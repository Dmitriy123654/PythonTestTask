import os
import uuid
import zipfile
from multiprocessing import Pool
from typing import List, Dict

import geojson
import numpy as np
from PIL import Image
from fastapi.responses import JSONResponse
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.responses import RedirectResponse
from geojson import MultiPolygon
from pydantic import BaseModel
from rasterio.features import shapes
from shapely.geometry import Polygon
from shapely.ops import unary_union
from multiprocessing import shared_memory

# Увеличиваем лимит на размер изображения
Image.MAX_IMAGE_PIXELS = None

app = FastAPI()
from fastapi.staticfiles import StaticFiles

app.mount("/static", StaticFiles(directory="static"), name="static")

# Параметры для фрагментации
class FragmentationParams(BaseModel):
    overlap_size: int
    max_fragment_size: int


# --- Утилиты ---
def cleanup_files(file_paths: List[str]):
    """Удаление временных файлов."""
    for path in file_paths:
        if os.path.exists(path):
            os.remove(path)

# Функция для создания векторной маски
def create_mask(image_array: np.ndarray) -> np.ndarray:
    """
    Создание маски полезной области.
    Пиксели с яркостью 0 или 255 считаются неинформативными (фон).
    Полезные области имеют значение 1, фон — 0.
    """
    # Преобразуем в градации серого, если изображение цветное
    if image_array.ndim == 3:
        image = Image.fromarray(image_array)
        image_array = np.array(image.convert("L"))

    # Полезная область: пиксели между 0 и 255
    mask = np.where((image_array > 0) & (image_array < 255), 1, 0).astype(np.uint8)
    return mask


def create_polygon_from_mask(mask: np.ndarray) -> Dict:
    """
    Создает векторный полигон (GeoJSON) из маски.
    """
    if mask.ndim == 3:
        mask = mask[:, :, 0]

    # Генерация полигонов
    shapes_generator = shapes(mask, mask=mask > 0)
    polygons = []

    for shape_data, value in shapes_generator:
        if value == 1:  # Учитываем только полезные области
            polygons.append(Polygon(shape_data['coordinates'][0]))

    # Объединяем все полигоны в один объект
    if polygons:
        merged_polygon = unary_union(polygons)
        if isinstance(merged_polygon, Polygon):
            merged_polygon = [merged_polygon]
        elif isinstance(merged_polygon, MultiPolygon):
            merged_polygon = list(merged_polygon.geoms)

        return geojson.FeatureCollection(
            [geojson.Feature(geometry=poly.__geo_interface__) for poly in merged_polygon]
        )
    else:
        print("No polygons created.")
        return geojson.FeatureCollection([])  # Если никаких полигонов не создано


def prepare_shared_array(image_array, mask):
    """
    Создаёт разделяемую память для изображения и маски.
    Возвращает объекты разделяемой памяти и их параметры.
    """
    # Проверяем, что размеры корректны
    image_array = np.ascontiguousarray(image_array)  # Убедимся, что массив непрерывный
    mask = np.ascontiguousarray(mask)  # Убедимся, что массив непрерывный

    # Создаём разделяемую память для изображения
    image_shm = shared_memory.SharedMemory(create=True, size=image_array.nbytes)
    image_shared = np.ndarray(image_array.shape, dtype=image_array.dtype, buffer=image_shm.buf)
    np.copyto(image_shared, image_array)

    # Создаём разделяемую память для маски
    mask_shm = shared_memory.SharedMemory(create=True, size=mask.nbytes)
    mask_shared = np.ndarray(mask.shape, dtype=mask.dtype, buffer=mask_shm.buf)
    np.copyto(mask_shared, mask)

    # Возвращаем объекты разделяемой памяти и параметры
    return image_shm, mask_shm, image_array.shape, mask.shape



def determine_global_background(image_array):
    """
    Определяет общий фон изображения (черный или белый).
    Returns:
        int: 0 для черного фона или 255 для белого фона.
    """
    # Убедимся, что работаем с градациями серого
    if image_array.ndim == 3:
        image_array = image_array[:, :, 0]

    # Подсчет количества черных и белых пикселей
    black_count = np.sum(image_array == 0)
    white_count = np.sum(image_array == 255)

    # Возвращаем фон: больше белого — фон белый, больше черного — фон черный
    return 255 if white_count > black_count else 0


def pad_to_square(fragment, fragment_size, constant_value=0):
    """
    Дополняет фрагмент до квадратного размера, добавляя пиксели указанного цвета.
    """
    height, width = fragment.shape[:2]
    pad_height = max(0, fragment_size - height)
    pad_width = max(0, fragment_size - width)

    # Дополняем указанным цветом
    padded_fragment = np.pad(
        fragment,
        ((0, pad_height), (0, pad_width)) + ((0, 0),) if fragment.ndim == 3 else ((0, pad_height), (0, pad_width)),
        mode='constant',
        constant_values=constant_value
    )
    return padded_fragment



def is_fragment_informative(mask_fragment, min_coverage):
    """
    Проверка, содержит ли фрагмент достаточно полезной области.
    """
    informative_pixels = np.count_nonzero(mask_fragment)
    total_pixels = mask_fragment.size
    return (informative_pixels / total_pixels) >= min_coverage

def fragment_image_worker_shared(args):
    """
    Обрабатывает один блок изображения, используя разделяемую память.
    """
    (image_shape, mask_shape, image_shm_name, mask_shm_name,
     x_start, y_start, fragment_size,overlap_size, fragment_dir,
     index, min_coverage, global_background) = args
    # Подключаемся к разделяемой памяти
    existing_image_shm = shared_memory.SharedMemory(name=image_shm_name)
    existing_mask_shm = shared_memory.SharedMemory(name=mask_shm_name)
    try:
        image_array = np.ndarray(image_shape, dtype=np.uint8, buffer=existing_image_shm.buf)
        mask = np.ndarray(mask_shape, dtype=np.uint8, buffer=existing_mask_shm.buf)

        # Обработка фрагментов
        x_end = min(x_start + fragment_size, image_shape[1])
        y_end = min(y_start + fragment_size, image_shape[0])

        fragment = image_array[y_start:y_end, x_start:x_end]
        fragment_mask = mask[y_start:y_end, x_start:x_end]

        # Проверка покрытия
        if np.sum(fragment_mask) < min_coverage * fragment_size * fragment_size:
            return None

        # Дополнение до квадрата
        padded_fragment = pad_to_square(fragment, fragment_size, constant_value=global_background)

        # Сохраняем фрагмент
        fragment_path = os.path.join(fragment_dir, f"fragment_{index}.tif")
        Image.fromarray(padded_fragment).save(fragment_path)
        return fragment_path
    finally:
        existing_image_shm.close()
        existing_image_shm.unlink()  # Удаляем разделяемую память
        existing_mask_shm.close()
        existing_mask_shm.unlink()  # Удаляем разделяемую память




def fragment_image_multiprocessing_shared(image_array, mask, params, fragment_dir, min_coverage):
    """
    Разделение изображения на фрагменты с использованием разделяемой памяти и многопроцессорности.
    Все фрагменты дополняются до квадратного размера, фон определяется по всему изображению.
    """
    height, width = image_array.shape[:2]
    fragment_size = params.max_fragment_size
    overlap_size = params.overlap_size

    # Определяем общий фон изображения (черный или белый)
    global_background = determine_global_background(image_array)

    # Создаём разделяемую память
    image_shm = shared_memory.SharedMemory(create=True, size=image_array.nbytes)
    mask_shm = shared_memory.SharedMemory(create=True, size=mask.nbytes)
    try:
        # Копируем данные в разделяемую память
        shared_image = np.ndarray(image_array.shape, dtype=image_array.dtype, buffer=image_shm.buf)
        shared_mask = np.ndarray(mask.shape, dtype=mask.dtype, buffer=mask_shm.buf)
        np.copyto(shared_image, image_array)
        np.copyto(shared_mask, mask)

        # Генерируем задачи
        tasks = []
        index = 0
        for y in range(0, height, fragment_size - overlap_size):
            for x in range(0, width, fragment_size - overlap_size):
                tasks.append((
                    image_array.shape, mask.shape, image_shm.name, mask_shm.name,
                    x, y, fragment_size, overlap_size, fragment_dir, index,
                    min_coverage, global_background
                ))
                index += 1

        # Обрабатываем задачи с помощью multiprocessing.Pool
        with Pool() as pool:
            fragment_paths = pool.map(fragment_image_worker_shared, tasks)

        # Убираем пустые результаты (None)
        return [path for path in fragment_paths if path is not None]

    finally:
        # Закрываем разделяемую память
        image_shm.close()
        image_shm.unlink()
        mask_shm.close()
        mask_shm.unlink()

@app.post("/create_mask")
async def create_mask_endpoint(image: UploadFile = File(...)):
    """
    Эндпоинт для создания маски полезной области изображения.
    Возвращает ZIP-архив с маской и GeoJSON.
    """
    temp_image_path = "temp_image.tif"
    mask_image_path = "mask_image.tif"
    geojson_path = "polygon.json"
    archive_name = f"mask_polygon_{uuid.uuid4().hex}.zip"

    with open(temp_image_path, "wb") as f:
        f.write(await image.read())

    try:
        # Загрузка изображения
        image = Image.open(temp_image_path)
        image_array = np.array(image)

        # Создание маски
        mask = create_mask(image_array)
        mask_image = Image.fromarray((mask * 255).astype(np.uint8))
        mask_image.save(mask_image_path)

        # Создание GeoJSON
        polygon_geojson = create_polygon_from_mask(mask)
        with open(geojson_path, "w") as geojson_file:
            geojson.dump(polygon_geojson, geojson_file)

        # Создание архива
        with zipfile.ZipFile(archive_name, "w") as zipf:
            zipf.write(mask_image_path, "mask.tif")
            zipf.write(geojson_path, "polygon.json")

        # Очистка временных файлов
        cleanup_files([temp_image_path, mask_image_path, geojson_path])

        # Возвращение архива
        return JSONResponse({"archive_name": archive_name}, status_code=200)

    except Exception as e:
        cleanup_files([temp_image_path, mask_image_path, geojson_path, archive_name])
        raise HTTPException(status_code=500, detail=f"Ошибка: {str(e)}")


@app.post("/fragment_image")
async def fragment_image_endpoint(
    image: UploadFile = File(...),
    overlap_size: int = Form(...),
    max_fragment_size: int = Form(...),
    min_coverage: float = Form(0.05)
):
    """
    Эндпоинт для фрагментации изображения.
    Возвращает ZIP-архив с фрагментами.
    """
    temp_image_path = "temp_image.tif"
    mask_image_path = "mask_image.tif"
    fragment_dir = "fragments"
    archive_name = f"fragments_{uuid.uuid4().hex}.zip"
    os.makedirs(fragment_dir, exist_ok=True)

    with open(temp_image_path, "wb") as f:
        f.write(await image.read())

    try:
        # Загрузка изображения
        image = Image.open(temp_image_path)
        image_array = np.array(image)

        # Создание маски
        mask = create_mask(image_array)
        mask_image = Image.fromarray((mask * 255).astype(np.uint8))
        mask_image.save(mask_image_path)

        # Фрагментация
        fragments = fragment_image_multiprocessing_shared(
            image_array,
            mask,
            FragmentationParams(overlap_size=overlap_size, max_fragment_size=max_fragment_size),
            fragment_dir,
            min_coverage
        )

        # Создание архива
        with zipfile.ZipFile(archive_name, "w") as zipf:
            for fragment_path in fragments:
                zipf.write(fragment_path, os.path.basename(fragment_path))

        # Очистка временных файлов
        cleanup_files([temp_image_path, mask_image_path])
        for fragment_path in fragments:
            os.remove(fragment_path)
        os.rmdir(fragment_dir)

        # Возвращение архива
        return JSONResponse({"archive_name": archive_name}, status_code=200)

    except Exception as e:
        cleanup_files([temp_image_path, mask_image_path, archive_name])
        if os.path.exists(fragment_dir):
            for file in os.listdir(fragment_dir):
                os.remove(os.path.join(fragment_dir, file))
            os.rmdir(fragment_dir)
        raise HTTPException(status_code=500, detail=f"Ошибка: {str(e)}")


@app.get("/download_archive")
def download_archive(archive_name: str, background_tasks: BackgroundTasks):
    if not os.path.exists(archive_name):
        raise HTTPException(status_code=404, detail="Архив не найден.")

    # Возвращаем файл
    response = FileResponse(archive_name, media_type="application/zip", filename=os.path.basename(archive_name))

    # Запланируем удаление архива после отправки
    background_tasks.add_task(cleanup_files, [archive_name])

    return response

@app.get("/")
async def root():
    """
    Редирект на статическую страницу.
    """
    return RedirectResponse(url="/static/index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)