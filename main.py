import os
import uuid
import zipfile

import geojson
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.responses import JSONResponse
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from services.fragmentation import fragment_image_multiprocessing_shared
from services.mask_creator import create_mask, create_polygon_from_mask
from utils.utils import cleanup_files

Image.MAX_IMAGE_PIXELS = None
app = FastAPI()
from fastapi.staticfiles import StaticFiles

app.mount("/static", StaticFiles(directory="static"), name="static")

class FragmentationParams(BaseModel):
    overlap_size: int
    max_fragment_size: int

@app.post("/create_mask")
async def create_mask_endpoint(image: UploadFile = File(...)):
    """
    Эндпоинт для создания маски полезной области изображения и полигона
    """
    temp_image_path = "temp_image.tif"
    mask_image_path = "mask_image.tif"
    geojson_path = "polygon.json"
    archive_name = f"mask_polygon_{uuid.uuid4().hex}.zip"

    with open(temp_image_path, "wb") as f:
        f.write(await image.read())

    try:
        image = Image.open(temp_image_path)
        image_array = np.array(image)

        mask = create_mask(image_array)
        mask_image = Image.fromarray((mask * 255).astype(np.uint8))
        mask_image.save(mask_image_path)

        polygon_geojson = create_polygon_from_mask(mask)
        with open(geojson_path, "w") as geojson_file:
            geojson.dump(polygon_geojson, geojson_file)

        with zipfile.ZipFile(archive_name, "w") as zipf:
            zipf.write(mask_image_path, "mask.tif")
            zipf.write(geojson_path, "polygon.json")

        cleanup_files([temp_image_path, mask_image_path, geojson_path])

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
    """
    temp_image_path = "temp_image.tif"
    mask_image_path = "mask_image.tif"
    fragment_dir = "fragments"
    archive_name = f"fragments_{uuid.uuid4().hex}.zip"
    os.makedirs(fragment_dir, exist_ok=True)

    with open(temp_image_path, "wb") as f:
        f.write(await image.read())

    try:
        image = Image.open(temp_image_path)
        image_array = np.array(image)

        mask = create_mask(image_array)
        mask_image = Image.fromarray((mask * 255).astype(np.uint8))
        mask_image.save(mask_image_path)

        fragments = fragment_image_multiprocessing_shared(
            image_array,
            mask,
            FragmentationParams(overlap_size=overlap_size, max_fragment_size=max_fragment_size),
            fragment_dir,
            min_coverage
        )

        with zipfile.ZipFile(archive_name, "w") as zipf:
            for fragment_path in fragments:
                zipf.write(fragment_path, os.path.basename(fragment_path))

        cleanup_files([temp_image_path, mask_image_path])
        for fragment_path in fragments:
            os.remove(fragment_path)
        os.rmdir(fragment_dir)

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

    response = FileResponse(archive_name, media_type="application/zip", filename=os.path.basename(archive_name))

    background_tasks.add_task(cleanup_files, [archive_name])

    return response

@app.get("/")
async def root():
    return RedirectResponse(url="/static/index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)