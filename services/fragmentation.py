import numpy as np
from multiprocessing import shared_memory, Pool
from PIL import Image
import os

from utils.utils import determine_global_background

def fragment_image_multiprocessing_shared(image_array, mask, params, fragment_dir, min_coverage):
    """
    Разделение изображения на фрагменты с использованием разделяемой памяти и многопроцессорности.
    Все фрагменты дополняются до квадратного размера, фон определяется по всему изображению.
    """
    height, width = image_array.shape[:2]
    fragment_size = params.max_fragment_size
    overlap_size = params.overlap_size

    global_background = determine_global_background(image_array)

    image_shm = shared_memory.SharedMemory(create=True, size=image_array.nbytes)
    mask_shm = shared_memory.SharedMemory(create=True, size=mask.nbytes)
    try:
        shared_image = np.ndarray(image_array.shape, dtype=image_array.dtype, buffer=image_shm.buf)
        shared_mask = np.ndarray(mask.shape, dtype=mask.dtype, buffer=mask_shm.buf)
        np.copyto(shared_image, image_array)
        np.copyto(shared_mask, mask)

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

        with Pool() as pool:
            fragment_paths = pool.map(fragment_image_worker_shared, tasks)

        return [path for path in fragment_paths if path is not None]

    finally:
        image_shm.close()
        image_shm.unlink()
        mask_shm.close()
        mask_shm.unlink()

def fragment_image_worker_shared(args):
    """
    Обрабатывает один блок изображения, используя разделяемую память.
    """
    (image_shape, mask_shape, image_shm_name, mask_shm_name,
     x_start, y_start, fragment_size,overlap_size, fragment_dir,
     index, min_coverage, global_background) = args

    existing_image_shm = shared_memory.SharedMemory(name=image_shm_name)
    existing_mask_shm = shared_memory.SharedMemory(name=mask_shm_name)
    try:
        image_array = np.ndarray(image_shape, dtype=np.uint8, buffer=existing_image_shm.buf)
        mask = np.ndarray(mask_shape, dtype=np.uint8, buffer=existing_mask_shm.buf)

        x_end = min(x_start + fragment_size, image_shape[1])
        y_end = min(y_start + fragment_size, image_shape[0])

        fragment = image_array[y_start:y_end, x_start:x_end]
        fragment_mask = mask[y_start:y_end, x_start:x_end]

        if np.sum(fragment_mask) < min_coverage * fragment_size * fragment_size:
            return None

        padded_fragment = pad_to_square(fragment, fragment_size, constant_value=global_background)

        fragment_path = os.path.join(fragment_dir, f"fragment_{index}.tif")
        Image.fromarray(padded_fragment).save(fragment_path)
        return fragment_path
    finally:
        existing_image_shm.close()
        existing_image_shm.unlink()
        existing_mask_shm.close()
        existing_mask_shm.unlink()

def pad_to_square(fragment, fragment_size, constant_value=0):
    """
    Дополняет фрагмент до квадратного размера, добавляя пиксели указанного цвета.
    """
    height, width = fragment.shape[:2]
    pad_height = max(0, fragment_size - height)
    pad_width = max(0, fragment_size - width)

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


