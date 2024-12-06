import os
from multiprocessing import shared_memory
from typing import List

import numpy as np


def cleanup_files(file_paths: List[str]):
    """Удаление временных файлов."""
    for path in file_paths:
        if os.path.exists(path):
            os.remove(path)


def prepare_shared_array(image_array, mask):
    """
    Создаёт разделяемую память для изображения и маски.
    Возвращает объекты разделяемой памяти и их параметры.
    """
    image_array = np.ascontiguousarray(image_array)
    mask = np.ascontiguousarray(mask)

    image_shm = shared_memory.SharedMemory(create=True, size=image_array.nbytes)
    image_shared = np.ndarray(image_array.shape, dtype=image_array.dtype, buffer=image_shm.buf)
    np.copyto(image_shared, image_array)

    mask_shm = shared_memory.SharedMemory(create=True, size=mask.nbytes)
    mask_shared = np.ndarray(mask.shape, dtype=mask.dtype, buffer=mask_shm.buf)
    np.copyto(mask_shared, mask)

    return image_shm, mask_shm, image_array.shape, mask.shape

def determine_global_background(image_array):
    """
    Определяет общий фон изображения (черный или белый).
    Returns:
        int: 0 для черного фона или 255 для белого фона.
    """
    if image_array.ndim == 3:
        image_array = image_array[:, :, 0]

    black_count = np.sum(image_array == 0)
    white_count = np.sum(image_array == 255)

    return 255 if white_count > black_count else 0
