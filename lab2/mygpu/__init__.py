import logging
import time
import numpy as np

import pycuda.autoinit

import PIL.Image
import imageio.v3 as iio

from .filters import emboss

logger = logging.getLogger(__file__)
_CUDA_ATTRIBUTES = None
_CUDA_DEVICE = None

def print_info() -> None:
    device, attributes = get_device_attributes()
    print(device)

    for name, value in attributes.items():
        print(f'  {repr(name).lstrip("pycuda._driver.device_attribute")}={value}')


def get_device_attributes() -> (str, dict[str, str | int | float]):
    global _CUDA_DEVICE, _CUDA_ATTRIBUTES

    if _CUDA_DEVICE is not None and _CUDA_ATTRIBUTES is not None:
        return _CUDA_DEVICE, _CUDA_ATTRIBUTES
    
    dev = pycuda.autoinit.device
    _CUDA_DEVICE = dev.name()
    _CUDA_ATTRIBUTES = dev.get_attributes()

    return _CUDA_DEVICE, _CUDA_ATTRIBUTES


def convert(in_image_path: str, out_image_path: str) -> None:
    _prev = PIL.Image.MAX_IMAGE_PIXELS
    PIL.Image.MAX_IMAGE_PIXELS = 3000000000

    try:
        before = time.time_ns()
        image = iio.imread(in_image_path)
        after = time.time_ns()
        logger.info(f'Image reading: {after - before}ns')

        before = time.time_ns()
        image = image / 255
        after = time.time_ns()
        logger.info(f'Image conversion to float format: {after - before}ns')

        filtered = emboss(image)
        filtered = np.rint(filtered * 255).astype(np.uint8)

        before = time.time_ns()
        iio.imwrite(out_image_path, filtered)
        after = time.time_ns()
        logger.info(f'Image saving: {after - before}ns')
    finally:
        PIL.Image.MAX_IMAGE_PIXELS = _prev
