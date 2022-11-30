import logging
import numpy as np

import pycuda.autoinit

import PIL.Image
import imageio.v3 as iio

from .filters import emboss

logger = logging.getLogger(__file__)
_CUDA_ATTRIBUTES = None
_CUDA_DEVICE = None

def print_info() -> None:
    global _CUDA_ATTRIBUTES
    global _CUDA_DEVICE

    if _CUDA_ATTRIBUTES is None:
        dev = pycuda.autoinit.device
        _CUDA_DEVICE = dev.name()
        _CUDA_ATTRIBUTES = dev.get_attributes()

    print(_CUDA_DEVICE)

    for name, value in _CUDA_ATTRIBUTES.items():
        print(f'  {repr(name).lstrip("pycuda._driver.device_attribute")}={value}')


def convert(in_image_path: str, out_image_path: str) -> None:
    _prev = PIL.Image.MAX_IMAGE_PIXELS
    PIL.Image.MAX_IMAGE_PIXELS = 3000000000

    try:
        logger.debug('Reading image')
        image = iio.imread(in_image_path)
        image = image / 255
        # image = np.array([
        #     [[1,0,1,1], [0,1,1,1], [1,1,0,1], [1,0,0,1], [0,0,1,1],],
        #     [[1,0,1,1], [0,1,1,1], [1,1,0,1], [1,0,0,1], [0,0,1,1],],
        #     [[1,0,1,1], [0,1,1,1], [1,1,0,1], [1,0,0,1], [0,0,1,1],],
        #     [[1,0,1,1], [0,1,1,1], [1,1,0,1], [1,0,0,1], [0,0,1,1],],
        #     [[0,0,0,1], [0,0,0,1], [0,1,0,1], [0,1,1,1], [1,0,0,1],],
        # ])
        filtered = emboss(image)
        filtered = np.rint(filtered * 255).astype(np.uint8)

        logger.debug('Saving image')
        iio.imwrite(out_image_path, filtered)
    finally:
        PIL.Image.MAX_IMAGE_PIXELS = _prev
