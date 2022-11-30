import logging

import pycuda.autoinit

import numpy as np
import pycuda.driver as cuda

from .convolve import convolve3x3


logger = logging.getLogger(__file__)

EMBOSS_KERNEL = np.array([
    [-2, -1, 0],
    [-1,  1, 1],
    [ 0,  1, 2],
]).astype(np.float32)

IDENTITY_KERNEL = np.array([
    [0, 0, 0],
    [0, 1, 0],
    [0, 0, 0],
]).astype(np.float32)

BOX_BLUR_KERNEL = (1/3) * np.array([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1],
]).astype(np.float32)


def emboss(image: np.array) -> np.array:
    return apply_filter(image, EMBOSS_KERNEL)

def apply_filter(image: np.array, filter: np.array) -> np.array:
    width, height = image.shape[1], image.shape[0]
    dest = np.zeros_like(image).astype(np.float32)
    image = np.pad(
        image,
        ((1, 1), (1, 1), (0, 0)),
        mode='reflect',
    ).astype(np.float32)

    convolve3x3(
        cuda.Out(dest),
        cuda.In(image),
        cuda.In(filter),
        block=(4, 1, 1),
        grid=(width, height, 1),
    )

    return dest
