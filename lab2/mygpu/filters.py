import time
import logging

import pycuda.autoinit

import numpy as np
import pycuda.driver as cuda

from .kernels import convolve3x3, normalize, clip


logger = logging.getLogger(__file__)

EMBOSS_FILTER = np.array([
    [-2, -1, 0],
    [-1,  1, 1],
    [ 0,  1, 2],
]).astype(np.float32)

IDENTITY_FILTER = np.array([
    [0, 0, 0],
    [0, 1, 0],
    [0, 0, 0],
]).astype(np.float32)

BOX_BLUR_FILTER = (1/9) * np.array([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1],
]).astype(np.float32)

SHARPEN_FILTER = np.array([
    [ 0, -1,  0],
    [-1,  5, -1],
    [0,  -1,  0],
]).astype(np.float32)


def emboss(image: np.array) -> np.array:
    return apply_filter(image, EMBOSS_FILTER)

def apply_filter(image: np.array, filter: np.array) -> np.array:
    width, height = image.shape[1], image.shape[0]

    before = time.time_ns()
    dest_host = np.empty_like(image).astype(np.float32)
    dest_device = cuda.mem_alloc(dest_host.nbytes)
    after = time.time_ns()
    logger.info(f'Allocation of memory for output image: {after - before}ns')

    image = np.pad(
        image,
        ((1, 1), (1, 1), (0, 0)),
        mode='reflect',
    ).astype(np.float32)

    before = time.time_ns()
    image_device = cuda.mem_alloc(image.nbytes)
    cuda.memcpy_htod(image_device, image)
    after = time.time_ns()
    logger.info(f'Allocation and memory copy for source image: {after - before}ns')

    before = time.time_ns()
    convolve3x3(
        dest_device,
        image_device,
        cuda.In(filter),
        block=(4, 1, 1),
        grid=(width, height, 1),
    )
    after = time.time_ns()
    logger.info(f'Image convolution: {after - before}ns')

    before = time.time_ns()
    cuda.memcpy_dtoh(dest_host, dest_device)
    minimum = np.min(np.amin(dest_host), 0)
    maximum = max(np.amax(dest_host), 1)
    after = time.time_ns()
    logger.info(f'Find min and max for normalization: {after - before}ns')

    before = time.time_ns()
    normalize(
        dest_device,
        np.float32(1 / (maximum - minimum)),
        np.float32(minimum),
        block=(4, 1, 1),
        grid=(width * height, 1, 1),
    )
    # # instead of normalize:
    # clip(
    #     dest_device,
    #     np.float32(0),
    #     np.float32(1),
    #     block=(4, 1, 1),
    #     grid=(width * height, 1, 1),
    # )
    after = time.time_ns()
    logger.info(f'Image normalization: {after - before}ns')

    cuda.memcpy_dtoh(dest_host, dest_device)

    return dest_host
