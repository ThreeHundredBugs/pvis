import time
import logging

import pycuda.autoinit

import numpy as np
import pycuda.driver as cuda

from .kernels import convolve3x3, normalize


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

BOX_BLUR_FILTER = (1/3) * np.array([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1],
]).astype(np.float32)


def emboss(image: np.array) -> np.array:
    return apply_filter(image, EMBOSS_FILTER)

def apply_filter(image: np.array, filter: np.array) -> np.array:
    width, height = image.shape[1], image.shape[0]

    before = time.time_ns()
    dest_host = np.empty_like(image).astype(np.float32)
    dest_device = cuda.mem_alloc(dest_host.nbytes)
    after = time.time_ns()
    elapsed = (after - before) / 1000
    logger.info(f'Allocation of memory for output image: {elapsed}microseconds')

    image = np.pad(
        image,
        ((1, 1), (1, 1), (0, 0)),
        mode='reflect',
    ).astype(np.float32)

    before = time.time_ns()
    image_device = cuda.mem_alloc(image.nbytes)
    cuda.memcpy_htod(image_device, image)
    after = time.time_ns()
    elapsed = (after - before) / 1000
    logger.info(f'Allocation and memory copy for source image: {elapsed}microseconds')

    before = time.time_ns()
    convolve3x3(
        dest_device,
        image_device,
        cuda.In(filter),
        block=(4, 1, 1),
        grid=(width, height, 1),
    )
    after = time.time_ns()
    elapsed = (after - before) / 1000
    logger.info(f'Image convolution: {elapsed}microseconds')

    before = time.time_ns()
    cuda.memcpy_dtoh(dest_host, dest_device)
    minimum = np.min(np.amin(dest_host), 0)
    maximum = np.amax(dest_host)
    after = time.time_ns()
    elapsed = (after - before) / 1000
    logger.info(f'Find min and max for normalization: {elapsed}microseconds')

    before = time.time_ns()
    normalize(
        dest_device,
        np.float32(1 / (maximum - minimum)),
        np.float32(minimum),
        block=(4, 1, 1),
        grid=(width * height, 1, 1),
    )
    after = time.time_ns()
    elapsed = (after - before)
    logger.info(f'Image normalization: {elapsed}nanoseconds')

    cuda.memcpy_dtoh(dest_host, dest_device)

    return dest_host
