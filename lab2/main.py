import sys
import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
import imageio.v3 as iio

from pycuda.compiler import SourceModule

from mygpu import print_info


np.set_printoptions(threshold=sys.maxsize)
# print_info()

source = SourceModule("""
__global__ void invert(float *dest, float *image, int width, int height)
{
    const int x = threadIdx.x;
    const int y = threadIdx.y;
    const int channels = 4;
    
    // dest[(x + width * y) * channels + 0] = 1.0;
    // dest[(x + width * y) * channels + 1] = 0.0;
    // dest[(x + width * y) * channels + 2] = 0.0;
    // dest[(x + width * y) * channels + 3] = 1.0;

    dest[0] = 1.0;
    dest[1] = 0.5;
    dest[2] = 0.5;
    dest[3] = 1.0;
    dest[4] = 1.0;
    dest[5] = 0.0;
    dest[6] = 0.0;
    dest[7] = 1.0;
}
""")

invert = source.get_function('invert')

image_host = (iio.imread('../images/500x321.png') / 255).astype(np.float32)
dest_host = np.zeros_like(image_host).astype(np.float32)


image_device = drv.mem_alloc(image_host.nbytes)
drv.memcpy_htod(image_device, image_host)
dest_device = drv.mem_alloc(dest_host.nbytes)
drv.memcpy_htod(dest_device, dest_host)


invert(
    dest_device,
    image_device,
    np.int32(500),
    np.int32(321),
    block=(1024, 1, 1),
    grid=(1, 1, 1),
)

drv.memcpy_dtoh(dest_host, dest_device)

dest_host = np.rint(dest_host * 255).astype(np.uint8)

iio.imwrite('./output/test.png', dest_host)
