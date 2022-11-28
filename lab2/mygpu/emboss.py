from pycuda.compiler import SourceModule


# Kernel:
#  -2  -1  0
#  -1   1  1
#   0   1  2
source = SourceModule("""
__global__ void emboss(float *dest, float *image, int width, int height)
{
    // TODO
}
""")

emboss = source.get_function('emboss')
