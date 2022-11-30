from pycuda.compiler import SourceModule


source = SourceModule("""
__global__ void invert(float *dest, float *image)
{
    const int channel = threadIdx.x;
    const int channels = 4;

    const int index = (blockIdx.x + gridDim.x * blockIdx.y) * channels + channel;

    if (channel == 3) {
        dest[index] = 1.0f;
        return;
    }
    dest[index] = 1.0f - image[index];
}
""")

invert = source.get_function('invert')
