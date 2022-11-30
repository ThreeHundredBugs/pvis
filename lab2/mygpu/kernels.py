from pycuda.compiler import SourceModule


_source = SourceModule("""
__global__ void convolve3x3(float *dest, float *image, float kernel[9])
{
    const int channel = threadIdx.x;
    const int channels = 4;

    const int destIndex = (blockIdx.x + (gridDim.x) * blockIdx.y) * channels + channel;

    if (channel == 3) {
        dest[destIndex] = 1.0f;
        return;
    }

    float sum = 0.0f;
    for (int dy = 0; dy < 3; ++dy) {
        for (int dx = 0; dx < 3; ++dx)
        {
            const int origIndex = (blockIdx.x + dx + (gridDim.x + 2) * (blockIdx.y + dy)) * channels + channel;
            sum += image[origIndex] * kernel[dx + 3*dy];
        }
    }
    dest[destIndex] = sum;
}

__global__ void normalize(float *array, float invmax, float min)
{
    const int channels = 4;
    const int channel = threadIdx.x;
    if (channel == 3) {
        return;
    }

    const int index = blockIdx.x * channels + channel;
    const float val = array[index];
    array[index] = (val - min) * invmax;
}
""")

convolve3x3 = _source.get_function('convolve3x3')
normalize = _source.get_function('normalize')
