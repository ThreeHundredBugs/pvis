from pycuda.compiler import SourceModule

source = SourceModule("""
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

__global__ void multiply_by(float *array, float val)
{
}

""")

convolve3x3 = source.get_function('convolve3x3')
