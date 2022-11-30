__global__ void convolve3x3(float *dest, float *image, float* kernel)
{
    const int channel = threadIdx.x;
    const int channels = 4;

    const int destIndex = (blockIdx.x + (gridDim.x) * blockIdx.y) * channels + channel;

    if (channel == 3) {
        dest[destIndex] = 1.0f;
        return;
    }

    const int origIndex = (blockIdx.x + 1 + (gridDim.x + 2) * (blockIdx.y + 1)) * channels + channel;
    dest[destIndex] = image[origIndex];
}
