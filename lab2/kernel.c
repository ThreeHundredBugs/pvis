__global__ void invert(uint8_t *dest, uint8_t *image, int width, int height)
{
  const int i = threadIdx.x;
  const int j = threadIdx.y;
  dest[i] = 255 - image[i * width + j];
}