#pragma once

#include <cstdint>
#include "matrix.h"

namespace myimg {

matrix<float> convolve(const matrix<float>& image, const matrix<float>& filter);
matrix<float> get_gabor_kernel(
    unsigned int width,
    unsigned int height,
    float theta,  // rotation
    float gamma,  // elongation
    float sigma,  // deviation (size)
    float lambda, // wave length
    float phi     // offset of wave
);

template<typename T = float>
matrix<T> grayscale_filter(const matrix<pixel>& image){
    std::vector<T> grey_data(image.data.size());

    #pragma omp parallel for
    for (int i = 0; i < image.data.size(); ++i) {
        grey_data[i] = (T)(
            0.299 * image.data[i].r +
            0.587 * image.data[i].g +
            0.114 * image.data[i].b
        );
    }

    return matrix<T>(std::move(grey_data), image.width, image.height);
}

};
