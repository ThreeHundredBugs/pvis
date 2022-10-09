#include <cmath>
#include <iostream>
#include <numbers>
#include "filters.h"

extern unsigned int GLOBAL_THREADS_NUM;

namespace myimg {

/*
                (𝑥∙𝑐𝑜𝑠𝜃 + 𝑦∙𝑠𝑖𝑛𝜃)^2 + 𝛾^2∙(−𝑥∙𝑐𝑜𝑠𝜃 + 𝑦∙𝑠𝑖𝑛𝜃)^2                 𝑥∙𝑐𝑜𝑠𝜃 + 𝑦∙𝑠𝑖𝑛𝜃
𝐺(𝑥, 𝑦) = exp(-  ----------------------------------------------) × cos(2𝜋 ∙ -------------------- + 𝜙)
                                    2𝜎^2                                             𝜆
*/
matrix<float> get_gabor_kernel(
    unsigned int width,
    unsigned int height,
    float theta,
    float gamma,
    float sigma,
    float lambda,
    float phi
) {
    matrix<float> result((size_t)width, (size_t)height);
    auto width_half = width / 2;
    auto height_half = height / 2;

    #pragma omp parallel for collapse(2) num_threads(GLOBAL_THREADS_NUM)
    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < height; ++j) {
            float x = (int)height_half - j;
            float y = i - (int)width_half;

            auto sinTheta = std::sin(theta);
            auto cosTheta = std::cos(theta);

            auto x_complex = x * cosTheta + y * sinTheta;
            auto y_complex = -x * sinTheta + y * cosTheta;
            auto numerator = x_complex * x_complex + gamma * gamma * y_complex * y_complex;

            auto e = std::exp(-numerator / (2.0f * sigma * sigma));
            auto c = std::cos(2.0f * std::numbers::pi * x_complex / lambda + phi);

            result(i, j) = e * c;
        }
    }

    return result;
}

};