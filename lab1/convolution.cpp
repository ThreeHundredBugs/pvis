#include "filters.h"


extern unsigned int GLOBAL_THREADS_NUM;


namespace myimg {

matrix<float> convolve(const matrix<float>& image, const matrix<float>& filter) {
    matrix<float> result(
        image.width - (filter.width / 2) * 2,
        image.height - (filter.height / 2) * 2
    );

    #pragma omp parallel for collapse(2) num_threads(GLOBAL_THREADS_NUM)
    for (int i = 0; i < result.height; ++i) {
        for (int j = 0; j < result.width; ++j) {
            float sum = .0f;
            for (int h = i; h < i + filter.height; ++h) {
                for (int w = j; w < j + filter.width; ++w) {
                    sum += image(h, w) * filter(h - i, w - j);
                }
            }
            result(i, j) = sum;
       }
    }

    return result;
}

};
