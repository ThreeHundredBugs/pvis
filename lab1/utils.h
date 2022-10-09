#pragma once

#include <vector>
#include "CImg/CImg.h"
#include "matrix.h"

namespace myimg{

matrix<pixel> from_cimg(const cimg_library::CImg<>& image) {
    std::vector<pixel> tmp_buffer;
    tmp_buffer.resize(image.width() * image.height());

    cimg_forXY(image, i, j){
        pixel p{
            (uint8_t)image(i, j, 0, 0),
            (uint8_t)image(i, j, 0, 1),
            (uint8_t)image(i, j, 0, 2),
        };
        tmp_buffer[i + j*image.width()] = p;
    }
        
    return matrix<pixel>(
        std::move(tmp_buffer),
        image.width(),
        image.height()
    );
}

cimg_library::CImg<> to_cimg(const matrix<pixel>& m) {
    cimg_library::CImg image(m.width, m.height, 1, 3);

    cimg_forXY(image, x, y) {
        auto p = m(y,x);
        image(x, y, 0) = p.r;
        image(x, y, 1) = p.g;
        image(x, y, 2) = p.b;
    }
    
    return image;
}

template<typename T>
cimg_library::CImg<unsigned char> to_cimg(const matrix<T>& m) {
    cimg_library::CImg image(m.width, m.height, 1, 3);

    cimg_forXY(image, x, y) {
        auto p = m(y,x);
        image(x, y, 0) = p;
        image(x, y, 1) = p;
        image(x, y, 2) = p;
    }
    
    return image;
}

template<typename In = double, typename Out = uint8_t>
matrix<Out> translate(
    const matrix<In>& m,
    double in_lo = -1.0,
    double in_hi = 1.0,
    double out_lo = 0.0,
    double out_hi = 255.0
) {
    unsigned int elements_len = m.width * m.height;
    Out* buffer = new Out[elements_len];

    double in_len = in_hi - in_lo;
    double out_len = out_hi - out_lo;
    double ratio = out_len / in_len;

    #pragma omp parallel for
    for (int i = 0; i < elements_len; ++i) {
        buffer[i] = (m.data[i] + in_lo) * ratio + in_lo;
    }
    
    std::vector<Out> v;
    v.assign(buffer, buffer + elements_len);

    return matrix(std::move(v), m.width, m.height);
}

}