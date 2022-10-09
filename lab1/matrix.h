#pragma once
#include <vector>
#include <utility>
#include <cstdint>
#include <cassert>

namespace myimg{

struct pixel {
    uint8_t r;
    uint8_t g;
    uint8_t b;
};

template<typename T>
struct matrix {
    const size_t width;
    const size_t height;
    std::vector<T> data;

    matrix(std::vector<T>&& data, size_t width, size_t height) :
        width(width),
        height(height),
        data(data)
    {
        assert(((void)"Matrix data size is not equal width*height", data.size() == width * height));
    }

    matrix(size_t width, size_t height) :
        width(width),
        height(height),
        data(width * height)
    {
    }

    T operator()(int i, int j) const {
        return data[i*width + j];
    }
    T& operator()(int i, int j) {
        return data[i*width + j];
    }
};

}