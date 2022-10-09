/*
Вариант 6 Фильтрация матрицы фильтром Габора

Вход: Матрица M размерами 2^D×2^S
Выход: Отфильтрованная матрица размерами (2^D-Rf+1)×(2^S-Rf+1),
    где Rf – размер фильтра

Выражения (A) и (B) служат для вычисления значений фильтра Габора,
использующегося для нахождения локальных ориентаций (отрезков
определённой ориентации, детектирующих линии с заданным наклоном).

(A):
                    (𝑥∙𝑐𝑜𝑠𝜃 + 𝑦∙𝑠𝑖𝑛𝜃)^2 + 𝛾^2∙(−𝑥∙𝑐𝑜𝑠𝜃 + 𝑦∙𝑠𝑖𝑛𝜃)^2                 𝑥∙𝑐𝑜𝑠𝜃 + 𝑦∙𝑠𝑖𝑛𝜃
    𝐺(𝑥, 𝑦) = exp(-  ----------------------------------------------) × cos(2𝜋 ∙ -------------------- + 𝜙)
                                        2𝜎^2                                             𝜆

    где:
        x, y – координаты элемента фильтрующей матрицы,
        𝛾 = 0.3 – степень пространственного аспекта,
        𝜙 = 0 – отклонение локальной характеристики от центра фильтра,
        𝜆 = 𝜎 ⁄ 0.8,
        𝜎 – отклонение (ширина локальной характеристики),
            рассчитывающееся в предложенной модели из следующего уравнения
(B):
        𝜎 = 0.0036∙𝑅𝑓^22 + 0.35𝑅𝑓 + 0.18, 

        где: Rf – размер фильтра (использовалось значение 7).

Получить отфильтрованные матрицы для следующих случаев:
    a) D=12, S=12 (4096x4096);   Rf=12, 𝜃=𝜋/2.
    b) D=13, S=12 (8192x4096);   Rf=12, 𝜃=𝜋/2.
    c) D=14, S=14 (16384x16384); Rf=12, 𝜃=𝜋/2.
*/

#include <iostream>
#include <filesystem>
#include <chrono>
#include <string>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <numbers>

#include "CImg/CImg.h"
#include "matrix.h"
#include "filters.h"
#include "utils.h"

using namespace cimg_library;
using namespace myimg;

unsigned int GLOBAL_THREADS_NUM = 4;
static const double PI = std::numbers::pi;
// PI     ~= 3.14159265358979323846
// PI / 2 ~= 1.57079632679489661923
// PI / 4 ~= 0.78539816339744830961

int main(int argc, char **argv) {
    const char* image_path = cimg_option("--input-image", "../images/1920x1080.png", "Input image file");
    const unsigned int threads_num = cimg_option("--threads-num", 4, "Number of threads to use");

    const unsigned int kernel_size = cimg_option("--kernel-size", 12, "Size of square Gabor kernel");
    const double theta = cimg_option("--theta", PI / 2, "Theta parameter of Gabor function");
    const double gamma = cimg_option("--gamma", 0.3, "Gamma parameter of Gabor function");
    const double phi = cimg_option("--phi", 0.0, "Phi parameter of Gabor function");
    const double sigma = cimg_option("--sigma", 0.0036 * kernel_size * kernel_size + 0.35 * kernel_size + 0.18, "Sigma parameter of Gabor function");
    const double lambda = cimg_option("--lambda", sigma / 0.8, "Lambda parameter of Gabor function");

    if (threads_num > 0) {
        GLOBAL_THREADS_NUM = threads_num;
    }

    if (!std::filesystem::exists(image_path)) {
        std::cerr << image_path << " does not exist";
        return 1;
    }
    
    CImg source_image(image_path);
    auto img_matrix = from_cimg(source_image);

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    matrix<float> gray = grayscale_filter(img_matrix);
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout
        << "grayscale_time=" 
        << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()
        << " [microseconds]"
        << std::endl;
    
    begin = std::chrono::steady_clock::now();
    auto gabor_kernel = get_gabor_kernel(
        kernel_size,  // width
        kernel_size,  // height
        theta,        // theta: rotation
        gamma,        // gamma: elongation
        sigma,        // sigma: deviation (size)
        lambda,       // lambda: wave length
        phi           // phi: offset of wave
    );
    end = std::chrono::steady_clock::now();
    std::cout
        << "gabor_kernel_time=" 
        << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()
        << " [microseconds]"
        << std::endl;

    begin = std::chrono::steady_clock::now();
    auto filtered = convolve(gray, gabor_kernel);
    end = std::chrono::steady_clock::now();
    std::cout
        << "convolution_time=" 
        << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()
        << " [microseconds]"
        << std::endl;


    #ifndef MEASURE_BUILD
    double max = *std::max_element(filtered.data.begin(), filtered.data.end());
    double min = *std::min_element(filtered.data.begin(), filtered.data.end());

    auto filtered_img = to_cimg<uint8_t>(translate(filtered, min, max));
    CImgDisplay display_filtered(filtered_img, "Image");

    auto gabor_kernel_img = to_cimg<uint8_t>(translate(gabor_kernel));
    while (gabor_kernel_img.width() < 150){
        gabor_kernel_img.resize_doubleXY();
    }
    std::string title("Gabor kernel");
    CImgDisplay display_kernel(
        gabor_kernel_img,
        (
            title 
            + "[original size " + std::to_string(kernel_size) + "; upscaled size "
            + std::to_string(gabor_kernel_img.width()) + "]"
        ).c_str()
    );

    while (!(
        display_kernel.is_closed() ||
        display_kernel.is_keyENTER()
    )) {
        display_kernel.wait();
    }
    #endif

    return 0;
}
