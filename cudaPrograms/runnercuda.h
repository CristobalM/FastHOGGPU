
#ifndef PROJECT_RUNNER_CUDA_H
#define PROJECT_RUNNER_CUDA_H

#include <cstddef>
#include <memory>

#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "ISVMManagerGPU.h"
#include "IImageManagerGPU.h"

using uchar = unsigned char;

std::unique_ptr<ISVMManagerGPU>  loadSVMWeights(float *svmWeights, size_t size);
std::unique_ptr<IImageManagerGPU> loadImageToGPU(cv::Mat& imageMat);


#endif //PROJECT_RUNNER_CUDA_H
