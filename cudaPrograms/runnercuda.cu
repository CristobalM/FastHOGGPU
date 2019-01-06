
#include "runnercuda.h"
#include "SVMManagerGPU.cuh"
#include "ImageManagerGPU.cuh"

std::unique_ptr<ISVMManagerGPU> loadSVMWeights(float *svmWeights, size_t size) {
  std::unique_ptr<ISVMManagerGPU> out = std::unique_ptr<SVMManagerGPU>(new SVMManagerGPU(svmWeights, size));
  return out;
}

// 4 channels image
std::unique_ptr<IImageManagerGPU> loadImageToGPU(cv::Mat& imageMat){
  std::unique_ptr<IImageManagerGPU> out = std::unique_ptr<ImageManagerGPU>(new ImageManagerGPU(imageMat));
  return out;
}