//
// Created by cristobal, 2018
//

#ifndef PROJECT_IMAGEMANAGERGPU_H
#define PROJECT_IMAGEMANAGERGPU_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <memory>

#include "IImageManagerGPU.h"
#include "runnercuda.h"

__global__ void kConvertMatFromUcharToUchar4(uchar* data, uchar4* result, int step, int channels);

class ImageManagerGPU : public IImageManagerGPU {
  int cols, rows, channels;
  int step;
  uchar4* dResult;
  float4* dImagePaddedF4;
  uchar4* dImagePaddedU4;

  int padding;
  int paddedWidth, paddedHeight;

public:
  ImageManagerGPU(cv::Mat& imageBGRA);
  ~ImageManagerGPU();

  int getCols() override;

  int getRows() override;

  int getChannels() override;

  __host__ void convertMatFromUcharToUchar4(uchar* data, uchar4** dResult,
  int cols, int rows, int channels, int step);


  __host__ void ConvertUchar4ToFloat(uchar4* dInputU4, float4* dOutputF4, int width, int height);
  __host__ void ConvertFloat4ToUchar(float4* dInputF4, uchar4* dOutputU4, int width, int height);

  uchar4* getUchar4DeviceImage() override;

  std::unique_ptr<uchar4> getUchar4Image() override;
};


#endif //PROJECT_IMAGEMANAGERGPU_H
