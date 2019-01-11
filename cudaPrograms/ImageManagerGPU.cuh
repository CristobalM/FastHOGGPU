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

__global__ void
kConvertMatFromUcharToUchar3(uchar* data, uchar3* result, int step, int channels, int width, int height);

class ImageManagerGPU : public IImageManagerGPU {
  int cols, rows, channels;
  int step;
  uchar3* dResult;
  float3* dImagePaddedF3;
  uchar3* dImagePaddedU3;

  int padding;
  int paddedWidth, paddedHeight;

public:
  ImageManagerGPU(cv::Mat& imageBGR);
  ~ImageManagerGPU();

  int getCols() override;

  int getRows() override;

  int getChannels() override;

  __host__ void convertMatFromUcharToUchar3(uchar* data, uchar3** dResult,
                                            int cols, int rows, int channels, int step);

  __host__ void convertDResultToFloat3();

  __host__ void ConvertUchar3ToFloat(uchar3* dInputU3, float3* dOutputF3, int width, int height);
  __host__ void ConvertFloat3ToUchar(float3* dInputF3, uchar3* dOutputU3, int width, int height);

  __host__ void computeGradient(float3* dInputF3, float3* dOutputGradX, float3* dOutputGradY, int width, int height);

  __host__ void computeMagnitudeAndAngles(float3* gradX, float3* gradY, float* magnitude,
                                                  float* angle, int width, int height);

  uchar3* getUchar3DeviceImage() override;

  std::unique_ptr<float3> getFloat3Image();
  std::unique_ptr<uchar3> getUchar3Image() override;

  // debug

  void debugGradient() override;
  void debugGradient2() override;

  int getDetectionHistogramSize() override;
};


#endif //PROJECT_IMAGEMANAGERGPU_H
