//
// Created by cristobal, 2018
//

#ifndef PROJECT_IIMAGEMANAGERGPU_H
#define PROJECT_IIMAGEMANAGERGPU_H


#include <memory>
#include <vector>

#include "common_structs.h"

#ifndef NVCC
#ifndef UC3
#define UC3
using uchar = unsigned char;
typedef struct uchar3{
  uchar x, y, z;
} uchar3;
#endif
#endif

class IImageManagerGPU{
public:
  virtual int getCols() = 0;
  virtual int getRows() = 0;
  virtual int getChannels() = 0;
  virtual uchar3* getUchar3DeviceImage() = 0;
  virtual std::unique_ptr<uchar3> getUchar3Image() = 0;
  virtual void debugGradient() = 0;
  virtual void debugGradient2() = 0;

  // debug
  virtual int getDetectionHistogramSize()= 0;
  virtual std::vector<ResultSVMScore> detectWithSVM() = 0;

};
#endif //PROJECT_IIMAGEMANAGERGPU_H
