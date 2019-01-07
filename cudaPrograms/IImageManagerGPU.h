//
// Created by cristobal, 2018
//

#ifndef PROJECT_IIMAGEMANAGERGPU_H
#define PROJECT_IIMAGEMANAGERGPU_H


#include <memory>

#ifndef NVCC
using uchar = unsigned char;
typedef struct {
  uchar x, y, z, w;
} uchar4;
#endif

class IImageManagerGPU{
public:
  //virtual std::unique_ptr<uchar4> convertMatFromUcharToUchar4(uchar* data, int cols, int rows, int channels, int step) = 0;
  virtual int getCols() = 0;
  virtual int getRows() = 0;
  virtual int getChannels() = 0;
  virtual uchar4* getUchar4DeviceImage() = 0;
  virtual std::unique_ptr<uchar4> getUchar4Image() = 0;

};
#endif //PROJECT_IIMAGEMANAGERGPU_H
