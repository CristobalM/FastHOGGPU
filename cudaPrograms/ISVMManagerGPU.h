//
// Created by cristobal, 2018
//

#ifndef PROJECT_ISVMMANAGERGPU_H
#define PROJECT_ISVMMANAGERGPU_H

#include <cstddef>




class ISVMManagerGPU{
public:
  virtual size_t getWeightsArraySize() = 0;
  virtual void* getDeviceArray() = 0;

};

#endif //PROJECT_ISVMMANAGERGPU_H
