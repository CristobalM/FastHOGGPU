//
// Created by cristobal, 2018
//

#ifndef PROJECT_SVM_MANAGER_GPU_H
#define PROJECT_SVM_MANAGER_GPU_H
#include <cuda.h>
#include <cuda_runtime.h>
#include "ISVMManagerGPU.h"

#include <cstddef>
texture<float, 1, cudaReadModeElementType> texRef;


class SVMManagerGPU : public ISVMManagerGPU{
  size_t weightsArraySize;
  cudaArray *dSVMWeights;
  cudaChannelFormatDesc cf;
public:
  __host__ explicit SVMManagerGPU(float* SVMWeights, size_t size);
  __host__ ~SVMManagerGPU();

  size_t getWeightsArraySize() override;
  void* getDeviceArray() override;


};

#endif //PROJECT_SVM_MANAGER_GPU_H
