//
// Created by cristobal, 2018
//

#include <cstddef>
#include "SVMManagerGPU.cuh"
#include "defs.h"


__host__ SVMManagerGPU::SVMManagerGPU(float* SVMWeights, size_t size) : weightsArraySize(size) {
  size_t totalBytes = size * sizeof(float);


  cf = cudaCreateChannelDesc<float>();

  gpuErrchk(cudaMallocArray(&dSVMWeights, &cf, size, 1));
  gpuErrchk(cudaMemcpyToArray(dSVMWeights, 0, 0, SVMWeights, totalBytes, cudaMemcpyHostToDevice));
  gpuErrchk(cudaBindTextureToArray(texRef, dSVMWeights, cf));


}

__host__ SVMManagerGPU::~SVMManagerGPU() {
  gpuErrchk(cudaUnbindTexture(texRef));
  gpuErrchk(cudaFree(dSVMWeights));
}

size_t SVMManagerGPU::getWeightsArraySize() {
  return weightsArraySize;
}

void* SVMManagerGPU::getDeviceArray() {
  return (void*) dSVMWeights;
}
