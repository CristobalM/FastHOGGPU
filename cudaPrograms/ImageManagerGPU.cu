//
// Created by cristobal, 2018
//

#include <opencv2/core/mat.hpp>
#include "ImageManagerGPU.cuh"
#include <iostream>
#include "defs.h"

#include <cuda.h>
#include <cuda_runtime.h>
// KERNELS
__global__ void kConvertMatFromUcharToUchar4(uchar *data, uchar4 *result, int step, int channels) {
  int row = blockIdx.x;
  int col = threadIdx.x;

  int base = channels*(step*row + col);
  int b_i = base + 0;
  int g_i = base + 1;
  int r_i = base + 2;
  int a_i = base + 3;

  uchar b = data[b_i];
  uchar g = data[g_i];
  uchar r = data[r_i];
  uchar a = data[a_i];

  int id = row*blockDim.x + col;
  result[id]= {
          x: r,
          y: g,
          z: b,
          w: a
  };
}

__global__ void kConvertU4ToFloat(uchar4* dInputU4, float4* dOutputF4, int width, int height){
  int offsetX = blockIdx.x * blockDim.x + threadIdx.x;
  int offsetY = blockIdx.y * blockDim.y + threadIdx.y;

  if(offsetX < width && offsetY < height){
    int offsetBlock = blockIdx.x * blockDim.x + blockIdx.y * blockDim.y * width;
    int offset = offsetBlock + threadIdx.x + threadIdx.y * width;

    uchar4 pixelU4 = dInputU4[offset];
    dOutputF4[offset] = {
            .x = (float)pixelU4.x,
            .y = (float)pixelU4.y,
            .z = (float)pixelU4.z,
            .w = (float)pixelU4.w
    };
  }
}


__global__ void kConvertF4ToUchar(float4* dInputF4, uchar4* dOutputU4, int width, int height){
  int offsetX = blockIdx.x * blockDim.x + threadIdx.x;
  int offsetY = blockIdx.y * blockDim.y + threadIdx.y;

  if(offsetX < width && offsetY < height){
    int offsetBlock = blockIdx.x * blockDim.x + blockIdx.y * blockDim.y * width;
    int offset = offsetBlock + threadIdx.x + threadIdx.y * width;

    float4 pixelF4 = dInputF4[offset];
    dOutputU4[offset] = {
            .x = (uchar)pixelF4.x,
            .y = (uchar)pixelF4.y,
            .z = (uchar)pixelF4.z,
            .w = (uchar)pixelF4.w
    };
  }
}

// ImageManagerGPU

ImageManagerGPU::ImageManagerGPU(cv::Mat& imageBGRA) {
  auto rows_ = imageBGRA.rows;
  auto cols_ = imageBGRA.cols;
  rows = rows_;
  cols = cols_;
  channels = 4;
  step = cols;
  auto* data = (uchar*)imageBGRA.data;
  dResult = nullptr;
  convertMatFromUcharToUchar4(data, &dResult, cols, rows, channels, step);

  padding = 3;
  paddedWidth = cols + padding*2;
  paddedHeight = rows + padding*2;



}

__host__ void ImageManagerGPU::convertMatFromUcharToUchar4(uchar* data, uchar4** dResult,
        int cols, int rows, int channels, int step) {
  auto szResult = cols*rows;
  auto szResultBytes = szResult * sizeof(uchar4);
  auto szInput = cols*rows*channels;
  auto szInputBytes = szInput * sizeof(uchar);

  std::unique_ptr<uchar4> hResult(new uchar4[szResult]);
  uchar* dInput;

  gpuErrchk(cudaMalloc((void**)dResult, szResultBytes));
  gpuErrchk(cudaMalloc((void**)&dInput, szInputBytes));

  gpuErrchk(cudaMemcpy(dInput, data, szInputBytes, cudaMemcpyHostToDevice));

  dim3 grid(rows);
  dim3 block(cols);

  kConvertMatFromUcharToUchar4<<<grid, block>>> (dInput, *dResult, step, this->channels);

  gpuErrchk(cudaFree(dInput));
}

ImageManagerGPU::~ImageManagerGPU() {
  if(dResult != nullptr)
    gpuErrchk(cudaFree(dResult));

  if(dImagePaddedF4 != nullptr)
    gpuErrchk(cudaFree(dImagePaddedF4));

  if(dImagePaddedU4 != nullptr)
    gpuErrchk(cudaFree(dImagePaddedU4));

}




int ImageManagerGPU::getCols() {
  return cols;
}

int ImageManagerGPU::getRows() {
  return rows;
}

int ImageManagerGPU::getChannels() {
  return channels;
}

__host__ void ImageManagerGPU::initAllocate(){
  gpuErrchk(cudaMalloc((void**)&dImagePaddedF4, sizeof(float4) * paddedWidth * paddedHeight));
}

__host__ void ImageManagerGPU::initPadding(){
  gpuErrchk(cudaMalloc((void**)&dImagePaddedU4, sizeof(uchar4) * paddedWidth * paddedHeight));
}

__host__ void ImageManagerGPU::padStoredDeviceImage() {

  int paddedWidth = cols + padding*2;
  int paddedHeight = rows + padding*2;

  gpuErrchk(cudaMemset(dImagePaddedU4, 0, sizeof(uchar4) * paddedWidth * paddedHeight));
  /*
  gpuErrchk(cudaMemcpy2D(dImagePaddedU4 + padding + padding*paddedWidth,
          paddedWidth * sizeof(uchar4), dResult,
          cols * sizeof(uchar4), rows * sizeof(uchar4),
          rows, cudaMemcpyDeviceToDevice
          ));
  */
  gpuErrchk(cudaMemcpy(dImagePaddedU4 + padding + padding * paddedWidth, ))

  ConvertUchar4ToFloat(dImagePaddedU4, dImagePaddedF4, paddedWidth, paddedHeight);

}

__host__ inline int ceilDiv(int a, int b){
  return (a % b != 0) ? a/b + 1: a/b;
}

__host__ void ImageManagerGPU::ConvertUchar4ToFloat(uchar4* dInputU4, float4* dOutputF4, int width, int height) {
  dim3 threadsInBlock(16, 16);
  dim3 blocks(ceilDiv(width, 16), ceilDiv(height, 16));
  kConvertU4ToFloat<<< blocks, threadsInBlock >>>(dInputU4, dOutputF4, width, height);
}

__host__ void ImageManagerGPU::ConvertFloat4ToUchar(float4* dInputF4, uchar4* dOutputU4, int width, int height) {
  dim3 threadsInBlock(16, 16);
  dim3 blocks(ceilDiv(width, 16), ceilDiv(height, 16));
  kConvertF4ToUchar<<< blocks, threadsInBlock >>>(dInputF4, dOutputU4, width, height);
}


uchar4* ImageManagerGPU::getUchar4DeviceImage() {
  return dResult;
}

std::unique_ptr<uchar4> ImageManagerGPU::getUchar4Image() { // debug
  std::unique_ptr<uchar4> hostImage(new uchar4[rows*cols]);

  gpuErrchk(cudaMemcpy(hostImage.get(), dResult, sizeof(uchar4) * rows * cols, cudaMemcpyDeviceToHost));

  return hostImage;
}

void ImageManagerGPU::padImage() {
  initAllocate();
  initPadding();
  padStoredDeviceImage();
}

std::unique_ptr<uchar4> ImageManagerGPU::debugPadding(){
  padImage();
  std::unique_ptr<uchar4> result(new uchar4[paddedWidth * paddedHeight]);

  ConvertFloat4ToUchar(dImagePaddedF4, dImagePaddedU4, paddedWidth, paddedHeight);
  gpuErrchk(cudaMemcpy(result.get(), dImagePaddedU4, sizeof(uchar4) * paddedWidth * paddedHeight, cudaMemcpyDeviceToHost));

  return result;
}

