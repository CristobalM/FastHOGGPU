//
// Created by cristobal, 2018
//

#include "ImageManagerGPU.cuh"
#include <iostream>
#include "defs.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <math_constants.h>

#include "../SVM/persondetectorwt.tcc"
#include "../../../../../usr/local/cuda-10.0/include/texture_types.h"

#include <utility>


const int DETECTION_WINDOW_WIDTH = 64;
const int DETECTION_WINDOW_HEIGHT = 128;
const int DETECTION_WINDOW_STRIDE_X = DETECTION_WINDOW_WIDTH/2;
//const int DETECTION_WINDOW_STRIDE_Y = DETECTION_WINDOW_STRIDE_X;
const int DETECTION_WINDOW_STRIDE = DETECTION_WINDOW_STRIDE_X;

const int CELL_WIDTH = 8;
const int CELL_HEIGHT = 8;
const int BLOCK_DIM_X = 2;
const int BLOCK_DIM_Y = 2;
const int BLOCK_WIDTH = CELL_WIDTH * BLOCK_DIM_X;
const int BLOCK_HEIGHT = CELL_HEIGHT * BLOCK_DIM_Y;


// KERNELS
__global__ void
kConvertMatFromUcharToUchar3(uchar* data, uchar3* result, int step, int channels, int width, int height) {
  //int row = blockIdx.x;
  //int col = threadIdx.x;
  int idx = threadIdx.x + blockIdx.x*blockDim.x;
  if(idx >= width * height)
    return;

  int col = idx % width;
  int row = idx / width;

  int base = channels*(step*row + col);
  int b_i = base + 0;
  int g_i = base + 1;
  int r_i = base + 2;

  uchar b = data[b_i];
  uchar g = data[g_i];
  uchar r = data[r_i];

  result[idx]= {
          x: b,
          y: g,
          z: r
  };
}

__global__ void kConvertU3ToFloat(uchar3* dInputU3, float3* dOutputF3,float4* dOutputF4, int width, int height){
  //int idx = blockDim.x * blockIdx.x + threadIdx.x;
  int idx = threadIdx.x + blockIdx.x*blockDim.x;

  uchar3 pixelU3 = dInputU3[idx];
  dOutputF3[idx] = {
          .x = (float)pixelU3.x,
          .y = (float)pixelU3.y,
          .z = (float)pixelU3.z
  };
  dOutputF4[idx] = {
          .x = (float)pixelU3.x,
          .y = (float)pixelU3.y,
          .z = (float)pixelU3.z,
          .w = (float)0
  };

}


__global__ void kConvertF3ToUchar(float3* dInputF3, uchar3* dOutputU3, int width, int height){
  int offsetX = blockIdx.x * blockDim.x + threadIdx.x;
  int offsetY = blockIdx.y * blockDim.y + threadIdx.y;

  if(offsetX < width && offsetY < height){
    int offsetBlock = blockIdx.x * blockDim.x + blockIdx.y * blockDim.y * width;
    int offset = offsetBlock + threadIdx.x + threadIdx.y * width;

    float3 pixelF3 = dInputF3[offset];
    dOutputU3[offset] = {
            .x = (uchar)pixelF3.x,
            .y = (uchar)pixelF3.y,
            .z = (uchar)pixelF3.z
    };
  }
}

__global__ void kComputeGradient(float3* dInputF3, float3* gradX, float3* gradY, int width, int height) {
  int idx = threadIdx.x + blockIdx.x*blockDim.x;
  int j = idx % width;
  int i = idx / width;


  if(idx >= width*height){
    return;
  }

  float3 xgrad;
  float3 ygrad;
  float3 prevX, prevY, afterX, afterY;
  if(j > 0)
    prevX = dInputF3[idx-1];
  if(j < width-1)
    afterX = dInputF3[idx+1];
  if(i > 0)
    prevY = dInputF3[idx-width];
  if(i < height-1)
    afterY = dInputF3[idx+width];

  if(j > 0 && j < width-1){
    xgrad.x = afterX.x - prevX.x ;
    xgrad.y = afterX.y - prevX.y ;
    xgrad.z = afterX.z - prevX.z ;
  }
  else if(j == 0){
    xgrad.x = afterX.x;
    xgrad.y = afterX.y;
    xgrad.z = afterX.z;
  }
  else{
    xgrad.x = -prevX.x;
    xgrad.y = -prevX.y;
    xgrad.z = -prevX.z;
  }

  if(i > 0 && i < height-1){
    ygrad.x = -prevY.x + afterY.x;
    ygrad.y = -prevY.y + afterY.y;
    ygrad.z = -prevY.z + afterY.z;
  }
  else if(i == 0){
    ygrad.x = afterY.x;
    ygrad.y = afterY.y;
    ygrad.z = afterY.z;
  }
  else{
    ygrad.x = -prevY.x;
    ygrad.y = -prevY.y;
    ygrad.z = -prevY.z;
  }

  gradX[idx] = xgrad;
  gradY[idx] = ygrad;
}

__global__ void kComputeMagnitudeAndAngle(float3* gradX, float3* gradY, float* magnitude,
        float* angle, int width, int height){

  int idx = threadIdx.x + blockIdx.x*blockDim.x;

  float3 gX = gradX[idx];
  float3 gY = gradY[idx];

  float magnitude_0 = sqrtf(gX.x*gX.x + gY.x*gY.x);
  float magnitude_1 = sqrtf(gX.y*gX.y + gY.y*gY.y);
  float magnitude_2 = sqrtf(gX.z*gX.z + gY.z*gY.z);

  float aMagnitude;
  float anAngle;

  float max_02 = fmaxf(magnitude_0, magnitude_2);
  float max_12 = fmaxf(magnitude_1, magnitude_2);

  if(magnitude_0 > max_12){
    aMagnitude = magnitude_0;
    anAngle = atan2f(gY.x, gX.x);
  }
  else if(magnitude_1 > max_02){
    aMagnitude = magnitude_1;
    anAngle = atan2f(gY.y, gX.y);
  }
  else{
    aMagnitude = magnitude_2;
    anAngle = atan2f(gY.z, gX.z);
  }

  magnitude[idx] = aMagnitude;
  angle[idx] = anAngle;
}


int calcContainedWithStride(int width, int height, int boxWidth, int boxHeight, int strideWidth, int strideHeight){
  int width_contained = (width - boxWidth)/strideWidth;
  int height_contained = (height - boxHeight)/strideHeight;
  return (width_contained + 1) * (height_contained + 1);
}

const int BLOCK_WINDOW_HISTOGRAM_SIZE = 36;

const int DETECTION_WINDOW_HISTOGRAM_SIZE = calcContainedWithStride(DETECTION_WINDOW_WIDTH, DETECTION_WINDOW_HEIGHT,
        BLOCK_WIDTH, BLOCK_HEIGHT, CELL_WIDTH, CELL_HEIGHT) * BLOCK_WINDOW_HISTOGRAM_SIZE;

__global__ void kComputeBlockHistogram(float* magnitude, float* angle, float* histogram,
        int width, int height, int blockSize, int blocksDimX, int blocksDimY, int cellSize, int blockStride){

  int idx = threadIdx.x + blockIdx.x*blockDim.x;
  if(idx >= blocksDimX * blocksDimY)
    return;

  int blockX = idx%blocksDimX;
  int blockY = idx/blocksDimX;
  int baseJ = blockX * blockStride;
  int baseI = blockY * width * blockStride;


  int cell1I = baseI;
  int cell1J = baseJ;
  int cell2I = baseI;
  int cell2J = baseJ + cellSize;
  int cell3I = baseI + cellSize*width;
  int cell3J = baseJ;
  int cell4I = baseI + cellSize*width;
  int cell4J = baseJ + cellSize;

  int cellsToSelect[4][2];
  cellsToSelect[0][0] = cell1I;
  cellsToSelect[0][1] = cell1J;
  cellsToSelect[1][0] = cell2I;
  cellsToSelect[1][1] = cell2J;
  cellsToSelect[2][0] = cell3I;
  cellsToSelect[2][1] = cell3J;
  cellsToSelect[3][0] = cell4I;
  cellsToSelect[3][1] = cell4J;
  
  int selectedCell;

  int hist_base = 36 * idx;

  const int halfBlockSize = blockSize/2;

  const int a = halfBlockSize/2;
  const int b = halfBlockSize + a;
  const float delta_bins = (float)(CUDART_PI_F / 8.0);


  float blockHistogram[36];
  memset(blockHistogram, 0, 36 * sizeof(float));

  int inow, jnow;
  for(int i = 0; i < blockSize; i++) {
    for (int j = 0; j < blockSize; j++) {
      float alpha1 = fabsf(b - j);
      float beta1 = fabsf(b - i);
      float contr_c1 = 0, contr_c2 = 0, contr_c3 = 0, contr_c4 = 0;

      if(i < halfBlockSize && j < halfBlockSize){
        inow = i; jnow = j; selectedCell = 0;
      }
      else if(i < halfBlockSize && j >= halfBlockSize){
        inow = i; jnow = j - halfBlockSize; selectedCell = 1;
      }
      else if(i >= halfBlockSize && j < halfBlockSize){
        inow = i - halfBlockSize; jnow = j; selectedCell = 2;
      }
      else if(i >= halfBlockSize && j >= halfBlockSize){
        inow = i - halfBlockSize; jnow = j - halfBlockSize;selectedCell = 3;
      }
      else{
        printf("ERROR(kComputeBlockHistogram): UNEXPECTED CASE\n");
        return;
      }
      int* cell = cellsToSelect[selectedCell];
      int cellI = cell[0];
      int cellJ = cell[1];
      
      int pixelI = cellI + width*inow;
      int pixelJ = cellJ + jnow;
      int pixelIdx = pixelI + pixelJ;
      float pixelMagnitude = magnitude[pixelIdx];
      float pixelAngle = angle[pixelIdx];

      if(i < halfBlockSize/2 && j < halfBlockSize/2){
        contr_c1 = pixelMagnitude;
      }
      else if(i < halfBlockSize/2 && j >= halfBlockSize/2 && j < 3*halfBlockSize/2){
        contr_c1 = alpha1*pixelMagnitude;
        contr_c2 = (halfBlockSize - alpha1)*pixelMagnitude;
      }
      else if(i < halfBlockSize/2 && j >= 3*halfBlockSize/2){
        contr_c2 = pixelMagnitude;
      }
      else if(i >= halfBlockSize/2 && i < 3*halfBlockSize/2 && j < halfBlockSize/2){
        contr_c1 = beta1*pixelMagnitude;
        contr_c3 = (halfBlockSize - beta1)*pixelMagnitude;
      }
      else if(i >= 3*halfBlockSize/2 && j < halfBlockSize/2){
        contr_c3 = pixelMagnitude;
      }
      else if(i >= 3*halfBlockSize/2 && j >= halfBlockSize/2 && j < 3*halfBlockSize/2){
        contr_c3 = alpha1*pixelMagnitude;
        contr_c4 = (halfBlockSize - alpha1)*pixelMagnitude;
      }
      else if(i >= 3*halfBlockSize/2 && j >= 3*halfBlockSize/2){
        contr_c4 = pixelMagnitude;
      }
      else if(i >= halfBlockSize/2 && i < 3*halfBlockSize/2 && j >= 3*halfBlockSize/2){
        contr_c2 = beta1*pixelMagnitude;
        contr_c4 = (halfBlockSize - beta1)*pixelMagnitude;
      }
      else{
        contr_c1 = alpha1*beta1*pixelMagnitude;
        contr_c2 = (halfBlockSize - alpha1)*beta1*pixelMagnitude;
        contr_c3 = alpha1*(halfBlockSize - beta1)*pixelMagnitude;
        contr_c4 = (halfBlockSize - alpha1)*(halfBlockSize - beta1)*pixelMagnitude;
      }

      float bin = pixelAngle / delta_bins;
      float lower_bin_d = floorf(bin);

      int lower_bin = (int)lower_bin_d;
      int upper_bin = (int)ceilf(bin);

      float dist_low = bin - lower_bin_d;
      float dist_up = (float)(1.0 - dist_low);

      lower_bin = (9 + lower_bin) % 9;
      upper_bin = (9 + upper_bin) % 9;

      float contr_c1_lb =  (contr_c1 * dist_up);
      float contr_c1_ub =  (contr_c1 * dist_low);
      float contr_c2_lb =  (contr_c2 * dist_up);
      float contr_c2_ub =  (contr_c2 * dist_low);
      float contr_c3_lb =  (contr_c3 * dist_up);
      float contr_c3_ub =  (contr_c3 * dist_low);
      float contr_c4_lb =  (contr_c4 * dist_up);
      float contr_c4_ub =  (contr_c4 * dist_low);


      blockHistogram[lower_bin] += contr_c1_lb;
      blockHistogram[upper_bin] += contr_c1_ub;
      blockHistogram[lower_bin + 9] += contr_c2_lb;
      blockHistogram[upper_bin + 9] += contr_c2_ub;
      blockHistogram[lower_bin + 18]+= contr_c3_lb;
      blockHistogram[upper_bin + 18]+= contr_c3_ub;
      blockHistogram[lower_bin + 27]+= contr_c4_lb;
      blockHistogram[upper_bin + 27]+= contr_c4_ub;
    }
  }

  float sum = 0;
  for(int i = 0; i < 4; i++){
    for(int j = 0; j < 9; j++){
      const int idxji = j + i*9;
      float r = blockHistogram[idxji];
      sum += r*r;
    }
  }

  float norm = sqrtf(sum);
  float sum2 = 0;

  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 9; j++) {
      const int idxji = j + i * 9;
      blockHistogram[idxji] /= norm;
      blockHistogram[idxji] = fminf(0.2, blockHistogram[idxji]);
      float r = blockHistogram[idxji];
      sum2 += r*r;
    }
  }

  float norm2 = sqrtf(sum2);
  for(int i = 0; i < 4; i++) {
    for (int j = 0; j < 9; j++) {
      const int idxji = j + i * 9;
      blockHistogram[idxji] /= norm2;
      blockHistogram[idxji] = fminf(0.2, blockHistogram[idxji]);
    }
  }
  
  for(int i = 0; i < 36; i++){
    histogram[hist_base + i] = blockHistogram[i];
  }
}

__global__ void kEvalSVM(float* histograms, float* svmScores, float* trainedSVM,
        int detWindowsDimX, int detWindowsDimY, int winStride, int winSizeX, int winSizeY,
        int blockDimAllX, int blockDimAllY, int blockSize, int blockStride,
        int width, int height,
        float svmBias){
  int idx = threadIdx.x + blockIdx.x*blockDim.x; // idx of detection window

  if(idx >= detWindowsDimX * detWindowsDimY)
    return;

  int dWX = idx%detWindowsDimX;
  int dWY = idx/detWindowsDimX;

  int x = dWX * winStride;
  int y = dWY * winStride;

  //int baseJ = x;
  //int baseI = y * width;

  int blocksInWindowDimX = (winSizeX - blockSize)/blockStride + 1;
  int blocksInWindowDimY = (winSizeY - blockSize)/blockStride + 1;
  int totalBlocksInWindow = blocksInWindowDimX * blocksInWindowDimY;

  int blockJ = x/blockStride;
  int blockI = y/blockStride;


  //int blockI = (y + winSizeY - blockSize)/blockStride + 1;
  //int blockJ = (x + winSizeX - blockSize)/blockStride + 1;

  float dotProd = 0;
  for(int iB = 0; iB < blocksInWindowDimY; iB++){
    int current = (blockI + iB) * blockDimAllX + blockJ;
    for(int jB = 0; jB < blocksInWindowDimX; jB++){
      int blockIdx = current + jB;
      int iH = iB * blocksInWindowDimY + jB;
      for(int jH = 0; jH < 36; jH++){
        dotProd += histograms[blockIdx*36+jH] * trainedSVM[iH*36 + jH];
      }
    }
  }

  for(int iH = 0; iH < totalBlocksInWindow; iH++){
    for(int jH = 0; jH < 36; jH++){
      int histIdx = iH * 36 + jH;
      dotProd += histograms[histIdx] * trainedSVM[histIdx];
    }
  }

  float thisSVMScore = dotProd - svmBias;
  svmScores[idx] = thisSVMScore;
}
cudaArray *imageArray = 0;
texture<float4, 2, cudaReadModeElementType> tex;
cudaChannelFormatDesc channelDescDownscale;
bool isAllocated;
__host__ void InitScale(){
  channelDescDownscale = cudaCreateChannelDesc<float4>();
  tex.filterMode = cudaFilterModeLinear;
  tex.normalized = false;
  isAllocated = false;
}

__global__ void kDownscaleImg(float3* outputImg, int widthOutput, int heightOutput, float scale){
  int idx = threadIdx.x + blockIdx.x*blockDim.x; // idx of detection window

  if(idx >= widthOutput * heightOutput)
    return;

  int x = idx % widthOutput;
  int y = idx / widthOutput;

  float u = x * scale;
  float v = y * scale;

  float4 result = tex2D(tex, u, v);
  float3 result3;

  result3.x = sqrtf(result.x);
  result3.y = sqrtf(result.y);
  result3.z = sqrtf(result.z);

  outputImg[idx] = result3;
}

__global__ void kApplyGamma(float3* dInputImage, float3* dOutputImage, int width, int height){
  int idx = threadIdx.x + blockIdx.x*blockDim.x; // idx of detection window

  if(idx >= width * height)
    return;


  dOutputImage[idx].x = sqrtf(dInputImage[idx].x);
  dOutputImage[idx].y = sqrtf(dInputImage[idx].y);
  dOutputImage[idx].z = sqrtf(dInputImage[idx].z);
}

// ImageManagerGPU

ImageManagerGPU::ImageManagerGPU(cv::Mat& imageBGR) {
  auto rows_ = imageBGR.rows;
  auto cols_ = imageBGR.cols;

 blocksInDetWindowDimX = (DETECTION_WINDOW_WIDTH - BLOCK_WIDTH)/CELL_WIDTH + 1;
 blocksInDetWindowDimY = (DETECTION_WINDOW_HEIGHT - BLOCK_HEIGHT)/CELL_WIDTH + 1;
 totalBlocksInDetWindow = blocksInDetWindowDimX * blocksInDetWindowDimY;
 svmVectorSize = totalBlocksInDetWindow * 36;

  rows = rows_;
  cols = cols_;
  channels = 3;
  step = cols;
  auto* data = (uchar*)imageBGR.data;
  dResult = nullptr;
  InitScale();

  convertMatFromUcharToUchar3(data, &dResult, cols, rows, channels, step);

  paddedWidth = cols;
  paddedHeight = rows;
  dImagePaddedF3 = nullptr;
  dImagePaddedF4 = nullptr;
  dImagePaddedU3 = nullptr;
  trainedSVM = nullptr;

  gpuErrchk(cudaMalloc(&trainedSVM, sizeof(float) * svmVectorSize));

  assert(svmVectorSize == PERSON_WEIGHT_VEC_LENGTH);
  gpuErrchk(cudaMemcpy(trainedSVM, PERSON_WEIGHT_VEC, sizeof(float) * svmVectorSize, cudaMemcpyHostToDevice));

}


__host__ void ImageManagerGPU::convertMatFromUcharToUchar3(uchar* data, uchar3** dResult,
                                                           int cols, int rows, int channels, int step) {
  auto szResult = cols*rows;
  auto szResultBytes = szResult * sizeof(uchar3);
  auto szInput = cols*rows*channels;
  auto szInputBytes = szInput * sizeof(uchar);

  uchar* dInput;

  gpuErrchk(cudaMalloc((void**)dResult, szResultBytes));
  gpuErrchk(cudaMalloc((void**)&dInput, szInputBytes));

  gpuErrchk(cudaMemcpy(dInput, data, szInputBytes, cudaMemcpyHostToDevice));

  dim3 grid(rows);
  dim3 block(cols);
  int width = cols;
  int height = rows;

  //std::cout << "image rows = " << rows << ", cols=" << cols << std::endl;
  const int threadsByBlock = 32;
  double b = (width*height)/((double)threadsByBlock);
  int b_int = (int)std::ceil(b);
  kConvertMatFromUcharToUchar3 <<<b_int, threadsByBlock>>> (dInput, *dResult, step, this->channels, width, height);
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );
  gpuErrchk(cudaFree(dInput));
}

ImageManagerGPU::~ImageManagerGPU() {
  if(dResult != nullptr)
    gpuErrchk(cudaFree(dResult));

  if(dImagePaddedF3 != nullptr)
    gpuErrchk(cudaFree(dImagePaddedF3));

  if(dImagePaddedU3 != nullptr)
    gpuErrchk(cudaFree(dImagePaddedU3));
  if(trainedSVM != nullptr)
    gpuErrchk(cudaFree(trainedSVM));
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

__host__ inline int ceilDiv(int a, int b){
  return (a % b != 0) ? a/b + 1: a/b;
}

__host__ void ImageManagerGPU::ConvertUchar3ToFloat(uchar3* dInputU3, float3* dOutputF3, float4* dOutputF4, int width, int height) {
  const int threadsByBlock = 32;
  double b = (width*height)/((double)threadsByBlock);
  int b_int = (int)std::ceil(b);
  //std::cout << "ConvertUchar3ToFloat:: blocks = " << b_int << " threadsByBlock = " << threadsByBlock << std::endl;
  kConvertU3ToFloat <<< b_int, threadsByBlock >>>(dInputU3, dOutputF3, dOutputF4, width, height);
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );

}

__host__ void ImageManagerGPU::ConvertFloat3ToUchar(float3* dInputF3, uchar3* dOutputU3, int width, int height) {
  dim3 threadsInBlock(16, 16);
  dim3 blocks(ceilDiv(width, 16), ceilDiv(height, 16));
  kConvertF3ToUchar <<< blocks, threadsInBlock >>>(dInputF3, dOutputU3, width, height);
}


uchar3* ImageManagerGPU::getUchar3DeviceImage() {
  return dResult;
}

std::unique_ptr<uchar3> ImageManagerGPU::getUchar3Image() { // debug
  assert(dResult != nullptr);
  std::unique_ptr<uchar3> hostImage(new uchar3[rows*cols]);

  gpuErrchk(cudaMemcpy(hostImage.get(), dResult, sizeof(uchar3) * rows * cols, cudaMemcpyDeviceToHost));

  return hostImage;
}

__host__
void ImageManagerGPU::computeGradient(float3* dInputF3, float3* dOutputGradX, float3* dOutputGradY, int width,
                                      int height) {
  const int threadsByBlock = 1024;
  double b = (width*height)/((double)threadsByBlock);
  int b_int = (int)std::ceil(b);
  //dim3 threadsInblock(threadsByBlock);
  //dim3 blocks(b_int);
  //std::cout << "computeGradient:: blocks = " << b_int << " threadsPerBlock = " << threadsByBlock << std::endl;
  //std::cout << "width=" << width << ", height=" << height << std::endl;
  //threadIdx.x + blockIdx.x*blockDim.x
  //int maxIdx = b_int*threadsByBlock;
  //std::cout << "max Idx = " << maxIdx << "; width*height = " << width * height << std::endl;
  kComputeGradient<<< b_int, threadsByBlock >>>(dInputF3, dOutputGradX, dOutputGradY, width, height);
  //gpuErrchk( cudaPeekAtLastError() );
  //gpuErrchk( cudaDeviceSynchronize() );
  //std::cout << "hola" << std::endl;
}

__host__
void ImageManagerGPU::computeMagnitudeAndAngles(float3* gradX, float3* gradY, float* magnitude,
        float* angle, int width, int height) {
  const int threadsByBlock = 32;
  double b = (width*height)/((double)threadsByBlock);
  int b_int = (int)std::ceil(b);
  //std::cout << "computeMagnitudeAndAngles:: blocks = " << b_int << " threadsPerBlock = " << threadsByBlock << std::endl;
  //kComputeMagnitudeAndAngle<<< b_int, threadsByBlock >>>(gradX, gradY, magnitude, angle,
  kComputeMagnitudeAndAngle<<< b_int, threadsByBlock >>>(gradX, gradY, magnitude, angle,
          width, height);
}



template <typename T>
void debugPrintMatrixFirstFlat(T* matrix, int width, int height){
  for(int i  = 0; i < height; i++){
    for(int j = 0; j < width; j++){
      std::cout << matrix[i*width + j].x << " ";
    }
    std::cout << std::endl;
  }
}

template <typename T>
void debugPrintCVMatFirstFlat(cv::Mat& mat, int width, int height){
  for(int i  = 0; i < height; i++){
    for(int j = 0; j < width; j++){
      std::cout << mat.at<T>(i, j)[0] << " ";
    }
    std::cout << std::endl;
  }
}

template <typename inputImageType3, typename outputImageType1, typename cvMatAcc>
cv::Mat convertToCVImage(inputImageType3* img, int width, int height, int cvCode){
  //std::unique_ptr<float> preResult(new float[width*height*3]);
  cv::Mat result(cv::Size(width, height), cvCode);

  for(int i = 0; i < height; i++){
    for(int j = 0; j < width; j++){
      int base = width*i + j;
      cvMatAcc color;
      color[0] = (outputImageType1)img[base].x;
      color[1] = (outputImageType1)img[base].y;
      color[2] = (outputImageType1)img[base].z;
      result.at<cvMatAcc>(i, j) = color;
    }
  }

  return result;
}

template <typename inputImageType3, typename outputImageType1, typename cvMatAcc>
cv::Mat convertToCVImage1D(inputImageType3* img, int width, int height, int cvCode){
  //std::unique_ptr<float> preResult(new float[width*height*3]);
  cv::Mat result(cv::Size(width, height), cvCode);

  for(int i = 0; i < height; i++){
    for(int j = 0; j < width; j++){
      int base = width*i + j;
      cvMatAcc color;
      color = (outputImageType1)img[base];
      result.at<cvMatAcc>(i, j) = color;
    }
  }

  return result;
}

template <typename matType,  typename floatDim>
float debugCompareMatrices(cv::Mat& cvMat, floatDim* f3Mat, int width, int height){
  float sum = 0;
  for(int i = 0; i < height; i++){
    for(int j = 0; j < width; j++){
      int base = i * width + j;
      float currentDiff0 = cvMat.at<matType>(i, j)[0] - f3Mat[base].x;
      float currentDiff1 = cvMat.at<matType>(i, j)[1] - f3Mat[base].y;
      float currentDiff2 = cvMat.at<matType>(i, j)[2] - f3Mat[base].z;
      sum += currentDiff0*currentDiff0;
      sum += currentDiff1*currentDiff1;
      sum += currentDiff2*currentDiff2;
    }
  }
  sum = std::sqrt(sum);
  return sum;
}

std::pair<int, int> getWindowCoordinates(int windowIdx,
        int windowStrideX, int windowStrideY,
        int windowsDimX){

  int idxX = windowIdx%windowsDimX;
  int idxY = windowIdx/windowsDimX;

  int x = idxX*windowStrideX;
  int y = idxY*windowStrideY;

  return {x, y};
}


void ImageManagerGPU::debugGradient() {
  if(dImagePaddedF3 == nullptr)
    convertDResultToFloat3();

  auto realImage = getUchar3Image();
  auto f3Image = getFloat3Image();

  auto cvF3Image = convertToCVImage<float3, float, cv::Vec3f>(f3Image.get(), cols, rows, CV_32FC3);
  auto cvU3ImageReal = convertToCVImage<uchar3, uchar, cv::Vec3b>(realImage.get(), cols, rows, CV_8UC3);

  cv::imwrite("cvF3image.png", cvF3Image);
  cv::imwrite("cvU3ImageReal.png", cvU3ImageReal);

  float3* dGammaCorrectedImg;
  std::unique_ptr<float3> hGammaCorrectedImg(new float3[paddedWidth * paddedHeight]);
  gpuErrchk(cudaMalloc(&dGammaCorrectedImg, sizeof(float3) * paddedWidth * paddedHeight));
  /*
  gpuErrchk(cudaMemcpy(dGammaCorrectedImg, dImagePaddedF3, sizeof(float3) * paddedWidth * paddedHeight,
          cudaMemcpyDeviceToDevice ));
  */
  const int threadsByBlock = 32;
  double b = (paddedWidth*paddedHeight)/((double)threadsByBlock);
  int b_int = (int)std::ceil(b);

  kApplyGamma<<<b_int, threadsByBlock >>>(dImagePaddedF3, dGammaCorrectedImg, paddedWidth, paddedHeight);

  gpuErrchk(cudaMemcpy(hGammaCorrectedImg.get(), dGammaCorrectedImg, sizeof(float3) * paddedWidth * paddedHeight,
                       cudaMemcpyDeviceToHost));

  auto cvF3ImageGamma = convertToCVImage<float3, float, cv::Vec3f>(hGammaCorrectedImg.get(), cols, rows, CV_32FC3);
  cv::imwrite("cvF3ImageGamma.png", cvF3ImageGamma);


  std::unique_ptr<float3> gradX(new float3[paddedWidth * paddedHeight]);
  std::unique_ptr<float3> gradY(new float3[paddedWidth * paddedHeight]);

  float3 *dGradX, *dGradY;

  size_t f3Bytes = sizeof(float3) * paddedHeight * paddedWidth;

  gpuErrchk(cudaMalloc((void**)&dGradX, f3Bytes));
  gpuErrchk(cudaMalloc((void**)&dGradY, f3Bytes));

  computeGradient(dImagePaddedF3, dGradX, dGradY, paddedWidth, paddedHeight);

  gpuErrchk(cudaMemcpy(gradX.get(), dGradX, f3Bytes, cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(gradY.get(), dGradY, f3Bytes, cudaMemcpyDeviceToHost));

  auto gradXCV = convertToCVImage<float3, float, cv::Vec3f>(gradX.get(), paddedWidth, paddedHeight, CV_32FC3);
  auto gradYCV = convertToCVImage<float3, float, cv::Vec3f>(gradY.get(), paddedWidth, paddedHeight, CV_32FC3);
  auto sumGrad = cv::Mat(gradXCV + gradYCV);
  cv::imwrite("gradXCV.png", gradXCV);
  cv::imwrite("gradYCV.png", gradYCV);
  cv::imwrite("sumGrad.png", sumGrad);

  float *dMagnitudes, *dAngles;
  gpuErrchk(cudaMalloc(&dMagnitudes, sizeof(float) * paddedHeight * paddedWidth));
  gpuErrchk(cudaMalloc(&dAngles, sizeof(float) * paddedHeight * paddedWidth));

  computeMagnitudeAndAngles(dGradX, dGradY, dMagnitudes, dAngles, paddedWidth, paddedHeight);

  std::unique_ptr<float> hMagnitudes(new float[paddedHeight * paddedWidth]);
  std::unique_ptr<float> hAngles(new float[paddedHeight * paddedWidth]);

  gpuErrchk(cudaMemcpy(hMagnitudes.get(), dMagnitudes, sizeof(float) * paddedWidth * paddedHeight, cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(hAngles.get(), dAngles, sizeof(float) * paddedWidth * paddedHeight, cudaMemcpyDeviceToHost));
  auto cvMagnitudes = convertToCVImage1D<float, float, float>(hMagnitudes.get(), paddedWidth, paddedHeight, CV_32F);
  auto cvAngles = convertToCVImage1D<float, float, float>(hAngles.get(), paddedWidth, paddedHeight, CV_32F);
  cv::imwrite("cvMagnitudes.png", cvMagnitudes);
  cv::imwrite("cvAngles.png", cvAngles);


  int blockNum = calcContainedWithStride(paddedWidth, paddedHeight, BLOCK_WIDTH, BLOCK_HEIGHT, CELL_WIDTH, CELL_HEIGHT);
  int allHistogramsSize = 36 * blockNum;
  std::unique_ptr<float> hHistograms(new float[allHistogramsSize]);
  float* dHistograms;

  const int blocksDimX = ((paddedWidth - BLOCK_WIDTH)/CELL_WIDTH) + 1;
  const int blocksDimY = ((paddedHeight - BLOCK_HEIGHT)/CELL_HEIGHT) + 1;


  gpuErrchk(cudaMalloc(&dHistograms, sizeof(float) * allHistogramsSize));

  ComputeBlockHistogram(dMagnitudes, dAngles, dHistograms, paddedWidth, paddedHeight,
          BLOCK_WIDTH, blocksDimX, blocksDimY, CELL_WIDTH, CELL_WIDTH);

  cudaDeviceSynchronize();

  //std::cout << "Done ComputeBlockHistogram. allHistogramsSize=" << allHistogramsSize << std::endl;

  gpuErrchk(cudaMemcpy(hHistograms.get(), dHistograms, sizeof(float) * allHistogramsSize, cudaMemcpyDeviceToHost));

  for(int i = 0; i < 10; i++)
    std::cout << hHistograms.get()[i] << "  ";
  std::cout << std::endl;

  //int ddebug = DETECTION_WINDOW_STRIDE;
  const int detWindowsDimX = ((paddedWidth - DETECTION_WINDOW_WIDTH)/DETECTION_WINDOW_STRIDE) + 1;
  const int detWindowsDimY = ((paddedHeight - DETECTION_WINDOW_HEIGHT)/DETECTION_WINDOW_STRIDE) + 1;

  const int totalDetectionWindows = detWindowsDimX * detWindowsDimY;



  float* dSvmScores;

  gpuErrchk(cudaMalloc(&dSvmScores, sizeof(float) * totalDetectionWindows));
  //gpuErrchk(cudaMalloc(&trainedSVM, sizeof(float) * svmVectorSize));



  //assert(svmVectorSize == PERSON_WEIGHT_VEC_LENGTH);
  //gpuErrchk(cudaMemcpy(trainedSVM, PERSON_WEIGHT_VEC, sizeof(float) * svmVectorSize, cudaMemcpyHostToDevice));

  EvalSVM(dHistograms, dSvmScores, trainedSVM,
          detWindowsDimX, detWindowsDimY, DETECTION_WINDOW_STRIDE,
          DETECTION_WINDOW_WIDTH, DETECTION_WINDOW_HEIGHT,
          blocksDimX, blocksDimY, BLOCK_WIDTH, CELL_WIDTH,
          paddedWidth, paddedHeight, PERSON_LINEAR_BIAS);

  std::unique_ptr<float> svmScores(new float[totalDetectionWindows]);

  gpuErrchk(cudaMemcpy(svmScores.get(), dSvmScores, sizeof(float) * totalDetectionWindows, cudaMemcpyDeviceToHost));

  for(int i = 0; i < totalDetectionWindows; i++){
    //std::cout << svmScores.get()[i] << ", ";
    if(svmScores.get()[i] >= 1){
      //std::cout << std::endl;
      auto winCoordinates = getWindowCoordinates(i, DETECTION_WINDOW_STRIDE, DETECTION_WINDOW_STRIDE, detWindowsDimX);
      std::cout << "detection at window " << i <<
      " with coordinates (" << winCoordinates.first << ", " << winCoordinates.second << ")" << std::endl;
      cv::rectangle(cvU3ImageReal,
              cv::Rect(winCoordinates.first, winCoordinates.second, DETECTION_WINDOW_WIDTH, DETECTION_WINDOW_HEIGHT),
              cv::Scalar(0, 0, 255));
    }
  }
  std::cout << std::endl;

  cv::imshow("cvU3ImageReal", cvU3ImageReal);


  gpuErrchk(cudaFree(dGradX));
  gpuErrchk(cudaFree(dGradY));
  gpuErrchk(cudaFree(dMagnitudes));
  gpuErrchk(cudaFree(dAngles));
  gpuErrchk(cudaFree(dHistograms));
  gpuErrchk(cudaFree(dSvmScores));
  cv::waitKey(0);

}

void ImageManagerGPU::debugGradient2() {
  if(dImagePaddedF3 == nullptr){
    std::cout << "converting to float3" << std::endl;
    convertDResultToFloat3();
    std::cout << "converted to float3" << std::endl;
  }
  assert(dImagePaddedF3 != nullptr);

  std::cout << "getFloat3Image" << std::endl;
  auto debug_f3image = getFloat3Image();
  auto debug_u3image = convertToCVImage<float3, uchar, cv::Vec3b>(debug_f3image.get(), paddedWidth, paddedHeight, CV_8UC3);

}

std::unique_ptr<float3> ImageManagerGPU::getFloat3Image() {
  assert(dImagePaddedF3 != nullptr);
  std::unique_ptr<float3> hostImageF(new float3[paddedHeight*paddedWidth]);

  //gpuErrchk(cudaMemcpy(hostImage.get(), dResult, sizeof(uchar3) * rows * cols, cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(hostImageF.get(), dImagePaddedF3, sizeof(float3) * rows * cols, cudaMemcpyDeviceToHost));
  return hostImageF;
}

__host__ void ImageManagerGPU::convertDResultToFloat3() {
  assert(paddedHeight > 0 && paddedWidth > 0);
  gpuErrchk(cudaMalloc((void**)&dImagePaddedF3, sizeof(float3) * paddedHeight * paddedWidth));
  gpuErrchk(cudaMalloc((void**)&dImagePaddedF4, sizeof(float4) * paddedHeight * paddedWidth));
  assert(dImagePaddedF3 != nullptr);
  assert(dImagePaddedF4 != nullptr);
  ConvertUchar3ToFloat(dResult, dImagePaddedF3, dImagePaddedF4, paddedWidth, paddedHeight);
}

int ImageManagerGPU::getDetectionHistogramSize() {
  return DETECTION_WINDOW_HISTOGRAM_SIZE;
}

void ImageManagerGPU::ComputeBlockHistogram(float* magnitude, float* angle, float* histogram, int width, int height,
                                            int blockSize, int blocksDimX, int blocksDimY, int cellSize,
                                            int blockStride) {
  const int threadsByBlock = 32;
  double b = (blocksDimX*blocksDimY)/((double)threadsByBlock);
  int b_int = (int)std::ceil(b);

  kComputeBlockHistogram<<< b_int, threadsByBlock >>>(magnitude, angle, histogram,
          width, height, blockSize, blocksDimX, blocksDimY, cellSize, blockStride);

}

__host__
void
ImageManagerGPU::EvalSVM(float* histograms, float* svmScores, float* trainedSVM, int detWindowsDimX, int detWindowsDimY,
                         int winStride, int winSizeX, int winSizeY, int blockDimAllX, int blockDimAllY, int blockSize,
                         int blockStride, int width, int height, float svmBias) {


  const int threadsByBlock = 32;
  double b = (detWindowsDimX*detWindowsDimY)/((double)threadsByBlock);
  int b_int = (int)(std::max<double>(std::ceil(b), 1.0));

  kEvalSVM<<< b_int, threadsByBlock>>>(histograms, svmScores, trainedSVM, detWindowsDimX, detWindowsDimY,
          winStride,  winSizeX,  winSizeY,  blockDimAllX,  blockDimAllY,  blockSize,
           blockStride,  width,  height,  svmBias);
  gpuErrchk( cudaPeekAtLastError() );
}

std::vector<ResultSVMScore> ImageManagerGPU::detectWithSVM() {
  std::vector<ResultSVMScore> results;


  //int blocksInDetWindowDimX = (DETECTION_WINDOW_WIDTH - BLOCK_WIDTH)/CELL_WIDTH + 1;
  //int blocksInDetWindowDimY = (DETECTION_WINDOW_HEIGHT - BLOCK_HEIGHT)/CELL_WIDTH + 1;
  //int totalBlocksInDetWindow = blocksInDetWindowDimX * blocksInDetWindowDimY;
  //int svmVectorSize = totalBlocksInDetWindow * 36;

  if(dImagePaddedF3 == nullptr)
    convertDResultToFloat3();

  float3* dCurrentImage = dImagePaddedF3;
  int currentHeight = paddedHeight;
  int currentWidth = paddedWidth;

  for(int invScale = 1; invScale < 10; invScale++) {
    float3* dOutputImg;

      //float scale = (float) (1.0 / ((float)(1 << invScale)));
    float scale = (float) (1 << invScale);

    int widthOutput = (int) std::ceil(paddedWidth / scale);
    int heightOutput = (int) std::ceil(paddedHeight / scale);


    const int detWindowsDimX = ((widthOutput - DETECTION_WINDOW_WIDTH)/DETECTION_WINDOW_STRIDE) + 1;
    const int detWindowsDimY = ((heightOutput - DETECTION_WINDOW_HEIGHT)/DETECTION_WINDOW_STRIDE) + 1;

    const int totalDetectionWindows = detWindowsDimX * detWindowsDimY;

    if(widthOutput <= DETECTION_WINDOW_WIDTH+5 || heightOutput <= DETECTION_WINDOW_WIDTH+5)
      break;


    gpuErrchk(cudaMalloc(&dOutputImg, sizeof(float3) * widthOutput * heightOutput));

    DownscaleImg(dImagePaddedF4, dOutputImg, paddedWidth, paddedHeight,
                 widthOutput, heightOutput, scale);

    currentHeight = heightOutput;
    currentWidth = widthOutput;
    dCurrentImage = dOutputImg;


    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());



    float3* dGradX, * dGradY;

    size_t f3Bytes = sizeof(float3) * currentHeight * currentWidth;

    gpuErrchk(cudaMalloc((void**) &dGradX, f3Bytes));
    gpuErrchk(cudaMalloc((void**) &dGradY, f3Bytes));

    computeGradient(dCurrentImage, dGradX, dGradY, currentWidth, currentHeight);

    float* dMagnitudes, * dAngles;
    gpuErrchk(cudaMalloc(&dMagnitudes, sizeof(float) * currentHeight * currentWidth));
    gpuErrchk(cudaMalloc(&dAngles, sizeof(float) * currentHeight * currentWidth));

    computeMagnitudeAndAngles(dGradX, dGradY, dMagnitudes, dAngles, currentWidth, currentHeight);
    gpuErrchk(cudaFree(dGradX));
    gpuErrchk(cudaFree(dGradY));
    int blockNum = calcContainedWithStride(currentWidth, currentHeight, BLOCK_WIDTH, BLOCK_HEIGHT, CELL_WIDTH,
                                           CELL_HEIGHT);
    int allHistogramsSize = 36 * blockNum;
    float* dHistograms;

    const int blocksDimX = ((currentWidth - BLOCK_WIDTH) / CELL_WIDTH) + 1;
    const int blocksDimY = ((currentHeight - BLOCK_HEIGHT) / CELL_HEIGHT) + 1;


    gpuErrchk(cudaMalloc(&dHistograms, sizeof(float) * allHistogramsSize));

    ComputeBlockHistogram(dMagnitudes, dAngles, dHistograms, currentWidth, currentHeight,
                          BLOCK_WIDTH, blocksDimX, blocksDimY, CELL_WIDTH, CELL_WIDTH);
    gpuErrchk(cudaFree(dMagnitudes));
    gpuErrchk(cudaFree(dAngles));


    float* dSvmScores;//, * trainedSVM;

    gpuErrchk(cudaMalloc(&dSvmScores, sizeof(float) * totalDetectionWindows));
    //gpuErrchk(cudaMalloc(&trainedSVM, sizeof(float) * svmVectorSize));

    //assert(svmVectorSize == PERSON_WEIGHT_VEC_LENGTH);
    //gpuErrchk(cudaMemcpy(trainedSVM, PERSON_WEIGHT_VEC, sizeof(float) * svmVectorSize, cudaMemcpyHostToDevice));

    EvalSVM(dHistograms, dSvmScores, trainedSVM,
            detWindowsDimX, detWindowsDimY, DETECTION_WINDOW_STRIDE,
            DETECTION_WINDOW_WIDTH, DETECTION_WINDOW_HEIGHT,
            blocksDimX, blocksDimY, BLOCK_WIDTH, CELL_WIDTH,
            currentWidth, currentHeight, PERSON_LINEAR_BIAS);
    gpuErrchk(cudaFree(dHistograms));


    std::unique_ptr<float> svmScores(new float[totalDetectionWindows]);

    gpuErrchk(cudaMemcpy(svmScores.get(), dSvmScores, sizeof(float) * totalDetectionWindows, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaFree(dSvmScores));

    for (int i = 0; i < totalDetectionWindows; i++) {
      float currentScore = svmScores.get()[i];
      if (currentScore >= 1.0) {
        auto winCoordinates = getWindowCoordinates(i, DETECTION_WINDOW_STRIDE, DETECTION_WINDOW_STRIDE, detWindowsDimX);
        ResultSVMScore resultSVMScore = {
                .svmScore = currentScore,
                .x = (int)(winCoordinates.first*scale),
                .y = (int)(winCoordinates.second*scale),
                .width = (int)(DETECTION_WINDOW_WIDTH*scale),
                .height = (int)(DETECTION_WINDOW_HEIGHT*scale),
        };
        results.push_back(resultSVMScore);
      }
    }

    //std::cout << "invscale=" << invScale << std::endl;

    gpuErrchk(cudaFree(dOutputImg));

  }
  return results;
}

__host__ void ImageManagerGPU::DownscaleImg(float4* dInputImg, float3* dOutputImg, int width, int height,
                                            int widthOutput, int heightOutput, float scale){
  const int threadsByBlock = 32;
  double b = (widthOutput*heightOutput)/((double)threadsByBlock);
  int b_int = (int)std::ceil(b);

  //int widthOutput = iDivUp(width, scale);
  //int heightOutput = iDivUp(height, scale);
  
  if(isAllocated){
    gpuErrchk(cudaFreeArray(imageArray));
  }

  gpuErrchk(cudaMallocArray(&imageArray, &channelDescDownscale, width, height));
  gpuErrchk(cudaMemcpyToArray(imageArray, 0, 0, dInputImg, sizeof(float4) * width * height, cudaMemcpyDeviceToDevice));

  isAllocated = true;

  gpuErrchk(cudaBindTextureToArray(tex, imageArray, channelDescDownscale));

  kDownscaleImg<<< b_int, threadsByBlock >>>(dOutputImg, widthOutput, heightOutput, scale);

  gpuErrchk(cudaFreeArray(imageArray));
  isAllocated = false;
}
