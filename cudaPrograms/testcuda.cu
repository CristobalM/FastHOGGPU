#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "testcuda.h"

__global__ void checkIndex(void){
  int pos = blockDim.x * blockIdx.x + threadIdx.x;
  printf("pos = %d  -  threadIdx: (%d, %d, %d) blockIdx: (%d, %d, %d) blockDim:(%d,%d,%d) gridDim(%d, %d, %d)\n", pos,
         threadIdx.x, threadIdx.y, threadIdx.z,
         blockIdx.x, blockIdx.y, blockIdx.z,
         blockDim.x, blockDim.y, blockDim.z);

/*
  double a = 0.0;



  int z = threadIdx.z+blockDim.z*blockIdx.z;

  int xy = gridDim.x*blockDim.x*gridDim.y*blockDim.y;

  int y = threadIdx.y+blockDim.y*blockIdx.y;

  int xx = gridDim.x*blockDim.x;

  int x_X = threadIdx.x+blockDim.x*blockIdx.x;


  int id = z*xy + y*xx + x_X;
  int id2 = threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;

  for(int i = 0; i < 10000000; i++){
    a += i*0.000053123;

  }


  printf("a = %f... id=%d, id2=%d \n", a, id, id2);
*/
}

__host__ void cudaProgram(void){
  int nElem = 10;

  cudaSetDevice(0);

  dim3 block(3);
  dim3 grid(3);

  printf("grid.x %d grid.y %d grid.z %d\n", grid.x, grid.y, grid.z);
  printf("block.x %d block.y %d block.z %d\n", block.x, block.y, block.z);

  checkIndex<<<grid, block>>>();

  printf("SU?");
  cudaDeviceSynchronize();
  printf("SU?2");
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    printf("Error: %s\n", cudaGetErrorString(err));
  cudaDeviceReset();
}
