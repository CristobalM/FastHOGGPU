cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 11)
project(ProjectoFastHOGGPUCuda)
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-10.0/)
find_package(CUDA QUIET REQUIRED)
find_package(OpenCV REQUIRED )

cuda_add_library(HOGCudaLib SHARED
        SVMManagerGPU.cu SVMManagerGPU.cuh ISVMManagerGPU.h
        runnercuda.cu runnercuda.h
        ImageManagerGPU.cu ImageManagerGPU.cuh IImageManagerGPU.h
        defs.h)
target_link_libraries(HOGCudaLib ${CUDA_LIBRARIES} ${OpenCV_LIBS})

cuda_add_library(pruebacuda SHARED testcuda.cu testcuda.h)
target_link_libraries(pruebacuda ${CUDA_LIBRARIES})

add_executable(DebugGPU debugGPU.cpp)
target_link_libraries(DebugGPU HOGCudaLib ${CUDA_LIBRARIES} ${OpenCV_LIBS})