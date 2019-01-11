cmake_minimum_required(VERSION 3.10)

project(ProjectoFastHOGGPUHostGCC)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED )
add_executable(ProjectoFastHOGGPU runner.cpp runner.h  ImageHandler.cpp ImageHandler.h SVM/persondetectorwt.tcc)
add_executable(gpu_svm_detection gpu_svm_detection.cpp ImageHandler.cpp ImageHandler.h SVM/persondetectorwt.tcc)
target_link_libraries(ProjectoFastHOGGPU HOGCudaLib ${OpenCV_LIBS} )
target_link_libraries(gpu_svm_detection HOGCudaLib ${OpenCV_LIBS} )

add_executable(prueba ../main.cpp)
target_link_libraries(prueba pruebacuda)