cmake_minimum_required(VERSION 3.10)

project(SequentialHOG)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED )
set(HOST_SOURCES ../hostGPP)
set(SEQ_HOG_SOURCES SequentialHOG.cpp SequentialHOG.h
        ${HOST_SOURCES}/ImageHandler.h ${HOST_SOURCES}/ImageHandler.cpp
        ../SVM/persondetectorwt.tcc)
add_executable(SequentialHOGDebug sequentialDebug.cpp ${SEQ_HOG_SOURCES})

target_link_libraries(SequentialHOGDebug ${OpenCV_LIBS})