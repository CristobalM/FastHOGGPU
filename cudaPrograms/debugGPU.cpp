//
// Created by cristobal, 2019
//

#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "runnercuda.h"
#include "IImageManagerGPU.h"

#include <iostream>

void debugGradient(){
  const std::string filename = "pplwalking.jpeg";
  //const std::string filename = "hola.png";
  auto readImage = cv::imread(filename, 1);
  cv::Mat bgra;
  //cv::cvtColor(readImage, bgra, CV_BGR2BGRA);

  auto imanager = loadImageToGPU(readImage);
  std::cout << "DEBUG HISTOGRAM SIZE (DETECTION WINDOW) = "<< imanager->getDetectionHistogramSize() << std::endl;
  imanager->debugGradient();
}

int main(int argc, char ** argv){


  debugGradient();

  return 0;
}