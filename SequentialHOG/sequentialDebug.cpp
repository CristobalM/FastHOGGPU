//
// Created by cristobal, 2019
//


#include <iostream>
#include <utility>
#include <cmath>
#include <algorithm>

#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "../hostGPP/ImageHandler.h"

#include "SequentialHOG.h"


int main(int argc, char **argv){
  std::cout << "argc=" << argc << std::endl;
  if(argc < 2){
    std::cout << "Falta archivo de imagen!" << std::endl;
    exit(1);
  }
  std::string filename(argv[1]);
  std::cout << "Filename: " << filename << std::endl;
  ImageHandler imageHandler(filename);
  auto *image = imageHandler.getImage();
  SequentialHOG sequentialHOG(image);
  auto detections = sequentialHOG.runHOG();
  std::cout << "detections: " << detections.size() << std::endl;

  for(auto& detection : detections){
    cv::rectangle(*image, cv::Rect(detection.x, detection.y, detection.width, detection.height), cv::Scalar(0, 0, 255));
  }

  cv::imshow("detections", *image);
  cv::waitKey(0);
}