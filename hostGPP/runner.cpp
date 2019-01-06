//
// Created by cristobal, 2018
//

#include "runner.h"
#include "ImageHandler.h"
#include "SVM/persondetectorwt.tcc"
#include "../cudaPrograms/runnercuda.h"

#include <iostream>
#include <chrono>
#include <thread>


int main(int argc, char **argv){
  ImageHandler handler("hola.png");
  //loadSVMWeights(PERSON_WEIGHT_VEC, (size_t) PERSON_WEIGHT_VEC_LENGTH);
  auto isvmManagerGPU = loadSVMWeights(PERSON_WEIGHT_VEC, (size_t) PERSON_WEIGHT_VEC_LENGTH);
  auto imageManager = loadImageToGPU(handler.getImage());
  //std::this_thread::sleep_for(std::chrono::seconds(5));
  std::cout << isvmManagerGPU->getWeightsArraySize() << std::endl;

  //auto uchar4Mat_ = imageManager->getUchar4Image();
  auto uchar4Mat_ = imageManager->debugPadding();
  auto* uchar4Mat = uchar4Mat_.get();
  auto cols = imageManager->getCols();
  auto rows = imageManager->getRows();


  cv::Mat toshow = cv::Mat::zeros(cv::Size(cols, rows), CV_8UC4);
  for(int i = 0; i < rows; i++){
    for(int j = 0; j < cols; j++){
      //toshow[i][j][0] = uchar4Mat[i*rows + cols][2];
      //toshow[i][j][0] = uchar4Mat[i*rows + cols][2];
      cv::Vec4b color;
      int idx = i*cols + j;
      uchar4& current = uchar4Mat[idx];
      color[0] = current.z;
      color[1] = current.y;
      color[2] = current.x;
      color[3] = current.w;
      toshow.at<cv::Vec4b>(i, j) = color;
    }
  }


  cv::imshow("The window", handler.getImage());
  cv::imshow("The window2", toshow);
  cv::waitKey(0);
  //cv::imshow("The window", handler.getImage());
  //cv::waitKey(0);


  return 0;
}
