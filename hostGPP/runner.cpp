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

#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>


int main(int argc, char **argv){
  ImageHandler handler("hola.png");
  //loadSVMWeights(PERSON_WEIGHT_VEC, (size_t) PERSON_WEIGHT_VEC_LENGTH);
  auto isvmManagerGPU = loadSVMWeights(PERSON_WEIGHT_VEC, (size_t) PERSON_WEIGHT_VEC_LENGTH);

  auto imageManager = loadImageToGPU(*handler.getImage());
  //std::this_thread::sleep_for(std::chrono::seconds(5));
  std::cout << isvmManagerGPU->getWeightsArraySize() << std::endl;

  auto uchar3Mat_ = imageManager->getUchar3Image();
  auto* uchar3Mat = uchar3Mat_.get();
  auto cols = imageManager->getCols();
  auto rows = imageManager->getRows();
  std::cout << "cols=" << cols << ", rows=" << rows << std::endl;


  cv::Mat toshow = cv::Mat::zeros(cv::Size(cols, rows), CV_8UC3);
  for(int i = 0; i < rows; i++){
    for(int j = 0; j < cols; j++){
      //toshow[i][j][0] = uchar3Mat[i*rows + cols][2];
      //toshow[i][j][0] = uchar3Mat[i*rows + cols][2];
      cv::Vec3b color;
      int idx = i*cols + j;
      uchar3& current = uchar3Mat[idx];
      color[0] = current.x;
      color[1] = current.y;
      color[2] = current.z;
      toshow.at<cv::Vec3b>(i, j) = color;
    }
  }

  cv::Mat toshowWalpha;
  cv::cvtColor(toshow, toshowWalpha, CV_BGR2BGRA);


  cv::imshow("Awindow", toshow);

  //cv::imshow("The window", *handler.getImage());
  //cv::imshow("The window2", toshow);
  //cv::waitKey(0);
  cv::imshow("The window", *handler.getImage());
  cv::waitKey(0);


  return 0;
}
