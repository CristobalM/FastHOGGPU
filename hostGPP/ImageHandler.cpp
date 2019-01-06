//
// Created by cristobal, 2018
//

#include <opencv2/imgproc.hpp>
#include "ImageHandler.h"

void ImageHandler::readImageFromFile(const std::string& filename) {
  auto readImage = cv::imread(filename, 1);
  cv::Mat bgra;
  cv::cvtColor(readImage, bgra, CV_BGR2BGRA);
  image = std::make_unique<cv::Mat>(std::move(bgra));
}

cv::Mat& ImageHandler::getImage() {
  return *image;
}

ImageHandler::ImageHandler(const std::string& filename) {
  readImageFromFile(filename);
}
