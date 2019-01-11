//
// Created by cristobal, 2018
//

#include <opencv2/imgproc.hpp>
#include "ImageHandler.h"

void ImageHandler::readImageFromFile(const std::string& filename) {
  auto readImage = cv::imread(filename, 1);
  image = std::make_unique<cv::Mat>(std::move(readImage));
;
}

cv::Mat* ImageHandler::getImage() {
  return image.get();
}

ImageHandler::ImageHandler(const std::string& filename) {
  readImageFromFile(filename);
}

void ImageHandler::padImage(int padding) {
  cv::Mat result;

  result.create(image->rows + 2 * padding, image->cols + 2 * padding, image->type());
  result.setTo(cv::Scalar::all(0));

  image->copyTo(result(cv::Rect(padding, padding, image->cols, image->rows)));
  //std::swap(image, &result);
  image = std::make_unique<cv::Mat>(std::move(result));
}
