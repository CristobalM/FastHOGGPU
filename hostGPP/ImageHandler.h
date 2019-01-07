//
// Created by cristobal, 2018
//

#ifndef PROJECT_IMAGEHANDLER_H
#define PROJECT_IMAGEHANDLER_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <memory>


class ImageHandler {
  std::unique_ptr<cv::Mat> image;
public:
  ImageHandler() = default;
  explicit ImageHandler(const std::string& filename);

  void readImageFromFile(const std::string& filename);
  cv::Mat* getImage();

};


#endif //PROJECT_IMAGEHANDLER_H
