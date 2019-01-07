//
// Created by cristobal, 2019
//

#ifndef PROJECT_SEQUENTIALHOG_H
#define PROJECT_SEQUENTIALHOG_H


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>


const int HOG_STRIDE = 8;
const int HOG_BLOCK_SIZE = 16;
const int HALF_HOG_BLOCK_SIZE = HOG_BLOCK_SIZE/2;




using pairMA = std::pair<cv::Mat, cv::Mat>;
class SequentialHOG {
  cv::Mat* image;
public:
  const cv::Mat GRADIENT_KERNEL_X = cv::Mat(1, 2, CV_64F, {1, -1});
  const cv::Mat GRADIENT_KERNEL_Y = cv::Mat(2, 1, CV_64F, {1, -1});
  //const cv::Mat GRADIENT_KERNEL_Y = (cv::Mat_<double>(2, 1) << 1, -1);

  explicit SequentialHOG(cv::Mat* image);

  cv::Mat padImage(int padding);
  cv::Mat downscaleImage(double times);
  std::vector<double> computeHistograms(cv::Mat& detectionWindow);

  pairMA computeGradientMagnitudeAndAngle(cv::Mat& image);
  std::vector<double> computeHistogram(pairMA& c1, pairMA& c2, pairMA& c3, pairMA& c4);


};



#endif //PROJECT_SEQUENTIALHOG_H
