//
// Created by cristobal, 2019
//

#ifndef PROJECT_SEQUENTIALHOG_H
#define PROJECT_SEQUENTIALHOG_H


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <memory>

const int HOG_STRIDE = 8;
const int HOG_BLOCK_SIZE = 16;
const int HALF_HOG_BLOCK_SIZE = HOG_BLOCK_SIZE/2;



const double kernel[3] = {-1, 0, 1};

const cv::Mat GRADIENT_KERNEL_X = cv::Mat(1, 3, CV_64F, (void*)kernel);
const cv::Mat GRADIENT_KERNEL_Y = cv::Mat(3, 1, CV_64F, (void*)kernel);


std::vector<double> getGaussianWeights();


using pairMA = std::pair<cv::Mat, cv::Mat>;

struct Rect{
  int x, y, width, height;
};

class SequentialHOG {
  cv::Mat* image;
  std::vector<double> gaussianWeights;
public:

  explicit SequentialHOG(cv::Mat* image);

  cv::Mat padImage(int padding);
  cv::Mat downscaleImage(cv::Mat& anImage, double times);
  std::vector<double> computeHistograms(cv::Mat& detectionWindow);

  pairMA computeGradientMagnitudeAndAngle(cv::Mat& image);
  std::vector<double> computeHistogram(pairMA& c1, pairMA& c2, pairMA& c3, pairMA& c4);

  std::vector<Rect> runHOG();

  double getWeightInterpolated(double pos);
};



#endif //PROJECT_SEQUENTIALHOG_H
