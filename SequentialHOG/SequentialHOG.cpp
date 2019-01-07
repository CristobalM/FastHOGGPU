//
// Created by cristobal, 2019
//

#include <iostream>
#include <utility>
#include <cmath>
#include <algorithm>

#include <opencv2/imgproc.hpp>

#include "SequentialHOG.h"

#include "../hostGPP/ImageHandler.h"

#include "../SVM/persondetectorwt.tcc"

SequentialHOG::SequentialHOG(cv::Mat* image) : image(image) {}

cv::Mat SequentialHOG::padImage(int padding) {
  cv::Mat result;

  result.create(image->rows + 2 * padding, image->cols + 2 * padding, image->type());
  result.setTo(cv::Scalar::all(0));

  image->copyTo(result(cv::Rect(padding, padding, image->cols, image->rows)));
  return result;
}

cv::Mat SequentialHOG::downscaleImage(double times) {
  cv::Mat result;
  assert(times > 0.0);
  auto factor = ( 1.0 / times);
  cv::resize(*image, result, cv::Size(), factor, factor);
  return result;
}

std::vector<double> SequentialHOG::computeHistograms(cv::Mat& detectionWindow) {
  //cv::Mat result(3780,1, CV_64F);
  std::vector<double> result(3780, 0);
  int count = 0;
  const int blocks_in_horizontal = (detectionWindow.cols - HOG_BLOCK_SIZE)/HOG_STRIDE + 1;
  for(int i = 0; i + HOG_BLOCK_SIZE <= detectionWindow.rows; i += HOG_STRIDE){
    for(int j = 0; j + HOG_BLOCK_SIZE <= detectionWindow.cols; j+= HOG_STRIDE){
      count++;
      cv::Rect rect1(j, i, HALF_HOG_BLOCK_SIZE, HALF_HOG_BLOCK_SIZE);
      cv::Rect rect2(j + HALF_HOG_BLOCK_SIZE, i, HALF_HOG_BLOCK_SIZE, HALF_HOG_BLOCK_SIZE);
      cv::Rect rect3(j, i + HALF_HOG_BLOCK_SIZE, HALF_HOG_BLOCK_SIZE, HALF_HOG_BLOCK_SIZE);
      cv::Rect rect4(j + HALF_HOG_BLOCK_SIZE, i + HALF_HOG_BLOCK_SIZE, HALF_HOG_BLOCK_SIZE, HALF_HOG_BLOCK_SIZE);

      auto cell1 = detectionWindow(rect1);
      auto cell2 = detectionWindow(rect2);
      auto cell3 = detectionWindow(rect3);
      auto cell4 = detectionWindow(rect4);


      auto mAndA1 = computeGradientMagnitudeAndAngle(cell1);
      auto mAndA2 = computeGradientMagnitudeAndAngle(cell2);
      auto mAndA3 = computeGradientMagnitudeAndAngle(cell3);
      auto mAndA4 = computeGradientMagnitudeAndAngle(cell4);

      auto histogram = computeHistogram(mAndA1, mAndA2, mAndA3, mAndA4);


      //cv::Mat normalized_h;
      //cv::normalize(histogram, normalized_h);

      int idx_i = (i/HOG_STRIDE);
      int idx_j = (j/HOG_STRIDE);
      //int debugval = detectionWindow.cols/HOG_BLOCK_SIZE;
      int idx = idx_i * (blocks_in_horizontal) + idx_j;
      if(idx == 100){

        std::cout << std::endl;
        std::cout << std::endl;
        for(int hh = 0; hh < 36; hh++){
          std::cout << histogram[hh] << ", ";
        }
        std::cout << std::endl;
        std::cout << std::endl;
      }
      std::cout << idx <<  ", ";

      //histogram.copyTo(result(cv::Rect(0, idx*36, 1, 36)));
      std::copy(histogram.begin(), histogram.end(), result.begin() + idx*36);

    }
  }
  std::cout << std::endl;
  std::cout << "histograms: " << count << std::endl;
  return result;
}

void iterateCVMat(void (*f)(cv::Mat &, int, int), cv::Mat &mat){
  for(int i = 0; i < mat.rows; i++){
    for(int j = 0; j < mat.cols; j++){
      f(mat, i, j);
    }
  }
}

void fixAngle(cv::Mat& mat, int i, int j){
  auto current_angle  = mat.at<double>(i, j);
  if(current_angle > CV_PI){
    mat.at<cv::Vec4d>(i, j).val[0] -= CV_PI;
    mat.at<cv::Vec4d>(i, j).val[1] -= CV_PI;
    mat.at<cv::Vec4d>(i, j).val[2] -= CV_PI;
    //mat.at<double>(i, j, 0) -= CV_PI;
    //mat.at<double>(i, j, 1) -= CV_PI;
    //mat.at<double>(i, j, 2) -= CV_PI;
  }
}

pairMA SequentialHOG::computeGradientMagnitudeAndAngle(cv::Mat& image) {
  cv::Mat gradientX, gradientY;

  int channels_g_b = gradientX.channels();
  cv::filter2D(image, gradientX, -1, GRADIENT_KERNEL_X);
  cv::filter2D(image, gradientY, -1, GRADIENT_KERNEL_Y);

  int channels_g = gradientX.channels();

  cv::Mat magnitude, angle;

  cv::cartToPolar(gradientX, gradientY, magnitude, angle);
  iterateCVMat(fixAngle, angle);

  auto output = std::make_pair(magnitude, angle);
  return output;
}

int getArgMaxInChannels(cv::Mat& mat, int i, int j){
  double max_val = mat.at<double>(i, j, 0);
  int max_idx = 0;
  for(int k = 0; k < 3;  k++){
    auto current = mat.at<cv::Vec4d>(i, j).val[k];
    if(current > max_val){
      max_val = current;
      max_idx = k;
    }
  }
  return max_idx;
}


std::vector<double> SequentialHOG::computeHistogram(pairMA& c1, pairMA& c2, pairMA& c3, pairMA& c4) {
  assert(
          c1.first.rows == c1.second.rows && c1.second.rows == c2.first.rows &&
          c2.first.rows == c2.second.rows && c2.second.rows == c3.first.rows &&
          c3.first.rows == c3.second.rows && c2.second.rows == c3.first.rows &&
          c4.first.rows == c4.second.rows && c4.second.rows == HALF_HOG_BLOCK_SIZE &&
          c1.first.cols == c1.second.cols && c1.second.cols == c2.first.cols &&
          c2.first.cols == c2.second.cols && c2.second.cols == c3.first.cols &&
          c3.first.cols == c3.second.cols && c3.second.cols == c4.first.cols &&
          c4.first.cols == c4.second.cols && c4.second.cols == HALF_HOG_BLOCK_SIZE
  );
  const int a = HALF_HOG_BLOCK_SIZE/2;
  const int b = HALF_HOG_BLOCK_SIZE + a;
  const auto delta_bins = CV_PI / 8.0;

  //auto result = cv::Mat(9*4, 1, CV_64F);
  std::vector<double> result(36, 0);


  for(int i = 0; i < HOG_BLOCK_SIZE; i++){
    for(int j = 0; j < HOG_BLOCK_SIZE; j++){
      double alpha1 = std::abs(b - j);
      double beta1 = std::abs(b - i);
      int jnow, inow;
      pairMA * selC;
      if(i < HALF_HOG_BLOCK_SIZE && j < HALF_HOG_BLOCK_SIZE){
        inow = i; jnow = j; selC = &c1;
      }
      else if(i < HALF_HOG_BLOCK_SIZE && j >= HALF_HOG_BLOCK_SIZE){
        inow = i; jnow = j - HALF_HOG_BLOCK_SIZE; selC = &c2;
      }
      else if(i >= HALF_HOG_BLOCK_SIZE && j < HALF_HOG_BLOCK_SIZE){
        inow = i - HALF_HOG_BLOCK_SIZE; jnow = j; selC = &c3;
      }
      else if(i >= HALF_HOG_BLOCK_SIZE && j >= HALF_HOG_BLOCK_SIZE){
        inow = i - HALF_HOG_BLOCK_SIZE; jnow = j - HALF_HOG_BLOCK_SIZE; selC = &c4;
      }
      else{
        std::cout << "UNEXPECTED CASE " << std::endl; exit(1);
      }

      auto max_mag_idx_ch = getArgMaxInChannels(selC->first, inow, jnow);
      double magnitude = selC->first.at<cv::Vec4d>(inow, jnow).val[max_mag_idx_ch];
      double angle = selC->second.at<cv::Vec4d>(inow, jnow).val[max_mag_idx_ch];

      double contr_c1 = alpha1*beta1*magnitude;
      double contr_c2 = (HALF_HOG_BLOCK_SIZE - alpha1)*beta1*magnitude;
      double contr_c3 = alpha1*(HALF_HOG_BLOCK_SIZE - beta1)*magnitude;
      double contr_c4 = (HALF_HOG_BLOCK_SIZE - alpha1)*(HALF_HOG_BLOCK_SIZE - beta1)*magnitude;


      double bin = angle / delta_bins;
      double lower_bin_d = std::floor(bin);

      int lower_bin = (int)lower_bin_d;
      int upper_bin = (int)std::ceil(bin);

      double dist_low = bin - lower_bin_d;

      auto contr_c1_lb =  (contr_c1 * (1.0 - dist_low));
      auto contr_c1_ub =  (contr_c1 * dist_low);
      auto contr_c2_lb =  (contr_c2 * (1.0 - dist_low));
      auto contr_c2_ub =  (contr_c2 * dist_low);
      auto contr_c3_lb =  (contr_c3 * (1.0 - dist_low));
      auto contr_c3_ub =  (contr_c3 * dist_low);
      auto contr_c4_lb =  (contr_c4 * (1.0 - dist_low));
      auto contr_c4_ub =  (contr_c4 * dist_low);


      result[lower_bin] = contr_c1_lb;
      result[upper_bin] = contr_c1_ub;
      result[lower_bin + 9] = contr_c2_lb;
      result[upper_bin + 9] = contr_c2_ub;
      result[lower_bin + 18]= contr_c3_lb;
      result[upper_bin + 18]= contr_c3_ub;
      result[lower_bin + 27]= contr_c4_lb;
      result[upper_bin + 27]= contr_c4_ub;
    }
  }

  for(int i = 0; i < 4; i++){
    double sum = 0;
    for(int j = 0; j < 9; j++){
      sum += result[j + i*9];
    }
    if(sum != 0){
      for(int j = 0; j < 9; j++){
        result[j + i*9] /= sum;
      }
    }
  }
  return result;
}

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

  auto paddedImage = sequentialHOG.padImage(0);
  cv::Mat paddedImage64F;
  paddedImage.convertTo(paddedImage64F, CV_64FC4);
  cv::Rect firstRect(0, 0, 64, 128);
  auto detectionWindow0 = paddedImage64F(firstRect);
  int channels_dw = detectionWindow0.channels();
  int channels_pi = paddedImage64F.channels();

  auto histograms = sequentialHOG.computeHistograms(detectionWindow0);
  double dotp = 0;
  int nans = 0;
  for(int i = 0; i < histograms.size(); i++){
    //std::cout << histograms.at<double>(i, 0) << ", ";
    //dist += (double)std::pow(histogram_result[i] - PERSON_WEIGHT_VEC[i],2);

    if(histograms[i] != histograms[i]){
      nans++;
      continue;
    }


    dotp += histograms[i]*PERSON_WEIGHT_VEC[i];

  }
  std::cout << "Nans count = " << nans << std::endl;

  double val = dotp - PERSON_LINEAR_BIAS;

  std::cout << "val is = " << val << std::endl;

  std::cout << "Is pedestrian? " << std::endl;
  if(val >= 1)
    std::cout << "YES" << std::endl;
  else if(val <= -1)
    std::cout << "NO" << std::endl;
  else
    std::cout << "NOT SURE" << std::endl;

  auto maTest = sequentialHOG.computeGradientMagnitudeAndAngle(paddedImage64F);
  cv::Mat gradX;
  cv::filter2D(*image, gradX, -1, sequentialHOG.GRADIENT_KERNEL_X);
  //gradX.convertTo(gradX, CV_8UC3);
  auto &magnitudes = maTest.first;
  magnitudes.convertTo(magnitudes, CV_8UC3);

  cv::imshow("imagen", gradX);
  cv::waitKey(0);

  /*
  cv::imshow("imagen", histograms);
  cv::imwrite("imagenlinea.png", histograms);
  for(int i = 0; i < histograms.rows; i++){
    std::cout << histograms.at<double>(i, 0) << ", ";
  }
  std::cout << std::endl;
  cv::waitKey(0);
   */
}
