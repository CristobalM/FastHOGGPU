//
// Created by cristobal, 2019
//

#include <dirent.h>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <chrono>

#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "../cudaPrograms/runnercuda.h"

#include "../SequentialHOG/SequentialHOG.h"


bool isImage(const std::string &fname){
  auto i = fname.size();
  std::string ext = "";
  for(;i >= 0; i--){
    if(fname[i] == '.'){
      if( i < fname.size()-1){
        ext = fname.substr(i+1, fname.size() - i);
        break;
      }
      else
        return false;
    }
  }

  std::vector<std::string> allowedExtensions{
    "jpg", "jpeg", "png"
  };

  return std::find(allowedExtensions.begin(), allowedExtensions.end(), ext) != allowedExtensions.end();
}

int main(int argc, char **argv) {
  if(argc < 2){
    std::cout << "syntax:" << std::endl;
    std::cout << argv[0] << " <FOLDER_WITH_IMAGES>" << std::endl;
    exit(1);
  }


  std::vector<std::string> imgFilenames;
  DIR *dir;
  struct dirent *ent;
  if ((dir = opendir (argv[1])) != NULL) {
    /* print all the files and directories within directory */
    while ((ent = readdir (dir)) != NULL) {
      //printf ("%s\n", ent->d_name);
      imgFilenames.emplace_back(ent->d_name);
    }
    closedir (dir);
  } else {
    /* could not open directory */
    perror ("");
    return EXIT_FAILURE;
  }

  std::chrono::time_point<std::chrono::high_resolution_clock> timeStart;
  uint totalTime = 0;

  int imageCount = 0;

  timeStart = std::chrono::high_resolution_clock::now();
  for (const auto& imgFilename : imgFilenames) {
    if(!isImage(imgFilename))
      continue;

    imageCount++;
    auto readImage = cv::imread(std::string(argv[1]) + "/" + imgFilename, 1);
    auto imanager = loadImageToGPU(readImage);
    imanager->detectWithSVM();

  }
  totalTime = (uint)std::chrono::duration_cast<std::chrono::milliseconds>
          (std::chrono::high_resolution_clock::now() - timeStart).count();


  std::cout << "GPU Performance--------------------------------------------------------------------" << std::endl;
  std::cout << "Image count: " << imageCount << ". Total time: " << totalTime << " ms" << std::endl;
  std::cout << "-----------------------------------------------------------------------------------" << std::endl << std::endl;

  imageCount = 0;
  timeStart = std::chrono::high_resolution_clock::now();
  for (const auto& imgFilename : imgFilenames) {
    if(!isImage(imgFilename))
      continue;

    imageCount++;
    auto readImage = cv::imread(std::string(argv[1]) + "/" + imgFilename, 1);
    SequentialHOG sequentialHOG(&readImage);
    auto detections = sequentialHOG.runHOG();
  }
  totalTime = (uint)std::chrono::duration_cast<std::chrono::milliseconds>
          (std::chrono::high_resolution_clock::now() - timeStart).count();


  std::cout << "CPU Performance (Sequential)-------------------------------------------------------" << std::endl;
  std::cout << "Image count: " << imageCount << ". Total time: " << totalTime << " ms" << std::endl;
  std::cout << "-----------------------------------------------------------------------------------" << std::endl << std::endl;



  return 0;
}