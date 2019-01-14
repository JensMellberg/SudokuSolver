
#ifndef  IDENTIFY_DIGITS_INC
#define  IDENTIFY_DIGITS_INC

#include <string>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <iomanip> // setprecision
#include <sstream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/ml.hpp>


#include <stdio.h>
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/ml/ml.hpp>

using namespace cv;
using namespace cv::ml;
using namespace std;

namespace Identify {
    std::string TrainSVM(std::string pathName, int digitSize);
    int IdentifyDigit(cv::Mat &digitMat);
    Mat preprocessImage(Mat image);
}

#endif
