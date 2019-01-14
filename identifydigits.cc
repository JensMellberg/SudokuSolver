


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



    const double PCT_TRAIN = 0.8;
    string pathName = "computerdigits.png";
    int SZ = 28;

    float affineFlags = WARP_INVERSE_MAP|INTER_LINEAR;

    HOGDescriptor hog(
        Size(28,28), //winSize
        Size(14,14), //blocksize
        Size(7,7), //blockStride,
        Size(14,14), //cellSize,
                9, //nbins,
                1, //derivAper,
                -1, //winSigma,
                HOGDescriptor::L2Hys, //histogramNormType,
                0.2, //L2HysThresh,
                0,//gammal correction,
                64,//nlevels=64
                1);

    static string getEnvVar(string const& key)
    {
        char const* val = getenv(key.c_str());
        cout << "Looking for " << key.c_str() << endl;
        cout << "Found " << string(val) << endl;

        return val == NULL ? std::string() : std::string(val);
    }

    static Ptr<SVM> svmTrained;

    Mat deskew(Mat& img){
        Moments m = moments(img);
        if(abs(m.mu02) < 1e-2){
            return img.clone();
        }
        float skew = m.mu11/m.mu02;
        Mat warpMat = (Mat_<float>(2,3) << 1, skew, -0.5*SZ*skew, 0, 1, 0);
        Mat imgOut = Mat::zeros(img.rows, img.cols, img.type());
        warpAffine(img, imgOut, warpMat, imgOut.size(),affineFlags);

        return imgOut;
    }

    /**
    * Load image used to train SVM and extract digit image and label
    * The training image file is expected to consist of 9 rows of digit images with each digit being a square pixel range of size SZ
    */
    void loadTrainTestLabel(string &pathName, vector<Mat> &trainCells, vector<Mat> &testCells,vector<int> &trainLabels, vector<int> &testLabels){

        Mat img = imread(pathName, IMREAD_GRAYSCALE);
        int ImgCount = 0;
        for(int y = 0; y * SZ < img.rows; y += 1)
        {
            for(int x = 0; x < img.cols; x += SZ)
            {
                Mat digitImg = (img.colRange(x,x + SZ).rowRange((y * SZ), (y * SZ) + SZ)).clone();
                // source image may be jagged 2D array of images... ignore all empty areas
                if (countNonZero(digitImg) > 0) {
                    if (rand() / double(RAND_MAX) <= PCT_TRAIN)
                    {
                        trainCells.push_back(digitImg);
                        cout << "imgsize: "<< digitImg.rows << " "<< digitImg.cols << endl;
                        trainLabels.push_back(y + 1);
                    }
                    else
                    {
                        testCells.push_back(digitImg);
                        testLabels.push_back(y + 1);
                    }
                    ImgCount++;
                }
            }
        }

        cout << "Image Count : " << ImgCount << endl;
    }

    /**
    * Deskew digit images
    */
    void CreateDeskewedTrainTest(vector<Mat> &deskewedTrainCells,vector<Mat> &deskewedTestCells, vector<Mat> &trainCells, vector<Mat> &testCells){
        for(int i=0;i<trainCells.size();i++){

            Mat deskewedImg = deskew(trainCells[i]);
            deskewedTrainCells.push_back(deskewedImg);
        }

        for(int i=0;i<testCells.size();i++){

            Mat deskewedImg = deskew(testCells[i]);
            deskewedTestCells.push_back(deskewedImg);
        }
    }

    /**
    * Calculate HOGDescriptor vector for each training/test digit image
    */
    void CreateTrainTestHOG(vector<vector<float> > &trainHOG, vector<vector<float> > &testHOG, vector<Mat> &deskewedtrainCells, vector<Mat> &deskewedtestCells){

        // set the descriptor position to the middle of the image
        //std::vector<cv::Point> positions;
        //positions.push_back(cv::Point(grayImg.cols / 2, grayImg.rows / 2));
        //std::vector<float> descriptor;
        //hog.compute(grayImg,descriptor,cv::Size(),cv::Size(),positions);

        for(int y=0;y<deskewedtrainCells.size();y++){
            std::vector<float> descriptors;
            std::vector<cv::Point> positions;
            hog.compute(deskewedtrainCells[y],descriptors,cv::Size(),cv::Size(),positions);
            trainHOG.push_back(descriptors);
        }

        for(int y=0;y<deskewedtestCells.size();y++){

            vector<float> descriptors;
            hog.compute(deskewedtestCells[y],descriptors);
            testHOG.push_back(descriptors);
        }
    }

    void ConvertVectortoMatrix(vector<vector<float> > &trainHOG, vector<vector<float> > &testHOG, Mat &trainMat, Mat &testMat)
    {

        int descriptor_size = trainHOG[0].size();

        for(int i = 0;i<trainHOG.size();i++){
            for(int j = 0;j<descriptor_size;j++){
            trainMat.at<float>(i,j) = trainHOG[i][j];
            }
        }
        for(int i = 0;i<testHOG.size();i++){
            for(int j = 0;j<descriptor_size;j++){
                testMat.at<float>(i,j) = testHOG[i][j];
            }
        }
    }

    void getSVMParams(SVM *svm)
    {
        cout << "Kernel type     : " << svm->getKernelType() << endl;
        cout << "Type            : " << svm->getType() << endl;
        cout << "C               : " << svm->getC() << endl;
        cout << "Degree          : " << svm->getDegree() << endl;
        cout << "Nu              : " << svm->getNu() << endl;
        cout << "Gamma           : " << svm->getGamma() << endl;
    }

    void SVMtrain(Mat &trainMat,vector<int> &trainLabels, Mat &testResponse,Mat &testMat){

        Ptr<SVM> svm = SVM::create();
        svm->setGamma(0.50625);
        svm->setC(12.5);
        svm->setKernel(SVM::RBF);
        svm->setType(SVM::C_SVC);
        Ptr<TrainData> td = TrainData::create(trainMat, ROW_SAMPLE, trainLabels);
        svm->train(td);
        svm->save("moodel");
        svm->predict(testMat, testResponse);
        getSVMParams(svm);
    }

    void SVMevaluate(Mat &testResponse,float &count, float &accuracy,vector<int> &testLabels){

        for(int i=0;i<testResponse.rows;i++)
        {
            cout << testResponse.at<float>(i,0) << " " << testLabels[i] << endl;
            if(testResponse.at<float>(i,0) == testLabels[i]){
                count = count + 1;
            }
        }
        accuracy = (count/testResponse.rows)*100;
    }

    string TrainSVM(string pathName, int digitSize){

        /* initialize random seed: */
        srand (234);

        vector<Mat> trainCells;
        vector<Mat> testCells;
        vector<int> trainLabels;
        vector<int> testLabels;
        loadTrainTestLabel(pathName,trainCells,testCells,trainLabels,testLabels);

        vector<Mat> deskewedTrainCells;
        vector<Mat> deskewedTestCells;
        CreateDeskewedTrainTest(deskewedTrainCells,deskewedTestCells,trainCells,testCells);

        vector<std::vector<float> > trainHOG;
        vector<std::vector<float> > testHOG;
        CreateTrainTestHOG(trainHOG,testHOG,deskewedTrainCells,deskewedTestCells);

        int descriptor_size = trainHOG[0].size();
        cout << "Descriptor Size : " << descriptor_size << endl;

        Mat trainMat(trainHOG.size(),descriptor_size,CV_32FC1);
        Mat testMat(testHOG.size(),descriptor_size,CV_32FC1);

        ConvertVectortoMatrix(trainHOG,testHOG,trainMat,testMat);

        Mat testResponse;
        SVMtrain(trainMat,trainLabels,testResponse,testMat);

        float count = 0;
        float accuracy = 0 ;
        SVMevaluate(testResponse, count, accuracy, testLabels);

        stringstream stream;
        stream << fixed << setprecision(2) << accuracy;
        return stream.str();
    }

    /**
    * Use trained SVM to predict digit from Mat
    */
    int IdentifyDigit(Mat &digitMat) {

        vector<float> descriptors;
        vector<Point> positions;

        // load pre-trained SVM
        if (!svmTrained) {
            //auto model_file = getEnvVar(SVM_MODEL_ENV_VAR_NAME);
            auto model_file = "moodel";
            ifstream fs(model_file);
            if ( !fs.good()) {
                throw invalid_argument("Invalid model file: ");
            }
            svmTrained = Algorithm::load<SVM>(model_file);
            cout << "Initialized trained SVM from " << model_file << endl;
        }

        // Get HOG descriptor
        hog.compute(digitMat, descriptors, Size(), Size(), positions);
        //cout << "Computed HOGDescriptor for " << digitMat.cols << "x" << digitMat.rows << " image" << endl;
        // convert HOG descriptor to Mat
        Mat testMat = Mat::zeros(1, descriptors.size(), CV_32FC1);
        for (int x = 0; x < testMat.cols; x++) {
            testMat.at<float>(0, x) = descriptors[x];
        }

        // predict digit
        Mat testResponse;
        imwrite("testdigit.png", digitMat);
        svmTrained->predict(testMat, testResponse);
        //cout << "Predicted digit for " << digitMat.cols << "x" << digitMat.rows << " image" << endl;
        // extract prediction
        return int(testResponse.at<float>(0,0));
    }


    Mat preprocessImage(Mat img) {
      int rowTop=-1, rowBottom=-1, colLeft=-1, colRight=-1;

       Mat temp;
       int thresholdBottom = 50;
       int thresholdTop = 50;
       int thresholdLeft = 50;
       int thresholdRight = 50;
       int center = img.rows/2;
       for(int i=center;i<img.rows;i++)
       {
           if(rowBottom==-1)
           {
               temp = img.row(i);
               IplImage stub = temp;
               if(cvSum(&stub).val[0] < thresholdBottom || i==img.rows-1)
                   rowBottom = i;

           }

           if(rowTop==-1)
           {
               temp = img.row(img.rows-i);
               IplImage stub = temp;
               if(cvSum(&stub).val[0] < thresholdTop || i==img.rows-1)
                   rowTop = img.rows-i;

           }

           if(colRight==-1)
           {
               temp = img.col(i);
               IplImage stub = temp;
               if(cvSum(&stub).val[0] < thresholdRight|| i==img.cols-1)
                   colRight = i;

           }

           if(colLeft==-1)
           {
               temp = img.col(img.cols-i);
               IplImage stub = temp;
               if(cvSum(&stub).val[0] < thresholdLeft|| i==img.cols-1)
                   colLeft = img.cols-i;
    }
    }

    Mat newImg;

    newImg = newImg.zeros(img.rows, img.cols, CV_8UC1);

    int startAtX = (newImg.cols/2)-(colRight-colLeft)/2;

    int startAtY = (newImg.rows/2)-(rowBottom-rowTop)/2;

    for(int y=startAtY;y<(newImg.rows/2)+(rowBottom-rowTop)/2;y++)
    {
      uchar *ptr = newImg.ptr<uchar>(y);
      for(int x=startAtX;x<(newImg.cols/2)+(colRight-colLeft)/2;x++)
      {
          ptr[x] = img.at<uchar>(rowTop+(y-startAtY),colLeft+(x-startAtX));
      }
    }
    int numRows = 28;
    int numCols = 28;

    Mat cloneImg = Mat(numRows, numCols, CV_8UC1);

        resize(newImg, cloneImg, Size(numCols, numRows));

        // Now fill along the borders
        for(int i=0;i<cloneImg.rows;i++)
        {
            floodFill(cloneImg, cvPoint(0, i), cvScalar(0,0,0));

            floodFill(cloneImg, cvPoint(cloneImg.cols-1, i), cvScalar(0,0,0));

            floodFill(cloneImg, cvPoint(i, 0), cvScalar(0));
            floodFill(cloneImg, cvPoint(i, cloneImg.rows-1), cvScalar(0));
        }
        return cloneImg;
    }
}
