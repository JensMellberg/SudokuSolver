#!/bin/sh

g++ -I/usr/local/include/opencv -I/usr/local/include/opencv4 -L/usr/local/lib/ -g -o SudokuSolver *.cc *.h -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_ml -lopencv_video -lopencv_features2d -lopencv_calib3d -lopencv_objdetect -lopencv_stitching -lopencv_imgcodecs -std=c++11
