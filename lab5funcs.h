#ifndef LAB3FUNCS_H
#define LAB3FUNCS_H

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

class SobelFunctions {
    

    public:
    
    void to442_grayscale(Mat* frame, double a, double b);
    void to442_sobel(Mat *C, double a, double b);

    Mat combineFrames(Mat q1, Mat q2, Mat q3, Mat q4);

};
#endif 