//lab4funcs.cpp

#include <iostream>
#include <opencv2/opencv.hpp>
#include <pthread.h>
#include "lab4funcs.h"

using namespace std;
using namespace cv;



void SobelFunctions::to442_grayscale(Mat* frame, double a, double b){
    
    // Mat B(frame->size().height,frame->size().width,CV_8U);
    //apply greyscale to all pixels
    for(int i = floor(a*frame->size().height); i<floor(b*frame->size().height); i++){
        for(int j = 0; j<frame->size().width; j++){
            Vec3b colors = frame->at<Vec3b>(Point(j,i));
            unsigned char grey = colors.val[0]*0.0722 + colors.val[1]*0.7152 + colors.val[2]*0.2126;
            frame->at<unsigned char>(Point(j,i)) = grey;
        }
    }
}

void SobelFunctions::to442_sobel(Mat* C, double a, double b){
    
    int Gx, Gy, sum;
    // Mat D(C->size().height,C->size().width,CV_8U,Scalar(0));
    
    //apply sobel filter to all non border pixels
    for(int j = floor(a*C->size().height)+1; j<floor(b*C->size().height)-1; j++){
        for(int i = 1; i<(C->size().width - 1); i++){
            Gx = C->at<unsigned char>(Point(i+1,j-1)) + C->at<unsigned char>(Point(i+1,j+1)) - 
                C->at<unsigned char>(Point(i-1,j-1)) - C->at<unsigned char>(Point(i-1,j+1)) +
                2*(C->at<unsigned char>(Point(i+1,j)) - C->at<unsigned char>(Point(i-1,j)));
            
            Gy = C->at<unsigned char>(Point(i-1,j-1)) + C->at<unsigned char>(Point(i+1,j-1)) - 
                C->at<unsigned char>(Point(i-1,j+1)) - C->at<unsigned char>(Point(i+1,j+1)) +
                2*(C->at<unsigned char>(Point(i,j-1)) - C->at<unsigned char>(Point(i,j+1)));
            sum = (abs(Gx) + abs(Gy));
            //cap sum to 8-bit unsigned char
            if(sum > 255){
                sum = 255;
            }    
            C->at<unsigned char>(Point(i,j)) = sum;
        }
    }
}

Mat SobelFunctions::combineFrames(Mat q1, Mat q2, Mat q3, Mat q4) {
    Mat combinedFrames(q1.size().height + q4.size().height, q1.size().width + q2.size().width, CV_8UC3, Scalar(0));

    for (int i = 0; i < combinedFrames.size().width - 1; i++) {
        for (int j = 0; j < combinedFrames.size().height - 1; j++) {

            if (i < combinedFrames.size().width / 2 && j < combinedFrames.size().height / 2) {  // if quadrant 1
                combinedFrames.at<Vec3b>(j, i) = q1.at<Vec3b>(j, i);
            }
            else if (i >= combinedFrames.size().width / 2 && j < combinedFrames.size().height / 2) {   // if quadrant 2
                combinedFrames.at<Vec3b>(j, i) = q2.at<Vec3b>(j, i - q2.size().width);
            }
            else if (i >= combinedFrames.size().width / 2 && j >= combinedFrames.size().height / 2) {    // if quadrant 3
                combinedFrames.at<Vec3b>(j, i) = q3.at<Vec3b>(j - q3.size().height, i - q3.size().width);
            }
            else if (i < combinedFrames.size().width / 2 && j >= combinedFrames.size().height / 2) {  // if quadrant 4
                combinedFrames.at<Vec3b>(j,i) = q4.at<Vec3b>(j - q4.size().height, i);
            }
        }
    }

    return combinedFrames;
}

