/* Lab 4: Sobel Video Playback
Authors: Weston Keitz
         Natalie Tokhmakhian
         Quentin Monasterial
Date: 10/31/2022
Rev 1.0
*/
#include <opencv2/opencv.hpp>
#include <iostream>
#include <pthread.h>

using namespace cv;
using namespace std;

pthread_t thread[4];

pthread_barrier_t barrier;

void *thread0Status;

typedef struct threadArgs{
    Mat* inputFrame;
    Mat* procFrame;
    int rows;
    int cols;
    
    int start;
    int stop; 
} threadStruct;


void* grayscale_sobel(void* threadArgs);

void setupFunction(int numThreads){
  pthread_barrier_init(&barrier, NULL, numThreads);
}
int main() {
    int numThreads = 5; //main and thread
   

    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

  
    VideoCapture cap("ocean.mp4");
    
    Mat frame;

    if(!cap.isOpened()){
        return 0;
    }

 
    cap >> frame;
 
    Mat displayFrame = Mat(frame.size().height,frame.size().width,CV_8UC1,Scalar::all(0));

    threadStruct thread0 = {.inputFrame = &frame, .procFrame = &displayFrame, 
                      .rows = frame.size().width, .cols = frame.size().height, 
                      .start = 0, .stop = (thread0.rows*thread0.cols/4) + thread0.rows};

    threadStruct thread1 = {.inputFrame = &frame, .procFrame = &displayFrame, 
                      .rows = frame.size().width, .cols = frame.size().height, 
                      .start = (thread1.rows * thread1.cols/4) - thread1.rows, .stop = (thread1.rows*thread1.cols/2) + thread1.rows};

    threadStruct thread2 = {.inputFrame = &frame, .procFrame = &displayFrame, 
                      .rows = frame.size().width, .cols = frame.size().height, 
                      .start = (thread2.rows*thread2.cols/2) - thread2.rows, .stop = 3*(thread2.rows*thread2.cols/4) + thread2.rows};

    threadStruct thread3 = {.inputFrame = &frame, .procFrame = &displayFrame, 
                      .rows = frame.size().width, .cols = frame.size().height, 
                      .start = (3*(thread3.cols*thread3.rows/4)) - thread3.rows, .stop = thread3.cols*thread3.rows};
                      

    setupFunction(numThreads);

    int ret0 = pthread_create(&thread[0], NULL, grayscale_sobel,(void *)&thread0);
    int ret1 = pthread_create(&thread[1], NULL, grayscale_sobel,(void *)&thread1);
    int ret2 = pthread_create(&thread[2], NULL, grayscale_sobel,(void *)&thread2);
    int ret3 = pthread_create(&thread[3], NULL, grayscale_sobel,(void *)&thread3);

    while(1){
    pthread_barrier_wait(&barrier);
      //read frame
      cap >> frame;
      
      if(frame.empty())
        break;
      

      imshow("Display image", displayFrame);
      
      if(waitKey(1)==27)
        break; //wait for ESC keystroke in window
    pthread_barrier_wait(&barrier);
    } 

    //close stream
    cap.release();
    destroyAllWindows();

    int ret4 = pthread_join(ret0, NULL);
    int ret5 = pthread_join(ret1, NULL);
    int ret6 = pthread_join(ret2, NULL);
    int ret7 = pthread_join(ret3, NULL);
    
    return 0;
}

void* grayscale_sobel(void* threadArgs){
   
    int Gx, Gy, sum;

    threadStruct *args = (threadStruct *)threadArgs;
    Mat frame_int(args->cols,args->rows,CV_8UC1,Scalar::all(0));
    int size = args->rows * args->cols;
    int gray[size];

    while(1){

        for(int i = args->start; i < args->stop; i++){
            gray[i] = 0.0722*(args->inputFrame->data[3*i]) + 0.7152*(args->inputFrame->data[3*i+1]) + 0.2126*(args->inputFrame->data[3*i+2]);
        }

        for(int j = args->start + args->rows; j < args->stop - args->rows; j++){
            Gx = gray[j - 1 - args->rows] + gray[j - 1 + args->rows] 
                - gray[j + 1 - args->rows] - gray[j + 1 + args->rows] 
                + 2*(gray[j - 1] - gray[j + 1]);

            Gy = gray[j - 1 - args->rows] + gray[j + 1 - args->rows] 
                - gray[j - 1 + args->rows] - gray[j + 1 + args->rows] 
                + 2*(gray[j - args->rows] - gray[j + args->rows]);

            sum = (abs(Gx) + abs(Gy));
            
            //cap sum to 8-bit unsigned char
            if(sum > 255){
                sum = 255;
            }   

            args->procFrame->data[j] = (unsigned char) sum;
        }
        
        pthread_barrier_wait(&barrier);
    }
   
    return 0;
}