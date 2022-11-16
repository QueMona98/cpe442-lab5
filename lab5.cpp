/* Lab 5: Sobel Video Playback
Authors: Weston Keitz
         Natalie Tokhmakhian
         Quentin Monasterial
Date: 11/14/2022
Rev 2.0
*/
#include <opencv2/opencv.hpp>
#include <iostream>
#include <pthread.h>
#include <arm_neon.h>

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


void* grayscale_sobel_neon(void* threadArgs);

int main() {

    int numThreads = 5; //main(1) and pthreads(4)
    //~ VideoCapture cap("5SecVid.mp4");
    VideoCapture cap("ocean.mp4");
    Mat frame;

    if(!cap.isOpened()){
        return 0;
    }

    cap >> frame;
    
    Mat displayFrame = Mat(frame.size().height, frame.size().width, CV_8UC1, Scalar::all(0));
    
    int size_arg = frame.size().height*frame.size().width;

    threadStruct thread0 = {.inputFrame = &frame, .procFrame = &displayFrame, 
                      .rows = frame.size().width, .cols = frame.size().height, 
                      .start = 0, .stop = (size_arg/4) + 1};

    threadStruct thread1 = {.inputFrame = &frame, .procFrame = &displayFrame, 
                      .rows = frame.size().width, .cols = frame.size().height, 
                      .start = (size_arg/4) - frame.size().width, .stop = (size_arg/2) + 1};

    threadStruct thread2 = {.inputFrame = &frame, .procFrame = &displayFrame, 
                      .rows = frame.size().width, .cols = frame.size().height, 
                      .start = (size_arg/2) - frame.size().width, .stop = 3*(size_arg/4) + 1};

    threadStruct thread3 = {.inputFrame = &frame, .procFrame = &displayFrame, 
                      .rows = frame.size().width, .cols = frame.size().height, 
                      .start = (3*(size_arg/4)) - frame.size().width, .stop = size_arg - 3*frame.size().width};
                      
    pthread_barrier_init(&barrier, NULL, numThreads);

    int ret0 = pthread_create(&thread[0], NULL, grayscale_sobel_neon,(void *)&thread0);
    int ret1 = pthread_create(&thread[1], NULL, grayscale_sobel_neon,(void *)&thread1);
    int ret2 = pthread_create(&thread[2], NULL, grayscale_sobel_neon,(void *)&thread2);
    int ret3 = pthread_create(&thread[3], NULL, grayscale_sobel_neon,(void *)&thread3);

    while(1){
        pthread_barrier_wait(&barrier);
        //read frame
        cap >> frame;

        if(frame.empty())
        break;
        pthread_barrier_wait(&barrier);
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

void grayscale_neon(const uint8_t *input, uint8_t *gray, int num_pixels) {

    const int B_8L = 18;
    const int G_8L = 183;
    const int R_8L = 54;
    
    uint8x8_t NEON_B = vdup_n_u8(B_8L);
    uint8x8_t NEON_G = vdup_n_u8(G_8L);
    uint8x8_t NEON_R = vdup_n_u8(R_8L);
    
    uint16x8_t temp;
    uint16x8_t result2;
    int16x8_t x11 ,x12, x13, x21, x23, x31, x32, x33;
    int16x8_t result3;
    uint8x8_t result;
    
    for (int i = 0; i < num_pixels / 8; ++i, input += 24, gray += 8) {
        uint8x8x3_t rgb = vld3_u8(input);
        
        temp = vmull_u8(rgb.val[0],NEON_B);
        temp = vmlal_u8(temp,rgb.val[1],NEON_G);
        temp = vmlal_u8(temp,rgb.val[2],NEON_R);
        
        result = vshrn_n_u16(temp, 8);
        
        vst1_u8(gray, result);
    }

}

void* grayscale_sobel_neon(void* threadArgs){
    
    threadStruct *args = (threadStruct *)threadArgs;
    
    int size = ((args->stop - args->start)) + 1;
    
    //~ uint8_t *gray = new uint8_t[size];
    
    while(1) {
        grayscale_neon(args->inputFrame->data + args->start, args->procFrame->data + args->start, size);
        
        pthread_barrier_wait(&barrier);
    }
    
    //~ const int B_8L = 18;
    //~ const int G_8L = 183;
    //~ const int R_8L = 54;
    
    //~ uint8x8_t NEON_B = vdup_n_u8(B_8L);
    //~ uint8x8_t NEON_G = vdup_n_u8(G_8L);
    //~ uint8x8_t NEON_R = vdup_n_u8(R_8L);
    
    //~ uint16x8_t temp;
    //~ uint16x8_t result2;
    //~ int16x8_t x11 ,x12, x13, x21, x23, x31, x32, x33;
    //~ int16x8_t result3;
    //~ uint8x8_t result;

    //~ int16x8_t Gx1, Gx2, Gx3, Gx4, Gx5, Gx6;
    //~ int16x8_t Gy1, Gy2, Gy3, Gy4, Gy5, Gy6;
    //~ int16x8_t Gx, Gy;
    //~ int16x8_t sum;

    //~ threadStruct *args = (threadStruct *)threadArgs;
    //~ int size = args->cols * args->rows;
    //~ int16_t *gray = new int16_t[size];
    //~ int16_t *startGray = gray;
    //~ int start = args->start;
    //~ uchar *inputPointer = args->inputFrame->data;
    //~ uchar *procPointer = args->procFrame->data; 
    //~ while(1){

        //~ for(int i = args->start; i < args->stop && i + 8 < args->stop; i+=8 /*args->inputFrame->data+=24,*/, start+=24, gray+=8){
                //~ //load rgb values from data array, automatically assigns RGB to different registers
                //~ uint8x8x3_t rgb = vld3_u8((args->inputFrame->data+start));
                //~ //equation for 
                //~ temp = vmull_u8(rgb.val[0],NEON_B);
                //~ temp = vmlal_u8(temp,rgb.val[1],NEON_G);
                //~ temp = vmlal_u8(temp,rgb.val[2],NEON_R);
            
                //~ result = vshrn_n_u16(temp,8);
                //~ result2 = vmovl_u8(result);
                //~ result3 = vreinterpretq_s16_u16(result2);
                //~ vst1q_s16(gray,result3);
                
                //~ uint16x8_t sum2 = vreinterpretq_u16_s16(result3);
                //~ uint8x8_t sum3 = vqmovn_u16(sum2);
                //~ vst1_u8(args->procFrame->data+i,sum3);
                //~ //cout << i <<endl;
                //~ //gray = startGray;
                //~ //vst1_u8(gray, result);
        //~ }

        //~ gray = startGray;
        //~ args->procFrame->data = procPointer;
        //~ args->inputFrame->data = inputPointer;
        /*
        //sobel
        for(int j = args->start; j < args->stop; j+=8, args->procFrame->data+=8, gray+=8){
         //   int j = 0;
            //load rows and columns for convolution
            //int16x8_t vld1q_s16(uint16_t)
            x11 = vld1q_s16(gray + j - args->rows - args->cols);
            x12 = vld1q_s16(gray + j - args->rows);
            x13 = vld1q_s16(gray + j - args->rows + args->cols);
            x21 = vld1q_s16(gray + j - args->cols); 
        
            x23 = vld1q_s16(gray + j + args->cols);
            x31 = vld1q_s16(gray + j + args->rows - args->cols);
            x32 = vld1q_s16(gray + j + args->rows);
            x33 = vld1q_s16(gray + j + args->rows + args->cols);

            Gx1 = vaddq_s16(x13,x33); //+1
            Gx2 = vaddq_s16(x11,x31); //-1
            Gx3 = vmulq_n_s16(x21,2); //-2
            Gx4 = vmulq_n_s16(x23,2); //+2
            Gx5 = vsubq_s16(Gx5,Gx5); // +2 + (-2)
            Gx6 = vsubq_s16(Gx1,Gx2); // +1 + (-1)
            Gx = vaddq_s16(Gx5,Gx6);

            Gy1 = vaddq_s16(x11,x13); //+1
            Gy2 = vaddq_s16(x13,x33); //-1
            Gy3 = vmulq_n_s16(x12,2); //+2
            Gy4 = vmulq_n_s16(x32,2); //-2
            Gy5 = vsubq_s16(Gy3,Gy4); // +2 + (-2)
            Gy6 = vsubq_s16(Gy1,Gy2); // +1 + (-1)
            Gy = vaddq_s16(Gy5,Gy6);

            Gx = vabsq_s16(Gx); //take abs val Gx
            Gy = vabsq_s16(Gy); //take abs val Gy
            sum = vaddq_s16(Gx,Gy);
            
            //cap sum to 8-bit unsigned char
            
            uint16x8_t sum2 = vreinterpretq_u16_s16(sum);
            uint8x8_t sum3 = vqmovn_u16(sum2);
            vst1_u8(args->procFrame->data,sum3);
            cout << (int) args->procFrame->data << endl;
        }
        gray = startGray;
        args->procFrame->data = procPointer;
        */
        //~ pthread_barrier_wait(&barrier);
    //~ }
    //~ delete[] gray;
    //~ return 0;
}
