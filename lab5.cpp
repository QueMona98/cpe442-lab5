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
    
    int row; 
    int col;
    
    Mat* grayscaleFrame;

    int inputStart;
    int start;
    int sobelStart;
    int stop;
    int sobelStop;
     
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
    
    Mat grayFrame = Mat(frame.size().height, frame.size().width, CV_8UC1, Scalar::all(0));
    
    int size_arg = (frame.size().height*frame.size().width)/4;

    threadStruct thread0 = {.inputFrame = &frame, 
                          .procFrame = &displayFrame, 
                          .row = frame.size().width, 
                          .col = frame.size().height,
                          .grayscaleFrame = &grayFrame, 
                          .inputStart = 0, 
                          .start = 0, 
                          .sobelStart = frame.size().width,
                          .stop = (size_arg), 
                          .sobelStop = 8*size_arg};
                      
    threadStruct thread1 = {.inputFrame = &frame, 
                          .procFrame = &displayFrame, 
                          .row = frame.size().width, 
                          .col = frame.size().height,
                          .grayscaleFrame = &grayFrame, 
                          .inputStart = 3*size_arg,
                          .start = (size_arg), 
                          .sobelStart = (size_arg), 
                          .stop = 2*(size_arg), 
                          .sobelStop = 16*size_arg};

    threadStruct thread2 = {.inputFrame = &frame, 
                          .procFrame = &displayFrame, 
                          .row = frame.size().width, 
                          .col = frame.size().height,
                          .grayscaleFrame = &grayFrame, 
                          .inputStart = 6*size_arg, 
                          .start = 2*size_arg, 
                          .sobelStart = 2*(size_arg), 
                          .stop = 3*size_arg, 
                          .sobelStop = 24*size_arg};

    threadStruct thread3 = {.inputFrame = &frame, 
                            .procFrame = &displayFrame, 
                            .row = frame.size().width, 
                            .col = frame.size().height,
                            .grayscaleFrame = &grayFrame, 
                            .inputStart = 9*(size_arg), 
                            .start = 3*size_arg, 
                            .sobelStart = 3*size_arg, 
                            .stop = 4*size_arg, 
                            .sobelStop = 32*size_arg - frame.size().width};
                      
    pthread_barrier_init(&barrier, NULL, numThreads);

    int ret0 = pthread_create(&thread[0], NULL, grayscale_sobel_neon,(void *)&thread0);
    int ret1 = pthread_create(&thread[1], NULL, grayscale_sobel_neon,(void *)&thread1);
    int ret2 = pthread_create(&thread[2], NULL, grayscale_sobel_neon,(void *)&thread2);
    int ret3 = pthread_create(&thread[3], NULL, grayscale_sobel_neon,(void *)&thread3);

    while(1){
        pthread_barrier_wait(&barrier);
        //read frame
        cap >> frame;

        if(frame.empty()){
            //~ break;
                cout << "error" << endl;
                return 0;}
        //~ pthread_barrier_wait(&barrier);
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
    uint8x8_t result;
        
    pthread_barrier_wait(&barrier);

    for (int i = 0; i < num_pixels / 8; ++i, input += 24, gray += 8) {
        uint8x8x3_t rgb = vld3_u8(input);
        
        temp = vmull_u8(rgb.val[0],NEON_B);
        temp = vmlal_u8(temp,rgb.val[1],NEON_G);
        temp = vmlal_u8(temp,rgb.val[2],NEON_R);
        
        result = vshrn_n_u16(temp, 8);
        
        vst1_u8(gray, result);
    }
}

void sobel_neon(const uint8_t *input, uint8_t *sobel, int num_pixels, int rows, int cols){

    int16x8_t two = vdupq_n_s16(2);
    
    pthread_barrier_wait(&barrier);
    //sobel
    for(int j = 0; j < num_pixels/8; j+=8, input+=8, sobel+=8){
        const uint8_t* x11 = input - rows - 1; 
        const uint8_t* x12 = input - rows;
        const uint8_t* x13 = input - rows + 1;
        
        const uint8_t* x21 = input - 1;
        const uint8_t* x23 = input + 1;
        
        const uint8_t* x31 = input + rows - 1;
        const uint8_t* x32 = input + rows;
        const uint8_t* x33 = input + rows + 1;
        
        
        uint8x8_t loadx11 = vld1_u8(x11);
        uint8x8_t loadx12 = vld1_u8(x12);
        uint8x8_t loadx13 = vld1_u8(x13);
        uint8x8_t loadx21 = vld1_u8(x21);
        uint8x8_t loadx23 = vld1_u8(x23);
        uint8x8_t loadx31 = vld1_u8(x31);
        uint8x8_t loadx32 = vld1_u8(x32);
        uint8x8_t loadx33 = vld1_u8(x33);
        
        uint16x8_t x11_16bit = vmovl_u8(loadx11);
        uint16x8_t x12_16bit = vmovl_u8(loadx12);
        uint16x8_t x13_16bit = vmovl_u8(loadx13);
        uint16x8_t x21_16bit = vmovl_u8(loadx21);
        uint16x8_t x23_16bit = vmovl_u8(loadx23); 
        uint16x8_t x31_16bit = vmovl_u8(loadx31);
        uint16x8_t x32_16bit = vmovl_u8(loadx32);
        uint16x8_t x33_16bit = vmovl_u8(loadx33);
        
        uint16x8_t Gx1 = vaddq_u16(x13_16bit,x23_16bit); //+1
        uint16x8_t Gx2 = vaddq_u16(x11_16bit,x31_16bit); //-1
        uint16x8_t Gx3 = vshlq_u16(x21_16bit,two); //-2
        uint16x8_t Gx4 = vshlq_u16(x23_16bit,two); //+2
        uint16x8_t Gx5 = vsubq_u16(Gx4,Gx3); // +2 + (-2)
        uint16x8_t Gx6 = vsubq_u16(Gx1,Gx2); // +1 + (-1)
        uint16x8_t Gx = vaddq_u16(Gx5,Gx6);
        
        int16x8_t temp_x = vreinterpretq_s16_u16(Gx);
        temp_x = vabsq_s16(temp_x);
        Gx = vreinterpretq_u16_s16(temp_x);

        uint16x8_t Gy1 = vaddq_u16(x11_16bit,x13_16bit); //+1
        uint16x8_t Gy2 = vaddq_u16(x31_16bit,x33_16bit); //-1
        uint16x8_t Gy3 = vshlq_u16(x12_16bit,two); //+2
        uint16x8_t Gy4 = vshlq_u16(x32_16bit,two); //-2
        uint16x8_t Gy5 = vsubq_u16(Gy3,Gy4); // +2 + (-2)
        uint16x8_t Gy6 = vsubq_u16(Gy1,Gy2); // +1 + (-1)
        uint16x8_t Gy = vaddq_u16(Gy5,Gy6);
        
        int16x8_t temp_y = vreinterpretq_s16_u16(Gy);
        temp_y = vabsq_s16(temp_y);
        Gy = vreinterpretq_u16_s16(temp_x);
        
        uint16x8_t sum = vaddq_u16(Gx,Gy);

        uint8x8_t sum_8bit = vqmovn_u16(sum);
        vst1_u8(sobel,sum_8bit);   
          
    }
}

void* grayscale_sobel_neon(void* threadArgs){
    
    threadStruct *args = (threadStruct *)threadArgs;
    
    int size = ((args->stop - args->start)) + 1;
    
    int sobelSize = ((args->sobelStop - args->sobelStart)) + 1;
        
    while(1) {
        grayscale_neon((args->inputFrame->data + args->inputStart), args->grayscaleFrame->data + args->start, size);
        
        pthread_barrier_wait(&barrier);
        
        sobel_neon(args->grayscaleFrame->data + args->sobelStart, args->procFrame->data + args->sobelStart, sobelSize, args->row, args->col);
        
        pthread_barrier_wait(&barrier);
        
    }
}
