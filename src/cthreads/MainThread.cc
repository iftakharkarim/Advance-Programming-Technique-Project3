#include <iostream>
#include "complex.h"
#include "complex.cc"
#include "input_image.cc"
#include <fstream>
#include "input_image.h"
#include <thread>
#include <string>
#include <cmath>
#include <cstring>
//#include <chrono>

#define PI 3.14159265358979323846

void rowDFT(Complex *x1, Complex *y1, int start, int last, int width);
void colDFT(Complex *x1, Complex *y1, int start, int last, int width, int height);
void launchRowThreads(Complex *x, Complex *y, int imgWidth, int rowsPerThread);
void launchColThreads(Complex *x, Complex *y, int height, int width, int colPerThread);


int main(int argc, char** argv){
    
    std::ifstream inputFile;
    std::ofstream outputFile;
    if(argc == 4){
    }
    else{
        std::cout << "Unexpected call, please format ./file mode inputfile outputfile" << std::endl;
    }
    //auto timer = std::chrono::system_clock::now();
    std::cout << "begin" << std::endl;
    InputImage img = InputImage(argv[2]);
    int imgWidth = img.get_width();
    int imgHeight = img.get_height();

    std::cout << imgHeight + ", " +  imgWidth << std::endl;

    std::cout << "Image has been read in" << std::endl;

    //determine array size and then do some array splitting so that there is no need to sync threads
    //assuming all inputs are evenly divisible by 16
    int rowsPerThread = imgHeight/16*imgWidth;
    int colPerThread = imgWidth/16*imgWidth;

    Complex *arry = img.get_image_data();
    Complex *cpy = (Complex*) malloc(sizeof(Complex)*imgHeight*imgWidth);

    std::memcpy(cpy, arry, sizeof(Complex)*imgHeight*imgWidth);
    std::cout << "Beginning DFT calculation" << std::endl;

    launchRowThreads(arry,cpy, imgHeight, rowsPerThread);
    launchColThreads(cpy,arry,imgHeight,imgWidth,rowsPerThread);

    std::cout << "First element in array: " << cpy[0].real << ", " << cpy[0].imag << std::endl;
    img.save_image_data(argv[3],cpy,imgWidth, imgHeight);

    //auto end = std::chrono::system_clock::now();
    //std::chrono::duration<double> time = end-timer;
   //std::cout << "Time elapsed = " << time.count() << std::endl;
    return 0;
}

void launchRowThreads(Complex *x, Complex *y, int imgWidth, int rowsPerThread){
    //rows will be right after one another so math is simple

    std::thread F1(rowDFT, x, y, 0, rowsPerThread, imgWidth);
    std::thread F2(rowDFT, x, y, rowsPerThread, 2*rowsPerThread, imgWidth);
    std::thread F3(rowDFT, x, y, 2*rowsPerThread, 3*rowsPerThread, imgWidth);
    std::thread F4(rowDFT, x, y, 3*rowsPerThread, 4*rowsPerThread, imgWidth);
    std::thread F5(rowDFT, x, y, 4*rowsPerThread, 5*rowsPerThread, imgWidth);
    std::thread F6(rowDFT, x, y, 5*rowsPerThread, 6*rowsPerThread, imgWidth);
    std::thread F7(rowDFT, x, y, 6*rowsPerThread, 7*rowsPerThread, imgWidth);
    std::thread F8(rowDFT, x, y, 7*rowsPerThread, 8*rowsPerThread, imgWidth);
    std::thread F9(rowDFT, x, y, 8*rowsPerThread, 9*rowsPerThread, imgWidth);
    std::thread F10(rowDFT, x, y, 9*rowsPerThread, 10*rowsPerThread, imgWidth);
    std::thread F11(rowDFT, x, y, 10*rowsPerThread, 11*rowsPerThread, imgWidth);
    std::thread F12(rowDFT, x, y, 11*rowsPerThread, 12*rowsPerThread, imgWidth);
    std::thread F13(rowDFT, x, y, 12*rowsPerThread, 13*rowsPerThread, imgWidth);
    std::thread F14(rowDFT, x, y, 13*rowsPerThread, 14*rowsPerThread, imgWidth);
    std::thread F15(rowDFT, x, y, 14*rowsPerThread, 15*rowsPerThread, imgWidth);
    std::thread F16(rowDFT, x, y, 15*rowsPerThread, 16*rowsPerThread, imgWidth);

    F1.join();
    F2.join();
    F3.join();
    F4.join();
    F5.join();
    F6.join();
    F7.join();
    F8.join();
    F9.join();
    F10.join();
    F11.join();
    F12.join();
    F13.join();
    F14.join();
    F15.join();
    F16.join();
}

void launchColThreads(Complex *x, Complex *y, int height, int width, int colPerThread){
    //col math is taken care of in method

    std::thread F1(colDFT, x, y, 0*colPerThread, 1*colPerThread,width, height);
    std::thread F2(colDFT, x, y, 1*colPerThread, 2*colPerThread,width, height);
    std::thread F3(colDFT, x, y, 2*colPerThread, 3*colPerThread,width, height);
    std::thread F4(colDFT, x, y, 3*colPerThread, 4*colPerThread,width, height);
    std::thread F5(colDFT, x, y, 4*colPerThread, 5*colPerThread,width, height);
    std::thread F6(colDFT, x, y, 5*colPerThread, 6*colPerThread,width, height);
    std::thread F7(colDFT, x, y, 6*colPerThread, 7*colPerThread,width, height);
    std::thread F8(colDFT, x, y, 7*colPerThread, 8*colPerThread,width, height);
    std::thread F9(colDFT, x, y, 8*colPerThread, 9*colPerThread,width, height);
    std::thread F10(colDFT, x, y,9*colPerThread, 10*colPerThread,width, height);
    std::thread F11(colDFT, x, y, 10*colPerThread, 11*colPerThread,width, height);
    std::thread F12(colDFT, x, y, 11*colPerThread, 12*colPerThread,width, height);
    std::thread F13(colDFT, x, y, 12*colPerThread, 13*colPerThread,width, height);
    std::thread F14(colDFT, x, y, 13*colPerThread, 14*colPerThread,width, height);
    std::thread F15(colDFT, x, y, 14*colPerThread, 15*colPerThread,width, height);
    std::thread F16(colDFT, x, y, 15*colPerThread, 16*colPerThread,width, height);

    F1.join();
    F2.join();
    F3.join();
    F4.join();
    F5.join();
    F6.join();
    F7.join();
    F8.join();
    F9.join();
    F10.join();
    F11.join();
    F12.join();
    F13.join();
    F14.join();
    F15.join();
    F16.join();
}

void rowDFT(Complex *x1, Complex *y1, int start, int last, int width){
    int x, y;

    for(int k = start; k < last; k++) {
        Complex inter(0,0);
        y = k/width;
        x = k%width;
        for (int j = 0; j < width; j++) {
            Complex w(cos(2*PI*x*j/(float)width), -1*sin(2*PI*x*j/(float)width));
            Complex hold = y1[y*width+j];
            inter = inter + hold*w;
        }
        x1[k] = inter;
    }
}

void colDFT(Complex *x1, Complex *y1, int start, int last, int width, int height){
    int x, y;

    for(int k = start; k < last; k++) {
        Complex inter(0,0);
        x = k%height;
        y = k/width;
        for (int j = 0; j < height; j++) {
            Complex W(cos(2*PI*y*j/(float)height), -1*sin(2*PI*y*j/(float)height));
            Complex hold = y1[x+j*height];
            inter = inter + hold*W;
        }
        //x*width+y
        x1[k] = inter;
    }
}
