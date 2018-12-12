//
// Created by Yee and Mohammad on 12/11/2018.
//

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include "input_image.cc"
#include <string>
#include <chrono>
using namespace std;


// separate the even/odd elements to the lower halves of array respectively
// param: array --> the pointer to the array
// param: n--> the number of elements in the array

void separate (Complex *array, int n) {
    Complex *b = new Complex[n/2];
    for(int i = 0; i <n/2; i++)
        b[i] = array[i*2+1];  // copy all the odd elements to heap storage
    for(int i = 0; i <n/2; i++)
        array[i] = array[i*2];  // copy all the even elements to the lower half of the array
    for(int i = 0; i<n/2; i++)
        array[i+n/2] = b[i];    
    delete[] b;
}
/*this function apply the cooley tuckey formula
    recursively
    param: x--> pointer points to the array
    param: n--> the width of the array
*/
void fft2 (Complex *x, int n) {
    if(n < 2) {
        return;
    } else {
        separate(x, n); // separate the even and odd points
        //cout<<"got upto separate"<<endl;
        fft2(x, n/2); // recurse on even items
        fft2(x + n/2, n/2); // recurse on odd items
        for(int k = 0; k < n/2; k++) {
            Complex e = x[k];  // even index
            Complex o = x[k+n/2]; // odd index
            // w--> the twiddler factor
            Complex w( cos( 2 * M_PI * k/(float)n), -sin(2 * M_PI * k / (float)n) );
            //Complex w = polar(1.0, -2*M_PI*k /n);
            x[k] = e + w*o;
            x[k+n/2] = e - w*o;
        }
    }
}


int main(int argc, char** argv){
    
    InputImage img = InputImage(argv[2]);
    int width = img.get_width();
    int height = img.get_height();
    Complex *x = img.get_image_data();
    Complex *c =  new Complex [width];
    auto start = std::chrono::system_clock::now();
    //fft2(x, width);
    for(int i = 0; i < height; i++) {
        fft2(x+i*width, width);
    }
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);
    cout << "finished computation at " << std::ctime(&end_time)
              << "elapsed time: " << elapsed_seconds.count() << "s\n";
    img.save_image_data(argv[3], x, width, height);
    return 0;
}