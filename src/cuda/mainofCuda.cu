#include "input_image.cc"
#include "complex.cuh"
//#include <cuda.h>
#include <algorithm>
#include <cmath>
//#include <chrono>

using namespace std;

// device function which does row wise DFT
__global__ void DFT1(Complex *garray1, Complex *garray2, int width, int height, float PI) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if(index < width*height) {   // making sure index is not going out of bound of the array
		int y = index/width;    // current row
		int x = index % width;  // current column
		Complex value(0, 0);
		for(int k = 0; k < width; k++) {
			Complex w(cos(2*PI*x*k/width), -sin(2*PI*x*k/width) );    // twiddler factor w
			Complex o = garray2[y*width + k];
			value = value + o* w;
		}
		garray1[index] = value;
	}
}

// device function which does column wise DFT
__global__ void DFT2(Complex *garray1, Complex *garray2, int width, int height, float PI) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if(index < width*height) { // making sure index is not going out of bound of the array
		int x = index % width; // current column
		int y = index/width;   // current row
		Complex value(0, 0);
		for(int k = 0; k < height; k++) {
			Complex w(cos((2*PI*y*k)/width), -sin((2*PI*y*k)/width));  // twiddler factor w
			Complex o = garray2[k*width + x];
			value = value + o* w;
		}
		garray1[index] = value;
	}
}

// main function starts here

int main(int argc, char const **argv)
{	
	if(argc == 4){
		ifstream inFile(argv[2]);
		ifstream outFile(argv[3]);
	}else{
		cout<<"Expected ./file forward/reverse inputfile outputfile"<<endl;
	}
	
	InputImage input = InputImage(argv[2]);
	int width = input.get_width();
	int height = input.get_height();
	Complex *inputarray = input.get_image_data();

	// define, allocate and copy data for gpu
	Complex *garray1, *garray2;

	int size = width*height*sizeof(Complex);

	cudaMalloc((void**) &garray1, size);
	cudaMalloc((void**) &garray2, size);

	cudaMemcpy(garray1, inputarray, size, cudaMemcpyHostToDevice);
	cudaMemcpy(garray2, inputarray, size, cudaMemcpyHostToDevice);

	// do the first dft which is row wise
	int points = width*height;
	int numOfThreads = (points < 512)? 512: 1024;
	int numofgrid = ((points/numOfThreads) < 1)? points/numOfThreads + 1:points/numOfThreads;

	DFT1<<<numofgrid, numOfThreads>>>(garray1, garray2, width, height, PI);
	cudaDeviceSynchronize();

	// do the second dft which is column wise
	DFT2<<<numofgrid, numOfThreads>>>(garray2, garray1, width, height, PI);
	cudaDeviceSynchronize();

	// copy the final result into inputarray
	cudaMemcpy(inputarray, garray2, size, cudaMemcpyDeviceToHost);

	
	// free all the memory and write output
	cudaFree(garray1), cudaFree(garray2);
	input.save_image_data(argv[3], inputarray, width, height);

	return 0;
}