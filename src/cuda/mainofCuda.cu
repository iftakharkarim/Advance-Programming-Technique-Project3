#include "input_image.h"
#include "input_image.cu"
#include "complex.h"
#include "complex.cu"
#include <cuda.h>
#include <algorithm>
#include <cmath>
using namespace std;

#define M_PI 3.14159265358979323846

// easy copy paste qsub
// qsub -I -q coc-ice -l nodes=1:ppn=1:nvidiagpu:teslap100,walltime=2:00:00,pmem=2gb

// device function which does row wise DFT
__global__ void doDFT1(Complex *gpuArray1, Complex *gpuArray2, int width, int height, float M_PI) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	index = min(index, width*height - 1);
	int y = index / width; // current row
	Complex value = new Complex();
	for(int k = 0; k < width; k++) {
		Complex w = exp(new Complex(0, -2*M_PI*k*y/(width))); // twiddler factor
		value = value + gpuArray2[y*width + k] * w;
	}
	gpuArray1[index] = value;
}
// device function which does column wise DFT
__global__ void doDFT2(Complex *gpuArray1, Complex *gpuArray2, int width, int height, float M_PI) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	index = min(index, width*height - 1);
	int x = index % width; // current column
	Complex value = new Complex();
	for(int k = 0; k < height; k++) {
		Complex w = exp(new Complex(0, -2*M_PI*k*x/(height))); // twiddler factor
		value = value + gpuArray2[x + width*k] * w;
	}
	gpuArray1[index] = value;
}


int main(int argc, char const **argv)
{	
	if(argc == 4){
		ifstream inFile(argv[2]);
		ifstream outFile(argv[3]);
	}else{
		cout<<"Expected ./file forward/reverse inputfile outputfile"<<endl;
	}
	
	InputImage input = InputImage(argv[2]);
	InputImage output = InputImage(argv[3]);
	int width = input.get_width();
	int height = input.get_height();
	cout<<"w  "<<width<<"  h  "<<height<<endl;


	cout<<input.get_image_data()<<endl;
	Complex *inputArray = input.get_image_data();
	cout<<*inputArray<<endl; // reading from pointer
	cout<<*(inputArray+1)<<endl;

	// define, allocate and copy data for gpu
	Complex *gpuArray1, *gpuArray2;
	int size = width*height*sizeof(Complex);

	cudaMalloc((void**)&gpuArray1, size);
	cudaMalloc((void**)&gpuArray2, size);
	//cudaMalloc(&gpuArray1, size);
	//cudaMalloc(&gpuArray2, size);

	cudaMemcpy(gpuArray1, &inputArray, size, cudaMemcpyHostToDevice);
	cudaMemcpy(gpuArray2, &inputArray, size, cudaMemcpyHostToDevice);

	// do the first dft which is row wise
	int numOfThreads = (width*height < 1024)? 512 : 1024;
	doDFT1<<<(size/numOfThreads) + 1 , numOfThreads >>>(gpuArray1, gpuArray2, width, height, M_PI);
	cudaMemcpy(inputArray, gpuArray1, size, cudaMemcpyDeviceToHost);

	// copy everything back to gpu in the second array
	cudaMemcpy(gpuArray2, &inputArray, size, cudaMemcpyHostToDevice);

	// do the second dft which is column wise
	doDFT2<<<(size/numOfThreads)+1 , numOfThreads>>>(gpuArray1, gpuArray2, width, height, M_PI);

	// copy the final result
	cudaMemcpy(inputArray, gpuArray1, size, cudaMemcpyDeviceToHost);
	
	// free all the memory
	cudaFree(gpuArray1), cudaFree(gpuArray2);

	// write output into the given text file
	input.save_image_data(argv[3], inputArray, width, height);
	return 0;
}