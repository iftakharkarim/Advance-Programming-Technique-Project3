#include "input_image.cc"
#include "complex.cuh"
//#include <cuda.h>
#include <algorithm>
#include <cmath>

using namespace std;


// device function which does row wise DFT
__global__ void DFT1(Complex *garray1, Complex *garray2, int width, int height, float pi) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if(index < width*height) {
		//index = min(index, width*height - 1);
		//printf("index %d \n", index);
		int y = index/width;    // current row
		int x = index % width; // current column
		Complex value(0, 0);
		for(int k = 0; k < width; k++) {
			Complex w(cos(2*pi*x*k/width), -sin(2*pi*x*k/width) );    // twiddler factor
			Complex o = garray2[y*width + k];
			value = value + o* w;
		}
		garray1[index] = value;
	}
}

// device function which does column wise DFT
__global__ void DFT2(Complex *garray1, Complex *garray2, int width, int height, float pi) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if(index < width*height) {
		int x = index % width; // current column
		int y = index/width; // current row
		Complex value(0, 0);
		for(int k = 0; k < height; k++) {
			Complex w(cos((2*pi*y*k)/width), -sin((2*pi*y*k)/width)); // twiddler factor
			value = value + garray2[x + width*k] * w;
		}
		garray1[index] = value;
	}
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
	int width = input.get_width();
	int height = input.get_height();
	cout<<"w  "<<width<<"  h  "<<height<<endl;


	cout<<input.get_image_data()<<endl;
	Complex *inputarray = input.get_image_data();
	cout<<inputarray[0]<<endl;
	cout<<inputarray[1]<<endl;

	// define, allocate and copy data for gpu
	Complex *garray1, *garray2;
	int size = width*height*sizeof(Complex);
	cout<<"size of complex: "<<sizeof(Complex)<<endl;
	cudaMalloc((void**) &garray1, size);
	cudaMalloc((void**) &garray2, size);
	cout<<"afte declareing cudamalloc"<<endl;

	cudaMemcpy(garray1, inputarray, size, cudaMemcpyHostToDevice);
	cudaMemcpy(garray2, inputarray, size, cudaMemcpyHostToDevice);
	cout<<"after cuda copy"<<endl;

	// do the first dft which is row wise
	int numOfThreads = (width*height < 512)? 512: 1024;
	int numofgrid = ((width*height/numOfThreads) < 1)? width*height/numOfThreads + 1:width*height/numOfThreads;
	DFT1<<<numofgrid, numOfThreads >>>(garray1, garray2, width, height, 3.14159265358979323846);
	cout<<"number of grids: "<<numofgrid<<endl;
	cudaDeviceSynchronize();
	cudaMemcpy(inputarray, garray1, size, cudaMemcpyDeviceToHost);
	cout<<"copied data to gpu1 "<<endl;

	// copy everything back to gpu in the second array
	cudaMemcpy(garray2, inputarray, size, cudaMemcpyHostToDevice);
	cout<<"copied data to gpu2 "<<endl;

	// do the second dft which is column wise
	//DFT2<<<numofgrid, numOfThreads>>>(garray1, garray2, width, height, 3.14159265358979323846);
	cudaDeviceSynchronize();

	// copy the final result
	cudaMemcpy(inputarray, garray1, size, cudaMemcpyDeviceToHost);
	
	// free all the memory
	cudaFree(garray1), cudaFree(garray2);

	// write output into the given text file
	input.save_image_data(argv[3], inputarray, width, height);
	return 0;
}