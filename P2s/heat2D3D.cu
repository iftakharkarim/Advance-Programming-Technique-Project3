#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <stdlib.h>
#include <fstream>
#include <string.h>
#include <vector>
#include <stdio.h>
#include <float.h>
#include <sstream>
#include <algorithm>


using namespace std;

// helper method which takes a line and parse each number into integer
// and adds them into a vector
void stringToIntParser (string &temp, vector <float> &temp2)
{
  stringstream str(temp);
  while (str.good() )
  {
    string substr;
    getline(str,substr,',');
    temp2.push_back(atof(substr.c_str()));
  }
}

// helper method to check if a line has # character of not
bool hasHash (string s){
  for ( int i= 0; i < s.length(); i++) {
    if (s.at(i) == '#')
      return true;
  }
  return false;
}

// the kernel function which does the calculation for 3 dimension
__global__ void caclulation3D(float *a, float *b, float *c, int width, int height, int depth, float vK) { 

  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if(index < (width*height*depth)) {

    int x = index % width;   // current column -- x direction width
    int y = (index / width) % height; // current row y direction height
    int z = index / (width*height); // current depth in z direction

    int left = max (x-1, 0);
    int right = min (x+1, width-1);
    int top = max (0, y-1);
    int bottom = min(y+1, height-1);

    int front = max (z -1, 0);
    int back = min (z+1, depth - 1);

    if(c[index] != FLT_MIN) {
      a[index] = b[index] + vK * ( b[front*width*height + y*width + x] + b[back*width*height + y*width + x]
        +b[z*width*height + top*width + x] + b[z*width*height + bottom*width + x] 
        + b[z*width*height + y*width + left] + b[z*width*height + y*width + right]  - 6*b[index]);
    }
  }
}

// the kernel function which does the calculation
__global__ void caclulation2D(float *a, float *b, float *c, int width, int height, float vK) { 

  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if(index < width*height) {

    int y = index / width; // current row y direction height
    int x = index % width;   // current column -- x direction width

    int left = max (x-1, 0);
    int right = min (x+1, width-1);
    int top = max (0, y-1);
    int bottom= min(y+1, height-1);

    //printf(" the index are index = %d, (x,y) = (%d, %d) left =%d, right = %d, top = %d , bottom = %d \n", 
      //index, x, y, left, right, top, bottom);

    if(c[index] != FLT_MIN) {
      a[index] = b[index] + vK *(b[x+top*width] + b[x+bottom*width] + b[left+y*width] + b[right + y*width]  - 4*b[index]);
    }
  }
}


// initializing the cpu memory for two dimension grid points
void initialize2D(float *array1, float * array2, int width, int height, float fixed, std::vector<float> v) {
  for(int y = 0; y < height; y++) {
    for(int x = 0; x<width; x++) {
      array1[x+ width*y] = fixed;
      array2[x+ width*y] = fixed;
    }
  }
  int hx, hy, wx, wy;
  float heat;
  int idx = 0;
  while(idx < v.size()) {
    hx = (int) v[idx], hy = (int) v[idx+1];
    wx = (int) v[idx+2], wy = (int) v[idx+3];
    heat = v[idx+4];
    idx += 5;

    for(int y = hy; y < hy + wy; y++) {
      for (int x = hx; x < hx + wx; x++) {
        array1[x + width*y] = heat;
        array2[x+width*y] = FLT_MIN;
      }
    }

  }
}

// initializing the cpu array for 3D grid points 
void initialize3D(float *array1, float * array2, int width, int height, int depth, float fixed, std::vector<float> v) {
  for(int z = 0; z < depth; z++) {
    for (int y = 0; y < height; y++) {
      for(int x = 0; x < width; x++) {
        array1[z*width*height + y*width + x] = fixed;
        array2[z*width*height + y*width + x] = fixed;
      }
    }
  }
  int hx, hy, hz, wx, wy, wz;
  float heat;
  int idx = 0;
  while(idx < v.size()) {
  hx = (int) v[idx], hy = (int) v[idx+1], hz = (int) v[idx+2];
  wx = (int) v[idx+3], wy = (int) v[idx+4], wz =(int) v[idx+5];
  heat = v[idx+6];
  idx += 7;

  for(int z = hz; z <hz+wz; z++) {
    for(int y = hy; y < hy +wy; y++) {
      for (int x = hx; x < hx + wx; x++) {
        array1[z*width*height + y*width + x] = heat;
        array2[z*width*height + y*width + x] = FLT_MIN;
      }
    }
  }

 }
}


// helper method which write the final result into a csv file
int writeoutput(float *array, char const *s, int width, int height, int depth, int dimension) {
    FILE *f = fopen(s, "w+");
    if(f == NULL) return -1;
    if(dimension == 3) {
      for(int z = 0; z < depth; z++) {
        for(int y = 0; y < height; y++) {
          for(int x = 0; x < width; x++) {
            if(x != width-1)
              fprintf(f, "%f, ", array[z*width*height + y*width + x]);
            else
              fprintf(f, "%f\n", array[z*width*height + y*width + x]); // print a new line after each row
          }
        }
        fprintf(f, "\n"); // printing a blank line
      }
    } else {
      for(int y = 0; y < height; y++) {
        for(int x = 0; x < width; x++) {
          if(x != width-1) 
            fprintf(f, "%f, ", array[y*width + x]);
          else
            fprintf(f, "%f\n", array[y*width + x]);
        }
      }
    }
    fclose(f);
    return 0;
}
// main starts here
int main (int argc, char** argv ) {
  // initializing the local variable
  ifstream configfile(argv[1]);
  string line;
  vector <string> stringarray;  //string vector to store all the line from input file
  int dimension;     // dimension 2D or 3D
  vector<float>heightWeidthDepth;    //height (row), width (col) and depth of  the grid
  float startTemp;    // default starting temperature for the grid
  vector<float>heaterLocation;
  int timeStep;     // total time step
  float valueK;       //the k value which is constant

  // start reading all the line from the file
  while (getline(configfile, line)) {
    if (line.empty() || hasHash(line)) {}
    else
      stringarray.push_back(line);
  }

  // the first line is dimension get it and convert in into integer
  dimension = atoi(stringarray[0].c_str());
  // the second value is constant k
  valueK = atof(stringarray[1].c_str());
  // the third value is number of tiestep in integer
  timeStep   = atoi(stringarray[2].c_str());
  // the height and width and depth which is in one line
  stringToIntParser(stringarray[3], heightWeidthDepth);

  // the 4th value is the default starting temperature
  startTemp = atof(stringarray[4].c_str());

  // the rest of the values are heater location
  // which can be 0 or 1 or more
  for ( int i = 5; i < stringarray.size(); ++i)
  {
    stringToIntParser(stringarray[i], heaterLocation);
  }

  int height = (int) heightWeidthDepth[1]; // y axis
  int width = (int) heightWeidthDepth[0]; // x axis
  int depth = 0;
  int size;
  if(dimension == 3) {depth = (int) heightWeidthDepth[2];}

  if(dimension == 3)
    size = height*width*depth*sizeof(float); // total number of points
  else
    size = height*width*sizeof(float);

  // declare the cpu array
  float *array1 = (float*)malloc(size);
  float *array2 = (float*)malloc(size);

  // declare the gpu variable
  float *gArray1, *gArray2, *gArray3;

  cudaMalloc((void **) &gArray1, size);
  cudaMalloc((void **) &gArray2, size);
  cudaMalloc((void **) &gArray3, size);

  // initialize the cpu array
  if(dimension == 2)
    initialize2D(array1, array2, width, height, startTemp, heaterLocation);
  else
    initialize3D(array1, array2, width, height, depth, startTemp, heaterLocation);


  // copy all data to the device from cpu
  cudaMemcpy(gArray1, array1, size, cudaMemcpyHostToDevice);
  cudaMemcpy(gArray2, array1, size, cudaMemcpyHostToDevice);
  cudaMemcpy(gArray3, array2, size, cudaMemcpyHostToDevice);

  // call the kernel function swap the gArray1 has the updated value after each odd(1,3,5) 
  //time step and gArray2
  int points = (depth == 0)? height*width:height*width*depth;
  int numOfThreads = (points < 512) ? 512: 1024;
  for(int i = 0; i < timeStep; i++) {
    if(i%2 == 0) {
      if(dimension == 2)
        caclulation2D<<<(points/numOfThreads) + 1 , numOfThreads >>>(gArray1, gArray2, gArray3, width, height, valueK);
      else
        caclulation3D <<< (points/numOfThreads) + 1 , numOfThreads>>> (gArray1, gArray2, gArray3, width, height, depth, valueK);
    } else {
      if(dimension == 2)
        caclulation2D<<<(points/numOfThreads) + 1 , numOfThreads >>>(gArray2, gArray1, gArray3, width,height, valueK);
      else
        caclulation3D <<< (points/numOfThreads) + 1 , numOfThreads>>> (gArray2, gArray1, gArray3, width, height, depth, valueK);
    }
    cudaDeviceSynchronize();
  }
  // read from gpu to cpu based on timestep
  if(timeStep % 2 == 0) 
    cudaMemcpy(array1, gArray2, size, cudaMemcpyDeviceToHost);
  else
    cudaMemcpy(array1, gArray1, size, cudaMemcpyDeviceToHost);
  // write output to a .csv file
  writeoutput(array1, "heatOutput.csv",  width, height, depth, dimension);

  // free all the allocated memory both in gpu and cpu
  cudaFree(gArray1), cudaFree(gArray2), cudaFree(gArray3);
  free(array1), free(array2);
  return 0;
}
