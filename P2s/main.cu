#include <stdio.h>
#include <algorithm>
#include <fstream> 
#include <iostream>
#include <vector>
#include <math.h>
#include <string>
#include <stdlib.h>

// qsub -I -q coc-ice -l nodes=1:ppn=1:nvidiagpu,walltime=2:00:00,pmem=2gb
// qsub -I -q coc-ice -l nodes=1:ppn=1:nvidiagpu:teslap100,walltime=2:00:00,pmem=2gb
// ssh -x nolivares3@coc-ice.pace.gatech.edu
using namespace std;

// global code runs on the device
// need to have one thread correspond to multiple gridpoints
__global__ void gpuIt(float *tNew,float *tOld,float *tOrig,int x,int y,int z,float k,float st) {

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    // may want an if(i < x*y*z) to prevent overflowing, likea thisa
    if(i < x*y*z){

    if(i == 0){ // top left corner
        tNew[i] = tOld[i] + k*(tOld[i+1] + tOld[i] + tOld[i] + tOld[i+x] - 4*tOld[i]);
        //tNew[i] = 1;
    }
    else if(i == x-1){ // top right corner
        tNew[i] = tOld[i] + k*(tOld[i] + tOld[i-1] + tOld[i] + tOld[i+x] - 4*tOld[i]);
        //tNew[i] = 3;
    }
    else if(i == x*y - 1){ // bottom right corner
        tNew[i] = tOld[i] + k*(tOld[i] + tOld[i-1] + tOld[i-x] + tOld[i] - 4*tOld[i]);
        //tNew[i] = 5;
    }
    else if(i == x*y - x){ // bottom left corner
        tNew[i] = tOld[i] + k*(tOld[i+1] + tOld[i] + tOld[i-x] + tOld[i] - 4*tOld[i]);
        //tNew[i] = 7;
    }
    else if(i%x == 0){ // left side
        tNew[i] = tOld[i] + k*(tOld[i+1] + tOld[i] + tOld[i-x] + tOld[i+x] - 4*tOld[i]);
        //tNew[i] = 8;
    }
    else if(i%x == x-1){ // right side
        tNew[i] = tOld[i] + k*(tOld[i] + tOld[i-1] + tOld[i-x] + tOld[i+x] - 4*tOld[i]);
        //tNew[i] = 4;
    }
    else if(i - x < 0){ // top row
        tNew[i] = tOld[i] + k*(tOld[i+1] + tOld[i-1] + tOld[i] + tOld[i+x] - 4*tOld[i]);
        //tNew[i] = 2;
    }
    else if(i + x > x*y){ // bottom row
        tNew[i] = tOld[i] + k*(tOld[i+1] + tOld[i-1] + tOld[i-x] + tOld[i] - 4*tOld[i]);
        //tNew[i] = 6;
    }
    else{
        tNew[i] = tOld[i] + k*(tOld[i+1] + tOld[i-1] + tOld[i-x] + tOld[i+x] - 4*tOld[i]);
        //tNew[i] = 9;
    }
    //tNew[i] = i; // for debugging
    // replace heaters
    if(tOrig[i] != st){
        tNew[i] = tOrig[i];
    }
    //tNew[i] = i%x;
    }
}

// thisll work for 3d, less if/elses this way
__global__ void gpuIt3(float *tNew,float *tOld,float *tOrig,int x,int y,int z,float k,float st) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
if(i < x*y*z){

    if(i == 0){ // front upper left corner
        tNew[i] = tOld[i]+k*(tOld[i]+tOld[i+(x*y)]+tOld[i]+tOld[i+x]+tOld[i]+tOld[i+1]-6*tOld[i]);
        //tNew[i] = 0;
    }
    else if(i == x-1){ // front upper right corner
        tNew[i] = tOld[i]+k*(tOld[i]+tOld[i+(x*y)]+tOld[i]+tOld[i+x]+tOld[i-1]+tOld[i]-6*tOld[i]);
        //tNew[i] = .1;
    }
    else if(i == x*y-1){ // front lower right corner
        tNew[i] = tOld[i]+k*(tOld[i]+tOld[i+(x*y)]+tOld[i-x]+tOld[i]+tOld[i-1]+tOld[i]-6*tOld[i]);
        //tNew[i] = .2;
    }
    else if(i == x*y-x){ // front lower left corner
        tNew[i] = tOld[i]+k*(tOld[i]+tOld[i+(x*y)]+tOld[i-x]+tOld[i]+tOld[i]+tOld[i+1]-6*tOld[i]);
        //tNew[i] = .3;
    }
    else if(i == x*y*(z-1) ){ // back upper left corner
        tNew[i] = tOld[i]+k*(tOld[i-(x*y)]+tOld[i]+tOld[i]+tOld[i+x]+tOld[i]+tOld[i+1]-6*tOld[i]);
        //tNew[i] = .4;
    }
    else if(i == x*y*(z-1) + x-1){ // back upper right corner
        tNew[i] = tOld[i]+k*(tOld[i-(x*y)]+tOld[i]+tOld[i]+tOld[i+x]+tOld[i-1]+tOld[i]-6*tOld[i]);
        //tNew[i] = .5;
    }
    else if(i == x*y*z-1){ // back lower right corner
        tNew[i] = tOld[i]+k*(tOld[i-(x*y)]+tOld[i]+tOld[i-x]+tOld[i]+tOld[i-1]+tOld[i]-6*tOld[i]);
        //tNew[i] = .6;
    }
    else if(i == x*y*z - x){ // back lower left corner
        tNew[i] = tOld[i]+k*(tOld[i-(x*y)]+tOld[i]+tOld[i-x]+tOld[i]+tOld[i]+tOld[i+1]-6*tOld[i]);
        //tNew[i] = .7;
    }

    else if(i - x < 0){ // front top edge
        tNew[i] = tOld[i]+k*(tOld[i]+tOld[i+(x*y)]+tOld[i]+tOld[i+x]+tOld[i-1]+tOld[i+1]-6*tOld[i]);
        //tNew[i] = .8;
    }
    else if(i%x == x-1 && i<x*y){ // front right edge
        tNew[i] = tOld[i]+k*(tOld[i]+tOld[i+(x*y)]+tOld[i-x]+tOld[i+x]+tOld[i-1]+tOld[i]-6*tOld[i]);
        //tNew[i] = .9;
    }
    else if(i+x > x*y && i < (x*y)){ // front bottom edge
        tNew[i] = tOld[i]+k*(tOld[i]+tOld[i+(x*y)]+tOld[i-x]+tOld[i]+tOld[i-1]+tOld[i+1]-6*tOld[i]);
        //tNew[i] = 1;
    }
    else if(i%x == 0 && i<x*y){ // front left edge
        tNew[i] = tOld[i]+k*(tOld[i]+tOld[i+(x*y)]+tOld[i-x]+tOld[i+x]+tOld[i]+tOld[i+1]-6*tOld[i]);
        //tNew[i] = 2;
    }

    else if(i > (x*y*z - x*y) && i < (x*y*z - (x-1)*y)){ // back top edge
        tNew[i] = tOld[i]+k*(tOld[i-(x*y)]+tOld[i]+tOld[i]+tOld[i+x]+tOld[i-1]+tOld[i+1]-6*tOld[i]);
        //tNew[i] = 3;
    }
    else if(i%x == x-1 && i > (x*y*(z-1))){ // back right edge
        tNew[i] = tOld[i]+k*(tOld[i-(x*y)]+tOld[i]+tOld[i-x]+tOld[i+x]+tOld[i-1]+tOld[i]-6*tOld[i]);
        //tNew[i] = 4;
    }
    else if(i+x > x*y*z){ // back bottom edge
        tNew[i] = tOld[i]+k*(tOld[i-(x*y)]+tOld[i]+tOld[i-x]+tOld[i]+tOld[i-1]+tOld[i+1]-6*tOld[i]);
        //tNew[i] = 5;
    }
    else if(i%x == 0 && i > x*y*(z-1)){ // back left edge
        tNew[i] = tOld[i]+k*(tOld[i-(x*y)]+tOld[i]+tOld[i-x]+tOld[i+x]+tOld[i]+tOld[i+1]-6*tOld[i]);
        //tNew[i] = 6;
    }

    // the corner sides going front to back
    else if(i%(x*y) == 0){ // upper left edge
        tNew[i] = tOld[i]+k*(tOld[i-(x*y)]+tOld[i+(x*y)]+tOld[i]+tOld[i+x]+tOld[i]+tOld[i+1]-6*tOld[i]);
        //tNew[i] = 7;
    }
    else if(i%(x*y) == x-1){ // upper right edge
        tNew[i] = tOld[i]+k*(tOld[i-(x*y)]+tOld[i+(x*y)]+tOld[i]+tOld[i+x]+tOld[i-1]+tOld[i]-6*tOld[i]);
        //tNew[i] = 8;
    }
    else if(i%(x*y) == x*y-1){ // lower right edge
        tNew[i] = tOld[i]+k*(tOld[i-(x*y)]+tOld[i+(x*y)]+tOld[i-x]+tOld[i]+tOld[i-1]+tOld[i]-6*tOld[i]);
        //tNew[i] = 9;
    }
    else if(i%(x*y) == x*y-x){ // lower left edge
        tNew[i] = tOld[i]+k*(tOld[i-(x*y)]+tOld[i+(x*y)]+tOld[i-x]+tOld[i]+tOld[i]+tOld[i+1]-6*tOld[i]);
        //tNew[i] = 9.1;
    }

    // else ifs here are vague because other options already completed
    else if(i < x*y){ // front face
        tNew[i] = tOld[i]+k*(tOld[i]+tOld[i+(x*y)]+tOld[i-x]+tOld[i+x]+tOld[i-1]+tOld[i+1]-6*tOld[i]);
        //tNew[i] = 1.1;
    }
    else if(i > x*y*(z-1)){ // back face
        tNew[i] = tOld[i]+k*(tOld[i-(x*y)]+tOld[i]+tOld[i-x]+tOld[i+x]+tOld[i-1]+tOld[i+1]-6*tOld[i]);
        //tNew[i] = 1.2;
    }
    else if(i%(x*y) < x){ // top face
        tNew[i] = tOld[i]+k*(tOld[i-(x*y)]+tOld[i+(x*y)]+tOld[i]+tOld[i+x]+tOld[i-1]+tOld[i+1]-6*tOld[i]);
        //tNew[i] = 1.3;
    }
    else if(i%(x*y) > x*(y-1)){ // bottom face
        tNew[i] = tOld[i]+k*(tOld[i-(x*y)]+tOld[i+(x*y)]+tOld[i-x]+tOld[i]+tOld[i-1]+tOld[i+1]-6*tOld[i]);
        //tNew[i] = 1.4;
    }
    else if(i%(x) == x-1){ // right face
        tNew[i] = tOld[i]+k*(tOld[i-(x*y)]+tOld[i+(x*y)]+tOld[i-x]+tOld[i+x]+tOld[i-1]+tOld[i]-6*tOld[i]);
        //tNew[i] = 1.5;
    }
    else if(i%(x) == 0){ // left face
        tNew[i] = tOld[i]+k*(tOld[i-(x*y)]+tOld[i+(x*y)]+tOld[i-x]+tOld[i+x]+tOld[i]+tOld[i+1]-6*tOld[i]);
        //tNew[i] = 1.6;
    }
    else{ // all in the middle
        //                       front        back         top       bottom     left     right 
        tNew[i] = tOld[i]+k*(tOld[i-(x*y)]+tOld[i+(x*y)]+tOld[i-x]+tOld[i+x]+tOld[i-1]+tOld[i+1]-6*tOld[i]);
    }


//tNew[i] = i%(x*y);
// replace heaters
if(tOrig[i] != st){
    tNew[i] = tOrig[i];
}

}
}


// blockIdx.x for blocks, threadIdx.x for threads
// name<<<blocks,threads per block>>>
// 


// host is cpu




int main(int argc, char** argv) {
    //cout<<argv[1]<<endl;
    int dim;
    ifstream inFile(argv[1]);

    string line;
    string dims;
    //string heaters;
    getline(inFile,line);
    if(argv[1]==NULL){
        cout<<"Error, no config file specified."<<endl;
    }
    // move to next nums
    while(line[0] == '#' || line.length() == 1){getline(inFile,line);}
    // gets 2d/3d as an int of 2 or 3
    if(line[0] != '#' && line.length() != 1){
        dim = line[0] - 48; // ascii starts at 48
        //cout<<"dim  "<<dim<<endl;
    }
    getline(inFile,line);

    // k value
    while(line[0] == '#' || line.length() == 1){getline(inFile,line);}
    int delimPos = line.find('\n');
    string kstring = line.substr(0,delimPos);
    float K = strtof(&kstring[0],NULL);
    //cout<<"K "<<K<<endl;

    // timesteps
    getline(inFile,line);
    while(line[0] == '#' || line.length() == 1){getline(inFile,line);}
    delimPos = line.find('\n');
    string ts = line.substr(0,delimPos);
    float TS = strtof(&ts[0],NULL);
    //cout<<"TS  "<<TS<<endl;

    // dims
    getline(inFile,line);
    while(line[0] == '#' || line.length() == 1){getline(inFile,line);}
    delimPos = line.find(',');
    string xd = line.substr(0,delimPos);
    float XD = strtof(&xd[0],NULL);
    //cout<<"XD  "<<XD<<endl;
    float YD, ZD;
    if(dim == 3){
        int delimPos2 = line.find(',',delimPos+1);
        string yd = line.substr(delimPos+1,delimPos2);
        YD = strtof(&yd[0],NULL);
        //cout<<"YD  "<<YD<<endl;

        string zd = line.substr(delimPos2+1,line.length());
        ZD = strtof(&zd[0],NULL);
        //cout<<"ZD  "<<ZD<<endl;
        //cout<<"d1  "<<delimPos<<"  d2  "<<delimPos2<<endl;

    }else if (dim == 2){ // 2d only needs y
        string yd = line.substr(delimPos+1,line.length());
        YD = strtof(&yd[0],NULL);
        //cout<<"YD  "<<YD<<endl;
        ZD = 1;
    }else{
        if(argv[1]!=NULL){
            cout<<"Config file read error"<<endl;
        }
    }
    
    // starting temp
    getline(inFile,line);
    while(line[0] == '#' || line.length() == 1){getline(inFile,line);}
    delimPos = line.find('\n');
    string startstring = line.substr(0,delimPos);
    float StTmp = strtof(&startstring[0],NULL);
    //cout<<"StTmp "<<StTmp<<endl;

    // heaters

    int hDims[300];
    float hTemps[75];
    char*dup; // get around line being annoying

    int count = 0;
    while(getline(inFile,line)){
        if(line[0] != '#' && line.length() > 1){
            if(dim == 2){
                dup = new char[line.size() + 1];
                copy(line.begin(),line.end(),dup);
                dup[line.size()] = '\0';
                hDims[0+count*4] = atoi(strtok(dup,","));
                //cout<<"this boi"<<hDims[0+count*4]<<endl;
                for(int i = 1; i < 4; i++){
                    hDims[i+count*4] = atoi(strtok(NULL,","));
                }
                hTemps[count] = atof(strtok(NULL,","));
                delete dup;

                count++;
            }else{ // for 3d
                dup = new char[line.size() + 1];
                copy(line.begin(),line.end(),dup);
                dup[line.size()] = '\0';
                hDims[0+count*6] = atoi(strtok(dup,","));
                //cout<<"this boi"<<hDims[0+count*4]<<endl;
                for(int i = 1; i < 6; i++){
                    hDims[i+count*6] = atoi(strtok(NULL,","));
                }
                hTemps[count] = atof(strtok(NULL,","));
                delete dup;

                count++;
            }
            
        }
    }
    //for(int i = 0; i < count*4; i++){
    //    cout<<hDims[i]<<endl;
    //}
    //cout<<"htemps 0 and 1  "<<hTemps[0]<<" "<<hTemps[1]<<endl;

    // use count to figure out how many heaters there are
    inFile.close();




    // k can be from 0 to 1/(number of neighbors)
    float k = K;
    int xDim = XD;
    int yDim = YD;
    int zDim = ZD;
    int timeSteps = TS;
    int gSize = xDim*yDim*zDim; // -1 for zero indexing
    float tOrig[gSize];
    float tOld[gSize];
    float tNew[gSize];
    //cout<<"StTmp "<<StTmp<<endl;
    // sets starting temp for nodes
    for(int i = 0; i < gSize; i++){
        tOrig[i] = StTmp + 0.0;
    }

    // place heaters
    int hLoc; // as one coord
    int hWidth,hHeight,hDepth;
    if(zDim == 1){
        for(int i = 0; i < count; i++){
            hLoc = hDims[0 + 4*i] + hDims[1 + 4*i]*xDim;
            hWidth = hDims[2 + 4*i];
            hHeight = hDims[3 + 4*i];
            for(int j = 0; j < hHeight; j++){
                for(int k = 0; k < hWidth; k++){
                    //cout<<"k "<<k<<"  j  "<<j<<"  hTemps[c]  "<<hTemps[i]<<endl;
                    tOrig[hLoc+k + xDim*j] = hTemps[i];
                    //cout<<"tOrig[]"<<tOrig[hLoc+k + xDim*j]<<endl;
                }
            }
        }
    }if(zDim > 1){ // 3d
        for(int i = 0; i < count; i++){
            hLoc = hDims[0 + 6*i] + hDims[1 + 6*i]*xDim + hDims[2 + 6*i]*xDim*yDim;
            hWidth = hDims[3 + 6*i];
            hHeight = hDims[4 + 6*i];
            hDepth = hDims[5 + 6*i];
            for(int h = 0; h < hDepth; h++){
                for(int j = 0; j < hHeight; j++){
                    for(int k = 0; k < hWidth; k++){
                        //cout<<"k "<<k<<"  j  "<<j<<"  hTemps[c]  "<<hTemps[i]<<endl;
                        tOrig[hLoc+k + xDim*j + xDim*yDim*h] = hTemps[i];
                        //cout<<"tOrig[]"<<tOrig[hLoc+k + xDim*j]<<endl;
                    }
                }
            }
            
        }
    }
    


    memcpy(tNew,tOrig, sizeof(tOld));
    memcpy(tOld,tNew, sizeof(tOld));

    // gets block count with maxed threads
    // each thread should handle one gridpoint at a time
    float gS = gSize;
    int BLOCKS = ceil(gS/1024);
    //cout<<"BLOCKS  "<<BLOCKS<<endl;

    float *d_new, *d_old, *d_orig;// *d_temp;

    cudaMalloc((void**)&d_new, gSize*sizeof(float)); // may need to be sizeof(gSize)
    cudaMalloc((void**)&d_old, gSize*sizeof(float)); // but I don't think so
    cudaMalloc((void**)&d_orig, gSize*sizeof(float)); 

    cudaMemcpy(d_new, tNew, gSize*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_old, tOld, gSize*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_orig, tOrig, gSize*sizeof(float), cudaMemcpyHostToDevice);

    // have to alternate calls, d_old = d_new doesn't work anymore for some reason
    // cout<<"timeSteps/2  "<<timeSteps/2<<endl;
    if(zDim == 1){
        for(int i = 0; i < timeSteps; i++){
            gpuIt<<<BLOCKS,1024>>>(d_new,d_old,d_orig,xDim,yDim,zDim,k,StTmp);
            cudaDeviceSynchronize(); // blocks all CPU until GPU done, TA says correct
            //cout<<"2d Call "<<i<<endl;
            i++;
            if(i < timeSteps){
                gpuIt<<<BLOCKS,1024>>>(d_old,d_new,d_orig,xDim,yDim,zDim,k,StTmp);
                cudaDeviceSynchronize();
                //cout<<"2d Call "<<i<<endl;
            }
        }
    }
    if(zDim > 1){
        for(int i = 0; i < timeSteps; i++){
            gpuIt3<<<BLOCKS,1024>>>(d_new,d_old,d_orig,xDim,yDim,zDim,k,StTmp);
            cudaDeviceSynchronize(); // blocks all CPU until GPU done, TA says correct
            //cout<<"3d Call "<<i<<endl;
            i++;
            if(i < timeSteps){
                gpuIt3<<<BLOCKS,1024>>>(d_old,d_new,d_orig,xDim,yDim,zDim,k,StTmp);
                cudaDeviceSynchronize();
                //cout<<"3d Call "<<i<<endl;
            }
            
        }
    }
    // since d_new and d_old alternate, get the most recent one
    if(timeSteps%2 == 1){
        cudaMemcpy(tNew, d_new, gSize*sizeof(float), cudaMemcpyDeviceToHost);
    }else{
        cudaMemcpy(tNew, d_old, gSize*sizeof(float), cudaMemcpyDeviceToHost);
    }

    // outputs to terminal for easy viewing
    //for(int h = 0; h < zDim; h++){
    //    for(int i = 0; i < yDim; i++){
    //    	for(int j = 0; j < xDim; j++){
    //    		cout<<tNew[j + i*xDim + h*xDim*yDim]<<" ";
    //    }
    //        cout<<endl;
    //        }
    //    cout<<endl;
    //}



    ofstream outFile;
    outFile.open("heatDoutput.csv");
    for(int h = 0; h < zDim; h++){
        for(int i = 0; i < yDim; i++){
            for(int j = 0; j < xDim; j++){
                outFile<<tNew[j + i*xDim + h*xDim*yDim]<<", ";
            }
            outFile<<endl;
        }
        outFile<<endl;
    }
    
    outFile.close();

    cudaFree(d_new);
    cudaFree(d_old);
    cudaFree(d_orig);
}
