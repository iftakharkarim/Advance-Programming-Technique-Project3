//
// Created by Nick Landi on 12/8/2018.
//

#include <stdio>
#include "complex.h"
#include <fstream>
#include <vector>
#include "input_image.h"


int main(int argc, char** argv){
    ifsream inputFile;
    ofstream outputFile;
    if(argc == 4){
        inputFile(argv[2]);
        outputFile(argv[3]);
    }
    else{
        std::cout << "Unexpected call, please format ./file mode inputfile outputfile" << std::endl;
    }

    InputImage img = InputImage(argv[2]);
    int imgWidth = img.get_width();
    int imgHeight = img.get_height();

    std::cout << imgHeight + ", " +  imgWidth << std::endl;



    return 0;
}