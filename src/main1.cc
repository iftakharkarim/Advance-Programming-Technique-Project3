#include <vector>
#include <iostream>
#include <fstream>
#include "input_image.h"
#include "input_image.cc"
#include "complex.h"
#include "complex.cc"

using namespace std;






int main(int argc, char** argv){
	if(argc == 4){
		ifstream inFile(argv[2]);
		ifstream outFile(argv[3]);
	}else{
		cout<<"Expected ./file forward/reverse inputfile outputfile"<<endl;
	}
	
	InputImage tow = InputImage(argv[2]);
	int w = tow.get_width();
	int h = tow.get_height();
	cout<<"w  "<<w<<"  h  "<<h<<endl;


	cout<<tow.get_image_data()<<endl;
	Complex *thing = tow.get_image_data();
	cout<<*thing<<endl;
	cout<<*(thing+1)<<endl;




return 0;
}