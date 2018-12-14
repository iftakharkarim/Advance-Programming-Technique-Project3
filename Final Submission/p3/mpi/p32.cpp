
#include "input_image.cc"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <signal.h>
#include <cmath>
#include <mpi.h>
#include <chrono>

using namespace std;


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

void transpose (Complex *input, Complex *output,int tempwidth, int tempheight)
{
  int counter = 0;
  for ( int i = 0; i < tempheight; ++i)
  {
    for ( int j = 0; j < tempwidth; ++j)
    {
      output[counter] = input[tempwidth*j + i];
      counter++;
    }
  }
}


/*void gather ( Complex* inarray, int processors, int ranking, int totalsize, MPI_Datatype dt_complex, int temprow, int tempwidth)
{
      if(ranking != 0 )
      {
        MPI_Send(inarray, (totalsize/processors), dt_complex, 0, 0, MPI_COMM_WORLD);
      }

      if(ranking == 0)
      {
        for ( int i = 1; i < processors; ++i)
        {
          MPI_Status status;
          Complex* recvBuff = inarray + (i*tempwidth);
          MPI_Recv(recvBuff, (totalsize/processors), dt_complex, i, 0, MPI_COMM_WORLD,&status);
        }
      }
}*/

int main(int argc, char* argv [])
{
  auto start = std::chrono::system_clock::now();
  int np;        //number of processors
  int rank;      //number of ranks
  string fwd_rev;
  string typeimg;
  string inputimg;    //inputimage string
  string outputimg;   //outputimage string
  string s1("forward");
  string s2("reverse");
  typeimg = argv[1];
  inputimg = argv[2];
  outputimg = argv[3];
  InputImage array(inputimg.c_str());
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &np);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  MPI_Datatype dt_complex;
  MPI_Type_contiguous(2,MPI_FLOAT,&dt_complex);
  MPI_Type_commit(&dt_complex);



  int width = array.get_width();
  int height = array.get_height();


  Complex *data = array.get_image_data();
  Complex DFTarray[width*height];
  Complex DFTarrayT[width*height];
  int size = height*width;

  if ( typeimg  == s1 )
  {
    const int process_height = height/np;
    //int process_width = width/np;
    //int process_amount = (size*2)/np;
    const int process_amount = size/np;

    int row = (height*rank)/np;
    //cout<<"rank "<<rank<<" starts "<<row<<" ends "<<row+process_height<<endl;
    for (int i = 0; i < process_height; ++i)
    {

      fft2 ( data + ((i+row)*width), width);
    }


    if(rank != 0 )
    {
      MPI_Send(data + (rank*width*process_height), process_amount, dt_complex, 0, 0, MPI_COMM_WORLD);
    }

    if(rank == 0)
    {
      for ( int i = 1; i < np; ++i)
      {
        MPI_Status status;
        MPI_Recv( data + (i*width*process_height) , process_amount , dt_complex, i, 0, MPI_COMM_WORLD,&status);
      }
    }


    if ( rank == 0)
    {
      transpose(data, DFTarrayT, width, height);
    }

    MPI_Bcast ( DFTarrayT, size, dt_complex, 0, MPI_COMM_WORLD );


    for (int i = 0; i < process_height; ++i)
    {

      fft2 ( DFTarrayT + ((i+row)*width), width);
    }


    if(rank != 0 )
    {
      MPI_Send(DFTarrayT + (rank*width*process_height), process_amount, dt_complex, 0, 0, MPI_COMM_WORLD);
    }

    if(rank == 0)
    {
      for ( int i = 1; i < np; ++i)
      {
        MPI_Status status;
        MPI_Recv( DFTarrayT + (i*width*process_height) , process_amount , dt_complex, i, 0, MPI_COMM_WORLD,&status);
      }
    }

    if ( rank == 0)
    {
      transpose(DFTarrayT, DFTarray, width, height);
    }



    if (rank == 0) {
      array.save_image_data(outputimg.c_str(), DFTarray, width, height);
      auto end = std::chrono::system_clock::now();
      std::chrono::duration<double> elapsed_seconds = end-start;
      std::cout << elapsed_seconds.count() << std::endl;
    }



  }

  MPI_Finalize();



}
