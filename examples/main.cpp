#include "mfem.hpp"
#include "RAJA/RAJA.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

#define CUDA_BLOCK_SZ 256

// This example illustrates interoperability
// between MFEM's memory manager and RAJA.
//
// More importantly it illustrates how to maintain
// validity between a pointer and an alias. In this
// example, memory is allocated externally and
// given to an MFEM data structure.
//

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *device_config = "raja-cuda";

   // 2. Enable hardware devices such as GPUs, and programming models such as
   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
   Device device(device_config);
   device.Print();

   int N = 10;   //Base vector size
   int nSub = 5; //Sub vector size

   double * x = new double[N]; //Allocate base vector data
   double * xsub = &x[nSub];   //Pointer to subvector data

   //Poulate base vector on the host
   for(int i=0; i<N; ++i) x[i] = i+1;

   //Shallow copy into MFEM vector + read command to copy to the GPU
   Vector V(x, N); V.Read();

   //Shallow copy into MFEM vector + annotate as reference to a base vector
   Vector Vs(xsub, nSub); Vs.MakeRef(V, nSub);

   //Sync validity flags with base Vector
   Vs.SyncAliasMemory(V);

   std::cout<<"V - Flags"<<std::endl;
   V.GetMemory().PrintFlags();

   std::cout<<"Vs - Flags "<<std::endl;
   Vs.GetMemory().PrintFlags();

   const double *d_x = V.Read(); //pointer to device memory
   std::cout<<"Contents of V on the GPU"<<std::endl;
   RAJA::forall<RAJA::cuda_exec<CUDA_BLOCK_SZ>>
     (RAJA::RangeSegment(0,N), [=] RAJA_DEVICE (int i) {
       printf("%.1f ", d_x[i]);
   });
   printf("\n");


   //Write on host data using subvector
   double *h_xs = Vs.HostWrite();

   //Modify data on host
   RAJA::forall<RAJA::loop_exec>
     (RAJA::RangeSegment(0,nSub), [=]  (int i) {
       h_xs[i] = 777.0;
     });

   //Sync validity flags with sub vector
   V.SyncMemory(Vs);

   std::cout<<"\n V - Flags - post host write and sync"<<std::endl;
   V.GetMemory().PrintFlags();

   std::cout<<"Vs - Flags - post host write and sync"<<std::endl;
   Vs.GetMemory().PrintFlags();

   V.Read(); //Read data from host as it has changed
   std::cout<<"V - contents post host write and sync"<<std::endl;
   RAJA::forall<RAJA::cuda_exec<CUDA_BLOCK_SZ>>
     (RAJA::RangeSegment(0,N), [=] RAJA_HOST_DEVICE (int i) {
       printf("%.1f ", d_x[i]);
     });
   printf("\n");

   delete [] x;

   return 0;
}
