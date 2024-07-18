//                      MFEM cuBLAS and hipBLAS gemvBatched() Example
//
// Compile with: make blas
//
// Device sample runs:
//               blas -d cuda
//               blas -d cuda -n 2 -ne 2
//               blas -pa -d cuda
//               blas -fa -d cuda
//               blas -d hip
//               blas -d hip -pa
//               blas -d hip -fa
//               blas -d hip

// Description:  This example code demonstrates the use of cu or hipBLAS on MFEM
//               objects to multiply batched square matrices with batched vectors.
//               It utilizes the BLAS functions
//                  cublasDgemvBatched()  or  hipblasDgemvBatched().
//               Note that version cuda/11.7.0 or newer is needed.
//
//               User can specify the number of rows (cols) n for the (nxn) square
//               square matrices stored in DenseTensor A and (nx1) vectors stored in
//               Vector X, as well as scalars alpha and beta to transform the operations:
//                  alpha * A[i] * X[i] + beta = Y[i].
//               Output also solves the linear system Ab = x as needed:
//                  Ainv[i] * X[i] = B[i].
//               This function is specifically included to demonstrate the Batch Linear
//               Algebra functions ported with cu/hipBLAS (see batchlinalg.cpp).

#include "mfem.hpp"
#include "../fem/bilinearform.cpp" // TODO: causes Seg Fault when removed (Note: not included in batchlinalg.cpp so Seg Fault occurs when running inverse portion)
#include <fstream>
#include <iostream>
#define IDXT(i,j,k,ld) (((ld)*(ld)*(k))+((j)*(ld))+(i))
#define IDXM(i,j,ld) ((ld*j)+i)

using namespace std;
using namespace mfem;

int main (int argc, char *argv[])
{
   // 1. Parse command-line options.
   int ndof = 4;
   int num_elem = 3;
   double alpha = 1.;
   double beta = 0.;
   bool inverse = true;
   bool pa = false;
   bool fa = false;
   const char *device_config = "cuda";

   OptionsParser args(argc, argv);
   args.AddOption(&ndof, "-n", "--ndof",
                  "Number of Rows/Columns of batched square matrices.");
   args.AddOption(&num_elem, "-ne", "--num_elem",
                  "Number of matrices in tensor (batch size).");
   args.AddOption(&alpha, "-a", "--alpha",
                  "Scalar for batched matrices.");
   args.AddOption(&beta, "-b", "--beta",
                  "Scalar to add to multiplication.");
   args.AddOption(&inverse, "-inv", "--inverse", "-no-inv",
                  "--no-inverse", "Disable inverse of batched matrices.");
   args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
                  "--no-partial-assembly", "Enable Partial Assembly.");
   args.AddOption(&fa, "-fa", "--full-assembly", "-no-fa",
                  "--no-full-assembly", "Enable Full Assembly.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   // 2. Specify hardware devices such as GPUs.
   Device device(device_config);
   device.Print();
   printf ("\n");

   // 3. Create a DenseTensor A and Vector X to batch multiply. Create a Vector Y container for the result.
   DenseTensor A(ndof,ndof,num_elem);
   Vector X(ndof*num_elem);
   Vector Y(ndof*num_elem);
   printf ("A transpose is \n"); // col-major, so prints vectors of columns of A out together in each row
   for (int k = 0; k < num_elem; k++)
   {
      for (int j = 0; j < ndof; j++)
      {
         X.HostReadWrite()[IDXM(j,k,ndof)] = (double) (IDXM(j,k,ndof));
         for (int i = 0; i < ndof; i++)
         {
            if (i==j) {A.HostReadWrite()[IDXT(i,j,k,ndof)] = 1;}
            else {A.HostReadWrite()[IDXT(i,j,k,ndof)] = (double) (IDXT(j,i,k,ndof));}
            printf ("%9.3f", A.HostRead()[IDXT(i,j,k,ndof)]);
         }
         printf ("\n");
      }
      printf ("\n");
   }
   printf ("X transpose is \n");
   for (int k = 0; k < num_elem; k++)
   {
      for (int j = 0; j < ndof; j++)
      {
         printf ("%9.3f", X.HostRead()[IDXM(j,k,ndof)]);
      }
      printf ("\n");
   }
   printf ("\n");

   // 4. Run CUDA or HIP
   //    Full commands not written out; using MFEM_USE_CUDA_OR_HIP implementation.
   //    See commented section below for example in cuBLAS and hipBLAS.

   // gemvBatched() requires array of pointers for each A, X, Y
   Array<double *>devPtrA(num_elem);
   Array<double *>devPtrX(num_elem);
   Array<double *>devPtrY(num_elem);

   for (int k = 0; k < num_elem; k++)
   {
      devPtrA[k] = &A.ReadWrite()[ndof*ndof*k];
      devPtrX[k] = &X.ReadWrite()[ndof*k];
      devPtrY[k] = &Y.ReadWrite()[ndof*k];
   }

   MFEM_cu_or_hip(blasStatus_t) stat;  // collects generation of cublasStatus_t
   MFEM_cu_or_hip(blasHandle_t)
   handle;  // tracks handle into API; can be specified futher but NULL works too

   stat = MFEM_cu_or_hip(blasCreate)(
             &handle);  // create handle to start CUBLAS work on the device; i.e. initialize CUBLAS
   if (stat != MFEM_CU_or_HIP(BLAS_STATUS_SUCCESS))
   {
      printf ("BLAS initialization failed\n");
      return EXIT_FAILURE;
   }

   stat = MFEM_cu_or_hip(blasDgemvBatched) (handle, MFEM_CU_or_HIP(BLAS_OP_N),
                                            ndof, ndof,
                                            &alpha, devPtrA.Read(), ndof, devPtrX.Read(), 1,
                                            &beta, devPtrY.ReadWrite(), 1,
                                            num_elem);  // version 11.9.0 needs batchCount = W as the last parameter
   if (stat != MFEM_CU_or_HIP(BLAS_STATUS_SUCCESS))
   {
      printf ("BLAS gemvBatched() failed\n");
      printf("\n");
      return EXIT_FAILURE;
   }

   MFEM_cu_or_hip(blasDestroy)(handle);  // end API stream


   // 5. Done! We can now output.
   printf ("Y tranpose is \n");
   for (int k = 0; k < num_elem; k++)
   {
      for (int j = 0; j < ndof; j++)
      {
         printf ("%9.3f", Y.HostRead()[IDXM(j,k,ndof)]);
      }
      printf ("\n");
   }
   printf ("\n");


   // 6. If wanted, solve the linear systems
   if (inverse)
   {
      Vector B(ndof*num_elem);
      DenseTensor Ainv(ndof,ndof,num_elem);
      BatchSolver batchSolver(BatchSolver::SolveMode::INVERSE);
      batchSolver.AssignMatrices(A);
      batchSolver.GetInverse(Ainv);
      batchSolver.Mult(X, B);

      printf ("A inverse transpose is \n");
      for (int k = 0; k < num_elem; k++)
      {
         for (int j = 0; j < ndof; j++)
         {
            for (int i = 0; i < ndof; i++)
            {
               printf ("%9.3f", Ainv.HostRead()[IDXT(i,j,k,ndof)]);
            }
            printf ("\n");
         }
         printf ("\n");
      }

      printf ("B tranpose is \n");
      for (int k = 0; k < num_elem; k++)
      {
         for (int j = 0; j < ndof; j++)
         {
            printf ("%9.3f", B.HostRead()[IDXM(j,k,
                                               ndof)]);  // col-major, so prints vectors of columns out together in each row
         }
         printf ("\n");
      }
      printf ("\n");
   }

   return EXIT_SUCCESS;


   // *****CUBLAS IMPLEMENTATIONS*****
   // cublasStatus_t stat;  // collects generation of cublasStatus_t
   // cublasHandle_t handle;  // tracks handle into API; can be specified futher but NULL works too
   // stat = cublasCreate(&handle);  // create handle to start CUBLAS work on the device; i.e. initialize CUBLAS
   // if (stat != CUBLAS_STATUS_SUCCESS) {
   //     printf ("CUBLAS initialization failed\n");
   //     return EXIT_FAILURE;
   // }
   // stat = cublasDgemvBatched (handle, CUBLAS_OP_N, M, N,
   //                             &alpha, devPtrA.READ(), M, devPtrX.Read(), 1,
   //                             &beta, devPtrY.ReadWrite(), 1, W);  // version 11.9.0 needs batchCount = W as the last parameter
   // if (stat != CUBLAS_STATUS_SUCCESS) {
   //     printf ("CUBLAS gemvBatched() failed\n");
   //     printf("%s", cublasGetStatusString(stat));
   //     printf("\n");
   //     return EXIT_FAILURE;
   // }
   // cublasDestroy(handle);  // end API stream


   // *****HIPBLAS IMPLEMENTATIONS*****
   // hipblasStatus_t stat;  // collects generation of hipblasStatus_t
   // hipblasHandle_t handle = nullptr;  // tracks handle into API; can be specified futher but NULL works too
   // stat = hipblasCreate(&handle);  // create handle to start HIPBLAS work on the device; i.e. initialize HIPBLAS
   // if (stat != HIPBLAS_STATUS_SUCCESS) {
   //     printf ("HIPBLAS initialization failed\n");
   //     return EXIT_FAILURE;
   // }
   // stat = hipblasDgemvBatched (handle, HIPBLAS_OP_N, M, N,
   //                             &alpha, devPtrA.Read(), M, devPtrX.Read(), 1,
   //                             &beta, devPtrY.ReadWrite(), 1, W);
   // if (stat != HIPBLAS_STATUS_SUCCESS) {
   //     printf ("HIPBLAS gemvBatched() failed\n");
   //     printf("%s", hipblasStatusToString(stat));
   //     printf("\n");
   //     return EXIT_FAILURE;
   // }
   // hipblasDestroy(handle);  // end API stream
}