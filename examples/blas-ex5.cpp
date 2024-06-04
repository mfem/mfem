#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "mfem.hpp"
#include <fstream>
#include <iostream>
#define M 6
#define N 6   // need to have M = N in order to use definition of IDX2C below                                                                                                                                                                 
#define W 2
#define IDX2C(i,j,k,ld) (((ld)*(ld)*(k))+((j)*(ld))+(i))
#define IDXV(i,j,ld) ((ld*j)+i)

using namespace mfem;

int main (void){
  cudaError_t cudaStat;  // collects generation of cudaError_t                                                                                                                                                                                
  cublasStatus_t stat;  // collects generation of cudaStatus_t                                                                                                                                                                                
  cublasHandle_t handle;  // tracks handle into API; can be specified futher but NULL works too                                                                                                                                               
  int i, j, k;

  DenseMatrix A(M);
  Vector X(N);
  Vector Y(N);
  for (j = 0; j < N; j++) {
    X.GetData()[j] = j;
    for (i = 0; i < M; i++) {
      A.Data()[IDXV(i,j,6)] = IDXV(i,j,6);
    }
  }

  double* devPtrA[W];  // device pointer for a                                                                                                                                                                                                

  double* devPtrX[W];  // device pointer for host pointer x                                                                                                                                                                                   

  double* devPtrY[W];  // device pointer for host pointer y                                                                                                                                                                                   
  double* y = 0;  // host device pointer                                                                                                                                                                                                      

  y = (double *)malloc (N * 1 * sizeof(*Y.GetData()));
  if (!y) {
    printf ("host memory allocation failed for y");
    return EXIT_FAILURE;
  }
  for (k = 0; k < 1; k++) {
    for (j = 0; j < N; j++) {
      for (i = 0; i < M; i++) {
        printf ("%7.0f", A.Data()[IDXV(i,j,6)]);
      }
      printf ("\n");  // col-major, so prints vectors of columns out together in each row                                                                                                                                                     
    }
    printf ("\n");
  }
  for (k = 0; k < 1; k++) {
    for (j = 0; j < N; j++) {
      printf ("%7.0f", X.GetData()[IDXV(j,k,N)]);
    }
    printf ("\n");
  }

  cudaStat = cudaMalloc ((void**)&devPtrA[0], M*N*sizeof(*A.Data()));  // cudaMalloc - give matrix allocation of MxNxW to device pointer on GPU                                                                                               
  if (cudaStat != cudaSuccess) {
    printf ("device memory allocation failed for A");
    return EXIT_FAILURE;
  }
  cudaStat = cudaMalloc ((void**)&devPtrX[0], N*sizeof(*X.GetData()));  // cudaMalloc - give matrix allocation of NxW to device pointer on GPU                                                                                                
  if (cudaStat != cudaSuccess) {
    printf ("device memory allocation failed for X");
    return EXIT_FAILURE;
  }
  cudaStat = cudaMalloc ((void**)&devPtrY[0], N*sizeof(*Y.GetData()));  // cudaMalloc - give matrix allocation of NxW to device pointer on GPU                                                                                                
  if (cudaStat != cudaSuccess) {
    printf ("device memory allocation failed for Y");
    return EXIT_FAILURE;
  }

  // ***START OF DEVICE COMPUTATIONS***                                                                                                                                                                                                       
  // create handle to start CUBLAS work on the device; i.e. initialize CUBLAS                                                                                                                                                                 
  stat = cublasCreate(&handle);
  if (stat != CUBLAS_STATUS_SUCCESS) {
    printf ("CUBLAS initialization failed\n");
    cudaFree (devPtrA);
    return EXIT_FAILURE;
  }
  stat = cublasSetMatrix (M, N, sizeof(*A.Data()), A.Data(), M, devPtrA[0], M);  // fill in the device pointer matrix;                                                                                                                        
  // arguments are (rows, cols, NE, elemSize, source_matrix, ld of source, destination matrix, ld dest)                                                                                                                                       
  if (stat != CUBLAS_STATUS_SUCCESS) {
    printf ("data download failed");
    cudaFree (devPtrA);
    cublasDestroy(handle);
    return EXIT_FAILURE;
  }
                                                                                                                                                                                          
  stat = cublasSetMatrix (N, 1, sizeof(*X.GetData()), X.GetData(), N, devPtrX[0], N);  // fill in the device pointer matrix;                                                                                                                  
  // arguments are (rows, cols, NE, elemSize, source_matrix, ld of source, destination matrix, ld dest)                                                                                                                                       
  if (stat != CUBLAS_STATUS_SUCCESS) {
    printf ("data download failed");
    cudaFree (devPtrX);
    cublasDestroy(handle);
    return EXIT_FAILURE;
  }

  // handles the computations on the device                                                                                                                                                                                                   
  double alpha = 1.;
  double beta = 0.;
  stat = cublasDgemv (handle, CUBLAS_OP_N, M, N,
                      &alpha, devPtrA[0], M, devPtrX[0], 1,
                      &beta, devPtrY[0], 1);

  // copies device memory to host memory, for output from host                                                                                                                                                                                
  stat = cublasGetMatrix (N, 1, sizeof(*y), devPtrY[0], N, y, N);
  if (stat != CUBLAS_STATUS_SUCCESS) {
    printf ("data upload failed");
    free (y);
    cudaFree (devPtrY);
    cublasDestroy(handle);
    return EXIT_FAILURE;
  }
  printf("Get to here \n");
  // free up memory & end API connection                                                                                                                                                                                                      
  cudaFree (devPtrA);
  cudaFree (devPtrX);
  cudaFree (devPtrY);
  cublasDestroy(handle);
  // ***END OF DEVICE COMPUTATION***                                                                                                                                                                                                          

  // with memory on host device, we can now output it                                                                                                                                                                                         
  Y.SetData(y);
  for (k = 0; k < 1; k++) {
    for (j = 0; j < N; j++) {
      printf ("%7.0f", Y.GetData()[IDXV(j,k,N)]);  // col-major, so prints vectors of columns out together in each row                                                                                                                        
    }
    printf ("\n");
  }
  free(y);
  return EXIT_SUCCESS;
}