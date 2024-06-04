#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"  // do we need additional headers? especially for gemvBatched
#include "mfem.hpp"
#include <fstream>
#include <iostream>
#define M 6
#define N 6   // need to have M = N in order to use definition of IDXT below
#define W 2
#define IDXT(i,j,k,ld) (((ld)*(ld)*(k))+((j)*(ld))+(i))
#define IDXM(i,j,ld) ((ld*j)+i)

using namespace mfem;

int main (void){
    cudaError_t cudaStat;  // collects generation of cudaError_t
    cublasStatus_t stat;  // collects generation of cudaStatus_t
    cublasHandle_t handle;  // tracks handle into API; can be specified futher but NULL works too
    int i, j, k;

    // batchlinalg uses DenseTensor Mat and Vector x & y 
    DenseTensor A(M,N,W);
    Vector X(N*W);
    Vector Y(N*W);

    for (k = 0; k < W; k++) {
        for (j = 0; j < N; j++) {
            for (i = 0; i < M; i++) {
                X.GetData()[IDXM(i,j,N)] = IDXM(i,j,N);
                A.Data()[IDXT(i,j,k,M)] = IDXT(i,j,k,M);
            }
        }
    }

    double* devPtrA[W];  // device array pointer (for W many matrices) for tensor A
    double* devPtrX[W];  // device array pointer (for W many vectors) for host pointer x
    double* devPtrY[W];  // device array pointer (for W many vectors) for host pointer y

    double* y = 0;  // host device pointer; used to update Y (there may be a better version than this -- talk to Arturo)

    y = (double *)malloc (N * W * sizeof(*Y.GetData()));
    if (!y) {
        printf ("host memory allocation failed for y");
        return EXIT_FAILURE;
    }
    for (k = 0; k < W; k++) {
        for (j = 0; j < N; j++) {
            for (i = 0; i < M; i++) {
                printf ("%7.0f", A.Data()[IDXT(i,j,k,M)]);
            }
            printf ("\n");  // col-major, so prints vectors of columns out together in each row
        }
        printf ("\n");
    }
    for (k = 0; k < W; k++) {
        for (j = 0; j < N; j++) {
            printf ("%7.0f", X[IDXM(j,k,N)]);
        }
        printf ("\n");
    }

    cudaStat = cudaMalloc ((void**)&devPtrA, M*N*W*sizeof(*A.Data()));  // cudaMalloc - give matrix allocation of MxNxW to device pointer on GPU
    if (cudaStat != cudaSuccess) {
        printf ("device memory allocation failed for A");
        return EXIT_FAILURE;
    }
    cudaStat = cudaMalloc ((void**)&devPtrX, N*W*sizeof(*X.GetData()));  // cudaMalloc - give matrix allocation of NxW to device pointer on GPU
    if (cudaStat != cudaSuccess) {
        printf ("device memory allocation failed for X");
        return EXIT_FAILURE;
    }
    cudaStat = cudaMalloc ((void**)&devPtrY, N*W*sizeof(*Y.GetData()));  // cudaMalloc - give matrix allocation of NxW to device pointer on GPU
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

    // (assuming this works...)
    for (k = 0; k < W; k++) {
        stat = cublasSetMatrix (M, N, sizeof(*A.Data()), &A.Data()[M*N*k], M, devPtrA[k], M);  // fill in the device pointer matrix;
        // arguments are (rows, cols, elemSize (bytesize), source_matrix, ld of source, dest_matrix, ld of dest)
        if (stat != CUBLAS_STATUS_SUCCESS) {
            printf ("data download failed");
            cudaFree (devPtrA);
            cublasDestroy(handle);
            return EXIT_FAILURE;
        }
        stat = cublasSetMatrix (N, W, sizeof(*X.GetData()), &X.GetData()[N*k], N, devPtrX[k], N);  // fill in the device pointer matrix;
        // arguments are (rows, cols, NE, elemSize, source_matrix, ld of source, destination matrix, ld dest)
        if (stat != CUBLAS_STATUS_SUCCESS) {
            printf ("data download failed");
            cudaFree (devPtrX);
            cublasDestroy(handle);
            return EXIT_FAILURE;
        }
    }

    double alpha = 1.;
    double beta = 0.;
    stat = cublasDgemvBatched (handle, CUBLAS_OP_N, M, N,
                               &alpha, devPtrA, M, devPtrX, 1,
                               &beta, devPtrY, 1);

    // copies device memory to host memory, for output from host
    stat = cublasGetMatrix (N, W, sizeof(*y), devPtrY, N, y, N);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("data upload failed");
        free (y);
        cudaFree (devPtrY);
        cublasDestroy(handle);
        return EXIT_FAILURE;
    }
    // free up memory & end CUBLAS stream
    cudaFree (devPtrA);
    cudaFree (devPtrX);
    cudaFree (devPtrY);
    cublasDestroy(handle);
    // ***END OF DEVICE COMPUTATION***

    // with memory on host device, we can now output it
    Y.SetData(y);
    for (k = 0; k < W; k++) {
        for (j = 0; j < N; j++) {
            printf ("%7.0f", Y.GetData()[IDXM(j,k,N)]);  
        }
        printf ("\n");
    }
    free(y);
    return EXIT_SUCCESS;
}