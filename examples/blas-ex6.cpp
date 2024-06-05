#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "mfem.hpp"
#include "../linalg/batchlinalg.hpp"
#include "../general/backends.hpp"
#include "../general/forall.hpp"
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

    DenseTensor A(M,N,W);
    Vector X(N*W);
    Vector Y(N*W);
    printf ("A is \n");
    for (k = 0; k < W; k++) {
        for (j = 0; j < N; j++) {
            X.GetData()[IDXM(j,k,N)] = IDXM(j,k,N);
            for (i = 0; i < M; i++) {
                A.Data()[IDXT(i,j,k,M)] = IDXT(i,j,k,M);
                printf ("%7.0f", A.Data()[IDXT(i,j,k,M)]);
            }
            printf ("\n");  // col-major, so prints vectors of columns out together in each row
        }
        printf ("\n");
    }
    printf ("X is \n");
    for (k = 0; k < W; k++) {
        for (j = 0; j < N; j++) {
            printf ("%7.0f", X.GetData()[IDXM(j,k,N)]);
        }
        printf ("\n");
    }
    printf ("\n");
    

    double* devPtrA[W];  // device pointer for a
    double* devPtrX[W];  // device pointer for host pointer x
    double* devPtrY[W];  // device pointer for host pointer y

    // double* y = 0;  // host device pointer
    // y = (double *)malloc (N * W * sizeof(*Y.GetData()));
    // if (!y) {
    //     printf ("host memory allocation failed for y");
    //     return EXIT_FAILURE;
    // }


    // ***START OF DEVICE COMPUTATIONS***
    // create handle to start CUBLAS work on the device; i.e. initialize CUBLAS
    stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("CUBLAS initialization failed\n");
        cudaFree (devPtrA);
        return EXIT_FAILURE;
    }

    for (k = 0; k < W; k++) {
        cudaStat = cudaMalloc ((void**)&devPtrA[k], M*N*sizeof(*A.Data()));  // cudaMalloc - give matrix allocation of MxNxW to device pointer on GPU
        if (cudaStat != cudaSuccess) {
            printf ("device memory allocation failed for A");
            return EXIT_FAILURE;
        }
        cudaStat = cudaMalloc ((void**)&devPtrX[k], N*sizeof(*X.GetData()));  // cudaMalloc - give matrix allocation of NxW to device pointer on GPU
        if (cudaStat != cudaSuccess) {
            printf ("device memory allocation failed for X");
            return EXIT_FAILURE;
        }
        cudaStat = cudaMalloc ((void**)&devPtrY[k], N*sizeof(*Y.GetData()));  // cudaMalloc - give matrix allocation of NxW to device pointer on GPU
        if (cudaStat != cudaSuccess) {
            printf ("device memory allocation failed for Y");
            return EXIT_FAILURE;
        }

        stat = cublasSetMatrix (M, N, sizeof(*A.Data()), &A.Data()[IDXT(0,0,k,M)], M, devPtrA[k], M);  // fill in the device pointer matrix;
        // arguments are (rows, cols, NE, elemSize, source_matrix, ld of source, destination matrix, ld dest)
        // seems to be correct based off https://stackoverflow.com/questions/16090351/in-cublas-how-do-i-get-or-set-matrix-element-from-host
        if (stat != CUBLAS_STATUS_SUCCESS) {
            printf ("data download failed");
            printf (cublasGetStatusString(stat));  // note: only available after version 11.4.2
            cudaFree (devPtrA);
            cublasDestroy(handle);
            return EXIT_FAILURE;
        }
        stat = cublasSetMatrix (N, 1, sizeof(*X.GetData()), &X.GetData()[IDXM(0,k,N)], N, devPtrX[k], N);  // fill in the device pointer matrix;
        // arguments are (rows, cols, NE, elemSize, source_matrix, ld of source, destination matrix, ld dest)
        if (stat != CUBLAS_STATUS_SUCCESS) {
            printf ("data download failed");
            cudaFree (devPtrX);
            cublasDestroy(handle);
            return EXIT_FAILURE;
        }
    }

    // handles the computations on the device via batched function
    double alpha = 1.;
    double beta = 0.;
    stat = cublasDgemvBatched (handle, CUBLAS_OP_N, M, N,
                               &alpha, devPtrA, M, devPtrX, 1,
                               &beta, devPtrY, 1, W);  // version 11.7.0 needs batchCount = W as the last parameter

    // copies device memory to host memory, for output from host
    for (k = 0; k < W; k++) {
        stat = cublasGetMatrix (N, 1, sizeof(*Y.GetData()), devPtrY[k], N, &Y.GetData()[IDXM(0,k,N)], N);
        if (stat != CUBLAS_STATUS_SUCCESS) {
            printf ("data upload failed");
            cudaFree (devPtrY);
            cublasDestroy(handle);
            return EXIT_FAILURE;
        }
    }
    // free up memory & end API connection
    cudaFree (devPtrA);
    cudaFree (devPtrX);
    cudaFree (devPtrY);
    cublasDestroy(handle);
    // ***END OF DEVICE COMPUTATION***

    // with memory on host device, we can now output it
    // Y.SetData(y);
    printf ("Y is \n");
    for (k = 0; k < W; k++) {
        for (j = 0; j < N; j++) {
            printf ("%7.0f", Y.GetData()[IDXM(j,k,N)]);  // col-major, so prints vectors of columns out together in each row
        }
        printf ("\n");
    }
    printf ("\n");

    DenseTensor const& const_A = A;
    Vector const& const_X = X;
    Vector Z(N*W);
    BatchSolver batchSolver(const_A, BatchSolver::SolveMode::INVERSE);
    batchSolver.ApplyInverse(const_X, Z);

    printf ("Z is \n");
    for (k = 0; k < W; k++) {
        for (j = 0; j < N; j++) {
            printf ("%7.0f", Z.GetData()[IDXM(j,k,N)]);  // col-major, so prints vectors of columns out together in each row
        }
        printf ("\n");
    }
    printf ("\n");


    return EXIT_SUCCESS;
}