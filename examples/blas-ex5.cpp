#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "mfem.hpp"
#include <fstream>
#include <iostream>
#define M 6
#define N 6   // need to have M = N in order to use definition of IDXT below
#define W 2
#define IDXT(i,j,k,ld) (((ld)*(ld)*(k))+((j)*(ld))+(i))
#define IDXM(i,j,ld) ((ld*j)+i)

using namespace mfem;

// // Build a function that helps to modify <type> tensor m ON THE DEVICE with dimensions ldm x n by scaling certain values with alpha or beta
// // NOTE: cublas<t>scal scales the vector x by a scalar alpha and overwrites with the result (S double, D double, etc)
// static __inline__ void modify (cublasHandle_t handle, double *m, int ldm, int n, int p, int q, double alpha, double beta){
//     cublasDscal (handle, n-q, &alpha, &m[IDX2C(p,q,0,ldm)], ldm);  // numElems = n-q, scalar = alpha, start at devVec = m[p+ldm*q], inc = ldm
//     cublasDscal (handle, ldm-p, &beta, &m[IDX2C(p,q,0,ldm)], 1);  // numElems = ldm-p, scalar = beta, start at devVec = m[p+ldm*q], inc = 1
// }

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

    double* devPtrA[W];  // device pointer for a
    double* devPtrX[W];  // device pointer for host pointer x
    double* devPtrY[W];  // device pointer for host pointer y

    double* y = 0;  // host device pointer

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
    for (k = 0; k < W; k++) {
        stat = cublasSetMatrix (M, N, sizeof(*A.Data()), A.Data()[M*N*k], M, devPtrA[k], M);  // fill in the device pointer matrix;
        // arguments are (rows, cols, NE, elemSize, source_matrix, ld of source, destination matrix, ld dest)
        if (stat != CUBLAS_STATUS_SUCCESS) {
            printf ("data download failed");
            cudaFree (devPtrA);
            cublasDestroy(handle);
            return EXIT_FAILURE;
        }
        stat = cublasSetMatrix (N, W, sizeof(*X.GetData()), X.GetData()[N*k], N, devPtrX[k], N);  // fill in the device pointer matrix;
        // arguments are (rows, cols, NE, elemSize, source_matrix, ld of source, destination matrix, ld dest)
        if (stat != CUBLAS_STATUS_SUCCESS) {
            printf ("data download failed");
            cudaFree (devPtrX);
            cublasDestroy(handle);
            return EXIT_FAILURE;
        }
    }

    // handles the computations on the device
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
    // free up memory & end API connection
    cudaFree (devPtrA);
    cudaFree (devPtrX);
    cudaFree (devPtrY);
    cublasDestroy(handle);
    // ***END OF DEVICE COMPUTATION***

    // with memory on host device, we can now output it
    Y.SetData(y);
    for (k = 0; k < W; k++) {
        for (j = 0; j < N; j++) {
            printf ("%7.0f", Y.GetData()[IDXM(j,k,N)]);  // col-major, so prints vectors of columns out together in each row
        }
        printf ("\n");
    }
    free(y);
    return EXIT_SUCCESS;
}