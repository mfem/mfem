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
#define W 1
#define IDX2C(i,j,k,ld) (((ld)*(ld)*(k))+((j)*(ld))+(i))
#define IDXV(i,j,ld) ((ld*j)+i)

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

    DenseMatrix A(M);
    Vector X(N);
    Vector Y(N);
    for (j = 0; j < N; j++) {
        X.GetData()[j] = j;
        for (i = 0; i < M; i++) {
            A.Data()[IDXV(i,j,6)] = IDXV(i,j,6);
        }
    }

    double* devPtrA;  // device pointer for a
    double* a = 0;  // host device pointer

    double* devPtrX;  // device pointer for host pointer x
    double* x = 0;  // host device pointer

    double* devPtrY;  // device pointer for host pointer y
    double* y = 0;  // host device pointer

    a = (double *)malloc (M * N * W * sizeof (*a));  // malloc - give matrix allocation of MxNx1 to a on host
    if (!a) {
        printf ("host memory allocation failed for a");
        return EXIT_FAILURE;
    }
    x = (double *)malloc (N * W * sizeof(*x));
    if (!x) {
        printf ("host memory allocation failed for x");
        return EXIT_FAILURE;
    }
    y = (double *)malloc (N * W * sizeof(*y));
    if (!y) {
        printf ("host memory allocation failed for y");
        return EXIT_FAILURE;
    }
    for (k = 0; k < W; k++) {
        for (j = 0; j < N; j++) {
            for (i = 0; i < M; i++) {
                a[IDX2C(i,j,k,M)] = (double)(A.Data()[IDXV(i,j,6)]);  // fill up a with values i*N+j+1; e.g. a[0][0] = 0*N+0+1 = 1
                printf ("%7.0f", a[IDX2C(i,j,k,M)]);
            }
            printf ("\n");  // col-major, so prints vectors of columns out together in each row
        }
        printf ("\n");
    }
    for (k = 0; k < W; k++) {
        for (j = 0; j < N; j++) {
            x[IDXV(j,k,N)] = (double)(X[IDXV(j,k,N)]);
            printf ("%7.0f", x[IDXV(j,k,N)]);
        }
        printf ("\n");
    }

    cudaStat = cudaMalloc ((void**)&devPtrA, M*N*W*sizeof(*a));  // cudaMalloc - give matrix allocation of MxNxW to device pointer on GPU
    if (cudaStat != cudaSuccess) {
        printf ("device memory allocation failed for a");
        free (a);
        return EXIT_FAILURE;
    }
    cudaStat = cudaMalloc ((void**)&devPtrX, N*W*sizeof(*x));  // cudaMalloc - give matrix allocation of NxW to device pointer on GPU
    if (cudaStat != cudaSuccess) {
        printf ("device memory allocation failed for x");
        free (x);
        return EXIT_FAILURE;
    }
    cudaStat = cudaMalloc ((void**)&devPtrY, N*W*sizeof(*y));  // cudaMalloc - give matrix allocation of NxW to device pointer on GPU
    if (cudaStat != cudaSuccess) {
        printf ("device memory allocation failed for y");
        free (y);
        return EXIT_FAILURE;
    }

    // ***START OF DEVICE COMPUTATIONS***
    // create handle to start CUBLAS work on the device; i.e. initialize CUBLAS
    stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("CUBLAS initialization failed\n");
        free (a);
        free (x);
        cudaFree (devPtrA);
        return EXIT_FAILURE;
    }
    stat = cublasSetMatrix (M, N, sizeof(*a), a, M, devPtrA, M);  // fill in the device pointer matrix;
    // arguments are (rows, cols, NE, elemSize, source_matrix, ld of source, destination matrix, ld dest)
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("data download failed");
        free (a);
        cudaFree (devPtrA);
        cublasDestroy(handle);
        return EXIT_FAILURE;
    }
    stat = cublasSetMatrix (N, W, sizeof(*x), x, N, devPtrX, N);  // fill in the device pointer matrix;
    // arguments are (rows, cols, NE, elemSize, source_matrix, ld of source, destination matrix, ld dest)
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("data download failed");
        free (x);
        cudaFree (devPtrX);
        cublasDestroy(handle);
        return EXIT_FAILURE;
    }

    // handles the computations on the device
    double alpha = 1;
    double beta = 0;
    stat = cublasSgemv (handle, CUBLAS_OP_N, M, N,
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
    Vector Y(y);
    
    for (k = 0; k < W; k++) {
        for (j = 0; j < N; j++) {
            printf ("%7.0f", y[IDXV(j,k,N)]);  // col-major, so prints vectors of columns out together in each row
        }
        printf ("\n");
    }
    free(a);
    free(x);
    free(y);
    return EXIT_SUCCESS;
}