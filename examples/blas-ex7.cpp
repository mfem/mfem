#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "mfem.hpp"
#include "../fem/bilinearform.cpp"
#include <fstream>
#include <iostream>
#define M 2
#define N 2   // need to have M = N in order to use definition of IDXT below
#define W 2
#define IDXT(i,j,k,ld) (((ld)*(ld)*(k))+((j)*(ld))+(i))
#define IDXM(i,j,ld) ((ld*j)+i)

using namespace mfem;

int main (void){
    cublasStatus_t stat;  // collects generation of cudaStatus_t
    cublasHandle_t handle;  // tracks handle into API; can be specified futher but NULL works too
    int i, j, k;

    // create handle to start CUBLAS work on the device; i.e. initialize CUBLAS
    stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("CUBLAS initialization failed\n");
        return EXIT_FAILURE;
    }

    DenseTensor A(M,N,W);
    Vector X(N*W);
    Vector Y(N*W);
    printf ("A is \n");
    for (k = 0; k < W; k++) {
        for (j = 0; j < N; j++) {
            X.GetData()[IDXM(j,k,N)] = IDXM(j,k,N);
            for (i = 0; i < M; i++) {
                if (i==j) {A.Data()[IDXT(i,j,k,M)] = 1;}
                else {A.Data()[IDXT(i,j,k,M)] = IDXT(j,i,k,M)+1;}
                printf ("%9.0f", A.Data()[IDXT(i,j,k,M)]);
            }
            printf ("\n");  // col-major, so prints vectors of columns out together in each row
        }
        printf ("\n");
    }
    printf ("X is \n");
    for (k = 0; k < W; k++) {
        for (j = 0; j < N; j++) {
            printf ("%9.0f", X.GetData()[IDXM(j,k,N)]);
        }
        printf ("\n");
    }
    printf ("\n");
    

    double* devPtrA[W];  // device pointer for a
    double* devPtrX[W];  // device pointer for host pointer x
    double* devPtrY[W];  // device pointer for host pointer y

    for (k = 0; k < W; k++) {
        devPtrA[k] = &A.ReadWrite()[M*M*k];
        devPtrX[k] = &X.ReadWrite()[N*k];
        devPtrY[k] = &Y.ReadWrite()[N*k];
    }

    // handles the computations on the device via batched function
    double alpha = 1.;
    double beta = 0.;
    stat = cublasDgemvBatched (handle, CUBLAS_OP_N, M, N,
                               &alpha, devPtrA, M, devPtrX, 1,
                               &beta, devPtrY, 1, W);  // version 11.9.0 needs batchCount = W as the last parameter


    // end API connection
    cublasDestroy(handle);

    // with memory on host device, we can now output it
    printf ("Y is \n");
    for (k = 0; k < W; k++) {
        for (j = 0; j < N; j++) {
            printf ("%9.0f", Y.GetData()[IDXM(j,k,N)]);  // col-major, so prints vectors of columns out together in each row
        }
        printf ("\n");
    }
    printf ("\n");

    Vector B(N*W);
    DenseTensor Ainv(M,N,W);
    BatchSolver batchSolver(BatchSolver::SolveMode::INVERSE);
    batchSolver.AssignMatrices(A);
    batchSolver.GetInverse(Ainv);
    batchSolver.Mult(X, B);

    printf ("A inverse is \n");
    for (k = 0; k < W; k++) {
        for (j = 0; j < N; j++) {
            for (i = 0; i < M; i++) {
                printf ("%9.3f", Ainv.Data()[IDXT(i,j,k,M)]);
            }
            printf ("\n");  // col-major, so prints vectors of columns out together in each row
        }
        printf ("\n");
    }

    printf ("B is \n");
    for (k = 0; k < W; k++) {
        for (j = 0; j < N; j++) {
            printf ("%9.3f", B.GetData()[IDXM(j,k,N)]);  // col-major, so prints vectors of columns out together in each row
        }
        printf ("\n");
    }
    printf ("\n");


    return EXIT_SUCCESS;
}
