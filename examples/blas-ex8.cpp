// #include <stdio.h>
// #include <stdlib.h>
// #include <math.h>
// #include <cuda_runtime.h>
// #include "cublas_v2.h"
#include "mfem.hpp"
#include "../fem/bilinearform.cpp"
// #include <fstream>
// #include <iostream>
#define M 2
#define N 2   // need to have M = N in order to use definition of IDXT below
#define W 1
#define IDXT(i,j,k,ld) (((ld)*(ld)*(k))+((j)*(ld))+(i))
#define IDXM(i,j,ld) ((ld*j)+i)

using namespace mfem;

int main (void){
    hipblasStatus_t stat;  // collects generation of cudaStatus_t
    hipblasHandle_t handle = nullptr;  // tracks handle into API; can be specified futher but NULL works too

    // create handle to start CUBLAS work on the device; i.e. initialize CUBLAS
    stat = hipblasCreate(&handle);
    if (stat != HIPBLAS_STATUS_SUCCESS) {
        printf ("HIPBLAS initialization failed\n");
        return EXIT_FAILURE;
    }
    // std::cout << handle;

    DenseTensor A(M,N,W);
    Vector X(N*W);
    Vector Y(N*W);
    printf ("A is \n");
    for (int k = 0; k < W; k++) {
        for (int j = 0; j < N; j++) {
            X.HostReadWrite()[IDXM(j,k,N)] = (double) (IDXM(j,k,N));
            for (int i = 0; i < M; i++) {
                if (i==j) {A.HostReadWrite()[IDXT(i,j,k,M)] = 1.0;}
                else {A.HostReadWrite()[IDXT(i,j,k,M)] = (double) (IDXT(j,i,k,M)+1);}
                printf ("%9.3f", A.HostRead()[IDXT(i,j,k,M)]);
            }
            printf ("\n");  // col-major, so prints vectors of columns out together in each row
        }
        printf ("\n");
    }
    printf ("X is \n");
    for (int k = 0; k < W; k++) {
        for (int j = 0; j < N; j++) {
            printf ("%9.3f", X.HostRead()[IDXM(j,k,N)]);
        }
        printf ("\n");
    }
    printf ("\n");
    

    Array<double *>devPtrA(W);  // device pointer for a
    Array<double *>devPtrX(W);  // device pointer for host pointer x
    Array<double *>devPtrY(W);  // device pointer for host pointer y

    for (int k = 0; k < W; k++) {
        devPtrA[k] = &const_cast<DenseTensor &>(A).ReadWrite()[M*N*k];
        devPtrX[k] = &const_cast<Vector &>(X).ReadWrite()[N*k];
        devPtrY[k] = &Y.ReadWrite()[N*k];
    }


    // handles the computations on the device via batched function
    const double alpha = 1.;
    const double beta = 0.;
    stat = hipblasDgemvBatched (&handle, HIPBLAS_OP_N, M, N,
                                &alpha, devPtrA.Read(), M, devPtrX.Read(), 1,
                                &beta, devPtrY.ReadWrite(), 1, W);  // version 11.9.0 needs batchCount = W as the last parameter
    if (stat != HIPBLAS_STATUS_SUCCESS) {
        printf ("HIPBLAS gemvBatched() failed\n");
        printf("%s", hipblasStatusToString(stat));
        printf("\n");
        return EXIT_FAILURE;
    }
    

    // end API connection
    // hipFree(devPtrA);
    // hipFree(devPtrX);
    // hipFree(devPtrY);
    hipblasDestroy(handle);

    // with memory on host device, we can now output it
    printf ("Y is \n");
    for (int k = 0; k < W; k++) {
        for (int j = 0; j < N; j++) {
            printf ("%9.3f", Y.HostRead()[IDXM(j,k,N)]);  // col-major, so prints vectors of columns out together in each row
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
    for (int k = 0; k < W; k++) {
        for (int j = 0; j < N; j++) {
            for (int i = 0; i < M; i++) {
                printf ("%9.3f", Ainv.HostRead()[IDXT(i,j,k,M)]);
            }
            printf ("\n");  // col-major, so prints vectors of columns out together in each row
        }
        printf ("\n");
    }

    printf ("B is \n");
    for (int k = 0; k < W; k++) {
        for (int j = 0; j < N; j++) {
            printf ("%9.3f", B.HostRead()[IDXM(j,k,N)]);  // col-major, so prints vectors of columns out together in each row
        }
        printf ("\n");
    }
    printf ("\n");


    return EXIT_SUCCESS;
}
