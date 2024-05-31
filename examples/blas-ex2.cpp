#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#define M 6
#define N 6
#define W 2
#define IDX2C(i,j,k,ld) (((ld)*(ld)*(k))+((j)*(ld))+(i))

// Build a function that helps to modify <type> tensor m ON THE DEVICE with dimensions ldm x n by scaling certain values with alpha or beta 
// NOTE: cublas<t>scal scales the vector x by a scalar alpha and overwrites with the result (S float, D double, etc)
static __inline__ void modify (cublasHandle_t handle, float *m, int ldm, int n, int p, int q, float alpha, float beta){
    cublasSscal (handle, n-q, &alpha, &m[IDX2C(p,q,0,ldm)], ldm);  // numElems = n-q, scalar = alpha, start at devVec = m[p+ldm*q], inc = ldm
    cublasSscal (handle, ldm-p, &beta, &m[IDX2C(p,q,0,ldm)], 1);  // numElems = ldm-p, scalar = beta, start at devVec = m[p+ldm*q], inc = 1
}

int main (void){
    cudaError_t cudaStat;  // collects generation of cudaError_t
    cublasStatus_t stat;  // collects generation of cudaStatus_t 
    cublasHandle_t handle;  // tracks handle into API; can be specified futher but NULL works too 
    int i, j, k;
    float* devPtrA;  // device pointer for a 
    float* a = 0;  // host device pointer 
    a = (float *)malloc (M * N * W * sizeof (*a));  // malloc - give matrix allocation of MxNx1 to a on host
    if (!a) {
        printf ("host memory allocation failed");
        return EXIT_FAILURE;
    }
    for (k = 0; k < W; k++){
        for (j = 0; j < N; j++) {
            for (i = 0; i < M; i++) {
                a[IDX2C(i,j,k,M)] = (float)(IDX2C(i,j,k,M));  // fill up a with values i*N+j+1; e.g. a[0][0] = 0*N+0+1 = 1
                printf ("%7.0f", a[IDX2C(i,j,k,M)]);
            }
            printf ("\n");  // col-major, so prints vectors of columns out together in each row
        }
        printf ("\n");
    }

    cudaStat = cudaMalloc ((void**)&devPtrA, M*N*W*sizeof(*a));  // cudaMalloc - give matrix allocation of MxNx1 to device pointer on GPU
    if (cudaStat != cudaSuccess) {
        printf ("device memory allocation failed");
        free (a);
        return EXIT_FAILURE;
    }

    // ***START OF DEVICE COMPUTATIONS***
    // create handle to start CUBLAS work on the device; i.e. initialize CUBLAS
    stat = cublasCreate(&handle);  
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("CUBLAS initialization failed\n");
        free (a);
        cudaFree (devPtrA);
        return EXIT_FAILURE;
    }
    stat = cublasSetMatrix (M*N, W, sizeof(*a), a, M*N, devPtrA, M*N);  // fill in the device pointer matrix; 
    // arguments are (rows, cols, NE, elemSize, source_matrix, ld of source, destination matrix, ld dest)
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("data download failed");
        free (a);
        cudaFree (devPtrA);
        cublasDestroy(handle);
        return EXIT_FAILURE;
    }
    // handles the computations on the device 
    modify(handle, devPtrA, M, N, 1, 2, 16.0f, 12.0f); 
        // m[1 + 2*5] = m[13] = 8 * 16 = 128 --> happens N-2 = 3 times, with gap of M = 6 in between each  
        // m[1 + 2*5] = m[13] = 128 * 12 = 1536 --> happens M-1 = 5 times, with gap of 1 in between each 

    // copies device memory to host memory, for output from host 
    stat = cublasGetMatrix (M*N, W, sizeof(*a), devPtrA, M*N, a, M*N);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("data upload failed");
        free (a);
        cudaFree (devPtrA);
        cublasDestroy(handle);
        return EXIT_FAILURE;
    }
    // free up memory & end API connection 
    cudaFree (devPtrA);
    cublasDestroy(handle);
    // ***END OF DEVICE COMPUTATION***

    // with memory on host device, we can now output it
    for (k = 0; k < W; k++) {
        for (j = 0; j < N; j++) {
            for (i = 0; i < M; i++) {
                printf ("%7.0f", a[IDX2C(i,j,k,M)]);
            }
            printf ("\n");  // col-major, so prints vectors of columns out together in each row
        }
        printf ("\n");
    }
    free(a);
    return EXIT_SUCCESS;
}