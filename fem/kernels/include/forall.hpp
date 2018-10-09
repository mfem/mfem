// Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
// reserved. See files LICENSE and NOTICE for details.
//
// This file is part of CEED, a collection of benchmarks, miniapps, software
// libraries and APIs for efficient high-order finite element and spectral
// element discretizations for exascale applications. For more information and
// source code availability see http://github.com/ceed.
//
// The CEED research is supported by the Exascale Computing Project 17-SC-20-SC,
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.
#ifndef MFEM_KERNELS_FORALL
#define MFEM_KERNELS_FORALL

// *****************************************************************************
#define ELEMENT_BATCH 10
#define M2_ELEMENT_BATCH 32
#define A2_ELEMENT_BATCH 1
#define A2_QUAD_BATCH 1

// *****************************************************************************
#ifdef __RAJA__ // *************************************************************
//#warning RAJA KERNELS, WITH NVCC
#define sync
#define share
#define kernel
const int CUDA_BLOCK_SIZE = 256;
#define cu_device __device__
#define cu_exec RAJA::cuda_exec<CUDA_BLOCK_SIZE>
#define cu_reduce RAJA::cuda_reduce<CUDA_BLOCK_SIZE>
#define sq_device __host__
#define sq_exec RAJA::seq_exec
#define sq_reduce RAJA::seq_reduce
#define ReduceDecl(type,var,ini) \
  RAJA::Reduce ## type<sq_reduce, RAJA::Real_type> var(ini);
#define ReduceForall(i,max,body) \
  RAJA::forall<sq_exec>(0,max,[=]sq_device(RAJA::Index_type i) {body});
#define forall(i,max,body)                                              \
   if (mfem::config::Get().Cuda())                                      \
    RAJA::forall<cu_exec>(0,max,[=]cu_device(RAJA::Index_type i) {body}); \
  else                                                                  \
    RAJA::forall<sq_exec>(0,max,[=]sq_device(RAJA::Index_type i) {body});
#define forallS(i,max,step,body) {assert(false);forall(i,max,body)}
#define call0(name,id,grid,blck,...) call[id](__VA_ARGS__)
#define cuKerGBS(name,grid,block,end,...) name ## 0(end,__VA_ARGS__)
#define cuKer(name,end,...) name ## 0(end,__VA_ARGS__)


// *****************************************************************************
#else // KERNELS on GPU, CUDA Kernel launches  *********************************
#ifdef __NVCC__
//#warning GPU KERNELS, WITH NVCC direct launch
#define kernel __global__
#define share __shared__
#define sync __syncthreads();
const int CUDA_BLOCK_SIZE = 256;
#define cuKer(name,end,...) name ## 0<<<((end+256-1)/256),256>>>(end,__VA_ARGS__)
#define cuLaunchKer(name,args) {                                      \
    cuLaunchKernel(name ## 0,                                         \
                   ((end+256-1)/256),1,1,                             \
                   256,1,1,                                           \
                   0,0,                                               \
                   args);                                             \
      }
#define cuKerGBS(name,grid,block,end,...) name ## 0<<<grid,block>>>(end,__VA_ARGS__)
#define call0p(name,id,grid,blck,...)                               \
  printf("\033[32;1m[call0] name=%s grid:%d, block:%d\033[m\n",#name,grid,blck); \
  call[id]<<<grid,blck>>>(__VA_ARGS__);\
  cudaDeviceSynchronize();
#define call0(name,id,grid,blck,...) {call[id]<<<grid,blck>>>(__VA_ARGS__);\
   cudaDeviceSynchronize();}
#define ReduceDecl(type,var,ini) double var=ini;
#define ReduceForall(i,max,body) 

// *****************************************************************************
#else // KERNELS on CPU ********************************************************
//#warning NO RAJA, NO NVCC
#define sync
#define share
#define kernel
class ReduceSum{
public:
  double s;
public:
  inline ReduceSum(double d):s(d){}
  inline operator double() { return s; }
  inline ReduceSum& operator +=(const double d) { return *this=(s+d); }
};
class ReduceMin{
public:
  double m;
public:
  inline ReduceMin(double d):m(d){}
  inline operator double() { return m; }
  inline ReduceMin& min(const double d) { return *this=(m<d)?m:d; }
};
#define ReduceDecl(type,var,ini) Reduce##type var(ini);
#define forall(i,max,body) for(int i=0;i<max;i++){body}
#define forallS(i,max,step,body) for(int i=0;i<max;i+=step){body}
#define ReduceForall(i,max,body) forall(i,max,body)
#define call0(name,id,grid,blck,...) call[id](__VA_ARGS__)
#define cuKer(name,...) name ## 0(__VA_ARGS__)
#define cuKerGBS(name,grid,block,end,...) name ## 0(end,__VA_ARGS__)
#endif //__NVCC__
#endif // __RAJA__
#endif // MFEM_KERNELS_FORALL
