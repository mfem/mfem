// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#include "../../config/config.hpp"
#if defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_PA)

#include "bilinearform.hpp"

namespace mfem
{

namespace pa
{

void getSubvector(HostVector<double>& subvec, const HostVector<double>& X, const HostArray& constraint_list)
{
   const double *X_data = X.GetData();
   double *subvec_data = subvec.GetData();
   const int* constraint_data = constraint_list.template GetTypedData<int>();
   const std::size_t num_constraint = constraint_list.Size();
   for (std::size_t i = 0; i < num_constraint; i++) subvec_data[i] = X_data[constraint_data[i]];   
}

void setSubvector(HostVector<double>& X, const HostVector<double>& subvec, const HostArray& constraint_list)
{
   double *X_data = X.GetData();
   const double *subvec_data = subvec.GetData();
   const int* constraint_data = constraint_list.template GetTypedData<int>();
   const std::size_t num_constraint = constraint_list.Size();
   for (std::size_t i = 0; i < num_constraint; i++) X_data[constraint_data[i]] = subvec_data[i];   
}

void mapDofs(HostVector<double>& w, const HostVector<double>& x, const HostArray& constraint_list)
{
   const double *x_data = x.GetData();
   double *w_data = w.GetData();
   const int* constraint_data = constraint_list.template GetTypedData<int>();
   const std::size_t num_constraint = constraint_list.Size();
   for (std::size_t i = 0; i < num_constraint; i++)
      w_data[constraint_data[i]] = x_data[constraint_data[i]];
}

void mapDofsClear(HostVector<double>& w, const HostArray& constraint_list)
{
   double *w_data = w.GetData();
   const int* constraint_data = constraint_list.template GetTypedData<int>();
   const std::size_t num_constraint = constraint_list.Size();
   for (std::size_t i = 0; i < num_constraint; i++)
      w_data[constraint_data[i]] = 0.0;
}

#ifdef __NVCC__
__global__ void getSubvectorKernel(const int size, double* subvec, const double* X, const int* constraint_list)
{
   int idx = threadIdx.x + blockDim.x*blockIdx.x;
   if (idx>=size)
      return;
   subvec[idx] = X[constraint_list[idx]];
}

void getSubvector(CudaVector<double>& subvec, const CudaVector<double>& X, const CudaArray& constraint_list)
{
   const int bsize = 512;
   const int vecsize = subvec.Size();
   int gridsize = vecsize/bsize;
   if (bsize*gridsize < vecsize)
        gridsize += 1;
   getSubvectorKernel<<<gridsize,bsize>>>(vecsize, subvec.GetData(), X.GetData(), constraint_list.GetTypedData<int>());
}

__global__ void setSubvectorKernel(const int size, double* X, const double* subvec, const int* constraint_list)
{
   int idx = threadIdx.x + blockDim.x*blockIdx.x;
   if (idx>=size)
      return;
   X[constraint_list[idx]] = subvec[idx];
}

void setSubvector(CudaVector<double>& X, const CudaVector<double>& subvec, const CudaArray& constraint_list)
{
   const int bsize = 512;
   const int vecsize = X.Size();
   int gridsize = vecsize/bsize;
   if (bsize*gridsize < vecsize)
        gridsize += 1;
   setSubvectorKernel<<<gridsize,bsize>>>(vecsize, X.GetData(), subvec.GetData(), constraint_list.GetTypedData<int>());
}

__global__ void mapDofsKernel(const int size, double* w, const double* x, const int* constraint_list)
{
   int idx = threadIdx.x + blockDim.x*blockIdx.x;
   if (idx>=size)
      return;
   w[constraint_list[idx]] = x[constraint_list[idx]];
}

void mapDofs(CudaVector<double>& w, const CudaVector<double>& x, const CudaArray& constraint_list)
{
   const int bsize = 512;
   const int vecsize = w.Size();
   int gridsize = vecsize/bsize;
   if (bsize*gridsize < vecsize)
        gridsize += 1;
   mapDofsKernel<<<gridsize,bsize>>>(vecsize, w.GetData(), x.GetData(), constraint_list.GetTypedData<int>());
}

__global__ void mapDofsClearKernel(const int size, double* w, const int* constraint_list)
{
   int idx = threadIdx.x + blockDim.x*blockIdx.x;
   if (idx>=size)
      return;
   w[constraint_list[idx]] = 0.0;
}

void mapDofsClear(CudaVector<double>& w, const CudaArray& constraint_list)
{
   const int bsize = 512;
   const int vecsize = w.Size();
   int gridsize = vecsize/bsize;
   if (bsize*gridsize < vecsize)
        gridsize += 1;
   mapDofsClearKernel<<<gridsize,bsize>>>(vecsize, w.GetData(), constraint_list.GetTypedData<int>());
}
#endif


} // namespace mfem::pa

} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_PA)