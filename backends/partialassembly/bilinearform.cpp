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
void getSubvector(CudaVector<double>& subvec, const CudaVector<double>& X, const CudaArray& constraint_list)
{
   //TODO
}

void setSubvector(CudaVector<double>& X, const CudaVector<double>& subvec, const CudaArray& constraint_list)
{
   //TODO
}

void mapDofs(CudaVector<double>& w, const CudaVector<double>& x, const CudaArray& constraint_list)
{
   //TODO
}

void mapDofsClear(CudaVector<double>& w, const CudaArray& constraint_list)
{
   //TODO
}
#endif



} // namespace mfem::pa

} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_PA)