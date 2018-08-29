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

#include "fespace.hpp"

namespace mfem
{

namespace pa
{

void toLVector(const mfem::Array<int>& tensor_offsets, const mfem::Array<int>& tensor_indices
               , const HostVector<double>& e_vector, HostVector<double>& l_vector)
{
   const int lsize = l_vector.Size();
   const int *offsets = tensor_offsets.Get_PArray()->As<HostArray>().GetTypedData<int>();
   const int *indices = tensor_indices.Get_PArray()->As<HostArray>().GetTypedData<int>();

   const double *e_data = e_vector.GetData();
   double *l_data = l_vector.GetData();

   for (int i = 0; i < lsize; i++)
   {
      const int offset = offsets[i];
      const int next_offset = offsets[i + 1];
      double dof_value = 0;
      for (int j = offset; j < next_offset; j++)
      {
         dof_value += e_data[indices[j]];
      }
      l_data[i] = dof_value;
   }
}

void toEVector(const mfem::Array<int>& tensor_offsets, const mfem::Array<int>& tensor_indices
               , const HostVector<double>& l_vector, HostVector<double>& e_vector)
{
   const int lsize = l_vector.Size();
   const int *offsets = tensor_offsets.Get_PArray()->As<HostArray>().GetTypedData<int>();
   const int *indices = tensor_indices.Get_PArray()->As<HostArray>().GetTypedData<int>();

   const double *l_data = l_vector.GetData();
   double *e_data = e_vector.GetData();

   for (int i = 0; i < lsize; i++)
   {
      const int offset = offsets[i];
      const int next_offset = offsets[i + 1];
      const double dof_value = l_data[i];
      for (int j = offset; j < next_offset; j++)
      {
         e_data[indices[j]] = dof_value;
      }
   }
}

#ifdef __NVCC__
void toLVector(const mfem::Array<int>& tensor_offsets, const mfem::Array<int>& tensor_indices
               , const CudaVector<double>& e_vector, CudaVector<double>& l_vector)
{
   //TODO
}

void toEVector(const mfem::Array<int>& tensor_offsets, const mfem::Array<int>& tensor_indices
               , const CudaVector<double>& l_vector, CudaVector<double>& e_vector)
{
   //TODO
}
#endif

} // namespace mfem::pa

} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_PA)
