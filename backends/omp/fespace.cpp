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
#if defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_OMP)

#include "fespace.hpp"

namespace mfem
{

namespace omp
{

FiniteElementSpace::FiniteElementSpace(const Engine &e,
                                       mfem::FiniteElementSpace &fespace)
   : PFiniteElementSpace(e, fespace),
     e_layout(e, 0),
     tensor_offsets(),
     tensor_indices()
{
   std::size_t lsize = 0;
   for (int e = 0; e < fespace.GetNE(); e++) { lsize += fespace.GetFE(e)->GetDof(); }
   e_layout.Resize(lsize);
}

static void BuildDofMaps(FiniteElementSpace *fes,
                         mfem::Array<int> &offsets,
                         mfem::Array<int> &indices)
{
   mfem::FiniteElementSpace *mfem_fes = fes->GetFESpace();

   const int local_size = fes->GetELayout().Size();
   const int global_size = mfem_fes->GetVLayout()->Size();
   const int vdim = mfem_fes->GetVDim();

   // Now we can allocate and fill the global map
   offsets.Resize(global_size + 1);
   indices.Resize(local_size);

   mfem::Array<int> global_map(local_size);
   mfem::Array<int> elem_vdof;

   int offset = 0;
   for (int e = 0; e < mfem_fes->GetNE(); e++)
   {
      const FiniteElement *fe = mfem_fes->GetFE(e);
      const int dofs = fe->GetDof();
      const TensorBasisElement *tfe = dynamic_cast<const TensorBasisElement *>(fe);
      const mfem::Array<int> &dof_map = tfe->GetDofMap();

      mfem_fes->GetElementVDofs(e, elem_vdof);

      for (int vd = 0; vd < vdim; vd++)
         for (int i = 0; i < dofs; i++)
         {
            global_map[offset + dofs*vd + i] = elem_vdof[dofs*vd + dof_map[i]];
         }
      offset += dofs * vdim;
   }

   // global_map[i] = index in global vector for local dof i
   // NOTE: multiple i values will yield same global_map[i] for shared DOF.

   // We want to now invert this map so we have indices[j] = (local dof for global dof j).

   // Zero the offset vector
   offsets = 0;

   // Keep track of how many local dof point to its global dof
   // Count how many times each dof gets hit
   for (int i = 0; i < local_size; i++)
   {
      const int g = global_map[i];
      ++offsets[g + 1];
   }
   // Aggregate the offsets
   for (int i = 1; i <= global_size; i++)
   {
      offsets[i] += offsets[i - 1];
   }

   for (int i = 0; i < local_size; i++)
   {
      const int g = global_map[i];
      indices[offsets[g]++] = i;
   }

   // Shift the offset vector back by one, since it was used as a
   // counter above.
   for (int i = global_size; i > 0; i--)
   {
      offsets[i] = offsets[i - 1];
   }
   offsets[0] = 0;

   offsets.Push();
   indices.Push();
}

/// Convert an E vector to L vector
void FiniteElementSpace::ToLVector(const Vector &e_vector, Vector &l_vector)
{
   if (tensor_indices.Size() == 0)
   {
      BuildDofMaps(this, tensor_offsets, tensor_indices);
   }

   if (l_vector.Size() != (std::size_t) GetFESpace()->GetVSize())
   {
      l_vector.Resize<double>(GetFESpace()->GetVLayout(), NULL);
   }

   const int size = l_vector.Size();
   const int *offsets = tensor_offsets.GetData();
   const int *indices = tensor_indices.GetData();

   const double *e_data = e_vector.GetData();
   double *l_data = l_vector.GetData();

#pragma omp target teams distribute parallel for        \
   map(to: offsets, indices, l_data, e_data)            \
   if (target: l_vector.ComputeOnDevice())              \
   if (parallel: l_vector.Size() > 1000)
   for (int i = 0; i < size; i++)
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

/// Covert an L vector to E vector
void FiniteElementSpace::ToEVector(const Vector &l_vector, Vector &e_vector)
{
   if (tensor_indices.Size() == 0)
   {
      BuildDofMaps(this, tensor_offsets, tensor_indices);
      std::cout << "Building maps!" << std::endl;
   }

   if (e_vector.Size() != (std::size_t) e_layout.Size())
   {
      e_vector.Resize<double>(GetELayout(), NULL);
   }

   const int lsize = l_vector.Size();
   const int *offsets = tensor_offsets.GetData();
   const int *indices = tensor_indices.GetData();

   const double *l_data = l_vector.GetData();
   double *e_data = e_vector.GetData();

#pragma omp target teams distribute parallel for        \
   map(to: offsets, indices, l_data, e_data)            \
   if (target: l_vector.ComputeOnDevice())              \
   if (parallel: l_vector.Size() > 1000)
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


} // namespace mfem::omp

} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_OMP)
