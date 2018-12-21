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

#include "../config/config.hpp"

#include "fem.hpp"
#include "kfespace.hpp"
#include "kernels/kGlobalToLocal.hpp"
#include "kernels/kLocalToGlobal.hpp"

// *****************************************************************************
namespace mfem
{

// **************************************************************************
static void kArrayAssign(const int n, const int *src, int *dest)
{
   GET_CONST_ADRS_T(src,int);
   GET_ADRS_T(dest,int);
   MFEM_FORALL(i, n, d_dest[i] = d_src[i];);
}

// *****************************************************************************
// * kFiniteElementSpace
// *****************************************************************************
kFiniteElementSpace::kFiniteElementSpace(FiniteElementSpace *f)
   :fes(f),
    globalDofs(f->GetNDofs()),
    localDofs(f->GetFE(0)->GetDof()),
    offsets(globalDofs+1),
    indices(localDofs, f->GetNE()),
    map(localDofs, f->GetNE())
{
   const FiniteElement *fe = f->GetFE(0);
   const TensorBasisElement* el = dynamic_cast<const TensorBasisElement*>(fe);
   MFEM_ASSERT(el, "Finite element not supported with partial assembly");

   const Array<int> &dof_map = el->GetDofMap();
   const bool dof_map_is_identity = (dof_map.Size()==0);

   const Table& e2dTable = f->GetElementToDofTable();
   const int* elementMap = e2dTable.GetJ();
   const int elements = f->GetNE();
   Array<int> h_offsets(globalDofs+1);

   // We'll be keeping a count of how many local nodes point to its global dof
   for (int i = 0; i <= globalDofs; ++i)
   {
      h_offsets[i] = 0;
   }
   for (int e = 0; e < elements; ++e)
   {
      for (int d = 0; d < localDofs; ++d)
      {
         const int gid = elementMap[localDofs*e + d];
         ++h_offsets[gid + 1];
      }
   }
   // Aggregate to find offsets for each global dof
   for (int i = 1; i <= globalDofs; ++i)
   {
      h_offsets[i] += h_offsets[i - 1];
   }

   Array<int> h_indices(localDofs*elements);
   Array<int> h_map(localDofs*elements);
   // For each global dof, fill in all local nodes that point   to it
   for (int e = 0; e < elements; ++e)
   {
      for (int d = 0; d < localDofs; ++d)
      {
         const int did = dof_map_is_identity?d:dof_map[d];
         const int gid = elementMap[localDofs*e + did];
         const int lid = localDofs*e + d;
         h_indices[h_offsets[gid]++] = lid;
         h_map[lid] = gid;
      }
   }

   // We shifted the offsets vector by 1 by using it as a counter
   // Now we shift it back.
   for (int i = globalDofs; i > 0; --i)
   {
      h_offsets[i] = h_offsets[i - 1];
   }
   h_offsets[0] = 0;

   const int leN = localDofs*elements;
   const int guN = globalDofs+1;
   kArrayAssign(guN,h_offsets,offsets);
   kArrayAssign(leN,h_indices,indices);
   kArrayAssign(leN,h_map,map);
}

// ***************************************************************************
kFiniteElementSpace::~kFiniteElementSpace()
{
   // ::delete reorderIndices;
}

// ***************************************************************************
void kFiniteElementSpace::GlobalToLocal(const Vector& globalVec,
                                        Vector& localVec) const
{
   const int vdim = fes->GetVDim();
   const int localEntries = localDofs * fes->GetNE();
   const bool vdim_ordering = fes->GetOrdering() == Ordering::byVDIM;
   kGlobalToLocal(vdim,
                  vdim_ordering,
                  globalDofs,
                  localEntries,
                  offsets,
                  indices,
                  globalVec,
                  localVec);
}

// ***************************************************************************
// Aggregate local node values to their respective global dofs
void kFiniteElementSpace::LocalToGlobal(const Vector& localVec,
                                        Vector& globalVec) const
{
   const int vdim = fes->GetVDim();
   const int localEntries = localDofs * fes->GetNE();
   const bool vdim_ordering = fes->GetOrdering() == Ordering::byVDIM;
   kLocalToGlobal(vdim,
                  vdim_ordering,
                  globalDofs,
                  localEntries,
                  offsets,
                  indices,
                  localVec,
                  globalVec);
}

} // mfem
