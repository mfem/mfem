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
MFEM_NAMESPACE

// **************************************************************************
static void kArrayAssign(const int n, const int *src, int *dest)
{
   GET_CONST_ADRS_T(src,int);
   GET_ADRS_T(dest,int);
   forall(i, n, d_dest[i] = d_src[i];);
}

// *****************************************************************************
// * kFiniteElementSpace
// *****************************************************************************
kFiniteElementSpace::kFiniteElementSpace(FiniteElementSpace &fes)
   :FiniteElementSpace(fes),
    globalDofs(GetNDofs()),
    localDofs(GetFE(0)->GetDof()),
    offsets(globalDofs+1),
    indices(localDofs, GetNE()),
    map(localDofs, GetNE())
{
   const FiniteElement *fe = GetFE(0);
   const TensorBasisElement* el = dynamic_cast<const TensorBasisElement*>(fe);
   const Array<int> &dof_map = el->GetDofMap();
   const bool dof_map_is_identity = (dof_map.Size()==0);

   const Table& e2dTable = GetElementToDofTable();
   const int* elementMap = e2dTable.GetJ();
   const int elements = GetNE();
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

   // Pushing because the kernels above are still on CPU
   mm::Get().Push(h_offsets);
   mm::Get().Push(h_indices);
   mm::Get().Push(h_map);

   const int leN = localDofs*elements;
   const int guN = globalDofs+1;

   //offsets = h_offsets;
   kArrayAssign(guN,h_offsets,offsets);

   //indices = h_indices;
   kArrayAssign(leN,h_indices,indices);

   //map = h_map;
   kArrayAssign(leN,h_map,map);
}

// ***************************************************************************
kFiniteElementSpace::~kFiniteElementSpace()
{
   ::delete reorderIndices;
}

// ***************************************************************************
void kFiniteElementSpace::GlobalToLocal(const Vector& globalVec,
                                        Vector& localVec) const
{
   const int vdim = GetVDim();
   const int localEntries = localDofs * GetNE();
   const bool vdim_ordering = ordering == Ordering::byVDIM;
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
   const int vdim = GetVDim();
   const int localEntries = localDofs * GetNE();
   const bool vdim_ordering = ordering == Ordering::byVDIM;
   kLocalToGlobal(vdim,
                  vdim_ordering,
                  globalDofs,
                  localEntries,
                  offsets,
                  indices,
                  localVec,
                  globalVec);
}

} // namespace mfem
