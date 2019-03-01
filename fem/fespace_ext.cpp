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
#include "../linalg/device.hpp"
#include "fem.hpp"
#include "fespace_ext.hpp"

// *****************************************************************************
namespace mfem
{

// *****************************************************************************
// * FiniteElementSpaceExtension
// *****************************************************************************
FiniteElementSpaceExtension::FiniteElementSpaceExtension(
   const FiniteElementSpace &f)
   :fes(f),
    ne(fes.GetNE()),
    vdim(fes.GetVDim()),
    byvdim(fes.GetOrdering() == Ordering::byVDIM),
    NDofs(fes.GetNDofs()),
    Dof(fes.GetFE(0)->GetDof()),
    neDofs(ne*Dof),
    offsets(NDofs+1),
    indices(neDofs)
{
   const FiniteElement *fe = fes.GetFE(0);
   const TensorBasisElement* el = dynamic_cast<const TensorBasisElement*>(fe);
   MFEM_ASSERT(el, "Finite element not supported with partial assembly");
   const Array<int> &dof_map = el->GetDofMap();
   const bool dof_map_is_identity = (dof_map.Size()==0);
   const Table& e2dTable = fes.GetElementToDofTable();
   const int* elementMap = e2dTable.GetJ();
   // We'll be keeping a count of how many local nodes point to its global dof
   for (int i = 0; i <= NDofs; ++i)
   {
      offsets[i] = 0;
   }
   for (int e = 0; e < ne; ++e)
   {
      for (int d = 0; d < Dof; ++d)
      {
         const int gid = elementMap[Dof*e + d];
         ++offsets[gid + 1];
      }
   }
   // Aggregate to find offsets for each global dof
   for (int i = 1; i <= NDofs; ++i)
   {
      offsets[i] += offsets[i - 1];
   }
   // For each global dof, fill in all local nodes that point   to it
   for (int e = 0; e < ne; ++e)
   {
      for (int d = 0; d < Dof; ++d)
      {
         const int did = dof_map_is_identity?d:dof_map[d];
         const int gid = elementMap[Dof*e + did];
         const int lid = Dof*e + d;
         indices[offsets[gid]++] = lid;
      }
   }
   // We shifted the offsets vector by 1 by using it as a counter
   // Now we shift it back.
   for (int i = NDofs; i > 0; --i)
   {
      offsets[i] = offsets[i - 1];
   }
   offsets[0] = 0;
}

// ***************************************************************************
void FiniteElementSpaceExtension::L2E(const Vector& lVec, Vector& eVec) const
{
   const int nd = NDofs;
   const int vd = vdim;
   const bool t = byvdim;
   const int ne = neDofs;
   const DeviceArray d_offsets(offsets, nd+1);
   const DeviceArray d_indices(indices, neDofs);
   const DeviceMatrix d_lVec(lVec, t?vd:nd, t?nd:vd);
   DeviceMatrix d_eVec(eVec, t?vd:ne, t?ne:vd);
   MFEM_FORALL(i, nd,
   {
      const int offset = d_offsets[i];
      const int nextOffset = d_offsets[i+1];
      for (int c = 0; c < vd; ++c)
      {
         const double dofValue = d_lVec(t?c:i,t?i:c);
         for (int j = offset; j < nextOffset; ++j)
         {
            const int idx_j = d_indices[j];
            d_eVec(t?c:idx_j,t?idx_j:c) = dofValue;
         }
      }
   });
}

// ***************************************************************************
void FiniteElementSpaceExtension::E2L(const Vector& eVec, Vector& lVec) const
{
   const int nd = NDofs;
   const int vd = vdim;
   const bool t = byvdim;
   const int ne = neDofs;
   const DeviceArray d_offsets(offsets, nd+1);
   const DeviceArray d_indices(indices, neDofs);
   const DeviceMatrix d_eVec(eVec, t?vd:ne, t?ne:vd);
   DeviceMatrix d_lVec(lVec, t?vd:nd, t?nd:vd);
   MFEM_FORALL(i, nd,
   {
      const int offset = d_offsets[i];
      const int nextOffset = d_offsets[i + 1];
      for (int c = 0; c < vd; ++c)
      {
         double dofValue = 0;
         for (int j = offset; j < nextOffset; ++j)
         {
            const int idx_j = d_indices[j];
            dofValue +=  d_eVec(t?c:idx_j,t?idx_j:c);
         }
         d_lVec(t?c:i,t?i:c) = dofValue;
      }
   });
}

} // mfem
