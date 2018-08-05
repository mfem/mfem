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
#if defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_OCCA)

#include "backend.hpp"
#include "fespace.hpp"
#include "interpolation.hpp"

namespace mfem
{

namespace occa
{

FiniteElementSpace::FiniteElementSpace(const Engine &e,
                                       mfem::FiniteElementSpace &fespace)
   : PFiniteElementSpace(e, fespace),
     e_layout(e, 0) // resized in SetupLocalGlobalMaps()
{
   vdim     = fespace.GetVDim();
   ordering = fespace.GetOrdering();

   SetupLocalGlobalMaps();
   SetupOperators();
   SetupKernels();

   e_layout.DontDelete();
}

FiniteElementSpace::~FiniteElementSpace()
{
   delete [] elementDofMap;
   delete [] elementDofMapInverse;
   delete restrictionOp;
   delete prolongationOp;
}

void FiniteElementSpace::SetupLocalGlobalMaps()
{
   const mfem::FiniteElement &fe = *(fes->GetFE(0));
   const mfem::TensorBasisElement *el =
      dynamic_cast<const mfem::TensorBasisElement*>(&fe);

   const mfem::Table &e2dTable = fes->GetElementToDofTable();
   const int *elementMap = e2dTable.GetJ();
   const int elements = fes->GetNE();

   globalDofs = fes->GetNDofs();
   localDofs  = fe.GetDof();

   e_layout.Resize(localDofs * elements * fes->GetVDim());

   elementDofMap        = new int[localDofs];
   elementDofMapInverse = new int[localDofs];
   if (el)
   {
      ::memcpy(elementDofMap,
               el->GetDofMap().GetData(),
               localDofs * sizeof(int));
   }
   else
   {
      for (int i = 0; i < localDofs; ++i)
      {
         elementDofMap[i] = i;
      }
   }
   for (int i = 0; i < localDofs; ++i)
   {
      elementDofMapInverse[elementDofMap[i]] = i;
   }

   // Allocate device offsets and indices
   globalToLocalOffsets.allocate(GetDevice(),
                                 globalDofs + 1);
   globalToLocalIndices.allocate(GetDevice(),
                                 localDofs, elements);
   localToGlobalMap.allocate(GetDevice(),
                             localDofs, elements);

   int *offsets = globalToLocalOffsets.ptr();
   int *indices = globalToLocalIndices.ptr();
   int *l2gMap  = localToGlobalMap.ptr();

   // We'll be keeping a count of how many local nodes point
   //   to its global dof
   for (int i = 0; i <= globalDofs; ++i)
   {
      offsets[i] = 0;
   }

   for (int e = 0; e < elements; ++e)
   {
      for (int d = 0; d < localDofs; ++d)
      {
         const int gid = elementMap[localDofs*e + d];
         ++offsets[gid + 1];
      }
   }
   // Aggregate to find offsets for each global dof
   for (int i = 1; i <= globalDofs; ++i)
   {
      offsets[i] += offsets[i - 1];
   }
   // For each global dof, fill in all local nodes that point
   //   to it
   for (int e = 0; e < elements; ++e)
   {
      for (int d = 0; d < localDofs; ++d)
      {
         const int gid = elementMap[localDofs*e + elementDofMap[d]];
         const int lid = localDofs*e + d;
         indices[offsets[gid]++] = lid;
         l2gMap[lid] = gid;
      }
   }
   // We shifted the offsets vector by 1 by using it
   //   as a counter. Now we shift it back.
   for (int i = globalDofs; i > 0; --i)
   {
      offsets[i] = offsets[i - 1];
   }
   offsets[0] = 0;

   globalToLocalOffsets.keepInDevice();
   globalToLocalIndices.keepInDevice();
   localToGlobalMap.keepInDevice();
}

void FiniteElementSpace::SetupOperators()
{
   const mfem::SparseMatrix *R = fes->GetRestrictionMatrix();
   const mfem::Operator *P = fes->GetProlongationMatrix();
   CreateRPOperators(OccaVLayout(), OccaTrueVLayout(),
                     R, P,
                     restrictionOp,
                     prolongationOp);
}

void FiniteElementSpace::SetupKernels()
{
   ::occa::properties props("defines: {"
                            "  TILESIZE: 256,"
                            "}");
   props["defines/NUM_VDIM"] = vdim;

   props["defines/ORDERING_BY_NODES"] = 0;
   props["defines/ORDERING_BY_VDIM"]  = 1;
   props["defines/VDIM_ORDERING"] = (int) (ordering == Ordering::byVDIM);

   ::occa::device device = GetDevice();
   const std::string &okl_path = OccaEngine().GetOklPath();
   globalToLocalKernel = device.buildKernel(okl_path + "fespace.okl",
                                            "GlobalToLocal",
                                            props);
   localToGlobalKernel = device.buildKernel(okl_path + "fespace.okl",
                                            "LocalToGlobal",
                                            props);
}

} // namespace mfem::occa

} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_OCCA)
