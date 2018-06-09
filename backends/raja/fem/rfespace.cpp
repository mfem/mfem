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

#include "../../../config/config.hpp"
#if defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_RAJA)

#include "../raja.hpp"

namespace mfem
{

namespace raja
{


// **************************************************************************
static void CreateRPOperators(Layout &v_layout,
                              Layout &t_layout,
                              const mfem::SparseMatrix *R,
                              const mfem::Operator *P,
                              mfem::Operator *&RajaR,
                              mfem::Operator *&RajaP)
{
   if (!P)
   {
      RajaR = new IdentityOperator(t_layout);
      RajaP = new IdentityOperator(t_layout);
      return;
   }

   const mfem::SparseMatrix *pmat = dynamic_cast<const mfem::SparseMatrix*>(P);
   raja::device device = v_layout.RajaEngine().GetDevice();

   if (R)
   {
      RajaSparseMatrix *rajaR =
         CreateMappedSparseMatrix(v_layout, t_layout, *R);
      raja::array<int> reorderIndices = rajaR->reorderIndices;
      delete rajaR;
      RajaR = new RestrictionOperator(v_layout, t_layout, reorderIndices);
   }

   if (pmat)
   {
      const mfem::SparseMatrix *pmatT = Transpose(*pmat);

      RajaSparseMatrix *rajaP  =
         CreateMappedSparseMatrix(t_layout, v_layout, *pmat);
      RajaSparseMatrix *rajaPT =
         CreateMappedSparseMatrix(v_layout, t_layout, *pmatT);

      RajaP = new ProlongationOperator(*rajaP, *rajaPT);
   }
   else
   {
      RajaP = new ProlongationOperator(t_layout, v_layout, P);
   }
}
 
// **************************************************************************
RajaFiniteElementSpace::RajaFiniteElementSpace(const Engine &e,
                                               mfem::FiniteElementSpace &fespace)
   : PFiniteElementSpace(e, fespace),
     e_layout(e, 0), // resized in SetupLocalGlobalMaps()
     globalDofs(fes->GetNDofs()),
     localDofs(GetFE(0)->GetDof()),
     vdim(fespace.GetVDim()),
     ordering(fespace.GetOrdering()),
     globalToLocalOffsets(globalDofs+1),
     globalToLocalIndices(localDofs, GetNE()),
     localToGlobalMap(localDofs, GetNE())
{
   push(PowderBlue);
   const mfem::FiniteElement &fe = *(fes->GetFE(0));
   const mfem::TensorBasisElement* el =
      dynamic_cast<const mfem::TensorBasisElement*>(&fe);
   const mfem::Array<int> &dof_map = el->GetDofMap();
   const bool dof_map_is_identity = (dof_map.Size()==0);

   const Table& e2dTable = fes->GetElementToDofTable();
   const int* elementMap = e2dTable.GetJ();
   const int elements = GetNE();
   mfem::Array<int> h_offsets(globalDofs+1);

   e_layout.Resize(localDofs * elements * fes->GetVDim());

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

   mfem::Array<int> h_indices(localDofs*elements);
   mfem::Array<int> h_map(localDofs*elements);
   // For each global dof, fill in all local nodes that point to it
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

   globalToLocalOffsets = h_offsets;
   globalToLocalIndices = h_indices;
   localToGlobalMap = h_map;

   assert(fes);
   const mfem::SparseMatrix *R = fes->GetRestrictionMatrix();
   const mfem::Operator *P = fes->GetProlongationMatrix();

   CreateRPOperators(RajaVLayout(), RajaTrueVLayout(),
                     R, P,
                     restrictionOp,
                     prolongationOp);
}

// **************************************************************************
   RajaFiniteElementSpace::~RajaFiniteElementSpace()
{
   delete restrictionOp;
   delete prolongationOp;
}

// **************************************************************************
   void RajaFiniteElementSpace::GlobalToLocal(const raja::Vector &globalVec,
                                       Vector &localVec) const
{
   push(PowderBlue);
   const int vdim = GetVDim();
   const int localEntries = localDofs * GetNE();
   const bool vdim_ordering = ordering == Ordering::byVDIM;
   rGlobalToLocal(vdim,
                  vdim_ordering,
                  globalDofs,
                  localEntries,
                  globalToLocalOffsets,
                  globalToLocalIndices,
                  (const double*)globalVec.RajaMem().ptr(),
                  (double*)localVec.RajaMem().ptr());
   pop();
}

// **************************************************************************
void RajaFiniteElementSpace::LocalToGlobal(const Vector &localVec,
                                       Vector &globalVec) const
{
   push(PowderBlue);
   const int vdim = GetVDim();
   const int localEntries = localDofs * GetNE();
   const bool vdim_ordering = ordering == Ordering::byVDIM;
   rLocalToGlobal(vdim,
                  vdim_ordering,
                  globalDofs,
                  localEntries,
                  globalToLocalOffsets,
                  globalToLocalIndices,
                  (const double*)localVec.RajaMem().ptr(),
                  (double*)globalVec.RajaMem().ptr());
   pop();
}

} // namespace mfem::raja

} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_RAJA)
