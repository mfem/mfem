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
#if defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_KERNELS)

#include "../kernels.hpp"

namespace mfem {

namespace kernels {


// ************************************************************************** 
kFiniteElementSpace::
kFiniteElementSpace(const Engine& e,
                    mfem::FiniteElementSpace& fespace)
   : PFiniteElementSpace(e, fespace),
     e_layout(e, 0),
     globalDofs(fes->GetNDofs()),
     localDofs(GetFE(0)->GetDof()),
     vdim(fespace.GetVDim()),
     ordering(fespace.GetOrdering()),
     offsets(globalDofs+1),
     indices(localDofs, GetNE()),
     map(localDofs, GetNE()) {
  push(PowderBlue);
  const mfem::FiniteElement& fe = *(fes->GetFE(0));
  const mfem::TensorBasisElement* el =
    dynamic_cast<const mfem::TensorBasisElement*>(&fe);
  const mfem::Array<int>& dof_map = el->GetDofMap();
  const bool dof_map_is_identity = (dof_map.Size()==0);

  const Table& e2dTable = fes->GetElementToDofTable();
  const int* elementMap = e2dTable.GetJ();
  const int elements = GetNE();
  mfem::Array<int> h_offsets(globalDofs+1);

  const int e_size = localDofs * elements * fes->GetVDim();
  e_layout.Resize(e_size);
  dbg("\033[7m[kFiniteElementSpace] e_size/fes->GetVDim()=%d",e_size/fes->GetVDim());

  // We'll be keeping a count of how many local nodes point to its global dof
  for (int i = 0; i <= globalDofs; ++i) {
    h_offsets[i] = 0;
  }
  for (int e = 0; e < elements; ++e) {
    for (int d = 0; d < localDofs; ++d) {
      const int gid = elementMap[localDofs*e + d];
      ++h_offsets[gid + 1];
    }
  }
  // Aggregate to find offsets for each global dof
  for (int i = 1; i <= globalDofs; ++i) {
    h_offsets[i] += h_offsets[i - 1];
  }

  mfem::Array<int> h_indices(localDofs*elements);
  mfem::Array<int> h_map(localDofs*elements);
  // For each global dof, fill in all local nodes that point to it
  for (int e = 0; e < elements; ++e) {
    for (int d = 0; d < localDofs; ++d) {
      const int did = dof_map_is_identity?d:dof_map[d];
      const int gid = elementMap[localDofs*e + did];
      const int lid = localDofs*e + d;
      h_indices[h_offsets[gid]++] = lid;
      h_map[lid] = gid;
    }
  }

  // We shifted the offsets vector by 1 by using it as a counter
  // Now we shift it back.
  for (int i = globalDofs; i > 0; --i) {
    h_offsets[i] = h_offsets[i - 1];
  }
  h_offsets[0] = 0;

  offsets = h_offsets;
  indices = h_indices;
  map = h_map;

  const mfem::SparseMatrix* R = fes->GetRestrictionMatrix();
  const mfem::Operator* P = fes->GetProlongationMatrix();

  if (!P) {
    restrictionOp = new IdentityOperator(KernelsTrueVLayout());
    prolongationOp = new IdentityOperator(KernelsTrueVLayout());
    pop();
    return;
  }

  assert(R);
  assert(P);
  /*const RajaConformingProlongationOperator *P =
    new RajaConformingProlongationOperator(this->GetParFESpace());*/

  const int mHeight = R->Height();
  const int* I = R->GetI();
  const int* J = R->GetJ();
  int trueCount = 0;
  for (int i = 0; i < mHeight; ++i) {
     trueCount += ((I[i + 1] - I[i]) == 1); 
  }
  
  mfem::Array<int> h_reorderIndices(2*trueCount);
  for (int i = 0, trueIdx=0; i < mHeight; ++i) {
     if ((I[i + 1] - I[i]) == 1) {
        h_reorderIndices[trueIdx++] = J[I[i]];
        h_reorderIndices[trueIdx++] = i;
     }
  }

  reorderIndices = ::new kernels::array<int>(2*trueCount);
  *reorderIndices = h_reorderIndices;
  
  restrictionOp = new kernels::RestrictionOperator(KernelsVLayout(),
                                                   KernelsTrueVLayout(),
                                                   *reorderIndices);
  prolongationOp = new kernels::ProlongationOperator(KernelsTrueVLayout(),
                                                     KernelsVLayout(),
                                                     P);
  pop();
}

// **************************************************************************
kFiniteElementSpace::~kFiniteElementSpace() {
  delete restrictionOp;
  delete prolongationOp;
}

// **************************************************************************
void kFiniteElementSpace::GlobalToLocal(const Vector& globalVec,
                                        Vector& localVec) const {
  push(PowderBlue);
  const int vdim = GetVDim();
  const int localEntries = localDofs * GetNE();
  const bool vdim_ordering = ordering == Ordering::byVDIM;
  rGlobalToLocal(vdim,
                 vdim_ordering,
                 globalDofs,
                 localEntries,
                 offsets,
                 indices,
                 (const double*)globalVec.KernelsMem().ptr(),
                 (double*)localVec.KernelsMem().ptr());
  pop();
}

// **************************************************************************
void kFiniteElementSpace::LocalToGlobal(const Vector& localVec,
                                        Vector& globalVec) const {
  push(PowderBlue);
  const int vdim = GetVDim();
  const int localEntries = localDofs * GetNE();
  const bool vdim_ordering = ordering == Ordering::byVDIM;
  rLocalToGlobal(vdim,
                 vdim_ordering,
                 globalDofs,
                 localEntries,
                 offsets,
                 indices,
                 (const double*)localVec.KernelsMem().ptr(),
                 (double*)globalVec.KernelsMem().ptr());
  pop();
}

   
} // namespace mfem::kernels

} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_KERNELS)
