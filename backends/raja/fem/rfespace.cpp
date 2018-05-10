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
#include "../raja.hpp"

namespace mfem {
  
  // ***************************************************************************
  // * RajaFiniteElementSpace
  //  ***************************************************************************
  RajaFiniteElementSpace::RajaFiniteElementSpace(Mesh* mesh,
                                                 const FiniteElementCollection* fec,
                                                 const int vdim_,
                                                 Ordering::Type ordering_)
    :ParFiniteElementSpace(static_cast<ParMesh*>(mesh),fec,vdim_,ordering_),
     globalDofs(GetNDofs()),
     localDofs(GetFE(0)->GetDof()),
     offsets(globalDofs+1),
     indices(localDofs, GetNE()),  
     map(localDofs, GetNE()) {
    push(PowderBlue);
    const FiniteElement *fe = GetFE(0);
    const TensorBasisElement* el = dynamic_cast<const TensorBasisElement*>(fe);
    const Array<int> &dof_map = el->GetDofMap();
    const bool dof_map_is_identity = (dof_map.Size()==0);
  
    const Table& e2dTable = GetElementToDofTable();
    const int* elementMap = e2dTable.GetJ();
    const int elements = GetNE();
    Array<int> h_offsets(globalDofs+1);
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
  
    Array<int> h_indices(localDofs*elements);
    Array<int> h_map(localDofs*elements);
    // For each global dof, fill in all local nodes that point   to it
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
  
    const SparseMatrix* R = GetRestrictionMatrix(); assert(R);
    //const Operator* P = GetProlongationMatrix(); assert(P);
    const RajaConformingProlongationOperator *P = new RajaConformingProlongationOperator(*this);
  
    const int mHeight = R->Height();
    const int* I = R->GetI();
    const int* J = R->GetJ();
    int trueCount = 0;
    for (int i = 0; i < mHeight; ++i) {
      trueCount += ((I[i + 1] - I[i]) == 1); 
    }
  
    Array<int> h_reorderIndices(2*trueCount);
    for (int i = 0, trueIdx=0; i < mHeight; ++i) {
      if ((I[i + 1] - I[i]) == 1) {
        h_reorderIndices[trueIdx++] = J[I[i]];
        h_reorderIndices[trueIdx++] = i;
      }
    }

    reorderIndices = ::new RajaArray<int>(2*trueCount);
    *reorderIndices = h_reorderIndices;
  
    restrictionOp = new RajaRestrictionOperator(R->Height(),
                                                R->Width(),
                                                reorderIndices);
    prolongationOp = new RajaProlongationOperator(P);
    pop();
  }

  // ***************************************************************************
  RajaFiniteElementSpace::~RajaFiniteElementSpace() {
    ::delete reorderIndices;
  }

  // ***************************************************************************
  bool RajaFiniteElementSpace::hasTensorBasis() const {
    assert(dynamic_cast<const TensorBasisElement*>(GetFE(0)));
    return true;
  }

  // ***************************************************************************
  void RajaFiniteElementSpace::GlobalToLocal(const RajaVector& globalVec,
                                             RajaVector& localVec) const {
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
                   globalVec,
                   localVec);
    pop();
  }

  // ***************************************************************************
  // Aggregate local node values to their respective global dofs
  void RajaFiniteElementSpace::LocalToGlobal(const RajaVector& localVec,
                                             RajaVector& globalVec) const {
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
                   localVec,
                   globalVec);
    pop();
  }
  
} // namespace mfem
