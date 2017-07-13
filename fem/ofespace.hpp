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

#ifdef MFEM_USE_OCCA
#  ifndef MFEM_OCCA_FESPACE
#  define MFEM_OCCA_FESPACE

#include "occa.hpp"
#include "fespace.hpp"

namespace mfem {
  class OccaFiniteElementSpace {
  protected:
    occa::device device;
    FiniteElementSpace *fespace;

    int *elementDofMap;
    int *elementDofMapInverse;

    occa::array<int> globalToLocalOffsets;
    occa::array<int> globalToLocalIndices;
    occa::array<int> localToGlobalMap;
    occa::kernel globalToLocalKernel, localToGlobalKernel;

    int globalDofs, localDofs;
    int vdim;

    Operator *restrictionOp, *prolongationOp;

  public:
    OccaFiniteElementSpace(Mesh *mesh,
                           const FiniteElementCollection *fec,
                           const int vdim_ = 1);

    OccaFiniteElementSpace(occa::device device_,
                           Mesh *mesh,
                           const FiniteElementCollection *fec,
                           const int vdim_ = 1);

    ~OccaFiniteElementSpace();

    void Init(occa::device device_,
              Mesh *mesh,
              const FiniteElementCollection *fec,
              const int vdim_);

    void SetupLocalGlobalMaps();
    void SetupOperators();
    void SetupKernels();

    Mesh* GetMesh();
    FiniteElementSpace* GetFESpace();

    int GetGlobalDofs() const;
    int GetLocalDofs() const;
    int GetVDim() const;

    int GetVSize() const;
    int GetTrueVSize() const;
    int GetGlobalVSize() const;
    int GetGlobalTrueVSize() const;

    const int* GetElementDofMap() const;
    const int* GetElementDofMapInverse() const;

    const Operator* GetRestrictionOperator();
    const Operator* GetProlongationOperator();

    const occa::array<int> GetLocalToGlobalMap() const;

    void GlobalToLocal(const OccaVector &globalVec,
                       OccaVector &localVec) const;
    void LocalToGlobal(const OccaVector &localVec,
                       OccaVector &globalVec) const;
  };
}

#  endif
#endif
