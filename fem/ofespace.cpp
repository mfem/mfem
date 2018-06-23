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

#include "pfespace.hpp"
#include "ofespace.hpp"
#include "ointerpolation.hpp"

namespace mfem {
  OccaFiniteElementSpace::OccaFiniteElementSpace(Mesh *mesh,
                                                 const FiniteElementCollection *fec,
                                                 Ordering::Type ordering_) {
    Init(occa::getDevice(), mesh, fec, 1, ordering_);
  }

  OccaFiniteElementSpace::OccaFiniteElementSpace(occa::device device_,
                                                 Mesh *mesh,
                                                 const FiniteElementCollection *fec,
                                                 Ordering::Type ordering_) {
    Init(device_, mesh, fec, 1, ordering_);
  }

  OccaFiniteElementSpace::OccaFiniteElementSpace(Mesh *mesh,
                                                 const FiniteElementCollection *fec,
                                                 const int vdim_,
                                                 Ordering::Type ordering_) {
    Init(occa::getDevice(), mesh, fec, vdim_, ordering_);
  }

  OccaFiniteElementSpace::OccaFiniteElementSpace(occa::device device_,
                                                 Mesh *mesh,
                                                 const FiniteElementCollection *fec,
                                                 const int vdim_,
                                                 Ordering::Type ordering_) {
    Init(device_, mesh, fec, vdim_, ordering_);
  }

  OccaFiniteElementSpace::~OccaFiniteElementSpace() {
    delete fespace;
    delete [] elementDofMap;
    delete [] elementDofMapInverse;
    delete restrictionOp;
    delete prolongationOp;
  }

  void OccaFiniteElementSpace::Init(occa::device device_,
                                    Mesh *mesh,
                                    const FiniteElementCollection *fec,
                                    const int vdim_,
                                    Ordering::Type ordering_) {
    device   = device_;
    vdim     = vdim_;
    ordering = ordering_;

#ifndef MFEM_USE_MPI
    fespace = new FiniteElementSpace(mesh, fec, vdim, ordering);
#else
    ParMesh *pmesh = dynamic_cast<ParMesh*>(mesh);
    if (pmesh == NULL) {
      fespace = new FiniteElementSpace(mesh, fec, vdim, ordering);
    } else {
      fespace = new ParFiniteElementSpace(pmesh, fec, vdim, ordering);
    }
#endif

    SetupLocalGlobalMaps();
    SetupOperators();
    SetupKernels();
  }

  void OccaFiniteElementSpace::SetupLocalGlobalMaps() {
    const FiniteElement &fe = *(fespace->GetFE(0));
    const TensorBasisElement *el = dynamic_cast<const TensorBasisElement*>(&fe);

    const Table &e2dTable = fespace->GetElementToDofTable();
    const int *elementMap = e2dTable.GetJ();
    const int elements = fespace->GetNE();

    globalDofs = fespace->GetNDofs();
    localDofs  = fe.GetDof();

    elementDofMap        = new int[localDofs];
    elementDofMapInverse = new int[localDofs];
    if (el &&
        el->GetDofMap().GetData()) {
      ::memcpy(elementDofMap,
               el->GetDofMap().GetData(),
               localDofs * sizeof(int));
    } else {
      for (int i = 0; i < localDofs; ++i) {
        elementDofMap[i] = i;
      }
    }
    for (int i = 0; i < localDofs; ++i) {
      elementDofMapInverse[elementDofMap[i]] = i;
    }

    // Allocate device offsets and indices
    globalToLocalOffsets.allocate(device,
                                  globalDofs + 1);
    globalToLocalIndices.allocate(device,
                                  localDofs, elements);
    localToGlobalMap.allocate(device,
                              localDofs, elements);

    int *offsets = globalToLocalOffsets.ptr();
    int *indices = globalToLocalIndices.ptr();
    int *l2gMap  = localToGlobalMap.ptr();

    // We'll be keeping a count of how many local nodes point
    //   to its global dof
    for (int i = 0; i <= globalDofs; ++i) {
      offsets[i] = 0;
    }

    for (int e = 0; e < elements; ++e) {
      for (int d = 0; d < localDofs; ++d) {
        const int gid = elementMap[localDofs*e + d];
        ++offsets[gid + 1];
      }
    }
    // Aggregate to find offsets for each global dof
    for (int i = 1; i <= globalDofs; ++i) {
      offsets[i] += offsets[i - 1];
    }
    // For each global dof, fill in all local nodes that point
    //   to it
    for (int e = 0; e < elements; ++e) {
      for (int d = 0; d < localDofs; ++d) {
        const int gid = elementMap[localDofs*e + elementDofMap[d]];
        const int lid = localDofs*e + d;
        indices[offsets[gid]++] = lid;
        l2gMap[lid] = gid;
      }
    }
    // We shifted the offsets vector by 1 by using it
    //   as a counter. Now we shift it back.
    for (int i = globalDofs; i > 0; --i) {
      offsets[i] = offsets[i - 1];
    }
    offsets[0] = 0;

    globalToLocalOffsets.keepInDevice();
    globalToLocalIndices.keepInDevice();
    localToGlobalMap.keepInDevice();
  }

  void OccaFiniteElementSpace::SetupOperators() {
    const SparseMatrix *R = fespace->GetRestrictionMatrix();
    const Operator *P = fespace->GetProlongationMatrix();
    CreateRPOperators(device,
                      fespace->GetNDofs(),
                      R, P,
                      restrictionOp,
                      prolongationOp);
  }

  void OccaFiniteElementSpace::SetupKernels() {
    occa::properties props("defines: {"
                           "  TILESIZE: 256,"
                           "}");
    props["defines/NUM_VDIM"] = vdim;

    props["defines/ORDERING_BY_NODES"] = 0;
    props["defines/ORDERING_BY_VDIM"]  = 1;
    props["defines/VDIM_ORDERING"] = (int) (ordering == Ordering::byVDIM);

    globalToLocalKernel = device.buildKernel("occa://mfem/fem/fespace.okl",
                                             "GlobalToLocal",
                                             props);
    localToGlobalKernel = device.buildKernel("occa://mfem/fem/fespace.okl",
                                             "LocalToGlobal",
                                             props);
  }

  occa::device OccaFiniteElementSpace::GetDevice() {
    return device;
  }

  Mesh* OccaFiniteElementSpace::GetMesh() {
    return fespace->GetMesh();
  }

  const Mesh* OccaFiniteElementSpace::GetMesh() const {
    return fespace->GetMesh();
  }

  FiniteElementSpace* OccaFiniteElementSpace::GetFESpace() {
    return fespace;
  }

  const FiniteElementSpace* OccaFiniteElementSpace::GetFESpace() const {
    return fespace;
  }

  bool OccaFiniteElementSpace::isDistributed() const {
#ifndef MFEM_USE_MPI
    return false;
#else
    return dynamic_cast<ParMesh*>(fespace->GetMesh());
#endif
  }

  bool OccaFiniteElementSpace::hasTensorBasis() const {
    return dynamic_cast<const TensorBasisElement*>(fespace->GetFE(0));
  }

  Ordering::Type OccaFiniteElementSpace::GetOrdering() const {
    return ordering;
  }

  int OccaFiniteElementSpace::GetGlobalDofs() const {
    return globalDofs;
  }

  int OccaFiniteElementSpace::GetLocalDofs() const {
    return localDofs;
  }

  int OccaFiniteElementSpace::GetDim() const {
    return fespace->GetMesh()->Dimension();
  }

  int OccaFiniteElementSpace::GetVDim() const {
    return vdim;
  }

  int OccaFiniteElementSpace::GetVSize() const {
    return globalDofs * vdim;
  }

  int OccaFiniteElementSpace::GetTrueVSize() const {
    return fespace->GetTrueVSize();
  }

  int OccaFiniteElementSpace::GetGlobalVSize() const {
#ifdef MFEM_USE_MPI
    ParFiniteElementSpace *pfespace = dynamic_cast<ParFiniteElementSpace*>(fespace);
    if (pfespace) {
      return pfespace->GlobalVSize();
    }
#endif
    return globalDofs * vdim;
  }

  int OccaFiniteElementSpace::GetGlobalTrueVSize() const {
#ifdef MFEM_USE_MPI
    ParFiniteElementSpace *pfespace = dynamic_cast<ParFiniteElementSpace*>(fespace);
    if (pfespace) {
      return pfespace->GlobalTrueVSize();
    }
#endif
    return fespace->GetTrueVSize();
  }

  int OccaFiniteElementSpace::GetNE() const {
    return fespace->GetNE();
  }

  const FiniteElementCollection* OccaFiniteElementSpace::FEColl() const {
    return fespace->FEColl();
  }

  const FiniteElement* OccaFiniteElementSpace::GetFE(const int idx) const {
    return fespace->GetFE(idx);
  }

  const int* OccaFiniteElementSpace::GetElementDofMap() const {
    return elementDofMap;
  }

  const int* OccaFiniteElementSpace::GetElementDofMapInverse() const {
    return elementDofMapInverse;
  }

  const Operator* OccaFiniteElementSpace::GetRestrictionOperator() {
    return restrictionOp;
  }

  const Operator* OccaFiniteElementSpace::GetProlongationOperator() {
    return prolongationOp;
  }

  const occa::array<int> OccaFiniteElementSpace::GetLocalToGlobalMap() const {
    return localToGlobalMap;
  }

  void OccaFiniteElementSpace::GlobalToLocal(const OccaVector &globalVec,
                                             OccaVector &localVec) const {
    globalToLocalKernel(globalDofs,
                        localDofs * fespace->GetNE(),
                        globalToLocalOffsets,
                        globalToLocalIndices,
                        globalVec, localVec);
  }

  // Aggregate local node values to their respective global dofs
  void OccaFiniteElementSpace::LocalToGlobal(const OccaVector &localVec,
                                             OccaVector &globalVec) const {
    localToGlobalKernel(globalDofs,
                        localDofs * fespace->GetNE(),
                        globalToLocalOffsets,
                        globalToLocalIndices,
                        localVec, globalVec);
  }
}

#endif
