// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "../mesh/mesh_headers.hpp"
#include "eltrans/eltrans_basis.hpp"
#include "fem.hpp"

#include <cmath>

namespace mfem {

BatchInverseElementTransformation::~BatchInverseElementTransformation() {}

void BatchInverseElementTransformation::Setup(Mesh &m, MemoryType d_mt) {
  mesh = &m;
  MFEM_VERIFY(mesh->GetNodes(), "the provided mesh must have valid nodes.");
  const FiniteElementSpace *fespace = mesh->GetNodalFESpace();
  const bool use_tensor_products = UsesTensorBasis(*fespace);
  MFEM_VERIFY(
      use_tensor_products,
      "BatchInverseElementTransform only supports UsesTensorBasis() == true");
  const FiniteElement *fe = fespace->GetTypicalFE();
  const TensorBasisElement *tfe = dynamic_cast<const TensorBasisElement *>(fe);
  const int dim = fe->GetDim();
  const int vdim = fespace->GetVDim();
  const int NE = fespace->GetNE();
  const int ND = fe->GetDof();
  const int order = fe->GetOrder();

  // can't just use mesh->GetGeometricFactors since we need the raw element DOFs
  const ElementDofOrdering e_ordering = use_tensor_products
                                            ? ElementDofOrdering::LEXICOGRAPHIC
                                            : ElementDofOrdering::NATIVE;
  const Operator *elem_restr = fespace->GetElementRestriction(e_ordering);
  MemoryType my_d_mt =
      (d_mt != MemoryType::DEFAULT) ? d_mt : Device::GetDeviceMemoryType();
  node_pos.SetSize(vdim * ND * NE, my_d_mt);
  elem_restr->Mult(*mesh->GetNodes(), node_pos);
  points1d = poly1d.GetPointsArray(tfe->GetBasisType(), order);
}

// helper for finding the batch transform ClosestPhysNode initial guess
template <int Dim, int SDim> struct PhysNodeFinder {
  // physical space point coordinates to find
  const real_t *pptr;
  // element indices
  const int *eptr;
  // initial guess results
  real_t *xptr;
  eltrans::Lagrange poly1d;

  void MFEM_HOST_DEVICE operator()(int idx) const {
    // TODO
  }
};

template <int Dim, int SDim>
BatchInverseElementTransformation::ClosestPhysPointKernelType
BatchInverseElementTransformation::FindClosestPhysPoint::Kernel() {
  // TODO
  MFEM_ABORT("");
}

BatchInverseElementTransformation::ClosestPhysPointKernelType
BatchInverseElementTransformation::FindClosestPhysPoint::Fallback(int, int) {
  MFEM_ABORT("Invalid Dim/SDim combination");
}

template <int Dim, int SDim>
BatchInverseElementTransformation::ClosestRefPointKernelType
BatchInverseElementTransformation::FindClosestRefPoint::Kernel() {
  // TODO
  MFEM_ABORT("");
}

BatchInverseElementTransformation::ClosestRefPointKernelType
BatchInverseElementTransformation::FindClosestRefPoint::Fallback(int, int) {
  MFEM_ABORT("Invalid Dim/SDim combination");
}

void BatchInverseElementTransformation::Transform(const Vector &pts,
                                                  const Array<int> &elems,
                                                  Array<int> &types,
                                                  Vector &refs, bool use_dev) {
  const FiniteElementSpace *fespace = mesh->GetNodalFESpace();
  const FiniteElement *fe = fespace->GetTypicalFE();
  const int dim = fe->GetDim();
  const int vdim = fespace->GetVDim();
  const int NE = fespace->GetNE();
  const int ND = fe->GetDof();
  int npts = elems.Size();
  auto geom = fe->GetGeomType();

  types.SetSize(npts);
  refs.SetSize(npts * dim);

  auto pptr = pts.Read(use_dev);
  auto eptr = elems.Read(use_dev);
  auto tptr = types.Write(use_dev);
  auto xptr = refs.ReadWrite(use_dev);
  auto nptr = points1d->Read(use_dev);

  switch (init_guess_type) {
  case InverseElementTransformation::Center: {
    real_t cx, cy, cz;
    auto ip0 = Geometries.GetCenter(geom);
    cx = ip0.x;
    cy = ip0.y;
    cz = ip0.z;
    switch (dim) {
    case 1:
      forall_switch(use_dev, npts,
                    [=] MFEM_HOST_DEVICE(int i) { xptr[i] = cx; });
      break;
    case 2:
      forall_switch(use_dev, npts, [=] MFEM_HOST_DEVICE(int i) {
        xptr[i] = cx;
        xptr[i + npts] = cy;
      });
      break;
    case 3:
      forall_switch(use_dev, npts, [=] MFEM_HOST_DEVICE(int i) {
        xptr[i] = cx;
        xptr[i + npts] = cy;
        xptr[i + 2 * npts] = cz;
      });
      break;
    }
  } break;
  case InverseElementTransformation::ClosestRefNode:
  case InverseElementTransformation::ClosestPhysNode: {
    // TODO: get 1D geometry refine points
    auto qptr = nptr;
    if (init_guess_type == InverseElementTransformation::ClosestPhysNode) {
      FindClosestPhysPoint::Run(dim, vdim, npts, pptr, eptr, nptr, qptr, xptr);
    } else {
      FindClosestRefPoint::Run(dim, vdim, npts, pptr, eptr, nptr, qptr, xptr);
    }
  } break;
  case InverseElementTransformation::GivenPoint:
    // nothing to do here
    break;
  case InverseElementTransformation::EdgeScan: {
    // TODO
  }
    return;
  }
  // general case: for each point, use guess inside refs
}

BatchInverseElementTransformation::Kernels::Kernels() {
  BatchInverseElementTransformation::AddFindClosestSpecialization<1, 1>();
  BatchInverseElementTransformation::AddFindClosestSpecialization<1, 2>();
  BatchInverseElementTransformation::AddFindClosestSpecialization<1, 3>();

  BatchInverseElementTransformation::AddFindClosestSpecialization<2, 2>();
  BatchInverseElementTransformation::AddFindClosestSpecialization<2, 3>();

  BatchInverseElementTransformation::AddFindClosestSpecialization<3, 3>();
}
} // namespace mfem
