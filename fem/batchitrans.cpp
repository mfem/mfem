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

#include "general/forall.hpp"

#include <cmath>

namespace mfem {

BatchInverseElementTransformation::~BatchInverseElementTransformation() {}

void BatchInverseElementTransformation::Setup(Mesh &m, MemoryType d_mt) {
  static Kernels kernels;

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

// helper for finding the batch transform initial guess
struct NodeFinderBase {
  // physical space coordinates of mesh element nodes
  const real_t *mptr;
  // physical space point coordinates to find
  const real_t *pptr;
  // element indices
  const int *eptr;
  // reference space nodes to test
  const real_t *qptr;
  // initial guess results
  real_t *xptr;
  eltrans::Lagrange poly1d;

  // number of points in pptr
  int npts;
  // number of points per element along each dimension to test
  int nq1d;
  // total number of points to test
  int nq;
};

template <int Dim, int SDim, bool use_dev> struct PhysNodeFinder;

template <int SDim, bool use_dev>
struct PhysNodeFinder<1, SDim, use_dev> : public NodeFinderBase {

  void MFEM_HOST_DEVICE operator()(int idx) const {
    constexpr int Dim = 1;
    constexpr int max_team_x = use_dev ? 128 : 1;
    int n = (nq < max_team_x) ? nq : max_team_x;
    MFEM_SHARED real_t dists[max_team_x];
    MFEM_SHARED real_t ref_buf[Dim * max_team_x];
    // MFEM_SHARED real_t phys_buf[SDim * max_team_x];
    MFEM_FOREACH_THREAD(i, x, n) {
#ifdef MFEM_USE_DOUBLE
      dists[i] = HUGE_VAL;
#else
      dists[i] = HUGE_VALF;
#endif
    }
    MFEM_SYNC_THREAD;
    // team serial portion
    MFEM_FOREACH_THREAD(i, x, nq) {
      real_t phys_coord[SDim] = {0};
      // TODO: evaluate basis at q point i
      real_t b = 0;
      for (int d = 0; d < SDim; ++d) {
        // TODO: correct index into mptr for element eptr[idx] and node
        // corresponding to basis b
        phys_coord[d] = mptr[eptr[idx]] * b;
      }
    }
    // now do tree reduce
    for (int i = (MFEM_THREAD_SIZE(x) >> 1); i > 0; i >>= 1) {
      MFEM_SYNC_THREAD;
      if (MFEM_THREAD_ID(x) < i) {
        if (dists[MFEM_THREAD_ID(x) + i] < dists[MFEM_THREAD_ID(x)]) {
          dists[MFEM_THREAD_ID(x)] = dists[MFEM_THREAD_ID(x) + i];
          for (int d = 0; d < Dim; ++d) {
            ref_buf[MFEM_THREAD_ID(x) + d * n] =
                ref_buf[MFEM_THREAD_ID(x) + i + d * n];
          }
        }
      }
    }
    // write results out
    MFEM_SYNC_THREAD;
    MFEM_FOREACH_THREAD(d, x, Dim) {
      xptr[idx + d * npts] = ref_buf[d * n];
    }
  }
};

template <bool use_dev>
struct PhysNodeFinder<2, 2, use_dev> : public NodeFinderBase {

  void MFEM_HOST_DEVICE operator()(int idx) const {
    // TODO
  }
};

template <bool use_dev>
struct PhysNodeFinder<3, 3, use_dev> : public NodeFinderBase {

  void MFEM_HOST_DEVICE operator()(int idx) const {
    // TODO
  }
};

template <bool use_dev>
struct PhysNodeFinder<2, 3, use_dev> : public NodeFinderBase {

  void MFEM_HOST_DEVICE operator()(int idx) const {
    // TODO
  }
};

template <int Dim, int SDim, bool use_dev>
static void ClosestPhysNodeImpl(int npts, int ndof1d, int nq1d,
                                const real_t *mptr, const real_t *pptr,
                                const int *eptr, const real_t *nptr,
                                const real_t *qptr, real_t *xptr) {
  PhysNodeFinder<Dim, SDim, use_dev> func;
  constexpr int max_team_x = use_dev ? 128 : 1;
  func.poly1d.z = nptr;
  func.poly1d.pN = ndof1d;
  func.mptr = mptr;
  func.pptr = pptr;
  func.eptr = eptr;
  func.qptr = qptr;
  func.xptr = xptr;
  func.npts = npts;
  func.nq1d = nq1d;
  func.nq = nq1d;
  // MFEM_ASSERT(nq1d <= max_team_x, "requested nq1d must be <= 128");
  for (int d = 0; d < Dim; ++d) {
    func.nq *= nq1d;
  }
  // TODO: any batching of npts?
  int team_x = std::min<int>(max_team_x, func.nq);
  forall_2D(npts, team_x, 1, func);
}

template <int Dim, int SDim, bool use_dev>
static void ClosestRefNodeImpl(int npts, int ndof1d, int nq1d,
                               const real_t *mptr, const real_t *pptr,
                               const int *eptr, const real_t *nptr,
                               const real_t *qptr, real_t *xptr) {
  // TODO
  MFEM_ABORT("ClostestRefNodeImpl not implemented yet");
}

template <int Dim, int SDim, bool use_dev>
BatchInverseElementTransformation::ClosestPhysPointKernelType
BatchInverseElementTransformation::FindClosestPhysPoint::Kernel() {
  return ClosestPhysNodeImpl<Dim, SDim, use_dev>;
}

BatchInverseElementTransformation::ClosestPhysPointKernelType
BatchInverseElementTransformation::FindClosestPhysPoint::Fallback(int, int,
                                                                  bool) {
  MFEM_ABORT("Invalid Dim/SDim combination");
}

template <int Dim, int SDim, bool use_dev>
BatchInverseElementTransformation::ClosestRefPointKernelType
BatchInverseElementTransformation::FindClosestRefPoint::Kernel() {
  return ClosestRefNodeImpl<Dim, SDim, use_dev>;
}

BatchInverseElementTransformation::ClosestRefPointKernelType
BatchInverseElementTransformation::FindClosestRefPoint::Fallback(int, int,
                                                                 bool) {
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
  const int order = fe->GetOrder();
  int npts = elems.Size();
  auto geom = fe->GetGeomType();

  types.SetSize(npts);
  refs.SetSize(npts * dim);

  auto pptr = pts.Read(use_dev);
  auto eptr = elems.Read(use_dev);
  auto mptr = node_pos.Read(use_dev);
  auto tptr = types.Write(use_dev);
  auto xptr = refs.ReadWrite(use_dev);
  auto nptr = points1d->Read(use_dev);
  int ndof1d = points1d->Size();

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
    int nq1d = std::max(order + rel_qpts_order, 0) + 1;
    auto qpoints = poly1d.GetPointsArray(guess_points_type, nq1d - 1);
    auto qptr = qpoints->Read(use_dev);
    if (init_guess_type == InverseElementTransformation::ClosestPhysNode) {
      FindClosestPhysPoint::Run(dim, vdim, use_dev, npts, ndof1d, nq1d, mptr,
                                pptr, eptr, nptr, qptr, xptr);
    } else {
      FindClosestRefPoint::Run(dim, vdim, use_dev, npts, ndof1d, nq1d, mptr,
                               pptr, eptr, nptr, qptr, xptr);
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
  BatchInverseElementTransformation::AddFindClosestSpecialization<1, 1, true>();
  BatchInverseElementTransformation::AddFindClosestSpecialization<1, 2, true>();
  BatchInverseElementTransformation::AddFindClosestSpecialization<1, 3, true>();

  BatchInverseElementTransformation::AddFindClosestSpecialization<2, 2, true>();
  BatchInverseElementTransformation::AddFindClosestSpecialization<2, 3, true>();

  BatchInverseElementTransformation::AddFindClosestSpecialization<3, 3, true>();

  BatchInverseElementTransformation::AddFindClosestSpecialization<1, 1,
                                                                  false>();
  BatchInverseElementTransformation::AddFindClosestSpecialization<1, 2,
                                                                  false>();
  BatchInverseElementTransformation::AddFindClosestSpecialization<1, 3,
                                                                  false>();

  BatchInverseElementTransformation::AddFindClosestSpecialization<2, 2,
                                                                  false>();
  BatchInverseElementTransformation::AddFindClosestSpecialization<2, 3,
                                                                  false>();

  BatchInverseElementTransformation::AddFindClosestSpecialization<3, 3,
                                                                  false>();
}
} // namespace mfem
