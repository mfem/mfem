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

  // ndof * nelems
  int stride_sdim;
  // number of points in pptr
  int npts;
  // number of points per element along each dimension to test
  int nq1d;
  // total number of points to test
  int nq;
};

template <int Geom, int SDim, bool use_dev> struct PhysNodeFinder;

template <int SDim, bool use_dev>
struct PhysNodeFinder<Geometry::SEGMENT, SDim, use_dev>
    : public NodeFinderBase {

  static int compute_nq(int nq1d) { return nq1d; }

  static int compute_stride_sdim(int ndof1d, int nelems) {
    return ndof1d * nelems;
  }

  void MFEM_HOST_DEVICE operator()(int idx) const {
    constexpr int Dim = 1;
    constexpr int max_team_x = use_dev ? 128 : 1;
    int n = (nq < max_team_x) ? nq : max_team_x;
    // L-2 norm squared
    MFEM_SHARED real_t dists[max_team_x];
    MFEM_SHARED real_t ref_buf[Dim * max_team_x];
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
      for (int j = 0; j < poly1d.pN; ++j) {
        real_t b = poly1d.eval(qptr[i], j);
        for (int d = 0; d < SDim; ++d) {
          phys_coord[d] = mptr[j + eptr[idx] * poly1d.pN + d * stride_sdim] * b;
        }
      }
      real_t dist = 0;
      // L-2 norm squared
      for (int d = 0; d < SDim; ++d) {
        real_t tmp = phys_coord[d] - pptr[idx + SDim * npts];
        dist += tmp * tmp;
      }
      if (dist < dists[MFEM_THREAD_ID(x)]) {
        // closer guess in physical space
        dists[MFEM_THREAD_ID(x)] = dist;
        ref_buf[MFEM_THREAD_ID(x)] = qptr[MFEM_THREAD_ID(x)];
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
    MFEM_FOREACH_THREAD(d, x, Dim) { xptr[idx + d * npts] = ref_buf[d * n]; }
  }
};

template <int SDim, bool use_dev>
struct PhysNodeFinder<Geometry::SQUARE, SDim, use_dev> : public NodeFinderBase {

  static int compute_nq(int nq1d) { return nq1d * nq1d; }

  static int compute_stride_sdim(int ndof1d, int nelems) {
    return ndof1d * ndof1d * nelems;
  }

  void MFEM_HOST_DEVICE operator()(int idx) const {
    constexpr int Dim = 2;
    constexpr int max_team_x = use_dev ? 128 : 1;
    int n = (nq < max_team_x) ? nq : max_team_x;
    // L-2 norm squared
    MFEM_SHARED real_t dists[max_team_x];
    MFEM_SHARED real_t ref_buf[Dim * max_team_x];
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
      int idcs[Dim];
      idcs[0] = i % nq1d;
      idcs[1] = i / nq1d;
      for (int j0 = 0; j0 < poly1d.pN; ++j0) {
        real_t b0 = poly1d.eval(qptr[idcs[0]], j0);
        for (int j1 = 0; j1 < poly1d.pN; ++j1) {
          real_t b = b0 * poly1d.eval(qptr[idcs[1]], j1);
          for (int d = 0; d < SDim; ++d) {
            phys_coord[d] = mptr[j0 + (j1 + eptr[idx] * poly1d.pN) * poly1d.pN +
                                 d * stride_sdim] *
                            b;
          }
        }
      }
      real_t dist = 0;
      // L-2 norm squared
      for (int d = 0; d < SDim; ++d) {
        real_t tmp = phys_coord[d] - pptr[idx + SDim * npts];
        dist += tmp * tmp;
      }
      if (dist < dists[MFEM_THREAD_ID(x)]) {
        // closer guess in physical space
        dists[MFEM_THREAD_ID(x)] = dist;
        ref_buf[MFEM_THREAD_ID(x)] = qptr[MFEM_THREAD_ID(x)];
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
    MFEM_FOREACH_THREAD(d, x, Dim) { xptr[idx + d * npts] = ref_buf[d * n]; }
  }
};

template <bool use_dev>
struct PhysNodeFinder<Geometry::CUBE, 3, use_dev> : public NodeFinderBase {

  static int compute_nq(int nq1d) { return nq1d * nq1d * nq1d; }

  static int compute_stride_sdim(int ndof1d, int nelems) {
    return ndof1d * ndof1d * ndof1d * nelems;
  }

  void MFEM_HOST_DEVICE operator()(int idx) const {
    constexpr int Dim = 3;
    constexpr int SDim = 3;
    constexpr int max_team_x = use_dev ? 128 : 1;
    int n = (nq < max_team_x) ? nq : max_team_x;
    // L-2 norm squared
    MFEM_SHARED real_t dists[max_team_x];
    MFEM_SHARED real_t ref_buf[Dim * max_team_x];
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
      int idcs[Dim];
      idcs[0] = i % nq1d;
      idcs[1] = i / nq1d;
      idcs[2] = idcs[1] / nq1d;
      idcs[1] %= nq1d;
      for (int j0 = 0; j0 < poly1d.pN; ++j0) {
        real_t b0 = poly1d.eval(qptr[idcs[0]], j0);
        for (int j1 = 0; j1 < poly1d.pN; ++j1) {
          real_t b1 = b0 * poly1d.eval(qptr[idcs[1]], j1);
          for (int j2 = 0; j2 < poly1d.pN; ++j2) {
            real_t b = b1 * poly1d.eval(qptr[idcs[2]], j2);
            for (int d = 0; d < SDim; ++d) {
              phys_coord[d] =
                  mptr[j0 +
                       (j1 + (j2 + eptr[idx] * poly1d.pN) * poly1d.pN) *
                           poly1d.pN +
                       d * stride_sdim] *
                  b;
            }
          }
        }
      }
      real_t dist = 0;
      // L-2 norm squared
      for (int d = 0; d < SDim; ++d) {
        real_t tmp = phys_coord[d] - pptr[idx + SDim * npts];
        dist += tmp * tmp;
      }
      if (dist < dists[MFEM_THREAD_ID(x)]) {
        // closer guess in physical space
        dists[MFEM_THREAD_ID(x)] = dist;
        ref_buf[MFEM_THREAD_ID(x)] = qptr[MFEM_THREAD_ID(x)];
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
    MFEM_FOREACH_THREAD(d, x, Dim) { xptr[idx + d * npts] = ref_buf[d * n]; }
  }
};

template <int Geom, int SDim, bool use_dev>
static void ClosestPhysNodeImpl(int npts, int nelems, int ndof1d, int nq1d,
                                const real_t *mptr, const real_t *pptr,
                                const int *eptr, const real_t *nptr,
                                const real_t *qptr, real_t *xptr) {
  PhysNodeFinder<Geom, SDim, use_dev> func;
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
  func.nq = func.compute_nq(nq1d);
  func.stride_sdim = func.compute_stride_sdim(ndof1d, nelems);
  // TODO: any batching of npts?
  int team_x = std::min<int>(max_team_x, func.nq);
  forall_2D(npts, team_x, 1, func);
}

template <int Geom, int SDim, bool use_dev>
static void ClosestRefNodeImpl(int npts, int nelems, int ndof1d, int nq1d,
                               const real_t *mptr, const real_t *pptr,
                               const int *eptr, const real_t *nptr,
                               const real_t *qptr, real_t *xptr) {
  // TODO
  MFEM_ABORT("ClostestRefNodeImpl not implemented yet");
}

template <int Geom, int SDim, bool use_dev>
BatchInverseElementTransformation::ClosestPhysPointKernelType
BatchInverseElementTransformation::FindClosestPhysPoint::Kernel() {
  return ClosestPhysNodeImpl<Geom, SDim, use_dev>;
}

BatchInverseElementTransformation::ClosestPhysPointKernelType
BatchInverseElementTransformation::FindClosestPhysPoint::Fallback(int, int,
                                                                  bool) {
  MFEM_ABORT("Invalid Geom/SDim combination");
}

template <int Geom, int SDim, bool use_dev>
BatchInverseElementTransformation::ClosestRefPointKernelType
BatchInverseElementTransformation::FindClosestRefPoint::Kernel() {
  return ClosestRefNodeImpl<Geom, SDim, use_dev>;
}

BatchInverseElementTransformation::ClosestRefPointKernelType
BatchInverseElementTransformation::FindClosestRefPoint::Fallback(int, int,
                                                                 bool) {
  MFEM_ABORT("Invalid Geom/SDim combination");
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
      FindClosestPhysPoint::Run(geom, vdim, use_dev, npts, NE, ndof1d, nq1d,
                                mptr, pptr, eptr, nptr, qptr, xptr);
    } else {
      FindClosestRefPoint::Run(geom, vdim, use_dev, npts, NE, ndof1d, nq1d,
                               mptr, pptr, eptr, nptr, qptr, xptr);
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
  BatchInverseElementTransformation::AddFindClosestSpecialization<
      Geometry::SEGMENT, 1, true>();
  BatchInverseElementTransformation::AddFindClosestSpecialization<
      Geometry::SEGMENT, 2, true>();
  BatchInverseElementTransformation::AddFindClosestSpecialization<
      Geometry::SEGMENT, 3, true>();

  BatchInverseElementTransformation::AddFindClosestSpecialization<
      Geometry::SQUARE, 2, true>();
  BatchInverseElementTransformation::AddFindClosestSpecialization<
      Geometry::SQUARE, 3, true>();

  BatchInverseElementTransformation::AddFindClosestSpecialization<
      Geometry::CUBE, 3, true>();

  BatchInverseElementTransformation::AddFindClosestSpecialization<
      Geometry::SEGMENT, 1, false>();
  BatchInverseElementTransformation::AddFindClosestSpecialization<
      Geometry::SEGMENT, 2, false>();
  BatchInverseElementTransformation::AddFindClosestSpecialization<
      Geometry::SEGMENT, 3, false>();

  BatchInverseElementTransformation::AddFindClosestSpecialization<
      Geometry::SQUARE, 2, false>();
  BatchInverseElementTransformation::AddFindClosestSpecialization<
      Geometry::SQUARE, 3, false>();

  BatchInverseElementTransformation::AddFindClosestSpecialization<
      Geometry::CUBE, 3, false>();
}
} // namespace mfem
