// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.
#pragma once

/** @file vector.hpp
    Vector H1 simplex MMA helpers: Elasticity, VectorDivergence and
    VectorConvectionNL.

    A vector E-vector X(nd, ncomp, NE) is a column-major nd x (ncomp NE)
    matrix, so one GEMM interpolates every component of every element:
    U = P X, with P the dense (rows x nd) basis at the quadrature points,
    rows = nq * planes. Pointwise kernels couple the components, then
    Y += P^T V scatters back.

    Device (CUDA, or the debug device): SimplexVecApply launches one block
    per batch of elements; X is staged in shared memory and both products
    go through mma::Gemm / mma::GemmT, i.e. FP64 DMMA (mma.sync m8n8k4) on
    sm_80+ in double precision, the cooperative blas loops otherwise.
    HIP-only builds and batches that do not fit in shared memory use the
    dense forall path. Host: vendor GEMMs with LAPACK, else forall loops.
*/

#include "mma.hpp"
#include "mode/lapack.hpp"

namespace mfem
{

/// \cond DO_NOT_DOCUMENT

/** True if VectorDivergence can use simplex MMA: vector H1 trial with
    UsesSimplexMMA and vdim == dim, and a scalar simplex test space whose
    E-vector dof order the basis builder knows (nodal H1/L2 or Positive). */
inline bool UsesSimplexMmaVectorDivergence(const FiniteElementSpace &trial,
                                           const FiniteElementSpace &test)
{
   bool ok = UsesSimplexMMA(trial);
   Mesh *mesh = trial.GetMesh();
   const int dim = mesh->Dimension();
   ok = ok && test.GetMesh() == mesh && !test.IsVariableOrder() &&
        trial.GetVDim() == dim && test.GetVDim() == 1 &&
        trial.GetOrdering() == Ordering::byNODES;
   if (ok && TypicalFeAvailable(*mesh))
   {
      const FiniteElement &tr = *trial.GetTypicalFE();
      const FiniteElement &te = *test.GetTypicalFE();
      ok = te.GetGeomType() == tr.GetGeomType() &&
           te.GetRangeType() == FiniteElement::SCALAR &&
           (dynamic_cast<const NodalFiniteElement *>(&te) ||
            dynamic_cast<const H1Pos_TriangleElement *>(&te) ||
            dynamic_cast<const H1Pos_TetrahedronElement *>(&te));
   }
   return AllreduceMmaFlag(trial, ok);
}

namespace internal::mma
{

/** Dense basis P(q, p, i) at @a ir, in the E-vector dof order of @a fes for
    @a ordering. Plane p = 0 is the value when @a with_value, the next dim
    planes are reference gradients when @a with_grad. */
void SimplexVectorBasis(const FiniteElementSpace &fes,
                        const IntegrationRule &ir,
                        const ElementDofOrdering ordering,
                        const bool with_value, const bool with_grad,
                        Array<real_t> &P);

/** D(q, a + dim b, e) = w(q) c(q, e) adj(J)(a, b), divided by det(J) when
    @a inv_det. Coefficient @a c is FULL (nq NE) or a single constant. */
void SimplexAdjSetup(Mesh &mesh, const IntegrationRule &ir,
                     const Vector &c, const bool inv_det, Vector &d);

/** U(r, col) = sum_i P(r, i) X(i, col); P is (ldp x nd), r < rows. */
inline void SimplexInterp(const int rows, const int ldp, const int nd,
                          const int ncol, const Array<real_t> &P,
                          const Vector &x, Vector &u)
{
   if (ncol <= 0) { return; }
#ifdef MFEM_USE_LAPACK
   if (!Device::Allows(Backend::DEVICE_MASK))
   {
      lapack::Gemm('N', 'N', rows, ncol, nd, real_t(1), P.HostRead(), ldp,
                   x.HostRead(), nd, real_t(0), u.HostWrite(), rows);
      return;
   }
#endif
   const real_t *Pm = P.Read();
   const auto X = Reshape(x.Read(), nd, ncol);
   auto U = Reshape(u.Write(), rows, ncol);
   mfem::forall(rows * ncol, [=] MFEM_HOST_DEVICE (int idx)
   {
      const int r = idx % rows, col = idx / rows;
      real_t s = 0.0;
      for (int i = 0; i < nd; ++i) { s += Pm[r + ldp * i] * X(i, col); }
      U(r, col) = s;
   });
}

/** Y(i, col) += sum_r P(r, i) V(r, col); P is (ldp x nd), r < rows. */
inline void SimplexScatterAdd(const int rows, const int ldp, const int nd,
                              const int ncol, const Array<real_t> &P,
                              const Vector &v, Vector &y)
{
   if (ncol <= 0) { return; }
#ifdef MFEM_USE_LAPACK
   if (!Device::Allows(Backend::DEVICE_MASK))
   {
      lapack::Gemm('T', 'N', nd, ncol, rows, real_t(1), P.HostRead(), ldp,
                   v.HostRead(), rows, real_t(1), y.HostReadWrite(), nd);
      return;
   }
#endif
   const real_t *Pm = P.Read();
   const auto V = Reshape(v.Read(), rows, ncol);
   auto Y = Reshape(y.ReadWrite(), nd, ncol);
   mfem::forall(nd * ncol, [=] MFEM_HOST_DEVICE (int idx)
   {
      const int i = idx % nd, col = idx / nd;
      real_t s = 0.0;
      for (int r = 0; r < rows; ++r) { s += Pm[r + ldp * i] * V(r, col); }
      Y(i, col) += s;
   });
}

/** One side of a batched vector product: basis P (ldp x nd, first rows
    rows used) acting on nc E-vector components per element. */
struct SimplexVecGemm
{
   const Array<real_t> *P;
   int rows, ldp, nd, nc;
};

/** Device copy of SimplexVecGemm: P is a device pointer. */
struct SimplexVecSide
{
   const real_t *P;
   int rows, ldp, nd, nc;
};

/** Point functor placeholder for interpolation-only batches. */
struct SimplexVecNoPoint
{
   MFEM_HOST_DEVICE void operator()(int, int, const real_t *, int,
                                    real_t *, int) const { }
};

/** CUDA dyn-smem / host-alloca batch of nb elements: U = P_in X (Gemm),
    V = point(U), Y += P_out^T V (GemmT). The point functor sees one
    element: u(r, c) = u[r + u_ld c] for its nc_in components, and writes
    v(r, c) = v[r + v_ld c] for r = q + nq p. A null out.P stores U into
    Y as (in.rows, in.nc NE) instead (interpolation only). */
template <typename Point>
struct SimplexVecBatchKernel
{
   Point point;
   int NE, nq, nb, x_ld, u_ld, v_ld;
   SimplexVecSide in, out;
   const real_t *X;
   real_t *Y;

   MFEM_HOST_DEVICE void operator()(int batch) const
   {
      const int ci = in.nc * nb, co = out.nc * nb;
#if defined(__CUDA_ARCH__)
      real_t *XS = reinterpret_cast<real_t *>(SimplexMmaDynSmem());
#else
      const int n_smem = (x_ld + u_ld) * ci + v_ld * co;
      real_t *XS = static_cast<real_t *>(alloca(sizeof(real_t) *
                                                static_cast<size_t>(n_smem)));
#endif
      real_t *US = XS + x_ld * ci;
      real_t *VS = US + u_ld * ci;
      const int tid = getThreadIdx();
      const int nthr = getBlockNthreads();
      const int e0 = batch * nb;
      const int c0 = in.nc * e0, ncol = in.nc * NE;
      const auto Xm = ConstDeviceMatrix(X, in.nd, ncol);
      LoadXToSmem(XS, Xm, c0, ncol, in.nd, x_ld, ci, tid, nthr);
      MFEM_SYNC_THREAD;
      Gemm<MmaMapDefault, false>(in.rows, in.nd, ci,
                                 PAcc{in.P, in.ldp, in.nd},
                                 SmemMatAccRt{XS, x_ld},
                                 SmemMatAccRt{US, u_ld}, NullDAcc{}, 0, 0);
      MFEM_SYNC_THREAD;
      if (out.P == nullptr)
      {
         for (int i = tid; i < in.rows * ci; i += nthr)
         {
            const int r = i % in.rows, b = i / in.rows;
            if (c0 + b < ncol) { Y[r + in.rows * (c0 + b)] = US[r + u_ld * b]; }
         }
         return;
      }
      for (int i = tid; i < nq * nb; i += nthr)
      {
         const int q = i % nq, b = i / nq;
         if (e0 + b >= NE) { continue; }
         point(q, e0 + b, US + u_ld * in.nc * b, u_ld,
               VS + v_ld * out.nc * b, v_ld);
      }
      MFEM_SYNC_THREAD;
      // Columns past NE hold stale smem; GemmT never stores them.
      GemmT<MmaMapDefault>(out.rows, out.nd, co,
                           PAcc{out.P, out.ldp, out.nd},
                           SmemMatAccRt{VS, v_ld},
                           YBatchAcc{Y, out.nd, out.nc * e0},
                           out.nc * e0, out.nc * NE);
   }
};

/** Launch SimplexVecBatchKernel; false if one element does not fit. */
template <typename Point>
inline bool SimplexVecDeviceApply(const Point &point, const int NE,
                                  const int nq, const SimplexVecSide &in,
                                  const SimplexVecSide &out,
                                  const real_t *X, real_t *Y)
{
   const int x_ld = PadLdBankRuntime(in.nd);
   const int u_ld = PadLdBankRuntime(in.rows);
   const int v_ld = out.P ? PadLdBankRuntime(out.rows) : 0;
   const int per_e = int(sizeof(real_t)) *
                     ((x_ld + u_ld) * in.nc + v_ld * out.nc);
   // A GEMM covers at most NBATCH = MAX_N_TILES mmaN columns.
   const int nb_max = NBATCH / (in.nc > out.nc ? in.nc : out.nc);
   int cap = SharedMemBytesPerBlock;
   if (Device::Allows(Backend::CUDA_MASK | Backend::HIP_MASK))
   {
      const int dev_cap = Device::SharedMemoryPerBlock();
      if (dev_cap < cap) { cap = dev_cap; }
   }
   // Stay under the default 48 KB unless that shrinks the batch.
   int nb = SharedMemBytesPrefer / per_e;
   if (nb < nb_max) { nb = cap / per_e; }
   if (nb > nb_max) { nb = nb_max; }
   if (nb < 1) { return false; }
   const int smem_bytes = nb * per_e;
   VerifySharedMemBytes(smem_bytes);
   const int nthreads = LaunchNthreads(in.rows, in.nd);
   const int nbatches = (NE + nb - 1) / nb;
   SimplexVecBatchKernel<Point> body
   {
      point, NE, nq, nb, x_ld, u_ld, v_ld, in, out, X, Y};
   mfem::forall_3D_smem(nbatches, nthreads, 1, 1, smem_bytes, body);
   return true;
}

inline SimplexVecSide SimplexVecDevice(const SimplexVecGemm &g)
{
   return SimplexVecSide{g.P ? g.P->Read() : nullptr,
                         g.rows, g.ldp, g.nd, g.nc};
}

/** True when SimplexVec* should take the batched Gemm/GemmT path. */
inline bool SimplexVecUseDevice()
{
#if defined(MFEM_USE_HIP) && !defined(MFEM_USE_CUDA)
   return false; // no dynamic smem launch: dense forall path
#else
   return Device::Allows(Backend::DEVICE_MASK);
#endif
}

/** y += P_out^T point(P_in x) over NE elements; @a u, @a v are host-path
    scratch (in.rows x in.nc NE and out.rows x out.nc NE). */
template <typename Point>
inline void SimplexVecApply(const Point &point, const int NE, const int nq,
                            const SimplexVecGemm &in,
                            const SimplexVecGemm &out, const Vector &x,
                            Vector &u, Vector &v, Vector &y)
{
   if (NE == 0) { return; }
   if (SimplexVecUseDevice() &&
       SimplexVecDeviceApply(point, NE, nq, SimplexVecDevice(in),
                             SimplexVecDevice(out), x.Read(),
                             y.ReadWrite()))
   {
      return;
   }
   const int rin = in.rows, nci = in.nc, rout = out.rows, nco = out.nc;
   u.SetSize(rin * nci * NE);
   v.SetSize(rout * nco * NE);
   u.UseDevice(true);
   v.UseDevice(true);
   SimplexInterp(rin, in.ldp, in.nd, nci * NE, *in.P, x, u);
   const real_t *U = u.Read();
   real_t *V = v.Write();
   mfem::forall(nq * NE, [=] MFEM_HOST_DEVICE (int idx)
   {
      const int q = idx % nq, e = idx / nq;
      point(q, e, U + rin * nci * e, rin, V + rout * nco * e, rout);
   });
   SimplexScatterAdd(rout, out.ldp, out.nd, nco * NE, *out.P, v, y);
}

/** u = P x for all elements (u is in.rows x in.nc NE). */
inline void SimplexVecInterp(const int NE, const SimplexVecGemm &in,
                             const Vector &x, Vector &u)
{
   u.SetSize(in.rows * in.nc * NE);
   u.UseDevice(true);
   if (NE == 0) { return; }
   const SimplexVecSide none{nullptr, 0, 0, 0, 0};
   if (SimplexVecUseDevice() &&
       SimplexVecDeviceApply(SimplexVecNoPoint{}, NE, 0,
                             SimplexVecDevice(in), none, x.Read(),
                             u.Write()))
   {
      return;
   }
   SimplexInterp(in.rows, in.ldp, in.nd, in.nc * NE, *in.P, x, u);
}

} // namespace internal::mma

namespace internal
{

/** Elasticity PA at Q: (w|J| lambda, w|J| mu, J^{-1}), layout (nq, 2+dim^2,
    NE). Without @a lambda: lambda = q_lambda mu and mu = q_mu mu. */
void MmaElasticitySetupSimplex(Mesh &mesh, const IntegrationRule &ir,
                               const Vector &lambda, const Vector &mu,
                               const real_t q_lambda, const real_t q_mu,
                               Vector &d);

/** y += K x, K the elasticity operator; @a G is (nq, dim, nd). */
void MmaElasticityApplySimplex(const int dim, const int NE, const int nq,
                               const int nd, const Array<real_t> &G,
                               const Vector &d, const Vector &x,
                               Vector &u, Vector &v, Vector &y);

void MmaElasticityDiagonalSimplex(const int dim, const int NE, const int nq,
                                  const int nd, const Array<real_t> &G,
                                  const Vector &d, Vector &diag);

/** y += D x (test = scalar B, trial = vector G) or its transpose. */
void MmaVectorDivergenceApplySimplex(const bool transpose, const int dim,
                                     const int NE, const int nq,
                                     const int trial_nd, const int test_nd,
                                     const Array<real_t> &G,
                                     const Array<real_t> &B,
                                     const Vector &d, const Vector &x,
                                     Vector &u, Vector &v, Vector &y);

/** y += N(x) (@a uq empty) or y += N'(u) x, with @a uq = BG u at Q.
    @a BG is (nq, 1+dim, nd), @a d is w Q adj(J). */
void MmaVectorConvectionApplySimplex(const int dim, const int NE,
                                     const int nq, const int nd,
                                     const Array<real_t> &BG,
                                     const Vector &d, const Vector &uq,
                                     const Vector &x, Vector &xq,
                                     Vector &vq, Vector &y);

/** diag(N'(u)) with @a uq = BG u at Q. */
void MmaVectorConvectionGradDiagonalSimplex(const int dim, const int NE,
                                            const int nq, const int nd,
                                            const Array<real_t> &BG,
                                            const Vector &d,
                                            const Vector &uq, Vector &diag);

} // namespace internal

/// \endcond DO_NOT_DOCUMENT

} // namespace mfem
