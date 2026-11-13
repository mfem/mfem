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

#include "../../nonlininteg.hpp"
#include "../../qspace.hpp"
#include "vector.hpp"

namespace mfem
{

namespace internal
{

/** One element at q: x(q + nq p, c) with p = 0 value, p = 1 + a reference
    gradient d/dxhat_a; v(q, c) is the value-plane output. M = w Q adj(J),
    so (M u) . grad_ref(w_c) = w |J| (u . grad) w_c. A null @a uq gives
    N(x) = (M x) . grad x_c, otherwise N'(u) x = (M x) . grad u_c +
    (M u) . grad x_c with uq(q, p, k, e) = BG u. */
template <int DIM>
struct VectorConvectionPoint
{
   const real_t *d, *uq;
   int nq;

   MFEM_HOST_DEVICE void operator()(const int q, const int e,
                                    const real_t *x, const int x_ld,
                                    real_t *v, const int v_ld) const
   {
      constexpr int NP = 1 + DIM;
      const real_t *D = d + q + nq * DIM * DIM * e;
      const real_t *U = uq ? uq + q + nq * NP * DIM * e : nullptr;
      real_t Mu[DIM], Mx[DIM];
      for (int a = 0; a < DIM; ++a)
      {
         real_t su = 0.0, sx = 0.0;
         for (int k = 0; k < DIM; ++k)
         {
            const real_t m = D[nq * (a + DIM * k)];
            sx += m * x[q + x_ld * k];
            if (U) { su += m * U[nq * NP * k]; }
         }
         Mu[a] = su;
         Mx[a] = sx;
      }
      for (int c = 0; c < DIM; ++c)
      {
         real_t s = 0.0;
         for (int a = 0; a < DIM; ++a)
         {
            const real_t xg = x[q + nq * (1 + a) + x_ld * c];
            s += U ? Mx[a] * U[nq * (1 + a + NP * c)] + Mu[a] * xg
                 : Mx[a] * xg;
         }
         v[q + v_ld * c] = s;
      }
   }
};

namespace
{

template <int DIM>
void VectorConvectionGradDiagonal(const int NE, const int NQ, const int ND,
                                  const Array<real_t> &bg, const Vector &d,
                                  const Vector &uq, Vector &diag)
{
   constexpr int NP = 1 + DIM;
   const auto P = Reshape(bg.Read(), NQ, NP, ND);
   const auto D = Reshape(d.Read(), NQ, DIM, DIM, NE);
   const auto U = Reshape(uq.Read(), NQ, NP, DIM, NE);
   auto Y = Reshape(diag.ReadWrite(), ND, DIM, NE);
   mfem::forall(ND * NE, [=] MFEM_HOST_DEVICE (int idx)
   {
      const int i = idx % ND, e = idx / ND;
      real_t acc[DIM];
      for (int c = 0; c < DIM; ++c) { acc[c] = 0.0; }
      for (int q = 0; q < NQ; ++q)
      {
         const real_t b = P(q, 0, i);
         real_t Mug = 0.0;
         for (int a = 0; a < DIM; ++a)
         {
            real_t s = 0.0;
            for (int k = 0; k < DIM; ++k)
            {
               s += D(q, a, k, e) * U(q, 0, k, e);
            }
            Mug += s * P(q, 1 + a, i);
         }
         for (int c = 0; c < DIM; ++c)
         {
            // (M e_c b) . grad u_c + (M u) . grad b, tested with b.
            real_t s = 0.0;
            for (int a = 0; a < DIM; ++a)
            {
               s += D(q, a, c, e) * U(q, 1 + a, c, e);
            }
            acc[c] += b * (b * s + Mug);
         }
      }
      for (int c = 0; c < DIM; ++c) { Y(i, c, e) += acc[c]; }
   });
}

} // namespace

void MmaVectorConvectionApplySimplex(const int dim, const int NE,
                                     const int nq, const int nd,
                                     const Array<real_t> &BG,
                                     const Vector &d, const Vector &uq,
                                     const Vector &x, Vector &xq,
                                     Vector &vq, Vector &y)
{
   const int rows = nq * (1 + dim);
   const mma::SimplexVecGemm in{&BG, rows, rows, nd, dim};
   // Test with values only: the first nq rows of BG.
   const mma::SimplexVecGemm out{&BG, nq, rows, nd, dim};
   const real_t *U = uq.Size() > 0 ? uq.Read() : nullptr;
   if (dim == 2)
   {
      VectorConvectionPoint<2> point{d.Read(), U, nq};
      mma::SimplexVecApply(point, NE, nq, in, out, x, xq, vq, y);
   }
   else
   {
      VectorConvectionPoint<3> point{d.Read(), U, nq};
      mma::SimplexVecApply(point, NE, nq, in, out, x, xq, vq, y);
   }
}

void MmaVectorConvectionGradDiagonalSimplex(const int dim, const int NE,
                                            const int nq, const int nd,
                                            const Array<real_t> &BG,
                                            const Vector &d,
                                            const Vector &uq, Vector &diag)
{
   if (NE == 0) { return; }
   if (dim == 2)
   {
      VectorConvectionGradDiagonal<2>(NE, nq, nd, BG, d, uq, diag);
   }
   else
   {
      VectorConvectionGradDiagonal<3>(NE, nq, nd, BG, d, uq, diag);
   }
}

} // namespace internal

void VectorConvectionNLFIntegrator::AssembleSimplexMmaPA(
   const FiniteElementSpace &fes)
{
   Mesh &mesh = *fes.GetMesh();
   const FiniteElement &el = *fes.GetTypicalFE();
   ElementTransformation &Tr = *mesh.GetTypicalElementTransformation();
   const IntegrationRule *ir = IntRule ? IntRule : &GetRule(el, Tr);
   ne = mesh.GetNE();
   nq = ir->GetNPoints();
   dim = mesh.Dimension();
   simplex_nd = el.GetDof();

   // Match PANonlinearFormExtension, which packs E-vectors with
   // GetEVectorOrdering (NATIVE on simplices). Lexicographic agrees
   // at P1 and diverges at P2+.
   internal::mma::SimplexVectorBasis(fes, *ir, GetEVectorOrdering(fes),
                                     true, true, mma_BG);

   QuadratureSpace qs(mesh, *ir);
   CoefficientVector coeff(Q, qs, CoefficientStorage::FULL);
   internal::mma::SimplexAdjSetup(mesh, *ir, coeff, false, pa_adj);
   mma_uq.Destroy();
   use_simplex_mma = true;
}

void VectorConvectionNLFIntegrator::SetSimplexMmaState(const Vector &u)
{
   const int rows = nq * (1 + dim);
   const internal::mma::SimplexVecGemm in{&mma_BG, rows, rows, simplex_nd,
                                          dim};
   internal::mma::SimplexVecInterp(ne, in, u, mma_uq);
}

} // namespace mfem
