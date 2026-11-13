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

#include "../../bilininteg.hpp"
#include "../../qspace.hpp"
#include "vector.hpp"

namespace mfem
{

namespace internal
{

/** V(q,:,j) = J^{-1} sigma_j(J^{-T} U(q,:,:)), scaled by w|J|, for one
    element; rows q + nq a, columns c (see SimplexVecBatchKernel). Named
    namespace: CUDA kernels are instantiated on it. */
template <int DIM>
struct ElasticityPoint
{
   const real_t *d;
   int nq;

   MFEM_HOST_DEVICE void operator()(const int q, const int e,
                                    const real_t *u, const int u_ld,
                                    real_t *v, const int v_ld) const
   {
      constexpr int S = 2 + DIM * DIM;
      const real_t *D = d + q + nq * S * e;
      real_t Ji[DIM][DIM], gp[DIM][DIM];
      for (int b = 0; b < DIM; ++b)
      {
         for (int a = 0; a < DIM; ++a) { Ji[a][b] = D[nq * (2 + a + DIM * b)]; }
      }
      real_t div = 0.0;
      for (int c = 0; c < DIM; ++c)
      {
         for (int k = 0; k < DIM; ++k)
         {
            real_t s = 0.0;
            for (int a = 0; a < DIM; ++a)
            {
               s += Ji[a][k] * u[q + nq * a + u_ld * c];
            }
            gp[c][k] = s;
         }
         div += gp[c][c];
      }
      const real_t lw = D[0] * div, mw = D[nq];
      for (int j = 0; j < DIM; ++j)
      {
         real_t sigma[DIM];
         for (int k = 0; k < DIM; ++k)
         {
            sigma[k] = mw * (gp[j][k] + gp[k][j]) + ((j == k) ? lw : real_t(0));
         }
         for (int a = 0; a < DIM; ++a)
         {
            real_t s = 0.0;
            for (int k = 0; k < DIM; ++k) { s += Ji[a][k] * sigma[k]; }
            v[q + nq * a + v_ld * j] = s;
         }
      }
   }
};

namespace
{

/** J^{-1} (column-major, DIM x DIM) at (q, e); returns det(J). Not inline
    in the forall body: nvcc rejects first captures under if constexpr. */
template <int DIM, typename TE, typename TG>
MFEM_HOST_DEVICE inline real_t SimplexJinv(const TE &E, const TG &G,
                                           const int q, const int e,
                                           const int ND, real_t *Ji)
{
   if constexpr (DIM == 2)
   {
      real_t J11, J21, J12, J22;
      EvalSimplexJ2(E, G, q, e, ND, J11, J21, J12, J22);
      const real_t det = J11 * J22 - J21 * J12;
      Ji[0] = J22 / det;  Ji[2] = -J12 / det;
      Ji[1] = -J21 / det; Ji[3] = J11 / det;
      return det;
   }
   else
   {
      real_t J11, J21, J31, J12, J22, J32, J13, J23, J33;
      EvalSimplexJ3(E, G, q, e, ND,
                    J11, J21, J31, J12, J22, J32, J13, J23, J33);
      real_t C11, C12, C13, C21, C22, C23, C31, C32, C33;
      CofactorsJ3(J11, J21, J31, J12, J22, J32, J13, J23, J33,
                  C11, C12, C13, C21, C22, C23, C31, C32, C33);
      const real_t det = J11 * C11 + J12 * C21 + J13 * C31;
      Ji[0] = C11 / det; Ji[3] = C12 / det; Ji[6] = C13 / det;
      Ji[1] = C21 / det; Ji[4] = C22 / det; Ji[7] = C23 / det;
      Ji[2] = C31 / det; Ji[5] = C32 / det; Ji[8] = C33 / det;
      return det;
   }
}

template <int DIM>
void ElasticitySetup(const int NE, const int NQ, const int ND,
                     const Array<real_t> &w, const Array<real_t> &g,
                     const Vector &nodes_e, const Vector &lambda,
                     const Vector &mu, const real_t q_lambda,
                     const real_t q_mu, Vector &d)
{
   constexpr int S = 2 + DIM * DIM;
   const auto W = Reshape(w.Read(), NQ);
   const auto G = Reshape(g.Read(), NQ, DIM, ND);
   const auto E = Reshape(nodes_e.Read(), ND, DIM, NE);
   const auto L = Reshape(lambda.Read(), NQ, NE);
   const auto M = Reshape(mu.Read(), NQ, NE);
   auto D = Reshape(d.Write(), NQ, S, NE);
   mfem::forall(NQ * NE, [=] MFEM_HOST_DEVICE (int idx)
   {
      const int q = idx % NQ, e = idx / NQ;
      real_t Ji[DIM * DIM];
      const real_t det = SimplexJinv<DIM>(E, G, q, e, ND, Ji);
      const real_t wdet = W(q) * det;
      D(q, 0, e) = wdet * q_lambda * L(q, e);
      D(q, 1, e) = wdet * q_mu * M(q, e);
      for (int k = 0; k < DIM * DIM; ++k) { D(q, 2 + k, e) = Ji[k]; }
   });
}

template <int DIM>
void ElasticityDiagonal(const int NE, const int NQ, const int ND,
                        const Array<real_t> &g, const Vector &d,
                        Vector &diag)
{
   constexpr int S = 2 + DIM * DIM;
   const auto G = Reshape(g.Read(), NQ, DIM, ND);
   const auto D = Reshape(d.Read(), NQ, S, NE);
   auto Y = Reshape(diag.ReadWrite(), ND, DIM, NE);
   mfem::forall(ND * NE, [=] MFEM_HOST_DEVICE (int idx)
   {
      const int i = idx % ND, e = idx / ND;
      real_t acc[DIM];
      for (int c = 0; c < DIM; ++c) { acc[c] = 0.0; }
      for (int q = 0; q < NQ; ++q)
      {
         real_t gp[DIM], g2 = 0.0;
         for (int k = 0; k < DIM; ++k)
         {
            real_t s = 0.0;
            for (int a = 0; a < DIM; ++a)
            {
               s += D(q, 2 + a + DIM * k, e) * G(q, a, i);
            }
            gp[k] = s;
            g2 += s * s;
         }
         const real_t lw = D(q, 0, e), mw = D(q, 1, e);
         for (int c = 0; c < DIM; ++c)
         {
            acc[c] += (lw + mw) * gp[c] * gp[c] + mw * g2;
         }
      }
      for (int c = 0; c < DIM; ++c) { Y(i, c, e) += acc[c]; }
   });
}

} // namespace

void MmaElasticitySetupSimplex(Mesh &mesh, const IntegrationRule &ir,
                               const Vector &lambda, const Vector &mu,
                               const real_t q_lambda, const real_t q_mu,
                               Vector &d)
{
   const int dim = mesh.Dimension();
   const int NE = mesh.GetNE();
   const int NQ = ir.GetNPoints();
   const MemoryType mt = Device::GetDeviceMemoryType();
   d.SetSize(NQ * (2 + dim * dim) * NE, mt);
   if (NE == 0) { return; }
   MFEM_VERIFY(lambda.Size() == NQ * NE && mu.Size() == NQ * NE,
               "Elasticity MMA needs FULL coefficient storage");

   Vector nodes_e;
   const Array<real_t> *G = nullptr;
   int ND = 0;
   GetSimplexSetupGeom(mesh, ir, mt, nodes_e, G, ND);
   const Array<real_t> &w = ir.GetWeights();
   if (dim == 2)
   {
      ElasticitySetup<2>(NE, NQ, ND, w, *G, nodes_e, lambda, mu,
                         q_lambda, q_mu, d);
   }
   else
   {
      MFEM_VERIFY(dim == 3, "Simplex MMA needs dim 2 or 3");
      ElasticitySetup<3>(NE, NQ, ND, w, *G, nodes_e, lambda, mu,
                         q_lambda, q_mu, d);
   }
}

void MmaElasticityApplySimplex(const int dim, const int NE, const int nq,
                               const int nd, const Array<real_t> &G,
                               const Vector &d, const Vector &x,
                               Vector &u, Vector &v, Vector &y)
{
   const int rows = nq * dim;
   const mma::SimplexVecGemm g{&G, rows, rows, nd, dim};
   if (dim == 2)
   {
      mma::SimplexVecApply(ElasticityPoint<2> {d.Read(), nq}, NE, nq, g, g,
                           x, u, v, y);
   }
   else
   {
      mma::SimplexVecApply(ElasticityPoint<3> {d.Read(), nq}, NE, nq, g, g,
                           x, u, v, y);
   }
}

void MmaElasticityDiagonalSimplex(const int dim, const int NE, const int nq,
                                  const int nd, const Array<real_t> &G,
                                  const Vector &d, Vector &diag)
{
   if (NE == 0) { return; }
   if (dim == 2) { ElasticityDiagonal<2>(NE, nq, nd, G, d, diag); }
   else { ElasticityDiagonal<3>(NE, nq, nd, G, d, diag); }
}

} // namespace internal

void ElasticityIntegrator::AssembleSimplexMmaPA(const FiniteElementSpace &fes)
{
   Mesh &mesh = *fes.GetMesh();
   const FiniteElement &el = *fes.GetTypicalFE();
   const IntegrationRule *ir = IntRule;
   if (ir == nullptr)
   {
      // Same default as AssembleElementMatrix and the stock PA path.
      const auto &T = *mesh.GetTypicalElementTransformation();
      ir = &IntRules.Get(T.GetGeometryType(), 2 * T.OrderGrad(&el));
   }
   mma_nq = ir->GetNPoints();

   internal::mma::SimplexVectorBasis(fes, *ir, GetEVectorOrdering(fes),
                                     false, true, mma_G);

   QuadratureSpace qs(mesh, *ir);
   CoefficientVector mu_c(mu, qs, CoefficientStorage::FULL);
   if (lambda)
   {
      CoefficientVector lambda_c(lambda, qs, CoefficientStorage::FULL);
      internal::MmaElasticitySetupSimplex(mesh, *ir, lambda_c, mu_c,
                                          1.0, 1.0, mma_pa);
   }
   else
   {
      internal::MmaElasticitySetupSimplex(mesh, *ir, mu_c, mu_c,
                                          q_lambda, q_mu, mma_pa);
   }
   use_simplex_mma = true;
}

} // namespace mfem
