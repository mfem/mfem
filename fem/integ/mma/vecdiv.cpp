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

/** One element at q, D(q, a, c) = w Q adj(J)(a, c) [/ det J].
    Forward: u = grad(x) (rows q + nq a, columns c), v(q) = tr(D^T u).
    Transpose: u(q) = B x, v(q + nq a, c) = D(q, a, c) u(q). */
template <int DIM, bool TRANSPOSE>
struct VectorDivergencePoint
{
   const real_t *d;
   int nq;

   MFEM_HOST_DEVICE void operator()(const int q, const int e,
                                    const real_t *u, const int u_ld,
                                    real_t *v, const int v_ld) const
   {
      const real_t *D = d + q + nq * DIM * DIM * e;
      if constexpr (TRANSPOSE)
      {
         const real_t t = u[q];
         for (int c = 0; c < DIM; ++c)
         {
            for (int a = 0; a < DIM; ++a)
            {
               v[q + nq * a + v_ld * c] = D[nq * (a + DIM * c)] * t;
            }
         }
      }
      else
      {
         real_t s = 0.0;
         for (int c = 0; c < DIM; ++c)
         {
            for (int a = 0; a < DIM; ++a)
            {
               s += D[nq * (a + DIM * c)] * u[q + nq * a + u_ld * c];
            }
         }
         v[q] = s;
      }
   }
};

template <int DIM>
void VectorDivergenceApply(const bool transpose, const int NE, const int nq,
                           const mma::SimplexVecGemm &g,
                           const mma::SimplexVecGemm &b, const Vector &d,
                           const Vector &x, Vector &u, Vector &v, Vector &y)
{
   if (transpose)
   {
      // y += G^T (D (B x)): the test E-vector is the input.
      VectorDivergencePoint<DIM, true> point{d.Read(), nq};
      mma::SimplexVecApply(point, NE, nq, b, g, x, v, u, y);
   }
   else
   {
      VectorDivergencePoint<DIM, false> point{d.Read(), nq};
      mma::SimplexVecApply(point, NE, nq, g, b, x, u, v, y);
   }
}

void MmaVectorDivergenceApplySimplex(const bool transpose, const int dim,
                                     const int NE, const int nq,
                                     const int trial_nd, const int test_nd,
                                     const Array<real_t> &G,
                                     const Array<real_t> &B,
                                     const Vector &d, const Vector &x,
                                     Vector &u, Vector &v, Vector &y)
{
   const mma::SimplexVecGemm g{&G, nq * dim, nq * dim, trial_nd, dim};
   const mma::SimplexVecGemm b{&B, nq, nq, test_nd, 1};
   if (dim == 2)
   {
      VectorDivergenceApply<2>(transpose, NE, nq, g, b, d, x, u, v, y);
   }
   else
   {
      VectorDivergenceApply<3>(transpose, NE, nq, g, b, d, x, u, v, y);
   }
}

} // namespace internal

void VectorDivergenceIntegrator::AssembleSimplexMmaPA(
   const FiniteElementSpace &trial_fes, const FiniteElementSpace &test_fes)
{
   Mesh &mesh = *trial_fes.GetMesh();
   const FiniteElement &trial_fe = *trial_fes.GetTypicalFE();
   const FiniteElement &test_fe = *test_fes.GetTypicalFE();
   const ElementTransformation &T = *mesh.GetTypicalElementTransformation();
   const IntegrationRule *ir = IntRule ? IntRule :
                               &GetRule(trial_fe, test_fe, T);
   dim = mesh.Dimension();
   sdim = mesh.SpaceDimension();
   ne = trial_fes.GetNE();
   nq = ir->GetNPoints();
   simplex_trial_nd = trial_fe.GetDof();
   simplex_test_nd = test_fe.GetDof();

   // PAMixedBilinearFormExtension restricts both spaces lexicographically.
   const ElementDofOrdering lex = ElementDofOrdering::LEXICOGRAPHIC;
   internal::mma::SimplexVectorBasis(trial_fes, *ir, lex, false, true, mma_G);
   internal::mma::SimplexVectorBasis(test_fes, *ir, lex, true, false, mma_B);

   QuadratureSpace qs(mesh, *ir);
   CoefficientVector coeff(Q, qs, CoefficientStorage::FULL);
   // CalcPhysShape scales INTEGRAL-mapped test shapes by 1/|J|.
   const bool inv_det = test_fe.GetMapType() == FiniteElement::INTEGRAL;
   internal::mma::SimplexAdjSetup(mesh, *ir, coeff, inv_det, pa_data);
   use_simplex_mma = true;
}

} // namespace mfem
