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

/**
 * @file
 * @brief Header for small strain, isotropic, linear elasticity kernels.
 *
 *        Strong form:    -div(sigma(u))
 *
 *        The constitutive model is given in terms of Lame parameters,
 *        sigma(u) = lambda*div(u)I + 2*mu*sym(grad(u)).
 *        The weak form implemented is (suppressing integral)
 *
 *        Weak form :     lambda*div(u)*div(v) + 2*mu*sym(grad(u))*sym(grad(v))
 *
 *        DATA LAYOUT ASSUMPTIONS :
 *        Finite element space - Ordering::byNODES
 *        Finite element basis - NATIVE for the generic fallback;
 *                               LEXICOGRAPHIC for tensor kernels
 *        Quadrature functions - QVectorLayout::byNODES
 *        All elements in "fespace" are the same.
 */

#ifndef MFEM_BILININTEG_ELASTICITY_KERNELS_HPP
#define MFEM_BILININTEG_ELASTICITY_KERNELS_HPP

#include "../../config/config.hpp"
#include "../../general/array.hpp"
#include "../../general/forall.hpp"
#include "../../linalg/dtensor.hpp"
#include "../../linalg/vector.hpp"
#include "../../linalg/tensor.hpp"
#include "../quadinterpolator.hpp"
#include "../coefficient.hpp"
#include "../kernels.hpp"
#include "../qfunction.hpp"

namespace mfem
{
namespace internal
{

/// Precompute compact elasticity PA data at every element quadrature point.
/// Layout: [invJ(0,0), ..., invJ(d-1,d-1), lambda*detJ*w,
///          mu*detJ*w], with quadrature point as the fastest index.
void ElasticitySetupPAData(const int dim, const IntegrationRule &ir,
                           const CoefficientVector &lambda,
                           const CoefficientVector &mu,
                           const GeometricFactors &geom, Vector &pa_data);

/// Generic reference-gradient fallback, valid for all supported H1 elements.
void ElasticityAddMultPA(const int dim, const int nDofs,
                         const FiniteElementSpace &fespace,
                         const DofToQuad &maps, const Vector &pa_data,
                         const Vector &x, QuadratureFunction &QVec, Vector &y);

void ElasticityAssembleDiagonalPA(const int dim, const int nDofs,
                                  const DofToQuad &maps,
                                  const IntegrationRule &ir,
                                  const Vector &pa_data, Vector &diag);

void ElasticityAssembleEA(const int dim, const int i_block,
                          const int j_block, const int nDofs,
                          const IntegrationRule &ir,
                          const DofToQuad &maps,
                          const Vector &pa_data, Vector &emat,
                          const bool add);


template<int dim>
void ElasticitySetupPAData_(const IntegrationRule &ir,
                            const CoefficientVector &lambda,
                            const CoefficientVector &mu,
                            const GeometricFactors &geom,
                            Vector &pa_data)
{
   using future::inv;
   using future::make_tensor;

   static constexpr int d = dim;
   static constexpr int entries = d*d + 2;
   const int numPoints = ir.GetNPoints();
   const int numEls = lambda.Size()/numPoints;

   const auto lam = Reshape(lambda.Read(), numPoints, numEls);
   const auto muv = Reshape(mu.Read(), numPoints, numEls);
   const auto J = Reshape(geom.J.Read(), numPoints, d, d, numEls);
   const auto detJ = Reshape(geom.detJ.Read(), numPoints, numEls);
   auto D = Reshape(pa_data.Write(), numPoints, entries, numEls);
   const real_t *weights = ir.GetWeights().Read();

   mfem::forall_2D(numEls, numPoints, 1,
                   [=] MFEM_HOST_DEVICE (int e)
   {
      MFEM_FOREACH_THREAD_DIRECT(p, x, numPoints)
      {
         const auto invJ = inv(make_tensor<d, d>(
         [&](int i, int j) { return J(p, i, j, e); }));

         for (int i = 0; i < d; ++i)
         {
            for (int j = 0; j < d; ++j)
            {
               D(p, i*d + j, e) = invJ(i, j);
            }
         }

         const real_t JxW = detJ(p, e)*weights[p];
         D(p, d*d, e) = lam(p, e)*JxW;
         D(p, d*d + 1, e) = muv(p, e)*JxW;
      }
   });
}


template<int dim, int i_block = -1, int j_block = -1>
void ElasticityAddMultPA_(const int nDofs,
                          const FiniteElementSpace &fespace,
                          const DofToQuad &maps,
                          const Vector &pa_data,
                          const Vector &x,
                          QuadratureFunction &QVec,
                          Vector &y)
{
   using future::make_tensor;
   using future::tensor;

   static_assert((i_block < 0) == (j_block < 0),
                 "i_block and j_block must both be set or both omitted.");
   static constexpr int d = dim;
   static constexpr int qSize = i_block < 0 ? d : 1;
   static constexpr bool component = i_block >= 0;
   static constexpr int entries = d*d + 2;

   const auto &ir = QVec.GetIntRule(0);
   const QuadratureInterpolator *EToQ =
      fespace.GetQuadratureInterpolator(ir);
   EToQ->SetOutputLayout(QVectorLayout::byNODES);

   // Reference derivatives avoid the geometric inverse performed by
   // PhysDerivatives. The compact PA data supplies inv(J) below.
   EToQ->Derivatives(x, QVec);

   const int numPoints = ir.GetNPoints();
   const int numEls = fespace.GetNE();
   const int ntx = nDofs > numPoints ? nDofs : numPoints;
   const int nqad = numPoints * qSize * d * numEls;
   auto *q_mem = QVec.ReadWrite();
   const auto Q = Reshape(q_mem, numPoints, qSize, d, numEls);
   auto Flux = Reshape(q_mem + nqad, numPoints, qSize, d, numEls);
   const auto D = Reshape(pa_data.Read(), numPoints, entries, numEls);
   const auto G = Reshape(maps.G.Read(), numPoints, d, nDofs);
   auto Y = Reshape(y.ReadWrite(), nDofs, qSize, numEls);

   // Constitutive evaluation and G^T reduction in one launch. Tensor-product
   // elements normally bypass this path through ApplyPAKernels.
   mfem::forall_2D(numEls, ntx, qSize,
                   [=] MFEM_HOST_DEVICE (int e)
   {
      MFEM_FOREACH_THREAD_DIRECT(p, x, numPoints)
      {
         MFEM_FOREACH_THREAD_DIRECT(a, y, qSize)
         {
            const auto invJ = make_tensor<d, d>([&](int i, int j)
            {
               return D(p, i*d + j, e);
            });
            const real_t alpha = D(p, d*d, e);
            const real_t beta = D(p, d*d + 1, e);
            // NVCC extended lambdas cannot first-capture a variable inside
            // if constexpr; touch Q and Flux here so both branches can use them.
            MFEM_CONTRACT_VAR(Q);
            MFEM_CONTRACT_VAR(Flux);

            if constexpr (component)
            {
               tensor<real_t, d> dudi;
               tensor<real_t, d> grad;
               for (int m = 0; m < d; ++m)
               {
                  dudi(m) = Q(p, 0, m, e);
               }
               for (int b = 0; b < d; ++b)
               {
                  grad(b) = 0.0;
                  for (int m = 0; m < d; ++m)
                  {
                     grad(b) += dudi(m)*invJ(m, b);
                  }
               }

               for (int m = 0; m < d; ++m)
               {
                  real_t flux = alpha*invJ(m, i_block)*grad(j_block)
                                + beta*invJ(m, j_block)*grad(i_block);
                  if constexpr (i_block == j_block)
                  {
                     for (int b = 0; b < d; ++b)
                     {
                        flux += beta*invJ(m, b)*grad(b);
                     }
                  }
                  Flux(p, 0, m, e) = flux;
               }
            }
            else
            {
               tensor<real_t, d> grad_row;
               tensor<real_t, d> grad_col;
               for (int b = 0; b < d; ++b)
               {
                  real_t row = 0.0;
                  real_t col = 0.0;
                  for (int m = 0; m < d; ++m)
                  {
                     row += Q(p, a, m, e)*invJ(m, b);
                     col += Q(p, b, m, e)*invJ(m, a);
                  }
                  grad_row(b) = row;
                  grad_col(b) = col;
               }

               real_t div_u = 0.0;
               for (int c = 0; c < d; ++c)
               {
                  for (int m = 0; m < d; ++m)
                  {
                     div_u += Q(p, c, m, e)*invJ(m, c);
                  }
               }

               for (int m = 0; m < d; ++m)
               {
                  real_t flux = 0.0;
                  for (int b = 0; b < d; ++b)
                  {
                     real_t sw = beta*(grad_row(b) + grad_col(b));
                     if (a == b)
                     {
                        sw += alpha*div_u;
                     }
                     flux += invJ(m, b)*sw;
                  }
                  Flux(p, a, m, e) = flux;
               }
            }
         }
      }

      MFEM_SYNC_THREAD;

      MFEM_FOREACH_THREAD_DIRECT(i, x, nDofs)
      {
         MFEM_FOREACH_THREAD_DIRECT(a, y, qSize)
         {
            real_t value = 0.0;
            for (int m = 0; m < d; ++m)
            {
               for (int p = 0; p < numPoints; ++p)
               {
                  value += Flux(p, a, m, e)*G(p, m, i);
               }
            }
            Y(i, a, e) += value;
         }
      }
   });
}


template<int dim>
void ElasticityAssembleDiagonalPA_(const int nDofs,
                                   const DofToQuad &maps,
                                   const IntegrationRule &ir,
                                   const Vector &pa_data,
                                   Vector &diag)
{
   using future::make_tensor;
   using future::tensor;

   static constexpr int d = dim;
   static constexpr int entries = d*d + 2;
   const int numPoints = ir.GetNPoints();
   const int numEls = pa_data.Size()/(entries*numPoints);
   const auto D = Reshape(pa_data.Read(), numPoints, entries, numEls);
   const auto G = Reshape(maps.G.Read(), numPoints, d, nDofs);
   auto diagDev = Reshape(diag.ReadWrite(), nDofs, d, numEls);

   mfem::forall_2D(numEls, nDofs, 1,
                   [=] MFEM_HOST_DEVICE (int e)
   {
      MFEM_FOREACH_THREAD_DIRECT(i, x, nDofs)
      {
         for (int p = 0; p < numPoints; ++p)
         {
            const auto invJ = make_tensor<d, d>([&](int r, int c)
            {
               return D(p, r*d + c, e);
            });
            tensor<real_t, d> grad;
            for (int b = 0; b < d; ++b)
            {
               grad(b) = 0.0;
               for (int m = 0; m < d; ++m)
               {
                  grad(b) += G(p, m, i)*invJ(m, b);
               }
            }

            real_t norm2 = 0.0;
            for (int b = 0; b < d; ++b)
            {
               norm2 += grad(b)*grad(b);
            }
            const real_t alpha = D(p, d*d, e);
            const real_t beta = D(p, d*d + 1, e);
            for (int a = 0; a < d; ++a)
            {
               diagDev(i, a, e) += beta*norm2 + (alpha + beta)*grad(a)*grad(a);
            }
         }
      }
   });
}


template<int dim>
void ElasticityAssembleEA_(const int i_block,
                           const int j_block,
                           const int nDofs,
                           const IntegrationRule &ir,
                           const DofToQuad &maps,
                           const Vector &pa_data,
                           Vector &emat,
                           const bool add)
{
   using future::make_tensor;

   static constexpr int d = dim;
   static constexpr int entries = d*d + 2;
   const int numPoints = ir.GetNPoints();
   const int numEls = pa_data.Size()/(entries*numPoints);
   const auto D = Reshape(pa_data.Read(), numPoints, entries, numEls);
   const auto G = Reshape(maps.G.Read(), numPoints, d, nDofs);
   auto E = Reshape(add ? emat.ReadWrite() : emat.Write(),
                    nDofs, nDofs, numEls);

   Vector phys_grad;
   phys_grad.UseDevice(true);
   phys_grad.SetSize(numEls * numPoints * d * nDofs);
   auto g = Reshape(phys_grad.Write(), numPoints, d, nDofs, numEls);

   mfem::forall_2D(numEls, nDofs, 1,
                   [=] MFEM_HOST_DEVICE (int e)
   {
      MFEM_FOREACH_THREAD_DIRECT(i, x, nDofs)
      {
         for (int p = 0; p < numPoints; ++p)
         {
            const auto invJ = make_tensor<d, d>([&](int r, int c)
            {
               return D(p, r*d + c, e);
            });
            for (int b = 0; b < d; ++b)
            {
               real_t gi = 0.0;
               for (int m = 0; m < d; ++m)
               {
                  gi += G(p, m, i)*invJ(m, b);
               }
               g(p, b, i, e) = gi;
            }
         }
      }
   });

   const auto gR = Reshape(phys_grad.Read(), numPoints, d, nDofs, numEls);

   mfem::forall_2D(numEls, nDofs, nDofs,
                   [=] MFEM_HOST_DEVICE (int e)
   {
      MFEM_FOREACH_THREAD_DIRECT(trial, y, nDofs)
      {
         MFEM_FOREACH_THREAD_DIRECT(test, x, nDofs)
         {
            real_t sum = 0.0;
            for (int p = 0; p < numPoints; ++p)
            {
               real_t dot = 0.0;
               for (int b = 0; b < d; ++b)
               {
                  dot += gR(p, b, trial, e)*gR(p, b, test, e);
               }
               const real_t alpha = D(p, d*d, e);
               const real_t beta = D(p, d*d + 1, e);
               sum += alpha*gR(p, j_block, trial, e)*gR(p, i_block, test, e)
                      + beta*gR(p, i_block, trial, e)*gR(p, j_block, test, e);
               if (i_block == j_block)
               {
                  sum += beta*dot;
               }
            }

            // MFEM E-matrices use (trial dof, test dof, element).
            if (add)
            {
               E(trial, test, e) += sum;
            }
            else
            {
               E(trial, test, e) = sum;
            }
         }
      }
   });
}


// ---------------------------------------------------------------------------
// Fused tensor-product kernels
// ---------------------------------------------------------------------------

template<int D1D, int Q1D>
void ElasticityAddMultPATensor2D_(const int numEls,
                                  const DofToQuad &maps,
                                  const Vector &pa_data,
                                  const Vector &x,
                                  Vector &y)
{
   using future::tensor;
   static constexpr int d = 2;
   static constexpr int entries = 6;

   const auto b = Reshape(maps.B.Read(), Q1D, D1D);
   const auto g = Reshape(maps.G.Read(), Q1D, D1D);

   const auto D = Reshape(pa_data.Read(), Q1D, Q1D, entries, numEls);
   const auto X = Reshape(x.Read(), D1D, D1D, d, numEls);
   auto Y = Reshape(y.ReadWrite(), D1D, D1D, d, numEls);

   mfem::forall_2D(numEls, Q1D, Q1D,
                   [=] MFEM_HOST_DEVICE (int e)
   {
      MFEM_SHARED tensor<real_t, Q1D, D1D> B;
      MFEM_SHARED tensor<real_t, Q1D, D1D> G;
      MFEM_SHARED tensor<real_t, 2, Q1D, D1D> S;
      MFEM_SHARED tensor<real_t, Q1D, Q1D, d, d> Q;

      MFEM_FOREACH_THREAD_DIRECT(i, y, D1D)
      {
         MFEM_FOREACH_THREAD_DIRECT(q, x, Q1D)
         {
            B(q, i) = b(q, i);
            G(q, i) = g(q, i);
         }
      }
      MFEM_SYNC_THREAD;

      // Reference gradients: sum factorization in x, then y.
      for (int c = 0; c < d; ++c)
      {
         MFEM_FOREACH_THREAD_DIRECT(dy, y, D1D)
         {
            MFEM_FOREACH_THREAD_DIRECT(qx, x, Q1D)
            {
               real_t value = 0.0;
               real_t deriv = 0.0;
               for (int dx = 0; dx < D1D; ++dx)
               {
                  const real_t u = X(dx, dy, c, e);
                  value += B(qx, dx)*u;
                  deriv += G(qx, dx)*u;
               }
               S(0, qx, dy) = value;
               S(1, qx, dy) = deriv;
            }
         }
         MFEM_SYNC_THREAD;

         MFEM_FOREACH_THREAD_DIRECT(qy, y, Q1D)
         {
            MFEM_FOREACH_THREAD_DIRECT(qx, x, Q1D)
            {
               real_t du0 = 0.0;
               real_t du1 = 0.0;
               for (int dy = 0; dy < D1D; ++dy)
               {
                  du0 += S(1, qx, dy)*B(qy, dy);
                  du1 += S(0, qx, dy)*G(qy, dy);
               }
               Q(qx, qy, c, 0) = du0;
               Q(qx, qy, c, 1) = du1;
            }
         }
         MFEM_SYNC_THREAD;
      }

      // Constitutive operation and Piola pullback, in place.
      MFEM_FOREACH_THREAD_DIRECT(qy, y, Q1D)
      {
         MFEM_FOREACH_THREAD_DIRECT(qx, x, Q1D)
         {
            tensor<real_t, d, d> invJ;
            tensor<real_t, d, d> dudi;
            tensor<real_t, d, d> grad;
            tensor<real_t, d, d> sigma_w;
            for (int i = 0; i < d; ++i)
            {
               for (int j = 0; j < d; ++j)
               {
                  invJ(i, j) = D(qx, qy, i*d + j, e);
                  dudi(i, j) = Q(qx, qy, i, j);
               }
            }
            for (int a = 0; a < d; ++a)
            {
               for (int b = 0; b < d; ++b)
               {
                  grad(a, b) = 0.0;
                  for (int m = 0; m < d; ++m)
                  {
                     grad(a, b) += dudi(a, m)*invJ(m, b);
                  }
               }
            }
            const real_t alpha = D(qx, qy, d*d, e);
            const real_t beta = D(qx, qy, d*d + 1, e);
            const real_t div_u = grad(0, 0) + grad(1, 1);
            for (int a = 0; a < d; ++a)
            {
               for (int b = 0; b < d; ++b)
               {
                  sigma_w(a, b) = beta*(grad(a, b) + grad(b, a));
                  if (a == b) { sigma_w(a, b) += alpha*div_u; }
               }
            }
            for (int m = 0; m < d; ++m)
            {
               for (int a = 0; a < d; ++a)
               {
                  real_t flux = 0.0;
                  for (int b = 0; b < d; ++b)
                  {
                     flux += invJ(m, b)*sigma_w(a, b);
                  }
                  Q(qx, qy, m, a) = flux;
               }
            }
         }
      }
      MFEM_SYNC_THREAD;

      // Transposed gradient: x contraction, then y contraction.
      for (int c = 0; c < d; ++c)
      {
         MFEM_FOREACH_THREAD_DIRECT(qy, y, Q1D)
         {
            MFEM_FOREACH_THREAD_DIRECT(dx, x, D1D)
            {
               real_t r0 = 0.0;
               real_t r1 = 0.0;
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  r0 += G(qx, dx)*Q(qx, qy, 0, c);
                  r1 += B(qx, dx)*Q(qx, qy, 1, c);
               }
               S(0, qy, dx) = r0;
               S(1, qy, dx) = r1;
            }
         }
         MFEM_SYNC_THREAD;

         MFEM_FOREACH_THREAD_DIRECT(dy, y, D1D)
         {
            MFEM_FOREACH_THREAD_DIRECT(dx, x, D1D)
            {
               real_t value = 0.0;
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  value += B(qy, dy)*S(0, qy, dx)
                           + G(qy, dy)*S(1, qy, dx);
               }
               Y(dx, dy, c, e) += value;
            }
         }
         MFEM_SYNC_THREAD;
      }
   });
}


/// Static shared memory available to one thread block on every supported
/// device: 48 KiB is the CUDA limit for statically sized __shared__ arrays
/// (HIP allows 64 KiB), and larger blocks would need dynamic shared memory.
constexpr int ElasticityMaxStaticSharedBytes = 48*1024;

/// Bytes of shared memory used by ElasticityAddMultPATensor3D_: the B and G
/// matrices, three Q1D^3 work arrays, the 3x3 gradient/flux tensor at every
/// quadrature point, and a 3*D1D^2*Q1D transpose scratch tensor.
constexpr int ElasticityTensor3DSharedBytes(const int D1D, const int Q1D)
{
   return static_cast<int>(sizeof(real_t)) *
          (2*Q1D*D1D + 3*Q1D*Q1D*Q1D + 9*Q1D*Q1D*Q1D + 3*D1D*D1D*Q1D);
}

/// Whether ElasticityAddMultPATensor3D_ fits in static shared memory; other
/// (D1D, Q1D) pairs use ElasticityAddMultPATensor3DRegs_.
constexpr bool ElasticityTensor3DFitsShared(const int D1D, const int Q1D)
{
   return ElasticityTensor3DSharedBytes(D1D, Q1D) <=
          ElasticityMaxStaticSharedBytes;
}

template<int D1D, int Q1D>
void ElasticityAddMultPATensor3D_(const int numEls,
                                  const DofToQuad &maps,
                                  const Vector &pa_data,
                                  const Vector &x,
                                  Vector &y)
{
   using future::tensor;
   static constexpr int d = 3;
   static constexpr int entries = 11;
   static_assert(ElasticityTensor3DFitsShared(D1D, Q1D),
                 "Use ElasticityAddMultPATensor3DRegs_ for this (D1D, Q1D).");

   const auto b = Reshape(maps.B.Read(), Q1D, D1D);
   const auto g = Reshape(maps.G.Read(), Q1D, D1D);

   const auto D = Reshape(pa_data.Read(), Q1D, Q1D, Q1D, entries, numEls);
   const auto X = Reshape(x.Read(), D1D, D1D, D1D, d, numEls);
   auto Y = Reshape(y.ReadWrite(), D1D, D1D, D1D, d, numEls);

   mfem::forall_3D(numEls, Q1D, Q1D, Q1D,
                   [=] MFEM_HOST_DEVICE (int e)
   {
      // Shared data consist of the one-dimensional basis maps, three Q1D^3
      // work arrays, the 3x3 q-point gradient/flux tensor, and a
      // 3*D1D^2*Q1D transpose scratch tensor, see
      // ElasticityTensor3DSharedBytes (about 42 KiB at D1D=Q1D=7 in double
      // precision, 72 KiB at D1D=4, Q1D=9).
      MFEM_SHARED tensor<real_t, Q1D, D1D> B;
      MFEM_SHARED tensor<real_t, Q1D, D1D> G;
      MFEM_SHARED tensor<real_t, 3, Q1D, Q1D, Q1D> S;
      MFEM_SHARED tensor<real_t, Q1D, Q1D, Q1D, d, d> Q;
      MFEM_SHARED tensor<real_t, 3, D1D, D1D, Q1D> T;

      if (MFEM_THREAD_ID(z) == 0)
      {
         MFEM_FOREACH_THREAD_DIRECT(i, y, D1D)
         {
            MFEM_FOREACH_THREAD_DIRECT(q, x, Q1D)
            {
               B(q, i) = b(q, i);
               G(q, i) = g(q, i);
            }
         }
      }
      MFEM_SYNC_THREAD;

      for (int c = 0; c < d; ++c)
      {
         // x contraction: B and G.
         MFEM_FOREACH_THREAD_DIRECT(dz, z, D1D)
         {
            MFEM_FOREACH_THREAD_DIRECT(dy, y, D1D)
            {
               MFEM_FOREACH_THREAD_DIRECT(qx, x, Q1D)
               {
                  real_t value = 0.0;
                  real_t deriv = 0.0;
                  for (int dx = 0; dx < D1D; ++dx)
                  {
                     const real_t u = X(dx, dy, dz, c, e);
                     value += B(qx, dx)*u;
                     deriv += G(qx, dx)*u;
                  }
                  S(0, qx, dy, dz) = value;
                  S(1, qx, dy, dz) = deriv;
               }
            }
         }
         MFEM_SYNC_THREAD;

         // y contraction. Q temporarily stores data indexed by dz.
         MFEM_FOREACH_THREAD_DIRECT(dz, z, D1D)
         {
            MFEM_FOREACH_THREAD_DIRECT(qy, y, Q1D)
            {
               MFEM_FOREACH_THREAD_DIRECT(qx, x, Q1D)
               {
                  real_t du0 = 0.0;
                  real_t du1 = 0.0;
                  real_t value = 0.0;
                  for (int dy = 0; dy < D1D; ++dy)
                  {
                     du0 += S(1, qx, dy, dz)*B(qy, dy);
                     du1 += S(0, qx, dy, dz)*G(qy, dy);
                     value += S(0, qx, dy, dz)*B(qy, dy);
                  }
                  Q(qx, qy, dz, c, 0) = du0;
                  Q(qx, qy, dz, c, 1) = du1;
                  Q(qx, qy, dz, c, 2) = value;
               }
            }
         }
         MFEM_SYNC_THREAD;

         // z contraction into S.
         MFEM_FOREACH_THREAD_DIRECT(qz, z, Q1D)
         {
            MFEM_FOREACH_THREAD_DIRECT(qy, y, Q1D)
            {
               MFEM_FOREACH_THREAD_DIRECT(qx, x, Q1D)
               {
                  real_t du0 = 0.0;
                  real_t du1 = 0.0;
                  real_t du2 = 0.0;
                  for (int dz = 0; dz < D1D; ++dz)
                  {
                     du0 += Q(qx, qy, dz, c, 0)*B(qz, dz);
                     du1 += Q(qx, qy, dz, c, 1)*B(qz, dz);
                     du2 += Q(qx, qy, dz, c, 2)*G(qz, dz);
                  }
                  S(0, qx, qy, qz) = du0;
                  S(1, qx, qy, qz) = du1;
                  S(2, qx, qy, qz) = du2;
               }
            }
         }
         MFEM_SYNC_THREAD;

         MFEM_FOREACH_THREAD_DIRECT(qz, z, Q1D)
         {
            MFEM_FOREACH_THREAD_DIRECT(qy, y, Q1D)
            {
               MFEM_FOREACH_THREAD_DIRECT(qx, x, Q1D)
               {
                  for (int m = 0; m < d; ++m)
                  {
                     Q(qx, qy, qz, c, m) = S(m, qx, qy, qz);
                  }
               }
            }
         }
         MFEM_SYNC_THREAD;
      }

      // Constitutive operation and reference-space flux.
      MFEM_FOREACH_THREAD_DIRECT(qz, z, Q1D)
      {
         MFEM_FOREACH_THREAD_DIRECT(qy, y, Q1D)
         {
            MFEM_FOREACH_THREAD_DIRECT(qx, x, Q1D)
            {
               tensor<real_t, d, d> invJ;
               tensor<real_t, d, d> dudi;
               tensor<real_t, d, d> grad;
               tensor<real_t, d, d> sigma_w;
               for (int i = 0; i < d; ++i)
               {
                  for (int j = 0; j < d; ++j)
                  {
                     invJ(i, j) = D(qx, qy, qz, i*d + j, e);
                     dudi(i, j) = Q(qx, qy, qz, i, j);
                  }
               }
               for (int a = 0; a < d; ++a)
               {
                  for (int b = 0; b < d; ++b)
                  {
                     grad(a, b) = 0.0;
                     for (int m = 0; m < d; ++m)
                     {
                        grad(a, b) += dudi(a, m)*invJ(m, b);
                     }
                  }
               }
               const real_t alpha = D(qx, qy, qz, d*d, e);
               const real_t beta = D(qx, qy, qz, d*d + 1, e);
               const real_t div_u = grad(0, 0) + grad(1, 1) + grad(2, 2);
               for (int a = 0; a < d; ++a)
               {
                  for (int b = 0; b < d; ++b)
                  {
                     sigma_w(a, b) = beta*(grad(a, b) + grad(b, a));
                     if (a == b) { sigma_w(a, b) += alpha*div_u; }
                  }
               }
               for (int m = 0; m < d; ++m)
               {
                  for (int a = 0; a < d; ++a)
                  {
                     real_t flux = 0.0;
                     for (int b = 0; b < d; ++b)
                     {
                        flux += invJ(m, b)*sigma_w(a, b);
                     }
                     Q(qx, qy, qz, m, a) = flux;
                  }
               }
            }
         }
      }
      MFEM_SYNC_THREAD;

      for (int c = 0; c < d; ++c)
      {
         // x transpose.
         MFEM_FOREACH_THREAD_DIRECT(qz, z, Q1D)
         {
            MFEM_FOREACH_THREAD_DIRECT(qy, y, Q1D)
            {
               MFEM_FOREACH_THREAD_DIRECT(dx, x, D1D)
               {
                  real_t r0 = 0.0;
                  real_t r1 = 0.0;
                  real_t r2 = 0.0;
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     r0 += G(qx, dx)*Q(qx, qy, qz, 0, c);
                     r1 += B(qx, dx)*Q(qx, qy, qz, 1, c);
                     r2 += B(qx, dx)*Q(qx, qy, qz, 2, c);
                  }
                  S(0, dx, qy, qz) = r0;
                  S(1, dx, qy, qz) = r1;
                  S(2, dx, qy, qz) = r2;
               }
            }
         }
         MFEM_SYNC_THREAD;

         // y transpose. Use dedicated scratch so that processing one output
         // component cannot overwrite flux data needed by later components.
         MFEM_FOREACH_THREAD_DIRECT(qz, z, Q1D)
         {
            MFEM_FOREACH_THREAD_DIRECT(dy, y, D1D)
            {
               MFEM_FOREACH_THREAD_DIRECT(dx, x, D1D)
               {
                  real_t r0 = 0.0;
                  real_t r1 = 0.0;
                  real_t r2 = 0.0;
                  for (int qy = 0; qy < Q1D; ++qy)
                  {
                     r0 += B(qy, dy)*S(0, dx, qy, qz);
                     r1 += G(qy, dy)*S(1, dx, qy, qz);
                     r2 += B(qy, dy)*S(2, dx, qy, qz);
                  }
                  T(0, dx, dy, qz) = r0;
                  T(1, dx, dy, qz) = r1;
                  T(2, dx, dy, qz) = r2;
               }
            }
         }
         MFEM_SYNC_THREAD;

         // z transpose and accumulation.
         MFEM_FOREACH_THREAD_DIRECT(dz, z, D1D)
         {
            MFEM_FOREACH_THREAD_DIRECT(dy, y, D1D)
            {
               MFEM_FOREACH_THREAD_DIRECT(dx, x, D1D)
               {
                  real_t value = 0.0;
                  for (int qz = 0; qz < Q1D; ++qz)
                  {
                     value += B(qz, dz)*T(0, dx, dy, qz)
                              + B(qz, dz)*T(1, dx, dy, qz)
                              + G(qz, dz)*T(2, dx, dy, qz);
                  }
                  Y(dx, dy, dz, c, e) += value;
               }
            }
         }
         MFEM_SYNC_THREAD;
      }
   });
}

/** @brief Register-tiled tensor-product elasticity action in 3D, y += K x.

    Used for the (D1D, Q1D) pairs where ElasticityAddMultPATensor3D_ would
    exceed ElasticityMaxStaticSharedBytes, e.g. D1D = 4, Q1D = 9.

    The threads of an element form a Q1D x Q1D tile in (x, y), and each
    thread keeps its column of Q1D quadrature points along z in registers
    (see kernels::internal::s_regs3d_t). Shared memory only holds the 1D
    basis matrices and one Q1D x Q1D contraction slice, so its footprint,
    (2*D1D + Q1D)*Q1D values, grows quadratically with Q1D instead of
    cubically. The price is more contractions and synchronizations, which
    is why the shared-memory kernel is preferred whenever it fits.

    The action proceeds in three stages:
    1. reference gradients dU(c,m) = d u_c / d xi_m of every displacement
       component, by sum factorization with G along xi_m and B elsewhere;
    2. at each quadrature point, the physical gradient grad = dU J^{-1}, the
       weighted stress sigma_w = lambda_w div(u) I + mu_w (grad + grad^T),
       and the reference flux F(a,m) = sum_b J^{-1}(m,b) sigma_w(a,b);
    3. for every output component a, the transposed contraction of F(a,m)
       with G along xi_m and B elsewhere, summed over m. */
template<int D1D, int Q1D>
void ElasticityAddMultPATensor3DRegs_(const int numEls,
                                      const DofToQuad &maps,
                                      const Vector &pa_data,
                                      const Vector &x,
                                      Vector &y)
{
   using future::tensor;
   using kernels::internal::s_regs3d_t;
   using kernels::internal::vd_regs3d_t;
   using kernels::internal::Contract3d;
   static constexpr int d = 3;
   static constexpr int entries = 11;
   static_assert(D1D <= Q1D, "The thread tile is sized by Q1D.");

   const real_t *b = maps.B.Read();
   const real_t *g = maps.G.Read();

   const auto D = Reshape(pa_data.Read(), Q1D, Q1D, Q1D, entries, numEls);
   const auto X = Reshape(x.Read(), D1D, D1D, D1D, d, numEls);
   auto Y = Reshape(y.ReadWrite(), D1D, D1D, D1D, d, numEls);

   mfem::forall_2D<Q1D*Q1D>(numEls, Q1D, Q1D,
                            [=] MFEM_HOST_DEVICE (int e)
   {
      constexpr int MD1 = kernels::internal::SetMaxOf(D1D);
      constexpr int MQ1 = kernels::internal::SetMaxOf(Q1D);

      // Basis matrices are stored transposed: sB[dof][qpt] = B(qpt, dof).
      MFEM_SHARED real_t sB[MD1][MQ1], sG[MD1][MQ1], smem[MQ1][MQ1];
      kernels::internal::LoadMatrix(D1D, Q1D, b, sB);
      kernels::internal::LoadMatrix(D1D, Q1D, g, sG);

      // dU(c, m) at the quadrature points; overwritten by the flux F(c, m).
      vd_regs3d_t<d, d, MQ1> dU;
      // Contraction input/scratch and output-component accumulator.
      s_regs3d_t<MQ1> r0, r1;
      // The loops over components, directions and z points are unrolled so
      // that every register tensor index is a compile-time constant on the
      // device; otherwise dU would be placed in local (stack) memory.

      // 1. Reference gradients. Contract3d consumes its input r0, so the
      //    degrees of freedom of component c are reloaded for each m.
      MFEM_UNROLL(3)
      for (int c = 0; c < d; ++c)
      {
         MFEM_UNROLL(3)
         for (int m = 0; m < d; ++m)
         {
            for (int dz = 0; dz < D1D; ++dz)
            {
               MFEM_FOREACH_THREAD_DIRECT(dy, y, D1D)
               {
                  MFEM_FOREACH_THREAD_DIRECT(dx, x, D1D)
                  {
                     r0[dz][dy][dx] = X(dx, dy, dz, c, e);
                  }
               }
            }
            Contract3d<false>(D1D, Q1D, smem,
                              m == 0 ? sG : sB,
                              m == 1 ? sG : sB,
                              m == 2 ? sG : sB,
                              r0, dU[c][m]);
         }
      }

      // 2. Constitutive operation and reference-space flux, pointwise.
      MFEM_UNROLL(Q1D)
      for (int qz = 0; qz < Q1D; ++qz)
      {
         MFEM_FOREACH_THREAD_DIRECT(qy, y, Q1D)
         {
            MFEM_FOREACH_THREAD_DIRECT(qx, x, Q1D)
            {
               tensor<real_t, d, d> invJ, grad, sigma_w;
               for (int i = 0; i < d; ++i)
               {
                  for (int j = 0; j < d; ++j)
                  {
                     invJ(i, j) = D(qx, qy, qz, i*d + j, e);
                  }
               }
               for (int a = 0; a < d; ++a)
               {
                  for (int j = 0; j < d; ++j)
                  {
                     grad(a, j) = 0.0;
                     for (int m = 0; m < d; ++m)
                     {
                        grad(a, j) += dU[a][m][qz][qy][qx]*invJ(m, j);
                     }
                  }
               }
               const real_t alpha = D(qx, qy, qz, d*d, e);
               const real_t beta = D(qx, qy, qz, d*d + 1, e);
               const real_t div_u = grad(0, 0) + grad(1, 1) + grad(2, 2);
               for (int a = 0; a < d; ++a)
               {
                  for (int j = 0; j < d; ++j)
                  {
                     sigma_w(a, j) = beta*(grad(a, j) + grad(j, a));
                     if (a == j) { sigma_w(a, j) += alpha*div_u; }
                  }
               }
               for (int a = 0; a < d; ++a)
               {
                  for (int m = 0; m < d; ++m)
                  {
                     real_t flux = 0.0;
                     for (int j = 0; j < d; ++j)
                     {
                        flux += invJ(m, j)*sigma_w(a, j);
                     }
                     dU[a][m][qz][qy][qx] = flux;
                  }
               }
            }
         }
      }

      // 3. Transposed contractions, accumulated per output component.
      //    Contract3d<true> consumes the flux dU(a, m) and returns r0.
      MFEM_UNROLL(3)
      for (int a = 0; a < d; ++a)
      {
         MFEM_UNROLL(3)
         for (int m = 0; m < d; ++m)
         {
            Contract3d<true>(D1D, Q1D, smem,
                             m == 0 ? sG : sB,
                             m == 1 ? sG : sB,
                             m == 2 ? sG : sB,
                             dU[a][m], r0);
            for (int dz = 0; dz < D1D; ++dz)
            {
               MFEM_FOREACH_THREAD_DIRECT(dy, y, D1D)
               {
                  MFEM_FOREACH_THREAD_DIRECT(dx, x, D1D)
                  {
                     const real_t prev = (m == 0) ? 0.0 : r1[dz][dy][dx];
                     r1[dz][dy][dx] = prev + r0[dz][dy][dx];
                  }
               }
            }
         }
         for (int dz = 0; dz < D1D; ++dz)
         {
            MFEM_FOREACH_THREAD_DIRECT(dy, y, D1D)
            {
               MFEM_FOREACH_THREAD_DIRECT(dx, x, D1D)
               {
                  Y(dx, dy, dz, a, e) += r1[dz][dy][dx];
               }
            }
         }
      }
   });
}

} // namespace internal
} // namespace mfem

#endif // MFEM_BILININTEG_ELASTICITY_KERNELS_HPP
