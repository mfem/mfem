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
 *        Finite element basis - ElementDofOrdering::NATIVE
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
#include "../bilininteg.hpp"
#include "../coefficient.hpp"
#include "../qfunction.hpp"

namespace mfem
{

namespace internal
{

/// @brief Elasticity kernel for AddMultPA.
///
/// Performs y += Ax. Implemented for byNODES ordering only, and does not use
/// tensor basis, so it should work for any H1 element.
///
/// @param[in] dim 2 or 3
/// @param[in] nDofs Number of scalar dofs per element.
/// @param[in] fespace Vector-valued finite element space.
/// @param[in] lambda Quadrature function for first Lame param.
/// @param[in] mu Quadrature function for second Lame param.
/// @param[in] geom Geometric factors corresponding to fespace.
/// @param[in] maps DofToQuad maps for one element (assume elements all same).
/// @param[in] x Input vector. nDofs x dim x numEls.
/// @param Q Scratch Q-Vector. nQuad x dim x dim x numEls.
/// @param[in,out] y Ax gets added to this. nDofs x dim x numEls.
void ElasticityAddMultPA(const int dim, const int nDofs,
                         const FiniteElementSpace &fespace, const CoefficientVector &lambda,
                         const CoefficientVector &mu, const GeometricFactors &geom,
                         const DofToQuad &maps, const Vector &x, QuadratureFunction &QVec, Vector &y);

/// @brief Elasticity component kernel for AddMultPA.
///
/// Performs y += Ax. Implemented for byNODES ordering only, and does not use
/// tensor basis, so it should work for any H1 element. i_block and j_block are
/// the dimensional component that is integrated. They must both be
/// non-negative.
///
/// Example: In 2D, A = [A_00  A_01],  x = [x_0],  y = [y_0]
///                     [A_10  A_11]       [x_1]       [y_1].
/// So i_block = 0, j_block = 1 implies only y_0 += A_01*x_1 is evaluated.
///
/// @param[in] dim 2 or 3
/// @param[in] nDofs Number of scalar dofs per element.
/// @param[in] fespace Scalar-valued finite element space.
/// @param[in] lambda Quadrature function for first Lame param.
/// @param[in] mu Quadrature function for second Lame param.
/// @param[in] geom Geometric factors corresponding to fespace.
/// @param[in] maps DofToQuad maps for one element (assume elements all same).
/// @param[in] x Input vector. nDofs x numEls.
/// @param Q Scratch Q-Vector. nQuad x dim x numEls.
/// @param[in,out] y Ax gets added to this. nDofs x numEls.
/// @param[in] i_block The row dimensional component. <= dim - 1
/// @param[in] j_block The column dimensional component. <= dim -1
void ElasticityComponentAddMultPA(
   const int dim, const int nDofs, const FiniteElementSpace &fespace,
   const CoefficientVector &lambda, const CoefficientVector &mu,
   const GeometricFactors &geom, const DofToQuad &maps, const Vector &x,
   QuadratureFunction &QVec, Vector &y, const int i_block, const int j_block);

/// @brief Elasticity kernel for AssembleEA.
///
/// Assembles the E-Matrix for a single dimensional component. Does not require
/// tensor product elements.
///
/// Example: In 2D, A = [A_00  A_01]
///                     [A_10  A_11].
/// So i_block = 0, j_block = 1 implies only A_01 is assembled.
///
/// Mainly intended to be used for order 1 elements on gpus to enable
/// preconditioning with a LOR-AMG operator. It's expected behavior that higher
/// orders may request too many resources.
///
/// @param[in] dim 2 or 3
/// @param[in] i_block The row dimensional component. 0 <= i_block <= dim - 1
/// @param[in] j_block The column dimensional component. 0 <= j_block<= dim -1
/// @param[in] nDofs Number of scalar dofs per element.
/// @param[in] lambda Quadrature function for first Lame param.
/// @param[in] mu Quadrature function for second Lame param.
/// @param[in] geom Geometric factors corresponding to fespace.
/// @param[in] maps DofToQuad maps for one element (assume elements all same).
/// @param[out] emat Resulting E-Matrix Vector. nDofs x nDofs x numEls.
void ElasticityAssembleEA(const int dim, const int i_block, const int j_block,
                          const int nDofs, const IntegrationRule &ir,
                          const CoefficientVector &lambda,
                          const CoefficientVector &mu, const GeometricFactors &geom,
                          const DofToQuad &maps, Vector &emat);

/// @brief Elasticity kernel for AssembleDiagonalPA.
///
/// @param[in] dim 2 or 3
/// @param[in] nDofs Number of scalar dofs per element.
/// @param[in] lambda Quadrature function for first Lame param.
/// @param[in] mu Quadrature function for second Lame param.
/// @param[in] geom Geometric factors corresponding to fespace.
/// @param[in] maps DofToQuad maps for one element (assume elements all same).
/// @param QVec Scratch Q-Vector. nQuad x dim x dim x dim x dim x numEls.
/// @param[out] diag diagonal of A. nDofs x dim x numEls.
void ElasticityAssembleDiagonalPA(const int dim, const int nDofs,
                                  const CoefficientVector &lambda,
                                  const CoefficientVector &mu, const GeometricFactors &geom,
                                  const DofToQuad &maps, QuadratureFunction &QVec, Vector &diag);

/// Templated implementation of ElasticityAddMultPA.
template<int dim, int i_block = -1, int j_block = -1>
void ElasticityAddMultPA_(const int nDofs, const FiniteElementSpace &fespace,
                          const CoefficientVector &lambda, const CoefficientVector &mu,
                          const GeometricFactors &geom, const DofToQuad &maps, const Vector &x,
                          QuadratureFunction &QVec, Vector &y)
{
   using future::tensor;
   using future::make_tensor;
   using future::det;
   using future::inv;

   static_assert((i_block < 0) == (j_block < 0),
                 "i_block and j_block must both be non-negative or strictly negative.");
   static constexpr int d = dim;
   static constexpr int qLower = (i_block < 0) ? 0 : i_block;
   static constexpr int qUpper = (i_block < 0) ? d : i_block+1;
   static constexpr int qSize = qUpper-qLower;
   static constexpr int aLower = (j_block < 0) ? 0 : j_block;
   static constexpr int aUpper = (j_block < 0) ? d : j_block+1;
   static constexpr int aSize = aUpper-aLower;
   static constexpr bool isComponent = (i_block >= 0);

   // Assuming all elements are the same
   const auto &ir = QVec.GetIntRule(0);
   const QuadratureInterpolator *E_To_Q_Map = fespace.GetQuadratureInterpolator(
                                                 ir);
   E_To_Q_Map->SetOutputLayout(QVectorLayout::byNODES);
   // interpolate physical derivatives to quadrature points.
   E_To_Q_Map->PhysDerivatives(x, QVec);

   const int numPoints = ir.GetNPoints();
   const int numEls = fespace.GetNE();
   const auto lamDev = Reshape(lambda.Read(), numPoints, numEls);
   const auto muDev = Reshape(mu.Read(), numPoints, numEls);
   const auto J = Reshape(geom.J.Read(), numPoints, d, d, numEls);
   auto Q = Reshape(QVec.ReadWrite(), numPoints, d, qSize, numEls);
   const real_t *ipWeights = ir.GetWeights().Read();
   mfem::forall_2D(numEls, numPoints, 1, [=] MFEM_HOST_DEVICE (int e)
   {
      // for(int p = 0; p < numPoints, )
      MFEM_FOREACH_THREAD(p, x,numPoints)
      {
         auto invJ = inv(make_tensor<d, d>(
         [&](int i, int j) { return J(p, i, j, e); }));
         tensor<real_t, aSize, d> gradx;
         // load grad(x) into gradx
         if (isComponent)
         {
            for (int i = 0; i < d; i++)
            {
               gradx(0,i) = Q(p, i, 0, e);
            }
         }
         else
         {
            for (int j = 0; j < d; j++)
            {
               for (int i = 0; i < d; i++)
               {
                  gradx(i,j) = Q(p, i, j, e);
               }
            }
         }
         // compute divergence
         real_t div = 0.;
         for (int i = aLower; i < aUpper; i++)
         {
            // take size of gradx into account
            const int iIndex = isComponent ? 0 : i;
            div += gradx(iIndex,i);
         }
         const real_t w = ipWeights[p]/det(invJ);
         for (int m = 0; m < d; m++)
         {
            for (int q = qLower; q < qUpper; q++)
            {
               // compute contraction of 4*sym(grad(u))sym(grad(v)) term.
               // this contraction could be made slightly cheaper using Voigt
               // notation, but repeated entries are summed for simplicity.
               real_t contraction = 0.;
               // not sure how to combine cases
               if (isComponent)
               {
                  for (int a = 0; a < d; a++)
                  {
                     contraction += 2*((a == q)*invJ(m,j_block)
                                       + (j_block==q)*invJ(m,a))*(gradx(0, a));
                  }
               }
               else
               {
                  for (int a = 0; a < d; a++)
                  {
                     for (int b = 0; b < d; b++)
                     {
                        contraction += ((a == q)*invJ(m,b) + (b == q)*invJ(m,a))
                                       *(gradx(a,b) + gradx(b, a));
                     }
                  }
               }
               // lambda*div(u)*div(v) + 2*mu*sym(grad(u))*sym(grad(v))
               // contraction = 4*sym(grad(u))sym(grad(v))
               const int qIndex = isComponent ? 0 : q;
               Q(p,m,qIndex,e) = w*(lamDev(p, e)*invJ(m,q)*div
                                    + 0.5*muDev(p, e)*contraction);
            }
         }
      }
   });

   // Reduce quadrature function to an E-Vector
   const auto QRead = Reshape(QVec.Read(), numPoints, d, qSize, numEls);
   const auto G = Reshape(maps.G.Read(), numPoints, d, nDofs);
   auto yDev = Reshape(y.ReadWrite(), nDofs, qSize, numEls);
   mfem::forall_2D(numEls, qSize, nDofs, [=] MFEM_HOST_DEVICE (int e)
   {
      MFEM_FOREACH_THREAD(i, y, nDofs)
      {
         MFEM_FOREACH_THREAD(q, x, qSize)
         {
            const int qIndex = isComponent ? 0 : q;
            real_t sum = 0.;
            for (int m = 0; m < d; m++ )
            {
               for (int p = 0; p < numPoints; p++ )
               {
                  sum += QRead(p,m,qIndex,e)*G(p,m,i);
               }
            }
            yDev(i, qIndex, e) += sum;
         }
      }
   });
}

/// Templated implementation of ElasticityAssembleDiagonalPA.
template<int dim>
void ElasticityAssembleDiagonalPA_(const int nDofs,
                                   const CoefficientVector &lambda,
                                   const CoefficientVector &mu, const GeometricFactors &geom,
                                   const DofToQuad &maps, QuadratureFunction &QVec, Vector &diag)
{
   using future::tensor;
   using future::make_tensor;
   using future::det;
   using future::inv;

   // Assuming all elements are the same
   const auto &ir = QVec.GetIntRule(0);
   static constexpr int d = dim;
   const int numPoints = ir.GetNPoints();
   const int numEls = lambda.Size()/numPoints;
   const auto lamDev = Reshape(lambda.Read(), numPoints, numEls);
   const auto muDev = Reshape(mu.Read(), numPoints, numEls);
   const auto J = Reshape(geom.J.Read(), numPoints, d, d, numEls);
   auto Q = Reshape(QVec.ReadWrite(), numPoints, d,d, d, numEls);
   const real_t *ipWeights = ir.GetWeights().Read();
   mfem::forall_2D(numEls, numPoints,1, [=] MFEM_HOST_DEVICE (int e)
   {
      MFEM_FOREACH_THREAD(p, x,numPoints)
      {
         auto invJ = inv(make_tensor<d, d>(
         [&](int i, int j) { return J(p, i, j, e); }));
         const real_t w = ipWeights[p] /det(invJ);
         for (int n = 0; n < d; n++)
         {
            for (int m = 0; m < d; m++)
            {
               for (int q = 0; q < d; q++)
               {
                  // compute contraction of 4*sym(grad(u))sym(grad(v)) term.
                  // this contraction could be made slightly cheaper using Voigt
                  // notation, but repeated entries are summed for simplicity.
                  real_t contraction = 0.;
                  for (int a = 0; a < d; a++)
                  {
                     for (int b = 0; b < d; b++)
                     {
                        contraction += ((a == q)*invJ(m,b) + (b==q)*invJ(m,a))*((a == q)
                                                                                *invJ(n, b) + (b==q)*invJ(n,a));
                     }
                  }
                  // lambda*div(u)*div(v) + 2*mu*sym(grad(u))*sym(grad(v))
                  // contraction = 4*sym(grad(u))sym(grad(v))
                  Q(p,m,n,q,e) = w*(lamDev(p, e)*invJ(m,q)*invJ(n,q)
                                    + 0.5*muDev(p, e)*contraction);
               }
            }
         }
      }
   });

   // Reduce quadrature function to an E-Vector
   const auto QRead = Reshape(QVec.Read(), numPoints, d, d, d, numEls);
   auto diagDev = Reshape(diag.Write(), nDofs, d, numEls);
   const auto G = Reshape(maps.G.Read(), numPoints, d, nDofs);
   mfem::forall_2D(numEls, d, nDofs, [=] MFEM_HOST_DEVICE (int e)
   {
      MFEM_FOREACH_THREAD(i, y, nDofs)
      {
         MFEM_FOREACH_THREAD(q, x, d)
         {
            real_t sum = 0.;
            for (int n = 0; n < d; n++)
            {
               for (int m = 0; m < d; m++)
               {
                  for (int p = 0; p < numPoints; p++ )
                  {
                     sum += QRead(p,m,n,q,e)*G(p,m,i)*G(p,n,i);
                  }
               }
            }
            diagDev(i, q, e) = sum;
         }
      }
   });
}

// Templated implementation of ElasticityAssembleEA.
template<int dim>
void ElasticityAssembleEA_(const int i_block,
                           const int j_block,
                           const int nDofs,
                           const IntegrationRule &ir,
                           const CoefficientVector &lambda,
                           const CoefficientVector &mu,
                           const GeometricFactors &geom,
                           const DofToQuad &maps,
                           Vector &emat)
{
   using future::tensor;
   using future::make_tensor;
   using future::det;
   using future::inv;

   // Assuming all elements are the same
   static constexpr int d = dim;
   const int numPoints = ir.GetNPoints();
   const int numEls = lambda.Size()/numPoints;
   const auto lamDev = Reshape(lambda.Read(), numPoints, numEls);
   const auto muDev = Reshape(mu.Read(), numPoints, numEls);
   const auto J = Reshape(geom.J.Read(), numPoints, d, d, numEls);
   const auto G = Reshape(maps.G.Read(), numPoints, d, nDofs);
   auto ematDev = Reshape(emat.Write(), nDofs, nDofs, numEls);
   const real_t *ipWeights = ir.GetWeights().Read();
   mfem::forall_2D(numEls, nDofs, nDofs, [=] MFEM_HOST_DEVICE (int e)
   {
      MFEM_FOREACH_THREAD(JDof, y, nDofs)
      {
         MFEM_FOREACH_THREAD(IDof, x, nDofs)
         {
            real_t sum = 0;
            for (int p = 0 ; p < numPoints; p++)
            {
               auto invJ = inv(make_tensor<d, d>(
               [&](int i, int j) { return J(p, i, j, e); }));
               const real_t w = ipWeights[p] /det(invJ);
               for (int n = 0; n < d; n++)
               {
                  for (int m = 0; m < d; m++)
                  {
                     // compute contraction of 4*sym(grad(u))sym(grad(v)) term.
                     real_t contraction = 0.;
                     for (int a = 0; a < d; a++)
                     {
                        for (int b = 0; b < d; b++)
                        {
                           contraction += ((a == i_block)*invJ(m,b) + (b==i_block)*invJ(m,
                                                                                        a))*((a == j_block)*invJ(n,
                                                                                              b) + (b==j_block)*invJ(n,a));
                        }
                     }
                     // lambda*div(u)*div(v) + 2*mu*sym(grad(u))*sym(grad(v))
                     // contraction = 4*sym(grad(u))sym(grad(v))
                     sum += w*(lamDev(p, e)*invJ(m,i_block)*invJ(n,j_block)
                               + 0.5*muDev(p, e)*contraction)*G(p,m,IDof)*G(p,n,JDof);
                  }
               }
            }
            ematDev(IDof, JDof, e) = sum;
         }
      }
   });
}

} // namespace internal

} // namespace mfem

#endif
