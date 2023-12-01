// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
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

namespace mfem
{

namespace internal
{
/// @brief Elasticity kernel for AddMultPA.
///
/// Performs y += Ax. Implemented for byNODES ordering only, and does not
/// use tensor basis, so it should work for any H1 element. IBlock and JBlock
/// are the dimensional component that is integrated. They must both be
/// either non-negative or both be negative. Negative values imply that the
/// whole dimensional system is evaluated. Otherwise, only one block of the
/// system is evaluated.
///
/// Example: In 2D, A = [A_00  A_01],  x = [x_0],  y = [y_0]
///                     [A_10  A_11]       [x_1]       [y_1].
/// So IBlock = 0, JBlock = 1 implies only y_0 += A_01*x_1 is evaluated.
///
/// The sizes of x, y, and Q depend on whether or not a single component is
/// evaluated. Also, fespace is either a vector or scalar space depending on if
/// a single component is used.
/// @param[in] dim 2 or 3
/// @param[in] nDofs Number of scalar dofs per element.
/// @param[in] fespace Vector (IBlock, JBlock<0) or scalar FE space.
/// @param[in] lambda Quadrature function for first Lame param.
/// @param[in] mu Quadrature function for first Lame param.
/// @param[in] geom Geometric factors corresponding to fespace.
/// @param[in] maps DofToQuad maps for one element (assume elements all same).
/// @param[in] x Input vector. nDofs x dim x numEls or nDofs x numEls.
/// @param Q Scratch Q-Vector. nQuad x dim x dim x numEls or nQuad x dim x numEls.
/// @param[in,out] y Ax gets added to this. nDofs x dim x numEls or nDofs x numEls.
/// @param[in] IBlock The row dimensional component. <= dim - 1
/// @param[in] JBlock The column dimensional component. <= dim -1
void ElasticityAddMultPA(const int dim, const int nDofs,
                         const FiniteElementSpace &fespace, const QuadratureFunction &lambda,
                         const QuadratureFunction &mu, const GeometricFactors &geom,
                         const DofToQuad &maps, const Vector &x, QuadratureFunction &QVec, Vector &y,
                         const int IBlock = -1, const int JBlock = -1);

/// @brief Elasticity kernel for AssembleEA.
///
/// Assembles the E-Matrix for a single dimensional component. Does not require
/// tensor product elements.
///
/// Example: In 2D, A = [A_00  A_01]
///                     [A_10  A_11].
/// So IBlock = 0, JBlock = 1 implies only A_01 is assembled.
///
/// Mainly intended to be used for order 1 elements on gpus to enable
/// preconditioning with a LOR-AMG operator. It's expected behavior that higher
/// orders may request too many resources and crash.
/// @param[in] dim 2 or 3
/// @param[in] IBlock The row dimensional component. 0 <= IBlock <= dim - 1
/// @param[in] JBlock The column dimensional component. 0 <= JBlock<= dim -1
/// @param[in] nDofs Number of scalar dofs per element.
/// @param[in] fespace Scalar FE space.
/// @param[in] lambda Quadrature function for first Lame param.
/// @param[in] mu Quadrature function for first Lame param.
/// @param[in] geom Geometric factors corresponding to fespace.
/// @param[in] maps DofToQuad maps for one element (assume elements all same).
/// @param[out] emat Resulting E-Matrix Vector. nDofs x nDofs x numEls.
void ElasticityAssembleEA(const int dim, const int IBlock, const int JBlock,
                          const int nDofs,
                          const FiniteElementSpace &fespace, const QuadratureFunction &lambda,
                          const QuadratureFunction &mu, const GeometricFactors &geom,
                          const DofToQuad &maps, Vector &emat);

/// @brief Elasticity kernel for AssembleDiagonalPA. Whole system only.
///
/// @param[in] dim 2 or 3
/// @param[in] nDofs Number of scalar dofs per element.
/// @param[in] fespace Vector (IBlock, JBlock<0) or scalar FE space.
/// @param[in] lambda Quadrature function for first Lame param.
/// @param[in] mu Quadrature function for first Lame param.
/// @param[in] geom Geometric factors corresponding to fespace.
/// @param[in] maps DofToQuad maps for one element (assume elements all same).
/// @param QVec Scratch Q-Vector. nQuad x dim x dim x dim x dim x numEls.
/// @param[out] diag diagonal of A. nDofs x dim x numEls.
void ElasticityAssembleDiagonalPA(const int dim, const int nDofs,
                                  const FiniteElementSpace &fespace, const QuadratureFunction &lambda,
                                  const QuadratureFunction &mu, const GeometricFactors &geom,
                                  const DofToQuad &maps, QuadratureFunction &QVec, Vector &diag);

/// Templated implementation of ElasticityAddMultPA.
template<int dim, int IBlock = -1, int JBlock = -1>
void ElasticityAddMultPA(const int nDofs, const FiniteElementSpace &fespace,
                         const QuadratureFunction &lambda, const QuadratureFunction &mu,
                         const GeometricFactors &geom, const DofToQuad &maps, const Vector &x,
                         QuadratureFunction &QVec, Vector &y)
{
   static_assert((IBlock < 0) == (JBlock < 0),
                 "IBlock and JBlock must both be non-negative or strictly negative.");
   static constexpr int d = dim;
   static constexpr int qLower = (IBlock < 0) ? 0 : IBlock;
   static constexpr int qUpper = (IBlock < 0) ? d : IBlock+1;
   static constexpr int qSize = qUpper-qLower;
   static constexpr int aLower = (JBlock < 0) ? 0 : JBlock;
   static constexpr int aUpper = (JBlock < 0) ? d : JBlock+1;
   static constexpr int aSize = aUpper-aLower;
   static constexpr bool isComponent = (IBlock >= 0);

   //Assuming all elements are the same
   const auto &ir = lambda.GetIntRule(0);
   const QuadratureInterpolator *E_To_Q_Map = fespace.GetQuadratureInterpolator(
                                                 ir);
   E_To_Q_Map->SetOutputLayout(QVectorLayout::byNODES);
   //interpolate physical derivatives to quadrature points.
   Vector junk;
   E_To_Q_Map->Mult(x,QuadratureInterpolator::PHYSICAL_DERIVATIVES, junk,
                    QVec, junk);

   int numPoints = ir.GetNPoints();
   int numEls = lambda.Size()/numPoints;
   const auto lamDev = Reshape(lambda.Read(), numPoints, numEls);
   const auto muDev = Reshape(mu.Read(), numPoints, numEls);
   const auto J = Reshape(geom.J.Read(), numPoints, d, d, numEls);
   auto Q = Reshape(QVec.ReadWrite(), numPoints, d, qSize, numEls);
   const double *ipWeights = ir.GetWeights().Read();
   mfem::forall_2D(numEls, numPoints, 1, [=] MFEM_HOST_DEVICE (int e)
   {
      // for(int p = 0; p < numPoints, )
      MFEM_FOREACH_THREAD(p, x,numPoints)
      {
         auto invJ = inv(make_tensor<d, d>(
         [&](int i, int j) { return J(p, i, j, e); }));
         tensor<double, aSize, d> gradx;
         //load grad(x) into gradx
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
         //compute divergence
         double div = 0.;
         for (int i = aLower; i < aUpper; i++)
         {
            //take size of gradx into account
            const int iIndex = isComponent ? 0 : i;
            div += gradx(iIndex,i);
         }
         const double w = ipWeights[p] /det(invJ);
         for (int m = 0; m < d; m++)
         {
            for (int q = qLower; q < qUpper; q++)
            {
               //compute contraction of 4*sym(grad(u))sym(grad(v)) term.
               //this contraction could be made slightly cheaper using Voigt
               //notation, but repeated entries are summed for simplicity.
               double contraction = 0.;
               //not sure how to combine cases
               if (isComponent)
               {
                  for (int a = 0; a < d; a++)
                  {
                     contraction += 2*((a == q)*invJ(m,JBlock) + (JBlock==q)*invJ(m,a))*(gradx(0,a));
                  }
               }
               else
               {
                  for (int a = 0; a < d; a++)
                  {
                     for (int b = 0; b < d; b++)
                     {
                        contraction += ((a == q)*invJ(m,b) + (b==q)*invJ(m,a))
                                       *(gradx(a,b) + gradx(b, a));
                     }
                  }
               }
               // lambda*div(u)*div(v) + 2*mu*sym(grad(u))*sym(grad(v))
               // contraction = 4*sym(grad(u))sym(grad(v))
               const int qIndex = isComponent ? 0 : q;
               Q(p,m,qIndex,e) = w*(lamDev(p, e)*invJ(m,q)*div + 0.5*muDev(p, e)*contraction);
            }
         }
      }
   });

   //Reduce quadrature function to an E-Vector
   const auto QRead = Reshape(QVec.Read(), numPoints, d, qSize, numEls);
   const auto G = Reshape(maps.G.Read(), numPoints, d, numEls);
   auto yDev = Reshape(y.ReadWrite(), nDofs, qSize, numEls);
   mfem::forall_2D(numEls, qSize, nDofs, [=] MFEM_HOST_DEVICE (int e)
   {
      MFEM_FOREACH_THREAD(i, y, nDofs)
      {
         MFEM_FOREACH_THREAD(q, x, qSize)
         {
            const int qIndex = isComponent ? 0 : q;
            double sum = 0.;
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
void ElasticityAssembleDiagonalPA(const int nDofs,
                                  const FiniteElementSpace &fespace, const QuadratureFunction &lambda,
                                  const QuadratureFunction &mu, const GeometricFactors &geom,
                                  const DofToQuad &maps, QuadratureFunction &QVec, Vector &diag)
{
   //Assuming all elements are the same
   const auto &ir = lambda.GetIntRule(0);
   static constexpr int d = dim;
   int numPoints = ir.GetNPoints();
   int numEls = lambda.Size()/numPoints;
   const auto lamDev = Reshape(lambda.Read(), numPoints, numEls);
   const auto muDev = Reshape(mu.Read(), numPoints, numEls);
   const auto J = Reshape(geom.J.Read(), numPoints, d, d, numEls);
   auto Q = Reshape(QVec.ReadWrite(), numPoints, d,d, d, numEls);
   const double *ipWeights = ir.GetWeights().Read();
   mfem::forall_2D(numEls, numPoints,1, [=] MFEM_HOST_DEVICE (int e)
   {
      MFEM_FOREACH_THREAD(p, x,numPoints)
      {
         auto invJ = inv(make_tensor<d, d>(
         [&](int i, int j) { return J(p, i, j, e); }));
         const double w = ipWeights[p] /det(invJ);
         for (int n = 0; n < d; n++)
         {
            for (int m = 0; m < d; m++)
            {
               for (int q = 0; q < d; q++)
               {
                  //compute contraction of 4*sym(grad(u))sym(grad(v)) term.
                  //this contraction could be made slightly cheaper using Voigt
                  //notation, but repeated entries are summed for simplicity.
                  double contraction = 0.;
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

   //Reduce quadrature function to an E-Vector
   const auto QRead = Reshape(QVec.Read(), numPoints, d, d, d, numEls);
   auto diagDev = Reshape(diag.Write(), nDofs, d, numEls);
   const auto G = Reshape(maps.G.Read(), numPoints, d, numEls);
   mfem::forall_2D(numEls, d, nDofs, [=] MFEM_HOST_DEVICE (int e)
   {
      MFEM_FOREACH_THREAD(i, y, nDofs)
      {
         MFEM_FOREACH_THREAD(q, x, d)
         {
            double sum = 0.;
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

//Templated implementation of ElasticityAssembleEA.
template<int dim>
void ElasticityAssembleEA(const int IBlock, const int JBlock, const int nDofs,
                          const FiniteElementSpace &fespace, const QuadratureFunction &lambda,
                          const QuadratureFunction &mu, const GeometricFactors &geom,
                          const DofToQuad &maps, Vector &emat)
{
   //Assuming all elements are the same
   const auto &ir = lambda.GetIntRule(0);
   static constexpr int d = dim;
   int numPoints = ir.GetNPoints();
   int numEls = lambda.Size()/numPoints;
   const auto lamDev = Reshape(lambda.Read(), numPoints, numEls);
   const auto muDev = Reshape(mu.Read(), numPoints, numEls);
   const auto J = Reshape(geom.J.Read(), numPoints, d, d, numEls);
   const auto G = Reshape(maps.G.Read(), numPoints, d, nDofs);
   auto ematDev = Reshape(emat.Write(), nDofs, nDofs, numEls);
   const double *ipWeights = ir.GetWeights().Read();
   mfem::forall_2D(numEls, nDofs, nDofs, [=] MFEM_HOST_DEVICE (int e)
   {
      MFEM_FOREACH_THREAD(JDof, y, nDofs)
      {
         MFEM_FOREACH_THREAD(IDof, x, nDofs)
         {
            double sum = 0;
            for (int p = 0 ; p < numPoints; p++)
            {
               auto invJ = inv(make_tensor<d, d>(
               [&](int i, int j) { return J(p, i, j, e); }));
               const double w = ipWeights[p] /det(invJ);
               for (int n = 0; n < d; n++)
               {
                  for (int m = 0; m < d; m++)
                  {
                     //compute contraction of 4*sym(grad(u))sym(grad(v)) term.
                     double contraction = 0.;
                     for (int a = 0; a < d; a++)
                     {
                        for (int b = 0; b < d; b++)
                        {
                           contraction += ((a == IBlock)*invJ(m,b) + (b==IBlock)*invJ(m,
                                                                                      a))*((a == JBlock)*invJ(n,
                                                                                            b) + (b==JBlock)*invJ(n,a));
                        }
                     }
                     // lambda*div(u)*div(v) + 2*mu*sym(grad(u))*sym(grad(v))
                     // contraction = 4*sym(grad(u))sym(grad(v))
                     sum += w*(lamDev(p, e)*invJ(m,IBlock)*invJ(n,JBlock)
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
