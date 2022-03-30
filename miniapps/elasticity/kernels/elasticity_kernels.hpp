// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_ELASTICITY_KERNELS_HPP
#define MFEM_ELASTICITY_KERNELS_HPP

#include "kernel_helpers.hpp"
#include "linalg/vector.hpp"

using mfem::internal::tensor;
using mfem::internal::make_tensor;

namespace mfem
{

namespace ElasticityKernels
{

template <int d1d, int q1d, typename material_type> static inline
void Apply3D(const int ne,
             const Array<double> &B_,
             const Array<double> &G_,
             const Array<double> &W_,
             const Vector &Jacobian_,
             const Vector &detJ_,
             const Vector &X_, Vector &Y_,
             const material_type &material)
{
   constexpr int dim = 3;
   KernelHelpers::CheckMemoryRestriction(d1d, q1d);

   // Basis functions evaluated at quadrature points.
   const tensor<double, q1d, d1d> &B =
   make_tensor<q1d, d1d>([&](int i, int j) { return B_[i + q1d*j]; });

   // Gradients of basis functions evaluated at quadrature points.
   const tensor<double, q1d, d1d> &G =
   make_tensor<q1d, d1d>([&](int i, int j) { return G_[i + q1d*j]; });

   const auto qweights = Reshape(W_.Read(), q1d, q1d, q1d);
   // Jacobians of the element transformations at all quadrature points in
   // column-major layout q1d x q1d x q1d x sdim x dim x ne
   const auto J = Reshape(Jacobian_.Read(), q1d, q1d, q1d, dim, dim, ne);
   const auto detJ = Reshape(detJ_.Read(), q1d, q1d, q1d, ne);
   // Input vector
   // d1d x d1d x d1d x vdim x ne
   const auto U = Reshape(X_.Read(), d1d, d1d, d1d, dim, ne);
   // Output vector
   // d1d x d1d x d1d x vdim x ne
   auto force = Reshape(Y_.ReadWrite(), d1d, d1d, d1d, dim, ne);

   MFEM_FORALL_3D(e, ne, q1d, q1d, q1d,
   {
      // shared memory placeholders for temporary contraction results
      MFEM_SHARED tensor<double, 2, 3, q1d, q1d, q1d> smem;
      // cauchy stress
      MFEM_SHARED_3D_BLOCK_TENSOR(invJ_sigma_detJw, double, q1d, q1d, q1d, dim, dim);
      // du/dxi
      MFEM_SHARED_3D_BLOCK_TENSOR(dudxi, double, q1d, q1d, q1d, dim, dim);

      const auto U_el = Reshape(&U(0, 0, 0, 0, e), d1d, d1d, d1d, dim);
      KernelHelpers::CalcGrad(B, G, smem, U_el, dudxi);

      MFEM_FOREACH_THREAD(qx, x, q1d)
      {
         MFEM_FOREACH_THREAD(qy, y, q1d)
         {
            MFEM_FOREACH_THREAD(qz, z, q1d)
            {
               auto invJqp = inv(make_tensor<dim, dim>(
               [&](int i, int j) { return J(qx, qy, qz, i, j, e); }));

               auto dudx = dudxi(qz, qy, qx) * invJqp;

               auto sigma = material.stress(dudx);

               invJ_sigma_detJw(qx, qy, qz) =
                  invJqp * sigma * detJ(qx, qy, qz, e) * qweights(qx, qy, qz);
            }
         }
      }
      MFEM_SYNC_THREAD;

      auto F = Reshape(&force(0, 0, 0, 0, e), d1d, d1d, d1d, dim);
      KernelHelpers::CalcGradTSum(B, G, smem, invJ_sigma_detJw, F);
   }); // for each element
}

template <int d1d, int q1d, typename material_type> static inline
void ApplyGradient3D(const int ne,
                     const Array<double> &B_, const Array<double> &G_,
                     const Array<double> &W_, const Vector &Jacobian_,
                     const Vector &detJ_, const Vector &dU_, Vector &dF_,
                     const Vector &U_, const material_type &material,
                     const bool use_cache_, const bool recompute_cache_,
                     Vector &dsigma_cache_)
{
   constexpr int dim = 3;
   KernelHelpers::CheckMemoryRestriction(d1d, q1d);

   // Basis functions evaluated at quadrature points.
   const tensor<double, q1d, d1d> &B =
   make_tensor<q1d, d1d>([&](int i, int j) { return B_[i + q1d*j]; });

   // Gradients of basis functions evaluated at quadrature points.
   const tensor<double, q1d, d1d> &G =
   make_tensor<q1d, d1d>([&](int i, int j) { return G_[i + q1d*j]; });

   const auto qweights = Reshape(W_.Read(), q1d, q1d, q1d);
   // Jacobians of the element transformations at all quadrature points in
   // column-major layout q1d x q1d x q1d x sdim x dim x ne
   const auto J = Reshape(Jacobian_.Read(), q1d, q1d, q1d, dim, dim, ne);
   const auto detJ = Reshape(detJ_.Read(), q1d, q1d, q1d, ne);
   // Input vector
   // d1d x d1d x d1d x vdim x ne
   const auto dU = Reshape(dU_.Read(), d1d, d1d, d1d, dim, ne);
   // Output vector
   // d1d x d1d x d1d x vdim x ne
   auto force = Reshape(dF_.ReadWrite(), d1d, d1d, d1d, dim, ne);
   // Input vector
   // d1d x d1d x d1d x vdim x ne
   const auto U = Reshape(U_.Read(), d1d, d1d, d1d, dim, ne);

   auto dsigma_cache = Reshape(dsigma_cache_.ReadWrite(), ne, q1d, q1d, q1d,
                               dim, dim, dim, dim);

   MFEM_FORALL_3D(e, ne, q1d, q1d, q1d,
   {
      // shared memory placeholders for temporary contraction results
      MFEM_SHARED tensor<double, 2, 3, q1d, q1d, q1d> smem;
      // cauchy stress
      MFEM_SHARED tensor<double, q1d, q1d, q1d, dim, dim> invJ_dsigma_detJw;
      // du/dxi, ddu/dxi
      MFEM_SHARED_3D_BLOCK_TENSOR( dudxi, double, q1d, q1d, q1d, dim, dim);
      MFEM_SHARED_3D_BLOCK_TENSOR(ddudxi, double, q1d, q1d, q1d, dim, dim);

      const auto U_el = Reshape(&U(0, 0, 0, 0, e), d1d, d1d, d1d, dim);
      KernelHelpers::CalcGrad(B, G, smem, U_el, dudxi);

      const auto dU_el = Reshape(&dU(0, 0, 0, 0, e), d1d, d1d, d1d, dim);
      KernelHelpers::CalcGrad(B, G, smem, dU_el, ddudxi);

      MFEM_FOREACH_THREAD(qx, x, q1d)
      {
         MFEM_FOREACH_THREAD(qy, y, q1d)
         {
            MFEM_FOREACH_THREAD(qz, z, q1d)
            {
               auto invJqp = inv(make_tensor<dim, dim>(
               [&](int i, int j) { return J(qx, qy, qz, i, j, e); }));

               auto dudx = dudxi(qz, qy, qx) * invJqp;
               auto ddudx = ddudxi(qz, qy, qx) * invJqp;

               if (use_cache_)
               {
                  // C = dsigma/dudx
                  tensor<double, dim, dim, dim, dim> C;

                  auto C_cache = make_tensor<dim, dim, dim, dim>(
                  [&](int i, int j, int k, int l) { return dsigma_cache(e, qx, qy, qz, i, j, k, l); });

                  if (recompute_cache_)
                  {
                     C = material.gradient(dudx);
                     for (int i = 0; i < dim; i++)
                     {
                        for (int j = 0; j < dim; j++)
                        {
                           for (int k = 0; k < dim; k++)
                           {
                              for (int l = 0; l < dim; l++)
                              {
                                 dsigma_cache(e, qx, qy, qz, i, j, k, l) = C(i, j, k, l);
                              }
                           }
                        }
                     }
                     C_cache = C;
                  }
                  else
                  {
                     C = C_cache;
                  }
                  invJ_dsigma_detJw(qx, qy, qz) =
                     invJqp * ddot(C, ddudx) * detJ(qx, qy, qz, e) * qweights(qx, qy, qz);
               }
               else
               {
                  auto dsigma = material.action_of_gradient(dudx, ddudx);
                  invJ_dsigma_detJw(qx, qy, qz) =
                     invJqp * dsigma * detJ(qx, qy, qz, e) * qweights(qx, qy, qz);
               }
            }
         }
      }
      MFEM_SYNC_THREAD;
      auto F = Reshape(&force(0, 0, 0, 0, e), d1d, d1d, d1d, dim);
      KernelHelpers::CalcGradTSum(B, G, smem, invJ_dsigma_detJw, F);
   }); // for each element
}

template <int d1d, int q1d, typename material_type> static inline
void AssembleGradientDiagonal3D(const int ne,
                                const Array<double> &B_,
                                const Array<double> &G_,
                                const Array<double> &W_,
                                const Vector &Jacobian_,
                                const Vector &detJ_,
                                const Vector &X_,
                                Vector &Ke_diag_memory,
                                const material_type &material)
{
   constexpr int dim = 3;
   KernelHelpers::CheckMemoryRestriction(d1d, q1d);

   // Basis functions evaluated at quadrature points.
   const tensor<double, q1d, d1d> &B =
   make_tensor<q1d, d1d>([&](int i, int j) { return B_[i + q1d*j]; });

   // Gradients of basis functions evaluated at quadrature points.
   const tensor<double, q1d, d1d> &G =
   make_tensor<q1d, d1d>([&](int i, int j) { return G_[i + q1d*j]; });

   const auto qweights = Reshape(W_.Read(), q1d, q1d, q1d);
   // Jacobians of the element transformations at all quadrature points. This
   // array uses a column-major layout
   // (q1d x q1d x q1d x sdim x dim x ne)
   const auto J = Reshape(Jacobian_.Read(), q1d, q1d, q1d, dim, dim, ne);
   const auto detJ = Reshape(detJ_.Read(), q1d, q1d, q1d, ne);
   // Input vector
   // d1d x d1d x d1d x vdim x ne
   const auto U = Reshape(X_.Read(), d1d, d1d, d1d, dim, ne);
   // Output vector
   // d1d x d1d x d1d x vdim x ne
   auto Ke_diag_m =
      Reshape(Ke_diag_memory.ReadWrite(), d1d, d1d, d1d, dim, ne, dim);

   MFEM_FORALL_3D(e, ne, q1d, q1d, q1d,
   {
      // shared memory placeholders for temporary contraction results
      MFEM_SHARED tensor<double, 2, 3, q1d, q1d, q1d> smem;

      // du/dxi
      MFEM_SHARED_3D_BLOCK_TENSOR(dudxi, double, q1d, q1d, q1d, dim, dim);
      MFEM_SHARED_3D_BLOCK_TENSOR(Ke_diag, double, d1d, d1d, d1d, dim, dim);

      const auto U_el = Reshape(&U(0, 0, 0, 0, e), d1d, d1d, d1d, dim);
      KernelHelpers::CalcGrad(B, G, smem, U_el, dudxi);

      MFEM_FOREACH_THREAD(qx, x, q1d)
      {
         MFEM_FOREACH_THREAD(qy, y, q1d)
         {
            MFEM_FOREACH_THREAD(qz, z, q1d)
            {
               const auto invJqp = inv(make_tensor<dim, dim>([&](int i, int j)
               {
                  return J(qx, qy, qz, i, j, e);
               }));

               const auto dudx = dudxi(qz, qy, qx) * invJqp;

               const auto dsigma_ddudx = material.gradient(dudx);

               const double JxW = detJ(qx, qy, qz, e) * qweights(qx, qy, qz);
               const auto dphidx = KernelHelpers::GradAllPhis(qx, qy, qz, B, G, invJqp);

               for (int dx = 0; dx < d1d; dx++)
               {
                  for (int dy = 0; dy < d1d; dy++)
                  {
                     for (int dz = 0; dz < d1d; dz++)
                     {
                        // phi_i * f(...) * phi_i
                        // dphidx_i dsigma_ddudx_ijkl dphidx_l
                        const auto phi_i = dphidx(dx, dy, dz);
                        const auto val = JxW * ( phi_i * dsigma_ddudx * phi_i);
                        for (int l = 0; l < dim; l++)
                        {
                           for (int m = 0; m < dim; m++)
                           {
                              AtomicAdd(Ke_diag(dx, dy, dz, l, m), val[l][m]);
                           } // m
                        } // l
                     } // dz
                  } // dy
               } // dx
            } // qz
         } // qy
      } // qx
      MFEM_SYNC_THREAD;

      MFEM_FOREACH_THREAD(i, x, d1d)
      {
         MFEM_FOREACH_THREAD(j, y, d1d)
         {
            MFEM_FOREACH_THREAD(k, z, d1d)
            {
               for (int l = 0; l < dim; l++)
               {
                  for (int m = 0; m < dim; m++)
                  {
                     Ke_diag_m(i, j, k, l, e, m) = Ke_diag(i, j, k, l, m);
                  }
               }
            }
         }
      }
   }); // for each element
}

} // namespace ElasticityKernels

} // namespace mfem

#endif
