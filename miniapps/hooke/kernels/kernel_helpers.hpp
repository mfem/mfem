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

#ifndef MFEM_ELASTICITY_KERNEL_HELPERS_HPP
#define MFEM_ELASTICITY_KERNEL_HELPERS_HPP

#include "mfem.hpp"
#include "general/forall.hpp"
#include "linalg/tensor.hpp"

using mfem::internal::tensor;

namespace mfem
{

namespace KernelHelpers
{

// MFEM_SHARED_3D_BLOCK_TENSOR definition
// Should be moved in backends/cuda/hip header files.
#if defined(__CUDA_ARCH__)
#define MFEM_SHARED_3D_BLOCK_TENSOR(name,T,bx,by,bz,...)\
MFEM_SHARED tensor<T,bx,by,bz,__VA_ARGS__> name;\
name(threadIdx.x, threadIdx.y, threadIdx.z) = tensor<T,__VA_ARGS__> {};
#else
#define MFEM_SHARED_3D_BLOCK_TENSOR(name,...) tensor<__VA_ARGS__> name {};
#endif

// Kernel helper functions

/**
 * @brief Runtime check for memory restrictions that are determined at compile
 * time.
 *
 * @param d1d Number of degrees of freedom in 1D.
 * @param q1d Number of quadrature points in 1D.
 */
inline void CheckMemoryRestriction(int d1d, int q1d)
{
   MFEM_VERIFY(d1d <= q1d,
               "There should be more or equal quadrature points "
               "as there are dofs");
   MFEM_VERIFY(d1d <= MAX_D1D,
               "Maximum number of degrees of freedom in 1D reached."
               "This number can be increased globally in general/forall.hpp if "
               "device memory allows.");
   MFEM_VERIFY(q1d <= MAX_Q1D, "Maximum quadrature points 1D reached."
               "This number can be increased globally in "
               "general/forall.hpp if device memory allows.");
}

/**
 * @brief Multi-component gradient evaluation from DOFs to quadrature points in
 * reference coordinates.
 *
 * The implementation exploits sum factorization.
 *
 * @note DeviceTensor<2> means RANK=2
 *
 * @tparam dim Dimension.
 * @tparam d1d Number of degrees of freedom in 1D.
 * @tparam q1d er of quadrature points in 1D.
 * @param B Basis functions evaluated at quadrature points in column-major
 * layout q1d x d1d.
 * @param G Gradients of basis functions evaluated at quadrature points in
 * column major layout q1d x d1d.
 * @param smem Block of shared memory for scratch space. Size needed is 2 x 3 x
 * q1d x q1d x q1d.
 * @param U Input vector d1d x d1d x d1d x dim.
 * @param dUdxi Gradient of the input vector wrt to reference coordinates. Size
 * needed q1d x q1d x q1d x dim x dim.
 */
template <int dim, int d1d, int q1d>
static inline MFEM_HOST_DEVICE void
CalcGrad(const tensor<double, q1d, d1d> &B,
         const tensor<double, q1d, d1d> &G,
         tensor<double,2,3,q1d,q1d,q1d> &smem,
         const DeviceTensor<4, const double> &U,
         tensor<double, q1d, q1d, q1d, dim, dim> &dUdxi)
{
   for (int c = 0; c < dim; ++c)
   {
      MFEM_FOREACH_THREAD(dz,z,d1d)
      {
         MFEM_FOREACH_THREAD(dy,y,d1d)
         {
            MFEM_FOREACH_THREAD(dx,x,d1d)
            {
               smem(0,0,dx,dy,dz) = U(dx,dy,dz,c);
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(dz,z,d1d)
      {
         MFEM_FOREACH_THREAD(dy,y,d1d)
         {
            MFEM_FOREACH_THREAD(qx,x,q1d)
            {
               double u = 0.0;
               double v = 0.0;
               for (int dx = 0; dx < d1d; ++dx)
               {
                  const double input = smem(0,0,dx,dy,dz);
                  u += input * B(qx,dx);
                  v += input * G(qx,dx);
               }
               smem(0,1,dz,dy,qx) = u;
               smem(0,2,dz,dy,qx) = v;
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(dz,z,d1d)
      {
         MFEM_FOREACH_THREAD(qy,y,q1d)
         {
            MFEM_FOREACH_THREAD(qx,x,q1d)
            {
               double u = 0.0;
               double v = 0.0;
               double w = 0.0;
               for (int dy = 0; dy < d1d; ++dy)
               {
                  u += smem(0,2,dz,dy,qx) * B(qy,dy);
                  v += smem(0,1,dz,dy,qx) * G(qy,dy);
                  w += smem(0,1,dz,dy,qx) * B(qy,dy);
               }
               smem(1,0,dz,qy,qx) = u;
               smem(1,1,dz,qy,qx) = v;
               smem(1,2,dz,qy,qx) = w;
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(qz,z,q1d)
      {
         MFEM_FOREACH_THREAD(qy,y,q1d)
         {
            MFEM_FOREACH_THREAD(qx,x,q1d)
            {
               double u = 0.0;
               double v = 0.0;
               double w = 0.0;
               for (int dz = 0; dz < d1d; ++dz)
               {
                  u += smem(1,0,dz,qy,qx) * B(qz,dz);
                  v += smem(1,1,dz,qy,qx) * B(qz,dz);
                  w += smem(1,2,dz,qy,qx) * G(qz,dz);
               }
               dUdxi(qz,qy,qx,c,0) += u;
               dUdxi(qz,qy,qx,c,1) += v;
               dUdxi(qz,qy,qx,c,2) += w;
            }
         }
      }
      MFEM_SYNC_THREAD;
   } // vdim
}

/**
 * @brief Multi-component transpose gradient evaluation from DOFs to quadrature
 * points in reference coordinates with contraction of the D vector.
 *
 * @tparam dim Dimension.
 * @tparam d1d Number of degrees of freedom in 1D.
 * @tparam q1d er of quadrature points in 1D.
 * @param B Basis functions evaluated at quadrature points in column-major
 * layout q1d x d1d.
 * @param G Gradients of basis functions evaluated at quadrature points in
 * column major layout q1d x d1d.
 * @param smem Block of shared memory for scratch space. Size needed is 2 x 3 x
 * q1d x q1d x q1d.
 * @param U Input vector q1d x q1d x q1d x dim.
 * @param F Output vector that applied the gradient evaluation from DOFs to
 * quadrature points in reference coordinates with contraction of the D operator
 * on the input vector. Size is d1d x d1d x d1d x dim.
 */
template <int dim, int d1d, int q1d>
static inline MFEM_HOST_DEVICE void
CalcGradTSum(const tensor<double, q1d, d1d> &B,
             const tensor<double, q1d, d1d> &G,
             tensor<double, 2, 3, q1d, q1d, q1d> &smem,
             const tensor<double, q1d, q1d, q1d, dim, dim> &U,
             DeviceTensor<4, double> &F)
{
   for (int c = 0; c < dim; ++c)
   {
      MFEM_FOREACH_THREAD(qz, z, q1d)
      {
         MFEM_FOREACH_THREAD(qy, y, q1d)
         {
            MFEM_FOREACH_THREAD(dx, x, d1d)
            {
               double u = 0.0, v = 0.0, w = 0.0;
               for (int qx = 0; qx < q1d; ++qx)
               {
                  u += U(qx, qy, qz, 0, c) * G(qx, dx);
                  v += U(qx, qy, qz, 1, c) * B(qx, dx);
                  w += U(qx, qy, qz, 2, c) * B(qx, dx);
               }
               smem(0, 0, qz, qy, dx) = u;
               smem(0, 1, qz, qy, dx) = v;
               smem(0, 2, qz, qy, dx) = w;
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(qz, z, q1d)
      {
         MFEM_FOREACH_THREAD(dy, y, d1d)
         {
            MFEM_FOREACH_THREAD(dx, x, d1d)
            {
               double u = 0.0, v = 0.0, w = 0.0;
               for (int qy = 0; qy < q1d; ++qy)
               {
                  u += smem(0, 0, qz, qy, dx) * B(qy, dy);
                  v += smem(0, 1, qz, qy, dx) * G(qy, dy);
                  w += smem(0, 2, qz, qy, dx) * B(qy, dy);
               }
               smem(1, 0, qz, dy, dx) = u;
               smem(1, 1, qz, dy, dx) = v;
               smem(1, 2, qz, dy, dx) = w;
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(dz, z, d1d)
      {
         MFEM_FOREACH_THREAD(dy, y, d1d)
         {
            MFEM_FOREACH_THREAD(dx, x, d1d)
            {
               double u = 0.0, v = 0.0, w = 0.0;
               for (int qz = 0; qz < q1d; ++qz)
               {
                  u += smem(1, 0, qz, dy, dx) * B(qz, dz);
                  v += smem(1, 1, qz, dy, dx) * B(qz, dz);
                  w += smem(1, 2, qz, dy, dx) * G(qz, dz);
               }
               const double sum = u + v + w;
               F(dx, dy, dz, c) += sum;
            }
         }
      }
      MFEM_SYNC_THREAD;
   }
}

/**
 * @brief Compute the gradient of all shape functions.
 *
 * @note TODO: Does not make use of shared memory on the GPU.
 *
 * @tparam dim Dimension.
 * @tparam d1d Number of degrees of freedom in 1D.
 * @tparam q1d er of quadrature points in 1D.
 * @param qx Quadrature point x index.
 * @param qy Quadrature point y index.
 * @param qz Quadrature point z index.
 * @param B Basis functions evaluated at quadrature points in column-major
 * layout q1d x d1d.
 * @param G Gradients of basis functions evaluated at quadrature points in
 * column major layout q1d x d1d.
 * @param invJ Inverse of the Jacobian at the quadrature point. Size is dim x
 * dim.
 *
 * @return Gradient of all shape functions at the quadrature point. Size is d1d
 * x d1d x d1d x dim.
 */
template <int dim, int d1d, int q1d>
static inline MFEM_HOST_DEVICE tensor<double, d1d, d1d, d1d, dim>
GradAllShapeFunctions(int qx, int qy, int qz,
                      const tensor<double, q1d, d1d> &B,
                      const tensor<double, q1d, d1d> &G,
                      const tensor<double, dim, dim> &invJ)
{
   tensor<double, d1d, d1d, d1d, dim> dphi_dx;
   // G (x) B (x) B
   // B (x) G (x) B
   // B (x) B (x) G
   for (int dx = 0; dx < d1d; dx++)
   {
      for (int dy = 0; dy < d1d; dy++)
      {
         for (int dz = 0; dz < d1d; dz++)
         {
            dphi_dx(dx,dy,dz) =
               transpose(invJ) *
               tensor<double, dim> {G(qx, dx) * B(qy, dy) * B(qz, dz),
                                    B(qx, dx) * G(qy, dy) * B(qz, dz),
                                    B(qx, dx) * B(qy, dy) * G(qz, dz)
                                   };
         }
      }
   }
   return dphi_dx;
}

} // namespace KernelHelpers

} // namespace mfem

#endif
