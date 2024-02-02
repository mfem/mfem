// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_NAVIER_STRESS_INTEGRATOR_HPP
#define MFEM_NAVIER_STRESS_INTEGRATOR_HPP

#include <mfem.hpp>
#include "kernel_helpers.hpp"

namespace mfem
{
namespace navier
{

using mfem::internal::tensor;
using mfem::internal::make_tensor;

// Bilinear form for S
class ShearStressIntegrator : public BilinearFormIntegrator
{
public:
   ShearStressIntegrator(const GridFunctionCoefficient &n,
                         const IntegrationRule &ir, const int m = 0) :
      BilinearFormIntegrator(&ir), nu(n), mode(m) {}

   void AssemblePA(const FiniteElementSpace &fes) override
   {
      // WARNING: Assumes tensor-product elements
      Mesh *mesh = fes.GetMesh();
      const FiniteElement &el = *fes.GetFE(0);
      dim = mesh->Dimension();
      ne = fes.GetNE();
      geom = mesh->GetGeometricFactors(*IntRule,
                                       GeometricFactors::JACOBIANS | GeometricFactors::DETERMINANTS);
      maps = &el.GetDofToQuad(*IntRule, DofToQuad::TENSOR);
      d1d = maps->ndof;
      q1d = maps->nqpt;

      const GridFunction *gf = nu.GetGridFunction();
      const FiniteElementSpace &gf_fes = *gf->FESpace();
      const QuadratureInterpolator *qi(gf_fes.GetQuadratureInterpolator(*IntRule));
      const Operator *R = gf_fes.GetElementRestriction(
                             ElementDofOrdering::LEXICOGRAPHIC);

      Vector xe(R->Height());
      xe.UseDevice(true);

      R->Mult(*gf, xe);
      qi->EnableTensorProducts();
      nu_wrap.SetSize(q1d * q1d * (dim == 3 ? q1d : 1) * ne);
      qi->Values(xe, nu_wrap);
   }

   void AddMultPA(const Vector &x, Vector &y) const override
   {
      if (dim == 2)
      {
         const int id = (d1d << 4) | q1d;
         switch (id)
         {
            case 0x22:
            {
               return ShearStressIntegratorApply2D<2, 2>(ne, maps->B, maps->G,
                                                         IntRule->GetWeights(),
                                                         geom->J,
                                                         geom->detJ, x, y, nu_wrap);
               break;
            }
            case 0x33:
            {
               return ShearStressIntegratorApply2D<3, 3>(ne, maps->B, maps->G,
                                                         IntRule->GetWeights(),
                                                         geom->J,
                                                         geom->detJ, x, y, nu_wrap);
               break;
            }
            case 0x55:
            {
               return ShearStressIntegratorApply2D<5, 5>(ne, maps->B, maps->G,
                                                         IntRule->GetWeights(),
                                                         geom->J,
                                                         geom->detJ, x, y, nu_wrap);
               break;
            }
            case 0x66:
            {
               return ShearStressIntegratorApply2D<6, 6>(ne, maps->B, maps->G,
                                                         IntRule->GetWeights(),
                                                         geom->J,
                                                         geom->detJ, x, y, nu_wrap);
               break;
            }
            case 0x77:
            {
               return ShearStressIntegratorApply2D<7, 7>(ne, maps->B, maps->G,
                                                         IntRule->GetWeights(),
                                                         geom->J,
                                                         geom->detJ, x, y, nu_wrap);
               break;
            }
            default:
               MFEM_ABORT("unknown kernel");
         }
      }
      else if (dim == 3)
      {
         const int id = (d1d << 4) | q1d;
         switch (id)
         {
            case 0x22:
            {
               return ShearStressIntegratorApply3D<2, 2>(ne, maps->B, maps->G,
                                                         IntRule->GetWeights(),
                                                         geom->J,
                                                         geom->detJ, x, y, nu_wrap);
               break;
            }
            case 0x33:
            {
               return ShearStressIntegratorApply3D<3, 3>(ne, maps->B, maps->G,
                                                         IntRule->GetWeights(),
                                                         geom->J,
                                                         geom->detJ, x, y, nu_wrap);
               break;
            }
            case 0x55:
            {
               return ShearStressIntegratorApply3D<5, 5>(ne, maps->B, maps->G,
                                                         IntRule->GetWeights(),
                                                         geom->J,
                                                         geom->detJ, x, y, nu_wrap);
               break;
            }
            case 0x77:
            {
               return ShearStressIntegratorApply3D<7, 7>(ne, maps->B, maps->G,
                                                         IntRule->GetWeights(),
                                                         geom->J,
                                                         geom->detJ, x, y, nu_wrap);
               break;
            }
            default:
               MFEM_ABORT("unknown kernel");
         }
      }
      MFEM_ABORT("unknown kernel");
   };

   template <int d1d, int q1d, int dim = 2> static inline
   void ShearStressIntegratorApply2D(
      const int ne,
      const Array<double> &B_,
      const Array<double> &G_,
      const Array<double> &W_,
      const Vector &Jacobian_,
      const Vector &detJ_,
      const Vector &X_, Vector &Y_,
      const Vector &nu_
   )
   {
      KernelHelpers::CheckMemoryRestriction(d1d, q1d);

      const tensor<double, q1d, d1d> &B =
      make_tensor<q1d, d1d>([&](int i, int j) { return B_[i + q1d*j]; });

      const tensor<double, q1d, d1d> &G =
      make_tensor<q1d, d1d>([&](int i, int j) { return G_[i + q1d*j]; });

      const auto qweights = Reshape(W_.Read(), q1d, q1d);
      const auto J = Reshape(Jacobian_.Read(), q1d, q1d, dim, dim, ne);
      const auto detJ = Reshape(detJ_.Read(), q1d, q1d, ne);
      const auto U = Reshape(X_.Read(), d1d, d1d, dim, ne);
      auto force = Reshape(Y_.ReadWrite(), d1d, d1d, dim, ne);
      auto nu = Reshape(nu_.Read(), q1d, q1d, ne);

      MFEM_FORALL_2D(e, ne, q1d, q1d, 1,
      {
         //  shared memory placeholders for temporary contraction results
         MFEM_SHARED tensor<double, 2, 3, q1d, q1d> smem;
         MFEM_SHARED tensor<double, q1d, q1d, dim, dim> invJ_nuS_detJw;
         MFEM_SHARED tensor<double, q1d, q1d, dim, dim> dudxi;

         const auto U_el = Reshape(&U(0, 0, 0, e), d1d, d1d, dim);
         KernelHelpers::CalcGrad(B, G, smem, U_el, dudxi);

         MFEM_FOREACH_THREAD(qx, x, q1d)
         {
            MFEM_FOREACH_THREAD(qy, y, q1d)
            {
               auto invJqp = inv(make_tensor<dim, dim>(
               [&](int i, int j) { return J(qx, qy, i, j, e); }));

               const auto dudx = dudxi(qy, qx) * invJqp;

               const auto nuS = nu(qx, qy, e) * (dudx + transpose(dudx));
               invJ_nuS_detJw(qx, qy) = invJqp * transpose(nuS)
                                        * detJ(qx, qy, e) * qweights(qx, qy);
            }
         }
         MFEM_SYNC_THREAD;
         auto F = Reshape(&force(0, 0, 0, e), d1d, d1d, dim);
         KernelHelpers::CalcGradTSum(B, G, smem, invJ_nuS_detJw, F);
      });
   }

   template <int d1d, int q1d, int dim = 3> static inline
   void ShearStressIntegratorApply3D(
      const int ne,
      const Array<double> &B_,
      const Array<double> &G_,
      const Array<double> &W_,
      const Vector &Jacobian_,
      const Vector &detJ_,
      const Vector &X_, Vector &Y_,
      const Vector &nu_
   )
   {
      KernelHelpers::CheckMemoryRestriction(d1d, q1d);

      const tensor<double, q1d, d1d> &B =
      make_tensor<q1d, d1d>([&](int i, int j) { return B_[i + q1d*j]; });

      const tensor<double, q1d, d1d> &G =
      make_tensor<q1d, d1d>([&](int i, int j) { return G_[i + q1d*j]; });

      const auto qweights = Reshape(W_.Read(), q1d, q1d, q1d);
      const auto J = Reshape(Jacobian_.Read(), q1d, q1d, q1d, dim, dim, ne);
      const auto detJ = Reshape(detJ_.Read(), q1d, q1d, q1d, ne);
      const auto U = Reshape(X_.Read(), d1d, d1d, d1d, dim, ne);
      auto force = Reshape(Y_.ReadWrite(), d1d, d1d, d1d, dim, ne);
      auto nu = Reshape(nu_.Read(), q1d, q1d, q1d, ne);

      MFEM_FORALL_3D(e, ne, q1d, q1d, q1d,
      {
         //  shared memory placeholders for temporary contraction results
         MFEM_SHARED tensor<double, 2, 3, q1d, q1d, q1d> smem;
         MFEM_SHARED tensor<double, q1d, q1d, q1d, dim, dim> invJ_nuS_detJw;
         MFEM_SHARED tensor<double, q1d, q1d, q1d, dim, dim> dudxi;

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

                  const auto dudx = dudxi(qz, qy, qx) * invJqp;

                  const auto nuS = nu(qx, qy, qz, e) * (dudx + transpose(dudx));
                  invJ_nuS_detJw(qx, qy, qz) = invJqp * transpose(nuS)
                                               * detJ(qx, qy, qz, e) * qweights(qx, qy, qz);
               }
            }
         }
         MFEM_SYNC_THREAD;
         auto F = Reshape(&force(0, 0, 0, 0, e), d1d, d1d, d1d, dim);
         KernelHelpers::CalcGradTSum(B, G, smem, invJ_nuS_detJw, F);
      });
   }

protected:
   int dim;
   /// Number of elements in the mesh (rank local)
   int ne;
   const DofToQuad *maps = nullptr;
   const GeometricFactors *geom = nullptr;
   const GridFunctionCoefficient &nu;
   // const IntegrationRule &ir;
   // Number of degrees of freedom in 1D
   int d1d;
   /// Number of quadrature points in 1D
   int q1d;
   Vector nu_wrap;
   int mode;
};

} // namespace navier
} // namespace mfem

#endif // MFEM_NAVIER_STRESS_INTEGRATOR_HPP