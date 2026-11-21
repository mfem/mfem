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

#include "shear_stress_integrator.hpp"

namespace mfem
{
namespace navier
{

using mfem::future::tensor;
using mfem::future::make_tensor;

ShearStressIntegrator::ShearStressIntegrator(GridFunctionCoefficient &n,
                                             const IntegrationRule &ir, const int m)
   : BilinearFormIntegrator(&ir), nu(n), mode(m) {}

void ShearStressIntegrator::AssembleElementMatrix(const FiniteElement &el,
                                                  ElementTransformation &Trans,
                                                  DenseMatrix &elmat)
{
   const int dof = el.GetDof();
   dim = el.GetDim();
   vdim = dim;

#ifdef MFEM_THREAD_SAFE
   DenseMatrix dshape(dof, dim), pelmat(dof);
#else
   dshape.SetSize(dof, dim);
   pelmat.SetSize(dof);
#endif

   elmat.SetSize(vdim * dof);

   const IntegrationRule *ir = GetIntegrationRule();
   if (ir == NULL)
   {
      ir = &DiffusionIntegrator::GetRule(el, el);
   }

   elmat = 0.0;

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      Trans.SetIntPoint(&ip);
      real_t norm = ip.weight * Trans.Weight();

      el.CalcPhysDShape(Trans, dshape);
      MultAAt(dshape, pelmat);

      // Evaluate viscosity coefficient
      norm *= nu.Eval(Trans, ip);

      // First term: ν * ∇u_i · ∇v_i (diagonal blocks)
      for (int d = 0; d < dim; d++)
      {
         for (int k = 0; k < dof; k++)
            for (int l = 0; l < dof; l++)
            {
               elmat(dof * d + k, dof * d + l) += norm * pelmat(k, l);
            }
      }

      // Second term: ν * ∇_j u_i · ∇_i v_j (off-diagonal terms)
      for (int ii = 0; ii < dim; ii++)
         for (int jj = 0; jj < dim; jj++)
         {
            for (int kk = 0; kk < dof; kk++)
               for (int ll = 0; ll < dof; ll++)
               {
                  elmat(dof * ii + kk, dof * jj + ll) +=
                      norm * dshape(kk, jj) * dshape(ll, ii);
               }
         }

         // Otherwise we could compute S = nu (∇u + ∇u^T), and then compute S : ∇v ?
   }
}

void ShearStressIntegrator::AssembleElementVector(const FiniteElement &el,
                                                  ElementTransformation &Trans,
                                                  const Vector &elfun, Vector &elvect)
{
   // TODO: Implement element vector assembly
   // This method should assemble the local element vector for the shear stress integrator
   MFEM_ABORT("AssembleElementVector not implemented for ShearStressIntegrator");
}


void ShearStressIntegrator::AssemblePA(const FiniteElementSpace &fes)
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

void ShearStressIntegrator::AddMultPA(const Vector &x, Vector &y) const
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
         /*case 0x77:   // disabled as uses too much shared data in ShearStressIntegratorApply3D kernel
         {
            return ShearStressIntegratorApply3D<7, 7>(ne, maps->B, maps->G,
                                                      IntRule->GetWeights(),
                                                      geom->J,
                                                      geom->detJ, x, y, nu_wrap);
            break;
         } */
         default:
            MFEM_ABORT("unknown kernel");
      }
   }
   MFEM_ABORT("unknown kernel");
}

template <int d1d, int q1d, int dim>
void ShearStressIntegrator::ShearStressIntegratorApply2D(
   const int ne,
   const Array<real_t> &B_,
   const Array<real_t> &G_,
   const Array<real_t> &W_,
   const Vector &Jacobian_,
   const Vector &detJ_,
   const Vector &X_, Vector &Y_,
   const Vector &nu_
)
{
   KernelHelpers::CheckMemoryRestriction(d1d, q1d);

   const tensor<real_t, q1d, d1d> &B =
   make_tensor<q1d, d1d>([&](int i, int j) { return B_[i + q1d*j]; });

   const tensor<real_t, q1d, d1d> &G =
   make_tensor<q1d, d1d>([&](int i, int j) { return G_[i + q1d*j]; });

   const auto qweights = Reshape(W_.Read(), q1d, q1d);
   const auto J = Reshape(Jacobian_.Read(), q1d, q1d, dim, dim, ne);
   const auto detJ = Reshape(detJ_.Read(), q1d, q1d, ne);
   const auto U = Reshape(X_.Read(), d1d, d1d, dim, ne);
   auto force = Reshape(Y_.ReadWrite(), d1d, d1d, dim, ne);
   auto nu = Reshape(nu_.Read(), q1d, q1d, ne);

   mfem::forall_2D(ne, q1d, q1d, [=] MFEM_HOST_DEVICE (int e)
   {
      //  shared memory placeholders for temporary contraction results
      MFEM_SHARED tensor<real_t, 2, 3, q1d, q1d> smem;
      MFEM_SHARED tensor<real_t, q1d, q1d, dim, dim> invJ_nuS_detJw;
      MFEM_SHARED tensor<real_t, q1d, q1d, dim, dim> dudxi;

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

template <int d1d, int q1d, int dim>
void ShearStressIntegrator::ShearStressIntegratorApply3D(
   const int ne,
   const Array<real_t> &B_,
   const Array<real_t> &G_,
   const Array<real_t> &W_,
   const Vector &Jacobian_,
   const Vector &detJ_,
   const Vector &X_, Vector &Y_,
   const Vector &nu_
)
{
   KernelHelpers::CheckMemoryRestriction(d1d, q1d);

   const tensor<real_t, q1d, d1d> &B =
   make_tensor<q1d, d1d>([&](int i, int j) { return B_[i + q1d*j]; });

   const tensor<real_t, q1d, d1d> &G =
   make_tensor<q1d, d1d>([&](int i, int j) { return G_[i + q1d*j]; });

   const auto qweights = Reshape(W_.Read(), q1d, q1d, q1d);
   const auto J = Reshape(Jacobian_.Read(), q1d, q1d, q1d, dim, dim, ne);
   const auto detJ = Reshape(detJ_.Read(), q1d, q1d, q1d, ne);
   const auto U = Reshape(X_.Read(), d1d, d1d, d1d, dim, ne);
   auto force = Reshape(Y_.ReadWrite(), d1d, d1d, d1d, dim, ne);
   auto nu = Reshape(nu_.Read(), q1d, q1d, q1d, ne);

   mfem::forall_3D(ne, q1d, q1d, q1d, [=] MFEM_HOST_DEVICE (int e)
   {
      //  shared memory placeholders for temporary contraction results
      MFEM_SHARED tensor<real_t, 2, 3, q1d, q1d, q1d> smem;
      MFEM_SHARED tensor<real_t, q1d, q1d, q1d, dim, dim> invJ_nuS_detJw;
      // du/dxi
      MFEM_SHARED_3D_BLOCK_TENSOR( dudxi, real_t, q1d, q1d, q1d, dim, dim);
      
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

// Explicit template instantiations for commonly used cases
template void ShearStressIntegrator::ShearStressIntegratorApply2D<2, 2, 2>(const int, const Array<real_t>&, const Array<real_t>&, const Array<real_t>&, const Vector&, const Vector&, const Vector&, Vector&, const Vector&);
template void ShearStressIntegrator::ShearStressIntegratorApply2D<3, 3, 2>(const int, const Array<real_t>&, const Array<real_t>&, const Array<real_t>&, const Vector&, const Vector&, const Vector&, Vector&, const Vector&);
template void ShearStressIntegrator::ShearStressIntegratorApply2D<5, 5, 2>(const int, const Array<real_t>&, const Array<real_t>&, const Array<real_t>&, const Vector&, const Vector&, const Vector&, Vector&, const Vector&);
template void ShearStressIntegrator::ShearStressIntegratorApply2D<6, 6, 2>(const int, const Array<real_t>&, const Array<real_t>&, const Array<real_t>&, const Vector&, const Vector&, const Vector&, Vector&, const Vector&);
template void ShearStressIntegrator::ShearStressIntegratorApply2D<7, 7, 2>(const int, const Array<real_t>&, const Array<real_t>&, const Array<real_t>&, const Vector&, const Vector&, const Vector&, Vector&, const Vector&);

template void ShearStressIntegrator::ShearStressIntegratorApply3D<2, 2, 3>(const int, const Array<real_t>&, const Array<real_t>&, const Array<real_t>&, const Vector&, const Vector&, const Vector&, Vector&, const Vector&);
template void ShearStressIntegrator::ShearStressIntegratorApply3D<3, 3, 3>(const int, const Array<real_t>&, const Array<real_t>&, const Array<real_t>&, const Vector&, const Vector&, const Vector&, Vector&, const Vector&);
template void ShearStressIntegrator::ShearStressIntegratorApply3D<5, 5, 3>(const int, const Array<real_t>&, const Array<real_t>&, const Array<real_t>&, const Vector&, const Vector&, const Vector&, Vector&, const Vector&);

} // namespace navier
} // namespace mfem