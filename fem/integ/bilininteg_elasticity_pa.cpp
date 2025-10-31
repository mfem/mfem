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

#include "../bilininteg.hpp"
#include "../gridfunc.hpp"
#include "../qfunction.hpp"
#include "../../mesh/nurbs.hpp"
#include "bilininteg_elasticity_kernels.hpp"
#include "bilininteg_patch.hpp"
#include "../integrator.hpp"
#include "../../linalg/dtensor.hpp"
#include "../../linalg/tensor.hpp"
using mfem::future::tensor;
using mfem::future::make_tensor;

namespace mfem
{

void ElasticityIntegrator::SetUpQuadratureSpaceAndCoefficients(
   const FiniteElementSpace &fes)
{
   if (IntRule == nullptr)
   {
      // This is where it's assumed that all elements are the same.
      const auto &T = *fes.GetMesh()->GetTypicalElementTransformation();
      int quad_order = 2 * T.OrderGrad(fes.GetTypicalFE());
      IntRule = &IntRules.Get(T.GetGeometryType(), quad_order);
   }

   Mesh &mesh = *fespace->GetMesh();

   q_space.reset(new QuadratureSpace(mesh, *IntRule));
   lambda_quad.reset(new CoefficientVector(lambda, *q_space,
                                           CoefficientStorage::FULL));
   mu_quad.reset(new CoefficientVector(mu, *q_space, CoefficientStorage::FULL));
   q_vec.reset(new QuadratureFunction(*q_space, vdim*vdim));
}

void ElasticityIntegrator::AssemblePA(const FiniteElementSpace &fes)
{
   MFEM_VERIFY(fes.GetOrdering() == Ordering::byNODES,
               "Elasticity PA only implemented for byNODES ordering.");

   fespace = &fes;
   Mesh &mesh = *fespace->GetMesh();
   vdim = fespace->GetVDim();
   MFEM_VERIFY(vdim == mesh.Dimension(),
               "Vector dimension and geometric dimension must match.");
   ndofs = fespace->GetTypicalFE()->GetDof();

   SetUpQuadratureSpaceAndCoefficients(fes);

   auto ordering = GetEVectorOrdering(*fespace);
   auto mode = ordering == ElementDofOrdering::NATIVE ? DofToQuad::FULL :
               DofToQuad::LEXICOGRAPHIC_FULL;
   maps = &fespace->GetTypicalFE()->GetDofToQuad(*IntRule, mode);
   geom = mesh.GetGeometricFactors(*IntRule, GeometricFactors::JACOBIANS);
}

void ElasticityIntegrator::AssembleDiagonalPA(Vector &diag)
{
   q_vec->SetVDim(vdim*vdim*vdim*vdim);
   internal::ElasticityAssembleDiagonalPA(vdim, ndofs, *lambda_quad, *mu_quad,
                                          *geom, *maps, *q_vec, diag);
}

void ElasticityIntegrator::AddMultPA(const Vector &x, Vector &y) const
{
   internal::ElasticityAddMultPA(vdim, ndofs, *fespace, *lambda_quad, *mu_quad,
                                 *geom, *maps, x, *q_vec, y);
}

void ElasticityIntegrator::AddMultTransposePA(const Vector &x, Vector &y) const
{
   AddMultPA(x, y); // Operator is symmetric
}

void ElasticityComponentIntegrator::AssemblePA(const FiniteElementSpace &fes)
{
   fespace = &fes;

   // Avoid projecting the coefficients more than once. If the coefficients
   // change, the parent ElasticityIntegrator must be reassembled.
   if (!parent.q_space)
   {
      parent.SetUpQuadratureSpaceAndCoefficients(fes);
   }
   else
   {
      IntRule = parent.IntRule;
   }

   auto ordering = GetEVectorOrdering(*fespace);
   auto mode = ordering == ElementDofOrdering::NATIVE ? DofToQuad::FULL :
               DofToQuad::LEXICOGRAPHIC_FULL;
   geom = fes.GetMesh()->GetGeometricFactors(*IntRule,
                                             GeometricFactors::JACOBIANS);
   maps = &fespace->GetTypicalFE()->GetDofToQuad(*IntRule, mode);
}

void ElasticityComponentIntegrator::AddMultPA(const Vector &x, Vector &y) const
{
   internal::ElasticityComponentAddMultPA(
      parent.vdim, parent.ndofs, *fespace, *parent.lambda_quad, *parent.mu_quad,
      *geom, *maps, x, *parent.q_vec, y, i_block, j_block);
}

void ElasticityComponentIntegrator::AddMultTransposePA(const Vector &x,
                                                       Vector &y) const
{
   // Each block in the operator is symmetric, so we can just switch the roles
   // of i_block and j_block
   internal::ElasticityComponentAddMultPA(
      parent.vdim, parent.ndofs, *fespace, *parent.lambda_quad, *parent.mu_quad,
      *geom, *maps, x, *parent.q_vec, y, j_block, i_block);
}

void ElasticityIntegrator::AssembleNURBSPA(const FiniteElementSpace &fes)
{

   fespace = &fes;
   Mesh &mesh = *fespace->GetMesh();
   vdim = fespace->GetVDim();
   MFEM_VERIFY(vdim == mesh.Dimension(),
               "Vector dimension and geometric dimension must match.");

   numPatches = mesh.NURBSext->GetNP();
   ppa_data.resize(numPatches);
   for (int p=0; p<numPatches; ++p)
   {
      AssemblePatchPA(p, fes);
   }
}

void ElasticityIntegrator::AssemblePatchPA(const int patch,
                                           const FiniteElementSpace &fes)
{
   Mesh *mesh = fes.GetMesh();
   SetupPatchBasisData(mesh, patch);
   SetupPatchPA(patch, mesh);  // For full quadrature, unitWeights = false
}

/**
 * Transform grad_uhat (reference) to physical space:
 *    grad_u = grad_uhat * J^{-1}
 *
 * Compute stress:
 *    sigma = lambda*tr(grad_u)*I + mu*(grad_u + grad_u^T)
 *
 * lambda and mu have W*detJ factored in
 *
 * Transform back to reference space:
 *    return sigma * J^{-T}
 */
template <int dim>
tensor<real_t, dim, dim>
LinearElasticStress(const tensor<real_t, dim, dim> Jinvt,
                    const real_t lambda,
                    const real_t mu,
                    const tensor<real_t, dim, dim> gradu_ref)
{
   // Convert gradu_ref to physical space
   const auto gradu = gradu_ref * transpose(Jinvt);
   // Compute stress
   constexpr auto I = mfem::future::IsotropicIdentity<dim>();
   const tensor<real_t, dim, dim> strain = sym(gradu);
   const tensor<real_t, dim, dim> stress =
      lambda * tr(strain) * I + static_cast<real_t>(2.0) * mu * strain;
   // Transform back to reference space
   return stress * Jinvt;
}

/**
 * Transforms grad_u into physical space, computes stress,
   then transforms back to reference space
 */
void PatchApplyKernel3D(const PatchBasisInfo &pb,
                        const Vector &pa_data,
                        DeviceTensor<5, real_t> &gradu,
                        DeviceTensor<5, real_t> &S)
{
   // Unpack patch basis info
   static constexpr int dim = 3;
   const Array<int>& Q1D = pb.Q1D;
   const int NQ = pb.NQ;
   // Quadrature data. 11 values per quadrature point.
   // First 9 entries are J^{-T}; then lambda*W*detJ and mu*W*detJ
   const auto qd = Reshape(pa_data.HostRead(), NQ, 11);

   for (int qz = 0; qz < Q1D[2]; ++qz)
   {
      for (int qy = 0; qy < Q1D[1]; ++qy)
      {
         for (int qx = 0; qx < Q1D[0]; ++qx)
         {
            const int q = qx + ((qy + (qz * Q1D[1])) * Q1D[0]);
            const real_t lambda  = qd(q,9);
            const real_t mu      = qd(q,10);
            const auto Jinvt = make_tensor<dim, dim>(
            [&](int i, int j) { return qd(q, i*dim + j); });
            const auto gradu_q = make_tensor<dim, dim>(
            [&](int i, int j) { return gradu(i,j,qx,qy,qz); });
            const auto Sq = LinearElasticStress<dim>(Jinvt, lambda, mu, gradu_q);

            for (int i = 0; i < dim; ++i)
            {
               for (int j = 0; j < dim; ++j)
               {
                  S(i,j,qx,qy,qz) = Sq(i,j);
               }
            }
         } // qx
      } // qy
   } // qz
}


// This version uses full 1D quadrature rules, taking into account the
// minimum interaction between basis functions and integration points.
void ElasticityIntegrator::AddMultPatchPA3D(const Vector &pa_data,
                                            const PatchBasisInfo &pb,
                                            const Vector &x,
                                            Vector &y) const
{
   // Unpack patch basis info
   const Array<int>& Q1D = pb.Q1D;
   const int NQ = pb.NQ;
   MFEM_VERIFY((pb.dim == 3) && (vdim == 3), "Dimension mismatch.");

   // gradu(i,j,q): d/d(x_j) u_i(x_q)
   Vector graduv(vdim*vdim*NQ);
   graduv = 0.0;
   auto gradu = Reshape(graduv.HostReadWrite(), vdim, vdim, Q1D[0], Q1D[1],
                        Q1D[2]);

   // S[i,j,q] = D( gradu )
   //          = stress[i,j,q] * J^{-T}[q]
   Vector Sv(vdim*vdim*NQ);
   Sv = 0.0;
   auto S = Reshape(Sv.HostReadWrite(), vdim, vdim, Q1D[0], Q1D[1], Q1D[2]);

   Vector sumXYv(vdim*vdim*pb.MAX1D[0]*pb.MAX1D[1]);
   Vector sumXv(vdim*vdim*pb.MAX1D[0]);

   // 1) Interpolate U at dofs to gradu in reference quadrature space
   PatchG3D<3>(pb, x, sumXYv, sumXv, gradu);

   // 2) Apply the "D" operator at each quadrature point: D( gradu )
   PatchApplyKernel3D(pb, pa_data, gradu, S);

   // 3) Apply test function gradv
   PatchGT3D<3>(pb, S, sumXYv, sumXv, y);

}

void ElasticityIntegrator::AddMultPatchPA(const int patch, const Vector &x,
                                          Vector &y) const
{
   if (vdim == 3)
   {
      AddMultPatchPA3D(ppa_data[patch], pbinfo[patch], x, y);
   }
   else
   {
      MFEM_ABORT("Only 3D is supported.");
   }
}


void ElasticityIntegrator::AddMultNURBSPA(const Vector &x, Vector &y) const
{
   Vector xp, yp;

   for (int p=0; p<numPatches; ++p)
   {
      Array<int> vdofs;
      fespace->GetPatchVDofs(p, vdofs);

      x.GetSubVector(vdofs, xp);
      yp.SetSize(vdofs.Size());
      yp = 0.0;

      AddMultPatchPA(p, xp, yp);

      y.AddElementVector(vdofs, yp);
   }
}

void ElasticityIntegrator::AssembleDiagonalPatchPA(const Vector &pa_data,
                                                   const PatchBasisInfo &pb,
                                                   Vector &diag) const
{
   // Unpack patch basis info
   const Array<int>& D1D = pb.D1D;
   const Array<int>& Q1D = pb.Q1D;
   const int NQ = pb.NQ;
   const std::vector<Array2D<real_t>>& B = pb.B;
   const std::vector<Array2D<real_t>>& G = pb.G;
   const std::vector<std::vector<int>> minD = pb.minD;
   const std::vector<std::vector<int>> maxD = pb.maxD;

   const auto qd = Reshape(pa_data.HostRead(), NQ, 11);

   auto Y = Reshape(diag.HostReadWrite(), D1D[0], D1D[1], D1D[2], 3);

   for (int dz = 0; dz < D1D[2]; ++dz)
   {
      for (int dy = 0; dy < D1D[1]; ++dy)
      {
         for (int dx = 0; dx < D1D[0]; ++dx)
         {
            for (int qz = minD[2][dz]; qz <= maxD[2][dz]; ++qz)
            {
               const real_t Bz = B[2](qz,dz);
               const real_t Gz = G[2](qz,dz);
               for (int qy = minD[1][dy]; qy <= maxD[1][dy]; ++qy)
               {
                  const real_t By = B[1](qy,dy);
                  const real_t Gy = G[1](qy,dy);
                  for (int qx = minD[0][dx]; qx <= maxD[0][dx]; ++qx)
                  {
                     const real_t Bx = B[0](qx,dx);
                     const real_t Gx = G[0](qx,dx);
                     const real_t grad[3] =
                     {
                        Gx * By * Bz,
                        Bx * Gy * Bz,
                        Bx * By * Gz
                     };

                     const int q = qx + ((qy + (qz * Q1D[1])) * Q1D[0]);
                     const real_t lambda_q  = qd(q,9);
                     const real_t mu_q      = qd(q,10);
                     const auto Jinvt = make_tensor<3,3>(
                     [&](int i, int j) { return qd(q, i*3 + j); });
                     // As second order tensor: e[i] * grad_phi[j]
                     const auto grad_phi = make_tensor<3,3>(
                     [&](int i, int j) { return grad[j]; });

                     const auto Sq = LinearElasticStress<3>(
                                        Jinvt, lambda_q, mu_q, grad_phi);

                     const auto grad_phiv = make_tensor<3>(
                     [&](int i) { return grad[i]; });
                     const auto Yq = dot(Sq, grad_phiv);
                     for (int c = 0; c < 3; ++c)
                     {
                        Y(dx,dy,dz,c) += Yq[c];
                     }
                  } // qx
               } // qy
            } // qz
         } // dx
      } // dy
   } // dz
}


void ElasticityIntegrator::AssembleDiagonalNURBSPA(Vector &diag)
{
   Vector diagp;

   for (int p=0; p<numPatches; ++p)
   {
      Array<int> vdofs;
      fespace->GetPatchVDofs(p, vdofs);
      diagp.SetSize(vdofs.Size());
      diagp = 0.0;

      AssembleDiagonalPatchPA(ppa_data[p], pbinfo[p], diagp);

      diag.AddElementVector(vdofs, diagp);
   }
}

} // namespace mfem
