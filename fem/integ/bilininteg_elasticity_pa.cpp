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
#include "linalg/dtensor.hpp"

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
   MFEM_VERIFY(vdim == mesh.Dimension(), "Vector dimension and geometric dimension must match.");
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
   MFEM_VERIFY(vdim == mesh.Dimension(), "Vector dimension and geometric dimension must match.");
   ndofs = fespace->GetTypicalFE()->GetDof();

   numPatches = mesh.NURBSext->GetNP();
   for (int p=0; p<numPatches; ++p)
   {
      mfem::out << "assembling patch " << p << std::endl;
      AssemblePatchPA(p, fes);
   }
}

void ElasticityIntegrator::AssemblePatchPA(const int patch,
                                           const FiniteElementSpace &fes)
{
   // TODO
   Mesh *mesh = fes.GetMesh();
   mfem::out << "AssemblePatchPA() " << patch << std::endl;
   SetupPatchBasisData(mesh, patch);

   SetupPatchPA(patch, mesh);  // For full quadrature, unitWeights = false
}

// This version uses full 1D quadrature rules, taking into account the
// minimum interaction between basis functions and integration points.
void ElasticityIntegrator::AddMultPatchPA(const int patch, const Vector &x,
                                          Vector &y) const
{
   MFEM_VERIFY(3 == vdim, "Only 3D so far");
   mfem::out << "AddMultPatchPA() " << patch << std::endl;

   // # of quadrature points in each dimension for this patch
   const Array<int>& Q1D = pQ1D[patch];
   // # of DOFs in each dimension for this patch
   const Array<int>& D1D = pD1D[patch];
   // Shape functions (B) and their derivatives (G) for this patch
   const std::vector<Array2D<real_t>>& B = pB[patch];
   const std::vector<Array2D<real_t>>& G = pG[patch];
   // for reduced rules
   const IntArrayVar2D& minD = pminD[patch];
   const IntArrayVar2D& maxD = pmaxD[patch];
   const IntArrayVar2D& minQ = pminQ[patch];
   const IntArrayVar2D& maxQ = pmaxQ[patch];

   const int NQ = Q1D[0] * Q1D[1] * Q1D[2];
   const int max_pts_x = std::max(Q1D[0], D1D[0]);
   const int max_pts_y = std::max(Q1D[1], D1D[1]);

   mfem::out << "AddMultPatchPA(): Size of x = " << x.Size() << std::endl;
   auto X = Reshape(x.HostRead(), vdim, D1D[0], D1D[1], D1D[2]);
   auto Y = Reshape(y.HostReadWrite(), vdim, D1D[0], D1D[1], D1D[2]);

   // First 9 entries are J^{-T}, last entry is W*detJ
   const auto qd = Reshape(pa_data.HostRead(), NQ, 10);

   // grad(c,d,qx,qy,qz)
   // derivative of u_c w.r.t. d evaluated at (qx,qy,qz)
   Vector gradv(NQ);
   auto grad = Reshape(gradv.HostReadWrite(), vdim, vdim, Q1D[0], Q1D[1], Q1D[2]);

   // Accumulators
   Vector gradXYv(vdim*vdim*max_pts_x*max_pts_y);
   Vector gradXv(vdim*vdim*max_pts_x);
   auto gradXY = Reshape(gradXYv.HostReadWrite(), vdim, vdim, max_pts_x, max_pts_y);
   auto gradX = Reshape(gradXv.HostReadWrite(), vdim, vdim, max_pts_x);

   mfem::out << "AddMultPatchPA() " << patch << " - finished 1) init grad u" << std::endl;
   mfem::out << "grad[0][0](1,1,1) = " << grad(0,0,1,1,1) << std::endl;

   // 2) compute grad(u) interpolated at quadrature points
   // B_{nq} U_n = \sum_n u_n \tilde{\nabla} \tilde{\phi}_n (\tilde{x}_q)
   // Because shape functions are tensor prodcuts they are decomposed
   // for (int c = 0; c < vdim; ++c)
   // {
   //    for (int dz = 0; dz < D1D[2]; ++dz)
   //    {
   //       gradXY = 0.0;
   //       for (int dy = 0; dy < D1D[1]; ++dy)
   //       {
   //          gradX = 0.0;
   //          for (int dx = 0; dx < D1D[0]; ++dx)
   //          {
   //             const real_t s = X(c, dx,dy,dz);
   //             for (int qx = minD[0][dx]; qx <= maxD[0][dx]; ++qx)
   //             {
   //                gradX(0,qx) += s * B[0](qx,dx);
   //                gradX(1,qx) += s * G[0](qx,dx);
   //             }
   //          }
   //          for (int qy = minD[1][dy]; qy <= maxD[1][dy]; ++qy)
   //          {
   //             const real_t wy  = B[1](qy,dy);
   //             const real_t wDy = G[1](qy,dy);
   //             // This full range of qx values is generally necessary.
   //             for (int qx = 0; qx < Q1D[0]; ++qx)
   //             {
   //                const real_t wx  = gradX(0,qx);
   //                const real_t wDx = gradX(1,qx);
   //                gradXY(0,qx,qy) += wDx * wy;
   //                gradXY(1,qx,qy) += wx  * wDy;
   //                gradXY(2,qx,qy) += wx  * wy;
   //             }
   //          }
   //       }
   //       for (int qz = minD[2][dz]; qz <= maxD[2][dz]; ++qz)
   //       {
   //          const real_t wz  = B[2](qz,dz);
   //          const real_t wDz = G[2](qz,dz);
   //          for (int qy = 0; qy < Q1D[1]; ++qy)
   //          {
   //             for (int qx = 0; qx < Q1D[0]; ++qx)
   //             {
   //                grad[c][0](qx,qy,qz) += gradXY(0,qx,qy) * wz;
   //                grad[c][1](qx,qy,qz) += gradXY(1,qx,qy) * wz;
   //                grad[c][2](qx,qy,qz) += gradXY(2,qx,qy) * wDz;
   //             }
   //          }
   //       }
   //    }
   // }

   // // mfem::out << "AddMultPatchPA() " << patch << " - finished 2) compute grad u" << std::endl;
   // // mfem::out << "grad[0](1,2,1) = " << grad[0](1,2,1) << std::endl;

   // // 3) Apply the "D" operator at each quadrature point
   // // D ( grad(u) )
   // for (int qz = 0; qz < Q1D[2]; ++qz)
   // {
   //    for (int qy = 0; qy < Q1D[1]; ++qy)
   //    {
   //       for (int qx = 0; qx < Q1D[0]; ++qx)
   //       {
   //          const int q = qx + ((qy + (qz * Q1D[1])) * Q1D[0]);
   //          const real_t Jinvt00 = qd(q,0);
   //          const real_t Jinvt01 = qd(q,1);
   //          const real_t Jinvt02 = qd(q,2);
   //          const real_t Jinvt10 = qd(q,3);
   //          const real_t Jinvt11 = qd(q,4);
   //          const real_t Jinvt12 = qd(q,5);
   //          const real_t Jinvt20 = qd(q,6);
   //          const real_t Jinvt21 = qd(q,7);
   //          const real_t Jinvt22 = qd(q,8);
   //          const real_t wdetj   = qd(q,9);

   //          // get grad_u = J^{-T} * grad_uhat
   //          // ...
   //          // grad[0](qx,qy,qz) = (O00*grad0)+(O01*grad1)+(O02*grad2);
   //          // grad[1](qx,qy,qz) = (O10*grad0)+(O11*grad1)+(O12*grad2);
   //          // grad[2](qx,qy,qz) = (O20*grad0)+(O21*grad1)+(O22*grad2);


   //          // lambda*div(u)

   //          // mu*strain(u)

   //          const real_t grad0 = grad[0](qx,qy,qz);
   //          const real_t grad1 = grad[1](qx,qy,qz);
   //          const real_t grad2 = grad[2](qx,qy,qz);

   //          // ... grad[8]
   //          // apply D(grad(u))


   //       } // qx
   //    } // qy
   // } // qz

   // // mfem::out << "AddMultPatchPA() " << patch << " - finished 3) apply D" << std::endl;
   // // mfem::out << "grad[0](1,1,1) = " << grad[0](1,1,1) << std::endl;

   // // 4) Add the contributions
   // // for c, vdim
   // for (int qz = 0; qz < Q1D[2]; ++qz)
   // {
   //    for (int dy = 0; dy < D1D[1]; ++dy)
   //    {
   //       for (int dx = 0; dx < D1D[0]; ++dx)
   //       {
   //          for (int d=0; d<3; ++d)
   //          {
   //             gradXY(d,dx,dy) = 0.0;
   //          }
   //       }
   //    }
   //    for (int qy = 0; qy < Q1D[1]; ++qy)
   //    {
   //       for (int dx = 0; dx < D1D[0]; ++dx)
   //       {
   //          for (int d=0; d<3; ++d)
   //          {
   //             gradX(d,dx) = 0.0;
   //          }
   //       }
   //       for (int qx = 0; qx < Q1D[0]; ++qx)
   //       {
   //          const real_t gX = grad[c][0](qx,qy,qz);
   //          const real_t gY = grad[c][1](qx,qy,qz);
   //          const real_t gZ = grad[c][2](qx,qy,qz);
   //          for (int dx = minQ[0][qx]; dx <= maxQ[0][qx]; ++dx)
   //          {
   //             const real_t wx  = B[0](qx,dx);
   //             const real_t wDx = G[0](qx,dx);
   //             gradX(0,dx) += gX * wDx;
   //             gradX(1,dx) += gY * wx;
   //             gradX(2,dx) += gZ * wx;
   //          }
   //       }
   //       for (int dy = minQ[1][qy]; dy <= maxQ[1][qy]; ++dy)
   //       {
   //          const real_t wy  = B[1](qy,dy);
   //          const real_t wDy = G[1](qy,dy);
   //          for (int dx = 0; dx < D1D[0]; ++dx)
   //          {
   //             gradXY(0,dx,dy) += gradX(0,dx) * wy;
   //             gradXY(1,dx,dy) += gradX(1,dx) * wDy;
   //             gradXY(2,dx,dy) += gradX(2,dx) * wy;
   //          }
   //       }
   //    }
   //    for (int dz = minQ[2][qz]; dz <= maxQ[2][qz]; ++dz)
   //    {
   //       const real_t wz  = B[2](qz,dz);
   //       const real_t wDz = G[2](qz,dz);
   //       for (int dy = 0; dy < D1D[1]; ++dy)
   //       {
   //          for (int dx = 0; dx < D1D[0]; ++dx)
   //          {
   //             Y(c,dx,dy,dz) +=
   //                ((gradXY(0,dx,dy) * wz) +
   //                 (gradXY(1,dx,dy) * wz) +
   //                 (gradXY(2,dx,dy) * wDz));
   //          }
   //       }
   //    } // dz
   // } // qz
}

void ElasticityIntegrator::AddMultNURBSPA(const Vector &x, Vector &y) const
{
   mfem::out << "AddMultNURBSPA() " << std::endl;
   Vector xp, yp;

   for (int p=0; p<numPatches; ++p)
   {
      Array<int> vdofs;
      fespace->GetPatchVDofs(p, vdofs);

      mfem::out << "AddMultNURBSPA(): vdofs (subvector) size = " << vdofs.Size() << std::endl;
      mfem::out << "AddMultNURBSPA(): size of x = " << x.Size() << std::endl;

      // TODO: reorder based on byVDIM or byNODE

      x.GetSubVector(vdofs, xp);
      mfem::out << "AddMultNURBSPA(): size of xp = " << xp.Size() << std::endl;
      yp.SetSize(vdofs.Size());
      yp = 0.0;

      AddMultPatchPA(p, xp, yp);

      y.AddElementVector(vdofs, yp);
   }
}

} // namespace mfem
