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
   mfem::out << "START ElasticityIntegrator::AssembleDiagonalPA" << std::endl;
   q_vec->SetVDim(vdim*vdim*vdim*vdim);
   internal::ElasticityAssembleDiagonalPA(vdim, ndofs, *lambda_quad, *mu_quad,
                                          *geom, *maps, *q_vec, diag);
   mfem::out << "END ElasticityIntegrator::AssembleDiagonalPA" << std::endl;
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

   // minD/maxD : shape function/dof index |-> min/max quadrature index within support
   const IntArrayVar2D& minD = pminD[patch];
   const IntArrayVar2D& maxD = pmaxD[patch];
   // minQ/maxQ : quadrature index |-> min/max shape function/dof index that supports
   const IntArrayVar2D& minQ = pminQ[patch];
   const IntArrayVar2D& maxQ = pmaxQ[patch];

   const int NQ = Q1D[0] * Q1D[1] * Q1D[2];

   mfem::out << "AddMultPatchPA(): Size of x = " << x.Size() << std::endl;
   auto X = Reshape(x.HostRead(), vdim, D1D[0], D1D[1], D1D[2]);
   auto Y = Reshape(y.HostReadWrite(), vdim, D1D[0], D1D[1], D1D[2]);

   // First 9 entries are J^{-T}, last entry is W*detJ
   const auto qd = Reshape(pa_data.HostRead(), NQ, 12);

   // grad(c,d,qx,qy,qz)
   // derivative of u_c w.r.t. d evaluated at (qx,qy,qz)
   Vector gradv(vdim*vdim*NQ);
   auto grad = Reshape(gradv.HostReadWrite(), vdim, vdim, Q1D[0], Q1D[1], Q1D[2]);
   // TODO: Make this method explicitly 3D
   Vector Sv(vdim*vdim*NQ);
   auto S = Reshape(Sv.HostReadWrite(), vdim, vdim, Q1D[0], Q1D[1], Q1D[2]);

   // Accumulators
   // Data is stored in Vectors but shaped as a DeviceTensor for easy access
   // const int max_pts_x = std::max(Q1D[0], D1D[0]);
   // const int max_pts_y = std::max(Q1D[1], D1D[1]);
   Vector gradXYv(vdim*vdim*Q1D[0]*Q1D[1]);
   Vector gradXv(vdim*2*Q1D[0]);
   auto gradXY = Reshape(gradXYv.HostReadWrite(), vdim, vdim, Q1D[0], Q1D[1]);
   auto gradX = Reshape(gradXv.HostReadWrite(), vdim, 2, Q1D[0]);

   mfem::out << "AddMultPatchPA() " << patch << " - finished 1) init grad u" << std::endl;
   mfem::out << "Q1D[0] = " << Q1D[0] << ", Q1D[1] = " << Q1D[1] << ", Q1D[2] = " << Q1D[2] << std::endl;
   mfem::out << "grad(0,1,1,1,1) = " << grad(0,1,1,1,1) << std::endl;

   /*
   2) Compute grad_uhat (reference space) interpolated at quadrature points

   B_{nq} U_n = \sum_n u_n \tilde{\nabla} \tilde{\phi}_n (\tilde{x}_q)

   Variables:
   - c  = component
   - d  = derivative
   - dx, dy, dz = degrees of freedom (DOF) indices
   - qx, qy, qz = quadrature indices

   Tensor product shape functions:
   phi[dx,dy,dz](qx,qy,qz) = phix[dx](qx) * phiy[dy](qy) * phiz[dz](qz)

   Gradient of shape functions:
   grad_phi[dx,dy,dz](qx,qy,qz) = [
      dphix[dx](qx) * phiy[dy](qy)  * phiz[dz](qz),
      phix[dx](qx)  * dphiy[dy](qy) * phiz[dz](qz),
      phix[dx](qx)  * phiy[dy](qy)  * dphiz[dz](qz)
   ]

   Computation of grad_uhat[c,0] (sum factorization):
   grad_uhat[c,0](qx,qy,qz)
      = \sum_{dx,dy,dz} U[c][dx,dy,dz] * grad_phi[dx,dy,dz][0](qx,qy,qz)
      = \sum_{dx,dy,dz} U[c][dx,dy,dz] * dphix[dx](qx) * phiy[dy](qy) * phiz[dz](qz)
      = \sum_{dz} phiz[dz](qz)
         \sum_{dy} phiy[dy](qy)
            \sum_{dx} U[c][dx,dy,dz] * dphix[dx](qx)

   Because a nurbs patch is "sparse" compared to an element in terms of shape
   function support - we can further optimize the computation by restricting
   interpolation to only the qudrature points supported in each dimension.
   */
   for (int dz = 0; dz < D1D[2]; ++dz)
   {
      gradXYv = 0.0;
      for (int dy = 0; dy < D1D[1]; ++dy)
      {
         gradXv = 0.0;
         for (int dx = 0; dx < D1D[0]; ++dx)
         {
            for (int c = 0; c < vdim; ++c)
            {
               const real_t U = X(c,dx,dy,dz);
               for (int qx = minD[0][dx]; qx <= maxD[0][dx]; ++qx)
               {
                  gradX(c,0,qx) += U * B[0](qx,dx);
                  gradX(c,1,qx) += U * G[0](qx,dx);
               }
            }
         }
         for (int qy = minD[1][dy]; qy <= maxD[1][dy]; ++qy)
         {
            const real_t wy  = B[1](qy,dy);
            const real_t wDy = G[1](qy,dy);
            for (int c = 0; c < vdim; ++c)
            {
               // This full range of qx values is generally necessary.
               for (int qx = 0; qx < Q1D[0]; ++qx)
               {
                  const real_t wx  = gradX(c,0,qx);
                  const real_t wDx = gradX(c,1,qx);
                  gradXY(c,0,qx,qy) += wDx * wy;
                  gradXY(c,1,qx,qy) += wx  * wDy;
                  gradXY(c,2,qx,qy) += wx  * wy;
               }
            }
         }
      }
      for (int qz = minD[2][dz]; qz <= maxD[2][dz]; ++qz)
      {
         const real_t wz  = B[2](qz,dz);
         const real_t wDz = G[2](qz,dz);
         for (int c = 0; c < vdim; ++c)
         {
            for (int qy = 0; qy < Q1D[1]; ++qy)
            {
               for (int qx = 0; qx < Q1D[0]; ++qx)
               {
                  grad(c,0,qx,qy,qz) += gradXY(c,0,qx,qy) * wz;
                  grad(c,1,qx,qy,qz) += gradXY(c,1,qx,qy) * wz;
                  grad(c,2,qx,qy,qz) += gradXY(c,2,qx,qy) * wDz;
               }
            }
         }
      }
   }

   mfem::out << "AddMultPatchPA() " << patch << " - finished 2) compute grad u" << std::endl;
   mfem::out << "grad(0,1,1,1,1) = " << grad(0,1,1,1,1) << std::endl;

   // 3) Apply the "D" operator at each quadrature point: D( grad_uhat )
   for (int qz = 0; qz < Q1D[2]; ++qz)
   {
      for (int qy = 0; qy < Q1D[1]; ++qy)
      {
         for (int qx = 0; qx < Q1D[0]; ++qx)
         {
            const int q = qx + ((qy + (qz * Q1D[1])) * Q1D[0]);
            const real_t Jinvt00 = qd(q,0);
            const real_t Jinvt01 = qd(q,1);
            const real_t Jinvt02 = qd(q,2);
            const real_t Jinvt10 = qd(q,3);
            const real_t Jinvt11 = qd(q,4);
            const real_t Jinvt12 = qd(q,5);
            const real_t Jinvt20 = qd(q,6);
            const real_t Jinvt21 = qd(q,7);
            const real_t Jinvt22 = qd(q,8);
            const real_t wdetj   = qd(q,9);
            const real_t lambda  = qd(q,10);
            const real_t mu      = qd(q,11);

            // grad_u = J^{-T} * grad_uhat
            for (int c = 0; c < vdim; ++c)
            {
               const real_t grad0 = grad(c,0,qx,qy,qz);
               const real_t grad1 = grad(c,1,qx,qy,qz);
               const real_t grad2 = grad(c,2,qx,qy,qz);
               grad(c,0,qx,qy,qz) = (Jinvt00*grad0)+(Jinvt01*grad1)+(Jinvt02*grad2);
               grad(c,1,qx,qy,qz) = (Jinvt10*grad0)+(Jinvt11*grad1)+(Jinvt12*grad2);
               grad(c,2,qx,qy,qz) = (Jinvt20*grad0)+(Jinvt21*grad1)+(Jinvt22*grad2);
               // grad(c,0,qx,qy,qz) = (Jinvt00*grad0)+(Jinvt10*grad1)+(Jinvt20*grad2);
               // grad(c,1,qx,qy,qz) = (Jinvt01*grad0)+(Jinvt11*grad1)+(Jinvt21*grad2);
               // grad(c,2,qx,qy,qz) = (Jinvt02*grad0)+(Jinvt12*grad1)+(Jinvt22*grad2);
            }

            // Compute stress tensor
            // [s00, s11, s22, s12, s02, s01]
            real_t div = grad(0,0,qx,qy,qz) + grad(1,1,qx,qy,qz) + grad(2,2,qx,qy,qz);
            real_t sigma00 = (lambda*div + 2.0*mu*grad(0,0,qx,qy,qz)) * wdetj;
            real_t sigma11 = (lambda*div + 2.0*mu*grad(1,1,qx,qy,qz)) * wdetj;
            real_t sigma22 = (lambda*div + 2.0*mu*grad(2,2,qx,qy,qz)) * wdetj;
            real_t sigma12 = mu * (grad(1,2,qx,qy,qz) + grad(2,1,qx,qy,qz)) * wdetj;
            real_t sigma02 = mu * (grad(0,2,qx,qy,qz) + grad(2,0,qx,qy,qz)) * wdetj;
            real_t sigma01 = mu * (grad(0,1,qx,qy,qz) + grad(1,0,qx,qy,qz)) * wdetj;

            // S = J^{-1} * stress
            /*
               Jinvt00, Jinvt10, Jinvt20,
               Jinvt01, Jinvt11, Jinvt21,
               Jinvt02, Jinvt12, Jinvt22,

               s00, s01, s02,
               s01, s11, s12,
               s02, s12, s22,
            */
            // (J^{-1} term comes from test function)
            S(0,0,qx,qy,qz) = Jinvt00*sigma00 + Jinvt10*sigma01 + Jinvt20*sigma02;
            S(0,1,qx,qy,qz) = Jinvt00*sigma01 + Jinvt10*sigma11 + Jinvt20*sigma12;
            S(0,2,qx,qy,qz) = Jinvt00*sigma02 + Jinvt10*sigma12 + Jinvt20*sigma22;
            S(1,0,qx,qy,qz) = Jinvt01*sigma00 + Jinvt11*sigma01 + Jinvt21*sigma02;
            S(1,1,qx,qy,qz) = Jinvt01*sigma01 + Jinvt11*sigma11 + Jinvt21*sigma12;
            S(1,2,qx,qy,qz) = Jinvt01*sigma02 + Jinvt11*sigma12 + Jinvt21*sigma22;
            S(2,0,qx,qy,qz) = Jinvt02*sigma00 + Jinvt12*sigma01 + Jinvt22*sigma02;
            S(2,1,qx,qy,qz) = Jinvt02*sigma01 + Jinvt12*sigma11 + Jinvt22*sigma12;
            S(2,2,qx,qy,qz) = Jinvt02*sigma02 + Jinvt12*sigma12 + Jinvt22*sigma22;


         } // qx
      } // qy
   } // qz

   mfem::out << "AddMultPatchPA() " << patch << " - finished 3) apply D" << std::endl;
   mfem::out << "S(0,0,1,1,1) = " << S(0,0,1,1,1) << std::endl;

   /*
   4) Contraction with grad_v (quads -> dofs)

   S[ij] = [
      s00, s01, s02,
      s10, s11, s12,
      s20, s21, s22,
   ]
   grad_v[ij] = e[i] * grad_phi[j]
             = e[i] * [ dX*Y*Z, X*dY*Z, X*Y*dZ ]

   Y[i] = S[ij] * grad_phi[j] = [
      s00*dX*Y*Z + s01*X*dY*Z + s02*X*Y*dZ,
      s10*dX*Y*Z + s11*X*dY*Z + s12*X*Y*dZ,
      s20*dX*Y*Z + s21*X*dY*Z + s22*X*Y*dZ,
   ]

   sX = [
      s00*dX, s01*X, s02*X,
      s10*dX, s11*X, s12*X,
      s20*dX, s21*X, s22*X,
   ]

   sXY = [
      (s00*dX) * Y + (s01*X) * dY, (s02*X) * Y,
      (s10*dX) * Y + (s11*X) * dY, (s12*X) * Y,
      (s20*dX) * Y + (s21*X) * dY, (s22*X) * Y,
   ]

   Y[i] = [
      ((s00*dX) * Y + (s01*X) * dY) * Z + ((s02*X) * Y) * dZ,
      ((s10*dX) * Y + (s11*X) * dY) * Z + ((s12*X) * Y) * dZ,
      ((s20*dX) * Y + (s21*X) * dY) * Z + ((s22*X) * Y) * dZ,
   ]
   */

   Vector sXYv(vdim*2*D1D[0]*D1D[1]);
   Vector sXv(vdim*vdim*D1D[0]);
   auto sXY = Reshape(sXYv.HostReadWrite(), vdim, 2, D1D[0], D1D[1]);
   auto sX = Reshape(sXv.HostReadWrite(), vdim, vdim, D1D[0]);

   for (int qz = 0; qz < Q1D[2]; ++qz)
   {
      sXYv = 0.0;
      for (int qy = 0; qy < Q1D[1]; ++qy)
      {
         sXv = 0.0;
         for (int qx = 0; qx < Q1D[0]; ++qx)
         {
            const real_t s[3][3] = {
               { S(0,0,qx,qy,qz), S(0,1,qx,qy,qz), S(0,2,qx,qy,qz) },
               { S(1,0,qx,qy,qz), S(1,1,qx,qy,qz), S(1,2,qx,qy,qz) },
               { S(2,0,qx,qy,qz), S(2,1,qx,qy,qz), S(2,2,qx,qy,qz) }
            };
            for (int dx = minQ[0][qx]; dx <= maxQ[0][qx]; ++dx)
            {
               const real_t wx  = B[0](qx,dx);
               const real_t wDx = G[0](qx,dx);

               /*
               sX = [
                  s00*dX, s01*X, s02*X,
                  s01*dX, s11*X, s12*X,
                  s02*dX, s12*X, s22*X,
               ]

               (could optimize this slightly since sX[1][2] == sX[2][1])
               */
               for (int c = 0; c < vdim; ++c)
               {
                  sX(c,0,dx) = s[c][0] * wDx;
                  sX(c,1,dx) = s[c][1] * wx;
                  sX(c,2,dx) = s[c][2] * wx;
               }
            }
         }
         for (int dy = minQ[1][qy]; dy <= maxQ[1][qy]; ++dy)
         {
            /*
            sXY = [
               (s00*dX) * Y + (s01*X) * dY, (s02*X) * Y,
               (s01*dX) * Y + (s11*X) * dY, (s12*X) * Y,
               (s02*dX) * Y + (s12*X) * dY, (s22*X) * Y,
            ]
            */
            const real_t wy  = B[1](qy,dy);
            const real_t wDy = G[1](qy,dy);
            for (int dx = 0; dx < D1D[0]; ++dx)
            {
               for (int c = 0; c < vdim; ++c)
               {
                  sXY(c,0,dx,dy) += sX(c,0,dx) * wy + sX(c,1,dx) * wDy;
                  sXY(c,1,dx,dy) += sX(c,2,dx) * wy;
               }
            }
         }
      }
      for (int dz = minQ[2][qz]; dz <= maxQ[2][qz]; ++dz)
      {
         const real_t wz  = B[2](qz,dz);
         const real_t wDz = G[2](qz,dz);
         for (int dy = 0; dy < D1D[1]; ++dy)
         {
            for (int dx = 0; dx < D1D[0]; ++dx)
            {
               for (int c = 0; c < vdim; ++c)
               {
                  Y(c,dx,dy,dz) +=
                     (sXY(c,0,dx,dy) * wz +
                      sXY(c,1,dx,dy) * wDz);
               }
            }
         }
      } // dz
   } // qz

   mfem::out << "AddMultPatchPA() " << patch << " - finished 4) contraction" << std::endl;
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
