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

#include "../bilininteg.hpp"
#include "../gridfunc.hpp"
#include "../qfunction.hpp"
#include "../../mesh/nurbs.hpp"
#include "../ceed/integrators/diffusion/diffusion.hpp"
#include "bilininteg_diffusion_kernels.hpp"

namespace mfem
{

void DiffusionIntegrator::AssembleDiagonalPA(Vector &diag)
{
   if (DeviceCanUseCeed())
   {
      ceedOp->GetDiagonal(diag);
   }
   else
   {
      if (pa_data.Size() == 0) { AssemblePA(*fespace); }
      const Array<real_t> &B = maps->B;
      const Array<real_t> &G = maps->G;
      const Vector &Dv = pa_data;
      DiagonalPAKernels::Run(dim, dofs1D, quad1D, ne, symmetric, B, G, Dv,
                             diag, dofs1D, quad1D);
   }
}

// PA Diffusion Apply kernel
void DiffusionIntegrator::AddMultPA(const Vector &x, Vector &y) const
{
   if (DeviceCanUseCeed())
   {
      ceedOp->AddMult(x, y);
   }
   else
   {
      const Array<real_t> &B = maps->B;
      const Array<real_t> &G = maps->G;
      const Array<real_t> &Bt = maps->Bt;
      const Array<real_t> &Gt = maps->Gt;
      const Vector &Dv = pa_data;

#ifdef MFEM_USE_OCCA
      if (DeviceCanUseOcca())
      {
         if (dim == 2)
         {
            internal::OccaPADiffusionApply2D(dofs1D,quad1D,ne,B,G,Bt,Gt,Dv,x,y);
            return;
         }
         if (dim == 3)
         {
            internal::OccaPADiffusionApply3D(dofs1D,quad1D,ne,B,G,Bt,Gt,Dv,x,y);
            return;
         }
         MFEM_ABORT("OCCA PADiffusionApply unknown kernel!");
      }
#endif // MFEM_USE_OCCA

      ApplyPAKernels::Run(dim, dofs1D, quad1D, ne, symmetric, B, G, Bt,
                          Gt, Dv, x, y, dofs1D, quad1D);
   }
}

void DiffusionIntegrator::AddMultTransposePA(const Vector &x, Vector &y) const
{
   if (symmetric)
   {
      AddMultPA(x, y);
   }
   else
   {
      MFEM_ABORT("DiffusionIntegrator::AddMultTransposePA only implemented in "
                 "the symmetric case.")
   }
}

void DiffusionIntegrator::AssemblePA(const FiniteElementSpace &fes)
{
   const MemoryType mt = (pa_mt == MemoryType::DEFAULT) ?
                         Device::GetDeviceMemoryType() : pa_mt;
   // Assuming the same element type
   fespace = &fes;
   Mesh *mesh = fes.GetMesh();
   if (mesh->GetNE() == 0) { return; }
   const FiniteElement &el = *fes.GetFE(0);
   const IntegrationRule *ir = IntRule ? IntRule : &GetRule(el, el);
   if (DeviceCanUseCeed())
   {
      delete ceedOp;
      MFEM_VERIFY(!VQ && !MQ,
                  "Only scalar coefficient supported for DiffusionIntegrator"
                  " with libCEED");
      const bool mixed = mesh->GetNumGeometries(mesh->Dimension()) > 1 ||
                         fes.IsVariableOrder();
      if (mixed)
      {
         ceedOp = new ceed::MixedPADiffusionIntegrator(*this, fes, Q);
      }
      else
      {
         ceedOp = new ceed::PADiffusionIntegrator(fes, *ir, Q);
      }
      return;
   }
   const int dims = el.GetDim();
   const int symmDims = (dims * (dims + 1)) / 2; // 1x1: 1, 2x2: 3, 3x3: 6
   const int nq = ir->GetNPoints();
   dim = mesh->Dimension();
   ne = fes.GetNE();
   geom = mesh->GetGeometricFactors(*ir, GeometricFactors::JACOBIANS, mt);
   const int sdim = mesh->SpaceDimension();
   maps = &el.GetDofToQuad(*ir, DofToQuad::TENSOR);
   dofs1D = maps->ndof;
   quad1D = maps->nqpt;

   QuadratureSpace qs(*mesh, *ir);
   CoefficientVector coeff(qs, CoefficientStorage::COMPRESSED);

   if (MQ) { coeff.ProjectTranspose(*MQ); }
   else if (VQ) { coeff.Project(*VQ); }
   else if (Q) { coeff.Project(*Q); }
   else { coeff.SetConstant(1.0); }

   const int coeff_dim = coeff.GetVDim();
   symmetric = (coeff_dim != dims*dims);
   const int pa_size = symmetric ? symmDims : dims*dims;

   pa_data.SetSize(pa_size * nq * ne, mt);
   internal::PADiffusionSetup(dim, sdim, dofs1D, quad1D, coeff_dim, ne,
                              ir->GetWeights(), geom->J, coeff, pa_data);
}

void DiffusionIntegrator::AssembleNURBSPA(const FiniteElementSpace &fes)
{
   fespace = &fes;
   Mesh *mesh = fes.GetMesh();
   dim = mesh->Dimension();
   MFEM_VERIFY(3 == dim, "Only 3D so far");

   numPatches = mesh->NURBSext->GetNP();
   for (int p=0; p<numPatches; ++p)
   {
      AssemblePatchPA(p, fes);
   }
}

void DiffusionIntegrator::AssemblePatchPA(const int patch,
                                          const FiniteElementSpace &fes)
{
   Mesh *mesh = fes.GetMesh();
   SetupPatchBasisData(mesh, patch);

   SetupPatchPA(patch, mesh);  // For full quadrature, unitWeights = false
}

// This version uses full 1D quadrature rules, taking into account the
// minimum interaction between basis functions and integration points.
void DiffusionIntegrator::AddMultPatchPA(const int patch, const Vector &x,
                                         Vector &y) const
{
   MFEM_VERIFY(3 == dim, "Only 3D so far");

   const Array<int>& Q1D = pQ1D[patch];
   const Array<int>& D1D = pD1D[patch];

   const std::vector<Array2D<real_t>>& B = pB[patch];
   const std::vector<Array2D<real_t>>& G = pG[patch];

   const IntArrayVar2D& minD = pminD[patch];
   const IntArrayVar2D& maxD = pmaxD[patch];
   const IntArrayVar2D& minQ = pminQ[patch];
   const IntArrayVar2D& maxQ = pmaxQ[patch];

   auto X = Reshape(x.Read(), D1D[0], D1D[1], D1D[2]);
   auto Y = Reshape(y.ReadWrite(), D1D[0], D1D[1], D1D[2]);

   const auto qd = Reshape(pa_data.Read(), Q1D[0]*Q1D[1]*Q1D[2],
                           (symmetric ? 6 : 9));

   // NOTE: the following is adapted from AssemblePatchMatrix_fullQuadrature
   std::vector<Array3D<real_t>> grad(dim);
   // TODO: Can an optimal order of dimensions be determined, for each patch?
   Array3D<real_t> gradXY(3, std::max(Q1D[0], D1D[0]), std::max(Q1D[1], D1D[1]));
   Array2D<real_t> gradX(3, std::max(Q1D[0], D1D[0]));

   for (int d=0; d<dim; ++d)
   {
      grad[d].SetSize(Q1D[0], Q1D[1], Q1D[2]);

      for (int qz = 0; qz < Q1D[2]; ++qz)
      {
         for (int qy = 0; qy < Q1D[1]; ++qy)
         {
            for (int qx = 0; qx < Q1D[0]; ++qx)
            {
               grad[d](qx,qy,qz) = 0.0;
            }
         }
      }
   }

   for (int dz = 0; dz < D1D[2]; ++dz)
   {
      for (int qy = 0; qy < Q1D[1]; ++qy)
      {
         for (int qx = 0; qx < Q1D[0]; ++qx)
         {
            for (int d=0; d<dim; ++d)
            {
               gradXY(d,qx,qy) = 0.0;
            }
         }
      }
      for (int dy = 0; dy < D1D[1]; ++dy)
      {
         for (int qx = 0; qx < Q1D[0]; ++qx)
         {
            gradX(0,qx) = 0.0;
            gradX(1,qx) = 0.0;
         }
         for (int dx = 0; dx < D1D[0]; ++dx)
         {
            const real_t s = X(dx,dy,dz);
            for (int qx = minD[0][dx]; qx <= maxD[0][dx]; ++qx)
            {
               gradX(0,qx) += s * B[0](qx,dx);
               gradX(1,qx) += s * G[0](qx,dx);
            }
         }
         for (int qy = minD[1][dy]; qy <= maxD[1][dy]; ++qy)
         {
            const real_t wy  = B[1](qy,dy);
            const real_t wDy = G[1](qy,dy);
            // This full range of qx values is generally necessary.
            for (int qx = 0; qx < Q1D[0]; ++qx)
            {
               const real_t wx  = gradX(0,qx);
               const real_t wDx = gradX(1,qx);
               gradXY(0,qx,qy) += wDx * wy;
               gradXY(1,qx,qy) += wx  * wDy;
               gradXY(2,qx,qy) += wx  * wy;
            }
         }
      }
      for (int qz = minD[2][dz]; qz <= maxD[2][dz]; ++qz)
      {
         const real_t wz  = B[2](qz,dz);
         const real_t wDz = G[2](qz,dz);
         for (int qy = 0; qy < Q1D[1]; ++qy)
         {
            for (int qx = 0; qx < Q1D[0]; ++qx)
            {
               grad[0](qx,qy,qz) += gradXY(0,qx,qy) * wz;
               grad[1](qx,qy,qz) += gradXY(1,qx,qy) * wz;
               grad[2](qx,qy,qz) += gradXY(2,qx,qy) * wDz;
            }
         }
      }
   }

   for (int qz = 0; qz < Q1D[2]; ++qz)
   {
      for (int qy = 0; qy < Q1D[1]; ++qy)
      {
         for (int qx = 0; qx < Q1D[0]; ++qx)
         {
            const int q = qx + ((qy + (qz * Q1D[1])) * Q1D[0]);
            const real_t O00 = qd(q,0);
            const real_t O01 = qd(q,1);
            const real_t O02 = qd(q,2);
            const real_t O10 = symmetric ? O01 : qd(q,3);
            const real_t O11 = symmetric ? qd(q,3) : qd(q,4);
            const real_t O12 = symmetric ? qd(q,4) : qd(q,5);
            const real_t O20 = symmetric ? O02 : qd(q,6);
            const real_t O21 = symmetric ? O12 : qd(q,7);
            const real_t O22 = symmetric ? qd(q,5) : qd(q,8);

            const real_t grad0 = grad[0](qx,qy,qz);
            const real_t grad1 = grad[1](qx,qy,qz);
            const real_t grad2 = grad[2](qx,qy,qz);

            grad[0](qx,qy,qz) = (O00*grad0)+(O01*grad1)+(O02*grad2);
            grad[1](qx,qy,qz) = (O10*grad0)+(O11*grad1)+(O12*grad2);
            grad[2](qx,qy,qz) = (O20*grad0)+(O21*grad1)+(O22*grad2);
         } // qx
      } // qy
   } // qz

   for (int qz = 0; qz < Q1D[2]; ++qz)
   {
      for (int dy = 0; dy < D1D[1]; ++dy)
      {
         for (int dx = 0; dx < D1D[0]; ++dx)
         {
            for (int d=0; d<3; ++d)
            {
               gradXY(d,dx,dy) = 0.0;
            }
         }
      }
      for (int qy = 0; qy < Q1D[1]; ++qy)
      {
         for (int dx = 0; dx < D1D[0]; ++dx)
         {
            for (int d=0; d<3; ++d)
            {
               gradX(d,dx) = 0.0;
            }
         }
         for (int qx = 0; qx < Q1D[0]; ++qx)
         {
            const real_t gX = grad[0](qx,qy,qz);
            const real_t gY = grad[1](qx,qy,qz);
            const real_t gZ = grad[2](qx,qy,qz);
            for (int dx = minQ[0][qx]; dx <= maxQ[0][qx]; ++dx)
            {
               const real_t wx  = B[0](qx,dx);
               const real_t wDx = G[0](qx,dx);
               gradX(0,dx) += gX * wDx;
               gradX(1,dx) += gY * wx;
               gradX(2,dx) += gZ * wx;
            }
         }
         for (int dy = minQ[1][qy]; dy <= maxQ[1][qy]; ++dy)
         {
            const real_t wy  = B[1](qy,dy);
            const real_t wDy = G[1](qy,dy);
            for (int dx = 0; dx < D1D[0]; ++dx)
            {
               gradXY(0,dx,dy) += gradX(0,dx) * wy;
               gradXY(1,dx,dy) += gradX(1,dx) * wDy;
               gradXY(2,dx,dy) += gradX(2,dx) * wy;
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
               Y(dx,dy,dz) +=
                  ((gradXY(0,dx,dy) * wz) +
                   (gradXY(1,dx,dy) * wz) +
                   (gradXY(2,dx,dy) * wDz));
            }
         }
      } // dz
   } // qz
}

void DiffusionIntegrator::AddMultNURBSPA(const Vector &x, Vector &y) const
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

} // namespace mfem
