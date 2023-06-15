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

#include "../general/forall.hpp"
#include "bilininteg.hpp"
#include "gridfunc.hpp"
#include "qfunction.hpp"
#include "restriction.hpp"

using namespace std;

namespace mfem
{
static void PADGDiffusionsetup2D(const int Q1D,
                                 const int NF,
                                 const Array<double>& w,
                                 const Vector& det,
                                 const Vector& nor,
                                 const Vector& q,
                                 const double sigma,
                                 const double kappa,
                                 Vector& pa_Q,
                                 Vector& pa_hi,
                                 Vector& pa_nJi)
{
   auto detJ = Reshape(det.Read(), Q1D, NF);
   auto n = Reshape(nor.Read(), Q1D, 2, NF);

   const bool const_q = (q.Size() == 1);
   auto Q =
      const_q ? Reshape(q.Read(), 1,1) : Reshape(q.Read(), Q1D,NF);
   
   auto W = w.Read();

   auto d_q = Reshape(pa_Q.Write(), Q1D, NF);
   auto hi = Reshape(pa_hi.Write(), Q1D, NF);
   auto nJi = Reshape(pa_nJi.Write(), Q1D, 2, NF);

   for (int f = 0; f < NF; ++f)
   {
      for (int p = 0; p < Q1D; ++p)
      {
         const double Qp = const_q ? Q(0,0) : Q(p, f);
         d_q(p, f) = Qp * W[p] * detJ(p, f);
      }
   }
}

static void PADGDiffusionSetup(const int dim,
                               const int D1D,
                               const int Q1D,
                               const int NF,
                               const Array<double> &W,
                               const Vector &det,
                               const Vector &nor,
                               const Vector &q,
                               const double sigma,
                               const double kappa,
                               Vector &pa_Q,
                               Vector &pa_hi,
                               Vector &pa_nJi)
{
   if (dim == 1) { MFEM_ABORT("dim==1 not supported in PADGTraceSetup"); }
   if (dim == 2)
   {
      PADGDiffusionsetup2D(Q1D, NF, W, det, nor, q, sigma, kappa, pa_Q, pa_hi, pa_nJi);
   }
   if (dim == 3)
   {
      MFEM_ABORT("Not yet implemented");
   }
}

void DGDiffusionIntegrator::SetupPA(const FiniteElementSpace &fes,
                                    FaceType type)
{
   const MemoryType mt = (pa_mt == MemoryType::DEFAULT) ?
                         Device::GetDeviceMemoryType() : pa_mt;

   nf = fes.GetNFbyType(type);
   if (nf==0) { return; }
   // // Assumes tensor-product elements
   Mesh *mesh = fes.GetMesh();
   const FiniteElement &el =
      *fes.GetTraceElement(0, mesh->GetFaceGeometry(0));
   FaceElementTransformations &T0 =
      *fes.GetMesh()->GetFaceElementTransformations(0);
   const IntegrationRule *ir = IntRule?
                               IntRule:
                               &GetRule(el.GetGeomType(), el.GetOrder(), T0);
   // const int symmDims = 4;
   nq = ir->GetNPoints();
   dim = mesh->Dimension();
   geom = mesh->GetFaceGeometricFactors(
             *ir,
             FaceGeometricFactors::DETERMINANTS |
             FaceGeometricFactors::NORMALS, type, mt);
   maps = &el.GetDofToQuad(*ir, DofToQuad::TENSOR);
   dofs1D = maps->ndof;
   quad1D = maps->nqpt;
   // pa_data.SetSize(symmDims * nq * nf, Device::GetMemoryType());
   pa_Q.SetSize(nq * nf, Device::GetDeviceMemoryType());
   pa_hi.SetSize(nq * nf, Device::GetDeviceMemoryType());
   pa_nJi.SetSize(2 * nq * nf, Device::GetDeviceMemoryType());

   FaceQuadratureSpace fqs(*mesh, *ir, type);
   CoefficientVector q(fqs, CoefficientStorage::COMPRESSED);
   if (Q)
   {
      q.Project(*Q);
   }
   else if (MQ)
   {
      MFEM_ABORT("Not yet implemented");
      // q.Project(*MQ);
   }
   else
   {
      q.SetConstant(1.0);
   }

   PADGDiffusionSetup(dim, dofs1D, quad1D, nf, ir->GetWeights(),
                  geom->detJ, geom->normal, q,
                  sigma, kappa, pa_Q, pa_hi, pa_nJi);
}

void DGDiffusionIntegrator::AssemblePAInteriorFaces(const FiniteElementSpace&
                                                    fes)
{
   SetupPA(fes, FaceType::Interior);
}

void DGDiffusionIntegrator::AssemblePABoundaryFaces(const FiniteElementSpace&
                                                    fes)
{
   SetupPA(fes, FaceType::Boundary);
}

template<int T_D1D = 0, int T_Q1D = 0> static
void PADGDiffusionApply2D(const int NF,
                          const Array<double> &b,
                          const Array<double> &bt,
                          const double sigma,
                          const double kappa,
                          const Vector &pa_Q,
                          const Vector &pa_hi,
                          const Vector &pa_nJi,
                          const Vector &x_,
                          const Vector &dxdn_,
                          Vector &y_,
                          Vector &dydn_,
                          const int d1d = 0,
                          const int q1d = 0)
{
   const int VDIM = 1;
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   MFEM_VERIFY(D1D <= MAX_D1D, "");
   MFEM_VERIFY(Q1D <= MAX_Q1D, "");
   auto B = Reshape(b.Read(), Q1D, D1D);
   auto Bt = Reshape(bt.Read(), D1D, Q1D);
   auto Q = Reshape(pa_Q.Read(), Q1D, NF);
   auto x =    Reshape(x_.Read(),         D1D, VDIM, 2, NF);
   auto y =    Reshape(y_.ReadWrite(),    D1D, VDIM, 2, NF);
   auto dxdn = Reshape(dxdn_.Read(),      D1D, VDIM, 2, NF);
   auto dydn = Reshape(dydn_.ReadWrite(), D1D, VDIM, 2, NF);

   for (int f = 0; f < NF; ++f)
   {
      constexpr int max_D1D = T_D1D ? T_D1D : MAX_D1D;
      constexpr int max_Q1D = T_Q1D ? T_Q1D : MAX_Q1D;
      double u0[max_D1D][VDIM];
      double u1[max_D1D][VDIM];
      double du0[max_D1D][VDIM];
      double du1[max_D1D][VDIM];

      // copy edge values to u0, u1 and copy edge normals to du0, du1
      for (int d = 0; d < D1D; ++d)
      {
         for (int c = 0; c < VDIM; ++c)
         {
            u0[d][c] = x(d, c, 0, f);
            u1[d][c] = x(d, c, 1, f);
            du0[d][c] = dxdn(d, c, 0, f);
            du1[d][c] = dxdn(d, c, 1, f);
         }
      }

      // eval @ quad points
      double Bu0[max_Q1D][VDIM];
      double Bu1[max_Q1D][VDIM];
      for (int p = 0; p < Q1D; ++p)
      {
         for (int c = 0; c < VDIM; ++c)
         {
            Bu0[p][c] = 0.0;
            Bu1[p][c] = 0.0;
         }
         for (int d = 0; d < D1D; ++d)
         {
            const double b = B(p,d);
            for (int c = 0; c < VDIM; ++c)
            {
               Bu0[p][c] += b*u0[d][c];
               Bu1[p][c] += b*u1[d][c];
            }
         }
      }

      double r[max_Q1D][VDIM]; //  Q * [u] * w * det(J)
      for (int p=0; p < Q1D; ++p)
      {
         const double q = Q(p, f);
         for (int c=0; c < VDIM; ++c)
         {
            const double jump = Bu0[p][c] - Bu1[p][c];
            r[p][c] = q * jump;
         }
      }

      double Br[VDIM]; // B' * r
      for (int d = 0; d < D1D; ++d)
      {
         for (int c = 0; c < VDIM; ++c)
         {
            Br[c] = 0.0;
         }

         for (int p = 0; p < Q1D; ++p)
         {
            double bt = Bt(d, p);
            for (int c = 0; c < VDIM; ++c)
            {
               Br[c] += bt * r[p][c];
            }
         }

         for (int c = 0; c < VDIM; ++c)
         {
            y(d, c, 0, f) +=  Br[c];
            y(d, c, 1, f) += -Br[c];
         } // for c
      } // for d

   } // for f
}

static void PADGDiffusionApply(const int dim,
                               const int D1D,
                               const int Q1D,
                               const int NF,
                               const Array<double> &B,
                               const Array<double> &Bt,
                               const double sigma,
                               const double kappa,
                               const Vector &pa_Q,
                               const Vector &pa_hi,
                               const Vector &pa_nJi,
                               const Vector &x,
                               const Vector &dxdn,
                               Vector &y,
                               Vector &dydn)
{
   if (dim == 2)
   {
      switch ((D1D << 4 ) | Q1D)
      {
         case 0x22: return PADGDiffusionApply2D<2,2>(NF,B,Bt,sigma,kappa,pa_Q,pa_hi,pa_nJi,x,dxdn,y,dydn);
         case 0x33: return PADGDiffusionApply2D<3,3>(NF,B,Bt,sigma,kappa,pa_Q,pa_hi,pa_nJi,x,dxdn,y,dydn);
         case 0x44: return PADGDiffusionApply2D<4,4>(NF,B,Bt,sigma,kappa,pa_Q,pa_hi,pa_nJi,x,dxdn,y,dydn);
         case 0x55: return PADGDiffusionApply2D<5,5>(NF,B,Bt,sigma,kappa,pa_Q,pa_hi,pa_nJi,x,dxdn,y,dydn);
         case 0x66: return PADGDiffusionApply2D<6,6>(NF,B,Bt,sigma,kappa,pa_Q,pa_hi,pa_nJi,x,dxdn,y,dydn);
         case 0x77: return PADGDiffusionApply2D<7,7>(NF,B,Bt,sigma,kappa,pa_Q,pa_hi,pa_nJi,x,dxdn,y,dydn);
         case 0x88: return PADGDiffusionApply2D<8,8>(NF,B,Bt,sigma,kappa,pa_Q,pa_hi,pa_nJi,x,dxdn,y,dydn);
         case 0x99: return PADGDiffusionApply2D<9,9>(NF,B,Bt,sigma,kappa,pa_Q,pa_hi,pa_nJi,x,dxdn,y,dydn);
         default:   return PADGDiffusionApply2D(NF,B,Bt,sigma,kappa,pa_Q,pa_hi,pa_nJi,x,dxdn,y,dydn,D1D,Q1D);
      }
   }
   else if (dim == 3)
   {
      MFEM_ABORT("Not yet implemented");
   }
   MFEM_ABORT("Unknown kernel.");
}

void DGDiffusionIntegrator::AddMultPAFaceNormalDerivatives(const Vector &x,
                                                           const Vector &dxdn, Vector &y, Vector &dydn) const
{
   PADGDiffusionApply(dim, dofs1D, quad1D, nf,
                      maps->B, maps->Bt,
                      sigma, kappa, pa_Q, pa_hi, pa_nJi, x, dxdn, y, dydn);
}

const IntegrationRule &DGDiffusionIntegrator::GetRule(
   Geometry::Type geom, int order, FaceElementTransformations &T)
{
   int int_order = T.Elem1->OrderW() + 2*order;
   return IntRules.Get(geom, int_order);
}

} // namespace mfem
