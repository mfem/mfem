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
#include "fe/face_map_utils.hpp"

using namespace std;

namespace mfem
{
static void PADGDiffusionsetup2D(const int Q1D,
                                 const int NE,
                                 const int NF,
                                 const Array<double>& w,
                                 const Vector& jacE,
                                 const Vector& detE,
                                 const Vector& detF,
                                 const Vector& nor,
                                 const Vector& q,
                                 const double sigma,
                                 const double kappa,
                                 Vector& pa_data,
                                 const Array<int>& iwork_)
{
   auto J = Reshape(jacE.Read(), Q1D, Q1D, 2, 2, NE);
   auto detJe = Reshape(detE.Read(), Q1D, Q1D, NE);
   auto detJf = Reshape(detF.Read(), Q1D, NF);
   auto n = Reshape(nor.Read(), Q1D, 2, NF);

   const bool const_q = (q.Size() == 1);
   auto Q =
      const_q ? Reshape(q.Read(), 1,1) : Reshape(q.Read(), Q1D,NF);

   auto W = w.Read();

   auto pa = Reshape(pa_data.Write(), 6, Q1D, NF); // (q, 1/h, J00, J01, J10, J11)
   auto iwork = Reshape(iwork_.Read(), 6,
                        NF); // (flip0, flip1, e0, e1, fid0, fid1)

   mfem::forall(NF, [=] MFEM_HOST_DEVICE (int f) -> void
   {
      for (int p = 0; p < Q1D; ++p)
      {
         const int flip[] = {iwork(0, f), iwork(1, f)};
         const int el[] = {iwork(2, f), iwork(3, f)};
         const int fid[] = {iwork(4, f), iwork(5, f)};

         const double Qp = const_q ? Q(0,0) : Q(p, f);
         // d_q(p, f) = kappa * Qp * W[p] * detJf(p, f);
         pa(0, p, f) = kappa * Qp * W[p] * detJf(p, f);

         const bool interior = el[1] >= 0;
         const double factor = interior ? 0.5 : 1.0;
         // hi(p, f) = 0.0;
         double hi = 0.0;

         const int nsides = (interior) ? 2 : 1;

         for (int side = 0; side < nsides; ++side)
         {
            int i, j;
            internal::EdgeQuad2Lex2D(p, Q1D, fid[0], fid[1], side, i, j);

            const double nJi0 = n(p,0,f)*J(i,j, 1,1, el[side])
                                - n(p,1,f)*J(i,j,0,1,el[side]);
            const double nJi1 = -n(p,0,f)*J(i,j,1,0, el[side])
                                + n(p,1,f)*J(i,j,0,0,el[side]);

            const double dJe = detJe(i,j,el[side]);
            const double dJf = detJf(p, f);

            // nJi(flip[side], p, side, f) = factor * nJi0 / dJe * Qp * W[p] * dJf;
            // nJi(1-flip[side], p, side, f) = factor * nJi1 / dJe * Qp * W[p] * dJf;
            pa(2 + 2*side + flip[side], p, f) = factor * nJi0 / dJe * Qp * W[p] * dJf;
            pa(2 + 2*side + 1 - flip[side], p, f) = factor * nJi1 / dJe * Qp * W[p] * dJf;

            // hi(p, f) += factor * dJf / dJe;
            hi += factor * dJf / dJe;
         }

         pa(1, p, f) = hi;
      }
   });
}

static void PADGDiffusionSetup(const int dim,
                               const int D1D,
                               const int Q1D,
                               const int NE,
                               const int NF,
                               const Array<double> &W,
                               const Vector& jacE,
                               const Vector& detE,
                               const Vector& detF,
                               const Vector &nor,
                               const Vector &q,
                               const double sigma,
                               const double kappa,
                               Vector &pa_data,
                               Array<int>& iwork)
{
   if (dim == 1) { MFEM_ABORT("dim==1 not supported in PADGTraceSetup"); }
   if (dim == 2)
   {
      PADGDiffusionsetup2D(Q1D, NE, NF, W, jacE, detE, detF, nor, q, sigma, kappa, pa_data, iwork);
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

   int ne = fes.GetNE();
   nf = fes.GetNFbyType(type);
   if (nf==0) { return; }
   // // Assumes tensor-product elements
   Mesh *mesh = fes.GetMesh();
   const FiniteElement &el =
      *fes.GetTraceElement(0, mesh->GetFaceGeometry(0));
   FaceElementTransformations &T0 =
      *fes.GetMesh()->GetFaceElementTransformations(0);
   const IntegrationRule *ir = IntRule ?
                               IntRule :
                               &GetRule(el.GetOrder(), T0);
   // const int symmDims = 4;
   dim = mesh->Dimension();
   nq = ir->Size();
   const int nq1d = pow(double(ir->Size()), 1.0/(dim - 1));
   // nq = ir->GetNPoints();

   auto vol_ir = irs.Get(mesh->GetElementGeometry(0), 2*nq1d - 3);
   elgeom = mesh->GetGeometricFactors(
               vol_ir,
               GeometricFactors::JACOBIANS | GeometricFactors::DETERMINANTS,
               mt);

   fgeom = mesh->GetFaceGeometricFactors(
              *ir,
              FaceGeometricFactors::DETERMINANTS |
              FaceGeometricFactors::NORMALS, type, mt);
   maps = &el.GetDofToQuad(*ir, DofToQuad::TENSOR);
   dofs1D = maps->ndof;
   quad1D = maps->nqpt;

   pa_data.SetSize(6 * nq * nf, Device::GetMemoryType());

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

   int fidx = 0;
   Array<int> iwork_(6 * nf);
   auto iwork = Reshape(iwork_.HostWrite(), 6,
                        nf); // (flip0, flip1, e0, e1, fid0, fid1)
   for (int f = 0; f < mesh->GetNumFaces(); ++f)
   {
      auto info = mesh->GetFaceInformation(f);

      if (info.IsOfFaceType(type))
      {
         const int face_id_1 = info.element[0].local_face_id;
         iwork(0, fidx) = (face_id_1 == 0 || face_id_1 == 2) ? 1 : 0;
         iwork(2, fidx) = info.element[0].index;
         iwork(4, fidx) = face_id_1;

         if (info.IsInterior())
         {
            const int face_id_2 = info.element[1].local_face_id;
            iwork(1, fidx) = (face_id_2 == 0 || face_id_2 == 2) ? 1 : 0;
            iwork(3, fidx) = info.element[1].index;
            iwork(5, fidx) = face_id_2;
         }
         else
         {
            iwork(1, fidx) = -1;
            iwork(3, fidx) = -1;
            iwork(5, fidx) = -1;
         }

         fidx++;
      }
   }

   PADGDiffusionSetup(dim, dofs1D, quad1D, ne, nf,
                      ir->GetWeights(), elgeom->J, elgeom->detJ, fgeom->detJ,
                      fgeom->normal, q, sigma, kappa, pa_data, iwork_);
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
                          const Array<double>& g,
                          const Array<double>& gt,
                          const double sigma,
                          const double kappa,
                          const Vector &pa_data,
                          const Vector &x_,
                          const Vector &dxdn_,
                          Vector &y_,
                          Vector &dydn_,
                          const int d1d = 0,
                          const int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   MFEM_VERIFY(D1D <= MAX_D1D, "");
   MFEM_VERIFY(Q1D <= MAX_Q1D, "");

   auto B = Reshape(b.Read(), Q1D, D1D);
   auto Bt = Reshape(bt.Read(), D1D, Q1D);
   auto G = Reshape(g.Read(), Q1D, D1D);
   auto Gt = Reshape(gt.Read(), D1D, Q1D);

   auto pa = Reshape(pa_data.Read(), 6, Q1D, NF);

   auto x =    Reshape(x_.Read(),         D1D, 2, NF);
   auto y =    Reshape(y_.ReadWrite(),    D1D, 2, NF);
   auto dxdn = Reshape(dxdn_.Read(),      D1D, 2, NF);
   auto dydn = Reshape(dydn_.ReadWrite(), D1D, 2, NF);

   mfem::forall(NF, [=] MFEM_HOST_DEVICE (int f) -> void
   {
      constexpr int max_D1D = T_D1D ? T_D1D : MAX_D1D;
      constexpr int max_Q1D = T_Q1D ? T_Q1D : MAX_Q1D;
      
      double u0[max_D1D];
      double u1[max_D1D];
      double du0[max_D1D];
      double du1[max_D1D];

      double Bu0[max_Q1D];
      double Bu1[max_Q1D];
      double Bdu0[max_Q1D];
      double Bdu1[max_Q1D];

      double r[max_Q1D];
      double w[max_Q1D];

      // copy edge values to u0, u1 and copy edge normals to du0, du1
      for (int d = 0; d < D1D; ++d)
      {
         u0[d] = x(d, 0, f);
         u1[d] = x(d, 1, f);
         du0[d] = dxdn(d, 0, f);
         du1[d] = dxdn(d, 1, f);
      }

      // eval @ quad points
      for (int p = 0; p < Q1D; ++p)
      {
         const double Je0[] = {pa(2, p, f), pa(3, p, f)};
         const double Je1[] = {pa(4, p, f), pa(5, p, f)};

         Bu0[p] = 0.0;
         Bu1[p] = 0.0;
         Bdu0[p] = 0.0;
         Bdu1[p] = 0.0;

         for (int d = 0; d < D1D; ++d)
         {
            const double b = B(p,d);
            const double g = G(p,d);
      
            Bu0[p] += b*u0[d];
            Bu1[p] += b*u1[d];

            Bdu0[p] += Je0[0] * b * du0[d] + Je0[1] * g * u0[d];
            Bdu1[p] += Je1[0] * b * du1[d] + Je1[1] * g * u1[d];
         }
      }

      // term - < {Q du/dn}, [v] > +  kappa * < {Q/h} [u], [v] >:
      for (int p = 0; p < Q1D; ++p)
      {
         const double q = pa(0, p, f);
         const double hi = pa(1, p, f);
         const double jump = Bu0[p] - Bu1[p];
         const double avg = Bdu0[p] + Bdu1[p]; // = {Q du/dn} * w * det(J)
         r[p] = -avg + hi * q * jump;
      }

      for (int d = 0; d < D1D; ++d)
      {
         double Br = 0.0;

         for (int p = 0; p < Q1D; ++p)
         {
            Br += Bt(d, p) * r[p];
         }

         u0[d] =  Br; // overwrite u0, u1
         u1[d] = -Br;
      } // for d

      // term sigma * < [u], {Q dv/dn} >
      MFEM_UNROLL(2)
      for (int side = 0; side < 2; ++side)
      {
         double * const du = (side == 0) ? du0 : du1;
         double * const u = (side == 0) ? u0 : u1;

         for (int p = 0; p < Q1D; ++p)
         {
            const double Je[] = {pa(2 + 2*side, p, f), pa(2 + 2*side + 1, p, f)};
            const double jump = Bu0[p] - Bu1[p];
            r[p] = Je[0] * jump;
            w[p] = Je[1] * jump;
         }

         for (int d = 0; d < D1D; ++d)
         {
            double Br = 0.0;
            double Gw = 0.0;

            for (int p = 0; p < Q1D; ++p)
            {
               Br += Bt(d, p) * r[p];
               Gw += Gt(d, p) * w[p];
            }

            du[d] = sigma * Br; // overwrite du0, du1
            u[d] += sigma * Gw;  
         }
      }

      for (int d = 0; d < D1D; ++d)
      {
         y(d, 0, f) += u0[d];
         y(d, 1, f) += u1[d];
         dydn(d, 0, f) += du0[d];
         dydn(d, 1, f) += du1[d];
      }
   }); // mfem::forall
}

static void PADGDiffusionApply(const int dim,
                               const int D1D,
                               const int Q1D,
                               const int NF,
                               const Array<double> &B,
                               const Array<double> &Bt,
                               const Array<double> &G,
                               const Array<double> &Gt,
                               const double sigma,
                               const double kappa,
                               const Vector &pa_data,
                               const Vector &x,
                               const Vector &dxdn,
                               Vector &y,
                               Vector &dydn)
{
   if (dim == 2)
   {
      switch ((D1D << 4 ) | Q1D)
      {
         case 0x23: return PADGDiffusionApply2D<2,3>(NF,B,Bt,G,Gt,sigma,kappa,pa_data,x,dxdn,y,dydn);
         case 0x34: return PADGDiffusionApply2D<3,4>(NF,B,Bt,G,Gt,sigma,kappa,pa_data,x,dxdn,y,dydn);
         case 0x45: return PADGDiffusionApply2D<4,5>(NF,B,Bt,G,Gt,sigma,kappa,pa_data,x,dxdn,y,dydn);
         case 0x56: return PADGDiffusionApply2D<5,6>(NF,B,Bt,G,Gt,sigma,kappa,pa_data,x,dxdn,y,dydn);
         case 0x67: return PADGDiffusionApply2D<6,7>(NF,B,Bt,G,Gt,sigma,kappa,pa_data,x,dxdn,y,dydn);
         case 0x78: return PADGDiffusionApply2D<7,8>(NF,B,Bt,G,Gt,sigma,kappa,pa_data,x,dxdn,y,dydn);
         case 0x89: return PADGDiffusionApply2D<8,9>(NF,B,Bt,G,Gt,sigma,kappa,pa_data,x,dxdn,y,dydn);
         case 0x9A: return PADGDiffusionApply2D<9,10>(NF,B,Bt,G,Gt,sigma,kappa,pa_data,x,dxdn,y,dydn);
         default:   return PADGDiffusionApply2D(NF,B,Bt,G,Gt,sigma,kappa,pa_data,x,dxdn,y,dydn,D1D,Q1D);
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
                      maps->B, maps->Bt, maps->G, maps->Gt,
                      sigma, kappa, pa_data, x, dxdn, y, dydn);
}

const IntegrationRule &DGDiffusionIntegrator::GetRule(
   int order, FaceElementTransformations &T)
{
   int int_order = T.Elem1->OrderW() + 2*order;
   return irs.Get(T.GetGeometryType(), int_order);
}

} // namespace mfem
