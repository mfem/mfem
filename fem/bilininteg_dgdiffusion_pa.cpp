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
static int ToLexOrdering2D(const int face_id, const int size1d, const int i)
{
   if (face_id==2 || face_id==3)
   {
      return size1d-1-i;
   }
   else
   {
      return i;
   }
}

static int PermuteFace2D(const int face_id1, const int face_id2,
                         const int orientation,
                         const int size1d, const int index)
{
   int new_index;
   // Convert from lex ordering
   if (face_id1==2 || face_id1==3)
   {
      new_index = size1d-1-index;
   }
   else
   {
      new_index = index;
   }
   // Permute based on face orientations
   if (orientation==1)
   {
      new_index = size1d-1-new_index;
   }
   return ToLexOrdering2D(face_id2, size1d, new_index);
}

static std::pair<int, int> EdgeQuad2Lex(const int qi, const int nq,
                                        const int face_id0, const int face_id1, const int side)
{
   const int face_id = (side == 0) ? face_id0 : face_id1;
   const int edge_idx = (side == 0) ? qi : PermuteFace2D(face_id0, face_id1, side,
                                                         nq, qi);
   int i, j;
   if (face_id == 0 || face_id == 2)
   {
      i = edge_idx;
      j = (face_id == 0) ? 0 : (nq-1);
   }
   else
   {
      j = edge_idx;
      i = (face_id == 3) ? 0 : (nq-1);
   }

   return std::make_pair(i, j);
}

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
                                 const double lambda,
                                 Vector& pa_Q,
                                 Vector& pa_hi,
                                 Vector& pa_nJi,
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

   auto d_q = Reshape(pa_Q.Write(), Q1D, NF);
   auto hi = Reshape(pa_hi.Write(), Q1D, NF);
   auto nJi = Reshape(pa_nJi.Write(), 2, Q1D, 2, NF);
   auto iwork = Reshape(iwork_.Read(), 6,
                        NF); // (flip0, flip1, e0, e1, fid0, fid1)

   for (int f = 0; f < NF; ++f)
   {
      for (int p = 0; p < Q1D; ++p)
      {
         const int flip[] = {iwork(0, f), iwork(1, f)};
         const int el[] = {iwork(2, f), iwork(3, f)};
         const int fid[] = {iwork(4, f), iwork(5, f)};

         const double Qp = const_q ? Q(0,0) : Q(p, f);
         d_q(p, f) = kappa * Qp * W[p] * detJf(p, f);

         for (int side = 0; side < 2; ++side)
         {
            if (el[side] < 0)
            {
               continue;
            }

            auto [i, j] = EdgeQuad2Lex(p, Q1D, fid[0], fid[1], side);

            const double nJi0 = n(p, 0, f) * J(i,j,  1,1,  el[side]) - n(p,1, f) * J(i,j,
                                                                                     0,1,  el[side]);
            const double nJi1 = -n(p, 0, f) * J(i,j,  1,0,  el[side]) + n(p,1, f) * J(i,j,
                                                                                      0,0,  el[side]);
            const double dJe = detJe(i,j,el[side]);

            nJi(flip[side], p, side, f) = lambda * nJi0 / dJe * Qp * W[p] * detJf(p, f);
            nJi(1-flip[side], p, side, f) = lambda * nJi1 / dJe * Qp * W[p] * detJf(p, f);
         }
      }
   }
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
                               const double lambda,
                               Vector &pa_Q,
                               Vector &pa_hi,
                               Vector &pa_nJi,
                               Array<int>& iwork)
{
   if (dim == 1) { MFEM_ABORT("dim==1 not supported in PADGTraceSetup"); }
   if (dim == 2)
   {
      PADGDiffusionsetup2D(Q1D, NE, NF, W, jacE, detE, detF, nor, q, sigma, kappa,
                           lambda, pa_Q, pa_hi, pa_nJi, iwork);
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

   // pa_data.SetSize(symmDims * nq * nf, Device::GetMemoryType());
   pa_Q.SetSize(nq * nf, Device::GetDeviceMemoryType());
   pa_hi.SetSize(nq * nf, Device::GetDeviceMemoryType());
   pa_nJi.SetSize(2 * 2 * nq * nf, Device::GetDeviceMemoryType());

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
   auto iwork = Reshape(iwork_.Write(), 6,
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
                      fgeom->normal, q, sigma, kappa, lambda, pa_Q, pa_hi, pa_nJi, iwork_);
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
   auto G = Reshape(g.Read(), Q1D, D1D);

   auto Q = Reshape(pa_Q.Read(), Q1D, NF);
   auto J = Reshape(pa_nJi.Read(), 2, Q1D, 2, NF);

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
      double Bdu0[max_Q1D][VDIM];
      double Bdu1[max_Q1D][VDIM];
      for (int p = 0; p < Q1D; ++p)
      {
         for (int c = 0; c < VDIM; ++c)
         {
            Bu0[p][c] = 0.0;
            Bu1[p][c] = 0.0;
            Bdu0[p][c] = 0.0;
            Bdu1[p][c] = 0.0;
         }
         for (int d = 0; d < D1D; ++d)
         {
            const double b = B(p,d);
            for (int c = 0; c < VDIM; ++c)
            {
               Bu0[p][c] += b*u0[d][c];
               Bu1[p][c] += b*u1[d][c];
               Bdu0[p][c] += b*du0[d][c];
               Bdu1[p][c] += b*du1[d][c];
            }
         }
      }

      // term - < {Q du/dn}, [v] >

      // --> compute reference tangential derivative from face values
      double ut0[max_Q1D][VDIM];
      double ut1[max_Q1D][VDIM];
      for (int p = 0; p < Q1D; ++p)
      {
         for (int c = 0; c < VDIM; ++c)
         {
            ut0[p][c] = 0.0;
            ut1[p][c] = 0.0;
         }

         for (int d = 0; d < D1D; ++d)
         {
            for (int c = 0; c < VDIM; ++c)
            {
               ut0[p][c] += G(p, d) * u0[d][c];
               ut1[p][c] += G(p, d) * u1[d][c];
            }
         }
      }

      // --> compute physical normal derivatives from reference gradient
      double v[max_Q1D][VDIM]; // {Q du/dn} * w * det(J)
      for (int p = 0; p < Q1D; ++p)
      {
         const double Je0[] = {J(0, p, 0, f), J(1, p, 0, f)};
         const double Je1[] = {J(0, p, 1, f), J(1, p, 1, f)};

         for (int c = 0; c < VDIM; ++c)
         {
            const double dudn0 = Je0[0] * Bdu0[p][c] + Je0[1] * ut0[p][c];
            const double dudn1 = Je1[0] * Bdu1[p][c] + Je1[1] * ut1[p][c];

            std::cout << std::setw(16) << std::setprecision(4) << std::left << dudn0
                      << std::setw(16) << std::setprecision(4) << std::left << dudn1 << '\n';

            v[p][c] = 0.5 * (dudn0 + dudn1);
         }
      }

      double Bv[max_D1D][VDIM]; // B' * v
      for (int d = 0; d < D1D; ++d)
      {
         for (int c = 0; c < VDIM; ++c)
         {
            Bv[d][c] = 0.0;
         }

         for (int p = 0; p < Q1D; ++p)
         {
            double bt = Bt(d, p);
            for (int c = 0; c < VDIM; ++c)
            {
               Bv[d][c] += bt * v[p][c];
            }
         }
      }

      for (int d = 0; d < D1D; ++d)
      {
         for (int c = 0; c < VDIM; ++c)
         {
            y(d, c, 0, f) += -Bv[d][c];
            y(d, c, 1, f) +=  Bv[d][c];
         }
      }

      // term sigma * < [u], {Q dv/dn} >

      // term kappa * < {Q/h} [u], [v] >:

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
                               const Array<double> &G,
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
         case 0x22: return PADGDiffusionApply2D<2,2>(NF,B,Bt,G,sigma,kappa,pa_Q,pa_hi,
                                                        pa_nJi,x,dxdn,y,dydn);
         case 0x33: return PADGDiffusionApply2D<3,3>(NF,B,Bt,G,sigma,kappa,pa_Q,pa_hi,
                                                        pa_nJi,x,dxdn,y,dydn);
         case 0x44: return PADGDiffusionApply2D<4,4>(NF,B,Bt,G,sigma,kappa,pa_Q,pa_hi,
                                                        pa_nJi,x,dxdn,y,dydn);
         case 0x55: return PADGDiffusionApply2D<5,5>(NF,B,Bt,G,sigma,kappa,pa_Q,pa_hi,
                                                        pa_nJi,x,dxdn,y,dydn);
         case 0x66: return PADGDiffusionApply2D<6,6>(NF,B,Bt,G,sigma,kappa,pa_Q,pa_hi,
                                                        pa_nJi,x,dxdn,y,dydn);
         case 0x77: return PADGDiffusionApply2D<7,7>(NF,B,Bt,G,sigma,kappa,pa_Q,pa_hi,
                                                        pa_nJi,x,dxdn,y,dydn);
         case 0x88: return PADGDiffusionApply2D<8,8>(NF,B,Bt,G,sigma,kappa,pa_Q,pa_hi,
                                                        pa_nJi,x,dxdn,y,dydn);
         case 0x99: return PADGDiffusionApply2D<9,9>(NF,B,Bt,G,sigma,kappa,pa_Q,pa_hi,
                                                        pa_nJi,x,dxdn,y,dydn);
         default:   return PADGDiffusionApply2D(NF,B,Bt,G,sigma,kappa,pa_Q,pa_hi,pa_nJi,
                                                   x,dxdn,y,dydn,D1D,Q1D);
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
                      maps->B, maps->Bt, maps->G,
                      sigma, kappa, pa_Q, pa_hi, pa_nJi, x, dxdn, y, dydn);
}

const IntegrationRule &DGDiffusionIntegrator::GetRule(
   int order, FaceElementTransformations &T)
{
   int int_order = T.Elem1->OrderW() + 2*order;
   return irs.Get(T.GetGeometryType(), int_order);
}

} // namespace mfem
