// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
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
#include "restriction.hpp"

using namespace std;

namespace mfem
{
// PA DG Trace Integrator
static void PADGTraceSetup2D(const int Q1D,
                             const int NF,
                             const Array<double> &w,
                             const Vector &det,
                             const Vector &nor,
                             const Vector &rho,
                             const Vector &vel,
                             const double alpha,
                             const double beta,
                             Vector &op)
{
   const int VDIM = 2;

   auto d = Reshape(det.Read(), Q1D, NF);
   auto n = Reshape(nor.Read(), Q1D, VDIM, NF);
   const bool const_r = rho.Size() == 1;
   auto R =
      const_r ? Reshape(rho.Read(), 1,1) : Reshape(rho.Read(), Q1D,NF);
   const bool const_v = vel.Size() == 2;
   auto V =
      const_v ? Reshape(vel.Read(), 2,1,1) : Reshape(vel.Read(), 2,Q1D,NF);
   auto W = w.Read();
   auto qd = Reshape(op.Write(), Q1D, 2, 2, NF);

   MFEM_FORALL(f, NF,//can be optimized with Q1D thread for NF blocks
   {
      for (int q = 0; q < Q1D; ++q)
      {
         const double r = const_r ? R(0,0) : R(q,f);
         const double v0 = const_v ? V(0,0,0) : V(0,q,f);
         const double v1 = const_v ? V(1,0,0) : V(1,q,f);
         const double dot = n(q,0,f) * v0 + n(q,1,f) * v1;
         const double abs = dot > 0.0 ? dot : -dot;
         const double w = W[q]*r*d(q,f);
         qd(q,0,0,f) = w*( alpha/2 * dot + beta * abs );
         qd(q,1,0,f) = w*( alpha/2 * dot - beta * abs );
         qd(q,0,1,f) = w*(-alpha/2 * dot - beta * abs );
         qd(q,1,1,f) = w*(-alpha/2 * dot + beta * abs );
      }
   });
}

static void PADGTraceSetup3D(const int Q1D,
                             const int NF,
                             const Array<double> &w,
                             const Vector &det,
                             const Vector &nor,
                             const Vector &rho,
                             const Vector &vel,
                             const double alpha,
                             const double beta,
                             Vector &op)
{
   const int VDIM = 3;

   auto d = Reshape(det.Read(), Q1D, Q1D, NF);
   auto n = Reshape(nor.Read(), Q1D, Q1D, VDIM, NF);
   const bool const_r = rho.Size() == 1;
   auto R =
      const_r ? Reshape(rho.Read(), 1,1,1) : Reshape(rho.Read(), Q1D,Q1D,NF);
   const bool const_v = vel.Size() == 3;
   auto V =
      const_v ? Reshape(vel.Read(), 3,1,1,1) : Reshape(vel.Read(), 3,Q1D,Q1D,NF);
   auto W = w.Read();
   auto qd = Reshape(op.Write(), Q1D, Q1D, 2, 2, NF);

   MFEM_FORALL(f, NF,//can be optimized with Q1D*Q1D threads for NF blocks
   {
      for (int q1 = 0; q1 < Q1D; ++q1)
      {
         for (int q2 = 0; q2 < Q1D; ++q2)
         {
            const double r = const_r ? R(0,0,0) : R(q1,q2,f);
            const double v0 = const_v ? V(0,0,0,0) : V(0,q1,q2,f);
            const double v1 = const_v ? V(1,0,0,0) : V(1,q1,q2,f);
            const double v2 = const_v ? V(2,0,0,0) : V(2,q1,q2,f);
            const double dot = n(q1,q2,0,f) * v0 + n(q1,q2,1,f) * v1 +
            /* */              n(q1,q2,2,f) * v2;
            const double abs = dot > 0.0 ? dot : -dot;
            const double w = W[q1+q2*Q1D]*r*d(q1,q2,f);
            qd(q1,q2,0,0,f) = w*( alpha/2 * dot + beta * abs );
            qd(q1,q2,1,0,f) = w*( alpha/2 * dot - beta * abs );
            qd(q1,q2,0,1,f) = w*(-alpha/2 * dot - beta * abs );
            qd(q1,q2,1,1,f) = w*(-alpha/2 * dot + beta * abs );
         }
      }
   });
}

static void PADGTraceSetup(const int dim,
                           const int D1D,
                           const int Q1D,
                           const int NF,
                           const Array<double> &W,
                           const Vector &det,
                           const Vector &nor,
                           const Vector &rho,
                           const Vector &u,
                           const double alpha,
                           const double beta,
                           Vector &op)
{
   if (dim == 1) { MFEM_ABORT("dim==1 not supported in PADGTraceSetup"); }
   if (dim == 2)
   {
      PADGTraceSetup2D(Q1D, NF, W, det, nor, rho, u, alpha, beta, op);
   }
   if (dim == 3)
   {
      PADGTraceSetup3D(Q1D, NF, W, det, nor, rho, u, alpha, beta, op);
   }
}

void DGTraceIntegrator::SetupPA(const FiniteElementSpace &fes, FaceType type)
{
   nf = fes.GetNFbyType(type);
   if (nf==0) { return; }
   // Assumes tensor-product elements
   Mesh *mesh = fes.GetMesh();
   const FiniteElement &el =
      *fes.GetTraceElement(0, fes.GetMesh()->GetFaceBaseGeometry(0));
   FaceElementTransformations &T =
      *fes.GetMesh()->GetFaceElementTransformations(0);
   const IntegrationRule *ir = IntRule?
                               IntRule:
                               &GetRule(el.GetGeomType(), el.GetOrder(), T);
   const int symmDims = 4;
   const int nq = ir->GetNPoints();
   dim = mesh->Dimension();
   geom = mesh->GetFaceGeometricFactors(
             *ir,
             FaceGeometricFactors::DETERMINANTS |
             FaceGeometricFactors::NORMALS, type);
   maps = &el.GetDofToQuad(*ir, DofToQuad::TENSOR);
   dofs1D = maps->ndof;
   quad1D = maps->nqpt;
   pa_data.SetSize(symmDims * nq * nf, Device::GetMemoryType());
   Vector r;
   if (rho==nullptr)
   {
      r.SetSize(1);
      r(0) = 1.0;
   }
   else if (ConstantCoefficient *c_rho = dynamic_cast<ConstantCoefficient*>(rho))
   {
      r.SetSize(1);
      r(0) = c_rho->constant;
   }
   else
   {
      r.SetSize(nq * nf);
      auto C = Reshape(r.HostWrite(), nq, nf);
      int f_ind = 0;
      for (int f = 0; f < fes.GetNF(); ++f)
      {
         int e1, e2;
         int inf1, inf2;
         fes.GetMesh()->GetFaceElements(f, &e1, &e2);
         fes.GetMesh()->GetFaceInfos(f, &inf1, &inf2);
         int face_id = inf1 / 64;
         if ((type==FaceType::Interior && (e2>=0 || (e2<0 && inf2>=0))) ||
             (type==FaceType::Boundary && e2<0 && inf2<0) )
         {
            ElementTransformation& T = *fes.GetMesh()->GetFaceTransformation(f);
            for (int q = 0; q < nq; ++q)
            {
               // Convert to lexicographic ordering
               int iq = ToLexOrdering(dim, face_id, quad1D, q);
               C(iq,f_ind) = rho->Eval(T, ir->IntPoint(q));
            }
            f_ind++;
         }
      }
      MFEM_VERIFY(f_ind==nf, "Incorrect number of faces.");
   }
   Vector vel;
   if (VectorConstantCoefficient *c_u = dynamic_cast<VectorConstantCoefficient*>
                                        (u))
   {
      vel = c_u->GetVec();
   }
   else
   {
      vel.SetSize(dim * nq * nf);
      auto C = Reshape(vel.HostWrite(), dim, nq, nf);
      Vector Vq(dim);
      int f_ind = 0;
      for (int f = 0; f < fes.GetNF(); ++f)
      {
         int e1, e2;
         int inf1, inf2;
         fes.GetMesh()->GetFaceElements(f, &e1, &e2);
         fes.GetMesh()->GetFaceInfos(f, &inf1, &inf2);
         int face_id = inf1 / 64;
         if ((type==FaceType::Interior && (e2>=0 || (e2<0 && inf2>=0))) ||
             (type==FaceType::Boundary && e2<0 && inf2<0) )
         {
            ElementTransformation& T = *fes.GetMesh()->GetFaceTransformation(f);
            for (int q = 0; q < nq; ++q)
            {
               // Convert to lexicographic ordering
               int iq = ToLexOrdering(dim, face_id, quad1D, q);
               u->Eval(Vq, T, ir->IntPoint(q));
               for (int i = 0; i < dim; ++i)
               {
                  C(i,iq,f_ind) = Vq(i);
               }
            }
            f_ind++;
         }
      }
      MFEM_VERIFY(f_ind==nf, "Incorrect number of faces.");
   }
   PADGTraceSetup(dim, dofs1D, quad1D, nf, ir->GetWeights(),
                  geom->detJ, geom->normal, r, vel,
                  alpha, beta, pa_data);
}

void DGTraceIntegrator::AssemblePAInteriorFaces(const FiniteElementSpace& fes)
{
   SetupPA(fes, FaceType::Interior);
}

void DGTraceIntegrator::AssemblePABoundaryFaces(const FiniteElementSpace& fes)
{
   SetupPA(fes, FaceType::Boundary);
}

// PA DGTrace Apply 2D kernel for Gauss-Lobatto/Bernstein
template<int T_D1D = 0, int T_Q1D = 0> static
void PADGTraceApply2D(const int NF,
                      const Array<double> &b,
                      const Array<double> &bt,
                      const Vector &_op,
                      const Vector &_x,
                      Vector &_y,
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
   auto op = Reshape(_op.Read(), Q1D, 2, 2, NF);
   auto x = Reshape(_x.Read(), D1D, VDIM, 2, NF);
   auto y = Reshape(_y.ReadWrite(), D1D, VDIM, 2, NF);

   MFEM_FORALL(f, NF,
   {
      const int VDIM = 1;
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      // the following variables are evaluated at compile time
      constexpr int max_D1D = T_D1D ? T_D1D : MAX_D1D;
      constexpr int max_Q1D = T_Q1D ? T_Q1D : MAX_Q1D;
      double u0[max_D1D][VDIM];
      double u1[max_D1D][VDIM];
      for (int d = 0; d < D1D; d++)
      {
         for (int c = 0; c < VDIM; c++)
         {
            u0[d][c] = x(d,c,0,f);
            u1[d][c] = x(d,c,1,f);
         }
      }
      double Bu0[max_Q1D][VDIM];
      double Bu1[max_Q1D][VDIM];
      for (int q = 0; q < Q1D; ++q)
      {
         for (int c = 0; c < VDIM; c++)
         {
            Bu0[q][c] = 0.0;
            Bu1[q][c] = 0.0;
         }
         for (int d = 0; d < D1D; ++d)
         {
            const double b = B(q,d);
            for (int c = 0; c < VDIM; c++)
            {
               Bu0[q][c] += b*u0[d][c];
               Bu1[q][c] += b*u1[d][c];
            }
         }
      }
      double DBu[max_Q1D][VDIM];
      for (int q = 0; q < Q1D; ++q)
      {
         for (int c = 0; c < VDIM; c++)
         {
            DBu[q][c] = op(q,0,0,f)*Bu0[q][c] + op(q,1,0,f)*Bu1[q][c];
         }
      }
      double BDBu[max_D1D][VDIM];
      for (int d = 0; d < D1D; ++d)
      {
         for (int c = 0; c < VDIM; c++)
         {
            BDBu[d][c] = 0.0;
         }
         for (int q = 0; q < Q1D; ++q)
         {
            const double b = Bt(d,q);
            for (int c = 0; c < VDIM; c++)
            {
               BDBu[d][c] += b*DBu[q][c];
            }
         }
         for (int c = 0; c < VDIM; c++)
         {
            y(d,c,0,f) +=  BDBu[d][c];
            y(d,c,1,f) += -BDBu[d][c];
         }
      }
   });
}

// PA DGTrace Apply 3D kernel for Gauss-Lobatto/Bernstein
template<int T_D1D = 0, int T_Q1D = 0> static
void PADGTraceApply3D(const int NF,
                      const Array<double> &b,
                      const Array<double> &bt,
                      const Vector &_op,
                      const Vector &_x,
                      Vector &_y,
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
   auto op = Reshape(_op.Read(), Q1D, Q1D, 2, 2, NF);
   auto x = Reshape(_x.Read(), D1D, D1D, VDIM, 2, NF);
   auto y = Reshape(_y.ReadWrite(), D1D, D1D, VDIM, 2, NF);

   MFEM_FORALL(f, NF,
   {
      const int VDIM = 1;
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      // the following variables are evaluated at compile time
      constexpr int max_D1D = T_D1D ? T_D1D : MAX_D1D;
      constexpr int max_Q1D = T_Q1D ? T_Q1D : MAX_Q1D;
      double u0[max_D1D][max_D1D][VDIM];
      double u1[max_D1D][max_D1D][VDIM];
      for (int d1 = 0; d1 < D1D; d1++)
      {
         for (int d2 = 0; d2 < D1D; d2++)
         {
            for (int c = 0; c < VDIM; c++)
            {
               u0[d1][d2][c] = x(d1,d2,c,0,f);
               u1[d1][d2][c] = x(d1,d2,c,1,f);
            }
         }
      }
      double Bu0[max_Q1D][max_D1D][VDIM];
      double Bu1[max_Q1D][max_D1D][VDIM];
      for (int q = 0; q < Q1D; ++q)
      {
         for (int d2 = 0; d2 < D1D; d2++)
         {
            for (int c = 0; c < VDIM; c++)
            {
               Bu0[q][d2][c] = 0.0;
               Bu1[q][d2][c] = 0.0;
            }
            for (int d1 = 0; d1 < D1D; ++d1)
            {
               const double b = B(q,d1);
               for (int c = 0; c < VDIM; c++)
               {
                  Bu0[q][d2][c] += b*u0[d1][d2][c];
                  Bu1[q][d2][c] += b*u1[d1][d2][c];
               }
            }
         }
      }
      double BBu0[max_Q1D][max_Q1D][VDIM];
      double BBu1[max_Q1D][max_Q1D][VDIM];
      for (int q1 = 0; q1 < Q1D; ++q1)
      {
         for (int q2 = 0; q2 < Q1D; q2++)
         {
            for (int c = 0; c < VDIM; c++)
            {
               BBu0[q1][q2][c] = 0.0;
               BBu1[q1][q2][c] = 0.0;
            }
            for (int d2 = 0; d2 < D1D; ++d2)
            {
               const double b = B(q2,d2);
               for (int c = 0; c < VDIM; c++)
               {
                  BBu0[q1][q2][c] += b*Bu0[q1][d2][c];
                  BBu1[q1][q2][c] += b*Bu1[q1][d2][c];
               }
            }
         }
      }
      double DBBu[max_Q1D][max_Q1D][VDIM];
      for (int q1 = 0; q1 < Q1D; ++q1)
      {
         for (int q2 = 0; q2 < Q1D; q2++)
         {
            for (int c = 0; c < VDIM; c++)
            {
               DBBu[q1][q2][c] = op(q1,q2,0,0,f)*BBu0[q1][q2][c] +
                                 op(q1,q2,1,0,f)*BBu1[q1][q2][c];
            }
         }
      }
      double BDBBu[max_Q1D][max_D1D][VDIM];
      for (int q1 = 0; q1 < Q1D; ++q1)
      {
         for (int d2 = 0; d2 < D1D; d2++)
         {
            for (int c = 0; c < VDIM; c++)
            {
               BDBBu[q1][d2][c] = 0.0;
            }
            for (int q2 = 0; q2 < Q1D; ++q2)
            {
               const double b = Bt(d2,q2);
               for (int c = 0; c < VDIM; c++)
               {
                  BDBBu[q1][d2][c] += b*DBBu[q1][q2][c];
               }
            }
         }
      }
      double BBDBBu[max_D1D][max_D1D][VDIM];
      for (int d1 = 0; d1 < D1D; ++d1)
      {
         for (int d2 = 0; d2 < D1D; d2++)
         {
            for (int c = 0; c < VDIM; c++)
            {
               BBDBBu[d1][d2][c] = 0.0;
            }
            for (int q1 = 0; q1 < Q1D; ++q1)
            {
               const double b = Bt(d1,q1);
               for (int c = 0; c < VDIM; c++)
               {
                  BBDBBu[d1][d2][c] += b*BDBBu[q1][d2][c];
               }
            }
            for (int c = 0; c < VDIM; c++)
            {
               y(d1,d2,c,0,f) +=  BBDBBu[d1][d2][c];
               y(d1,d2,c,1,f) += -BBDBBu[d1][d2][c];
            }
         }
      }
   });
}

// Optimized PA DGTrace Apply 3D kernel for Gauss-Lobatto/Bernstein
template<int T_D1D = 0, int T_Q1D = 0, int T_NBZ = 0> static
void SmemPADGTraceApply3D(const int NF,
                          const Array<double> &b,
                          const Array<double> &bt,
                          const Vector &_op,
                          const Vector &_x,
                          Vector &_y,
                          const int d1d = 0,
                          const int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   constexpr int NBZ = T_NBZ ? T_NBZ : 1;
   MFEM_VERIFY(D1D <= MAX_D1D, "");
   MFEM_VERIFY(Q1D <= MAX_Q1D, "");
   auto B = Reshape(b.Read(), Q1D, D1D);
   auto Bt = Reshape(bt.Read(), D1D, Q1D);
   auto op = Reshape(_op.Read(), Q1D, Q1D, 2, 2, NF);
   auto x = Reshape(_x.Read(), D1D, D1D, 2, NF);
   auto y = Reshape(_y.ReadWrite(), D1D, D1D, 2, NF);

   MFEM_FORALL_2D(f, NF, Q1D, Q1D, NBZ,
   {
      const int tidz = MFEM_THREAD_ID(z);
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      // the following variables are evaluated at compile time
      constexpr int NBZ = T_NBZ ? T_NBZ : 1;
      constexpr int max_D1D = T_D1D ? T_D1D : MAX_D1D;
      constexpr int max_Q1D = T_Q1D ? T_Q1D : MAX_Q1D;
      MFEM_SHARED double u0[NBZ][max_D1D][max_D1D];
      MFEM_SHARED double u1[NBZ][max_D1D][max_D1D];
      MFEM_FOREACH_THREAD(d1,x,D1D)
      {
         MFEM_FOREACH_THREAD(d2,y,D1D)
         {
            u0[tidz][d1][d2] = x(d1,d2,0,f+tidz);
            u1[tidz][d1][d2] = x(d1,d2,1,f+tidz);
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_SHARED double Bu0[NBZ][max_Q1D][max_D1D];
      MFEM_SHARED double Bu1[NBZ][max_Q1D][max_D1D];
      MFEM_FOREACH_THREAD(q1,x,Q1D)
      {
         MFEM_FOREACH_THREAD(d2,y,D1D)
         {
            double Bu0_ = 0.0;
            double Bu1_ = 0.0;
            for (int d1 = 0; d1 < D1D; ++d1)
            {
               const double b = B(q1,d1);
               Bu0_ += b*u0[tidz][d1][d2];
               Bu1_ += b*u1[tidz][d1][d2];
            }
            Bu0[tidz][q1][d2] = Bu0_;
            Bu1[tidz][q1][d2] = Bu1_;
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_SHARED double BBu0[NBZ][max_Q1D][max_Q1D];
      MFEM_SHARED double BBu1[NBZ][max_Q1D][max_Q1D];
      MFEM_FOREACH_THREAD(q1,x,Q1D)
      {
         MFEM_FOREACH_THREAD(q2,y,Q1D)
         {
            double BBu0_ = 0.0;
            double BBu1_ = 0.0;
            for (int d2 = 0; d2 < D1D; ++d2)
            {
               const double b = B(q2,d2);
               BBu0_ += b*Bu0[tidz][q1][d2];
               BBu1_ += b*Bu1[tidz][q1][d2];
            }
            BBu0[tidz][q1][q2] = BBu0_;
            BBu1[tidz][q1][q2] = BBu1_;
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_SHARED double DBBu[NBZ][max_Q1D][max_Q1D];
      MFEM_FOREACH_THREAD(q1,x,Q1D)
      {
         MFEM_FOREACH_THREAD(q2,y,Q1D)
         {
            DBBu[tidz][q1][q2] = op(q1,q2,0,0,f+tidz)*BBu0[tidz][q1][q2] +
                                 op(q1,q2,1,0,f+tidz)*BBu1[tidz][q1][q2];
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_SHARED double BDBBu[NBZ][max_Q1D][max_D1D];
      MFEM_FOREACH_THREAD(q1,x,Q1D)
      {
         MFEM_FOREACH_THREAD(d2,y,D1D)
         {
            double BDBBu_ = 0.0;
            for (int q2 = 0; q2 < Q1D; ++q2)
            {
               const double b = Bt(d2,q2);
               BDBBu_ += b*DBBu[tidz][q1][q2];
            }
            BDBBu[tidz][q1][d2] = BDBBu_;
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(d1,x,D1D)
      {
         MFEM_FOREACH_THREAD(d2,y,D1D)
         {
            double BBDBBu_ = 0.0;
            for (int q1 = 0; q1 < Q1D; ++q1)
            {
               const double b = Bt(d1,q1);
               BBDBBu_ += b*BDBBu[tidz][q1][d2];
            }
            y(d1,d2,0,f+tidz) +=  BBDBBu_;
            y(d1,d2,1,f+tidz) += -BBDBBu_;
         }
      }
   });
}

static void PADGTraceApply(const int dim,
                           const int D1D,
                           const int Q1D,
                           const int NF,
                           const Array<double> &B,
                           const Array<double> &Bt,
                           const Vector &op,
                           const Vector &x,
                           Vector &y)
{
   if (dim == 2)
   {
      switch ((D1D << 4 ) | Q1D)
      {
         case 0x22: return PADGTraceApply2D<2,2>(NF,B,Bt,op,x,y);
         case 0x33: return PADGTraceApply2D<3,3>(NF,B,Bt,op,x,y);
         case 0x44: return PADGTraceApply2D<4,4>(NF,B,Bt,op,x,y);
         case 0x55: return PADGTraceApply2D<5,5>(NF,B,Bt,op,x,y);
         case 0x66: return PADGTraceApply2D<6,6>(NF,B,Bt,op,x,y);
         case 0x77: return PADGTraceApply2D<7,7>(NF,B,Bt,op,x,y);
         case 0x88: return PADGTraceApply2D<8,8>(NF,B,Bt,op,x,y);
         case 0x99: return PADGTraceApply2D<9,9>(NF,B,Bt,op,x,y);
         default:   return PADGTraceApply2D(NF,B,Bt,op,x,y,D1D,Q1D);
      }
   }
   else if (dim == 3)
   {
      switch ((D1D << 4 ) | Q1D)
      {
         case 0x23: return SmemPADGTraceApply3D<2,3,1>(NF,B,Bt,op,x,y);
         case 0x34: return SmemPADGTraceApply3D<3,4,2>(NF,B,Bt,op,x,y);
         case 0x45: return SmemPADGTraceApply3D<4,5,2>(NF,B,Bt,op,x,y);
         case 0x56: return SmemPADGTraceApply3D<5,6,1>(NF,B,Bt,op,x,y);
         case 0x67: return SmemPADGTraceApply3D<6,7,1>(NF,B,Bt,op,x,y);
         case 0x78: return SmemPADGTraceApply3D<7,8,1>(NF,B,Bt,op,x,y);
         case 0x89: return SmemPADGTraceApply3D<8,9,1>(NF,B,Bt,op,x,y);
         default:   return PADGTraceApply3D(NF,B,Bt,op,x,y,D1D,Q1D);
      }
   }
   MFEM_ABORT("Unknown kernel.");
}

// PA DGTrace Apply 2D kernel for Gauss-Lobatto/Bernstein
template<int T_D1D = 0, int T_Q1D = 0> static
void PADGTraceApplyTranspose2D(const int NF,
                               const Array<double> &b,
                               const Array<double> &bt,
                               const Vector &_op,
                               const Vector &_x,
                               Vector &_y,
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
   auto op = Reshape(_op.Read(), Q1D, 2, 2, NF);
   auto x = Reshape(_x.Read(), D1D, VDIM, 2, NF);
   auto y = Reshape(_y.ReadWrite(), D1D, VDIM, 2, NF);

   MFEM_FORALL(f, NF,
   {
      const int VDIM = 1;
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      // the following variables are evaluated at compile time
      constexpr int max_D1D = T_D1D ? T_D1D : MAX_D1D;
      constexpr int max_Q1D = T_Q1D ? T_Q1D : MAX_Q1D;
      double u0[max_D1D][VDIM];
      double u1[max_D1D][VDIM];
      for (int d = 0; d < D1D; d++)
      {
         for (int c = 0; c < VDIM; c++)
         {
            u0[d][c] = x(d,c,0,f);
            u1[d][c] = x(d,c,1,f);
         }
      }
      double Bu0[max_Q1D][VDIM];
      double Bu1[max_Q1D][VDIM];
      for (int q = 0; q < Q1D; ++q)
      {
         for (int c = 0; c < VDIM; c++)
         {
            Bu0[q][c] = 0.0;
            Bu1[q][c] = 0.0;
         }
         for (int d = 0; d < D1D; ++d)
         {
            const double b = B(q,d);
            for (int c = 0; c < VDIM; c++)
            {
               Bu0[q][c] += b*u0[d][c];
               Bu1[q][c] += b*u1[d][c];
            }
         }
      }
      double DBu0[max_Q1D][VDIM];
      double DBu1[max_Q1D][VDIM];
      for (int q = 0; q < Q1D; ++q)
      {
         for (int c = 0; c < VDIM; c++)
         {
            DBu0[q][c] = op(q,0,0,f)*Bu0[q][c] + op(q,0,1,f)*Bu1[q][c];
            DBu1[q][c] = op(q,1,0,f)*Bu0[q][c] + op(q,1,1,f)*Bu1[q][c];
         }
      }
      double BDBu0[max_D1D][VDIM];
      double BDBu1[max_D1D][VDIM];
      for (int d = 0; d < D1D; ++d)
      {
         for (int c = 0; c < VDIM; c++)
         {
            BDBu0[d][c] = 0.0;
            BDBu1[d][c] = 0.0;
         }
         for (int q = 0; q < Q1D; ++q)
         {
            const double b = Bt(d,q);
            for (int c = 0; c < VDIM; c++)
            {
               BDBu0[d][c] += b*DBu0[q][c];
               BDBu1[d][c] += b*DBu1[q][c];
            }
         }
         for (int c = 0; c < VDIM; c++)
         {
            y(d,c,0,f) += BDBu0[d][c];
            y(d,c,1,f) += BDBu1[d][c];
         }
      }
   });
}

// PA DGTrace Apply Transpose 3D kernel for Gauss-Lobatto/Bernstein
template<int T_D1D = 0, int T_Q1D = 0> static
void PADGTraceApplyTranspose3D(const int NF,
                               const Array<double> &b,
                               const Array<double> &bt,
                               const Vector &_op,
                               const Vector &_x,
                               Vector &_y,
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
   auto op = Reshape(_op.Read(), Q1D, Q1D, 2, 2, NF);
   auto x = Reshape(_x.Read(), D1D, D1D, VDIM, 2, NF);
   auto y = Reshape(_y.ReadWrite(), D1D, D1D, VDIM, 2, NF);

   MFEM_FORALL(f, NF,
   {
      const int VDIM = 1;
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      // the following variables are evaluated at compile time
      constexpr int max_D1D = T_D1D ? T_D1D : MAX_D1D;
      constexpr int max_Q1D = T_Q1D ? T_Q1D : MAX_Q1D;
      double u0[max_D1D][max_D1D][VDIM];
      double u1[max_D1D][max_D1D][VDIM];
      for (int d1 = 0; d1 < D1D; d1++)
      {
         for (int d2 = 0; d2 < D1D; d2++)
         {
            for (int c = 0; c < VDIM; c++)
            {
               u0[d1][d2][c] = x(d1,d2,c,0,f);
               u1[d1][d2][c] = x(d1,d2,c,1,f);
            }
         }
      }
      double Bu0[max_Q1D][max_D1D][VDIM];
      double Bu1[max_Q1D][max_D1D][VDIM];
      for (int q1 = 0; q1 < Q1D; ++q1)
      {
         for (int d2 = 0; d2 < D1D; ++d2)
         {
            for (int c = 0; c < VDIM; c++)
            {
               Bu0[q1][d2][c] = 0.0;
               Bu1[q1][d2][c] = 0.0;
            }
            for (int d1 = 0; d1 < D1D; ++d1)
            {
               const double b = B(q1,d1);
               for (int c = 0; c < VDIM; c++)
               {
                  Bu0[q1][d2][c] += b*u0[d1][d2][c];
                  Bu1[q1][d2][c] += b*u1[d1][d2][c];
               }
            }
         }
      }
      double BBu0[max_Q1D][max_Q1D][VDIM];
      double BBu1[max_Q1D][max_Q1D][VDIM];
      for (int q1 = 0; q1 < Q1D; ++q1)
      {
         for (int q2 = 0; q2 < Q1D; ++q2)
         {
            for (int c = 0; c < VDIM; c++)
            {
               BBu0[q1][q2][c] = 0.0;
               BBu1[q1][q2][c] = 0.0;
            }
            for (int d2 = 0; d2 < D1D; ++d2)
            {
               const double b = B(q2,d2);
               for (int c = 0; c < VDIM; c++)
               {
                  BBu0[q1][q2][c] += b*Bu0[q1][d2][c];
                  BBu1[q1][q2][c] += b*Bu1[q1][d2][c];
               }
            }
         }
      }
      double DBu0[max_Q1D][max_Q1D][VDIM];
      double DBu1[max_Q1D][max_Q1D][VDIM];
      for (int q1 = 0; q1 < Q1D; ++q1)
      {
         for (int q2 = 0; q2 < Q1D; ++q2)
         {
            const double D00 = op(q1,q2,0,0,f);
            const double D01 = op(q1,q2,0,1,f);
            const double D10 = op(q1,q2,1,0,f);
            const double D11 = op(q1,q2,1,1,f);
            for (int c = 0; c < VDIM; c++)
            {
               DBu0[q1][q2][c] = D00*BBu0[q1][q2][c] + D01*BBu1[q1][q2][c];
               DBu1[q1][q2][c] = D10*BBu0[q1][q2][c] + D11*BBu1[q1][q2][c];
            }
         }
      }
      double BDBu0[max_D1D][max_Q1D][VDIM];
      double BDBu1[max_D1D][max_Q1D][VDIM];
      for (int d1 = 0; d1 < D1D; ++d1)
      {
         for (int q2 = 0; q2 < Q1D; ++q2)
         {
            for (int c = 0; c < VDIM; c++)
            {
               BDBu0[d1][q2][c] = 0.0;
               BDBu1[d1][q2][c] = 0.0;
            }
            for (int q1 = 0; q1 < Q1D; ++q1)
            {
               const double b = Bt(d1,q1);
               for (int c = 0; c < VDIM; c++)
               {
                  BDBu0[d1][q2][c] += b*DBu0[q1][q2][c];
                  BDBu1[d1][q2][c] += b*DBu1[q1][q2][c];
               }
            }
         }
      }
      double BBDBu0[max_D1D][max_D1D][VDIM];
      double BBDBu1[max_D1D][max_D1D][VDIM];
      for (int d1 = 0; d1 < D1D; ++d1)
      {
         for (int d2 = 0; d2 < D1D; ++d2)
         {
            for (int c = 0; c < VDIM; c++)
            {
               BBDBu0[d1][d2][c] = 0.0;
               BBDBu1[d1][d2][c] = 0.0;
            }
            for (int q2 = 0; q2 < Q1D; ++q2)
            {
               const double b = Bt(d2,q2);
               for (int c = 0; c < VDIM; c++)
               {
                  BBDBu0[d1][d2][c] += b*BDBu0[d1][q2][c];
                  BBDBu1[d1][d2][c] += b*BDBu1[d1][q2][c];
               }
            }
            for (int c = 0; c < VDIM; c++)
            {
               y(d1,d2,c,0,f) += BBDBu0[d1][d2][c];
               y(d1,d2,c,1,f) += BBDBu1[d1][d2][c];
            }
         }
      }
   });
}

// Optimized PA DGTrace Apply Transpose 3D kernel for Gauss-Lobatto/Bernstein
template<int T_D1D = 0, int T_Q1D = 0, int T_NBZ = 0> static
void SmemPADGTraceApplyTranspose3D(const int NF,
                                   const Array<double> &b,
                                   const Array<double> &bt,
                                   const Vector &_op,
                                   const Vector &_x,
                                   Vector &_y,
                                   const int d1d = 0,
                                   const int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   constexpr int NBZ = T_NBZ ? T_NBZ : 1;
   MFEM_VERIFY(D1D <= MAX_D1D, "");
   MFEM_VERIFY(Q1D <= MAX_Q1D, "");
   auto B = Reshape(b.Read(), Q1D, D1D);
   auto Bt = Reshape(bt.Read(), D1D, Q1D);
   auto op = Reshape(_op.Read(), Q1D, Q1D, 2, 2, NF);
   auto x = Reshape(_x.Read(), D1D, D1D, 2, NF);
   auto y = Reshape(_y.ReadWrite(), D1D, D1D, 2, NF);

   MFEM_FORALL_2D(f, NF, Q1D, Q1D, NBZ,
   {
      const int tidz = MFEM_THREAD_ID(z);
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      // the following variables are evaluated at compile time
      constexpr int NBZ = T_NBZ ? T_NBZ : 1;
      constexpr int max_D1D = T_D1D ? T_D1D : MAX_D1D;
      constexpr int max_Q1D = T_Q1D ? T_Q1D : MAX_Q1D;
      MFEM_SHARED double u0[NBZ][max_D1D][max_D1D];
      MFEM_SHARED double u1[NBZ][max_D1D][max_D1D];
      MFEM_FOREACH_THREAD(d1,x,D1D)
      {
         MFEM_FOREACH_THREAD(d2,y,D1D)
         {
            u0[tidz][d1][d2] = x(d1,d2,0,f+tidz);
            u1[tidz][d1][d2] = x(d1,d2,1,f+tidz);
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_SHARED double Bu0[NBZ][max_Q1D][max_D1D];
      MFEM_SHARED double Bu1[NBZ][max_Q1D][max_D1D];
      MFEM_FOREACH_THREAD(q1,x,Q1D)
      {
         MFEM_FOREACH_THREAD(d2,y,D1D)
         {
            double Bu0_ = 0.0;
            double Bu1_ = 0.0;
            for (int d1 = 0; d1 < D1D; ++d1)
            {
               const double b = B(q1,d1);
               Bu0_ += b*u0[tidz][d1][d2];
               Bu1_ += b*u1[tidz][d1][d2];
            }
            Bu0[tidz][q1][d2] = Bu0_;
            Bu1[tidz][q1][d2] = Bu1_;
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_SHARED double BBu0[NBZ][max_Q1D][max_Q1D];
      MFEM_SHARED double BBu1[NBZ][max_Q1D][max_Q1D];
      MFEM_FOREACH_THREAD(q1,x,Q1D)
      {
         MFEM_FOREACH_THREAD(q2,y,Q1D)
         {
            double BBu0_ = 0.0;
            double BBu1_ = 0.0;
            for (int d2 = 0; d2 < D1D; ++d2)
            {
               const double b = B(q2,d2);
               BBu0_ += b*Bu0[tidz][q1][d2];
               BBu1_ += b*Bu1[tidz][q1][d2];
            }
            BBu0[tidz][q1][q2] = BBu0_;
            BBu1[tidz][q1][q2] = BBu1_;
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_SHARED double DBBu0[NBZ][max_Q1D][max_Q1D];
      MFEM_SHARED double DBBu1[NBZ][max_Q1D][max_Q1D];
      MFEM_FOREACH_THREAD(q1,x,Q1D)
      {
         MFEM_FOREACH_THREAD(q2,y,Q1D)
         {
            const double D00 = op(q1,q2,0,0,f+tidz);
            const double D01 = op(q1,q2,0,1,f+tidz);
            const double D10 = op(q1,q2,1,0,f+tidz);
            const double D11 = op(q1,q2,1,1,f+tidz);
            const double u0 = BBu0[tidz][q1][q2];
            const double u1 = BBu1[tidz][q1][q2];
            DBBu0[tidz][q1][q2] = D00*u0 + D01*u1;
            DBBu1[tidz][q1][q2] = D10*u0 + D11*u1;
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_SHARED double BDBBu0[NBZ][max_Q1D][max_D1D];
      MFEM_SHARED double BDBBu1[NBZ][max_Q1D][max_D1D];
      MFEM_FOREACH_THREAD(q1,x,Q1D)
      {
         MFEM_FOREACH_THREAD(d2,y,D1D)
         {
            double BDBBu0_ = 0.0;
            double BDBBu1_ = 0.0;
            for (int q2 = 0; q2 < Q1D; ++q2)
            {
               const double b = Bt(d2,q2);
               BDBBu0_ += b*DBBu0[tidz][q1][q2];
               BDBBu1_ += b*DBBu1[tidz][q1][q2];
            }
            BDBBu0[tidz][q1][d2] = BDBBu0_;
            BDBBu1[tidz][q1][d2] = BDBBu1_;
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(d1,x,D1D)
      {
         MFEM_FOREACH_THREAD(d2,y,D1D)
         {
            double BBDBBu0_ = 0.0;
            double BBDBBu1_ = 0.0;
            for (int q1 = 0; q1 < Q1D; ++q1)
            {
               const double b = Bt(d1,q1);
               BBDBBu0_ += b*BDBBu0[tidz][q1][d2];
               BBDBBu1_ += b*BDBBu1[tidz][q1][d2];
            }
            y(d1,d2,0,f+tidz) += BBDBBu0_;
            y(d1,d2,1,f+tidz) += BBDBBu1_;
         }
      }
   });
}

static void PADGTraceApplyTranspose(const int dim,
                                    const int D1D,
                                    const int Q1D,
                                    const int NF,
                                    const Array<double> &B,
                                    const Array<double> &Bt,
                                    const Vector &op,
                                    const Vector &x,
                                    Vector &y)
{
   if (dim == 2)
   {
      switch ((D1D << 4 ) | Q1D)
      {
         case 0x22: return PADGTraceApplyTranspose2D<2,2>(NF,B,Bt,op,x,y);
         case 0x33: return PADGTraceApplyTranspose2D<3,3>(NF,B,Bt,op,x,y);
         case 0x44: return PADGTraceApplyTranspose2D<4,4>(NF,B,Bt,op,x,y);
         case 0x55: return PADGTraceApplyTranspose2D<5,5>(NF,B,Bt,op,x,y);
         case 0x66: return PADGTraceApplyTranspose2D<6,6>(NF,B,Bt,op,x,y);
         case 0x77: return PADGTraceApplyTranspose2D<7,7>(NF,B,Bt,op,x,y);
         case 0x88: return PADGTraceApplyTranspose2D<8,8>(NF,B,Bt,op,x,y);
         case 0x99: return PADGTraceApplyTranspose2D<9,9>(NF,B,Bt,op,x,y);
         default: return PADGTraceApplyTranspose2D(NF,B,Bt,op,x,y,D1D,Q1D);
      }
   }
   else if (dim == 3)
   {
      switch ((D1D << 4 ) | Q1D)
      {
         case 0x23: return SmemPADGTraceApplyTranspose3D<2,3>(NF,B,Bt,op,x,y);
         case 0x34: return SmemPADGTraceApplyTranspose3D<3,4>(NF,B,Bt,op,x,y);
         case 0x45: return SmemPADGTraceApplyTranspose3D<4,5>(NF,B,Bt,op,x,y);
         case 0x56: return SmemPADGTraceApplyTranspose3D<5,6>(NF,B,Bt,op,x,y);
         case 0x67: return SmemPADGTraceApplyTranspose3D<6,7>(NF,B,Bt,op,x,y);
         case 0x78: return SmemPADGTraceApplyTranspose3D<7,8>(NF,B,Bt,op,x,y);
         case 0x89: return SmemPADGTraceApplyTranspose3D<8,9>(NF,B,Bt,op,x,y);
         default: return PADGTraceApplyTranspose3D(NF,B,Bt,op,x,y,D1D,Q1D);
      }
   }
   MFEM_ABORT("Unknown kernel.");
}

// PA DGTraceIntegrator Apply kernel
void DGTraceIntegrator::AddMultPA(const Vector &x, Vector &y) const
{
   PADGTraceApply(dim, dofs1D, quad1D, nf,
                  maps->B, maps->Bt,
                  pa_data, x, y);
}

void DGTraceIntegrator::AddMultTransposePA(const Vector &x, Vector &y) const
{
   PADGTraceApplyTranspose(dim, dofs1D, quad1D, nf,
                           maps->B, maps->Bt,
                           pa_data, x, y);
}

} // namespace mfem
