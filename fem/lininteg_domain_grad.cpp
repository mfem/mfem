// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "fem.hpp"

#include "../fem/kernels.hpp"
#include "../linalg/kernels.hpp"
#include "../general/forall.hpp"

namespace mfem
{

template<int T_D1D = 0, int T_Q1D = 0> static
void VectorDomainLFGradIntegratorAssemble(const int vdim,
                                          const int ne,
                                          const int d,
                                          const int q,
                                          const int *markers,
                                          const double *b,
                                          const double *g,
                                          const double *jacobians,
                                          const double *detJ,
                                          const double *weights,
                                          const Vector &coeff,
                                          double *y);

using kernel_t = decltype(&VectorDomainLFGradIntegratorAssemble<>);

template<int T_D1D = 0, int T_Q1D = 0> static
void VectorDomainLFGradIntegratorAssemble2D(const int vdim,
                                            const int ne,
                                            const int d,
                                            const int q,
                                            const int *markers,
                                            const double *b,
                                            const double *g,
                                            const double *jacobians,
                                            const double *detJ,
                                            const double *weights,
                                            const Vector &coeff,
                                            double *y)
{
   constexpr int DIM = 2;

   const auto F = coeff.Read();
   const auto M = Reshape(markers, ne);
   const auto B = Reshape(b, q, d);
   const auto G = Reshape(g, q, d);
   const auto J = Reshape(jacobians, q, q, DIM,DIM, ne);
   const auto DetJ = Reshape(detJ, q, q, ne);
   const auto W = Reshape(weights, q, q);
   const bool cst_coeff = coeff.Size() == vdim*DIM;
   const auto C =
      cst_coeff ? Reshape(F,DIM,vdim,1,1,1) : Reshape(F,DIM,vdim,q,q,ne);

   auto Y = Reshape(y, d,d, vdim, ne);

   MFEM_FORALL_2D(e, ne, q, q, 1,
   {
      if (M(e) == 0) { return; } // ignore

      constexpr int Q = T_Q1D ? T_Q1D : MAX_Q1D;
      constexpr int D = T_D1D ? T_D1D : MAX_D1D;

      MFEM_SHARED double sBGt[2][Q*D];
      MFEM_SHARED double sQQ[2][Q*Q];
      MFEM_SHARED double sDQ[2][D*Q];

      const DeviceMatrix Bt(sBGt[0], q,d);
      const DeviceMatrix Gt(sBGt[1], q,d);
      kernels::internal::LoadBGt<D,Q>(d,q,B,G,sBGt);

      const DeviceMatrix QQ0(sQQ[0], q,q);
      const DeviceMatrix QQ1(sQQ[1], q,q);

      const DeviceMatrix DQ0(sDQ[0], d,q);
      const DeviceMatrix DQ1(sDQ[1], d,q);

      for (int c = 0; c < vdim; ++c)
      {
         const double cst_val0 = C(0,c,0,0,0);
         const double cst_val1 = C(1,c,0,0,0);

         MFEM_FOREACH_THREAD(x,x,q)
         {
            MFEM_FOREACH_THREAD(y,y,q)
            {
               double Jloc[4], Jinv[4];
               Jloc[0] = J(x,y,0,0,e);
               Jloc[1] = J(x,y,1,0,e);
               Jloc[2] = J(x,y,0,1,e);
               Jloc[3] = J(x,y,1,1,e);
               const double detJ = DetJ(x,y,e);
               kernels::CalcInverse<2>(Jloc, Jinv);
               const double weight = W(x,y);
               const double u = cst_coeff ? cst_val0 : C(0,c,x,y,e);
               const double v = cst_coeff ? cst_val1 : C(1,c,x,y,e);
               QQ0(y,x) = Jinv[0]*u + Jinv[2]*v;
               QQ1(y,x) = Jinv[1]*u + Jinv[3]*v;
               QQ0(y,x) *= weight * detJ;
               QQ1(y,x) *= weight * detJ;
            }
         }
         MFEM_SYNC_THREAD;
         kernels::internal::GradYt(d,q,Bt,Gt,QQ0,QQ1,DQ0,DQ1);
         kernels::internal::GradXt(d,q,Bt,Gt,DQ0,DQ1,Y,c,e);
      }
   });
}

template<int T_D1D = 0, int T_Q1D = 0> static
void VectorDomainLFGradIntegratorAssemble3D(const int vdim,
                                            const int ne,
                                            const int d,
                                            const int q,
                                            const int *markers,
                                            const double *b,
                                            const double *g,
                                            const double *jacobians,
                                            const double *detJ,
                                            const double *weights,
                                            const Vector &coeff,
                                            double *output)
{
   constexpr int DIM = 3;

   const auto F = coeff.Read();
   const auto M = Reshape(markers, ne);
   const auto B = Reshape(b, q,d);
   const auto G = Reshape(g, q,d);
   const auto J = Reshape(jacobians, q,q,q, DIM,DIM, ne);
   const auto DetJ = Reshape(detJ, q,q,q, ne);
   const auto W = Reshape(weights, q,q,q);
   const bool cst_coeff = coeff.Size() == vdim*DIM;
   const auto C =
      cst_coeff ? Reshape(F,DIM,vdim,1,1,1,1) : Reshape(F,DIM,vdim,q,q,q,ne);

   auto Y = Reshape(output, d,d,d, vdim, ne);

   MFEM_FORALL_2D(e, ne, q, q, 1,
   {
      if (M(e) == 0) { return; } // ignore

#warning MAX_Q = 9
      constexpr int MAX_Q = 9;
      constexpr int Q = T_Q1D ? T_Q1D : MAX_Q;
      constexpr int D = T_D1D ? T_D1D : MAX_Q;

      MFEM_SHARED double sBGt[2][Q*D];
      MFEM_SHARED double sQQ[3][Q*Q*Q];
      MFEM_SHARED double sQD[3][Q*Q*D];

      const DeviceMatrix Bt(sBGt[0], q,d);
      const DeviceMatrix Gt(sBGt[1], q,d);
      kernels::internal::LoadBGt<D,Q>(d,q,B,G,sBGt);

      const DeviceCube QQ0(sQQ[0], q,q,q);
      const DeviceCube QQ1(sQQ[1], q,q,q);
      const DeviceCube QQ2(sQQ[2], q,q,q);

      const DeviceCube QD0(sQD[0], q,q,d);
      const DeviceCube QD1(sQD[1], q,q,d);
      const DeviceCube QD2(sQD[2], q,q,d);

      const DeviceCube DD0(QQ0, q,d,d);
      const DeviceCube DD1(QQ1, q,d,d);
      const DeviceCube DD2(QQ2, q,d,d);

      for (int c = 0; c < vdim; ++c)
      {
         const double cst_val_0 = C(0,c,0,0,0,0);
         const double cst_val_1 = C(1,c,0,0,0,0);
         const double cst_val_2 = C(2,c,0,0,0,0);

         MFEM_FOREACH_THREAD(x,x,q)
         {
            MFEM_FOREACH_THREAD(y,y,q)
            {
               for (int z = 0; z < q; ++z)
               {
                  double Jloc[9], Jinv[9];
                  for (int j = 0; j < 3; j++)
                  {
                     for (int i = 0; i < 3; i++)
                     {
                        Jloc[i+3*j] = J(x,y,z,i,j,e);
                     }
                  }
                  kernels::CalcInverse<3>(Jloc, Jinv);

                  const double u = cst_coeff ? cst_val_0 : C(0,c,x,y,z,e);
                  const double v = cst_coeff ? cst_val_1 : C(1,c,x,y,z,e);
                  const double w = cst_coeff ? cst_val_2 : C(2,c,x,y,z,e);
                  QQ0(z,y,x) = Jinv[0]*u + Jinv[3]*v + Jinv[6]*w;
                  QQ1(z,y,x) = Jinv[1]*u + Jinv[4]*v + Jinv[7]*w;
                  QQ2(z,y,x) = Jinv[2]*u + Jinv[5]*v + Jinv[8]*w;

                  const double dJ = DetJ(x,y,z,e);
                  const double weight = W(x,y,z);
                  QQ0(z,y,x) *= weight * dJ;
                  QQ1(z,y,x) *= weight * dJ;
                  QQ2(z,y,x) *= weight * dJ;
               }
            }
         }
         MFEM_SYNC_THREAD;
         kernels::internal::GradZt(d,q,Bt,Gt,QQ0,QQ1,QQ2,QD0,QD1,QD2);
         kernels::internal::GradYt(d,q,Bt,Gt,QD0,QD1,QD2,DD0,DD1,DD2);
         kernels::internal::GradXt(d,q,Bt,Gt,DD0,DD1,DD2,Y,c,e);
      }
   });
}

static void LaunchDeviceKernel(const FiniteElementSpace &fes,
                               const IntegrationRule *ir,
                               const Array<int> &markers,
                               const Vector &coeff,
                               Vector &y)
{
   kernel_t ker = nullptr;
   Mesh *mesh = fes.GetMesh();
   const int dim = mesh->Dimension();
   const FiniteElement &el = *fes.GetFE(0);
   const MemoryType mt = Device::GetDeviceMemoryType();
   const DofToQuad &maps = el.GetDofToQuad(*ir, DofToQuad::TENSOR);
   constexpr int flags =
      GeometricFactors::JACOBIANS | GeometricFactors::DETERMINANTS;
   const GeometricFactors *geom = mesh->GetGeometricFactors(*ir, flags, mt);

   const int d = maps.ndof, q = maps.nqpt;

   if (dim==2) { ker=VectorDomainLFGradIntegratorAssemble2D; }
   if (dim==3) { ker=VectorDomainLFGradIntegratorAssemble3D; }

   if (dim==2)
   {
      if (d==2 && q==2) { ker=VectorDomainLFGradIntegratorAssemble2D<2,2>; }
      if (d==3 && q==3) { ker=VectorDomainLFGradIntegratorAssemble2D<3,3>; }
      if (d==4 && q==4) { ker=VectorDomainLFGradIntegratorAssemble2D<4,4>; }
      if (d==5 && q==5) { ker=VectorDomainLFGradIntegratorAssemble2D<5,5>; }
      if (d==2 && q==3) { ker=VectorDomainLFGradIntegratorAssemble2D<2,3>; }
      if (d==3 && q==4) { ker=VectorDomainLFGradIntegratorAssemble2D<3,4>; }
      if (d==4 && q==5) { ker=VectorDomainLFGradIntegratorAssemble2D<4,5>; }
      if (d==5 && q==6) { ker=VectorDomainLFGradIntegratorAssemble2D<5,6>; }
   }

   if (dim==3)
   {
      if (d==2 && q==2) { ker=VectorDomainLFGradIntegratorAssemble3D<2,2>; }
      if (d==3 && q==3) { ker=VectorDomainLFGradIntegratorAssemble3D<3,3>; }
      if (d==4 && q==4) { ker=VectorDomainLFGradIntegratorAssemble3D<4,4>; }
      if (d==5 && q==5) { ker=VectorDomainLFGradIntegratorAssemble3D<5,5>; }
      if (d==2 && q==3) { ker=VectorDomainLFGradIntegratorAssemble3D<2,3>; }
      if (d==3 && q==4) { ker=VectorDomainLFGradIntegratorAssemble3D<3,4>; }
      if (d==4 && q==5) { ker=VectorDomainLFGradIntegratorAssemble3D<4,5>; }
      if (d==5 && q==6) { ker=VectorDomainLFGradIntegratorAssemble3D<5,6>; }
   }

   MFEM_VERIFY(ker, "No kernel ndof " << d << " nqpt " << q);

   const int vdim = fes.GetVDim();
   const int ne = fes.GetMesh()->GetNE();
   const int *M = markers.Read();
   const double *B = maps.B.Read();
   const double *G = maps.G.Read();
   const double *J = geom->J.Read();
   const double *detJ = geom->detJ.Read();
   const double *W = ir->GetWeights().Read();
   double *Y = y.ReadWrite();
   ker(vdim, ne, d, q, M, B, G, J, detJ, W, coeff, Y);
}

void DomainLFGradIntegrator::AssembleDevice(const FiniteElementSpace &fes,
                                            const Array<int> &markers,
                                            Vector &b)
{

   const FiniteElement &fe = *fes.GetFE(0);
   const int qorder = 2 * fe.GetOrder();
   const Geometry::Type gtype = fe.GetGeomType();
   const IntegrationRule *ir = IntRule ? IntRule : &IntRules.Get(gtype, qorder);
   const int nq = ir->GetNPoints(), ne = fes.GetMesh()->GetNE();

   if (VectorConstantCoefficient *vcQ =
          dynamic_cast<VectorConstantCoefficient*>(&Q))
   {
      Qvec = vcQ->GetVec();
   }
   else if (VectorQuadratureFunctionCoefficient *vqfQ =
               dynamic_cast<VectorQuadratureFunctionCoefficient*>(&Q))
   {
      const QuadratureFunction &qfun = vqfQ->GetQuadFunction();
      MFEM_VERIFY(qfun.Size() == ne*nq,
                  "Incompatible QuadratureFunction dimension \n");
      MFEM_VERIFY(ir == &qfun.GetSpace()->GetElementIntRule(0),
                  "IntegrationRule used within integrator and in"
                  " QuadratureFunction appear to be different.\n");
      qfun.Read();
      Qvec.MakeRef(const_cast<QuadratureFunction&>(qfun),0);
   }
   else
   {
      const int qvdim = Q.GetVDim();
      Vector qvec(qvdim);
      Qvec.SetSize(qvdim * nq * ne);
      auto C = Reshape(Qvec.HostWrite(), qvdim, nq, ne);
      for (int e = 0; e < ne; ++e)
      {
         ElementTransformation& T = *fes.GetElementTransformation(e);
         for (int q = 0; q < nq; ++q)
         {
            Q.Eval(qvec, T, ir->IntPoint(q));
            for (int c=0; c < qvdim; ++c)
            {
               C(c,q,e) = qvec[c];
            }
         }
      }
   }
   LaunchDeviceKernel(fes, ir, markers, Qvec, b);
}

void VectorDomainLFGradIntegrator::AssembleDevice(const FiniteElementSpace &fes,
                                                  const Array<int> &markers,
                                                  Vector &b)
{
   const int vdim = fes.GetVDim();
   const FiniteElement &fe = *fes.GetFE(0);
   const int qorder = 2 * fe.GetOrder();
   const Geometry::Type gtype = fe.GetGeomType();
   const IntegrationRule *ir = IntRule ? IntRule : &IntRules.Get(gtype, qorder);
   const int nq = ir->GetNPoints(), ne = fes.GetMesh()->GetNE(),
             ns = fes.GetMesh()->SpaceDimension();

   if (VectorConstantCoefficient *vcQ =
          dynamic_cast<VectorConstantCoefficient*>(&Q))
   {
      Qvec = vcQ->GetVec();
   }
   else if (QuadratureFunctionCoefficient *qfQ =
               dynamic_cast<QuadratureFunctionCoefficient*>(&Q))
   {
      const QuadratureFunction &qfun = qfQ->GetQuadFunction();
      MFEM_VERIFY(qfun.Size() == ne*nq,
                  "Incompatible QuadratureFunction dimension \n");
      MFEM_VERIFY(ir == &qfun.GetSpace()->GetElementIntRule(0),
                  "IntegrationRule used within integrator and in"
                  " QuadratureFunction appear to be different.\n");
      qfun.Read();
      Qvec.MakeRef(const_cast<QuadratureFunction&>(qfun),0);
   }
   else if (VectorQuadratureFunctionCoefficient* vqfQ =
               dynamic_cast<VectorQuadratureFunctionCoefficient*>(&Q))
   {
      const QuadratureFunction &qFun = vqfQ->GetQuadFunction();
      MFEM_VERIFY(qFun.Size() == vdim * ns * nq * ne,
                  "Incompatible QuadratureFunction dimension \n");
      MFEM_VERIFY(ir == &qFun.GetSpace()->GetElementIntRule(0),
                  "IntegrationRule used within integrator and in"
                  " QuadratureFunction appear to be different");
      qFun.Read();
      Qvec.MakeRef(const_cast<QuadratureFunction &>(qFun),0);
   }
   else
   {
      Vector qvec(vdim);
      Qvec.SetSize(vdim * nq * ne);
      auto C = Reshape(Qvec.HostWrite(), vdim, nq, ne);
      for (int e = 0; e < ne; ++e)
      {
         ElementTransformation &Tr = *fes.GetElementTransformation(e);
         for (int q = 0; q < nq; ++q)
         {
            Q.Eval(qvec, Tr, ir->IntPoint(q));
            for (int c = 0; c<vdim; ++c) { C(c,q,e) = qvec[c]; }
         }
      }
   }
   LaunchDeviceKernel(fes, ir, markers, Qvec, b);
}

} // namespace mfem
