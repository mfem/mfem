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

#include "fem.hpp"
#include "../fem/kernels.hpp"
#include "../general/forall.hpp"

namespace mfem
{

template<int T_D1D = 0, int T_Q1D = 0> static
void DLFEvalAssemble2D(const int vdim, const int ne, const int d, const int q,
                       const int map_type, const int *markers, const double *b,
                       const double *detj, const double *weights,
                       const Vector &coeff, double *y)
{
   const auto F = coeff.Read();
   const auto M = Reshape(markers, ne);
   const auto B = Reshape(b, q, d);
   const auto DETJ = Reshape(detj, q, q, ne);
   const auto W = Reshape(weights, q, q);
   const bool cst = coeff.Size() == vdim;
   const auto C = cst ? Reshape(F,vdim,1,1,1) : Reshape(F,vdim,q,q,ne);
   auto Y = Reshape(y, d,d, vdim, ne);

   mfem::forall_2D(ne, q, q, [=] MFEM_HOST_DEVICE (int e)
   {
      if (M(e) == 0) { return; } // ignore

      constexpr int Q = T_Q1D ? T_Q1D : MAX_Q1D;
      constexpr int D = T_D1D ? T_D1D : MAX_D1D;

      MFEM_SHARED double sBt[Q*D];
      MFEM_SHARED double sQQ[Q*Q];
      MFEM_SHARED double sQD[Q*D];

      const DeviceMatrix Bt(sBt, d, q);
      kernels::internal::LoadB<D,Q>(d, q, B, sBt);

      const DeviceMatrix QQ(sQQ, q, q);
      const DeviceMatrix QD(sQD, q, d);

      for (int c = 0; c < vdim; ++c)
      {
         const double cst_val = C(c,0,0,0);
         MFEM_FOREACH_THREAD(x,x,q)
         {
            MFEM_FOREACH_THREAD(y,y,q)
            {
               const double detJ = (map_type == FiniteElement::VALUE) ? DETJ(x,y,e) : 1.0;
               const double coeff_val = cst ? cst_val : C(c,x,y,e);
               QQ(y,x) = W(x,y) * coeff_val * detJ;
            }
         }
         MFEM_SYNC_THREAD;
         MFEM_FOREACH_THREAD(qy,y,q)
         {
            MFEM_FOREACH_THREAD(dx,x,d)
            {
               double u = 0.0;
               for (int qx = 0; qx < q; ++qx) { u += QQ(qy,qx) * Bt(dx,qx); }
               QD(qy,dx) = u;
            }
         }
         MFEM_SYNC_THREAD;
         MFEM_FOREACH_THREAD(dy,y,d)
         {
            MFEM_FOREACH_THREAD(dx,x,d)
            {
               double u = 0.0;
               for (int qy = 0; qy < q; ++qy) { u += QD(qy,dx) * Bt(dy,qy); }
               Y(dx,dy,c,e) += u;
            }
         }
         MFEM_SYNC_THREAD;
      }
   });
}

template<int T_D1D = 0, int T_Q1D = 0> static
void DLFEvalAssemble3D(const int vdim, const int ne, const int d, const int q,
                       const int map_type, const int *markers, const double *b,
                       const double *detj, const double *weights,
                       const Vector &coeff, double *y)
{
   const auto F = coeff.Read();
   const auto M = Reshape(markers, ne);
   const auto B = Reshape(b, q,d);
   const auto DETJ = Reshape(detj, q, q, q, ne);
   const auto W = Reshape(weights, q,q,q);
   const bool cst_coeff = coeff.Size() == vdim;
   const auto C = cst_coeff ? Reshape(F,vdim,1,1,1,1):Reshape(F,vdim,q,q,q,ne);

   auto Y = Reshape(y, d,d,d, vdim, ne);

   mfem::forall_2D(ne, q, q, [=] MFEM_HOST_DEVICE (int e)
   {
      if (M(e) == 0) { return; } // ignore

      constexpr int Q = T_Q1D ? T_Q1D : MAX_Q1D;
      constexpr int D = T_D1D ? T_D1D : MAX_D1D;
      constexpr int MQD = (Q >= D) ? Q : D;

      double u[D];

      MFEM_SHARED double sBt[Q*D];
      const DeviceMatrix Bt(sBt, d,q);
      kernels::internal::LoadB<D,Q>(d,q,B,sBt);

      MFEM_SHARED double sQQQ[MQD*MQD*MQD];
      const DeviceCube QQQ(sQQQ, MQD, MQD, MQD);

      for (int c = 0; c < vdim; ++c)
      {
         const double cst_val = C(c,0,0,0,0);
         MFEM_FOREACH_THREAD(x,x,q)
         {
            MFEM_FOREACH_THREAD(y,y,q)
            {
               for (int z = 0; z < q; ++z)
               {
                  const double detJ = (map_type == FiniteElement::VALUE) ? DETJ(x,y,z,e) : 1.0;
                  const double coeff_val = cst_coeff ? cst_val : C(c,x,y,z,e);
                  QQQ(z,y,x) = W(x,y,z) * coeff_val * detJ;
               }
            }
         }
         MFEM_SYNC_THREAD;
         MFEM_FOREACH_THREAD(qx,x,q)
         {
            MFEM_FOREACH_THREAD(qy,y,q)
            {
               for (int dz = 0; dz < d; ++dz) { u[dz] = 0.0; }
               for (int qz = 0; qz < q; ++qz)
               {
                  const double ZYX = QQQ(qz,qy,qx);
                  for (int dz = 0; dz < d; ++dz) { u[dz] += ZYX * Bt(dz,qz); }
               }
               for (int dz = 0; dz < d; ++dz) { QQQ(dz,qy,qx) = u[dz]; }
            }
         }
         MFEM_SYNC_THREAD;
         MFEM_FOREACH_THREAD(dz,y,d)
         {
            MFEM_FOREACH_THREAD(qx,x,q)
            {
               for (int dy = 0; dy < d; ++dy) { u[dy] = 0.0; }
               for (int qy = 0; qy < q; ++qy)
               {
                  const double zYX = QQQ(dz,qy,qx);
                  for (int dy = 0; dy < d; ++dy) { u[dy] += zYX * Bt(dy,qy); }
               }
               for (int dy = 0; dy < d; ++dy) { QQQ(dz,dy,qx) = u[dy]; }
            }
         }
         MFEM_SYNC_THREAD;
         MFEM_FOREACH_THREAD(dz,y,d)
         {
            MFEM_FOREACH_THREAD(dy,x,d)
            {
               for (int dx = 0; dx < d; ++dx) { u[dx] = 0.0; }
               for (int qx = 0; qx < q; ++qx)
               {
                  const double zyX = QQQ(dz,dy,qx);
                  for (int dx = 0; dx < d; ++dx) { u[dx] += zyX * Bt(dx,qx); }
               }
               for (int dx = 0; dx < d; ++dx) { Y(dx,dy,dz,c,e) += u[dx]; }
            }
         }
         MFEM_SYNC_THREAD;
      }
   });
}

static void DLFEvalAssemble(const FiniteElementSpace &fes,
                            const IntegrationRule *ir,
                            const Array<int> &markers,
                            const Vector &coeff,
                            Vector &y)
{
   Mesh *mesh = fes.GetMesh();
   const int dim = mesh->Dimension();
   const FiniteElement &el = *fes.GetFE(0);
   const MemoryType mt = Device::GetDeviceMemoryType();
   const DofToQuad &maps = el.GetDofToQuad(*ir, DofToQuad::TENSOR);
   const int d = maps.ndof, q = maps.nqpt;
   constexpr int flags = GeometricFactors::DETERMINANTS;
   const GeometricFactors *geom = mesh->GetGeometricFactors(*ir, flags, mt);
   const int map_type = fes.GetFE(0)->GetMapType();
   decltype(&DLFEvalAssemble2D<>) ker =
      dim == 2 ? DLFEvalAssemble2D<> : DLFEvalAssemble3D<>;

   if (dim==2)
   {
      if (d==1 && q==1) { ker=DLFEvalAssemble2D<1,1>; }
      if (d==2 && q==2) { ker=DLFEvalAssemble2D<2,2>; }
      if (d==3 && q==3) { ker=DLFEvalAssemble2D<3,3>; }
      if (d==4 && q==4) { ker=DLFEvalAssemble2D<4,4>; }
      if (d==5 && q==5) { ker=DLFEvalAssemble2D<5,5>; }
      if (d==2 && q==3) { ker=DLFEvalAssemble2D<2,3>; }
      if (d==3 && q==4) { ker=DLFEvalAssemble2D<3,4>; }
      if (d==4 && q==5) { ker=DLFEvalAssemble2D<4,5>; }
      if (d==5 && q==6) { ker=DLFEvalAssemble2D<5,6>; }
   }

   if (dim==3)
   {
      if (d==1 && q==1) { ker=DLFEvalAssemble3D<1,1>; }
      if (d==2 && q==2) { ker=DLFEvalAssemble3D<2,2>; }
      if (d==3 && q==3) { ker=DLFEvalAssemble3D<3,3>; }
      if (d==4 && q==4) { ker=DLFEvalAssemble3D<4,4>; }
      if (d==5 && q==5) { ker=DLFEvalAssemble3D<5,5>; }
      if (d==2 && q==3) { ker=DLFEvalAssemble3D<2,3>; }
      if (d==3 && q==4) { ker=DLFEvalAssemble3D<3,4>; }
      if (d==4 && q==5) { ker=DLFEvalAssemble3D<4,5>; }
      if (d==5 && q==6) { ker=DLFEvalAssemble3D<5,6>; }
   }

   MFEM_VERIFY(ker, "No kernel ndof " << d << " nqpt " << q);

   const int vdim = fes.GetVDim();
   const int ne = fes.GetMesh()->GetNE();
   const int *M = markers.Read();
   const double *B = maps.B.Read();
   const double *detJ = geom->detJ.Read();
   const double *W = ir->GetWeights().Read();
   double *Y = y.ReadWrite();
   ker(vdim, ne, d, q, map_type, M, B, detJ, W, coeff, Y);
}

void DomainLFIntegrator::AssembleDevice(const FiniteElementSpace &fes,
                                        const Array<int> &markers,
                                        Vector &b)
{
   const FiniteElement &fe = *fes.GetFE(0);
   const int qorder = oa * fe.GetOrder() + ob;
   const Geometry::Type gtype = fe.GetGeomType();
   const IntegrationRule *ir = IntRule ? IntRule : &IntRules.Get(gtype, qorder);

   QuadratureSpace qs(*fes.GetMesh(), *ir);
   CoefficientVector coeff(Q, qs, CoefficientStorage::COMPRESSED);
   DLFEvalAssemble(fes, ir, markers, coeff, b);
}

void VectorDomainLFIntegrator::AssembleDevice(const FiniteElementSpace &fes,
                                              const Array<int> &markers,
                                              Vector &b)
{
   const FiniteElement &fe = *fes.GetFE(0);
   const int qorder = 2 * fe.GetOrder();
   const Geometry::Type gtype = fe.GetGeomType();
   const IntegrationRule *ir = IntRule ? IntRule : &IntRules.Get(gtype, qorder);

   QuadratureSpace qs(*fes.GetMesh(), *ir);
   CoefficientVector coeff(Q, qs, CoefficientStorage::COMPRESSED);
   DLFEvalAssemble(fes, ir, markers, coeff, b);
}

} // namespace mfem
