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

#include "../../general/forall.hpp"
#include "../../fem/kernels.hpp"
#include "../fem.hpp"

namespace mfem
{

template<int T_D1D = 0, int T_Q1D = 0> static
void DLFGradAssemble2D(const int vdim, const int ne, const int d, const int q,
                       const int *markers, const real_t *b, const real_t *g,
                       const real_t *jacobians,
                       const real_t *weights, const Vector &coeff, real_t *y)
{
   const auto F = coeff.Read();
   const auto M = Reshape(markers, ne);
   const auto B = Reshape(b, q, d);
   const auto G = Reshape(g, q, d);
   const auto J = Reshape(jacobians, q, q, 2,2, ne);
   const auto W = Reshape(weights, q, q);
   const bool cst = coeff.Size() == vdim*2;
   const auto C = cst ? Reshape(F,2,vdim,1,1,1) : Reshape(F,2,vdim,q,q,ne);
   auto Y = Reshape(y, d,d, vdim, ne);

   mfem::forall_2D(ne, q, q, [=] MFEM_HOST_DEVICE (int e)
   {
      if (M(e) == 0) { return; } // ignore

      constexpr int Q = T_Q1D ? T_Q1D : DofQuadLimits::MAX_Q1D;
      constexpr int D = T_D1D ? T_D1D : DofQuadLimits::MAX_D1D;

      MFEM_SHARED real_t sBGt[2][Q*D];
      MFEM_SHARED real_t sQQ[2][Q*Q];
      MFEM_SHARED real_t sDQ[2][D*Q];

      const DeviceMatrix Bt(sBGt[0], q, d);
      const DeviceMatrix Gt(sBGt[1], q, d);
      kernels::internal::LoadBGt<D,Q>(d, q, B, G, sBGt);

      const DeviceMatrix QQ0(sQQ[0], q, q);
      const DeviceMatrix QQ1(sQQ[1], q, q);

      const DeviceMatrix DQ0(sDQ[0], d, q);
      const DeviceMatrix DQ1(sDQ[1], d, q);

      for (int c = 0; c < vdim; ++c)
      {
         const real_t cst_val0 = C(0,c,0,0,0);
         const real_t cst_val1 = C(1,c,0,0,0);

         MFEM_FOREACH_THREAD(x,x,q)
         {
            MFEM_FOREACH_THREAD(y,y,q)
            {
               const real_t w = W(x,y);
               const real_t J11 = J(x,y,0,0,e);
               const real_t J21 = J(x,y,1,0,e);
               const real_t J12 = J(x,y,0,1,e);
               const real_t J22 = J(x,y,1,1,e);
               const real_t u = cst ? cst_val0 : C(0,c,x,y,e);
               const real_t v = cst ? cst_val1 : C(1,c,x,y,e);
               // QQ = w * det(J) * J^{-1} . C = w * adj(J) . { u, v }
               QQ0(y,x) = w * (J22*u - J12*v);
               QQ1(y,x) = w * (J11*v - J21*u);
            }
         }
         MFEM_SYNC_THREAD;
         MFEM_FOREACH_THREAD(qx,x,q)
         {
            MFEM_FOREACH_THREAD(dy,y,d)
            {
               real_t u = 0.0, v = 0.0;
               for (int qy = 0; qy < q; ++qy)
               {
                  u += QQ0(qy,qx) * Bt(qy,dy);
                  v += QQ1(qy,qx) * Gt(qy,dy);
               }
               DQ0(dy,qx) = u;
               DQ1(dy,qx) = v;
            }
         }
         MFEM_SYNC_THREAD;
         MFEM_FOREACH_THREAD(dx,x,d)
         {
            MFEM_FOREACH_THREAD(dy,y,d)
            {
               real_t u = 0.0, v = 0.0;
               for (int qx = 0; qx < q; ++qx)
               {
                  u += DQ0(dy,qx) * Gt(qx,dx);
                  v += DQ1(dy,qx) * Bt(qx,dx);
               }
               Y(dx,dy,c,e) += u + v;
            }
         }
         MFEM_SYNC_THREAD;
      }
   });
}

template<int T_D1D = 0, int T_Q1D = 0> static
void DLFGradAssemble3D(const int vdim, const int ne, const int d, const int q,
                       const int *markers, const real_t *b, const real_t *g,
                       const real_t *jacobians,
                       const real_t *weights, const Vector &coeff,
                       real_t *output)
{
   const auto F = coeff.Read();
   const auto M = Reshape(markers, ne);
   const auto B = Reshape(b, q,d);
   const auto G = Reshape(g, q,d);
   const auto J = Reshape(jacobians, q,q,q, 3,3, ne);
   const auto W = Reshape(weights, q,q,q);
   const bool cst = coeff.Size() == vdim*3;
   const auto C = cst ? Reshape(F,3,vdim,1,1,1,1) : Reshape(F,3,vdim,q,q,q,ne);

   auto Y = Reshape(output, d,d,d, vdim, ne);

   mfem::forall_2D(ne, q, q, [=] MFEM_HOST_DEVICE (int e)
   {
      if (M(e) == 0) { return; } // ignore

      constexpr int Q = T_Q1D ? T_Q1D : DofQuadLimits::MAX_Q1D;
      constexpr int D = T_D1D ? T_D1D : DofQuadLimits::MAX_D1D;
      constexpr int MQD = (Q >= D) ? Q : D;

      MFEM_SHARED real_t sBGt[2][Q*D];
      const DeviceMatrix Bt(sBGt[0], q,d), Gt(sBGt[1], q,d);

      MFEM_SHARED real_t sQQQ[MQD*MQD*MQD];
      const DeviceCube QQQ(sQQQ, MQD,MQD,MQD);

      kernels::internal::LoadBGt<D,Q>(d,q,B,G,sBGt);

      for (int c = 0; c < vdim; ++c)
      {
         const real_t cst_val_0 = C(0,c,0,0,0,0);
         const real_t cst_val_1 = C(1,c,0,0,0,0);
         const real_t cst_val_2 = C(2,c,0,0,0,0);

         for (int k = 0; k < 3; ++k)
         {
            for (int z = 0; z < q; ++z)
            {
               MFEM_FOREACH_THREAD(y,y,q)
               {
                  MFEM_FOREACH_THREAD(x,x,q)
                  {
                     const real_t J11 = J(x,y,z,0,0,e);
                     const real_t J21 = J(x,y,z,1,0,e);
                     const real_t J31 = J(x,y,z,2,0,e);
                     const real_t J12 = J(x,y,z,0,1,e);
                     const real_t J22 = J(x,y,z,1,1,e);
                     const real_t J32 = J(x,y,z,2,1,e);
                     const real_t J13 = J(x,y,z,0,2,e);
                     const real_t J23 = J(x,y,z,1,2,e);
                     const real_t J33 = J(x,y,z,2,2,e);

                     const real_t u = cst ? cst_val_0 : C(0,c,x,y,z,e);
                     const real_t v = cst ? cst_val_1 : C(1,c,x,y,z,e);
                     const real_t w = cst ? cst_val_2 : C(2,c,x,y,z,e);

                     if (k == 0)
                     {
                        const real_t A11 = (J22 * J33) - (J23 * J32);
                        const real_t A12 = (J32 * J13) - (J12 * J33);
                        const real_t A13 = (J12 * J23) - (J22 * J13);
                        QQQ(z,y,x) = A11*u + A12*v + A13*w;

                     }

                     if (k == 1)
                     {
                        const real_t A21 = (J31 * J23) - (J21 * J33);
                        const real_t A22 = (J11 * J33) - (J13 * J31);
                        const real_t A23 = (J21 * J13) - (J11 * J23);
                        QQQ(z,y,x) = A21*u + A22*v + A23*w;
                     }

                     if (k == 2)
                     {
                        const real_t A31 = (J21 * J32) - (J31 * J22);
                        const real_t A32 = (J31 * J12) - (J11 * J32);
                        const real_t A33 = (J11 * J22) - (J12 * J21);
                        QQQ(z,y,x) = A31*u + A32*v + A33*w;
                     }

                     QQQ(z,y,x) *= W(x,y,z);
                  }
               }
               MFEM_SYNC_THREAD;
            }
            MFEM_FOREACH_THREAD(qz,x,q)
            {
               MFEM_FOREACH_THREAD(qy,y,q)
               {
                  real_t r_u[Q];
                  for (int qx = 0; qx < q; ++qx) { r_u[qx] = QQQ(qz,qy,qx); }
                  for (int dx = 0; dx < d; ++dx)
                  {
                     real_t u = 0.0;
                     for (int qx = 0; qx < q; ++qx)
                     {
                        u += (k == 0 ? Gt(qx,dx) : Bt(qx,dx)) * r_u[qx];
                     }
                     QQQ(qz,qy,dx) = u;
                  }
               }
            }
            MFEM_SYNC_THREAD;
            MFEM_FOREACH_THREAD(qz,y,q)
            {
               MFEM_FOREACH_THREAD(dx,x,d)
               {
                  real_t r_u[Q];
                  for (int qy = 0; qy < q; ++qy) { r_u[qy] = QQQ(qz,qy,dx); }
                  for (int dy = 0; dy < d; ++dy)
                  {
                     real_t u = 0.0;
                     for (int qy = 0; qy < q; ++qy)
                     {
                        u += (k == 1 ? Gt(qy,dy) : Bt(qy,dy)) * r_u[qy];
                     }
                     QQQ(qz,dy,dx) = u;
                  }
               }
            }
            MFEM_SYNC_THREAD;
            MFEM_FOREACH_THREAD(dy,y,d)
            {
               MFEM_FOREACH_THREAD(dx,x,d)
               {
                  real_t r_u[Q];
                  for (int qz = 0; qz < q; ++qz) { r_u[qz] = QQQ(qz,dy,dx); }
                  for (int dz = 0; dz < d; ++dz)
                  {
                     real_t u = 0.0;
                     for (int qz = 0; qz < q; ++qz)
                     {
                        u += (k == 2 ? Gt(qz,dz) : Bt(qz,dz)) * r_u[qz];
                     }
                     Y(dx,dy,dz,c,e) += u;
                  }
               }
            }
            MFEM_SYNC_THREAD;
         } // dim
      } // vdim
   });
}

static void DLFGradAssemble(const FiniteElementSpace &fes,
                            const IntegrationRule *ir,
                            const Array<int> &markers,
                            const Vector &coeff,
                            Vector &y)
{
   Mesh *mesh = fes.GetMesh();
   const int dim = mesh->Dimension();
   const FiniteElement &el = *fes.GetTypicalFE();
   const MemoryType mt = Device::GetDeviceMemoryType();
   const DofToQuad &maps = el.GetDofToQuad(*ir, DofToQuad::TENSOR);
   const int d = maps.ndof, q = maps.nqpt;
   constexpr int flags = GeometricFactors::JACOBIANS;
   const GeometricFactors *geom = mesh->GetGeometricFactors(*ir, flags, mt);
   decltype(&DLFGradAssemble2D<>) ker =
      dim == 2 ? DLFGradAssemble2D<> :  DLFGradAssemble3D<>;

   if (dim==2)
   {
      if (d==1 && q==1) { ker=DLFGradAssemble2D<1,1>; }
      if (d==2 && q==2) { ker=DLFGradAssemble2D<2,2>; }
      if (d==3 && q==3) { ker=DLFGradAssemble2D<3,3>; }
      if (d==4 && q==4) { ker=DLFGradAssemble2D<4,4>; }
      if (d==5 && q==5) { ker=DLFGradAssemble2D<5,5>; }
      if (d==2 && q==3) { ker=DLFGradAssemble2D<2,3>; }
      if (d==3 && q==4) { ker=DLFGradAssemble2D<3,4>; }
      if (d==4 && q==5) { ker=DLFGradAssemble2D<4,5>; }
      if (d==5 && q==6) { ker=DLFGradAssemble2D<5,6>; }
   }

   if (dim==3)
   {
      if (d==1 && q==1) { ker=DLFGradAssemble3D<1,1>; }
      if (d==2 && q==2) { ker=DLFGradAssemble3D<2,2>; }
      if (d==3 && q==3) { ker=DLFGradAssemble3D<3,3>; }
      if (d==4 && q==4) { ker=DLFGradAssemble3D<4,4>; }
      if (d==5 && q==5) { ker=DLFGradAssemble3D<5,5>; }
      if (d==2 && q==3) { ker=DLFGradAssemble3D<2,3>; }
      if (d==3 && q==4) { ker=DLFGradAssemble3D<3,4>; }
      if (d==4 && q==5) { ker=DLFGradAssemble3D<4,5>; }
      if (d==5 && q==6) { ker=DLFGradAssemble3D<5,6>; }
   }

   MFEM_VERIFY(ker, "No kernel ndof " << d << " nqpt " << q);

   const int vdim = fes.GetVDim();
   const int ne = fes.GetMesh()->GetNE();
   const int *M = markers.Read();
   const real_t *B = maps.B.Read();
   const real_t *G = maps.G.Read();
   const real_t *J = geom->J.Read();
   const real_t *W = ir->GetWeights().Read();
   real_t *Y = y.ReadWrite();
   ker(vdim, ne, d, q, M, B, G, J, W, coeff, Y);
}

void DomainLFGradIntegrator::AssembleDevice(const FiniteElementSpace &fes,
                                            const Array<int> &markers,
                                            Vector &b)
{

   const FiniteElement &fe = *fes.GetTypicalFE();
   const int qorder = 2 * fe.GetOrder();
   const Geometry::Type gtype = fe.GetGeomType();
   const IntegrationRule *ir = IntRule ? IntRule : &IntRules.Get(gtype, qorder);

   QuadratureSpace qs(*fes.GetMesh(), *ir);
   CoefficientVector coeff(Q, qs, CoefficientStorage::COMPRESSED);
   DLFGradAssemble(fes, ir, markers, coeff, b);
}

void VectorDomainLFGradIntegrator::AssembleDevice(const FiniteElementSpace &fes,
                                                  const Array<int> &markers,
                                                  Vector &b)
{
   const FiniteElement &fe = *fes.GetTypicalFE();
   const int qorder = 2 * fe.GetOrder();
   const Geometry::Type gtype = fe.GetGeomType();
   const IntegrationRule *ir = IntRule ? IntRule : &IntRules.Get(gtype, qorder);

   QuadratureSpace qs(*fes.GetMesh(), *ir);
   CoefficientVector coeff(Q, qs, CoefficientStorage::COMPRESSED);
   DLFGradAssemble(fes, ir, markers, coeff, b);
}

} // namespace mfem
