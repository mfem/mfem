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

template<int T_D1D = 0, int T_Q1D = 0>
static void HdivDLFAssemble2D(
   const int ne, const int d, const int q, const int *markers, const real_t *bo,
   const real_t *bc, const real_t *j, const real_t *weights,
   const Vector &coeff, real_t *y)
{
   MFEM_VERIFY(T_D1D || d <= DeviceDofQuadLimits::Get().HDIV_MAX_D1D,
               "Problem size too large.");
   MFEM_VERIFY(T_Q1D || q <= DeviceDofQuadLimits::Get().HDIV_MAX_Q1D,
               "Problem size too large.");

   static constexpr int vdim = 2;
   const auto F = coeff.Read();
   const auto M = Reshape(markers, ne);
   const auto BO = Reshape(bo, q, d-1);
   const auto BC = Reshape(bc, q, d);
   const auto J = Reshape(j, q, q, vdim, vdim, ne);
   const auto W = Reshape(weights, q, q);
   const bool cst = coeff.Size() == vdim;
   const auto C = cst ? Reshape(F,vdim,1,1,1) : Reshape(F,vdim,q,q,ne);
   auto Y = Reshape(y, 2*(d-1)*d, ne);

   mfem::forall_3D(ne, q, q, vdim, [=] MFEM_HOST_DEVICE (int e)
   {
      if (M(e) == 0) { return; } // ignore

      constexpr int Q = T_Q1D ? T_Q1D : DofQuadLimits::HDIV_MAX_Q1D;
      constexpr int D = T_D1D ? T_D1D : DofQuadLimits::HDIV_MAX_D1D;

      MFEM_SHARED real_t sBot[Q*D];
      MFEM_SHARED real_t sBct[Q*D];
      MFEM_SHARED real_t sQQ[vdim*Q*Q];
      MFEM_SHARED real_t sQD[vdim*Q*D];

      // Bo and Bc into shared memory
      const DeviceMatrix Bot(sBot, d-1, q);
      kernels::internal::LoadB<D,Q>(d-1, q, BO, sBot);
      const DeviceMatrix Bct(sBct, d, q);
      kernels::internal::LoadB<D,Q>(d, q, BC, sBct);

      const DeviceCube QQ(sQQ, q, q, vdim);
      const DeviceCube QD(sQD, q, d, vdim);

      MFEM_FOREACH_THREAD(vd,z,vdim)
      {
         const real_t cst_val_0 = C(0,0,0,0);
         const real_t cst_val_1 = C(1,0,0,0);
         MFEM_FOREACH_THREAD(y,y,q)
         {
            MFEM_FOREACH_THREAD(x,x,q)
            {
               const real_t J0 = J(x,y,0,vd,e);
               const real_t J1 = J(x,y,1,vd,e);
               const real_t C0 = cst ? cst_val_0 : C(0,x,y,e);
               const real_t C1 = cst ? cst_val_1 : C(1,x,y,e);
               QQ(x,y,vd) = W(x,y)*(J0*C0 + J1*C1);
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(vd,z,vdim)
      {
         const int nx = (vd == 0) ? d : d-1;
         DeviceMatrix Btx = (vd == 0) ? Bct : Bot;
         MFEM_FOREACH_THREAD(qy,y,q)
         {
            MFEM_FOREACH_THREAD(dx,x,nx)
            {
               real_t qd = 0.0;
               for (int qx = 0; qx < q; ++qx)
               {
                  qd += QQ(qx,qy,vd) * Btx(dx,qx);
               }
               QD(dx,qy,vd) = qd;
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(vd,z,vdim)
      {
         const int nx = (vd == 0) ? d : d-1;
         const int ny = (vd == 1) ? d : d-1;
         DeviceMatrix Bty = (vd == 1) ? Bct : Bot;
         DeviceTensor<4> Yxy(Y, nx, ny, vdim, ne);
         MFEM_FOREACH_THREAD(dy,y,ny)
         {
            MFEM_FOREACH_THREAD(dx,x,nx)
            {
               real_t dd = 0.0;
               for (int qy = 0; qy < q; ++qy)
               {
                  dd += QD(dx,qy,vd) * Bty(dy,qy);
               }
               Yxy(dx,dy,vd,e) += dd;
            }
         }
      }
      MFEM_SYNC_THREAD;
   });
}

template<int T_D1D = 0, int T_Q1D = 0>
static void HdivDLFAssemble3D(
   const int ne, const int d, const int q, const int *markers, const real_t *bo,
   const real_t *bc, const real_t *j, const real_t *weights,
   const Vector &coeff, real_t *y)
{
   MFEM_VERIFY(T_D1D || d <= DeviceDofQuadLimits::Get().HDIV_MAX_D1D,
               "Problem size too large.");
   MFEM_VERIFY(T_Q1D || q <= DeviceDofQuadLimits::Get().HDIV_MAX_Q1D,
               "Problem size too large.");

   static constexpr int vdim = 3;
   const auto F = coeff.Read();
   const auto M = Reshape(markers, ne);
   const auto BO = Reshape(bo, q, d-1);
   const auto BC = Reshape(bc, q, d);
   const auto J = Reshape(j, q, q, q, vdim, vdim, ne);
   const auto W = Reshape(weights, q, q, q);
   const bool cst = coeff.Size() == vdim;
   const auto C = cst ? Reshape(F,vdim,1,1,1,1) : Reshape(F,vdim,q,q,q,ne);
   auto Y = Reshape(y, 2*(d-1)*(d-1)*d, ne);

   mfem::forall_3D(ne, q, q, vdim, [=] MFEM_HOST_DEVICE (int e)
   {
      if (M(e) == 0) { return; } // ignore

      constexpr int Q = T_Q1D ? T_Q1D : DofQuadLimits::HDIV_MAX_Q1D;
      constexpr int D = T_D1D ? T_D1D : DofQuadLimits::HDIV_MAX_D1D;

      MFEM_SHARED real_t sBot[Q*D];
      MFEM_SHARED real_t sBct[Q*D];

      // Bo and Bc into shared memory
      const DeviceMatrix Bot(sBot, d-1, q);
      kernels::internal::LoadB<D,Q>(d-1, q, BO, sBot);
      const DeviceMatrix Bct(sBct, d, q);
      kernels::internal::LoadB<D,Q>(d, q, BC, sBct);

      MFEM_SHARED real_t sm0[vdim*Q*Q*Q];
      MFEM_SHARED real_t sm1[vdim*Q*Q*Q];
      DeviceTensor<4> QQQ(sm1, q, q, q, vdim);
      DeviceTensor<4> DQQ(sm0, d, q, q, vdim);
      DeviceTensor<4> DDQ(sm1, d, d, q, vdim);

      MFEM_FOREACH_THREAD(vd,z,vdim)
      {
         const real_t cst_val_0 = C(0,0,0,0,0);
         const real_t cst_val_1 = C(1,0,0,0,0);
         const real_t cst_val_2 = C(2,0,0,0,0);
         MFEM_FOREACH_THREAD(y,y,q)
         {
            MFEM_FOREACH_THREAD(x,x,q)
            {
               for (int z = 0; z < q; ++z)
               {
                  const real_t J0 = J(x,y,z,0,vd,e);
                  const real_t J1 = J(x,y,z,1,vd,e);
                  const real_t J2 = J(x,y,z,2,vd,e);
                  const real_t C0 = cst ? cst_val_0 : C(0,x,y,z,e);
                  const real_t C1 = cst ? cst_val_1 : C(1,x,y,z,e);
                  const real_t C2 = cst ? cst_val_2 : C(2,x,y,z,e);
                  QQQ(x,y,z,vd) = W(x,y,z)*(J0*C0 + J1*C1 + J2*C2);
               }
            }
         }
      }
      MFEM_SYNC_THREAD;
      // Apply Bt operator
      MFEM_FOREACH_THREAD(vd,z,vdim)
      {
         const int nx = (vd == 0) ? d : d-1;
         DeviceMatrix Btx = (vd == 0) ? Bct : Bot;
         MFEM_FOREACH_THREAD(qy,y,q)
         {
            MFEM_FOREACH_THREAD(dx,x,nx)
            {
               real_t u[Q];
               MFEM_UNROLL(Q)
               for (int qz = 0; qz < q; ++qz) { u[qz] = 0.0; }
               MFEM_UNROLL(Q)
               for (int qx = 0; qx < q; ++qx)
               {
                  MFEM_UNROLL(Q)
                  for (int qz = 0; qz < q; ++qz)
                  {
                     u[qz] += QQQ(qx,qy,qz,vd) * Btx(dx,qx);
                  }
               }
               MFEM_UNROLL(Q)
               for (int qz = 0; qz < q; ++qz) { DQQ(dx,qy,qz,vd) = u[qz]; }
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(vd,z,vdim)
      {
         const int nx = (vd == 0) ? d : d-1;
         const int ny = (vd == 1) ? d : d-1;
         DeviceMatrix Bty = (vd == 1) ? Bct : Bot;
         MFEM_FOREACH_THREAD(dy,y,ny)
         {
            MFEM_FOREACH_THREAD(dx,x,nx)
            {
               real_t u[Q];
               MFEM_UNROLL(Q)
               for (int qz = 0; qz < q; ++qz) { u[qz] = 0.0; }
               MFEM_UNROLL(Q)
               for (int qy = 0; qy < q; ++qy)
               {
                  MFEM_UNROLL(Q)
                  for (int qz = 0; qz < q; ++qz)
                  {
                     u[qz] += DQQ(dx,qy,qz,vd) * Bty(dy,qy);
                  }
               }
               MFEM_UNROLL(Q)
               for (int qz = 0; qz < q; ++qz) { DDQ(dx,dy,qz,vd) = u[qz]; }
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(vd,z,vdim)
      {
         const int nx = (vd == 0) ? d : d-1;
         const int ny = (vd == 1) ? d : d-1;
         const int nz = (vd == 2) ? d : d-1;
         DeviceTensor<5> Yxyz(Y, nx, ny, nz, vdim, ne);
         DeviceMatrix Btz = (vd == 2) ? Bct : Bot;
         MFEM_FOREACH_THREAD(dy,y,ny)
         {
            MFEM_FOREACH_THREAD(dx,x,nx)
            {
               real_t u[D];
               MFEM_UNROLL(D)
               for (int dz = 0; dz < nz; ++dz) { u[dz] = 0.0; }
               MFEM_UNROLL(Q)
               for (int qz = 0; qz < q; ++qz)
               {
                  MFEM_UNROLL(D)
                  for (int dz = 0; dz < nz; ++dz)
                  {
                     u[dz] += DDQ(dx,dy,qz,vd) * Btz(dz,qz);
                  }
               }
               MFEM_UNROLL(D)
               for (int dz = 0; dz < nz; ++dz) { Yxyz(dx,dy,dz,vd,e) += u[dz]; }
            }
         }
      }
      MFEM_SYNC_THREAD;
   });
}

static void HdivDLFAssemble(const FiniteElementSpace &fes,
                            const IntegrationRule *ir,
                            const Array<int> &markers,
                            const Vector &coeff,
                            Vector &y)
{
   Mesh &mesh = *fes.GetMesh();
   const int dim = mesh.Dimension();
   const FiniteElement *el = fes.GetTypicalFE();
   const auto *vel = dynamic_cast<const VectorTensorFiniteElement *>(el);
   MFEM_VERIFY(vel != nullptr, "Must be VectorTensorFiniteElement");
   const MemoryType mt = Device::GetDeviceMemoryType();
   const DofToQuad &maps_o = vel->GetDofToQuadOpen(*ir, DofToQuad::TENSOR);
   const DofToQuad &maps_c = vel->GetDofToQuad(*ir, DofToQuad::TENSOR);
   const int d = maps_c.ndof, q = maps_c.nqpt;
   constexpr int flags = GeometricFactors::JACOBIANS;
   const GeometricFactors *geom = mesh.GetGeometricFactors(*ir, flags, mt);
   decltype(&HdivDLFAssemble2D<>) ker =
      dim == 2 ? HdivDLFAssemble2D<> : HdivDLFAssemble3D<>;

   if (dim==2)
   {
      if (d==1 && q==1) { ker=HdivDLFAssemble2D<1,1>; }
      if (d==2 && q==2) { ker=HdivDLFAssemble2D<2,2>; }
      if (d==3 && q==3) { ker=HdivDLFAssemble2D<3,3>; }
      if (d==4 && q==4) { ker=HdivDLFAssemble2D<4,4>; }
      if (d==5 && q==5) { ker=HdivDLFAssemble2D<5,5>; }
      if (d==6 && q==6) { ker=HdivDLFAssemble2D<6,6>; }
      if (d==7 && q==7) { ker=HdivDLFAssemble2D<7,7>; }
      if (d==8 && q==8) { ker=HdivDLFAssemble2D<8,8>; }
   }

   if (dim==3)
   {
      if (d==2 && q==2) { ker=HdivDLFAssemble3D<2,2>; }
      if (d==3 && q==3) { ker=HdivDLFAssemble3D<3,3>; }
      if (d==4 && q==4) { ker=HdivDLFAssemble3D<4,4>; }
      if (d==5 && q==5) { ker=HdivDLFAssemble3D<5,5>; }
      if (d==6 && q==6) { ker=HdivDLFAssemble3D<6,6>; }
      if (d==7 && q==7) { ker=HdivDLFAssemble3D<7,7>; }
      if (d==8 && q==8) { ker=HdivDLFAssemble3D<8,8>; }
   }

   MFEM_VERIFY(ker, "No kernel ndof " << d << " nqpt " << q);

   const int ne = mesh.GetNE();
   const int *M = markers.Read();
   const real_t *Bo = maps_o.B.Read();
   const real_t *Bc = maps_c.B.Read();
   const real_t *J = geom->J.Read();
   const real_t *W = ir->GetWeights().Read();
   real_t *Y = y.ReadWrite();
   ker(ne, d, q, M, Bo, Bc, J, W, coeff, Y);
}

void VectorFEDomainLFIntegrator::AssembleDevice(const FiniteElementSpace &fes,
                                                const Array<int> &markers,
                                                Vector &b)
{
   const FiniteElement &fe = *fes.GetTypicalFE();
   const int qorder = 2 * fe.GetOrder();
   const Geometry::Type gtype = fe.GetGeomType();
   const IntegrationRule *ir = IntRule ? IntRule : &IntRules.Get(gtype, qorder);

   QuadratureSpace qs(*fes.GetMesh(), *ir);
   CoefficientVector coeff(QF, qs, CoefficientStorage::COMPRESSED);

   const int fe_type = fe.GetDerivType();
   if (fe_type == FiniteElement::DIV)
   {
      HdivDLFAssemble(fes, ir, markers, coeff, b);
   }
   else
   {
      MFEM_ABORT("Not implemented.");
   }
}

} // namespace mfem
