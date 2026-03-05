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

#include "../../general/forall.hpp"
#include "../../fem/kernels.hpp"
#include "../fem.hpp"

namespace mfem
{

template<int T_D1D = 0, int T_Q1D = 0> static
void BFLFEvalAssemble2D(const int nbe, const int d, const int q,
                        const int *markers, const real_t *b,
                        const real_t *weights, const Vector &coeff, real_t *y)
{
   const auto F = coeff.Read();
   const auto M = Reshape(markers, nbe);
   const auto B = Reshape(b, q, d);
   const auto W = Reshape(weights, q);
   const bool const_coeff = coeff.Size() == 1;
   const auto C = const_coeff ? Reshape(F,1,1) : Reshape(F,q,nbe);
   auto Y = Reshape(y, d, nbe);

   mfem::forall(nbe, [=] MFEM_HOST_DEVICE (int e)
   {
      if (M(e) == 0) { return; } // ignore (in a lambda return acts as continue)

      constexpr int Q = T_Q1D ? T_Q1D : DofQuadLimits::MAX_Q1D;
      real_t QQ[Q];

      for (int qx = 0; qx < q; ++qx)
      {
         const real_t coeff_val = const_coeff ? C(0,0) : C(qx,e);
         QQ[qx] = W(qx) * coeff_val;
      }
      for (int dx = 0; dx < d; ++dx)
      {
         real_t u = 0;
         for (int qx = 0; qx < q; ++qx) { u += QQ[qx] * B(qx,dx); }
         Y(dx,e) += u;
      }
   });
}

template<int T_D1D = 0, int T_Q1D = 0> static
void BFLFEvalAssemble3D(const int nbe, const int d, const int q,
                        const int *markers, const real_t *b,
                        const real_t *weights, const Vector &coeff, real_t *y)
{
   const auto F = coeff.Read();
   const auto M = Reshape(markers, nbe);
   const auto B = Reshape(b, q, d);
   const auto W = Reshape(weights, q, q);
   const bool const_coeff = coeff.Size() == 1;
   const auto C = const_coeff ? Reshape(F,1,1,1) : Reshape(F,q,q,nbe);
   auto Y = Reshape(y, d, d, nbe);

   mfem::forall_2D(nbe, q, q, [=] MFEM_HOST_DEVICE (int e)
   {
      if (M(e) == 0) { return; } // ignore

      constexpr int Q = T_Q1D ? T_Q1D : DofQuadLimits::MAX_Q1D;
      constexpr int D = T_D1D ? T_D1D : DofQuadLimits::MAX_D1D;

      MFEM_SHARED real_t sBt[Q*D];
      MFEM_SHARED real_t sQQ[Q*Q];
      MFEM_SHARED real_t sQD[Q*D];

      const DeviceMatrix Bt(sBt, d, q);
      kernels::internal::LoadB<D,Q>(d, q, B, sBt);

      const DeviceMatrix QQ(sQQ, q, q);
      const DeviceMatrix QD(sQD, q, d);

      MFEM_FOREACH_THREAD(x,x,q)
      {
         MFEM_FOREACH_THREAD(y,y,q)
         {
            const real_t coeff_val = const_coeff ? C(0,0,0) : C(x,y,e);
            QQ(y,x) = W(x,y) * coeff_val;
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(qy,y,q)
      {
         MFEM_FOREACH_THREAD(dx,x,d)
         {
            real_t u = 0.0;
            for (int qx = 0; qx < q; ++qx) { u += QQ(qy,qx) * Bt(dx,qx); }
            QD(qy,dx) = u;
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(dy,y,d)
      {
         MFEM_FOREACH_THREAD(dx,x,d)
         {
            real_t u = 0.0;
            for (int qy = 0; qy < q; ++qy) { u += QD(qy,dx) * Bt(dy,qy); }
            Y(dx,dy,e) += u;
         }
      }
      MFEM_SYNC_THREAD;
   });
}

static void BFLFEvalAssemble(const FiniteElementSpace &fes,
                             const IntegrationRule &ir,
                             const Array<int> &markers,
                             const Vector &coeff,
                             Vector &y)
{
   Mesh &mesh = *fes.GetMesh();
   const int dim = mesh.Dimension();
   const FiniteElement &el = *fes.GetBE(0);
   const DofToQuad &maps = el.GetDofToQuad(ir, DofToQuad::TENSOR);
   const int d = maps.ndof, q = maps.nqpt;
   auto ker = (dim == 2) ? BFLFEvalAssemble2D<> : BFLFEvalAssemble3D<>;

   if (dim==2)
   {
      if (d==1 && q==1) { ker=BFLFEvalAssemble2D<1,1>; }
      if (d==2 && q==2) { ker=BFLFEvalAssemble2D<2,2>; }
      if (d==3 && q==3) { ker=BFLFEvalAssemble2D<3,3>; }
      if (d==4 && q==4) { ker=BFLFEvalAssemble2D<4,4>; }
      if (d==5 && q==5) { ker=BFLFEvalAssemble2D<5,5>; }
      if (d==2 && q==3) { ker=BFLFEvalAssemble2D<2,3>; }
      if (d==3 && q==4) { ker=BFLFEvalAssemble2D<3,4>; }
      if (d==4 && q==5) { ker=BFLFEvalAssemble2D<4,5>; }
      if (d==5 && q==6) { ker=BFLFEvalAssemble2D<5,6>; }
   }

   if (dim==3)
   {
      if (d==1 && q==1) { ker=BFLFEvalAssemble3D<1,1>; }
      if (d==2 && q==2) { ker=BFLFEvalAssemble3D<2,2>; }
      if (d==3 && q==3) { ker=BFLFEvalAssemble3D<3,3>; }
      if (d==4 && q==4) { ker=BFLFEvalAssemble3D<4,4>; }
      if (d==5 && q==5) { ker=BFLFEvalAssemble3D<5,5>; }
      if (d==2 && q==3) { ker=BFLFEvalAssemble3D<2,3>; }
      if (d==3 && q==4) { ker=BFLFEvalAssemble3D<3,4>; }
      if (d==4 && q==5) { ker=BFLFEvalAssemble3D<4,5>; }
      if (d==5 && q==6) { ker=BFLFEvalAssemble3D<5,6>; }
   }

   MFEM_VERIFY(ker, "No kernel ndof " << d << " nqpt " << q);

   const int nbe = fes.GetMesh()->GetNFbyType(FaceType::Boundary);
   const int *M = markers.Read();
   const real_t *B = maps.B.Read();
   const real_t *W = ir.GetWeights().Read();
   real_t *Y = y.ReadWrite();
   ker(nbe, d, q, M, B, W, coeff, Y);
}

void VectorFEBoundaryFluxLFIntegrator::AssembleDevice(
   const FiniteElementSpace &fes,
   const Array<int> &markers,
   Vector &b)
{
   const FiniteElement &fe = *fes.GetBE(0);
   const int qorder = oa * fe.GetOrder() + ob;
   const Geometry::Type gtype = fe.GetGeomType();
   const IntegrationRule &ir = IntRule ? *IntRule : IntRules.Get(gtype, qorder);
   Mesh &mesh = *fes.GetMesh();

   FaceQuadratureSpace qs(mesh, ir, FaceType::Boundary);
   CoefficientVector coeff(F, qs, CoefficientStorage::COMPRESSED);
   BFLFEvalAssemble(fes, ir, markers, coeff, b);
}

} // namespace mfem
