// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

// Internal header, included only by .cpp files

#include "fem.hpp"

#include "../fem/kernels.hpp"
#include "../general/forall.hpp"
#include "../linalg/kernels.hpp"

namespace mfem
{

namespace internal
{

namespace linearform_extension
{

////////////////////////////////////////////////////////////////////////////////
using Kernel_f = void (*)(const int vdim,
                          const bool byVDIM,
                          const int ND,
                          const int NE,
                          const int d,
                          const int q,
                          const int *markers,
                          const double *b,
                          const double *g,
                          const int *idx,
                          const double *jacobians,
                          const double *weights,
                          const Vector &coeff,
                          double *output);

using GetOrder_f = std::function<int(int)>;

////////////////////////////////////////////////////////////////////////////////
inline const IntegrationRule*GetIntegrationRule(const FiniteElementSpace &fes,
                                                const IntegrationRule *IntRule,
                                                GetOrder_f &GetIrOrder)
{
   const FiniteElement &fe = *fes.GetFE(0);
   const int qorder = GetIrOrder(fe.GetOrder());
   const Geometry::Type geom_type = fe.GetGeomType();
   return IntRule ? IntRule : &IntRules.Get(geom_type, qorder);
}

////////////////////////////////////////////////////////////////////////////////
inline int GetKernelId(const FiniteElementSpace &fes,
                       const IntegrationRule *ir)
{
   Mesh *mesh = fes.GetMesh();
   const int dim = mesh->Dimension();
   const FiniteElement &el = *fes.GetFE(0);
   const DofToQuad &maps = el.GetDofToQuad(*ir, DofToQuad::TENSOR);
   const int d = maps.ndof;
   const int q = maps.nqpt;
   const int id = (dim << 8) | (d << 4) | q;
   return id;
}

////////////////////////////////////////////////////////////////////////////////
inline void Launch(const Kernel_f &kernel,
                   const FiniteElementSpace &fes,
                   const IntegrationRule *ir,
                   const Vector &coeff,
                   const Array<int> &markers,
                   Vector &y)
{
   Mesh *mesh = fes.GetMesh();
   const int vdim = fes.GetVDim();
   const bool byVDIM = fes.GetOrdering() == Ordering::byVDIM;

   const FiniteElement &el = *fes.GetFE(0);
   constexpr int flags = GeometricFactors::JACOBIANS;
   const MemoryType mt = Device::GetDeviceMemoryType();
   const GeometricFactors *geom = mesh->GetGeometricFactors(*ir, flags, mt);
   const DofToQuad &maps = el.GetDofToQuad(*ir, DofToQuad::TENSOR);
   constexpr ElementDofOrdering ordering = ElementDofOrdering::LEXICOGRAPHIC;
   const Operator *ERop = fes.GetElementRestriction(ordering);
   const ElementRestriction* ER = dynamic_cast<const ElementRestriction*>(ERop);
   MFEM_ASSERT(ER, "Not supported!");

   const int *M = markers.Read();
   const double *B = maps.B.Read();
   const double *G = maps.G.Read();
   const double *J = geom->J.Read();
   const int *I = ER->GatherMap().Read();
   const double *W = ir->GetWeights().Read();
   double *Y = y.ReadWrite();

   const int ND = fes.GetNDofs();
   const int NE = fes.GetMesh()->GetNE();

   const int d = maps.ndof;
   const int q = maps.nqpt;

   kernel(vdim, byVDIM, ND, NE, d, q, M, B, G, I, J, W, coeff, Y);
}

////////////////////////////////////////////////////////////////////////////////
template<int D=0, int Q=0> static
void VectorDomainLFIntegratorAssemble2D(const int vdim,
                                        const bool byVDIM,
                                        const int ND,
                                        const int NE,
                                        const int d,
                                        const int q,
                                        const int *markers,
                                        const double *b,
                                        const double */*g*/, // unused
                                        const int *idx,
                                        const double *jacobians,
                                        const double *weights,
                                        const Vector &coeff,
                                        double *y)
{
   constexpr int DIM = 2;
   constexpr bool USE_SMEM = D > 0 && Q > 0;

   const bool cst_coeff = coeff.Size() == vdim;

   const auto F = coeff.Read();
   const auto M = Reshape(markers, NE);
   const auto B = Reshape(b, q,d);
   const auto J = Reshape(jacobians, q,q, DIM,DIM, NE);
   const auto W = Reshape(weights, q,q);
   const auto I = Reshape(idx, d,d, NE);
   const auto C = cst_coeff ? Reshape(F,vdim,1,1,1) : Reshape(F,vdim,q,q,NE);

   auto Y = Reshape(y, byVDIM ? vdim : ND, byVDIM ? ND : vdim);

   const int sm_size = 2*q*(d+q);
   constexpr int GRID = USE_SMEM ? 0 : 128;
   double *gmem = ScratchMem<GRID>(sm_size);

   MFEM_FORALL_3D_GRID(e, NE, q,q,1, GRID,
   {
      if (M(e) == 0) { return; }

      const int bid = MFEM_BLOCK_ID(x);
      constexpr int SM_SIZE = 2*Q*(D+Q);
      constexpr bool USE_SMEM = D > 0 && Q > 0;
      MFEM_SHARED double SMEM[USE_SMEM ? SM_SIZE : 1];
      double *sm = USE_SMEM ? SMEM : (gmem + sm_size*bid);
      const DeviceMatrix Bt(DeviceMemAlloc(sm,q*d), q,d);
      const DeviceMatrix QQ(DeviceMemAlloc(sm,q*q), q,q);
      const DeviceMatrix QD(DeviceMemAlloc(sm,q*d), q,d);

      kernels::internal::LoadB(d,q,B,Bt);

      for (int c = 0; c < vdim; ++c)
      {
         const double cst_val = C(c,0,0,0);
         MFEM_FOREACH_THREAD(x,x,q)
         {
            MFEM_FOREACH_THREAD(y,y,q)
            {
               double Jloc[4];
               Jloc[0] = J(x,y,0,0,e);
               Jloc[1] = J(x,y,1,0,e);
               Jloc[2] = J(x,y,0,1,e);
               Jloc[3] = J(x,y,1,1,e);
               const double detJ = kernels::Det<2>(Jloc);
               const double coeff_val = cst_coeff ? cst_val : C(c,x,y,e);
               QQ(y,x) = W(x,y) * coeff_val * detJ;

            }
         }
         MFEM_SYNC_THREAD;
         kernels::internal::Atomic2DEvalTranspose(d,q,Bt,QQ,QD,I,Y,c,e,byVDIM);
      }
   });
}

template<int D=0, int Q=0> static
void VectorDomainLFIntegratorAssemble3D(const int vdim,
                                        const bool byVDIM,
                                        const int ND,
                                        const int NE,
                                        const int d,
                                        const int q,
                                        const int *markers,
                                        const double *b,
                                        const double */*g*/, // unused
                                        const int *idx,
                                        const double *jacobians,
                                        const double *weights,
                                        const Vector &coeff,
                                        double *y)
{
   constexpr int DIM = 3;
   constexpr bool USE_SMEM = D > 0 && Q > 0;

   const bool cst_coeff = coeff.Size() == vdim;

   const auto F = coeff.Read();
   const auto M = Reshape(markers, NE);
   const auto B = Reshape(b, q,d);
   const auto J = Reshape(jacobians, q,q,q, DIM,DIM, NE);
   const auto W = Reshape(weights, q,q,q);
   const auto I = Reshape(idx, d,d,d, NE);
   const auto C = cst_coeff ?
                  Reshape(F,vdim,1,1,1,1) :
                  Reshape(F,vdim,q,q,q,NE);

   auto Y = Reshape(y, byVDIM ? vdim : ND, byVDIM ? ND : vdim);

   const int sm_size = q*d + q*q*q;
   const int GRID = USE_SMEM ? 0 : 128;
   double *gmem = ScratchMem<GRID>(sm_size);
   MFEM_VERIFY(q < 32, "Unsupported quadrature order!");

   MFEM_FORALL_3D_GRID(e, NE, q,q,1, GRID,
   {
      if (M(e) == 0) { return; }

      double u[Q>0?Q:32];

      const int bid = MFEM_BLOCK_ID(x);
      constexpr int SM_SIZE = Q*D + Q*Q*Q;
      constexpr bool USE_SMEM = D > 0 && Q > 0;
      MFEM_SHARED double SMEM[USE_SMEM ? SM_SIZE : 1];
      double *sm = USE_SMEM ? SMEM : (gmem + sm_size*bid);
      const DeviceCube QQQ(DeviceMemAlloc(sm,q*q*q), q,q,q);
      const DeviceMatrix Bt(DeviceMemAlloc(sm,q*d), q,d);
      kernels::internal::LoadB(d,q,B,Bt);

      for (int c = 0; c < vdim; ++c)
      {
         const double cst_val = C(c,0,0,0,0);
         MFEM_FOREACH_THREAD(x,x,q)
         {
            MFEM_FOREACH_THREAD(y,y,q)
            {
               for (int z = 0; z < q; ++z)
               {
                  double Jloc[9];
                  for (int j = 0; j < 3; j++)
                  {
                     for (int i = 0; i < 3; i++)
                     {
                        Jloc[i+3*j] = J(x,y,z,i,j,e);
                     }
                  }
                  const double detJ = kernels::Det<3>(Jloc);
                  const double coeff_val = cst_coeff ? cst_val : C(c,x,y,z,e);
                  QQQ(z,y,x) = W(x,y,z) * coeff_val * detJ;
               }
            }
         }
         MFEM_SYNC_THREAD;
         kernels::internal::Atomic3DEvalTranspose(d,q,u,Bt,QQQ,I,Y,c,e,byVDIM);
      }
   });
}

} // namespace linearform_extension

} // namespace internal

} // namespace mfem
