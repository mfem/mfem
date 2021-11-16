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
                          const int d1d,
                          const int q1d,
                          const double *marks,
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
   const int D1D = maps.ndof;
   const int Q1D = maps.nqpt;
   const int id = (dim << 8) | (D1D << 4) | Q1D;
   return id;
}

////////////////////////////////////////////////////////////////////////////////
inline void Launch(const Kernel_f &kernel,
                   const FiniteElementSpace &fes,
                   const IntegrationRule *ir,
                   const Vector &coeff,
                   const Vector &mark,
                   Vector &y)
{
   Mesh *mesh = fes.GetMesh();
   const int vdim = fes.GetVDim();
   const bool byVDIM = fes.GetOrdering() == Ordering::byVDIM;

   const FiniteElement &el = *fes.GetFE(0);
   const int flags = GeometricFactors::JACOBIANS;
   const MemoryType mt = Device::GetDeviceMemoryType();
   const GeometricFactors *geom = mesh->GetGeometricFactors(*ir, flags, mt);
   const DofToQuad &maps = el.GetDofToQuad(*ir, DofToQuad::TENSOR);
   constexpr ElementDofOrdering ordering = ElementDofOrdering::LEXICOGRAPHIC;
   const Operator *ERop = fes.GetElementRestriction(ordering);
   const ElementRestriction* ER = dynamic_cast<const ElementRestriction*>(ERop);
   MFEM_ASSERT(ER, "Not supported!");

   const double *M = mark.Read();
   const double *B = maps.B.Read();
   const double *G = maps.G.Read();
   const int *I = ER->GatherMap().Read();
   const double *J = geom->J.Read();
   const double *W = ir->GetWeights().Read();
   double *Y = y.ReadWrite();

   const int ND = fes.GetNDofs();
   const int NE = fes.GetMesh()->GetNE();

   const int d1d = maps.ndof;
   const int q1d = maps.nqpt;

   kernel(vdim,byVDIM,ND,NE,d1d,q1d,M,B,G,I,J,W,coeff,Y);
}

////////////////////////////////////////////////////////////////////////////////
template<typename T> static MFEM_HOST_DEVICE inline
T *alloc(T* &mem, size_t size, T* base = 0) noexcept
{
   return (base = mem, mem += size, base);
}

////////////////////////////////////////////////////////////////////////////////
template<int D1D=0, int Q1D=0> static
void VectorDomainLFIntegratorAssemble2D(const int vdim,
                                        const bool byVDIM,
                                        const int ND,
                                        const int NE,
                                        const int d1d,
                                        const int q1d,
                                        const double *marks,
                                        const double *b,
                                        const double */*g*/, // unused
                                        const int *idx,
                                        const double *jacobians,
                                        const double *weights,
                                        const Vector &coeff,
                                        double *y)
{
   constexpr int DIM = 2;
   constexpr bool USE_SMEM = D1D > 0 && Q1D > 0;

   const bool cst_coeff = coeff.Size() == vdim;

   const auto F = coeff.Read();
   const auto M = Reshape(marks, NE);
   const auto B = Reshape(b, q1d,d1d);
   const auto J = Reshape(jacobians, q1d,q1d,DIM,DIM,NE);
   const auto W = Reshape(weights, q1d,q1d);
   const auto I = Reshape(idx, d1d,d1d, NE);
   const auto C = cst_coeff ?
                  Reshape(F,vdim,1,1,1):
                  Reshape(F,vdim,q1d,q1d,NE);

   auto Y = Reshape(y, byVDIM ? vdim : ND, byVDIM ? ND : vdim);

   const int sm_size = 2*q1d*(d1d+q1d);

   const int GRID = USE_SMEM ? 0 : 128;
   double *gmem = nullptr;
   static Vector *d_buffer = nullptr;
   if (!USE_SMEM)
   {
      if (!d_buffer)
      {
         d_buffer = new Vector;
         d_buffer->UseDevice(true);
      }
      d_buffer->SetSize(sm_size*GRID);
      gmem = d_buffer->Write();
   }

   MFEM_FORALL_3D_GRID(e, NE, q1d,q1d,1, GRID,
   {
      if (M(e) < 1.0) return;

      const int bid = MFEM_BLOCK_ID(x);
      const int sm_SIZE = 2*Q1D*(D1D+Q1D);
      MFEM_SHARED double SMEM[USE_SMEM ? sm_SIZE : 1];
      double *sm = USE_SMEM ? SMEM : (gmem + sm_size*bid);
      const DeviceMatrix Bt(alloc(sm,q1d*d1d),q1d,d1d);
      const DeviceMatrix QQ(alloc(sm,q1d*q1d),q1d,q1d);
      const DeviceMatrix QD(alloc(sm,q1d*d1d),q1d,d1d);

      kernels::internal::LoadB(d1d,q1d,B,Bt);

      for (int c = 0; c < vdim; ++c)
      {
         const double cst_val = C(c,0,0,0);
         MFEM_FOREACH_THREAD(qx,x,q1d)
         {
            MFEM_FOREACH_THREAD(qy,y,q1d)
            {
               double Jloc[4];
               Jloc[0] = J(qx,qy,0,0,e);
               Jloc[1] = J(qx,qy,1,0,e);
               Jloc[2] = J(qx,qy,0,1,e);
               Jloc[3] = J(qx,qy,1,1,e);
               const double detJ = kernels::Det<2>(Jloc);
               const double coeff_val = cst_coeff ? cst_val : C(c,qx,qy,e);
               QQ(qy,qx) = W(qx,qy) * coeff_val * detJ;

            }
         }
         MFEM_SYNC_THREAD;
         kernels::internal::Atomic2DEvalTranspose(d1d,q1d,Bt,QQ,QD,I,Y,c,e,byVDIM);
      }
   });
}

template<int D1D=0, int Q1D=0> static
void VectorDomainLFIntegratorAssemble3D(const int vdim,
                                        const bool byVDIM,
                                        const int ND,
                                        const int NE,
                                        const int d1d,
                                        const int q1d,
                                        const double *marks,
                                        const double *b,
                                        const double */*g*/, // unused
                                        const int *idx,
                                        const double *jacobians,
                                        const double *weights,
                                        const Vector &coeff,
                                        double *y)
{
   constexpr int DIM = 3;
   constexpr bool USE_SMEM = D1D > 0 && Q1D > 0;

   const bool cst_coeff = coeff.Size() == vdim;

   const auto F = coeff.Read();
   const auto M = Reshape(marks, NE);
   const auto B = Reshape(b, q1d,d1d);
   const auto J = Reshape(jacobians, q1d,q1d,q1d,DIM,DIM,NE);
   const auto W = Reshape(weights, q1d,q1d,q1d);
   const auto I = Reshape(idx, d1d,d1d,d1d, NE);
   const auto C = cst_coeff ?
                  Reshape(F,vdim,1,1,1,1) :
                  Reshape(F,vdim,q1d,q1d,q1d,NE);

   auto Y = Reshape(y, byVDIM ? vdim : ND, byVDIM ? ND : vdim);

   const int sm_size = q1d*d1d + q1d*q1d*q1d;

   const int GRID = USE_SMEM ? 0 : 128;
   double *gmem = nullptr;
   static Vector *d_buffer = nullptr;
   if (!USE_SMEM)
   {
      if (!d_buffer)
      {
         d_buffer = new Vector;
         d_buffer->UseDevice(true);
      }
      d_buffer->SetSize(sm_size*GRID);
      gmem = d_buffer->Write();
   }

   MFEM_FORALL_3D_GRID(e, NE, q1d,q1d,1, GRID,
   {
      if (M(e) < 1.0) return;

      double u[Q1D>0?Q1D:32];

      const int bid = MFEM_BLOCK_ID(x);
      const int sm_SIZE = Q1D*D1D + Q1D*Q1D*Q1D;
      MFEM_SHARED double SMEM[USE_SMEM ? sm_SIZE : 1];
      double *sm = USE_SMEM ? SMEM : (gmem + sm_size*bid);
      const DeviceCube Q(alloc(sm,q1d*q1d*q1d),q1d,q1d,q1d);
      const DeviceMatrix Bt(alloc(sm,q1d*d1d),q1d,d1d);
      kernels::internal::LoadB(d1d,q1d,B,Bt);

      for (int c = 0; c < vdim; ++c)
      {
         const double cst_val = C(c,0,0,0,0);
         MFEM_FOREACH_THREAD(qx,x,q1d)
         {
            MFEM_FOREACH_THREAD(qy,y,q1d)
            {
               for (int qz = 0; qz < q1d; ++qz)
               {
                  double Jloc[9];
                  for (int col = 0; col < 3; col++)
                  {
                     for (int row = 0; row < 3; row++)
                     {
                        Jloc[row+3*col] = J(qx,qy,qz,row,col,e);
                     }
                  }
                  const double detJ = kernels::Det<3>(Jloc);
                  const double coeff_val = cst_coeff ? cst_val : C(c,qx,qy,qz,e);
                  Q(qz,qy,qx) = W(qx,qy,qz) * coeff_val * detJ;
               }
            }
         }
         MFEM_SYNC_THREAD;
         kernels::internal::Atomic3DEvalTranspose(d1d,q1d,u,Bt,Q,I,Y,c,e,byVDIM);
      }
   });
}

} // namespace linearform_extension

} // namespace internal

} // namespace mfem
