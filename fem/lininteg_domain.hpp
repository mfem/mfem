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

using Kernel_f = void (*)(const int vdim,
                          const bool byVDIM,
                          const int ND,
                          const int NE,
                          const double *marks,
                          const double *b,
                          const double *g,
                          const int *idx,
                          const double *jacobians,
                          const double *weights,
                          const Vector &coeff,
                          double *output);

using GetOrder_f = std::function<int(int)>;

inline const IntegrationRule*GetIntegrationRule(const FiniteElementSpace &fes,
                                                const IntegrationRule *IntRule,
                                                GetOrder_f &GetIrOrder)
{
   const FiniteElement &fe = *fes.GetFE(0);
   const int qorder = GetIrOrder(fe.GetOrder());
   const Geometry::Type geom_type = fe.GetGeomType();
   return IntRule ? IntRule : &IntRules.Get(geom_type, qorder);
};

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

   kernel(vdim,byVDIM,ND,NE,M,B,G,I,J,W,coeff,Y);
}

template<int D1D, int Q1D> static
void VectorDomainLFIntegratorAssemble2D(const int vdim,
                                        const bool byVDIM,
                                        const int ND,
                                        const int NE,
                                        const double *marks,
                                        const double *b,
                                        const double */*g*/, // unused
                                        const int *idx,
                                        const double *jacobians,
                                        const double *weights,
                                        const Vector &coeff,
                                        double * __restrict y)
{
   constexpr int DIM = 2;

   const bool cst_coeff = coeff.Size() == vdim;

   const auto F = coeff.Read();
   const auto M = Reshape(marks, NE);
   const auto B = Reshape(b, Q1D,D1D);
   const auto J = Reshape(jacobians, Q1D,Q1D,DIM,DIM,NE);
   const auto W = Reshape(weights, Q1D,Q1D);
   const auto I = Reshape(idx, D1D,D1D, NE);
   const auto C = cst_coeff ?
                  Reshape(F,vdim,1,1,1):
                  Reshape(F,vdim,Q1D,Q1D,NE);

   auto Y = Reshape(y, byVDIM ? vdim : ND, byVDIM ? ND : vdim);

   MFEM_FORALL_2D(e, NE, Q1D,Q1D,1,
   {
      if (M(e) < 1.0) return;

      MFEM_SHARED double sB[D1D*Q1D];
      const DeviceMatrix Bt(sB,Q1D,D1D);
      kernels::internal::LoadB(D1D,Q1D,B,Bt);

      MFEM_SHARED double sQQ[Q1D*Q1D];
      MFEM_SHARED double sQD[Q1D*D1D];
      const DeviceMatrix QQ(sQQ,Q1D,Q1D);
      const DeviceMatrix QD(sQD,Q1D,D1D);

      for (int c = 0; c < vdim; ++c)
      {
         const double cst_val = C(c,0,0,0);
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            MFEM_FOREACH_THREAD(qy,y,Q1D)
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
         kernels::internal::Atomic2DEvalTranspose(D1D,Q1D,Bt,QQ,QD,I,Y,c,e,byVDIM);
      }
   });
}

template<int D1D, int Q1D> static
void VectorDomainLFIntegratorAssemble3D(const int vdim,
                                        const bool byVDIM,
                                        const int ND,
                                        const int NE,
                                        const double *marks,
                                        const double *b,
                                        const double */*g*/, // unused
                                        const int *idx,
                                        const double *jacobians,
                                        const double *weights,
                                        const Vector &coeff,
                                        double * __restrict y)
{
   constexpr int DIM = 3;

   const bool cst_coeff = coeff.Size() == vdim;

   const auto F = coeff.Read();
   const auto M = Reshape(marks, NE);
   const auto B = Reshape(b, Q1D,D1D);
   const auto J = Reshape(jacobians, Q1D,Q1D,Q1D,DIM,DIM,NE);
   const auto W = Reshape(weights, Q1D,Q1D,Q1D);
   const auto I = Reshape(idx, D1D,D1D,D1D, NE);
   const auto C = cst_coeff ?
                  Reshape(F,vdim,1,1,1,1) :
                  Reshape(F,vdim,Q1D,Q1D,Q1D,NE);

   auto Y = Reshape(y, byVDIM ? vdim : ND, byVDIM ? ND : vdim);

   MFEM_FORALL_2D(e, NE, Q1D, Q1D, 1,
   {
      if (M(e) < 1.0) return;

      double u[Q1D];

      MFEM_SHARED double sB[Q1D*D1D];
      const DeviceMatrix Bt(sB,Q1D,D1D);
      kernels::internal::LoadB(D1D,Q1D,B,Bt);

      MFEM_SHARED double sq[Q1D*Q1D*Q1D];
      const DeviceCube Q(sq,Q1D,Q1D,Q1D);

      for (int c = 0; c < vdim; ++c)
      {
         const double cst_val = C(c,0,0,0,0);
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            MFEM_FOREACH_THREAD(qy,y,Q1D)
            {
               for (int qz = 0; qz < Q1D; ++qz)
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
         kernels::internal::Atomic3DEvalTranspose(D1D,Q1D,u,Bt,Q,I,Y,c,e,byVDIM);
      }
   });
}

} // namespace linearform_extension

} // namespace internal

} // namespace mfem
