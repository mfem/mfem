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

#include "../../fem/kernels.hpp"
#include "../../general/forall.hpp"
#include "../fem.hpp"

/// \cond DO_NOT_DOCUMENT

namespace mfem
{

template <int T_D1D = 0, int T_Q1D = 0>
static void DLFEvalAssemble1D(const int vdim, const int ne, const int d,
                              const int q, const int map_type,
                              const int *markers, const real_t *b,
                              const real_t *detj, const real_t *weights,
                              const Vector &coeff, real_t *y)
{
   {
      constexpr int Q = T_Q1D ? T_Q1D : DofQuadLimits::MAX_Q1D;
      constexpr int D = T_D1D ? T_D1D : DofQuadLimits::MAX_D1D;
      MFEM_VERIFY(q <= Q, "");
      MFEM_VERIFY(d <= D, "");
   }

   const auto F = coeff.Read();
   const auto B = Reshape(b, q, d);
   const auto DETJ = Reshape(detj, q, ne);
   const bool cst = coeff.Size() == vdim;
   const auto C = cst ? Reshape(F, vdim, 1, 1) : Reshape(F, vdim, q, ne);
   auto Y = Reshape(y, d, vdim, ne);

   mfem::forall_2D(ne, d, 1, [=] MFEM_HOST_DEVICE(int e)
   {
      if (markers[e] == 0)
      {
         return;
      } // ignore

      constexpr int Q = T_Q1D ? T_Q1D : DofQuadLimits::MAX_Q1D;
      constexpr int D = T_D1D ? T_D1D : DofQuadLimits::MAX_D1D;

      MFEM_SHARED real_t sBt[Q * D];
      const DeviceMatrix Bt(sBt, d, q);
      kernels::internal::LoadB<D, Q>(d, q, B, sBt);

      for (int c = 0; c < vdim; ++c)
      {
         const real_t cst_val = C(c, 0, 0);
         MFEM_FOREACH_THREAD(dx, x, d)
         {
            real_t u = 0;
            for (int qx = 0; qx < q; ++qx)
            {
               const real_t detJ =
                  (map_type == FiniteElement::VALUE) ? DETJ(qx, e) : 1.0;
               const real_t coeff_val = cst ? cst_val : C(c, qx, e);
               u += weights[qx] * coeff_val * detJ * Bt(dx, qx);
            }
            Y(dx, c, e) += u;
         }
      }
   });
}

template <int T_D1D = 0, int T_Q1D = 0>
static void DLFEvalAssemble2D(const int vdim, const int ne, const int d,
                              const int q, const int map_type,
                              const int *markers, const real_t *b,
                              const real_t *detj, const real_t *weights,
                              const Vector &coeff, real_t *y)
{
   {
      constexpr int Q = T_Q1D ? T_Q1D : DofQuadLimits::MAX_Q1D;
      constexpr int D = T_D1D ? T_D1D : DofQuadLimits::MAX_D1D;
      MFEM_VERIFY(q <= Q, "");
      MFEM_VERIFY(d <= D, "");
   }

   const auto F = coeff.Read();
   const auto B = Reshape(b, q, d);
   const auto DETJ = Reshape(detj, q, q, ne);
   const auto W = Reshape(weights, q, q);
   const bool cst = coeff.Size() == vdim;
   const auto C = cst ? Reshape(F, vdim, 1, 1, 1) : Reshape(F, vdim, q, q, ne);
   auto Y = Reshape(y, d, d, vdim, ne);

   mfem::forall_2D(ne, q, q, [=] MFEM_HOST_DEVICE(int e)
   {
      if (markers[e] == 0)
      {
         return;
      } // ignore

      constexpr int Q = T_Q1D ? T_Q1D : DofQuadLimits::MAX_Q1D;
      constexpr int D = T_D1D ? T_D1D : DofQuadLimits::MAX_D1D;

      MFEM_SHARED real_t sBt[Q * D];
      MFEM_SHARED real_t sQQ[Q * Q];
      MFEM_SHARED real_t sQD[Q * D];

      const DeviceMatrix Bt(sBt, d, q);
      kernels::internal::LoadB<D, Q>(d, q, B, sBt);

      const DeviceMatrix QQ(sQQ, q, q);
      const DeviceMatrix QD(sQD, q, d);

      for (int c = 0; c < vdim; ++c)
      {
         const real_t cst_val = C(c, 0, 0, 0);
         MFEM_FOREACH_THREAD(x, x, q)
         {
            MFEM_FOREACH_THREAD(y, y, q)
            {
               const real_t detJ =
                  (map_type == FiniteElement::VALUE) ? DETJ(x, y, e) : 1.0;
               const real_t coeff_val = cst ? cst_val : C(c, x, y, e);
               QQ(y, x) = W(x, y) * coeff_val * detJ;
            }
         }
         MFEM_SYNC_THREAD;
         MFEM_FOREACH_THREAD(qy, y, q)
         {
            MFEM_FOREACH_THREAD(dx, x, d)
            {
               real_t u = 0.0;
               for (int qx = 0; qx < q; ++qx)
               {
                  u += QQ(qy, qx) * Bt(dx, qx);
               }
               QD(qy, dx) = u;
            }
         }
         MFEM_SYNC_THREAD;
         MFEM_FOREACH_THREAD(dy, y, d)
         {
            MFEM_FOREACH_THREAD(dx, x, d)
            {
               real_t u = 0.0;
               for (int qy = 0; qy < q; ++qy)
               {
                  u += QD(qy, dx) * Bt(dy, qy);
               }
               Y(dx, dy, c, e) += u;
            }
         }
         MFEM_SYNC_THREAD;
      }
   });
}

template <int T_D1D = 0, int T_Q1D = 0>
static void DLFEvalAssemble3D(const int vdim, const int ne, const int d,
                              const int q, const int map_type,
                              const int* markers, const real_t *b,
                              const real_t *detj, const real_t *weights,
                              const Vector &coeff, real_t *y)
{
   {
      constexpr int Q = T_Q1D ? T_Q1D : DofQuadLimits::MAX_Q1D;
      constexpr int D = T_D1D ? T_D1D : DofQuadLimits::MAX_D1D;
      MFEM_VERIFY(q <= Q, "");
      MFEM_VERIFY(d <= D, "");
   }

   const auto F = coeff.Read();
   const auto B = Reshape(b, q, d);
   const auto DETJ = Reshape(detj, q, q, q, ne);
   const auto W = Reshape(weights, q, q, q);
   const bool cst_coeff = coeff.Size() == vdim;
   const auto C =
      cst_coeff ? Reshape(F, vdim, 1, 1, 1, 1) : Reshape(F, vdim, q, q, q, ne);

   auto Y = Reshape(y, d, d, d, vdim, ne);

   mfem::forall_2D(ne, q, q, [=] MFEM_HOST_DEVICE(int e)
   {
      if (markers[e] == 0)
      {
         return;
      } // ignore

      constexpr int Q = T_Q1D ? T_Q1D : DofQuadLimits::MAX_Q1D;
      constexpr int D = T_D1D ? T_D1D : DofQuadLimits::MAX_D1D;
      constexpr int MQD = (Q >= D) ? Q : D;

      real_t u[D];

      MFEM_SHARED real_t sBt[Q * D];
      const DeviceMatrix Bt(sBt, d, q);
      kernels::internal::LoadB<D, Q>(d, q, B, sBt);

      MFEM_SHARED real_t sQQQ[MQD * MQD * MQD];
      const DeviceCube QQQ(sQQQ, MQD, MQD, MQD);

      for (int c = 0; c < vdim; ++c)
      {
         const real_t cst_val = C(c, 0, 0, 0, 0);
         MFEM_FOREACH_THREAD(x, x, q)
         {
            MFEM_FOREACH_THREAD(y, y, q)
            {
               for (int z = 0; z < q; ++z)
               {
                  const real_t detJ = (map_type == FiniteElement::VALUE)
                                      ? DETJ(x, y, z, e)
                                      : 1.0;
                  const real_t coeff_val =
                     cst_coeff ? cst_val : C(c, x, y, z, e);
                  QQQ(z, y, x) = W(x, y, z) * coeff_val * detJ;
               }
            }
         }
         MFEM_SYNC_THREAD;
         MFEM_FOREACH_THREAD(qx, x, q)
         {
            MFEM_FOREACH_THREAD(qy, y, q)
            {
               for (int dz = 0; dz < d; ++dz)
               {
                  u[dz] = 0.0;
               }
               for (int qz = 0; qz < q; ++qz)
               {
                  const real_t ZYX = QQQ(qz, qy, qx);
                  for (int dz = 0; dz < d; ++dz)
                  {
                     u[dz] += ZYX * Bt(dz, qz);
                  }
               }
               for (int dz = 0; dz < d; ++dz)
               {
                  QQQ(dz, qy, qx) = u[dz];
               }
            }
         }
         MFEM_SYNC_THREAD;
         MFEM_FOREACH_THREAD(dz, y, d)
         {
            MFEM_FOREACH_THREAD(qx, x, q)
            {
               for (int dy = 0; dy < d; ++dy)
               {
                  u[dy] = 0.0;
               }
               for (int qy = 0; qy < q; ++qy)
               {
                  const real_t zYX = QQQ(dz, qy, qx);
                  for (int dy = 0; dy < d; ++dy)
                  {
                     u[dy] += zYX * Bt(dy, qy);
                  }
               }
               for (int dy = 0; dy < d; ++dy)
               {
                  QQQ(dz, dy, qx) = u[dy];
               }
            }
         }
         MFEM_SYNC_THREAD;
         MFEM_FOREACH_THREAD(dz, y, d)
         {
            MFEM_FOREACH_THREAD(dy, x, d)
            {
               for (int dx = 0; dx < d; ++dx)
               {
                  u[dx] = 0.0;
               }
               for (int qx = 0; qx < q; ++qx)
               {
                  const real_t zyX = QQQ(dz, dy, qx);
                  for (int dx = 0; dx < d; ++dx)
                  {
                     u[dx] += zyX * Bt(dx, qx);
                  }
               }
               for (int dx = 0; dx < d; ++dx)
               {
                  Y(dx, dy, dz, c, e) += u[dx];
               }
            }
         }
         MFEM_SYNC_THREAD;
      }
   });
}

static void DLFEvalAssemble(const FiniteElementSpace &fes,
                            const IntegrationRule *ir,
                            const Array<int> &markers, const Vector &coeff,
                            Vector &y)
{
   Mesh *mesh = fes.GetMesh();
   const int dim = mesh->Dimension();
   const FiniteElement &el = *fes.GetTypicalFE();
   const MemoryType mt = Device::GetDeviceMemoryType();
   const DofToQuad &maps = el.GetDofToQuad(*ir, DofToQuad::TENSOR);
   const int d = maps.ndof, q = maps.nqpt;
   constexpr int flags = GeometricFactors::DETERMINANTS;
   const GeometricFactors *geom = mesh->GetGeometricFactors(*ir, flags, mt);
   const int map_type = fes.GetTypicalFE()->GetMapType();

   const int vdim = fes.GetVDim();
   const int ne = fes.GetMesh()->GetNE();
   const real_t *B = maps.B.Read();
   const int *M = markers.Read();
   const real_t *detJ = geom->detJ.Read();
   const real_t *W = ir->GetWeights().Read();
   real_t *Y = y.ReadWrite();
   DomainLFIntegrator::AssembleKernels::Run(dim, d, q, vdim, ne, d, q, map_type,
                                            M, B, detJ, W, coeff, Y);
}

void DomainLFIntegrator::AssembleDevice(const FiniteElementSpace &fes,
                                        const Array<int> &markers, Vector &b)
{
   const FiniteElement &fe = *fes.GetTypicalFE();
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
   const FiniteElement &fe = *fes.GetTypicalFE();
   const int qorder = 2 * fe.GetOrder();
   const Geometry::Type gtype = fe.GetGeomType();
   const IntegrationRule *ir = IntRule ? IntRule : &IntRules.Get(gtype, qorder);

   QuadratureSpace qs(*fes.GetMesh(), *ir);
   CoefficientVector coeff(Q, qs, CoefficientStorage::COMPRESSED);
   DLFEvalAssemble(fes, ir, markers, coeff, b);
}

template <int DIM, int T_D1D, int T_Q1D>
DomainLFIntegrator::AssembleKernelType
DomainLFIntegrator::AssembleKernels::Kernel()
{
   switch (DIM)
   {
      case 1:
         return DLFEvalAssemble1D<T_D1D, T_Q1D>;
      case 2:
         return DLFEvalAssemble2D<T_D1D, T_Q1D>;
      case 3:
         return DLFEvalAssemble3D<T_D1D, T_Q1D>;
   }
   MFEM_ABORT("");
}

DomainLFIntegrator::AssembleKernelType
DomainLFIntegrator::AssembleKernels::Fallback(int DIM, int, int)
{
   switch (DIM)
   {
      case 1:
         return DLFEvalAssemble1D<0, 0>;
      case 2:
         return DLFEvalAssemble2D<0, 0>;
      case 3:
         return DLFEvalAssemble3D<0, 0>;
   }
   MFEM_ABORT("");
}

DomainLFIntegrator::Kernels::Kernels()
{
   // 2D
   // Q = P+1
   DomainLFIntegrator::AddSpecialization<2, 1, 1>();
   DomainLFIntegrator::AddSpecialization<2, 2, 2>();
   DomainLFIntegrator::AddSpecialization<2, 3, 3>();
   DomainLFIntegrator::AddSpecialization<2, 4, 4>();
   DomainLFIntegrator::AddSpecialization<2, 5, 5>();
   // Q = P+2
   DomainLFIntegrator::AddSpecialization<2, 2, 3>();
   DomainLFIntegrator::AddSpecialization<2, 3, 4>();
   DomainLFIntegrator::AddSpecialization<2, 4, 5>();
   DomainLFIntegrator::AddSpecialization<2, 5, 6>();
   // 3D
   // Q = P+1
   DomainLFIntegrator::AddSpecialization<3, 1, 1>();
   DomainLFIntegrator::AddSpecialization<3, 2, 2>();
   DomainLFIntegrator::AddSpecialization<3, 3, 3>();
   DomainLFIntegrator::AddSpecialization<3, 4, 4>();
   DomainLFIntegrator::AddSpecialization<3, 5, 5>();
   // Q = P+2
   DomainLFIntegrator::AddSpecialization<3, 2, 3>();
   DomainLFIntegrator::AddSpecialization<3, 3, 4>();
   DomainLFIntegrator::AddSpecialization<3, 4, 5>();
   DomainLFIntegrator::AddSpecialization<3, 5, 6>();
}

/// \endcond DO_NOT_DOCUMENT

} // namespace mfem
