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

#include "../pa.hpp"
#include "../../tmop.hpp"
#include "../../kernels.hpp"
#include "../../gridfunc.hpp"
#include "../../../general/forall.hpp"
#include "../../../linalg/kernels.hpp"

using namespace mfem;

namespace mfem
{

template <int T_Q1D = 0>
void TMOP_TcIdealShapeUnitSize_3D(const int NE,
                                  const ConstDeviceMatrix &W,
                                  DeviceTensor<6> &J,
                                  const int q1d = 0)
{
   constexpr int DIM = 3;
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   mfem::forall_3D(NE, Q1D, Q1D, Q1D, [=] MFEM_HOST_DEVICE(int e)
   {
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      MFEM_FOREACH_THREAD(qy, y, Q1D)
      {
         MFEM_FOREACH_THREAD(qx, x, Q1D)
         {
            MFEM_FOREACH_THREAD(qz, z, Q1D)
            {
               kernels::Set(DIM, DIM, 1.0, &W(0, 0), &J(0, 0, qx, qy, qz, e));
            }
         }
      }
   });
}

MFEM_TMOP_REGISTER_KERNELS_1(TMOPTcIdealShapeUnitSize3D,
                             TMOP_TcIdealShapeUnitSize_3D);
MFEM_TMOP_ADD_SPECIALIZED_KERNELS_1(TMOPTcIdealShapeUnitSize3D);

template <int T_D1D = 0, int T_Q1D = 0>
void TMOP_TcIdealShapeGivenSize_3D(const int NE,
                                   const real_t detW,
                                   const ConstDeviceMatrix &B,
                                   const ConstDeviceMatrix &G,
                                   const ConstDeviceMatrix &W,
                                   const DeviceTensor<5, const real_t> &X,
                                   DeviceTensor<6> &J,
                                   const int d1d,
                                   const int q1d)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   MFEM_VERIFY(D1D <= DeviceDofQuadLimits::Get().MAX_D1D, "");
   MFEM_VERIFY(Q1D <= DeviceDofQuadLimits::Get().MAX_Q1D, "");

   mfem::forall_3D(NE, Q1D, Q1D, Q1D, [=] MFEM_HOST_DEVICE(int e)
   {
      constexpr int DIM = 3;
      constexpr int MQ1 = T_Q1D ? T_Q1D : DofQuadLimits::MAX_Q1D;
      constexpr int MD1 = T_D1D ? T_D1D : DofQuadLimits::MAX_D1D;

      MFEM_SHARED real_t BG[2][MQ1 * MD1];
      MFEM_SHARED real_t DDD[3][MD1 * MD1 * MD1];
      MFEM_SHARED real_t DDQ[6][MD1 * MD1 * MQ1];
      MFEM_SHARED real_t DQQ[9][MD1 * MQ1 * MQ1];
      MFEM_SHARED real_t QQQ[9][MQ1 * MQ1 * MQ1];

      kernels::internal::LoadX<MD1>(e, D1D, X, DDD);
      kernels::internal::LoadBG<MD1, MQ1>(D1D, Q1D, B, G, BG);

      kernels::internal::GradX<MD1, MQ1>(D1D, Q1D, BG, DDD, DDQ);
      kernels::internal::GradY<MD1, MQ1>(D1D, Q1D, BG, DDQ, DQQ);
      kernels::internal::GradZ<MD1, MQ1>(D1D, Q1D, BG, DQQ, QQQ);

      MFEM_FOREACH_THREAD(qz, z, Q1D)
      {
         MFEM_FOREACH_THREAD(qy, y, Q1D)
         {
            MFEM_FOREACH_THREAD(qx, x, Q1D)
            {
               real_t Jtr[9];
               const real_t *Wid = &W(0, 0);
               kernels::internal::PullGrad<MQ1>(Q1D, qx, qy, qz, QQQ, Jtr);
               const real_t detJ = kernels::Det<3>(Jtr);
               const real_t alpha = std::pow(detJ / detW, 1. / 3);
               kernels::Set(DIM, DIM, alpha, Wid, &J(0, 0, qx, qy, qz, e));
            }
         }
      }
   });
}

MFEM_TMOP_REGISTER_KERNELS(TMOPTcIdealShapeGivenSize3D,
                           TMOP_TcIdealShapeGivenSize_3D);
MFEM_TMOP_ADD_SPECIALIZED_KERNELS(TMOPTcIdealShapeGivenSize3D);

template <>
bool TargetConstructor::ComputeAllElementTargets<3>(
   const FiniteElementSpace &fes,
   const IntegrationRule &ir,
   const Vector &,
   DenseTensor &Jtr) const
{
   MFEM_ASSERT(target_type == IDEAL_SHAPE_UNIT_SIZE || nodes != nullptr, "");
   const Mesh *mesh = fes.GetMesh();
   const int NE = mesh->GetNE();
   // Quick return for empty processors:
   if (NE == 0) { return true; }
   const int dim = mesh->Dimension();
   MFEM_VERIFY(mesh->GetNumGeometries(dim) <= 1,
               "mixed meshes are not supported");
   MFEM_VERIFY(!fes.IsVariableOrder(), "variable orders are not supported");
   const FiniteElement &fe = *fes.GetFE(0);
   MFEM_VERIFY(fe.GetGeomType() == Geometry::CUBE, "");
   const DenseMatrix &w = Geometries.GetGeomToPerfGeomJac(Geometry::CUBE);
   const real_t detW = w.Det();
   const DofToQuad::Mode mode = DofToQuad::TENSOR;
   const DofToQuad &maps = fe.GetDofToQuad(ir, mode);
   const int d = maps.ndof, q = maps.nqpt;

   constexpr int DIM = 3;
   const auto W = Reshape(w.Read(), DIM, DIM);
   const auto B = Reshape(maps.B.Read(), q, d);
   const auto G = Reshape(maps.G.Read(), q, d);
   auto J = Reshape(Jtr.Write(), DIM, DIM, q, q, q, NE);

   switch (target_type)
   {
      case IDEAL_SHAPE_UNIT_SIZE: // Jtr(i) = Wideal;
      {
         TMOPTcIdealShapeUnitSize3D::Run(q, NE, W, J, q);
         return true;
      }
      case IDEAL_SHAPE_EQUAL_SIZE: return false;
      case IDEAL_SHAPE_GIVEN_SIZE:
      {
         MFEM_VERIFY(nodes, "");
         const ElementDofOrdering ordering = ElementDofOrdering::LEXICOGRAPHIC;
         const Operator *R = fes.GetElementRestriction(ordering);
         Vector x(R->Height(), Device::GetDeviceMemoryType());
         x.UseDevice(true);
         R->Mult(*nodes, x);
         MFEM_ASSERT(nodes->FESpace()->GetVDim() == 3, "");
         const auto X = Reshape(x.Read(), d, d, d, DIM, NE);

         TMOPTcIdealShapeGivenSize3D::Run(d, q, NE, detW, B, G, W, X, J, d, q);
         return true;
      }
      case GIVEN_SHAPE_AND_SIZE: return false;
      default:                   return false;
   }
   return false;
}

} // namespace mfem
