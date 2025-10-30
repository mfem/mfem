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
#include "../../gridfunc.hpp" // IWYU pragma: keep
#include "../../../general/forall.hpp"
#include "../../../linalg/kernels.hpp"

namespace mfem
{

template <int T_Q1D = 0>
void TMOP_TcIdealShapeUnitSize_3D(const int NE, const ConstDeviceMatrix &W,
                                  DeviceTensor<6> &J, const int q1d = 0)
{
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   mfem::forall_3D(NE, Q1D, Q1D, Q1D,
                   [=] MFEM_HOST_DEVICE(int e)
   {
      MFEM_FOREACH_THREAD_DIRECT(qy, y, Q1D)
      {
         MFEM_FOREACH_THREAD_DIRECT(qx, x, Q1D)
         {
            MFEM_FOREACH_THREAD_DIRECT(qz, z, Q1D)
            {
               kernels::Set(3, 3, 1.0, &W(0, 0), &J(0, 0, qx, qy, qz, e));
            }
         }
      }
   });
}

MFEM_TMOP_REGISTER_KERNELS_1(TMOPTcIdealShapeUnitSize3D,
                             TMOP_TcIdealShapeUnitSize_3D);
MFEM_TMOP_ADD_SPECIALIZED_KERNELS_1(TMOPTcIdealShapeUnitSize3D);

template <int MD1, int MQ1, int T_D1D = 0, int T_Q1D = 0>
void TMOP_TcIdealShapeGivenSize_3D(const int NE,
                                   const real_t detW,
                                   const real_t *b,
                                   const real_t *g,
                                   const ConstDeviceMatrix &W,
                                   const DeviceTensor<5, const real_t> &X,
                                   DeviceTensor<6> &J, const int d1d,
                                   const int q1d)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   mfem::forall_2D(NE, Q1D, Q1D, [=] MFEM_HOST_DEVICE(int e)
   {
      MFEM_SHARED real_t smem[MQ1][MQ1];
      MFEM_SHARED real_t sB[MD1][MQ1], sG[MD1][MQ1];
      kernels::internal::vd_regs3d_t<3, 3, MQ1> r0, r1;

      kernels::internal::LoadMatrix(D1D, Q1D, b, sB);
      kernels::internal::LoadMatrix(D1D, Q1D, g, sG);

      kernels::internal::LoadDofs3d(e, D1D, X, r0);
      kernels::internal::Grad3d(D1D, Q1D, smem, sB, sG, r0, r1);

      for (int qz = 0; qz < Q1D; ++qz)
      {
         MFEM_FOREACH_THREAD_DIRECT(qy, y, Q1D)
         {
            MFEM_FOREACH_THREAD_DIRECT(qx, x, Q1D)
            {
               const real_t *Wid = &W(0, 0);
               const real_t Jtr[9] =
               {
                  r1(0, 0, qz, qy, qx), r1(1, 0, qz, qy, qx), r1(2, 0, qz, qy, qx),
                  r1(0, 1, qz, qy, qx), r1(1, 1, qz, qy, qx), r1(2, 1, qz, qy, qx),
                  r1(0, 2, qz, qy, qx), r1(1, 2, qz, qy, qx), r1(2, 2, qz, qy, qx)
               };
               const real_t detJ = kernels::Det<3>(Jtr);
               const real_t alpha = std::pow(detJ / detW, 1. / 3);
               kernels::Set(3, 3, alpha, Wid, &J(0, 0, qx, qy, qz, e));
            }
         }
      }
   });
}

MFEM_TMOP_MDQ_REGISTER(TMOPTcIdealShapeGivenSize3D,
                       TMOP_TcIdealShapeGivenSize_3D);
MFEM_TMOP_MDQ_SPECIALIZE(TMOPTcIdealShapeGivenSize3D);

template <>
bool TargetConstructor::ComputeAllElementTargets<3>(
   const FiniteElementSpace &fes, const IntegrationRule &ir, const Vector &,
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

   MFEM_VERIFY(d <= DeviceDofQuadLimits::Get().MAX_D1D, "");
   MFEM_VERIFY(q <= DeviceDofQuadLimits::Get().MAX_Q1D, "");

   const auto W = Reshape(w.Read(), 3, 3);
   const auto *b = maps.B.Read(), *g = maps.G.Read();
   auto J = Reshape(Jtr.Write(), 3, 3, q, q, q, NE);

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
         MFEM_VERIFY(GetNodes(), "No target-matrix nodes!");
         const ElementDofOrdering ordering = ElementDofOrdering::LEXICOGRAPHIC;
         const Operator *R = fes.GetElementRestriction(ordering);
         Vector x(R->Height(), Device::GetDeviceMemoryType());
         x.UseDevice(true);
         R->Mult(*GetNodes(), x);
         MFEM_ASSERT(nodes->FESpace()->GetVDim() == 3, "");
         const auto X = Reshape(x.Read(), d, d, d, 3, NE);

         TMOPTcIdealShapeGivenSize3D::Run(d, q, NE, detW, b, g, W, X, J, d, q);
         return true;
      }
      case GIVEN_SHAPE_AND_SIZE: return false;
      default:                   return false;
   }
   return false;
}

} // namespace mfem
