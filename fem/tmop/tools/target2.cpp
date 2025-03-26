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
#include "../../gridfunc.hpp" // IWYU pragma: keep
#include "../../../general/forall.hpp"
#include "../../../linalg/kernels.hpp"

namespace mfem
{

template <int T_Q1D = 0>
void TMOP_TcIdealShapeUnitSize_2D(const int NE,
                                  const ConstDeviceMatrix &W,
                                  DeviceTensor<5> &J,
                                  const int q1d)
{
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   mfem::forall_2D(NE, Q1D, Q1D, [=] MFEM_HOST_DEVICE(int e)
   {
      constexpr int DIM = 2;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      mfem::tmop::foreach_y_thread(Q1D, [&](int qy)
      {
         mfem::tmop::foreach_x_thread(Q1D, [&](int qx)
         {
            kernels::Set(DIM, DIM, 1.0, &W(0, 0), &J(0, 0, qx, qy, e));
         });
      });
   });
}

template <int T_D1D = 0, int T_Q1D = 0>
void TMOP_TcIdealShapeGivenSize_2D(const int NE,
                                   const real_t detW,
                                   const real_t *b,
                                   const real_t *g,
                                   const ConstDeviceMatrix &W,
                                   const DeviceTensor<4, const real_t> &X,
                                   DeviceTensor<5> &J,
                                   const int d1d,
                                   const int q1d)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   mfem::forall_2D(NE, Q1D, Q1D, [=] MFEM_HOST_DEVICE(int e)
   {
      static constexpr int DIM = 2, VDIM = 2;
      static constexpr int MD1 = T_D1D ? T_D1D : DofQuadLimits::MAX_D1D;
      static constexpr int MQ1 = T_Q1D ? T_Q1D : DofQuadLimits::MAX_Q1D;

      MFEM_SHARED real_t smem[MQ1][MQ1];
      MFEM_SHARED real_t sB[MD1][MQ1], sG[MD1][MQ1];
      regs4d_t<VDIM, DIM, MQ1> r0, r1;

      LoadMatrix(D1D, Q1D, b, sB);
      LoadMatrix(D1D, Q1D, g, sG);

      LoadDofs2d(e, D1D, X, r0);
      Grad2d(D1D, Q1D, smem, sB, sG, r0, r1);

      mfem::tmop::foreach_y_thread(Q1D, [&](int qy)
      {
         mfem::tmop::foreach_x_thread(Q1D, [&](int qx)
         {
            const real_t *Wid = &W(0, 0);
            const real_t Jtr[4] =
            {
               r1[0][0][qy][qx], r1[1][0][qy][qx],
               r1[0][1][qy][qx], r1[1][1][qy][qx]
            };
            const real_t detJ = kernels::Det<2>(Jtr);
            const real_t alpha = std::pow(detJ / detW, 1. / 2);
            kernels::Set(DIM, DIM, alpha, Wid, &J(0, 0, qx, qy, e));
         });
      });
   });
}

MFEM_TMOP_REGISTER_KERNELS_1(TMOPTcIdealShapeUnitSize2D,
                             TMOP_TcIdealShapeUnitSize_2D);
MFEM_TMOP_ADD_SPECIALIZED_KERNELS_1(TMOPTcIdealShapeUnitSize2D);

MFEM_TMOP_REGISTER_KERNELS(TMOPTcIdealShapeGivenSize2D,
                           TMOP_TcIdealShapeGivenSize_2D);
MFEM_TMOP_ADD_SPECIALIZED_KERNELS(TMOPTcIdealShapeGivenSize2D);

template <>
bool TargetConstructor::ComputeAllElementTargets<2>(
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
   MFEM_VERIFY(fe.GetGeomType() == Geometry::SQUARE, "");
   const DenseMatrix &w = Geometries.GetGeomToPerfGeomJac(Geometry::SQUARE);
   const real_t detW = w.Det();
   const DofToQuad::Mode mode = DofToQuad::TENSOR;
   const DofToQuad &maps = fe.GetDofToQuad(ir, mode);
   const int d = maps.ndof, q = maps.nqpt;

   MFEM_VERIFY(d <= DeviceDofQuadLimits::Get().MAX_D1D, "");
   MFEM_VERIFY(q <= DeviceDofQuadLimits::Get().MAX_Q1D, "");

   static constexpr int DIM = 2;
   const auto W = Reshape(w.Read(), DIM, DIM);
   const auto *b = maps.B.Read(), *g = maps.G.Read();
   auto J = Reshape(Jtr.Write(), DIM, DIM, q, q, NE);

   switch (target_type)
   {
      case IDEAL_SHAPE_UNIT_SIZE: // Jtr(i) = Wideal;
      {
         TMOPTcIdealShapeUnitSize2D::Run(q, NE, W, J, q);
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
         MFEM_ASSERT(nodes->FESpace()->GetVDim() == 2, "");
         const auto X = Reshape(x.Read(), d, d, DIM, NE);

         TMOPTcIdealShapeGivenSize2D::Run(d, q, NE, detW, b, g, W, X, J, d, q);
         return true;
      }
      case GIVEN_SHAPE_AND_SIZE: return false;
      default:                   return false;
   }
   return false;
}

} // namespace mfem
