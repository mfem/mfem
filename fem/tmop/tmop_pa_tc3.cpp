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

#include "../tmop.hpp"
#include "tmop_pa.hpp"
#include "../gridfunc.hpp"
#include "../../general/forall.hpp"
#include "../../linalg/kernels.hpp"

using namespace mfem;

namespace mfem
{

MFEM_REGISTER_TMOP_KERNELS(bool, TC_IDEAL_SHAPE_UNIT_SIZE_3D_KERNEL,
                           const int NE,
                           const DenseMatrix &w_,
                           DenseTensor &j_,
                           const int d1d,
                           const int q1d)
{
   constexpr int DIM = 3;

   const int Q1D = T_Q1D ? T_Q1D : q1d;

   const auto W = Reshape(w_.Read(), DIM,DIM);
   auto J = Reshape(j_.Write(), DIM,DIM, Q1D,Q1D,Q1D, NE);

   mfem::forall_3D(NE, Q1D, Q1D, Q1D, [=] MFEM_HOST_DEVICE (int e)
   {
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      MFEM_FOREACH_THREAD(qy,y,Q1D)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            MFEM_FOREACH_THREAD(qz,z,Q1D)
            {
               kernels::Set(DIM,DIM, 1.0, &W(0,0), &J(0,0,qx,qy,qz,e));
            }
         }
      }
   });
   return true;
}

MFEM_REGISTER_TMOP_KERNELS(bool, TC_IDEAL_SHAPE_GIVEN_SIZE_3D_KERNEL,
                           const int NE,
                           const Array<real_t> &b_,
                           const Array<real_t> &g_,
                           const DenseMatrix &w_,
                           const Vector &x_,
                           DenseTensor &j_,
                           const int d1d,
                           const int q1d)
{
   constexpr int DIM = 3;

   const real_t detW = w_.Det();
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   const auto b = Reshape(b_.Read(), Q1D, D1D);
   const auto g = Reshape(g_.Read(), Q1D, D1D);
   const auto W = Reshape(w_.Read(), DIM,DIM);
   const auto X = Reshape(x_.Read(), D1D, D1D, D1D, DIM, NE);
   auto J = Reshape(j_.Write(), DIM,DIM, Q1D,Q1D,Q1D, NE);

   mfem::forall_3D(NE, Q1D, Q1D, Q1D, [=] MFEM_HOST_DEVICE (int e)
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;

      constexpr int MQ1 = T_Q1D ? T_Q1D : T_MAX;
      constexpr int MD1 = T_D1D ? T_D1D : T_MAX;

      MFEM_SHARED real_t BG[2][MQ1*MD1];
      MFEM_SHARED real_t DDD[3][MD1*MD1*MD1];
      MFEM_SHARED real_t DDQ[6][MD1*MD1*MQ1];
      MFEM_SHARED real_t DQQ[9][MD1*MQ1*MQ1];
      MFEM_SHARED real_t QQQ[9][MQ1*MQ1*MQ1];

      kernels::internal::LoadX<MD1>(e,D1D,X,DDD);
      kernels::internal::LoadBG<MD1,MQ1>(D1D,Q1D,b,g,BG);

      kernels::internal::GradX<MD1,MQ1>(D1D,Q1D,BG,DDD,DDQ);
      kernels::internal::GradY<MD1,MQ1>(D1D,Q1D,BG,DDQ,DQQ);
      kernels::internal::GradZ<MD1,MQ1>(D1D,Q1D,BG,DQQ,QQQ);

      MFEM_FOREACH_THREAD(qz,z,Q1D)
      {
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               real_t Jtr[9];
               const real_t *Wid = &W(0,0);
               kernels::internal::PullGrad<MQ1>(Q1D,qx,qy,qz,QQQ,Jtr);
               const real_t detJ = kernels::Det<3>(Jtr);
               const real_t alpha = std::pow(detJ/detW,1./3);
               kernels::Set(DIM,DIM,alpha,Wid,&J(0,0,qx,qy,qz,e));
            }
         }
      }
   });
   return true;
}

template<> bool
TargetConstructor::ComputeAllElementTargets<3>(const FiniteElementSpace &fes,
                                               const IntegrationRule &ir,
                                               const Vector &,
                                               DenseTensor &Jtr) const
{
   MFEM_ASSERT(target_type == IDEAL_SHAPE_UNIT_SIZE || nodes != nullptr, "");
   const Mesh *mesh = fes.GetMesh();
   const int NE = mesh->GetNE();
   const int dim = mesh->Dimension();
   MFEM_VERIFY(mesh->GetNumGeometries(dim) <= 1,
               "mixed meshes are not supported");
   MFEM_VERIFY(!fes.IsVariableOrder(), "variable orders are not supported");
   const FiniteElement &fe = *fes.GetTypicalFE();
   MFEM_VERIFY(fe.GetGeomType() == Geometry::CUBE, "");
   const DenseMatrix &W = Geometries.GetGeomToPerfGeomJac(Geometry::CUBE);
   const DofToQuad::Mode mode = DofToQuad::TENSOR;
   const DofToQuad &maps = fe.GetDofToQuad(ir, mode);
   const Array<real_t> &B = maps.B;
   const Array<real_t> &G = maps.G;
   const int D1D = maps.ndof;
   const int Q1D = maps.nqpt;
   const int id = (D1D << 4 ) | Q1D;

   switch (target_type)
   {
      case IDEAL_SHAPE_UNIT_SIZE: // Jtr(i) = Wideal;
      {
         MFEM_LAUNCH_TMOP_KERNEL(TC_IDEAL_SHAPE_UNIT_SIZE_3D_KERNEL,
                                 id,NE,W,Jtr);
      }
      case IDEAL_SHAPE_EQUAL_SIZE: return false;
      case IDEAL_SHAPE_GIVEN_SIZE:
      {
         MFEM_VERIFY(nodes, "");
         const ElementDofOrdering ordering = ElementDofOrdering::LEXICOGRAPHIC;
         const Operator *R = fes.GetElementRestriction(ordering);
         Vector X(R->Height(), Device::GetDeviceMemoryType());
         X.UseDevice(true);
         R->Mult(*nodes, X);
         MFEM_ASSERT(nodes->FESpace()->GetVDim() == 3, "");
         MFEM_LAUNCH_TMOP_KERNEL(TC_IDEAL_SHAPE_GIVEN_SIZE_3D_KERNEL,
                                 id,NE,B,G,W,X,Jtr);
      }
      case GIVEN_SHAPE_AND_SIZE: return false;
      default: return false;
   }
   return false;
}

} // namespace mfem
