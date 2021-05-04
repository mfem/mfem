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

#include "../tmop.hpp"
#include "tmop_pa.hpp"
#include "../../general/forall.hpp"
#include "../../linalg/kernels.hpp"

namespace mfem
{

MFEM_REGISTER_TMOP_KERNELS(bool, DatcSize,
                           const int NE,
                           const int ncomp,
                           const int sizeidx,
                           const DenseMatrix w_, // Copy
                           const Array<double> &b_,
                           const Vector &x_,
                           DenseTensor &j_,
                           const int d1d,
                           const int q1d)
{
   MFEM_VERIFY(ncomp==1,"");
   constexpr int DIM = 3;
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   MFEM_VERIFY(D1D <= Q1D, "");

   const auto b = Reshape(b_.Read(), Q1D, D1D);
   const auto W = Reshape(w_.Read(), DIM,DIM);
   const auto X = Reshape(x_.Read(), D1D, D1D, D1D, ncomp, NE);
   auto J = Reshape(j_.Write(), DIM,DIM, Q1D,Q1D,Q1D, NE);

   const double infinity = std::numeric_limits<double>::infinity();
   MFEM_VERIFY(sizeidx == 0,"");
   MFEM_VERIFY(MFEM_CUDA_BLOCKS==256,"");

   MFEM_FORALL_3D(e, NE, Q1D, Q1D, Q1D,
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      constexpr int MQ1 = T_Q1D ? T_Q1D : T_MAX;
      constexpr int MD1 = T_D1D ? T_D1D : T_MAX;

      MFEM_SHARED double B[MQ1*MD1];
      MFEM_SHARED double DDD[MD1*MD1*MD1];
      MFEM_SHARED double DDQ[MD1*MD1*MQ1];
      MFEM_SHARED double DQQ[MD1*MQ1*MQ1];
      MFEM_SHARED double QQQ[MQ1*MQ1*MQ1];

      kernels::internal::LoadX<MD1>(e,D1D,sizeidx,X,DDD);

      double min;
      MFEM_SHARED double min_size[MFEM_CUDA_BLOCKS];
      DeviceTensor<3,double> M((double*)(min_size),D1D,D1D,D1D);
      const DeviceTensor<3,const double> D((double*)(DDD+sizeidx),D1D,D1D,D1D);
      MFEM_FOREACH_THREAD(t,x,MFEM_CUDA_BLOCKS) { min_size[t] = infinity; }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(dz,z,D1D)
      {
         MFEM_FOREACH_THREAD(dy,y,D1D)
         {
            MFEM_FOREACH_THREAD(dx,x,D1D)
            {
               M(dx,dy,dz) = D(dx,dy,dz);
            }
         }
      }
      MFEM_SYNC_THREAD;
      for (int wrk = MFEM_CUDA_BLOCKS >> 1; wrk > 0; wrk >>= 1)
      {
         MFEM_FOREACH_THREAD(t,x,MFEM_CUDA_BLOCKS)
         { if (t < wrk) { min_size[t] = fmin(min_size[t], min_size[t+wrk]); } }
         MFEM_SYNC_THREAD;
      }
      min = min_size[0];

      kernels::internal::LoadB<MD1,MQ1>(D1D,Q1D,b,B);
      kernels::internal::EvalX<MD1,MQ1>(D1D,Q1D,B,DDD,DDQ);
      kernels::internal::EvalY<MD1,MQ1>(D1D,Q1D,B,DDQ,DQQ);
      kernels::internal::EvalZ<MD1,MQ1>(D1D,Q1D,B,DQQ,QQQ);
      MFEM_FOREACH_THREAD(qx,x,Q1D)
      {
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(qz,z,Q1D)
            {
               double T;
               kernels::internal::PullEval<MQ1>(qx,qy,qz,QQQ,T);
               const double shape_par_vals = T;
               const double size = fmax(shape_par_vals, min);
               const double alpha = std::pow(size, 1.0/DIM);
               for (int i = 0; i < DIM; i++)
               {
                  for (int j = 0; j < DIM; j++)
                  {
                     J(i,j,qx,qy,qz,e) = alpha * W(i,j);
                  }
               }
            }
         }
      }
   });
   return true;
}

// PA.Jtr Size = (dim, dim, PA.ne*PA.nq);
bool DiscreteAdaptTC::ComputeElementTargetsPA(const FiniteElementSpace *pa_fes,
                                              const IntegrationRule *ir,
                                              DenseTensor &Jtr,
                                              const Vector &xe) const
{
   MFEM_VERIFY(target_type == IDEAL_SHAPE_GIVEN_SIZE ||
               target_type == GIVEN_SHAPE_AND_SIZE,"");

   const FiniteElementSpace *fes = tspec_fesv;

   if (!fes) { return false;}

   const FiniteElement &fe = *fes->GetFE(0);
   const DenseMatrix &W = Geometries.GetGeomToPerfGeomJac(fe.GetGeomType());
   const int DIM = W.Height();
   const int NE = fes->GetMesh()->GetNE();
   const DofToQuad::Mode mode = DofToQuad::TENSOR;
   const DofToQuad &maps = fe.GetDofToQuad(*ir, mode);
   const Array<double> &B = maps.B;
   const int D1D = maps.ndof;
   const int Q1D = maps.nqpt;

   const bool SizeKernel = sizeidx != -1;

   // Until it is not implemented, return on host
   if (skewidx != -1) { return false; }
   if (aspectratioidx != -1) { return false; }
   if (orientationidx != -1) { return false; }

   if (DIM == 3 && SizeKernel)
   {
      Vector tspec_e;
      const ElementDofOrdering ordering = ElementDofOrdering::LEXICOGRAPHIC;
      const Operator *R = fes->GetElementRestriction(ordering);
      MFEM_VERIFY(R,"");
      MFEM_VERIFY(R->Height() == NE*ncomp*D1D*D1D*D1D,"");
      tspec_e.SetSize(R->Height(), Device::GetDeviceMemoryType());
      tspec_e.UseDevice(true);
      tspec.UseDevice(true);
      R->Mult(tspec, tspec_e);
      const int id = (D1D << 4 ) | Q1D;
      MFEM_LAUNCH_TMOP_KERNEL(DatcSize,id,NE,ncomp,sizeidx,W,B,tspec_e,Jtr);
   }
   return false;
}

} // namespace mfem
