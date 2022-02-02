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
#include "../linearform.hpp"
#include "../../general/forall.hpp"
#include "../../linalg/kernels.hpp"
#include "../quadinterpolator.hpp"
#include "../qinterp/dispatch.hpp"
#include "../qinterp/grad.hpp"

namespace mfem
{

MFEM_REGISTER_TMOP_KERNELS(void, AddMultPA_Kernel_SF_2D,
                           const double surf_fit_normal,
                           const Vector &surf_fit_gf,
                           const Vector &surf_fit_mask,
                           const Vector &gradq,
                           const Vector &c0sf_,
                           const int NE,
                           Vector &y_,
                           const int d1d,
                           const int q1d)
{
   const bool const_c0 = c0sf_.Size() == 1;
   constexpr int DIM = 2;

   constexpr int NBZ = 1;

   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   const auto C0SF = const_c0 ?
                   Reshape(c0sf_.Read(), 1, 1, 1) :
                   Reshape(c0sf_.Read(), Q1D, Q1D, NE);
   const auto SFG = Reshape(surf_fit_gf.Read(), D1D, D1D, NE);
   const auto SFM = Reshape(surf_fit_mask.Read(), D1D, D1D, NE);
   const auto G = Reshape(gradq.Read(), Q1D, Q1D, DIM, NE);

   auto Y = Reshape(y_.ReadWrite(), D1D, D1D, DIM, NE);

   MFEM_FORALL_2D(e, NE, Q1D, Q1D, NBZ,
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      constexpr int NBZ = 1;
      constexpr int MQ1 = T_Q1D ? T_Q1D : T_MAX;
      constexpr int MD1 = T_D1D ? T_D1D : T_MAX;

      MFEM_FOREACH_THREAD(qy,y,Q1D)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            const double coeff = const_c0 ? C0SF(0, 0, 0) :
                                 C0SF(qx, qy, e);
            for (int d = 0; d < DIM; d++) {
               Y(qx, qy, d, e) += 2 * coeff * SFM(qx, qy, e) *
                              SFG(qx, qy, e) * G(qx, qy, d, e);
            }
         }
      }
      MFEM_SYNC_THREAD;
   });
}

void TMOP_Integrator::AddMultPA_SF_2D(const Vector &X, Vector &Y) const
{
   const int N = PA.ne;
   const int D1D = PA.maps_surf->ndof;
   const int Q1D = PA.maps_surf->nqpt;
   const int id = (D1D << 4 ) | Q1D;
   const double sn = surf_fit_normal;
   const Vector &SFG = PA.SFG;
   const Vector &SFM = PA.SFM;
   const Array<double> &W   = PA.irsf->GetWeights();
   const Array<double> &B   = PA.maps_surf->B;
   MFEM_VERIFY(PA.maps_surf->ndof == D1D, "");
   MFEM_VERIFY(PA.maps_surf->nqpt == Q1D, "");
   const Vector &C0SF = PA.C0sf;
   const Array<double> &G = PA.maps_surf->G;

   PA.fessf->GetMesh()->DeleteGeometricFactors();
   QuadratureInterpolator qi = QuadratureInterpolator(*PA.fessf, *PA.irsf);

   const int vdim = PA.fes->GetVDim();
   const int dim = PA.fes->GetFE(0)->GetDim();
   const int nqp = PA.irsf->GetNPoints();
   Vector jacobians(nqp*N*dim*dim);
   using namespace internal::quadrature_interpolator;
   const DofToQuad &maps = PA.fes->GetFE(0)->GetDofToQuad(*PA.irsf, DofToQuad::TENSOR);
   TensorDerivatives<QVectorLayout::byNODES>(N, vdim, maps, X, jacobians);
   constexpr QVectorLayout L = QVectorLayout::byNODES;
   constexpr bool P = true; // GRAD_PHYS
   Vector gradq(nqp*N*dim);
   const double *J = jacobians.Read();

   constexpr int MD = MAX_D1D;
   constexpr int MQ = MAX_Q1D;
   MFEM_VERIFY(D1D <= MD, "Orders higher than " << MD-1
               << " are not supported!");
   MFEM_VERIFY(Q1D <= MQ, "Quadrature rules with more than "
               << MQ << " 1D points are not supported!");
   Derivatives2D<L,P,0,0,0,0,MD,MQ>(N,B,G,J,SFG,gradq,1,D1D,Q1D);

   const bool const_c0 = C0SF.Size() == 1;
   MFEM_LAUNCH_TMOP_KERNEL(AddMultPA_Kernel_SF_2D,id,sn,SFG,SFM,gradq,C0SF,N,Y);
}

} // namespace mfem
