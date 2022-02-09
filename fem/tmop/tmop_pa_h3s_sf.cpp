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
MFEM_REGISTER_TMOP_KERNELS(void, SetupGradPA_SF_3D,
                           const double surf_fit_normal,
                           const Vector &surf_fit_mask,
                           const Vector &surf_fit_gf,
                           const Vector &gradq,
                           const Vector &hessq,
                           const Vector &c0sf_,
                           const int NE,
                           Vector &h0_,
                           const int d1d,
                           const int q1d)
{
   constexpr int DIM = 3;
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   const bool const_c0 = c0sf_.Size() == 1;
   const auto C0SF = const_c0 ?
                     Reshape(c0sf_.Read(), 1, 1, 1, 1) :
                     Reshape(c0sf_.Read(), Q1D, Q1D, Q1D, NE);
   const auto SFG = Reshape(surf_fit_gf.Read(), D1D, D1D, D1D, NE);
   const auto SFM = Reshape(surf_fit_mask.Read(), D1D, D1D, D1D, NE);
   const auto G = Reshape(gradq.Read(), Q1D, Q1D, Q1D, DIM, NE);
   const auto Hin = Reshape(hessq.Read(), DIM, DIM, Q1D, Q1D, Q1D, NE);
   auto H0 = Reshape(h0_.Write(), DIM, DIM, Q1D, Q1D, Q1D, NE);

   MFEM_FORALL_3D(e, NE, Q1D, Q1D, Q1D,
   {
      const int Q1D = T_Q1D ? T_Q1D : q1d;

      MFEM_FOREACH_THREAD(qz,z,Q1D)
      {
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               const double coeff0 = const_c0 ? C0SF(0,0,0,0) : C0SF(qx,qy,qz,e);

               for (int i = 0; i < DIM; i++)
               {
                  for (int j = 0; j <= i; j++)
                  {
                     const double entry = coeff0 * 2 * surf_fit_normal *
                     SFM(qx, qy, qz, e) *
                     (Hin(i, j, qx, qy, qz, e) *
                      SFG(qx, qy, qz, e) +
                      G(qx, qy, qz, j, e) *
                      G(qx, qy, qz, i, e));
                     H0(i, j, qx, qy, qz, e) = entry;
                     if (j != i)
                     {
                        H0(j, i, qx, qy, qz, e) = entry;
                     }
                  }
               }
            }
         }
      }
   });
}

void TMOP_Integrator::AssembleGradPA_SF_3D(const Vector &X) const
{
   MFEM_CONTRACT_VAR(X);
   const int N = PA.ne;
   const int D1D = PA.maps_surf->ndof;
   const int Q1D = PA.maps_surf->nqpt;
   const int id = (D1D << 4 ) | Q1D;
   const double sn = surf_fit_normal;
   const Vector &SFG = PA.SFG;
   const Vector &SFM = PA.SFM;
   const Array<double> &B   = PA.maps_surf->B;
   MFEM_VERIFY(PA.maps_surf->ndof == D1D, "");
   MFEM_VERIFY(PA.maps_surf->nqpt == Q1D, "");
   const Vector &C0SF = PA.C0sf;
   const Array<double> &G = PA.maps_surf->G;
   Vector &gradq = PA.Gsf;

   QuadratureInterpolator qi = QuadratureInterpolator(*PA.fessf, *PA.irsf);

   const int dim = PA.fes->GetFE(0)->GetDim();
   const int vdim = PA.fes->GetVDim();
   const int nqp = PA.irsf->GetNPoints();
   Vector jacobians(nqp*N*dim*dim);
   using namespace internal::quadrature_interpolator;
   const DofToQuad &maps = PA.fes->GetFE(0)->GetDofToQuad(*PA.irsf,
                                                          DofToQuad::TENSOR);
   TensorDerivatives<QVectorLayout::byNODES>(N, vdim, maps, X, jacobians);
   constexpr QVectorLayout L = QVectorLayout::byNODES;
   constexpr bool P = true; // GRAD_PHYS
   //Vector gradq(nqp*N*dim);
   const double *J = jacobians.Read();

   Vector hessq(nqp*N*dim*dim);
   Vector &Hsf = PA.Hsf;

   constexpr int MD = MAX_D1D;
   constexpr int MQ = MAX_Q1D;
   MFEM_VERIFY(D1D <= MD, "Orders higher than " << MD-1
               << " are not supported!");
   MFEM_VERIFY(Q1D <= MQ, "Quadrature rules with more than "
               << MQ << " 1D points are not supported!");
   //Derivatives3D<L,P,0,0,0,MD,MQ>(N,B,G,J,SFG,gradq,1,D1D,Q1D);

   Derivatives3D<QVectorLayout::byVDIM,P,0,0,0,MD,MQ>(N,B,G,J,gradq.GetData(),
                                                      hessq,dim,D1D,Q1D);

   MFEM_LAUNCH_TMOP_KERNEL(SetupGradPA_SF_3D,id,sn,SFM,SFG,gradq,hessq,C0SF,N,Hsf);
}

} // namespace mfem
