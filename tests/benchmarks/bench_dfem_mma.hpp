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
#pragma once

#include "fem/bilininteg.hpp"
#include <fem/quadinterpolator.hpp>
#include "general/forall.hpp"
#include "linalg/dtensor.hpp"
#include "linalg/kernels.hpp"

using namespace mfem;

/// PADiffMmaIntegrator ///////////////////////////////////////////////////////
struct PADiffMmaIntegrator : public BilinearFormIntegrator
{
   const FiniteElementSpace *fes;
   const real_t *B, *G, *DX;
   int ne, d1d, q1d;
   Vector J0, dx;

public: // for nvcc
   //////////////////////////////////////////////////////////////////
   template <int T_Q1D = 0>
   static void PADiffMmaMult(const int ne, const int d1d,
                             const real_t *b, const real_t *g,
                             const real_t *dx, const real_t *xe,
                             real_t *ye,
                             const int q1d)
   {
      constexpr int DIM = 3, VDIM = 1;

      const auto XE = Reshape(xe, d1d, d1d, d1d, VDIM, ne);
      auto YE = Reshape(ye, d1d, d1d, d1d, VDIM, ne);

      mfem::forall_3D<T_Q1D*T_Q1D*T_Q1D>(ne, q1d, q1d, q1d,
                                         [=] MFEM_HOST_DEVICE(int e)
      {
         constexpr int MQ1 = T_Q1D;

         MFEM_SHARED real_t sm0[MQ1][MQ1][MQ1][3];
         MFEM_SHARED real_t sm1[MQ1][MQ1][MQ1][3];
         MFEM_SHARED real_t sB[MQ1][MQ1];
         MFEM_SHARED real_t sG[MQ1][MQ1];

      });
   }

   using PADiffMmaKernelType = decltype(&PADiffMmaMult<>);
   MFEM_REGISTER_KERNELS(PADiffMmaKernels, PADiffMmaKernelType, (int));

public:
   PADiffMmaIntegrator()
   {
      // PADiffMmaKernels::Specialization<3>::Add();  // 1
      PADiffMmaKernels::Specialization<4>::Add();  // 2
      // PADiffMmaKernels::Specialization<5>::Add();  // 3
      PADiffMmaKernels::Specialization<6>::Add();  // 4
      // PADiffMmaKernels::Specialization<7>::Add();  // 5
      PADiffMmaKernels::Specialization<8>::Add();  // 6
   }

   void AssemblePA(const FiniteElementSpace &fespace) override
   {
      NVTX();
      fes = &fespace;
      auto *mesh = fes->GetMesh();
      const int DIM = mesh->Dimension();
      ne = mesh->GetNE();
      const auto p = fes->GetFE(0)->GetOrder();
      const auto q = 2 * p + mesh->GetElementTransformation(0)->OrderW();
      const auto type = mesh->GetElementBaseGeometry(0);
      const IntegrationRule &ir = IntRules.Get(type, q);
      const int NQPT = ir.GetNPoints();
      d1d = p + 1;
      q1d = IntRules.Get(Geometry::SEGMENT, ir.GetOrder()).GetNPoints();
      MFEM_VERIFY(NQPT == q1d * q1d * q1d, "");
      const DofToQuad *maps =
         &fes->GetFE(0)->GetDofToQuad(ir, DofToQuad::TENSOR);
      const GridFunction *nodes = (mesh->EnsureNodes(), mesh->GetNodes());
      const FiniteElementSpace *nfes = nodes->FESpace();
      const int nVDIM = nfes->GetVDim();
      dx.SetSize(nVDIM * DIM * NQPT * ne, Device::GetDeviceMemoryType());
      J0.SetSize(nVDIM * DIM * NQPT * ne, Device::GetDeviceMemoryType());
      dx.UseDevice(true), J0.UseDevice(true);
      B = maps->B.Read(), G = maps->G.Read(), DX = dx.Read();

      const Operator *NR =
         nfes->GetElementRestriction(ElementDofOrdering::LEXICOGRAPHIC);
      const QuadratureInterpolator *nqi = nfes->GetQuadratureInterpolator(ir);
      nqi->SetOutputLayout(QVectorLayout::byVDIM);
      const int nd = nfes->GetFE(0)->GetDof();
      Vector xe(nVDIM * nd * ne, Device::GetDeviceMemoryType());
      NR->Mult(*nodes, (xe.UseDevice(true), xe));
      nqi->Derivatives(xe, J0);

      const int Q1D = q1d;
      const auto w_r = ir.GetWeights().Read();
      const auto W = Reshape(w_r, q1d, q1d, q1d);
      const auto J = Reshape(J0.Read(), 3, 3, q1d, q1d, q1d, ne);
      auto DX_w = Reshape(dx.Write(), 3, 3, q1d, q1d, q1d, ne);

      mfem::forall_3D(ne, Q1D, Q1D, Q1D,[=] MFEM_HOST_DEVICE(int e)
      {
         MFEM_FOREACH_THREAD_DIRECT(qz, z, Q1D)
         {
            MFEM_FOREACH_THREAD_DIRECT(qy, y, Q1D)
            {
               MFEM_FOREACH_THREAD_DIRECT(qx, x, Q1D)
               {
                  const real_t w = W(qx, qy, qz);
                  const real_t *Jtr = &J(0, 0, qx, qy, qz, e);
                  const real_t detJ = kernels::Det<3>(Jtr);
                  const real_t wd = w * detJ;
                  const real_t D[9] = { wd, 0.0, 0.0,
                                        0.0, wd, 0.0,
                                        0.0, 0.0, wd
                                      };
                  real_t Jrt[9], A[9];
                  kernels::CalcInverse<3>(Jtr, Jrt);
                  kernels::MultABt(3, 3, 3, D, Jrt, A);
                  kernels::Mult(3, 3, 3, A, Jrt, &DX_w(0, 0, qz, qy, qx, e));
               }
            }
         }
         MFEM_SYNC_THREAD;
      });
   }

   void AddMultPA(const Vector &x, Vector &y) const override
   {
      db1("\x1b[32md1d:{} q1d:{}", d1d, q1d);
      PADiffMmaKernels::Run(q1d,
                            ne, d1d, B, G, DX, x.Read(), y.ReadWrite(),
                            q1d);
   }
};
template <int Q1D>
PADiffMmaIntegrator::PADiffMmaKernelType
PADiffMmaIntegrator::PADiffMmaKernels::Kernel()
{
   db1("Q1D:{}", Q1D);
   return PADiffMmaMult<Q1D>;
}

PADiffMmaIntegrator::PADiffMmaKernelType
PADiffMmaIntegrator::PADiffMmaKernels::Fallback(int q1d)
{
   dbg("\x1b[33mFallback d1d:{} q1d:{}", q1d);
   return PADiffMmaMult;
}
