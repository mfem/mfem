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

#include "../bilininteg.hpp"
#include "../../general/forall.hpp"
#include "../ceed/integrators/diffusion/diffusion.hpp"

#include "./bilininteg_vecdiffusion_pa.hpp" // IWYU pragma: keep

// #include "bilininteg_vecdiffusion_kernels.hpp"
// #include "bilininteg_vecdiffusion_pa.hpp"

namespace mfem
{

VectorDiffusionIntegrator::VectorDiffusionIntegrator(const IntegrationRule *ir)
   : BilinearFormIntegrator(ir)
{
   // static Kernels kernels;
}

VectorDiffusionIntegrator::VectorDiffusionIntegrator(Coefficient &q)
   : VectorDiffusionIntegrator()
{
   Q = &q;
}

VectorDiffusionIntegrator::VectorDiffusionIntegrator(int vector_dimension)
   : VectorDiffusionIntegrator()
{
   vdim = vector_dimension;
}

VectorDiffusionIntegrator::VectorDiffusionIntegrator(Coefficient &q,
                                                     const IntegrationRule *ir)
   : VectorDiffusionIntegrator(ir)
{
   Q = &q;
}

VectorDiffusionIntegrator::VectorDiffusionIntegrator(Coefficient &q,
                                                     int vector_dimension)
   : VectorDiffusionIntegrator()
{
   Q = &q;
   vdim = vector_dimension;
}

VectorDiffusionIntegrator::VectorDiffusionIntegrator(VectorCoefficient &vq)
   : VectorDiffusionIntegrator()
{
   VQ = &vq;
   vdim = vq.GetVDim();
}

VectorDiffusionIntegrator::VectorDiffusionIntegrator(MatrixCoefficient &mq)
   : VectorDiffusionIntegrator()
{
   MQ = &mq;
   vdim = mq.GetVDim();
}

void VectorDiffusionIntegrator::AssemblePA(const FiniteElementSpace &fes)
{
   Mesh *mesh = fes.GetMesh();
   const FiniteElement &el = *fes.GetTypicalFE();
   const auto *ir = IntRule ? IntRule : &DiffusionIntegrator::GetRule(el, el);

   if (DeviceCanUseCeed())
   {
      delete ceedOp;
      const bool mixed =
         mesh->GetNumGeometries(mesh->Dimension()) > 1 || fes.IsVariableOrder();
      if (mixed) { ceedOp = new ceed::MixedPADiffusionIntegrator(*this, fes, Q); }
      else { ceedOp = new ceed::PADiffusionIntegrator(fes, *ir, Q); }
      return;
   }

   // If vdim is not set, set it to the space dimension
   vdim = (vdim == -1) ? fes.GetVDim() : vdim;
   MFEM_VERIFY(vdim == fes.GetVDim(), "vdim != fes.GetVDim()");

   const MemoryType mt = pa_mt == MemoryType::DEFAULT
                         ? Device::GetDeviceMemoryType()
                         : pa_mt;

   ne = fes.GetNE();
   dim = mesh->Dimension();
   sdim = mesh->SpaceDimension();
   const int nq = ir->GetNPoints();
   geom = mesh->GetGeometricFactors(*ir, GeometricFactors::JACOBIANS, mt);
   maps = &el.GetDofToQuad(*ir, DofToQuad::TENSOR);
   dofs1D = maps->ndof;
   quad1D = maps->nqpt;
   const int q1d = quad1D;

   if (!(dim == 2 || dim == 3)) { MFEM_ABORT("Dimension not supported."); }

   QuadratureSpace qs(*mesh, *ir);
   CoefficientVector coeff(qs, CoefficientStorage::FULL);

   if (Q)
   {
      coeff.Project(*Q);
   }
   else if (VQ)
   {
      coeff.Project(*VQ);
      MFEM_VERIFY(VQ->GetVDim() == vdim, "VQ vdim vs. vdim error");
   }
   else if (MQ)
   {
      coeff.ProjectTranspose(*MQ);
      MFEM_VERIFY(MQ->GetVDim() == vdim, "MQ dimension vs. vdim error");
      MFEM_VERIFY(coeff.Size() == (vdim*vdim) * ne * nq, "MQ size error");
   }
   else { coeff.SetConstant(1.0); }

   coeff_vdim = coeff.GetVDim();
   const bool scalar_coeff = coeff_vdim == 1;
   const bool vector_coeff = coeff_vdim == vdim;
   const bool matrix_coeff = coeff_vdim == vdim * vdim;
   MFEM_VERIFY(scalar_coeff + vector_coeff + matrix_coeff == 1, "");

   const int pa_size = dim * dim;
   pa_data.SetSize(nq * pa_size * vdim * (matrix_coeff ? dim : 1) * ne, mt);

   if (dim == 2 && sdim == 3)
   {
      MFEM_VERIFY(scalar_coeff, "");
      const int nc = vdim;
      const auto W = Reshape(ir->GetWeights().Read(), q1d, q1d);
      const auto J = Reshape(geom->J.Read(), q1d, q1d, sdim, dim, ne);
      const auto C = Reshape(coeff.Read(), coeff_vdim, q1d, q1d, ne);
      auto D = Reshape(pa_data.Write(), q1d, q1d, pa_size,
                       vdim * (matrix_coeff ? dim : 1), ne);

      mfem::forall_2D(ne, q1d, q1d, [=] MFEM_HOST_DEVICE(int e)
      {
         MFEM_FOREACH_THREAD(qy, y, q1d)
         {
            MFEM_FOREACH_THREAD(qx, x, q1d)
            {
               for (int i = 0; i < nc; ++i)
               {
                  const real_t wq = W(qx, qy);
                  const real_t J11 = J(qx, qy, 0, 0, e);
                  const real_t J21 = J(qx, qy, 1, 0, e);
                  const real_t J31 = J(qx, qy, 2, 0, e);
                  const real_t J12 = J(qx, qy, 0, 1, e);
                  const real_t J22 = J(qx, qy, 1, 1, e);
                  const real_t J32 = J(qx, qy, 2, 1, e);
                  const real_t E = J11*J11 + J21*J21 + J31*J31;
                  const real_t G = J12*J12 + J22*J22 + J32*J32;
                  const real_t F = J11*J12 + J21*J22 + J31*J32;
                  const real_t iw = 1.0 / sqrt(E*G - F*F);
                  const auto C0 = C(0, qx, qy, e);
                  const real_t alpha = wq * C0 * iw;
                  D(qx, qy, 0, i, e) =  alpha * G; // 1,1
                  D(qx, qy, 1, i, e) = -alpha * F; // 1,2
                  D(qx, qy, 2, i, e) = -alpha * F; // 2,1 == 1,2
                  D(qx, qy, 3, i, e) =  alpha * E; // 2,2
               }
            }
         }
      });
   }
   else if (dim == 2 && sdim == 2)
   {
      const int nc = vdim, cvdim = coeff_vdim;
      const auto W = Reshape(ir->GetWeights().Read(), q1d, q1d);
      const auto J = Reshape(geom->J.Read(), q1d, q1d, sdim, dim, ne);
      const auto C = Reshape(coeff.Read(), coeff_vdim, q1d, q1d, ne);
      auto DE = Reshape(pa_data.Write(), q1d, q1d, pa_size,
                        vdim * (matrix_coeff ? dim : 1), ne);
      mfem::forall_2D(ne, q1d, q1d, [=] MFEM_HOST_DEVICE(int e)
      {
         MFEM_FOREACH_THREAD(qy, y, q1d)
         {
            MFEM_FOREACH_THREAD(qx, x, q1d)
            {
               const real_t J11 = J(qx, qy, 0, 0, e);
               const real_t J21 = J(qx, qy, 1, 0, e);
               const real_t J12 = J(qx, qy, 0, 1, e);
               const real_t J22 = J(qx, qy, 1, 1, e);
               const real_t w_detJ = W(qx, qy) / ((J11*J22)-(J21*J12));
               const real_t D0 =  w_detJ * (J12*J12 + J22*J22);
               const real_t D1 = -w_detJ * (J12*J11 + J22*J21);
               const real_t D2 =  w_detJ * (J11*J11 + J21*J21);
               const int map[4] = {0, 2, 1, 3};

               for (int i = 0; i < (matrix_coeff ? cvdim : nc); ++i)
               {
                  const auto k = matrix_coeff ? map[i] : (vector_coeff ? i : 0);
                  const auto Cc = C(k, qx, qy, e);
                  DE(qx, qy, 0, i, e) = D0 * Cc;
                  DE(qx, qy, 1, i, e) = D1 * Cc;
                  DE(qx, qy, 2, i, e) = D1 * Cc;
                  DE(qx, qy, 3, i, e) = D2 * Cc;
               }
            }
         }
      });
   }
   else if (dim == 3 && sdim == 3)
   {
      const int nc = vdim, cvdim = coeff_vdim;
      const auto W = Reshape(ir->GetWeights().Read(), q1d, q1d, q1d);
      const auto J = Reshape(geom->J.Read(), q1d, q1d, q1d, sdim, dim, ne);
      const auto C = Reshape(coeff.Read(), coeff_vdim, q1d, q1d, q1d, ne);
      auto DE = Reshape(pa_data.Write(), q1d, q1d, q1d, pa_size,
                        vdim * (matrix_coeff ? dim : 1), ne);

      mfem::forall_3D(ne, q1d, q1d, q1d, [=] MFEM_HOST_DEVICE(int e)
      {
         MFEM_FOREACH_THREAD(qz, z, q1d)
         {
            MFEM_FOREACH_THREAD(qy, y, q1d)
            {
               MFEM_FOREACH_THREAD(qx, x, q1d)
               {
                  const real_t J11 = J(qx, qy, qz, 0, 0, e);
                  const real_t J21 = J(qx, qy, qz, 1, 0, e);
                  const real_t J31 = J(qx, qy, qz, 2, 0, e);
                  const real_t J12 = J(qx, qy, qz, 0, 1, e);
                  const real_t J22 = J(qx, qy, qz, 1, 1, e);
                  const real_t J32 = J(qx, qy, qz, 2, 1, e);
                  const real_t J13 = J(qx, qy, qz, 0, 2, e);
                  const real_t J23 = J(qx, qy, qz, 1, 2, e);
                  const real_t J33 = J(qx, qy, qz, 2, 2, e);
                  const real_t detJ = J11 * (J22 * J33 - J32 * J23) -
                                      J21 * (J12 * J33 - J32 * J13) +
                                      J31 * (J12 * J23 - J22 * J13);
                  const real_t c_detJ = W(qx, qy, qz) / detJ;
                  // adj(J)
                  const real_t A11 = (J22 * J33) - (J23 * J32);
                  const real_t A12 = (J32 * J13) - (J12 * J33);
                  const real_t A13 = (J12 * J23) - (J22 * J13);
                  const real_t A21 = (J31 * J23) - (J21 * J33);
                  const real_t A22 = (J11 * J33) - (J13 * J31);
                  const real_t A23 = (J21 * J13) - (J11 * J23);
                  const real_t A31 = (J21 * J32) - (J31 * J22);
                  const real_t A32 = (J31 * J12) - (J11 * J32);
                  const real_t A33 = (J11 * J22) - (J12 * J21);
                  // detJ J^{-1} J^{-T} = (1/detJ) adj(J) adj(J)^T
                  const real_t D11 = c_detJ * (A11*A11 + A12*A12 + A13*A13); // 1,1
                  const real_t D21 = c_detJ * (A11*A21 + A12*A22 + A13*A23); // 2,1
                  const real_t D31 = c_detJ * (A11*A31 + A12*A32 + A13*A33); // 3,1
                  const real_t D22 = c_detJ * (A21*A21 + A22*A22 + A23*A23); // 2,2
                  const real_t D32 = c_detJ * (A21*A31 + A22*A32 + A23*A33); // 3,2
                  const real_t D33 = c_detJ * (A31*A31 + A32*A32 + A33*A33); // 3,3
                  const int map[9] = {0, 3, 6, 1, 4, 7, 2, 5, 8};

                  for (int i = 0; i < (matrix_coeff ? cvdim : nc); ++i)
                  {
                     const auto k = matrix_coeff ? map[i] : vector_coeff ? i : 0;
                     const auto Ck = C(k, qx, qy, qz, e);
                     DE(qx, qy, qz, 0, i, e) = D11 * Ck;
                     DE(qx, qy, qz, 1, i, e) = D21 * Ck;
                     DE(qx, qy, qz, 2, i, e) = D31 * Ck;
                     DE(qx, qy, qz, 3, i, e) = D22 * Ck;
                     DE(qx, qy, qz, 4, i, e) = D32 * Ck;
                     DE(qx, qy, qz, 5, i, e) = D33 * Ck;
                  }
               }
            }
         }
      });
   }
   else
   {
      MFEM_ABORT("Unknown VectorDiffusionIntegrator::AssemblePA kernel for"
                 << " dim:" << dim << ", vdim:" << vdim << ", sdim:" << sdim);
   }
}

// PA Diffusion Apply kernel
void VectorDiffusionIntegrator::AddMultPA(const Vector &x, Vector &y) const
{
   // Use CEED backend if available
   if (DeviceCanUseCeed()) { return ceedOp->AddMult(x, y); }

   // Add the VectorDiffusionAddMultPA specializations
   static const auto vector_diffusion_kernel_specializations =
      (
         // 2D, SDIM = 2
         VectorDiffusionIntegrator::ApplyPAKernels::Specialization<2,2, 2,2>::Add(),
         VectorDiffusionIntegrator::ApplyPAKernels::Specialization<2,2, 3,3>::Add(),
         VectorDiffusionIntegrator::ApplyPAKernels::Specialization<2,2, 4,4>::Add(),
         VectorDiffusionIntegrator::ApplyPAKernels::Specialization<2,2, 5,5>::Add(),
         VectorDiffusionIntegrator::ApplyPAKernels::Specialization<2,2, 6,6>::Add(),
         VectorDiffusionIntegrator::ApplyPAKernels::Specialization<2,2, 7,7>::Add(),
         VectorDiffusionIntegrator::ApplyPAKernels::Specialization<2,2, 8,8>::Add(),
         VectorDiffusionIntegrator::ApplyPAKernels::Specialization<2,2, 9,9>::Add(),
         // 2D, SDIM = 3
         VectorDiffusionIntegrator::ApplyPAKernels::Specialization<2,3, 2,2>::Add(),
         VectorDiffusionIntegrator::ApplyPAKernels::Specialization<2,3, 3,3>::Add(),
         VectorDiffusionIntegrator::ApplyPAKernels::Specialization<2,3, 4,4>::Add(),
         VectorDiffusionIntegrator::ApplyPAKernels::Specialization<2,3, 5,5>::Add(),
         // 3D, SDIM = 3
         VectorDiffusionIntegrator::ApplyPAKernels::Specialization<3,3, 2,2>::Add(),
         VectorDiffusionIntegrator::ApplyPAKernels::Specialization<3,3, 2,3>::Add(),
         VectorDiffusionIntegrator::ApplyPAKernels::Specialization<3,3, 3,4>::Add(),
         VectorDiffusionIntegrator::ApplyPAKernels::Specialization<3,3, 4,5>::Add(),
         VectorDiffusionIntegrator::ApplyPAKernels::Specialization<3,3, 4,6>::Add(),
         VectorDiffusionIntegrator::ApplyPAKernels::Specialization<3,3, 5,6>::Add(),
         VectorDiffusionIntegrator::ApplyPAKernels::Specialization<3,3, 5,8>::Add(),
         VectorDiffusionIntegrator::ApplyPAKernels::Specialization<3,3, 6,7>::Add(),
         VectorDiffusionIntegrator::ApplyPAKernels::Specialization<3,3, 7,8>::Add(),
         VectorDiffusionIntegrator::ApplyPAKernels::Specialization<3,3, 8,9>::Add(),
         true);
   MFEM_CONTRACT_VAR(vector_diffusion_kernel_specializations);

   ApplyPAKernels::Run(dim, sdim, dofs1D, quad1D,
                       ne, coeff_vdim, maps->B, maps->G, pa_data, x, y,
                       sdim, dofs1D, quad1D);

}

template<int T_D1D = 0, int T_Q1D = 0>
static void PAVectorDiffusionDiagonal2D(const int NE,
                                        const Array<real_t> &b,
                                        const Array<real_t> &g,
                                        const Vector &d,
                                        Vector &y,
                                        const int d1d = 0,
                                        const int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   MFEM_VERIFY(D1D <= DeviceDofQuadLimits::Get().MAX_D1D, "");
   MFEM_VERIFY(Q1D <= DeviceDofQuadLimits::Get().MAX_Q1D, "");

   const auto B = Reshape(b.Read(), Q1D, D1D);
   const auto G = Reshape(g.Read(), Q1D, D1D);
   // note the different shape for D, this is a (symmetric) matrix so we only
   // store necessary entries
   MFEM_VERIFY(d.Size() == Q1D*Q1D*4*2*NE, "");
   const auto D = Reshape(d.Read(), Q1D*Q1D, /*3*/4, 2, NE);
   auto Y = Reshape(y.ReadWrite(), D1D, D1D, 2, NE);

   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      constexpr int MD1 = T_D1D ? T_D1D : DofQuadLimits::MAX_D1D;
      constexpr int MQ1 = T_Q1D ? T_Q1D : DofQuadLimits::MAX_Q1D;
      // gradphi \cdot Q \gradphi has four terms
      real_t QD0[MQ1][MD1];
      real_t QD1[MQ1][MD1];
      real_t QD2[MQ1][MD1];
      for (int qx = 0; qx < Q1D; ++qx)
      {
         for (int dy = 0; dy < D1D; ++dy)
         {
            QD0[qx][dy] = 0.0;
            QD1[qx][dy] = 0.0;
            QD2[qx][dy] = 0.0;
            for (int qy = 0; qy < Q1D; ++qy)
            {
               const int q = qx + qy * Q1D;
               const real_t D0 = D(q,0,0,e);
               const real_t D1 = D(q,1,0,e);
               const real_t D2 = D(q,3/*2*/,0,e); // size from 3 (symmetric) to 4 (dims x dims)
               QD0[qx][dy] += B(qy, dy) * B(qy, dy) * D0;
               QD1[qx][dy] += B(qy, dy) * G(qy, dy) * D1;
               QD2[qx][dy] += G(qy, dy) * G(qy, dy) * D2;
            }
         }
      }
      for (int dy = 0; dy < D1D; ++dy)
      {
         for (int dx = 0; dx < D1D; ++dx)
         {
            real_t temp = 0.0;
            for (int qx = 0; qx < Q1D; ++qx)
            {
               temp += G(qx, dx) * G(qx, dx) * QD0[qx][dy];
               temp += G(qx, dx) * B(qx, dx) * QD1[qx][dy];
               temp += B(qx, dx) * G(qx, dx) * QD1[qx][dy];
               temp += B(qx, dx) * B(qx, dx) * QD2[qx][dy];
            }
            Y(dx,dy,0,e) += temp;
            Y(dx,dy,1,e) += temp;
         }
      }
   });
}

template<int T_D1D = 0, int T_Q1D = 0>
static void PAVectorDiffusionDiagonal3D(const int NE,
                                        const Array<real_t> &b,
                                        const Array<real_t> &g,
                                        const Vector &d,
                                        Vector &y,
                                        const int d1d = 0,
                                        const int q1d = 0)
{
   constexpr int DIM = 3;
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   const int max_q1d = T_Q1D ? T_Q1D : DeviceDofQuadLimits::Get().MAX_Q1D;
   const int max_d1d = T_D1D ? T_D1D : DeviceDofQuadLimits::Get().MAX_D1D;
   MFEM_VERIFY(D1D <= max_d1d, "");
   MFEM_VERIFY(Q1D <= max_q1d, "");
   auto B = Reshape(b.Read(), Q1D, D1D);
   auto G = Reshape(g.Read(), Q1D, D1D);
   MFEM_VERIFY(d.Size() == Q1D*Q1D*Q1D*9*3*NE, "");
   auto Q = Reshape(d.Read(), Q1D*Q1D*Q1D, 9/*PA_SIZE:dims*dims*/, 3/*VDIM*/, NE);
   auto Y = Reshape(y.ReadWrite(), D1D, D1D, D1D, 3, NE);
   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      constexpr int MD1 = T_D1D ? T_D1D : DofQuadLimits::MAX_D1D;
      constexpr int MQ1 = T_Q1D ? T_Q1D : DofQuadLimits::MAX_Q1D;
      real_t QQD[MQ1][MQ1][MD1];
      real_t QDD[MQ1][MD1][MD1];
      for (int i = 0; i < DIM; ++i)
      {
         for (int j = 0; j < DIM; ++j)
         {
            // first tensor contraction, along z direction
            for (int qx = 0; qx < Q1D; ++qx)
            {
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  for (int dz = 0; dz < D1D; ++dz)
                  {
                     QQD[qx][qy][dz] = 0.0;
                     for (int qz = 0; qz < Q1D; ++qz)
                     {
                        const int q = qx + (qy + qz * Q1D) * Q1D;
                        const int k = j >= i ?
                                      3 - (3-i)*(2-i)/2 + j:
                                      3 - (3-j)*(2-j)/2 + i;
                        // using 6 symmetric values
                        const real_t O = Q(q,k,0,e);
                        const real_t Bz = B(qz,dz);
                        const real_t Gz = G(qz,dz);
                        const real_t L = i==2 ? Gz : Bz;
                        const real_t R = j==2 ? Gz : Bz;
                        QQD[qx][qy][dz] += L * O * R;
                     }
                  }
               }
            }
            // second tensor contraction, along y direction
            for (int qx = 0; qx < Q1D; ++qx)
            {
               for (int dz = 0; dz < D1D; ++dz)
               {
                  for (int dy = 0; dy < D1D; ++dy)
                  {
                     QDD[qx][dy][dz] = 0.0;
                     for (int qy = 0; qy < Q1D; ++qy)
                     {
                        const real_t By = B(qy,dy);
                        const real_t Gy = G(qy,dy);
                        const real_t L = i==1 ? Gy : By;
                        const real_t R = j==1 ? Gy : By;
                        QDD[qx][dy][dz] += L * QQD[qx][qy][dz] * R;
                     }
                  }
               }
            }
            // third tensor contraction, along x direction
            for (int dz = 0; dz < D1D; ++dz)
            {
               for (int dy = 0; dy < D1D; ++dy)
               {
                  for (int dx = 0; dx < D1D; ++dx)
                  {
                     real_t temp = 0.0;
                     for (int qx = 0; qx < Q1D; ++qx)
                     {
                        const real_t Bx = B(qx,dx);
                        const real_t Gx = G(qx,dx);
                        const real_t L = i==0 ? Gx : Bx;
                        const real_t R = j==0 ? Gx : Bx;
                        temp += L * QDD[qx][dy][dz] * R;
                     }
                     Y(dx, dy, dz, 0, e) += temp;
                     Y(dx, dy, dz, 1, e) += temp;
                     Y(dx, dy, dz, 2, e) += temp;
                  }
               }
            }
         }
      }
   });
}

static void PAVectorDiffusionAssembleDiagonal(const int dim,
                                              const int D1D,
                                              const int Q1D,
                                              const int NE,
                                              const Array<real_t> &B,
                                              const Array<real_t> &G,
                                              const Vector &op,
                                              Vector &y)
{
   if (dim == 2)
   {
      return PAVectorDiffusionDiagonal2D(NE, B, G, op, y, D1D, Q1D);
   }
   else if (dim == 3)
   {
      return PAVectorDiffusionDiagonal3D(NE, B, G, op, y, D1D, Q1D);
   }
   MFEM_ABORT("Dimension not implemented.");
}

void VectorDiffusionIntegrator::AssembleDiagonalPA(Vector &diag)
{
   if (DeviceCanUseCeed())
   {
      ceedOp->GetDiagonal(diag);
   }
   else
   {
      MFEM_VERIFY(!VQ && !MQ, "VQ and MQ not supported.");
      PAVectorDiffusionAssembleDiagonal(dim, dofs1D, quad1D, ne,
                                        maps->B, maps->G,
                                        pa_data, diag);
   }
}

/*
// PA Diffusion Apply kernel
void VectorDiffusionIntegrator::AddMultPA(const Vector &x, Vector &y) const
{
   if (DeviceCanUseCeed())
   {
      ceedOp->AddMult(x, y);
   }
   else
   {
      const int D1D = dofs1D;
      const int Q1D = quad1D;
      const Array<real_t> &B = maps->B;
      const Array<real_t> &G = maps->G;
      const Array<real_t> &Bt = maps->Bt;
      const Array<real_t> &Gt = maps->Gt;
      const Vector &D = pa_data;
      ApplyPAKernels::Run(dim, sdim, D1D, Q1D, ne, B, G, Bt, Gt, D, x, y, D1D,
                          Q1D, sdim);
   }
}

/// \cond DO_NOT_DOCUMENT

VectorDiffusionIntegrator::ApplyKernelType
VectorDiffusionIntegrator::ApplyPAKernels::Fallback(int DIM, int, int, int)
{
   if (DIM == 2) { return internal::PAVectorDiffusionApply2D; }
   else if (DIM == 3) { return internal::PAVectorDiffusionApply3D; }
   else { MFEM_ABORT(""); }
}

VectorDiffusionIntegrator::Kernels::Kernels()
{
   VectorDiffusionIntegrator::AddSpecialization<2, 3, 2, 2>();
   VectorDiffusionIntegrator::AddSpecialization<2, 3, 3, 3>();
   VectorDiffusionIntegrator::AddSpecialization<2, 3, 4, 4>();
   VectorDiffusionIntegrator::AddSpecialization<2, 3, 5, 5>();
}

/// \endcond DO_NOT_DOCUMENT
*/

} // namespace mfem
