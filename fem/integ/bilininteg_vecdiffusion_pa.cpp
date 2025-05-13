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

#include "../../general/forall.hpp"
#include "../bilininteg.hpp"
#include "../ceed/integrators/diffusion/diffusion.hpp"

#include "./bilininteg_vecdiffusion_pa.hpp" // IWYU pragma: keep

#if __has_include("general/nvtx.hpp")
#undef NVTX_COLOR
#define NVTX_COLOR ::nvtx::kNvidia
#include "general/nvtx.hpp"
#else
#define dbg(...)
#endif

namespace mfem
{

void VectorDiffusionIntegrator::AssemblePA(const FiniteElementSpace &fes)
{

   // Assumes tensor-product elements
   Mesh *mesh = fes.GetMesh();
   const FiniteElement &el = *fes.GetTypicalFE();
   const auto *ir = IntRule ? IntRule : &DiffusionIntegrator::GetRule(el, el);

   if (DeviceCanUseCeed())
   {
      delete ceedOp;
      const bool mixed = mesh->GetNumGeometries(mesh->Dimension()) > 1 ||
                         fes.IsVariableOrder();
      if (mixed) { ceedOp = new ceed::MixedPADiffusionIntegrator(*this, fes, Q); }
      else { ceedOp = new ceed::PADiffusionIntegrator(fes, *ir, Q);      }
      return;
   }

   ne = fes.GetNE();
   dim = mesh->Dimension();
   sdim = mesh->SpaceDimension();
   const int nq = ir->GetNPoints();
   const int nd = el.GetDof();
   const int dims = el.GetDim();

   dbg("dim:{} vdim:{} fes.VDim():{} sdim:{} nq:{} nd:{} dims:{}",
       dim, vdim, fes.GetVDim(), sdim, nq, nd, dims);

   // If vdim is not set, set it to the space dimension
   dbg("\x1b[31mvdim:{} fes.VDim:{}", vdim, fes.GetVDim());
   if (vdim != -1) { MFEM_VERIFY(vdim == fes.GetVDim(), ""); }
   vdim = (vdim == -1) ? fes.GetVDim() : vdim;
   dbg("\x1b[33mvdim: {}", vdim);

   const MemoryType mt = pa_mt == MemoryType::DEFAULT
                         ? Device::GetDeviceMemoryType()
                         : pa_mt;
   geom = mesh->GetGeometricFactors(*ir, GeometricFactors::JACOBIANS, mt);
   maps = &el.GetDofToQuad(*ir, DofToQuad::TENSOR);
   dofs1D = maps->ndof;
   quad1D = maps->nqpt;
   const int q1d = quad1D;

   QuadratureSpace qs(*mesh, *ir);
   CoefficientVector coeff(qs, CoefficientStorage::FULL);

   if (Q)
   {
      dbg("\x1b[33mQ"); coeff.Project(*Q);
   }
   else if (VQ)
   {
      dbg("\x1b[33mVQ"); coeff.Project(*VQ);
      MFEM_VERIFY(VQ->GetVDim() == vdim, "VQ dimension vs. vdim error");
   }
   else if (MQ)
   {
      dbg("\x1b[33mMQ");
      coeff.ProjectTranspose(*MQ);
      MFEM_VERIFY(MQ->GetVDim() == vdim, "MQ dimension vs. vdim error");
      MFEM_VERIFY(coeff.Size() == (vdim*vdim) * ne * nq, "Coefficient size error");
   }
   else { dbg("\x1b[33m1.0"); coeff.SetConstant(1.0); }
   dbg("\x1b[33m[coeff] size:{} vdim:{}", coeff.Size(), coeff.GetVDim());

   const int pa_size = dims*dims;
   coeff_vdim = coeff.GetVDim();
   const bool const_coeff = coeff.Size() == 1;
   assert(!const_coeff);
   dbg("\x1b[33mconst_coeff:{}", const_coeff);

   const bool matrix_coeff = coeff_vdim == vdim*vdim;
   dbg("\x1b[33mcoeff_vdim:{} matrix_coeff:{}", coeff_vdim, matrix_coeff);

   if (dim == 2 && sdim == 3) // 🔥🔥🔥 PA data size
   {
      const int symmDims = (dims * (dims + 1)) / 2; // 1x1: 1, 2x2: 3, 3x3: 6
      dbg("symmDims:{}", symmDims);
      assert(coeff_vdim == 1);
      assert(coeff_vdim == symmDims);
      pa_data.SetSize(symmDims * nq * ne, mt);
   }
   else
   {
      pa_data.SetSize(vdim*pa_size * nq * ne * (matrix_coeff ? dim : 1), mt);
      dbg("pa_data size:{} = (vdim:{})x(pa_size:{}*dim)x{}x{}",
          vdim, pa_data.Size(), pa_size, nq, ne);
   }
   // MFEM_VERIFY(vdim == dim, "vdim != dim");

   const auto w_r = ir->GetWeights().Read();

   if (!(dim == 2 || dim == 3)) { MFEM_ABORT("Dimension not supported."); }


   if (dim == 2 && sdim == 3)
   {
      constexpr int DIM = 2;
      constexpr int SDIM = 3;
      const int NQ = quad1D*quad1D;
      auto J = Reshape(geom->J.Read(), NQ, SDIM, DIM, ne);
      auto D = Reshape(pa_data.Write(), NQ, SDIM, ne);

      const bool const_c = coeff.Size() == 1;
      const auto C = const_c
                     ? Reshape(coeff.Read(), 1, 1)
                     : Reshape(coeff.Read(), NQ, ne);

      mfem::forall(ne, [=] MFEM_HOST_DEVICE (int e)
      {
         for (int q = 0; q < NQ; ++q)
         {
            const real_t wq = w_r[q];
            const real_t J11 = J(q,0,0,e);
            const real_t J21 = J(q,1,0,e);
            const real_t J31 = J(q,2,0,e);
            const real_t J12 = J(q,0,1,e);
            const real_t J22 = J(q,1,1,e);
            const real_t J32 = J(q,2,1,e);
            const real_t E = J11*J11 + J21*J21 + J31*J31;
            const real_t G = J12*J12 + J22*J22 + J32*J32;
            const real_t F = J11*J12 + J21*J22 + J31*J32;
            const real_t iw = 1.0 / sqrt(E*G - F*F);
            const real_t C1 = const_c ? C(0,0) : C(q,e);
            const real_t alpha = wq * C1 * iw;
            D(q,0,e) =  alpha * G; // 1,1
            D(q,1,e) = -alpha * F; // 1,2
            D(q,2,e) =  alpha * E; // 2,2
         }
      });
   }
   else
   {
      // PAVectorDiffusionSetup(dim, quad1D, ne, w, j, coeff, d);
      if (!(dim == 2 || dim == 3)) { MFEM_ABORT("Dimension not supported."); }

      if (dim == 2)
      {
         dbg("dim:{}(==2!) sdim:{}(==2!) vdim:{}", dim, sdim, vdim);
         assert(dim == 2 && sdim == 2);

         const auto W = Reshape(w_r, q1d, q1d);
         const auto J = Reshape(geom->J.Read(), q1d, q1d, sdim, dim, ne);
         const auto C = Reshape(coeff.Read(), coeff_vdim, q1d, q1d, ne);
         auto DE = Reshape(pa_data.Write(), q1d, q1d, pa_size,
                           vdim * (matrix_coeff ? dim : 1),
                           ne);

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

                  if (coeff_vdim != dim*dim)
                  {
                     for (int c = 0; c < vdim; ++c)
                     {
                        const real_t Cc = C(coeff_vdim == vdim ? c : 0, qx, qy, e);
                        DE(qx, qy, 0, c, e) = D0 * Cc;
                        DE(qx, qy, 1, c, e) = D1 * Cc;
                        DE(qx, qy, 2, c, e) = D1 * Cc;
                        DE(qx, qy, 3, c, e) = D2 * Cc;
                     }
                  }
                  else if (coeff_vdim == 2*2) // Matrix coefficient
                  {
                     const real_t C0 = C(0, qx, qy, e);
                     const real_t C1 = C(1, qx, qy, e);
                     const real_t C2 = C(2, qx, qy, e);
                     const real_t C3 = C(3, qx, qy, e);

                     {
                        // k = 0
                        DE(qx, qy, 0, 0, e) = D0 * C0;
                        DE(qx, qy, 1, 0, e) = D1 * C0;
                        DE(qx, qy, 2, 0, e) = D1 * C0;
                        DE(qx, qy, 3, 0, e) = D2 * C0;
                     }
                     {
                        // k = 1
                        DE(qx, qy, 0, 1, e) = D0 * C3;
                        DE(qx, qy, 1, 1, e) = D1 * C3;
                        DE(qx, qy, 2, 1, e) = D1 * C3;
                        DE(qx, qy, 3, 1, e) = D2 * C3;
                     }
                     {
                        // k = 2
                        DE(qx, qy, 0, 2, e) = D0 * C1;
                        DE(qx, qy, 1, 2, e) = D1 * C1;
                        DE(qx, qy, 2, 2, e) = D1 * C1;
                        DE(qx, qy, 3, 2, e) = D2 * C1;
                     }
                     {
                        // k = 3
                        DE(qx, qy, 0, 3, e) = D0 * C2;
                        DE(qx, qy, 1, 3, e) = D1 * C2;
                        DE(qx, qy, 2, 3, e) = D1 * C2;
                        DE(qx, qy, 3, 3, e) = D2 * C2;
                     }
                  }
                  else { assert(false); }
               }
            }
         });
      }

      if (dim == 3)
      {
         dbg("dim:{}(==3!) sdim:{}(==3!) vdim:{}(3)", dim, sdim, vdim);
         assert(dim == 3 && sdim == 3);

         const auto W = Reshape(w_r, q1d, q1d, q1d);
         const auto J = Reshape(geom->J.Read(), q1d, q1d, q1d, sdim, dim, ne);
         const auto C = Reshape(coeff.Read(), coeff_vdim, q1d, q1d, q1d, ne);
         auto DE = Reshape(pa_data.Write(), q1d, q1d, q1d, pa_size,
                           vdim * (matrix_coeff ? dim : 1),
                           ne);

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

                     if (coeff_vdim != dim*dim)
                     {
                        for (int c = 0; c < vdim; ++c)
                        {
                           const real_t Ck = C(coeff_vdim == vdim ? c : 0, qx, qy, qz, e);
                           DE(qx, qy, qz, 0, c, e) = D11 * Ck;
                           DE(qx, qy, qz, 1, c, e) = D21 * Ck;
                           DE(qx, qy, qz, 2, c, e) = D31 * Ck;
                           DE(qx, qy, qz, 3, c, e) = D22 * Ck;
                           DE(qx, qy, qz, 4, c, e) = D32 * Ck;
                           DE(qx, qy, qz, 5, c, e) = D33 * Ck;
                        }
                     }
                     else if (coeff_vdim == dim*dim) // Matrix coefficient
                     {
                        for (int k = 0; k < 9; ++k)
                        {
                           const real_t Ck = C(k, qx, qy, qz, e);
                           DE(qx, qy, qz, 0, k, e) = D11 * Ck;
                           DE(qx, qy, qz, 1, k, e) = D21 * Ck;
                           DE(qx, qy, qz, 2, k, e) = D31 * Ck;
                           DE(qx, qy, qz, 3, k, e) = D21 * Ck;
                           DE(qx, qy, qz, 4, k, e) = D22 * Ck;
                           DE(qx, qy, qz, 5, k, e) = D32 * Ck;
                           DE(qx, qy, qz, 6, k, e) = D31 * Ck;
                           DE(qx, qy, qz, 7, k, e) = D32 * Ck;
                           DE(qx, qy, qz, 8, k, e) = D33 * Ck;
                        }
                     }
                     else { assert(false); }
                  }
               }
            }
         });
      }
   }
}

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

      if (dim == 2 && sdim == 3)
      {
         switch ((dofs1D << 4 ) | quad1D)
         {
            case 0x22:
               return internal::PAVectorDiffusionApply2D<2,2,3>(ne,coeff_vdim,B,G,Bt,Gt,D,x,y);
            case 0x33:
               return internal::PAVectorDiffusionApply2D<3,3,3>(ne,coeff_vdim,B,G,Bt,Gt,D,x,y);
            case 0x44:
               return internal::PAVectorDiffusionApply2D<4,4,3>(ne,coeff_vdim,B,G,Bt,Gt,D,x,y);
            case 0x55:
               return internal::PAVectorDiffusionApply2D<5,5,3>(ne,coeff_vdim,B,G,Bt,Gt,D,x,y);
            default:
               return internal::PAVectorDiffusionApply2D(ne,coeff_vdim,B,G,Bt,Gt,D,x,y,D1D,Q1D,
                                                         sdim);
         }
      }
      if (dim == 2 && sdim == 2)
      {
         dbg("dim:{} sdim:{} vdim:{}", dim, sdim, vdim);
         return internal::PAVectorDiffusionApply2D(ne, coeff_vdim,
                                                   B, G, Bt, Gt, D, x, y,
                                                   D1D, Q1D, vdim);
      }

      if (dim == 3 && sdim == 3)
      {
         return internal::PAVectorDiffusionApply3D(ne, coeff_vdim,
                                                   B, G, Bt, Gt, D, x, y,
                                                   D1D, Q1D);
      }

      MFEM_ABORT("Unknown kernel.");
   }
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
   auto B = Reshape(b.Read(), Q1D, D1D);
   auto G = Reshape(g.Read(), Q1D, D1D);
   // note the different shape for D, this is a (symmetric) matrix so we only
   // store necessary entries
   auto D = Reshape(d.Read(), Q1D*Q1D, 3, NE);
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
               const real_t D0 = D(q,0,e);
               const real_t D1 = D(q,1,e);
               const real_t D2 = D(q,2,e);
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
   auto Q = Reshape(d.Read(), Q1D*Q1D*Q1D, 6, NE);
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
                        const real_t O = Q(q,k,e);
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
      PAVectorDiffusionAssembleDiagonal(dim, dofs1D, quad1D, ne,
                                        maps->B, maps->G,
                                        pa_data, diag);
   }
}

} // namespace mfem
