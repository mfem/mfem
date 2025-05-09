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
#include "../ceed/integrators/mass/mass.hpp"

#if __has_include("general/nvtx.hpp")
#undef NVTX_COLOR
#define NVTX_COLOR ::nvtx::kNvidia
#include "general/nvtx.hpp"
#else
#define dbg(...)
#endif

namespace mfem
{

void VectorMassIntegrator::AssemblePA(const FiniteElementSpace &fes)
{
   dbg();
   const MemoryType mt = pa_mt == MemoryType::DEFAULT
                         ? Device::GetDeviceMemoryType()
                         : pa_mt;
   // Assuming the same element type
   Mesh *mesh = fes.GetMesh();
   const FiniteElement &el = *fes.GetTypicalFE();
   ElementTransformation &T = *mesh->GetTypicalElementTransformation();
   const auto *ir = IntRule ? IntRule : &MassIntegrator::GetRule(el, el, T);
   if (DeviceCanUseCeed())
   {
      delete ceedOp;
      const bool mixed =
         mesh->GetNumGeometries(mesh->Dimension()) > 1 || fes.IsVariableOrder();
      if (mixed) { ceedOp = new ceed::MixedPAMassIntegrator(*this, fes, Q); }
      else { ceedOp = new ceed::PAMassIntegrator(fes, *ir, Q); }
      return;
   }

   dim = mesh->Dimension();
   ne = mesh->GetNE();
   dbg("ne: {}", ne);
   nq = ir->GetNPoints();
   dbg("nq: {}", nq);
   const int flags = GeometricFactors::COORDINATES | GeometricFactors::JACOBIANS;
   geom = mesh->GetGeometricFactors(*ir, flags, mt);
   maps = &el.GetDofToQuad(*ir, DofToQuad::TENSOR);
   dofs1D = maps->ndof;
   quad1D = maps->nqpt;

   QuadratureSpace qs(*mesh, *ir);
   CoefficientVector coeff(qs);

   if (Q) { coeff.Project(*Q); }
   else if (VQ) { coeff.Project(*VQ); }
   else if (MQ) { MFEM_ABORT("MatrixCoefficient not supported for partial assembly"); }
   else { coeff.SetConstant(1.0); }
   dbg("coeff VDim:{} Size:{}", coeff.GetVDim(), coeff.Size());

   const int coeff_vdim = coeff.GetVDim();
   MFEM_VERIFY(coeff_vdim == 1 || coeff_vdim == dim, "Incorrect coefficient size");

   dbg("\x1b[33mcoeff_vdim: {}", coeff_vdim);
   const bool const_coeff = coeff_vdim == 1;

   pa_data.SetSize(dim * nq * ne, mt);

   if (!(dim == 2 || dim == 3)) { MFEM_ABORT("Dimension not supported."); }

   const int NE = ne, NQ = nq;
   const auto W = ir->GetWeights().Read();
   const auto C = Reshape(coeff.Read(), coeff_vdim, NQ, NE);
   const auto J = Reshape(geom->J.Read(), NQ, dim, dim, NE);
   auto y = Reshape(pa_data.Write(), NQ, dim, NE);

   if (dim == 2)
   {

      mfem::forall(NE, [=] MFEM_HOST_DEVICE(int e)
      {
         for (int q = 0; q < NQ; ++q)
         {
            const real_t J11 = J(q, 0, 0, e), J12 = J(q, 1, 0, e);
            const real_t J21 = J(q, 0, 1, e), J22 = J(q, 1, 1, e);
            const real_t detJ = (J11 * J22) - (J21 * J12);
            const real_t w_det = W[q] * detJ;
            y(q, 0, e) = C(0, q, e) * w_det;
            y(q, 1, e) = C(const_coeff ? 0 : 1, q, e) * w_det;
         }
      });
   }

   if (dim == 3)
   {
      mfem::forall(NE, [=] MFEM_HOST_DEVICE(int e)
      {
         for (int q = 0; q < NQ; ++q)
         {
            const real_t J11 = J(q, 0, 0, e), J12 = J(q, 0, 1, e), J13 = J(q, 0, 2, e);
            const real_t J21 = J(q, 1, 0, e), J22 = J(q, 1, 1, e), J23 = J(q, 1, 2, e);
            const real_t J31 = J(q, 2, 0, e), J32 = J(q, 2, 1, e), J33 = J(q, 2, 2, e);
            const real_t detJ = J11 * (J22 * J33 - J32 * J23) -
                                J21 * (J12 * J33 - J32 * J13) +
                                J31 * (J12 * J23 - J22 * J13);
            const real_t w_det = W[q] * detJ;
            y(q, 0, e) = C(0, q, e) * w_det;
            y(q, 1, e) = C(const_coeff ? 0 : 1, q, e) * w_det;
            y(q, 2, e) = C(const_coeff ? 0 : 2, q, e) * w_det;
         }
      });
   }
}

template <const int T_D1D = 0, const int T_Q1D = 0>
static void PAVectorMassAssembleDiagonal2D(const int NE,
                                           const Array<real_t> &B_,
                                           const Array<real_t> &Bt_,
                                           const Vector &op_, Vector &diag_,
                                           const int d1d = 0, const int q1d = 0)
{
   dbg();
   constexpr int VDIM = 2;
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   MFEM_VERIFY(D1D <= DeviceDofQuadLimits::Get().MAX_D1D, "");
   MFEM_VERIFY(Q1D <= DeviceDofQuadLimits::Get().MAX_Q1D, "");
   auto B = Reshape(B_.Read(), Q1D, D1D);
   auto op = Reshape(op_.Read(), Q1D, Q1D, 2, NE);
   auto y = Reshape(diag_.ReadWrite(), D1D, D1D, VDIM, NE);
   mfem::forall(NE, [=] MFEM_HOST_DEVICE(int e)
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      constexpr int max_D1D = T_D1D ? T_D1D : DofQuadLimits::MAX_D1D;
      constexpr int max_Q1D = T_Q1D ? T_Q1D : DofQuadLimits::MAX_Q1D;

      real_t temp[max_Q1D][max_D1D];
      for (int c = 0; c < 2; ++c)
      {
         for (int qx = 0; qx < Q1D; ++qx)
         {
            for (int dy = 0; dy < D1D; ++dy)
            {
               temp[qx][dy] = 0.0;
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  temp[qx][dy] += B(qy, dy) * B(qy, dy) * op(qx, qy, c, e);
               }
            }
         }
         for (int dy = 0; dy < D1D; ++dy)
         {
            for (int dx = 0; dx < D1D; ++dx)
            {
               real_t temp1 = 0.0;
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  temp1 += B(qx, dx) * B(qx, dx) * temp[qx][dy];
               }
               y(dx, dy, c, e) = temp1;
            }
         }
      }
   });
}

template <const int T_D1D = 0, const int T_Q1D = 0>
static void PAVectorMassAssembleDiagonal3D(const int NE,
                                           const Array<real_t> &B_,
                                           const Array<real_t> &Bt_,
                                           const Vector &op_, Vector &diag_,
                                           const int d1d = 0, const int q1d = 0)
{
   dbg();
   constexpr int VDIM = 3;
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   MFEM_VERIFY(D1D <= DeviceDofQuadLimits::Get().MAX_D1D, "");
   MFEM_VERIFY(Q1D <= DeviceDofQuadLimits::Get().MAX_Q1D, "");
   auto B = Reshape(B_.Read(), Q1D, D1D);
   auto op = Reshape(op_.Read(), Q1D, Q1D, Q1D, 3, NE);
   auto y = Reshape(diag_.ReadWrite(), D1D, D1D, D1D, VDIM, NE);
   mfem::forall(NE, [=] MFEM_HOST_DEVICE(int e)
   {
      const int D1D = T_D1D ? T_D1D : d1d; // nvcc workaround
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      // the following variables are evaluated at compile time
      constexpr int max_D1D = T_D1D ? T_D1D : DofQuadLimits::MAX_D1D;
      constexpr int max_Q1D = T_Q1D ? T_Q1D : DofQuadLimits::MAX_Q1D;

      real_t temp[max_Q1D][max_Q1D][max_D1D];
      for (int c = 0; c < 2; ++c)
      {
         for (int qx = 0; qx < Q1D; ++qx)
         {
            for (int qy = 0; qy < Q1D; ++qy)
            {
               for (int dz = 0; dz < D1D; ++dz)
               {
                  temp[qx][qy][dz] = 0.0;
                  for (int qz = 0; qz < Q1D; ++qz)
                  {
                     temp[qx][qy][dz] +=
                        B(qz, dz) * B(qz, dz) * op(qx, qy, qz, c, e);
                  }
               }
            }
         }
         real_t temp2[max_Q1D][max_D1D][max_D1D];
         for (int qx = 0; qx < Q1D; ++qx)
         {
            for (int dz = 0; dz < D1D; ++dz)
            {
               for (int dy = 0; dy < D1D; ++dy)
               {
                  temp2[qx][dy][dz] = 0.0;
                  for (int qy = 0; qy < Q1D; ++qy)
                  {
                     temp2[qx][dy][dz] +=
                        B(qy, dy) * B(qy, dy) * temp[qx][qy][dz];
                  }
               }
            }
         }
         for (int dz = 0; dz < D1D; ++dz)
         {
            for (int dy = 0; dy < D1D; ++dy)
            {
               for (int dx = 0; dx < D1D; ++dx)
               {
                  real_t temp3 = 0.0;
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     temp3 += B(qx, dx) * B(qx, dx) * temp2[qx][dy][dz];
                  }
                  y(dx, dy, dz, c, e) = temp3;
               }
            }
         }
      }
   });
}

static void PAVectorMassAssembleDiagonal(const int dim, const int D1D,
                                         const int Q1D, const int NE,
                                         const Array<real_t> &B,
                                         const Array<real_t> &Bt,
                                         const Vector &op, Vector &y)
{
   dbg();
   if (dim == 2)
   {
      return PAVectorMassAssembleDiagonal2D(NE, B, Bt, op, y, D1D, Q1D);
   }
   else if (dim == 3)
   {
      return PAVectorMassAssembleDiagonal3D(NE, B, Bt, op, y, D1D, Q1D);
   }
   MFEM_ABORT("Dimension not implemented.");
}

void VectorMassIntegrator::AssembleDiagonalPA(Vector &diag)
{
   dbg();
   if (DeviceCanUseCeed()) { ceedOp->GetDiagonal(diag); }
   else
   {
      PAVectorMassAssembleDiagonal(dim, dofs1D, quad1D, ne, maps->B, maps->Bt,
                                   pa_data, diag);
   }
}

template <const int T_D1D = 0, const int T_Q1D = 0>
static void PAVectorMassApply2D(const int NE, const Array<real_t> &B_,
                                const Array<real_t> &Bt_, const Vector &op_,
                                const Vector &x_, Vector &y_, const int d1d = 0,
                                const int q1d = 0)
{
   dbg();
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   constexpr int VDIM = 2;
   MFEM_VERIFY(D1D <= DeviceDofQuadLimits::Get().MAX_D1D, "");
   MFEM_VERIFY(Q1D <= DeviceDofQuadLimits::Get().MAX_Q1D, "");
   auto B = Reshape(B_.Read(), Q1D, D1D);
   auto Bt = Reshape(Bt_.Read(), D1D, Q1D);
   auto op = Reshape(op_.Read(), Q1D, Q1D, VDIM, NE);
   auto x = Reshape(x_.Read(), D1D, D1D, VDIM, NE);
   auto y = Reshape(y_.ReadWrite(), D1D, D1D, VDIM, NE);
   mfem::forall(NE, [=] MFEM_HOST_DEVICE(int e)
   {
      const int D1D = T_D1D ? T_D1D : d1d; // nvcc workaround
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      // the following variables are evaluated at compile time
      constexpr int max_D1D = T_D1D ? T_D1D : DofQuadLimits::MAX_D1D;
      constexpr int max_Q1D = T_Q1D ? T_Q1D : DofQuadLimits::MAX_Q1D;
      real_t sol_xy[max_Q1D][max_Q1D];

      for (int c = 0; c < VDIM; ++c)
      {
         for (int qy = 0; qy < Q1D; ++qy)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               sol_xy[qy][qx] = 0.0;
            }
         }
         for (int dy = 0; dy < D1D; ++dy)
         {
            real_t sol_x[max_Q1D];
            for (int qy = 0; qy < Q1D; ++qy) { sol_x[qy] = 0.0; }
            for (int dx = 0; dx < D1D; ++dx)
            {
               const real_t s = x(dx, dy, c, e);
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  sol_x[qx] += B(qx, dx) * s;
               }
            }
            for (int qy = 0; qy < Q1D; ++qy)
            {
               const real_t d2q = B(qy, dy);
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  sol_xy[qy][qx] += d2q * sol_x[qx];
               }
            }
         }
         for (int qy = 0; qy < Q1D; ++qy)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               sol_xy[qy][qx] *= op(qx, qy, c, e);
            }
         }
         for (int qy = 0; qy < Q1D; ++qy)
         {
            real_t sol_x[max_D1D];
            for (int dx = 0; dx < D1D; ++dx) { sol_x[dx] = 0.0; }
            for (int qx = 0; qx < Q1D; ++qx)
            {
               const real_t s = sol_xy[qy][qx];
               for (int dx = 0; dx < D1D; ++dx)
               {
                  sol_x[dx] += Bt(dx, qx) * s;
               }
            }
            for (int dy = 0; dy < D1D; ++dy)
            {
               const real_t q2d = Bt(dy, qy);
               for (int dx = 0; dx < D1D; ++dx)
               {
                  y(dx, dy, c, e) += q2d * sol_x[dx];
               }
            }
         }
      }
   });
}

template <const int T_D1D = 0, const int T_Q1D = 0>
static void PAVectorMassApply3D(const int NE, const Array<real_t> &B_,
                                const Array<real_t> &Bt_, const Vector &op_,
                                const Vector &x_, Vector &y_, const int d1d = 0,
                                const int q1d = 0)
{
   dbg();
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   constexpr int VDIM = 3;
   MFEM_VERIFY(D1D <= DeviceDofQuadLimits::Get().MAX_D1D, "");
   MFEM_VERIFY(Q1D <= DeviceDofQuadLimits::Get().MAX_Q1D, "");
   auto B = Reshape(B_.Read(), Q1D, D1D);
   auto Bt = Reshape(Bt_.Read(), D1D, Q1D);
   auto op = Reshape(op_.Read(), Q1D, Q1D, Q1D, VDIM, NE);
   auto x = Reshape(x_.Read(), D1D, D1D, D1D, VDIM, NE);
   auto y = Reshape(y_.ReadWrite(), D1D, D1D, D1D, VDIM, NE);
   mfem::forall(NE, [=] MFEM_HOST_DEVICE(int e)
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      constexpr int max_D1D = T_D1D ? T_D1D : DofQuadLimits::MAX_D1D;
      constexpr int max_Q1D = T_Q1D ? T_Q1D : DofQuadLimits::MAX_Q1D;
      real_t sol_xyz[max_Q1D][max_Q1D][max_Q1D];
      for (int c = 0; c < VDIM; ++c)
      {
         for (int qz = 0; qz < Q1D; ++qz)
         {
            for (int qy = 0; qy < Q1D; ++qy)
            {
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  sol_xyz[qz][qy][qx] = 0.0;
               }
            }
         }
         for (int dz = 0; dz < D1D; ++dz)
         {
            real_t sol_xy[max_Q1D][max_Q1D];
            for (int qy = 0; qy < Q1D; ++qy)
            {
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  sol_xy[qy][qx] = 0.0;
               }
            }
            for (int dy = 0; dy < D1D; ++dy)
            {
               real_t sol_x[max_Q1D];
               for (int qx = 0; qx < Q1D; ++qx) { sol_x[qx] = 0; }
               for (int dx = 0; dx < D1D; ++dx)
               {
                  const real_t s = x(dx, dy, dz, c, e);
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     sol_x[qx] += B(qx, dx) * s;
                  }
               }
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  const real_t wy = B(qy, dy);
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     sol_xy[qy][qx] += wy * sol_x[qx];
                  }
               }
            }
            for (int qz = 0; qz < Q1D; ++qz)
            {
               const real_t wz = B(qz, dz);
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     sol_xyz[qz][qy][qx] += wz * sol_xy[qy][qx];
                  }
               }
            }
         }
         for (int qz = 0; qz < Q1D; ++qz)
         {
            for (int qy = 0; qy < Q1D; ++qy)
            {
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  sol_xyz[qz][qy][qx] *= op(qx, qy, qz, c, e);
               }
            }
         }
         for (int qz = 0; qz < Q1D; ++qz)
         {
            real_t sol_xy[max_D1D][max_D1D];
            for (int dy = 0; dy < D1D; ++dy)
            {
               for (int dx = 0; dx < D1D; ++dx)
               {
                  sol_xy[dy][dx] = 0;
               }
            }
            for (int qy = 0; qy < Q1D; ++qy)
            {
               real_t sol_x[max_D1D];
               for (int dx = 0; dx < D1D; ++dx) { sol_x[dx] = 0; }
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  const real_t s = sol_xyz[qz][qy][qx];
                  for (int dx = 0; dx < D1D; ++dx)
                  {
                     sol_x[dx] += Bt(dx, qx) * s;
                  }
               }
               for (int dy = 0; dy < D1D; ++dy)
               {
                  const real_t wy = Bt(dy, qy);
                  for (int dx = 0; dx < D1D; ++dx)
                  {
                     sol_xy[dy][dx] += wy * sol_x[dx];
                  }
               }
            }
            for (int dz = 0; dz < D1D; ++dz)
            {
               const real_t wz = Bt(dz, qz);
               for (int dy = 0; dy < D1D; ++dy)
               {
                  for (int dx = 0; dx < D1D; ++dx)
                  {
                     y(dx, dy, dz, c, e) += wz * sol_xy[dy][dx];
                  }
               }
            }
         }
      }
   });
}

static void PAVectorMassApply(const int dim, const int D1D, const int Q1D,
                              const int NE, const Array<real_t> &B,
                              const Array<real_t> &Bt, const Vector &op,
                              const Vector &x, Vector &y)
{
   dbg();
   if (dim == 2) { return PAVectorMassApply2D(NE, B, Bt, op, x, y, D1D, Q1D); }
   if (dim == 3) { return PAVectorMassApply3D(NE, B, Bt, op, x, y, D1D, Q1D); }
   MFEM_ABORT("Unknown kernel.");
}

void VectorMassIntegrator::AddMultPA(const Vector &x, Vector &y) const
{
   dbg();
   if (DeviceCanUseCeed()) { ceedOp->AddMult(x, y); }
   else
   {
      PAVectorMassApply(dim, dofs1D, quad1D, ne, maps->B, maps->Bt, pa_data, x, y);
   }
}

} // namespace mfem
