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
#include "../ceed/integrators/mass/mass.hpp"

#include "./bilininteg_vecmass_pa.hpp" // IWYU pragma: keep

namespace mfem
{

void VectorMassIntegrator::AssemblePA(const FiniteElementSpace &fes)
{
   Mesh *mesh = fes.GetMesh();
   const FiniteElement &el = *fes.GetTypicalFE();
   ElementTransformation &Trans = *mesh->GetTypicalElementTransformation();
   const auto *ir = IntRule ? IntRule : &MassIntegrator::GetRule(el, el, Trans);

   if (DeviceCanUseCeed())
   {
      delete ceedOp;
      const bool mixed =
         mesh->GetNumGeometries(mesh->Dimension()) > 1 || fes.IsVariableOrder();
      if (mixed) { ceedOp = new ceed::MixedPAMassIntegrator(*this, fes, Q); }
      else { ceedOp = new ceed::PAMassIntegrator(fes, *ir, Q); }
      return;
   }

   // If vdim is not set, set it to the space dimension
   vdim = (vdim == -1) ? Trans.GetSpaceDim() : vdim;
   MFEM_VERIFY(vdim == fes.GetVDim(), "vdim != fes.GetVDim()");
   MFEM_VERIFY(vdim == mesh->Dimension(), "vdim != dim");

   const MemoryType mt = pa_mt == MemoryType::DEFAULT
                         ? Device::GetDeviceMemoryType()
                         : pa_mt;

   ne = mesh->GetNE();
   dim = mesh->Dimension();
   const int nq = ir->GetNPoints();
   const int sdim = mesh->SpaceDimension();
   geom = mesh->GetGeometricFactors(*ir, GeometricFactors::JACOBIANS, mt);
   maps = &el.GetDofToQuad(*ir, DofToQuad::TENSOR);
   dofs1D = maps->ndof;
   quad1D = maps->nqpt;
   const int q1d = quad1D;

   if (!(dim == 2 || dim == 3)) { MFEM_ABORT("Dimension not supported."); }

   QuadratureSpace qs(*mesh, *ir);
   CoefficientVector coeff(qs);

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
   const bool const_coeff = coeff_vdim == 1;
   const bool vector_coeff = coeff_vdim == vdim;
   const bool matrix_coeff = coeff_vdim == vdim * vdim;
   MFEM_VERIFY(const_coeff + vector_coeff + matrix_coeff == 1, "");

   pa_data.SetSize(coeff_vdim * nq * ne, mt);

   const auto w_r = ir->GetWeights().Read();

   if (dim == 2)
   {
      const auto W = Reshape(w_r, q1d, q1d);
      const auto C = Reshape(coeff.Read(), coeff_vdim, q1d, q1d, ne);
      const auto J = Reshape(geom->J.Read(), q1d, q1d, sdim, dim, ne);
      auto D = Reshape(pa_data.Write(), q1d, q1d, coeff_vdim, ne);

      mfem::forall_2D(ne, q1d, q1d, [=] MFEM_HOST_DEVICE(int e)
      {
         MFEM_FOREACH_THREAD(qy, y, q1d)
         {
            MFEM_FOREACH_THREAD(qx, x, q1d)
            {
               const real_t J11 = J(qx, qy, 0, 0, e), J12 = J(qx, qy, 1, 0, e);
               const real_t J21 = J(qx, qy, 0, 1, e), J22 = J(qx, qy, 1, 1, e);
               const real_t detJ = (J11 * J22) - (J21 * J12);
               const real_t w_det = W(qx, qy) * detJ;
               D(qx, qy, 0, e) = C(0, qx, qy, e) * w_det;
               if (const_coeff) { continue; }
               D(qx, qy, 1, e) = C(1, qx, qy, e) * w_det;
               if (vector_coeff) { continue; }
               assert(matrix_coeff);
               D(qx, qy, 2, e) = C(2, qx, qy, e) * w_det;
               D(qx, qy, 3, e) = C(3, qx, qy, e) * w_det;
            }
         }
      });
   }
   else if (dim == 3)
   {
      const auto W = Reshape(w_r, q1d, q1d, q1d);
      const auto C = Reshape(coeff.Read(), coeff_vdim, q1d, q1d, q1d, ne);
      const auto J = Reshape(geom->J.Read(), q1d, q1d, q1d, sdim, dim, ne);
      auto D = Reshape(pa_data.Write(), q1d, q1d, q1d, coeff_vdim, ne);

      mfem::forall_3D(ne, q1d, q1d, q1d, [=] MFEM_HOST_DEVICE(int e)
      {
         MFEM_FOREACH_THREAD(qz, z, q1d)
         {
            MFEM_FOREACH_THREAD(qy, y, q1d)
            {
               MFEM_FOREACH_THREAD(qx, x, q1d)
               {
                  const real_t J11 = J(qx, qy, qz, 0, 0, e),
                               J12 = J(qx, qy, qz, 0, 1, e),
                               J13 = J(qx, qy, qz, 0, 2, e);
                  const real_t J21 = J(qx, qy, qz, 1, 0, e),
                               J22 = J(qx, qy, qz, 1, 1, e),
                               J23 = J(qx, qy, qz, 1, 2, e);
                  const real_t J31 = J(qx, qy, qz, 2, 0, e),
                               J32 = J(qx, qy, qz, 2, 1, e),
                               J33 = J(qx, qy, qz, 2, 2, e);
                  const real_t detJ = J11 * (J22 * J33 - J32 * J23) -
                                      J21 * (J12 * J33 - J32 * J13) +
                                      J31 * (J12 * J23 - J22 * J13);
                  const real_t w_det = W(qx, qy, qz) * detJ;
                  D(qx, qy, qz, 0, e) = C(0, qx, qy, qz, e) * w_det;
                  if (const_coeff) { continue; }
                  D(qx, qy, qz, 1, e) = C(1, qx, qy, qz, e) * w_det;
                  D(qx, qy, qz, 2, e) = C(2, qx, qy, qz, e) * w_det;
                  if (vector_coeff) { continue; }
                  D(qx, qy, qz, 3, e) = C(3, qx, qy, qz, e) * w_det;
                  D(qx, qy, qz, 4, e) = C(4, qx, qy, qz, e) * w_det;
                  D(qx, qy, qz, 5, e) = C(5, qx, qy, qz, e) * w_det;
                  D(qx, qy, qz, 6, e) = C(6, qx, qy, qz, e) * w_det;
                  D(qx, qy, qz, 7, e) = C(7, qx, qy, qz, e) * w_det;
                  D(qx, qy, qz, 8, e) = C(8, qx, qy, qz, e) * w_det;
               }
            }
         }
      });
   }
   else
   {
      MFEM_ABORT("Unknown VectorMassIntegrator::AssemblePA kernel for"
                 << " dim:" << dim << ", vdim:" << vdim << ", sdim:" << sdim);
   }
}

void VectorMassIntegrator::AddMultPA(const Vector &x, Vector &y) const
{
   // Use CEED backend if available
   if (DeviceCanUseCeed()) { return ceedOp->AddMult(x, y); }

   // Add the VectorMassAddMultPA specializations
   static const auto vector_mass_kernel_specializations =
      ( // 2D
         VectorMassIntegrator::VectorMassAddMultPA::Specialization<2, 2,2>::Add(),
         VectorMassIntegrator::VectorMassAddMultPA::Specialization<2, 3,3>::Add(),
         VectorMassIntegrator::VectorMassAddMultPA::Specialization<2, 3,4>::Add(),
         VectorMassIntegrator::VectorMassAddMultPA::Specialization<2, 4,4>::Add(),
         VectorMassIntegrator::VectorMassAddMultPA::Specialization<2, 4,6>::Add(),
         VectorMassIntegrator::VectorMassAddMultPA::Specialization<2, 5,5>::Add(),
         VectorMassIntegrator::VectorMassAddMultPA::Specialization<2, 6,6>::Add(),
         VectorMassIntegrator::VectorMassAddMultPA::Specialization<2, 7,7>::Add(),
         VectorMassIntegrator::VectorMassAddMultPA::Specialization<2, 8,8>::Add(),
         VectorMassIntegrator::VectorMassAddMultPA::Specialization<2, 9,9>::Add(),
         // 3D
         VectorMassIntegrator::VectorMassAddMultPA::Specialization<3, 2,2>::Add(),
         VectorMassIntegrator::VectorMassAddMultPA::Specialization<3, 2,3>::Add(),
         VectorMassIntegrator::VectorMassAddMultPA::Specialization<3, 3,4>::Add(),
         VectorMassIntegrator::VectorMassAddMultPA::Specialization<3, 3,5>::Add(),
         VectorMassIntegrator::VectorMassAddMultPA::Specialization<3, 4,5>::Add(),
         VectorMassIntegrator::VectorMassAddMultPA::Specialization<3, 4,6>::Add(),
         VectorMassIntegrator::VectorMassAddMultPA::Specialization<3, 4,8>::Add(),
         VectorMassIntegrator::VectorMassAddMultPA::Specialization<3, 5,6>::Add(),
         VectorMassIntegrator::VectorMassAddMultPA::Specialization<3, 5,8>::Add(),
         VectorMassIntegrator::VectorMassAddMultPA::Specialization<3, 6,7>::Add(),
         VectorMassIntegrator::VectorMassAddMultPA::Specialization<3, 7,8>::Add(),
         VectorMassIntegrator::VectorMassAddMultPA::Specialization<3, 8,9>::Add(),
         true);
   MFEM_CONTRACT_VAR(vector_mass_kernel_specializations);

   VectorMassAddMultPA::Run(dim, dofs1D, quad1D,
                            ne, coeff_vdim, maps->B, pa_data, x, y,
                            dofs1D, quad1D);

}

template <const int T_D1D = 0, const int T_Q1D = 0>
static void PAVectorMassAssembleDiagonal2D(const int NE,
                                           const Array<real_t> &b,
                                           const Vector &pa_data, Vector &diag,
                                           const int d1d = 0, const int q1d = 0)
{
   constexpr int VDIM = 2;
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   MFEM_VERIFY(D1D <= DeviceDofQuadLimits::Get().MAX_D1D, "");
   MFEM_VERIFY(Q1D <= DeviceDofQuadLimits::Get().MAX_Q1D, "");
   const auto B = Reshape(b.Read(), Q1D, D1D);
   const auto D = Reshape(pa_data.Read(), Q1D, Q1D, NE);
   auto Y = Reshape(diag.ReadWrite(), D1D, D1D, VDIM, NE);

   mfem::forall(NE, [=] MFEM_HOST_DEVICE(int e)
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      constexpr int max_D1D = T_D1D ? T_D1D : DofQuadLimits::MAX_D1D;
      constexpr int max_Q1D = T_Q1D ? T_Q1D : DofQuadLimits::MAX_Q1D;

      real_t temp[max_Q1D][max_D1D];
      for (int qx = 0; qx < Q1D; ++qx)
      {
         for (int dy = 0; dy < D1D; ++dy)
         {
            temp[qx][dy] = 0.0;
            for (int qy = 0; qy < Q1D; ++qy)
            {
               temp[qx][dy] += B(qy, dy) * B(qy, dy) * D(qx, qy, e);
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
            Y(dx, dy, 0, e) = temp1;
            Y(dx, dy, 1, e) = temp1;
         }
      }
   });
}

template <const int T_D1D = 0, const int T_Q1D = 0>
static void PAVectorMassAssembleDiagonal3D(const int NE,
                                           const Array<real_t> &B_,
                                           const Vector &pa_data, Vector &diag,
                                           const int d1d = 0, const int q1d = 0)
{
   constexpr int VDIM = 3;
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   MFEM_VERIFY(D1D <= DeviceDofQuadLimits::Get().MAX_D1D, "");
   MFEM_VERIFY(Q1D <= DeviceDofQuadLimits::Get().MAX_Q1D, "");
   const auto B = Reshape(B_.Read(), Q1D, D1D);
   MFEM_VERIFY(pa_data.Size() == Q1D * Q1D * Q1D * NE, "pa_data size error");
   const auto D = Reshape(pa_data.Read(), Q1D, Q1D, Q1D, NE);
   auto Y = Reshape(diag.ReadWrite(), D1D, D1D, D1D, VDIM, NE);
   mfem::forall(NE, [=] MFEM_HOST_DEVICE(int e)
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      // the following variables are evaluated at compile time
      constexpr int max_D1D = T_D1D ? T_D1D : DofQuadLimits::MAX_D1D;
      constexpr int max_Q1D = T_Q1D ? T_Q1D : DofQuadLimits::MAX_Q1D;

      real_t temp[max_Q1D][max_Q1D][max_D1D];
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
                     B(qz, dz) * B(qz, dz) * D(qx, qy, qz, e);
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
               Y(dx, dy, dz, 0, e) = temp3;
               Y(dx, dy, dz, 1, e) = temp3;
               Y(dx, dy, dz, 2, e) = temp3;
            }
         }
      }
   });
}

static void PAVectorMassAssembleDiagonal(const int dim, const int D1D,
                                         const int Q1D, const int NE,
                                         const Array<real_t> &B,
                                         const Vector &pa_data,
                                         Vector &diag)
{
   if (dim == 2)
   {
      return PAVectorMassAssembleDiagonal2D(NE, B, pa_data, diag, D1D, Q1D);
   }
   else if (dim == 3)
   {
      return PAVectorMassAssembleDiagonal3D(NE, B, pa_data, diag, D1D, Q1D);
   }
   MFEM_ABORT("Dimension not implemented.");
}

void VectorMassIntegrator::AssembleDiagonalPA(Vector &diag)
{
   if (DeviceCanUseCeed()) { ceedOp->GetDiagonal(diag); }
   else
   {
      MFEM_VERIFY(coeff_vdim == 1, "coeff_vdim != 1");
      MFEM_VERIFY(!VQ && !MQ, "VQ and MQ not supported");
      PAVectorMassAssembleDiagonal(dim, dofs1D, quad1D, ne, maps->B, pa_data, diag);
   }
}

} // namespace mfem
