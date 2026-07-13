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

#include "../ceed/integrators/nlconvection/nlconvection.hpp"
#include "./nonlininteg_vecconvection_pa.hpp" // IWYU pragma: keep

namespace mfem
{

void VectorConvectionNLFIntegrator::AssemblePA(const FiniteElementSpace &fes)
{
   MFEM_ASSERT(fes.GetOrdering() == Ordering::byNODES,
               "PA Only supports Ordering::byNODES!");
   Mesh *mesh = fes.GetMesh();
   const FiniteElement &el = *fes.GetTypicalFE();
   ElementTransformation &Tr = *mesh->GetTypicalElementTransformation();
   const IntegrationRule *ir = IntRule ? IntRule : &GetRule(el, Tr);

   if (DeviceCanUseCeed())
   {
      delete ceedOp;
      const bool mixed = mesh->GetNumGeometries(mesh->Dimension()) > 1 ||
                         fes.IsVariableOrder();
      if (mixed)
      {
         ceedOp = new ceed::MixedPAVectorConvectionNLIntegrator(*this, fes, Q);
      }
      else
      {
         ceedOp = new ceed::PAVectorConvectionNLFIntegrator(fes, *ir, Q);
      }
      return;
   }

   ne = mesh->GetNE();
   nq = ir->GetNPoints();
   dim = mesh->Dimension();
   MFEM_VERIFY(dim == 2 || dim == 3, "Dimension not supported");

   const MemoryType mt = pa_mt == MemoryType::DEFAULT
                         ? Device::GetDeviceMemoryType()
                         : pa_mt;
   pa_adj.SetSize(ne * nq * dim * dim, mt);
   geom = mesh->GetGeometricFactors(*ir, GeometricFactors::JACOBIANS, mt);
   maps = &el.GetDofToQuad(*ir, DofToQuad::TENSOR);
   d1d = maps->ndof;
   q1d = maps->nqpt;

   QuadratureSpace qs(*mesh, *ir);
   CoefficientVector coeff(Q, qs, CoefficientStorage::COMPRESSED);

   const int nq1d = q1d * q1d * (dim==3 ? q1d : 1);
   MFEM_VERIFY(coeff.Size() == 1 || coeff.Size() == nq1d*ne, "Invalid coeff");
   MFEM_VERIFY(ir->GetWeights().Size() == nq1d, "Invalid weights size");

   const auto w_r = ir->GetWeights().Read();
   const bool const_coeff = coeff.Size() == 1;

   if (dim == 2)
   {
      const int Q1D = q1d;
      constexpr int VDIM = 2, DIM = 2;
      const auto W = Reshape(w_r, Q1D, Q1D);
      const auto C = const_coeff ?
                     Reshape(coeff.Read(), 1, 1, 1) :
                     Reshape(coeff.Read(), Q1D, Q1D, ne);
      const auto J = Reshape(geom->J.Read(), Q1D, Q1D, VDIM, DIM, ne);
      auto A = Reshape(pa_adj.Write(), VDIM, DIM, Q1D, Q1D, ne);

      mfem::forall_2D(ne, Q1D, Q1D, [=] MFEM_HOST_DEVICE(int e)
      {
         MFEM_FOREACH_THREAD_DIRECT(qy, y, Q1D)
         {
            MFEM_FOREACH_THREAD_DIRECT(qx, x, Q1D)
            {
               const real_t J11 = J(qx, qy, 0, 0, e), J12 = J(qx, qy, 0, 1, e);
               const real_t J21 = J(qx, qy, 1, 0, e), J22 = J(qx, qy, 1, 1, e);
               // adj(J)
               const real_t A11 = +J22, A12 = -J12;
               const real_t A21 = -J21, A22 = +J11;
               // Store w * coeff * adj(J)
               const real_t w = W(qx, qy);
               const real_t c = const_coeff ? C(0, 0, 0) : C(qx, qy, e);
               A(0, 0, qx, qy, e) = w * c * A11;
               A(1, 0, qx, qy, e) = w * c * A12;
               A(0, 1, qx, qy, e) = w * c * A21;
               A(1, 1, qx, qy, e) = w * c * A22;
            }
         }
      });
   }
   else if (dim == 3)
   {
      const int Q1D = q1d;
      constexpr int VDIM = 3, DIM = 3;
      const auto W = Reshape(w_r, Q1D, Q1D, Q1D);
      const auto C = const_coeff ?
                     Reshape(coeff.Read(), 1, 1, 1, 1) :
                     Reshape(coeff.Read(), Q1D, Q1D, Q1D, ne);
      const auto J = Reshape(geom->J.Read(), Q1D, Q1D, Q1D, VDIM, DIM, ne);
      auto A = Reshape(pa_adj.Write(), VDIM, DIM, Q1D, Q1D, Q1D, ne);

      mfem::forall_3D(ne, Q1D, Q1D, Q1D, [=] MFEM_HOST_DEVICE(int e)
      {
         MFEM_FOREACH_THREAD_DIRECT(qz, z, Q1D)
         {
            MFEM_FOREACH_THREAD_DIRECT(qy, y, Q1D)
            {
               MFEM_FOREACH_THREAD_DIRECT(qx, x, Q1D)
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
                  const real_t c =
                     const_coeff ? C(0, 0, 0, 0) : C(qx, qy, qz, e);
                  const real_t cw = W(qx, qy, qz) * c;
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
                  // Store wq * coeff * adj(J)
                  A(0, 0, qx, qy, qz, e) = cw * A11;
                  A(1, 0, qx, qy, qz, e) = cw * A12;
                  A(2, 0, qx, qy, qz, e) = cw * A13;
                  A(0, 1, qx, qy, qz, e) = cw * A21;
                  A(1, 1, qx, qy, qz, e) = cw * A22;
                  A(2, 1, qx, qy, qz, e) = cw * A23;
                  A(0, 2, qx, qy, qz, e) = cw * A31;
                  A(1, 2, qx, qy, qz, e) = cw * A32;
                  A(2, 2, qx, qy, qz, e) = cw * A33;
               }
            }
         }
      });
   }
   else
   {
      MFEM_ABORT("dim " << dim << " not supported!");
   }
}

void VectorConvectionNLFIntegrator::AddMultPA(const Vector &x, Vector &y) const
{
   if (DeviceCanUseCeed())
   {
      ceedOp->AddMult(x, y);
   }
   else
   {
      static const auto specializations =
         (AddMultPAKernels::Specialization<2, 2,2>::Add(),
          AddMultPAKernels::Specialization<2, 2,3>::Add(),
          AddMultPAKernels::Specialization<2, 3,4>::Add(),
          AddMultPAKernels::Specialization<2, 3,5>::Add(),
          AddMultPAKernels::Specialization<2, 4,5>::Add(),
          AddMultPAKernels::Specialization<2, 4,6>::Add(),
          AddMultPAKernels::Specialization<2, 5,7>::Add(),
          AddMultPAKernels::Specialization<2, 5,8>::Add(),
          AddMultPAKernels::Specialization<2, 6,8>::Add(),
          // 3D
          AddMultPAKernels::Specialization<3, 2,3>::Add(),
          AddMultPAKernels::Specialization<3, 2,4>::Add(),
          AddMultPAKernels::Specialization<3, 2,5>::Add(),
          AddMultPAKernels::Specialization<3, 3,4>::Add(),
          AddMultPAKernels::Specialization<3, 3,5>::Add(),
          AddMultPAKernels::Specialization<3, 3,6>::Add(),
          AddMultPAKernels::Specialization<3, 4,5>::Add(),
          AddMultPAKernels::Specialization<3, 4,6>::Add(),
          AddMultPAKernels::Specialization<3, 4,7>::Add(),
          AddMultPAKernels::Specialization<3, 4,8>::Add(),
          AddMultPAKernels::Specialization<3, 5,6>::Add(),
          AddMultPAKernels::Specialization<3, 5,7>::Add(),
          AddMultPAKernels::Specialization<3, 5,8>::Add(),
          true);
      MFEM_CONTRACT_VAR(specializations);

      AddMultPAKernels::Run(dim, d1d, q1d, ne,
                            maps->B.Read(),
                            maps->G.Read(),
                            pa_adj.Read(),
                            x.Read(),
                            y.ReadWrite(),
                            d1d, q1d);
   }
}

} // namespace mfem
