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
#include "./bilininteg_vecdiv_pa.hpp" // IWYU pragma: keep // IWYU pragma: keep

namespace mfem
{

// PA Divergence Assemble 2D kernel
static void PADivergenceSetup2D(const int Q1D,
                                const int NE,
                                const Array<real_t> &w,
                                const Vector &j,
                                const real_t COEFF,
                                Vector &op)
{
   const auto W = Reshape(w.Read(), Q1D, Q1D);
   const auto J = Reshape(j.Read(), Q1D, Q1D, 2, 2, NE);
   auto y = Reshape(op.Write(), Q1D, Q1D, 2, 2, NE);

   mfem::forall_2D(NE, Q1D, Q1D, [=] MFEM_HOST_DEVICE (int e)
   {
      MFEM_FOREACH_THREAD_DIRECT(qx, x, Q1D)
      {
         MFEM_FOREACH_THREAD_DIRECT(qy, y, Q1D)
         {
            const real_t J11 = J(qx, qy, 0, 0, e);
            const real_t J21 = J(qx, qy, 1, 0, e);
            const real_t J12 = J(qx, qy, 0, 1, e);
            const real_t J22 = J(qx, qy, 1, 1, e);
            const real_t cw = W(qx, qy) * COEFF;
            y(qx, qy, 0, 0, e) = cw *  J22;
            y(qx, qy, 0, 1, e) = cw * -J12;
            y(qx, qy, 1, 0, e) = cw * -J21;
            y(qx, qy, 1, 1, e) = cw *  J11;
         }
      }
   });
}

// PA Divergence Assemble 3D kernel
static void PADivergenceSetup3D(const int Q1D,
                                const int NE,
                                const Array<real_t> &w,
                                const Vector &j,
                                const real_t COEFF,
                                Vector &op)
{
   const auto W = Reshape(w.Read(), Q1D, Q1D, Q1D);
   const auto J = Reshape(j.Read(), Q1D, Q1D, Q1D, 3, 3, NE);
   auto y = Reshape(op.Write(), Q1D, Q1D, Q1D, 3, 3, NE);

   mfem::forall_3D(NE, Q1D, Q1D, Q1D, [=] MFEM_HOST_DEVICE (int e)
   {
      MFEM_FOREACH_THREAD_DIRECT(qx, x, Q1D)
      {
         MFEM_FOREACH_THREAD_DIRECT(qy, y, Q1D)
         {
            MFEM_FOREACH_THREAD_DIRECT(qz, z, Q1D)
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
               const real_t cw = W(qx, qy, qz) * COEFF;
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
               // Store wq * Q * adj(J)
               y(qx, qy, qz, 0, 0, e) = cw * A11;
               y(qx, qy, qz, 0, 1, e) = cw * A12;
               y(qx, qy, qz, 0, 2, e) = cw * A13;
               y(qx, qy, qz, 1, 0, e) = cw * A21;
               y(qx, qy, qz, 1, 1, e) = cw * A22;
               y(qx, qy, qz, 1, 2, e) = cw * A23;
               y(qx, qy, qz, 2, 0, e) = cw * A31;
               y(qx, qy, qz, 2, 1, e) = cw * A32;
               y(qx, qy, qz, 2, 2, e) = cw * A33;
            }
         }
      }
   });
}

static void PADivergenceSetup(const int dim,
                              const int Q1D,
                              const int NE,
                              const Array<real_t> &W,
                              const Vector &J,
                              const real_t COEFF,
                              Vector &op)
{
   if (dim == 1) { MFEM_ABORT("dim==1 not supported in PADivergenceSetup"); }
   if (dim == 2)
   {
      PADivergenceSetup2D(Q1D, NE, W, J, COEFF, op);
   }
   if (dim == 3)
   {
      PADivergenceSetup3D(Q1D, NE, W, J, COEFF, op);
   }
}

void VectorDivergenceIntegrator::AssemblePA(const FiniteElementSpace &trial_fes,
                                            const FiniteElementSpace &test_fes)
{
   // Assumes tensor-product elements ordered by nodes
   MFEM_ASSERT(trial_fes.GetOrdering() == Ordering::byNODES,
               "PA Only supports Ordering::byNODES!");
   auto *mesh = trial_fes.GetMesh();
   const auto &trial_fe = *trial_fes.GetTypicalFE();
   const auto &test_fe = *test_fes.GetTypicalFE();
   const auto *ir = IntRule ? IntRule :
                    &GetRule(trial_fe, test_fe,
                             *mesh->GetTypicalElementTransformation());
   const int dims = trial_fe.GetDim();
   nq = ir->GetNPoints();
   dim = mesh->Dimension();
   ne = trial_fes.GetNE();
   geom = mesh->GetGeometricFactors(*ir, GeometricFactors::JACOBIANS);
   trial_maps = &trial_fe.GetDofToQuad(*ir, DofToQuad::TENSOR);
   trial_dofs1D = trial_maps->ndof;
   quad1D = trial_maps->nqpt;
   test_maps = &test_fe.GetDofToQuad(*ir, DofToQuad::TENSOR);
   test_dofs1D = test_maps->ndof;
   MFEM_ASSERT(quad1D == test_maps->nqpt,
               "PA requires test and trial space to have same number of "
               "quadrature points!");
   pa_data.SetSize(nq * dims * dims * ne, Device::GetMemoryType());

   real_t coeff = 1.0;
   if (Q)
   {
      auto *cQ = dynamic_cast<ConstantCoefficient *>(Q);
      MFEM_VERIFY(cQ, "only ConstantCoefficient is supported!");
      coeff = cQ->constant;
   }
   PADivergenceSetup(dim, quad1D, ne, ir->GetWeights(), geom->J, coeff, pa_data);
}

// PA Divergence Apply kernel
void VectorDivergenceIntegrator::AddMultPA(const Vector &x, Vector &y) const
{
   using Kernels = VectorDivergenceAddMultPA;

   static const auto specializations =
      ( // 2D
         Kernels::Specialization<2, 2, 2, 2>::Add(),
         Kernels::Specialization<2, 2, 2, 3>::Add(),
         Kernels::Specialization<2, 2, 2, 4>::Add(),
         Kernels::Specialization<2, 3, 3, 3>::Add(),
         Kernels::Specialization<2, 3, 3, 4>::Add(),
         Kernels::Specialization<2, 3, 3, 5>::Add(),
         Kernels::Specialization<2, 4, 4, 4>::Add(),
         Kernels::Specialization<2, 4, 4, 5>::Add(),
         Kernels::Specialization<2, 4, 4, 6>::Add(),
         Kernels::Specialization<2, 5, 5, 5>::Add(),
         Kernels::Specialization<2, 5, 5, 6>::Add(),
         Kernels::Specialization<2, 5, 5, 7>::Add(),
         // 3D
         Kernels::Specialization<3, 2, 2, 3>::Add(),
         Kernels::Specialization<3, 2, 2, 4>::Add(),
         Kernels::Specialization<3, 2, 2, 6>::Add(),
         Kernels::Specialization<3, 3, 2, 5>::Add(),
         Kernels::Specialization<3, 3, 3, 4>::Add(),
         Kernels::Specialization<3, 3, 3, 5>::Add(),
         Kernels::Specialization<3, 3, 3, 7>::Add(),
         Kernels::Specialization<3, 4, 4, 5>::Add(),
         Kernels::Specialization<3, 4, 4, 6>::Add(),
         Kernels::Specialization<3, 4, 4, 8>::Add(),
         Kernels::Specialization<3, 5, 5, 6>::Add(),
         Kernels::Specialization<3, 5, 5, 7>::Add(),
         Kernels::Specialization<3, 5, 5, 9>::Add(),
         true);
   MFEM_CONTRACT_VAR(specializations);

   Kernels::Run(dim, trial_dofs1D, test_dofs1D, quad1D, ne,
                trial_maps->B, trial_maps->G, test_maps->Bt,
                pa_data, x, y,
                trial_dofs1D, test_dofs1D, quad1D);
}

// PA Divergence Apply kernel transpose
void VectorDivergenceIntegrator::AddMultTransposePA(const Vector &x,
                                                    Vector &y) const
{
   using Kernels = VectorDivergenceAddMultTransposePA;

   static const auto specializations =
      ( // 2D
         Kernels::Specialization<2, 2, 2, 2>::Add(),
         Kernels::Specialization<2, 2, 2, 3>::Add(),
         Kernels::Specialization<2, 2, 2, 4>::Add(),
         Kernels::Specialization<2, 3, 3, 3>::Add(),
         Kernels::Specialization<2, 3, 3, 4>::Add(),
         Kernels::Specialization<2, 3, 3, 5>::Add(),
         Kernels::Specialization<2, 4, 4, 4>::Add(),
         Kernels::Specialization<2, 4, 4, 5>::Add(),
         Kernels::Specialization<2, 4, 4, 6>::Add(),
         Kernels::Specialization<2, 5, 5, 5>::Add(),
         Kernels::Specialization<2, 5, 5, 6>::Add(),
         Kernels::Specialization<2, 5, 5, 7>::Add(),
         // 3D
         Kernels::Specialization<3, 2, 2, 3>::Add(),
         Kernels::Specialization<3, 2, 2, 4>::Add(),
         Kernels::Specialization<3, 2, 2, 6>::Add(),
         Kernels::Specialization<3, 3, 2, 5>::Add(),
         Kernels::Specialization<3, 3, 3, 4>::Add(),
         Kernels::Specialization<3, 3, 3, 5>::Add(),
         Kernels::Specialization<3, 3, 3, 7>::Add(),
         Kernels::Specialization<3, 4, 4, 5>::Add(),
         Kernels::Specialization<3, 4, 4, 6>::Add(),
         Kernels::Specialization<3, 4, 4, 8>::Add(),
         Kernels::Specialization<3, 5, 5, 6>::Add(),
         Kernels::Specialization<3, 5, 5, 7>::Add(),
         Kernels::Specialization<3, 5, 5, 9>::Add(),
         true);
   MFEM_CONTRACT_VAR(specializations);

   Kernels::Run(dim, trial_dofs1D, test_dofs1D, quad1D, ne,
                trial_maps->Bt, trial_maps->Gt, test_maps->B,
                pa_data, x, y,
                trial_dofs1D, test_dofs1D, quad1D);
}

} // namespace mfem
