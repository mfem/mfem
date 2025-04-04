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

#include "unit_tests.hpp"

#include "mfem.hpp"

#include "fem/dfem/doperator.hpp"
#include "linalg/tensor.hpp"

using namespace mfem;
using mfem::internal::tensor;
using DOperator = DifferentiableOperator;

namespace dfem_pa_kernels
{

template <int DIM>
struct Diffusion
{
   using vecd_t = tensor<real_t, DIM>;
   using matd_t = tensor<real_t, DIM, DIM>;

   struct MFApply
   {
      MFEM_HOST_DEVICE inline auto operator()(const vecd_t &dudxi,
                                              const real_t &rho,
                                              const matd_t &J,
                                              const real_t &w) const
      {
         const auto invJ = inv(J), TinJ = transpose(invJ);
         return mfem::tuple{ (dudxi * invJ) * TinJ * det(J) * w * rho };
      }
   };

   struct PASetup
   {
      MFEM_HOST_DEVICE inline auto operator()(const real_t &u,
                                              const real_t &rho,
                                              const matd_t &J,
                                              const real_t &w) const
      {
         return mfem::tuple{ inv(J) * transpose(inv(J)) * det(J) * w * rho };
      }
   };

   struct PAApply
   {
      MFEM_HOST_DEVICE inline auto operator()(const vecd_t &dudxi,
                                              const matd_t &q) const
      {
         return mfem::tuple{ q * dudxi };
      };
   };
};

template <int DIM>
void DFemDiffusion(const char *filename, const int p, const int q_inc)
{
   CAPTURE(filename, DIM, p, q_inc);

   Mesh smesh(filename);
   smesh.EnsureNodes();
   ParMesh pmesh(MPI_COMM_WORLD, smesh);
   MFEM_VERIFY(pmesh.Dimension() == DIM, "Mesh dimension mismatch");
   pmesh.SetCurvature(p); // 🔥 necessary with 3D q3 ?!
   auto *nodes = static_cast<ParGridFunction *>(pmesh.GetNodes());
   smesh.Clear();

   Array<int> all_domain_attr(pmesh.bdr_attributes.Max());
   all_domain_attr = 1;

   H1_FECollection fec(p, DIM);
   ParFiniteElementSpace pfes(&pmesh, &fec);
   ParFiniteElementSpace *mfes = nodes->ParFESpace();

   const int NE = pfes.GetNE(), d1d(p + 1), q = 2 * p + q_inc;
   const auto *ir = &IntRules.Get(pmesh.GetTypicalElementGeometry(), q);
   const int q1d(IntRules.Get(Geometry::SEGMENT, ir->GetOrder()).GetNPoints());

   ParGridFunction x(&pfes), y(&pfes), z(&pfes);
   x.Randomize(1);

   auto rho = [](const Vector &xyz)
   {
      const real_t x = xyz(0), y = xyz(1), z = DIM == 3 ? xyz(2) : 0.0;
      real_t r = M_PI * pow(x, 2);
      if (DIM >= 2) { r += pow(y, 3); }
      if (DIM >= 3) { r += pow(z, 4); }
      return r;
   };
   FunctionCoefficient rho_coeff(rho);

   ParBilinearForm blf_fa(&pfes);
   blf_fa.AddDomainIntegrator(new DiffusionIntegrator(rho_coeff, ir));
   blf_fa.Assemble();
   blf_fa.Finalize();

   QuadratureSpace qs(pmesh, *ir);
   CoefficientVector rho_coeff_cv(rho_coeff, qs);
   REQUIRE(rho_coeff_cv.GetVDim() == 1);
   REQUIRE(rho_coeff_cv.Size() == q1d * q1d * (DIM == 3 ? q1d : 1) * NE);

   const int rho_local_size = 1;
   const int rho_elem_size(rho_local_size * ir->GetNPoints());
   const int rho_total_size(rho_elem_size * NE);
   // 🔥 2D workaround for is_none_fop Reshape access
   ParametricSpace rho_ps(DIM, rho_local_size, rho_elem_size, rho_total_size,
                          DIM == 3 ? d1d : d1d * d1d,
                          DIM == 3 ? q1d : q1d * q1d);

   static constexpr int U = 0, Coords = 1, QData = 2, Rho = 3;
   const auto sol = std::vector{ FieldDescriptor{ U, &pfes } };

   SECTION("Matrix free")
   {
      DOperator dop_mf(sol, { { Rho, &rho_ps }, { Coords, mfes } }, pmesh);
      typename Diffusion<DIM>::MFApply mf_apply_qf;
      dop_mf.AddDomainIntegrator(mf_apply_qf,
                                 mfem::tuple{ Gradient<U>{}, None<Rho>{},
                                              Gradient<Coords>{}, Weight{} },
                                 mfem::tuple{ Gradient<U>{} }, *ir,
                                 all_domain_attr);
      dop_mf.SetParameters({ &rho_coeff_cv, nodes });
      dop_mf.Mult(x, z);
      blf_fa.Mult(x, y);
      y -= z;
      REQUIRE(y.Normlinf() == MFEM_Approx(0.0));
   }

   SECTION("Partial assembly")
   {
      const int qd_local_size = DIM * DIM;
      const int qd_elem_size(qd_local_size * ir->GetNPoints());
      const int qd_total_size(qd_elem_size * NE);
      // 🔥 2D workaround for is_none_fop Reshape access
      ParametricSpace qd_ps(DIM, qd_local_size, qd_elem_size, qd_total_size,
                            DIM == 3 ? d1d : d1d * d1d,
                            DIM == 3 ? q1d : q1d * q1d);
      ParametricFunction qdata(qd_ps);
      qdata.UseDevice(true);

      DOperator dSetup(
      sol, { { Rho, &rho_ps }, { Coords, mfes }, { QData, &qd_ps } }, pmesh);
      typename Diffusion<DIM>::PASetup pa_setup_qf;
      dSetup.AddDomainIntegrator(
         pa_setup_qf,
         mfem::tuple{ None<U>{}, None<Rho>{}, Gradient<Coords>{}, Weight{} },
         mfem::tuple{ None<QData>{} }, *ir, all_domain_attr);
      dSetup.SetParameters({ &rho_coeff_cv, nodes, &qdata });
      dSetup.Mult(x, qdata);

      DOperator dop_pa(sol, { { QData, &qd_ps } }, pmesh);
      typename Diffusion<DIM>::PAApply pa_apply_qf;
      dop_pa.AddDomainIntegrator(
         pa_apply_qf, mfem::tuple{ Gradient<U>{}, None<QData>{} },
         mfem::tuple{ Gradient<U>{} }, *ir, all_domain_attr);
      dop_pa.SetParameters({ &qdata });
      dop_pa.Mult(x, z);
      blf_fa.Mult(x, y);
      y -= z;
      REQUIRE(y.Normlinf() == MFEM_Approx(0.0));
   }
}

TEST_CASE("DFEM Diffusion", "[Parallel][DFEM]")
{
   const bool all_tests = launch_all_non_regression_tests;

   const auto order = !all_tests ? 2 : GENERATE(1, 2, 3);
   const auto q_order_inc = !all_tests ? 0 : GENERATE(0, 1, 2, 3);

   SECTION("2D order " + std::to_string(order) + " q_order_inc " +
           std::to_string(q_order_inc))
   {
      const auto filename =
         GENERATE("../../data/star.mesh",
                  "../../data/star-q3.mesh",
                  "../../data/inline-quad.mesh"
                  // "../../data/periodic-square.mesh", // ❌
                 );
      DFemDiffusion<2>(filename, order, q_order_inc);
   }

   SECTION("3D order " + std::to_string(order) + " q_order_inc " +
           std::to_string(q_order_inc))
   {
      const auto filename =
         GENERATE("../../data/fichera.mesh",
                  "../../data/fichera-q3.mesh",
                  "../../data/inline-hex.mesh",
                  "../../data/periodic-cube.mesh");
      DFemDiffusion<3>(filename, order, q_order_inc);
   }
}

} // namespace dfem_pa_kernels
