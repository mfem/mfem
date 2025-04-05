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
#include "fem/qinterp/grad.hpp" // IWYU pragma: keep
#include "linalg/tensor.hpp"

using namespace mfem;
using mfem::internal::tensor;
using DOperator = DifferentiableOperator;

#undef NVTX_COLOR
#define NVTX_COLOR nvtx::kAquamarine
#include "general/nvtx.hpp"

namespace dfem_pa_kernels
{

///////////////////////////////////////////////////////////////////////////////
template <int DIM> struct Diffusion
{
   using vecd_t = tensor<real_t, DIM>;
   using matd_t = tensor<real_t, DIM, DIM>;

   struct MFApply
   {
      MFEM_HOST_DEVICE inline auto operator()(const vecd_t &dudxi,
                                              //  const real_t &rho,
                                              const matd_t &J,
                                              const real_t &w) const
      {
         const auto invJ = inv(J), TinJ = transpose(invJ);
         return mfem::tuple{ (dudxi * invJ) * TinJ * det(J) * w /** rho*/ };
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

///////////////////////////////////////////////////////////////////////////////
auto print_vec = [](const char *header, const Vector &v)
{
   dbl("{}:", header);
   for (int i=0; i < v.Size(); i++) { dba("{:f} ", v(i)); }
   dbc();
};

///////////////////////////////////////////////////////////////////////////////
template <int DIM>
void DFemDiffusion(const char *filename, int p, const int r, const bool no_dfem)
{
   CAPTURE(filename, DIM, p, r);

   Mesh smesh(filename);
   ParMesh pmesh(MPI_COMM_WORLD, smesh);
   MFEM_VERIFY(pmesh.Dimension() == DIM, "Mesh dimension mismatch");

   pmesh.EnsureNodes();
   auto *nodes = static_cast<ParGridFunction *>(pmesh.GetNodes());
   p = std::max(p, pmesh.GetNodalFESpace()->GetMaxElementOrder());
   smesh.Clear();

   Array<int> all_domain_attr;
   if (pmesh.bdr_attributes.Size() > 0)
   {
      all_domain_attr.SetSize(pmesh.bdr_attributes.Max());
      all_domain_attr = 1;
   }

   H1_FECollection fec(p, DIM);
   ParFiniteElementSpace pfes(&pmesh, &fec);
   const auto *R = pfes.GetRestrictionOperator();
   MFEM_VERIFY(R, "Restriction operator not set");
   ParFiniteElementSpace *mfes = nodes->ParFESpace();

   const int NE = pfes.GetNE(), d1d(p + 1), q = 2 * p + r;
   const auto *ir = &IntRules.Get(pmesh.GetTypicalElementGeometry(), q);
   const int q1d(IntRules.Get(Geometry::SEGMENT, ir->GetOrder()).GetNPoints());
   MFEM_VERIFY(d1d <= q1d, "q1d should be >= d1d");

   ParGridFunction x(&pfes), y(&pfes), z(&pfes);
   Vector X(pfes.GetTrueVSize()), Y(pfes.GetTrueVSize()), Z(pfes.GetTrueVSize());

   dbg("x:{} X:{}", x.Size(), X.Size());
   // static int seed = 1;
   // x.Randomize(1);
   for (int i=0; i < x.Size(); i++) { x(i) = (real_t)i; }
   // x = M_PI;
   x.SetTrueVector(), x.SetFromTrueVector();

   print_vec("x", x);
   R->Mult(x, Y);
   print_vec("Y", Y);
   R->MultTranspose(Y, y);
   print_vec("     y", y);
   y.SetTrueVector(), y.SetFromTrueVector();
   print_vec("sync y", y);
   y -= x;
   REQUIRE(y.Normlinf() == MFEM_Approx(0.0));
   MPI_Barrier(MPI_COMM_WORLD);

   auto rho = [](const Vector &xyz)
   {
#if 0
      const real_t x = xyz(0), y = xyz(1), z = DIM == 3 ? xyz(2) : 0.0;
      real_t r = M_PI * pow(x, 2);
      if (DIM >= 2) { r += pow(y, 3); }
      if (DIM >= 3) { r += pow(z, 4); }
      return r;
#else
      return 1.0;
#endif
   };
   FunctionCoefficient rho_coeff(rho);

   ParBilinearForm blf_fa(&pfes);
   blf_fa.AddDomainIntegrator(new DiffusionIntegrator(rho_coeff, ir));
   blf_fa.Assemble();
   blf_fa.Finalize();

   // SECTION("Partial assembly")
   {
      dbg("Partial assembly");
      ParBilinearForm blf_pa(&pfes);
      blf_pa.AddDomainIntegrator(new DiffusionIntegrator(rho_coeff, ir));
      blf_pa.SetAssemblyLevel(AssemblyLevel::FULL);
      blf_pa.Assemble();
      print_vec("[PA] x", x);
      blf_pa.Mult(x, z);
      print_vec("[PA] z", z);
      blf_fa.Mult(x, y);
      print_vec("[FA] y", y);
      y -= z;
      REQUIRE(y.Normlinf() == MFEM_Approx(0.0));
   }

   if (no_dfem) { return; } // Skip if DFEM is disabled

   QuadratureSpace qs(pmesh, *ir);
   CoefficientVector rho_coeff_cv(rho_coeff, qs);
   MFEM_VERIFY(rho_coeff_cv.GetVDim() == 1, "Coefficient should be scalar");
   MFEM_VERIFY(rho_coeff_cv.Size() == q1d * q1d * (DIM == 3 ? q1d : 1) * NE, "");

   const int rho_local_size = 1;
   const int rho_elem_size(rho_local_size * ir->GetNPoints());
   const int rho_total_size(rho_elem_size * NE);
#warning 🔥 2D workaround for is_none_fop Reshape access
   ParametricSpace rho_ps(DIM, rho_local_size, rho_elem_size, rho_total_size,
                          DIM == 3 ? d1d : d1d * d1d,
                          DIM == 3 ? q1d : q1d * q1d);

   static constexpr int U = 0, Coords = 1, Rho = 3;
   const auto sol = std::vector{ FieldDescriptor{ U, &pfes } };

#if 1
#warning 🔥 2D parallel errors
   // SECTION("DFEM Matrix free")
   {
      dbg("DFEM Matrix free");
      DOperator dop_mf(sol, {/*{Rho, &rho_ps},*/ {Coords, mfes}}, pmesh);
      typename Diffusion<DIM>::MFApply mf_apply_qf;
      dop_mf.AddDomainIntegrator(mf_apply_qf,
                                 mfem::tuple{ Gradient<U>{}, /*None<Rho>{},*/
                                              Gradient<Coords>{}, Weight{} },
                                 mfem::tuple{ Gradient<U>{} }, *ir,
                                 all_domain_attr);
      dop_mf.SetParameters({ /*&rho_coeff_cv,*/ nodes });

      // X = 0.0, Z = 0.0, y = 0.0, z = 0.0;
      x.SetTrueVector(), x.SetFromTrueVector();
      // y.SetTrueVector(), y.SetFromTrueVector();
      z.SetTrueVector(), z.SetFromTrueVector();

      print_vec("[MF] x", x);
      R->Mult(x, X);
      dop_mf.Mult(X, Z);
      R->MultTranspose(Z, z);
      print_vec("[MF]       z", z);
      // z.SetTrueVector();
      // print_vec("[MF][set]  z", z);
      z.SetFromTrueVector();
      print_vec("[MF][from] z", z);

      blf_fa.Mult(x, y);
      print_vec("[FA]       y", y);
      y.SetTrueVector(), y.SetFromTrueVector();
      print_vec("[FA][sync] y", y);
      y -= z;
      MPI_Barrier(MPI_COMM_WORLD);
      REQUIRE(y.Normlinf() == MFEM_Approx(0.0));
   }
#endif

#if 0
#warning 🔥 3D parallel errors
   // SECTION("DFEM Partial assembly")
   {
      static constexpr int QData = 2;
      const int qd_local_size = DIM * DIM;
      const int qd_elem_size(qd_local_size * ir->GetNPoints());
      const int qd_total_size(qd_elem_size * NE);
#warning 🔥 2D workaround for is_none_fop Reshape access
      ParametricSpace qd_ps(DIM, qd_local_size, qd_elem_size, qd_total_size,
                            DIM == 3 ? d1d : d1d * d1d,
                            DIM == 3 ? q1d : q1d * q1d);
      ParametricFunction qdata(qd_ps);
      qdata.UseDevice(true);

      DOperator dSetup(sol, {{Rho, &rho_ps}, {Coords, mfes}, {QData, &qd_ps}}, pmesh);
      typename Diffusion<DIM>::PASetup pa_setup_qf;
      dSetup.AddDomainIntegrator(
         pa_setup_qf,
         mfem::tuple{ None<U>{}, None<Rho>{}, Gradient<Coords>{}, Weight{} },
         mfem::tuple{ None<QData>{} }, *ir, all_domain_attr);
      dSetup.SetParameters({ &rho_coeff_cv, nodes, &qdata });
      pfes.GetRestrictionMatrix()->Mult(x, X);
      dSetup.Mult(X, qdata);

      DOperator dop_pa(sol, { { QData, &qd_ps } }, pmesh);
      typename Diffusion<DIM>::PAApply pa_apply_qf;
      dop_pa.AddDomainIntegrator(pa_apply_qf,
                                 mfem::tuple{ Gradient<U>{}, None<QData>{} },
                                 mfem::tuple{ Gradient<U>{} },
                                 *ir, all_domain_attr);
      dop_pa.SetParameters({ &qdata });
      {
         R->Mult(x, X);
         dop_pa.Mult(X, Z);
         R->MultTranspose(Z, z);
         z.SetTrueVector(), z.SetFromTrueVector();
      }
      blf_fa.Mult(x, y);
      y -= z;
      REQUIRE(y.Normlinf() == MFEM_Approx(0.0));
   }
#endif
}

///////////////////////////////////////////////////////////////////////////////
TEST_CASE("DFEM Diffusion", "[Parallel][DFEM]")
{
   // using Grad = QuadratureInterpolator::GradKernels;
   // Grad::Specialization<3, QVectorLayout::byNODES, false, 3, 2, 2>::Add();
   // Grad::Specialization<3, QVectorLayout::byNODES, false, 3, 4, 5>::Add();

   static const bool no_dfem = ::getenv("MFEM_NO_DFEM") != nullptr;
   // const bool all_tests = launch_all_non_regression_tests;

   // const auto p = !all_tests ? 2 : GENERATE(1, 2, 3);
   // const auto r = !all_tests ? 1 : GENERATE(0, 1, 2, 3);
   const int p = 1, r = 0;

   dbg("p:{}, r:{}", p, r);
   // SECTION("2D p=" + std::to_string(p) + " r=" + std::to_string(r))
   {
      const auto filename =
         GENERATE("../../data/star.mesh"//,
                  // "../../data/star-q3.mesh",
                  // "../../data/rt-2d-q3.mesh",
                  //"../../data/inline-quad.mesh"//,
                  // "../../data/periodic-square.mesh"
                 );
      DFemDiffusion<2>(filename, p, r, no_dfem);
   }

   /*SECTION("3D p=" + std::to_string(p) + " r=" + std::to_string(r))
   {
      const auto filename =
         GENERATE("../../data/fichera.mesh"//,
                  // "../../data/fichera-q3.mesh",
                  // "../../data/inline-hex.mesh",
                  // "../../data/toroid-hex.mesh",
                  // "../../data/periodic-cube.mesh"
                 );
      DFemDiffusion<3>(filename, p, r, no_dfem);
   }*/
}

} // namespace dfem_pa_kernels
