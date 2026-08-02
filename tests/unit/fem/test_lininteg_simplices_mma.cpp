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

#ifdef _WIN32
#define _USE_MATH_DEFINES
#include <cmath>
#endif

#include "unit_tests.hpp"
#include "mfem.hpp"
#include "fem/integ/bilininteg_pa_mma.hpp"

using namespace mfem;

namespace lininteg_simplex_mma
{

/** ir_order < 0 → default (use_2p_ir ? 2p : 2p+OrderW+4). */
void test_domain_lf_simplex_mma(const char *filename, int p, bool use_2p_ir,
                                int ir_order = -1)
{
   CAPTURE(filename, p, use_2p_ir, ir_order);

   Mesh mesh(filename);
   MFEM_VERIFY((mesh.Dimension() == 2 || mesh.Dimension() == 3),
               "Mesh dimension must be 2 or 3");
   MFEM_VERIFY(!mesh.IsMixedMesh(), "Mesh is mixed");
   MFEM_VERIFY(mesh.SpaceDimension() == mesh.Dimension(),
               "Simplex MMA requires volumetric meshes (sdim == dim)");

   H1_FECollection fec(p, mesh.Dimension(), BasisType::GaussLobatto);
   FiniteElementSpace fes(&mesh, &fec);

   REQUIRE(UsesSimplexMMA(fes));

   const auto &fe = *fes.GetTypicalFE();
   const auto &Tr = *mesh.GetTypicalElementTransformation();
   const int order = (ir_order >= 0) ? ir_order
                     : (use_2p_ir ? (2 * p)
                        : (2 * fe.GetOrder() + Tr.OrderW() + 4));
   const IntegrationRule *ir = &IntRules.Get(fe.GetGeomType(), order);
   CAPTURE(order, ir->GetNPoints());

   const int max_q1d = DeviceDofQuadLimits::Get().MAX_Q1D;
   const int max_nq = (mesh.Dimension() == 2) ? max_q1d * max_q1d : 256;
   if (ir->GetNPoints() > max_nq) { return; }

   for (int e = 0; e < mesh.GetNE(); e++) { mesh.SetAttribute(e, e % 2 ? 1 : 2); }
   mesh.SetAttributes();
   Array<int> elem_marker(mesh.attributes.Max());
   elem_marker = 1;
   if (elem_marker.Size() >= 2) { elem_marker[0] = 0; }

   ConstantCoefficient const_coeff(M_2_SQRTPI);
   FunctionCoefficient funct_coeff([](const Vector &pt)
   { return M_1_PI + pt[0] * pt[0]; });

   auto compare = [&](Coefficient &Q)
   {
      LinearForm lf_dev(&fes), lf_std(&fes);
      lf_dev.AddDomainIntegrator(new DomainLFIntegrator(Q, ir), elem_marker);
      lf_std.AddDomainIntegrator(new DomainLFIntegrator(Q, ir), elem_marker);

      REQUIRE(lf_dev.SupportsDevice());
      lf_dev.UseFastAssembly(true);
      REQUIRE(lf_dev.SupportsDevice());
      lf_dev.Assemble();

      lf_std.UseFastAssembly(false);
      lf_std.Assemble();

      lf_std -= lf_dev;
      REQUIRE(lf_std.Norml2() == MFEM_Approx(0.0, 1e-10));
   };

   compare(const_coeff);
   compare(funct_coeff);
}

TEST_CASE("DomainLF Simplices MMA", "[LinearFormExtension][MMA][GPU]")
{
   const auto all_tests = launch_all_non_regression_tests;
   const auto p = !all_tests ? GENERATE(1, 2, 5, 6) : GENERATE(1, 2, 3, 4, 5, 6);
   const auto use_2p_ir = GENERATE(false, true);

   SECTION("2D")
   {
      auto meshs = { "../../data/beam-tri.mesh",
                     "../../data/inline-tri.mesh",
                     "../../data/ref-triangle.mesh"
                   };
      test_domain_lf_simplex_mma(GENERATE_REF(from_range(meshs)), p, use_2p_ir);
   }

   SECTION("3D")
   {
      auto meshs = { "../../data/beam-tet.mesh",
                     "../../data/inline-tet.mesh",
                     "../../data/ref-tetrahedron.mesh"
                   };
      test_domain_lf_simplex_mma(GENERATE_REF(from_range(meshs)), p, use_2p_ir);
   }

   // Unregistered (D1D,nq) → AssembleSimplexMmaKernels::Fallback.
   SECTION("Fallback 2D triangle nq=7")
   {
      // Tables register (2,3/12/...), not (2,7).
      test_domain_lf_simplex_mma("../../data/ref-triangle.mesh", 1, false, 5);
      test_domain_lf_simplex_mma("../../data/inline-tri.mesh", 1, false, 5);
   }
   SECTION("Fallback 3D tet nq=35")
   {
      // Tables register (2,4/8/14/24), not (2,35).
      test_domain_lf_simplex_mma("../../data/ref-tetrahedron.mesh", 1, false, 7);
      test_domain_lf_simplex_mma("../../data/inline-tet.mesh", 1, false, 7);
   }
}

} // namespace lininteg_simplex_mma
