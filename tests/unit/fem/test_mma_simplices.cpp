// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
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
#include "fem/integ/mma/mma.hpp"

using namespace mfem;

namespace pa_simplices_mma
{

namespace
{

void AddMassDiffIntegrators(BilinearForm &a, const IntegrationRule *ir,
                            Coefficient &const_coeff,
                            Coefficient &funct_coeff)
{
   a.AddDomainIntegrator(new MassIntegrator(ir));
   a.AddDomainIntegrator(new MassIntegrator(const_coeff, ir));
   a.AddDomainIntegrator(new MassIntegrator(funct_coeff, ir));
   a.AddDomainIntegrator(new DiffusionIntegrator(ir));
   a.AddDomainIntegrator(new DiffusionIntegrator(const_coeff, ir));
   a.AddDomainIntegrator(new DiffusionIntegrator(funct_coeff, ir));
}

/** MMA-PA vs stock Simplex PA (Positive / MMAForce). FA vs PA is covered in
    test_pa_simplices.cpp. */
void test_pa_simplices_mma_positive(const char *filename, int p)
{
   CAPTURE(filename, p);

   Mesh mesh(filename);
   MFEM_VERIFY((mesh.Dimension() == 2 || mesh.Dimension() == 3),
               "Mesh dimension must be 2 or 3");
   MFEM_VERIFY(!mesh.IsMixedMesh(), "Mesh is mixed");
   MFEM_VERIFY(mesh.SpaceDimension() == mesh.Dimension(),
               "Simplex MMA requires volumetric meshes (sdim == dim)");

   H1_FECollection fec(p, mesh.Dimension(), BasisType::Positive);
   FiniteElementSpace fes(&mesh, &fec);

   {
      MMAForce on(true);
      if (!UsesSimplexMMA(fes)) { return; }
   }

   GridFunction x(&fes), y_mma(&fes), y_sum(&fes);
   x.Randomize(0x100001b3);
   y_mma.Randomize(0x9e3779b9);
   y_sum = y_mma;

   const auto &fe = *fes.GetTypicalFE();
   const auto &Tr = *mesh.GetTypicalElementTransformation();
   // Stock Positive PA uses ragged-tensor maps with Stroud rules; MMA accepts
   // the same IR via CalcShape, so both paths share Stroud quadrature.
   const auto order = 2 * fe.GetOrder() + Tr.OrderW();
   const IntegrationRule *ir = &StroudIntRules.Get(fe.GetGeomType(), order);

   // Runtime (non-specialized) simplex MMA apply caps.
   const int max_q1d = DeviceDofQuadLimits::Get().MAX_Q1D;
   const int max_nq = (mesh.Dimension() == 2) ? max_q1d * max_q1d : 256;
   if (ir->GetNPoints() > max_nq) { return; }

   ConstantCoefficient const_coeff(M_2_SQRTPI);
   FunctionCoefficient funct_coeff([](const Vector &pt)
   { return M_1_PI + pt[0] * pt[0]; });

   BilinearForm pa_mma(&fes), pa_sum(&fes);
   AddMassDiffIntegrators(pa_mma, ir, const_coeff, funct_coeff);
   AddMassDiffIntegrators(pa_sum, ir, const_coeff, funct_coeff);
   pa_mma.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   pa_sum.SetAssemblyLevel(AssemblyLevel::PARTIAL);

   {
      MMAForce on(true);
      pa_mma.Assemble();
   }
   {
      MMAForce off(false);
      pa_sum.Assemble();
   }

   pa_mma.Mult(x, y_mma);
   pa_sum.Mult(x, y_sum);

   y_sum -= y_mma;
   REQUIRE(y_sum.Normlinf() == MFEM_Approx(0.0, 1e-9, 1e-9));
}

/** ir_order < 0 → default smoke order 2p+OrderW+4.
    compare_stock → MMA Mult vs stock PA (used for Fallback sizes). */
void test_pa_simplices_mma_h1(const char *filename, int p,
                              int ir_order = -1, bool compare_stock = false)
{
   CAPTURE(filename, p, ir_order, compare_stock);

   Mesh mesh(filename);
   MFEM_VERIFY((mesh.Dimension() == 2 || mesh.Dimension() == 3),
               "Mesh dimension must be 2 or 3");
   MFEM_VERIFY(!mesh.IsMixedMesh(), "Mesh is mixed");
   MFEM_VERIFY(mesh.SpaceDimension() == mesh.Dimension(),
               "Simplex MMA requires volumetric meshes (sdim == dim)");

   H1_FECollection fec(p, mesh.Dimension(), BasisType::GaussLobatto);
   FiniteElementSpace fes(&mesh, &fec);

   if (!UsesSimplexMMA(fes)) { return; }

   const auto &fe = *fes.GetTypicalFE();
   const auto &Tr = *mesh.GetTypicalElementTransformation();
   const int order = (ir_order < 0)
                     ? (2 * fe.GetOrder() + Tr.OrderW() + 4)
                     : ir_order;
   const IntegrationRule *ir = &IntRules.Get(fe.GetGeomType(), order);
   CAPTURE(order, ir->GetNPoints());

   const int max_q1d = DeviceDofQuadLimits::Get().MAX_Q1D;
   const int max_nq = (mesh.Dimension() == 2) ? max_q1d * max_q1d : 256;
   if (ir->GetNPoints() > max_nq) { return; }

   ConstantCoefficient const_coeff(M_2_SQRTPI);
   FunctionCoefficient funct_coeff([](const Vector &pt)
   { return M_1_PI + pt[0] * pt[0]; });

   GridFunction x(&fes), y_mma(&fes), y_sum(&fes);
   x.Randomize(0x100001b3);
   y_mma.Randomize(0x9e3779b9);
   y_sum = y_mma;

   BilinearForm pa_mma(&fes);
   AddMassDiffIntegrators(pa_mma, ir, const_coeff, funct_coeff);
   pa_mma.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   pa_mma.Assemble();
   pa_mma.Mult(x, y_mma);

   if (!compare_stock)
   {
      REQUIRE(y_mma.Norml2() >= 0.0);
      return;
   }

   BilinearForm pa_sum(&fes);
   AddMassDiffIntegrators(pa_sum, ir, const_coeff, funct_coeff);
   pa_sum.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   {
      MMAForce off(false);
      pa_sum.Assemble();
   }
   pa_sum.Mult(x, y_sum);
   y_sum -= y_mma;
   REQUIRE(y_sum.Normlinf() == MFEM_Approx(0.0, 1e-9, 1e-9));
}

} // namespace

TEST_CASE("PA Simplices MMA vs stock PA", "[PartialAssembly][MMA][GPU]")
{
   const auto all_tests = launch_all_non_regression_tests;
   const auto p = !all_tests ? GENERATE(1, 2, 5, 6, 7) :
                  GENERATE(1, 2, 3, 4, 5, 6, 7);

   const auto GenMesh = [&](const auto &meshs, const auto &extra)
   {
      return !all_tests
             ? GENERATE_REF(from_range(meshs))
             : GENERATE_REF(from_range(meshs), from_range(extra));
   };

   SECTION("2D")
   {
      auto meshs = { "../../data/beam-tri.mesh",
                     "../../data/inline-tri.mesh",
                     "../../data/ref-triangle.mesh",
                     "../../data/rt-2d-p4-tri.mesh",
                     "../../data/square-disc-p2.mesh",
                     "../../data/square-disc-p3.mesh",
                     "../../data/periodic-annulus-sector.msh"
                   };
      test_pa_simplices_mma_positive(GENERATE_REF(from_range(meshs)), p);
   }

   SECTION("3D")
   {
      auto meshs = { "../../data/beam-tet.mesh",
                     "../../data/inline-tet.mesh",
                     "../../data/ref-tetrahedron.mesh"
                   };
      auto extra = { "../../data/escher.mesh",
                     "../../data/escher-p2.mesh"
                   };
      test_pa_simplices_mma_positive(GenMesh(meshs, extra), p);
   }
}

TEST_CASE("PA Simplices MMA GLL", "[PartialAssembly][MMA][GPU]")
{
   const auto all_tests = launch_all_non_regression_tests;
   const auto p = !all_tests ? GENERATE(1, 2, 5, 6, 7) :
                  GENERATE(1, 2, 3, 4, 5, 6, 7);

   SECTION("smoke 2D")
   {
      auto meshs = { "../../data/ref-triangle.mesh",
                     "../../data/inline-tri.mesh",
                     "../../data/beam-tri.mesh"
                   };
      test_pa_simplices_mma_h1(GENERATE_REF(from_range(meshs)), p);
   }

   SECTION("smoke 3D")
   {
      auto meshs = { "../../data/ref-tetrahedron.mesh",
                     "../../data/inline-tet.mesh",
                     "../../data/beam-tet.mesh"
                   };
      test_pa_simplices_mma_h1(GENERATE_REF(from_range(meshs)), p);
   }

   // Unregistered (D1D,nq) → ApplySimplexMmaPAKernels::Fallback.
   SECTION("Fallback 2D triangle nq=7")
   {
      // Tables register (2,3/4/9/...), not (2,7).
      test_pa_simplices_mma_h1("../../data/ref-triangle.mesh", 1, 5, true);
      test_pa_simplices_mma_h1("../../data/inline-tri.mesh", 1, 5, true);
   }
   SECTION("Fallback 3D tet nq=35")
   {
      // Tables register (2,4/8/14/24), not (2,35).
      test_pa_simplices_mma_h1("../../data/ref-tetrahedron.mesh", 1, 7, true);
      test_pa_simplices_mma_h1("../../data/inline-tet.mesh", 1, 7, true);
   }
}

TEST_CASE("PA Simplices Positive force MMA", "[PartialAssembly][MMA][GPU]")
{
   Mesh mesh("../../data/ref-triangle.mesh");
   H1_FECollection fec(3, mesh.Dimension(), BasisType::Positive);
   FiniteElementSpace fes(&mesh, &fec);

   REQUIRE_FALSE(UsesSimplexMMA(fes));
   REQUIRE(GetEVectorOrdering(fes) == ElementDofOrdering::LEXICOGRAPHIC);

   const auto &fe = *fes.GetTypicalFE();
   const auto &Tr = *mesh.GetTypicalElementTransformation();
   const IntegrationRule *ir =
      &StroudIntRules.Get(fe.GetGeomType(), 2 * fe.GetOrder() + Tr.OrderW());

   GridFunction x(&fes), y_mma(&fes), y_sum(&fes);
   x.Randomize(0x100001b3);
   y_mma.Randomize(0x9e3779b9);
   y_sum = y_mma;

   BilinearForm pa_mma(&fes), pa_sum(&fes);
   pa_mma.AddDomainIntegrator(new MassIntegrator(ir));
   pa_mma.AddDomainIntegrator(new DiffusionIntegrator(ir));
   pa_sum.AddDomainIntegrator(new MassIntegrator(ir));
   pa_sum.AddDomainIntegrator(new DiffusionIntegrator(ir));
   pa_mma.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   pa_sum.SetAssemblyLevel(AssemblyLevel::PARTIAL);

   {
      MMAForce on(true);
      REQUIRE(UsesSimplexMMA(fes));
      REQUIRE(GetEVectorOrdering(fes) == ElementDofOrdering::NATIVE);
      pa_mma.Assemble();
   }
   {
      MMAForce off(false);
      pa_sum.Assemble();
   }

   pa_mma.Mult(x, y_mma);
   pa_sum.Mult(x, y_sum);
   y_sum -= y_mma;
   REQUIRE(y_sum.Normlinf() == MFEM_Approx(0.0, 1e-9, 1e-9));

   REQUIRE_FALSE(UsesSimplexMMA(fes));
   REQUIRE(GetEVectorOrdering(fes) == ElementDofOrdering::LEXICOGRAPHIC);
}

/** Vector Mass/Diffusion simplex MMA PA vs full assembly (FA). */
void test_pa_vec_simplices_mma_fa(Mesh &mesh, int p, bool positive,
                                  int ir_order = -1)
{
   const int dim = mesh.Dimension();
   CAPTURE(dim, p, positive, ir_order, mesh.GetNE());
   MFEM_VERIFY(mesh.SpaceDimension() == dim, "");
   MFEM_VERIFY(!mesh.IsMixedMesh(), "");

   const int btype = positive ? BasisType::Positive : BasisType::GaussLobatto;
   H1_FECollection fec(p, dim, btype);
   FiniteElementSpace fes(&mesh, &fec, dim);

   if (positive)
   {
      MMAForce on(true);
      if (!UsesSimplexMMA(fes)) { return; }
   }
   else if (!UsesSimplexMMA(fes)) { return; }

   const auto &fe = *fes.GetTypicalFE();
   const auto &Tr = *mesh.GetTypicalElementTransformation();
   const int order = (ir_order < 0)
                     ? (2 * fe.GetOrder() + Tr.OrderW() + (positive ? 0 : 4))
                     : ir_order;
   const IntegrationRule *ir = positive
                               ? &StroudIntRules.Get(fe.GetGeomType(), order)
                               : &IntRules.Get(fe.GetGeomType(), order);
   CAPTURE(order, ir->GetNPoints());

   const int max_q1d = DeviceDofQuadLimits::Get().MAX_Q1D;
   const int max_nq = (dim == 2) ? max_q1d * max_q1d : 256;
   if (ir->GetNPoints() > max_nq) { return; }

   ConstantCoefficient const_coeff(M_2_SQRTPI);
   FunctionCoefficient funct_coeff([](const Vector &pt)
   { return M_1_PI + pt[0] * pt[0]; });

   GridFunction x(&fes), y_mma(&fes), y_fa(&fes);
   x.Randomize(0x100001b3);
   y_mma.Randomize(0x9e3779b9);
   y_fa = y_mma;

   BilinearForm pa(&fes), fa(&fes);
   auto add_vec = [&](BilinearForm &a)
   {
      auto *vm0 = new VectorMassIntegrator;
      auto *vm1 = new VectorMassIntegrator(const_coeff, ir);
      auto *vm2 = new VectorMassIntegrator(funct_coeff, ir);
      vm0->SetIntRule(ir);
      a.AddDomainIntegrator(vm0);
      a.AddDomainIntegrator(vm1);
      a.AddDomainIntegrator(vm2);
      a.AddDomainIntegrator(new VectorDiffusionIntegrator(ir));
      a.AddDomainIntegrator(new VectorDiffusionIntegrator(const_coeff, ir));
      a.AddDomainIntegrator(new VectorDiffusionIntegrator(funct_coeff, ir));
   };
   add_vec(pa);
   add_vec(fa);
   pa.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   // Legacy FA (FULL/EA not implemented for VectorMass/Diffusion).

   {
      MMAForce on(true);
      pa.Assemble();
   }
   fa.Assemble();
   fa.Finalize();

   pa.Mult(x, y_mma);
   fa.Mult(x, y_fa);
   y_fa -= y_mma;
   REQUIRE(y_fa.Normlinf() == MFEM_Approx(0.0, 1e-9, 1e-9));
}

TEST_CASE("PA Simplices MMA VectorMass/Diffusion vs FA",
          "[PartialAssembly][MMA][GPU]")
{
   const auto all_tests = launch_all_non_regression_tests;
   const auto p = !all_tests ? GENERATE(1, 2, 5) : GENERATE(1, 2, 3, 4, 5);

   SECTION("GLL 2D")
   {
      auto meshs = { "../../data/ref-triangle.mesh",
                     "../../data/inline-tri.mesh"
                   };
      Mesh mesh(GENERATE_REF(from_range(meshs)));
      test_pa_vec_simplices_mma_fa(mesh, p, false);
   }
   SECTION("GLL 3D")
   {
      auto meshs = { "../../data/ref-tetrahedron.mesh",
                     "../../data/inline-tet.mesh"
                   };
      Mesh mesh(GENERATE_REF(from_range(meshs)));
      test_pa_vec_simplices_mma_fa(mesh, p, false);
   }
   SECTION("Positive 2D")
   {
      Mesh mesh("../../data/ref-triangle.mesh");
      test_pa_vec_simplices_mma_fa(mesh, p, true);
   }
   SECTION("Positive 3D")
   {
      Mesh mesh("../../data/ref-tetrahedron.mesh");
      test_pa_vec_simplices_mma_fa(mesh, p, true);
   }
   SECTION("Fallback GLL triangle")
   {
      Mesh mesh("../../data/ref-triangle.mesh");
      test_pa_vec_simplices_mma_fa(mesh, 1, false, 5);
   }
}

/** Simplex VectorMass/Diffusion VQ or MQ MMA vs FA (smoke). */
void test_pa_vec_coeff_simplices_mma_fa(Mesh &mesh, int p, bool diffusion,
                                        bool mq)
{
   const int dim = mesh.Dimension();
   CAPTURE(dim, p, diffusion, mq, mesh.GetNE());

   H1_FECollection fec(p, dim, BasisType::GaussLobatto);
   FiniteElementSpace fes(&mesh, &fec, dim);
   if (!UsesSimplexMMA(fes)) { return; }

   const auto &fe = *fes.GetTypicalFE();
   const auto &Tr = *mesh.GetTypicalElementTransformation();
   const int order = 2 * fe.GetOrder() + Tr.OrderW() + 4;
   const IntegrationRule *ir = &IntRules.Get(fe.GetGeomType(), order);

   VectorFunctionCoefficient vq(dim, [](const Vector &pt, Vector &v)
   {
      for (int i = 0; i < v.Size(); ++i) { v(i) = M_1_PI + pt[0] + real_t(i); }
   });
   MatrixFunctionCoefficient mq_coeff(dim, [](const Vector &pt, DenseMatrix &m)
   {
      m = 0.0;
      for (int i = 0; i < m.Height(); ++i)
      {
         m(i, i) = 1.0 + M_1_PI * pt[0] + real_t(i);
         for (int j = 0; j < i; ++j)
         {
            m(i, j) = m(j, i) = 0.1 * (pt[0] + real_t(i + j));
         }
      }
   });

   GridFunction x(&fes), y_mma(&fes), y_fa(&fes);
   x.Randomize(0x100001b3);
   y_mma.Randomize(0x9e3779b9);
   y_fa = y_mma;

   BilinearForm pa(&fes), fa(&fes);
   auto add = [&](BilinearForm &a)
   {
      BilinearFormIntegrator *integ = nullptr;
      if (diffusion)
      {
         integ = mq ? static_cast<BilinearFormIntegrator *>(
                    new VectorDiffusionIntegrator(mq_coeff))
                 : static_cast<BilinearFormIntegrator *>(
                    new VectorDiffusionIntegrator(vq));
      }
      else
      {
         integ = mq ? static_cast<BilinearFormIntegrator *>(
                    new VectorMassIntegrator(mq_coeff))
                 : static_cast<BilinearFormIntegrator *>(
                    new VectorMassIntegrator(vq));
      }
      integ->SetIntRule(ir);
      a.AddDomainIntegrator(integ);
   };
   add(pa);
   add(fa);
   pa.SetAssemblyLevel(AssemblyLevel::PARTIAL);

   {
      MMAForce on(true);
      pa.Assemble();
   }
   fa.Assemble();
   fa.Finalize();

   pa.Mult(x, y_mma);
   fa.Mult(x, y_fa);
   y_fa -= y_mma;
   REQUIRE(y_fa.Normlinf() == MFEM_Approx(0.0, 1e-9, 1e-9));
}

TEST_CASE("PA Simplices MMA Vector VQ/MQ vs FA",
          "[PartialAssembly][MMA][GPU]")
{
   SECTION("2D VQ Mass")
   {
      Mesh mesh("../../data/ref-triangle.mesh");
      test_pa_vec_coeff_simplices_mma_fa(mesh, 2, false, false);
   }
   SECTION("2D MQ Mass")
   {
      Mesh mesh("../../data/ref-triangle.mesh");
      test_pa_vec_coeff_simplices_mma_fa(mesh, 2, false, true);
   }
   SECTION("2D VQ Diffusion")
   {
      Mesh mesh("../../data/ref-triangle.mesh");
      test_pa_vec_coeff_simplices_mma_fa(mesh, 2, true, false);
   }
   SECTION("2D MQ Diffusion")
   {
      Mesh mesh("../../data/ref-triangle.mesh");
      test_pa_vec_coeff_simplices_mma_fa(mesh, 2, true, true);
   }
   SECTION("3D VQ Mass")
   {
      Mesh mesh("../../data/ref-tetrahedron.mesh");
      test_pa_vec_coeff_simplices_mma_fa(mesh, 2, false, false);
   }
   SECTION("3D MQ Diffusion")
   {
      Mesh mesh("../../data/ref-tetrahedron.mesh");
      test_pa_vec_coeff_simplices_mma_fa(mesh, 2, true, true);
   }
}

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

static void fvec_dim(const Vector &xvec, Vector &v)
{
   const int dim = xvec.Size();
   real_t val = 2 * xvec[0];
   if (dim >= 2) { val += 3 * xvec[1] * xvec[0]; }
   if (dim >= 3) { val += real_t(0.25) * xvec[2] * xvec[1]; }
   v.SetSize(dim);
   for (int d = 0; d < dim; ++d) { v[d] = val / real_t(d + 1); }
}

/** ir_order < 0 → default (use_2p_ir ? 2*GetOrder() : MassIntegrator::GetRule). */
void test_vectorfe_domain_lf_simplex_mma(const char *filename, int p, bool hcurl,
                                         bool use_2p_ir, int ir_order = -1)
{
   CAPTURE(filename, p, hcurl, use_2p_ir, ir_order);

   Mesh mesh(filename);
   MFEM_VERIFY((mesh.Dimension() == 2 || mesh.Dimension() == 3),
               "Mesh dimension must be 2 or 3");
   MFEM_VERIFY(!mesh.IsMixedMesh(), "Mesh is mixed");
   MFEM_VERIFY(mesh.SpaceDimension() == mesh.Dimension(),
               "Simplex MMA requires volumetric meshes (sdim == dim)");

   const int dim = mesh.Dimension();
   std::unique_ptr<FiniteElementCollection> fec;
   if (hcurl) { fec.reset(new ND_FECollection(p, dim)); }
   else { fec.reset(new RT_FECollection(p, dim)); }
   FiniteElementSpace fes(&mesh, fec.get());

   {
      MMAForce on(true);
      if (hcurl) { REQUIRE(UsesSimplexMmaHcurl(fes)); }
      else { REQUIRE(UsesSimplexMmaHdiv(fes)); }
   }

   const auto &fe = *fes.GetTypicalFE();
   const auto &Tr = *mesh.GetTypicalElementTransformation();
   const IntegrationRule *ir = nullptr;
   if (ir_order >= 0)
   {
      ir = &IntRules.Get(fe.GetGeomType(), ir_order);
   }
   else if (use_2p_ir)
   {
      ir = &IntRules.Get(fe.GetGeomType(), 2 * fe.GetOrder());
   }
   else
   {
      ir = &MassIntegrator::GetRule(fe, fe, Tr);
   }
   CAPTURE(ir->GetNPoints());

   const int max_q1d = DeviceDofQuadLimits::Get().MAX_Q1D;
   const int max_nq = (dim == 2) ? max_q1d * max_q1d : 256;
   if (ir->GetNPoints() > max_nq) { return; }

   for (int e = 0; e < mesh.GetNE(); e++) { mesh.SetAttribute(e, e % 2 ? 1 : 2); }
   mesh.SetAttributes();
   Array<int> elem_marker(mesh.attributes.Max());
   elem_marker = 1;
   if (elem_marker.Size() >= 2) { elem_marker[0] = 0; }

   Vector cst(dim);
   cst = M_2_SQRTPI;
   VectorConstantCoefficient const_coeff(cst);
   VectorFunctionCoefficient funct_coeff(dim, fvec_dim);

   auto compare = [&](VectorCoefficient &Q)
   {
      MMAForce on(true);
      LinearForm lf_dev(&fes), lf_std(&fes);
      lf_dev.AddDomainIntegrator(new VectorFEDomainLFIntegrator(Q, ir),
                                 elem_marker);
      lf_std.AddDomainIntegrator(new VectorFEDomainLFIntegrator(Q, ir),
                                 elem_marker);

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

TEST_CASE("VectorFE DomainLF Simplices MMA",
          "[LinearFormExtension][MMA][GPU]")
{
   const auto all_tests = launch_all_non_regression_tests;
   const auto p = !all_tests ? GENERATE(1, 2, 5, 6) : GENERATE(1, 2, 3, 4, 5, 6);
   const auto use_2p_ir = GENERATE(false, true);
   const auto hcurl = GENERATE(true, false);

   SECTION("2D")
   {
      auto meshs = { "../../data/inline-tri.mesh",
                     "../../data/ref-triangle.mesh"
                   };
      test_vectorfe_domain_lf_simplex_mma(GENERATE_REF(from_range(meshs)), p,
                                          hcurl, use_2p_ir);
   }

   SECTION("3D")
   {
      auto meshs = { "../../data/inline-tet.mesh",
                     "../../data/ref-tetrahedron.mesh"
                   };
      test_vectorfe_domain_lf_simplex_mma(GENERATE_REF(from_range(meshs)), p,
                                          hcurl, use_2p_ir);
   }

   // Unregistered (D1D,nq) → AssembleSimplexMmaKernels::Fallback.
   SECTION("Fallback 2D triangle nq=7")
   {
      test_vectorfe_domain_lf_simplex_mma("../../data/ref-triangle.mesh", 1,
                                          hcurl, false, 5);
      test_vectorfe_domain_lf_simplex_mma("../../data/inline-tri.mesh", 1,
                                          hcurl, false, 5);
   }
   SECTION("Fallback 3D tet nq=35")
   {
      test_vectorfe_domain_lf_simplex_mma("../../data/ref-tetrahedron.mesh", 1,
                                          hcurl, false, 7);
      test_vectorfe_domain_lf_simplex_mma("../../data/inline-tet.mesh", 1,
                                          hcurl, false, 7);
   }
}

enum class VecFeOp { Mass, CurlCurl, DivDiv };

void test_hcurl_hdiv_simplex_fa_vs_mma(Mesh &mesh, int p, bool hcurl, VecFeOp op)
{
   const int dim = mesh.Dimension();
   CAPTURE(dim, p, hcurl, int(op), mesh.GetNE());

   std::unique_ptr<FiniteElementCollection> fec;
   if (hcurl) { fec.reset(new ND_FECollection(p, dim)); }
   else { fec.reset(new RT_FECollection(p, dim)); }
   FiniteElementSpace fes(&mesh, fec.get());

   {
      MMAForce on(true);
      if (hcurl) { REQUIRE(UsesSimplexMmaHcurl(fes)); }
      else { REQUIRE(UsesSimplexMmaHdiv(fes)); }
   }
   {
      MMAForce off(false);
      REQUIRE_FALSE((UsesSimplexMmaHcurl(fes) || UsesSimplexMmaHdiv(fes)));
   }

   GridFunction x(&fes), y_fa(&fes), y_mma(&fes);
   x.Randomize(0x100001b3);
   y_fa = 0.0;
   y_mma = 0.0;

   ConstantCoefficient one(1.0);
   ConstantCoefficient c2(M_2_SQRTPI);
   FunctionCoefficient cf([](const Vector &pt)
   { return M_1_PI + pt[0] * pt[0]; });

   const auto &fe = *fes.GetTypicalFE();
   ElementTransformation &T = *mesh.GetTypicalElementTransformation();
   const IntegrationRule *ir = &MassIntegrator::GetRule(fe, fe, T);

   auto add_ops = [&](BilinearForm &a)
   {
      if (op == VecFeOp::Mass)
      {
         auto *i0 = new VectorFEMassIntegrator;
         auto *i1 = new VectorFEMassIntegrator(c2);
         auto *i2 = new VectorFEMassIntegrator(cf);
         i0->SetIntRule(ir); i1->SetIntRule(ir); i2->SetIntRule(ir);
         a.AddDomainIntegrator(i0);
         a.AddDomainIntegrator(i1);
         a.AddDomainIntegrator(i2);
      }
      else if (op == VecFeOp::CurlCurl)
      {
         auto *i0 = new CurlCurlIntegrator;
         auto *i1 = new CurlCurlIntegrator(c2);
         auto *i2 = new CurlCurlIntegrator(cf);
         i0->SetIntRule(ir); i1->SetIntRule(ir); i2->SetIntRule(ir);
         a.AddDomainIntegrator(i0);
         a.AddDomainIntegrator(i1);
         a.AddDomainIntegrator(i2);
      }
      else
      {
         auto *i0 = new DivDivIntegrator(one);
         auto *i1 = new DivDivIntegrator(c2);
         auto *i2 = new DivDivIntegrator(cf);
         i0->SetIntRule(ir); i1->SetIntRule(ir); i2->SetIntRule(ir);
         a.AddDomainIntegrator(i0);
         a.AddDomainIntegrator(i1);
         a.AddDomainIntegrator(i2);
      }
   };

   BilinearForm fa(&fes), pa(&fes);
   add_ops(fa);
   add_ops(pa);
   pa.SetAssemblyLevel(AssemblyLevel::PARTIAL);

   fa.Assemble();
   fa.Finalize();
   {
      MMAForce on(true);
      pa.Assemble();
   }

   fa.Mult(x, y_fa);
   pa.Mult(x, y_mma);
   y_fa -= y_mma;
   REQUIRE(y_fa.Normlinf() == MFEM_Approx(0.0, 1e-8, 1e-8));
}

TEST_CASE("Hcurl/Hdiv simplex MMA PA vs FA",
          "[PA][MMA][Hcurl][Hdiv][Simplex][GPU]")
{
   const int p = GENERATE(1, 2, 3);
   SECTION("2D ND triangle")
   {
      Mesh mesh = Mesh::MakeCartesian2D(2, 2, Element::TRIANGLE);
      test_hcurl_hdiv_simplex_fa_vs_mma(mesh, p, true, VecFeOp::Mass);
      test_hcurl_hdiv_simplex_fa_vs_mma(mesh, p, true, VecFeOp::CurlCurl);
   }
   SECTION("2D RT triangle")
   {
      Mesh mesh = Mesh::MakeCartesian2D(2, 2, Element::TRIANGLE);
      test_hcurl_hdiv_simplex_fa_vs_mma(mesh, p, false, VecFeOp::Mass);
      test_hcurl_hdiv_simplex_fa_vs_mma(mesh, p, false, VecFeOp::DivDiv);
   }
   SECTION("3D ND tet")
   {
      Mesh mesh = Mesh::MakeCartesian3D(2, 2, 2, Element::TETRAHEDRON);
      test_hcurl_hdiv_simplex_fa_vs_mma(mesh, p, true, VecFeOp::Mass);
      test_hcurl_hdiv_simplex_fa_vs_mma(mesh, p, true, VecFeOp::CurlCurl);
   }
   SECTION("3D RT tet")
   {
      Mesh mesh = Mesh::MakeCartesian3D(2, 2, 2, Element::TETRAHEDRON);
      test_hcurl_hdiv_simplex_fa_vs_mma(mesh, p, false, VecFeOp::Mass);
      test_hcurl_hdiv_simplex_fa_vs_mma(mesh, p, false, VecFeOp::DivDiv);
   }
}

} // namespace pa_simplices_mma
