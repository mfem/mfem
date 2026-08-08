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
#include "fem/integ/mma/mma.hpp"

using namespace mfem;

namespace pa_tensors_mma
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

/** ir_order < 0 → specialized 2p+2 ((D1D,Q1D)=(p+1,p+2)).
    ir_order = 2p+5 → Fallback Q1D=p+3 (unregistered). */
void test_pa_tensors_mma(Mesh &mesh, int p, int ir_order = -1)
{
   const int dim = mesh.Dimension();
   const int order = (ir_order < 0) ? (2 * p + 2) : ir_order;
   CAPTURE(dim, p, order, mesh.GetNE());

   H1_FECollection fec(p, dim, BasisType::GaussLobatto);
   FiniteElementSpace fes(&mesh, &fec);

   {
      MMAForce on(true);
      REQUIRE(UsesTensorMMA(fes));
   }

   GridFunction x(&fes), y_mma(&fes), y_sum(&fes);
   x.Randomize(0x100001b3);
   y_mma.Randomize(0x9e3779b9);
   y_sum = y_mma;

   const auto &fe = *fes.GetTypicalFE();
   const IntegrationRule *ir = &IntRules.Get(fe.GetGeomType(), order);
   const DofToQuad &maps = fe.GetDofToQuad(*ir, DofToQuad::TENSOR);
   CAPTURE(maps.ndof, maps.nqpt);
   REQUIRE(maps.ndof == p + 1);
   REQUIRE(maps.nqpt <= internal::mma::TensorsMmaMaxQ1D);
   if (order == 2 * p + 5)
   {
      REQUIRE(maps.nqpt == p + 3);
   }

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

void test_pa_tensors_mma_cartesian(int dim, int p, int ir_order = -1)
{
   Mesh mesh = (dim == 2)
               ? Mesh::MakeCartesian2D(3, 3, Element::QUADRILATERAL)
               : Mesh::MakeCartesian3D(2, 2, 2, Element::HEXAHEDRON);
   test_pa_tensors_mma(mesh, p, ir_order);
}

} // namespace

TEST_CASE("Tensors MMA PA vs SUM-PA", "[MMA][GPU]")
{
   const int dim = GENERATE(2, 3);
   // p=2 uses SUM (m8n8k4 pad); MMA path starts at p>=3.
   const int p = GENERATE(3, 4, 5, 6, 7);

   SECTION("Specialized")
   {
      test_pa_tensors_mma_cartesian(dim, p);
   }
   SECTION("Fallback")
   {
      // Registered table is only (p+1,p+2); order 2p+5 → Q1D=p+3 <= MaxQ1D.
      if (p + 3 > internal::mma::TensorsMmaMaxQ1D) { return; }
      test_pa_tensors_mma_cartesian(dim, p, 2 * p + 5);
   }
}

TEST_CASE("Tensors MMA PA vs SUM-PA uneven NE", "[MMA][GPU]")
{
   // Partial serial NB batches: mass NB=8, diffusion NB=4.
   SECTION("quad 5x3")
   {
      const int p = GENERATE(3, 4, 7);
      Mesh mesh = Mesh::MakeCartesian2D(5, 3, Element::QUADRILATERAL);
      test_pa_tensors_mma(mesh, p);
   }
   SECTION("hex 3x3x2")
   {
      const int p = GENERATE(3, 4, 7);
      Mesh mesh = Mesh::MakeCartesian3D(3, 3, 2, Element::HEXAHEDRON);
      test_pa_tensors_mma(mesh, p);
   }
}

TEST_CASE("Tensors MMA PA vs SUM-PA on meshes", "[MMA][GPU]")
{
   // Paths assume cwd = build/tests/unit (MFEM unit-test convention).
   const int p = GENERATE(3, 4, 7);

   SECTION("2d")
   {
      const char *filename =
         GENERATE("../../data/inline-quad.mesh",
                  "../../data/periodic-square.mesh",
                  "../../data/fichera-quad.mesh");
      CAPTURE(filename);
      Mesh mesh(filename);
      REQUIRE(mesh.Dimension() == 2);
      test_pa_tensors_mma(mesh, p);
   }

   SECTION("3d")
   {
      const char *filename =
         GENERATE("../../data/fichera.mesh",
                  "../../data/fichera-q3.mesh",
                  "../../data/inline-hex.mesh",
                  "../../data/toroid-hex.mesh",
                  "../../data/periodic-cube.mesh");
      CAPTURE(filename);
      Mesh mesh(filename);
      REQUIRE(mesh.Dimension() == 3);
      test_pa_tensors_mma(mesh, p);
   }
}

/** VectorMass (scalar Q, block-diag) tensor MMA vs stock PA.
    ir_order < 0 → specialized 2p+2; ir_order = 2p+5 → Fallback Q1D=p+3. */
void test_pa_vecmass_tensors_mma(Mesh &mesh, int p, int ir_order = -1)
{
   const int dim = mesh.Dimension();
   const int order = (ir_order < 0) ? (2 * p + 2) : ir_order;
   CAPTURE(dim, p, order, mesh.GetNE());

   H1_FECollection fec(p, dim, BasisType::GaussLobatto);
   FiniteElementSpace fes(&mesh, &fec, dim); // vdim = dim

   {
      MMAForce on(true);
      REQUIRE(UsesTensorMMA(fes));
   }

   GridFunction x(&fes), y_mma(&fes), y_sum(&fes);
   x.Randomize(0x100001b3);
   y_mma.Randomize(0x9e3779b9);
   y_sum = y_mma;

   const auto &fe = *fes.GetTypicalFE();
   const IntegrationRule *ir = &IntRules.Get(fe.GetGeomType(), order);
   const DofToQuad &maps = fe.GetDofToQuad(*ir, DofToQuad::TENSOR);
   CAPTURE(maps.ndof, maps.nqpt);
   REQUIRE(maps.ndof == p + 1);
   REQUIRE(maps.nqpt <= internal::mma::TensorsMmaMaxQ1D);
   if (order == 2 * p + 5)
   {
      REQUIRE(maps.nqpt == p + 3);
   }

   ConstantCoefficient const_coeff(M_2_SQRTPI);
   FunctionCoefficient funct_coeff([](const Vector &pt)
   { return M_1_PI + pt[0] * pt[0]; });

   BilinearForm pa_mma(&fes), pa_sum(&fes);
   auto *vm_mma0 = new VectorMassIntegrator;
   auto *vm_mma1 = new VectorMassIntegrator(const_coeff, ir);
   auto *vm_mma2 = new VectorMassIntegrator(funct_coeff, ir);
   vm_mma0->SetIntRule(ir);
   pa_mma.AddDomainIntegrator(vm_mma0);
   pa_mma.AddDomainIntegrator(vm_mma1);
   pa_mma.AddDomainIntegrator(vm_mma2);
   auto *vm_sum0 = new VectorMassIntegrator;
   auto *vm_sum1 = new VectorMassIntegrator(const_coeff, ir);
   auto *vm_sum2 = new VectorMassIntegrator(funct_coeff, ir);
   vm_sum0->SetIntRule(ir);
   pa_sum.AddDomainIntegrator(vm_sum0);
   pa_sum.AddDomainIntegrator(vm_sum1);
   pa_sum.AddDomainIntegrator(vm_sum2);
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

void test_pa_vecmass_tensors_mma_cartesian(int dim, int p, int ir_order = -1)
{
   Mesh mesh = (dim == 2)
               ? Mesh::MakeCartesian2D(3, 3, Element::QUADRILATERAL)
               : Mesh::MakeCartesian3D(2, 2, 2, Element::HEXAHEDRON);
   test_pa_vecmass_tensors_mma(mesh, p, ir_order);
}

/** VectorDiffusion (scalar Q, block-diag) tensor MMA vs stock PA. */
void test_pa_vecdiffusion_tensors_mma(Mesh &mesh, int p, int ir_order = -1)
{
   const int dim = mesh.Dimension();
   const int order = (ir_order < 0) ? (2 * p + 2) : ir_order;
   CAPTURE(dim, p, order, mesh.GetNE());

   H1_FECollection fec(p, dim, BasisType::GaussLobatto);
   FiniteElementSpace fes(&mesh, &fec, dim);

   {
      MMAForce on(true);
      REQUIRE(UsesTensorMMA(fes));
   }

   GridFunction x(&fes), y_mma(&fes), y_sum(&fes);
   x.Randomize(0x100001b3);
   y_mma.Randomize(0x9e3779b9);
   y_sum = y_mma;

   const auto &fe = *fes.GetTypicalFE();
   const IntegrationRule *ir = &IntRules.Get(fe.GetGeomType(), order);
   const DofToQuad &maps = fe.GetDofToQuad(*ir, DofToQuad::TENSOR);
   CAPTURE(maps.ndof, maps.nqpt);
   REQUIRE(maps.ndof == p + 1);
   REQUIRE(maps.nqpt <= internal::mma::TensorsMmaMaxQ1D);
   if (order == 2 * p + 5)
   {
      REQUIRE(maps.nqpt == p + 3);
   }

   ConstantCoefficient const_coeff(M_2_SQRTPI);
   FunctionCoefficient funct_coeff([](const Vector &pt)
   { return M_1_PI + pt[0] * pt[0]; });

   BilinearForm pa_mma(&fes), pa_sum(&fes);
   auto *vd_mma0 = new VectorDiffusionIntegrator(ir);
   auto *vd_mma1 = new VectorDiffusionIntegrator(const_coeff, ir);
   auto *vd_mma2 = new VectorDiffusionIntegrator(funct_coeff, ir);
   pa_mma.AddDomainIntegrator(vd_mma0);
   pa_mma.AddDomainIntegrator(vd_mma1);
   pa_mma.AddDomainIntegrator(vd_mma2);
   auto *vd_sum0 = new VectorDiffusionIntegrator(ir);
   auto *vd_sum1 = new VectorDiffusionIntegrator(const_coeff, ir);
   auto *vd_sum2 = new VectorDiffusionIntegrator(funct_coeff, ir);
   pa_sum.AddDomainIntegrator(vd_sum0);
   pa_sum.AddDomainIntegrator(vd_sum1);
   pa_sum.AddDomainIntegrator(vd_sum2);
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

void test_pa_vecdiffusion_tensors_mma_cartesian(int dim, int p,
                                                int ir_order = -1)
{
   Mesh mesh = (dim == 2)
               ? Mesh::MakeCartesian2D(3, 3, Element::QUADRILATERAL)
               : Mesh::MakeCartesian3D(2, 2, 2, Element::HEXAHEDRON);
   test_pa_vecdiffusion_tensors_mma(mesh, p, ir_order);
}

/** VQ keeps MMA off; MMAForce on/off Mult must still agree (both stock). */
void test_pa_vec_vq_stays_stock(Mesh &mesh, int p, bool diffusion)
{
   const int dim = mesh.Dimension();
   CAPTURE(dim, p, diffusion, mesh.GetNE());

   H1_FECollection fec(p, dim, BasisType::GaussLobatto);
   FiniteElementSpace fes(&mesh, &fec, dim);

   {
      MMAForce on(true);
      REQUIRE(UsesTensorMMA(fes));
   }

   GridFunction x(&fes), y_on(&fes), y_off(&fes);
   x.Randomize(0x100001b3);
   y_on.Randomize(0x9e3779b9);
   y_off = y_on;

   const auto &fe = *fes.GetTypicalFE();
   const IntegrationRule *ir = &IntRules.Get(fe.GetGeomType(), 2 * p + 2);

   VectorFunctionCoefficient vq(dim, [](const Vector &pt, Vector &v)
   {
      for (int i = 0; i < v.Size(); ++i) { v(i) = M_1_PI + pt[0] + real_t(i); }
   });

   BilinearForm pa_on(&fes), pa_off(&fes);
   if (diffusion)
   {
      auto *on = new VectorDiffusionIntegrator(vq);
      auto *off = new VectorDiffusionIntegrator(vq);
      on->SetIntRule(ir);
      off->SetIntRule(ir);
      pa_on.AddDomainIntegrator(on);
      pa_off.AddDomainIntegrator(off);
   }
   else
   {
      auto *on = new VectorMassIntegrator(vq);
      auto *off = new VectorMassIntegrator(vq);
      on->SetIntRule(ir);
      off->SetIntRule(ir);
      pa_on.AddDomainIntegrator(on);
      pa_off.AddDomainIntegrator(off);
   }
   pa_on.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   pa_off.SetAssemblyLevel(AssemblyLevel::PARTIAL);

   {
      MMAForce on(true);
      pa_on.Assemble();
   }
   {
      MMAForce off(false);
      pa_off.Assemble();
   }

   pa_on.Mult(x, y_on);
   pa_off.Mult(x, y_off);
   y_off -= y_on;
   REQUIRE(y_off.Normlinf() == MFEM_Approx(0.0, 1e-9, 1e-9));
}

TEST_CASE("Tensors MMA VectorMass PA vs SUM-PA", "[MMA][GPU]")
{
   const int dim = GENERATE(2, 3);
   const int p = GENERATE(3, 4, 5, 6, 7);

   SECTION("Specialized")
   {
      test_pa_vecmass_tensors_mma_cartesian(dim, p);
   }
   SECTION("Fallback")
   {
      if (p + 3 > internal::mma::TensorsMmaMaxQ1D) { return; }
      test_pa_vecmass_tensors_mma_cartesian(dim, p, 2 * p + 5);
   }
}

TEST_CASE("Tensors MMA VectorMass PA vs SUM-PA uneven NE", "[MMA][GPU]")
{
   SECTION("quad 5x3")
   {
      const int p = GENERATE(3, 4, 7);
      Mesh mesh = Mesh::MakeCartesian2D(5, 3, Element::QUADRILATERAL);
      test_pa_vecmass_tensors_mma(mesh, p);
   }
   SECTION("hex 3x3x2")
   {
      const int p = GENERATE(3, 4, 7);
      Mesh mesh = Mesh::MakeCartesian3D(3, 3, 2, Element::HEXAHEDRON);
      test_pa_vecmass_tensors_mma(mesh, p);
   }
}

TEST_CASE("Tensors MMA VectorMass PA vs SUM-PA on meshes", "[MMA][GPU]")
{
   const int p = GENERATE(3, 4, 7);

   SECTION("2d")
   {
      const char *filename =
         GENERATE("../../data/inline-quad.mesh",
                  "../../data/periodic-square.mesh",
                  "../../data/fichera-quad.mesh");
      CAPTURE(filename);
      Mesh mesh(filename);
      REQUIRE(mesh.Dimension() == 2);
      test_pa_vecmass_tensors_mma(mesh, p);
   }

   SECTION("3d")
   {
      const char *filename =
         GENERATE("../../data/fichera.mesh",
                  "../../data/fichera-q3.mesh",
                  "../../data/inline-hex.mesh",
                  "../../data/toroid-hex.mesh",
                  "../../data/periodic-cube.mesh");
      CAPTURE(filename);
      Mesh mesh(filename);
      REQUIRE(mesh.Dimension() == 3);
      test_pa_vecmass_tensors_mma(mesh, p);
   }
}

TEST_CASE("Tensors MMA VectorDiffusion PA vs SUM-PA", "[MMA][GPU]")
{
   const int dim = GENERATE(2, 3);
   const int p = GENERATE(3, 4, 5, 6, 7);

   SECTION("Specialized")
   {
      test_pa_vecdiffusion_tensors_mma_cartesian(dim, p);
   }
   SECTION("Fallback")
   {
      if (p + 3 > internal::mma::TensorsMmaMaxQ1D) { return; }
      test_pa_vecdiffusion_tensors_mma_cartesian(dim, p, 2 * p + 5);
   }
}

TEST_CASE("Tensors MMA VectorDiffusion PA vs SUM-PA uneven NE", "[MMA][GPU]")
{
   SECTION("quad 5x3")
   {
      const int p = GENERATE(3, 4, 7);
      Mesh mesh = Mesh::MakeCartesian2D(5, 3, Element::QUADRILATERAL);
      test_pa_vecdiffusion_tensors_mma(mesh, p);
   }
   SECTION("hex 3x3x2")
   {
      const int p = GENERATE(3, 4, 7);
      Mesh mesh = Mesh::MakeCartesian3D(3, 3, 2, Element::HEXAHEDRON);
      test_pa_vecdiffusion_tensors_mma(mesh, p);
   }
}

TEST_CASE("Tensors MMA VectorDiffusion PA vs SUM-PA on meshes", "[MMA][GPU]")
{
   const int p = GENERATE(3, 4, 7);

   SECTION("2d")
   {
      const char *filename =
         GENERATE("../../data/inline-quad.mesh",
                  "../../data/periodic-square.mesh",
                  "../../data/fichera-quad.mesh");
      CAPTURE(filename);
      Mesh mesh(filename);
      REQUIRE(mesh.Dimension() == 2);
      test_pa_vecdiffusion_tensors_mma(mesh, p);
   }

   SECTION("3d")
   {
      const char *filename =
         GENERATE("../../data/fichera.mesh",
                  "../../data/fichera-q3.mesh",
                  "../../data/inline-hex.mesh",
                  "../../data/toroid-hex.mesh",
                  "../../data/periodic-cube.mesh");
      CAPTURE(filename);
      Mesh mesh(filename);
      REQUIRE(mesh.Dimension() == 3);
      test_pa_vecdiffusion_tensors_mma(mesh, p);
   }
}

TEST_CASE("Tensors MMA Vector VQ stays on stock PA", "[MMA][GPU]")
{
   const int dim = GENERATE(2, 3);
   const int p = 3;
   Mesh mesh = (dim == 2)
               ? Mesh::MakeCartesian2D(3, 3, Element::QUADRILATERAL)
               : Mesh::MakeCartesian3D(2, 2, 2, Element::HEXAHEDRON);

   SECTION("VectorMass")
   {
      test_pa_vec_vq_stays_stock(mesh, p, false);
   }
   SECTION("VectorDiffusion")
   {
      test_pa_vec_vq_stays_stock(mesh, p, true);
   }
}

TEST_CASE("Tensors MMA eligibility", "[MMA][GPU]")
{
   Mesh mesh = Mesh::MakeCartesian3D(2, 2, 2, Element::HEXAHEDRON);
   H1_FECollection fec(3, 3, BasisType::GaussLobatto);
   FiniteElementSpace fes(&mesh, &fec);

   {
      MMAForce off(false);
      REQUIRE_FALSE(UsesTensorMMA(fes));
   }

   {
      MMAForce on(true);
      REQUIRE(UsesTensorMMA(fes));
   }

   REQUIRE_FALSE(UsesTensorMMA(fes));

   // p=1 and p=2 are intentionally unsupported (pad / SUM preferred)
   H1_FECollection fec1(1, 3, BasisType::GaussLobatto);
   FiniteElementSpace fes1(&mesh, &fec1);
   {
      MMAForce on(true);
      REQUIRE_FALSE(UsesTensorMMA(fes1));
      H1_FECollection fec2(2, 3, BasisType::GaussLobatto);
      FiniteElementSpace fes2(&mesh, &fec2);
      REQUIRE_FALSE(UsesTensorMMA(fes2));
   }
}

} // namespace pa_tensors_mma
