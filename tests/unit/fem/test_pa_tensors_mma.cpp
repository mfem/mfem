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

enum class Kind { Scalar, VectorMass, VectorDiffusion };

namespace
{

void AddIntegrators(Kind kind, BilinearForm &a, const IntegrationRule *ir,
                    Coefficient &const_coeff, Coefficient &funct_coeff)
{
   switch (kind)
   {
      case Kind::Scalar:
         a.AddDomainIntegrator(new MassIntegrator(ir));
         a.AddDomainIntegrator(new MassIntegrator(const_coeff, ir));
         a.AddDomainIntegrator(new MassIntegrator(funct_coeff, ir));
         a.AddDomainIntegrator(new DiffusionIntegrator(ir));
         a.AddDomainIntegrator(new DiffusionIntegrator(const_coeff, ir));
         a.AddDomainIntegrator(new DiffusionIntegrator(funct_coeff, ir));
         break;
      case Kind::VectorMass:
      {
         auto *i0 = new VectorMassIntegrator;
         auto *i1 = new VectorMassIntegrator(const_coeff, ir);
         auto *i2 = new VectorMassIntegrator(funct_coeff, ir);
         i0->SetIntRule(ir);
         a.AddDomainIntegrator(i0);
         a.AddDomainIntegrator(i1);
         a.AddDomainIntegrator(i2);
         break;
      }
      case Kind::VectorDiffusion:
         a.AddDomainIntegrator(new VectorDiffusionIntegrator(ir));
         a.AddDomainIntegrator(new VectorDiffusionIntegrator(const_coeff, ir));
         a.AddDomainIntegrator(new VectorDiffusionIntegrator(funct_coeff, ir));
         break;
   }
}

/** ir_order < 0 → specialized 2p+2 ((D1D,Q1D)=(p+1,p+2)).
    ir_order = 2p+5 → Fallback Q1D=p+3 (unregistered). */
void test_pa_tensors_mma(Mesh &mesh, int p, Kind kind, int ir_order = -1)
{
   const int dim = mesh.Dimension();
   const int order = (ir_order < 0) ? (2 * p + 2) : ir_order;
   const int vdim = (kind == Kind::Scalar) ? 1 : dim;
   CAPTURE(dim, p, order, int(kind), mesh.GetNE());

   H1_FECollection fec(p, dim, BasisType::GaussLobatto);
   FiniteElementSpace fes(&mesh, &fec, vdim);

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
   AddIntegrators(kind, pa_mma, ir, const_coeff, funct_coeff);
   AddIntegrators(kind, pa_sum, ir, const_coeff, funct_coeff);
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

void test_pa_tensors_mma_cartesian(int dim, int p, Kind kind, int ir_order = -1)
{
   Mesh mesh = (dim == 2)
               ? Mesh::MakeCartesian2D(3, 3, Element::QUADRILATERAL)
               : Mesh::MakeCartesian3D(2, 2, 2, Element::HEXAHEDRON);
   test_pa_tensors_mma(mesh, p, kind, ir_order);
}

/** VQ/MQ VectorMass/VectorDiffusion: MMA Mult matches stock SUM-PA. */
void test_pa_vec_coeff_tensors_mma(Mesh &mesh, int p, bool diffusion, bool mq)
{
   const int dim = mesh.Dimension();
   CAPTURE(dim, p, diffusion, mq, mesh.GetNE());

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
   const IntegrationRule *ir = &IntRules.Get(fe.GetGeomType(), 2 * p + 2);

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

   BilinearForm pa_mma(&fes), pa_sum(&fes);
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
   add(pa_mma);
   add(pa_sum);
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

} // namespace

TEST_CASE("Tensors MMA PA vs SUM-PA", "[MMA][GPU]")
{
   const Kind kind = GENERATE(Kind::Scalar, Kind::VectorMass,
                              Kind::VectorDiffusion);
   const int dim = GENERATE(2, 3);
   // p=2 uses SUM (m8n8k4 pad); MMA path starts at p>=3.
   const int p = GENERATE(3, 4, 5, 6, 7);

   SECTION("Specialized")
   {
      test_pa_tensors_mma_cartesian(dim, p, kind);
   }
   SECTION("Fallback")
   {
      // Registered table is only (p+1,p+2); order 2p+5 → Q1D=p+3 <= MaxQ1D.
      if (p + 3 > internal::mma::TensorsMmaMaxQ1D) { return; }
      test_pa_tensors_mma_cartesian(dim, p, kind, 2 * p + 5);
   }
}

TEST_CASE("Tensors MMA PA vs SUM-PA uneven NE", "[MMA][GPU]")
{
   const Kind kind = GENERATE(Kind::Scalar, Kind::VectorMass,
                              Kind::VectorDiffusion);
   // Partial serial NB batches: mass NB=8, diffusion NB=4.
   SECTION("quad 5x3")
   {
      const int p = GENERATE(3, 4, 7);
      Mesh mesh = Mesh::MakeCartesian2D(5, 3, Element::QUADRILATERAL);
      test_pa_tensors_mma(mesh, p, kind);
   }
   SECTION("hex 3x3x2")
   {
      const int p = GENERATE(3, 4, 7);
      Mesh mesh = Mesh::MakeCartesian3D(3, 3, 2, Element::HEXAHEDRON);
      test_pa_tensors_mma(mesh, p, kind);
   }
}

TEST_CASE("Tensors MMA PA vs SUM-PA on meshes", "[MMA][GPU]")
{
   // Paths assume cwd = build/tests/unit (MFEM unit-test convention).
   const Kind kind = GENERATE(Kind::Scalar, Kind::VectorMass,
                              Kind::VectorDiffusion);
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
      test_pa_tensors_mma(mesh, p, kind);
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
      test_pa_tensors_mma(mesh, p, kind);
   }
}

TEST_CASE("Tensors MMA Vector VQ/MQ vs SUM-PA", "[MMA][GPU]")
{
   const int dim = GENERATE(2, 3);
   const int p = 3;
   Mesh mesh = (dim == 2)
               ? Mesh::MakeCartesian2D(3, 3, Element::QUADRILATERAL)
               : Mesh::MakeCartesian3D(2, 2, 2, Element::HEXAHEDRON);

   SECTION("VectorMass VQ")
   {
      test_pa_vec_coeff_tensors_mma(mesh, p, false, false);
   }
   SECTION("VectorMass MQ")
   {
      test_pa_vec_coeff_tensors_mma(mesh, p, false, true);
   }
   SECTION("VectorDiffusion VQ")
   {
      test_pa_vec_coeff_tensors_mma(mesh, p, true, false);
   }
   SECTION("VectorDiffusion MQ")
   {
      test_pa_vec_coeff_tensors_mma(mesh, p, true, true);
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
