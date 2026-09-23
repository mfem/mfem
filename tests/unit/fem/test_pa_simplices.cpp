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

using namespace mfem;

namespace pa_kernels
{

void test_pa_simplices(const char *filename, int p)
{
   CAPTURE(filename, p);

   Mesh mesh(filename);
   if (mesh.GetTypicalElementGeometry() == Geometry::SQUARE ||
       mesh.GetTypicalElementGeometry() == Geometry::CUBE)
   {
      mesh = Mesh::MakeSimplicial(mesh);
   }
   const int dim = mesh.Dimension();

   MFEM_VERIFY(!mesh.IsMixedMesh(), "Mesh is mixed");

   H1_FECollection fec(p, dim, BasisType::Positive);
   FiniteElementSpace fes(&mesh, &fec);

   GridFunction x(&fes), y_fa(&fes), y_pa(&fes);
   x.Randomize(0x100001b3);
   y_fa.Randomize(0x9e3779b9);
   y_pa = y_fa;

   const auto &fe = *fes.GetTypicalFE();
   const auto &Tr = *mesh.GetTypicalElementTransformation();
   const auto order = 2 * fe.GetOrder() + Tr.OrderW();
   const auto *ir = &StroudIntRules.Get(fe.GetGeomType(), order);
   const auto *ir1 = &StroudIntRules.Get(fe.GetGeomType(), 2*fe.GetOrder()-1);

   ConstantCoefficient const_coeff(M_2_SQRTPI);
   FunctionCoefficient funct_coeff([](const Vector &x)
   { return M_1_PI + x[0] * x[0]; });

   BilinearForm fa(&fes), pa(&fes);
   fa.AddDomainIntegrator(new MassIntegrator(ir));
   fa.AddDomainIntegrator(new MassIntegrator(ir));
   fa.AddDomainIntegrator(new MassIntegrator(const_coeff, ir));
   fa.AddDomainIntegrator(new MassIntegrator(funct_coeff, ir));
   fa.AddDomainIntegrator(new DiffusionIntegrator(ir1));
   fa.AddDomainIntegrator(new DiffusionIntegrator(ir));
   fa.AddDomainIntegrator(new DiffusionIntegrator(const_coeff, ir));
   fa.AddDomainIntegrator(new DiffusionIntegrator(funct_coeff, ir));
   fa.Assemble();
   fa.Finalize();

   pa.AddDomainIntegrator(new MassIntegrator());
   pa.AddDomainIntegrator(new MassIntegrator(ir));
   pa.AddDomainIntegrator(new MassIntegrator(const_coeff, ir));
   pa.AddDomainIntegrator(new MassIntegrator(funct_coeff, ir));
   pa.AddDomainIntegrator(new DiffusionIntegrator());
   pa.AddDomainIntegrator(new DiffusionIntegrator(ir));
   pa.AddDomainIntegrator(new DiffusionIntegrator(const_coeff, ir));
   pa.AddDomainIntegrator(new DiffusionIntegrator(funct_coeff, ir));
   pa.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   pa.Assemble();

   fa.Mult(x, y_fa);
   pa.Mult(x, y_pa);
   y_fa -= y_pa;
   REQUIRE(y_fa.Norml2() == MFEM_Approx(0.0));
}

TEST_CASE("PA Simplices", "[PartialAssembly][Simplices][GPU]")
{
   const auto all_tests = launch_all_non_regression_tests;
   const auto p = !all_tests ? GENERATE(1, 2) : GENERATE(1, 2, 3, 4);

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
      auto extra = { "../../data/star-q2.mesh",
                     "../../data/star-q3.mesh",
                     "../../data/inline-quad.mesh",
                     "../../data/klein-donut.mesh",
                     "../../data/fichera-quad.mesh",
                     "../../data/periodic-square.mesh"
                   };
      test_pa_simplices(GenMesh(meshs, extra), p);
   }

   SECTION("3D")
   {
      auto meshs = { "../../data/beam-tet.mesh",
                     "../../data/inline-tet.mesh",
                     "../../data/ref-tetrahedron.mesh"
                   };
      auto extra = { "../../data/escher.mesh",
                     "../../data/escher-p2.mesh",
                     "../../data/inline-hex.mesh",
                     "../../data/fichera-q2.mesh",
                     "../../data/periodic-cube.mesh"
                   };
      test_pa_simplices(GenMesh(meshs, extra), p);
   }
}

// y(a, e) += sum_q B(q, a) w_q det J(q, e) u(q, e): the element action of a mass
// operator, from the values of u at the quadrature points.
static void AddMassAction(const DofToQuad &maps, const IntegrationRule &ir,
                          const GeometricFactors &geom, const Vector &u_q,
                          Vector &y)
{
   const int nq = ir.GetNPoints(), nd = maps.ndof;
   const int ne = u_q.Size() / nq;
   const auto B = Reshape(maps.B.Read(), nq, nd);
   const auto W = ir.GetWeights().Read();
   const auto detJ = Reshape(geom.detJ.Read(), nq, ne);
   const auto U = Reshape(u_q.Read(), nq, ne);
   auto Y = Reshape(y.ReadWrite(), nd, ne);
   mfem::forall(nd * ne, [=] MFEM_HOST_DEVICE (int i)
   {
      const int a = i % nd, e = i / nd;
      real_t sum = 0.0;
      for (int q = 0; q < nq; q++) { sum += B(q, a) * W[q] * detJ(q, e) * U(q, e); }
      Y(a, e) += sum;
   });
}

// A mass operator written the way a partial-assembly or matrix-free nonlinear-form
// integrator is: basis data in the dof ordering GetEVectorOrdering() names, and
// values at the points from the space's QuadratureInterpolator. It agrees with
// MassIntegrator only if the form hands it E-vectors in that ordering.
class EVectorMassIntegrator : public NonlinearFormIntegrator
{
   const FiniteElementSpace *fes = nullptr;
   const DofToQuad *maps = nullptr;
   const GeometricFactors *geom = nullptr;
   mutable Vector u_q;

   void Setup(const FiniteElementSpace &fespace)
   {
      fes = &fespace;
      const bool native =
         GetEVectorOrdering(fespace) == ElementDofOrdering::NATIVE;
      maps = &fespace.GetTypicalFE()->GetDofToQuad(
                *IntRule, native ? DofToQuad::FULL : DofToQuad::LEXICOGRAPHIC_FULL);
      geom = fespace.GetMesh()->GetGeometricFactors(
                *IntRule, GeometricFactors::DETERMINANTS);
      u_q.SetSize(IntRule->GetNPoints() * fespace.GetNE());
   }

   void Apply(const Vector &x, Vector &y) const
   {
      fes->GetQuadratureInterpolator(*IntRule)->Values(x, u_q);
      AddMassAction(*maps, *IntRule, *geom, u_q, y);
   }

public:
   EVectorMassIntegrator(const IntegrationRule &ir)
      : NonlinearFormIntegrator(&ir) { }

   using NonlinearFormIntegrator::AssemblePA;
   void AssemblePA(const FiniteElementSpace &fespace) override { Setup(fespace); }
   void AddMultPA(const Vector &x, Vector &y) const override { Apply(x, y); }
   void AssembleMF(const FiniteElementSpace &fespace) override { Setup(fespace); }
   void AddMultMF(const Vector &x, Vector &y) const override { Apply(x, y); }
};

TEST_CASE("PA NonlinearForm Simplices",
          "[PartialAssembly][NonlinearPA][Simplices][GPU]")
{
   // From degree 2, the lexicographic ordering of a nodal tetrahedron's dofs
   // differs from the native one.
   const int p = GENERATE(2, 3);
   const auto assembly = GENERATE(AssemblyLevel::PARTIAL, AssemblyLevel::NONE);
   CAPTURE(p, int(assembly));

   Mesh mesh = Mesh::MakeCartesian3D(2, 2, 2, Element::TETRAHEDRON);
   H1_FECollection fec(p, 3);
   FiniteElementSpace fes(&mesh, &fec);
   const IntegrationRule &ir = IntRules.Get(Geometry::TETRAHEDRON, 2*p);

   NonlinearForm nlf(&fes);
   nlf.SetAssemblyLevel(assembly);
   nlf.AddDomainIntegrator(new EVectorMassIntegrator(ir));
   nlf.Setup();

   BilinearForm mass(&fes);
   mass.AddDomainIntegrator(new MassIntegrator(&ir));
   mass.Assemble();
   mass.Finalize();

   Vector x(fes.GetVSize()), y_nlf(fes.GetVSize()), y_mass(fes.GetVSize());
   x.Randomize(1);
   nlf.Mult(x, y_nlf);
   mass.Mult(x, y_mass);
   y_nlf -= y_mass;
   REQUIRE(y_nlf.Normlinf() == MFEM_Approx(0.0));
}

} // namespace pa_kernels
