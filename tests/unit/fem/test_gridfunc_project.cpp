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

#include "mfem.hpp"
#include "unit_tests.hpp"

using namespace mfem;

namespace gridfunc_project
{

// Exactly representable by any space of order >= 2.
real_t Quadratic(const Vector &x)
{
   real_t v = 1.0 + 2.0 * x(0) - 0.5 * x(1) + 0.75 * x(0) * x(1);
   return v + 0.25 * x(0) * x(0);
}

// Not in any polynomial space, so the projection variants must differ.
real_t Transcendental(const Vector &x)
{
   return std::exp(x(0)) * std::sin(3.0 * x(1)) + std::cos(4.0 * x(0));
}

// Degree five, so it is outside every space used below, but every quadrature
// rule involved integrates it exactly -- which lets the L2 projection
// equations be checked to round-off rather than to quadrature accuracy.
real_t Quintic(const Vector &x)
{
   const real_t s = x(0) + 0.5 * x(1);
   return 1.0 + s * s * s * s * s - 0.3 * x(1) * x(1) * x(1);
}

real_t L2Error(GridFunction &u, Coefficient &c)
{
   const int quad_order = 20;
   const IntegrationRule *irs[Geometry::NumGeom];
   for (int i = 0; i < Geometry::NumGeom; i++)
   {
      irs[i] = &(IntRules.Get(i, quad_order));
   }
   return u.ComputeL2Error(c, irs);
}

} // namespace gridfunc_project

TEST_CASE("GridFunction ProjectType variants agree where they must",
          "[GridFunction][ProjectType]")
{
   using namespace gridfunc_project;

   const int order = GENERATE(2, 3);
   const bool discontinuous = GENERATE(false, true);
   CAPTURE(order, discontinuous);

   Mesh mesh = Mesh::MakeCartesian2D(3, 2, Element::QUADRILATERAL, false,
                                     1.0, 1.0);
   const int dim = mesh.Dimension();

   std::unique_ptr<FiniteElementCollection> fec;
   if (discontinuous) { fec.reset(new L2_FECollection(order, dim)); }
   else               { fec.reset(new H1_FECollection(order, dim)); }
   FiniteElementSpace fes(&mesh, fec.get());

   SECTION("all variants reproduce a function that is in the space")
   {
      FunctionCoefficient c(Quadratic);

      GridFunction a(&fes), b(&fes), d(&fes), e(&fes);
      a.ProjectCoefficient(c, ProjectType::DEFAULT);
      b.ProjectCoefficient(c, ProjectType::ELEMENT);
      d.ProjectCoefficient(c, ProjectType::ELEMENT_L2);
      e.ProjectCoefficient(c, ProjectType::GLOBAL_L2);

      for (int i = 0; i < a.Size(); i++)
      {
         CAPTURE(i);
         REQUIRE(b(i) == MFEM_Approx(a(i), 1e-10, 1e-9));
         REQUIRE(d(i) == MFEM_Approx(a(i), 1e-10, 1e-9));
         REQUIRE(e(i) == MFEM_Approx(a(i), 1e-10, 1e-9));
      }
   }

   SECTION("GLOBAL_L2 satisfies the L2 projection equations")
   {
      // The defining property, checked against an independently assembled
      // mass matrix and load vector rather than against another projection.
      FunctionCoefficient c(Quintic);

      GridFunction u(&fes);
      u.ProjectCoefficient(c, ProjectType::GLOBAL_L2);

      BilinearForm m(&fes);
      m.AddDomainIntegrator(new MassIntegrator);
      m.Assemble();
      m.Finalize();

      // The same default quadrature the projection itself uses for its load
      // vector, so that what is left is the linear solve and not a difference
      // between two integration rules.
      LinearForm b(&fes);
      b.AddDomainIntegrator(new DomainLFIntegrator(c));
      b.Assemble();

      Vector Mu(u.Size());
      m.SpMat().Mult(u, Mu);
      Mu -= b;
      INFO("residual of M u = b is " << Mu.Normlinf());
      REQUIRE(Mu.Normlinf() < 1e-8 * std::max(b.Normlinf(), real_t(1.0)));
   }

   SECTION("element-local and global L2 coincide only on a broken space")
   {
      FunctionCoefficient c(Transcendental);

      GridFunction ge(&fes), gg(&fes), gi(&fes);
      ge.ProjectCoefficient(c, ProjectType::ELEMENT_L2);
      gg.ProjectCoefficient(c, ProjectType::GLOBAL_L2);
      gi.ProjectCoefficient(c, ProjectType::ELEMENT);

      Vector diff(ge);
      diff -= gg;

      if (discontinuous)
      {
         // The mass matrix is block diagonal, so the two are the same solve.
         REQUIRE(diff.Normlinf() < 1e-9);
      }
      else
      {
         REQUIRE(diff.Normlinf() > 1e-6);
      }

      // And the global L2 projection is the best approximation in L2, so it
      // cannot be beaten by interpolation.
      const real_t err_g = L2Error(gg, c);
      const real_t err_i = L2Error(gi, c);
      const real_t err_e = L2Error(ge, c);
      INFO("global " << err_g << ", element L2 " << err_e
           << ", interpolation " << err_i);
      REQUIRE(err_g <= err_i * (1.0 + 1e-10));
      REQUIRE(err_g <= err_e * (1.0 + 1e-10));
   }
}

TEST_CASE("GridFunction::GetValue on a face of an H(div) space",
          "[GridFunction]")
{
   using namespace gridfunc_project;

   // Evaluating on a face is what the FACE branch of GetValue() allows for any
   // space that is not discontinuous; for an RT space that is the normal
   // trace. The mesh is deliberately anisotropic so that the face Jacobian is
   // not one, which distinguishes u.n from u.n |J|.
   const int order = GENERATE(0, 1, 2);
   CAPTURE(order);

   Mesh mesh = Mesh::MakeCartesian2D(2, 1, Element::QUADRILATERAL, false,
                                     2.0, 3.0);
   const int dim = mesh.Dimension();

   RT_FECollection fec(order, dim);
   FiniteElementSpace fes(&mesh, &fec);

   Vector c(dim);
   c(0) = 3.0;
   c(1) = -1.0;
   VectorConstantCoefficient vc(c);      // in RT of every order

   GridFunction u(&fes);
   u.ProjectCoefficient(vc);

   int f = -1;
   for (int i = 0; i < mesh.GetNumFaces(); i++)
   {
      if (mesh.FaceIsInterior(i)) { f = i; break; }
   }
   REQUIRE(f >= 0);

   FaceElementTransformations *Tr = mesh.GetFaceElementTransformations(f);

   IntegrationPoint ip;
   ip.Set2(0.37, 0.0);
   Tr->SetAllIntPoints(&ip);

   Vector nor(dim);
   CalcOrtho(Tr->Jacobian(), nor);
   const real_t len = nor.Norml2();
   REQUIRE(len > 1.5);          // the Jacobian is genuinely not one

   Vector uv(dim);
   u.GetVectorValue(*Tr->Elem1, Tr->GetElement1IntPoint(), uv);
   const real_t un = (uv * nor) / len;

   const real_t face_value = u.GetValue(*Tr, ip);
   INFO("face value " << face_value << " vs u.n " << un
        << " (|J| = " << len << ")");
   REQUIRE(face_value == MFEM_Approx(un, 1e-10, 1e-9));

   // The normal trace is single valued, which is the whole point of RT.
   Vector uv2(dim);
   u.GetVectorValue(*Tr->Elem2, Tr->GetElement2IntPoint(), uv2);
   REQUIRE((uv2 * nor) / len == MFEM_Approx(un, 1e-10, 1e-9));
}

TEST_CASE("GridFunction::GetValue on a face of an H1 space", "[GridFunction]")
{
   using namespace gridfunc_project;

   const int order = GENERATE(1, 2);
   CAPTURE(order);

   Mesh mesh = Mesh::MakeCartesian2D(2, 1, Element::QUADRILATERAL, false,
                                     2.0, 3.0);
   H1_FECollection fec(order, mesh.Dimension());
   FiniteElementSpace fes(&mesh, &fec);

   FunctionCoefficient c(Quadratic);
   GridFunction u(&fes);
   u.ProjectCoefficient(c);

   int f = -1;
   for (int i = 0; i < mesh.GetNumFaces(); i++)
   {
      if (mesh.FaceIsInterior(i)) { f = i; break; }
   }

   FaceElementTransformations *Tr = mesh.GetFaceElementTransformations(f);
   IntegrationPoint ip;
   ip.Set2(0.42, 0.0);
   Tr->SetAllIntPoints(&ip);

   const real_t face_value = u.GetValue(*Tr, ip);
   const real_t elem_value = u.GetValue(*Tr->Elem1, Tr->GetElement1IntPoint());
   REQUIRE(face_value == MFEM_Approx(elem_value, 1e-10, 1e-9));
}

TEST_CASE("Projecting a VectorDeltaCoefficient", "[GridFunction]")
{
   using namespace gridfunc_project;

   // A delta projected onto a space is normalized so that its integral is the
   // coefficient's scale; for the vector form that must hold component by
   // component, along the given direction.
   const int order = GENERATE(1, 2);
   CAPTURE(order);

   Mesh mesh = Mesh::MakeCartesian2D(4, 4, Element::QUADRILATERAL, false,
                                     1.0, 1.0);
   const int dim = mesh.Dimension();

   H1_FECollection fec(order, dim);
   FiniteElementSpace fes_scalar(&mesh, &fec);
   FiniteElementSpace fes(&mesh, &fec, dim);
   const int nd = fes_scalar.GetNDofs();

   Vector dir(dim);
   dir(0) = 2.0;
   dir(1) = -3.0;
   const real_t scale = 1.5;

   // Two things this call has to get right, both of which silently produce a
   // zero field rather than an error. The four-argument form is (dir, x, y,
   // s); the three-argument overload is (dir, x, s), which would place the
   // delta at (0.37, 0, 0). And the centre must lie within DeltaCoefficient's
   // tolerance of a mesh vertex -- ProjectVectorDeltaCoefficient() searches
   // for the nearest vertex and returns an all-zero projection if none is
   // close enough. (0.25, 0.5) is a vertex of this 4x4 mesh.
   VectorDeltaCoefficient vdelta(dir, 0.25, 0.5, scale);

   GridFunction u(&fes);
   u.ProjectCoefficient(vdelta);

   // Integrate each component against one.
   ConstantCoefficient one(1.0);
   LinearForm w(&fes_scalar);
   w.AddDomainIntegrator(new DomainLFIntegrator(one, 4, 8));
   w.Assemble();

   for (int d = 0; d < dim; d++)
   {
      real_t integral = 0.0;
      for (int i = 0; i < nd; i++) { integral += w[i] * u(d * nd + i); }
      CAPTURE(d, integral);
      REQUIRE(integral == MFEM_Approx(scale * dir(d), 1e-9, 1e-8));
   }
}
