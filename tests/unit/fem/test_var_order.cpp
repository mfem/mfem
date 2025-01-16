// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
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

namespace mfem
{

static double exact_sln(const Vector &p);
static void TestSolve(FiniteElementSpace &fespace);

// Check basic functioning of variable order spaces, hp interpolation and
// some corner cases.
TEST_CASE("Variable Order FiniteElementSpace",
          "[FiniteElementCollection]"
          "[FiniteElementSpace]"
          "[NCMesh]")
{
   SECTION("Quad mesh")
   {
      // 2-element quad mesh
      Mesh mesh = Mesh::MakeCartesian2D(2, 1, Element::QUADRILATERAL);
      mesh.EnsureNCMesh();

      // standard H1 space with order 1 elements
      H1_FECollection fec(1, mesh.Dimension());
      FiniteElementSpace fespace(&mesh, &fec);

      REQUIRE(fespace.GetNDofs() == 6);
      REQUIRE(fespace.GetNConformingDofs() == 6);

      // convert to variable order space: p-refine second element
      fespace.SetElementOrder(1, 2);
      fespace.Update(false);

      REQUIRE(fespace.GetNDofs() == 11);
      REQUIRE(fespace.GetNConformingDofs() == 10);

      // h-refine first element in the y axis
      Array<Refinement> refs;
      refs.Append(Refinement(0, 2));
      mesh.GeneralRefinement(refs);
      fespace.Update();

      REQUIRE(fespace.GetNDofs() == 13);
      REQUIRE(fespace.GetNConformingDofs() == 11);

      // relax the master edge to be quadratic
      fespace.SetRelaxedHpConformity(true);

      REQUIRE(fespace.GetNDofs() == 13);
      REQUIRE(fespace.GetNConformingDofs() == 12);

      // increase order
      for (int i = 0; i < mesh.GetNE(); i++)
      {
         fespace.SetElementOrder(i, fespace.GetElementOrder(i) + 1);
      }
      fespace.Update(false);

      // 15 quadratic + 16 cubic DOFs - 2 shared vertices:
      REQUIRE(fespace.GetNDofs() == 29);
      // 3 constrained DOFs on slave side, inexact interpolation
      REQUIRE(fespace.GetNConformingDofs() == 26);

      // relaxed off
      fespace.SetRelaxedHpConformity(false);

      // new quadratic DOF on master edge:
      REQUIRE(fespace.GetNDofs() == 30);
      // 3 constrained DOFs on slave side, 2 on master side:
      REQUIRE(fespace.GetNConformingDofs() == 25);

      TestSolve(fespace);

      // refine
      mesh.UniformRefinement();
      fespace.Update();

      REQUIRE(fespace.GetNDofs() == 93);
      REQUIRE(fespace.GetNConformingDofs() == 83);

      TestSolve(fespace);
   }

   SECTION("Quad/hex mesh projection")
   {
      for (int dim=2; dim<=3; ++dim)
      {
         // 2-element mesh
         Mesh mesh = dim == 2 ? Mesh::MakeCartesian2D(2, 1, Element::QUADRILATERAL) :
                     Mesh::MakeCartesian3D(2, 1, 1, Element::HEXAHEDRON);
         mesh.EnsureNCMesh();

         // h-refine element 1
         Array<Refinement> refinements;
         refinements.Append(Refinement(1));

         int nonconformity_limit = 0; // 0 meaning allow unlimited ratio
         mesh.GeneralRefinement(refinements, 1, nonconformity_limit);  // h-refinement

         // standard H1 space with order 2 elements
         H1_FECollection fec(2, mesh.Dimension());
         FiniteElementSpace fespace(&mesh, &fec);

         GridFunction x(&fespace);

         // p-refine element 0
         fespace.SetElementOrder(0, 3);

         fespace.Update(false);
         x.SetSpace(&fespace);

         // Test projection of the coefficient
         FunctionCoefficient exsol(exact_sln);
         x.ProjectCoefficient(exsol);

         // Enforce space constraints on locally interpolated GridFunction x
         const SparseMatrix *R = fespace.GetHpRestrictionMatrix();
         const SparseMatrix *P = fespace.GetConformingProlongation();
         Vector y(fespace.GetTrueVSize());
         R->Mult(x, y);
         P->Mult(y, x);

         const double error = x.ComputeL2Error(exsol);
         REQUIRE(error == MFEM_Approx(0.0));
      }
   }

   SECTION("Hex mesh")
   {
      // 2-element hex mesh
      Mesh mesh = Mesh::MakeCartesian3D(2, 1, 1, Element::HEXAHEDRON);
      mesh.EnsureNCMesh();

      // standard H1 space with order 1 elements
      H1_FECollection fec(1, mesh.Dimension());
      FiniteElementSpace fespace(&mesh, &fec);

      REQUIRE(fespace.GetNDofs() == 12);
      REQUIRE(fespace.GetNConformingDofs() == 12);

      // convert to variable order space: p-refine second element
      fespace.SetElementOrder(1, 2);
      fespace.Update(false);

      REQUIRE(fespace.GetNDofs() == 31);
      REQUIRE(fespace.GetNConformingDofs() == 26);

      // h-refine first element in the z axis
      Array<Refinement> refs;
      refs.Append(Refinement(0, 4));
      mesh.GeneralRefinement(refs);
      fespace.Update();

      REQUIRE(fespace.GetNDofs() == 35);
      REQUIRE(fespace.GetNConformingDofs() == 28);

      // relax the master face to be quadratic
      fespace.SetRelaxedHpConformity(true);

      REQUIRE(fespace.GetNDofs() == 35);
      REQUIRE(fespace.GetNConformingDofs() == 31);

      // increase order
      for (int i = 0; i < mesh.GetNE(); i++)
      {
         fespace.SetElementOrder(i, fespace.GetElementOrder(i) + 1);
      }
      fespace.Update(false);

      REQUIRE(fespace.GetNDofs() == 105);
      REQUIRE(fespace.GetNConformingDofs() == 92);

      // relaxed off
      fespace.SetRelaxedHpConformity(false);

      REQUIRE(fespace.GetNDofs() == 108);
      REQUIRE(fespace.GetNConformingDofs() == 87);

      // refine one of the small elements into four
      refs[0].SetType(3);
      mesh.GeneralRefinement(refs);
      fespace.Update();

      REQUIRE(fespace.GetNDofs() == 162);
      REQUIRE(fespace.GetNConformingDofs() == 115);

      TestSolve(fespace);

      // lower the order of one of the four new elements to 1 - this minimum
      // order will propagate through two master faces and severely constrain
      // the space (since relaxed hp is off)
      fespace.SetElementOrder(0, 1);
      fespace.Update(false);

      REQUIRE(fespace.GetNDofs() == 152);
      REQUIRE(fespace.GetNConformingDofs() == 92);
   }

   SECTION("Prism mesh")
   {
      // 2-element prism mesh
      Mesh mesh = Mesh::MakeCartesian3D(1, 1, 1, Element::WEDGE);
      mesh.EnsureNCMesh();

      // standard H1 space with order 2 elements
      H1_FECollection fec(2, mesh.Dimension());
      FiniteElementSpace fespace(&mesh, &fec);

      REQUIRE(fespace.GetNDofs() == 27);
      REQUIRE(fespace.GetNConformingDofs() == 27);

      // convert to variable order space: p-refine first element
      fespace.SetElementOrder(0, 3);
      fespace.Update(false);

      REQUIRE(fespace.GetNDofs() == 54);
      REQUIRE(fespace.GetNConformingDofs() == 42);

      // refine to form an edge-face constraint similar to
      // https://github.com/mfem/mfem/pull/713#issuecomment-495786362
      Array<Refinement> refs;
      refs.Append(Refinement(1, 3));
      mesh.GeneralRefinement(refs);
      fespace.Update(false);

      refs[0].SetType(4);
      refs.Append(Refinement(2, 4));
      mesh.GeneralRefinement(refs);
      fespace.Update(false);

      REQUIRE(fespace.GetNDofs() == 113);
      REQUIRE(fespace.GetNConformingDofs() == 67);

      TestSolve(fespace);
   }
}


// Exact solution: x^2 + y^2 + z^2
static double exact_sln(const Vector &p)
{
   double x = p(0), y = p(1);
   if (p.Size() == 3)
   {
      double z = p(2);
      return x*x + y*y + z*z;
   }
   else
   {
      return x*x + y*y;
   }
}

static double exact_rhs(const Vector &p)
{
   return (p.Size() == 3) ? -6.0 : -4.0;
}

static void TestSolve(FiniteElementSpace &fespace)
{
   Mesh *mesh = fespace.GetMesh();

   // exact solution and RHS for the problem -\Delta u = 1
   FunctionCoefficient exsol(exact_sln);
   FunctionCoefficient rhs(exact_rhs);

   // set up Dirichlet BC on the boundary
   Array<int> ess_attr(mesh->bdr_attributes.Max());
   ess_attr = 1;

   Array<int> ess_tdof_list;
   fespace.GetEssentialTrueDofs(ess_attr, ess_tdof_list);

   GridFunction x(&fespace);
   x = 0.0;
   x.ProjectBdrCoefficient(exsol, ess_attr);

   // assemble the linear form
   LinearForm lf(&fespace);
   lf.AddDomainIntegrator(new DomainLFIntegrator(rhs));
   lf.Assemble();

   // assemble the bilinear form.
   BilinearForm bf(&fespace);
   bf.AddDomainIntegrator(new DiffusionIntegrator());
   bf.Assemble();

   OperatorPtr A;
   Vector B, X;
   bf.FormLinearSystem(ess_tdof_list, x, lf, A, X, B);

   // solve
   GSSmoother M((SparseMatrix&)(*A));
   PCG(*A, M, B, X, 0, 500, 1e-30, 0.0);

   bf.RecoverFEMSolution(X, lf, x);

   // compute L2 error from the exact solution
   double error = x.ComputeL2Error(exsol);

   REQUIRE(error == MFEM_Approx(0.0));

   // visualize
#ifdef MFEM_UNIT_DEBUG_VISUALIZE
   const char vishost[] = "localhost";
   const int  visport   = 19916;
   GridFunction *vis_x = ProlongToMaxOrder(&x);
   socketstream sol_sock(vishost, visport);
   sol_sock.precision(8);
   sol_sock << "solution\n" << *mesh << *vis_x;
   delete vis_x;
#endif
}

}
