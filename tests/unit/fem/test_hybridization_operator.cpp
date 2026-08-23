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

namespace hybridization_operator
{

// The grad-div problem of examples/ex4.cpp: (div u, div v) + (u, v) = (f, v)
// in an RT space, which is what BilinearForm::EnableHybridization() is built
// for.
void FExact(const Vector &x, Vector &f)
{
   f(0) = std::sin(M_PI * x(1)) + 0.3;
   f(1) = std::cos(M_PI * x(0)) - 0.2;
   if (x.Size() == 3) { f(2) = std::sin(M_PI * x(0) * x(1)); }
}

} // namespace hybridization_operator

TEST_CASE("Hybridization is an Operator over the constraint space",
          "[Hybridization]")
{
   using namespace hybridization_operator;

   const int order = GENERATE(1, 2);
   CAPTURE(order);

   Mesh mesh = Mesh::MakeCartesian2D(3, 3, Element::QUADRILATERAL, false,
                                     1.0, 1.0);
   const int dim = mesh.Dimension();

   RT_FECollection fec(order - 1, dim);
   FiniteElementSpace fes(&mesh, &fec);

   DG_Interface_FECollection c_fec(order - 1, dim);
   FiniteElementSpace c_fes(&mesh, &c_fec);

   Array<int> ess_tdofs;
   Array<int> ess_bdr(mesh.bdr_attributes.Max());
   ess_bdr = 1;
   fes.GetEssentialTrueDofs(ess_bdr, ess_tdofs);

   VectorFunctionCoefficient fcoeff(dim, FExact);
   ConstantCoefficient one(1.0);

   LinearForm b(&fes);
   b.AddDomainIntegrator(new VectorFEDomainLFIntegrator(fcoeff));
   b.Assemble();

   GridFunction x(&fes);
   x = 0.0;

   BilinearForm a(&fes);
   a.AddDomainIntegrator(new DivDivIntegrator(one));
   a.AddDomainIntegrator(new VectorFEMassIntegrator(one));
   a.EnableHybridization(&c_fes, new NormalTraceJumpIntegrator(), ess_tdofs);
   a.Assemble();

   OperatorPtr A;
   Vector B, X;
   a.FormLinearSystem(ess_tdofs, x, b, A, X, B);

   Hybridization *hybr = a.GetHybridization();
   REQUIRE(hybr != nullptr);

   SECTION("the constraint space and shape are what was handed in")
   {
      REQUIRE(hybr->ConstraintFESpace() == &c_fes);
      REQUIRE(hybr->Height() == hybr->Width());
      REQUIRE(hybr->Height() == X.Size());
   }

   SECTION("Mult agrees with the assembled reduced matrix")
   {
      // The class became an Operator on this branch; its action has to be the
      // matrix it assembles, or the two ways of using it disagree silently.
      SparseMatrix &H = hybr->GetMatrix();
      REQUIRE(H.Height() == hybr->Height());

      Vector v(hybr->Width()), y_op(hybr->Height()), y_mat(hybr->Height());
      for (int i = 0; i < v.Size(); i++)
      {
         v(i) = std::sin(1.7 * i) + 0.5 * std::cos(0.3 * i);
      }

      hybr->Mult(v, y_op);
      H.Mult(v, y_mat);

      y_op -= y_mat;
      REQUIRE(y_op.Normlinf() < 1e-11 * std::max(y_mat.Normlinf(),
                                                 real_t(1.0)));
   }

   SECTION("GetGradient is the same operator for a linear form")
   {
      Vector v(hybr->Width());
      v = 0.0;
      Operator &G = hybr->GetGradient(v);
      REQUIRE(G.Height() == hybr->Height());

      Vector w(hybr->Width()), y_g(hybr->Height()), y_m(hybr->Height());
      for (int i = 0; i < w.Size(); i++) { w(i) = std::cos(0.9 * i); }

      G.Mult(w, y_g);
      hybr->GetMatrix().Mult(w, y_m);
      y_g -= y_m;
      REQUIRE(y_g.Normlinf() < 1e-11 * std::max(y_m.Normlinf(), real_t(1.0)));
   }

   SECTION("the hybridized solve reproduces the direct one")
   {
      GSSmoother prec(*A.As<SparseMatrix>());
      CGSolver cg;
      cg.SetOperator(*A);
      cg.SetPreconditioner(prec);
      cg.SetRelTol(0.0);
      cg.SetAbsTol(1e-14);
      cg.SetMaxIter(5000);
      cg.Mult(B, X);
      REQUIRE(cg.GetConverged());
      a.RecoverFEMSolution(X, b, x);

      // The same problem with no hybridization at all.
      GridFunction x_ref(&fes);
      x_ref = 0.0;
      BilinearForm a_ref(&fes);
      a_ref.AddDomainIntegrator(new DivDivIntegrator(one));
      a_ref.AddDomainIntegrator(new VectorFEMassIntegrator(one));
      a_ref.Assemble();

      OperatorPtr A_ref;
      Vector B_ref, X_ref;
      a_ref.FormLinearSystem(ess_tdofs, x_ref, b, A_ref, X_ref, B_ref);

      GSSmoother prec_ref(*A_ref.As<SparseMatrix>());
      CGSolver cg_ref;
      cg_ref.SetOperator(*A_ref);
      cg_ref.SetPreconditioner(prec_ref);
      cg_ref.SetRelTol(0.0);
      cg_ref.SetAbsTol(1e-14);
      cg_ref.SetMaxIter(5000);
      cg_ref.Mult(B_ref, X_ref);
      REQUIRE(cg_ref.GetConverged());
      a_ref.RecoverFEMSolution(X_ref, b, x_ref);

      Vector diff(x);
      diff -= x_ref;
      REQUIRE(diff.Normlinf() < 1e-8 * std::max(x_ref.Normlinf(),
                                                real_t(1.0)));
   }
}

TEST_CASE("Hybridization boundary constraint integrator bookkeeping",
          "[Hybridization]")
{
   Mesh mesh = Mesh::MakeCartesian2D(2, 2, Element::QUADRILATERAL, false,
                                     1.0, 1.0);
   const int dim = mesh.Dimension();

   RT_FECollection fec(0, dim);
   FiniteElementSpace fes(&mesh, &fec);
   DG_Interface_FECollection c_fec(0, dim);
   FiniteElementSpace c_fes(&mesh, &c_fec);

   Hybridization hybr(&fes, &c_fes);
   REQUIRE(hybr.ConstraintFESpace() == &c_fes);
   REQUIRE(hybr.NumBdrConstraintIntegrators() == 0);

   BilinearFormIntegrator *i0 = new NormalTraceJumpIntegrator();
   BilinearFormIntegrator *i1 = new NormalTraceJumpIntegrator();
   Array<int> marker(mesh.bdr_attributes.Max());
   marker = 0;
   marker[0] = 1;

   hybr.AddBdrConstraintIntegrator(i0);            // no marker
   hybr.AddBdrConstraintIntegrator(i1, marker);    // with a marker

   REQUIRE(hybr.NumBdrConstraintIntegrators() == 2);
   REQUIRE(&hybr.GetBdrConstraintIntegrator(0) == i0);
   REQUIRE(&hybr.GetBdrConstraintIntegrator(1) == i1);

   // The documented contract: no marker means a null pointer, not an
   // all-ones array.
   REQUIRE(hybr.GetBdrConstraintIntegratorMarker(0) == nullptr);
   REQUIRE(hybr.GetBdrConstraintIntegratorMarker(1) == &marker);
}
