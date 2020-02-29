// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#include "catch.hpp"
#include "mfem.hpp"

namespace mfem
{

double f1(const Vector &x)
{
   double r = cos(x(0)) + sin(x(1));
   if (x.Size() == 3)
   {
      r += cos(x(2));
   }
   return r;
}

void gradf1(const Vector &x, Vector &u)
{
   u(0) = -sin(x(0));
   u(1) = cos(x(1));
   if (x.Size() == 3)
   {
      u(2) = -sin(x(2));
   }
}

TEST_CASE("FormRectangular", "[FormRectangularSystemMatrix]")
{
   SECTION("MixedBilinearForm::FormRectangularSystemMatrix")
   {
      Mesh mesh(10, 10, Element::QUADRILATERAL, 0, 1.0, 1.0);
      int dim = mesh.Dimension();
      int order = 2;

      int nattr = mesh.bdr_attributes.Max();
      Array<int> ess_trial_tdof_list, ess_test_tdof_list;

      Array<int> ess_bdr(nattr);
      ess_bdr = 0;
      ess_bdr[0] = 1;

      // Scalar
      H1_FECollection fec1(order, dim);
      FiniteElementSpace fes1(&mesh, &fec1);

      // Vector valued
      H1_FECollection fec2(order, dim);
      FiniteElementSpace fes2(&mesh, &fec2, dim);

      fes1.GetEssentialTrueDofs(ess_bdr, ess_trial_tdof_list);
      fes2.GetEssentialTrueDofs(ess_bdr, ess_test_tdof_list);

      GridFunction field(&fes1), field2(&fes2);

      MixedBilinearForm gform(&fes1, &fes2);
      gform.AddDomainIntegrator(new GradientIntegrator);
      gform.Assemble();

      // Project u = f1
      FunctionCoefficient fcoeff1(f1);
      field.ProjectCoefficient(fcoeff1);

      VectorFunctionCoefficient fcoeff2(dim, gradf1);
      LinearForm lf(&fes2);
      lf.AddDomainIntegrator(new VectorDomainLFIntegrator(fcoeff2));
      lf.Assemble();

      OperatorHandle G;
      Vector X, B;
      gform.FormRectangularLinearSystem(ess_trial_tdof_list,
                                        ess_test_tdof_list,
                                        field,
                                        lf,
                                        G,
                                        X,
                                        B);

      G->Mult(field, field2);

      subtract(B, field2, field2);
      REQUIRE(field2.Norml2() == Approx(0.0));
   }
}

#ifdef MFEM_USE_MPI

TEST_CASE("ParallelFormRectangular",
          "[Parallel], [FormRectangularSystemMatrix]")
{
   SECTION("ParMixedBilinearForm::FormRectangularSystemMatrix")
   {
      Mesh mesh(10, 10, Element::QUADRILATERAL, 0, 1.0, 1.0);
      int dim = mesh.Dimension();
      int order = 2;

      int nattr = mesh.bdr_attributes.Max();
      Array<int> ess_trial_tdof_list, ess_test_tdof_list;

      Array<int> ess_bdr(nattr);
      ess_bdr = 0;
      ess_bdr[0] = 1;

      ParMesh pmesh(MPI_COMM_WORLD, mesh);

      // Scalar
      H1_FECollection fec1(order, dim);
      ParFiniteElementSpace fes1(&pmesh, &fec1);

      // Vector valued
      H1_FECollection fec2(order, dim);
      ParFiniteElementSpace fes2(&pmesh, &fec2, dim);

      fes1.GetEssentialTrueDofs(ess_bdr, ess_trial_tdof_list);
      fes2.GetEssentialTrueDofs(ess_bdr, ess_test_tdof_list);

      ParGridFunction field(&fes1), field2(&fes2);

      ParMixedBilinearForm gform(&fes1, &fes2);
      gform.AddDomainIntegrator(new GradientIntegrator);
      gform.Assemble();

      // Project u = f1
      FunctionCoefficient fcoeff1(f1);
      field.ProjectCoefficient(fcoeff1);

      VectorFunctionCoefficient fcoeff2(dim, gradf1);
      ParLinearForm lf(&fes2);
      lf.AddDomainIntegrator(new VectorDomainLFIntegrator(fcoeff2));
      lf.Assemble();

      OperatorHandle G;
      Vector X, B;
      gform.FormRectangularLinearSystem(ess_trial_tdof_list,
                                        ess_test_tdof_list,
                                        field,
                                        lf,
                                        G,
                                        X,
                                        B);

      Vector *field_tdof = field.ParallelProject();
      Vector *field2_tdof = field2.ParallelProject();

      G->Mult(*field_tdof, *field2_tdof);

      subtract(B, *field2_tdof, *field2_tdof);
      REQUIRE(field2_tdof->Norml2() == Approx(0.0));
   }
}

#endif

} // namespace mfem
