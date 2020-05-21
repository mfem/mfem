// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

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

TEST_CASE("HypreParMatrixBlocks",
          "[Parallel], [BlockMatrix]")
{
   SECTION("HypreParMatrixFromBlocks")
   {
      int rank;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);

      Mesh mesh(10, 10, Element::QUADRILATERAL, 0, 1.0, 1.0);
      int dim = mesh.Dimension();
      int order = 2;

      int nattr = mesh.bdr_attributes.Max();
      Array<int> ess_trial_tdof_list, ess_test_tdof_list;

      Array<int> ess_bdr(nattr);
      ess_bdr = 0;
      ess_bdr[0] = 1;

      ParMesh pmesh(MPI_COMM_WORLD, mesh);

      FiniteElementCollection *hdiv_coll(new RT_FECollection(order, dim));
      FiniteElementCollection *l2_coll(new L2_FECollection(order, dim));

      ParFiniteElementSpace R_space(&pmesh, hdiv_coll);
      ParFiniteElementSpace W_space(&pmesh, l2_coll);

      ParBilinearForm RmVarf(&R_space);
      ParBilinearForm WmVarf(&W_space);
      ParMixedBilinearForm bVarf(&R_space, &W_space);

      HypreParMatrix *MR, *MW, *B;

      RmVarf.AddDomainIntegrator(new VectorFEMassIntegrator());
      RmVarf.Assemble();
      RmVarf.Finalize();
      MR = RmVarf.ParallelAssemble();

      WmVarf.AddDomainIntegrator(new MassIntegrator());
      WmVarf.Assemble();
      WmVarf.Finalize();
      MW = WmVarf.ParallelAssemble();

      bVarf.AddDomainIntegrator(new VectorFEDivergenceIntegrator);
      bVarf.Assemble();
      bVarf.Finalize();
      B = bVarf.ParallelAssemble();
      (*B) *= -1;

      HypreParMatrix *BT = B->Transpose();

      Array<int> blockRow_trueOffsets(3); // number of variables + 1
      blockRow_trueOffsets[0] = 0;
      blockRow_trueOffsets[1] = R_space.TrueVSize();
      blockRow_trueOffsets[2] = W_space.TrueVSize();
      blockRow_trueOffsets.PartialSum();

      Array<int> blockCol_trueOffsets(4); // number of variables + 1
      blockCol_trueOffsets[0] = 0;
      blockCol_trueOffsets[1] = R_space.TrueVSize();
      blockCol_trueOffsets[2] = W_space.TrueVSize();
      blockCol_trueOffsets[3] = W_space.TrueVSize();
      blockCol_trueOffsets.PartialSum();

      BlockOperator blockOper(blockRow_trueOffsets, blockCol_trueOffsets);
      blockOper.SetBlock(0, 0, MR);
      blockOper.SetBlock(0, 1, BT);
      blockOper.SetBlock(1, 0, B);
      blockOper.SetBlock(0, 2, BT, 3.14);
      blockOper.SetBlock(1, 2, MW);

      Array2D<HypreParMatrix*> hBlocks(2,3);
      hBlocks = NULL;
      hBlocks(0, 0) = MR;
      hBlocks(0, 1) = BT;
      hBlocks(1, 0) = B;
      hBlocks(0, 2) = BT;
      hBlocks(1, 2) = MW;

      Array2D<double> blockCoeff(2,3);
      blockCoeff = 1.0;
      blockCoeff(0, 2) = 3.14;
      HypreParMatrix *H = HypreParMatrixFromBlocks(hBlocks, &blockCoeff);

      Vector x(blockCol_trueOffsets[3]);
      Vector yB(blockRow_trueOffsets[2]);
      Vector yH(blockRow_trueOffsets[2]);

      x.Randomize();
      yB = 0.0;
      yH = 0.0;

      blockOper.Mult(x, yB);
      H->Mult(x, yH);

      yH -= yB;
      double error = yH.Norml2();
      std::cout << "  order: " << order
                << ", block matrix error norm on rank " << rank << ": " << error << std::endl;
      REQUIRE(error < 1.e-12);

      delete H;
   }
}

#endif

} // namespace mfem
