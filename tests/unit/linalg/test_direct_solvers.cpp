// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
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

using namespace mfem;

#ifdef MFEM_USE_SUITESPARSE
#define DIRECT_SOLVE_SERIAL
#endif
#ifdef MFEM_USE_MUMPS
#define DIRECT_SOLVE_PARALLEL
#endif
#ifdef MFEM_USE_SUPERLU
#define DIRECT_SOLVE_PARALLEL
#endif

#if defined(DIRECT_SOLVE_SERIAL) || defined(DIRECT_SOLVE_PARALLEL)

int dim;
double uexact(const Vector& x)
{
   double u;
   switch (dim)
   {
      case 1:
         u  = 3.0 + 2.0 * x(0) - 0.5 * x(0) * x(0);
         break;
      case 2:
         u = 1.0 + 0.2 * x(0) - 0.9 * x(0) * x(1) + x(1) * x(1) * x(0);
         break;
      default:
         u = x(2) * x(2) * x(2) - 5.0 * x(0) * x(0) * x(1) * x(2);
         break;
   }
   return u;
}

void gradexact(const Vector& x, Vector & grad)
{
   grad.SetSize(dim);
   switch (dim)
   {
      case 1:
         grad[0] = 2.0 - x(0);
         break;
      case 2:
         grad[0] = 0.2 - 0.9 * x(1) + x(1) * x (1);
         grad[1] = - 0.9 * x(0) + 2.0 * x(0) * x(1);
         break;
      default:
         grad[0] = -10.0 * x(0) * x(1) * x(2);
         grad[1] = - 5.0 * x(0) * x(0) * x(2);
         grad[2] = 3.0 * x(2) * x(2) - 5.0 * x(0) * x(0) * x(1);
         break;
   }
}

double d2uexact(const Vector& x) // retuns \Delta u
{
   double d2u;
   switch (dim)
   {
      case 1:
         d2u  = -1.0;
         break;
      case 2:
         d2u = 2.0 * x(0);
         break;
      default:
         d2u = -10.0 * x(1) * x(2) + 6.0 * x(2);
         break;
   }
   return d2u;
}

double fexact(const Vector& x) // retuns -\Delta u
{
   double d2u = d2uexact(x);
   return -d2u;
}

#endif

#ifdef DIRECT_SOLVE_SERIAL

TEST_CASE("direct-serial","[CUDA]")
{
   const int ne = 2;
   for (dim = 1; dim < 4; ++dim)
   {
      Mesh mesh;
      if (dim == 1)
      {
         mesh = Mesh::MakeCartesian1D(ne,  1.0);
      }
      else if (dim == 2)
      {
         mesh = Mesh::MakeCartesian2D(
                   ne, ne, Element::QUADRILATERAL, 1, 1.0, 1.0);
      }
      else
      {
         mesh = Mesh::MakeCartesian3D(
                   ne, ne, ne, Element::HEXAHEDRON, 1.0, 1.0, 1.0);
      }
      int order = 3;
      FiniteElementCollection* fec = new H1_FECollection(order, dim);
      FiniteElementSpace fespace(&mesh, fec);
      Array<int> ess_tdof_list;
      Array<int> ess_bdr(mesh.bdr_attributes.Max());
      ess_bdr = 1;
      fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

      FunctionCoefficient f(fexact);
      LinearForm b(&fespace);
      b.AddDomainIntegrator(new DomainLFIntegrator(f));
      b.Assemble();

      BilinearForm a(&fespace);
      ConstantCoefficient one(1.0);
      a.AddDomainIntegrator(new DiffusionIntegrator(one));
      a.Assemble();

      GridFunction x(&fespace);
      FunctionCoefficient uex(uexact);
      x = 0.0;
      x.ProjectBdrCoefficient(uex,ess_bdr);

      OperatorPtr A;
      Vector B, X;
      a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);

      UMFPackSolver umf_solver;
      umf_solver.Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
      umf_solver.SetOperator(*A);
      umf_solver.Mult(B, X);

      Vector Y(X.Size());
      A->Mult(X,Y);
      Y-=B;
      REQUIRE(Y.Norml2() < 1.e-12);

      a.RecoverFEMSolution(X, b, x);
      VectorFunctionCoefficient grad(dim,gradexact);
      double err = x.ComputeH1Error(&uex,&grad);
      REQUIRE(err < 1.e-12);
      delete fec;
   }
}

#endif

#ifdef DIRECT_SOLVE_PARALLEL

TEST_CASE("direct-parallel", "[Parallel], [CUDA]")
{
   int rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   const int ne = 2;
   for (dim = 1; dim < 4; ++dim)
   {
      Mesh mesh;
      if (dim == 1)
      {
         mesh = Mesh::MakeCartesian1D(ne,  1.0);
      }
      else if (dim == 2)
      {
         mesh = Mesh::MakeCartesian2D(
                   ne, ne, Element::QUADRILATERAL, 1, 1.0, 1.0);
      }
      else
      {
         mesh = Mesh::MakeCartesian3D(
                   ne, ne, ne, Element::HEXAHEDRON, 1.0, 1.0, 1.0);
      }

      ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, mesh);
      mesh.Clear();
      int order = 3;
      FiniteElementCollection* fec = new H1_FECollection(order, dim);
      ParFiniteElementSpace fespace(pmesh, fec);
      Array<int> ess_tdof_list;
      Array<int> ess_bdr;
      if (pmesh->bdr_attributes.Size())
      {
         ess_bdr.SetSize(pmesh->bdr_attributes.Max());
         ess_bdr = 1;
         fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
      }
      FunctionCoefficient f(fexact);
      ParLinearForm b(&fespace);
      b.AddDomainIntegrator(new DomainLFIntegrator(f));
      b.Assemble();

      ParBilinearForm a(&fespace);
      ConstantCoefficient one(1.0);
      a.AddDomainIntegrator(new DiffusionIntegrator(one));
      a.Assemble();

      ParGridFunction x(&fespace);
      FunctionCoefficient uex(uexact);
      x = 0.0;
      x.ProjectBdrCoefficient(uex,ess_bdr);

      OperatorPtr A;
      Vector B, X;
      a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);

#ifdef MFEM_USE_MUMPS
      {
         MUMPSSolver mumps;
         mumps.SetPrintLevel(0);
         mumps.SetOperator(*A.As<HypreParMatrix>());
         mumps.Mult(B,X);
         Vector Y(X.Size());
         A->Mult(X,Y);
         Y-=B;
         REQUIRE(Y.Norml2() < 1.e-12);

         a.RecoverFEMSolution(X, b, x);
         VectorFunctionCoefficient grad(dim,gradexact);
         double err = x.ComputeH1Error(&uex,&grad);
         REQUIRE(err < 1.e-12);
      }
#endif
#ifdef MFEM_USE_SUPERLU
      // Transform to monolithic HypreParMatrix
      {
         SuperLURowLocMatrix SA(*A.As<HypreParMatrix>());
         SuperLUSolver superlu(MPI_COMM_WORLD);
         superlu.SetPrintStatistics(false);
         superlu.SetSymmetricPattern(false);
         superlu.SetColumnPermutation(superlu::METIS_AT_PLUS_A);
         superlu.SetOperator(SA);
         superlu.Mult(B, X);
         Vector Y(X.Size());
         A->Mult(X,Y);
         Y-=B;
         REQUIRE(Y.Norml2() < 1.e-12);
         a.RecoverFEMSolution(X, b, x);
         VectorFunctionCoefficient grad(dim,gradexact);
         double err = x.ComputeH1Error(&uex,&grad);
         REQUIRE(err < 1.e-12);
      }
#endif
      delete fec;
      delete pmesh;
   }
}

#endif
