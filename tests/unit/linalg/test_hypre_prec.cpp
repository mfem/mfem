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

#include "unit_tests.hpp"
#include "mfem.hpp"
#include <memory>

namespace mfem
{

#ifdef MFEM_USE_MPI

enum PartType {ALL, FIRST, LAST, ALL_BUT_LAST, ALL_BUT_FIRST};

double sin3d(const Vector &x)
{
   return sin(x[0]) * sin(x[1]) * sin(x[2]);
}

void sin2d_vec(const Vector &x, Vector &v)
{
   v.SetSize(2);
   v[0] = cos(x[0]) * sin(x[1]);
   v[1] = sin(x[0]) * cos(x[1]);
}

void sin3d_vec(const Vector &x, Vector &v)
{
   v.SetSize(3);
   v[0] = cos(x[0]) * sin(x[1]) * sin(x[2]);
   v[1] = sin(x[0]) * cos(x[1]) * sin(x[2]);
   v[2] = sin(x[0]) * sin(x[1]) * cos(x[2]);
}

void GeneratePart(PartType part_type, int nelems, int world_size,
                  int *partitioning)
{
   if (world_size == 1)
   {
      for (int i=0; i<nelems; i++)
      {
         partitioning[i] = 0;
      }
      return;
   }
   switch (part_type)
   {
      case ALL:
         for (int i=0; i<nelems; i++)
         {
            partitioning[i] = i % world_size;
         }
         break;
      case FIRST:
         for (int i=0; i<nelems; i++)
         {
            partitioning[i] = 0;
         }
         break;
      case LAST:
         for (int i=0; i<nelems; i++)
         {
            partitioning[i] = world_size - 1;
         }
         break;
      case ALL_BUT_LAST:
         for (int i=0; i<nelems; i++)
         {
            partitioning[i] = i % (world_size-1);
         }
         break;
      case ALL_BUT_FIRST:
         for (int i=0; i<nelems; i++)
         {
            partitioning[i] = i % (world_size-1) + 1;
         }
         break;
   }
}

TEST_CASE("HypreBoomerAMG", "[Parallel][HypreBoomerAMG]")
{
   int world_size, rank;
   MPI_Comm_size(MPI_COMM_WORLD, &world_size);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);

   int n = 3;
   int dim = 3;
   int order = 2;

   Mesh mesh = Mesh::MakeCartesian3D(n, n, n, Element::HEXAHEDRON);

   int nelems = mesh.GetNE();
   auto partitioning = std::make_unique<int[]>(nelems);

   PartType last_type = (world_size == 1) ? ALL : ALL_BUT_FIRST;
   for (int part_type = ALL; part_type <= last_type; part_type++)
   {
      GeneratePart((PartType)part_type, nelems, world_size, partitioning.get());

      ParMesh pmesh(MPI_COMM_WORLD, mesh, partitioning.get());

      H1_FECollection fec(order, dim);
      ParFiniteElementSpace fespace(&pmesh, &fec);

      ParBilinearForm a(&fespace);
      a.AddDomainIntegrator(new DiffusionIntegrator);
      a.AddDomainIntegrator(new MassIntegrator);
      a.Assemble();

      ParGridFunction x(&fespace);
      FunctionCoefficient sin3dCoef(sin3d);
      x.ProjectCoefficient(sin3dCoef);
      double err0 = x.ComputeL2Error(sin3dCoef);

      ParLinearForm b(&fespace);
      a.Mult(x, b);
      x = 0.0;

      OperatorPtr A;
      Vector B, X;
      Array<int> ess_tdof_list;
      a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);

      HypreBoomerAMG amg;
      amg.SetPrintLevel(0);

      HyprePCG pcg(MPI_COMM_WORLD);
      pcg.SetTol(1e-10);
      pcg.SetMaxIter(2000);
      pcg.SetPrintLevel(3);
      pcg.SetPreconditioner(amg);
      pcg.SetOperator(*A);
      pcg.Mult(B, X);

      int its = -1;
      pcg.GetNumIterations(its);

      a.RecoverFEMSolution(X, b, x);

      double err1 = x.ComputeL2Error(sin3dCoef);
      REQUIRE(fabs(err1 - err0) < 1e-6 * err0);
   }
}

TEST_CASE("HypreAMS", "[Parallel][HypreAMS]")
{
   int world_size, rank;
   MPI_Comm_size(MPI_COMM_WORLD, &world_size);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);

   int n = 3;
   int dim = GENERATE(2, 3);
   int order = 2;

   Mesh mesh = (dim == 2) ?
               Mesh::MakeCartesian2D(n, n, Element::QUADRILATERAL):
               Mesh::MakeCartesian3D(n, n, n, Element::HEXAHEDRON);

   int nelems = mesh.GetNE();
   auto partitioning = std::make_unique<int[]>(nelems);

   PartType last_type = (world_size == 1) ? ALL : ALL_BUT_FIRST;
   for (int part_type = ALL; part_type <= last_type; part_type++)
   {
      GeneratePart((PartType)part_type, nelems, world_size, partitioning.get());

      ParMesh pmesh(MPI_COMM_WORLD, mesh, partitioning.get());

      ND_FECollection fec(order, dim);
      ParFiniteElementSpace fespace(&pmesh, &fec);

      ParBilinearForm a(&fespace);
      a.AddDomainIntegrator(new CurlCurlIntegrator);
      a.AddDomainIntegrator(new VectorFEMassIntegrator);
      a.Assemble();

      ParGridFunction x(&fespace);
      VectorFunctionCoefficient sinCoef(dim,
                                        (dim == 2) ? sin2d_vec : sin3d_vec);
      x.ProjectCoefficient(sinCoef);
      double err0 = x.ComputeL2Error(sinCoef);

      ParLinearForm b(&fespace);
      a.Mult(x, b);
      x = 0.0;

      OperatorPtr A;
      Vector B, X;
      Array<int> ess_tdof_list;
      a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);

      HypreAMS ams(*A.As<HypreParMatrix>(), &fespace);
      ams.SetPrintLevel(0);

      HyprePCG pcg(MPI_COMM_WORLD);
      pcg.SetTol(1e-10);
      pcg.SetMaxIter(2000);
      pcg.SetPrintLevel(3);
      pcg.SetPreconditioner(ams);
      pcg.SetOperator(*A);
      pcg.Mult(B, X);

      int its = -1;
      pcg.GetNumIterations(its);

      a.RecoverFEMSolution(X, b, x);

      double err1 = x.ComputeL2Error(sinCoef);
      REQUIRE(fabs(err1 - err0) < 1e-6 * err0);
   }
}

TEST_CASE("HypreADS", "[Parallel][HypreADS]")
{
   int world_size, rank;
   MPI_Comm_size(MPI_COMM_WORLD, &world_size);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);

   int n = 3;
   int dim = 3;
   int order = 2;

   Mesh mesh = Mesh::MakeCartesian3D(n, n, n, Element::HEXAHEDRON);

   int nelems = mesh.GetNE();
   auto partitioning = std::make_unique<int[]>(nelems);

   PartType last_type = (world_size == 1) ? ALL : ALL_BUT_FIRST;
   for (int part_type = ALL; part_type <= last_type; part_type++)
   {
      GeneratePart((PartType)part_type, nelems, world_size, partitioning.get());

      ParMesh pmesh(MPI_COMM_WORLD, mesh, partitioning.get());

      RT_FECollection fec(order, dim);
      ParFiniteElementSpace fespace(&pmesh, &fec);

      ParBilinearForm a(&fespace);
      a.AddDomainIntegrator(new DivDivIntegrator);
      a.AddDomainIntegrator(new VectorFEMassIntegrator);
      a.Assemble();

      ParGridFunction x(&fespace);
      VectorFunctionCoefficient sin3dCoef(3, sin3d_vec);
      x.ProjectCoefficient(sin3dCoef);
      double err0 = x.ComputeL2Error(sin3dCoef);

      ParLinearForm b(&fespace);
      a.Mult(x, b);
      x = 0.0;

      OperatorPtr A;
      Vector B, X;
      Array<int> ess_tdof_list;
      a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);

      HypreADS ads(*A.As<HypreParMatrix>(), &fespace);
      ads.SetPrintLevel(0);

      HyprePCG pcg(MPI_COMM_WORLD);
      pcg.SetTol(1e-10);
      pcg.SetMaxIter(2000);
      pcg.SetPrintLevel(3);
      pcg.SetPreconditioner(ads);
      pcg.SetOperator(*A);
      pcg.Mult(B, X);

      int its = -1;
      pcg.GetNumIterations(its);

      a.RecoverFEMSolution(X, b, x);

      double err1 = x.ComputeL2Error(sin3dCoef);
      REQUIRE(fabs(err1 - err0) < 1e-6 * err0);
   }
}

// The preconditioner is given a new operator after the destruction of the
// previous one, as a NewtonSolver does with the gradient of a
// ParNonlinearForm.
TEST_CASE("Hypre preconditioners SetOperator",
          "[Parallel][HypreParaSails][HypreEuclid]")
{
   Mesh mesh = Mesh::MakeCartesian3D(4, 4, 4, Element::HEXAHEDRON);
   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   H1_FECollection fec(1, pmesh.Dimension());
   ParFiniteElementSpace fespace(&pmesh, &fec);
   ParBilinearForm a(&fespace);
   a.AddDomainIntegrator(new DiffusionIntegrator);
   a.AddDomainIntegrator(new MassIntegrator);
   a.Assemble();
   a.Finalize();
   std::unique_ptr<HypreParMatrix> A0(a.ParallelAssemble());
   HypreParVector B(*A0), X(*A0), R(*A0);
   B = 1.0;

   auto test = [&](HypreSolver &prec)
   {
      HyprePCG pcg(MPI_COMM_WORLD);
      pcg.SetTol(1e-10);
      pcg.SetMaxIter(500);
      pcg.SetPreconditioner(prec);
      // Three operators, since a read of a destroyed operator does not
      // always fail.
      std::unique_ptr<HypreParMatrix> A;
      for (int i = 0; i < 3; i++)
      {
         // The assignment destroys the previous operator, still known to the
         // preconditioner, before SetOperator is called.
         A = std::make_unique<HypreParMatrix>(*A0);
         *A *= 1.0 + i;
         pcg.SetOperator(*A);
         X = 0.0;
         pcg.Mult(B, X);
         A->Mult(X, R);
         R -= B;
         REQUIRE(InnerProduct(R, R) < 1e-16 * InnerProduct(B, B));
      }
   };

   SECTION("DiagScale") { HypreDiagScale prec; test(prec); }
   SECTION("BoomerAMG")
   {
      HypreBoomerAMG prec;
      prec.SetPrintLevel(0);
      test(prec);
   }
   // ILU, ParaSails and Euclid are tested on the host only, as in the
   // HypreILU test. ParaSails and Euclid are not available in mixedint mode.
   if (HypreUsingGPU()) { return; }
#if MFEM_HYPRE_VERSION >= 21900
   SECTION("ILU") { HypreILU prec; test(prec); }
#endif
#ifndef HYPRE_MIXEDINT
   SECTION("ParaSails")
   {
      HypreParaSails prec(MPI_COMM_WORLD);
      prec.SetSymmetry(1);
      prec.SetLogging(0);
      test(prec);
   }
   SECTION("Euclid") { HypreEuclid prec(MPI_COMM_WORLD); test(prec); }
#endif
}

TEST_CASE("HypreAMS preconditions parallel variable-order ND systems",
          "[Parallel][HypreAMS][NCMesh]")
{
   const int ob_type = GENERATE(BasisType::GaussLegendre,
                                BasisType::IntegratedGLL);
   CAPTURE(ob_type);

   Mesh serial = Mesh::MakeCartesian3D(2, 2, 2, Element::HEXAHEDRON);
   serial.EnsureNCMesh();
   Array<int> refine;
   refine.Append(0);
   serial.GeneralRefinement(refine);

   ParMesh pmesh(MPI_COMM_WORLD, serial);
   ND_FECollection fec(1, 3, BasisType::GaussLobatto, ob_type);
   ParFiniteElementSpace pfes(&pmesh, &fec);
   Vector c(3);
   for (int e = 0; e < pmesh.GetNE(); ++e)
   {
      pmesh.GetElementCenter(e, c);
      pfes.SetElementOrder(e, (c(0) > 0.5) ? 2 : 1);
   }
   pfes.Update(false);

   ParBilinearForm a(&pfes);
   a.AddDomainIntegrator(new CurlCurlIntegrator);
   a.AddDomainIntegrator(new VectorFEMassIntegrator);
   a.Assemble();

   VectorFunctionCoefficient exsol(3, sin3d_vec);
   ParGridFunction x(&pfes);
   x.ProjectCoefficient(exsol);
   Vector x_true(pfes.GetTrueVSize());
   x.ParallelProject(x_true);
   x.Distribute(x_true);
   const real_t err0 = x.ComputeL2Error(exsol);

   ParLinearForm b(&pfes);
   a.Mult(x, b);
   x = 0.0;

   Array<int> ess_tdof_list;
   OperatorPtr A;
   Vector B, X;
   a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);

   HypreAMS ams(*A.As<HypreParMatrix>(), &pfes);
   ams.SetPrintLevel(0);

   // Verify internal discrete gradient G and interpolation Pi operators.
   const HypreParMatrix *G_ams = ams.GetDiscreteGradient();
   const HypreParMatrix *Pix = ams.GetPix();
   const HypreParMatrix *Piy = ams.GetPiy();
   const HypreParMatrix *Piz = ams.GetPiz();
   REQUIRE(G_ams != nullptr);
   REQUIRE(Pix != nullptr);
   REQUIRE(Piy != nullptr);
   REQUIRE(Piz != nullptr);

   H1_FECollection h1_fec(1, 3);
   ParFiniteElementSpace h1_fes(&pmesh, &h1_fec);
   for (int e = 0; e < pmesh.GetNE(); ++e)
   {
      h1_fes.SetElementOrder(e, pfes.GetElementOrder(e));
   }
   h1_fes.Update(false);

   FunctionCoefficient linear_phi([](const Vector &p)
   {
      return 1.2 * p(0) - 0.7 * p(1) + 0.5 * p(2) + 0.3;
   });
   Vector const_grad({1.2, -0.7, 0.5});
   VectorConstantCoefficient grad_phi_coeff(const_grad);

   ParGridFunction phi_gf(&h1_fes);
   phi_gf.ProjectCoefficient(linear_phi);
   Vector phi_t(h1_fes.GetTrueVSize()), gphi_t(pfes.GetTrueVSize());
   phi_gf.GetTrueDofs(phi_t);
   G_ams->Mult(phi_t, gphi_t);
   ParGridFunction gphi_gf(&pfes);
   gphi_gf.SetFromTrueDofs(gphi_t);
   REQUIRE(gphi_gf.ComputeL2Error(grad_phi_coeff) < 1e-11);

   Vector one_t(h1_fes.GetTrueVSize()), pi_t(pfes.GetTrueVSize());
   one_t = 1.0;
   const HypreParMatrix *pi_ops[3] = {Pix, Piy, Piz};
   for (int d = 0; d < 3; ++d)
   {
      pi_ops[d]->Mult(one_t, pi_t);
      ParGridFunction pi_gf(&pfes);
      pi_gf.SetFromTrueDofs(pi_t);
      Vector evec(3);
      evec = 0.0;
      evec(d) = 1.0;
      VectorConstantCoefficient ec(evec);
      REQUIRE(pi_gf.ComputeL2Error(ec) < 1e-11);
   }

   CGSolver cg(MPI_COMM_WORLD);
   cg.SetRelTol(1e-10);
   cg.SetMaxIter(200);
   cg.SetPrintLevel(0);
   cg.SetPreconditioner(ams);
   cg.SetOperator(*A);
   cg.Mult(B, X);
   REQUIRE(cg.GetConverged());
   REQUIRE(cg.GetNumIterations() < 40);

   a.RecoverFEMSolution(X, b, x);
   REQUIRE(std::abs(x.ComputeL2Error(exsol) - err0) < 1e-6 * err0);
}

TEST_CASE("HypreADS preconditions parallel variable-order RT systems",
          "[Parallel][HypreADS][NCMesh]")
{
   Mesh serial = Mesh::MakeCartesian3D(2, 2, 2, Element::HEXAHEDRON);
   serial.EnsureNCMesh();
   Array<int> refine;
   refine.Append(0);
   serial.GeneralRefinement(refine);

   ParMesh pmesh(MPI_COMM_WORLD, serial);
   RT_FECollection fec(0, 3);
   ParFiniteElementSpace pfes(&pmesh, &fec);
   Vector c(3);
   for (int e = 0; e < pmesh.GetNE(); ++e)
   {
      pmesh.GetElementCenter(e, c);
      pfes.SetElementOrder(e, (c(0) > 0.5) ? 2 : 1);
   }
   pfes.Update(false);

   ParBilinearForm a(&pfes);
   a.AddDomainIntegrator(new DivDivIntegrator);
   a.AddDomainIntegrator(new VectorFEMassIntegrator);
   a.Assemble();

   VectorFunctionCoefficient exsol(3, sin3d_vec);
   ParGridFunction x(&pfes);
   x.ProjectCoefficient(exsol);
   Vector x_true(pfes.GetTrueVSize());
   x.ParallelProject(x_true);
   x.Distribute(x_true);
   const real_t err0 = x.ComputeL2Error(exsol);

   ParLinearForm b(&pfes);
   a.Mult(x, b);
   x = 0.0;

   Array<int> ess_tdof_list;
   OperatorPtr A;
   Vector B, X;
   a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);

   HypreADS ads(*A.As<HypreParMatrix>(), &pfes);
   ads.SetPrintLevel(0);

   CGSolver cg(MPI_COMM_WORLD);
   cg.SetRelTol(1e-10);
   cg.SetMaxIter(200);
   cg.SetPrintLevel(0);
   cg.SetPreconditioner(ads);
   cg.SetOperator(*A);
   cg.Mult(B, X);
   REQUIRE(cg.GetConverged());
   REQUIRE(cg.GetNumIterations() < 40);

   a.RecoverFEMSolution(X, b, x);
   REQUIRE(std::abs(x.ComputeL2Error(exsol) - err0) < 1e-6 * err0);
}

#endif // MFEM_USE_MPI

} // namespace mfem
