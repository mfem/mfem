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
#include <functional>
#include <map>
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
   using PrecFactory = std::function<std::unique_ptr<HypreSolver>()>;
   std::map<std::string, PrecFactory> precs =
   {
      {"DiagScale", [] { return std::make_unique<HypreDiagScale>(); }},
      {
         "BoomerAMG", []
         {
            auto amg = std::make_unique<HypreBoomerAMG>();
            amg->SetPrintLevel(0);
            return amg;
         }
      },
   };
   // ILU, ParaSails and Euclid are tested on the host only, as in the
   // HypreILU test. ParaSails and Euclid are not available in mixedint mode.
   if (!HypreUsingGPU())
   {
#if MFEM_HYPRE_VERSION >= 21900
      precs["ILU"] = [] { return std::make_unique<HypreILU>(); };
#endif
#ifndef HYPRE_MIXEDINT
      precs["ParaSails"] = []
      {
         auto parasails = std::make_unique<HypreParaSails>(MPI_COMM_WORLD);
         parasails->SetSymmetry(1);
         parasails->SetLogging(0);
         return parasails;
      };
      precs["Euclid"] = []
      { return std::make_unique<HypreEuclid>(MPI_COMM_WORLD); };
#endif
   }

   Mesh mesh = Mesh::MakeCartesian3D(4, 4, 4, Element::HEXAHEDRON);
   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   H1_FECollection fec(1, pmesh.Dimension());
   ParFiniteElementSpace fespace(&pmesh, &fec);
   Array<int> ess_tdof_list;
   fespace.GetBoundaryTrueDofs(ess_tdof_list);

   ConstantCoefficient one(1.0);
   ParLinearForm b(&fespace);
   b.AddDomainIntegrator(new DomainLFIntegrator(one));
   b.Assemble();
   std::unique_ptr<HypreParVector> B(b.ParallelAssemble());
   B->SetSubVector(ess_tdof_list, 0.0);
   HypreParVector X(*B), R(*B);

   auto assemble = [&](real_t k)
   {
      ConstantCoefficient coef(k);
      ParBilinearForm a(&fespace);
      a.AddDomainIntegrator(new DiffusionIntegrator(coef));
      a.Assemble();
      a.Finalize();
      std::unique_ptr<HypreParMatrix> A(a.ParallelAssemble());
      delete A->EliminateRowsCols(ess_tdof_list);
      return A;
   };

   for (const auto &[name, make_prec] : precs)
   {
      DYNAMIC_SECTION(name)
      {
         std::unique_ptr<HypreSolver> prec = make_prec();
         HyprePCG pcg(MPI_COMM_WORLD);
         pcg.SetTol(1e-10);
         pcg.SetMaxIter(500);
         pcg.SetPreconditioner(*prec);

         // Three operators, since a read of a destroyed operator does not
         // always fail.
         std::unique_ptr<HypreParMatrix> A;
         for (int i = 0; i < 3; i++)
         {
            // The move assignment destroys the previous operator, still known
            // to the preconditioner, before SetOperator is called.
            A = assemble(1.0 + i);
            pcg.SetOperator(*A);
            X = 0.0;
            pcg.Mult(*B, X);
            A->Mult(X, R);
            R -= *B;
            REQUIRE(InnerProduct(R, R) < 1e-16 * InnerProduct(*B, *B));
         }
      }
   }
}

#endif // MFEM_USE_MPI

} // namespace mfem
