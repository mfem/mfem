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

#endif // MFEM_USE_MPI

} // namespace mfem
