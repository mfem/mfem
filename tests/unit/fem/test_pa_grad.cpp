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

#include "catch.hpp"
#include "mfem.hpp"
#include "unit_tests.hpp"

using namespace mfem;

Mesh MakeCartesianNonaligned(const int dim, const int ne)
{
   Mesh mesh;
   if (dim == 2)
   {
      mesh = Mesh::MakeCartesian2D(ne, ne, Element::QUADRILATERAL, 1, 1.0, 1.0);
   }
   else
   {
      mesh = Mesh::MakeCartesian3D(ne, ne, ne, Element::HEXAHEDRON, 1.0, 1.0, 1.0);
   }

   // Remap vertices so that the mesh is not aligned with axes.
   for (int i=0; i<mesh.GetNV(); ++i)
   {
      real_t *vcrd = mesh.GetVertex(i);
      vcrd[1] += 0.2 * vcrd[0];
      if (dim == 3) { vcrd[2] += 0.3 * vcrd[0]; }
   }

   return mesh;
}

real_t compare_pa_assembly(int dim, int num_elements, int order, bool transpose)
{
   Mesh mesh;
   if (num_elements == 0)
   {
      if (dim == 2)
      {
         mesh = Mesh::LoadFromFile("../../data/star.mesh", order);
      }
      else
      {
         mesh = Mesh::LoadFromFile("../../data/beam-hex.mesh", order);
      }
   }
   else
   {
      mesh = MakeCartesianNonaligned(dim, num_elements);
   }

   FiniteElementCollection *h1_fec = new H1_FECollection(order, dim);
   FiniteElementCollection *nd_fec = new ND_FECollection(order, dim);
   FiniteElementSpace h1_fespace(&mesh, h1_fec);
   FiniteElementSpace nd_fespace(&mesh, nd_fec);

   DiscreteLinearOperator assembled_grad(&h1_fespace, &nd_fespace);
   assembled_grad.AddDomainInterpolator(new GradientInterpolator);
   const int skip_zeros = 1;
   assembled_grad.Assemble(skip_zeros);
   assembled_grad.Finalize(skip_zeros);
   const SparseMatrix& assembled_grad_mat = assembled_grad.SpMat();

   DiscreteLinearOperator pa_grad(&h1_fespace, &nd_fespace);
   pa_grad.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   pa_grad.AddDomainInterpolator(new GradientInterpolator);
   pa_grad.Assemble();
   pa_grad.Finalize();

   int insize, outsize;
   if (transpose)
   {
      insize = nd_fespace.GetVSize();
      outsize = h1_fespace.GetVSize();
   }
   else
   {
      insize = h1_fespace.GetVSize();
      outsize = nd_fespace.GetVSize();
   }
   Vector xv(insize);
   Vector assembled_y(outsize);
   Vector pa_y(outsize);

   xv.Randomize();
   if (transpose)
   {
      assembled_grad_mat.MultTranspose(xv, assembled_y);
      pa_grad.MultTranspose(xv, pa_y);
   }
   else
   {
      assembled_grad_mat.Mult(xv, assembled_y);
      pa_grad.Mult(xv, pa_y);
   }

   pa_y -= assembled_y;
   real_t error = pa_y.Norml2() / assembled_y.Norml2();
   INFO("dim " << dim << " ne " << num_elements << " order " << order
        << (transpose ? " T:" : ":") << " error in PA gradient: " << error);

   delete h1_fec;
   delete nd_fec;

   return error;
}

TEST_CASE("PAGradient", "[GPU]")
{
   auto transpose = GENERATE(true, false);
   auto order = GENERATE(1, 2, 3, 4);
   auto dim = GENERATE(2, 3);
   auto num_elements = GENERATE(0, 1, 2, 3, 4);

   real_t error = compare_pa_assembly(dim, num_elements, order, transpose);
   REQUIRE(error == MFEM_Approx(0.0, 1.0e-14));
}

#ifdef MFEM_USE_MPI

real_t par_compare_pa_assembly(int dim, int num_elements, int order,
                               bool transpose)
{
   int rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   int size;
   MPI_Comm_size(MPI_COMM_WORLD, &size);

   Mesh smesh = MakeCartesianNonaligned(dim, num_elements);
   ParMesh * mesh = new ParMesh(MPI_COMM_WORLD, smesh);
   smesh.Clear();
   FiniteElementCollection *h1_fec = new H1_FECollection(order, dim);
   FiniteElementCollection *nd_fec = new ND_FECollection(order, dim);
   ParFiniteElementSpace h1_fespace(mesh, h1_fec);
   ParFiniteElementSpace nd_fespace(mesh, nd_fec);

   ParDiscreteLinearOperator assembled_grad(&h1_fespace, &nd_fespace);
   assembled_grad.AddDomainInterpolator(new GradientInterpolator);
   const int skip_zeros = 1;
   assembled_grad.Assemble(skip_zeros);
   assembled_grad.Finalize(skip_zeros);
   HypreParMatrix * assembled_grad_mat = assembled_grad.ParallelAssemble();

   ParDiscreteLinearOperator pa_grad(&h1_fespace, &nd_fespace);
   pa_grad.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   pa_grad.AddDomainInterpolator(new GradientInterpolator);
   pa_grad.Assemble();
   OperatorPtr pa_grad_oper;
   pa_grad.FormRectangularSystemMatrix(pa_grad_oper);

   int insize, outsize;
   if (transpose)
   {
      insize = assembled_grad_mat->Height();
      outsize = assembled_grad_mat->Width();
   }
   else
   {
      insize = assembled_grad_mat->Width();
      outsize = assembled_grad_mat->Height();
   }
   Vector xv(insize);
   Vector assembled_y(outsize);
   Vector pa_y(outsize);
   assembled_y = 0.0;
   pa_y = 0.0;

   xv.Randomize();
   if (transpose)
   {
      assembled_grad_mat->MultTranspose(xv, assembled_y);
      pa_grad_oper->MultTranspose(xv, pa_y);
   }
   else
   {
      assembled_grad_mat->Mult(xv, assembled_y);
      pa_grad_oper->Mult(xv, pa_y);
   }

   Vector error_vec(pa_y);
   error_vec -= assembled_y;
   // serial norms and serial error; we are enforcing equality on each processor
   // in the test
   real_t error = error_vec.Norml2() / assembled_y.Norml2();

   for (int p = 0; p < size; ++p)
   {
      if (rank == p)
      {
         INFO("[" << rank << "][par] dim " << dim << " ne " << num_elements
              << " order " << order << (transpose ? " T:" : ":")
              << " error in PA gradient: " << error);
      }
      MPI_Barrier(MPI_COMM_WORLD);
   }

   delete h1_fec;
   delete nd_fec;
   delete assembled_grad_mat;
   delete mesh;

   return error;
}

TEST_CASE("ParallelPAGradient", "[Parallel], [ParallelPAGradient]")
{
   auto transpose = GENERATE(true, false);
   auto order = GENERATE(1, 2, 3, 4);
   auto dim = GENERATE(2, 3);
   auto num_elements = GENERATE(4, 5);

   real_t error = par_compare_pa_assembly(dim, num_elements, order, transpose);
   REQUIRE(error == MFEM_Approx(0.0, 1.0e-14));
}

#endif
