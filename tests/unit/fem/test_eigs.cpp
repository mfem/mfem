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

#include "mfem.hpp"
#include "unit_tests.hpp"
#include "common_get_mesh.hpp"

using namespace mfem;
using namespace mfem_test_fem;

namespace eigs
{

#if defined MFEM_USE_LAPACK || defined MFEM_USE_MPI

static double a_ = M_PI;
static double b_ = M_PI / sqrt(2.0);
static double c_ = M_PI / 2.0;

#endif

int eigs[21] =
{
   1,4,9,16,25,36,49,
   3,6,9,11,12,17,18,
   7,10,13,15,16,19,21
};

#ifdef MFEM_USE_LAPACK
#
TEST_CASE("Laplacian Eigenvalues",
          "[H1_FECollection]"
          "[GridFunction]"
          "[BilinearForm]")
{
   int order = 3;

   for (int mt = (int)MeshType::SEGMENT;
        mt <= (int)MeshType::MIXED3D8; mt++)
   {
      Mesh *mesh = GetMesh((MeshType)mt, a_, b_, c_);
      int  dim = mesh->Dimension();
      if (dim < 3 ||
          mt == MeshType::HEXAHEDRON ||
          mt == MeshType::WEDGE2     ||
          mt == MeshType::TETRAHEDRA ||
          mt == MeshType::WEDGE4     ||
          mt == MeshType::MIXED3D8 )
      {
         mesh->UniformRefinement();
      }

      H1_FECollection fec(order, dim);
      FiniteElementSpace fespace(mesh, &fec);
      int size = fespace.GetTrueVSize();
      std::cout << mt << " Eigenvalue system size: " << size << std::endl;

      Array<int> ess_bdr;
      if (mesh->bdr_attributes.Size())
      {
         ess_bdr.SetSize(mesh->bdr_attributes.Max());
         ess_bdr = 1;
      }
      Array<int> ess_bdr_tdofs;
      fespace.GetEssentialTrueDofs(ess_bdr, ess_bdr_tdofs);
      int bsize = ess_bdr_tdofs.Size();

      BilinearForm a(&fespace);
      a.AddDomainIntegrator(new DiffusionIntegrator);
      a.Assemble();
      a.EliminateEssentialBCDiag(ess_bdr, 1.0);
      a.Finalize();

      BilinearForm m(&fespace);
      m.AddDomainIntegrator(new MassIntegrator);
      m.Assemble();
      // shift the eigenvalue corresponding to eliminated dofs to a large value
      m.EliminateEssentialBCDiag(ess_bdr, std::numeric_limits<double>::min());
      m.Finalize();

      DenseMatrix Ad(size);
      DenseMatrix Md(size);
      DenseMatrix vd(size);

      Ad = 0.0;
      Md = 0.0;
      Vector one(size);
      Vector done(size);
      one = 0.0;
      for (int i=0; i<size; i++)
      {
         one[i] = 1.0;
         a.Mult(one, done);
         for (int j=0; j<size; j++)
         {
            Ad(j, i) = done[j];
         }
         m.Mult(one, done);
         for (int j=0; j<size; j++)
         {
            Md(j, i) = done[j];
         }
         one[i] = 0.0;
      }
      for (int i=0; i<bsize; i++)
      {
         int ei = ess_bdr_tdofs[i];
         Ad(ei,ei) = 0.0;
         Md(ei,ei) = 1.0;
      }

      int nev = dim;
      Vector deigs(size);
      Ad.Eigenvalues(Md, deigs, vd);

      Array<int> exact_eigs(&eigs[7 * (dim - 1)], 7);

      double max_err = 0.0;
      for (int i=bsize; i<std::min(size,bsize+nev); i++)
      {
         double lc = deigs[i];
         double le = exact_eigs[i-bsize];
         double err = 100.0 * fabs(le - lc) / le;
         max_err = std::max(max_err, err);
         REQUIRE(err < 5.0);
      }
      std::cout << mt << " Maximum relative error: " << max_err << "%"
                << std::endl;

      delete mesh;
   }
}

#endif // MFEM_USE_LAPACK

#ifdef MFEM_USE_MPI
#
TEST_CASE("Laplacian Eigenvalues in Parallel",
          "[H1_FECollection]"
          "[GridFunction]"
          "[BilinearForm]"
          "[Parallel]")
{
   int num_procs;
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

   int my_rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

   int order = 3;
   int seed = 75;

   for (int mt = (int)MeshType::SEGMENT;
        mt <= (int)MeshType::MIXED3D8; mt++)
   {
      Mesh *mesh = GetMesh((MeshType)mt, a_, b_, c_);
      int  dim = mesh->Dimension();
      if (dim < 3 ||
          mt == MeshType::HEXAHEDRON ||
          mt == MeshType::WEDGE2     ||
          mt == MeshType::TETRAHEDRA ||
          mt == MeshType::WEDGE4     ||
          mt == MeshType::MIXED3D8 )
      {
         mesh->UniformRefinement();
      }
      while (mesh->GetNE() < num_procs)
      {
         mesh->UniformRefinement();
      }
      ParMesh pmesh(MPI_COMM_WORLD, *mesh);
      delete mesh;

      H1_FECollection fec(order, dim);
      ParFiniteElementSpace fespace(&pmesh, &fec);
      HYPRE_Int size = fespace.GlobalTrueVSize();
      if (my_rank == 0)
      {
         std::cout << mt << " Eigenvalue system size: " << size << std::endl;
      }

      Array<int> ess_bdr;
      if (pmesh.bdr_attributes.Size())
      {
         ess_bdr.SetSize(pmesh.bdr_attributes.Max());
         ess_bdr = 1;
      }
      Array<int> ess_bdr_tdofs;
      fespace.GetEssentialTrueDofs(ess_bdr, ess_bdr_tdofs);

      ParBilinearForm a(&fespace);
      a.AddDomainIntegrator(new DiffusionIntegrator);
      a.Assemble();
      a.EliminateEssentialBCDiag(ess_bdr, 1.0);
      a.Finalize();

      ParBilinearForm m(&fespace);
      m.AddDomainIntegrator(new MassIntegrator);
      m.Assemble();
      // shift the eigenvalue corresponding to eliminated dofs to a large value
      m.EliminateEssentialBCDiag(ess_bdr, std::numeric_limits<double>::min());
      m.Finalize();

      HypreParMatrix *A = a.ParallelAssemble();
      HypreParMatrix *M = m.ParallelAssemble();

      HypreBoomerAMG amg(*A);
      amg.SetPrintLevel(0);

      int nev = dim;

      HypreLOBPCG lobpcg(MPI_COMM_WORLD);
      lobpcg.SetNumModes(nev);
      lobpcg.SetRandomSeed(seed);
      lobpcg.SetPreconditioner(amg);
      lobpcg.SetMaxIter(200);
      lobpcg.SetTol(1e-8);
      lobpcg.SetPrecondUsageMode(1);
      lobpcg.SetPrintLevel(0);
      lobpcg.SetMassMatrix(*M);
      lobpcg.SetOperator(*A);

      Array<double> eigenvalues;
      lobpcg.Solve();
      lobpcg.GetEigenvalues(eigenvalues);

      Array<int> exact_eigs(&eigs[7 * (dim - 1)], 7);

      double max_err = 0.0;
      for (int i=0; i<nev; i++)
      {
         double lc = eigenvalues[i];
         double le = exact_eigs[i];
         double err = 100.0 * fabs(le - lc) / le;
         max_err = std::max(max_err, err);
         REQUIRE(err < 5.0);
      }
      if (my_rank == 0)
      {
         std::cout << mt << " Maximum relative error: " << max_err << "%"
                   << std::endl;
      }

      delete A;
      delete M;
   }
}

#endif // MFEM_USE_MPI

} // namespace eigs
