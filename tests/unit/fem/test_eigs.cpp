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

using namespace mfem;

namespace eigs
{

static double a_ = M_PI;
static double b_ = M_PI / sqrt(2.0);
static double c_ = M_PI / 2.0;

enum MeshType
{
   SEGMENT = 0,
   QUADRILATERAL = 1,
   TRIANGLE2A = 2,
   TRIANGLE2B = 3,
   TRIANGLE2C = 4,
   TRIANGLE4 = 5,
   MIXED2D = 6,
   HEXAHEDRON = 7,
   HEXAHEDRON2A = 8,
   HEXAHEDRON2B = 9,
   HEXAHEDRON2C = 10,
   HEXAHEDRON2D = 11,
   WEDGE2 = 12,
   TETRAHEDRA = 13,
   WEDGE4 = 14,
   MIXED3D6 = 15,
   MIXED3D8 = 16
};

Mesh * GetMesh(MeshType type);

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
      Mesh *mesh = GetMesh((MeshType)mt);
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
      Mesh *mesh = GetMesh((MeshType)mt);
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

Mesh * GetMesh(MeshType type)
{
   Mesh * mesh = NULL;
   double c[3];
   int    v[8];

   switch (type)
   {
      case SEGMENT:
         mesh = new Mesh(1, 2, 1);
         c[0] = 0.0;
         mesh->AddVertex(c);
         c[0] = a_;
         mesh->AddVertex(c);
         v[0] = 0; v[1] = 1;
         mesh->AddSegment(v);
         {
            Element * el = mesh->NewElement(Geometry::POINT);
            el->SetAttribute(1);
            el->SetVertices(&v[0]);
            mesh->AddBdrElement(el);
         }
         {
            Element * el = mesh->NewElement(Geometry::POINT);
            el->SetAttribute(2);
            el->SetVertices(&v[1]);
            mesh->AddBdrElement(el);
         }
         break;
      case QUADRILATERAL:
         mesh = new Mesh(2, 4, 1);
         c[0] = 0.0; c[1] = 0.0;
         mesh->AddVertex(c);
         c[0] = a_; c[1] = 0.0;
         mesh->AddVertex(c);
         c[0] = a_; c[1] = b_;
         mesh->AddVertex(c);
         c[0] = 0.0; c[1] = b_;
         mesh->AddVertex(c);

         v[0] = 0; v[1] = 1; v[2] = 2; v[3] = 3;
         mesh->AddQuad(v);
         break;
      case TRIANGLE2A:
         mesh = new Mesh(2, 4, 2);
         c[0] = 0.0; c[1] = 0.0;
         mesh->AddVertex(c);
         c[0] = a_; c[1] = 0.0;
         mesh->AddVertex(c);
         c[0] = a_; c[1] = b_;
         mesh->AddVertex(c);
         c[0] = 0.0; c[1] = b_;
         mesh->AddVertex(c);

         v[0] = 0; v[1] = 1; v[2] = 2;
         mesh->AddTri(v);
         v[0] = 2; v[1] = 3; v[2] = 0;
         mesh->AddTri(v);
         break;
      case TRIANGLE2B:
         mesh = new Mesh(2, 4, 2);
         c[0] = 0.0; c[1] = 0.0;
         mesh->AddVertex(c);
         c[0] = a_; c[1] = 0.0;
         mesh->AddVertex(c);
         c[0] = a_; c[1] = b_;
         mesh->AddVertex(c);
         c[0] = 0.0; c[1] = b_;
         mesh->AddVertex(c);

         v[0] = 1; v[1] = 2; v[2] = 0;
         mesh->AddTri(v);
         v[0] = 3; v[1] = 0; v[2] = 2;
         mesh->AddTri(v);
         break;
      case TRIANGLE2C:
         mesh = new Mesh(2, 4, 2);
         c[0] = 0.0; c[1] = 0.0;
         mesh->AddVertex(c);
         c[0] = a_; c[1] = 0.0;
         mesh->AddVertex(c);
         c[0] = a_; c[1] = b_;
         mesh->AddVertex(c);
         c[0] = 0.0; c[1] = b_;
         mesh->AddVertex(c);

         v[0] = 2; v[1] = 0; v[2] = 1;
         mesh->AddTri(v);
         v[0] = 0; v[1] = 2; v[2] = 3;
         mesh->AddTri(v);
         break;
      case TRIANGLE4:
         mesh = new Mesh(2, 5, 4);
         c[0] = 0.0; c[1] = 0.0;
         mesh->AddVertex(c);
         c[0] = a_; c[1] = 0.0;
         mesh->AddVertex(c);
         c[0] = a_; c[1] = b_;
         mesh->AddVertex(c);
         c[0] = 0.0; c[1] = b_;
         mesh->AddVertex(c);
         c[0] = 0.5 * a_; c[1] = 0.5 * b_;
         mesh->AddVertex(c);

         v[0] = 0; v[1] = 1; v[2] = 4;
         mesh->AddTri(v);
         v[0] = 1; v[1] = 2; v[2] = 4;
         mesh->AddTri(v);
         v[0] = 2; v[1] = 3; v[2] = 4;
         mesh->AddTri(v);
         v[0] = 3; v[1] = 0; v[2] = 4;
         mesh->AddTri(v);
         break;
      case MIXED2D:
         mesh = new Mesh(2, 6, 4);
         c[0] = 0.0; c[1] = 0.0;
         mesh->AddVertex(c);
         c[0] = a_; c[1] = 0.0;
         mesh->AddVertex(c);
         c[0] = a_; c[1] = b_;
         mesh->AddVertex(c);
         c[0] = 0.0; c[1] = b_;
         mesh->AddVertex(c);
         c[0] = 0.5 * b_; c[1] = 0.5 * b_;
         mesh->AddVertex(c);
         c[0] = a_ - 0.5 * b_; c[1] = 0.5 * b_;
         mesh->AddVertex(c);

         v[0] = 0; v[1] = 1; v[2] = 5; v[3] = 4;
         mesh->AddQuad(v);
         v[0] = 1; v[1] = 2; v[2] = 5;
         mesh->AddTri(v);
         v[0] = 2; v[1] = 3; v[2] = 4; v[3] = 5;
         mesh->AddQuad(v);
         v[0] = 3; v[1] = 0; v[2] = 4;
         mesh->AddTri(v);
         break;
      case HEXAHEDRON:
         mesh = new Mesh(3, 8, 1);
         c[0] = 0.0; c[1] = 0.0; c[2] = 0.0;
         mesh->AddVertex(c);
         c[0] = a_; c[1] = 0.0; c[2] = 0.0;
         mesh->AddVertex(c);
         c[0] = a_; c[1] = b_; c[2] = 0.0;
         mesh->AddVertex(c);
         c[0] = 0.0; c[1] = b_; c[2] = 0.0;
         mesh->AddVertex(c);
         c[0] = 0.0; c[1] = 0.0; c[2] = c_;
         mesh->AddVertex(c);
         c[0] = a_; c[1] = 0.0; c[2] = c_;
         mesh->AddVertex(c);
         c[0] = a_; c[1] = b_; c[2] = c_;
         mesh->AddVertex(c);
         c[0] = 0.0; c[1] = b_; c[2] = c_;
         mesh->AddVertex(c);

         v[0] = 0; v[1] = 1; v[2] = 2; v[3] = 3;
         v[4] = 4; v[5] = 5; v[6] = 6; v[7] = 7;
         mesh->AddHex(v);
         break;
      case HEXAHEDRON2A:
      case HEXAHEDRON2B:
      case HEXAHEDRON2C:
      case HEXAHEDRON2D:
         mesh = new Mesh(3, 12, 2);
         c[0] = 0.0; c[1] = 0.0; c[2] = 0.0;
         mesh->AddVertex(c);
         c[0] = 0.5 * a_; c[1] = 0.0; c[2] = 0.0;
         mesh->AddVertex(c);
         c[0] = a_; c[1] = 0.0; c[2] = 0.0;
         mesh->AddVertex(c);
         c[0] = a_; c[1] = b_; c[2] = 0.0;
         mesh->AddVertex(c);
         c[0] = 0.5 * a_; c[1] = b_; c[2] = 0.0;
         mesh->AddVertex(c);
         c[0] = 0.0; c[1] = b_; c[2] = 0.0;
         mesh->AddVertex(c);
         c[0] = 0.0; c[1] = 0.0; c[2] = c_;
         mesh->AddVertex(c);
         c[0] = 0.5 * a_; c[1] = 0.0; c[2] = c_;
         mesh->AddVertex(c);
         c[0] = a_; c[1] = 0.0; c[2] = c_;
         mesh->AddVertex(c);
         c[0] = a_; c[1] = b_; c[2] = c_;
         mesh->AddVertex(c);
         c[0] = 0.5 * a_; c[1] = b_; c[2] = c_;
         mesh->AddVertex(c);
         c[0] = 0.0; c[1] = b_; c[2] = c_;
         mesh->AddVertex(c);

         v[0] = 0; v[1] = 5; v[2] = 11; v[3] = 6;
         v[4] = 1; v[5] = 4; v[6] = 10; v[7] = 7;
         mesh->AddHex(v);

         switch (type)
         {
            case HEXAHEDRON2A: // Face Orientation 1
               v[0] = 4; v[1] = 10; v[2] = 7; v[3] = 1;
               v[4] = 3; v[5] = 9; v[6] = 8; v[7] = 2;
               mesh->AddHex(v);
               break;
            case HEXAHEDRON2B: // Face Orientation 3
               v[0] = 10; v[1] = 7; v[2] = 1; v[3] = 4;
               v[4] = 9; v[5] = 8; v[6] = 2; v[7] = 3;
               mesh->AddHex(v);
               break;
            case HEXAHEDRON2C: // Face Orientation 5
               v[0] = 7; v[1] = 1; v[2] = 4; v[3] = 10;
               v[4] = 8; v[5] = 2; v[6] = 3; v[7] = 9;
               mesh->AddHex(v);
               break;
            case HEXAHEDRON2D: // Face Orientation 7
               v[0] = 1; v[1] = 4; v[2] = 10; v[3] = 7;
               v[4] = 2; v[5] = 3; v[6] = 9; v[7] = 8;
               mesh->AddHex(v);
               break;
            default:
               // Cannot happen
               break;
         }
         break;
      case WEDGE2:
         mesh = new Mesh(3, 8, 2);
         c[0] = 0.0; c[1] = 0.0; c[2] = 0.0;
         mesh->AddVertex(c);
         c[0] = a_; c[1] = 0.0; c[2] = 0.0;
         mesh->AddVertex(c);
         c[0] = a_; c[1] = b_; c[2] = 0.0;
         mesh->AddVertex(c);
         c[0] = 0.0; c[1] = b_; c[2] = 0.0;
         mesh->AddVertex(c);
         c[0] = 0.0; c[1] = 0.0; c[2] = c_;
         mesh->AddVertex(c);
         c[0] = a_; c[1] = 0.0; c[2] = c_;
         mesh->AddVertex(c);
         c[0] = a_; c[1] = b_; c[2] = c_;
         mesh->AddVertex(c);
         c[0] = 0.0; c[1] = b_; c[2] = c_;
         mesh->AddVertex(c);

         v[0] = 0; v[1] = 1; v[2] = 2; v[3] = 4; v[4] = 5; v[5] = 6;
         mesh->AddWedge(v);
         v[0] = 0; v[1] = 2; v[2] = 3; v[3] = 4; v[4] = 6; v[5] = 7;
         mesh->AddWedge(v);
         break;
      case TETRAHEDRA:
         mesh = new Mesh(3, 8, 5);
         c[0] = 0.0; c[1] = 0.0; c[2] = 0.0;
         mesh->AddVertex(c);
         c[0] = a_; c[1] = 0.0; c[2] = 0.0;
         mesh->AddVertex(c);
         c[0] = a_; c[1] = b_; c[2] = 0.0;
         mesh->AddVertex(c);
         c[0] = 0.0; c[1] = b_; c[2] = 0.0;
         mesh->AddVertex(c);
         c[0] = 0.0; c[1] = 0.0; c[2] = c_;
         mesh->AddVertex(c);
         c[0] = a_; c[1] = 0.0; c[2] = c_;
         mesh->AddVertex(c);
         c[0] = a_; c[1] = b_; c[2] = c_;
         mesh->AddVertex(c);
         c[0] = 0.0; c[1] = b_; c[2] = c_;
         mesh->AddVertex(c);

         v[0] = 0; v[1] = 2; v[2] = 7; v[3] = 5;
         mesh->AddTet(v);
         v[0] = 6; v[1] = 7; v[2] = 2; v[3] = 5;
         mesh->AddTet(v);
         v[0] = 4; v[1] = 7; v[2] = 5; v[3] = 0;
         mesh->AddTet(v);
         v[0] = 1; v[1] = 0; v[2] = 5; v[3] = 2;
         mesh->AddTet(v);
         v[0] = 3; v[1] = 7; v[2] = 0; v[3] = 2;
         mesh->AddTet(v);
         break;
      case WEDGE4:
         mesh = new Mesh(3, 10, 4);
         c[0] = 0.0; c[1] = 0.0; c[2] = 0.0;
         mesh->AddVertex(c);
         c[0] = a_; c[1] = 0.0; c[2] = 0.0;
         mesh->AddVertex(c);
         c[0] = a_; c[1] = b_; c[2] = 0.0;
         mesh->AddVertex(c);
         c[0] = 0.0; c[1] = b_; c[2] = 0.0;
         mesh->AddVertex(c);
         c[0] = 0.5 * a_; c[1] = 0.5 * b_; c[2] = 0.0;
         mesh->AddVertex(c);
         c[0] = 0.0; c[1] = 0.0; c[2] = c_;
         mesh->AddVertex(c);
         c[0] = a_; c[1] = 0.0; c[2] = c_;
         mesh->AddVertex(c);
         c[0] = a_; c[1] = b_; c[2] = c_;
         mesh->AddVertex(c);
         c[0] = 0.0; c[1] = b_; c[2] = c_;
         mesh->AddVertex(c);
         c[0] = 0.5 * a_; c[1] = 0.5 * b_; c[2] = c_;
         mesh->AddVertex(c);

         v[0] = 0; v[1] = 1; v[2] = 4; v[3] = 5; v[4] = 6; v[5] = 9;
         mesh->AddWedge(v);
         v[0] = 1; v[1] = 2; v[2] = 4; v[3] = 6; v[4] = 7; v[5] = 9;
         mesh->AddWedge(v);
         v[0] = 2; v[1] = 3; v[2] = 4; v[3] = 7; v[4] = 8; v[5] = 9;
         mesh->AddWedge(v);
         v[0] = 3; v[1] = 0; v[2] = 4; v[3] = 8; v[4] = 5; v[5] = 9;
         mesh->AddWedge(v);
         break;
      case MIXED3D6:
         mesh = new Mesh(3, 12, 6);
         c[0] = 0.0; c[1] = 0.0; c[2] = 0.0;
         mesh->AddVertex(c);
         c[0] = a_; c[1] = 0.0; c[2] = 0.0;
         mesh->AddVertex(c);
         c[0] = a_; c[1] = b_; c[2] = 0.0;
         mesh->AddVertex(c);
         c[0] = 0.0; c[1] = b_; c[2] = 0.0;
         mesh->AddVertex(c);
         c[0] = 0.5 * c_; c[1] = 0.5 * c_; c[2] = 0.5 * c_;
         mesh->AddVertex(c);
         c[0] = a_ - 0.5 * c_; c[1] = 0.5 * c_; c[2] = 0.5 * c_;
         mesh->AddVertex(c);
         c[0] = a_ - 0.5 * c_; c[1] = b_ - 0.5 * c_; c[2] = 0.5 * c_;
         mesh->AddVertex(c);
         c[0] = 0.5 * c_; c[1] = b_ - 0.5 * c_; c[2] = 0.5 * c_;
         mesh->AddVertex(c);
         c[0] = 0.0; c[1] = 0.0; c[2] = c_;
         mesh->AddVertex(c);
         c[0] = a_; c[1] = 0.0; c[2] = c_;
         mesh->AddVertex(c);
         c[0] = a_; c[1] = b_; c[2] = c_;
         mesh->AddVertex(c);
         c[0] = 0.0; c[1] = b_; c[2] = c_;
         mesh->AddVertex(c);

         v[0] = 0; v[1] = 1; v[2] = 2; v[3] = 3;
         v[4] = 4; v[5] = 5; v[6] = 6; v[7] = 7;
         mesh->AddHex(v);
         v[0] = 0; v[1] = 4; v[2] = 8; v[3] = 1; v[4] = 5; v[5] = 9;
         mesh->AddWedge(v);
         v[0] = 1; v[1] = 5; v[2] = 9; v[3] = 2; v[4] = 6; v[5] = 10;
         mesh->AddWedge(v);
         v[0] = 2; v[1] = 6; v[2] = 10; v[3] = 3; v[4] = 7; v[5] = 11;
         mesh->AddWedge(v);
         v[0] = 3; v[1] = 7; v[2] = 11; v[3] = 0; v[4] = 4; v[5] = 8;
         mesh->AddWedge(v);
         v[0] = 4; v[1] = 5; v[2] = 6; v[3] = 7;
         v[4] = 8; v[5] = 9; v[6] = 10; v[7] = 11;
         mesh->AddHex(v);
         break;
      case MIXED3D8:
         mesh = new Mesh(3, 10, 8);
         c[0] = 0.0; c[1] = 0.0; c[2] = 0.0;
         mesh->AddVertex(c);
         c[0] = a_; c[1] = 0.0; c[2] = 0.0;
         mesh->AddVertex(c);
         c[0] = a_; c[1] = b_; c[2] = 0.0;
         mesh->AddVertex(c);
         c[0] = 0.0; c[1] = b_; c[2] = 0.0;
         mesh->AddVertex(c);

         c[0] = 0.25 * a_; c[1] = 0.5 * b_; c[2] = 0.5 * c_;
         mesh->AddVertex(c);
         c[0] = 0.75 * a_; c[1] = 0.5 * b_; c[2] = 0.5 * c_;
         mesh->AddVertex(c);

         c[0] = 0.0; c[1] = 0.0; c[2] = c_;
         mesh->AddVertex(c);
         c[0] = a_; c[1] = 0.0; c[2] = c_;
         mesh->AddVertex(c);
         c[0] = a_; c[1] = b_; c[2] = c_;
         mesh->AddVertex(c);
         c[0] = 0.0; c[1] = b_; c[2] = c_;
         mesh->AddVertex(c);

         v[0] = 0; v[1] = 3; v[2] = 4; v[3] = 1; v[4] = 2; v[5] = 5;
         mesh->AddWedge(v);
         v[0] = 3; v[1] = 9; v[2] = 4; v[3] = 2; v[4] = 8; v[5] = 5;
         mesh->AddWedge(v);
         v[0] = 9; v[1] = 6; v[2] = 4; v[3] = 8; v[4] = 7; v[5] = 5;
         mesh->AddWedge(v);
         v[0] = 6; v[1] = 0; v[2] = 4; v[3] = 7; v[4] = 1; v[5] = 5;
         mesh->AddWedge(v);
         v[0] = 0; v[1] = 3; v[2] = 9; v[3] = 4;
         mesh->AddTet(v);
         v[0] = 0; v[1] = 9; v[2] = 6; v[3] = 4;
         mesh->AddTet(v);
         v[0] = 1; v[1] = 7; v[2] = 2; v[3] = 5;
         mesh->AddTet(v);
         v[0] = 8; v[1] = 2; v[2] = 7; v[3] = 5;
         mesh->AddTet(v);
         break;
   }
   mesh->FinalizeTopology();

   return mesh;
}

} // namespace eigs
