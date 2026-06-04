// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory. All Rights reserved.
// See files LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

// Tied Poisson Problem
// ====================
//
// ******************************************************************************
// WARNING: This implementation has KNOWN BUGS for parallel execution with
//          multiple MPI ranks:
//
// 1. Incorrect parallel matrix partitioning (lines 227-232):
//    - Creates imbalanced penalty matrix where rank 0 owns all rows
//    - Violates expected row distribution, causing incorrect results in ParAdd()
//
// 2. Missing global DOF correspondence (lines 143-155):
//    - Uses only local tied DOF indices without MPI communication
//    - Causes incorrect pairing of tied DOFs across ranks
//
// 3. Race condition in tied DOF indexing (lines 213-220):
//    - Uses local indices tied_dofs1[k] and tied_dofs2[k]
//    - These refer to different physical locations on different ranks
//
// 4. Missing diagonal/off-diagonal matrix split:
//    - Doesn't implement MFEM's required CSR format for parallel matrices
//
// 5. No boundary condition elimination (lines 170-171):
//    - Uses FormSystemMatrix() with empty essential DOF array
//    - Tries to eliminate BCs later after block matrix construction
//
// ==> USE tied-poisson.cpp FOR CORRECT PARALLEL IMPLEMENTATION <==
// This file is kept for reference purposes only.
// ******************************************************************************
//
// This miniapp demonstrates solving a Poisson equation on two copies of a mesh
// with one face tied together via penalty constraints. The tied interface is
// enforced by adding alpha * P * P^T to the stiffness matrix, where P is a
// constraint matrix with +1/-1 entries for each tied node pair.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   // Initialize MPI
   Mpi::Init(argc, argv);
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();

   // Parse command-line options
   const char *mesh_file = "../../data/beam-tri.mesh";
   int order = 1;
   int ref_levels = 0;
   double alpha = 1e3;
   int tied_bdr_attr = 1;
   bool visualization = true;
   double separation = 0.0;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly.");
   args.AddOption(&alpha, "-a", "--alpha",
                  "Penalty parameter for tied interface.");
   args.AddOption(&tied_bdr_attr, "-t", "--tied-attr",
                  "Boundary attribute to tie between mesh copies.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&separation, "-sep", "--separation",
                  "Separation distance for visualization (0 = auto).");
   args.Parse();
   if (!args.Good())
   {
      if (myid == 0)
      {
         args.PrintUsage(cout);
      }
      return 1;
   }
   if (myid == 0)
   {
      args.PrintOptions(cout);
   }

   // Read the mesh from the given mesh file
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   // Refine the mesh uniformly if requested
   for (int l = 0; l < ref_levels; l++)
   {
      mesh->UniformRefinement();
   }

   // Compute separation distance if not specified
   if (separation == 0.0)
   {
      Vector bb_min, bb_max;
      mesh->GetBoundingBox(bb_min, bb_max);
      separation = 2.0 * (bb_max(0) - bb_min(0));
   }

   // Create two copies of the mesh and partition them
   Mesh *mesh1 = new Mesh(*mesh);
   Mesh *mesh2 = new Mesh(*mesh);

   // Translate mesh2 for visualization (we'll undo this for computation)
   Vector translate(dim);
   translate = 0.0;
   translate(0) = separation;
   GridFunction *nodes2 = mesh2->GetNodes();
   if (nodes2)
   {
      for (int i = 0; i < nodes2->Size() / dim; i++)
      {
         for (int d = 0; d < dim; d++)
         {
            (*nodes2)(i * dim + d) += translate(d);
         }
      }
   }
   else
   {
      for (int i = 0; i < mesh2->GetNV(); i++)
      {
         double *coord = mesh2->GetVertex(i);
         coord[0] += separation;
      }
   }

   // Create parallel meshes
   ParMesh *pmesh1 = new ParMesh(MPI_COMM_WORLD, *mesh1);
   ParMesh *pmesh2 = new ParMesh(MPI_COMM_WORLD, *mesh2);
   delete mesh1;
   delete mesh2;
   delete mesh;

   // Create finite element spaces
   H1_FECollection fec(order, dim);
   ParFiniteElementSpace *fespace1 = new ParFiniteElementSpace(pmesh1, &fec);
   ParFiniteElementSpace *fespace2 = new ParFiniteElementSpace(pmesh2, &fec);

   HYPRE_BigInt glob_size1 = fespace1->GlobalTrueVSize();
   HYPRE_BigInt glob_size2 = fespace2->GlobalTrueVSize();
   HYPRE_BigInt glob_size_total = glob_size1 + glob_size2;

   if (myid == 0)
   {
      cout << "Number of DOFs in mesh 1: " << glob_size1 << endl;
      cout << "Number of DOFs in mesh 2: " << glob_size2 << endl;
      cout << "Total number of DOFs: " << glob_size_total << endl;
   }

   // Identify tied boundary DOFs
   Array<int> tied_dofs1, tied_dofs2;
   Array<int> bdr_marker1(pmesh1->bdr_attributes.Max());
   Array<int> bdr_marker2(pmesh2->bdr_attributes.Max());
   bdr_marker1 = 0; bdr_marker1[tied_bdr_attr-1] = 1;
   bdr_marker2 = 0; bdr_marker2[tied_bdr_attr-1] = 1;

   fespace1->GetEssentialTrueDofs(bdr_marker1, tied_dofs1);
   fespace2->GetEssentialTrueDofs(bdr_marker2, tied_dofs2);

   if (myid == 0)
   {
      cout << "Number of tied DOFs per mesh (rank 0): " << tied_dofs1.Size() << endl;
   }

   // Build the stiffness matrices
   ParBilinearForm *a1 = new ParBilinearForm(fespace1);
   ParBilinearForm *a2 = new ParBilinearForm(fespace2);
   ConstantCoefficient one(1.0);
   a1->AddDomainIntegrator(new DiffusionIntegrator(one));
   a2->AddDomainIntegrator(new DiffusionIntegrator(one));
   a1->Assemble();
   a2->Assemble();
   a1->Finalize();
   a2->Finalize();

   HypreParMatrix A1, A2;
   Array<int> empty;
   a1->FormSystemMatrix(empty, A1);
   a2->FormSystemMatrix(empty, A2);

   // Create combined matrix in block form
   Array<int> block_offsets(3);
   block_offsets[0] = 0;
   block_offsets[1] = fespace1->GetTrueVSize();
   block_offsets[2] = fespace1->GetTrueVSize() + fespace2->GetTrueVSize();

   BlockOperator A(block_offsets);
   A.SetBlock(0, 0, &A1);
   A.SetBlock(1, 1, &A2);

   // Compute sizes for later use
   int size1 = fespace1->GetTrueVSize();
   int size2 = fespace2->GetTrueVSize();
   int total_size = size1 + size2;

   if (myid == 0)
   {
      cout << "Building block diagonal matrix..." << endl;
   }

   // Build the full block matrix: diag(A1, A2)
   Array2D<HypreParMatrix*> diag_blocks(2, 2);
   diag_blocks = NULL;
   diag_blocks(0, 0) = &A1;
   diag_blocks(1, 1) = &A2;

   HypreParMatrix *A_block = HypreParMatrixFromBlocks(diag_blocks);

   if (myid == 0)
   {
      cout << "Building penalty matrix..." << endl;
   }

   // Build a sparse penalty matrix manually
   // For serial or simple parallel, build on rank 0
   SparseMatrix penalty_local(total_size, total_size);

   // Add penalty terms: alpha * (u_i - u_j)^2 expands to contributions:
   //   alpha to (i,i), alpha to (j,j), -alpha to (i,j), -alpha to (j,i)
   for (int k = 0; k < tied_dofs1.Size(); k++)
   {
      int i = tied_dofs1[k];
      int j = fespace1->GetTrueVSize() + tied_dofs2[k];

      penalty_local.Add(i, i, alpha);
      penalty_local.Add(i, j, -alpha);
      penalty_local.Add(j, i, -alpha);
      penalty_local.Add(j, j, alpha);
   }

   penalty_local.Finalize();

   // Convert to HypreParMatrix
   // Simple partitioning: rank 0 owns all rows, others own nothing
   HYPRE_BigInt row_starts[2] = {0, (myid == 0) ? glob_size_total : 0};
   HYPRE_BigInt col_starts[2] = {0, glob_size_total};

   HypreParMatrix *penalty = new HypreParMatrix(MPI_COMM_WORLD,
                                                 glob_size_total,
                                                 row_starts, &penalty_local);

   if (myid == 0)
   {
      cout << "Adding penalty to system matrix..." << endl;
   }

   HypreParMatrix *A_final = ParAdd(A_block, penalty);

   delete penalty;
   delete A_block;

   if (myid == 0)
   {
      cout << "Matrix assembly complete." << endl;
   }

   // Set up the linear form (RHS)
   ParLinearForm *b1 = new ParLinearForm(fespace1);
   ParLinearForm *b2 = new ParLinearForm(fespace2);
   ConstantCoefficient rhs_coef(1.0);
   b1->AddDomainIntegrator(new DomainLFIntegrator(rhs_coef));
   b2->AddDomainIntegrator(new DomainLFIntegrator(rhs_coef));
   b1->Assemble();
   b2->Assemble();

   // Create combined RHS vector
   Vector b_vec1, b_vec2;
   b1->ParallelAssemble(b_vec1);
   b2->ParallelAssemble(b_vec2);

   Vector b_full(total_size);
   for (int i = 0; i < b_vec1.Size(); i++)
   {
      b_full(i) = b_vec1(i);
   }
   for (int i = 0; i < b_vec2.Size(); i++)
   {
      b_full(size1 + i) = b_vec2(i);
   }

   // Apply essential boundary conditions (u=1 on non-tied boundaries)
   Array<int> ess_bdr1(pmesh1->bdr_attributes.Max());
   Array<int> ess_bdr2(pmesh2->bdr_attributes.Max());
   ess_bdr1 = 1; ess_bdr1[tied_bdr_attr-1] = 0;
   ess_bdr2 = 1; ess_bdr2[tied_bdr_attr-1] = 0;

   Array<int> ess_tdof1, ess_tdof2;
   fespace1->GetEssentialTrueDofs(ess_bdr1, ess_tdof1);
   fespace2->GetEssentialTrueDofs(ess_bdr2, ess_tdof2);

   // Create combined ess_tdof array
   Array<int> ess_tdof_full;
   for (int i = 0; i < ess_tdof1.Size(); i++)
   {
      ess_tdof_full.Append(ess_tdof1[i]);
   }
   for (int i = 0; i < ess_tdof2.Size(); i++)
   {
      ess_tdof_full.Append(size1 + ess_tdof2[i]);
   }

   // Create solution vector
   Vector x_full(total_size);
   x_full = 0.0;

   // Set boundary values
   for (int i = 0; i < ess_tdof_full.Size(); i++)
   {
      x_full(ess_tdof_full[i]) = 1.0;
   }

   if (myid == 0)
   {
      cout << "Eliminating essential DOFs..." << endl;
   }

   // Eliminate essential DOFs
   HypreParVector X(MPI_COMM_WORLD, glob_size_total, x_full.GetData(),
                    A_final->GetRowStarts());
   HypreParVector B(MPI_COMM_WORLD, glob_size_total, b_full.GetData(),
                    A_final->GetRowStarts());
   A_final->EliminateRowsCols(ess_tdof_full, X, B);

   if (myid == 0)
   {
      cout << "Setting up preconditioner..." << endl;
   }

   // Solve the system using PCG with HypreBoomerAMG preconditioner
   HypreBoomerAMG M(*A_final);
   M.SetSystemsOptions(dim);

   if (myid == 0)
   {
      cout << "Setting up PCG solver..." << endl;
   }

   HyprePCG pcg(*A_final);
   pcg.SetTol(1e-12);
   pcg.SetMaxIter(1000);
   pcg.SetPrintLevel(0);
   pcg.SetPreconditioner(M);

   if (myid == 0)
   {
      cout << "Solving system..." << endl;
   }

   pcg.Mult(b_full, x_full);

   if (myid == 0)
   {
      int num_iter;
      pcg.GetNumIterations(num_iter);
      cout << "Solve complete. Iterations: " << num_iter << endl;
   }

   // Extract solutions for each mesh
   Vector x1_vec(size1);
   Vector x2_vec(size2);
   for (int i = 0; i < x1_vec.Size(); i++)
   {
      x1_vec(i) = x_full(i);
   }
   for (int i = 0; i < x2_vec.Size(); i++)
   {
      x2_vec(i) = x_full(size1 + i);
   }

   ParGridFunction x1(fespace1);
   ParGridFunction x2(fespace2);
   x1.Distribute(x1_vec);
   x2.Distribute(x2_vec);

   // Save the solution
   if (myid == 0)
   {
      ofstream mesh1_ofs("tied-poisson-mesh1.mesh");
      ofstream mesh2_ofs("tied-poisson-mesh2.mesh");
      ofstream sol1_ofs("tied-poisson-sol1.gf");
      ofstream sol2_ofs("tied-poisson-sol2.gf");

      mesh1_ofs.precision(8);
      mesh2_ofs.precision(8);
      sol1_ofs.precision(8);
      sol2_ofs.precision(8);

      pmesh1->Print(mesh1_ofs);
      pmesh2->Print(mesh2_ofs);
      x1.Save(sol1_ofs);
      x2.Save(sol2_ofs);
   }

   // Send solutions to GLVis
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;

      socketstream sol1_sock(vishost, visport);
      sol1_sock << "parallel " << num_procs << " " << myid << "\n";
      sol1_sock.precision(8);
      sol1_sock << "solution\n" << *pmesh1 << x1
                << "window_title 'Solution on Mesh 1'\n"
                << "window_geometry 0 0 400 400\n" << flush;

      socketstream sol2_sock(vishost, visport);
      sol2_sock << "parallel " << num_procs << " " << myid << "\n";
      sol2_sock.precision(8);
      sol2_sock << "solution\n" << *pmesh2 << x2
                << "window_title 'Solution on Mesh 2'\n"
                << "window_geometry 420 0 400 400\n" << flush;
   }

   // Clean up
   delete A_final;
   delete b2;
   delete b1;
   delete a2;
   delete a1;
   delete fespace2;
   delete fespace1;
   delete pmesh2;
   delete pmesh1;

   return 0;
}
