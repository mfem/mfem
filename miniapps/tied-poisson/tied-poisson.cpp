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
// This miniapp demonstrates solving a Poisson equation on two copies of a mesh
// with one face tied together via penalty constraints. The tied interface is
// enforced by adding alpha * (u_i - u_j)^2 penalty terms for each tied DOF pair.

#include "mfem.hpp"

#include "solver_utils.hpp"

#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <random>

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   // Initialize MPI and HYPRE
   Mpi::Init(argc, argv);
   Hypre::Init();

   // Parse command line options
   string mesh_file = "../../data/beam-tri.mesh";
   int order = 1;
   int ref_levels = 0;
   double alpha = 1e3;
   int tied_bdr_attr = 1;
   bool visualization = true;
   bool compute_eigenvalues = false;
   double separation = 0.0;
   int pcg_max_iters = 10000;
   double diffusion_ratio = 100.0;
   bool do_amgf = false;
   bool iterative_filter = false;
   bool one_level_amg = false;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly.");
   args.AddOption(&alpha, "-a", "--alpha",
                  "Penalty parameter for tied interface.");
   args.AddOption(&tied_bdr_attr, "-t", "--tied-attr",
                  "Boundary attribute to tie between mesh copies.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&compute_eigenvalues, "-eig", "--eigenvalues", "-no-eig",
                  "--no-eigenvalues",
                  "Compute and display min/max eigenvalues.");
   args.AddOption(&separation, "-sep", "--separation",
                  "Separation distance for visualization (0 = auto).");
   args.AddOption(&pcg_max_iters, "-p", "--pcg-max-iters",
                  "Max PCG Iterations");
   args.AddOption(&diffusion_ratio, "-d", "--diffusion-ratio",
                  "Ratio of the diffusion coeffecients on each mesh copy.");
   args.AddOption(&do_amgf, "-amgf", "--amgf", "-no-amgf", "--no-amgf",
                  "Enable or disable AMG with Filtering solver.");
   args.AddOption(&iterative_filter, "-iterative-filter", "--iterative-filter", "-no-iterative-filter", "--no-iterative-filter",
                  "Enable or disable use of Schwarz solver on subspace in AMGF.");
   args.AddOption(&one_level_amg, "-one-level-amg", "--one-level-amg", "-no-one-level-amg", "--no-one-level-amg",
                  "Enable or disable using a non multigrid (AMG w/ just one level) preconditioner.");
   args.ParseCheck();

   // Read the serial mesh
   Mesh serial_mesh(mesh_file);
   int dim = serial_mesh.Dimension();

   for (int i = 0; i<serial_mesh.GetNE(); i++)  { serial_mesh.GetElement(i)->SetAttribute(1); }
   serial_mesh.SetAttributes();

   // Refine the mesh uniformly if requested
   for (int l = 0; l < ref_levels; l++)
   {
      serial_mesh.UniformRefinement();
   }

   Mesh serial_mesh_copy(serial_mesh, true);
   for (int i = 0; i<serial_mesh_copy.GetNE(); i++)  { serial_mesh_copy.GetElement(i)->SetAttribute(2); }
   // for (int i = 0; i<serial_mesh.GetNBE(); i++) { serial_mesh_copy.GetBdrElement(i)->SetAttribute(serial_mesh.GetBdrAttribute(i) + 100); }
   serial_mesh_copy.SetAttributes();

   // Compute separation distance if not specified
   if (separation == 0.0)
   {
      Vector bb_min, bb_max;
      serial_mesh.GetBoundingBox(bb_min, bb_max);
      separation = 2.0 * (bb_max(0) - bb_min(0));
   }

   // Translate mesh2 for visualization (shift in x-direction)
   // Linear mesh - translate vertices
   for (int i = 0; i < serial_mesh_copy.GetNV(); i++)
   {
      double *coord = serial_mesh_copy.GetVertex(i);
      coord[0] += separation;
   }

   Mesh * mesh_array[] = {&serial_mesh, &serial_mesh_copy};
   Mesh * combined_mesh = new Mesh(mesh_array, 2);

   ParMesh pmesh(MPI_COMM_WORLD, *combined_mesh);

   // Create finite element spaces
   H1_FECollection fec(order, dim);
   ParFiniteElementSpace fespace(&pmesh, &fec);

   HYPRE_BigInt total_size = fespace.GlobalTrueVSize();

   // Identify tied boundary DOFs
   Array<int> tied_dofs;
   Array<int> bdr_marker(pmesh.bdr_attributes.Max());
   bdr_marker = 0;
   bdr_marker[tied_bdr_attr-1] = 1;

   fespace.GetEssentialTrueDofs(bdr_marker, tied_dofs);

   Array<int> global_tied_dofs(tied_dofs.Size());
   for (int i = 0; i < tied_dofs.Size(); ++i)
   {
      global_tied_dofs[i] = tied_dofs[i] + fespace.GetMyTDofOffset();
   } 

   // Gather all tied DOF global numbers across all ranks
   int num_procs = Mpi::WorldSize();
   int myrank = Mpi::WorldRank();

   // Gather sizes
   int my_num_tied = global_tied_dofs.Size();
   Array<int> all_num_tied(num_procs);
   MPI_Allgather(&my_num_tied, 1, MPI_INT, all_num_tied.GetData(), 1, MPI_INT, MPI_COMM_WORLD);

   int total_num_tied = 0;
   Array<int> tied_offsets(num_procs + 1);
   tied_offsets[0] = 0;
   for (int p = 0; p < num_procs; p++)
   {
      total_num_tied += all_num_tied[p];
      tied_offsets[p+1] = tied_offsets[p] + all_num_tied[p];
   }

   // Gather all global DOF numbers from mesh1
   Array<int> all_tied_gdofs(total_num_tied);
   MPI_Allgatherv(global_tied_dofs.GetData(), my_num_tied, MPI_INT,
                  all_tied_gdofs.GetData(), all_num_tied.GetData(),
                  tied_offsets.GetData(), MPI_INT, MPI_COMM_WORLD);


   // Build map of tied DOF pairs for penalty term construction
   std::map<HYPRE_BigInt, HYPRE_BigInt> tied_pairs;
   for (int i = 0; i < all_tied_gdofs.Size() / 2; ++i)
   {
      HYPRE_BigInt gdof1 = all_tied_gdofs[i];
      HYPRE_BigInt gdof2 = all_tied_gdofs[i + all_tied_gdofs.Size() / 2];
      tied_pairs[gdof1] = gdof2;
      //tied_pairs[gdof2] = gdof1;
   }

   if (Mpi::Root())
   {
      mfem::out << "Total DOFs: " << total_size << endl;
      mfem::out << "Total Tied Pairs: " << tied_pairs.size() / 2 << endl;
   }

   // Identify essential (non-tied) boundary DOFs
   Array<int> ess_bdr(pmesh.bdr_attributes.Max());
   ess_bdr = 1;
   ess_bdr[tied_bdr_attr-1] = 0;

   Array<int> ess_tdof;
   fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof);

   // Set up the linear forms (RHS)
   ConstantCoefficient one(1.0);
   ParLinearForm b(&fespace);
   b.AddDomainIntegrator(new DomainLFIntegrator(one));
   b.Assemble();

   // Set up the bilinear forms (stiffness matrices)
   PWConstCoefficient diffusion_coeff(pmesh.GetElementAttributes().Max());
   diffusion_coeff(1) = 1.0;
   diffusion_coeff(2) = 1.0 * diffusion_ratio;
   ParBilinearForm a(&fespace);
   a.AddDomainIntegrator(new DiffusionIntegrator(diffusion_coeff));
   a.Assemble();

   // Set up solution vectors
   ParGridFunction x(&fespace);
   x = 0.0;

   // Set boundary values to 0
   for (int i = 0; i < ess_tdof.Size(); i++)
   {
      x(ess_tdof[i]) = 0.0;
   }

   // Form the linear systems for each mesh WITH boundary condition elimination
   HypreParMatrix A;
   Vector B, X;
   a.FormLinearSystem(ess_tdof, x, b, A, X, B);

   // Merge diagonal and off-diagonal blocks to get complete local rows
   SparseMatrix merged_A;
   A.MergeDiagAndOffd(merged_A);

   // Get partitioning info
   HYPRE_BigInt row_start = A.GetRowStarts()[0];
   HYPRE_BigInt row_end = A.GetRowStarts()[1];
   int local_num_rows = A.GetNumRows();
   HYPRE_BigInt global_num_cols = A.N();

   // Create new SparseMatrix with penalty terms added
   SparseMatrix modified_A(local_num_rows, global_num_cols);

   // Process each row
   for (int local_row = 0; local_row < local_num_rows; ++local_row)
   {
      HYPRE_BigInt global_row = row_start + local_row;

      // Copy original row entries first
      const int *cols = merged_A.GetRowColumns(local_row);
      const real_t *vals = merged_A.GetRowEntries(local_row);
      const int row_size = merged_A.RowSize(local_row);

      for (int j = 0; j < row_size; ++j)
      {
         modified_A.Add(local_row, cols[j], vals[j]);
      }

      // Check whether this row is tied
      auto it = tied_pairs.find(global_row);
      if (it == tied_pairs.end())
      {
         continue;
      }

      const HYPRE_BigInt paired_gdof = it->second;
      const real_t value = alpha;

      // Build neighbor group:
      //   group = { paired_gdof } U {other tied dofs found in this row stencil}
      std::vector<HYPRE_BigInt> group;
      group.push_back(paired_gdof);

      const int *paired_cols = merged_A.GetRowColumns(local_row);
      const real_t *paired_vals = merged_A.GetRowEntries(local_row);
      const int paired_row_size = merged_A.RowSize(local_row);

      for (int j = 0; j < paired_row_size; ++j)
      {
         const HYPRE_BigInt c = paired_cols[j];

         if (c == global_row || c == paired_gdof)
         {
            continue;
         }

         if (all_tied_gdofs.Find(c) != -1)
         {
            group.push_back(c);
         }
      }

      // Optional dedup, in case the stencil can contain repeats
      std::sort(group.begin(), group.end());
      group.erase(std::unique(group.begin(), group.end()), group.end());

      const int m = (int)group.size();
      if (m == 0)
      {
         continue;
      }

      const real_t w = 1.0 / m;

      // Add SPD penalty:
      // value * (u_i - sum_k w v_k)^2
      //
      // (i,i)                    += value
      // (i,v_k), (v_k,i)         += -value * w
      // (v_k,v_l)                +=  value * w * w

      // Diagonal for u_i
      modified_A.Add(global_row, global_row, value);

      // Cross terms
      for (int k = 0; k < m; ++k)
      {
         const HYPRE_BigInt vk = group[k];
         modified_A.Add(global_row, vk, -value * w);
         modified_A.Add(vk, global_row, -value * w);
      }

      // Dense group-group block
      for (int k = 0; k < m; ++k)
      {
         const HYPRE_BigInt vk = group[k];
         for (int l = 0; l < m; ++l)
         {
            const HYPRE_BigInt vl = group[l];
            modified_A.Add(vk, vl, value * w * w);
         }
      }
   }
   
   /*for (int local_row = 0; local_row < local_num_rows; ++local_row)
   {
      HYPRE_BigInt global_row = row_start + local_row;

      // Get current row entries from merged_A
      const int *cols = merged_A.GetRowColumns(local_row);
      const real_t *vals = merged_A.GetRowEntries(local_row);
      int row_size = merged_A.RowSize(local_row);

      // Check if this is a tied DOF row
      auto it = tied_pairs.find(global_row);
      bool is_tied = (it != tied_pairs.end());

      if (!is_tied)
      {
         // Just copy the row as-is
         for (int j = 0; j < row_size; ++j)
         {
            modified_A.Add(local_row, cols[j], vals[j]);
         }
      }
      else
      {
         // Copy existing entries and modify/add penalty terms
         HYPRE_BigInt paired_gdof = it->second;
         bool found_diag = false;
         bool found_paired = false;
         const real_t value = alpha; // std::min(global_row, paired_gdof) % 2 == 0 ? alpha : 1;

         for (int j = 0; j < row_size; ++j)
         {
            real_t val = vals[j];

            // Modify diagonal entry
            if (cols[j] == global_row)
            {
               val += value;
               found_diag = true;
            }
            // Modify paired entry
            else if (cols[j] == paired_gdof)
            {
               val -= value;
               found_paired = true;
               assert(false);
            }

            modified_A.Add(local_row, cols[j], val);
         }

         // Add diagonal if it didn't exist
         if (!found_diag)
         {
            modified_A.Add(local_row, global_row, value);
            assert(false);
         }

         // Add paired coupling if it didn't exist
         if (!found_paired)
         {
            //modified_A.Add(local_row, paired_gdof, -value);
            
            const int *paired_cols = merged_A.GetRowColumns(local_row);
            int paired_row_size = merged_A.RowSize(local_row);
            int num_pairs = 0;

            for (int i = 0; i < paired_row_size; ++i)
            {
               if (all_tied_gdofs.Find(paired_cols[i]) != -1 && paired_cols[i] != paired_gdof)
               {
                  ++num_pairs;
               }
            }   

            for (int i = 0; i < paired_row_size; ++i)
            {
               if (all_tied_gdofs.Find(paired_cols[i]) != -1 && paired_cols[i] != paired_gdof)
               {
                  modified_A.Add(local_row, paired_cols[i], -value / 2.0 / num_pairs);
               }
            }
            modified_A.Add(local_row, paired_gdof, -value / 2.0);            
         }
      }
   }*/

   modified_A.Finalize();

   // Split back into diagonal and off-diagonal blocks
   HYPRE_BigInt col_start = A.GetColStarts()[0];
   HYPRE_BigInt col_end = A.GetColStarts()[1];

   // Separate into diagonal and off-diagonal entries
   int *modified_I = modified_A.GetI();
   int *modified_J = modified_A.GetJ();
   real_t *modified_data = modified_A.GetData();

   std::vector<int> diag_I_vec, diag_J_vec;
   std::vector<real_t> diag_data_vec;
   std::vector<int> offd_I_vec, offd_J_vec;
   std::vector<real_t> offd_data_vec;
   std::set<HYPRE_BigInt> offd_cols_set;

   diag_I_vec.push_back(0);
   offd_I_vec.push_back(0);

   for (int i = 0; i < local_num_rows; ++i)
   {
      for (int idx = modified_I[i]; idx < modified_I[i+1]; ++idx)
      {
         HYPRE_BigInt global_col = modified_J[idx];
         real_t value = modified_data[idx];

         if (global_col >= col_start && global_col < col_end)
         {
            // Diagonal block
            diag_J_vec.push_back(global_col - col_start);
            diag_data_vec.push_back(value);
         }
         else
         {
            // Off-diagonal block
            offd_cols_set.insert(global_col);
            offd_J_vec.push_back(global_col);  // Temporarily use global index
            offd_data_vec.push_back(value);
         }
      }
      diag_I_vec.push_back(diag_J_vec.size());
      offd_I_vec.push_back(offd_J_vec.size());
   }

   // Build column map for off-diagonal
   int offd_num_cols = offd_cols_set.size();
   HYPRE_BigInt *new_cmap = mfem::Memory<HYPRE_BigInt>(offd_num_cols,
                                                        Device::GetHostMemoryType());
   std::map<HYPRE_BigInt, int> global_to_local_offd;
   int idx = 0;
   for (HYPRE_BigInt gcol : offd_cols_set)
   {
      new_cmap[idx] = gcol;
      global_to_local_offd[gcol] = idx;
      idx++;
   }

   // Convert off-diagonal global columns to local
   for (int& col : offd_J_vec)
   {
      col = global_to_local_offd[col];
   }

   // Copy to MFEM memory
   HYPRE_Int *new_diag_I = mfem::Memory<HYPRE_Int>(local_num_rows + 1,
                                                     Device::GetHostMemoryType());
   HYPRE_Int *new_diag_J = mfem::Memory<HYPRE_Int>(diag_J_vec.size(),
                                                     Device::GetHostMemoryType());
   real_t *new_diag_data = mfem::Memory<real_t>(diag_data_vec.size(),
                                                  Device::GetHostMemoryType());

   HYPRE_Int *new_offd_I = mfem::Memory<HYPRE_Int>(local_num_rows + 1,
                                                     Device::GetHostMemoryType());
   HYPRE_Int *new_offd_J = mfem::Memory<HYPRE_Int>(offd_J_vec.size(),
                                                     Device::GetHostMemoryType());
   real_t *new_offd_data = mfem::Memory<real_t>(offd_data_vec.size(),
                                                  Device::GetHostMemoryType());

   std::copy(diag_I_vec.begin(), diag_I_vec.end(), new_diag_I);
   std::copy(diag_J_vec.begin(), diag_J_vec.end(), new_diag_J);
   std::copy(diag_data_vec.begin(), diag_data_vec.end(), new_diag_data);

   std::copy(offd_I_vec.begin(), offd_I_vec.end(), new_offd_I);
   std::copy(offd_J_vec.begin(), offd_J_vec.end(), new_offd_J);
   std::copy(offd_data_vec.begin(), offd_data_vec.end(), new_offd_data);

   // Construct new HypreParMatrix with penalty terms
   HypreParMatrix *tiedA = new HypreParMatrix(
      MPI_COMM_WORLD,
      A.M(), A.N(),
      A.GetRowStarts(),
      A.GetColStarts(),
      new_diag_I, new_diag_J, new_diag_data,
      new_offd_I, new_offd_J, new_offd_data,
      offd_num_cols,
      new_cmap,
      false  // hypre_arrays = false
   );

   //tiedA->Print("tiedA");

   // Optionally compute eigenvalues
   if (compute_eigenvalues)
   {
      if (Mpi::Root())
      {
         mfem::out << "\nComputing eigenvalues..." << endl;
      }

      // Use power iteration to estimate largest eigenvalue
      Vector x_rand(total_size);
      x_rand.Randomize(12345);
      Vector y(total_size);

      // Power iteration for largest eigenvalue
      double lambda_max = 0.0;
      for (int i = 0; i < 20; i++)
      {
         A.Mult(x_rand, y);
         lambda_max = InnerProduct(x_rand, y);
         double norm = sqrt(InnerProduct(y, y));
         x_rand = y;
         x_rand *= 1.0/norm;
      }

      // Inverse power iteration for smallest eigenvalue
      HypreBoomerAMG amg_inv(A);
      amg_inv.SetPrintLevel(0);
      GMRESSolver inv_solver(MPI_COMM_WORLD);
      inv_solver.SetRelTol(1e-6);
      inv_solver.SetMaxIter(100);
      inv_solver.SetPrintLevel(0);
      inv_solver.SetOperator(A);
      inv_solver.SetPreconditioner(amg_inv);

      x_rand.Randomize(54321);
      double lambda_min = 0.0;
      for (int i = 0; i < 20; i++)
      {
         inv_solver.Mult(x_rand, y);
         lambda_min = InnerProduct(x_rand, y);
         double norm = sqrt(InnerProduct(y, y));
         x_rand = y;
         x_rand *= 1.0/norm;
      }
      lambda_min = 1.0 / lambda_min;

      if (Mpi::Root())
      {
         mfem::out << "\nEstimated eigenvalue range:" << endl;
         mfem::out << "  lambda_min = " << lambda_min << endl;
         mfem::out << "  lambda_max = " << lambda_max << endl;
         mfem::out << "  Condition number estimate = " << lambda_max/lambda_min << endl;
         mfem::out << endl;
      }
   }

   Solver * prec = nullptr;
   Solver * subspacesolver = nullptr;
   HypreParMatrix * P_tied_T = nullptr;
   HypreParMatrix * P_tied = nullptr;

   if (do_amgf)
   {
      prec = new AMGFSolver();
      auto * amgfprec = dynamic_cast<AMGFSolver *>(prec);
      amgfprec->GetAMG().SetPrintLevel(0);
      amgfprec->GetAMG().SetStrengthThresh(0.5);

      if (one_level_amg)
      {
         amgfprec->GetAMG().SetMaxLevels(1);
      }

      if (iterative_filter)
      {
         subspacesolver = new HypreBoomerAMG();
         HypreBoomerAMG * subspaceamgsolver = dynamic_cast<HypreBoomerAMG*>(subspacesolver);
         subspaceamgsolver->SetMaxLevels(1);
         subspaceamgsolver->SetRelaxType(0);
         subspaceamgsolver->SetPrintLevel(0);
         HYPRE_BoomerAMGSetNumFunctions(*subspaceamgsolver, 1);
         HYPRE_BoomerAMGSetDomainType(*subspaceamgsolver, 0);
         HYPRE_BoomerAMGSetOverlap(*subspaceamgsolver, 1);
         HYPRE_BoomerAMGSetVariant(*subspaceamgsolver, 0);
         HYPRE_BoomerAMGSetSchwarzRlxWeight(*subspaceamgsolver, 0);
         HYPRE_BoomerAMGSetSmoothType(*subspaceamgsolver, 6);
         HYPRE_BoomerAMGSetSmoothNumLevels(*subspaceamgsolver, 1);
      }
      else{
         subspacesolver = new ParallelDirectSolver(MPI_COMM_WORLD, "superlu");
         ParallelDirectSolver * subspacedirectsolver = dynamic_cast<ParallelDirectSolver*>(subspacesolver);
         subspacedirectsolver->SetPrintLevel(0);
      }

      amgfprec->SetFilteredSubspaceSolver(*subspacesolver);

      HYPRE_BigInt row_start = A.GetRowStarts()[0];
      HYPRE_BigInt row_end = A.GetRowStarts()[1];

      Array<int> owned_tied_gdofs;
      for (int i = Mpi::WorldRank(); i < all_tied_gdofs.Size() / 2; i += Mpi::WorldSize())
      {
         owned_tied_gdofs.Append(all_tied_gdofs[i]);
         owned_tied_gdofs.Append(all_tied_gdofs[i + all_tied_gdofs.Size() / 2]);
      }

      int nrows_tied = owned_tied_gdofs.Size();
      SparseMatrix Pct(nrows_tied,fespace.GlobalTrueVSize());

      for (int i = 0; i < nrows_tied; ++i)
      {
         Pct.Set(i, owned_tied_gdofs[i], 1.0);
      }
      Pct.Finalize();

      HYPRE_BigInt rows_c[2];

      HYPRE_BigInt row_offset_tied;
      HYPRE_BigInt nrows_tied_bigint = nrows_tied;
      MPI_Scan(&nrows_tied_bigint,&row_offset_tied,1,MPI_INT,
               MPI_SUM,MPI_COMM_WORLD);

      row_offset_tied-=nrows_tied_bigint;
      rows_c[0] = row_offset_tied;
      rows_c[1] = row_offset_tied+nrows_tied;

      HYPRE_BigInt glob_nrows_tied;
      HYPRE_BigInt glob_ncols_tied = fespace.GlobalTrueVSize();
      MPI_Allreduce(&nrows_tied_bigint, &glob_nrows_tied,1,
                  MPI_INT,
                  MPI_SUM,MPI_COMM_WORLD);
   HYPRE_BigInt * J;
#ifndef HYPRE_BIGINT
      J = Pct.GetJ();
#else
      J = new HYPRE_BigInt[Pct.NumNonZeroElems()];
      for (int i = 0; i < Pct.NumNonZeroElems(); i++)
      {
         J[i] = Pct.GetJ()[i];
      }
#endif
      P_tied_T = new HypreParMatrix(MPI_COMM_WORLD, nrows_tied, glob_nrows_tied,
                                                glob_ncols_tied, Pct.GetI(), J,
                                                Pct.GetData(), rows_c,fespace.GetTrueDofOffsets());
      P_tied = P_tied_T->Transpose();
         
      amgfprec->SetFilteredSubspaceTransferOperator(
         *P_tied);

#ifdef HYPRE_BIGINT
      delete [] J;
#endif
   }
   else
   {
      prec = new HypreBoomerAMG();
      auto * amgprec = dynamic_cast<HypreBoomerAMG *>(prec);
      amgprec->SetPrintLevel(0);
      amgprec->SetStrengthThresh(0.125);
      
      if (one_level_amg)
      {
         amgprec->SetMaxLevels(1);
      }
   }

   CGSolver cg(MPI_COMM_WORLD);
   cg.SetRelTol(1e-12);
   cg.SetMaxIter(pcg_max_iters);
   cg.SetPrintLevel(1);
   cg.SetPreconditioner(*prec);
   cg.SetOperator(*tiedA);
   cg.Mult(B, X);

   // Recover the grid function solution
   a.RecoverFEMSolution(X, b, x);

   if (Mpi::Root())
   {
      mfem::out << "Solutions saved to tied-poisson-sol[1,2] and tied-poisson-mesh[1,2]" << endl;
   }

   // Send to GLVis if requested
   if (visualization)
   {
      // Save solutions
      x.Save("tied-poisson-sol");
      pmesh.Save("tied-poisson-mesh");

      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock << "parallel " << Mpi::WorldSize() << " " << Mpi::WorldRank() << "\n";
      sol_sock.precision(8);
      sol_sock << "solution\n" << pmesh << x << flush;
   }

   delete combined_mesh;
   delete tiedA;
   delete prec;

   if (do_amgf)
   {
      delete P_tied_T;
      delete P_tied;
      delete subspacesolver;
   }

   return 0;
}
