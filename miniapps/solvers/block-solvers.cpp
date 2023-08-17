// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.
//
//          ----------------------------------------------------------
//          Block Solvers Miniapp: Compare Saddle Point System Solvers
//          ----------------------------------------------------------
//
// This miniapp compares various linear solvers for the saddle point system
// obtained from mixed finite element discretization of the simple mixed Darcy
// problem in ex5p
//
//                            k*u + grad p = f
//                           - div u      = g
//
// with natural boundary condition -p = <given pressure>. We use a given exact
// solution (u,p) and compute the corresponding r.h.s. (f,g). We discretize
// with Raviart-Thomas finite elements (velocity u) and piecewise discontinuous
// polynomials (pressure p).
//
// The solvers being compared include:
//    1. The divergence free solver (couple and decoupled modes)
//    2. MINRES preconditioned by a block diagonal preconditioner
//    3. CG with a Bramble-Pasciak transformation
//
// We recommend viewing example 5 before viewing this miniapp.
//
// Sample runs:
//
//    mpirun -np 8 block-solvers -r 2 -o 0
//    mpirun -np 8 block-solvers -m anisotropic.mesh -c anisotropic.coeff -be anisotropic.bdr
//
//
// NOTE:  The coefficient file (provided through -c) defines a piecewise constant
//        scalar coefficient k. The number of entries in this file should equal
//        to the number of "element attributes" in the mesh file. The value of
//        the coefficient in elements with the i-th attribute is given by the
//        i-th entry of the coefficient file.
//
//
// NOTE:  The essential boundary attribute file (provided through -eb) defines
//        which attributes to impose essential boundary condition (on u). The
//        number of entries in this file should equal to the number of "boundary
//        attributes" in the mesh file. If the i-th entry of the file is nonzero
//        (respectively 0), essential (respectively natural) boundary condition
//        will be imposed on boundary with the i-th attribute.

#include "mfem.hpp"
#include "bramble_pasciak.hpp"
#include "div_free_solver.hpp"
#include <fstream>
#include <iostream>
#include <memory>

using namespace std;
using namespace mfem;
using namespace blocksolvers;

int main(int argc, char *argv[])
{
#ifdef HYPRE_USING_GPU
   cout << "\nAs of mfem-4.3 and hypre-2.22.0 (July 2021) this miniapp\n"
        << "is NOT supported with the GPU version of hypre.\n\n";
   return 242;
#endif

   // Initialize MPI and HYPRE.
   Mpi::Init(argc, argv);
   Hypre::Init();

   StopWatch chrono;
   auto ResetTimer = [&chrono]() { chrono.Clear(); chrono.Start(); };

   // Parse command-line options.
   const char *mesh_file = "../../data/beam-hex.mesh";
   const char *coef_file = "";
   const char *ess_bdr_attr_file = "";
   int order = 0;
   int par_ref_levels = 2;
   bool show_error = false;
   bool visualization = false;
   bool enable_bpcg = true;
   bool enable_hpc = false;

   DFSParameters param;
   BPCGParameters bpcg_param;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&par_ref_levels, "-r", "--ref",
                  "Number of parallel refinement steps.");
   args.AddOption(&coef_file, "-c", "--coef",
                  "Coefficient file to use.");
   args.AddOption(&ess_bdr_attr_file, "-eb", "--ess-bdr",
                  "Essential boundary attribute file to use.");
   args.AddOption(&show_error, "-se", "--show-error", "-no-se",
                  "--no-show-error",
                  "Show or not show approximation error.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&enable_bpcg, "-bp", "--bpcg", "-no-bp",
                  "--no-bpcg",
                  "Enable or disable Bramble-Pasciak CG method (BPCG-only).");
   args.AddOption(&enable_hpc, "-hp", "--h-pc", "-no-hp",
                  "--no-h-pc",
                  "Enable or disable H preconditioner (BPCG-only).");
   args.Parse();
   if (!args.Good())
   {
      if (Mpi::Root()) { args.PrintUsage(cout); }
      return 1;
   }
   if (Mpi::Root()) { args.PrintOptions(cout); }

   if (Mpi::Root() && par_ref_levels == 0)
   {
      std::cout << "WARNING: DivFree solver is equivalent to BDPMinresSolver "
                << "when par_ref_levels == 0.\n";
   }

   bpcg_param.use_bpcg = enable_bpcg;
   bpcg_param.use_hpc = enable_hpc;

   // Initialize the mesh, boundary attributes, and solver parameters
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();
   int ser_ref_lvls =
      (int)ceil(log(Mpi::WorldSize()/mesh->GetNE())/log(2.)/dim);
   for (int i = 0; i < ser_ref_lvls; ++i)
   {
      mesh->UniformRefinement();
   }

   Array<int> ess_bdr(mesh->bdr_attributes.Max());
   ess_bdr = 0;
   if (std::strcmp(ess_bdr_attr_file, ""))
   {
      ifstream ess_bdr_attr_str(ess_bdr_attr_file);
      ess_bdr.Load(mesh->bdr_attributes.Max(), ess_bdr_attr_str);
   }
   if (IsAllNeumannBoundary(ess_bdr))
   {
      if (Mpi::Root())
      {
         cout << "\nSolution is not unique when Neumann boundary condition is "
              << "imposed on the entire boundary. \nPlease provide a different "
              << "boundary condition.\n";
      }
      delete mesh;
      return 0;
   }

   string line = "**********************************************************\n";

   ResetTimer();

   // Generate components of the saddle point problem
   DarcyProblem darcy(*mesh, par_ref_levels, order, coef_file, ess_bdr, param);
   HypreParMatrix& M = darcy.GetM();
   HypreParMatrix& B = darcy.GetB();
   const DFSData& DFS_data = darcy.GetDFSData();
   delete mesh;

   if (Mpi::Root())
   {
      cout << line << "System assembled in " << chrono.RealTime() << "s.\n";
      cout << "Dimension of the physical space: " << dim << "\n";
      cout << "Size of the discrete Darcy system: " << M.M() + B.M() << "\n";
      if (par_ref_levels > 0)
      {
         cout << "Dimension of the divergence free subspace: "
              << DFS_data.C.back().Ptr()->NumCols() << "\n\n";
      }
   }

   // Setup various solvers for the discrete problem
   std::map<const DarcySolver*, double> setup_time;
   ResetTimer();
   BDPMinresSolver bdp(M, B, param);
   setup_time[&bdp] = chrono.RealTime();

   ResetTimer();
   DivFreeSolver dfs_dm(M, B, DFS_data);
   setup_time[&dfs_dm] = chrono.RealTime();

   ResetTimer();
   const_cast<bool&>(DFS_data.param.coupled_solve) = true;
   DivFreeSolver dfs_cm(M, B, DFS_data);
   setup_time[&dfs_cm] = chrono.RealTime();

   ResetTimer();
   BramblePasciakSolver bp(darcy.GetMform(), darcy.GetBform(), bpcg_param);
   setup_time[&bp] = chrono.RealTime();

   std::map<const DarcySolver*, std::string> solver_to_name;
   solver_to_name[&bdp] = "Block-diagonal-preconditioned MINRES";
   solver_to_name[&dfs_dm] = "Divergence free (decoupled mode)";
   solver_to_name[&dfs_cm] = "Divergence free (coupled mode)";
   solver_to_name[&bp] = bpcg_param.use_bpcg ? "Bramble Pasciak CG (BPCG)" :
                         "Bramble Pasciak CG (BP Transformation + PCG)";

   // Solve the problem using all solvers
   for (const auto& solver_pair : solver_to_name)
   {
      auto& solver = solver_pair.first;
      auto& name = solver_pair.second;

      Vector sol = darcy.GetEssentialBC();

      ResetTimer();
      solver->Mult(darcy.GetRHS(), sol);
      chrono.Stop();

      if (Mpi::Root())
      {
         cout << line << name << " solver:\n   Setup time: "
              << setup_time[solver] << "s.\n   Solve time: "
              << chrono.RealTime() << "s.\n   Total time: "
              << setup_time[solver] + chrono.RealTime() << "s.\n"
              << "   Iteration count: " << solver->GetNumIterations() <<"\n\n";
      }
      if (show_error && std::strcmp(coef_file, "") == 0)
      {
         darcy.ShowError(sol, Mpi::Root());
      }
      else if (show_error && Mpi::Root())
      {
         cout << "Exact solution is unknown for coefficient '" << coef_file
              << "'.\nApproximation error is computed in this case!\n\n";
      }

      if (visualization) { darcy.VisualizeSolution(sol, name); }

   }

   return 0;
}
