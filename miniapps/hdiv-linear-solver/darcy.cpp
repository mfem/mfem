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
//
//                     ---------------------------------
//                     Poisson/Darcy Mixed Method Solver
//                     ---------------------------------
//
// Solves a Poisson problem -Delta p = f using a mixed finite element
// formulation). The right-hand side of the Poisson problem is the same as that
// used in the LOR Solvers miniapp (see miniapps/solvers). Dirichlet boundary
// conditions are enforced on all domain boundaries.
//
// Optionally, the equation alpha*p - Delta p = f can be solved by setting the
// alpha parameter to a nonzero value.

// This can be written in the form of a Darcy problem
//
//                           -u - grad(p) = 0
//                       alpha*p + div(u) = f
//
// where natural boundary conditions are enforced on the flux u, and the
// Dirichlet condition on p is enforced by modifying the right-hand side.
//
// The resulting saddle-point system is solved using MINRES with a matrix-free
// block-diagonal preconditioner.
//
// See also example 5 and its parallel version.
//
// Sample runs:
//
//    darcy
//    mpirun -np 4 darcy -m ../../data/fichera-q2.mesh

#include "mfem.hpp"
#include <cstring>
#include <iostream>
#include <memory>

#ifdef MFEM_USE_UMPIRE
#include <umpire/Allocator.hpp>
#include <umpire/ResourceManager.hpp>
#endif

#include "discrete_divergence.hpp"
#include "hdiv_linear_solver.hpp"

#include "../solvers/lor_mms.hpp"

using namespace std;
using namespace mfem;

ParMesh LoadParMesh(const char *mesh_file, int ser_ref = 0, int par_ref = 0);

namespace
{

HdivSaddlePointSolver::L2InverseType ParseL2InverseType(const char *name)
{
   if (!name || strcmp(name, "cg") == 0)
   {
      return HdivSaddlePointSolver::L2InverseType::CG;
   }
   if (strcmp(name, "magma-packed") == 0)
   {
      return HdivSaddlePointSolver::L2InverseType::MAGMA_PACKED;
   }
   if (strcmp(name, "magma-full") == 0)
   {
      return HdivSaddlePointSolver::L2InverseType::MAGMA_FULL;
   }
   MFEM_ABORT("Unknown -l2inv value: " << name
              << " (expected: cg | magma-packed | magma-full)");
   return HdivSaddlePointSolver::L2InverseType::CG;
}

#ifdef MFEM_USE_UMPIRE
void ReportUmpireAllocator(const char *label, const char *alloc_name)
{
   auto &rm = umpire::ResourceManager::getInstance();
   if (!rm.isAllocator(alloc_name))
   {
      if (Mpi::Root())
      {
         cout << label << ": allocator '" << alloc_name
              << "' not found (no allocations yet?)\n";
      }
      return;
   }

   auto alloc = rm.getAllocator(alloc_name);
   const unsigned long long cur = alloc.getCurrentSize();
   const unsigned long long hwm = alloc.getHighWatermark();
   unsigned long long cur_sum = 0, cur_max = 0;
   unsigned long long hwm_sum = 0, hwm_max = 0;

   MPI_Reduce(&cur, &cur_sum, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0,
              MPI_COMM_WORLD);
   MPI_Reduce(&cur, &cur_max, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, 0,
              MPI_COMM_WORLD);
   MPI_Reduce(&hwm, &hwm_sum, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0,
              MPI_COMM_WORLD);
   MPI_Reduce(&hwm, &hwm_max, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, 0,
              MPI_COMM_WORLD);

   if (Mpi::Root())
   {
      cout << label << " (Umpire '" << alloc_name << "'): "
           << "current(sum/max)=(" << cur_sum << "/" << cur_max << ") bytes, "
           << "hwm(sum/max)=(" << hwm_sum << "/" << hwm_max << ") bytes\n";
   }
}

void ReportUmpireMemory(const char *label)
{
   if (Mpi::Root()) { cout << label << '\n'; }
   ReportUmpireAllocator("  host", MemoryManager::GetUmpireHostAllocatorName());
   ReportUmpireAllocator("  device",
                         MemoryManager::GetUmpireDeviceAllocatorName());
}
#else
void ReportUmpireMemory(const char *) { }
#endif

} // namespace

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();

   const char *mesh_file = "../../data/star.mesh";
   const char *device_config = "cpu";
   int ser_ref = 1;
   int par_ref = 1;
   int order = 3;
   real_t alpha = 0.0;
   const char *l2inv = "cg";
   bool bench_l2inv = false;
   int l2inv_reps = 100;
   bool use_umpire_pool = false;
   bool report_umpire_mem = false;

   OptionsParser args(argc, argv);
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&ser_ref, "-rs", "--serial-refine",
                  "Number of times to refine the mesh in serial.");
   args.AddOption(&par_ref, "-rp", "--parallel-refine",
                  "Number of times to refine the mesh in parallel.");
   args.AddOption(&order, "-o", "--order", "Polynomial degree.");
   args.AddOption(&alpha, "-a", "--alpha", "Value of alpha coefficient.");
   args.AddOption(&l2inv, "-l2inv", "--l2-inverse",
                  "Local L2 mass inverse: cg | magma-packed | magma-full.");
   args.AddOption(&bench_l2inv, "-l2bench", "--l2-bench",
                  "-no-l2bench", "--no-l2-bench",
                  "Benchmark the local L2 inverse apply.");
   args.AddOption(&l2inv_reps, "-l2reps", "--l2-repetitions",
                  "Repetitions for -l2bench timing.");
   args.AddOption(&use_umpire_pool, "-umpire-pool", "--umpire-pool",
                  "-no-umpire-pool", "--no-umpire-pool",
                  "Use Umpire QuickPool allocators for MFEM allocations.");
   args.AddOption(&report_umpire_mem, "-mem", "--report-memory",
                  "-no-mem", "--no-report-memory",
                  "Report Umpire allocator memory usage.");
   args.ParseCheck();

#ifdef MFEM_USE_UMPIRE
   if (use_umpire_pool)
   {
      MemoryManager::SetUmpireHostAllocatorName("mfem_host_pool");
      MemoryManager::SetUmpireDeviceAllocatorName("mfem_device_pool");
   }
#else
   MFEM_VERIFY(!use_umpire_pool, "MFEM was built without Umpire support.");
   MFEM_VERIFY(!report_umpire_mem, "MFEM was built without Umpire support.");
#endif

   Device device(device_config);
   if (Mpi::Root()) { device.Print(); }

   ParMesh mesh = LoadParMesh(mesh_file, ser_ref, par_ref);
   const int dim = mesh.Dimension();
   MFEM_VERIFY(dim == 2 || dim == 3, "Spatial dimension must be 2 or 3.");

   const int b1 = BasisType::GaussLobatto, b2 = BasisType::GaussLegendre;
   const int mt = FiniteElement::VALUE;
   RT_FECollection fec_rt(order-1, dim, b1, b2);
   L2_FECollection fec_l2(order-1, dim, b2, mt);
   ParFiniteElementSpace fes_rt(&mesh, &fec_rt);
   ParFiniteElementSpace fes_l2(&mesh, &fec_l2);

   HYPRE_BigInt ndofs_rt = fes_rt.GlobalTrueVSize();
   HYPRE_BigInt ndofs_l2 = fes_l2.GlobalTrueVSize();

   if (Mpi::Root())
   {
      cout << "\nRT DOFs: " << ndofs_rt << "\nL2 DOFs: " << ndofs_l2 << endl;
   }

   Array<int> ess_rt_dofs; // empty

   // f is the RHS, u is the exact solution
   FunctionCoefficient f_coeff(f(alpha)), u_coeff(u);

   // Assemble the right-hand side for the scalar (L2) unknown.
   ParLinearForm b_l2(&fes_l2);
   b_l2.AddDomainIntegrator(new DomainLFIntegrator(f_coeff));
   b_l2.UseFastAssembly(true);
   b_l2.Assemble();

   // Enforce Dirichlet boundary conditions on the scalar unknown by adding
   // the boundary term to the flux equation.
   ParLinearForm b_rt(&fes_rt);
   b_rt.AddBoundaryIntegrator(new VectorFEBoundaryFluxLFIntegrator(u_coeff));
   b_rt.UseFastAssembly(true);
   b_rt.Assemble();

   if (Mpi::Root()) { cout << "\nSaddle point solver... " << flush; }
   tic_toc.Clear(); tic_toc.Start();

   // Set up the block system of the form
   //
   // [  W     D ][ u ] = [  f  ]
   // [ D^T   -M ][ q ] = [ g_D ]
   //
   // where W is the L2 mass matrix, D is the discrete divergence, and M is
   // the RT mass matrix.
   //
   // If the coefficient alpha is set to zero, the system takes the form
   //
   // [  0     D ][ u ] = [  f  ]
   // [ D^T   -M ][ q ] = [ g_D ]
   //
   // u is the scalar unknown, and q is the flux. f is the right-hand side from
   // the Poisson problem, and g_D is the contribution to the right-hand side
   // from the Dirichlet boundary condition.

   ConstantCoefficient one(1.0);
   ConstantCoefficient alpha_coeff(alpha);
   const auto solver_mode = HdivSaddlePointSolver::Mode::DARCY;
   const auto l2inv_type = ParseL2InverseType(l2inv);

   StopWatch setup_sw;
   setup_sw.Start();
   HdivSaddlePointSolver saddle_point_solver(
      mesh, fes_rt, fes_l2, alpha_coeff, one, ess_rt_dofs, solver_mode, l2inv_type);
   MFEM_DEVICE_SYNC;
   setup_sw.Stop();

   if (Mpi::Root())
   {
      const int n = fes_l2.GetTypicalFE()->GetDof();
      const int ne = mesh.GetNE();
      const size_t full_bytes = static_cast<size_t>(ne)*n*n*sizeof(real_t);
      const size_t packed_bytes = static_cast<size_t>(ne)*n*(n+1)/2*sizeof(real_t);
      cout << "Setup time: " << setup_sw.RealTime() << " s\n"
           << "Local element matrices (theoretical): full=" << full_bytes
           << " bytes, packed=" << packed_bytes << " bytes\n";
   }
   if (report_umpire_mem) { ReportUmpireMemory("After setup"); }

   const Array<int> &offsets = saddle_point_solver.GetOffsets();
   BlockVector X_block(offsets), B_block(offsets);

   b_l2.ParallelAssemble(B_block.GetBlock(0));
   b_rt.ParallelAssemble(B_block.GetBlock(1));
   B_block.SyncFromBlocks();

   X_block = 0.0;
   saddle_point_solver.Mult(B_block, X_block);
   X_block.SyncToBlocks();

   if (Mpi::Root())
   {
      cout << "Done.\nIterations: "
           << saddle_point_solver.GetNumIterations()
           << "\nElapsed: " << tic_toc.RealTime() << endl;
   }
   if (report_umpire_mem) { ReportUmpireMemory("After solve"); }

   if (bench_l2inv)
   {
      const int n_l2 = offsets[1];
      Vector rhs(n_l2), x_l2(n_l2);
      rhs.UseDevice(true);
      x_l2.UseDevice(true);
      rhs.Randomize(1);

      // Warm up and time repeated applications.
      saddle_point_solver.GetL2Inverse().Mult(rhs, x_l2);
      MFEM_DEVICE_SYNC;

      StopWatch sw;
      sw.Start();
      for (int r = 0; r < l2inv_reps; ++r)
      {
         saddle_point_solver.GetL2Inverse().Mult(rhs, x_l2);
      }
      MFEM_DEVICE_SYNC;
      sw.Stop();

      const double local_ms = 1000.0*sw.RealTime()/l2inv_reps;
      double max_ms = 0.0;
      MPI_Reduce(&local_ms, &max_ms, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      if (Mpi::Root())
      {
         cout << "L2 inverse apply (ms/apply, max over ranks): " << max_ms << '\n';
      }
   }

   ParGridFunction x(&fes_l2);
   x.SetFromTrueDofs(X_block.GetBlock(0));
   const real_t error = x.ComputeL2Error(u_coeff);
   if (Mpi::Root()) { cout << "L2 error: " << error << endl; }

   return 0;
}

ParMesh LoadParMesh(const char *mesh_file, int ser_ref, int par_ref)
{
   Mesh serial_mesh = Mesh::LoadFromFile(mesh_file);
   for (int i = 0; i < ser_ref; ++i) { serial_mesh.UniformRefinement(); }
   ParMesh mesh(MPI_COMM_WORLD, serial_mesh);
   serial_mesh.Clear();
   for (int i = 0; i < par_ref; ++i) { mesh.UniformRefinement(); }
   return mesh;
}
