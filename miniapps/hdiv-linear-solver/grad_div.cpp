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
//                     H(div) saddle-point system solver
//                     ---------------------------------
//
// Solves the grad-div problem u - grad(div(u)) = f using a variety of solver
// techniques. This miniapp supports solving this problem using a variety of
// matrix-free and matrix-based preconditioning methods, including:
//
// * Matrix-free block-diagonal preconditioning for the saddle-point system.
// * ADS-AMG preconditioning.
// * Low-order-refined ADS-AMG preconditioning (matrix-free).
// * Hybridization with AMG preconditioning.
//
// The problem setup is the same as in the LOR solvers miniapps (in the
// miniapps/solvers directory). Dirichlet conditions are enforced on the normal
// component of u.
//
// Sample runs:
//
//    grad_div -sp -ams -lor -hb
//    mpirun -np 4 grad_div -sp -ams -lor -hb -m ../../data/fichera-q2.mesh -rp 0

#include "mfem.hpp"
#include <cstring>
#include <iostream>
#include <memory>
#ifdef MFEM_USE_UMPIRE
#include <umpire/Allocator.hpp>
#include <umpire/ResourceManager.hpp>
#endif
#include "hdiv_linear_solver.hpp"
#include "../solvers/lor_mms.hpp"

using namespace std;
using namespace mfem;

ParMesh LoadParMesh(const char *mesh_file, int ser_ref = 0, int par_ref = 0);
void SolveCG(Operator &A, Solver &P, const Vector &B, Vector &X);

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
   if (!rm.isAllocator(alloc_name)) { return; }
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
   bool use_saddle_point = false;
   bool use_ams = false;
   bool use_lor_ams = false;
   bool use_hybridization = false;
   const char *l2inv = "cg";
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
   args.AddOption(&use_saddle_point,
                  "-sp", "--saddle-point", "-no-sp", "--no-saddle-point",
                  "Enable or disable saddle-point solver.");
   args.AddOption(&use_ams, "-ams", "--ams", "-no-ams", "--no-ams",
                  "Enable or disable AMS solver.");
   args.AddOption(&use_lor_ams, "-lor", "--lor-ams", "-no-lor", "--no-lor-ams",
                  "Enable or disable LOR-AMS solver.");
   args.AddOption(&use_hybridization,
                  "-hb", "--hybridization", "-no-hb", "--no-hybridization",
                  "Enable or disable hybridization solver.");
   args.AddOption(&l2inv, "-l2inv", "--l2-inverse",
                  "Local L2 mass inverse: cg | magma-packed | magma-full.");
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

   if (!use_saddle_point && !use_ams && !use_lor_ams && !use_hybridization)
   {
      if (Mpi::Root()) { cout << "No solver enabled. Exiting.\n"; }
      return 0;
   }

   Device device(device_config);
   if (Mpi::Root()) { device.Print(); }

   ParMesh mesh = LoadParMesh(mesh_file, ser_ref, par_ref);
   const int dim = mesh.Dimension();
   MFEM_VERIFY(dim == 2 || dim == 3, "Spatial dimension must be 2 or 3.");

   const int b1 = BasisType::GaussLobatto, b2 = BasisType::GaussLegendre;
   RT_FECollection fec_rt(order-1, dim, b1, b2);
   ParFiniteElementSpace fes_rt(&mesh, &fec_rt);

   Array<int> ess_rt_dofs;
   fes_rt.GetBoundaryTrueDofs(ess_rt_dofs);

   VectorFunctionCoefficient f_vec_coeff(dim, f_vec(true)), u_vec_coeff(dim,
                                                                        u_vec);

   ParLinearForm b(&fes_rt);
   b.AddDomainIntegrator(new VectorFEDomainLFIntegrator(f_vec_coeff));
   b.UseFastAssembly(true);
   b.Assemble();

   ConstantCoefficient alpha_coeff(1.0);
   ConstantCoefficient beta_coeff(1.0);

   ParGridFunction x(&fes_rt);
   x.ProjectCoefficient(u_vec_coeff);

   cout.precision(4);
   cout << scientific;

   if (use_saddle_point)
   {
      if (Mpi::Root()) { cout << "\nSaddle point solver... " << flush; }
      tic_toc.Clear(); tic_toc.Start();

      const int mt = FiniteElement::INTEGRAL;
      L2_FECollection fec_l2(order-1, dim, b2, mt);
      ParFiniteElementSpace fes_l2(&mesh, &fec_l2);

      const auto l2inv_type = ParseL2InverseType(l2inv);
      HdivSaddlePointSolver saddle_point_solver(
         mesh, fes_rt, fes_l2, alpha_coeff, beta_coeff, ess_rt_dofs,
         HdivSaddlePointSolver::Mode::GRAD_DIV, l2inv_type);
      if (report_umpire_mem) { ReportUmpireMemory("After saddle-point setup"); }

      const Array<int> &offsets = saddle_point_solver.GetOffsets();

      BlockVector X_block(offsets), B_block(offsets);
      B_block.GetBlock(0) = 0.0;
      b.ParallelAssemble(B_block.GetBlock(1));
      B_block.GetBlock(1) *= -1.0;
      B_block.SyncFromBlocks();

      x.ParallelProject(X_block.GetBlock(1));
      saddle_point_solver.SetBC(X_block.GetBlock(1));

      X_block = 0.0;
      saddle_point_solver.Mult(B_block, X_block);

      if (Mpi::Root())
      {
         cout << "Done.\nIterations: "
              << saddle_point_solver.GetNumIterations()
              << "\nElapsed: " << tic_toc.RealTime() << endl;
      }

      X_block.SyncToBlocks();
      x.SetFromTrueDofs(X_block.GetBlock(1));
      const real_t error = x.ComputeL2Error(u_vec_coeff);
      if (Mpi::Root()) { cout << "L2 error: " << error << endl; }
   }

   if (use_ams)
   {
      x.ProjectCoefficient(u_vec_coeff);

      if (Mpi::Root()) { cout << "\nAMS solver... " << flush; }
      tic_toc.Clear(); tic_toc.Start();

      ParBilinearForm a(&fes_rt);
      a.AddDomainIntegrator(new DivDivIntegrator(alpha_coeff));
      a.AddDomainIntegrator(new VectorFEMassIntegrator(beta_coeff));
      a.Assemble();

      OperatorHandle A;
      Vector B, X;
      a.FormLinearSystem(ess_rt_dofs, x, b, A, X, B);
      HypreParMatrix &Ah = *A.As<HypreParMatrix>();

      std::unique_ptr<Solver> prec;
      if (dim == 2) { prec.reset(new HypreAMS(Ah, &fes_rt)); }
      else  { prec.reset(new HypreADS(Ah, &fes_rt)); }

      SolveCG(Ah, *prec, B, X);
      x.SetFromTrueDofs(X);
      const real_t error = x.ComputeL2Error(u_vec_coeff);
      if (Mpi::Root()) { cout << "L2 error: " << error << endl; }
   }

   if (use_lor_ams)
   {
      const int b2_lor = BasisType::IntegratedGLL;
      RT_FECollection fec_rt_lor(order-1, dim, b1, b2_lor);
      ParFiniteElementSpace fes_rt_lor(&mesh, &fec_rt_lor);

      ParGridFunction x_lor(&fes_rt_lor);
      x_lor.ProjectCoefficient(u_vec_coeff);

      ParLinearForm b_lor(&fes_rt_lor);
      b_lor.AddDomainIntegrator(new VectorFEDomainLFIntegrator(f_vec_coeff));
      b_lor.UseFastAssembly(true);
      b_lor.Assemble();

      if (Mpi::Root()) { cout << "\nLOR-AMS solver... " << flush; }
      tic_toc.Clear(); tic_toc.Start();

      ParBilinearForm a(&fes_rt_lor);
      a.SetAssemblyLevel(AssemblyLevel::PARTIAL);
      a.AddDomainIntegrator(new DivDivIntegrator(alpha_coeff));
      a.AddDomainIntegrator(new VectorFEMassIntegrator(beta_coeff));
      a.Assemble();

      OperatorHandle A;
      Vector B, X;
      a.FormLinearSystem(ess_rt_dofs, x_lor, b_lor, A, X, B);

      std::unique_ptr<Solver> prec;
      if (dim == 2) { prec.reset(new LORSolver<HypreAMS>(a, ess_rt_dofs)); }
      else { prec.reset(new LORSolver<HypreADS>(a, ess_rt_dofs)); }

      SolveCG(*A, *prec, B, X);
      a.RecoverFEMSolution(X, b_lor, x_lor);
      const real_t error = x_lor.ComputeL2Error(u_vec_coeff);
      if (Mpi::Root()) { cout << "L2 error: " << error << endl; }
   }

   if (use_hybridization)
   {
      // Don't include cuBLAS setup time in the hybridization timings
      GPUBlas::Handle();

      x.ProjectCoefficient(u_vec_coeff);

      if (Mpi::Root()) { cout << "\nHybridization solver... " << flush; }
      tic_toc.Clear(); tic_toc.Start();

      DG_Interface_FECollection fec_hb(order-1, dim);
      ParFiniteElementSpace fes_hb(&mesh, &fec_hb);

      ParBilinearForm a(&fes_rt);
      a.AddDomainIntegrator(new DivDivIntegrator(alpha_coeff));
      a.AddDomainIntegrator(new VectorFEMassIntegrator(beta_coeff));
      a.SetAssemblyLevel(AssemblyLevel::ELEMENT);
      a.EnableHybridization(&fes_hb, new NormalTraceJumpIntegrator, ess_rt_dofs);
      a.Assemble();

      OperatorHandle A;
      Vector B, X;
      a.FormLinearSystem(ess_rt_dofs, x, b, A, X, B);

      HypreBoomerAMG amg_hb(*A.As<HypreParMatrix>());
      amg_hb.SetPrintLevel(0);

      SolveCG(*A, amg_hb, B, X);
      a.RecoverFEMSolution(X, b, x);
      const real_t error = x.ComputeL2Error(u_vec_coeff);
      if (Mpi::Root()) { cout << "L2 error: " << error << endl; }
   }

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

void SolveCG(Operator &A, Solver &P, const Vector &B, Vector &X)
{
   CGSolver cg(MPI_COMM_WORLD);
   cg.SetAbsTol(0.0);
   cg.SetRelTol(1e-12);
   cg.SetMaxIter(500);
   cg.SetPrintLevel(0);
   cg.SetOperator(A);
   cg.SetPreconditioner(P);
   X = 0.0;
   cg.Mult(B, X);
   if (Mpi::Root())
   {
      cout << "Done.\nIterations: " << cg.GetNumIterations()
           << "\nElapsed: " << tic_toc.RealTime() << endl;
   }
}
