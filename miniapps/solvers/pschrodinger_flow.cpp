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
//                 -------------------------------------
//                 Incompressible Schrödinger Flow (ISF)
//                 -------------------------------------
//
// This miniapp introduces the Incompressible Schrödinger Flow (ISF) method,
// an approach for simulating inviscid fluid dynamics by solving the linear
// Schrödinger equation, leveraging the hydrodynamical analogy to quantum
// mechanics proposed by Madelung in 1926. ISF offers a simple and efficient
// framework, particularly effective for capturing vortex dynamics.
//
// Compile with: make pschrodinger_flow
//
// Sample runs:
//    mpirun -np 4 pschrodinger_flow --leapfrog
//    mpirun -np 4 pschrodinger_flow --leapfrog -nx 128 -ny 128 -hbar 5e-2
//    mpirun -np 4 pschrodinger_flow --leapfrog -o 2 -nx 32 -ny 32 -sx 8 -sy 8
//    mpirun -np 4 pschrodinger_flow --leapfrog -o 3 -nx 32 -ny 32
//    mpirun -np 4 pschrodinger_flow --leapfrog -o 4 -nx 16 -ny 16 -lr1 0.32 -lr2 0.16 -dt 0.1
//    mpirun -np 4 pschrodinger_flow --jet -vd 0 -hbar 5e-2
//
// Device sample runs:
//    mpirun -np 4 pschrodinger_flow -d debug --leapfrog

#include "schrodinger_flow.hpp"

namespace mfem
{

// Kernel definitions for parallel computations
using Kernels = SchrodingerBaseKernels<
                ParMesh,
                ParFiniteElementSpace,
                ParComplexGridFunction,
                ParGridFunction,
                ParBilinearForm,
                ParMixedBilinearForm,
                ParLinearForm>;

// Crank-Nicolson solver for time stepping
using CrankNicolsonSolver = CrankNicolsonBaseSolver<
                            ParFiniteElementSpace,
                            ParSesquilinearForm,
                            ParComplexGridFunction>;

// Parallel mesh and solvers functions
static auto SetParMesh = [](Mesh &mesh) { return ParMesh(MPI_COMM_WORLD, mesh); };
static auto SetOrthoSolver = []() { return OrthoSolver(MPI_COMM_WORLD); };
static auto SetCGSolver = []() { return CGSolver(MPI_COMM_WORLD); };
static auto SetGMRESSolver = []() { return GMRESSolver(MPI_COMM_WORLD); };

// Main solver class for Schrödinger flow
struct SchrodingerSolver : public Kernels
{
   // Crank-Nicolson solver for time evolution
   struct CrankNicolsonSchrodingerTimeSolver: public CrankNicolsonSolver
   {
      Vector PSI, Z;
      CrankNicolsonSchrodingerTimeSolver(ParFiniteElementSpace &fes,
                                         real_t hbar, real_t dt,
                                         real_t rtol, real_t atol, int maxiter,
                                         int print_level):
         CrankNicolsonBaseSolver(fes, hbar, dt, SetGMRESSolver,
                                 rtol, atol, maxiter, print_level),
         PSI(2*fes.GetTrueVSize()),
         Z(2*fes.GetTrueVSize()) { }

      void Mult(ParComplexGridFunction &psi) override
      {
         psi.ParallelProject(PSI);
         R_op->Mult(PSI, Z), Cm1.Mult(Z, PSI);
         psi.Distribute(PSI);
         MFEM_VERIFY(Cm1.GetConverged(), "Crank Nicolson solver failed");
      }
   } cn1, cn2;
   Vector B;

public:
   SchrodingerSolver(Options &config):
      Kernels(config, SetParMesh, SetOrthoSolver, SetCGSolver),
      cn1(h1_fes, hbar, dt, rtol, atol, max_iters, print_level),
      cn2(h1_fes, hbar, dt, rtol, atol, max_iters, print_level),
      B(Km1_h1.Width())
   {
      B.UseDevice(true);
      serial_mesh.Clear();
      nodes.SetTrueVector();
      nodes.SetFromTrueVector();
   }

   void Step() { cn1.Mult(psi1), cn2.Mult(psi2); }

   void GradPsi()
   {
      const auto Grad_nd = [&](ParGridFunction &in_h1, ParGridFunction &out_nd)
      {
         in_h1.SetTrueVector(), in_h1.SetFromTrueVector();
         grad_nd_op->Mult(in_h1.GetTrueVector(), nd.GetTrueVector());
         Mm1_nd.Mult(nd.GetTrueVector(), out_nd.GetTrueVector());
         out_nd.SetFromTrueVector();
      };
      const auto x_dot_Mm1 = [&](ParGridFunction &x, ParGridFunction &y)
      {
         x.SetTrueVector(), x.SetFromTrueVector();
         nd_dot_x_h1_op->Mult(x.GetTrueVector(), h1.GetTrueVector());
         Mm1_h1.Mult(h1.GetTrueVector(), y.GetTrueVector());
         y.SetFromTrueVector();
      };
      const auto y_dot_Mm1 = [&](ParGridFunction &x, ParGridFunction &y)
      {
         x.SetTrueVector(), x.SetFromTrueVector();
         nd_dot_y_h1_op->Mult(x.GetTrueVector(), h1.GetTrueVector());
         Mm1_h1.Mult(h1.GetTrueVector(), y.GetTrueVector());
         y.SetFromTrueVector();
      };
      const auto z_dot_Mm1 = [&](ParGridFunction &x, ParGridFunction &y)
      {
         x.SetTrueVector(), x.SetFromTrueVector();
         nd_dot_z_h1_op->Mult(x.GetTrueVector(), h1.GetTrueVector());
         Mm1_h1.Mult(h1.GetTrueVector(), y.GetTrueVector());
         y.SetFromTrueVector();
      };
      Kernels::GradPsi(Grad_nd, x_dot_Mm1, y_dot_Mm1, z_dot_Mm1);
   }

   void VelocityOneForm(ParGridFunction &ux, ParGridFunction &uy,
                        ParGridFunction &uz)
   {
      GradPsi();
      GradPsiVelocity(hbar, ux, uy, uz);
   }

   void ComputeDivU()
   {
      const auto diff_Mm1 = [&](ParGridFunction &x, ParGridFunction &y)
      {
         x.SetTrueVector(), x.SetFromTrueVector();
         diff_h1_op->Mult(x.GetTrueVector(), h1.GetTrueVector());
         Mm1_h1.Mult(h1.GetTrueVector(), y.GetTrueVector());
         y.SetFromTrueVector();
      };
      diff_Mm1(psi1.real(), delta_psi1.real());
      diff_Mm1(psi2.real(), delta_psi2.real());
      diff_Mm1(psi1.imag(), delta_psi1.imag());
      diff_Mm1(psi2.imag(), delta_psi2.imag());
      Kernels::ComputeDivU();
   }

   void PoissonSolve()
   {
      rhs.Assemble();
      rhs.ParallelAssemble(B);
      Km1_h1.Mult(B, q.GetTrueVector());
      q.SetFromTrueVector();
      MFEM_VERIFY(Km1_h1.GetConverged(), "Km1_h1 solver did not converge");
   }

   void PressureProject() { ComputeDivU(); PoissonSolve(); GaugeTransform(); }
};

using IncompressibleFlow =
   IncompressibleBaseFlow<SchrodingerSolver, ParGridFunction>;

using FlowVis =
   VisualizerBase<ParMesh, ParGridFunction, ParFiniteElementSpace, SchrodingerSolver, IncompressibleFlow>;

} // namespace mfem

int main(int argc, char *argv[])
{
   Mpi::Init();
   Hypre::Init();

   Options opt(argc, argv);
   Device device(opt.device);
   if (Mpi::Root()) { device.Print(); }

   SchrodingerSolver schrodinger(opt);
   IncompressibleFlow flow(opt, schrodinger);
   const auto glvis_prefix = []()
   {
      std::ostringstream os;
      os << "parallel " << Mpi::WorldSize() << " " << Mpi::WorldRank() << "\n";
      return os.str();
   };
   FlowVis vis(opt, schrodinger, flow, glvis_prefix);

   for (int ti = 1; ti <= opt.max_steps; ti++)
   {
      const real_t t = ti * opt.dt;
      const bool vis_steps = (ti % opt.vis_steps) == 0;
      if (Mpi::Root() && vis_steps) { mfem::out << "#" << ti << std::endl; }
      flow.Step(t);
      if (opt.visualization && vis_steps) { vis.GLVis(); vis.Save(ti, t); }
   }

   return EXIT_SUCCESS;
}
