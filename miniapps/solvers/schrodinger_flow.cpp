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
//             ---------------------------------------------
//             Incompressible Schrödinger Flow (ISF) Miniapp
//             ---------------------------------------------
//
// This miniapp introduces the Incompressible Schrödinger Flow (ISF) method,
// a novel approach for simulating inviscid fluid dynamics by solving the
// linear Schrödinger equation, leveraging the hydrodynamical analogy to
// quantum mechanics proposed by Madelung in 1926. ISF offers a simple and
// efficient framework, particularly effective for capturing vortex dynamics.
//
// Compile with: make schrodinger_flow
//
// Sample runs:
//  * schrodinger_flow --leapfrog
//  * schrodinger_flow --leapfrog -o 4 -nx 10 -ny 10 -lr1 0.36 -lr2 0.26
//  * schrodinger_flow --leapfrog -o 4 -nx 10 -ny 10 -lr1 0.3 -lr2 0.2
//  * schrodinger_flow --leapfrog -o 4 -nx 10 -ny 10 -lr1 0.24 -lr2 0.12 -ms 256
//  * schrodinger_flow --jet -vd 0 -hbar 5e-2
//
// Device sample runs:
//  * schrodinger_flow -d hip --leapfrog -nx 128 -ny 128 -hbar 5e-2

#include "schrodinger_flow.hpp"

namespace mfem
{

// Kernel definitions for serial computations
using Kernels = SchrodingerBaseKernels<
                Mesh,
                FiniteElementSpace,
                ComplexGridFunction,
                GridFunction,
                BilinearForm,
                MixedBilinearForm,
                LinearForm>;

// Crank-Nicolson solver for time stepping
using CrankNicolsonSolver = CrankNicolsonBaseSolver<
                            FiniteElementSpace,
                            SesquilinearForm,
                            ComplexGridFunction>;

// Factory functions for serial mesh and solvers
static auto SetMesh = [](Mesh &mesh) { return Mesh(mesh);};
static auto SetOrthoSolver = []() { return OrthoSolver(); };
static auto SetCGSolver = []() { return CGSolver();};
static auto SetGMRESSolver = []() { return GMRESSolver(); };

// Main solver class for Schrödinger flow
struct SchrodingerSolver : public Kernels
{
   // Crank-Nicolson solver for time evolution
   struct CrankNicolsonSchrodingerTimeSolver: public CrankNicolsonSolver
   {
      CrankNicolsonSchrodingerTimeSolver(FiniteElementSpace &fes,
                                         real_t hbar, real_t dt,
                                         real_t rtol, real_t atol, int maxiter,
                                         int print_level):
         CrankNicolsonBaseSolver(fes, hbar, dt, SetGMRESSolver,
                                 rtol, atol, maxiter, print_level) { }

      void Mult(ComplexGridFunction &psi) override
      {
         R_op->Mult(psi, z), Cm1.Mult(z, psi);
         MFEM_VERIFY(Cm1.GetConverged(), "Crank Nicolson solver failed");
      }
   } cn1, cn2;

public:
   SchrodingerSolver(Options &config):
      Kernels(config, SetMesh, SetOrthoSolver, SetCGSolver),
      cn1(h1_fes, hbar, dt, rtol, atol, max_iters, print_level),
      cn2(h1_fes, hbar, dt, rtol, atol, max_iters, print_level) { }

   void Step() { cn1.Mult(psi1), cn2.Mult(psi2); }

   void GradPsi()
   {
      const auto Grad_nd = [&](GridFunction &in_h1, GridFunction &out_nd)
      {
         grad_nd_op->Mult(in_h1, nd);
         Mm1_nd.Mult(nd, out_nd);
         assert(Mm1_nd.GetConverged());
      };
      const auto x_dot_Mm1 = [&](GridFunction &x, GridFunction &y)
      {
         nd_dot_x_h1_op->Mult(x, h1), Mm1_h1.Mult(h1, y);
      };
      const auto y_dot_Mm1 = [&](GridFunction &x, GridFunction &y)
      {
         nd_dot_y_h1_op->Mult(x, h1), Mm1_h1.Mult(h1, y);
      };
      const auto z_dot_Mm1 = [&](GridFunction &x, GridFunction &y)
      {
         nd_dot_z_h1_op->Mult(x, h1), Mm1_h1.Mult(h1, y);
      };
      Kernels::GradPsi(Grad_nd, x_dot_Mm1, y_dot_Mm1, z_dot_Mm1);
   }

   void VelocityOneForm(GridFunction &ux, GridFunction &uy, GridFunction &uz)
   {
      GradPsi(), GradPsiVelocity(hbar, ux, uy, uz);
   }

   void ComputeDivU()
   {
      diff_h1_op->Mult(psi1.real(), h1), Mm1_h1.Mult(h1, delta_psi1.real());
      diff_h1_op->Mult(psi2.real(), h1), Mm1_h1.Mult(h1, delta_psi2.real());
      diff_h1_op->Mult(psi1.imag(), h1), Mm1_h1.Mult(h1, delta_psi1.imag());
      diff_h1_op->Mult(psi2.imag(), h1), Mm1_h1.Mult(h1, delta_psi2.imag());
      Kernels::ComputeDivU();
   }

   void PoissonSolve()
   {
      rhs.Assemble();
      Km1_h1.Mult(rhs, q);
      MFEM_VERIFY(Km1_h1.GetConverged(), "Km1_h1 solver did not converge");
   }

   void PressureProject() { ComputeDivU(), PoissonSolve(), GaugeTransform(); }
};

using IncompressibleFlow =
   IncompressibleBaseFlow<SchrodingerSolver, GridFunction>;

using FlowVis =
   VisualizerBase<Mesh, GridFunction, FiniteElementSpace, SchrodingerSolver, IncompressibleFlow>;

} // namespace mfem

int main(int argc, char *argv[])
{
   Options opt(argc, argv);
   Device device(opt.device);
   device.Print();

   SchrodingerSolver schrodinger(opt);
   IncompressibleFlow flow(opt, schrodinger);
   FlowVis vis(opt, schrodinger, flow, []() { return ""; });

   for (int ti = 1; ti <= opt.max_steps; ti++)
   {
      const real_t t = ti * opt.dt;
      const bool vis_steps = (ti % opt.vis_steps) == 0;
      if (vis_steps) { mfem::out << "#" << ti << std::endl; }
      flow.Step(t);
      if (opt.visualization && vis_steps) { vis.GLVis(), vis.Save(ti, t); }
   }

   return EXIT_SUCCESS;
}
