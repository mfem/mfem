// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
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
//      ---------------------------------------------------------
//      Optimization Examples: problems built with the Problem API
//      ---------------------------------------------------------
//
// Volume-constrained Dirichlet energy on the unit square:
//
//    min_u  1/2 (grad u, grad u) - (1, u)   s.t.  (1, u) = V,  u = 0 on dOmega
//
// The objective is a BilinearFunctional, the constraint a LinearFunctional and
// the boundary condition essential dofs of the field. The solution comes from
// one Newton step on the KKT system, using the derivatives and the Hessian of
// the problem's objective and equality constraint.
//
// Sample runs:  mpirun -np 4 opt_examples
//               mpirun -np 4 opt_examples -o 3 -r 4 -V 0.1
//               mpirun -np 4 opt_examples -d debug

#include "mfem.hpp"
#include <iostream>

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();
   MPI_Comm comm = MPI_COMM_WORLD;

   int order = 2;
   int ref_levels = 3;
   real_t volume = 0.05;
   const char *device_config = "cpu";

   OptionsParser args(argc, argv);
   args.AddOption(&order, "-o", "--order", "Finite element order.");
   args.AddOption(&ref_levels, "-r", "--refine", "Uniform refinements.");
   args.AddOption(&volume, "-V", "--volume", "Target value of (1, u).");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.ParseCheck();
   Device device(device_config);

   Mesh mesh = Mesh::MakeCartesian2D(4, 4, Element::QUADRILATERAL);
   for (int l = 0; l < ref_levels; l++) { mesh.UniformRefinement(); }
   ParMesh pmesh(comm, mesh);
   mesh.Clear();

   H1_FECollection fec(order, pmesh.Dimension());
   ParFiniteElementSpace fes(&pmesh, &fec);
   const int n = fes.GetTrueVSize();
   Array<int> ess_bdr(pmesh.bdr_attributes.Max()), ess_tdofs;
   ess_bdr = 1;
   fes.GetEssentialTrueDofs(ess_bdr, ess_tdofs);

   ConstantCoefficient one(1.0);
   ParBilinearForm a(&fes);
   a.AddDomainIntegrator(new DiffusionIntegrator);
   ParLinearForm l(&fes);
   l.AddDomainIntegrator(new DomainLFIntegrator(one));
   optimization::BilinearFunctional energy(&a, &l);
   energy.Assemble();

   ParLinearForm vol_form(&fes);
   vol_form.AddDomainIntegrator(new DomainLFIntegrator(one));
   vol_form.Assemble();
   Vector vol_vec(n);
   vol_form.ParallelAssemble(vol_vec);
   optimization::LinearFunctional vol(comm, &vol_vec);

   optimization::Problem prob(comm);
   const int u_id = prob.RegisterField(n);
   prob.SetFieldEssentialDofs(u_id, ess_tdofs);
   const int energy_id = prob.RegisterFunctional(Array<int>({u_id}), &energy);
   const int vol_id = prob.RegisterFunctional(Array<int>({u_id}), &vol);
   prob.SetRole(energy_id, optimization::Objective(1.0));
   prob.SetRole(vol_id, optimization::Equality(volume));
   prob.Finalize();

   const optimization::GlobalFunctional &obj = prob.GetObjective();
   const auto equality = prob.GetEquality();
   const optimization::GlobalFunctional &eq = std::get<0>(equality);
   const Vector &eq_target = std::get<1>(equality).target;

   Vector u(n), g(n), m(n), c;
   u.UseDevice(true); g.UseDevice(true); m.UseDevice(true);
   u = 0.0;
   real_t lambda = 0.0;
   MultiVector U(u), G(g), M(m);

   // Hessian of the objective, the identity on the essential dofs. AMG uses
   // the energy matrix with the essential dofs eliminated.
   const Operator &H = *obj.ComponentHessian(U, 0);
   HypreParMatrix A_elim(*static_cast<const HypreParMatrix*>(
                            energy.ComponentHessian(U, 0)));
   delete A_elim.EliminateRowsCols(ess_tdofs);
   HypreBoomerAMG amg(A_elim);
   amg.SetPrintLevel(0);
   CGSolver cg(comm);
   cg.SetRelTol(1e-14);
   cg.SetMaxIter(500);
   cg.SetOperator(H);
   cg.SetPreconditioner(amg);

   auto kkt_residual = [&](real_t &grad_norm, real_t &constraint)
   {
      obj.ComponentDerivative(U, 0, G);
      eq.ComponentDerivative(U, 0, M);
      eq.Eval(U, c);
      c -= eq_target;
      Vector grad_L(g);
      grad_L.Add(lambda, m);
      grad_norm = sqrt(InnerProduct(comm, grad_L, grad_L));
      constraint = c(0);
   };

   real_t grad_norm0, constraint0;
   kkt_residual(grad_norm0, constraint0);

   // One Newton step on [H m; m^T 0][du; dlambda] = -[g + lambda m; c].
   Vector r(g), w1(n), w2(n);
   w1.UseDevice(true); w2.UseDevice(true);
   r.Add(lambda, m);
   w1 = 0.0; cg.Mult(r, w1);
   w2 = 0.0; cg.Mult(m, w2);
   const real_t dlambda = (c(0) - InnerProduct(comm, m, w1)) /
                          InnerProduct(comm, m, w2);
   u.Add(-1.0, w1);
   u.Add(-dlambda, w2);
   lambda += dlambda;

   real_t grad_norm, constraint;
   kkt_residual(grad_norm, constraint);
   Vector fx;
   obj.Eval(U, fx);

   const bool converged = grad_norm <= 1e-8 * grad_norm0 &&
                          std::abs(constraint) <= 1e-10 * volume;
   if (Mpi::Root())
   {
      cout << "dofs                 " << fes.GlobalTrueVSize() << '\n'
           << "objective            " << fx(0) << '\n'
           << "multiplier           " << lambda << '\n'
           << "|grad L| / |grad L0| " << grad_norm / grad_norm0 << '\n'
           << "(1, u) - V           " << constraint << '\n'
           << (converged ? "converged" : "NOT converged") << endl;
   }
   return converged ? 0 : 1;
}
