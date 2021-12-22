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
//
//            ------------------------------------------------
//            Extrapolation Miniapp: PDE-based extrapolation
//            ------------------------------------------------
//
// Compile with: make extrapolate
//
// Sample runs:
//     mpirun -np 4 extrapolate -o 3
//     mpirun -np 4 extrapolate -rs 3 -o 2 -p 1

#include <fstream>
#include <iostream>
#include "../common/mfem-common.hpp"
#include "marking.hpp"

using namespace std;
using namespace mfem;

int problem = 0;

double domainLS(const Vector &coord)
{
   // Map from [0,1] to [-1,1].
   const int dim = coord.Size();
   const double x = coord(0)*2.0 - 1.0,
                y = coord(1)*2.0 - 1.0,
                z = (dim > 2) ? coord(2)*2.0 - 1.0 : 0.0;
   switch(problem)
   {
      case 0:
      {
         // 2d circle.
         return 0.75 - sqrt(x*x + y*y + 1e-12);
      }
      case 1:
      {
         // 2d star.
         return 0.60 - sqrt(x*x + y*y + 1e-12) +
                0.25 * (y*y*y*y*y + 5.0*x*x*x*x*y - 10.0*x*x*y*y*y) /
                       pow(x*x + y*y + 1e-12, 2.5);
      }
      default: MFEM_ABORT("Bad option for --problem!"); return 0.0;
   }
}

double solution0(const Vector &coord)
{
   // Map from [0,1] to [-1,1].
   const int dim = coord.Size();
   const double x = coord(0)*2.0 - 1.0,
                y = coord(1)*2.0 - 1.0,
                z = (dim > 2) ? coord(2)*2.0 - 1.0 : 0.0;

   if (domainLS(coord) > 0.0)
   {
      return std::cos(M_PI * x) * std::sin(M_PI * y);
   }
   else { return 0.0; }
}

class LevelSetNormalGradCoeff : public VectorCoefficient
{
private:
   const ParGridFunction &ls_gf;

public:
   LevelSetNormalGradCoeff(const ParGridFunction &ls) :
      VectorCoefficient(ls.ParFESpace()->GetMesh()->Dimension()), ls_gf(ls) { }

   virtual void Eval(Vector &V, ElementTransformation &T,
                     const IntegrationPoint &ip)
   {
      Vector grad_ls(vdim), n(vdim);
      ls_gf.GetGradient(T, grad_ls);
      const double norm_grad = grad_ls.Norml2();
      V = grad_ls;
      if (norm_grad > 0.0) { V /= norm_grad; }

      // Since positive level set values correspond to the known region, we
      // transport into the opposite direction of the gradient.
      V *= -1;
   }
};

class NormalGradCoeff : public Coefficient
{
private:
   const ParGridFunction &u_gf;
   LevelSetNormalGradCoeff &n_coeff;

public:
   NormalGradCoeff(const ParGridFunction &u, LevelSetNormalGradCoeff &n) :
      u_gf(u), n_coeff(n) { }

   virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip)
   {
      const int dim = T.GetDimension();
      Vector n(dim), grad_u(dim);
      n_coeff.Eval(n, T, ip);
      u_gf.GetGradient(T, grad_u);
      return n * grad_u;
   }

};

class Extrapolator : public TimeDependentOperator
{
private:
   Array<bool> &active_zones;
   ParBilinearForm &M, &K;
   const Vector &b;
   Solver *M_prec;
   CGSolver M_solver;

public:
   Extrapolator(Array<bool> &zones,
                ParBilinearForm &Mbf, ParBilinearForm &Kbf, const Vector &rhs)
      : TimeDependentOperator(Mbf.Size()),
        active_zones(zones),
        M(Mbf), K(Kbf),
        b(rhs), M_prec(NULL), M_solver(M.ParFESpace()->GetComm()) { }

   virtual void Mult(const Vector &x, Vector &dx) const
   {
      K.BilinearForm::operator=(0.0);
      K.Assemble();

      ParFiniteElementSpace &pfes = *M.ParFESpace();
      const int NE = pfes.GetNE();
      const int nd = pfes.GetFE(0)->GetDof();

      Vector rhs(x.Size());
      HypreParMatrix *K_mat = K.ParallelAssemble(&K.SpMat());
      K_mat->Mult(x, rhs);
      rhs += b;

      Array<int> dofs(nd);
      DenseMatrix M_loc(nd);
      DenseMatrixInverse M_loc_inv(&M_loc);
      Vector rhs_loc(nd), dx_loc(nd);
      for (int k = 0; k < NE; k++)
      {
         pfes.GetElementDofs(k, dofs);

         if (active_zones[k] == false)
         {
            dx.SetSubVector(dofs, 0.0);
            continue;
         }

         rhs.GetSubVector(dofs, rhs_loc);
         M.SpMat().GetSubMatrix(dofs, dofs, M_loc);
         M_loc_inv.Factor();
         M_loc_inv.Mult(rhs_loc, dx_loc);
         dx.SetSubVector(dofs, dx_loc);
      }
   }
};

int main(int argc, char *argv[])
{
   // Initialize MPI.
   MPI_Session mpi;
   int myid = mpi.WorldRank();

   // Parse command-line options.
   const char *mesh_file = "../../data/inline-quad.mesh";
   int rs_levels = 2;
   int order = 2;
   int ode_solver_type = 2;
   bool visualization = true;
   int vis_steps = 5;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&rs_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&ode_solver_type, "-s", "--ode-solver",
                  "ODE solver: 1 - Forward Euler,\n\t"
                  "            2 - RK2 SSP,\n\t"
                  "            3 - RK3 SSP");
   args.AddOption(&problem, "-p", "--problem",
                  "0 - 2D circle,\n\t"
                  "1 - 2D star");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&vis_steps, "-vs", "--visualization-steps",
                  "Visualize every n-th timestep.");
   args.Parse();
   if (!args.Good())
   {
      if (myid == 0) { args.PrintUsage(cout); }
      return 1;
   }
   if (myid == 0) { args.PrintOptions(cout); }

   char vishost[] = "localhost";
   int  visport   = 19916, wsize = 500;
   socketstream sock_grad_u_n, sock_u;

   // Refine the mesh.
   Mesh mesh(mesh_file, 1, 1);
   for (int lev = 0; lev < rs_levels; lev++) { mesh.UniformRefinement(); }

   // MPI distribution.
   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();
   const int dim = pmesh.Dimension(), NE = pmesh.GetNE();

   FunctionCoefficient ls_coeff(domainLS), u0_coeff(solution0);

   L2_FECollection fec_L2(order, dim, BasisType::GaussLobatto);
   ParFiniteElementSpace pfes_L2(&pmesh, &fec_L2);
   ParGridFunction u(&pfes_L2);
   u.ProjectCoefficient(u0_coeff);

   // Initialize the level set.
   H1_FECollection fec(order, dim);
   ParFiniteElementSpace pfes_H1(&pmesh, &fec);
   ParGridFunction level_set(&pfes_H1);
   level_set.ProjectCoefficient(ls_coeff);
   if (visualization)
   {
      socketstream sol_sock_w;
      common::VisualizeField(sol_sock_w, vishost, visport, level_set,
                             "Domain level set", 0, 0, wsize, wsize,
                             "rRjmm********A");
      MPI_Barrier(pmesh.GetComm());
   }
   // Setup a VectorCoefficient for n = - grad_ls / |grad_ls|.
   // The sign makes it point out of the known region.
   LevelSetNormalGradCoeff ls_n_coeff(level_set);

   // Mark elements.
   Array<int> elem_marker, dofs;
   ShiftedFaceMarker marker(pmesh, pfes_L2, false);
   level_set.ExchangeFaceNbrData();
   marker.MarkElements(level_set, elem_marker);

   // Trim to the known values (only elements inside the known region).
   ParGridFunction u_known(u);
   for (int k = 0; k < NE; k++)
   {
      pfes_L2.GetElementDofs(k, dofs);
      if (elem_marker[k] != ShiftedFaceMarker::INSIDE)
      { u_known.SetSubVector(dofs, 0.0); }
   }
   if (visualization)
   {
      socketstream sol_socks;
      common::VisualizeField(sol_socks, vishost, visport, u_known,
                             "Fixed (known) u values", wsize, 0, wsize, wsize,
                             "rRjmm********A");
   }
   u = u_known;

   // Normal derivative function.
   ParGridFunction grad_u_n(&pfes_L2);
   NormalGradCoeff grad_u_n_coeff(u, ls_n_coeff);
   grad_u_n.ProjectCoefficient(grad_u_n_coeff);
   if (visualization)
   {
      socketstream sol_sock;
      common::VisualizeField(sol_sock, vishost, visport, grad_u_n,
                             "grad_u n", 2*wsize, 0, wsize, wsize,
                             "rRjmm********A");
   }

   // The active zones are where we extrapolate (where the PDE is solved).
   Array<bool> active_zones(NE);
   for (int k = 0; k < NE; k++)
   {
      // Extrapolation is done in zones that are CUT or OUTSIDE.
      active_zones[k] = (elem_marker[k] == ShiftedFaceMarker::INSIDE) ? false : true;
   }

   ParBilinearForm lhs_bf(&pfes_L2), rhs_bf(&pfes_L2);
   lhs_bf.AddDomainIntegrator(new MassIntegrator);
   const double alpha = -1.0;
   rhs_bf.AddDomainIntegrator(new ConvectionIntegrator(ls_n_coeff, alpha));
   auto trace_i = new NonconservativeDGTraceIntegrator(ls_n_coeff, alpha);
   rhs_bf.AddInteriorFaceIntegrator(trace_i);
   rhs_bf.KeepNbrBlock(true);

   lhs_bf.Assemble();
   lhs_bf.Finalize();
   rhs_bf.Assemble(0);
   rhs_bf.Finalize(0);

   Vector rhs(pfes_L2.GetVSize());
   rhs = 0.0;

   // Compute a CFL time step.
   double h_min = std::numeric_limits<double>::infinity();
   for (int k = 0; k < NE; k++)
   {
      h_min = std::min(h_min, pmesh.GetElementSize(k));
   }
   MPI_Allreduce(MPI_IN_PLACE, &h_min, 1, MPI_DOUBLE, MPI_MIN,
                 pfes_L2.GetComm());
   h_min /= order;
   // The propagation speed is 1.
   double dt = 0.25 * h_min / 1.0;

   // Time loop
   double t = 0.0;
   ODESolver *ode_solver = NULL;
   switch (ode_solver_type)
   {
      case 1: ode_solver = new ForwardEulerSolver; break;
      case 2: ode_solver = new RK2Solver(1.0); break;
      case 3: ode_solver = new RK3SSPSolver; break;
      default:
      {
         if (myid == 0)
         { cout << "Unknown ODE solver type: " << ode_solver_type << '\n'; }
         return 3;
      }
   }

   Extrapolator ext(active_zones, lhs_bf, rhs_bf, rhs);
   ode_solver->Init(ext);

   bool done = false;
   const double t_final = 0.4;
   for (int ti = 0; !done;)
   {
      double dt_real = min(dt, t_final - t);
      ode_solver->Step(grad_u_n, t, dt_real);
      ti++;

      done = (t >= t_final - 1e-8*dt);

      if (done || ti % vis_steps == 0)
      {
         if (myid == 0)
         {
            cout << "time step: " << ti << ", time: " << t << endl;
         }

         if (visualization)
         {
            common::VisualizeField(sock_grad_u_n, vishost, visport, grad_u_n,
                                   "Solution", 2*wsize, 570, wsize, wsize,
                                   "rRjmm********A");
            MPI_Barrier(pmesh.GetComm());
         }
      }
   }

   lhs_bf.Mult(grad_u_n, rhs);
   done = false;
   t = 0.0;
   for (int ti = 0; !done;)
   {
      double dt_real = min(dt, t_final - t);
      ode_solver->Step(u, t, dt_real);
      ti++;

      done = (t >= t_final - 1e-8*dt);

      if (done || ti % vis_steps == 0)
      {
         if (myid == 0)
         {
            cout << "time step: " << ti << ", time: " << t << endl;
         }

         if (visualization)
         {
            common::VisualizeField(sock_u, vishost, visport, u,
                                   "Solution", wsize, 570, wsize, wsize,
                                   "rRjmm********A");
            MPI_Barrier(pmesh.GetComm());
         }
      }
   }

   // ParaView output.
   ParaViewDataCollection dacol("ParaViewExtrapolate", &pmesh);
   dacol.SetLevelsOfDetail(order);
   dacol.RegisterField("filtered_level_set", &level_set);
   dacol.SetTime(1.0);
   dacol.SetCycle(1);
   dacol.Save();

   return 0;
}
