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
//     mpirun -np 4 extrapolate
//

#include <fstream>
#include <iostream>
#include "../common/mfem-common.hpp"
#include "marking.hpp"

using namespace std;
using namespace mfem;

double domainLS(const Vector &coord)
{
   // [0, 1] to [-pi, pi]
   const double x = (coord(0)*2.0 - 1.0) * M_PI,
                y = (coord(1)*2.0 - 1.0) * M_PI,
                z = (coord.Size() > 2) ? (coord(2)*2.0 - 1.0) * M_PI : 0.0;

   const double radius = 2.0;
   double rv = x*x + y*y + z*z;
   rv = rv > 0.0 ? sqrt(rv) : 0.0;
   return rv - radius;
}

double solution0(const Vector &coord)
{
   // [0, 1] to [-pi, pi]
   const double x = (coord(0)*2.0 - 1.0) * M_PI,
                y = (coord(1)*2.0 - 1.0) * M_PI,
                z = (coord.Size() > 2) ? (coord(2)*2.0 - 1.0) * M_PI : 0.0;

   if (domainLS(coord) < 0.0)
   {
      return std::cos(x) * std::sin(y);
   }
   else { return 0.0; }
}

class LevelSetNormalCoeff : public VectorCoefficient
{
private:
   const ParGridFunction &ls_gf;

public:
   LevelSetNormalCoeff(const ParGridFunction &ls) :
      VectorCoefficient(ls.ParFESpace()->GetMesh()->Dimension()), ls_gf(ls) { }

   virtual void Eval(Vector &V, ElementTransformation &T,
                     const IntegrationPoint &ip)
   {
      const double psi = ls_gf.GetValue(T, ip);

      if (psi > 0.0) { V = 0.0; return; }

      Vector grad_psi(vdim), n(vdim);
      ls_gf.GetGradient(T, grad_psi);
      const double norm_grad = grad_psi.Norml2();
      n = grad_psi;
      if (norm_grad > 0.0) { n /= norm_grad; }
      V = n;
   }
};

class Extrapolator : public TimeDependentOperator
{
private:
   OperatorHandle M, K;
   const Vector &b;
   Solver *M_prec;
   CGSolver M_solver;

   mutable Vector z;

public:
   Extrapolator(ParBilinearForm &M_, ParBilinearForm &K_, const Vector &b_);

   virtual void Mult(const Vector &x, Vector &y) const;

   virtual ~Extrapolator();
};


int main(int argc, char *argv[])
{
   // Initialize MPI.
   MPI_Session mpi;
   int num_procs = mpi.WorldSize();
   int myid = mpi.WorldRank();

   // Parse command-line options.
   const char *mesh_file = "../../data/inline-quad.mesh";
   int rs_levels = 2;
   int order = 1;
   int ode_solver_type = 2;
   double dt = 0.01;
   bool visualization = true;

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
   args.AddOption(&dt, "-dt", "--time-step", "Time step.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();
   if (!args.Good())
   {
      if (myid == 0) { args.PrintUsage(cout); }
      return 1;
   }
   if (myid == 0) { args.PrintOptions(cout); }

   // Refine the mesh.
   Mesh mesh(mesh_file, 1, 1);
   const int dim = mesh.Dimension();
   for (int lev = 0; lev < rs_levels; lev++) { mesh.UniformRefinement(); }

   // MPI distribution.
   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();

   FunctionCoefficient ls_coeff(domainLS), u0_coeff(solution0);

   L2_FECollection fec_L2(order, dim);
   ParFiniteElementSpace pfes_L2(&pmesh, &fec_L2);
   ParGridFunction u(&pfes_L2);
   u.ProjectCoefficient(u0_coeff);
   HypreParVector *U = u.GetTrueDofs();

   H1_FECollection fec(order, dim);
   ParFiniteElementSpace pfes(&pmesh, &fec);
   ParGridFunction level_set(&pfes);
   level_set.ProjectCoefficient(ls_coeff);

   LevelSetNormalCoeff ls_n_coeff(level_set);

   // Mark elements.
   Array<int> elem_marker;
   ShiftedFaceMarker marker(pmesh, pfes_L2, true);
   level_set.ExchangeFaceNbrData();
   marker.MarkElements(level_set, elem_marker);
   L2_FECollection fec_mark(0, dim);
   ParFiniteElementSpace pfes_mark(&pmesh, &fec_mark);
   ParGridFunction gf_mark(&pfes_mark);
   for (int i = 0; i < gf_mark.Size(); i++) { gf_mark(i) = elem_marker[i]; }
   if (visualization)
   {
      socketstream sol_sock_mark;
      int size = 500;
      char vishost[] = "localhost";
      int  visport   = 19916;
      common::VisualizeField(sol_sock_mark, vishost, visport, gf_mark,
                             "Marking", 0, size, size, size,
                             "rRjmm********A");
   }

   ParBilinearForm lhs(&pfes_L2), rhs_bf(&pfes_L2);
   lhs.AddDomainIntegrator(new MassIntegrator);
   const double alpha = -1.0;
   rhs_bf.AddDomainIntegrator(new ConvectionIntegrator(ls_n_coeff, alpha));
   auto trace_i = new NonconservativeDGTraceIntegrator(ls_n_coeff, alpha);
   rhs_bf.AddInteriorFaceIntegrator(trace_i);

   lhs.Assemble();
   lhs.Finalize();
   rhs_bf.Assemble(0);
   rhs_bf.Finalize(0);

   Vector rhs(pfes_L2.GetVSize());
   rhs_bf.Mult(u, rhs);

   // Time loop
   double t = 0.0;
   ODESolver *ode_solver = NULL;
   switch (ode_solver_type)
   {
      case 1: ode_solver = new ForwardEulerSolver; break;
      case 2: ode_solver = new RK2Solver(1.0); break;
      case 3: ode_solver = new RK3SSPSolver; break;
      default:
         if (myid == 0)
         {
            cout << "Unknown ODE solver type: " << ode_solver_type << '\n';
         }
         return 3;
   }

   bool done = true;
   const double t_final = 1.0;
   for (int ti = 0; !done;)
   {
      double dt_real = min(dt, t_final - t);
      ode_solver->Step(*U, t, dt_real);
      ti++;

      done = (t >= t_final - 1e-8*dt);

      if (done)
      {
         if (mpi.Root())
         {
            cout << "time step: " << ti << ", time: " << t << endl;
         }

         u = *U;
      }
   }

   // Send the solution by socket to a GLVis server.
   if (visualization)
   {
      int size = 500;
      char vishost[] = "localhost";
      int  visport   = 19916;

      socketstream sol_sock_w;
      common::VisualizeField(sol_sock_w, vishost, visport, level_set,
                             "Domain Level Set", 0, 0, size, size);
      MPI_Barrier(pmesh.GetComm());

      socketstream sol_sock_ds;
      common::VisualizeField(sol_sock_ds, vishost, visport, u,
                             "Solution", size, 0, size, size,
                             "rRjmm********A");
      MPI_Barrier(pmesh.GetComm());
   }

   // ParaView output.
   ParaViewDataCollection dacol("ParaViewExtrapolate", &pmesh);
   dacol.SetLevelsOfDetail(order);
   dacol.RegisterField("filtered_level_set", &level_set);
   dacol.SetTime(1.0);
   dacol.SetCycle(1);
   dacol.Save();

   delete U;

   return 0;
}
