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
//            --------------------------------------------------
//            Overlapping Grids Miniapp: Conjugate heat transfer
//            --------------------------------------------------
//
// This example code demonstrates use of MFEM to solve different physics in
// different domains using overlapping grids: A solid block with its base at a
// fixed temperature is cooled by incoming flow. The Fluid domain models the
// entire domain, minus the solid block, and the incompressible Navier-Stokes
// equations are solved on it:
//
//                 ________________________________________
//                |                                        |
//                |             FLUID DOMAIN               |
//                |                                        |
//   -->inflow    |                ______                  | --> outflow
//     (attr=1)   |               |      |                 |     (attr=2)
//                |_______________|      |_________________|
//
// Inhomogeneous Dirichlet conditions are imposed at inflow (attr=1) and
// homogeneous Dirichlet conditions are imposed on all surface (attr=3) except
// the outflow (attr=2) which has Neumann boundary conditions for velocity.
//
// In contrast to the Fluid domain, the Thermal domain includes the solid block,
// and the advection-diffusion equation is solved on it:
//
//                     dT/dt + u.grad T = kappa \nabla^2 T
//
//                                (attr=3)
//                 ________________________________________
//                |                                        |
//                |           THERMAL DOMAIN               |
//   (attr=1)     |                kappa1                  |
//     T=0        |                ______                  |
//                |               |kappa2|                 |
//                |_______________|______|_________________|
//                   (attr=4)     (attr=2)      (attr=4)
//                                  T=10
//
// Inhomogeneous boundary conditions (T=10) are imposed on the base of the solid
// block (attr=2) and homogeneous boundary conditions are imposed at the inflow
// region (attr=1). All other surfaces have Neumann condition.
//
// The one-sided coupling between the two domains is via transfer of the
// advection velocity (u) from fluid domain to thermal domain at each time step.
//
// Sample run:
//   mpirun -np 4 navier_bifurcation -rs 2

#include "mfem.hpp"
#include "navier_solver.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;
using namespace navier;

struct schwarz_common
{
   // common
   real_t dt = 2e-4;
   real_t t_final = 10000*dt;
   // fluid
   int fluid_order = 4;
   real_t fluid_kin_vis = 0.001;
} schwarz;

// Dirichlet conditions for velocity
void vel_dbc(const Vector &x, real_t t, Vector &u);
// solid conductivity
real_t kappa_fun(const Vector &x);
// initial condition for temperature
real_t temp_init(const Vector &x);

class ConductionOperator : public TimeDependentOperator
{
protected:
   ParFiniteElementSpace &fespace;
   Array<int> ess_tdof_list; // this list remains empty for pure Neumann b.c.

   mutable ParBilinearForm *M;
   ParBilinearForm *K;

   HypreParMatrix Mmat;
   HypreParMatrix Kmat;
   HypreParMatrix *T; // T = M + dt K
   real_t current_dt;

   mutable CGSolver M_solver; // Krylov solver for inverting the mass matrix M
   HypreSmoother M_prec;      // Preconditioner for the mass matrix M

   CGSolver T_solver;    // Implicit solver for T = M + dt K
   HypreSmoother T_prec; // Preconditioner for the implicit solver

   real_t alpha, kappa, udir;

   mutable Vector z; // auxiliary vector

public:
   ConductionOperator(ParFiniteElementSpace &f, real_t alpha, real_t kappa,
                      VectorGridFunctionCoefficient adv_gf_c);

   void Mult(const Vector &u, Vector &du_dt) const override;
   /** Solve the Backward-Euler equation: k = f(u + dt*k, t), for the unknown k.
       This is the only requirement for high-order SDIRK implicit integration.*/
   void ImplicitSolve(const real_t dt, const Vector &u, Vector &k) override;

   /// Update the diffusion BilinearForm K using the given true-dof vector `u`.
   void SetParameters(VectorGridFunctionCoefficient adv_gf_c);

   virtual ~ConductionOperator();
};

void VisualizeField(socketstream &sock, const char *vishost, int visport,
                    ParGridFunction &gf, const char *title,
                    int x = 0, int y = 0, int w = 400, int h = 400,
                    bool vec = false);

int main(int argc, char *argv[])
{
   // Initialize MPI and HYPRE.
   Mpi::Init(argc, argv);
   int myid = Mpi::WorldRank();
   Hypre::Init();

   // Parse command-line options.
   int rs_levels                 = 0;
   int visport               = 19916;
   bool visualization        = true;

   OptionsParser args(argc, argv);
   args.AddOption(&rs_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh in serial.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&visport, "-p", "--send-port", "Socket for GLVis.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   if (myid == 0)
   {
      args.PrintOptions(cout);
   }

   Mesh mesh("../../data/channel2.mesh");

   int dim = mesh.Dimension();
   mesh.SetCurvature(1);

   for (int lev = 0; lev < rs_levels; lev++)
   {
      mesh.UniformRefinement();
   }

   // Setup ParMesh based on the communicator for each mesh
   auto *pmesh = new ParMesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();

   // Create the flow solver.
   NavierSolver flowsolver(pmesh, schwarz.fluid_order, schwarz.fluid_kin_vis);
   flowsolver.EnablePA(true);
   // flowsolver.EnableNI(ctx.ni);

   // Setup pointer for FESpaces, GridFunctions, and Solvers
   ParGridFunction *u_gf             = NULL; // Velocity solution

   real_t t       = 0,
          dt      = schwarz.dt,
          t_final = schwarz.t_final;
   bool last_step = false;

   {
      u_gf = flowsolver.GetCurrentVelocity();
      Vector init_vel(dim);
      init_vel = 0.;
      VectorConstantCoefficient u_excoeff(init_vel);
      u_gf->ProjectCoefficient(u_excoeff);

      // Dirichlet boundary conditions for fluid
      Array<int> attr(pmesh->bdr_attributes.Max());
      attr = 0;
      // Inlet is attribute 1.
      attr[0] = 1;
      // Walls is attribute 3.
      attr[2] = 1;
      flowsolver.AddVelDirichletBC(vel_dbc, attr);

      flowsolver.Setup(dt);
      u_gf = flowsolver.GetCurrentVelocity();
   }
   // Visualize the solution.
   char vishost[] = "localhost";
   socketstream vis_sol;
   int Ww = 350, Wh = 350; // window size
   int Wx = 10, Wy = 0; // window position
   if (visualization)
   {
      VisualizeField(vis_sol, vishost, visport, *u_gf,
                        "Velocity", Wx, Wy, Ww, Wh);
   }

   for (int step = 0; !last_step; ++step)
   {
      if (t + dt >= t_final - dt / 2)
      {
         last_step = true;
      }

      real_t cfl;
      flowsolver.Step(t, dt, step);
      cfl = flowsolver.ComputeCFL(*u_gf, dt);

      if (visualization)
      {
         VisualizeField(vis_sol, vishost, visport, *u_gf,
                        "Velocity", Wx, Wy, Ww, Wh);
      }

      if (myid == 0)
      {
         printf("%11s %11s %11s\n", "Time", "dt", "CFL");
         printf("%.5E %.5E %.5E\n", t, dt,cfl);
         fflush(stdout);
      }
   }

   flowsolver.PrintTimingData();

   delete pmesh;

   return 0;
}

void VisualizeField(socketstream &sock, const char *vishost, int visport,
                    ParGridFunction &gf, const char *title,
                    int x, int y, int w, int h, bool vec)
{
   gf.HostRead();
   ParMesh &pmesh = *gf.ParFESpace()->GetParMesh();
   MPI_Comm comm = pmesh.GetComm();

   int num_procs, myid;
   MPI_Comm_size(comm, &num_procs);
   MPI_Comm_rank(comm, &myid);

   bool newly_opened = false;
   int connection_failed;

   do
   {
      if (myid == 0)
      {
         if (!sock.is_open() || !sock)
         {
            sock.open(vishost, visport);
            sock.precision(8);
            newly_opened = true;
         }
         sock << "solution\n";
      }

      pmesh.PrintAsOne(sock);
      gf.SaveAsOne(sock);

      if (myid == 0 && newly_opened)
      {
         const char* keys = (gf.FESpace()->GetMesh()->Dimension() == 2)
                            ? "mAcRjlmm]]]]]]]]]" : "mmaaAcl";

         sock << "window_title '" << title << "'\n"
              << "window_geometry "
              << x << " " << y << " " << w << " " << h << "\n"
              << "keys " << keys;
         if ( vec ) { sock << "vvv"; }
         sock << std::endl;
      }

      if (myid == 0)
      {
         connection_failed = !sock && !newly_opened;
      }
      MPI_Bcast(&connection_failed, 1, MPI_INT, 0, comm);
   }
   while (connection_failed);
}

/// Fluid data
// Dirichlet conditions for velocity
void vel_dbc(const Vector &x, real_t t, Vector &u)
{
   real_t xi = x(0);
   real_t yi = x(1);

   u(0) = 0.;
   u(1) = 0.;
   if (std::fabs(xi)<1.e-5) { u(0) = 1.5*yi*(2-yi); }
   double epsilon = 0.0;
   double omega = 2.0*M_PI/5.0;
   double scale = epsilon*sin(omega*t)*yi*(2-yi)*(yi-1);
   u(0) += scale;
}