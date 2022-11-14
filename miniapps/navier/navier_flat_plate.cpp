// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
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
// Incompressible Navier Stokes Flat Plate stagnation flow example
//
// Solve for the steady Flat Plate stagnation flow at Re =  defined by
// P = 101325Pa (1 atm)
// u = 68.058 m/s (Mach 0.2)
// mu = 1.7894x10^-5 Kg /m * s (for atmospheric)
// mu = 1.7894x10^-3 Kg /m * s (scaled by 100 to remain laminar)
// Re_x = 4.66x10^4
// rho = 1.225 km/m^3
//
// The 2D problem domain is set up like this:
//
// 
// Uniform inflow passes over a flat plate of 1m is with its leading edge 
// located at 0.1m into the domain. The Fluid domain models the entire domain, 
// minus the flat plate, and the incompressible Navier-Stokes equations are 
// solved on it:
//
//                                Atmosphere
//                 ________________________________________
//                |                                        |
//                |             FLUID DOMAIN               |
//                |                                        |
//   -->inflow    |                                        | --> outflow
//     (attr=1)   |                                        |     (attr=2)
//                |_____------------------------------_____|
//                               Flat Plate
//
// Uniform Dirichlet velocity conditions are imposed at inflow (attr=1) and
// homogeneous Dirichlet conditions are imposed on all surface (attr=3) except
// the outflow (attr=2) which has Neumann boundary conditions for velocity.

#include "mfem.hpp"
#include "navier_solver.hpp"
#include <fstream>
#include <iostream>

using namspace std;
using namespace mfem;
using namespace navier;

struct s_NavierContext
{
   int ser_ref_levels = 1; //Serial Refinement Levels
   int order = 6; // Finite Element function space order
   double kinvis = 1.0 / 40.0; //Kinematic viscocity
   double t_final = 10 * 0.001; //Final time of simulation
   double dt = 0.001; //Time-step size
   double reference_pressure = 0.0; //Reference Pressure
   bool pa = true;
   bool ni = false;
   bool visualization = false;
   bool paraview = true;
   bool checkres = false;
} ctx; // Might also be called Schwartz in some of the functions

// Dirichlet conditions for velocity
void vel_dbc(const Vector &x, double t, Vector &u);

// 
void VisualizeField(socketstream &sock, const char *vishost, int visport,
                    ParGridFunction &gf, const char *title,
                    int x = 0, int y = 0, int w = 400, int h = 400,
                    bool vec = false);

int main(int argc, char *argv[])
{
   // Initialize MPI and HYPRE.
   Mpi::Init(argc, argv);
   Hypre::Init();

   OptionsParser args(argc, argv);
   args.AddOption(&ctx.ser_ref_levels,
                  "-rs",
                  "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&ctx.order,
                  "-o",
                  "--order",
                  "Order (degree) of the finite elements.");
   args.AddOption(&ctx.dt, "-dt", "--time-step", "Time step.");
   args.AddOption(&ctx.t_final, "-tf", "--final-time", "Final time.");
   args.AddOption(&ctx.pa,
                  "-pa",
                  "--enable-pa",
                  "-no-pa",
                  "--disable-pa",
                  "Enable partial assembly.");
   args.AddOption(&ctx.ni,
                  "-ni",
                  "--enable-ni",
                  "-no-ni",
                  "--disable-ni",
                  "Enable numerical integration rules.");
   args.AddOption(&ctx.visualization,
                  "-vis",
                  "--visualization",
                  "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&ctx.paraview,
                  "-pv",
                  "--paraview",
                  "-no-pv",
                  "--no-paraview",
                  "Enable or disable Paraview file creation.");
   args.AddOption(
      &ctx.checkres,
      "-cr",
      "--checkresult",
      "-no-cr",
      "--no-checkresult",
      "Enable or disable checking of the result. Returns -1 on failure.");
   args.Parse();

   if (!args.Good())
   {
      if (Mpi::Root())
      {
         args.PrintUsage(mfem::out);
      }
      return 1;
   }
   if (Mpi::Root())
   {
      args.PrintOptions(mfem::out);
   }

   Mesh *mesh = new Mesh("flat-plate.mesh"); // Need to name our mesh flat-plate.mesh 
   int dim = mesh->Dimension();

   //Mesh refinement
   for (int i = 0; i < ctx.ser_ref_levels; ++i)
   {
      mesh.UniformRefinement();
   }

   if (Mpi::Root())
   {
      std::cout << "Number of elements: " << mesh.GetNE() << std::endl;
   }

   auto *pmesh = new ParMesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();


   // Create the flow solver.
   NavierSolver flowsolver(pmesh, ctx.order, ctx.kinvis);
   flowsolver.EnablePA(ctx.pa);
   flowsolver.EnableNI(ctx.ni);

   // Set the initial condition.
   ParGridFunction *u_ic = flowsolver.GetCurrentVelocity(); // u_ic is u_gf in the rest of the code. 
   VectorFunctionCoefficient u_excoeff(pmesh->Dimension(), vel_dbc);
   u_ic->ProjectCoefficient(u_excoeff);

   FunctionCoefficient p_excoeff(pres_kovasznay);

   // Add Dirichlet boundary conditions to velocity space restricted to
   // selected attributes on the mesh.
   Array<int> attr(pmesh->bdr_attributes.Max());
   attr = 1;
   flowsolver.AddVelDirichletBC(vel_kovasznay, attr);
   // ===============================================================
   // Setup flow solver on mesh for fluid
   if (color == 0)
   {
      flowsolver = new NavierSolver(pmesh, schwarz.fluid_order,
                                    schwarz.fluid_kin_vis);
      flowsolver->EnablePA(true);
      u_gf = flowsolver->GetCurrentVelocity();
      Vector init_vel(dim);
      init_vel = 0.;
      VectorConstantCoefficient u_excoeff(init_vel);
      u_gf->ProjectCoefficient(u_excoeff);

      // Dirichlet boundary conditions for fluid
      Array<int> attr(pmesh->bdr_attributes.Max());
      // Inlet is attribute 1.
      attr[0] = 1;
      // Walls is attribute 3.
      attr[2] = 1;
      flowsolver->AddVelDirichletBC(vel_dbc, attr);

      flowsolver->Setup(dt);
      u_gf = flowsolver->GetCurrentVelocity();
   }
   // ===============================================================

   // Setup temperature solver for mesh on solid
   ODESolver *ode_solver = NULL;
   Vector vxyz;
   if (color == 1)
   {
      switch (schwarz.ode_solver_type)
      {
         // Implicit L-stable methods
         case 1:  ode_solver = new BackwardEulerSolver; break;
         case 2:  ode_solver = new SDIRK23Solver(2); break;
         case 3:  ode_solver = new SDIRK33Solver; break;
         // Explicit methods
         case 11: ode_solver = new ForwardEulerSolver; break;
         case 12: ode_solver = new RK2Solver(0.5); break; // midpoint method
         case 13: ode_solver = new RK3SSPSolver; break;
         case 14: ode_solver = new RK4Solver; break;
         case 15: ode_solver = new GeneralizedAlphaSolver(0.5); break;
         // Implicit A-stable methods (not L-stable)
         case 22: ode_solver = new ImplicitMidpointSolver; break;
         case 23: ode_solver = new SDIRK23Solver; break;
         case 24: ode_solver = new SDIRK34Solver; break;
         default:
            std::cout << "Unknown ODE solver type: " << schwarz.ode_solver_type << '\n';
            delete mesh;
            return 3;
      }
      fec_s = new H1_FECollection(schwarz.solid_order, dim);
      fes_s = new ParFiniteElementSpace(pmesh, fec_s);
      adv_fes_s = new ParFiniteElementSpace(pmesh, fec_s, 2);
      t_gf = new ParGridFunction(fes_s);
      u_gf = new ParGridFunction(adv_fes_s);

      FunctionCoefficient t_0(temp_init);
      t_gf->ProjectCoefficient(t_0);
      t_gf->SetTrueVector();
      t_gf->GetTrueDofs(t_tdof);

      // Create a list of points for the interior where the gridfunction will
      // be interpolate from the fluid mesh
      vxyz = *pmesh->GetNodes();
   }

   // Setup FindPointsGSLIB. Note: we set it up with MPI_COMM_WORLD to enable
   // communication between ParMesh for solid and fluid zones.
   OversetFindPointsGSLIB finder(MPI_COMM_WORLD);
   finder.Setup(*pmesh, color);

   // Tag each point to be found with the same id as the mesh
   Array<unsigned int> color_array;
   color_array.SetSize(vxyz.Size());
   for (int i = 0; i < color_array.Size(); i++)
   {
      color_array[i] = (unsigned int)color;
   }
   Vector interp_vals(vxyz.Size());

   // Interpolate velocity solution on both meshes. Since the velocity solution
   // does not exist on the temperature mesh, it just passes in a dummy
   // gridfunction that is not used in any way on the fluid mesh.
   finder.Interpolate(vxyz, color_array, *u_gf, interp_vals);

   // Transfer the interpolated solution to solid mesh and setup a coefficient.
   VectorGridFunctionCoefficient adv_gf_c;
   if (color == 1)
   {
      *u_gf = interp_vals;
      adv_gf_c.SetGridFunction(u_gf);
      coper = new ConductionOperator(*fes_s, schwarz.alpha, schwarz.kappa,
                                     adv_gf_c);
      coper->SetParameters(adv_gf_c);
   }

   // Visualize the solution.
   char vishost[] = "localhost";
   int visport = 19916;
   socketstream vis_sol;
   int Ww = 350, Wh = 350; // window size
   int Wx = color*Ww+10, Wy = 0; // window position
   if (visualization)
   {
      if (color == 0)
      {
         VisualizeField(vis_sol, vishost, visport, *u_gf,
                        "Velocity", Wx, Wy, Ww, Wh);
      }
      else
      {
         VisualizeField(vis_sol, vishost, visport, *t_gf,
                        "Temperature", Wx, Wy, Ww, Wh);
      }
   }

   if (ode_solver) { ode_solver->Init(*coper); }

   for (int step = 0; !last_step; ++step)
   {
      if (t + dt >= t_final - dt / 2)
      {
         last_step = true;
      }

      double cfl;
      if (flowsolver)
      {
         flowsolver->Step(t, dt, step);
         cfl = flowsolver->ComputeCFL(*u_gf, dt);
      }
      if (ode_solver)
      {
         ode_solver->Step(t_tdof, t, dt);
         t_gf->SetFromTrueDofs(t_tdof);
      }
      finder.Interpolate(vxyz, color_array, *u_gf, interp_vals);
      if (color == 1)
      {
         *u_gf = interp_vals;
         adv_gf_c.SetGridFunction(u_gf);
         coper->SetParameters(adv_gf_c);
      }

      if (visualization)
      {
         if (color == 0)
         {
            VisualizeField(vis_sol, vishost, visport, *u_gf,
                           "Velocity", Wx, Wy, Ww, Wh);
         }
         else
         {
            VisualizeField(vis_sol, vishost, visport, *t_gf,
                           "Temperature", Wx, Wy, Ww, Wh);
         }
      }

      if (color == 0 && myidlocal == 0)
      {
         printf("%11s %11s %11s\n", "Time", "dt", "CFL");
         printf("%.5E %.5E %.5E\n", t, dt,cfl);
         fflush(stdout);
      }
   }

   if (flowsolver) { flowsolver->PrintTimingData(); }

   finder.FreeData();
   delete coper;
   delete t_gf;
   if (color == 1) { delete u_gf; }
   delete adv_fes_s;
   delete fes_s;
   delete fec_s;
   delete ode_solver;
   delete flowsolver;
   delete pmesh;
   delete comml;

   return 0;
}

ConductionOperator::ConductionOperator(ParFiniteElementSpace &f, double al,
                                       double kap,
                                       VectorGridFunctionCoefficient adv_gf_c)
   : TimeDependentOperator(f.GetTrueVSize(), 0.0), fespace(f), M(NULL), K(NULL),
     T(NULL), current_dt(0.0),
     M_solver(f.GetComm()), T_solver(f.GetComm()), udir(10), z(height)
{
   const double rel_tol = 1e-8;

   Array<int> ess_bdr(f.GetParMesh()->bdr_attributes.Max());
   // Dirichlet boundary condition on inlet and isothermal section of wall.
   ess_bdr = 0;
   ess_bdr[0] = 1; // inlet
   ess_bdr[1] = 1; // homogeneous isothermal section of bottom wall
   ess_bdr[2] = 0; // top wall
   ess_bdr[3] = 0; // inhomogeneous isothermal section of bottom wall
   f.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

   M = new ParBilinearForm(&fespace);
   M->AddDomainIntegrator(new MassIntegrator());
   M->Assemble(0); // keep sparsity pattern of M and K the same
   M->FormSystemMatrix(ess_tdof_list, Mmat);

   M_solver.iterative_mode = false;
   M_solver.SetRelTol(rel_tol);
   M_solver.SetAbsTol(0.0);
   M_solver.SetMaxIter(100);
   M_solver.SetPrintLevel(0);
   M_prec.SetType(HypreSmoother::Jacobi);
   M_solver.SetPreconditioner(M_prec);
   M_solver.SetOperator(Mmat);

   alpha = al;
   kappa = kap;

   T_solver.iterative_mode = false;
   T_solver.SetRelTol(rel_tol);
   T_solver.SetAbsTol(0.0);
   T_solver.SetMaxIter(100);
   T_solver.SetPrintLevel(0);
   T_solver.SetPreconditioner(T_prec);

   SetParameters(adv_gf_c);
}

void ConductionOperator::Mult(const Vector &u, Vector &du_dt) const
{
   // Compute:
   //    du_dt = M^{-1}*-K(u)
   // for du_dt

   Kmat.Mult(u, z);
   z.Neg(); // z = -z
   K->EliminateVDofsInRHS(ess_tdof_list, u, z);

   M_solver.Mult(z, du_dt);
   du_dt.Print();
   du_dt.SetSubVector(ess_tdof_list, 0.0);
}

void ConductionOperator::ImplicitSolve(const double dt,
                                       const Vector &u, Vector &du_dt)
{
   // Solve the equation:
   //    du_dt = M^{-1}*[-K(u + dt*du_dt)]
   // for du_dt
   if (!T)
   {
      T = Add(1.0, Mmat, dt, Kmat);
      current_dt = dt;
      T_solver.SetOperator(*T);
   }
   MFEM_VERIFY(dt == current_dt, ""); // SDIRK methods use the same dt
   Kmat.Mult(u, z);
   z.Neg();
   K->EliminateVDofsInRHS(ess_tdof_list, u, z);

   T_solver.Mult(z, du_dt);
   du_dt.SetSubVector(ess_tdof_list, 0.0);
}

void ConductionOperator::SetParameters(VectorGridFunctionCoefficient adv_gf_c)
{
   ParGridFunction u_alpha_gf(&fespace);
   FunctionCoefficient kapfuncoef(kappa_fun);
   u_alpha_gf.ProjectCoefficient(kapfuncoef);

   delete K;
   K = new ParBilinearForm(&fespace);

   GridFunctionCoefficient u_coeff(&u_alpha_gf);

   K->AddDomainIntegrator(new DiffusionIntegrator(u_coeff));
   K->AddDomainIntegrator(new MixedDirectionalDerivativeIntegrator(adv_gf_c));
   K->Assemble(0); // keep sparsity pattern of M and K the same
   K->FormSystemMatrix(ess_tdof_list, Kmat);
   delete T;
   T = NULL; // re-compute T on the next ImplicitSolve
}

ConductionOperator::~ConductionOperator()
{
   delete T;
   delete M;
   delete K;
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
                            ? "mAcRjlmm" : "mmaaAcl";

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
void vel_dbc(const Vector &x, double t, Vector &u)
{
   double xi = x(0);
   double yi = x(1);

   u(0) = 0.;
   u(1) = 0.;
   if (std::fabs(xi+2.5)<1.e-5) { u(0) = 0.25*yi*(3-yi)/(1.5*1.5); }
}

/// Solid data
// solid conductivity
double kappa_fun(const Vector &x)
{
   return x(1) <= 1.0 && std::fabs(x(0)) < 0.5 ? 5.: 1.0;
}

// initial temperature
double temp_init(const Vector &x)
{
   double t_init = 1.0;
   if (x(1) < 0.5)
   {
      t_init = 10*(std::exp(-x(1)*x(1)));
   }
   if (std::fabs(x(0)) >= 0.5)
   {
      double dx = std::fabs(x(0))-0.5;
      t_init *= std::exp(-10*dx*dx);
   }
   return t_init;
}
