// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
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
//   mpirun -np 4 navier_cht -r1 3 -r2 2 -np1 2 -np2 2

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
   real_t dt = 2e-2;
   real_t t_final = 250*dt;
   // fluid
   int fluid_order = 4;
   real_t fluid_kin_vis = 0.001;
   // solid
   int solid_order = 4;
   int ode_solver_type = 3;
   real_t alpha = 1.0e-2;
   real_t kappa = 0.5;
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
   int lim_meshes = 2; // should be greater than nmeshes
   Array <const char *> mesh_file_list(lim_meshes);
   Array <int> np_list(lim_meshes),
         rs_levels(lim_meshes);
   rs_levels                 = 0;
   np_list                   = 1;
   int visport               = 19916;
   bool visualization        = true;

   OptionsParser args(argc, argv);
   args.AddOption(&np_list[0], "-np1", "--np1",
                  "number of MPI ranks for mesh 1");
   args.AddOption(&np_list[1], "-np2", "--np2",
                  "number of MPI ranks for mesh 1");
   args.AddOption(&rs_levels[0], "-r1", "--refine-serial 1",
                  "Number of times to refine the mesh 1 uniformly in serial.");
   args.AddOption(&rs_levels[1], "-r2", "--refine-serial 2",
                  "Number of times to refine the mesh 2 uniformly in serial.");
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

   const int nmeshes         = 2;
   mesh_file_list[0]         = "fluid-cht.mesh";
   mesh_file_list[1]         = "solid-cht.mesh";

   // Setup MPI communicator for each mesh
   MPI_Comm *comml = new MPI_Comm;
   int color = 0;
   int npsum = 0;
   for (int i = 0; i < nmeshes; i++)
   {
      npsum += np_list[i];
      if (myid < npsum) { color = i; break; }
   }
   MPI_Comm_split(MPI_COMM_WORLD, color, myid, comml);
   int myidlocal, numproclocal;
   MPI_Comm_rank(*comml, &myidlocal);
   MPI_Comm_size(*comml, &numproclocal);

   Mesh *mesh = new Mesh(mesh_file_list[color], 1, 1);
   int dim = mesh->Dimension();
   mesh->SetCurvature(color == 0 ? schwarz.fluid_order : schwarz.solid_order);

   for (int lev = 0; lev < rs_levels[color]; lev++)
   {
      mesh->UniformRefinement();
   }


   if (color == 0 && myidlocal == 0)
   {
      std::cout << "Number of elements: " << mesh->GetNE() << std::endl;
   }

   // Setup ParMesh based on the communicator for each mesh
   ParMesh *pmesh;
   pmesh = new ParMesh(*comml, *mesh);
   delete mesh;

   // Setup pointer for FESpaces, GridFunctions, and Solvers
   H1_FECollection *fec_s            = NULL; //FECollection for solid
   ParFiniteElementSpace *fes_s      = NULL; //FESpace for solid
   ParFiniteElementSpace *adv_fes_s  = NULL; //FESpace for advection in solid
   ParGridFunction *u_gf             = NULL; //Velocity solution on both meshes
   ParGridFunction *t_gf             = NULL; //Temperature solution
   NavierSolver *flowsolver          = NULL; //Fluid solver
   ConductionOperator *coper         = NULL; //Temperature solver
   Vector t_tdof;                            //Temperature true-dof vector

   real_t t       = 0,
          dt      = schwarz.dt,
          t_final = schwarz.t_final;
   bool last_step = false;

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

      real_t cfl;
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

ConductionOperator::ConductionOperator(ParFiniteElementSpace &f, real_t al,
                                       real_t kap,
                                       VectorGridFunctionCoefficient adv_gf_c)
   : TimeDependentOperator(f.GetTrueVSize(), 0.0), fespace(f), M(NULL), K(NULL),
     T(NULL), current_dt(0.0),
     M_solver(f.GetComm()), T_solver(f.GetComm()), udir(10), z(height)
{
   const real_t rel_tol = 1e-8;

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

void ConductionOperator::ImplicitSolve(const real_t dt,
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
void vel_dbc(const Vector &x, real_t t, Vector &u)
{
   real_t xi = x(0);
   real_t yi = x(1);

   u(0) = 0.;
   u(1) = 0.;
   if (std::fabs(xi+2.5)<1.e-5) { u(0) = 0.25*yi*(3-yi)/(1.5*1.5); }
}

/// Solid data
// solid conductivity
real_t kappa_fun(const Vector &x)
{
   return x(1) <= 1.0 && std::fabs(x(0)) < 0.5 ? 5.: 1.0;
}

// initial temperature
real_t temp_init(const Vector &x)
{
   real_t t_init = 1.0;
   if (x(1) < 0.5)
   {
      t_init = 10*(std::exp(-x(1)*x(1)));
   }
   if (std::fabs(x(0)) >= 0.5)
   {
      real_t dx = std::fabs(x(0))-0.5;
      t_init *= std::exp(-10*dx*dx);
   }
   return t_init;
}
