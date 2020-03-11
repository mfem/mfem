// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
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
//            -----------------------------------------------------
//            Joule Miniapp:  Transient Magnetics and Joule Heating
//            -----------------------------------------------------
//
// This miniapp solves a time dependent eddy current problem, resulting in Joule
// heating.
//
// This version has electrostatic potential, Phi, which is a source term in the
// EM diffusion equation. The potential itself is driven by essential BC's
//
//               Div sigma Grad Phi = 0
//               sigma E  =  Curl B/mu - sigma grad Phi
//               dB/dt = - Curl E
//               F = -k Grad T
//               c dT/dt = -Div(F) + sigma E.E,
//
// where B is the magnetic flux, E is the electric field, T is the temperature,
// F is the thermal flux, sigma is electrical conductivity, mu is the magnetic
// permeability, and alpha is the thermal diffusivity.  The geometry of the
// domain is assumed to be as follows:
//
//                              boundary attribute 3
//                            +---------------------+
//               boundary --->|                     | boundary
//               attribute 1  |                     | attribute 2
//               (driven)     +---------------------+
//
// The voltage BC condition is essential BC on attribute 1 (front) and 2 (rear)
// given by function p_bc() at bottom of this file.
//
// The E-field boundary condition specifies the essential BC (n cross E) on
// attribute 1 (front) and 2 (rear) given by function edot_bc at bottom of this
// file. The E-field can be set on attribute 3 also.
//
// The thermal boundary condition for the flux F is the natural BC on attribute 1
// (front) and 2 (rear). This means that dT/dt = 0 on the boundaries, and the
// initial T = 0.
//
// See Section 3 for how the material propertied are assigned to mesh
// attributes, this needs to be changed for different applications.
//
// See Section 5 for how the boundary conditions are assigned to mesh
// attributes, this needs to be changed for different applications.
//
// This code supports a simple version of AMR, all elements containing material
// attribute 1 are (optionally) refined.
//
// Compile with: make joule
//
// Sample runs:
//
//    mpirun -np 8 joule -m cylinder-hex.mesh -p rod
//    mpirun -np 8 joule -m cylinder-tet.mesh -sc 1 -amr 1 -p rod
//    mpirun -np 8 joule -m cylinder-hex-q2.gen -s 22 -dt 0.1 -tf 240.0 -p rod
//
// Options:
//
//    -m [string]   the mesh file name
//    -o [int]      the order of the basis
//    -rs [int]     number of times to serially refine the mesh
//    -rp [int]     number of times to refine the mesh in parallel
//    -s [int]      time integrator 1=Backward Euler, 2=SDIRK2, 3=SDIRK3,
//                  22=Midpoint, 23=SDIRK23, 34=SDIRK34
//    -tf [double]  the final time
//    -dt [double]  time step
//    -mu [double]  the magnetic permeability
//    -cnd [double] the electrical conductivity
//    -f [double]   the frequency of the applied EM BC
//    -vis [int]    GLVis -vis = true -no-vis = false
//    -vs [int]     visualization step
//    -k [string]   base file name for output file
//    -print [int]  print solution (gridfunctions) to disk  0 = no, 1 = yes
//    -amr [int]    0 = no amr, 1 = amr
//    -sc [int]     0 = no static condensation, 1 = use static condensation
//    -p [string]   specify the problem to run, "rod", "coil", or "test"
//
//
// NOTE:  Example meshes for this miniapp are the included cylinder/rod meshes:
//        cylinder-hex.mesh, cylinder-tet.mesh, cylinder-hex-q2.gen,
//        cylinder-tet-q2.gen, as well as the coil.gen mesh which can be
//        downloaded from github.com/mfem/data (its size is 21MB). Note that the
//        meshes with the "gen" extension require MFEM to be built with NetCDF.
//
// NOTE:  This code is set up to solve two example problems, 1) a straight metal
//        rod surrounded by air, 2) a metal rod surrounded by a metal coil all
//        surrounded by air. To specify problem (1) use the command line options
//        "-p rod -m cylinder-hex-q2.gen", to specify problem (2) use the
//        command line options "-p coil -m coil.gen". Problem (1) has two
//        materials and problem (2) has three materials, and the BC's are
//        different.
//
// NOTE:  We write out, optionally, grid functions for P, E, B, W, F, and
//        T. These can be visualized using "glvis -np 4 -m mesh.mesh -g E",
//        assuming we used 4 processors.

#include "joule_solver.hpp"
#include <memory>
#include <iostream>
#include <fstream>

using namespace std;
using namespace mfem;
using namespace mfem::common;
using namespace mfem::electromagnetics;

void display_banner(ostream & os);

static double mj_ = 0.0;
static double sj_ = 0.0;
static double wj_ = 0.0;

// Initialize variables used in joule_solver.cpp
int electromagnetics::SOLVER_PRINT_LEVEL = 0;
int electromagnetics::STATIC_COND        = 0;

int main(int argc, char *argv[])
{
   // 1. Initialize MPI.
   MPI_Session mpi(argc, argv);
   int myid = mpi.WorldRank();

   // print the cool banner
   if (mpi.Root()) { display_banner(cout); }

   // 2. Parse command-line options.
   const char *mesh_file = "cylinder-hex.mesh";
   int ser_ref_levels = 0;
   int par_ref_levels = 0;
   int order = 2;
   int ode_solver_type = 1;
   double t_final = 100.0;
   double dt = 0.5;
   double mu = 1.0;
   double sigma = 2.0*M_PI*10;
   double Tcapacity = 1.0;
   double Tconductivity = 0.01;
   double freq = 1.0/60.0;
   bool visualization = true;
   bool visit = true;
   int vis_steps = 1;
   int gfprint = 0;
   const char *basename = "Joule";
   int amr = 0;
   int debug = 0;
   const char *problem = "rod";

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ser_ref_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&par_ref_levels, "-rp", "--refine-parallel",
                  "Number of times to refine the mesh uniformly in parallel.");
   args.AddOption(&order, "-o", "--order",
                  "Order (degree) of the finite elements.");
   args.AddOption(&ode_solver_type, "-s", "--ode-solver",
                  "ODE solver: 1 - Backward Euler, 2 - SDIRK2, 3 - SDIRK3\n\t."
                  "\t   22 - Mid-Point, 23 - SDIRK23, 34 - SDIRK34.");
   args.AddOption(&t_final, "-tf", "--t-final",
                  "Final time; start time is 0.");
   args.AddOption(&dt, "-dt", "--time-step",
                  "Time step.");
   args.AddOption(&mu, "-mu", "--permeability",
                  "Magnetic permeability coefficient.");
   args.AddOption(&sigma, "-cnd", "--sigma",
                  "Conductivity coefficient.");
   args.AddOption(&freq, "-f", "--frequency",
                  "Frequency of oscillation.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&visit, "-visit", "--visit", "-no-visit", "--no-visit",
                  "Enable or disable VisIt visualization.");
   args.AddOption(&vis_steps, "-vs", "--visualization-steps",
                  "Visualize every n-th timestep.");
   args.AddOption(&basename, "-k", "--outputfilename",
                  "Name of the visit dump files");
   args.AddOption(&gfprint, "-print", "--print",
                  "Print results (grid functions) to disk.");
   args.AddOption(&amr, "-amr", "--amr",
                  "Enable AMR");
   args.AddOption(&STATIC_COND, "-sc", "--static-condensation",
                  "Enable static condensation");
   args.AddOption(&debug, "-debug", "--debug",
                  "Print matrices and vectors to disk");
   args.AddOption(&SOLVER_PRINT_LEVEL, "-hl", "--hypre-print-level",
                  "Hypre print level");
   args.AddOption(&problem, "-p", "--problem",
                  "Name of problem to run");
   args.Parse();
   if (!args.Good())
   {
      if (mpi.Root())
      {
         args.PrintUsage(cout);
      }
      return 1;
   }
   if (mpi.Root())
   {
      args.PrintOptions(cout);
   }

   mj_  = mu;
   sj_  = sigma;
   wj_  = 2.0*M_PI*freq;

   if (mpi.Root())
   {
      cout << "\nSkin depth sqrt(2.0/(wj*mj*sj)) = " << sqrt(2.0/(wj_*mj_*sj_))
           << "\nSkin depth sqrt(2.0*dt/(mj*sj)) = " << sqrt(2.0*dt/(mj_*sj_))
           << endl;
   }

   // 3. Here material properties are assigned to mesh attributes.  This code is
   //    not general, it is assumed the mesh has 3 regions each with a different
   //    integer attribute: 1, 2 or 3.
   //
   //       The coil problem has three regions: 1) coil, 2) air, 3) the rod.
   //       The rod problem has two regions: 1) rod, 2) air.
   //
   //    We can use the same material maps for both problems.

   std::map<int, double> sigmaMap, InvTcondMap, TcapMap, InvTcapMap;
   double sigmaAir;
   double TcondAir;
   double TcapAir;
   if (strcmp(problem,"rod")==0 || strcmp(problem,"coil")==0)
   {
      sigmaAir     = 1.0e-6 * sigma;
      TcondAir     = 1.0e6  * Tconductivity;
      TcapAir      = 1.0    * Tcapacity;
   }
   else
   {
      cerr << "Problem " << problem << " not recognized\n";
      mfem_error();
   }

   if (strcmp(problem,"rod")==0 || strcmp(problem,"coil")==0)
   {

      sigmaMap.insert(pair<int, double>(1, sigma));
      sigmaMap.insert(pair<int, double>(2, sigmaAir));
      sigmaMap.insert(pair<int, double>(3, sigmaAir));

      InvTcondMap.insert(pair<int, double>(1, 1.0/Tconductivity));
      InvTcondMap.insert(pair<int, double>(2, 1.0/TcondAir));
      InvTcondMap.insert(pair<int, double>(3, 1.0/TcondAir));

      TcapMap.insert(pair<int, double>(1, Tcapacity));
      TcapMap.insert(pair<int, double>(2, TcapAir));
      TcapMap.insert(pair<int, double>(3, TcapAir));

      InvTcapMap.insert(pair<int, double>(1, 1.0/Tcapacity));
      InvTcapMap.insert(pair<int, double>(2, 1.0/TcapAir));
      InvTcapMap.insert(pair<int, double>(3, 1.0/TcapAir));
   }
   else
   {
      cerr << "Problem " << problem << " not recognized\n";
      mfem_error();
   }

   // 4. Read the serial mesh from the given mesh file on all processors. We can
   //    handle triangular, quadrilateral, tetrahedral and hexahedral meshes
   //    with the same code.
   Mesh *mesh;
   mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   // 5. Assign the boundary conditions
   Array<int> ess_bdr(mesh->bdr_attributes.Max());
   Array<int> thermal_ess_bdr(mesh->bdr_attributes.Max());
   Array<int> poisson_ess_bdr(mesh->bdr_attributes.Max());
   if (strcmp(problem,"coil")==0)
   {
      // BEGIN CODE FOR THE COIL PROBLEM
      // For the coil in a box problem we have surfaces 1) coil end (+),
      // 2) coil end (-), 3) five sides of box, 4) side of box with coil BC

      ess_bdr = 0;
      ess_bdr[0] = 1; // boundary attribute 4 (index 3) is fixed
      ess_bdr[1] = 1; // boundary attribute 4 (index 3) is fixed
      ess_bdr[2] = 1; // boundary attribute 4 (index 3) is fixed
      ess_bdr[3] = 1; // boundary attribute 4 (index 3) is fixed

      // Same as above, but this is for the thermal operator for HDiv
      // formulation the essential BC is the flux

      thermal_ess_bdr = 0;
      thermal_ess_bdr[2] = 1; // boundary attribute 4 (index 3) is fixed

      // Same as above, but this is for the poisson eq for H1 formulation the
      // essential BC is the value of Phi

      poisson_ess_bdr = 0;
      poisson_ess_bdr[0] = 1; // boundary attribute 1 (index 0) is fixed
      poisson_ess_bdr[1] = 1; // boundary attribute 2 (index 1) is fixed
      // END CODE FOR THE COIL PROBLEM
   }
   else if (strcmp(problem,"rod")==0)
   {
      // BEGIN CODE FOR THE STRAIGHT ROD PROBLEM
      // the boundary conditions below are for the straight rod problem

      ess_bdr = 0;
      ess_bdr[0] = 1; // boundary attribute 1 (index 0) is fixed (front)
      ess_bdr[1] = 1; // boundary attribute 2 (index 1) is fixed (rear)
      ess_bdr[2] = 1; // boundary attribute 3 (index 2) is fixed (outer)

      // Same as above, but this is for the thermal operator.  For HDiv
      // formulation the essential BC is the flux, which is zero on the front
      // and sides. Note the Natural BC is T = 0 on the outer surface.

      thermal_ess_bdr = 0;
      thermal_ess_bdr[0] = 1; // boundary attribute 1 (index 0) is fixed (front)
      thermal_ess_bdr[1] = 1; // boundary attribute 2 (index 1) is fixed (rear)

      // Same as above, but this is for the poisson eq for H1 formulation the
      // essential BC is the value of Phi

      poisson_ess_bdr = 0;
      poisson_ess_bdr[0] = 1; // boundary attribute 1 (index 0) is fixed (front)
      poisson_ess_bdr[1] = 1; // boundary attribute 2 (index 1) is fixed (back)
      // END CODE FOR THE STRAIGHT ROD PROBLEM
   }
   else
   {
      cerr << "Problem " << problem << " not recognized\n";
      mfem_error();
   }

   // The following is required for mesh refinement
   mesh->EnsureNCMesh();

   // 6. Define the ODE solver used for time integration. Several implicit
   //    methods are available, including singly diagonal implicit Runge-Kutta
   //    (SDIRK).
   ODESolver *ode_solver;
   switch (ode_solver_type)
   {
      // Implicit L-stable methods
      case 1:  ode_solver = new BackwardEulerSolver; break;
      case 2:  ode_solver = new SDIRK23Solver(2); break;
      case 3:  ode_solver = new SDIRK33Solver; break;
      // Implicit A-stable methods (not L-stable)
      case 22: ode_solver = new ImplicitMidpointSolver; break;
      case 23: ode_solver = new SDIRK23Solver; break;
      case 34: ode_solver = new SDIRK34Solver; break;
      default:
         if (mpi.Root())
         {
            cout << "Unknown ODE solver type: " << ode_solver_type << '\n';
         }
         delete mesh;
         return 3;
   }

   // 7. Refine the mesh in serial to increase the resolution. In this example
   //    we do 'ser_ref_levels' of uniform refinement, where 'ser_ref_levels' is
   //    a command-line parameter.
   for (int lev = 0; lev < ser_ref_levels; lev++)
   {
      mesh->UniformRefinement();
   }

   // 8. Define a parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh further in parallel to increase the resolution. Once the
   //    parallel mesh is defined, the serial mesh can be deleted.
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   for (int lev = 0; lev < par_ref_levels; lev++)
   {
      pmesh->UniformRefinement();
   }
   // Make sure tet-only meshes are marked for local refinement.
   pmesh->Finalize(true);

   // 9. Apply non-uniform non-conforming mesh refinement to the mesh.  The
   //    whole metal region is refined once, before the start of the time loop,
   //    i.e. this is not based on any error estimator.
   if (amr == 1)
   {
      Array<int> ref_list;
      int numElems = pmesh->GetNE();
      for (int ielem = 0; ielem < numElems; ielem++)
      {
         int thisAtt = pmesh->GetAttribute(ielem);
         if (thisAtt == 1)
         {
            ref_list.Append(ielem);
         }
      }

      pmesh->GeneralRefinement(ref_list);

      ref_list.DeleteAll();
   }

   // 10. Reorient the mesh. Must be done after refinement but before definition
   //     of higher order Nedelec spaces
   pmesh->ReorientTetMesh();

   // 11. Rebalance the mesh. Since the mesh was adaptively refined in a
   //     non-uniform way it will be computationally unbalanced.
   if (pmesh->Nonconforming())
   {
      pmesh->Rebalance();
   }

   // 12. Define the parallel finite element spaces. We use:
   //
   //     H(curl) for electric field,
   //     H(div) for magnetic flux,
   //     H(div) for thermal flux,
   //     H(grad)/H1 for electrostatic potential,
   //     L2 for temperature

   // L2 contains discontinuous "cell-center" finite elements, type 2 is
   // "positive"
   L2_FECollection L2FEC(order-1, dim);

   // ND contains Nedelec "edge-centered" vector finite elements with continuous
   // tangential component.
   ND_FECollection HCurlFEC(order, dim);

   // RT contains Raviart-Thomas "face-centered" vector finite elements with
   // continuous normal component.
   RT_FECollection HDivFEC(order-1, dim);

   // H1 contains continuous "node-centered" Lagrange finite elements.
   H1_FECollection HGradFEC(order, dim);

   ParFiniteElementSpace    L2FESpace(pmesh, &L2FEC);
   ParFiniteElementSpace HCurlFESpace(pmesh, &HCurlFEC);
   ParFiniteElementSpace  HDivFESpace(pmesh, &HDivFEC);
   ParFiniteElementSpace  HGradFESpace(pmesh, &HGradFEC);

   // The terminology is TrueVSize is the unique (non-redundant) number of dofs
   HYPRE_Int glob_size_l2 = L2FESpace.GlobalTrueVSize();
   HYPRE_Int glob_size_nd = HCurlFESpace.GlobalTrueVSize();
   HYPRE_Int glob_size_rt = HDivFESpace.GlobalTrueVSize();
   HYPRE_Int glob_size_h1 = HGradFESpace.GlobalTrueVSize();

   if (mpi.Root())
   {
      cout << "Number of Temperature Flux unknowns:  " << glob_size_rt << endl;
      cout << "Number of Temperature unknowns:       " << glob_size_l2 << endl;
      cout << "Number of Electric Field unknowns:    " << glob_size_nd << endl;
      cout << "Number of Magnetic Field unknowns:    " << glob_size_rt << endl;
      cout << "Number of Electrostatic unknowns:     " << glob_size_h1 << endl;
   }

   int Vsize_l2 = L2FESpace.GetVSize();
   int Vsize_nd = HCurlFESpace.GetVSize();
   int Vsize_rt = HDivFESpace.GetVSize();
   int Vsize_h1 = HGradFESpace.GetVSize();

   // the big BlockVector stores the fields as
   //    0 Temperature
   //    1 Temperature Flux
   //    2 P field
   //    3 E field
   //    4 B field
   //    5 Joule Heating

   Array<int> true_offset(7);
   true_offset[0] = 0;
   true_offset[1] = true_offset[0] + Vsize_l2;
   true_offset[2] = true_offset[1] + Vsize_rt;
   true_offset[3] = true_offset[2] + Vsize_h1;
   true_offset[4] = true_offset[3] + Vsize_nd;
   true_offset[5] = true_offset[4] + Vsize_rt;
   true_offset[6] = true_offset[5] + Vsize_l2;

   // The BlockVector is a large contiguous chunk of memory for storing required
   // data for the hypre vectors, in this case: the temperature L2, the T-flux
   // HDiv, the E-field HCurl, and the B-field HDiv, and scalar potential P.
   BlockVector F(true_offset);

   // grid functions E, B, T, F, P, and w which is the Joule heating
   ParGridFunction E_gf, B_gf, T_gf, F_gf, w_gf, P_gf;
   T_gf.MakeRef(&L2FESpace,F,   true_offset[0]);
   F_gf.MakeRef(&HDivFESpace,F, true_offset[1]);
   P_gf.MakeRef(&HGradFESpace,F,true_offset[2]);
   E_gf.MakeRef(&HCurlFESpace,F,true_offset[3]);
   B_gf.MakeRef(&HDivFESpace,F, true_offset[4]);
   w_gf.MakeRef(&L2FESpace,F,   true_offset[5]);

   // 13. Get the boundary conditions, set up the exact solution grid functions
   //     These VectorCoefficients have an Eval function.  Note that e_exact and
   //     b_exact in this case are exact analytical solutions, taking a 3-vector
   //     point as input and returning a 3-vector field
   VectorFunctionCoefficient E_exact(3, e_exact);
   VectorFunctionCoefficient B_exact(3, b_exact);
   FunctionCoefficient T_exact(t_exact);
   E_exact.SetTime(0.0);
   B_exact.SetTime(0.0);

   // 14. Initialize the Diffusion operator, the GLVis visualization and print
   //     the initial energies.
   MagneticDiffusionEOperator oper(true_offset[6], L2FESpace, HCurlFESpace,
                                   HDivFESpace, HGradFESpace,
                                   ess_bdr, thermal_ess_bdr, poisson_ess_bdr,
                                   mu, sigmaMap, TcapMap, InvTcapMap,
                                   InvTcondMap);

   // This function initializes all the fields to zero or some provided IC
   oper.Init(F);

   socketstream vis_T, vis_E, vis_B, vis_w, vis_P;
   char vishost[] = "localhost";
   int  visport   = 19916;
   if (visualization)
   {
      // Make sure all ranks have sent their 'v' solution before initiating
      // another set of GLVis connections (one from each rank):
      MPI_Barrier(pmesh->GetComm());

      vis_T.precision(8);
      vis_E.precision(8);
      vis_B.precision(8);
      vis_P.precision(8);
      vis_w.precision(8);

      int Wx = 0, Wy = 0; // window position
      int Ww = 350, Wh = 350; // window size
      int offx = Ww+10, offy = Wh+45; // window offsets

      VisualizeField(vis_P, vishost, visport,
                     P_gf, "Electric Potential (Phi)", Wx, Wy, Ww, Wh);
      Wx += offx;

      VisualizeField(vis_E, vishost, visport,
                     E_gf, "Electric Field (E)", Wx, Wy, Ww, Wh);
      Wx += offx;

      VisualizeField(vis_B, vishost, visport,
                     B_gf, "Magnetic Field (B)", Wx, Wy, Ww, Wh);
      Wx = 0;
      Wy += offy;

      VisualizeField(vis_w, vishost, visport,
                     w_gf, "Joule Heating", Wx, Wy, Ww, Wh);

      Wx += offx;

      VisualizeField(vis_T, vishost, visport,
                     T_gf, "Temperature", Wx, Wy, Ww, Wh);
   }
   // VisIt visualization
   VisItDataCollection visit_dc(basename, pmesh);
   if ( visit )
   {
      visit_dc.RegisterField("E", &E_gf);
      visit_dc.RegisterField("B", &B_gf);
      visit_dc.RegisterField("T", &T_gf);
      visit_dc.RegisterField("w", &w_gf);
      visit_dc.RegisterField("Phi", &P_gf);
      visit_dc.RegisterField("F", &F_gf);

      visit_dc.SetCycle(0);
      visit_dc.SetTime(0.0);
      visit_dc.Save();
   }

   E_exact.SetTime(0.0);
   B_exact.SetTime(0.0);

   // 15. Perform time-integration (looping over the time iterations, ti, with a
   //     time-step dt). The object oper is the MagneticDiffusionOperator which
   //     has a Mult() method and an ImplicitSolve() method which are used by
   //     the time integrators.
   ode_solver->Init(oper);
   double t = 0.0;

   bool last_step = false;
   for (int ti = 1; !last_step; ti++)
   {
      if (t + dt >= t_final - dt/2)
      {
         last_step = true;
      }

      // F is the vector of dofs, t is the current time, and dt is the time step
      // to advance.
      ode_solver->Step(F, t, dt);

      if (debug == 1)
      {
         oper.Debug(basename,t);
      }

      if (gfprint == 1)
      {
         ostringstream T_name, E_name, B_name, F_name, w_name, P_name, mesh_name;
         T_name << basename << "_" << setfill('0') << setw(6) << t << "_"
                << "T." << setfill('0') << setw(6) << myid;
         E_name << basename << "_" << setfill('0') << setw(6) << t << "_"
                << "E." << setfill('0') << setw(6) << myid;
         B_name << basename << "_" << setfill('0') << setw(6) << t << "_"
                << "B." << setfill('0') << setw(6) << myid;
         F_name << basename << "_" << setfill('0') << setw(6) << t << "_"
                << "F." << setfill('0') << setw(6) << myid;
         w_name << basename << "_" << setfill('0') << setw(6) << t << "_"
                << "w." << setfill('0') << setw(6) << myid;
         P_name << basename << "_" << setfill('0') << setw(6) << t << "_"
                << "P." << setfill('0') << setw(6) << myid;
         mesh_name << basename << "_" << setfill('0') << setw(6) << t << "_"
                   << "mesh." << setfill('0') << setw(6) << myid;

         ofstream mesh_ofs(mesh_name.str().c_str());
         mesh_ofs.precision(8);
         pmesh->Print(mesh_ofs);
         mesh_ofs.close();

         ofstream T_ofs(T_name.str().c_str());
         T_ofs.precision(8);
         T_gf.Save(T_ofs);
         T_ofs.close();

         ofstream E_ofs(E_name.str().c_str());
         E_ofs.precision(8);
         E_gf.Save(E_ofs);
         E_ofs.close();

         ofstream B_ofs(B_name.str().c_str());
         B_ofs.precision(8);
         B_gf.Save(B_ofs);
         B_ofs.close();

         ofstream F_ofs(F_name.str().c_str());
         F_ofs.precision(8);
         F_gf.Save(B_ofs);
         F_ofs.close();

         ofstream P_ofs(P_name.str().c_str());
         P_ofs.precision(8);
         P_gf.Save(P_ofs);
         P_ofs.close();

         ofstream w_ofs(w_name.str().c_str());
         w_ofs.precision(8);
         w_gf.Save(w_ofs);
         w_ofs.close();
      }

      if (last_step || (ti % vis_steps) == 0)
      {
         double el = oper.ElectricLosses(E_gf);

         if (mpi.Root())
         {
            cout << fixed;
            cout << "step " << setw(6) << ti << ",\tt = " << setw(6)
                 << setprecision(3) << t
                 << ",\tdot(E, J) = " << setprecision(8) << el << endl;
         }

         // Make sure all ranks have sent their 'v' solution before initiating
         // another set of GLVis connections (one from each rank):
         MPI_Barrier(pmesh->GetComm());

         if (visualization)
         {
            int Wx = 0, Wy = 0; // window position
            int Ww = 350, Wh = 350; // window size
            int offx = Ww+10, offy = Wh+45; // window offsets

            VisualizeField(vis_P, vishost, visport,
                           P_gf, "Electric Potential (Phi)", Wx, Wy, Ww, Wh);
            Wx += offx;

            VisualizeField(vis_E, vishost, visport,
                           E_gf, "Electric Field (E)", Wx, Wy, Ww, Wh);
            Wx += offx;

            VisualizeField(vis_B, vishost, visport,
                           B_gf, "Magnetic Field (B)", Wx, Wy, Ww, Wh);

            Wx = 0;
            Wy += offy;

            VisualizeField(vis_w, vishost, visport,
                           w_gf, "Joule Heating", Wx, Wy, Ww, Wh);

            Wx += offx;

            VisualizeField(vis_T, vishost, visport,
                           T_gf, "Temperature", Wx, Wy, Ww, Wh);
         }

         if (visit)
         {
            visit_dc.SetCycle(ti);
            visit_dc.SetTime(t);
            visit_dc.Save();
         }
      }
   }
   if (visualization)
   {
      vis_T.close();
      vis_E.close();
      vis_B.close();
      vis_w.close();
      vis_P.close();
   }

   // 16. Free the used memory.
   delete ode_solver;
   delete pmesh;

   return 0;
}

namespace mfem
{

namespace electromagnetics
{

void edot_bc(const Vector &x, Vector &E)
{
   E = 0.0;
}

void e_exact(const Vector &x, double t, Vector &E)
{
   E[0] = 0.0;
   E[1] = 0.0;
   E[2] = 0.0;
}

void b_exact(const Vector &x, double t, Vector &B)
{
   B[0] = 0.0;
   B[1] = 0.0;
   B[2] = 0.0;
}

double t_exact(Vector &x)
{
   double T = 0.0;
   return T;
}

double p_bc(const Vector &x, double t)
{
   // the value
   double T;
   if (x[2] < 0.0)
   {
      T = 1.0;
   }
   else
   {
      T = -1.0;
   }

   return T*cos(wj_ * t);
}

} // namespace electromagnetics

} // namespace mfem

void display_banner(ostream & os)
{
   os << "     ____.            .__             " << endl
      << "    |    | ____  __ __|  |   ____     " << endl
      << "    |    |/  _ \\|  |  \\  | _/ __ \\ " << endl
      << "/\\__|    (  <_> )  |  /  |_\\  ___/  " << endl
      << "\\________|\\____/|____/|____/\\___  >" << endl
      << "                                \\/   " << endl
      << flush;
}
