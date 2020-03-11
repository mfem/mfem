//                                MFEM modified from Example 10 and 16
//
// Compile with: make imMHDp
//
// Description:  It solves a time dependent resistive MHD problem 
//               There three versions:
//               1. explicit scheme
//               2. implicit scheme using a very simple linear preconditioner
//               3. implicit scheme using physcis-based preconditioner
// Author: QT

#include "mfem.hpp"
#include "myCoefficient.hpp"
#include "myIntegrator.hpp"
#include "imScale.hpp"
#include "PCSolver.hpp"
#include "InitialConditions.hpp"
#include <memory>
#include <iostream>
#include <fstream>

#ifndef MFEM_USE_PETSC
#error This example requires that MFEM is built with MFEM_USE_PETSC=YES
#endif

int main(int argc, char *argv[])
{
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   //++++Parse command-line options.
   const char *mesh_file = "./Meshes/xperiodic-square.mesh";
   int ser_ref_levels = 2;
   int par_ref_levels = 0;
   int order = 2;
   int ode_solver_type = 3;
   double t_final = 5.0;
   double dt = 0.0001;
   double visc = 1e-3;
   double resi = 1e-3;
   bool visit = false;
   bool use_petsc = false;
   bool use_factory = false;
   const char *petscrc_file = "";
   beta = 0.001; 
   Lx=3.0;
   lambda=5.0;

   int vis_steps = 1;


   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ser_ref_levels, "-rs", "--refine",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&par_ref_levels, "-rp", "--refineP",
                  "Number of times to refine the mesh uniformly in parallel.");
   args.AddOption(&order, "-o", "--order",
                  "Order (degree) of the finite elements.");
   args.AddOption(&ode_solver_type, "-s", "--ode-solver",
                  "ODE solver: 1 - Backward Euler, 2 - Brailovskaya,\n\t"
                  "            3 - L-stable SDIRK23, 4 - L-stable SDIRK33,\n\t"
                  "            22 - Implicit Midpoint, 23 - SDIRK23, 24 - SDIRK34.");
   args.AddOption(&t_final, "-tf", "--t-final",
                  "Final time; start time is 0.");
   args.AddOption(&dt, "-dt", "--time-step",
                  "Time step.");
   args.AddOption(&visc, "-visc", "--viscosity",
                  "Viscosity coefficient.");
   args.AddOption(&resi, "-resi", "--resistivity",
                  "Resistivity coefficient.");
   args.AddOption(&vis_steps, "-vs", "--visualization-steps",
                  "Visualize every n-th timestep.");
   args.AddOption(&use_petsc, "-usepetsc", "--usepetsc", "-no-petsc",
                  "--no-petsc",
                  "Use or not PETSc to solve the nonlinear system.");
   args.AddOption(&use_factory, "-shell", "--shell", "-no-shell",
                  "--no-shell",
                  "Use user-defined preconditioner factory (PCSHELL).");
   args.AddOption(&petscrc_file, "-petscopts", "--petscopts",
                  "PetscOptions file to use.");

   args.Parse();
   if (!args.Good())
   {
      if (myid == 0)
      {
         args.PrintUsage(cout);
      }
      MPI_Finalize();
      return 1;
   }

   lambda=.5/M_PI;
   resiG=resi;
  
   if (myid == 0) args.PrintOptions(cout);

   if (use_petsc)
   {
      MFEMInitializePetsc(NULL,NULL,petscrc_file,NULL);
   }

   //+++++Read the mesh from the given mesh file.    
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   //++++Define the ODE solver used for time integration. Several implicit
   //    singly diagonal implicit Runge-Kutta (SDIRK) methods, as well as
   //    backward Euler methods are available.
   PCSolver *ode_solver=NULL;
   ODESolver *ode_solver2=NULL;
   bool explicitSolve=false;
   switch (ode_solver_type)
   {
      //Explicit methods (first-order Predictor-Corrector)
      case 2: ode_solver = new PCSolver; explicitSolve = true; break;
      //Implict L-stable methods 
      case 1: ode_solver2 = new BackwardEulerSolver; break;
      case 3: ode_solver2 = new SDIRK23Solver(2); break;
      case 4: ode_solver2 = new SDIRK33Solver; break;
      // Implicit A-stable methods (not L-stable)
      case 12: ode_solver2 = new ImplicitMidpointSolver; break;
      case 13: ode_solver2 = new SDIRK23Solver; break;
      case 14: ode_solver2 = new SDIRK34Solver; break;
     default:
         if (myid == 0) cout << "Unknown ODE solver type: " << ode_solver_type << '\n';
         delete mesh;
         MPI_Finalize();
         return 3;
   }

   //++++++Refine the mesh to increase the resolution.    
   for (int lev = 0; lev < ser_ref_levels; lev++)
   {
      mesh->UniformRefinement();
   }

   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   for (int lev = 0; lev < par_ref_levels; lev++)
   {
      pmesh->UniformRefinement();
   }

   //+++++Define the vector finite element spaces representing  [Psi, Phi, w]
   // in block vector bv, with offsets given by the fe_offset array.
   // All my fespace is 1D but the problem is multi-dimensional
   H1_FECollection fe_coll(order, dim);
   ParFiniteElementSpace fespace(pmesh, &fe_coll); 

   HYPRE_Int global_size = fespace.GlobalTrueVSize();
   if (myid == 0)
   {
      cout << "Number of total scalar unknowns: " << global_size << endl;
   }

   int fe_size = fespace.TrueVSize();
   Array<int> fe_offset(4);
   fe_offset[0] = 0;
   fe_offset[1] = fe_size;
   fe_offset[2] = 2*fe_size;
   fe_offset[3] = 3*fe_size;

   BlockVector vx(fe_offset);
   ParGridFunction psi, phi, w;
   phi.MakeTRef(&fespace, vx, fe_offset[0]);
   psi.MakeTRef(&fespace, vx, fe_offset[1]);
     w.MakeTRef(&fespace, vx, fe_offset[2]);

   //+++++Set the initial conditions, and the boundary conditions
   FunctionCoefficient phiInit(InitialPhi);
   phi.ProjectCoefficient(phiInit);
   phi.SetTrueVector();

   FunctionCoefficient psiInit3(InitialPsi3);
   psi.ProjectCoefficient(psiInit3);
   psi.SetTrueVector();

   FunctionCoefficient wInit(InitialW);
   w.ProjectCoefficient(wInit);
   w.SetTrueVector();
   
   //this step is necessary to make sure unknows are updated!
   phi.SetFromTrueVector(); psi.SetFromTrueVector(); w.SetFromTrueVector();

   //++++++this is a periodic boundary condition in x and Direchlet in y 
   Array<int> ess_bdr(fespace.GetMesh()->bdr_attributes.Max());
   ess_bdr = 0;
   ess_bdr[0] = 1;  //set attribute 1 to Direchlet boundary fixed
   if(ess_bdr.Size()!=1 || false)
   {
    if (myid==0) cout <<"ess_bdr size should be 1 but it is "<<ess_bdr.Size()<<endl;
    delete ode_solver;
    delete ode_solver2;
    delete mesh;
    delete pmesh;
    MPI_Finalize();
    return 2;
   }

   //++++Initialize the MHD operator, the GLVis visualization    
   ResistiveMHDOperator oper(fespace, ess_bdr, visc, resi, use_petsc, use_factory);
   FunctionCoefficient e0(E0rhs3);
   oper.SetRHSEfield(e0);

   //set initial J
   FunctionCoefficient jInit3(InitialJ3);
   oper.SetInitialJ(jInit3);

   double t = 0.0;
   oper.SetTime(t);
   ode_solver2->Init(oper);

   MPI_Barrier(MPI_COMM_WORLD); 
   double start = MPI_Wtime();

   //++++Perform time-integration (looping over the time iterations, ti, with a
   //    time-step dt).
   bool last_step = false;
   for (int ti = 1; !last_step; ti++)
   {
      //ignore the first time step
      if (ti==2)
      {
          MPI_Barrier(MPI_COMM_WORLD); 
          start = MPI_Wtime();
      }

      ode_solver2->Step(vx, t, dt);
      last_step = (t >= t_final - 1e-8*dt);

      if (last_step || (ti % vis_steps) == 0)
      {
         if (myid==0) cout << "step " << ti << ", t = " << t <<endl;
      }
   }
   MPI_Barrier(MPI_COMM_WORLD); 
   double end = MPI_Wtime();

   if (myid == 0) 
       cout <<"######Runtime = "<<end-start<<" ######"<<endl;

   //++++++Save the solutions.
   {
      phi.SetFromTrueVector(); psi.SetFromTrueVector(); w.SetFromTrueVector();

      ostringstream mesh_name, phi_name, psi_name, w_name,j_name;
      mesh_name << "mesh." << setfill('0') << setw(6) << myid;
      phi_name << "sol_phi." << setfill('0') << setw(6) << myid;
      psi_name << "sol_psi." << setfill('0') << setw(6) << myid;
      w_name << "sol_omega." << setfill('0') << setw(6) << myid;

      ofstream omesh(mesh_name.str().c_str());
      omesh.precision(8);
      pmesh->Print(omesh);

      ofstream osol(phi_name.str().c_str());
      osol.precision(8);
      phi.Save(osol);

      ofstream osol3(psi_name.str().c_str());
      osol3.precision(8);
      psi.Save(osol3);

      ofstream osol4(w_name.str().c_str());
      osol4.precision(8);
      w.Save(osol4);
   }

   //+++++Free the used memory.
   delete ode_solver;
   delete ode_solver2;
   delete pmesh;

   oper.DestroyHypre();

   if (use_petsc) { MFEMFinalizePetsc(); }

   MPI_Finalize();

   return 0;
}



