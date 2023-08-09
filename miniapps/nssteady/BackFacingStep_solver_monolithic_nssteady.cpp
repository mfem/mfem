// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
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
// Flow around cylinder in 2D/3D
//
// The problem domain is set up like this
//                
//                                  no slip
//              |\    + ---------------------------------- +
// Parabolic -->  |   |                                    |   
//  inflow      |/    + ------ +                           |  Traction free (outflow)
//                             |                           |
//                             + ------------------------- +                           
//                                      no slip
//
// Mesh attributes for 2D/3D are:
// inflow = 1, outflow = 2, wall = 3
//
// Run with:
// mpirun -np 4 ./BackFacingStep_solver_monolithic_nssteady -d 2 -u 1.0 -kv 0.01 -pic -1 -rs 0 -rp 0  -abs 1e-6 -g 1.0 -a 1.0 -p --verbose -o Output-Folder
//
 
// Include mfem and I/O
#include "mfem.hpp"
#include <fstream>
#include <iostream>

// Include for defining function definition
#include <math.h>

// Include steady ns miniapp
#include "snavier_monolithic.hpp"


using namespace mfem;

// Forward declarations of functions
double Umax = 1.0;    // maximum inflow velocity (parabolic profile)
double L    = 1.0;    // length of inflow section
void   V_inflow(const Vector &X, Vector &v);
void   noSlip(const Vector &X, Vector &v);

// Test
int main(int argc, char *argv[])
{
   //
   /// 1. Initialize MPI and HYPRE.
   //
   int nprocs, myrank;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
   Hypre::Init();


   //
   /// 2. Parse command-line options. 
   //
   int dim = 2;              // dimension of problem

   int porder = 1;           // fe
   int vorder = 2;

   int ser_ref_levels = 0;   // mesh
   int par_ref_levels = 0;

   double kin_vis = 0.01;    // physical params

   double  rel_tol = 1e-7;   // solvers
   double  abs_tol = 1e-6;
   int    tot_iter = 1000;
   int print_level = 0;

   double alpha = 1.;          // steady-state scheme
   double gamma = 1.;          // relaxation
   int nPicardIterations = -1; // number of picard iterations (before switching to newton)

   bool paraview = false;     // postprocessing (paraview)
   bool visit    = false;     // postprocessing (VISit)
   
   bool verbose = false;
   const char *outFolder = "./";

   mfem::OptionsParser args(argc, argv);
   args.AddOption(&dim,
                     "-d",
                     "--dimension",
                     "Dimension of problem (run 2D flow around cylinder, or 3D flow around sphere).");
   args.AddOption(&ser_ref_levels,
                     "-rs",
                     "--refine-serial",
                     "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&par_ref_levels,
                     "-rp",
                     "--refine-parallel",
                     "Number of times to refine the mesh uniformly in parallel.");
   args.AddOption(&Umax, "-u", "--max-velocity",
                  "Max velocity");   
   args.AddOption(&kin_vis, "-kv", "--kinematic-viscosity",
                  "Kinematic viscosity");
   args.AddOption(&nPicardIterations,
                  "-pic",
                  "--picard-iterations",
                  "Number of Picard iterations before switching to newton (-1) Picard solver, (0) Newton solver, (>=0) Mixed.");
   args.AddOption(&vorder, "-ov", "--order_vel",
                     "Finite element order for velocity (polynomial degree) or -1 for"
                     " isoparametric space.");
   args.AddOption(&porder, "-op", "--order_pres",
                     "Finite element order for pressure(polynomial degree) or -1 for"
                     " isoparametric space.");
   args.AddOption(&rel_tol,
                  "-rel",
                  "--relative-tolerance",
                  "Relative tolerance for the Newton solve.");
   args.AddOption(&abs_tol,
                  "-abs",
                  "--absolute-tolerance",
                  "Absolute tolerance for the Outer solve.");
   args.AddOption(&tot_iter,
                  "-it",
                  "--linear-iterations",
                  "Maximum iterations for the linear solve.");
   args.AddOption(&outFolder,
                  "-o",
                  "--output-folder",
                  "Output folder.");
   args.AddOption(&paraview, "-p", "--paraview", "-no-p",
                  "--no-paraview",
                  "Enable Paraview output.");
   args.AddOption(&visit, "-v", "--visit", "-no-v",
                  "--no-visit",
                  "Enable VisIt output.");    
   args.AddOption(&verbose, "-vb", "--verbose", "-no-verb",
                  "--no-verbose",
                  "Verbosity level of code");    
   args.AddOption(&print_level,
                  "-pl",
                  "--print-level",
                  "Print level.");     
   args.AddOption(&gamma,
                     "-g",
                     "--gamma",
                     "Relaxation parameter");
   args.AddOption(&alpha,
                     "-a",
                     "--alpha",
                     "Parameter controlling linearization");


   args.Parse();
   if (!args.Good())
   {
      if (myrank == 0)
      {
         args.PrintUsage(mfem::out);
      }
      MPI_Finalize();
      return 1;
   }
   if (myrank == 0)
   {
       args.PrintOptions(mfem::out);
   }



   //
   /// 3. Read the (serial) mesh from the given mesh file on all processors.
   //

   Mesh mesh;

   switch (dim)
   {
   case 2:
      {mesh = Mesh::LoadFromFile("./Mesh/back_facing_step_2D.msh"); break;}
   case 3:
      {mesh = Mesh::LoadFromFile("./Mesh/back_facing_step_3D.msh"); break;}   
   default:
      break;
   }


   for (int l = 0; l < ser_ref_levels; l++)
   {
       mesh.UniformRefinement();
   }



   //
   /// 4. Define a parallel mesh by a partitioning of the serial mesh. 
   // Refine this mesh further in parallel to increase the resolution. Once the
   // parallel mesh is defined, the serial mesh can be deleted.
   //
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();
   {
       for (int l = 0; l < par_ref_levels; l++)
       {
          pmesh->UniformRefinement();
       }
   }



   //
   /// 5. Create solver
   // 
   SNavierMonolithicSolver* NSSolver = new SNavierMonolithicSolver(pmesh,vorder,porder,kin_vis,verbose);



   //
   /// 6. Set parameters of the Fixed Point Solver
   // 
   SolverParams sFP = {rel_tol, abs_tol, tot_iter, print_level}; // rtol, atol, maxIter, print level
   NSSolver->SetOuterSolver(sFP, nPicardIterations);

   NSSolver->SetAlpha(alpha, AlphaType::CONSTANT);
   NSSolver->SetGamma(gamma);


   //
   /// 7. Add boundary conditions (Velocity-Dirichlet, Traction) and forcing term/s
   //
   // Essential velocity bcs
   int inflow_attr = 1;
   int outflow_attr = 2;
   Array<int> ess_attr(pmesh->bdr_attributes.Max());
   ess_attr = 1;                    // Mark walls/cylinder for no-slip condition 
   ess_attr[inflow_attr - 1]  = 0;
   ess_attr[outflow_attr - 1] = 0;

   // Inflow
   NSSolver->AddVelDirichletBC(V_inflow,inflow_attr);
         
   // No Slip
   NSSolver->AddVelDirichletBC(noSlip,ess_attr);



   //
   /// 8. Finalize Setup of solver
   //
   NSSolver->Setup();
   //DataCollection::Format forma = DataCollection::PARALLEL_FORMAT
   NSSolver->SetupOutput( outFolder, visit, paraview );



   //
   /// 9. Solve the forward problem
   //
   NSSolver->FSolve(); 


   // Free memory
   delete pmesh;
   delete NSSolver;

   // Finalize Hypre and MPI
   HYPRE_Finalize();
   MPI_Finalize();

   return 0;
}


// Functions
void V_inflow(const Vector &X, Vector &v)
{
   const int dim = X.Size();

   double x = X[0];
   double y = X[1];

   v = 0.0;

   v(1) = 0.0;

   if( dim == 3) {
      double z = X[2];
      v(0) = Umax * 32 / L *( 1 - y/L) * ( 1 - 2 * y / L) * ( z/L - 1) * z; 
      v(2) = 0.0;  
   }
   else
   {
      v(0) = Umax / ( ( (2*L - 3)*(4*L - 3) ) / ( 8 * std::pow(L,2) ) ) * ( 1 - y/L) * ( 1 - 2 * y / L);;   
   }

   
   
}

void noSlip(const Vector &X, Vector &v)
{
   v = 0.0;
}



