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
//    ------------------------------------------------------------------
//    Maxwell Miniapp:  Simple Full-Wave Electromagnetic Simulation Code
//    ------------------------------------------------------------------
//
// This miniapp solves a simple 3D full-wave electromagnetic problem using the
// coupled, first-order equations:
//
//                 epsilon dE/dt = Curl 1/mu B - sigma E - J
//                         dB/dt = - Curl E
//
// The permittivity function is that of the vacuum with an optional dielectric
// sphere. The permeability function is that of the vacuum with an optional
// diamagnetic or paramagnetic spherical shell. The optional conductivity
// function is also a user-defined sphere.
//
// The optional current density is a pulse of current in the shape of a cylinder
// with a time dependence resembling the derivative of a Gaussian distribution.
//
// Boundary conditions can be 'natural' meaning zero tangential current,
// 'Dirichlet' which sets the time-derivative of the tangential components of E,
// or 'absorbing' (we use a simple Sommerfeld first order absorbing boundary
// condition).
//
// We discretize the electric field with H(Curl) finite elements (Nedelec edge
// elements) and the magnetic flux with H(Div) finite elements (Raviart-Thomas
// elements).
//
// The symplectic time integration algorithm used below is designed to conserve
// energy unless lossy materials or absorbing boundary conditions are used.
// When losses are expected, the algorithm uses an implicit method which
// includes the loss operators in the left hand side of the linear system.
//
// For increased accuracy the time integration order can be set to 2, 3, or 4
// (the default is 1st order).
//
// Compile with: make maxwell
//
// Sample runs:
//
//   Current source in a sphere with absorbing boundary conditions:
//     mpirun -np 4 maxwell -m ../../data/ball-nurbs.mesh -rs 2
//                          -abcs '-1'
//                          -dp '-0.3 0.0 0.0 0.3 0.0 0.0 0.1 1 .5 .5'
//
//   Current source in a metal sphere with dielectric and conducting materials:
//     mpirun -np 4 maxwell -m ../../data/ball-nurbs.mesh -rs 2
//                          -dbcs '-1'
//                          -dp '-0.3 0.0 0.0 0.3 0.0 0.0 0.1 1 .5 .5'
//                          -cs '0.0 0.0 -0.5 .2 3e6'
//                          -ds '0.0 0.0 0.5 .2 10'
//
//   Current source in a metal box:
//     mpirun -np 4 maxwell -m ../../data/fichera.mesh -rs 3
//                          -ts 0.25 -tf 10 -dbcs '-1'
//                          -dp '-0.5 -0.5 0.0 -0.5 -0.5 1.0 0.1 1 .5 1'
//
//   Current source with a mixture of absorbing and reflecting boundaries:
//     mpirun -np 4 maxwell -m ../../data/fichera.mesh -rs 3
//                          -ts 0.25 -tf 10
//                          -dp '-0.5 -0.5 0.0 -0.5 -0.5 1.0 0.1 1 .5 1'
//                          -dbcs '4 8 19 21' -abcs '5 18'
//
//   By default the sources and fields are all zero:
//     mpirun -np 4 maxwell

#include "maxwell_solver.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;
using namespace mfem::common;
using namespace mfem::electromagnetics;

// Permittivity Function
static Vector ds_params_(0);  // Center, Radius, and Permittivity
//                               of dielectric sphere
double dielectric_sphere(const Vector &);
double epsilon(const Vector &x) { return dielectric_sphere(x); }

// Permeability Function
static Vector ms_params_(0);  // Center, Inner and Outer Radii, and
//                               Permeability of magnetic shell
double magnetic_shell(const Vector &);
double muInv(const Vector & x) { return 1.0/magnetic_shell(x); }

// Conductivity Function
static Vector cs_params_(0);  // Center, Radius, and Conductivity
//                               of conductive sphere
double conductive_sphere(const Vector &);
double sigma(const Vector &x) { return conductive_sphere(x); }

// Current Density Function
static Vector dp_params_(0);  // Axis Start, Axis End, Rod Radius,
//                               Total Current of Rod, and Frequency
void dipole_pulse(const Vector &x, double t, Vector &j);
void j_src(const Vector &x, double t, Vector &j) { dipole_pulse(x, t, j); }

// dE/dt Boundary Condition: The following function returns zero but any time
// dependent function could be used.
void dEdtBCFunc(const Vector &x, double t, Vector &E);

// The following functions return zero but they could be modified to set initial
// conditions for the electric and magnetic fields
void EFieldFunc(const Vector &, Vector&);
void BFieldFunc(const Vector &, Vector&);

// Scale factor between input time units and seconds
static double tScale_ = 1e-9;  // Input time in nanosecond

int SnapTimeStep(double tmax, double dtmax, double & dt);

// Prints the program's logo to the given output stream
void display_banner(ostream & os);

int main(int argc, char *argv[])
{
   MPI_Session mpi(argc, argv);

   if ( mpi.Root() ) { display_banner(cout); }

   // Parse command-line options.
   const char *mesh_file = "../../data/ball-nurbs.mesh";
   int sOrder = 1;
   int tOrder = 1;
   int serial_ref_levels = 0;
   int parallel_ref_levels = 0;
   bool visualization = true;
   bool visit = true;
   double dt = 1.0e-12;
   double dtsf = 0.95;
   double ti = 0.0;
   double ts = 1.0;
   double tf = 40.0;

   Array<int> abcs;
   Array<int> dbcs;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&sOrder, "-so", "--spatial-order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&tOrder, "-to", "--temporal-order",
                  "Time integration order.");
   args.AddOption(&serial_ref_levels, "-rs", "--serial-ref-levels",
                  "Number of serial refinement levels.");
   args.AddOption(&parallel_ref_levels, "-rp", "--parallel-ref-levels",
                  "Number of parallel refinement levels.");
   args.AddOption(&dtsf, "-sf", "--dt-safety-factor",
                  "Used to reduce the time step below the upper bound.");
   args.AddOption(&ti, "-ti", "--initial-time",
                  "Beginning of time interval to simulate (ns).");
   args.AddOption(&tf, "-tf", "--final-time",
                  "End of time interval to simulate (ns).");
   args.AddOption(&ts, "-ts", "--snapshot-time",
                  "Time between snapshots (ns).");
   args.AddOption(&ds_params_, "-ds", "--dielectric-sphere-params",
                  "Center, Radius, and Permittivity of Dielectric Sphere");
   args.AddOption(&ms_params_, "-ms", "--magnetic-shell-params",
                  "Center, Inner Radius, Outer Radius, and Permeability "
                  "of Magnetic Shell");
   args.AddOption(&cs_params_, "-cs", "--conductive-sphere-params",
                  "Center, Radius, and Conductivity of Conductive Sphere");
   args.AddOption(&dp_params_, "-dp", "--dipole-pulse-params",
                  "Axis End Points, Radius, Amplitude, "
                  "Pulse Center (ns), Pulse Width (ns)");
   args.AddOption(&abcs, "-abcs", "--absorbing-bc-surf",
                  "Absorbing Boundary Condition Surfaces");
   args.AddOption(&dbcs, "-dbcs", "--dirichlet-bc-surf",
                  "Dirichlet Boundary Condition Surfaces");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&visit, "-visit", "--visit", "-no-visit",
                  "--no-visualization",
                  "Enable or disable VisIt visualization.");
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

   // Read the (serial) mesh from the given mesh file on all processors.  We can
   // handle triangular, quadrilateral, tetrahedral, hexahedral, surface and
   // volume meshes with the same code.
   Mesh *mesh;
   ifstream imesh(mesh_file);
   if (!imesh)
   {
      if (mpi.Root())
      {
         cerr << "\nCan not open mesh file: " << mesh_file << '\n' << endl;
      }
      return 2;
   }
   mesh = new Mesh(imesh, 1, 1);
   imesh.close();

   // Project a NURBS mesh to a piecewise-quadratic curved mesh
   if (mesh->NURBSext)
   {
      mesh->UniformRefinement();
      if (serial_ref_levels > 0) { serial_ref_levels--; }

      mesh->SetCurvature(2);
   }

   // Refine the serial mesh on all processors to increase the resolution. In
   // this example we do 'ref_levels' of uniform refinement.
   for (int l = 0; l < serial_ref_levels; l++)
   {
      mesh->UniformRefinement();
   }

   // Define a parallel mesh by a partitioning of the serial mesh. Refine this
   // mesh further in parallel to increase the resolution. Once the parallel
   // mesh is defined, the serial mesh can be deleted.
   ParMesh pmesh(MPI_COMM_WORLD, *mesh);
   delete mesh;

   // Refine this mesh in parallel to increase the resolution.
   for (int l = 0; l < parallel_ref_levels; l++)
   {
      pmesh.UniformRefinement();
   }

   // Create the Electromagnetic solver
   MaxwellSolver Maxwell(pmesh, sOrder,
                         ( ds_params_.Size() > 0 ) ? epsilon     : NULL,
                         ( ms_params_.Size() > 0 ) ? muInv       : NULL,
                         ( cs_params_.Size() > 0 ) ? sigma       : NULL,
                         ( dp_params_.Size() > 0 ) ? j_src       : NULL,
                         abcs, dbcs,
                         (       dbcs.Size() > 0 ) ? dEdtBCFunc  : NULL
                        );

   // Display the current number of DoFs in each finite element space
   Maxwell.PrintSizes();

   // Set the initial conditions for both the electric and magnetic fields
   VectorFunctionCoefficient EFieldCoef(3,EFieldFunc);
   VectorFunctionCoefficient BFieldCoef(3,BFieldFunc);

   Maxwell.SetInitialEField(EFieldCoef);
   Maxwell.SetInitialBField(BFieldCoef);

   // Compute the energy of the initial fields
   double energy = Maxwell.GetEnergy();
   if ( mpi.Root() )
   {
      cout << "Energy(" << ti << "ns):  " << energy << "J" << endl;
   }

   // Approximate the largest stable time step
   double dtmax = Maxwell.GetMaximumTimeStep();

   // Convert times from nanoseconds to seconds
   ti *= tScale_;
   tf *= tScale_;
   ts *= tScale_;

   if ( mpi.Root() )
   {
      cout << "Maximum Time Step:     " << dtmax / tScale_ << "ns" << endl;
   }

   // Round down the time step so that tf-ti is an integer multiple of dt
   int nsteps = SnapTimeStep(tf-ti, dtsf * dtmax, dt);
   if ( mpi.Root() )
   {
      cout << "Number of Time Steps:  " << nsteps << endl;
      cout << "Time Step Size:        " << dt / tScale_ << "ns" << endl;
   }

   // Create the ODE solver
   SIAVSolver siaSolver(tOrder);
   siaSolver.Init(Maxwell.GetNegCurl(), Maxwell);


   // Initialize GLVis visualization
   if (visualization)
   {
      Maxwell.InitializeGLVis();
   }

   // Initialize VisIt visualization
   VisItDataCollection visit_dc("Maxwell-Parallel", &pmesh);

   double t = ti;
   Maxwell.SetTime(t);

   if ( visit )
   {
      Maxwell.RegisterVisItFields(visit_dc);
   }

   // Write initial fields to disk for VisIt
   if ( visit )
   {
      Maxwell.WriteVisItFields(0);
   }

   // Send the initial condition by socket to a GLVis server.
   if (visualization)
   {
      Maxwell.DisplayToGLVis();
   }

   // The main time evolution loop.
   int it = 1;
   while (t < tf)
   {
      // Run the simulation until a snapshot is needed
      siaSolver.Run(Maxwell.GetBField(), Maxwell.GetEField(), t, dt,
                    max(t + dt, ti + ts * it));

      // Approximate the current energy if the fields
      energy = Maxwell.GetEnergy();
      if ( mpi.Root() )
      {
         cout << "Energy(" << t/tScale_ << "ns):  " << energy << "J" << endl;
      }

      // Update local DoFs with current true DoFs
      Maxwell.SyncGridFuncs();

      // Write fields to disk for VisIt
      if ( visit )
      {
         Maxwell.WriteVisItFields(it);
      }

      // Send the solution by socket to a GLVis server.
      if (visualization)
      {
         Maxwell.DisplayToGLVis();
      }

      it++;
   }

   return 0;
}

// Print the Maxwell ascii logo to the given ostream
void display_banner(ostream & os)
{
   os << "     ___    ____                                      " << endl
      << "    /   |  /   /                           __   __    " << endl
      << "   /    |_/ _ /__  ___  _____  _  __ ____ |  | |  |   " << endl
      << "  /         \\__  \\ \\  \\/  /\\ \\/ \\/ // __ \\|  | |  |   "
      << endl
      << " /   /|_/   // __ \\_>    <  \\     /\\  ___/|  |_|  |__ " << endl
      << "/___/  /_  /(____  /__/\\_ \\  \\/\\_/  \\___  >____/____/ " << endl
      << "         \\/       \\/      \\/             \\/           " << endl
      << flush;
}

// A sphere with constant permittivity.  The sphere has a radius, center, and
// permittivity specified on the command line and stored in ds_params_.
double dielectric_sphere(const Vector &x)
{
   double r2 = 0.0;

   for (int i=0; i<x.Size(); i++)
   {
      r2 += (x(i)-ds_params_(i))*(x(i)-ds_params_(i));
   }

   if ( sqrt(r2) <= ds_params_(x.Size()) )
   {
      return ds_params_(x.Size()+1) * epsilon0_;
   }
   return epsilon0_;
}

// A spherical shell with constant permeability.  The sphere has inner and outer
// radii, center, and relative permeability specified on the command line and
// stored in ms_params_.
double magnetic_shell(const Vector &x)
{
   double r2 = 0.0;

   for (int i=0; i<x.Size(); i++)
   {
      r2 += (x(i)-ms_params_(i))*(x(i)-ms_params_(i));
   }

   if ( sqrt(r2) >= ms_params_(x.Size()) &&
        sqrt(r2) <= ms_params_(x.Size()+1) )
   {
      return mu0_*ms_params_(x.Size()+2);
   }
   return mu0_;
}

// A sphere with constant conductivity.  The sphere has a radius, center, and
// conductivity specified on the command line and stored in ls_params_.
double conductive_sphere(const Vector &x)
{
   double r2 = 0.0;

   for (int i=0; i<x.Size(); i++)
   {
      r2 += (x(i)-cs_params_(i))*(x(i)-cs_params_(i));
   }

   if ( sqrt(r2) <= cs_params_(x.Size()) )
   {
      return cs_params_(x.Size()+1);
   }
   return 0.0;
}

// A cylindrical rod of current density.  The rod has two axis end points, a
// radius, a current amplitude in Amperes, a center time, and a width.  All of
// these parameters are stored in dp_params_.
void dipole_pulse(const Vector &x, double t, Vector &j)
{
   MFEM_ASSERT(x.Size() == 3, "current source requires 3D space.");

   j.SetSize(x.Size());
   j = 0.0;

   Vector  v(x.Size());  // Normalized Axis vector
   Vector xu(x.Size());  // x vector relative to the axis end-point

   xu = x;

   for (int i=0; i<x.Size(); i++)
   {
      xu[i] -= dp_params_[i];
      v[i]   = dp_params_[x.Size()+i] - dp_params_[i];
   }

   double h = v.Norml2();

   if ( h == 0.0 )
   {
      return;
   }
   v /= h;

   double r = dp_params_[2*x.Size()+0];
   double a = dp_params_[2*x.Size()+1] * tScale_;
   double b = dp_params_[2*x.Size()+2] * tScale_;
   double c = dp_params_[2*x.Size()+3] * tScale_;

   double xv = xu * v;

   // Compute perpendicular vector from axis to x
   xu.Add(-xv, v);

   double xp = xu.Norml2();

   if ( xv >= 0.0 && xv <= h && xp <= r )
   {
      j = v;
   }

   j *= a * (t - b) * exp(-0.5 * pow((t-b)/c, 2)) / (c * c);
}

void
EFieldFunc(const Vector &x, Vector &E)
{
   E.SetSize(3);
   E = 0.0;
}

void
BFieldFunc(const Vector &x, Vector &B)
{
   B.SetSize(3);
   B = 0.0;
}

void
dEdtBCFunc(const Vector &x, double t, Vector &dE)
{
   dE.SetSize(3);
   dE = 0.0;
}

int
SnapTimeStep(double tmax, double dtmax, double & dt)
{
   double dsteps = tmax/dtmax;

   int nsteps = pow(10,(int)ceil(log10(dsteps)));

   for (int i=1; i<=5; i++)
   {
      int a = (int)ceil(log10(dsteps/pow(5.0,i)));
      int nstepsi = (int)pow(5,i)*max(1,(int)pow(10,a));

      nsteps = min(nsteps,nstepsi);
   }

   dt = tmax / nsteps;

   return nsteps;
}
