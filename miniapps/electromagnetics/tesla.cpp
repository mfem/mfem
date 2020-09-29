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
//            Tesla Miniapp:  Simple Magnetostatics Simulation Code
//            -----------------------------------------------------
//
// This miniapp solves a simple 3D magnetostatic problem.
//
//                     Curl 1/mu Curl A = J + Curl mu0/mu M
//
// The permeability function is that of the vacuum with an optional diamagnetic
// or paramagnetic spherical shell. The optional current density takes the form
// of a user defined ring of current. The optional magnetization consists of a
// cylindrical bar of constant magnetization.
//
// The boundary conditions either apply a user selected uniform magnetic flux
// density or a surface current flowing between user defined surfaces.
//
// We discretize the vector potential with H(Curl) finite elements. The magnetic
// flux B is discretized with H(Div) finite elements.
//
// Compile with: make tesla
//
// Sample runs:
//
//   A cylindrical bar magnet in a metal sphere:
//      mpirun -np 4 tesla -bm '0 -0.5 0 0 0.5 0 0.2 1'
//
//   A spherical shell of paramagnetic material in a uniform B field:
//      mpirun -np 4 tesla -ubbc '0 0 1' -ms '0 0 0 0.2 0.4 10'
//
//   A ring of current in a metal sphere:
//      mpirun -np 4 tesla -cr '0 0 -0.2 0 0 0.2 0.2 0.4 1'
//
//   A Halbach array of permanent magnets:
//      mpirun -np 4 tesla -m ../../data/beam-hex.mesh -rs 2
//                         -ha '1 0.1 0.3 7 0.9 0.7 0 1 12'
//
//   An example demonstrating the use of surface currents:
//      mpirun -np 4 tesla -m square-angled-pipe.mesh
//                         -kbcs '3' -vbcs '1 2' -vbcv '-0.5 0.5'
//
//   An example combining the paramagnetic shell, permanent magnet,
//   and current ring:
//      mpirun -np 4 tesla -m ../../data/inline-hex.mesh
//                         -ms '0.5 0.5 0.5 0.4 0.45 20'
//                         -bm '0.5 0.5 0.3 0.5 0.5 0.7 0.1 1'
//                         -cr '0.5 0.5 0.45 0.5 0.5 0.55 0.2 0.3 1'
//
//   By default the sources and fields are all zero:
//      mpirun -np 4 tesla

#include "tesla_solver.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;
using namespace mfem::electromagnetics;

// Permeability Function
Coefficient * SetupInvPermeabilityCoefficient();

static Vector pw_mu_(0);      // Piecewise permeability values
static Vector pw_mu_inv_(0);  // Piecewise inverse permeability values
static Vector ms_params_(0);  // Center, Inner and Outer Radii, and
//                               Permeability of magnetic shell
double magnetic_shell(const Vector &);
double magnetic_shell_inv(const Vector & x) { return 1.0/magnetic_shell(x); }

// Current Density Function
static Vector cr_params_(0);  // Axis Start, Axis End, Inner Ring Radius,
//                               Outer Ring Radius, and Total Current
//                               of current ring (annulus)
void current_ring(const Vector &, Vector &);

// Magnetization
static Vector bm_params_(0);  // Axis Start, Axis End, Bar Radius,
//                               and Magnetic Field Magnitude
void bar_magnet(const Vector &, Vector &);

static Vector ha_params_(0);  // Bounding box,
//                               axis index (0->'x', 1->'y', 2->'z'),
//                               rotation axis index
//                               and number of segments
void halbach_array(const Vector &, Vector &);

// A Field Boundary Condition for B = (Bx,By,Bz)
static Vector b_uniform_(0);
void a_bc_uniform(const Vector &, Vector&);

// Phi_M Boundary Condition for H = (0,0,1)
double phi_m_bc_uniform(const Vector &x);

// Prints the program's logo to the given output stream
void display_banner(ostream & os);

int main(int argc, char *argv[])
{
   MPI_Session mpi(argc, argv);

   if ( mpi.Root() ) { display_banner(cout); }

   // Parse command-line options.
   const char *mesh_file = "../../data/ball-nurbs.mesh";
   int order = 1;
   int maxit = 100;
   int serial_ref_levels = 0;
   int parallel_ref_levels = 0;
   bool visualization = true;
   bool visit = true;

   Array<int> kbcs;
   Array<int> vbcs;

   Vector vbcv;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&serial_ref_levels, "-rs", "--serial-ref-levels",
                  "Number of serial refinement levels.");
   args.AddOption(&parallel_ref_levels, "-rp", "--parallel-ref-levels",
                  "Number of parallel refinement levels.");
   args.AddOption(&b_uniform_, "-ubbc", "--uniform-b-bc",
                  "Specify if the three components of the constant magnetic flux density");
   args.AddOption(&pw_mu_, "-pwm", "--piecewise-mu",
                  "Piecewise values of Permeability");
   args.AddOption(&ms_params_, "-ms", "--magnetic-shell-params",
                  "Center, Inner Radius, Outer Radius, and Permeability of Magnetic Shell");
   args.AddOption(&cr_params_, "-cr", "--current-ring-params",
                  "Axis End Points, Inner Radius, Outer Radius and Total Current of Annulus");
   args.AddOption(&bm_params_, "-bm", "--bar-magnet-params",
                  "Axis End Points, Radius, and Magnetic Field of Cylindrical Magnet");
   args.AddOption(&ha_params_, "-ha", "--halbach-array-params",
                  "Bounding Box Corners and Number of Segments");
   args.AddOption(&kbcs, "-kbcs", "--surface-current-bc",
                  "Surfaces for the Surface Current (K) Boundary Condition");
   args.AddOption(&vbcs, "-vbcs", "--voltage-bc-surf",
                  "Voltage Boundary Condition Surfaces (to drive K)");
   args.AddOption(&vbcv, "-vbcv", "--voltage-bc-vals",
                  "Voltage Boundary Condition Values (to drive K)");
   args.AddOption(&maxit, "-maxit", "--max-amr-iterations",
                  "Max number of iterations in the main AMR loop.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&visit, "-visit", "--visit", "-no-visit", "--no-visit",
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

   // Read the (serial) mesh from the given mesh file on all processors.  We
   // can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
   // and volume meshes with the same code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);

   if (mpi.Root())
   {
      cout << "Starting initialization." << endl;
   }

   // Project a NURBS mesh to a piecewise-quadratic curved mesh
   if (mesh->NURBSext)
   {
      mesh->UniformRefinement();
      if (serial_ref_levels > 0) { serial_ref_levels--; }

      mesh->SetCurvature(2);
   }

   // Ensure that quad and hex meshes are treated as non-conforming.
   mesh->EnsureNCMesh();

   // Refine the serial mesh on all processors to increase the resolution. In
   // this example we do 'ref_levels' of uniform refinement.
   for (int l = 0; l < serial_ref_levels; l++)
   {
      mesh->UniformRefinement();
   }

   // Define a parallel mesh by a partitioning of the serial mesh. Refine
   // this mesh further in parallel to increase the resolution. Once the
   // parallel mesh is defined, the serial mesh can be deleted.
   ParMesh pmesh(MPI_COMM_WORLD, *mesh);
   delete mesh;

   // Refine this mesh in parallel to increase the resolution.
   int par_ref_levels = parallel_ref_levels;
   for (int l = 0; l < par_ref_levels; l++)
   {
      pmesh.UniformRefinement();
   }
   // Make sure tet-only meshes are marked for local refinement.
   pmesh.Finalize(true);

   // If values for Voltage BCs were not set issue a warning and exit
   if ( ( vbcs.Size() > 0 && kbcs.Size() == 0 ) ||
        ( kbcs.Size() > 0 && vbcs.Size() == 0 ) ||
        ( vbcv.Size() < vbcs.Size() ) )
   {
      if ( mpi.Root() )
      {
         cout << "The surface current (K) boundary condition requires "
              << "surface current boundary condition surfaces (with -kbcs), "
              << "voltage boundary condition surface (with -vbcs), "
              << "and voltage boundary condition values (with -vbcv)."
              << endl;
      }
      return 3;
   }

   // Create a coefficient describing the magnetic permeability
   Coefficient * muInvCoef = SetupInvPermeabilityCoefficient();

   // Create the Magnetostatic solver
   TeslaSolver Tesla(pmesh, order, kbcs, vbcs, vbcv, *muInvCoef,
                     (b_uniform_.Size() > 0 ) ? a_bc_uniform  : NULL,
                     (cr_params_.Size() > 0 ) ? current_ring  : NULL,
                     (bm_params_.Size() > 0 ) ? bar_magnet    :
                     (ha_params_.Size() > 0 ) ? halbach_array : NULL);

   // Initialize GLVis visualization
   if (visualization)
   {
      Tesla.InitializeGLVis();
   }

   // Initialize VisIt visualization
   VisItDataCollection visit_dc("Tesla-AMR-Parallel", &pmesh);

   if ( visit )
   {
      Tesla.RegisterVisItFields(visit_dc);
   }
   if (mpi.Root()) { cout << "Initialization done." << endl; }

   // The main AMR loop. In each iteration we solve the problem on the current
   // mesh, visualize the solution, estimate the error on all elements, refine
   // the worst elements and update all objects to work with the new mesh. We
   // refine until the maximum number of dofs in the Nedelec finite element
   // space reaches 10 million.
   const int max_dofs = 10000000;
   for (int it = 1; it <= maxit; it++)
   {
      if (mpi.Root())
      {
         cout << "\nAMR Iteration " << it << endl;
      }

      // Display the current number of DoFs in each finite element space
      Tesla.PrintSizes();

      // Assemble all forms
      Tesla.Assemble();

      // Solve the system and compute any auxiliary fields
      Tesla.Solve();

      // Determine the current size of the linear system
      int prob_size = Tesla.GetProblemSize();

      // Write fields to disk for VisIt
      if ( visit )
      {
         Tesla.WriteVisItFields(it);
      }

      // Send the solution by socket to a GLVis server.
      if (visualization)
      {
         Tesla.DisplayToGLVis();
      }

      if (mpi.Root())
      {
         cout << "AMR iteration " << it << " complete." << endl;
      }

      // Check stopping criteria
      if (prob_size > max_dofs)
      {
         if (mpi.Root())
         {
            cout << "Reached maximum number of dofs, exiting..." << endl;
         }
         break;
      }
      if ( it == maxit )
      {
         break;
      }

      // Wait for user input. Ask every 10th iteration.
      char c = 'c';
      if (mpi.Root() && (it % 10 == 0))
      {
         cout << "press (q)uit or (c)ontinue --> " << flush;
         cin >> c;
      }
      MPI_Bcast(&c, 1, MPI_CHAR, 0, MPI_COMM_WORLD);

      if (c != 'c')
      {
         break;
      }

      // Estimate element errors using the Zienkiewicz-Zhu error estimator.
      Vector errors(pmesh.GetNE());
      Tesla.GetErrorEstimates(errors);

      double local_max_err = errors.Max();
      double global_max_err;
      MPI_Allreduce(&local_max_err, &global_max_err, 1,
                    MPI_DOUBLE, MPI_MAX, pmesh.GetComm());

      // Refine the elements whose error is larger than a fraction of the
      // maximum element error.
      const double frac = 0.5;
      double threshold = frac * global_max_err;
      if (mpi.Root()) { cout << "Refining ..." << endl; }
      pmesh.RefineByError(errors, threshold);

      // Update the magnetostatic solver to reflect the new state of the mesh.
      Tesla.Update();

      if (pmesh.Nonconforming() && mpi.WorldSize() > 1)
      {
         if (mpi.Root()) { cout << "Rebalancing ..." << endl; }
         pmesh.Rebalance();

         // Update again after rebalancing
         Tesla.Update();
      }
   }

   delete muInvCoef;

   return 0;
}

// Print the Volta ascii logo to the given ostream
void display_banner(ostream & os)
{
   os << "  ___________            __            " << endl
      << "  \\__    ___/___   _____|  | _____     " << endl
      << "    |    |_/ __ \\ /  ___/  | \\__  \\    " << endl
      << "    |    |\\  ___/ \\___ \\|  |__/ __ \\_  " << endl
      << "    |____| \\___  >____  >____(____  /  " << endl
      << "               \\/     \\/          \\/   " << endl << flush;
}

// The Permeability is a required coefficient which may be defined in
// various ways so we'll determine the appropriate coefficient type here.
Coefficient *
SetupInvPermeabilityCoefficient()
{
   Coefficient * coef = NULL;

   if ( ms_params_.Size() > 0 )
   {
      coef = new FunctionCoefficient(magnetic_shell_inv);
   }
   else if ( pw_mu_.Size() > 0 )
   {
      pw_mu_inv_.SetSize(pw_mu_.Size());
      for (int i = 0; i < pw_mu_.Size(); i++)
      {
         MFEM_ASSERT( pw_mu_[i] > 0.0, "permeability values must be positive" );
         pw_mu_inv_[i] = 1.0/pw_mu_[i];
      }
      coef = new PWConstCoefficient(pw_mu_inv_);
   }
   else
   {
      coef = new ConstantCoefficient(1.0/mu0_);
   }

   return coef;
}

// A spherical shell with constant permeability.  The sphere has inner
// and outer radii, center, and relative permeability specified on the
// command line and stored in ms_params_.
double magnetic_shell(const Vector &x)
{
   double r2 = 0.0;

   for (int i = 0; i < x.Size(); i++)
   {
      r2 += (x(i) - ms_params_(i))*(x(i) - ms_params_(i));
   }

   if ( sqrt(r2) >= ms_params_(x.Size()) &&
        sqrt(r2) <= ms_params_(x.Size()+1) )
   {
      return mu0_*ms_params_(x.Size()+2);
   }
   return mu0_;
}

// An annular ring of current density.  The ring has two axis end
// points, inner and outer radii, and a constant current in Amperes.
void current_ring(const Vector &x, Vector &j)
{
   MFEM_ASSERT(x.Size() == 3, "current_ring source requires 3D space.");

   j.SetSize(x.Size());
   j = 0.0;

   Vector  a(x.Size());  // Normalized Axis vector
   Vector xu(x.Size());  // x vector relative to the axis end-point
   Vector ju(x.Size());  // Unit vector in direction of current

   xu = x;

   for (int i=0; i<x.Size(); i++)
   {
      xu[i] -= cr_params_[i];
      a[i]   = cr_params_[x.Size()+i] - cr_params_[i];
   }

   double h = a.Norml2();

   if ( h == 0.0 )
   {
      return;
   }

   double ra = cr_params_[2*x.Size()+0];
   double rb = cr_params_[2*x.Size()+1];
   if ( ra > rb )
   {
      double rc = ra;
      ra = rb;
      rb = rc;
   }
   double xa = xu*a;

   if ( h > 0.0 )
   {
      xu.Add(-xa/(h*h),a);
   }

   double xp = xu.Norml2();

   if ( xa >= 0.0 && xa <= h*h && xp >= ra && xp <= rb )
   {
      ju(0) = a(1) * xu(2) - a(2) * xu(1);
      ju(1) = a(2) * xu(0) - a(0) * xu(2);
      ju(2) = a(0) * xu(1) - a(1) * xu(0);
      ju /= h;

      j.Add(cr_params_[2*x.Size()+2]/(h*(rb-ra)),ju);
   }
}

// A Cylindrical Rod of constant magnetization.  The cylinder has two
// axis end points, a radius, and a constant magnetic field oriented
// along the axis.
void bar_magnet(const Vector &x, Vector &m)
{
   m.SetSize(x.Size());
   m = 0.0;

   Vector  a(x.Size());  // Normalized Axis vector
   Vector xu(x.Size());  // x vector relative to the axis end-point

   xu = x;

   for (int i=0; i<x.Size(); i++)
   {
      xu[i] -= bm_params_[i];
      a[i]   = bm_params_[x.Size()+i] - bm_params_[i];
   }

   double h = a.Norml2();

   if ( h == 0.0 )
   {
      return;
   }

   double  r = bm_params_[2*x.Size()];
   double xa = xu*a;

   if ( h > 0.0 )
   {
      xu.Add(-xa/(h*h),a);
   }

   double xp = xu.Norml2();

   if ( xa >= 0.0 && xa <= h*h && xp <= r )
   {
      m.Add(bm_params_[2*x.Size()+1]/h,a);
   }
}

// A Square Rod of rotating magnetized segments.  The rod is defined
// by a bounding box and a number of segments.  The magnetization in
// each segment is constant and follows a rotating pattern.
void halbach_array(const Vector &x, Vector &m)
{
   m.SetSize(x.Size());
   m = 0.0;

   // Check Bounding Box
   if ( x[0] < ha_params_[0] || x[0] > ha_params_[3] ||
        x[1] < ha_params_[1] || x[1] > ha_params_[4] ||
        x[2] < ha_params_[2] || x[2] > ha_params_[5] )
   {
      return;
   }

   int ai = (int)ha_params_[6];
   int ri = (int)ha_params_[7];
   int n  = (int)ha_params_[8];

   int i = (int)n * (x[ai] - ha_params_[ai]) /
           (ha_params_[ai+3] - ha_params_[ai]);

   m[(ri + 1 + (i % 2)) % 3] = pow(-1.0,i/2);
}

// To produce a uniform magnetic flux the vector potential can be set
// to ( By z, Bz x, Bx y).
void a_bc_uniform(const Vector & x, Vector & a)
{
   a.SetSize(3);
   a(0) = b_uniform_(1) * x(2);
   a(1) = b_uniform_(2) * x(0);
   a(2) = b_uniform_(0) * x(1);
}

// To produce a uniform magnetic field the scalar potential can be set
// to -z (or -y in 2D).
double phi_m_bc_uniform(const Vector &x)
{
   return -x(x.Size()-1);
}
