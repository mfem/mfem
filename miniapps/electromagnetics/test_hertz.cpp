// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.
//
//   -----------------------------------------------------------------------
//   Hertz Miniapp:  Simple Frequency-Domain Electromagnetic Simulation Code
//   -----------------------------------------------------------------------
//
//   Assumes that all sources and boundary conditions oscillate with the same
//   frequency although not necessarily in phase with one another.  This
//   assumptions implies that we can factor out the time dependence which we
//   take to be of the form exp(i omega t).  With these assumptions we can
//   write the Maxwell equations in the form:
//
//   i omega epsilon E = Curl mu^{-1} B - J - sigma E
//   i omega B         = - Curl E
//
//   Which combine to yield:
//
//   Curl mu^{-1} Curl E - omega^2 epsilon E + i omega sigma E = - i omega J
//
//   We discretize this equation with H(Curl) a.k.a Nedelec basis
//   functions.  The curl curl operator must be handled with
//   integration by parts which yields a surface integral:
//
//   (W, Curl mu^{-1} Curl E) = (Curl W, mu^{-1} Curl E)
//               + (W, n x (mu^{-1} Curl E))_{\Gamma}
//
//   or
//
//   (W, Curl mu^{-1} Curl E) = (Curl W, mu^{-1} Curl E)
//               - i omega (W, n x H)_{\Gamma}
//
//   For plane waves
//     omega B = - k x E
//     omega D = k x H, assuming n x k = 0 => n x H = omega epsilon E / |k|
//
//   c = omega/|k|
//
//   (W, Curl mu^{-1} Curl E) = (Curl W, mu^{-1} Curl E)
//               - i omega sqrt{epsilon/mu} (W, E)_{\Gamma}
//
//
// Compile with: make hertz
//
// Sample runs:
//
//   By default the sources and fields are all zero
//     mpirun -np 4 hertz
//
//   Current source in a sphere with absorbing boundary conditions
//     mpirun -np 4 hertz -m ../../data/ball-nurbs.mesh -rs 2
//                        -abcs '-1' -f 3e8
//                        -do '-0.3 0.0 0.0 0.3 0.0 0.0 0.1 1 .5 .5'
//
//   Current source in a metal sphere with dielectric and conducting materials
//     mpirun -np 4 hertz -m ../../data/ball-nurbs.mesh -rs 2
//                        -dbcs '-1' -f 3e8
//                        -do '-0.3 0.0 0.0 0.3 0.0 0.0 0.1 1 .5 .5'
//                        -cs '0.0 0.0 -0.5 .2 10'
//                        -ds '0.0 0.0 0.5 .2 10'
//
//   Current source in a metal box
//     mpirun -np 4 hertz -m ../../data/fichera.mesh -rs 3
//                        -dbcs '-1' -f 3e8
//                        -do '-0.5 -0.5 0.0 -0.5 -0.5 1.0 0.1 1 .5 1'
//
//   Current source with a mixture of absorbing and reflecting boundaries
//     mpirun -np 4 hertz -m ../../data/fichera.mesh -rs 3
//                        -do '-0.5 -0.5 0.0 -0.5 -0.5 1.0 0.1 1 .5 1'
//                        -dbcs '4 8 19 21' -abcs '5 18' -f 3e8
//

#include "test_hertz_solver.hpp"
#include <fstream>
#include <iostream>
#include <cmath>
#include <complex>

using namespace std;
using namespace mfem;
using namespace mfem::electromagnetics;

// Physical Constants
static double qe = 1.60217662e-19;
static double u = 1.660539040e-27;
static double me = 9.10938356e-31;

// Permittivity Functions
Coefficient * SetupPermittivityCoefficient();

static Vector pw_eps_(0);     // Piecewise permittivity values
static Vector ds_params_(0);  // Center, Radius, and Permittivity
//                               of dielectric sphere
class DielectricTensor: public MatrixCoefficient
{
public:
   DielectricTensor(ParGridFunction & B,
                    BlockVector & T,
                    BlockVector & density,
                    ParFiniteElementSpace & H1FESpace,
                    ParFiniteElementSpace & L2FESpace,
                    int nspecies,
                    double omega,
                    bool realPart = true);
   virtual void Eval(DenseMatrix &K, ElementTransformation &T,
                     const IntegrationPoint &ip);
   // virtual void Dval(DenseMatrix &K, ElementTransformation &T,
   //                   const IntegrationPoint &ip);
   virtual ~DielectricTensor() {}

private:
   ParGridFunction * B_;
   BlockVector * temperature_;
   BlockVector * density_;
   ParFiniteElementSpace * H1FESpace_;
   ParFiniteElementSpace * L2FESpace_;
   int nspecies_;
   double omega_;
   bool realPart_;

   ParGridFunction density_gf_;
   ParGridFunction temperature_gf_;

   Vector density_vals_;
   Vector temperature_vals_;
};

double dielectric_sphere(const Vector &);

// Permeability Function
Coefficient * SetupInvPermeabilityCoefficient();

static Vector pw_mu_(0);      // Piecewise permeability values
static Vector pw_mu_inv_(0);  // Piecewise inverse permeability values
static Vector ms_params_(0);  // Center, Inner and Outer Radii, and
//                               Permeability of magnetic shell
double magnetic_shell(const Vector &);
double magnetic_shell_inv(const Vector & x) { return 1.0/magnetic_shell(x); }

// Conductivity Functions
//Coefficient * SetupConductivityCoefficient();

static Vector pw_sigma_(0);   // Piecewise conductivity values
static Vector cs_params_(0);  // Center, Radius, and Conductivity
//                               of conductive sphere
double conductive_sphere(const Vector &);

// Impedance
Coefficient * SetupAdmittanceCoefficient(const Mesh & mesh,
                                         const Array<int> & abcs);

static Vector pw_eta_(0);      // Piecewise impedance values
static Vector pw_eta_inv_(0);  // Piecewise inverse impedance values

// Current Density Function
static Vector do_params_(0);  // Axis Start, Axis End, Rod Radius,
//                               Total Current of Rod
static Vector slab_params_(0); // Amplitude of x, y, z current source

void dipole_oscillator(const Vector &x, Vector &j);
void slab_current_source(const Vector &x, Vector &j);
void j_src(const Vector &x, Vector &j)
{
   if (do_params_.Size() > 0)
   {
      dipole_oscillator(x, j);
   }
   else
   {
      slab_current_source(x, j);
   }
}

// Electric Field Boundary Condition: The following function returns zero but
// any function could be used.
void e_bc_r(const Vector &x, Vector &E);
void e_bc_i(const Vector &x, Vector &E);

static double freq_ = 1.0;

// Mesh Size
static Vector mesh_dim_(0); // x, y, z dimensions of mesh

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
   int sol = 2;
   int nspecies = 2;
   bool herm_conv = false;
   bool visualization = true;
   bool visit = true;

   Array<int> abcs;
   Array<int> dbcs;
   Array<int> num_elements;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   // args.AddOption(&nspecies, "-ns", "--num-species",
   //               "Number of ion species.");
   args.AddOption(&serial_ref_levels, "-rs", "--serial-ref-levels",
                  "Number of serial refinement levels.");
   args.AddOption(&parallel_ref_levels, "-rp", "--parallel-ref-levels",
                  "Number of parallel refinement levels.");
   args.AddOption(&freq_, "-f", "--frequency",
                  "Frequency in Hertz (of course...)");
   args.AddOption(&sol, "-s", "--solver",
                  "Solver: 1 - GMRES, 2 - FGMRES w/AMS");
   args.AddOption(&pw_eps_, "-pwe", "--piecewise-eps",
                  "Piecewise values of Permittivity");
   args.AddOption(&ds_params_, "-ds", "--dielectric-sphere-params",
                  "Center, Radius, and Permittivity of Dielectric Sphere");
   args.AddOption(&pw_mu_, "-pwm", "--piecewise-mu",
                  "Piecewise values of Permeability");
   args.AddOption(&ms_params_, "-ms", "--magnetic-shell-params",
                  "Center, Inner Radius, Outer Radius, "
                  "and Permeability of Magnetic Shell");
   args.AddOption(&pw_sigma_, "-pws", "--piecewise-sigma",
                  "Piecewise values of Conductivity");
   args.AddOption(&cs_params_, "-cs", "--conductive-sphere-params",
                  "Center, Radius, and Conductivity of Conductive Sphere");
   args.AddOption(&pw_eta_, "-pwz", "--piecewise-eta",
                  "Piecewise values of Impedance (one value per abc surface)");
   args.AddOption(&do_params_, "-do", "--dipole-oscillator-params",
                  "Axis End Points, Radius, and Amplitude");
   args.AddOption(&slab_params_, "-slab", "--slab_params",
                  "Amplitude");
   args.AddOption(&abcs, "-abcs", "--absorbing-bc-surf",
                  "Absorbing Boundary Condition Surfaces");
   args.AddOption(&dbcs, "-dbcs", "--dirichlet-bc-surf",
                  "Dirichlet Boundary Condition Surfaces");
   args.AddOption(&mesh_dim_, "-md", "--mesh_dimensions",
                  "The x, y, z mesh dimensions");
   args.AddOption(&num_elements, "-ne", "--num_elements",
                  "The number of elements in each x, y, z of mesh");
   /*
   args.AddOption(&dbcv, "-dbcv", "--dirichlet-bc-vals",
                  "Dirichlet Boundary Condition Values");
   args.AddOption(&dbcg, "-dbcg", "--dirichlet-bc-gradient",
                  "-no-dbcg", "--no-dirichlet-bc-gradient",
                  "Dirichlet Boundary Condition Gradient (phi = -z)");
   args.AddOption(&nbcs, "-nbcs", "--neumann-bc-surf",
                  "Neumann Boundary Condition Surfaces");
   args.AddOption(&nbcv, "-nbcv", "--neumann-bc-vals",
                  "Neumann Boundary Condition Values");
   args.AddOption(&kbcs, "-kbcs", "--surface-current-bc",
                  "Surfaces for the Surface Current (K) Boundary Condition");
   args.AddOption(&vbcs, "-vbcs", "--voltage-bc-surf",
                  "Voltage Boundary Condition Surfaces (to drive K)");
   args.AddOption(&vbcv, "-vbcv", "--voltage-bc-vals",
                  "Voltage Boundary Condition Values (to drive K)");
   */
   args.AddOption(&maxit, "-maxit", "--max-amr-iterations",
                  "Max number of iterations in the main AMR loop.");
   args.AddOption(&herm_conv, "-herm", "--hermitian", "-no-herm",
                  "--no-hermitian", "Use convention for Hermitian operators.");
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

   ComplexOperator::Convention conv =
      herm_conv ? ComplexOperator::HERMITIAN : ComplexOperator::BLOCK_SYMMETRIC;

   // Read the (serial) mesh from the given mesh file on all processors.  We
   // can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
   // and volume meshes with the same code.

   Mesh *mesh = NULL;

   if (mesh_dim_(0) == -1 && mesh_dim_(1) == -1 && mesh_dim_(2) == -1)
   {
      mesh = new Mesh(mesh_file, 1, 1);
   }
   else
   {
      mesh = new Mesh(num_elements[0], num_elements[1], num_elements[2],
                      Element::HEXAHEDRON, 1,
                      mesh_dim_(0), mesh_dim_(1), mesh_dim_(2));
   }

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

   // If values for Voltage BCs were not set issue a warning and exit
   /*
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
   */
   Vector BVec(3);
   BVec = 0.0; BVec(0) = 0.1;
   VectorConstantCoefficient BCoef(BVec);

   double ion_frac = 0.0;
   ConstantCoefficient rhoCoef1(1.0e19);
   ConstantCoefficient rhoCoef2(rhoCoef1.constant * ion_frac);
   ConstantCoefficient rhoCoef3(rhoCoef1.constant * (1.0 - ion_frac));
   ConstantCoefficient tempCoef(10.0 * q_);

   H1_ParFESpace H1FESpace(&pmesh, order, pmesh.Dimension());
   RT_ParFESpace HDivFESpace(&pmesh, order, pmesh.Dimension());
   L2_ParFESpace L2FESpace(&pmesh, order, pmesh.Dimension());

   ParGridFunction BField(&HDivFESpace);
   ParGridFunction temperature_gf;
   ParGridFunction density_gf;

   BField.ProjectCoefficient(BCoef);

   int size_h1 = H1FESpace.GetVSize();
   int size_l2 = L2FESpace.GetVSize();

   Array<int> density_offsets(nspecies + 2);
   Array<int> temperature_offsets(nspecies + 2);

   density_offsets[0] = 0;
   temperature_offsets[0] = 0;
   for (int i=0; i<=nspecies; i++)
   {
      density_offsets[i + 1]     = density_offsets[i] + size_l2;
      temperature_offsets[i + 1] = temperature_offsets[i] + size_h1;
   }

   BlockVector density(density_offsets);
   BlockVector temperature(temperature_offsets);

   for (int i=0; i<=nspecies; i++)
   {
      temperature_gf.MakeRef(&H1FESpace, temperature.GetBlock(i));
      temperature_gf.ProjectCoefficient(tempCoef);
   }
   density_gf.MakeRef(&L2FESpace, density.GetBlock(0));
   density_gf.ProjectCoefficient(rhoCoef1);

   density_gf.MakeRef(&L2FESpace, density.GetBlock(1));
   density_gf.ProjectCoefficient(rhoCoef2);

   density_gf.MakeRef(&L2FESpace, density.GetBlock(2));
   density_gf.ProjectCoefficient(rhoCoef3);

   //TField  = 10.0*q (10 eV);
   // density = dependent on electron density;

   // Create a coefficient describing the dielectric permittivity
   // Coefficient * epsCoef = SetupPermittivityCoefficient();

   // Create a coefficient describing the magnetic permeability
   Coefficient * muInvCoef = SetupInvPermeabilityCoefficient();

   // Create a tensor coefficient describing the electrical conductivity
   DielectricTensor conductivity_tensor(BField, temperature, density,
                                        H1FESpace, L2FESpace,
                                        nspecies, 2.0 * M_PI * freq_, false);

   // Create a coefficient describing the surface admittance
   Coefficient * etaInvCoef = SetupAdmittanceCoefficient(pmesh, abcs);

   // Create a tensor coefficient describing the dielectric permittivity
   DielectricTensor dielectric_tensor(BField, temperature, density,
                                      H1FESpace, L2FESpace,
                                      nspecies, 2.0 * M_PI * freq_);

   // Create the Magnetostatic solver
   HertzSolver Hertz(pmesh, order, freq_, (HertzSolver::SolverType)sol,
                     conv, dielectric_tensor, *muInvCoef, conductivity_tensor, etaInvCoef,
                     abcs, dbcs,
                     e_bc_r, e_bc_i,
                     (do_params_.Size() > 0 || slab_params_.Size() > 0 ) ? j_src : NULL, NULL
                    );

   //(b_uniform_.Size() > 0 ) ? a_bc_uniform  : NULL,
   //(cr_params_.Size() > 0 ) ? current_ring  : NULL,
   //(bm_params_.Size() > 0 ) ? bar_magnet    :
   //(ha_params_.Size() > 0 ) ? halbach_array : NULL);

   // Initialize GLVis visualization
   if (visualization)
   {
      Hertz.InitializeGLVis();
   }

   // Initialize VisIt visualization
   VisItDataCollection visit_dc("Hertz-AMR-Parallel", &pmesh);

   if ( visit )
   {
      Hertz.RegisterVisItFields(visit_dc);
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
      Hertz.PrintSizes();

      // Assemble all forms
      Hertz.Assemble();

      // Solve the system and compute any auxiliary fields
      Hertz.Solve();

      // Determine the current size of the linear system
      int prob_size = Hertz.GetProblemSize();

      // Write fields to disk for VisIt
      if ( visit )
      {
         Hertz.WriteVisItFields(it);
      }

      // Send the solution by socket to a GLVis server.
      if (visualization)
      {
         Hertz.DisplayToGLVis();
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
      Hertz.GetErrorEstimates(errors);

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
      Hertz.Update();

      if (pmesh.Nonconforming() && mpi.WorldSize() > 1 && false)
      {
         if (mpi.Root()) { cout << "Rebalancing ..." << endl; }
         pmesh.Rebalance();

         // Update again after rebalancing
         Hertz.Update();
      }
   }

   // delete epsCoef;
   delete muInvCoef;
   // delete sigmaCoef;

   return 0;
}

// Print the Hertz ascii logo to the given ostream
void display_banner(ostream & os)
{
   os << "     ____  ____              __           " << endl
      << "    /   / /   / ____________/  |_________ " << endl
      << "   /   /_/   /_/ __ \\_  __ \\   __\\___   / " << endl
      << "  /   __    / \\  ___/|  | \\/|  |  /   _/  " << endl
      << " /___/ /_  /   \\___  >__|   |__| /_____ \\ " << endl
      << "         \\/        \\/                  \\/ " << endl << flush;
}

// The Permittivity is a required coefficient which may be defined in
// various ways so we'll determine the appropriate coefficient type here.
Coefficient *
SetupPermittivityCoefficient()
{
   Coefficient * coef = NULL;

   if ( ds_params_.Size() > 0 )
   {
      coef = new FunctionCoefficient(dielectric_sphere);
   }
   else if ( pw_eps_.Size() > 0 )
   {
      coef = new PWConstCoefficient(pw_eps_);
   }
   else
   {
      coef = new ConstantCoefficient(epsilon0_);
   }

   return coef;
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

// The Conductivity is an optional coefficient which may be defined in
// various ways so we'll determine the appropriate coefficient type here.
Coefficient *
SetupConductivityCoefficient()
{
   Coefficient * coef = NULL;

   if ( cs_params_.Size() > 0 )
   {
      coef = new FunctionCoefficient(conductive_sphere);
   }
   else if ( pw_sigma_.Size() > 0 )
   {
      coef = new PWConstCoefficient(pw_sigma_);
   }

   return coef;
}

// The Admittance is an optional coefficient defined on boundary surfaces which
// can be used in conjunction with absorbing boundary conditions.
Coefficient *
SetupAdmittanceCoefficient(const Mesh & mesh, const Array<int> & abcs)
{
   Coefficient * coef = NULL;

   if ( pw_eta_.Size() > 0 )
   {
      MFEM_VERIFY(pw_eta_.Size() == abcs.Size(),
                  "Each impedance value must be associated with exactly one "
                  "absorbing boundary surface.");

      pw_eta_inv_.SetSize(mesh.bdr_attributes.Size());

      if ( abcs[0] == -1 )
      {
         pw_eta_inv_ = 1.0 / pw_eta_[0];
      }
      else
      {
         pw_eta_inv_ = 0.0;

         for (int i=0; i<pw_eta_.Size(); i++)
         {
            pw_eta_inv_[abcs[i]-1] = 1.0 / pw_eta_[i];
         }
      }
      coef = new PWConstCoefficient(pw_eta_inv_);
   }

   return coef;
}

// A sphere with constant permittivity.  The sphere has a radius,
// center, and permittivity specified on the command line and stored
// in ds_params_.
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

// A sphere with constant conductivity.  The sphere has a radius,
// center, and conductivity specified on the command line and stored
// in ls_params_.
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

// A cylindrical rod of current density.  The rod has two axis end
// points, a radus, a current amplitude in Amperes.  All of these
// parameters are stored in do_params_.
void dipole_oscillator(const Vector &x, Vector &j)
{
   MFEM_ASSERT(x.Size() == 3, "current source requires 3D space.");

   j.SetSize(x.Size());
   j = 0.0;

   Vector  v(x.Size());  // Normalized Axis vector
   Vector xu(x.Size());  // x vector relative to the axis end-point

   xu = x;

   for (int i=0; i<x.Size(); i++)
   {
      xu[i] -= do_params_[i];
      v[i]   = do_params_[x.Size()+i] - do_params_[i];
   }

   double h = v.Norml2();

   if ( h == 0.0 )
   {
      return;
   }
   v /= h;

   double r = do_params_[2*x.Size()+0];
   double a = do_params_[2*x.Size()+1];

   double xv = xu * v;

   // Compute perpendicular vector from axis to x
   xu.Add(-xv, v);

   double xp = xu.Norml2();

   if ( xv >= 0.0 && xv <= h && xp <= r )
   {
      j.Add(a, v);
   }
}

void slab_current_source(const Vector &x, Vector &j)
{
   MFEM_ASSERT(x.Size() == 3, "current source requires 3D space.");

   j.SetSize(x.Size());
   j = 0.0;

   double half_x_l = mesh_dim_(0) * (0.5 - 0.005);
   double half_x_r = mesh_dim_(0) * (0.5 + 0.005);

   if (x(0) <= half_x_r && x(0) >= half_x_l)
   {
      j(0) = slab_params_(0);
      j(1) = slab_params_(1);
      j(2) = slab_params_(2);
   }
}

void e_bc_r(const Vector &x, Vector &E)
{
   E.SetSize(3);
   E = 0.0;

}

void e_bc_i(const Vector &x, Vector &E)
{
   E.SetSize(3);
   E = 0.0;
}

void real_epsilon_sigma(double omega, const Vector &B,
                        const Vector &density_vals,
                        const Vector &temperature_vals,
                        double *real_epsilon, double *real_sigma)
{
   complex<double> I(0.0, 1.0);

   double Bnorm = B.Norml2();
   double phi = 0.0;
   double MData[9] = {cos(phi), 0, -sin(phi), 0.0, 1.0, 0.0, sin(phi), 0, cos(phi)};
   DenseMatrix M(MData, 3, 3);
   Vector Blocal(3);
   M.Mult(B, Blocal);
   double th = atan2(B(2), B(0)), ph = atan2(B(0)*cos(th)+B(2)*sin(th), -B(1));
   double Z1 = 1.0, Z2 = 18.0;
   double qi1 = Z1*qe, qi2 = Z2*qe;
   double mi1 = 2.01410178*u, mi2 = 39.948*u;
   double ne = density_vals[0], ni1 = density_vals[1], ni2 = density_vals[2];
   double Te = temperature_vals[0], Ti = temperature_vals[1];
   double vTe = sqrt(2*Te/me);
   double debye_length = sqrt((epsilon0_*Te)/(ne*pow(qe, 2)));
   double b90_1 = (qe*qi1)/(4*M_PI*epsilon0_*me*pow(vTe, 2)),
          b90_2 = (qe*qi2)/(4*M_PI*epsilon0_*me*pow(vTe, 2));
   double nu_ei1 = (pow(qe, 2)*pow(qi1,
                                   2)*ni1*log(debye_length/b90_1))/(4*M_PI*pow(epsilon0_, 2)*sqrt(me)*pow(Te,
                                                                    3.0/2.0));
   double nu_ei2 = (pow(qe, 2)*pow(qi2,
                                   2)*ni2*log(debye_length/b90_2))/(4*M_PI*pow(epsilon0_, 2)*sqrt(me)*pow(Te,
                                                                    3.0/2.0));

   double wpe = (ne * pow(qe, 2))/(me*epsilon0_); // Squared plasma frequency
   double wpi1 = (ne * pow(qi1, 2))/(mi1*epsilon0_); // Squared plasma frequency
   double wpi2 = (ne * pow(qi2, 2))/(mi2*epsilon0_); // Squared plasma frequency
   double wce = qe*Bnorm/me, wci1 = qi1 * Bnorm/mi1, wci2 = qi2 * Bnorm/mi2;

   double S = (1.0 - wpe/(pow(omega, 2) - pow(wce, 2)) - wpi1/(pow(omega,
                                                                   2)-pow(wci1, 2)) - wpi2/(pow(omega, 2) - pow(wci2, 2)));
   double P = (1.0 - wpe/pow(omega, 2) - wpi1/pow(omega, 2) - wpi2/pow(omega, 2));
   double D = (wce*wpe/(omega*(pow(omega, 2) - pow(wce,
                                                   2))) + wci1*wpi1/(omega*(pow(omega, 2) - pow(wci1,
                                                                            2))) + wci2*wpi2/(omega*(pow(omega, 2) - pow(wci2, 2))));

   double e_xx = (-P * pow(sin(ph), 2)*pow(sin(th), 2) + P*pow(sin(ph),
                                                               2) + S*pow(sin(ph), 2)*pow(sin(th), 2) - S*pow(sin(ph), 2) + S);
   complex<double> e_xy = (I*D*sin(th) + P*cos(ph)*cos(th) - S*cos(ph)*cos(
                              th))*sin(ph);
   complex<double> e_xz = -I*D*cos(ph) + P*pow(sin(ph),
                                               2)*sin(th)*cos(th) - S*pow(sin(ph), 2)*sin(th)*cos(th);

   complex<double> e_yx = -(I*D*sin(th) - P*cos(ph)*cos(th)+S*cos(ph)*cos(
                               th))*sin(ph);
   double e_yy = P*pow(cos(ph), 2)+S*pow(sin(ph), 2);
   complex<double> e_yz = (I*D*cos(th)+P*sin(th)*cos(ph)-S*sin(th)*cos(ph))*sin(
                             ph);

   complex<double> e_zx = I*D*cos(ph) + P*pow(sin(ph),
                                              2)*sin(th)*cos(th) - S*pow(sin(ph), 2)*sin(th)*cos(th);
   complex<double> e_zy = -(I*D*cos(th) - P*sin(th)*cos(ph)+S*sin(th)*cos(
                               ph))*sin(ph);
   double e_zz = P*pow(sin(ph), 2)*pow(sin(th), 2) - S*pow(sin(ph), 2)*pow(sin(th),
                                                                           2) + S;

   if (real_epsilon != NULL)
   {
      real_epsilon[0] = epsilon0_ * e_xx;
      real_epsilon[1] = epsilon0_ * e_yx.real();
      real_epsilon[2] = epsilon0_ * e_zx.real();
      real_epsilon[3] = epsilon0_ * e_xy.real();
      real_epsilon[4] = epsilon0_ * e_yy;
      real_epsilon[5] = epsilon0_ * e_zy.real();
      real_epsilon[6] = epsilon0_ * e_xz.real();
      real_epsilon[7] = epsilon0_ * e_yz.real();
      real_epsilon[8] = epsilon0_ * e_zz;
   }
   if (real_sigma != NULL)
   {
      real_sigma[0] = 0.0;
      real_sigma[1] = e_yx.imag() * omega * epsilon0_;
      real_sigma[2] = e_zx.imag() * omega * epsilon0_;
      real_sigma[3] = e_xy.imag() * omega * epsilon0_;
      real_sigma[4] = 0.0;
      real_sigma[5] = e_zy.imag() * omega * epsilon0_;
      real_sigma[6] = e_xz.imag() * omega * epsilon0_;
      real_sigma[7] = e_yz.imag() * omega * epsilon0_;
      real_sigma[8] = 0.0;
   }

}

DielectricTensor::DielectricTensor(ParGridFunction & B,
                                   BlockVector & temperature,
                                   BlockVector & density,
                                   ParFiniteElementSpace & H1FESpace,
                                   ParFiniteElementSpace & L2FESpace,
                                   int nspecies,
                                   double omega,
                                   bool realPart)
   : MatrixCoefficient(3),
     B_(&B),
     temperature_(&temperature),
     density_(&density),
     H1FESpace_(&H1FESpace),
     L2FESpace_(&L2FESpace),
     nspecies_(nspecies),
     omega_(omega),
     realPart_(realPart)
{
   density_vals_.SetSize(nspecies_ + 1);
   temperature_vals_.SetSize(nspecies_ + 1);
}

void DielectricTensor::Eval(DenseMatrix &epsilon, ElementTransformation &T,
                            const IntegrationPoint &ip)
{
   // Initialize dielectric tensor to all zeros
   epsilon.SetSize(3); epsilon = 0.0;

   // Collect density, temperature, and magnetic field values
   Vector B(3);
   B_->GetVectorValue(T.ElementNo, ip, B);

   for (int i=0; i<=nspecies_; i++)
   {
      density_gf_.MakeRef(L2FESpace_, density_->GetBlock(i));
      temperature_gf_.MakeRef(H1FESpace_, temperature_->GetBlock(i));
      density_vals_[i]     = density_gf_.GetValue(T.ElementNo, ip);
      temperature_vals_[i] = temperature_gf_.GetValue(T.ElementNo, ip);
   }

   if (realPart_)
   {
      // Populate the dielectric tensor
      real_epsilon_sigma(omega_, B, density_vals_, temperature_vals_,
                         epsilon.Data(), NULL);
   }
   else
   {
      // Populate the conductivity tensor
      real_epsilon_sigma(omega_, B, density_vals_, temperature_vals_,
                         NULL, epsilon.Data());

   }
   Vector lambda(3);
   epsilon.Eigenvalues(lambda);
   if (realPart_)
      cout << "Dielectric tensor eigenvalues: "
           << lambda[0] << " " << lambda[1] << " " << lambda[2] << endl;
   else
      cout << "Conductivity tensor eigenvalues: "
           << lambda[0] << " " << lambda[1] << " " << lambda[2] << endl;
}
/*
void DielectricTensor::Dval(DenseMatrix &sigma, ElementTransformation &T,
                            const IntegrationPoint &ip)
{
   // Initialize dielectric tensor to all zeros
   sigma.SetSize(3); sigma = 0.0;

   // Collect density, temperature, and magnetic field values
   Vector B(3);
   B_->GetVectorValue(T.ElementNo, ip, B);

   for (int i=0; i<=nspecies_; i++)
   {
      density_gf_.MakeRef(L2FESpace_, density_->GetBlock(i));
      temperature_gf_.MakeRef(H1FESpace_, temperature_->GetBlock(i));
      density_vals_[i]     = density_gf_.GetValue(T.ElementNo, ip);
      temperature_vals_[i] = temperature_gf_.GetValue(T.ElementNo, ip);
   }
   // Populate the dielectric tensor
   real_epsilon_sigma(omega_, B, density_vals_, temperature_vals_,
                      NULL, sigma.Data());
}
*/

