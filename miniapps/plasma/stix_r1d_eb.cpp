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
//   -----------------------------------------------------------------------
//      Stix_R1D_EB Miniapp: Cold Plasma Electromagnetic Simulation Code
//   -----------------------------------------------------------------------
//
//   Assumes that all sources and boundary conditions oscillate with
//   the same frequency although not necessarily in phase with one
//   another.  This assumption implies that we can factor out the time
//   dependence which we take to be of the form exp(-i omega t).  With
//   these assumptions we can write the Maxwell equations for E and B
//   in the form:
//
//   -i omega epsilon E = Curl mu^{-1} B - J
//    i omega B         = Curl E
//
//   Which combine to yield:
//
//   Curl mu^{-1} Curl E - omega^2 epsilon E = i omega J
//
//   In a cold magnetized plasma the dielectric tensor, epsilon, is
//   complex-valued and anisotropic.  The anisotropy aligns with the
//   external magnetic field and the values depend on the properties
//   of the plasma including the masses and charges of its constituent
//   ion species.
//
//   We have the option of defining a real or complex valued phase
//   vector, beta, which appears in the definition of the electric
//   field as E = \tilde{E} exp(i beta.x). If beta is non-zero the
//   system of equations is modified to solve for \tilde{E} rather
//   than E. In this 1D mini-application this produces no noticeable
//   effect on the solution E. In higher dimensional domains this
//   enables support for imposing a known phase difference across
//   periodic domains.
//
//   For a magnetic field aligned with the z-axis the dielectric
//   tensor has the form:
//
//                | S -iD 0 |
//      epsilon = |iD  S  0 |
//                | 0  0  P |
//
//   In this example we will use only the simplest coefficients;
//   constants or linear functions of position.
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
//               + i omega (W, n x H)_{\Gamma}
//
//   The resulting surface integral, (W, n x H)_{\Gamma}, can be used
//   to impose surface current boundary conditions which are an
//   example of Neumann boundary conditions. This term can also be
//   used to impose a first order Sommerfeld absorbing boundary
//   condition. The code also supports fixing the values of the
//   electric field on certain boundaries as Dirichlet (or essential)
//   boundary conditions.
//
//   This mini-application makes use of adaptive mesh refinement (AMR)
//   in two manners. First, it adaptively refines the mesh to capture
//   variations of the dielectric tensor due to changes in the ion
//   densities (or temperatures). Second, it further refines the mesh
//   to reduce errors in the computed electric field.
//
//   (By default the sources, boundary conditions, and fields are all zero)
//
// Compile with: make stix_r1d_eb
//
// Sample runs:
//
// Any of the following sample runs could be run in parallel but as
// they are 1D examples this is hardly necessary.
//
// Dirichlet BCs:
//   ./stix_r1d_eb -w R -dbcs 1 -abcs 2 -rs 4 -s 4 -vis -f 16e6
//   ./stix_r1d_eb -w L -dbcs 1 -abcs 2 -rs 4 -s 4 -vis -f 4e6
//   ./stix_r1d_eb -w O -dbcs 1 -abcs 2 -rs 4 -s 4 -vis -f 1
//   ./stix_r1d_eb -w X -dbcs 1 -abcs 2 -rs 4 -s 4 -vis -f 8e6
//
// Neumann BCs:
//   ./stix_r1d_eb -w R -nbcs 1 -abcs 2 -rs 4 -s 4 -vis -f 16e6
//   ./stix_r1d_eb -w L -nbcs 1 -abcs 2 -rs 4 -s 4 -vis -f 4e6
//   ./stix_r1d_eb -w O -nbcs 1 -abcs 2 -rs 4 -s 4 -vis -f 1
//   ./stix_r1d_eb -w X -nbcs 1 -abcs 2 -rs 4 -s 4 -vis -f 8e6
//
// Current Source:
//   ./stix_r1d_eb -w R -slab '0.4 0.2' -abcs '1 2' -rs 4 -s 4 -vis -f 16e6
//   ./stix_r1d_eb -w L -slab '0.4 0.2' -abcs '1 2' -rs 4 -s 4 -vis -f 4e6
//   ./stix_r1d_eb -w O -slab '0.4 0.2' -abcs '1 2' -rs 4 -s 4 -vis -f 1
//   ./stix_r1d_eb -w X -slab '0.4 0.2' -abcs '1 2' -rs 4 -s 4 -vis -f 8e6
//
//   ./stix_r1d_eb -w R -slab '0.4 0.2' -jp 1 -abcs '1 2' -rs 4 -s 4 -vis -f 16e6
//   ./stix_r1d_eb -w L -slab '0.4 0.2' -jp 1 -abcs '1 2' -rs 4 -s 4 -vis -f 4e6
//   ./stix_r1d_eb -w O -slab '0.4 0.2' -jp 1 -abcs '1 2' -rs 4 -s 4 -vis -f 1
//   ./stix_r1d_eb -w X -slab '0.4 0.2' -jp 1 -abcs '1 2' -rs 4 -s 4 -vis -f 8e6
//
// Variable Plasma Density:
//   ./stix_r1d_eb -w R -dbcs 1 -abcs 2 -rs 4 -s 4 -vis -f 8e7 -dp 1 -dpp '1e16 0 0 0 5e19 0 0' -viid
//   ./stix_r1d_eb -w R -dbcs 1 -abcs 2 -rs 4 -s 4 -vis -f 16e6 -dp 2 -dpp '1e16 1e19 0.01 0.5 0 0 1 0 0' -viid
//   ./stix_r1d_eb -w R -dbcs 1 -abcs 2 -rs 4 -s 4 -vis -f 16e6 -ic '1 1' -im '1 2' -dp '1 2' -dpp '1e19 0 0 0 -999e16 0 0 1e16 1e19 0.01 0.5 0 0 1 0 0' -viid
//
// Adaptive Mesh Refinement:
//   ./stix_r1d_eb -w R -dbcs 1 -abcs 2 -rs 2 -s 4 -vis -f 16e6 -ic '1 1' -im '1 2' -dp '1 2' -dpp '1e19 0 0 0 -999e16 0 0 1e16 1e19 0.01 0.5 0 0 1 0 0' -viid -visisp -iamr ISP -iatol 1e-4
//

#include "cold_plasma_dielectric_coefs.hpp"
#include "cold_plasma_dielectric_eb_solver.hpp"
#include "../common/mesh_extras.hpp"
#include <fstream>
#include <iostream>
#include <cmath>
#include <complex>

using namespace std;
using namespace mfem;
using namespace mfem::common;
using namespace mfem::plasma;

// Create density and temperature profiles for each ion species and
// electrons.  The number of ion species is determined by the lengths
// of the `charges` and `masses` vectors. The density and temperature
// profiles are either constant or defined by user input. The electron
// density is computed to create a neutral plasma. The electron
// temperature is set so that the electron energy equals the average
// of the ion energies.
void SetDensityAndTempProfiles(Vector &charges, Vector &masses,
                               Array<int> &dpt, Vector &dpp,
                               Array<int> &tpt, Vector &tpp);

// Adaptively refine the initial mesh in order to adequately capture
// features of the Stix coefficients. The `coef` argument selects the
// function used to guide the refinement. Choices for `coef` are: S,
// D, P, L, R, or ISP.
void AdaptInitialMesh(ParMesh &pmesh,
                      int order, int p, StixParams &stixParams,
                      const char * coef, real_t tol, int max_its, int max_dofs,
                      bool visualization);

// Wrapper for the exact solution fields and sources
class Stix1DExactSol : public VectorCoefficient
{
public:
   enum FieldType {E_FIELD, H_FIELD, J_SOURCE};

private:
   FieldType field_;
   char type_;
   Vector kVec_;
   StixParams &stixParams_;
   StixCoef::ReImPart re_im_part_;

   real_t j_pos_;
   real_t j_dx_;
   int    j_prof_;

   ColdPlasmaPlaneWaveBase *coef_;

   void initCoef()
   {
      delete coef_;

      switch (field_)
      {
         case E_FIELD:
            if (j_prof_ < 0)
            {
               coef_ = new ColdPlasmaPlaneWaveE(kVec_, type_,
                                                stixParams_, re_im_part_);
            }
            else
            {
               coef_ = new ColdPlasmaCenterFeedE(kVec_, type_,
                                                 j_pos_, j_dx_, j_prof_,
                                                 stixParams_, re_im_part_);
            }
            break;
         case H_FIELD:
            if (j_prof_ < 0)
            {
               coef_ = new ColdPlasmaPlaneWaveH(kVec_, type_,
                                                stixParams_, re_im_part_);
            }
            else
            {
               coef_ = new ColdPlasmaCenterFeedH(kVec_, type_,
                                                 j_pos_, j_dx_, j_prof_,
                                                 stixParams_, re_im_part_);
            }
            break;
         case J_SOURCE:
            if (j_prof_ < 0)
            {
               coef_ = NULL;
            }
            else
            {
               coef_ = new ColdPlasmaCenterFeedJ(kVec_, type_,
                                                 j_pos_, j_dx_, j_prof_,
                                                 stixParams_, re_im_part_);
            }
            break;
      }
   }

public:
   Stix1DExactSol(FieldType field, Vector kVec, char type,
                  StixParams & stixParams,
                  StixCoef::ReImPart re_im_part)
      : VectorCoefficient(3),
        field_(field),
        type_(type),
        kVec_(kVec),
        stixParams_(stixParams),
        re_im_part_(re_im_part),
        j_pos_(0.0),
        j_dx_(0.0),
        j_prof_(-1),
        coef_(NULL)
   {
      initCoef();
   }

   ~Stix1DExactSol() { delete coef_; }

   void SetCurrentSlab(real_t j_pos, real_t j_dx, int j_profile)
   { j_pos_ = j_pos; j_dx_ = j_dx; j_prof_ = j_profile; initCoef(); }

   void SetPhaseShift(const Vector & beta_r,
                      const Vector & beta_i)
   { coef_->SetPhaseShift(beta_r, beta_i); }

   void ClearPhaseShift()
   { coef_->ClearPhaseShift(); }

   void InvertPhaseShift()
   { coef_->InvertPhaseShift(); }

   void Eval(Vector &V, ElementTransformation &T,
             const IntegrationPoint &ip)
   {
      if (coef_ != NULL )
      {
         coef_->Eval(V, T, ip);
      }
      else
      {
         V.SetSize(3); V = 0.0;
      }
   }

};


// Prints the program's logo to the given output stream
void display_banner(ostream & os);

// Prints a copy of the program's command line to a text file
void record_cmd_line(int argc, char *argv[]);

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   if (!Mpi::Root()) { mfem::out.Disable(); mfem::err.Disable(); }

   display_banner(mfem::out);

   if (Mpi::Root()) { record_cmd_line(argc, argv); }

   int logging = 1;

   // Parse command-line options.
   const char *init_amr = "";
   real_t init_amr_tol = 1e-3;
   int init_amr_max_its = 10;
   int init_amr_max_dofs = 100000;
   int ser_ref_levels = 0;
   int par_ref_levels = 0;
   int order = 2;
   int maxit = 1;
   int sol = 2;
   int prec = 1;
   bool herm_conv = false;
   bool exact_sol = false;
   bool visualization = false;

   real_t freq = 1.0e6;
   Vector phaseVec(0);
   Vector kVec({1.0, 0.0, 0.0});

   // Type of plane wave being simulated
   char wave_type = 'Z';

   // Current Density Function
   // Position in 1D, and size in 1D
   Vector slab_params(0);
   // Piecewise constant or a single sinusoidal bump
   int slab_profile = 0;

   Vector charges(1); charges[0] = 1.0;
   Vector masses(1); masses[0] = 2.01410178;

   PlasmaProfile::Type nept = PlasmaProfile::CONSTANT;
   PlasmaProfile::Type nipt = PlasmaProfile::CONSTANT;
   BFieldProfile::Type bpt  = BFieldProfile::CONSTANT;

   Array<int> dpt(1); dpt = PlasmaProfile::CONSTANT;
   Array<int> tpt(1); tpt = PlasmaProfile::CONSTANT;

   Vector dpp(1); dpp = 1e19;
   Vector tpp(1); tpp = 1e3;
   Vector bpp;
   Vector nepp;
   Vector nipp;
   int nuprof = 0;
   real_t res_lim = 0.01;

   Array<int> abcs; // Absorbing BC attributes
   Array<int> peca; // Perfect Electric Conductor BC attributes
   Array<int> dbca; // Dirichlet BC attributes
   Array<int> nbca; // Neumann BC attributes
   Array<int> jsrca; // Current Source region attributes

   Array<char> abct; // Absorbing BC plane wave types

   int num_elements = 10;
   real_t length_x = 1.0;

   CPDSolverEB::SolverOptions solOpts;
   solOpts.maxIter = 1000;
   solOpts.kDim = 50;
   solOpts.printLvl = 1;
   solOpts.relTol = 1e-4;
   solOpts.euLvl = 1;

   Array<bool>  inputVisOpts = CPDInputVis::GetDefaultVisFlags();
   Array<bool>  fieldVisOpts = CPDFieldVis::GetDefaultVisFlags();
   Array<bool> outputVisOpts = CPDOutputVis::GetDefaultVisFlags();
   Array<bool> fieldAnimOpts = CPDFieldAnim::GetDefaultVisFlags();

   bool logo = false;
   const char *device_config = "cpu";

   OptionsParser args(argc, argv);
   args.AddOption(&logo, "-logo", "--print-logo", "-no-logo",
                  "--no-print-logo", "Print logo and exit.");
   args.AddOption(&num_elements, "-ne", "--num-elems",
                  "Number of elements in initial mesh.");
   args.AddOption(&length_x, "-Lx", "--length-x",
                  "Length of the mesh in the x direction.");
   args.AddOption(&ser_ref_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&par_ref_levels, "-rp", "--refine-parallel",
                  "Number of times to refine the mesh uniformly in parallel.");
   args.AddOption(&init_amr, "-iamr", "--init-amr",
                  "Initial AMR to capture Stix coefficient: S, D, P, L, R"
                  " or 1 / (S P).");
   args.AddOption(&init_amr_tol, "-iatol", "--init-amr-tol",
                  "Initial AMR tolerance.");
   args.AddOption(&init_amr_max_its, "-iamit", "--init-amr-max-its",
                  "Initial AMR Maximum Number of Iterations.");
   args.AddOption(&init_amr_max_dofs, "-iamdof", "--init-amr-max-dofs",
                  "Initial AMR Maximum Number of DoFs.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&freq, "-f", "--frequency",
                  "Frequency in Hertz (of course...)");
   args.AddOption(&phaseVec, "-p-vec", "--phase-vector",
                  "Phase shift vector across periodic directions."
                  " For complex phase shifts input 3 real phase shifts "
                  "followed by 3 imaginary phase shifts");
   args.AddOption(&dpt, "-dp", "--density-profile",
                  "Density Profile Type (for each ion species, "
                  "electrons will be set to enforce charge neutrality): \n"
                  "0 - Constant, 1 - Constant Gradient, "
                  "2 - Hyprebolic Tangent.");
   args.AddOption(&dpp, "-dpp", "--density-profile-params",
                  "Density Profile Parameters:\n"
                  "   CONSTANT: density value\n"
                  "   GRADIENT: value, location, gradient (7 params)\n"
                  "   TANH:     starting value, final value, skin depth, "
                  "location of mid point, unit vector along gradient.");
   args.AddOption((int*)&bpt, "-bp", "--Bfield-profile",
                  "BField Profile Type: \n"
                  "0 - Constant.");
   args.AddOption(&bpp, "-bpp", "--Bfield-profile-params",
                  "BField Profile Parameters:\n"
                  "  Bx, By, Bz.");
   args.AddOption(&tpt, "-tp", "--temperature-profile",
                  "Temperature Profile Type "
                  "(for each ion species and electrons): \n"
                  "0 - Constant, 1 - Constant Gradient, "
                  "2 - Hyperbolic Tangent.");
   args.AddOption(&tpp, "-tpp", "--temperature-profile-params",
                  "Temperature Profile Parameters: \n"
                  "   CONSTANT: temperature value \n"
                  "   GRADIENT: value, location, gradient (7 params)\n"
                  "   TANH:     starting value, final value, skin depth, "
                  "location of mid point, unit vector along gradient.");
   args.AddOption((int*)&nept, "-nep", "--electron-collision-profile",
                  "Electron Collisions Profile Type: \n"
                  "0 - Constant, 1 - Constant Gradient, "
                  "2 - Hyperbolic Tangent.");
   args.AddOption(&nepp, "-nepp", "--electron-collisions-profile-params",
                  "Electron Collisions Profile Parameters: \n"
                  "   CONSTANT: temperature value \n"
                  "   GRADIENT: value, location, gradient (7 params)\n"
                  "   TANH:     starting value, final value, skin depth, "
                  "location of mid point, unit vector along gradient.");
   args.AddOption((int*)&nipt, "-nip", "--ion-collision-profile",
                  "Ion Collisions Profile Type: \n"
                  "0 - Constant, 1 - Constant Gradient, "
                  "2 - Hyperbolic Tangent.");
   args.AddOption(&nipp, "-nipp", "--ion-collisions-profile-params",
                  "Ion Collisions Profile Parameters: \n"
                  "   CONSTANT: temperature value \n"
                  "   GRADIENT: value, location, gradient (7 params)\n"
                  "   TANH:     starting value, final value, skin depth, "
                  "location of mid point, unit vector along gradient.");
   args.AddOption(&nuprof, "-nuprof", "--collisional-profile",
                  "Temperature Profile Type: \n"
                  "0 - Standard e-i Collision Freq, 1 - Custom Freq.");
   args.AddOption(&res_lim, "-res-lim", "--resonance-limiter",
                  "Resonance limit factor [0,1).");
   args.AddOption(&wave_type, "-w", "--wave-type",
                  "Wave type: 'R' - Right Circularly Polarized, "
                  "'L' - Left Circularly Polarized, "
                  "'O' - Ordinary, 'X' - Extraordinary, "
                  "'Z' - Zero");
   args.AddOption(&charges, "-ic", "--ion-charges",
                  "Charges of the various ion species "
                  "(in units of electron charge)");
   args.AddOption(&masses, "-im", "--ion-masses",
                  "Masses of the various ion species "
                  "(in atomic mass units)");
   args.AddOption(&prec, "-pc", "--precond",
                  "Preconditioner: 1 - Diagonal Scaling, 2 - ParaSails, "
                  "3 - Euclid, 4 - AMS");
   args.AddOption(&sol, "-s", "--solver",
                  "Solver: 1 - GMRES, 2 - FGMRES, 3 - MINRES"
#ifdef MFEM_USE_SUPERLU
                  ", 4 - SuperLU"
#endif
#ifdef MFEM_USE_STRUMPACK
                  ", 5 - STRUMPACK"
#endif
#ifdef MFEM_USE_MUMPS
                  ", 6 - MUMPS"
#endif
                 );
   args.AddOption(&solOpts.maxIter, "-sol-it", "--solver-iterations",
                  "Maximum number of solver iterations.");
   args.AddOption(&solOpts.kDim, "-sol-k-dim", "--solver-krylov-dimension",
                  "Krylov space dimension for GMRES and FGMRES.");
   args.AddOption(&solOpts.relTol, "-sol-tol", "--solver-tolerance",
                  "Relative tolerance for GMRES or FGMRES.");
   args.AddOption(&solOpts.printLvl, "-sol-prnt-lvl", "--solver-print-level",
                  "Logging level for solvers.");
   args.AddOption(&solOpts.euLvl, "-eu-lvl", "--euclid-level",
                  "Euclid factorization level for ILU(k).");
   args.AddOption(&jsrca, "-jsrc", "--j-src-reg",
                  "Current source region attributes");
   args.AddOption(&slab_params, "-slab", "--slab_params",
                  "3D Vector Amplitude (Real x,y,z, Imag x,y,z), "
                  "1D Midpoint Position, 1D Size");
   args.AddOption(&slab_profile, "-jp", "--current-profile",
                  "0 (Constant) or 1 (Sin Function)");
   args.AddOption(&abcs, "-abcs", "--absorbing-bc-surf",
                  "Absorbing Boundary Condition Surfaces");
   args.AddOption(&abct, "-abct", "--absorbing-bc-types",
                  "Absorbing Boundary Condition Plane Wave Types");
   args.AddOption(&peca, "-pecs", "--pec-bc-surf",
                  "Perfect Electrical Conductor Boundary Condition Surfaces");
   args.AddOption(&dbca, "-dbcs", "--dirichlet-bc-surf",
                  "Dirichlet Boundary Condition Surfaces");
   args.AddOption(&nbca, "-nbcs", "--neumann-bc-surf",
                  "Neumann Boundary Condition Surfaces");
   args.AddOption(&maxit, "-maxit", "--max-amr-iterations",
                  "Max number of iterations in the main AMR loop.");
   args.AddOption(&herm_conv, "-herm", "--hermitian", "-no-herm",
                  "--no-hermitian", "Use convention for Hermitian operators.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   CPDInputVis::AddOptions(args, inputVisOpts);
   CPDFieldVis::AddOptions(args, fieldVisOpts);
   CPDOutputVis::AddOptions(args, outputVisOpts);
   CPDFieldAnim::AddOptions(args, fieldAnimOpts);
   args.Parse();
   if (!args.Good())
   {
      if (Mpi::Root())
      {
         args.PrintUsage(cout);
      }
      return 1;
   }
   if (logo)
   {
      return 1;
   }
   Device device(device_config);
   if (Mpi::Root())
   {
      device.Print();
   }
   SetDensityAndTempProfiles(charges, masses, dpt, dpp, tpt, tpp);
   if (nepp.Size() == 0)
   {
      nepp.SetSize(1);
      nepp[0] = 0;
   }
   if (nipp.Size() == 0)
   {
      nipp.SetSize(1);
      nipp[0] = 0;
   }
   if (bpt == BFieldProfile::CONSTANT && bpp.Size() == 0)
   {
      bpp.SetSize(3);
      bpp = 0.0;
      if (wave_type == 'O' || wave_type == 'X' )
      {
         bpp[1] = 1.0;
      }
      else
      {
         bpp[0] = 1.0;
      }
   }
   if (num_elements <= 0)
   {
      num_elements = 10;
   }
   real_t omega = 2.0 * M_PI * freq;
   bool phase_shift = false;
   if (phaseVec.Size() != 0)
   {
      phase_shift = true;
   }

   if (Mpi::Root())
   {
      args.PrintOptions(cout);
   }

   ComplexOperator::Convention conv =
      herm_conv ? ComplexOperator::HERMITIAN : ComplexOperator::BLOCK_SYMMETRIC;

   // Read the (serial) mesh from the given mesh file on all processors.
   if ( Mpi::Root() && logging > 0 )
   {
      cout << "Building 1D Mesh ..." << endl;
   }

   tic_toc.Clear();
   tic_toc.Start();

   Mesh mesh = Mesh::MakeCartesian1D(num_elements, length_x);

   if (Mpi::Root())
   {
      cout << "Created mesh object with element attributes: ";
      mesh.attributes.Print(cout);
      cout << "and boundary attributes: ";
      mesh.bdr_attributes.Print(cout);
   }
   for (int lev = 0; lev < ser_ref_levels; lev++)
   {
      mesh.UniformRefinement();
   }
   tic_toc.Stop();

   if (Mpi::Root() && logging > 0 )
   {
      cout << " done in " << tic_toc.RealTime() << " seconds." << endl;
   }

   // Ensure that quad meshes are treated as non-conforming.
   mesh.EnsureNCMesh();

   // Define a parallel mesh by a partitioning of the serial mesh. Refine
   // this mesh further in parallel to increase the resolution. Once the
   // parallel mesh is defined, the serial mesh can be deleted.
   if ( Mpi::Root() && logging > 0 )
   { cout << "Building Parallel Mesh ..." << endl; }
   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();

   for (int lev = 0; lev < par_ref_levels; lev++)
   {
      pmesh.UniformRefinement();
   }

   if (Mpi::Root())
   {
      cout << "Starting initialization." << endl;
   }

   PlasmaProfile nueCoef(nept, nepp);
   PlasmaProfile nuiCoef(nipt, nipp);

   BFieldProfile BCoef(bpt, bpp, false);
   BFieldProfile BUnitCoef(bpt, bpp, true);

   if (Mpi::Root())
   {
      cout << "Creating plasma profile." << endl;
   }

   MultiSpeciesPlasmaProfiles temperatureCoef(tpt, tpp);
   MultiSpeciesPlasmaProfiles densityCoef(dpt, dpp, charges);

   StixParams stixParams(BCoef, nueCoef, nuiCoef, densityCoef, temperatureCoef,
                         omega, charges, masses, nuprof, res_lim);

   if (strcmp(init_amr,""))
   {
      AdaptInitialMesh(pmesh, order, 2, stixParams,
                       init_amr, init_amr_tol, init_amr_max_its,
                       init_amr_max_dofs,
                       visualization);
   }

   Vector phaseReVec(3); phaseReVec = 0.0;
   Vector phaseImVec(3); phaseImVec = 0.0;
   if (phase_shift)
   {
      if (phaseVec.Size() >= 3)
      {
         phaseReVec.SetDataAndSize(&phaseVec[0], 3);
      }
      if (phaseVec.Size() >= 6)
      {
         phaseImVec.SetDataAndSize(&phaseVec[3], 3);
      }

      mfem::out << "Setting phase shift of ("
                << complex_t(phaseReVec[0],phaseImVec[0]) << ","
                << complex_t(phaseReVec[1],phaseImVec[1]) << ","
                << complex_t(phaseReVec[2],phaseImVec[2]) << ")" << endl;

   }
   VectorConstantCoefficient phaseReCoef(phaseReVec);
   VectorConstantCoefficient phaseImCoef(phaseImVec);

   if (dpt.Size() == 2 && dpt[0] == 0 ) { exact_sol = true; }
   if (Mpi::Root() && exact_sol)
   {
      cout << "Creating exact solution coefficients." << endl;
   }


   Stix1DExactSol EReCoef(Stix1DExactSol::E_FIELD, kVec, wave_type,
                          stixParams, StixCoef::REAL_PART);
   Stix1DExactSol EImCoef(Stix1DExactSol::E_FIELD, kVec, wave_type,
                          stixParams, StixCoef::IMAG_PART);
   Stix1DExactSol HReCoef(Stix1DExactSol::H_FIELD, kVec, wave_type,
                          stixParams, StixCoef::REAL_PART);
   Stix1DExactSol HImCoef(Stix1DExactSol::H_FIELD, kVec, wave_type,
                          stixParams, StixCoef::IMAG_PART);
   ScalarVectorProductCoefficient BReCoef(mu0_, HReCoef);
   ScalarVectorProductCoefficient BImCoef(mu0_, HImCoef);

   if (Mpi::Root() && slab_params.Size() > 0)
   {
      cout << "Setup current source." << endl;
   }
   if (slab_params.Size() > 0)
   {
      EReCoef.SetCurrentSlab(slab_params[0], slab_params[1], slab_profile);
      EImCoef.SetCurrentSlab(slab_params[0], slab_params[1], slab_profile);
      HReCoef.SetCurrentSlab(slab_params[0], slab_params[1], slab_profile);
      HImCoef.SetCurrentSlab(slab_params[0], slab_params[1], slab_profile);
   }
   if (phase_shift)
   {
      EReCoef.SetPhaseShift(phaseReVec, phaseImVec);
      EImCoef.SetPhaseShift(phaseReVec, phaseImVec);
      HReCoef.SetPhaseShift(phaseReVec, phaseImVec);
      HImCoef.SetPhaseShift(phaseReVec, phaseImVec);
   }


   if (Mpi::Root())
   {
      cout << "Setup boundary conditions." << endl;
   }

   // Setup coefficients for Dirichlet BC
   int dbcsSize = (peca.Size() > 0) + (dbca.Size() > 0);

   StixBCs stixBCs(pmesh.attributes, pmesh.bdr_attributes);

   Vector zeroVec(3); zeroVec = 0.0;
   VectorConstantCoefficient zeroCoef(zeroVec);

   if (dbcsSize > 0)
   {
      if (peca.Size() > 0)
      {
         stixBCs.AddDirichletBC(peca, zeroCoef, zeroCoef);
      }
      if (dbca.Size() > 0)
      {
         stixBCs.AddDirichletBC(dbca, EReCoef, EImCoef);
      }
   }

   int nbcsSize = nbca.Size() > 0;

   if (nbcsSize > 0)
   {
      stixBCs.AddNeumannBC(nbca, HReCoef, HImCoef);
   }

   Array<Coefficient*> etaInvReCoef(abcs.Size());
   Array<Coefficient*> etaInvImCoef(abcs.Size());
   if (abcs.Size() > 0)
   {
      if (abct.Size() < abcs.Size())
      {
         abct.SetSize(abcs.Size());
         abct = wave_type;
      }

      for (int i=0; i<abcs.Size(); i++)
      {
         Array<int> bdr(1);
         bdr = abcs[i];

         etaInvReCoef[i] = new StixAdmittanceCoef(abct[i], stixParams,
                                                  StixCoef::REAL_PART);
         etaInvImCoef[i] = new StixAdmittanceCoef(abct[i], stixParams,
                                                  StixCoef::IMAG_PART);

         stixBCs.AddSommerfeldBC(bdr, *etaInvReCoef[i], *etaInvImCoef[i]);
      }
   }

   // Note that the phase shift, if present, is applied to the current
   // source as part of the PDE so these coefficients do not need to
   // have the phase shift applied.
   Stix1DExactSol JReCoef(Stix1DExactSol::J_SOURCE, kVec, wave_type,
                          stixParams, StixCoef::REAL_PART);
   Stix1DExactSol JImCoef(Stix1DExactSol::J_SOURCE, kVec, wave_type,
                          stixParams, StixCoef::IMAG_PART);
   if (slab_params.Size() > 0)
   {
      if (Mpi::Root())
      {
         cout << "Adding volumetric current source." << endl;
      }
      if (jsrca.Size() == 0)
      {
         jsrca = pmesh.attributes;
      }
      JReCoef.SetCurrentSlab(slab_params[0], slab_params[1], slab_profile);
      JImCoef.SetCurrentSlab(slab_params[0], slab_params[1], slab_profile);
      stixBCs.AddCurrentSrc(jsrca, JReCoef, JImCoef);
   }

   if (Mpi::Root())
   {
      cout << "Creating Cold Plasma Dielectric solver." << endl;
   }

   // Create the cold plasma EM solver
   CPDSolverEB CPD(pmesh, order,
                   (CPDSolverEB::SolverType)sol, solOpts,
                   (CPDSolverEB::PrecondType)prec,
                   conv, stixParams,
                   (phase_shift) ? &phaseReCoef : NULL,
                   (phase_shift) ? &phaseImCoef : NULL,
                   stixBCs);

   // Set the visualization flags
   CPD.GetInputVis().SetOptions(inputVisOpts);
   CPD.GetFieldVis().SetOptions(fieldVisOpts);
   CPD.GetOutputVis().SetOptions(outputVisOpts);
   CPD.GetFieldAnim().SetOptions(fieldAnimOpts);

   // Initialize visualization objects for exact solutions
   ComplexVectorFieldVisObject VisExactE("Exact E",
                                         CPD.GetVectorVisFES());

   ComplexVectorFieldVisObject VisExactB("Exact B",
                                         CPD.GetVectorVisFES());

   if (Mpi::Root()) { cout << "Initialization done." << endl; }

   // The main AMR loop. In each iteration we solve the problem on the current
   // mesh, visualize the solution, estimate the error on all elements, refine
   // the worst elements and update all objects to work with the new mesh. We
   // refine until the maximum number of dofs in the Nedelec finite element
   // space reaches 10 million.
   const int max_dofs = 10000000;
   for (int it = 1; it <= maxit; it++)
   {
      if (Mpi::Root())
      {
         cout << "\nAMR Iteration " << it << endl;
      }

      // Display the current number of DoFs in each finite element space
      CPD.PrintSizes();

      // Assemble all forms
      CPD.Assemble();

      // Solve the system and compute any auxiliary fields
      CPD.Solve();

      if (exact_sol)
      {
         real_t glb_error_E = CPD.GetEFieldError(EReCoef, EImCoef);
         real_t glb_error_B = CPD.GetBFieldError(BReCoef, BImCoef);
         if (Mpi::Root())
         {
            cout << "Global L2 Error in E field " << glb_error_E << endl;
            cout << "Global L2 Error in B field " << glb_error_B << endl;
         }

         if (visualization)
         {
            if (CPD.GetFieldVis().CheckVisFlag(CPDFieldVis::ELECTRIC_FIELD))
            {
               VisExactE.PrepareVisField(EReCoef, EImCoef,
                                         (phase_shift) ? &phaseReCoef : NULL,
                                         (phase_shift) ? &phaseImCoef : NULL);
               VisExactE.DisplayToGLVis();
            }
            if (CPD.GetFieldVis().CheckVisFlag(CPDFieldVis::MAGNETIC_FLUX))
            {
               VisExactB.PrepareVisField(BReCoef, BImCoef,
                                         (phase_shift) ? &phaseReCoef : NULL,
                                         (phase_shift) ? &phaseImCoef : NULL);
               VisExactB.DisplayToGLVis();
            }
         }
      }

      // Send the solution by socket to a GLVis server.
      if (visualization)
      {
         CPD.DisplayToGLVis();
      }

      if (Mpi::Root())
      {
         cout << "AMR iteration " << it << " complete." << endl;
      }

      // Determine the current size of the linear system
      int prob_size = CPD.GetProblemSize();

      // Check stopping criteria
      if (prob_size > max_dofs)
      {
         if (Mpi::Root())
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
      if (Mpi::Root() && (it % 50 == 0))
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
      CPD.GetErrorEstimates(errors);

      real_t local_max_err = errors.Max();
      real_t global_max_err;
      MPI_Allreduce(&local_max_err, &global_max_err, 1,
                    MPITypeMap<real_t>::mpi_type, MPI_MAX, pmesh.GetComm());

      // Refine the elements whose error is larger than a fraction of the
      // maximum element error.
      const real_t frac = 0.5;
      real_t threshold = frac * global_max_err;
      if (Mpi::Root()) { cout << "Refining ..." << endl; }
      {
         pmesh.RefineByError(errors, threshold);
      }

      // Update the electromagnetic solver to reflect the new state of the mesh.
      CPD.Update();
      VisExactE.Update();
      VisExactB.Update();

      if (pmesh.Nonconforming() && Mpi::WorldSize() > 1 && false)
      {
         if (Mpi::Root()) { cout << "Rebalancing ..." << endl; }
         pmesh.Rebalance();

         // Update again after rebalancing
         CPD.Update();
         VisExactE.Update();
         VisExactB.Update();
      }
   }

   return 0;
}

void SetDensityAndTempProfiles(Vector &charges, Vector &masses,
                               Array<int> &dpt, Vector &dpp,
                               Array<int> &tpt, Vector &tpp)
{
   int num_ion_spec = 0;
   cout << "Initial Masses:  "; masses.Print(cout);
   cout << "Initial Charges: "; charges.Print(cout);

   // Check masses of species
   if (std::abs(masses[masses.Size() - 1] - me_u_) < me_u_ * 1e-4)
   {
      // Vector already contains the electron mass at the end
      num_ion_spec = masses.Size() - 1;
   }
   else
   {
      // Append electron mass to the vector of masses
      num_ion_spec = masses.Size();
      Vector ion_masses(masses);
      masses.SetSize(num_ion_spec + 1);
      masses.SetVector(ion_masses, 0);
      masses[num_ion_spec] = me_u_;
   }
   cout << "Number of ion species: " << num_ion_spec << endl;
   cout << "Masses: "; masses.Print(cout);

   // Check species charges
   if (charges.Size() == num_ion_spec)
   {
      Vector ion_charges(charges);
      charges.SetSize(num_ion_spec + 1);
      charges.SetVector(ion_charges, 0);
      charges[num_ion_spec] = -1.0;
   }
   cout << "Charges: "; charges.Print(cout);
   MFEM_VERIFY(charges.Size() == num_ion_spec + 1 &&
               std::abs(charges[charges.Size() - 1] + 1.0) < 1e-4,
               "The number and types of charges are incompatible "
               "with the species masses.");

   // Check density profiles
   if (dpt.Size() == 1 && dpt.Size() < num_ion_spec)
   {
      Vector den_params(dpp);
      dpp.SetSize(num_ion_spec * den_params.Size());
      dpp.SetVector(den_params, 0);
      int o = den_params.Size();
      for (int i=1; i<num_ion_spec; i++)
      {
         dpt.Append(dpt[0]);
         dpp.SetVector(den_params, o);
         o += den_params.Size();
      }
   }
   if (dpt.Size() == num_ion_spec)
   {
      dpt.Append(PlasmaProfile::NEUTRALITY);
   }
   MFEM_VERIFY(dpt.Size() == num_ion_spec + 1,
               "The number of denisty profiles is incompatible "
               "with the species masses.");

   // Check temperature profiles
   if (tpt.Size() == 1 && tpt.Size() < num_ion_spec)
   {
      Vector temp_params(tpp);
      tpp.SetSize(num_ion_spec * temp_params.Size());
      tpp.SetVector(temp_params, 0);
      int o = temp_params.Size();
      for (int i=1; i<num_ion_spec; i++)
      {
         tpt.Append(dpt[0]);
         tpp.SetVector(temp_params, o);
         o += temp_params.Size();
      }
   }
   if (tpt.Size() == num_ion_spec)
   {
      tpt.Append(PlasmaProfile::AVERAGE);
   }
   MFEM_VERIFY(tpt.Size() == num_ion_spec + 1,
               "The number of temperature profiles is incompatible "
               "with the species masses.");
}

void AdaptInitialMesh(ParMesh &pmesh,
                      int order, int p, StixParams &stixParams,
                      const char * coef, real_t tol, int max_its, int max_dofs,
                      bool visualization)
{
   if (strcmp(coef,"S") && strcmp(coef,"D") && strcmp(coef,"P") &&
       strcmp(coef,"L") && strcmp(coef,"R") && strcmp(coef,"ISP"))
   {
      if (Mpi::Root())
      {
         cout << "Unrecognized parameter for initial AMR loop '"
              << coef << "' coefficient." << endl;
      }
      return;
   }
   if (Mpi::Root())
   {
      if (strcmp(coef,"ISP"))
      {
         cout << "Adapting mesh to Stix '" << coef << "' coefficient."
              << endl;
      }
      else
      {
         cout << "Adapting mesh to Stix coefficient function '1/(SP)'."
              << endl;
      }
   }

   Coefficient *ReCoefPtr = NULL;
   Coefficient *ImCoefPtr = NULL;
   if (!strcmp(coef,"S"))
   {
      ReCoefPtr = new StixSCoef(stixParams, StixCoef::REAL_PART);
      ImCoefPtr = new StixSCoef(stixParams, StixCoef::IMAG_PART);
   }
   else if (!strcmp(coef,"D"))
   {
      ReCoefPtr = new StixDCoef(stixParams, StixCoef::REAL_PART);
      ImCoefPtr = new StixDCoef(stixParams, StixCoef::IMAG_PART);
   }
   else if (!strcmp(coef,"P"))
   {
      ReCoefPtr = new StixPCoef(stixParams, StixCoef::REAL_PART);
      ImCoefPtr = new StixPCoef(stixParams, StixCoef::IMAG_PART);
   }
   else if (!strcmp(coef,"L"))
   {
      ReCoefPtr = new StixLCoef(stixParams, StixCoef::REAL_PART);
      ImCoefPtr = new StixLCoef(stixParams, StixCoef::IMAG_PART);
   }
   else if (!strcmp(coef,"R"))
   {
      ReCoefPtr = new StixRCoef(stixParams, StixCoef::REAL_PART);
      ImCoefPtr = new StixRCoef(stixParams, StixCoef::IMAG_PART);
   }
   else if (!strcmp(coef,"ISP"))
   {
      ReCoefPtr = new StixInvSPCoef(stixParams, StixCoef::REAL_PART);
      ImCoefPtr = new StixInvSPCoef(stixParams, StixCoef::IMAG_PART);
   }

   ConstantCoefficient zeroCoef(0.0);

   L2_ParFESpace L2FESpace(&pmesh, order, pmesh.Dimension());
   ParComplexGridFunction gf(&L2FESpace);

   ComplexLpErrorEstimator estimator(p, *ReCoefPtr, *ImCoefPtr, gf, true);

   ThresholdRefiner refiner(estimator);
   refiner.SetTotalErrorFraction(0);
   refiner.SetTotalErrorNormP(p);
   refiner.SetLocalErrorGoal(tol);

   for (int it = 0; it < max_its; it++)
   {
      HYPRE_Int global_dofs = L2FESpace.GlobalTrueVSize();
      if (Mpi::Root())
      {
         cout << "\nAMR iteration " << it << endl;
         cout << "Number of L2 unknowns: " << global_dofs << endl;
      }

      gf.ProjectCoefficient(*ReCoefPtr, *ImCoefPtr);

      real_t l2_nrm = gf.ComputeL2Error(zeroCoef, zeroCoef);
      real_t l2_err = gf.ComputeL2Error(*ReCoefPtr, *ImCoefPtr);
      if (Mpi::Root())
      {
         if (l2_nrm > 0.0)
         {
            cout << "Relative L2 Error: " << l2_err << " / " << l2_nrm
                 << " = " << l2_err / l2_nrm << endl;
         }
         else
         {
            cout << "L2 Error: " << l2_err << endl;
         }
      }

      if (global_dofs > max_dofs)
      {
         if (Mpi::Root())
         {
            cout << "Reached the maximum number of dofs. Stop." << endl;
         }
         break;
      }

      // Call the refiner to modify the mesh. The refiner calls the error
      // estimator to obtain element errors, then it selects elements to be
      // refined and finally it modifies the mesh. The Stop() method can be
      // used to determine if a stopping criterion was met.
      refiner.Apply(pmesh);
      if (refiner.Stop())
      {
         if (Mpi::Root())
         {
            cout << "Stopping criterion satisfied. Stop." << endl;
         }
         break;
      }

      // Update the finite element space (recalculate the number of DOFs,
      // etc.)
      L2FESpace.Update();

      // Load balance the mesh, and update the space and solution. Currently
      // available only for nonconforming meshes.
      if (pmesh.Nonconforming())
      {
         pmesh.Rebalance();

         // Update the space.
         L2FESpace.Update();
      }
      gf.SetSpace(&L2FESpace);
   }
   if (Mpi::Root())
   {
      cout << endl;
   }

   delete ReCoefPtr;
   delete ImCoefPtr;
}

const char * banner[6] =
{
   R"(  _________ __   __       __________  ____     ________________________ )",
   R"( /   _____//  |_|__|__  __\______   \/_   | __| _/\_   _____/\______   \)",
   R"( \_____  \\   __\  \  \/  /|       _/ |   |/ __ |  |    __)_  |    |  _/)",
   R"( /        \|  | |  |>    < |    |   \ |   / /_/ |  |        \ |    |   \)",
   R"(/_______  /|__| |__/__/\_ \|____|_  / |___\____ | /_______  / |______  /)",
   R"(        \/               \/       \/           \/         \/         \/ )"
};

// Print the stix_r2d_eb ascii logo to the given ostream
void display_banner(ostream & os)
{
   for (int i=0; i<6; i++)
   {
      os << banner[i] << endl;
   }
   os << endl
      << "* Thomas H. Stix was a pioneer in the use of radio frequency"
      << " waves to heat" << endl
      << "  terrestrial plasmas to solar temperatures. He made important"
      << " contributions" << endl
      << "  to experimental and theoretic plasma physics. In the Stix"
      << " applications, the" << endl
      << "  plasma dielectric for the wave equation is formulated using"
      << " the \"Stix\"" << endl
      << "  notation, \"S, D, P\"." << endl<< endl << flush;
}

void record_cmd_line(int argc, char *argv[])
{
   ofstream ofs("stix_r1d_eb_cmd.txt");

   // We need to include single quotes around arguments which may
   // contain white space such as arguments corresponding to Vector or
   // Array objects.

   for (int i=0; i<argc; i++)
   {
      ofs << argv[i] << " ";
      if (strcmp(argv[i], "-im"     ) == 0 ||
          strcmp(argv[i], "-ic"     ) == 0 ||
          strcmp(argv[i], "-bpp"    ) == 0 ||
          strcmp(argv[i], "-dp"     ) == 0 ||
          strcmp(argv[i], "-dpp"     ) == 0 ||
          strcmp(argv[i], "-tp"    ) == 0 ||
          strcmp(argv[i], "-tpp"    ) == 0 ||
          strcmp(argv[i], "-nepp"   ) == 0 ||
          strcmp(argv[i], "-nipp"   ) == 0 ||
          strcmp(argv[i], "-p-vec"  ) == 0 ||
          strcmp(argv[i], "-jsrc"   ) == 0 ||
          strcmp(argv[i], "-slab"   ) == 0 ||
          strcmp(argv[i], "-abcs"   ) == 0 ||
          strcmp(argv[i], "-abct"   ) == 0 ||
          strcmp(argv[i], "-pecs"   ) == 0 ||
          strcmp(argv[i], "-dbcs"   ) == 0 ||
          strcmp(argv[i], "-nbcs"   ) == 0)
      {
         ofs << "'" << argv[++i] << "' ";
      }
   }
   ofs << endl << flush;
   ofs.close();
}
