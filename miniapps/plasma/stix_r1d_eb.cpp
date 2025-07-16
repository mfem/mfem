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
//   Assumes that all sources and boundary conditions oscillate with the same
//   frequency although not necessarily in phase with one another.  This
//   assumption implies that we can factor out the time dependence which we
//   take to be of the form exp(-i omega t).  With these assumptions we can
//   write the Maxwell equations for E and B in the form:
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
//   For a magnetic field aligned with the z-axis the dielectric tensor has
//   the form:
//              | S  -iD 0 |
//    epsilon = |iD   S  0 |
//              | 0   0  P |
//
//   In this example we will use only the simplest coefficients; constants
//   or linear functions of position.
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
// (By default the sources and fields are all zero)
//
// Compile with: make stix_r1d_eb
//
// Sample runs:
//   ./stix_r1d_eb -dbcs1 1 -dbcv1 '0 0 1 0 -1 0' -abcs '2' -abct 'R' -rs 4 -s 4 -vis -f 16e6
//   ./stix_r1d_eb -dbcs1 1 -dbcv1 '0 0 1 0 1 0' -abcs '2' -abct 'L' -rs 4 -s 4 -vis -f 4e6
//   ./stix_r1d_eb -dbcs1 1 -dbcv1 '0 0 1 1 0 0' -abcs '2' -abct 'X' -rs 4 -s 4 -vis -f 8e6 -bpp '0 1 0'
//   ./stix_r1d_eb -dbcs1 1 -dbcv1 '0 1 0' -abcs '2' -abct 'O' -rs 4 -s 4 -vis -f 1 -bpp '0 1 0'
//   ./stix_r1d_eb -dbcs1 1 -dbcv1 '0 0 1 0 -1 0' -abcs '2' -abct 'R' -rs 4 -s 4 -vis -f 16e6 -dp '1 1' -dpp '1e19 0.5 0 0 2e19 0 0 1e19 0.5 0 0 2e19 0 0'
//   ./stix_r1d_eb -dbcs1 1 -dbcv1 '0 0 1 0 1 0' -abcs '2' -abct 'L' -rs 4 -s 4 -vis -f 4e6 -dp '1 1' -dpp '1e19 0.5 0 0 2e19 0 0 1e19 0.5 0 0 2e19 0 0'
//   ./stix_r1d_eb -dbcs1 1 -dbcv1 '0 0 1 1 0 0' -abcs '2' -abct 'X' -rs 4 -s 4 -vis -f 8e6 -bpp '0 1 0' -dp '1 1' -dpp '1e19 0.5 0 0 2e19 0 0 1e19 0.5 0 0 2e19 0 0'
//   ./stix_r1d_eb -dbcs1 1 -dbcv1 '0 1 0' -abcs '2' -abct 'O' -rs 4 -s 4 -vis -f 1 -bpp '0 1 0' -dp '1 1' -dpp '1e19 0.5 0 0 2e19 0 0 1e19 0.5 0 0 2e19 0 0'
//  ./stix_r1d_eb -abcs '1 2' -abct 'R R' -slab '0 0 1 0 -1 0 0.5 0.05' -rs 4 -s 4 -vis -f 16e6
// Parallel sample runs:
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

void AdaptInitialMesh(ParMesh &pmesh,
                      int order, int p, StixParams &stixParams,
		      const char * coef, real_t tol, int max_its, int max_dofs,
                      bool visualization);

// Current Density Function
// Amplitude of x, y, z current source, position in 1D, and size in 1D
static Vector slab_params_(0);
// Piecewise constant or a single sinusoidal bump
static int slab_profile_ = 0;

void slab_current_source_r(const Vector &x, Vector &j);
void slab_current_source_i(const Vector &x, Vector &j);
void j_src_r(const Vector &x, Vector &j)
{
   if (slab_params_.Size() > 0)
   {
      slab_current_source_r(x, j);
   }
}
void j_src_i(const Vector &x, Vector &j)
{
   if (slab_params_.Size() > 0)
   {
      slab_current_source_i(x, j);
   }
}

// Electric Field Boundary Condition: The following function returns zero but
// any function could be used.
void e_bc_r(const Vector &x, Vector &E);
void e_bc_i(const Vector &x, Vector &E);

class ColdPlasmaPlaneWaveH: public VectorCoefficient
{
public:
   ColdPlasmaPlaneWaveH(char type,
                        real_t omega,
                        const Vector & B,
                        const Vector & number,
                        const Vector & charge,
                        const Vector & mass,
                        const Vector & temp,
                        int nuprof,
                        real_t res_lim,
                        bool realPart);

   void SetCurrentSlab(real_t Jy, real_t xJ, real_t delta, real_t Lx)
   { Jy_ = Jy; xJ_ = xJ; dx_ = delta, Lx_ = Lx; }

   void SetPhaseShift(const Vector & beta)
   { beta_r_ = beta; beta_i_ = 0.0; }
   void SetPhaseShift(const Vector & beta_r,
                      const Vector & beta_i)
   { beta_r_ = beta_r; beta_i_ = beta_i; }

   void GetWaveVector(Vector & k_r, Vector & k_i) const
   { k_r = k_r_; k_i = k_i_; }

   void Eval(Vector &V, ElementTransformation &T,
             const IntegrationPoint &ip);

private:
   char type_;
   bool realPart_;
   int nuprof_;
   real_t res_lim_;
   real_t omega_;
   real_t Bmag_;
   real_t Jy_;
   real_t xJ_;
   real_t dx_;
   real_t Lx_;
   complex<real_t> kappa_;
   Vector b_;   // Normalized vector in direction of B
   Vector bc_;  // Normalized vector perpendicular to b_, (by-bz,bz-bx,bx-by)
   Vector bcc_; // Normalized vector perpendicular to b_ and bc_
   Vector h_r_;
   Vector h_i_;
   Vector k_r_;
   Vector k_i_;
   Vector beta_r_;
   Vector beta_i_;

   // const Vector & B_;
   const Vector & numbers_;
   const Vector & charges_;
   const Vector & masses_;
   const Vector & temps_;

   complex<real_t> S_;
   complex<real_t> D_;
   complex<real_t> P_;
};

class ColdPlasmaPlaneWaveE: public StixCoefBase, public VectorCoefficient
{
public:
   ColdPlasmaPlaneWaveE(char type,
			StixParams & stixParams,
                        const Vector & B,
                        const Vector & number,
                        const Vector & charge,
                        const Vector & mass,
                        const Vector & temp,
                        ReImPart re_im_part);

   void SetCurrentSlab(real_t Jy, real_t xJ, real_t delta, real_t Lx)
   { Jy_ = Jy; xJ_ = xJ; dx_ = delta, Lx_ = Lx; }

   void SetPhaseShift(const Vector & beta)
   { beta_r_ = beta; beta_i_ = 0.0; }
   void SetPhaseShift(const Vector & beta_r,
                      const Vector & beta_i)
   { beta_r_ = beta_r; beta_i_ = beta_i; }

   void GetWaveVector(Vector & k_r, Vector & k_i) const
   { k_r = k_r_; k_i = k_i_; }

   void Eval(Vector &V, ElementTransformation &T,
             const IntegrationPoint &ip);

private:
   char type_;
   bool realPart_;
   int nuprof_;
   real_t res_lim_;
   real_t omega_;
   real_t Bmag_;
   real_t Jy_;
   real_t xJ_;
   real_t dx_;
   real_t Lx_;
   complex<real_t> kappa_;
   Vector b_;   // Normalized vector in direction of B
   Vector bc_;  // Normalized vector perpendicular to b_, (by-bz,bz-bx,bx-by)
   Vector bcc_; // Normalized vector perpendicular to b_ and bc_
   Vector e_r_;
   Vector e_i_;
   Vector k_r_;
   Vector k_i_;
   Vector beta_r_;
   Vector beta_i_;

   // const Vector & B_;
   const Vector & numbers_;
   const Vector & charges_;
   const Vector & masses_;
   const Vector & temps_;

   complex<real_t> S_;
   complex<real_t> D_;
   complex<real_t> P_;
};

// Prints the program's logo to the given output stream
void display_banner(ostream & os);

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
   int order = 1;
   int maxit = 1;
   int sol = 2;
   int prec = 1;
   bool herm_conv = false;
   bool exact_sol = false;
   bool vis_u = false;
   bool visualization = false;
   bool visit = false;

   real_t freq = 1.0e6;
   const char * wave_type = " ";

   bool phase_shift = false;
   Vector kVec;
   Vector kReVec;
   Vector kImVec;

   Vector charges(2); charges[0] = -1.0; charges[1] = 1.0;
   Vector masses(2); masses[0] = me_u_; masses[1] = 2.01410178;

   PlasmaProfile::Type nept = PlasmaProfile::CONSTANT;
   PlasmaProfile::Type nipt = PlasmaProfile::CONSTANT;
   BFieldProfile::Type bpt  = BFieldProfile::CONSTANT;

   Array<int> dpt(2); dpt = PlasmaProfile::CONSTANT;
   Array<int> tpt(2); tpt = PlasmaProfile::CONSTANT;

   Vector dpp(2); dpp = 1e19;
   Vector tpp(2); tpp = 1e3;
   Vector bpp;
   Vector nepp;
   Vector nipp;
   int nuprof = 0;
   real_t res_lim = 0.01;

   Array<int> abcs; // Absorbing BC attributes
   Array<int> peca; // Perfect Electric Conductor BC attributes
   Array<int> dbca1; // Dirichlet BC attributes
   Array<int> dbca2; // Dirichlet BC attributes
   Array<int> nbca; // Neumann BC attributes
   Array<int> nbca1; // Neumann BC attributes
   Array<int> nbca2; // Neumann BC attributes
   Array<int> jsrca; // Current Source region attributes

   Vector dbcv1; // Dirichlet BC values
   Vector dbcv2; // Dirichlet BC values
   Vector nbcv; // Neumann BC values
   Vector nbcv1; // Neumann BC values
   Vector nbcv2; // Neumann BC values
   Array<char> abct; // Absorbing BC plane wave types
   
   int msa_n = 0;
   Vector msa_p(0);

   int num_elements = 10;
   real_t length_x = 1.0;
   
   CPDSolverEB::SolverOptions solOpts;
   solOpts.maxIter = 1000;
   solOpts.kDim = 50;
   solOpts.printLvl = 1;
   solOpts.relTol = 1e-4;
   solOpts.euLvl = 1;

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
                  "Initial AMR to capture Stix coefficient: S, D, L, or R.");
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
   args.AddOption(&dpt, "-dp", "--density-profile",
                  "Density Profile Type (for ions): \n"
                  "0 - Constant, 1 - Constant Gradient, "
                  "2 - Hyprebolic Tangent, 3 - Elliptic Cosine.");
   args.AddOption(&dpp, "-dpp", "--density-profile-params",
                  "Density Profile Parameters:\n"
                  "   CONSTANT: density value\n"
                  "   GRADIENT: value, location, gradient (7 params)\n"
                  "   TANH:     value at 0, value at 1, skin depth, "
                  "location of 0 point, unit vector along gradient, "
                  "   ELLIPTIC_COS: value at -1, value at 1, "
                  "radius in x, radius in y, location of center.");
   args.AddOption((int*)&bpt, "-bp", "--Bfield-profile",
                  "BField Profile Type: \n"
                  "0 - Constant, 1 - Constant Gradient, "
                  "2 - Hyprebolic Tangent, 3 - Elliptic Cosine.");
   args.AddOption(&bpp, "-bpp", "--Bfield-profile-params",
                  "BField Profile Parameters:\n"
                  "  B_P: value at -1, value at 1, "
                  "radius in x, radius in y, location of center, Bz, placeholder.");
   args.AddOption(&tpt, "-tp", "--temperature-profile",
                  "Temperature Profile Type: \n"
                  "0 - Constant, 1 - Constant Gradient, "
                  "2 - Hyperbolic Tangent, 3 - Elliptic Cosine.");
   args.AddOption(&tpp, "-tpp", "--temperature-profile-params",
                  "Temperature Profile Parameters: \n"
                  "   CONSTANT: temperature value \n"
                  "   GRADIENT: value, location, gradient (7 params)\n"
                  "   TANH:     value at 0, value at 1, skin depth, "
                  "location of 0 point, unit vector along gradient, "
                  "   ELLIPTIC_COS: value at -1, value at 1, "
                  "radius in x, radius in y, location of center.");
   args.AddOption((int*)&nept, "-nep", "--electron-collision-profile",
                  "Electron Collisions Profile Type: \n"
                  "0 - Constant, 1 - Constant Gradient, "
                  "2 - Hyperbolic Tangent, 3 - Elliptic Cosine.");
   args.AddOption(&nepp, "-nepp", "--electron-collisions-profile-params",
                  "Electron Collisions Profile Parameters: \n"
                  "   CONSTANT: temperature value \n"
                  "   GRADIENT: value, location, gradient (7 params)\n"
                  "   TANH:     value at 0, value at 1, skin depth, "
                  "location of 0 point, unit vector along gradient, "
                  "   ELLIPTIC_COS: value at -1, value at 1, "
                  "radius in x, radius in y, location of center.");
   args.AddOption((int*)&nipt, "-nip", "--ion-collision-profile",
                  "Ion Collisions Profile Type: \n"
                  "0 - Constant, 1 - Constant Gradient, "
                  "2 - Hyperbolic Tangent, 3 - Elliptic Cosine.");
   args.AddOption(&nipp, "-nipp", "--ion-collisions-profile-params",
                  "Ion Collisions Profile Parameters: \n"
                  "   CONSTANT: temperature value \n"
                  "   GRADIENT: value, location, gradient (7 params)\n"
                  "   TANH:     value at 0, value at 1, skin depth, "
                  "location of 0 point, unit vector along gradient, "
                  "   ELLIPTIC_COS: value at -1, value at 1, "
                  "radius in x, radius in y, location of center.");
   args.AddOption(&nuprof, "-nuprof", "--collisional-profile",
                  "Temperature Profile Type: \n"
                  "0 - Standard e-i Collision Freq, 1 - Custom Freq.");
   args.AddOption(&res_lim, "-res-lim", "--resonance-limiter",
                  "Resonance limit factor [0,1).");
   args.AddOption(&wave_type, "-w", "--wave-type",
                  "Wave type: 'R' - Right Circularly Polarized, "
                  "'L' - Left Circularly Polarized, "
                  "'O' - Ordinary, 'X' - Extraordinary, "
                  "'J' - Current Slab (in conjunction with -slab), "
                  "'Z' - Zero");
   args.AddOption(&kVec, "-k-vec", "--phase-vector",
                  "Phase shift vector across periodic directions."
                  " For complex phase shifts input 3 real phase shifts "
                  "followed by 3 imaginary phase shifts");
   args.AddOption(&msa_n, "-ns", "--num-straps","");
   args.AddOption(&msa_p, "-sp", "--strap-params","");
   args.AddOption(&charges, "-q", "--charges",
                  "Charges of the various species "
                  "(in units of electron charge)");
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
   args.AddOption(&slab_params_, "-slab", "--slab_params",
                  "3D Vector Amplitude (Real x,y,z, Imag x,y,z), "
                  "1D Midpoint Position, 1D Size");
   args.AddOption(&slab_profile_, "-slab-prof", "--slab_profile",
                  "0 (Constant) or 1 (Sin Function)");
   args.AddOption(&abcs, "-abcs", "--absorbing-bc-surf",
                  "Absorbing Boundary Condition Surfaces");
   args.AddOption(&abct, "-abct", "--absorbing-bc-types",
                  "Absorbing Boundary Condition Plane Wave Types");
   args.AddOption(&peca, "-pecs", "--pec-bc-surf",
                  "Perfect Electrical Conductor Boundary Condition Surfaces");
   args.AddOption(&dbca1, "-dbcs1", "--dirichlet-bc-1-surf",
                  "Dirichlet Boundary Condition Surfaces Using Value 1");
   args.AddOption(&dbca2, "-dbcs2", "--dirichlet-bc-2-surf",
                  "Dirichlet Boundary Condition Surfaces Using Value 2");
   args.AddOption(&dbcv1, "-dbcv1", "--dirichlet-bc-1-vals",
                  "Dirichlet Boundary Condition Value 1 (v_x v_y v_z)"
                  " or (Re(v_x) Re(v_y) Re(v_z) Im(v_x) Im(v_y) Im(v_z))");
   args.AddOption(&dbcv2, "-dbcv2", "--dirichlet-bc-2-vals",
                  "Dirichlet Boundary Condition Value 2 (v_x v_y v_z)"
                  " or (Re(v_x) Re(v_y) Re(v_z) Im(v_x) Im(v_y) Im(v_z))");
   args.AddOption(&nbca, "-nbcs", "--neumann-bc-surf",
                  "Neumann Boundary Condition Surfaces Using Piecewise Values");
   args.AddOption(&nbca1, "-nbcs1", "--neumann-bc-1-surf",
                  "Neumann Boundary Condition Surfaces Using Value 1");
   args.AddOption(&nbca2, "-nbcs2", "--neumann-bc-2-surf",
                  "Neumann Boundary Condition Surfaces Using Value 2");
   args.AddOption(&nbcv, "-nbcv", "--neumann-bc-vals",
                  "Neuamnn Boundary Condition (surface current) "
                  "Six values per boundary attribute: "
                  "(Re(v_x) Re(v_y) Re(v_z) Im(v_x) Im(v_y) Im(v_z)) ...");
   args.AddOption(&nbcv1, "-nbcv1", "--neumann-bc-1-vals",
                  "Neuamnn Boundary Condition (surface current) "
                  "Value 1 (v_x v_y v_z) or "
                  "(Re(v_x) Re(v_y) Re(v_z) Im(v_x) Im(v_y) Im(v_z))");
   args.AddOption(&nbcv2, "-nbcv2", "--neumann-bc-2-vals",
                  "Neumann Boundary Condition (surface current) "
                  "Value 2 (v_x v_y v_z) or "
                  "(Re(v_x) Re(v_y) Re(v_z) Im(v_x) Im(v_y) Im(v_z))");
   args.AddOption(&maxit, "-maxit", "--max-amr-iterations",
                  "Max number of iterations in the main AMR loop.");
   args.AddOption(&herm_conv, "-herm", "--hermitian", "-no-herm",
                  "--no-hermitian", "Use convention for Hermitian operators.");
   args.AddOption(&vis_u, "-vis-u", "--visualize-energy", "-no-vis-u",
                  "--no-visualize-energy",
                  "Enable or disable visualization of energy density.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&visit, "-visit", "--visit", "-no-visit", "--no-visit",
                  "Enable or disable VisIt visualization.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
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
   if (dpp.Size() == 0)
   {
      dpp.SetSize(1);
      dpp[0] = 1.0e19;
   }
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
      bpp[0] = 1.0;
   }
   if (num_elements <= 0)
   {
      num_elements = 10;
   }
   real_t omega = 2.0 * M_PI * freq;
   if (kVec.Size() != 0)
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

   VisItDataCollection visit_dc(MPI_COMM_WORLD, "STIX-R1D-EB-AMR-Parallel",
                                &pmesh);

   PlasmaProfile nueCoef(nept, nepp);
   PlasmaProfile nuiCoef(nipt, nipp);

   BFieldProfile BCoef(bpt, bpp, false);
   BFieldProfile BUnitCoef(bpt, bpp, true);

   if (Mpi::Root())
   {
      cout << "Creating plasma profile." << endl;
   }

   MultiSpeciesPlasmaProfiles temperatureCoef(tpt, tpp);
   MultiSpeciesPlasmaProfiles densityCoef(dpt, dpp);

   StixParams stixParams(BCoef, nueCoef, nuiCoef, densityCoef, temperatureCoef,
			 omega, charges, masses, nuprof, res_lim);
   
   if (strcmp(init_amr,""))
   {
      AdaptInitialMesh(pmesh, order, 2, stixParams,
                       init_amr, init_amr_tol, init_amr_max_its,
		       init_amr_max_dofs,
                       visualization);
   }

   if (Mpi::Root())
   {
      cout << "Creating phase shift vectors." << endl;
   }

   if (kVec.Size() >= 3)
   {
     kReVec.SetDataAndSize(&kVec[0], 3);
   }
   else
   {
     kReVec.SetSize(3);
     kReVec = 0.0;
   }
   if (kVec.Size() >= 6)
     {
       kImVec.SetDataAndSize(&kVec[3], 3);
     }
   else
     {
       kImVec.SetSize(3);
       kImVec = 0.0;
     }

   mfem::out << "Setting phase shift of ("
             << complex<real_t>(kReVec[0],kImVec[0]) << ","
             << complex<real_t>(kReVec[1],kImVec[1]) << ","
             << complex<real_t>(kReVec[2],kImVec[2]) << ")" << endl;

   VectorConstantCoefficient kReCoef(kReVec);
   VectorConstantCoefficient kImCoef(kImVec);

   /*
   if (visualization && wave_type[0] != ' ')
   {
      if (Mpi::Root())
      {
         cout << "Visualize input fields." << endl;
      }
      ParComplexGridFunction HField(&HCurlFESpace);
      HField.ProjectCoefficient(HReCoef, HImCoef);
      ParComplexGridFunction EField(&HCurlFESpace);
      EField.ProjectCoefficient(EReCoef, EImCoef);

      Vector zeroVec(3); zeroVec = 0.0;
      VectorConstantCoefficient zeroCoef(zeroVec);
      real_t max_Hr = HField.real().ComputeMaxError(zeroCoef);
      real_t max_Hi = HField.imag().ComputeMaxError(zeroCoef);
      real_t max_Er = EField.real().ComputeMaxError(zeroCoef);
      real_t max_Ei = EField.imag().ComputeMaxError(zeroCoef);

      char vishost[] = "localhost";
      int  visport   = 19916;

      int Wx = 0, Wy = 0; // window position
      int Ww = 350, Wh = 350; // window size
      int offx = Ww+10, offy = Wh+45; // window offsets

      socketstream sock_Hr, sock_Hi, sock_Er, sock_Ei, sock_B;
      sock_Hr.precision(8);
      sock_Hi.precision(8);
      sock_Er.precision(8);
      sock_Ei.precision(8);
      sock_B.precision(8);

      ostringstream hr_keys, hi_keys;
      hr_keys << "aaAcPPPPvvv valuerange 0.0 " << max_Hr;
      hi_keys << "aaAcPPPPvvv valuerange 0.0 " << max_Hi;

      ostringstream er_keys, ei_keys;
      er_keys << "aaAcpppppvvv valuerange 0.0 " << max_Er;
      ei_keys << "aaAcpppppvvv valuerange 0.0 " << max_Ei;

      Wy += offy;
      VisualizeField(sock_Hr, vishost, visport,
                     HField.real(), "Exact Magnetic Field, Re(H)",
                     Wx, Wy, Ww, Wh, hr_keys.str().c_str());

      Wx += offx;
      VisualizeField(sock_Hi, vishost, visport,
                     HField.imag(), "Exact Magnetic Field, Im(H)",
                     Wx, Wy, Ww, Wh, hi_keys.str().c_str());

      Wx += offx;
      VisualizeField(sock_Er, vishost, visport,
                     EField.real(), "Exact Electric Field, Re(E)",
                     Wx, Wy, Ww, Wh, er_keys.str().c_str());
      Wx += offx;
      VisualizeField(sock_Ei, vishost, visport,
                     EField.imag(), "Exact Electric Field, Im(E)",
                     Wx, Wy, Ww, Wh, ei_keys.str().c_str());

      Wx -= offx;
      Wy += offy;

      VisualizeField(sock_B, vishost, visport,
                     BField, "Background Magnetic Field",
                     Wx, Wy, Ww, Wh);

      for (int i=0; i<charges.Size(); i++)
      {
         Wx += offx;

         socketstream sock;
         sock.precision(8);

         stringstream oss;
         oss << "Density Species " << i;
         density_gf.MakeRef(&L2FESpace, density.GetBlock(i));
         VisualizeField(sock, vishost, visport,
                        density_gf, oss.str().c_str(),
                        Wx, Wy, Ww, Wh);
      }
   }
   */
   if (Mpi::Root())
   {
      cout << "Setup boundary conditions." << endl;
   }

   // Setup coefficients for Dirichlet BC
   int dbcsSize = (peca.Size() > 0) + (dbca1.Size() > 0) + (dbca2.Size() > 0);

   StixBCs stixBCs(pmesh.attributes, pmesh.bdr_attributes);

   Vector zeroVec(3); zeroVec = 0.0;
   Vector dbc1ReVec;
   Vector dbc1ImVec;
   Vector dbc2ReVec;
   Vector dbc2ImVec;

   if (dbcv1.Size() >= 3)
   {
      dbc1ReVec.SetDataAndSize(&dbcv1[0], 3);
   }
   else
   {
      dbc1ReVec.SetDataAndSize(&zeroVec[0], 3);
   }
   if (dbcv1.Size() >= 6)
   {
      dbc1ImVec.SetDataAndSize(&dbcv1[3], 3);
   }
   else
   {
      dbc1ImVec.SetDataAndSize(&zeroVec[0], 3);
   }
   if (dbcv2.Size() >= 3)
   {
      dbc2ReVec.SetDataAndSize(&dbcv2[0], 3);
   }
   else
   {
      dbc2ReVec.SetDataAndSize(&zeroVec[0], 3);
   }
   if (dbcv2.Size() >= 6)
   {
      dbc2ImVec.SetDataAndSize(&dbcv2[3], 3);
   }
   else
   {
      dbc2ImVec.SetDataAndSize(&zeroVec[0], 3);
   }

   VectorConstantCoefficient zeroCoef(zeroVec);
   VectorConstantCoefficient dbc1ReCoef(dbc1ReVec);
   VectorConstantCoefficient dbc1ImCoef(dbc1ImVec);
   VectorConstantCoefficient dbc2ReCoef(dbc2ReVec);
   VectorConstantCoefficient dbc2ImCoef(dbc2ImVec);

   if (dbcsSize > 0)
   {
      if (peca.Size() > 0)
      {
         stixBCs.AddDirichletBC(peca, zeroCoef, zeroCoef);
      }
      if (dbca1.Size() > 0)
      {
         stixBCs.AddDirichletBC(dbca1, dbc1ReCoef, dbc1ImCoef);
      }
      if (dbca2.Size() > 0)
      {
         stixBCs.AddDirichletBC(dbca2, dbc2ReCoef, dbc2ImCoef);
      }
   }

   int nbcsSize = (nbca.Size() > 0) + (nbca1.Size() > 0) +
                  (nbca2.Size() > 0);

   Vector nbc1ReVec;
   Vector nbc1ImVec;
   Vector nbc2ReVec;
   Vector nbc2ImVec;

   if (nbcv1.Size() >= 3)
   {
      nbc1ReVec.SetDataAndSize(&nbcv1[0], 3);
   }
   else
   {
      nbc1ReVec.SetDataAndSize(&zeroVec[0], 3);
   }
   if (nbcv1.Size() >= 6)
   {
      nbc1ImVec.SetDataAndSize(&nbcv1[3], 3);
   }
   else
   {
      nbc1ImVec.SetDataAndSize(&zeroVec[0], 3);
   }
   if (nbcv2.Size() >= 3)
   {
      nbc2ReVec.SetDataAndSize(&nbcv2[0], 3);
   }
   else
   {
      nbc2ReVec.SetDataAndSize(&zeroVec[0], 3);
   }
   if (nbcv2.Size() >= 6)
   {
      nbc2ImVec.SetDataAndSize(&nbcv2[3], 3);
   }
   else
   {
      nbc2ImVec.SetDataAndSize(&zeroVec[0], 3);
   }

   Array<VectorConstantCoefficient*> nbcVecs(2 * nbca.Size());
   VectorConstantCoefficient nbc1ReCoef(nbc1ReVec);
   VectorConstantCoefficient nbc1ImCoef(nbc1ImVec);
   VectorConstantCoefficient nbc2ReCoef(nbc2ReVec);
   VectorConstantCoefficient nbc2ImCoef(nbc2ImVec);

   if (nbcsSize > 0)
   {
      if (nbca.Size() > 0)
      {
         Array<int> nbca0(1);
         Vector nbc0ReVec, nbc0ImVec;
         for (int i=0; i<nbca.Size(); i++)
         {
            nbca0[0] = nbca[i];

            nbc0ReVec.SetDataAndSize(&nbcv[6*i+0], 3);
            nbc0ImVec.SetDataAndSize(&nbcv[6*i+3], 3);

            nbcVecs[2*i+0] = new VectorConstantCoefficient(nbc0ReVec);
            nbcVecs[2*i+1] = new VectorConstantCoefficient(nbc0ImVec);

            stixBCs.AddNeumannBC(nbca0, *nbcVecs[2*i+0], *nbcVecs[2*i+1]);
         }
      }
      if (nbca1.Size() > 0)
      {
         stixBCs.AddNeumannBC(nbca1, nbc1ReCoef, nbc1ImCoef);
      }
      if (nbca2.Size() > 0)
      {
         stixBCs.AddNeumannBC(nbca2, nbc2ReCoef, nbc2ImCoef);
      }
   }

   Array<Coefficient*> etaInvReCoef(abcs.Size());
   Array<Coefficient*> etaInvImCoef(abcs.Size());
   if (abcs.Size() > 0 && abcs.Size() == abct.Size())
   {
     for (int i=0; i<abcs.Size(); i++)
     {
       Array<int> bdr(1);
       bdr = abcs[i];

       etaInvReCoef[i] = new StixAdmittanceCoef(abct[i], stixParams, StixCoef::REAL_PART);
       etaInvImCoef[i] = new StixAdmittanceCoef(abct[i], stixParams, StixCoef::IMAG_PART);

       stixBCs.AddSommerfeldBC(bdr, *etaInvReCoef[i], *etaInvImCoef[i]);
     }
   }
   

   VectorFunctionCoefficient jrCoef(3, j_src_r);
   VectorFunctionCoefficient jiCoef(3, j_src_i);
   if (slab_params_.Size() > 0)
   {
      if (Mpi::Root())
      {
         cout << "Adding volumetric current source." << endl;
      }
      if (jsrca.Size() == 0)
      {
         jsrca = pmesh.attributes;
      }
      stixBCs.AddCurrentSrc(jsrca, jrCoef, jiCoef);
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
                   (phase_shift) ? &kReCoef : NULL,
                   (phase_shift) ? &kImCoef : NULL,
                   stixBCs,
                   vis_u);

   // Initialize GLVis visualization
   if (visualization)
   {
      CPD.InitializeGLVis();
   }

   // Initialize VisIt visualization
   if ( visit )
   {
      CPD.RegisterVisItFields(visit_dc);
      CPD.WriteVisItFields(0);
   }
   if (Mpi::Root()) { cout << "Initialization done." << endl; }

   ColdPlasmaPlaneWaveE EReCoef(wave_type, stixParams, ,,,,,,, StixCoef::REAL_PART);
   ColdPlasmaPlaneWaveE EImCoef(wave_type, stixParams, ,,,,,,, StixCoef::IMAG_PART);
   
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
            // Compute error
            /*
             real_t glb_error_H = CPD.GetHFieldError(HReCoef, HImCoef);
                  if (Mpi::Root())
                  {
                     cout << "Global L2 Error in H field " << glb_error_H << endl;
                  }
            */
                 real_t glb_error_E = CPD.GetEFieldError(EReCoef, EImCoef);
                 if (Mpi::Root())
                 {
                    cout << "Global L2 Error in E field " << glb_error_E << endl;
                 }
      }

      // Write fields to disk for VisIt
      if ( visit )
	{
	  CPD.WriteVisItFields(it);
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

      if (pmesh.Nonconforming() && Mpi::WorldSize() > 1 && false)
      {
         if (Mpi::Root()) { cout << "Rebalancing ..." << endl; }
         pmesh.Rebalance();

         // Update again after rebalancing
         CPD.Update();
      }
   }

   // Send the solution by socket to a GLVis server.
   if (visualization)
   {
      CPD.DisplayAnimationToGLVis();
   }
   std::cout << "deleting nbcvecs" << std::endl;
   for (int i=0; i<nbcVecs.Size(); i++)
   {
      delete nbcVecs[i];
   }

   // delete mesh3d;
   std::cout << "exiting main" << std::endl;
   return 0;
}

void AdaptInitialMesh(ParMesh &pmesh,
                      int order, int p, StixParams &stixParams,
		      const char * coef, real_t tol, int max_its, int max_dofs,
                      bool visualization)
{
      if (strcmp(coef,"S") && strcmp(coef,"D") &&
          strcmp(coef,"L") && strcmp(coef,"R") &&
          strcmp(coef,"ISP"))
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

   vector<socketstream> sout(2);
   char vishost[] = "localhost";
   int  visport   = 19916;

   int Wx = 0, Wy = 0; // window position
   int Ww = 275, Wh = 250; // window size
   int offx = Ww + 3;

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

      // 19. Send the solution by socket to a GLVis server.
      if (visualization)
      {
         VisualizeField(sout[0], vishost, visport, gf.real(),
                        "Real Stix Coef",
                        Wx, Wy, Ww, Wh);
         Wx += offx;

         VisualizeField(sout[1], vishost, visport, gf.imag(),
                        "Imaginary Stix Coef",
                        Wx, Wy, Ww, Wh);
      }

      if (global_dofs > max_dofs)
      {
         if (Mpi::Root())
         {
            cout << "Reached the maximum number of dofs. Stop." << endl;
         }
         break;
      }

      // 20. Call the refiner to modify the mesh. The refiner calls the error
      //     estimator to obtain element errors, then it selects elements to be
      //     refined and finally it modifies the mesh. The Stop() method can be
      //     used to determine if a stopping criterion was met.
      refiner.Apply(pmesh);
      if (refiner.Stop())
      {
         if (Mpi::Root())
         {
            cout << "Stopping criterion satisfied. Stop." << endl;
         }
         break;
      }

      // 21. Update the finite element space (recalculate the number of DOFs,
      //     etc.)
      L2FESpace.Update();

      // 22. Load balance the mesh, and update the space and solution. Currently
      //     available only for nonconforming meshes.
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
/*
void Update(// ParFiniteElementSpace & H1FESpace,
       // ParFiniteElementSpace & HCurlFESpace,
       // ParFiniteElementSpace & HDivFESpace,
            // ParFiniteElementSpace & L2FESpace,
            // ParFiniteElementSpace & L2V2FESpace,
            VectorCoefficient & BCoef,
            Coefficient & rhoCoef,
            Coefficient & TCoef,
            Coefficient & nueCoef,
            Coefficient & nuiCoef,
            int & size_h1,
            int & size_l2,
            Array<int> & density_offsets,
            Array<int> & temperature_offsets,
            BlockVector & density,
            BlockVector & temperature,
            ParGridFunction & BField,
            VectorFieldVisObject & BField_v,
            ParGridFunction & density_gf,
            ParGridFunction & temperature_gf,
            ParGridFunction & nue_gf,
            ParGridFunction & nui_gf)
{
   H1FESpace.Update();
   // HCurlFESpace.Update();
   // HDivFESpace.Update();
   L2FESpace.Update();
   L2V2FESpace.Update();

   BField.Update();
   BField.ProjectCoefficient(BCoef);

   BField_v.Update();
   BField_v.PrepareVisField(BCoef);

   nue_gf.Update();
   nue_gf.ProjectCoefficient(nueCoef);

   nui_gf.Update();
   nui_gf.ProjectCoefficient(nuiCoef);

   size_l2 = L2FESpace.GetVSize();
   for (int i=1; i<density_offsets.Size(); i++)
   {
      density_offsets[i] = density_offsets[i - 1] + size_l2;
   }
   density.Update(density_offsets);
   for (int i=0; i<density_offsets.Size()-1; i++)
   {
      density_gf.MakeRef(&L2FESpace, density.GetBlock(i).GetData());
      density_gf.ProjectCoefficient(rhoCoef);
   }

   size_h1 = H1FESpace.GetVSize();
   for (int i=1; i<temperature_offsets.Size(); i++)
   {
      temperature_offsets[i] = temperature_offsets[i - 1] + size_h1;
   }
   temperature.Update(temperature_offsets);
   for (int i=0; i<temperature_offsets.Size()-1; i++)
   {
      temperature_gf.MakeRef(&H1FESpace, temperature.GetBlock(i).GetData());
      temperature_gf.ProjectCoefficient(TCoef);
   }
}
*/
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
   ofstream ofs("stix_r2d_eb_cmd.txt");

   for (int i=0; i<argc; i++)
   {
      ofs << argv[i] << " ";
      if (strcmp(argv[i], "-bm"     ) == 0 ||
          strcmp(argv[i], "-cm"     ) == 0 ||
          strcmp(argv[i], "-sm"     ) == 0 ||
          strcmp(argv[i], "-bpp"    ) == 0 ||
          strcmp(argv[i], "-dpp"    ) == 0 ||
          strcmp(argv[i], "-tpp"    ) == 0 ||
          strcmp(argv[i], "-nepp"   ) == 0 ||
          strcmp(argv[i], "-nipp"   ) == 0 ||
          strcmp(argv[i], "-B"      ) == 0 ||
          strcmp(argv[i], "-k-vec"  ) == 0 ||
          strcmp(argv[i], "-q"      ) == 0 ||
          strcmp(argv[i], "-min"    ) == 0 ||
          strcmp(argv[i], "-jsrc"   ) == 0 ||
          strcmp(argv[i], "-slab"   ) == 0 ||
          strcmp(argv[i], "-curve"  ) == 0 ||
          strcmp(argv[i], "-sp"     ) == 0 ||
          strcmp(argv[i], "-abcs"   ) == 0 ||
          strcmp(argv[i], "-abct"   ) == 0 ||
          strcmp(argv[i], "-sbcs"   ) == 0 ||
          strcmp(argv[i], "-pecs"   ) == 0 ||
          strcmp(argv[i], "-dbcs-msa") == 0 ||
          strcmp(argv[i], "-dbcs-pw") == 0 ||
          strcmp(argv[i], "-dbcs1"  ) == 0 ||
          strcmp(argv[i], "-dbcs2"  ) == 0 ||
          strcmp(argv[i], "-dbcv1"  ) == 0 ||
          strcmp(argv[i], "-dbcv2"  ) == 0 ||
          strcmp(argv[i], "-nbcs"   ) == 0 ||
          strcmp(argv[i], "-nbcs1"  ) == 0 ||
          strcmp(argv[i], "-nbcs2"  ) == 0 ||
          strcmp(argv[i], "-nbcv"   ) == 0 ||
          strcmp(argv[i], "-nbcv1"  ) == 0 ||
          strcmp(argv[i], "-nbcv2"  ) == 0)
      {
         ofs << "'" << argv[++i] << "' ";
      }
   }
   ofs << endl << flush;
   ofs.close();
}
/*
// The Impedance is an optional coefficient defined on boundary surfaces which
// can be used in conjunction with absorbing boundary conditions.
Coefficient *
SetupImpedanceCoefficient(const Mesh & mesh, const Array<int> & abcs)
{
   Coefficient * coef = NULL;

   if ( pw_eta_.Size() > 0 )
   {
      MFEM_VERIFY(pw_eta_.Size() == abcs.Size(),
                  "Each impedance value must be associated with exactly one "
                  "absorbing boundary surface.");

      pw_bdr_eta_.SetSize(mesh.bdr_attributes.Size());

      if ( abcs[0] == -1 )
      {
         pw_bdr_eta_ = pw_eta_[0];
      }
      else
      {
         pw_bdr_eta_ = 0.0;

         for (int i=0; i<pw_eta_.Size(); i++)
         {
            pw_bdr_eta_[abcs[i]-1] = pw_eta_[i];
         }
      }
      coef = new PWConstCoefficient(pw_bdr_eta_);
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

      pw_bdr_eta_inv_.SetSize(mesh.bdr_attributes.Size());

      if ( abcs[0] == -1 )
      {
         pw_bdr_eta_inv_ = 1.0 / pw_eta_[0];
      }
      else
      {
         pw_bdr_eta_inv_ = 0.0;

         for (int i=0; i<pw_eta_.Size(); i++)
         {
            pw_bdr_eta_inv_[abcs[i]-1] = 1.0 / pw_eta_[i];
         }
      }
      coef = new PWConstCoefficient(pw_bdr_eta_inv_);
   }

   return coef;
}
*/
void slab_current_source_r(const Vector &x, Vector &j)
{
   int sdim = x.Size();

   j.SetSize(3);
   j = 0.0;

   bool cmplx = slab_params_.Size() == 6 + 2 * sdim;

   int o = 3 + (cmplx ? 3 : 0);

   real_t x0 = slab_params_(o+0);
   real_t dx = slab_params_(o+1);

   if (x[0] >= x0-0.5*dx && x[0] <= x0+0.5*dx)
   {
      j(0) = slab_params_(0);
      j(1) = slab_params_(1);
      j(2) = slab_params_(2);
      if (slab_profile_ == 1)
      {
	j *= 0.5 * (1.0 + cos(2.0 * M_PI * (x[0] - x0)/dx));
      }
   }
}

void slab_current_source_i(const Vector &x, Vector &j)
{
   int sdim = x.Size();

   j.SetSize(3);
   j = 0.0;

   bool cmplx = slab_params_.Size() == 6 + 2 * sdim;

   int o = 3 + (cmplx ? 3 : 0);

   real_t x0 = slab_params_(o+0);
   real_t dx = slab_params_(o+1);

   if (x[0] >= x0-0.5*dx && x[0] <= x0+0.5*dx)
   {
      if (cmplx)
      {
         j(0) = slab_params_(3);
         j(1) = slab_params_(4);
         j(2) = slab_params_(5);
         if (slab_profile_ == 1)
         {
	   j *= 0.5 * (1.0 + cos(2.0 * M_PI * (x[0] - x0)/dx));
	 }
      }
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

ColdPlasmaPlaneWaveH::ColdPlasmaPlaneWaveH(char type,
                                           real_t omega,
                                           const Vector & B,
                                           const Vector & number,
                                           const Vector & charge,
                                           const Vector & mass,
                                           const Vector & temp,
                                           int nuprof,
                                           real_t res_lim,
                                           bool realPart)
   : VectorCoefficient(3),
     type_(type),
     realPart_(realPart),
     nuprof_(nuprof),
     res_lim_(res_lim),
     omega_(omega),
     Bmag_(B.Norml2()),
     Jy_(0.0),
     xJ_(0.5),
     dx_(0.05),
     Lx_(1.0),
     kappa_(0.0),
     b_(B),
     bc_(3),
     bcc_(3),
     h_r_(3),
     h_i_(3),
     k_r_(3),
     k_i_(3),
     beta_r_(3),
     beta_i_(3),
     numbers_(number),
     charges_(charge),
     masses_(mass),
     temps_(temp)
{
   b_ *= 1.0 / Bmag_;

   {
      real_t bx = b_(0);
      real_t by = b_(1);
      real_t bz = b_(2);

      bc_(0) = by - bz;
      bc_(1) = bz - bx;
      bc_(2) = bx - by;

      bcc_(0) = by*by + bz*bz - bx*(by + bz);
      bcc_(1) = bz*bz + bx*bx - by*(bz + bx);
      bcc_(2) = bx*bx + by*by - bz*(bx + by);

      bc_  *= 1.0 / bc_.Norml2();
      bcc_ *= 1.0 / bcc_.Norml2();
   }

   beta_r_ = 0.0;
   beta_i_ = 0.0;

   S_ = S_cold_plasma(omega_, Bmag_, 0.0, 0.0, numbers_, charges_, masses_,
                      temps_, nuprof_, res_lim_);
   D_ = D_cold_plasma(omega_, Bmag_, 0.0, 0.0, numbers_, charges_, masses_,
                      temps_, nuprof_, res_lim_);
   P_ = P_cold_plasma(omega_, 0.0, numbers_, charges_, masses_, temps_,
                      nuprof_);

   switch (type_)
   {
      case 'L':
      {
         kappa_ = omega_ * sqrt(S_ - D_) / c0_;
         if (kappa_.imag() < 0.0) { kappa_ *= -1.0; }

         k_r_.Set(kappa_.real(), b_);
         k_i_.Set(kappa_.imag(), b_);

         complex<real_t> h = sqrt((S_ - D_) * (epsilon0_ / mu0_));

         h_r_.Set(-M_SQRT1_2 * h.real(), bcc_);
         h_r_.Add(-M_SQRT1_2 * h.imag(), bc_);
         h_i_.Set( M_SQRT1_2 * h.real(), bc_);
         h_i_.Add(-M_SQRT1_2 * h.imag(), bcc_);
      }
      break;
      case 'R':
      {
         kappa_ = omega_ * sqrt(S_ + D_) / c0_;
         if (kappa_.imag() < 0.0) { kappa_ *= -1.0; }

         k_r_.Set(kappa_.real(), b_);
         k_i_.Set(kappa_.imag(), b_);

         complex<real_t> h = sqrt((S_ + D_) * (epsilon0_ / mu0_));

         h_r_.Set(-M_SQRT1_2 * h.real(), bcc_);
         h_r_.Add( M_SQRT1_2 * h.imag(), bc_);
         h_i_.Set(-M_SQRT1_2 * h.real(), bc_);
         h_i_.Add(-M_SQRT1_2 * h.imag(), bcc_);
      }
      break;
      case 'O':
      {
         kappa_ = omega_ * sqrt(P_) / c0_;
         if (kappa_.imag() < 0.0) { kappa_ *= -1.0; }

         k_r_.Set(kappa_.real(), bc_);
         k_i_.Set(kappa_.imag(), bc_);

         complex<real_t> h = sqrt(P_ * (epsilon0_ / mu0_));

         h_r_.Set(h.real(), bcc_);
         h_i_.Set(h.imag(), bcc_);
      }
      break;
      case 'X':
      {
         kappa_ = omega_ * sqrt(S_ - D_ * D_ / S_) / c0_;
         if (kappa_.imag() < 0.0) { kappa_ *= -1.0; }

         k_r_.Set(kappa_.real(), bc_);
         k_i_.Set(kappa_.imag(), bc_);

         complex<real_t> h = S_ * sqrt(S_ - D_ * D_ / S_);
         h *= sqrt((epsilon0_ / mu0_) / (S_ * S_ + D_ * D_));

         h_r_.Set(-h.real(), b_);
         h_i_.Set(-h.imag(), b_);
      }
      break;
      case 'J':
         // MFEM_VERIFY(fabs(B_[2]) == Bmag_,
         //             "Current slab require a magnetic field in the z-direction.");
         break;
   }
}

void ColdPlasmaPlaneWaveH::Eval(Vector &V, ElementTransformation &T,
                                const IntegrationPoint &ip)
{
   V.SetSize(3);

   real_t x_data[3];
   Vector x(x_data, 3);
   T.Transform(ip, x);

   complex<real_t> i = complex<real_t>(0.0,1.0);

   switch (type_)
   {
      case 'L': // Left Circularly Polarized, propagating along B
      case 'R': // Right Circularly Polarized, propagating along B
      case 'O': // Ordinary wave propagating perpendicular to B
      case 'X': // eXtraordinary wave propagating perpendicular to B
      {
         complex<real_t> kx = 0.0;
         for (int d=0; d<3; d++)
         {
            kx += (k_r_[d] - beta_r_[d] + i * (k_i_[d] - beta_i_[d])) * x[d];
         }
         complex<real_t> phase = exp(i * kx);
         real_t phase_r = phase.real();
         real_t phase_i = phase.imag();

         if (realPart_)
         {
            for (int d=0; d<3; d++)
            {
               V[d] = h_r_[d] * phase_r - h_i_[d] * phase_i;
            }
         }
         else
         {
            for (int d=0; d<3; d++)
            {
               V[d] = h_r_[d] * phase_i + h_i_[d] * phase_r;
            }
         }
      }
      break;
      case 'J':  // Slab of current density perpendicular to propagation
      {
         if (true/* k_.Size() == 0 */)
         {
            complex<real_t> kE = omega_ * sqrt(S_ - D_ * D_ / S_) / c0_;

            complex<real_t> skL = sin(kE * Lx_);
            complex<real_t> E0 = i * Jy_ /
                                 (omega_ * epsilon0_ * skL *
                                  (S_ * S_ - D_ * D_));

            complex<real_t> Ex = i * D_ * E0;
            complex<real_t> Ey = S_ * E0;

            if (x[0] <= xJ_ - 0.5 * dx_)
            {
               complex<real_t> skLJ = sin(kE * (Lx_ - xJ_));
               complex<real_t> skd  = sin(kE * 0.5 * dx_);
               complex<real_t> skx  = sin(kE * x[0]);

               Ex *= -2.0 * skLJ * skd * skx;
               Ey *= -2.0 * skLJ * skd * skx;
            }
            else if (x[0] <= xJ_ + 0.5 * dx_)
            {
               complex<real_t> ck1  = cos(kE * (Lx_ - xJ_ - 0.5 * dx_));
               complex<real_t> ck2  = cos(kE * (xJ_ - 0.5 * dx_));
               complex<real_t> skx  = sin(kE * x[0]);
               complex<real_t> skLx = sin(kE * (Lx_ - x[0]));

               Ex *= skL - ck1 * skx - ck2 * skLx;
               Ey *= skL - ck1 * skx - ck2 * skLx;
            }
            else
            {
               complex<real_t> skJ  = sin(kE * xJ_);
               complex<real_t> skd  = sin(kE * 0.5 * dx_);
               complex<real_t> skLx = sin(kE * (Lx_ - x[0]));

               Ex *= -2.0 * skJ * skd * skLx;
               Ey *= -2.0 * skJ * skd * skLx;
            }

            if (realPart_)
            {
               V[0] = Ex.real();
               V[1] = Ey.real();
               V[2] = 0.0;
            }
            else
            {
               V[0] = Ex.imag();
               V[1] = Ey.imag();
               V[2] = 0.0;
            }
         }
         else
         {
            // General phase shift
            V = 0.0; // For now...
         }
      }
      break;
      case 'Z':
         V = 0.0;
         break;
   }
}

ColdPlasmaPlaneWaveE::ColdPlasmaPlaneWaveE(char type,
					   StixParams & stixParams,
					   const Vector & B,
                                           const Vector & number,
                                           const Vector & charge,
                                           const Vector & mass,
                                           const Vector & temp,
                                           int nuprof,
                                           real_t res_lim,
                                           ReImPart re_im_part)
:  StixCoefBase(stixParams, re_im_part),
  VectorCoefficient(3),
     type_(type),
     nuprof_(nuprof),
     Bmag_(B.Norml2()),
     Jy_(0.0),
     xJ_(0.5),
     dx_(0.05),
     Lx_(1.0),
     kappa_(0.0),
     b_(B),
     bc_(3),
     bcc_(3),
     e_r_(3),
     e_i_(3),
     k_r_(3),
     k_i_(3),
     beta_r_(3),
     beta_i_(3),
     numbers_(number),
     charges_(charge),
     masses_(mass),
     temps_(temp)
{
   b_ *= 1.0 / Bmag_;

   {
      real_t bx = b_(0);
      real_t by = b_(1);
      real_t bz = b_(2);

      bc_(0) = by - bz;
      bc_(1) = bz - bx;
      bc_(2) = bx - by;

      bcc_(0) = by*by + bz*bz - bx*(by + bz);
      bcc_(1) = bz*bz + bx*bx - by*(bz + bx);
      bcc_(2) = bx*bx + by*by - bz*(bx + by);

      bc_  *= 1.0 / bc_.Norml2();
      bcc_ *= 1.0 / bcc_.Norml2();
   }

   beta_r_ = 0.0;
   beta_i_ = 0.0;

   S_ = S_cold_plasma(omega_, Bmag_, 0.0, 0.0, numbers_, charges_, masses_,
                      temps_, nuprof_, res_lim_);
   D_ = D_cold_plasma(omega_, Bmag_, 0.0, 0.0, numbers_, charges_, masses_,
                      temps_, nuprof_, res_lim_);
   P_ = P_cold_plasma(omega_, 0.0, numbers_, charges_, masses_, temps_,
                      nuprof_);

   switch (type_)
   {
      case 'L':
      {
         kappa_ = omega_ * sqrt(S_ - D_) / c0_;
         if (kappa_.imag() < 0.0) { kappa_ *= -1.0; }

         k_r_.Set(kappa_.real(), b_);
         k_i_.Set(kappa_.imag(), b_);

         e_r_.Set(M_SQRT1_2, bc_);
         e_i_.Set(M_SQRT1_2, bcc_);
      }
      break;
      case 'R':
      {
         kappa_ = omega_ * sqrt(S_ + D_) / c0_;
         if (kappa_.imag() < 0.0) { kappa_ *= -1.0; }

         k_r_.Set(kappa_.real(), b_);
         k_i_.Set(kappa_.imag(), b_);

         e_r_.Set( M_SQRT1_2, bc_);
         e_i_.Set(-M_SQRT1_2, bcc_);
      }
      break;
      case 'O':
      {
         kappa_ = omega_ * sqrt(P_) / c0_;
         if (kappa_.imag() < 0.0) { kappa_ *= -1.0; }

         k_r_.Set(kappa_.real(), bc_);
         k_i_.Set(kappa_.imag(), bc_);

         e_r_.Set(1.0, b_);
         e_i_ = 0.0;
      }
      break;
      case 'X':
      {
         kappa_ = omega_ * sqrt(S_ - D_ * D_ / S_) / c0_;
         if (kappa_.imag() < 0.0) { kappa_ *= -1.0; }

         k_r_.Set(kappa_.real(), bc_);
         k_i_.Set(kappa_.imag(), bc_);

         complex<real_t> den = sqrt(S_ * S_ + D_ * D_);
         complex<real_t> ec  = D_ / den;
         complex<real_t> ecc = S_ / den;

         e_r_.Set(ecc.real(), bcc_);
         e_r_.Add(ec.imag(), bc_);
         e_i_.Set(-ec.real(), bc_);
         e_i_.Add(ecc.imag(), bcc_);
      }
      break;
      case 'J':
         // MFEM_VERIFY(fabs(B_[2]) == Bmag_,
         //           "Current slab require a magnetic field in the z-direction.");
         break;
   }
}

void ColdPlasmaPlaneWaveE::Eval(Vector &V, ElementTransformation &T,
                                const IntegrationPoint &ip)
{
   V.SetSize(3);

   real_t x_data[3];
   Vector x(x_data, 3);
   T.Transform(ip, x);

   complex<real_t> i = complex<real_t>(0.0,1.0);

   switch (type_)
   {
      case 'L': // Left Circularly Polarized, propagating along B
      case 'R': // Right Circularly Polarized, propagating along B
      case 'O': // Ordinary wave propagating perpendicular to B
      case 'X': // eXtraordinary wave propagating perpendicular to B
      {
         complex<real_t> kx = 0.0;
         for (int d=0; d<3; d++)
         {
            kx += (k_r_[d] - beta_r_[d] + i * (k_i_[d] - beta_i_[d])) * x[d];
         }
         complex<real_t> phase = exp(i * kx);
         real_t phase_r = phase.real();
         real_t phase_i = phase.imag();

         if (re_im_part_ == REAL_PART)
         {
            for (int d=0; d<3; d++)
            {
               V[d] = e_r_[d] * phase_r - e_i_[d] * phase_i;
            }
         }
         else
         {
            for (int d=0; d<3; d++)
            {
               V[d] = e_r_[d] * phase_i + e_i_[d] * phase_r;
            }
         }
      }
      break;
      case 'J':  // Slab of current density perpendicular to propagation
      {
         /*
          if (k_.Size() == 0)
               {
                  complex<real_t> kE = omega_ * sqrt(S_ - D_ * D_ / S_) / c0_;

                  complex<real_t> skL = sin(kE * Lx_);
                  complex<real_t> E0 = i * Jy_ /
                                       (omega_ * epsilon0_ * skL *
                                        (S_ * S_ - D_ * D_));

                  complex<real_t> Ex = i * D_ * E0;
                  complex<real_t> Ey = S_ * E0;

                  if (x[0] <= xJ_ - 0.5 * dx_)
                  {
                     complex<real_t> skLJ = sin(kE * (Lx_ - xJ_));
                     complex<real_t> skd  = sin(kE * 0.5 * dx_);
                     complex<real_t> skx  = sin(kE * x[0]);

                     Ex *= -2.0 * skLJ * skd * skx;
                     Ey *= -2.0 * skLJ * skd * skx;
                  }
                  else if (x[0] <= xJ_ + 0.5 * dx_)
                  {
                     complex<real_t> ck1  = cos(kE * (Lx_ - xJ_ - 0.5 * dx_));
                     complex<real_t> ck2  = cos(kE * (xJ_ - 0.5 * dx_));
                     complex<real_t> skx  = sin(kE * x[0]);
                     complex<real_t> skLx = sin(kE * (Lx_ - x[0]));

                     Ex *= skL - ck1 * skx - ck2 * skLx;
                     Ey *= skL - ck1 * skx - ck2 * skLx;
                  }
                  else
                  {
                     complex<real_t> skJ  = sin(kE * xJ_);
                     complex<real_t> skd  = sin(kE * 0.5 * dx_);
                     complex<real_t> skLx = sin(kE * (Lx_ - x[0]));

                     Ex *= -2.0 * skJ * skd * skLx;
                     Ey *= -2.0 * skJ * skd * skLx;
                  }

                  if (realPart_)
                  {
                     V[0] = Ex.real();
                     V[1] = Ey.real();
                     V[2] = 0.0;
                  }
                  else
                  {
                     V[0] = Ex.imag();
                     V[1] = Ey.imag();
                     V[2] = 0.0;
                  }
               }
               else
               {
                  // General phase shift
                  V = 0.0; // For now...
               }
         */
      }
      break;
      case 'Z':
         V = 0.0;
         break;
   }
}
