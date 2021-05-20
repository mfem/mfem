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
//       Stix1D Miniapp: Cold Plasma Electromagnetic Simulation Code
//   -----------------------------------------------------------------------
//
//   Assumes that all sources and boundary conditions oscillate with the same
//   frequency although not necessarily in phase with one another. This
//   assumption implies that we can factor out the time dependence which we
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
//   functions. The curl curl operator must be handled with
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
// (By default the sources and fields are all zero)
//
// Compile with: make stix1d
//
// Sample runs:
//   ./stix1d -md 0.55 -ne  50 -dbcs '3 5' -s 1 -f 80e6 -B '5.4 0 0' -w R
//   ./stix1d -md 0.10 -ne  50 -dbcs '3 5' -s 1 -f 80e6 -B '5.4 0 0' -w L
//   ./stix1d -md 0.01 -ne  50 -dbcs '3 5' -s 1 -f 80e6 -B '0 5.4 0' -w O
//   ./stix1d -md 20 -ne 80 -dbcs '3 5' -s 1 -f 80e4 -B '0 5.4 0' -w X
//   ./stix1d -md 0.24 -ne 40 -dbcs '3 5' -s 1 -f 80e6 -B '0 0 5.4' -w J -slab '0 1 0 0.132 0.024'
//
// Sample runs with partial assembly:
//   ./stix1d -md 0.24  -ne  50 -dbcs '3 5' -s 1 -f 80e6 -B '5.4 0 0' -w R -pa
//   ./stix1d -md 0.24  -ne  50 -dbcs '3 5' -s 1 -f 80e6 -B '5.4 0 0' -w L -pa
//   ./stix1d -md 0.007 -ne  50 -dbcs '3 5' -s 1 -f 80e6 -B '0 5.4 0' -w O -pa
//
// Device sample runs:
//   ./stix1d -md 0.24  -ne  50 -dbcs '3 5' -s 1 -f 80e6 -B '5.4 0 0' -w R -pa -d cuda
//   ./stix1d -md 0.24  -ne  50 -dbcs '3 5' -s 1 -f 80e6 -B '5.4 0 0' -w L -pa -d cuda
//   ./stix1d -md 0.007 -ne  50 -dbcs '3 5' -s 1 -f 80e6 -B '0 5.4 0' -w O -pa -d cuda
//
// Parallel sample runs:
//   mpirun -np 4 ./stix1d -md 0.24  -ne  50 -dbcs '3 5' -s 1 -f 80e6 -B '5.4 0 0' -w R
//   mpirun -np 4 ./stix1d -md 0.24  -ne  50 -dbcs '3 5' -s 1 -f 80e6 -B '5.4 0 0' -w L
//   mpirun -np 4 ./stix1d -md 0.007 -ne  50 -dbcs '3 5' -s 1 -f 80e6 -B '0 5.4 0' -w O
//

#include "cold_plasma_dielectric_coefs.hpp"
#include "cold_plasma_dielectric_solver.hpp"
#include "../common/mesh_extras.hpp"
#include <fstream>
#include <iostream>
#include <cmath>
#include <complex>

using namespace std;
using namespace mfem;
using namespace mfem::common;
using namespace mfem::plasma;

// Admittance for Absorbing Boundary Condition
Coefficient * SetupRealAdmittanceCoefficient(const Mesh & mesh,
                                             const Array<int> & abcs);

// Admittance for Complex-Valued Sheath Boundary Condition
void SetupComplexAdmittanceCoefs(const Mesh & mesh, const Array<int> & sbcs,
                                 Coefficient *& etaInvReCoef,
                                 Coefficient *& etaInvImCoef);

// Storage for user-supplied, real-valued impedance
static Vector pw_eta_(0);      // Piecewise impedance values
static Vector pw_eta_inv_(0);  // Piecewise inverse impedance values

// Storage for user-supplied, complex-valued impedance
static Vector pw_eta_re_(0);      // Piecewise real impedance
static Vector pw_eta_inv_re_(0);  // Piecewise inverse real impedance
static Vector pw_eta_im_(0);      // Piecewise imaginary impedance
static Vector pw_eta_inv_im_(0);  // Piecewise inverse imaginary impedance

// Current Density Function
static Vector slab_params_(0); // Amplitude of x, y, z current source

void slab_current_source(const Vector &x, Vector &j);
void j_src(const Vector &x, Vector &j)
{
   if (slab_params_.Size() > 0)
   {
      slab_current_source(x, j);
   }
}

static Vector B_params_(0);
void B_func(const Vector &x, Vector &E);

// Electric Field Boundary Condition: The following function returns zero but
// any function could be used.
void e_bc_r(const Vector &x, Vector &E);
void e_bc_i(const Vector &x, Vector &E);

class ColdPlasmaPlaneWave: public VectorCoefficient
{
public:
   ColdPlasmaPlaneWave(char type,
                       double omega,
                       const Vector & B,
                       const Vector & number,
                       const Vector & charge,
                       const Vector & mass,
                       const Vector & temp,
                       bool realPart = true);

   void SetCurrentSlab(double Jy, double xJ, double delta, double Lx)
   { Jy_ = Jy; xJ_ = xJ; dx_ = delta, Lx_ = Lx; }

   void SetPhaseShift(const Vector &k) { k_ = k; }

   void Eval(Vector &V, ElementTransformation &T,
             const IntegrationPoint &ip);

private:
   char type_;
   bool realPart_;
   double omega_;
   double Bmag_;
   double Jy_;
   double xJ_;
   double dx_;
   double Lx_;
   Vector k_;

   const Vector & B_;
   const Vector & numbers_;
   const Vector & charges_;
   const Vector & masses_;
   const Vector & temps_;

   complex<double> S_;
   complex<double> D_;
   complex<double> P_;
};

void Update(ParFiniteElementSpace & H1FESpace,
            ParFiniteElementSpace & HCurlFESpace,
            ParFiniteElementSpace & HDivFESpace,
            ParFiniteElementSpace & L2FESpace,
            ParGridFunction & BField,
            VectorCoefficient & BCoef,
            Coefficient & rhoCoef,
            Coefficient & TCoef,
            int & size_h1,
            int & size_l2,
            Array<int> & density_offsets,
            Array<int> & temperature_offsets,
            BlockVector & density,
            BlockVector & temperature,
            ParGridFunction & density_gf,
            ParGridFunction & temperature_gf);

// Mesh Size
static Vector mesh_dim_(0); // x, y, z dimensions of mesh

// Prints the program's logo to the given output stream
void display_banner(ostream & os);

int main(int argc, char *argv[])
{
   MPI_Session mpi(argc, argv);

   if ( mpi.Root() ) { display_banner(cout); }

   int logging = 1;

   // Parse command-line options.
   int order = 1;
   int maxit = 1;
   int sol = 2;
   int prec = 1;
   bool herm_conv = false;
   bool vis_u = false;
   bool visualization = true;
   bool visit = true;

   double freq = 1.0e9;
   const char * wave_type = "R";

   Vector BVec(3);
   BVec = 0.0; BVec(0) = 0.1;

   bool phase_shift = false;
   Vector kVec;
   Vector kReVec;
   Vector kImVec;

   Vector numbers;
   Vector charges;
   Vector masses;
   Vector temps;

   PlasmaProfile::Type dpt = PlasmaProfile::CONSTANT;
   PlasmaProfile::Type tpt = PlasmaProfile::CONSTANT;
   Vector dpp;
   Vector tpp;

   Array<int> abcs; // Absorbing BC attributes
   Array<int> sbca; // Sheath BC attributes
   Array<int> dbca; // Dirichlet BC attributes
   int num_elements = 10;

   SolverOptions solOpts;
   solOpts.maxIter = 1000;
   solOpts.kDim = 50;
   solOpts.printLvl = 1;
   solOpts.relTol = 1e-4;
   solOpts.euLvl = 1;

   bool pa = false;
   const char *device_config = "cpu";

   OptionsParser args(argc, argv);
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&freq, "-f", "--frequency",
                  "Frequency in Hertz (of course...)");
   args.AddOption((int*)&dpt, "-dp", "--density-profile",
                  "Density Profile Type (for ions): \n"
                  "0 - Constant, 1 - Constant Gradient, "
                  "2 - Hyperbolic Tangent.");
   args.AddOption(&dpp, "-dpp", "--density-profile-params",
                  "Density Profile Parameters: \n"
                  "   CONSTANT: density value \n"
                  "   GRADIENT: value, location, gradient (7 params)\n"
                  "   TANH:     value at 0, value at 1, skin depth, "
                  "location of 0 point, unit vector along gradient.");
   args.AddOption((int*)&tpt, "-tp", "--temperature-profile",
                  "Temperature Profile Type: \n"
                  "0 - Constant, 1 - Constant Gradient, "
                  "2 - Hyperbolic Tangent.");
   args.AddOption(&tpp, "-tpp", "--temperature-profile-params",
                  "Temperature Profile Parameters: \n"
                  "   CONSTANT: temperature value \n"
                  "   GRADIENT: value, location, gradient (7 params)\n"
                  "   TANH:     value at 0, value at 1, skin depth, "
                  "location of 0 point, unit vector along gradient.");
   args.AddOption(&wave_type, "-w", "--wave-type",
                  "Wave type: 'R' - Right Circularly Polarized, "
                  "'L' - Left Circularly Polarized, "
                  "'O' - Ordinary, 'X' - Extraordinary, "
                  "'J' - Current Slab (in conjunction with -slab), "
                  "'Z' - Zero");
   args.AddOption(&B_params_, "-b", "--magnetic-flux",
                  "Background magnetic flux parameters");
   args.AddOption(&BVec, "-B", "--magnetic-flux",
                  "Background magnetic flux vector");
   args.AddOption(&kVec, "-k-vec", "--phase-vector",
                  "Phase shift vector across periodic directions."
                  " For complex phase shifts input 3 real phase shifts "
                  "followed by 3 imaginary phase shifts");
   args.AddOption(&numbers, "-num", "--number-densites",
                  "Number densities of the various species");
   args.AddOption(&charges, "-q", "--charges",
                  "Charges of the various species "
                  "(in units of electron charge)");
   args.AddOption(&masses, "-m", "--masses",
                  "Masses of the various species (in amu)");
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
   args.AddOption(&pw_eta_, "-pwz", "--piecewise-eta",
                  "Piecewise values of Impedance (one value per abc surface)");
   args.AddOption(&pw_eta_re_, "-pwz-r", "--piecewise-eta-r",
                  "Piecewise values of Real part of Complex Impedance "
                  "(one value per abc surface)");
   args.AddOption(&pw_eta_im_, "-pwz-i", "--piecewise-eta-i",
                  "Piecewise values of Imaginary part of Complex Impedance "
                  "(one value per abc surface)");
   args.AddOption(&slab_params_, "-slab", "--slab_params",
                  "Amplitude");
   args.AddOption(&abcs, "-abcs", "--absorbing-bc-surf",
                  "Absorbing Boundary Condition Surfaces");
   args.AddOption(&sbca, "-sbcs", "--sheath-bc-surf",
                  "Sheath Boundary Condition Surfaces");
   args.AddOption(&dbca, "-dbcs", "--dirichlet-bc-surf",
                  "Dirichlet Boundary Condition Surfaces");
   args.AddOption(&mesh_dim_, "-md", "--mesh_dimensions",
                  "The x, y, z mesh dimensions");
   args.AddOption(&num_elements, "-ne", "--num-elements",
                  "The number of mesh elements in x");
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
   args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
                  "--no-partial-assembly", "Enable Partial Assembly.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.Parse();
   if (!args.Good())
   {
      if (mpi.Root())
      {
         args.PrintUsage(cout);
      }
      return 1;
   }
   Device device(device_config);
   if (mpi.Root())
   {
      device.Print();
   }

   if (numbers.Size() == 0)
   {
      numbers.SetSize(2);
      if (dpp.Size() == 0)
      {
         numbers[0] = 1.0e19;
         numbers[1] = 1.0e19;
      }
      else
      {
         switch (dpt)
         {
            case PlasmaProfile::CONSTANT:
               numbers[0] = dpp[0];
               numbers[1] = dpp[0];
               break;
            case PlasmaProfile::GRADIENT:
               numbers[0] = dpp[0];
               numbers[1] = dpp[0];
               break;
            case PlasmaProfile::TANH:
               numbers[0] = dpp[1];
               numbers[1] = dpp[1];
               break;
            default:
               numbers[0] = 1.0e19;
               numbers[1] = 1.0e19;
               break;
         }
      }
   }
   if (dpp.Size() == 0)
   {
      dpp.SetSize(1);
      dpp[0] = 1.0e19;
   }
   if (charges.Size() == 0)
   {
      charges.SetSize(2);
      charges[0] = -1.0;
      charges[1] =  1.0;
   }
   if (masses.Size() == 0)
   {
      masses.SetSize(2);
      masses[0] = me_u_;
      masses[1] = 2.01410178;
   }
   if (temps.Size() == 0)
   {
      temps.SetSize(2);
      if (tpp.Size() == 0)
      {
         tpp.SetSize(1);
         tpp[0] = 1.0e3;
         temps[0] = tpp[0];
         temps[1] = tpp[0];
      }
      else
      {
         switch (tpt)
         {
            case PlasmaProfile::CONSTANT:
               temps[0] = tpp[0];
               temps[1] = tpp[0];
               break;
            case PlasmaProfile::GRADIENT:
               temps[0] = tpp[0];
               temps[1] = tpp[0];
               break;
            case PlasmaProfile::TANH:
               temps[0] = tpp[1];
               temps[1] = tpp[1];
               break;
            default:
               temps[0] = 1.0e3;
               temps[1] = 1.0e3;
               break;
         }
      }
   }
   if (num_elements <= 0)
   {
      num_elements = 10;
   }
   if (mesh_dim_.Size() == 0)
   {
      mesh_dim_.SetSize(3);
      mesh_dim_ = 0.0;
   }
   else if (mesh_dim_.Size() < 3)
   {
      double d0 = mesh_dim_[0];
      double d1 = (mesh_dim_.Size() == 2) ? mesh_dim_[1] : 0.1 * d0;
      mesh_dim_.SetSize(3);
      mesh_dim_[0] = d0;
      mesh_dim_[1] = d1;
      mesh_dim_[2] = d1;
   }
   if (mesh_dim_[0] == 0.0)
   {
      mesh_dim_[0] = 1.0;
      mesh_dim_[1] = 0.1;
      mesh_dim_[2] = 0.1;
   }
   double omega = 2.0 * M_PI * freq;
   if (kVec.Size() != 0)
   {
      phase_shift = true;
   }

   if (mpi.Root())
   {
      args.PrintOptions(cout);
   }

   ComplexOperator::Convention conv =
      herm_conv ? ComplexOperator::HERMITIAN : ComplexOperator::BLOCK_SYMMETRIC;

   if (mpi.Root())
   {
      double lam0 = c0_ / freq;
      double Bmag = BVec.Norml2();
      std::complex<double> S = S_cold_plasma(omega, Bmag, numbers,
                                             charges, masses, temps);
      std::complex<double> P = P_cold_plasma(omega, numbers,
                                             charges, masses, temps);
      std::complex<double> D = D_cold_plasma(omega, Bmag, numbers,
                                             charges, masses, temps);
      std::complex<double> R = R_cold_plasma(omega, Bmag, numbers,
                                             charges, masses, temps);
      std::complex<double> L = L_cold_plasma(omega, Bmag, numbers,
                                             charges, masses, temps);

      cout << "\nConvenient Terms:\n";
      cout << "R = " << R << ",\tL = " << L << endl;
      cout << "S = " << S << ",\tD = " << D << ",\tP = " << P << endl;

      cout << "\nSpecies Properties (number, charge, mass):\n";
      for (int i=0; i<numbers.Size(); i++)
      {
         cout << numbers[i] << '\t' << charges[i] << '\t' << masses[i] << '\n';
      }
      cout << "\nPlasma and Cyclotron Frequencies by Species (GHz):\n";
      for (int i=0; i<numbers.Size(); i++)
      {
         cout << omega_p(numbers[i], charges[i], masses[i]) / (2.0e9 * M_PI)
              << '\t'
              << omega_c(Bmag, charges[i], masses[i]) / (2.0e9 * M_PI) << '\n';
      }

      cout << "\nWavelengths (meters):\n";
      cout << "   Free Space Wavelength: " << lam0 << '\n';
      if (real(sqrt(S-D)) != 0.0 )
      {
         cout << "   L mode:    " << lam0 / real(sqrt(S-D)) << '\n';
      }
      if (real(sqrt(S+D)) != 0.0 )
      {
         cout << "   R mode:    " << lam0 / real(sqrt(S+D)) << '\n';
      }
      if (real(sqrt(P)) != 0.0 )
      {
         cout << "   O mode:    " << lam0 / real(sqrt(P)) << '\n';
      }
      if (real(sqrt(S-D*D/S)) != 0.0 )
      {
         cout << "   X mode:    " << lam0 * real(sqrt(S/(S*S-D*D))) << '\n';
      }

      cout << "\nSkin Depth (meters):\n";
      if (imag(sqrt(S-D)) != 0.0 )
      {
         cout << "   L mode:    " << lam0 / imag(sqrt(S-D)) << '\n';
      }
      if (imag(sqrt(S+D)) != 0.0 )
      {
         cout << "   R mode:    " << lam0 / imag(sqrt(S+D)) << '\n';
      }
      if (imag(sqrt(P)) != 0.0 )
      {
         cout << "   O mode:    " << lam0 / imag(sqrt(P)) << '\n';
      }
      if (imag(sqrt(S-D*D/S)) != 0.0 )
      {
         cout << "   X mode:    " << lam0 * imag(sqrt(S/(S*S-D*D))) << '\n';
      }
      cout << endl;
   }

   // Read the (serial) mesh from the given mesh file on all processors.  We
   // can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
   // and volume meshes with the same code.
   if ( mpi.Root() && logging > 0 ) { cout << "Building Mesh ..." << endl; }

   tic_toc.Clear();
   tic_toc.Start();

   Mesh * mesh = new Mesh(num_elements, 3, 3, Element::HEXAHEDRON, 1,
                          mesh_dim_(0), mesh_dim_(1), mesh_dim_(2));
   {
      Array<int> v2v(mesh->GetNV());
      for (int i=0; i<v2v.Size(); i++) { v2v[i] = i; }
      for (int i=0; i<=num_elements; i++)
      {
         v2v[ 3 * num_elements +  3 + i] = i; // y = sy, z = 0
         v2v[12 * num_elements + 12 + i] = i; // y =  0, z = sz
         v2v[15 * num_elements + 15 + i] = i; // y = sy, z = sz
      }
      for (int j=1; j<3; j++)
      {
         for (int i=0; i<=num_elements; i++)
         {
            v2v[(j + 12) * (num_elements +  1) + i] =
               j * (num_elements +  1) + i;
         }
      }
      for (int k=1; k<3; k++)
      {
         for (int i=0; i<=num_elements; i++)
         {
            v2v[(4 * k + 3) * (num_elements +  1) + i] =
               4 * k * (num_elements +  1) + i;
         }
      }

      Mesh * per_mesh = MakePeriodicMesh(mesh, v2v);
      delete mesh;
      mesh = per_mesh;
   }

   // Ensure that quad and hex meshes are treated as non-conforming.
   mesh->EnsureNCMesh();

   // Define a parallel mesh by a partitioning of the serial mesh. Refine
   // this mesh further in parallel to increase the resolution. Once the
   // parallel mesh is defined, the serial mesh can be deleted.
   ParMesh pmesh(MPI_COMM_WORLD, *mesh);
   delete mesh;

   tic_toc.Stop();

   if (mpi.Root() && logging > 0 )
   {
      cout << " done in " << tic_toc.RealTime() << " seconds." << endl;
   }

   if (mpi.Root() && logging > 0)
   {
      cout << "Initializing coefficients..." << endl;
   }
   tic_toc.Clear();
   tic_toc.Start();

   VectorCoefficient * BCoef = NULL;
   if (B_params_.Size()  == 7)
   {
      BCoef = new VectorFunctionCoefficient(3, B_func);
   }
   else
   {
      BCoef = new VectorConstantCoefficient(BVec);
   }
   VectorConstantCoefficient kCoef(kVec);

   if (mpi.Root() && logging > 0)
   {
      cout << "Building Finite Element Spaces..." << endl;
   }
   H1_ParFESpace H1FESpace(&pmesh, order, pmesh.Dimension());
   ND_ParFESpace HCurlFESpace(&pmesh, order, pmesh.Dimension());
   RT_ParFESpace HDivFESpace(&pmesh, order, pmesh.Dimension());
   L2_ParFESpace L2FESpace(&pmesh, order, pmesh.Dimension());

   ParGridFunction BField(&HDivFESpace);
   ParGridFunction temperature_gf;
   ParGridFunction density_gf;

   BField.ProjectCoefficient(*BCoef);

   if (mpi.Root() && logging > 0)
   {
      cout << "Setting up density and temperature..." << endl;
   }
   int size_h1 = H1FESpace.GetVSize();
   int size_l2 = L2FESpace.GetVSize();

   Array<int> density_offsets(numbers.Size() + 1);
   Array<int> temperature_offsets(numbers.Size() + 2);

   density_offsets[0] = 0;
   temperature_offsets[0] = 0;
   temperature_offsets[1] = size_h1;

   for (int i=1; i<=numbers.Size(); i++)
   {
      density_offsets[i]     = density_offsets[i - 1] + size_l2;
      temperature_offsets[i + 1] = temperature_offsets[i] + size_h1;
   }

   BlockVector density(density_offsets);
   BlockVector temperature(temperature_offsets);

   PlasmaProfile tempCoef(tpt, tpp);
   PlasmaProfile rhoCoef(dpt, dpp);

   for (int i=0; i<=numbers.Size(); i++)
   {
      temperature_gf.MakeRef(&H1FESpace, temperature.GetBlock(i));
      temperature_gf.ProjectCoefficient(tempCoef);
   }

   for (int i=0; i<numbers.Size(); i++)
   {
      density_gf.MakeRef(&L2FESpace, density.GetBlock(i));
      density_gf.ProjectCoefficient(rhoCoef);
   }

   if (mpi.Root() && logging > 0)
   {
      cout << "Initializing more coefficients..." << endl;
   }

   // Create a coefficient describing the magnetic permeability
   ConstantCoefficient muInvCoef(1.0 / mu0_);

   // Create a coefficient describing the surface admittance
   Coefficient * etaInvCoef = SetupRealAdmittanceCoefficient(pmesh, abcs);

   // Create tensor coefficients describing the dielectric permittivity
   DielectricTensor epsilon_real(BField, density, temperature,
                                 L2FESpace, H1FESpace,
                                 omega, charges, masses, true);
   DielectricTensor epsilon_imag(BField, density, temperature,
                                 L2FESpace, H1FESpace,
                                 omega, charges, masses, false);
   SPDDielectricTensor epsilon_abs(BField, density, temperature,
                                   L2FESpace, H1FESpace,
                                   omega, charges, masses);
   SheathImpedance z_r(BField, density, temperature,
                       L2FESpace, H1FESpace,
                       omega, charges, masses, true);
   SheathImpedance z_i(BField, density, temperature,
                       L2FESpace, H1FESpace,
                       omega, charges, masses, false);

   ColdPlasmaPlaneWave EReCoef(wave_type[0], omega, BVec,
                               numbers, charges, masses, temps, true);
   ColdPlasmaPlaneWave EImCoef(wave_type[0], omega, BVec,
                               numbers, charges, masses, temps, false);

   if (wave_type[0] == 'J' && slab_params_.Size() == 5)
   {
      EReCoef.SetCurrentSlab(slab_params_[1], slab_params_[3], slab_params_[4],
                             mesh_dim_[0]);
      EImCoef.SetCurrentSlab(slab_params_[1], slab_params_[3], slab_params_[4],
                             mesh_dim_[0]);
   }
   if (phase_shift)
   {
      EReCoef.SetPhaseShift(kVec);
      EImCoef.SetPhaseShift(kVec);
   }

   mfem::out << "Setting phase shift of ("
             << complex<double>(kReVec[0],kImVec[0]) << ","
             << complex<double>(kReVec[1],kImVec[1]) << ","
             << complex<double>(kReVec[2],kImVec[2]) << ")" << endl;

   VectorConstantCoefficient kReCoef(kReVec);
   VectorConstantCoefficient kImCoef(kImVec);

   tic_toc.Stop();

   if (mpi.Root() && logging > 0 )
   {
      cout << " done in " << tic_toc.RealTime() << " seconds." << endl;
   }

   if (visualization)
   {
      ParComplexGridFunction EField(&HCurlFESpace);
      EField.ProjectCoefficient(EReCoef, EImCoef);

      char vishost[] = "localhost";
      int  visport   = 19916;

      int Wx = 0, Wy = 0; // window position
      int Ww = 350, Wh = 350; // window size
      int offx = Ww+10, offy = Wh+45; // window offsets

      socketstream sock_Er, sock_Ei, sock_B;//, sock_L;
      sock_Er.precision(8);
      sock_Ei.precision(8);
      sock_B.precision(8);

      Wx += 2 * offx;
      VisualizeField(sock_Er, vishost, visport,
                     EField.real(), "Exact Electric Field, Re(E)",
                     Wx, Wy, Ww, Wh);

      Wx += offx;
      VisualizeField(sock_Ei, vishost, visport,
                     EField.imag(), "Exact Electric Field, Im(E)",
                     Wx, Wy, Ww, Wh);

      Wx -= offx;
      Wy += offy;

      VisualizeField(sock_B, vishost, visport,
                     BField, "Background Magnetic Field", Wx, Wy, Ww, Wh);
   }

   // Setup coefficients for Dirichlet BC
   Array<ComplexVectorCoefficientByAttr> dbcs((dbca.Size()==0)?0:1);
   if (dbca.Size() > 0)
   {
      dbcs[0].attr = dbca;
      dbcs[0].real = &EReCoef;
      dbcs[0].imag = &EImCoef;
   }

   Array<ComplexVectorCoefficientByAttr> nbcs(0);

   Array<ComplexCoefficientByAttr> sbcs((sbca.Size() > 0)? 1 : 0);
   if (sbca.Size() > 0)
   {
      sbcs[0].real = &z_r;
      sbcs[0].imag = &z_i;
      sbcs[0].attr = sbca;
      AttrToMarker(pmesh.bdr_attributes.Max(), sbcs[0].attr,
                   sbcs[0].attr_marker);
   }

   // Create the Magnetostatic solver
   if (mpi.Root() && logging > 0)
   {
      cout << "Creating CPDSolver..." << endl;
   }
   CPDSolver CPD(pmesh, order, omega,
                 (CPDSolver::SolverType)sol, solOpts,
                 (CPDSolver::PrecondType)prec,
                 conv, *BCoef, epsilon_real, epsilon_imag, epsilon_abs,
                 muInvCoef, etaInvCoef,
                 (phase_shift) ? &kReCoef : NULL,
                 (phase_shift) ? &kImCoef : NULL,
                 abcs, dbcs, nbcs, sbcs,
                 (slab_params_.Size() > 0) ? j_src : NULL, NULL, vis_u, pa);

   // Initialize GLVis visualization
   if (visualization)
   {
      CPD.InitializeGLVis();
   }

   // Initialize VisIt visualization
   VisItDataCollection visit_dc("STIX1D-AMR-Parallel", &pmesh);

   if ( visit )
   {
      CPD.RegisterVisItFields(visit_dc);
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
      CPD.PrintSizes();

      // Assemble all forms
      CPD.Assemble();

      // Solve the system and compute any auxiliary fields
      CPD.Solve();

      // Compute error
      double glb_error = CPD.GetError(EReCoef, EImCoef);
      if (mpi.Root())
      {
         cout << "Global L2 Error " << glb_error << endl;
      }

      // Determine the current size of the linear system
      int prob_size = CPD.GetProblemSize();

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
      CPD.GetErrorEstimates(errors);

      double local_max_err = errors.Max();
      double global_max_err;
      MPI_Allreduce(&local_max_err, &global_max_err, 1,
                    MPI_DOUBLE, MPI_MAX, pmesh.GetComm());

      // Refine the elements whose error is larger than a fraction of the
      // maximum element error.
      const double frac = 0.5;
      double threshold = frac * global_max_err;
      if (mpi.Root()) { cout << "Refining ..." << endl; }
      {
         pmesh.RefineByError(errors, threshold);
      }

      // Update the magnetostatic solver to reflect the new state of the mesh.
      Update(H1FESpace, HCurlFESpace, HDivFESpace, L2FESpace, BField, *BCoef,
             rhoCoef, tempCoef, size_h1, size_l2, density_offsets,
             temperature_offsets, density, temperature, density_gf,
             temperature_gf);
      CPD.Update();

      if (pmesh.Nonconforming() && mpi.WorldSize() > 1 && false)
      {
         if (mpi.Root()) { cout << "Rebalancing ..." << endl; }
         pmesh.Rebalance();

         // Update again after rebalancing
         Update(H1FESpace, HCurlFESpace, HDivFESpace, L2FESpace, BField, *BCoef,
                rhoCoef, tempCoef, size_h1, size_l2, density_offsets,
                temperature_offsets, density, temperature, density_gf,
                temperature_gf);
         CPD.Update();
      }
   }

   // Send the solution by socket to a GLVis server.
   if (visualization)
   {
      CPD.DisplayAnimationToGLVis();
   }

   return 0;
}

void Update(ParFiniteElementSpace & H1FESpace,
            ParFiniteElementSpace & HCurlFESpace,
            ParFiniteElementSpace & HDivFESpace,
            ParFiniteElementSpace & L2FESpace,
            ParGridFunction & BField,
            VectorCoefficient & BCoef,
            Coefficient & rhoCoef,
            Coefficient & TCoef,
            int & size_h1,
            int & size_l2,
            Array<int> & density_offsets,
            Array<int> & temperature_offsets,
            BlockVector & density,
            BlockVector & temperature,
            ParGridFunction & density_gf,
            ParGridFunction & temperature_gf)
{
   H1FESpace.Update();
   HCurlFESpace.Update();
   HDivFESpace.Update();
   L2FESpace.Update();

   BField.Update();
   BField.ProjectCoefficient(BCoef);

   size_l2 = L2FESpace.GetVSize();
   for (int i=1; i<density_offsets.Size(); i++)
   {
      density_offsets[i] = density_offsets[i - 1] + size_l2;
   }
   density.Update(density_offsets);
   for (int i=0; i<density_offsets.Size()-1; i++)
   {
      density_gf.MakeRef(&L2FESpace, density.GetBlock(i));
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
      temperature_gf.MakeRef(&H1FESpace, temperature.GetBlock(i));
      temperature_gf.ProjectCoefficient(TCoef);
   }
}

const char * banner[6] =
{
   R"(  _________ __   __        ____     ___)",
   R"( /   _____//  |_|__|__  __/_   | __| _/)",
   R"( \_____  \\   __\  \  \/  /|   |/ __ | )",
   R"( /        \|  | |  |>    < |   / /_/ | )",
   R"(/_______  /|__| |__/__/\_ \|___\____ | )",
   R"(        \/               \/         \/ )"
};

// Print the STIX1D ascii logo to the given ostream
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
      << " application, the" << endl
      << "  plasma dielectric for the wave equation is formulated using"
      << " the \"Stix\"" << endl
      << "  notation, \"S, D, P\"." << endl<< endl << flush;
}

// The Admittance is an optional coefficient defined on boundary surfaces which
// can be used in conjunction with absorbing boundary conditions.
Coefficient *
SetupRealAdmittanceCoefficient(const Mesh & mesh, const Array<int> & abcs)
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

// Complex Admittance is an optional pair of coefficients, defined on boundary
// surfaces, which can be used to approximate a sheath boundary condition.
void
SetupComplexAdmittanceCoefs(const Mesh & mesh, const Array<int> & sbcs,
                            Coefficient *& etaInvReCoef,
                            Coefficient *& etaInvImCoef )
{
   MFEM_VERIFY(pw_eta_re_.Size() == sbcs.Size() &&
               pw_eta_im_.Size() == sbcs.Size(),
               "Each impedance value must be associated with exactly one "
               "sheath boundary surface.");

   if (pw_eta_re_.Size() > 0)
   {

      pw_eta_inv_re_.SetSize(mesh.bdr_attributes.Size());
      pw_eta_inv_im_.SetSize(mesh.bdr_attributes.Size());

      if ( sbcs[0] == -1 )
      {
         double zmag2 = pow(pw_eta_re_[0], 2) + pow(pw_eta_im_[0], 2);
         pw_eta_inv_re_ =  pw_eta_re_[0] / zmag2;
         pw_eta_inv_im_ = -pw_eta_im_[0] / zmag2;
      }
      else
      {
         pw_eta_inv_re_ = 0.0;
         pw_eta_inv_im_ = 0.0;

         for (int i=0; i<pw_eta_re_.Size(); i++)
         {
            double zmag2 = pow(pw_eta_re_[i], 2) + pow(pw_eta_im_[i], 2);
            if ( zmag2 > 0.0 )
            {
               pw_eta_inv_re_[sbcs[i]-1] =  pw_eta_re_[i] / zmag2;
               pw_eta_inv_im_[sbcs[i]-1] = -pw_eta_im_[i] / zmag2;
            }
         }
      }
      etaInvReCoef = new PWConstCoefficient(pw_eta_inv_re_);
      etaInvImCoef = new PWConstCoefficient(pw_eta_inv_im_);
   }
}

void slab_current_source(const Vector &x, Vector &j)
{
   MFEM_ASSERT(x.Size() == 3, "current source requires 3D space.");

   j.SetSize(x.Size());
   j = 0.0;

   double width = slab_params_(4);
   double half_x_l = slab_params_(3) - 0.5 * width;
   double half_x_r = slab_params_(3) + 0.5 * width;

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

void B_func(const Vector &x, Vector &B)
{
   B.SetSize(3);

   for (int i=0; i<3; i++)
   {
      B[i] = B_params_[i] +
             (B_params_[i+3] - B_params_[i]) * x[0] / B_params_[6];
   }
}

ColdPlasmaPlaneWave::ColdPlasmaPlaneWave(char type,
                                         double omega,
                                         const Vector & B,
                                         const Vector & number,
                                         const Vector & charge,
                                         const Vector & mass,
                                         const Vector & temp,
                                         bool realPart)
   : VectorCoefficient(3),
     type_(type),
     realPart_(realPart),
     omega_(omega),
     Bmag_(B.Norml2()),
     Jy_(0.0),
     xJ_(0.5),
     Lx_(1.0),
     k_(0),
     B_(B),
     numbers_(number),
     charges_(charge),
     masses_(mass),
     temps_(temp)
{
   switch (type_)
   {
      case 'L':
         MFEM_VERIFY(fabs(B_[0]) == Bmag_,
                     "L waves require a magnetic field in the x-direction.");
         break;
      case 'R':
         MFEM_VERIFY(fabs(B_[0]) == Bmag_,
                     "R waves require a magnetic field in the x-direction.");
         break;
      case 'O':
         MFEM_VERIFY(fabs(B_[1]) == Bmag_,
                     "O waves require a magnetic field in the y-direction.");
         break;
      case 'X':
         MFEM_VERIFY(fabs(B_[1]) == Bmag_,
                     "X waves require a magnetic field in the y-direction.");
         break;
      case 'J':
         MFEM_VERIFY(fabs(B_[2]) == Bmag_,
                     "Current slab require a magnetic field in the z-direction.");
         break;
   }

   S_ = S_cold_plasma(omega_, Bmag_, numbers_, charges_, masses_, temps_);
   D_ = D_cold_plasma(omega_, Bmag_, numbers_, charges_, masses_, temps_);
   P_ = P_cold_plasma(omega_, numbers_, charges_, masses_, temps_);
}

void ColdPlasmaPlaneWave::Eval(Vector &V, ElementTransformation &T,
                               const IntegrationPoint &ip)
{
   V.SetSize(3);

   double x_data[3];
   Vector x(x_data, 3);
   T.Transform(ip, x);

   complex<double> i = complex<double>(0.0,1.0);

   switch (type_)
   {
      case 'L': // Left Circularly Polarized, propagating along B
      {
         complex<double> kL = omega_ * sqrt(S_ - D_) / c0_;
         if (kL.imag() < 0.0) { kL *= -1.0; }

         complex<double> Ez = exp(i * kL * x[0]);
         complex<double> Ey = i * Ez;

         if (realPart_)
         {
            V[0] = 0.0;
            V[1] = Ey.real();
            V[2] = Ez.real();
         }
         else
         {
            V[0] = 0.0;
            V[1] = Ey.imag();
            V[2] = Ez.imag();
         }
      }
      break;
      case 'R': // Right Circularly Polarized, propagating along B
      {
         complex<double> kR = omega_ * sqrt(S_ + D_) / c0_;
         if (kR.imag() < 0.0) { kR *= -1.0; }

         complex<double> Ez = exp(i * kR * x[0]);
         complex<double> Ey = -i * Ez;

         if (realPart_)
         {
            V[0] = 0.0;
            V[1] = Ey.real();
            V[2] = Ez.real();
         }
         else
         {
            V[0] = 0.0;
            V[1] = Ey.imag();
            V[2] = Ez.imag();
         }
      }
      break;
      case 'O': // Ordinary wave propagating perpendicular to B
      {
         complex<double> kO = omega_ * sqrt(P_) / c0_;
         if (kO.imag() < 0.0) { kO *= -1.0; }

         complex<double> Ey = exp(i * kO * x[0]);

         if (realPart_)
         {
            V[0] = 0.0;
            V[1] = Ey.real();
            V[2] = 0.0;
         }
         else
         {
            V[0] = 0.0;
            V[1] = Ey.imag();
            V[2] = 0.0;
         }
      }
      break;
      case 'X': // eXtraordinary wave propagating perpendicular to B
      {
         complex<double> kX = omega_ * sqrt(S_ - D_ * D_ / S_) / c0_;
         if (kX.imag() < 0.0) { kX *= -1.0; }

         complex<double> Ez = exp(i * kX * x[0]);
         complex<double> Ex = -i * D_ * Ez / S_;

         if (realPart_)
         {
            V[0] = Ex.real();
            V[1] = 0.0;
            V[2] = Ez.real();
         }
         else
         {
            V[0] = Ex.imag();
            V[1] = 0.0;
            V[2] = Ez.imag();
         }
      }
      break;
      case 'J':  // Slab of current density perpendicular to propagation
      {
         if (k_.Size() == 0)
         {
            complex<double> kE = omega_ * sqrt(S_ - D_ * D_ / S_) / c0_;

            complex<double> skL = sin(kE * Lx_);
            complex<double> E0 = i * Jy_ /
                                 (omega_ * epsilon0_ * skL *
                                  (S_ * S_ - D_ * D_));

            complex<double> Ex = i * D_ * E0;
            complex<double> Ey = S_ * E0;

            if (x[0] <= xJ_ - 0.5 * dx_)
            {
               complex<double> skLJ = sin(kE * (Lx_ - xJ_));
               complex<double> skd  = sin(kE * 0.5 * dx_);
               complex<double> skx  = sin(kE * x[0]);

               Ex *= -2.0 * skLJ * skd * skx;
               Ey *= -2.0 * skLJ * skd * skx;
            }
            else if (x[0] <= xJ_ + 0.5 * dx_)
            {
               complex<double> ck1  = cos(kE * (Lx_ - xJ_ - 0.5 * dx_));
               complex<double> ck2  = cos(kE * (xJ_ - 0.5 * dx_));
               complex<double> skx  = sin(kE * x[0]);
               complex<double> skLx = sin(kE * (Lx_ - x[0]));

               Ex *= skL - ck1 * skx - ck2 * skLx;
               Ey *= skL - ck1 * skx - ck2 * skLx;
            }
            else
            {
               complex<double> skJ  = sin(kE * xJ_);
               complex<double> skd  = sin(kE * 0.5 * dx_);
               complex<double> skLx = sin(kE * (Lx_ - x[0]));

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
