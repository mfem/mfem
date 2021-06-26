// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
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
//
// ./stix2d -rod '0 0 1 0 0 0.1' -o 3 -s 5 -rs 0 -maxit 1 -f 1e6
//
// mpirun -np 8 ./stix2d -rod '0 0 1 0 0 0.1' -dbcs '1' -w Z -o 3 -s 5 -rs 0 -maxit 1 -f 1e6
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
// ./stix2d_CM -m simple_antenna.mesh -f 8e7 -s 3 -rs 0 -o 1 -maxit 1 -ants '5 6' -antv '0.0 1.0 0.0 0.0 1.0 0.0' -B '0 0 5.4' -dp 0 -dpp 2e20 -tp 1 -tpp '2 0 0 0 1e2 1e2 1e2' -kz 10.8
//
// An attempt to match the Kohno paper (work in progress)
// mpirun -np 10 ./stix2d_CM -m simple_antenna_Kohno_per.mesh -f 8e7 -s 1 -rs 0 -o 1 -maxit 1 -ants '5 6' -antv '0.0 1.0 0.0 0.0 1.0 0.0' -dbcs '2' -dbcv '0 0 0' -B '1.5 0.5 4' -dp 0 -dpp 2e18 -tp 0 -tpp '10' -kz 10.8
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
using namespace mfem::miniapps;
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

// Storage for user-supplied, Dirichlet boundary conditions
static Vector dbcv_(0);    // Real Dirichlet value
static Vector dbcv_re_(0); // Real component of complex Dirichlet value
static Vector dbcv_im_(0); // Imaginary component of complex Dirichlet value

// Current Density Function
static Vector rod_params_
(0); // Amplitude of x, y, z current source, position in 2D, and radius
static Vector antv_(0); // Values of Efield for each antenna attribute

void rod_current_source(const Vector &x, Vector &j);
void j_src(const Vector &x, Vector &j)
{
   if (rod_params_.Size() > 0)
   {
      rod_current_source(x, j);
   }
}

void antenna_func_lhs_r(const Vector &x, Vector &Ej);
void antenna_func_rhs_r(const Vector &x, Vector &Ej);
void antenna_func_lhs_i(const Vector &x, Vector &Ej);
void antenna_func_rhs_i(const Vector &x, Vector &Ej);

// Electric Field Boundary Condition: The following function returns zero but
// any function could be used.
void e_bc_r(const Vector &x, Vector &E);
void e_bc_i(const Vector &x, Vector &E);

/*
class ColdPlasmaPlaneWave: public VectorCoefficient
{
public:
   ColdPlasmaPlaneWave(char type,
                       double omega,
                       const Vector & B,
                       const Vector & number,
                       const Vector & charge,
                       const Vector & mass,
                       double temp,
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
   double temp_;

   std::complex<double> S_;
   std::complex<double> D_;
   std::complex<double> P_;
};
*/
void Update(ParFiniteElementSpace & H1FESpace,
            ParFiniteElementSpace & HCurlFESpace,
            ParFiniteElementSpace & HDivFESpace,
            ParFiniteElementSpace & L2FESpace,
            ParGridFunction & BField,
            VectorCoefficient & BCoef,
            Coefficient & TCoef,
            Coefficient & rhoCoef,
            int & size_l2,
            int & size_h1,
            const Vector & numbers,
            Array<int> & density_offsets,
            Array<int> & temperature_offsets,
            BlockVector & density,
            BlockVector & temperature,
            ParGridFunction & density_gf,
            ParGridFunction & temperature_gf);

//static double freq_ = 1.0e9;

// Mesh Size
//static Vector mesh_dim_(0); // x, y, z dimensions of mesh

// Prints the program's logo to the given output stream
void display_banner(ostream & os);

int main(int argc, char *argv[])
{
   MPI_Session mpi(argc, argv);

   if ( mpi.Root() ) { display_banner(cout); }

   int logging = 1;

   // Parse command-line options.
   const char *mesh_file = "ellipse_origin_h0pt0625_o3.mesh";
   int ser_ref_levels = 0;
   int order = 1;
   int maxit = 100;
   int sol = 2;
   int prec = 1;
   // int nspecies = 2;
   bool herm_conv = false;
   bool vis_u = false;
   bool visualization = true;
   bool visit = true;

   double freq = 1.0e6;
   const char * wave_type = "R";

   Vector BVec(3);
   BVec = 0.0; BVec(0) = 0.1;

   bool phase_shift = false;
   Vector kVec(3);
   kVec = 0.0;

   double hz = -1.0;

   Vector numbers;
   Vector charges;
   Vector masses;
   Vector temp;

   PlasmaProfile::Type dpt = PlasmaProfile::CONSTANT;
   PlasmaProfile::Type tpt = PlasmaProfile::CONSTANT;
   Vector dpp;
   Vector tpp;

   Array<int> abcs;  // Absorbing BC attributes
   Array<int> sbcs;  // Sheath BC attributes
   Array<int> dbca;  // Dirichlet BC attributes
   Array<int> nbca;  // Dirichlet BC attributes
   Array<int> ants_; // Antenna Neumann BC attributes
   int num_elements = 10;

   SolverOptions solOpts;
   solOpts.maxIter = 1000;
   solOpts.kDim = 50;
   solOpts.printLvl = 1;
   solOpts.relTol = 1e-4;
   solOpts.euLvl = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ser_ref_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   // args.AddOption(&nspecies, "-ns", "--num-species",
   //               "Number of ion species.");
   args.AddOption(&freq, "-f", "--frequency",
                  "Frequency in Hertz (of course...)");
   args.AddOption(&hz, "-mh", "--mesh-height",
                  "Thickness of extruded mesh in meters.");
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
   args.AddOption(&BVec, "-B", "--magnetic-flux",
                  "Background magnetic flux vector");
   args.AddOption(&kVec[2], "-kz", "--wave-vector-z",
                  "z-Component of wave vector.");
   //args.AddOption(&numbers, "-num", "--number-densites",
   //               "Number densities of the various species");
   args.AddOption(&charges, "-q", "--charges",
                  "Charges of the various species "
                  "(in units of electron charge)");
   args.AddOption(&masses, "-m", "--masses",
                  "Masses of the various species (in amu)");
   //args.AddOption(&temp, "-t", "--temperature",
   //                "Temperature of the electrons (in eV)");
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
   args.AddOption(&rod_params_, "-rod", "--rod_params",
                  "3D Vector Amplitude, 2D Position, Radius");
   args.AddOption(&abcs, "-abcs", "--absorbing-bc-surf",
                  "Absorbing Boundary Condition Surfaces");
   args.AddOption(&sbcs, "-sbcs", "--sheath-bc-surf",
                  "Sheath Boundary Condition Surfaces");
   args.AddOption(&dbca, "-dbcs", "--dirichlet-bc-surf",
                  "Dirichlet Boundary Condition Surfaces");
   args.AddOption(&dbcv_, "-dbcv", "--dirichlet-bc-val",
                  "Real Dirichlet Boundary Condition Values");
   args.AddOption(&dbcv_re_, "-dbcv_re", "--dirichlet-bc-re-val",
                  "Imaginary part of complex Dirichlet values");
   args.AddOption(&dbcv_im_, "-dbcv_im", "--dirichlet-bc-im-val",
                  "Real part of complex Dirichlet values");
   args.AddOption(&nbca, "-nbcs", "--neumann-bc-surf",
                  "Neumann Boundary Condition Surfaces");
   args.AddOption(&ants_, "-ants", "--antenna-attr",
                  "Boundary attributes of antenna");
   args.AddOption(&antv_, "-antv", "--antenna-vals",
                  "Efield values at antenna attributes");
   // args.AddOption(&num_elements, "-ne", "--num-elements",
   //             "The number of mesh elements in x");
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
   args.Parse();
   if (!args.Good())
   {
      if (mpi.Root())
      {
         args.PrintUsage(cout);
      }
      return 1;
   }
   if (numbers.Size() == 0)
   {
      numbers.SetSize(2);
      if (dpp.Size() == 0)
      {
         dpp.SetSize(1);
         dpp[0] = 1.0e19;
         numbers[0] = dpp[0];
         numbers[1] = dpp[0];
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
   if (temp.Size() == 0)
   {
      temp.SetSize(2);
      if (tpp.Size() == 0)
      {
         tpp.SetSize(1);
         tpp[0] = 1.0e3;
         temp[0] = tpp[0];
         temp[1] = tpp[0];
      }
      else
      {
         switch (tpt)
         {
            case PlasmaProfile::CONSTANT:
               temp[0] = tpp[0];
               temp[1] = tpp[0];
               break;
            case PlasmaProfile::GRADIENT:
               temp[0] = tpp[0];
               temp[1] = tpp[0];
               break;
            case PlasmaProfile::TANH:
               temp[0] = tpp[1];
               temp[1] = tpp[1];
               break;
            default:
               temp[0] = 1.0e3;
               temp[1] = 1.0e3;
               break;
         }
      }
   }
   if (num_elements <= 0)
   {
      num_elements = 10;
   }
   if (hz < 0.0)
   {
      hz = 0.1;
   }
   double omega = 2.0 * M_PI * freq;
   if (kVec[2] != 0.0)
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
      // double lam0 = c0_ / freq;
      double Bmag = BVec.Norml2();
      std::complex<double> S = S_cold_plasma(omega, Bmag, numbers,
                                             charges, masses, temp);
      std::complex<double> P = P_cold_plasma(omega, numbers,
                                             charges, masses, temp);
      std::complex<double> D = D_cold_plasma(omega, Bmag, numbers,
                                             charges, masses, temp);
      std::complex<double> R = R_cold_plasma(omega, Bmag, numbers,
                                             charges, masses, temp);
      std::complex<double> L = L_cold_plasma(omega, Bmag, numbers,
                                             charges, masses, temp);

      cout << "\nConvenient Terms:\n";
      cout << "R = " << R << ",\tL = " << L << endl;
      cout << "S = " << S << ",\tD = " << D << ",\tP = " << P << endl;

      cout << "\nSpecies Properties (number, charge, mass):\n";
      for (int i=0; i<numbers.Size(); i++)
      {
         cout << numbers[i] << '\t' << charges[i] << '\t' << masses[i] <<'\n';
      }
      cout << "\nPlasma and Cyclotron Frequencies by Species (GHz):\n";
      for (int i=0; i<numbers.Size(); i++)
      {
         cout << real(omega_p(numbers[i], charges[i], masses[i]) / (2.0e9 * M_PI))
              << '\t'
              << real(omega_c(Bmag, charges[i], masses[i]) / (2.0e9 * M_PI)) << '\n';
      }
      /*
            cout << "\nWavelengths (meters):\n";
            cout << "   Free Space Wavelength: " << lam0 << '\n';
            if (S < D)
            {
               cout << "   Decaying L mode:       " << lam0 / sqrt(D-S) << '\n';
            }
            else
            {
               cout << "   Oscillating L mode:    " << lam0 / sqrt(S-D) << '\n';
            }
            if (S < - D)
            {
               cout << "   Decaying R mode:       " << lam0 / sqrt(-S-D) << '\n';
            }
            else
            {
               cout << "   Oscillating R mode:    " << lam0 / sqrt(S+D) << '\n';
            }
            if (P < 0)
            {
               cout << "   Decaying O mode:       " << lam0 / sqrt(-P) << '\n';
            }
            else
            {
               cout << "   Oscillating O mode:    " << lam0 / sqrt(P) << '\n';
            }
            if ((S * S - D * D) / S < 0)
            {
               cout << "   Decaying X mode:       " << lam0 * sqrt(-S/(S*S-D*D))
                    << '\n';
            }
            else
            {
               cout << "   Oscillating X mode:    " << lam0 * sqrt(S/(S*S-D*D))
                    << '\n';
            }
            cout << endl;
       */
   }

   // Read the (serial) mesh from the given mesh file on all processors.  We
   // can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
   // and volume meshes with the same code.
   if ( mpi.Root() && logging > 0 ) { cout << "Building Mesh ..." << endl; }

   tic_toc.Clear();
   tic_toc.Start();

   // Mesh * mesh = new Mesh(num_elements, 3, 3, Element::HEXAHEDRON, 1,
   //                      mesh_dim_(0), mesh_dim_(1), mesh_dim_(2));
   Mesh * mesh2d = new Mesh(mesh_file, 1, 1);
   for (int lev = 0; lev < ser_ref_levels; lev++)
   {
      mesh2d->UniformRefinement();
   }
   Mesh * mesh = Extrude2D(mesh2d, 3, hz);
   delete mesh2d;
   {
      /*
      vector<Vector> trans(1);
      trans[0].SetSize(3);
      trans[0] = 0.0; trans[0][2] = hz;
      */
      Array<int> v2v(mesh->GetNV());
      for (int i=0; i<v2v.Size(); i++) { v2v[i] = i; }
      for (int i=0; i<mesh->GetNV() / 4; i++) { v2v[4 * i + 3] = 4 * i; }

      Mesh * per_mesh = miniapps::MakePeriodicMesh(mesh, v2v);
      /*
      ofstream ofs("per_mesh.mesh");
      per_mesh->Print(ofs);
      ofs.close();
      cout << "Chekcing eltrans from mesh" << endl;
      for (int i=0; i<mesh->GetNBE(); i++)
      {
        ElementTransformation * eltrans = mesh->GetBdrElementTransformation(i);
        cout << i
        << '\t' << eltrans->ElementNo
        << '\t' << eltrans->Attribute
        << endl;
      }
      cout << "Chekcing eltrans from per_mesh" << endl;
      for (int i=0; i<per_mesh->GetNBE(); i++)
      {
        ElementTransformation * eltrans = per_mesh->GetBdrElementTransformation(i);
        cout << i
        << '\t' << eltrans->ElementNo
        << '\t' << eltrans->Attribute
        << endl;
      }
      */
      delete mesh;
      mesh = per_mesh;
   }
   tic_toc.Stop();

   if (mpi.Root() && logging > 0 )
   {
      cout << " done in " << tic_toc.RealTime() << " seconds." << endl;
   }
   if (mpi.Root())
   {
      cout << "Starting initialization." << endl;
   }

   // Ensure that quad and hex meshes are treated as non-conforming.
   mesh->EnsureNCMesh();

   // Define a parallel mesh by a partitioning of the serial mesh. Refine
   // this mesh further in parallel to increase the resolution. Once the
   // parallel mesh is defined, the serial mesh can be deleted.
   ParMesh pmesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   /*
   {
     for (int i=0; i<pmesh.GetNBE(); i++)
       {
    cout << i << '\t' << pmesh.GetBdrElementBaseGeometry(i)
         << '\t' << pmesh.GetBdrAttribute(i) << endl;
       }
   }
   */
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
   VectorConstantCoefficient BCoef(BVec);
   VectorConstantCoefficient kCoef(kVec);
   /*
   double ion_frac = 0.0;
   ConstantCoefficient rhoCoef1(rho1);
   ConstantCoefficient rhoCoef2(rhoCoef1.constant * (1.0 - ion_frac));
   ConstantCoefficient rhoCoef3(rhoCoef1.constant * ion_frac);
   */
   // ConstantCoefficient tempCoef(temp * q_);
   H1_ParFESpace H1FESpace(&pmesh, order, pmesh.Dimension());
   ND_ParFESpace HCurlFESpace(&pmesh, order, pmesh.Dimension());
   RT_ParFESpace HDivFESpace(&pmesh, order, pmesh.Dimension());
   L2_ParFESpace L2FESpace(&pmesh, order, pmesh.Dimension());

   ParGridFunction BField(&HDivFESpace);
   ParGridFunction temperature_gf;
   ParGridFunction density_gf;

   if (mpi.Root())
   {
      cout << "Projecting B." << endl;
   }
   BField.ProjectCoefficient(BCoef);

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

   PlasmaProfile rhoCoef(dpt, dpp);
   PlasmaProfile tempCoef(tpt, tpp);

   if (mpi.Root())
   {
      cout << "Projecting rho." << endl;
   }
   for (int i=0; i<numbers.Size(); i++)
   {
      // ConstantCoefficient rhoCoef(numbers[i]);
      density_gf.MakeRef(&L2FESpace, density.GetBlock(i));
      density_gf.ProjectCoefficient(rhoCoef);
   }

   if (mpi.Root())
   {
      cout << "Projecting T." << endl;
   }
   for (int i=0; i<=numbers.Size(); i++)
   {
      temperature_gf.MakeRef(&H1FESpace, temperature.GetBlock(i));
      temperature_gf.ProjectCoefficient(tempCoef);
   }
   /*
   density_gf.MakeRef(&L2FESpace, density.GetBlock(0));
   density_gf.ProjectCoefficient(rhoCoef1);

   density_gf.MakeRef(&L2FESpace, density.GetBlock(1));
   density_gf.ProjectCoefficient(rhoCoef2);

   density_gf.MakeRef(&L2FESpace, density.GetBlock(2));
   density_gf.ProjectCoefficient(rhoCoef3);
   */

   // Create a coefficient describing the magnetic permeability
   ConstantCoefficient muInvCoef(1.0 / mu0_);

   // Create a coefficient describing the surface admittance
   Coefficient * etaInvCoef = SetupRealAdmittanceCoefficient(pmesh, abcs);

   Coefficient * etaInvReCoef = NULL;
   Coefficient * etaInvImCoef = NULL;
   SetupComplexAdmittanceCoefs(pmesh, sbcs, etaInvReCoef, etaInvImCoef);

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

   /*
   ColdPlasmaPlaneWave EReCoef(wave_type[0], omega, BVec,
                              numbers, charges, masses, temp, true);
   ColdPlasmaPlaneWave EImCoef(wave_type[0], omega, BVec,
                              numbers, charges, masses, temp, false);
   */
   /*
   if (wave_type[0] == 'J' && slab_params_.Size() == 5)
   {
      EReCoef.SetCurrentSlab(slab_params_[1], slab_params_[3], slab_params_[4],
                             mesh_dim_[0]);
      EImCoef.SetCurrentSlab(slab_params_[1], slab_params_[3], slab_params_[4],
                             mesh_dim_[0]);
   }
   */
   /*
   if (phase_shift)
   {
     EReCoef.SetPhaseShift(kVec);
     EImCoef.SetPhaseShift(kVec);
   }
    */
   /*if (visualization)
   {
      ParComplexGridFunction EField(&HCurlFESpace);
      EField.ProjectCoefficient(EReCoef, EImCoef);

      char vishost[] = "localhost";
      int  visport   = 19916;

      int Wx = 0, Wy = 0; // window position
      int Ww = 350, Wh = 350; // window size
      int offx = Ww+10, offy = Wh+45; // window offsets

      socketstream sock_Er, sock_Ei, sock_B;
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
                     BField, "Background Magnetic Field",
                     Wx, Wy, Ww, Wh);

   }
   */

   // Setup coefficients for Dirichlet BC
   /*
   Array<ComplexVectorCoefficientByAttr> dbcs(1);
   dbcs[0].attr = dbca;
   dbcs[0].real = &EReCoef;
   dbcs[0].imag = &EImCoef;
   */

   Vector zeroVec(3); zeroVec = 0.0;
   VectorConstantCoefficient zeroCoef(zeroVec);
   Array<ComplexVectorCoefficientByAttr> dbcs(dbca.Size());

   if ( dbcv_.Size() > 0 )
   {
      MFEM_VERIFY(dbcv_.Size() == 3*dbca.Size(),
                  "Each Dirichlet boundary condition value must be associated"
                  "with exactly one Dirichlet boundary surface.");

      for (int i=0; i<dbca.Size(); i++)
      {
         Vector dbcz_(3); dbcz_[0] = dbcv_[i];
         dbcz_[1] = dbcv_[i+1]; dbcz_[2] = dbcv_[i+2];
         VectorCoefficient *dbcz_Coef = new VectorConstantCoefficient(dbcz_);

         Array<int> resize(1);
         dbcs[i].attr = resize;

         dbcs[i].attr = dbca[i];
         dbcs[i].real = dbcz_Coef;
         dbcs[i].imag = &zeroCoef;
      }
   }

   if ( dbcv_re_.Size() > 0 )
   {
      MFEM_VERIFY(dbcv_re_.Size() == 3*dbca.Size() &&
                  dbcv_im_.Size() == 3*dbca.Size(),
                  "Each Dirichlet boundary condition value must be associated"
                  "with exactly one Dirichlet boundary surface.");

      for (int i=0; i<dbca.Size(); i++)
      {
         Vector dbcz_re_(3); dbcz_re_[0] = dbcv_re_[i];
         dbcz_re_[1] = dbcv_re_[i+1]; dbcz_re_[2] = dbcv_re_[i+2];
         Vector dbcz_im_(3); dbcz_im_[0] = dbcv_im_[i];
         dbcz_im_[1] = dbcv_im_[i+1]; dbcz_im_[2] = dbcv_im_[i+2];
         VectorCoefficient *dbcz_reCoef = new VectorConstantCoefficient(dbcz_re_);
         VectorCoefficient *dbcz_imCoef = new VectorConstantCoefficient(dbcz_im_);

         Array<int> resize(1);
         dbcs[i].attr = resize;

         dbcs[i].attr = dbca[i];
         dbcs[i].real = dbcz_reCoef;
         dbcs[i].imag = dbcz_imCoef;
      }
   }

   /*
   Vector y(3); yVec = 0.0; yVec[1] = 1.0;
   Vector yNegVec(3); yNegVec = 0.0; yNegVec[1] = -1.0;
   VectorConstantCoefficient zeroCoef(zeroVec);
   VectorConstantCoefficient yCoef(yVec);
   VectorConstantCoefficient yNegCoef(yNegVec);

   Array<ComplexVectorCoefficientByAttr> dbcs(3);

   Array<int> dbcz(4); dbcz[0] = 1; dbcz[1] = 2; dbcz[2] = 3; dbcz[3] = 4;
   dbcs[0].attr = dbcz;
   dbcs[0].real = &zeroCoef;
   dbcs[0].imag = &zeroCoef;

   Array<int> dbcf(1); dbcf[0] = 5;
   dbcs[1].attr = dbcf;
   dbcs[1].real = &yCoef;
   dbcs[1].imag = &zeroCoef;

   Array<int> dbcb(1); dbcb[0] = 6;
   dbcs[2].attr = dbcb;
   dbcs[2].real = &yNegCoef;
   dbcs[2].imag = &zeroCoef;
   */
   Array<ComplexVectorCoefficientByAttr> nbcs(ants_.Size());

   if ( ants_.Size() > 0 )
   {
      MFEM_VERIFY(antv_.Size() == 3*ants_.Size(),
                  "Each Neumann boundary condition value must be associated"
                  "with exactly one Neumann boundary surface.");

      VectorFunctionCoefficient *antz_Coef_lhs_r = new VectorFunctionCoefficient(
         pmesh.SpaceDimension(), antenna_func_lhs_r);
      VectorFunctionCoefficient *antz_Coef_rhs_r = new VectorFunctionCoefficient(
         pmesh.SpaceDimension(), antenna_func_rhs_r);
      VectorFunctionCoefficient *antz_Coef_lhs_i = new VectorFunctionCoefficient(
         pmesh.SpaceDimension(), antenna_func_lhs_i);
      VectorFunctionCoefficient *antz_Coef_rhs_i = new VectorFunctionCoefficient(
         pmesh.SpaceDimension(), antenna_func_rhs_i);

      Array<int> resize(1);
      nbcs[nbca.Size()].attr = resize;
      nbcs[nbca.Size()].attr = ants_[0];
      nbcs[nbca.Size()].real = antz_Coef_lhs_r;
      nbcs[nbca.Size()].imag = antz_Coef_lhs_i;

      nbcs[1+nbca.Size()].attr = resize;
      nbcs[1+nbca.Size()].attr = ants_[1];
      nbcs[1+nbca.Size()].real = antz_Coef_rhs_r;
      nbcs[1+nbca.Size()].imag = antz_Coef_rhs_i;
   }

   cout << "boundary attr: " << pmesh.bdr_attributes.Size() << endl;

   // Create the Magnetostatic solver
   if (mpi.Root())
   {
      cout << "Creating Cold Plasma Dielectric Solver." << endl;
   }
   CPDSolver CPD(pmesh, order, omega,
                 (CPDSolver::SolverType)sol, solOpts,
                 (CPDSolver::PrecondType)prec,
                 conv, BCoef, epsilon_real, epsilon_imag, epsilon_abs,
                 muInvCoef, etaInvCoef, etaInvReCoef, etaInvImCoef,
                 (phase_shift) ? &kCoef : NULL,
                 abcs, sbcs, dbcs, nbcs,
                 // e_bc_r, e_bc_i,
                 // EReCoef, EImCoef,
                 (rod_params_.Size() > 0) ? j_src : NULL, NULL, vis_u);

   // Initialize GLVis visualization
   if (visualization)
   {
      CPD.InitializeGLVis();
   }

   // Initialize VisIt visualization
   VisItDataCollection visit_dc("STIX2D-AMR-Parallel", &pmesh);

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

      /*
      // Compute error
      double glb_error = CPD.GetError(EReCoef, EImCoef);
      if (mpi.Root())
      {
        cout << "Global L2 Error " << glb_error << endl;
      }
       */

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
         /*
              Array<Refinement> refs;
              for (int i=0; i<pmesh.GetNE(); i++)
              {
                 if (errors[i] > threshold)
                 {
                    refs.Append(Refinement(i, 3));
                 }
              }
              if (refs.Size() > 0)
              {
                 pmesh.GeneralRefinement(refs);
              }
         */
      }

      // Update the magnetostatic solver to reflect the new state of the mesh.
      Update(H1FESpace, HCurlFESpace, HDivFESpace, L2FESpace, BField, BCoef,
             tempCoef, rhoCoef, size_h1, size_l2, numbers, density_offsets,
             temperature_offsets, density, temperature, density_gf,
             temperature_gf);
      CPD.Update();

      if (pmesh.Nonconforming() && mpi.WorldSize() > 1 && false)
      {
         if (mpi.Root()) { cout << "Rebalancing ..." << endl; }
         pmesh.Rebalance();

         // Update again after rebalancing
         Update(H1FESpace, HCurlFESpace, HDivFESpace, L2FESpace, BField, BCoef,
                tempCoef, rhoCoef, size_h1, size_l2, numbers, density_offsets,
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

   // delete epsCoef;
   // delete muInvCoef;
   // delete sigmaCoef;

   for ( int i=0; i<dbcs.Size(); i++)
   {
      if ( dbcs[i].real != NULL && dbcs[i].real != &zeroCoef)
      {
         delete dbcs[i].real;
      }
      if ( dbcs[i].imag != NULL && dbcs[i].imag != &zeroCoef)
      {
         delete dbcs[i].imag;
      }
   }

   return 0;
}

void Update(ParFiniteElementSpace & H1FESpace,
            ParFiniteElementSpace & HCurlFESpace,
            ParFiniteElementSpace & HDivFESpace,
            ParFiniteElementSpace & L2FESpace,
            ParGridFunction & BField,
            VectorCoefficient & BCoef,
            Coefficient & TCoef,
            Coefficient & rhoCoef,
            int & size_h1,
            int & size_l2,
            const Vector & numbers,
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
   size_h1 = H1FESpace.GetVSize();
   for (int i=1; i<=numbers.Size(); i++)
   {
      density_offsets[i]     = density_offsets[i - 1] + size_l2;
      temperature_offsets[i + 1] = temperature_offsets[i] + size_h1;
   }
   density.Update(density_offsets);
   temperature.Update(temperature_offsets);
   for (int i=0; i<numbers.Size(); i++)
   {
      density_gf.MakeRef(&L2FESpace, density.GetBlock(i));
      density_gf.ProjectCoefficient(rhoCoef);
   }
   for (int i=0; i<=numbers.Size(); i++)
   {
      temperature_gf.MakeRef(&H1FESpace, temperature.GetBlock(i));
      temperature_gf.ProjectCoefficient(TCoef);
   }
}

// Print the stix2d ascii logo to the given ostream
void display_banner(ostream & os)
{
   os << "  _________ __   __       ________      ___" << endl
      << " /   _____//  |_|__|__  __\\_____  \\  __| _/" << endl
      << " \\_____  \\\\   __\\  \\  \\/  //  ____/ / __ | " << endl
      << " /        \\|  | |  |>    </       \\/ /_/ | " << endl
      << "/_______  /|__| |__/__/\\_ \\_______ \\____ | " << endl
      << "        \\/               \\/       \\/    \\/ "  << endl
      << endl
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
   if (pw_eta_re_.Size() > 0)
   {
      MFEM_VERIFY(pw_eta_re_.Size() == sbcs.Size() &&
                  pw_eta_im_.Size() == sbcs.Size(),
                  "Each impedance value must be associated with exactly one "
                  "sheath boundary surface.");

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

void rod_current_source(const Vector &x, Vector &j)
{
   MFEM_ASSERT(x.Size() == 3, "current source requires 3D space.");

   j.SetSize(x.Size());
   j = 0.0;

   double x0 = rod_params_(3);
   double y0 = rod_params_(4);
   double radius = rod_params_(5);

   double r2 = (x(0) - x0) * (x(0) - x0) + (x(1) - y0) * (x(1) - y0);

   if (r2 <= radius * radius)
   {
      j(0) = rod_params_(0);
      j(1) = rod_params_(1);
      j(2) = rod_params_(2);
   }
   // j *= height;
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

void antenna_func_lhs_r(const Vector &x, Vector &Ej)
{
   MFEM_ASSERT(x.Size() == 3, "current source requires 3D space.");

   double kz = 0.0;

   Ej.SetSize(x.Size());
   Ej = 0.0;

   Ej(0) = antv_[0];
   Ej(1) = (antv_[1] / 2.0) * ( sin(M_PI * (((2.0 * x(1) - 0.4 + 0.05 ) / 0.05) -
                                            0.5)) + 1.0 );
   Ej(2) = antv_[2];
   Ej *= cos(kz*x(2));
}

void antenna_func_rhs_r(const Vector &x, Vector &Ej)
{
   MFEM_ASSERT(x.Size() == 3, "current source requires 3D space.");

   double kz = 0.0;

   Ej.SetSize(x.Size());
   Ej = 0.0;

   Ej(0) = antv_[3];
   Ej(1) = (antv_[4] / 2.0) * ( sin(M_PI * (((2.0 * x(1) - 0.4 + 0.05 ) / 0.05) -
                                            0.5)) + 1.0 );
   Ej(2) = antv_[5];
   Ej *= cos(kz*x(2));
}

void antenna_func_lhs_i(const Vector &x, Vector &Ej)
{
   MFEM_ASSERT(x.Size() == 3, "current source requires 3D space.");

   double kz = 0.0;

   Ej.SetSize(x.Size());
   Ej = 0.0;

   Ej(0) = antv_[0];
   Ej(1) = (antv_[1] / 2.0) * ( sin(M_PI * (((2.0 * x(1) - 0.4 + 0.05 ) / 0.05) -
                                            0.5)) + 1.0 );
   Ej(2) = antv_[2];
   Ej *= sin(kz*x(2));
}

void antenna_func_rhs_i(const Vector &x, Vector &Ej)
{
   MFEM_ASSERT(x.Size() == 3, "current source requires 3D space.");

   double kz = 0.0;

   Ej.SetSize(x.Size());
   Ej = 0.0;

   Ej(0) = antv_[3];
   Ej(1) = (antv_[4] / 2.0) * ( sin(M_PI * (((2.0 * x(1) - 0.4 + 0.05 ) / 0.05) -
                                            0.5)) + 1.0 );
   Ej(2) = antv_[5];
   Ej *= sin(kz*x(2));
}


/*
ColdPlasmaPlaneWave::ColdPlasmaPlaneWave(char type,
                                         double omega,
                                         const Vector & B,
                                         const Vector & number,
                                         const Vector & charge,
                                         const Vector & mass,
                                         double temp,
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
     masses_(mass)
     temp_(temp)
{
   S_ = S_cold_plasma(omega_, Bmag_, numbers_, charges_, masses_, temp_);
   D_ = D_cold_plasma(omega_, Bmag_, numbers_, charges_, masses_, temp_);
   P_ = P_cold_plasma(omega_, numbers_, charges_, masses_, temp_);
}

void ColdPlasmaPlaneWave::Eval(Vector &V, ElementTransformation &T,
                               const IntegrationPoint &ip)
{
   V.SetSize(3);

   double x_data[3];
   Vector x(x_data, 3);
   T.Transform(ip, x);

   switch (type_)
   {
      case 'L':
      {
         bool osc = S_ - D_ > 0.0;
         double kL = omega_ * sqrt(fabs(S_-D_)) / c0_;

         if (realPart_)
         {
            V[0] = 0.0;
            V[1] = osc ?  sin(kL * x[0]) : 0.0;
            V[2] = osc ?  cos(kL * x[0]) : exp(-kL * x[0]);
         }
         else
         {
            V[0] = 0.0;
            V[1] = osc ?  cos(kL * x[0]) : exp(-kL * x[0]);
            V[2] = osc ? -sin(kL * x[0]) : 0.0;
         }
      }
      break;
      case 'R':
      {
         bool osc = S_ + D_ > 0.0;
         double kR = omega_ * sqrt(fabs(S_+D_)) / c0_;

         if (realPart_)
         {
            V[0] = 0.0;
            V[1] = osc ? -sin(kR * x[0]) : 0.0;
            V[2] = osc ?  cos(kR * x[0]) : exp(-kR * x[0]);
         }
         else
         {
            V[0] = 0.0;
            V[1] = osc ? -cos(kR * x[0]) : -exp(-kR * x[0]);
            V[2] = osc ? -sin(kR * x[0]) : 0.0;
         }
      }
      break;
      case 'O':
      {
         bool osc = P_ > 0.0;
         double kO = omega_ * sqrt(fabs(P_)) / c0_;

         if (realPart_)
         {
            V[0] = 0.0;
            V[1] = osc ? cos(kO * x[0]) : exp(-kO * x[0]);
            V[2] = 0.0;
         }
         else
         {
            V[0] = 0.0;
            V[1] = osc ? -sin(kO * x[0]) : 0.0;
            V[2] = 0.0;
         }
      }
      break;
      case 'X':
      {
         bool osc = (S_ * S_ - D_ * D_) / S_ > 0.0;
         double kE = omega_ * sqrt(fabs((S_ * S_ - D_ * D_) / S_)) / c0_;

         if (realPart_)
         {
            V[0] = osc ? -D_ * sin(kE * x[0]) : 0.0;
            V[1] = 0.0;
            V[2] = osc ?  S_ * cos(kE * x[0]) : S_ * exp(-kE * x[0]);
         }
         else
         {
            V[0] = osc ? -D_ * cos(kE * x[0]) : -D_ * exp(-kE * x[0]);
            V[1] = 0.0;
            V[2] = osc ? -S_ * sin(kE * x[0]) : 0.0;
         }
         V /= sqrt(S_ * S_ + D_ * D_);
      }
      break;
      case 'J':
      {
         if (k_.Size() == 0)
         {
            bool osc = (S_ * S_ - D_ * D_) / S_ > 0.0;
            double kE = omega_ * sqrt(fabs((S_ * S_ - D_ * D_) / S_)) / c0_;

            double (*sfunc)(double) = osc ?
                                      static_cast<double (*)(double)>(&sin) :
                                      static_cast<double (*)(double)>(&sinh);
            double (*cfunc)(double) = osc ?
                                      static_cast<double (*)(double)>(&cos) :
                                      static_cast<double (*)(double)>(&cosh);

            double skL   = (*sfunc)(kE * Lx_);
            double csckL = 1.0 / skL;

            if (realPart_)
            {
               V[0] = D_ / S_;
               V[1] = 0.0;
               V[2] = 0.0;
            }
            else
            {
               V[0] = 0.0;
               V[1] = -1.0;
               V[2] = 0.0;
            }

            if (x[0] <= xJ_ - 0.5 * dx_)
            {
               double skx    = (*sfunc)(kE * x[0]);
               double skLxJ  = (*sfunc)(kE * (Lx_ - xJ_));
               double skd    = (*sfunc)(kE * 0.5 * dx_);
               double a = skx * skLxJ * skd;

               V *= 2.0 * omega_ * mu0_ * Jy_ * a * csckL / (kE * kE);
               if (!osc) { V *= -1.0; }
            }
            else if (x[0] <= xJ_ + 0.5 * dx_)
            {
               double skx      = (*sfunc)(kE * x[0]);
               double skLx     = (*sfunc)(kE * (Lx_ - x[0]));
               double ckxJmd   = (*cfunc)(kE * (xJ_ - 0.5 * dx_));
               double ckLxJmd  = (*cfunc)(kE * (Lx_ - xJ_ - 0.5 * dx_));
               double a = skx * ckLxJmd + skLx * ckxJmd - skL;

               V *= omega_ * mu0_ * Jy_ * a * csckL / (kE * kE);
            }
            else
            {
               double skLx = (*sfunc)(kE * (Lx_ - x[0]));
               double skxJ = (*sfunc)(kE * xJ_);
               double skd  = (*sfunc)(kE * 0.5 * dx_);
               double a = skLx * skxJ * skd;

               V *= 2.0 * omega_ * mu0_ * Jy_ * a * csckL / (kE * kE);
               if (!osc) { V *= -1.0; }
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
 */
