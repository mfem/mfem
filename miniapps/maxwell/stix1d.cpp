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


// sample run 

/*
./stix1d -md 0.24 -nex 4 -ney 4 -nez 4 -dbcs '3 5' -f 80e6 
         -B '0 0 5.4' -w J -slab '0 1 0 0.06 0.02' -num '2e20 2e20' -herm
*/

#include "mfem.hpp"
#include <fstream>
#include "../common/mesh_extras.hpp"
#include <iostream>
#include <cmath>
#include <complex>
#include "stix1d.hpp"

using namespace std;
using namespace mfem;
using namespace mfem::miniapps;

#ifndef MFEM_USE_PETSC
#error This example requires that MFEM is built with MFEM_USE_PETSC=YES
#endif

void visualize_approx(ParComplexGridFunction &);
void visualize_animation(ParMesh &, ParComplexGridFunction &);
void visualize_exact(ParFiniteElementSpace &, ColdPlasmaPlaneWave &, ColdPlasmaPlaneWave &, ParGridFunction &, ParGridFunction &);
void set_plasma_defaults(Vector & , Vector & , Vector &, Vector &);
void print_plasma_info(double &, double & ,Vector & , Vector &, Vector &, Vector &);
void slab_current_source(const Vector &x, Vector &j);
void j_src(const Vector &x, Vector &j);
void e_bc_r(const Vector &x, Vector &E);
void e_bc_i(const Vector &x, Vector &E);
void B_func(const Vector &x, Vector &B);
// Used for combining scalar coefficients
double prodFunc(double a, double b) { return a * b; }

// Mesh Size
Vector mesh_dim_(0); // x, y, z dimensions of mesh

int main(int argc, char *argv[])
{
   StopWatch chrono;

   MPI_Session mpi(argc, argv);

   int order = 1;
   double freq = 1.0e9; //frequecy in hertz
   const char * wave_type = "R";
   //
   Vector BVec(3); BVec = 0.0; BVec(0) = 0.1;
   //
   bool phase_shift = false;
   //
   Vector kVec(3); kVec = 0.0;

   Vector numbers;
   Vector charges;
   Vector masses;

   Array<int> abcs;
   Array<int> dbcs;

   int num_elements_x = 8;
   int num_elements_y = 8;
   int num_elements_z = 8;
   bool herm_conv = false;

   // visualization flag
   bool visualization = true;
   // number of initial ref
   int initref = 1;
   // number of mg levels
   int maxref = 1;
   const char *petscrc_file = "petscrc_mult_options";
   OptionsParser args(argc, argv);
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree)."); 
   args.AddOption(&freq, "-f", "--frequency",
                  "Frequency in Hertz (of course...)");        
   args.AddOption(&wave_type, "-w", "--wave-type",
                  "Wave type: 'R' - Right Circularly Polarized, "
                  "'L' - Left Circularly Polarized, "
                  "'O' - Ordinary, 'X' - Extraordinary, "
                  "'J' - Current Slab (in conjunction with -slab), "
                  "'Z' - Zero");
   args.AddOption(&B_params_, "-b", "--magnetic-flux-param",
                  "Background magnetic flux parameters");
   args.AddOption(&BVec, "-B", "--magnetic-flux",
                  "Background magnetic flux vector");
   args.AddOption(&kVec[1], "-ky", "--wave-vector-y",
                  "y-Component of wave vector.");
   args.AddOption(&kVec[2], "-kz", "--wave-vector-z",
                  "z-Component of wave vector.");
   args.AddOption(&numbers, "-num", "--number-densites",
                  "Number densities of the various species");
   args.AddOption(&charges, "-q", "--charges",
                  "Charges of the various species "
                  "(in units of electron charge)");
   args.AddOption(&masses, "-m", "--masses",
                  "Masses of the various species (in amu)");
   args.AddOption(&initref, "-initref", "--initref",
                  "Number of initial refinements.");
   args.AddOption(&maxref, "-maxref", "--maxref",
                  "Number of Refinements.");   
   args.AddOption(&pw_eta_, "-pwz", "--piecewise-eta",
                  "Piecewise values of Impedance (one value per abc surface)");
   args.AddOption(&slab_params_, "-slab", "--slab_params",
                  "Amplitude");
   args.AddOption(&abcs, "-abcs", "--absorbing-bc-surf",
                  "Absorbing Boundary Condition Surfaces");
   args.AddOption(&dbcs, "-dbcs", "--dirichlet-bc-surf",
                  "Dirichlet Boundary Condition Surfaces");
   args.AddOption(&mesh_dim_, "-md", "--mesh_dimensions",
                  "The x, y, z mesh dimensions");
   args.AddOption(&num_elements_x, "-nex", "--num-elements",
                  "The number of mesh elements in x");
   args.AddOption(&num_elements_y, "-ney", "--num-elements",
                  "The number of mesh elements in y");
   args.AddOption(&num_elements_z, "-nez", "--num-elements",
                  "The number of mesh elements in z");                       
   args.AddOption(&herm_conv, "-herm", "--hermitian", "-no-herm",
                  "--no-hermitian", "Use convention for Hermitian operators.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();
   // check if the inputs are correct
   if (!args.Good())
   {
      if (mpi.Root())
      {
         args.PrintUsage(cout);
      }
      MPI_Finalize();
      return 1;
   }

   set_plasma_defaults(numbers, charges,masses,mesh_dim_);
   //
   double omega = 2.0 * M_PI * freq;
   if (kVec[1] != 0.0 || kVec[2] != 0.0)
   {
      phase_shift = true;
   }
   if (mpi.Root())
   {
      print_plasma_info(freq, omega, BVec, numbers, charges, masses);  
      args.PrintOptions(cout);
   }

   // Initialize PETSc
   MFEMInitializePetsc(NULL, NULL, petscrc_file, NULL);

   // Create serial mesh and enforce periodic bc
   Mesh * mesh = new Mesh(num_elements_x, num_elements_y, num_elements_z, Element::HEXAHEDRON, 1,
                          mesh_dim_(0), mesh_dim_(1), mesh_dim_(2));
   {
      vector<Vector> trans(2);
      trans[0].SetSize(3);
      trans[1].SetSize(3);
      trans[0] = 0.0; trans[0][1] = mesh_dim_[1];
      trans[1] = 0.0; trans[1][2] = mesh_dim_[2];
      Mesh * per_mesh = miniapps::MakePeriodicMesh(mesh, trans);
      delete mesh;
      mesh = per_mesh;
   }

   //3. Executing uniform h-refinement
   for (int i = 0; i < initref; i++ )
   {
      mesh->UniformRefinement();
   }

   // create parallel mesh and delete the serial one
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   int dim = mesh->Dimension();
   delete mesh;

   VectorCoefficient * BCoef = NULL;
   if (B_params_.Size()  == 7)
   {
      BCoef = new VectorFunctionCoefficient(3, B_func);
   }
   else
   {
      BCoef = new VectorConstantCoefficient(BVec);
   }
   VectorConstantCoefficient * kCoef = new VectorConstantCoefficient(kVec);
   StixLCoefficient LCoef(omega, *BCoef, numbers, charges, masses);
   // Create H(curl) (Nedelec) Finite element space
   // 4. Define a finite element space on the mesh.
   FiniteElementCollection *fec   = new ND_FECollection(order, dim);
   ParFiniteElementSpace *HCurlFESpace = new ParFiniteElementSpace(pmesh, fec);

   std::vector<HypreParMatrix*>  P(maxref);
   for (int i = 0; i < maxref; i++)
   {
      const ParFiniteElementSpace cfespace(*HCurlFESpace);
      pmesh->UniformRefinement();
      // Update fespace
      HCurlFESpace->Update();
      OperatorHandle Tr(Operator::Hypre_ParCSR);
      HCurlFESpace->GetTrueTransferOperator(cfespace, Tr);
      Tr.SetOperatorOwner(false);
      Tr.Get(P[i]);
   }


   RT_ParFESpace HDivFESpace(pmesh, order, pmesh->Dimension());
   L2_ParFESpace L2FESpace(pmesh, order, pmesh->Dimension());
   ParGridFunction BField(&HDivFESpace);
   ParGridFunction LField(&L2FESpace);
   ParGridFunction density_gf;

   BField.ProjectCoefficient(*BCoef);
   LField.ProjectCoefficient(LCoef);
   int size_l2 = L2FESpace.GetVSize();

   Array<int> density_offsets(numbers.Size() + 1);
   density_offsets[0] = 0;
   for (int i=1; i<=numbers.Size(); i++)
   {
      density_offsets[i]     = density_offsets[i - 1] + size_l2;
   }
   BlockVector density(density_offsets);
   for (int i=0; i<numbers.Size(); i++)
   {
      ConstantCoefficient rhoCoef(numbers[i]);
      density_gf.MakeRef(&L2FESpace, density.GetBlock(i));
      density_gf.ProjectCoefficient(rhoCoef);
   }

   // Create a coefficient describing the magnetic permeability
   ConstantCoefficient muInvCoef(1.0 / mu0_);

   // Create a coefficient describing the surface admittance
   Coefficient * etaInvCoef = SetupAdmittanceCoefficient(*pmesh, abcs);

   // Create tensor coefficients describing the dielectric permittivity
   DielectricTensor epsilon_real(BField, density,L2FESpace,
                                 omega, charges, masses, true);
   DielectricTensor epsilon_imag(BField, density, L2FESpace,
                                 omega, charges, masses, false);

   ColdPlasmaPlaneWave EReCoef(wave_type[0], omega, BVec, numbers, charges, masses, true);
   ColdPlasmaPlaneWave EImCoef(wave_type[0], omega, BVec, numbers, charges, masses, false);


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
   if (visualization) 
   {
      visualize_exact(*HCurlFESpace,EReCoef,EImCoef,BField, LField);
   }

   Array<int> ess_bdr;
   Array<int> ess_bdr_tdofs;

   ess_bdr.SetSize(pmesh->bdr_attributes.Max());
   if ( dbcs != NULL )
   {
      if ( dbcs.Size() == 1 && dbcs[0] == -1 )
      {
         ess_bdr = 1;
      }
      else
      {
         ess_bdr = 0;
         for (int i=0; i<dbcs.Size(); i++)
         {
            ess_bdr[dbcs[i]-1] = 1;
         }
      }
      HCurlFESpace->GetEssentialTrueDofs(ess_bdr, ess_bdr_tdofs);
   }
   ConstantCoefficient OmegaCoef(omega);
   ConstantCoefficient negOmegaCoef(-omega);

   ConstantCoefficient Omega2Coef(pow(omega,2));
   ConstantCoefficient negOmega2Coef(-pow(omega,2));

   ScalarMatrixProductCoefficient massReCoef(negOmega2Coef,epsilon_real) ;
   ScalarMatrixProductCoefficient massImCoef(negOmega2Coef,epsilon_imag) ;

   // Array of 0's and 1's marking the location of absorbing surfaces
   Array<int> abc_marker;
   Coefficient * abcCoef=nullptr;       // -omega eta^{-1}
   // Impedance of free space
   if ( abcs.Size() > 0 )
   {
      abc_marker.SetSize(pmesh->bdr_attributes.Max());
      if ( abcs.Size() == 1 && abcs[0] < 0 )
      {
         // Mark all boundaries as absorbing
         abc_marker = 1;
      }
      else
      {
         // Mark select boundaries as absorbing
         abc_marker = 0;
         for (int i=0; i<abcs.Size(); i++)
         {
            abc_marker[abcs[i]-1] = 1;
         }
      }
      if ( etaInvCoef == NULL )
      {
         etaInvCoef = new ConstantCoefficient(sqrt(epsilon0_/mu0_));
      }
      abcCoef = new TransformedCoefficient(&negOmegaCoef, etaInvCoef,prodFunc);
   }

   // Volume Current Density
   VectorCoefficient * jrCoef=nullptr;
   if (slab_params_.Size() > 0)
   {
      jrCoef = new VectorFunctionCoefficient(pmesh->SpaceDimension(),j_src);
   }
   else
   {
      Vector j(3); j=0.0;
      jrCoef = new VectorConstantCoefficient(j);
   }
   VectorCoefficient * jiCoef=nullptr;
   {
      Vector j(3); j = 0.0;
      jiCoef = new VectorConstantCoefficient(j);
   }

   VectorCoefficient * rhsrCoef=nullptr;     // Volume Current Density Function
   VectorCoefficient * rhsiCoef=nullptr;     // Volume Current Density Function

   rhsrCoef = new ScalarVectorProductCoefficient( omega, *jiCoef);
   rhsiCoef = new ScalarVectorProductCoefficient(-omega, *jrCoef);


//----------------------------------------------------------------------------------------

   ComplexOperator::Convention conv =
   herm_conv ? ComplexOperator::HERMITIAN : ComplexOperator::BLOCK_SYMMETRIC;
   // Set up the Bilinear forms
   ParSesquilinearForm *a = new ParSesquilinearForm(HCurlFESpace, conv);
   a->AddDomainIntegrator(new CurlCurlIntegrator(muInvCoef), NULL); // Only real part
   a->AddDomainIntegrator(new VectorFEMassIntegrator(massReCoef),
                          new VectorFEMassIntegrator(massImCoef));  // Both real and Imag

   Coefficient * negMuInvCoef;
   VectorCoefficient * negMuInvkCoef;
   MatrixCoefficient * negMuInvkxkxCoef;
   if (kCoef)
   {
      negMuInvCoef = new ProductCoefficient(-1.0, muInvCoef);
      negMuInvkCoef = new ScalarVectorProductCoefficient(*negMuInvCoef,*kCoef);
      negMuInvkxkxCoef = new CrossCrossCoefficient(muInvCoef, *kCoef);
      a->AddDomainIntegrator(new VectorFEMassIntegrator(*negMuInvkxkxCoef),NULL);
      a->AddDomainIntegrator(NULL, new MixedCrossCurlIntegrator(*negMuInvkCoef));
      a->AddDomainIntegrator(NULL, new MixedWeakCurlCrossIntegrator(*negMuInvkCoef));
   }
   if (abcCoef)
   {
      a->AddBoundaryIntegrator(NULL, new VectorFEMassIntegrator(*abcCoef),abc_marker);
   }
   a->Assemble();
   a->Finalize();
//----------------------------------------------------------------------------------------
   // Build grid functions
   ParComplexGridFunction *E_gf = new ParComplexGridFunction(HCurlFESpace); *E_gf = 0.0;
   ParComplexGridFunction *J_gf = new ParComplexGridFunction(HCurlFESpace); 
   J_gf->ProjectCoefficient(*jrCoef, *jiCoef);

//----------------------------------------------------------------------------------------

   // Right hand side complex linear form
   ParComplexLinearForm *b = new ParComplexLinearForm(HCurlFESpace, conv);
   b->AddDomainIntegrator(new VectorFEDomainLFIntegrator(*rhsrCoef),
                          new VectorFEDomainLFIntegrator(*rhsiCoef));
   b->real().Vector::operator=(0.0);
   b->imag().Vector::operator=(0.0);
   b->Assemble();

   OperatorHandle A;
   Vector X, B;

   E_gf->ProjectCoefficient(EReCoef,EImCoef);
   a->FormLinearSystem(ess_bdr_tdofs, *E_gf, *b, A, X, B);
   ComplexHypreParMatrix * AZ = A.As<ComplexHypreParMatrix>();

   HypreParMatrix * Ah = AZ->GetSystemMatrix();
   if (mpi.Root())
   {
      cout << "Size of fine grid system: "
           << Ah->GetGlobalNumRows() << " x " << Ah->GetGlobalNumCols() << endl;
   }
   ComplexGMGSolver M(AZ, P,ComplexGMGSolver::PETSC);
   M.SetTheta(0.5);
   M.SetSmootherType(HypreSmoother::Jacobi);


   int maxit(5000);
   double rtol(1.e-8);
   double atol(0.0);
   X = 0.0;
   GMRESSolver gmres(MPI_COMM_WORLD);
   gmres.SetAbsTol(atol);
   gmres.SetRelTol(rtol);
   gmres.SetMaxIter(maxit);
   gmres.SetOperator(*AZ);
   gmres.SetPreconditioner(M);
   gmres.SetPrintLevel(1);
   gmres.Mult(B, X);

   a->RecoverFEMSolution(X,B,*E_gf);

   // Compute error
   int order_quad = max(2, 2 * order + 1);
   const IntegrationRule *irs[Geometry::NumGeom];
   for (int i = 0; i < Geometry::NumGeom; ++i)
   {
      irs[i] = &(IntRules.Get(i, order_quad));
   }

   double L2Error_Re = E_gf->real().ComputeL2Error(EReCoef, irs);
   double norm_E_Re = ComputeGlobalLpNorm(2, EReCoef, *pmesh, irs);

   double L2Error_Im = E_gf->imag().ComputeL2Error(EImCoef, irs);
   double norm_E_Im = ComputeGlobalLpNorm(2, EImCoef, *pmesh, irs);


   if (mpi.Root())
   {
      cout << " Real Part: || E_h - E || / ||E|| = " << L2Error_Re / norm_E_Re << '\n' << endl;
      cout << " Imag Part: || E_h - E || / ||E|| = " << L2Error_Im / norm_E_Im << '\n' << endl;

      cout << " Real Part: || E_h - E || = " << L2Error_Re << '\n' << endl;
      cout << " Imag Part: || E_h - E || = " << L2Error_Im << '\n' << endl;
   }


   // visualization   
   if (visualization)
   {
      visualize_approx(*E_gf);
      // visualize_animation(*pmesh,*E_gf);
   }

   delete b;
   delete E_gf;
   delete J_gf;
   if (kCoef)
   {
      delete negMuInvCoef;
      delete negMuInvkCoef;
      delete negMuInvkxkxCoef;
   }   
   delete a;
   delete rhsrCoef;
   delete rhsiCoef;
   delete jrCoef;
   delete jiCoef;
   delete etaInvCoef;
   delete HCurlFESpace;
   delete fec;
   delete kCoef;
   delete BCoef;
   delete pmesh;

   MFEMFinalizePetsc();
}


void set_plasma_defaults(Vector & numbers, Vector & charges, Vector &masses, Vector &mesh_dim_)
{
   if (numbers.Size() == 0)
   {
      numbers.SetSize(2);
      numbers[0] = 1.0e19;
      numbers[1] = 1.0e19;
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
   if (mesh_dim_.Size() < 2 )
   {
      mesh_dim_.SetSize(3);
      mesh_dim_[0] = 1.0;
      mesh_dim_[1] = 0.1;
      mesh_dim_[2] = 0.1;
   }   
}


void print_plasma_info(double & freq , double & omega,Vector & BVec, Vector & numbers,Vector & charges, Vector & masses)
{
   double lam0 = c0_ / freq;
   double Bmag = BVec.Norml2();
   double S = S_cold_plasma(omega, Bmag, numbers, charges, masses);
   double P = P_cold_plasma(omega, numbers, charges, masses);
   double D = D_cold_plasma(omega, Bmag, numbers, charges, masses);
   double R = R_cold_plasma(omega, Bmag, numbers, charges, masses);
   double L = L_cold_plasma(omega, Bmag, numbers, charges, masses);

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
}   



void slab_current_source(const Vector &x, Vector &j)
{
   MFEM_ASSERT(x.Size() == 3, "current source requires 3D space.");

   j.SetSize(x.Size());
   j = 0.0;

   double width = slab_params_(4);
   // double height = 1.0 / width;
   double half_x_l = slab_params_(3) - 0.5 * width;
   double half_x_r = slab_params_(3) + 0.5 * width;

   if (x(0) <= half_x_r && x(0) >= half_x_l)
   {
      j(0) = slab_params_(0);
      j(1) = slab_params_(1);
      j(2) = slab_params_(2);
   }
}

void j_src(const Vector &x, Vector &j)
{
   if (slab_params_.Size() > 0)
   {
      slab_current_source(x, j);
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


void visualize_exact(ParFiniteElementSpace & HCurlFESpace, ColdPlasmaPlaneWave & EReCoef, ColdPlasmaPlaneWave & EImCoef,
                     ParGridFunction &  BField, ParGridFunction & LField)
{
   ParComplexGridFunction EField(&HCurlFESpace);
   EField.ProjectCoefficient(EReCoef, EImCoef);

   char vishost[] = "localhost";
   int  visport   = 19916;

   int Ww = 960, Wh = 540; // window size

   socketstream sock_Er, sock_Ei, sock_B, sock_L;
   sock_Er.precision(8);
   sock_Ei.precision(8);
   sock_B.precision(8);
   sock_L.precision(8);

   int Wx = 0, Wy = 0; // window position
   VisualizeField(sock_Er, vishost, visport,
                  EField.real(), "Exact Electric Field, Re(E)",
                  Wx, Wy, Ww, Wh);
   Wx += Ww;
   VisualizeField(sock_Ei, vishost, visport,
                  EField.imag(), "Exact Electric Field, Im(E)",
                  Wx, Wy, Ww, Wh);
   // Wx += Ww;
   // VisualizeField(sock_B, vishost, visport,
   //                BField, "Background Magnetic Field", Wx, Wy, Ww, Wh);

   // Wx += Ww;
   // VisualizeField(sock_L, vishost, visport,
   //                LField, "L", Wx, Wy, Ww, Wh);
}



void visualize_approx(ParComplexGridFunction &  E_gf)
{
   char vishost[] = "localhost";
   int  visport   = 19916;

   int Wx = 0, Wy = 0; // window position
   int Ww = 960, Wh = 540; // window size

   socketstream sock_cEr, sock_cEi;
   sock_cEr.precision(8);
   sock_cEi.precision(8);

   Wx += Ww;
   Wy += Wh+100;
   VisualizeField(sock_cEr, vishost, visport,
                  E_gf.real(), "Computed Electric Field, Re(E)",
                  Wx, Wy, Ww, Wh);
   Wx += Ww;
   VisualizeField(sock_cEi, vishost, visport,
                  E_gf.imag(), "Computed Electric Field, Im(E)",
                  Wx, Wy, Ww, Wh);
}


void visualize_animation(ParMesh & pmesh, ParComplexGridFunction &  E_gf)
{
   MPI_Comm comm = pmesh.GetComm();

   int num_procs, myid;
   MPI_Comm_size(comm, &num_procs);
   MPI_Comm_rank(comm, &myid);

   Vector zeroVec(3); zeroVec = 0.0;
   VectorConstantCoefficient zeroCoef(zeroVec);

   double norm_r = E_gf.real().ComputeMaxError(zeroCoef);
   double norm_i = E_gf.imag().ComputeMaxError(zeroCoef);

   char vishost[] = "localhost";
   int  visport   = 19916;

   int Wx = 0, Wy = 0; // window position
   int Ww = 1920, Wh = 1080; // window size

   socketstream sol_sock(vishost, visport);
   sol_sock << "parallel " << num_procs << " " << myid << "\n";
   sol_sock.precision(8);
   sol_sock << "solution\n" << pmesh << E_gf.real()
            << "window_title 'Harmonic Solution (t = 0.0 T)'"
            << "valuerange 0.0 " << max(norm_r, norm_i) << "\n"
            << "autoscale off\n"
            << "keys cvvv\n"
            << "pause\n" << flush;
   if (myid == 0)
      cout << "GLVis visualization paused."
           << " Press space (in the GLVis window) to resume it.\n";
   int num_frames = 24;
   int i = 0;
   while (sol_sock)
   {
      double t = (double)(i % num_frames) / num_frames;
      ostringstream oss;
      oss << "Harmonic Solution (t = " << t << " T)";

      add( cos( 2.0 * M_PI * t), E_gf.real(),
           -sin( 2.0 * M_PI * t), E_gf.imag(), E_gf.real());
      sol_sock << "parallel " << num_procs << " " << myid << "\n";
      sol_sock << "solution\n" << pmesh << E_gf.real()
               << "window_title '" << oss.str() << "'" << flush;
      i++;
   }
}