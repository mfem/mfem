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

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

// #define DEFINITE

// #ifndef MFEM_USE_PETSC
// #error This example requires that MFEM is built with MFEM_USE_PETSC=YES
// #endif

// Define exact solution
void E_exact_Re(const Vector & x, Vector & E);
void f_exact_Re(const Vector & x, Vector & f);
void get_maxwell_solution_Re(const Vector & x, double E[], double curl2E[]);

void E_exact_Im(const Vector & x, Vector & E);
void f_exact_Im(const Vector & x, Vector & f);
void get_maxwell_solution_Im(const Vector & x, double E[], double curl2E[]);

double pml_detJ_inv_Re(const Vector &x);
double pml_detJ_inv_Im(const Vector &x);
void pml_detJ_JT_J_inv_Re(const Vector &x, DenseMatrix &M);
void pml_detJ_JT_J_inv_Im(const Vector &x, DenseMatrix &M);
void pml_detJ_inv_JT_J_Re(const Vector &x, DenseMatrix &M);
void pml_detJ_inv_JT_J_Im(const Vector &x, DenseMatrix &M);

// Mesh Size
int dim;
double omega;
int sol = 1;
bool scatter = false;
bool pml = false;
double length = 1.0;
double pml_length = 0.25;

int main(int argc, char *argv[])
{
   StopWatch chrono;

   // 1. Initialise MPI
   MPI_Session mpi(argc, argv);
   // 1. Parse command-line options.
   // geometry file
   const char *mesh_file = "../../data/one-hex.mesh";
   int order = 1;
   // number of wavelengths
   double k = 0.5;
   //
   const char *petscrc_file = "petscrc_mult_options";
   // visualization flag
   bool visualization = 1;
   // number of initial ref
   int initref = 1;
   // number of mg levels
   int ref = 1;
   // dimension
   int nd = 2;
   //
   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree)."); 
   args.AddOption(&nd, "-nd", "--dim","Problem space dimension");   
   args.AddOption(&pml, "-pml", "--pml", "-no-pml",
                  "--no-pml", "Enable PML.");         
   args.AddOption(&pml_length, "-pml_length", "--pml_length",
                  "Length of the PML region in each direction");        
   args.AddOption(&length, "-length", "--length",
                  "length of the domainin in each direction.");                  
   args.AddOption(&k, "-k", "--wavelengths",
                  "Number of wavelengths"); 
   args.AddOption(&sol, "-sol", "--exact",
                  "Exact solution flag - 0:polynomial, 1: plane wave");
   args.AddOption(&initref, "-initref", "--initref",
                  "Number of initial refinements.");
   args.AddOption(&ref, "-ref", "--refinements",
                  "Number of Refinements.");   
   args.AddOption(&scatter, "-scat", "--scattering-prob", "-no-scat",
                  "--no-scattering", "Solve a scattering problem");                  
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization."); 
   args.Parse();
   // check if the inputs are correct
   if (!args.Good())
   {
      if ( mpi.Root() )
      {
         args.PrintUsage(cout);
      }
      MPI_Finalize();
      return 1;
   }
   if ( mpi.Root() )
   {
      args.PrintOptions(cout);
   }

   // Angular frequency
   omega = 2.0*k*M_PI;

   Mesh *mesh;

   // Create serial mesh 
   double l = 1.0;
   if (nd == 2) 
   {
      if (scatter) 
      {
         mesh_file = "../../data/rectwhole7_2attr.e";
         mesh = new Mesh(mesh_file, 1, 1);
      }
      else
      {
         mesh = new Mesh(1, 1, Element::QUADRILATERAL, true, l, l, false);
      }
   }
   else
   {
      if (scatter) 
      {
         mesh_file = "../../data/hexwhole7.e";
         mesh = new Mesh(mesh_file, 1, 1);
      }
      else
      {
         mesh = new Mesh(1, 1, 1, Element::HEXAHEDRON, true, l, l, l, false);
      }
   }
    // normalize mesh
   mesh->EnsureNodes();
   GridFunction * nodes = mesh->GetNodes();
   // Assuming square/cubic domain 
   double min_coord =  nodes->Min();
   double max_coord =  nodes->Max();
   double domain_length = abs(max_coord-min_coord); 
   // shift to zero
   *nodes -= min_coord;
   // scale to one
   *nodes *= 1./domain_length;

   // Mesh *mesh = new Mesh(mesh_file, 1, 1);
   dim = mesh->Dimension();

   // 3. Executing uniform h-refinement
   for (int i = 0; i < initref; i++ )
   {
      mesh->UniformRefinement();
   }

   // create parallel mesh and delete the serial one
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;

   // Create H(curl) (Nedelec) Finite element space
   FiniteElementCollection *fec   = new ND_FECollection(order, dim);
   ParFiniteElementSpace *ND_fespace = new ParFiniteElementSpace(pmesh, fec);

   for (int i = 0; i < ref; i++)
   {
      pmesh->UniformRefinement();
      // Update fespace
      ND_fespace->Update();
   }

   // 7. Linear form b(.) (Right hand side)
   VectorFunctionCoefficient f_Re(dim, f_exact_Re);
   VectorFunctionCoefficient f_Im(dim, f_exact_Im);
   ParComplexLinearForm b(ND_fespace,ComplexOperator::HERMITIAN);
   if (! scatter)
   {
      b.AddDomainIntegrator(new VectorFEDomainLFIntegrator(f_Re), 
                            new VectorFEDomainLFIntegrator(f_Im));
   }
   b.real().Vector::operator=(0.0);
   b.imag().Vector::operator=(0.0);
   b.Assemble();

   // setup coefficients
   ConstantCoefficient muinv(1.0);
   ConstantCoefficient sigma(-pow(omega, 2));
   // pml coefficients
   FunctionCoefficient det_inv_Re(pml_detJ_inv_Re);
   FunctionCoefficient det_inv_Im(pml_detJ_inv_Im);


   MatrixFunctionCoefficient c1_Re(dim,pml_detJ_inv_JT_J_Re);
   MatrixFunctionCoefficient c1_Im(dim,pml_detJ_inv_JT_J_Im);

   MatrixFunctionCoefficient temp_c2_Re(dim,pml_detJ_JT_J_inv_Re);
   MatrixFunctionCoefficient temp_c2_Im(dim,pml_detJ_JT_J_inv_Im);

   ScalarMatrixProductCoefficient c2_Re(sigma,temp_c2_Re);
   ScalarMatrixProductCoefficient c2_Im(sigma,temp_c2_Im);

   // 7. Bilinear form a(.,.) on the finite element space
   ParSesquilinearForm a(ND_fespace, ComplexOperator::HERMITIAN);

   // a.AddDomainIntegrator(new CurlCurlIntegrator(muinv),NULL); 
   // a.AddDomainIntegrator(new VectorFEMassIntegrator(sigma),NULL);
   if (dim == 3)
   {
      a.AddDomainIntegrator(new CurlCurlIntegrator(c1_Re),
                            new CurlCurlIntegrator(c1_Im)); 
   }
   else
   {
      a.AddDomainIntegrator(new CurlCurlIntegrator(det_inv_Re),
                            new CurlCurlIntegrator(det_inv_Im)); 
   }
   a.AddDomainIntegrator(new VectorFEMassIntegrator(c2_Re),
                         new VectorFEMassIntegrator(c2_Im));

   
   a.Assemble();
   a.Finalize();


   Array<int> ess_tdof_list;
   Array<int> ess_bdr(pmesh->bdr_attributes.Max());
   ess_bdr = 1;
   ND_fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

   // Solution grid function
   ParComplexGridFunction E_gf(ND_fespace);
   E_gf = 0.0;
   VectorFunctionCoefficient E_Re(dim, E_exact_Re);
   VectorFunctionCoefficient E_Im(dim, E_exact_Im);
   if (sol >=0)  E_gf.ProjectBdrCoefficientTangent(E_Re,E_Im,ess_bdr);
   // E_gf.ProjectCoefficient(E_Re,E_Im);

   OperatorHandle Ah;
   Vector X, B;
   a.FormLinearSystem(ess_tdof_list, E_gf, b, Ah, X, B);

   ComplexHypreParMatrix * AZ = Ah.As<ComplexHypreParMatrix>();
   HypreParMatrix * A = AZ->GetSystemMatrix();

   if ( mpi.Root() )
   {
      cout << "Size of fine grid system: "
           << A->GetGlobalNumRows() << " x " << A->GetGlobalNumCols() << endl;
   }

   chrono.Clear();
   chrono.Start();
  
   MFEMInitializePetsc(NULL, NULL, petscrc_file, NULL);

   PetscLinearSolver * invA = new PetscLinearSolver(MPI_COMM_WORLD, "direct");
   PetscParMatrix *PA = new PetscParMatrix(A, Operator::PETSC_MATAIJ);
   invA->SetOperator(*PA);
   invA->Mult(B,X);
   delete PA;
   MFEMFinalizePetsc();
   a.RecoverFEMSolution(X,B,E_gf);


   // Compute error
   if (sol >= 0 && !pml)
   {
      int order_quad = max(2, 2 * order + 1);
      const IntegrationRule *irs[Geometry::NumGeom];
      for (int i = 0; i < Geometry::NumGeom; ++i)
      {
         irs[i] = &(IntRules.Get(i, order_quad));
      }

      double L2Error_Re = E_gf.real().ComputeL2Error(E_Re, irs);
      double norm_E_Re = ComputeGlobalLpNorm(2, E_Re, *pmesh, irs);

      double L2Error_Im = E_gf.imag().ComputeL2Error(E_Im, irs);
      double norm_E_Im = ComputeGlobalLpNorm(2, E_Im, *pmesh, irs);

      if (mpi.Root())
      {
         cout << " Real Part: || E_h - E || / ||E|| = " << L2Error_Re / norm_E_Re << '\n' << endl;
         cout << " Imag Part: || E_h - E || / ||E|| = " << L2Error_Im / norm_E_Im << '\n' << endl;

         cout << " Real Part: || E_h - E || = " << L2Error_Re << '\n' << endl;
         cout << " Imag Part: || E_h - E || = " << L2Error_Im << '\n' << endl;
      }
   }

   // visualization   
   if (visualization)
   {
      int num_procs, myid;
      MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
      MPI_Comm_rank(MPI_COMM_WORLD, &myid);
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock << "parallel " << num_procs << " " << myid << "\n";
      sol_sock.precision(8);
      string keys;
      if (dim ==3) 
      {
         keys = "keys mF\n";
      }
      else
      {
         keys = "keys mrRljcUUuu\n";
      }
      sol_sock << "solution\n" << *pmesh << E_gf.real() << "window_title 'Real part'" 
      << keys << flush;
   }
   delete fec;
   delete ND_fespace;
   delete pmesh;
   return 0;
}
//define exact solution
void E_exact_Re(const Vector &x, Vector &E)
{
   double curl2E[3];
   get_maxwell_solution_Re(x, E, curl2E);
}

//calculate RHS from exact solution
void f_exact_Re(const Vector &x, Vector &f)
{
   double E_Re[3], curl2E_Re[3];
   double E_Im[3], curl2E_Im[3];

   get_maxwell_solution_Re(x, E_Re, curl2E_Re);
   get_maxwell_solution_Re(x, E_Im, curl2E_Im);

   // curl ( curl E) - omega^2 E = f
   double coeff;
   coeff = -omega * omega;
   f(0) = curl2E_Re[0] + coeff * E_Re[0];
   f(1) = curl2E_Re[1] + coeff * E_Re[1];
   if (dim == 3)
   {
      f(2) = curl2E_Re[2] + coeff * E_Re[2];
   }

   if (sol < 0)
   {
      double x0 = length/2.0;
      double x1 = length/2.0;
      double x2 = length/2.0;
      double alpha,beta;
      double n = 5.0 * omega/M_PI;
      double coeff = pow(n,2)/M_PI;
      beta = pow(x0-x(0),2) + pow(x1-x(1),2);
      if (dim == 3) beta += pow(x2-x(2),2);
      alpha = -pow(n,2) * beta;
      f = 0.0;
      f(0) = coeff*exp(alpha);
      f(1) = coeff*exp(alpha);
      if (dim == 3) f(2) = coeff*exp(alpha);
   }

}

void get_maxwell_solution_Re(const Vector & x, double E[], double curl2E[])
{
   if (sol == 0) // polynomial
   {
      if (dim == 2)
      {
         E[0] = x[1] * (1.0 - x[1])* x[0] * (1.0 - x[0]);
         E[1] = x[1] * (1.0 - x[1])* x[0] * (1.0 - x[0]);
         E[2] = 0.0;
         curl2E[0] = -2.0 * x[0] * x[0] + 4.0*x[0]*x[1] - 2.0*x[1] + 1.0;
         curl2E[1] = x[0] * (4.0 * x[1] - 2.0) - 2.0 * x[1] * x[1] + 1.0;
         curl2E[2] = 0.0;
      }
      else
      {
         E[0] = x[1] * x[2]      * (1.0 - x[1]) * (1.0 - x[2]);
         E[1] = x[0] * x[1] * x[2] * (1.0 - x[0]) * (1.0 - x[2]);
         E[2] = x[0] * x[1]      * (1.0 - x[0]) * (1.0 - x[1]);
         curl2E[0] = 2.0 * x[1] * (1.0 - x[1]) - (2.0 * x[0] - 3.0) * x[2] * (1 - x[2]);
         curl2E[1] = 2.0 * x[1] * (x[0] * (1.0 - x[0]) + (1.0 - x[2]) * x[2]);
         curl2E[2] = 2.0 * x[1] * (1.0 - x[1]) + x[0] * (3.0 - 2.0 * x[2]) * (1.0 - x[0]);
      }
   }
   else if (sol == 1)
   {
      if (dim == 2)
      {
         double alpha = omega / sqrt(2);
         E[0] = cos(alpha*(x(0) + x(1)));
         E[1] = 0.0;
         E[2] = 0.0;
         curl2E[0] = alpha * alpha * E[0];
         curl2E[1] = -alpha * alpha * E[0];
         curl2E[2] = 0.0;
      }
      else
      {
         double alpha = omega / sqrt(3);
         E[0] = cos(alpha*(x(0) + x(1) + x(2)));
         E[1] = 0.0;
         E[2] = 0.0;

         curl2E[0] = 2.0 * alpha * alpha * E[0];
         curl2E[1] = -alpha * alpha * E[0];
         curl2E[2] = -alpha * alpha * E[0];
      }
   }
   else if (sol == 2)
   {
      if (dim == 2)
      {
         double shift = 0.1;
         if (scatter) shift = -0.5;
         double x0 = x(0) + shift;
         double x1 = x(1) + shift;
         double r = sqrt(x0 * x0 + x1 * x1);
         E[0] = cos(x0);
         E[1] = 0.0;
         double r_x = x0 / r;
         double r_y = x1 / r;
         double r_xx = (1.0 / r) * (1.0 - r_x * r_x);
         double r_xy = -(r_x / r) * r_y;
         double r_yx = r_xy;
         double r_yy = (1.0 / r) * (1.0 - r_y * r_y);
         curl2E[0] =  omega*(r_yy * sin(omega * r) + omega * r_y * r_y * cos(omega * r));
         curl2E[1] = -omega*(r_yx * sin(omega * r) + omega * r_y * r_x * cos(omega * r));
      }
      else
      {
         double shift = 0.1;
         if (scatter) shift = -0.5;
         double x0 = x(0) + shift;
         double x1 = x(1) + shift;
         double x2 = x(2) + shift;
         double r = sqrt(x0 * x0 + x1 * x1 + x2 * x2);
         E[0] = cos(omega * r);
         E[1] = 0.0;
         E[2] = 0.0;
         double r_x = x0 / r;
         double r_y = x1 / r;
         double r_z = x2 / r;
         double r_xy = -(r_x / r) * r_y;
         double r_xz = -(r_x / r) * r_z;
         double r_yx = r_xy;
         double r_yy = (1.0 / r) * (1.0 - r_y * r_y);
         double r_zx = r_xz;
         double r_zz = (1.0 / r) * (1.0 - r_z * r_z);

         curl2E[0] = omega * ((r_yy + r_zz) * sin(omega * r) +
                              (omega * r_y * r_y + omega * r_z * r_z) * cos(omega * r));
         curl2E[1] = -omega * (r_yx * sin(omega * r) + omega * r_y * r_x * cos(omega * r));
         curl2E[2] = -omega * (r_zx * sin(omega * r) + omega * r_z * r_x * cos(omega * r));
      }
   }
   if (pml)
   {
      if (abs(x(0)-1.) < 1e-13 || abs(x(1)-1.0) < 1e-13 ||
          abs(x(0)) < 1e-13    || abs(x(1)) < 1e-13)
      {
         E[0] = 0.0;
         E[1] = 0.0;
         curl2E[0] = 0.0;
         curl2E[1] = 0.0;
      }
      if (dim == 3)
      {
         if (abs(x(2)-1.) < 1e-13 || abs(x(2)) < 1e-13) 
         {
            E[0] = 0.0;
            E[1] = 0.0;
            E[2]=0.0;
            curl2E[0] = 0.0;
            curl2E[1] = 0.0;
            curl2E[2] = 0.0;
         }
      }
   }
}

//define exact solution
void E_exact_Im(const Vector &x, Vector &E)
{
   double curl2E[3];
   get_maxwell_solution_Re(x, E, curl2E);
}

//calculate RHS from exact solution
void f_exact_Im(const Vector &x, Vector &f)
{
   double E_Re[3], curl2E_Re[3];
   double E_Im[3], curl2E_Im[3];

   get_maxwell_solution_Re(x, E_Im, curl2E_Im);
   get_maxwell_solution_Re(x, E_Re, curl2E_Re);

   // curl ( curl E) - omega^2 E = f
   double coeff;
   coeff = -omega * omega;
   f(0) = curl2E_Im[0] + coeff * E_Im[0];
   f(1) = curl2E_Im[1] + coeff * E_Im[1];
   if (dim == 3)
   {
      f(2) = curl2E_Im[2] + coeff * E_Im[2];
   }
}

// PML
void pml_function(const Vector &x, std::vector<std::complex<double>> & dxs)
{
   double L = length;
   double n = 2.0;
   double lbeg, lend;
   double c = 50.0;
   double c1 = pml_length; 
   double c2 = length - pml_length; 
   double coeff;
   // initialize to one
   for (int i = 0; i<dim; ++i) dxs[i] = complex<double>(1.0,0.0); 

   if (pml)
   {
   // Stretch in each direction independenly
      for (int i = 0; i<dim; ++i)
      {
         if (x(i) >= c2)
         {
            lbeg = c2;
            lend = L;
            coeff = n * c / omega / pow(lend-lbeg,n); 
            dxs[i] = complex<double>(1.0,0.0) + complex<double>(0.0,coeff * pow(x(i)-lbeg, n-1.0)); 
         }
         if (x(i) <= c1)
         {
            lbeg = c1;
            lend = 0.0;
            coeff = n * c / omega / pow(lend-lbeg,n); 
            dxs[i] = complex<double>(1.0,0.0) - complex<double>(0.0, coeff * pow(x(i)-lbeg, n-1.0)); 
         } 
      }
   }
}


double pml_detJ_inv_Re(const Vector &x)
{
   std::vector<std::complex<double>> dxs(dim);
   complex<double> det(1.0,0.0);
   pml_function(x, dxs);
   for (int i=0; i<dim; ++i) det *= dxs[i];
   
   complex<double> det_inv = complex<double>(1.0,0.0)/det;
   return det_inv.real();
}
double pml_detJ_inv_Im(const Vector &x)
{
   std::vector<std::complex<double>> dxs(dim);
   complex<double> det(1.0,0.0);
   pml_function(x, dxs);
   for (int i=0; i<dim; ++i) det *= dxs[i];

   complex<double> det_inv = complex<double>(1.0,0.0)/det;
   return det_inv.imag();
}
void pml_detJ_JT_J_inv_Re(const Vector &x, DenseMatrix &M)
{
   std::vector<complex<double>> diag(dim);
   std::vector<std::complex<double>> dxs(dim);
   complex<double> det(1.0,0.0);
   pml_function(x, dxs);

   for (int i = 0; i<dim; ++i)
   {
      diag[i] = complex<double>(1.0,0.0) / pow(dxs[i],2);
      det *= dxs[i];
   }

   M.SetSize(dim);
   M=0.0;

   for (int i = 0; i<dim; ++i)
   {
      complex<double> temp = det * diag[i];
      M(i,i) = temp.real();
   }
}
void pml_detJ_JT_J_inv_Im(const Vector &x, DenseMatrix &M)
{
   std::vector<std::complex<double>> diag(dim);
   std::vector<std::complex<double>> dxs(dim);
   complex<double> det = 1.0;
   pml_function(x, dxs);

   for (int i = 0; i<dim; ++i)
   {
      diag[i] = complex<double>(1.0,0.0) / pow(dxs[i],2);
      det *= dxs[i];
   }

   M.SetSize(dim);
   M=0.0;

   for (int i = 0; i<dim; ++i)
   {
      complex<double> temp = det * diag[i];
      M(i,i) = temp.imag();
   }
}
void pml_detJ_inv_JT_J_Re(const Vector &x, DenseMatrix &M)
{
   std::vector<complex<double>> diag(dim);
   std::vector<std::complex<double>> dxs(dim);
   complex<double> det(1.0,0.0);
   pml_function(x, dxs);

   for (int i = 0; i<dim; ++i)
   {
      diag[i] = pow(dxs[i],2);
      det *= dxs[i];
   }

   M.SetSize(dim);
   M=0.0;

   for (int i = 0; i<dim; ++i)
   {
      complex<double> temp = diag[i]/det;
      M(i,i) = temp.real();
   }
}
void pml_detJ_inv_JT_J_Im(const Vector &x, DenseMatrix &M)
{
   std::vector<std::complex<double>> diag(dim);
   std::vector<std::complex<double>> dxs(dim);
   complex<double> det = 1.0;
   pml_function(x, dxs);

   for (int i = 0; i<dim; ++i)
   {
      diag[i] = pow(dxs[i],2);
      det *= dxs[i];
   }

   M.SetSize(dim);
   M=0.0;

   for (int i = 0; i<dim; ++i)
   {
      complex<double> temp = diag[i]/det;
      M(i,i) = temp.imag();
   }
}