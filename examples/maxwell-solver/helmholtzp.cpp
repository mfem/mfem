//
// Compile with: make helmholtz
//
// Sample runs:  helmholtz -m ../data/one-hex.mesh
//               helmholtz -m ../data/fichera.mesh
//               helmholtz -m ../data/fichera-mixed.mesh
//
// Description:  This example code demonstrates the use of MFEM to define a
//               simple finite element discretization of the Helmholtz problem
//               -Delta p - omega^2 p = 1 with impedance boundary condition.
//
#include "mfem.hpp"
#include <fstream>
#include <iostream>
// #include "DST/DST.hpp"
#include "ParDST/ParDST.hpp"
#include "common/PML.hpp"

using namespace std;
using namespace mfem;

// Exact solution and r.h.s., see below for implementation.
double f_exact_Re(const Vector &x);
double f_exact_Im(const Vector &x);

double wavespeed(const Vector &x);

double funccoeff_re(const Vector & x);
double funccoeff_im(const Vector & x);


int dim;
double omega;
int sol = 1;
double length = 1.0;
double pml_length = 0.25;
Array2D<double>comp_bdr;

#ifndef MFEM_USE_SUPERLU
#error This example requires that MFEM is built with MFEM_USE_PETSC=YES
#endif

int main(int argc, char *argv[])
{
   // 1. Initialize MPI.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);
   // 2. Parse command-line options.
   // geometry file
   const char *mesh_file = "../../data/one-hex.mesh";
   // finite element order of approximation
   int order = 1;
   bool visualization = 1;
   // number of wavelengths
   double k = 0.5;
   // number of serial refinements
   int ser_ref_levels = 1;
   // number of parallel refinements
   int par_ref_levels = 2;
   // dimension
   int nd = 2;
   int nx=2;
   int ny=2;
   int nz=2;
   bool herm_conv = true;

   // optional command line inputs
   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&nd, "-nd", "--dim","Problem space dimension");
   args.AddOption(&nx, "-nx", "--nx","Number of subdomains in x direction");
   args.AddOption(&ny, "-ny", "--ny","Number of subdomains in y direction");
   args.AddOption(&nz, "-nz", "--nz","Number of subdomains in z direction");
   args.AddOption(&sol, "-sol", "--exact",
                  "Exact solution flag - 0:polynomial, 1: plane wave, -1: unknown exact");
   args.AddOption(&k, "-k", "--wavelengths",
                  "Number of wavelengths.");
   args.AddOption(&pml_length, "-pml_length", "--pml_length",
                  "Length of the PML region in each direction");
   args.AddOption(&length, "-length", "--length",
                  "length of the domain in each direction.");
   args.AddOption(&ser_ref_levels, "-sr", "--ser_ref_levels",
                  "Number of Serial Refinements.");
   args.AddOption(&par_ref_levels, "-pr", "--par_ref_levels",
                  "Number of Parallel Refinements.");                  
   args.AddOption(&herm_conv, "-herm", "--hermitian", "-no-herm",
                  "--no-hermitian", "Use convention for Hermitian operators.");                  
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();
   // check if the inputs are correct
   if (!args.Good())
   {
      if (myid == 0)
      {
         args.PrintUsage(cout);
      }
      MPI_Finalize();
      return 1;
   }
   if (myid == 0)
   {
      args.PrintOptions(cout);
   }
   // Angular frequency
   omega = 2.0 * M_PI * k;

   // 3. Read the mesh from the given mesh file.
   Mesh *mesh;

   if (nd == 2)
   {
      // mesh = new Mesh(mesh_file,1,1);
      mesh = new Mesh(1, 1, Element::QUADRILATERAL, true, length, length, false);
   }
   else
   {
      mesh = new Mesh(1, 1, 1, Element::HEXAHEDRON, true, length, length, length,false);
   }

   // 3. Executing uniform h-refinement
   for (int i = 0; i < ser_ref_levels; i++ )
   {
      mesh->UniformRefinement();
   }
   dim = mesh->Dimension();

   double hl = GetUniformMeshElementSize(mesh);
   int nrlayers = 1;

   Array2D<double> lengths(dim,2);
   lengths = hl*nrlayers;
   // lengths[0][1] = 0.0;
   // lengths[1][1] = 0.0;
   // lengths[1][0] = 0.0;
   // lengths[0][0] = 0.0;
   CartesianPML pml(mesh,lengths);
   pml.SetOmega(omega);
   comp_bdr.SetSize(dim,2);
   comp_bdr = pml.GetCompDomainBdr(); 

      // 4. Define a parallel mesh by a partitioning of the serial mesh.
   // ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   int nprocs = sqrt(num_procs);
   // MFEM_VERIFY(nprocs*nprocs == num_procs, "Check MPI partitioning");
   // int nxyz[3] = {num_procs,1,1};
   int nxyz[3] = {nprocs,nprocs,1};
   int * part = mesh->CartesianPartitioning(nxyz);
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD,*mesh,part);
   delete mesh;
   {
      for (int l = 0; l < par_ref_levels; l++)
      {
         pmesh->UniformRefinement();
      }
   }

   // {
   //    int nx = 2;
   //    int ny = 2;
   //    int nz = 2;
   //    int nlayers = 1;
   //    // CartesianParMeshPartition part(pmesh,nx,ny,nz,nlayers);
   //    ParMeshPartition part(pmesh,nx,ny,nz,nlayers);


      // if (visualization)
      // {
      //    char vishost[] = "localhost";
      //    int  visport   = 19916;
      //    socketstream mesh_sock(vishost, visport);
      //    mesh_sock.precision(8);
      //    mesh_sock << "parallel " << num_procs << " " << myid << "\n"
      //                << "mesh\n" << *pmesh  << flush;
      // }



   //    // cout << "myid = " << myid << endl;
   //    MPI_Finalize();
   //    return 0;
   // }

   // 6. Define a finite element space on the mesh.
   FiniteElementCollection *fec = new H1_FECollection(order, dim);
   ParFiniteElementSpace *fespace = new ParFiniteElementSpace(pmesh, fec);
   HYPRE_Int size = fespace->GlobalTrueVSize();

   if (myid == 0)
   {
      cout << "Number of finite element unknowns: " << size << endl;
   }
   // 6. Set up the linear form (Real and Imaginary part)
   FunctionCoefficient f_Re(f_exact_Re);
   FunctionCoefficient f_Im(f_exact_Im);

   // 8. Setup Complex Operator convention
   ComplexOperator::Convention conv =
      herm_conv ? ComplexOperator::HERMITIAN : ComplexOperator::BLOCK_SYMMETRIC;

   // ParLinearForm *b_Re(new ParLinearForm);
   ParComplexLinearForm b(fespace, conv);
   b.AddDomainIntegrator(new DomainLFIntegrator(f_Re),
                         new DomainLFIntegrator(f_Im));
   b.real().Vector::operator=(0.0);
   b.imag().Vector::operator=(0.0);
   b.Assemble();

   // 7. Set up the bilinear form (Real and Imaginary part)
   ConstantCoefficient one(1.0);
   ConstantCoefficient sigma(-pow(omega, 2));

   FunctionCoefficient ws(wavespeed);

   PmlMatrixCoefficient c1_re(dim,pml_detJ_JT_J_inv_Re,&pml);
   PmlMatrixCoefficient c1_im(dim,pml_detJ_JT_J_inv_Im,&pml);

   PmlCoefficient detJ_re(pml_detJ_Re,&pml);
   PmlCoefficient detJ_im(pml_detJ_Im,&pml);

   ProductCoefficient c2_re0(sigma, detJ_re);
   ProductCoefficient c2_im0(sigma, detJ_im);

   ProductCoefficient c2_re(c2_re0, ws);
   ProductCoefficient c2_im(c2_im0, ws);

   ParSesquilinearForm a(fespace,conv);
   a.AddDomainIntegrator(new DiffusionIntegrator(c1_re),
                         new DiffusionIntegrator(c1_im));
   a.AddDomainIntegrator(new MassIntegrator(c2_re),
                         new MassIntegrator(c2_im));
   a.Assemble();
   a.Finalize();

   Array<int> ess_tdof_list;
   Array<int> ess_bdr(pmesh->bdr_attributes.Max());
   ess_bdr = 1;
   fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

   // Solution grid function
   ParComplexGridFunction p_gf(fespace); p_gf = 0.0;
   OperatorHandle Ah;
   Vector X, B;

   a.FormLinearSystem(ess_tdof_list, p_gf, b, Ah, X, B);


   {
      // int nx = 4;
      // int ny = 4;
      // int nz = 2;
      int nlayers = 1;
      // ParMeshPartition part(pmesh,nx,ny,nz,nlayers);
      ParDST S(&a,lengths,omega, &ws,nlayers,nx,ny,nz);

      // ParComplexGridFunction pgf(fespace);
      // FunctionCoefficient cf_re(funccoeff_re);
      // FunctionCoefficient cf_im(funccoeff_im);
      // pgf.ProjectCoefficient(cf_re,cf_im);


      // Vector y1_re(fespace->GetTrueVSize());
      // Vector y1_im(fespace->GetTrueVSize());
      // const SparseMatrix * R = fespace->GetRestrictionMatrix();

      // Array<int> blockoffsets(3);
      // blockoffsets[0] = 0;
      // blockoffsets[1] = fespace->GetTrueVSize();
      // blockoffsets[2] = fespace->GetTrueVSize();
      // blockoffsets.PartialSum();
      // BlockVector y1(blockoffsets);


      // R->Mult(pgf.real(),y1_re);
      // R->Mult(pgf.imag(),y1_im);

      // Vector y1(2*fespace->GetTrueVSize());
      // y1.SetVector(y1_re,0);
      // y1.SetVector(y1_im,fespace->GetTrueVSize());
      
      // S.Mult(y1,X);
      S.Mult(B,X);

      a.RecoverFEMSolution(X,B,p_gf);

      if (visualization)
      {
         char vishost[] = "localhost";
         int  visport   = 19916;
         string keys;
         if (dim ==2 )
         {
            keys = "keys mrRljc\n";
         }
         else
         {
            keys = "keys mc\n";
         }
         // socketstream mesh_sock(vishost, visport);
         // mesh_sock.precision(8);
         // mesh_sock << "parallel " << num_procs << " " << myid << "\n"
         //             << "mesh\n" << *pmesh  << flush;
         socketstream sol_sock_re(vishost, visport);
         sol_sock_re.precision(8);
         sol_sock_re << "parallel " << num_procs << " " << myid << "\n"
                     << "solution\n" << *pmesh << p_gf.real() << keys 
                     << "window_title 'Numerical Pressure: Real Part' " << flush;                     

         socketstream sol_sock_im(vishost, visport);
         sol_sock_im.precision(8);
         sol_sock_im << "parallel " << num_procs << " " << myid << "\n"
                     << "solution\n" << *pmesh << p_gf.imag() << keys 
                     << "window_title 'Numerical Pressure: Imag Part' " << flush;                     
      }

      // cout << "myid = " << myid << endl;
      MPI_Finalize();
      return 0;
   }



   // solve
{
   HypreParMatrix *A = Ah.As<ComplexHypreParMatrix>()->GetSystemMatrix();
   SuperLURowLocMatrix SA(*A);
   SuperLUSolver superlu(MPI_COMM_WORLD);
   superlu.SetPrintStatistics(false);
   superlu.SetSymmetricPattern(false);
   superlu.SetColumnPermutation(superlu::PARMETIS);
   superlu.SetOperator(SA);
   superlu.Mult(B, X);
   delete A;
}
   a.RecoverFEMSolution(X,B,p_gf);

   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      string keys;
      if (dim ==2 )
      {
         keys = "keys mrRljc\n";
      }
      else
      {
         keys = "keys mc\n";
      }
      socketstream sol_sock_re(vishost, visport);
      sol_sock_re.precision(8);
      sol_sock_re << "parallel " << num_procs << " " << myid << "\n"
                  << "solution\n" << *pmesh << p_gf.real() <<
                  "window_title 'Numerical Pressure' "
                  << keys << "valuerange -0.08 0.08 \n" << flush;
   }
   delete fespace;
   delete fec;
	delete pmesh;
   MPI_Finalize();
   return 0;
}


double f_exact_Re(const Vector &x)
{
   double f_re = 0.0;
   double x0 = length/2.0;
   double x1 = length/2.0;
   double x2 = length/2.0;
   // x0 = 0.59;
   // x0 = 0.19;
   x0 = 0.25;
   // x1 = 0.768;
   // x1 = 0.168;
   x1 = 0.25;
   x2 = 0.25;
   double alpha,beta;
   // double n = 5.0*omega/M_PI;
   double n = 4.0*omega/M_PI;
   // double n = 1.0;
   // double coeff = pow(n,2)/M_PI;
   beta = pow(x0-x(0),2) + pow(x1-x(1),2);
   if (dim == 3) { beta += pow(x2-x(2),2); }
   // alpha = -pow(n,2) * beta;
   // double coeff = pow(n,2)/M_PI;
   double coeff = 16.0*omega*omega/M_PI/M_PI/M_PI;
   alpha = -pow(n,2) * beta;
   f_re = coeff*exp(alpha);

   // x0 = 0.85;
   // x1 = 0.15;
   // beta = pow(x0-x(0),2) + pow(x1-x(1),2);
   // if (dim == 3) { beta += pow(x2-x(2),2); }
   // alpha = -pow(n,2) * beta;
   // f_re += coeff*exp(alpha);

   x0 = 0.8;
   x1 = 0.4;
   beta = pow(x0-x(0),2) + pow(x1-x(1),2);
   if (dim == 3) { beta += pow(x2-x(2),2); }
   alpha = -pow(n,2) * beta;
   // f_re += coeff*exp(alpha);

   bool in_pml = false;
   for (int i = 0; i<dim; i++)
   {
      if (x(i)<=comp_bdr(i,0) || x(i)>=comp_bdr(i,1))
      {
         in_pml = true;
         break;
      }
   }
   if (in_pml) f_re = 0.0;

   return f_re;

}
double f_exact_Im(const Vector &x)
{
   double f_im;
   f_im = 0.0;
   return f_im;
}

double wavespeed(const Vector &x)
{
   double ws;
   ws = 1.0;
   return ws;
}


double funccoeff_re(const Vector & x)
{
   return sin(3*M_PI*(x.Sum()));
}

double funccoeff_im(const Vector & x)
{
   return cos(10*M_PI*(x.Sum()));
}























