//
// Compile with: make maxwellp
//
//               mpirun maxwell -o 2 -f 8.0 -ref 3 -prob 4 -m ../data/inline-quad.mesh
//

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include "ParDST/ParDST.hpp"
#include "common/PML.hpp"
#include "gsl/gsl_sf_airy.h"

using namespace std;
using namespace mfem;
  
void source_re(const Vector &x, Vector & f);
void source_im(const Vector &x, Vector & f);

void ExactRe(const Vector &x, Vector & E);
void ExactIm(const Vector &x, Vector & E);
void get_maxwell_solution(const Vector & x, double E[]);
void epsilon_func(const Vector &x, DenseMatrix &M);

double wavespeed(const Vector &x);

double mu = 1.0;
double epsilon = 1.0;
double omega;
int dim;
double length = 1.0;

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);
   const char *mesh_file = "../../data/inline-quad.mesh";
   int order = 1;
   // number of serial refinements
   int ser_ref_levels = 1;
   // number of parallel refinements
   int par_ref_levels = 2;
   double freq = 5.0;
   bool herm_conv = true;
   bool visualization = 1;
   int nd=2;
   int nx=2;
   int ny=2;
   int nz=2;
   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&nd, "-nd", "--dim","Problem space dimension");
   args.AddOption(&nx, "-nx", "--nx","Number of subdomains in x direction");
   args.AddOption(&ny, "-ny", "--ny","Number of subdomains in y direction");
   args.AddOption(&nz, "-nz", "--nz","Number of subdomains in z direction");               
   args.AddOption(&ser_ref_levels, "-sr", "--ser_ref_levels",
                  "Number of Serial Refinements.");
   args.AddOption(&par_ref_levels, "-pr", "--par_ref_levels",
                  "Number of Parallel Refinements."); 
   args.AddOption(&mu, "-mu", "--permeability",
                  "Permeability of free space (or 1/(spring constant)).");
   args.AddOption(&epsilon, "-eps", "--permittivity",
                  "Permittivity of free space (or mass constant).");
   args.AddOption(&freq, "-f", "--frequency",
                  "Frequency (in Hz).");
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
   omega = 2.0 * M_PI * freq;

   Mesh *mesh;


   int nel = 1;
   length = 0.5;
   mesh = new Mesh(nel, nel, nel, Element::HEXAHEDRON, true, length, length, length,false);

   dim = mesh->Dimension();

   // 4. Refine the mesh to increase the resolution.
   for (int l = 0; l < ser_ref_levels; l++)
   {
      mesh->UniformRefinement();
   }

   // 4. Define a parallel mesh by a partitioning of the serial mesh.
   // ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   // int nprocs = sqrt(num_procs);
   // MFEM_VERIFY(nprocs*nprocs == num_procs, "Check MPI partitioning");
   // int nxyz[3] = {num_procs,1,1};
   // int nxyz[3] = {nprocs,nprocs,1};
   // int nxyz[3] = {1,num_procs,1};
   int nxyz[3] = {num_procs,1,1};
   int * part = mesh->CartesianPartitioning(nxyz);
   // ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD,*mesh,part);
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD,*mesh);
   delete [] part;
   delete mesh;

   for (int l = 0; l < par_ref_levels; l++)
   {
      pmesh->UniformRefinement();
   }

   // 6. Define a finite element space on the mesh. Here we use the Nedelec
   //    finite elements of the specified order.
   FiniteElementCollection *fec = new ND_FECollection(order, dim);
   ParFiniteElementSpace *fespace = new ParFiniteElementSpace(pmesh, fec);
   HYPRE_Int size = fespace->GlobalTrueVSize();

   if (myid == 0)
   {
      cout << "Number of finite element unknowns: " << size << endl;
   }

   // 8. Setup Complex Operator convention
   ComplexOperator::Convention conv =
      herm_conv ? ComplexOperator::HERMITIAN : ComplexOperator::BLOCK_SYMMETRIC;

   // 9. Set up the linear form b(.) which corresponds to the right-hand side of
   //    the FEM linear system.
   VectorFunctionCoefficient f_re(dim, source_re);
   VectorFunctionCoefficient f_im(dim, source_im);
   ParComplexLinearForm b(fespace, conv);
   b.AddDomainIntegrator(new VectorFEDomainLFIntegrator(f_re),
                         new VectorFEDomainLFIntegrator(f_im));
   b.Vector::operator=(0.0);
   b.Assemble();

   ConstantCoefficient one(1.0);
   ConstantCoefficient omeg(-pow(omega, 2));
   MatrixFunctionCoefficient epsilon(dim,epsilon_func);

   DenseMatrix M(dim); M = 0.0;
   M(0,0) = -pow(omega, 2);
   M(1,1) = -pow(omega, 2);
   M(2,2) = -pow(omega, 2);
   MatrixConstantCoefficient Momeg(M);

   // ScalarMatrixProductCoefficient coeff(omeg,epsilon);
   MatMatCoefficient coeff(Momeg,epsilon);
   ParSesquilinearForm a(fespace, conv);
   a.AddDomainIntegrator(new CurlCurlIntegrator(one), NULL);
   // a.AddDomainIntegrator(new CurlCurlIntegrator(one), new CurlCurlIntegrator(one));
   // a.AddDomainIntegrator(new VectorFEMassIntegrator(coeff), new VectorFEMassIntegrator(coeff));
   a.AddDomainIntegrator(new VectorFEMassIntegrator(coeff),NULL);
   a.Assemble();

   Array<int> ess_tdof_list;
   Array<int> ess_bdr;
   if (pmesh->bdr_attributes.Size())
   {
      ess_bdr.SetSize(pmesh->bdr_attributes.Max());
      ess_bdr = 1;
      fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   ParComplexGridFunction x(fespace);
   x = 0.0;
   VectorFunctionCoefficient E_Re(dim, ExactRe);
   VectorFunctionCoefficient E_Im(dim, ExactIm);
   x.ProjectBdrCoefficientTangent(E_Re,E_Im,ess_bdr);
   ParComplexGridFunction Eex(fespace);
   Eex.ProjectCoefficient(E_Re,E_Im);



   OperatorHandle Ah;
   Vector B, X;
   a.FormLinearSystem(ess_tdof_list, x, b, Ah, X, B);

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


    a.RecoverFEMSolution(X, b, x);

   // If exact is known compute the error
   VectorFunctionCoefficient E_ex_Re(dim, ExactRe);
   VectorFunctionCoefficient E_ex_Im(dim, ExactIm);

   double L2Error_Re = x.real().ComputeL2Error(E_ex_Re);
   double L2Error_Im = x.imag().ComputeL2Error(E_ex_Im);

   if (myid == 0)
   {
      cout << "\n Error (Re part): || E_h - E || = "
           << L2Error_Re 
           << "\n Error (Im part): || E_h - E || = "
           << L2Error_Im 
           << "\n Total Error: "
           << sqrt(L2Error_Re*L2Error_Re + L2Error_Im*L2Error_Im) << "\n\n";
   }



   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      string keys;
      keys = "keys mc\n";
      socketstream sol_sock_re(vishost, visport);
      sol_sock_re.precision(8);
      sol_sock_re << "parallel " << num_procs << " " << myid << "\n"
                  << "solution\n" << *pmesh << x.real() << keys 
                  << "window_title 'E: Real Part' " << flush;                     

      socketstream sol_sock_im(vishost, visport);
      sol_sock_im.precision(8);
      sol_sock_im << "parallel " << num_procs << " " << myid << "\n"
                  << "solution\n" << *pmesh << x.imag() << keys 
                  << "window_title 'E: Imag Part' " << flush;                     
   }

   // // 18. Free the used memory.
   delete fespace;
   delete fec;
   delete pmesh;
   MPI_Finalize();
   return 0;
}

void source_re(const Vector &x, Vector &f)
{
   f = 0.0;
}

void source_im(const Vector &x, Vector &f)
{
   f = 0.0;
}

//define exact solution
void ExactRe(const Vector &x, Vector &E)
{
   get_maxwell_solution(x, E);
}
void ExactIm(const Vector &x, Vector &E)
{
   get_maxwell_solution(x, E);
}

void get_maxwell_solution(const Vector &x, double E[])
{
// Airy function
   E[0] = 0;
   E[1] = 0;
   double b = -pow(omega/4.0,2.0/3.0)*(4.0*x(0)-1.0);
   E[2] = gsl_sf_airy_Ai(b,GSL_PREC_DOUBLE);
}

void epsilon_func(const Vector &x, DenseMatrix &M)
{
   M.SetSize(3); M = 0.0;
   M(0,0) = 1.0;
   M(1,1) = 1.0;
   M(2,2) = 4.0*x(0)-1.0;
}