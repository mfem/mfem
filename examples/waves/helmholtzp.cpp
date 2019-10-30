//
// Compile with: make helmholtz
//
// Sample runs:  helmholtz -m ../data/one-hex.mesh
//               helmholtz -m ../data/fichera.mesh
//               helmholtz -m ../data/fichera-mixed.mesh
//
// Description:  This example code demonstrates the use of MFEM to define a
//               simple finite element discretization of the Helmholtz problem
//               -Delta p - omega^2 p = 1 with impedance boundary conditiones.
//
#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

// Exact solution and r.h.s., see below for implementation.
void get_helmholtz_solution_Re(const Vector &x, double & p, double dp[], double & d2p);
void get_helmholtz_solution_Im(const Vector &x, double & p, double dp[], double & d2p);
double p_exact_Re(const Vector &x);
double p_exact_Im(const Vector &x);
double f_exact_Re(const Vector &x);
double f_exact_Im(const Vector &x);
double g_exact_Re(const Vector &x);
double g_exact_Im(const Vector &x);
void grad_exact_Re(const Vector &x, Vector &grad_Re);
void grad_exact_Im(const Vector &x, Vector &grad_Im);

int dim;
double omega;
double complex_shift;
int isol = 1;

#ifndef MFEM_USE_PETSC
#error This example requires that MFEM is built with MFEM_USE_PETSC=YES
#endif

int main(int argc, char *argv[])
{

   // 1. Initialise MPI
   int num_procs, myid;
   MPI_Init(&argc, &argv);                    // Initialise MPI
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs); //total number of processors available
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);      // Determine process identifier

   //-----------------------------------------------------------------------------

   // 2. Parse command-line options.
   // geometry file
   const char *mesh_file = "../../data/one-hex.mesh";
   // finite element order of approximation
   int order = 1;
   // static condensation flag
   bool static_cond = false;
   bool visualization = 1;
   // number of wavelengths
   double k = 0.5;
   // number of mg levels
   int maxref = 1;
   // number of initial ref
   int initref = 1;
   // complex shift
   complex_shift = 0.0;

   // PETSC
   // const char *petscrc_file = "";
   const char *petscrc_file = "petscrc_mult_options";
   // optional command line inputs
   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&isol, "-isol", "--exact",
                  "Exact solution flag - 0:polynomial, 1: plane wave");               
   args.AddOption(&k, "-k", "--wavelengths",
                  "Number of wavelengths.");
   args.AddOption(&complex_shift, "-cs", "--complex_shift",
                  "Complex shift");                    
   args.AddOption(&maxref, "-ref", "--maxref",
                  "Number of Refinements.");
   args.AddOption(&initref, "-initref", "--initref",
                  "Number of initial refinements.");
   args.AddOption(&petscrc_file, "-petscopts", "--petscopts",
                  "PetscOptions file to use.");
   args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                  "--no-static-condensation", "Enable static condensation.");
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

   // 2b. Initialize PETSc
   MFEMInitializePetsc(NULL, NULL, petscrc_file, NULL);
   //-----------------------------------------------------------------------------

   // 3. Read the mesh from the given mesh file.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   // Mesh *mesh = new Mesh(1, 1, Element::QUADRILATERAL, true, 1.0, 1.0, false);
   dim = mesh->Dimension();
   int sdim = mesh->SpaceDimension();

   // ----------------------------------------------------------------------------

   // 3. Executing uniform h-refinement
   for (int i = 0; i < initref; i++ )
   {
      mesh->UniformRefinement();
   }

   // ----------------------------------------------------------------------------

   // 5. Define a parallel mesh and delete the serial mesh.
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;

   // ----------------------------------------------------------------------------

   // 6. Define a finite element space on the mesh.
   FiniteElementCollection *fec = new H1_FECollection(order, dim);
   ParFiniteElementSpace *fespace = new ParFiniteElementSpace(pmesh, fec);

   std::vector<HypreParMatrix*>  P(maxref);

   for (int i = 0; i < maxref; i++)
   {
      const ParFiniteElementSpace cfespace(*fespace);
      pmesh->UniformRefinement();
      fespace->Update();
      OperatorHandle Tr(Operator::Hypre_ParCSR);
      fespace->GetTrueTransferOperator(cfespace, Tr);
      HypreParMatrix * Paux;
      Tr.Get(Paux);
      P[i] = new HypreParMatrix(*Paux);
   }

   // 6. Set up the linear form (Real and Imaginary part)
   FunctionCoefficient f_Re(f_exact_Re);
   FunctionCoefficient g_Re(g_exact_Re);
   VectorFunctionCoefficient grad_Re(sdim, grad_exact_Re);
   FunctionCoefficient f_Im(f_exact_Im);
   FunctionCoefficient g_Im(g_exact_Im);
   VectorFunctionCoefficient grad_Im(sdim, grad_exact_Im);

   // ParLinearForm *b_Re(new ParLinearForm);
   ParComplexLinearForm b(fespace, ComplexOperator::HERMITIAN);
   b.AddDomainIntegrator(new DomainLFIntegrator(f_Re),new DomainLFIntegrator(f_Im));
   b.AddBoundaryIntegrator(new BoundaryNormalLFIntegrator(grad_Re),
                           new BoundaryNormalLFIntegrator(grad_Im));
   b.AddBoundaryIntegrator(new BoundaryLFIntegrator(g_Re),new BoundaryLFIntegrator(g_Im));
   b.real().Vector::operator=(0.0);
   b.imag().Vector::operator=(0.0);                        
   b.Assemble();

   // 7. Set up the bilinear form (Real and Imaginary part)
   ConstantCoefficient one(1.0);
   ConstantCoefficient sigma(-pow(omega, 2));
   ConstantCoefficient alpha(complex_shift);

   ParSesquilinearForm a(fespace,ComplexOperator::HERMITIAN);
   ConstantCoefficient impedance(omega);
   a.AddDomainIntegrator(new DiffusionIntegrator(one),NULL);
   a.AddDomainIntegrator(new MassIntegrator(sigma),new MassIntegrator(alpha));
   a.AddBoundaryIntegrator(NULL,new BoundaryMassIntegrator(impedance));
   a.Assemble();
   a.Finalize();

   Array<int> ess_tdof_list;
   if (pmesh->bdr_attributes.Size())
   {
      Array<int> ess_bdr(pmesh->bdr_attributes.Max());
      ess_bdr = 0;
      fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   // Solution grid function
   ParComplexGridFunction p_gf(fespace);
   FunctionCoefficient p_Re(p_exact_Re);
   FunctionCoefficient p_Im(p_exact_Im);
   p_gf.ProjectCoefficient(p_Re,p_Im);

   OperatorHandle Ah;
   Vector X, B;
   a.FormLinearSystem(ess_tdof_list, p_gf, b, Ah, X, B);

   ComplexHypreParMatrix * AZ = Ah.As<ComplexHypreParMatrix>();
   HypreParMatrix * A = AZ->GetSystemMatrix();

   if (myid == 0)
   {
      cout << "Size of fine grid system: "
           << A->GetGlobalNumRows() << " x " << A->GetGlobalNumCols() << endl;
   }


   ComplexGMGSolver M(AZ, P,ComplexGMGSolver::CoarseSolver::PETSC);
   M.SetTheta(0.5);
   M.SetSmootherType(HypreSmoother::Jacobi);

   int maxit(5000);
   double rtol(1.e-12);
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

   a.RecoverFEMSolution(X,B,p_gf);

   int order_quad = max(2, 2*order+1);
   const IntegrationRule *irs[Geometry::NumGeom];
   for (int i=0; i < Geometry::NumGeom; ++i)
   {
      irs[i] = &(IntRules.Get(i, order_quad));
   }

   const int h1_norm_type = 1;
   double L2error;
   double H1error;

   double L2err_Re  = p_gf.real().ComputeL2Error(p_Re);
   double L2err_Im  = p_gf.imag().ComputeL2Error(p_Im);

   double loc_H1err_Re = p_gf.real().ComputeH1Error(&p_Re, &grad_Re, &one, 1.0, h1_norm_type);
   double loc_H1err_Im = p_gf.imag().ComputeH1Error(&p_Im, &grad_Im, &one, 1.0, h1_norm_type);
   double H1err_Re = GlobalLpNorm(2.0, loc_H1err_Re, MPI_COMM_WORLD);
   double H1err_Im = GlobalLpNorm(2.0, loc_H1err_Im, MPI_COMM_WORLD);
   // double norm_Re = ComputeGlobalLpNorm(2, p_Re, *pmesh, irs);
   // double norm_Im = ComputeGlobalLpNorm(2, p_Im, *pmesh, irs);
      
   L2error = sqrt(L2err_Re*L2err_Re + L2err_Im*L2err_Im);
   H1error = sqrt(H1err_Re*H1err_Re + H1err_Im*H1err_Im);
   // double L2norm = sqrt(norm_Re*norm_Re + norm_Im*norm_Im);
   if (myid == 0)
   {
      cout << " || p_h - p ||_{H^1} = " << H1error <<  endl;
      cout << " || p_h - p ||_{L^2} = " << L2error <<  endl;
      // cout << " || p_h - p ||_{L^2}/||p||_{L^2} = " << L2error/L2norm <<  endl;
   }

   delete fespace;
   delete fec;
   delete pmesh;
   MFEMFinalizePetsc();
   MPI_Finalize();
   return 0;
}

//define exact solution plane wave

void get_helmholtz_solution_Re(const Vector &x, double & p, double dp[], double & d2p)
{

   if (isol == 0) // polynomial
   {  
      p = x[0]*(1.0 - x[0]) * x[1]*(1.0 - x[1]) * x[2]*(1.0 - x[2]);
      dp[0] = (1.0 - 2.0 *x[0]) * x[1]*(1.0 - x[1]) * x[2]*(1.0 - x[2]);
      dp[1] = (1.0 - 2.0 *x[1]) * x[0]*(1.0 - x[0]) * x[2]*(1.0 - x[2]);
      dp[2] = (1.0 - 2.0 *x[2]) * x[0]*(1.0 - x[0]) * x[1]*(1.0 - x[1]);
      d2p = -2.0*(-1.0 + x[0]) * x[0] * (-1.0 + x[1]) * x[1] 
            -2.0*(-1.0 + x[0]) * x[0] * (-1.0 + x[2]) * x[2] 
            -2.0*(-1.0 + x[1]) * x[1] * (-1.0 + x[2]) * x[2];
   }
   else
   {
      double alpha;
      if (dim == 2)
      {
         alpha = omega/sqrt(2);
         p = cos(alpha * ( x(0) + x(1) ) );
         dp[0] = -alpha * sin(alpha * ( x(0) + x(1) ) );
         dp[1] = dp[0];
         d2p = -2.0 * alpha * alpha * p;
      }
      else 
      {
         alpha = omega/sqrt(3);
         p = cos(alpha * ( x(0) + x(1) + x(2) ) );
         dp[0] = -alpha * sin(alpha * ( x(0) + x(1) + x(2) ) );
         dp[1] = dp[0];
         dp[2] = dp[0];
         d2p = -3.0 * alpha * alpha * p;
      }
   }
   
}

void get_helmholtz_solution_Im(const Vector &x, double & p, double dp[], double & d2p)
{

   if (isol == 0) // polynomial
   {
      p = x[0]*(1.0 - x[0]) * x[1]*(1.0 - x[1]) * x[2]*(1.0 - x[2]);
      dp[0] = (1.0 - 2.0 *x[0]) * x[1]*(1.0 - x[1]) * x[2]*(1.0 - x[2]);
      dp[1] = (1.0 - 2.0 *x[1]) * x[0]*(1.0 - x[0]) * x[2]*(1.0 - x[2]);
      dp[2] = (1.0 - 2.0 *x[2]) * x[0]*(1.0 - x[0]) * x[1]*(1.0 - x[1]);
      d2p = -2.0*(-1.0 + x[0]) * x[0] * (-1.0 + x[1]) * x[1] 
            -2.0*(-1.0 + x[0]) * x[0] * (-1.0 + x[2]) * x[2] 
            -2.0*(-1.0 + x[1]) * x[1] * (-1.0 + x[2]) * x[2];
   }
   else
   {
      double alpha;
      if (dim == 2)
      {
         alpha = omega/sqrt(2);
         p = -sin(alpha * ( x(0) + x(1) ) );
         dp[0] = -alpha * cos(alpha * ( x(0) + x(1) ) );
         dp[1] = dp[0];
         d2p = -2.0 * alpha * alpha * p;
      }
      else 
      {
         alpha = omega/sqrt(3);
         p = -sin(alpha * ( x(0) + x(1) + x(2) ) );
         dp[0] = -alpha * cos(alpha * ( x(0) + x(1) + x(2) ) );
         dp[1] = dp[0];
         dp[2] = dp[0];
         d2p = -3.0 * alpha * alpha * p;
      }   
   }
}


double p_exact_Re(const Vector &x)
{
   double p, d2p;
   double dp[3];
   get_helmholtz_solution_Re(x, p, dp, d2p);
   return p;
}
double p_exact_Im(const Vector &x)
{
   double p, d2p;
   double dp[3];
   get_helmholtz_solution_Im(x, p, dp, d2p);
   return p;
}

//calculate RHS from exact solution f = - \Delta u
double f_exact_Re(const Vector &x)
{
   double p_re, d2p_re, p_im, d2p_im;
   double dp_re[3], dp_im[3];

   get_helmholtz_solution_Re(x, p_re, dp_re, d2p_re);
   get_helmholtz_solution_Im(x, p_im, dp_im, d2p_im);
   return -d2p_re - omega * omega * p_re - complex_shift*p_im ;
}
double f_exact_Im(const Vector &x)
{
   double p_re, d2p_re, p_im, d2p_im;
   double dp_re[3], dp_im[3];

   get_helmholtz_solution_Re(x, p_re, dp_re, d2p_re);
   get_helmholtz_solution_Im(x, p_im, dp_im, d2p_im);
   return -d2p_im - omega * omega * p_im + complex_shift*p_re;
}

void grad_exact_Re(const Vector &x, Vector &dp)
{

   double alpha = omega/sqrt(3);
   double p, d2p;
   get_helmholtz_solution_Re(x, p, dp, d2p);
}
void grad_exact_Im(const Vector &x, Vector &dp)
{
   double alpha = omega/sqrt(3);
   double p, d2p;
   get_helmholtz_solution_Im(x, p, dp, d2p);
}

//define impedence coefficient: i omega p
double g_exact_Re(const Vector &x)
{
   double p, d2p;
   double dp[3];
   get_helmholtz_solution_Im(x, p, dp, d2p);

   return -omega * p;
}
double g_exact_Im(const Vector &x)
{
   double p, d2p;
   double dp[3];
   get_helmholtz_solution_Re(x, p, dp, d2p);

   return omega * p;
}