//
// Compile with: make helmholtz
//
// Sample runs:  helmholtz -m ../data/one-hex.mesh
//               helmholtz -m ../data/fichera.mesh
//               helmholtz -m ../data/fichera-mixed.mesh
//
// Description:  This example code demonstrates the use of MFEM to define a
//               simple finite element discretization of the Helmholtz problem
//               -Delta p - omega^2 p = f with impedance boundary conditiones.
//
#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include "mg.hpp"

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
int sol = 1;
double length = 1.0;

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
   bool visualization = 1;
   // number of wavelengths
   double k = 0.5;
   // number of mg levels
   int href = 1;
   // number of initial ref
   int initref = 1;
   // dimension
   int nd = 2;


   // optional command line inputs
   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&nd, "-nd", "--dim","Problem space dimension");               
   args.AddOption(&sol, "-sol", "--exact",
                  "Exact solution flag - 0:polynomial, 1: plane wave, -1: unknown exact");               
   args.AddOption(&k, "-k", "--wavelengths",
                  "Number of wavelengths.");
   args.AddOption(&length, "-length", "--length",
                  "length of the domainin in each direction.");                                       
   args.AddOption(&href, "-href", "--href",
                  "Number of Geometric Refinements.");
   args.AddOption(&initref, "-initref", "--initref",
                  "Number of initial refinements.");
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

   //-----------------------------------------------------------------------------

   // if (scatter) pml = true; // for now only scattering problems with pml

   // 3. Read the mesh from the given mesh file.
   Mesh mesh;
   
   if (nd == 2) 
   {
      mesh = Mesh::MakeCartesian2D(1, 1, Element::QUADRILATERAL, true, length, length, false);

   }
   else
   {
      mesh = Mesh::MakeCartesian3D(1, 1, 1, Element::HEXAHEDRON, true, length, length, length);
   }

   dim = mesh.Dimension();
   int sdim = mesh.SpaceDimension();

   // 3. Executing uniform h-refinement
   for (int i = 0; i < initref; i++ )
   {
      mesh.UniformRefinement();
   }

   // 5. Define a parallel mesh and delete the serial mesh.
   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear(); // the serial mesh is no longer needed
   pmesh.UniformRefinement();

   // ----------------------------------------------------------------------------

   // 6. Define a finite element space on the mesh.
   H1_FECollection fec(order, pmesh.Dimension());
   ParFiniteElementSpace fespace(&pmesh, &fec);
   std::vector<ParFiniteElementSpace * > fespaces(href+1);
   std::vector<ParMesh * > ParMeshes(href+1);
   std::vector<HypreParMatrix*>  P(href);
   for (int i = 0; i < href; i++)
   {
      ParMeshes[i] = new ParMesh(pmesh);
      fespaces[i]  = new ParFiniteElementSpace(fespace, *ParMeshes[i]);
      pmesh.UniformRefinement();
      // Update fespace
      fespace.Update();
      OperatorHandle Tr(Operator::Hypre_ParCSR);
      fespace.GetTrueTransferOperator(*fespaces[i], Tr);
      Tr.SetOperatorOwner(false);
      Tr.Get(P[i]);
   }
   fespaces[href] = new ParFiniteElementSpace(fespace);

   // 6. Set up the linear form (Real and Imaginary part)
   FunctionCoefficient f_Re(f_exact_Re);
   FunctionCoefficient g_Re(g_exact_Re);
   VectorFunctionCoefficient grad_Re(sdim, grad_exact_Re);
   FunctionCoefficient f_Im(f_exact_Im);
   FunctionCoefficient g_Im(g_exact_Im);
   VectorFunctionCoefficient grad_Im(sdim, grad_exact_Im);

   // ParLinearForm *b_Re(new ParLinearForm);
   ParComplexLinearForm b(&fespace, ComplexOperator::HERMITIAN);
   b.AddDomainIntegrator(new DomainLFIntegrator(f_Re),new DomainLFIntegrator(f_Im));
   if(sol >=0) // if exact solution exists. Otherwise use homogeneous impedence (gradp . n + i omega p = 0)
   {
      b.AddBoundaryIntegrator(new BoundaryNormalLFIntegrator(grad_Re),
                              new BoundaryNormalLFIntegrator(grad_Im));
      b.AddBoundaryIntegrator(new BoundaryLFIntegrator(g_Re),
                              new BoundaryLFIntegrator(g_Im));
   }
   b.real().Vector::operator=(0.0);
   b.imag().Vector::operator=(0.0);                        
   b.Assemble();

   // 7. Set up the bilinear form (Real and Imaginary part)
   ConstantCoefficient one(1.0);
   ConstantCoefficient zero(0.0);
   ConstantCoefficient neg_omega(-pow(omega, 2));

   ParSesquilinearForm a(&fespace,ComplexOperator::HERMITIAN);
   ConstantCoefficient impedance(omega);

   Array<int> bdr_attr(pmesh.bdr_attributes.Max());
   bdr_attr = 1;
   RestrictedCoefficient imp_rest(impedance,bdr_attr);

   a.AddDomainIntegrator(new DiffusionIntegrator(one),NULL);
   // Just putting imaginary integrator with zero just to keep sparsity pattern of real 
   // and imaginary part the same 
   a.AddDomainIntegrator(new MassIntegrator(neg_omega),new MassIntegrator(zero));
   a.AddBoundaryIntegrator(new BoundaryMassIntegrator(zero),new BoundaryMassIntegrator(imp_rest));
   a.Assemble(0);

   Array<int> ess_tdof_list;
   Array<int> ess_bdr(pmesh.bdr_attributes.Max());
   ess_bdr = 0;
   fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

   // Solution grid function
   ParComplexGridFunction p_gf(&fespace);
   ParComplexGridFunction p_gf_ex(&fespace);
   FunctionCoefficient p_Re(p_exact_Re);
   FunctionCoefficient p_Im(p_exact_Im);
   p_gf = 0.0;
   p_gf_ex.ProjectCoefficient(p_Re,p_Im);
   if (sol >= 0) 
   {
      p_gf.ProjectBdrCoefficient(p_Re,p_Im,ess_bdr);
   }

   OperatorHandle Ah;
   Vector X, B;
   a.FormLinearSystem(ess_tdof_list, p_gf, b, Ah, X, B);

   ComplexHypreParMatrix * AZ = Ah.As<ComplexHypreParMatrix>();


   ComplexMGSolver * M = new ComplexMGSolver(AZ,P,fespaces);
   M->SetTheta(0.25);

   ComplexSchwarzSmoother * M = new ComplexSchwarzSmoother(
      fespace.GetParMesh(),0,&fespace,AZ);


   GMRESSolver gmres(MPI_COMM_WORLD);
   gmres.SetPrintLevel(1);
   gmres.SetMaxIter(2000);
   gmres.SetKDim(200);
   gmres.SetRelTol(1e-12);
   gmres.SetAbsTol(0.0);
   gmres.SetOperator(*AZ);
   gmres.SetPreconditioner(*M);
   gmres.Mult(B, X);


   {
      ComplexMUMPSSolver cmumps;
      cmumps.SetPrintLevel(0);
      cmumps.SetOperator(*AZ);
      cmumps.Mult(B,X);
   }


   a.RecoverFEMSolution(X,B,p_gf);


   if (sol >= 0)
   {
      const int h1_norm_type = 1;
      double L2error;
      double H1error;

      double L2err_Re  = p_gf.real().ComputeL2Error(p_Re);
      double L2err_Im  = p_gf.imag().ComputeL2Error(p_Im);

      double loc_H1err_Re = p_gf.real().ComputeH1Error(&p_Re, &grad_Re, &one, 1.0, h1_norm_type);
      double loc_H1err_Im = p_gf.imag().ComputeH1Error(&p_Im, &grad_Im, &one, 1.0, h1_norm_type);
      double H1err_Re = GlobalLpNorm(2.0, loc_H1err_Re, MPI_COMM_WORLD);
      double H1err_Im = GlobalLpNorm(2.0, loc_H1err_Im, MPI_COMM_WORLD);
         
      L2error = sqrt(L2err_Re*L2err_Re + L2err_Im*L2err_Im);
      H1error = sqrt(H1err_Re*H1err_Re + H1err_Im*H1err_Im);
      if (myid == 0)
      {
         cout << " || p_h - p ||_{H^1} = " << H1error <<  endl;
         cout << " || p_h - p ||_{L^2} = " << L2error <<  endl;
      }
   }

   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      string keys;
      if(dim ==2 )
      {
         keys = "keys mrRljc\n";
      }
      else
      {
         keys = "keys mc\n";
      }
      socketstream sol_sock_re(vishost, visport);
      sol_sock_re << "parallel " << num_procs << " " << myid << "\n";
      sol_sock_re.precision(8);
      sol_sock_re << "solution\n" << pmesh << p_gf.real() << "window_title 'Numerical Pressure (real part)' " 
               << keys << flush;
      socketstream sol_sock_im(vishost, visport);
      sol_sock_im << "parallel " << num_procs << " " << myid << "\n";
      sol_sock_im.precision(8);
      sol_sock_im << "solution\n" << pmesh << p_gf.imag() << "window_title 'Numerical Pressure (imag part)' " 
               << keys << flush;         
   }
   MPI_Finalize();
   return 0;
}

//define exact solutions
void get_helmholtz_solution_Re(const Vector &x, double & p, double dp[], double & d2p)
{

   if (sol == 0) // polynomial
   {  
      if (dim == 3)
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
         p = x[1] * (1.0 - x[1])* x[0] * (1.0 - x[0]);
         dp[0] = (1.0 - 2.0 *x[0]) * x[1]*(1.0 - x[1]);
         dp[1] = (1.0 - 2.0 *x[1]) * x[0]*(1.0 - x[0]);
         d2p = - 2.0 * x[1] * (1.0 - x[1]) 
               - 2.0 * x[0] * (1.0 - x[0]);   
      }
   }

   else if(sol == 1) // Plane wave
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
   else if (sol == 2)
   {
      if (dim == 2 )
      {
         // shift to avoid singularity
         double shift = 0.1;
         double x0 = x(0) + shift;
         double x1 = x(1) + shift;
         //
         double r = sqrt(x0 * x0 + x1 * x1);

         p = cos(omega * r);
         double r_x = x0 / r;
         double r_y = x1 / r;
         double r_xx = (1.0 / r) * (1.0 - r_x * r_x);
         double r_yy = (1.0 / r) * (1.0 - r_y * r_y);

         dp[0] = - omega * sin(omega * r) * r_x; 
         dp[1] = - omega * sin(omega * r) * r_y; 

         d2p = -omega*omega * cos(omega * r)*r_x * r_x - omega * sin(omega*r) * r_xx
               -omega*omega * cos(omega * r)*r_y * r_y - omega * sin(omega*r) * r_yy;
      }
      else
      {
         // shift to avoid singularity
         double shift = 0.1;
         double x0 = x(0) + shift;
         double x1 = x(1) + shift;
         double x2 = x(2) + shift;
         //
         double r = sqrt(x0 * x0 + x1 * x1 + x2 * x2);

         p = cos(omega * r);

         double r_x = x0 / r;
         double r_y = x1 / r;
         double r_z = x2 / r;
         double r_xx = (1.0 / r) * (1.0 - r_x * r_x);
         double r_yy = (1.0 / r) * (1.0 - r_y * r_y);
         double r_zz = (1.0 / r) * (1.0 - r_z * r_z);

         dp[0] = - omega * sin(omega * r) * r_x; 
         dp[1] = - omega * sin(omega * r) * r_y; 
         dp[2] = - omega * sin(omega * r) * r_z; 

         d2p = -omega*omega * cos(omega * r)*r_x * r_x - omega * sin(omega*r) * r_xx
               -omega*omega * cos(omega * r)*r_y * r_y - omega * sin(omega*r) * r_yy
               -omega*omega * cos(omega * r)*r_z * r_z - omega * sin(omega*r) * r_zz;
      }
   }
}

void get_helmholtz_solution_Im(const Vector &x, double & p, double dp[], double & d2p)
{
   if (sol == 0) // polynomial
   {  
      if (dim == 3)
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
         p = x[1] * (1.0 - x[1])* x[0] * (1.0 - x[0]);
         dp[0] = (1.0 - 2.0 *x[0]) * x[1]*(1.0 - x[1]);
         dp[1] = (1.0 - 2.0 *x[1]) * x[0]*(1.0 - x[0]);
         d2p = - 2.0 * x[1] * (1.0 - x[1]) 
               - 2.0 * x[0] * (1.0 - x[0]);   
      }
   }
   else if (sol == 1)// plane wave
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
   double p_re, d2p_re;
   double dp_re[3];
   double f_re;
   f_re = 0.0;
   get_helmholtz_solution_Re(x, p_re, dp_re, d2p_re);
   f_re = -d2p_re - omega * omega * p_re;
   return f_re;
}
double f_exact_Im(const Vector &x)
{
   double p_im, d2p_im;
   double dp_im[3];
   double f_im;
   f_im = 0.0;
   get_helmholtz_solution_Im(x, p_im, dp_im, d2p_im);
   f_im = -d2p_im - omega * omega * p_im;
   return f_im;   
}

void grad_exact_Re(const Vector &x, Vector &dp)
{
   double p, d2p;
   get_helmholtz_solution_Re(x, p, dp, d2p);
}
void grad_exact_Im(const Vector &x, Vector &dp)
{
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
