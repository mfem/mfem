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
#include "complex_additive_schwarzp.hpp"

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

// pml
void pml_function(const Vector &x, std::vector<std::complex<double>> & dxs);
double pml_detJ_Re(const Vector &x);
double pml_detJ_Im(const Vector &x);
void pml_detJ_JT_J_inv_Re(const Vector &x, DenseMatrix &M);
void pml_detJ_JT_J_inv_Im(const Vector &x, DenseMatrix &M);


int dim;
double omega;
int sol = 1;
bool pml = false;
double length = 1.0;
double pml_length = 0.25;
bool scatter = false;

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
   int ref = 1;
   // number of initial ref
   int initref = 1;
   // dimension
   int nd = 2;


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
   args.AddOption(&nd, "-nd", "--dim","Problem space dimension");               
   args.AddOption(&sol, "-sol", "--exact",
                  "Exact solution flag - 0:polynomial, 1: plane wave, -1: unknown exact");               
   args.AddOption(&k, "-k", "--wavelengths",
                  "Number of wavelengths.");
   args.AddOption(&pml, "-pml", "--pml", "-no-pml",
                  "--no-pml", "Enable PML.");         
   args.AddOption(&pml_length, "-pml_length", "--pml_length",
                  "Length of the PML region in each direction");        
   args.AddOption(&length, "-length", "--length",
                  "length of the domainin in each direction.");                                       
   args.AddOption(&ref, "-ref", "--ref",
                  "Number of Refinements.");
   args.AddOption(&initref, "-initref", "--initref",
                  "Number of initial refinements.");
   args.AddOption(&petscrc_file, "-petscopts", "--petscopts",
                  "PetscOptions file to use.");
   args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                  "--no-static-condensation", "Enable static condensation.");
   args.AddOption(&scatter, "-scat", "--scattering-prob", "-no-scat",
                  "--no-scattering", "Solve a scattering problem");               
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
   // omega = k;

   // 2b. Initialize PETSc
   MFEMInitializePetsc(NULL, NULL, petscrc_file, NULL);
   //-----------------------------------------------------------------------------

   // if (scatter) pml = true; // for now only scattering problems with pml

   // 3. Read the mesh from the given mesh file.
   Mesh *mesh;
   
   if (nd == 2) 
   {
      if (scatter) 
      {
         mesh_file = "../../data/rectwhole7_2attr.e";
         mesh = new Mesh(mesh_file, 1, 1);
      }
      else
      {
         mesh = new Mesh(1, 1, Element::QUADRILATERAL, true, length, length, false);
      }
   }
   else
   {
      if (scatter)
      {
         // mesh_file = "../../data/hexwhole7.e";
         // mesh_file = "../../data/hexwhole.e";
         // mesh_file = "../../data/hexa728.mesh";
         // mesh_file = "./hexa728.mesh";
         mesh_file = "../../data/hexwhole7.e";
         mesh = new Mesh(mesh_file, 1, 1);
      }
      else
      {
         mesh = new Mesh(1, 1, 1, Element::HEXAHEDRON, true, length, length, length, false);
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

   dim = mesh->Dimension();
   int sdim = mesh->SpaceDimension();

   // 3. Executing uniform h-refinement
   for (int i = 0; i < initref; i++ )
   {
      mesh->UniformRefinement();
   }

   // 5. Define a parallel mesh and delete the serial mesh.
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   // delete mesh;

   // ----------------------------------------------------------------------------

   // 6. Define a finite element space on the mesh.
   FiniteElementCollection *fec = new H1_FECollection(order, dim);
   ParFiniteElementSpace *fespace = new ParFiniteElementSpace(pmesh, fec);
   std::vector<HypreParMatrix*>  P(ref);

   for (int i = 0; i < ref; i++)
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
   if (!scatter) // if scattering problem the source is zero and is driven by bc
   {
      b.AddDomainIntegrator(new DomainLFIntegrator(f_Re),new DomainLFIntegrator(f_Im));
      if (!pml && sol >=0) // if exact solution exists. Otherwise use homogeneous impedence (gradp . n + i omega p = 0)
      {
         b.AddBoundaryIntegrator(new BoundaryNormalLFIntegrator(grad_Re),
                                 new BoundaryNormalLFIntegrator(grad_Im));
         b.AddBoundaryIntegrator(new BoundaryLFIntegrator(g_Re),
                                 new BoundaryLFIntegrator(g_Im));
      }
   }
   b.real().Vector::operator=(0.0);
   b.imag().Vector::operator=(0.0);                        
   b.Assemble();

   // 7. Set up the bilinear form (Real and Imaginary part)
   ConstantCoefficient one(1.0);
   ConstantCoefficient sigma(-pow(omega, 2));

   ParSesquilinearForm * a = new ParSesquilinearForm(fespace,ComplexOperator::HERMITIAN);
   ConstantCoefficient impedance(omega);


   MatrixFunctionCoefficient c1_Re(dim,pml_detJ_JT_J_inv_Re);
   MatrixFunctionCoefficient c1_Im(dim,pml_detJ_JT_J_inv_Im);

   FunctionCoefficient det_Re(pml_detJ_Re);
   FunctionCoefficient det_Im(pml_detJ_Im);
   ProductCoefficient c2_Re(det_Re,sigma);
   ProductCoefficient c2_Im(det_Im,sigma);

   Array<int> bdr_attr(pmesh->bdr_attributes.Max());
   bdr_attr = 0;
   bdr_attr[0] = 1;
   if (!scatter) bdr_attr = 1;
   RestrictedCoefficient imp_rest(impedance,bdr_attr);

   a->AddDomainIntegrator(new DiffusionIntegrator(c1_Re),new DiffusionIntegrator(c1_Im));
   a->AddDomainIntegrator(new MassIntegrator(c2_Re),new MassIntegrator(c2_Im));
   if (!pml)
   {
      a->AddBoundaryIntegrator(NULL,new BoundaryMassIntegrator(imp_rest));
      // a.AddBoundaryIntegrator(NULL,new BoundaryMassIntegrator(impedance));
   }
   a->Assemble();
   a->Finalize();

   Array<int> ess_tdof_list;
   Array<int> ess_bdr(pmesh->bdr_attributes.Max());
   ess_bdr = 0;
   if (scatter)
   {
      if (pml)
      {
         ess_bdr = 1;
      }
      else
      {
         ess_bdr[1] = 1;
      }
   }
   fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

   // Solution grid function
   ParComplexGridFunction p_gf(fespace);
   ParComplexGridFunction p_gf_ex(fespace);
   FunctionCoefficient p_Re(p_exact_Re);
   FunctionCoefficient p_Im(p_exact_Im);
   p_gf = 0.0;
   p_gf_ex.ProjectCoefficient(p_Re,p_Im);
   if (!pml && sol >= 0 ) 
   {
      p_gf.ProjectBdrCoefficient(p_Re,p_Im,ess_bdr);
   }
   if (scatter) 
   {
      p_gf.ProjectBdrCoefficient(p_Re,p_Im,ess_bdr);
   }

   OperatorHandle Ah;
   Vector X, B;

   a->FormLinearSystem(ess_tdof_list, p_gf, b, Ah, X, B);

   ComplexHypreParMatrix * AZ = Ah.As<ComplexHypreParMatrix>();
   HypreParMatrix * A = AZ->GetSystemMatrix();


   if (myid == 0)
   {
      cout << "Size of fine grid system: "
           << A->GetGlobalNumRows() << " x " << A->GetGlobalNumCols() << endl;
   }

   PetscLinearSolver * petsc = new PetscLinearSolver(MPI_COMM_WORLD, "direct");
   // Convert to PetscParMatrix
   petsc->SetOperator(PetscParMatrix(A, Operator::PETSC_MATAIJ));
   petsc->Mult(B,X);

   a->RecoverFEMSolution(X,B,p_gf);

   

   ComplexParAddSchwarz * test = new ComplexParAddSchwarz(a);

   delete test;



   if (!scatter && sol >= 0 )
   {
      // int order_quad = max(2, 2*order+1);
      // const IntegrationRule *irs[Geometry::NumGeom];
      // for (int i=0; i < Geometry::NumGeom; ++i)
      // {
         // irs[i] = &(IntRules.Get(i, order_quad));
      // }

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
      sol_sock_re << "solution\n" << *pmesh << p_gf.real() << "window_title 'Numerical Pressure (real part)' " 
               << keys << flush;
   }

   delete a;
   delete fespace;
   delete fec;
   delete pmesh;
   MFEMFinalizePetsc();
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
         if (scatter) shift = -0.5;
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
         if (scatter) shift = -0.5;
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
   if (scatter)
   {
      if (dim == 2)
      {
         if (abs(x(0)-1.) < 1e-13 || abs(x(1)-1.0) < 1e-13 ||
             abs(x(0)) < 1e-13    || abs(x(1)) < 1e-13)
         {
            p = 0.0;
         }
      }
      else
      {
         if (abs(x(0)-1.) < 1e-13 || abs(x(1)-1.0) < 1e-13 || abs(x(2)-1.0) < 1e-13 ||
             abs(x(0)) < 1e-13    || abs(x(1)) < 1e-13 || abs(x(2)) < 1e-13)
         {
            p = 0.0;
         }
      }
   }
   return p;
}
double p_exact_Im(const Vector &x)
{
   double p, d2p;
   double dp[3];
   get_helmholtz_solution_Im(x, p, dp, d2p);
   if (scatter)
   {
      if (dim == 2)
      {
         if (abs(x(0)-1.) < 1e-13 || abs(x(1)-1.0) < 1e-13 ||
             abs(x(0)) < 1e-13    || abs(x(1)) < 1e-13)
         {
            p = 0.0;
         }
      }
      else
      {
         if (abs(x(0)-1.) < 1e-13 || abs(x(1)-1.0) < 1e-13 || abs(x(2)-1.0) < 1e-13 ||
             abs(x(0)) < 1e-13    || abs(x(1)) < 1e-13 || abs(x(2)) < 1e-13)
         {
            p = 0.0;
         }
      }
   }
   // p *= -1.0;
   return p;
}

//calculate RHS from exact solution f = - \Delta u
double f_exact_Re(const Vector &x)
{
   double p_re, d2p_re, p_im, d2p_im;
   double dp_re[3], dp_im[3];
   double f_re;
   f_re = 0.0;
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
      f_re = coeff*exp(alpha);
   }
   else
   {
      get_helmholtz_solution_Re(x, p_re, dp_re, d2p_re);
      get_helmholtz_solution_Im(x, p_im, dp_im, d2p_im);
      f_re = -d2p_re - omega * omega * p_re;
   }
   return f_re;
}
double f_exact_Im(const Vector &x)
{
   double p_re, d2p_re, p_im, d2p_im;
   double dp_re[3], dp_im[3];
   double f_im;
   f_im = 0.0;
   if (sol < 0)
   {
      // double x0 = 0.6;
      // double x1 = 0.6;
      // double alpha;
      // double n = 5.0 * omega/M_PI;
      // double coeff = pow(n,2)/M_PI;
      // alpha = -pow(n,2) * sqrt(pow(x0-x(0),2) + pow(x1-x(1),2));
      // f_im = coeff*exp(alpha);
   }
   else
   {
      get_helmholtz_solution_Re(x, p_re, dp_re, d2p_re);
      get_helmholtz_solution_Im(x, p_im, dp_im, d2p_im);
      f_im = -d2p_im - omega * omega * p_im;
   }
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







void pml_function(const Vector &x, std::vector<std::complex<double>> & dxs)
{
   double L = length;
   double n = 2.0;
   double lbeg, lend;
   double c = 50.0;
   double c1 = pml_length; 
   double c2 = length-pml_length; 
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
            dxs[i] = complex<double>(1.0,0.0) - complex<double>(0.0,coeff * pow(x(i)-lbeg, n-1.0)); 
         }
         if (x(i) <= c1)
         {
            lbeg = c1;
            lend = 0.0;
            coeff = n * c / omega / pow(lend-lbeg,n); 
            dxs[i] = complex<double>(1.0,0.0) + complex<double>(0.0, coeff * pow(x(i)-lbeg, n-1.0)); 
         } 
      }
   }
}

double pml_detJ_Re(const Vector &x)
{
   std::vector<std::complex<double>> dxs(dim);
   complex<double> det(1.0,0.0);
   pml_function(x, dxs);
   for (int i=0; i<dim; ++i) det *= dxs[i];
   return det.real();
}

double pml_detJ_Im(const Vector &x)
{
   std::vector<std::complex<double>> dxs(dim);
   complex<double> det(1.0,0.0);
   pml_function(x, dxs);
   for (int i=0; i<dim; ++i) det *= dxs[i];
   return det.imag();
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




























// int ndofs = nodes->FESpace()->GetNDofs();
//    Vector xcoords(ndofs), ycoords(ndofs), zcoords(ndofs);

//    for (int comp = 0; comp < nodes->FESpace()->GetVDim(); comp++)
//    {
//       for (int i = 0; i < ndofs; i++)
//       {
//          if (comp == 0)
//          {
//             xcoords(i) = *nodes[nodes->FESpace()->DofToVDof(i, comp)];
//          }
//          else if (comp == 1)
//          {
//             ycoords(i) = *nodes[nodes->FESpace()->DofToVDof(i, comp)];
//          }
//          else if (comp == 2)
//          {
//             zcoords(i) = *nodes[nodes->FESpace()->DofToVDof(i, comp)];
//          }
//       }
//    }
