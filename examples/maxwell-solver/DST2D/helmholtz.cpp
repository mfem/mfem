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
#include "DST2D.hpp"
#include "AdditiveST2D.hpp"

using namespace std;
using namespace mfem;

// Exact solution and r.h.s., see below for implementation.
double f_exact_Re(const Vector &x);
double f_exact_Im(const Vector &x);

double wavespeed(const Vector &x);


int dim;
double omega;
int sol = 1;
bool pml = false;
double length = 1.0;
double pml_length = 0.25;
bool scatter = false;
Array2D<double>comp_bdr;

#ifndef MFEM_USE_SUPERLU
#error This example requires that MFEM is built with MFEM_USE_PETSC=YES
#endif

int main(int argc, char *argv[])
{

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
   args.AddOption(&pml, "-pml", "--pml", "-no-pml",
                  "--no-pml", "Enable PML.");
   args.AddOption(&pml_length, "-pml_length", "--pml_length",
                  "Length of the PML region in each direction");
   args.AddOption(&length, "-length", "--length",
                  "length of the domainin in each direction.");
   args.AddOption(&ref, "-ref", "--ref",
                  "Number of Refinements.");
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
		args.PrintUsage(cout);
		return 1;
   }
	args.PrintOptions(cout);
   // Angular frequency
   omega = 2.0 * M_PI * k;

   // 3. Read the mesh from the given mesh file.
   Mesh *mesh;

   if (nd == 2)
   {
      // mesh = new Mesh(mesh_file,1,1);
      mesh = new Mesh(4, 4, Element::QUADRILATERAL, true, length, length, false);
   }
   else
   {
      mesh = new Mesh(1, 1, 1, Element::HEXAHEDRON, true, length, length, length,false);
   }

   // 3. Executing uniform h-refinement
   for (int i = 0; i < ref; i++ )
   {
      mesh->UniformRefinement();
   }
   dim = mesh->Dimension();

   double hl = GetUniformMeshElementSize(mesh);
   Vector pmin, pmax;
   mesh->GetBoundingBox(pmin,pmax);
   // double domain_length = pmax[0] - pmin[0];
   // double pml_thickness = 0.125/domain_length;
   // int nrlayers = pml_thickness/hl;
   int nrlayers = 4;
   Array<int> directions;
   
   for (int i = 0; i<nrlayers; i++)
   {
      for (int comp=0; comp<dim; ++comp)
      {
         directions.Append(comp+1);
         directions.Append(-comp-1);
      }
   }
   // Find uniform h size of the original mesh
   cout << "pml layers = " << nrlayers << endl;
   cout << "pml length = " << hl*nrlayers << endl;
   Mesh *mesh_ext = ExtendMesh(mesh,directions);


   Array2D<double> lengths(dim,2);
   lengths = hl*nrlayers;
   // lengths[0][1] = 0.0;
   // lengths[1][1] = 0.0;
   // lengths[1][0] = 0.0;
   // lengths[0][0] = 0.0;
   CartesianPML pml(mesh_ext,lengths);
   pml.SetOmega(omega);
   comp_bdr.SetSize(dim,2);
   comp_bdr = pml.GetCompDomainBdr(); 

   // 6. Define a finite element space on the mesh.
   FiniteElementCollection *fec = new H1_FECollection(order, dim);
   FiniteElementSpace *fespace = new FiniteElementSpace(mesh_ext, fec);

   // 6. Set up the linear form (Real and Imaginary part)
   FunctionCoefficient f_Re(f_exact_Re);
   FunctionCoefficient f_Im(f_exact_Im);

   // ParLinearForm *b_Re(new ParLinearForm);
   ComplexLinearForm b(fespace, ComplexOperator::HERMITIAN);
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


   SesquilinearForm a(fespace,ComplexOperator::HERMITIAN);

   a.AddDomainIntegrator(new DiffusionIntegrator(c1_re),
                         new DiffusionIntegrator(c1_im));
   a.AddDomainIntegrator(new MassIntegrator(c2_re),new MassIntegrator(c2_im));

   a.Assemble();
   a.Finalize();

   Array<int> ess_tdof_list;
   Array<int> ess_bdr(mesh_ext->bdr_attributes.Max());
   ess_bdr = 1;
   fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

   // Solution grid function
   ComplexGridFunction p_gf(fespace);
   OperatorHandle Ah;
   Vector X, B;

   a.FormLinearSystem(ess_tdof_list, p_gf, b, Ah, X, B);

   ComplexSparseMatrix * AZ = Ah.As<ComplexSparseMatrix>();
   SparseMatrix * A = AZ->GetSystemMatrix();


   cout << "Size of fine grid system: "
         << A->Height() << " x " << A->Width() << endl;


   DST2D S1(&a,lengths, omega, &ws, nrlayers);
   // AdditiveST S2(&a,lengths, omega, &ws, nrlayers);
   

   StopWatch chrono;

   // chrono.Clear();
   // chrono.Start();
   X = 0.0;
	GMRESSolver gmres;
	// gmres.iterative_mode = true;
   gmres.SetPreconditioner(S1);
	gmres.SetOperator(*A);
	gmres.SetRelTol(1e-10);
	gmres.SetMaxIter(50);
	gmres.SetPrintLevel(1);
	gmres.Mult(B, X);

   // X = 0.0;
	// gmres.SetPreconditioner(S2);
	// gmres.Mult(B, X);

   // chrono.Stop();
   // cout << "GMRES time: " << chrono.RealTime() << endl; 

   X = 0.0;
   SLISolver sli;
   sli.iterative_mode = true;
   sli.SetPreconditioner(S1);
   sli.SetOperator(*A);
   sli.SetRelTol(1e-10);
   sli.SetMaxIter(50);
   sli.SetPrintLevel(1);
   sli.Mult(B,X);

   // int n= 200;
   // X = 0.0;
   // Vector z(X.Size()); z = 0.0;
   // Vector r(B);
   // Vector ztemp(r.Size());
   // Vector Ax(X.Size());
   // double tol = 1e-10;
   // cout << endl;
   // chrono.Clear();
   // chrono.Start();
   // for (int i = 0; i<n; i++)
   // {
   //    A->Mult(X,Ax); Ax *=-1.0;
   //    r = b; r+=Ax;
   //    cout << "   ST Solver   Iteration :   " << i <<"  || r || = " <<  r.Norml2() << endl;
   //    if (r.Norml2() < tol) 
   //    {
   //       cout << "Convergence in " << i << " iterations" << endl;
   //       break;
   //    }
   //    S1.Mult(r,z); 
   //    X += z;

   //    // X1-=z;
   //    // p_gf = 0.0;
   //    // a.RecoverFEMSolution(X,B,p_gf);
   //    //    char vishost[] = "localhost";
   //    //    int  visport   = 19916;
   //    //    string keys;
   //    //    if (dim ==2 )
   //    //    {
   //    //       keys = "keys mrRljc\n";
   //    //    }
   //    //    else
   //    //    {
   //    //       keys = "keys mc\n";
   //    //    }
   //    //    socketstream sol1_sock_re(vishost, visport);
   //    //    sol1_sock_re.precision(8);
   //    //    sol1_sock_re << "solution\n" << *mesh_ext << p_gf.real() <<
   //    //                "window_title 'Numerical Pressure (real part)' "
   //    //                << keys << flush;
   //    //    cin.get();
   // }

   // chrono.Stop();
   // cout << "Solver time: " << chrono.RealTime() << endl; 

   a.RecoverFEMSolution(X,B,p_gf);

   KLUSolver klu(*A);
   Vector X1(X.Size());
   klu.Mult(B,X1);
   X1-= X;

   ComplexGridFunction error_gf(fespace);

   a.RecoverFEMSolution(X1,B,error_gf);


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
      sol_sock_re << "solution\n" << *mesh_ext << p_gf.real() <<
                  "window_title 'Numerical Pressure (real part from DST)' "
                  // << keys << flush;
                  << keys << "valuerange -0.08 0.08 \n" << flush;
      socketstream err_sock_re(vishost, visport);
      err_sock_re.precision(8);
      err_sock_re << "solution\n" << *mesh_ext << error_gf.real() <<
                  "window_title 'Numerical Pressure (real part from KLU)' "
                  << keys << flush;
   }
   delete fespace;
   delete fec;
	delete mesh_ext;
	delete mesh;
   return 0;
}


//calculate RHS from exact solution f = - \Delta u
double f_exact_Re(const Vector &x)
{
   double f_re = 0.0;
   double x0 = length/2.0;
   double x1 = length/2.0;
   double x2 = length/2.0;
   // x0 = 0.59;
   // x0 = 0.19;
   x0 = 0.5;
   // x1 = 0.768;
   // x1 = 0.168;
   x1 = 0.5;
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

   x0 = 0.85;
   x1 = 0.15;
   beta = pow(x0-x(0),2) + pow(x1-x(1),2);
   if (dim == 3) { beta += pow(x2-x(2),2); }
   alpha = -pow(n,2) * beta;
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
   // if (x(0) <= 0.25)
   // {
   //    ws = 1.0;
   // }
   // else if(x(0)<=0.5)
   // {
   //    ws = 1.0;
   // }
   // else if(x(0)<=0.75)
   // {
   //    ws = 0.75;
   //    // ws = 0.5;
   // }
   // else 
   // {
   //    ws = 0.75;
   //    // ws = 1.0;
   // }
   // if (x(1) <= 1.0/3.0)
   // {
   //    ws = 2.0;
   // }
   // else if(x(1)<=2.0/3.0)
   // {
   //    ws = 1.0;
   // }
   // else 
   // {
   //    // ws = 0.75;
   //    ws = 0.25;
   // }

   // if (x(0) <= 0.33)
   // {
   //    ws = 1.0;
   // }
   // else if(x(0)<=0.66)
   // {
   //    ws = -0.65 + 5.0*x(0);
   // }
   // else 
   // {
   //    ws = 2.65;
   //    // ws = 0.5;
   // }

   // if (x(0) <= x(1) && x(1) >= 1.0-x(0))
   // {
   //    ws = 1.0;
   // }
   // else if (x(0) > x(1) && x(1) >= 1.0-x(0))
   // {
   //    ws = 3.0;
   // }
   // else if (x(0) <= x(1) && x(1) < 1.0-x(0))
   // {
   //    ws = 2.0;
   // }
   // else
   // {
   //    ws = 4.0;
   // }
   


   ws = 1.0;
   return ws;
}


























