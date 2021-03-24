//
// Compile with: make maxwell
//
//               maxwell -o 2 -f 8.0 -ref 3 -prob 4 -m ../data/inline-quad.mesh
//

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include "DST/DST.hpp"

using namespace std;
using namespace mfem;

void source_re(const Vector &x, Vector & f);
void source_im(const Vector &x, Vector & f);
double wavespeed(const Vector &x);



double mu = 1.0;
double epsilon = 1.0;
double omega;
int dim;
double length = 1.0;
Array2D<double> comp_bdr;

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../../data/inline-quad.mesh";
   int order = 1;
   int ref_levels = 3;
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
   args.AddOption(&ref_levels, "-ref", "--refinements",
                  "Number of refinements");
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


   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);


   Mesh *mesh;

   if (nd == 2)
   {
      mesh = new Mesh(4, 4, Element::QUADRILATERAL, true, length, length, false);
   }
   else
   {
      mesh = new Mesh(1, 1, 1, Element::HEXAHEDRON, true, length, length, length,false);
   }

   dim = mesh->Dimension();

   // Angular frequency
   omega = 2.0 * M_PI * freq;

   // 4. Refine the mesh to increase the resolution.
   for (int l = 0; l < ref_levels; l++)
   {
      mesh->UniformRefinement();
   }

   // char vishost[] = "localhost";
   // int visport = 19916;

   // socketstream mesh_sock(vishost, visport);
   // mesh_sock.precision(8);
   // mesh_sock << "mesh\n"
   //           << *mesh << "window_title 'Global mesh'" << flush;

   // Setup PML length
   int nrlayers = 2;
   double hl = GetUniformMeshElementSize(mesh);
   Array2D<double> lengths(dim, 2); 
   lengths = hl*nrlayers;

   CartesianPML * pml = new CartesianPML(mesh,lengths);
   pml->SetOmega(omega);
   comp_bdr.SetSize(dim,2);
   comp_bdr = pml->GetCompDomainBdr(); 


   // 6. Define a finite element space on the mesh. Here we use the Nedelec
   //    finite elements of the specified order.
   FiniteElementCollection *fec = new ND_FECollection(order, dim);
   FiniteElementSpace *fespace = new FiniteElementSpace(mesh, fec);
   int size = fespace->GetTrueVSize();
   cout << "Number of finite element unknowns: " << size << endl;

   // 7. Determine the list of true essential boundary dofs. In this example,
   //    the boundary conditions are defined based on the specific mesh and the
   //    problem type.
   Array<int> ess_tdof_list;
   if (mesh->bdr_attributes.Size())
   {
      Array<int> ess_bdr(mesh->bdr_attributes.Max());
      ess_bdr = 1;
      fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   // 8. Setup Complex Operator convention
   ComplexOperator::Convention conv =
      herm_conv ? ComplexOperator::HERMITIAN : ComplexOperator::BLOCK_SYMMETRIC;

   // 9. Set up the linear form b(.) which corresponds to the right-hand side of
   //    the FEM linear system.
   VectorFunctionCoefficient f_re(dim, source_re);
   VectorFunctionCoefficient f_im(dim, source_re);
   ComplexLinearForm b(fespace, conv);
   b.AddDomainIntegrator(new VectorFEDomainLFIntegrator(f_re),
                         new VectorFEDomainLFIntegrator(f_im));
   b.Vector::operator=(0.0);
   b.Assemble();

   // 10. Define the solution vector x as a complex finite element grid function
   //     corresponding to fespace.
   ComplexGridFunction x(fespace);
   x = 0.0;
   // 11. Set up the sesquilinear form a(.,.)
   //
   //       1/mu (1/det(J) J^T J Curl E, Curl F)
   //        - omega^2 * epsilon (det(J) * (J^T J)^-1 * E, F)
   //
   FunctionCoefficient ws(wavespeed);
   ConstantCoefficient omeg(-pow(omega, 2));
   int cdim = (dim == 2) ? 1 : dim;
   PmlMatrixCoefficient pml_c1_Re(cdim,detJ_inv_JT_J_Re, pml);
   PmlMatrixCoefficient pml_c1_Im(cdim,detJ_inv_JT_J_Im, pml);

   PmlMatrixCoefficient pml_c2_Re(dim, detJ_JT_J_inv_Re,pml);
   PmlMatrixCoefficient pml_c2_Im(dim, detJ_JT_J_inv_Im,pml);
   ScalarMatrixProductCoefficient c2_Re0(omeg,pml_c2_Re);
   ScalarMatrixProductCoefficient c2_Im0(omeg,pml_c2_Im);
   ScalarMatrixProductCoefficient c2_Re(ws,c2_Re0);
   ScalarMatrixProductCoefficient c2_Im(ws,c2_Im0);

   SesquilinearForm a(fespace, conv);
   a.AddDomainIntegrator(new CurlCurlIntegrator(pml_c1_Re),
                         new CurlCurlIntegrator(pml_c1_Im));
   a.AddDomainIntegrator(new VectorFEMassIntegrator(c2_Re),
                         new VectorFEMassIntegrator(c2_Im));

   a.Assemble(0);

   OperatorHandle Ah;
   Vector B, X;
   a.FormLinearSystem(ess_tdof_list, x, b, Ah, X, B);

   ComplexSparseMatrix * Ac = Ah.As<ComplexSparseMatrix>();
   StopWatch chrono;
   // chrono.Clear();
   // chrono.Start();
   // {
   //    ComplexUMFPackSolver csolver;
   //    csolver.Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
   //    csolver.SetOperator(*Ac);
   //    // csolver.SetPrintLevel(2);
   //    csolver.Mult(B,X);
   // }
   // chrono.Stop();
   // cout << "Time 1 = " << chrono.RealTime() << endl;


   chrono.Clear();
   chrono.Start();
   DST S(&a,lengths, omega, &ws, nrlayers, nx, ny, nz);
   chrono.Stop();
   cout << "Time 2 = " << chrono.RealTime() << endl;
   chrono.Clear();
   chrono.Start();
   X = 0.0;
	GMRESSolver gmres;
	// gmres.iterative_mode = true;
   gmres.SetPreconditioner(S);
	gmres.SetOperator(*Ac);
	gmres.SetRelTol(1e-8);
	gmres.SetMaxIter(50);
	gmres.SetPrintLevel(1);
	gmres.Mult(B, X);
   chrono.Stop();
   cout << "Time 3 = " << chrono.RealTime() << endl;
   // 14. Solve using a direct or an iterative solver
   // Vector Y(X);

   // chrono.Stop();
   // cout << "Time 3 = " << chrono.RealTime() << endl;

   // cout << endl;

   // cout << "X norm = " << X.Norml2() << endl;
   // cout << "Y norm = " << Y.Norml2() << endl;
   // Y-=X;
   // cout << "diff norm = " << Y.Norml2() << endl;



   a.RecoverFEMSolution(X, b, x);

   // 17. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      // Define visualization keys for GLVis (see GLVis documentation)
      string keys;
      keys = (dim == 3) ? "keys acF\n" : keys = "keys amrRljcUUuu\n";

      char vishost[] = "localhost";
      int visport = 19916;

      socketstream sol_sock_re(vishost, visport);
      sol_sock_re.precision(8);
      sol_sock_re << "solution\n"
                  << *mesh << x.real() << keys
                  << "window_title 'Solution real part'" << flush;

      socketstream sol_sock_im(vishost, visport);
      sol_sock_im.precision(8);
      sol_sock_im << "solution\n"
                  << *mesh << x.imag() << keys
                  << "window_title 'Solution imag part'" << flush;

      GridFunction x_t(fespace);
      x_t = x.real();
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "solution\n"
               << *mesh << x_t << keys << "autoscale off\n"
               << "window_title 'Harmonic Solution (t = 0.0 T)'"
               << "pause\n" << flush;
      cout << "GLVis visualization paused."
           << " Press space (in the GLVis window) to resume it.\n";
      int num_frames = 32;
      int i = 0;
      while (sol_sock)
      {
         double t = (double)(i % num_frames) / num_frames;
         ostringstream oss;
         oss << "Harmonic Solution (t = " << t << " T)";

         add(cos(2.0 * M_PI * t), x.real(),
             sin(2.0 * M_PI * t), x.imag(), x_t);
         sol_sock << "solution\n"
                  << *mesh << x_t
                  << "window_title '" << oss.str() << "'" << flush;
         i++;
      }
   }

   // 18. Free the used memory.
   delete pml;
   delete fespace;
   delete fec;
   delete mesh;
   return 0;
}

void source_re(const Vector &x, Vector &f)
{
   f = 0.0;
   double x0 = length/2.0;
   double x1 = length/2.0;
   double x2 = length/2.0;
   x0 = 0.45;
   x1 = 0.35;
   x2 = 0.25;
   double alpha,beta;
   double n = 4.0*omega/M_PI;
   beta = pow(x0-x(0),2) + pow(x1-x(1),2);
   if (dim == 3) { beta += pow(x2-x(2),2); }
   double coeff = 16.0*omega*omega/M_PI/M_PI/M_PI;
   alpha = -pow(n,2) * beta;
   f[0] = coeff*exp(alpha);
   // f[1] = coeff*exp(alpha);



   x0 = 0.8;
   x1 = 0.8;
   beta = pow(x0-x(0),2) + pow(x1-x(1),2);
   if (dim == 3) { beta += pow(x2-x(2),2); }
   alpha = -pow(n,2) * beta;
   // f[0] += coeff*exp(alpha);


   bool in_pml = false;
   for (int i = 0; i<dim; i++)
   {
      if (x(i)<=comp_bdr(i,0) || x(i)>=comp_bdr(i,1))
      {
         in_pml = true;
         break;
      }
   }
   if (in_pml) f = 0.0;
}

void source_im(const Vector &x, Vector &f)
{
   f = 0.0;
}

double wavespeed(const Vector &x)
{
   double ws;
   ws = 1.0;
   return ws;
}