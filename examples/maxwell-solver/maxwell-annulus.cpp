

// sample runs: ./maxwell-annulus -ref 2 -o 2 -f 0.6

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

void maxwell_solution(const Vector &x, vector<complex<double>> &E);
void maxwell_curl(const Vector &x, vector<complex<double>> &curlE);

void E_bdr_data_Re(const Vector &x, Vector &E);
void E_bdr_data_Im(const Vector &x, Vector &E);

void E_exact_Re(const Vector &x, Vector &E);
void E_exact_Im(const Vector &x, Vector &E);

void E_exact_Curl_Re(const Vector &x, Vector &E);
void E_exact_Curl_Im(const Vector &x, Vector &E);

void source(const Vector &x, Vector & f);
double sigma_func(const Vector &x);

double mu = 1.0;
double epsilon = 1.0;
double omega;
int dim;

int main(int argc, char *argv[])
{
   // 0. Initialize MPI.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_SELF, &num_procs);
   MPI_Comm_rank(MPI_COMM_SELF, &myid);
   // 1. Parse command-line options.
   // const char *mesh_file = "torus1_4.mesh";
   // const char *mesh_file = "waveguide-bend2.mesh";
   const char *mesh_file = "meshes/annulus-quad-o3.mesh";

   int order = 1;
   int ref_levels = 1;
   double freq = 5.0;
   bool herm_conv = true;
   bool visualization = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
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

   // 2. Setup the mesh
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   Mesh * mesh = new Mesh(mesh_file, 1, 1);
   dim = mesh->Dimension();
   mesh->RemoveInternalBoundaries();

   mesh->UniformRefinement();

   // Angular frequency
   omega = 2.0 * M_PI * freq;

   ComplexOperator::Convention conv =
      herm_conv ? ComplexOperator::HERMITIAN : ComplexOperator::BLOCK_SYMMETRIC;

   FiniteElementCollection *fec = new ND_FECollection(order, dim);
   FiniteElementSpace *fespace = new FiniteElementSpace(mesh, fec);

   ComplexGridFunction x(fespace);
   x = 0.0;
   VectorFunctionCoefficient E_Re(dim, E_bdr_data_Re);
   VectorFunctionCoefficient E_Im(dim, E_bdr_data_Im);


   H1_FECollection H1fec(order, dim);
   FiniteElementSpace H1fes(mesh, &H1fec);

   GridFunction bump(&H1fes);
   FunctionCoefficient bump_coeff(sigma_func);
   bump.ProjectCoefficient(bump_coeff);
   {
      char vishost[] = "localhost";
      int visport = 19916;

      socketstream sol_sock_sigma(vishost, visport);
      sol_sock_sigma.precision(8);
      sol_sock_sigma << "solution\n"
                  << *mesh << bump 
                  << "window_title 'bump function'" << flush;
   }


   for (int iter = 0; iter<ref_levels; iter++)
   {

      int size = fespace->GetTrueVSize();
      cout << "Number of finite element unknowns: " << size << endl;

      Array<int> ess_tdof_list;
      Array<int> ess_bdr;
      if (mesh->bdr_attributes.Size())
      {
         ess_bdr.SetSize(mesh->bdr_attributes.Max());
         ess_bdr = 1;
      }
      fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

      ComplexLinearForm b(fespace, conv);
      b.Vector::operator=(0.0);
      b.Assemble();

      x.ProjectBdrCoefficientTangent(E_Re, E_Im, ess_bdr);


      ConstantCoefficient muinv(1.0/mu);
      ConstantCoefficient omeg(-pow(omega, 2) * epsilon);
      ConstantCoefficient sigma(-pow(omega, 2) * epsilon);
      ProductCoefficient c1(sigma,bump_coeff);
      ConstantCoefficient sigma1(-omega * epsilon);

      // Integrators inside the computational domain (excluding the PML region)
      SesquilinearForm a(fespace, conv);
      a.AddDomainIntegrator(new CurlCurlIntegrator(muinv),NULL);
      a.AddDomainIntegrator(new VectorFEMassIntegrator(omeg),NULL);

      a.AddDomainIntegrator(NULL,new VectorFEMassIntegrator(c1));
      a.Assemble(0);


      SesquilinearForm prec(fespace, conv);
      prec.AddDomainIntegrator(new CurlCurlIntegrator(muinv),NULL);
      prec.AddDomainIntegrator(new VectorFEMassIntegrator(omeg),NULL);


      prec.AddDomainIntegrator(NULL,new VectorFEMassIntegrator(c1));
      prec.Assemble(0);

      OperatorPtr A;
      Vector B, X;
      a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);

      OperatorPtr pA;
      prec.FormSystemMatrix(ess_tdof_list, pA);


      SparseMatrix * SpMat = (*pA.As<ComplexSparseMatrix>()).GetSystemMatrix();
      HYPRE_Int global_size = SpMat->Height();
      HYPRE_Int row_starts[2]; row_starts[0] = 0; row_starts[1] = global_size;
      HypreParMatrix * HypreMat = new HypreParMatrix(MPI_COMM_SELF,global_size,row_starts,SpMat);
      {
         MUMPSSolver mumps;
         mumps.SetOperator(*HypreMat);
         mumps.Mult(B,X);

         GMRESSolver gmres(MPI_COMM_WORLD);
         gmres.SetRelTol(1e-12);
         gmres.SetMaxIter(2000);
         gmres.SetPrintLevel(1);
         gmres.SetOperator(*A);
         gmres.SetPreconditioner(mumps); 
         gmres.Mult(B, X);
      }

      a.RecoverFEMSolution(X, b, x);

      int cdim = (dim == 2) ? 1 : dim;

      if (iter == ref_levels) break;
      mesh->UniformRefinement();
      fespace->Update();
      x.Update();
   }


   // 16. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      // Define visualization keys for GLVis (see GLVis documentation)
      string keys;
      keys = (dim == 3) ? "keys macF\n" : keys = "keys amrRljcUUuuu\n";

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
      {
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
      int num_frames = 16;

         while (sol_sock)
         {
            for (int i = 1; i<num_frames; i++)
            {   
               double t = (double)(i % num_frames) / num_frames;
               ostringstream oss;
               oss << "Harmonic Solution (t = " << t << " T)";

               add(cos(2.0 * M_PI * t), x.real(),
                  sin(2.0 * M_PI * t), x.imag(), x_t);
               sol_sock << "solution\n"
                        << *mesh << x_t 
                        << "window_title '" << oss.str() << "'" << flush;
            }
         }
      }
   }

   // 17. Free the used memory.
   delete fespace;
   delete fec;
   delete mesh;

   return 0;
}

double sigma_func(const Vector &x)
{
   double r = x.Norml2();
   double val = 0.0;
   if (r < 0.3)
   {
      val = 1.0;
   }
   else
   {
      r*=1.5;
      if (r<1)
      {
         // r*=.5;
         double d = r*r;
         double factor = 0.1;
         val = exp(factor) * exp(-factor/(1.-d));
      }
   }

   return 1.-val;
}


void E_bdr_data_Re(const Vector &x, Vector &E)
{
   // vector<complex<double>> Eval(E.Size());
   // maxwell_solution(x, Eval);
   // for (int i = 0; i < dim; ++i)
   // {
   //    E[i] = Eval[i].real();
   // }
   E_exact_Re(x,E);
}

// Define bdr_data solution
void E_bdr_data_Im(const Vector &x, Vector &E)
{
   // double r = x.Norml2();
   // vector<complex<double>> Eval(E.Size());
   // maxwell_solution(x, Eval);
   // for (int i = 0; i < dim; ++i)
   // {
   //    E[i] = Eval[i].imag();
   // }
   E_exact_Im(x,E);
}

void E_exact_Re(const Vector &x, Vector &E)
{
   E = 0.0;
   if (x.Norml2() < 0.3 )
   {
      vector<complex<double>> Eval(E.Size());
      maxwell_solution(x, Eval);
      for (int i = 0; i < dim; ++i)
      {
         E[i] = Eval[i].real();
      }
   }
}

void E_exact_Im(const Vector &x, Vector &E)
{
   E = 0.0;
   if (x.Norml2() < 0.3 )
   {
      vector<complex<double>> Eval(E.Size());
      maxwell_solution(x, Eval);
      for (int i = 0; i < dim; ++i)
      {
         E[i] = Eval[i].imag();
      }
   }
}

void maxwell_solution(const Vector &x, vector<complex<double>> &E)
{
   complex<double> zi = complex<double>(0., 1.);
   double k = omega * sqrt(epsilon * mu);
   Vector shift(dim);
   shift = 0.0;
   double x0 = x(0) + shift(0);
   double x1 = x(1) + shift(1);
   double r = sqrt(x0 * x0 + x1 * x1);
   double beta = k * r;

   // Bessel functions
   complex<double> H0, H0_r, H0_rr;
   complex<double> H1; 
   complex<double> H2;
   H0 = jn(0,beta) + zi * yn(0,beta);
   H1 = jn(1,beta) + zi * yn(1,beta);
   H2 = jn(2,beta) + zi * yn(2,beta);
   // H3 = jn(3,beta) + zi * yn(3,beta);

   H0_r = - k * H1;
   H0_rr = - k * k * (1.0/beta * H1 - H2); 

   // First derivatives
   double r_x = x0 / r;
   double r_y = x1 / r;
   double r_xy = -(r_x / r) * r_y;
   double r_xx = (1.0 / r) * (1.0 - r_x * r_x);

   complex<double> val, val_xx, val_xy ;
   val = 0.25 * zi * H0;
   val_xx = 0.25 * zi * (r_xx * H0_r + r_x * r_x * H0_rr);
   val_xy = 0.25 * zi * (r_xy * H0_r + r_x * r_y * H0_rr);
   E[0] = zi / k * (k * k * val + val_xx);
   E[1] = zi / k * val_xy;
}

void E_exact_Curl_Re(const Vector &x, Vector &E)
{
   E = 0.0;
   vector<complex<double>> Eval(E.Size());
   maxwell_curl(x, Eval);
   for (int i = 0; i < E.Size(); ++i)
   {
      E[i] = Eval[i].real();
   }
}

void E_exact_Curl_Im(const Vector &x, Vector &E)
{
   E = 0.0;
   vector<complex<double>> Eval(E.Size());
   maxwell_curl(x, Eval);
   for (int i = 0; i < E.Size(); ++i)
   {
      E[i] = Eval[i].imag();
   }
}


void maxwell_curl(const Vector &x, vector<complex<double>> &curlE)
{
   complex<double> zi = complex<double>(0., 1.);

   double k = omega * sqrt(epsilon * mu);
   Vector shift(dim);
   shift = 0.0;
   double x0 = x(0) + shift(0);
   double x1 = x(1) + shift(1);
   double r = sqrt(x0 * x0 + x1 * x1);
   double beta = k * r;

   // Bessel functions
   complex<double> H0_r;
   complex<double> H1;
   H1 = jn(1,beta) + zi * yn(1,beta);
   H0_r   = - k * H1;

   double r_y = x1 / r;
   complex<double> val_y;
   val_y = 0.25 * zi * H0_r * r_y;
   curlE[0] = zi / k * (- k * k * val_y);
}