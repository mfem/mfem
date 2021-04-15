
//
// Compile with: make maxwell-impedanceBC
//
//               maxwell-impedanceBC -o 2 -f 1.6 -ref 2 -prob 0 -m ../../data/beam-hex.mesh
//               maxwell-impedanceBC -o 2 -f 1.6 -ref 2 -prob 0 -m ../../data/beam-tet.mesh
//
#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace mfem;
using namespace std;

void E_exact_Re(const Vector &x, Vector &E);
void E_exact_Im(const Vector &x, Vector &E);
void Curl_exact_Re(const Vector &x, Vector &Curl);
void Curl_exact_Im(const Vector &x, Vector &Curl);
void maxwell_solution(const Vector &x, vector<complex<double>> &Eval);
void maxwell_curl(const Vector &x, vector<complex<double>> &Curl);
void maxwell_curlcurl(const Vector &x, vector<complex<double>> &CurlCurl);
void f_exact_Re(const Vector &x, Vector &E);
void f_exact_Im(const Vector &x, Vector &E);

double omega;
int prob_kind = 0;
int dim;

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../../data/beam-hex.mesh";
   // const char *mesh_file = "../../data/beam-tet.mesh";
   int order = 2;
   int ref_levels = 2;
   double freq = 1.6;
   bool herm_conv = true;
   bool visualization = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&prob_kind, "-prob", "--problem_kind",
                  "Choice of problem");                  
   args.AddOption(&ref_levels, "-ref", "--refinements",
                  "Number of refinements");
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


   Mesh * mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   // Angular frequency
   omega = 2.0 * M_PI * freq;
   if (prob_kind == 0)  MFEM_VERIFY (omega > M_PI * M_PI, "increase fequency");

   mesh->ReorientTetMesh();

   // 7. Define a finite element space on the mesh. Here we use the Nedelec
   //    finite elements of the specified order.
   FiniteElementCollection *fec = new ND_FECollection(order, dim);
   FiniteElementSpace *fespace = new FiniteElementSpace(mesh, fec);

   ComplexOperator::Convention conv =
   herm_conv ? ComplexOperator::HERMITIAN : ComplexOperator::BLOCK_SYMMETRIC;

   Array<int> ess_bdr;
   Array<int> imp_bdr;
   cout << mesh->bdr_attributes.Max() << endl;
   if (mesh->bdr_attributes.Size())
   {
      ess_bdr.SetSize(mesh->bdr_attributes.Max());
      imp_bdr.SetSize(mesh->bdr_attributes.Max());
      ess_bdr = 0;
      imp_bdr = 1;
   }

   // required coefficients
   ConstantCoefficient one(1.0);
   ConstantCoefficient omeg(-pow(omega, 2));
   ConstantCoefficient om(omega);
   VectorFunctionCoefficient E_Re(dim, E_exact_Re);
   VectorFunctionCoefficient E_Im(dim, E_exact_Im);
   VectorFunctionCoefficient Curl_Re(dim, Curl_exact_Re);
   VectorFunctionCoefficient Curl_Im(dim, Curl_exact_Im);
   VectorFunctionCoefficient f_Re(dim,f_exact_Re);
   VectorFunctionCoefficient f_Im(dim,f_exact_Im);
   // For - <n x curl E> 
   ScalarVectorProductCoefficient c1_Re(-1.0,Curl_Re);                        
   ScalarVectorProductCoefficient c1_Im(-1.0,Curl_Im);
   // For  i omega (n x n x E)
   ScalarVectorProductCoefficient c2_Re(-omega,E_Im);                        
   ScalarVectorProductCoefficient c2_Im(omega,E_Re);


   // Weak form with impedance condition
   // n x curl E + i omega (n x n x E) = G on \partial \Omega 
   // (curlE, curlH) - omega^2 (E,H) + < n x curlE, H> = (F, H)
   // (curlE, curlH) - omega^2 (E,H) - i omega <n x n x E, H> = (F,H) - <G,H>

   ComplexLinearForm b(fespace, conv);
   b.AddDomainIntegrator(new VectorFEDomainLFIntegrator(f_Re), 
                         new VectorFEDomainLFIntegrator(f_Im));
   b.AddBoundaryIntegrator(new VectorFEBoundaryTangentLFIntegrator(c1_Re), 
                           new VectorFEBoundaryTangentLFIntegrator(c1_Im), imp_bdr);
   b.AddBoundaryIntegrator(new VectorFEDomainLFIntegrator(c2_Re), 
                           new VectorFEDomainLFIntegrator(c2_Im), imp_bdr);   

   SesquilinearForm a(fespace, conv);
   a.AddDomainIntegrator(new CurlCurlIntegrator(one), NULL);
   a.AddDomainIntegrator(new VectorFEMassIntegrator(omeg),NULL);
   a.AddBoundaryIntegrator(NULL, new VectorFEMassIntegrator(om),imp_bdr);

   ComplexGridFunction x(fespace);
   x = 0.0;
   x.ProjectBdrCoefficientTangent(E_Re, E_Im, ess_bdr);


   ConvergenceStudy rates_re, rates_im;

   for (int l = 0; l<=ref_levels; l++)
   {
      int size = fespace->GetTrueVSize();
      cout << "Number of finite element unknowns: " << size << endl;
   
      b.Assemble();
      a.Assemble();

      OperatorPtr A;
      Vector B, X;
      Array<int> ess_tdof_list;
      fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
      a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);

      // 14. Solve using a direct solver
      UMFPackSolver csolver(*A.As<ComplexSparseMatrix>()->GetSystemMatrix());
      csolver.Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
      csolver.SetPrintLevel(1);
      csolver.Mult(B, X);

      a.RecoverFEMSolution(X, b, x);

      rates_re.AddHcurlGridFunction(&x.real(),&E_Re,&Curl_Re);
      rates_im.AddHcurlGridFunction(&x.imag(),&E_Im,&Curl_Im);

      if (l==ref_levels) break;

      mesh->UniformRefinement();
      mesh->ReorientTetMesh();
      fespace->Update();
      a.Update();
      b.Update();
      x.Update();
   }

   rates_re.Print();
   rates_im.Print();


   ComplexGridFunction x_ex(fespace);
   x_ex.ProjectCoefficient(E_Re, E_Im);

   if (visualization)
   {

      char vishost[] = "localhost";
      int visport = 19916;

      socketstream sol_sock_re(vishost, visport);
      sol_sock_re.precision(8);
      sol_sock_re << "solution\n"
                  << *mesh << x.real() 
                  << "window_title 'Solution real part'" << flush;

      socketstream sol_sock_im(vishost, visport);
      sol_sock_im.precision(8);
      sol_sock_im << "solution\n"
                  << *mesh << x.imag()
                  << "window_title 'Solution imag part'" << flush;

      socketstream sol_sock_re_ex(vishost, visport);
      sol_sock_re_ex.precision(8);
      sol_sock_re_ex << "solution\n"
                  << *mesh << x_ex.real() 
                  << "window_title 'Exact real part'" << flush;

      socketstream sol_sock_im_ex(vishost, visport);
      sol_sock_im_ex.precision(8);
      sol_sock_im_ex << "solution\n"
                  << *mesh << x_ex.imag()
                  << "window_title 'Exact imag part'" << flush;            

   }
   // 18. Free the used memory.
   delete fespace;
   delete fec;
   delete mesh;
}


void maxwell_solution(const Vector &x, vector<complex<double>> &E)
{
   // Initialize
   int dim = x.Size();
   for (int i = 0; i < dim; ++i)
   {
      E[i] = 0.0;
   }
   if (prob_kind == 0)
   {
      complex<double> zi = complex<double>(0., 1.);
      double k10 = sqrt(omega * omega - M_PI * M_PI);
      E[1] = -zi * omega / M_PI * sin(M_PI*x(2))*exp(zi * k10 * x(0));
   }
   else if (prob_kind == 1)
   {
      E[0] = x(0)*x(1);
      E[1] = x(1)*x(2);
      E[2] = x(2)*x(0);
   }
   else
   {
      E[0] = x(0)*x(1)*x(2)*x(2);
      E[1] = x(1)*x(2)*x(0)*x(0)*x(0);
      E[2] = x(2)*x(0)*x(1);
   }
   


}

void maxwell_curl(const Vector &x, vector<complex<double>> &Curl)
{
   // Initialize
   int dim = x.Size();
   for (int i = 0; i < dim; ++i)
   {
      Curl[i] = 0.0;
   }
   if (prob_kind == 0)
   {
      complex<double> zi = complex<double>(0., 1.);
      double k10 = sqrt(omega * omega - M_PI * M_PI);
      Curl[0] = zi * omega * cos(M_PI*x(2)) * exp(zi*k10*x(0));
      Curl[1] = 0.0;
      Curl[2] = omega * k10 / M_PI * sin(M_PI * x(2)) * exp(zi*k10*x(0));
   }
   else if (prob_kind == 1)
   {
      Curl[0] = -x(1);
      Curl[1] = -x(2);
      Curl[2] = -x(0);
   }
   else
   {
      Curl[0] = x(0)*x(2) - x(0)*x(0)*x(0)*x(1);
      Curl[1] = (2.0*x(0)-1.0)*x(1)*x(2);
      Curl[2] = x(0)*x(2)*(3.0*x(0)*x(1)-x(2));
   }
   
}

void maxwell_curlcurl(const Vector &x, vector<complex<double>> &CurlCurl)
{
   // Initialize
   int dim = x.Size();
   for (int i = 0; i < dim; ++i)
   {
      CurlCurl[i] = 0.0;
   }

   if (prob_kind == 0)
   {
      complex<double> zi = complex<double>(0., 1.);
      double k10 = sqrt(omega * omega - M_PI * M_PI);
   // complex<double> f = -zi * omega / M_PI * sin(M_PI*x(2))*exp(zi * k10 * x(0));
   // complex<double> f_x = omega * k10 /M_PI * sin(M_PI*x(2))*exp(zi * k10 * x(0));
      complex<double> f_xx = zi * omega * k10 * k10 /M_PI * sin(M_PI*x(2))*exp(zi * k10 * x(0));
      complex<double> f_xy = 0.0;
   // complex<double> f_z = -zi * omega * cos(M_PI*x(2))*exp(zi * k10 * x(0));
      complex<double> f_zy = 0.0;
      complex<double> f_zz = zi * omega * M_PI * sin(M_PI*x(2))*exp(zi * k10 * x(0));
      CurlCurl[0] = f_xy;
      CurlCurl[1] = -f_zz - f_xx;
      CurlCurl[2] = f_zy;
   }
   else if (prob_kind == 1)
   {
      CurlCurl[0] = 1.0;
      CurlCurl[1] = 1.0;
      CurlCurl[2] = 1.0;
   }
   else
   {
      CurlCurl[0] = 3.0*x(0)*x(0)*x(2) - 2*x(0)*x(1) + x(1);
      CurlCurl[1] = -6*x(0)*x(1)*x(2) + x(0) + x(2)*x(2);
      CurlCurl[2] = x(0)*x(0)*x(0) + 2.0*x(1)*x(2);
   }
}

void E_exact_Re(const Vector &x, Vector &E)
{
   int dim = x.Size();
   vector<complex<double>> Eval(E.Size());
   maxwell_solution(x, Eval);
   for (int i = 0; i < dim; ++i)
   {
      E[i] = Eval[i].real();
   }
}

void E_exact_Im(const Vector &x, Vector &E)
{
   int dim = x.Size();
   vector<complex<double>> Eval(E.Size());
   maxwell_solution(x, Eval);
   for (int i = 0; i < dim; ++i)
   {
      E[i] = Eval[i].imag();
   }
}

void Curl_exact_Re(const Vector &x, Vector &Curl)
{
   int dim = x.Size();
   vector<complex<double>> Eval(Curl.Size());
   maxwell_curl(x, Eval);
   for (int i = 0; i < dim; ++i)
   {
      Curl[i] = Eval[i].real();
   }
}

void Curl_exact_Im(const Vector &x, Vector &Curl)
{
   int dim = x.Size();
   vector<complex<double>> Eval(Curl.Size());
   maxwell_curl(x, Eval);
   for (int i = 0; i < dim; ++i)
   {
      Curl[i] = Eval[i].imag();
   }
}

void f_exact_Re(const Vector &x, Vector &E)
{
   int dim = x.Size();
   vector<complex<double>> Eval(E.Size());
   vector<complex<double>> CurlCurl(E.Size());
   maxwell_solution(x, Eval);
   maxwell_curlcurl(x, CurlCurl);
   for (int i = 0; i < dim; ++i)
   {
      E[i] = (CurlCurl[i] - omega * omega * Eval[i]).real();
   }
}

void f_exact_Im(const Vector &x, Vector &E)
{
   int dim = x.Size();
   vector<complex<double>> Eval(E.Size());
   vector<complex<double>> CurlCurl(E.Size());
   maxwell_solution(x, Eval);
   maxwell_curlcurl(x, CurlCurl);
   for (int i = 0; i < dim; ++i)
   {
      E[i] = (CurlCurl[i] - omega * omega * Eval[i]).imag();
   }
}
