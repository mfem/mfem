// Test integrator
// (∇ × E, F) 


#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

void E_exact_r(const Vector &x, Vector & E_r);

void curlE_exact_r(const Vector &x, Vector &curlE_r);

void maxwell_solution(const Vector & X, 
                      std::vector<complex<double>> &E, 
                      std::vector<complex<double>> &curlE, 
                      std::vector<complex<double>> &curlcurlE);

void maxwell_solution_r(const Vector & X, Vector &E_r, 
                      Vector &curlE_r, 
                      Vector &curlcurlE_r);

int dim;
int dimc;
double omega;

enum prob_type
{
   polynomial,
   plane_wave,
   fichera_oven  
};

prob_type prob;

int main(int argc, char *argv[])
{
   const char *mesh_file = "../../../data/inline-hex.mesh";
   int order = 1;
   int delta_order = 1;
   bool visualization = true;
   double rnum=1.0;
   int ref = 1;
   double theta = 0.0;
   bool adjoint_graph_norm = false;
   bool static_cond = false;
   int iprob = 0;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree)");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&rnum, "-rnum", "--number_of_wavelenths",
                  "Number of wavelengths");    
   args.AddOption(&iprob, "-prob", "--problem", "Problem case"
                  " 0: polynomial, 1: plane wave, 2: Gaussian beam");                     
   args.AddOption(&delta_order, "-do", "--delta_order",
                  "Order enrichment for DPG test space.");     
   args.AddOption(&theta, "-theta", "--theta",
                  "Theta parameter for AMR");                    
   args.AddOption(&adjoint_graph_norm, "-graph-norm", "--adjoint-graph-norm",
                  "-no-graph-norm", "--no-adjoint-graph-norm",
                  "Enable or disable Adjoint Graph Norm on the test space");                                
   args.AddOption(&ref, "-ref", "--serial_ref",
                  "Number of serial refinements.");       
   args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                  "--no-static-condensation", "Enable static condensation.");                                            
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   if (iprob > 2) { iprob = 0; }
   prob = (prob_type)iprob;

   omega = 2.*M_PI*rnum;


   Mesh mesh(mesh_file, 1, 1);
   dim = mesh.Dimension();

   dimc = (dim == 3) ? 3 : 1;

   // Define spaces
   // L2 space for E
   FiniteElementCollection *E_fec = new ND_FECollection(order,dim);
   FiniteElementSpace *E_fes = new FiniteElementSpace(&mesh,E_fec);


   FiniteElementCollection *curlE_fec = new L2_FECollection(order-1,dim);
   FiniteElementSpace *curlE_fes = new FiniteElementSpace(&mesh,curlE_fec,dimc);

   mfem::out << "E_fes space true dofs = " << E_fes->GetTrueVSize() << endl;
   mfem::out << "curlE_fes space true dofs = " << curlE_fes->GetTrueVSize() << endl;


   GridFunction E_gf(E_fes);
   VectorFunctionCoefficient E_cf(dim,E_exact_r);
   E_gf.ProjectCoefficient(E_cf);

   GridFunction curlE_gf(curlE_fes);
   VectorFunctionCoefficient curlE_cf(dimc,curlE_exact_r);
   curlE_gf.ProjectCoefficient(curlE_cf);


   char vishost[] = "localhost";
   int visport = 19916;
   socketstream E_sock(vishost, visport);
   E_sock.precision(8);
   E_sock << "solution\n"
               << mesh << E_gf 
               << "window_title 'Exact E'" << flush;

   socketstream curlE_sock(vishost, visport);
   curlE_sock.precision(8);
   curlE_sock << "solution\n"
               << mesh << curlE_gf 
               << "window_title 'Exact curlE'" << flush;



   MixedBilinearForm a(E_fes,curlE_fes);
   a.AddDomainIntegrator(new CurlIntegrator());
   a.Assemble();
   Array<int> empty;
   SparseMatrix A;
   a.FormRectangularSystemMatrix(empty,empty,A);


   Vector curl_load(A.Height());
   A.Mult(E_gf,curl_load);
   BilinearForm m(curlE_fes);
   m.AddDomainIntegrator(new VectorMassIntegrator);
   m.Assemble();
   SparseMatrix M;
   m.FormSystemMatrix(empty, M);


   GSSmoother prec(M);
   PCG(M, prec, curl_load, curlE_gf, 1, 200, 1e-12, 0.0);


   socketstream curlE2_sock(vishost, visport);
   curlE2_sock.precision(8);
   curlE2_sock << "solution\n"
               << mesh << curlE_gf 
               << "window_title 'Numerical curlE'" << flush;


   delete E_fec;
   delete E_fes;
   delete curlE_fec;
   delete curlE_fes;

   return 0;
}                                       

void E_exact_r(const Vector &x, Vector & E_r)
{
   Vector curlE_r;
   Vector curlcurlE_r;

   maxwell_solution_r(x,E_r,curlE_r,curlcurlE_r);
}

void curlE_exact_r(const Vector &x, Vector &curlE_r)
{
   Vector E_r;
   Vector curlcurlE_r;

   maxwell_solution_r(x,E_r,curlE_r,curlcurlE_r);
}



void maxwell_solution(const Vector & X, std::vector<complex<double>> &E, 
                      std::vector<complex<double>> &curlE, 
                      std::vector<complex<double>> &curlcurlE)
{
   double x = X(0);
   double y = X(1);
   double z;
   if (dim == 3)
   {
      z = X(2);
   } 

   E.resize(dim);
   curlE.resize(dimc);
   curlcurlE.resize(dim);

   if (dim == 3)
   {
      E[0] = y * z * (1.0 - y) * (1.0 - z);
      E[1] = x * y * z * (1.0 - x) * (1.0 - z);
      E[2] = x * y * (1.0 - x) * (1.0 - y);
      
      curlE[0] = (1.0 - x) * x * (y*(2.0*z-3.0)+1.0);
      curlE[1] = 2.0*(1.0 - y)*y*(x-z);
      curlE[2] = (z-1)*z*(1.0+y*(2.0*x-3.0));
      
      curlcurlE[0] = 2.0 * y * (1.0 - y) - (2.0 * x - 3.0) * z * (1 - z);
      curlcurlE[1] = 2.0 * y * (x * (1.0 - x) + (1.0 - z) * z);
      curlcurlE[2] = 2.0 * y * (1.0 - y) + x * (3.0 - 2.0 * z) * (1.0 - x);
   }
   else if (dim == 2)
   {
      double c = 2.0*M_PI;
      E[0] = sin(c * y);
      E[1] = sin(c * x);
      curlE[0] = c * (cos(c*x) - cos(c*y));
      curlcurlE[0] = c*c * sin(c*y);
      curlcurlE[1] = c*c * sin(c*x);
   }
   else
   {
      MFEM_ABORT("Dimension cannot be 1");
   }

}

void maxwell_solution_r(const Vector & X, Vector &E_r, 
                        Vector &curlE_r, 
                        Vector &curlcurlE_r)
{
   E_r.SetSize(dim);
   curlE_r.SetSize(dimc);
   curlcurlE_r.SetSize(dim);

   std::vector<complex<double>> E;
   std::vector<complex<double>> curlE;
   std::vector<complex<double>> curlcurlE;

   maxwell_solution(X,E,curlE,curlcurlE);
   for (int i = 0; i<dim; i++)
   {
      E_r(i) = E[i].real();
      curlcurlE_r(i) = curlcurlE[i].real();
   }
   for (int i = 0; i<dimc; i++)
   {
      curlE_r(i) = curlE[i].real();
   }
}
