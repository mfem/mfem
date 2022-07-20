#include "mfem.hpp"
#include <fstream>
#include <iostream>

#include "common/fpde.hpp"
#include "common/rational_approximation.hpp"


using namespace std;
using namespace mfem;

double alpha = 1.0;
double beta = 0.0;


double u_function(const Vector & x)
{
   double u = sin(M_PI*x(0))*sin(M_PI*x(1));
   return u;
}


double rhs_function(const Vector & x)
{
   double c = pow(2*M_PI*M_PI,alpha)+beta;
   double u = sin(M_PI*x(0))*sin(M_PI*x(1));
   return c*u;
}


int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../../mfem/data/star.mesh";
   int order = 1;
   int num_refs = 3;
   bool visualization = true;
   double diff_coeff = 1.0;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&num_refs, "-r", "--refs",
                  "Number of uniform refinements");
   args.AddOption(&alpha, "-alpha", "--alpha",
                  "Fractional exponent");
   args.AddOption(&beta, "-beta", "--beta",
                  "Shift in f(z) = z^-α ");        
   args.AddOption(&diff_coeff, "-diff-coeff", "--diff-coeff",
                  "Diffusion coefficient ");                              
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


   Mesh mesh(mesh_file, 1, 1);


   Vector center;
   for (int i = 0; i<mesh.GetNE(); i++)
   {
      mesh.GetElementCenter(i,center);
      if (center[0]>0.5)
      {
         mesh.SetAttribute(i,2);
      }
      else
      {
         mesh.SetAttribute(i,1);
      }
   }
   mesh.SetAttributes();

   Vector constants(mesh.attributes.Max());
   constants(0) = -1.; 
   constants(1) = 1;

   PWConstCoefficient pw_cf(constants); 



   for (int i = 0; i < num_refs; i++)
   {
      mesh.UniformRefinement();
   }

   FunctionCoefficient rhs(rhs_function);
   ConstantCoefficient cf(pow(diff_coeff,1./alpha));
   // FPDESolver fpde(&mesh,order,&f,alpha);
   FPDESolver fpde;
   fpde.SetMesh(&mesh);
   fpde.SetOrder(order);
   fpde.SetAlpha(alpha);
   fpde.SetBeta(beta);

   Array<int> ess_bdr(mesh.bdr_attributes.Max());
   ess_bdr = 0;
   fpde.SetEssentialBoundary(ess_bdr);

   fpde.Init();
   fpde.SetDiffusionCoefficient(&cf);
   // fpde.SetRHSCoefficient(&rhs);
   fpde.SetRHSCoefficient(&pw_cf);
   fpde.SetupFEM();

   fpde.Solve();

   GridFunction * u = fpde.GetFEMSolution();


   L2_FECollection L2fec(order, mesh.Dimension());
   FiniteElementSpace L2fes(&mesh, &L2fec);

   GridFunction rhs_gf(&L2fes);
   rhs_gf.ProjectCoefficient(pw_cf);


   // FunctionCoefficient u_ex(u_function);

   // double L2error = u->ComputeL2Error(u_ex);

   // mfem::out << "L2error = " << L2error << std::endl;

   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "solution\n" << mesh << *u << flush;

      socketstream rhs_sock(vishost, visport);
      rhs_sock.precision(8);
      rhs_sock << "solution\n" << mesh << rhs_gf << flush;
   }



   // Array<double> coeffs, poles; 
   // ComputePartialFractionApproximation(alpha,beta,coeffs,  poles);

   // mfem::out << "coeffs = " << endl;
   // coeffs.Print();

   // mfem::out << "poles = " << endl;
   // poles.Print();




   return 0;
}