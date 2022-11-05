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

// Neuman_data on x = 1;
void grad_u(const Vector & x, Vector & gradu)
{
   gradu.SetSize(2);

   gradu(0) = M_PI * cos(M_PI*x(0))*sin(M_PI*x(1));
   gradu(1) = M_PI * sin(M_PI*x(0))*cos(M_PI*x(1));
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

   mesh.UniformRefinement();


   Vector center;
   for (int i = 0; i<mesh.GetNBE(); i++)
   {
      Element * be = mesh.GetBdrElement(i);
      Array<int> vertices;
      be->GetVertices(vertices);

      double * coords1 = mesh.GetVertex(vertices[0]);
      double * coords2 = mesh.GetVertex(vertices[1]);

      Vector center(2);
      center(0) = 0.5*(coords1[0] + coords2[0]);
      center(1) = 0.5*(coords1[1] + coords2[1]);


      if (abs(center(1) - 1.0) < 1e-10)
      {
         // the top edge
         be->SetAttribute(1);
      }
      else if(abs(center(1)) < 1e-10 && (center(0) < 0.125 || center(0) > 0.875))
      {
         // bottom edge (left and right "corners")
         be->SetAttribute(2);
      }
      else
      {
         be->SetAttribute(3);
      }
   }
   mesh.SetAttributes();

   for (int i = 0; i < num_refs; i++)
   {
      mesh.UniformRefinement();
   }

   FunctionCoefficient rhs(rhs_function);
   ConstantCoefficient cf(diff_coeff);

   FPDESolver fpde;
   fpde.SetMesh(&mesh);
   fpde.SetOrder(order);
   fpde.SetAlpha(alpha);
   fpde.SetBeta(beta);

   Array<int> ess_bdr(mesh.bdr_attributes.Max());
   Array<int> inhomogenous_neuman_bdr(mesh.bdr_attributes.Max());
   ess_bdr = 0;
   ess_bdr[1] = 1;

   inhomogenous_neuman_bdr = 1;
   inhomogenous_neuman_bdr[1] = 0;
   fpde.SetEssentialBoundary(ess_bdr);
   fpde.SetNeumannBoundary(inhomogenous_neuman_bdr);

   fpde.Init();
   fpde.SetDiffusionCoefficient(&cf);
   fpde.SetRHSCoefficient(&rhs);
   // fpde.SetNeumannData(&neumann_cf);
   VectorFunctionCoefficient grad_cf(mesh.Dimension(), grad_u);
   fpde.SetGradientData(&grad_cf);
   fpde.SetupFEM();
   fpde.Solve();

   GridFunction * u = fpde.GetFEMSolution();

   FunctionCoefficient u_ex(u_function);
   double L2error = u->ComputeL2Error(u_ex);
   mfem::out << "L2error = " << L2error << std::endl;

   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "solution\n" << mesh << *u << flush;
   }
   return 0;
}