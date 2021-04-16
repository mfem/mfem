#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <memory>

using namespace std;
using namespace mfem;

int exact_sol = 0;
int dim = 0;

void E_exact(const Vector &xvec, Vector &E)
{
   double x=xvec[0], y=xvec[1], z=xvec[2];
   constexpr double pi = M_PI;

   if (exact_sol == 0)
   {
      E[0] = sin(2*pi*x)*sin(4*pi*y);
      E[1] = sin(4*pi*x)*sin(2*pi*y);
   }
   else if (exact_sol == 1)
   {
      if (dim == 3)
      {
         E(0) = x * y * (1.0-y) * z * (1.0-z);
         E(1) = x * y * (1.0-x) * z * (1.0-z);
         E(2) = x * z * (1.0-x) * y * (1.0-y);
      }
      else
      {
         E(0) = y * (1.0-y);
         E(1) = x * (1.0-x);
      }
   }
   else if (exact_sol == 2)
   {
      E[0] = cos(2*pi*x)*cos(4*pi*y);
      E[1] = cos(4*pi*x)*cos(2*pi*y);
   }
   else if (exact_sol == 3)
   {
      constexpr double kappa = 2.0 * pi;

      if (dim == 3)
      {
         E(0) = sin(kappa * y);
         E(1) = sin(kappa * z);
         E(2) = sin(kappa * x);
      }
   }
   else
   {
      constexpr double kappa = 2.0 * pi;
      if (dim == 3)
      {
         E(0) = sin(kappa * y) * sin(kappa * z);
         E(1) = sin(kappa * x) * sin(kappa * z);
         E(2) = sin(kappa * x) * sin(kappa * y);
      }
      else
      {
         E(0) = sin(kappa * y);
         E(1) = sin(kappa * x);
         if (xvec.Size() == 3) { E(2) = 0.0; }
      }
   }
}

void f_exact(const Vector &xvec, Vector &f)
{
   double x=xvec[0], y=xvec[1], z=xvec[2];
   constexpr double pi = M_PI;
   constexpr double pi2 = M_PI*M_PI;

   if (exact_sol == 0)
   {
      f[0] = 8*pi2*cos(4*pi*x)*cos(2*pi*y) + (1 + 16*pi2)*sin(2*pi*x)*sin(4*pi*y);
      f[1] = 8*pi2*cos(2*pi*x)*cos(4*pi*y) + (1 + 16*pi2)*sin(4*pi*x)*sin(2*pi*y);
   }
   else if (exact_sol == 1)
   {
      if (dim == 3)
      {
         f(0) = x * y * (1.0-y) * z * (1.0-z);
         f(1) = x * y * (1.0-x) * z * (1.0-z);
         f(2) = x * z * (1.0-x) * y * (1.0-y);

         f(0) += y * (1.0-y) + z * (1.0-z);
         f(1) += x * (1.0-x) + z * (1.0-z);
         f(2) += x * (1.0-x) + y * (1.0-y);
      }
      else
      {
         f(0) = y * (1.0-y);
         f(1) = x * (1.0-x);

         f(0) += 2.0;
         f(1) += 2.0;
      }
   }
   else if (exact_sol == 2)
   {
      f[0] = 8*pi2*sin(4*pi*x)*sin(2*pi*y) + (1 + 16*pi2)*cos(2*pi*x)*cos(4*pi*y);
      f[1] = 8*pi2*sin(2*pi*x)*sin(4*pi*y) + (1 + 16*pi2)*cos(4*pi*x)*cos(2*pi*y);
   }
   else if (exact_sol == 3)
   {
      constexpr double kappa = 2.0 * pi;
      if (dim == 3)
      {
         f(0) = (1. + kappa * kappa) * sin(kappa * y);
         f(1) = (1. + kappa * kappa) * sin(kappa * z);
         f(2) = (1. + kappa * kappa) * sin(kappa * x);
      }
   }
   else
   {
      constexpr double kappa = 2.0 * pi;
      if (dim == 3)
      {
         f(0) = (1.0 + 2.0 * kappa * kappa) * sin(kappa * y) * sin(kappa * z);
         f(1) = (1.0 + 2.0 * kappa * kappa) * sin(kappa * x) * sin(kappa * z);
         f(2) = (1.0 + 2.0 * kappa * kappa) * sin(kappa * x) * sin(kappa * y);
      }
      else
      {
         f(0) = (1. + kappa * kappa) * sin(kappa * y);
         f(1) = (1. + kappa * kappa) * sin(kappa * x);
         if (xvec.Size() == 3) { f(2) = 0.0; }
      }
   }
}

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../data/star.mesh";
   int ref_levels = 0;
   int order = 3;
   const char *fe = "n";

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&ref_levels, "-r", "--refine", "Uniform refinements.");
   args.AddOption(&order, "-o", "--order", "Polynomial degree.");
   args.AddOption(&exact_sol, "-p", "--problem", "Exact solution.");
   args.AddOption(&fe, "-fe", "--fe-type", "FE type. n for Hcurl, r for Hdiv");
   args.ParseCheck();

   bool ND;
   if (string(fe) == "n") { ND = true; }
   else if (string(fe) == "r") { ND = false; }
   else { MFEM_ABORT("Bad FE type. Must be 'n' or 'r'."); }

   Mesh mesh(mesh_file, 1, 1);
   dim = mesh.Dimension();
   for (int l = 0; l < ref_levels; l++) { mesh.UniformRefinement(); }

   int b1 = BasisType::GaussLobatto, b2 = BasisType::Integrated;
   unique_ptr<FiniteElementCollection> fec;
   if (ND) { fec.reset(new ND_FECollection(order, dim, b1, b2)); }
   else { fec.reset(new RT_FECollection(order-1, dim, b1, b2)); }

   FiniteElementSpace fes(&mesh, fec.get());
   Array<int> ess_dofs;
   fes.GetBoundaryTrueDofs(ess_dofs);

   BilinearForm a(&fes);
   a.AddDomainIntegrator(new VectorFEMassIntegrator);
   if (ND) { a.AddDomainIntegrator(new CurlCurlIntegrator); }
   else { a.AddDomainIntegrator(new DivDivIntegrator); }
   a.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   a.Assemble();

   LinearForm b(&fes);
   VectorFunctionCoefficient f_coeff(dim, f_exact);
   b.AddDomainIntegrator(new VectorFEDomainLFIntegrator(f_coeff));
   b.Assemble();

   GridFunction x(&fes);
   VectorFunctionCoefficient exact_coeff(dim, E_exact);
   x.ProjectCoefficient(exact_coeff);

   Vector X, B;
   OperatorHandle A;
   a.FormLinearSystem(ess_dofs, x, b, A, X, B);

   LORSolver<UMFPackSolver> lor_solver(a, ess_dofs);

   CGSolver cg;
   cg.SetAbsTol(0.0);
   cg.SetRelTol(1e-12);
   cg.SetMaxIter(100);
   cg.SetPrintLevel(1);
   cg.SetOperator(*A);
   cg.SetPreconditioner(lor_solver);
   cg.Mult(B, X);
   a.RecoverFEMSolution(X, b, x);

   double er = x.ComputeL2Error(exact_coeff);
   std::cout << "L^2 error: " << er << '\n';

   ParaViewDataCollection dc("LOR", &mesh);
   dc.SetPrefixPath("ParaView");
   dc.SetHighOrderOutput(true);
   dc.SetLevelsOfDetail(order);
   dc.RegisterField("u", &x);
   dc.SetCycle(0);
   dc.SetTime(0.0);
   dc.Save();

   return 0;
}
