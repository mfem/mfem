#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <memory>

using namespace std;
using namespace mfem;

struct Exact : Coefficient
{
   enum struct Fn { SIN, X4, EXP };
   Fn f;
   Exact(Fn f_) : f(f_) { }
   double u(double xyz[3])
   {
      constexpr double pi = M_PI;
      double x = xyz[0]; double y = xyz[1]; double z = xyz[2];
      switch (f)
      {
         case Fn::SIN: return sin(2*pi*x)*cos(2*pi*y)*cos(2*pi*z);
         case Fn::X4: return pow(x, 4);
         case Fn::EXP:
            return exp(0.1*sin(5.1*x - 6.2*y) + 0.3*cos(4.3*x +3.4*y));
      }
   }
   void du(double xyz[3], double du[3])
   {
      constexpr double pi = M_PI;
      double x = xyz[0]; double y = xyz[1]; double z = xyz[2];
      switch (f)
      {
         case Fn::SIN:
            du[0] = 2*pi*cos(2*pi*x)*cos(2*pi*y)*cos(2*pi*z);
            du[1] = -2*pi*sin(2*pi*x)*sin(2*pi*y)*cos(2*pi*z);
            du[2] = -2*pi*sin(2*pi*x)*cos(2*pi*y)*sin(2*pi*z);
            break;
         case Fn::X4: du[0] = 4*pow(x,3); du[1] = 0.0; du[2] = 0.0; break;
         case Fn::EXP:
         {
            double uu = u(xyz);
            du[0] = uu*(0.51*cos(5.1*x - 6.2*y) - 1.29*sin(4.3*x +3.4*y));
            du[1] = uu*(-0.62*cos(5.1*x - 6.2*y) - 1.02*sin(4.3*x +3.4*y));
            du[2] = 0.0;
            break;
         }
      }
   }
   double Eval(ElementTransformation &T, const IntegrationPoint &ip)
   {
      double xyz[3];
      Vector transip(xyz, 3);
      T.Transform(ip, transip);
      if (T.GetDimension() < 3) { xyz[2] = 0.0; }
      return u(xyz);
   }
};

struct Vel : VectorCoefficient
{
   enum struct Fn { CONST_XYZ, CONST_Y, ROT };
   Fn f;
   Vel(int dim_, Fn f_) : VectorCoefficient(dim_), f(f_) { }
   void vel(double xyz[3], double v[3])
   {
      double x = xyz[0]; double y = xyz[1]; double z = xyz[2];
      (void)z; // Get rid of warning
      v[2] = 0.0;
      switch (f)
      {
         case Fn::CONST_XYZ: v[0] = 1.0; v[1] = 1.0; v[2] = 1.0; break;
         case Fn::CONST_Y: v[0] = 0.0; v[1] = 1.0; break;
         // case Fn::ROT: v[0] = 1-2*y; v[1] = 2*x-1; break;
         case Fn::ROT: v[0] = -y; v[1] = x; break;
      }
   }
   double div(double xyz[3])
   {
      // All the example velocities have zero divergence
      return 0.0;
   }
   void Eval(Vector &V, ElementTransformation &T, const IntegrationPoint &ip)
   {
      double xyz[3], vxyz[3];
      Vector transip(xyz, 3);
      T.Transform(ip, transip);
      if (T.GetDimension() < 3) { xyz[2] = 0.0; }
      vel(xyz, vxyz);
      V = vxyz;
   }
};

struct RHS : Coefficient
{
   double g;
   Vel &v;
   Exact &e;
   RHS(double g_, Vel &v_, Exact &e_) : g(g_), v(v_), e(e_) { }
   double Eval(ElementTransformation &T, const IntegrationPoint &ip)
   {
      double xyz[3];
      Vector transip(xyz, 3);
      T.Transform(ip, transip);

      double u = e.u(xyz);
      double du[3];
      e.du(xyz, du);
      double ux = du[0]; double uy = du[1]; double uz = du[2];
      double b[3];
      v.vel(xyz, b);
      double divb = v.div(xyz);
      double b1 = b[0]; double b2 = b[1]; double b3 = b[2];
      return (g + divb)*u + b1*ux + b2*uy + b3*uz;
   }
};

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../data/star.mesh";
   int ref_levels = 0;
   int order = 3;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&ref_levels, "-r", "--refine", "Uniform refinements.");
   args.AddOption(&order, "-o", "--order", "Polynomial degree.");
   args.ParseCheck();

   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();
   for (int l = 0; l < ref_levels; l++) { mesh.UniformRefinement(); }

   int b1 = BasisType::GaussLobatto;
   L2_FECollection fec(order, dim, b1);

   FiniteElementSpace fes(&mesh, &fec);
   Array<int> ess_dofs;

   Exact exact_coeff(Exact::Fn(2));
   Vel vel_coeff(dim, Vel::Fn(1));
   RHS rhs_coeff(1.0, vel_coeff, exact_coeff);

   double alpha = 1.0;

   BilinearForm a(&fes);
   a.AddDomainIntegrator(new MassIntegrator);
   a.AddDomainIntegrator(new ConvectionIntegrator(vel_coeff, alpha));
   a.AddInteriorFaceIntegrator(
      new NonconservativeDGTraceIntegrator(vel_coeff, alpha));
   a.AddBdrFaceIntegrator(
      new NonconservativeDGTraceIntegrator(vel_coeff, alpha));
   // a.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   a.Assemble();

   LinearForm b(&fes);
   b.AddDomainIntegrator(new DomainLFIntegrator(rhs_coeff));
   b.AddBdrFaceIntegrator(
      new BoundaryFlowIntegrator(exact_coeff, vel_coeff, -1.0));
   b.Assemble();

   GridFunction x(&fes);
   x = 0.0;

   Vector X, B;
   OperatorHandle A;
   a.FormLinearSystem(ess_dofs, x, b, A, X, B);

   LORSolver<UMFPackSolver> lor_solver(a, ess_dofs);

   GMRESSolver solver;
   solver.SetAbsTol(0.0);
   solver.SetRelTol(1e-12);
   solver.SetMaxIter(300);
   solver.SetKDim(300);
   solver.SetPrintLevel(1);
   solver.SetOperator(*A);
   solver.SetPreconditioner(lor_solver);
   solver.Mult(B, X);
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
