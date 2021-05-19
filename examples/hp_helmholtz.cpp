//                                MFEM Example 6
//
// Compile with: make ex6
//
// Sample runs:  ./hp_helmholtz -m ../data/inline-quad.mesh -o 1

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

void gaussian_beam(const Vector & X, complex<double> & p, std::vector<complex<double>> & dp, complex<double> & d2p);
double exact_beam(const Vector & X);
void grad_beam(const Vector & X, Vector & gradp);
double rhs(const Vector & X);

double omega = 21.0 * 2.*M_PI;


int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../data/inline-quad.mesh";
   int order = 1;
   int max_dofs = 100000;
   bool visualization = true;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&max_dofs, "-md", "--max-dofs",
                  "Stop after reaching this many degrees of freedom.");
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
   int dim = mesh.Dimension();
   int sdim = mesh.SpaceDimension();
   mesh.UniformRefinement();


   H1_FECollection fec(order, dim);
   FiniteElementSpace fespace(&mesh, &fec);

   BilinearForm a(&fespace);
   LinearForm b(&fespace);

   ConstantCoefficient one(1.0);
   ConstantCoefficient zero(0.0);
   FunctionCoefficient frhs(rhs);
   ConstantCoefficient omeg(-omega*omega);

   BilinearFormIntegrator *integ = new DiffusionIntegrator(one);
   BilinearFormIntegrator *minteg = new MassIntegrator(omeg);
   a.AddDomainIntegrator(integ);
   // a.AddDomainIntegrator(minteg);
   // b.AddDomainIntegrator(new DomainLFIntegrator(zero));
   b.AddDomainIntegrator(new DomainLFIntegrator(frhs));

   FunctionCoefficient gbeam(exact_beam);
   VectorFunctionCoefficient gradbeam(dim,grad_beam);

   GridFunction x(&fespace);
   x = 0.0;
   // x.ProjectCoefficient(gbeam);



   // 8. All boundary attributes will be used for essential (Dirichlet) BC.
   MFEM_VERIFY(mesh.bdr_attributes.Size() > 0,
               "Boundary attributes required in the mesh.");
   Array<int> ess_bdr(mesh.bdr_attributes.Max());
   ess_bdr = 1;

   // 9. Connect to GLVis.
   char vishost[] = "localhost";
   int  visport   = 19916;
   socketstream sol_sock;
   if (visualization)
   {
      sol_sock.open(vishost, visport);
      // sol_sock.precision(8);
      // sol_sock << "solution\n" << mesh << x << flush;
   }
   // cin.get();

   
   FiniteElementSpace flux_fespace(&mesh, &fec, sdim);
   ZienkiewiczZhuEstimator estimator(*integ, x, flux_fespace);
   estimator.SetAnisotropic(false);

   ThresholdRefiner refiner(estimator);
   refiner.SetTotalErrorFraction(0.7);

   ConvergenceStudy rates;

   for (int it = 0; ; it++)
   {
      int cdofs = fespace.GetTrueVSize();
      cout << "\nAMR iteration " << it << endl;
      cout << "Number of unknowns: " << cdofs << endl;

      // 13. Assemble the right-hand side.
      b.Assemble();

      // 14. Set Dirichlet boundary values in the GridFunction x.
      //     Determine the list of Dirichlet true DOFs in the linear system.
      Array<int> ess_tdof_list;
      x.ProjectBdrCoefficient(gbeam, ess_bdr);
      fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

      // 15. Assemble the stiffness matrix.
      a.Assemble();

      OperatorPtr A;
      Vector B, X;

      const int copy_interior = 1;
      a.FormLinearSystem(ess_tdof_list, x, b, A, X, B, copy_interior);

#ifndef MFEM_USE_SUITESPARSE
         // Use a simple symmetric Gauss-Seidel preconditioner with PCG.
         GSSmoother M((SparseMatrix&)(*A));
         PCG(*A, M, B, X, 3, 200, 1e-12, 0.0);
#else
         // If MFEM was compiled with SuiteSparse, use UMFPACK to solve the system.
         UMFPackSolver umf_solver;
         umf_solver.Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
         umf_solver.SetOperator(*A);
         umf_solver.Mult(B, X);
#endif
      a.RecoverFEMSolution(X, b, x);

      rates.AddH1GridFunction(&x,&gbeam,&gradbeam);

      // 19. Send solution by socket to the GLVis server.
      if (visualization && sol_sock.good())
      {
         sol_sock.precision(8);
         sol_sock << "solution\n" << mesh << x << flush;
      }

      if (cdofs > max_dofs)
      {
         cout << "Reached the maximum number of dofs. Stop." << endl;
         break;
      }

      refiner.Apply(mesh);
      if (refiner.Stop())
      {
         cout << "Stopping criterion satisfied. Stop." << endl;
         break;
      }

      fespace.Update();
      x.Update();

      // 22. Inform also the bilinear and linear forms that the space has
      //     changed.
      a.Update();
      b.Update();
   }
   rates.Print();

   return 0;
}



double exact_beam(const Vector & X)
{
   complex<double> p, d2p;
   std::vector<complex<double>> dp(2);
   gaussian_beam(X,p,dp,d2p);
   return p.real();

}

void grad_beam(const Vector & X, Vector & gradp)
{
   complex<double> p, d2p;
   std::vector<complex<double>> dp(2);
   gradp.SetSize(2);
   gaussian_beam(X,p,dp,d2p);
   for (int i = 0; i<2; i++)
   {
      gradp[i] = dp[i].real();
   }
}

double rhs(const Vector & X)
{
   complex<double> p, d2p;
   std::vector<complex<double>> dp(2);
   gaussian_beam(X,p,dp,d2p);
   return -d2p.real();
   // return -d2p.real() - omega * omega * p.real();
}


void gaussian_beam(const Vector & X, complex<double> & p, std::vector<complex<double>>  & dp, complex<double> & d2p)
{
   double rk = omega;
   double alpha = 45 * M_PI/180.;
   double sina = sin(alpha); 
   double cosa = cos(alpha);
   // shift the origin
   double xprim=X(0) + 0.1; 
   double yprim=X(1) + 0.1;

   double  x = xprim*sina - yprim*cosa;
   double  y = xprim*cosa + yprim*sina;
   double  dxdxprim = sina, dxdyprim = -cosa;
   double  dydxprim = cosa, dydyprim =  sina;
   //wavelength
   double rl = 2.*M_PI/rk;

   // beam waist radius
   double w0 = 0.05;

   // function w
   double fact = rl/M_PI/(w0*w0);
   double aux = 1. + (fact*y)*(fact*y);

   double w = w0*sqrt(aux);
   double dwdy = w0*fact*fact*y/sqrt(aux);
   double d2wdydy = w0*fact*fact*(1. - (fact*y)*(fact*y)/aux)/sqrt(aux);

   double phi0 = atan(fact*y);
   double dphi0dy = cos(phi0)*cos(phi0)*fact;
   double d2phi0dydy = -2.*cos(phi0)*sin(phi0)*fact*dphi0dy;

   double r = y + 1./y/(fact*fact);
   double drdy = 1. - 1./(y*y)/(fact*fact);
   double d2rdydy = 2./(y*y*y)/(fact*fact);

   // pressure
   complex<double> zi = complex<double>(0., 1.);
   complex<double> ze = - x*x/(w*w) - zi*rk*y - zi * M_PI * x * x/rl/r + zi*phi0/2.;

   complex<double> zdedx = -2.*x/(w*w) - 2.*zi*M_PI*x/rl/r;
   complex<double> zdedy = 2.*x*x/(w*w*w)*dwdy - zi*rk + zi*M_PI*x*x/rl/(r*r)*drdy + zi*dphi0dy/2.;
   complex<double> zd2edxdx = -2./(w*w) - 2.*zi*M_PI/rl/r;
   complex<double> zd2edxdy = 4.*x/(w*w*w)*dwdy + 2.*zi*M_PI*x/rl/(r*r)*drdy;
   complex<double> zd2edydx = zd2edxdy;
   complex<double> zd2edydy = -6.*x*x/(w*w*w*w)*dwdy*dwdy + 2.*x*x/(w*w*w)*d2wdydy - 2.*zi*M_PI*x*x/rl/(r*r*r)*drdy*drdy
                            + zi*M_PI*x*x/rl/(r*r)*d2rdydy + zi/2.*d2phi0dydy;

   double pf = pow(2.0/M_PI/(w*w),0.25);
   double dpfdy = -pow(2./M_PI/(w*w),-0.75)/M_PI/(w*w*w)*dwdy;
   double d2pfdydy = -1./M_PI*pow(2./M_PI,-0.75)*(-1.5*pow(w,-2.5)
                     *dwdy*dwdy + pow(w,-1.5)*d2wdydy);


   complex<double> zp = pf*exp(ze);
   complex<double> zdpdx = zp*zdedx;
   complex<double> zdpdy = dpfdy*exp(ze)+zp*zdedy;
   complex<double> zd2pdxdx = zdpdx*zdedx + zp*zd2edxdx;
   complex<double> zd2pdxdy = zdpdy*zdedx + zp*zd2edxdy;
   complex<double> zd2pdydx = dpfdy*exp(ze)*zdedx + zdpdx*zdedy + zp*zd2edydx;
   complex<double> zd2pdydy = d2pfdydy*exp(ze) + dpfdy*exp(ze)*zdedy + zdpdy*zdedy + zp*zd2edydy;

   p = zp;
   dp[0] = zdpdx*dxdxprim + zdpdy*dydxprim;
   dp[1] = zdpdx*dxdyprim + zdpdy*dydyprim;

   d2p = (zd2pdxdx*dxdxprim + zd2pdydx*dydxprim)*dxdxprim + (zd2pdxdy*dxdxprim + zd2pdydy*dydxprim)*dydxprim
       + (zd2pdxdx*dxdyprim + zd2pdydx*dydyprim)*dxdyprim + (zd2pdxdy*dxdyprim + zd2pdydy*dydyprim)*dydyprim;
}