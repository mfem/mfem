#include "mfem.hpp"
#include "Problems.hpp"
#include "IPsolver.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;



double dmanufacturedFun(const Vector &);
double fRhs(const Vector &);
double obstacle(const Vector &);

int main(int argc, char *argv[])
{
  int FEorder = 1; // order of the finite elements
  int linSolver = 0;
  int maxIPMiters = 30;
  bool iAmRoot = true;
  
  OptionsParser args(argc, argv);
  args.AddOption(&FEorder, "-o", "--order",\
		  "Order of the finite elements.");
  args.AddOption(&linSolver, "-linSolver", "--linearSolver", \
       "IP-Newton linear system solution strategy.");
  args.AddOption(&maxIPMiters, "-IPMiters", "--IPMiters",\
		  "Maximum number of IPM iterations");
  
  args.Parse();
  if(!args.Good())
  {
    args.PrintUsage(cout);
    return 1;
  }
  else
  {
    if( iAmRoot )
    {
      args.PrintOptions(cout);
    }
  }

  const char *meshFile = "../../data/inline-quad.mesh";
  Mesh *mesh = new Mesh(meshFile, 1, 1);
  int dim = mesh->Dimension(); // geometric dimension of the domain
  {
     int ref_levels = 3;
     for (int l = 0; l < ref_levels; l++)
     {
        mesh->UniformRefinement();
     }
  }

  FiniteElementCollection *fec = new H1_FECollection(FEorder, dim);
  FiniteElementSpace      *Vh  = new FiniteElementSpace(mesh, fec);
  ObstacleProblem problem(Vh, &fRhs, &obstacle);
  
  int dimD = problem.GetDimD();
  Vector x0(dimD); x0 = 0.0;
  Vector xf(dimD); xf = 0.0;

  InteriorPointSolver optimizer(&problem); 
  optimizer.SetTol(1.e-7);
  optimizer.SetLinearSolver(linSolver);
  optimizer.SetMaxIter(maxIPMiters);
  optimizer.Mult(x0, xf);

  double Einitial = problem->E(x0);
  double Efinal = problem->E(xf);
  cout << "Energy objective at initial point = " << Einitial << endl;
  cout << "Energy objective at QP optimizer = " << Efinal << endl;


  GridFunction d_gf(Vh);

  d_gf = xf;

  FunctionCoefficient dm_fc(dmanufacturedFun); // pseudo-manufactured solution
  GridFunction dm_gf(Vh);
  dm_gf.ProjectCoefficient(dm_fc);
  
  ParaViewDataCollection paraview_dc("BarrierProblemSolution", mesh);
  paraview_dc.SetPrefixPath("ParaView");
  paraview_dc.SetLevelsOfDetail(FEorder);
  paraview_dc.SetDataFormat(VTKFormat::BINARY);
  paraview_dc.SetHighOrderOutput(true);
  paraview_dc.SetCycle(0);
  paraview_dc.SetTime(0.0);
  paraview_dc.RegisterField("d(x) (numerical)", &d_gf);
  paraview_dc.RegisterField("d(x) (pseudo-manufactured)", &dm_gf);
  paraview_dc.Save();
  
  
  delete Vh;
  delete fec;
  delete mesh;
  return 0;
}


double dmanufacturedFun(const Vector &x)
{
  return cos(2*M_PI*x(0)) + 0.2 - 2.0*(pow(x(0),3) - 1.5*pow(x(0),2));
}


// f(x) forcing term... which enters the objective energy functional
// E(d) = 0.5 d^T K d - f^T d, where f is a discrete vector representation
// of f(x). f(x) is such that in the absence of bound-constraints then
// the solution of the optimization problem satisfies the PDE
// -div(grad(d)) + d = f + homogeneous Neumann conditions on the unit interval,
// for d(x) = cos(2 \pi x) + a0 + a3 (x^3 - 1.5 x^2), a2 = 0.2, a3 = -2
double fRhs(const Vector &x)
{
  double fx = 0.;
  fx = 0.2 - 2.0 * (pow(x(0),3)- 1.5*pow(x(0),2.) - 6 * x(0) + 3.) + (1. + pow(2.*M_PI,2))*cos(2.*M_PI*x(0));
  return fx;
}

double obstacle(const Vector &x)
{
  return 0.0;
}
