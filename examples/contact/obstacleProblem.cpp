#include "mfem.hpp"
#include "problems.hpp"
#include "IPsolver.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;



double dmanufacturedFun(const Vector &);
double fRhs(const Vector &);

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
     int ref_levels =
        (int)floor(log(2000./mesh->GetNE())/log(2.)/dim);
     for (int l = 0; l < ref_levels; l++)
     {
        mesh->UniformRefinement();
     }
  }

  FiniteElementCollection *fec = new H1_FECollection(FEorder, dim);
  FiniteElementSpace      *Vh  = new FiniteElementSpace(mesh, fec);
  ObstacleProblem problem(Vh, &fRhs);
  
  int dimD = problem.GetDimD();
  cout << "dimension of displacement field = " << dimD << endl;
  Array<int> offsets(3);
  offsets[0] = 0;
  offsets[1] = dimD;
  offsets[2] = dimD;
  offsets.PartialSum();


  InteriorPointSolver optimizer(&problem); 
  BlockVector x0(offsets); x0 = 100.0;
  BlockVector xf(offsets); xf = 0.0;
  optimizer.SetTol(1.e-6);
  optimizer.SetLinearSolver(linSolver);
  optimizer.SetMaxIter(maxIPMiters);
  optimizer.Mult(x0, xf);

  GridFunction d_gf(Vh);
  GridFunction s_gf(Vh);

  d_gf = xf.GetBlock(0);
  s_gf = xf.GetBlock(1);

  FunctionCoefficient dm_fc(dmanufacturedFun); // pseudo-manufactured solution
  GridFunction dm_gf(Vh);
  dm_gf.ProjectCoefficient(dm_fc);
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

