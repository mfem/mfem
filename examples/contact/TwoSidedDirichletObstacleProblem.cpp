#include "mfem.hpp"
#include "Problems.hpp"
#include "IPsolver.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;



double fRhs(const Vector &pt);
double obstaclel(const Vector &pt);
double obstacleu(const Vector &pt);
double dmanufacturedFun(const Vector &pt);

int main(int argc, char *argv[])
{
  int FEorder = 1; // order of the finite elements
  int linSolver = 0;
  int maxIPMiters = 30;
  bool iAmRoot = true;
  int ref_levels = 1; 
  OptionsParser args(argc, argv);
  args.AddOption(&FEorder, "-o", "--order",\
		  "Order of the finite elements.");
  args.AddOption(&linSolver, "-linSolver", "--linearSolver", \
       "IP-Newton linear system solution strategy.");
  args.AddOption(&maxIPMiters, "-IPMiters", "--IPMiters",\
		  "Maximum number of IPM iterations");
  args.AddOption(&ref_levels, "-r", "--mesh_refinement", \
		  "Mesh Refinement");
  
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
  for (int l = 0; l < ref_levels; l++)
  {
      mesh->UniformRefinement();
  }

  FiniteElementCollection *fec = new H1_FECollection(FEorder, dim);
  FiniteElementSpace      *Vh  = new FiniteElementSpace(mesh, fec);
  Array<int> ess_tdof_list;
  if (mesh->bdr_attributes.Size())
  {
     Array<int> ess_bdr(mesh->bdr_attributes.Max());
     ess_bdr = 1;
     Vh->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
  }
  
  double DC_val = 0.0;
  int dimD = Vh->GetTrueVSize();

  Vector x0(dimD); x0 = DC_val;
  Vector xf(dimD); xf = 0.0;

  ObstacleProblem problem(Vh, x0, &fRhs, &obstaclel, &obstacleu, ess_tdof_list);

  InteriorPointSolver optimizer(&problem); 
  optimizer.SetTol(1.e-7);
  optimizer.SetLinearSolver(linSolver);
  optimizer.SetMaxIter(maxIPMiters);
  optimizer.Mult(x0, xf);
  

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



double dmanufacturedFun(const Vector &pt)
{
  double alpha = 16.5;
  return sin(M_PI * pt(1)) * (sin(M_PI * pt(0)) - alpha * pow(pt(0) * (1. - pt(0)), 2));
}


// f(x) forcing term... which enters the objective energy functional
// E(d) = 0.5 d^T K d - f^T d, where f is a discrete vector representation
// of f(x). f(x) is such that in the absence of bound-constraints then
// the solution of the optimization problem satisfies the PDE
// -div(grad(d)) + d = f + homogeneous Neumann conditions on the unit interval,
// for d(x) = cos(2 \pi x) + a0 + a3 (x^3 - 1.5 x^2), a2 = 0.2, a3 = -2

double fRhs(const Vector &pt)
{
  double alpha = 16.5;
  double fx;
  fx = pow(M_PI, 2) * sin(M_PI * pt(0));
  fx += alpha * (2. * pow(pt(0), 2) + 2. * pow(1.-pt(0), 2) - 8. * pt(0) * (1.-pt(0)));
  fx += pow(M_PI, 2) * sin(M_PI * pt(0)) * dmanufacturedFun(pt);
  fx *= sin(M_PI * pt(1));
  return fx;
}

double obstaclel(const Vector &pt)
{
  return 0.0;
}

double obstacleu(const Vector &pt)
{
  return 0.08;
}
