//                          Spherical Obstacle Problem
//
//
// Compile with: make SphericalobstacleProblem
//
// Sample runs: ./SphericalobstacleProblem
//
//
// Description: This example code demonstrates the use of MFEM to solve the
//              bound-constrained energy minimization problem
//
//                      minimize ||∇u||² subject to u ≥ ϕ in H¹₀.


#include "mfem.hpp"
#include "Problems.hpp"
#include "IPsolver.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;


double fRhs(const Vector &);
double spherical_obstacle(const Vector &);
double exact_solution_obstacle(const Vector &);

int main(int argc, char *argv[])
{
  int FEorder = 1;     // finite element order
  int linSolver = 0;   // linear solver 0 (direct), 1 (iterative) or 2 (iterative)
  int maxIPMiters = 30;
  bool iAmRoot = true;
  int ref_levels = 3;
  
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

  const char *meshFile = "../../data/disk.mesh";
  Mesh *mesh = new Mesh(meshFile, 1, 1);
  int dim = mesh->Dimension(); // geometric dimension of the domain
  {
     for (int l = 0; l < ref_levels; l++)
     {
        mesh->UniformRefinement();
     }
  }
  double h_min, h_max, kappa_min, kappa_max;
  mesh->GetCharacteristics(h_min, h_max, kappa_min, kappa_max);

  FiniteElementCollection *fec = new H1_FECollection(FEorder, dim);
  FiniteElementSpace      *Vh  = new FiniteElementSpace(mesh, fec);
  Array<int> ess_tdof_list;
  if (mesh->bdr_attributes.Size())
  {
     Array<int> ess_bdr(mesh->bdr_attributes.Max());
     ess_bdr = 1;
     Vh->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
  }
  
  int dimD = Vh->GetTrueVSize();
  Vector x0(dimD); x0 = 0.0;
  Vector xf(dimD); xf = 0.0;
  
  ObstacleProblem problem(Vh, x0, &fRhs, &spherical_obstacle, ess_tdof_list);
  InteriorPointSolver optimizer(&problem); 
  
  optimizer.SetTol(1.e-7);
  optimizer.SetLinearSolver(linSolver);
  optimizer.SetMaxIter(maxIPMiters);
  optimizer.Mult(x0, xf);
  
  
  double Einitial = problem.E(x0);
  double Efinal = problem.E(xf);
  cout << "Energy objective at initial point = " << Einitial << endl;
  cout << "Energy objective at optimizer = " << Efinal << endl;

  GridFunction d_gf(Vh);
  d_gf = xf;

  FunctionCoefficient dtrue_fc(exact_solution_obstacle); // exact solution
  GridFunction dtrue_gf(Vh);
  dtrue_gf.ProjectCoefficient(dtrue_fc);
  
  ParaViewDataCollection paraview_dc("BarrierProblemSolution", mesh);
  paraview_dc.SetPrefixPath("ParaView");
  paraview_dc.SetLevelsOfDetail(FEorder);
  paraview_dc.SetDataFormat(VTKFormat::BINARY);
  paraview_dc.SetHighOrderOutput(true);
  paraview_dc.SetCycle(0);
  paraview_dc.SetTime(0.0);
  paraview_dc.RegisterField("d(x) (numerical)", &d_gf);
  paraview_dc.RegisterField("d(x) (true)", &dtrue_gf);
  paraview_dc.Save();
  
  FunctionCoefficient exact_coef(exact_solution_obstacle);
  double L2_error = d_gf.ComputeL2Error(exact_coef);
  cout << "||u - u_true||_L^2(Omega) = " << L2_error << ", hmax = " << h_max << ", hmin = " << h_min << endl;
  
  delete Vh;
  delete fec;
  delete mesh;
  return 0;
}


double fRhs(const Vector &x)
{
  return 0.;
}


double spherical_obstacle(const Vector &pt)
{
   double x = pt(0), y = pt(1);
   double r = sqrt(x*x + y*y);
   double r0 = 0.5;
   double beta = 0.9;

   double b = r0*beta;
   double tmp = sqrt(r0*r0 - b*b);
   double B = tmp + b*b/tmp;
   double C = -b/tmp;

   if (r > b)
   {
      return B + r * C;
   }
   else
   {
      return sqrt(r0*r0 - r*r);
   }
}

double exact_solution_obstacle(const Vector &pt)
{
   double x = pt(0), y = pt(1);
   double r = sqrt(x*x + y*y);
   double r0 = 0.5;
   double a =  0.348982574111686;
   double A = -0.340129705945858;

   if (r > a)
   {
      return A * log(r);
   }
   else
   {
      return sqrt(r0*r0-r*r);
   }
} 

