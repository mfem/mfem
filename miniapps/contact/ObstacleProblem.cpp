//                         Obstacle Problem
//
//
// Compile with: make ParObstacleProblem
//
// Sample runs: mpirun -np 4 ./ParObstacleProblem
//
//
// Description: This example code demonstrates the use of MFEM to solve the
//              bound-constrained energy minimization problem
//
//                      minimize (||∇u||² + ||u||²) subject to u ≥ ϕ in H¹.

#include "mfem.hpp"
#include "ipsolver/IPsolver.hpp"
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
   int linSolver = 2;
   int maxIPMiters = 30;
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
     args.PrintOptions(cout);
   }

   const char *meshFile = "../../data/inline-quad.mesh";
   Mesh mesh(meshFile, 1, 1);
   int dim = mesh.Dimension(); // geometric dimension of the meshed domain
   {
      for (int l = 0; l < ref_levels; l++)
      {
         mesh.UniformRefinement();
      }
   }

   FiniteElementCollection *fec = new H1_FECollection(FEorder, dim);
   FiniteElementSpace   *Vh  = new FiniteElementSpace(&mesh, fec);

   ObstacleProblem problem(Vh,&fRhs, &obstacle);
  
   int dimD = problem.GetDimU();
   Vector x0(dimD); x0 = 100.0;
   Vector xf(dimD); xf = 0.0;

   InteriorPointSolver optimizer(&problem); 
   optimizer.SetTol(1.e-8);
   optimizer.SetLinearSolveTol(1.e-10);
   optimizer.SetLinearSolver(linSolver);
   optimizer.SetMaxIter(maxIPMiters);
   optimizer.Mult(x0, xf);

   //ParGridFunction d_gf(Vh);

   //d_gf.SetFromTrueDofs(xf);


   //FunctionCoefficient dm_fc(dmanufacturedFun); // manufactured solution
   //ParGridFunction dm_gf(Vh);
   //dm_gf.ProjectCoefficient(dm_fc);
   //ParaViewDataCollection paraview_dc("BarrierProblemSolution", &pmesh);
   //paraview_dc.SetPrefixPath("ParaView");
   //paraview_dc.SetLevelsOfDetail(FEorder);
   //paraview_dc.SetDataFormat(VTKFormat::BINARY);
   //paraview_dc.SetHighOrderOutput(true);
   //paraview_dc.SetCycle(0);
   //paraview_dc.SetTime(0.0);
   //paraview_dc.RegisterField("d(x) (numerical)", &d_gf);
   //paraview_dc.RegisterField("d(x) (pseudo-manufactured)", &dm_gf);
   //paraview_dc.Save();

   delete Vh;
   delete fec;
   return 0;
}


double dmanufacturedFun(const Vector &x)
{
  return cos(2*M_PI*x(0)) + 0.2 - 2.0*(pow(x(0),3) - 1.5*pow(x(0),2));
}

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
