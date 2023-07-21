//                       Spherical Obstacle Problem
//
//
// Compile with: make ParSphericalObstacleProblem
//
// Sample runs: mpirun -np 4 ./ParSphericalObstacleProblem -linSolver 0
//              mpirun -np 4 ./ParSphericalObstacleProblem -linSolver 1
//              mpirun -np 4 ./ParSphericalObstacleProblem -linSolver 2
//
//
// Description: This example code demonstrates the use of MFEM to solve the
//              bound-constrained energy minimization problem
//
//                      minimize ||∇u||² subject to u ≥ ϕ in H¹₀.

#include "mfem.hpp"
#include "ParProblems.hpp"
#include "ParIPsolver.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

double fRhs(const Vector &);
double spherical_obstacle(const Vector &);
double exact_solution_obstacle(const Vector &);


int main(int argc, char *argv[])
{
  // Initialize MPI
   Mpi::Init();
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();

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
     if(myid == 0)
     {  
       args.PrintOptions(cout);
     }
   }

   const char *meshFile = "../../data/disk.mesh";
   Mesh mesh(meshFile, 1, 1);
   int dim = mesh.Dimension(); // geometric dimension of the meshed domain
   {
      for (int l = 0; l < ref_levels; l++)
      {
         mesh.UniformRefinement();
      }
   }

   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   
   FiniteElementCollection *fec = new H1_FECollection(FEorder, dim);
   ParFiniteElementSpace   *Vh  = new ParFiniteElementSpace(&pmesh, fec);
   Array<int> boundary_dofs;
   Vh->GetBoundaryTrueDofs(boundary_dofs);
   int dimD = Vh->GetTrueVSize();
   Vector xDC(dimD); xDC = 0.0;

   ParObstacleProblem problem(Vh, Vh, &fRhs, &spherical_obstacle, boundary_dofs, xDC);
   Vector x0(dimD); x0.Set(1.0, xDC);
   Vector xf(dimD); xf = 0.0;

   ParInteriorPointSolver optimizer(&problem); 
   optimizer.SetTol(1.e-7);
   optimizer.SetLinearSolveTol(1.e-10);
   optimizer.SetLinearSolver(linSolver);
   optimizer.SetMaxIter(maxIPMiters);
   optimizer.Mult(x0, xf);

   ParGridFunction d_gf(Vh);

   d_gf.SetFromTrueDofs(xf);


   FunctionCoefficient dtrue_fc(exact_solution_obstacle); // analytic solution
   ParGridFunction dtrue_gf(Vh);
   dtrue_gf.ProjectCoefficient(dtrue_fc);

   double L2error = d_gf.ComputeL2Error(dtrue_fc);
   if (myid == 0)
   {
     cout << "\n|| u_h - u ||_{L^2} = " << L2error << '\n' << endl;
   } 
   
   ParaViewDataCollection paraview_dc("SphericalObstacleProblem", &pmesh);
   paraview_dc.SetPrefixPath("ParaView");
   paraview_dc.SetLevelsOfDetail(FEorder);
   paraview_dc.SetDataFormat(VTKFormat::BINARY);
   paraview_dc.SetHighOrderOutput(true);
   paraview_dc.SetCycle(0);
   paraview_dc.SetTime(0.0);
   paraview_dc.RegisterField("u(x,y) (analytic)", &dtrue_gf);
   paraview_dc.RegisterField("u(x,y) (numerical)", &d_gf);
   paraview_dc.Save();

   delete Vh;
   delete fec;
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

