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
#include "ParProblems.hpp"
#include "ParIPsolver.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

double dmanufacturedFun(const Vector &);
double fRhs(const Vector &);

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
     if(Mpi::Root())
     {  
       args.PrintOptions(cout);
     }
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

   ParMesh pmesh(MPI_COMM_WORLD, mesh);

   FiniteElementCollection *fec = new H1_FECollection(FEorder, dim);
   ParFiniteElementSpace   *Vh  = new ParFiniteElementSpace(&pmesh, fec);

   ParObstacleProblem problem(Vh,Vh,&fRhs);
  
   int dimD = problem.GetDimD();
   Vector x0(dimD); x0 = 100.0;
   Vector xf(dimD); xf = 0.0;

   ParInteriorPointSolver optimizer(&problem); 
   optimizer.SetTol(1.e-8);
   optimizer.SetLinearSolveTol(1.e-10);
   optimizer.SetLinearSolver(linSolver);
   optimizer.SetMaxIter(maxIPMiters);
   optimizer.Mult(x0, xf);

   ParGridFunction d_gf(Vh);

   d_gf.SetFromTrueDofs(xf);


   FunctionCoefficient dm_fc(dmanufacturedFun); // manufactured solution
   ParGridFunction dm_gf(Vh);
   dm_gf.ProjectCoefficient(dm_fc);

   char vishost[] = "localhost";
   int  visport   = 19916;
   socketstream exact_sock(vishost, visport);
   exact_sock.precision(8);
   exact_sock << "parallel " << num_procs << " " << myid << "\n";
   exact_sock << "solution\n" << pmesh << dm_gf
              << "window_title 'Manufactured solution'" << flush;

   socketstream sol_sock(vishost, visport);
   sol_sock.precision(8);
   sol_sock << "parallel " << num_procs << " " << myid << "\n";
   sol_sock << "solution\n" << pmesh << d_gf 
            << "window_title 'Numerical solution'" << flush;

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
