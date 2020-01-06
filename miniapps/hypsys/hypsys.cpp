#include <fstream>
#include <iostream>
#include "lib/hypsys.hpp"
#include "apps/advection.hpp"
// #include "apps/swe.hpp"

int dim;

int main(int argc, char *argv[])
{
   int ProblemNum = 0;
   int ConfigNum = 0;
   const char *MeshFile = "data/inline-quad.mesh";
   int refinements = 2;
   int order = 3;
   double tEnd = 1.;
   double dt = 0.002;
   int odeSolverType = 3;
   int VisSteps = 100;

   int precision = 8;
   cout.precision(precision);

   OptionsParser args(argc, argv);
   args.AddOption(&ProblemNum, "-p", "--problem",
                  "Hyperbolic system of equations to solve.");
   args.AddOption(&ConfigNum, "-c", "--configuration",
                  "Problem setup to use.");
   args.AddOption(&MeshFile, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&refinements, "-r", "--refine",
                  "Number of times to refine the mesh uniformly.");
   args.AddOption(&order, "-o", "--order",
                  "Order (degree) of the finite elements.");
   args.AddOption(&tEnd, "-tf", "--t-final",
                  "Final time; start time is 0.");
   args.AddOption(&dt, "-dt", "--time-step",
                  "Time step.");
   args.AddOption(&odeSolverType, "-s", "--ode-solver",
                  "ODE solver: 1 - Forward Euler,\n\t"
                  "            2 - RK2 SSP, 3 - RK3 SSP.");
   args.AddOption(&VisSteps, "-vs", "--visualization-steps",
                  "Visualize every n-th timestep.");

   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return -1;
   }
   
   ODESolver *odeSolver = NULL;
   switch (odeSolverType)
   {
      case 1: odeSolver = new ForwardEulerSolver; break;
      case 2: odeSolver = new RK2Solver(1.0); break;
      case 3: odeSolver = new RK3SSPSolver; break;
      default:
         cout << "Unknown ODE solver type: " << odeSolverType << '\n';
         return -1;
   }

   Mesh mesh(MeshFile, 1, 1);
   dim = mesh.Dimension();

   for (int lev = 0; lev < refinements; lev++)
   {
      mesh.UniformRefinement();
   }
   if (mesh.NURBSext)
   {
      mesh.SetCurvature(max(order, 1));
   }
   
   Vector bbMin, bbMax;
   mesh.GetBoundingBox(bbMin, bbMax, max(order, 1));
   
   // Create Bernstein Finite Element Space.
   const int btype = BasisType::Positive;
   DG_FECollection fec(order, dim, btype);
   FiniteElementSpace fes(&mesh, &fec);
	cout << "Number of unknowns: " << fes.GetVSize() << endl;
	return 0;
	HyperbolicSystem *hyp = NULL;
	switch (ProblemNum)
	{
		case 0: { hyp =  new Advection(&fes, ConfigNum, tEnd, bbMin, bbMax); break; }
		default:
			cout << "Unknown Hyperbolic system: " << ProblemNum << '\n';
         return -1;
   }
	
	GridFunction u(&fes);
	hyp->PreprocessProblem(&fes, u);
	
	// Visualization with GLVis, VisIt is currently not supported.
	{
      ofstream omesh("hypsys.mesh");
      omesh.precision(precision);
      mesh.Print(omesh);
      ofstream osol("sol-init.gf");
      osol.precision(precision);
      u.Save(osol);
   }
	
	socketstream sout;
	bool visualization = true;
	{
      char vishost[] = "localhost";
      int  visport   = 19916;
      sout.open(vishost, visport);
      if (!sout)
      {
         cout << "Unable to connect to GLVis server at "
              << vishost << ':' << visport << endl;
         visualization = false;
         cout << "GLVis visualization disabled.\n";
      }
      else
      {
         sout.precision(precision);
         sout << "solution\n" << mesh << u;
         sout << "pause\n";
         sout << flush;
         cout << "GLVis visualization paused."
              << " Press space (in the GLVis window) to resume it.\n";
      }
   }
   
   hyp->t = 0.;
   odeSolver->Init(*hyp);
	
	bool done = false;
   for (int ti = 0; !done; )
   {
      double dtReal = min(dt, tEnd - hyp->t);
      odeSolver->Step(u, hyp->t, dtReal);
      ti++;

      done = (hyp->t >= tEnd - 1.e-8*dt);

      if (done || ti % VisSteps == 0)
      {
         cout << "time step: " << ti << ", time: " << hyp->t << endl;
         if (visualization)
         {
            sout << "solution\n" << mesh << u << flush;
         }
      }
   }
   
   {
      ofstream osol("sol-final.gf");
      osol.precision(precision);
      u.Save(osol);
   }
	
	delete odeSolver;
	delete hyp;
   return 0;
}

// void HyperbolicSystem::Mult(const Vector &x, Vector &y) const
// {
// 	
// }

