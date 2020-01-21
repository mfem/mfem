#include "hypsys.hpp"
#include "lib/evolve.cpp"
#include "lib/tools.cpp"
#include "apps/advection.hpp"
// #include "apps/swe.hpp"

int main(int argc, char *argv[])
{
	Configuration config;
   config.ProblemNum = 0;
   config.ConfigNum = 1;
   const char *MeshFile = "data/unstr.mesh";
   int refinements = 1;
   config.order = 3;
   config.tFinal = 1.;
   config.dt = 0.001;
   config.odeSolverType = 3;
   config.VisSteps = 100;
	config.scheme = Standard;
	
   config.precision = 8;
   cout.precision(config.precision);

   OptionsParser args(argc, argv);
   args.AddOption(&config.ProblemNum, "-p", "--problem",
                  "Hyperbolic system of equations to solve.");
   args.AddOption(&config.ConfigNum, "-c", "--configuration",
                  "Problem setup to use.");
   args.AddOption(&MeshFile, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&refinements, "-r", "--refine",
                  "Number of times to refine the mesh uniformly.");
   args.AddOption(&config.order, "-o", "--order",
                  "Order (polynomial degree) of the finite element space.");
   args.AddOption(&config.tFinal, "-tf", "--t-final",
                  "Final time; start time is 0.");
   args.AddOption(&config.dt, "-dt", "--time-step",
                  "Time step.");
   args.AddOption(&config.odeSolverType, "-s", "--ode-solver",
                  "ODE solver: 1 - Forward Euler,\n\t"
                  "            2 - RK2 SSP, 3 - RK3 SSP.");
   args.AddOption(&config.VisSteps, "-vs", "--visualization-steps",
                  "Visualize every n-th timestep.");
	args.AddOption((int*)(&config.scheme), "-e", "--EvolutionScheme",
                  "Scheme: 0 - Standard Finite Element Approximation,\n\t"
                  "        1 - Monolithic Convex Limiting.");

   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return -1;
   }
   args.PrintOptions(cout);
   
   ODESolver *odeSolver = NULL;
   switch (config.odeSolverType)
   {
      case 1: odeSolver = new ForwardEulerSolver; break;
      case 2: odeSolver = new RK2Solver(1.0); break;
      case 3: odeSolver = new RK3SSPSolver; break;
      default:
         cout << "Unknown ODE solver type: " << config.odeSolverType << endl;
         return -1;
   }

   Mesh mesh(MeshFile, 1, 1);
   const int dim = mesh.Dimension();

   for (int lev = 0; lev < refinements; lev++)
   {
      mesh.UniformRefinement();
   }
   if (mesh.NURBSext)
   {
      mesh.SetCurvature(max(config.order, 1));
   }
   
   mesh.GetBoundingBox(config.bbMin, config.bbMax, max(config.order, 1));
   
   // Create Bernstein Finite Element Space.
   const int btype = BasisType::Positive;
   L2_FECollection fec(config.order, dim, btype);
   FiniteElementSpace fes(&mesh, &fec);
	cout << "Number of unknowns: " << fes.GetVSize() << endl;

	HyperbolicSystem *hyp;
	switch (config.ProblemNum)
	{
		case 0: { hyp =  new Advection(&fes, config); break; }
		default:
			cout << "Unknown hyperbolic system: " << config.ProblemNum << '\n';
         return -1;
   }
   
   if (config.odeSolverType != 1 && hyp->SteadyState)
		MFEM_WARNING("You should use forward Euler for pseudo time stepping.");

	double InitialMass = hyp->LumpedMassMat * hyp->u;
	
	// Visualization with GLVis, VisIt is currently not supported.
	{
      ofstream omesh("grid.mesh");
      omesh.precision(config.precision);
      mesh.Print(omesh);
      ofstream osol("initial.gf");
      osol.precision(config.precision);
      hyp->u.Save(osol);
   }
	
	socketstream sout;
   char vishost[] = "localhost";
   int  visport   = 19916;
	bool VectorOutput = false; // TODO
   {
		VisualizeField(sout, vishost, visport, hyp->u, VectorOutput);
   }
   
   odeSolver->Init(*hyp);

	bool done = false;
	double dt, res, tol = 1.e-12;
   for (int ti = 0; !done; )
   {
      dt = min(config.dt, config.tFinal - hyp->t);
      odeSolver->Step(hyp->u, hyp->t, dt);
      ti++;

      done = (hyp->t >= config.tFinal - 1.e-8*config.dt);
		
		if (hyp->SteadyState)
		{
			res = hyp->ConvergenceCheck(dt, tol);
			if (res < tol)
				done = true;
		}

      if (done || ti % config.VisSteps == 0)
      {
			if (hyp->SteadyState)
			{
				cout << "time step: " << ti << ", time: " << hyp->t << 
					", residual: " << res << endl;
			}
			else
			{
				cout << "time step: " << ti << ", time: " << hyp->t << endl;
			}
         VisualizeField(sout, vishost, visport, hyp->u, VectorOutput);
      }
   }
   
   double DomainSize = hyp->LumpedMassMat.Sum();
	cout << "Difference in solution mass: "
		  << abs(InitialMass - hyp->LumpedMassMat * hyp->u) / DomainSize << endl;
	
	if (hyp->SolutionKnown)
   {
		Array<double> errors;
		hyp->ComputeErrors(errors, DomainSize);
		hyp->WriteErrors(errors);
	}

   {
      ofstream osol("final.gf");
      osol.precision(config.precision);
      hyp->u.Save(osol);
   }

	delete odeSolver;
	delete hyp;
   return 0;
}
