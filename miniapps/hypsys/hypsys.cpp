#include <fstream>
#include <iostream>
#include "lib/hypsys.hpp"
#include "lib/dofs.hpp"
#include "apps/advection.hpp"
// #include "apps/swe.hpp"
#include "evolve.cpp"

using namespace mfem;

int main(int argc, char *argv[])
{
	Configuration config;
	
   config.ProblemNum = 0;
   config.ConfigNum = 0;
   const char *MeshFile = "data/periodic-square.mesh";
   int refinements = 2;
   config.order = 3;
   config.tEnd = 1.;
   config.dt = 0.002;
   config.odeSolverType = 3;
   config.VisSteps = 100;
	config.scheme = Standard;

   int precision = 8;
   cout.precision(precision);

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
                  "Order (degree) of the finite elements.");
   args.AddOption(&config.tEnd, "-tf", "--t-final",
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
         cout << "Unknown ODE solver type: " << config.odeSolverType << '\n';
         return -1;
   }

   Mesh mesh(MeshFile, 1, 1);
   int dim = mesh.Dimension();

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
			cout << "Unknown Hyperbolic system: " << config.ProblemNum << '\n';
         return -1;
   }
   
	GridFunction u(&fes);
	hyp->PreprocessProblem(&fes, u);
	
   // Compute the lumped mass matrix.
   Vector LumpedMassMat;
   BilinearForm ml(&fes);
   ml.AddDomainIntegrator(new LumpedIntegrator(new MassIntegrator));
   ml.Assemble();
   ml.Finalize();
   ml.SpMat().GetDiag(LumpedMassMat);
	const double InitialMass = LumpedMassMat * u;
	
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
      double dt = min(config.dt, config.tEnd - hyp->t);
      odeSolver->Step(u, hyp->t, dt);
      ti++;

      done = (hyp->t >= config.tEnd - 1.e-8*config.dt);

      if (done || ti % config.VisSteps == 0)
      {
         cout << "time step: " << ti << ", time: " << hyp->t << endl;
         if (visualization)
         {
            sout << "solution\n" << mesh << u << flush;
         }
      }
   }
   
   cout << "Difference in solution mass: " <<
		abs(InitialMass - LumpedMassMat * u) << endl;
   
   {
      ofstream osol("sol-final.gf");
      osol.precision(precision);
      u.Save(osol);
   }
	
	delete odeSolver;
	delete hyp;
   return 0;
}

// TODO restructure code
const IntegrationRule* GetElementIntegrationRule(FiniteElementSpace *fes)
{
	const FiniteElement *el = fes->GetFE(0);
	ElementTransformation *eltrans = fes->GetElementTransformation(0);
	int order = eltrans->OrderGrad(el) + eltrans->Order() + el->GetOrder();
   return &IntRules.Get(el->GetGeomType(), order);
}

// Appropriate quadrature rule for faces according to DGTraceIntegrator.
const IntegrationRule *GetFaceIntegrationRule(FiniteElementSpace *fes)
{
   int i, order;
   // Use the first mesh face and element as indicator.
   const FaceElementTransformations *Trans =
      fes->GetMesh()->GetFaceElementTransformations(0);
   const FiniteElement *el = fes->GetFE(0);

   if (Trans->Elem2No >= 0)
   {
      order = min(Trans->Elem1->OrderW(), Trans->Elem2->OrderW()) + 2*el->GetOrder();
   }
   else
   {
      order = Trans->Elem1->OrderW() + 2*el->GetOrder();
   }
   if (el->Space() == FunctionSpace::Pk)
   {
      order++;
   }
   return &IntRules.Get(Trans->FaceGeom, order);
}

void HyperbolicSystem::PreprocessProblem(FiniteElementSpace *fes, GridFunction &u)
{ }

void HyperbolicSystem::EvaluateSolution(const Vector &u, Vector &v,
													 const int QuadNum) const
{
	int nd = ShapeEval.Height();
	Vector shape(nd);// TODO optimize.
	ShapeEval.GetColumn(QuadNum, shape);
	v(0) = u * shape; // TODO Vector valued soultion.
}

void HyperbolicSystem::EvaluateSolution(const Vector &u, Vector &v,
													 const int QuadNum,
													 const int BdrNum) const
{
	v(0) = 0.; // TODO Vector valued soultion.
	for (int j = 0; j < dofs->NumFaceDofs; j++)
	{
		v(0) += u(dofs->BdrDofs(j,BdrNum)) * ShapeEvalFace(BdrNum,j,QuadNum);
	}
}

void HyperbolicSystem::Mult(const Vector &x, Vector &y) const
{
	switch (scheme)
	{
		case 0: // Standard Finite Element Approximation.
		{
			EvolveStandard(x, y);
			break;
		}
		case 1: // Monolithic Convex Limiting.
		{
			EvolveMCL(x, y);
			break;
		}
		default:
		{
			MFEM_ABORT("Unknown Evolution Scheme.");
		}
	}
}
