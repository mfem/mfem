#include "lib/lib.hpp"
#include "fe_evol/lib.hpp"
#include "apps/lib.hpp"

int main(int argc, char *argv[])
{
   Configuration config;
   int ProblemNum = 0;
   config.ConfigNum = 1;
   int VisSteps = 100;
   config.tFinal = 1.;
   int odeSolverType = 3;
   double dt = 0.001;
   const char *MeshFile = "data/unstr.mesh";
   int order = 3;
   int refinements = 1;
   EvolutionScheme scheme = MonolithicConvexLimiting;
   const char *OutputDir = "."; // Directory has to exist to produce output.
   bool TransOutput = false; // Use this to produce output for videos.

   int precision = 8;
   cout.precision(precision);

   OptionsParser args(argc, argv);
   args.AddOption(&ProblemNum, "-p", "--problem",
                  "Hyperbolic system of equations to solve.");
   args.AddOption(&config.ConfigNum, "-c", "--configuration",
                  "Problem setup to use.");
   args.AddOption(&VisSteps, "-vs", "--visualization-steps",
                  "Visualize every n-th timestep.");
   args.AddOption(&config.tFinal, "-tf", "--t-final",
                  "Final time; start time is 0.");
   args.AddOption(&odeSolverType, "-s", "--ode-solver",
                  "ODE solver: 0 - RK6 solver, 1 - Forward Euler,\n\t"
                  "            2 - RK2 SSP, 3 - RK3 SSP.");
   args.AddOption(&dt, "-dt", "--time-step", "Time step.");
   args.AddOption(&MeshFile, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Order (polynomial degree) of the finite element space.");
   args.AddOption(&refinements, "-r", "--refine",
                  "Number of times to refine the mesh uniformly.");
   args.AddOption((int*)(&scheme), "-e", "--EvolutionScheme",
                  "Scheme: 0 - Galerkin Finite Element Approximation,\n\t"
                  "        1 - Monolithic Convex Limiting.");
   args.AddOption(&OutputDir, "-out", "--output", "Output directory.");
   args.AddOption(&TransOutput, "-t", "--transitional-output", "-no-t",
                  "--transitional-output", "Print transitional output files.");

   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return -1;
   }
   args.PrintOptions(cout);

   if (order == 0)
   {
      scheme = Galerkin;
   }

   ODESolver *odeSolver = NULL;
   switch (odeSolverType)
   {
      case 0: odeSolver = new RK6Solver; break;
      case 1: odeSolver = new ForwardEulerSolver; break;
      case 2: odeSolver = new RK2Solver(1.0); break;
      case 3: odeSolver = new RK3SSPSolver; break;
      default:
         cout << "Unknown ODE solver type: " << odeSolverType << endl;
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
      mesh.SetCurvature(max(order, 1));
   }

   mesh.GetBoundingBox(config.bbMin, config.bbMax, max(order, 1));

   int NumEq;
   switch (ProblemNum)
   {
      case 0:
      case 1:
      case 2:
      case 3: NumEq = 1; break;
      case 4: NumEq = 1 + dim; break;
      case 5: NumEq = 2 + dim; break;
      default:
         cout << "Unknown hyperbolic system: " << ProblemNum << endl;
         delete odeSolver;
         return -1;
   }

   // Create Bernstein Finite Element Space.
   const int btype = BasisType::Positive;
   L2_FECollection fec(order, dim, btype);
   FiniteElementSpace fes(&mesh, &fec);
   FiniteElementSpace vfes(&mesh, &fec, NumEq, Ordering::byNODES);

   Array<int> offsets(NumEq + 1);
   for (int k = 0; k <= NumEq; k++) { offsets[k] = k * fes.GetNDofs(); }
   BlockVector u_block(offsets);

   const int ProblemSize = vfes.GetVSize();
   cout << "Number of unknowns: " << ProblemSize << endl;

   DofInfo dofs(&fes);

   bool NodalQuadRule = false;
   if (scheme == MonolithicConvexLimiting)
   {
      NodalQuadRule = true;
   }

   HyperbolicSystem *hyp;
   switch (ProblemNum)
   {
      case 0: { hyp = new Advection(&vfes, u_block, config, NodalQuadRule); break; }
      case 1: { hyp = new Burgers(&vfes, u_block, config); break; }
      case 2: { hyp = new KPP(&vfes, u_block, config); break; }
      case 3: { hyp = new BuckleyLeverett(&vfes, u_block, config); break; }
      case 4: { hyp = new ShallowWater(&vfes, u_block, config); break; }
      case 5: { hyp = new Euler(&vfes, u_block, config); break; }
      default:
         return -1;
   }

   if (odeSolverType != 1 && hyp->SteadyState)
   {
      MFEM_WARNING("Better use forward Euler pseudo time stepping for steady state simulations.");
   }

   GridFunction u(&vfes, u_block);
   u = hyp->u0;

   // The main is variable is visualized, printed, and used to check for mass leaks or violation of maximum principles.
   GridFunction main(&fes, u_block.GetBlock(0));

   ostringstream MeshName, InitName;
   MeshName << OutputDir << "/grid.mesh";
   InitName << OutputDir << "/initial.gf";
   ofstream omesh(MeshName.str().c_str());
   omesh.precision(precision);
   mesh.Print(omesh);
   ofstream initial(InitName.str().c_str());
   initial.precision(precision);
   main.Save(initial);

   socketstream sout;
   char vishost[] = "localhost";
   int visport = 19916;
   VisualizeField(sout, vishost, visport, hyp->ProblemName, main, hyp->glvis_scale);

   FE_Evolution *evol;
   switch (scheme)
   {
      case Galerkin: { evol = new GalerkinEvolution(&vfes, hyp, dofs); break; }
      case MonolithicConvexLimiting: { evol = new MCL_Evolution(&vfes, hyp, dofs, dt); break; }
      default:
         MFEM_ABORT("Unknown evolution scheme");
   }

   Vector LumpedMassMat(fes.GetVSize());
   BilinearForm ml(&fes);
   ml.AddDomainIntegrator(new LumpedIntegrator(new MassIntegrator()));
   ml.Assemble();
   ml.Finalize();
   ml.SpMat().GetDiag(LumpedMassMat);

   double InitialMass = LumpedMassMat * main;

   odeSolver->Init(*evol);
   if (hyp->SteadyState)
   {
      evol->uOld.SetSize(ProblemSize);
      evol->uOld = 0.;
   }

   int TransStep = 0;
   double dtLast, res, t = 0., tol = 1.e-12;
   bool done = t >= config.tFinal;
   tic_toc.Clear();
   tic_toc.Start();
   cout << "Preprocessing done. Entering time stepping loop.\n";

   for (int ti = 0; !done;)
   {
      dtLast = min(dt, config.tFinal - t);
      odeSolver->Step(u, t, dtLast);
      ti++;

      done = (t >= config.tFinal - 1.e-8 * dt);

      if (hyp->SteadyState)
      {
         res = evol->ConvergenceCheck(dt, u);
         if (res < tol)
         {
            done = true;
            u = evol->uOld;
         }
      }

      if (done || ti % VisSteps == 0)
      {
         if (hyp->SteadyState)
         {
            cout << "time step: " << ti << ", time: " << t <<
                 ", residual: " << res << endl;
         }
         else
         {
            cout << "time step: " << ti << ", time: " << t << endl;
         }

         VisualizeField(sout, vishost, visport, hyp->ProblemName, main, hyp->glvis_scale);
         if (TransOutput)
         {
            ostringstream TransName;
            TransName << OutputDir << "/trans" << TransStep << ".gf";
            ofstream trans(TransName.str().c_str());
            trans.precision(precision);
            main.Save(trans);
            TransStep++;
         }
      }
   }

   tic_toc.Stop();
   cout << "Time stepping loop done in " << tic_toc.RealTime() << " seconds.\n\n";

   double DomainSize = LumpedMassMat.Sum();
   if (hyp->SolutionKnown)
   {
      Array<double> errors;
      hyp->ComputeErrors(errors, u, DomainSize, t);
      cout << "L1 error:                    " << errors[0] << endl;
      hyp->WriteErrors(errors);
   }

   cout << "Min of primary field:        " << main.Min() << endl
        << "Max of primary field:        " << main.Max() << endl
        << "Difference in solution mass: "
        << abs(InitialMass - LumpedMassMat * main) / DomainSize << "\n\n";

   ostringstream FinalName;
   FinalName << OutputDir << "/ultimate.gf";
   ofstream ultimate(FinalName.str().c_str());
   ultimate.precision(precision);
   main.Save(ultimate);

   delete evol;
   delete hyp;
   delete odeSolver;
   return 0;
}
