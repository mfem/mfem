#include "lib/lib.hpp"
#include "fe_evol/plib.hpp"
#include "apps/lib.hpp"

int main(int argc, char *argv[])
{
   MPI_Session mpi(argc, argv);
   const int myid = mpi.WorldRank();

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
   int prefinements = 0;
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
   args.AddOption(&VisSteps, "-vf", "--visualization-frequency",
                  "Visualize every n-th timestep.");
   args.AddOption(&config.tFinal, "-tf", "--final-time",
                  "Final time; start time is 0.");
   args.AddOption(&odeSolverType, "-s", "--ode-solver",
                  "ODE solver: 0 - RK6 solver, 1 - Forward Euler,\n\t"
                  "            2 - RK2 SSP, 3 - RK3 SSP.");
   args.AddOption(&dt, "-dt", "--time-step", "Time step.");
   args.AddOption(&MeshFile, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Order (polynomial degree) of the finite element space.");
   args.AddOption(&refinements, "-r", "--serial-refinements",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&prefinements, "-pr", "--parallel-refinements",
                  "Number of times to refine the mesh uniformly in parallel.");
   args.AddOption((int*)(&scheme), "-e", "--evolution-scheme",
                  "Scheme: 0 - Galerkin Finite Element Approximation,\n\t"
                  "        1 - Monolithic Convex Limiting.");
   args.AddOption(&OutputDir, "-out", "--output-directory", "Output directory.");
   args.AddOption(&TransOutput, "-t", "--transitional-output", "-no-t",
                  "--no-transitional-output", "Print transitional output files.");

   args.Parse();
   if (!args.Good())
   {
      if (myid == 0) { args.PrintUsage(cout); }
      return -1;
   }
   if (myid == 0) { args.PrintOptions(cout); }

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

   // Read the serial mesh from the given mesh file on all processors.
   Mesh *mesh = new Mesh(MeshFile, 1, 1);
   const int dim = mesh->Dimension();
   for (int lev = 0; lev < refinements; lev++)
   {
      mesh->UniformRefinement();
   }
   mesh->GetBoundingBox(config.bbMin, config.bbMax, max(order, 1));

   // Parallel partitioning of the mesh.
   ParMesh pmesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   for (int lev = 0; lev < prefinements; lev++)
   {
      pmesh.UniformRefinement();
   }

   if (pmesh.NURBSext)
   {
      pmesh.SetCurvature(max(order, 1));
   }
   MPI_Comm comm = pmesh.GetComm();

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
   ParFiniteElementSpace pfes(&pmesh, &fec);
   ParFiniteElementSpace vfes(&pmesh, &fec, NumEq, Ordering::byNODES);

   Array<int> offsets(NumEq + 1);
   for (int k = 0; k <= NumEq; k++) { offsets[k] = k * pfes.GetNDofs(); }
   BlockVector u_block(offsets);

   const int ProblemSize = vfes.GlobalVSize();
   if (myid == 0)
   {
      cout << "Number of unknowns: " << ProblemSize << endl;
   }

   ParDofInfo pdofs(&pfes);

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

   if (odeSolverType != 1 && hyp->SteadyState && myid == 0)
   {
      MFEM_WARNING("Better use forward Euler pseudo time stepping for steady state simulations.");
   }

   ParGridFunction u(&vfes, u_block);
   u = hyp->u0;

   // The main is variable is visualized, printed, and used to check for mass leaks or violation of maximum principles.
   ParGridFunction main(&pfes, u_block.GetBlock(0));

   ostringstream MeshName, InitName;
   MeshName << OutputDir << "/grid-mesh." << setfill('0') << setw(6) << myid;
   InitName << OutputDir << "/initial-gf." << setfill('0') << setw(6) << myid;
   ofstream omesh(MeshName.str().c_str());
   omesh.precision(precision);
   pmesh.Print(omesh);
   ofstream initial(InitName.str().c_str());
   initial.precision(precision);
   main.Save(initial);

   socketstream sout;
   char vishost[] = "localhost";
   int visport = 19916;
   {
      // Make sure all MPI ranks have sent their 'v' solution before initiating
      // another set of GLVis connections (one from each rank):
      MPI_Barrier(comm);
      ParVisualizeField(sout, vishost, visport, hyp->ProblemName, main,
                        hyp->glvis_scale);
   }

   FE_Evolution *evol;
   switch (scheme)
   {
      case Galerkin: { evol = new ParGalerkinEvolution(&vfes, hyp, pdofs); break; }
      case MonolithicConvexLimiting: { evol = new ParMCL_Evolution(&vfes, hyp, pdofs, dt); break; }
      default:
         MFEM_ABORT("Unknown evolution scheme");
   }

   Vector LumpedMassMat(pfes.GetVSize());
   ParBilinearForm ml(&pfes);
   ml.AddDomainIntegrator(new LumpedIntegrator(new MassIntegrator()));
   ml.Assemble();
   ml.Finalize();
   ml.SpMat().GetDiag(LumpedMassMat);

   double InitialMass, MassMPI = LumpedMassMat * main;
   MPI_Allreduce(&MassMPI, &InitialMass, 1, MPI_DOUBLE, MPI_SUM, comm);

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
   if (myid == 0)
   {
      cout << "Preprocessing done. Entering time stepping loop.\n";
   }

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
         if (myid == 0)
         {
            if (hyp->SteadyState)
            {
               cout << "time step: " << ti << ", time: " << t << ", residual: " << res << endl;
            }
            else
            {
               cout << "time step: " << ti << ", time: " << t << endl;
            }
         }

         ParVisualizeField(sout, vishost, visport, hyp->ProblemName, main,
                           hyp->glvis_scale);
         if (TransOutput)
         {
            ostringstream TransName;
            TransName << OutputDir << "/trans-gf-" << TransStep << "."
                      << setfill('0') << setw(6) << myid;
            ofstream trans(TransName.str().c_str());
            trans.precision(precision);
            main.Save(trans);
            TransStep++;
         }
      }
   }

   tic_toc.Stop();
   if (myid == 0)
   {
      cout << "Time stepping loop done in " << tic_toc.RealTime() << " seconds.\n\n";
   }

   double FinalMass, DomainSize, DomainSizeMPI = LumpedMassMat.Sum();
   MPI_Allreduce(&DomainSizeMPI, &DomainSize, 1, MPI_DOUBLE, MPI_SUM, comm);

   if (hyp->SolutionKnown)
   {
      Array<double> errors;
      hyp->ComputeErrors(errors, u, DomainSize, t);
      if (myid == 0)
      {
         cout << "L1 error:                    " << errors[0] << endl;
         hyp->WriteErrors(errors);
      }
   }

   double mainMin, mainMax, mainLoc = main.Min();
   MPI_Allreduce(&mainLoc, &mainMin, 1, MPI_DOUBLE, MPI_MIN, comm);
   mainLoc = main.Max();
   MPI_Allreduce(&mainLoc, &mainMax, 1, MPI_DOUBLE, MPI_MAX, comm);

   MassMPI = LumpedMassMat * main;
   MPI_Allreduce(&MassMPI, &FinalMass, 1, MPI_DOUBLE, MPI_SUM, comm);

   if (myid == 0)
   {
      cout << "Min of primary field:        " << mainMin << endl
           << "Max of primary field:        " << mainMax << endl
           << "Difference in solution mass: "
           << abs(InitialMass - FinalMass) / DomainSize << "\n\n";
   }

   ostringstream FinalName;
   FinalName << OutputDir << "/ultimate-gf." << setfill('0') << setw(6) << myid;
   ofstream ultimate(FinalName.str().c_str());
   ultimate.precision(precision);
   main.Save(ultimate);

   if (ProblemNum > 3)
   {
      ParGridFunction v(&pfes), p(&pfes);
      hyp->ComputeDerivedQuantities(u, v, p);

      ostringstream VelocityName;
      VelocityName << OutputDir << "/velocity-gf." << setfill('0') << setw(6) << myid;
      ofstream velocity(VelocityName.str().c_str());
      velocity.precision(precision);
      v.Save(velocity);

      if (ProblemNum == 5)
      {
         ostringstream PressureName;
         PressureName << OutputDir << "/pressure-gf." << setfill('0') << setw(6) << myid;
         ofstream pressure(PressureName.str().c_str());
         pressure.precision(precision);
         p.Save(pressure);
      }
   }

   delete evol;
   delete hyp;
   delete odeSolver;
   return 0;
}
