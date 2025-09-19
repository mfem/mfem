//                                MFEM Example 18
//
// Compile with: make ex18
//
// Sample runs:
//
//       ex18 -p 1 -r 2 -o 1 -s 3
//       ex18 -p 1 -r 1 -o 3 -s 4
//       ex18 -p 1 -r 0 -o 5 -s 6
//       ex18 -p 2 -r 1 -o 1 -s 3 -mf
//       ex18 -p 2 -r 0 -o 3 -s 3 -mf
//
// Description:  This example code solves the compressible Euler system of
//               equations, a model nonlinear hyperbolic PDE, with a
//               discontinuous Galerkin (DG) formulation.
//
//                (u_t, v)_T - (F(u), ∇ v)_T + <F̂(u,n), [[v]]>_F = 0
//
//               where (⋅,⋅)_T is volume integration, and <⋅,⋅>_F is face
//               integration, F is the Euler flux function, and F̂ is the
//               numerical flux.
//
//               Specifically, it solves for an exact solution of the equations
//               whereby a vortex is transported by a uniform flow. Since all
//               boundaries are periodic here, the method's accuracy can be
//               assessed by measuring the difference between the solution and
//               the initial condition at a later time when the vortex returns
//               to its initial location.
//
//               Note that as the order of the spatial discretization increases,
//               the timestep must become smaller. This example currently uses a
//               simple estimate derived by Cockburn and Shu for the 1D RKDG
//               method. An additional factor can be tuned by passing the --cfl
//               (or -c shorter) flag.
//
//               The example demonstrates usage of DGHyperbolicConservationLaws
//               that wraps NonlinearFormIntegrators containing element and face
//               integration schemes. In this case the system also involves an
//               external approximate Riemann solver for the DG interface flux.
//               By default, weak-divergence is pre-assembled in element-wise
//               manner, which corresponds to (I_h(F(u_h)), ∇ v). This yields
//               better performance and similar accuracy for the included test
//               problems. This can be turned off and use nonlinear assembly
//               similar to matrix-free assembly when -mf flag is provided.
//               It also demonstrates how to use GLVis for in-situ visualization
//               of vector grid function and how to set top-view.
//
//               We recommend viewing examples 9, 14 and 17 before viewing this
//               example.

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <sstream>
#include "ex18.hpp"
#include "darcyop.hpp"

#define EX18_NONLINEAR
#define EX18_ADAPTIVE_DT
#define EX18_EULER

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   int problem = 1;
   const real_t specific_heat_ratio = 1.4;
   const real_t gas_constant = 1.0;

   string mesh_file = "";
   int IntOrderOffset = 1;
   int ref_levels = 1;
   int order = 3;
   bool dg = false;
   bool brt = false;
   int ode_solver_type = 4;
   real_t t_final = 2.0;
   real_t dt = -0.01;
   real_t cfl = 0.3;
   real_t k = 1.;
   real_t td = 0.5;
   bool reduction = false;
   bool hybridization = false;
   bool visualization = true;
   bool preassembleWeakDiv = true;
   int vis_steps = 50;

   int precision = 8;
   cout.precision(precision);

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use. If not provided, then a periodic square"
                  " mesh will be used.");
   args.AddOption(&problem, "-p", "--problem",
                  "Problem setup to use. See EulerInitialCondition().");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly.");
   args.AddOption(&order, "-o", "--order",
                  "Order (degree) of the finite elements.");
   args.AddOption(&dg, "-dg", "--discontinuous", "-no-dg",
                  "--no-discontinuous", "Enable DG elements for fluxes.");
   args.AddOption(&brt, "-brt", "--broken-RT", "-no-brt",
                  "--no-broken-RT", "Enable broken RT elements for fluxes.");
   args.AddOption(&ode_solver_type, "-s", "--ode-solver",
                  ODESolver::ImplicitTypes.c_str());
   args.AddOption(&t_final, "-tf", "--t-final", "Final time; start time is 0.");
   args.AddOption(&dt, "-dt", "--time-step",
                  "Time step. Positive number skips CFL timestep calculation.");
   args.AddOption(&cfl, "-c", "--cfl-number",
                  "CFL number for timestep calculation.");
   args.AddOption(&k, "-k", "--kappa",
                  "Diffusivity coefficient");
   args.AddOption(&td, "-td", "--stab_diff",
                  "Diffusion stabilization factor (1/2=default)");
   args.AddOption(&reduction, "-rd", "--reduction", "-no-rd",
                  "--no-reduction", "Enable reduction.");
   args.AddOption(&hybridization, "-hb", "--hybridization", "-no-hb",
                  "--no-hybridization", "Enable hybridization.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&preassembleWeakDiv, "-ea", "--element-assembly-divergence",
                  "-mf", "--matrix-free-divergence",
                  "Weak divergence assembly level\n"
                  "    ea - Element assembly with interpolated F\n"
                  "    mf - Nonlinear assembly in matrix-free manner");
   args.AddOption(&vis_steps, "-vs", "--visualization-steps",
                  "Visualize every n-th timestep.");
   args.ParseCheck();

   // 2. Read the mesh from the given mesh file. When the user does not provide
   //    mesh file, use the default mesh file for the problem.
   Mesh mesh = mesh_file.empty() ? EulerMesh(problem) : Mesh(mesh_file);
   const int dim = mesh.Dimension();
   const int num_equations = dim + 2;

   // Refine the mesh to increase the resolution. In this example we do
   // 'ref_levels' of uniform refinement, where 'ref_levels' is a command-line
   // parameter.
   for (int lev = 0; lev < ref_levels; lev++)
   {
      mesh.UniformRefinement();
   }

   // 3. Define the ODE solver used for time integration. Several explicit
   //    Runge-Kutta methods are available.
   unique_ptr<ODESolver> ode_solver = ODESolver::SelectImplicit(ode_solver_type);

   // 4. Define the discontinuous DG finite element space of the given
   //    polynomial order on the refined mesh.
   DG_FECollection fec(order, dim);
   // Finite element space for a scalar (thermodynamic quantity)
   FiniteElementSpace fes(&mesh, &fec);
   // Finite element space for a mesh-dim vector quantity (momentum)
   FiniteElementSpace dfes(&mesh, &fec, dim, Ordering::byNODES);
   // Finite element space for all variables together (total thermodynamic state)
   FiniteElementSpace vfes(&mesh, &fec, num_equations, Ordering::byNODES);

   unique_ptr<FiniteElementCollection> qfec;
   if (dg)
   {
      // In the case of LDG formulation, we chose a closed basis as it
      // is customary for HDG to match trace DOFs, but an open basis can
      // be used instead.
      qfec.reset(new L2_FECollection(order, dim, BasisType::GaussLobatto));
   }
   else if (brt)
   {
      qfec.reset(new BrokenRT_FECollection(order, dim));
   }
   else
   {
      qfec.reset(new RT_FECollection(order, dim));
   }
   FiniteElementSpace qfes(&mesh, qfec.get(), (dg)?(dim):(1));
   FiniteElementSpace qvfes(&mesh, qfec.get(), ((dg)?(dim):(1)) * num_equations);

   DarcyForm darcy(&qvfes, &vfes);

   // This example depends on this ordering of the space.
   MFEM_ASSERT(fes.GetOrdering() == Ordering::byNODES, "");

   cout << "Number of unknowns: " << vfes.GetVSize() << endl;

   // 5. Set up the nonlinear form with euler flux and numerical flux

   BilinearForm *Mq = darcy.GetFluxMassForm();
   MixedBilinearForm *B = darcy.GetFluxDivForm();
#ifdef EX18_NONLINEAR
   NonlinearForm *Mtnl = darcy.GetPotentialMassNonlinearForm();
#else
   BilinearForm *Mt = darcy.GetPotentialMassForm();
#endif

   //linear diffusion
   ConstantCoefficient kcoeff(k);
   ConstantCoefficient ikcoeff(1./k);

   if (dg)
   {
      Mq->AddDomainIntegrator(new VectorBlockDiagonalIntegrator(num_equations,
                                                                new VectorMassIntegrator(ikcoeff)));
   }
   else
   {
      Mq->AddDomainIntegrator(new VectorBlockDiagonalIntegrator(num_equations,
                                                                new VectorFEMassIntegrator(ikcoeff)));
   }

   if (dg && td > 0.)
   {
#ifdef EX18_NONLINEAR
      Mtnl->AddInteriorFaceIntegrator(new VectorBlockDiagonalIntegrator(num_equations,
                                                                        new HDGDiffusionIntegrator(kcoeff, td)));
#else
      Mt->AddInteriorFaceIntegrator(new VectorBlockDiagonalIntegrator(num_equations,
                                                                      new HDGDiffusionIntegrator(kcoeff, td)));
#endif
   }

   //divergence/weak gradient

   if (dg)
   {
      B->AddDomainIntegrator(new VectorBlockDiagonalIntegrator(num_equations,
                                                               new VectorDivergenceIntegrator()));
   }
   else
   {
      B->AddDomainIntegrator(new VectorBlockDiagonalIntegrator(num_equations,
                                                               new VectorFEDivergenceIntegrator()));
   }

   if (dg || brt)
   {
      B->AddInteriorFaceIntegrator(new VectorBlockDiagonalIntegrator(
                                      num_equations, new TransposeIntegrator(new DGNormalTraceIntegrator(-1.))));
   }

   //nonlinear convection
#ifdef EX18_NONLINEAR
#ifdef EX18_EULER
   EulerFlux flux(dim, specific_heat_ratio);
#else
   Vector c {1., 0.};
   VectorConstantCoefficient ccoeff(c);
   AdvectionFlux flux1D(ccoeff);
   CompoundFlux flux(num_equations, flux1D);
#endif
   RusanovFlux numericalFlux(flux);

   unique_ptr<HyperbolicFormIntegrator> hyperbolicIntegrator(
      new HyperbolicFormIntegrator(numericalFlux, 0, -1.));
   Mtnl->AddDomainIntegrator(hyperbolicIntegrator.get());
   Mtnl->AddInteriorFaceIntegrator(hyperbolicIntegrator.get());
   Mtnl->UseExternalIntegrators();
#else
   Vector c {1., 0.};
   VectorConstantCoefficient ccoeff(c);
   Mt->AddDomainIntegrator(new VectorBlockDiagonalIntegrator(num_equations,
                                                             new ConservativeConvectionIntegrator(ccoeff)));
   Mt->AddInteriorFaceIntegrator(new VectorBlockDiagonalIntegrator(num_equations,
                                                                   new HDGConvectionCenteredIntegrator(ccoeff)));
#endif

   //set hybridization / reduction

   Array<int> ess_flux_tdofs_list;

   unique_ptr<FiniteElementCollection> trace_coll;
   unique_ptr<FiniteElementSpace> trace_space;
   if (hybridization)
   {
      trace_coll.reset(new DG_Interface_FECollection(order, dim));
      trace_space.reset(new FiniteElementSpace(&mesh, trace_coll.get(),
                                               num_equations));
      darcy.EnableHybridization(trace_space.get(),
                                new VectorBlockDiagonalIntegrator(
                                   num_equations, new NormalTraceJumpIntegrator()),
                                ess_flux_tdofs_list);

   }
   else if (reduction)
   {
      if (dg || brt)
      {
         darcy.EnableFluxReduction();
      }
      else
      {
         std::cerr << "No possible reduction!" << std::endl;
         return 1;
      }
   }

   // 6. Define the initial conditions, save the corresponding mesh and grid
   //    functions to files. These can be opened with GLVis using:
   //    "glvis -m euler-mesh.mesh -g euler-1-init.gf" (for x-momentum).

   // Initialize the state.
   VectorFunctionCoefficient u0 = EulerInitialCondition(problem,
                                                        specific_heat_ratio,
                                                        gas_constant);

   const Array<int> block_offsets(DarcyOperator::ConstructOffsets(darcy));
   BlockVector x(block_offsets);
   x = 0.;
   GridFunction sol(&vfes, x.GetBlock(1), 0);
   sol.ProjectCoefficient(u0);
   GridFunction den(&fes, sol, 0);
   GridFunction mom(&dfes, sol, fes.GetNDofs());

   // Output the initial solution.
   /*{
      ostringstream mesh_name;
      mesh_name << "euler-mesh.mesh";
      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(precision);
      mesh_ofs << mesh;

      for (int k = 0; k < num_equations; k++)
      {
         GridFunction uk(&fes, sol.GetData() + k * fes.GetNDofs());
         ostringstream sol_name;
         sol_name << "euler-" << k << "-init.gf";
         ofstream sol_ofs(sol_name.str().c_str());
         sol_ofs.precision(precision);
         sol_ofs << uk;
      }
   }*/

   Array<Coefficient*> coeffs;
   DarcyOperator op(ess_flux_tdofs_list, &darcy, NULL, NULL, NULL, coeffs,
                    DarcyOperator::SolverType::Newton, false, true);

   // 7. Visualize momentum with its magnitude
   socketstream sout;
   if (visualization)
   {
      char vishost[] = "localhost";
      int visport = 19916;

      sout.open(vishost, visport);
      if (!sout)
      {
         visualization = false;
         cout << "Unable to connect to GLVis server at " << vishost << ':'
              << visport << endl;
         cout << "GLVis visualization disabled.\n";
      }
      else
      {
         sout.precision(precision);
         // Plot magnitude of vector-valued momentum
         sout << "solution\n" << mesh << mom;
         sout << "window_title 'momentum, t = 0'\n";
         sout << "view 0 0\n";  // view from top
         sout << "keys jlm\n";  // turn off perspective and light, show mesh
         //sout << "pause\n";
         sout << flush;
         cout << "GLVis visualization paused."
              << " Press space (in the GLVis window) to resume it.\n";
      }
   }

   // 8. Time integration

   // When dt is not specified, use CFL condition.
   // Compute h_min and initial maximum characteristic speed
#if defined(EX18_NONLINEAR) && defined(EX18_ADAPTIVE_DT)
   real_t hmin = infinity();
   if (cfl > 0)
   {
      for (int i = 0; i < mesh.GetNE(); i++)
      {
         hmin = min(mesh.GetElementSize(i, 1), hmin);
      }
      // Find a safe dt, using a temporary vector. Calling Mult() computes the
      // maximum char speed at all quadrature points on all faces (and all
      // elements with -mf).
      Vector z(x.Size());
      hyperbolicIntegrator->ResetMaxCharSpeed();
      op.ImplicitSolve(dt, x, z);

      real_t max_char_speed = hyperbolicIntegrator->GetMaxCharSpeed();
      dt = cfl * hmin / max_char_speed / (2 * order + 1);
   }
#endif

   // Start the timer.
   tic_toc.Clear();
   tic_toc.Start();

   // Init time integration
   real_t t = 0.0;
   op.SetTime(t);
   ode_solver->Init(op);

   // Integrate in time.
   bool done = false;
   for (int ti = 0; !done;)
   {
      real_t dt_real = min(dt, t_final - t);
#if defined(EX18_NONLINEAR) && defined(EX18_ADAPTIVE_DT)
      hyperbolicIntegrator->ResetMaxCharSpeed();
#endif
      ode_solver->Step(x, t, dt_real);
#if defined(EX18_NONLINEAR) && defined(EX18_ADAPTIVE_DT)
      if (cfl > 0) // update time step size with CFL
      {
         real_t max_char_speed = hyperbolicIntegrator->GetMaxCharSpeed();
         dt = cfl * hmin / max_char_speed / (2 * order + 1);
      }
#endif
      ti++;

      done = (t >= t_final - 1e-8 * dt);
      if (done || ti % vis_steps == 0)
      {
         cout << "time step: " << ti << ", time: " << t << endl;
         if (visualization)
         {
            sout << "window_title 'momentum, t = " << t << "'\n";
            sout << "solution\n" << mesh << mom << flush;
         }
      }
   }

   tic_toc.Stop();
   cout << " done, " << tic_toc.RealTime() << "s." << endl;

   // 9. Save the final solution. This output can be viewed later using GLVis:
   //    "glvis -m euler-mesh-final.mesh -g euler-1-final.gf" (for x-momentum).
   /*{
      ostringstream mesh_name;
      mesh_name << "euler-mesh-final.mesh";
      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(precision);
      mesh_ofs << mesh;

      for (int k = 0; k < num_equations; k++)
      {
         GridFunction uk(&fes, sol.GetData() + k * fes.GetNDofs());
         ostringstream sol_name;
         sol_name << "euler-" << k << "-final.gf";
         ofstream sol_ofs(sol_name.str().c_str());
         sol_ofs.precision(precision);
         sol_ofs << uk;
      }
   }*/

   // 10. Compute the L2 solution error summed for all components.
   const real_t error = sol.ComputeLpError(2, u0);
   cout << "Solution error: " << error << endl;

   return 0;
}
