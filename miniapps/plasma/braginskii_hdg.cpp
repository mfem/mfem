// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.
//
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
#include "../../examples/ex18.hpp"
#include "braginskii_hdg_solver.hpp"

#define BRAGINSKII_IMEX

using namespace std;
using namespace mfem;
using namespace mfem::plasma;
using namespace mfem::hdg;

enum class Problem
{
   FastVortex = 1,
   SlowVortex,
   SineWave,
   Inflow,
};

enum Quantity
{
   Density,
   Momentum,
   Energy,
};

struct QuantityMap
{
   int dim;
   QuantityMap(int dim_) : dim(dim_) { }

   Quantity operator()(int i) const
   {
      if (i == 0) { return Quantity::Density; }
      else if (i > 0 && i <= dim) { return Quantity::Momentum; }
      else if (i == dim + 1) { return Quantity::Energy; }
      else { MFEM_ABORT("Unknown index"); }
   }

   inline int size() const { return dim + 2; }
};

struct ProblemParams
{
   Problem prob;
   real_t k{1.}, ks{1.}, ka{0.};
   real_t specific_heat_ratio{1.4};
   real_t gas_constant{1.};
   struct
   {
      real_t den_inf{1.};
      real_t ene_inf{1.};
      real_t den_cen{2.};
      Vector vel_cen{1.};
      real_t ene_cen{2.};
      Vector x_cen{0.5};
      real_t dx_cen{0.2};
      real_t dt_cen{0.2};
   } inflow;
};

class AnisotropicMatrixFunctionCoefficient : public MatrixCoefficient
{
public:
   using AnisoMatFunc =
      std::function<void(const Vector &, const Vector&, DenseMatrix &)>;

   AnisotropicMatrixFunctionCoefficient(int dim, AnisoMatFunc F_,
                                        VectorCoefficient &B_)
      : MatrixCoefficient(dim), F(std::move(F_)), B(B_) { }

   void Eval(DenseMatrix &K, ElementTransformation &T,
             const IntegrationPoint &ip) override
   {
      K.SetSize(height, width);
      Vector x, vB;
      B.Eval(vB, T, ip);
      T.Transform(ip, x);
      F(x, vB, K);
   }

private:
   AnisoMatFunc F;
   VectorCoefficient &B;
};

// Prints the program's logo to the given output stream
void display_banner(ostream & os);

// Define the analytical solution and forcing terms / boundary conditions
typedef std::function<void(const Vector &, Vector &)> VecFunc;
typedef std::function<void(const Vector &, real_t, Vector &)> VecTFunc;
typedef AnisotropicMatrixFunctionCoefficient::AnisoMatFunc MatFunc;

MatFunc GetKFun(const ProblemParams &params, Quantity q);
VecFunc GetU0Fun(const ProblemParams &params);
VecTFunc GetBCFun(const ProblemParams &params);

// Open the named VisItDataCollection and read the named field.
// Returns pointers to the two new objects.
int ReadGridFunction(const char * coll_name, const char * field_name,
                     int pad_digits_cycle, int cycle,
                     std::unique_ptr<VisItDataCollection> &dc,
                     GridFunction* &gf);

bool VisualizeField(socketstream &sout, const GridFunction &gf,
                    const char *name, real_t t = 0.);

int main(int argc, char *argv[])
{
   display_banner(cout);

   // 1. Parse command-line options.
   string mesh_file = "";
   int problem = 1;
   int IntOrderOffset = 1;
   int ref_levels = 1;
   int order = 3;
   bool dg = false;
   bool brt = false;
   int ode_solver_type = 4;
   ProblemParams pars;
   real_t t_final = 1.0;
   real_t dt = -0.01;
   real_t cfl = 0.3;
   real_t td = 0.5;

   const char *B_coll_name = "";
   const char *B_field_name = "B";
   int B_cycle = 10;
   int B_pad_digits_cycle = 6;

   bool hybridization = false;
   bool visualization = true;
   int vis_steps = 50;

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
   args.AddOption(&td, "-td", "--stab_diff",
                  "Diffusion stabilization factor (1/2=default)");
   args.AddOption(&ode_solver_type, "-s", "--ode-solver",
                  ODESolver::ImplicitTypes.c_str());
   args.AddOption(&t_final, "-tf", "--t-final", "Final time; start time is 0.");
   args.AddOption(&dt, "-dt", "--time-step",
                  "Time step. Positive number skips CFL timestep calculation.");
   args.AddOption(&cfl, "-c", "--cfl-number",
                  "CFL number for timestep calculation.");
   args.AddOption(&pars.k, "-k", "--kappa",
                  "Diffusivity coefficient");
   args.AddOption(&pars.ks, "-ks", "--kappa_sym",
                  "Symmetric anisotropy of the diffusivity tensor");
   args.AddOption(&pars.ka, "-ka", "--kappa_anti",
                  "Antisymmetric anisotropy of the diffusivity tensor");
   //inflow
   args.AddOption(&pars.inflow.den_inf, "-dinf", "--densitiy-infinity",
                  "Density at infinity");
   args.AddOption(&pars.inflow.ene_inf, "-einf", "--energy-infinity",
                  "Internal energy at infinity");
   args.AddOption(&pars.inflow.den_cen, "-dcen", "--densitiy-center",
                  "Density at the center of the source");
   args.AddOption(&pars.inflow.vel_cen, "-vcen", "--velocity-center",
                  "Velocity at the center of the source");
   args.AddOption(&pars.inflow.ene_cen, "-ecen", "--energy-center",
                  "Internal energy at the center of the source");
   args.AddOption(&pars.inflow.x_cen, "-xcen", "--x-center",
                  "Location of the source");
   args.AddOption(&pars.inflow.dx_cen, "-dxcen", "--width-center",
                  "Width of the source");
   args.AddOption(&pars.inflow.dt_cen, "-dtcen", "--time-center",
                  "Ramp-up time of the source");

   args.AddOption(&B_coll_name, "-bdc", "--b-data-collection",
                  "Set the VisIt data collection B field root file prefix.");
   args.AddOption(&B_field_name, "-bf", "--b-field-name",
                  "Set the VisIt data collection B field name");
   args.AddOption(&B_cycle, "-bcyc", "--b-cycle",
                  "Set the B field cycle index to read.");
   args.AddOption(&B_pad_digits_cycle, "-bpdc", "--b-pad-digits-cycle",
                  "Number of digits in B field cycle.");
   args.AddOption(&hybridization, "-hb", "--hybridization", "-no-hb",
                  "--no-hybridization", "Enable hybridization.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&vis_steps, "-vs", "--visualization-steps",
                  "Visualize every n-th timestep.");
   args.ParseCheck();

   pars.prob = (Problem)problem;

   // 2. Read the mesh from the given mesh file. When the user does not provide
   //    mesh file, use the default mesh file for the problem.
   Mesh mesh;
   if (!mesh_file.empty())
   {
      mesh = std::move(Mesh(mesh_file));
   }
   else
   {
      switch (pars.prob)
      {
         case Problem::FastVortex:
         case Problem::SlowVortex:
         case Problem::SineWave:
            mesh = std::move(Mesh("../../data/periodic-square.mesh"));
            break;
         default:
            mesh = std::move(Mesh("../../data/inline-quad.mesh"));
            break;
      }
   }

   const int dim = mesh.Dimension();
   QuantityMap qmap(dim);
   const int num_equations = qmap.size();

   // Refine the mesh to increase the resolution. In this example we do
   // 'ref_levels' of uniform refinement, where 'ref_levels' is a command-line
   // parameter.
   for (int lev = 0; lev < ref_levels; lev++)
   {
      mesh.UniformRefinement();
   }

   // Set boundary conditions

   bool bc_dirichlet = false;

   switch (pars.prob)
   {
      case Problem::FastVortex:
      case Problem::SlowVortex:
      case Problem::SineWave:
         break;
      case Problem::Inflow:
         bc_dirichlet = true;
         break;
      default:
         cerr << "Unknown problem" << std::endl;
         return 1;
   }

   Array<int> bdr_dirichlet_marker(mesh.bdr_attributes.Size()?(
                                      mesh.bdr_attributes.Max()):(0));
   Array<int> bdr_free_marker(mesh.bdr_attributes.Size()?(
                                 mesh.bdr_attributes.Max()):(0));

   if (bc_dirichlet)
   {
      bdr_dirichlet_marker = 0;
      bdr_dirichlet_marker[0] = -1;
      bdr_free_marker = -1;
      bdr_free_marker[0] = 0;
   }

   // Read the magnetic field grid function from the given VisIt data
   // collection or otherwise align anisotropy with the axes
   std::unique_ptr<VisItDataCollection> B_dc;
   GridFunction *B_gf;
   std::unique_ptr<VectorCoefficient> B_coeff;
   Vector B_vec;

   if (strlen(B_coll_name) > 0)
   {
      if (ReadGridFunction(B_coll_name, B_field_name, B_pad_digits_cycle,
                           B_cycle, B_dc, B_gf))
      {
         mfem::out << "Error loading B field" << endl;
         return 1;
      }

      B_coeff.reset(new VectorGridFunctionCoefficient(B_gf));
   }
   else
   {
      B_vec.SetSize(dim);
      B_vec = 0.;
      B_vec[0] = 1.;
      B_coeff.reset(new VectorConstantCoefficient(B_vec));
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
#ifndef BRAGINSKII_IMEX
   NonlinearForm *Mtnl = darcy.GetPotentialMassNonlinearForm();
#else // BRAGINSKII_IMEX
   BilinearForm *Mt = darcy.GetPotentialMassForm();
#endif // BRAGINSKII_IMEX

   //linear diffusion
   MatFunc kFun[] = { GetKFun(pars, Quantity::Density),
                      GetKFun(pars, Quantity::Momentum),
                      GetKFun(pars, Quantity::Energy)
                    };
   AnisotropicMatrixFunctionCoefficient kcoeff[] =
   {
      AnisotropicMatrixFunctionCoefficient(dim, kFun[Quantity::Density], *B_coeff),
      AnisotropicMatrixFunctionCoefficient(dim, kFun[Quantity::Momentum], *B_coeff),
      AnisotropicMatrixFunctionCoefficient(dim, kFun[Quantity::Energy], *B_coeff)
   }; //tensor diffusivity
   InverseMatrixCoefficient ikcoeff[] =
   {
      InverseMatrixCoefficient(kcoeff[Quantity::Density]),
      InverseMatrixCoefficient(kcoeff[Quantity::Momentum]),
      InverseMatrixCoefficient(kcoeff[Quantity::Energy]),
   }; //inverse tensor diffusivity

   if (dg)
   {
      std::vector<BilinearFormIntegrator*> bfis(num_equations);
      for (int i = 0; i < num_equations; i++)
      {
         bfis[i] = new VectorMassIntegrator(ikcoeff[qmap(i)]);
      }

      Mq->AddDomainIntegrator(new VectorBlockDiagonalIntegrator(bfis));
   }
   else
   {
      std::vector<BilinearFormIntegrator*> bfis(num_equations);
      for (int i = 0; i < num_equations; i++)
      {
         bfis[i] = new VectorFEMassIntegrator(ikcoeff[qmap(i)]);
      }

      Mq->AddDomainIntegrator(new VectorBlockDiagonalIntegrator(bfis));
   }

   if (dg && td > 0.)
   {
      std::vector<BilinearFormIntegrator*> bfis(num_equations);
      for (int i = 0; i < num_equations; i++)
      {
         bfis[i] = new HDGDiffusionIntegrator(kcoeff[qmap(i)], td);
      }
#ifndef BRAGINSKII_IMEX
      Mtnl->AddInteriorFaceIntegrator(new VectorBlockDiagonalIntegrator(bfis));
#else
      Mt->AddInteriorFaceIntegrator(new VectorBlockDiagonalIntegrator(bfis));
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
      B->AddBdrFaceIntegrator(new VectorBlockDiagonalIntegrator(
                                 num_equations, new TransposeIntegrator(new DGNormalTraceIntegrator(-1.))),
                              bdr_free_marker);
   }

   //nonlinear convection

   EulerFlux flux(dim, pars.specific_heat_ratio);
   RusanovFlux numericalFlux(flux);

   unique_ptr<HyperbolicFormIntegrator> hyperbolicIntegrator(
      new HyperbolicFormIntegrator(numericalFlux, IntOrderOffset, -1.));
#ifdef BRAGINSKII_IMEX
   std::unique_ptr<NonlinearForm> Mt_ex(new NonlinearForm(&vfes));
   Mt_ex->AddDomainIntegrator(hyperbolicIntegrator.get());
   Mt_ex->AddInteriorFaceIntegrator(hyperbolicIntegrator.get());
   Mt_ex->UseExternalIntegrators();
#else //BRAGINSKII_IMEX
   Mtnl->AddDomainIntegrator(hyperbolicIntegrator.get());
   Mtnl->AddInteriorFaceIntegrator(hyperbolicIntegrator.get());
   Mtnl->UseExternalIntegrators();//not handled by DarcyOperator!!
#endif //BRAGINSKII_IMEX

   // diffusion Dirichlet boundary condition

   auto bc_fun = GetBCFun(pars);
   VectorFunctionCoefficient bc_coeff(num_equations, bc_fun);
   ScalarVectorProductCoefficient negbc_coeff(-1., bc_coeff);

   if (bc_dirichlet)
   {
      LinearForm *bq = darcy.GetFluxRHS();

      if (dg)
      {
         bq->AddBdrFaceIntegrator(new VectorBoundaryFluxLFIntegrator(negbc_coeff),
                                  bdr_dirichlet_marker);
      }
      else if (brt)
      {
         bq->AddBdrFaceIntegrator(new VectorFEBoundaryFluxLFIntegrator(negbc_coeff),
                                  bdr_dirichlet_marker);
      }
      else
      {
         bq->AddBoundaryIntegrator(new VectorFEBoundaryFluxLFIntegrator(negbc_coeff),
                                   bdr_dirichlet_marker);
      }
   }

   // convection Dirichlet boundary condition

   unique_ptr<BdrHyperbolicFormIntegrator> hyperbolicIntegrator_bc;



   if (bc_dirichlet)
   {
      hyperbolicIntegrator_bc.reset(new BdrHyperbolicFormIntegrator(
                                       numericalFlux, bc_coeff, IntOrderOffset, -1.));

      Mt_ex->AddBdrFaceIntegrator(hyperbolicIntegrator_bc.get(),
                                  bdr_dirichlet_marker);
      Mt_ex->AddBdrFaceIntegrator(hyperbolicIntegrator.get(),
                                  bdr_free_marker);
   }

   //set hybridization / reduction

   Array<int> ess_flux_tdofs_list;

   if (!dg)
   {
      qvfes.GetEssentialTrueDofs(bdr_free_marker, ess_flux_tdofs_list);
   }

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

   // 6. Define the initial conditions, save the corresponding mesh and grid
   //    functions to files. These can be opened with GLVis using:
   //    "glvis -m euler-mesh.mesh -g euler-1-init.gf" (for x-momentum).

   // Initialize the state.
   auto u0Fun = GetU0Fun(pars);
   VectorFunctionCoefficient u0(num_equations, u0Fun);

   const Array<int> block_offsets(DarcyOperator::ConstructOffsets(darcy));
   BlockVector x(block_offsets);
   x = 0.;
   GridFunction sol(&vfes, x.GetBlock(1), 0);
   sol.ProjectCoefficient(u0);
   GridFunction den(&fes, sol, 0);
   GridFunction mom(&dfes, sol, fes.GetNDofs());
   GridFunction ene(&fes, sol, (1 + dim) * fes.GetNDofs());

   if (hybridization)
   {
      darcy.GetHybridization()->ProjectSolution(x, x.GetBlock(2));
   }

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

   // Setup the system operator

   Array<Coefficient*> coeffs;
   DarcyOperator darcy_op(ess_flux_tdofs_list, &darcy, NULL, NULL, NULL, coeffs,
                          DarcyOperator::SolverType::Newton, false, true);

#ifdef BRAGINSKII_IMEX
   Mt_ex->Setup();
   ConvectionDiffusionOp op(fes, darcy_op, *Mt_ex);
#else //BRAGINSKII_IMEX
   DarcyOperator &op = darcy_op;
#endif //BRAGINSKII_IMEX

   constexpr real_t rtol = 1e-6;
   constexpr real_t atol = 0.;
   constexpr int max_iter = 1000;

   op.SetTolerance(rtol, atol);
   op.SetMaxIters(max_iter);

   // 7. Visualize state
   socketstream sden, smom, sene;
   if (visualization && !VisualizeField(sden, den, "density")) { visualization = false; }
   if (visualization && !VisualizeField(smom, mom, "momentum")) { visualization = false; }
   if (visualization && !VisualizeField(sene, ene, "energy")) { visualization = false; }

   // 8. Time integration

   // When dt is not specified, use CFL condition.
   // Compute h_min and initial maximum characteristic speed
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
   cout << "initial dt: " << dt << endl;

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
      hyperbolicIntegrator->ResetMaxCharSpeed();

      bc_coeff.SetTime(t);

      ode_solver->Step(x, t, dt_real);

      if (cfl > 0) // update time step size with CFL
      {
         real_t max_char_speed = hyperbolicIntegrator->GetMaxCharSpeed();
         dt = cfl * hmin / max_char_speed / (2 * order + 1);
      }
      ti++;

      done = (t >= t_final - 1e-8 * dt);
      if (done || ti % vis_steps == 0)
      {
         cout << "time step: " << ti << ", time: " << t << ", dt: " << dt << endl;
         if (visualization)
         {
            VisualizeField(sden, den, "density", t);
            VisualizeField(smom, mom, "momentum", t);
            VisualizeField(sene, ene, "energy", t);
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

void display_banner(ostream & os)
{
   os << "__________                        __               __    __ __ " << endl
      << "\\______   \\____________     ____ |__| ____   _____|  | _|__|__|" << endl
      << " |    |  _/\\_  __ \\__  \\   / ___\\|  |/    \\ /  ___/  |/ /  |  |" <<
      endl
      << " |    |   \\ |  | \\// __ \\_/ /_/  >  |   |  \\\\___ \\|    <|  |  |" <<
      endl
      << " |______  / |__|  (____  /\\___  /|__|___|  /____  >__|_ \\__|__|" << endl
      << "        \\/             \\//_____/         \\/     \\/     \\/      " <<
      endl
      << endl<< endl << flush;
}

MatFunc GetKFun(const ProblemParams &params, Quantity q)
{
   const real_t &k = params.k;
   const real_t &ks = params.ks;
   const real_t &ka = params.ka;

   switch (params.prob)
   {
      case Problem::FastVortex:
      case Problem::SlowVortex:
      case Problem::SineWave:
      case Problem::Inflow:
         return [=](const Vector &x, const Vector &B, DenseMatrix &kappa)
         {
            const int ndim = x.Size();
            Vector b(B);
            b /= b.Norml2();
            kappa.Diag(ks * k, ndim);
            if (ks != 1.)
            {
               AddMult_a_VVt((1. - ks) * k, b, kappa);
            }
         };
   }

   return MatFunc();
}

VecFunc GetU0Fun(const ProblemParams &params)
{
   switch (params.prob)
   {
      case Problem::FastVortex:
         return GetMovingVortexInit(0.2, 0.5, 1. / 5., params.gas_constant,
                                    params.specific_heat_ratio);
      case Problem::SlowVortex:
         return GetMovingVortexInit(0.2, 0.05, 1. / 50., params.gas_constant,
                                    params.specific_heat_ratio);
      case Problem::SineWave:
         return [](const Vector &x, Vector &y)
         {
            MFEM_ASSERT(x.Size() == 2, "");
            const real_t density = 1.0 + 0.2 * std::sin(M_PI*(x(0) + x(1)));
            const real_t velocity_x = 0.7;
            const real_t velocity_y = 0.3;
            const real_t pressure = 1.0;
            const real_t energy =
               pressure / (1.4 - 1.0) +
               density * 0.5 * (velocity_x * velocity_x + velocity_y * velocity_y);

            y(0) = density;
            y(1) = density * velocity_x;
            y(2) = density * velocity_y;
            y(3) = energy;
         };
      case Problem::Inflow:
         return [&params](const Vector &x, Vector &y)
         {
            const int dim = x.Size();

            const real_t &den_inf = params.inflow.den_inf;
            const real_t &ene_inf = params.inflow.ene_inf;

            y(0) = den_inf;
            for (int d = 0; d < dim; d++)
            {
               y(d+1) = 0.;
            }
            y(dim+1) = den_inf * ene_inf;
         };
   }

   return VecFunc();
}

VecTFunc GetBCFun(const ProblemParams &params)
{
   switch (params.prob)
   {
      case Problem::FastVortex:
      case Problem::SlowVortex:
      case Problem::SineWave:
         break;
      case Problem::Inflow:
         return [&params](const Vector &x, real_t t, Vector &y)
         {
            const int dim = x.Size();

            const real_t &den_inf = params.inflow.den_inf;
            const real_t &ene_inf = params.inflow.ene_inf;
            const real_t &den_c = params.inflow.den_cen;
            const Vector &vel_c = params.inflow.vel_cen;
            const real_t &ene_c = params.inflow.ene_cen;
            const Vector &x_c = params.inflow.x_cen;
            const real_t &dx_c = params.inflow.dx_cen;
            const real_t &dt_c = params.inflow.dt_cen;

            Vector dx(dim);
            for (int d = 0; d < dim; d++)
            {
               const real_t x_cd = (x_c.Size() > d)?(x_c(d)):(0.);
               dx(d) = (x(d) - x_cd) / dx_c;
            }
            const real_t exp_dx = exp(-(dx*dx));
            const real_t dt = t / dt_c;
            const real_t exp_dt = 1. - exp(-dt*dt);
            const real_t exp_dxt = exp_dx * exp_dt;

            const real_t den = den_inf + (den_c - den_inf) * exp_dxt;
            const real_t ene = ene_inf + (ene_c - ene_inf) * exp_dxt;

            y(0) = den;
            for (int d = 0; d < dim; d++)
            {
               const real_t v_c = (vel_c.Size() > d)?(vel_c(d)):(0.);
               y(d+1) = den * v_c * exp_dxt;
            }
            y(dim+1) = den * ene;
         };
   }

   return VecTFunc();
}

int ReadGridFunction(const char * coll_name, const char * field_name,
                     int pad_digits_cycle, int cycle,
                     std::unique_ptr<VisItDataCollection> &dc,
                     GridFunction* &gf)
{
   dc.reset(new VisItDataCollection(coll_name));
   dc->SetPadDigitsCycle(pad_digits_cycle);
   dc->Load(cycle);

   if (dc->Error() != DataCollection::No_Error)
   {
      mfem::out << "Error loading VisIt data collection: "
                << coll_name << endl;
      return 1;
   }

   if (dc->HasField(field_name))
   {
      gf = dc->GetField(field_name);
   }

   return 0;
}

bool VisualizeField(socketstream &sout, const GridFunction &gf,
                    const char *name, real_t t)
{
   const char vishost[] = "localhost";
   const int visport = 19916;
   if (!sout.is_open())
   {
      sout.open(vishost, visport);
   }
   if (!sout)
   {
      cout << "Unable to connect to GLVis server at " << vishost << ':'
           << visport << endl;
      cout << "GLVis visualization disabled.\n";
      return false;
   }
   else
   {
      constexpr int precision = 8;
      sout.precision(precision);
      // Plot magnitude of vector-valued momentum
      sout << "solution\n" << *gf.FESpace()->GetMesh() << gf;
      if (t == 0.)
      {
         sout << "window_title '" << name << ", t = 0'\n";
         sout << "view 0 0\n";  // view from top
         sout << "keys jlm\n";  // turn off perspective and light, show mesh
      }
      else
      {
         sout << "window_title '" << name << ", t = " << t << "'\n";
      }
      //sout << "pause\n";
      sout << flush;
      //cout << "GLVis visualization paused."
      //      << " Press space (in the GLVis window) to resume it.\n";
   }
   return true;
}
