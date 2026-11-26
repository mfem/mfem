#include "examples/support/manufactured.hpp"
#include <cmath>
#include <iomanip>
#include <string>

using namespace mfem;
using namespace mfem::flow;

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();
   Device device("cpu");
   const char *mesh_file = "";
   const char *element = "quad";
   const char *solver_name = "fgmres";
   const char *preconditioner_name = "block-diagonal-amg";
   const char *method_name = "pspg";
   const char *pressure_gauge_name = "auto";
   const char *output_directory = "stokes-output";
   int dimension = 0, cells = 4, order = 2, pressure_order = -1, refinements = 0;
   real_t viscosity_scale = 1.0, shift = 0.0;
   bool variable_viscosity = false, paraview = false;
   FlowSolverOptions options;
   IterativeOptions iterative;
   Stabilization discretization = Stabilization{PSPGOptions{}};
   PressureGauge gauge_policy = PressureGauge::Auto();
   LinearSolverType linear_solver = LinearSolverType::FGMRES;
   int integration_order = -1;
   real_t pspg_constant = 1.0, cip_gamma = 0.01, gls_tau_scale = 1.0;
   options.linear.print_level = 1;
   OptionsParser args(argc, argv);
   args.AddOption(&pressure_gauge_name, "-pg", "--pressure-gauge",
                  "auto (default), mean-zero (require a pressure nullspace), or none (require no pressure nullspace).");
   args.AddOption(&preconditioner_name, "-pc", "--preconditioner",
                  "Preconditioner: block-diagonal-amg, block-diagonal-p1-amg, block-upper-p1-amg, block-lower-p1-amg; append -assembled to a P1-AMG choice for direct P1 coarse assembly.");
   args.AddOption(&iterative.p1_smoothing_steps, "-p1s", "--p1-smoothing-steps",
                  "Positive fixed number of l1-Jacobi sweeps before/after P1-AMG correction.");
   args.AddOption(&mesh_file, "-m", "--mesh", "Optional mesh file (unit domain for MMS).");
   args.AddOption(&element, "-e", "--element", "Generated element type: tri, quad, tet, hex.");
   args.AddOption(&dimension, "-dim", "--dim", "Required dimension, or 0 to infer from mesh/element.");
   args.AddOption(&cells, "-n", "--cells", "Cells per coordinate in the generated mesh.");
   args.AddOption(&order, "-o", "--order", "Velocity H1 polynomial order, at least 2.");
   args.AddOption(&pressure_order, "-po", "--pressure-order", "Pressure order, or -1 for method-dependent selection.");
   args.AddOption(&method_name, "-method", "--method", "pspg, cip, gls, or taylor-hood.");
   args.AddOption(&refinements, "-r", "--refine", "Uniform refinements before partitioning.");
   args.AddOption(&viscosity_scale, "-mu", "--mu", "Positive viscosity scale.");
   args.AddOption(&variable_viscosity, "-vm", "--variable-viscosity", "-cm", "--constant-viscosity",
                  "Use mu*(1+x+y[+z]) instead of constant viscosity.");
   args.AddOption(&shift, "-sh", "--shift", "Constant offset multiplier for manufactured velocity.");
   args.AddOption(&solver_name, "-s", "--solver", "mumps, gmres, or fgmres.");
   args.AddOption(&pspg_constant, "-tau", "--tau-constant",
                  "Positive dimensionless PSPG multiplier C (default 1), not a raw tau.\n\t"
                  "Automatic tau_K = C*h_K^2/(mean(mu)_K*(k+1)^4), with k the velocity degree.\n\t"
                  "h_K is the local mesh size metric; mean(mu)_K is the cell-average viscosity.\n\t"
                  "Omit for the default multiplier; specifying this option overrides C only.\n\t"
                  "Used only with --method pspg.");
   args.AddOption(&cip_gamma, "-cip", "--cip-gamma",
                  "Positive dimensionless CIP multiplier gamma (default 0.01), not a raw face weight.\n\t"
                  "Automatic weight_F = gamma*h_F^3*(1/mean(mu)_K+ + 1/mean(mu)_K-)/2.\n\t"
                  "h_F = min(h_K+,h_K-), using adjacent element size metrics and cell-average viscosities.\n\t"
                  "No additional degree scaling. Omit for the default; this option overrides gamma only.\n\t"
                  "Used only with --method cip.");
   args.AddOption(&gls_tau_scale, "-gls", "--gls-tau-scale",
                  "Positive dimensionless GLS multiplier C (default 1), not a raw tau.\n\t"
                  "Automatic tau_K = C*h_K^2/(mean(mu)_K*(k+1)^4), with k the velocity degree.\n\t"
                  "h_K is the local mesh size metric; mean(mu)_K is the cell-average viscosity.\n\t"
                  "Omit for the default multiplier; specifying this option overrides C only.\n\t"
                  "Used only with --method gls.");
   args.AddOption(&integration_order, "-q", "--quadrature-order", "Integration order; -1 selects 2*k+2+2*dim*(g-1) for coordinate degree g.");
   args.AddOption(&options.linear.relative_tolerance, "-rtol", "--relative-tolerance", "Relative solver tolerance.");
   args.AddOption(&options.linear.absolute_tolerance, "-atol", "--absolute-tolerance", "Absolute solver tolerance.");
   args.AddOption(&iterative.max_iterations, "-mi", "--max-iterations", "Maximum Krylov iterations.");
   args.AddOption(&iterative.restart_dimension, "-kd", "--restart", "Krylov restart dimension.");
   args.AddOption(&iterative.amg_elasticity_interp_refine, "-air", "--amg-interp-refine",
                  "-no-air", "--no-amg-interp-refine", "Refine elasticity AMG interpolation.");
   args.AddOption(&options.linear.print_level, "-pl", "--print-level", "Solver verbosity.");
   args.AddOption(&paraview, "-pv", "--paraview", "-no-pv", "--no-paraview", "Write ParaView output.");
   args.AddOption(&output_directory, "-out", "--output", "ParaView output directory.");
   args.Parse();
   if (!args.Good())
   {
      if (Mpi::Root()) { args.PrintUsage(out); }
      return args.Help() ? 0 : 1;
   }
   const std::string selected_solver(solver_name);
   const std::string selected_preconditioner(preconditioner_name);
   bool valid_preconditioner = false;
   for (auto preconditioner : {PreconditionerType::BlockDiagonalAMG,
                              PreconditionerType::BlockDiagonalP1AMG,
                              PreconditionerType::BlockUpperTriangularP1AMG,
                              PreconditionerType::BlockLowerTriangularP1AMG,
                              PreconditionerType::BlockDiagonalP1AMGAssembled,
                              PreconditionerType::BlockUpperTriangularP1AMGAssembled,
                              PreconditionerType::BlockLowerTriangularP1AMGAssembled})
   {
      if (selected_preconditioner == PreconditionerName(preconditioner))
      {
         iterative.preconditioner = preconditioner;
         valid_preconditioner = true;
      }
   }
   MFEM_VERIFY(valid_preconditioner, "Unknown Stokes preconditioner");
   bool valid_pressure_gauge = false;
   for (auto pressure_gauge : {PressureGauge::Auto(), PressureGauge::MeanValue(0.0),
                              PressureGauge::None()})
   {
      if (std::string(pressure_gauge_name) == PressureGaugeName(pressure_gauge))
      {
         gauge_policy = pressure_gauge;
         valid_pressure_gauge = true;
      }
   }
   MFEM_VERIFY(valid_pressure_gauge, "Unknown Stokes pressure-gauge policy");
   const std::string selected_method(method_name);
   MFEM_VERIFY(selected_method == "pspg" || selected_method == "cip" ||
               selected_method == "gls" || selected_method == "taylor-hood", "Unknown Stokes method");
   discretization = selected_method == "pspg" ? Stabilization{PSPGOptions{pspg_constant}} :
                    selected_method == "cip" ? Stabilization{CIPOptions{cip_gamma}} :
                    selected_method == "gls" ? Stabilization{GLSOptions{gls_tau_scale}} : Stabilization{NoStabilization{}};
   const int required_pressure_order = std::holds_alternative<NoStabilization>(discretization) ? order - 1 : order;
   MFEM_VERIFY(order >= 2 && (pressure_order == -1 || pressure_order == required_pressure_order),
               "Require velocity order >=2 and method-compatible pressure order");
   pressure_order = required_pressure_order;
   MFEM_VERIFY(selected_solver == "mumps" || selected_solver == "gmres" || selected_solver == "fgmres",
               "Unknown solver");
   linear_solver = selected_solver == "mumps" ? LinearSolverType::MUMPS :
                           selected_solver == "gmres" ? LinearSolverType::GMRES : LinearSolverType::FGMRES;
   MFEM_VERIFY(refinements >= 0 && std::isfinite(viscosity_scale) && viscosity_scale > 0.0,
               "Invalid refinement count or viscosity");
   Mesh serial_mesh = std::string(mesh_file).empty() ? MakeStokesMesh(element, cells) :
                      Mesh(mesh_file, 1, 1);
   MFEM_VERIFY(dimension == 0 || dimension == serial_mesh.Dimension(), "Mesh dimension mismatch");
   dimension = serial_mesh.Dimension();
   for (int refinement = 0; refinement < refinements; ++refinement) { serial_mesh.UniformRefinement(); }
   ParMesh mesh(MPI_COMM_WORLD, serial_mesh);
   H1_FECollection collection(order, dimension);
   H1_FECollection pressure_collection(pressure_order, dimension);
   ParFiniteElementSpace velocity_space(&mesh, &collection, dimension, Ordering::byVDIM);
   ParFiniteElementSpace pressure_space(&mesh, &pressure_collection);
   ManufacturedSolution exact(dimension, viscosity_scale, variable_viscosity, shift);
   if (linear_solver == LinearSolverType::MUMPS) { options.linear.backend = MumpsOptions{}; }
   else { iterative.method = linear_solver; options.linear.backend = iterative; }
   FlowProblem definition(velocity_space, pressure_space, exact.viscosity);
   definition.SetViscosityGradient(exact.viscosity_gradient);
   definition.SetIntegrationOrder(integration_order);
   definition.SetStabilization(discretization);
   definition.SetPressureGauge(gauge_policy);
   BoundaryConditions boundary;
   ParGridFunction velocity(&velocity_space), pressure(&pressure_space);
   for (int index = 0; index < mesh.bdr_attributes.Size(); ++index)
   {
      boundary.SetVelocity(mesh.bdr_attributes[index], exact.velocity);
   }
   definition.SetBoundaryConditions(boundary);
   for (int attribute : mesh.attributes) { definition.SetBodyForce(attribute, exact.force); }
   FlowSolver solver(definition, options);
   MPI_Barrier(MPI_COMM_WORLD);
   const double started = MPI_Wtime();
   const auto info = solver.Solve(velocity, pressure);
   const double elapsed = MPI_Wtime() - started;
   double seconds = 0.0;
   MPI_Allreduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
   const StokesErrors errors = ComputeErrors(velocity, pressure, exact,
                                             info.integration_order + 4);
   if (Mpi::Root())
   {
      out << std::setprecision(12)
          << "RESULT method=" << method_name << " element=" << element << " dim=" << dimension << " order=" << order
          << " pressure_order=" << pressure_order
          << " geometry_order=" << info.geometry_order << " integration_order=" << info.integration_order
          << " velocity_dofs=" << velocity_space.GlobalTrueVSize()
          << " pressure_dofs=" << pressure_space.GlobalTrueVSize()
          << " cells=" << cells << " solver=" << solver_name
          << " preconditioner=" << preconditioner_name
          << " pressure_gauge=" << PressureGaugeName(gauge_policy)
          << " pressure_nullspace_detected=" << info.pressure_nullspace_detected
          << " pressure_gauge_active=" << info.gauge_multiplier.has_value()
          << " p1_smoothing_steps=" << iterative.p1_smoothing_steps
          << " variable_mu=" << variable_viscosity << " shift=" << shift
          << " amg_elasticity=" << (linear_solver != LinearSolverType::MUMPS)
          << " interp_refine=" << iterative.amg_elasticity_interp_refine
          << " velocity_l2=" << errors.velocity_l2 << " velocity_h1=" << errors.velocity_h1
          << " pressure_l2=" << errors.pressure_l2 << " divergence_l2=" << errors.divergence_l2
          << " residual=" << info.relative_residual << " iterations=" << info.linear_iterations
          << " seconds=" << seconds << " assembly_seconds=" << info.assembly_seconds
          << " assembly_performed=" << info.assembly_performed
          << " preconditioner_setup_seconds=" << (linear_solver != LinearSolverType::MUMPS ? info.setup_seconds : 0.0)
          << " direct_setup_seconds=" << (linear_solver == LinearSolverType::MUMPS ? info.setup_seconds : 0.0)
          << " linear_solve_seconds=" << info.solve_seconds
          << " finalization_seconds=" << info.finalization_seconds << '\n';
   }
   if (paraview)
   {
      ParaViewDataCollection output("stokes", &mesh);
      output.SetPrefixPath(output_directory);
      output.SetLevelsOfDetail(order);
      output.SetHighOrderOutput(true);
      output.RegisterField("velocity", const_cast<ParGridFunction*>(&velocity));
      output.RegisterField("pressure", const_cast<ParGridFunction*>(&pressure));
      output.Save();
   }
   return info.converged ? 0 : 2;
}
