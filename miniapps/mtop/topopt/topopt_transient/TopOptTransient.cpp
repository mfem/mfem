// =============================================================================
// Transient Topology Optimization Driver
// =============================================================================
//
// Minimizes wave amplitude in a protected subdomain by optimizing the material
// distribution of a linear-elastodynamic domain with absorbing boundaries:
//
//   minimize   J(rho) = int_0^T int_Omega_hat |u(t)|^2 dx dt
//   subject to M(rho) u'' + C u' + K(rho) u = f(t),   u(0)=u'(0)=0
//              (1/V*) int rho dx - 1 <= 0,   0 <= rho <= 1
//
// Pipeline per MMA iteration:
//   1. raw control density rho (L2) -> Helmholtz filter -> rho_tilde (H1),
//   2. rho_tilde drives SIMP mass/stiffness coefficients,
//   3. DesignObjectiveAdjointGradient runs the RK4 forward sweep (J) and the
//      discrete adjoint backward sweep with stage-consistent design
//      sensitivity, returning dJ/drho (already filter-transposed),
//   4. MMA updates rho subject to the volume constraint + move limits.
//
// The adjoint + design gradient are verified in test_adjoint_verification.
//
// COMPILE:
//   make TopOptTransient -j8
//
// RUN (short wiring smoke test):
//   mpirun -np 4 ./TopOptTransient -r 0 -o 1 -tf 0.3 -dt 1e-4 -vf 0.5 \
//   -fr 0.03 -mi 150 -mv 0.2 -pv
//
// =============================================================================

#include "mfem.hpp"
#include "ElastodynamicsSolver.hpp"
#include "ProblemSpecification.hpp"
#include "../../pde_filter.hpp"
#include "../../mma/MMA_MFEM.hpp"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>

using namespace std;
using namespace mfem;

namespace
{

string ToLower(const char *text)
{
   string value(text ? text : "");
   transform(value.begin(), value.end(), value.begin(),
             [](unsigned char c) { return static_cast<char>(tolower(c)); });
   return value;
}

unique_ptr<HypreParVector> AssembleVolumeWeights(ParFiniteElementSpace &fes,
                                                 real_t &domain_volume)
{
   ConstantCoefficient one(1.0);
   ParLinearForm volume_form(&fes);
   volume_form.AddDomainIntegrator(new DomainLFIntegrator(one));
   volume_form.Assemble();

   unique_ptr<HypreParVector> weights(volume_form.ParallelAssemble());

   const real_t local_volume = weights->Sum();
   MPI_Allreduce(&local_volume, &domain_volume, 1,
                 MPITypeMap<real_t>::mpi_type, MPI_SUM, fes.GetComm());

   return weights;
}

Array<int> MakeBoundaryMarker(const ParMesh &pmesh, const Array<int> &attrs)
{
   Array<int> marker(pmesh.bdr_attributes.Max());
   marker = 0;

   for (int i = 0; i < attrs.Size(); i++)
   {
      const int attr = attrs[i];
      if (attr >= 1 && attr <= marker.Size())
      {
         marker[attr - 1] = 1;
      }
   }

   return marker;
}

bool InitializeDesign(ParGridFunction &rho, const char *design_init,
                      real_t vol_frac, real_t x_max, real_t y_max)
{
   const string mode = ToLower(design_init);

   if (mode == "uniform")
   {
      rho = vol_frac;
      return true;
   }

   if (mode == "solid")
   {
      rho = 1.0;
      return true;
   }

   if (mode == "void")
   {
      rho = 0.0;
      return true;
   }

   if (mode == "gaussian")
   {
      GaussianDesignCoefficient gaussian(x_max/2.0, y_max/2.0,
                                         0.25*x_max, 0.25*y_max,
                                         0.10, 1.0);
      rho.ProjectCoefficient(gaussian);
      return true;
   }

   return false;
}

} // namespace

int main(int argc, char *argv[])
{
   Mpi::Init();
   Hypre::Init();

   const MPI_Comm comm = MPI_COMM_WORLD;
   const int myid = Mpi::WorldRank();

   Device device("cpu");

   TransientTopOptConfig cfg;
   bool paraview = false;
   // Mass discretization for the forward/adjoint sweeps. Both give a
   // gradient-consistent dJ/drho (the design sensitivity differentiates whichever
   // mass matrix is used - see StageMassDesignLFIntegrator - and both are verified
   // by test_adjoint_verification). Consistent (CG+AMG) is the default reference;
   // lumped (diagonal, row-sum) is faster for explicit RK4. User's choice via flag.
   bool use_iterative_mass = true;
   const char *mesh_file = cfg.mesh_file.c_str();
   const char *design_init = "uniform";
   const char *problem_name = "wave";
   bool damping = true;   // -damp / -no-damp: apply the problem's damping

   OptionsParser args(argc, argv);
   args.AddOption(&problem_name, "-problem", "--problem",
                  "Forward problem: wave, cantilever-compliance, or "
                  "band-waveguide");
   args.AddOption(&damping, "-damp", "--damp", "-no-damp", "--no-damp",
                  "Apply the problem's damping (bulk + absorbing). -no-damp zeroes "
                  "all dissipation: free (Neumann) boundaries, conservative system.");
   args.AddOption(&cfg.ref_levels, "-r", "--refine", "Refinement level");
   args.AddOption(&cfg.order, "-o", "--order", "H1 finite element order");
   args.AddOption(&cfg.t_final, "-tf", "--t-final", "Final time");
   args.AddOption(&cfg.dt, "-dt", "--time-step", "Time step");
   args.AddOption(&cfg.vol_frac, "-vf", "--vol-frac", "Target volume fraction");
   args.AddOption(&cfg.filter_radius, "-fr", "--filter-radius",
                  "Helmholtz filter radius");
   args.AddOption(&cfg.max_it, "-mi", "--max-it", "Max MMA iterations");
   args.AddOption(&cfg.move, "-mv", "--move", "MMA move limit");
   args.AddOption(&cfg.change_tol, "-tol", "--tol",
                  "Stop early when the L1 design change drops below this");
   args.AddOption(&design_init, "-init", "--design-init",
                  "Initial design: uniform, solid, void, gaussian");
   args.AddOption(&mesh_file, "-mesh", "--mesh-file", "Mesh file");
   args.AddOption(&paraview, "-pv", "--paraview", "-no-pv",
                  "--no-paraview", "Write ParaView output");
   args.AddOption(&use_iterative_mass, "-iterative-mass", "--iterative-mass",
                  "-lumped-mass", "--lumped-mass",
                  "Mass solver: consistent CG+AMG (default) or faster lumped. "
                  "Both are gradient-consistent (verified).");
   args.Parse();

   if (!args.Good())
   {
      if (myid == 0) { args.PrintUsage(cout); }
      return 1;
   }
   cfg.mesh_file = mesh_file;

   if (cfg.order < 1)
   {
      if (myid == 0) { cerr << "Error: -o/--order must be at least 1.\n"; }
      return 1;
   }
   if (cfg.dt <= 0.0 || cfg.t_final <= 0.0)
   {
      if (myid == 0)
      {
         cerr << "Error: -dt and -tf must both be positive.\n";
      }
      return 1;
   }
   if (cfg.vol_frac <= 0.0 || cfg.vol_frac > 1.0)
   {
      if (myid == 0)
      {
         cerr << "Error: -vf/--vol-frac must be in (0, 1].\n";
      }
      return 1;
   }
   if (cfg.max_it < 1)
   {
      if (myid == 0) { cerr << "Error: -mi/--max-it must be >= 1.\n"; }
      return 1;
   }

   unique_ptr<TransientTopOptProblem> problem_owner;
   const string problem_sel = ToLower(problem_name);
   if (problem_sel == "cantilever-compliance")
   {
      problem_owner = make_unique<CantileverComplianceProblem>(cfg);
   }
   else if (problem_sel == "band-waveguide")
   {
      problem_owner = make_unique<BandWaveguideProblem>(cfg);
   }
   else if (problem_sel == "wave")
   {
      problem_owner = make_unique<WaveShieldingProblem>(cfg);
   }
   else
   {
      if (myid == 0)
      {
         cerr << "Error: unknown -problem '" << problem_name
              << "'. Use wave, cantilever-compliance, or band-waveguide.\n";
      }
      return 1;
   }
   TransientTopOptProblem &problem = *problem_owner;

   ostringstream problem_errors;
   if (!problem.Validate(problem_errors))
   {
      if (myid == 0) { cerr << problem_errors.str(); }
      return 1;
   }

   if (myid == 0) { args.PrintOptions(cout); }

   // The problem builds its own coarse mesh (file-based by default; generated
   // geometry for e.g. the cantilever).
   Mesh mesh = problem.CreateMesh();
   const int dim = mesh.Dimension();

   for (int l = 0; l < problem.GetRefinementLevel(); l++)
   {
      mesh.UniformRefinement();
   }

   ParMesh pmesh(comm, mesh);
   mesh.Clear();

   const BoundaryLoadSpec &load_spec = problem.GetBoundaryLoad();
   if (load_spec.direction.Size() != dim)
   {
      if (myid == 0)
      {
         cerr << "Error: boundary load direction dimension ("
              << load_spec.direction.Size()
              << ") does not match mesh dimension (" << dim << ").\n";
      }
      return 1;
   }

   H1_FECollection state_fec(problem.GetOrder(), dim);
   H1_FECollection filter_fec(problem.GetOrder(), dim);
   const int control_order = max(0, problem.GetOrder() - 1);
   L2_FECollection control_fec(control_order, dim, BasisType::GaussLobatto);

   ParFiniteElementSpace state_fes(&pmesh, &state_fec, dim);
   ParFiniteElementSpace filter_fes(&pmesh, &filter_fec);
   ParFiniteElementSpace control_fes(&pmesh, &control_fec);

   const HYPRE_BigInt state_dofs = state_fes.GlobalTrueVSize();
   const HYPRE_BigInt filter_dofs = filter_fes.GlobalTrueVSize();
   const HYPRE_BigInt control_dofs = control_fes.GlobalTrueVSize();

   ParGridFunction rho(&control_fes);
   ParGridFunction rho_tilde(&filter_fes);

   real_t ref_x_max = 0.0, ref_y_max = 0.0;
   problem.GetReferenceDomainExtents(ref_x_max, ref_y_max);

   if (!InitializeDesign(rho, design_init, problem.GetVolumeFraction(),
                         ref_x_max, ref_y_max))
   {
      if (myid == 0)
      {
         cerr << "Error: unknown -init value '" << design_init
              << "'. Use uniform, solid, void, or gaussian.\n";
      }
      return 1;
   }
   rho_tilde = 0.0;

   toopt::PDEFilterOptions filter_opts;
   filter_opts.filter_radius = problem.GetFilterRadius();
   toopt::PDEFilter filter(filter_fes, control_fes, filter_opts);
   filter.Assemble();
   filter.Mult(rho, rho_tilde);

   real_t domain_volume = 0.0;
   unique_ptr<HypreParVector> volume_weights =
      AssembleVolumeWeights(control_fes, domain_volume);
   const real_t target_volume = problem.GetVolumeFraction() * domain_volume;

   // Material and problem constants (match test_adjoint_verification).
   const MaterialParams &mat = problem.GetMaterialParams();

   // Sponge-layer damping coefficient + absorbing-boundary impedance, assembled
   // by the problem from the material and damping parameters.
   unique_ptr<DampingField> damping_field = problem.CreateDampingField(damping);
   Coefficient &gamma_coef = damping_field->GetCoefficient();
   const real_t impedance = damping_field->GetImpedance();

   Array<int> absorbing_bdr_attributes;
   problem.GetAbsorbingBoundaryAttributes(absorbing_bdr_attributes);
   Array<int> exterior_bdr_attr =
      MakeBoundaryMarker(pmesh, absorbing_bdr_attributes);

   Array<int> essential_bdr_attributes;
   problem.GetEssentialBoundaryAttributes(essential_bdr_attributes);
   Array<int> essential_bdr_attr =
      MakeBoundaryMarker(pmesh, essential_bdr_attributes);

   unique_ptr<TimeIntegratedObjective> objective =
      problem.CreateObjective(&state_fes, comm);
   unique_ptr<VectorCoefficient> load_coef =
      problem.CreateBoundaryLoadCoefficient();

   const int num_steps =
      max(1, static_cast<int>(ceil(problem.GetFinalTime() / problem.GetTimeStep())));
   const real_t dt_eff = problem.GetFinalTime() / num_steps;

   if (myid == 0)
   {
      cout << "\n=== Transient TopOpt (MMA) ===\n";
      cout << "Mesh: " << problem.GetMeshFile() << "\n";
      cout << "Refinement levels: " << problem.GetRefinementLevel() << "\n";
      cout << "State DOFs:   " << state_dofs << "\n";
      cout << "Filter DOFs:  " << filter_dofs << " (H1 rho_tilde)\n";
      cout << "Control DOFs: " << control_dofs << " (L2 rho)\n";
      cout << "Target volume fraction: " << problem.GetVolumeFraction() << "\n";
      cout << "Filter radius: " << problem.GetFilterRadius() << "\n";
      cout << "Time interval: [0, " << problem.GetFinalTime() << "],  steps: " << num_steps
           << ",  dt_eff: " << dt_eff << "\n";
      cout << "Max MMA iterations: " << problem.GetMaxIterations()
           << ",  move limit: " << problem.GetMoveLimit()
           << ",  stop tol (L1 dRho): " << problem.GetChangeTolerance() << "\n";
   }

   // --- MMA setup -----------------------------------------------------------
   const int n = control_fes.GetTrueVSize();
   const int num_con = 1;  // single volume constraint

   Vector rho_tv(n), rho_old(n);
   rho.GetTrueDofs(rho_tv);

   Vector dJ_drho(n);
   Vector fival(num_con);

   // Volume-constraint gradient d/drho [ (1/V*) int rho - 1 ] = w / V*
   // (FE volume weights, not a constant vector).
   Vector dvol(*volume_weights);
   dvol /= target_volume;
   Vector dfidx[num_con];
   dfidx[0] = dvol;

   mfem_mma::MMAOptimizerParallel mma(comm, n, num_con, rho_tv);
   mma.SetAsymptotes(0.5, 0.7, 1.2);

   Vector rho_min(n), rho_max(n);

   ParaViewDataCollection paraview_dc("TopOptTransient", &pmesh);
   if (paraview)
   {
      paraview_dc.SetPrefixPath("ParaView");
      paraview_dc.SetLevelsOfDetail(problem.GetOrder());
      paraview_dc.SetDataFormat(VTKFormat::BINARY);
      paraview_dc.SetHighOrderOutput(true);
      paraview_dc.RegisterField("rho", &rho);
      paraview_dc.RegisterField("rho_tilde", &rho_tilde);
   }

   ofstream history;
   if (myid == 0)
   {
      history.open("optimization_history.txt");
      history << "# iter    J                 vol_frac      g\n";
   }

   // --- Optimization loop ---------------------------------------------------
   MassSolverType mass_solver = use_iterative_mass ?
                                MassSolverType::ITERATIVE : MassSolverType::LUMPED;

   // Bundle the invariant setup once; the loop evaluates a design in one call.
   TransientDesignSolver design_solver(
      state_fes, filter_fes, control_fes, filter, gamma_coef,
      exterior_bdr_attr, essential_bdr_attr, *objective, mat, load_spec,
      *load_coef, impedance, num_steps, dt_eff, mass_solver, rho, rho_tilde);

   GridFunctionCoefficient rho_cf(&rho);
   int k = 0;
   real_t iterationError = 1.0;
   for (; k < problem.GetMaxIterations() &&
          iterationError > problem.GetChangeTolerance(); k++)
   {
      design_solver.FilterFSolve(rho_tv);              // forward filter:  rho -> rho_tilde
      const real_t J = design_solver.PhysicsFSolve(k); // forward physics: -> J
      design_solver.PhysicsASolve();                   // adjoint physics: -> dJ/drho_tilde
      design_solver.FilterASolve(dJ_drho);             // adjoint filter:  -> dJ/drho

      // Volume constraint and current fraction.
      const real_t cur_volume = InnerProduct(comm, *volume_weights, rho_tv);
      const real_t cur_vol_frac = cur_volume / domain_volume;
      fival(0) = cur_volume / target_volume - 1.0;

      // Box constraints with move limits.
      rho_old = rho_tv;
      for (int i = 0; i < n; i++)
      {
         rho_min[i] = max(real_t(0.0), rho_tv[i] - problem.GetMoveLimit());
         rho_max[i] = min(real_t(1.0), rho_tv[i] + problem.GetMoveLimit());
      }

      // MMA outer iteration (minimizes J subject to fival <= 0).
      mma.Update(rho_tv, dJ_drho, J, fival, dfidx, rho_min, rho_max);
      rho.SetFromTrueDofs(rho_tv);

      // Design change (L1 norm, matches ElastTopOpt_static) for the
      // early-stop test and progress monitoring: iterationError = int |dRho|.
      ParGridFunction rho_old_gf(&control_fes);
      rho_old_gf.SetFromTrueDofs(rho_old);
      iterationError = rho_old_gf.ComputeL1Error(rho_cf);

      if (myid == 0)
      {
         cout << "it " << setw(3) << k + 1
              << "   J = " << scientific << setprecision(6) << J
              << "   vol = " << fixed << setprecision(4) << cur_vol_frac
              << "   g = " << scientific << setprecision(3) << fival(0)
              << "   dRho(L1) = " << setprecision(3) << iterationError << "\n";
         history << setw(5) << k + 1 << "  "
                 << scientific << setprecision(8) << J << "  "
                 << fixed << setprecision(6) << cur_vol_frac << "  "
                 << scientific << setprecision(6) << fival(0) << "\n";
      }

      if (paraview)
      {
         paraview_dc.SetCycle(k + 1);
         paraview_dc.SetTime(k + 1);
         paraview_dc.Save();
      }
   }

   if (myid == 0)
   {
      history.close();
      cout << "\nOptimization stopped after " << k << " iterations"
            << " (final L1 design change = " << scientific << setprecision(3)
            << iterationError << ", tol = "
            << problem.GetChangeTolerance() << ").\n";
      if (paraview)
      {
         cout << "ParaView output: ParaView/TopOptTransient.pvd\n";
      }
      cout << "History: optimization_history.txt\n";
   }

   return 0;
}
