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
#include "OptimizationCheckpoint.hpp"
#include "../../pde_filter.hpp"
#include "../../mma/MMA_MFEM.hpp"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <sys/stat.h>

using namespace std;
using namespace mfem;

namespace
{

// =============================================================================
// OUTPUT DIRECTORY HELPER: Timestamp + SLURM Job ID
// =============================================================================
string GenerateOutputDirectory()
{
   // Get current time
   time_t now = time(nullptr);
   struct tm* tm_info = localtime(&now);

   char timestamp[64];
   strftime(timestamp, sizeof(timestamp), "%Y%m%d_%H%M%S", tm_info);

   // Get SLURM job ID (if available)
   const char* job_id_env = std::getenv("SLURM_JOB_ID");
   string job_id = job_id_env ? job_id_env : "local";

   ostringstream dirname;
   dirname << timestamp << "_job" << job_id;
   return dirname.str();
}

string ToLower(const char *text)
{
   string value(text ? text : "");
   transform(value.begin(), value.end(), value.begin(),
             [](unsigned char c) { return static_cast<char>(tolower(c)); });
   return value;
}

unique_ptr<HypreParVector> AssembleVolumeWeights(ParFiniteElementSpace &fes,
                                                 real_t &domain_volume,
                                                 Coefficient *active_region = nullptr)
{
   // If active_region is provided, integrate only over active (designable) region.
   // Otherwise integrate over entire domain.
   Coefficient *integrand = active_region ? active_region : new ConstantCoefficient(1.0);

   ParLinearForm volume_form(&fes);
   volume_form.AddDomainIntegrator(new DomainLFIntegrator(*integrand));
   volume_form.Assemble();

   unique_ptr<HypreParVector> weights(volume_form.ParallelAssemble());

   const real_t local_volume = weights->Sum();
   MPI_Allreduce(&local_volume, &domain_volume, 1,
                 MPITypeMap<real_t>::mpi_type, MPI_SUM, fes.GetComm());

   if (!active_region)
   {
      delete integrand;
   }

   return weights;
}

// =============================================================================
// Active/Passive DOF Management
// =============================================================================
// Identifies which DOFs are in active (designable) vs passive (fixed) regions.
// A DOF is passive if its associated element has ANY integration point where
// the passive_region_coef evaluates to > 0.5.
void IdentifyActivePassiveDOFs(ParFiniteElementSpace &fes,
                                Coefficient &passive_region_coef,
                                Array<int> &active_tdof_list,
                                Array<int> &passive_tdof_list,
                                ParGridFunction &passive_marker)
{
   const int n_local = fes.GetVSize();
   Array<int> local_is_passive(n_local);
   local_is_passive = 0;

   // Mark DOFs that touch passive regions
   ParMesh *pmesh = fes.GetParMesh();
   for (int e = 0; e < pmesh->GetNE(); e++)
   {
      const FiniteElement *fe = fes.GetFE(e);
      ElementTransformation *T = fes.GetElementTransformation(e);
      const IntegrationRule &ir = IntRules.Get(fe->GetGeomType(), 2 * fe->GetOrder());

      bool element_is_passive = false;
      for (int q = 0; q < ir.GetNPoints(); q++)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         T->SetIntPoint(&ip);
         if (passive_region_coef.Eval(*T, ip) > 0.5)
         {
            element_is_passive = true;
            break;
         }
      }

      if (element_is_passive)
      {
         Array<int> dofs;
         fes.GetElementVDofs(e, dofs);
         for (int i = 0; i < dofs.Size(); i++)
         {
            int dof = dofs[i];
            if (dof < 0) { dof = -1 - dof; }
            local_is_passive[dof] = 1;
         }
      }
   }

   // Convert to true DOF markers
   const int n_true = fes.GetTrueVSize();
   Array<int> true_is_passive(n_true);
   true_is_passive = 0;

   const SparseMatrix *P = fes.GetRestrictionMatrix();
   if (P)
   {
      // Has parallel restriction: map local to true DOFs
      for (int i = 0; i < n_local; i++)
      {
         if (local_is_passive[i])
         {
            const int *cols = P->GetRowColumns(i);
            const int ncols = P->RowSize(i);
            for (int j = 0; j < ncols; j++)
            {
               true_is_passive[cols[j]] = 1;
            }
         }
      }
   }
   else
   {
      // Serial or no restriction: true DOFs = local DOFs
      for (int i = 0; i < n_true; i++)
      {
         true_is_passive[i] = local_is_passive[i];
      }
   }

   // Build active and passive lists
   active_tdof_list.SetSize(0);
   passive_tdof_list.SetSize(0);
   for (int i = 0; i < n_true; i++)
   {
      if (true_is_passive[i])
      {
         passive_tdof_list.Append(i);
      }
      else
      {
         active_tdof_list.Append(i);
      }
   }

   // Store marker in grid function for visualization
   for (int i = 0; i < n_local; i++)
   {
      passive_marker(i) = local_is_passive[i];
   }
}

// Map from reduced active design vector to full grid function
void MapActiveToFull(const Vector &rho_active,
                     const Array<int> &active_tdof_list,
                     const Array<int> &passive_tdof_list,
                     real_t passive_value,
                     Vector &rho_full_tv)
{
   // Set active DOFs from reduced vector
   for (int i = 0; i < active_tdof_list.Size(); i++)
   {
      rho_full_tv[active_tdof_list[i]] = rho_active[i];
   }

   // Set passive DOFs to fixed value
   for (int i = 0; i < passive_tdof_list.Size(); i++)
   {
      rho_full_tv[passive_tdof_list[i]] = passive_value;
   }
}

// Map from full grid function to reduced active design vector
void MapFullToActive(const Vector &rho_full_tv,
                     const Array<int> &active_tdof_list,
                     Vector &rho_active)
{
   for (int i = 0; i < active_tdof_list.Size(); i++)
   {
      rho_active[i] = rho_full_tv[active_tdof_list[i]];
   }
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
   int pv_freq = 0;   // ParaView design-output interval; 0 = auto (~100 frames)
   // Mass discretization for the forward/adjoint sweeps. Both give a
   // gradient-consistent dJ/drho (the design sensitivity differentiates whichever
   // mass matrix is used - see StageMassDesignLFIntegrator - and both are verified
   // by test_adjoint_verification). Consistent (CG+AMG) is the default reference;
   // lumped (diagonal, row-sum) is faster for explicit RK4. User's choice via flag.
   bool use_iterative_mass = true;
   const string default_mesh_file = cfg.mesh_file;
   const char *mesh_file = default_mesh_file.c_str();
   const char *design_init = "uniform";
   const char *problem_name = "wave";
   bool damping = true;   // -damp / -no-damp: apply the problem's damping
   // Carrier/pulse overrides (0 = keep the problem's default). Same problem can
   // then run cheap (low f, coarse mesh) locally and rich (high f) on HPC.
   real_t load_frequency = 0.0;
   real_t load_duration = 0.0;
   int num_checkpoints = -1;   // REVOLVE snapshot count; -1 = auto-size

   // === Checkpoint and output directory options ===
   const char *output_parent = nullptr;  // Parent directory for all outputs
   bool restart = false;                 // Restart from checkpoint?
   bool auto_checkpoint = true;          // Auto-save checkpoint each iteration

   OptionsParser args(argc, argv);
   args.AddOption(&problem_name, "-problem", "--problem",
                  "Forward problem: wave, cantilever-compliance, "
                  "band-waveguide, or spherical-bandgap");
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
   args.AddOption(&output_parent, "-out", "--output-parent",
                  "Parent directory for all outputs (ParaView, checkpoint, history). "
                  "If not specified, auto-generates: YYYYMMDD_HHMMSS_jobSLURM_ID");
   args.AddOption(&restart, "-restart", "--restart", "-no-restart", "--no-restart",
                  "Restart optimization from checkpoint in output-parent directory");
   args.AddOption(&auto_checkpoint, "-ckpt", "--checkpoint", "-no-ckpt", "--no-checkpoint",
                  "Enable automatic checkpointing at each iteration");
   args.AddOption(&pv_freq, "-pvf", "--paraview-freq",
                  "Save ParaView design output every N optimization iterations "
                  "(0 = auto: ~100 evenly spaced frames). The first and last "
                  "iterations are always saved.");
   args.AddOption(&use_iterative_mass, "-iterative-mass", "--iterative-mass",
                  "-lumped-mass", "--lumped-mass",
                  "Mass solver: consistent CG+AMG (default) or faster lumped. "
                  "Both are gradient-consistent (verified).");
   args.AddOption(&load_frequency, "-freq", "--load-frequency",
                  "Carrier frequency override for modulated/harmonic loads "
                  "(0 = problem default). Resolving the carrier needs "
                  "mesh size <~ c_p/(7 f).");
   args.AddOption(&load_duration, "-dur", "--load-duration",
                  "Pulse duration override (0 = problem default; the "
                  "spherical problem otherwise keeps ~3 carrier cycles).");
   args.AddOption(&num_checkpoints, "-nchk", "--num-checkpoints",
                  "REVOLVE trajectory checkpoints per forward sweep "
                  "(-1 = auto). More checkpoints = more memory, fewer "
                  "forward recomputations in the adjoint.");
   args.Parse();

   if (!args.Good())
   {
      if (myid == 0) { args.PrintUsage(cout); }
      return 1;
   }
   cfg.mesh_file = mesh_file;
   cfg.mesh_file_is_user = (cfg.mesh_file != default_mesh_file);
   if (load_frequency > 0.0)
   {
      cfg.boundary_load.frequency = load_frequency;
      cfg.load_frequency_is_user = true;
   }
   if (load_duration > 0.0)
   {
      cfg.boundary_load.duration = load_duration;
      cfg.load_duration_is_user = true;
   }

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
   else if (problem_sel == "spherical-bandgap")
   {
      problem_owner = make_unique<SphericalBandGapProblem>(cfg);
   }
   else
   {
      if (myid == 0)
      {
         cerr << "Error: unknown -problem '" << problem_name
              << "'. Use wave, cantilever-compliance, band-waveguide, or spherical-bandgap.\n";
      }
      return 1;
   }
   TransientTopOptProblem &problem = *problem_owner;

   ostringstream problem_errors;
   const bool problem_valid = problem.Validate(problem_errors);
   // Print whatever the problem reported: hard errors on failure, but also
   // non-fatal warnings (e.g. -tf too short for the pulse to reach the
   // objective region) when validation succeeds.
   if (myid == 0 && !problem_errors.str().empty())
   {
      cerr << problem_errors.str();
   }
   if (!problem_valid) { return 1; }

   if (myid == 0) { args.PrintOptions(cout); }

   // =========================================================================
   // SETUP OUTPUT DIRECTORY STRUCTURE (timestamp_jobID)
   // =========================================================================
   string output_parent_dir;
   if (output_parent)
   {
      output_parent_dir = output_parent;
   }
   else
   {
      // Auto-generate timestamped directory
      if (myid == 0)
      {
         output_parent_dir = GenerateOutputDirectory();
      }
      // Broadcast to all ranks
      char dir_buf[256];
      if (myid == 0)
      {
         strncpy(dir_buf, output_parent_dir.c_str(), 255);
         dir_buf[255] = '\0';
      }
      MPI_Bcast(dir_buf, 256, MPI_CHAR, 0, comm);
      if (myid != 0)
      {
         output_parent_dir = string(dir_buf);
      }
   }

   // Create the output directory (rank 0 acts; the verdict is broadcast so a
   // failure exits ALL ranks - a rank-0-only return here would leave the
   // other ranks hanging in a barrier).
   bool outdir_ok = true;
   if (myid == 0)
   {
      struct stat st;
      if (stat(output_parent_dir.c_str(), &st) != 0)
      {
         outdir_ok = (mkdir(output_parent_dir.c_str(), 0755) == 0);
         if (!outdir_ok)
         {
            cerr << "ERROR: Failed to create output directory: "
                 << output_parent_dir << "\n"
                 << "       (parent directories must already exist)\n";
         }
      }
      else if (!restart)
      {
         cerr << "WARNING: Output directory already exists: "
              << output_parent_dir << "\n"
              << "         Use -restart to continue from checkpoint, or specify different -out\n";
      }
   }
   MPI_Bcast(&outdir_ok, 1, MPI_C_BOOL, 0, comm);
   if (!outdir_ok) { return 1; }

   // Subdirectories
   string paraview_dir = output_parent_dir + "/ParaView";
   string checkpoint_dir = output_parent_dir + "/optimization_checkpoint";
   string history_file = output_parent_dir + "/optimization_history.txt";

   if (myid == 0)
   {
      cout << "\n=== Output Configuration ===\n";
      cout << "Parent directory: " << output_parent_dir << "\n";
      cout << "ParaView output:  " << paraview_dir << "\n";
      cout << "Checkpoint:       " << checkpoint_dir << "\n";
      cout << "History file:     " << history_file << "\n";
      cout << "Restart mode:     " << (restart ? "YES" : "NO") << "\n";
      cout << "============================\n\n";
   }

   // The problem builds its own coarse mesh (file-based by default; generated
   // geometry for e.g. the cantilever).
   Mesh mesh = problem.CreateMesh();
   const int dim = mesh.Dimension();

   // Apply periodic boundary conditions in y-direction for band-waveguide
   if (problem_sel == "band-waveguide" && dim == 2)
   {
      real_t y_max = 0.0, x_max = 0.0;
      problem.GetReferenceDomainExtents(x_max, y_max);

      // Make mesh periodic in y-direction only
      // Use large x-translation to make x effectively non-periodic
      std::vector<Vector> translations = {
         Vector({1000.0, 0.0}),  // x: effectively non-periodic
         Vector({0.0, y_max})    // y: periodic with period = domain height
      };
      std::vector<int> v2v = mesh.CreatePeriodicVertexMapping(translations);
      mesh = Mesh::MakePeriodic(mesh, v2v);

      if (myid == 0)
      {
         cout << "Applied periodic boundary conditions in y-direction (y_period="
              << y_max << ")\n";
      }
   }

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

   // =========================================================================
   // CHECKPOINT LOAD (if restarting)
   // =========================================================================
   // Minimal restart: only the control density is restored and used as the
   // initial guess for a fresh MMA run (no optimizer internals - MMA rebuilds
   // its asymptotes within a couple of iterations). The iteration counter
   // continues so -mi budgets and the history file stay meaningful.
   OptimizationCheckpoint checkpoint(checkpoint_dir, comm);
   OptimizationCheckpointMetadata ckpt_meta;
   bool restarting = false;
   int start_iteration = 0;

   if (restart && checkpoint.Exists())
   {
      if (!checkpoint.ValidateCompatibility(cfg.ref_levels, cfg.order, ckpt_meta))
      {
         if (myid == 0)
         {
            cerr << "ERROR: Checkpoint exists but is incompatible. Exiting.\n";
            cerr << "       Remove -restart flag to start fresh run.\n";
         }
         return 1;
      }

      Vector rho_tv_restart(control_fes.GetTrueVSize());
      if (!checkpoint.Load(rho_tv_restart))
      {
         if (myid == 0)
         {
            cerr << "ERROR: Failed to load checkpoint design. Exiting.\n";
         }
         return 1;
      }
      rho.SetFromTrueDofs(rho_tv_restart);
      restarting = true;
      start_iteration = ckpt_meta.iteration + 1;

      if (myid == 0)
      {
         cout << "\n=== RESTARTING FROM CHECKPOINT ===\n";
         cout << "Previous run stopped at iteration: " << ckpt_meta.iteration << "\n";
         cout << "Objective value: " << scientific << ckpt_meta.objective << "\n";
         cout << "Volume fraction: " << fixed << ckpt_meta.volume_fraction << "\n";
         cout << "Design restored as initial guess; MMA restarts fresh.\n";
         cout << "Resuming from iteration " << start_iteration << "\n";
         cout << "==================================\n\n";
      }
   }
   else if (restart && !checkpoint.Exists())
   {
      if (myid == 0)
      {
         cerr << "ERROR: -restart specified but no checkpoint found in: "
              << checkpoint_dir << "\n";
      }
      return 1;
   }

   // =========================================================================
   // INITIALIZE DESIGN (skip if restarting - already loaded from checkpoint)
   // =========================================================================
   real_t ref_x_max = 0.0, ref_y_max = 0.0;
   problem.GetReferenceDomainExtents(ref_x_max, ref_y_max);

   if (!restarting)
   {
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
   }
   rho_tilde = 0.0;

   toopt::PDEFilterOptions filter_opts;
   filter_opts.filter_radius = problem.GetFilterRadius();
   toopt::PDEFilter filter(filter_fes, control_fes, filter_opts);
   filter.Assemble();
   filter.Mult(rho, rho_tilde);

   // --- Active/Passive Region Setup -----------------------------------------
   // Check if problem defines passive (non-designable) regions
   unique_ptr<Coefficient> passive_region_coef = problem.CreatePassiveRegionCoefficient();
   ConstantCoefficient one_coef(1.0);
   unique_ptr<Coefficient> active_region_coef;
   Array<int> active_tdof_list, passive_tdof_list;
   ParGridFunction passive_marker(&control_fes);
   // Passive regions frozen at the problem's reference density (default:
   // the volume fraction; the spherical problem pins 0.5 regardless of -vf).
   const real_t passive_rho_value = problem.GetPassiveDensity();

   if (passive_region_coef)
   {
      // Active region = 1 - passive region
      active_region_coef = std::make_unique<SumCoefficient>(
         one_coef, *passive_region_coef, 1.0, -1.0);

      // Identify which DOFs are active vs passive
      IdentifyActivePassiveDOFs(control_fes, *passive_region_coef,
                                active_tdof_list, passive_tdof_list,
                                passive_marker);

      // Passive DOFs are already initialized to vol_frac by InitializeDesign.
      // We just need to ensure they stay at that value throughout optimization.
      // No need to re-set them here.

      // Get global DOF counts across all ranks
      HYPRE_BigInt global_control_dofs = control_fes.GlobalTrueVSize();
      HYPRE_BigInt local_active = active_tdof_list.Size();
      HYPRE_BigInt local_passive = passive_tdof_list.Size();
      HYPRE_BigInt global_active = 0, global_passive = 0;
      MPI_Allreduce(&local_active, &global_active, 1,
                    HYPRE_MPI_BIG_INT, MPI_SUM, comm);
      MPI_Allreduce(&local_passive, &global_passive, 1,
                    HYPRE_MPI_BIG_INT, MPI_SUM, comm);

      if (myid == 0)
      {
         cout << "Passive regions defined:\n";
         cout << "  Total control DOFs: " << global_control_dofs << "\n";
         cout << "  Active DOFs:  " << global_active << "\n";
         cout << "  Passive DOFs: " << global_passive << " (fixed at rho="
              << passive_rho_value << ")\n";
      }
   }

   // Assemble volume weights over active region only
   real_t domain_volume = 0.0;
   unique_ptr<HypreParVector> volume_weights =
      AssembleVolumeWeights(control_fes, domain_volume,
                           active_region_coef.get());
   const real_t target_volume = problem.GetVolumeFraction() * domain_volume;

   // Material and problem constants (match test_adjoint_verification).
   const MaterialParams &mat = problem.GetMaterialParams();

   // Carrier-resolution report: an under-resolved carrier wave silently turns
   // the forward physics into numerical dispersion, so make the elements-per-
   // wavelength budget visible up front. (Collective: GetCharacteristics.)
   if (load_spec.time_profile == LoadTimeProfile::MODULATED_GAUSSIAN ||
       load_spec.time_profile == LoadTimeProfile::HARMONIC)
   {
      real_t h_min, h_max, kappa_min, kappa_max;
      pmesh.GetCharacteristics(h_min, h_max, kappa_min, kappa_max);
      const real_t c_p = sqrt((mat.lambda0 + 2.0 * mat.mu0) / mat.rho0);
      const real_t lambda_p = c_p / load_spec.frequency;
      if (myid == 0)
      {
         cout << "Carrier: f = " << load_spec.frequency
              << ", c_p = " << c_p << ", lambda_p = " << lambda_p
              << ", mesh h = [" << h_min << ", " << h_max << "]"
              << " -> elements/wavelength = [" << lambda_p / h_max
              << ", " << lambda_p / h_min << "]\n";
         if (lambda_p < 4.0 * h_max)
         {
            cerr << "WARNING: fewer than ~4 elements per P-wavelength in the "
                 << "coarsest cells - the carrier will be strongly dispersed. "
                 << "Refine the mesh or lower -freq (need h <~ lambda_p/7 = "
                 << lambda_p / 7.0 << " where the wave must propagate).\n";
         }
      }
   }

   // Sponge-layer damping coefficient + absorbing-boundary impedance, assembled
   // by the problem from the material and damping parameters.
   unique_ptr<DampingFieldBase> damping_field = problem.CreateDampingField(damping);
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
   // MMA works with active (designable) DOFs only
   const int n_full = control_fes.GetTrueVSize();
   const int n_active = passive_region_coef ? active_tdof_list.Size() : n_full;
   const int num_con = 1;  // single volume constraint

   Vector rho_tv_full(n_full);
   rho.GetTrueDofs(rho_tv_full);

   Vector rho_active(n_active);
   Vector rho_active_old(n_active);
   if (passive_region_coef)
   {
      MapFullToActive(rho_tv_full, active_tdof_list, rho_active);
   }
   else
   {
      rho_active = rho_tv_full;
   }

   // (When restarting, rho already holds the checkpointed design, so
   // rho_active was just filled from it above - nothing more to restore.)

   Vector dJ_drho_full(n_full);
   Vector dJ_drho_active(n_active);
   Vector fival(num_con);

   // Volume-constraint gradient d/drho [ (1/V*) int rho - 1 ] = w / V*
   // Extract only active DOF contributions
   Vector dvol_full(*volume_weights);
   dvol_full /= target_volume;
   Vector dvol_active(n_active);
   if (passive_region_coef)
   {
      MapFullToActive(dvol_full, active_tdof_list, dvol_active);
   }
   else
   {
      dvol_active = dvol_full;
   }
   Vector dfidx[num_con];
   dfidx[0] = dvol_active;

   mfem_mma::MMAOptimizerParallel mma(comm, n_active, num_con, rho_active);
   mma.SetAsymptotes(0.5, 0.7, 1.2);

   Vector rho_active_min(n_active), rho_active_max(n_active);

   ParaViewDataCollection paraview_dc("TopOptTransient", &pmesh);
   if (paraview)
   {
      paraview_dc.SetPrefixPath(paraview_dir);  // Use subdirectory in parent
      paraview_dc.SetLevelsOfDetail(problem.GetOrder());
      paraview_dc.SetDataFormat(VTKFormat::BINARY);
      paraview_dc.SetHighOrderOutput(true);
      paraview_dc.RegisterField("rho", &rho);
      paraview_dc.RegisterField("rho_tilde", &rho_tilde);
      if (passive_region_coef)
      {
         paraview_dc.RegisterField("passive_region", &passive_marker);
      }
   }

   ofstream history;
   if (myid == 0)
   {
      if (restarting)
      {
         // APPEND to existing history
         history.open(history_file, ios::app);
         history << "\n# === RESTART at iteration " << start_iteration << " ===\n";
      }
      else
      {
         // NEW history file
         history.open(history_file);
         history << "# Transient Topology Optimization History\n";
         history << "# Output directory: " << output_parent_dir << "\n";
         history << "# Problem: " << problem_name << "\n";
         history << "#\n";
         history << "# iter    J                 vol_frac      g\n";
      }
   }

   // --- Optimization loop ---------------------------------------------------
   MassSolverType mass_solver = use_iterative_mass ?
                                MassSolverType::ITERATIVE : MassSolverType::LUMPED;

   // Bundle the invariant setup once; the loop evaluates a design in one call.
   TransientDesignSolver design_solver(
      state_fes, filter_fes, control_fes, filter, gamma_coef,
      exterior_bdr_attr, essential_bdr_attr, *objective, mat, load_spec,
      *load_coef, impedance, num_steps, dt_eff, mass_solver, rho, rho_tilde,
      num_checkpoints);

   GridFunctionCoefficient rho_cf(&rho);
   int k = start_iteration;  // Start from checkpoint iteration (or 0 if fresh)
   real_t iterationError = 1.0;   // fresh MMA on restart -> fresh stop test

   // ParaView design snapshots: cap the number of files written to the shared
   // filesystem. Saving every iteration exhausts the inode quota (each Save is
   // ~1 directory + one .vtu per MPI rank). Default to ~100 evenly spaced
   // frames; -pvf overrides the interval. First/last iterations always saved.
   const int pv_save_interval =
      (pv_freq > 0) ? pv_freq : max(1, problem.GetMaxIterations() / 100);

   for (; k < problem.GetMaxIterations() &&
          iterationError > problem.GetChangeTolerance(); k++)
   {
      // Map active design to full for forward solve
      if (passive_region_coef)
      {
         MapActiveToFull(rho_active, active_tdof_list, passive_tdof_list,
                        passive_rho_value, rho_tv_full);
         rho.SetFromTrueDofs(rho_tv_full);
      }
      else
      {
         rho.SetFromTrueDofs(rho_active);
      }

      design_solver.FilterFSolve(rho_tv_full);              // forward filter:  rho -> rho_tilde
      const real_t J = design_solver.PhysicsFSolve(k);      // forward physics: -> J
      design_solver.PhysicsASolve();                        // adjoint physics: -> dJ/drho_tilde
      design_solver.FilterASolve(dJ_drho_full);             // adjoint filter:  -> dJ/drho

      // Extract gradients for active DOFs only
      if (passive_region_coef)
      {
         MapFullToActive(dJ_drho_full, active_tdof_list, dJ_drho_active);
      }
      else
      {
         dJ_drho_active = dJ_drho_full;
      }

      // Volume constraint and current fraction (over active region only)
      const real_t cur_volume = InnerProduct(comm, *volume_weights, rho_tv_full);
      const real_t cur_vol_frac = cur_volume / domain_volume;
      fival(0) = cur_volume / target_volume - 1.0;

      // Box constraints with move limits (active DOFs only)
      rho_active_old = rho_active;
      for (int i = 0; i < n_active; i++)
      {
         rho_active_min[i] = max(real_t(0.0), rho_active[i] - problem.GetMoveLimit());
         rho_active_max[i] = min(real_t(1.0), rho_active[i] + problem.GetMoveLimit());
      }

      // MMA outer iteration (minimizes J subject to fival <= 0) - active DOFs only
      mma.Update(rho_active, dJ_drho_active, J, fival, dfidx,
                rho_active_min, rho_active_max);

      // Refresh the full design (visualization + checkpoint) from the update
      if (passive_region_coef)
      {
         MapActiveToFull(rho_active, active_tdof_list, passive_tdof_list,
                        passive_rho_value, rho_tv_full);
      }
      else
      {
         rho_tv_full = rho_active;
      }
      rho.SetFromTrueDofs(rho_tv_full);

      // Design change (L1 norm, matches ElastTopOpt_static) for the
      // early-stop test and progress monitoring: iterationError = int |dRho|.
      Vector rho_active_change(n_active);
      for (int i = 0; i < n_active; i++)
      {
         rho_active_change[i] = fabs(rho_active[i] - rho_active_old[i]);
      }
      iterationError = rho_active_change.Sum();
      MPI_Allreduce(MPI_IN_PLACE, &iterationError, 1,
                   MPITypeMap<real_t>::mpi_type, MPI_SUM, comm);

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
         history.flush();  // Ensure history is written to disk
      }

      // ======================================================================
      // SAVE CHECKPOINT (at end of each successful iteration)
      // ======================================================================
      // Only the post-update control density (rho_tv_full, already refreshed
      // above) plus small metadata. Atomic per file; safe against wall-clock
      // kills mid-save.
      if (auto_checkpoint)
      {
         ckpt_meta.iteration = k;
         ckpt_meta.objective = J;
         ckpt_meta.volume_fraction = cur_vol_frac;
         ckpt_meta.refinement_level = cfg.ref_levels;
         ckpt_meta.fe_order = cfg.order;

         if (!checkpoint.Save(ckpt_meta, rho_tv_full))
         {
            if (myid == 0)
            {
               cerr << "WARNING: Failed to save checkpoint at iteration " << k << "\n";
            }
         }
      }

      const bool is_last_iter =
         (k + 1 >= problem.GetMaxIterations()) ||
         (iterationError <= problem.GetChangeTolerance());
      if (paraview)
      {
         // Design snapshot: first, last, and every pv_save_interval-th iteration.
         if (k == 0 || is_last_iter || (k + 1) % pv_save_interval == 0)
         {
            paraview_dc.SetCycle(k + 1);
            paraview_dc.SetTime(k + 1);
            paraview_dc.Save();
         }

         // Forward wave visualization (first and last iteration only).
         // NOTE: This runs a separate forward-only sweep that stores ALL states.
         // Memory usage temporarily increases, but only for 2 iterations total.
         if (k == 0 || is_last_iter)
         {
            if (myid == 0)
            {
               cout << "    Generating forward wave visualization...\n";
            }

            // Run forward-only sweep for visualization
            std::vector<Vector> viz_states;
            std::vector<real_t> viz_times;
            design_solver.ForwardVisualizationSweep(rho_tv_full, viz_states, viz_times);

            // Save wave propagation to ParaView (the data collection creates
            // its own directories under the prefix path).
            string wave_collection_name = "wave_iter" + to_string(k);
            string wave_full_dir = output_parent_dir + "/ParaView/" + wave_collection_name;

            ParaViewDataCollection wave_dc(wave_collection_name.c_str(), &pmesh);
            wave_dc.SetLevelsOfDetail(problem.GetOrder());
            wave_dc.SetDataFormat(VTKFormat::BINARY);
            wave_dc.SetHighOrderOutput(true);
            wave_dc.SetPrefixPath((output_parent_dir + "/ParaView").c_str());

            // Create grid function for displacement only (not velocity to save space)
            ParGridFunction u_gf(&state_fes);
            wave_dc.RegisterField("displacement", &u_gf);

            // Save sampled timesteps (not all, to avoid millions of files)
            const int nsteps = design_solver.GetNumSteps();
            const int wave_viz_freq = max(1, nsteps / 20);  // Save ~20 frames total (5x reduction)
            int frames_saved = 0;

            for (int step = 0; step <= nsteps; step++)
            {
               // Save first, last, and every Nth step
               if (step == 0 || step == nsteps || step % wave_viz_freq == 0)
               {
                  const Vector &state = viz_states[step];
                  const int half_size = state.Size() / 2;

                  // Extract u (displacement) only
                  Vector u_vec(state.GetData(), half_size);

                  u_gf.SetFromTrueDofs(u_vec);

                  wave_dc.SetCycle(step);
                  wave_dc.SetTime(viz_times[step]);
                  wave_dc.Save();
                  frames_saved++;
               }
            }

            if (myid == 0)
            {
               cout << "    Saved " << frames_saved << " frames (every " << wave_viz_freq
                    << " steps from " << nsteps << " total)\n";
            }

            if (myid == 0)
            {
               cout << "    Wave visualization saved to: " << wave_full_dir << "/\n";
            }
         }
      }
   }

   if (myid == 0)
   {
      history.close();
      cout << "\n=== Optimization Complete ===\n";
      cout << "Output directory: " << output_parent_dir << "\n";
      cout << "Total iterations: " << k << "\n";
      cout << "Final convergence error: " << scientific << setprecision(3)
           << iterationError << " (tol = " << problem.GetChangeTolerance() << ")\n";
      if (paraview)
      {
         cout << "ParaView output: " << paraview_dir << "/TopOptTransient.pvd\n";
      }
      cout << "History file: " << history_file << "\n";
      if (auto_checkpoint)
      {
         cout << "Checkpoint: " << checkpoint_dir << "\n";
         cout << "  (Use '-out " << output_parent_dir << " -restart' to continue)\n";
      }
      cout << "=============================\n";
   }

   return 0;
}
