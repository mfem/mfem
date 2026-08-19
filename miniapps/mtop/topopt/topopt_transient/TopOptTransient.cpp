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
//   3. TransientDesignSolver runs the RK4 forward sweep and either the default
//      exact discrete adjoint or the opt-in continuous RK4/Hermite adjoint,
//      returning dJ/drho after the filter transpose,
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
#include "ReferenceBoundaryData.hpp"
#include "ProblemSpecification.hpp"
#include "OptimizationCheckpoint.hpp"
#include "../../pde_filter.hpp"
#include "../../mma/MMA_MFEM.hpp"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
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

   // A passive element on one rank must also mark every shared copy of its
   // interface DOFs before restriction to the owned true-DOF vector.
   fes.Synchronize(local_is_passive);

   // Convert the synchronized local marker to a true-DOF marker.  MFEM's
   // restriction matrix has shape (true dofs) x (local dofs).
   const int n_true = fes.GetTrueVSize();
   Array<int> true_is_passive(n_true);
   true_is_passive = 0;

   const SparseMatrix *R = fes.GetRestrictionMatrix();
   if (R)
   {
      R->BooleanMult(local_is_passive, true_is_passive);
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

// A deterministic, volume-neutral transverse density perturbation for the 2D
// periodic band converter. The sin^2 axial envelope is zero at the active-strip
// interfaces; cos(m*pi*y/H) has zero cross-sectional mean for m>0.
class ModalSeedDesignCoefficient : public Coefficient
{
private:
   real_t rho_mean_, amplitude_, x_min_, x_max_, y_max_;
   int transverse_mode_;

public:
   ModalSeedDesignCoefficient(real_t rho_mean, real_t amplitude,
                              real_t x_min, real_t x_max, real_t y_max,
                              int transverse_mode)
      : rho_mean_(rho_mean), amplitude_(amplitude), x_min_(x_min),
        x_max_(x_max), y_max_(y_max), transverse_mode_(transverse_mode)
   {
      MFEM_VERIFY(x_max_ > x_min_ && y_max_ > 0.0 && transverse_mode_ > 0,
                  "Modal seed requires a positive 2D active region and mode.");
   }

   real_t Eval(ElementTransformation &T, const IntegrationPoint &ip) override
   {
      Vector x(2);
      T.Transform(ip, x);
      if (x(0) < x_min_ || x(0) > x_max_)
      {
         return rho_mean_;
      }
      constexpr real_t pi = 3.1415926535897932384626433832795;
      const real_t xi = (x(0) - x_min_) / (x_max_ - x_min_);
      const real_t axial_envelope = std::pow(std::sin(pi * xi), 2);
      const real_t transverse_mode =
         std::cos(transverse_mode_ * pi * x(1) / y_max_);
      return rho_mean_ + amplitude_ * axial_envelope * transverse_mode;
   }
};

// Deliberately high-contrast material seed for fixed-design RK4 stress tests.
// Its scale is specified in physical coordinates, rather than by element index,
// so it remains an unambiguous input when the mesh is changed.  Passive cells
// are imposed after initialization by the normal problem mask.
class CheckerboardDesignCoefficient : public Coefficient
{
private:
   real_t rho_mean_, amplitude_, x_period_, y_period_;

public:
   CheckerboardDesignCoefficient(real_t rho_mean, real_t amplitude,
                                 real_t x_period, real_t y_period)
      : rho_mean_(rho_mean), amplitude_(amplitude),
        x_period_(x_period), y_period_(y_period)
   {
      MFEM_VERIFY(x_period_ > 0.0 && y_period_ > 0.0,
                  "Checkerboard periods must be positive.");
   }

   real_t Eval(ElementTransformation &T, const IntegrationPoint &ip) override
   {
      Vector x(2);
      T.Transform(ip, x);
      const long long ix = static_cast<long long>(std::floor(x(0) / x_period_));
      const long long iy = static_cast<long long>(std::floor(x(1) / y_period_));
      return rho_mean_ + (((ix + iy) & 1LL) ? amplitude_ : -amplitude_);
   }
};

bool InitializeDesign(ParGridFunction &rho, const char *design_init,
                      real_t vol_frac, real_t x_max, real_t y_max,
                      const TransientTopOptProblem &problem)
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

   if (mode == "checkerboard")
   {
      const real_t amplitude = std::min(
         real_t(0.35), real_t(0.70) * std::min(vol_frac, 1.0 - vol_frac));
      CheckerboardDesignCoefficient checkerboard(
         vol_frac, amplitude, 0.10 * x_max, 0.10 * y_max);
      rho.ProjectCoefficient(checkerboard);
      return true;
   }

   if (mode == "modal-seed")
   {
      real_t active_x_min = 0.0, active_x_max = 0.0;
      int transverse_mode = 0;
      if (!problem.GetModalSeedRegion(active_x_min, active_x_max,
                                      transverse_mode))
      {
         return false;
      }
      // At vf=0.5 this gives rho in [0.3,0.7]. Reduce the perturbation for
      // volume fractions near a box bound, leaving room for a roundoff-level
      // volume correction after Q0 projection.
      const real_t amplitude = std::min(
         real_t(0.20), real_t(0.40) * std::min(vol_frac, 1.0 - vol_frac));
      ModalSeedDesignCoefficient modal_seed(
         vol_frac, amplitude, active_x_min, active_x_max, y_max,
         transverse_mode);
      rho.ProjectCoefficient(modal_seed);
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
   int mpi_ranks = 1;
   MPI_Comm_size(comm, &mpi_ranks);

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
   // -1 preserves the historical coupling: design/filter order follows -o.
   int design_order = -1;
   bool damping = true;   // -damp / -no-damp: apply the problem's damping
   bool forward_only = false;  // single forward objective sweep; no MMA/adjoint
   bool reference_only = false; // generate inclusion data, then exit
   int reference_order = -1;    // -1 = reconstruction order + 1
   real_t reference_time_step = -1.0; // -1 = effective coarse dt / 4
   bool forward_modal_probe = false;  // terminal projection for temporal studies
   bool forward_modal_history = false; // launched/converted output time series
   int modal_sample_every = 1;
   bool rhs_spectrum = false;  // compare modal content of F and dJ/du
   bool continuous_adjoint_check = false;  // full-storage state-adjoint study
   bool continuous_adjoint_refinement_check = false; // Hermite fine-adjoint study
   bool continuous_storage_check = false; // production FULL/REVOLVE equivalence
   bool rk4_adjoint_comparison = false; // DO/transformed/naive same-grid study
   int rk4_taylor_levels = 0;
   real_t rk4_taylor_epsilon = 1e-2;
   bool rk4_taylor_linf_normalized = false;
   int maximum_adjoint_coarsening = 16;
   int minimum_adjoint_refinement = 1;
   int maximum_adjoint_refinement = 16;
   const char *adjoint_mode_name = "discrete";
   const char *objective_quadrature_name = "legacy";
   const char *trajectory_storage_name = "revolve";
   int adjoint_refinement = 1;
   int adjoint_coarsening = 1;
   // Positive rescaling used only by the MMA subproblem.  It leaves the
   // reported physical objective and all forward/adjoint calculations intact.
   real_t mma_objective_scale = 1.0;
   // Carrier/pulse overrides (0 = keep the problem's default). Same problem can
   // then run cheap (low f, coarse mesh) locally and rich (high f) on HPC.
   real_t load_frequency = 0.0;
   real_t load_duration = 0.0;
   // Negative sentinels retain each problem's material-law default.
   real_t simp_r_min = -1.0;
   real_t simp_r_max = -1.0;
   real_t simp_p = -1.0;
   int num_checkpoints = -1;   // REVOLVE snapshot count; -1 = auto-size

   // === Checkpoint and output directory options ===
   // Optional C-string options must stay non-null because OptionsParser prints
   // them even when the user leaves them unset.
   const char *output_parent = "";       // Parent directory for all outputs
   bool restart = false;                 // Restart from checkpoint?
   // OptionsParser prints every C-string option.  Keep the optional path
   // non-null so PrintOptions() cannot set cout's failbit on a null char*.
   const char *restart_from = "";        // Optional checkpoint source output
   bool auto_checkpoint = true;          // Auto-save checkpoint each iteration
   bool checkpoint_history = false;      // Preserve one raw design checkpoint per update

   OptionsParser args(argc, argv);
   args.AddOption(&problem_name, "-problem", "--problem",
                  "Forward problem: wave, cantilever-compliance, "
                  "cantilever-harmonic-l2, cantilever-harmonic-tracking, "
                  "elastic-inclusion-identification, "
                  "band-waveguide, band-waveguide-legacy, band-mode-converter, "
                  "band-mode-converter-correlation, "
                  "band-mode-converter-energy, "
                  "band-mode-converter-reverse, spherical-bandgap, "
                  "mode-converter-3d, or mode-converter-reverse-3d");
   args.AddOption(&damping, "-damp", "--damp", "-no-damp", "--no-damp",
                  "Apply the problem's damping (bulk + absorbing). -no-damp zeroes "
                  "all dissipation: free (Neumann) boundaries, conservative system.");
   args.AddOption(&cfg.damping_reflection,
                  "-damp-reflection", "--damp-reflection",
                  "Target amplitude reflection for the exterior damping sponge "
                  "(must lie in (0,1); default is problem-specific). ");
   args.AddOption(&forward_only, "-forward-only", "--forward-only",
                  "-no-forward-only", "--no-forward-only",
                  "Run one forward objective sweep at the initial/restarted design; "
                  "skip MMA, adjoint, and optimization checkpointing. With -pv, "
                  "write sampled wave fields without storing the full trajectory.");
   args.AddOption(&reference_only,
                  "-reference-only", "--reference-only",
                  "-no-reference-only", "--no-reference-only",
                  "Generate the elastic-inclusion reference boundary data and "
                  "exit before constructing the reconstruction solver or MMA. "
                  "Truth fields and metadata are written; trace samples remain "
                  "in memory only.");
   args.AddOption(&reference_order,
                  "-reference-order", "--reference-order",
                  "Reference state H1 order for elastic inclusion data "
                  "(-1 selects reconstruction order + 1).");
   args.AddOption(&reference_time_step,
                  "-reference-dt", "--reference-time-step",
                  "Reference RK4 timestep for elastic inclusion data "
                  "(-1 selects effective reconstruction dt / 4).");
   args.AddOption(&forward_modal_probe,
                  "-forward-modal-probe", "--forward-modal-probe",
                  "-no-forward-modal-probe", "--no-forward-modal-probe",
                  "In -forward-only mode, report the terminal projection onto the "
                  "problem-provided forward temporal-convergence mode.");
   args.AddOption(&forward_modal_history,
                  "-forward-modal-history", "--forward-modal-history",
                  "-no-forward-modal-history", "--no-forward-modal-history",
                  "In -forward-only mode, write launched- and converted-mode "
                  "receiver projections versus time and report carrier-demodulated "
                  "amplitudes. Requires both problem-provided modal probes.");
   args.AddOption(&modal_sample_every,
                  "-modal-sample-every", "--modal-sample-every",
                  "Accepted forward steps between modal-history samples "
                  "(default 1; the final endpoint is always included).");
   args.AddOption(&rhs_spectrum, "-rhs-spectrum", "--rhs-spectrum",
                  "-no-rhs-spectrum", "--no-rhs-spectrum",
                  "Analyze the generalized K phi = lambda M phi spectral measures "
                  "seen by the spatial forward RHS F and adjoint RHS dJ/du, then exit.");
   args.AddOption(&continuous_adjoint_check,
                  "-continuous-adjoint-check", "--continuous-adjoint-check",
                  "-no-continuous-adjoint-check",
                  "--no-continuous-adjoint-check",
                  "Run a full-storage continuous state-adjoint coarsening study "
                  "on the selected forward trajectory, then exit.");
   args.AddOption(&maximum_adjoint_coarsening,
                  "-adjoint-coarsening-max", "--adjoint-coarsening-max",
                  "Largest power-of-two dt_a/dt_f ratio tested by "
                  "-continuous-adjoint-check (minimum 2; q=2 is the "
                  "interpolation-free reference).");
   args.AddOption(
      &continuous_adjoint_refinement_check,
      "-continuous-adjoint-refinement-check",
      "--continuous-adjoint-refinement-check",
      "-no-continuous-adjoint-refinement-check",
      "--no-continuous-adjoint-refinement-check",
      "Run a fixed-design full-storage coarse-forward/fine-continuous-adjoint "
      "study of J, p(0), and filtered/active-raw design gradients using cubic "
      "Hermite forward-state reconstruction, then exit.");
   args.AddOption(
      &continuous_storage_check,
      "-continuous-storage-check", "--continuous-storage-check",
      "-no-continuous-storage-check", "--no-continuous-storage-check",
      "At the initial design, run the exact production continuous objective/"
      "gradient path once with full trajectory storage and once with REVOLVE, "
      "compare J, p(0), and filtered/active-raw gradients, write timing/storage/"
      "replay telemetry, then exit. Uses -nchk for the REVOLVE case.");
   args.AddOption(
      &rk4_adjoint_comparison,
      "-rk4-adjoint-comparison", "--rk4-adjoint-comparison",
      "-no-rk4-adjoint-comparison", "--no-rk4-adjoint-comparison",
      "At the initial design, evaluate one common four-stage RK4 objective "
      "and compare its exact DO gradient, an independently implemented "
      "transformed adjoint, and naive continuous RK4 over the accepted/Hermite "
      "trajectory; write design-metric errors and then exit.");
   args.AddOption(
      &rk4_taylor_levels,
      "-rk4-taylor-levels", "--rk4-taylor-levels",
      "Number of halved, symmetric raw-design perturbations for the optional "
      "volume-neutral Taylor test in -rk4-adjoint-comparison (0 disables).");
   args.AddOption(
      &rk4_taylor_epsilon,
      "-rk4-taylor-epsilon", "--rk4-taylor-epsilon",
      "Initial symmetric perturbation for -rk4-adjoint-comparison Taylor test.");
   args.AddOption(
      &rk4_taylor_linf_normalized,
      "-rk4-taylor-linf-normalized", "--rk4-taylor-linf-normalized",
      "-no-rk4-taylor-linf-normalized", "--no-rk4-taylor-linf-normalized",
      "Rescale the volume-neutral L2-Riesz Taylor direction to unit active "
      "L-infinity norm before applying -rk4-taylor-epsilon. This makes the "
      "Taylor parameter a maximum raw-density perturbation.");
   args.AddOption(
      &minimum_adjoint_refinement,
      "-adjoint-refinement-min", "--adjoint-refinement-min",
      "Smallest power-of-two dt_f/dt_a ratio tested by "
      "-continuous-adjoint-refinement-check (default 1). Set this above 1 "
      "to continue an expensive convergence study without rerunning coarser "
      "candidates; at least two consecutive ratios are still required.");
   args.AddOption(
      &maximum_adjoint_refinement,
      "-adjoint-refinement-max", "--adjoint-refinement-max",
      "Largest power-of-two dt_f/dt_a ratio tested by "
      "-continuous-adjoint-refinement-check (minimum 2; the largest "
      "ratio is the adjoint/gradient reference; a 2x-finer forward audit samples "
      "all of its RK4 stage times).");
   args.AddOption(
      &adjoint_mode_name, "-adjoint-mode", "--adjoint-mode",
      "Optimization gradient: discrete (exact derivative of the selected "
      "RK4 objective) or continuous (RK4/Hermite adjoint integral on a "
      "nested time grid).");
   args.AddOption(
      &objective_quadrature_name,
      "-objective-quadrature", "--objective-quadrature",
      "Running-objective time quadrature: legacy (trapezoid for discrete, "
      "Simpson/Hermite for continuous) or rk4-stage (one common four-stage "
      "functional; same-grid experiment only).");
   args.AddOption(
      &trajectory_storage_name,
      "-trajectory-storage", "--trajectory-storage",
      "Forward trajectory storage: revolve (default checkpoint/replay path) "
      "or full (retain all forward endpoint states; continuous mode only).");
   args.AddOption(
      &adjoint_refinement, "-adjoint-refinement", "--adjoint-refinement",
      "Fine continuous-adjoint steps per coarse forward interval. "
      "Must be 1 in discrete mode.");
   args.AddOption(
      &adjoint_coarsening, "-adjoint-coarsening", "--adjoint-coarsening",
      "Fine forward steps per coarse continuous-adjoint step. Must be 1 "
      "in discrete mode and is mutually exclusive with refinement > 1.");
   args.AddOption(&cfg.ref_levels, "-r", "--refine", "Refinement level");
   args.AddOption(&cfg.order, "-o", "--order",
                  "Forward/adjoint state H1 finite element order");
   args.AddOption(&design_order, "-do", "--design-order",
                  "H1 order of rho_tilde; rho uses paired L2 order "
                  "max(0, design-order-1). Default: --order (legacy). ");
   args.AddOption(&cfg.t_final, "-tf", "--t-final", "Final time");
   args.AddOption(&cfg.dt, "-dt", "--time-step", "Time step");
   args.AddOption(&cfg.vol_frac, "-vf", "--vol-frac", "Target volume fraction");
   args.AddOption(&cfg.filter_radius, "-fr", "--filter-radius",
                  "Helmholtz filter radius");
   args.AddOption(&simp_r_min, "-simp-rmin", "--simp-rmin",
                  "SIMP material scale at rho_tilde=0 (positive = override "
                  "the band-waveguide default)");
   args.AddOption(&simp_r_max, "-simp-rmax", "--simp-rmax",
                  "SIMP material scale at rho_tilde=1 (positive = override "
                  "the band-waveguide default)");
   args.AddOption(&simp_p, "-simp-p", "--simp-p",
                  "SIMP exponent (positive = override the band-waveguide default)");
   args.AddOption(&cfg.max_it, "-mi", "--max-it", "Max MMA iterations");
   args.AddOption(&cfg.move, "-mv", "--move", "MMA move limit");
   args.AddOption(&mma_objective_scale,
                  "-mma-objective-scale", "--mma-objective-scale",
                  "Positive objective/gradient scale used only inside the "
                  "MMA subproblem (default 1). ");
   args.AddOption(&cfg.change_tol, "-tol", "--tol",
                  "Stop early when the L1 design change drops below this");
   args.AddOption(&design_init, "-init", "--design-init",
                  "Initial design: uniform, solid, void, gaussian, checkerboard, or "
                  "modal-seed (2D band converter only).");
   args.AddOption(&mesh_file, "-mesh", "--mesh-file", "Mesh file");
   args.AddOption(&paraview, "-pv", "--paraview", "-no-pv",
                  "--no-paraview", "Write ParaView output");
   args.AddOption(&output_parent, "-out", "--output-parent",
                  "Parent directory for all outputs (ParaView, checkpoint, history). "
                  "If not specified, auto-generates: YYYYMMDD_HHMMSS_jobSLURM_ID");
   args.AddOption(&restart, "-restart", "--restart", "-no-restart", "--no-restart",
                  "Restart optimization from checkpoint in output-parent directory");
   args.AddOption(&restart_from, "-restart-from", "--restart-from",
                  "Source output directory for -restart (default: --output-parent). "
                  "Use this to seed a separate output branch.");
   args.AddOption(&auto_checkpoint, "-ckpt", "--checkpoint", "-no-ckpt", "--no-checkpoint",
                  "Enable automatic checkpointing at each iteration");
   args.AddOption(&checkpoint_history, "-ckpt-history", "--checkpoint-history",
                  "-no-ckpt-history", "--no-checkpoint-history",
                  "Preserve every post-update raw-control checkpoint under "
                  "optimization_checkpoint_history (for paired-run diagnostics). ");
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
                  "Pulse duration override (0 = problem default; the legacy "
                  "band and spherical loads span t_final, while the 2D mode "
                  "converter uses a one-unit pulse).");
   args.AddOption(&cfg.mode_converter_target_mode,
                  "-tm", "--target-mode",
                  "Transverse cos(n*pi*y/H) target mode for the 2D band "
                  "mode-converter (positive and even because y is periodic).");
   args.AddOption(&cfg.mode_converter_target_amplitude,
                  "-ta", "--target-amplitude",
                  "Harmonic target/correlation amplitude for the 2D band "
                  "mode-converter.");
   args.AddOption(&cfg.mode_converter_energy_residual_weight,
                  "-energy-low-penalty", "--energy-low-penalty",
                  "Relative penalty on residual mode-0 energy in the "
                  "windowed modal-energy mode-converter objective.");
   args.AddOption(&cfg.mode_converter_energy_window_start,
                  "-energy-window-start", "--energy-window-start",
                  "Start time of the output window in the modal-energy "
                  "mode-converter objective.");
   args.AddOption(&cfg.mode_converter_energy_window_ramp,
                  "-energy-window-ramp", "--energy-window-ramp",
                  "Sin-squared ramp duration of the output window in the "
                  "modal-energy mode-converter objective (0 = step). ");
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
   const auto command_line_has_option = [&](const char *short_name,
                                             const char *long_name)
   {
      for (int i = 1; i < argc; i++)
      {
         if (std::strcmp(argv[i], short_name) == 0 ||
             std::strcmp(argv[i], long_name) == 0)
         {
            return true;
         }
      }
      return false;
   };
   cfg.order_is_user = command_line_has_option("-o", "--order");
   cfg.t_final_is_user = command_line_has_option("-tf", "--t-final");
   cfg.time_step_is_user = command_line_has_option("-dt", "--time-step");
   cfg.volume_fraction_is_user =
      command_line_has_option("-vf", "--vol-frac");
   cfg.filter_radius_is_user =
      command_line_has_option("-fr", "--filter-radius");
   cfg.mesh_file = mesh_file;
   cfg.mesh_file_is_user =
      command_line_has_option("-mesh", "--mesh-file");
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
   if (simp_r_min > 0.0)
   {
      cfg.material.r_min = simp_r_min;
      cfg.simp_r_min_is_user = true;
   }
   if (simp_r_max > 0.0)
   {
      cfg.material.r_max = simp_r_max;
      cfg.simp_r_max_is_user = true;
   }
   if (simp_p > 0.0)
   {
      cfg.material.simp_p = simp_p;
      cfg.simp_p_is_user = true;
   }
   if ((cfg.simp_r_min_is_user || cfg.simp_r_max_is_user) &&
       cfg.material.r_max <= cfg.material.r_min)
   {
      if (myid == 0)
      {
         cerr << "Error: --simp-rmax must exceed --simp-rmin.\n";
      }
      return 1;
   }
   if (reference_order != -1 && reference_order < 1)
   {
      if (myid == 0)
      {
         cerr << "Error: --reference-order must be -1 (automatic) or a "
                 "positive integer.\n";
      }
      return 1;
   }
   if (reference_time_step != -1.0 &&
       (!std::isfinite(reference_time_step) || reference_time_step <= 0.0))
   {
      if (myid == 0)
      {
         cerr << "Error: --reference-time-step must be -1 (automatic) or "
                 "finite and positive.\n";
      }
      return 1;
   }

   if (cfg.order < 1)
   {
      if (myid == 0) { cerr << "Error: -o/--order must be at least 1.\n"; }
      return 1;
   }
   if (design_order < 0) { design_order = cfg.order; }
   if (design_order < 1)
   {
      if (myid == 0)
      {
         cerr << "Error: -do/--design-order must be at least 1.\n";
      }
      return 1;
   }

   const string adjoint_mode_sel = ToLower(adjoint_mode_name);
   TransientAdjointMode adjoint_mode = TransientAdjointMode::DISCRETE;
   if (adjoint_mode_sel == "discrete")
   {
      adjoint_mode = TransientAdjointMode::DISCRETE;
   }
   else if (adjoint_mode_sel == "continuous")
   {
      adjoint_mode = TransientAdjointMode::CONTINUOUS;
   }
   else
   {
      if (myid == 0)
      {
         cerr << "Error: -adjoint-mode must be discrete or continuous.\n";
      }
      return 1;
   }

   const string trajectory_storage_sel = ToLower(trajectory_storage_name);
   TrajectoryStorageMode trajectory_storage = TrajectoryStorageMode::REVOLVE;
   if (trajectory_storage_sel == "revolve")
   {
      trajectory_storage = TrajectoryStorageMode::REVOLVE;
   }
   else if (trajectory_storage_sel == "full")
   {
      trajectory_storage = TrajectoryStorageMode::FULL;
   }
   else
   {
      if (myid == 0)
      {
         cerr << "Error: -trajectory-storage must be revolve or full.\n";
      }
      return 1;
   }
   const string objective_quadrature_sel =
      ToLower(objective_quadrature_name);
   bool rk4_stage_objective = false;
   if (objective_quadrature_sel == "legacy")
   {
      rk4_stage_objective = false;
   }
   else if (objective_quadrature_sel == "rk4-stage")
   {
      rk4_stage_objective = true;
   }
   else
   {
      if (myid == 0)
      {
         cerr << "Error: -objective-quadrature must be legacy or "
                 "rk4-stage.\n";
      }
      return 1;
   }
   // Resolve the inclusion problem's mandatory common RK4-stage functional
   // before the generic grid/diagnostic compatibility checks below.  Doing
   // this only after constructing the concrete problem would let a legacy
   // continuous diagnostic slip through with unsupported observation times,
   // and would unnecessarily require an explicit quadrature option for the
   // fixed-design DO/modified/naive comparison.
   const bool inclusion_problem_requested =
      ToLower(problem_name) == "elastic-inclusion-identification";
   if (inclusion_problem_requested && !reference_only &&
       !rk4_stage_objective)
   {
      rk4_stage_objective = true;
      objective_quadrature_name = "rk4-stage";
      if (myid == 0)
      {
         cout << "Elastic inclusion identification: selecting the required "
                 "common RK4-stage objective quadrature.\n";
      }
   }
   if (adjoint_refinement < 1)
   {
      if (myid == 0)
      {
         cerr << "Error: -adjoint-refinement must be >= 1.\n";
      }
      return 1;
   }
   if (adjoint_coarsening < 1)
   {
      if (myid == 0)
      {
         cerr << "Error: -adjoint-coarsening must be >= 1.\n";
      }
      return 1;
   }
   if (adjoint_refinement > 1 && adjoint_coarsening > 1)
   {
      if (myid == 0)
      {
         cerr << "Error: -adjoint-refinement and -adjoint-coarsening are "
                 "mutually exclusive.\n";
      }
      return 1;
   }
   if (rk4_stage_objective &&
       (adjoint_refinement != 1 || adjoint_coarsening != 1))
   {
      if (myid == 0)
      {
         cerr << "Error: -objective-quadrature rk4-stage requires the "
                 "same forward/adjoint grid (refinement=coarsening=1).\n";
      }
      return 1;
   }
   if (adjoint_mode == TransientAdjointMode::DISCRETE &&
       (adjoint_refinement != 1 || adjoint_coarsening != 1))
   {
      if (myid == 0)
      {
         cerr << "Error: adjoint refinement/coarsening is only available "
                 "with -adjoint-mode continuous.\n";
      }
      return 1;
   }
   if (adjoint_mode == TransientAdjointMode::DISCRETE &&
       trajectory_storage == TrajectoryStorageMode::FULL)
   {
      if (myid == 0)
      {
         cerr << "Error: -trajectory-storage full currently requires "
                 "-adjoint-mode continuous.\n";
      }
      return 1;
   }
   if (num_checkpoints == 0 || num_checkpoints < -1)
   {
      if (myid == 0)
      {
         cerr << "Error: -nchk/--num-checkpoints must be -1 (automatic) "
                 "or a positive integer.\n";
      }
      return 1;
   }
   if (trajectory_storage == TrajectoryStorageMode::FULL &&
       num_checkpoints != -1)
   {
      if (myid == 0)
      {
         cerr << "Error: -nchk/--num-checkpoints is only meaningful with "
                 "-trajectory-storage revolve.\n";
      }
      return 1;
   }
   if (restart_from[0] != '\0' && !restart)
   {
      if (myid == 0) { cerr << "Error: --restart-from requires -restart.\n"; }
      return 1;
   }
   if (modal_sample_every < 1)
   {
      if (myid == 0)
      {
         cerr << "Error: -modal-sample-every must be >= 1.\n";
      }
      return 1;
   }
   if ((forward_modal_probe || forward_modal_history) && !forward_only)
   {
      if (myid == 0)
      {
         cerr << "Error: forward modal diagnostics require -forward-only.\n";
      }
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
   if (!(mma_objective_scale > 0.0) || !std::isfinite(mma_objective_scale))
   {
      if (myid == 0)
      {
         cerr << "Error: -mma-objective-scale must be finite and positive.\n";
      }
      return 1;
   }
   if (maximum_adjoint_coarsening < 2)
   {
      if (myid == 0)
      {
         cerr << "Error: -adjoint-coarsening-max must be >= 2.\n";
      }
      return 1;
   }
   if (maximum_adjoint_refinement < 2)
   {
      if (myid == 0)
      {
         cerr << "Error: -adjoint-refinement-max must be >= 2.\n";
      }
      return 1;
   }
   if (rk4_taylor_levels < 0 || !std::isfinite(rk4_taylor_epsilon) ||
       rk4_taylor_epsilon <= 0.0)
   {
      if (myid == 0)
      {
         cerr << "Error: -rk4-taylor-levels must be nonnegative and "
                 "-rk4-taylor-epsilon must be positive.\n";
      }
      return 1;
   }
   if (rk4_taylor_levels > 0 && !rk4_adjoint_comparison)
   {
      if (myid == 0)
      {
         cerr << "Error: -rk4-taylor-levels requires "
                 "-rk4-adjoint-comparison.\n";
      }
      return 1;
   }
   if (minimum_adjoint_refinement < 1 ||
       (minimum_adjoint_refinement & (minimum_adjoint_refinement - 1)) != 0 ||
       minimum_adjoint_refinement > maximum_adjoint_refinement / 2)
   {
      if (myid == 0)
      {
         cerr << "Error: -adjoint-refinement-min must be a positive power "
                 "of two with at least one consecutive doubled ratio not "
                 "exceeding -adjoint-refinement-max.\n";
      }
      return 1;
   }
   const int continuous_diagnostic_count =
      (continuous_adjoint_check ? 1 : 0) +
      (continuous_adjoint_refinement_check ? 1 : 0) +
      (continuous_storage_check ? 1 : 0);
   if (continuous_diagnostic_count > 1)
   {
      if (myid == 0)
      {
         cerr << "Error: select only one continuous-adjoint diagnostic "
                 "at a time.\n";
      }
      return 1;
   }
   if (rk4_adjoint_comparison &&
       (continuous_diagnostic_count > 0 || forward_only || rhs_spectrum))
   {
      if (myid == 0)
      {
         cerr << "Error: -rk4-adjoint-comparison is mutually exclusive with "
                 "the continuous diagnostics, -forward-only, and "
                 "-rhs-spectrum.\n";
      }
      return 1;
   }
   if (rk4_adjoint_comparison && !rk4_stage_objective)
   {
      if (myid == 0)
      {
         cerr << "Error: -rk4-adjoint-comparison requires "
                 "-objective-quadrature rk4-stage.\n";
      }
      return 1;
   }
   if (rk4_stage_objective && continuous_diagnostic_count > 0)
   {
      if (myid == 0)
      {
         cerr << "Error: the legacy continuous refinement/storage diagnostics "
                 "cannot be combined with -objective-quadrature rk4-stage.\n";
      }
      return 1;
   }
   if (continuous_storage_check &&
       adjoint_mode != TransientAdjointMode::CONTINUOUS)
   {
      if (myid == 0)
      {
         cerr << "Error: -continuous-storage-check requires "
                 "-adjoint-mode continuous.\n";
      }
      return 1;
   }
   if (continuous_storage_check &&
       trajectory_storage != TrajectoryStorageMode::REVOLVE)
   {
      if (myid == 0)
      {
         cerr << "Error: -continuous-storage-check requires the default "
                 "-trajectory-storage revolve configuration so -nchk can "
                 "define its REVOLVE leg; the diagnostic runs the FULL leg "
                 "internally.\n";
      }
      return 1;
   }
   if (continuous_storage_check && (forward_only || rhs_spectrum))
   {
      if (myid == 0)
      {
         cerr << "Error: -continuous-storage-check is mutually exclusive "
                 "with -forward-only and -rhs-spectrum.\n";
      }
      return 1;
   }

   unique_ptr<TransientTopOptProblem> problem_owner;
   const string problem_sel = ToLower(problem_name);
   if (problem_sel == "cantilever-compliance")
   {
      problem_owner = make_unique<CantileverComplianceProblem>(cfg);
   }
   else if (problem_sel == "elastic-inclusion-identification")
   {
      problem_owner =
         make_unique<ElasticInclusionIdentificationProblem>(cfg);
   }
   else if (problem_sel == "cantilever-harmonic-l2")
   {
      problem_owner = make_unique<CantileverComplianceProblem>(
         cfg, /*harmonic_l2=*/true);
   }
   else if (problem_sel == "cantilever-harmonic-tracking")
   {
      problem_owner = make_unique<CantileverComplianceProblem>(
         cfg, /*harmonic_l2=*/false, /*harmonic_tracking=*/true);
   }
   else if (problem_sel == "band-waveguide")
   {
      problem_owner = make_unique<BandWaveguideProblem>(cfg);
   }
   else if (problem_sel == "band-waveguide-legacy")
   {
      problem_owner = make_unique<BandWaveguideProblem>(cfg, true);
   }
   else if (problem_sel == "band-mode-converter")
   {
      problem_owner = make_unique<BandModeConverterProblem>(cfg);
   }
   else if (problem_sel == "band-mode-converter-correlation")
   {
      problem_owner = make_unique<BandModeConverterProblem>(
         cfg, /*reverse_spectral_roles=*/false,
         /*modal_correlation_objective=*/true);
   }
   else if (problem_sel == "band-mode-converter-energy")
   {
      problem_owner = make_unique<BandModeConverterProblem>(
         cfg, /*reverse_spectral_roles=*/false,
         /*modal_correlation_objective=*/false,
         /*modal_energy_objective=*/true);
   }
   else if (problem_sel == "band-mode-converter-reverse")
   {
      problem_owner = make_unique<BandModeConverterProblem>(
         cfg, /*reverse_spectral_roles=*/true);
   }
   else if (problem_sel == "wave")
   {
      problem_owner = make_unique<WaveShieldingProblem>(cfg);
   }
   else if (problem_sel == "spherical-bandgap")
   {
      problem_owner = make_unique<SphericalBandGapProblem>(cfg);
   }
   else if (problem_sel == "mode-converter-3d")
   {
      problem_owner = make_unique<ModeConverterWaveguideProblem>(cfg);
   }
   else if (problem_sel == "mode-converter-reverse-3d")
   {
      problem_owner = make_unique<ModeConverterWaveguideProblem>(
         cfg, /*reverse_spectral_roles=*/true);
   }
   else
   {
      if (myid == 0)
      {
        cerr << "Error: unknown -problem '" << problem_name
              << "'. Use wave, cantilever-compliance, cantilever-harmonic-l2, "
                 "cantilever-harmonic-tracking, "
                 "elastic-inclusion-identification, "
                 "band-waveguide, band-waveguide-legacy, "
                 "band-mode-converter, band-mode-converter-correlation, "
                 "band-mode-converter-energy, "
                 "band-mode-converter-reverse, "
                 "spherical-bandgap, mode-converter-3d, or "
                 "mode-converter-reverse-3d.\n";
      }
      return 1;
   }
   TransientTopOptProblem &problem = *problem_owner;

   // A problem-owned state-order default is resolved only after the concrete
   // problem exists.  Preserve an explicit design-order override; otherwise
   // keep the documented design/filter-order-follows-state behavior.
   if (!command_line_has_option("-do", "--design-order"))
   {
      design_order = problem.GetOrder();
   }

   const bool requires_reference_data =
      problem.RequiresReferenceBoundaryData();
   if (!requires_reference_data &&
       (reference_only || reference_order != -1 ||
        reference_time_step != -1.0))
   {
      if (myid == 0)
      {
         cerr << "Error: reference-data options require "
                 "-problem elastic-inclusion-identification.\n";
      }
      return 1;
   }
   if (reference_only &&
       (forward_only || restart || rhs_spectrum || rk4_adjoint_comparison ||
        continuous_diagnostic_count > 0))
   {
      if (myid == 0)
      {
         cerr << "Error: -reference-only is mutually exclusive with restart, "
                 "forward-only, adjoint diagnostics/comparisons, and RHS "
                 "spectrum analysis.\n";
      }
      return 1;
   }
   if (requires_reference_data && rk4_stage_objective &&
       (adjoint_refinement != 1 || adjoint_coarsening != 1))
   {
      if (myid == 0)
      {
         cerr << "Error: elastic inclusion identification uses the common "
                 "same-grid RK4-stage objective; adjoint refinement and "
                 "coarsening must both equal 1.\n";
      }
      return 1;
   }

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

   // OptionsParser retains pointers to the driver-owned cfg fields.  Refresh
   // them from the concrete problem so its problem-specific defaults (rather
   // than the generic startup defaults) are what PrintOptions reports.
   cfg = problem.GetConfig();
   mesh_file = problem.GetMeshFile().c_str();
   if (myid == 0)
   {
      args.PrintOptions(cout);
      problem.PrintSummary(cout);
   }

   // =========================================================================
   // SETUP OUTPUT DIRECTORY STRUCTURE (timestamp_jobID)
   // =========================================================================
   string output_parent_dir;
   if (output_parent[0] != '\0')
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
   string checkpoint_history_root =
      output_parent_dir + "/optimization_checkpoint_history";
   const string restart_checkpoint_dir = restart_from[0] != '\0' ?
      string(restart_from) + "/optimization_checkpoint" : checkpoint_dir;
   string history_file = output_parent_dir + "/optimization_history.txt";

   bool checkpoint_history_ok = true;
   if (checkpoint_history && myid == 0)
   {
      struct stat st;
      if (stat(checkpoint_history_root.c_str(), &st) != 0)
      {
         checkpoint_history_ok =
            (mkdir(checkpoint_history_root.c_str(), 0755) == 0);
      }
      else
      {
         checkpoint_history_ok = S_ISDIR(st.st_mode);
      }
      if (!checkpoint_history_ok)
      {
         cerr << "ERROR: Failed to create checkpoint-history directory: "
              << checkpoint_history_root << "\n";
      }
   }
   MPI_Bcast(&checkpoint_history_ok, 1, MPI_C_BOOL, 0, comm);
   if (!checkpoint_history_ok) { return 1; }

   if (myid == 0)
   {
      cout << "\n=== Output Configuration ===\n";
      cout << "Parent directory: " << output_parent_dir << "\n";
      cout << "ParaView output:  " << paraview_dir << "\n";
      cout << "Checkpoint:       " << checkpoint_dir << "\n";
      if (checkpoint_history)
      {
         cout << "Checkpoint history: " << checkpoint_history_root << "\n";
      }
      cout << "History file:     " << history_file << "\n";
      cout << "Restart mode:     " << (restart ? "YES" : "NO") << "\n";
      if (restart) { cout << "Restart source:   " << restart_checkpoint_dir << "\n"; }
      cout << "============================\n\n";
   }

   // The problem builds its own coarse mesh (file-based by default; generated
   // geometry for e.g. the cantilever).
   Mesh mesh = problem.CreateMesh();
   const int dim = mesh.Dimension();

   // Apply a problem-owned periodic identification in the transverse direction.
   if (problem.UsesPeriodicYBoundary() && dim == 2)
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

   const int state_order = problem.GetOrder();
   H1_FECollection state_fec(state_order, dim);
   H1_FECollection filter_fec(design_order, dim);
   const int control_order = max(0, design_order - 1);
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
   OptimizationCheckpoint restart_checkpoint(restart_checkpoint_dir, comm);
   OptimizationCheckpointMetadata ckpt_meta;
   const auto save_checkpoint_history =
      [&](const OptimizationCheckpointMetadata &meta, const Vector &design)
   {
      if (!checkpoint_history) { return true; }
      ostringstream iter_dir;
      iter_dir << checkpoint_history_root << "/iter_"
               << setfill('0') << setw(6) << meta.design_iteration;
      OptimizationCheckpoint snapshot(iter_dir.str(), comm);
      const bool saved = snapshot.Save(meta, design);
      if (!saved && myid == 0)
      {
         cerr << "WARNING: Failed to preserve checkpoint-history design rho^"
              << meta.design_iteration << "\n";
      }
      return saved;
   };
   bool restarting = false;
   int start_iteration = 0;

   if (restart && restart_checkpoint.Exists())
   {
      if (!restart_checkpoint.ValidateCompatibility(cfg.ref_levels, design_order,
                                                    ckpt_meta))
      {
         if (myid == 0)
         {
            cerr << "ERROR: Checkpoint exists but is incompatible. Exiting.\n";
            cerr << "       Remove -restart flag to start fresh run.\n";
         }
         return 1;
      }

      Vector rho_tv_restart(control_fes.GetTrueVSize());
      if (!restart_checkpoint.Load(rho_tv_restart, &ckpt_meta))
      {
         if (myid == 0)
         {
            cerr << "ERROR: Failed to load checkpoint design. Exiting.\n";
         }
         return 1;
      }
      rho.SetFromTrueDofs(rho_tv_restart);
      restarting = true;
      start_iteration = ckpt_meta.design_iteration;

      if (myid == 0)
      {
         cout << "\n=== RESTARTING FROM CHECKPOINT ===\n";
         cout << "Restored design rho^" << ckpt_meta.design_iteration
              << " after " << ckpt_meta.design_iteration
              << " completed MMA updates.\n";
         if (ckpt_meta.objective_valid_for_design)
         {
            cout << "Objective value for restored design: " << scientific
                 << ckpt_meta.objective << "\n";
         }
         else
         {
            cout << "The restored design has not yet been forward-evaluated "
                    "in its checkpoint metadata.\n";
         }
         cout << "Restored-design volume fraction: " << fixed
              << ckpt_meta.volume_fraction << "\n";
         cout << "State FE order: checkpoint " << ckpt_meta.fe_order
              << " -> current " << state_order << "\n";
         cout << "Design FE order: " << ckpt_meta.design_fe_order << "\n";
         cout << "Design restored as initial guess; MMA restarts fresh.\n";
         cout << "Resuming from iteration " << start_iteration << "\n";
         cout << "==================================\n\n";
      }
   }
   else if (restart && !restart_checkpoint.Exists())
   {
      if (myid == 0)
      {
         cerr << "ERROR: -restart specified but no checkpoint found in: "
              << restart_checkpoint_dir << "\n";
      }
      return 1;
   }

   rho_tilde = 0.0;

   toopt::PDEFilterOptions filter_opts;
   filter_opts.filter_radius = problem.GetFilterRadius();
   toopt::PDEFilter filter(filter_fes, control_fes, filter_opts);
   filter.Assemble();

   // --- Active/Passive Region Setup -----------------------------------------
   // Check if problem defines passive (non-designable) regions
   unique_ptr<Coefficient> passive_region_coef = problem.CreatePassiveRegionCoefficient();
   ConstantCoefficient one_coef(1.0);
   unique_ptr<Coefficient> active_region_coef;
   Array<int> active_tdof_list, passive_tdof_list;
   ParGridFunction passive_marker(&control_fes);
   // Passive regions frozen at the problem's reference density (default:
   // the volume fraction; individual experiments can pin a reference medium
   // independent of -vf).
   const real_t passive_rho_value = problem.GetPassiveDensity();
   HYPRE_BigInt global_active_control_dofs = control_dofs;

   if (passive_region_coef)
   {
      // Active region = 1 - passive region
      active_region_coef = std::make_unique<SumCoefficient>(
         one_coef, *passive_region_coef, 1.0, -1.0);

      // Identify which DOFs are active vs passive
      IdentifyActivePassiveDOFs(control_fes, *passive_region_coef,
                                active_tdof_list, passive_tdof_list,
                                passive_marker);

      // Get global DOF counts across all ranks
      HYPRE_BigInt global_control_dofs = control_fes.GlobalTrueVSize();
      HYPRE_BigInt local_active = active_tdof_list.Size();
      HYPRE_BigInt local_passive = passive_tdof_list.Size();
      HYPRE_BigInt global_active = 0, global_passive = 0;
      MPI_Allreduce(&local_active, &global_active, 1,
                    HYPRE_MPI_BIG_INT, MPI_SUM, comm);
      MPI_Allreduce(&local_passive, &global_passive, 1,
                    HYPRE_MPI_BIG_INT, MPI_SUM, comm);
      global_active_control_dofs = global_active;

      if (myid == 0)
      {
         cout << "Passive regions defined:\n";
         cout << "  Total control DOFs: " << global_control_dofs << "\n";
         cout << "  Active DOFs:  " << global_active << "\n";
         cout << "  Passive DOFs: " << global_passive << " (raw rho fixed at "
              << passive_rho_value << ")\n";
      }
   }

   // Assemble volume weights over active region only
   real_t domain_volume = 0.0;
   unique_ptr<HypreParVector> volume_weights =
      AssembleVolumeWeights(control_fes, domain_volume,
                           active_region_coef.get());
   MFEM_VERIFY(std::isfinite(domain_volume) && domain_volume > 0.0,
               "The active design region has non-positive measure.");

   // --- Prescribed inverse truth --------------------------------------------
   // Project rho_dagger to the actual raw-control space, freeze every passive
   // control at background density, and use the same discrete active-volume
   // weights as MMA.  Only this mesh/order-dependent value may initialize the
   // inclusion reconstruction or define its equality constraint.
   unique_ptr<ParGridFunction> truth_rho;
   unique_ptr<ParGridFunction> truth_rho_tilde;
   real_t prescribed_truth_volume =
      std::numeric_limits<real_t>::quiet_NaN();
   if (problem.HasReferenceTruth())
   {
      unique_ptr<Coefficient> truth_coefficient =
         problem.CreateTruthDensityCoefficient();
      MFEM_VERIFY(truth_coefficient,
                  "Problem advertised a reference truth but did not create it.");
      truth_rho = make_unique<ParGridFunction>(&control_fes);
      truth_rho->ProjectCoefficient(*truth_coefficient);

      Vector truth_rho_tv(control_fes.GetTrueVSize());
      truth_rho->GetTrueDofs(truth_rho_tv);
      if (passive_region_coef)
      {
         for (int i = 0; i < passive_tdof_list.Size(); i++)
         {
            truth_rho_tv[passive_tdof_list[i]] = passive_rho_value;
         }
      }
      int local_invalid_truth = 0;
      for (int i = 0; i < truth_rho_tv.Size(); i++)
      {
         if (!std::isfinite(truth_rho_tv[i]) ||
             truth_rho_tv[i] < 0.0 || truth_rho_tv[i] > 1.0)
         {
            local_invalid_truth = 1;
            break;
         }
      }
      int global_invalid_truth = 0;
      MPI_Allreduce(&local_invalid_truth, &global_invalid_truth, 1, MPI_INT,
                    MPI_MAX, comm);
      MFEM_VERIFY(global_invalid_truth == 0,
                  "Projected reference truth violates density bounds.");
      truth_rho->SetFromTrueDofs(truth_rho_tv);

      const real_t truth_volume =
         InnerProduct(comm, *volume_weights, truth_rho_tv);
      prescribed_truth_volume = truth_volume;
      const real_t truth_volume_fraction = truth_volume / domain_volume;
      problem.SetComputedTruthVolumeFraction(truth_volume_fraction);

      truth_rho_tilde = make_unique<ParGridFunction>(&filter_fes);
      *truth_rho_tilde = 0.0;
      filter.Mult(*truth_rho, *truth_rho_tilde);
      if (myid == 0)
      {
         cout << "Reference truth: discrete active volume fraction = "
              << setprecision(16) << truth_volume_fraction
              << ", active measure = " << domain_volume << "\n";
      }
   }

   const real_t target_volume = problem.HasReferenceTruth() ?
      prescribed_truth_volume : problem.GetVolumeFraction() * domain_volume;

   // =========================================================================
   // INITIALIZE DESIGN (skip if restarting - already loaded from checkpoint)
   // =========================================================================
   real_t ref_x_max = 0.0, ref_y_max = 0.0;
   problem.GetReferenceDomainExtents(ref_x_max, ref_y_max);
   if (!restarting)
   {
      if (!InitializeDesign(rho, design_init, problem.GetVolumeFraction(),
                            ref_x_max, ref_y_max, problem))
      {
         if (myid == 0)
         {
            cerr << "Error: unknown or unsupported -init value '"
                 << design_init
                 << "'. Use uniform, solid, void, gaussian, checkerboard, or modal-seed "
                    "(2D band converter only).\n";
         }
         return 1;
      }
   }

   // Enforce raw passive material after either fresh initialization or restart,
   // before every diagnostic/forward-only early exit and before filtering.
   if (passive_region_coef)
   {
      Vector initialized_rho_tv(control_fes.GetTrueVSize());
      rho.GetTrueDofs(initialized_rho_tv);
      for (int i = 0; i < passive_tdof_list.Size(); i++)
      {
         initialized_rho_tv[passive_tdof_list[i]] = passive_rho_value;
      }
      rho.SetFromTrueDofs(initialized_rho_tv);
   }

   // Nonuniform seeds are not generally volume-exact after Q0 projection.
   // Correct their raw active volume before the first MMA subproblem so that
   // the first step resolves topology rather than a spurious global-volume
   // violation. The Gaussian, checkerboard, and modal-seed bounds leave room
   // for this deterministic shift in their supported volume-fraction range.
   const string initial_design_name = ToLower(design_init);
   if (!restarting &&
       (initial_design_name == "modal-seed" || initial_design_name == "gaussian" ||
        initial_design_name == "checkerboard"))
   {
      Vector rho_tv(control_fes.GetTrueVSize());
      rho.GetTrueDofs(rho_tv);
      const real_t current_volume =
         InnerProduct(comm, *volume_weights, rho_tv);
      const real_t volume_shift =
         (target_volume - current_volume) / domain_volume;
      if (passive_region_coef)
      {
         for (int i = 0; i < active_tdof_list.Size(); i++)
         {
            rho_tv[active_tdof_list[i]] += volume_shift;
         }
      }
      else
      {
         rho_tv += volume_shift;
      }
      int local_out_of_bounds = 0;
      for (int i = 0; i < rho_tv.Size(); i++)
      {
         if (rho_tv[i] < -1e-12 || rho_tv[i] > 1.0 + 1e-12)
         {
            local_out_of_bounds = 1;
            break;
         }
      }
      int global_out_of_bounds = 0;
      MPI_Allreduce(&local_out_of_bounds, &global_out_of_bounds, 1, MPI_INT,
                    MPI_MAX, comm);
      MFEM_VERIFY(global_out_of_bounds == 0,
                  "Initial-design volume correction violates raw density bounds.");
      rho.SetFromTrueDofs(rho_tv);
      if (myid == 0)
      {
         cout << "Initial design '" << initial_design_name
              << "': raw active-volume correction=" << scientific
              << volume_shift << "\n";
      }
   }

   // Keep the displayed/initial filtered field consistent with the raw design
   // supplied to the Helmholtz filter.
   filter.Mult(rho, rho_tilde);

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

   const int num_steps =
      max(1, static_cast<int>(ceil(problem.GetFinalTime() /
                                  problem.GetTimeStep())));
   const real_t dt_eff = problem.GetFinalTime() / num_steps;

   // Generate fixed synthetic observations once, before the reconstruction
   // objective or solver is built.  The high-order state lives only inside the
   // generator; the retained history contains reconstruction-space boundary
   // values and is immutable throughout MMA/adjoint sweeps.
   shared_ptr<const BoundaryTraceHistory> reference_trace_history;
   ReferenceBoundaryDataMetadata reference_metadata;
   int resolved_reference_order = 0;
   real_t resolved_reference_dt = 0.0;
   if (requires_reference_data)
   {
      MFEM_VERIFY(truth_rho_tilde,
                  "Reference boundary data requires a filtered truth field.");
      Array<int> observation_attributes;
      problem.GetObservationBoundaryAttributes(observation_attributes);
      Array<int> observation_marker =
         MakeBoundaryMarker(pmesh, observation_attributes);
      resolved_reference_order =
         reference_order < 0 ? state_order + 1 : reference_order;
      resolved_reference_dt =
         reference_time_step < 0.0 ? 0.25 * dt_eff : reference_time_step;

      ReferenceBoundaryDataGenerator reference_generator(
         problem, state_fes, *truth_rho_tilde, observation_marker,
         num_steps, dt_eff, resolved_reference_order, resolved_reference_dt,
         damping,
         // Synthetic data use the higher-fidelity consistent mass model even
         // when a reconstruction experiment opts into row-sum lumping.
         MassSolverType::ITERATIVE);
      reference_trace_history = reference_generator.Generate();
      reference_metadata = reference_generator.Metadata();
      resolved_reference_dt = reference_metadata.effective_time_step;
      problem.SetBoundaryTraceHistory(reference_trace_history);
      if (myid == 0)
      {
         cout << defaultfloat;
         ofstream reference_history(
            output_parent_dir + "/reference_boundary_data_history.txt");
         reference_history
            << "# Elastic inclusion reference boundary data\n"
            << "# Output directory: " << output_parent_dir << "\n";
         ostringstream reference_problem_summary;
         problem.PrintSummary(reference_problem_summary);
         const MaterialParams &reference_material =
            problem.GetMaterialParams();
         const BoundaryLoadSpec &reference_load = problem.GetBoundaryLoad();
         reference_history << "# " << reference_problem_summary.str()
            << "# Reconstruction: state_order=" << state_order
            << ", design_filter_order=" << design_order
            << ", raw_control_order=" << control_order
            << ", coarse_steps=" << num_steps
            << ", coarse_dt=" << setprecision(16) << dt_eff
            << ", final_time=" << problem.GetFinalTime() << "\n"
            << "# Inversion: active_volume=" << domain_volume
            << ", truth_volume=" << prescribed_truth_volume
            << ", truth_volume_fraction=" << problem.GetVolumeFraction()
            << ", filter_radius=" << problem.GetFilterRadius()
            << ", SIMP=[r_min=" << reference_material.r_min
            << ", r_max=" << reference_material.r_max
            << ", p=" << reference_material.simp_p << "]\n"
            << "# Load: profile="
            << LoadTimeProfileName(reference_load.time_profile)
            << ", amplitude=" << reference_load.amplitude
            << ", duration=" << reference_load.duration
            << ", frequency=" << reference_load.frequency
            << ", phase=" << reference_load.phase
            << ", damping=" << (damping ? "enabled" : "disabled") << "\n"
            << "# Reference: state_order="
            << reference_metadata.state_order
            << ", steps=" << reference_metadata.reference_steps
            << ", requested_dt=" << setprecision(16)
            << reference_metadata.requested_time_step
            << ", effective_dt="
            << reference_metadata.effective_time_step
            << ", steps_per_coarse_half_step="
            << reference_metadata.reference_steps_per_half_step
            << ", mass=consistent, damping="
            << (reference_metadata.damping_enabled ? "enabled" : "disabled")
            << "\n# Reference state global true DOFs: "
            << reference_metadata.global_state_true_dofs
            << "\n# Trace history bytes: local_rank0="
            << reference_metadata.local_trace_memory_bytes
            << ", maximum_per_rank="
            << reference_metadata.maximum_trace_memory_bytes_per_rank
            << ", global="
            << reference_metadata.global_trace_memory_bytes
            << "\n# Reference forward seconds: "
            << reference_metadata.forward_seconds
            << "\n# Reference convergence audit: pending (not part of "
               "the first common-mesh implementation milestone)\n";
      }

      // One static density snapshot is available in every inverse mode,
      // including the forward-only and reference-only early exits.
      if (paraview)
      {
         ParaViewDataCollection density_dc(
            "ElasticInclusionDensities", &pmesh);
         density_dc.SetPrefixPath(paraview_dir);
         density_dc.SetLevelsOfDetail(design_order);
         density_dc.SetDataFormat(VTKFormat::BINARY);
         density_dc.SetHighOrderOutput(true);
         density_dc.RegisterField("rho_reconstruction_raw", &rho);
         density_dc.RegisterField("rho_reconstruction_filtered", &rho_tilde);
         density_dc.RegisterField("rho_truth_raw", truth_rho.get());
         density_dc.RegisterField("rho_truth_filtered", truth_rho_tilde.get());
         if (passive_region_coef)
         {
            density_dc.RegisterField("passive_region", &passive_marker);
         }
         density_dc.SetCycle(0);
         density_dc.SetTime(0.0);
         density_dc.Save();
      }

      if (reference_only)
      {
         if (myid == 0)
         {
            cout << "\n=== Reference-Only Complete ===\n";
            problem.PrintSummary(cout);
            cout << "Reference order: " << resolved_reference_order
                 << ", reference dt: " << resolved_reference_dt << "\n"
                 << "Trace memory: global "
                 << reference_metadata.global_trace_memory_bytes
                 << " bytes, maximum/rank "
                 << reference_metadata.maximum_trace_memory_bytes_per_rank
                 << " bytes\n"
                 << "Configuration: " << output_parent_dir
                 << "/reference_boundary_data_history.txt\n"
                 << "No reconstruction solve, adjoint, MMA update, or "
                    "optimization checkpoint was written.\n"
                 << "================================\n";
         }
         return 0;
      }
   }

   // Sponge-layer damping coefficient + absorbing-boundary impedance, assembled
   // by the problem from the material and damping parameters.
   unique_ptr<DampingFieldBase> damping_field = problem.CreateDampingField(damping);
   Coefficient &gamma_coef = damping_field->GetCoefficient();
   const real_t impedance = damping_field->GetImpedance();
   if (myid == 0 && damping) { damping_field->PrintSummary(cout); }

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

   if (adjoint_mode == TransientAdjointMode::CONTINUOUS &&
       adjoint_coarsening > 1 && num_steps % adjoint_coarsening != 0)
   {
      if (myid == 0)
      {
         cerr << "Error: N_f=" << num_steps
              << " must be divisible by -adjoint-coarsening "
              << adjoint_coarsening << ".\n";
      }
      return 1;
   }
   if (adjoint_mode == TransientAdjointMode::CONTINUOUS &&
       adjoint_refinement > 1 &&
       num_steps > std::numeric_limits<int>::max() / adjoint_refinement)
   {
      if (myid == 0)
      {
         cerr << "Error: refined adjoint grid exceeds the supported step count.\n";
      }
      return 1;
   }
   const int selected_adjoint_steps =
      adjoint_refinement > 1 ? num_steps * adjoint_refinement :
      (adjoint_coarsening > 1 ? num_steps / adjoint_coarsening : num_steps);
   const real_t selected_adjoint_dt =
      problem.GetFinalTime() / selected_adjoint_steps;
   const NestedTimeGridRelation selected_grid_relation =
      adjoint_refinement > 1 ? NestedTimeGridRelation::ADJOINT_FINER :
      (adjoint_coarsening > 1 ? NestedTimeGridRelation::FORWARD_FINER :
       NestedTimeGridRelation::SAME);

   if (myid == 0)
   {
      cout << "\n=== Transient "
           << (forward_only ? "Forward-Only" : "TopOpt (MMA)") << " ===\n";
      cout << "Mesh: " << problem.GetMeshFile() << "\n";
      cout << "Refinement levels: " << problem.GetRefinementLevel() << "\n";
      cout << "FE orders: state H1 = " << state_order
           << ", design/filter H1 = " << design_order
           << ", control L2 = " << control_order << "\n";
      cout << "State DOFs:   " << state_dofs << "\n";
      cout << "Filter DOFs:  " << filter_dofs << " (H1 rho_tilde)\n";
      cout << "Control DOFs: " << control_dofs << " (L2 rho)\n";
      cout << "Target volume fraction: " << problem.GetVolumeFraction() << "\n";
      cout << "Filter radius: " << problem.GetFilterRadius() << "\n";
      cout << "Time interval: [0, " << problem.GetFinalTime() << "],  steps: " << num_steps
           << ",  dt_eff: " << dt_eff << "\n";
      if (!forward_only)
      {
         cout << "Max MMA iterations: " << problem.GetMaxIterations()
              << ",  move limit: " << problem.GetMoveLimit()
              << ",  stop tol (L1 dRho): " << problem.GetChangeTolerance() << "\n";
         cout << "Adjoint gradient: "
              << TransientAdjointModeName(adjoint_mode);
         if (adjoint_mode == TransientAdjointMode::CONTINUOUS)
         {
            cout << ", " << NestedTimeGridRelationName(selected_grid_relation)
                 << ", N_a = " << selected_adjoint_steps
                 << ", dt_a = " << selected_adjoint_dt
                 << ", m = " << adjoint_refinement
                 << ", q = " << adjoint_coarsening;
         }
         cout << ", trajectory storage = "
              << TrajectoryStorageModeName(trajectory_storage)
              << ", objective quadrature = "
              << (rk4_stage_objective ? "rk4-stage" : "legacy") << "\n";
      }
   }

   // Bundle the invariant setup once. In forward-only mode this is the only
   // solver path used: no optimizer, adjoint sweep, or optimization checkpoint.
   MassSolverType mass_solver = use_iterative_mass ?
                                MassSolverType::ITERATIVE : MassSolverType::LUMPED;
   TransientDesignSolver design_solver(
      state_fes, filter_fes, control_fes, filter, gamma_coef,
      exterior_bdr_attr, essential_bdr_attr, *objective, mat, load_spec,
      *load_coef, impedance, num_steps, dt_eff, mass_solver, rho, rho_tilde,
      num_checkpoints, adjoint_mode, adjoint_refinement,
      trajectory_storage, adjoint_coarsening, rk4_stage_objective);

   if (rk4_adjoint_comparison)
   {
      Vector rho_tv_comparison(control_fes.GetTrueVSize());
      rho.GetTrueDofs(rho_tv_comparison);
      design_solver.AnalyzeRK4AdjointComparison(
         rho_tv_comparison, *volume_weights, output_parent_dir,
         passive_region_coef ? &active_tdof_list : nullptr,
         rk4_taylor_levels, rk4_taylor_epsilon,
         rk4_taylor_linf_normalized);
      return 0;
   }

   if (rhs_spectrum)
   {
      Vector rho_tv_spectrum(control_fes.GetTrueVSize());
      rho.GetTrueDofs(rho_tv_spectrum);
      design_solver.AnalyzeRightHandSideSpectra(
         rho_tv_spectrum, output_parent_dir, paraview);
      if (myid == 0)
      {
         cout << "Spectrum table: " << output_parent_dir
              << "/rhs_spectrum.csv\n";
      }
      return 0;
   }

   if (continuous_adjoint_check)
   {
      Vector rho_tv_continuous(control_fes.GetTrueVSize());
      rho.GetTrueDofs(rho_tv_continuous);
      design_solver.AnalyzeContinuousAdjointCoarsening(
         rho_tv_continuous, output_parent_dir, maximum_adjoint_coarsening);
      return 0;
   }

   if (continuous_adjoint_refinement_check)
   {
      Vector rho_tv_continuous(control_fes.GetTrueVSize());
      rho.GetTrueDofs(rho_tv_continuous);
      design_solver.AnalyzeContinuousAdjointRefinement(
         rho_tv_continuous, output_parent_dir,
         minimum_adjoint_refinement,
         maximum_adjoint_refinement,
         passive_region_coef ? &active_tdof_list : nullptr);
      return 0;
   }

   if (continuous_storage_check)
   {
      Vector rho_tv_storage(control_fes.GetTrueVSize());
      rho.GetTrueDofs(rho_tv_storage);
      design_solver.AnalyzeContinuousStorageEquivalence(
         rho_tv_storage, output_parent_dir,
         passive_region_coef ? &active_tdof_list : nullptr);
      return 0;
   }

   if (forward_only)
   {
      if ((forward_modal_probe || forward_modal_history) && paraview)
      {
         if (myid == 0)
         {
            cerr << "Error: forward modal diagnostics cannot currently be "
                    "combined with -pv. Run them without ParaView.\n";
         }
         return 1;
      }

      Vector rho_tv_forward(control_fes.GetTrueVSize());
      rho.GetTrueDofs(rho_tv_forward);
      real_t J = 0.0;
      int frames_saved = 0;
      real_t terminal_modal_projection = 0.0;
      real_t terminal_target_modal_projection = 0.0;
      bool modal_projection_evaluated = false;
      bool modal_history_evaluated = false;
      real_t launched_demodulated_amplitude = 0.0;
      real_t target_demodulated_amplitude = 0.0;
      real_t launched_demodulated_phase = 0.0;
      real_t target_demodulated_phase = 0.0;
      real_t modal_amplitude_ratio = 0.0;

      if (paraview)
      {
         const string wave_collection_name = "wave_forward";
         const string wave_full_dir = output_parent_dir + "/ParaView/" + wave_collection_name;
         ParaViewDataCollection wave_dc(wave_collection_name.c_str(), &pmesh);
         wave_dc.SetLevelsOfDetail(state_order);
         wave_dc.SetDataFormat(VTKFormat::BINARY);
         wave_dc.SetHighOrderOutput(true);
         wave_dc.SetPrefixPath((output_parent_dir + "/ParaView").c_str());

         ParGridFunction u_gf(&state_fes);
         wave_dc.RegisterField("displacement", &u_gf);

         const int wave_viz_freq = max(1, num_steps / 20);
         const auto save_wave = [&](int step, real_t time, const Vector &state)
         {
            const int half_size = state.Size() / 2;
            Vector u_vec(state.GetData(), half_size);
            u_gf.SetFromTrueDofs(u_vec);
            wave_dc.SetCycle(step);
            wave_dc.SetTime(time);
            wave_dc.Save();
            frames_saved++;
         };
         J = design_solver.ForwardVisualizationSweepStream(
            rho_tv_forward, wave_viz_freq, save_wave);

         if (myid == 0)
         {
            cout << "    Saved " << frames_saved << " forward-wave frames to: "
                 << wave_full_dir << "/\n";
         }
      }
      else
      {
         if (forward_modal_probe || forward_modal_history)
         {
            unique_ptr<VectorCoefficient> launched_modal_probe =
               problem.CreateForwardModalProbe();
            MFEM_VERIFY(launched_modal_probe,
                        "The selected problem does not provide a forward modal probe.");
            unique_ptr<VectorCoefficient> target_modal_probe;
            if (forward_modal_history)
            {
               target_modal_probe = problem.CreateTargetModalProbe();
               MFEM_VERIFY(
                  target_modal_probe,
                  "The selected problem does not provide a converted-mode probe.");
            }

            ParGridFunction u_gf(&state_fes);
            vector<real_t> modal_times;
            vector<real_t> launched_projections;
            vector<real_t> target_projections;
            ofstream modal_csv;
            if (myid == 0 && forward_modal_history)
            {
               modal_csv.open(output_parent_dir + "/forward_modal_history.csv");
               MFEM_VERIFY(modal_csv.is_open(),
                           "Could not open the forward modal-history CSV.");
               modal_csv << "step,time,launched_mode_projection,"
                            "converted_mode_projection\n";
            }

            const auto sample_modal_output =
               [&](int step, real_t time, const Vector &state)
            {
               const int half_size = state.Size() / 2;
               Vector u_vec(const_cast<real_t *>(state.GetData()), half_size);
               u_gf.SetFromTrueDofs(u_vec);
               const real_t launched_projection =
                  EvaluateDisplacementModalProjection(
                     state_fes, u_gf, *launched_modal_probe, comm);
               real_t target_projection = 0.0;
               if (forward_modal_history)
               {
                  target_projection = EvaluateDisplacementModalProjection(
                     state_fes, u_gf, *target_modal_probe, comm);
                  if (myid == 0)
                  {
                     modal_times.push_back(time);
                     launched_projections.push_back(launched_projection);
                     target_projections.push_back(target_projection);
                     modal_csv << step << ',' << std::setprecision(16)
                               << time << ',' << launched_projection << ','
                               << target_projection << '\n';
                  }
               }
               if (step == num_steps)
               {
                  terminal_modal_projection = launched_projection;
                  terminal_target_modal_projection = target_projection;
                  modal_projection_evaluated = true;
               }
            };

            J = design_solver.ForwardVisualizationSweepStream(
               rho_tv_forward,
               forward_modal_history ? modal_sample_every : num_steps,
               sample_modal_output);

            if (myid == 0 && forward_modal_history)
            {
               MFEM_VERIFY(modal_times.size() >= 2,
                           "Modal history requires at least two time samples.");
               constexpr real_t two_pi =
                  2.0 * 3.1415926535897932384626433832795;
               const real_t omega = two_pi * load_spec.frequency;
               real_t launched_real = 0.0, launched_imag = 0.0;
               real_t target_real = 0.0, target_imag = 0.0;
               for (std::size_t i = 1; i < modal_times.size(); i++)
               {
                  const real_t dt_sample = modal_times[i] - modal_times[i - 1];
                  const real_t cos_left = std::cos(omega * modal_times[i - 1]);
                  const real_t sin_left = std::sin(omega * modal_times[i - 1]);
                  const real_t cos_right = std::cos(omega * modal_times[i]);
                  const real_t sin_right = std::sin(omega * modal_times[i]);
                  launched_real += 0.5 * dt_sample *
                     (launched_projections[i - 1] * cos_left +
                      launched_projections[i] * cos_right);
                  launched_imag -= 0.5 * dt_sample *
                     (launched_projections[i - 1] * sin_left +
                      launched_projections[i] * sin_right);
                  target_real += 0.5 * dt_sample *
                     (target_projections[i - 1] * cos_left +
                      target_projections[i] * cos_right);
                  target_imag -= 0.5 * dt_sample *
                     (target_projections[i - 1] * sin_left +
                      target_projections[i] * sin_right);
               }
               const real_t observation_time =
                  modal_times.back() - modal_times.front();
               MFEM_VERIFY(observation_time > 0.0,
                           "Modal history has a non-positive time span.");
               launched_demodulated_amplitude =
                  2.0 * std::hypot(launched_real, launched_imag) /
                  observation_time;
               target_demodulated_amplitude =
                  2.0 * std::hypot(target_real, target_imag) /
                  observation_time;
               launched_demodulated_phase =
                  std::atan2(launched_imag, launched_real);
               target_demodulated_phase =
                  std::atan2(target_imag, target_real);
               if (launched_demodulated_amplitude > 0.0)
               {
                  modal_amplitude_ratio = target_demodulated_amplitude /
                                          launched_demodulated_amplitude;
               }
               modal_history_evaluated = true;
               modal_csv.flush();
            }
         }
         else
         {
            J = design_solver.Objective(rho_tv_forward, "forward");
         }
      }

      if (myid == 0)
      {
         cout << "\n=== Forward-Only Complete ===\n";
         cout << "Objective J = " << scientific << setprecision(8) << J << "\n";
         if (modal_projection_evaluated)
         {
            cout << "Terminal launched-mode projection = "
                 << scientific << setprecision(16)
                 << terminal_modal_projection << "\n";
         }
         if (modal_history_evaluated)
         {
            cout << "Terminal converted-mode projection = "
                 << scientific << setprecision(16)
                 << terminal_target_modal_projection << "\n"
                 << "Carrier-demodulated launched amplitude = "
                 << launched_demodulated_amplitude
                 << ", phase = " << launched_demodulated_phase << "\n"
                 << "Carrier-demodulated converted amplitude = "
                 << target_demodulated_amplitude
                 << ", phase = " << target_demodulated_phase << "\n"
                 << "Converted/launched modal amplitude ratio = "
                 << modal_amplitude_ratio << "\n"
                 << "Modal history: " << output_parent_dir
                 << "/forward_modal_history.csv\n";
         }
         cout << "No MMA update, adjoint, or optimization checkpoint was written.\n";
         cout << "=============================\n";
      }
      return 0;
   }

   // --- MMA setup -----------------------------------------------------------
   // MMA works with active (designable) DOFs only
   const int n_full = control_fes.GetTrueVSize();
   const int n_active = passive_region_coef ? active_tdof_list.Size() : n_full;
   // Wave-shielding objectives can reduce the response by removing material.
   // Enforce an active-region volume equality rather than the compliance-style
   // upper budget alone. MMA represents h=V/V* - 1=0 internally as (+h,-h),
   // with an unconstrained equality multiplier; treating those rows as two
   // ordinary inequalities makes the two dual barriers oppose one another at
   // feasibility and can stall the topology update.
   const int num_con = 2;

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

   // Packed volume-equality gradients in the required (+h,-h) order.
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
   dfidx[1] = dvol_active;
   dfidx[1] *= -1.0;

   mfem_mma::MMAOptimizerParallel mma =
      mfem_mma::MMAOptimizerParallel::WithEqualities(
         comm, n_active, /*n_ineq=*/0, /*n_eq=*/1, rho_active);
   mma.SetAsymptotes(0.5, 0.7, 1.2);

   Vector rho_active_min(n_active), rho_active_max(n_active);

   ParaViewDataCollection paraview_dc("TopOptTransient", &pmesh);
   if (paraview)
   {
      paraview_dc.SetPrefixPath(paraview_dir);  // Use subdirectory in parent
      paraview_dc.SetLevelsOfDetail(design_order);
      paraview_dc.SetDataFormat(VTKFormat::BINARY);
      paraview_dc.SetHighOrderOutput(true);
      paraview_dc.RegisterField("rho", &rho);
      paraview_dc.RegisterField("rho_tilde", &rho_tilde);
      if (truth_rho && truth_rho_tilde)
      {
         paraview_dc.RegisterField("rho_truth_raw", truth_rho.get());
         paraview_dc.RegisterField("rho_truth_filtered",
                                   truth_rho_tilde.get());
      }
      if (passive_region_coef)
      {
         paraview_dc.RegisterField("passive_region", &passive_marker);
      }
   }

   ofstream history;
   if (myid == 0)
   {
      const auto write_history_run_configuration = [&]()
      {
         const NestedTimeGrid &grid = design_solver.TimeGrid();
         history << "# Load: profile="
                 << LoadTimeProfileName(load_spec.time_profile)
                 << ", amplitude=" << load_spec.amplitude
                 << ", duration=" << load_spec.duration
                 << ", frequency=" << load_spec.frequency
                 << ", phase=" << load_spec.phase << "\n";
         ostringstream problem_summary;
         problem.PrintSummary(problem_summary);
         if (!problem_summary.str().empty())
         {
            history << "# " << problem_summary.str();
         }
         history << "# Adjoint mode: "
                 << TransientAdjointModeName(adjoint_mode) << "\n";
         history << "# Objective quadrature: "
                 << (rk4_stage_objective ? "rk4-stage" : "legacy") << "\n";
         history << "# Time grid: relation="
                 << NestedTimeGridRelationName(grid.relation)
                 << ", N_f=" << grid.forward_steps
                 << ", N_a=" << grid.adjoint_steps
                 << ", dt_f=" << std::setprecision(16) << grid.dt_forward
                 << ", dt_a=" << grid.dt_adjoint
                 << ", m=" << adjoint_refinement
                 << ", q=" << adjoint_coarsening << "\n";
         history << "# Trajectory storage: "
                 << TrajectoryStorageModeName(trajectory_storage)
                 << ", checkpoints="
                 << design_solver.NumTrajectoryCheckpoints()
                 << ", estimated_MB_per_rank="
                 << design_solver.EstimatedTrajectoryMemoryMB() << "\n";
         history << "# Discretization: refinement=" << cfg.ref_levels
                 << ", state_order=" << state_order
                 << ", design_order=" << design_order
                 << ", control_order=" << control_order
                 << ", state_dofs=" << state_dofs
                 << ", filter_dofs=" << filter_dofs
                 << ", control_dofs=" << control_dofs << "\n";
         if (requires_reference_data)
         {
            history << "# Reference data: state_order="
                    << reference_metadata.state_order
                    << ", steps=" << reference_metadata.reference_steps
                    << ", requested_dt="
                    << reference_metadata.requested_time_step
                    << ", effective_dt="
                    << reference_metadata.effective_time_step
                    << ", steps_per_coarse_half_step="
                    << reference_metadata.reference_steps_per_half_step
                    << ", mass=consistent, damping="
                    << (reference_metadata.damping_enabled ?
                        "enabled" : "disabled")
                    << ", global_state_true_dofs="
                    << reference_metadata.global_state_true_dofs << "\n";
            history << "# Reference trace history: samples="
                    << (reference_trace_history ?
                        reference_trace_history->SampleCount() : 0)
                    << ", local_rank0_bytes="
                    << reference_metadata.local_trace_memory_bytes
                    << ", maximum_per_rank_bytes="
                    << reference_metadata.maximum_trace_memory_bytes_per_rank
                    << ", global_bytes="
                    << reference_metadata.global_trace_memory_bytes
                    << ", generation_seconds="
                    << reference_metadata.forward_seconds << "\n";
            history << "# Reference convergence audit: pending (not part of "
                       "the first common-mesh implementation milestone)\n";
         }
         history << "# Physics: mass="
                 << (use_iterative_mass ? "consistent" : "lumped")
                 << ", damping=" << (damping ? "enabled" : "disabled")
                 << "\n";
         history << "# SIMP: s(rho_tilde)=" << mat.r_min << "+("
                 << mat.r_max - mat.r_min << ") rho_tilde^" << mat.simp_p
                 << "\n";
         history << "# Design: initialization=" << design_init
                 << ", target_volume_fraction="
                 << problem.GetVolumeFraction()
                 << ", filter_radius=" << problem.GetFilterRadius()
                 << ", active_dofs=" << global_active_control_dofs
                 << ", total_control_dofs=" << control_dofs
                 << ", passive_density=" << passive_rho_value << "\n";
         history << "# Optimizer: max_updates=" << problem.GetMaxIterations()
                 << ", move_limit=" << problem.GetMoveLimit()
                 << ", change_tolerance=" << problem.GetChangeTolerance()
                 << ", mma_objective_scale=" << mma_objective_scale
                 << "\n";
         history << "# MPI ranks: " << mpi_ranks << "\n";
         if (adjoint_mode == TransientAdjointMode::CONTINUOUS)
         {
            history << "# Forward reconstruction: cubic Hermite with physical "
                       "endpoint slopes\n";
         }
      };
      const auto write_history_header = [&]()
      {
         history << "# Transient Topology Optimization History\n";
         history << "# Output directory: " << output_parent_dir << "\n";
         history << "# Problem: " << problem_name << "\n";
         write_history_run_configuration();
         history << "#\n";
         history << "# Row semantics: row i evaluates rho^(i-1), records its "
                    "objective/gradients, and then produces rho^i.\n";
         history << "# iter    J                 vol_frac      g_upper       g_lower"
                    "       grad_raw_l2   grad_active_raw_l2  "
                    "grad_filtered_l2  dRho_L1      forward_s     "
                    "adjoint_s     trajectory_MB  controller_blocks  "
                    "local_blocks  controller_intervals  local_intervals\n";
      };

      if (restarting)
      {
         // A same-directory restart appends to its existing history.  A
         // -restart-from branch normally has a new output directory and no
         // history yet, so give that branch the complete standalone header
         // before recording its checkpoint provenance.
         bool existing_history_is_nonempty = false;
         {
            ifstream existing_history(history_file, ios::binary | ios::ate);
            existing_history_is_nonempty =
               existing_history.good() && existing_history.tellg() > 0;
         }
         if (existing_history_is_nonempty)
         {
            history.open(history_file, ios::app);
         }
         else
         {
            history.open(history_file);
            write_history_header();
         }
         history << "\n# === RESTART at iteration " << start_iteration << " ===\n";
         history << "# Restart checkpoint: " << restart_checkpoint_dir << "\n";
         history << "# Source checkpoint design iteration: "
                 << ckpt_meta.design_iteration
                 << "\n";
         write_history_run_configuration();
      }
      else
      {
         // NEW history file
         history.open(history_file);
         write_history_header();
      }
   }

   // --- Optimization loop ---------------------------------------------------

   GridFunctionCoefficient rho_cf(&rho);
   int k = start_iteration;  // Start from checkpoint iteration (or 0 if fresh)
   real_t iterationError = 1.0;   // fresh MMA on restart -> fresh stop test
   real_t final_design_objective =
      std::numeric_limits<real_t>::quiet_NaN();
   bool final_design_objective_available = false;

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
      const Vector &dJ_drho_tilde = design_solver.FilteredDesignGradient();
      real_t local_gradient_norm_sq[2] =
      {
         dJ_drho_full * dJ_drho_full,
         dJ_drho_tilde * dJ_drho_tilde
      };
      real_t global_gradient_norm_sq[2] = {0.0, 0.0};
      MPI_Allreduce(local_gradient_norm_sq, global_gradient_norm_sq, 2,
                    MPITypeMap<real_t>::mpi_type, MPI_SUM, comm);
      const real_t raw_gradient_norm =
         std::sqrt(global_gradient_norm_sq[0]);
      const real_t filtered_gradient_norm =
         std::sqrt(global_gradient_norm_sq[1]);
      const ContinuousStorageTelemetry iteration_telemetry =
         design_solver.StorageTelemetry();
      design_solver.ReleasePhysicsIterationStorage();       // free matrices + REVOLVE snapshots

      // Never let an unstable forward/adjoint sweep reach MMA. A NaN objective
      // previously drove every active density through a bogus update and was
      // then committed as an apparently restartable optimization checkpoint.
      int local_nonfinite =
         (std::isfinite(J) && std::isfinite(raw_gradient_norm) &&
          std::isfinite(filtered_gradient_norm)) ? 0 : 1;
      for (int i = 0; i < dJ_drho_full.Size(); i++)
      {
         if (!std::isfinite(dJ_drho_full[i]))
         {
            local_nonfinite = 1;
            break;
         }
      }
      for (int i = 0; i < dJ_drho_tilde.Size(); i++)
      {
         if (!std::isfinite(dJ_drho_tilde[i]))
         {
            local_nonfinite = 1;
            break;
         }
      }
      int global_nonfinite = 0;
      MPI_Allreduce(&local_nonfinite, &global_nonfinite, 1, MPI_INT, MPI_MAX,
                    comm);
      MFEM_VERIFY(global_nonfinite == 0,
                  "Non-finite objective/design gradient; refusing the MMA "
                  "update and optimization checkpoint.");

      // Extract gradients for active DOFs only
      if (passive_region_coef)
      {
         MapFullToActive(dJ_drho_full, active_tdof_list, dJ_drho_active);
      }
      else
      {
         dJ_drho_active = dJ_drho_full;
      }
      const real_t local_active_gradient_norm_sq =
         dJ_drho_active * dJ_drho_active;
      real_t global_active_gradient_norm_sq = 0.0;
      MPI_Allreduce(&local_active_gradient_norm_sq,
                    &global_active_gradient_norm_sq, 1,
                    MPITypeMap<real_t>::mpi_type, MPI_SUM, comm);
      const real_t active_raw_gradient_norm =
         std::sqrt(global_active_gradient_norm_sq);
      MFEM_VERIFY(std::isfinite(active_raw_gradient_norm),
                  "Active raw-design gradient norm is non-finite.");

      // Volume constraint and current fraction (over active region only)
      const real_t cur_volume = InnerProduct(comm, *volume_weights, rho_tv_full);
      const real_t cur_vol_frac = cur_volume / domain_volume;
      fival(0) = cur_volume / target_volume - 1.0;  // +h
      fival(1) = -fival(0);                         // -h

      // Box constraints with move limits (active DOFs only)
      rho_active_old = rho_active;
      for (int i = 0; i < n_active; i++)
      {
         rho_active_min[i] = max(real_t(0.0), rho_active[i] - problem.GetMoveLimit());
         rho_active_max[i] = min(real_t(1.0), rho_active[i] + problem.GetMoveLimit());
      }

      // A positive scalar changes only the conditioning of MMA's local
      // rational model. It does not change the physical objective or its
      // stationary points, and is valuable when a receiver-energy objective
      // is many orders of magnitude smaller than the design variables.
      Vector mma_objective_gradient(dJ_drho_active);
      mma_objective_gradient *= mma_objective_scale;

      // MMA outer iteration (minimizes J subject to the packed volume
      // equality) - active DOFs only.
      mma.Update(rho_active, mma_objective_gradient,
                 mma_objective_scale * J, fival, dfidx,
                rho_active_min, rho_active_max);

      // The equality-aware MMA subproblem is normally feasible to solver
      // tolerance.  Project its trial point onto the *exact* active volume
      // equality nevertheless: in a long, strongly scaled transient run the
      // internal dual solve can otherwise accumulate an infeasible volume
      // drift.  The water-filling shift preserves both the box constraints
      // and this iteration's MMA move bounds.  Since the pre-update design is
      // feasible, the target lies in the interval obtained by saturating the
      // lower and upper move bounds.
      const auto bounded_active_volume = [&](const real_t shift)
      {
         real_t local_volume = 0.0;
         for (int i = 0; i < n_active; i++)
         {
            const real_t value = min(rho_active_max[i],
                                     max(rho_active_min[i],
                                         rho_active[i] + shift));
            local_volume += target_volume * dvol_active[i] * value;
         }
         real_t global_volume = 0.0;
         MPI_Allreduce(&local_volume, &global_volume, 1,
                       MPITypeMap<real_t>::mpi_type, MPI_SUM, comm);
         return global_volume;
      };
      const real_t lower_volume = bounded_active_volume(-1.0);
      const real_t upper_volume = bounded_active_volume(1.0);
      const real_t volume_tolerance =
         128.0 * numeric_limits<real_t>::epsilon() * max(target_volume, 1.0);
      MFEM_VERIFY(lower_volume <= target_volume + volume_tolerance &&
                  upper_volume >= target_volume - volume_tolerance,
                  "Active box/move bounds cannot satisfy the volume equality.");
      real_t lower_shift = -1.0;
      real_t upper_shift = 1.0;
      for (int bisection_step = 0; bisection_step < 64; bisection_step++)
      {
         const real_t mid_shift = 0.5 * (lower_shift + upper_shift);
         if (bounded_active_volume(mid_shift) < target_volume)
         {
            lower_shift = mid_shift;
         }
         else
         {
            upper_shift = mid_shift;
         }
      }
      const real_t volume_shift = 0.5 * (lower_shift + upper_shift);
      for (int i = 0; i < n_active; i++)
      {
         rho_active[i] = min(rho_active_max[i],
                             max(rho_active_min[i],
                                 rho_active[i] + volume_shift));
      }

      local_nonfinite = 0;
      for (int i = 0; i < rho_active.Size(); i++)
      {
         if (!std::isfinite(rho_active[i]))
         {
            local_nonfinite = 1;
            break;
         }
      }
      MPI_Allreduce(&local_nonfinite, &global_nonfinite, 1, MPI_INT, MPI_MAX,
                    comm);
      MFEM_VERIFY(global_nonfinite == 0,
                  "MMA produced a non-finite design; refusing to checkpoint it.");

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
      const real_t post_update_volume =
         InnerProduct(comm, *volume_weights, rho_tv_full);
      const real_t post_update_vol_frac = post_update_volume / domain_volume;

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
              << "   g = [" << scientific << setprecision(3) << fival(0)
              << ", " << fival(1) << "]"
              << "   ||grad_active|| = " << active_raw_gradient_norm
              << "   dRho(L1) = " << setprecision(3) << iterationError << "\n";
         history << setw(5) << k + 1 << "  "
                 << scientific << setprecision(8) << J << "  "
                 << fixed << setprecision(6) << cur_vol_frac << "  "
                 << scientific << setprecision(6) << fival(0) << "  "
                 << fival(1) << "  "
                 << raw_gradient_norm << "  "
                 << active_raw_gradient_norm << "  "
                 << filtered_gradient_norm << "  "
                 << iterationError << "  "
                 << iteration_telemetry.forward_seconds << "  "
                 << iteration_telemetry.adjoint_seconds << "  "
                 << iteration_telemetry.trajectory_memory_mb << "  "
                 << iteration_telemetry.controller_replayed_blocks << "  "
                 << iteration_telemetry.locally_replayed_blocks << "  "
                 << iteration_telemetry.controller_replayed_intervals << "  "
                 << iteration_telemetry.locally_replayed_intervals << "\n";
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
         ckpt_meta.format_version = 2;
         ckpt_meta.design_iteration = k + 1;
         ckpt_meta.objective_valid_for_design = false;
         ckpt_meta.objective = 0.0;
         ckpt_meta.volume_fraction = post_update_vol_frac;
         ckpt_meta.refinement_level = cfg.ref_levels;
         ckpt_meta.fe_order = state_order;
         ckpt_meta.design_fe_order = design_order;

         if (!checkpoint.Save(ckpt_meta, rho_tv_full))
         {
            if (myid == 0)
            {
               cerr << "WARNING: Failed to save checkpoint for design rho^"
                    << k + 1 << "\n";
            }
         }
         save_checkpoint_history(ckpt_meta, rho_tv_full);
      }

      const bool is_last_iter =
         (k + 1 >= problem.GetMaxIterations()) ||
         (iterationError <= problem.GetChangeTolerance());
      if (paraview)
      {
         // Design snapshot: first, last, and every pv_save_interval-th iteration.
         if (k == 0 || is_last_iter || (k + 1) % pv_save_interval == 0)
         {
            // MMA has just updated rho.  Refresh rho_tilde before saving so
            // both fields in this snapshot describe the same post-update
            // design (the physics sweep above used the pre-update design).
            design_solver.FilterFSolve(rho_tv_full);
            paraview_dc.SetCycle(k + 1);
            paraview_dc.SetTime(k + 1);
            paraview_dc.Save();
         }

         // Forward wave visualization (first and last iteration only).
         // Stream only the sampled frames to ParaView.  Retaining every state
         // would require O(num_steps * state_size) memory (about 15.5 GiB per
         // rank for the p=3 spherical production run).
         if (k == 0 || is_last_iter)
         {
            if (myid == 0)
            {
               cout << "    Generating forward wave visualization...\n";
            }

            // Save wave propagation to ParaView (the data collection creates
            // its own directories under the prefix path).
            string wave_collection_name = "wave_iter" + to_string(k);
            string wave_full_dir = output_parent_dir + "/ParaView/" + wave_collection_name;

            ParaViewDataCollection wave_dc(wave_collection_name.c_str(), &pmesh);
            wave_dc.SetLevelsOfDetail(state_order);
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

            const auto save_wave = [&](int step, real_t time,
                                       const Vector &state)
            {
               const int half_size = state.Size() / 2;
               Vector u_vec(state.GetData(), half_size);
               u_gf.SetFromTrueDofs(u_vec);
               wave_dc.SetCycle(step);
               wave_dc.SetTime(time);
               wave_dc.Save();
               frames_saved++;
            };
            const real_t visualization_objective =
               design_solver.ForwardVisualizationSweepStream(
               rho_tv_full, wave_viz_freq, save_wave,
               /*refilter_design=*/false);
            if (is_last_iter)
            {
               final_design_objective = visualization_objective;
               final_design_objective_available = true;
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

   // Every numeric history row evaluates the design before its MMA update.
   // Evaluate the final post-update density once so its reported/checkpointed
   // objective actually belongs to the saved rho^k. A final ParaView wave
   // sweep already computed the identical continuous/discrete objective.
   if (!final_design_objective_available)
   {
      final_design_objective =
         design_solver.Objective(rho_tv_full, "final design");
      final_design_objective_available = true;
   }
   const real_t final_design_volume =
      InnerProduct(comm, *volume_weights, rho_tv_full);
   const real_t final_design_vol_frac = final_design_volume / domain_volume;
   MFEM_VERIFY(std::isfinite(final_design_objective) &&
               std::isfinite(final_design_vol_frac),
               "Final post-update design evaluation is non-finite.");

   if (auto_checkpoint)
   {
      ckpt_meta.format_version = 2;
      ckpt_meta.design_iteration = k;
      ckpt_meta.objective_valid_for_design = true;
      ckpt_meta.objective = final_design_objective;
      ckpt_meta.volume_fraction = final_design_vol_frac;
      ckpt_meta.refinement_level = cfg.ref_levels;
      ckpt_meta.fe_order = state_order;
      ckpt_meta.design_fe_order = design_order;
      if (!checkpoint.Save(ckpt_meta, rho_tv_full) && myid == 0)
      {
         cerr << "WARNING: Failed to commit the evaluated final checkpoint "
              << "for design rho^" << k << "\n";
      }
      save_checkpoint_history(ckpt_meta, rho_tv_full);
   }

   if (myid == 0)
   {
      history << "# Final design index: " << k << "\n"
              << "# Final design objective: " << scientific
              << setprecision(16) << final_design_objective << "\n"
              << "# Final design volume fraction: " << final_design_vol_frac
              << "\n"
              << "# Final design evaluation: forward-only\n";
      history.close();
      cout << "\n=== Optimization Complete ===\n";
      cout << "Output directory: " << output_parent_dir << "\n";
      cout << "Total iterations: " << k << "\n";
      cout << "Final convergence error: " << scientific << setprecision(3)
           << iterationError << " (tol = " << problem.GetChangeTolerance() << ")\n";
      cout << "Final design J(rho^" << k << ") = " << scientific
           << setprecision(8) << final_design_objective
           << ", volume fraction = " << fixed << setprecision(6)
           << final_design_vol_frac << "\n";
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
