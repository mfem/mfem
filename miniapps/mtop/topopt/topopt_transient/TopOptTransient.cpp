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
#include "ObjectiveFunctional.hpp"
#include "../../pde_filter.hpp"
#include "../../mma/MMA_MFEM.hpp"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
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

   int ref_levels = 0;
   int order = 1;
   real_t t_final = 0.006;
   real_t dt = 5e-5;
   real_t vol_frac = 0.5;
   real_t filter_radius = 0.05;
   int max_it = 20;
   real_t move = 0.2;
   real_t change_tol = 1e-3;
   bool paraview = false;
   bool use_iterative_mass = false;
   const char *mesh_file = "lamb-problem-damping-mesh-triangs.msh";
   const char *design_init = "uniform";

   OptionsParser args(argc, argv);
   args.AddOption(&ref_levels, "-r", "--refine", "Refinement level");
   args.AddOption(&order, "-o", "--order", "H1 finite element order");
   args.AddOption(&t_final, "-tf", "--t-final", "Final time");
   args.AddOption(&dt, "-dt", "--time-step", "Time step");
   args.AddOption(&vol_frac, "-vf", "--vol-frac", "Target volume fraction");
   args.AddOption(&filter_radius, "-fr", "--filter-radius",
                  "Helmholtz filter radius");
   args.AddOption(&max_it, "-mi", "--max-it", "Max MMA iterations");
   args.AddOption(&move, "-mv", "--move", "MMA move limit");
   args.AddOption(&change_tol, "-tol", "--tol",
                  "Stop early when the L1 design change drops below this");
   args.AddOption(&design_init, "-init", "--design-init",
                  "Initial design: uniform, solid, void, gaussian");
   args.AddOption(&mesh_file, "-mesh", "--mesh-file", "Mesh file");
   args.AddOption(&paraview, "-pv", "--paraview", "-no-pv",
                  "--no-paraview", "Write ParaView output");
   args.AddOption(&use_iterative_mass, "-iterative-mass", "--iterative-mass",
                  "-lumped-mass", "--lumped-mass",
                  "Use iterative (CG+AMG) mass solver instead of lumped (default: lumped)");
   args.Parse();

   if (!args.Good())
   {
      if (myid == 0) { args.PrintUsage(cout); }
      return 1;
   }

   if (order < 1)
   {
      if (myid == 0) { cerr << "Error: -o/--order must be at least 1.\n"; }
      return 1;
   }
   if (dt <= 0.0 || t_final <= 0.0)
   {
      if (myid == 0)
      {
         cerr << "Error: -dt and -tf must both be positive.\n";
      }
      return 1;
   }
   if (vol_frac <= 0.0 || vol_frac > 1.0)
   {
      if (myid == 0)
      {
         cerr << "Error: -vf/--vol-frac must be in (0, 1].\n";
      }
      return 1;
   }
   if (max_it < 1)
   {
      if (myid == 0) { cerr << "Error: -mi/--max-it must be >= 1.\n"; }
      return 1;
   }

   if (myid == 0) { args.PrintOptions(cout); }

   const real_t x_max = 1.5;
   const real_t y_max = 0.75;

   ifstream imesh(mesh_file);
   if (!imesh)
   {
      if (myid == 0)
      {
         cerr << "Error: Cannot open mesh file '" << mesh_file << "'.\n";
      }
      return 1;
   }

   Mesh mesh(imesh, 1, 1);
   imesh.close();
   const int dim = mesh.Dimension();

   for (int l = 0; l < ref_levels; l++)
   {
      mesh.UniformRefinement();
   }

   ParMesh pmesh(comm, mesh);
   mesh.Clear();

   H1_FECollection state_fec(order, dim);
   H1_FECollection filter_fec(order, dim);
   const int control_order = max(0, order - 1);
   L2_FECollection control_fec(control_order, dim, BasisType::GaussLobatto);

   ParFiniteElementSpace state_fes(&pmesh, &state_fec, dim);
   ParFiniteElementSpace filter_fes(&pmesh, &filter_fec);
   ParFiniteElementSpace control_fes(&pmesh, &control_fec);

   const HYPRE_BigInt state_dofs = state_fes.GlobalTrueVSize();
   const HYPRE_BigInt filter_dofs = filter_fes.GlobalTrueVSize();
   const HYPRE_BigInt control_dofs = control_fes.GlobalTrueVSize();

   ParGridFunction rho(&control_fes);
   ParGridFunction rho_tilde(&filter_fes);

   if (!InitializeDesign(rho, design_init, vol_frac, x_max, y_max))
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
   filter_opts.filter_radius = filter_radius;
   toopt::PDEFilter filter(filter_fes, control_fes, filter_opts);
   filter.Assemble();
   filter.Mult(rho, rho_tilde);

   real_t domain_volume = 0.0;
   unique_ptr<HypreParVector> volume_weights =
      AssembleVolumeWeights(control_fes, domain_volume);
   const real_t target_volume = vol_frac * domain_volume;

   // Material and problem constants (match test_adjoint_verification).
   MaterialParams mat;

   const real_t c_p = sqrt((mat.lambda0 + 2.0*mat.mu0) / mat.rho0);
   const real_t damping_thickness = 0.25;
   DampingProfile phi_profile(damping_thickness, x_max, y_max);
   const real_t gamma_max = (2.0 * c_p / 0.2136) * log(1.0 / 1e-4);
   SpatialDampingCoefficient gamma_coef(&phi_profile, gamma_max,
                                        mat.rho0, 2.0, 2);

   const real_t pulse_duration = 0.005;
   const real_t pulse_amplitude = 30.0;
   const real_t impedance = mat.rho0 * c_p;

   Array<int> exterior_bdr_attr(pmesh.bdr_attributes.Max());
   exterior_bdr_attr = 0;
   if (pmesh.bdr_attributes.Max() >= 10) { exterior_bdr_attr[9] = 1; }
   if (pmesh.bdr_attributes.Max() >= 11) { exterior_bdr_attr[10] = 1; }
   if (pmesh.bdr_attributes.Max() >= 12) { exterior_bdr_attr[11] = 1; }

   Array<int> empty_bdr_attr(pmesh.bdr_attributes.Max());
   empty_bdr_attr = 0;

   const real_t protected_radius = 0.2;
   SubdomainIndicator subdomain_indicator(x_max/2.0, y_max/2.0,
                                          protected_radius);

   const int num_steps = max(1, static_cast<int>(ceil(t_final / dt)));
   const real_t dt_eff = t_final / num_steps;

   // Rest initial state z0 = [u0, v0] = 0; the pulse load drives the dynamics.
   Vector x0(2 * state_fes.GetTrueVSize());
   x0 = 0.0;

   if (myid == 0)
   {
      cout << "\n=== Transient TopOpt (MMA) ===\n";
      cout << "Mesh: " << mesh_file << "\n";
      cout << "Refinement levels: " << ref_levels << "\n";
      cout << "State DOFs:   " << state_dofs << "\n";
      cout << "Filter DOFs:  " << filter_dofs << " (H1 rho_tilde)\n";
      cout << "Control DOFs: " << control_dofs << " (L2 rho)\n";
      cout << "Target volume fraction: " << vol_frac << "\n";
      cout << "Filter radius: " << filter_radius << "\n";
      cout << "Time interval: [0, " << t_final << "],  steps: " << num_steps
           << ",  dt_eff: " << dt_eff << "\n";
      cout << "Max MMA iterations: " << max_it << ",  move limit: " << move
           << ",  stop tol (L1 dRho): " << change_tol << "\n";
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
      paraview_dc.SetLevelsOfDetail(order);
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

   GridFunctionCoefficient rho_cf(&rho);
   int k = 0;
   real_t iterationError = 1.0;
   for (; k < max_it && iterationError > change_tol; k++)
   {
      // Objective + design gradient (forward sweep + discrete adjoint).
      // On entry this sets rho, rho_tilde from rho_tv.
      const real_t J = DesignObjectiveAdjointGradient(
         rho_tv, x0, state_fes, filter_fes, control_fes, mass_solver,
         rho, rho_tilde,
         filter, gamma_coef, exterior_bdr_attr, empty_bdr_attr,
         subdomain_indicator, mat, pulse_amplitude, pulse_duration,
         impedance, num_steps, dt_eff, dJ_drho, k);

      // Volume constraint and current fraction.
      const real_t cur_volume = InnerProduct(comm, *volume_weights, rho_tv);
      const real_t cur_vol_frac = cur_volume / domain_volume;
      fival(0) = cur_volume / target_volume - 1.0;

      // Box constraints with move limits.
      rho_old = rho_tv;
      for (int i = 0; i < n; i++)
      {
         rho_min[i] = max(real_t(0.0), rho_tv[i] - move);
         rho_max[i] = min(real_t(1.0), rho_tv[i] + move);
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
           << iterationError << ", tol = " << change_tol << ").\n";
      if (paraview)
      {
         cout << "ParaView output: ParaView/TopOptTransient.pvd\n";
      }
      cout << "History: optimization_history.txt\n";
   }

   return 0;
}
