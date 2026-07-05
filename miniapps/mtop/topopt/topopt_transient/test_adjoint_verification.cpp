// =============================================================================
// Adjoint verification for the transient elastodynamics operator
// =============================================================================
//
// These checks intentionally stop before design sensitivities and MMA:
//   1. <J(x) v, w> = <v, J(x)^T w>
//   2. <D Phi_h(x) v, w> = <v, D Phi_h(x)^T w> for one RK4 step
//   3. the same identity for an n-step RK4 map
//
// MFEM in this checkout does not expose RK4 AdjointStep, so the RK4 transpose
// used here is a local reverse-mode transcription of MFEM's RK4Solver::Step.
//
// =============================================================================

#include "mfem.hpp"
#include "ElastodynamicsSolver.hpp"
#include "../../pde_filter.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>

using namespace mfem;
using namespace std;

namespace
{

double GlobalDot(MPI_Comm comm, const Vector &a, const Vector &b)
{
   return InnerProduct(comm, a, b);
}

double RelativeError(double lhs, double rhs)
{
   double scale = 1.0;
   scale = max(scale, fabs(lhs));
   scale = max(scale, fabs(rhs));
   return fabs(lhs - rhs) / scale;
}

void RandomState(Vector &x, int seed)
{
   x.Randomize(seed);
   x *= 0.1;
}

void Normalize(MPI_Comm comm, Vector &x)
{
   const real_t norm = std::sqrt(GlobalDot(comm, x, x));
   MFEM_VERIFY(norm > 0.0, "Cannot normalize a zero vector.");
   x /= norm;
}

// DirectionalBoundaryLoadCoefficient: shared from BoundaryLoadSpec.hpp
// (pulled in via ElastodynamicsSolver.hpp).

// EvalRHS / EvalJacobianTranspose: promoted to ElastodynamicsSolver.hpp

void RK4OneStep(ElastodynamicsOperator &oper,
                const Vector &x0, real_t t0, real_t h, Vector &x1)
{
   x1 = x0;
   real_t t = t0;
   real_t dt = h;
   RK4Solver solver;
   solver.Init(oper);
   solver.Step(x1, t, dt);
}

// RK4Stages / RK4AdjointOneStep: promoted to ElastodynamicsSolver.hpp

void RolloutRK4(ElastodynamicsOperator &oper,
                const Vector &x_init, int nsteps,
                real_t t_init, real_t h,
                Vector &x_final,
                vector<Vector> *states,
                vector<real_t> *times)
{
   const int n = x_init.Size();
   x_final = x_init;
   real_t t = t_init;

   RK4Solver solver;
   solver.Init(oper);

   if (states)
   {
      states->resize(nsteps + 1);
      for (int i = 0; i <= nsteps; i++) { (*states)[i].SetSize(n); }
      (*states)[0] = x_final;
   }
   if (times)
   {
      times->assign(nsteps + 1, 0.0);
      (*times)[0] = t;
   }

   for (int i = 0; i < nsteps; i++)
   {
      real_t dt = h;
      solver.Step(x_final, t, dt);

      if (states) { (*states)[i + 1] = x_final; }
      if (times)  { (*times)[i + 1] = t; }
   }
}

double CheckJacobianTranspose(ElastodynamicsOperator &oper,
                              MPI_Comm comm, int n,
                              int ntrials, real_t eps,
                              real_t tolerance)
{
   double worst = 0.0;
   const real_t t0 = 0.0137;

   Vector x(n), v(n), w(n), xp(n), xm(n), fp(n), fm(n), jv(n), jtw(n);

   for (int trial = 0; trial < ntrials; trial++)
   {
      RandomState(x, 100 + 3*trial);
      RandomState(v, 101 + 3*trial);
      RandomState(w, 102 + 3*trial);

      xp = x;
      xm = x;
      xp.Add(eps, v);
      xm.Add(-eps, v);

      EvalRHS(oper, xp, t0, fp);
      EvalRHS(oper, xm, t0, fm);

      jv = fp;
      jv -= fm;
      jv *= 0.5 / eps;

      EvalJacobianTranspose(oper, x, t0, w, jtw);

      const double lhs = GlobalDot(comm, jv, w);
      const double rhs = GlobalDot(comm, v, jtw);
      const double rel = RelativeError(lhs, rhs);
      worst = max(worst, rel);

      if (Mpi::Root())
      {
         mfem::out << "Jacobian transpose trial " << trial
                   << ": lhs=" << setprecision(16) << lhs
                   << ", rhs=" << rhs
                   << ", rel_err=" << rel << '\n';
      }
   }

   MFEM_VERIFY(worst < tolerance, "Jacobian transpose verification failed.");
   return worst;
}

double CheckRK4OneStepTranspose(ElastodynamicsOperator &oper,
                                MPI_Comm comm, int n,
                                int ntrials, real_t h, real_t eps,
                                real_t tolerance)
{
   double worst = 0.0;
   const real_t t0 = 0.0137;

   Vector x0(n), v(n), w(n), xp(n), xm(n), x_plus(n), x_minus(n);
   Vector jv(n), lambda_prev(n);

   for (int trial = 0; trial < ntrials; trial++)
   {
      RandomState(x0, 200 + 3*trial);
      RandomState(v, 201 + 3*trial);
      RandomState(w, 202 + 3*trial);

      xp = x0;
      xm = x0;
      xp.Add(eps, v);
      xm.Add(-eps, v);

      RK4OneStep(oper, xp, t0, h, x_plus);
      RK4OneStep(oper, xm, t0, h, x_minus);

      jv = x_plus;
      jv -= x_minus;
      jv *= 0.5 / eps;

      RK4AdjointOneStep(oper, x0, t0, h, w, lambda_prev);

      const double lhs = GlobalDot(comm, jv, w);
      const double rhs = GlobalDot(comm, v, lambda_prev);
      const double rel = RelativeError(lhs, rhs);
      worst = max(worst, rel);

      if (Mpi::Root())
      {
         mfem::out << "RK4 one-step transpose trial " << trial
                   << ": lhs=" << setprecision(16) << lhs
                   << ", rhs=" << rhs
                   << ", rel_err=" << rel << '\n';
      }
   }

   MFEM_VERIFY(worst < tolerance, "RK4 one-step transpose verification failed.");
   return worst;
}

double CheckRK4NStepTranspose(ElastodynamicsOperator &oper,
                              MPI_Comm comm, int n,
                              int nsteps, int ntrials,
                              real_t h, real_t eps,
                              real_t tolerance)
{
   double worst = 0.0;
   const real_t t0 = 0.0;

   Vector x0(n), v(n), w(n), xp(n), xm(n), x_plus(n), x_minus(n);
   Vector jv(n), lambda(n), lambda_prev(n);

   for (int trial = 0; trial < ntrials; trial++)
   {
      RandomState(x0, 300 + 3*trial);
      RandomState(v, 301 + 3*trial);
      RandomState(w, 302 + 3*trial);

      vector<Vector> states;
      vector<real_t> times;
      Vector x_base(n);
      RolloutRK4(oper, x0, nsteps, t0, h, x_base, &states, &times);

      xp = x0;
      xm = x0;
      xp.Add(eps, v);
      xm.Add(-eps, v);

      RolloutRK4(oper, xp, nsteps, t0, h, x_plus, nullptr, nullptr);
      RolloutRK4(oper, xm, nsteps, t0, h, x_minus, nullptr, nullptr);

      jv = x_plus;
      jv -= x_minus;
      jv *= 0.5 / eps;

      lambda = w;
      for (int i = nsteps - 1; i >= 0; i--)
      {
         const real_t hi = times[i + 1] - times[i];
         RK4AdjointOneStep(oper, states[i], times[i], hi, lambda, lambda_prev);
         lambda = lambda_prev;
      }

      const double lhs = GlobalDot(comm, jv, w);
      const double rhs = GlobalDot(comm, v, lambda);
      const double rel = RelativeError(lhs, rhs);
      worst = max(worst, rel);

      if (Mpi::Root())
      {
         mfem::out << "RK4 " << nsteps << "-step transpose trial " << trial
                   << ": lhs=" << setprecision(16) << lhs
                   << ", rhs=" << rhs
                   << ", rel_err=" << rel << '\n';
      }
   }

   MFEM_VERIFY(worst < tolerance, "RK4 n-step transpose verification failed.");
   return worst;
}

// AddObjectiveContribution / ObjectiveGradientAtState / RolloutObjective:
// promoted to ElastodynamicsSolver.hpp

real_t ObjectiveAdjointGradient(ElastodynamicsOperator &oper,
                                ParFiniteElementSpace &state_fes,
                                const Array<int> &offsets,
                                TimeIntegratedObjective &objective,
                                const Vector &x0,
                                int nsteps, real_t t_init, real_t h,
                                Vector &gradient)
{
   vector<Vector> states;
   vector<real_t> times;
   const real_t J = RolloutObjective(oper, state_fes, offsets, objective,
                                     x0, nsteps, t_init, h,
                                     &states, &times);

   const int n = x0.Size();
   const int total_steps = nsteps + 1;
   Vector q(n), lambda(n), lambda_prev(n);

   ObjectiveGradientAtState(state_fes, offsets, objective, states[nsteps],
                            h, nsteps, total_steps, lambda);

   for (int i = nsteps - 1; i >= 0; i--)
   {
      const real_t hi = times[i + 1] - times[i];
      RK4AdjointOneStep(oper, states[i], times[i], hi, lambda, lambda_prev);

      ObjectiveGradientAtState(state_fes, offsets, objective, states[i],
                               h, i, total_steps, q);
      lambda = lambda_prev;
      lambda += q;
   }

   gradient = lambda;
   return J;
}

double CheckObjectiveTaylor(ElastodynamicsOperator &oper,
                            ParFiniteElementSpace &state_fes,
                            const Array<int> &offsets,
                            TimeIntegratedObjective &objective,
                            MPI_Comm comm, int n,
                            int nsteps, int ntrials,
                            real_t h, int nscales,
                            real_t initial_scale,
                            real_t tolerance)
{
   double worst_best_fd_rel = 0.0;
   const real_t t0 = 0.0;

   Vector x0(n), direction(n), gradient(n), xp(n), xm(n);

   for (int trial = 0; trial < ntrials; trial++)
   {
      RandomState(x0, 400 + 2*trial);
      RandomState(direction, 401 + 2*trial);
      Normalize(comm, direction);

      const real_t J0 = ObjectiveAdjointGradient(oper, state_fes, offsets,
                                                 objective, x0, nsteps,
                                                 t0, h, gradient);
      const real_t projected_grad = GlobalDot(comm, gradient, direction);

      if (Mpi::Root())
      {
         mfem::out << "\nObjective Taylor trial " << trial
                   << ": J0=" << setprecision(16) << J0
                   << ", <grad,p>=" << projected_grad << '\n';
      }

      real_t scale = initial_scale;
      double previous_remainder = -1.0;
      double trial_best_fd_rel = numeric_limits<double>::infinity();
      for (int s = 0; s < nscales; s++)
      {
         xp = x0;
         xm = x0;
         xp.Add(scale, direction);
         xm.Add(-scale, direction);

         const real_t Jp = RolloutObjective(oper, state_fes, offsets, objective,
                                            xp, nsteps, t0, h, nullptr, nullptr);
         const real_t Jm = RolloutObjective(oper, state_fes, offsets, objective,
                                            xm, nsteps, t0, h, nullptr, nullptr);

         const real_t fd = (Jp - Jm) / (2.0 * scale);
         const double derivative_scale =
            max(max(fabs(static_cast<double>(fd)),
                    fabs(static_cast<double>(projected_grad))), 1e-30);
         const double fd_rel = fabs(static_cast<double>(fd - projected_grad))
                               / derivative_scale;
         trial_best_fd_rel = min(trial_best_fd_rel, fd_rel);

         const real_t first_order_remainder =
            fabs(Jp - J0 - scale * projected_grad);
         const double remainder_ratio =
            (previous_remainder > 0.0) ?
            previous_remainder / first_order_remainder : 0.0;

         if (Mpi::Root())
         {
            mfem::out << "  scale=" << scientific << setprecision(3) << scale
                      << "  FD=" << setprecision(12) << fd
                      << "  rel_err=" << fd_rel
                      << "  first_order_rem=" << first_order_remainder;
            if (previous_remainder > 0.0)
            {
               mfem::out << "  rem_ratio=" << remainder_ratio;
            }
            mfem::out << '\n';
         }

         previous_remainder = first_order_remainder;
         scale *= 0.1;
      }

      worst_best_fd_rel = max(worst_best_fd_rel, trial_best_fd_rel);
   }

   MFEM_VERIFY(worst_best_fd_rel < tolerance,
               "Objective adjoint finite-difference Taylor check failed.");
   return worst_best_fd_rel;
}

// MaterialParams / SimpDerivative / StageMassDesignLFIntegrator /
// StageStiffnessDesignLFIntegrator: promoted to ElastodynamicsSolver.hpp

double CheckDesignTaylor(ParFiniteElementSpace &state_fes,
                         ParFiniteElementSpace &filter_fes,
                         ParFiniteElementSpace &control_fes,
                         ParGridFunction &rho,
                         ParGridFunction &rho_tilde,
                         toopt::PDEFilter &filter,
                         SpatialDampingCoefficient &gamma_coef,
                         Array<int> &exterior_bdr_attr,
                         Array<int> &empty_bdr_attr,
                         TimeIntegratedObjective &objective,
                         const MaterialParams &mat,
                         const BoundaryLoadSpec &load_spec,
                         VectorCoefficient &load_coef,
                         real_t impedance,
                         int nsteps,
                         real_t h,
                         int ntrials,
                         int nscales,
                         real_t initial_scale,
                         real_t state_scale,
                         real_t tolerance,
                         MassSolverType mass_type)
{
   MPI_Comm comm = state_fes.GetComm();
   double worst_best_fd_rel = 0.0;

   const char *mass_label =
      (mass_type == MassSolverType::LUMPED) ? "LUMPED" : "CONSISTENT";
   if (Mpi::Root())
   {
      mfem::out << "\n--- Design Taylor check (" << mass_label
                << " mass) ---\n";
   }

   Vector rho0;
   rho.GetTrueDofs(rho0);

   const int state_size = 2 * state_fes.GetTrueVSize();
   const int design_size = control_fes.GetTrueVSize();

   Vector x0(state_size), direction(design_size);
   Vector grad(design_size), rho_plus(design_size), rho_minus(design_size);

   for (int trial = 0; trial < ntrials; trial++)
   {
      RandomState(x0, 500 + 2*trial);
      x0 *= state_scale;
      RandomState(direction, 501 + 2*trial);
      Normalize(comm, direction);

      // The design sensitivity integrators differentiate whichever mass matrix
      // (consistent or row-lumped) drives the forward solve, so J and dJ/drho are
      // self-consistent for both mass_type choices (see StageMassDesignLFIntegrator).
      const real_t J0 = DesignObjectiveAdjointGradient(
         rho0, x0, state_fes, filter_fes, control_fes, mass_type,
         rho, rho_tilde, filter,
         gamma_coef, exterior_bdr_attr, empty_bdr_attr, objective,
         mat, load_spec, load_coef, impedance, nsteps, h, grad);

      const real_t projected_grad = GlobalDot(comm, grad, direction);

      if (Mpi::Root())
      {
         mfem::out << "\nDesign Taylor trial " << trial
                   << ": J0=" << setprecision(16) << J0
                   << ", <dJ/drho,p>=" << projected_grad << '\n';
      }

      real_t scale = initial_scale;
      double previous_remainder = -1.0;
      double trial_best_fd_rel = numeric_limits<double>::infinity();
      bool trial_has_quadratic_drop = false;
      for (int s = 0; s < nscales; s++)
      {
         rho_plus = rho0;
         rho_minus = rho0;
         rho_plus.Add(scale, direction);
         rho_minus.Add(-scale, direction);

         const real_t Jp = EvaluateDesignObjective(
            rho_plus, x0, state_fes, control_fes, rho, rho_tilde, filter,
            gamma_coef, exterior_bdr_attr, empty_bdr_attr,
            objective, mat, load_spec, load_coef,
            impedance, nsteps, h, mass_type);

         const real_t Jm = EvaluateDesignObjective(
            rho_minus, x0, state_fes, control_fes, rho, rho_tilde, filter,
            gamma_coef, exterior_bdr_attr, empty_bdr_attr,
            objective, mat, load_spec, load_coef,
            impedance, nsteps, h, mass_type);

         const real_t fd = (Jp - Jm) / (2.0 * scale);
         const double derivative_scale =
            max(max(fabs(static_cast<double>(fd)),
                    fabs(static_cast<double>(projected_grad))), 1e-30);
         const double fd_rel = fabs(static_cast<double>(fd - projected_grad))
                               / derivative_scale;
         trial_best_fd_rel = min(trial_best_fd_rel, fd_rel);

         const real_t first_order_remainder =
            fabs(Jp - J0 - scale * projected_grad);
         const double remainder_ratio =
            (previous_remainder > 0.0) ?
            previous_remainder / first_order_remainder : 0.0;

         if (Mpi::Root())
         {
            mfem::out << "  scale=" << scientific << setprecision(3) << scale
                      << "  FD=" << setprecision(12) << fd
                      << "  rel_err=" << fd_rel
                      << "  first_order_rem=" << first_order_remainder;
            if (previous_remainder > 0.0)
            {
               mfem::out << "  rem_ratio=" << remainder_ratio;
            }
            mfem::out << '\n';
         }

         if (previous_remainder > 0.0 && remainder_ratio > 50.0)
         {
            trial_has_quadratic_drop = true;
         }

         previous_remainder = first_order_remainder;
         scale *= 0.1;
      }

      worst_best_fd_rel = max(worst_best_fd_rel, trial_best_fd_rel);
      MFEM_VERIFY(trial_best_fd_rel < tolerance,
                  "Raw design Taylor check did not find an accurate scale.");
      MFEM_VERIFY(trial_has_quadratic_drop,
                  "Raw design Taylor check did not show quadratic remainder decay.");
   }

   return worst_best_fd_rel;
}

} // namespace

int main(int argc, char *argv[])
{
   Mpi::Init();
   Hypre::Init();

   MPI_Comm comm = MPI_COMM_WORLD;
   const int myid = Mpi::WorldRank();

   int ref_levels = 0;
   int order = 1;
   int ntrials = 3;
   int nsteps = 8;
   int taylor_scales = 6;
   real_t dt = 5e-5;
   real_t eps = 1.0;
   real_t tolerance = 1e-7;
   real_t taylor_initial_scale = 1e-1;
   real_t taylor_tolerance = 1e-5;
   real_t design_initial_scale = 1e-1;
   real_t design_state_scale = 100.0;
   real_t design_tolerance = 1e-4;
   real_t vol_frac = 0.5;
   real_t filter_radius = 0.05;
   real_t protected_radius = 0.2;
   const char *mesh_file = "lamb-problem-damping-mesh-triangs.msh";

   OptionsParser args(argc, argv);
   args.AddOption(&ref_levels, "-r", "--refine", "Refinement level");
   args.AddOption(&order, "-o", "--order", "H1 finite element order");
   args.AddOption(&ntrials, "-nt", "--num-trials", "Random trials per check");
   args.AddOption(&nsteps, "-ns", "--num-steps", "RK4 steps for n-step check");
   args.AddOption(&taylor_scales, "-ts", "--taylor-scales",
                  "Number of Taylor finite-difference scales");
   args.AddOption(&taylor_initial_scale, "-t0", "--taylor-initial-scale",
                  "Initial Taylor finite-difference scale");
   args.AddOption(&dt, "-dt", "--time-step", "RK4 time step");
   args.AddOption(&eps, "-eps", "--epsilon", "Centered finite-difference step");
   args.AddOption(&tolerance, "-tol", "--tolerance", "Relative error tolerance");
   args.AddOption(&taylor_tolerance, "-ttol", "--taylor-tolerance",
                  "Objective Taylor relative derivative tolerance");
   args.AddOption(&design_initial_scale, "-d0", "--design-initial-scale",
                  "Initial raw-design Taylor finite-difference scale");
   args.AddOption(&design_state_scale, "-ds", "--design-state-scale",
                  "Initial-state amplification used only by design Taylor test");
   args.AddOption(&design_tolerance, "-dtol", "--design-tolerance",
                  "Raw-design Taylor relative derivative tolerance");
   args.AddOption(&vol_frac, "-vf", "--vol-frac", "Uniform control density");
   args.AddOption(&filter_radius, "-fr", "--filter-radius",
                  "Helmholtz filter radius");
   args.AddOption(&protected_radius, "-pr", "--protected-radius",
                  "Circular protected-zone radius for objective");
   args.AddOption(&mesh_file, "-mesh", "--mesh-file", "Mesh file");
   args.Parse();

   if (!args.Good())
   {
      if (myid == 0) { args.PrintUsage(cout); }
      return 1;
   }
   if (myid == 0) { args.PrintOptions(cout); }

   Device device("cpu");

   ifstream imesh(mesh_file);
   if (!imesh)
   {
      if (myid == 0)
      {
         cerr << "Error: cannot open mesh file '" << mesh_file << "'.\n";
      }
      return 1;
   }

   Mesh mesh(imesh, 1, 1);
   imesh.close();
   const int dim = mesh.Dimension();

   for (int l = 0; l < ref_levels; l++) { mesh.UniformRefinement(); }

   ParMesh pmesh(comm, mesh);
   mesh.Clear();

   H1_FECollection state_fec(order, dim);
   H1_FECollection filter_fec(order, dim);
   L2_FECollection control_fec(max(0, order - 1), dim, BasisType::GaussLobatto);

   ParFiniteElementSpace state_fes(&pmesh, &state_fec, dim);
   ParFiniteElementSpace filter_fes(&pmesh, &filter_fec);
   ParFiniteElementSpace control_fes(&pmesh, &control_fec);

   ParGridFunction rho(&control_fes);
   ParGridFunction rho_tilde(&filter_fes);
   rho = vol_frac;

   toopt::PDEFilterOptions filter_opts;
   filter_opts.filter_radius = filter_radius;
   toopt::PDEFilter filter(filter_fes, control_fes, filter_opts);
   filter.Assemble();
   filter.Mult(rho, rho_tilde);

   MaterialParams mat;

   ConstantCoefficient rho_0_coef(mat.rho0);
   ConstantCoefficient lambda_0_coef(mat.lambda0);
   ConstantCoefficient mu_0_coef(mat.mu0);

   SIMPCoefficient simp_mass(&rho_tilde, mat.r_min, mat.r_max, mat.simp_p);
   SIMPCoefficient simp_stiff(&rho_tilde, mat.r_min, mat.r_max, mat.simp_p);

   ProductCoefficient mass_coef(simp_mass, rho_0_coef);
   ProductCoefficient lambda_coef(simp_stiff, lambda_0_coef);
   ProductCoefficient mu_coef(simp_stiff, mu_0_coef);

   const real_t x_max = 1.5;
   const real_t y_max = 0.75;
   const real_t c_p = sqrt((mat.lambda0 + 2.0*mat.mu0) / mat.rho0);
   const real_t damping_thickness = 0.25;
   DampingProfile phi_profile(damping_thickness, x_max, y_max);
   const real_t gamma_max = (2.0 * c_p / 0.2136) * log(1.0 / 1e-4);
   SpatialDampingCoefficient gamma_coef(&phi_profile, gamma_max,
                                        mat.rho0, 2.0, 2);

   BoundaryLoadSpec load_spec;
   DirectionalBoundaryLoadCoefficient load_coef(load_spec.direction);
   const real_t impedance = mat.rho0 * c_p;

   Array<int> exterior_bdr_attr(pmesh.bdr_attributes.Max());
   exterior_bdr_attr = 0;
   if (pmesh.bdr_attributes.Max() >= 10) { exterior_bdr_attr[9] = 1; }
   if (pmesh.bdr_attributes.Max() >= 11) { exterior_bdr_attr[10] = 1; }
   if (pmesh.bdr_attributes.Max() >= 12) { exterior_bdr_attr[11] = 1; }

   Array<int> empty_bdr_attr(pmesh.bdr_attributes.Max());
   empty_bdr_attr = 0;

   ElastodynamicsOperator oper(
      state_fes, mass_coef, lambda_coef, mu_coef,
      load_spec.amplitude, load_spec.duration, load_spec.time_profile,
      load_spec.phase, load_spec.frequency, load_spec.bdr_attributes, load_coef,
      &gamma_coef, impedance, exterior_bdr_attr, empty_bdr_attr);

   SubdomainIndicator subdomain_indicator(x_max/2.0, y_max/2.0,
                                          protected_radius);
   DisplacementL2Objective objective(&state_fes, subdomain_indicator, comm);

   const int state_size = oper.Height();

   if (myid == 0)
   {
      mfem::out << "\n=== Adjoint Verification ===\n"
                << "State size: " << state_size << '\n'
                << "Trials: " << ntrials << '\n'
                << "RK4 n-step count: " << nsteps << '\n'
                << "dt: " << dt << '\n'
                << "eps: " << eps << '\n'
                << "transpose tolerance: " << tolerance << '\n'
                << "state Taylor tolerance: " << taylor_tolerance << '\n'
                << "design Taylor tolerance: " << design_tolerance << "\n\n";
   }

   const double jac_err =
      CheckJacobianTranspose(oper, comm, state_size, ntrials, eps, tolerance);
   const double one_step_err =
      CheckRK4OneStepTranspose(oper, comm, state_size, ntrials, dt, eps,
                               tolerance);
   const double n_step_err =
      CheckRK4NStepTranspose(oper, comm, state_size, nsteps, ntrials, dt, eps,
                             tolerance);
   const double taylor_err =
      CheckObjectiveTaylor(oper, state_fes, oper.GetBlockOffsets(), objective,
                           comm, state_size, nsteps, ntrials, dt,
                           taylor_scales, taylor_initial_scale,
                           taylor_tolerance);
   // Verify the design gradient for BOTH mass discretizations: the sensitivity
   // must be self-consistent with whichever mass solver drives the forward solve.
   const double design_taylor_err_cg =
      CheckDesignTaylor(state_fes, filter_fes, control_fes, rho, rho_tilde,
                        filter, gamma_coef, exterior_bdr_attr, empty_bdr_attr,
                        objective, mat, load_spec, load_coef,
                        impedance, nsteps, dt, ntrials,
                        taylor_scales, design_initial_scale,
                        design_state_scale, design_tolerance,
                        MassSolverType::ITERATIVE);
   const double design_taylor_err_lumped =
      CheckDesignTaylor(state_fes, filter_fes, control_fes, rho, rho_tilde,
                        filter, gamma_coef, exterior_bdr_attr, empty_bdr_attr,
                        objective, mat, load_spec, load_coef,
                        impedance, nsteps, dt, ntrials,
                        taylor_scales, design_initial_scale,
                        design_state_scale, design_tolerance,
                        MassSolverType::LUMPED);

   if (myid == 0)
   {
      mfem::out << "\nAll adjoint and objective Taylor checks passed.\n"
                << "Worst Jacobian transpose error: "
                << scientific << setprecision(6) << jac_err << '\n'
                << "Worst RK4 one-step transpose error: "
                << one_step_err << '\n'
                << "Worst RK4 n-step transpose error: "
                << n_step_err << '\n'
                << "Worst objective Taylor FD error: "
                << taylor_err << '\n'
                << "Worst raw-design Taylor FD error (consistent mass): "
                << design_taylor_err_cg << '\n'
                << "Worst raw-design Taylor FD error (lumped mass): "
                << design_taylor_err_lumped << '\n';
   }

   return 0;
}
