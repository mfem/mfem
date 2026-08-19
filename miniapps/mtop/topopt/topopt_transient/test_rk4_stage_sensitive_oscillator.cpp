// A manufactured, stage-sensitive RK4 DO/OD comparison.
//
// This is deliberately not a topology-optimization production case.  It
// isolates the finite-step distinction in a single design-dependent oscillator
// with a state-dependent running functional.  The common forward equation is
//
//    u' = v,  v' = -omega_0^2 (0.1 + 0.9 rho^3) u,
//    J_h = h sum_{n,i} b_i (u_i^n)^2 / 2.
//
// DO differentiates the stored RK4 stages.  OD_modified implements the
// transformed RK4 adjoint independently.  OD_naive samples the same accepted
// forward endpoints through cubic Hermite reconstruction while integrating the
// continuous adjoint backwards with classical RK4.  Sweeping omega*h makes the
// stage-sampling effect visible without changing h between methods.

#include "mfem.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

using mfem::real_t;

namespace
{

struct State
{
   real_t u = 0.0;
   real_t v = 0.0;

   State &operator+=(const State &other)
   {
      u += other.u;
      v += other.v;
      return *this;
   }
};

State operator+(State left, const State &right) { return left += right; }

State operator-(const State &left, const State &right)
{
   return {left.u - right.u, left.v - right.v};
}

State operator*(real_t scale, const State &value)
{
   return {scale * value.u, scale * value.v};
}

real_t Dot(const State &left, const State &right)
{
   return left.u * right.u + left.v * right.v;
}

struct StepData
{
   State x_left;
   State x_right;
   std::array<State, 4> y;
   std::array<State, 4> k;
};

struct ForwardResult
{
   real_t objective = 0.0;
   std::vector<StepData> steps;
};

real_t Scale(real_t rho) { return 0.1 + 0.9 * rho * rho * rho; }
real_t ScaleDerivative(real_t rho) { return 2.7 * rho * rho; }

State Rhs(const State &x, real_t omega0_squared, real_t rho)
{
   return {x.v, -omega0_squared * Scale(rho) * x.u};
}

State JacobianTransposeTimes(const State &p, real_t omega0_squared,
                             real_t rho)
{
   // [[0, 1], [-omega^2 scale, 0]]^T p
   return {-omega0_squared * Scale(rho) * p.v, p.u};
}

State RhoDerivative(const State &x, real_t omega0_squared, real_t rho)
{
   return {0.0, -omega0_squared * ScaleDerivative(rho) * x.u};
}

State ObjectiveGradient(const State &x) { return {x.u, 0.0}; }

ForwardResult Forward(real_t rho, real_t omega0_squared,
                      int num_steps, real_t h)
{
   constexpr std::array<real_t, 4> b = {
      1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0};
   ForwardResult result;
   result.steps.reserve(num_steps);
   State x = {1.0, 0.0};
   for (int n = 0; n < num_steps; n++)
   {
      StepData step;
      step.x_left = x;
      step.y[0] = x;
      step.k[0] = Rhs(step.y[0], omega0_squared, rho);
      step.y[1] = x + 0.5 * h * step.k[0];
      step.k[1] = Rhs(step.y[1], omega0_squared, rho);
      step.y[2] = x + 0.5 * h * step.k[1];
      step.k[2] = Rhs(step.y[2], omega0_squared, rho);
      step.y[3] = x + h * step.k[2];
      step.k[3] = Rhs(step.y[3], omega0_squared, rho);
      x += h * (b[0] * step.k[0] + b[1] * step.k[1] +
                b[2] * step.k[2] + b[3] * step.k[3]);
      step.x_right = x;
      for (int i = 0; i < 4; i++)
      {
         result.objective += 0.5 * h * b[i] * step.y[i].u * step.y[i].u;
      }
      result.steps.push_back(step);
   }
   return result;
}

struct GradientResult
{
   State initial_adjoint;
   real_t gradient = 0.0;
};

GradientResult DiscreteReverseAD(const ForwardResult &forward, real_t rho,
                                 real_t omega0_squared, real_t h)
{
   constexpr std::array<real_t, 4> b = {
      1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0};
   State p = {0.0, 0.0};
   real_t gradient = 0.0;
   for (int n = static_cast<int>(forward.steps.size()) - 1; n >= 0; n--)
   {
      const StepData &step = forward.steps[n];
      std::array<State, 4> bar_k = {
         h * b[0] * p, h * b[1] * p, h * b[2] * p, h * b[3] * p};
      State bar_x = p;
      for (int i = 3; i >= 0; i--)
      {
         gradient += Dot(RhoDerivative(step.y[i], omega0_squared, rho),
                         bar_k[i]);
         State bar_y = JacobianTransposeTimes(
            bar_k[i], omega0_squared, rho);
         bar_y += h * b[i] * ObjectiveGradient(step.y[i]);
         bar_x += bar_y;
         if (i == 3) { bar_k[2] += h * bar_y; }
         if (i == 2) { bar_k[1] += 0.5 * h * bar_y; }
         if (i == 1) { bar_k[0] += 0.5 * h * bar_y; }
      }
      p = bar_x;
   }
   return {p, gradient};
}

GradientResult TransformedAdjoint(const ForwardResult &forward, real_t rho,
                                  real_t omega0_squared, real_t h)
{
   constexpr std::array<real_t, 4> b = {
      1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0};
   State p = {0.0, 0.0};
   real_t gradient = 0.0;
   for (int n = static_cast<int>(forward.steps.size()) - 1; n >= 0; n--)
   {
      const StepData &step = forward.steps[n];
      const auto g = [&](int i, const State &stage_adjoint)
      {
         return JacobianTransposeTimes(stage_adjoint, omega0_squared, rho) +
                ObjectiveGradient(step.y[i]);
      };
      const State p4 = p;
      const State g4 = g(3, p4);
      const State p3 = p + 0.5 * h * g4;
      const State g3 = g(2, p3);
      const State p2 = p + 0.5 * h * g3;
      const State g2 = g(1, p2);
      const State p1 = p + h * g2;
      const State g1 = g(0, p1);
      gradient += h * (b[0] * Dot(RhoDerivative(step.y[0], omega0_squared,
                                                  rho), p1) +
                       b[1] * Dot(RhoDerivative(step.y[1], omega0_squared,
                                                  rho), p2) +
                       b[2] * Dot(RhoDerivative(step.y[2], omega0_squared,
                                                  rho), p3) +
                       b[3] * Dot(RhoDerivative(step.y[3], omega0_squared,
                                                  rho), p4));
      p += h * (b[0] * g1 + b[1] * g2 + b[2] * g3 + b[3] * g4);
   }
   return {p, gradient};
}

State CubicHermite(const StepData &step, real_t theta, real_t h,
                   real_t omega0_squared, real_t rho)
{
   const real_t theta2 = theta * theta;
   const real_t theta3 = theta2 * theta;
   const real_t h00 = 2.0 * theta3 - 3.0 * theta2 + 1.0;
   const real_t h10 = theta3 - 2.0 * theta2 + theta;
   const real_t h01 = -2.0 * theta3 + 3.0 * theta2;
   const real_t h11 = theta3 - theta2;
   return h00 * step.x_left +
          h10 * h * Rhs(step.x_left, omega0_squared, rho) +
          h01 * step.x_right +
          h11 * h * Rhs(step.x_right, omega0_squared, rho);
}

GradientResult NaiveHermiteAdjoint(const ForwardResult &forward, real_t rho,
                                   real_t omega0_squared, real_t h)
{
   constexpr std::array<real_t, 4> b = {
      1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0};
   State p = {0.0, 0.0};
   real_t gradient = 0.0;
   for (int n = static_cast<int>(forward.steps.size()) - 1; n >= 0; n--)
   {
      const StepData &step = forward.steps[n];
      // Reverse-time RK4 samples physical times right, midpoint, midpoint,
      // left. The timestep remains h; only the forward state is reconstructed.
      const State x_right = step.x_right;
      const State x_mid = CubicHermite(step, 0.5, h, omega0_squared, rho);
      const State x_left = step.x_left;
      const auto g = [&](const State &state, const State &stage_adjoint)
      {
         return JacobianTransposeTimes(stage_adjoint, omega0_squared, rho) +
                ObjectiveGradient(state);
      };
      const State p_right = p;
      const State g_right = g(x_right, p_right);
      const State p_mid_1 = p + 0.5 * h * g_right;
      const State g_mid_1 = g(x_mid, p_mid_1);
      const State p_mid_2 = p + 0.5 * h * g_mid_1;
      const State g_mid_2 = g(x_mid, p_mid_2);
      const State p_left = p + h * g_mid_2;
      const State g_left = g(x_left, p_left);
      gradient += h * (b[0] * Dot(RhoDerivative(x_right, omega0_squared,
                                                  rho), p_right) +
                       b[1] * Dot(RhoDerivative(x_mid, omega0_squared,
                                                  rho), p_mid_1) +
                       b[2] * Dot(RhoDerivative(x_mid, omega0_squared,
                                                  rho), p_mid_2) +
                       b[3] * Dot(RhoDerivative(x_left, omega0_squared,
                                                  rho), p_left));
      p += h * (b[0] * g_right + b[1] * g_mid_1 +
                b[2] * g_mid_2 + b[3] * g_left);
   }
   return {p, gradient};
}

real_t RelativeError(real_t value, real_t reference)
{
   return std::abs(value - reference) /
          std::max(std::abs(reference), real_t(1e-30));
}

real_t StateNorm(const State &value)
{
   return std::sqrt(std::max(Dot(value, value), real_t(0.0)));
}

struct OptimizationOptions
{
   bool run = false;
   real_t initial_omega_h = 1.8;
   real_t rho_initial = 0.60;
   real_t rho_min = 0.45;
   real_t rho_max = 0.65;
   real_t step_size = 0.005;
   int iterations = 50;
};

bool ParseRealArgument(const char *text, real_t &value)
{
   char *end = nullptr;
   const double parsed = std::strtod(text, &end);
   if (end == text || *end != '\0' || !std::isfinite(parsed)) { return false; }
   value = parsed;
   return true;
}

bool ParseIntArgument(const char *text, int &value)
{
   char *end = nullptr;
   const long parsed = std::strtol(text, &end, 10);
   if (end == text || *end != '\0' || parsed < 0 ||
       parsed > std::numeric_limits<int>::max())
   {
      return false;
   }
   value = static_cast<int>(parsed);
   return true;
}

bool ParseOptimizationOptions(int argc, char *argv[], OptimizationOptions &options)
{
   for (int i = 1; i < argc; i++)
   {
      const std::string argument(argv[i]);
      if (argument == "--optimize")
      {
         options.run = true;
      }
      else if (argument == "--omega-h" || argument == "--rho-initial" ||
               argument == "--rho-min" || argument == "--rho-max" ||
               argument == "--step-size")
      {
         if (++i == argc) { return false; }
         real_t value = 0.0;
         if (!ParseRealArgument(argv[i], value)) { return false; }
         if (argument == "--omega-h") { options.initial_omega_h = value; }
         if (argument == "--rho-initial") { options.rho_initial = value; }
         if (argument == "--rho-min") { options.rho_min = value; }
         if (argument == "--rho-max") { options.rho_max = value; }
         if (argument == "--step-size") { options.step_size = value; }
      }
      else if (argument == "--iterations")
      {
         if (++i == argc || !ParseIntArgument(argv[i], options.iterations))
         {
            return false;
         }
      }
      else
      {
         return false;
      }
   }
   return !options.run ||
          (std::isfinite(options.initial_omega_h) &&
           options.initial_omega_h > 0.0 &&
           std::isfinite(options.rho_initial) &&
           std::isfinite(options.rho_min) && std::isfinite(options.rho_max) &&
           options.rho_min < options.rho_initial &&
           options.rho_initial < options.rho_max &&
           options.rho_min > 0.0 && options.rho_max <= 1.0 &&
           std::isfinite(options.step_size) && options.step_size > 0.0 &&
           options.iterations > 0);
}

void PrintOptimizationUsage()
{
   if (!mfem::Mpi::Root()) { return; }
   mfem::err << "Usage: test_rk4_stage_sensitive_oscillator [--optimize "
                "--omega-h VALUE --iterations N --step-size VALUE "
                "--rho-initial VALUE --rho-min VALUE --rho-max VALUE]\n";
}

void RunProjectedGradientOptimization(const OptimizationOptions &options)
{
   constexpr real_t h = 0.05;
   constexpr int num_steps = 80;
   constexpr real_t rk4_imaginary_stability_limit =
      2.8284271247461900976033774484194; // 2 sqrt(2)
   const real_t omega_initial = options.initial_omega_h / h;
   const real_t omega0_squared =
      omega_initial * omega_initial / Scale(options.rho_initial);
   const auto omega_h = [&](real_t rho)
   {
      return h * std::sqrt(omega0_squared * Scale(rho));
   };
   MFEM_VERIFY(omega_h(options.rho_min) < rk4_imaginary_stability_limit &&
               omega_h(options.rho_max) < rk4_imaginary_stability_limit,
               "Oscillator optimization bounds exceed RK4's imaginary-axis "
               "stability interval.");

   real_t rho_do = options.rho_initial;
   real_t rho_naive = options.rho_initial;
   if (mfem::Mpi::Root())
   {
      mfem::out << std::scientific << std::setprecision(12)
                << "# projected oscillator optimization: h=" << h
                << ", N=" << num_steps
                << ", rho_initial=" << options.rho_initial
                << ", rho_bounds=[" << options.rho_min << ','
                << options.rho_max << "]"
                << ", step_size=" << options.step_size
                << ", initial_omega_h=" << options.initial_omega_h << '\n'
                << "method,iteration,rho,omega_h,J_h,gradient,"
                   "initial_adjoint_norm,relative_gradient_error_to_do_"
                   "at_same_rho\n";
   }

   for (int iteration = 0; iteration <= options.iterations; iteration++)
   {
      const ForwardResult forward_do = Forward(
         rho_do, omega0_squared, num_steps, h);
      const GradientResult gradient_do = DiscreteReverseAD(
         forward_do, rho_do, omega0_squared, h);
      const GradientResult gradient_modified = TransformedAdjoint(
         forward_do, rho_do, omega0_squared, h);
      const ForwardResult forward_naive = Forward(
         rho_naive, omega0_squared, num_steps, h);
      const GradientResult gradient_naive = NaiveHermiteAdjoint(
         forward_naive, rho_naive, omega0_squared, h);
      const real_t modified_error = RelativeError(
         gradient_modified.gradient, gradient_do.gradient);

      MFEM_VERIFY(modified_error < 5e-12,
                  "OD_modified changed the DO optimization trajectory.");
      MFEM_VERIFY(std::isfinite(gradient_do.gradient) &&
                  std::isfinite(gradient_naive.gradient),
                  "Oscillator optimization produced a non-finite gradient.");
      if (mfem::Mpi::Root())
      {
         mfem::out << "DO," << iteration << ',' << rho_do << ','
                   << omega_h(rho_do) << ',' << forward_do.objective << ','
                   << gradient_do.gradient << ','
                   << StateNorm(gradient_do.initial_adjoint) << ",0\n"
                   << "OD_modified," << iteration << ',' << rho_do << ','
                   << omega_h(rho_do) << ',' << forward_do.objective << ','
                   << gradient_modified.gradient << ','
                   << StateNorm(gradient_modified.initial_adjoint) << ','
                   << modified_error << '\n'
                   << "OD_naive_Hermite," << iteration << ',' << rho_naive
                   << ',' << omega_h(rho_naive) << ','
                   << forward_naive.objective << ','
                   << gradient_naive.gradient << ','
                   << StateNorm(gradient_naive.initial_adjoint)
                   << ",nan\n";
      }
      if (iteration == options.iterations) { break; }
      rho_do = std::clamp(rho_do - options.step_size * gradient_do.gradient,
                          options.rho_min, options.rho_max);
      rho_naive = std::clamp(
         rho_naive - options.step_size * gradient_naive.gradient,
         options.rho_min, options.rho_max);
   }
}

} // namespace

int main(int argc, char *argv[])
{
   mfem::Mpi::Init(argc, argv);
   mfem::Hypre::Init();

   OptimizationOptions optimization;
   if (!ParseOptimizationOptions(argc, argv, optimization))
   {
      PrintOptimizationUsage();
      return 2;
   }
   if (optimization.run)
   {
      RunProjectedGradientOptimization(optimization);
      return 0;
   }

   constexpr real_t rho = 0.60;
   constexpr real_t h = 0.05;
   constexpr int num_steps = 80;
   constexpr real_t pi = 3.1415926535897932384626433832795;
   // Values well inside the explicit RK4 imaginary-axis stability interval
   // (omega*h < 2 sqrt(2)), with 1.8 intentionally stage-sensitive.
   const std::array<real_t, 4> nondimensional_frequencies = {
      0.40, 1.00, 1.80, 2.40};

   if (mfem::Mpi::Root())
   {
      mfem::out << std::scientific << std::setprecision(12)
                << "# Manufactured stage-sensitive oscillator: rho=" << rho
                << ", h=" << h << ", N=" << num_steps << '\n'
                << "# omega*h, omega, periods-per-step, J_h, g_DO, "
                   "abs(OD_modified-DO), rel(OD_modified,DO), "
                   "abs(OD_naive-DO), rel(OD_naive,DO), "
                   "abs(p_naive(0)-p_DO(0)), rel(p_naive(0),p_DO(0)), "
                   "DO centered-FD error\n";
   }

   for (const real_t omega_h : nondimensional_frequencies)
   {
      const real_t omega = omega_h / h;
      const real_t omega0_squared = omega * omega / Scale(rho);
      const ForwardResult forward = Forward(
         rho, omega0_squared, num_steps, h);
      const GradientResult do_result = DiscreteReverseAD(
         forward, rho, omega0_squared, h);
      const GradientResult modified = TransformedAdjoint(
         forward, rho, omega0_squared, h);
      const GradientResult naive = NaiveHermiteAdjoint(
         forward, rho, omega0_squared, h);

      const real_t epsilon = 1e-6;
      const real_t centered =
         (Forward(rho + epsilon, omega0_squared, num_steps, h).objective -
          Forward(rho - epsilon, omega0_squared, num_steps, h).objective) /
         (2.0 * epsilon);
      const real_t modified_error = RelativeError(
         modified.gradient, do_result.gradient);
      const real_t naive_error = RelativeError(
         naive.gradient, do_result.gradient);
      const State p_difference =
         naive.initial_adjoint - do_result.initial_adjoint;
      const real_t p_naive_absolute_error = std::sqrt(
         Dot(p_difference, p_difference));
      const real_t p_naive_error = p_naive_absolute_error /
         std::max(std::sqrt(Dot(do_result.initial_adjoint,
                                do_result.initial_adjoint)), real_t(1e-30));
      const real_t modified_absolute_error = std::abs(
         modified.gradient - do_result.gradient);
      const real_t naive_absolute_error = std::abs(
         naive.gradient - do_result.gradient);
      const real_t fd_error = RelativeError(centered, do_result.gradient);

      MFEM_VERIFY(modified_error < 5e-12,
                  "Transformed oscillator adjoint must match reverse AD.");
      MFEM_VERIFY(fd_error < 5e-7,
                  "Oscillator reverse-AD gradient failed centered FD.");
      MFEM_VERIFY(std::isfinite(naive_error) && std::isfinite(p_naive_error),
                  "Naive oscillator adjoint produced a non-finite metric.");

      if (mfem::Mpi::Root())
      {
         mfem::out << omega_h << ',' << omega << ','
                   << omega_h / (2.0 * pi) << ','
                   << forward.objective << ',' << do_result.gradient << ','
                   << modified_absolute_error << ',' << modified_error << ','
                   << naive_absolute_error << ',' << naive_error << ','
                   << p_naive_absolute_error << ',' << p_naive_error << ','
                   << fd_error << '\n';
      }
   }

   return 0;
}
