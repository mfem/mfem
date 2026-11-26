#include "stokes.hpp"
#include <type_traits>

static_assert(std::is_aggregate_v<mfem::flow::FlowSolverOptions>);
static_assert(std::is_constructible_v<mfem::flow::FlowProblem,
              mfem::ParFiniteElementSpace &, mfem::ParFiniteElementSpace &, mfem::real_t>);
static_assert(std::is_constructible_v<mfem::flow::FlowSolver, const mfem::flow::FlowProblem &>);
static_assert(std::is_constructible_v<mfem::flow::FlowAdjointSolver,
              const mfem::flow::FlowProblem &, const mfem::flow::FlowSolverOptions &>);
static_assert(std::is_same_v<decltype(std::declval<mfem::flow::FlowSolver &>().Solve(
              std::declval<mfem::ParGridFunction &>(), std::declval<mfem::ParGridFunction &>())),
              mfem::flow::FlowSolveReport>);

int main()
{
   mfem::flow::FlowSolverOptions options;
   mfem::flow::IterativeOptions iterative;
   iterative.schur = mfem::flow::SchurApproximation::ExperimentalReactionDiagonal;
   options.linear.backend = iterative;
   const auto copied = options;
   const auto *settings = std::get_if<mfem::flow::IterativeOptions>(&copied.linear.backend);
   return copied.convection || !settings || settings->schur != iterative.schur ? 1 : 0;
}
