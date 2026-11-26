#include "stokes.hpp"
#include <string>
#include <type_traits>

static_assert(std::is_constructible_v<mfem::flow::FlowSolver,
              const mfem::flow::FlowProblem &, const mfem::flow::FlowSolverOptions &>);
static_assert(std::is_constructible_v<mfem::flow::FlowAdjointSolver,
              const mfem::flow::FlowProblem &, const mfem::flow::FlowSolverOptions &>);
static_assert(std::is_same_v<decltype(mfem::flow::IterativeOptions{}.preconditioner),
              mfem::flow::PreconditionerType>);
static_assert(std::variant_size_v<mfem::flow::Stabilization> == 4);
static_assert(!std::is_constructible_v<mfem::flow::PressureGauge,
              mfem::flow::PressureGauge::Mode>);

int main()
{
   using namespace mfem::flow;
   FlowSolverOptions options;
   Stabilization stabilization = CIPOptions{0.125};
   const Stabilization copied = stabilization;
   stabilization = GLSOptions{2.5};
   const auto automatic = PressureGauge::Auto(3.0);
   const auto prescribed = PressureGauge::MeanValue(-2.0);
   const auto none = PressureGauge::None();
   const bool valid = !options.convection &&
      std::get<IterativeOptions>(options.linear.backend).preconditioner == PreconditionerType::BlockUpperTriangularP1AMG &&
      std::get<CIPOptions>(copied).gamma == 0.125 && std::get<GLSOptions>(stabilization).tau_constant == 2.5 &&
      std::string(StabilizationName(copied)) == "cip" && std::string(StabilizationName(stabilization)) == "gls" &&
      std::string(StabilizationName(PSPGOptions{0.5})) == "pspg" &&
      std::string(StabilizationName(NoStabilization{})) == "taylor-hood" &&
      automatic.GetMode() == PressureGauge::Mode::Auto && automatic.GetMean() == 3.0 &&
      prescribed.GetMode() == PressureGauge::Mode::MeanValue && prescribed.GetMean() == -2.0 &&
      none.GetMode() == PressureGauge::Mode::None && !none.GetMean() &&
      std::string(PressureGaugeName(automatic)) == "auto" && std::string(PressureGaugeName(none)) == "none" &&
      std::string(PressureGaugeName(prescribed)) == "mean-value" &&
      std::string(PressureGaugeName(PressureGauge::MeanValue(0.0))) == "mean-zero";
   return valid ? 0 : 1;
}
