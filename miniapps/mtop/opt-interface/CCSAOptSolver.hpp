/**
 * @file CCSAOptSolver.hpp
 * @brief Bridge the entropy-CCSA ("Bregman") optimiser to
 *        mfem::OptimizationSolver.
 *
 *   - mfem_mma::CCSAOptimizationSolver          (serial, CCSAOptimizer)
 *   - mfem_mma::CCSAOptimizationSolverParallel  (MPI, CCSAOptimizerParallel)
 *
 * Reuses SeparableApproxSolver (MMAOptSolver.hpp) for the OptimizationProblem ->
 * packed-constraint mapping, Riesz map, GCMMA toggle, monitors and stopping.
 * The difference from the MMA family is the outer loop: CCSA is a Bregman /
 * mirror-descent method whose optimiser variable is a LATENT eta (unconstrained
 * — the box bounds live in a BoundsGeometry). Update()/UpdateGCMMA()/
 * KKTresidual() take the latent eta and physical-space gradients and DO NOT take
 * xmin/xmax; the physical design is x = ToPhysical(eta).
 *
 * Same public API as the MMA solvers (SetRelTol/SetAbsTol/SetMaxIter,
 * SetInitialGuess, SetRieszMap, SetGCMMA, GetConverged/GetNumIterations/
 * GetFinalNorm/GetLambda). CCSA already defaults to the robust d=1 elastic
 * weight, so no penalty override is needed.
 *
 * @note The CCSA KKT residual is a LATENT-space stationarity measure (chain
 * rule dL/deta), so its absolute scale differs from the MMA physical
 * projected-gradient residual; size abs_tol accordingly.
 */

#pragma once

#include "MMAOptSolver.hpp"        // SeparableApproxSolver
#include "CCSA_Bregman_MFEM.hpp"   // CCSAOptimizer / CCSAOptimizerParallel

namespace mfem_mma
{

/// @brief CCSA (entropy/Bregman) family: latent-variable outer loop.
/// @tparam Opt  CCSAOptimizer or CCSAOptimizerParallel.
template <class Opt>
class CCSAFamilySolver : public SeparableApproxSolver
{
public:
   using SeparableApproxSolver::SeparableApproxSolver;

   void Mult(const mfem::Vector &xt, mfem::Vector &x) const override
   {
      MFEM_VERIFY(problem && opt_, "SetOptimizationProblem must be called first");
      x.SetSize(n_);
      x = has_x0_ ? x0_ : xt;                 // initial physical design
      eta_.SetSize(n_); eta_.UseDevice(true);
      opt_->ToLatent(x, eta_);                // physical -> latent

      converged = false; initial_norm = -1.0; final_iter = 0; final_norm = -1.0;
      bool stop = false;
      for (int it = 0; it <= max_iter; it++)
      {
         opt_->ToPhysical(eta_, x);           // current physical iterate
         mfem::real_t f0;
         EvalGradientsAt_(x, f0);             // physical-space gradients at x
         const mfem::real_t kkt =
            opt_->KKTresidual(eta_, df0_, f0, fival_, dfidx_.data());
         stop = RecordIterate_(it, kkt, x, f0, "CCSA");
         if (stop || it == max_iter) { break; }

         int inner = 0;
         if (!gcmma_)
         { opt_->Update(eta_, df0_, f0, fival_, dfidx_.data()); }
         else if (gcmma_conservative_)
         { opt_->UpdateGCMMA(eta_, df0_, f0, fival_, dfidx_.data(),
                             MakeGCMMACallback_(), gcmma_max_inner_, &inner); }
         else
         { opt_->UpdateGCMMA(eta_, df0_, f0, fival_, dfidx_.data(), &inner); }
      }
      opt_->ToPhysical(eta_, x);              // final physical design
      ReportSummary_(stop, x, "CCSAOptimizationSolver");
   }

   const std::vector<double> &GetLambda() const { return opt_->GetLambda(); }

protected:
   mutable std::unique_ptr<Opt> opt_;
   mutable mfem::Vector eta_;
};

/// @brief Serial CCSA (Bregman) solver.
class CCSAOptimizationSolver : public CCSAFamilySolver<CCSAOptimizer>
{
protected:
   void BuildOptimizer_(int n, int n_ineq, int n_eq) override
   {
      opt_.reset(new CCSAOptimizer(
                    CCSAOptimizer::WithEqualities(n, n_ineq, n_eq)));
      opt_->SetBounds(xmin_, xmax_);   // TwoSided geometry from dof bounds
   }
};

#ifdef MFEM_USE_MPI
/// @brief Distributed (MPI) CCSA (Bregman) solver. Local design chunk per rank;
/// the problem must return globally-reduced objective/constraint values.
class CCSAOptimizationSolverParallel : public CCSAFamilySolver<CCSAOptimizerParallel>
{
public:
   explicit CCSAOptimizationSolverParallel(MPI_Comm comm)
      : CCSAFamilySolver<CCSAOptimizerParallel>(comm), comm_(comm) {}
protected:
   void BuildOptimizer_(int n_local, int n_ineq, int n_eq) override
   {
      opt_.reset(new CCSAOptimizerParallel(
                    CCSAOptimizerParallel::WithEqualities(comm_, n_local,
                                                          n_ineq, n_eq)));
      opt_->SetBounds(xmin_, xmax_);   // local chunks
   }
private:
   MPI_Comm comm_;
};
#endif // MFEM_USE_MPI

} // namespace mfem_mma
