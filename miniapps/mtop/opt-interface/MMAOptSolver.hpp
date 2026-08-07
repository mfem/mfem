/**
 * @file MMAOptSolver.hpp
 * @brief Bridge the Svanberg-family optimisers (MMA/GCMMA and the entropy-CCSA
 *        "Bregman" variant) to mfem::OptimizationSolver.
 *
 * Solvers provided:
 *   - mfem_mma::MMAOptimizationSolver          (serial, wraps MMAOptimizer)
 *   - mfem_mma::MMAOptimizationSolverParallel  (MPI, MMAOptimizerParallel)
 *   - mfem_mma::CCSAOptimizationSolver          (serial, CCSAOptimizer)   [CCSAOptSolver.hpp]
 *   - mfem_mma::CCSAOptimizationSolverParallel  (MPI, CCSAOptimizerParallel)
 *
 * All share SeparableApproxSolver, which maps an mfem::OptimizationProblem
 *   min F(x) s.t. C(x)=c_e, d_lo<=D(x)<=d_hi, x_lo<=x<=x_hi
 * onto the packed MMA/CCSA constraint form (min f0 s.t. fi<=0):
 *   - finite upper side  D_i<=d_hi_i -> D_i-d_hi_i<=0
 *   - finite lower side  d_lo_i<=D_i -> d_lo_i-D_i<=0
 *   - equality           C_i=c_e_i   -> h_i=C_i-c_e_i=0   (±h encoding)
 * The packed arrays are [ ineq | +h | -h ] (fival) and [ dineq | +dh | -dh ]
 * (dfidx), filled in place each iteration (no per-iteration allocation).
 * Constraint-Jacobian rows come from a unit-seed VJP (Jᵀe_i), so any Operator
 * gradient works (DenseMatrix or the matrix-free MFGrad).
 *
 * @section gcmma GCMMA
 * SetGCMMA() switches the outer step from the plain Update() to the globally
 * convergent UpdateGCMMA().  In conservative mode (default) the wrapper hands
 * MMA/CCSA a true-model callback that re-evaluates the problem's objective and
 * constraint VALUES at each trial point, so the inner loop enforces the
 * conservatism condition — the robust variant for non-convex problems.
 *
 * @section riesz Riesz map
 * SetRieszMap() applies a metric R to the objective and constraint gradients.
 * @warning R must be DIAGONAL (see the per-method note); a non-diagonal R can
 * make the per-component box-KKT check report false convergence at active
 * bounds and breaks the per-component separable model.
 *
 * @section stop Stopping
 * The optimiser's KKT residual is the IterativeSolver "norm": convergence when
 * kkt <= max(abs_tol, rel_tol*kkt_0).  Defaults: abs_tol=1e-8, rel_tol=0,
 * max_iter=100 (overridable).
 */

#pragma once

#include "MMA_MFEM.hpp"
#include <mfem.hpp>
#include <vector>
#include <memory>
#include <functional>

namespace mfem_mma
{

/// @brief Optimizer-agnostic base: maps an OptimizationProblem onto the packed
/// MMA/CCSA constraint arrays and holds all shared buffers/config. Concrete
/// solvers implement BuildOptimizer_() (construct the typed optimiser) and
/// Mult() (drive it).
class SeparableApproxSolver : public mfem::OptimizationSolver
{
public:
   SeparableApproxSolver() { SetDefaults_(); }
#ifdef MFEM_USE_MPI
   SeparableApproxSolver(MPI_Comm comm) : mfem::OptimizationSolver(comm)
   { SetDefaults_(); }
#endif

   /// @brief Diagonal Riesz map R applied to all gradients (not owned).
   /// @warning R must be DIAGONAL; a non-diagonal metric can cause false
   ///          convergence at active variable bounds.
   void SetRieszMap(const mfem::Operator &riesz) { riesz_ = &riesz; }

   /// @brief Initial design. If unset, Mult() uses its @p xt argument.
   void SetInitialGuess(const mfem::Vector &x0) { x0_ = x0; has_x0_ = true; }

   /// @brief Use the globally-convergent GCMMA step (UpdateGCMMA).
   /// @param on           enable GCMMA (default: enable).
   /// @param conservative use the true-model conservative inner loop (default);
   ///                     if false, the single-subproblem GCMMA step is used.
   /// @param max_inner    max conservative inner iterations (conservative only).
   void SetGCMMA(bool on = true, bool conservative = true, int max_inner = 15)
   { gcmma_ = on; gcmma_conservative_ = conservative; gcmma_max_inner_ = max_inner; }

   void SetOptimizationProblem(const mfem::OptimizationProblem &prob) override
   {
      mfem::OptimizationSolver::SetOptimizationProblem(prob);
      SetupMapping_(prob);
      BuildOptimizer_(n_, n_ineq_, n_eq_);
   }

protected:
   /// Construct the concrete (typed) optimiser into the derived class.
   virtual void BuildOptimizer_(int n, int n_ineq, int n_eq) = 0;

   void SetDefaults_()
   {
      max_iter = 100;
      abs_tol  = mfem::real_t(1e-8);
      rel_tol  = mfem::real_t(0.0);
   }

   /// Compute the constraint layout and allocate all work buffers once.
   void SetupMapping_(const mfem::OptimizationProblem &prob)
   {
      n_  = prob.input_size;
      const mfem::Operator *C = prob.GetC();
      const mfem::Operator *D = prob.GetD();
      nC_ = C ? C->Height() : 0;
      nD_ = D ? D->Height() : 0;

      ineq_row_.clear(); ineq_sign_.clear(); ineq_bnd_.clear();
      const mfem::Vector *dlo = prob.GetInequalityVec_Lo();
      const mfem::Vector *dhi = prob.GetInequalityVec_Hi();
      if (dlo) { dlo->HostRead(); }
      if (dhi) { dhi->HostRead(); }
      for (int i = 0; i < nD_; i++)
      {
         if (dhi && (*dhi)(i) < mfem::infinity())
         { ineq_row_.push_back(i); ineq_sign_.push_back(1.0); ineq_bnd_.push_back((*dhi)(i)); }
         if (dlo && (*dlo)(i) > -mfem::infinity())
         { ineq_row_.push_back(i); ineq_sign_.push_back(-1.0); ineq_bnd_.push_back((*dlo)(i)); }
      }
      n_ineq_ = (int)ineq_row_.size();
      n_eq_   = nC_;
      m_      = n_ineq_ + 2 * n_eq_;

      c_e_.SetSize(nC_); c_e_ = 0.0;
      if (nC_ > 0 && prob.GetEqualityVec())
      { prob.GetEqualityVec()->HostRead(); c_e_ = *prob.GetEqualityVec(); }
      c_e_.HostRead();

      MFEM_VERIFY(prob.GetBoundsVec_Lo() && prob.GetBoundsVec_Hi(),
                  "SeparableApproxSolver requires solution bounds "
                  "(set SetDofBounds on the OptimProblem)");
      xmin_.SetSize(n_); xmin_.UseDevice(true); xmin_ = *prob.GetBoundsVec_Lo();
      xmax_.SetSize(n_); xmax_.UseDevice(true); xmax_ = *prob.GetBoundsVec_Hi();

      // O(n), device-resident.
      df0_.SetSize(n_); df0_.UseDevice(true);
      dfidx_.assign(m_, mfem::Vector());
      for (auto &v : dfidx_) { v.SetSize(n_); v.UseDevice(true); }
      riesz_tmp_.SetSize(n_); riesz_tmp_.UseDevice(true);
      mon_r_.SetSize(n_);
      // O(m), host.
      fival_.SetSize(m_);
      dvals_.SetSize(nD_);
      cvals_.SetSize(nC_);
      seedC_.SetSize(nC_);
      seedD_.SetSize(nD_);
   }

   /// Fill df0_ (Riesz'd), fival_ and dfidx_ with values+gradients at @p x.
   void EvalGradientsAt_(const mfem::Vector &x, mfem::real_t &f0) const
   {
      f0 = problem->CalcObjective(x);
      problem->CalcObjectiveGrad(x, df0_);
      ApplyRiesz_(df0_);
      if (nD_ > 0)
      {
         const mfem::Operator *D = problem->GetD();
         D->Mult(x, dvals_);
         mfem::Operator &Jd = D->GetGradient(x);
         for (int k = 0; k < n_ineq_; k++)
         {
            const int i = ineq_row_[k];
            const mfem::real_t s = ineq_sign_[k];
            fival_(k) = s * (dvals_(i) - ineq_bnd_[k]);
            GradRow_(Jd, seedD_, i, dfidx_[k]);
            dfidx_[k] *= s;
            ApplyRiesz_(dfidx_[k]);
         }
      }
      if (nC_ > 0)
      {
         const mfem::Operator *C = problem->GetC();
         C->Mult(x, cvals_);
         mfem::Operator &Jc = C->GetGradient(x);
         for (int i = 0; i < nC_; i++)
         {
            const mfem::real_t h = cvals_(i) - c_e_(i);
            fival_(n_ineq_ + i)         =  h;
            fival_(n_ineq_ + n_eq_ + i) = -h;
            mfem::Vector &gp = dfidx_[n_ineq_ + i];
            GradRow_(Jc, seedC_, i, gp);
            ApplyRiesz_(gp);
            mfem::Vector &gm = dfidx_[n_ineq_ + n_eq_ + i];
            gm = gp; gm *= mfem::real_t(-1);
         }
      }
   }

   /// Fill @p fout (size m_) and @p f0 with true objective/constraint VALUES at
   /// @p x — no gradients. Used as the GCMMA conservative-inner-loop callback.
   void EvalValuesAt_(const mfem::Vector &x, mfem::Vector &fout,
                      mfem::real_t &f0) const
   {
      f0 = problem->CalcObjective(x);
      fout.SetSize(m_);
      if (nD_ > 0)
      {
         const mfem::Operator *D = problem->GetD();
         D->Mult(x, dvals_);
         for (int k = 0; k < n_ineq_; k++)
         { fout(k) = ineq_sign_[k] * (dvals_(ineq_row_[k]) - ineq_bnd_[k]); }
      }
      if (nC_ > 0)
      {
         const mfem::Operator *C = problem->GetC();
         C->Mult(x, cvals_);
         for (int i = 0; i < nC_; i++)
         {
            const mfem::real_t h = cvals_(i) - c_e_(i);
            fout(n_ineq_ + i)         =  h;
            fout(n_ineq_ + n_eq_ + i) = -h;
         }
      }
   }

   /// GCMMA true-model callback bound to EvalValuesAt_ (physical trial point).
   GCMMAEvalCallback MakeGCMMACallback_() const
   {
      return [this](const mfem::Vector &xc, mfem::Vector &fi, mfem::real_t &f0)
      { this->EvalValuesAt_(xc, fi, f0); };
   }

   void GradRow_(const mfem::Operator &J, mfem::Vector &seed, int i,
                 mfem::Vector &row) const
   { seed = 0.0; seed(i) = mfem::real_t(1.0); J.MultTranspose(seed, row); }

   void ApplyRiesz_(mfem::Vector &g) const
   { if (riesz_) { riesz_->Mult(g, riesz_tmp_); g = riesz_tmp_; } }

   /// Report/monitor + convergence bookkeeping for one iterate. Returns true if
   /// the stopping test is met.
   bool RecordIterate_(int it, mfem::real_t kkt, const mfem::Vector &x,
                       mfem::real_t f0, const char *tag) const
   {
      if (it == 0) { initial_norm = kkt; }
      final_iter = it;
      final_norm = kkt;
      Monitor(it, kkt, mon_r_, x, /*final=*/false);
      if (print_options.iterations)
      {
         mfem::out << tag << " iteration " << it << " : kkt = " << kkt
                   << " , f0 = " << f0 << '\n';
      }
      const mfem::real_t stop =
         std::max(abs_tol, rel_tol * (initial_norm > 0 ? initial_norm : 1.0));
      return kkt <= stop;
   }

   void ReportSummary_(bool ok, const mfem::Vector &x, const char *tag) const
   {
      converged = ok;
      Monitor(final_iter, final_norm, mon_r_, x, /*final=*/true);
      if (print_options.summary || (print_options.errors && !ok))
      {
         mfem::out << tag << ": " << (ok ? "converged" : "NOT converged")
                   << " in " << final_iter << " iterations, kkt = "
                   << final_norm << '\n';
      }
   }

   // Layout (set in SetupMapping_).
   int n_ = 0, nC_ = 0, nD_ = 0, n_ineq_ = 0, n_eq_ = 0, m_ = 0;
   std::vector<int>          ineq_row_;
   std::vector<mfem::real_t> ineq_sign_, ineq_bnd_;
   mfem::Vector c_e_;

   const mfem::Operator *riesz_ = nullptr;
   mfem::Vector x0_;
   bool has_x0_ = false;

   bool gcmma_ = false, gcmma_conservative_ = true;
   int  gcmma_max_inner_ = 15;

   // Work buffers (mutable: Mult is const). Allocated once.
   mutable mfem::Vector df0_, xmin_, xmax_, riesz_tmp_, mon_r_;
   mutable std::vector<mfem::Vector> dfidx_;
   mutable mfem::Vector fival_, dvals_, cvals_, seedC_, seedD_;
};

/// @brief MMA/GCMMA family: physical-variable outer loop with move-limit bounds.
/// @tparam Opt  MMAOptimizer or MMAOptimizerParallel.
template <class Opt>
class MMAFamilySolver : public SeparableApproxSolver
{
public:
   using SeparableApproxSolver::SeparableApproxSolver;

   void Mult(const mfem::Vector &xt, mfem::Vector &x) const override
   {
      MFEM_VERIFY(problem && opt_, "SetOptimizationProblem must be called first");
      x.SetSize(n_);
      x = has_x0_ ? x0_ : xt;

      converged = false; initial_norm = -1.0; final_iter = 0; final_norm = -1.0;

      bool stop = false;
      for (int it = 0; it <= max_iter; it++)
      {
         mfem::real_t f0;
         EvalGradientsAt_(x, f0);
         const mfem::real_t kkt = opt_->KKTresidual(
            x, df0_, f0, fival_, dfidx_.data(), xmin_, xmax_);
         stop = RecordIterate_(it, kkt, x, f0, "MMA");
         if (stop || it == max_iter) { break; }

         int inner = 0;
         OuterStep_(x, f0, inner);   // one Update/UpdateGCMMA (optimiser-specific)
      }
      ReportSummary_(stop, x, "MMAOptimizationSolver");
   }

   const std::vector<double> &GetLambda() const { return opt_->GetLambda(); }

protected:
   mutable std::unique_ptr<Opt> opt_;

   /// One physical outer step. MMA and SQ differ in their UpdateGCMMA API
   /// (SQ folds the callback into one overload with a different callback type),
   /// so the step is supplied by a family mixin (MMAGcmmaMixin / SQGcmmaMixin).
   virtual void OuterStep_(mfem::Vector &x, mfem::real_t f0, int &inner) const = 0;

   // a=0, c=max(1000,10*n[_global]), d=1 (see the derived factories). d=1 only
   // penalises the zero-at-optimum elastic slack (optimum unchanged) but is far
   // more robust than the default d=0 on multi-inequality problems.
   static void FillPenalties_(int n, int m, std::vector<double> &a,
                              std::vector<double> &c, std::vector<double> &d)
   {
      a.assign(m, 0.0);
      c.assign(m, std::max(1000.0, 10.0 * n));
      d.assign(m, 1.0);
   }
};

/// @brief Outer-step mixin for MMA-style optimisers (MMAOptimizer[Parallel]):
/// UpdateGCMMA takes a (x, fival, f0) true-model callback plus max_inner.
template <class Opt>
class MMAGcmmaMixin : public MMAFamilySolver<Opt>
{
public:
   using MMAFamilySolver<Opt>::MMAFamilySolver;
protected:
   void OuterStep_(mfem::Vector &x, mfem::real_t f0, int &inner) const override
   {
      auto &o = *this->opt_;
      if (!this->gcmma_)
      { o.Update(x, this->df0_, f0, this->fival_, this->dfidx_.data(),
                 this->xmin_, this->xmax_); }
      else if (this->gcmma_conservative_)
      { o.UpdateGCMMA(x, this->df0_, f0, this->fival_, this->dfidx_.data(),
                      this->xmin_, this->xmax_, this->MakeGCMMACallback_(),
                      this->gcmma_max_inner_, &inner); }
      else
      { o.UpdateGCMMA(x, this->df0_, f0, this->fival_, this->dfidx_.data(),
                      this->xmin_, this->xmax_, &inner); }
   }
};

/// @brief Outer-step mixin for SQ optimisers (SQOptimizer[Parallel]): a single
/// UpdateGCMMA overload with an optional (x, fival, dfidx*) callback. GCMMA here
/// is the single-subproblem (non-conservative) variant — SQ's conservative
/// callback has a different signature and is not routed through the wrapper.
template <class Opt>
class SQGcmmaMixin : public MMAFamilySolver<Opt>
{
public:
   using MMAFamilySolver<Opt>::MMAFamilySolver;
protected:
   void OuterStep_(mfem::Vector &x, mfem::real_t f0, int &inner) const override
   {
      auto &o = *this->opt_;
      if (!this->gcmma_)
      { o.Update(x, this->df0_, f0, this->fival_, this->dfidx_.data(),
                 this->xmin_, this->xmax_); }
      else
      { o.UpdateGCMMA(x, this->df0_, f0, this->fival_, this->dfidx_.data(),
                      this->xmin_, this->xmax_, nullptr, &inner); }
   }
};

/// @brief Serial MMA/GCMMA solver.
class MMAOptimizationSolver : public MMAGcmmaMixin<MMAOptimizer>
{
protected:
   void BuildOptimizer_(int n, int n_ineq, int n_eq) override
   {
      const int m = n_ineq + 2 * n_eq;
      std::vector<double> a, c, d; FillPenalties_(n, m, a, c, d);
      opt_.reset(new MMAOptimizer(MMAOptimizer::WithEqualities(
                                     n, n_ineq, n_eq, a.data(), c.data(), d.data())));
   }
};

#ifdef MFEM_USE_MPI
/// @brief Distributed (MPI) MMA/GCMMA solver. Each rank holds local chunks; the
/// problem must return globally-reduced objective/constraint values.
class MMAOptimizationSolverParallel : public MMAGcmmaMixin<MMAOptimizerParallel>
{
public:
   explicit MMAOptimizationSolverParallel(MPI_Comm comm)
      : MMAGcmmaMixin<MMAOptimizerParallel>(comm), comm_(comm) {}
protected:
   void BuildOptimizer_(int n_local, int n_ineq, int n_eq) override
   {
      int n_global = 0;
      MPI_Allreduce(&n_local, &n_global, 1, MPI_INT, MPI_SUM, comm_);
      const int m = n_ineq + 2 * n_eq;
      std::vector<double> a, c, d; FillPenalties_(n_global, m, a, c, d);
      opt_.reset(new MMAOptimizerParallel(
                    MMAOptimizerParallel::WithEqualities(
                       comm_, n_local, n_ineq, n_eq, a.data(), c.data(), d.data())));
   }
private:
   MPI_Comm comm_;
};
#endif // MFEM_USE_MPI

/// @brief Serial SQ (separable-quadratic, Svanberg 2007 §5.1) solver. Same API
/// and MMA family loop (physical variable, move-limit bounds, GCMMA) as
/// MMAOptimizationSolver, plus SetSigmaScale (the SQ trust-region σ scale).
/// @note SQ's WithEqualities has no (a,c,d) overload, so it uses SQ's default
/// penalties (a=0, c=max(1000,10n), d=0); SQ's separable-quadratic model does
/// not exhibit the multi-inequality fragility the MMA d=1 fix addresses.
class SQOptimizationSolver : public SQGcmmaMixin<SQOptimizer>
{
public:
   /// Scale the SQ trust-region σ. May be called before or after
   /// SetOptimizationProblem.
   void SetSigmaScale(mfem::real_t s)
   { sigma_ = s; has_sigma_ = true; if (opt_) { opt_->SetSigmaScale(s); } }
protected:
   void BuildOptimizer_(int n, int n_ineq, int n_eq) override
   {
      opt_.reset(new SQOptimizer(SQOptimizer::WithEqualities(n, n_ineq, n_eq)));
      if (has_sigma_) { opt_->SetSigmaScale(sigma_); }
   }
private:
   mfem::real_t sigma_ = 0; bool has_sigma_ = false;
};

#ifdef MFEM_USE_MPI
/// @brief Distributed (MPI) SQ solver.
class SQOptimizationSolverParallel : public SQGcmmaMixin<SQOptimizerParallel>
{
public:
   explicit SQOptimizationSolverParallel(MPI_Comm comm)
      : SQGcmmaMixin<SQOptimizerParallel>(comm), comm_(comm) {}
   void SetSigmaScale(mfem::real_t s)
   { sigma_ = s; has_sigma_ = true; if (opt_) { opt_->SetSigmaScale(s); } }
protected:
   void BuildOptimizer_(int n_local, int n_ineq, int n_eq) override
   {
      opt_.reset(new SQOptimizerParallel(
                    SQOptimizerParallel::WithEqualities(comm_, n_local,
                                                        n_ineq, n_eq)));
      if (has_sigma_) { opt_->SetSigmaScale(sigma_); }
   }
private:
   MPI_Comm comm_; mfem::real_t sigma_ = 0; bool has_sigma_ = false;
};
#endif // MFEM_USE_MPI

} // namespace mfem_mma
