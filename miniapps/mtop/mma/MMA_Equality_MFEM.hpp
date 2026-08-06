#ifndef MFEM_MMA_EQUALITY_MFEM_HPP
#define MFEM_MMA_EQUALITY_MFEM_HPP

#include "mfem.hpp"
#include "MMA_MFEM.hpp"

#ifdef MFEM_USE_MPI
#include <mpi.h>
#endif

#include <functional>
#include <vector>

namespace mfem_mma {

/**
 * Equality-only Method of Moving Asymptotes optimizer.
 *
 * The objective uses the usual convex separable MMA approximation.  Every
 * constraint is kept affine in the subproblem,
 *
 *   h_i(x_k) + grad(h_i(x_k))^T (x - x_k) = 0,
 *
 * so its multiplier is free-sign.  There are no inequality slacks, elastic
 * variables y, scalar z, or penalty coefficients a/c/d.
 *
 * Design-sized vector operations, separable primal minimization, and dot
 * products use the MFEM device selected by the input design vector.  The
 * reduced dense system in the equality multipliers remains on the host; its
 * dimension is the number of equalities rather than the number of controls.
 *
 * Because no feasibility-restoration variables are introduced, the affine
 * equalities must intersect the current move-limit box for the subproblem to
 * be exactly feasible.  If they do not, the reduced Newton solve returns the
 * candidate with the smallest equality residual it found.
 */
class MMAEqualityOptimizer
{
public:
   /** Construct an optimizer for @p n design variables and @p m_equalities
    *  affine-modeled equality constraints. */
   MMAEqualityOptimizer(int n, int m_equalities);

   /** Set the initial asymptote distance (fraction of the move-limit box)
    *  and the shrink/grow factors applied on oscillation/monotonic
    *  progression of consecutive iterates. */
   void SetAsymptotes(mfem::real_t init,
                      mfem::real_t decrease,
                      mfem::real_t increase);

   /**
    * Plain (non-globalized) equality-only MMA update.
    *
    * Like UpdateGCMMA, the objective model always carries a small baseline
    * curvature regularization derived from df0dx and the move-limit box (see
    * ObjectiveRho in the .cpp); this update does not grow that curvature
    * across attempts the way UpdateGCMMA does, it is fixed for the step.
    */
   void Update(mfem::Vector &x,
               const mfem::Vector &df0dx,
               mfem::real_t f0val,
               const mfem::Vector &hval,
               const mfem::Vector *dhdx,
               const mfem::Vector &xmin,
               const mfem::Vector &xmax);

   /**
    * Globally convergent equality-only MMA update.
    *
    * Only the objective MMA model receives a GCMMA curvature parameter.
    * Equality constraints remain affine on every inner attempt.  The
    * callback evaluates the true objective and nonlinear equalities at each
    * candidate.  A candidate is accepted only when the objective model is
    * conservative and it passes a merit/filter reduction test.  Rejected
    * candidates cause objective-curvature growth and move-limit contraction.
    * If all optimization attempts fail, one affine restoration step is tried.
    * MMA history advances only for an accepted optimization or restoration
    * point.
    *
    * @param evaluate Callback that fills true h and f0 at a candidate (see
    *                 GCMMAEvalCallback). In the parallel class it is called
    *                 on every rank and must return globally replicated
    *                 values.
    * @param max_inner Maximum optimization attempts before restoration.
    * @param inner_iterations Optional number of attempted subproblems.
    */
   void UpdateGCMMA(mfem::Vector &x,
                    const mfem::Vector &df0dx,
                    mfem::real_t f0val,
                    const mfem::Vector &hval,
                    const mfem::Vector *dhdx,
                    const mfem::Vector &xmin,
                    const mfem::Vector &xmax,
                    GCMMAEvalCallback evaluate,
                    int max_inner = 15,
                    int *inner_iterations = nullptr);

   /**
    * Restore feasibility of the current affine equality model.
    *
    * Solves the convex projection
    *
    *   min  1/2 sum_j ((x_j-xk_j)/(xmax_j-xmin_j))^2
    *   s.t. h(xk) + J(xk)(x-xk) = 0,
    *        xmin <= x <= xmax.
    *
    * Bound activity is handled inside the reduced free-multiplier Newton
    * solve; clipping is never applied after the solve because that would
    * destroy equality feasibility.  This method does not advance the MMA
    * iteration, modify MMA history, or overwrite the optimization
    * multipliers returned by GetLambda().
    *
    * For nonlinear equalities this is one restoration linearization.  The
    * caller should reevaluate h and dhdx at the returned x and call this
    * method again until the true nonlinear residual is acceptable.
    * Typical nonlinear restoration usage is:
    * @code
    * for (int r=0; r<max_restore; ++r) {
    *    EvaluateEqualities(x, h, dh);
    *    if (h.Norml2() < feasibility_tolerance) { break; }
    *    const real_t affine_residual =
    *       opt.RestoreFeasibility(x, h, dh.data(), xmin, xmax);
    *    if (!std::isfinite(double(affine_residual))) { break; }
    * }
    * @endcode
    *
    * @return Euclidean norm of the remaining affine equality residual.
    */
   mfem::real_t RestoreFeasibility(mfem::Vector &x,
                                   const mfem::Vector &hval,
                                   const mfem::Vector *dhdx,
                                   const mfem::Vector &xmin,
                                   const mfem::Vector &xmax,
                                   int max_iterations = 80);

   /**
    * Bound-projected KKT stationarity residual at (x, lambda).
    *
    * Uses the last-accepted equality multipliers (see GetLambda()) rather
    * than resolving them, so this is a diagnostic of the current iterate,
    * not a fresh optimality solve. The Lagrangian gradient is projected
    * against active bounds before its norm is taken.
    *
    * @param lambda_out Optional buffer of size NumEqualities() that
    *                    receives the multipliers used in the residual.
    * @return Mean squared projected-gradient term plus sum_i h_i(x)^2.
    */
   mfem::real_t KKTresidual(const mfem::Vector &x,
                            const mfem::Vector &df0dx,
                            mfem::real_t f0val,
                            const mfem::Vector &hval,
                            const mfem::Vector *dhdx,
                            const mfem::Vector &xmin,
                            const mfem::Vector &xmax,
                            double *lambda_out = nullptr) const;

   /// Number of accepted MMA iterations so far (alias of NumIterations()).
   int GetIteration() const { return iter_; }
   /// Number of accepted MMA iterations so far.
   int NumIterations() const { return iter_; }
   /// Number of equality constraints this optimizer was constructed with.
   int NumEqualities() const { return m_; }
   /// Whether the most recent Update/UpdateGCMMA call advanced MMA history.
   bool LastStepAccepted() const { return last_step_accepted_; }
   /// Equality multipliers from the most recently accepted step.
   const std::vector<double> &GetLambda() const { return lambda_; }

private:
   /** Lazily allocate p0_/q0_/L_/U_/alpha_/beta_/xo1_/xo2_ on first use so
    *  their device flag matches @p x; a no-op on later calls. */
   void EnsureInitialized_(const mfem::Vector &x);

   int n_ = 0;
   int m_ = 0;
   int iter_ = 0;
   bool last_step_accepted_ = false;
   double asyminit_ = 0.5;
   double asymdec_ = 0.7;
   double asyminc_ = 1.2;
   mfem::Vector p0_, q0_, L_, U_, alpha_, beta_, xo1_, xo2_;
   std::vector<double> lambda_;
};

/**
 * Equality-only Sequential Quadratic (SQ) optimizer.
 *
 * Shares its equality-handling machinery (affine constraint model,
 * free-sign multiplier, reduced Newton solve, restoration) verbatim with
 * MMAEqualityOptimizer -- there are no inequality slacks, elastic
 * variables y, scalar z, or penalty coefficients a/c/d here either, so
 * none of that carries over from SQOptimizer.
 *
 * The only thing that differs from MMAEqualityOptimizer is how the
 * objective's separable model is built:
 *  - the move-limit box is a stateless, symmetric trust region
 *    L = x_k - sigma, U = x_k + sigma with sigma = sigma_scale * range,
 *    instead of history-adaptive asymmetric asymptotes (no oscillation
 *    tracking, no previous-iterate memory);
 *  - the objective curvature comes from a closed-form estimate designed to
 *    already be a valid global upper bound on the true objective over the
 *    trust region (the same formula SQOptimizer uses), rather than a
 *    heuristic starting guess that UpdateGCMMA grows from scratch.
 */
class SQEqualityOptimizer
{
public:
   /** Construct an optimizer for @p n design variables and @p m_equalities
    *  affine-modeled equality constraints. */
   SQEqualityOptimizer(int n, int m_equalities);

   /**
    * Set the trust-region scale: the move limit is sigma_j = s * (xmax_j -
    * xmin_j), applied symmetrically around the current iterate. Unlike
    * MMAEqualityOptimizer::SetAsymptotes(), there is no history-based
    * shrink/grow -- the same scale is used unchanged on every call.
    */
   void SetSigmaScale(mfem::real_t s);

   /**
    * Plain (non-globalized) equality-only SQ update.
    *
    * Builds the symmetric trust region and a closed-form conservative
    * curvature estimate for the objective, then solves for the equality
    * multipliers exactly as MMAEqualityOptimizer::Update() does.
    */
   void Update(mfem::Vector &x,
               const mfem::Vector &df0dx,
               mfem::real_t f0val,
               const mfem::Vector &hval,
               const mfem::Vector *dhdx,
               const mfem::Vector &xmin,
               const mfem::Vector &xmax);

   /**
    * Globally convergent equality-only SQ update.
    *
    * Same globalization loop as MMAEqualityOptimizer::UpdateGCMMA()
    * (objective-curvature growth and move-limit contraction on rejection,
    * one affine restoration attempt if every retry fails), but starting
    * from the SQ trust region and conservative curvature estimate rather
    * than history-adaptive asymptotes.
    *
    * @param evaluate Callback that fills true h and f0 at a candidate (see
    *                 GCMMAEvalCallback). In the parallel class it is called
    *                 on every rank and must return globally replicated
    *                 values.
    * @param max_inner Maximum optimization attempts before restoration.
    * @param inner_iterations Optional number of attempted subproblems.
    */
   void UpdateGCMMA(mfem::Vector &x,
                    const mfem::Vector &df0dx,
                    mfem::real_t f0val,
                    const mfem::Vector &hval,
                    const mfem::Vector *dhdx,
                    const mfem::Vector &xmin,
                    const mfem::Vector &xmax,
                    GCMMAEvalCallback evaluate,
                    int max_inner = 15,
                    int *inner_iterations = nullptr);

   /**
    * Restore feasibility of the current affine equality model; identical
    * contract to MMAEqualityOptimizer::RestoreFeasibility() (same convex
    * projection, same non-mutation of iteration count / history / lambda).
    */
   mfem::real_t RestoreFeasibility(mfem::Vector &x,
                                   const mfem::Vector &hval,
                                   const mfem::Vector *dhdx,
                                   const mfem::Vector &xmin,
                                   const mfem::Vector &xmax,
                                   int max_iterations = 80);

   /**
    * Bound-projected KKT stationarity residual at (x, lambda); identical
    * contract to MMAEqualityOptimizer::KKTresidual().
    *
    * @param lambda_out Optional buffer of size NumEqualities() that
    *                    receives the multipliers used in the residual.
    * @return Mean squared projected-gradient term plus sum_i h_i(x)^2.
    */
   mfem::real_t KKTresidual(const mfem::Vector &x,
                            const mfem::Vector &df0dx,
                            mfem::real_t f0val,
                            const mfem::Vector &hval,
                            const mfem::Vector *dhdx,
                            const mfem::Vector &xmin,
                            const mfem::Vector &xmax,
                            double *lambda_out = nullptr) const;

   /// Number of accepted SQ iterations so far (alias of NumIterations()).
   int GetIteration() const { return iter_; }
   /// Number of accepted SQ iterations so far.
   int NumIterations() const { return iter_; }
   /// Number of equality constraints this optimizer was constructed with.
   int NumEqualities() const { return m_; }
   /// Whether the most recent Update/UpdateGCMMA call advanced SQ history.
   bool LastStepAccepted() const { return last_step_accepted_; }
   /// Equality multipliers from the most recently accepted step.
   const std::vector<double> &GetLambda() const { return lambda_; }

private:
   /** Lazily allocate p0_/q0_/L_/U_/alpha_/beta_ on first use so their
    *  device flag matches @p x; a no-op on later calls. */
   void EnsureInitialized_(const mfem::Vector &x);

   int n_ = 0;
   int m_ = 0;
   int iter_ = 0;
   bool last_step_accepted_ = false;
   double sigma_scale_ = 0.5;
   mfem::Vector p0_, q0_, L_, U_, alpha_, beta_;
   std::vector<double> lambda_;
};

#ifdef MFEM_USE_MPI
/**
 * MPI-distributed equality-only MMA optimizer.
 * The equality values are global replicated scalars; each dhdx entry stores
 * this rank's local portion of the corresponding equality gradient.
 * Local design-sized work runs on the MFEM device selected by x_local.  MPI
 * reductions and the replicated dense equality-multiplier solve remain on
 * the host.
 */
class MMAEqualityOptimizerParallel
{
public:
   /** Construct an optimizer over @p comm for @p n_local design variables
    *  owned by this rank and @p m_equalities globally replicated equality
    *  constraints. Collectively reduces the global design size. */
   MMAEqualityOptimizerParallel(MPI_Comm comm,
                                int n_local,
                                int m_equalities);

   /** Set the initial asymptote distance and shrink/grow factors; see
    *  MMAEqualityOptimizer::SetAsymptotes(). Must be called identically on
    *  every rank. */
   void SetAsymptotes(mfem::real_t init,
                      mfem::real_t decrease,
                      mfem::real_t increase);

   /**
    * Plain (non-globalized) equality-only MMA update, distributed over
    * @p comm. Equivalent to MMAEqualityOptimizer::Update() but design-sized
    * work stays local to this rank while the equality-multiplier solve
    * reduces gradient dot products across ranks. Must be called identically
    * (same @p hval) on every rank.
    */
   void Update(mfem::Vector &x_local,
               const mfem::Vector &df0dx_local,
               mfem::real_t f0val,
               const mfem::Vector &hval,
               const mfem::Vector *dhdx_local,
               const mfem::Vector &xmin_local,
               const mfem::Vector &xmax_local);

   /**
    * MPI-distributed objective-only GCMMA globalization; see
    * MMAEqualityOptimizer::UpdateGCMMA(). The @p evaluate callback runs on
    * every rank and must return identical, globally replicated values.
    */
   void UpdateGCMMA(mfem::Vector &x_local,
                    const mfem::Vector &df0dx_local,
                    mfem::real_t f0val,
                    const mfem::Vector &hval,
                    const mfem::Vector *dhdx_local,
                    const mfem::Vector &xmin_local,
                    const mfem::Vector &xmax_local,
                    GCMMAEvalCallback evaluate,
                    int max_inner = 15,
                    int *inner_iterations = nullptr);

   /**
    * MPI-distributed affine feasibility restoration.
    *
    * This is the distributed counterpart of
    * MMAEqualityOptimizer::RestoreFeasibility().  Equality values are
    * replicated, gradient dot products are reduced across @p comm, and all
    * ranks obtain the same restoration multipliers and residual norm.
    */
   mfem::real_t RestoreFeasibility(mfem::Vector &x_local,
                                   const mfem::Vector &hval,
                                   const mfem::Vector *dhdx_local,
                                   const mfem::Vector &xmin_local,
                                   const mfem::Vector &xmax_local,
                                   int max_iterations = 80);

   /**
    * MPI-distributed bound-projected KKT stationarity residual; see
    * MMAEqualityOptimizer::KKTresidual(). The projected-gradient term is
    * summed across ranks before normalizing by the global design size.
    * Must be called identically (same @p hval) on every rank.
    */
   mfem::real_t KKTresidual(const mfem::Vector &x_local,
                            const mfem::Vector &df0dx_local,
                            mfem::real_t f0val,
                            const mfem::Vector &hval,
                            const mfem::Vector *dhdx_local,
                            const mfem::Vector &xmin_local,
                            const mfem::Vector &xmax_local,
                            double *lambda_out = nullptr) const;

   /// Number of accepted MMA iterations so far (alias of NumIterations()).
   int GetIteration() const { return iter_; }
   /// Number of accepted MMA iterations so far.
   int NumIterations() const { return iter_; }
   /// Number of equality constraints this optimizer was constructed with.
   int NumEqualities() const { return m_; }
   /// Whether the most recent Update/UpdateGCMMA call advanced MMA history.
   bool LastStepAccepted() const { return last_step_accepted_; }
   /// Total design size summed across all ranks.
   long long GlobalSize() const { return n_global_; }
   /// Equality multipliers from the most recently accepted step.
   const std::vector<double> &GetLambda() const { return lambda_; }

private:
   /** Lazily allocate the local-sized work vectors on first use so their
    *  device flag matches @p x_local; a no-op on later calls. */
   void EnsureInitialized_(const mfem::Vector &x_local);

   MPI_Comm comm_;
   long long n_global_ = 0;
   int n_local_ = 0;
   int m_ = 0;
   int iter_ = 0;
   bool last_step_accepted_ = false;
   double asyminit_ = 0.5;
   double asymdec_ = 0.7;
   double asyminc_ = 1.2;
   mfem::Vector p0_, q0_, L_, U_, alpha_, beta_, xo1_, xo2_;
   std::vector<double> lambda_;
};

/**
 * MPI-distributed equality-only Sequential Quadratic (SQ) optimizer.
 * Distributed counterpart of SQEqualityOptimizer, mirroring
 * MMAEqualityOptimizerParallel's relationship to MMAEqualityOptimizer:
 * equality values are global replicated scalars, dhdx entries hold this
 * rank's local gradient portion, and MPI reductions/the replicated dense
 * multiplier solve remain on the host.
 */
class SQEqualityOptimizerParallel
{
public:
   /** Construct an optimizer over @p comm for @p n_local design variables
    *  owned by this rank and @p m_equalities globally replicated equality
    *  constraints. Collectively reduces the global design size. */
   SQEqualityOptimizerParallel(MPI_Comm comm,
                               int n_local,
                               int m_equalities);

   /** Set the trust-region scale; see SQEqualityOptimizer::SetSigmaScale().
    *  Must be called identically on every rank. */
   void SetSigmaScale(mfem::real_t s);

   /**
    * Plain (non-globalized) equality-only SQ update, distributed over
    * @p comm. Equivalent to SQEqualityOptimizer::Update() but design-sized
    * work stays local to this rank while the equality-multiplier solve and
    * the conservative curvature estimate reduce across ranks. Must be
    * called identically (same @p hval) on every rank.
    */
   void Update(mfem::Vector &x_local,
               const mfem::Vector &df0dx_local,
               mfem::real_t f0val,
               const mfem::Vector &hval,
               const mfem::Vector *dhdx_local,
               const mfem::Vector &xmin_local,
               const mfem::Vector &xmax_local);

   /**
    * MPI-distributed objective-only GCMMA globalization; see
    * SQEqualityOptimizer::UpdateGCMMA(). The @p evaluate callback runs on
    * every rank and must return identical, globally replicated values.
    */
   void UpdateGCMMA(mfem::Vector &x_local,
                    const mfem::Vector &df0dx_local,
                    mfem::real_t f0val,
                    const mfem::Vector &hval,
                    const mfem::Vector *dhdx_local,
                    const mfem::Vector &xmin_local,
                    const mfem::Vector &xmax_local,
                    GCMMAEvalCallback evaluate,
                    int max_inner = 15,
                    int *inner_iterations = nullptr);

   /**
    * MPI-distributed affine feasibility restoration; identical contract to
    * MMAEqualityOptimizerParallel::RestoreFeasibility().
    */
   mfem::real_t RestoreFeasibility(mfem::Vector &x_local,
                                   const mfem::Vector &hval,
                                   const mfem::Vector *dhdx_local,
                                   const mfem::Vector &xmin_local,
                                   const mfem::Vector &xmax_local,
                                   int max_iterations = 80);

   /**
    * MPI-distributed bound-projected KKT stationarity residual; see
    * SQEqualityOptimizer::KKTresidual(). The projected-gradient term is
    * summed across ranks before normalizing by the global design size.
    * Must be called identically (same @p hval) on every rank.
    */
   mfem::real_t KKTresidual(const mfem::Vector &x_local,
                            const mfem::Vector &df0dx_local,
                            mfem::real_t f0val,
                            const mfem::Vector &hval,
                            const mfem::Vector *dhdx_local,
                            const mfem::Vector &xmin_local,
                            const mfem::Vector &xmax_local,
                            double *lambda_out = nullptr) const;

   /// Number of accepted SQ iterations so far (alias of NumIterations()).
   int GetIteration() const { return iter_; }
   /// Number of accepted SQ iterations so far.
   int NumIterations() const { return iter_; }
   /// Number of equality constraints this optimizer was constructed with.
   int NumEqualities() const { return m_; }
   /// Whether the most recent Update/UpdateGCMMA call advanced SQ history.
   bool LastStepAccepted() const { return last_step_accepted_; }
   /// Total design size summed across all ranks.
   long long GlobalSize() const { return n_global_; }
   /// Equality multipliers from the most recently accepted step.
   const std::vector<double> &GetLambda() const { return lambda_; }

private:
   /** Lazily allocate the local-sized work vectors on first use so their
    *  device flag matches @p x_local; a no-op on later calls. */
   void EnsureInitialized_(const mfem::Vector &x_local);

   MPI_Comm comm_;
   long long n_global_ = 0;
   int n_local_ = 0;
   int m_ = 0;
   int iter_ = 0;
   bool last_step_accepted_ = false;
   double sigma_scale_ = 0.5;
   mfem::Vector p0_, q0_, L_, U_, alpha_, beta_;
   std::vector<double> lambda_;
};
#endif

} // namespace mfem_mma

#endif
