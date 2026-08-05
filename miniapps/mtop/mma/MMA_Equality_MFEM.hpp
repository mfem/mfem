#ifndef MFEM_MMA_EQUALITY_MFEM_HPP
#define MFEM_MMA_EQUALITY_MFEM_HPP

#include "mfem.hpp"

#ifdef MFEM_USE_MPI
#include <mpi.h>
#endif

#include <functional>
#include <vector>

namespace mfem_mma {

using MMAEqualityEvalCallback =
   std::function<void(const mfem::Vector &,mfem::real_t &,mfem::Vector &)>;

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
   MMAEqualityOptimizer(int n, int m_equalities);

   void SetAsymptotes(mfem::real_t init,
                      mfem::real_t decrease,
                      mfem::real_t increase);

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
    * @param evaluate Callback that fills true f0 and h at a candidate.
    *                 In the parallel class it is called on every rank and
    *                 must return globally replicated values.
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
                    MMAEqualityEvalCallback evaluate,
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

   mfem::real_t KKTresidual(const mfem::Vector &x,
                            const mfem::Vector &df0dx,
                            mfem::real_t f0val,
                            const mfem::Vector &hval,
                            const mfem::Vector *dhdx,
                            const mfem::Vector &xmin,
                            const mfem::Vector &xmax,
                            double *lambda_out = nullptr) const;

   int GetIteration() const { return iter_; }
   int NumIterations() const { return iter_; }
   int NumEqualities() const { return m_; }
   bool LastStepAccepted() const { return last_step_accepted_; }
   const std::vector<double> &GetLambda() const { return lambda_; }

private:
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
   MMAEqualityOptimizerParallel(MPI_Comm comm,
                                int n_local,
                                int m_equalities);

   void SetAsymptotes(mfem::real_t init,
                      mfem::real_t decrease,
                      mfem::real_t increase);

   void Update(mfem::Vector &x_local,
               const mfem::Vector &df0dx_local,
               mfem::real_t f0val,
               const mfem::Vector &hval,
               const mfem::Vector *dhdx_local,
               const mfem::Vector &xmin_local,
               const mfem::Vector &xmax_local);

   /** MPI-distributed objective-only GCMMA globalization. */
   void UpdateGCMMA(mfem::Vector &x_local,
                    const mfem::Vector &df0dx_local,
                    mfem::real_t f0val,
                    const mfem::Vector &hval,
                    const mfem::Vector *dhdx_local,
                    const mfem::Vector &xmin_local,
                    const mfem::Vector &xmax_local,
                    MMAEqualityEvalCallback evaluate,
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

   mfem::real_t KKTresidual(const mfem::Vector &x_local,
                            const mfem::Vector &df0dx_local,
                            mfem::real_t f0val,
                            const mfem::Vector &hval,
                            const mfem::Vector *dhdx_local,
                            const mfem::Vector &xmin_local,
                            const mfem::Vector &xmax_local,
                            double *lambda_out = nullptr) const;

   int GetIteration() const { return iter_; }
   int NumIterations() const { return iter_; }
   int NumEqualities() const { return m_; }
   bool LastStepAccepted() const { return last_step_accepted_; }
   long long GlobalSize() const { return n_global_; }
   const std::vector<double> &GetLambda() const { return lambda_; }

private:
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
#endif

} // namespace mfem_mma

#endif
