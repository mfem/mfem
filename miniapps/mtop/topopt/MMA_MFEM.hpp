/**
 * @file MMA_MFEM.hpp
 * @brief Device-aware Method of Moving Asymptotes (MMA/GCMMA) for MFEM.
 *
 * Provides two optimiser classes that share the same dual interior-point
 * mathematics but differ in their parallelism model:
 *
 *   - mfem_mma::MMAOptimizer          – single-process (serial)
 *   - mfem_mma::MMAOptimizerParallel  – distributed MPI, one local
 *                                       mfem::Vector chunk per rank
 *
 * @section problem Problem statement
 * Solves a sequence of convex separable sub-problems of the form (Svanberg
 * 2007, eq. 3.1):
 * @code
 *   min_x  sum_j [ p0j/(Uj-xj) + q0j/(xj-Lj) ]
 *          + a0*z + sum_i [ ci*yi + 0.5*di*yi^2 ]
 *   s.t.   sum_j [ pij/(Uj-xj) + qij/(xj-Lj) ] - ai*z - yi <= bi  (i=1..m)
 *          alpha_j <= x_j <= beta_j,  yi >= 0,  z >= 0
 * @endcode
 * The outer iteration updates asymptotes L, U and rebuilds the p/q
 * coefficients; the inner loop is a dual interior-point Newton method.
 *
 * @section device Device execution
 * All O(n) primal loops execute on whichever device the input Vector @p x
 * lives on, using @c mfem::forall_switch and @c MFEM_HOST_DEVICE lambdas.
 * The O(m²) dual Newton system is always assembled and solved on the CPU
 * (m << n in practice).  Device↔host traffic is limited to m or m² scalars
 * per Newton step, never the full n-vector.
 *
 * Device selection is inherited from @p x:
 * @code
 *   x.UseDevice(true);   // GPU (CUDA or HIP depending on MFEM build)
 *   x.UseDevice(false);  // CPU (default)
 * @endcode
 *
 * @section precision Floating-point precision
 * All quantities that touch @c mfem::Vector data use @c mfem::real_t, which
 * is either @c double or @c float depending on @c MFEM_USE_SINGLE.  The dual
 * Newton system is always computed in @c double regardless of the MFEM build.
 *
 * @section unconstrained Unconstrained problems (m=0)
 * Pass @p m=0, @p fival=nullptr, @p dfidx=nullptr.  The dual IP loop is
 * skipped and the primal solution is computed in closed form entirely on the
 * device.
 *
 * @section redundant Redundant constraints
 * @c detail::SolveDense() attempts an LU factorisation first.  If the dual
 * Hessian is (near-)singular due to linearly dependent constraints, it
 * automatically falls back to a minimum-norm least-squares solve via LAPACK
 * @c dgelsd (divide-and-conquer SVD).
 *
 * @section refs References
 *   - Svanberg K. (2007) MMA and GCMMA – two methods for nonlinear
 *     optimization. KTH Tech Note.
 *   - Aage N. & Lazarov B.S. (2013) Parallel framework for topology
 *     optimization using MMA. Struct Multidisc Optim 47:493–505.
 */

#pragma once

#include <mfem.hpp>
#include <mpi.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <functional>
#include <stdexcept>

namespace mfem_mma {

// ============================================================
/// @namespace mfem_mma::detail
/// @brief Internal helpers shared by serial and parallel optimisers.
///
/// These are not part of the public API and may change without notice.
// ============================================================
namespace detail {

/**
 * @brief Solve the m×m dense linear system  K * x = rhs  in-place.
 *
 * Tries LAPACK @c dgesv (LU) first because it is O(m³) and cheapest.
 * If @c dgesv reports a singular factor (info > 0), which occurs when
 * constraints are linearly dependent, falls back to LAPACK @c dgelsd
 * (divide-and-conquer SVD) to compute the minimum-norm least-squares
 * solution.  Singular values below @f$ 2.2 \times 10^{-16} \cdot m \cdot
 * \sigma_{\max} @f$ are treated as zero.
 *
 * @param[in,out] K    System matrix (m×m, row-major).  Overwritten.
 * @param[in,out] rhs  Right-hand side (length m).  Solution on return.
 * @param[in]     m    System dimension.  m=0 is a no-op; m=1 is solved
 *                     by direct division without calling LAPACK.
 */
void SolveDense(std::vector<double>& K, std::vector<double>& rhs, int m);

/**
 * @brief Dual interior-point solver for the MMA sub-problem.
 *
 * Implements a primal-dual interior-point Newton method on the dual of the
 * MMA sub-problem (Svanberg 2007 §5).  The barrier parameter @f$ \varepsilon
 * @f$ is reduced by a factor of 0.1 each outer loop until convergence.
 *
 * @par Device execution
 * All O(n_loc) loops execute via @c mfem::forall_switch(use_dev, …) so they
 * run on the GPU when @p use_dev is @c true.  Reductions over j (dual
 * gradient and Hessian) use @c mfem::Vector::Sum() and
 * @c mfem::InnerProduct(), which dispatch to cuBLAS/rocBLAS on GPU.
 * The m-vector dual variables and the m×m Newton system live entirely on
 * the host.
 *
 * @par m=0 fast path
 * When @p m=0 the dual IP loop is skipped.  The primal solution is computed
 * analytically as @f$ x_j = \mathrm{clamp}(\sqrt{p_{0j}}\,L_j +
 * \sqrt{q_{0j}}\,U_j)/(\sqrt{p_{0j}}+\sqrt{q_{0j}}),\ [\alpha_j,\beta_j])
 * @f$ entirely on the device.
 *
 * @param comm      MPI communicator.  Pass @c MPI_COMM_SELF for serial use.
 * @param n_loc     Number of local design variables on this rank.
 * @param m         Number of constraints (may be 0).
 * @param use_dev   Execute O(n) loops on the device when @c true.
 * @param L_loc     Lower asymptotes (device pointer, length n_loc).
 * @param U_loc     Upper asymptotes (device pointer, length n_loc).
 * @param alpha_loc Lower move limits (device pointer, length n_loc).
 * @param beta_loc  Upper move limits (device pointer, length n_loc).
 * @param p0_loc    Objective p coefficients (host pointer, length n_loc,
 *                  always @c double).
 * @param q0_loc    Objective q coefficients (host pointer, length n_loc).
 * @param pij_loc   Constraint p coefficients; @c pij_loc[i] is a device
 *                  Vector of length n_loc for constraint i.
 * @param qij_loc   Constraint q coefficients; same layout as @p pij_loc.
 * @param b         Sub-problem RHS constants (host, length m).
 * @param a_pen     Penalty parameter a (host, length m).
 * @param c_pen     Penalty parameter c (host, length m).
 * @param d_pen     Penalty parameter d (host, length m).
 * @param lam       Dual variable λ (host, length m).  Warm-started on entry;
 *                  updated in-place.
 * @param mu        Barrier dual variable μ (host, length m).  Updated in-place.
 * @param y         Slack variable y (host, length m).  Updated in-place.
 * @param z         Scalar dual variable z.  Updated in-place.
 * @param x_loc     Primal solution (device pointer, length n_loc).
 *                  Written on exit.
 */
void SolveDualIP(
    MPI_Comm comm,
    int n_loc, int m,
    bool use_dev,
    const mfem::real_t* L_loc,
    const mfem::real_t* U_loc,
    const mfem::real_t* alpha_loc,
    const mfem::real_t* beta_loc,
    const double*  p0_loc,
    const double*  q0_loc,
    const std::vector<mfem::Vector>& pij_loc,
    const std::vector<mfem::Vector>& qij_loc,
    const std::vector<double>& b,
    const std::vector<double>& a_pen,
    const std::vector<double>& c_pen,
    const std::vector<double>& d_pen,
    std::vector<double>& lam,
    std::vector<double>& mu,
    std::vector<double>& y,
    double& z,
    mfem::real_t* x_loc);

} // namespace detail


// ============================================================
/**
 * @class MMAOptimizer
 * @brief Serial (single-process) MMA/GCMMA optimiser using mfem::Vector.
 *
 * Solves a sequence of nonlinear programming problems of the form
 * @code
 *   min   f0(x)
 *   s.t.  fi(x) <= 0,  i = 1 … m
 *         xmin <= x <= xmax
 * @endcode
 * using the Method of Moving Asymptotes (MMA, Svanberg 1987) with an
 * optional globally convergent variant (GCMMA, Svanberg 2007).
 *
 * Typical usage:
 * @code
 *   MMAOptimizer opt(n, m, x);
 *   for (int k = 0; k < max_iter; ++k) {
 *       ComputeObjectiveAndGradients(x, f0, df0, fi, dfi);
 *       opt.Update(x, df0, f0, fi, dfi.data(), xmin, xmax);
 *       if (opt.KKTresidual(x, df0, f0, fi, dfi.data(),
 *                           xmin, xmax) < tol) break;
 *   }
 * @endcode
 *
 * @note The optimiser is @b not thread-safe.  Each instance maintains mutable
 *       state (asymptotes, dual variables, iteration counter).
 */
// ============================================================

// ── Equality-constraint helpers ──────────────────────────────────────────────
// Equality constraints h_i(x) = 0 are encoded as two inequalities:
//   fival[n_ineq + i]         =  h_i(x) <= 0   (upper half)
//   fival[n_ineq + n_eq + i]  = -h_i(x) <= 0   (lower half)
// where m = n_ineq + 2*n_eq.
//
// Both halves become active (lam > 0) simultaneously at the optimum h_i = 0,
// so this encoding exactly represents the equality without any special dual
// treatment — standard MMA multipliers (always >= 0) are sufficient.
//
// Use MMAOptimizer::WithEqualities() to create the optimiser with the correct
// m, and PackFival()/PackedDfidx to pack arguments at call sites.

/**
 * @brief Pack inequality and equality constraint values into one fival vector.
 *
 * @param fi_ineq  Values of the n_ineq inequality constraints fᵢ(x) ≤ 0.
 * @param h_eq     Values of the n_eq equality constraints hᵢ(x) = 0.
 * @return Vector of length n_ineq + 2*n_eq: [fi_ineq | +h_eq | -h_eq].
 */
inline mfem::Vector PackFival(const mfem::Vector& fi_ineq,
                               const mfem::Vector& h_eq)
{
    const int ni = fi_ineq.Size(), ne = h_eq.Size();
    mfem::Vector out(ni + 2*ne);
    for (int i=0; i<ni; ++i) out(i)              = fi_ineq(i);
    for (int i=0; i<ne; ++i) out(ni + i)          =  h_eq(i);
    for (int i=0; i<ne; ++i) out(ni + ne + i)     = -h_eq(i);
    return out;
}

/**
 * @brief RAII container for packed inequality + equality constraint gradients.
 *
 * Layout: [dfi_ineq | +dh_eq | -dh_eq] (n_ineq + 2*n_eq rows).
 * Pass data() to the dfidx argument of Update/UpdateGCMMA.
 */
struct PackedDfidx {
    std::vector<mfem::Vector> rows;
    PackedDfidx(const mfem::Vector* dfi_ineq, int n_ineq,
                const mfem::Vector* dh_eq,   int n_eq)
        : rows(n_ineq + 2*n_eq)
    {
        for (int i=0; i<n_ineq; ++i)
            rows[i] = dfi_ineq[i];
        for (int i=0; i<n_eq; ++i)
            rows[n_ineq + i] = dh_eq[i];
        for (int i=0; i<n_eq; ++i) {
            rows[n_ineq + n_eq + i]  = dh_eq[i];
            rows[n_ineq + n_eq + i] *= mfem::real_t(-1);
        }
    }
    const mfem::Vector* data() const { return rows.data(); }
    int size() const { return (int)rows.size(); }
};

class MMAOptimizer {
public:
    // ── Constructors ────────────────────────────────────────────────────

    /**
     * @brief Construct with default penalty parameters.
     *
     * Sets a=0, c=max(1000, 10n), d=1 for all constraints.
     * The default @p c ensures @f$ c > \lambda^* @f$ for topology
     * optimisation problems where @f$ \lambda^* \sim n / V_\text{frac}^2 @f$.
     *
     * @param n  Number of design variables.
     * @param m  Number of constraints (≥ 0).  Pass 0 for unconstrained.
     * @param x  Initial design vector (n × 1).  The device flag
     *           (@c x.UseDevice()) is inherited by all internal Vectors.
     */
    MMAOptimizer(int n, int m, const mfem::Vector& x);

    /**
     * @brief Construct with custom penalty parameters (raw arrays).
     *
     * The MMA sub-problem includes elastic variables y_i and a scalar z
     * weighted by penalty parameters a_i, c_i, d_i:
     * @f[
     *   \text{penalty term} = a_0 z + \sum_i \bigl(c_i y_i +
     *   \tfrac{1}{2} d_i y_i^2\bigr)
     * @f]
     * Typical values: a=0, c=max(1000,10n), d=1.
     *
     * @param n  Number of design variables.
     * @param m  Number of constraints.
     * @param x  Initial design vector (n × 1).
     * @param a  Constraint weight on z (length m, usually all zeros).
     * @param c  Elastic penalty weight (length m, must satisfy c > λ*).
     * @param d  Quadratic elastic weight (length m, usually all ones).
     */
    MMAOptimizer(int n, int m, const mfem::Vector& x,
                 const double* a, const double* c, const double* d);

    /**
     * @brief Construct with custom penalty parameters as mfem::Vector.
     *
     * Convenience overload.  Values are converted from @c real_t to
     * @c double internally via @c HostRead().  The Vector may live on either
     * host or device; a single @c HostRead() is called at construction time.
     *
     * @param n  Number of design variables.
     * @param m  Number of constraints.
     * @param x  Initial design vector (n × 1).
     * @param a  Constraint weight on z (length m).
     * @param c  Elastic penalty weight (length m).
     * @param d  Quadratic elastic weight (length m).
     */
    MMAOptimizer(int n, int m, const mfem::Vector& x,
                 const mfem::Vector& a, const mfem::Vector& c,
                 const mfem::Vector& d);

    ~MMAOptimizer() = default;

    // ── Configuration ───────────────────────────────────────────────────

    /**
     * @brief Set asymptote adaptation speeds.
     *
     * Controls how quickly asymptotes L, U expand or contract between
     * outer iterations (Svanberg 2007, eq. 3.11–3.14):
     *   - @p init    : initial half-width as a fraction of (xmax−xmin).
     *                  Used for the first two iterations.  Default 0.5.
     *   - @p decrease: contraction factor when oscillation is detected
     *                  (consecutive sign changes in x−xold).  Default 0.7.
     *   - @p increase: expansion factor when progress is monotone.
     *                  Default 1.2.
     *
     * Must be called before the first @c Update().
     *
     * @param init      Initial asymptote half-width fraction  (0 < init < 1).
     * @param decrease  Contraction factor  (0 < decrease < 1).
     * @param increase  Expansion factor    (> 1).
     */
    void SetAsymptotes(mfem::real_t init, mfem::real_t decrease,
                       mfem::real_t increase);

    // ── Outer iteration ──────────────────────────────────────────────────

    /**
     * @brief Perform one MMA outer iteration.
     *
     * 1. Updates asymptotes L, U from the history of x.
     * 2. Computes move limits alpha, beta.
     * 3. Builds p0, q0, pij, qij, b (sub-problem coefficients) on the device.
     * 4. Solves the dual sub-problem via @c detail::SolveDualIP.
     * 5. Writes the new iterate into @p x.
     *
     * @par Device execution
     * Steps 1–3 run via @c mfem::forall_switch on the same device as @p x.
     * Step 4 runs on the CPU (dual variables are small).
     *
     * @param[in,out] x      Design variables (n × 1).  Updated in-place.
     * @param[in]  df0dx     Objective gradient ∂f₀/∂x (n × 1, same device as x).
     * @param[in]  f0val     Objective value f₀(x).  Accepted but not used in
     *                       the sub-problem (only gradients matter for MMA).
     * @param[in]  fival     Constraint values, length m.
     *                       Layout when using equalities (see WithEqualities()):
     *                         fival[0 .. n_ineq-1] : inequality values fᵢ(x) ≤ 0
     *                         fival[n_ineq .. m-1] : equality values   hᵢ(x) = 0
     *                       Use PackFival() to build this vector conveniently.
     * @param[in]  dfidx     Constraint gradients; dfidx[i] is ∂fᵢ/∂x
     *                       (n × 1, same device as x).
     *                       Pass @c nullptr for unconstrained (m=0).
     * @param[in]  xmin      Lower bounds (n × 1).
     * @param[in]  xmax      Upper bounds (n × 1).
     */
    void Update(mfem::Vector& x,
                const mfem::Vector& df0dx,
                mfem::real_t f0val,
                const mfem::Vector& fival,
                const mfem::Vector* dfidx,
                const mfem::Vector& xmin,
                const mfem::Vector& xmax);

    /**
     * @brief Perform one GCMMA outer iteration (globally convergent MMA).
     *
     * Implements the single-inner-iteration variant of Svanberg (2007) §4.
     * Asymptotes are updated once; @f$ \rho^{(k,0)} @f$ is initialised from
     * gradient magnitudes (@f$ 0.5/n \sum_j |\partial f/\partial x_j|
     * (x_{\max,j}-x_{\min,j}) @f$) and a single sub-problem is solved.  The
     * @f$ \rho @f$ values are re-computed from fresh gradients on every call,
     * providing implicit conservatism adaptation through the outer loop.
     *
     * @par When to use GCMMA vs MMA
     * GCMMA is more robust for non-convex problems or when MMA oscillates.
     * For strictly convex objectives (e.g. compliance minimisation) plain
     * MMA converges faster.
     *
     * @param[in,out] x         Design variables (n × 1).  Updated in-place.
     * @param[in]  df0dx        Objective gradient (n × 1).
     * @param[in]  f0val        Objective value.
     * @param[in]  fival        Constraint values, length m.
     *                          For equality constraints see WithEqualities() and PackFival().
     * @param[in]  dfidx        Constraint gradients (array of m Vectors).
     * @param[in]  xmin         Lower bounds (n × 1).
     * @param[in]  xmax         Upper bounds (n × 1).
     * @param[out] innerIter    If non-null, receives the inner iteration count
     *                          (always 1 in this implementation).
     */
    void UpdateGCMMA(mfem::Vector& x,
                     const mfem::Vector& df0dx,
                     mfem::real_t f0val,
                     const mfem::Vector& fival,
                     const mfem::Vector* dfidx,
                     const mfem::Vector& xmin,
                     const mfem::Vector& xmax,
                     int* innerIter = nullptr);

    /**
     * @brief GCMMA with full inner loop and user-supplied function evaluator.
     *
     * Implements the complete Svanberg (2007) §4 inner loop:
     * after solving the sub-problem to get a candidate x̂, the callback
     * @p eval_fi is called with x̂ to obtain the true constraint values.
     * If any fᵢ(x̂) > f̃ᵢ(x̂) (approximation is not conservative), ρ is
     * increased and the sub-problem is re-solved.  This repeats until the
     * approximation is conservative or @p max_inner steps are exhausted.
     *
     * @param eval_fi  Callable with signature
     *   @code
     *   void eval_fi(const mfem::Vector& x_candidate,
     *                mfem::Vector&       fi_out,
     *                mfem::real_t&       f0_out);
     *   @endcode
     *   Called on every inner iteration with the candidate iterate.
     *   Must fill fi_out[0..m-1] and f0_out with the true function values.
     *   @b Called on all MPI ranks (single-rank for serial class).
     *
     * @param max_inner  Maximum inner iterations.  Default 15.
     *
     * All other parameters are identical to the basic UpdateGCMMA overload.
     */
    using EvalCallback = std::function<void(const mfem::Vector&,
                                             mfem::Vector&,
                                             mfem::real_t&)>;

    void UpdateGCMMA(mfem::Vector& x,
                     const mfem::Vector& df0dx,
                     mfem::real_t f0val,
                     const mfem::Vector& fival,
                     const mfem::Vector* dfidx,
                     const mfem::Vector& xmin,
                     const mfem::Vector& xmax,
                     EvalCallback eval_fi,
                     int  max_inner = 15,
                     int* innerIter = nullptr);

    // ── Convergence ──────────────────────────────────────────────────────

    /**
     * @brief Compute the projected-gradient KKT residual.
     *
     * Returns the quantity
     * @f[
     *   \text{KKT} = \frac{1}{n}\left(
     *     \sum_j \bigl(\mathrm{proj}[\nabla_x \mathcal{L}]_j\bigr)^2 +
     *     \sum_i \bigl(\lambda_i f_i\bigr)^2
     *   \right),
     * @f]
     * where the projected Lagrangian gradient is
     * @f[
     *   \mathrm{proj}[g]_j = \begin{cases}
     *     \min(0,g_j) & x_j \le x_{\min,j} + \tau,\\
     *     \max(0,g_j) & x_j \ge x_{\max,j} - \tau,\\
     *     g_j         & \text{otherwise},
     *   \end{cases}
     * @f]
     * with @f$ \tau = 10^{-3}(x_{\max,j}-x_{\min,j}) @f$.
     *
     * The O(n) projection loop runs on the device; only the scalar result is
     * returned to the host.
     *
     * @param[in]  x           Current design (n × 1).
     * @param[in]  df0dx       Objective gradient (n × 1).
     * @param[in]  f0val       Objective value (unused, kept for API symmetry).
     * @param[in]  fival       Constraint values, length m.
     *                         For equality constraints see WithEqualities() and PackFival().
     * @param[in]  dfidx       Constraint gradients (array of m Vectors).
     * @param[in]  xmin        Lower bounds (n × 1).
     * @param[in]  xmax        Upper bounds (n × 1).
     * @param[out] lambda_out  If non-null, filled with current Lagrange
     *                         multiplier estimates λᵢ (length m, always double).
     * @return  KKT residual (≥ 0).  Convergence is typically declared when
     *          this value falls below ~1e-4.
     */
    mfem::real_t KKTresidual(const mfem::Vector& x,
                             const mfem::Vector& df0dx,
                             mfem::real_t f0val,
                             const mfem::Vector& fival,
                             const mfem::Vector* dfidx,
                             const mfem::Vector& xmin,
                             const mfem::Vector& xmax,
                             double* lambda_out = nullptr) const;

    // ── Accessors ────────────────────────────────────────────────────────

    /// @brief Return the number of completed outer iterations.
    int GetIteration() const { return iter_; }

    /**
     * @brief Return the current Lagrange multiplier estimates (read-only).
     *
     * These are the dual variables λᵢ from the last @c Update() or
     * @c UpdateGCMMA() call.  They are always stored as @c double regardless
     * of the MFEM precision setting.
     *
     * @return  Const reference to the internal λ vector (length m).
     */
    const std::vector<double>& GetLambda() const { return lam_; }

    // ── Unconstrained convenience overloads (m=0) ────────────────────────
    
    // ── Equality-constraint factory ───────────────────────────────────────
    /** Build optimiser with n_ineq inequality + n_eq equality constraints.
     *  m = n_ineq + 2*n_eq internally (each equality encoded as ±h pair).
     *  Use PackFival/PackedDfidx at call sites. */
    static MMAOptimizer WithEqualities(int n, int n_ineq, int n_eq,
                                       const mfem::Vector& x)
    { MMAOptimizer o(n, n_ineq+2*n_eq, x); o.n_eq_=n_eq;
      // Equality slots: set c very large so elastic variable stays zero
      for (int i=n_ineq; i<n_ineq+2*n_eq; ++i) o.c_[i]=1e30;
      return o; }

    static MMAOptimizer WithEqualities(int n, int n_ineq, int n_eq,
                                       const mfem::Vector& x,
                                       const double* a, const double* c,
                                       const double* d)
    { MMAOptimizer o(n, n_ineq+2*n_eq, x, a, c, d); o.n_eq_=n_eq;
      for (int i=n_ineq; i<n_ineq+2*n_eq; ++i) o.c_[i]=1e30;
      return o; }

    int NumEqualities()   const { return n_eq_; }
    int NumInequalities() const { return m_ - 2*n_eq_; }
    int NumConstraints()  const { return m_ - n_eq_; }  // n_ineq + n_eq (user-visible)


    // Drops fival and dfidx — no empty-vector workaround needed at call sites.
    void Update(mfem::Vector& x,
                const mfem::Vector& df0dx,
                mfem::real_t f0val,
                const mfem::Vector& xmin,
                const mfem::Vector& xmax)
    {
        static const mfem::Vector empty;
        Update(x, df0dx, f0val, empty, nullptr, xmin, xmax);
    }

    void UpdateGCMMA(mfem::Vector& x,
                     const mfem::Vector& df0dx,
                     mfem::real_t f0val,
                     const mfem::Vector& xmin,
                     const mfem::Vector& xmax,
                     int* innerIter = nullptr)
    {
        static const mfem::Vector empty;
        UpdateGCMMA(x, df0dx, f0val, empty, nullptr, xmin, xmax, innerIter);
    }

    mfem::real_t KKTresidual(const mfem::Vector& x,
                              const mfem::Vector& df0dx,
                              mfem::real_t f0val,
                              const mfem::Vector& xmin,
                              const mfem::Vector& xmax,
                              double* lambda_out = nullptr) const
    {
        static const mfem::Vector empty;
        return KKTresidual(x, df0dx, f0val, empty, nullptr, xmin, xmax,
                           lambda_out);
    }

private:
    // ── Problem dimensions ───────────────────────────────────────────────

    int n_;     ///< Total number of design variables.
    int m_;     ///< Number of constraints (may be 0).
    int n_eq_ = 0; ///< Number of equality constraints (each encoded as ±h pair, uses 2 slots).
    int iter_;  ///< Completed outer iteration count.

    // ── Asymptote parameters ─────────────────────────────────────────────

    mfem::real_t asyminit_;  ///< Initial asymptote half-width fraction.
    mfem::real_t asymdec_;   ///< Asymptote contraction factor (oscillation).
    mfem::real_t asyminc_;   ///< Asymptote expansion factor (monotone).

    // ── Sub-problem penalty (host, always double) ─────────────────────────

    std::vector<double> a_; ///< Weight on z in constraint i  (length m).
    std::vector<double> c_; ///< Elastic penalty weight c_i   (length m).
    std::vector<double> d_; ///< Quadratic elastic weight d_i (length m).

    // ── Dual variables (host, always double) ─────────────────────────────

    std::vector<double> lam_; ///< Lagrange multipliers λᵢ          (length m).
    std::vector<double> mu_;  ///< Barrier dual variables μᵢ         (length m).
    std::vector<double> y_;   ///< Elastic slack variables yᵢ        (length m).
    double z_;                ///< Scalar dual variable z.

    // ── Design history (device, real_t) ──────────────────────────────────

    mfem::Vector xo1_; ///< Previous iterate x^(k-1) — for asymptote adaptation.
    mfem::Vector xo2_; ///< Two-step-old iterate x^(k-2).

    // ── Asymptotes and move limits (device, real_t) ───────────────────────

    mfem::Vector L_;     ///< Lower asymptotes L_j (length n).
    mfem::Vector U_;     ///< Upper asymptotes U_j (length n).
    mfem::Vector alpha_; ///< Lower move limits α_j = max(xmin, ...) (length n).
    mfem::Vector beta_;  ///< Upper move limits β_j = min(xmax, ...) (length n).

    // ── Sub-problem approximation coefficients (device, real_t) ──────────

    mfem::Vector p0_; ///< Objective p coefficients (length n).
    mfem::Vector q0_; ///< Objective q coefficients (length n).

    /// Constraint p coefficients: pij_[i] has length n, one per constraint.
    std::vector<mfem::Vector> pij_;
    /// Constraint q coefficients: qij_[i] has length n, one per constraint.
    std::vector<mfem::Vector> qij_;

    // ── Sub-problem RHS and GCMMA rho (host, double) ─────────────────────

    /// Sub-problem RHS constants bᵢ = Σⱼ[pij/(U-x)+qij/(x-L)] - fi (length m).
    std::vector<double> b_;
    /// GCMMA conservatism parameters: rho_[0]=objective, rho_[i+1]=constraint i.
    std::vector<double> rho_;

    // ── Private helpers ───────────────────────────────────────────────────

    /**
     * @brief Update asymptotes and build sub-problem coefficients (plain MMA).
     *
     * Called at the start of each @c Update() call.  Updates L_, U_, alpha_,
     * beta_ using the asymptote adaptation rule, then fills p0_, q0_, pij_,
     * qij_, b_ with the standard MMA approximation (rho=0).
     */
    void BuildSubproblem_(const mfem::Vector& x,
                          const mfem::Vector& df0dx,
                          const mfem::Vector& fival,
                          const mfem::Vector* dfidx,
                          const mfem::Vector& xmin,
                          const mfem::Vector& xmax);

    /**
     * @brief Build sub-problem coefficients with explicit rho (GCMMA).
     *
     * Same as @c BuildSubproblem_() but adds the @p rho conservatism term
     * to the p/q coefficients.  Asymptotes must already be current (updated
     * by the caller before the inner loop).
     *
     * @param rho  Conservatism parameters (host, length m+1).
     *             rho[0] is for the objective; rho[i+1] for constraint i.
     */
    void BuildSubproblemRho_(const mfem::Vector& x,
                              const mfem::Vector& df0dx,
                              const mfem::Vector& fival,
                              const mfem::Vector* dfidx,
                              const mfem::Vector& xmin,
                              const mfem::Vector& xmax,
                              const std::vector<double>& rho);
};


// ============================================================
/**
 * @class MMAOptimizerParallel
 * @brief Distributed MPI MMA/GCMMA optimiser using local mfem::Vector chunks.
 *
 * Distributes the n design variables across MPI ranks.  Each rank owns
 * @p n_local contiguous entries and holds local chunks of x, df0dx, dfidx,
 * xmin, xmax.  Constraint values fᵢ and function value f₀ must be globally
 * consistent (identical on all ranks) before each @c Update() call.
 *
 * All O(n_local) loops run on the device of the local x Vector.  The dual
 * gradient and Hessian are assembled via @c MPI_Allreduce of the local
 * contributions; the m×m Newton system is solved identically on every rank
 * (replicated computation).
 *
 * Typical parallel usage:
 * @code
 *   auto [n_local, offset] = Distribute(n_global);
 *   Vector x_loc(n_local);
 *   x_loc.UseDevice(true);            // GPU if available
 *   MMAOptimizerParallel opt(MPI_COMM_WORLD, n_global, n_local, m, x_loc);
 *
 *   for (int k = 0; k < max_iter; ++k) {
 *       // Compute f0, fi on host (global sum); compute gradients on device
 *       double f0, fi[m];
 *       ComputeLocal(x_loc, df0_loc, fi_loc, dfi_loc, f0, fi);
 *
 *       opt.Update(x_loc, df0_loc, f0, fi, dfi_loc, xmin_loc, xmax_loc);
 *       real_t kkt = opt.KKTresidual(...);
 *       if (kkt < tol) break;
 *   }
 * @endcode
 */
// ============================================================
class MMAOptimizerParallel {
public:
    // ── Constructors ────────────────────────────────────────────────────

    /**
     * @brief Construct with default penalty parameters.
     *
     * @param comm      MPI communicator (e.g. @c MPI_COMM_WORLD).
     * @param n_local   Number of design variables owned by this rank.
     * @param m         Number of constraints (may be 0).
     * @param x_local   Local initial design chunk (n_local × 1).
     *                  The device flag is inherited by internal Vectors.
     */
    /// @note n_global is computed automatically as MPI_Allreduce(SUM) of n_local.
    MMAOptimizerParallel(MPI_Comm comm,
                         int n_local, int m,
                         const mfem::Vector& x_local);

    /**
     * @brief Construct with custom penalty parameters (raw double arrays).
     *
     * @param comm      MPI communicator.
     * @param n_local   Local number of design variables on this rank.
     * @param m         Number of constraints.
     * @param x_local   Local initial design chunk (n_local × 1).
     * @param a         Constraint weight on z (length m).
     * @param c         Elastic penalty weight  (length m).
     * @param d         Quadratic elastic weight (length m).
     */
    MMAOptimizerParallel(MPI_Comm comm,
                         int n_local, int m,
                         const mfem::Vector& x_local,
                         const double* a, const double* c, const double* d);

    /**
     * @brief Construct with custom penalty parameters as mfem::Vector.
     *
     * Penalty arrays are converted from @c real_t to @c double via a single
     * @c HostRead() call at construction time.
     *
     * @param comm      MPI communicator.
     * @param n_local   Local number of design variables on this rank.
     * @param m         Number of constraints.
     * @param x_local   Local initial design chunk (n_local × 1).
     * @param a         Constraint weight on z (length m).
     * @param c         Elastic penalty weight  (length m).
     * @param d         Quadratic elastic weight (length m).
     */
    MMAOptimizerParallel(MPI_Comm comm,
                         int n_local, int m,
                         const mfem::Vector& x_local,
                         const mfem::Vector& a, const mfem::Vector& c,
                         const mfem::Vector& d);

    ~MMAOptimizerParallel() = default;

    // ── Configuration ───────────────────────────────────────────────────

    /**
     * @brief Set asymptote adaptation speeds (see MMAOptimizer::SetAsymptotes).
     *
     * @param init      Initial asymptote half-width fraction.
     * @param decrease  Contraction factor on oscillation.
     * @param increase  Expansion factor on monotone progress.
     */
    void SetAsymptotes(mfem::real_t init, mfem::real_t decrease,
                       mfem::real_t increase);

    // ── Outer iteration ──────────────────────────────────────────────────

    /**
     * @brief Perform one parallel MMA outer iteration.
     *
     * Each rank supplies its local chunk of x and gradients.  The dual
     * gradient and Hessian are assembled via @c MPI_Allreduce so that every
     * rank runs the same Newton steps and reaches identical dual variables.
     * The primal update is computed locally on each rank's device.
     *
     * @par Preconditions
     *   - @p f0val and @p fival must be globally consistent (same value on
     *     all ranks) before calling.  The caller is responsible for computing
     *     global sums (e.g. @c MPI_Allreduce) if the objective/constraints
     *     are assembled from local contributions.
     *
     * @param[in,out] x_local       Local design chunk (n_local × 1).
     * @param[in]  df0dx_local      Local objective gradient (n_local × 1).
     * @param[in]  f0val            Global objective value.
     * @param[in]  fival            Global constraint values, length m.
     *                              For equality constraints see WithEqualities() and PackFival().
     * @param[in]  dfidx_local      Local constraint gradients (array of m
     *                              Vectors, each n_local × 1).
     * @param[in]  xmin_local       Local lower bounds (n_local × 1).
     * @param[in]  xmax_local       Local upper bounds (n_local × 1).
     */
    void Update(mfem::Vector& x_local,
                const mfem::Vector& df0dx_local,
                mfem::real_t f0val,
                const mfem::Vector& fival,
                const mfem::Vector* dfidx_local,
                const mfem::Vector& xmin_local,
                const mfem::Vector& xmax_local);

    /**
     * @brief Perform one parallel GCMMA outer iteration.
     *
     * Parallel variant of @c MMAOptimizer::UpdateGCMMA().  The ρ
     * initialisation performs an @c MPI_Allreduce of the local gradient-
     * magnitude sums so all ranks use the same ρ values.
     *
     * @param[in,out] x_local     Local design chunk.
     * @param[in]  df0dx_local    Local objective gradient.
     * @param[in]  f0val          Global objective value.
     * @param[in]  fival          Global constraint values, length m.
     *                            For equality constraints see WithEqualities() and PackFival().
     * @param[in]  dfidx_local    Local constraint gradients.
     * @param[in]  xmin_local     Local lower bounds.
     * @param[in]  xmax_local     Local upper bounds.
     * @param[out] innerIter      Inner iteration count (always 1 here).
     */
    void UpdateGCMMA(mfem::Vector& x_local,
                     const mfem::Vector& df0dx_local,
                     mfem::real_t f0val,
                     const mfem::Vector& fival,
                     const mfem::Vector* dfidx_local,
                     const mfem::Vector& xmin_local,
                     const mfem::Vector& xmax_local,
                     int* innerIter = nullptr);

    /**
     * @brief Parallel GCMMA with full inner loop and user-supplied evaluator.
     *
     * Parallel variant of the callback-based UpdateGCMMA.  The callback
     * receives the LOCAL candidate chunk x_local and must fill fi_out with
     * the GLOBAL constraint values (consistent across all ranks) and
     * f0_out with the global objective value.
     *
     * @param eval_fi  Callable with signature
     *   @code
     *   void eval_fi(const mfem::Vector& x_local_candidate,
     *                mfem::Vector&       fi_out,   // global, length m
     *                mfem::real_t&       f0_out);  // global scalar
     *   @endcode
     *   Called on ALL ranks.  The callback is responsible for any
     *   MPI_Allreduce needed to compute global values.
     *
     * @param max_inner  Maximum inner iterations.  Default 15.
     */
    using EvalCallback = std::function<void(const mfem::Vector&,
                                             mfem::Vector&,
                                             mfem::real_t&)>;

    void UpdateGCMMA(mfem::Vector& x_local,
                     const mfem::Vector& df0dx_local,
                     mfem::real_t f0val,
                     const mfem::Vector& fival,
                     const mfem::Vector* dfidx_local,
                     const mfem::Vector& xmin_local,
                     const mfem::Vector& xmax_local,
                     EvalCallback eval_fi,
                     int  max_inner = 15,
                     int* innerIter = nullptr);

    // ── Convergence ──────────────────────────────────────────────────────

    /**
     * @brief Compute the parallel projected-gradient KKT residual.
     *
     * Identical formula to @c MMAOptimizer::KKTresidual() but the primal
     * term is summed over all ranks via @c MPI_Allreduce before dividing by
     * @p n_global.  Returns the same value on all ranks.
     *
     * @param[in]  x_local         Local design chunk (n_local × 1).
     * @param[in]  df0dx_local     Local objective gradient.
     * @param[in]  f0val           Global objective value (unused).
     * @param[in]  fival           Global constraint values, length m.
     *                             For equality constraints see WithEqualities() and PackFival().
     * @param[in]  dfidx_local     Local constraint gradients.
     * @param[in]  xmin_local      Local lower bounds.
     * @param[in]  xmax_local      Local upper bounds.
     * @param[out] lambda_out      If non-null, filled with λᵢ (length m,
     *                             double).  Identical on all ranks.
     * @return  Global KKT residual (same value on every rank).
     */
    mfem::real_t KKTresidual(const mfem::Vector& x_local,
                             const mfem::Vector& df0dx_local,
                             mfem::real_t f0val,
                             const mfem::Vector& fival,
                             const mfem::Vector* dfidx_local,
                             const mfem::Vector& xmin_local,
                             const mfem::Vector& xmax_local,
                             double* lambda_out = nullptr) const;

    // ── Accessors ────────────────────────────────────────────────────────

    /// @brief Return the number of completed outer iterations.
    int GetIteration() const { return iter_; }

    /**
     * @brief Return the current Lagrange multiplier estimates (read-only).
     *
     * Replicated on every rank; always stored as @c double.
     *
     * @return  Const reference to λ (length m).
     */
    const std::vector<double>& GetLambda() const { return lam_; }

    // ── Unconstrained convenience overloads (m=0) ────────────────────────
    
    // ── Equality-constraint factory ───────────────────────────────────────
    static MMAOptimizerParallel WithEqualities(MPI_Comm comm, int n_local,
                                               int n_ineq, int n_eq,
                                               const mfem::Vector& x_local)
    { MMAOptimizerParallel o(comm,n_local,n_ineq+2*n_eq,x_local);
      o.n_eq_=n_eq;
      for (int i=n_ineq; i<n_ineq+2*n_eq; ++i) o.c_[i]=1e30;
      return o; }

    static MMAOptimizerParallel WithEqualities(MPI_Comm comm, int n_local,
                                               int n_ineq, int n_eq,
                                               const mfem::Vector& x_local,
                                               const double* a, const double* c,
                                               const double* d)
    { MMAOptimizerParallel o(comm,n_local,n_ineq+2*n_eq,x_local,a,c,d);
      o.n_eq_=n_eq;
      for (int i=n_ineq; i<n_ineq+2*n_eq; ++i) o.c_[i]=1e30;
      return o; }

    int NumEqualities()   const { return n_eq_; }
    int NumInequalities() const { return m_ - 2*n_eq_; }
    int NumConstraints()  const { return m_ - n_eq_; }  // n_ineq + n_eq (user-visible)


    void Update(mfem::Vector& x_local,
                const mfem::Vector& df0dx_local,
                mfem::real_t f0val,
                const mfem::Vector& xmin_local,
                const mfem::Vector& xmax_local)
    {
        static const mfem::Vector empty;
        Update(x_local, df0dx_local, f0val, empty, nullptr,
               xmin_local, xmax_local);
    }

    void UpdateGCMMA(mfem::Vector& x_local,
                     const mfem::Vector& df0dx_local,
                     mfem::real_t f0val,
                     const mfem::Vector& xmin_local,
                     const mfem::Vector& xmax_local,
                     int* innerIter = nullptr)
    {
        static const mfem::Vector empty;
        UpdateGCMMA(x_local, df0dx_local, f0val, empty, nullptr,
                    xmin_local, xmax_local, innerIter);
    }

    mfem::real_t KKTresidual(const mfem::Vector& x_local,
                              const mfem::Vector& df0dx_local,
                              mfem::real_t f0val,
                              const mfem::Vector& xmin_local,
                              const mfem::Vector& xmax_local,
                              double* lambda_out = nullptr) const
    {
        static const mfem::Vector empty;
        return KKTresidual(x_local, df0dx_local, f0val, empty, nullptr,
                           xmin_local, xmax_local, lambda_out);
    }

private:
    // ── Communicator and dimensions ───────────────────────────────────────

    MPI_Comm comm_;    ///< MPI communicator.
    int n_eq_ = 0; ///< Number of equality constraints (each encoded as ±h pair, uses 2 slots).
    int n_global_;     ///< Total DOFs across all ranks — computed from n_local_ via MPI_Allreduce.
    int n_local_;      ///< Number of design variables on this rank.
    int m_;            ///< Number of constraints.
    int iter_;         ///< Completed outer iteration count.

    // ── Asymptote parameters ─────────────────────────────────────────────

    mfem::real_t asyminit_; ///< Initial asymptote half-width fraction.
    mfem::real_t asymdec_;  ///< Contraction factor (oscillation detected).
    mfem::real_t asyminc_;  ///< Expansion factor (monotone progress).

    // ── Sub-problem penalty (host, always double) ─────────────────────────

    std::vector<double> a_; ///< Weight on z in constraint i  (length m).
    std::vector<double> c_; ///< Elastic penalty weight c_i   (length m).
    std::vector<double> d_; ///< Quadratic elastic weight d_i (length m).

    // ── Dual variables (host, always double, replicated on all ranks) ─────

    std::vector<double> lam_; ///< Lagrange multipliers λᵢ   (length m).
    std::vector<double> mu_;  ///< Barrier dual variables μᵢ  (length m).
    std::vector<double> y_;   ///< Elastic slack variables yᵢ (length m).
    double z_;                ///< Scalar dual variable z.

    // ── Design history — local chunks (device, real_t) ────────────────────

    mfem::Vector xo1_; ///< Previous local iterate x^(k-1).
    mfem::Vector xo2_; ///< Two-step-old local iterate x^(k-2).

    // ── Asymptotes and move limits — local chunks (device, real_t) ────────

    mfem::Vector L_;     ///< Local lower asymptotes L_j     (length n_local).
    mfem::Vector U_;     ///< Local upper asymptotes U_j     (length n_local).
    mfem::Vector alpha_; ///< Local lower move limits α_j    (length n_local).
    mfem::Vector beta_;  ///< Local upper move limits β_j    (length n_local).

    // ── Sub-problem coefficients — local chunks (device, real_t) ──────────

    mfem::Vector p0_; ///< Local objective p coefficients  (length n_local).
    mfem::Vector q0_; ///< Local objective q coefficients  (length n_local).

    /// Local constraint p coefficients: pij_[i] has length n_local.
    std::vector<mfem::Vector> pij_;
    /// Local constraint q coefficients: qij_[i] has length n_local.
    std::vector<mfem::Vector> qij_;

    // ── Sub-problem RHS and GCMMA rho (host, double, replicated) ──────────

    /// Global sub-problem RHS bᵢ (length m).  Assembled via MPI_Allreduce.
    std::vector<double> b_;
    /// GCMMA conservatism parameters ρ (length m+1): ρ[0]=obj, ρ[i+1]=cstr i.
    std::vector<double> rho_;

    // ── Private helpers ───────────────────────────────────────────────────

    /**
     * @brief Update local asymptotes and build local sub-problem coefficients.
     *
     * Performs MPI_Allreduce of the local b contributions to obtain the
     * global bᵢ.  Subtracts fᵢ on every rank.
     */
    void BuildSubproblem_(const mfem::Vector& x_local,
                          const mfem::Vector& df0dx_local,
                          const mfem::Vector& fival,
                          const mfem::Vector* dfidx_local,
                          const mfem::Vector& xmin_local,
                          const mfem::Vector& xmax_local);

    /**
     * @brief Build local sub-problem coefficients with explicit rho (GCMMA).
     *
     * Asymptotes must already be current.  The rho term is added to each
     * p/q coefficient locally; the global b is assembled via MPI_Allreduce.
     *
     * @param rho  Conservatism parameters (host, length m+1).
     */
    void BuildSubproblemRho_(const mfem::Vector& x_local,
                              const mfem::Vector& df0dx_local,
                              const mfem::Vector& fival,
                              const mfem::Vector* dfidx_local,
                              const mfem::Vector& xmin_local,
                              const mfem::Vector& xmax_local,
                              const std::vector<double>& rho);
};

} // namespace mfem_mma
