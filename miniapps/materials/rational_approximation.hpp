//                  MFEM Example 33 - Serial/Parallel Shared Code
//                      (Implementation of the AAA algorithm)
//
//  Here, we implement the triple-A algorithm [1] for the rational approximation
//  of complex-valued functions,
//
//          p(z)/q(z) ≈ f(z).
//
//  In this file, we always assume f(z) = z^{-α}. The triple-A algorithm
//  provides a robust, accurate approximation in rational barycentric form.
//  This representation must be transformed into a partial fraction
//  representation in order to be used to solve a spectral FPDE.
//
//  More specifically, we first expand the numerator in terms of the zeros of
//  the rational approximation,
//
//          p(z) ∝ Π_i (z - z_i),
//
//  and expand the denominator in terms of the poles of the rational
//  approximation,
//
//          q(z) ∝ Π_i (z - p_i).
//
//  We then use these zeros and poles to derive the partial fraction expansion
//
//          f(z) ≈ p(z)/q(z) = Σ_i c_i / (z - p_i).
//
//  [1] Nakatsukasa, Y., Sète, O., & Trefethen, L. N. (2018). The AAA algorithm
//      for rational approximation. SIAM Journal on Scientific Computing, 40(3),
//      A1494-A1522.

#include "mfem.hpp"

namespace mfem {

/** RationalApproximation_AAA: compute the rational approximation (RA) of data
    @a val [in] at the set of points @a pt [in].

    @param[in]  val        Vector of data values
    @param[in]  pt         Vector of sample points
    @param[in]  tol        Relative tolerance
    @param[in]  max_order  Maximum number of terms (order) of the RA
    @param[out] z          Support points of the RA in rational barycentric form
    @param[out] f          Data values at support points @a z
    @param[out] w          Weights of the RA in rational barycentric form

    See pg. A1501 of Nakatsukasa et al. [1]. */
void RationalApproximation_AAA(const Vector &val, const Vector &pt,
                               Array<double> &z, Array<double> &f, Vector &w,
                               double tol, int max_order);


/** ComputePolesAndZeros: compute the @a poles [out] and @a zeros [out] of the
    rational function f(z) = C p(z)/q(z) from its ration barycentric form.

    @param[in]  z      Support points in rational barycentric form
    @param[in]  f      Data values at support points @a z
    @param[in]  w      Weights in rational barycentric form
    @param[out] poles  Array of poles (roots of p(z))
    @param[out] zeros  Array of zeros (roots of q(z))
    @param[out] scale  Scaling constant in f(z) = C p(z)/q(z)

    See pg. A1501 of Nakatsukasa et al. [1]. */
void ComputePolesAndZeros(const Vector &z, const Vector &f, const Vector &w,
                          Array<double> & poles, Array<double> & zeros, double &scale);

/** PartialFractionExpansion: compute the partial fraction expansion of the
    rational function f(z) = Σ_i c_i / (z - p_i) from its @a poles [in] and
    @a zeros [in].

    @param[in]  poles   Array of poles (same as p_i above)
    @param[in]  zeros   Array of zeros
    @param[in]  scale   Scaling constant
    @param[out] coeffs  Coefficients c_i */
void PartialFractionExpansion(double scale, Array<double> & poles,
                              Array<double> & zeros, Array<double> & coeffs);

/** ComputePartialFractionApproximation: compute a rational approximation (RA)
    in partial fraction form, e.g., f(z) ≈ Σ_i c_i / (z - p_i), from sampled
    values of the function f(z) = z^{-a}, 0 < a < 1.

    @param[in]  alpha         Exponent a in f(z) = z^-a
    @param[in] lmax, npoints  f(z) is uniformly sampled @a npoints times in the
                              interval [ 0, @a lmax ]
    @param[in]  tol           Relative tolerance
    @param[in]  max_order     Maximum number of terms (order) of the RA
    @param[out] coeffs        Coefficients c_i
    @param[out] poles         Poles p_i

    NOTES: When MFEM is not built with LAPACK support, only @a alpha = 0.33,
           0.5, and 0.99 are possible. In this case, if @a alpha != 0.33 and
           @a alpha != 0.99, then @a alpha = 0.5 is used by default.

   See pg. A1501 of Nakatsukasa et al. [1]. */
void ComputePartialFractionApproximation(double & alpha,
                                         Array<double> & coeffs, Array<double> & poles,
                                         double lmax = 1000.,
                                         double tol=1e-10, int npoints = 1000,
                                         int max_order = 100);

} // namespace mfem
