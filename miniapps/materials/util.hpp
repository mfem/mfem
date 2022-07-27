#ifndef MATERIALS_UTIL_HPP
#define MATERIALS_UTIL_HPP

#include "mfem.hpp"
#include <random>
#include <vector>

namespace mfem {

/// Fills the vector x with random numbers between a and b.
void FillWithRandomNumbers(std::vector<double> &x, double a = 0.0,
                           double b = 1.0);

/// This function creates random rotation matrices (3 x 3) and stores them in
/// the vector. That means, x[0 - 8] is the first rotation matrix. x[9 - 17] is
/// the second and so forth. Size of the vector determines the number of
/// rotation that fit into the vector and should be a multiple of 9.
void FillWithRandomRotations(std::vector<double> &x);

/// Construct the second order tensor (matrix coefficient) Theta from the
/// equation R^T(e1,e2,e3) diag(l1,l2,l3) R (e1,e2,e3).
DenseMatrix ConstructMatrixCoefficient(double l1, double l2, double l3,
                                       double e1, double e2, double e3,
                                       double nu, int dim);

/// Construct the normalization coefficient eta of the white noise right hands
/// side.
double ConstructNormalizationCoefficient(double nu, double l1, double l2,
                                         double l3, int dim);

} // namespace mfem

#endif // MATERIALS_UTIL_HPP
