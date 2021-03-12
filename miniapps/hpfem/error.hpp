#ifndef HPTEST_ERROR_HPP
#define HPTEST_ERROR_HPP

namespace mfem
{

/**
 * @brief Calcuate the square of the H^1_0 error of 'sol' against exact
 *        solution 'exgrad'.
 *
 * @param sol Approximate solution.
 * @param exgrad Gradient of the exact solution.
 * @param elemError Optional array that receives per-element error.
 * @param elemRef Optional array that receives per-element anisotropy flag.
 * @param intOrder Integration rule order to use for evaluating the error.
 * @return
 */
double CalculateH10Error2(GridFunction *sol,
                          VectorCoefficient *exgrad,
                          Array<double> *elemError,
                          Array<int> *elemRef,
                          int intOrder);

} // namespace mfem

#endif // HPTEST_ERROR_HPP
