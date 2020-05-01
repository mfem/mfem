#ifndef HPTEST_UTIL_HPP
#define HPTEST_UTIL_HPP

namespace mfem
{

/** Find the maximum element order of a variable-order solution 'x' and
 *  prolong it to an L2 GridFunction of constant order.
 */
GridFunction* ProlongToMaxOrder(const GridFunction *x);


} // namespace mfem

#endif // HPTEST_UTIL_HPP
