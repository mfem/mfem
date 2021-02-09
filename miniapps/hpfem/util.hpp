#ifndef HPTEST_UTIL_HPP
#define HPTEST_UTIL_HPP

namespace mfem
{

/** Find the maximum element order of a variable-order solution 'x' and
 *  prolong it to an L2 GridFunction of constant order.
 */
GridFunction* ProlongToMaxOrder(const GridFunction *x);


void VisualizeField(socketstream &sock, GridFunction &gf, const char *title,
                    const char * keys = NULL, int w = 400, int h = 400,
                    int x = 0, int y = 0, bool vec = false);


} // namespace mfem

#endif // HPTEST_UTIL_HPP
