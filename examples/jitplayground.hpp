#ifndef MFEM_EXAMPLES_JITPLAYGROUND_HPP
#define MFEM_EXAMPLES_JITPLAYGROUND_HPP

#include <cstddef>
#include <vector>
#include <iostream>

inline void daxpy(const double *a, std::vector<double> *x,
                  const std::vector<double> *y,
                  size_t n)
{
   for (size_t i = 0; i < n; ++i) { (*x)[i] = *a * (*x)[i] + (*y)[i]; }
}

template<size_t T_N>
void daxpy_wrapper(const double *a, std::vector<double> *x,
                   const std::vector<double> *y)
{
   daxpy(a, x, y, T_N);
}

struct daxpy_op
{
   inline void operator()(double *a, std::vector<double> *x,
                          const std::vector<double> *y) const
   {
      const size_t n = x->size();
      daxpy(a, x, y, n);
   }
};

template <typename return_type, typename... Args>
return_type __enzyme_fwddiff(Args...);

extern int enzyme_const;
extern int enzyme_dup;

template<size_t T_N>
void ddaxpy_wrapper(const double *a, std::vector<double> *x,
                    std::vector<double> *dx, const std::vector<double> *y)
{
   __enzyme_fwddiff<void>(
      (void*)daxpy_wrapper<T_N>,
      enzyme_const, a, enzyme_dup, x, dx, enzyme_const, y);
}

#endif // MFEM_EXAMPLES_JITPLAYGROUND_HPP
