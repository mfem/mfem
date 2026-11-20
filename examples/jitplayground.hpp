#pragma once

#include <cstddef>
#include <vector>
#include <type_traits>

#include "proteus/JitInterface.h"

struct daxpy_op
{
   void operator()(
      const double *a,
      double *x,
      const double *y,
      const size_t *N) const
   {
      const size_t n = *N;
      auto lam = [=, n = proteus::jit_variable(n)]
                 () __attribute__((annotate("jit")))
      {
         printf("N = %zu\n", n);
         for (size_t i = 0; i < n; ++i)
         {
            printf("x[%zu] = %f, y[%zu] = %f\n", i, x[i], i, y[i]);
            x[i] = *a * x[i] + y[i];
            printf("updated x[%zu] = %f\n", i, x[i]);
         }
      };

      proteus::register_lambda(lam);

      lam();
   }
};

template <typename return_type, typename... Args>
return_type __enzyme_fwddiff(Args...);

extern int enzyme_const;
extern int enzyme_dup;

void daxpy_op_wrapper(const double * Arg0, double * Arg1,
                      const double * Arg2, const size_t *Arg3)
{
   daxpy_op qf;
   qf(Arg0, Arg1, Arg2, Arg3);
}

void daxpy_op_fwddiff(const double * Arg0, double * Arg1,
                      double * dArg1, const double * Arg2, const size_t *Arg3)
{
   __enzyme_fwddiff<void>(
      (void*)daxpy_op_wrapper,
      enzyme_const, Arg0,
      enzyme_dup, Arg1, dArg1,
      enzyme_const, Arg2,
      enzyme_const, Arg3);
}
