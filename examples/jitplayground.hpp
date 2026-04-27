#pragma once

#include <cstddef>
#include <vector>
#include <type_traits>

struct daxpy_op
{
   constexpr void operator()(const double *a,
                             std::vector<double> *x,
                             const std::vector<double> *y) const
   {
      if constexpr (std::is_constant_evaluated())
      {

      }
      else
      {

      }

      // $JIT [size_t, n, generic]
      const size_t n = x->size();
      // $JIT [size_t, num_elements, generic]
      const size_t num_elements = x->size();
      for (size_t i = 0; i < n; ++i) { (*x)[i] = *a * (*x)[i] + (*y)[i]; }
   }

};
