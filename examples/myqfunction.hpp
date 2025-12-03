#include <mfem.hpp>

using namespace mfem;
using mfem::future::tensor;

constexpr int dim = 2;

tensor<real_t, dim, dim> myqfunction0(
   const tensor<real_t, dim, dim> &dvdxi,
   const tensor<real_t, dim, dim> &J,
   const real_t &w)
{
   const auto invJ = inv(J);
   const auto dvdx = dvdxi * invJ;
   const auto test_function_terms = inv(J);
   return dot(dvdx, J) * det(J) * w * test_function_terms;
}
