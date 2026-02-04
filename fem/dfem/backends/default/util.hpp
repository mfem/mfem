#pragma once

#include "../fem/quadinterpolator.hpp"

namespace mfem::future
{

template <typename input_t>
void interpolate(
   const input_t &input,
   const std::vector<const QuadratureInterpolator *> &qis)
{
   std::cout << input.GetFieldId() << "\n";
}

template <
   typename inputs_t,
   typename std::size_t... Is>
void interpolate(
   const inputs_t &inputs,
   const std::vector<const QuadratureInterpolator *> &qis,
   const std::vector<Vector> &ue,
   std::vector<Vector> &uq,
   std::index_sequence<Is...>)
{
   (interpolate(get<Is>(inputs), qis), ...);
}

}
