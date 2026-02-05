#pragma once

#include "../fem/quadinterpolator.hpp"
#include "../../util.hpp"

namespace mfem::future
{

template <typename inputs_t>
void interpolate(
   const inputs_t &inputs,
   const std::unordered_map<int, const QuadratureInterpolator *> &qis,
   const BlockVector &xe,
   BlockVector &xq)
{
   constexpr int ninputs = tuple_size<inputs_t>::value;

   constexpr_for<0, ninputs, 1>([&](auto i)
   {
      const auto input = get<i>(inputs);
      using input_t = std::decay_t<decltype(input)>;

      if constexpr (is_weight_fop<input_t>::value)
      {
         xq.GetBlock(i) = xe.GetBlock(i);
         return;
      }

      auto search = qis.find(input.GetFieldId());
      MFEM_ASSERT(search != qis.end(),
                  "can't find QuadratureInterpolator for given ID " << input.GetFieldId());
      auto qi = search->second;

      qi->SetOutputLayout(QVectorLayout::byVDIM);

      if constexpr (is_value_fop<input_t>::value)
      {
         qi->Values(xe.GetBlock(i), xq.GetBlock(i));
      }
      else if constexpr (is_gradient_fop<input_t>::value)
      {
         qi->Derivatives(xe.GetBlock(i), xq.GetBlock(i));
      }
      else
      {
         MFEM_ABORT("default backend doesn't support " << get_type_name<input_t>());
      }
   });
}

}
