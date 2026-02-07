#pragma once

#include "../fem/quadinterpolator.hpp"
#include "../../util.hpp"

namespace mfem::future
{

template <typename inputs_t>
void interpolate(
   const inputs_t &inputs,
   const std::unordered_map<int, const QuadratureInterpolator *> &qis,
   const IntegrationRule &ir,
   const std::vector<Vector *> &xe,
   BlockVector &xq)
{
   constexpr int ninputs = tuple_size<inputs_t>::value;

   constexpr_for<0, ninputs>([&](auto i)
   {
      const auto input = get<i>(inputs);
      using input_t = std::decay_t<decltype(input)>;

      if constexpr (is_weight_fop<input_t>::value)
      {
         Vector &w = xq.GetBlock(i);

         const int nqp = ir.GetNPoints();
         MFEM_ASSERT(w.Size() % nqp == 0, "weight block has unexpected size");

         const int ne = w.Size() / nqp;
         const real_t *wref = ir.GetWeights().Read();

         for (int e = 0; e < ne; e++)
         {
            std::memcpy(w.GetData() + e*nqp, wref, nqp*sizeof(real_t));
         }
         return;
      }

      auto search = qis.find(input.GetFieldId());
      MFEM_ASSERT(search != qis.end(),
                  "can't find QuadratureInterpolator for given ID " << input.GetFieldId());
      auto qi = search->second;

      qi->SetOutputLayout(QVectorLayout::byVDIM);

      if constexpr (is_value_fop<input_t>::value)
      {
         qi->Values(*xe[i], xq.GetBlock(i));
      }
      else if constexpr (is_gradient_fop<input_t>::value)
      {
         qi->Derivatives(*xe[i], xq.GetBlock(i));
      }
      else
      {
         MFEM_ABORT("default backend doesn't support " << get_type_name<input_t>());
      }
   });
}

template <typename outputs_t>
void integrate(
   const outputs_t &outputs,
   const std::unordered_map<int, const QuadratureInterpolator *> &qis,
   const BlockVector &xq,
   std::vector<Vector *> &ye)
{
   constexpr int noutputs = tuple_size<outputs_t>::value;
   constexpr_for<0, noutputs>([&](auto i)
   {
      const auto output = get<i>(outputs);
      using output_t = std::decay_t<decltype(output)>;

      // Weights are not outputs - they're only inputs
      if constexpr (is_weight_fop<output_t>::value)
      {
         MFEM_ABORT("Weight cannot be an output field");
         return;
      }

      // Check that output vector is allocated
      MFEM_ASSERT(ye[i] != nullptr, "output vector ye[" << i << "] is null");

      auto search = qis.find(output.GetFieldId());
      MFEM_ASSERT(search != qis.end(),
                  "can't find QuadratureInterpolator for given ID " << output.GetFieldId());
      auto qi = search->second;
      qi->SetOutputLayout(QVectorLayout::byVDIM);

      // Initialize output vector to zero (MultTranspose accumulates)
      *ye[i] = 0.0;

      Vector empty; // For unused arguments

      if constexpr (is_value_fop<output_t>::value)
      {
         // Integrate values: Q -> E
         qi->MultTranspose(QuadratureInterpolator::VALUES,
                           xq.GetBlock(i), empty, *ye[i]);
      }
      else if constexpr (is_gradient_fop<output_t>::value)
      {
         MFEM_ABORT("errrr");
         // Integrate gradients: Q -> E
         // qi->MultTranspose(QuadratureInterpolator::DERIVATIVES,
         //                   empty, yq.GetBlock(i), *ye[i]);
      }
      else
      {
         MFEM_ABORT("default backend doesn't support " << get_type_name<output_t>());
      }
   });
}

}
