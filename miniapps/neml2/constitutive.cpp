// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "constitutive.hpp"
#include "neml2/tensors/functions/mv.h"
#include <torch/torch.h>

namespace mfem
{

static neml2::Tensor mfem_to_neml2_tensor(const neml2::TensorOptions &options,
                                          ParameterFunction &pf)
{
   // Pointer to host/device data (MFEM owns it)
   real_t *strain_data = pf.ReadWrite();

   // Use torch::from_blob to create a tensor wrapper around the data
   // (without copying!)
   const auto &q_space = pf.GetParameterSpace();
   auto neml2_tensor = neml2::Tensor(torch::from_blob(strain_data,
                                                      {(int64_t)(q_space.GetTrueVSize() /
                                                                 q_space.GetVDim()),
                                                       (int64_t)q_space.GetVDim()},
                                                      options),
                                     /*batch_dim=*/1);
   return neml2_tensor;
}

static void neml2_to_mfem_tensor(const neml2::Tensor &neml2_tensor,
                                 ParameterFunction &pf)
{
   // Copy the stress data back to MFEM
   // (either a host-host or device-device copy)
   const auto &q_space = pf.GetParameterSpace();
   Vector mfem_tensor(neml2_tensor.data_ptr<real_t>(), q_space.GetTrueVSize());
   pf = mfem_tensor;
}

ConstitutiveModel::ConstitutiveModel(std::shared_ptr<neml2::Model> cmodel)
    : _cmodel(cmodel), _time_name(neml2::VariableName("forces", "t")),
      _strain_name(neml2::VariableName("forces", "strain")),
      _stress_name(neml2::VariableName("state", "stress"))
{
}

neml2::ValueMap ConstitutiveModel::MakeInputs(ParameterFunction &strain,
                                              real_t time) const
{
   auto strain_tensor = mfem_to_neml2_tensor(_cmodel->variable_options(),
                                             strain);
   neml2::ValueMap inputs = {{_strain_name, strain_tensor}};
   if (_cmodel->input_variables().count(_time_name))
   {
      inputs[_time_name] = neml2::Scalar(time, _cmodel->variable_options());
   }
   return inputs;
}

void ConstitutiveModel::Mult(ParameterFunction &strain,
                             ParameterFunction &stress, real_t time) const
{
   const auto inputs = MakeInputs(strain, time);
   auto outputs = _cmodel->value(inputs);
   auto &stress_tensor = outputs.at(_stress_name);
   neml2_to_mfem_tensor(stress_tensor, stress);
}

void ConstitutiveModel::Tangent(ParameterFunction &strain,
                                neml2::Tensor &tangent, real_t time) const
{
   const auto inputs = MakeInputs(strain, time);
   auto outputs = _cmodel->dvalue(inputs);
   tangent = outputs.at(_stress_name).at(_strain_name);
}

void ConstitutiveModel::ApplyTangent(const neml2::Tensor &tangent,
                                     ParameterFunction &dstrain,
                                     ParameterFunction &dstress,
                                     real_t time) const
{
   const auto inputs = MakeInputs(dstrain, time);
   auto dstress_tensor = neml2::mv(tangent, inputs.at(_strain_name));
   neml2_to_mfem_tensor(dstress_tensor, dstress);
}

} // namespace mfem
