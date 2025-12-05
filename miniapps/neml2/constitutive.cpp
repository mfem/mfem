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

namespace op {

static neml2::Tensor mfem_to_neml2_tensor(const neml2::TensorOptions &options,
                                          ParameterFunction &pf) {
  // Pointer to host/device data (MFEM owns it)
  real_t *strain_data = pf.ReadWrite();

  // Use torch::from_blob to create a tensor wrapper around the data
  // (without copying!)
  const auto &q_space = pf.GetParameterSpace();
  auto neml2_tensor = neml2::Tensor(
      torch::from_blob(strain_data,
                       {(int64_t)(q_space.GetTrueVSize() / q_space.GetVDim()),
                        (int64_t)q_space.GetVDim()},
                       options),
      /*batch_dim=*/1);
  return neml2_tensor;
}

static void neml2_to_mfem_tensor(const neml2::Tensor &neml2_tensor,
                                 ParameterFunction &pf) {
  // Copy the stress data back to MFEM
  // (either a host-host or device-device copy)
  const auto &q_space = pf.GetParameterSpace();
  auto nbyte = q_space.GetTrueVSize() * sizeof(real_t);
  if (neml2_tensor.options().device().is_cpu())
    std::memcpy(pf.HostWrite(), neml2_tensor.data_ptr<real_t>(), nbyte);
  else if (neml2_tensor.options().device().is_cuda())
    MFEM_GPU_CHECK(cudaMemcpy(pf.ReadWrite(), neml2_tensor.data_ptr<real_t>(),
                              nbyte, cudaMemcpyDeviceToDevice));
  else
    MFEM_ABORT("Unsupported device backend for NEML2");
}

ConstitutiveModel::ConstitutiveModel(std::shared_ptr<neml2::Model> cmodel)
    : _cmodel(cmodel), _strain_name(neml2::VariableName("state", "strain")),
      _stress_name(neml2::VariableName("state", "stress")) {}

void ConstitutiveModel::Mult(ParameterFunction &strain,
                             ParameterFunction &stress) const {
  auto strain_tensor =
      mfem_to_neml2_tensor(_cmodel->variable_options(), strain);
  auto outputs = _cmodel->value({{_strain_name, strain_tensor}});
  auto &stress_tensor = outputs.at(_stress_name);
  neml2_to_mfem_tensor(stress_tensor, stress);
}

void ConstitutiveModel::Tangent(ParameterFunction &strain,
                                neml2::Tensor &tangent) const {
  auto strain_tensor =
      mfem_to_neml2_tensor(_cmodel->variable_options(), strain);
  auto outputs = _cmodel->dvalue({{_strain_name, strain_tensor}});
  tangent = outputs.at(_stress_name).at(_strain_name);
}

void ConstitutiveModel::ApplyTangent(const neml2::Tensor &tangent,
                                     ParameterFunction &dstrain,
                                     ParameterFunction &dstress) const {
  auto dstrain_tensor =
      mfem_to_neml2_tensor(_cmodel->variable_options(), dstrain);
  auto dstress_tensor = neml2::mv(tangent, dstrain_tensor);
  neml2_to_mfem_tensor(dstress_tensor, dstress);
}

} // namespace op
