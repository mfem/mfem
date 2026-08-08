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

#include "../bilininteg.hpp"
#include "bilininteg_vecmass_pa_tensors_mma.hpp"

namespace mfem
{

void VectorMassIntegrator::RegisterTensorsMmaKernels()
{
   // Match scalar mass / stock VectorMass specializations (p = 3..7).
   AddTensorsMmaSpecialization<2,4,5>();
   AddTensorsMmaSpecialization<2,5,6>();
   AddTensorsMmaSpecialization<2,6,7>();
   AddTensorsMmaSpecialization<2,7,8>();
   AddTensorsMmaSpecialization<2,8,9>();

   AddTensorsMmaSpecialization<3,4,5>();
   AddTensorsMmaSpecialization<3,5,6>();
   AddTensorsMmaSpecialization<3,6,7>();
   AddTensorsMmaSpecialization<3,7,8>();
   AddTensorsMmaSpecialization<3,8,9>();
}

VectorMassIntegrator::ApplyTensorsMmaKernelType
VectorMassIntegrator::ApplyTensorsMmaPAKernels::Fallback(int dim, int, int)
{
   if (dim == 2) { return internal::MmaVectorMassApplyTensors2D; }
   if (dim == 3) { return internal::MmaVectorMassApplyTensors3D; }
   MFEM_ABORT("Tensors MMA VectorMass PA is only implemented for dim 2 or 3");
   return nullptr;
}

} // namespace mfem
