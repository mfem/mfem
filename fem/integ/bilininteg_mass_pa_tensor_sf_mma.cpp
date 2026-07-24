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
#include "bilininteg_mass_pa_tensor_sf_mma.hpp"

namespace mfem
{

void MassIntegrator::RegisterTensorSfMmaKernels()
{
   AddTensorSfMmaSpecialization<2,3,4>();
   AddTensorSfMmaSpecialization<2,4,5>();
   AddTensorSfMmaSpecialization<2,5,6>();
   AddTensorSfMmaSpecialization<2,6,7>();
   AddTensorSfMmaSpecialization<2,7,8>();

   AddTensorSfMmaSpecialization<3,3,4>();
   AddTensorSfMmaSpecialization<3,4,5>();
   AddTensorSfMmaSpecialization<3,5,6>();
   AddTensorSfMmaSpecialization<3,6,7>();
   AddTensorSfMmaSpecialization<3,7,8>();
}

} // namespace mfem
