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

#ifndef MFEM_BILININTEG_VECTORFEMASS_KERNELS_HPP
#define MFEM_BILININTEG_VECTORFEMASS_KERNELS_HPP

#include "../../config/config.hpp"
#include "../../general/array.hpp"
#include "../../general/forall.hpp"
#include "../../linalg/dtensor.hpp"
#include "../../linalg/vector.hpp"
#include "../bilininteg.hpp"

#include "bilininteg_diffusion_kernels.hpp"
#include "bilininteg_hcurl_kernels.hpp"
#include "bilininteg_hdiv_kernels.hpp"
#include "bilininteg_hcurlhdiv_kernels.hpp"

namespace mfem
{
/// \cond DO_NOT_DOCUMENT
namespace internal
{
namespace hcurlmass
{
constexpr int NBZ3D(int MDQ) { return std::min(128 / MDQ, 64); }
} // namespace hcurlmass
} // namespace internal

template <FiniteElement::DerivType TrialType, FiniteElement::DerivType TestType,
          int DIM, int TrialD1D, int TestD1D, int Q1D>
VectorFEMassIntegrator::ApplyKernelType
VectorFEMassIntegrator::ApplyPAKernels::Kernel()
{
   constexpr bool trial_curl = (TrialType == mfem::FiniteElement::CURL);
   constexpr bool trial_div = (TrialType == mfem::FiniteElement::DIV);
   constexpr bool test_curl = (TestType == mfem::FiniteElement::CURL);
   constexpr bool test_div = (TestType == mfem::FiniteElement::DIV);

   if constexpr (DIM == 3)
   {
      if constexpr (trial_curl && test_curl)
      {
         if (Device::Allows(Backend::DEVICE_MASK))
         {
            // assume TrialD1D == TestD1D
            return internal::SmemPAHcurlMassApply3D<
                   TrialD1D, Q1D,
                   internal::hcurlmass::NBZ3D(std::max(TrialD1D, Q1D))>;
         }
         else
         {
            return internal::PAHcurlMassApply3D;
         }
      }
      else if constexpr (trial_div && test_div)
      {
         if (Device::Allows(Backend::DEVICE_MASK))
         {
            // assumes TrialD1D == TestD1D
            return internal::SmemPAHdivMassApply3D<TrialD1D, Q1D>;
         }
         else
         {
            return internal::PAHdivMassApply3D;
         }
      }
      else if constexpr (trial_curl && test_div)
      {
         return internal::PAHdivHcurlMassApply3D;
      }
      else if constexpr (trial_div && test_curl)
      {
         return internal::PAHcurlHdivMassApply3D;
      }
   }
   else if constexpr (DIM == 2) // 2D
   {
      if constexpr (trial_curl && test_curl)
      {
         return internal::PAHcurlMassApply2D;
      }
      else if constexpr (trial_div && test_div)
      {
         if (Device::Allows(Backend::DEVICE_MASK))
         {
            // assumes TrialD1D == TestD1D
            return internal::SmemPAHdivMassApply2D<TrialD1D, Q1D>;
         }
         else
         {
            return internal::PAHdivMassApply2D;
         }
      }
      else if constexpr (trial_curl && test_div)
      {
         return internal::PAHdivHcurlMassApply2D;
      }
      else if constexpr (trial_div && test_curl)
      {
         return internal::PAHcurlHdivMassApply2D;
      }
   }
   MFEM_ABORT("Unknown kernel.");
}
/// \endcond DO_NOT_DOCUMENT
}

#endif
