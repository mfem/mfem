// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_TENSOR_INTERP_TRAITS
#define MFEM_TENSOR_INTERP_TRAITS

#include "../../tensor.hpp"
#include "../../factories/basis/basis.hpp"
#include "../contractions/contractions.hpp"

namespace mfem
{

enum class InterpAlgo {
   NonTensor,
   Tensor,
   Untensorized,
   Legacy,
   Nedelec,
   NA
};

// Default
template <typename Basis, typename Dofs, typename Enable = void>
struct get_interp_algo_v
{
   static constexpr InterpAlgo value = InterpAlgo::NA;
};

// Non-tensor
template <typename Basis, typename Dofs>
struct get_interp_algo_v<Basis, Dofs,
   std::enable_if_t<
      is_non_tensor_basis<Basis>
   > >
{
   static constexpr InterpAlgo value = InterpAlgo::NonTensor;
};

// Tensor
template <typename Basis, typename Dofs>
struct get_interp_algo_v<Basis, Dofs,
   std::enable_if_t<
      is_tensor_basis<Basis> &&
      !(get_basis_dim<Basis> == 3 &&
        get_tensor_rank<Dofs> == 3 &&
        is_device)
   > >
{
   static constexpr InterpAlgo value = InterpAlgo::Tensor;
};

// Legacy
template <typename Basis, typename Dofs>
struct get_interp_algo_v<Basis, Dofs,
   std::enable_if_t<
      is_tensor_basis<Basis> &&
      (get_basis_dim<Basis> == 3 &&
       get_tensor_rank<Dofs> == 3 &&
       is_device)
   > >
{
   static constexpr InterpAlgo value = InterpAlgo::Legacy;
};

// Nedelec
template <typename Basis, typename Dofs>
struct get_interp_algo_v<Basis, Dofs,
   std::enable_if_t<
      is_nedelec_basis<Basis>
   > >
{
   static constexpr InterpAlgo value = InterpAlgo::Nedelec;
};

// Interpolation algorithm selection trait.
template <typename Basis, typename Dofs>
constexpr InterpAlgo get_interp_algo = get_interp_algo_v<Basis, Dofs>::value;

} // namespace mfem

#endif // MFEM_TENSOR_INTERP_TRAITS
