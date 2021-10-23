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

#ifndef MFEM_NEDELEC_BASIS
#define MFEM_NEDELEC_BASIS

#include "../../utilities/utilities.hpp"
#include "../../tensor.hpp"

namespace mfem
{

template <typename KernelConfig>
struct NedelecBasis
{

// TODO

};

///////////////////////
// Nedelec Basis Traits

// is_nedelec_basis
template <typename Basis>
struct is_nedelec_basis_v
{
   static constexpr bool value = false;
};

template <typename Config>
struct is_nedelec_basis_v<NedelecBasis<Config>>
{
   static constexpr bool value = true;
};

template <typename Basis>
constexpr bool is_nedelec_basis = is_nedelec_basis_v<Basis>::value;

// get_open_basis_dofs
template <typename Basis, typename Enable = void>
struct get_open_basis_dofs_v
{
   static constexpr int value = DynamicMaxSize*DynamicMaxSize;
};

// TODO

template <typename Basis>
constexpr int get_open_basis_dofs = get_open_basis_dofs_v<Basis>::value;

// get_close_basis_dofs
template <typename Basis, typename Enable = void>
struct get_close_basis_dofs_v
{
   static constexpr int value = DynamicMaxSize*DynamicMaxSize;
};

// TODO

template <typename Basis>
constexpr int get_close_basis_dofs = get_close_basis_dofs_v<Basis>::value;

// get_open_basis_capacity
template <typename Basis, typename Enable = void>
struct get_open_basis_capacity_v
{
   static constexpr int value = DynamicMaxSize*DynamicMaxSize;
};

// TODO

template <typename Basis>
constexpr int get_open_basis_capacity = get_open_basis_capacity_v<Basis>::value;

// get_close_basis_capacity
template <typename Basis, typename Enable = void>
struct get_close_basis_capacity_v
{
   static constexpr int value = DynamicMaxSize*DynamicMaxSize;
};

// TODO

template <typename Basis>
constexpr int get_close_basis_capacity = get_close_basis_capacity_v<Basis>::value;

} // mfem namespace

#endif // MFEM_NEDELEC_BASIS
