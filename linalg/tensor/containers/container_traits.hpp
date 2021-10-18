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

#ifndef MFEM_CONTAINER_TRAITS
#define MFEM_CONTAINER_TRAITS

namespace mfem
{

////////////////////
// Container Traits

// get_container_type
template <typename Container>
struct get_container_type_t;

template <typename Container>
using get_container_type = typename get_container_type_t<Container>::type;

// get_container_sizes
template <typename Container>
struct get_container_sizes_t;

template <typename Container>
using get_container_sizes = typename get_container_sizes_t<Container>::type;

// get_unsized_container
template <typename Container>
struct get_unsized_container;

// is_pointer_container
template <typename Container>
struct is_pointer_container_v
{
   static constexpr bool value = false;
};

template <typename Tensor>
constexpr bool is_pointer_container = is_pointer_container_v<Tensor>::value;

} // namespace mfem

#endif // MFEM_CONTAINER_TRAITS
