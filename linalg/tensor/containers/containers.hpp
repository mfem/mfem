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

#ifndef MFEM_CONTAINERS
#define MFEM_CONTAINERS

/**
 *  The container classes represent different types of linear memories
 *  storing the values of a tensor.
 * */

/// A read/write access pointer container
#include "device_container.hpp"
/// A read only access pointer container
#include "read_container.hpp"
/// A container using the Memory<T> class
#include "memory_container.hpp"
/// A statically sized container allocated on the stack
#include "static_container.hpp"
/// A view container (reference) to another container
#include "view_container.hpp"

#endif // MFEM_CONTAINERS
