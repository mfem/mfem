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

#ifndef MFEM_SYCL_HPP
#define MFEM_SYCL_HPP

#include "../config/config.hpp"

#ifdef MFEM_USE_SYCL
#include "mem_manager.hpp"
#include "device.hpp"

namespace mfem
{

/// Get the number of SYCL devices
int SyGetDeviceCount();

/** @brief Function that determines if an SYCL kernel should be used, based on
    the current mfem::Device configuration. */
inline bool DeviceCanUseSycl() { return Device::Allows(Backend::SYCL); }

} // namespace mfem

#endif // MFEM_USE_SYCL

#endif // MFEM_SYCL_HPP
