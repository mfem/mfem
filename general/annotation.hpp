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

#ifndef MFEM_ANNOTATION_HPP
#define MFEM_ANNOTATION_HPP

#include "../config/config.hpp"

#ifdef MFEM_USE_CALIPER

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#define MFEM_PERF_FUNCTION CALI_CXX_MARK_FUNCTION
#define MFEM_PERF_BEGIN(s) CALI_MARK_BEGIN(s)
#define MFEM_PERF_END(s) CALI_MARK_END(s)
#define MFEM_PERF_SCOPE(name) CALI_CXX_MARK_SCOPE(std::string(name).c_str())

#else

#define MFEM_PERF_FUNCTION
#define MFEM_PERF_BEGIN(s)
#define MFEM_PERF_END(s)
#define MFEM_PERF_SCOPE(name)

#endif // MFEM_USE_CALIPER

#endif // MFEM_ANNOTATION_HPP
