// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_MIXED_SYSTEM_HPP
#define MFEM_MIXED_SYSTEM_HPP

#include "../../config/config.hpp"

#include "mixed_system_form.hpp"
#include "mixed_system_hybridization.hpp"
#include "mixed_system_reduction.hpp"
#include "bilininteg_hdg.hpp"
#include "estimators_hdg.hpp"

#ifdef MFEM_USE_MPI
#include "pmixed_system_form.hpp"
#endif // MFEM_USE_MPI

#endif // MFEM_MIXED_SYSTEM_HPP
