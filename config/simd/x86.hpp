// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#ifndef MFEM_TEMPLATE_CONFIG_X86INTRIN_HPP
#define MFEM_TEMPLATE_CONFIG_X86INTRIN_HPP

#include "x86intrin.h"

template <typename, int, int> struct AutoSIMD;

#include "m64.hpp"

#include "m128.hpp"

#include "m256.hpp"

#include "m512.hpp"

#endif // MFEM_TEMPLATE_CONFIG_X86INTRIN_HPP
