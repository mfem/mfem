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

#ifndef MFEM_OKINA_HPP
#define MFEM_OKINA_HPP

#ifdef __NVCC__
#include <cuda.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#endif

#include <cmath>
#include <cassert>
#include <iostream>
#include <cstring>

#include <stdarg.h>
#include <signal.h>

#include "/home/camier1/home/stk/stk.hpp"

#include "dbg.hpp"
#include "config.hpp"
#include "memcpy.hpp"
#include "memmng.hpp"
#include "kernels.hpp"

#endif // MFEM_OKINA_HPP
