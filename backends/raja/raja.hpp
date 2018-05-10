// Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
// reserved. See files LICENSE and NOTICE for details.
//
// This file is part of CEED, a collection of benchmarks, miniapps, software
// libraries and APIs for efficient high-order finite element and spectral
// element discretizations for exascale applications. For more information and
// source code availability see http://github.com/ceed.
//
// The CEED research is supported by the Exascale Computing Project 17-SC-20-SC,
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.
#ifndef BACKEND_RAJA
#define BACKEND_RAJA

// stdincs *********************************************************************
#include <stdio.h>
#include <stdarg.h>
#include <assert.h>

// __NVCC__ ********************************************************************
#ifdef __NVCC__
#include <cuda.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#define cuCheck checkCudaErrors
#endif

// MFEM/base/backend ***********************************************************
#include "../base/backend.hpp"

// MFEM ************************************************************************
#include "../../config/config.hpp"
#include "../../general/array.hpp"
#include "../../fem/gridfunc.hpp"
#include "../../general/communication.hpp"
#include "../../fem/pfespace.hpp"

// backend/raja ****************************************************************
#include "device.hpp"
#include "memory.hpp"
#include "layout.hpp"
#include "array.hpp"
#include "vector.hpp"
#include "engine.hpp"

// backend/raja/config *********************************************************
#include "config/rdbg.hpp"
#include "config/rnvvp.hpp"
#include "config/rconfig.hpp"

// backend/raja/general ********************************************************
#include "general/rmemcpy.hpp"
#include "general/rmalloc.hpp"
#include "general/rarray.hpp"
#include "general/rtable.hpp"
#include "general/rcommd.hpp"

// backend/raja/linalg *********************************************************
#include "linalg/rvector.hpp"
#include "linalg/roperator.hpp"
#include "linalg/rode.hpp"
#include "linalg/rsolvers.hpp"

// backend/raja/kernels ********************************************************
#include "kernels/include/kernels.hpp"

// backend/raja/fem ************************************************************
#include "fem/rconform.hpp"
#include "fem/rprolong.hpp"
#include "fem/rrestrict.hpp"
#include "fem/rfespace.hpp"
#include "fem/rbilinearform.hpp"
#include "fem/rgridfunc.hpp"
#include "fem/rbilininteg.hpp"

// backend/raja/tests **********************************************************
#include "tests/tests.hpp"

#endif // BACKEND_RAJA

