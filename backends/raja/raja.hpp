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
#ifndef MFEM_BACKENDS_RAJA_HPP
#define MFEM_BACKENDS_RAJA_HPP

// stdincs *********************************************************************
#include <stdio.h>
#include <stdarg.h>
#include <assert.h>
#include <sys/time.h>

// __NVCC__ ********************************************************************
#ifdef __NVCC__
#include <cuda.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#define cuCheck checkCudaErrors
#endif

// backend/raja/kernels ********************************************************
#include "kernels/include/kernels.hpp"

// MFEM/base/backend ***********************************************************
#include "../base/backend.hpp"

// MFEM ************************************************************************
#include "../../config/config.hpp"
#include "../../general/array.hpp"
#include "../../fem/gridfunc.hpp"
#include "../../general/communication.hpp"
#include "../../fem/pfespace.hpp"
#include "../../fem/fespace.hpp"
#include "../../fem/bilinearform.hpp"

// backend/raja ****************************************************************
#include "memory.hpp"
#include "device.hpp"
#include "engine.hpp"
#include "backend.hpp"
#include "linalg.hpp"
#include "layout.hpp"
#include "array.hpp"
#include "vector.hpp"

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

// backend/raja ****************************************************************
#include "operator.hpp"
#include "fespace.hpp"

// backend/raja/fem ************************************************************
#include "fem/rconform.hpp"
#include "fem/rprolong.hpp"
#include "fem/rrestrict.hpp"
//#include "fem/rfespace.hpp"
#include "fem/rbilinearform.hpp"
#include "fem/rgridfunc.hpp"
#include "fem/rbilininteg.hpp"

// backend/raja/tests **********************************************************
#include "tests/tests.hpp"

#endif // MFEM_BACKENDS_RAJA_HPP

