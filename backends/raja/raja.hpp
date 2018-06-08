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

// backend/raja/kernels ********************************************************
#include "kernels/include/kernels.hpp"

// MFEM/base/backend ***********************************************************
#include "../base/backend.hpp"

// MFEM ************************************************************************
#include "../../config/config.hpp"
#include "../../general/array.hpp"
#include "../../fem/gridfunc.hpp"
#include "../../fem/fem.hpp"
#include "../../fem/fespace.hpp"
#include "../../fem/bilinearform.hpp"

// backend/raja/* **************************************************************
#include "engine/memory.hpp"
#include "engine/device.hpp"
#include "engine/engine.hpp"
#include "engine/backend.hpp"
#include "linalg/linalg.hpp"
#include "general/layout.hpp"
#include "general/array.hpp"
#include "linalg/vector.hpp"

// backend/raja/config *********************************************************
#include "config/rdbg.hpp"
#include "config/rnvvp.hpp"
#include "config/rconfig.hpp"

// backend/raja/general ********************************************************
#include "general/rmemcpy.hpp"
#include "general/rmalloc.hpp"
#include "general/rarray.hpp"
#include "general/rtable.hpp"

// backend/raja/linalg *********************************************************
#include "linalg/rvector.hpp"

// backend/raja/fem ************************************************************
#include "linalg/operator.hpp"
#include "fem/fespace.hpp"

#endif // MFEM_BACKENDS_RAJA_HPP

