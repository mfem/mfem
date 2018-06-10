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
#include <math.h>
#include <stdio.h>
#include <stdarg.h>
#include <assert.h>
#include <assert.h>
#include <sys/time.h>
#include <unistd.h>

// MPI *************************************************************************
#ifdef MFEM_USE_MPI
#include <mpi.h>
#include <mpi-ext.h>
#endif

// MFEM ************************************************************************
#include "../../config/config.hpp"
#include "../../general/array.hpp"
#include "../../fem/gridfunc.hpp"
#include "../../fem/fem.hpp"
#include "../../fem/fespace.hpp"
#include "../../fem/bilinearform.hpp"

// RAJA/kernels ****************************************************************
#include "kernels/include/kernels.hpp"

// MFEM/backends ***************************************************************
#include "../base/backend.hpp"

// RAJA/engine *****************************************************************
#include "engine/memory.hpp"
#include "engine/device.hpp"
#include "engine/engine.hpp"
#include "engine/backend.hpp"

// RAJA/config *****************************************************************
#include "config/rdbg.hpp"
#include "config/rnvvp.hpp"
#include "config/rconfig.hpp"

// RAJA/general ****************************************************************
#include "linalg/linalg.hpp"
#include "general/layout.hpp"
#include "general/array.hpp"
#include "general/rmemcpy.hpp"
#include "general/rmalloc.hpp"
#include "general/rarray.hpp"
#include "general/rtable.hpp"

// RAJA/linalg *****************************************************************
#include "linalg/vector.hpp"
#include "linalg/rvector.hpp"
#include "linalg/operator.hpp"
#include "linalg/sparsemat.hpp"

// RAJA/fem ********************************************************************
#include "fem/fespace.hpp"
#include "fem/gridfunc.hpp"
#include "fem/bilinearform.hpp"
#include "fem/coefficient.hpp"
#include "fem/bilininteg.hpp"
#include "fem/bilinIntDiffusion.hpp"
#include "fem/bilinIntMass.hpp"
#include "fem/bilinIntVMass.hpp"
#include "fem/restrict.hpp"
#include "fem/prolong.hpp"


#endif // MFEM_BACKENDS_RAJA_HPP

