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
#ifndef MFEM_BACKENDS_KERNELS_HPP
#define MFEM_BACKENDS_KERNELS_HPP

// *****************************************************************************
#define __LAMBDA__

// stdincs *********************************************************************
#include <math.h>
#include <stdio.h>
#include <stdarg.h>
#include <assert.h>
#include <assert.h>
#include <sys/time.h>
#include <unistd.h>
#include <string.h>

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

// backends ********************************************************************
#include "../base/backend.hpp"

// kernels *********************************************************************
#include "kernels/blas/blas.hpp"
#include "kernels/diffusion/diffusion.hpp"
#include "kernels/force/force.hpp"
#include "kernels/geom/geom.hpp"
#include "kernels/mapping/mapping.hpp"
#include "kernels/mass/mass.hpp"
#include "kernels/quad/quad.hpp"

// engine **********************************************************************
#include "engine/memory.hpp"
#include "engine/device.hpp"
#include "engine/engine.hpp"
#include "engine/backend.hpp"

// config **********************************************************************
#include "config/dbg.hpp"
#include "config/nvvp.hpp"
#include "config/config.hpp"

// general *********************************************************************
#include "linalg/linalg.hpp"
#include "general/layout.hpp"
#include "general/array.hpp"
#include "general/memcpy.hpp"
#include "general/malloc.hpp"
#include "general/karray.hpp"
#include "general/table.hpp"

// linalg **********************************************************************
#include "linalg/vector.hpp"
#include "linalg/kvector.hpp"
#include "linalg/operator.hpp"
#include "linalg/constrained.hpp"
#include "linalg/sparsemat.hpp"
#include "linalg/restrict.hpp"
#include "linalg/prolong.hpp"

// fem *************************************************************************
#include "fem/fespace.hpp"
#include "fem/gridfunc.hpp"
#include "fem/kbilinearform.hpp"
#include "fem/bilinearform.hpp"
#include "fem/coefficient.hpp"
#include "fem/doftoquad.hpp"
#include "fem/geom.hpp"
#include "fem/bilininteg.hpp"
#include "fem/bilinintDiffusion.hpp"
#include "fem/bilinintMass.hpp"
#include "fem/bilinintVMass.hpp"

#endif // MFEM_BACKENDS_KERNELS_HPP

