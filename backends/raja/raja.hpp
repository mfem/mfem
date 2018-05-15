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
//#include "../../general/communication.hpp"
//#include "../../fem/pfespace.hpp"
#include "../../fem/fem.hpp"
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
#include "rdbg.hpp"
#include "rnvvp.hpp"
#include "rconfig.hpp"

// backend/raja/general ********************************************************
#include "rmemcpy.hpp"
#include "rmalloc.hpp"
#include "rarray.hpp"
#include "rtable.hpp"

// backend/raja/linalg *********************************************************
#include "rvector.hpp"

// backend/raja ****************************************************************
#include "operator.hpp"
#include "fespace.hpp"

#endif // MFEM_BACKENDS_RAJA_HPP

