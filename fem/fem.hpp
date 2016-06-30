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

#ifndef MFEM_FEM
#define MFEM_FEM

#include "../general/array.hpp"
#include "../linalg/linalg.hpp"
#include "../mesh/mesh_headers.hpp"

#include "intrules.hpp"
#include "geom.hpp"
#include "fe.hpp"
#include "fe_coll.hpp"
#include "eltrans.hpp"
#include "coefficient.hpp"
#include "lininteg.hpp"
#include "nonlininteg.hpp"
#include "bilininteg.hpp"
#include "fespace.hpp"
#include "gridfunc.hpp"
#include "linearform.hpp"
#include "nonlinearform.hpp"
#include "bilinearform.hpp"
#include "hybridization.hpp"
#include "datacollection.hpp"
#include "estimators.hpp"

#ifdef MFEM_USE_MPI
#include <mpi.h>
#include "pfespace.hpp"
#include "pgridfunc.hpp"
#include "plinearform.hpp"
#include "pbilinearform.hpp"
#include "pnonlinearform.hpp"
#endif

#endif
