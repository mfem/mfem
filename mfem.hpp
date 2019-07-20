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

#ifndef MFEM_HPP
#define MFEM_HPP

#include "config/config.hpp"

#include "general/error.hpp"
#include "general/device.hpp"
#include "general/array.hpp"
#include "general/sets.hpp"
#include "general/hash.hpp"
#include "general/mem_alloc.hpp"
#include "general/sort_pairs.hpp"
#include "general/stable3d.hpp"
#include "general/table.hpp"
#include "general/tic_toc.hpp"
#include "general/isockstream.hpp"
#include "general/osockstream.hpp"
#include "general/socketstream.hpp"
#include "general/optparser.hpp"
#include "general/gzstream.hpp"
#include "general/version.hpp"
#include "general/globals.hpp"
#ifdef MFEM_USE_MPI
#include "general/communication.hpp"
#endif

#include "linalg/linalg.hpp"
#include "mesh/mesh_headers.hpp"
#include "fem/fem.hpp"

#endif
