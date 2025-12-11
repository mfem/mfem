// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_HPP
#define MFEM_HPP

#include "config/config.hpp"

#include "general/error.hpp"
#include "general/device.hpp"
#include "general/forall.hpp"
#include "general/array.hpp"
#include "general/arrays_by_name.hpp"
#include "general/sets.hpp"
#include "general/hash.hpp"
#include "general/mem_alloc.hpp"
#include "general/sort_pairs.hpp"
#include "general/stable3d.hpp"
#include "general/table.hpp"
#include "general/tic_toc.hpp"
#include "general/annotation.hpp"
#ifdef MFEM_USE_ADIOS2
#include "general/adios2stream.hpp"
#endif // MFEM_USE_ADIOS2
#include "general/isockstream.hpp"
#include "general/osockstream.hpp"
#include "general/socketstream.hpp"
#include "general/optparser.hpp"
#include "general/zstr.hpp"
#include "general/version.hpp"
#include "general/globals.hpp"
#include "general/kdtree.hpp"
#include "general/enzyme.hpp"
#ifdef MFEM_USE_MPI
#include "general/communication.hpp"
#endif // MFEM_USE_MPI

#include "linalg/linalg.hpp"

#include "mesh/mesh_headers.hpp"

#include "fem/fem.hpp"
#ifdef MFEM_USE_MOONOLITH
#include "fem/moonolith/transfer.hpp"
#endif // MFEM_USE_MOONOLITH

#endif
