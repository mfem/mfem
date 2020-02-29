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

#ifndef MFEM_VTK
#define MFEM_VTK

#include "../fem/geom.hpp"

namespace mfem
{

// Helpers for writing to the VTK format

enum class VTKFormat
{
   ASCII,
   BINARY,
   BINARY32
};

/// Create the VTK element connectivity array for a given element geometry and
/// refinement level. Converts node numbers from MFEM to VTK ordering.
void CreateVTKElementConnectivity(Array<int> &con, Geometry::Type geom,
                                  int ref);

/// Outputs encoded binary data in the format needed by VTK. The binary data
/// will be base 64 encoded, and compressed if @a compression_level is not
/// zero. The proper header will be prepended to the data.
void WriteVTKEncodedCompressed(std::ostream &out, const void *bytes,
                               uint32_t nbytes, int compression_level);

const char *VTKByteOrder();

} // namespace mfem

#endif
