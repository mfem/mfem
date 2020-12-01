// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

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

extern const int vtk_prism_perm[6];

/// Create the VTK element connectivity array for a given element geometry and
/// refinement level. Converts node numbers from MFEM to VTK ordering.
void CreateVTKElementConnectivity(Array<int> &con, Geometry::Type geom,
                                  int ref);

/// Outputs encoded binary data in the format needed by VTK. The binary data
/// will be base 64 encoded, and compressed if @a compression_level is not
/// zero. The proper header will be prepended to the data.
void WriteVTKEncodedCompressed(std::ostream &out, const void *bytes,
                               uint32_t nbytes, int compression_level);

int BarycentricToVTKTriangle(int *b, int ref);

const char *VTKByteOrder();

} // namespace mfem

#endif
