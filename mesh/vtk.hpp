// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
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

// Helpers for reading and writing VTK format

// VTK element types defined at: https://git.io/JvZLm
struct VTKGeometry
{
   static const int POINT = 1;
   static const int SEGMENT = 3;
   static const int TRIANGLE = 5;
   static const int SQUARE = 9;
   static const int TETRAHEDRON = 10;
   static const int CUBE = 12;
   static const int PRISM = 13;

   static const int QUADRATIC_SEGMENT = 21;
   static const int QUADRATIC_TRIANGLE = 22;
   static const int BIQUADRATIC_SQUARE = 28;
   static const int QUADRATIC_TETRAHEDRON = 24;
   static const int TRIQUADRATIC_CUBE = 29;
   static const int QUADRATIC_PRISM = 26;
   static const int BIQUADRATIC_QUADRATIC_PRISM = 32;

   static const int LAGRANGE_SEGMENT = 68;
   static const int LAGRANGE_TRIANGLE = 69;
   static const int LAGRANGE_SQUARE = 70;
   static const int LAGRANGE_TETRAHEDRON = 71;
   static const int LAGRANGE_CUBE = 72;
   static const int LAGRANGE_PRISM = 73;

   static const int PrismMap[6];
   static const int *VertexPermutation[Geometry::NUM_GEOMETRIES];

   static const int Map[Geometry::NUM_GEOMETRIES];
   static const int QuadraticMap[Geometry::NUM_GEOMETRIES];
   static const int HighOrderMap[Geometry::NUM_GEOMETRIES];

   static Geometry::Type GetMFEMGeometry(int vtk_geom);
   static bool IsLagrange(int vtk_geom);
   static bool IsQuadratic(int vtk_geom);
   static int GetOrder(int vtk_geom, int npoints);
};

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

int BarycentricToVTKTriangle(int *b, int ref);

const char *VTKByteOrder();

} // namespace mfem

#endif
