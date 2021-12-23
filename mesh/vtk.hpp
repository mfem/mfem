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
#include "../general/binaryio.hpp"

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

/// Determine the byte order and return either "BigEndian" or "LittleEndian"
const char *VTKByteOrder();

/// @brief Write either ASCII data to the stream or binary data to the buffer
/// depending on the given format.
///
/// If @a format is VTK::ASCII, write the canonical ASCII representation of
/// @a val to the output stream. Otherwise, append its raw binary data to the
/// byte buffer @a buf.
///
/// Note that there are specializations for @a uint8_t (to write as a numeric
/// value rather than a character), and for @a float and @a double values to
/// use the precision specified by @a format.
template <typename T>
void WriteBinaryOrASCII(std::ostream &out, std::vector<char> &buf, const T &val,
                        const char *suffix, VTKFormat format)
{
   if (format == VTKFormat::ASCII) { out << val << suffix; }
   else { bin_io::AppendBytes(buf, val); }
}

template <>
void WriteBinaryOrASCII<uint8_t>(std::ostream &out, std::vector<char> &buf,
                                 const uint8_t &val, const char *suffix,
                                 VTKFormat format);

template <>
void WriteBinaryOrASCII<double>(std::ostream &out, std::vector<char> &buf,
                                const double &val, const char *suffix,
                                VTKFormat format);

template <>
void WriteBinaryOrASCII<float>(std::ostream &out, std::vector<char> &buf,
                               const float &val, const char *suffix,
                               VTKFormat format);

/// @brief Encode in base 64 (and potentially compress) the given data, write it
/// to the output stream (with a header) and clear the buffer.
void WriteBase64WithSizeAndClear(std::ostream &out, std::vector<char> &buf,
                                 int compression_level);

} // namespace mfem

#endif
