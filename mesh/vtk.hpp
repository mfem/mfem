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

#ifndef MFEM_VTK
#define MFEM_VTK

#include <cstdint>
#include <string>

#include "../fem/geom.hpp"
#include "../general/binaryio.hpp"

namespace mfem
{

// Helpers for reading and writing VTK format

/// @brief Helper class for converting between MFEM and VTK geometry types.
///
/// Note: The VTK element types defined are at: https://git.io/JvZLm
struct VTKGeometry
{
   /// @name VTK geometry types
   ///@{
   static const int POINT = 1;

   /// @name Low-order (linear, straight-sided) VTK geometric types
   ///@{
   static const int SEGMENT = 3;
   static const int TRIANGLE = 5;
   static const int SQUARE = 9;
   static const int TETRAHEDRON = 10;
   static const int CUBE = 12;
   static const int PRISM = 13;
   static const int PYRAMID = 14;
   ///@}

   /// @name Legacy quadratic VTK geometric types
   ///@{
   static const int QUADRATIC_SEGMENT = 21;
   static const int QUADRATIC_TRIANGLE = 22;
   static const int BIQUADRATIC_SQUARE = 28;
   static const int QUADRATIC_TETRAHEDRON = 24;
   static const int TRIQUADRATIC_CUBE = 29;
   static const int QUADRATIC_PRISM = 26;
   static const int BIQUADRATIC_QUADRATIC_PRISM = 32;
   static const int QUADRATIC_PYRAMID = 27;
   ///@}

   /// @name Arbitrary-order VTK geometric types
   ///@{
   static const int LAGRANGE_SEGMENT = 68;
   static const int LAGRANGE_TRIANGLE = 69;
   static const int LAGRANGE_SQUARE = 70;
   static const int LAGRANGE_TETRAHEDRON = 71;
   static const int LAGRANGE_CUBE = 72;
   static const int LAGRANGE_PRISM = 73;
   static const int LAGRANGE_PYRAMID = 74;
   ///@}
   ///@}

   /// Permutation from MFEM's prism ordering to VTK's prism ordering.
   static const int PrismMap[6];

   /// @brief Permutation from MFEM's vertex ordering to VTK's vertex ordering.
   /// @note If the MFEM and VTK orderings are the same, the vertex permutation
   /// will be NULL.
   static const int *VertexPermutation[Geometry::NUM_GEOMETRIES];

   /// Map from MFEM's Geometry::Type to linear VTK geometries.
   static const int Map[Geometry::NUM_GEOMETRIES];
   /// Map from MFEM's Geometry::Type to legacy quadratic VTK geometries/
   static const int QuadraticMap[Geometry::NUM_GEOMETRIES];
   /// Map from MFEM's Geometry::Type to arbitrary-order Lagrange VTK geometries
   static const int HighOrderMap[Geometry::NUM_GEOMETRIES];

   /// Given a VTK geometry type, return the corresponding MFEM Geometry::Type.
   static Geometry::Type GetMFEMGeometry(int vtk_geom);
   /// @brief Does the given VTK geometry type describe an arbitrary-order
   /// Lagrange element?
   static bool IsLagrange(int vtk_geom);
   /// @brief Does the given VTK geometry type describe a legacy quadratic
   /// element?
   static bool IsQuadratic(int vtk_geom);
   /// @brief For the given VTK geometry type and number of points, return the
   /// order of the element.
   static int GetOrder(int vtk_geom, int npoints);
};

/// Data array format for VTK and VTU files.
enum class VTKFormat
{
   /// Data arrays will be written in ASCII format.
   ASCII,
   /// Data arrays will be written in binary format. Floating point numbers will
   /// be output with 64 bits of precision.
   BINARY,
   /// Data arrays will be written in binary format. Floating point numbers will
   /// be output with 32 bits of precision.
   BINARY32
};

/// @brief Create the VTK element connectivity array for a given element
/// geometry and refinement level.
///
/// The output array @a con will be such that, for the @a ith VTK node index,
/// con[i] will contain the index of the corresponding node in MFEM ordering.
void CreateVTKElementConnectivity(Array<int> &con, Geometry::Type geom,
                                  int ref);

/// @brief Outputs encoded binary data in the base 64 format needed by VTK.
///
/// The binary data will be base 64 encoded, and compressed if @a
/// compression_level is not zero. The proper header will be prepended to the
/// data.
void WriteVTKEncodedCompressed(std::ostream &os, const void *bytes,
                               uint32_t nbytes, int compression_level);

/// @brief Return the VTK node index of the barycentric point @a b in a
/// triangle with refinement level @a ref.
///
/// The barycentric index @a b has three components, satisfying b[0] + b[1] +
/// b[2] == ref.
int BarycentricToVTKTriangle(int *b, int ref);

/// Determine the byte order and return either "BigEndian" or "LittleEndian"
const char *VTKByteOrder();

/// @brief Write either ASCII data to the stream or binary data to the buffer
/// depending on the given format.
///
/// If @a format is VTK::ASCII, write the canonical ASCII representation of @a
/// val to the output stream. Subnormal floating point numbers are rounded to
/// zero. Otherwise, append its raw binary data to the byte buffer @a buf.
///
/// Note that there are specializations for @a uint8_t (to write as a numeric
/// value rather than a character), and for @a float and @a double values to use
/// the precision specified by @a format.
template <typename T>
void WriteBinaryOrASCII(std::ostream &os, std::vector<char> &buf, const T &val,
                        const char *suffix, VTKFormat format)
{
   if (format == VTKFormat::ASCII) { os << val << suffix; }
   else { bin_io::AppendBytes(buf, val); }
}

/// @brief Specialization of @ref WriteBinaryOrASCII for @a uint8_t to ensure
/// ASCII output is numeric (rather than interpreting @a val as a character.)
template <>
void WriteBinaryOrASCII<uint8_t>(std::ostream &os, std::vector<char> &buf,
                                 const uint8_t &val, const char *suffix,
                                 VTKFormat format);

/// @brief Specialization of @ref WriteBinaryOrASCII for @a double.
///
/// If @a format is equal to VTKFormat::BINARY32, @a val is converted to a @a
/// float and written as 32 bits. Subnormals are rounded to zero in ASCII
/// output.
template <>
void WriteBinaryOrASCII<double>(std::ostream &os, std::vector<char> &buf,
                                const double &val, const char *suffix,
                                VTKFormat format);

/// @brief Specialization of @ref WriteBinaryOrASCII<T> for @a float.
///
/// If @a format is equal to VTKFormat::BINARY, @a val is converted to a @a
/// double and written as 64 bits. Subnormals are rounded to zero in ASCII
/// output.
template <>
void WriteBinaryOrASCII<float>(std::ostream &os, std::vector<char> &buf,
                               const float &val, const char *suffix,
                               VTKFormat format);

/// @brief Encode in base 64 (and potentially compress) the given data, write it
/// to the output stream (with a header) and clear the buffer.
///
/// @sa WriteVTKEncodedCompressed.
void WriteBase64WithSizeAndClear(std::ostream &os, std::vector<char> &buf,
                                 int compression_level);

/// @brief Returns a string defining the component labels for vector-valued data
/// arrays for use in XML VTU files.
std::string VTKComponentLabels(int vdim);

} // namespace mfem

#endif
