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

#include "vtk.hpp"
#include "../general/binaryio.hpp"
#ifdef MFEM_USE_ZLIB
#include <zlib.h>
#endif

namespace mfem
{

const int VTKGeometry::Map[Geometry::NUM_GEOMETRIES] =
{
   POINT, SEGMENT, TRIANGLE, SQUARE, TETRAHEDRON, CUBE, PRISM, PYRAMID
};

const int VTKGeometry::QuadraticMap[Geometry::NUM_GEOMETRIES] =
{
   POINT, QUADRATIC_SEGMENT, QUADRATIC_TRIANGLE, BIQUADRATIC_SQUARE,
   QUADRATIC_TETRAHEDRON, TRIQUADRATIC_CUBE, BIQUADRATIC_QUADRATIC_PRISM,
   QUADRATIC_PYRAMID
};

const int VTKGeometry::HighOrderMap[Geometry::NUM_GEOMETRIES] =
{
   POINT, LAGRANGE_SEGMENT, LAGRANGE_TRIANGLE, LAGRANGE_SQUARE,
   LAGRANGE_TETRAHEDRON, LAGRANGE_CUBE, LAGRANGE_PRISM, LAGRANGE_PYRAMID
};

const int VTKGeometry::PrismMap[6] = {0, 2, 1, 3, 5, 4};

const int *VTKGeometry::VertexPermutation[Geometry::NUM_GEOMETRIES] =
{
   NULL, NULL, NULL, NULL, NULL, NULL, VTKGeometry::PrismMap, NULL
};

Geometry::Type VTKGeometry::GetMFEMGeometry(int vtk_geom)
{
   switch (vtk_geom)
   {
      case POINT:
         return Geometry::POINT;
      case SEGMENT:
      case QUADRATIC_SEGMENT:
      case LAGRANGE_SEGMENT:
         return Geometry::SEGMENT;
      case TRIANGLE:
      case QUADRATIC_TRIANGLE:
      case LAGRANGE_TRIANGLE:
         return Geometry::TRIANGLE;
      case SQUARE:
      case BIQUADRATIC_SQUARE:
      case LAGRANGE_SQUARE:
         return Geometry::SQUARE;
      case TETRAHEDRON:
      case QUADRATIC_TETRAHEDRON:
      case LAGRANGE_TETRAHEDRON:
         return Geometry::TETRAHEDRON;
      case CUBE:
      case TRIQUADRATIC_CUBE:
      case LAGRANGE_CUBE:
         return Geometry::CUBE;
      case PRISM:
      case BIQUADRATIC_QUADRATIC_PRISM:
      case LAGRANGE_PRISM:
         return Geometry::PRISM;
      case PYRAMID:
      case QUADRATIC_PYRAMID:
      case LAGRANGE_PYRAMID:
         return Geometry::PYRAMID;
      default:
         return Geometry::INVALID;
   }
}

bool VTKGeometry::IsLagrange(int vtk_geom)
{
   return vtk_geom >= LAGRANGE_SEGMENT && vtk_geom <= LAGRANGE_PYRAMID;
}

bool VTKGeometry::IsQuadratic(int vtk_geom)
{
   return vtk_geom >= QUADRATIC_SEGMENT
          && vtk_geom <= BIQUADRATIC_QUADRATIC_PRISM;
}

int VTKGeometry::GetOrder(int vtk_geom, int npoints)
{
   if (IsQuadratic(vtk_geom))
   {
      return 2;
   }
   else if (IsLagrange(vtk_geom))
   {
      switch (vtk_geom)
      {
         case LAGRANGE_SEGMENT:
            return npoints - 1;
         case LAGRANGE_TRIANGLE:
            return static_cast<int>(std::sqrt(8*npoints + 1) - 3)/2;
         case LAGRANGE_SQUARE:
            return static_cast<int>(std::round(std::sqrt(npoints))) - 1;
         case LAGRANGE_TETRAHEDRON:
            switch (npoints)
            {
               // Note that for given order, npoints is given by
               // npoints_order = (order + 1)*(order + 2)*(order + 3)/6,
               case 4: return 1;
               case 10: return 2;
               case 20: return 3;
               case 35: return 4;
               case 56: return 5;
               case 84: return 6;
               case 120: return 7;
               case 165: return 8;
               case 220: return 9;
               case 286: return 10;
               default:
               {
                  constexpr int max_order = 20;
                  int order = 11, npoints_order;
                  for (; order<max_order; ++order)
                  {
                     npoints_order = (order + 1)*(order + 2)*(order + 3)/6;
                     if (npoints_order == npoints) { break; }
                  }
                  MFEM_VERIFY(npoints == npoints_order, "");
                  return order;
               }
            }
         case LAGRANGE_CUBE:
            return static_cast<int>(std::round(std::cbrt(npoints))) - 1;
         case LAGRANGE_PRISM:
         {
            const double n = npoints;
            static const double third = 1.0/3.0;
            static const double ninth = 1.0/9.0;
            static const double twentyseventh = 1.0/27.0;
            const double term =
               std::cbrt(third*sqrt(third)*sqrt((27.0*n - 2.0)*n) + n
                         - twentyseventh);
            return static_cast<int>(std::round(term + ninth / term - 4*third));
         }
         case LAGRANGE_PYRAMID:
            MFEM_ABORT("Lagrange pyramids not currently supported in VTK.");
            return 0;
      }
   }
   return 1;
}

int BarycentricToVTKTriangle(int *b, int ref)
{
   // Cf. https://git.io/JvW8f
   int max = ref;
   int min = 0;
   int bmin = std::min(std::min(b[0], b[1]), b[2]);
   int idx = 0;

   // scope into the correct triangle
   while (bmin > min)
   {
      idx += 3*ref;
      max -= 2;
      ++min;
      ref -= 3;
   }
   for (int d=0; d<3; ++d)
   {
      if (b[(d+2)%3] == max)
      {
         // we are on a vertex
         return idx;
      }
      ++idx;
   }
   for (int d=0; d<3; ++d)
   {
      if (b[(d+1)%3] == min)
      {
         // we are on an edge
         return idx + b[d] - (min + 1);
      }
      idx += max - (min + 1);
   }
   return idx;
}

int BarycentricToVTKTetra(int *b, int ref)
{
   // Cf. https://git.io/JvW8c
   int idx = 0;

   int max = ref;
   int min = 0;

   int bmin = std::min(std::min(std::min(b[0], b[1]), b[2]), b[3]);

   // scope into the correct tetra
   while (bmin > min)
   {
      idx += 2*(ref*ref + 1);
      max -= 3;
      min++;
      ref -= 4;
   }

   // When a linearized tetra vertex is cast into barycentric coordinates, one of
   // its coordinates is maximal and the other three are minimal. These are the
   // indices of the maximal barycentric coordinate for each vertex.
   static const int VertexMaxCoords[4] = {3,0,1,2};
   // Each linearized tetra edge holds two barycentric tetra coordinates constant
   // and varies the other two. These are the coordinates that are held constant
   // for each edge.
   static const int EdgeMinCoords[6][2] = {{1,2},{2,3},{0,2}, {0,1},{1,3},{0,3}};
   // The coordinate that increments when traversing an edge (i.e. the coordinate
   // of the nonzero component of the second vertex of the edge).
   static const int EdgeCountingCoord[6] = {0,1,3,2,2,2};
   // When describing a linearized tetra face, there is a mapping between the
   // four-component barycentric tetra system and the three-component barycentric
   // triangle system. These are the constant indices within the four-component
   // system for each face (e.g. face 0 holds barycentric tetra coordinate 1
   // constant).
   static const int FaceMinCoord[4] = {1,3,0,2};
   // When describing a linearized tetra face, there is a mapping between the
   // four-component barycentric tetra system and the three-component barycentric
   // triangle system. These are the relevant indices within the four-component
   // system for each face (e.g. face 0 varies across the barycentric tetra
   // coordinates 0, 2 and 3).
   static const int FaceBCoords[4][3] = {{0,2,3}, {2,0,1}, {2,1,3}, {1,0,3}};


   for (int vertex = 0; vertex < 4; vertex++)
   {
      if (b[VertexMaxCoords[vertex]] == max)
      {
         // we are on a vertex
         return idx;
      }
      idx++;
   }

   for (int edge = 0; edge < 6; edge++)
   {
      if (b[EdgeMinCoords[edge][0]] == min && b[EdgeMinCoords[edge][1]] == min)
      {
         // we are on an edge
         return idx + b[EdgeCountingCoord[edge]] - (min + 1);
      }
      idx += max - (min + 1);
   }

   for (int face = 0; face < 4; face++)
   {
      if (b[FaceMinCoord[face]] == min)
      {
         // we are on a face
         int projectedb[3];
         for (int i = 0; i < 3; i++)
         {
            projectedb[i] = b[FaceBCoords[face][i]] - min;
         }
         // we must subtract the indices of the face's vertices and edges, which
         // total to 3*ref
         return (idx + BarycentricToVTKTriangle(projectedb, ref) - 3*ref);
      }
      idx += (ref+1)*(ref+2)/2 - 3*ref;
   }
   return idx;
}

int VTKTriangleDOFOffset(int ref, int i, int j)
{
   return i + ref*(j - 1) - (j*(j + 1))/2;
}

int CartesianToVTKPrism(int i, int j, int k, int ref)
{
   // Cf. https://t.ly/3Yl9m
   int om1 = ref - 1;
   int ibdr = (i == 0);
   int jbdr = (j == 0);
   int ijbdr = (i + j == ref);
   int kbdr = (k == 0 || k == ref);
   // How many boundaries do we lie on at once?
   int nbdr = ibdr + jbdr + ijbdr + kbdr;

   // Return an invalid index given invalid coordinates
   if (i < 0 || i > ref || j < 0 || j > ref || i + j > ref || k < 0 || k > ref)
   {
      MFEM_ABORT("Invalid index")
   }

   if (nbdr == 3) // Vertex DOF
   {
      // ijk is a corner node. Return the proper index (somewhere in [0,5]):
      return (ibdr && jbdr ? 0 : (jbdr && ijbdr ? 1 : 2)) + (k ? 3 : 0);
   }

   int offset = 6;
   if (nbdr == 2) // Edge DOF
   {
      if (!kbdr)
      {
         // Must be on a vertical edge and 2 of {ibdr, jbdr, ijbdr} are true
         offset += om1*6;
         return offset + (k-1)
                + ((ibdr && jbdr) ? 0 : (jbdr && ijbdr ? 1 : 2))*om1;
      }
      else
      {
         // Must be on a horizontal edge and kbdr plus 1 of {ibdr, jbdr, ijbdr} is true
         // Skip past first 3 edges if we are on the top (k = ref) face:
         offset += (k == ref ? 3*om1 : 0);
         if (jbdr)
         {
            return offset + i - 1;
         }
         offset += om1; // Skip the i-axis edge
         if (ijbdr)
         {
            return offset + j - 1;
         }
         offset += om1; // Skip the ij-axis edge
         // if (ibdr)
         return offset + (ref - j - 1);
      }
   }

   offset += 9*om1; // Skip all the edges

   // Number of points on a triangular face (but not on edge/corner):
   int ntfdof = (om1 - 1)*om1/2;
   int nqfdof = om1*om1;
   if (nbdr == 1) // Face DOF
   {
      if (kbdr)
      {
         // We are on a triangular face.
         if (k > 0)
         {
            offset += ntfdof;
         }
         return offset + VTKTriangleDOFOffset(ref, i, j);
      }
      // Not a k-normal face, so skip them:
      offset += 2*ntfdof;

      // Face is quadrilateral (ref - 1) x (ref - 1)
      // First face is i-normal, then ij-normal, then j-normal
      if (jbdr) // On i-normal face
      {
         return offset + (i - 1) + om1*(k - 1);
      }
      offset += nqfdof; // Skip i-normal face
      if (ijbdr) // on ij-normal face
      {
         return offset + (j - 1) + om1*(k - 1);
      }
      offset += nqfdof; // Skip ij-normal face
      return offset + (ref - j - 1) + om1*(k - 1);
   }

   // Skip all face DOF
   offset += 2*ntfdof + 3*nqfdof;

   // nbdr == 0: Body DOF
   return offset + VTKTriangleDOFOffset(ref, i, j) + ntfdof*(k - 1);
   // (i - 1) + (ref-1)*((j - 1) + (ref - 1)*(k - 1)));
}

int CartesianToVTKTensor(int idx_in, int ref, Geometry::Type geom)
{
   int n = ref + 1;
   switch (geom)
   {
      case Geometry::POINT:
         return idx_in;
      case Geometry::SEGMENT:
         if (idx_in == 0 || idx_in == ref)
         {
            return idx_in ? 1 : 0;
         }
         return idx_in + 1;
      case Geometry::SQUARE:
      {
         // Cf: https://git.io/JvZLT
         int i = idx_in % n;
         int j = idx_in / n;
         // Do we lie on any of the edges
         bool ibdr = (i == 0 || i == ref);
         bool jbdr = (j == 0 || j == ref);
         if (ibdr && jbdr) // Vertex DOF
         {
            return (i ? (j ? 2 : 1) : (j ? 3 : 0));
         }
         int offset = 4;
         if (jbdr) // Edge DOF on j==0 or j==ref
         {
            return (i - 1) + (j ? ref - 1 + ref - 1 : 0) + offset;
         }
         else if (ibdr) // Edge DOF on i==0 or i==ref
         {
            return (j - 1) + (i ? ref - 1 : 2 * (ref - 1) + ref - 1) + offset;
         }
         else // Interior DOF
         {
            offset += 2 * (ref - 1 + ref - 1);
            return offset + (i - 1) + (ref - 1) * ((j - 1));
         }
      }
      case Geometry::CUBE:
      {
         // Cf: https://t.ly/HEGbX
         int i = idx_in % n;
         int j = (idx_in / n) % n;
         int k = idx_in / (n*n);
         bool ibdr = (i == 0 || i == ref);
         bool jbdr = (j == 0 || j == ref);
         bool kbdr = (k == 0 || k == ref);
         // How many boundaries do we lie on at once?
         int nbdr = (ibdr ? 1 : 0) + (jbdr ? 1 : 0) + (kbdr ? 1 : 0);
         if (nbdr == 3) // Vertex DOF
         {
            // ijk is a corner node. Return the proper index (in [0,7])
            return (i ? (j ? 2 : 1) : (j ? 3 : 0)) + (k ? 4 : 0);
         }

         int offset = 8;
         if (nbdr == 2) // Edge DOF
         {
            if (!ibdr)
            {
               // On i axis
               return (i - 1) +
                      (j ? ref - 1 + ref - 1 : 0) +
                      (k ? 2*(ref - 1 + ref - 1) : 0) +
                      offset;
            }
            if (!jbdr)
            {
               // On j axis
               return (j - 1) +
                      (i ? ref - 1 : 2*(ref - 1) + ref - 1) +
                      (k ? 2*(ref - 1 + ref - 1) : 0) +
                      offset;
            }
            // !kbdr, On k axis
            offset += 4*(ref - 1) + 4*(ref - 1);
            return (k - 1) + (ref - 1)*(i ? (j ? 2 : 1) : (j ? 3 : 0))
                   + offset;
         }

         offset += 4*(ref - 1 + ref - 1 + ref - 1);
         if (nbdr == 1) // Face DOF
         {
            if (ibdr) // On i-normal face
            {
               return (j - 1) + ((ref - 1)*(k - 1))
                      + (i ? (ref - 1)*(ref - 1) : 0) + offset;
            }
            offset += 2*(ref - 1)*(ref - 1);
            if (jbdr) // On j-normal face
            {
               return (i - 1)
                      + ((ref - 1)*(k - 1))
                      + (j ? (ref - 1)*(ref - 1) : 0) + offset;
            }
            offset += 2*(ref - 1)*(ref - 1);
            // kbdr, On k-normal face
            return (i - 1) + ((ref - 1)*(j - 1))
                   + (k ? (ref - 1)*(ref - 1) : 0) + offset;
         }

         // nbdr == 0: Interior DOF
         offset += 2*((ref - 1)*(ref - 1) +
                      (ref - 1)*(ref - 1) +
                      (ref - 1)*(ref - 1));
         return offset + (i - 1) + (ref - 1)*((j - 1) + (ref - 1)*(k - 1));
      }
      default:
         MFEM_ABORT("CartesianToVTKOrderingTensor only supports tensor"
                    " geometries.");
         return -1;
   }
}

void CreateVTKElementConnectivity(Array<int> &con, Geometry::Type geom, int ref)
{

   RefinedGeometry *RefG = GlobGeometryRefiner.Refine(geom, ref, 1);
   int nnodes = RefG->RefPts.GetNPoints();
   con.SetSize(nnodes);
   if (geom == Geometry::TRIANGLE)
   {
      int b[3];
      int idx = 0;
      for (b[1]=0; b[1]<=ref; ++b[1])
      {
         for (b[0]=0; b[0]<=ref-b[1]; ++b[0])
         {
            b[2] = ref - b[0] - b[1];
            con[BarycentricToVTKTriangle(b, ref)] = idx++;
         }
      }
   }
   else if (geom == Geometry::TETRAHEDRON)
   {
      int idx = 0;
      int b[4];
      for (b[2]=0; b[2]<=ref; b[2]++)
      {
         for (b[1]=0; b[1]<=ref-b[2]; b[1]++)
         {
            for (b[0]=0; b[0]<=ref-b[1]-b[2]; b[0]++)
            {
               b[3] = ref-b[0]-b[1]-b[2];
               con[BarycentricToVTKTetra(b, ref)] = idx++;
            }
         }
      }
   }
   else if (geom == Geometry::PRISM)
   {
      int idx = 0;
      for (int k=0; k<=ref; k++)
      {
         for (int j=0; j<=ref; j++)
         {
            for (int i=0; i<=ref-j; i++)
            {
               con[CartesianToVTKPrism(i, j, k, ref)] = idx++;
            }
         }
      }
   }
   else if (geom == Geometry::PYRAMID)
   {
      MFEM_ABORT("Lagrange pyramid elements not currently supported in VTK.");
   }
   else
   {
      for (int idx=0; idx<nnodes; ++idx)
      {
         con[CartesianToVTKTensor(idx, ref, geom)] = idx;
      }
   }
}

void WriteVTKEncodedCompressed(std::ostream &os, const void *bytes,
                               uint32_t nbytes, int compression_level)
{
   if (compression_level == 0)
   {
      // First write size of buffer (as uint32_t), encoded with base 64
      bin_io::WriteBase64(os, &nbytes, sizeof(nbytes));
      // Then write all the bytes in the buffer, encoded with base 64
      bin_io::WriteBase64(os, bytes, nbytes);
   }
   else
   {
#ifdef MFEM_USE_ZLIB
      MFEM_ASSERT(compression_level >= -1 && compression_level <= 9,
                  "Compression level must be between -1 and 9 (inclusive).");
      uLongf buf_sz = compressBound(nbytes);
      std::vector<unsigned char> buf(buf_sz);
      compress2(buf.data(), &buf_sz, static_cast<const Bytef *>(bytes), nbytes,
                compression_level);

      // Write the header
      std::vector<uint32_t> header(4);
      header[0] = 1; // number of blocks
      header[1] = nbytes; // uncompressed size
      header[2] = 0; // size of partial block
      header[3] = buf_sz; // compressed size
      bin_io::WriteBase64(os, header.data(), header.size()*sizeof(uint32_t));
      // Write the compressed data
      bin_io::WriteBase64(os, buf.data(), buf_sz);
#else
      MFEM_ABORT("MFEM must be compiled with ZLib support to output "
                 "compressed binary data.")
#endif
   }
}

bool IsBigEndian()
{
   int16_t x16 = 1;
   int8_t *x8 = reinterpret_cast<int8_t *>(&x16);
   return !*x8;
}

const char *VTKByteOrder()
{
   if (IsBigEndian())
   {
      return "BigEndian";
   }
   else
   {
      return "LittleEndian";
   }

}

// Ensure ASCII output of uint8_t to stream is integer rather than character
template <>
void WriteBinaryOrASCII<uint8_t>(std::ostream &os, std::vector<char> &buf,
                                 const uint8_t &val, const char *suffix,
                                 VTKFormat format)
{
   if (format == VTKFormat::ASCII) { os << static_cast<int>(val) << suffix; }
   else { bin_io::AppendBytes(buf, val); }
}

template <>
void WriteBinaryOrASCII<double>(std::ostream &os, std::vector<char> &buf,
                                const double &val, const char *suffix,
                                VTKFormat format)
{
   if (format == VTKFormat::BINARY32)
   {
      bin_io::AppendBytes<float>(buf, float(val));
   }
   else if (format == VTKFormat::BINARY)
   {
      bin_io::AppendBytes(buf, val);
   }
   else
   {
      os << ZeroSubnormal(val) << suffix;
   }
}

template <>
void WriteBinaryOrASCII<float>(std::ostream &os, std::vector<char> &buf,
                               const float &val, const char *suffix,
                               VTKFormat format)
{
   if (format == VTKFormat::BINARY) { bin_io::AppendBytes<double>(buf, val); }
   else if (format == VTKFormat::BINARY32) { bin_io::AppendBytes(buf, val); }
   else { os << ZeroSubnormal(val) << suffix; }
}

void WriteBase64WithSizeAndClear(std::ostream &os, std::vector<char> &buf,
                                 int compression_level)
{
   WriteVTKEncodedCompressed(os, buf.data(), buf.size(), compression_level);
   os << '\n';
   buf.clear();
}

std::string VTKComponentLabels(int vdim)
{
   if (vdim == 1)
   {
      return "";
   }
   else
   {
      std::stringstream s;
      for (int i = 0; i < vdim; ++i)
      {
         s << "ComponentName" << i << "=\"" << i << "\"";
         if (i < vdim - 1) { s << " "; }
      }
      return s.str();
   }
}

} // namespace mfem
