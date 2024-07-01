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

#include "mesh_headers.hpp"
#include "vtk.hpp"

using namespace std;

namespace mfem
{

static int BarycentricToGmshTet(int *b, int ref)
{
   int i = b[0];
   int j = b[1];
   int k = b[2];
   int l = b[3];
   bool ibdr = (i == 0);
   bool jbdr = (j == 0);
   bool kbdr = (k == 0);
   bool lbdr = (l == 0);
   if (ibdr && jbdr && kbdr)
   {
      return 0;
   }
   else if (jbdr && kbdr && lbdr)
   {
      return 1;
   }
   else if (ibdr && kbdr && lbdr)
   {
      return 2;
   }
   else if (ibdr && jbdr && lbdr)
   {
      return 3;
   }
   int offset = 4;
   if (jbdr && kbdr) // Edge DOF on j == 0 and k == 0
   {
      return offset + i - 1;
   }
   else if (kbdr && lbdr) // Edge DOF on k == 0 and l == 0
   {
      return offset + ref - 1 + j - 1;
   }
   else if (ibdr && kbdr) // Edge DOF on i == 0 and k == 0
   {
      return offset + 2 * (ref - 1) + ref - j - 1;
   }
   else if (ibdr && jbdr) // Edge DOF on i == 0 and j == 0
   {
      return offset + 3 * (ref - 1) + ref - k - 1;
   }
   else if (ibdr && lbdr) // Edge DOF on i == 0 and l == 0
   {
      return offset + 4 * (ref - 1) + ref - k - 1;
   }
   else if (jbdr && lbdr) // Edge DOF on j == 0 and l == 0
   {
      return offset + 5 * (ref - 1) + ref - k - 1;
   }

   // Recursive numbering for the faces
   offset += 6 * (ref - 1);
   if (kbdr)
   {
      int b_out[3];
      b_out[0] = j-1;
      b_out[1] = i-1;
      b_out[2] = ref - i - j - 1;
      return offset + BarycentricToVTKTriangle(b_out, ref-3);
   }
   else if (jbdr)
   {
      int b_out[3];
      b_out[0] = i-1;
      b_out[1] = k-1;
      b_out[2] = ref - i - k - 1;
      offset += (ref - 1) * (ref - 2) / 2;
      return offset + BarycentricToVTKTriangle(b_out, ref-3);
   }
   else if (ibdr)
   {
      int b_out[3];
      b_out[0] = k-1;
      b_out[1] = j-1;
      b_out[2] = ref - j - k - 1;
      offset += (ref - 1) * (ref - 2);
      return offset + BarycentricToVTKTriangle(b_out, ref-3);
   }
   else if (lbdr)
   {
      int b_out[3];
      b_out[0] = ref-j-k-1;
      b_out[1] = j-1;
      b_out[2] = k-1;
      offset += 3 * (ref - 1) * (ref - 2) / 2;
      return offset + BarycentricToVTKTriangle(b_out, ref-3);
   }

   // Recursive numbering for interior
   {
      int b_out[4];
      b_out[0] = i-1;
      b_out[1] = j-1;
      b_out[2] = k-1;
      b_out[3] = ref - i - j - k - 1;
      offset += 2 * (ref - 1) * (ref - 2);
      return offset + BarycentricToGmshTet(b_out, ref-4);
   }
}

static int CartesianToGmshQuad(int idx_in[], int ref)
{
   int i = idx_in[0];
   int j = idx_in[1];
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
      return offset + (j ? 3*ref - 3 - i : i - 1);
   }
   else if (ibdr) // Edge DOF on i==0 or i==ref
   {
      return offset + (i ? ref - 1 + j - 1 : 4*ref - 4 - j);
   }
   else // Recursive numbering for interior
   {
      int idx_out[2];
      idx_out[0] = i-1;
      idx_out[1] = j-1;
      offset += 4 * (ref - 1);
      return offset + CartesianToGmshQuad(idx_out, ref-2);
   }
}

static int CartesianToGmshHex(int idx_in[], int ref)
{
   int i = idx_in[0];
   int j = idx_in[1];
   int k = idx_in[2];
   // Do we lie on any of the edges
   bool ibdr = (i == 0 || i == ref);
   bool jbdr = (j == 0 || j == ref);
   bool kbdr = (k == 0 || k == ref);
   if (ibdr && jbdr && kbdr) // Vertex DOF
   {
      return (i ? (j ? (k ? 6 : 2) : (k ? 5 : 1)) :
              (j ? (k ? 7 : 3) : (k ? 4 : 0)));
   }
   int offset = 8;
   if (jbdr && kbdr) // Edge DOF on x-directed edge
   {
      return offset + (j ? (k ? 12*ref-12-i: 6*ref-6-i) :
                       (k ? 8*ref-9+i: i-1));
   }
   else if (ibdr && kbdr) // Edge DOF on y-directed edge
   {
      return offset + (k ? (i ? 10*ref-11+j: 9*ref-10+j) :
                       (i ? 3*ref-4+j: ref-2+j));
   }
   else if (ibdr && jbdr) // Edge DOF on z-directed edge
   {
      return offset + (i ? (j ? 6*ref-7+k: 4*ref-5+k) :
                       (j ? 7*ref-8+k: 2*ref-3+k));
   }
   else if (ibdr) // Face DOF on x-directed face
   {
      int idx_out[2];
      idx_out[0] = i ? j-1 : k-1;
      idx_out[1] = i ? k-1 : j-1;
      offset += (12 + (i ? 3 : 2) * (ref - 1)) * (ref - 1);
      return offset + CartesianToGmshQuad(idx_out, ref-2);
   }
   else if (jbdr) // Face DOF on y-directed face
   {
      int idx_out[2];
      idx_out[0] = j ? ref-i-1 : i-1;
      idx_out[1] = j ? k-1 : k-1;
      offset += (12 + (j ? 4 : 1) * (ref - 1)) * (ref - 1);
      return offset + CartesianToGmshQuad(idx_out, ref-2);
   }
   else if (kbdr) // Face DOF on z-directed face
   {
      int idx_out[2];
      idx_out[0] = k ? i-1 : j-1;
      idx_out[1] = k ? j-1 : i-1;
      offset += (12 + (k ? 5 : 0) * (ref - 1)) * (ref - 1);
      return offset + CartesianToGmshQuad(idx_out, ref-2);
   }
   else // Recursive numbering for interior
   {
      int idx_out[3];
      idx_out[0] = i-1;
      idx_out[1] = j-1;
      idx_out[2] = k-1;

      offset += (12 + 6 * (ref - 1)) * (ref - 1);
      return offset + CartesianToGmshHex(idx_out, ref-2);
   }
}

static int WedgeToGmshPri(int idx_in[], int ref)
{
   int i = idx_in[0];
   int j = idx_in[1];
   int k = idx_in[2];
   int l = ref - i -j;
   bool ibdr = (i == 0);
   bool jbdr = (j == 0);
   bool kbdr = (k == 0 || k == ref);
   bool lbdr = (l == 0);
   if (ibdr && jbdr && kbdr)
   {
      return k ? 3 : 0;
   }
   else if (jbdr && lbdr && kbdr)
   {
      return k ? 4 : 1;
   }
   else if (ibdr && lbdr && kbdr)
   {
      return k ? 5 : 2;
   }
   int offset = 6;
   if (jbdr && kbdr)
   {
      return offset + (k ? 6 * (ref - 1) + i - 1: i - 1);
   }
   else if (ibdr && kbdr)
   {
      return offset + (k ? 7 * (ref -1) + j-1 : ref - 1 + j - 1);
   }
   else if (ibdr && jbdr)
   {
      return offset + 2 * (ref - 1) + k - 1;
   }
   else if (lbdr && kbdr)
   {
      return offset + (k ? 8 * (ref -1) + j - 1 : 3 * (ref - 1) + j - 1);
   }
   else if (jbdr && lbdr)
   {
      return offset + 4 * (ref - 1) + k - 1;
   }
   else if (ibdr && lbdr)
   {
      return offset + 5 * (ref - 1) + k - 1;
   }
   offset += 9 * (ref-1);
   if (kbdr) // Triangular faces at k=0 and k=ref
   {
      int b_out[3];
      b_out[0] = k ? i-1 : j-1;
      b_out[1] = k ? j-1 : i-1;
      b_out[2] = ref - i - j - 1;
      offset += k ? (ref-1)*(ref-2) / 2: 0;
      return offset + BarycentricToVTKTriangle(b_out, ref-3);
   }
   offset += (ref-1)*(ref-2);
   if (jbdr) // Quadrilateral face at j=0
   {
      int idx_out[2];
      idx_out[0] = i-1;
      idx_out[1] = k-1;
      return offset + CartesianToGmshQuad(idx_out, ref-2);
   }
   else if (ibdr) // Quadrilateral face at i=0
   {
      int idx_out[2];
      idx_out[0] = k-1;
      idx_out[1] = j-1;
      offset += (ref-1)*(ref-1);
      return offset + CartesianToGmshQuad(idx_out, ref-2);
   }
   else if (lbdr) // Quadrilateral face at l=ref-i-j=0
   {
      int idx_out[2];
      idx_out[0] = j-1;
      idx_out[1] = k-1;
      offset += 2*(ref-1)*(ref-1);
      return offset + CartesianToGmshQuad(idx_out, ref-2);
   }
   offset += 3*(ref-1)*(ref-1);
   // The Gmsh Prism interiors are a tensor product of segments of order ref-2
   // and triangles of order ref-3
   {
      int b_out[3];
      b_out[0] = i-1;
      b_out[1] = j-1;
      b_out[2] = ref - i - j - 1;
      int ot = BarycentricToVTKTriangle(b_out, ref-3);
      int os = (k==1) ? 0 : (k == ref-1 ? 1 : k);
      return offset + (ref-1) * ot + os;
   }
}

static int CartesianToGmshPyramid(int idx_in[], int ref)
{
   int i = idx_in[0];
   int j = idx_in[1];
   int k = idx_in[2];
   // Do we lie on any of the edges
   bool ibdr = (i == 0 || i == ref-k);
   bool jbdr = (j == 0 || j == ref-k);
   bool kbdr = (k == 0);
   if (ibdr && jbdr && kbdr)
   {
      return i ? (j ? 2 : 1): (j ? 3 : 0);
   }
   else if (k == ref)
   {
      return 4;
   }
   int offset = 5;
   if (jbdr && kbdr)
   {
      return offset + (j ? (6 * ref - 6 - i) : (i - 1));
   }
   else if (ibdr && kbdr)
   {
      return offset + (i ? (3 * ref - 4 + j) : (ref - 2 + j));
   }
   else if (ibdr && jbdr)
   {
      return offset + (i ? (j ? 6 : 4) : (j ? 7 : 2 )) * (ref-1) + k - 1;
   }
   offset += 8*(ref-1);
   if (jbdr)
   {
      int b_out[3];
      b_out[0] = j ? ref - i - k - 1 : i - 1;
      b_out[1] = k - 1;
      b_out[2] = (j ? i - 1 : ref - i - k - 1);
      offset += (j ? 3 : 0) * (ref - 1) * (ref - 2) / 2;
      return offset + BarycentricToVTKTriangle(b_out, ref-3);
   }
   else if (ibdr)
   {
      int b_out[3];
      b_out[0] = i ? j - 1: ref - j - k - 1;
      b_out[1] = k - 1;
      b_out[2] = (i ? ref - j - k - 1: j - 1);
      offset += (i ? 2 : 1) * (ref - 1) * (ref - 2) / 2;
      return offset + BarycentricToVTKTriangle(b_out, ref-3);
   }
   else if (kbdr)
   {
      int idx_out[2];
      idx_out[0] = k ? i-1 : j-1;
      idx_out[1] = k ? j-1 : i-1;
      offset += 2 * (ref - 1) * (ref - 2);
      return offset + CartesianToGmshQuad(idx_out, ref-2);
   }
   offset += (2 * (ref - 2) + (ref - 1)) * (ref - 1) ;
   {
      int idx_out[3];
      idx_out[0] = i-1;
      idx_out[1] = j-1;
      idx_out[2] = k-1;
      return offset + CartesianToGmshPyramid(idx_out, ref-3);
   }
}

static void GmshHOSegmentMapping(int order, int *map)
{
   map[0] = 0;
   map[order] = 1;
   for (int i=1; i<order; i++)
   {
      map[i] = i + 1;
   }
}

static void GmshHOTriangleMapping(int order, int *map)
{
   int b[3];
   int o = 0;
   for (b[1]=0; b[1]<=order; ++b[1])
   {
      for (b[0]=0; b[0]<=order-b[1]; ++b[0])
      {
         b[2] = order - b[0] - b[1];
         map[o] = BarycentricToVTKTriangle(b, order);
         o++;
      }
   }
}

static void GmshHOQuadrilateralMapping(int order, int *map)
{
   int b[2];
   int o = 0;
   for (b[1]=0; b[1]<=order; b[1]++)
   {
      for (b[0]=0; b[0]<=order; b[0]++)
      {
         map[o] = CartesianToGmshQuad(b, order);
         o++;
      }
   }
}

static void GmshHOTetrahedronMapping(int order, int *map)
{
   int b[4];
   int o = 0;
   for (b[2]=0; b[2]<=order; ++b[2])
   {

      for (b[1]=0; b[1]<=order-b[2]; ++b[1])
      {
         for (b[0]=0; b[0]<=order-b[1]-b[2]; ++b[0])
         {
            b[3] = order - b[0] - b[1] - b[2];
            map[o] = BarycentricToGmshTet(b, order);
            o++;
         }
      }
   }
}

static void GmshHOHexahedronMapping(int order, int *map)
{
   int b[3];
   int o = 0;
   for (b[2]=0; b[2]<=order; b[2]++)
   {
      for (b[1]=0; b[1]<=order; b[1]++)
      {
         for (b[0]=0; b[0]<=order; b[0]++)
         {
            map[o] = CartesianToGmshHex(b, order);
            o++;
         }
      }
   }
}

static void GmshHOWedgeMapping(int order, int *map)
{
   int b[3];
   int o = 0;
   for (b[2]=0; b[2]<=order; b[2]++)
   {
      for (b[1]=0; b[1]<=order; b[1]++)
      {
         for (b[0]=0; b[0]<=order - b[1]; b[0]++)
         {
            map[o] = WedgeToGmshPri(b, order);
            o++;
         }
      }
   }
}

static void GmshHOPyramidMapping(int order, int *map)
{
   int b[3];
   int o = 0;
   for (b[2]=0; b[2]<=order; b[2]++)
   {
      for (b[1]=0; b[1]<=order - b[2]; b[1]++)
      {
         for (b[0]=0; b[0]<=order - b[2]; b[0]++)
         {
            map[o] = CartesianToGmshPyramid(b, order);
            o++;
         }
      }
   }
}

static std::pair<Geometry::Type, int> GmshGetGeometryAndOrder(int elem_type)
{
   Geometry::Type geom = Geometry::INVALID;

   int el_order = 11;
   switch (elem_type)
   {
      case  1: el_order--; //   2-node line
      case  8: el_order--; //   3-node line (2nd order)
      case 26: el_order--; //   4-node line (3rd order)
      case 27: el_order--; //   5-node line (4th order)
      case 28: el_order--; //   6-node line (5th order)
      case 62: el_order--; //   7-node line (6th order)
      case 63: el_order--; //   8-node line (7th order)
      case 64: el_order--; //   9-node line (8th order)
      case 65: el_order--; //  10-node line (9th order)
      case 66: el_order--; //  11-node line (10th order)
         geom = Geometry::SEGMENT;
         break;
      case  2: el_order--; //  3-node triangle
      case  9: el_order--; //  6-node triangle (2nd order)
      case 21: el_order--; // 10-node triangle (3rd order)
      case 23: el_order--; // 15-node triangle (4th order)
      case 25: el_order--; // 21-node triangle (5th order)
      case 42: el_order--; // 28-node triangle (6th order)
      case 43: el_order--; // 36-node triangle (7th order)
      case 44: el_order--; // 45-node triangle (8th order)
      case 45: el_order--; // 55-node triangle (9th order)
      case 46: el_order--; // 66-node triangle (10th order)
         geom = Geometry::TRIANGLE;
         break;
      case  3: el_order--; //   4-node quadrangle
      case 10: el_order--; //   9-node quadrangle (2nd order)
      case 36: el_order--; //  16-node quadrangle (3rd order)
      case 37: el_order--; //  25-node quadrangle (4th order)
      case 38: el_order--; //  36-node quadrangle (5th order)
      case 47: el_order--; //  49-node quadrangle (6th order)
      case 48: el_order--; //  64-node quadrangle (7th order)
      case 49: el_order--; //  81-node quadrangle (8th order)
      case 50: el_order--; // 100-node quadrangle (9th order)
      case 51: el_order--; // 121-node quadrangle (10th order)
         geom = Geometry::SQUARE;
         break;
      case  4: el_order--; //   4-node tetrahedron
      case 11: el_order--; //  10-node tetrahedron (2nd order)
      case 29: el_order--; //  20-node tetrahedron (3rd order)
      case 30: el_order--; //  35-node tetrahedron (4th order)
      case 31: el_order--; //  56-node tetrahedron (5th order)
      case 71: el_order--; //  84-node tetrahedron (6th order)
      case 72: el_order--; // 120-node tetrahedron (7th order)
      case 73: el_order--; // 165-node tetrahedron (8th order)
      case 74: el_order--; // 220-node tetrahedron (9th order)
      case 75: el_order--; // 286-node tetrahedron (10th order)
         geom = Geometry::TETRAHEDRON;
         break;
      case  5: el_order--; //    8-node hexahedron
      case 12: el_order--; //   27-node hexahedron (2nd order)
      case 92: el_order--; //   64-node hexahedron (3rd order)
      case 93: el_order--; //  125-node hexahedron (4th order)
      case 94: el_order--; //  216-node hexahedron (5th order)
      case 95: el_order--; //  343-node hexahedron (6th order)
      case 96: el_order--; //  512-node hexahedron (7th order)
      case 97: el_order--; //  729-node hexahedron (8th order)
      case 98: el_order--; // 1000-node hexahedron (9th order)
         geom = Geometry::CUBE;
         break;
      case   6: el_order--; //   6-node wedge
      case  13: el_order--; //  18-node wedge (2nd order)
      case  90: el_order--; //  40-node wedge (3rd order)
      case  91: el_order--; //  75-node wedge (4th order)
      case 106: el_order--; // 126-node wedge (5th order)
      case 107: el_order--; // 196-node wedge (6th order)
      case 108: el_order--; // 288-node wedge (7th order)
      case 109: el_order--; // 405-node wedge (8th order)
      case 110: el_order--; // 550-node wedge (9th order)
         geom = Geometry::PRISM;
         break;
      case   7: el_order--; //   5-node pyramid
      case  14: el_order--; //  14-node pyramid (2nd order)
      case 118: el_order--; //  30-node pyramid (3rd order)
      case 119: el_order--; //  55-node pyramid (4th order)
      case 120: el_order--; //  91-node pyramid (5th order)
      case 121: el_order--; // 140-node pyramid (6th order)
      case 122: el_order--; // 204-node pyramid (7th order)
      case 123: el_order--; // 285-node pyramid (8th order)
      case 124: el_order--; // 385-node pyramid (9th order)
         geom = Geometry::PYRAMID;
         break;
      case 15: // 1-node point
         el_order = 1;
         geom = Geometry::POINT;
         break;
      default: // any other element
         MFEM_WARNING("Unsupported Gmsh element type.");
         break;
   } // switch (type_of_element)

   return std::make_pair(geom, el_order);
}

static int GmshNumNodesInElement(Geometry::Type geom, int order)
{
   return GlobGeometryRefiner.Refine(geom, order, 1)->RefPts.GetNPoints();
}

void Mesh::ReadGmshMesh(std::istream &input, int &curved, int &read_gf)
{
   string buff;
   real_t version;
   int binary, dsize;
   input >> version >> binary >> dsize;
   if (version < 2.2)
   {
      MFEM_ABORT("Gmsh file version < 2.2");
   }
   MFEM_VERIFY(dsize == sizeof(double), "Gmsh file : dsize != sizeof(double)");
   getline(input, buff);
   // There is a number 1 in binary format
   if (binary)
   {
      int one;
      input.read(reinterpret_cast<char*>(&one), sizeof(one));
      if (one != 1)
      {
         MFEM_ABORT("Gmsh file : wrong binary format");
      }
   }

   // A map between a serial number of the vertex and its number in the file
   // (there may be gaps in the numbering, and also Gmsh enumerates vertices
   // starting from 1, not 0)
   map<int, int> vertices_map;

   // A map containing names of physical curves, surfaces, and volumes.
   // The first index is the dimension of the physical manifold, the second
   // index is the element attribute number of the set, and the string is
   // the assigned name.
   map<int,map<int,std::string> > phys_names_by_dim;

   // Gmsh always outputs coordinates in 3D, but MFEM distinguishes between the
   // mesh element dimension (Dim) and the dimension of the space in which the
   // mesh is embedded (spaceDim). For example, a 2D MFEM mesh has Dim = 2 and
   // spaceDim = 2, while a 2D surface mesh in 3D has Dim = 2 but spaceDim = 3.
   // Below we set spaceDim by measuring the mesh bounding box and checking for
   // a lower dimensional subspace. The assumption is that the mesh is at least
   // 2D if the y-dimension of the box is non-trivial and 3D if the z-dimension
   // is non-trivial. Note that with these assumptions a 2D mesh parallel to the
   // yz plane will be considered a surface mesh embedded in 3D whereas the same
   // 2D mesh parallel to the xy plane will be considered a 2D mesh.
   real_t bb_tol = 1e-14;
   real_t bb_min[3];
   real_t bb_max[3];

   // Mesh order
   int mesh_order = 1;

   // Mesh type
   bool periodic = false;

   // Vector field to store uniformly spaced Gmsh high order mesh coords
   GridFunction Nodes_gf;

   // Read the lines of the mesh file. If we face specific keyword, we'll treat
   // the section.
   while (input >> buff)
   {
      if (buff == "$Nodes") // reading mesh vertices
      {
         input >> NumOfVertices;
         getline(input, buff);
         vertices.SetSize(NumOfVertices);
         int serial_number;
         const int gmsh_dim = 3; // Gmsh always outputs 3 coordinates
         real_t coord[gmsh_dim];
         for (int ver = 0; ver < NumOfVertices; ++ver)
         {
            if (binary)
            {
               input.read(reinterpret_cast<char*>(&serial_number), sizeof(int));
               input.read(reinterpret_cast<char*>(coord), gmsh_dim*sizeof(double));
            }
            else // ASCII
            {
               input >> serial_number;
               for (int ci = 0; ci < gmsh_dim; ++ci)
               {
                  input >> coord[ci];
               }
            }
            vertices[ver] = Vertex(coord, gmsh_dim);
            vertices_map[serial_number] = ver;

            for (int ci = 0; ci < gmsh_dim; ++ci)
            {
               bb_min[ci] = (ver == 0) ? coord[ci] :
                            std::min(bb_min[ci], coord[ci]);
               bb_max[ci] = (ver == 0) ? coord[ci] :
                            std::max(bb_max[ci], coord[ci]);
            }
         }
         real_t bb_size = std::max(bb_max[0] - bb_min[0],
                                   std::max(bb_max[1] - bb_min[1],
                                            bb_max[2] - bb_min[2]));
         spaceDim = 1;
         if (bb_max[1] - bb_min[1] > bb_size * bb_tol)
         {
            spaceDim++;
         }
         if (bb_max[2] - bb_min[2] > bb_size * bb_tol)
         {
            spaceDim++;
         }

         if (static_cast<int>(vertices_map.size()) != NumOfVertices)
         {
            MFEM_ABORT("Gmsh file : vertices indices are not unique");
         }
      } // section '$Nodes'
      else if (buff == "$Elements") // reading mesh elements
      {
         int num_of_all_elements;
         input >> num_of_all_elements;
         // = NumOfElements + NumOfBdrElements + (maybe, PhysicalPoints)
         getline(input, buff);

         int serial_number; // serial number of an element
         int type_of_element; // ID describing a type of a mesh element
         int n_tags; // number of different tags describing an element
         int phys_domain; // element's attribute
         int elem_domain; // another element's attribute (rarely used)
         int n_partitions; // number of partitions where an element takes place

         vector<Element*> elements_0D, elements_1D, elements_2D, elements_3D;

         // Temporary storage for high order vertices, if present
         vector<vector<int>> ho_verts_1D, ho_verts_2D, ho_verts_3D;

         // Temporary storage for order of elements
         vector<int> ho_el_order_1D, ho_el_order_2D, ho_el_order_3D;

         // Vertex order mappings
         vector<vector<int>> ho_lin(11);
         vector<vector<int>> ho_tri(11);
         vector<vector<int>> ho_sqr(11);
         vector<vector<int>> ho_tet(11);
         vector<vector<int>> ho_hex(10);
         vector<vector<int>> ho_wdg(10);
         vector<vector<int>> ho_pyr(10);

         bool has_nonpositive_phys_domain = false;
         bool has_positive_phys_domain = false;

         if (binary)
         {
            int n_elem_part = 0; // partial sum of elements that are read
            const int header_size = 3;
            // header consists of 3 numbers: type of the element, number of
            // elements of this type, and number of tags
            int header[header_size];
            int n_elem_one_type; // number of elements of a specific type

            while (n_elem_part < num_of_all_elements)
            {
               input.read(reinterpret_cast<char*>(header),
                          header_size*sizeof(int));
               type_of_element = header[0];
               n_elem_one_type = header[1];
               n_tags          = header[2];

               n_elem_part += n_elem_one_type;

               const auto geom_and_order = GmshGetGeometryAndOrder(type_of_element);
               const Geometry::Type geom = geom_and_order.first;
               const int el_order = geom_and_order.second;

               const int n_elem_nodes = GmshNumNodesInElement(geom, el_order);
               vector<int> data(1+n_tags+n_elem_nodes);
               for (int el = 0; el < n_elem_one_type; ++el)
               {
                  input.read(reinterpret_cast<char*>(&data[0]),
                             data.size()*sizeof(int));
                  int dd = 0; // index for data array
                  serial_number = data[dd++];
                  // physical domain - the most important value (to distinguish
                  // materials with different properties)
                  phys_domain = (n_tags > 0) ? data[dd++] : 1;
                  // elementary domain - to distinguish different geometrical
                  // domains (typically, it's used rarely)
                  elem_domain = (n_tags > 1) ? data[dd++] : 0;
                  // the number of tags is bigger than 2 if there are some
                  // partitions (domain decompositions)
                  n_partitions = (n_tags > 2) ? data[dd++] : 0;
                  // we currently just skip the partitions if they exist, and go
                  // directly to vertices describing the mesh element
                  vector<int> vert_indices(n_elem_nodes);
                  for (int vi = 0; vi < n_elem_nodes; ++vi)
                  {
                     map<int, int>::const_iterator it =
                        vertices_map.find(data[1+n_tags+vi]);
                     if (it == vertices_map.end())
                     {
                        MFEM_ABORT("Gmsh file : vertex index doesn't exist");
                     }
                     vert_indices[vi] = it->second;
                  }

                  // Non-positive attributes are not allowed in MFEM. However,
                  // by default, Gmsh sets the physical domain of all elements
                  // to zero. In the case that all elements have physical domain
                  // zero, we will given them attribute 1. If only some elements
                  // have physical domain zero, we will throw an error.
                  if (phys_domain <= 0)
                  {
                     has_nonpositive_phys_domain = true;
                     phys_domain = 1;
                  }
                  else
                  {
                     has_positive_phys_domain = true;
                  }

                  // initialize the mesh element
                  const int dim = Geometry::Dimension[geom];

                  Element *new_elem = NewElement(geom);
                  new_elem->SetVertices(&vert_indices[0]);
                  new_elem->SetAttribute(phys_domain);

                  vector<vector<int>> *ho_verts = nullptr;
                  vector<int> *ho_el_order = nullptr;

                  switch (dim)
                  {
                     case 0:
                        elements_0D.push_back(new_elem);
                        break;
                     case 1:
                        elements_1D.push_back(new_elem);
                        ho_verts = &ho_verts_1D;
                        ho_el_order = &ho_el_order_1D;
                        break;
                     case 2:
                        elements_2D.push_back(new_elem);
                        ho_verts = &ho_verts_2D;
                        ho_el_order = &ho_el_order_2D;
                        break;
                     case 3:
                        elements_3D.push_back(new_elem);
                        ho_verts = &ho_verts_3D;
                        ho_el_order = &ho_el_order_3D;
                        break;
                  }

                  if (el_order > 1)
                  {
                     ho_verts->emplace_back(
                        &vert_indices[0], &vert_indices[0] + n_elem_nodes);
                     ho_el_order->push_back(el_order);
                  }
               } // el (elements of one type)
            } // all elements
         } // if binary
         else // ASCII
         {
            for (int el = 0; el < num_of_all_elements; ++el)
            {
               input >> serial_number >> type_of_element >> n_tags;
               vector<int> data(n_tags);
               for (int i = 0; i < n_tags; ++i) { input >> data[i]; }
               // physical domain - the most important value (to distinguish
               // materials with different properties)
               phys_domain = (n_tags > 0) ? data[0] : 1;
               // elementary domain - to distinguish different geometrical
               // domains (typically, it's used rarely)
               elem_domain = (n_tags > 1) ? data[1] : 0;
               // the number of tags is bigger than 2 if there are some
               // partitions (domain decompositions)
               n_partitions = (n_tags > 2) ? data[2] : 0;
               // we currently just skip the partitions if they exist, and go
               // directly to vertices describing the mesh element

               const auto geom_and_order = GmshGetGeometryAndOrder(type_of_element);
               const Geometry::Type geom = geom_and_order.first;
               const int el_order = geom_and_order.second;
               const int n_elem_nodes = GmshNumNodesInElement(geom, el_order);
               vector<int> vert_indices(n_elem_nodes);
               int index;
               for (int vi = 0; vi < n_elem_nodes; ++vi)
               {
                  input >> index;
                  map<int, int>::const_iterator it = vertices_map.find(index);
                  if (it == vertices_map.end())
                  {
                     MFEM_ABORT("Gmsh file : vertex index doesn't exist");
                  }
                  vert_indices[vi] = it->second;
               }

               // Non-positive attributes are not allowed in MFEM. However,
               // by default, Gmsh sets the physical domain of all elements
               // to zero. In the case that all elements have physical domain
               // zero, we will given them attribute 1. If only some elements
               // have physical domain zero, we will throw an error.
               if (phys_domain <= 0)
               {
                  has_nonpositive_phys_domain = true;
                  phys_domain = 1;
               }
               else
               {
                  has_positive_phys_domain = true;
               }

               // initialize the mesh element
               const int dim = Geometry::Dimension[geom];

               Element *new_elem = NewElement(geom);
               new_elem->SetVertices(&vert_indices[0]);
               new_elem->SetAttribute(phys_domain);

               vector<vector<int>> *ho_verts = nullptr;
               vector<int> *ho_el_order = nullptr;

               switch (dim)
               {
                  case 0:
                     elements_0D.push_back(new_elem);
                     break;
                  case 1:
                     elements_1D.push_back(new_elem);
                     ho_verts = &ho_verts_1D;
                     ho_el_order = &ho_el_order_1D;
                     break;
                  case 2:
                     elements_2D.push_back(new_elem);
                     ho_verts = &ho_verts_2D;
                     ho_el_order = &ho_el_order_2D;
                     break;
                  case 3:
                     elements_3D.push_back(new_elem);
                     ho_verts = &ho_verts_3D;
                     ho_el_order = &ho_el_order_3D;
                     break;
               }

               if (el_order > 1)
               {
                  ho_verts->emplace_back(
                     &vert_indices[0], &vert_indices[0] + n_elem_nodes);
                  ho_el_order->push_back(el_order);
               }
            } // el (all elements)
         } // if ASCII

         if (has_positive_phys_domain && has_nonpositive_phys_domain)
         {
            MFEM_ABORT("Non-positive element attribute in Gmsh mesh!\n"
                       "By default Gmsh sets element tags (attributes)"
                       " to '0' but MFEM requires that they be"
                       " positive integers.\n"
                       "Use \"Physical Curve\", \"Physical Surface\","
                       " or \"Physical Volume\" to set tags/attributes"
                       " for all curves, surfaces, or volumes in your"
                       " Gmsh geometry to values which are >= 1.");
         }
         else if (has_nonpositive_phys_domain)
         {
            mfem::out << "\nGmsh reader: all element attributes were zero.\n"
                      << "MFEM only supports positive element attributes.\n"
                      << "Setting element attributes to 1.\n\n";
         }

         if (!elements_3D.empty())
         {
            Dim = 3;
            NumOfElements = elements_3D.size();
            elements.SetSize(NumOfElements);
            for (int el = 0; el < NumOfElements; ++el)
            {
               elements[el] = elements_3D[el];
            }
            NumOfBdrElements = elements_2D.size();
            boundary.SetSize(NumOfBdrElements);
            for (int el = 0; el < NumOfBdrElements; ++el)
            {
               boundary[el] = elements_2D[el];
            }
            for (size_t el = 0; el < ho_el_order_3D.size(); el++)
            {
               mesh_order = max(mesh_order, ho_el_order_3D[el]);
            }
            // discard other elements
            for (size_t el = 0; el < elements_1D.size(); ++el)
            {
               delete elements_1D[el];
            }
            for (size_t el = 0; el < elements_0D.size(); ++el)
            {
               delete elements_0D[el];
            }
         }
         else if (!elements_2D.empty())
         {
            Dim = 2;
            NumOfElements = elements_2D.size();
            elements.SetSize(NumOfElements);
            for (int el = 0; el < NumOfElements; ++el)
            {
               elements[el] = elements_2D[el];
            }
            NumOfBdrElements = elements_1D.size();
            boundary.SetSize(NumOfBdrElements);
            for (int el = 0; el < NumOfBdrElements; ++el)
            {
               boundary[el] = elements_1D[el];
            }
            for (size_t el = 0; el < ho_el_order_2D.size(); el++)
            {
               mesh_order = max(mesh_order, ho_el_order_2D[el]);
            }
            // discard other elements
            for (size_t el = 0; el < elements_0D.size(); ++el)
            {
               delete elements_0D[el];
            }
         }
         else if (!elements_1D.empty())
         {
            Dim = 1;
            NumOfElements = elements_1D.size();
            elements.SetSize(NumOfElements);
            for (int el = 0; el < NumOfElements; ++el)
            {
               elements[el] = elements_1D[el];
            }
            NumOfBdrElements = elements_0D.size();
            boundary.SetSize(NumOfBdrElements);
            for (int el = 0; el < NumOfBdrElements; ++el)
            {
               boundary[el] = elements_0D[el];
            }
            for (size_t el = 0; el < ho_el_order_1D.size(); el++)
            {
               mesh_order = max(mesh_order, ho_el_order_1D[el]);
            }
         }
         else
         {
            MFEM_ABORT("Gmsh file : no elements found");
            return;
         }

         if (mesh_order > 1)
         {
            curved = 1;
            read_gf = 0;

            // initialize mesh_geoms so we can create Nodes FE space below
            SetMeshGen();

            // Generate faces and edges so that we can define
            // FE space on the mesh
            FinalizeTopology();

            // Construct GridFunction for uniformly spaced high order coords
            FiniteElementCollection* nfec;
            FiniteElementSpace* nfes;
            nfec = new L2_FECollection(mesh_order, Dim,
                                       BasisType::ClosedUniform);
            nfes = new FiniteElementSpace(this, nfec, spaceDim,
                                          Ordering::byVDIM);
            Nodes_gf.SetSpace(nfes);
            Nodes_gf.MakeOwner(nfec);

            int o = 0;
            int el_order = 1;
            for (int el = 0; el < NumOfElements; el++)
            {
               const int *vm = NULL;
               vector<int> *ho_verts = NULL;
               switch (GetElementType(el))
               {
                  case Element::SEGMENT:
                     ho_verts = &ho_verts_1D[el];
                     el_order = ho_el_order_1D[el];
                     if (ho_lin[el_order].empty())
                     {
                        ho_lin[el_order].resize(ho_verts->size());
                        GmshHOSegmentMapping(el_order, ho_lin[el_order].data());
                     }
                     vm = ho_lin[el_order].data();
                     break;
                  case Element::TRIANGLE:
                     ho_verts = &ho_verts_2D[el];
                     el_order = ho_el_order_2D[el];
                     if (ho_tri[el_order].empty())
                     {
                        ho_tri[el_order].resize(ho_verts->size());
                        GmshHOTriangleMapping(el_order, ho_tri[el_order].data());
                     }
                     vm = ho_tri[el_order].data();
                     break;
                  case Element::QUADRILATERAL:
                     ho_verts = &ho_verts_2D[el];
                     el_order = ho_el_order_2D[el];
                     if (ho_sqr[el_order].empty())
                     {
                        ho_sqr[el_order].resize(ho_verts->size());
                        GmshHOQuadrilateralMapping(el_order, ho_sqr[el_order].data());
                     }
                     vm = ho_sqr[el_order].data();
                     break;
                  case Element::TETRAHEDRON:
                     ho_verts = &ho_verts_3D[el];
                     el_order = ho_el_order_3D[el];
                     if (ho_tet[el_order].empty())
                     {
                        ho_tet[el_order].resize(ho_verts->size());
                        GmshHOTetrahedronMapping(el_order, ho_tet[el_order].data());
                     }
                     vm = ho_tet[el_order].data();
                     break;
                  case Element::HEXAHEDRON:
                     ho_verts = &ho_verts_3D[el];
                     el_order = ho_el_order_3D[el];
                     if (ho_hex[el_order].empty())
                     {
                        ho_hex[el_order].resize(ho_verts->size());
                        GmshHOHexahedronMapping(el_order, ho_hex[el_order].data());
                     }
                     vm = ho_hex[el_order].data();
                     break;
                  case Element::WEDGE:
                     ho_verts = &ho_verts_3D[el];
                     el_order = ho_el_order_3D[el];
                     if (ho_wdg[el_order].empty())
                     {
                        ho_wdg[el_order].resize(ho_verts->size());
                        GmshHOWedgeMapping(el_order, ho_wdg[el_order].data());
                     }
                     vm = ho_wdg[el_order].data();
                     break;
                  case Element::PYRAMID:
                     ho_verts = &ho_verts_3D[el];
                     el_order = ho_el_order_3D[el];
                     if (ho_pyr[el_order].empty())
                     {
                        ho_pyr[el_order].resize(ho_verts->size());
                        GmshHOPyramidMapping(el_order, ho_pyr[el_order].data());
                     }
                     vm = ho_pyr[el_order].data();
                     break;
                  default: // Any other element type
                     MFEM_WARNING("Unsupported Gmsh element type.");
                     break;
               }
               int nv = (ho_verts) ? ho_verts->size() : 0;

               for (int v = 0; v<nv; v++)
               {
                  real_t * c = GetVertex((*ho_verts)[vm[v]]);
                  for (int d=0; d<spaceDim; d++)
                  {
                     Nodes_gf(spaceDim * (o + v) + d) = c[d];
                  }
               }
               o += nv;
            }
         }

         // Suppress warnings (MFEM_CONTRACT_VAR does not work here with nvcc):
         ++n_partitions;
         ++elem_domain;
         MFEM_CONTRACT_VAR(n_partitions);
         MFEM_CONTRACT_VAR(elem_domain);

      } // section '$Elements'
      else if (buff == "$PhysicalNames") // Named element sets
      {
         int num_names = 0;
         int mdim,num;
         string name;
         input >> num_names;
         for (int i=0; i < num_names; i++)
         {
            input >> mdim >> num;
            getline(input, name);

            // Trim leading white space
            while (!name.empty() &&
                   (*name.begin() == ' ' || *name.begin() == '\t'))
            { name.erase(0,1);}

            // Trim trailing white space
            while (!name.empty() &&
                   (*name.rbegin() == ' ' || *name.rbegin() == '\t' ||
                    *name.rbegin() == '\n' || *name.rbegin() == '\r'))
            { name.resize(name.length()-1);}

            // Remove enclosing quotes
            if ( (*name.begin() == '"' || *name.begin() == '\'') &&
                 (*name.rbegin() == '"' || *name.rbegin() == '\''))
            {
               name = name.substr(1,name.length()-2);
            }

            phys_names_by_dim[mdim][num] = name;
         }
      }
      else if (buff == "$Periodic") // Reading master/slave node pairs
      {
         curved = 1;
         read_gf = 0;
         periodic = true;

         Array<int> v2v(NumOfVertices);
         for (int i = 0; i < v2v.Size(); i++)
         {
            v2v[i] = i;
         }
         int num_per_ent;
         int num_nodes;
         input >> num_per_ent;
         getline(input, buff); // Read end-of-line
         for (int i = 0; i < num_per_ent; i++)
         {
            getline(input, buff); // Read and ignore entity dimension and tags
            getline(input, buff); // If affine mapping exist, read and ignore
            if (!strncmp(buff.c_str(), "Affine", 6))
            {
               input >> num_nodes;
            }
            else
            {
               num_nodes = atoi(buff.c_str());
            }
            for (int j=0; j<num_nodes; j++)
            {
               int slave, master;
               input >> slave >> master;
               v2v[slave - 1] = master - 1;
            }
            getline(input, buff); // Read end-of-line
         }

         // Follow existing long chains of slave->master in v2v array.
         // Upon completion of this loop, each v2v[slave] will point to a true
         // master vertex. This algorithm is useful for periodicity defined in
         // multiple directions.
         for (int slave = 0; slave < v2v.Size(); slave++)
         {
            int master = v2v[slave];
            if (master != slave)
            {
               // This loop will end if it finds a circular dependency.
               while (v2v[master] != master && master != slave)
               {
                  master = v2v[master];
               }
               if (master == slave)
               {
                  // if master and slave are the same vertex, circular dependency
                  // exists. We need to fix the problem, we choose slave.
                  v2v[slave] = slave;
               }
               else
               {
                  // the long chain has ended on the true master vertex.
                  v2v[slave] = master;
               }
            }
         }

         // Convert nodes to discontinuous GridFunction (if they aren't already)
         if (mesh_order == 1)
         {
            FinalizeTopology();
            SetMeshGen();
            SetCurvature(1, true, spaceDim, Ordering::byVDIM);
         }

         // Replace "slave" vertex indices in the element connectivity
         // with their corresponding "master" vertex indices.
         for (int i = 0; i < GetNE(); i++)
         {
            Element *el = GetElement(i);
            int *v = el->GetVertices();
            int nv = el->GetNVertices();
            for (int j = 0; j < nv; j++)
            {
               v[j] = v2v[v[j]];
            }
         }
         // Replace "slave" vertex indices in the boundary element connectivity
         // with their corresponding "master" vertex indices.
         for (int i = 0; i < GetNBE(); i++)
         {
            Element *el = GetBdrElement(i);
            int *v = el->GetVertices();
            int nv = el->GetNVertices();
            for (int j = 0; j < nv; j++)
            {
               v[j] = v2v[v[j]];
            }
         }
      }
   } // we reach the end of the file

   // Process set names
   if (phys_names_by_dim.size() > 0)
   {
      // Process boundary attribute set names
      for (auto const &bdr_attr : phys_names_by_dim[Dim-1])
      {
         if (!bdr_attribute_sets.AttributeSetExists(bdr_attr.second))
         {
            bdr_attribute_sets.CreateAttributeSet(bdr_attr.second);
         }
         bdr_attribute_sets.AddToAttributeSet(bdr_attr.second, bdr_attr.first);
      }

      // Process element attribute set names
      for (auto const &attr : phys_names_by_dim[Dim])
      {
         if (!attribute_sets.AttributeSetExists(attr.second))
         {
            attribute_sets.CreateAttributeSet(attr.second);
         }
         attribute_sets.AddToAttributeSet(attr.second, attr.first);
      }
   }

   RemoveUnusedVertices();
   if (periodic)
   {
      RemoveInternalBoundaries();
   }
   FinalizeTopology();

   // If a high order coordinate field was created project it onto the mesh
   if (mesh_order > 1)
   {
      SetCurvature(mesh_order, periodic, spaceDim, Ordering::byVDIM);

      VectorGridFunctionCoefficient NodesCoef(&Nodes_gf);
      Nodes->ProjectCoefficient(NodesCoef);
   }
}

} // namespace mfem
