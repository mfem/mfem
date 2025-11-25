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

namespace gmsh
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

static void HOSegmentMapping(int order, int *map)
{
   map[0] = 0;
   map[order] = 1;
   for (int i=1; i<order; i++)
   {
      map[i] = i + 1;
   }
}

static void HOTriangleMapping(int order, int *map)
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

static void HOQuadrilateralMapping(int order, int *map)
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

static void HOTetrahedronMapping(int order, int *map)
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

static void HOHexahedronMapping(int order, int *map)
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

static void HOWedgeMapping(int order, int *map)
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

static void HOPyramidMapping(int order, int *map)
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

static int NumNodesInElement(Geometry::Type geom, int order)
{
   return GlobGeometryRefiner.Refine(geom, order, 1)->RefPts.GetNPoints();
}

static void AddPhysicalNames(Mesh &mesh,
                             map<int, map<int, string>> &phys_names_by_dim)
{
   // Process boundary attribute set names
   for (auto const &bdr_attr : phys_names_by_dim[mesh.Dimension() - 1])
   {
      if (!mesh.bdr_attribute_sets.AttributeSetExists(bdr_attr.second))
      {
         mesh.bdr_attribute_sets.CreateAttributeSet(bdr_attr.second);
      }
      mesh.bdr_attribute_sets.AddToAttributeSet(bdr_attr.second, bdr_attr.first);
   }

   // Process element attribute set names
   for (auto const &attr : phys_names_by_dim[mesh.Dimension()])
   {
      if (!mesh.attribute_sets.AttributeSetExists(attr.second))
      {
         mesh.attribute_sets.CreateAttributeSet(attr.second);
      }
      mesh.attribute_sets.AddToAttributeSet(attr.second, attr.first);
   }
}

/// Helper object that caches data used in Gmsh mesh creation.
class Gmsh
{
   vector<vector<int>> types;
   map<pair<Geometry::Type, int>, vector<int>> node_maps;
public:
   Gmsh()
      : types(
   {
      {15}, // point
      {1, 8, 26, 27, 28, 62, 63, 64, 65, 66}, // segment
      {2, 9, 21, 23, 25, 42, 43, 44, 45, 46}, // triangle
      {3, 10, 36, 37, 38, 47, 48, 49, 50, 51}, // quadrilateral
      {4, 11, 29, 30, 31, 71, 72, 73, 74, 75}, // tetrahedron
      {5, 12, 92, 93, 94, 95, 96, 97, 98}, // hexahedron
      {6, 13, 90, 91, 106, 107, 108, 109, 110}, // prism
      {7, 14, 118, 119, 120, 121, 122, 123, 124} // pyramid
   }) { }

   pair<Geometry::Type, int> GetGeometryAndOrder(int element_type)
   {
      for (int g = Geometry::POINT; g < Geometry::NUM_GEOMETRIES; ++g)
      {
         vector<int> &types_g = types[g];
         const auto it = lower_bound(types_g.begin(), types_g.end(), element_type);
         if (it != types_g.end() && *it == element_type)
         {
            return {Geometry::Type(g), int(distance(types_g.begin(), it) + 1)};
         }
      }
      MFEM_WARNING("Unknown Gmsh element type.");
      return {Geometry::INVALID, 0};
   }

   int GetNumNodesByType(int element_type)
   {
      auto [geom, order] = GetGeometryAndOrder(element_type);
      return NumNodesInElement(geom, order);
   }

   const vector<int> &GetNodeMap(Geometry::Type geom, int order)
   {
      auto it = node_maps.find(make_pair(geom, order));
      if (it == node_maps.end())
      {
         const int n_nodes = NumNodesInElement(geom, order);
         auto ret = node_maps.emplace(piecewise_construct,
                                      forward_as_tuple(geom, order),
                                      forward_as_tuple(n_nodes));
         auto &map = ret.first->second;
         auto data = map.data();
         switch (geom)
         {
            case Geometry::SEGMENT: HOSegmentMapping(order, data); break;
            case Geometry::TRIANGLE: HOSegmentMapping(order, data); break;
            case Geometry::SQUARE: HOSegmentMapping(order, data); break;
            case Geometry::TETRAHEDRON: HOTetrahedronMapping(order, data); break;
            case Geometry::CUBE: HOHexahedronMapping(order, data); break;
            case Geometry::PRISM: HOWedgeMapping(order, data); break;
            case Geometry::PYRAMID: HOPyramidMapping(order, data); break;
            default: break; // Unknown element type
         }
         return map;
      }
      else
      {
         return it->second;
      }
   }
};

enum BinaryOrASCII : bool
{
   ASCII = false,
   BINARY = true
};

static string GoToNextSection(istream &input)
{
   string line;
   while (getline(input, line))
   {
      // Find the next line that starts with '$', but does not start with "$End"
      if (line.size() >= 1 &&
          line[0] == '$' &&
          (line.size() < 4 || line.compare(1, 3, "End") != 0))
      {
         return line.substr(1, string::npos);
      }
   }
   return "";
}

template <typename T>
T ReadBinaryOrASCII(istream &input, gmsh::BinaryOrASCII binary)
{
   if (binary)
   {
      return bin_io::read<T>(input);
   }
   else
   {
      T val;
      input >> val;
      return val;
   }
}

template <typename T>
void Skip(istream &input, gmsh::BinaryOrASCII binary, int num)
{
   for (int i = 0; i < num; ++i) { ReadBinaryOrASCII<T>(input, binary); }
}

static string ReadQuotedString(istream &input)
{
   char c;
   // Find opening quote
   while (input.get(c))
   {
      if (c == '"') { break; }
   }
   MFEM_VERIFY(input, "Error reading string.");

   string result;
   while (input.get(c))
   {
      // Find closing quote
      if (c == '"')
      {
         return result;
      }
      result.push_back(c);
   }

   MFEM_ABORT("Failed to read string.");
}

/// @brief Return the space dimension (at least 1) given a 3D bounding box.
///
/// If some of the sides of the box have zero (or very small) sides, then that
/// dimension is not counted.
static int GetSpaceDimension(double bb_min[3], double bb_max[3])
{
   static constexpr double bb_tol = 1e-14;
   const double bb_size = max(bb_max[0] - bb_min[0],
                              max(bb_max[1] - bb_min[1],
                                  bb_max[2] - bb_min[2]));
   int sd = 1;
   if (bb_max[1] - bb_min[1] > bb_size * bb_tol)
   {
      sd += 1;
   }
   if (bb_max[2] - bb_min[2] > bb_size * bb_tol)
   {
      sd += 1;
   }
   return sd;
}

} // namespace gmsh

void Mesh::ReadGmsh4Mesh(istream &input, int &curved, int &read_gf)
{
   using namespace gmsh;

   // $MeshFormat is always encoded in ASCII
   const BinaryOrASCII is_binary = BinaryOrASCII(
                                      ReadBinaryOrASCII<bool>(input, ASCII));
   const int data_size = ReadBinaryOrASCII<int>(input, ASCII);

   MFEM_VERIFY(data_size == sizeof(size_t), "Incompatible Gmsh mesh.");

   bool periodic = false;
   int mesh_order = -1;
   unordered_map<int, int> vertex_map;
   map<int, map<int, string>> phys_names_by_dim;
   map<pair<int,int>, int> entity_physical_tag;

   vector<vector<int>> el_nodes;

   string section;
   do
   {
      section = GoToNextSection(input);
      if (section == "PhysicalNames")
      {
         // $PhysicalNames is always encoded in ASCII
         const int n_phys_names = ReadBinaryOrASCII<int>(input, ASCII);
         for (int i = 0; i < n_phys_names; ++i)
         {
            const int phys_name_dim = ReadBinaryOrASCII<int>(input, ASCII);
            const int phys_name_tag = ReadBinaryOrASCII<int>(input, ASCII);
            const string phys_name = ReadQuotedString(input);

            phys_names_by_dim[phys_name_dim][phys_name_tag] = phys_name;
         }
      }
      else if (section == "Entities")
      {
         const size_t n_points = ReadBinaryOrASCII<size_t>(input, is_binary);
         const size_t n_curves = ReadBinaryOrASCII<size_t>(input, is_binary);
         const size_t n_surfaces = ReadBinaryOrASCII<size_t>(input, is_binary);
         const size_t n_volumes = ReadBinaryOrASCII<size_t>(input, is_binary);

         const size_t n_entities[4] = {n_points, n_curves, n_surfaces, n_volumes};

         if (n_volumes > 0) { Dim = 3; }
         else if (n_surfaces > 0) { Dim = 2; }
         else { Dim = 1; }

         for (int d = 0; d <= 3; ++d)
         {
            for (size_t i = 0; i < n_entities[d]; ++i)
            {
               const int tag = ReadBinaryOrASCII<int>(input, is_binary);
               Skip<double>(input, is_binary, d == 0 ? 3 : 6); // Skip X, Y, Z
               const int n_phys_tags = ReadBinaryOrASCII<size_t>(input, is_binary);
               for (size_t iphys = 0; iphys < n_phys_tags; ++iphys)
               {
                  const int phys_tag = ReadBinaryOrASCII<int>(input, is_binary);
                  // Keep track of codim-0 and codim-1 entities.
                  if (d == Dim || d == Dim - 1)
                  {
                     entity_physical_tag[ {d, tag}] = phys_tag;
                  }
               }
               if (d > 0)
               {
                  const size_t n_bounding = ReadBinaryOrASCII<size_t>(input, is_binary);
                  Skip<int>(input, is_binary, n_bounding);
               }
            }
         }
      }
      else if (section == "Nodes")
      {
         const size_t n_blocks = ReadBinaryOrASCII<size_t>(input, is_binary);
         const size_t n_nodes = ReadBinaryOrASCII<size_t>(input, is_binary);
         Skip<size_t>(input, is_binary, 2); // Skip min and max tags

         NumOfVertices = n_nodes;
         vertices.SetSize(n_nodes);
         size_t vertex_counter = 0;

         const double inf = numeric_limits<double>::infinity();
         double bb_min[3] = {inf, inf, inf};
         double bb_max[3] = {-inf, -inf, -inf};
         double c[3];

         for (int iblock = 0; iblock < n_blocks; ++iblock)
         {
            Skip<int>(input, is_binary, 2); // Skip entity dim and tag.
            const int is_parametric = ReadBinaryOrASCII<int>(input, is_binary);
            const size_t n_nodes_in_block = ReadBinaryOrASCII<size_t>(input, is_binary);

            MFEM_VERIFY(!is_parametric, "Parametric nodes not supported.");

            vector<size_t> node_tags(n_nodes_in_block);
            for (int i = 0; i < n_nodes_in_block; ++i)
            {
               const size_t node_tag = ReadBinaryOrASCII<size_t>(input, is_binary);
               node_tags[i] = node_tag;
            }
            for (int i = 0; i < n_nodes_in_block; ++i)
            {
               for (int d = 0; d < 3; ++d)
               {
                  c[d] = ReadBinaryOrASCII<double>(input, is_binary);
                  bb_min[d] = min(bb_min[d], c[d]);
                  bb_max[d] = max(bb_min[d], c[d]);
               }
               vertex_map[node_tags[i]] = vertex_counter;
               vertices[vertex_counter] = Vertex(c[0], c[1], c[2]);
               vertex_counter += 1;
            }
         }
         spaceDim = GetSpaceDimension(bb_min, bb_max);
      }
      else if (section == "Elements")
      {
         Gmsh g;

         const size_t n_blocks = ReadBinaryOrASCII<size_t>(input, is_binary);
         Skip<size_t>(input, is_binary, 3); // Skip n_elements and min/max tags.

         for (int iblock = 0; iblock < n_blocks; ++iblock)
         {
            const int entity_dim = ReadBinaryOrASCII<int>(input, is_binary);
            const int entity_tag = ReadBinaryOrASCII<int>(input, is_binary);
            const int element_type = ReadBinaryOrASCII<int>(input, is_binary);
            const size_t n_elements = ReadBinaryOrASCII<size_t>(input, is_binary);

            for (int ie = 0; ie < n_elements; ++ie)
            {
               Skip<size_t>(input, is_binary, 1); // Skip element tag
               const auto [geom, el_order] = g.GetGeometryAndOrder(element_type);

               if (mesh_order < 0) { mesh_order = el_order; }
               MFEM_VERIFY(mesh_order == el_order,
                           "Variable order Gmsh meshes are not supported");

               const int n_elem_nodes = NumNodesInElement(geom, el_order);
               vector<size_t> node_tags(n_elem_nodes);
               for (int inode = 0; inode < n_elem_nodes; ++inode)
               {
                  node_tags[inode] = ReadBinaryOrASCII<size_t>(input, is_binary);
               }

               // We only add codim-0 and codim-1 elements.
               if (entity_dim != Dim && entity_dim != Dim - 1) { continue; }

               auto e = NewElement(geom);
               Array<int> v(e->GetNVertices());
               for (int i = 0; i < e->GetNVertices(); ++i)
               {
                  v[i] = vertex_map[node_tags[i]];
               }
               e->SetVertices(v);
               e->SetAttribute(entity_physical_tag[ {entity_dim, entity_tag}]);

               if (entity_dim == Dim) { elements.Append(e); }
               else if (entity_dim == Dim - 1) { boundary.Append(e); }

               // Store high-order node locations
               if (el_order > 1 && entity_dim == Dim)
               {
                  const vector<int> &map = g.GetNodeMap(geom, el_order);
                  vector<int> &nodes = el_nodes.emplace_back(n_elem_nodes);
                  for (int i = 0; i < n_elem_nodes; ++i)
                  {
                     nodes[i] = vertex_map[node_tags[map[i]]];
                  }
               }
            }
         }
         NumOfElements = elements.Size();
         NumOfBdrElements = boundary.Size();
      }
      else if (section == "Periodic")
      {
         const size_t n_periodic = ReadBinaryOrASCII<size_t>(input, is_binary);
         if (n_periodic > 0) { periodic = true; }

         MFEM_ABORT("Not currently supported");
      }
   }
   while (!section.empty());

   // If the elements are high-order, keep a copy of the nodes before removing
   // unused vertices.
   Array<Vertex> ho_vertices;
   if (mesh_order > 1) { ho_vertices = vertices; }

   AddPhysicalNames(*this, phys_names_by_dim);
   RemoveUnusedVertices();
   RemoveInternalBoundaries();
   FinalizeTopology();

   // Now that the mesh topology has been fully created, set the high-order
   // nodal information (if needed). For periodic meshes, we need to set the
   // L2 grid function.
   if (mesh_order > 1 || periodic)
   {
      SetCurvature(mesh_order, periodic, spaceDim, Ordering::byVDIM);
      const FiniteElementSpace &fes = *GetNodalFESpace();
      GridFunction &nodes_gf = *GetNodes();

      Array<int> vdofs;
      for (int e = 0; e < NumOfElements; ++e)
      {
         const FiniteElement *fe = fes.GetFE(e);
         auto *nfe = dynamic_cast<const NodalFiniteElement*>(fe);
         MFEM_ASSERT(nfe, "Invalid FE");
         const Array<int> &lex = nfe->GetLexicographicOrdering();
         fes.GetElementVDofs(e, vdofs);
         const int n = vdofs.Size() / spaceDim;
         for (int i = 0; i < n; ++i)
         {
            Vertex v = ho_vertices[el_nodes[e][i]];
            for (int d = 0; d < spaceDim; ++d)
            {
               nodes_gf[vdofs[lex[i] + d*n]] = v(d);
            }
         }
      }
   }
}

void Mesh::ReadGmsh2Mesh(istream &input, int &curved, int &read_gf)
{
   using namespace gmsh;

   int binary, dsize;
   input >> binary >> dsize;

   MFEM_VERIFY(dsize == sizeof(double), "Gmsh file : dsize != sizeof(double)");

   string buff;

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
   map<int,map<int,string> > phys_names_by_dim;

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
   double bb_min[3];
   double bb_max[3];

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
         double coord[gmsh_dim];
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
            vertices[ver] = Vertex(coord[0], coord[1], coord[2]);
            vertices_map[serial_number] = ver;

            for (int ci = 0; ci < gmsh_dim; ++ci)
            {
               bb_min[ci] = (ver == 0) ? coord[ci] :
                            min(bb_min[ci], coord[ci]);
               bb_max[ci] = (ver == 0) ? coord[ci] :
                            max(bb_max[ci], coord[ci]);
            }
         }
         spaceDim = GetSpaceDimension(bb_min, bb_max);

         if (static_cast<int>(vertices_map.size()) != NumOfVertices)
         {
            MFEM_ABORT("Gmsh file : vertices indices are not unique");
         }
      } // section '$Nodes'
      else if (buff == "$Elements") // reading mesh elements
      {
         Gmsh g;

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

               const auto [geom, el_order] = g.GetGeometryAndOrder(type_of_element);
               const int n_elem_nodes = NumNodesInElement(geom, el_order);
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

               const auto [geom, el_order] = g.GetGeometryAndOrder(type_of_element);
               const int n_elem_nodes = NumNodesInElement(geom, el_order);
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
                        HOSegmentMapping(el_order, ho_lin[el_order].data());
                     }
                     vm = ho_lin[el_order].data();
                     break;
                  case Element::TRIANGLE:
                     ho_verts = &ho_verts_2D[el];
                     el_order = ho_el_order_2D[el];
                     if (ho_tri[el_order].empty())
                     {
                        ho_tri[el_order].resize(ho_verts->size());
                        HOTriangleMapping(el_order, ho_tri[el_order].data());
                     }
                     vm = ho_tri[el_order].data();
                     break;
                  case Element::QUADRILATERAL:
                     ho_verts = &ho_verts_2D[el];
                     el_order = ho_el_order_2D[el];
                     if (ho_sqr[el_order].empty())
                     {
                        ho_sqr[el_order].resize(ho_verts->size());
                        HOQuadrilateralMapping(el_order, ho_sqr[el_order].data());
                     }
                     vm = ho_sqr[el_order].data();
                     break;
                  case Element::TETRAHEDRON:
                     ho_verts = &ho_verts_3D[el];
                     el_order = ho_el_order_3D[el];
                     if (ho_tet[el_order].empty())
                     {
                        ho_tet[el_order].resize(ho_verts->size());
                        HOTetrahedronMapping(el_order, ho_tet[el_order].data());
                     }
                     vm = ho_tet[el_order].data();
                     break;
                  case Element::HEXAHEDRON:
                     ho_verts = &ho_verts_3D[el];
                     el_order = ho_el_order_3D[el];
                     if (ho_hex[el_order].empty())
                     {
                        ho_hex[el_order].resize(ho_verts->size());
                        HOHexahedronMapping(el_order, ho_hex[el_order].data());
                     }
                     vm = ho_hex[el_order].data();
                     break;
                  case Element::WEDGE:
                     ho_verts = &ho_verts_3D[el];
                     el_order = ho_el_order_3D[el];
                     if (ho_wdg[el_order].empty())
                     {
                        ho_wdg[el_order].resize(ho_verts->size());
                        HOWedgeMapping(el_order, ho_wdg[el_order].data());
                     }
                     vm = ho_wdg[el_order].data();
                     break;
                  case Element::PYRAMID:
                     ho_verts = &ho_verts_3D[el];
                     el_order = ho_el_order_3D[el];
                     if (ho_pyr[el_order].empty())
                     {
                        ho_pyr[el_order].resize(ho_verts->size());
                        HOPyramidMapping(el_order, ho_pyr[el_order].data());
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
            phys_names_by_dim[mdim][num] = ReadQuotedString(input);
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

   AddPhysicalNames(*this, phys_names_by_dim);
   RemoveUnusedVertices();
   RemoveInternalBoundaries();
   FinalizeTopology();

   // If a high order coordinate field was created project it onto the mesh
   if (mesh_order > 1)
   {
      SetCurvature(mesh_order, periodic, spaceDim, Ordering::byVDIM);

      VectorGridFunctionCoefficient NodesCoef(&Nodes_gf);
      Nodes->ProjectCoefficient(NodesCoef);
   }
}

void Mesh::ReadGmshMesh(istream &input, int &curved, int &read_gf)
{
   real_t version;
   input >> version;

   // Versions supported: 4.0, 4.1 and 2.2
   if (version == 4.0 || version == 4.1)
   {
      ReadGmsh4Mesh(input, curved, read_gf);
      return;
   }
   else if (version == 2.2)
   {
      ReadGmsh2Mesh(input, curved, read_gf);
   }
   else
   {
      MFEM_ABORT("Unsupported Gmsh file version. "
                 "Supported versions: 2.2, 4.0, 4.1");
   }
}

} // namespace mfem
