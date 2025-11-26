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

/// Number of nodes in an element of type @a geom with order @a order.
static int NumNodesInElement(Geometry::Type geom, int order)
{
   return GlobGeometryRefiner.Refine(geom, order, 1)->RefPts.GetNPoints();
}

/// Enum to specify if values should be read in binary or ASCII format.
enum BinaryOrASCII : bool
{
   ASCII = false,
   BINARY = true
};

/// Helper class for reading Gmsh meshes.
class GmshReader
{
   vector<vector<int>> types =
   {
      {15}, // point
      {1, 8, 26, 27, 28, 62, 63, 64, 65, 66}, // segment
      {2, 9, 21, 23, 25, 42, 43, 44, 45, 46}, // triangle
      {3, 10, 36, 37, 38, 47, 48, 49, 50, 51}, // quadrilateral
      {4, 11, 29, 30, 31, 71, 72, 73, 74, 75}, // tetrahedron
      {5, 12, 92, 93, 94, 95, 96, 97, 98}, // hexahedron
      {6, 13, 90, 91, 106, 107, 108, 109, 110}, // prism
      {7, 14, 118, 119, 120, 121, 122, 123, 124} // pyramid
   };
   map<pair<Geometry::Type, int>, vector<int>> node_maps;

   bool has_positive_attrs = false;
   bool has_non_positive_attrs = false;

public:
   BinaryOrASCII is_binary; ///< Is the file in binary or ascii format.
   int data_size; ///< Data size in bytes (meaning depends on file format).

   /// A map between a serial number of the vertex and its number in the file
   /// (there may be gaps in the numbering, and also Gmsh enumerates vertices
   /// starting from 1, not 0)
   map<int, int> vertex_map;

   /// A map containing names of physical curves, surfaces, and volumes. The
   /// first index is the dimension of the physical manifold, the second index is
   /// the element attribute number of the set, and the string is the assigned
   /// name.
   map<int,map<int,string> > phys_names_by_dim;

   /// Gmsh always outputs coordinates in 3D, but MFEM distinguishes between the
   /// mesh element dimension (Dim) and the dimension of the space in which the
   /// mesh is embedded (spaceDim). For example, a 2D MFEM mesh has Dim = 2 and
   /// spaceDim = 2, while a 2D surface mesh in 3D has Dim = 2 but spaceDim = 3.
   /// Below we set spaceDim by measuring the mesh bounding box and checking for
   /// a lower dimensional subspace. The assumption is that the mesh is at least
   /// 2D if the y-dimension of the box is non-trivial and 3D if the z-dimension
   /// is non-trivial. Note that with these assumptions a 2D mesh parallel to the
   /// yz plane will be considered a surface mesh embedded in 3D whereas the same
   /// 2D mesh parallel to the xy plane will be considered a 2D mesh.
   ///@{
   const double inf = numeric_limits<double>::infinity();
   double bb_min[3] = {inf, inf, inf};
   double bb_max[3] = {-inf, -inf, -inf};
   ///@}

   int mesh_order = -1; /// Mesh order. Variable order meshes are not supported.
   bool periodic = false; ///< Is the mesh periodic?

   /// Node indices of high-order elements, such that ho_el_nodes[dim][e][i] is
   /// the i-th node index of the e-th element of dimension dim.
   vector<vector<vector<int>>> ho_el_nodes{4};

   vector<int> v2v; ///< Periodic vertex mapping (for periodic meshes only).

   GmshReader() = default;

   /// Get the geometry type and polynomial degree for a given Gmsh element type.
   pair<Geometry::Type, int> GetGeometryAndOrder(int element_type) const
   {
      for (int g = Geometry::POINT; g < Geometry::NUM_GEOMETRIES; ++g)
      {
         const vector<int> &types_g = types[g];
         const auto it = lower_bound(types_g.begin(), types_g.end(), element_type);
         if (it != types_g.end() && *it == element_type)
         {
            return {Geometry::Type(g), int(distance(types_g.begin(), it) + 1)};
         }
      }
      MFEM_ABORT("Unknown Gmsh element type.");
   }

   /// Return node map if it exists, otherwise lazily construct it.
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
            case Geometry::TRIANGLE: HOTriangleMapping(order, data); break;
            case Geometry::SQUARE: HOQuadrilateralMapping(order, data); break;
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

   /// Add the physical names (in @a phys_names_by_dim) to the mesh's attribute
   /// sets and boundary attribute sets.
   void AddPhysicalNames(Mesh &mesh)
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

   /// In the periodic vertex mapping @a v2v, there may be chains or cycles. This
   /// will simplify all chains so that they are one link only, and break any
   /// cycles.
   void SimplifyPeriodicLinks()
   {
      // Follow existing long chains of duplicate->primary in v2v array. Upon
      // completion of this loop, each v2v[duplicate] will point to a true
      // primary vertex. This algorithm is useful for periodicity defined in
      // multiple directions.
      for (int duplicate = 0; duplicate < v2v.size(); duplicate++)
      {
         int primary = v2v[duplicate];
         if (primary != duplicate)
         {
            // This loop will end if it finds a circular dependency.
            while (v2v[primary] != primary && primary != duplicate)
            {
               primary = v2v[primary];
            }
            if (primary == duplicate)
            {
               // If primary and duplicate are the same vertex, circular
               // dependency exists. We need to fix the problem, we choose
               // duplicate.
               v2v[duplicate] = duplicate;
            }
            else
            {
               // The long chain has ended on the true primary vertex.
               v2v[duplicate] = primary;
            }
         }
      }
   }

   /// In the list of Elements @a els, replace periodic vertices using the periodic
   /// identification map @a v2v.
   void ReplacePeriodicVertices(Array<Element*> &els) const
   {
      for (int i = 0; i < els.Size(); i++)
      {
         Element *e = els[i];
         int *v = e->GetVertices();
         for (int j = 0; j < e->GetNVertices(); j++)
         {
            v[j] = v2v[v[j]];
         }
      }
   }

   /// Set the attribute of element @a e to @a attribute. If the attribute is
   /// non-positive, set it to 1. Keep track if non-positive or positive
   /// attributes are encountered to potentially report errors to the user.
   void SetAttribute(Element *e, int attribute)
   {
      if (attribute < 1)
      {
         has_non_positive_attrs = true;
         attribute = 1; // Resetting non-positive attributes to be 1.
      }
      else
      {
         has_positive_attrs = true;
      }
      e->SetAttribute(attribute);
   }

   /// Check that all attributes are positive (or, if none are positive, give a
   /// warning that they have been replaced by 1).
   void CheckAttributes() const
   {
      if (has_non_positive_attrs)
      {
         // If mesh has a mix of positive and non-positive attributes, this is
         // a user error. All attributes should be positive.
         MFEM_VERIFY(!has_positive_attrs,
                     "Non-positive element attribute in Gmsh mesh!\n"
                     "By default Gmsh sets element tags (attributes)"
                     " to '0' but MFEM requires that they be"
                     " positive integers.\n"
                     "Use \"Physical Curve\", \"Physical Surface\","
                     " or \"Physical Volume\" to set tags/attributes"
                     " for all curves, surfaces, or volumes in your"
                     " Gmsh geometry to values which are >= 1.");
         // If the mesh has only non-positive attributes, this could be because
         // Gmsh by default will set zero attributes if no physical entities are
         // defined. In this case, we warn the user, and set attributes to 1.
         MFEM_WARNING("Gmsh reader: all element attributes were zero.\n"
                      "MFEM only supports positive element attributes.\n"
                      "Setting all element attributes to 1.\n");
      }
   }
};

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

/// Skip ahead in the input stream until the next section, which opens on a new
/// line beginning with $ (but not beginning with $End, which ends the previous
/// section).
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

/// Read a value of type @a T from the input stream, in either binary or ASCII
/// format, depending on the value of @a binary.
template <typename T>
T ReadBinaryOrASCII(istream &input, BinaryOrASCII binary)
{
   if (binary)
   {
      return bin_io::read<T>(input);
   }
   else
   {
      T val;
      input >> val;
      if (input.peek() == '\n') { input.get(); } // Chomp up to one newline
      return val;
   }
}

/// Skip @a num values of type @a T from the input stream, in either binary or
/// ASCII format, depending on the value of @a binary.
template <typename T>
void Skip(istream &input, BinaryOrASCII binary, int num)
{
   for (int i = 0; i < num; ++i) { ReadBinaryOrASCII<T>(input, binary); }
}

/// Read a double-quoted string from the input stream, and return the result
/// (without the enclosing quotes).
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

} // namespace gmsh

using namespace gmsh;

void Mesh::ReadGmsh4Mesh(GmshReader &g, istream &input, int &curved,
                         int &read_gf)
{
   MFEM_VERIFY(g.data_size == sizeof(size_t), "Incompatible Gmsh mesh.");

   const auto b = g.is_binary;

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

            g.phys_names_by_dim[phys_name_dim][phys_name_tag] = phys_name;
         }
      }
      else if (section == "Entities")
      {
         const size_t n_points = ReadBinaryOrASCII<size_t>(input, b);
         const size_t n_curves = ReadBinaryOrASCII<size_t>(input, b);
         const size_t n_surfaces = ReadBinaryOrASCII<size_t>(input, b);
         const size_t n_volumes = ReadBinaryOrASCII<size_t>(input, b);

         const size_t n_entities[4] = {n_points, n_curves, n_surfaces, n_volumes};

         if (n_volumes > 0) { Dim = 3; }
         else if (n_surfaces > 0) { Dim = 2; }
         else { Dim = 1; }

         for (int d = 0; d <= 3; ++d)
         {
            for (size_t i = 0; i < n_entities[d]; ++i)
            {
               const int tag = ReadBinaryOrASCII<int>(input, b);
               Skip<double>(input, b, d == 0 ? 3 : 6); // Skip X, Y, Z
               const int n_phys_tags = ReadBinaryOrASCII<size_t>(input, b);
               for (size_t iphys = 0; iphys < n_phys_tags; ++iphys)
               {
                  const int phys_tag = ReadBinaryOrASCII<int>(input, b);
                  // Keep track of codim-0 and codim-1 entities.
                  if (d == Dim || d == Dim - 1)
                  {
                     entity_physical_tag[ {d, tag}] = phys_tag;
                  }
               }
               if (d > 0)
               {
                  const size_t n_bounding = ReadBinaryOrASCII<size_t>(input, b);
                  Skip<int>(input, b, n_bounding);
               }
            }
         }
      }
      else if (section == "Nodes")
      {
         const size_t n_blocks = ReadBinaryOrASCII<size_t>(input, b);
         const size_t n_nodes = ReadBinaryOrASCII<size_t>(input, b);
         Skip<size_t>(input, b, 2); // Skip min and max tags

         NumOfVertices = n_nodes;
         vertices.SetSize(n_nodes);
         size_t vertex_counter = 0;

         double c[3];

         for (int iblock = 0; iblock < n_blocks; ++iblock)
         {
            Skip<int>(input, b, 2); // Skip entity dim and tag.
            const int is_parametric = ReadBinaryOrASCII<int>(input, b);
            const size_t n_nodes_in_block = ReadBinaryOrASCII<size_t>(input, b);

            MFEM_VERIFY(!is_parametric, "Parametric nodes not supported.");

            vector<size_t> node_tags(n_nodes_in_block);
            for (int i = 0; i < n_nodes_in_block; ++i)
            {
               const size_t node_tag = ReadBinaryOrASCII<size_t>(input, b);
               node_tags[i] = node_tag;
            }
            for (int i = 0; i < n_nodes_in_block; ++i)
            {
               for (int d = 0; d < 3; ++d)
               {
                  c[d] = ReadBinaryOrASCII<double>(input, b);
                  g.bb_min[d] = min(g.bb_min[d], c[d]);
                  g.bb_max[d] = max(g.bb_min[d], c[d]);
               }
               g.vertex_map[node_tags[i]] = vertex_counter;
               vertices[vertex_counter] = Vertex(c[0], c[1], c[2]);
               vertex_counter += 1;
            }
         }
         spaceDim = GetSpaceDimension(g.bb_min, g.bb_max);
      }
      else if (section == "Elements")
      {
         const size_t n_blocks = ReadBinaryOrASCII<size_t>(input, b);
         Skip<size_t>(input, b, 3); // Skip n_elements and min/max tags.

         for (int iblock = 0; iblock < n_blocks; ++iblock)
         {
            const int entity_dim = ReadBinaryOrASCII<int>(input, b);
            const int entity_tag = ReadBinaryOrASCII<int>(input, b);
            const int element_type = ReadBinaryOrASCII<int>(input, b);
            const size_t n_elements = ReadBinaryOrASCII<size_t>(input, b);

            for (int ie = 0; ie < n_elements; ++ie)
            {
               Skip<size_t>(input, b, 1); // Skip element tag
               const auto [geom, el_order] = g.GetGeometryAndOrder(element_type);

               if (g.mesh_order < 0) { g.mesh_order = el_order; }
               MFEM_VERIFY(g.mesh_order == el_order,
                           "Variable order Gmsh meshes are not supported");

               const int n_elem_nodes = NumNodesInElement(geom, el_order);
               vector<size_t> node_tags(n_elem_nodes);
               for (int inode = 0; inode < n_elem_nodes; ++inode)
               {
                  node_tags[inode] = ReadBinaryOrASCII<size_t>(input, b);
               }

               // We only add codim-0 and codim-1 elements.
               if (entity_dim != Dim && entity_dim != Dim - 1) { continue; }

               auto e = NewElement(geom);
               Array<int> v(e->GetNVertices());
               for (int i = 0; i < e->GetNVertices(); ++i)
               {
                  v[i] = g.vertex_map[node_tags[i]];
               }
               e->SetVertices(v);
               g.SetAttribute(e, entity_physical_tag[ {entity_dim, entity_tag}]);

               if (entity_dim == Dim) { elements.Append(e); }
               else if (entity_dim == Dim - 1) { boundary.Append(e); }

               // Store high-order node locations
               if (el_order > 1 && entity_dim == Dim)
               {
                  const vector<int> &map = g.GetNodeMap(geom, el_order);
                  auto &nodes = g.ho_el_nodes[Dim].emplace_back(n_elem_nodes);
                  for (int i = 0; i < n_elem_nodes; ++i)
                  {
                     nodes[i] = g.vertex_map[node_tags[map[i]]];
                  }
               }
            }
         }
         NumOfElements = elements.Size();
         NumOfBdrElements = boundary.Size();
      }
      else if (section == "Periodic")
      {
         const size_t n_periodic = ReadBinaryOrASCII<size_t>(input, b);
         if (n_periodic == 0) { continue; }

         g.periodic = true;
         g.v2v.resize(NumOfVertices);
         for (int i = 0; i < NumOfVertices; i++) { g.v2v[i] = i; }

         for (int i = 0; i < n_periodic; ++i)
         {
            Skip<int>(input, b, 3); // Skip entity information
            const int n_affine = ReadBinaryOrASCII<size_t>(input, b);
            Skip<double>(input, b, n_affine); // Skip affine information
            const size_t n_nodes = ReadBinaryOrASCII<size_t>(input, b);
            for (int j = 0; j < n_nodes; ++j)
            {
               const int node_num = ReadBinaryOrASCII<int>(input, b);
               const int primary_node_num = ReadBinaryOrASCII<int>(input, b);
               g.v2v[node_num - 1] = primary_node_num - 1;
            }
         }
      }
   }
   while (!section.empty());
}

void Mesh::ReadGmsh2Mesh(GmshReader &g, istream &input, int &curved,
                         int &read_gf)
{
   const auto b = g.is_binary;
   MFEM_VERIFY(g.data_size == sizeof(double), "Incompatible data size.");

   string section;
   do
   {
      section = GoToNextSection(input);
      if (section == "Nodes")
      {
         NumOfVertices = ReadBinaryOrASCII<int>(input, ASCII);
         vertices.SetSize(NumOfVertices);
         double c[3];
         for (int v = 0; v < NumOfVertices; ++v)
         {
            const int node_num = ReadBinaryOrASCII<int>(input, b);
            for (int d = 0; d < 3; ++d)
            {
               c[d] = ReadBinaryOrASCII<double>(input, b);
               g.bb_min[d] = min(g.bb_min[d], c[d]);
               g.bb_max[d] = max(g.bb_max[d], c[d]);
            }
            vertices[v] = Vertex(c[0], c[1], c[2]);
            g.vertex_map[node_num] = v;
         }
         spaceDim = GetSpaceDimension(g.bb_min, g.bb_max);
         MFEM_VERIFY(g.vertex_map.size() == NumOfVertices,
                     "Gmsh node indices are not unique.");
      }
      else if (section == "Elements")
      {
         const int num_elements = ReadBinaryOrASCII<int>(input, ASCII);
         int num_el_read = 0;

         vector<vector<unique_ptr<Element>>> elems_by_dim(4);

         while (num_el_read < num_elements)
         {
            auto add_element = [&](int el_type, int el_phys_tag, Geometry::Type geom,
                                   int el_order, const vector<int> &el_nodes)
            {
               if (g.mesh_order < 0) { g.mesh_order = el_order; }
               MFEM_VERIFY(g.mesh_order == el_order,
                           "Variable order Gmsh meshes are not supported");

               const int dim = Geometry::Dimension[geom];
               Element *e = NewElement(geom);
               Array<int> v(e->GetNVertices());
               for (int i = 0; i < e->GetNVertices(); ++i)
               {
                  v[i] = g.vertex_map[el_nodes[i]];
               }
               e->SetVertices(v);
               g.SetAttribute(e, el_phys_tag);

               elems_by_dim[dim].emplace_back(e);

               if (el_order > 1)
               {
                  const vector<int> &map = g.GetNodeMap(geom, el_order);
                  auto &nodes = g.ho_el_nodes[dim].emplace_back(el_nodes.size());
                  for (int i = 0; i < el_nodes.size(); ++i)
                  {
                     nodes[i] = g.vertex_map[el_nodes[map[i]]];
                  }
               }
            };

            if (b)
            {
               // Header
               const int el_type  = ReadBinaryOrASCII<int>(input, BINARY);
               const int n_els  = ReadBinaryOrASCII<int>(input, BINARY);
               const int n_tags = ReadBinaryOrASCII<int>(input, BINARY);
               const auto [geom, el_order] = g.GetGeometryAndOrder(el_type);
               const int n_el_nodes = NumNodesInElement(geom, el_order);
               vector<int> el_nodes(n_el_nodes);
               // Element blocks
               for (int e = 0; e < n_els; ++e)
               {
                  Skip<int>(input, BINARY, 1); // Skip element number
                  int el_phys_tag = 0;
                  if (n_tags > 0)
                  {
                     el_phys_tag = ReadBinaryOrASCII<int>(input, BINARY);
                     Skip<int>(input, BINARY, n_tags - 1);
                  }
                  for (int i = 0; i < n_el_nodes; ++i)
                  {
                     el_nodes[i] = ReadBinaryOrASCII<int>(input, BINARY);
                  }
                  add_element(el_type, el_phys_tag, geom, el_order, el_nodes);
                  num_el_read += 1;
               }
            }
            else
            {
               Skip<int>(input, ASCII, 1); // Skip element number
               const int el_type = ReadBinaryOrASCII<int>(input, ASCII);
               const int n_tags = ReadBinaryOrASCII<int>(input, ASCII);
               int el_phys_tag = 0;
               if (n_tags > 0)
               {
                  el_phys_tag = ReadBinaryOrASCII<int>(input, ASCII);
                  Skip<int>(input, ASCII, n_tags - 1);
               }
               const auto [geom, el_order] = g.GetGeometryAndOrder(el_type);
               const int n_el_nodes = NumNodesInElement(geom, el_order);
               vector<int> el_nodes(n_el_nodes);
               for (int i = 0; i < n_el_nodes; ++i)
               {
                  el_nodes[i] = ReadBinaryOrASCII<int>(input, ASCII);
               }
               add_element(el_type, el_phys_tag, geom, el_order, el_nodes);
               num_el_read += 1;
            }
         }

         if (elems_by_dim[3].size() > 0) { Dim = 3; }
         else if (elems_by_dim[2].size() > 0) { Dim = 2; }
         else { Dim = 1; }

         NumOfElements = elems_by_dim[Dim].size();
         elements.SetSize(NumOfElements);
         for (int i = 0; i < NumOfElements; ++i)
         {
            elements[i] = elems_by_dim[Dim][i].release();
         }
         NumOfBdrElements = elems_by_dim[Dim - 1].size();
         boundary.SetSize(NumOfBdrElements);
         for (int i = 0; i < NumOfBdrElements; ++i)
         {
            boundary[i] = elems_by_dim[Dim - 1][i].release();
         }
      }
      else if (section == "PhysicalNames")
      {
         const int num_names = ReadBinaryOrASCII<int>(input, ASCII);
         for (int i = 0; i < num_names; ++i)
         {
            const int phys_dim = ReadBinaryOrASCII<int>(input, ASCII);
            const int phys_tag = ReadBinaryOrASCII<int>(input, ASCII);
            g.phys_names_by_dim[phys_dim][phys_tag] = ReadQuotedString(input);
         }
      }
      else if (section == "Periodic")
      {
         const int n_periodic_entities = ReadBinaryOrASCII<int>(input, ASCII);
         if (n_periodic_entities == 0) { continue; }

         g.periodic = true;
         g.v2v.resize(NumOfVertices);
         for (int i = 0; i < NumOfVertices; i++) { g.v2v[i] = i; }

         for (int i = 0; i < n_periodic_entities; i++)
         {
            Skip<int>(input, ASCII, 3); // Skip dimension, tag, and master tag
            // Next section might be "Affine"; if so, skip.
            auto pos = input.tellg();
            if (ReadBinaryOrASCII<string>(input, ASCII) == "Affine")
            {
               string line;
               getline(input, line);
            }
            else
            {
               input.clear();
               input.seekg(pos);
            }
            const int n_nodes = ReadBinaryOrASCII<int>(input, ASCII);
            for (int j = 0; j < n_nodes; ++j)
            {
               const int node_num = ReadBinaryOrASCII<int>(input, ASCII);
               const int primary_node_num = ReadBinaryOrASCII<int>(input, ASCII);
               g.v2v[node_num - 1] = primary_node_num - 1;
            }
         }
      }
   }
   while (section != "");
}

void Mesh::ReadGmshMesh(istream &input, int &curved, int &read_gf)
{
   const real_t version = ReadBinaryOrASCII<real_t>(input, ASCII);

   MFEM_VERIFY(version == 2.2 || version == 4.1,
               "Unsupported Gmsh file version. Supported versions: 2.2 and 4.1");

   GmshReader g;
   g.is_binary = BinaryOrASCII(ReadBinaryOrASCII<bool>(input, ASCII));
   g.data_size = ReadBinaryOrASCII<int>(input, ASCII);

   if (g.is_binary)
   {
      const int one = ReadBinaryOrASCII<int>(input, BINARY);
      MFEM_VERIFY(one == 1, "Incompatible endianness.");
   }

   // Versions supported: 4.1 and 2.2
   if (version == 4.1)
   {
      ReadGmsh4Mesh(g, input, curved, read_gf);
   }
   else if (version == 2.2)
   {
      ReadGmsh2Mesh(g, input, curved, read_gf);
   }

   // Make sure all element and boundary attributes are positive.
   g.CheckAttributes();

   // Merge periodic vertices
   if (g.periodic)
   {
      g.SimplifyPeriodicLinks();
      // If the mesh is low-order, we need to populate g.ho_el_nodes before
      // periodic vertices are identified in order to set the L2 nodes grid
      // function.
      if (g.mesh_order == 1)
      {
         g.ho_el_nodes[Dim].resize(NumOfElements);
         for (int ie = 0; ie < NumOfElements; ++ie)
         {
            const Element *e = elements[ie];
            const int nv = e->GetNVertices();
            const int *v = e->GetVertices();
            g.ho_el_nodes[Dim][ie].resize(nv);
            for (int i = 0; i < nv; ++i)
            {
               g.ho_el_nodes[Dim][ie][i] = v[i];
            }
         }
      }
      g.ReplacePeriodicVertices(elements);
      g.ReplacePeriodicVertices(boundary);
   }

   // If the elements are high-order, keep a copy of the nodes before removing
   // unused vertices.
   Array<Vertex> ho_vertices;
   if (g.mesh_order > 1 || g.periodic) { ho_vertices = vertices; }

   g.AddPhysicalNames(*this);
   RemoveUnusedVertices();
   FinalizeTopology();

   // Now that the mesh topology has been fully created, set the high-order
   // nodal information (if needed). For periodic meshes, we need to set the
   // L2 grid function.
   if (g.mesh_order > 1 || g.periodic)
   {
      SetCurvature(g.mesh_order, g.periodic, spaceDim, Ordering::byVDIM);
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
            const int ii = lex.IsEmpty() ? i : lex[i];
            Vertex v = ho_vertices[g.ho_el_nodes[Dim][e][i]];
            for (int d = 0; d < spaceDim; ++d)
            {
               nodes_gf[vdofs[ii + d*n]] = v(d);
            }
         }
      }
   }
}

} // namespace mfem
