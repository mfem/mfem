// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
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
#include "../fem/fem.hpp"
#include "../general/binaryio.hpp"
#include "../general/text.hpp"
#include "../general/tinyxml2.h"
#include "gmsh.hpp"

#include <iostream>
#include <cstdio>
#include <vector>
#include <algorithm>

#ifdef MFEM_USE_NETCDF
#include "netcdf.h"
#endif

#ifdef MFEM_USE_ZLIB
#include <zlib.h>
#endif

using namespace std;

namespace mfem
{

bool Mesh::remove_unused_vertices = true;

void Mesh::ReadMFEMMesh(std::istream &input, int version, int &curved)
{
   // Read MFEM mesh v1.0, v1.2, or v1.3 format
   MFEM_VERIFY(version == 10 || version == 12 || version == 13,
               "unknown MFEM mesh version");

   string ident;

   // read lines beginning with '#' (comments)
   skip_comment_lines(input, '#');
   input >> ident; // 'dimension'

   MFEM_VERIFY(ident == "dimension", "invalid mesh file");
   input >> Dim;

   skip_comment_lines(input, '#');
   input >> ident; // 'elements'

   MFEM_VERIFY(ident == "elements", "invalid mesh file");
   input >> NumOfElements;
   elements.SetSize(NumOfElements);
   for (int j = 0; j < NumOfElements; j++)
   {
      elements[j] = ReadElement(input);
   }

   if (version == 13)
   {
      skip_comment_lines(input, '#');
      input >> ident; // 'attribute_sets'

      MFEM_VERIFY(ident == "attribute_sets", "invalid mesh file");

      attribute_sets.attr_sets.Load(input);
      attribute_sets.attr_sets.SortAll();
      attribute_sets.attr_sets.UniqueAll();
   }

   skip_comment_lines(input, '#');
   input >> ident; // 'boundary'

   MFEM_VERIFY(ident == "boundary", "invalid mesh file");
   input >> NumOfBdrElements;
   boundary.SetSize(NumOfBdrElements);
   for (int j = 0; j < NumOfBdrElements; j++)
   {
      boundary[j] = ReadElement(input);
   }

   if (version == 13)
   {
      skip_comment_lines(input, '#');
      input >> ident; // 'bdr_attribute_sets'

      MFEM_VERIFY(ident == "bdr_attribute_sets", "invalid mesh file");

      bdr_attribute_sets.attr_sets.Load(input);
      bdr_attribute_sets.attr_sets.SortAll();
      bdr_attribute_sets.attr_sets.UniqueAll();
   }

   skip_comment_lines(input, '#');
   input >> ident; // 'vertices'

   MFEM_VERIFY(ident == "vertices", "invalid mesh file");
   input >> NumOfVertices;
   vertices.SetSize(NumOfVertices);

   input >> ws >> ident;
   if (ident != "nodes")
   {
      // read the vertices
      spaceDim = atoi(ident.c_str());
      for (int j = 0; j < NumOfVertices; j++)
      {
         for (int i = 0; i < spaceDim; i++)
         {
            input >> vertices[j](i);
         }
      }
   }
   else
   {
      // prepare to read the nodes
      input >> ws;
      curved = 1;
   }

   // When visualizing solutions on non-conforming grids, PETSc
   // may dump additional vertices
   if (remove_unused_vertices) { RemoveUnusedVertices(); }
}

void Mesh::ReadLineMesh(std::istream &input)
{
   int j,p1,p2,a;

   Dim = 1;

   input >> NumOfVertices;
   vertices.SetSize(NumOfVertices);
   // Sets vertices and the corresponding coordinates
   for (j = 0; j < NumOfVertices; j++)
   {
      input >> vertices[j](0);
   }

   input >> NumOfElements;
   elements.SetSize(NumOfElements);
   // Sets elements and the corresponding indices of vertices
   for (j = 0; j < NumOfElements; j++)
   {
      input >> a >> p1 >> p2;
      elements[j] = new Segment(p1-1, p2-1, a);
   }

   int ind[1];
   input >> NumOfBdrElements;
   boundary.SetSize(NumOfBdrElements);
   for (j = 0; j < NumOfBdrElements; j++)
   {
      input >> a >> ind[0];
      ind[0]--;
      boundary[j] = new Point(ind,a);
   }
}

void Mesh::ReadNetgen2DMesh(std::istream &input, int &curved)
{
   int ints[32], attr, n;

   // Read planar mesh in Netgen format.
   Dim = 2;

   // Read the boundary elements.
   input >> NumOfBdrElements;
   boundary.SetSize(NumOfBdrElements);
   for (int i = 0; i < NumOfBdrElements; i++)
   {
      input >> attr
            >> ints[0] >> ints[1];
      ints[0]--; ints[1]--;
      boundary[i] = new Segment(ints, attr);
   }

   // Read the elements.
   input >> NumOfElements;
   elements.SetSize(NumOfElements);
   for (int i = 0; i < NumOfElements; i++)
   {
      input >> attr >> n;
      for (int j = 0; j < n; j++)
      {
         input >> ints[j];
         ints[j]--;
      }
      switch (n)
      {
         case 2:
            elements[i] = new Segment(ints, attr);
            break;
         case 3:
            elements[i] = new Triangle(ints, attr);
            break;
         case 4:
            elements[i] = new Quadrilateral(ints, attr);
            break;
      }
   }

   if (!curved)
   {
      // Read the vertices.
      input >> NumOfVertices;
      vertices.SetSize(NumOfVertices);
      for (int i = 0; i < NumOfVertices; i++)
         for (int j = 0; j < Dim; j++)
         {
            input >> vertices[i](j);
         }
   }
   else
   {
      input >> NumOfVertices;
      vertices.SetSize(NumOfVertices);
      input >> ws;
   }
}

void Mesh::ReadNetgen3DMesh(std::istream &input)
{
   int ints[32], attr;

   // Read a Netgen format mesh of tetrahedra.
   Dim = 3;

   // Read the vertices
   input >> NumOfVertices;

   vertices.SetSize(NumOfVertices);
   for (int i = 0; i < NumOfVertices; i++)
      for (int j = 0; j < Dim; j++)
      {
         input >> vertices[i](j);
      }

   // Read the elements
   input >> NumOfElements;
   elements.SetSize(NumOfElements);
   for (int i = 0; i < NumOfElements; i++)
   {
      input >> attr;
      for (int j = 0; j < 4; j++)
      {
         input >> ints[j];
         ints[j]--;
      }
#ifdef MFEM_USE_MEMALLOC
      Tetrahedron *tet;
      tet = TetMemory.Alloc();
      tet->SetVertices(ints);
      tet->SetAttribute(attr);
      elements[i] = tet;
#else
      elements[i] = new Tetrahedron(ints, attr);
#endif
   }

   // Read the boundary information.
   input >> NumOfBdrElements;
   boundary.SetSize(NumOfBdrElements);
   for (int i = 0; i < NumOfBdrElements; i++)
   {
      input >> attr;
      for (int j = 0; j < 3; j++)
      {
         input >> ints[j];
         ints[j]--;
      }
      boundary[i] = new Triangle(ints, attr);
   }
}

void Mesh::ReadTrueGridMesh(std::istream &input)
{
   int i, j, ints[32], attr;
   const int buflen = 1024;
   char buf[buflen];

   // TODO: find the actual dimension
   Dim = 3;

   if (Dim == 2)
   {
      int vari;
      real_t varf;

      input >> vari >> NumOfVertices >> vari >> vari >> NumOfElements;
      input.getline(buf, buflen);
      input.getline(buf, buflen);
      input >> vari;
      input.getline(buf, buflen);
      input.getline(buf, buflen);
      input.getline(buf, buflen);

      // Read the vertices.
      vertices.SetSize(NumOfVertices);
      for (i = 0; i < NumOfVertices; i++)
      {
         input >> vari >> varf >> vertices[i](0) >> vertices[i](1);
         input.getline(buf, buflen);
      }

      // Read the elements.
      elements.SetSize(NumOfElements);
      for (i = 0; i < NumOfElements; i++)
      {
         input >> vari >> attr;
         for (j = 0; j < 4; j++)
         {
            input >> ints[j];
            ints[j]--;
         }
         input.getline(buf, buflen);
         input.getline(buf, buflen);
         elements[i] = new Quadrilateral(ints, attr);
      }
   }
   else if (Dim == 3)
   {
      int vari;
      real_t varf;
      input >> vari >> NumOfVertices >> NumOfElements;
      input.getline(buf, buflen);
      input.getline(buf, buflen);
      input >> vari >> vari >> NumOfBdrElements;
      input.getline(buf, buflen);
      input.getline(buf, buflen);
      input.getline(buf, buflen);
      // Read the vertices.
      vertices.SetSize(NumOfVertices);
      for (i = 0; i < NumOfVertices; i++)
      {
         input >> vari >> varf >> vertices[i](0) >> vertices[i](1)
               >> vertices[i](2);
         input.getline(buf, buflen);
      }
      // Read the elements.
      elements.SetSize(NumOfElements);
      for (i = 0; i < NumOfElements; i++)
      {
         input >> vari >> attr;
         for (j = 0; j < 8; j++)
         {
            input >> ints[j];
            ints[j]--;
         }
         input.getline(buf, buflen);
         elements[i] = new Hexahedron(ints, attr);
      }
      // Read the boundary elements.
      boundary.SetSize(NumOfBdrElements);
      for (i = 0; i < NumOfBdrElements; i++)
      {
         input >> attr;
         for (j = 0; j < 4; j++)
         {
            input >> ints[j];
            ints[j]--;
         }
         input.getline(buf, buflen);
         boundary[i] = new Quadrilateral(ints, attr);
      }
   }
}

// see Tetrahedron::edges
const int Mesh::vtk_quadratic_tet[10] =
{ 0, 1, 2, 3, 4, 7, 5, 6, 8, 9 };

// see Pyramid::edges & Mesh::GenerateFaces
// https://www.vtk.org/doc/nightly/html/classvtkBiQuadraticQuadraticWedge.html
const int Mesh::vtk_quadratic_pyramid[13] =
{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

// see Wedge::edges & Mesh::GenerateFaces
// https://www.vtk.org/doc/nightly/html/classvtkBiQuadraticQuadraticWedge.html
const int Mesh::vtk_quadratic_wedge[18] =
{ 0, 2, 1, 3, 5, 4, 8, 7, 6, 11, 10, 9, 12, 14, 13, 17, 16, 15};

// see Hexahedron::edges & Mesh::GenerateFaces
const int Mesh::vtk_quadratic_hex[27] =
{
   0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
   24, 22, 21, 23, 20, 25, 26
};

void Mesh::CreateVTKMesh(const Vector &points, const Array<int> &cell_data,
                         const Array<int> &cell_offsets,
                         const Array<int> &cell_types,
                         const Array<int> &cell_attributes,
                         int &curved, int &read_gf, bool &finalize_topo)
{
   int np = points.Size()/3;
   Dim = -1;
   NumOfElements = cell_types.Size();
   elements.SetSize(NumOfElements);

   int order = -1;
   bool legacy_elem = false, lagrange_elem = false;

   for (int i = 0; i < NumOfElements; i++)
   {
      int j = (i > 0) ? cell_offsets[i-1] : 0;
      int ct = cell_types[i];
      Geometry::Type geom = VTKGeometry::GetMFEMGeometry(ct);
      elements[i] = NewElement(geom);
      if (cell_attributes.Size() > 0)
      {
         elements[i]->SetAttribute(cell_attributes[i]);
      }
      // VTK ordering of vertices is the same as MFEM ordering of vertices
      // for all element types *except* prisms, which require a permutation
      if (geom == Geometry::PRISM && ct != VTKGeometry::LAGRANGE_PRISM)
      {
         int prism_vertices[6];
         for (int k=0; k<6; ++k)
         {
            prism_vertices[k] = cell_data[j+VTKGeometry::PrismMap[k]];
         }
         elements[i]->SetVertices(prism_vertices);
      }
      else
      {
         elements[i]->SetVertices(&cell_data[j]);
      }

      int elem_dim = Geometry::Dimension[geom];
      int elem_order = VTKGeometry::GetOrder(ct, cell_offsets[i] - j);

      if (VTKGeometry::IsLagrange(ct)) { lagrange_elem = true; }
      else { legacy_elem = true; }

      MFEM_VERIFY(Dim == -1 || Dim == elem_dim,
                  "Elements with different dimensions are not supported");
      MFEM_VERIFY(order == -1 || order == elem_order,
                  "Elements with different orders are not supported");
      MFEM_VERIFY(legacy_elem != lagrange_elem,
                  "Mixing of legacy and Lagrange cell types is not supported");
      Dim = elem_dim;
      order = elem_order;
   }

   // determine spaceDim based on min/max differences detected each dimension
   spaceDim = 0;
   if (np > 0)
   {
      real_t min_value, max_value;
      for (int d = 3; d > 0; --d)
      {
         min_value = max_value = points(3*0 + d-1);
         for (int i = 1; i < np; i++)
         {
            min_value = std::min(min_value, points(3*i + d-1));
            max_value = std::max(max_value, points(3*i + d-1));
            if (min_value != max_value)
            {
               spaceDim = d;
               break;
            }
         }
         if (spaceDim > 0) { break; }
      }
   }

   if (order == 1 && !lagrange_elem)
   {
      NumOfVertices = np;
      vertices.SetSize(np);
      for (int i = 0; i < np; i++)
      {
         vertices[i](0) = points(3*i+0);
         vertices[i](1) = points(3*i+1);
         vertices[i](2) = points(3*i+2);
      }
      // No boundary is defined in a VTK mesh
      NumOfBdrElements = 0;
      FinalizeTopology();
      CheckElementOrientation(true);
   }
   else
   {
      // The following section of code is shared for legacy quadratic and the
      // Lagrange high order elements
      curved = 1;

      // generate new enumeration for the vertices
      Array<int> pts_dof(np);
      pts_dof = -1;
      // mark vertex points
      for (int i = 0; i < NumOfElements; i++)
      {
         int *v = elements[i]->GetVertices();
         int nv = elements[i]->GetNVertices();
         for (int j = 0; j < nv; j++)
         {
            if (pts_dof[v[j]] == -1) { pts_dof[v[j]] = 0; }
         }
      }

      // The following loop reorders pts_dofs so vertices are visited in
      // canonical order

      // Keep the original ordering of the vertices
      NumOfVertices = 0;
      for (int i = 0; i < np; i++)
      {
         if (pts_dof[i] != -1)
         {
            pts_dof[i] = NumOfVertices++;
         }
      }
      // update the element vertices
      for (int i = 0; i < NumOfElements; i++)
      {
         int *v = elements[i]->GetVertices();
         int nv = elements[i]->GetNVertices();
         for (int j = 0; j < nv; j++)
         {
            v[j] = pts_dof[v[j]];
         }
      }
      // Define the 'vertices' from the 'points' through the 'pts_dof' map
      vertices.SetSize(NumOfVertices);
      for (int i = 0; i < np; i++)
      {
         int j = pts_dof[i];
         if (j != -1)
         {
            vertices[j](0) = points(3*i+0);
            vertices[j](1) = points(3*i+1);
            vertices[j](2) = points(3*i+2);
         }
      }

      // No boundary is defined in a VTK mesh
      NumOfBdrElements = 0;

      // Generate faces and edges so that we can define FE space on the mesh
      FinalizeTopology();

      FiniteElementCollection *fec;
      FiniteElementSpace *fes;
      if (legacy_elem)
      {
         // Define quadratic FE space
         fec = new QuadraticFECollection;
         fes = new FiniteElementSpace(this, fec, spaceDim);
         Nodes = new GridFunction(fes);
         Nodes->MakeOwner(fec); // Nodes will destroy 'fec' and 'fes'
         own_nodes = 1;

         // Map vtk points to edge/face/element dofs
         Array<int> dofs;
         for (int i = 0; i < NumOfElements; i++)
         {
            fes->GetElementDofs(i, dofs);
            const int *vtk_mfem;
            switch (elements[i]->GetGeometryType())
            {
               case Geometry::TRIANGLE:
               case Geometry::SQUARE:
                  vtk_mfem = vtk_quadratic_hex; break; // identity map
               case Geometry::TETRAHEDRON:
                  vtk_mfem = vtk_quadratic_tet; break;
               case Geometry::CUBE:
                  vtk_mfem = vtk_quadratic_hex; break;
               case Geometry::PRISM:
                  vtk_mfem = vtk_quadratic_wedge; break;
               case Geometry::PYRAMID:
                  vtk_mfem = vtk_quadratic_pyramid; break;
               default:
                  vtk_mfem = NULL; // suppress a warning
                  break;
            }

            int offset = (i == 0) ? 0 : cell_offsets[i-1];
            for (int j = 0; j < dofs.Size(); j++)
            {
               if (pts_dof[cell_data[offset+j]] == -1)
               {
                  pts_dof[cell_data[offset+j]] = dofs[vtk_mfem[j]];
               }
               else
               {
                  if (pts_dof[cell_data[offset+j]] != dofs[vtk_mfem[j]])
                  {
                     MFEM_ABORT("VTK mesh: inconsistent quadratic mesh!");
                  }
               }
            }
         }
      }
      else
      {
         // Define H1 FE space
         fec = new H1_FECollection(order,Dim,BasisType::ClosedUniform);
         fes = new FiniteElementSpace(this, fec, spaceDim);
         Nodes = new GridFunction(fes);
         Nodes->MakeOwner(fec); // Nodes will destroy 'fec' and 'fes'
         own_nodes = 1;
         Array<int> dofs;

         std::map<Geometry::Type,Array<int>> vtk_inv_maps;
         std::map<Geometry::Type,const Array<int>*> lex_orderings;

         int i, n;
         for (n = i = 0; i < NumOfElements; i++)
         {
            Geometry::Type geom = GetElementBaseGeometry(i);
            fes->GetElementDofs(i, dofs);

            Array<int> &vtk_inv_map = vtk_inv_maps[geom];
            if (vtk_inv_map.Size() == 0)
            {
               Array<int> vtk_map;
               CreateVTKElementConnectivity(vtk_map, geom, order);
               vtk_inv_map.SetSize(vtk_map.Size());
               for (int j=0; j<vtk_map.Size(); ++j)
               {
                  vtk_inv_map[vtk_map[j]] = j;
               }
            }
            const Array<int> *&lex_ordering = lex_orderings[geom];
            if (!lex_ordering)
            {
               const FiniteElement *fe = fes->GetFE(i);
               const NodalFiniteElement *nodal_fe =
                  dynamic_cast<const NodalFiniteElement*>(fe);
               MFEM_ASSERT(nodal_fe != NULL, "Unsupported element type");
               lex_ordering = &nodal_fe->GetLexicographicOrdering();
            }

            for (int lex_idx = 0; lex_idx < dofs.Size(); lex_idx++)
            {
               int mfem_idx = (*lex_ordering)[lex_idx];
               int vtk_idx = vtk_inv_map[lex_idx];
               int pt_idx = cell_data[n + vtk_idx];
               if (pts_dof[pt_idx] == -1)
               {
                  pts_dof[pt_idx] = dofs[mfem_idx];
               }
               else
               {
                  if (pts_dof[pt_idx] != dofs[mfem_idx])
                  {
                     MFEM_ABORT("VTK mesh: inconsistent Lagrange mesh!");
                  }
               }
            }
            n += dofs.Size();
         }
      }
      // Define the 'Nodes' from the 'points' through the 'pts_dof' map
      Array<int> dofs;
      for (int i = 0; i < np; i++)
      {
         dofs.SetSize(1);
         if (pts_dof[i] != -1)
         {
            dofs[0] = pts_dof[i];
            fes->DofsToVDofs(dofs);
            for (int d = 0; d < dofs.Size(); d++)
            {
               (*Nodes)(dofs[d]) = points(3*i+d);
            }
         }
      }
      read_gf = 0;
   }
}

namespace vtk_xml
{

using namespace tinyxml2;

/// Return false if either string is NULL or if the strings differ, return true
/// if the strings are the same.
bool StringCompare(const char *s1, const char *s2)
{
   if (s1 == NULL || s2 == NULL) { return false; }
   return strcmp(s1, s2) == 0;
}

/// Abstract base class for reading contiguous arrays of (potentially
/// compressed, potentially base-64 encoded) binary data from a buffer into a
/// destination array. The types of the source and destination arrays may be
/// different (e.g. read data of type uint8_t into destination array of
/// uint32_t), which is handled by the templated derived class @a BufferReader.
struct BufferReaderBase
{
   enum HeaderType { UINT32_HEADER, UINT64_HEADER };
   virtual void ReadBinary(const char *buf, void *dest, int n) const = 0;
   virtual void ReadBase64(const char *txt, void *dest, int n) const = 0;
   virtual ~BufferReaderBase() { }
};

/// Read an array of source data stored as (potentially compressed, potentially
/// base-64 encoded) into a destination array. The types of the elements in the
/// source array are given by template parameter @a F ("from") and the types of
/// the elements of the destination array are given by @a T ("to"). The binary
/// data has a header, which is one integer if the data is uncompressed, and is
/// four integers if the data is compressed. The integers may either by uint32_t
/// or uint64_t, according to the @a header_type. If the data is compressed and
/// base-64 encoded, then the header is encoded separately from the data. If the
/// data is uncompressed and base-64 encoded, then the header and data are
/// encoded together.
template <typename T, typename F>
struct BufferReader : BufferReaderBase
{
   bool compressed;
   HeaderType header_type;
   BufferReader(bool compressed_, HeaderType header_type_)
      : compressed(compressed_), header_type(header_type_) { }

   /// Return the number of bytes of each header entry.
   size_t HeaderEntrySize() const
   {
      return header_type == UINT64_HEADER ? sizeof(uint64_t) : sizeof(uint32_t);
   }

   /// Return the value of the header entry pointer to by @a header_buf. The
   /// value is stored as either uint32_t or uint64_t, according to the @a
   /// header_type, and is returned as uint64_t.
   uint64_t ReadHeaderEntry(const char *header_buf) const
   {
      return (header_type == UINT64_HEADER) ? bin_io::read<uint64_t>(header_buf)
             : bin_io::read<uint32_t>(header_buf);
   }

   /// Return the number of bytes in the header. The header consists of one
   /// integer if the data is uncompressed, and @a N + 3 integers if the data is
   /// compressed, where @a N is the number of blocks. The integers are either
   /// 32 or 64 bytes depending on the value of @a header_type. The number of
   /// blocks is determined by reading the first integer (of type @a
   /// header_type) pointed to by @a header_buf.
   int NumHeaderBytes(const char *header_buf) const
   {
      if (!compressed) { return HeaderEntrySize(); }
      return (3 + ReadHeaderEntry(header_buf))*HeaderEntrySize();
   }

   /// Read @a n elements of type @a F from the source buffer @a buf into the
   /// (pre-allocated) destination array of elements of type @a T stored in
   /// the buffer @a dest_void. The header is stored @b separately from the
   /// rest of the data, in the buffer @a header_buf. The data buffer @a buf
   /// does @b not contain a header.
   void ReadBinaryWithHeader(const char *header_buf, const char *buf,
                             void *dest_void, int n) const
   {
      std::vector<char> uncompressed_data;
      T *dest = static_cast<T*>(dest_void);

      if (compressed)
      {
#ifdef MFEM_USE_ZLIB
         // The header has format (where header_t is uint32_t or uint64_t):
         //    header_t number_of_blocks;
         //    header_t uncompressed_block_size;
         //    header_t uncompressed_last_block_size;
         //    header_t compressed_size[number_of_blocks];
         int header_entry_size = HeaderEntrySize();
         int nblocks = ReadHeaderEntry(header_buf);
         header_buf += header_entry_size;
         std::vector<int> header(nblocks + 2);
         for (int i=0; i<nblocks+2; ++i)
         {
            header[i] = ReadHeaderEntry(header_buf);
            header_buf += header_entry_size;
         }
         uncompressed_data.resize((nblocks-1)*header[0] + header[1]);
         Bytef *dest_ptr = (Bytef *)uncompressed_data.data();
         Bytef *dest_start = dest_ptr;
         const Bytef *source_ptr = (const Bytef *)buf;
         for (int i=0; i<nblocks; ++i)
         {
            uLongf source_len = header[i+2];
            uLong dest_len = (i == nblocks-1) ? header[1] : header[0];
            int res = uncompress(dest_ptr, &dest_len, source_ptr, source_len);
            MFEM_VERIFY(res == Z_OK, "Error uncompressing");
            dest_ptr += dest_len;
            source_ptr += source_len;
         }
         MFEM_VERIFY(int(sizeof(F)*n) == (dest_ptr - dest_start),
                     "AppendedData: wrong data size");
         buf = uncompressed_data.data();
#else
         MFEM_ABORT("MFEM must be compiled with zlib enabled to uncompress.")
#endif
      }
      else
      {
         // Each "data block" is preceded by a header that is either UInt32 or
         // UInt64. The rest of the data follows.
         uint64_t data_size;
         if (header_type == UINT32_HEADER)
         {
            uint32_t *data_size_32 = (uint32_t *)header_buf;
            data_size = *data_size_32;
         }
         else
         {
            uint64_t *data_size_64 = (uint64_t *)header_buf;
            data_size = *data_size_64;
         }
         MFEM_VERIFY(sizeof(F)*n == data_size, "AppendedData: wrong data size");
      }

      if (std::is_same<T, F>::value)
      {
         // Special case: no type conversions necessary, so can just memcpy
         memcpy(dest, buf, sizeof(T)*n);
      }
      else
      {
         for (int i=0; i<n; ++i)
         {
            // Read binary data as type F, place in array as type T
            dest[i] = bin_io::read<F>(buf + i*sizeof(F));
         }
      }
   }

   /// Read @a n elements of type @a F from source buffer @a buf into
   /// (pre-allocated) array of elements of type @a T stored in destination
   /// buffer @a dest. The input buffer contains both the header and the data.
   void ReadBinary(const char *buf, void *dest, int n) const override
   {
      ReadBinaryWithHeader(buf, buf + NumHeaderBytes(buf), dest, n);
   }

   /// Read @a n elements of type @a F from base-64 encoded source buffer into
   /// (pre-allocated) array of elements of type @a T stored in destination
   /// buffer @a dest. The base-64-encoded data is given by the null-terminated
   /// string @a txt, which contains both the header and the data.
   void ReadBase64(const char *txt, void *dest, int n) const override
   {
      // Skip whitespace
      while (*txt)
      {
         if (*txt != ' ' && *txt != '\n') { break; }
         ++txt;
      }
      if (compressed)
      {
         // Decode the first entry of the header, which we need to determine
         // how long the rest of the header is.
         std::vector<char> nblocks_buf;
         int nblocks_b64 = bin_io::NumBase64Chars(HeaderEntrySize());
         bin_io::DecodeBase64(txt, nblocks_b64, nblocks_buf);
         std::vector<char> data, header;
         // Compute number of characters needed to encode header in base 64,
         // then round to nearest multiple of 4 to take padding into account.
         int header_b64 = bin_io::NumBase64Chars(NumHeaderBytes(nblocks_buf.data()));
         // If data is compressed, header is encoded separately
         bin_io::DecodeBase64(txt, header_b64, header);
         bin_io::DecodeBase64(txt + header_b64, strlen(txt)-header_b64, data);
         ReadBinaryWithHeader(header.data(), data.data(), dest, n);
      }
      else
      {
         std::vector<char> data;
         bin_io::DecodeBase64(txt, strlen(txt), data);
         ReadBinary(data.data(), dest, n);
      }
   }
};

/// Class to read data from VTK's @a DataArary elements. Each @a DataArray can
/// contain inline ASCII data, inline base-64-encoded data (potentially
/// compressed), or reference "appended data", which may be raw or base-64, and
/// may be compressed or uncompressed.
struct XMLDataReader
{
   const char *appended_data, *byte_order, *compressor;
   enum AppendedDataEncoding { RAW, BASE64 };
   map<string,BufferReaderBase*> type_map;
   AppendedDataEncoding encoding;

   /// Create the data reader, where @a vtk is the @a VTKFile XML element, and
   /// @a vtu is the child @a UnstructuredGrid XML element. This will determine
   /// the header type (32 or 64 bit integers) and whether compression is
   /// enabled or not. The appended data will be loaded.
   XMLDataReader(const XMLElement *vtk, const XMLElement *vtu)
   {
      // Determine whether binary data header is 32 or 64 bit integer
      BufferReaderBase::HeaderType htype;
      if (StringCompare(vtk->Attribute("header_type"), "UInt64"))
      {
         htype = BufferReaderBase::UINT64_HEADER;
      }
      else
      {
         htype = BufferReaderBase::UINT32_HEADER;
      }

      // Get the byte order of the file (will check if we encounter binary data)
      byte_order = vtk->Attribute("byte_order");

      // Get the compressor. We will check that MFEM can handle the compression
      // if we encounter binary data.
      compressor = vtk->Attribute("compressor");
      bool compressed = (compressor != NULL);

      // Find the appended data.
      appended_data = NULL;
      for (const XMLElement *xml_elem = vtu->NextSiblingElement();
           xml_elem != NULL;
           xml_elem = xml_elem->NextSiblingElement())
      {
         if (StringCompare(xml_elem->Name(), "AppendedData"))
         {
            const char *encoding_str = xml_elem->Attribute("encoding");
            if (StringCompare(encoding_str, "raw"))
            {
               appended_data = xml_elem->GetAppendedData();
               encoding = RAW;
            }
            else if (StringCompare(encoding_str, "base64"))
            {
               appended_data = xml_elem->GetText();
               encoding = BASE64;
            }
            MFEM_VERIFY(appended_data != NULL, "Invalid AppendedData");
            // Appended data follows first underscore
            bool found_leading_underscore = false;
            while (*appended_data)
            {
               ++appended_data;
               if (*appended_data == '_')
               {
                  found_leading_underscore = true;
                  ++appended_data;
                  break;
               }
            }
            MFEM_VERIFY(found_leading_underscore, "Invalid AppendedData");
            break;
         }
      }

      type_map["Int8"] = new BufferReader<int, int8_t>(compressed, htype);
      type_map["Int16"] = new BufferReader<int, int16_t>(compressed, htype);
      type_map["Int32"] = new BufferReader<int, int32_t>(compressed, htype);
      type_map["Int64"] = new BufferReader<int, int64_t>(compressed, htype);
      type_map["UInt8"] = new BufferReader<int, uint8_t>(compressed, htype);
      type_map["UInt16"] = new BufferReader<int, uint16_t>(compressed, htype);
      type_map["UInt32"] = new BufferReader<int, uint32_t>(compressed, htype);
      type_map["UInt64"] = new BufferReader<int, uint64_t>(compressed, htype);
      type_map["Float32"] = new BufferReader<double, float>(compressed, htype);
      type_map["Float64"] = new BufferReader<double, double>(compressed, htype);
   }

   /// Read the @a DataArray XML element given by @a xml_elem into
   /// (pre-allocated) destination array @a dest, where @a dest stores @a n
   /// elements of type @a T.
   template <typename T>
   void Read(const XMLElement *xml_elem, T *dest, int n)
   {
      static const char *erstr = "Error reading XML DataArray";
      MFEM_VERIFY(StringCompare(xml_elem->Name(), "DataArray"), erstr);
      const char *format = xml_elem->Attribute("format");
      if (StringCompare(format, "ascii"))
      {
         const char *txt = xml_elem->GetText();
         MFEM_VERIFY(txt != NULL, erstr);
         std::istringstream data_stream(txt);
         for (int i=0; i<n; ++i) { data_stream >> dest[i]; }
      }
      else if (StringCompare(format, "appended"))
      {
         VerifyBinaryOptions();
         int offset = xml_elem->IntAttribute("offset");
         const char *type = xml_elem->Attribute("type");
         MFEM_VERIFY(type != NULL, erstr);
         BufferReaderBase *reader = type_map[type];
         MFEM_VERIFY(reader != NULL, erstr);
         MFEM_VERIFY(appended_data != NULL, "No AppendedData found");
         if (encoding == RAW)
         {
            reader->ReadBinary(appended_data + offset, dest, n);
         }
         else
         {
            reader->ReadBase64(appended_data + offset, dest, n);
         }
      }
      else if (StringCompare(format, "binary"))
      {
         VerifyBinaryOptions();
         const char *txt = xml_elem->GetText();
         MFEM_VERIFY(txt != NULL, erstr);
         const char *type = xml_elem->Attribute("type");
         if (type == NULL) { MFEM_ABORT(erstr); }
         BufferReaderBase *reader = type_map[type];
         if (reader == NULL) { MFEM_ABORT(erstr); }
         reader->ReadBase64(txt, dest, n);
      }
      else
      {
         MFEM_ABORT("Invalid XML VTK DataArray format");
      }
   }

   /// Check that the byte order of the file is the same as the native byte
   /// order that we're running with. We don't currently support converting
   /// between byte orders. The byte order is only verified if we encounter
   /// binary data.
   void VerifyByteOrder() const
   {
      // Can't handle reading big endian from little endian or vice versa
      if (byte_order && !StringCompare(byte_order, VTKByteOrder()))
      {
         MFEM_ABORT("Converting between different byte orders is unsupported.");
      }
   }

   /// Check that the compressor is compatible (MFEM currently only supports
   /// zlib compression). If MFEM is not compiled with zlib, then we cannot
   /// read binary data with compression.
   void VerifyCompressor() const
   {
      if (compressor && !StringCompare(compressor, "vtkZLibDataCompressor"))
      {
         MFEM_ABORT("Unsupported compressor. Only zlib is supported.")
      }
#ifndef MFEM_USE_ZLIB
      MFEM_VERIFY(compressor == NULL, "MFEM must be compiled with zlib enabled "
                  "to support reading compressed data.");
#endif
   }

   /// Verify that the binary data is stored with compatible options (i.e.
   /// native byte order and compatible compression).
   void VerifyBinaryOptions() const
   {
      VerifyByteOrder();
      VerifyCompressor();
   }

   ~XMLDataReader()
   {
      for (auto &x : type_map) { delete x.second; }
   }
};

} // namespace vtk_xml

void Mesh::ReadXML_VTKMesh(std::istream &input, int &curved, int &read_gf,
                           bool &finalize_topo, const std::string &xml_prefix)
{
   using namespace vtk_xml;

   static const char *erstr = "XML parsing error";

   // Create buffer beginning with xml_prefix, then read the rest of the stream
   std::vector<char> buf(xml_prefix.begin(), xml_prefix.end());
   std::istreambuf_iterator<char> eos;
   buf.insert(buf.end(), std::istreambuf_iterator<char>(input), eos);
   buf.push_back('\0'); // null-terminate buffer

   XMLDocument xml;
   xml.Parse(buf.data(), buf.size());
   if (xml.ErrorID() != XML_SUCCESS)
   {
      MFEM_ABORT("Error parsing XML VTK file.\n" << xml.ErrorStr());
   }

   const XMLElement *vtkfile = xml.FirstChildElement();
   MFEM_VERIFY(vtkfile, erstr);
   MFEM_VERIFY(StringCompare(vtkfile->Name(), "VTKFile"), erstr);
   const XMLElement *vtu = vtkfile->FirstChildElement();
   MFEM_VERIFY(vtu, erstr);
   MFEM_VERIFY(StringCompare(vtu->Name(), "UnstructuredGrid"), erstr);

   XMLDataReader data_reader(vtkfile, vtu);

   // Count the number of points and cells
   const XMLElement *piece = vtu->FirstChildElement();
   MFEM_VERIFY(StringCompare(piece->Name(), "Piece"), erstr);
   MFEM_VERIFY(piece->NextSiblingElement() == NULL,
               "XML VTK meshes with more than one Piece are not supported");
   int npts = piece->IntAttribute("NumberOfPoints");
   int ncells = piece->IntAttribute("NumberOfCells");

   // Read the points
   Vector points(3*npts);
   const XMLElement *pts_xml;
   for (pts_xml = piece->FirstChildElement();
        pts_xml != NULL;
        pts_xml = pts_xml->NextSiblingElement())
   {
      if (StringCompare(pts_xml->Name(), "Points"))
      {
         const XMLElement *pts_data = pts_xml->FirstChildElement();
         MFEM_VERIFY(pts_data->IntAttribute("NumberOfComponents") == 3,
                     "XML VTK Points DataArray must have 3 components");
         data_reader.Read(pts_data, points.GetData(), points.Size());
         break;
      }
   }
   if (pts_xml == NULL) { MFEM_ABORT(erstr); }

   // Read the cells
   Array<int> cell_data, cell_offsets(ncells), cell_types(ncells);
   const XMLElement *cells_xml;
   for (cells_xml = piece->FirstChildElement();
        cells_xml != NULL;
        cells_xml = cells_xml->NextSiblingElement())
   {
      if (StringCompare(cells_xml->Name(), "Cells"))
      {
         const XMLElement *cell_data_xml = NULL;
         for (const XMLElement *data_xml = cells_xml->FirstChildElement();
              data_xml != NULL;
              data_xml = data_xml->NextSiblingElement())
         {
            const char *data_name = data_xml->Attribute("Name");
            if (StringCompare(data_name, "offsets"))
            {
               data_reader.Read(data_xml, cell_offsets.GetData(), ncells);
            }
            else if (StringCompare(data_name, "types"))
            {
               data_reader.Read(data_xml, cell_types.GetData(), ncells);
            }
            else if (StringCompare(data_name, "connectivity"))
            {
               // Have to read the connectivity after the offsets, because we
               // don't know how many points to read until we have the offsets
               // (size of connectivity array is equal to the last offset), so
               // store the XML element pointer and read this data later.
               cell_data_xml = data_xml;
            }
         }
         MFEM_VERIFY(cell_data_xml != NULL, erstr);
         int cell_data_size = cell_offsets.Last();
         cell_data.SetSize(cell_data_size);
         data_reader.Read(cell_data_xml, cell_data.GetData(), cell_data_size);
         break;
      }
   }
   if (cells_xml == NULL) { MFEM_ABORT(erstr); }

   // Read the element attributes, which are stored as CellData named "material"
   Array<int> cell_attributes;
   for (const XMLElement *cell_data_xml = piece->FirstChildElement();
        cell_data_xml != NULL;
        cell_data_xml = cell_data_xml->NextSiblingElement())
   {
      if (StringCompare(cell_data_xml->Name(), "CellData")
          && StringCompare(cell_data_xml->Attribute("Scalars"), "material"))
      {
         const XMLElement *data_xml = cell_data_xml->FirstChildElement();
         if (data_xml != NULL && StringCompare(data_xml->Name(), "DataArray"))
         {
            cell_attributes.SetSize(ncells);
            data_reader.Read(data_xml, cell_attributes.GetData(), ncells);
         }
      }
   }

   CreateVTKMesh(points, cell_data, cell_offsets, cell_types, cell_attributes,
                 curved, read_gf, finalize_topo);
}

void Mesh::ReadVTKMesh(std::istream &input, int &curved, int &read_gf,
                       bool &finalize_topo)
{
   // VTK resources:
   //   * https://www.vtk.org/doc/nightly/html/vtkCellType_8h_source.html
   //   * https://www.vtk.org/doc/nightly/html/classvtkCell.html
   //   * https://lorensen.github.io/VTKExamples/site/VTKFileFormats
   //   * https://www.kitware.com/products/books/VTKUsersGuide.pdf

   string buff;
   getline(input, buff); // comment line
   getline(input, buff);
   filter_dos(buff);
   if (buff != "ASCII")
   {
      MFEM_ABORT("VTK mesh is not in ASCII format!");
      return;
   }
   do
   {
      getline(input, buff);
      filter_dos(buff);
      if (!input.good()) { MFEM_ABORT("VTK mesh is not UNSTRUCTURED_GRID!"); }
   }
   while (buff != "DATASET UNSTRUCTURED_GRID");

   // Read the points, skipping optional sections such as the FIELD data from
   // VisIt's VTK export (or from Mesh::PrintVTK with field_data==1).
   do
   {
      input >> buff;
      if (!input.good())
      {
         MFEM_ABORT("VTK mesh does not have POINTS data!");
      }
   }
   while (buff != "POINTS");

   Vector points;
   int np;
   input >> np >> ws;
   getline(input, buff); // "double"
   points.Load(input, 3*np);

   //skip metadata
   // Looks like:
   // METADATA
   //INFORMATION 2
   //NAME L2_NORM_RANGE LOCATION vtkDataArray
   //DATA 2 0 5.19615
   //NAME L2_NORM_FINITE_RANGE LOCATION vtkDataArray
   //DATA 2 0 5.19615
   do
   {
      input >> buff;
      if (!input.good())
      {
         MFEM_ABORT("VTK mesh does not have CELLS data!");
      }
   }
   while (buff != "CELLS");

   // Read the cells
   Array<int> cell_data, cell_offsets;
   if (buff == "CELLS")
   {
      int ncells, n;
      input >> ncells >> n >> ws;
      cell_offsets.SetSize(ncells);
      cell_data.SetSize(n - ncells);
      int offset = 0;
      for (int i=0; i<ncells; ++i)
      {
         int nv;
         input >> nv;
         cell_offsets[i] = offset + nv;
         for (int j=0; j<nv; ++j)
         {
            input >> cell_data[offset + j];
         }
         offset += nv;
      }
   }

   // Read the cell types
   input >> ws >> buff;
   Array<int> cell_types;
   int ncells;
   MFEM_VERIFY(buff == "CELL_TYPES", "CELL_TYPES not provided in VTK mesh.")
   input >> ncells;
   cell_types.Load(ncells, input);

   while ((input.good()) && (buff != "CELL_DATA"))
   {
      input >> buff;
   }
   getline(input, buff); // finish the line

   // Read the cell materials
   // bool found_material = false;
   Array<int> cell_attributes;
   while ((input.good()))
   {
      getline(input, buff);
      if (buff.rfind("POINT_DATA") == 0)
      {
         break; // We have entered the POINT_DATA block. Quit.
      }
      else if (buff.rfind("SCALARS material") == 0)
      {
         getline(input, buff); // LOOKUP_TABLE default
         if (buff.rfind("LOOKUP_TABLE default") != 0)
         {
            MFEM_ABORT("Invalid LOOKUP_TABLE for material array in VTK file.");
         }
         cell_attributes.Load(ncells, input);
         // found_material = true;
         break;
      }
   }

   // if (!found_material)
   // {
   //    MFEM_WARNING("Material array not found in VTK file. "
   //                 "Assuming uniform material composition.");
   // }

   CreateVTKMesh(points, cell_data, cell_offsets, cell_types, cell_attributes,
                 curved, read_gf, finalize_topo);
} // end ReadVTKMesh

void Mesh::ReadNURBSMesh(std::istream &input, int &curved, int &read_gf,
                         bool spacing)
{
   NURBSext = new NURBSExtension(input, spacing);

   Dim              = NURBSext->Dimension();
   NumOfVertices    = NURBSext->GetNV();
   NumOfElements    = NURBSext->GetNE();
   NumOfBdrElements = NURBSext->GetNBE();

   NURBSext->GetElementTopo(elements);
   NURBSext->GetBdrElementTopo(boundary);

   vertices.SetSize(NumOfVertices);
   curved = 1;
   if (NURBSext->HavePatches())
   {
      NURBSFECollection  *fec = new NURBSFECollection(NURBSext->GetOrder());
      FiniteElementSpace *fes = new FiniteElementSpace(this, fec, Dim,
                                                       Ordering::byVDIM);
      Nodes = new GridFunction(fes);
      Nodes->MakeOwner(fec);
      NURBSext->SetCoordsFromPatches(*Nodes);
      own_nodes = 1;
      read_gf = 0;
      spaceDim = Nodes->VectorDim();
      for (int i = 0; i < spaceDim; i++)
      {
         Vector vert_val;
         Nodes->GetNodalValues(vert_val, i+1);
         for (int j = 0; j < NumOfVertices; j++)
         {
            vertices[j](i) = vert_val(j);
         }
      }
   }
   else
   {
      read_gf = 1;
   }
}

void Mesh::ReadInlineMesh(std::istream &input, bool generate_edges)
{
   // Initialize to negative numbers so that we know if they've been set.  We're
   // using Element::POINT as our flag, since we're not going to make a 0D mesh,
   // ever.
   int nx = -1;
   int ny = -1;
   int nz = -1;
   real_t sx = -1.0;
   real_t sy = -1.0;
   real_t sz = -1.0;
   Element::Type type = Element::POINT;

   while (true)
   {
      skip_comment_lines(input, '#');
      // Break out if we reached the end of the file after gobbling up the
      // whitespace and comments after the last keyword.
      if (!input.good())
      {
         break;
      }

      // Read the next keyword
      std::string name;
      input >> name;
      input >> std::ws;
      // Make sure there's an equal sign
      MFEM_VERIFY(input.get() == '=',
                  "Inline mesh expected '=' after keyword " << name);
      input >> std::ws;

      if (name == "nx")
      {
         input >> nx;
      }
      else if (name == "ny")
      {
         input >> ny;
      }
      else if (name == "nz")
      {
         input >> nz;
      }
      else if (name == "sx")
      {
         input >> sx;
      }
      else if (name == "sy")
      {
         input >> sy;
      }
      else if (name == "sz")
      {
         input >> sz;
      }
      else if (name == "type")
      {
         std::string eltype;
         input >> eltype;
         if (eltype == "segment")
         {
            type = Element::SEGMENT;
         }
         else if (eltype == "quad")
         {
            type = Element::QUADRILATERAL;
         }
         else if (eltype == "tri")
         {
            type = Element::TRIANGLE;
         }
         else if (eltype == "hex")
         {
            type = Element::HEXAHEDRON;
         }
         else if (eltype == "wedge")
         {
            type = Element::WEDGE;
         }
         else if (eltype == "pyramid")
         {
            type = Element::PYRAMID;
         }
         else if (eltype == "tet")
         {
            type = Element::TETRAHEDRON;
         }
         else
         {
            MFEM_ABORT("unrecognized element type (read '" << eltype
                       << "') in inline mesh format.  "
                       "Allowed: segment, tri, quad, tet, hex, wedge");
         }
      }
      else
      {
         MFEM_ABORT("unrecognized keyword (" << name
                    << ") in inline mesh format.  "
                    "Allowed: nx, ny, nz, type, sx, sy, sz");
      }

      input >> std::ws;
      // Allow an optional semi-colon at the end of each line.
      if (input.peek() == ';')
      {
         input.get();
      }

      // Done reading file
      if (!input)
      {
         break;
      }
   }

   // Now make the mesh.
   if (type == Element::SEGMENT)
   {
      MFEM_VERIFY(nx > 0 && sx > 0.0,
                  "invalid 1D inline mesh format, all values must be "
                  "positive\n"
                  << "   nx = " << nx << "\n"
                  << "   sx = " << sx << "\n");
      Make1D(nx, sx);
   }
   else if (type == Element::TRIANGLE || type == Element::QUADRILATERAL)
   {
      MFEM_VERIFY(nx > 0 && ny > 0 && sx > 0.0 && sy > 0.0,
                  "invalid 2D inline mesh format, all values must be "
                  "positive\n"
                  << "   nx = " << nx << "\n"
                  << "   ny = " << ny << "\n"
                  << "   sx = " << sx << "\n"
                  << "   sy = " << sy << "\n");
      Make2D(nx, ny, type, sx, sy, generate_edges, true);
   }
   else if (type == Element::TETRAHEDRON || type == Element::WEDGE ||
            type == Element::HEXAHEDRON  || type == Element::PYRAMID)
   {
      MFEM_VERIFY(nx > 0 && ny > 0 && nz > 0 &&
                  sx > 0.0 && sy > 0.0 && sz > 0.0,
                  "invalid 3D inline mesh format, all values must be "
                  "positive\n"
                  << "   nx = " << nx << "\n"
                  << "   ny = " << ny << "\n"
                  << "   nz = " << nz << "\n"
                  << "   sx = " << sx << "\n"
                  << "   sy = " << sy << "\n"
                  << "   sz = " << sz << "\n");
      Make3D(nx, ny, nz, type, sx, sy, sz, true);
      // TODO: maybe have an option in the file to control ordering?
   }
   else
   {
      MFEM_ABORT("For inline mesh, must specify an element type ="
                 " [segment, tri, quad, tet, hex, wedge]");
   }
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
   if (dsize != sizeof(double))
   {
      MFEM_ABORT("Gmsh file : dsize != sizeof(double)");
   }
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

         // number of nodes for each type of Gmsh elements, type is the index of
         // the array + 1
         int nodes_of_gmsh_element[] =
         {
            2,   // 2-node line.
            3,   // 3-node triangle.
            4,   // 4-node quadrangle.
            4,   // 4-node tetrahedron.
            8,   // 8-node hexahedron.
            6,   // 6-node prism.
            5,   // 5-node pyramid.
            3,   /* 3-node second order line (2 nodes associated with the
                    vertices and 1 with the edge). */
            6,   /* 6-node second order triangle (3 nodes associated with the
                    vertices and 3 with the edges). */
            9,   /* 9-node second order quadrangle (4 nodes associated with the
                    vertices, 4 with the edges and 1 with the face). */
            10,  /* 10-node second order tetrahedron (4 nodes associated with
                    the vertices and 6 with the edges). */
            27,  /* 27-node second order hexahedron (8 nodes associated with the
                    vertices, 12 with the edges, 6 with the faces and 1 with the
                    volume). */
            18,  /* 18-node second order prism (6 nodes associated with the
                    vertices, 9 with the edges and 3 with the quadrangular
                    faces). */
            14,  /* 14-node second order pyramid (5 nodes associated with the
                    vertices, 8 with the edges and 1 with the quadrangular
                    face). */
            1,   // 1-node point.
            8,   /* 8-node second order quadrangle (4 nodes associated with the
                    vertices and 4 with the edges). */
            20,  /* 20-node second order hexahedron (8 nodes associated with the
                    vertices and 12 with the edges). */
            15,  /* 15-node second order prism (6 nodes associated with the
                    vertices and 9 with the edges). */
            13,  /* 13-node second order pyramid (5 nodes associated with the
                    vertices and 8 with the edges). */
            9,   /* 9-node third order incomplete triangle (3 nodes associated
                    with the vertices, 6 with the edges) */
            10,  /* 10-node third order triangle (3 nodes associated with the
                    vertices, 6 with the edges, 1 with the face) */
            12,  /* 12-node fourth order incomplete triangle (3 nodes associated
                    with the vertices, 9 with the edges) */
            15,  /* 15-node fourth order triangle (3 nodes associated with the
                    vertices, 9 with the edges, 3 with the face) */
            15,  /* 15-node fifth order incomplete triangle (3 nodes associated
                    with the vertices, 12 with the edges) */
            21,  /* 21-node fifth order complete triangle (3 nodes associated
                    with the vertices, 12 with the edges, 6 with the face) */
            4,   /* 4-node third order edge (2 nodes associated with the
                    vertices, 2 internal to the edge) */
            5,   /* 5-node fourth order edge (2 nodes associated with the
                    vertices, 3 internal to the edge) */
            6,   /* 6-node fifth order edge (2 nodes associated with the
                    vertices, 4 internal to the edge) */
            20,  /* 20-node third order tetrahedron (4 nodes associated with the
                    vertices, 12 with the edges, 4 with the faces) */
            35,  /* 35-node fourth order tetrahedron (4 nodes associated with
                    the vertices, 18 with the edges, 12 with the faces, and 1
                    with the volume) */
            56,  /* 56-node fifth order tetrahedron (4 nodes associated with the
                    vertices, 24 with the edges, 24 with the faces, and 4 with
                    the volume) */
            -1,-1, /* unsupported tetrahedral types */
            -1,-1, /* unsupported polygonal and polyhedral types */
            16,  /* 16-node third order quadrilateral (4 nodes associated with
                    the vertices, 8 with the edges, 4 with the face) */
            25,  /* 25-node fourth order quadrilateral (4 nodes associated with
                    the vertices, 12 with the edges, 9 with the face) */
            36,  /* 36-node fifth order quadrilateral (4 nodes associated with
                    the vertices, 16 with the edges, 16 with the face) */
            -1,-1,-1, /* unsupported quadrilateral types */
            28,  /* 28-node sixth order complete triangle (3 nodes associated
                    with the vertices, 15 with the edges, 10 with the face) */
            36,  /* 36-node seventh order complete triangle (3 nodes associated
                    with the vertices, 18 with the edges, 15 with the face) */
            45,  /* 45-node eighth order complete triangle (3 nodes associated
                    with the vertices, 21 with the edges, 21 with the face) */
            55,  /* 55-node ninth order complete triangle (3 nodes associated
                    with the vertices, 24 with the edges, 28 with the face) */
            66,  /* 66-node tenth order complete triangle (3 nodes associated
                    with the vertices, 27 with the edges, 36 with the face) */
            49,  /* 49-node sixth order quadrilateral (4 nodes associated with
                    the vertices, 20 with the edges, 25 with the face) */
            64,  /* 64-node seventh order quadrilateral (4 nodes associated with
                    the vertices, 24 with the edges, 36 with the face) */
            81,  /* 81-node eighth order quadrilateral (4 nodes associated with
                    the vertices, 28 with the edges, 49 with the face) */
            100, /* 100-node ninth order quadrilateral (4 nodes associated with
                    the vertices, 32 with the edges, 64 with the face) */
            121, /* 121-node tenth order quadrilateral (4 nodes associated with
                    the vertices, 36 with the edges, 81 with the face) */
            -1,-1,-1,-1,-1, /* unsupported triangular types */
            -1,-1,-1,-1,-1, /* unsupported quadrilateral types */
            7,   /* 7-node sixth order edge (2 nodes associated with the
                    vertices, 5 internal to the edge) */
            8,   /* 8-node seventh order edge (2 nodes associated with the
                    vertices, 6 internal to the edge) */
            9,   /* 9-node eighth order edge (2 nodes associated with the
                    vertices, 7 internal to the edge) */
            10,  /* 10-node ninth order edge (2 nodes associated with the
                    vertices, 8 internal to the edge) */
            11,  /* 11-node tenth order edge (2 nodes associated with the
                    vertices, 9 internal to the edge) */
            -1,  /* unsupported linear types */
            -1,-1,-1, /* unsupported types */
            84,  /* 84-node sixth order tetrahedron (4 nodes associated with the
                    vertices, 30 with the edges, 40 with the faces, and 10 with
                    the volume) */
            120, /* 120-node seventh order tetrahedron (4 nodes associated with
                    the vertices, 36 with the edges, 60 with the faces, and 20
                    with the volume) */
            165, /* 165-node eighth order tetrahedron (4 nodes associated with
                    the vertices, 42 with the edges, 84 with the faces, and 35
                    with the volume) */
            220, /* 220-node ninth order tetrahedron (4 nodes associated with
                    the vertices, 48 with the edges, 112 with the faces, and 56
                    with the volume) */
            286, /* 286-node tenth order tetrahedron (4 nodes associated with
                    the vertices, 54 with the edges, 144 with the faces, and 84
                    with the volume) */
            -1,-1,-1,       /* undefined types */
            -1,-1,-1,-1,-1, /* unsupported tetrahedral types */
            -1,-1,-1,-1,-1,-1, /* unsupported types */
            40,  /* 40-node third order prism (6 nodes associated with the
                    vertices, 18 with the edges, 14 with the faces, and 2 with
                    the volume) */
            75,  /* 75-node fourth order prism (6 nodes associated with the
                    vertices, 27 with the edges, 33 with the faces, and 9 with
                    the volume) */
            64,  /* 64-node third order hexahedron (8 nodes associated with the
                    vertices, 24 with the edges, 24 with the faces and 8 with
                    the volume).*/
            125, /* 125-node fourth order hexahedron (8 nodes associated with
                    the vertices, 36 with the edges, 54 with the faces and 27
                    with the volume).*/
            216, /* 216-node fifth order hexahedron (8 nodes associated with the
                    vertices, 48 with the edges, 96 with the faces and 64 with
                    the volume).*/
            343, /* 343-node sixth order hexahedron (8 nodes associated with the
                    vertices, 60 with the edges, 150 with the faces and 125 with
                    the volume).*/
            512, /* 512-node seventh order hexahedron (8 nodes associated with
                    the vertices, 72 with the edges, 216 with the faces and 216
                    with the volume).*/
            729, /* 729-node eighth order hexahedron (8 nodes associated with
                    the vertices, 84 with the edges, 294 with the faces and 343
                    with the volume).*/
            1000,/* 1000-node ninth order hexahedron (8 nodes associated with
                    the vertices, 96 with the edges, 384 with the faces and 512
                    with the volume).*/
            -1,-1,-1,-1,-1,-1,-1, /* unsupported hexahedron types */
            126, /* 126-node fifth order prism (6 nodes associated with the
                    vertices, 36 with the edges, 60 with the faces, and 24 with
                    the volume) */
            196, /* 196-node sixth order prism (6 nodes associated with the
                    vertices, 45 with the edges, 95 with the faces, and 50 with
                    the volume) */
            288, /* 288-node seventh order prism (6 nodes associated with the
                    vertices, 54 with the edges, 138 with the faces, and 90 with
                    the volume) */
            405, /* 405-node eighth order prism (6 nodes associated with the
                    vertices, 63 with the edges, 189 with the faces, and 147
                    with the volume) */
            550, /* 550-node ninth order prism (6 nodes associated with the
                    vertices, 72 with the edges, 248 with the faces, and 224
                    with the volume) */
            -1,-1,-1,-1,-1,-1,-1, /* unsupported prism types */
            30,  /* 30-node third order pyramid (5 nodes associated with the
                    vertices, 16 with the edges and 8 with the faces, and 1 with
                    the volume). */
            55,  /* 55-node fourth order pyramid (5 nodes associated with the
                    vertices, 24 with the edges and 21 with the faces, and 5
                    with the volume). */
            91,  /* 91-node fifth order pyramid (5 nodes associated with the
                    vertices, 32 with the edges and 40 with the faces, and 14
                    with the volume). */
            140, /* 140-node sixth order pyramid (5 nodes associated with the
                    vertices, 40 with the edges and 65 with the faces, and 30
                    with the volume). */
            204, /* 204-node seventh order pyramid (5 nodes associated with the
                    vertices, 48 with the edges and 96 with the faces, and 55
                    with the volume). */
            285, /* 285-node eighth order pyramid (5 nodes associated with the
                    vertices, 56 with the edges and 133 with the faces, and 91
                    with the volume). */
            385  /* 385-node ninth order pyramid (5 nodes associated with the
                    vertices, 64 with the edges and 176 with the faces, and 140
                    with the volume). */
         };

         /** The following mappings convert the Gmsh node orderings for high
             order elements to MFEM's L2 degree of freedom ordering. To support
             more options examine Gmsh's ordering and read off the indices in
             MFEM's order. For example 2nd order Gmsh quadrilaterals use the
             following ordering:

                3--6--2
                |  |  |
                7  8  5
                |  |  |
                0--4--1

             (from https://gmsh.info/doc/texinfo/gmsh.html#Node-ordering)

             Whereas MFEM uses a tensor product ordering with the horizontal
             axis cycling fastest so we would read off:

                0 4 1 7 8 5 3 6 2

             This corresponds to the quad9 mapping below.
         */
         int lin3[] = {0,2,1};                // 2nd order segment
         int lin4[] = {0,2,3,1};              // 3rd order segment
         int tri6[] = {0,3,1,5,4,2};          // 2nd order triangle
         int tri10[] = {0,3,4,1,8,9,5,7,6,2}; // 3rd order triangle
         int quad9[] = {0,4,1,7,8,5,3,6,2};   // 2nd order quadrilateral
         int quad16[] = {0,4,5,1,11,12,13,6,  // 3rd order quadrilateral
                         10,15,14,7,3,9,8,2
                        };
         int tet10[] {0,4,1,6,5,2,7,9,8,3};   // 2nd order tetrahedron
         int tet20[] = {0,4,5,1,9,16,6,8,7,2, // 3rd order tetrahedron
                        11,17,15,18,19,13,10,14,12,3
                       };
         int hex27[] {0,8,1,9,20,11,3,13,2,   // 2nd order hexahedron
                      10,21,12,22,26,23,15,24,14,
                      4,16,5,17,25,18,7,19,6
                     };
         int hex64[] {0,8,9,1,10,32,35,14,    // 3rd order hexahedron
                      11,33,34,15,3,19,18,2,
                      12,36,37,16,40,56,57,44,
                      43,59,58,45,22,49,48,20,
                      13,39,38,17,41,60,61,47,
                      42,63,62,46,23,50,51,21,
                      4,24,25,5,26,52,53,28,
                      27,55,54,29,7,31,30,6
                     };

         int wdg18[] = {0,6,1,7,9,2,8,15,10,    // 2nd order wedge/prism
                        16,17,11,3,12,4,13,14,5
                       };
         int wdg40[] = {0,6,7,1,8,24,12,9,13,2, // 3rd order wedge/prism
                        10,26,27,14,30,38,34,33,35,16,
                        11,29,28,15,31,39,37,32,36,17,
                        3,18,19,4,20,25,22,21,23,5
                       };

         int pyr14[] = {0,5,1,6,13,8,3,          // 2nd order pyramid
                        10,2,7,9,12,11,4
                       };
         int pyr30[] = {0,5,6,1,7,25,28,11,8,26, // 3rd order pyramid
                        27,12,3,16,15,2,9,21,13,22,
                        29,23,19,24,17,10,14,20,18,4
                       };

         vector<Element*> elements_0D, elements_1D, elements_2D, elements_3D;
         elements_0D.reserve(num_of_all_elements);
         elements_1D.reserve(num_of_all_elements);
         elements_2D.reserve(num_of_all_elements);
         elements_3D.reserve(num_of_all_elements);

         // Temporary storage for high order vertices, if present
         vector<Array<int>*> ho_verts_1D, ho_verts_2D, ho_verts_3D;
         ho_verts_1D.reserve(num_of_all_elements);
         ho_verts_2D.reserve(num_of_all_elements);
         ho_verts_3D.reserve(num_of_all_elements);

         // Temporary storage for order of elements
         vector<int> ho_el_order_1D, ho_el_order_2D, ho_el_order_3D;
         ho_el_order_1D.reserve(num_of_all_elements);
         ho_el_order_2D.reserve(num_of_all_elements);
         ho_el_order_3D.reserve(num_of_all_elements);

         // Vertex order mappings
         Array<int*> ho_lin(11); ho_lin = NULL;
         Array<int*> ho_tri(11); ho_tri = NULL;
         Array<int*> ho_sqr(11); ho_sqr = NULL;
         Array<int*> ho_tet(11); ho_tet = NULL;
         Array<int*> ho_hex(10); ho_hex = NULL;
         Array<int*> ho_wdg(10); ho_wdg = NULL;
         Array<int*> ho_pyr(10); ho_pyr = NULL;

         // Use predefined arrays at lowest orders (for efficiency)
         ho_lin[2] = lin3;  ho_lin[3] = lin4;
         ho_tri[2] = tri6;  ho_tri[3] = tri10;
         ho_sqr[2] = quad9; ho_sqr[3] = quad16;
         ho_tet[2] = tet10; ho_tet[3] = tet20;
         ho_hex[2] = hex27; ho_hex[3] = hex64;
         ho_wdg[2] = wdg18; ho_wdg[3] = wdg40;
         ho_pyr[2] = pyr14; ho_pyr[3] = pyr30;

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

               const int n_elem_nodes = nodes_of_gmsh_element[type_of_element-1];
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
                  int el_order = 11;
                  switch (type_of_element)
                  {
                     case  1: //   2-node line
                     case  8: //   3-node line (2nd order)
                     case 26: //   4-node line (3rd order)
                     case 27: //   5-node line (4th order)
                     case 28: //   6-node line (5th order)
                     case 62: //   7-node line (6th order)
                     case 63: //   8-node line (7th order)
                     case 64: //   9-node line (8th order)
                     case 65: //  10-node line (9th order)
                     case 66: //  11-node line (10th order)
                     {
                        elements_1D.push_back(
                           new Segment(&vert_indices[0], phys_domain));
                        if (type_of_element != 1)
                        {
                           el_order = n_elem_nodes - 1;
                           Array<int> * hov = new Array<int>;
                           hov->Append(&vert_indices[0], n_elem_nodes);
                           ho_verts_1D.push_back(hov);
                           ho_el_order_1D.push_back(el_order);
                        }
                        break;
                     }
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
                        {
                           elements_2D.push_back(
                              new Triangle(&vert_indices[0], phys_domain));
                           if (el_order > 1)
                           {
                              Array<int> * hov = new Array<int>;
                              hov->Append(&vert_indices[0], n_elem_nodes);
                              ho_verts_2D.push_back(hov);
                              ho_el_order_2D.push_back(el_order);
                           }
                           break;
                        }
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
                        {
                           elements_2D.push_back(
                              new Quadrilateral(&vert_indices[0], phys_domain));
                           if (el_order > 1)
                           {
                              Array<int> * hov = new Array<int>;
                              hov->Append(&vert_indices[0], n_elem_nodes);
                              ho_verts_2D.push_back(hov);
                              ho_el_order_2D.push_back(el_order);
                           }
                           break;
                        }
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
                        {
#ifdef MFEM_USE_MEMALLOC
                           elements_3D.push_back(TetMemory.Alloc());
                           elements_3D.back()->SetVertices(&vert_indices[0]);
                           elements_3D.back()->SetAttribute(phys_domain);
#else
                           elements_3D.push_back(
                              new Tetrahedron(&vert_indices[0], phys_domain));
#endif
                           if (el_order > 1)
                           {
                              Array<int> * hov = new Array<int>;
                              hov->Append(&vert_indices[0], n_elem_nodes);
                              ho_verts_3D.push_back(hov);
                              ho_el_order_3D.push_back(el_order);
                           }
                           break;
                        }
                     case  5: el_order--; //    8-node hexahedron
                     case 12: el_order--; //   27-node hexahedron (2nd order)
                     case 92: el_order--; //   64-node hexahedron (3rd order)
                     case 93: el_order--; //  125-node hexahedron (4th order)
                     case 94: el_order--; //  216-node hexahedron (5th order)
                     case 95: el_order--; //  343-node hexahedron (6th order)
                     case 96: el_order--; //  512-node hexahedron (7th order)
                     case 97: el_order--; //  729-node hexahedron (8th order)
                     case 98: el_order--; // 1000-node hexahedron (9th order)
                        {
                           el_order--; // Gmsh does not define an order 10 hex
                           elements_3D.push_back(
                              new Hexahedron(&vert_indices[0], phys_domain));
                           if (el_order > 1)
                           {
                              Array<int> * hov = new Array<int>;
                              hov->Append(&vert_indices[0], n_elem_nodes);
                              ho_verts_3D.push_back(hov);
                              ho_el_order_3D.push_back(el_order);
                           }
                           break;
                        }
                     case   6: el_order--; //   6-node wedge
                     case  13: el_order--; //  18-node wedge (2nd order)
                     case  90: el_order--; //  40-node wedge (3rd order)
                     case  91: el_order--; //  75-node wedge (4th order)
                     case 106: el_order--; // 126-node wedge (5th order)
                     case 107: el_order--; // 196-node wedge (6th order)
                     case 108: el_order--; // 288-node wedge (7th order)
                     case 109: el_order--; // 405-node wedge (8th order)
                     case 110: el_order--; // 550-node wedge (9th order)
                        {
                           el_order--; // Gmsh does not define an order 10 wedge
                           elements_3D.push_back(
                              new Wedge(&vert_indices[0], phys_domain));
                           if (el_order > 1)
                           {
                              Array<int> * hov = new Array<int>;
                              hov->Append(&vert_indices[0], n_elem_nodes);
                              ho_verts_3D.push_back(hov);
                              ho_el_order_3D.push_back(el_order);
                           }
                           break;
                        }
                     case   7: el_order--; //   5-node pyramid
                     case  14: el_order--; //  14-node pyramid (2nd order)
                     case 118: el_order--; //  30-node pyramid (3rd order)
                     case 119: el_order--; //  55-node pyramid (4th order)
                     case 120: el_order--; //  91-node pyramid (5th order)
                     case 121: el_order--; // 140-node pyramid (6th order)
                     case 122: el_order--; // 204-node pyramid (7th order)
                     case 123: el_order--; // 285-node pyramid (8th order)
                     case 124: el_order--; // 385-node pyramid (9th order)
                        {
                           el_order--; // Gmsh does not define an order 10 pyr
                           elements_3D.push_back(
                              new Pyramid(&vert_indices[0], phys_domain));
                           if (el_order > 1)
                           {
                              Array<int> * hov = new Array<int>;
                              hov->Append(&vert_indices[0], n_elem_nodes);
                              ho_verts_3D.push_back(hov);
                              ho_el_order_3D.push_back(el_order);
                           }
                           break;
                        }
                     case 15: // 1-node point
                     {
                        elements_0D.push_back(
                           new Point(&vert_indices[0], phys_domain));
                        break;
                     }
                     default: // any other element
                        MFEM_WARNING("Unsupported Gmsh element type.");
                        break;

                  } // switch (type_of_element)
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
               const int n_elem_nodes = nodes_of_gmsh_element[type_of_element-1];
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
               int el_order = 11;
               switch (type_of_element)
               {
                  case  1: //  2-node line
                  case  8: //  3-node line (2nd order)
                  case 26: //  4-node line (3rd order)
                  case 27: //  5-node line (4th order)
                  case 28: //  6-node line (5th order)
                  case 62: //  7-node line (6th order)
                  case 63: //  8-node line (7th order)
                  case 64: //  9-node line (8th order)
                  case 65: // 10-node line (9th order)
                  case 66: // 11-node line (10th order)
                  {
                     elements_1D.push_back(
                        new Segment(&vert_indices[0], phys_domain));
                     if (type_of_element != 1)
                     {
                        Array<int> * hov = new Array<int>;
                        hov->Append(&vert_indices[0], n_elem_nodes);
                        ho_verts_1D.push_back(hov);
                        el_order = n_elem_nodes - 1;
                        ho_el_order_1D.push_back(el_order);
                     }
                     break;
                  }
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
                     {
                        elements_2D.push_back(
                           new Triangle(&vert_indices[0], phys_domain));
                        if (el_order > 1)
                        {
                           Array<int> * hov = new Array<int>;
                           hov->Append(&vert_indices[0], n_elem_nodes);
                           ho_verts_2D.push_back(hov);
                           ho_el_order_2D.push_back(el_order);
                        }
                        break;
                     }
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
                     {
                        elements_2D.push_back(
                           new Quadrilateral(&vert_indices[0], phys_domain));
                        if (el_order > 1)
                        {
                           Array<int> * hov = new Array<int>;
                           hov->Append(&vert_indices[0], n_elem_nodes);
                           ho_verts_2D.push_back(hov);
                           ho_el_order_2D.push_back(el_order);
                        }
                        break;
                     }
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
                     {
#ifdef MFEM_USE_MEMALLOC
                        elements_3D.push_back(TetMemory.Alloc());
                        elements_3D.back()->SetVertices(&vert_indices[0]);
                        elements_3D.back()->SetAttribute(phys_domain);
#else
                        elements_3D.push_back(
                           new Tetrahedron(&vert_indices[0], phys_domain));
#endif
                        if (el_order > 1)
                        {
                           Array<int> * hov = new Array<int>;
                           hov->Append(&vert_indices[0], n_elem_nodes);
                           ho_verts_3D.push_back(hov);
                           ho_el_order_3D.push_back(el_order);
                        }
                        break;
                     }
                  case  5: el_order--; //    8-node hexahedron
                  case 12: el_order--; //   27-node hexahedron (2nd order)
                  case 92: el_order--; //   64-node hexahedron (3rd order)
                  case 93: el_order--; //  125-node hexahedron (4th order)
                  case 94: el_order--; //  216-node hexahedron (5th order)
                  case 95: el_order--; //  343-node hexahedron (6th order)
                  case 96: el_order--; //  512-node hexahedron (7th order)
                  case 97: el_order--; //  729-node hexahedron (8th order)
                  case 98: el_order--; // 1000-node hexahedron (9th order)
                     {
                        el_order--;
                        elements_3D.push_back(
                           new Hexahedron(&vert_indices[0], phys_domain));
                        if (el_order > 1)
                        {
                           Array<int> * hov = new Array<int>;
                           hov->Append(&vert_indices[0], n_elem_nodes);
                           ho_verts_3D.push_back(hov);
                           ho_el_order_3D.push_back(el_order);
                        }
                        break;
                     }
                  case   6: el_order--; //   6-node wedge
                  case  13: el_order--; //  18-node wedge (2nd order)
                  case  90: el_order--; //  40-node wedge (3rd order)
                  case  91: el_order--; //  75-node wedge (4th order)
                  case 106: el_order--; // 126-node wedge (5th order)
                  case 107: el_order--; // 196-node wedge (6th order)
                  case 108: el_order--; // 288-node wedge (7th order)
                  case 109: el_order--; // 405-node wedge (8th order)
                  case 110: el_order--; // 550-node wedge (9th order)
                     {
                        el_order--;
                        elements_3D.push_back(
                           new Wedge(&vert_indices[0], phys_domain));
                        if (el_order > 1)
                        {
                           Array<int> * hov = new Array<int>;
                           hov->Append(&vert_indices[0], n_elem_nodes);
                           ho_verts_3D.push_back(hov);
                           ho_el_order_3D.push_back(el_order);
                        }
                        break;
                     }
                  case   7: el_order--; //   5-node pyramid
                  case  14: el_order--; //  14-node pyramid (2nd order)
                  case 118: el_order--; //  30-node pyramid (3rd order)
                  case 119: el_order--; //  55-node pyramid (4th order)
                  case 120: el_order--; //  91-node pyramid (5th order)
                  case 121: el_order--; // 140-node pyramid (6th order)
                  case 122: el_order--; // 204-node pyramid (7th order)
                  case 123: el_order--; // 285-node pyramid (8th order)
                  case 124: el_order--; // 385-node pyramid (9th order)
                     {
                        el_order--;
                        elements_3D.push_back(
                           new Pyramid(&vert_indices[0], phys_domain));
                        if (el_order > 1)
                        {
                           Array<int> * hov = new Array<int>;
                           hov->Append(&vert_indices[0], n_elem_nodes);
                           ho_verts_3D.push_back(hov);
                           ho_el_order_3D.push_back(el_order);
                        }
                        break;
                     }
                  case 15: // 1-node point
                  {
                     elements_0D.push_back(
                        new Point(&vert_indices[0], phys_domain));
                     break;
                  }
                  default: // any other element
                     MFEM_WARNING("Unsupported Gmsh element type.");
                     break;

               } // switch (type_of_element)
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
            this->SetMeshGen();

            // Generate faces and edges so that we can define
            // FE space on the mesh
            this->FinalizeTopology();

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
               const int * vm = NULL;
               Array<int> * ho_verts = NULL;
               switch (GetElementType(el))
               {
                  case Element::SEGMENT:
                     ho_verts = ho_verts_1D[el];
                     el_order = ho_el_order_1D[el];
                     if (!ho_lin[el_order])
                     {
                        ho_lin[el_order] = new int[ho_verts->Size()];
                        GmshHOSegmentMapping(el_order, ho_lin[el_order]);
                     }
                     vm = ho_lin[el_order];
                     break;
                  case Element::TRIANGLE:
                     ho_verts = ho_verts_2D[el];
                     el_order = ho_el_order_2D[el];
                     if (!ho_tri[el_order])
                     {
                        ho_tri[el_order] = new int[ho_verts->Size()];
                        GmshHOTriangleMapping(el_order, ho_tri[el_order]);
                     }
                     vm = ho_tri[el_order];
                     break;
                  case Element::QUADRILATERAL:
                     ho_verts = ho_verts_2D[el];
                     el_order = ho_el_order_2D[el];
                     if (!ho_sqr[el_order])
                     {
                        ho_sqr[el_order] = new int[ho_verts->Size()];
                        GmshHOQuadrilateralMapping(el_order, ho_sqr[el_order]);
                     }
                     vm = ho_sqr[el_order];
                     break;
                  case Element::TETRAHEDRON:
                     ho_verts = ho_verts_3D[el];
                     el_order = ho_el_order_3D[el];
                     if (!ho_tet[el_order])
                     {
                        ho_tet[el_order] = new int[ho_verts->Size()];
                        GmshHOTetrahedronMapping(el_order, ho_tet[el_order]);
                     }
                     vm = ho_tet[el_order];
                     break;
                  case Element::HEXAHEDRON:
                     ho_verts = ho_verts_3D[el];
                     el_order = ho_el_order_3D[el];
                     if (!ho_hex[el_order])
                     {
                        ho_hex[el_order] = new int[ho_verts->Size()];
                        GmshHOHexahedronMapping(el_order, ho_hex[el_order]);
                     }
                     vm = ho_hex[el_order];
                     break;
                  case Element::WEDGE:
                     ho_verts = ho_verts_3D[el];
                     el_order = ho_el_order_3D[el];
                     if (!ho_wdg[el_order])
                     {
                        ho_wdg[el_order] = new int[ho_verts->Size()];
                        GmshHOWedgeMapping(el_order, ho_wdg[el_order]);
                     }
                     vm = ho_wdg[el_order];
                     break;
                  case Element::PYRAMID:
                     ho_verts = ho_verts_3D[el];
                     el_order = ho_el_order_3D[el];
                     if (!ho_pyr[el_order])
                     {
                        ho_pyr[el_order] = new int[ho_verts->Size()];
                        GmshHOPyramidMapping(el_order, ho_pyr[el_order]);
                     }
                     vm = ho_pyr[el_order];
                     break;
                  default: // Any other element type
                     MFEM_WARNING("Unsupported Gmsh element type.");
                     break;
               }
               int nv = (ho_verts) ? ho_verts->Size() : 0;

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

         // Delete any high order element to vertex connectivity
         for (size_t el=0; el<ho_verts_1D.size(); el++)
         {
            delete ho_verts_1D[el];
         }
         for (size_t el=0; el<ho_verts_2D.size(); el++)
         {
            delete ho_verts_2D[el];
         }
         for (size_t el=0; el<ho_verts_3D.size(); el++)
         {
            delete ho_verts_3D[el];
         }

         // Delete dynamically allocated high vertex order mappings
         for (int ord=4; ord<ho_lin.Size(); ord++)
         {
            if (ho_lin[ord] != NULL) { delete [] ho_lin[ord]; }
         }
         for (int ord=4; ord<ho_tri.Size(); ord++)
         {
            if (ho_tri[ord] != NULL) { delete [] ho_tri[ord]; }
         }
         for (int ord=4; ord<ho_sqr.Size(); ord++)
         {
            if (ho_sqr[ord] != NULL) { delete [] ho_sqr[ord]; }
         }
         for (int ord=4; ord<ho_tet.Size(); ord++)
         {
            if (ho_tet[ord] != NULL) { delete [] ho_tet[ord]; }
         }
         for (int ord=4; ord<ho_hex.Size(); ord++)
         {
            if (ho_hex[ord] != NULL) { delete [] ho_hex[ord]; }
         }
         for (int ord=4; ord<ho_wdg.Size(); ord++)
         {
            if (ho_wdg[ord] != NULL) { delete [] ho_wdg[ord]; }
         }
         for (int ord=4; ord<ho_pyr.Size(); ord++)
         {
            if (ho_pyr[ord] != NULL) { delete [] ho_pyr[ord]; }
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
            this->FinalizeTopology();
            this->SetMeshGen();
            this->SetCurvature(1, true, spaceDim, Ordering::byVDIM);
         }

         // Replace "slave" vertex indices in the element connectivity
         // with their corresponding "master" vertex indices.
         for (int i = 0; i < this->GetNE(); i++)
         {
            Element *el = this->GetElement(i);
            int *v = el->GetVertices();
            int nv = el->GetNVertices();
            for (int j = 0; j < nv; j++)
            {
               v[j] = v2v[v[j]];
            }
         }
         // Replace "slave" vertex indices in the boundary element connectivity
         // with their corresponding "master" vertex indices.
         for (int i = 0; i < this->GetNBE(); i++)
         {
            Element *el = this->GetBdrElement(i);
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

   this->RemoveUnusedVertices();
   if (periodic)
   {
      this->RemoveInternalBoundaries();
   }
   this->FinalizeTopology();

   // If a high order coordinate field was created project it onto the mesh
   if (mesh_order > 1)
   {
      SetCurvature(mesh_order, periodic, spaceDim, Ordering::byVDIM);

      VectorGridFunctionCoefficient NodesCoef(&Nodes_gf);
      Nodes->ProjectCoefficient(NodesCoef);
   }
}


#ifdef MFEM_USE_NETCDF

namespace cubit
{

const int mfem_to_genesis_tet10[10] =
{
   // 1,2,3,4,5,6,7,8,9,10
   1,2,3,4,5,7,8,6,9,10
};

const int mfem_to_genesis_hex27[27] =
{
   // 1,2,3,4,5,6,7,8,9,10,11,
   1,2,3,4,5,6,7,8,9,10,11,

   // 12,13,14,15,16,17,18,19
   12,17,18,19,20,13,14,15,

   // 20,21,22,23,24,25,26,27
   16,22,26,25,27,24,23,21
};

const int mfem_to_genesis_tri6[6]   =
{
   1,2,3,4,5,6
};

const int mfem_to_genesis_quad9[9]  =
{
   1,2,3,4,5,6,7,8,9
};

const int cubit_side_map_tri3[3][2] =
{
   {1,2},
   {2,3},
   {3,1},
};

const int cubit_side_map_quad4[4][2] =
{
   {1,2},
   {2,3},
   {3,4},
   {4,1},
};

const int cubit_side_map_tri6[3][3] =
{
   {1,2,4},
   {2,3,5},
   {3,1,6},
};

const int cubit_side_map_quad9[4][3] =
{
   {1,2,5},
   {2,3,6},
   {3,4,7},
   {4,1,8},
};

const int cubit_side_map_tet4[4][3] =
{
   {1,2,4},
   {2,3,4},
   {1,4,3},
   {1,3,2}
};

const int cubit_side_map_tet10[4][6] =
{
   {1,2,4,5,9,8},
   {2,3,4,6,10,9},
   {1,4,3,8,10,7},
   {1,3,2,7,6,5}
};

const int cubit_side_map_hex8[6][4] =
{
   {1,2,6,5},
   {2,3,7,6},
   {4,3,7,8},
   {1,4,8,5},
   {1,4,3,2},
   {5,8,7,6}
};

const int cubit_side_map_hex27[6][9] =
{
   {1,2,6,5,9,14,17,13,26},
   {2,3,7,6,10,15,18,14,25},
   {4,3,7,8,11,15,19,16,27},
   {1,4,8,5,12,16,20,13,24},
   {1,4,3,2,12,11,10,9,22},
   {5,8,7,6,20,19,18,17,23}
};


enum CubitElementType
{
   ELEMENT_TRI3,
   ELEMENT_TRI6,
   ELEMENT_QUAD4,
   ELEMENT_QUAD9,
   ELEMENT_TET4,
   ELEMENT_TET10,
   ELEMENT_HEX8,
   ELEMENT_HEX27
};

enum CubitFaceType
{
   FACE_EDGE2,
   FACE_EDGE3,
   FACE_TRI3,
   FACE_TRI6,
   FACE_QUAD4,
   FACE_QUAD9
};


static void HandleNetCDFError(const int error)
{
   if (error == NC_NOERR) { return; }

   MFEM_ABORT("Fatal NetCDF error: " << nc_strerror(error));
}


static void ReadCubitNodeCoordinates(const int netcdf_descriptor,
                                     double *coordx,
                                     double *coordy,
                                     double *coordz)
{
   if (!coordx || !coordy) { return; }  // Only allow coordz to be NULL if dimensions < 3.

   int variable_id, netcdf_status;

   netcdf_status = nc_inq_varid(netcdf_descriptor, "coordx", &variable_id);
   netcdf_status = nc_get_var_double(netcdf_descriptor, variable_id, coordx);

   netcdf_status = nc_inq_varid(netcdf_descriptor, "coordy", &variable_id);
   netcdf_status = nc_get_var_double(netcdf_descriptor, variable_id, coordy);

   if (coordz)
   {
      netcdf_status = nc_inq_varid(netcdf_descriptor, "coordz", &variable_id);
      netcdf_status = nc_get_var_double(netcdf_descriptor, variable_id, coordz);
   }

   if (netcdf_status != NC_NOERR) { HandleNetCDFError(netcdf_status); }
}


static void ReadCubitNumElementsInBlock(const int netcdf_descriptor,
                                        const int num_element_blocks,
                                        std::vector<std::size_t> &num_elements_for_block)
{
   int netcdf_status, variable_id;

   // NB: need to add 1 for '\0' terminating character.
   const int buffer_size = NC_MAX_NAME + 1;

   char string_buffer[buffer_size];

   for (int iblock = 0; iblock < num_element_blocks; iblock++)
   {
      // Write variable name to buffer.
      snprintf(string_buffer, buffer_size, "num_el_in_blk%d", iblock + 1);

      // Set variable ID for variable name.
      netcdf_status = nc_inq_dimid(netcdf_descriptor, string_buffer, &variable_id);

      // Sets name and length for variable ID. We discard the name (just write to our string buffer).
      netcdf_status = nc_inq_dim(netcdf_descriptor, variable_id, string_buffer,
                                 &num_elements_for_block[iblock]);

      if (netcdf_status != NC_NOERR) { break; }
   }

   if (netcdf_status != NC_NOERR) { HandleNetCDFError(netcdf_status); }
}


static void ReadCubitNumNodesPerElement(const int netcdf_descriptor,
                                        const int num_element_blocks,
                                        size_t &num_nodes_per_element)
{
   int netcdf_status, variable_id;

   // NB: need to add 1 for '\0' terminating character.
   const int buffer_size = NC_MAX_NAME + 1;

   char string_buffer[buffer_size];

   size_t new_num_nodes_per_element, last_num_nodes_per_element;

   bool different_element_types_detected = false;

   for (int iblock = 0; iblock < num_element_blocks; iblock++)
   {
      // Write variable name to buffer.
      snprintf(string_buffer, buffer_size, "num_nod_per_el%d", iblock + 1);

      // Set variable ID.
      netcdf_status = nc_inq_dimid(netcdf_descriptor, string_buffer, &variable_id);

      // Set name and length. We discard the name.
      netcdf_status = nc_inq_dim(netcdf_descriptor, variable_id, string_buffer,
                                 &new_num_nodes_per_element);
      if (netcdf_status != NC_NOERR) { break; }

      // NB: currently can only support one element type!
      if (iblock > 0 && new_num_nodes_per_element != last_num_nodes_per_element)
      {
         different_element_types_detected = true;
         break;
      }

      last_num_nodes_per_element = new_num_nodes_per_element;
   }

   if (netcdf_status != NC_NOERR)
   {
      HandleNetCDFError(netcdf_status);
   }
   else if (different_element_types_detected)
   {
      MFEM_ABORT("Element blocks of different types are not supported!\n");
   }

   // Set:
   num_nodes_per_element = new_num_nodes_per_element;
}


static void ReadCubitDimensions(const int netcdf_descriptor,
                                size_t &num_dim,
                                size_t &num_nodes,
                                size_t &num_elem,
                                size_t &num_el_blk,
                                size_t &num_side_sets)
{
   int netcdf_status, variable_id;

   const int buffer_size = NC_MAX_NAME + 1;
   char string_buffer[buffer_size];

   netcdf_status = nc_inq_dimid(netcdf_descriptor, "num_dim", &variable_id);
   netcdf_status = nc_inq_dim(netcdf_descriptor, variable_id, string_buffer,
                              &num_dim);
   if (netcdf_status != NC_NOERR) { HandleNetCDFError(netcdf_status); }

   netcdf_status = nc_inq_dimid(netcdf_descriptor, "num_nodes", &variable_id);
   netcdf_status = nc_inq_dim(netcdf_descriptor, variable_id, string_buffer,
                              &num_nodes);
   if (netcdf_status != NC_NOERR) { HandleNetCDFError(netcdf_status); }

   netcdf_status = nc_inq_dimid(netcdf_descriptor, "num_elem", &variable_id);
   netcdf_status = nc_inq_dim(netcdf_descriptor, variable_id, string_buffer,
                              &num_elem);
   if (netcdf_status != NC_NOERR) { HandleNetCDFError(netcdf_status); }

   netcdf_status = nc_inq_dimid(netcdf_descriptor, "num_el_blk", &variable_id);
   netcdf_status = nc_inq_dim(netcdf_descriptor, variable_id, string_buffer,
                              &num_el_blk);
   if (netcdf_status != NC_NOERR) { HandleNetCDFError(netcdf_status); }

   netcdf_status = nc_inq_dimid(netcdf_descriptor, "num_side_sets", &variable_id);
   netcdf_status = nc_inq_dim(netcdf_descriptor, variable_id, string_buffer,
                              &num_side_sets);

   // Special case: there may be no sidesets. Ignore error.
   if (netcdf_status != NC_NOERR)
   {
      num_side_sets = 0;
   }
}


static void ReadCubitBoundaries(const int netcdf_descriptor,
                                const int num_boundaries,
                                vector<size_t> &num_boundary_elements,
                                vector<vector<int>> &boundary_elements,
                                vector<vector<int>> &boundary_sides)
{
   int netcdf_status, variable_id;

   const int buffer_size = NC_MAX_NAME + 1;
   char string_buffer[buffer_size];

   for (int iboundary = 0; iboundary < num_boundaries; iboundary++)
   {
      // 1. Extract number of elements/sides for boundary.
      size_t num_sides = 0;

      snprintf(string_buffer, buffer_size, "num_side_ss%d", iboundary + 1);

      netcdf_status = nc_inq_dimid(netcdf_descriptor, string_buffer, &variable_id);
      netcdf_status = nc_inq_dim(netcdf_descriptor, variable_id, string_buffer,
                                 &num_sides);

      if (netcdf_status != NC_NOERR) { break; }

      num_boundary_elements[iboundary] = num_sides;

      // 2. Extract elements and sides on each boundary.
      boundary_elements[iboundary].resize(num_sides); // (element, face) pairs.
      boundary_sides[iboundary].resize(num_sides);

      //
      snprintf(string_buffer, buffer_size, "elem_ss%d", iboundary + 1);

      netcdf_status = nc_inq_varid(netcdf_descriptor, string_buffer, &variable_id);
      netcdf_status = nc_get_var_int(netcdf_descriptor, variable_id,
                                     boundary_elements[iboundary].data());

      if (netcdf_status != NC_NOERR) { break; }

      //
      snprintf(string_buffer, buffer_size,"side_ss%d", iboundary + 1);

      netcdf_status = nc_inq_varid(netcdf_descriptor, string_buffer, &variable_id);
      netcdf_status = nc_get_var_int(netcdf_descriptor, variable_id,
                                     boundary_sides[iboundary].data());

      if (netcdf_status != NC_NOERR) { break; }
   }

   if (netcdf_status != NC_NOERR) { HandleNetCDFError(netcdf_status); }
}


static void ReadCubitElementBlocks(const int netcdf_descriptor,
                                   const int num_element_blocks, const int num_nodes_per_element,
                                   const vector<size_t> &num_elements_for_block,
                                   vector<vector<int>> &block_elements)
{
   int netcdf_status, variable_id;

   const int buffer_size = NC_MAX_NAME + 1;
   char string_buffer[buffer_size];

   for (int iblock = 0; iblock < num_element_blocks; iblock++)
   {
      block_elements[iblock].resize(
         num_elements_for_block[iblock]*num_nodes_per_element);

      // Write variable name to buffer.
      snprintf(string_buffer, buffer_size, "connect%d", iblock + 1);

      // Get variable ID and then set all nodes of element in block.
      netcdf_status = nc_inq_varid(netcdf_descriptor, string_buffer, &variable_id);
      netcdf_status = nc_get_var_int(netcdf_descriptor, variable_id,
                                     block_elements[iblock].data());

      if (netcdf_status != NC_NOERR) { break; }
   }

   if (netcdf_status != NC_NOERR) { HandleNetCDFError(netcdf_status); }
}


static void SetCubitElementAndFaceType2D(const int num_nodes_per_element,
                                         CubitElementType &cubit_element_type,
                                         CubitFaceType & cubit_face_type,
                                         int & num_element_linear_nodes)
{
   switch (num_nodes_per_element)
   {
      case 3:
      {
         cubit_element_type = ELEMENT_TRI3;
         cubit_face_type = FACE_EDGE2;
         num_element_linear_nodes = 3;
         break;
      }
      case 6:
      {
         cubit_element_type = ELEMENT_TRI6;
         cubit_face_type = FACE_EDGE3;
         num_element_linear_nodes = 3;
         break;
      }
      case 4:
      {
         cubit_element_type = ELEMENT_QUAD4;
         cubit_face_type = FACE_EDGE2;
         num_element_linear_nodes = 4;
         break;
      }
      case 9:
      {
         cubit_element_type = ELEMENT_QUAD9;
         cubit_face_type = FACE_EDGE3;
         num_element_linear_nodes = 4;
         break;
      }
      default:
      {
         MFEM_ABORT("Don't know what to do with a " << num_nodes_per_element <<
                    " node 2D element\n");
         break;
      }
   }
}


static void SetCubitElementAndFaceType3D(const int num_nodes_per_element,
                                         CubitElementType &cubit_element_type,
                                         CubitFaceType & cubit_face_type,
                                         int & num_element_linear_nodes)
{
   switch (num_nodes_per_element)
   {
      case 4:
      {
         cubit_element_type = ELEMENT_TET4;
         cubit_face_type = FACE_TRI3;
         num_element_linear_nodes = 4;
         break;
      }
      case 10:
      {
         cubit_element_type = ELEMENT_TET10;
         cubit_face_type = FACE_TRI6;
         num_element_linear_nodes = 4;
         break;
      }
      case 8:
      {
         cubit_element_type = ELEMENT_HEX8;
         cubit_face_type = FACE_QUAD4;
         num_element_linear_nodes = 8;
         break;
      }
      case 27:
      {
         cubit_element_type = ELEMENT_HEX27;
         cubit_face_type = FACE_QUAD9;
         num_element_linear_nodes = 8;
         break;
      }
      default:
      {
         MFEM_ABORT("Don't know what to do with a " << num_nodes_per_element <<
                    " node 3D element\n");
         break;
      }
   }
}


static void SetCubitElementAndFaceType(const int num_dimensions,
                                       const int num_nodes_per_element,
                                       CubitElementType &cubit_element_type,
                                       CubitFaceType &cubit_face_type,
                                       int & num_element_linear_nodes)
{
   switch (num_dimensions)
   {
      case 2:
      {
         SetCubitElementAndFaceType2D(num_nodes_per_element, cubit_element_type,
                                      cubit_face_type, num_element_linear_nodes);
         break;
      }
      case 3:
      {
         SetCubitElementAndFaceType3D(num_nodes_per_element, cubit_element_type,
                                      cubit_face_type, num_element_linear_nodes);
         break;
      }
      default:
      {
         MFEM_ABORT("Invalid dimension: " << num_dimensions);
         break;
      }
   }
}


static void SetCubitFaceInfo(const CubitFaceType cubit_face_type,
                             int & num_face_nodes,
                             int & num_face_linear_nodes)
{
   switch (cubit_face_type)
   {
      case FACE_EDGE2:
      {
         num_face_nodes = 2;
         num_face_linear_nodes = 2;
         break;
      }
      case FACE_EDGE3:
      {
         num_face_nodes = 3;
         num_face_linear_nodes = 2;
         break;
      }
      case FACE_TRI3:
      {
         num_face_nodes = 3;
         num_face_linear_nodes = 3;
         break;
      }
      case FACE_TRI6:
      {
         num_face_nodes = 6;
         num_face_linear_nodes = 3;
         break;
      }
      case FACE_QUAD4:
      {
         num_face_nodes = 4;
         num_face_linear_nodes = 4;
         break;
      }
      case FACE_QUAD9:
      {
         num_face_nodes = 9;
         num_face_linear_nodes = 4;
         break;
      }
      default:
      {
         MFEM_ABORT("Unsupported cubit face type encountered.");
         break;
      }
   }
}

static int GetOrderFromCubitElementType(const CubitElementType element_type)
{
   int order;

   switch (element_type)
   {
      case ELEMENT_TRI3:
      case ELEMENT_QUAD4:
      case ELEMENT_TET4:
      case ELEMENT_HEX8:
      {
         order = 1;
         break;
      }
      case ELEMENT_TRI6:
      case ELEMENT_QUAD9:
      case ELEMENT_TET10:
      case ELEMENT_HEX27:
      {
         order = 2;
         break;
      }
      default:
      {
         MFEM_ABORT("Unsupported cubit element type encountered.");
         break;
      }
   }

   return order;
}


static int GetCubitBlockIndexForElement(const int global_element_index,
                                        const int num_element_blocks,
                                        const int *start_of_block)
{
   int iblock = 0;

   while (iblock < num_element_blocks &&
          global_element_index >= start_of_block[iblock + 1])
   {
      iblock++;
   }

   if (iblock >= num_element_blocks)
   {
      MFEM_ABORT("Element is not part of any blocks.");
   }

   return iblock;
}

mfem::Element *NewElement(Mesh &mesh, Geometry::Type geom, const int *vertices,
                          const int attribute)
{
   Element *new_element = mesh.NewElement(geom);
   new_element->SetVertices(vertices);
   new_element->SetAttribute(attribute);
   return new_element;
}

/// @brief Returns a pointer to a new mfem::Element based on the provided cubit
/// element type. This is used to create the mesh elements from a Genesis file.
mfem::Element *CreateCubitElement(Mesh &mesh,
                                  const int cubit_element_type,
                                  const int *vertex_ids,
                                  const int block_id)
{
   switch (cubit_element_type)
   {
      case ELEMENT_TRI3:
      case ELEMENT_TRI6:
         return NewElement(mesh, Geometry::TRIANGLE, vertex_ids, block_id);
      case ELEMENT_QUAD4:
      case ELEMENT_QUAD9:
         return NewElement(mesh, Geometry::SQUARE, vertex_ids, block_id);
      case ELEMENT_TET4:
      case ELEMENT_TET10:
         return NewElement(mesh, Geometry::TETRAHEDRON, vertex_ids, block_id);
      case ELEMENT_HEX8:
      case ELEMENT_HEX27:
         return NewElement(mesh, Geometry::CUBE, vertex_ids, block_id);
      default:
         MFEM_ABORT("Unsupported cubit element type encountered.");
         return nullptr;
   }
}

/// @brief Returns a pointer to a new mfem::Element based on the provided cubit
/// face type. This is used to create the boundary elements from a Genesis file.
mfem::Element *CreateCubitBoundaryElement(Mesh &mesh,
                                          const int cubit_face_type,
                                          const int *vertex_ids,
                                          const int sideset_id)
{
   switch (cubit_face_type)
   {
      case FACE_EDGE2:
      case FACE_EDGE3:
         return NewElement(mesh, Geometry::SEGMENT, vertex_ids, sideset_id);
      case FACE_TRI3:
      case FACE_TRI6:
         return NewElement(mesh, Geometry::TRIANGLE, vertex_ids, sideset_id);
      case FACE_QUAD4:
      case FACE_QUAD9:
         return NewElement(mesh, Geometry::SQUARE, vertex_ids, sideset_id);
      default:
         MFEM_ABORT("Unsupported cubit face type encountered.");
         return nullptr;
   }
}

/// @brief The final step in constructing the mesh from a Genesis file. This is
/// only called if the mesh order == 2 (determined internally from the cubit
/// element type).
void FinalizeCubitSecondOrderMesh(Mesh &mesh,
                                  const int cubit_element_type,
                                  const int num_element_blocks,
                                  const int num_nodes_per_element,
                                  const int *start_of_block,
                                  const double *coordx,
                                  const double *coordy,
                                  const double *coordz,
                                  const vector<vector<int>> &element_blocks)
{
   int *mfem_to_genesis_map = nullptr;

   switch (cubit_element_type)
   {
      case ELEMENT_TRI6:
         mfem_to_genesis_map = (int *) mfem_to_genesis_tri6;
         break;
      case ELEMENT_QUAD9:
         mfem_to_genesis_map = (int *) mfem_to_genesis_quad9;
         break;
      case ELEMENT_TET10:
         mfem_to_genesis_map = (int *) mfem_to_genesis_tet10;
         break;
      case ELEMENT_HEX27:
         mfem_to_genesis_map = (int *) mfem_to_genesis_hex27;
         break;
      default:
         MFEM_ABORT("Something went wrong. Linear elements detected when order is 2.");
   }

   mesh.FinalizeTopology();

   // Define quadratic FE space.
   const int Dim = mesh.Dimension();
   FiniteElementCollection *fec = new H1_FECollection(2,3);
   FiniteElementSpace *fes = new FiniteElementSpace(&mesh, fec, Dim,
                                                    Ordering::byVDIM);
   GridFunction *Nodes = new GridFunction(fes);
   Nodes->MakeOwner(fec); // Nodes will destroy 'fec' and 'fes'
   mesh.SetNodalGridFunction(Nodes, true);

   for (int ielement = 0; ielement < mesh.GetNE(); ielement++)
   {
      Array<int> dofs;
      fes->GetElementDofs(ielement, dofs);

      Array<int> vdofs = dofs;   // Deep copy.
      fes->DofsToVDofs(vdofs);

      // Find block that element is part of.
      const int iblock = GetCubitBlockIndexForElement(ielement,
                                                      num_element_blocks,
                                                      start_of_block);

      // Find element offset in block.
      const int element_offset = ielement - start_of_block[iblock];
      const int node_offset    = element_offset * num_nodes_per_element;

      for (int jnode = 0; jnode < dofs.Size(); jnode++)
      {
         const int node_index = element_blocks[iblock][node_offset +
                                                       mfem_to_genesis_map[jnode] - 1] - 1;

         (*Nodes)(vdofs[jnode])     = coordx[node_index];
         (*Nodes)(vdofs[jnode] + 1) = coordy[node_index];

         if (Dim == 3)
         {
            (*Nodes)(vdofs[jnode] + 2) = coordz[node_index];
         }
      }
   }
}

}  // namespace cubit.


void Mesh::ReadCubit(const std::string &filename, int &curved, int &read_gf)
{
   using namespace cubit;

   read_gf  = 0;
   curved   = 0; // Set to 1 if mesh is curved.

   // Setup buffer used to write variable names to.
   int variable_id;

   // Open the file.
   int netcdf_status, netcdf_descriptor;

   netcdf_status = nc_open(filename.c_str(), NC_NOWRITE, &netcdf_descriptor);
   if (netcdf_status != NC_NOERR) { HandleNetCDFError(netcdf_status); }

   // Read important dimensions from file.
   size_t num_dimensions, num_nodes, num_elements, num_element_blocks,
          num_boundaries;

   ReadCubitDimensions(netcdf_descriptor, num_dimensions, num_nodes, num_elements,
                       num_element_blocks, num_boundaries);

   Dim = num_dimensions;

   // Read the number of elements for each block.
   std::vector<std::size_t> num_elements_for_block(num_element_blocks);

   ReadCubitNumElementsInBlock(netcdf_descriptor, num_element_blocks,
                               num_elements_for_block);

   // Read number of nodes for each element. NB: we currently only support
   // reading in a single type of element!
   size_t num_nodes_per_element;

   ReadCubitNumNodesPerElement(netcdf_descriptor, num_element_blocks,
                               num_nodes_per_element);

   // Determine the cubit element and face type.
   CubitElementType cubit_element_type;
   CubitFaceType cubit_face_type;
   int num_element_linear_nodes;

   SetCubitElementAndFaceType(num_dimensions, num_nodes_per_element,
                              cubit_element_type,
                              cubit_face_type, num_element_linear_nodes);

   // Read the face info. NB: "linear" refers to nodes on the edges.
   int num_face_nodes, num_face_linear_nodes;

   SetCubitFaceInfo(cubit_face_type, num_face_nodes, num_face_linear_nodes);

   // Read the (element, corresponding side) on each of the boundaries.
   vector<size_t> num_boundary_elements(num_boundaries);

   vector<vector<int>> boundary_elements(num_boundaries);
   vector<vector<int>> boundary_sides(num_boundaries);

   ReadCubitBoundaries(netcdf_descriptor, num_boundaries, num_boundary_elements,
                       boundary_elements, boundary_sides);

   // Read the boundary ids.
   vector<int> boundary_ids;

   if (num_boundaries > 0)
   {
      boundary_ids.resize(num_boundaries);

      netcdf_status = nc_inq_varid(netcdf_descriptor, "ss_prop1", &variable_id);
      netcdf_status = nc_get_var_int(netcdf_descriptor, variable_id,
                                     boundary_ids.data());

      if (netcdf_status != NC_NOERR) { HandleNetCDFError(netcdf_status); }
   }

   // Read the xyz coordinates for each node.
   vector<double> coordx(num_nodes);
   vector<double> coordy(num_nodes);
   vector<double> coordz(num_dimensions == 3 ? num_nodes : 0);

   ReadCubitNodeCoordinates(netcdf_descriptor, coordx.data(), coordy.data(),
                            coordz.data());

   // Read the elements that make-up each block.
   vector<vector<int>> block_elements(num_element_blocks);

   ReadCubitElementBlocks(netcdf_descriptor, num_element_blocks,
                          num_nodes_per_element, num_elements_for_block,
                          block_elements);

   // Read the block IDs.
   vector<int> block_ids(num_element_blocks);

   {
      netcdf_status = nc_inq_varid(netcdf_descriptor, "eb_prop1", &variable_id);
      netcdf_status = nc_get_var_int(netcdf_descriptor, variable_id,
                                     block_ids.data());

      if (netcdf_status != NC_NOERR) { HandleNetCDFError(netcdf_status); }
   }

   // Create an array holding the index of the first element in each block. This
   // will allow the determination of the block that each element is in.
   vector<int> start_of_block(num_element_blocks + 1);

   start_of_block[0] = 0;

   for (size_t iblock = 1; iblock < num_element_blocks + 1; iblock++)
   {
      start_of_block[iblock] = start_of_block[iblock - 1] +
                               num_elements_for_block[iblock - 1];
   }

   // Iterate over each boundary. For each boundary, we run through the
   // (element, side) pairs and extract the face nodes of each element on the
   // corresponding side.
   vector<vector<int>> boundary_nodes(num_boundaries);

   // Iterate over boundaries.
   for (size_t iboundary = 0; iboundary < num_boundaries; iboundary++)
   {
      const int num_elements_on_boundary = num_boundary_elements[iboundary];
      const int num_nodes_on_boundary = num_elements_on_boundary * num_face_nodes;

      boundary_nodes[iboundary].resize(num_nodes_on_boundary);

      // Iterate over (element, side) pairs on boundary.
      for (int jelement = 0; jelement < num_elements_on_boundary; jelement++)
      {
         const int element_global_index = boundary_elements[iboundary][jelement] - 1;

         // NB: Sides numbers start at 1.
         const int this_side = boundary_sides[iboundary][jelement];

         // Determine the block the element is part-of.
         const int iblock = GetCubitBlockIndexForElement(element_global_index,
                                                         num_element_blocks,
                                                         start_of_block.data());

         const int element_block_offset = element_global_index - start_of_block[iblock];
         const int node_block_offset    = element_block_offset * num_nodes_per_element;

         // Iterate over the element's face nodes on the matching side.
         for (int knode = 0; knode < num_face_nodes; knode++)
         {
            int inode;

            switch (cubit_element_type)
            {
               case (ELEMENT_TRI3):
               {
                  inode = cubit_side_map_tri3[this_side - 1][knode];
                  break;
               }
               case (ELEMENT_TRI6):
               {
                  inode = cubit_side_map_tri6[this_side - 1][knode];
                  break;
               }
               case (ELEMENT_QUAD4):
               {
                  inode = cubit_side_map_quad4[this_side - 1][knode];
                  break;
               }
               case (ELEMENT_QUAD9):
               {
                  inode = cubit_side_map_quad9[this_side - 1][knode];
                  break;
               }
               case (ELEMENT_TET4):
               {
                  inode = cubit_side_map_tet4[this_side - 1][knode];
                  break;
               }
               case (ELEMENT_TET10):
               {
                  inode = cubit_side_map_tet10[this_side - 1][knode];
                  break;
               }
               case (ELEMENT_HEX8):
               {
                  inode = cubit_side_map_hex8[this_side - 1][knode];
                  break;
               }
               case (ELEMENT_HEX27):
               {
                  inode = cubit_side_map_hex27[this_side - 1][knode];
                  break;
               }
               default:
               {
                  MFEM_ABORT("Unsupported element type encountered.\n");
                  break;
               }
            }

            boundary_nodes[iboundary][jelement * num_face_nodes + knode] =
               block_elements[iblock][node_block_offset + inode - 1];
         }
      }
   }

   // We need another node ID mapping since MFEM needs contiguous vertex ids.
   vector<int> unique_vertex_ids;

   for (size_t iblock = 0; iblock < num_element_blocks; iblock++)
   {
      const vector<int> &nodes_in_block = block_elements[iblock];

      for (size_t jelement = 0; jelement < num_elements_for_block[iblock]; jelement++)
      {
         const int element_block_offset = jelement * num_nodes_per_element;

         for (int knode = 0; knode < num_element_linear_nodes; knode++)
         {
            unique_vertex_ids.push_back(nodes_in_block[element_block_offset + knode]);
         }
      }
   }

   // Sort and only retain unique node IDs.
   std::sort(unique_vertex_ids.begin(), unique_vertex_ids.end());

   auto new_end = std::unique(unique_vertex_ids.begin(), unique_vertex_ids.end());
   unique_vertex_ids.resize(std::distance(unique_vertex_ids.begin(), new_end));

   // unique_vertex_ids now contains a 1-based sorted list of node IDs for each
   // node used by the mesh. We now create a map by running over the node IDs
   // and remapping to contiguous 1-based integers.
   // ie. [1, 4, 5, 8, 9] --> [1, 2, 3, 4, 5].
   std::map<int,int> cubit_to_mfem_vertex_map;

   for (size_t ivertex = 0; ivertex < unique_vertex_ids.size(); ivertex++)
   {
      const int key     = unique_vertex_ids[ivertex];
      const int value   = ivertex + 1;

      cubit_to_mfem_vertex_map[key] = value;
   }

   //
   // Load up the vertices.
   //
   NumOfVertices = unique_vertex_ids.size();
   vertices.SetSize(NumOfVertices);

   for (int ivertex = 0; ivertex < NumOfVertices; ivertex++)
   {
      const int original_1based_id = unique_vertex_ids[ivertex];

      vertices[ivertex](0) = coordx[original_1based_id - 1];
      vertices[ivertex](1) = coordy[original_1based_id - 1];

      if (Dim == 3)
      {
         vertices[ivertex](2) = coordz[original_1based_id - 1];
      }
   }

   //
   // Now load the elements.
   //
   NumOfElements = num_elements;
   elements.SetSize(num_elements);

   std::vector<int> renumbered_vertex_ids(max(num_element_linear_nodes,
                                              num_face_linear_nodes));

   int element_counter = 0;

   // Iterate over blocks.
   for (size_t iblock = 0; iblock < num_element_blocks; iblock++)
   {
      const vector<int> &nodes_ids_for_block = block_elements[iblock];

      // Iterate over elements in block.
      for (size_t jelement = 0; jelement < num_elements_for_block[iblock]; jelement++)
      {
         // Iterate over linear nodes in block.
         for (int knode = 0; knode < num_element_linear_nodes; knode++)
         {
            const int node_id = nodes_ids_for_block[jelement * num_nodes_per_element +
                                                             knode];

            // Renumber using the mapping.
            renumbered_vertex_ids[knode] = cubit_to_mfem_vertex_map[node_id] - 1;
         }

         // Create element.
         elements[element_counter++] = CreateCubitElement(*this, cubit_element_type,
                                                          renumbered_vertex_ids.data(),
                                                          block_ids[iblock]);
      }
   }

   //
   // Load up the boundary elements.
   //
   NumOfBdrElements = 0;
   for (size_t iboundary = 0; iboundary < num_boundaries; iboundary++)
   {
      NumOfBdrElements += num_boundary_elements[iboundary];
   }

   boundary.SetSize(NumOfBdrElements);

   int boundary_counter = 0;

   // Iterate over boundaries.
   for (size_t iboundary = 0; iboundary < num_boundaries; iboundary++)
   {
      const vector<int> &nodes_on_boundary = boundary_nodes[iboundary];

      // Iterate over elements on boundary.
      for (size_t jelement = 0; jelement < num_boundary_elements[iboundary];
           jelement++)
      {
         // Iterate over element's face linear nodes.
         for (int knode = 0; knode < num_face_linear_nodes; knode++)
         {
            const int node_id = nodes_on_boundary[jelement * num_face_nodes + knode];

            // Renumber using the mapping.
            renumbered_vertex_ids[knode] = cubit_to_mfem_vertex_map[node_id] - 1;
         }

         // Create boundary element.
         boundary[boundary_counter++] = CreateCubitBoundaryElement(*this,
                                                                   cubit_face_type,
                                                                   renumbered_vertex_ids.data(),
                                                                   boundary_ids[iboundary]);
      }
   }

   // Additional setup for second order.
   const int order = GetOrderFromCubitElementType(cubit_element_type);

   if (order == 2)
   {
      curved = 1;

      FinalizeCubitSecondOrderMesh(*this,
                                   cubit_element_type,
                                   num_element_blocks,
                                   num_nodes_per_element,
                                   start_of_block.data(),
                                   coordx.data(),
                                   coordy.data(),
                                   coordz.data(),
                                   block_elements);
   }

   // Clean up all netcdf stuff.
   nc_close(netcdf_descriptor);
}

#endif // #ifdef MFEM_USE_NETCDF

} // namespace mfem
