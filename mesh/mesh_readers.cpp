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
#include "ncnurbs.hpp"
#include "../fem/fem.hpp"
#include "../general/binaryio.hpp"
#include "../general/text.hpp"
#include "../general/tinyxml2.h"

#include <iostream>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <map>

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
      if (!compressed) { return static_cast<int>(HeaderEntrySize()); }
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
         std::vector<size_t> header(nblocks + 2);
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
         MFEM_VERIFY(size_t(sizeof(F)*n) == size_t(dest_ptr - dest_start),
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
         MFEM_VERIFY(sizeof(F)*n == ReadHeaderEntry(header_buf),
                     "AppendedData: wrong data size");
      }

      if (std::is_same_v<T, F>)
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
         int nblocks_b64 = static_cast<int>(bin_io::NumBase64Chars(HeaderEntrySize()));
         bin_io::DecodeBase64(txt, nblocks_b64, nblocks_buf);
         std::vector<char> data, header;
         // Compute number of characters needed to encode header in base 64,
         // then round to nearest multiple of 4 to take padding into account.
         int header_b64 = static_cast<int>(bin_io::NumBase64Chars(NumHeaderBytes(
                                                                     nblocks_buf.data())));
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

   // Read the element attributes, which are stored as CellData named either
   // "material" or "attribute". We prioritize "material" over "attribute" for
   // backwards compatibility.
   Array<int> cell_attributes;
   bool found_attributes = false;
   for (const XMLElement *cell_data_xml = piece->FirstChildElement();
        cell_data_xml != NULL;
        cell_data_xml = cell_data_xml->NextSiblingElement())
   {
      const bool is_cell_data =
         StringCompare(cell_data_xml->Name(), "CellData");
      const bool is_material =
         StringCompare(cell_data_xml->Attribute("Scalars"), "material");
      const bool is_attribute =
         StringCompare(cell_data_xml->Attribute("Scalars"), "attribute");
      if (is_cell_data && (is_material || (is_attribute && !found_attributes)))
      {
         found_attributes = true;
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
   bool found_attributes = false;
   while ((input.good()))
   {
      getline(input, buff);
      if (buff.rfind("POINT_DATA") == 0)
      {
         break; // We have entered the POINT_DATA block. Quit.
      }
      else if (buff.rfind("SCALARS material") == 0 ||
               (buff.rfind("SCALARS attribute") == 0 && !found_attributes))
      {
         found_attributes = true;
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
                         bool spacing, bool nc)
{
   NURBSext = nc ? new NCNURBSExtension(input, spacing):
              new NURBSExtension(input, spacing);

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
      const int vdim = NURBSext->GetPatchSpaceDimension();
      FiniteElementSpace *fes = new FiniteElementSpace(this, fec, vdim,
                                                       Ordering::byVDIM);
      Nodes = new GridFunction(fes);
      Nodes->MakeOwner(fec);
      NURBSext->SetCoordsFromPatches(*Nodes, vdim);
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

const int mfem_to_genesis_pyramid14[14] =
{
   1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14
};

const int mfem_to_genesis_wedge18[18] =
{
   1, 2, 3, 4, 5, 6, 7, 8, 9, 13, 14, 15, 10, 11, 12, 16, 17, 18
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
   {1,2}, // 1
   {2,3}, // 2
   {3,1}, // 3
};

const int cubit_side_map_quad4[4][2] =
{
   {1,2}, // 1
   {2,3}, // 2
   {3,4}, // 3
   {4,1}, // 4
};

const int cubit_side_map_tet4[4][3] =
{
   {1,2,4}, // 1
   {2,3,4}, // 2
   {1,4,3}, // 3
   {1,3,2}  // 4
};

const int cubit_side_map_hex8[6][4] =
{
   {1,2,6,5},  // 1 <-- Exodus II side_ids
   {2,3,7,6},  // 2
   {3,4,8,7},  // 3
   {1,5,8,4},  // 4
   {1,4,3,2},  // 5
   {5,6,7,8}   // 6
};

const int cubit_side_map_wedge6[5][4] =
{
   {1,2,5,4},  // 1 (Quad4)
   {2,3,6,5},  // 2
   {3,1,4,6},  // 3
   {1,3,2,0},  // 4 (Tri3; NB: 0 is placeholder!)
   {4,5,6,0}   // 5
};

const int cubit_side_map_pyramid5[5][4] =
{
   {1, 2, 5, 0},  // 1 (Tri3)
   {2, 3, 5, 0},  // 2
   {3, 4, 5, 0},  // 3
   {1, 5, 4, 0},  // 4
   {1, 4, 3, 2}   // 5 (Quad4)
};

enum CubitFaceType
{
   FACE_EDGE2,
   FACE_EDGE3,
   FACE_TRI3,
   FACE_TRI6,
   FACE_QUAD4,
   FACE_QUAD9  // Order = 2; center node.
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
   ELEMENT_HEX27,
   ELEMENT_WEDGE6,
   ELEMENT_WEDGE18,
   ELEMENT_PYRAMID5,
   ELEMENT_PYRAMID14
};


/**
 * CubitElement
 *
 * Stores information about a particular element.
 */
class CubitElement
{
public:
   /// Default constructor.
   CubitElement(CubitElementType element_type);
   CubitElement() = delete;

   /// Destructor.
   ~CubitElement() = default;

   /// Returns the Cubit element type.
   inline CubitElementType GetElementType() const { return _element_type; }

   /// Returns the face type for a specified face. NB: sides have 1-based indexing.
   CubitFaceType GetFaceType(size_t side_id = 1) const;

   /// Returns the number of faces.
   inline size_t GetNumFaces() const { return _num_faces; }

   /// Returns the number of vertices.
   inline size_t GetNumVertices() const { return _num_vertices; }

   /// Returns the number of nodes (vertices + higher-order control points).
   inline size_t GetNumNodes() const { return _num_nodes; }

   /// Returns the number of vertices for a particular face.
   size_t GetNumFaceVertices(size_t iface = 1) const;

   /// Returns the order of the element.
   inline uint8_t GetOrder() const { return _order; }

   /// Creates an MFEM equivalent element using the supplied vertex IDs and block ID.
   Element * BuildElement(Mesh & mesh, const int * vertex_ids,
                          const int block_id) const;

   /// Creates an MFEM boundary element using the supplied vertex IDs and block ID.
   Element * BuildBoundaryElement(Mesh & mesh, const int iface,
                                  const int * vertex_ids, const int sideset_id) const;

   /// Static method returning the element type for a given number of nodes per element and dimension.
   static CubitElementType GetElementType(size_t num_nodes,
                                          uint8_t dimension = 3);
protected:
   /// Static method which returns the 2D Cubit element type for the number of nodes per element.
   static CubitElementType Get2DElementType(size_t num_nodes);

   /// Static method which returns the 3D Cubit element type for the number of nodes per element.
   static CubitElementType Get3DElementType(size_t num_nodes);

   /// Creates a new MFEM element. Used internally in BuildElement and BuildBoundaryElement.
   Element * NewElement(Mesh & mesh, Geometry::Type geom, const int *vertices,
                        const int attribute) const;

private:
   CubitElementType _element_type;

   uint8_t _order;

   size_t _num_vertices;
   size_t _num_faces;
   size_t _num_nodes;
};

CubitElement::CubitElement(CubitElementType element_type)
{
   _element_type = element_type;

   switch (element_type)
   {
      case ELEMENT_TRI3:   // 2D.
         _order = 1;
         _num_vertices = 3;
         _num_nodes = 3;
         _num_faces = 3;
         break;
      case ELEMENT_TRI6:
         _order = 2;
         _num_vertices = 3;
         _num_nodes = 6;
         _num_faces = 3;
         break;
      case ELEMENT_QUAD4:
         _order = 1;
         _num_vertices = 4;
         _num_nodes = 4;
         _num_faces = 4;
         break;
      case ELEMENT_QUAD9:
         _order = 2;
         _num_vertices = 4;
         _num_nodes = 9;
         _num_faces = 4;
         break;
      case ELEMENT_TET4:   // 3D.
         _order = 1;
         _num_vertices = 4;
         _num_nodes = 4;
         _num_faces = 4;
         break;
      case ELEMENT_TET10:
         _order = 2;
         _num_vertices = 4;
         _num_nodes = 10;
         _num_faces = 4;
         break;
      case ELEMENT_HEX8:
         _order = 1;
         _num_vertices = 8;
         _num_nodes = 8;
         _num_faces = 6;
         break;
      case ELEMENT_HEX27:
         _order = 2;
         _num_vertices = 8;
         _num_nodes = 27;
         _num_faces = 6;
         break;
      case ELEMENT_WEDGE6:
         _order = 1;
         _num_vertices = 6;
         _num_nodes = 6;
         _num_faces = 5;
         break;
      case ELEMENT_WEDGE18:
         _order = 2;
         _num_vertices = 6;
         _num_nodes = 18;
         _num_faces = 5;
         break;
      case ELEMENT_PYRAMID5:
         _order = 1;
         _num_vertices = 5;
         _num_nodes = 5;
         _num_faces = 5;
         break;
      case ELEMENT_PYRAMID14:
         _order = 2;
         _num_vertices = 5;
         _num_nodes = 14;
         _num_faces = 5;
         break;
      default:
         MFEM_ABORT("Unsupported Cubit element type " << element_type << ".");
         break;
   }
}

CubitElementType CubitElement::Get3DElementType(size_t num_nodes)
{
   switch (num_nodes)
   {
      case 4:
         return ELEMENT_TET4;
      case 10:
         return ELEMENT_TET10;
      case 8:
         return ELEMENT_HEX8;
      case 27:
         return ELEMENT_HEX27;
      case 6:
         return ELEMENT_WEDGE6;
      case 18:
         return ELEMENT_WEDGE18;
      case 5:
         return ELEMENT_PYRAMID5;
      case 14:
         return ELEMENT_PYRAMID14;
      default:
         MFEM_ABORT("Unsupported 3D element with " << num_nodes << " nodes.");
   }
}

CubitElementType CubitElement::Get2DElementType(size_t num_nodes)
{
   switch (num_nodes)
   {
      case 3:
         return ELEMENT_TRI3;
      case 6:
         return ELEMENT_TRI6;
      case 4:
         return ELEMENT_QUAD4;
      case 9:
         return ELEMENT_QUAD9;
      default:
         MFEM_ABORT("Unsupported 2D element with " << num_nodes << " nodes.");
   }
}

CubitElementType CubitElement::GetElementType(size_t num_nodes,
                                              uint8_t dimension)
{
   if (dimension == 2)
   {
      return Get2DElementType(num_nodes);
   }
   else if (dimension == 3)
   {
      return Get3DElementType(num_nodes);
   }
   else
   {
      MFEM_ABORT("Unsupported Cubit dimension " << dimension << ".");
   }
}

CubitFaceType CubitElement::GetFaceType(size_t side_id) const
{
   // NB: 1-based indexing. See Exodus II file format specifications.
   bool valid_id = (side_id >= 1 &&
                    side_id <= GetNumFaces());
   if (!valid_id)
   {
      MFEM_ABORT("Encountered invalid side ID: " << side_id << ".");
   }

   switch (_element_type)
   {
      case ELEMENT_TRI3:      // 2D.
         return FACE_EDGE2;
      case ELEMENT_TRI6:
         return FACE_EDGE3;
      case ELEMENT_QUAD4:
         return FACE_EDGE2;
      case ELEMENT_QUAD9:
         return FACE_EDGE3;
      case ELEMENT_TET4:      // 3D.
         return FACE_TRI3;
      case ELEMENT_TET10:
         return FACE_TRI6;
      case ELEMENT_HEX8:
         return FACE_QUAD4;
      case ELEMENT_HEX27:
         return FACE_QUAD9;
      case ELEMENT_WEDGE6:    // [Quad4, Quad4, Quad4, Tri3, Tri3]
         return (side_id < 4 ? FACE_QUAD4 : FACE_TRI3);
      case ELEMENT_WEDGE18:   // [Quad9, Quad9, Quad9, Tri6, Tri6]
         return (side_id < 4 ? FACE_QUAD9 : FACE_TRI6);
      case ELEMENT_PYRAMID5:  // [Tri3, Tri3, Tri3, Tri3, Quad4]
         return (side_id < 5 ? FACE_TRI3 : FACE_QUAD4);
      case ELEMENT_PYRAMID14: // [Tri6, Tri6, Tri6, Tri6, Quad9]
         return (side_id < 5 ? FACE_TRI6 : FACE_QUAD9);
      default:
         MFEM_ABORT("Unknown element type: " << _element_type << ".");
   }
}


size_t CubitElement::GetNumFaceVertices(size_t side_id) const
{
   switch (GetFaceType(side_id))
   {
      case FACE_EDGE2:
      case FACE_EDGE3:
         return 2;
      case FACE_TRI3:
      case FACE_TRI6:
         return 3;
      case FACE_QUAD4:
      case FACE_QUAD9:
         return 4;
      default:
         MFEM_ABORT("Unrecognized Cubit face type " << GetFaceType(side_id) << ".");
   }
}


mfem::Element * CubitElement::NewElement(Mesh &mesh, Geometry::Type geom,
                                         const int *vertices,
                                         const int attribute) const
{
   Element *new_element = mesh.NewElement(geom);
   new_element->SetVertices(vertices);
   new_element->SetAttribute(attribute);
   return new_element;
}


mfem::Element * CubitElement::BuildElement(Mesh &mesh,
                                           const int *vertex_ids,
                                           const int block_id) const
{
   switch (GetElementType())
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
      case ELEMENT_WEDGE6:
      case ELEMENT_WEDGE18:
         return NewElement(mesh, Geometry::PRISM, vertex_ids, block_id);
      case ELEMENT_PYRAMID5:
      case ELEMENT_PYRAMID14:
         return NewElement(mesh, Geometry::PYRAMID, vertex_ids, block_id);
      default:
         MFEM_ABORT("Unsupported Cubit element type encountered.");
   }
}


mfem::Element * CubitElement::BuildBoundaryElement(Mesh &mesh,
                                                   const int face_id,
                                                   const int *vertex_ids,
                                                   const int sideset_id) const
{
   switch (GetFaceType(face_id))
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
         MFEM_ABORT("Unsupported Cubit face type encountered.");
   }
}

/**
 * CubitBlock
 *
 * Stores the information about each block in a mesh. Each block can contain a different
 * element type (although all element types must be of the same order and dimension).
 */
class CubitBlock
{
public:
   CubitBlock() = delete;
   ~CubitBlock() = default;

   /**
    * Default initializer.
    */
   CubitBlock(int dimension);

   /**
    * Returns a constant reference to the element info for a particular block.
    */
   const CubitElement & GetBlockElement(int block_id) const;

   /**
    * Call to add each block individually.
    */
   void AddBlockElement(int block_id, CubitElementType element_type);

   /**
    * Accessors.
    */
   uint8_t GetOrder() const;
   inline uint8_t GetDimension() const { return _dimension; }

   inline size_t GetNumBlocks() const { return BlockIDs().size(); }
   inline bool HasBlocks() const { return !BlockIDs().empty(); }

protected:
   /**
    * Checks that the order of a new block element matches the order of existing blocks. Called
    * internally in method "addBlockElement".
    */
   void CheckElementBlockIsCompatible(const CubitElement & new_block_element)
   const;

   /**
    * Reset all block elements. Called internally in initializer.
    */
   void ClearBlockElements();

   /**
    * Helper methods.
    */
   inline const std::set<int> & BlockIDs() const { return _block_ids; }

   bool HasBlockID(int block_id) const;
   bool ValidBlockID(int block_id) const;
   bool ValidDimension(int dimension) const;

private:
   /**
    * Stores all block IDs.
    */
   std::set<int> _block_ids;

   /**
    * Maps from block ID to element.
    */
   std::map<int, CubitElement> _block_element_for_block_id;

   /**
    * Dimension and order of block elements.
    */
   uint8_t _dimension;
   uint8_t _order;
};

CubitBlock::CubitBlock(int dimension)
{
   if (!ValidDimension(dimension))
   {
      MFEM_ABORT("Invalid dimension '" << dimension << "' specified.");
   }

   _dimension = dimension;

   ClearBlockElements();
}

void
CubitBlock::AddBlockElement(int block_id, CubitElementType element_type)
{
   if (HasBlockID(block_id))
   {
      MFEM_ABORT("Block with ID '" << block_id << "' has already been added.");
   }
   else if (!ValidBlockID(block_id))
   {
      MFEM_ABORT("Illegal block ID '" << block_id << "'.");
   }

   CubitElement block_element = CubitElement(element_type);

   /**
    * Check element is compatible with existing element blocks.
    */
   CheckElementBlockIsCompatible(block_element);

   if (!HasBlocks()) // Set order of elements.
   {
      _order = block_element.GetOrder();
   }

   _block_ids.insert(block_id);
   _block_element_for_block_id.emplace(block_id,
                                       block_element);
}

uint8_t
CubitBlock::GetOrder() const
{
   if (!HasBlocks())
   {
      MFEM_ABORT("No elements have been added.");
   }

   return _order;
}

void
CubitBlock::ClearBlockElements()
{
   _order = 0;
   _block_ids.clear();
   _block_element_for_block_id.clear();
}

bool
CubitBlock::HasBlockID(int block_id) const
{
   return (_block_ids.count(block_id) > 0);
}

bool
CubitBlock::ValidBlockID(int block_id) const
{
   return (block_id > 0); // 1-based indexing.
}

bool
CubitBlock::ValidDimension(int dimension) const
{
   return (dimension == 2 || dimension == 3);
}

const CubitElement &
CubitBlock::GetBlockElement(int block_id) const
{
   if (!HasBlockID(block_id))
   {
      MFEM_ABORT("No element info for block ID '" << block_id << "'.");
   }

   return _block_element_for_block_id.at(block_id);
}

void
CubitBlock::CheckElementBlockIsCompatible(const CubitElement &
                                          new_block_element) const
{
   if (!HasBlocks())
   {
      return;
   }

   // Enforce block orders to be the same for now.
   if (GetOrder() != new_block_element.GetOrder())
   {
      MFEM_ABORT("All block elements must be of the same order.");
   }
}

/**
 * Lightweight wrapper around NetCDF C functions.
 */
class NetCDFReader
{
public:
   NetCDFReader() = delete;
   NetCDFReader(const std::string fname);

   ~NetCDFReader();

   /// Returns true if variable id for that name exists.
   bool HasVariable(const char * name);

   /// Read variable info from file and write to int buffer.
   void ReadVariable(const char * name, int * data);

   /// Read variable info from file and write to double buffer.
   void ReadVariable(const char * name, double * data);

   /// Returns true if dimension id for that name exists.
   bool HasDimension(const char * name);

   /// Read dimension info from file.
   void ReadDimension(const char * name, size_t *dimension);

   /// Build the map from quantity ID to name, e.g. block ID to block name or boundary ID to boundary name
   void BuildIDToNameMap(const vector<int> & ids,
                         unordered_map<int, string> & ids_to_names,
                         const string & quantity_name);

protected:
   /// Called internally. Calls HandleNetCDFError if _netcdf_status is not "NC_NOERR".
   void CheckForNetCDFError();

   /// Called in "ReadVariable" methods to extract variable id.
   int ReadVariableID(const char * name);

   /// Called in "ReadDimension" to extract dimension id.
   int ReadDimensionID(const char * name);

private:
   /// Calls MFEM_Abort with string description of NetCDF error.
   void HandleNetCDFError(const int error_code);

   int _netcdf_status{NC_NOERR};
   int _netcdf_descriptor;

   /// Internal buffer. Used in ReadDimension to write unwanted name to.
   char *_name_buffer{NULL};
};


NetCDFReader::NetCDFReader(const std::string fname)
{
   _netcdf_status = nc_open(fname.c_str(), NC_NOWRITE, &_netcdf_descriptor);
   CheckForNetCDFError();

   // NB: add byte for '\0' terminating char.
   _name_buffer = new char[NC_MAX_NAME + 1];
}

NetCDFReader::~NetCDFReader()
{
   _netcdf_status = nc_close(_netcdf_descriptor);
   CheckForNetCDFError();

   if (_name_buffer)
   {
      delete[] _name_buffer;
   }
}

void NetCDFReader::CheckForNetCDFError()
{
   if (_netcdf_status != NC_NOERR)
   {
      HandleNetCDFError(_netcdf_status);
   }
}

void NetCDFReader::HandleNetCDFError(const int error_code)
{
   MFEM_ABORT("Fatal NetCDF error: " << nc_strerror(error_code));
}

int NetCDFReader::ReadVariableID(const char * var_name)
{
   int variable_id;

   _netcdf_status = nc_inq_varid(_netcdf_descriptor, var_name,
                                 &variable_id);
   CheckForNetCDFError();

   return variable_id;
}

int NetCDFReader::ReadDimensionID(const char * name)
{
   int dim_id;

   _netcdf_status = nc_inq_dimid(_netcdf_descriptor, name, &dim_id);
   CheckForNetCDFError();

   return dim_id;
}

void NetCDFReader::ReadDimension(const char * name, size_t *dimension)
{
   const int dimension_id = ReadDimensionID(name);

   // NB: ignore name output (write to private buffer).
   _netcdf_status = nc_inq_dim(_netcdf_descriptor, dimension_id, _name_buffer,
                               dimension);
   CheckForNetCDFError();
}

bool NetCDFReader::HasVariable(const char * name)
{
   int var_id;
   const int status = nc_inq_varid(_netcdf_descriptor, name, &var_id);

   switch (status)
   {
      case NC_NOERR:    // Found!
         return true;
      case NC_ENOTVAR:  // Not found.
         return false;
      default:
         HandleNetCDFError(status);
         return false;
   }
}

bool NetCDFReader::HasDimension(const char * name)
{
   int dim_id;
   const int status = nc_inq_dimid(_netcdf_descriptor, name, &dim_id);

   switch (status)
   {
      case NC_NOERR:    // Found!
         return true;
      case NC_EBADDIM:  // Not found.
         return false;
      default:
         HandleNetCDFError(status);
         return false;
   }
}

void NetCDFReader::ReadVariable(const char * name, int * data)
{
   const int variable_id = ReadVariableID(name);

   _netcdf_status = nc_get_var_int(_netcdf_descriptor, variable_id, data);
   CheckForNetCDFError();
}


void NetCDFReader::ReadVariable(const char * name, double * data)
{
   const int variable_id = ReadVariableID(name);

   _netcdf_status = nc_get_var_double(_netcdf_descriptor, variable_id, data);
   CheckForNetCDFError();
}


void NetCDFReader::BuildIDToNameMap(const vector<int> & ids,
                                    unordered_map<int, string> & ids_to_names,
                                    const string & quantity_name)
{
   int varid_names;

   // Find the variable ID for the given quantity_name (e.g. eb_names, ss_names)
   _netcdf_status = nc_inq_varid(_netcdf_descriptor, quantity_name.c_str(),
                                 &varid_names);
   // It's possible the netcdf file doesn't contain the variable, in which case
   // there's nothing to do
   if (_netcdf_status == NC_ENOTVAR)
   {
      return;
   }
   else
   {
      CheckForNetCDFError();
   }

   // Get type of quantity_name
   nc_type var_type;
   _netcdf_status = nc_inq_vartype(_netcdf_descriptor, varid_names,
                                   &var_type);
   CheckForNetCDFError();

   if (var_type == NC_CHAR)
   {
      int dimids_names[2], names_ndim;
      size_t num_names, name_len;

      _netcdf_status = nc_inq_varndims(_netcdf_descriptor, varid_names,
                                       &names_ndim);
      CheckForNetCDFError();
      MFEM_ASSERT(names_ndim == 2, "This variable should have two dimensions");

      _netcdf_status = nc_inq_vardimid(_netcdf_descriptor, varid_names,
                                       dimids_names);
      CheckForNetCDFError();

      _netcdf_status = nc_inq_dimlen(_netcdf_descriptor, dimids_names[0], &num_names);
      CheckForNetCDFError();
      MFEM_ASSERT(num_names == ids.size(),
                  "The block id and block name lengths don't match");
      // Check the maximum string length
      _netcdf_status = nc_inq_dimlen(_netcdf_descriptor, dimids_names[1], &name_len);
      CheckForNetCDFError();

      // Read the block names
      vector<char> names(ids.size() * name_len);
      _netcdf_status = nc_get_var_text(_netcdf_descriptor, varid_names,
                                       names.data());
      CheckForNetCDFError();

      for (size_t i = 0; i < ids.size(); ++i)
      {
         string name(&names[i * name_len], name_len);
         // shorten string
         name.resize(name.find('\0'));
         ids_to_names[ids[i]] = name;
      }
   }
   else
   {
      mfem_error("Unexpected netcdf variable type");
   }
}


/// @brief Reads the coordinate data from the Genesis file.
static void ReadCubitNodeCoordinates(NetCDFReader & cubit_reader,
                                     double *coordx,
                                     double *coordy,
                                     double *coordz)
{
   cubit_reader.ReadVariable("coordx", coordx);
   cubit_reader.ReadVariable("coordy", coordy);

   if (coordz)
   {
      cubit_reader.ReadVariable("coordz", coordz);
   }
}


/// @brief Reads the number of elements in each block.
static void ReadCubitNumElementsInBlock(NetCDFReader & cubit_reader,
                                        const vector<int> & block_ids,
                                        map<int, size_t> &num_elements_for_block_id)
{
   num_elements_for_block_id.clear();

   // NB: need to add 1 for '\0' terminating character.
   const int buffer_size = NC_MAX_NAME + 1;
   char string_buffer[buffer_size];

   int iblock = 1;
   for (const auto block_id : block_ids)
   {
      // Write variable name to buffer.
      snprintf(string_buffer, buffer_size, "num_el_in_blk%d", iblock++);

      size_t num_elements_for_block = 0;
      cubit_reader.ReadDimension(string_buffer, &num_elements_for_block);

      num_elements_for_block_id[block_id] = num_elements_for_block;
   }
}

/// @brief Builds the mappings:
/// (blockID --> (elements in block)); (elementID --> blockID)
static void BuildElementIDsForBlockID(
   const vector<int> & block_ids,
   const map<int, size_t> & num_elements_for_block_id,
   map<int, vector<int>> & element_ids_for_block_id,
   map<int, int> & block_id_for_element_id)
{
   element_ids_for_block_id.clear();
   block_id_for_element_id.clear();

   // From the Exodus II specifications, the element ID is numbered contiguously starting
   // from 1 across the element blocks.
   int element_id = 1;
   for (int block_id : block_ids)
   {
      const int num_elements_for_block = num_elements_for_block_id.at(block_id);

      vector<int> element_ids(num_elements_for_block);

      for (int i = 0; i < num_elements_for_block; i++, element_id++)
      {
         element_ids[i] = element_id;
         block_id_for_element_id[element_id] = block_id;
      }

      element_ids_for_block_id[block_id] = std::move(element_ids);
   }
}

/// @brief Reads the element types for each block.
static void ReadCubitBlocks(NetCDFReader & cubit_reader,
                            const vector<int> block_ids,
                            CubitBlock & cubit_blocks)
{
   const int buffer_size = NC_MAX_NAME + 1;
   char string_buffer[buffer_size];

   size_t num_nodes_per_element;

   int iblock = 1;
   for (int block_id : block_ids)
   {
      // Write variable name to buffer.
      snprintf(string_buffer, buffer_size, "num_nod_per_el%d", iblock++);

      cubit_reader.ReadDimension(string_buffer, &num_nodes_per_element);

      // Determine the element type:
      CubitElementType element_type = CubitElement::GetElementType(
                                         num_nodes_per_element, cubit_blocks.GetDimension());
      cubit_blocks.AddBlockElement(block_id, element_type);
   }
}


/// @brief Extracts core dimension information from Genesis file.
static void ReadCubitDimensions(NetCDFReader & cubit_reader,
                                size_t &num_dim,
                                size_t &num_nodes,
                                size_t &num_elem,
                                size_t &num_el_blk,
                                size_t &num_side_sets)
{
   cubit_reader.ReadDimension("num_dim", &num_dim);
   cubit_reader.ReadDimension("num_nodes", &num_nodes);
   cubit_reader.ReadDimension("num_elem", &num_elem);
   cubit_reader.ReadDimension("num_el_blk", &num_el_blk);

   // Optional: if not present, num_side_sets = 0.
   if (cubit_reader.HasDimension("num_side_sets"))
   {
      cubit_reader.ReadDimension("num_side_sets", &num_side_sets);
   }
   else
   {
      num_side_sets = 0;
   }
}

/// @brief Extracts the element ids corresponding to elements that lie on each boundary;
/// also extracts the side ids of those elements which lie on the boundary.
static void ReadCubitBoundaries(NetCDFReader & cubit_reader,
                                const vector<int> & boundary_ids,
                                map<int, vector<int>> & element_ids_for_boundary_id,
                                map<int, vector<int>> & side_ids_for_boundary_id)
{
   const int buffer_size = NC_MAX_NAME + 1;
   char string_buffer[buffer_size];

   int ibdr = 1;
   for (int boundary_id : boundary_ids)
   {
      // 1. Extract number of elements/sides for boundary.
      size_t num_sides = 0;

      snprintf(string_buffer, buffer_size, "num_side_ss%d", ibdr);
      cubit_reader.ReadDimension(string_buffer, &num_sides);

      // 2. Extract elements and sides on each boundary (1-indexed!)
      vector<int> boundary_element_ids(num_sides); // (element, face) pairs.
      vector<int> boundary_side_ids(num_sides);

      //
      snprintf(string_buffer, buffer_size, "elem_ss%d", ibdr);
      cubit_reader.ReadVariable(string_buffer, boundary_element_ids.data());

      //
      snprintf(string_buffer, buffer_size,"side_ss%d", ibdr++);
      cubit_reader.ReadVariable(string_buffer, boundary_side_ids.data());

      // 3. Add to maps.
      element_ids_for_boundary_id[boundary_id] = std::move(boundary_element_ids);
      side_ids_for_boundary_id[boundary_id] = std::move(boundary_side_ids);
   }
}

/// @brief Reads the block ids from the Genesis file.
static void BuildCubitBlockIDs(NetCDFReader & cubit_reader,
                               const int num_element_blocks,
                               vector<int> & block_ids)
{
   block_ids.resize(num_element_blocks);
   cubit_reader.ReadVariable("eb_prop1", block_ids.data());
}

/// @brief Reads the boundary ids from the Genesis file.
static void ReadCubitBoundaryIDs(NetCDFReader & cubit_reader,
                                 const int num_boundaries, vector<int> & boundary_ids)
{
   boundary_ids.clear();

   if (num_boundaries < 1) { return; }

   boundary_ids.resize(num_boundaries);
   cubit_reader.ReadVariable("ss_prop1", boundary_ids.data());
}

/// @brief Reads the node ids for each element from the Genesis file.
static void ReadCubitElementBlocks(NetCDFReader & cubit_reader,
                                   const CubitBlock & cubit_blocks,
                                   const vector<int> & block_ids,
                                   const map<int, vector<int>> & element_ids_for_block_id,
                                   map<int, vector<int>> &node_ids_for_element_id)
{
   const int buffer_size = NC_MAX_NAME + 1;
   char string_buffer[buffer_size];

   int iblock = 1;
   for (const int block_id : block_ids)
   {
      const CubitElement & block_element = cubit_blocks.GetBlockElement(block_id);

      const vector<int> & block_element_ids = element_ids_for_block_id.at(block_id);

      const size_t num_nodes_for_block = block_element_ids.size() *
                                         block_element.GetNumNodes();

      vector<int> node_ids_for_block(num_nodes_for_block);

      // Write variable name to buffer.
      snprintf(string_buffer, buffer_size, "connect%d", iblock++);

      cubit_reader.ReadVariable(string_buffer, node_ids_for_block.data());

      // Now map from the element id to the nodes:
      int ielement = 0;
      for (int element_id : block_element_ids)
      {
         vector<int> element_node_ids(block_element.GetNumNodes());

         for (int i = 0; i < (int)block_element.GetNumNodes(); i++)
         {
            element_node_ids[i] = node_ids_for_block[ielement * block_element.GetNumNodes()
                                                     + i];
         }

         ielement++;

         node_ids_for_element_id[element_id] = std::move(element_node_ids);
      }
   }
}

/// @brief Builds a mapping from the boundary ID to the face vertices of each element that lie on the boundary.
static void BuildBoundaryNodeIDs(const vector<int> & boundary_ids,
                                 const CubitBlock & blocks,
                                 const map<int, vector<int>> & node_ids_for_element_id,
                                 const map<int, vector<int>> & element_ids_for_boundary_id,
                                 const map<int, vector<int>> & side_ids_for_boundary_id,
                                 const map<int, int> & block_id_for_element_id,
                                 map<int, vector<vector<int>>> & node_ids_for_boundary_id)
{
   for (int boundary_id : boundary_ids)
   {
      // Get element IDs of element on boundary (and their sides that are on boundary).
      auto & boundary_element_ids = element_ids_for_boundary_id.at(
                                       boundary_id);
      auto & boundary_element_sides = side_ids_for_boundary_id.at(
                                         boundary_id);

      // Create vector to store the node ids of all boundary nodes.
      vector<vector<int>> boundary_node_ids(
                          boundary_element_ids.size());

      // Iterate over elements on boundary.
      for (int jelement = 0; jelement < (int)boundary_element_ids.size(); jelement++)
      {
         // Get element ID and the boundary side.
         const int boundary_element_global_id = boundary_element_ids[jelement];
         const int boundary_side = boundary_element_sides[jelement];

         // Get the element information:
         const int block_id = block_id_for_element_id.at(boundary_element_global_id);
         const CubitElement & block_element = blocks.GetBlockElement(block_id);

         const int num_face_vertices = block_element.GetNumFaceVertices(boundary_side);
         vector<int> nodes_of_element_on_side(num_face_vertices);

         // Get all of the element's nodes on boundary side of element.
         const vector<int> & element_node_ids =
            node_ids_for_element_id.at(boundary_element_global_id);

         // Iterate over the element's face nodes on the matching side.
         // NB: only adding vertices on face (ignore higher-order).
         for (int knode = 0; knode < num_face_vertices; knode++)
         {
            int inode;

            switch (block_element.GetElementType())
            {
               case ELEMENT_TRI3:
               case ELEMENT_TRI6:
                  inode = cubit_side_map_tri3[boundary_side - 1][knode];
                  break;
               case ELEMENT_QUAD4:
               case ELEMENT_QUAD9:
                  inode = cubit_side_map_quad4[boundary_side - 1][knode];
                  break;
               case ELEMENT_TET4:
               case ELEMENT_TET10:
                  inode = cubit_side_map_tet4[boundary_side - 1][knode];
                  break;
               case ELEMENT_HEX8:
               case ELEMENT_HEX27:
                  inode = cubit_side_map_hex8[boundary_side - 1][knode];
                  break;
               case ELEMENT_WEDGE6:
               case ELEMENT_WEDGE18:
                  inode = cubit_side_map_wedge6[boundary_side - 1][knode];
                  break;
               case ELEMENT_PYRAMID5:
               case ELEMENT_PYRAMID14:
                  inode = cubit_side_map_pyramid5[boundary_side - 1][knode];
                  break;
               default:
                  MFEM_ABORT("Unsupported element type encountered.\n");
                  break;
            }

            nodes_of_element_on_side[knode] = element_node_ids[inode - 1];
         }

         boundary_node_ids[jelement] = std::move(nodes_of_element_on_side);
      }

      // Add to the map.
      node_ids_for_boundary_id[boundary_id] = std::move(boundary_node_ids);
   }
}

/// @brief Generates a vector of unique vertex ID.
static void BuildUniqueVertexIDs(const vector<int> & unique_block_ids,
                                 const CubitBlock & blocks,
                                 const map<int, vector<int>> & element_ids_for_block_id,
                                 const map<int, vector<int>> & node_ids_for_element_id,
                                 vector<int> & unique_vertex_ids)
{
   // Iterate through all vertices and add their global IDs to the unique_vertex_ids vector.
   for (int block_id : unique_block_ids)
   {
      auto & element_ids = element_ids_for_block_id.at(block_id);

      auto & block_element = blocks.GetBlockElement(block_id);

      for (int element_id : element_ids)
      {
         auto & node_ids = node_ids_for_element_id.at(element_id);

         for (size_t knode = 0; knode < block_element.GetNumVertices(); knode++)
         {
            unique_vertex_ids.push_back(node_ids[knode]);
         }
      }
   }

   // Sort unique_vertex_ids in ascending order and remove duplicate node IDs.
   std::sort(unique_vertex_ids.begin(), unique_vertex_ids.end());

   auto new_end = std::unique(unique_vertex_ids.begin(), unique_vertex_ids.end());

   unique_vertex_ids.resize(std::distance(unique_vertex_ids.begin(), new_end));
}

/// @brief unique_vertex_ids contains a 1-based sorted list of vertex IDs used by the mesh. We
/// now create a map by running over the vertex IDs and remapping to a contiguous
/// 1-based array of integers.
static void BuildCubitToMFEMVertexMap(const vector<int> & unique_vertex_ids,
                                      map<int, int> & cubit_to_mfem_vertex_map)
{
   cubit_to_mfem_vertex_map.clear();

   int ivertex = 1;
   for (int vertex_id : unique_vertex_ids)
   {
      cubit_to_mfem_vertex_map[vertex_id] = ivertex++;
   }
}


/// @brief The final step in constructing the mesh from a Genesis file. This is
/// only called if the mesh order == 2 (determined internally from the cubit
/// element type).
static void FinalizeCubitSecondOrderMesh(Mesh &mesh,
                                         const vector<int> & unique_block_ids,
                                         const CubitBlock & blocks,
                                         const map<int, vector<int>> & element_ids_for_block_id,
                                         const map<int, vector<int>> & node_ids_for_element_id,
                                         const double *coordx,
                                         const double *coordy,
                                         const double *coordz)
{
   mesh.FinalizeTopology();

   // Define quadratic FE space.
   const int Dim = mesh.Dimension();
   FiniteElementCollection *fec = new H1_FECollection(2, Dim);
   FiniteElementSpace *fes = new FiniteElementSpace(&mesh, fec, Dim,
                                                    Ordering::byVDIM);
   GridFunction *Nodes = new GridFunction(fes);
   Nodes->MakeOwner(fec); // Nodes will destroy 'fec' and 'fes'
   mesh.SetNodalGridFunction(Nodes, true);

   for (int block_id : unique_block_ids)
   {
      const CubitElement & block_element = blocks.GetBlockElement(block_id);

      int *mfem_to_genesis_map = NULL;

      switch (block_element.GetElementType())
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
         case ELEMENT_WEDGE18:
            mfem_to_genesis_map = (int *) mfem_to_genesis_wedge18;
            break;
         case ELEMENT_PYRAMID14:
            mfem_to_genesis_map = (int *) mfem_to_genesis_pyramid14;
            break;
         default:
            MFEM_ABORT("Something went wrong. Linear elements detected when order is 2.");
      }

      auto & element_ids = element_ids_for_block_id.at(block_id);

      for (int element_id : element_ids)
      {
         // NB: 1-index (Exodus) --> 0-index (MFEM).
         Array<int> dofs;
         fes->GetElementDofs(element_id - 1, dofs);

         Array<int> vdofs = dofs;   // Deep copy.
         fes->DofsToVDofs(vdofs);

         const vector<int> & element_node_ids = node_ids_for_element_id.at(element_id);

         for (int jnode = 0; jnode < dofs.Size(); jnode++)
         {
            const int node_index = element_node_ids[mfem_to_genesis_map[jnode] - 1] - 1;

            (*Nodes)(vdofs[jnode])     = coordx[node_index];
            (*Nodes)(vdofs[jnode] + 1) = coordy[node_index];

            if (Dim == 3)
            {
               (*Nodes)(vdofs[jnode] + 2) = coordz[node_index];
            }
         }
      }
   }
}

}  // end of namespace cubit.

/// @brief Set the coordinates of the Cubit vertices.
void Mesh::BuildCubitVertices(const vector<int> & unique_vertex_ids,
                              const vector<double> & coordx,
                              const vector<double> & coordy,
                              const vector<double> & coordz)
{
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
}

/// @brief Create Cubit elements.
void Mesh::BuildCubitElements(const int num_elements,
                              const cubit::CubitBlock * blocks,
                              const vector<int> & block_ids,
                              const map<int, vector<int>> & element_ids_for_block_id,
                              const map<int, vector<int>> & node_ids_for_element_id,
                              const map<int, int> & cubit_to_mfem_vertex_map)
{
   using namespace cubit;

   NumOfElements = num_elements;
   elements.SetSize(num_elements);

   int element_counter = 0;

   // Iterate over blocks.
   for (int block_id : block_ids)
   {
      const CubitElement & block_element = blocks->GetBlockElement(block_id);

      vector<int> renumbered_vertex_ids(block_element.GetNumVertices());

      const vector<int> &block_element_ids = element_ids_for_block_id.at(block_id);

      // Iterate over elements in block.
      for (int element_id : block_element_ids)
      {
         const vector<int> & element_node_ids = node_ids_for_element_id.at(element_id);

         // Iterate over linear (vertex) nodes in block.
         for (size_t knode = 0; knode < block_element.GetNumVertices(); knode++)
         {
            const int node_id = element_node_ids[knode];

            // Renumber using the mapping.
            renumbered_vertex_ids[knode] = cubit_to_mfem_vertex_map.at(node_id) - 1;
         }

         // Create element.
         elements[element_counter++] = block_element.BuildElement(*this,
                                                                  renumbered_vertex_ids.data(),
                                                                  block_id);
      }
   }
}

/// @brief Build the Cubit boundaries.
void Mesh::BuildCubitBoundaries(
   const cubit::CubitBlock * blocks,
   const vector<int> & boundary_ids,
   const map<int, vector<int>> & element_ids_for_boundary_id,
   const map<int, vector<vector<int>>> & node_ids_for_boundary_id,
   const map<int, vector<int>> & side_ids_for_boundary_id,
   const map<int, int> & block_id_for_element_id,
   const map<int, int> & cubit_to_mfem_vertex_map)
{
   using namespace cubit;

   NumOfBdrElements = 0;
   for (int boundary_id : boundary_ids)
   {
      NumOfBdrElements += element_ids_for_boundary_id.at(boundary_id).size();
   }

   boundary.SetSize(NumOfBdrElements);

   array<int, 8> renumbered_vertex_ids;   // Set to max number of vertices (Hex27).

   // Iterate over boundaries.
   int boundary_counter = 0;
   for (int boundary_id : boundary_ids)
   {
      const vector<int> &elements_on_boundary = element_ids_for_boundary_id.at(
                                                   boundary_id);

      const vector<vector<int>> &nodes_on_boundary = node_ids_for_boundary_id.at(
                                                        boundary_id);

      int jelement = 0;
      for (int side_id : side_ids_for_boundary_id.at(boundary_id))
      {
         // Determine the block the element originates from and the element type.
         const int element_id = elements_on_boundary.at(jelement);
         const int element_block = block_id_for_element_id.at(element_id);
         const CubitElement & block_element = blocks->GetBlockElement(element_block);

         const vector<int> & element_nodes_on_side = nodes_on_boundary.at(jelement);

         // Iterate over element's face vertices.
         for (size_t knode = 0; knode < element_nodes_on_side.size(); knode++)
         {
            const int node_id = element_nodes_on_side[knode];

            // Renumber using the mapping.
            renumbered_vertex_ids[knode] = cubit_to_mfem_vertex_map.at(node_id) - 1;
         }

         // Create boundary element.
         boundary[boundary_counter++] = block_element.BuildBoundaryElement(*this,
                                                                           side_id,
                                                                           renumbered_vertex_ids.data(),
                                                                           boundary_id);

         jelement++;
      }
   }
}

void Mesh::ReadCubit(const std::string &filename, int &curved, int &read_gf)
{
   using namespace cubit;

   read_gf  = 0;
   curved   = 0; // Set to 1 if mesh is curved.

   //
   // Open the file.
   //
   NetCDFReader cubit_reader(filename);

   //
   // Read important dimensions from file.
   //
   size_t num_dimensions, num_nodes, num_elements, num_element_blocks,
          num_boundaries;

   ReadCubitDimensions(cubit_reader, num_dimensions, num_nodes, num_elements,
                       num_element_blocks, num_boundaries);

   Dim = num_dimensions;

   //
   // Read the blocks.
   //
   vector<int> block_ids;
   BuildCubitBlockIDs(cubit_reader, num_element_blocks, block_ids);
   unordered_map<int, string> blk_ids_to_names;
   cubit_reader.BuildIDToNameMap(block_ids, blk_ids_to_names, "eb_names");
   for (const auto & pr : blk_ids_to_names)
   {
      const auto blk_id = pr.first;
      const auto & blk_name = pr.second;
      if (!blk_name.empty())
      {
         if (!attribute_sets.AttributeSetExists(blk_name))
         {
            attribute_sets.CreateAttributeSet(blk_name);
         }
         attribute_sets.AddToAttributeSet(blk_name, blk_id);
      }
   }

   map<int, size_t> num_elements_for_block_id;
   ReadCubitNumElementsInBlock(cubit_reader, block_ids,
                               num_elements_for_block_id);

   map<int, vector<int>> element_ids_for_block_id;
   map<int, int> block_id_for_element_id;
   BuildElementIDsForBlockID(
      block_ids, num_elements_for_block_id, element_ids_for_block_id,
      block_id_for_element_id);

   //
   // Read number of nodes for each element.
   CubitBlock blocks(num_dimensions);
   ReadCubitBlocks(cubit_reader, block_ids, blocks);

   // Read the elements that make-up each block.
   map<int, vector<int>> node_ids_for_element_id;
   ReadCubitElementBlocks(cubit_reader,
                          blocks,
                          block_ids,
                          element_ids_for_block_id,
                          node_ids_for_element_id);

   //
   // Read the boundary ids.
   //
   vector<int> boundary_ids;
   ReadCubitBoundaryIDs(cubit_reader, num_boundaries, boundary_ids);
   unordered_map<int, string> bnd_ids_to_names;
   cubit_reader.BuildIDToNameMap(boundary_ids, bnd_ids_to_names, "ss_names");
   for (const auto & pr : bnd_ids_to_names)
   {
      const auto bnd_id = pr.first;
      const auto & bnd_name = pr.second;
      if (!bnd_name.empty())
      {
         if (!bdr_attribute_sets.AttributeSetExists(bnd_name))
         {
            bdr_attribute_sets.CreateAttributeSet(bnd_name);
         }
         bdr_attribute_sets.AddToAttributeSet(bnd_name, bnd_id);
      }
   }


   //
   // Read the (element, corresponding side) on each of the boundaries.
   //
   map<int, vector<int>> element_ids_for_boundary_id;
   map<int, vector<int>> side_ids_for_boundary_id;

   ReadCubitBoundaries(cubit_reader, boundary_ids,
                       element_ids_for_boundary_id, side_ids_for_boundary_id);

   map<int, vector<vector<int>>> node_ids_for_boundary_id;

   BuildBoundaryNodeIDs(boundary_ids, blocks, node_ids_for_element_id,
                        element_ids_for_boundary_id, side_ids_for_boundary_id,
                        block_id_for_element_id,
                        node_ids_for_boundary_id);

   //
   // Read the xyz coordinates for each node.
   //
   vector<double> coordx(num_nodes);
   vector<double> coordy(num_nodes);
   vector<double> coordz(num_dimensions == 3 ? num_nodes : 0);

   ReadCubitNodeCoordinates(cubit_reader, coordx.data(), coordy.data(),
                            coordz.data());

   //
   // We need another node ID mapping since MFEM needs contiguous vertex ids.
   //
   vector<int> unique_vertex_ids;
   BuildUniqueVertexIDs(block_ids, blocks, element_ids_for_block_id,
                        node_ids_for_element_id, unique_vertex_ids);

   //
   // unique_vertex_ids now contains a 1-based sorted list of node IDs for each
   // node used by the mesh. We now create a map by running over the node IDs
   // and remapping to contiguous 1-based integers.
   // ie. [1, 4, 5, 8, 9] --> [1, 2, 3, 4, 5].
   //
   map<int, int> cubit_to_mfem_vertex_map;
   BuildCubitToMFEMVertexMap(unique_vertex_ids, cubit_to_mfem_vertex_map);

   //
   // Load up the vertices.
   //
   BuildCubitVertices(unique_vertex_ids, coordx, coordy, coordz);

   //
   // Now load the elements.
   //
   BuildCubitElements(num_elements, &blocks, block_ids,
                      element_ids_for_block_id,
                      node_ids_for_element_id, cubit_to_mfem_vertex_map);

   //
   // Load up the boundary elements.
   //
   BuildCubitBoundaries(&blocks, boundary_ids,
                        element_ids_for_boundary_id, node_ids_for_boundary_id, side_ids_for_boundary_id,
                        block_id_for_element_id,
                        cubit_to_mfem_vertex_map);

   //
   // Additional setup for second order.
   //
   if (blocks.GetOrder() == 2)
   {
      curved = 1;

      FinalizeCubitSecondOrderMesh(*this,
                                   block_ids,
                                   blocks,
                                   element_ids_for_block_id,
                                   node_ids_for_element_id,
                                   coordx.data(),
                                   coordy.data(),
                                   coordz.data());
   }
}

#endif // #ifdef MFEM_USE_NETCDF

} // namespace mfem
