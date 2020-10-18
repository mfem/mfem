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

#include "mesh_headers.hpp"
#include "../fem/fem.hpp"
#include "../general/text.hpp"
#include "gmsh.hpp"

#include <iostream>
#include <cstdio>

#ifdef MFEM_USE_NETCDF
#include "netcdf.h"
#endif

using namespace std;

namespace mfem
{

bool Mesh::remove_unused_vertices = true;

void Mesh::ReadMFEMMesh(std::istream &input, bool mfem_v11, int &curved)
{
   // Read MFEM mesh v1.0 format
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

   skip_comment_lines(input, '#');
   input >> ident; // 'boundary'

   MFEM_VERIFY(ident == "boundary", "invalid mesh file");
   input >> NumOfBdrElements;
   boundary.SetSize(NumOfBdrElements);
   for (int j = 0; j < NumOfBdrElements; j++)
   {
      boundary[j] = ReadElement(input);
   }

   skip_comment_lines(input, '#');
   input >> ident;

   if (mfem_v11 && ident == "vertex_parents")
   {
      ncmesh = new NCMesh(this, &input);
      // NOTE: the constructor above will call LoadVertexParents

      skip_comment_lines(input, '#');
      input >> ident;

      if (ident == "coarse_elements")
      {
         ncmesh->LoadCoarseElements(input);

         skip_comment_lines(input, '#');
         input >> ident;
      }
   }

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

      // initialize vertex positions in NCMesh
      if (ncmesh) { ncmesh->SetVertexPositions(vertices); }
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
      double varf;

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
      double varf;
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

void Mesh::ReadVTKMesh(std::istream &input, int &curved, int &read_gf,
                       bool &finalize_topo)
{
   // VTK resources:
   //   * https://www.vtk.org/doc/nightly/html/vtkCellType_8h_source.html
   //   * https://www.vtk.org/doc/nightly/html/classvtkCell.html
   //   * https://lorensen.github.io/VTKExamples/site/VTKFileFormats
   //   * https://www.kitware.com/products/books/VTKUsersGuide.pdf

   int i, j, n, attr;

   string buff;
   getline(input, buff); // comment line
   getline(input, buff);
   filter_dos(buff);
   if (buff != "ASCII")
   {
      MFEM_ABORT("VTK mesh is not in ASCII format!");
      return;
   }
   getline(input, buff);
   filter_dos(buff);
   if (buff != "DATASET UNSTRUCTURED_GRID")
   {
      MFEM_ABORT("VTK mesh is not UNSTRUCTURED_GRID!");
      return;
   }

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
   int np = 0;
   Vector points;
   {
      input >> np >> ws;
      points.SetSize(3*np);
      getline(input, buff); // "double"
      for (i = 0; i < points.Size(); i++)
      {
         input >> points(i);
      }
   }

   // Read the cells
   NumOfElements = n = 0;
   Array<int> cells_data;
   input >> ws >> buff;
   if (buff == "CELLS")
   {
      input >> NumOfElements >> n >> ws;
      cells_data.SetSize(n);
      for (i = 0; i < n; i++)
      {
         input >> cells_data[i];
      }
   }

   // Read the cell types
   Dim = -1;
   int order = -1;
   input >> ws >> buff;
   if (buff == "CELL_TYPES")
   {
      input >> NumOfElements;
      elements.SetSize(NumOfElements);
      for (j = i = 0; i < NumOfElements; i++)
      {
         int ct, elem_dim, elem_order = 1;
         input >> ct;
         switch (ct)
         {
            case 5:   // triangle
               elem_dim = 2;
               elements[i] = new Triangle(&cells_data[j+1]);
               break;
            case 9:   // quadrilateral
               elem_dim = 2;
               elements[i] = new Quadrilateral(&cells_data[j+1]);
               break;
            case 10:  // tetrahedron
               elem_dim = 3;
#ifdef MFEM_USE_MEMALLOC
               elements[i] = TetMemory.Alloc();
               elements[i]->SetVertices(&cells_data[j+1]);
#else
               elements[i] = new Tetrahedron(&cells_data[j+1]);
#endif
               break;
            case 12:  // hexahedron
               elem_dim = 3;
               elements[i] = new Hexahedron(&cells_data[j+1]);
               break;
            case 13:  // wedge
               elem_dim = 3;
               // switch between vtk vertex ordering and mfem vertex ordering:
               // swap vertices (1,2) and (4,5)
               elements[i] =
                  new Wedge(cells_data[j+1], cells_data[j+3], cells_data[j+2],
                            cells_data[j+4], cells_data[j+6], cells_data[j+5]);
               break;

            case 22:  // quadratic triangle
               elem_dim = 2;
               elem_order = 2;
               elements[i] = new Triangle(&cells_data[j+1]);
               break;
            case 28:  // biquadratic quadrilateral
               elem_dim = 2;
               elem_order = 2;
               elements[i] = new Quadrilateral(&cells_data[j+1]);
               break;
            case 24:  // quadratic tetrahedron
               elem_dim = 3;
               elem_order = 2;
#ifdef MFEM_USE_MEMALLOC
               elements[i] = TetMemory.Alloc();
               elements[i]->SetVertices(&cells_data[j+1]);
#else
               elements[i] = new Tetrahedron(&cells_data[j+1]);
#endif
               break;
            case 32: // biquadratic-quadratic wedge
               elem_dim = 3;
               elem_order = 2;
               // switch between vtk vertex ordering and mfem vertex ordering:
               // swap vertices (1,2) and (4,5)
               elements[i] =
                  new Wedge(cells_data[j+1], cells_data[j+3], cells_data[j+2],
                            cells_data[j+4], cells_data[j+6], cells_data[j+5]);
               break;
            case 29:  // triquadratic hexahedron
               elem_dim = 3;
               elem_order = 2;
               elements[i] = new Hexahedron(&cells_data[j+1]);
               break;
            default:
               MFEM_ABORT("VTK mesh : cell type " << ct << " is not supported!");
               return;
         }
         MFEM_VERIFY(Dim == -1 || Dim == elem_dim,
                     "elements with different dimensions are not supported");
         MFEM_VERIFY(order == -1 || order == elem_order,
                     "elements with different orders are not supported");
         Dim = elem_dim;
         order = elem_order;
         j += cells_data[j] + 1;
      }
   }

   // Read attributes
   streampos sp = input.tellg();
   input >> ws >> buff;
   if (buff == "CELL_DATA")
   {
      input >> n >> ws;
      getline(input, buff);
      filter_dos(buff);
      // "SCALARS material dataType numComp"
      if (!strncmp(buff.c_str(), "SCALARS material", 16))
      {
         getline(input, buff); // "LOOKUP_TABLE default"
         for (i = 0; i < NumOfElements; i++)
         {
            input >> attr;
            elements[i]->SetAttribute(attr);
         }
      }
      else
      {
         input.seekg(sp);
      }
   }
   else
   {
      input.seekg(sp);
   }

   if (order == 1)
   {
      cells_data.DeleteAll();
      NumOfVertices = np;
      vertices.SetSize(np);
      for (i = 0; i < np; i++)
      {
         vertices[i](0) = points(3*i+0);
         vertices[i](1) = points(3*i+1);
         vertices[i](2) = points(3*i+2);
      }
      points.Destroy();

      // No boundary is defined in a VTK mesh
      NumOfBdrElements = 0;
   }
   else if (order == 2)
   {
      curved = 1;

      // generate new enumeration for the vertices
      Array<int> pts_dof(np);
      pts_dof = -1;
      for (n = i = 0; i < NumOfElements; i++)
      {
         int *v = elements[i]->GetVertices();
         int nv = elements[i]->GetNVertices();
         for (j = 0; j < nv; j++)
            if (pts_dof[v[j]] == -1)
            {
               pts_dof[v[j]] = n++;
            }
      }
      // keep the original ordering of the vertices
      for (n = i = 0; i < np; i++)
         if (pts_dof[i] != -1)
         {
            pts_dof[i] = n++;
         }
      // update the element vertices
      for (i = 0; i < NumOfElements; i++)
      {
         int *v = elements[i]->GetVertices();
         int nv = elements[i]->GetNVertices();
         for (j = 0; j < nv; j++)
         {
            v[j] = pts_dof[v[j]];
         }
      }
      // Define the 'vertices' from the 'points' through the 'pts_dof' map
      NumOfVertices = n;
      vertices.SetSize(n);
      for (i = 0; i < np; i++)
      {
         if ((j = pts_dof[i]) != -1)
         {
            vertices[j](0) = points(3*i+0);
            vertices[j](1) = points(3*i+1);
            vertices[j](2) = points(3*i+2);
         }
      }

      // No boundary is defined in a VTK mesh
      NumOfBdrElements = 0;

      // Generate faces and edges so that we can define quadratic
      // FE space on the mesh
      FinalizeTopology();
      finalize_topo = false;

      // Define quadratic FE space
      FiniteElementCollection *fec = new QuadraticFECollection;
      FiniteElementSpace *fes = new FiniteElementSpace(this, fec, Dim);
      Nodes = new GridFunction(fes);
      Nodes->MakeOwner(fec); // Nodes will destroy 'fec' and 'fes'
      own_nodes = 1;

      // Map vtk points to edge/face/element dofs
      Array<int> dofs;
      for (n = i = 0; i < NumOfElements; i++)
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
            default:
               vtk_mfem = NULL; // suppress a warning
               break;
         }

         for (n++, j = 0; j < dofs.Size(); j++, n++)
         {
            if (pts_dof[cells_data[n]] == -1)
            {
               pts_dof[cells_data[n]] = dofs[vtk_mfem[j]];
            }
            else
            {
               if (pts_dof[cells_data[n]] != dofs[vtk_mfem[j]])
               {
                  MFEM_ABORT("VTK mesh : inconsistent quadratic mesh!");
               }
            }
         }
      }

      // Define the 'Nodes' from the 'points' through the 'pts_dof' map
      for (i = 0; i < np; i++)
      {
         dofs.SetSize(1);
         if ((dofs[0] = pts_dof[i]) != -1)
         {
            fes->DofsToVDofs(dofs);
            for (j = 0; j < dofs.Size(); j++)
            {
               (*Nodes)(dofs[j]) = points(3*i+j);
            }
         }
      }

      read_gf = 0;
   }
}

void Mesh::ReadNURBSMesh(std::istream &input, int &curved, int &read_gf)
{
   NURBSext = new NURBSExtension(input);

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
      int vd = Nodes->VectorDim();
      for (int i = 0; i < vd; i++)
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
   double sx = -1.0;
   double sy = -1.0;
   double sz = -1.0;
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
            type == Element::HEXAHEDRON)
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
   double version;
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
   double bb_tol = 1e-14;
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
         double bb_size = std::max(bb_max[0] - bb_min[0],
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
                    the vertices, 8 with the edges, 4 wth the face) */
            25,  /* 25-node fourth order quadrilateral (4 nodes associated with
                    the vertices, 12 with the edges, 9 wth the face) */
            36,  /* 36-node fifth order quadrilateral (4 nodes associated with
                    the vertices, 16 with the edges, 16 wth the face) */
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
                    the vertices, 20 with the edges, 25 wth the face) */
            64,  /* 64-node seventh order quadrilateral (4 nodes associated with
                    the vertices, 24 with the edges, 36 wth the face) */
            81,  /* 81-node eighth order quadrilateral (4 nodes associated with
                    the vertices, 28 with the edges, 49 wth the face) */
            100, /* 100-node ninth order quadrilateral (4 nodes associated with
                    the vertices, 32 with the edges, 64 wth the face) */
            121, /* 121-node tenth order quadrilateral (4 nodes associated with
                    the vertices, 36 with the edges, 81 wth the face) */
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

                  // non-positive attributes are not allowed in MFEM
                  if (phys_domain <= 0)
                  {
                     MFEM_ABORT("Non-positive element attribute in Gmsh mesh!");
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
                     /*
                     // MFEM does not support pyramids yet
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
                     */
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

               // non-positive attributes are not allowed in MFEM
               if (phys_domain <= 0)
               {
                  MFEM_ABORT("Non-positive element attribute in Gmsh mesh!");
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
                  /*
                  // MFEM does not support pyramids yet
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
                  */
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
                  // case Element::PYRAMID:
                  //    ho_verts = ho_verts_3D[el];
                  //    el_order = ho_el_order_3D[el];
                  //    if (ho_pyr[el_order])
                  //    {
                  //      ho_pyr[el_order] = new int[ho_verts->Size()];
                  //      GmshHOPyramidMapping(el_order, ho_pyr[el_order]);
                  //    }
                  //    vm = ho_pyr[el_order];
                  //    break;
                  default: // Any other element type
                     MFEM_WARNING("Unsupported Gmsh element type.");
                     break;
               }
               int nv = (ho_verts) ? ho_verts->Size() : 0;

               for (int v = 0; v<nv; v++)
               {
                  double * c = GetVertex((*ho_verts)[vm[v]]);
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

         MFEM_CONTRACT_VAR(n_partitions);
         MFEM_CONTRACT_VAR(elem_domain);

      } // section '$Elements'
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
         int slave, master;
         input >> num_per_ent;
         getline(input, buff); // Read end-of-line
         for (int i = 0; i < num_per_ent; i++)
         {
            getline(input, buff); // Read and ignore entity dimension and tags
            getline(input, buff); // Read and ignore affine mapping
            // Read master/slave vertex pairs
            input >> num_nodes;
            for (int j=0; j<num_nodes; j++)
            {
               input >> slave >> master;
               v2v[slave - 1] = master - 1;
            }
            getline(input, buff); // Read end-of-line
         }

         // Convert nodes to discontinuous GridFunction (if they aren't already)
         if (mesh_order == 1)
         {
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
void Mesh::ReadCubit(const char *filename, int &curved, int &read_gf)
{
   read_gf = 0;

   // curved set to zero will change if mesh is indeed curved
   curved = 0;

   const int sideMapTri3[3][2] =
   {
      {1,2},
      {2,3},
      {3,1},
   };

   const int sideMapQuad4[4][2] =
   {
      {1,2},
      {2,3},
      {3,4},
      {4,1},
   };

   const int sideMapTri6[3][3] =
   {
      {1,2,4},
      {2,3,5},
      {3,1,6},
   };

   const int sideMapQuad9[4][3] =
   {
      {1,2,5},
      {2,3,6},
      {3,4,7},
      {4,1,8},
   };

   const int sideMapTet4[4][3] =
   {
      {1,2,4},
      {2,3,4},
      {1,4,3},
      {1,3,2}
   };

   const int sideMapTet10[4][6] =
   {
      {1,2,4,5,9,8},
      {2,3,4,6,10,9},
      {1,4,3,8,10,7},
      {1,3,2,7,6,5}
   };

   const int sideMapHex8[6][4] =
   {
      {1,2,6,5},
      {2,3,7,6},
      {4,3,7,8},
      {1,4,8,5},
      {1,4,3,2},
      {5,8,7,6}
   };

   const int sideMapHex27[6][9] =
   {
      {1,2,6,5,9,14,17,13,26},
      {2,3,7,6,10,15,18,14,25},
      {4,3,7,8,11,15,19,16,27},
      {1,4,8,5,12,16,20,13,24},
      {1,4,3,2,12,11,10,9,22},
      {5,8,7,6,20,19,18,17,23}
   };


   //                                  1,2,3,4,5,6,7,8,9,10
   const int mfemToGenesisTet10[10] = {1,2,3,4,5,7,8,6,9,10};

   //                                  1,2,3,4,5,6,7,8,9,10,11,
   const int mfemToGenesisHex27[27] = {1,2,3,4,5,6,7,8,9,10,11,
                                       // 12,13,14,15,16,17,18,19
                                       12,17,18,19,20,13,14,15,
                                       // 20,21,22,23,24,25,26,27
                                       16,22,26,25,27,24,23,21
                                      };

   const int mfemToGenesisTri6[6]   = {1,2,3,4,5,6};
   const int mfemToGenesisQuad9[9]  = {1,2,3,4,5,6,7,8,9};


   // error handling.
   int retval;

   // dummy string
   char str_dummy[256];

   char temp_str[256];
   int temp_id;

   // open the file.
   int ncid;
   if ((retval = nc_open(filename, NC_NOWRITE, &ncid)))
   {
      MFEM_ABORT("Fatal NetCDF error: " << nc_strerror(retval));
   }

   // read important dimensions

   int id;
   size_t num_dim=0, num_nodes=0, num_elem=0, num_el_blk=0, num_side_sets=0;

   if ((retval = nc_inq_dimid(ncid, "num_dim", &id)) ||
       (retval = nc_inq_dim(ncid, id, str_dummy, &num_dim)) ||

       (retval = nc_inq_dimid(ncid, "num_nodes", &id)) ||
       (retval = nc_inq_dim(ncid, id, str_dummy, &num_nodes)) ||

       (retval = nc_inq_dimid(ncid, "num_elem", &id)) ||
       (retval = nc_inq_dim(ncid, id, str_dummy, &num_elem)) ||

       (retval = nc_inq_dimid(ncid, "num_el_blk", &id)) ||
       (retval = nc_inq_dim(ncid, id, str_dummy, &num_el_blk)))
   {
      MFEM_ABORT("Fatal NetCDF error: " << nc_strerror(retval));
   }
   if ((retval = nc_inq_dimid(ncid, "num_side_sets", &id)) ||
       (retval = nc_inq_dim(ncid, id, str_dummy, &num_side_sets)))
   {
      num_side_sets = 0;
   }

   Dim = num_dim;

   // create arrays for element blocks
   size_t *num_el_in_blk   = new size_t[num_el_blk];
   size_t num_node_per_el;

   int previous_num_node_per_el = 0;
   for (int i = 0; i < (int) num_el_blk; i++)
   {
      sprintf(temp_str, "num_el_in_blk%d", i+1);
      if ((retval = nc_inq_dimid(ncid, temp_str, &temp_id)) ||
          (retval = nc_inq_dim(ncid, temp_id, str_dummy, &num_el_in_blk[i])))
      {
         MFEM_ABORT("Fatal NetCDF error: " << nc_strerror(retval));
      }

      sprintf(temp_str, "num_nod_per_el%d", i+1);
      if ((retval = nc_inq_dimid(ncid, temp_str, &temp_id)) ||
          (retval = nc_inq_dim(ncid, temp_id, str_dummy, &num_node_per_el)))
      {
         MFEM_ABORT("Fatal NetCDF error: " << nc_strerror(retval));
      }

      // check for different element types in each block
      // which is not currently supported
      if (i != 0)
      {
         if ((int) num_node_per_el != previous_num_node_per_el)
         {
            MFEM_ABORT("Element blocks of different element types not supported");
         }
      }
      previous_num_node_per_el = num_node_per_el;
   }

   // Determine CUBIT element and face type
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

   CubitElementType cubit_element_type = ELEMENT_TRI3; // suppress a warning
   CubitFaceType cubit_face_type = FACE_EDGE2; // suppress a warning
   int num_element_linear_nodes = 0; // initialize to suppress a warning

   if (num_dim == 2)
   {
      switch (num_node_per_el)
      {
         case (3) :
         {
            cubit_element_type = ELEMENT_TRI3;
            cubit_face_type = FACE_EDGE2;
            num_element_linear_nodes = 3;
            break;
         }
         case (6) :
         {
            cubit_element_type = ELEMENT_TRI6;
            cubit_face_type = FACE_EDGE3;
            num_element_linear_nodes = 3;
            break;
         }
         case (4) :
         {
            cubit_element_type = ELEMENT_QUAD4;
            cubit_face_type = FACE_EDGE2;
            num_element_linear_nodes = 4;
            break;
         }
         case (9) :
         {
            cubit_element_type = ELEMENT_QUAD9;
            cubit_face_type = FACE_EDGE3;
            num_element_linear_nodes = 4;
            break;
         }
         default :
         {
            MFEM_ABORT("Don't know what to do with a " << num_node_per_el <<
                       " node 2D element\n");
         }
      }
   }
   else if (num_dim == 3)
   {
      switch (num_node_per_el)
      {
         case (4) :
         {
            cubit_element_type = ELEMENT_TET4;
            cubit_face_type = FACE_TRI3;
            num_element_linear_nodes = 4;
            break;
         }
         case (10) :
         {
            cubit_element_type = ELEMENT_TET10;
            cubit_face_type = FACE_TRI6;
            num_element_linear_nodes = 4;
            break;
         }
         case (8) :
         {
            cubit_element_type = ELEMENT_HEX8;
            cubit_face_type = FACE_QUAD4;
            num_element_linear_nodes = 8;
            break;
         }
         case (27) :
         {
            cubit_element_type = ELEMENT_HEX27;
            cubit_face_type = FACE_QUAD9;
            num_element_linear_nodes = 8;
            break;
         }
         default :
         {
            MFEM_ABORT("Don't know what to do with a " << num_node_per_el <<
                       " node 3D element\n");
         }
      }
   }
   else
   {
      MFEM_ABORT("Invalid dimension: num_dim = " << num_dim);
   }

   // Determine order of elements
   int order = 0;
   if (cubit_element_type == ELEMENT_TRI3 || cubit_element_type == ELEMENT_QUAD4 ||
       cubit_element_type == ELEMENT_TET4 || cubit_element_type == ELEMENT_HEX8)
   {
      order = 1;
   }
   else if (cubit_element_type == ELEMENT_TRI6 ||
            cubit_element_type == ELEMENT_QUAD9 ||
            cubit_element_type == ELEMENT_TET10 || cubit_element_type == ELEMENT_HEX27)
   {
      order = 2;
   }

   // create array for number of sides in side sets
   size_t *num_side_in_ss  = new size_t[num_side_sets];
   for (int i = 0; i < (int) num_side_sets; i++)
   {
      sprintf(temp_str, "num_side_ss%d", i+1);
      if ((retval = nc_inq_dimid(ncid, temp_str, &temp_id)) ||
          (retval = nc_inq_dim(ncid, temp_id, str_dummy, &num_side_in_ss[i])))
      {
         MFEM_ABORT("Fatal NetCDF error: " << nc_strerror(retval));
      }
   }

   // read the coordinates
   double *coordx = new double[num_nodes];
   double *coordy = new double[num_nodes];
   double *coordz = new double[num_nodes];

   if ((retval = nc_inq_varid(ncid, "coordx", &id)) ||
       (retval = nc_get_var_double(ncid, id, coordx)) ||
       (retval = nc_inq_varid(ncid, "coordy", &id)) ||
       (retval = nc_get_var_double(ncid, id, coordy)))
   {
      MFEM_ABORT("Fatal NetCDF error: " << nc_strerror(retval));
   }

   if (num_dim == 3)
   {
      if ((retval = nc_inq_varid(ncid, "coordz", &id)) ||
          (retval = nc_get_var_double(ncid, id, coordz)))
      {
         MFEM_ABORT("Fatal NetCDF error: " << nc_strerror(retval));
      }
   }

   // read the element blocks
   int **elem_blk = new int*[num_el_blk];
   for (int i = 0; i < (int) num_el_blk; i++)
   {
      elem_blk[i] = new int[num_el_in_blk[i] * num_node_per_el];
      sprintf(temp_str, "connect%d", i+1);
      if ((retval = nc_inq_varid(ncid, temp_str, &temp_id)) ||
          (retval = nc_get_var_int(ncid, temp_id, elem_blk[i])))
      {
         MFEM_ABORT("Fatal NetCDF error: " << nc_strerror(retval));
      }
   }
   int *ebprop = new int[num_el_blk];
   if ((retval = nc_inq_varid(ncid, "eb_prop1", &id)) ||
       (retval = nc_get_var_int(ncid, id, ebprop)))
   {
      MFEM_ABORT("Fatal NetCDF error: " << nc_strerror(retval));
   }

   // read the side sets, a side is is given by (element, face) pairs

   int **elem_ss = new int*[num_side_sets];
   int **side_ss = new int*[num_side_sets];

   for (int i = 0; i < (int) num_side_sets; i++)
   {
      elem_ss[i] = new int[num_side_in_ss[i]];
      side_ss[i] = new int[num_side_in_ss[i]];

      sprintf(temp_str, "elem_ss%d", i+1);
      if ((retval = nc_inq_varid(ncid, temp_str, &temp_id)) ||
          (retval = nc_get_var_int(ncid, temp_id, elem_ss[i])))
      {
         MFEM_ABORT("Fatal NetCDF error: " << nc_strerror(retval));
      }

      sprintf(temp_str,"side_ss%d",i+1);
      if ((retval = nc_inq_varid(ncid, temp_str, &temp_id)) ||
          (retval = nc_get_var_int(ncid, temp_id, side_ss[i])))
      {
         MFEM_ABORT("Fatal NetCDF error: " << nc_strerror(retval));
      }
   }

   int *ssprop = new int[num_side_sets];
   if ((num_side_sets > 0) &&
       ((retval = nc_inq_varid(ncid, "ss_prop1", &id)) ||
        (retval = nc_get_var_int(ncid, id, ssprop))))
   {
      MFEM_ABORT("Fatal NetCDF error: " << nc_strerror(retval));
   }

   // convert (elem,side) pairs to 2D elements


   int num_face_nodes = 0;
   int num_face_linear_nodes = 0;

   switch (cubit_face_type)
   {
      case (FACE_EDGE2):
      {
         num_face_nodes = 2;
         num_face_linear_nodes = 2;
         break;
      }
      case (FACE_EDGE3):
      {
         num_face_nodes = 3;
         num_face_linear_nodes = 2;
         break;
      }
      case (FACE_TRI3):
      {
         num_face_nodes = 3;
         num_face_linear_nodes = 3;
         break;
      }
      case (FACE_TRI6):
      {
         num_face_nodes = 6;
         num_face_linear_nodes = 3;
         break;
      }
      case (FACE_QUAD4):
      {
         num_face_nodes = 4;
         num_face_linear_nodes = 4;
         break;
      }
      case (FACE_QUAD9):
      {
         num_face_nodes = 9;
         num_face_linear_nodes = 4;
         break;
      }
   }

   // given a global element number, determine the element block and local
   // element number
   int *start_of_block = new int[num_el_blk+1];
   start_of_block[0] = 0;
   for (int i = 1; i < (int) num_el_blk+1; i++)
   {
      start_of_block[i] = start_of_block[i-1] + num_el_in_blk[i-1];
   }

   int **ss_node_id = new int*[num_side_sets];

   for (int i = 0; i < (int) num_side_sets; i++)
   {
      ss_node_id[i] = new int[num_side_in_ss[i]*num_face_nodes];
      for (int j = 0; j < (int) num_side_in_ss[i]; j++)
      {
         int glob_ind = elem_ss[i][j]-1;
         int iblk = 0;
         int loc_ind;
         while (iblk < (int) num_el_blk && glob_ind >= start_of_block[iblk+1])
         {
            iblk++;
         }
         if (iblk >= (int) num_el_blk)
         {
            MFEM_ABORT("Sideset element does not exist");
         }
         loc_ind = glob_ind - start_of_block[iblk];
         int this_side = side_ss[i][j];
         int ielem = loc_ind*num_node_per_el;

         for (int k = 0; k < num_face_nodes; k++)
         {
            int inode;
            switch (cubit_element_type)
            {
               case (ELEMENT_TRI3):
               {
                  inode = sideMapTri3[this_side-1][k];
                  break;
               }
               case (ELEMENT_TRI6):
               {
                  inode = sideMapTri6[this_side-1][k];
                  break;
               }
               case (ELEMENT_QUAD4):
               {
                  inode = sideMapQuad4[this_side-1][k];
                  break;
               }
               case (ELEMENT_QUAD9):
               {
                  inode = sideMapQuad9[this_side-1][k];
                  break;
               }
               case (ELEMENT_TET4):
               {
                  inode = sideMapTet4[this_side-1][k];
                  break;
               }
               case (ELEMENT_TET10):
               {
                  inode = sideMapTet10[this_side-1][k];
                  break;
               }
               case (ELEMENT_HEX8):
               {
                  inode = sideMapHex8[this_side-1][k];
                  break;
               }
               case (ELEMENT_HEX27):
               {
                  inode = sideMapHex27[this_side-1][k];
                  break;
               }
            }
            ss_node_id[i][j*num_face_nodes+k] =
               elem_blk[iblk][ielem + inode - 1];
         }
      }
   }

   // we need another node ID mapping since MFEM needs contiguous vertex IDs
   std::vector<int> uniqueVertexID;

   for (int iblk = 0; iblk < (int) num_el_blk; iblk++)
   {
      for (int i = 0; i < (int) num_el_in_blk[iblk]; i++)
      {
         for (int j = 0; j < num_element_linear_nodes; j++)
         {
            uniqueVertexID.push_back(elem_blk[iblk][i*num_node_per_el + j]);
         }
      }
   }
   std::sort(uniqueVertexID.begin(), uniqueVertexID.end());
   std::vector<int>::iterator newEnd;
   newEnd = std::unique(uniqueVertexID.begin(), uniqueVertexID.end());
   uniqueVertexID.resize(std::distance(uniqueVertexID.begin(), newEnd));

   // OK at this point uniqueVertexID contains a list of all the nodes that are
   // actually used by the mesh, 1-based, and sorted. We need to invert this
   // list, the inverse is a map

   std::map<int,int> cubitToMFEMVertMap;
   for (int i = 0; i < (int) uniqueVertexID.size(); i++)
   {
      cubitToMFEMVertMap[uniqueVertexID[i]] = i+1;
   }
   MFEM_ASSERT(cubitToMFEMVertMap.size() == uniqueVertexID.size(),
               "This should never happen\n");

   // OK now load up the MFEM mesh structures

   // load up the vertices

   NumOfVertices = uniqueVertexID.size();
   vertices.SetSize(NumOfVertices);
   for (int i = 0; i < (int) uniqueVertexID.size(); i++)
   {
      vertices[i](0) = coordx[uniqueVertexID[i] - 1];
      vertices[i](1) = coordy[uniqueVertexID[i] - 1];
      if (Dim == 3)
      {
         vertices[i](2) = coordz[uniqueVertexID[i] - 1];
      }
   }

   NumOfElements = num_elem;
   elements.SetSize(num_elem);
   int elcount = 0;
   int renumberedVertID[8];
   for (int iblk = 0; iblk < (int) num_el_blk; iblk++)
   {
      int NumNodePerEl = num_node_per_el;
      for (int i = 0; i < (int) num_el_in_blk[iblk]; i++)
      {
         for (int j = 0; j < num_element_linear_nodes; j++)
         {
            renumberedVertID[j] =
               cubitToMFEMVertMap[elem_blk[iblk][i*NumNodePerEl+j]]-1;
         }

         switch (cubit_element_type)
         {
            case (ELEMENT_TRI3):
            case (ELEMENT_TRI6):
            {
               elements[elcount] = new Triangle(renumberedVertID,ebprop[iblk]);
               break;
            }
            case (ELEMENT_QUAD4):
            case (ELEMENT_QUAD9):
            {
               elements[elcount] = new Quadrilateral(renumberedVertID,ebprop[iblk]);
               break;
            }
            case (ELEMENT_TET4):
            case (ELEMENT_TET10):
            {
#ifdef MFEM_USE_MEMALLOC
               elements[elcount] = TetMemory.Alloc();
               elements[elcount]->SetVertices(renumberedVertID);
               elements[elcount]->SetAttribute(ebprop[iblk]);
#else
               elements[elcount] = new Tetrahedron(renumberedVertID,
                                                   ebprop[iblk]);
#endif
               break;
            }
            case (ELEMENT_HEX8):
            case (ELEMENT_HEX27):
            {
               elements[elcount] = new Hexahedron(renumberedVertID,ebprop[iblk]);
               break;
            }
         }
         elcount++;
      }
   }

   // load up the boundary elements

   NumOfBdrElements = 0;
   for (int iss = 0; iss < (int) num_side_sets; iss++)
   {
      NumOfBdrElements += num_side_in_ss[iss];
   }
   boundary.SetSize(NumOfBdrElements);
   int sidecount = 0;
   for (int iss = 0; iss < (int) num_side_sets; iss++)
   {
      for (int i = 0; i < (int) num_side_in_ss[iss]; i++)
      {
         for (int j = 0; j < num_face_linear_nodes; j++)
         {
            renumberedVertID[j] =
               cubitToMFEMVertMap[ss_node_id[iss][i*num_face_nodes+j]] - 1;
         }
         switch (cubit_face_type)
         {
            case (FACE_EDGE2):
            case (FACE_EDGE3):
            {
               boundary[sidecount] = new Segment(renumberedVertID,ssprop[iss]);
               break;
            }
            case (FACE_TRI3):
            case (FACE_TRI6):
            {
               boundary[sidecount] = new Triangle(renumberedVertID,ssprop[iss]);
               break;
            }
            case (FACE_QUAD4):
            case (FACE_QUAD9):
            {
               boundary[sidecount] = new Quadrilateral(renumberedVertID,ssprop[iss]);
               break;
            }
         }
         sidecount++;
      }
   }

   if (order == 2)
   {
      curved = 1;
      int *mymap = NULL;

      switch (cubit_element_type)
      {
         case (ELEMENT_TRI6):
         {
            mymap = (int *) mfemToGenesisTri6;
            break;
         }
         case (ELEMENT_QUAD9):
         {
            mymap = (int *) mfemToGenesisQuad9;
            break;
         }
         case (ELEMENT_TET10):
         {
            mymap = (int *) mfemToGenesisTet10;
            break;
         }
         case (ELEMENT_HEX27):
         {
            mymap = (int *) mfemToGenesisHex27;
            break;
         }
         case (ELEMENT_TRI3):
         case (ELEMENT_QUAD4):
         case (ELEMENT_TET4):
         case (ELEMENT_HEX8):
         {
            MFEM_ABORT("Something went wrong. Linear elements detected when order is 2.");
            break;
         }
      }

      FinalizeTopology();

      // Define quadratic FE space
      FiniteElementCollection *fec = new H1_FECollection(2,3);
      FiniteElementSpace *fes = new FiniteElementSpace(this, fec, Dim,
                                                       Ordering::byVDIM);
      Nodes = new GridFunction(fes);
      Nodes->MakeOwner(fec); // Nodes will destroy 'fec' and 'fes'
      own_nodes = 1;

      // int nTotDofs = fes->GetNDofs();
      // int nTotVDofs = fes->GetVSize();
      //    mfem::out << endl << "nTotDofs = " << nTotDofs << "  nTotVDofs "
      //              << nTotVDofs << endl << endl;

      for (int i = 0; i < NumOfElements; i++)
      {
         Array<int> dofs;

         fes->GetElementDofs(i, dofs);
         Array<int> vdofs;
         vdofs.SetSize(dofs.Size());
         for (int l = 0; l < dofs.Size(); l++) { vdofs[l] = dofs[l]; }
         fes->DofsToVDofs(vdofs);
         int iblk = 0;
         int loc_ind;
         while (iblk < (int) num_el_blk && i >= start_of_block[iblk+1]) { iblk++; }
         loc_ind = i - start_of_block[iblk];
         for (int j = 0; j < dofs.Size(); j++)
         {
            int point_id = elem_blk[iblk][loc_ind*num_node_per_el + mymap[j] - 1] - 1;
            (*Nodes)(vdofs[j])   = coordx[point_id];
            (*Nodes)(vdofs[j]+1) = coordy[point_id];
            if (Dim == 3)
            {
               (*Nodes)(vdofs[j]+2) = coordz[point_id];
            }
         }
      }
   }

   // clean up all netcdf stuff

   nc_close(ncid);

   for (int i = 0; i < (int) num_side_sets; i++)
   {
      delete [] elem_ss[i];
      delete [] side_ss[i];
   }

   delete [] elem_ss;
   delete [] side_ss;
   delete [] num_el_in_blk;
   delete [] num_side_in_ss;
   delete [] coordx;
   delete [] coordy;
   delete [] coordz;

   for (int i = 0; i < (int) num_el_blk; i++)
   {
      delete [] elem_blk[i];
   }

   delete [] elem_blk;
   delete [] start_of_block;

   for (int i = 0; i < (int) num_side_sets; i++)
   {
      delete [] ss_node_id[i];
   }
   delete [] ss_node_id;
   delete [] ebprop;
   delete [] ssprop;

}
#endif // #ifdef MFEM_USE_NETCDF

} // namespace mfem
