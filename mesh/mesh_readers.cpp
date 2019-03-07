// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#include "mesh_headers.hpp"
#include "../fem/fem.hpp"
#include "../general/text.hpp"

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
      Make3D(nx, ny, nz, type, sx, sy, sz, generate_edges, true);
      // TODO: maybe have an option in the file to control ordering?
   }
   else
   {
      MFEM_ABORT("For inline mesh, must specify an element type ="
                 " [segment, tri, quad, tet, hex, wedge]");
   }
}

void Mesh::ReadGmshMesh(std::istream &input)
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
            2, // 2-node line.
            3, // 3-node triangle.
            4, // 4-node quadrangle.
            4, // 4-node tetrahedron.
            8, // 8-node hexahedron.
            6, // 6-node prism.
            5, // 5-node pyramid.
            3, /* 3-node second order line (2 nodes associated with the vertices
                    and 1 with the edge). */
            6, /* 6-node second order triangle (3 nodes associated with the
                    vertices and 3 with the edges). */
            9, /* 9-node second order quadrangle (4 nodes associated with the
                    vertices, 4 with the edges and 1 with the face). */
            10,/* 10-node second order tetrahedron (4 nodes associated with the
                     vertices and 6 with the edges). */
            27,/* 27-node second order hexahedron (8 nodes associated with the
                     vertices, 12 with the edges, 6 with the faces and 1 with
                     the volume). */
            18,/* 18-node second order prism (6 nodes associated with the
                     vertices, 9 with the edges and 3 with the quadrangular
                     faces). */
            14,/* 14-node second order pyramid (5 nodes associated with the
                     vertices, 8 with the edges and 1 with the quadrangular
                     face). */
            1, // 1-node point.
            8, /* 8-node second order quadrangle (4 nodes associated with the
                    vertices and 4 with the edges). */
            20,/* 20-node second order hexahedron (8 nodes associated with the
                     vertices and 12 with the edges). */
            15,/* 15-node second order prism (6 nodes associated with the
                     vertices and 9 with the edges). */
            13,/* 13-node second order pyramid (5 nodes associated with the
                     vertices and 8 with the edges). */
            9, /* 9-node third order incomplete triangle (3 nodes associated
                    with the vertices, 6 with the edges) */
            10,/* 10-node third order triangle (3 nodes associated with the
                     vertices, 6 with the edges, 1 with the face) */
            12,/* 12-node fourth order incomplete triangle (3 nodes associated
                     with the vertices, 9 with the edges) */
            15,/* 15-node fourth order triangle (3 nodes associated with the
                     vertices, 9 with the edges, 3 with the face) */
            15,/* 15-node fifth order incomplete triangle (3 nodes associated
                     with the vertices, 12 with the edges) */
            21,/* 21-node fifth order complete triangle (3 nodes associated with
                     the vertices, 12 with the edges, 6 with the face) */
            4, /* 4-node third order edge (2 nodes associated with the vertices,
                    2 internal to the edge) */
            5, /* 5-node fourth order edge (2 nodes associated with the
                    vertices, 3 internal to the edge) */
            6, /* 6-node fifth order edge (2 nodes associated with the vertices,
                    4 internal to the edge) */
            20 /* 20-node third order tetrahedron (4 nodes associated with the
                     vertices, 12 with the edges, 4 with the faces) */
         };

         vector<Element*> elements_0D, elements_1D, elements_2D, elements_3D;
         elements_0D.reserve(num_of_all_elements);
         elements_1D.reserve(num_of_all_elements);
         elements_2D.reserve(num_of_all_elements);
         elements_3D.reserve(num_of_all_elements);

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
                  switch (type_of_element)
                  {
                     case 1: // 2-node line
                     {
                        elements_1D.push_back(
                           new Segment(&vert_indices[0], phys_domain));
                        break;
                     }
                     case 2: // 3-node triangle
                     {
                        elements_2D.push_back(
                           new Triangle(&vert_indices[0], phys_domain));
                        break;
                     }
                     case 3: // 4-node quadrangle
                     {
                        elements_2D.push_back(
                           new Quadrilateral(&vert_indices[0], phys_domain));
                        break;
                     }
                     case 4: // 4-node tetrahedron
                     {
                        elements_3D.push_back(
                           new Tetrahedron(&vert_indices[0], phys_domain));
                        break;
                     }
                     case 5: // 8-node hexahedron
                     {
                        elements_3D.push_back(
                           new Hexahedron(&vert_indices[0], phys_domain));
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

               // non-positive attributes are not allowed in MFEM
               if (phys_domain <= 0)
               {
                  MFEM_ABORT("Non-positive element attribute in Gmsh mesh!");
               }

               // initialize the mesh element
               switch (type_of_element)
               {
                  case 1: // 2-node line
                  {
                     elements_1D.push_back(
                        new Segment(&vert_indices[0], phys_domain));
                     break;
                  }
                  case 2: // 3-node triangle
                  {
                     elements_2D.push_back(
                        new Triangle(&vert_indices[0], phys_domain));
                     break;
                  }
                  case 3: // 4-node quadrangle
                  {
                     elements_2D.push_back(
                        new Quadrilateral(&vert_indices[0], phys_domain));
                     break;
                  }
                  case 4: // 4-node tetrahedron
                  {
                     elements_3D.push_back(
                        new Tetrahedron(&vert_indices[0], phys_domain));
                     break;
                  }
                  case 5: // 8-node hexahedron
                  {
                     elements_3D.push_back(
                        new Hexahedron(&vert_indices[0], phys_domain));
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
         }
         else
         {
            MFEM_ABORT("Gmsh file : no elements found");
            return;
         }

         MFEM_CONTRACT_VAR(n_partitions);
         MFEM_CONTRACT_VAR(elem_domain);

      } // section '$Elements'
   } // we reach the end of the file
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
               elements[elcount] = new Tetrahedron(renumberedVertID,ebprop[iblk]);
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

      // Generate faces and edges so that we can define quadratic
      // FE space on the mesh

      // Generate faces
      if (Dim > 2)
      {
         GetElementToFaceTable();
         GenerateFaces();
      }
      else
      {
         NumOfFaces = 0;
      }

      // Generate edges
      el_to_edge = new Table;
      NumOfEdges = GetElementToEdgeTable(*el_to_edge, be_to_edge);
      if (Dim == 2)
      {
         GenerateFaces(); // 'Faces' in 2D refers to the edges
      }

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
