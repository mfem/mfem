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

#include "common_get_mesh.hpp"

using namespace mfem;

namespace mfem_test_fem
{

Mesh * GetMesh(MeshType type, double lx, double ly, double lz)
{
   Mesh * mesh = NULL;
   double c[3];
   int    v[8];

   switch (type)
   {
      case SEGMENT:
         mesh = new Mesh(1, 2, 1);
         c[0] = 0.0;
         mesh->AddVertex(c);
         c[0] = lx;
         mesh->AddVertex(c);
         v[0] = 0; v[1] = 1;
         mesh->AddSegment(v);
         {
            Element * el = mesh->NewElement(Geometry::POINT);
            el->SetAttribute(1);
            el->SetVertices(&v[0]);
            mesh->AddBdrElement(el);
         }
         {
            Element * el = mesh->NewElement(Geometry::POINT);
            el->SetAttribute(2);
            el->SetVertices(&v[1]);
            mesh->AddBdrElement(el);
         }
         break;
      case QUADRILATERAL:
         mesh = new Mesh(2, 4, 1);
         c[0] = 0.0; c[1] = 0.0;
         mesh->AddVertex(c);
         c[0] = lx; c[1] = 0.0;
         mesh->AddVertex(c);
         c[0] = lx; c[1] = ly;
         mesh->AddVertex(c);
         c[0] = 0.0; c[1] = ly;
         mesh->AddVertex(c);

         v[0] = 0; v[1] = 1; v[2] = 2; v[3] = 3;
         mesh->AddQuad(v);
         break;
      case TRIANGLE2A:
         mesh = new Mesh(2, 4, 2);
         c[0] = 0.0; c[1] = 0.0;
         mesh->AddVertex(c);
         c[0] = lx; c[1] = 0.0;
         mesh->AddVertex(c);
         c[0] = lx; c[1] = ly;
         mesh->AddVertex(c);
         c[0] = 0.0; c[1] = ly;
         mesh->AddVertex(c);

         v[0] = 0; v[1] = 1; v[2] = 2;
         mesh->AddTri(v);
         v[0] = 2; v[1] = 3; v[2] = 0;
         mesh->AddTri(v);
         break;
      case TRIANGLE2B:
         mesh = new Mesh(2, 4, 2);
         c[0] = 0.0; c[1] = 0.0;
         mesh->AddVertex(c);
         c[0] = lx; c[1] = 0.0;
         mesh->AddVertex(c);
         c[0] = lx; c[1] = ly;
         mesh->AddVertex(c);
         c[0] = 0.0; c[1] = ly;
         mesh->AddVertex(c);

         v[0] = 1; v[1] = 2; v[2] = 0;
         mesh->AddTri(v);
         v[0] = 3; v[1] = 0; v[2] = 2;
         mesh->AddTri(v);
         break;
      case TRIANGLE2C:
         mesh = new Mesh(2, 4, 2);
         c[0] = 0.0; c[1] = 0.0;
         mesh->AddVertex(c);
         c[0] = lx; c[1] = 0.0;
         mesh->AddVertex(c);
         c[0] = lx; c[1] = ly;
         mesh->AddVertex(c);
         c[0] = 0.0; c[1] = ly;
         mesh->AddVertex(c);

         v[0] = 2; v[1] = 0; v[2] = 1;
         mesh->AddTri(v);
         v[0] = 0; v[1] = 2; v[2] = 3;
         mesh->AddTri(v);
         break;
      case TRIANGLE4:
         mesh = new Mesh(2, 5, 4);
         c[0] = 0.0; c[1] = 0.0;
         mesh->AddVertex(c);
         c[0] = lx; c[1] = 0.0;
         mesh->AddVertex(c);
         c[0] = lx; c[1] = ly;
         mesh->AddVertex(c);
         c[0] = 0.0; c[1] = ly;
         mesh->AddVertex(c);
         c[0] = 0.5 * lx; c[1] = 0.5 * ly;
         mesh->AddVertex(c);

         v[0] = 0; v[1] = 1; v[2] = 4;
         mesh->AddTri(v);
         v[0] = 1; v[1] = 2; v[2] = 4;
         mesh->AddTri(v);
         v[0] = 2; v[1] = 3; v[2] = 4;
         mesh->AddTri(v);
         v[0] = 3; v[1] = 0; v[2] = 4;
         mesh->AddTri(v);
         break;
      case MIXED2D:
         mesh = new Mesh(2, 6, 4);
         c[0] = 0.0; c[1] = 0.0;
         mesh->AddVertex(c);
         c[0] = lx; c[1] = 0.0;
         mesh->AddVertex(c);
         c[0] = lx; c[1] = ly;
         mesh->AddVertex(c);
         c[0] = 0.0; c[1] = ly;
         mesh->AddVertex(c);
         c[0] = 0.5 * ly; c[1] = 0.5 * ly;
         mesh->AddVertex(c);
         c[0] = lx - 0.5 * ly; c[1] = 0.5 * ly;
         mesh->AddVertex(c);

         v[0] = 0; v[1] = 1; v[2] = 5; v[3] = 4;
         mesh->AddQuad(v);
         v[0] = 1; v[1] = 2; v[2] = 5;
         mesh->AddTri(v);
         v[0] = 2; v[1] = 3; v[2] = 4; v[3] = 5;
         mesh->AddQuad(v);
         v[0] = 3; v[1] = 0; v[2] = 4;
         mesh->AddTri(v);
         break;
      case HEXAHEDRON:
         mesh = new Mesh(3, 8, 1);
         c[0] = 0.0; c[1] = 0.0; c[2] = 0.0;
         mesh->AddVertex(c);
         c[0] = lx; c[1] = 0.0; c[2] = 0.0;
         mesh->AddVertex(c);
         c[0] = lx; c[1] = ly; c[2] = 0.0;
         mesh->AddVertex(c);
         c[0] = 0.0; c[1] = ly; c[2] = 0.0;
         mesh->AddVertex(c);
         c[0] = 0.0; c[1] = 0.0; c[2] = lz;
         mesh->AddVertex(c);
         c[0] = lx; c[1] = 0.0; c[2] = lz;
         mesh->AddVertex(c);
         c[0] = lx; c[1] = ly; c[2] = lz;
         mesh->AddVertex(c);
         c[0] = 0.0; c[1] = ly; c[2] = lz;
         mesh->AddVertex(c);

         v[0] = 0; v[1] = 1; v[2] = 2; v[3] = 3;
         v[4] = 4; v[5] = 5; v[6] = 6; v[7] = 7;
         mesh->AddHex(v);
         break;
      case HEXAHEDRON2A:
      case HEXAHEDRON2B:
      case HEXAHEDRON2C:
      case HEXAHEDRON2D:
         mesh = new Mesh(3, 12, 2);
         c[0] = 0.0; c[1] = 0.0; c[2] = 0.0;
         mesh->AddVertex(c);
         c[0] = 0.5 * lx; c[1] = 0.0; c[2] = 0.0;
         mesh->AddVertex(c);
         c[0] = lx; c[1] = 0.0; c[2] = 0.0;
         mesh->AddVertex(c);
         c[0] = lx; c[1] = ly; c[2] = 0.0;
         mesh->AddVertex(c);
         c[0] = 0.5 * lx; c[1] = ly; c[2] = 0.0;
         mesh->AddVertex(c);
         c[0] = 0.0; c[1] = ly; c[2] = 0.0;
         mesh->AddVertex(c);
         c[0] = 0.0; c[1] = 0.0; c[2] = lz;
         mesh->AddVertex(c);
         c[0] = 0.5 * lx; c[1] = 0.0; c[2] = lz;
         mesh->AddVertex(c);
         c[0] = lx; c[1] = 0.0; c[2] = lz;
         mesh->AddVertex(c);
         c[0] = lx; c[1] = ly; c[2] = lz;
         mesh->AddVertex(c);
         c[0] = 0.5 * lx; c[1] = ly; c[2] = lz;
         mesh->AddVertex(c);
         c[0] = 0.0; c[1] = ly; c[2] = lz;
         mesh->AddVertex(c);

         v[0] = 0; v[1] = 5; v[2] = 11; v[3] = 6;
         v[4] = 1; v[5] = 4; v[6] = 10; v[7] = 7;
         mesh->AddHex(v);

         switch (type)
         {
            case HEXAHEDRON2A: // Face Orientation 1
               v[0] = 4; v[1] = 10; v[2] = 7; v[3] = 1;
               v[4] = 3; v[5] = 9; v[6] = 8; v[7] = 2;
               mesh->AddHex(v);
               break;
            case HEXAHEDRON2B: // Face Orientation 3
               v[0] = 10; v[1] = 7; v[2] = 1; v[3] = 4;
               v[4] = 9; v[5] = 8; v[6] = 2; v[7] = 3;
               mesh->AddHex(v);
               break;
            case HEXAHEDRON2C: // Face Orientation 5
               v[0] = 7; v[1] = 1; v[2] = 4; v[3] = 10;
               v[4] = 8; v[5] = 2; v[6] = 3; v[7] = 9;
               mesh->AddHex(v);
               break;
            case HEXAHEDRON2D: // Face Orientation 7
               v[0] = 1; v[1] = 4; v[2] = 10; v[3] = 7;
               v[4] = 2; v[5] = 3; v[6] = 9; v[7] = 8;
               mesh->AddHex(v);
               break;
            default:
               // Cannot happen
               break;
         }
         break;
      case WEDGE2:
         mesh = new Mesh(3, 8, 2);
         c[0] = 0.0; c[1] = 0.0; c[2] = 0.0;
         mesh->AddVertex(c);
         c[0] = lx; c[1] = 0.0; c[2] = 0.0;
         mesh->AddVertex(c);
         c[0] = lx; c[1] = ly; c[2] = 0.0;
         mesh->AddVertex(c);
         c[0] = 0.0; c[1] = ly; c[2] = 0.0;
         mesh->AddVertex(c);
         c[0] = 0.0; c[1] = 0.0; c[2] = lz;
         mesh->AddVertex(c);
         c[0] = lx; c[1] = 0.0; c[2] = lz;
         mesh->AddVertex(c);
         c[0] = lx; c[1] = ly; c[2] = lz;
         mesh->AddVertex(c);
         c[0] = 0.0; c[1] = ly; c[2] = lz;
         mesh->AddVertex(c);

         v[0] = 0; v[1] = 1; v[2] = 2; v[3] = 4; v[4] = 5; v[5] = 6;
         mesh->AddWedge(v);
         v[0] = 0; v[1] = 2; v[2] = 3; v[3] = 4; v[4] = 6; v[5] = 7;
         mesh->AddWedge(v);
         break;
      case TETRAHEDRA:
         mesh = new Mesh(3, 8, 5);
         c[0] = 0.0; c[1] = 0.0; c[2] = 0.0;
         mesh->AddVertex(c);
         c[0] = lx; c[1] = 0.0; c[2] = 0.0;
         mesh->AddVertex(c);
         c[0] = lx; c[1] = ly; c[2] = 0.0;
         mesh->AddVertex(c);
         c[0] = 0.0; c[1] = ly; c[2] = 0.0;
         mesh->AddVertex(c);
         c[0] = 0.0; c[1] = 0.0; c[2] = lz;
         mesh->AddVertex(c);
         c[0] = lx; c[1] = 0.0; c[2] = lz;
         mesh->AddVertex(c);
         c[0] = lx; c[1] = ly; c[2] = lz;
         mesh->AddVertex(c);
         c[0] = 0.0; c[1] = ly; c[2] = lz;
         mesh->AddVertex(c);

         v[0] = 0; v[1] = 2; v[2] = 7; v[3] = 5;
         mesh->AddTet(v);
         v[0] = 6; v[1] = 7; v[2] = 2; v[3] = 5;
         mesh->AddTet(v);
         v[0] = 4; v[1] = 7; v[2] = 5; v[3] = 0;
         mesh->AddTet(v);
         v[0] = 1; v[1] = 0; v[2] = 5; v[3] = 2;
         mesh->AddTet(v);
         v[0] = 3; v[1] = 7; v[2] = 0; v[3] = 2;
         mesh->AddTet(v);
         break;
      case WEDGE4:
         mesh = new Mesh(3, 10, 4);
         c[0] = 0.0; c[1] = 0.0; c[2] = 0.0;
         mesh->AddVertex(c);
         c[0] = lx; c[1] = 0.0; c[2] = 0.0;
         mesh->AddVertex(c);
         c[0] = lx; c[1] = ly; c[2] = 0.0;
         mesh->AddVertex(c);
         c[0] = 0.0; c[1] = ly; c[2] = 0.0;
         mesh->AddVertex(c);
         c[0] = 0.5 * lx; c[1] = 0.5 * ly; c[2] = 0.0;
         mesh->AddVertex(c);
         c[0] = 0.0; c[1] = 0.0; c[2] = lz;
         mesh->AddVertex(c);
         c[0] = lx; c[1] = 0.0; c[2] = lz;
         mesh->AddVertex(c);
         c[0] = lx; c[1] = ly; c[2] = lz;
         mesh->AddVertex(c);
         c[0] = 0.0; c[1] = ly; c[2] = lz;
         mesh->AddVertex(c);
         c[0] = 0.5 * lx; c[1] = 0.5 * ly; c[2] = lz;
         mesh->AddVertex(c);

         v[0] = 0; v[1] = 1; v[2] = 4; v[3] = 5; v[4] = 6; v[5] = 9;
         mesh->AddWedge(v);
         v[0] = 1; v[1] = 2; v[2] = 4; v[3] = 6; v[4] = 7; v[5] = 9;
         mesh->AddWedge(v);
         v[0] = 2; v[1] = 3; v[2] = 4; v[3] = 7; v[4] = 8; v[5] = 9;
         mesh->AddWedge(v);
         v[0] = 3; v[1] = 0; v[2] = 4; v[3] = 8; v[4] = 5; v[5] = 9;
         mesh->AddWedge(v);
         break;
      case MIXED3D6:
         mesh = new Mesh(3, 12, 6);
         c[0] = 0.0; c[1] = 0.0; c[2] = 0.0;
         mesh->AddVertex(c);
         c[0] = lx; c[1] = 0.0; c[2] = 0.0;
         mesh->AddVertex(c);
         c[0] = lx; c[1] = ly; c[2] = 0.0;
         mesh->AddVertex(c);
         c[0] = 0.0; c[1] = ly; c[2] = 0.0;
         mesh->AddVertex(c);
         c[0] = 0.5 * lz; c[1] = 0.5 * lz; c[2] = 0.5 * lz;
         mesh->AddVertex(c);
         c[0] = lx - 0.5 * lz; c[1] = 0.5 * lz; c[2] = 0.5 * lz;
         mesh->AddVertex(c);
         c[0] = lx - 0.5 * lz; c[1] = ly - 0.5 * lz; c[2] = 0.5 * lz;
         mesh->AddVertex(c);
         c[0] = 0.5 * lz; c[1] = ly - 0.5 * lz; c[2] = 0.5 * lz;
         mesh->AddVertex(c);
         c[0] = 0.0; c[1] = 0.0; c[2] = lz;
         mesh->AddVertex(c);
         c[0] = lx; c[1] = 0.0; c[2] = lz;
         mesh->AddVertex(c);
         c[0] = lx; c[1] = ly; c[2] = lz;
         mesh->AddVertex(c);
         c[0] = 0.0; c[1] = ly; c[2] = lz;
         mesh->AddVertex(c);

         v[0] = 0; v[1] = 1; v[2] = 2; v[3] = 3;
         v[4] = 4; v[5] = 5; v[6] = 6; v[7] = 7;
         mesh->AddHex(v);
         v[0] = 0; v[1] = 4; v[2] = 8; v[3] = 1; v[4] = 5; v[5] = 9;
         mesh->AddWedge(v);
         v[0] = 1; v[1] = 5; v[2] = 9; v[3] = 2; v[4] = 6; v[5] = 10;
         mesh->AddWedge(v);
         v[0] = 2; v[1] = 6; v[2] = 10; v[3] = 3; v[4] = 7; v[5] = 11;
         mesh->AddWedge(v);
         v[0] = 3; v[1] = 7; v[2] = 11; v[3] = 0; v[4] = 4; v[5] = 8;
         mesh->AddWedge(v);
         v[0] = 4; v[1] = 5; v[2] = 6; v[3] = 7;
         v[4] = 8; v[5] = 9; v[6] = 10; v[7] = 11;
         mesh->AddHex(v);
         break;
      case MIXED3D8:
         mesh = new Mesh(3, 10, 8);
         c[0] = 0.0; c[1] = 0.0; c[2] = 0.0;
         mesh->AddVertex(c);
         c[0] = lx; c[1] = 0.0; c[2] = 0.0;
         mesh->AddVertex(c);
         c[0] = lx; c[1] = ly; c[2] = 0.0;
         mesh->AddVertex(c);
         c[0] = 0.0; c[1] = ly; c[2] = 0.0;
         mesh->AddVertex(c);

         c[0] = 0.25 * lx; c[1] = 0.5 * ly; c[2] = 0.5 * lz;
         mesh->AddVertex(c);
         c[0] = 0.75 * lx; c[1] = 0.5 * ly; c[2] = 0.5 * lz;
         mesh->AddVertex(c);

         c[0] = 0.0; c[1] = 0.0; c[2] = lz;
         mesh->AddVertex(c);
         c[0] = lx; c[1] = 0.0; c[2] = lz;
         mesh->AddVertex(c);
         c[0] = lx; c[1] = ly; c[2] = lz;
         mesh->AddVertex(c);
         c[0] = 0.0; c[1] = ly; c[2] = lz;
         mesh->AddVertex(c);

         v[0] = 0; v[1] = 3; v[2] = 4; v[3] = 1; v[4] = 2; v[5] = 5;
         mesh->AddWedge(v);
         v[0] = 3; v[1] = 9; v[2] = 4; v[3] = 2; v[4] = 8; v[5] = 5;
         mesh->AddWedge(v);
         v[0] = 9; v[1] = 6; v[2] = 4; v[3] = 8; v[4] = 7; v[5] = 5;
         mesh->AddWedge(v);
         v[0] = 6; v[1] = 0; v[2] = 4; v[3] = 7; v[4] = 1; v[5] = 5;
         mesh->AddWedge(v);
         v[0] = 0; v[1] = 3; v[2] = 9; v[3] = 4;
         mesh->AddTet(v);
         v[0] = 0; v[1] = 9; v[2] = 6; v[3] = 4;
         mesh->AddTet(v);
         v[0] = 1; v[1] = 7; v[2] = 2; v[3] = 5;
         mesh->AddTet(v);
         v[0] = 8; v[1] = 2; v[2] = 7; v[3] = 5;
         mesh->AddTet(v);
         break;
   }
   mesh->FinalizeTopology();

   return mesh;
}

} // namespace mfem_test_fem
