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

#include "make_permuted_mesh.hpp"

namespace mfem
{

Mesh Mesh2D_Orientation(int face_perm_1, int face_perm_2)
{
   static const int dim = 2;
   static const int nv = 6;
   static const int nel = 2;
   Mesh mesh(dim, nv, nel);
   real_t x[dim];
   x[0] = 0.0;   x[1] = 0.0;
   mesh.AddVertex(x);
   x[0] = 1.0;   x[1] = 0.0;
   mesh.AddVertex(x);
   x[0] = 2.0;   x[1] = 0.0;
   mesh.AddVertex(x);
   x[0] = 0.0;   x[1] = 1.0;
   mesh.AddVertex(x);
   x[0] = 1.0;   x[1] = 1.0;
   mesh.AddVertex(x);
   x[0] = 2.0;   x[1] = 1.0;
   mesh.AddVertex(x);
   int el[4];
   el[0] = 0;
   el[1] = 1;
   el[2] = 4;
   el[3] = 3;
   std::rotate(&el[0], &el[face_perm_1], &el[3] + 1);

   mesh.AddQuad(el);

   el[0] = 1;
   el[1] = 2;
   el[2] = 5;
   el[3] = 4;
   std::rotate(&el[0], &el[face_perm_2], &el[3] + 1);
   mesh.AddQuad(el);

   mesh.FinalizeQuadMesh(true);
   mesh.GenerateBoundaryElements();
   mesh.Finalize();
   return mesh;
}

void Rotation3DVertices(int *v, int ref_face, int rot)
{
   std::vector<int> face_1, face_2;

   switch (ref_face/2)
   {
      case 0:
         face_1 = {v[0], v[1], v[2], v[3]};
         face_2 = {v[4], v[5], v[6], v[7]};
         break;
      case 1:
         face_1 = {v[1], v[5], v[6], v[2]};
         face_2 = {v[0], v[4], v[7], v[3]};
         break;
      case 2:
         face_1 = {v[4], v[5], v[1], v[0]};
         face_2 = {v[7], v[6], v[2], v[3]};
         break;
   }
   if (ref_face % 2 == 0)
   {
      std::reverse(face_1.begin(), face_1.end());
      std::reverse(face_2.begin(), face_2.end());
      std::swap(face_1, face_2);
   }

   std::rotate(face_1.begin(), face_1.begin() + rot, face_1.end());
   std::rotate(face_2.begin(), face_2.begin() + rot, face_2.end());

   for (int i=0; i<4; ++i)
   {
      v[i] = face_1[i];
      v[i+4] = face_2[i];
   }
}

Mesh Mesh3D_Orientation(int face_perm_1, int face_perm_2)
{
   static const int dim = 3;
   static const int nv = 12;
   static const int nel = 2;
   Mesh mesh(dim, nv, nel);
   real_t x[dim];
   x[0] = 0.0;   x[1] = 0.0;   x[2] = 0.0;
   mesh.AddVertex(x);
   x[0] = 1.0;   x[1] = 0.0;   x[2] = 0.0;
   mesh.AddVertex(x);
   x[0] = 2.0;   x[1] = 0.0;   x[2] = 0.0;
   mesh.AddVertex(x);
   x[0] = 0.0;   x[1] = 1.0;   x[2] = 0.0;
   mesh.AddVertex(x);
   x[0] = 1.0;   x[1] = 1.0;   x[2] = 0.0;
   mesh.AddVertex(x);
   x[0] = 2.0;   x[1] = 1.0;   x[2] = 0.0;
   mesh.AddVertex(x);
   x[0] = 0.0;   x[1] = 0.0;   x[2] = 1.0;
   mesh.AddVertex(x);
   x[0] = 1.0;   x[1] = 0.0;   x[2] = 1.0;
   mesh.AddVertex(x);
   x[0] = 2.0;   x[1] = 0.0;   x[2] = 1.0;
   mesh.AddVertex(x);
   x[0] = 0.0;   x[1] = 1.0;   x[2] = 1.0;
   mesh.AddVertex(x);
   x[0] = 1.0;   x[1] = 1.0;   x[2] = 1.0;
   mesh.AddVertex(x);
   x[0] = 2.0;   x[1] = 1.0;   x[2] = 1.0;
   mesh.AddVertex(x);

   int el[8];

   el[0] = 0;
   el[1] = 1;
   el[2] = 4;
   el[3] = 3;
   el[4] = 6;
   el[5] = 7;
   el[6] = 10;
   el[7] = 9;
   Rotation3DVertices(el, face_perm_1/4, face_perm_1%4);
   mesh.AddHex(el);

   el[0] = 1;
   el[1] = 2;
   el[2] = 5;
   el[3] = 4;
   el[4] = 7;
   el[5] = 8;
   el[6] = 11;
   el[7] = 10;
   Rotation3DVertices(el, face_perm_2/4, face_perm_2%4);
   mesh.AddHex(el);

   mesh.FinalizeHexMesh(true);
   mesh.Finalize();
   return mesh;
}

Mesh MeshOrientation(int dim, int o1, int o2)
{
   if (dim == 2) { return Mesh2D_Orientation(o1, o2); }
   else if (dim == 3) { return Mesh3D_Orientation(o1, o2); }
   else { MFEM_ABORT("Unsupported dimension."); }
}

}
