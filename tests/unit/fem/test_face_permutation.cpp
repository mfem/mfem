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

#include "mfem.hpp"
#include "unit_tests.hpp"

using namespace mfem;

Mesh *mesh_2d_orientation(int face_perm_1, int face_perm_2)
{
   static const int dim = 2;
   static const int nv = 6;
   static const int nel = 2;
   Mesh *mesh = new Mesh(dim, nv, nel);
   double x[dim];
   x[0] = 0.0;   x[1] = 0.0;
   mesh->AddVertex(x);
   x[0] = 1.0;   x[1] = 0.0;
   mesh->AddVertex(x);
   x[0] = 2.0;   x[1] = 0.0;
   mesh->AddVertex(x);
   x[0] = 0.0;   x[1] = 1.0;
   mesh->AddVertex(x);
   x[0] = 1.0;   x[1] = 1.0;
   mesh->AddVertex(x);
   x[0] = 2.0;   x[1] = 1.0;
   mesh->AddVertex(x);
   int el[4];
   el[0] = 0;
   el[1] = 1;
   el[2] = 4;
   el[3] = 3;
   std::rotate(&el[0], &el[face_perm_1], &el[3] + 1);

   mesh->AddQuad(el);

   el[0] = 1;
   el[1] = 2;
   el[2] = 5;
   el[3] = 4;
   std::rotate(&el[0], &el[face_perm_2], &el[3] + 1);
   mesh->AddQuad(el);

   mesh->FinalizeQuadMesh(true);
   mesh->GenerateBoundaryElements();
   mesh->Finalize();
   return mesh;
}

void rotate_3d_vertices(int *v, int ref_face, int rot)
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

Mesh *mesh_3d_orientation(int face_perm_1, int face_perm_2)
{
   static const int dim = 3;
   static const int nv = 12;
   static const int nel = 2;
   Mesh *mesh = new Mesh(dim, nv, nel);
   double x[dim];
   x[0] = 0.0;   x[1] = 0.0;   x[2] = 0.0;
   mesh->AddVertex(x);
   x[0] = 1.0;   x[1] = 0.0;   x[2] = 0.0;
   mesh->AddVertex(x);
   x[0] = 2.0;   x[1] = 0.0;   x[2] = 0.0;
   mesh->AddVertex(x);
   x[0] = 0.0;   x[1] = 1.0;   x[2] = 0.0;
   mesh->AddVertex(x);
   x[0] = 1.0;   x[1] = 1.0;   x[2] = 0.0;
   mesh->AddVertex(x);
   x[0] = 2.0;   x[1] = 1.0;   x[2] = 0.0;
   mesh->AddVertex(x);
   x[0] = 0.0;   x[1] = 0.0;   x[2] = 1.0;
   mesh->AddVertex(x);
   x[0] = 1.0;   x[1] = 0.0;   x[2] = 1.0;
   mesh->AddVertex(x);
   x[0] = 2.0;   x[1] = 0.0;   x[2] = 1.0;
   mesh->AddVertex(x);
   x[0] = 0.0;   x[1] = 1.0;   x[2] = 1.0;
   mesh->AddVertex(x);
   x[0] = 1.0;   x[1] = 1.0;   x[2] = 1.0;
   mesh->AddVertex(x);
   x[0] = 3.0;   x[1] = 1.0;   x[2] = 1.0;
   mesh->AddVertex(x);

   int el[8];

   el[0] = 0;
   el[1] = 1;
   el[2] = 4;
   el[3] = 3;
   el[4] = 6;
   el[5] = 7;
   el[6] = 10;
   el[7] = 9;
   rotate_3d_vertices(el, face_perm_1/4, face_perm_1%4);
   mesh->AddHex(el);

   el[0] = 1;
   el[1] = 2;
   el[2] = 5;
   el[3] = 4;
   el[4] = 7;
   el[5] = 8;
   el[6] = 11;
   el[7] = 10;
   rotate_3d_vertices(el, face_perm_2/4, face_perm_2%4);
   mesh->AddHex(el);

   mesh->FinalizeHexMesh(true);
   mesh->GenerateBoundaryElements();
   mesh->Finalize();
   return mesh;
}

double x_fn(const Vector &xvec) { return xvec[0]; }
double y_fn(const Vector &xvec) { return xvec[1]; }
double z_fn(const Vector &xvec) { return xvec[2]; }

double TestFaceRestriction(Mesh &mesh, int order)
{
   int dim = mesh.Dimension();
   DG_FECollection fec(order, dim, BasisType::GaussLobatto);
   FiniteElementSpace fes(&mesh, &fec);
   GridFunction gf(&fes);

   L2FaceRestriction restr(fes, ElementDofOrdering::LEXICOGRAPHIC,
                           FaceType::Interior, L2FaceValues::DoubleValued);

   const int ndof_face = pow(order+1, dim-1);

   Vector face_values(ndof_face*2);

   double max_err = 0.0;
   for (int d=0; d<dim; ++d)
   {
      double (*fn)(const Vector &);
      if (d == 0)
      {
         fn = x_fn;
      }
      else if (d == 1)
      {
         fn = y_fn;
      }
      else if (d == 2)
      {
         fn = z_fn;
      }
      else
      {
         MFEM_ABORT("Bad dimension");
         return infinity();
      }
      FunctionCoefficient coeff(fn);
      gf.ProjectCoefficient(coeff);
      restr.Mult(gf, face_values);

      for (int i=0; i<ndof_face; ++i)
      {
         double err = std::abs(face_values(i) - face_values(i + ndof_face));
         max_err = std::max(max_err, err);
      }
   }
   return max_err;
}

TEST_CASE("2D Face Permutation", "[Face Permutation]")
{
   int order = 3;
   double max_err = 0.0;
   for (int fp2=0; fp2<4; ++fp2)
   {
      for (int fp1=0; fp1<4; ++fp1)
      {
         Mesh *mesh = mesh_2d_orientation(fp1, fp2);
         double err = TestFaceRestriction(*mesh, order);
         max_err = std::max(max_err, err);
         delete mesh;
      }
   }
   std::cout << "2D Face Permutation: max_err = " << max_err << '\n';
   REQUIRE(max_err < 1e-15);
}

TEST_CASE("3D Face Permutation", "[Face Permutation]")
{
   int order = 3;
   double max_err = 0.0;
   for (int fp2=0; fp2<24; ++fp2)
   {
      for (int fp1=0; fp1<24; ++fp1)
      {
         Mesh *mesh = mesh_3d_orientation(fp1, fp2);
         double err = TestFaceRestriction(*mesh, order);
         max_err = std::max(max_err, err);
         delete mesh;
      }
   }
   std::cout << "3D Face Permutation: max_err = " << max_err << '\n';
   REQUIRE(max_err < 1e-15);
}
