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

#include "mfem.hpp"
#include "unit_tests.hpp"

using namespace mfem;

class TestMesh : public Mesh
{
public:
   using Mesh::GetTriOrientation;
   using Mesh::ComposeTriOrientations;
   using Mesh::InvertTriOrientation;
   using Mesh::GetQuadOrientation;
   using Mesh::ComposeQuadOrientations;
   using Mesh::InvertQuadOrientation;
};

void TriPerm(int i, int *v)
{
   v[0] = int((i + 1) / 2) % 3;
   v[1] = (7 - i) % 3;
   v[2] = (int(i / 2)+ 2) % 3;
}

int QuadPermGen(int i, int s)
{
   if (i % 2 == 0)
   {
      return (4 + s - i / 2) % 4;
   }
   else
   {
      return (4 - s + (i - 1)/2) % 4;
   }
}

void QuadPerm(int i, int *v)
{
   for (int j=0; j<4; j++)
   {
      v[j] = QuadPermGen(i, j);
   }
}

TEST_CASE("Face Orientation", "[FaceOrientation]")
{
   SECTION("Triangle")
   {
      const int va[3] = {0,1,2};
      int vb[3] = {0,1,2};
      int vc[3] = {0,1,2};

      for (int i=0; i<6; i++)
      {
         TriPerm(i, vb);

         int ori_a_b = TestMesh::GetTriOrientation(va, vb);
         int ori_b_a = TestMesh::GetTriOrientation(vb, va);

         int inv_ori_a_b = TestMesh::InvertTriOrientation(ori_a_b);

         REQUIRE(inv_ori_a_b == ori_b_a);

         for (int j=0; j<6; j++)
         {
            TriPerm(j, vc);

            int ori_b_c = TestMesh::GetTriOrientation(vb, vc);

            int test_ori = TestMesh::ComposeTriOrientations(ori_a_b, ori_b_c);

            int ori_a_c = TestMesh::GetTriOrientation(va, vc);

            REQUIRE(test_ori == ori_a_c);
         }
      }
   }
   SECTION("Quadrilateral")
   {
      const int va[4] = {0,1,2,3};
      int vb[4] = {0,1,2,3};
      int vc[4] = {0,1,2,3};

      for (int i=0; i<8; i++)
      {
         QuadPerm(i, vb);

         int ori_a_b = TestMesh::GetQuadOrientation(va, vb);
         int ori_b_a = TestMesh::GetQuadOrientation(vb, va);

         int inv_ori_a_b = TestMesh::InvertQuadOrientation(ori_a_b);

         REQUIRE(inv_ori_a_b == ori_b_a);

         for (int j=0; j<8; j++)
         {
            QuadPerm(j, vc);

            int ori_b_c = TestMesh::GetQuadOrientation(vb, vc);

            int test_ori = TestMesh::ComposeQuadOrientations(ori_a_b, ori_b_c);

            int ori_a_c = TestMesh::GetQuadOrientation(va, vc);

            REQUIRE(test_ori == ori_a_c);
         }
      }
   }
}

template <Geometry::Type geom_t>
constexpr Geometry::Type GetFaceType();

template <>
constexpr Geometry::Type GetFaceType<Geometry::SEGMENT>()
{
   return Geometry::POINT;
}

template <>
constexpr Geometry::Type GetFaceType<Geometry::TRIANGLE>()
{
   return Geometry::SEGMENT;
}

template <>
constexpr Geometry::Type GetFaceType<Geometry::SQUARE>()
{
   return Geometry::SEGMENT;
}

template <>
constexpr Geometry::Type GetFaceType<Geometry::TETRAHEDRON>()
{
   return Geometry::TRIANGLE;
}

template <>
constexpr Geometry::Type GetFaceType<Geometry::CUBE>()
{
   return Geometry::SQUARE;
}

TEMPLATE_TEST_CASE_SIG("Boundary Element Face Orientation", "[FaceOrientation]",
                       ((Geometry::Type geom_t), geom_t),
                       Geometry::SEGMENT, Geometry::TRIANGLE, Geometry::SQUARE,
                       Geometry::TETRAHEDRON, Geometry::CUBE)
{
   constexpr auto face_t = GetFaceType<geom_t>();
   using face_t_consts = Geometry::Constants<face_t>;

   Mesh mesh;
   constexpr int n1d = 1;
   switch (geom_t)
   {
      case Geometry::SEGMENT:
         mesh = Mesh::MakeCartesian1D(n1d, Element::SEGMENT);
         break;
      case Geometry::TRIANGLE:
         mesh = Mesh::MakeCartesian2D(n1d, n1d, Element::TRIANGLE);
         break;
      case Geometry::SQUARE:
         mesh = Mesh::MakeCartesian2D(n1d, n1d, Element::QUADRILATERAL);
         break;
      case Geometry::TETRAHEDRON:
         mesh = Mesh::MakeCartesian3D(n1d, n1d, n1d, Element::TETRAHEDRON);
         break;
      case Geometry::CUBE:
         mesh = Mesh::MakeCartesian3D(n1d, n1d, n1d, Element::HEXAHEDRON);
         break;
      default:
         MFEM_ABORT("");
   }
   Element *be0 = mesh.GetBdrElement(0);
   MFEM_VERIFY(be0->GetGeometryType() == face_t, "");
   int f, o;
   mesh.GetBdrElementFace(0, &f, &o);
   const Element *face = mesh.GetFace(f);
   int *be0_v = be0->GetVertices();
   const int *face_v = face->GetVertices();
   for (o = 0; o < face_t_consts::NumOrient; o++)
   {
      const int *face_perm = face_t_consts::Orient[o];
      for (int i = 0; i < face_t_consts::NumVert; i++)
      {
         be0_v[i] = face_v[face_perm[i]];
      }
      IsoparametricTransformation bdr_tr, face_tr;
      mesh.GetBdrElementTransformation(0, &bdr_tr);
      mesh.GetFaceTransformation(f, &face_tr);
      IntegrationPoint bdr_ip;
      bdr_ip.Set3(0.1, 0.3, 0.0);
      int inv_o;
      mesh.GetBdrElementFace(0, &f, &inv_o);
      MFEM_VERIFY(inv_o == face_t_consts::InvOrient[o], "");
      IntegrationPoint face_ip = Mesh::TransformBdrElementToFace(
                                    be0->GetGeometryType(), inv_o, bdr_ip);
      Vector bdr_pt, face_pt;
      bdr_tr.Transform(bdr_ip, bdr_pt);
      face_tr.Transform(face_ip, face_pt);
      REQUIRE(bdr_pt.DistanceTo(face_pt) == MFEM_Approx(0.0));
   }
}
