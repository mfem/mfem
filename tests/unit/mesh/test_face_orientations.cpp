// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
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
using namespace mfem;

#include "unit_tests.hpp"

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
