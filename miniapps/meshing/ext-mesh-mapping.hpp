// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

// MFEM External Mesh Mapping Miniapp

#include <array>

// A very simple example of an external mesh class with a baby mesh in it.
//The element and vertex ids are as follows:
// 6-----7-----8
// |     |     |
// |  3  |  4  |
// |     |     |
// 3-----4-----5
// |  1  |     |
// 9----10  2  |
// |  0  |     |
// 0-----1-----2
class DummyMesh
{
   class Vertex
   {
   public:
      Vertex(): x(0.0), y(0.0) {}

      void Set(double x_, double y_)
      {
         x = x_;
         y = y_;
      }

      double x;
      double y;
   };

   class Element
   {
   public:
      Element() {}

      void Set(int id0, int id1, int id2, int id3)
      {
         vertex_ids[0] = id0; vertex_ids[1] = id1;
         vertex_ids[2] = id2; vertex_ids[3] = id3;
      }

      const int num_vertices = 4;
      std::array<int,4> vertex_ids;
   };

   class BElement
   {
   public:
      BElement() {}

      void Set(int id0, int id1)
      {
         vertex_ids[0] = id0; vertex_ids[1] = id1;
      }

      const int num_vertices = 2;
      std::array<int,2> vertex_ids;
   };

public:
   DummyMesh() : num_vertices(11), num_elements(5), num_belements(9),
      num_vparents(2)
   {
      //Vertices
      V[6].Set(0.0, 2.0);  V[7].Set(1.0, 2.0);  V[8].Set(2.0, 2.0);
      V[3].Set(0.0, 1.0);  V[4].Set(1.0, 1.0);  V[5].Set(2.0, 1.0);
      V[0].Set(0.0, 0.0);  V[1].Set(1.0, 0.0);  V[2].Set(2.0, 0.0);

      V[9].Set(0.0, 0.5);  V[10].Set(1.0, 0.5);

      // Elements in this dummy format have their vertex indices listed
      // lexographic order rather than going around the element as is normal
      // in MFEM.
      E[0].Set(0,1,9,10);
      E[1].Set(9,10,3,4);
      E[2].Set(1,2,4,5);
      E[3].Set(3,4,6,7);
      E[4].Set(4,5,7,8);

      //Boundary Elements
      //Bottom
      B[0].Set(0,1);       B[1].Set(1,2);
      //Top
      B[2].Set(6,7);       B[3].Set(7,8);
      //Left
      B[4].Set(0,9);       B[5].Set(9,3);       B[6].Set(3,6);
      //Right
      B[7].Set(2,5);       B[8].Set(5,8);

      //Set the vertex parents
      //In the future replace this with element parents and demonstrate
      //computation of vertex parents
      VP[0] = std::make_tuple(9,0,3);
      VP[1] = std::make_tuple(10,1,4);
   }

   const int num_vertices;
   const int num_elements;
   const int num_belements;
   const int num_vparents;
   Vertex V[11];                    //Mesh vertices
   Element E[5];                    //Mesh elements
   BElement B[9];                   //Mesh boundary elements
   std::tuple<int,int,int> VP[2];   //Vertex parents (vid, parent1id, parent2id)
};
