// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.googlecode.com.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.


#include "mesh_headers.hpp"

namespace mfem
{

const int Tesseract::edges[32][2] =
{
   {0, 1}, {1, 2}, {3, 2}, {0, 3},
   {4, 5}, {5, 6}, {7, 6}, {4, 7},
   {0, 4}, {1, 5}, {2, 6}, {3, 7},
   {8, 9}, {9, 10}, {11, 10}, {8, 11},
   {12, 13}, {13, 14}, {15, 14}, {12, 15},
   {8, 12}, {9, 13}, {10, 14}, {11, 15},
   {0, 8}, {1, 9}, {2, 10}, {3, 11},
   {4, 12}, {5, 13}, {6, 14}, {7, 15}
};

// same as Mesh::hex_faces
const int Tesseract::faces[8][8] =
{
   // {8,11,12,15,0,3,4,7},   //x bottom
   // {1,2,6,5,9,10,14,13},   //x top
   // {0,1,5,4,8,9,13,12},    //y bottom
   // {2,3,7,6,10,11,15,14},  //y top
   // {8,9,10,11,0,1,2,3},    // z bottom
   // {4,5,6,7,12,13,14,15},  //z top
   // {0,1,2,3,4,5,6,7},      //t botom
   // {12,13,14,15,8,9,10,11} //t top
   {8,11,15,12,0,3,7,4},   //x bottom
   {1,2,6,5,9,10,14,13},   //x top
   {0,1,5,4,8,9,13,12},    //y bottom
   {2,3,7,6,10,11,15,14},  //y top
   {8,9,10,11,0,1,2,3},    // z bottom
   {4,5,6,7,12,13,14,15},  //z top
   {0,1,2,3,4,5,6,7},      //t botom
   {12,13,14,15,8,9,10,11} //t top
};


Tesseract::Tesseract(const int *ind, int attr)
   : Element(Geometry::TESSERACT)
{
   attribute = attr;
   for (int i = 0; i < 16; i++)
   {
      indices[i] = ind[i];
   }
}

Tesseract::Tesseract(int ind1, int ind2, int ind3, int ind4,
                     int ind5, int ind6, int ind7, int ind8,
                     int ind9, int ind10, int ind11, int ind12,
                     int ind13, int ind14, int ind15, int ind16,
                     int attr) : Element(Geometry::TESSERACT)
{
   attribute  = attr;
   indices[0] = ind1;
   indices[1] = ind2;
   indices[2] = ind3;
   indices[3] = ind4;
   indices[4] = ind5;
   indices[5] = ind6;
   indices[6] = ind7;
   indices[7] = ind8;
   indices[8] = ind9;
   indices[9] = ind10;
   indices[10] = ind11;
   indices[11] = ind12;
   indices[12] = ind13;
   indices[13] = ind14;
   indices[14] = ind15;
   indices[15] = ind16;
}

void Tesseract::GetVertices(Array<int> &v) const
{
   v.SetSize(16);
   for (int i = 0; i < 16; i++)
   {
      v[i] = indices[i];
   }
}

QuadLinear4DFiniteElement TesseractFE;

}
