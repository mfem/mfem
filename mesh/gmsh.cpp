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

#include "gmsh.hpp"
#include "vtk.hpp"

namespace mfem
{

void GmshHOSegmentMapping(int order, int *map)
{
   map[0] = 0;
   map[order] = 1;
   for (int i=1; i<order; i++)
   {
      map[i] = i + 1;
   }
}

void GmshHOTriangleMapping(int order, int *map)
{
   int b[3];
   int o = 0;
   for (b[1]=0; b[1]<=order; ++b[1])
   {
      for (b[0]=0; b[0]<=order-b[1]; ++b[0])
      {
         b[2] = order - b[0] - b[1];
         int o_gmsh =  BarycentricToVTKTriangle(b, order);
         map[o] = o_gmsh;
         o++;
      }
   }
}

void GmshHOQuadrilateralMapping(int order, int *map)
{}

void GmshHOTetrahedronMapping(int order, int *map)
{}

void GmshHOHexahedronMapping(int order, int *map)
{}

void GmshHOWedgeMapping(int order, int *map)
{}

void GmshHOPyramidMapping(int order, int *map)
{}

} // namespace mfem
