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

#include "fem.hpp"
#include <cstdlib>
#include <cstring>
#include <cstdio>

int FiniteElementCollection::HasFaceDofs(int GeomType) const
{
   switch (GeomType)
   {
   case Geometry::TETRAHEDRON: return DofForGeometry (Geometry::TRIANGLE);
   case Geometry::CUBE:        return DofForGeometry (Geometry::SQUARE);
   default:
      mfem_error ("FiniteElementCollection::HasFaceDofs:"
                  " unknown geometry type.");
   }
   return 0;
}

FiniteElementCollection *FiniteElementCollection::New(const char *name)
{
   FiniteElementCollection *fec = NULL;

   if (!strcmp(name, "Linear"))
      fec = new LinearFECollection;
   else if (!strcmp(name, "Quadratic"))
      fec = new QuadraticFECollection;
   else if (!strcmp(name, "QuadraticPos"))
      fec = new QuadraticPosFECollection;
   else if (!strcmp(name, "Cubic"))
      fec = new CubicFECollection;
   else if (!strcmp(name, "Const3D"))
      fec = new Const3DFECollection;
   else if (!strcmp(name, "Const2D"))
      fec = new Const2DFECollection;
   else if (!strcmp(name, "LinearDiscont2D"))
      fec = new LinearDiscont2DFECollection;
   else if (!strcmp(name, "GaussLinearDiscont2D"))
      fec = new GaussLinearDiscont2DFECollection;
   else if (!strcmp(name, "P1OnQuad"))
      fec = new P1OnQuadFECollection;
   else if (!strcmp(name, "QuadraticDiscont2D"))
      fec = new QuadraticDiscont2DFECollection;
   else if (!strcmp(name, "QuadraticPosDiscont2D"))
      fec = new QuadraticPosDiscont2DFECollection;
   else if (!strcmp(name, "GaussQuadraticDiscont2D"))
      fec = new GaussQuadraticDiscont2DFECollection;
   else if (!strcmp(name, "CubicDiscont2D"))
      fec = new CubicDiscont2DFECollection;
   else if (!strcmp(name, "LinearDiscont3D"))
      fec = new LinearDiscont3DFECollection;
   else if (!strcmp(name, "QuadraticDiscont3D"))
      fec = new QuadraticDiscont3DFECollection;
   else if (!strcmp(name, "LinearNonConf3D"))
      fec = new LinearNonConf3DFECollection;
   else if (!strcmp(name, "CrouzeixRaviart"))
      fec = new CrouzeixRaviartFECollection;
   else if (!strcmp(name, "ND1_3D"))
      fec = new ND1_3DFECollection;
   else if (!strcmp(name, "RT0_2D"))
      fec = new RT0_2DFECollection;
   else if (!strcmp(name, "RT1_2D"))
      fec = new RT1_2DFECollection;
   else if (!strcmp(name, "RT2_2D"))
      fec = new RT2_2DFECollection;
   else if (!strcmp(name, "RT0_3D"))
      fec = new RT0_3DFECollection;
   else if (!strcmp(name, "RT1_3D"))
      fec = new RT1_3DFECollection;
   else if (!strncmp(name, "H1_", 3))
      fec = new H1_FECollection(atoi(name + 7), atoi(name + 3));
   else if (!strncmp(name, "L2_", 3))
      fec = new L2_FECollection(atoi(name + 7), atoi(name + 3));
   else if (!strncmp(name, "RT_", 3))
      fec = new RT_FECollection(atoi(name + 7), atoi(name + 3));
   else if (!strncmp(name, "ND_", 3))
      fec = new ND_FECollection(atoi(name + 7), atoi(name + 3));
   else if (!strncmp(name, "Local_", 6))
      fec = new Local_FECollection(name + 6);
   else if (!strncmp(name, "NURBS", 5))
      fec = new NURBSFECollection(atoi(name + 5));
   else
      mfem_error ("FiniteElementCollection::New : "
                  "Unknown FiniteElementCollection!");

   return fec;
}

const FiniteElement *
LinearFECollection::FiniteElementForGeometry(int GeomType) const
{
   switch (GeomType)
   {
   case Geometry::POINT:       return &PointFE;
   case Geometry::SEGMENT:     return &SegmentFE;
   case Geometry::TRIANGLE:    return &TriangleFE;
   case Geometry::SQUARE:      return &QuadrilateralFE;
   case Geometry::TETRAHEDRON: return &TetrahedronFE;
   case Geometry::CUBE:        return &ParallelepipedFE;
   default:
      mfem_error ("LinearFECollection: unknown geometry type.");
   }
   return &SegmentFE; // Make some compilers happy
}

int LinearFECollection::DofForGeometry(int GeomType) const
{
   switch (GeomType)
   {
   case Geometry::POINT:       return 1;
   case Geometry::SEGMENT:     return 0;
   case Geometry::TRIANGLE:    return 0;
   case Geometry::SQUARE:      return 0;
   case Geometry::TETRAHEDRON: return 0;
   case Geometry::CUBE:        return 0;
   default:
      mfem_error ("LinearFECollection: unknown geometry type.");
   }
   return 0; // Make some compilers happy
}

int * LinearFECollection::DofOrderForOrientation(int GeomType, int Or) const
{
   return NULL;
}


const FiniteElement *
QuadraticFECollection::FiniteElementForGeometry(int GeomType) const
{
   switch (GeomType)
   {
   case Geometry::POINT:       return &PointFE;
   case Geometry::SEGMENT:     return &SegmentFE;
   case Geometry::TRIANGLE:    return &TriangleFE;
   case Geometry::SQUARE:      return &QuadrilateralFE;
   case Geometry::TETRAHEDRON: return &TetrahedronFE;
   case Geometry::CUBE:        return &ParallelepipedFE;
   default:
      mfem_error ("QuadraticFECollection: unknown geometry type.");
   }
   return &SegmentFE; // Make some compilers happy
}

int QuadraticFECollection::DofForGeometry(int GeomType) const
{
   switch (GeomType)
   {
   case Geometry::POINT:       return 1;
   case Geometry::SEGMENT:     return 1;
   case Geometry::TRIANGLE:    return 0;
   case Geometry::SQUARE:      return 1;
   case Geometry::TETRAHEDRON: return 0;
   case Geometry::CUBE:        return 1;
   default:
      mfem_error ("QuadraticFECollection: unknown geometry type.");
   }
   return 0; // Make some compilers happy
}

int * QuadraticFECollection::DofOrderForOrientation(int GeomType, int Or) const
{
   static int indexes[] = { 0 };

   return indexes;
}


const FiniteElement *
QuadraticPosFECollection::FiniteElementForGeometry(int GeomType) const
{
   switch (GeomType)
   {
   case Geometry::SEGMENT:     return &SegmentFE;
   case Geometry::SQUARE:      return &QuadrilateralFE;
   default:
      mfem_error ("QuadraticPosFECollection: unknown geometry type.");
   }
   return NULL; // Make some compilers happy
}

int QuadraticPosFECollection::DofForGeometry(int GeomType) const
{
   switch (GeomType)
   {
   case Geometry::POINT:       return 1;
   case Geometry::SEGMENT:     return 1;
   case Geometry::SQUARE:      return 1;
   default:
      mfem_error ("QuadraticPosFECollection: unknown geometry type.");
   }
   return 0; // Make some compilers happy
}

int * QuadraticPosFECollection::DofOrderForOrientation(int GeomType, int Or)
   const
{
   static int indexes[] = { 0 };

   return indexes;
}


const FiniteElement *
CubicFECollection::FiniteElementForGeometry(int GeomType) const
{
   switch (GeomType)
   {
   case Geometry::POINT:       return &PointFE;
   case Geometry::SEGMENT:     return &SegmentFE;
   case Geometry::TRIANGLE:    return &TriangleFE;
   case Geometry::SQUARE:      return &QuadrilateralFE;
   case Geometry::TETRAHEDRON: return &TetrahedronFE;
   case Geometry::CUBE:        return &ParallelepipedFE;
   default:
      mfem_error ("CubicFECollection: unknown geometry type.");
   }
   return &SegmentFE; // Make some compilers happy
}

int CubicFECollection::DofForGeometry(int GeomType) const
{
   switch (GeomType)
   {
   case Geometry::POINT:       return 1;
   case Geometry::SEGMENT:     return 2;
   case Geometry::TRIANGLE:    return 1;
   case Geometry::SQUARE:      return 4;
   case Geometry::TETRAHEDRON: return 0;
   case Geometry::CUBE:        return 8;
   default:
      mfem_error ("CubicFECollection: unknown geometry type.");
   }
   return 0; // Make some compilers happy
}

int * CubicFECollection::DofOrderForOrientation(int GeomType, int Or) const
{
   if (GeomType == Geometry::SEGMENT)
   {
      static int ind_pos[] = { 0, 1 };
      static int ind_neg[] = { 1, 0 };

      if (Or < 0)
         return ind_neg;
      return ind_pos;
   }
   else if (GeomType == Geometry::TRIANGLE)
   {
      static int indexes[] = { 0 };

      return indexes;
   }
   else if (GeomType == Geometry::SQUARE)
   {
      static int sq_ind[8][4] = {{0, 1, 2, 3}, {0, 2, 1, 3},
                                 {2, 0, 3, 1}, {1, 0, 3, 2},
                                 {3, 2, 1, 0}, {3, 1, 2, 0},
                                 {1, 3, 0, 2}, {2, 3, 0, 1}};
      return sq_ind[Or];
   }

   return NULL;
}


const FiniteElement *
CrouzeixRaviartFECollection::FiniteElementForGeometry(int GeomType) const
{
   switch (GeomType)
   {
   case Geometry::SEGMENT:     return &SegmentFE;
   case Geometry::TRIANGLE:    return &TriangleFE;
   case Geometry::SQUARE:      return &QuadrilateralFE;
   default:
      mfem_error ("CrouzeixRaviartFECollection: unknown geometry type.");
   }
   return &SegmentFE; // Make some compilers happy
}

int CrouzeixRaviartFECollection::DofForGeometry(int GeomType) const
{
   switch (GeomType)
   {
   case Geometry::POINT:       return 0;
   case Geometry::SEGMENT:     return 1;
   case Geometry::TRIANGLE:    return 0;
   case Geometry::SQUARE:      return 0;
   default:
      mfem_error ("CrouzeixRaviartFECollection: unknown geometry type.");
   }
   return 0; // Make some compilers happy
}

int * CrouzeixRaviartFECollection::DofOrderForOrientation(int GeomType, int Or)
   const
{
   static int indexes[] = { 0 };

   return indexes;
}


const FiniteElement *
RT0_2DFECollection::FiniteElementForGeometry(int GeomType) const
{
   switch (GeomType)
   {
   case Geometry::SEGMENT:     return &SegmentFE;
   case Geometry::TRIANGLE:    return &TriangleFE;
   case Geometry::SQUARE:      return &QuadrilateralFE;
   default:
      mfem_error ("RT0_2DFECollection: unknown geometry type.");
   }
   return &SegmentFE; // Make some compilers happy
}

int RT0_2DFECollection::DofForGeometry(int GeomType) const
{
   switch (GeomType)
   {
   case Geometry::POINT:       return 0;
   case Geometry::SEGMENT:     return 1;
   case Geometry::TRIANGLE:    return 0;
   case Geometry::SQUARE:      return 0;
   default:
      mfem_error ("RT0_2DFECollection: unknown geometry type.");
   }
   return 0; // Make some compilers happy
}

int * RT0_2DFECollection::DofOrderForOrientation(int GeomType, int Or)
   const
{
   static int ind_pos[] = { 0 };
   static int ind_neg[] = { -1 };

   if (Or > 0)
      return ind_pos;
   return ind_neg;
}


const FiniteElement *
RT1_2DFECollection::FiniteElementForGeometry(int GeomType) const
{
   switch (GeomType)
   {
   case Geometry::SEGMENT:     return &SegmentFE;
   case Geometry::TRIANGLE:    return &TriangleFE;
   case Geometry::SQUARE:      return &QuadrilateralFE;
   default:
      mfem_error ("RT1_2DFECollection: unknown geometry type.");
   }
   return &SegmentFE; // Make some compilers happy
}

int RT1_2DFECollection::DofForGeometry(int GeomType) const
{
   switch (GeomType)
   {
   case Geometry::POINT:       return 0;
   case Geometry::SEGMENT:     return 2;
   case Geometry::TRIANGLE:    return 2;
   case Geometry::SQUARE:      return 4;
   default:
      mfem_error ("RT1_2DFECollection: unknown geometry type.");
   }
   return 0; // Make some compilers happy
}

int * RT1_2DFECollection::DofOrderForOrientation(int GeomType, int Or)
   const
{
   static int ind_pos[] = {  0,  1 };
   static int ind_neg[] = { -2, -1 };

   if (Or > 0)
      return ind_pos;
   return ind_neg;
}

const FiniteElement *
RT2_2DFECollection::FiniteElementForGeometry(int GeomType) const
{
   switch (GeomType)
   {
   case Geometry::SEGMENT:     return &SegmentFE;
   case Geometry::TRIANGLE:    return &TriangleFE;
   case Geometry::SQUARE:      return &QuadrilateralFE;
   default:
      mfem_error ("RT2_2DFECollection: unknown geometry type.");
   }
   return &SegmentFE; // Make some compilers happy
}

int RT2_2DFECollection::DofForGeometry(int GeomType) const
{
   switch (GeomType)
   {
   case Geometry::POINT:       return 0;
   case Geometry::SEGMENT:     return 3;
   case Geometry::TRIANGLE:    return 6;
   case Geometry::SQUARE:      return 12;
   default:
      mfem_error ("RT2_2DFECollection: unknown geometry type.");
   }
   return 0; // Make some compilers happy
}

int * RT2_2DFECollection::DofOrderForOrientation(int GeomType, int Or)
   const
{
   static int ind_pos[] = { 0, 1, 2 };
   static int ind_neg[] = { -3, -2, -1 };

   if (Or > 0)
      return ind_pos;
   return ind_neg;
}


const FiniteElement *
Const2DFECollection::FiniteElementForGeometry(int GeomType) const
{
   switch (GeomType)
   {
   case Geometry::TRIANGLE:    return &TriangleFE;
   case Geometry::SQUARE:      return &QuadrilateralFE;
   default:
      mfem_error ("Const2DFECollection: unknown geometry type.");
   }
   return &TriangleFE; // Make some compilers happy
}

int Const2DFECollection::DofForGeometry(int GeomType) const
{
   switch (GeomType)
   {
   case Geometry::POINT:       return 0;
   case Geometry::SEGMENT:     return 0;
   case Geometry::TRIANGLE:    return 1;
   case Geometry::SQUARE:      return 1;
   default:
      mfem_error ("Const2DFECollection: unknown geometry type.");
   }
   return 0; // Make some compilers happy
}

int * Const2DFECollection::DofOrderForOrientation(int GeomType, int Or)
   const
{
   return NULL;
}


const FiniteElement *
LinearDiscont2DFECollection::FiniteElementForGeometry(int GeomType) const
{
   switch (GeomType)
   {
   case Geometry::TRIANGLE:    return &TriangleFE;
   case Geometry::SQUARE:      return &QuadrilateralFE;
   default:
      mfem_error ("LinearDiscont2DFECollection: unknown geometry type.");
   }
   return &TriangleFE; // Make some compilers happy
}

int LinearDiscont2DFECollection::DofForGeometry(int GeomType) const
{
   switch (GeomType)
   {
   case Geometry::POINT:       return 0;
   case Geometry::SEGMENT:     return 0;
   case Geometry::TRIANGLE:    return 3;
   case Geometry::SQUARE:      return 4;
   default:
      mfem_error ("LinearDiscont2DFECollection: unknown geometry type.");
   }
   return 0; // Make some compilers happy
}

int * LinearDiscont2DFECollection::DofOrderForOrientation(int GeomType, int Or)
   const
{
   return NULL;
}


const FiniteElement *
GaussLinearDiscont2DFECollection::FiniteElementForGeometry(int GeomType) const
{
   switch (GeomType)
   {
   case Geometry::TRIANGLE:    return &TriangleFE;
   case Geometry::SQUARE:      return &QuadrilateralFE;
   default:
      mfem_error ("GaussLinearDiscont2DFECollection:"
                  " unknown geometry type.");
   }
   return &TriangleFE; // Make some compilers happy
}

int GaussLinearDiscont2DFECollection::DofForGeometry(int GeomType) const
{
   switch (GeomType)
   {
   case Geometry::POINT:       return 0;
   case Geometry::SEGMENT:     return 0;
   case Geometry::TRIANGLE:    return 3;
   case Geometry::SQUARE:      return 4;
   default:
      mfem_error ("GaussLinearDiscont2DFECollection:"
                  " unknown geometry type.");
   }
   return 0; // Make some compilers happy
}

int * GaussLinearDiscont2DFECollection::DofOrderForOrientation(
   int GeomType, int Or) const
{
   return NULL;
}


const FiniteElement *
P1OnQuadFECollection::FiniteElementForGeometry(int GeomType) const
{
   if (GeomType != Geometry::SQUARE)
   {
      mfem_error ("P1OnQuadFECollection: unknown geometry type.");
   }
   return &QuadrilateralFE;
}

int P1OnQuadFECollection::DofForGeometry(int GeomType) const
{
   switch (GeomType)
   {
   case Geometry::POINT:       return 0;
   case Geometry::SEGMENT:     return 0;
   case Geometry::SQUARE:      return 3;
   default:
      mfem_error ("P1OnQuadFECollection: unknown geometry type.");
   }
   return 0; // Make some compilers happy
}

int * P1OnQuadFECollection::DofOrderForOrientation(
   int GeomType, int Or) const
{
   return NULL;
}


const FiniteElement *
QuadraticDiscont2DFECollection::FiniteElementForGeometry(int GeomType) const
{
   switch (GeomType)
   {
   case Geometry::TRIANGLE:    return &TriangleFE;
   case Geometry::SQUARE:      return &QuadrilateralFE;
   default:
      mfem_error ("QuadraticDiscont2DFECollection: unknown geometry type.");
   }
   return &TriangleFE; // Make some compilers happy
}

int QuadraticDiscont2DFECollection::DofForGeometry(int GeomType) const
{
   switch (GeomType)
   {
   case Geometry::POINT:       return 0;
   case Geometry::SEGMENT:     return 0;
   case Geometry::TRIANGLE:    return 6;
   case Geometry::SQUARE:      return 9;
   default:
      mfem_error ("QuadraticDiscont2DFECollection: unknown geometry type.");
   }
   return 0; // Make some compilers happy
}

int * QuadraticDiscont2DFECollection::DofOrderForOrientation(
   int GeomType, int Or) const
{
   return NULL;
}


const FiniteElement *
QuadraticPosDiscont2DFECollection::FiniteElementForGeometry(int GeomType) const
{
   switch (GeomType)
   {
   case Geometry::SQUARE:  return &QuadrilateralFE;
   default:
      mfem_error ("QuadraticPosDiscont2DFECollection: unknown geometry type.");
   }
   return NULL; // Make some compilers happy
}

int QuadraticPosDiscont2DFECollection::DofForGeometry(int GeomType) const
{
   switch (GeomType)
   {
   case Geometry::POINT:       return 0;
   case Geometry::SEGMENT:     return 0;
   case Geometry::SQUARE:      return 9;
   default:
      mfem_error ("QuadraticPosDiscont2DFECollection: unknown geometry type.");
   }
   return 0; // Make some compilers happy
}


const FiniteElement *
GaussQuadraticDiscont2DFECollection::FiniteElementForGeometry(int GeomType)
   const
{
   switch (GeomType)
   {
   case Geometry::TRIANGLE:    return &TriangleFE;
   case Geometry::SQUARE:      return &QuadrilateralFE;
   default:
      mfem_error ("GaussQuadraticDiscont2DFECollection:"
                  " unknown geometry type.");
   }
   return &QuadrilateralFE; // Make some compilers happy
}

int GaussQuadraticDiscont2DFECollection::DofForGeometry(int GeomType) const
{
   switch (GeomType)
   {
   case Geometry::POINT:       return 0;
   case Geometry::SEGMENT:     return 0;
   case Geometry::TRIANGLE:    return 6;
   case Geometry::SQUARE:      return 9;
   default:
      mfem_error ("GaussQuadraticDiscont2DFECollection:"
                  " unknown geometry type.");
   }
   return 0; // Make some compilers happy
}

int * GaussQuadraticDiscont2DFECollection::DofOrderForOrientation(
   int GeomType, int Or) const
{
   return NULL;
}


const FiniteElement *
CubicDiscont2DFECollection::FiniteElementForGeometry(int GeomType) const
{
   switch (GeomType)
   {
   case Geometry::TRIANGLE:    return &TriangleFE;
   case Geometry::SQUARE:      return &QuadrilateralFE;
   default:
      mfem_error ("CubicDiscont2DFECollection: unknown geometry type.");
   }
   return &TriangleFE; // Make some compilers happy
}

int CubicDiscont2DFECollection::DofForGeometry(int GeomType) const
{
   switch (GeomType)
   {
   case Geometry::POINT:       return 0;
   case Geometry::SEGMENT:     return 0;
   case Geometry::TRIANGLE:    return 10;
   case Geometry::SQUARE:      return 16;
   default:
      mfem_error ("CubicDiscont2DFECollection: unknown geometry type.");
   }
   return 0; // Make some compilers happy
}

int * CubicDiscont2DFECollection::DofOrderForOrientation(int GeomType, int Or)
   const
{
   return NULL;
}


const FiniteElement *
LinearNonConf3DFECollection::FiniteElementForGeometry(int GeomType) const
{
   switch (GeomType)
   {
   case Geometry::TRIANGLE:    return &TriangleFE;
   case Geometry::SQUARE:      return &QuadrilateralFE;
   case Geometry::TETRAHEDRON: return &TetrahedronFE;
   case Geometry::CUBE:        return &ParallelepipedFE;
   default:
      mfem_error ("LinearNonConf3DFECollection: unknown geometry type.");
   }
   return &TriangleFE; // Make some compilers happy
}

int LinearNonConf3DFECollection::DofForGeometry(int GeomType) const
{
   switch (GeomType)
   {
   case Geometry::POINT:       return 0;
   case Geometry::SEGMENT:     return 0;
   case Geometry::TRIANGLE:    return 1;
   case Geometry::SQUARE:      return 1;
   case Geometry::TETRAHEDRON: return 0;
   case Geometry::CUBE:        return 0;
   default:
      mfem_error ("LinearNonConf3DFECollection: unknown geometry type.");
   }
   return 0; // Make some compilers happy
}

int * LinearNonConf3DFECollection::DofOrderForOrientation(int GeomType, int Or)
   const
{
   static int indexes[] = { 0 };

   return indexes;
}


const FiniteElement *
Const3DFECollection::FiniteElementForGeometry(int GeomType) const
{
   switch (GeomType)
   {
   case Geometry::TETRAHEDRON: return &TetrahedronFE;
   case Geometry::CUBE:        return &ParallelepipedFE;
   default:
      mfem_error ("Const3DFECollection: unknown geometry type.");
   }
   return &TetrahedronFE; // Make some compilers happy
}

int Const3DFECollection::DofForGeometry(int GeomType) const
{
   switch (GeomType)
   {
   case Geometry::POINT:       return 0;
   case Geometry::SEGMENT:     return 0;
   case Geometry::TRIANGLE:    return 0;
   case Geometry::TETRAHEDRON: return 1;
   case Geometry::SQUARE:      return 0;
   case Geometry::CUBE:        return 1;
   default:
      mfem_error ("Const3DFECollection: unknown geometry type.");
   }
   return 0; // Make some compilers happy
}

int * Const3DFECollection::DofOrderForOrientation(int GeomType, int Or)
   const
{
   return NULL;
}


const FiniteElement *
LinearDiscont3DFECollection::FiniteElementForGeometry(int GeomType) const
{
   switch (GeomType)
   {
   case Geometry::TETRAHEDRON: return &TetrahedronFE;
   case Geometry::CUBE:        return &ParallelepipedFE;
   default:
      mfem_error ("LinearDiscont3DFECollection: unknown geometry type.");
   }
   return &TetrahedronFE; // Make some compilers happy
}

int LinearDiscont3DFECollection::DofForGeometry(int GeomType) const
{
   switch (GeomType)
   {
   case Geometry::POINT:       return 0;
   case Geometry::SEGMENT:     return 0;
   case Geometry::TRIANGLE:    return 0;
   case Geometry::SQUARE:      return 0;
   case Geometry::TETRAHEDRON: return 4;
   case Geometry::CUBE:        return 8;
   default:
      mfem_error ("LinearDiscont3DFECollection: unknown geometry type.");
   }
   return 0; // Make some compilers happy
}

int * LinearDiscont3DFECollection::DofOrderForOrientation(int GeomType, int Or)
   const
{
   return NULL;
}


const FiniteElement *
QuadraticDiscont3DFECollection::FiniteElementForGeometry(int GeomType) const
{
   switch (GeomType)
   {
   case Geometry::TETRAHEDRON: return &TetrahedronFE;
   case Geometry::CUBE:        return &ParallelepipedFE;
   default:
      mfem_error ("QuadraticDiscont3DFECollection: unknown geometry type.");
   }
   return &TetrahedronFE; // Make some compilers happy
}

int QuadraticDiscont3DFECollection::DofForGeometry(int GeomType) const
{
   switch (GeomType)
   {
   case Geometry::POINT:       return 0;
   case Geometry::SEGMENT:     return 0;
   case Geometry::TRIANGLE:    return 0;
   case Geometry::SQUARE:      return 0;
   case Geometry::TETRAHEDRON: return 10;
   case Geometry::CUBE:        return 27;
   default:
      mfem_error ("QuadraticDiscont3DFECollection: unknown geometry type.");
   }
   return 0; // Make some compilers happy
}

int * QuadraticDiscont3DFECollection::DofOrderForOrientation(
   int GeomType, int Or) const
{
   return NULL;
}

const FiniteElement *
RefinedLinearFECollection::FiniteElementForGeometry(int GeomType) const
{
   switch (GeomType)
   {
   case Geometry::POINT:       return &PointFE;
   case Geometry::SEGMENT:     return &SegmentFE;
   case Geometry::TRIANGLE:    return &TriangleFE;
   case Geometry::SQUARE:      return &QuadrilateralFE;
   case Geometry::TETRAHEDRON: return &TetrahedronFE;
   case Geometry::CUBE:        return &ParallelepipedFE;
   default:
      mfem_error ("RefinedLinearFECollection: unknown geometry type.");
   }
   return &SegmentFE; // Make some compilers happy
}

int RefinedLinearFECollection::DofForGeometry(int GeomType) const
{
   switch (GeomType)
   {
   case Geometry::POINT:       return 1;
   case Geometry::SEGMENT:     return 1;
   case Geometry::TRIANGLE:    return 0;
   case Geometry::SQUARE:      return 1;
   case Geometry::TETRAHEDRON: return 0;
   case Geometry::CUBE:        return 1;
   default:
      mfem_error ("RefinedLinearFECollection: unknown geometry type.");
   }
   return 0; // Make some compilers happy
}

int * RefinedLinearFECollection::DofOrderForOrientation(int GeomType, int Or) const
{
   static int indexes[] = { 0 };

   return indexes;
}


const FiniteElement *
ND1_3DFECollection::FiniteElementForGeometry(int GeomType) const
{
   switch (GeomType)
   {
   case Geometry::CUBE:        return &HexahedronFE;
   case Geometry::TETRAHEDRON: return &TetrahedronFE;
   default:
      mfem_error ("ND1_3DFECollection: unknown geometry type.");
   }
   return &HexahedronFE; // Make some compilers happy
}

int ND1_3DFECollection::DofForGeometry(int GeomType) const
{
   switch (GeomType)
   {
   case Geometry::POINT:       return 0;
   case Geometry::SEGMENT:     return 1;
   case Geometry::TRIANGLE:    return 0;
   case Geometry::SQUARE:      return 0;
   case Geometry::TETRAHEDRON: return 0;
   case Geometry::CUBE:        return 0;
   default:
      mfem_error ("ND1_3DFECollection: unknown geometry type.");
   }
   return 0; // Make some compilers happy
}

int * ND1_3DFECollection::DofOrderForOrientation(int GeomType, int Or)
   const
{
   static int ind_pos[] = { 0 };
   static int ind_neg[] = { -1 };

   if (Or > 0)
      return ind_pos;
   return ind_neg;
}


const FiniteElement *
RT0_3DFECollection::FiniteElementForGeometry(int GeomType) const
{
   switch (GeomType)
   {
   case Geometry::TRIANGLE:    return &TriangleFE;
   case Geometry::SQUARE:      return &QuadrilateralFE;
   case Geometry::CUBE:        return &HexahedronFE;
   case Geometry::TETRAHEDRON: return &TetrahedronFE;
   default:
      mfem_error ("RT0_3DFECollection: unknown geometry type.");
   }
   return &HexahedronFE; // Make some compilers happy
}

int RT0_3DFECollection::DofForGeometry(int GeomType) const
{
   switch (GeomType)
   {
   case Geometry::POINT:       return 0;
   case Geometry::SEGMENT:     return 0;
   case Geometry::TRIANGLE:    return 1;
   case Geometry::SQUARE:      return 1;
   case Geometry::TETRAHEDRON: return 0;
   case Geometry::CUBE:        return 0;
   default:
      mfem_error ("RT0_3DFECollection: unknown geometry type.");
   }
   return 0; // Make some compilers happy
}

int * RT0_3DFECollection::DofOrderForOrientation(int GeomType, int Or)
   const
{
   static int ind_pos[] = { 0 };
   static int ind_neg[] = { -1 };

   if ((GeomType == Geometry::TRIANGLE) || (GeomType == Geometry::SQUARE))
   {
      if (Or % 2 == 0)
         return ind_pos;
      return ind_neg;
   }
   return NULL;
}

const FiniteElement *
RT1_3DFECollection::FiniteElementForGeometry(int GeomType) const
{
   switch (GeomType)
   {
   case Geometry::TRIANGLE:    return &TriangleFE;
   case Geometry::SQUARE:      return &QuadrilateralFE;
   case Geometry::CUBE:        return &HexahedronFE;
   default:
      mfem_error ("RT1_3DFECollection: unknown geometry type.");
   }
   return &HexahedronFE; // Make some compilers happy
}

int RT1_3DFECollection::DofForGeometry(int GeomType) const
{
   switch (GeomType)
   {
   case Geometry::POINT:       return 0;
   case Geometry::SEGMENT:     return 0;
   case Geometry::TRIANGLE:    return 2;
   case Geometry::SQUARE:      return 4;
   case Geometry::CUBE:        return 12;
   default:
      mfem_error ("RT1_3DFECollection: unknown geometry type.");
   }
   return 0; // Make some compilers happy
}

int * RT1_3DFECollection::DofOrderForOrientation(int GeomType, int Or)
   const
{
   if (GeomType == Geometry::SQUARE)
   {
      static int sq_ind[8][4] = {
         {0, 1, 2, 3}, {-1, -3, -2, -4},
         {2, 0, 3, 1}, {-2, -1, -4, -3},
         {3, 2, 1, 0}, {-4, -2, -3, -1},
         {1, 3, 0, 2}, {-3, -4, -1, -2}
      };

      return sq_ind[Or];
   }
   else
      return NULL;
}


H1_FECollection::H1_FECollection(const int p, const int dim)
{
   const int pm1 = p - 1, pm2 = pm1 - 1, pm3 = pm2 - 1;
   snprintf(h1_name, 32, "H1_%dD_P%d", dim, p);

   for (int g = 0; g < Geometry::NumGeom; g++)
   {
      H1_dof[g] = 0;
      H1_Elements[g] = NULL;
   }
   for (int i = 0; i < 2; i++)
      SegDofOrd[i] = NULL;
   for (int i = 0; i < 6; i++)
      TriDofOrd[i] = NULL;
   for (int i = 0; i < 8; i++)
      QuadDofOrd[i] = NULL;

   H1_dof[Geometry::POINT] = 1;
   H1_dof[Geometry::SEGMENT] = pm1;
   H1_Elements[Geometry::SEGMENT] = new H1_SegmentElement(p);

   if (dim >= 2)
   {
      H1_dof[Geometry::TRIANGLE] = (pm1*pm2)/2;
      H1_dof[Geometry::SQUARE] = pm1*pm1;
      H1_Elements[Geometry::TRIANGLE] = new H1_TriangleElement(p);
      H1_Elements[Geometry::SQUARE] = new H1_QuadrilateralElement(p);

      SegDofOrd[0] = new int[2*pm1];
      SegDofOrd[1] = SegDofOrd[0] + pm1;
      for (int i = 0; i < pm1; i++)
      {
         SegDofOrd[0][i] = i;
         SegDofOrd[1][i] = pm2 - i;
      }

      if (dim >= 3)
      {
         const int &TriDof = H1_dof[Geometry::TRIANGLE];
         const int &QuadDof = H1_dof[Geometry::SQUARE];
         H1_dof[Geometry::TETRAHEDRON] = (TriDof*pm3)/3;
         H1_dof[Geometry::CUBE] = QuadDof*pm1;
         H1_Elements[Geometry::TETRAHEDRON] = new H1_TetrahedronElement(p);
         H1_Elements[Geometry::CUBE] = new H1_HexahedronElement(p);

         TriDofOrd[0] = new int[6*TriDof];
         for (int i = 1; i < 6; i++)
            TriDofOrd[i] = TriDofOrd[i-1] + TriDof;
         // see Mesh::GetTriOrientation in mesh/mesh.cpp
         for (int j = 0; j < pm2; j++)
            for (int i = 0; i + j < pm2; i++)
            {
               int o = TriDof - ((pm1 - j)*(pm2 - j))/2 + i;
               int k = pm3 - j - i;
               TriDofOrd[0][o] = o;  // (0,1,2)
               TriDofOrd[1][o] = TriDof - ((pm1-j)*(pm2-j))/2 + k;  // (1,0,2)
               TriDofOrd[2][o] = TriDof - ((pm1-i)*(pm2-i))/2 + k;  // (2,0,1)
               TriDofOrd[3][o] = TriDof - ((pm1-k)*(pm2-k))/2 + i;  // (2,1,0)
               TriDofOrd[4][o] = TriDof - ((pm1-k)*(pm2-k))/2 + j;  // (1,2,0)
               TriDofOrd[5][o] = TriDof - ((pm1-i)*(pm2-i))/2 + j;  // (0,2,1)
            }

         QuadDofOrd[0] = new int[8*QuadDof];
         for (int i = 1; i < 8; i++)
            QuadDofOrd[i] = QuadDofOrd[i-1] + QuadDof;
         // see Mesh::GetQuadOrientation in mesh/mesh.cpp
         for (int j = 0; j < pm1; j++)
            for (int i = 0; i < pm1; i++)
            {
               int o = i + j*pm1;
               QuadDofOrd[0][o] = i + j*pm1;  // (0,1,2,3)
               QuadDofOrd[1][o] = j + i*pm1;  // (0,3,2,1)
               QuadDofOrd[2][o] = j + (pm2 - i)*pm1;  // (1,2,3,0)
               QuadDofOrd[3][o] = (pm2 - i) + j*pm1;  // (1,0,3,2)
               QuadDofOrd[4][o] = (pm2 - i) + (pm2 - j)*pm1;  // (2,3,0,1)
               QuadDofOrd[5][o] = (pm2 - j) + (pm2 - i)*pm1;  // (2,1,0,3)
               QuadDofOrd[6][o] = (pm2 - j) + i*pm1;  // (3,0,1,2)
               QuadDofOrd[7][o] = i + (pm2 - j)*pm1;  // (3,2,1,0)
            }
      }
   }
}

int *H1_FECollection::DofOrderForOrientation(int GeomType, int Or) const
{
   if (GeomType == Geometry::SEGMENT)
   {
      if (Or > 0)
         return SegDofOrd[0];
      return SegDofOrd[1];
   }
   else if (GeomType == Geometry::TRIANGLE)
   {
      return TriDofOrd[Or%6];
   }
   else if (GeomType == Geometry::SQUARE)
   {
      return QuadDofOrd[Or%8];
   }
   return NULL;
}

H1_FECollection::~H1_FECollection()
{
   delete [] SegDofOrd[0];
   delete [] TriDofOrd[0];
   delete [] QuadDofOrd[0];
   for (int g = 0; g < Geometry::NumGeom; g++)
      delete H1_Elements[g];
}


L2_FECollection::L2_FECollection(const int p, const int dim)
{
   snprintf(d_name, 32, "L2_%dD_P%d", dim, p);

   for (int g = 0; g < Geometry::NumGeom; g++)
      L2_Elements[g] = NULL;

   if (dim == 1)
   {
      L2_Elements[Geometry::SEGMENT] = new L2_SegmentElement(p);
   }
   else if (dim == 2)
   {
      L2_Elements[Geometry::TRIANGLE] = new L2_TriangleElement(p);
      L2_Elements[Geometry::SQUARE] = new L2_QuadrilateralElement(p);
   }
   else if (dim == 3)
   {
      L2_Elements[Geometry::TETRAHEDRON] = new L2_TetrahedronElement(p);
      L2_Elements[Geometry::CUBE] = new L2_HexahedronElement(p);
   }
   else
   {
      cerr << "L2_FECollection::L2_FECollection : dim = "
           << dim << endl;
      mfem_error();
   }
}

L2_FECollection::~L2_FECollection()
{
   for (int i = 0; i < Geometry::NumGeom; i++)
      delete L2_Elements[i];
}


RT_FECollection::RT_FECollection(const int p, const int dim)
{
   const int pp1 = p + 1, pp2 = p + 2;

   snprintf(rt_name, 32, "RT_%dD_P%d", dim, p);

   for (int g = 0; g < Geometry::NumGeom; g++)
   {
      RT_Elements[g] = NULL;
      RT_dof[g] = 0;
   }
   for (int i = 0; i < 2; i++)
      SegDofOrd[i] = NULL;
   for (int i = 0; i < 6; i++)
      TriDofOrd[i] = NULL;
   for (int i = 0; i < 8; i++)
      QuadDofOrd[i] = NULL;

   if (dim == 2)
   {
      RT_Elements[Geometry::TRIANGLE] = new RT_TriangleElement(p);
      RT_dof[Geometry::TRIANGLE] = p*pp1;

      RT_Elements[Geometry::SQUARE] = new RT_QuadrilateralElement(p);
      RT_dof[Geometry::SQUARE] = 2*p*pp1;

      RT_Elements[Geometry::SEGMENT] = new L2_SegmentElement(p);
      RT_dof[Geometry::SEGMENT] = pp1;

      SegDofOrd[0] = new int[2*pp1];
      SegDofOrd[1] = SegDofOrd[0] + pp1;
      for (int i = 0; i <= p; i++)
      {
         SegDofOrd[0][i] = i;
         SegDofOrd[1][i] = -1 - (p - i);
      }
   }
   else if (dim == 3)
   {
      RT_Elements[Geometry::TETRAHEDRON] = new RT_TetrahedronElement(p);
      RT_dof[Geometry::TETRAHEDRON] = p*pp1*pp2/2;

      RT_Elements[Geometry::CUBE] = new RT_HexahedronElement(p);
      RT_dof[Geometry::CUBE] = 3*p*pp1*pp1;

      RT_Elements[Geometry::TRIANGLE] = new L2_TriangleElement(p);
      RT_dof[Geometry::TRIANGLE] = pp1*pp2/2;

      RT_Elements[Geometry::SQUARE] = new L2_QuadrilateralElement(p);
      RT_dof[Geometry::SQUARE] = pp1*pp1;

      int TriDof = RT_dof[Geometry::TRIANGLE];
      TriDofOrd[0] = new int[6*TriDof];
      for (int i = 1; i < 6; i++)
         TriDofOrd[i] = TriDofOrd[i-1] + TriDof;
      // see Mesh::GetTriOrientation in mesh/mesh.cpp,
      // the constructor of H1_FECollection
      for (int j = 0; j <= p; j++)
         for (int i = 0; i + j <= p; i++)
         {
            int o = TriDof - ((pp2 - j)*(pp1 - j))/2 + i;
            int k = p - j - i;
            TriDofOrd[0][o] = o;  // (0,1,2)
            TriDofOrd[1][o] = -1-(TriDof-((pp2-j)*(pp1-j))/2+k);  // (1,0,2)
            TriDofOrd[2][o] =     TriDof-((pp2-i)*(pp1-i))/2+k;   // (2,0,1)
            TriDofOrd[3][o] = -1-(TriDof-((pp2-k)*(pp1-k))/2+i);  // (2,1,0)
            TriDofOrd[4][o] =     TriDof-((pp2-k)*(pp1-k))/2+j;   // (1,2,0)
            TriDofOrd[5][o] = -1-(TriDof-((pp2-i)*(pp1-i))/2+j);  // (0,2,1)
         }

      int QuadDof = RT_dof[Geometry::SQUARE];
      QuadDofOrd[0] = new int[8*QuadDof];
      for (int i = 1; i < 8; i++)
         QuadDofOrd[i] = QuadDofOrd[i-1] + QuadDof;
      // see Mesh::GetQuadOrientation in mesh/mesh.cpp
      for (int j = 0; j <= p; j++)
         for (int i = 0; i <= p; i++)
         {
            int o = i + j*pp1;
            QuadDofOrd[0][o] = i + j*pp1;                    // (0,1,2,3)
            QuadDofOrd[1][o] = -1 - (j + i*pp1);             // (0,3,2,1)
            QuadDofOrd[2][o] = j + (p - i)*pp1;              // (1,2,3,0)
            QuadDofOrd[3][o] = -1 - ((p - i) + j*pp1);       // (1,0,3,2)
            QuadDofOrd[4][o] = (p - i) + (p - j)*pp1;        // (2,3,0,1)
            QuadDofOrd[5][o] = -1 - ((p - j) + (p - i)*pp1); // (2,1,0,3)
            QuadDofOrd[6][o] = (p - j) + i*pp1;              // (3,0,1,2)
            QuadDofOrd[7][o] = -1 - (i + (p - j)*pp1);       // (3,2,1,0)
         }
   }
   else
   {
      cerr << "RT_FECollection::RT_FECollection : dim = " << dim << endl;
      mfem_error();
   }
}

int *RT_FECollection::DofOrderForOrientation(int GeomType, int Or) const
{
   if (GeomType == Geometry::SEGMENT)
   {
      if (Or > 0)
         return SegDofOrd[0];
      return SegDofOrd[1];
   }
   else if (GeomType == Geometry::TRIANGLE)
   {
      return TriDofOrd[Or%6];
   }
   else if (GeomType == Geometry::SQUARE)
   {
      return QuadDofOrd[Or%8];
   }
   return NULL;
}

RT_FECollection::~RT_FECollection()
{
   delete [] SegDofOrd[0];
   delete [] TriDofOrd[0];
   delete [] QuadDofOrd[0];
   for (int g = 0; g < Geometry::NumGeom; g++)
      delete RT_Elements[g];
}

ND_FECollection::ND_FECollection(const int p, const int dim)
{
   const int pm1 = p - 1, pm2 = p - 2;

   snprintf(nd_name, 32, "ND_%dD_P%d", dim, p);

   for (int g = 0; g < Geometry::NumGeom; g++)
   {
      ND_Elements[g] = NULL;
      ND_dof[g] = 0;
   }
   for (int i = 0; i < 2; i++)
      SegDofOrd[i] = NULL;
   for (int i = 0; i < 6; i++)
      TriDofOrd[i] = NULL;
   for (int i = 0; i < 8; i++)
      QuadDofOrd[i] = NULL;

   if (dim == 2 || dim == 3)
   {
      ND_Elements[Geometry::SQUARE] = new ND_QuadrilateralElement(p);
      ND_dof[Geometry::SQUARE] = 2*p*pm1;

      ND_Elements[Geometry::TRIANGLE] = new ND_TriangleElement(p);
      ND_dof[Geometry::TRIANGLE] = p*pm1;

      // ND_Elements[Geometry::SEGMENT] = NULL;
      ND_dof[Geometry::SEGMENT] = p;

      SegDofOrd[0] = new int[2*p];
      SegDofOrd[1] = SegDofOrd[0] + p;
      for (int i = 0; i < p; i++)
      {
         SegDofOrd[0][i] = i;
         SegDofOrd[1][i] = -1 - (pm1 - i);
      }
   }
   else
   {
      mfem_error("ND_FECollection::ND_FECollection : dim != 2 or 3");
   }

   if (dim == 3)
   {
      ND_Elements[Geometry::CUBE] = new ND_HexahedronElement(p);
      ND_dof[Geometry::CUBE] = 3*p*pm1*pm1;

      ND_Elements[Geometry::TETRAHEDRON] = new ND_TetrahedronElement(p);
      ND_dof[Geometry::TETRAHEDRON] = p*pm1*pm2/2;

      int QuadDof = ND_dof[Geometry::SQUARE];
      QuadDofOrd[0] = new int[8*QuadDof];
      for (int i = 1; i < 8; i++)
         QuadDofOrd[i] = QuadDofOrd[i-1] + QuadDof;
      // see Mesh::GetQuadOrientation in mesh/mesh.cpp
      for (int j = 0; j < pm1; j++)
         for (int i = 0; i < p; i++)
         {
            int d1 = i + j*p;            // x-component
            int d2 = p*pm1 + j + i*pm1;  // y-component
            // (0,1,2,3)
            QuadDofOrd[0][d1] = d1;
            QuadDofOrd[0][d2] = d2;
            // (0,3,2,1)
            QuadDofOrd[1][d1] = d2;
            QuadDofOrd[1][d2] = d1;
            // (1,2,3,0)
            // QuadDofOrd[2][d1] = p*pm1 + (pm2 - j) + i*pm1;
            // QuadDofOrd[2][d2] = -1 - ((pm1 - i) + j*p);
            QuadDofOrd[2][d1] = -1 - (p*pm1 + j + (pm1 - i)*pm1);
            QuadDofOrd[2][d2] = i + (pm2 - j)*p;
            // (1,0,3,2)
            QuadDofOrd[3][d1] = -1 - ((pm1 - i) + j*p);
            QuadDofOrd[3][d2] = p*pm1 + (pm2 - j) + i*pm1;
            // (2,3,0,1)
            QuadDofOrd[4][d1] = -1 - ((pm1 - i) + (pm2 - j)*p);
            QuadDofOrd[4][d2] = -1 - (p*pm1 + (pm2 - j) + (pm1 - i)*pm1);
            // (2,1,0,3)
            QuadDofOrd[5][d1] = -1 - (p*pm1 + (pm2 - j) + (pm1 - i)*pm1);
            QuadDofOrd[5][d2] = -1 - ((pm1 - i) + (pm2 - j)*p);
            // (3,0,1,2)
            // QuadDofOrd[6][d1] = -1 - (p*pm1 + j + (pm1 - i)*pm1);
            // QuadDofOrd[6][d2] = i + (pm2 - j)*p;
            QuadDofOrd[6][d1] = p*pm1 + (pm2 - j) + i*pm1;
            QuadDofOrd[6][d2] = -1 - ((pm1 - i) + j*p);
            // (3,2,1,0)
            QuadDofOrd[7][d1] = i + (pm2 - j)*p;
            QuadDofOrd[7][d2] = -1 - (p*pm1 + j + (pm1 - i)*pm1);
         }

      int TriDof = ND_dof[Geometry::TRIANGLE];
      TriDofOrd[0] = new int[6*TriDof];
      for (int i = 1; i < 6; i++)
         TriDofOrd[i] = TriDofOrd[i-1] + TriDof;
      // see Mesh::GetTriOrientation in mesh/mesh.cpp,
      // the constructor of H1_FECollection
      for (int j = 0; j <= pm2; j++)
         for (int i = 0; i + j <= pm2; i++)
         {
            int k1 = p*pm1 - (p - j)*(pm1 - j) + 2*i;
            int k2 = p*pm1 - (p - i)*(pm1 - i) + 2*j;
            // (0,1,2)
            TriDofOrd[0][k1  ] = k1;
            TriDofOrd[0][k1+1] = k1 + 1;
            // (0,2,1)
            TriDofOrd[5][k1  ] = k2 + 1;
            TriDofOrd[5][k1+1] = k2;

            // The other orientations can not be supported with the current
            // interface. The method Mesh::ReorientTetMesh will ensure that
            // only orientations 0 and 5 are generated.
         }
   }
}

int *ND_FECollection::DofOrderForOrientation(int GeomType, int Or) const
{
   if (GeomType == Geometry::SEGMENT)
   {
      if (Or > 0)
         return SegDofOrd[0];
      return SegDofOrd[1];
   }
   else if (GeomType == Geometry::TRIANGLE)
   {
      if (Or != 0 && Or != 5)
      {
         cerr <<
            "ND_FECollection::DofOrderForOrientation :\n"
            "  triangle face orientation " << Or << " is not supported!\n"
            "  Use Mesh::ReorientTetMesh to fix it." << endl;
         mfem_error();
      }
      return TriDofOrd[Or%6];
   }
   else if (GeomType == Geometry::SQUARE)
   {
      return QuadDofOrd[Or%8];
   }
   return NULL;
}

ND_FECollection::~ND_FECollection()
{
   delete [] SegDofOrd[0];
   delete [] TriDofOrd[0];
   delete [] QuadDofOrd[0];
   for (int g = 0; g < Geometry::NumGeom; g++)
      delete ND_Elements[g];
}


Local_FECollection::Local_FECollection(const char *fe_name)
{
   snprintf(d_name, 32, "Local_%s", fe_name);

   Local_Element = NULL;

   if (!strcmp(fe_name, "BiCubic2DFiniteElement") ||
       !strcmp(fe_name, "Quad_Q3"))
   {
      GeomType = Geometry::SQUARE;
      Local_Element = new BiCubic2DFiniteElement;
   }
   else if (!strcmp(fe_name, "Nedelec1HexFiniteElement") ||
            !strcmp(fe_name, "Hex_ND1"))
   {
      GeomType = Geometry::CUBE;
      Local_Element = new Nedelec1HexFiniteElement;
   }
   else
   {
      cerr << "Local_FECollection::Local_FECollection : fe_name = "
           << fe_name << endl;
      mfem_error();
   }
}


void NURBSFECollection::Allocate(int Order)
{
   SegmentFE        = new NURBS1DFiniteElement(Order);
   QuadrilateralFE  = new NURBS2DFiniteElement(Order);
   ParallelepipedFE = new NURBS3DFiniteElement(Order);

   sprintf(name, "NURBS%i", Order);
}

void NURBSFECollection::Deallocate()
{
   delete ParallelepipedFE;
   delete QuadrilateralFE;
   delete SegmentFE;
}

const FiniteElement *
NURBSFECollection::FiniteElementForGeometry(int GeomType) const
{
   switch (GeomType)
   {
   case Geometry::SEGMENT:     return SegmentFE;
   case Geometry::SQUARE:      return QuadrilateralFE;
   case Geometry::CUBE:        return ParallelepipedFE;
   default:
      mfem_error ("NURBSFECollection: unknown geometry type.");
   }
   return SegmentFE; // Make some compilers happy
}

int NURBSFECollection::DofForGeometry(int GeomType) const
{
   mfem_error("NURBSFECollection::DofForGeometry");
   return 0; // Make some compilers happy
}

int *NURBSFECollection::DofOrderForOrientation(int GeomType, int Or) const
{
   mfem_error("NURBSFECollection::DofOrderForOrientation");
   return NULL;
}
