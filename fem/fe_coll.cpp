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
#include <stdlib.h>
#include <string.h>

int FiniteElementCollection::HasFaceDofs (int GeomType) const
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
                                 {1, 3, 0, 2}, {1, 0, 3, 2},
                                 {3, 2, 1, 0}, {3, 1, 2, 0},
                                 {2, 0, 3, 1}, {2, 3, 0, 1}};
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
   default:
      mfem_error ("RT1_2DFECollection: unknown geometry type.");
   }
   return 0; // Make some compilers happy
}

int * RT1_2DFECollection::DofOrderForOrientation(int GeomType, int Or)
   const
{
   static int ind_pos[] = { 0, 1 };
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
   case Geometry::TETRAHEDRON: return &TetrahedronFE;
   default:
      mfem_error ("RT0_3DFECollection: unknown geometry type.");
   }
   return &TetrahedronFE; // Make some compilers happy
}

int RT0_3DFECollection::DofForGeometry(int GeomType) const
{
   switch (GeomType)
   {
   case Geometry::POINT:       return 0;
   case Geometry::SEGMENT:     return 0;
   case Geometry::TRIANGLE:    return 1;
   case Geometry::TETRAHEDRON: return 0;
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

   if (GeomType == Geometry::TRIANGLE)
   {
      if (Or % 2 == 0)
         return ind_pos;
      return ind_neg;
   }
   return NULL;
}
