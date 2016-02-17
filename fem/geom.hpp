// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#ifndef MFEM_GEOM
#define MFEM_GEOM

#include "../config/config.hpp"
#include "../linalg/densemat.hpp"
#include "intrules.hpp"

namespace mfem
{

/** Types of domains for integration rules and reference finite elements:
    Geometry::POINT    - a point
    Geometry::SEGMENT  - the interval [0,1]
    Geometry::TRIANGLE - triangle with vertices (0,0), (1,0), (0,1)
    Geometry::SQUARE   - the unit square (0,1)x(0,1)
    Geometry::TETRAHEDRON - w/ vert. (0,0,0),(1,0,0),(0,1,0),(0,0,1)
    Geometry::CUBE - the unit cube                                    */
class Geometry
{
public:
   enum Type { POINT, SEGMENT, TRIANGLE, SQUARE, TETRAHEDRON, CUBE };

   static const int NumGeom = 6;
   static const int NumBdrArray[];
   static const char *Name[NumGeom];
   static const double Volume[NumGeom];

private:
   IntegrationRule *GeomVert[NumGeom];
   IntegrationPoint GeomCenter[NumGeom];
   DenseMatrix *PerfGeomToGeomJac[NumGeom];

public:

   Geometry();

   ~Geometry();

   const IntegrationRule *GetVertices(int GeomType);
   const IntegrationPoint &GetCenter(int GeomType)
   { return GeomCenter[GeomType]; }
   /** Get a random point in the reference element specified by GeomType.
       This method uses the function rand() for random number generation. */
   static void GetRandomPoint(int GeomType, IntegrationPoint &ip);
   /// Check if the given point is inside the given reference element.
   static bool CheckPoint(int GeomType, const IntegrationPoint &ip);
   /** Check if the end point is inside the reference element, if not overwrite
       it with the point on the boundary that lies on the line segment between
       beg and end (beg must be inside the element). Return true if end is
       inside the element, and false otherwise. */
   static bool ProjectPoint(int GeomType, const IntegrationPoint &beg,
                            IntegrationPoint &end);

   DenseMatrix *GetPerfGeomToGeomJac(int GeomType)
   { return PerfGeomToGeomJac[GeomType]; }
   void GetPerfPointMat(int GeomType, DenseMatrix &pm);
   void JacToPerfJac(int GeomType, const DenseMatrix &J,
                     DenseMatrix &PJ) const;

   int NumBdr (int GeomType) { return NumBdrArray[GeomType]; }
};

extern Geometry Geometries;

class RefinedGeometry
{
public:
   int Times, ETimes;
   IntegrationRule RefPts;
   Array<int> RefGeoms, RefEdges;

   RefinedGeometry (int NPts, int NRefG, int NRefE) :
      RefPts (NPts), RefGeoms (NRefG), RefEdges (NRefE) { }
};

class GeometryRefiner
{
private:
   int type; // 0 - uniform points, otherwise - poly1d.ClosedPoints
   RefinedGeometry *RGeom[Geometry::NumGeom];
   IntegrationRule *IntPts[Geometry::NumGeom];
public:
   GeometryRefiner();

   void SetType(const int t) { type = t; }
   RefinedGeometry *Refine (int Geom, int Times, int ETimes = 1);
   const IntegrationRule *RefineInterior(int Geom, int Times);

   ~GeometryRefiner();
};

extern GeometryRefiner GlobGeometryRefiner;

}

#endif
