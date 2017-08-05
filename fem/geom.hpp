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
    Geometry::CUBE - the unit cube
    Geometry::PENTATOPE - w/ vert. (0,0,0,0),(1,0,0,0),(0,1,0,0),(0,0,1,0),(0,0,0,1)
    Geometry::TESSERACT - the 4d unit cube                                           */
class Geometry
{
public:
   enum Type { POINT, SEGMENT, TRIANGLE, SQUARE, TETRAHEDRON, CUBE, PENTATOPE, TESSERACT };

   static const int NumGeom = 8;
   static const int NumBdrArray[];
   static const char *Name[NumGeom];
   static const double Volume[NumGeom];
   static const int Dimension[NumGeom];
   static const int NumVerts[NumGeom];
   static const int NumEdges[NumGeom];
   static const int NumFaces[NumGeom];

   // Structure that holds constants describing the Geometries.
   // Currently it contains just the space dimension.
   template <Type Geom> struct Constants;

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

template <> struct Geometry::Constants<Geometry::POINT>
{
   static const int Dimension = 0;
   static const int NumVert = 1;

   static const int NumOrient = 1;
   static const int Orient[NumOrient][NumVert];
   static const int InvOrient[NumOrient];
};

template <> struct Geometry::Constants<Geometry::SEGMENT>
{
   static const int Dimension = 1;
   static const int NumVert = 2;
   static const int NumEdges = 1;
   static const int Edges[NumEdges][2];

   static const int NumOrient = 2;
   static const int Orient[NumOrient][NumVert];
   static const int InvOrient[NumOrient];
};

template <> struct Geometry::Constants<Geometry::TRIANGLE>
{
   static const int Dimension = 2;
   static const int NumVert = 3;
   static const int NumEdges = 3;
   static const int Edges[NumEdges][2];
   // Lower-triangular part of the local vertex-to-vertex graph.
   struct VertToVert
   {
      static const int I[NumVert];
      static const int J[NumEdges][2]; // {end,edge_idx}
   };
   static const int NumFaces = 1;
   static const int FaceVert[NumFaces][NumVert];

   // For a given base tuple v={v0,v1,v2}, the orientation of a permutation
   // u={u0,u1,u2} of v, is an index 'j' such that u[i]=v[Orient[j][i]].
   // The static method Mesh::GetTriOrientation, computes the index 'j' of the
   // permutation that maps the second argument 'test' to the first argument
   // 'base': test[Orient[j][i]]=base[i].
   static const int NumOrient = 6;
   static const int Orient[NumOrient][NumVert];
   // The inverse of orientation 'j' is InvOrient[j].
   static const int InvOrient[NumOrient];
};

template <> struct Geometry::Constants<Geometry::SQUARE>
{
   static const int Dimension = 2;
   static const int NumVert = 4;
   static const int NumEdges = 4;
   static const int Edges[NumEdges][2];
   // Lower-triangular part of the local vertex-to-vertex graph.
   struct VertToVert
   {
      static const int I[NumVert];
      static const int J[NumEdges][2]; // {end,edge_idx}
   };
   static const int NumFaces = 1;
   static const int FaceVert[NumFaces][NumVert];

   static const int NumOrient = 8;
   static const int Orient[NumOrient][NumVert];
   static const int InvOrient[NumOrient];
};

template <> struct Geometry::Constants<Geometry::TETRAHEDRON>
{
   static const int Dimension = 3;
   static const int NumVert = 4;
   static const int NumEdges = 6;
   static const int Edges[NumEdges][2];
   static const int NumFaces = 4;
   static const int FaceTypes[NumFaces];
   static const int MaxFaceVert = 3;
   static const int FaceVert[NumFaces][MaxFaceVert];
   // Lower-triangular part of the local vertex-to-vertex graph.
   struct VertToVert
   {
      static const int I[NumVert];
      static const int J[NumEdges][2]; // {end,edge_idx}
   };

   static const int NumOrient = 24;
   static const int Orient[NumOrient][NumVert];
   // The inverse of orientation 'j' is InvOrient[j].
   static const int InvOrient[NumOrient];

};

template <> struct Geometry::Constants<Geometry::CUBE>
{
   static const int Dimension = 3;
   static const int NumVert = 8;
   static const int NumEdges = 12;
   static const int Edges[NumEdges][2];
   static const int NumFaces = 6;
   static const int FaceTypes[NumFaces];
   static const int MaxFaceVert = 4;
   static const int FaceVert[NumFaces][MaxFaceVert];
   // Lower-triangular part of the local vertex-to-vertex graph.
   struct VertToVert
   {
      static const int I[NumVert];
      static const int J[NumEdges][2]; // {end,edge_idx}
   };
};

template <> struct Geometry::Constants<Geometry::PENTATOPE>
{
   static const int Dimension = 4;
   static const int NumVert = 5;
   static const int NumEdges = 10;
   static const int Edges[NumEdges][2];
   static const int NumFaces = 5;
   static const int FaceTypes[NumFaces];
   static const int MaxFaceVert = 4;
   static const int FaceVert[NumFaces][MaxFaceVert];
   static const int NumPlanar = 10;
   static const int MaxPlanarVert = 3;
   static const int PlanarVert[NumPlanar][MaxPlanarVert];
   // Lower-triangular part of the local vertex-to-vertex graph.
   struct VertToVert
   {
      static const int I[NumVert];
      static const int J[NumEdges][2]; // {end,edge_idx}
   };
};

template <> struct Geometry::Constants<Geometry::TESSERACT>
{
   static const int Dimension = 4;
   static const int NumVert = 16;
   static const int NumEdges = 32;
   static const int Edges[NumEdges][2];
   static const int NumFaces = 8;
   static const int FaceTypes[NumFaces];
   static const int MaxFaceVert = 8;
   static const int FaceVert[NumFaces][MaxFaceVert];
   // Lower-triangular part of the local vertex-to-vertex graph.
   struct VertToVert
   {
      static const int I[NumVert];
      static const int J[NumEdges][2]; // {end,edge_idx}
   };
};

extern Geometry Geometries;

class RefinedGeometry
{
public:
   int Times, ETimes;
   IntegrationRule RefPts;
   Array<int> RefGeoms, RefEdges;
   int NumBdrEdges; // at the begining of RefEdges

   RefinedGeometry(int NPts, int NRefG, int NRefE, int NBdrE = 0) :
      RefPts(NPts), RefGeoms(NRefG), RefEdges(NRefE), NumBdrEdges(NBdrE) { }
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
