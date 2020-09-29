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
    Geometry::PRISM - w/ vert. (0,0,0),(1,0,0),(0,1,0),(0,0,1),(1,0,1),(0,1,1)
*/
class Geometry
{
public:
   enum Type
   {
      INVALID = -1,
      POINT = 0, SEGMENT, TRIANGLE, SQUARE, TETRAHEDRON, CUBE, PRISM,
      NUM_GEOMETRIES
   };

   static const int NumGeom = NUM_GEOMETRIES;
   static const int MaxDim = 3;
   static const int NumBdrArray[NumGeom];
   static const char *Name[NumGeom];
   static const double Volume[NumGeom];
   static const int Dimension[NumGeom];
   static const int DimStart[MaxDim+2]; // including MaxDim+1
   static const int NumVerts[NumGeom];
   static const int NumEdges[NumGeom];
   static const int NumFaces[NumGeom];

   // Structure that holds constants describing the Geometries.
   template <Type Geom> struct Constants;

private:
   IntegrationRule *GeomVert[NumGeom];
   IntegrationPoint GeomCenter[NumGeom];
   DenseMatrix *GeomToPerfGeomJac[NumGeom];
   DenseMatrix *PerfGeomToGeomJac[NumGeom];

public:
   Geometry();
   ~Geometry();

   /** @brief Return an IntegrationRule consisting of all vertices of the given
       Geometry::Type, @a GeomType. */
   const IntegrationRule *GetVertices(int GeomType);

   /// Return the center of the given Geometry::Type, @a GeomType.
   const IntegrationPoint &GetCenter(int GeomType)
   { return GeomCenter[GeomType]; }

   /// Get a random point in the reference element specified by @a GeomType.
   /** This method uses the function rand() for random number generation. */
   static void GetRandomPoint(int GeomType, IntegrationPoint &ip);

   /// Check if the given point is inside the given reference element.
   static bool CheckPoint(int GeomType, const IntegrationPoint &ip);
   /** @brief Check if the given point is inside the given reference element.
       Overload for fuzzy tolerance. */
   static bool CheckPoint(int GeomType, const IntegrationPoint &ip, double eps);

   /// Project a point @a end, onto the given Geometry::Type, @a GeomType.
   /** Check if the @a end point is inside the reference element, if not
       overwrite it with the point on the boundary that lies on the line segment
       between @a beg and @a end (@a beg must be inside the element). Return
       true if @a end is inside the element, and false otherwise. */
   static bool ProjectPoint(int GeomType, const IntegrationPoint &beg,
                            IntegrationPoint &end);

   /// Project a point @a ip, onto the given Geometry::Type, @a GeomType.
   /** If @a ip is outside the element, replace it with the point on the
       boundary that is closest to the original @a ip and return false;
       otherwise, return true without changing @a ip. */
   static bool ProjectPoint(int GeomType, IntegrationPoint &ip);

   const DenseMatrix &GetGeomToPerfGeomJac(int GeomType) const
   { return *GeomToPerfGeomJac[GeomType]; }
   DenseMatrix *GetPerfGeomToGeomJac(int GeomType)
   { return PerfGeomToGeomJac[GeomType]; }
   void GetPerfPointMat(int GeomType, DenseMatrix &pm);
   void JacToPerfJac(int GeomType, const DenseMatrix &J,
                     DenseMatrix &PJ) const;

   /// Return the number of boundary "faces" of a given Geometry::Type.
   int NumBdr(int GeomType) { return NumBdrArray[GeomType]; }
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
   // Upper-triangular part of the local vertex-to-vertex graph.
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
   // Upper-triangular part of the local vertex-to-vertex graph.
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
   // Upper-triangular part of the local vertex-to-vertex graph.
   struct VertToVert
   {
      static const int I[NumVert];
      static const int J[NumEdges][2]; // {end,edge_idx}
   };

   static const int NumOrient = 24;
   static const int Orient[NumOrient][NumVert];
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
   // Upper-triangular part of the local vertex-to-vertex graph.
   struct VertToVert
   {
      static const int I[NumVert];
      static const int J[NumEdges][2]; // {end,edge_idx}
   };
};

template <> struct Geometry::Constants<Geometry::PRISM>
{
   static const int Dimension = 3;
   static const int NumVert = 6;
   static const int NumEdges = 9;
   static const int Edges[NumEdges][2];
   static const int NumFaces = 5;
   static const int FaceTypes[NumFaces];
   static const int MaxFaceVert = 4;
   static const int FaceVert[NumFaces][MaxFaceVert];
   // Upper-triangular part of the local vertex-to-vertex graph.
   struct VertToVert
   {
      static const int I[NumVert];
      static const int J[NumEdges][2]; // {end,edge_idx}
   };
};

// Defined in fe.cpp to ensure construction after 'mfem::WedgeFE'.
extern Geometry Geometries;


class RefinedGeometry
{
public:
   int Times, ETimes;
   IntegrationRule RefPts;
   Array<int> RefGeoms, RefEdges;
   int NumBdrEdges; // at the beginning of RefEdges
   int Type;

   RefinedGeometry(int NPts, int NRefG, int NRefE, int NBdrE = 0) :
      RefPts(NPts), RefGeoms(NRefG), RefEdges(NRefE), NumBdrEdges(NBdrE) { }
};

class GeometryRefiner
{
private:
   int type; // Quadrature1D type (ClosedUniform is default)
   Array<RefinedGeometry *> RGeom[Geometry::NumGeom];
   Array<IntegrationRule *> IntPts[Geometry::NumGeom];

   RefinedGeometry *FindInRGeom(Geometry::Type Geom, int Times, int ETimes,
                                int Type);
   IntegrationRule *FindInIntPts(Geometry::Type Geom, int NPts);

public:
   GeometryRefiner();

   /// Set the Quadrature1D type of points to use for subdivision.
   void SetType(const int t) { type = t; }
   /// Get the Quadrature1D type of points used for subdivision.
   int GetType() const { return type; }

   RefinedGeometry *Refine(Geometry::Type Geom, int Times, int ETimes = 1);

   /// @note This method always uses Quadrature1D::OpenUniform points.
   const IntegrationRule *RefineInterior(Geometry::Type Geom, int Times);

   /// Get the Refinement level based on number of points
   virtual int GetRefinementLevelFromPoints(Geometry::Type Geom, int Npts);

   /// Get the Refinement level based on number of elements
   virtual int GetRefinementLevelFromElems(Geometry::Type geom, int Npts);

   ~GeometryRefiner();
};

extern GeometryRefiner GlobGeometryRefiner;

}

#endif
