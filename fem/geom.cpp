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

Geometry::Geometry()
{
   // Vertices for Geometry::POINT
   GeomVert[0] = NULL; // No vertices, dimension is 0

   // Vertices for Geometry::SEGMENT
   GeomVert[1] = new IntegrationRule(2);

   GeomVert[1]->IntPoint(0).x = 0.0;
   GeomVert[1]->IntPoint(1).x = 1.0;

   // Vertices for Geometry::TRIANGLE
   GeomVert[2] = new IntegrationRule(3);

   GeomVert[2]->IntPoint(0).x = 0.0;
   GeomVert[2]->IntPoint(0).y = 0.0;

   GeomVert[2]->IntPoint(1).x = 1.0;
   GeomVert[2]->IntPoint(1).y = 0.0;

   GeomVert[2]->IntPoint(2).x = 0.0;
   GeomVert[2]->IntPoint(2).y = 1.0;

   // Vertices for Geometry::SQUARE
   GeomVert[3] = new IntegrationRule(4);

   GeomVert[3]->IntPoint(0).x = 0.0;
   GeomVert[3]->IntPoint(0).y = 0.0;

   GeomVert[3]->IntPoint(1).x = 1.0;
   GeomVert[3]->IntPoint(1).y = 0.0;

   GeomVert[3]->IntPoint(2).x = 1.0;
   GeomVert[3]->IntPoint(2).y = 1.0;

   GeomVert[3]->IntPoint(3).x = 0.0;
   GeomVert[3]->IntPoint(3).y = 1.0;

   // Vertices for Geometry::TETRAHEDRON
   GeomVert[4] = new IntegrationRule(4);
   GeomVert[4]->IntPoint(0).x = 0.0;
   GeomVert[4]->IntPoint(0).y = 0.0;
   GeomVert[4]->IntPoint(0).z = 0.0;

   GeomVert[4]->IntPoint(1).x = 1.0;
   GeomVert[4]->IntPoint(1).y = 0.0;
   GeomVert[4]->IntPoint(1).z = 0.0;

   GeomVert[4]->IntPoint(2).x = 0.0;
   GeomVert[4]->IntPoint(2).y = 1.0;
   GeomVert[4]->IntPoint(2).z = 0.0;

   GeomVert[4]->IntPoint(3).x = 0.0;
   GeomVert[4]->IntPoint(3).y = 0.0;
   GeomVert[4]->IntPoint(3).z = 1.0;

   // Vertices for Geometry::CUBE
   GeomVert[5] = new IntegrationRule(8);

   GeomVert[5]->IntPoint(0).x = 0.0;
   GeomVert[5]->IntPoint(0).y = 0.0;
   GeomVert[5]->IntPoint(0).z = 0.0;

   GeomVert[5]->IntPoint(1).x = 1.0;
   GeomVert[5]->IntPoint(1).y = 0.0;
   GeomVert[5]->IntPoint(1).z = 0.0;

   GeomVert[5]->IntPoint(2).x = 1.0;
   GeomVert[5]->IntPoint(2).y = 1.0;
   GeomVert[5]->IntPoint(2).z = 0.0;

   GeomVert[5]->IntPoint(3).x = 0.0;
   GeomVert[5]->IntPoint(3).y = 1.0;
   GeomVert[5]->IntPoint(3).z = 0.0;

   GeomVert[5]->IntPoint(4).x = 0.0;
   GeomVert[5]->IntPoint(4).y = 0.0;
   GeomVert[5]->IntPoint(4).z = 1.0;

   GeomVert[5]->IntPoint(5).x = 1.0;
   GeomVert[5]->IntPoint(5).y = 0.0;
   GeomVert[5]->IntPoint(5).z = 1.0;

   GeomVert[5]->IntPoint(6).x = 1.0;
   GeomVert[5]->IntPoint(6).y = 1.0;
   GeomVert[5]->IntPoint(6).z = 1.0;

   GeomVert[5]->IntPoint(7).x = 0.0;
   GeomVert[5]->IntPoint(7).y = 1.0;
   GeomVert[5]->IntPoint(7).z = 1.0;

   GeomCenter[POINT].x = 0.0;
   GeomCenter[POINT].y = 0.0;
   GeomCenter[POINT].z = 0.0;

   GeomCenter[SEGMENT].x = 0.5;
   GeomCenter[SEGMENT].y = 0.0;
   GeomCenter[SEGMENT].z = 0.0;

   GeomCenter[TRIANGLE].x = 1.0 / 3.0;
   GeomCenter[TRIANGLE].y = 1.0 / 3.0;
   GeomCenter[TRIANGLE].z = 0.0;

   GeomCenter[SQUARE].x = 0.5;
   GeomCenter[SQUARE].y = 0.5;
   GeomCenter[SQUARE].z = 0.0;

   GeomCenter[TETRAHEDRON].x = 0.25;
   GeomCenter[TETRAHEDRON].y = 0.25;
   GeomCenter[TETRAHEDRON].z = 0.25;

   GeomCenter[CUBE].x = 0.5;
   GeomCenter[CUBE].y = 0.5;
   GeomCenter[CUBE].z = 0.5;

   PerfGeomToGeomJac[POINT]       = NULL;
   PerfGeomToGeomJac[SEGMENT]     = NULL;
   PerfGeomToGeomJac[TRIANGLE]    = new DenseMatrix(2);
   PerfGeomToGeomJac[SQUARE]      = NULL;
   PerfGeomToGeomJac[TETRAHEDRON] = new DenseMatrix(3);
   PerfGeomToGeomJac[CUBE]        = NULL;

   {
      Linear2DFiniteElement TriFE;
      IsoparametricTransformation tri_T;
      tri_T.SetFE(&TriFE);
      GetPerfPointMat (TRIANGLE, tri_T.GetPointMat());
      tri_T.SetIntPoint(&GeomCenter[TRIANGLE]);
      CalcInverse(tri_T.Jacobian(), *PerfGeomToGeomJac[TRIANGLE]);
   }
   {
      Linear3DFiniteElement TetFE;
      IsoparametricTransformation tet_T;
      tet_T.SetFE(&TetFE);
      GetPerfPointMat (TETRAHEDRON, tet_T.GetPointMat());
      tet_T.SetIntPoint(&GeomCenter[TETRAHEDRON]);
      CalcInverse(tet_T.Jacobian(), *PerfGeomToGeomJac[TETRAHEDRON]);
   }
}

Geometry::~Geometry()
{
   for (int i = 0; i < NumGeom; i++)
   {
      delete PerfGeomToGeomJac[i];
      delete GeomVert[i];
   }
}

const IntegrationRule * Geometry::GetVertices(int GeomType)
{
   switch (GeomType)
   {
   case Geometry::POINT:       return GeomVert[0];
   case Geometry::SEGMENT:     return GeomVert[1];
   case Geometry::TRIANGLE:    return GeomVert[2];
   case Geometry::SQUARE:      return GeomVert[3];
   case Geometry::TETRAHEDRON: return GeomVert[4];
   case Geometry::CUBE:        return GeomVert[5];
   default:
      mfem_error ("Geometry::GetVertices(...)");
   }
   // make some compilers happy.
   return GeomVert[0];
}

void Geometry::GetPerfPointMat (int GeomType, DenseMatrix &pm)
{
   switch (GeomType)
   {
   case Geometry::SEGMENT:
   {
      pm.SetSize (1, 2);
      pm(0,0) = 0.0;
      pm(0,1) = 1.0;
   }
   break;

   case Geometry::TRIANGLE:
   {
      pm.SetSize (2, 3);
      pm(0,0) = 0.0;  pm(1,0) = 0.0;
      pm(0,1) = 1.0;  pm(1,1) = 0.0;
      pm(0,2) = 0.5;  pm(1,2) = 0.86602540378443864676;
   }
   break;

   case Geometry::SQUARE:
   {
      pm.SetSize (2, 4);
      pm(0,0) = 0.0;  pm(1,0) = 0.0;
      pm(0,1) = 1.0;  pm(1,1) = 0.0;
      pm(0,2) = 1.0;  pm(1,2) = 1.0;
      pm(0,3) = 0.0;  pm(1,3) = 1.0;
   }
   break;

   case Geometry::TETRAHEDRON:
   {
      pm.SetSize (3, 4);
      pm(0,0) = 0.0;  pm(1,0) = 0.0;  pm(2,0) = 0.0;
      pm(0,1) = 1.0;  pm(1,1) = 0.0;  pm(2,1) = 0.0;
      pm(0,2) = 0.5;  pm(1,2) = 0.86602540378443864676;  pm(2,2) = 0.0;
      pm(0,3) = 0.5;  pm(1,3) = 0.28867513459481288225;
      pm(2,3) = 0.81649658092772603273;
   }
   break;

   case Geometry::CUBE:
   {
      pm.SetSize (3, 8);
      pm(0,0) = 0.0;  pm(1,0) = 0.0;  pm(2,0) = 0.0;
      pm(0,1) = 1.0;  pm(1,1) = 0.0;  pm(2,1) = 0.0;
      pm(0,2) = 1.0;  pm(1,2) = 1.0;  pm(2,2) = 0.0;
      pm(0,3) = 0.0;  pm(1,3) = 1.0;  pm(2,3) = 0.0;
      pm(0,4) = 0.0;  pm(1,4) = 0.0;  pm(2,4) = 1.0;
      pm(0,5) = 1.0;  pm(1,5) = 0.0;  pm(2,5) = 1.0;
      pm(0,6) = 1.0;  pm(1,6) = 1.0;  pm(2,6) = 1.0;
      pm(0,7) = 0.0;  pm(1,7) = 1.0;  pm(2,7) = 1.0;
   }
   break;

   default:
      mfem_error ("Geometry::GetPerfPointMat (...)");
   }
}

void Geometry::JacToPerfJac(int GeomType, const DenseMatrix &J,
                            DenseMatrix &PJ) const
{
   if (PerfGeomToGeomJac[GeomType])
   {
      Mult(J, *PerfGeomToGeomJac[GeomType], PJ);
   }
   else
   {
      PJ = J;
   }
}

const int Geometry::NumGeom;

const int Geometry::NumBdrArray[] = { 0, 2, 3, 4, 4, 6 };

Geometry Geometries;


GeometryRefiner::GeometryRefiner()
{
   for (int i = 0; i < Geometry::NumGeom; i++)
   {
      RGeom[i] = NULL;
      IntPts[i] = NULL;
   }
}

GeometryRefiner::~GeometryRefiner()
{
   for (int i = 0; i < Geometry::NumGeom; i++)
   {
      delete RGeom[i];
      delete IntPts[i];
   }
}

RefinedGeometry * GeometryRefiner::Refine (int Geom, int Times, int ETimes)
{
   int i, j, k, l;

   switch (Geom)
   {
   case Geometry::TRIANGLE:
   {
      if (RGeom[2] != NULL && RGeom[2]->Times == Times &&
          RGeom[2]->ETimes == ETimes)
         return RGeom[2];

      if (RGeom[2] != NULL)
         delete RGeom[2];
      RGeom[2] = new RefinedGeometry ((Times+1)*(Times+2)/2, 3*Times*Times,
                                      3*Times*(ETimes+1));
      RGeom[2]->Times = Times;
      RGeom[2]->ETimes = ETimes;
      for (k = j = 0; j <= Times; j++)
         for (i = 0; i <= Times-j; i++, k++)
         {
            IntegrationPoint &ip = RGeom[2]->RefPts.IntPoint(k);
            ip.x = double(i) / Times;
            ip.y = double(j) / Times;
         }
      Array<int> &G = RGeom[2]->RefGeoms;
      for (l = k = j = 0; j < Times; j++, k++)
         for (i = 0; i < Times-j; i++, k++)
         {
            G[l++] = k;
            G[l++] = k+1;
            G[l++] = k+Times-j+1;
            if (i+j+1 < Times)
            {
               G[l++] = k+1;
               G[l++] = k+Times-j+2;
               G[l++] = k+Times-j+1;
            }
         }
      Array<int> &E = RGeom[2]->RefEdges;
      // horizontal edges
      for (l = k = 0; k < Times; k += Times/ETimes)
      {
         j = k*(Times+1)-((k-1)*k)/2;
         for (i = 0; i < Times-k; i++)
         {
            E[l++] = j; j++;
            E[l++] = j;
         }
      }
      // diagonal edges
      for (k = Times; k > 0; k -= Times/ETimes)
      {
         j = k;
         for (i = 0; i < k; i++)
         {
            E[l++] = j; j += Times-i;
            E[l++] = j;
         }
      }
      // vertical edges
      for (k = 0; k < Times; k += Times/ETimes)
      {
         j = k;
         for (i = 0; i < Times-k; i++)
         {
            E[l++] = j; j += Times-i+1;
            E[l++] = j;
         }
      }

      return RGeom[2];
   }

   case Geometry::SQUARE:
   {
      if (RGeom[3] != NULL && RGeom[3]->Times == Times &&
          RGeom[3]->ETimes == ETimes)
         return RGeom[3];

      if (RGeom[3] != NULL)
         delete RGeom[3];
      RGeom[3] = new RefinedGeometry ((Times+1)*(Times+1), 4*Times*Times,
                                      4*(ETimes+1)*Times);
      RGeom[3]->Times = Times;
      RGeom[3]->ETimes = ETimes;
      for (k = j = 0; j <= Times; j++)
         for (i = 0; i <= Times; i++, k++)
         {
            IntegrationPoint &ip = RGeom[3]->RefPts.IntPoint(k);
            ip.x = double(i) / Times;
            ip.y = double(j) / Times;
         }
      Array<int> &G = RGeom[3]->RefGeoms;
      for (l = k = j = 0; j < Times; j++, k++)
         for (i = 0; i < Times; i++, k++)
         {
            G[l++] = k;
            G[l++] = k+1;
            G[l++] = k+Times+2;
            G[l++] = k+Times+1;
         }
      Array<int> &E = RGeom[3]->RefEdges;
      // horizontal edges
      for (l = k = 0; k <= Times; k += Times/ETimes)
      {
         for (i = 0, j = k*(Times+1); i < Times; i++)
         {
            E[l++] = j; j++;
            E[l++] = j;
         }
      }
      // vertical edges (in right-to-left order)
      for (k = Times; k >= 0; k -= Times/ETimes)
      {
         for (i = 0, j = k; i < Times; i++)
         {
            E[l++] = j; j += Times+1;
            E[l++] = j;
         }
      }

      return RGeom[3];
   }

   default:

      return RGeom[0];
   }
}

const IntegrationRule *GeometryRefiner::RefineInterior(int Geom, int Times)
{
   int g = Geom;

   switch (g)
   {
   case Geometry::SEGMENT:
   {
      if (Times < 2)
         return NULL;
      if (IntPts[g] == NULL || IntPts[g]->GetNPoints() != Times-1)
      {
         delete IntPts[g];
         IntPts[g] = new IntegrationRule(Times-1);
         for (int i = 1; i < Times; i++)
         {
            IntegrationPoint &ip = IntPts[g]->IntPoint(i-1);
            ip.x = double(i) / Times;
            ip.y = ip.z = 0.0;
         }
      }
   }
   break;

   case Geometry::TRIANGLE:
   {
      if (Times < 3)
         return NULL;
      if (IntPts[g] == NULL ||
          IntPts[g]->GetNPoints() != ((Times-1)*(Times-2))/2)
      {
         delete IntPts[g];
         IntPts[g] = new IntegrationRule(((Times-1)*(Times-2))/2);
         for (int k = 0, j = 1; j < Times-1; j++)
            for (int i = 1; i < Times-j; i++, k++)
            {
               IntegrationPoint &ip = IntPts[g]->IntPoint(k);
               ip.x = double(i) / Times;
               ip.y = double(j) / Times;
               ip.z = 0.0;
            }
      }
   }
   break;

   case Geometry::SQUARE:
   {
      if (Times < 2)
         return NULL;
      if (IntPts[g] == NULL || IntPts[g]->GetNPoints() != (Times-1)*(Times-1))
      {
         delete IntPts[g];
         IntPts[g] = new IntegrationRule((Times-1)*(Times-1));
         for (int k = 0, j = 1; j < Times; j++)
            for (int i = 1; i < Times; i++, k++)
            {
               IntegrationPoint &ip = IntPts[g]->IntPoint(k);
               ip.x = double(i) / Times;
               ip.y = double(j) / Times;
               ip.z = 0.0;
            }
      }
   }
   break;

   default:
      mfem_error("GeometryRefiner::RefineInterior(...)");
   }

   return IntPts[g];
}

GeometryRefiner GlobGeometryRefiner;
