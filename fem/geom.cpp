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

#include "fem.hpp"

namespace mfem
{

const char *Geometry::Name[NumGeom] =
{ "Point", "Segment", "Triangle", "Square", "Tetrahedron", "Cube" };

const double Geometry::Volume[NumGeom] =
{ 1.0, 1.0, 0.5, 1.0, 1./6, 1.0 };

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

// static method
void Geometry::GetRandomPoint(int GeomType, IntegrationPoint &ip)
{
   switch (GeomType)
   {
      case Geometry::POINT:
         ip.x = 0.0;
         break;
      case Geometry::SEGMENT:
         ip.x = double(rand()) / RAND_MAX;
         break;
      case Geometry::TRIANGLE:
         ip.x = double(rand()) / RAND_MAX;
         ip.y = double(rand()) / RAND_MAX;
         if (ip.x + ip.y > 1.0)
         {
            ip.x = 1.0 - ip.x;
            ip.y = 1.0 - ip.y;
         }
         break;
      case Geometry::SQUARE:
         ip.x = double(rand()) / RAND_MAX;
         ip.y = double(rand()) / RAND_MAX;
         break;
      case Geometry::TETRAHEDRON:
         ip.x = double(rand()) / RAND_MAX;
         ip.y = double(rand()) / RAND_MAX;
         ip.z = double(rand()) / RAND_MAX;
         // map to the triangular prism obtained by extruding the reference
         // triangle in z direction
         if (ip.x + ip.y > 1.0)
         {
            ip.x = 1.0 - ip.x;
            ip.y = 1.0 - ip.y;
         }
         // split the prism into 3 parts: 1 is the reference tet, and the
         // other two tets (as given below) are mapped to the reference tet
         if (ip.x + ip.z > 1.0)
         {
            // tet with vertices: (0,0,1),(1,0,1),(0,1,1),(1,0,0)
            ip.x = ip.x + ip.z - 1.0;
            // ip.y = ip.y;
            ip.z = 1.0 - ip.z;
            // mapped to: (0,0,0),(1,0,0),(0,1,0),(0,0,1)
         }
         else if (ip.x + ip.y + ip.z > 1.0)
         {
            // tet with vertices: (0,1,1),(0,1,0),(0,0,1),(1,0,0)
            double x = ip.x;
            ip.x = 1.0 - x - ip.z;
            ip.y = 1.0 - x - ip.y;
            ip.z = x;
            // mapped to: (0,0,0),(1,0,0),(0,1,0),(0,0,1)
         }
         break;
      case Geometry::CUBE:
         ip.x = double(rand()) / RAND_MAX;
         ip.y = double(rand()) / RAND_MAX;
         ip.z = double(rand()) / RAND_MAX;
         break;
      default:
         MFEM_ABORT("Unknown type of reference element!");
   }
}

// static method
bool Geometry::CheckPoint(int GeomType, const IntegrationPoint &ip)
{
   switch (GeomType)
   {
      case Geometry::POINT:
         if (ip.x != 0.0) { return false; }
         break;
      case Geometry::SEGMENT:
         if (ip.x < 0.0 || ip.x > 1.0) { return false; }
         break;
      case Geometry::TRIANGLE:
         if (ip.x < 0.0 || ip.y < 0.0 || ip.x+ip.y > 1.0) { return false; }
         break;
      case Geometry::SQUARE:
         if (ip.x < 0.0 || ip.x > 1.0 || ip.y < 0.0 || ip.y > 1.0)
         { return false; }
         break;
      case Geometry::TETRAHEDRON:
         if (ip.x < 0.0 || ip.y < 0.0 || ip.z < 0.0 ||
             ip.x+ip.y+ip.z > 1.0) { return false; }
         break;
      case Geometry::CUBE:
         if (ip.x < 0.0 || ip.x > 1.0 || ip.y < 0.0 || ip.y > 1.0 ||
             ip.z < 0.0 || ip.z > 1.0) { return false; }
         break;
      default:
         MFEM_ABORT("Unknown type of reference element!");
   }
   return true;
}

namespace internal
{

template <int N, int dim>
inline bool IntersectSegment(double lbeg[N], double lend[N],
                             IntegrationPoint &end)
{
   double t = 1.0;
   bool out = false;
   for (int i = 0; i < N; i++)
   {
      lbeg[i] = std::max(lbeg[i], 0.0); // remove round-off
      if (lend[i] < 0.0)
      {
         out = true;
         t = std::min(t, lbeg[i]/(lbeg[i]-lend[i]));
      }
   }
   if (out)
   {
      if (dim >= 1) { end.x = t*lend[0] + (1.0-t)*lbeg[0]; }
      if (dim >= 2) { end.y = t*lend[1] + (1.0-t)*lbeg[1]; }
      if (dim >= 3) { end.z = t*lend[2] + (1.0-t)*lbeg[2]; }
      return false;
   }
   return true;
}

}

// static method
bool Geometry::ProjectPoint(int GeomType, const IntegrationPoint &beg,
                            IntegrationPoint &end)
{
   switch (GeomType)
   {
      case Geometry::POINT:
      {
         if (end.x != 0.0) { end.x = 0.0; return false; }
         break;
      }
      case Geometry::SEGMENT:
      {
         if (end.x < 0.0) { end.x = 0.0; return false; }
         if (end.x > 1.0) { end.x = 1.0; return false; }
         break;
      }
      case Geometry::TRIANGLE:
      {
         double lend[3] = { end.x, end.y, 1-end.x-end.y };
         double lbeg[3] = { beg.x, beg.y, 1-beg.x-beg.y };
         return internal::IntersectSegment<3,2>(lbeg, lend, end);
      }
      case Geometry::SQUARE:
      {
         double lend[4] = { end.x, end.y, 1-end.x, 1.0-end.y };
         double lbeg[4] = { beg.x, beg.y, 1-beg.x, 1.0-beg.y };
         return internal::IntersectSegment<4,2>(lbeg, lend, end);
      }
      case Geometry::TETRAHEDRON:
      {
         double lend[4] = { end.x, end.y, end.z, 1.0-end.x-end.y-end.z };
         double lbeg[4] = { beg.x, beg.y, beg.z, 1.0-beg.x-beg.y-beg.z };
         return internal::IntersectSegment<4,3>(lbeg, lend, end);
      }
      case Geometry::CUBE:
      {
         double lend[6] = { end.x, end.y, end.z,
                            1.0-end.x, 1.0-end.y, 1.0-end.z
                          };
         double lbeg[6] = { beg.x, beg.y, beg.z,
                            1.0-beg.x, 1.0-beg.y, 1.0-beg.z
                          };
         return internal::IntersectSegment<6,3>(lbeg, lend, end);
      }
      default:
         MFEM_ABORT("Unknown type of reference element!");
   }
   return true;
}

void Geometry::GetPerfPointMat(int GeomType, DenseMatrix &pm)
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

const int Geometry::NumBdrArray[] = { 0, 2, 3, 4, 4, 6 };

Geometry Geometries;


GeometryRefiner::GeometryRefiner()
{
   type = 0;
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

   const double *cp = NULL;
   if (type)
   {
      cp = poly1d.ClosedPoints(Times);
   }

   switch (Geom)
   {
      case Geometry::SEGMENT:
      {
         const int g = Geometry::SEGMENT;
         if (RGeom[g] != NULL && RGeom[g]->Times == Times)
         {
            return RGeom[g];
         }
         delete RGeom[g];
         RGeom[g] = new RefinedGeometry(Times+1, 2*Times, 0);
         RGeom[g]->Times = Times;
         RGeom[g]->ETimes = 0;
         for (i = 0; i <= Times; i++)
         {
            IntegrationPoint &ip = RGeom[g]->RefPts.IntPoint(i);
            ip.x = (type == 0) ? double(i) / Times : cp[i];
         }
         Array<int> &G = RGeom[g]->RefGeoms;
         for (i = 0; i < Times; i++)
         {
            G[2*i+0] = i;
            G[2*i+1] = i+1;
         }

         return RGeom[g];
      }

      case Geometry::TRIANGLE:
      {
         if (RGeom[2] != NULL && RGeom[2]->Times == Times &&
             RGeom[2]->ETimes == ETimes)
         {
            return RGeom[2];
         }

         if (RGeom[2] != NULL)
         {
            delete RGeom[2];
         }
         RGeom[2] = new RefinedGeometry ((Times+1)*(Times+2)/2, 3*Times*Times,
                                         3*Times*(ETimes+1));
         RGeom[2]->Times = Times;
         RGeom[2]->ETimes = ETimes;
         for (k = j = 0; j <= Times; j++)
            for (i = 0; i <= Times-j; i++, k++)
            {
               IntegrationPoint &ip = RGeom[2]->RefPts.IntPoint(k);
               if (type == 0)
               {
                  ip.x = double(i) / Times;
                  ip.y = double(j) / Times;
               }
               else
               {
                  ip.x = cp[i]/(cp[i] + cp[j] + cp[Times-i-j]);
                  ip.y = cp[j]/(cp[i] + cp[j] + cp[Times-i-j]);
               }
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
         {
            return RGeom[3];
         }

         if (RGeom[3] != NULL)
         {
            delete RGeom[3];
         }
         RGeom[3] = new RefinedGeometry ((Times+1)*(Times+1), 4*Times*Times,
                                         4*(ETimes+1)*Times);
         RGeom[3]->Times = Times;
         RGeom[3]->ETimes = ETimes;
         for (k = j = 0; j <= Times; j++)
            for (i = 0; i <= Times; i++, k++)
            {
               IntegrationPoint &ip = RGeom[3]->RefPts.IntPoint(k);
               if (type == 0)
               {
                  ip.x = double(i) / Times;
                  ip.y = double(j) / Times;
               }
               else
               {
                  ip.x = cp[i];
                  ip.y = cp[j];
               }
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

      case Geometry::CUBE:
      {
         const int g = Geometry::CUBE;
         if (RGeom[g] != NULL && RGeom[g]->Times == Times &&
             RGeom[g]->ETimes == ETimes)
         {
            return RGeom[g];
         }

         if (RGeom[g] != NULL)
         {
            delete RGeom[g];
         }
         RGeom[g] = new RefinedGeometry ((Times+1)*(Times+1)*(Times+1),
                                         8*Times*Times*Times, 0);
         RGeom[g]->Times = Times;
         RGeom[g]->ETimes = ETimes;
         for (l = k = 0; k <= Times; k++)
            for (j = 0; j <= Times; j++)
               for (i = 0; i <= Times; i++, l++)
               {
                  IntegrationPoint &ip = RGeom[g]->RefPts.IntPoint(l);
                  if (type == 0)
                  {
                     ip.x = double(i) / Times;
                     ip.y = double(j) / Times;
                     ip.z = double(k) / Times;
                  }
                  else
                  {
                     ip.x = cp[i];
                     ip.y = cp[j];
                     ip.z = cp[k];
                  }
               }
         Array<int> &G = RGeom[g]->RefGeoms;
         for (l = k = 0; k < Times; k++)
            for (j = 0; j < Times; j++)
               for (i = 0; i < Times; i++)
               {
                  G[l++] = i+0 + (j+0 + (k+0) * (Times+1)) * (Times+1);
                  G[l++] = i+1 + (j+0 + (k+0) * (Times+1)) * (Times+1);
                  G[l++] = i+1 + (j+1 + (k+0) * (Times+1)) * (Times+1);
                  G[l++] = i+0 + (j+1 + (k+0) * (Times+1)) * (Times+1);
                  G[l++] = i+0 + (j+0 + (k+1) * (Times+1)) * (Times+1);
                  G[l++] = i+1 + (j+0 + (k+1) * (Times+1)) * (Times+1);
                  G[l++] = i+1 + (j+1 + (k+1) * (Times+1)) * (Times+1);
                  G[l++] = i+0 + (j+1 + (k+1) * (Times+1)) * (Times+1);
               }

         return RGeom[g];
      }

      case Geometry::TETRAHEDRON:
      {
         const int g = Geometry::TETRAHEDRON;
         if (RGeom[g] != NULL && RGeom[g]->Times == Times &&
             RGeom[g]->ETimes == ETimes)
         {
            return RGeom[g];
         }

         if (RGeom[g] != NULL)
         {
            delete RGeom[g];
         }

         // subdivide the tetrahedron with vertices
         // (0,0,0), (0,0,1), (1,1,1), (0,1,1)

         // vertices: 0 <= i <= j <= k <= Times
         // (3-combination with repetitions)
         // number of vertices: (n+3)*(n+2)*(n+1)/6, n = Times

         // elements: the vertices are: v1=(i,j,k), v2=v1+u1, v3=v2+u2, v4=v3+u3
         // where 0 <= i <= j <= k <= n-1 and
         // u1,u2,u3 is a permutation of (1,0,0),(0,1,0),(0,0,1)
         // such that all v2,v3,v4 have non-decreasing components
         // number of elements: n^3

         const int n = Times;
         RGeom[g] = new RefinedGeometry((n+3)*(n+2)*(n+1)/6, 4*n*n*n, 0);
         // enumerate and define the vertices
         Array<int> vi((n+1)*(n+1)*(n+1));
         vi = -1;
         int m = 0;
         for (k = 0; k <= n; k++)
            for (j = 0; j <= k; j++)
               for (i = 0; i <= j; i++)
               {
                  IntegrationPoint &ip = RGeom[g]->RefPts.IntPoint(m);
                  // map the coordinates to the reference tetrahedron
                  // (0,0,0) -> (0,0,0)
                  // (0,0,1) -> (1,0,0)
                  // (1,1,1) -> (0,1,0)
                  // (0,1,1) -> (0,0,1)
                  if (type == 0)
                  {
                     ip.x = double(k - j) / n;
                     ip.y = double(i) / n;
                     ip.z = double(j - i) / n;
                  }
                  else
                  {
                     double w = cp[k-j] + cp[i] + cp[j-i] + cp[Times-k];
                     ip.x = cp[k-j]/w;
                     ip.y = cp[i]/w;
                     ip.z = cp[j-i]/w;
                  }
                  l = i + (j + k * (n+1)) * (n+1);
                  vi[l] = m;
                  m++;
               }
         if (m != (n+3)*(n+2)*(n+1)/6)
         {
            mfem_error("GeometryRefiner::Refine() for TETRAHEDRON #1");
         }
         // elements
         Array<int> &G = RGeom[g]->RefGeoms;
         m = 0;
         for (k = 0; k < n; k++)
            for (j = 0; j <= k; j++)
               for (i = 0; i <= j; i++)
               {
                  // the ordering of the vertices is chosen to ensure:
                  // 1) correct orientation
                  // 2) the x,y,z edges are in the set of edges
                  //    {(0,1),(2,3), (0,2),(1,3)}
                  //    (goal is to ensure that subsequent refinement using
                  //    this procedure preserves the six tetrahedral shapes)

                  // zyx: (i,j,k)-(i,j,k+1)-(i+1,j+1,k+1)-(i,j+1,k+1)
                  G[m++] = vi[i+0 + (j+0 + (k+0) * (n+1)) * (n+1)];
                  G[m++] = vi[i+0 + (j+0 + (k+1) * (n+1)) * (n+1)];
                  G[m++] = vi[i+1 + (j+1 + (k+1) * (n+1)) * (n+1)];
                  G[m++] = vi[i+0 + (j+1 + (k+1) * (n+1)) * (n+1)];
                  if (j < k)
                  {
                     // yzx: (i,j,k)-(i+1,j+1,k+1)-(i,j+1,k)-(i,j+1,k+1)
                     G[m++] = vi[i+0 + (j+0 + (k+0) * (n+1)) * (n+1)];
                     G[m++] = vi[i+1 + (j+1 + (k+1) * (n+1)) * (n+1)];
                     G[m++] = vi[i+0 + (j+1 + (k+0) * (n+1)) * (n+1)];
                     G[m++] = vi[i+0 + (j+1 + (k+1) * (n+1)) * (n+1)];
                     // yxz: (i,j,k)-(i,j+1,k)-(i+1,j+1,k+1)-(i+1,j+1,k)
                     G[m++] = vi[i+0 + (j+0 + (k+0) * (n+1)) * (n+1)];
                     G[m++] = vi[i+0 + (j+1 + (k+0) * (n+1)) * (n+1)];
                     G[m++] = vi[i+1 + (j+1 + (k+1) * (n+1)) * (n+1)];
                     G[m++] = vi[i+1 + (j+1 + (k+0) * (n+1)) * (n+1)];
                  }
                  if (i < j)
                  {
                     // xzy: (i,j,k)-(i+1,j,k)-(i+1,j+1,k+1)-(i+1,j,k+1)
                     G[m++] = vi[i+0 + (j+0 + (k+0) * (n+1)) * (n+1)];
                     G[m++] = vi[i+1 + (j+0 + (k+0) * (n+1)) * (n+1)];
                     G[m++] = vi[i+1 + (j+1 + (k+1) * (n+1)) * (n+1)];
                     G[m++] = vi[i+1 + (j+0 + (k+1) * (n+1)) * (n+1)];
                     if (j < k)
                     {
                        // xyz: (i,j,k)-(i+1,j+1,k+1)-(i+1,j,k)-(i+1,j+1,k)
                        G[m++] = vi[i+0 + (j+0 + (k+0) * (n+1)) * (n+1)];
                        G[m++] = vi[i+1 + (j+1 + (k+1) * (n+1)) * (n+1)];
                        G[m++] = vi[i+1 + (j+0 + (k+0) * (n+1)) * (n+1)];
                        G[m++] = vi[i+1 + (j+1 + (k+0) * (n+1)) * (n+1)];
                     }
                     // zxy: (i,j,k)-(i+1,j+1,k+1)-(i,j,k+1)-(i+1,j,k+1)
                     G[m++] = vi[i+0 + (j+0 + (k+0) * (n+1)) * (n+1)];
                     G[m++] = vi[i+1 + (j+1 + (k+1) * (n+1)) * (n+1)];
                     G[m++] = vi[i+0 + (j+0 + (k+1) * (n+1)) * (n+1)];
                     G[m++] = vi[i+1 + (j+0 + (k+1) * (n+1)) * (n+1)];
                  }
               }
         if (m != 4*n*n*n)
         {
            mfem_error("GeometryRefiner::Refine() for TETRAHEDRON #2");
         }
         for (i = 0; i < m; i++)
            if (G[i] < 0)
            {
               mfem_error("GeometryRefiner::Refine() for TETRAHEDRON #3");
            }

         return RGeom[g];
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
         {
            return NULL;
         }
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
         {
            return NULL;
         }
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
         {
            return NULL;
         }
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

}
