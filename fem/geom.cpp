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

#include "fem.hpp"
#include "../mesh/wedge.hpp"

namespace mfem
{

const char *Geometry::Name[NumGeom] =
{ "Point", "Segment", "Triangle", "Square", "Tetrahedron", "Cube", "Prism" };

const double Geometry::Volume[NumGeom] =
{ 1.0, 1.0, 0.5, 1.0, 1./6, 1.0, 0.5 };

Geometry::Geometry()
{
   // Vertices for Geometry::POINT
   GeomVert[0] =  new IntegrationRule(1);
   GeomVert[0]->IntPoint(0).x = 0.0;

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

   // Vertices for Geometry::PRISM
   GeomVert[6] = new IntegrationRule(6);
   GeomVert[6]->IntPoint(0).x = 0.0;
   GeomVert[6]->IntPoint(0).y = 0.0;
   GeomVert[6]->IntPoint(0).z = 0.0;

   GeomVert[6]->IntPoint(1).x = 1.0;
   GeomVert[6]->IntPoint(1).y = 0.0;
   GeomVert[6]->IntPoint(1).z = 0.0;

   GeomVert[6]->IntPoint(2).x = 0.0;
   GeomVert[6]->IntPoint(2).y = 1.0;
   GeomVert[6]->IntPoint(2).z = 0.0;

   GeomVert[6]->IntPoint(3).x = 0.0;
   GeomVert[6]->IntPoint(3).y = 0.0;
   GeomVert[6]->IntPoint(3).z = 1.0;

   GeomVert[6]->IntPoint(4).x = 1.0;
   GeomVert[6]->IntPoint(4).y = 0.0;
   GeomVert[6]->IntPoint(4).z = 1.0;

   GeomVert[6]->IntPoint(5).x = 0.0;
   GeomVert[6]->IntPoint(5).y = 1.0;
   GeomVert[6]->IntPoint(5).z = 1.0;

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

   GeomCenter[PRISM].x = 1.0 / 3.0;
   GeomCenter[PRISM].y = 1.0 / 3.0;
   GeomCenter[PRISM].z = 0.5;

   GeomToPerfGeomJac[POINT]       = NULL;
   GeomToPerfGeomJac[SEGMENT]     = new DenseMatrix(1);
   GeomToPerfGeomJac[TRIANGLE]    = new DenseMatrix(2);
   GeomToPerfGeomJac[SQUARE]      = new DenseMatrix(2);
   GeomToPerfGeomJac[TETRAHEDRON] = new DenseMatrix(3);
   GeomToPerfGeomJac[CUBE]        = new DenseMatrix(3);
   GeomToPerfGeomJac[PRISM]       = new DenseMatrix(3);

   PerfGeomToGeomJac[POINT]       = NULL;
   PerfGeomToGeomJac[SEGMENT]     = NULL;
   PerfGeomToGeomJac[TRIANGLE]    = new DenseMatrix(2);
   PerfGeomToGeomJac[SQUARE]      = NULL;
   PerfGeomToGeomJac[TETRAHEDRON] = new DenseMatrix(3);
   PerfGeomToGeomJac[CUBE]        = NULL;
   PerfGeomToGeomJac[PRISM]       = new DenseMatrix(3);

   GeomToPerfGeomJac[SEGMENT]->Diag(1.0, 1);
   {
      IsoparametricTransformation tri_T;
      tri_T.SetFE(&TriangleFE);
      GetPerfPointMat (TRIANGLE, tri_T.GetPointMat());
      tri_T.SetIntPoint(&GeomCenter[TRIANGLE]);
      *GeomToPerfGeomJac[TRIANGLE] = tri_T.Jacobian();
      CalcInverse(tri_T.Jacobian(), *PerfGeomToGeomJac[TRIANGLE]);
   }
   GeomToPerfGeomJac[SQUARE]->Diag(1.0, 2);
   {
      IsoparametricTransformation tet_T;
      tet_T.SetFE(&TetrahedronFE);
      GetPerfPointMat (TETRAHEDRON, tet_T.GetPointMat());
      tet_T.SetIntPoint(&GeomCenter[TETRAHEDRON]);
      *GeomToPerfGeomJac[TETRAHEDRON] = tet_T.Jacobian();
      CalcInverse(tet_T.Jacobian(), *PerfGeomToGeomJac[TETRAHEDRON]);
   }
   GeomToPerfGeomJac[CUBE]->Diag(1.0, 3);
   {
      IsoparametricTransformation pri_T;
      pri_T.SetFE(&WedgeFE);
      GetPerfPointMat (PRISM, pri_T.GetPointMat());
      pri_T.SetIntPoint(&GeomCenter[PRISM]);
      *GeomToPerfGeomJac[PRISM] = pri_T.Jacobian();
      CalcInverse(pri_T.Jacobian(), *PerfGeomToGeomJac[PRISM]);
   }
}

Geometry::~Geometry()
{
   for (int i = 0; i < NumGeom; i++)
   {
      delete PerfGeomToGeomJac[i];
      delete GeomToPerfGeomJac[i];
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
      case Geometry::PRISM:       return GeomVert[6];
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
      case Geometry::PRISM:
         ip.x = double(rand()) / RAND_MAX;
         ip.y = double(rand()) / RAND_MAX;
         ip.z = double(rand()) / RAND_MAX;
         if (ip.x + ip.y > 1.0)
         {
            ip.x = 1.0 - ip.x;
            ip.y = 1.0 - ip.y;
         }
         break;
      default:
         MFEM_ABORT("Unknown type of reference element!");
   }
}


namespace internal
{

// Fuzzy equality operator with absolute tolerance eps.
inline bool NearlyEqual(double x, double y, double eps)
{
   return std::abs(x-y) <= eps;
}

// Fuzzy greater than comparison operator with absolute tolerance eps.
// Returns true when x is greater than y by at least eps.
inline bool FuzzyGT(double x, double y, double eps)
{
   return (x > y + eps);
}

// Fuzzy less than comparison operator with absolute tolerance eps.
// Returns true when x is less than y by at least eps.
inline bool FuzzyLT(double x, double y, double eps)
{
   return (x < y - eps);
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
      case Geometry::PRISM:
         if (ip.x < 0.0 || ip.y < 0.0 || ip.x+ip.y > 1.0 ||
             ip.z < 0.0 || ip.z > 1.0) { return false; }
         break;
      default:
         MFEM_ABORT("Unknown type of reference element!");
   }
   return true;
}

bool Geometry::CheckPoint(int GeomType, const IntegrationPoint &ip, double eps)
{
   switch (GeomType)
   {
      case Geometry::POINT:
         if (! internal::NearlyEqual(ip.x, 0.0, eps))
         {
            return false;
         }
         break;
      case Geometry::SEGMENT:
         if ( internal::FuzzyLT(ip.x, 0.0, eps)
              || internal::FuzzyGT(ip.x, 1.0, eps) )
         {
            return false;
         }
         break;
      case Geometry::TRIANGLE:
         if ( internal::FuzzyLT(ip.x, 0.0, eps)
              || internal::FuzzyLT(ip.y, 0.0, eps)
              || internal::FuzzyGT(ip.x+ip.y, 1.0, eps) )
         {
            return false;
         }
         break;
      case Geometry::SQUARE:
         if ( internal::FuzzyLT(ip.x, 0.0, eps)
              || internal::FuzzyGT(ip.x, 1.0, eps)
              || internal::FuzzyLT(ip.y, 0.0, eps)
              || internal::FuzzyGT(ip.y, 1.0, eps) )
         {
            return false;
         }
         break;
      case Geometry::TETRAHEDRON:
         if ( internal::FuzzyLT(ip.x, 0.0, eps)
              || internal::FuzzyLT(ip.y, 0.0, eps)
              || internal::FuzzyLT(ip.z, 0.0, eps)
              || internal::FuzzyGT(ip.x+ip.y+ip.z, 1.0, eps) )
         {
            return false;
         }
         break;
      case Geometry::CUBE:
         if ( internal::FuzzyLT(ip.x, 0.0, eps)
              || internal::FuzzyGT(ip.x, 1.0, eps)
              || internal::FuzzyLT(ip.y, 0.0, eps)
              || internal::FuzzyGT(ip.y, 1.0, eps)
              || internal::FuzzyLT(ip.z, 0.0, eps)
              || internal::FuzzyGT(ip.z, 1.0, eps) )
         {
            return false;
         }
         break;
      case Geometry::PRISM:
         if ( internal::FuzzyLT(ip.x, 0.0, eps)
              || internal::FuzzyLT(ip.y, 0.0, eps)
              || internal::FuzzyGT(ip.x+ip.y, 1.0, eps)
              || internal::FuzzyLT(ip.z, 0.0, eps)
              || internal::FuzzyGT(ip.z, 1.0, eps) )
         {
            return false;
         }
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

inline bool ProjectTriangle(double &x, double &y)
{
   if (x < 0.0)
   {
      x = 0.0;
      if (y < 0.0)      { y = 0.0; }
      else if (y > 1.0) { y = 1.0; }
      return false;
   }
   if (y < 0.0)
   {
      if (x > 1.0) { x = 1.0; }
      y = 0.0;
      return false;
   }
   const double l3 = 1.0-x-y;
   if (l3 < 0.0)
   {
      if (y - x > 1.0)       { x = 0.0; y = 1.0; }
      else if (y - x < -1.0) { x = 1.0; y = 0.0; }
      else                   { x += l3/2; y += l3/2; }
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
         double lend[3] = { end.x, end.y, 1.0-end.x-end.y };
         double lbeg[3] = { beg.x, beg.y, 1.0-beg.x-beg.y };
         return internal::IntersectSegment<3,2>(lbeg, lend, end);
      }
      case Geometry::SQUARE:
      {
         double lend[4] = { end.x, end.y, 1.0-end.x, 1.0-end.y };
         double lbeg[4] = { beg.x, beg.y, 1.0-beg.x, 1.0-beg.y };
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
      case Geometry::PRISM:
      {
         double lend[5] = { end.x, end.y, end.z, 1.0-end.x-end.y, 1.0-end.z };
         double lbeg[5] = { beg.x, beg.y, beg.z, 1.0-beg.x-beg.y, 1.0-beg.z };
         return internal::IntersectSegment<5,3>(lbeg, lend, end);
      }
      default:
         MFEM_ABORT("Unknown type of reference element!");
   }
   return true;
}

// static method
bool Geometry::ProjectPoint(int GeomType, IntegrationPoint &ip)
{
   // If ip is outside the element, replace it with the point on the boundary
   // that is closest to the original ip and return false; otherwise, return
   // true without changing ip.

   switch (GeomType)
   {
      case SEGMENT:
      {
         if (ip.x < 0.0)      { ip.x = 0.0; return false; }
         else if (ip.x > 1.0) { ip.x = 1.0; return false; }
         return true;
      }

      case TRIANGLE:
      {
         return internal::ProjectTriangle(ip.x, ip.y);
      }

      case SQUARE:
      {
         bool in_x, in_y;
         if (ip.x < 0.0)      { in_x = false; ip.x = 0.0; }
         else if (ip.x > 1.0) { in_x = false; ip.x = 1.0; }
         else                 { in_x = true; }
         if (ip.y < 0.0)      { in_y = false; ip.y = 0.0; }
         else if (ip.y > 1.0) { in_y = false; ip.y = 1.0; }
         else                 { in_y = true; }
         return in_x && in_y;
      }

      case TETRAHEDRON:
      {
         if (ip.z < 0.0)
         {
            ip.z = 0.0;
            internal::ProjectTriangle(ip.x, ip.y);
            return false;
         }
         if (ip.y < 0.0)
         {
            ip.y = 0.0;
            internal::ProjectTriangle(ip.x, ip.z);
            return false;
         }
         if (ip.x < 0.0)
         {
            ip.x = 0.0;
            internal::ProjectTriangle(ip.y, ip.z);
            return false;
         }
         const double l4 = 1.0-ip.x-ip.y-ip.z;
         if (l4 < 0.0)
         {
            const double l4_3 = l4/3;
            ip.x += l4_3;
            ip.y += l4_3;
            internal::ProjectTriangle(ip.x, ip.y);
            ip.z = 1.0-ip.x-ip.y;
            return false;
         }
         return true;
      }

      case CUBE:
      {
         bool in_x, in_y, in_z;
         if (ip.x < 0.0)      { in_x = false; ip.x = 0.0; }
         else if (ip.x > 1.0) { in_x = false; ip.x = 1.0; }
         else                 { in_x = true; }
         if (ip.y < 0.0)      { in_y = false; ip.y = 0.0; }
         else if (ip.y > 1.0) { in_y = false; ip.y = 1.0; }
         else                 { in_y = true; }
         if (ip.z < 0.0)      { in_z = false; ip.z = 0.0; }
         else if (ip.z > 1.0) { in_z = false; ip.z = 1.0; }
         else                 { in_z = true; }
         return in_x && in_y && in_z;
      }

      case PRISM:
      {
         bool in_tri, in_z;
         in_tri = internal::ProjectTriangle(ip.x, ip.y);
         if (ip.z < 0.0)      { in_z = false; ip.z = 0.0; }
         else if (ip.z > 1.0) { in_z = false; ip.z = 1.0; }
         else                 { in_z = true; }
         return in_tri && in_z;
      }

      default:
         MFEM_ABORT("Reference element type is not supported!");
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

      case Geometry::PRISM:
      {
         pm.SetSize (3, 6);
         pm(0,0) = 0.0;  pm(1,0) = 0.0;  pm(2,0) = 0.0;
         pm(0,1) = 1.0;  pm(1,1) = 0.0;  pm(2,1) = 0.0;
         pm(0,2) = 0.5;  pm(1,2) = 0.86602540378443864676;  pm(2,2) = 0.0;
         pm(0,3) = 0.0;  pm(1,3) = 0.0;  pm(2,3) = 1.0;
         pm(0,4) = 1.0;  pm(1,4) = 0.0;  pm(2,4) = 1.0;
         pm(0,5) = 0.5;  pm(1,5) = 0.86602540378443864676;  pm(2,5) = 1.0;
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

const int Geometry::NumBdrArray[NumGeom] = { 0, 2, 3, 4, 4, 6, 5 };
const int Geometry::Dimension[NumGeom] = { 0, 1, 2, 2, 3, 3, 3 };
const int Geometry::DimStart[MaxDim+2] =
{ POINT, SEGMENT, TRIANGLE, TETRAHEDRON, NUM_GEOMETRIES };
const int Geometry::NumVerts[NumGeom] = { 1, 2, 3, 4, 4, 8, 6 };
const int Geometry::NumEdges[NumGeom] = { 0, 1, 3, 4, 6, 12, 9 };
const int Geometry::NumFaces[NumGeom] = { 0, 0, 1, 1, 4, 6, 5 };

const int Geometry::
Constants<Geometry::POINT>::Orient[1][1] = {{0}};
const int Geometry::
Constants<Geometry::POINT>::InvOrient[1] = {0};

const int Geometry::
Constants<Geometry::SEGMENT>::Edges[1][2] = { {0, 1} };
const int Geometry::
Constants<Geometry::SEGMENT>::Orient[2][2] = { {0, 1}, {1, 0} };
const int Geometry::
Constants<Geometry::SEGMENT>::InvOrient[2] = { 0, 1 };

const int Geometry::
Constants<Geometry::TRIANGLE>::Edges[3][2] = {{0, 1}, {1, 2}, {2, 0}};
const int Geometry::
Constants<Geometry::TRIANGLE>::VertToVert::I[3] = {0, 2, 3};
const int Geometry::
Constants<Geometry::TRIANGLE>::VertToVert::J[3][2] = {{1, 0}, {2, -3}, {2, 1}};
const int Geometry::
Constants<Geometry::TRIANGLE>::FaceVert[1][3] = {{0, 1, 2}};
const int Geometry::
Constants<Geometry::TRIANGLE>::Orient[6][3] =
{
   {0, 1, 2}, {1, 0, 2}, {2, 0, 1},
   {2, 1, 0}, {1, 2, 0}, {0, 2, 1}
};
const int Geometry::
Constants<Geometry::TRIANGLE>::InvOrient[6] = { 0, 1, 4, 3, 2, 5 };

const int Geometry::
Constants<Geometry::SQUARE>::Edges[4][2] = {{0, 1}, {1, 2}, {2, 3}, {3, 0}};
const int Geometry::
Constants<Geometry::SQUARE>::VertToVert::I[4] = {0, 2, 3, 4};
const int Geometry::
Constants<Geometry::SQUARE>::VertToVert::J[4][2] =
{{1, 0}, {3, -4}, {2, 1}, {3, 2}};
const int Geometry::
Constants<Geometry::SQUARE>::FaceVert[1][4] = {{0, 1, 2, 3}};
const int Geometry::
Constants<Geometry::SQUARE>::Orient[8][4] =
{
   {0, 1, 2, 3}, {0, 3, 2, 1}, {1, 2, 3, 0}, {1, 0, 3, 2},
   {2, 3, 0, 1}, {2, 1, 0, 3}, {3, 0, 1, 2}, {3, 2, 1, 0}
};
const int Geometry::
Constants<Geometry::SQUARE>::InvOrient[8] = { 0, 1, 6, 3, 4, 5, 2, 7 };

const int Geometry::
Constants<Geometry::TETRAHEDRON>::Edges[6][2] =
{{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}};
const int Geometry::
Constants<Geometry::TETRAHEDRON>::FaceTypes[4] =
{
   Geometry::TRIANGLE, Geometry::TRIANGLE,
   Geometry::TRIANGLE, Geometry::TRIANGLE
};
const int Geometry::
Constants<Geometry::TETRAHEDRON>::FaceVert[4][3] =
{{1, 2, 3}, {0, 3, 2}, {0, 1, 3}, {0, 2, 1}};
const int Geometry::
Constants<Geometry::TETRAHEDRON>::VertToVert::I[4] = {0, 3, 5, 6};
const int Geometry::
Constants<Geometry::TETRAHEDRON>::VertToVert::J[6][2] =
{
   {1, 0}, {2, 1}, {3, 2}, // 0,1:0   0,2:1   0,3:2
   {2, 3}, {3, 4},         // 1,2:3   1,3:4
   {3, 5}                  // 2,3:5
};
const int Geometry::
Constants<Geometry::TETRAHEDRON>::Orient[24][4] =
{
   {0, 1, 2, 3}, {0, 1, 3, 2}, {0, 2, 3, 1}, {0, 2, 1, 3},
   {0, 3, 1, 2}, {0, 3, 2, 1},
   {1, 2, 0, 3}, {1, 2, 3, 0}, {1, 3, 2, 0}, {1, 3, 0, 2},
   {1, 0, 3, 2}, {1, 0, 2, 3},
   {2, 3, 0, 1}, {2, 3, 1, 0}, {2, 0, 1, 3}, {2, 0, 3, 1},
   {2, 1, 3, 0}, {2, 1, 0, 3},
   {3, 0, 2, 1}, {3, 0, 1, 2}, {3, 1, 0, 2}, {3, 1, 2, 0},
   {3, 2, 1, 0}, {3, 2, 0, 1}
};
const int Geometry::
Constants<Geometry::TETRAHEDRON>::InvOrient[24] =
{
   0,   1,  4,  3,  2,  5,
   14, 19, 18, 15, 10, 11,
   12, 23,  6,  9, 20, 17,
   8,   7, 16, 21, 22, 13
};

const int Geometry::
Constants<Geometry::CUBE>::Edges[12][2] =
{
   {0, 1}, {1, 2}, {3, 2}, {0, 3}, {4, 5}, {5, 6},
   {7, 6}, {4, 7}, {0, 4}, {1, 5}, {2, 6}, {3, 7}
};
const int Geometry::
Constants<Geometry::CUBE>::FaceTypes[6] =
{
   Geometry::SQUARE, Geometry::SQUARE, Geometry::SQUARE,
   Geometry::SQUARE, Geometry::SQUARE, Geometry::SQUARE
};
const int Geometry::
Constants<Geometry::CUBE>::FaceVert[6][4] =
{
   {3, 2, 1, 0}, {0, 1, 5, 4}, {1, 2, 6, 5},
   {2, 3, 7, 6}, {3, 0, 4, 7}, {4, 5, 6, 7}
};
const int Geometry::
Constants<Geometry::CUBE>::VertToVert::I[8] = {0, 3, 5, 7, 8, 10, 11, 12};
const int Geometry::
Constants<Geometry::CUBE>::VertToVert::J[12][2] =
{
   {1, 0}, {3, 3}, {4, 8}, // 0,1:0   0,3:3   0,4:8
   {2, 1}, {5, 9},         // 1,2:1   1,5:9
   {3,-3}, {6,10},         // 2,3:-3  2,6:10
   {7,11},                 // 3,7:11
   {5, 4}, {7, 7},         // 4,5:4   4,7:7
   {6, 5},                 // 5,6:5
   {7,-7}                  // 6,7:-7
};

const int Geometry::
Constants<Geometry::PRISM>::Edges[9][2] =
{{0, 1}, {1, 2}, {2, 0}, {3, 4}, {4, 5}, {5, 3}, {0, 3}, {1, 4}, {2, 5}};
const int Geometry::
Constants<Geometry::PRISM>::FaceTypes[5] =
{
   Geometry::TRIANGLE, Geometry::TRIANGLE,
   Geometry::SQUARE, Geometry::SQUARE, Geometry::SQUARE
};
const int Geometry::
Constants<Geometry::PRISM>::FaceVert[5][4] =
{{0, 2, 1, -1}, {3, 4, 5, -1}, {0, 1, 4, 3}, {1, 2, 5, 4}, {2, 0, 3, 5}};
const int Geometry::
Constants<Geometry::PRISM>::VertToVert::I[6] = {0, 3, 5, 6, 8, 9};
const int Geometry::
Constants<Geometry::PRISM>::VertToVert::J[9][2] =
{
   {1, 0}, {2, -3}, {3, 6}, // 0,1:0   0,2:-3  0,3:6
   {2, 1}, {4, 7},          // 1,2:1   1,4:7
   {5, 8},                  // 2,5:8
   {4, 3}, {5, -6},         // 3,4:3   3,5:-6
   {5, 4}                   // 4,5:4
};


GeometryRefiner::GeometryRefiner()
{
   type = Quadrature1D::ClosedUniform;
}

GeometryRefiner::~GeometryRefiner()
{
   for (int i = 0; i < Geometry::NumGeom; i++)
   {
      for (int j = 0; j < RGeom[i].Size(); j++) { delete RGeom[i][j]; }
      for (int j = 0; j < IntPts[i].Size(); j++) { delete IntPts[i][j]; }
   }
}

RefinedGeometry *GeometryRefiner::FindInRGeom(Geometry::Type Geom,
                                              int Times, int ETimes,
                                              int Type)
{
   Array<RefinedGeometry *> &RGA = RGeom[Geom];
   for (int i = 0; i < RGA.Size(); i++)
   {
      RefinedGeometry &RG = *RGA[i];
      if (RG.Times == Times && RG.ETimes == ETimes && RG.Type == Type)
      {
         return &RG;
      }
   }
   return NULL;
}

IntegrationRule *GeometryRefiner::FindInIntPts(Geometry::Type Geom, int NPts)
{
   Array<IntegrationRule *> &IPA = IntPts[Geom];
   for (int i = 0; i < IPA.Size(); i++)
   {
      IntegrationRule &ir = *IPA[i];
      if (ir.GetNPoints() == NPts) { return &ir; }
   }
   return NULL;
}

RefinedGeometry * GeometryRefiner::Refine(Geometry::Type Geom,
                                          int Times, int ETimes)
{
   int i, j, k, l, m;

   Times = std::max(Times, 1);
   ETimes = std::max(ETimes, 1);
   const double *cp = poly1d.GetPoints(Times, BasisType::GetNodalBasis(type));

   RefinedGeometry *RG = FindInRGeom(Geom, Times, ETimes, type);
   if (RG) { return RG; }

   switch (Geom)
   {
      case Geometry::POINT:
      {
         RG = new RefinedGeometry(1, 1, 0);
         RG->Times = 1;
         RG->ETimes = 0;
         RG->Type = type;
         RG->RefPts.IntPoint(0).x = cp[0];
         RG->RefGeoms[0] = 0;

         RGeom[Geometry::POINT].Append(RG);
         return RG;
      }

      case Geometry::SEGMENT:
      {
         RG = new RefinedGeometry(Times+1, 2*Times, 0);
         RG->Times = Times;
         RG->ETimes = 0;
         RG->Type = type;
         for (i = 0; i <= Times; i++)
         {
            IntegrationPoint &ip = RG->RefPts.IntPoint(i);
            ip.x = cp[i];
         }
         Array<int> &G = RG->RefGeoms;
         for (i = 0; i < Times; i++)
         {
            G[2*i+0] = i;
            G[2*i+1] = i+1;
         }

         RGeom[Geometry::SEGMENT].Append(RG);
         return RG;
      }

      case Geometry::TRIANGLE:
      {
         RG = new RefinedGeometry((Times+1)*(Times+2)/2, 3*Times*Times,
                                  3*Times*(ETimes+1), 3*Times);
         RG->Times = Times;
         RG->ETimes = ETimes;
         RG->Type = type;
         for (k = j = 0; j <= Times; j++)
            for (i = 0; i <= Times-j; i++, k++)
            {
               IntegrationPoint &ip = RG->RefPts.IntPoint(k);
               ip.x = cp[i]/(cp[i] + cp[j] + cp[Times-i-j]);
               ip.y = cp[j]/(cp[i] + cp[j] + cp[Times-i-j]);
            }
         Array<int> &G = RG->RefGeoms;
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
         Array<int> &E = RG->RefEdges;
         int lb = 0, li = 2*RG->NumBdrEdges;
         // horizontal edges
         for (k = 0; k < Times; k += Times/ETimes)
         {
            int &lt = (k == 0) ? lb : li;
            j = k*(Times+1)-((k-1)*k)/2;
            for (i = 0; i < Times-k; i++)
            {
               E[lt++] = j; j++;
               E[lt++] = j;
            }
         }
         // diagonal edges
         for (k = Times; k > 0; k -= Times/ETimes)
         {
            int &lt = (k == Times) ? lb : li;
            j = k;
            for (i = 0; i < k; i++)
            {
               E[lt++] = j; j += Times-i;
               E[lt++] = j;
            }
         }
         // vertical edges
         for (k = 0; k < Times; k += Times/ETimes)
         {
            int &lt = (k == 0) ? lb : li;
            j = k;
            for (i = 0; i < Times-k; i++)
            {
               E[lt++] = j; j += Times-i+1;
               E[lt++] = j;
            }
         }

         RGeom[Geometry::TRIANGLE].Append(RG);
         return RG;
      }

      case Geometry::SQUARE:
      {
         RG = new RefinedGeometry((Times+1)*(Times+1), 4*Times*Times,
                                  4*(ETimes+1)*Times, 4*Times);
         RG->Times = Times;
         RG->ETimes = ETimes;
         RG->Type = type;
         for (k = j = 0; j <= Times; j++)
            for (i = 0; i <= Times; i++, k++)
            {
               IntegrationPoint &ip = RG->RefPts.IntPoint(k);
               ip.x = cp[i];
               ip.y = cp[j];
            }
         Array<int> &G = RG->RefGeoms;
         for (l = k = j = 0; j < Times; j++, k++)
            for (i = 0; i < Times; i++, k++)
            {
               G[l++] = k;
               G[l++] = k+1;
               G[l++] = k+Times+2;
               G[l++] = k+Times+1;
            }
         Array<int> &E = RG->RefEdges;
         int lb = 0, li = 2*RG->NumBdrEdges;
         // horizontal edges
         for (k = 0; k <= Times; k += Times/ETimes)
         {
            int &lt = (k == 0 || k == Times) ? lb : li;
            for (i = 0, j = k*(Times+1); i < Times; i++)
            {
               E[lt++] = j; j++;
               E[lt++] = j;
            }
         }
         // vertical edges (in right-to-left order)
         for (k = Times; k >= 0; k -= Times/ETimes)
         {
            int &lt = (k == Times || k == 0) ? lb : li;
            for (i = 0, j = k; i < Times; i++)
            {
               E[lt++] = j; j += Times+1;
               E[lt++] = j;
            }
         }

         RGeom[Geometry::SQUARE].Append(RG);
         return RG;
      }

      case Geometry::CUBE:
      {
         RG = new RefinedGeometry ((Times+1)*(Times+1)*(Times+1),
                                   8*Times*Times*Times, 0);
         RG->Times = Times;
         RG->ETimes = ETimes;
         RG->Type = type;
         for (l = k = 0; k <= Times; k++)
            for (j = 0; j <= Times; j++)
               for (i = 0; i <= Times; i++, l++)
               {
                  IntegrationPoint &ip = RG->RefPts.IntPoint(l);
                  ip.x = cp[i];
                  ip.y = cp[j];
                  ip.z = cp[k];
               }
         Array<int> &G = RG->RefGeoms;
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

         RGeom[Geometry::CUBE].Append(RG);
         return RG;
      }

      case Geometry::TETRAHEDRON:
      {
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
         RG = new RefinedGeometry((n+3)*(n+2)*(n+1)/6, 4*n*n*n, 0);
         RG->Times = Times;
         RG->ETimes = ETimes;
         RG->Type = type;
         // enumerate and define the vertices
         Array<int> vi((n+1)*(n+1)*(n+1));
         vi = -1;
         m = 0;
         for (k = 0; k <= n; k++)
            for (j = 0; j <= k; j++)
               for (i = 0; i <= j; i++)
               {
                  IntegrationPoint &ip = RG->RefPts.IntPoint(m);
                  // map the coordinates to the reference tetrahedron
                  // (0,0,0) -> (0,0,0)
                  // (0,0,1) -> (1,0,0)
                  // (1,1,1) -> (0,1,0)
                  // (0,1,1) -> (0,0,1)
                  double w = cp[k-j] + cp[i] + cp[j-i] + cp[Times-k];
                  ip.x = cp[k-j]/w;
                  ip.y = cp[i]/w;
                  ip.z = cp[j-i]/w;
                  l = i + (j + k * (n+1)) * (n+1);
                  vi[l] = m;
                  m++;
               }
         if (m != (n+3)*(n+2)*(n+1)/6)
         {
            mfem_error("GeometryRefiner::Refine() for TETRAHEDRON #1");
         }
         // elements
         Array<int> &G = RG->RefGeoms;
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

         RGeom[Geometry::TETRAHEDRON].Append(RG);
         return RG;
      }

      case Geometry::PRISM:
      {
         const int n = Times;
         RG = new RefinedGeometry ((n+1)*(n+1)*(n+2)/2, 6*n*n*n, 0);
         RG->Times = Times;
         RG->ETimes = ETimes;
         RG->Type = type;
         // enumerate and define the vertices
         m = 0;
         for (l = k = 0; k <= n; k++)
            for (j = 0; j <= n; j++)
               for (i = 0; i <= n-j; i++, l++)
               {
                  IntegrationPoint &ip = RG->RefPts.IntPoint(l);
                  if (type == 0)
                  {
                     ip.x = double(i) / n;
                     ip.y = double(j) / n;
                     ip.z = double(k) / n;
                  }
                  else
                  {
                     ip.x = cp[i]/(cp[i] + cp[j] + cp[n-i-j]);
                     ip.y = cp[j]/(cp[i] + cp[j] + cp[n-i-j]);
                     ip.z = cp[k];
                  }
                  m++;
               }
         if (m != (n+1)*(n+1)*(n+2)/2)
         {
            mfem_error("GeometryRefiner::Refine() for PRISM #1");
         }
         // elements
         Array<int> &G = RG->RefGeoms;
         m = 0;
         for (m = k = 0; k < n; k++)
            for (l = j = 0; j < n; j++, l++)
               for (i = 0; i < n-j; i++, l++)
               {
                  G[m++] = l + (k+0) * (n+1) * (n+2) / 2;
                  G[m++] = l + 1 + (k+0) * (n+1) * (n+2) / 2;
                  G[m++] = l - j + (2 + (k+0) * (n+2)) * (n+1) / 2;
                  G[m++] = l + (k+1) * (n+1) * (n+2) / 2;
                  G[m++] = l + 1 + (k+1) * (n+1) * (Times+2) / 2;
                  G[m++] = l - j + (2 + (k+1) * (n+2)) * (n+1) / 2;
                  if (i+j+1 < n)
                  {
                     G[m++] = l + 1 + (k+0) * (n+1) * (n+2)/2;
                     G[m++] = l - j + (2 + (k+0) * (n+1)) * (n+2) / 2;
                     G[m++] = l - j + (2 + (k+0) * (n+2)) * (n+1) / 2;
                     G[m++] = l + 1 + (k+1) * (n+1) * (n+2) / 2;
                     G[m++] = l - j + (2 + (k+1) * (n+1)) * (n+2) / 2;
                     G[m++] = l - j + (2 + (k+1) * (n+2)) * (n+1) / 2;
                  }
               }
         if (m != 6*n*n*n)
         {
            mfem_error("GeometryRefiner::Refine() for PRISM #2");
         }
         for (i = 0; i < m; i++)
            if (G[i] < 0)
            {
               mfem_error("GeometryRefiner::Refine() for PRISM #3");
            }

         RGeom[Geometry::PRISM].Append(RG);
         return RG;
      }

      default:

         return NULL;
   }
}

const IntegrationRule *GeometryRefiner::RefineInterior(Geometry::Type Geom,
                                                       int Times)
{
   IntegrationRule *ir = NULL;

   switch (Geom)
   {
      case Geometry::SEGMENT:
      {
         if (Times < 2)
         {
            return NULL;
         }
         ir = FindInIntPts(Geom, Times-1);
         if (ir) { return ir; }

         ir = new IntegrationRule(Times-1);
         for (int i = 1; i < Times; i++)
         {
            IntegrationPoint &ip = ir->IntPoint(i-1);
            ip.x = double(i) / Times;
            ip.y = ip.z = 0.0;
         }
      }
      break;

      case Geometry::TRIANGLE:
      {
         if (Times < 3)
         {
            return NULL;
         }
         ir = FindInIntPts(Geom, ((Times-1)*(Times-2))/2);
         if (ir) { return ir; }

         ir = new IntegrationRule(((Times-1)*(Times-2))/2);
         for (int k = 0, j = 1; j < Times-1; j++)
            for (int i = 1; i < Times-j; i++, k++)
            {
               IntegrationPoint &ip = ir->IntPoint(k);
               ip.x = double(i) / Times;
               ip.y = double(j) / Times;
               ip.z = 0.0;
            }
      }
      break;

      case Geometry::SQUARE:
      {
         if (Times < 2)
         {
            return NULL;
         }
         ir = FindInIntPts(Geom, (Times-1)*(Times-1));
         if (ir) { return ir; }

         ir = new IntegrationRule((Times-1)*(Times-1));
         for (int k = 0, j = 1; j < Times; j++)
            for (int i = 1; i < Times; i++, k++)
            {
               IntegrationPoint &ip = ir->IntPoint(k);
               ip.x = double(i) / Times;
               ip.y = double(j) / Times;
               ip.z = 0.0;
            }
      }
      break;

      default:
         mfem_error("GeometryRefiner::RefineInterior(...)");
   }

   MFEM_ASSERT(ir != NULL, "Failed to construct the refined IntegrationRule.");
   IntPts[Geom].Append(ir);

   return ir;
}


int GeometryRefiner::GetRefinementLevelFromPoints(Geometry::Type geom, int Npts)
{
   switch (geom)
   {
      case Geometry::POINT:
      {
         return -1;
      }
      case Geometry::SEGMENT:
      {
         return Npts -1;
      }
      case Geometry::TRIANGLE:
      {
         for (int n = 0, np = 0; (n < 15) && (np < Npts) ; n++)
         {
            np = (n+1)*(n+2)/2;
            if (np == Npts) { return n; }
         }
         return -1;
      }
      case Geometry::SQUARE:
      {
         for (int n = 0, np = 0; (n < 15) && (np < Npts) ; n++)
         {
            np = (n+1)*(n+1);
            if (np == Npts) { return n; }
         }
         return -1;
      }
      case Geometry::CUBE:
      {
         for (int n = 0, np = 0; (n < 15) && (np < Npts) ; n++)
         {
            np = (n+1)*(n+1)*(n+1);
            if (np == Npts) { return n; }
         }
         return -1;
      }
      case Geometry::TETRAHEDRON:
      {
         for (int n = 0, np = 0; (n < 15) && (np < Npts) ; n++)
         {
            np = (n+3)*(n+2)*(n+1)/6;
            if (np == Npts) { return n; }
         }
         return -1;
      }
      case Geometry::PRISM:
      {
         for (int n = 0, np = 0; (n < 15) && (np < Npts) ; n++)
         {
            np = (n+1)*(n+1)*(n+2)/2;
            if (np == Npts) { return n; }
         }
         return -1;
      }
      default:
      {
         mfem_error("Non existing Geometry.");
      }
   }

   return -1;
}


int GeometryRefiner::GetRefinementLevelFromElems(Geometry::Type geom, int Nels)
{
   switch (geom)
   {
      case Geometry::POINT:
      {
         return -1;
      }
      case Geometry::SEGMENT:
      {
         return Nels;
      }
      case Geometry::TRIANGLE:
      case Geometry::SQUARE:
      {
         for (int n = 0; (n < 15) && (n*n < Nels+1) ; n++)
         {
            if (n*n == Nels) { return n-1; }
         }
         return -1;
      }
      case Geometry::CUBE:
      case Geometry::TETRAHEDRON:
      case Geometry::PRISM:
      {
         for (int n = 0; (n < 15) && (n*n*n < Nels+1) ; n++)
         {
            if (n*n*n == Nels) { return n-1; }
         }
         return -1;
      }
      default:
      {
         mfem_error("Non existing Geometry.");
      }
   }

   return -1;
}


GeometryRefiner GlobGeometryRefiner;

}
