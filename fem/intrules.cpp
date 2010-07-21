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

// Implementation of IntegrationRule(s) classes

// Acknowledgment: Some of the high-precision triangular and tetrahedral
// quadrature rules below were obtained from the Encyclopaedia of Cubature
// Formulas at http://nines.cs.kuleuven.be/research/ecf/ecf.html

#include <math.h>
#include "fem.hpp"

IntegrationRule::IntegrationRule (int NP)
{
   NPoints   = NP;
   IntPoints = new IntegrationPoint[NP];
}

IntegrationRule::IntegrationRule (IntegrationRule &irx, IntegrationRule &iry)
{
   int i, j, nx, ny;

   nx = irx.GetNPoints();
   ny = iry.GetNPoints();
   NPoints = nx * ny;
   IntPoints = new IntegrationPoint[NPoints];

   for (j = 0; j < ny; j++)
   {
      IntegrationPoint &ipy = iry.IntPoint(j);
      for (i = 0; i < nx; i++)
      {
         IntegrationPoint &ipx = irx.IntPoint(i);
         IntegrationPoint &ip  = IntPoints[j*nx+i];

         ip.x = ipx.x;
         ip.y = ipy.x;
         ip.weight = ipx.weight * ipy.weight;
      }
   }
}

void IntegrationRule::GaussianRule()
{
   int n = NPoints;
   int m = (n+1)/2;
   int i, j;
   double p1, p2, p3;
   double pp, z, z1;
   for (i = 1; i <= m; i++)
   {
      z = cos ( M_PI * (i - 0.25) / (n + 0.5));

      while(1)
      {
         p1 = 1;
         p2 = 0;
         for (j = 1; j <= n; j++)
         {
            p3 = p2;
            p2 = p1;
            p1 = ((2 * j - 1) * z * p2 - (j - 1) * p3) / j;
         }
         // p1 is Legendre polynomial

         pp = n * (z*p1-p2) / (z*z - 1);
         z1 = z;
         z = z1-p1/pp;

         if (fabs (z - z1) < 1e-14) break;
      }

      IntPoints[i-1].x  = 0.5 * (1 - z);
      IntPoints[n-i].x  = 0.5 * (1 + z);
      IntPoints[i-1].weight = IntPoints[n-i].weight =
         1.0 / ( (1  - z * z) * pp * pp);
   }
}

void IntegrationRule::UniformRule()
{
   int i;
   double h;

   h = 1.0 / (NPoints - 1);
   for (i = 0; i < NPoints; i++)
   {
      IntPoints[i].x = double(i) / (NPoints - 1);
      IntPoints[i].weight = h;
   }
   IntPoints[0].weight = 0.5 * h;
   IntPoints[NPoints-1].weight = 0.5 * h;
}

IntegrationRule::~IntegrationRule ()
{
   delete [] IntPoints;
}



IntegrationRules IntRules(0);

IntegrationRules RefinedIntRules(1);

IntegrationRules::IntegrationRules (int refined)
{
   if (refined < 0) { own_rules = 0; return; }

   own_rules = 1;
   PointIntegrationRules();
   SegmentIntegrationRules(refined);
   TriangleIntegrationRules(0);
   SquareIntegrationRules();
   TetrahedronIntegrationRules(0);
   CubeIntegrationRules();
}

const IntegrationRule & IntegrationRules::Get (int GeomType, int Order)
{
   switch (GeomType)
   {
   case Geometry::POINT:    return *PointIntRules[Order];
   case Geometry::SEGMENT:  return *SegmentIntRules[Order];
   case Geometry::TRIANGLE: return *TriangleIntRules[Order];
   case Geometry::SQUARE:   return *SquareIntRules[Order];
   case Geometry::TETRAHEDRON: return *TetrahedronIntRules[Order];
   case Geometry::CUBE:        return *CubeIntRules[Order];
   default:
#ifdef MFEM_DEBUG
      mfem_error ("IntegrationRules::Get (...)");
#endif
      ;
   }
   return *TriangleIntRules[Order];
}

void IntegrationRules::Set (int GeomType, int Order, IntegrationRule &IntRule)
{
   Array<IntegrationRule *> *ir_array;

   switch (GeomType)
   {
   case Geometry::POINT:       ir_array = &PointIntRules; break;
   case Geometry::SEGMENT:     ir_array = &SegmentIntRules; break;
   case Geometry::TRIANGLE:    ir_array = &TriangleIntRules; break;
   case Geometry::SQUARE:      ir_array = &SquareIntRules; break;
   case Geometry::TETRAHEDRON: ir_array = &TetrahedronIntRules; break;
   case Geometry::CUBE:        ir_array = &CubeIntRules; break;
   default:
#ifdef MFEM_DEBUG
      mfem_error ("IntegrationRules::Set (...)");
#endif
      ;
   }

   if (ir_array -> Size() <= Order)
   {
      int i = ir_array -> Size();

      ir_array -> SetSize (Order + 1);

      for ( ; i < Order; i++)
         (*ir_array)[i] = NULL;
   }

   (*ir_array)[Order] = &IntRule;
}

IntegrationRules::~IntegrationRules ()
{
   int i;

   if (!own_rules) return;

   for (i = 0; i < PointIntRules.Size(); i++)
      if (PointIntRules[i] != NULL)
         delete PointIntRules[i];

   for (i = 0; i < SegmentIntRules.Size(); i++)
      if (SegmentIntRules[i] != NULL)
         delete SegmentIntRules[i];

   for (i = 0; i < TriangleIntRules.Size(); i++)
      if (TriangleIntRules[i] != NULL)
         delete TriangleIntRules[i];

   for (i = 0; i < SquareIntRules.Size(); i++)
      if (SquareIntRules[i] != NULL)
         delete SquareIntRules[i];

   for (i = 0; i < TetrahedronIntRules.Size(); i++)
      if (TetrahedronIntRules[i] != NULL)
         delete TetrahedronIntRules[i];

   for (i = 0; i < CubeIntRules.Size(); i++)
      if (CubeIntRules[i] != NULL)
         delete CubeIntRules[i];
}

// Integration rules for a point
void IntegrationRules::PointIntegrationRules()
{
   PointIntRules.SetSize(2);

   PointIntRules[0] = new IntegrationRule (1);
   PointIntRules[0] -> IntPoint(0).x = .0;
   PointIntRules[0] -> IntPoint(0).weight = 1.;

   PointIntRules[1] = new IntegrationRule (1);
   PointIntRules[1] -> IntPoint(0).x = .0;
   PointIntRules[1] -> IntPoint(0).weight = 1.;
}

// Integration rules for line segment [0,1]
void IntegrationRules::SegmentIntegrationRules(int refined)
{
   int i, j;

   SegmentIntRules.SetSize(32);

   if (refined) {
      int n;
      IntegrationRule * tmp;
      for (i = 0; i < SegmentIntRules.Size(); i++) {
         n = i/2+1; tmp = new IntegrationRule(n); tmp -> GaussianRule();
         SegmentIntRules[i] = new IntegrationRule (2*n);
         for (j = 0; j < n; j++) {
            SegmentIntRules[i]->IntPoint(j).x = tmp->IntPoint(j).x/2.0;
            SegmentIntRules[i]->IntPoint(j).weight = tmp->IntPoint(j).weight/2.0;
            SegmentIntRules[i]->IntPoint(j+n).x = 0.5 + tmp->IntPoint(j).x/2.0;
            SegmentIntRules[i]->IntPoint(j+n).weight = tmp->IntPoint(j).weight/2.0;
         }
         delete tmp;
      }
      return;
   }

   for (i = 0; i < SegmentIntRules.Size(); i++)
      SegmentIntRules[i] = NULL;

   // 1 point - 1 degree
   SegmentIntRules[0] = new IntegrationRule (1);

   SegmentIntRules[0] -> IntPoint(0).x = .5;
   SegmentIntRules[0] -> IntPoint(0).weight = 1.;

   // 1 point - 1 degree
   SegmentIntRules[1] = new IntegrationRule (1);

   SegmentIntRules[1] -> IntPoint(0).x = .5;
   SegmentIntRules[1] -> IntPoint(0).weight = 1.;

   // 2 point - 3 degree
   SegmentIntRules[2] = new IntegrationRule (2);

   SegmentIntRules[2] -> IntPoint(0).x = 0.211324865405187;
   SegmentIntRules[2] -> IntPoint(0).weight = .5;
   SegmentIntRules[2] -> IntPoint(1).x = 0.788675134594812;
   SegmentIntRules[2] -> IntPoint(1).weight = .5;

   // 2 point - 3 degree
   SegmentIntRules[3] = new IntegrationRule (2);

   SegmentIntRules[3] -> IntPoint(0).x = 0.211324865405187;
   SegmentIntRules[3] -> IntPoint(0).weight = .5;
   SegmentIntRules[3] -> IntPoint(1).x = 0.788675134594812;
   SegmentIntRules[3] -> IntPoint(1).weight = .5;

   // 3 point - 5 degree
   SegmentIntRules[4] = new IntegrationRule (3);
   SegmentIntRules[4] -> GaussianRule();

   SegmentIntRules[4] -> IntPoint(0).x = 0.11270166537925831148;
   SegmentIntRules[4] -> IntPoint(0).weight = 0.2777777777777777777777778;
   SegmentIntRules[4] -> IntPoint(1).x = 0.5;
   SegmentIntRules[4] -> IntPoint(1).weight = 0.4444444444444444444444444;
   SegmentIntRules[4] -> IntPoint(2).x = 0.88729833462074168852;
   SegmentIntRules[4] -> IntPoint(2).weight = 0.2777777777777777777777778;

   // 3 point - 5 degree
   SegmentIntRules[5] = new IntegrationRule (3);

   SegmentIntRules[5] -> IntPoint(0).x = 0.11270166537925831148;
   SegmentIntRules[5] -> IntPoint(0).weight = 0.2777777777777777777777778;
   SegmentIntRules[5] -> IntPoint(1).x = 0.5;
   SegmentIntRules[5] -> IntPoint(1).weight = 0.4444444444444444444444444;
   SegmentIntRules[5] -> IntPoint(2).x = 0.88729833462074168852;
   SegmentIntRules[5] -> IntPoint(2).weight = 0.2777777777777777777777778;

   /*
   // 4 point - 7 degree
   SegmentIntRules[6] = new IntegrationRule (4);

   SegmentIntRules[6] -> IntPoint(0).x = 0.069431844202973;
   SegmentIntRules[6] -> IntPoint(0).weight = .1739274226587269286;
   SegmentIntRules[6] -> IntPoint(1).x = 0.330009478207572;
   SegmentIntRules[6] -> IntPoint(1).weight = .3260725774312730713;
   SegmentIntRules[6] -> IntPoint(2).x = 0.669990521792428;
   SegmentIntRules[6] -> IntPoint(2).weight = .3260725774312730713;
   SegmentIntRules[6] -> IntPoint(3).x = 0.930568155797026;
   SegmentIntRules[6] -> IntPoint(3).weight = .1739274226587269286;

   // 4 point - 7 degree
   SegmentIntRules[7] = new IntegrationRule (4);

   SegmentIntRules[7] -> IntPoint(0).x = 0.069431844202973;
   SegmentIntRules[7] -> IntPoint(0).weight = .1739274226587269286;
   SegmentIntRules[7] -> IntPoint(1).x = 0.330009478207572;
   SegmentIntRules[7] -> IntPoint(1).weight = .3260725774312730713;
   SegmentIntRules[7] -> IntPoint(2).x = 0.669990521792428;
   SegmentIntRules[7] -> IntPoint(2).weight = .3260725774312730713;
   SegmentIntRules[7] -> IntPoint(3).x = 0.930568155797026;
   SegmentIntRules[7] -> IntPoint(3).weight = .1739274226587269286;

   // 5 point - 9 degree
   SegmentIntRules[8] = new IntegrationRule (5);

   SegmentIntRules[8] -> IntPoint(0).x = 0.046910077030668;
   SegmentIntRules[8] -> IntPoint(0).weight = .1655506920379504390;
   SegmentIntRules[8] -> IntPoint(1).x = 0.230765344947158;
   SegmentIntRules[8] -> IntPoint(1).weight = .3344378060412287839;
   SegmentIntRules[8] -> IntPoint(2).x = 0.5;
   SegmentIntRules[8] -> IntPoint(2).weight = .0000230038416415541;
   SegmentIntRules[8] -> IntPoint(3).x = 0.769234655052842;
   SegmentIntRules[8] -> IntPoint(3).weight = .3344378060412287839;
   SegmentIntRules[8] -> IntPoint(4).x = 0.953089922969332;
   SegmentIntRules[8] -> IntPoint(4).weight = .1655506920379504390;

   // 5 point - 9 degree
   SegmentIntRules[9] = new IntegrationRule (5);

   SegmentIntRules[9] -> IntPoint(0).x = 0.046910077030668;
   SegmentIntRules[9] -> IntPoint(0).weight = .1655506920379504390;
   SegmentIntRules[9] -> IntPoint(1).x = 0.230765344947158;
   SegmentIntRules[9] -> IntPoint(1).weight = .3344378060412287839;
   SegmentIntRules[9] -> IntPoint(2).x = 0.5;
   SegmentIntRules[9] -> IntPoint(2).weight = .0000230038416415541;
   SegmentIntRules[9] -> IntPoint(3).x = 0.769234655052842;
   SegmentIntRules[9] -> IntPoint(3).weight = .3344378060412287839;
   SegmentIntRules[9] -> IntPoint(4).x = 0.953089922969332;
   SegmentIntRules[9] -> IntPoint(4).weight = .1655506920379504390;

   // 6 point - 11 degree
   SegmentIntRules[10] = new IntegrationRule (6);

   SegmentIntRules[10] -> IntPoint(0).x = 0.033765242898424;
   SegmentIntRules[10] -> IntPoint(0).weight = .0856622461895851724;
   SegmentIntRules[10] -> IntPoint(1).x = 0.169395306766868;
   SegmentIntRules[10] -> IntPoint(1).weight = .1803807865240693038;
   SegmentIntRules[10] -> IntPoint(2).x = 0.380690406958402;
   SegmentIntRules[10] -> IntPoint(2).weight = .2339569672863455237;
   SegmentIntRules[10] -> IntPoint(3).x = 0.619309593041598;
   SegmentIntRules[10] -> IntPoint(3).weight = .2339569672863455237;
   SegmentIntRules[10] -> IntPoint(4).x = 0.830604693233132;
   SegmentIntRules[10] -> IntPoint(4).weight = .1803807865240693038;
   SegmentIntRules[10] -> IntPoint(5).x = 0.966234757101576;
   SegmentIntRules[10] -> IntPoint(5).weight = .0856622461895851724;

   // 6 point - 11 degree
   SegmentIntRules[11] = new IntegrationRule (6);

   SegmentIntRules[11] -> IntPoint(0).x = 0.033765242898424;
   SegmentIntRules[11] -> IntPoint(0).weight = .0856622461895851724;
   SegmentIntRules[11] -> IntPoint(1).x = 0.169395306766868;
   SegmentIntRules[11] -> IntPoint(1).weight = .1803807865240693038;
   SegmentIntRules[11] -> IntPoint(2).x = 0.380690406958402;
   SegmentIntRules[11] -> IntPoint(2).weight = .2339569672863455237;
   SegmentIntRules[11] -> IntPoint(3).x = 0.619309593041598;
   SegmentIntRules[11] -> IntPoint(3).weight = .2339569672863455237;
   SegmentIntRules[11] -> IntPoint(4).x = 0.830604693233132;
   SegmentIntRules[11] -> IntPoint(4).weight = .1803807865240693038;
   SegmentIntRules[11] -> IntPoint(5).x = 0.966234757101576;
   SegmentIntRules[11] -> IntPoint(5).weight = .0856622461895851724;

   // 7 point - 13 degree
   SegmentIntRules[12] = new IntegrationRule (7);

   SegmentIntRules[12] -> IntPoint(0).x = 0.025446043828621;
   SegmentIntRules[12] -> IntPoint(0).weight = .0818467915961606551;
   SegmentIntRules[12] -> IntPoint(1).x = 0.129234407200303;
   SegmentIntRules[12] -> IntPoint(1).weight = .1768003619484993849;
   SegmentIntRules[12] -> IntPoint(2).x = 0.297077424311301;
   SegmentIntRules[12] -> IntPoint(2).weight = .2413528419051118830;
   SegmentIntRules[12] -> IntPoint(3).x = 0.5;
   SegmentIntRules[12] -> IntPoint(3).weight = 9.100456214e-9;
   SegmentIntRules[12] -> IntPoint(4).x = 0.702922575688699;
   SegmentIntRules[12] -> IntPoint(4).weight = .2413528419051118830;
   SegmentIntRules[12] -> IntPoint(5).x = 0.870765592799697;
   SegmentIntRules[12] -> IntPoint(5).weight = .1768003619484993849;
   SegmentIntRules[12] -> IntPoint(6).x = 0.974553956171379;
   SegmentIntRules[12] -> IntPoint(6).weight = .0818467915961606551;

   // 7 point - 13 degree
   SegmentIntRules[13] = new IntegrationRule (7);

   SegmentIntRules[13] -> IntPoint(0).x = 0.025446043828621;
   SegmentIntRules[13] -> IntPoint(0).weight = .0818467915961606551;
   SegmentIntRules[13] -> IntPoint(1).x = 0.129234407200303;
   SegmentIntRules[13] -> IntPoint(1).weight = .1768003619484993849;
   SegmentIntRules[13] -> IntPoint(2).x = 0.297077424311301;
   SegmentIntRules[13] -> IntPoint(2).weight = .2413528419051118830;
   SegmentIntRules[13] -> IntPoint(3).x = 0.5;
   SegmentIntRules[13] -> IntPoint(3).weight = 9.100456214e-9;
   SegmentIntRules[13] -> IntPoint(4).x = 0.702922575688699;
   SegmentIntRules[13] -> IntPoint(4).weight = .2413528419051118830;
   SegmentIntRules[13] -> IntPoint(5).x = 0.870765592799697;
   SegmentIntRules[13] -> IntPoint(5).weight = .1768003619484993849;
   SegmentIntRules[13] -> IntPoint(6).x = 0.974553956171379;
   SegmentIntRules[13] -> IntPoint(6).weight = .0818467915961606551;
   */

   for (i = 6; i < SegmentIntRules.Size(); i++)
   {
      SegmentIntRules[i] = new IntegrationRule (i/2+1);
      SegmentIntRules[i] -> GaussianRule();
   }
}

// Integration rules for reference triangle {[0,0],[1,0],[0,1]}
void IntegrationRules::TriangleIntegrationRules(int refined)
{
   TriangleIntRules.SetSize(15);

   if (refined)
      mfem_error ("Refined TriangleIntegrationRules are not implemented!");

   TriangleIntRules = NULL;

   // 1 point - 0 degree
   TriangleIntRules[0] = new IntegrationRule (1);

   TriangleIntRules[0] -> IntPoint(0).x = 1./3.;
   TriangleIntRules[0] -> IntPoint(0).y = 1./3.;
   TriangleIntRules[0] -> IntPoint(0).weight = 0.5;

   /*
   // 3 point - 1 degree (vertices)
   TriangleIntRules[1] = new IntegrationRule (3);

   TriangleIntRules[1] -> IntPoint(0).x      = 0.;
   TriangleIntRules[1] -> IntPoint(0).y      = 0.;
   TriangleIntRules[1] -> IntPoint(0).weight = 0.16666666666667;

   TriangleIntRules[1] -> IntPoint(1).x      = 1.;
   TriangleIntRules[1] -> IntPoint(1).y      = 0.;
   TriangleIntRules[1] -> IntPoint(1).weight = 0.16666666666667;

   TriangleIntRules[1] -> IntPoint(2).x      = 0.;
   TriangleIntRules[1] -> IntPoint(2).y      = 1.;
   TriangleIntRules[1] -> IntPoint(2).weight = 0.16666666666667;
   */

   // 1 point - 1 degree
   TriangleIntRules[1] = new IntegrationRule (1);

   TriangleIntRules[1] -> IntPoint(0).x = 1./3.;
   TriangleIntRules[1] -> IntPoint(0).y = 1./3.;
   TriangleIntRules[1] -> IntPoint(0).weight = 0.5;

   // 3 point - 2 degree
   TriangleIntRules[2] = new IntegrationRule (3);

   /*
   //   midpoints of the edges
   TriangleIntRules[2] -> IntPoint(0).x      = 0.5;
   TriangleIntRules[2] -> IntPoint(0).y      = 0.0;
   TriangleIntRules[2] -> IntPoint(0).weight = 1./6.;

   TriangleIntRules[2] -> IntPoint(1).x      = 0.5;
   TriangleIntRules[2] -> IntPoint(1).y      = 0.5;
   TriangleIntRules[2] -> IntPoint(1).weight = 1./6.;

   TriangleIntRules[2] -> IntPoint(2).x      = 0.0;
   TriangleIntRules[2] -> IntPoint(2).y      = 0.5;
   TriangleIntRules[2] -> IntPoint(2).weight = 1./6.;
   */

   //   interior points
   TriangleIntRules[2] -> IntPoint(0).x      = 1./6.;
   TriangleIntRules[2] -> IntPoint(0).y      = 1./6.;
   TriangleIntRules[2] -> IntPoint(0).weight = 1./6.;

   TriangleIntRules[2] -> IntPoint(1).x      = 2./3.;
   TriangleIntRules[2] -> IntPoint(1).y      = 1./6.;
   TriangleIntRules[2] -> IntPoint(1).weight = 1./6.;

   TriangleIntRules[2] -> IntPoint(2).x      = 1./6.;
   TriangleIntRules[2] -> IntPoint(2).y      = 2./3.;
   TriangleIntRules[2] -> IntPoint(2).weight = 1./6.;

   // 4 point - 3 degree (has one negative weight)
   TriangleIntRules[3] = new IntegrationRule (4);

   TriangleIntRules[3] -> IntPoint(0).x      = 1./3.;
   TriangleIntRules[3] -> IntPoint(0).y      = 1./3.;
   TriangleIntRules[3] -> IntPoint(0).weight = -0.28125; // -9./32.

   TriangleIntRules[3] -> IntPoint(1).x      = 0.2;
   TriangleIntRules[3] -> IntPoint(1).y      = 0.2;
   TriangleIntRules[3] -> IntPoint(1).weight = 25./96.;

   TriangleIntRules[3] -> IntPoint(2).x      = 0.6;
   TriangleIntRules[3] -> IntPoint(2).y      = 0.2;
   TriangleIntRules[3] -> IntPoint(2).weight = 25./96.;

   TriangleIntRules[3] -> IntPoint(3).x      = 0.2;
   TriangleIntRules[3] -> IntPoint(3).y      = 0.6;
   TriangleIntRules[3] -> IntPoint(3).weight = 25./96.;

   // 6 point - 4 degree
   TriangleIntRules[4] = new IntegrationRule (6);
   const double rule_6[] =
      {
         0.0915762135097707434595714634022015,
         0.0915762135097707434595714634022015,
         0.0549758718276609338191631624501052,
         0.0915762135097707434595714634022015,
         0.8168475729804585130808570731955970,
         0.0549758718276609338191631624501052,
         0.8168475729804585130808570731955970,
         0.0915762135097707434595714634022015,
         0.0549758718276609338191631624501052,
         0.445948490915964886318329253883051,
         0.445948490915964886318329253883051,
         0.111690794839005732847503504216561,
         0.445948490915964886318329253883051,
         0.108103018168070227363341492233898,
         0.111690794839005732847503504216561,
         0.108103018168070227363341492233898,
         0.445948490915964886318329253883051,
         0.111690794839005732847503504216561
      };
   for (int i = 0; i < 6; i++)
   {
      TriangleIntRules[4]->IntPoint(i).x      = rule_6[3*i+0];
      TriangleIntRules[4]->IntPoint(i).y      = rule_6[3*i+1];
      TriangleIntRules[4]->IntPoint(i).weight = rule_6[3*i+2];
   }

   // 7 point - 5 degree
   TriangleIntRules[5] = new IntegrationRule (7);
   const double rule_7[] =
      {
         0.3333333333333333333333333333333,
         0.3333333333333333333333333333333,
         0.1125,
         0.1012865073234563388009873619151,
         0.1012865073234563388009873619151,
         0.06296959027241357629784197275009,
         0.1012865073234563388009873619151,
         0.7974269853530873223980252761698,
         0.06296959027241357629784197275009,
         0.7974269853530873223980252761698,
         0.1012865073234563388009873619151,
         0.06296959027241357629784197275009,
         0.4701420641051150897704412095134,
         0.4701420641051150897704412095134,
         0.06619707639425309036882469391658,
         0.4701420641051150897704412095134,
         0.0597158717897698204591175809731,
         0.06619707639425309036882469391658,
         0.0597158717897698204591175809731,
         0.4701420641051150897704412095134,
         0.06619707639425309036882469391658
      };
   for (int i = 0; i < 7; i++)
   {
      TriangleIntRules[5]->IntPoint(i).x      = rule_7[3*i+0];
      TriangleIntRules[5]->IntPoint(i).y      = rule_7[3*i+1];
      TriangleIntRules[5]->IntPoint(i).weight = rule_7[3*i+2];
   }

   // 12 point - 6 degree
   TriangleIntRules[6] = new IntegrationRule (12);
   const double rule_12_6[] =
      {
         0.0630890144915022283403316028708191,
         0.0630890144915022283403316028708191,
         0.0254224531851034084604684045534344,
         0.0630890144915022283403316028708191,
         0.8738219710169955433193367942583618,
         0.0254224531851034084604684045534344,
         0.8738219710169955433193367942583618,
         0.0630890144915022283403316028708191,
         0.0254224531851034084604684045534344,
         0.249286745170910421291638553107019,
         0.249286745170910421291638553107019,
         0.0583931378631896830126448056927897,
         0.249286745170910421291638553107019,
         0.501426509658179157416722893785962,
         0.0583931378631896830126448056927897,
         0.501426509658179157416722893785962,
         0.249286745170910421291638553107019,
         0.0583931378631896830126448056927897,
         0.0531450498448169473532496716313981,
         0.310352451033784405416607733956552,
         0.0414255378091867875967767282102212,
         0.310352451033784405416607733956552,
         0.0531450498448169473532496716313981,
         0.0414255378091867875967767282102212,
         0.0531450498448169473532496716313981,
         0.6365024991213986472301425944120499,
         0.0414255378091867875967767282102212,
         0.6365024991213986472301425944120499,
         0.0531450498448169473532496716313981,
         0.0414255378091867875967767282102212,
         0.310352451033784405416607733956552,
         0.6365024991213986472301425944120499,
         0.0414255378091867875967767282102212,
         0.6365024991213986472301425944120499,
         0.310352451033784405416607733956552,
         0.0414255378091867875967767282102212
      };
   for (int i = 0; i < 12; i++)
   {
      TriangleIntRules[6]->IntPoint(i).x      = rule_12_6[3*i+0];
      TriangleIntRules[6]->IntPoint(i).y      = rule_12_6[3*i+1];
      TriangleIntRules[6]->IntPoint(i).weight = rule_12_6[3*i+2];
   }

   // 13 point - 7 degree (has 1 negative weight)
   // TriangleIntRules[7] = new IntegrationRule (13);
#if 0
   const double rule_13[] =
      {
         3.33333333333333E-01,  3.33333333333333E-01, -7.47850222338410E-02,
         4.79308067841920E-01,  2.60345966079040E-01,  8.78076287166040E-02,
         2.60345966079040E-01,  4.79308067841920E-01,  8.78076287166040E-02,
         2.60345966079040E-01,  2.60345966079040E-01,  8.78076287166040E-02,
         8.69739794195568E-01,  6.51301029022160E-02,  2.66736178044190E-02,
         6.51301029022160E-02,  8.69739794195568E-01,  2.66736178044190E-02,
         6.51301029022160E-02,  6.51301029022160E-02,  2.66736178044190E-02,
         4.86903154253160E-02,  3.12865496004874E-01,  3.85568804451285E-02,
         6.38444188569810E-01,  4.86903154253160E-02,  3.85568804451285E-02,
         3.12865496004874E-01,  6.38444188569810E-01,  3.85568804451285E-02,
         4.86903154253160E-02,  6.38444188569810E-01,  3.85568804451285E-02,
         6.38444188569810E-01,  3.12865496004874E-01,  3.85568804451285E-02,
         3.12865496004874E-01,  4.86903154253160E-02,  3.85568804451285E-02
      };
   for (int i = 0; i < 13; i++)
   {
      TriangleIntRules[7]->IntPoint(i).x      = rule_13[3*i+0];
      TriangleIntRules[7]->IntPoint(i).y      = rule_13[3*i+1];
      TriangleIntRules[7]->IntPoint(i).weight = rule_13[3*i+2];
   }
#endif

   // 12 point - degree 7
   TriangleIntRules[7] = new IntegrationRule(12);
   const double rule_12_7[] =
      {
         0.0623822650944021181736830009963499,
         0.0675178670739160854425571310508685,
         0.0265170281574362514287541804607391,
         0.8700998678316817963837598679527816,
         0.0623822650944021181736830009963499,
         0.0265170281574362514287541804607391,
         0.0675178670739160854425571310508685,
         0.8700998678316817963837598679527816,
         0.0265170281574362514287541804607391,
         0.0552254566569266117374791902756449,
         0.321502493851981822666307849199202,
         0.0438814087144460550367699031392875,
         0.6232720494910915655962129605251531,
         0.0552254566569266117374791902756449,
         0.0438814087144460550367699031392875,
         0.321502493851981822666307849199202,
         0.6232720494910915655962129605251531,
         0.0438814087144460550367699031392875,
         0.0343243029450971464696306424839376,
         0.660949196186735657611980310197799,
         0.0287750427849815857384454969002185,
         0.3047265008681671959183890473182634,
         0.0343243029450971464696306424839376,
         0.0287750427849815857384454969002185,
         0.660949196186735657611980310197799,
         0.3047265008681671959183890473182634,
         0.0287750427849815857384454969002185,
         0.515842334353591779257463386826430,
         0.277716166976391782569581871393723,
         0.0674931870098027744626970861664214,
         0.206441498670016438172954741779847,
         0.515842334353591779257463386826430,
         0.0674931870098027744626970861664214,
         0.277716166976391782569581871393723,
         0.206441498670016438172954741779847,
         0.0674931870098027744626970861664214
      };
   for (int i = 0; i < 12; i++)
   {
      TriangleIntRules[7]->IntPoint(i).x      = rule_12_7[3*i+0];
      TriangleIntRules[7]->IntPoint(i).y      = rule_12_7[3*i+1];
      TriangleIntRules[7]->IntPoint(i).weight = rule_12_7[3*i+2];
   }

   // 16 point - 8 degree
   TriangleIntRules[8] = new IntegrationRule(16);
   const double rule_16[] =
      {
         3.33333333333333E-01,  3.33333333333333E-01,  7.21578038388935E-02,
         8.14148234145540E-02,  4.59292588292723E-01,  4.75458171336425E-02,
         4.59292588292723E-01,  8.14148234145540E-02,  4.75458171336425E-02,
         4.59292588292723E-01,  4.59292588292723E-01,  4.75458171336425E-02,
         6.58861384496480E-01,  1.70569307751760E-01,  5.16086852673590E-02,
         1.70569307751760E-01,  6.58861384496480E-01,  5.16086852673590E-02,
         1.70569307751760E-01,  1.70569307751760E-01,  5.16086852673590E-02,
         8.98905543365938E-01,  5.05472283170310E-02,  1.62292488115990E-02,
         5.05472283170310E-02,  8.98905543365938E-01,  1.62292488115990E-02,
         5.05472283170310E-02,  5.05472283170310E-02,  1.62292488115990E-02,
         8.39477740995800E-03,  2.63112829634638E-01,  1.36151570872175E-02,
         7.28492392955404E-01,  8.39477740995800E-03,  1.36151570872175E-02,
         2.63112829634638E-01,  7.28492392955404E-01,  1.36151570872175E-02,
         8.39477740995800E-03,  7.28492392955404E-01,  1.36151570872175E-02,
         7.28492392955404E-01,  2.63112829634638E-01,  1.36151570872175E-02,
         2.63112829634638E-01,  8.39477740995800E-03,  1.36151570872175E-02
      };
   for (int i = 0; i < 16; i++)
   {
      TriangleIntRules[8]->IntPoint(i).x      = rule_16[3*i+0];
      TriangleIntRules[8]->IntPoint(i).y      = rule_16[3*i+1];
      TriangleIntRules[8]->IntPoint(i).weight = rule_16[3*i+2];
   }

   // 19 point - 9 degree
   TriangleIntRules[9] = new IntegrationRule(19);
   const double rule_19[] =
      {
         3.33333333333333E-01,  3.33333333333333E-01,  4.85678981413995E-02,
         2.06349616025250E-02,  4.89682519198738E-01,  1.56673501135695E-02,
         4.89682519198738E-01,  2.06349616025250E-02,  1.56673501135695E-02,
         4.89682519198738E-01,  4.89682519198738E-01,  1.56673501135695E-02,
         1.25820817014127E-01,  4.37089591492937E-01,  3.89137705023870E-02,
         4.37089591492937E-01,  1.25820817014127E-01,  3.89137705023870E-02,
         4.37089591492937E-01,  4.37089591492937E-01,  3.89137705023870E-02,
         6.23592928761935E-01,  1.88203535619033E-01,  3.98238694636050E-02,
         1.88203535619033E-01,  6.23592928761935E-01,  3.98238694636050E-02,
         1.88203535619033E-01,  1.88203535619033E-01,  3.98238694636050E-02,
         9.10540973211095E-01,  4.47295133944530E-02,  1.27888378293490E-02,
         4.47295133944530E-02,  9.10540973211095E-01,  1.27888378293490E-02,
         4.47295133944530E-02,  4.47295133944530E-02,  1.27888378293490E-02,
         3.68384120547360E-02,  2.21962989160766E-01,  2.16417696886445E-02,
         7.41198598784498E-01,  3.68384120547360E-02,  2.16417696886445E-02,
         2.21962989160766E-01,  7.41198598784498E-01,  2.16417696886445E-02,
         3.68384120547360E-02,  7.41198598784498E-01,  2.16417696886445E-02,
         7.41198598784498E-01,  2.21962989160766E-01,  2.16417696886445E-02,
         2.21962989160766E-01,  3.68384120547360E-02,  2.16417696886445E-02
      };
   for (int i = 0; i < 19; i++)
   {
      TriangleIntRules[9]->IntPoint(i).x      = rule_19[3*i+0];
      TriangleIntRules[9]->IntPoint(i).y      = rule_19[3*i+1];
      TriangleIntRules[9]->IntPoint(i).weight = rule_19[3*i+2];
   }

   // 25 point - 10 degree
   TriangleIntRules[10] = new IntegrationRule(25);
   const double rule_25[] =
      {
         3.33333333333333E-01,  3.33333333333333E-01,  4.54089951913770E-02,
         2.88447332326850E-02,  4.85577633383657E-01,  1.83629788782335E-02,
         4.85577633383657E-01,  2.88447332326850E-02,  1.83629788782335E-02,
         4.85577633383657E-01,  4.85577633383657E-01,  1.83629788782335E-02,
         7.81036849029926E-01,  1.09481575485037E-01,  2.26605297177640E-02,
         1.09481575485037E-01,  7.81036849029926E-01,  2.26605297177640E-02,
         1.09481575485037E-01,  1.09481575485037E-01,  2.26605297177640E-02,
         1.41707219414880E-01,  3.07939838764121E-01,  3.63789584227100E-02,
         5.50352941820999E-01,  1.41707219414880E-01,  3.63789584227100E-02,
         3.07939838764121E-01,  5.50352941820999E-01,  3.63789584227100E-02,
         1.41707219414880E-01,  5.50352941820999E-01,  3.63789584227100E-02,
         5.50352941820999E-01,  3.07939838764121E-01,  3.63789584227100E-02,
         3.07939838764121E-01,  1.41707219414880E-01,  3.63789584227100E-02,
         2.50035347626860E-02,  2.46672560639903E-01,  1.41636212655285E-02,
         7.28323904597411E-01,  2.50035347626860E-02,  1.41636212655285E-02,
         2.46672560639903E-01,  7.28323904597411E-01,  1.41636212655285E-02,
         2.50035347626860E-02,  7.28323904597411E-01,  1.41636212655285E-02,
         7.28323904597411E-01,  2.46672560639903E-01,  1.41636212655285E-02,
         2.46672560639903E-01,  2.50035347626860E-02,  1.41636212655285E-02,
         9.54081540029900E-03,  6.68032510122000E-02,  4.71083348186650E-03,
         9.23655933587500E-01,  9.54081540029900E-03,  4.71083348186650E-03,
         6.68032510122000E-02,  9.23655933587500E-01,  4.71083348186650E-03,
         9.54081540029900E-03,  9.23655933587500E-01,  4.71083348186650E-03,
         9.23655933587500E-01,  6.68032510122000E-02,  4.71083348186650E-03,
         6.68032510122000E-02,  9.54081540029900E-03,  4.71083348186650E-03
      };
   for (int i = 0; i < 25; i++)
   {
      TriangleIntRules[10]->IntPoint(i).x      = rule_25[3*i+0];
      TriangleIntRules[10]->IntPoint(i).y      = rule_25[3*i+1];
      TriangleIntRules[10]->IntPoint(i).weight = rule_25[3*i+2];
   }

   // 27 point - 11 degree  (has points outside the triangle)
#if 0
   TriangleIntRules[11] = new IntegrationRule(27);
   const double rule_27[] =
      {
         -6.92220965415170E-02,  5.34611048270758E-01,  4.63503164480500E-04,
         5.34611048270758E-01, -6.92220965415170E-02,  4.63503164480500E-04,
         5.34611048270758E-01,  5.34611048270758E-01,  4.63503164480500E-04,
         2.02061394068290E-01,  3.98969302965855E-01,  3.85747674574065E-02,
         3.98969302965855E-01,  2.02061394068290E-01,  3.85747674574065E-02,
         3.98969302965855E-01,  3.98969302965855E-01,  3.85747674574065E-02,
         5.93380199137435E-01,  2.03309900431282E-01,  2.96614886903870E-02,
         2.03309900431282E-01,  5.93380199137435E-01,  2.96614886903870E-02,
         2.03309900431282E-01,  2.03309900431282E-01,  2.96614886903870E-02,
         7.61298175434837E-01,  1.19350912282581E-01,  1.80922702517090E-02,
         1.19350912282581E-01,  7.61298175434837E-01,  1.80922702517090E-02,
         1.19350912282581E-01,  1.19350912282581E-01,  1.80922702517090E-02,
         9.35270103777448E-01,  3.23649481112760E-02,  6.82986550133900E-03,
         3.23649481112760E-02,  9.35270103777448E-01,  6.82986550133900E-03,
         3.23649481112760E-02,  3.23649481112760E-02,  6.82986550133900E-03,
         5.01781383104950E-02,  3.56620648261293E-01,  2.61685559811020E-02,
         5.93201213428213E-01,  5.01781383104950E-02,  2.61685559811020E-02,
         3.56620648261293E-01,  5.93201213428213E-01,  2.61685559811020E-02,
         5.01781383104950E-02,  5.93201213428213E-01,  2.61685559811020E-02,
         5.93201213428213E-01,  3.56620648261293E-01,  2.61685559811020E-02,
         3.56620648261293E-01,  5.01781383104950E-02,  2.61685559811020E-02,
         2.10220165361660E-02,  1.71488980304042E-01,  1.03538298195705E-02,
         8.07489003159792E-01,  2.10220165361660E-02,  1.03538298195705E-02,
         1.71488980304042E-01,  8.07489003159792E-01,  1.03538298195705E-02,
         2.10220165361660E-02,  8.07489003159792E-01,  1.03538298195705E-02,
         8.07489003159792E-01,  1.71488980304042E-01,  1.03538298195705E-02,
         1.71488980304042E-01,  2.10220165361660E-02,  1.03538298195705E-02
      };
   for (int i = 0; i < 27; i++)
   {
      TriangleIntRules[11]->IntPoint(i).x      = rule_27[3*i+0];
      TriangleIntRules[11]->IntPoint(i).y      = rule_27[3*i+1];
      TriangleIntRules[11]->IntPoint(i).weight = rule_27[3*i+2];
   }
#endif

   // 33 point - 12 degree
   TriangleIntRules[11] = new IntegrationRule(33);
   const double rule_33[] =
      {
         2.35652204523900E-02,  4.88217389773805E-01,  1.28655332202275E-02,
         4.88217389773805E-01,  2.35652204523900E-02,  1.28655332202275E-02,
         4.88217389773805E-01,  4.88217389773805E-01,  1.28655332202275E-02,
         1.20551215411079E-01,  4.39724392294460E-01,  2.18462722690190E-02,
         4.39724392294460E-01,  1.20551215411079E-01,  2.18462722690190E-02,
         4.39724392294460E-01,  4.39724392294460E-01,  2.18462722690190E-02,
         4.57579229975768E-01,  2.71210385012116E-01,  3.14291121089425E-02,
         2.71210385012116E-01,  4.57579229975768E-01,  3.14291121089425E-02,
         2.71210385012116E-01,  2.71210385012116E-01,  3.14291121089425E-02,
         7.44847708916828E-01,  1.27576145541586E-01,  1.73980564653545E-02,
         1.27576145541586E-01,  7.44847708916828E-01,  1.73980564653545E-02,
         1.27576145541586E-01,  1.27576145541586E-01,  1.73980564653545E-02,
         9.57365299093579E-01,  2.13173504532100E-02,  3.08313052577950E-03,
         2.13173504532100E-02,  9.57365299093579E-01,  3.08313052577950E-03,
         2.13173504532100E-02,  2.13173504532100E-02,  3.08313052577950E-03,
         1.15343494534698E-01,  2.75713269685514E-01,  2.01857788831905E-02,
         6.08943235779788E-01,  1.15343494534698E-01,  2.01857788831905E-02,
         2.75713269685514E-01,  6.08943235779788E-01,  2.01857788831905E-02,
         1.15343494534698E-01,  6.08943235779788E-01,  2.01857788831905E-02,
         6.08943235779788E-01,  2.75713269685514E-01,  2.01857788831905E-02,
         2.75713269685514E-01,  1.15343494534698E-01,  2.01857788831905E-02,
         2.28383322222570E-02,  2.81325580989940E-01,  1.11783866011515E-02,
         6.95836086787803E-01,  2.28383322222570E-02,  1.11783866011515E-02,
         2.81325580989940E-01,  6.95836086787803E-01,  1.11783866011515E-02,
         2.28383322222570E-02,  6.95836086787803E-01,  1.11783866011515E-02,
         6.95836086787803E-01,  2.81325580989940E-01,  1.11783866011515E-02,
         2.81325580989940E-01,  2.28383322222570E-02,  1.11783866011515E-02,
         2.57340505483300E-02,  1.16251915907597E-01,  8.65811555432950E-03,
         8.58014033544073E-01,  2.57340505483300E-02,  8.65811555432950E-03,
         1.16251915907597E-01,  8.58014033544073E-01,  8.65811555432950E-03,
         2.57340505483300E-02,  8.58014033544073E-01,  8.65811555432950E-03,
         8.58014033544073E-01,  1.16251915907597E-01,  8.65811555432950E-03,
         1.16251915907597E-01,  2.57340505483300E-02,  8.65811555432950E-03
      };
   for (int i = 0; i < 33; i++)
   {
      TriangleIntRules[11]->IntPoint(i).x      = rule_33[3*i+0];
      TriangleIntRules[11]->IntPoint(i).y      = rule_33[3*i+1];
      TriangleIntRules[11]->IntPoint(i).weight = rule_33[3*i+2];
   }

   // 33 point - 12 degree
   TriangleIntRules[12] = new IntegrationRule(33);
   for (int i = 0; i < 33; i++)
   {
      TriangleIntRules[12]->IntPoint(i).x      = rule_33[3*i+0];
      TriangleIntRules[12]->IntPoint(i).y      = rule_33[3*i+1];
      TriangleIntRules[12]->IntPoint(i).weight = rule_33[3*i+2];
   }
}

// Integration rules for unit square
void IntegrationRules::SquareIntegrationRules()
{
   SquareIntRules.SetSize(20);

   int i,k,s,np;

   for (i = 0; i < SquareIntRules.Size(); i++)
   {
      np = SegmentIntRules[i] -> GetNPoints();
      SquareIntRules[i] = new IntegrationRule(np*np);
      for (k = 0; k < np; k++)
         for (s = 0; s < np; s++)
         {
            SquareIntRules[i] -> IntPoint(k*np+s).x
               = SegmentIntRules[i] -> IntPoint(k).x ;

            SquareIntRules[i] -> IntPoint(k*np+s).y
               = SegmentIntRules[i] -> IntPoint(s).x ;

            SquareIntRules[i] -> IntPoint(k*np+s).weight
               = SegmentIntRules[i] -> IntPoint(k).weight
               * SegmentIntRules[i] -> IntPoint(s).weight;
         }
   }
}

/** Integration rules for reference tetrahedron
    {[0,0,0],[1,0,0],[0,1,0],[0,0,1]}          */
void IntegrationRules::TetrahedronIntegrationRules(int refined)
{
   TetrahedronIntRules.SetSize(9);

   if (refined)
      mfem_error ("Refined TetrahedronIntegrationRules are not implemented!");

   for (int i = 0; i < TetrahedronIntRules.Size(); i++)
      TetrahedronIntRules[i] = NULL;


   // 1 point - degree 1
   TetrahedronIntRules[0] = new IntegrationRule (1);

   TetrahedronIntRules[0] -> IntPoint(0).x = 0.25;
   TetrahedronIntRules[0] -> IntPoint(0).y = 0.25;
   TetrahedronIntRules[0] -> IntPoint(0).z = 0.25;
   TetrahedronIntRules[0] -> IntPoint(0).weight = 0.1666666666666666667;

   // 1 point - degree 1
   TetrahedronIntRules[1] = new IntegrationRule (1);

   TetrahedronIntRules[1] -> IntPoint(0).x = 0.25;
   TetrahedronIntRules[1] -> IntPoint(0).y = 0.25;
   TetrahedronIntRules[1] -> IntPoint(0).z = 0.25;
   TetrahedronIntRules[1] -> IntPoint(0).weight = 0.1666666666666666667;

   // 4 points - degree 2
   TetrahedronIntRules[2] = new IntegrationRule (4);

   TetrahedronIntRules[2] -> IntPoint(0).x = 0.58541019662496845446;
   TetrahedronIntRules[2] -> IntPoint(0).y = 0.13819660112501051518;
   TetrahedronIntRules[2] -> IntPoint(0).z = 0.13819660112501051518;
   TetrahedronIntRules[2] -> IntPoint(0).weight = 0.041666666666666666667;
   TetrahedronIntRules[2] -> IntPoint(1).x = 0.13819660112501051518;
   TetrahedronIntRules[2] -> IntPoint(1).y = 0.58541019662496845446;
   TetrahedronIntRules[2] -> IntPoint(1).z = 0.13819660112501051518;
   TetrahedronIntRules[2] -> IntPoint(1).weight = 0.041666666666666666667;
   TetrahedronIntRules[2] -> IntPoint(2).x = 0.13819660112501051518;
   TetrahedronIntRules[2] -> IntPoint(2).y = 0.13819660112501051518;
   TetrahedronIntRules[2] -> IntPoint(2).z = 0.58541019662496845446;
   TetrahedronIntRules[2] -> IntPoint(2).weight = 0.041666666666666666667;
   TetrahedronIntRules[2] -> IntPoint(3).x = 0.13819660112501051518;
   TetrahedronIntRules[2] -> IntPoint(3).y = 0.13819660112501051518;
   TetrahedronIntRules[2] -> IntPoint(3).z = 0.13819660112501051518;
   TetrahedronIntRules[2] -> IntPoint(3).weight = 0.041666666666666666667;

   // 5 points - degree 3
   TetrahedronIntRules[3] = new IntegrationRule (5);

   TetrahedronIntRules[3] -> IntPoint(0).x = 0.25;
   TetrahedronIntRules[3] -> IntPoint(0).y = 0.25;
   TetrahedronIntRules[3] -> IntPoint(0).z = 0.25;
   TetrahedronIntRules[3] -> IntPoint(0).weight = -0.13333333333333333333;
   TetrahedronIntRules[3] -> IntPoint(1).x = 0.5;
   TetrahedronIntRules[3] -> IntPoint(1).y = 0.1666666666666666667;
   TetrahedronIntRules[3] -> IntPoint(1).z = 0.1666666666666666667;
   TetrahedronIntRules[3] -> IntPoint(1).weight = 0.075;
   TetrahedronIntRules[3] -> IntPoint(2).x = 0.1666666666666666667;
   TetrahedronIntRules[3] -> IntPoint(2).y = 0.5;
   TetrahedronIntRules[3] -> IntPoint(2).z = 0.1666666666666666667;
   TetrahedronIntRules[3] -> IntPoint(2).weight = 0.075;
   TetrahedronIntRules[3] -> IntPoint(3).x = 0.1666666666666666667;
   TetrahedronIntRules[3] -> IntPoint(3).y = 0.1666666666666666667;
   TetrahedronIntRules[3] -> IntPoint(3).z = 0.5;
   TetrahedronIntRules[3] -> IntPoint(3).weight = 0.075;
   TetrahedronIntRules[3] -> IntPoint(4).x = 0.1666666666666666667;
   TetrahedronIntRules[3] -> IntPoint(4).y = 0.1666666666666666667;
   TetrahedronIntRules[3] -> IntPoint(4).z = 0.1666666666666666667;
   TetrahedronIntRules[3] -> IntPoint(4).weight = 0.075;

   // 11 points - degree 4
   TetrahedronIntRules[4] = new IntegrationRule (11);

   TetrahedronIntRules[4] -> IntPoint(0).x = 0.25;
   TetrahedronIntRules[4] -> IntPoint(0).y = 0.25;
   TetrahedronIntRules[4] -> IntPoint(0).z = 0.25;
   TetrahedronIntRules[4] -> IntPoint(0).weight = -0.013155555555555555556;
   TetrahedronIntRules[4] -> IntPoint(1).x = 0.78571428571428571429;
   TetrahedronIntRules[4] -> IntPoint(1).y = 0.071428571428571428571;
   TetrahedronIntRules[4] -> IntPoint(1).z = 0.071428571428571428571;
   TetrahedronIntRules[4] -> IntPoint(1).weight = 0.0076222222222222222222;
   TetrahedronIntRules[4] -> IntPoint(2).x = 0.071428571428571428571;
   TetrahedronIntRules[4] -> IntPoint(2).y = 0.78571428571428571429;
   TetrahedronIntRules[4] -> IntPoint(2).z = 0.071428571428571428571;
   TetrahedronIntRules[4] -> IntPoint(2).weight = 0.0076222222222222222222;
   TetrahedronIntRules[4] -> IntPoint(3).x = 0.071428571428571428571;
   TetrahedronIntRules[4] -> IntPoint(3).y = 0.071428571428571428571;
   TetrahedronIntRules[4] -> IntPoint(3).z = 0.78571428571428571429;
   TetrahedronIntRules[4] -> IntPoint(3).weight = 0.0076222222222222222222;
   TetrahedronIntRules[4] -> IntPoint(4).x = 0.071428571428571428571;
   TetrahedronIntRules[4] -> IntPoint(4).y = 0.071428571428571428571;
   TetrahedronIntRules[4] -> IntPoint(4).z = 0.071428571428571428571;
   TetrahedronIntRules[4] -> IntPoint(4).weight = 0.0076222222222222222222;
   TetrahedronIntRules[4] -> IntPoint(5).x = 0.39940357616679920500;
   TetrahedronIntRules[4] -> IntPoint(5).y = 0.39940357616679920500;
   TetrahedronIntRules[4] -> IntPoint(5).z = 0.10059642383320079500;
   TetrahedronIntRules[4] -> IntPoint(5).weight = 0.024888888888888888889;
   TetrahedronIntRules[4] -> IntPoint(6).x = 0.39940357616679920500;
   TetrahedronIntRules[4] -> IntPoint(6).y = 0.10059642383320079500;
   TetrahedronIntRules[4] -> IntPoint(6).z = 0.39940357616679920500;
   TetrahedronIntRules[4] -> IntPoint(6).weight = 0.024888888888888888889;
   TetrahedronIntRules[4] -> IntPoint(7).x = 0.39940357616679920500;
   TetrahedronIntRules[4] -> IntPoint(7).y = 0.10059642383320079500;
   TetrahedronIntRules[4] -> IntPoint(7).z = 0.10059642383320079500;
   TetrahedronIntRules[4] -> IntPoint(7).weight = 0.024888888888888888889;
   TetrahedronIntRules[4] -> IntPoint(8).x = 0.10059642383320079500;
   TetrahedronIntRules[4] -> IntPoint(8).y = 0.39940357616679920500;
   TetrahedronIntRules[4] -> IntPoint(8).z = 0.39940357616679920500;
   TetrahedronIntRules[4] -> IntPoint(8).weight = 0.024888888888888888889;
   TetrahedronIntRules[4] -> IntPoint(9).x = 0.10059642383320079500;
   TetrahedronIntRules[4] -> IntPoint(9).y = 0.39940357616679920500;
   TetrahedronIntRules[4] -> IntPoint(9).z = 0.10059642383320079500;
   TetrahedronIntRules[4] -> IntPoint(9).weight = 0.024888888888888888889;
   TetrahedronIntRules[4] -> IntPoint(10).x = 0.10059642383320079500;
   TetrahedronIntRules[4] -> IntPoint(10).y = 0.10059642383320079500;
   TetrahedronIntRules[4] -> IntPoint(10).z = 0.39940357616679920500;
   TetrahedronIntRules[4] -> IntPoint(10).weight = 0.024888888888888888889;

   // 15 points - degree 5
   TetrahedronIntRules[5] = new IntegrationRule (15);

   TetrahedronIntRules[5] -> IntPoint( 0).weight = +0.0060267857142857;
   TetrahedronIntRules[5] -> IntPoint( 1).weight = +0.0060267857142857;
   TetrahedronIntRules[5] -> IntPoint( 2).weight = +0.0060267857142857;
   TetrahedronIntRules[5] -> IntPoint( 3).weight = +0.0060267857142857;
   TetrahedronIntRules[5] -> IntPoint( 4).weight = +0.0302836780970892;
   TetrahedronIntRules[5] -> IntPoint( 5).weight = +0.0116452490860290;
   TetrahedronIntRules[5] -> IntPoint( 6).weight = +0.0116452490860290;
   TetrahedronIntRules[5] -> IntPoint( 7).weight = +0.0116452490860290;
   TetrahedronIntRules[5] -> IntPoint( 8).weight = +0.0116452490860290;
   TetrahedronIntRules[5] -> IntPoint( 9).weight = +0.0109491415613865;
   TetrahedronIntRules[5] -> IntPoint(10).weight = +0.0109491415613865;
   TetrahedronIntRules[5] -> IntPoint(11).weight = +0.0109491415613865;
   TetrahedronIntRules[5] -> IntPoint(12).weight = +0.0109491415613865;
   TetrahedronIntRules[5] -> IntPoint(13).weight = +0.0109491415613865;
   TetrahedronIntRules[5] -> IntPoint(14).weight = +0.0109491415613865;
   TetrahedronIntRules[5] -> IntPoint( 0).x = +0.3333333333333333;
   TetrahedronIntRules[5] -> IntPoint( 0).y = +0.3333333333333333;
   TetrahedronIntRules[5] -> IntPoint( 0).z = +0.3333333333333333;
   TetrahedronIntRules[5] -> IntPoint( 1).x = +0.3333333333333333;
   TetrahedronIntRules[5] -> IntPoint( 1).y = +0.3333333333333333;
   TetrahedronIntRules[5] -> IntPoint( 1).z = +0.0000000000000000;
   TetrahedronIntRules[5] -> IntPoint( 2).x = +0.3333333333333333;
   TetrahedronIntRules[5] -> IntPoint( 2).y = +0.0000000000000000;
   TetrahedronIntRules[5] -> IntPoint( 2).z = +0.3333333333333333;
   TetrahedronIntRules[5] -> IntPoint( 3).x = +0.0000000000000000;
   TetrahedronIntRules[5] -> IntPoint( 3).y = +0.3333333333333333;
   TetrahedronIntRules[5] -> IntPoint( 3).z = +0.3333333333333333;
   TetrahedronIntRules[5] -> IntPoint( 4).x = +0.2500000000000000;
   TetrahedronIntRules[5] -> IntPoint( 4).y = +0.2500000000000000;
   TetrahedronIntRules[5] -> IntPoint( 4).z = +0.2500000000000000;
   TetrahedronIntRules[5] -> IntPoint( 5).x = +0.0909090909090909;
   TetrahedronIntRules[5] -> IntPoint( 5).y = +0.0909090909090909;
   TetrahedronIntRules[5] -> IntPoint( 5).z = +0.0909090909090909;
   TetrahedronIntRules[5] -> IntPoint( 6).x = +0.0909090909090909;
   TetrahedronIntRules[5] -> IntPoint( 6).y = +0.0909090909090909;
   TetrahedronIntRules[5] -> IntPoint( 6).z = +0.7272727272727273;
   TetrahedronIntRules[5] -> IntPoint( 7).x = +0.0909090909090909;
   TetrahedronIntRules[5] -> IntPoint( 7).y = +0.7272727272727273;
   TetrahedronIntRules[5] -> IntPoint( 7).z = +0.0909090909090909;
   TetrahedronIntRules[5] -> IntPoint( 8).x = +0.7272727272727273;
   TetrahedronIntRules[5] -> IntPoint( 8).y = +0.0909090909090909;
   TetrahedronIntRules[5] -> IntPoint( 8).z = +0.0909090909090909;
   TetrahedronIntRules[5] -> IntPoint( 9).x = +0.0665501535736643;
   TetrahedronIntRules[5] -> IntPoint( 9).y = +0.0665501535736643;
   TetrahedronIntRules[5] -> IntPoint( 9).z = +0.4334498464263357;
   TetrahedronIntRules[5] -> IntPoint(10).x = +0.0665501535736643;
   TetrahedronIntRules[5] -> IntPoint(10).y = +0.4334498464263357;
   TetrahedronIntRules[5] -> IntPoint(10).z = +0.4334498464263357;
   TetrahedronIntRules[5] -> IntPoint(11).x = +0.4334498464263357;
   TetrahedronIntRules[5] -> IntPoint(11).y = +0.4334498464263357;
   TetrahedronIntRules[5] -> IntPoint(11).z = +0.0665501535736643;
   TetrahedronIntRules[5] -> IntPoint(12).x = +0.4334498464263357;
   TetrahedronIntRules[5] -> IntPoint(12).y = +0.0665501535736643;
   TetrahedronIntRules[5] -> IntPoint(12).z = +0.0665501535736643;
   TetrahedronIntRules[5] -> IntPoint(13).x = +0.0665501535736643;
   TetrahedronIntRules[5] -> IntPoint(13).y = +0.4334498464263357;
   TetrahedronIntRules[5] -> IntPoint(13).z = +0.0665501535736643;
   TetrahedronIntRules[5] -> IntPoint(14).x = +0.4334498464263357;
   TetrahedronIntRules[5] -> IntPoint(14).y = +0.0665501535736643;
   TetrahedronIntRules[5] -> IntPoint(14).z = +0.4334498464263357;

   // 24 points - degree 6
   TetrahedronIntRules[6] = new IntegrationRule (24);

   TetrahedronIntRules[6] -> IntPoint( 0).weight = +0.0066537917096946;
   TetrahedronIntRules[6] -> IntPoint( 1).weight = +0.0066537917096946;
   TetrahedronIntRules[6] -> IntPoint( 2).weight = +0.0066537917096946;
   TetrahedronIntRules[6] -> IntPoint( 3).weight = +0.0066537917096946;
   TetrahedronIntRules[6] -> IntPoint( 4).weight = +0.0016795351758868;
   TetrahedronIntRules[6] -> IntPoint( 5).weight = +0.0016795351758868;
   TetrahedronIntRules[6] -> IntPoint( 6).weight = +0.0016795351758868;
   TetrahedronIntRules[6] -> IntPoint( 7).weight = +0.0016795351758868;
   TetrahedronIntRules[6] -> IntPoint( 8).weight = +0.0092261969239424;
   TetrahedronIntRules[6] -> IntPoint( 9).weight = +0.0092261969239424;
   TetrahedronIntRules[6] -> IntPoint(10).weight = +0.0092261969239424;
   TetrahedronIntRules[6] -> IntPoint(11).weight = +0.0092261969239424;
   TetrahedronIntRules[6] -> IntPoint(12).weight = +0.0080357142857143;
   TetrahedronIntRules[6] -> IntPoint(13).weight = +0.0080357142857143;
   TetrahedronIntRules[6] -> IntPoint(14).weight = +0.0080357142857143;
   TetrahedronIntRules[6] -> IntPoint(15).weight = +0.0080357142857143;
   TetrahedronIntRules[6] -> IntPoint(16).weight = +0.0080357142857143;
   TetrahedronIntRules[6] -> IntPoint(17).weight = +0.0080357142857143;
   TetrahedronIntRules[6] -> IntPoint(18).weight = +0.0080357142857143;
   TetrahedronIntRules[6] -> IntPoint(19).weight = +0.0080357142857143;
   TetrahedronIntRules[6] -> IntPoint(20).weight = +0.0080357142857143;
   TetrahedronIntRules[6] -> IntPoint(21).weight = +0.0080357142857143;
   TetrahedronIntRules[6] -> IntPoint(22).weight = +0.0080357142857143;
   TetrahedronIntRules[6] -> IntPoint(23).weight = +0.0080357142857143;
   TetrahedronIntRules[6] -> IntPoint( 0).x = +0.2146028712591517;
   TetrahedronIntRules[6] -> IntPoint( 0).y = +0.2146028712591517;
   TetrahedronIntRules[6] -> IntPoint( 0).z = +0.2146028712591517;
   TetrahedronIntRules[6] -> IntPoint( 1).x = +0.2146028712591517;
   TetrahedronIntRules[6] -> IntPoint( 1).y = +0.2146028712591517;
   TetrahedronIntRules[6] -> IntPoint( 1).z = +0.3561913862225449;
   TetrahedronIntRules[6] -> IntPoint( 2).x = +0.2146028712591517;
   TetrahedronIntRules[6] -> IntPoint( 2).y = +0.3561913862225449;
   TetrahedronIntRules[6] -> IntPoint( 2).z = +0.2146028712591517;
   TetrahedronIntRules[6] -> IntPoint( 3).x = +0.3561913862225449;
   TetrahedronIntRules[6] -> IntPoint( 3).y = +0.2146028712591517;
   TetrahedronIntRules[6] -> IntPoint( 3).z = +0.2146028712591517;
   TetrahedronIntRules[6] -> IntPoint( 4).x = +0.0406739585346113;
   TetrahedronIntRules[6] -> IntPoint( 4).y = +0.0406739585346113;
   TetrahedronIntRules[6] -> IntPoint( 4).z = +0.0406739585346113;
   TetrahedronIntRules[6] -> IntPoint( 5).x = +0.0406739585346113;
   TetrahedronIntRules[6] -> IntPoint( 5).y = +0.0406739585346113;
   TetrahedronIntRules[6] -> IntPoint( 5).z = +0.8779781243961660;
   TetrahedronIntRules[6] -> IntPoint( 6).x = +0.0406739585346113;
   TetrahedronIntRules[6] -> IntPoint( 6).y = +0.8779781243961660;
   TetrahedronIntRules[6] -> IntPoint( 6).z = +0.0406739585346113;
   TetrahedronIntRules[6] -> IntPoint( 7).x = +0.8779781243961660;
   TetrahedronIntRules[6] -> IntPoint( 7).y = +0.0406739585346113;
   TetrahedronIntRules[6] -> IntPoint( 7).z = +0.0406739585346113;
   TetrahedronIntRules[6] -> IntPoint( 8).x = +0.3223378901422757;
   TetrahedronIntRules[6] -> IntPoint( 8).y = +0.3223378901422757;
   TetrahedronIntRules[6] -> IntPoint( 8).z = +0.3223378901422757;
   TetrahedronIntRules[6] -> IntPoint( 9).x = +0.3223378901422757;
   TetrahedronIntRules[6] -> IntPoint( 9).y = +0.3223378901422757;
   TetrahedronIntRules[6] -> IntPoint( 9).z = +0.0329863295731731;
   TetrahedronIntRules[6] -> IntPoint(10).x = +0.3223378901422757;
   TetrahedronIntRules[6] -> IntPoint(10).y = +0.0329863295731731;
   TetrahedronIntRules[6] -> IntPoint(10).z = +0.3223378901422757;
   TetrahedronIntRules[6] -> IntPoint(11).x = +0.0329863295731731;
   TetrahedronIntRules[6] -> IntPoint(11).y = +0.3223378901422757;
   TetrahedronIntRules[6] -> IntPoint(11).z = +0.3223378901422757;
   TetrahedronIntRules[6] -> IntPoint(12).x = +0.0636610018750175;
   TetrahedronIntRules[6] -> IntPoint(12).y = +0.0636610018750175;
   TetrahedronIntRules[6] -> IntPoint(12).z = +0.2696723314583159;
   TetrahedronIntRules[6] -> IntPoint(13).x = +0.0636610018750175;
   TetrahedronIntRules[6] -> IntPoint(13).y = +0.2696723314583159;
   TetrahedronIntRules[6] -> IntPoint(13).z = +0.6030056647916491;
   TetrahedronIntRules[6] -> IntPoint(14).x = +0.2696723314583159;
   TetrahedronIntRules[6] -> IntPoint(14).y = +0.6030056647916491;
   TetrahedronIntRules[6] -> IntPoint(14).z = +0.0636610018750175;
   TetrahedronIntRules[6] -> IntPoint(15).x = +0.6030056647916491;
   TetrahedronIntRules[6] -> IntPoint(15).y = +0.0636610018750175;
   TetrahedronIntRules[6] -> IntPoint(15).z = +0.0636610018750175;
   TetrahedronIntRules[6] -> IntPoint(16).x = +0.0636610018750175;
   TetrahedronIntRules[6] -> IntPoint(16).y = +0.6030056647916491;
   TetrahedronIntRules[6] -> IntPoint(16).z = +0.0636610018750175;
   TetrahedronIntRules[6] -> IntPoint(17).x = +0.6030056647916491;
   TetrahedronIntRules[6] -> IntPoint(17).y = +0.0636610018750175;
   TetrahedronIntRules[6] -> IntPoint(17).z = +0.2696723314583159;
   TetrahedronIntRules[6] -> IntPoint(18).x = +0.0636610018750175;
   TetrahedronIntRules[6] -> IntPoint(18).y = +0.2696723314583159;
   TetrahedronIntRules[6] -> IntPoint(18).z = +0.0636610018750175;
   TetrahedronIntRules[6] -> IntPoint(19).x = +0.0636610018750175;
   TetrahedronIntRules[6] -> IntPoint(19).y = +0.6030056647916491;
   TetrahedronIntRules[6] -> IntPoint(19).z = +0.2696723314583159;
   TetrahedronIntRules[6] -> IntPoint(20).x = +0.2696723314583159;
   TetrahedronIntRules[6] -> IntPoint(20).y = +0.0636610018750175;
   TetrahedronIntRules[6] -> IntPoint(20).z = +0.6030056647916491;
   TetrahedronIntRules[6] -> IntPoint(21).x = +0.6030056647916491;
   TetrahedronIntRules[6] -> IntPoint(21).y = +0.2696723314583159;
   TetrahedronIntRules[6] -> IntPoint(21).z = +0.0636610018750175;
   TetrahedronIntRules[6] -> IntPoint(22).x = +0.2696723314583159;
   TetrahedronIntRules[6] -> IntPoint(22).y = +0.0636610018750175;
   TetrahedronIntRules[6] -> IntPoint(22).z = +0.0636610018750175;
   TetrahedronIntRules[6] -> IntPoint(23).x = +0.0636610018750175;
   TetrahedronIntRules[6] -> IntPoint(23).y = +0.0636610018750175;
   TetrahedronIntRules[6] -> IntPoint(23).z = +0.6030056647916491;

   // 31 points - degree 7
   TetrahedronIntRules[7] = new IntegrationRule (31);

   TetrahedronIntRules[7] -> IntPoint( 0).weight = +0.0009700176366843;
   TetrahedronIntRules[7] -> IntPoint( 1).weight = +0.0009700176366843;
   TetrahedronIntRules[7] -> IntPoint( 2).weight = +0.0009700176366843;
   TetrahedronIntRules[7] -> IntPoint( 3).weight = +0.0009700176366843;
   TetrahedronIntRules[7] -> IntPoint( 4).weight = +0.0009700176366843;
   TetrahedronIntRules[7] -> IntPoint( 5).weight = +0.0009700176366843;
   TetrahedronIntRules[7] -> IntPoint( 6).weight = +0.0182642234661088;
   TetrahedronIntRules[7] -> IntPoint( 7).weight = +0.0105999415244142;
   TetrahedronIntRules[7] -> IntPoint( 8).weight = +0.0105999415244142;
   TetrahedronIntRules[7] -> IntPoint( 9).weight = +0.0105999415244142;
   TetrahedronIntRules[7] -> IntPoint(10).weight = +0.0105999415244142;
   TetrahedronIntRules[7] -> IntPoint(11).weight = -0.0625177401143300;
   TetrahedronIntRules[7] -> IntPoint(12).weight = -0.0625177401143300;
   TetrahedronIntRules[7] -> IntPoint(13).weight = -0.0625177401143300;
   TetrahedronIntRules[7] -> IntPoint(14).weight = -0.0625177401143300;
   TetrahedronIntRules[7] -> IntPoint(15).weight = +0.0048914252630735;
   TetrahedronIntRules[7] -> IntPoint(16).weight = +0.0048914252630735;
   TetrahedronIntRules[7] -> IntPoint(17).weight = +0.0048914252630735;
   TetrahedronIntRules[7] -> IntPoint(18).weight = +0.0048914252630735;
   TetrahedronIntRules[7] -> IntPoint(19).weight = +0.0275573192239851;
   TetrahedronIntRules[7] -> IntPoint(20).weight = +0.0275573192239851;
   TetrahedronIntRules[7] -> IntPoint(21).weight = +0.0275573192239851;
   TetrahedronIntRules[7] -> IntPoint(22).weight = +0.0275573192239851;
   TetrahedronIntRules[7] -> IntPoint(23).weight = +0.0275573192239851;
   TetrahedronIntRules[7] -> IntPoint(24).weight = +0.0275573192239851;
   TetrahedronIntRules[7] -> IntPoint(25).weight = +0.0275573192239851;
   TetrahedronIntRules[7] -> IntPoint(26).weight = +0.0275573192239851;
   TetrahedronIntRules[7] -> IntPoint(27).weight = +0.0275573192239851;
   TetrahedronIntRules[7] -> IntPoint(28).weight = +0.0275573192239851;
   TetrahedronIntRules[7] -> IntPoint(29).weight = +0.0275573192239851;
   TetrahedronIntRules[7] -> IntPoint(30).weight = +0.0275573192239851;
   TetrahedronIntRules[7] -> IntPoint( 0).x = +0.5000000000000000;
   TetrahedronIntRules[7] -> IntPoint( 0).y = +0.5000000000000000;
   TetrahedronIntRules[7] -> IntPoint( 0).z = +0.0000000000000000;
   TetrahedronIntRules[7] -> IntPoint( 1).x = +0.5000000000000000;
   TetrahedronIntRules[7] -> IntPoint( 1).y = +0.0000000000000000;
   TetrahedronIntRules[7] -> IntPoint( 1).z = +0.0000000000000000;
   TetrahedronIntRules[7] -> IntPoint( 2).x = +0.0000000000000000;
   TetrahedronIntRules[7] -> IntPoint( 2).y = +0.0000000000000000;
   TetrahedronIntRules[7] -> IntPoint( 2).z = +0.5000000000000000;
   TetrahedronIntRules[7] -> IntPoint( 3).x = +0.0000000000000000;
   TetrahedronIntRules[7] -> IntPoint( 3).y = +0.5000000000000000;
   TetrahedronIntRules[7] -> IntPoint( 3).z = +0.5000000000000000;
   TetrahedronIntRules[7] -> IntPoint( 4).x = +0.5000000000000000;
   TetrahedronIntRules[7] -> IntPoint( 4).y = +0.0000000000000000;
   TetrahedronIntRules[7] -> IntPoint( 4).z = +0.5000000000000000;
   TetrahedronIntRules[7] -> IntPoint( 5).x = +0.0000000000000000;
   TetrahedronIntRules[7] -> IntPoint( 5).y = +0.5000000000000000;
   TetrahedronIntRules[7] -> IntPoint( 5).z = +0.0000000000000000;
   TetrahedronIntRules[7] -> IntPoint( 6).x = +0.2500000000000000;
   TetrahedronIntRules[7] -> IntPoint( 6).y = +0.2500000000000000;
   TetrahedronIntRules[7] -> IntPoint( 6).z = +0.2500000000000000;
   TetrahedronIntRules[7] -> IntPoint( 7).x = +0.0782131923303186;
   TetrahedronIntRules[7] -> IntPoint( 7).y = +0.0782131923303186;
   TetrahedronIntRules[7] -> IntPoint( 7).z = +0.0782131923303186;
   TetrahedronIntRules[7] -> IntPoint( 8).x = +0.0782131923303186;
   TetrahedronIntRules[7] -> IntPoint( 8).y = +0.0782131923303186;
   TetrahedronIntRules[7] -> IntPoint( 8).z = +0.7653604230090441;
   TetrahedronIntRules[7] -> IntPoint( 9).x = +0.0782131923303186;
   TetrahedronIntRules[7] -> IntPoint( 9).y = +0.7653604230090441;
   TetrahedronIntRules[7] -> IntPoint( 9).z = +0.0782131923303186;
   TetrahedronIntRules[7] -> IntPoint(10).x = +0.7653604230090441;
   TetrahedronIntRules[7] -> IntPoint(10).y = +0.0782131923303186;
   TetrahedronIntRules[7] -> IntPoint(10).z = +0.0782131923303186;
   TetrahedronIntRules[7] -> IntPoint(11).x = +0.1218432166639044;
   TetrahedronIntRules[7] -> IntPoint(11).y = +0.1218432166639044;
   TetrahedronIntRules[7] -> IntPoint(11).z = +0.1218432166639044;
   TetrahedronIntRules[7] -> IntPoint(12).x = +0.1218432166639044;
   TetrahedronIntRules[7] -> IntPoint(12).y = +0.1218432166639044;
   TetrahedronIntRules[7] -> IntPoint(12).z = +0.6344703500082868;
   TetrahedronIntRules[7] -> IntPoint(13).x = +0.1218432166639044;
   TetrahedronIntRules[7] -> IntPoint(13).y = +0.6344703500082868;
   TetrahedronIntRules[7] -> IntPoint(13).z = +0.1218432166639044;
   TetrahedronIntRules[7] -> IntPoint(14).x = +0.6344703500082868;
   TetrahedronIntRules[7] -> IntPoint(14).y = +0.1218432166639044;
   TetrahedronIntRules[7] -> IntPoint(14).z = +0.1218432166639044;
   TetrahedronIntRules[7] -> IntPoint(15).x = +0.3325391644464206;
   TetrahedronIntRules[7] -> IntPoint(15).y = +0.3325391644464206;
   TetrahedronIntRules[7] -> IntPoint(15).z = +0.3325391644464206;
   TetrahedronIntRules[7] -> IntPoint(16).x = +0.3325391644464206;
   TetrahedronIntRules[7] -> IntPoint(16).y = +0.3325391644464206;
   TetrahedronIntRules[7] -> IntPoint(16).z = +0.0023825066607383;
   TetrahedronIntRules[7] -> IntPoint(17).x = +0.3325391644464206;
   TetrahedronIntRules[7] -> IntPoint(17).y = +0.0023825066607383;
   TetrahedronIntRules[7] -> IntPoint(17).z = +0.3325391644464206;
   TetrahedronIntRules[7] -> IntPoint(18).x = +0.0023825066607383;
   TetrahedronIntRules[7] -> IntPoint(18).y = +0.3325391644464206;
   TetrahedronIntRules[7] -> IntPoint(18).z = +0.3325391644464206;
   TetrahedronIntRules[7] -> IntPoint(19).x = +0.1000000000000000;
   TetrahedronIntRules[7] -> IntPoint(19).y = +0.1000000000000000;
   TetrahedronIntRules[7] -> IntPoint(19).z = +0.2000000000000000;
   TetrahedronIntRules[7] -> IntPoint(20).x = +0.1000000000000000;
   TetrahedronIntRules[7] -> IntPoint(20).y = +0.2000000000000000;
   TetrahedronIntRules[7] -> IntPoint(20).z = +0.6000000000000000;
   TetrahedronIntRules[7] -> IntPoint(21).x = +0.2000000000000000;
   TetrahedronIntRules[7] -> IntPoint(21).y = +0.6000000000000000;
   TetrahedronIntRules[7] -> IntPoint(21).z = +0.1000000000000000;
   TetrahedronIntRules[7] -> IntPoint(22).x = +0.6000000000000000;
   TetrahedronIntRules[7] -> IntPoint(22).y = +0.1000000000000000;
   TetrahedronIntRules[7] -> IntPoint(22).z = +0.1000000000000000;
   TetrahedronIntRules[7] -> IntPoint(23).x = +0.1000000000000000;
   TetrahedronIntRules[7] -> IntPoint(23).y = +0.6000000000000000;
   TetrahedronIntRules[7] -> IntPoint(23).z = +0.1000000000000000;
   TetrahedronIntRules[7] -> IntPoint(24).x = +0.6000000000000000;
   TetrahedronIntRules[7] -> IntPoint(24).y = +0.1000000000000000;
   TetrahedronIntRules[7] -> IntPoint(24).z = +0.2000000000000000;
   TetrahedronIntRules[7] -> IntPoint(25).x = +0.1000000000000000;
   TetrahedronIntRules[7] -> IntPoint(25).y = +0.2000000000000000;
   TetrahedronIntRules[7] -> IntPoint(25).z = +0.1000000000000000;
   TetrahedronIntRules[7] -> IntPoint(26).x = +0.1000000000000000;
   TetrahedronIntRules[7] -> IntPoint(26).y = +0.6000000000000000;
   TetrahedronIntRules[7] -> IntPoint(26).z = +0.2000000000000000;
   TetrahedronIntRules[7] -> IntPoint(27).x = +0.2000000000000000;
   TetrahedronIntRules[7] -> IntPoint(27).y = +0.1000000000000000;
   TetrahedronIntRules[7] -> IntPoint(27).z = +0.6000000000000000;
   TetrahedronIntRules[7] -> IntPoint(28).x = +0.6000000000000000;
   TetrahedronIntRules[7] -> IntPoint(28).y = +0.2000000000000000;
   TetrahedronIntRules[7] -> IntPoint(28).z = +0.1000000000000000;
   TetrahedronIntRules[7] -> IntPoint(29).x = +0.2000000000000000;
   TetrahedronIntRules[7] -> IntPoint(29).y = +0.1000000000000000;
   TetrahedronIntRules[7] -> IntPoint(29).z = +0.1000000000000000;
   TetrahedronIntRules[7] -> IntPoint(30).x = +0.1000000000000000;
   TetrahedronIntRules[7] -> IntPoint(30).y = +0.1000000000000000;
   TetrahedronIntRules[7] -> IntPoint(30).z = +0.6000000000000000;

   // 45 points - degree 8
   TetrahedronIntRules[8] = new IntegrationRule (45);

   TetrahedronIntRules[8] -> IntPoint( 0).weight = -0.0393270066412926;
   TetrahedronIntRules[8] -> IntPoint( 1).weight = +0.0040813160593427;
   TetrahedronIntRules[8] -> IntPoint( 2).weight = +0.0040813160593427;
   TetrahedronIntRules[8] -> IntPoint( 3).weight = +0.0040813160593427;
   TetrahedronIntRules[8] -> IntPoint( 4).weight = +0.0040813160593427;
   TetrahedronIntRules[8] -> IntPoint( 5).weight = +0.0006580867733043;
   TetrahedronIntRules[8] -> IntPoint( 6).weight = +0.0006580867733043;
   TetrahedronIntRules[8] -> IntPoint( 7).weight = +0.0006580867733043;
   TetrahedronIntRules[8] -> IntPoint( 8).weight = +0.0006580867733043;
   TetrahedronIntRules[8] -> IntPoint( 9).weight = +0.0043842588251228;
   TetrahedronIntRules[8] -> IntPoint(10).weight = +0.0043842588251228;
   TetrahedronIntRules[8] -> IntPoint(11).weight = +0.0043842588251228;
   TetrahedronIntRules[8] -> IntPoint(12).weight = +0.0043842588251228;
   TetrahedronIntRules[8] -> IntPoint(13).weight = +0.0043842588251228;
   TetrahedronIntRules[8] -> IntPoint(14).weight = +0.0043842588251228;
   TetrahedronIntRules[8] -> IntPoint(15).weight = +0.0138300638425098;
   TetrahedronIntRules[8] -> IntPoint(16).weight = +0.0138300638425098;
   TetrahedronIntRules[8] -> IntPoint(17).weight = +0.0138300638425098;
   TetrahedronIntRules[8] -> IntPoint(18).weight = +0.0138300638425098;
   TetrahedronIntRules[8] -> IntPoint(19).weight = +0.0138300638425098;
   TetrahedronIntRules[8] -> IntPoint(20).weight = +0.0138300638425098;
   TetrahedronIntRules[8] -> IntPoint(21).weight = +0.0042404374246837;
   TetrahedronIntRules[8] -> IntPoint(22).weight = +0.0042404374246837;
   TetrahedronIntRules[8] -> IntPoint(23).weight = +0.0042404374246837;
   TetrahedronIntRules[8] -> IntPoint(24).weight = +0.0042404374246837;
   TetrahedronIntRules[8] -> IntPoint(25).weight = +0.0042404374246837;
   TetrahedronIntRules[8] -> IntPoint(26).weight = +0.0042404374246837;
   TetrahedronIntRules[8] -> IntPoint(27).weight = +0.0042404374246837;
   TetrahedronIntRules[8] -> IntPoint(28).weight = +0.0042404374246837;
   TetrahedronIntRules[8] -> IntPoint(29).weight = +0.0042404374246837;
   TetrahedronIntRules[8] -> IntPoint(30).weight = +0.0042404374246837;
   TetrahedronIntRules[8] -> IntPoint(31).weight = +0.0042404374246837;
   TetrahedronIntRules[8] -> IntPoint(32).weight = +0.0042404374246837;
   TetrahedronIntRules[8] -> IntPoint(33).weight = +0.0022387397396142;
   TetrahedronIntRules[8] -> IntPoint(34).weight = +0.0022387397396142;
   TetrahedronIntRules[8] -> IntPoint(35).weight = +0.0022387397396142;
   TetrahedronIntRules[8] -> IntPoint(36).weight = +0.0022387397396142;
   TetrahedronIntRules[8] -> IntPoint(37).weight = +0.0022387397396142;
   TetrahedronIntRules[8] -> IntPoint(38).weight = +0.0022387397396142;
   TetrahedronIntRules[8] -> IntPoint(39).weight = +0.0022387397396142;
   TetrahedronIntRules[8] -> IntPoint(40).weight = +0.0022387397396142;
   TetrahedronIntRules[8] -> IntPoint(41).weight = +0.0022387397396142;
   TetrahedronIntRules[8] -> IntPoint(42).weight = +0.0022387397396142;
   TetrahedronIntRules[8] -> IntPoint(43).weight = +0.0022387397396142;
   TetrahedronIntRules[8] -> IntPoint(44).weight = +0.0022387397396142;
   TetrahedronIntRules[8] -> IntPoint( 0).x = +0.2500000000000000;
   TetrahedronIntRules[8] -> IntPoint( 0).y = +0.2500000000000000;
   TetrahedronIntRules[8] -> IntPoint( 0).z = +0.2500000000000000;
   TetrahedronIntRules[8] -> IntPoint( 1).x = +0.1274709365666390;
   TetrahedronIntRules[8] -> IntPoint( 1).y = +0.1274709365666390;
   TetrahedronIntRules[8] -> IntPoint( 1).z = +0.1274709365666390;
   TetrahedronIntRules[8] -> IntPoint( 2).x = +0.1274709365666390;
   TetrahedronIntRules[8] -> IntPoint( 2).y = +0.1274709365666390;
   TetrahedronIntRules[8] -> IntPoint( 2).z = +0.6175871903000830;
   TetrahedronIntRules[8] -> IntPoint( 3).x = +0.1274709365666390;
   TetrahedronIntRules[8] -> IntPoint( 3).y = +0.6175871903000830;
   TetrahedronIntRules[8] -> IntPoint( 3).z = +0.1274709365666390;
   TetrahedronIntRules[8] -> IntPoint( 4).x = +0.6175871903000830;
   TetrahedronIntRules[8] -> IntPoint( 4).y = +0.1274709365666390;
   TetrahedronIntRules[8] -> IntPoint( 4).z = +0.1274709365666390;
   TetrahedronIntRules[8] -> IntPoint( 5).x = +0.0320788303926323;
   TetrahedronIntRules[8] -> IntPoint( 5).y = +0.0320788303926323;
   TetrahedronIntRules[8] -> IntPoint( 5).z = +0.0320788303926323;
   TetrahedronIntRules[8] -> IntPoint( 6).x = +0.0320788303926323;
   TetrahedronIntRules[8] -> IntPoint( 6).y = +0.0320788303926323;
   TetrahedronIntRules[8] -> IntPoint( 6).z = +0.9037635088221031;
   TetrahedronIntRules[8] -> IntPoint( 7).x = +0.0320788303926323;
   TetrahedronIntRules[8] -> IntPoint( 7).y = +0.9037635088221031;
   TetrahedronIntRules[8] -> IntPoint( 7).z = +0.0320788303926323;
   TetrahedronIntRules[8] -> IntPoint( 8).x = +0.9037635088221031;
   TetrahedronIntRules[8] -> IntPoint( 8).y = +0.0320788303926323;
   TetrahedronIntRules[8] -> IntPoint( 8).z = +0.0320788303926323;
   TetrahedronIntRules[8] -> IntPoint( 9).x = +0.0497770956432810;
   TetrahedronIntRules[8] -> IntPoint( 9).y = +0.0497770956432810;
   TetrahedronIntRules[8] -> IntPoint( 9).z = +0.4502229043567190;
   TetrahedronIntRules[8] -> IntPoint(10).x = +0.0497770956432810;
   TetrahedronIntRules[8] -> IntPoint(10).y = +0.4502229043567190;
   TetrahedronIntRules[8] -> IntPoint(10).z = +0.4502229043567190;
   TetrahedronIntRules[8] -> IntPoint(11).x = +0.4502229043567190;
   TetrahedronIntRules[8] -> IntPoint(11).y = +0.4502229043567190;
   TetrahedronIntRules[8] -> IntPoint(11).z = +0.0497770956432810;
   TetrahedronIntRules[8] -> IntPoint(12).x = +0.4502229043567190;
   TetrahedronIntRules[8] -> IntPoint(12).y = +0.0497770956432810;
   TetrahedronIntRules[8] -> IntPoint(12).z = +0.0497770956432810;
   TetrahedronIntRules[8] -> IntPoint(13).x = +0.0497770956432810;
   TetrahedronIntRules[8] -> IntPoint(13).y = +0.4502229043567190;
   TetrahedronIntRules[8] -> IntPoint(13).z = +0.0497770956432810;
   TetrahedronIntRules[8] -> IntPoint(14).x = +0.4502229043567190;
   TetrahedronIntRules[8] -> IntPoint(14).y = +0.0497770956432810;
   TetrahedronIntRules[8] -> IntPoint(14).z = +0.4502229043567190;
   TetrahedronIntRules[8] -> IntPoint(15).x = +0.1837304473985499;
   TetrahedronIntRules[8] -> IntPoint(15).y = +0.1837304473985499;
   TetrahedronIntRules[8] -> IntPoint(15).z = +0.3162695526014501;
   TetrahedronIntRules[8] -> IntPoint(16).x = +0.1837304473985499;
   TetrahedronIntRules[8] -> IntPoint(16).y = +0.3162695526014501;
   TetrahedronIntRules[8] -> IntPoint(16).z = +0.3162695526014501;
   TetrahedronIntRules[8] -> IntPoint(17).x = +0.3162695526014501;
   TetrahedronIntRules[8] -> IntPoint(17).y = +0.3162695526014501;
   TetrahedronIntRules[8] -> IntPoint(17).z = +0.1837304473985499;
   TetrahedronIntRules[8] -> IntPoint(18).x = +0.3162695526014501;
   TetrahedronIntRules[8] -> IntPoint(18).y = +0.1837304473985499;
   TetrahedronIntRules[8] -> IntPoint(18).z = +0.1837304473985499;
   TetrahedronIntRules[8] -> IntPoint(19).x = +0.1837304473985499;
   TetrahedronIntRules[8] -> IntPoint(19).y = +0.3162695526014501;
   TetrahedronIntRules[8] -> IntPoint(19).z = +0.1837304473985499;
   TetrahedronIntRules[8] -> IntPoint(20).x = +0.3162695526014501;
   TetrahedronIntRules[8] -> IntPoint(20).y = +0.1837304473985499;
   TetrahedronIntRules[8] -> IntPoint(20).z = +0.3162695526014501;
   TetrahedronIntRules[8] -> IntPoint(21).x = +0.2319010893971509;
   TetrahedronIntRules[8] -> IntPoint(21).y = +0.2319010893971509;
   TetrahedronIntRules[8] -> IntPoint(21).z = +0.0229177878448171;
   TetrahedronIntRules[8] -> IntPoint(22).x = +0.2319010893971509;
   TetrahedronIntRules[8] -> IntPoint(22).y = +0.0229177878448171;
   TetrahedronIntRules[8] -> IntPoint(22).z = +0.5132800333608811;
   TetrahedronIntRules[8] -> IntPoint(23).x = +0.0229177878448171;
   TetrahedronIntRules[8] -> IntPoint(23).y = +0.5132800333608811;
   TetrahedronIntRules[8] -> IntPoint(23).z = +0.2319010893971509;
   TetrahedronIntRules[8] -> IntPoint(24).x = +0.5132800333608811;
   TetrahedronIntRules[8] -> IntPoint(24).y = +0.2319010893971509;
   TetrahedronIntRules[8] -> IntPoint(24).z = +0.2319010893971509;
   TetrahedronIntRules[8] -> IntPoint(25).x = +0.2319010893971509;
   TetrahedronIntRules[8] -> IntPoint(25).y = +0.5132800333608811;
   TetrahedronIntRules[8] -> IntPoint(25).z = +0.2319010893971509;
   TetrahedronIntRules[8] -> IntPoint(26).x = +0.5132800333608811;
   TetrahedronIntRules[8] -> IntPoint(26).y = +0.2319010893971509;
   TetrahedronIntRules[8] -> IntPoint(26).z = +0.0229177878448171;
   TetrahedronIntRules[8] -> IntPoint(27).x = +0.2319010893971509;
   TetrahedronIntRules[8] -> IntPoint(27).y = +0.0229177878448171;
   TetrahedronIntRules[8] -> IntPoint(27).z = +0.2319010893971509;
   TetrahedronIntRules[8] -> IntPoint(28).x = +0.2319010893971509;
   TetrahedronIntRules[8] -> IntPoint(28).y = +0.5132800333608811;
   TetrahedronIntRules[8] -> IntPoint(28).z = +0.0229177878448171;
   TetrahedronIntRules[8] -> IntPoint(29).x = +0.0229177878448171;
   TetrahedronIntRules[8] -> IntPoint(29).y = +0.2319010893971509;
   TetrahedronIntRules[8] -> IntPoint(29).z = +0.5132800333608811;
   TetrahedronIntRules[8] -> IntPoint(30).x = +0.5132800333608811;
   TetrahedronIntRules[8] -> IntPoint(30).y = +0.0229177878448171;
   TetrahedronIntRules[8] -> IntPoint(30).z = +0.2319010893971509;
   TetrahedronIntRules[8] -> IntPoint(31).x = +0.0229177878448171;
   TetrahedronIntRules[8] -> IntPoint(31).y = +0.2319010893971509;
   TetrahedronIntRules[8] -> IntPoint(31).z = +0.2319010893971509;
   TetrahedronIntRules[8] -> IntPoint(32).x = +0.2319010893971509;
   TetrahedronIntRules[8] -> IntPoint(32).y = +0.2319010893971509;
   TetrahedronIntRules[8] -> IntPoint(32).z = +0.5132800333608811;
   TetrahedronIntRules[8] -> IntPoint(33).x = +0.0379700484718286;
   TetrahedronIntRules[8] -> IntPoint(33).y = +0.0379700484718286;
   TetrahedronIntRules[8] -> IntPoint(33).z = +0.7303134278075384;
   TetrahedronIntRules[8] -> IntPoint(34).x = +0.0379700484718286;
   TetrahedronIntRules[8] -> IntPoint(34).y = +0.7303134278075384;
   TetrahedronIntRules[8] -> IntPoint(34).z = +0.1937464752488044;
   TetrahedronIntRules[8] -> IntPoint(35).x = +0.7303134278075384;
   TetrahedronIntRules[8] -> IntPoint(35).y = +0.1937464752488044;
   TetrahedronIntRules[8] -> IntPoint(35).z = +0.0379700484718286;
   TetrahedronIntRules[8] -> IntPoint(36).x = +0.1937464752488044;
   TetrahedronIntRules[8] -> IntPoint(36).y = +0.0379700484718286;
   TetrahedronIntRules[8] -> IntPoint(36).z = +0.0379700484718286;
   TetrahedronIntRules[8] -> IntPoint(37).x = +0.0379700484718286;
   TetrahedronIntRules[8] -> IntPoint(37).y = +0.1937464752488044;
   TetrahedronIntRules[8] -> IntPoint(37).z = +0.0379700484718286;
   TetrahedronIntRules[8] -> IntPoint(38).x = +0.1937464752488044;
   TetrahedronIntRules[8] -> IntPoint(38).y = +0.0379700484718286;
   TetrahedronIntRules[8] -> IntPoint(38).z = +0.7303134278075384;
   TetrahedronIntRules[8] -> IntPoint(39).x = +0.0379700484718286;
   TetrahedronIntRules[8] -> IntPoint(39).y = +0.7303134278075384;
   TetrahedronIntRules[8] -> IntPoint(39).z = +0.0379700484718286;
   TetrahedronIntRules[8] -> IntPoint(40).x = +0.0379700484718286;
   TetrahedronIntRules[8] -> IntPoint(40).y = +0.1937464752488044;
   TetrahedronIntRules[8] -> IntPoint(40).z = +0.7303134278075384;
   TetrahedronIntRules[8] -> IntPoint(41).x = +0.7303134278075384;
   TetrahedronIntRules[8] -> IntPoint(41).y = +0.0379700484718286;
   TetrahedronIntRules[8] -> IntPoint(41).z = +0.1937464752488044;
   TetrahedronIntRules[8] -> IntPoint(42).x = +0.1937464752488044;
   TetrahedronIntRules[8] -> IntPoint(42).y = +0.7303134278075384;
   TetrahedronIntRules[8] -> IntPoint(42).z = +0.0379700484718286;
   TetrahedronIntRules[8] -> IntPoint(43).x = +0.7303134278075384;
   TetrahedronIntRules[8] -> IntPoint(43).y = +0.0379700484718286;
   TetrahedronIntRules[8] -> IntPoint(43).z = +0.0379700484718286;
   TetrahedronIntRules[8] -> IntPoint(44).x = +0.0379700484718286;
   TetrahedronIntRules[8] -> IntPoint(44).y = +0.0379700484718286;
   TetrahedronIntRules[8] -> IntPoint(44).z = +0.1937464752488044;
}

/// Integration rules for reference cube
void IntegrationRules::CubeIntegrationRules()
{
   int i, k, l, m, np;

   CubeIntRules.SetSize(8);

   for (i = 0; i < CubeIntRules.Size(); i++)
      CubeIntRules[i] = NULL;


   for(i = 0; i < CubeIntRules.Size(); i++)
   {
      np = SegmentIntRules[i] -> GetNPoints();
      CubeIntRules[i] = new IntegrationRule(np*np*np);
      for (k = 0; k < np; k++)
         for (l = 0; l < np; l++)
            for (m = 0; m < np; m++)
            {
               CubeIntRules[i] -> IntPoint((k*np+l)*np+m).x =
                  SegmentIntRules[i] -> IntPoint(k).x;

               CubeIntRules[i] -> IntPoint((k*np+l)*np+m).y =
                  SegmentIntRules[i] -> IntPoint(l).x;

               CubeIntRules[i] -> IntPoint((k*np+l)*np+m).z =
                  SegmentIntRules[i] -> IntPoint(m).x;

               CubeIntRules[i] -> IntPoint((k*np+l)*np+m).weight =
                  SegmentIntRules[i] -> IntPoint(k).weight *
                  SegmentIntRules[i] -> IntPoint(l).weight *
                  SegmentIntRules[i] -> IntPoint(m).weight;
            }
   }

}
