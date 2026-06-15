// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

// Fixed Order Finite Element classes

#include "fe_fixed_order.hpp"
#include "../coefficient.hpp"

namespace mfem
{

using namespace std;

PointFiniteElement::PointFiniteElement()
   : NodalFiniteElement(0, Geometry::POINT, 1, 0)
{
   lex_ordering.SetSize(1);
   lex_ordering[0] = 0;
   Nodes.IntPoint(0).x = 0.0;
}

void PointFiniteElement::CalcShape(const IntegrationPoint &ip,
                                   Vector &shape) const
{
   shape(0) = 1.;
}

void PointFiniteElement::CalcDShape(const IntegrationPoint &ip,
                                    DenseMatrix &dshape) const
{
   // dshape is (1 x 0) - nothing to compute
}

Linear1DFiniteElement::Linear1DFiniteElement()
   : NodalFiniteElement(1, Geometry::SEGMENT, 2, 1)
{
   Nodes.IntPoint(0).x = 0.0;
   Nodes.IntPoint(1).x = 1.0;
}

void Linear1DFiniteElement::CalcShape(const IntegrationPoint &ip,
                                      Vector &shape) const
{
   shape(0) = 1. - ip.x;
   shape(1) = ip.x;
}

void Linear1DFiniteElement::CalcDShape(const IntegrationPoint &ip,
                                       DenseMatrix &dshape) const
{
   dshape(0,0) = -1.;
   dshape(1,0) =  1.;
}

void Linear1DFiniteElement::CalcHessian(const IntegrationPoint &ip,
                                        DenseMatrix &h) const
{
   h = 0.0;
}

Linear2DFiniteElement::Linear2DFiniteElement()
   : NodalFiniteElement(2, Geometry::TRIANGLE, 3, 1)
{
   Nodes.IntPoint(0).x = 0.0;
   Nodes.IntPoint(0).y = 0.0;
   Nodes.IntPoint(1).x = 1.0;
   Nodes.IntPoint(1).y = 0.0;
   Nodes.IntPoint(2).x = 0.0;
   Nodes.IntPoint(2).y = 1.0;
}

void Linear2DFiniteElement::CalcShape(const IntegrationPoint &ip,
                                      Vector &shape) const
{
   shape(0) = 1. - ip.x - ip.y;
   shape(1) = ip.x;
   shape(2) = ip.y;
}

void Linear2DFiniteElement::CalcDShape(const IntegrationPoint &ip,
                                       DenseMatrix &dshape) const
{
   dshape(0,0) = -1.; dshape(0,1) = -1.;
   dshape(1,0) =  1.; dshape(1,1) =  0.;
   dshape(2,0) =  0.; dshape(2,1) =  1.;
}

void Linear2DFiniteElement::CalcHessian(const IntegrationPoint &ip,
                                        DenseMatrix &h) const
{
   h = 0.0;
}

BiLinear2DFiniteElement::BiLinear2DFiniteElement()
   : NodalFiniteElement(2, Geometry::SQUARE, 4, 1, FunctionSpace::Qk)
{
   Nodes.IntPoint(0).x = 0.0;
   Nodes.IntPoint(0).y = 0.0;
   Nodes.IntPoint(1).x = 1.0;
   Nodes.IntPoint(1).y = 0.0;
   Nodes.IntPoint(2).x = 1.0;
   Nodes.IntPoint(2).y = 1.0;
   Nodes.IntPoint(3).x = 0.0;
   Nodes.IntPoint(3).y = 1.0;
}

void BiLinear2DFiniteElement::CalcShape(const IntegrationPoint &ip,
                                        Vector &shape) const
{
   shape(0) = (1. - ip.x) * (1. - ip.y) ;
   shape(1) = ip.x * (1. - ip.y) ;
   shape(2) = ip.x * ip.y ;
   shape(3) = (1. - ip.x) * ip.y ;
}

void BiLinear2DFiniteElement::CalcDShape(const IntegrationPoint &ip,
                                         DenseMatrix &dshape) const
{
   dshape(0,0) = -1. + ip.y; dshape(0,1) = -1. + ip.x ;
   dshape(1,0) =  1. - ip.y; dshape(1,1) = -ip.x ;
   dshape(2,0) =  ip.y ;     dshape(2,1) = ip.x ;
   dshape(3,0) = -ip.y ;     dshape(3,1) = 1. - ip.x ;
}

void BiLinear2DFiniteElement::CalcHessian(
   const IntegrationPoint &ip, DenseMatrix &h) const
{
   h(0,0) = 0.;   h(0,1) =  1.;   h(0,2) = 0.;
   h(1,0) = 0.;   h(1,1) = -1.;   h(1,2) = 0.;
   h(2,0) = 0.;   h(2,1) =  1.;   h(2,2) = 0.;
   h(3,0) = 0.;   h(3,1) = -1.;   h(3,2) = 0.;
}


GaussLinear2DFiniteElement::GaussLinear2DFiniteElement()
   : NodalFiniteElement(2, Geometry::TRIANGLE, 3, 1, FunctionSpace::Pk)
{
   Nodes.IntPoint(0).x = 1./6.;
   Nodes.IntPoint(0).y = 1./6.;
   Nodes.IntPoint(1).x = 2./3.;
   Nodes.IntPoint(1).y = 1./6.;
   Nodes.IntPoint(2).x = 1./6.;
   Nodes.IntPoint(2).y = 2./3.;
}

void GaussLinear2DFiniteElement::CalcShape(const IntegrationPoint &ip,
                                           Vector &shape) const
{
   const real_t x = ip.x, y = ip.y;

   shape(0) = 5./3. - 2. * (x + y);
   shape(1) = 2. * (x - 1./6.);
   shape(2) = 2. * (y - 1./6.);
}

void GaussLinear2DFiniteElement::CalcDShape(const IntegrationPoint &ip,
                                            DenseMatrix &dshape) const
{
   dshape(0,0) = -2.;  dshape(0,1) = -2.;
   dshape(1,0) =  2.;  dshape(1,1) =  0.;
   dshape(2,0) =  0.;  dshape(2,1) =  2.;
}

void GaussLinear2DFiniteElement::ProjectDelta(int vertex, Vector &dofs) const
{
   dofs(vertex)       = 2./3.;
   dofs((vertex+1)%3) = 1./6.;
   dofs((vertex+2)%3) = 1./6.;
}


// 0.5-0.5/sqrt(3) and 0.5+0.5/sqrt(3)
const real_t GaussBiLinear2DFiniteElement::p[] =
{ 0.2113248654051871177454256, 0.7886751345948128822545744 };

GaussBiLinear2DFiniteElement::GaussBiLinear2DFiniteElement()
   : NodalFiniteElement(2, Geometry::SQUARE, 4, 1, FunctionSpace::Qk)
{
   Nodes.IntPoint(0).x = p[0];
   Nodes.IntPoint(0).y = p[0];
   Nodes.IntPoint(1).x = p[1];
   Nodes.IntPoint(1).y = p[0];
   Nodes.IntPoint(2).x = p[1];
   Nodes.IntPoint(2).y = p[1];
   Nodes.IntPoint(3).x = p[0];
   Nodes.IntPoint(3).y = p[1];
}

void GaussBiLinear2DFiniteElement::CalcShape(const IntegrationPoint &ip,
                                             Vector &shape) const
{
   const real_t x = ip.x, y = ip.y;

   shape(0) = 3. * (p[1] - x) * (p[1] - y);
   shape(1) = 3. * (x - p[0]) * (p[1] - y);
   shape(2) = 3. * (x - p[0]) * (y - p[0]);
   shape(3) = 3. * (p[1] - x) * (y - p[0]);
}

void GaussBiLinear2DFiniteElement::CalcDShape(const IntegrationPoint &ip,
                                              DenseMatrix &dshape) const
{
   const real_t x = ip.x, y = ip.y;

   dshape(0,0) = 3. * (y - p[1]);  dshape(0,1) = 3. * (x - p[1]);
   dshape(1,0) = 3. * (p[1] - y);  dshape(1,1) = 3. * (p[0] - x);
   dshape(2,0) = 3. * (y - p[0]);  dshape(2,1) = 3. * (x - p[0]);
   dshape(3,0) = 3. * (p[0] - y);  dshape(3,1) = 3. * (p[1] - x);
}

void GaussBiLinear2DFiniteElement::ProjectDelta(int vertex, Vector &dofs) const
{
#if 1
   dofs(vertex)       = p[1]*p[1];
   dofs((vertex+1)%4) = p[0]*p[1];
   dofs((vertex+2)%4) = p[0]*p[0];
   dofs((vertex+3)%4) = p[0]*p[1];
#else
   dofs = 1.0;
#endif
}


P1OnQuadFiniteElement::P1OnQuadFiniteElement()
   : NodalFiniteElement(2, Geometry::SQUARE, 3, 1, FunctionSpace::Qk)
{
   Nodes.IntPoint(0).x = 0.0;
   Nodes.IntPoint(0).y = 0.0;
   Nodes.IntPoint(1).x = 1.0;
   Nodes.IntPoint(1).y = 0.0;
   Nodes.IntPoint(2).x = 0.0;
   Nodes.IntPoint(2).y = 1.0;
}

void P1OnQuadFiniteElement::CalcShape(const IntegrationPoint &ip,
                                      Vector &shape) const
{
   shape(0) = 1. - ip.x - ip.y;
   shape(1) = ip.x;
   shape(2) = ip.y;
}

void P1OnQuadFiniteElement::CalcDShape(const IntegrationPoint &ip,
                                       DenseMatrix &dshape) const
{
   dshape(0,0) = -1.; dshape(0,1) = -1.;
   dshape(1,0) =  1.; dshape(1,1) =  0.;
   dshape(2,0) =  0.; dshape(2,1) =  1.;
}


Quad1DFiniteElement::Quad1DFiniteElement()
   : NodalFiniteElement(1, Geometry::SEGMENT, 3, 2)
{
   Nodes.IntPoint(0).x = 0.0;
   Nodes.IntPoint(1).x = 1.0;
   Nodes.IntPoint(2).x = 0.5;
}

void Quad1DFiniteElement::CalcShape(const IntegrationPoint &ip,
                                    Vector &shape) const
{
   real_t x = ip.x;
   real_t l1 = 1.0 - x, l2 = x, l3 = 2. * x - 1.;

   shape(0) = l1 * (-l3);
   shape(1) = l2 * l3;
   shape(2) = 4. * l1 * l2;
}

void Quad1DFiniteElement::CalcDShape(const IntegrationPoint &ip,
                                     DenseMatrix &dshape) const
{
   real_t x = ip.x;

   dshape(0,0) = 4. * x - 3.;
   dshape(1,0) = 4. * x - 1.;
   dshape(2,0) = 4. - 8. * x;
}


Quad2DFiniteElement::Quad2DFiniteElement()
   : NodalFiniteElement(2, Geometry::TRIANGLE, 6, 2)
{
   Nodes.IntPoint(0).x = 0.0;
   Nodes.IntPoint(0).y = 0.0;
   Nodes.IntPoint(1).x = 1.0;
   Nodes.IntPoint(1).y = 0.0;
   Nodes.IntPoint(2).x = 0.0;
   Nodes.IntPoint(2).y = 1.0;
   Nodes.IntPoint(3).x = 0.5;
   Nodes.IntPoint(3).y = 0.0;
   Nodes.IntPoint(4).x = 0.5;
   Nodes.IntPoint(4).y = 0.5;
   Nodes.IntPoint(5).x = 0.0;
   Nodes.IntPoint(5).y = 0.5;
}

void Quad2DFiniteElement::CalcShape(const IntegrationPoint &ip,
                                    Vector &shape) const
{
   real_t x = ip.x, y = ip.y;
   real_t l1 = 1.-x-y, l2 = x, l3 = y;

   shape(0) = l1 * (2. * l1 - 1.);
   shape(1) = l2 * (2. * l2 - 1.);
   shape(2) = l3 * (2. * l3 - 1.);
   shape(3) = 4. * l1 * l2;
   shape(4) = 4. * l2 * l3;
   shape(5) = 4. * l3 * l1;
}

void Quad2DFiniteElement::CalcDShape(const IntegrationPoint &ip,
                                     DenseMatrix &dshape) const
{
   real_t x = ip.x, y = ip.y;

   dshape(0,0) =
      dshape(0,1) = 4. * (x + y) - 3.;

   dshape(1,0) = 4. * x - 1.;
   dshape(1,1) = 0.;

   dshape(2,0) = 0.;
   dshape(2,1) = 4. * y - 1.;

   dshape(3,0) = -4. * (2. * x + y - 1.);
   dshape(3,1) = -4. * x;

   dshape(4,0) = 4. * y;
   dshape(4,1) = 4. * x;

   dshape(5,0) = -4. * y;
   dshape(5,1) = -4. * (x + 2. * y - 1.);
}

void Quad2DFiniteElement::CalcHessian (const IntegrationPoint &ip,
                                       DenseMatrix &h) const
{
   h(0,0) = 4.;
   h(0,1) = 4.;
   h(0,2) = 4.;

   h(1,0) = 4.;
   h(1,1) = 0.;
   h(1,2) = 0.;

   h(2,0) = 0.;
   h(2,1) = 0.;
   h(2,2) = 4.;

   h(3,0) = -8.;
   h(3,1) = -4.;
   h(3,2) =  0.;

   h(4,0) = 0.;
   h(4,1) = 4.;
   h(4,2) = 0.;

   h(5,0) =  0.;
   h(5,1) = -4.;
   h(5,2) = -8.;
}

void Quad2DFiniteElement::ProjectDelta(int vertex, Vector &dofs) const
{
#if 0
   dofs = 1.;
#else
   dofs = 0.;
   dofs(vertex) = 1.;
   switch (vertex)
   {
      case 0: dofs(3) = 0.25; dofs(5) = 0.25; break;
      case 1: dofs(3) = 0.25; dofs(4) = 0.25; break;
      case 2: dofs(4) = 0.25; dofs(5) = 0.25; break;
   }
#endif
}


const real_t GaussQuad2DFiniteElement::p[] =
{ 0.0915762135097707434595714634022015, 0.445948490915964886318329253883051 };

GaussQuad2DFiniteElement::GaussQuad2DFiniteElement()
   : NodalFiniteElement(2, Geometry::TRIANGLE, 6, 2), A(6), D(6,2), pol(6)
{
   Nodes.IntPoint(0).x = p[0];
   Nodes.IntPoint(0).y = p[0];
   Nodes.IntPoint(1).x = 1. - 2. * p[0];
   Nodes.IntPoint(1).y = p[0];
   Nodes.IntPoint(2).x = p[0];
   Nodes.IntPoint(2).y = 1. - 2. * p[0];
   Nodes.IntPoint(3).x = p[1];
   Nodes.IntPoint(3).y = p[1];
   Nodes.IntPoint(4).x = 1. - 2. * p[1];
   Nodes.IntPoint(4).y = p[1];
   Nodes.IntPoint(5).x = p[1];
   Nodes.IntPoint(5).y = 1. - 2. * p[1];

   for (int i = 0; i < 6; i++)
   {
      const real_t x = Nodes.IntPoint(i).x, y = Nodes.IntPoint(i).y;
      A(0,i) = 1.;
      A(1,i) = x;
      A(2,i) = y;
      A(3,i) = x * x;
      A(4,i) = x * y;
      A(5,i) = y * y;
   }

   A.Invert();
}

void GaussQuad2DFiniteElement::CalcShape(const IntegrationPoint &ip,
                                         Vector &shape) const
{
   const real_t x = ip.x, y = ip.y;
   pol(0) = 1.;
   pol(1) = x;
   pol(2) = y;
   pol(3) = x * x;
   pol(4) = x * y;
   pol(5) = y * y;

   A.Mult(pol, shape);
}

void GaussQuad2DFiniteElement::CalcDShape(const IntegrationPoint &ip,
                                          DenseMatrix &dshape) const
{
   const real_t x = ip.x, y = ip.y;
   D(0,0) = 0.;      D(0,1) = 0.;
   D(1,0) = 1.;      D(1,1) = 0.;
   D(2,0) = 0.;      D(2,1) = 1.;
   D(3,0) = 2. *  x; D(3,1) = 0.;
   D(4,0) = y;       D(4,1) = x;
   D(5,0) = 0.;      D(5,1) = 2. * y;

   Mult(A, D, dshape);
}


BiQuad2DFiniteElement::BiQuad2DFiniteElement()
   : NodalFiniteElement(2, Geometry::SQUARE, 9, 2, FunctionSpace::Qk)
{
   Nodes.IntPoint(0).x = 0.0;
   Nodes.IntPoint(0).y = 0.0;
   Nodes.IntPoint(1).x = 1.0;
   Nodes.IntPoint(1).y = 0.0;
   Nodes.IntPoint(2).x = 1.0;
   Nodes.IntPoint(2).y = 1.0;
   Nodes.IntPoint(3).x = 0.0;
   Nodes.IntPoint(3).y = 1.0;
   Nodes.IntPoint(4).x = 0.5;
   Nodes.IntPoint(4).y = 0.0;
   Nodes.IntPoint(5).x = 1.0;
   Nodes.IntPoint(5).y = 0.5;
   Nodes.IntPoint(6).x = 0.5;
   Nodes.IntPoint(6).y = 1.0;
   Nodes.IntPoint(7).x = 0.0;
   Nodes.IntPoint(7).y = 0.5;
   Nodes.IntPoint(8).x = 0.5;
   Nodes.IntPoint(8).y = 0.5;
}

void BiQuad2DFiniteElement::CalcShape(const IntegrationPoint &ip,
                                      Vector &shape) const
{
   real_t x = ip.x, y = ip.y;
   real_t l1x, l2x, l3x, l1y, l2y, l3y;

   l1x = (x - 1.) * (2. * x - 1);
   l2x = 4. * x * (1. - x);
   l3x = x * (2. * x - 1.);
   l1y = (y - 1.) * (2. * y - 1);
   l2y = 4. * y * (1. - y);
   l3y = y * (2. * y - 1.);

   shape(0) = l1x * l1y;
   shape(4) = l2x * l1y;
   shape(1) = l3x * l1y;
   shape(7) = l1x * l2y;
   shape(8) = l2x * l2y;
   shape(5) = l3x * l2y;
   shape(3) = l1x * l3y;
   shape(6) = l2x * l3y;
   shape(2) = l3x * l3y;
}

void BiQuad2DFiniteElement::CalcDShape(const IntegrationPoint &ip,
                                       DenseMatrix &dshape) const
{
   real_t x = ip.x, y = ip.y;
   real_t l1x, l2x, l3x, l1y, l2y, l3y;
   real_t d1x, d2x, d3x, d1y, d2y, d3y;

   l1x = (x - 1.) * (2. * x - 1);
   l2x = 4. * x * (1. - x);
   l3x = x * (2. * x - 1.);
   l1y = (y - 1.) * (2. * y - 1);
   l2y = 4. * y * (1. - y);
   l3y = y * (2. * y - 1.);

   d1x = 4. * x - 3.;
   d2x = 4. - 8. * x;
   d3x = 4. * x - 1.;
   d1y = 4. * y - 3.;
   d2y = 4. - 8. * y;
   d3y = 4. * y - 1.;

   dshape(0,0) = d1x * l1y;
   dshape(0,1) = l1x * d1y;

   dshape(4,0) = d2x * l1y;
   dshape(4,1) = l2x * d1y;

   dshape(1,0) = d3x * l1y;
   dshape(1,1) = l3x * d1y;

   dshape(7,0) = d1x * l2y;
   dshape(7,1) = l1x * d2y;

   dshape(8,0) = d2x * l2y;
   dshape(8,1) = l2x * d2y;

   dshape(5,0) = d3x * l2y;
   dshape(5,1) = l3x * d2y;

   dshape(3,0) = d1x * l3y;
   dshape(3,1) = l1x * d3y;

   dshape(6,0) = d2x * l3y;
   dshape(6,1) = l2x * d3y;

   dshape(2,0) = d3x * l3y;
   dshape(2,1) = l3x * d3y;
}

void BiQuad2DFiniteElement::ProjectDelta(int vertex, Vector &dofs) const
{
#if 0
   dofs = 1.;
#else
   dofs = 0.;
   dofs(vertex) = 1.;
   switch (vertex)
   {
      case 0: dofs(4) = 0.25; dofs(7) = 0.25; break;
      case 1: dofs(4) = 0.25; dofs(5) = 0.25; break;
      case 2: dofs(5) = 0.25; dofs(6) = 0.25; break;
      case 3: dofs(6) = 0.25; dofs(7) = 0.25; break;
   }
   dofs(8) = 1./16.;
#endif
}


GaussBiQuad2DFiniteElement::GaussBiQuad2DFiniteElement()
   : NodalFiniteElement(2, Geometry::SQUARE, 9, 2, FunctionSpace::Qk)
{
   const real_t p1 = 0.5*(1.-sqrt(3./5.));

   Nodes.IntPoint(0).x = p1;
   Nodes.IntPoint(0).y = p1;
   Nodes.IntPoint(4).x = 0.5;
   Nodes.IntPoint(4).y = p1;
   Nodes.IntPoint(1).x = 1.-p1;
   Nodes.IntPoint(1).y = p1;
   Nodes.IntPoint(7).x = p1;
   Nodes.IntPoint(7).y = 0.5;
   Nodes.IntPoint(8).x = 0.5;
   Nodes.IntPoint(8).y = 0.5;
   Nodes.IntPoint(5).x = 1.-p1;
   Nodes.IntPoint(5).y = 0.5;
   Nodes.IntPoint(3).x = p1;
   Nodes.IntPoint(3).y = 1.-p1;
   Nodes.IntPoint(6).x = 0.5;
   Nodes.IntPoint(6).y = 1.-p1;
   Nodes.IntPoint(2).x = 1.-p1;
   Nodes.IntPoint(2).y = 1.-p1;
}

void GaussBiQuad2DFiniteElement::CalcShape(const IntegrationPoint &ip,
                                           Vector &shape) const
{
   const real_t a = sqrt(5./3.);
   const real_t p1 = 0.5*(1.-sqrt(3./5.));

   real_t x = a*(ip.x-p1), y = a*(ip.y-p1);
   real_t l1x, l2x, l3x, l1y, l2y, l3y;

   l1x = (x - 1.) * (2. * x - 1);
   l2x = 4. * x * (1. - x);
   l3x = x * (2. * x - 1.);
   l1y = (y - 1.) * (2. * y - 1);
   l2y = 4. * y * (1. - y);
   l3y = y * (2. * y - 1.);

   shape(0) = l1x * l1y;
   shape(4) = l2x * l1y;
   shape(1) = l3x * l1y;
   shape(7) = l1x * l2y;
   shape(8) = l2x * l2y;
   shape(5) = l3x * l2y;
   shape(3) = l1x * l3y;
   shape(6) = l2x * l3y;
   shape(2) = l3x * l3y;
}

void GaussBiQuad2DFiniteElement::CalcDShape(const IntegrationPoint &ip,
                                            DenseMatrix &dshape) const
{
   const real_t a = sqrt(5./3.);
   const real_t p1 = 0.5*(1.-sqrt(3./5.));

   real_t x = a*(ip.x-p1), y = a*(ip.y-p1);
   real_t l1x, l2x, l3x, l1y, l2y, l3y;
   real_t d1x, d2x, d3x, d1y, d2y, d3y;

   l1x = (x - 1.) * (2. * x - 1);
   l2x = 4. * x * (1. - x);
   l3x = x * (2. * x - 1.);
   l1y = (y - 1.) * (2. * y - 1);
   l2y = 4. * y * (1. - y);
   l3y = y * (2. * y - 1.);

   d1x = a * (4. * x - 3.);
   d2x = a * (4. - 8. * x);
   d3x = a * (4. * x - 1.);
   d1y = a * (4. * y - 3.);
   d2y = a * (4. - 8. * y);
   d3y = a * (4. * y - 1.);

   dshape(0,0) = d1x * l1y;
   dshape(0,1) = l1x * d1y;

   dshape(4,0) = d2x * l1y;
   dshape(4,1) = l2x * d1y;

   dshape(1,0) = d3x * l1y;
   dshape(1,1) = l3x * d1y;

   dshape(7,0) = d1x * l2y;
   dshape(7,1) = l1x * d2y;

   dshape(8,0) = d2x * l2y;
   dshape(8,1) = l2x * d2y;

   dshape(5,0) = d3x * l2y;
   dshape(5,1) = l3x * d2y;

   dshape(3,0) = d1x * l3y;
   dshape(3,1) = l1x * d3y;

   dshape(6,0) = d2x * l3y;
   dshape(6,1) = l2x * d3y;

   dshape(2,0) = d3x * l3y;
   dshape(2,1) = l3x * d3y;
}

BiCubic2DFiniteElement::BiCubic2DFiniteElement()
   : NodalFiniteElement (2, Geometry::SQUARE, 16, 3, FunctionSpace::Qk)
{
   Nodes.IntPoint(0).x = 0.;
   Nodes.IntPoint(0).y = 0.;
   Nodes.IntPoint(1).x = 1.;
   Nodes.IntPoint(1).y = 0.;
   Nodes.IntPoint(2).x = 1.;
   Nodes.IntPoint(2).y = 1.;
   Nodes.IntPoint(3).x = 0.;
   Nodes.IntPoint(3).y = 1.;
   Nodes.IntPoint(4).x = 1./3.;
   Nodes.IntPoint(4).y = 0.;
   Nodes.IntPoint(5).x = 2./3.;
   Nodes.IntPoint(5).y = 0.;
   Nodes.IntPoint(6).x = 1.;
   Nodes.IntPoint(6).y = 1./3.;
   Nodes.IntPoint(7).x = 1.;
   Nodes.IntPoint(7).y = 2./3.;
   Nodes.IntPoint(8).x = 2./3.;
   Nodes.IntPoint(8).y = 1.;
   Nodes.IntPoint(9).x = 1./3.;
   Nodes.IntPoint(9).y = 1.;
   Nodes.IntPoint(10).x = 0.;
   Nodes.IntPoint(10).y = 2./3.;
   Nodes.IntPoint(11).x = 0.;
   Nodes.IntPoint(11).y = 1./3.;
   Nodes.IntPoint(12).x = 1./3.;
   Nodes.IntPoint(12).y = 1./3.;
   Nodes.IntPoint(13).x = 2./3.;
   Nodes.IntPoint(13).y = 1./3.;
   Nodes.IntPoint(14).x = 1./3.;
   Nodes.IntPoint(14).y = 2./3.;
   Nodes.IntPoint(15).x = 2./3.;
   Nodes.IntPoint(15).y = 2./3.;
}

void BiCubic2DFiniteElement::CalcShape(
   const IntegrationPoint &ip, Vector &shape) const
{
   real_t x = ip.x, y = ip.y;

   real_t w1x, w2x, w3x, w1y, w2y, w3y;
   real_t l0x, l1x, l2x, l3x, l0y, l1y, l2y, l3y;

   w1x = x - 1./3.; w2x = x - 2./3.; w3x = x - 1.;
   w1y = y - 1./3.; w2y = y - 2./3.; w3y = y - 1.;

   l0x = (- 4.5) * w1x * w2x * w3x;
   l1x = ( 13.5) *   x * w2x * w3x;
   l2x = (-13.5) *   x * w1x * w3x;
   l3x = (  4.5) *   x * w1x * w2x;

   l0y = (- 4.5) * w1y * w2y * w3y;
   l1y = ( 13.5) *   y * w2y * w3y;
   l2y = (-13.5) *   y * w1y * w3y;
   l3y = (  4.5) *   y * w1y * w2y;

   shape(0)  = l0x * l0y;
   shape(1)  = l3x * l0y;
   shape(2)  = l3x * l3y;
   shape(3)  = l0x * l3y;
   shape(4)  = l1x * l0y;
   shape(5)  = l2x * l0y;
   shape(6)  = l3x * l1y;
   shape(7)  = l3x * l2y;
   shape(8)  = l2x * l3y;
   shape(9)  = l1x * l3y;
   shape(10) = l0x * l2y;
   shape(11) = l0x * l1y;
   shape(12) = l1x * l1y;
   shape(13) = l2x * l1y;
   shape(14) = l1x * l2y;
   shape(15) = l2x * l2y;
}

void BiCubic2DFiniteElement::CalcDShape(
   const IntegrationPoint &ip, DenseMatrix &dshape) const
{
   real_t x = ip.x, y = ip.y;

   real_t w1x, w2x, w3x, w1y, w2y, w3y;
   real_t l0x, l1x, l2x, l3x, l0y, l1y, l2y, l3y;
   real_t d0x, d1x, d2x, d3x, d0y, d1y, d2y, d3y;

   w1x = x - 1./3.; w2x = x - 2./3.; w3x = x - 1.;
   w1y = y - 1./3.; w2y = y - 2./3.; w3y = y - 1.;

   l0x = (- 4.5) * w1x * w2x * w3x;
   l1x = ( 13.5) *   x * w2x * w3x;
   l2x = (-13.5) *   x * w1x * w3x;
   l3x = (  4.5) *   x * w1x * w2x;

   l0y = (- 4.5) * w1y * w2y * w3y;
   l1y = ( 13.5) *   y * w2y * w3y;
   l2y = (-13.5) *   y * w1y * w3y;
   l3y = (  4.5) *   y * w1y * w2y;

   d0x = -5.5 + ( 18. - 13.5 * x) * x;
   d1x =  9.  + (-45. + 40.5 * x) * x;
   d2x = -4.5 + ( 36. - 40.5 * x) * x;
   d3x =  1.  + (- 9. + 13.5 * x) * x;

   d0y = -5.5 + ( 18. - 13.5 * y) * y;
   d1y =  9.  + (-45. + 40.5 * y) * y;
   d2y = -4.5 + ( 36. - 40.5 * y) * y;
   d3y =  1.  + (- 9. + 13.5 * y) * y;

   dshape( 0,0) = d0x * l0y;   dshape( 0,1) = l0x * d0y;
   dshape( 1,0) = d3x * l0y;   dshape( 1,1) = l3x * d0y;
   dshape( 2,0) = d3x * l3y;   dshape( 2,1) = l3x * d3y;
   dshape( 3,0) = d0x * l3y;   dshape( 3,1) = l0x * d3y;
   dshape( 4,0) = d1x * l0y;   dshape( 4,1) = l1x * d0y;
   dshape( 5,0) = d2x * l0y;   dshape( 5,1) = l2x * d0y;
   dshape( 6,0) = d3x * l1y;   dshape( 6,1) = l3x * d1y;
   dshape( 7,0) = d3x * l2y;   dshape( 7,1) = l3x * d2y;
   dshape( 8,0) = d2x * l3y;   dshape( 8,1) = l2x * d3y;
   dshape( 9,0) = d1x * l3y;   dshape( 9,1) = l1x * d3y;
   dshape(10,0) = d0x * l2y;   dshape(10,1) = l0x * d2y;
   dshape(11,0) = d0x * l1y;   dshape(11,1) = l0x * d1y;
   dshape(12,0) = d1x * l1y;   dshape(12,1) = l1x * d1y;
   dshape(13,0) = d2x * l1y;   dshape(13,1) = l2x * d1y;
   dshape(14,0) = d1x * l2y;   dshape(14,1) = l1x * d2y;
   dshape(15,0) = d2x * l2y;   dshape(15,1) = l2x * d2y;
}

void BiCubic2DFiniteElement::CalcHessian(
   const IntegrationPoint &ip, DenseMatrix &h) const
{
   real_t x = ip.x, y = ip.y;

   real_t w1x, w2x, w3x, w1y, w2y, w3y;
   real_t l0x, l1x, l2x, l3x, l0y, l1y, l2y, l3y;
   real_t d0x, d1x, d2x, d3x, d0y, d1y, d2y, d3y;
   real_t h0x, h1x, h2x, h3x, h0y, h1y, h2y, h3y;

   w1x = x - 1./3.; w2x = x - 2./3.; w3x = x - 1.;
   w1y = y - 1./3.; w2y = y - 2./3.; w3y = y - 1.;

   l0x = (- 4.5) * w1x * w2x * w3x;
   l1x = ( 13.5) *   x * w2x * w3x;
   l2x = (-13.5) *   x * w1x * w3x;
   l3x = (  4.5) *   x * w1x * w2x;

   l0y = (- 4.5) * w1y * w2y * w3y;
   l1y = ( 13.5) *   y * w2y * w3y;
   l2y = (-13.5) *   y * w1y * w3y;
   l3y = (  4.5) *   y * w1y * w2y;

   d0x = -5.5 + ( 18. - 13.5 * x) * x;
   d1x =  9.  + (-45. + 40.5 * x) * x;
   d2x = -4.5 + ( 36. - 40.5 * x) * x;
   d3x =  1.  + (- 9. + 13.5 * x) * x;

   d0y = -5.5 + ( 18. - 13.5 * y) * y;
   d1y =  9.  + (-45. + 40.5 * y) * y;
   d2y = -4.5 + ( 36. - 40.5 * y) * y;
   d3y =  1.  + (- 9. + 13.5 * y) * y;

   h0x = -27. * x + 18.;
   h1x =  81. * x - 45.;
   h2x = -81. * x + 36.;
   h3x =  27. * x -  9.;

   h0y = -27. * y + 18.;
   h1y =  81. * y - 45.;
   h2y = -81. * y + 36.;
   h3y =  27. * y -  9.;

   h( 0,0) = h0x * l0y;   h( 0,1) = d0x * d0y;   h( 0,2) = l0x * h0y;
   h( 1,0) = h3x * l0y;   h( 1,1) = d3x * d0y;   h( 1,2) = l3x * h0y;
   h( 2,0) = h3x * l3y;   h( 2,1) = d3x * d3y;   h( 2,2) = l3x * h3y;
   h( 3,0) = h0x * l3y;   h( 3,1) = d0x * d3y;   h( 3,2) = l0x * h3y;
   h( 4,0) = h1x * l0y;   h( 4,1) = d1x * d0y;   h( 4,2) = l1x * h0y;
   h( 5,0) = h2x * l0y;   h( 5,1) = d2x * d0y;   h( 5,2) = l2x * h0y;
   h( 6,0) = h3x * l1y;   h( 6,1) = d3x * d1y;   h( 6,2) = l3x * h1y;
   h( 7,0) = h3x * l2y;   h( 7,1) = d3x * d2y;   h( 7,2) = l3x * h2y;
   h( 8,0) = h2x * l3y;   h( 8,1) = d2x * d3y;   h( 8,2) = l2x * h3y;
   h( 9,0) = h1x * l3y;   h( 9,1) = d1x * d3y;   h( 9,2) = l1x * h3y;
   h(10,0) = h0x * l2y;   h(10,1) = d0x * d2y;   h(10,2) = l0x * h2y;
   h(11,0) = h0x * l1y;   h(11,1) = d0x * d1y;   h(11,2) = l0x * h1y;
   h(12,0) = h1x * l1y;   h(12,1) = d1x * d1y;   h(12,2) = l1x * h1y;
   h(13,0) = h2x * l1y;   h(13,1) = d2x * d1y;   h(13,2) = l2x * h1y;
   h(14,0) = h1x * l2y;   h(14,1) = d1x * d2y;   h(14,2) = l1x * h2y;
   h(15,0) = h2x * l2y;   h(15,1) = d2x * d2y;   h(15,2) = l2x * h2y;
}


Cubic1DFiniteElement::Cubic1DFiniteElement()
   : NodalFiniteElement(1, Geometry::SEGMENT, 4, 3)
{
   Nodes.IntPoint(0).x = 0.0;
   Nodes.IntPoint(1).x = 1.0;
   Nodes.IntPoint(2).x = 0.33333333333333333333;
   Nodes.IntPoint(3).x = 0.66666666666666666667;
}

void Cubic1DFiniteElement::CalcShape(const IntegrationPoint &ip,
                                     Vector &shape) const
{
   real_t x = ip.x;
   real_t l1 = x,
          l2 = (1.0-x),
          l3 = (0.33333333333333333333-x),
          l4 = (0.66666666666666666667-x);

   shape(0) =   4.5 * l2 * l3 * l4;
   shape(1) =   4.5 * l1 * l3 * l4;
   shape(2) =  13.5 * l1 * l2 * l4;
   shape(3) = -13.5 * l1 * l2 * l3;
}

void Cubic1DFiniteElement::CalcDShape(const IntegrationPoint &ip,
                                      DenseMatrix &dshape) const
{
   real_t x = ip.x;

   dshape(0,0) = -5.5 + x * (18. - 13.5 * x);
   dshape(1,0) = 1. - x * (9. - 13.5 * x);
   dshape(2,0) = 9. - x * (45. - 40.5 * x);
   dshape(3,0) = -4.5 + x * (36. - 40.5 * x);
}


Cubic2DFiniteElement::Cubic2DFiniteElement()
   : NodalFiniteElement(2, Geometry::TRIANGLE, 10, 3)
{
   Nodes.IntPoint(0).x = 0.0;
   Nodes.IntPoint(0).y = 0.0;
   Nodes.IntPoint(1).x = 1.0;
   Nodes.IntPoint(1).y = 0.0;
   Nodes.IntPoint(2).x = 0.0;
   Nodes.IntPoint(2).y = 1.0;
   Nodes.IntPoint(3).x = 0.33333333333333333333;
   Nodes.IntPoint(3).y = 0.0;
   Nodes.IntPoint(4).x = 0.66666666666666666667;
   Nodes.IntPoint(4).y = 0.0;
   Nodes.IntPoint(5).x = 0.66666666666666666667;
   Nodes.IntPoint(5).y = 0.33333333333333333333;
   Nodes.IntPoint(6).x = 0.33333333333333333333;
   Nodes.IntPoint(6).y = 0.66666666666666666667;
   Nodes.IntPoint(7).x = 0.0;
   Nodes.IntPoint(7).y = 0.66666666666666666667;
   Nodes.IntPoint(8).x = 0.0;
   Nodes.IntPoint(8).y = 0.33333333333333333333;
   Nodes.IntPoint(9).x = 0.33333333333333333333;
   Nodes.IntPoint(9).y = 0.33333333333333333333;
}

void Cubic2DFiniteElement::CalcShape(const IntegrationPoint &ip,
                                     Vector &shape) const
{
   real_t x = ip.x, y = ip.y;
   real_t l1 = (-1. + x + y),
          lx = (-1. + 3.*x),
          ly = (-1. + 3.*y);

   shape(0) = -0.5*l1*(3.*l1 + 1.)*(3.*l1 + 2.);
   shape(1) =  0.5*x*(lx - 1.)*lx;
   shape(2) =  0.5*y*(-1. + ly)*ly;
   shape(3) =  4.5*x*l1*(3.*l1 + 1.);
   shape(4) = -4.5*x*lx*l1;
   shape(5) =  4.5*x*lx*y;
   shape(6) =  4.5*x*y*ly;
   shape(7) = -4.5*y*l1*ly;
   shape(8) =  4.5*y*l1*(1. + 3.*l1);
   shape(9) = -27.*x*y*l1;
}

void Cubic2DFiniteElement::CalcDShape(const IntegrationPoint &ip,
                                      DenseMatrix &dshape) const
{
   real_t x = ip.x, y = ip.y;

   dshape(0,0) =  0.5*(-11. + 36.*y - 9.*(x*(-4. + 3.*x) + 6.*x*y + 3.*y*y));
   dshape(1,0) =  1. + 4.5*x*(-2. + 3.*x);
   dshape(2,0) =  0.;
   dshape(3,0) =  4.5*(2. + 9.*x*x - 5.*y + 3.*y*y + 2.*x*(-5. + 6.*y));
   dshape(4,0) = -4.5*(1. - 1.*y + x*(-8. + 9.*x + 6.*y));
   dshape(5,0) =  4.5*(-1. + 6.*x)*y;
   dshape(6,0) =  4.5*y*(-1. + 3.*y);
   dshape(7,0) =  4.5*(1. - 3.*y)*y;
   dshape(8,0) =  4.5*y*(-5. + 6.*x + 6.*y);
   dshape(9,0) =  -27.*y*(-1. + 2.*x + y);

   dshape(0,1) =  0.5*(-11. + 36.*y - 9.*(x*(-4. + 3.*x) + 6.*x*y + 3.*y*y));
   dshape(1,1) =  0.;
   dshape(2,1) =  1. + 4.5*y*(-2. + 3.*y);
   dshape(3,1) =  4.5*x*(-5. + 6.*x + 6.*y);
   dshape(4,1) =  4.5*(1. - 3.*x)*x;
   dshape(5,1) =  4.5*x*(-1. + 3.*x);
   dshape(6,1) =  4.5*x*(-1. + 6.*y);
   dshape(7,1) = -4.5*(1. + x*(-1. + 6.*y) + y*(-8. + 9.*y));
   dshape(8,1) =  4.5*(2. + 3.*x*x + y*(-10. + 9.*y) + x*(-5. + 12.*y));
   dshape(9,1) = -27.*x*(-1. + x + 2.*y);
}

void Cubic2DFiniteElement::CalcHessian (const IntegrationPoint &ip,
                                        DenseMatrix &h) const
{
   real_t x = ip.x, y = ip.y;

   h(0,0) = 18.-27.*(x+y);
   h(0,1) = 18.-27.*(x+y);
   h(0,2) = 18.-27.*(x+y);

   h(1,0) = -9.+27.*x;
   h(1,1) = 0.;
   h(1,2) = 0.;

   h(2,0) = 0.;
   h(2,1) = 0.;
   h(2,2) = -9.+27.*y;

   h(3,0) = -45.+81.*x+54.*y;
   h(3,1) = -22.5+54.*x+27.*y;
   h(3,2) = 27.*x;

   h(4,0) = 36.-81.*x-27.*y;
   h(4,1) = 4.5-27.*x;
   h(4,2) = 0.;

   h(5,0) = 27.*y;
   h(5,1) = -4.5+27.*x;
   h(5,2) = 0.;

   h(6,0) = 0.;
   h(6,1) = -4.5+27.*y;
   h(6,2) = 27.*x;

   h(7,0) = 0.;
   h(7,1) = 4.5-27.*y;
   h(7,2) = 36.-27.*x-81.*y;

   h(8,0) = 27.*y;
   h(8,1) = -22.5+27.*x+54.*y;
   h(8,2) = -45.+54.*x+81.*y;

   h(9,0) = -54.*y;
   h(9,1) = 27.-54.*(x+y);
   h(9,2) = -54.*x;
}


Cubic3DFiniteElement::Cubic3DFiniteElement()
   : NodalFiniteElement(3, Geometry::TETRAHEDRON, 20, 3)
{
   Nodes.IntPoint(0).x = 0;
   Nodes.IntPoint(0).y = 0;
   Nodes.IntPoint(0).z = 0;
   Nodes.IntPoint(1).x = 1.;
   Nodes.IntPoint(1).y = 0;
   Nodes.IntPoint(1).z = 0;
   Nodes.IntPoint(2).x = 0;
   Nodes.IntPoint(2).y = 1.;
   Nodes.IntPoint(2).z = 0;
   Nodes.IntPoint(3).x = 0;
   Nodes.IntPoint(3).y = 0;
   Nodes.IntPoint(3).z = 1.;
   Nodes.IntPoint(4).x = 0.3333333333333333333333333333;
   Nodes.IntPoint(4).y = 0;
   Nodes.IntPoint(4).z = 0;
   Nodes.IntPoint(5).x = 0.6666666666666666666666666667;
   Nodes.IntPoint(5).y = 0;
   Nodes.IntPoint(5).z = 0;
   Nodes.IntPoint(6).x = 0;
   Nodes.IntPoint(6).y = 0.3333333333333333333333333333;
   Nodes.IntPoint(6).z = 0;
   Nodes.IntPoint(7).x = 0;
   Nodes.IntPoint(7).y = 0.6666666666666666666666666667;
   Nodes.IntPoint(7).z = 0;
   Nodes.IntPoint(8).x = 0;
   Nodes.IntPoint(8).y = 0;
   Nodes.IntPoint(8).z = 0.3333333333333333333333333333;
   Nodes.IntPoint(9).x = 0;
   Nodes.IntPoint(9).y = 0;
   Nodes.IntPoint(9).z = 0.6666666666666666666666666667;
   Nodes.IntPoint(10).x = 0.6666666666666666666666666667;
   Nodes.IntPoint(10).y = 0.3333333333333333333333333333;
   Nodes.IntPoint(10).z = 0;
   Nodes.IntPoint(11).x = 0.3333333333333333333333333333;
   Nodes.IntPoint(11).y = 0.6666666666666666666666666667;
   Nodes.IntPoint(11).z = 0;
   Nodes.IntPoint(12).x = 0.6666666666666666666666666667;
   Nodes.IntPoint(12).y = 0;
   Nodes.IntPoint(12).z = 0.3333333333333333333333333333;
   Nodes.IntPoint(13).x = 0.3333333333333333333333333333;
   Nodes.IntPoint(13).y = 0;
   Nodes.IntPoint(13).z = 0.6666666666666666666666666667;
   Nodes.IntPoint(14).x = 0;
   Nodes.IntPoint(14).y = 0.6666666666666666666666666667;
   Nodes.IntPoint(14).z = 0.3333333333333333333333333333;
   Nodes.IntPoint(15).x = 0;
   Nodes.IntPoint(15).y = 0.3333333333333333333333333333;
   Nodes.IntPoint(15).z = 0.6666666666666666666666666667;
   Nodes.IntPoint(16).x = 0.3333333333333333333333333333;
   Nodes.IntPoint(16).y = 0.3333333333333333333333333333;
   Nodes.IntPoint(16).z = 0.3333333333333333333333333333;
   Nodes.IntPoint(17).x = 0;
   Nodes.IntPoint(17).y = 0.3333333333333333333333333333;
   Nodes.IntPoint(17).z = 0.3333333333333333333333333333;
   Nodes.IntPoint(18).x = 0.3333333333333333333333333333;
   Nodes.IntPoint(18).y = 0;
   Nodes.IntPoint(18).z = 0.3333333333333333333333333333;
   Nodes.IntPoint(19).x = 0.3333333333333333333333333333;
   Nodes.IntPoint(19).y = 0.3333333333333333333333333333;
   Nodes.IntPoint(19).z = 0;
}

void Cubic3DFiniteElement::CalcShape(const IntegrationPoint &ip,
                                     Vector &shape) const
{
   real_t x = ip.x, y = ip.y, z = ip.z;

   shape(0) = -((-1 + x + y + z)*(-2 + 3*x + 3*y + 3*z)*
                (-1 + 3*x + 3*y + 3*z))/2.;
   shape(4) = (9*x*(-1 + x + y + z)*(-2 + 3*x + 3*y + 3*z))/2.;
   shape(5) = (-9*x*(-1 + 3*x)*(-1 + x + y + z))/2.;
   shape(1) = (x*(2 + 9*(-1 + x)*x))/2.;
   shape(6) = (9*y*(-1 + x + y + z)*(-2 + 3*x + 3*y + 3*z))/2.;
   shape(19) = -27*x*y*(-1 + x + y + z);
   shape(10) = (9*x*(-1 + 3*x)*y)/2.;
   shape(7) = (-9*y*(-1 + 3*y)*(-1 + x + y + z))/2.;
   shape(11) = (9*x*y*(-1 + 3*y))/2.;
   shape(2) = (y*(2 + 9*(-1 + y)*y))/2.;
   shape(8) = (9*z*(-1 + x + y + z)*(-2 + 3*x + 3*y + 3*z))/2.;
   shape(18) = -27*x*z*(-1 + x + y + z);
   shape(12) = (9*x*(-1 + 3*x)*z)/2.;
   shape(17) = -27*y*z*(-1 + x + y + z);
   shape(16) = 27*x*y*z;
   shape(14) = (9*y*(-1 + 3*y)*z)/2.;
   shape(9) = (-9*z*(-1 + x + y + z)*(-1 + 3*z))/2.;
   shape(13) = (9*x*z*(-1 + 3*z))/2.;
   shape(15) = (9*y*z*(-1 + 3*z))/2.;
   shape(3) = (z*(2 + 9*(-1 + z)*z))/2.;
}

void Cubic3DFiniteElement::CalcDShape(const IntegrationPoint &ip,
                                      DenseMatrix &dshape) const
{
   real_t x = ip.x, y = ip.y, z = ip.z;

   dshape(0,0) = (-11 + 36*y + 36*z - 9*(3*pow(x,2) + 3*pow(y + z,2) +
                                         x*(-4 + 6*y + 6*z)))/2.;
   dshape(0,1) = (-11 + 36*y + 36*z - 9*(3*pow(x,2) + 3*pow(y + z,2) +
                                         x*(-4 + 6*y + 6*z)))/2.;
   dshape(0,2) = (-11 + 36*y + 36*z - 9*(3*pow(x,2) + 3*pow(y + z,2) +
                                         x*(-4 + 6*y + 6*z)))/2.;
   dshape(4,0) = (9*(9*pow(x,2) + (-1 + y + z)*(-2 + 3*y + 3*z) +
                     2*x*(-5 + 6*y + 6*z)))/2.;
   dshape(4,1) = (9*x*(-5 + 6*x + 6*y + 6*z))/2.;
   dshape(4,2) = (9*x*(-5 + 6*x + 6*y + 6*z))/2.;
   dshape(5,0) = (-9*(1 - y - z + x*(-8 + 9*x + 6*y + 6*z)))/2.;
   dshape(5,1) = (9*(1 - 3*x)*x)/2.;
   dshape(5,2) = (9*(1 - 3*x)*x)/2.;
   dshape(1,0) = 1 + (9*x*(-2 + 3*x))/2.;
   dshape(1,1) = 0;
   dshape(1,2) = 0;
   dshape(6,0) = (9*y*(-5 + 6*x + 6*y + 6*z))/2.;
   dshape(6,1) = (9*(2 + 3*pow(x,2) - 10*y - 5*z + 3*(y + z)*(3*y + z) +
                     x*(-5 + 12*y + 6*z)))/2.;
   dshape(6,2) = (9*y*(-5 + 6*x + 6*y + 6*z))/2.;
   dshape(19,0) = -27*y*(-1 + 2*x + y + z);
   dshape(19,1) = -27*x*(-1 + x + 2*y + z);
   dshape(19,2) = -27*x*y;
   dshape(10,0) = (9*(-1 + 6*x)*y)/2.;
   dshape(10,1) = (9*x*(-1 + 3*x))/2.;
   dshape(10,2) = 0;
   dshape(7,0) = (9*(1 - 3*y)*y)/2.;
   dshape(7,1) = (-9*(1 + x*(-1 + 6*y) - z + y*(-8 + 9*y + 6*z)))/2.;
   dshape(7,2) = (9*(1 - 3*y)*y)/2.;
   dshape(11,0) = (9*y*(-1 + 3*y))/2.;
   dshape(11,1) = (9*x*(-1 + 6*y))/2.;
   dshape(11,2) = 0;
   dshape(2,0) = 0;
   dshape(2,1) = 1 + (9*y*(-2 + 3*y))/2.;
   dshape(2,2) = 0;
   dshape(8,0) = (9*z*(-5 + 6*x + 6*y + 6*z))/2.;
   dshape(8,1) = (9*z*(-5 + 6*x + 6*y + 6*z))/2.;
   dshape(8,2) = (9*(2 + 3*pow(x,2) - 5*y - 10*z + 3*(y + z)*(y + 3*z) +
                     x*(-5 + 6*y + 12*z)))/2.;
   dshape(18,0) = -27*z*(-1 + 2*x + y + z);
   dshape(18,1) = -27*x*z;
   dshape(18,2) = -27*x*(-1 + x + y + 2*z);
   dshape(12,0) = (9*(-1 + 6*x)*z)/2.;
   dshape(12,1) = 0;
   dshape(12,2) = (9*x*(-1 + 3*x))/2.;
   dshape(17,0) = -27*y*z;
   dshape(17,1) = -27*z*(-1 + x + 2*y + z);
   dshape(17,2) = -27*y*(-1 + x + y + 2*z);
   dshape(16,0) = 27*y*z;
   dshape(16,1) = 27*x*z;
   dshape(16,2) = 27*x*y;
   dshape(14,0) = 0;
   dshape(14,1) = (9*(-1 + 6*y)*z)/2.;
   dshape(14,2) = (9*y*(-1 + 3*y))/2.;
   dshape(9,0) = (9*(1 - 3*z)*z)/2.;
   dshape(9,1) = (9*(1 - 3*z)*z)/2.;
   dshape(9,2) = (9*(-1 + x + y + 8*z - 6*(x + y)*z - 9*pow(z,2)))/2.;
   dshape(13,0) = (9*z*(-1 + 3*z))/2.;
   dshape(13,1) = 0;
   dshape(13,2) = (9*x*(-1 + 6*z))/2.;
   dshape(15,0) = 0;
   dshape(15,1) = (9*z*(-1 + 3*z))/2.;
   dshape(15,2) = (9*y*(-1 + 6*z))/2.;
   dshape(3,0) = 0;
   dshape(3,1) = 0;
   dshape(3,2) = 1 + (9*z*(-2 + 3*z))/2.;
}


P0TriangleFiniteElement::P0TriangleFiniteElement()
   : NodalFiniteElement(2, Geometry::TRIANGLE, 1, 0)
{
   Nodes.IntPoint(0).x = 0.333333333333333333;
   Nodes.IntPoint(0).y = 0.333333333333333333;
}

void P0TriangleFiniteElement::CalcShape(const IntegrationPoint &ip,
                                        Vector &shape) const
{
   shape(0) = 1.0;
}

void P0TriangleFiniteElement::CalcDShape(const IntegrationPoint &ip,
                                         DenseMatrix &dshape) const
{
   dshape(0,0) = 0.0;
   dshape(0,1) = 0.0;
}


P0QuadFiniteElement::P0QuadFiniteElement()
   : NodalFiniteElement(2, Geometry::SQUARE, 1, 0, FunctionSpace::Qk)
{
   Nodes.IntPoint(0).x = 0.5;
   Nodes.IntPoint(0).y = 0.5;
}

void P0QuadFiniteElement::CalcShape(const IntegrationPoint &ip,
                                    Vector &shape) const
{
   shape(0) = 1.0;
}

void P0QuadFiniteElement::CalcDShape(const IntegrationPoint &ip,
                                     DenseMatrix &dshape) const
{
   dshape(0,0) = 0.0;
   dshape(0,1) = 0.0;
}


Linear3DFiniteElement::Linear3DFiniteElement()
   : NodalFiniteElement(3, Geometry::TETRAHEDRON, 4, 1)
{
   Nodes.IntPoint(0).x = 0.0;
   Nodes.IntPoint(0).y = 0.0;
   Nodes.IntPoint(0).z = 0.0;
   Nodes.IntPoint(1).x = 1.0;
   Nodes.IntPoint(1).y = 0.0;
   Nodes.IntPoint(1).z = 0.0;
   Nodes.IntPoint(2).x = 0.0;
   Nodes.IntPoint(2).y = 1.0;
   Nodes.IntPoint(2).z = 0.0;
   Nodes.IntPoint(3).x = 0.0;
   Nodes.IntPoint(3).y = 0.0;
   Nodes.IntPoint(3).z = 1.0;
}

void Linear3DFiniteElement::CalcShape(const IntegrationPoint &ip,
                                      Vector &shape) const
{
   shape(0) = 1. - ip.x - ip.y - ip.z;
   shape(1) = ip.x;
   shape(2) = ip.y;
   shape(3) = ip.z;
}

void Linear3DFiniteElement::CalcDShape(const IntegrationPoint &ip,
                                       DenseMatrix &dshape) const
{
   if (dshape.Height() == 4)
   {
      real_t *A = &dshape(0,0);
      A[0] = -1.; A[4] = -1.; A[8]  = -1.;
      A[1] =  1.; A[5] =  0.; A[9]  =  0.;
      A[2] =  0.; A[6] =  1.; A[10] =  0.;
      A[3] =  0.; A[7] =  0.; A[11] =  1.;
   }
   else
   {
      dshape(0,0) = -1.; dshape(0,1) = -1.; dshape(0,2) = -1.;
      dshape(1,0) =  1.; dshape(1,1) =  0.; dshape(1,2) =  0.;
      dshape(2,0) =  0.; dshape(2,1) =  1.; dshape(2,2) =  0.;
      dshape(3,0) =  0.; dshape(3,1) =  0.; dshape(3,2) =  1.;
   }
}

void Linear3DFiniteElement::CalcHessian(const IntegrationPoint &ip,
                                        DenseMatrix &h) const
{
   h = 0.0;
}

void Linear3DFiniteElement::GetFaceDofs (int face, int **dofs, int *ndofs)
const
{
   static int face_dofs[4][3] = {{1, 2, 3}, {0, 2, 3}, {0, 1, 3}, {0, 1, 2}};

   *ndofs = 3;
   *dofs  = face_dofs[face];
}


// TODO: use a FunctionSpace specific to wedges instead of Qk.
LinearWedgeFiniteElement::LinearWedgeFiniteElement()
   : NodalFiniteElement(3, Geometry::PRISM, 6, 1, FunctionSpace::Qk)
{
   Nodes.IntPoint(0).x = 0.0;
   Nodes.IntPoint(0).y = 0.0;
   Nodes.IntPoint(0).z = 0.0;
   Nodes.IntPoint(1).x = 1.0;
   Nodes.IntPoint(1).y = 0.0;
   Nodes.IntPoint(1).z = 0.0;
   Nodes.IntPoint(2).x = 0.0;
   Nodes.IntPoint(2).y = 1.0;
   Nodes.IntPoint(2).z = 0.0;
   Nodes.IntPoint(3).x = 0.0;
   Nodes.IntPoint(3).y = 0.0;
   Nodes.IntPoint(3).z = 1.0;
   Nodes.IntPoint(4).x = 1.0;
   Nodes.IntPoint(4).y = 0.0;
   Nodes.IntPoint(4).z = 1.0;
   Nodes.IntPoint(5).x = 0.0;
   Nodes.IntPoint(5).y = 1.0;
   Nodes.IntPoint(5).z = 1.0;
}

void LinearWedgeFiniteElement::CalcShape(const IntegrationPoint &ip,
                                         Vector &shape) const
{
   shape(0) = (1. - ip.x - ip.y) * (1. - ip.z);
   shape(1) = ip.x * (1. - ip.z);
   shape(2) = ip.y * (1. - ip.z);
   shape(3) = (1. - ip.x - ip.y) * ip.z;
   shape(4) = ip.x * ip.z;
   shape(5) = ip.y * ip.z;
}

void LinearWedgeFiniteElement::CalcDShape(const IntegrationPoint &ip,
                                          DenseMatrix &dshape) const
{
   dshape(0,0) = -1. + ip.z;
   dshape(0,1) = -1. + ip.z;
   dshape(0,2) = -1. + ip.x + ip.y;

   dshape(1,0) =  1. - ip.z;
   dshape(1,1) =  0.;
   dshape(1,2) = -ip.x;

   dshape(2,0) =  0.;
   dshape(2,1) =  1. - ip.z;
   dshape(2,2) = -ip.y;

   dshape(3,0) = -ip.z;
   dshape(3,1) = -ip.z;
   dshape(3,2) =  1. - ip.x - ip.y;

   dshape(4,0) =  ip.z;
   dshape(4,1) =  0.;
   dshape(4,2) =  ip.x;

   dshape(5,0) =  0.;
   dshape(5,1) =  ip.z;
   dshape(5,2) =  ip.y;
}

void LinearWedgeFiniteElement::GetFaceDofs (int face, int **dofs, int *ndofs)
const
{
   static int face_dofs[5][4] =
   {{0, 2, 1, -1}, {3, 4, 5, -1}, {0, 1, 4, 3}, {1, 2, 5, 4}, {2, 0, 3, 5}};

   *ndofs = (face < 2) ? 3 : 4;
   *dofs  = face_dofs[face];
}


LinearPyramidFiniteElement::LinearPyramidFiniteElement()
   : NodalFiniteElement(3, Geometry::PYRAMID, 5, 1, FunctionSpace::Uk)
{
   Nodes.IntPoint(0).x = 0.0;
   Nodes.IntPoint(0).y = 0.0;
   Nodes.IntPoint(0).z = 0.0;
   Nodes.IntPoint(1).x = 1.0;
   Nodes.IntPoint(1).y = 0.0;
   Nodes.IntPoint(1).z = 0.0;
   Nodes.IntPoint(2).x = 1.0;
   Nodes.IntPoint(2).y = 1.0;
   Nodes.IntPoint(2).z = 0.0;
   Nodes.IntPoint(3).x = 0.0;
   Nodes.IntPoint(3).y = 1.0;
   Nodes.IntPoint(3).z = 0.0;
   Nodes.IntPoint(4).x = 0.0;
   Nodes.IntPoint(4).y = 0.0;
   Nodes.IntPoint(4).z = 1.0;
}

void LinearPyramidFiniteElement::CalcShape(const IntegrationPoint &ip,
                                           Vector &shape) const
{
   real_t x = ip.x, y = ip.y, z = ip.z;
   real_t ox = 1.-x-z, oy = 1.-y-z, oz = 1.-z;

   real_t tol = 1e-6;

   if (oz <= tol)
   {
      // We must return the limit of the basis functions as z->1.  In order to
      // remain inside the pyramid in this limit the x and y coordinates must
      // be approaching 0. The resulting limiting basis function values are:
      shape(0) = 0.;
      shape(1) = 0.;
      shape(2) = 0.;
      shape(3) = 0.;
      shape(4) = 1.;
      return;
   }

   real_t ozi = 1. / oz;

   shape(0) = ox * oy * ozi;
   shape(1) =  x * oy * ozi;
   shape(2) =  x *  y * ozi;
   shape(3) = ox *  y * ozi;
   shape(4) = z;
}

void LinearPyramidFiniteElement::CalcDShape(const IntegrationPoint &ip,
                                            DenseMatrix &dshape) const
{
   real_t x = ip.x, y = ip.y, z = ip.z;
   real_t ox = 1.-x-z, oy = 1.-y-z, oz = 1.-z;

   real_t tol = 1e-6;

   if (oz <= tol)
   {
      // At the apex of the pyramid the gradients of the basis functions are
      // multivalued and depend on the direction from which the limit is taken.
      // The following values correspond to the average of the gradients taken
      // over all possible directions approaching the apex of the pyramid from
      // within its interior.
      dshape(0,0) = - 0.5;
      dshape(0,1) = - 0.5;
      dshape(0,2) = - 0.75;

      dshape(1,0) =   0.5;
      dshape(1,1) = - 0.5;
      dshape(1,2) = - 0.25;

      dshape(2,0) =   0.5;
      dshape(2,1) =   0.5;
      dshape(2,2) =   0.25;

      dshape(3,0) = - 0.5;
      dshape(3,1) =   0.5;
      dshape(3,2) = - 0.25;

      dshape(4,0) =   0.;
      dshape(4,1) =   0.;
      dshape(4,2) =   1.;

      return;
   }

   real_t ozi = 1. / oz;

   dshape(0,0) = - oy * ozi;
   dshape(0,1) = - ox * ozi;
   dshape(0,2) =   x * y * ozi * ozi - 1.;

   dshape(1,0) =   oy * ozi;
   dshape(1,1) = -  x * ozi;
   dshape(1,2) = -  x * y * ozi * ozi;

   dshape(2,0) =    y * ozi;
   dshape(2,1) =    x * ozi;
   dshape(2,2) =    x *  y * ozi * ozi;

   dshape(3,0) = -  y * ozi;
   dshape(3,1) =   ox * ozi;
   dshape(3,2) = -  x *  y * ozi * ozi;

   dshape(4,0) =   0.;
   dshape(4,1) =   0.;
   dshape(4,2) =   1.;
}

void LinearPyramidFiniteElement::GetFaceDofs (int face, int **dofs, int *ndofs)
const
{
   static int face_dofs[5][4] =
   {{3, 2, 1, 0}, {0, 1, 4, -1}, {1, 2, 4, -1}, {2, 3, 4, -1}, {3, 0, 4, -1}};

   *ndofs = (face < 1) ? 4 : 3;
   *dofs  = face_dofs[face];
}


Quadratic3DFiniteElement::Quadratic3DFiniteElement()
   : NodalFiniteElement(3, Geometry::TETRAHEDRON, 10, 2)
{
   Nodes.IntPoint(0).x = 0.0;
   Nodes.IntPoint(0).y = 0.0;
   Nodes.IntPoint(0).z = 0.0;
   Nodes.IntPoint(1).x = 1.0;
   Nodes.IntPoint(1).y = 0.0;
   Nodes.IntPoint(1).z = 0.0;
   Nodes.IntPoint(2).x = 0.0;
   Nodes.IntPoint(2).y = 1.0;
   Nodes.IntPoint(2).z = 0.0;
   Nodes.IntPoint(3).x = 0.0;
   Nodes.IntPoint(3).y = 0.0;
   Nodes.IntPoint(3).z = 1.0;
   Nodes.IntPoint(4).x = 0.5;
   Nodes.IntPoint(4).y = 0.0;
   Nodes.IntPoint(4).z = 0.0;
   Nodes.IntPoint(5).x = 0.0;
   Nodes.IntPoint(5).y = 0.5;
   Nodes.IntPoint(5).z = 0.0;
   Nodes.IntPoint(6).x = 0.0;
   Nodes.IntPoint(6).y = 0.0;
   Nodes.IntPoint(6).z = 0.5;
   Nodes.IntPoint(7).x = 0.5;
   Nodes.IntPoint(7).y = 0.5;
   Nodes.IntPoint(7).z = 0.0;
   Nodes.IntPoint(8).x = 0.5;
   Nodes.IntPoint(8).y = 0.0;
   Nodes.IntPoint(8).z = 0.5;
   Nodes.IntPoint(9).x = 0.0;
   Nodes.IntPoint(9).y = 0.5;
   Nodes.IntPoint(9).z = 0.5;
}

void Quadratic3DFiniteElement::CalcShape(const IntegrationPoint &ip,
                                         Vector &shape) const
{
   real_t L0, L1, L2, L3;

   L0 = 1. - ip.x - ip.y - ip.z;
   L1 = ip.x;
   L2 = ip.y;
   L3 = ip.z;

   shape(0) = L0 * ( 2.0 * L0 - 1.0 );
   shape(1) = L1 * ( 2.0 * L1 - 1.0 );
   shape(2) = L2 * ( 2.0 * L2 - 1.0 );
   shape(3) = L3 * ( 2.0 * L3 - 1.0 );
   shape(4) = 4.0 * L0 * L1;
   shape(5) = 4.0 * L0 * L2;
   shape(6) = 4.0 * L0 * L3;
   shape(7) = 4.0 * L1 * L2;
   shape(8) = 4.0 * L1 * L3;
   shape(9) = 4.0 * L2 * L3;
}

void Quadratic3DFiniteElement::CalcDShape(const IntegrationPoint &ip,
                                          DenseMatrix &dshape) const
{
   real_t x, y, z, L0;

   x = ip.x;
   y = ip.y;
   z = ip.z;
   L0 = 1.0 - x - y - z;

   dshape(0,0) = dshape(0,1) = dshape(0,2) = 1.0 - 4.0 * L0;
   dshape(1,0) = -1.0 + 4.0 * x; dshape(1,1) = 0.0; dshape(1,2) = 0.0;
   dshape(2,0) = 0.0; dshape(2,1) = -1.0 + 4.0 * y; dshape(2,2) = 0.0;
   dshape(3,0) = dshape(3,1) = 0.0; dshape(3,2) = -1.0 + 4.0 * z;
   dshape(4,0) = 4.0 * (L0 - x); dshape(4,1) = dshape(4,2) = -4.0 * x;
   dshape(5,0) = dshape(5,2) = -4.0 * y; dshape(5,1) = 4.0 * (L0 - y);
   dshape(6,0) = dshape(6,1) = -4.0 * z; dshape(6,2) = 4.0 * (L0 - z);
   dshape(7,0) = 4.0 * y; dshape(7,1) = 4.0 * x; dshape(7,2) = 0.0;
   dshape(8,0) = 4.0 * z; dshape(8,1) = 0.0; dshape(8,2) = 4.0 * x;
   dshape(9,0) = 0.0; dshape(9,1) = 4.0 * z; dshape(9,2) = 4.0 * y;
}

TriLinear3DFiniteElement::TriLinear3DFiniteElement()
   : NodalFiniteElement(3, Geometry::CUBE, 8, 1, FunctionSpace::Qk)
{
   Nodes.IntPoint(0).x = 0.0;
   Nodes.IntPoint(0).y = 0.0;
   Nodes.IntPoint(0).z = 0.0;

   Nodes.IntPoint(1).x = 1.0;
   Nodes.IntPoint(1).y = 0.0;
   Nodes.IntPoint(1).z = 0.0;

   Nodes.IntPoint(2).x = 1.0;
   Nodes.IntPoint(2).y = 1.0;
   Nodes.IntPoint(2).z = 0.0;

   Nodes.IntPoint(3).x = 0.0;
   Nodes.IntPoint(3).y = 1.0;
   Nodes.IntPoint(3).z = 0.0;

   Nodes.IntPoint(4).x = 0.0;
   Nodes.IntPoint(4).y = 0.0;
   Nodes.IntPoint(4).z = 1.0;

   Nodes.IntPoint(5).x = 1.0;
   Nodes.IntPoint(5).y = 0.0;
   Nodes.IntPoint(5).z = 1.0;

   Nodes.IntPoint(6).x = 1.0;
   Nodes.IntPoint(6).y = 1.0;
   Nodes.IntPoint(6).z = 1.0;

   Nodes.IntPoint(7).x = 0.0;
   Nodes.IntPoint(7).y = 1.0;
   Nodes.IntPoint(7).z = 1.0;
}

void TriLinear3DFiniteElement::CalcShape(const IntegrationPoint &ip,
                                         Vector &shape) const
{
   real_t x = ip.x, y = ip.y, z = ip.z;
   real_t ox = 1.-x, oy = 1.-y, oz = 1.-z;

   shape(0) = ox * oy * oz;
   shape(1) =  x * oy * oz;
   shape(2) =  x *  y * oz;
   shape(3) = ox *  y * oz;
   shape(4) = ox * oy *  z;
   shape(5) =  x * oy *  z;
   shape(6) =  x *  y *  z;
   shape(7) = ox *  y *  z;
}

void TriLinear3DFiniteElement::CalcDShape(const IntegrationPoint &ip,
                                          DenseMatrix &dshape) const
{
   real_t x = ip.x, y = ip.y, z = ip.z;
   real_t ox = 1.-x, oy = 1.-y, oz = 1.-z;

   dshape(0,0) = - oy * oz;
   dshape(0,1) = - ox * oz;
   dshape(0,2) = - ox * oy;

   dshape(1,0) =   oy * oz;
   dshape(1,1) = -  x * oz;
   dshape(1,2) = -  x * oy;

   dshape(2,0) =    y * oz;
   dshape(2,1) =    x * oz;
   dshape(2,2) = -  x *  y;

   dshape(3,0) = -  y * oz;
   dshape(3,1) =   ox * oz;
   dshape(3,2) = - ox *  y;

   dshape(4,0) = - oy *  z;
   dshape(4,1) = - ox *  z;
   dshape(4,2) =   ox * oy;

   dshape(5,0) =   oy *  z;
   dshape(5,1) = -  x *  z;
   dshape(5,2) =    x * oy;

   dshape(6,0) =    y *  z;
   dshape(6,1) =    x *  z;
   dshape(6,2) =    x *  y;

   dshape(7,0) = -  y *  z;
   dshape(7,1) =   ox *  z;
   dshape(7,2) =   ox *  y;
}

void TriLinear3DFiniteElement::CalcHessian(const IntegrationPoint &ip,
                                           DenseMatrix &h) const
{
   real_t x = ip.x, y = ip.y, z = ip.z;
   real_t ox = 1.-x, oy = 1.-y, oz = 1.-z;

   h(0,0) = 0.;   h(0,1) =  oz;   h(0,2) = oy;
   h(0,3) = 0.;   h(0,4) =  ox;   h(0,5) = 0.;

   h(1,0) = 0.;   h(1,1) = -oz;   h(1,2) = -oy;
   h(1,3) = 0.;   h(1,4) =  x;    h(1,5) = 0.;

   h(2,0) = 0.;   h(2,1) =  oz;   h(2,2) = -y;
   h(2,3) = 0.;   h(2,4) =  -x;   h(2,5) = 0.;

   h(3,0) = 0.;   h(3,1) = -oz;   h(3,2) = y;
   h(3,3) = 0.;   h(3,4) = -ox;   h(3,5) = 0.;

   h(4,0) = 0.;   h(4,1) =  z;    h(4,2) = -oy;
   h(4,3) = 0.;   h(4,4) = -ox;   h(4,5) = 0.;

   h(5,0) = 0.;   h(5,1) = -z;    h(5,2) = oy;
   h(5,3) = 0.;   h(5,4) = -x;    h(5,5) = 0.;

   h(6,0) = 0.;   h(6,1) =  z;    h(6,2) = y;
   h(6,3) = 0.;   h(6,4) =  x;    h(6,5) = 0.;

   h(7,0) = 0.;   h(7,1) = -z;    h(7,2) = -y;
   h(7,3) = 0.;   h(7,4) = ox;    h(7,5) = 0.;
}


P0SegmentFiniteElement::P0SegmentFiniteElement(int Ord)
   : NodalFiniteElement(1, Geometry::SEGMENT, 1, Ord)   // default Ord = 0
{
   Nodes.IntPoint(0).x = 0.5;
}

void P0SegmentFiniteElement::CalcShape(const IntegrationPoint &ip,
                                       Vector &shape) const
{
   shape(0) = 1.0;
}

void P0SegmentFiniteElement::CalcDShape(const IntegrationPoint &ip,
                                        DenseMatrix &dshape) const
{
   dshape(0,0) = 0.0;
}

CrouzeixRaviartFiniteElement::CrouzeixRaviartFiniteElement()
   : NodalFiniteElement(2, Geometry::TRIANGLE, 3, 1)
{
   Nodes.IntPoint(0).x = 0.5;
   Nodes.IntPoint(0).y = 0.0;
   Nodes.IntPoint(1).x = 0.5;
   Nodes.IntPoint(1).y = 0.5;
   Nodes.IntPoint(2).x = 0.0;
   Nodes.IntPoint(2).y = 0.5;
}

void CrouzeixRaviartFiniteElement::CalcShape(const IntegrationPoint &ip,
                                             Vector &shape) const
{
   shape(0) =  1.0 - 2.0 * ip.y;
   shape(1) = -1.0 + 2.0 * ( ip.x + ip.y );
   shape(2) =  1.0 - 2.0 * ip.x;
}

void CrouzeixRaviartFiniteElement::CalcDShape(const IntegrationPoint &ip,
                                              DenseMatrix &dshape) const
{
   dshape(0,0) =  0.0; dshape(0,1) = -2.0;
   dshape(1,0) =  2.0; dshape(1,1) =  2.0;
   dshape(2,0) = -2.0; dshape(2,1) =  0.0;
}

CrouzeixRaviartQuadFiniteElement::CrouzeixRaviartQuadFiniteElement()
// the FunctionSpace should be rotated (45 degrees) Q_1
// i.e. the span of { 1, x, y, x^2 - y^2 }
   : NodalFiniteElement(2, Geometry::SQUARE, 4, 2, FunctionSpace::Qk)
{
   Nodes.IntPoint(0).x = 0.5;
   Nodes.IntPoint(0).y = 0.0;
   Nodes.IntPoint(1).x = 1.0;
   Nodes.IntPoint(1).y = 0.5;
   Nodes.IntPoint(2).x = 0.5;
   Nodes.IntPoint(2).y = 1.0;
   Nodes.IntPoint(3).x = 0.0;
   Nodes.IntPoint(3).y = 0.5;
}

void CrouzeixRaviartQuadFiniteElement::CalcShape(const IntegrationPoint &ip,
                                                 Vector &shape) const
{
   const real_t l1 = ip.x+ip.y-0.5, l2 = 1.-l1, l3 = ip.x-ip.y+0.5, l4 = 1.-l3;

   shape(0) = l2 * l3;
   shape(1) = l1 * l3;
   shape(2) = l1 * l4;
   shape(3) = l2 * l4;
}

void CrouzeixRaviartQuadFiniteElement::CalcDShape(const IntegrationPoint &ip,
                                                  DenseMatrix &dshape) const
{
   const real_t x2 = 2.*ip.x, y2 = 2.*ip.y;

   dshape(0,0) =  1. - x2; dshape(0,1) = -2. + y2;
   dshape(1,0) =       x2; dshape(1,1) =  1. - y2;
   dshape(2,0) =  1. - x2; dshape(2,1) =       y2;
   dshape(3,0) = -2. + x2; dshape(3,1) =  1. - y2;
}


RT0TriangleFiniteElement::RT0TriangleFiniteElement()
   : VectorFiniteElement(2, Geometry::TRIANGLE, 3, 1, H_DIV)
{
   Nodes.IntPoint(0).x = 0.5;
   Nodes.IntPoint(0).y = 0.0;
   Nodes.IntPoint(1).x = 0.5;
   Nodes.IntPoint(1).y = 0.5;
   Nodes.IntPoint(2).x = 0.0;
   Nodes.IntPoint(2).y = 0.5;
}

void RT0TriangleFiniteElement::CalcVShape(const IntegrationPoint &ip,
                                          DenseMatrix &shape) const
{
   real_t x = ip.x, y = ip.y;

   shape(0,0) = x;
   shape(0,1) = y - 1.;
   shape(1,0) = x;
   shape(1,1) = y;
   shape(2,0) = x - 1.;
   shape(2,1) = y;
}

void RT0TriangleFiniteElement::CalcDivShape(const IntegrationPoint &ip,
                                            Vector &divshape) const
{
   divshape(0) = 2.;
   divshape(1) = 2.;
   divshape(2) = 2.;
}

const real_t RT0TriangleFiniteElement::nk[3][2] =
{ {0, -1}, {1, 1}, {-1, 0} };

void RT0TriangleFiniteElement::GetLocalInterpolation (
   ElementTransformation &Trans, DenseMatrix &I) const
{
   int k, j;
#ifdef MFEM_THREAD_SAFE
   DenseMatrix vshape(dof, dim);
#endif

#ifdef MFEM_DEBUG
   for (k = 0; k < 3; k++)
   {
      CalcVShape (Nodes.IntPoint(k), vshape);
      for (j = 0; j < 3; j++)
      {
         real_t d = vshape(j,0)*nk[k][0]+vshape(j,1)*nk[k][1];
         if (j == k) { d -= 1.0; }
         if (fabs(d) > 1.0e-12)
         {
            mfem::err << "RT0TriangleFiniteElement::GetLocalInterpolation (...)\n"
                      " k = " << k << ", j = " << j << ", d = " << d << endl;
            mfem_error();
         }
      }
   }
#endif

   IntegrationPoint ip;
   ip.x = ip.y = 0.0;
   Trans.SetIntPoint (&ip);
   // Trans must be linear
   // set Jinv = |J| J^{-t} = adj(J)^t
   const DenseMatrix &Jinv = Trans.TransposeAdjugateJacobian();

   real_t vk[2];
   Vector xk (vk, 2);

   for (k = 0; k < 3; k++)
   {
      Trans.Transform (Nodes.IntPoint (k), xk);
      ip.x = vk[0]; ip.y = vk[1];
      CalcVShape (ip, vshape);
      //  vk = |J| J^{-t} nk
      vk[0] = Jinv(0,0)*nk[k][0]+Jinv(0,1)*nk[k][1];
      vk[1] = Jinv(1,0)*nk[k][0]+Jinv(1,1)*nk[k][1];
      for (j = 0; j < 3; j++)
         if (fabs (I(k,j) = vshape(j,0)*vk[0]+vshape(j,1)*vk[1]) < 1.0e-12)
         {
            I(k,j) = 0.0;
         }
   }
}

void RT0TriangleFiniteElement::Project (
   VectorCoefficient &vc, ElementTransformation &Trans,
   Vector &dofs) const
{
   real_t vk[2];
   Vector xk (vk, 2);

   for (int k = 0; k < 3; k++)
   {
      Trans.SetIntPoint (&Nodes.IntPoint (k));
      // set Jinv = |J| J^{-t} = adj(J)^t
      const DenseMatrix &Jinv = Trans.TransposeAdjugateJacobian();

      vc.Eval (xk, Trans, Nodes.IntPoint (k));
      //  xk^t |J| J^{-t} nk
      dofs(k) = (vk[0] * ( Jinv(0,0)*nk[k][0]+Jinv(0,1)*nk[k][1] ) +
                 vk[1] * ( Jinv(1,0)*nk[k][0]+Jinv(1,1)*nk[k][1] ));
   }
}

RT0QuadFiniteElement::RT0QuadFiniteElement()
   : VectorFiniteElement(2, Geometry::SQUARE, 4, 1, H_DIV,
                         FunctionSpace::Qk)
{
   Nodes.IntPoint(0).x = 0.5;
   Nodes.IntPoint(0).y = 0.0;
   Nodes.IntPoint(1).x = 1.0;
   Nodes.IntPoint(1).y = 0.5;
   Nodes.IntPoint(2).x = 0.5;
   Nodes.IntPoint(2).y = 1.0;
   Nodes.IntPoint(3).x = 0.0;
   Nodes.IntPoint(3).y = 0.5;
}

void RT0QuadFiniteElement::CalcVShape(const IntegrationPoint &ip,
                                      DenseMatrix &shape) const
{
   real_t x = ip.x, y = ip.y;

   shape(0,0) = 0;
   shape(0,1) = y - 1.;
   shape(1,0) = x;
   shape(1,1) = 0;
   shape(2,0) = 0;
   shape(2,1) = y;
   shape(3,0) = x - 1.;
   shape(3,1) = 0;
}

void RT0QuadFiniteElement::CalcDivShape(const IntegrationPoint &ip,
                                        Vector &divshape) const
{
   divshape(0) = 1.;
   divshape(1) = 1.;
   divshape(2) = 1.;
   divshape(3) = 1.;
}

const real_t RT0QuadFiniteElement::nk[4][2] =
{ {0, -1}, {1, 0}, {0, 1}, {-1, 0} };

void RT0QuadFiniteElement::GetLocalInterpolation (
   ElementTransformation &Trans, DenseMatrix &I) const
{
   int k, j;
#ifdef MFEM_THREAD_SAFE
   DenseMatrix vshape(dof, dim);
#endif

#ifdef MFEM_DEBUG
   for (k = 0; k < 4; k++)
   {
      CalcVShape (Nodes.IntPoint(k), vshape);
      for (j = 0; j < 4; j++)
      {
         real_t d = vshape(j,0)*nk[k][0]+vshape(j,1)*nk[k][1];
         if (j == k) { d -= 1.0; }
         if (fabs(d) > 1.0e-12)
         {
            mfem::err << "RT0QuadFiniteElement::GetLocalInterpolation (...)\n"
                      " k = " << k << ", j = " << j << ", d = " << d << endl;
            mfem_error();
         }
      }
   }
#endif

   IntegrationPoint ip;
   ip.x = ip.y = 0.0;
   Trans.SetIntPoint (&ip);
   // Trans must be linear (more to have embedding?)
   // set Jinv = |J| J^{-t} = adj(J)^t
   const DenseMatrix &Jinv = Trans.TransposeAdjugateJacobian();

   real_t vk[2];
   Vector xk (vk, 2);

   for (k = 0; k < 4; k++)
   {
      Trans.Transform (Nodes.IntPoint (k), xk);
      ip.x = vk[0]; ip.y = vk[1];
      CalcVShape (ip, vshape);
      //  vk = |J| J^{-t} nk
      vk[0] = Jinv(0,0)*nk[k][0]+Jinv(0,1)*nk[k][1];
      vk[1] = Jinv(1,0)*nk[k][0]+Jinv(1,1)*nk[k][1];
      for (j = 0; j < 4; j++)
         if (fabs (I(k,j) = vshape(j,0)*vk[0]+vshape(j,1)*vk[1]) < 1.0e-12)
         {
            I(k,j) = 0.0;
         }
   }
}

void RT0QuadFiniteElement::Project (
   VectorCoefficient &vc, ElementTransformation &Trans,
   Vector &dofs) const
{
   real_t vk[2];
   Vector xk (vk, 2);

   for (int k = 0; k < 4; k++)
   {
      Trans.SetIntPoint (&Nodes.IntPoint (k));
      // set Jinv = |J| J^{-t} = adj(J)^t
      const DenseMatrix &Jinv = Trans.TransposeAdjugateJacobian();

      vc.Eval (xk, Trans, Nodes.IntPoint (k));
      //  xk^t |J| J^{-t} nk
      dofs(k) = (vk[0] * ( Jinv(0,0)*nk[k][0]+Jinv(0,1)*nk[k][1] ) +
                 vk[1] * ( Jinv(1,0)*nk[k][0]+Jinv(1,1)*nk[k][1] ));
   }
}

RT1TriangleFiniteElement::RT1TriangleFiniteElement()
   : VectorFiniteElement(2, Geometry::TRIANGLE, 8, 2, H_DIV)
{
   Nodes.IntPoint(0).x = 0.33333333333333333333;
   Nodes.IntPoint(0).y = 0.0;
   Nodes.IntPoint(1).x = 0.66666666666666666667;
   Nodes.IntPoint(1).y = 0.0;
   Nodes.IntPoint(2).x = 0.66666666666666666667;
   Nodes.IntPoint(2).y = 0.33333333333333333333;
   Nodes.IntPoint(3).x = 0.33333333333333333333;
   Nodes.IntPoint(3).y = 0.66666666666666666667;
   Nodes.IntPoint(4).x = 0.0;
   Nodes.IntPoint(4).y = 0.66666666666666666667;
   Nodes.IntPoint(5).x = 0.0;
   Nodes.IntPoint(5).y = 0.33333333333333333333;
   Nodes.IntPoint(6).x = 0.33333333333333333333;
   Nodes.IntPoint(6).y = 0.33333333333333333333;
   Nodes.IntPoint(7).x = 0.33333333333333333333;
   Nodes.IntPoint(7).y = 0.33333333333333333333;
}

void RT1TriangleFiniteElement::CalcVShape(const IntegrationPoint &ip,
                                          DenseMatrix &shape) const
{
   real_t x = ip.x, y = ip.y;

   shape(0,0) = -2 * x * (-1 + x + 2 * y);
   shape(0,1) = -2 * (-1 + y) * (-1 + x + 2 * y);
   shape(1,0) =  2 * x * (x - y);
   shape(1,1) =  2 * (x - y) * (-1 + y);
   shape(2,0) =  2 * x * (-1 + 2 * x + y);
   shape(2,1) =  2 * y * (-1 + 2 * x + y);
   shape(3,0) =  2 * x * (-1 + x + 2 * y);
   shape(3,1) =  2 * y * (-1 + x + 2 * y);
   shape(4,0) = -2 * (-1 + x) * (x - y);
   shape(4,1) =  2 * y * (-x + y);
   shape(5,0) = -2 * (-1 + x) * (-1 + 2 * x + y);
   shape(5,1) = -2 * y * (-1 + 2 * x + y);
   shape(6,0) = -3 * x * (-2 + 2 * x + y);
   shape(6,1) = -3 * y * (-1 + 2 * x + y);
   shape(7,0) = -3 * x * (-1 + x + 2 * y);
   shape(7,1) = -3 * y * (-2 + x + 2 * y);
}

void RT1TriangleFiniteElement::CalcDivShape(const IntegrationPoint &ip,
                                            Vector &divshape) const
{
   real_t x = ip.x, y = ip.y;

   divshape(0) = -2 * (-4 + 3 * x + 6 * y);
   divshape(1) =  2 + 6 * x - 6 * y;
   divshape(2) = -4 + 12 * x + 6 * y;
   divshape(3) = -4 + 6 * x + 12 * y;
   divshape(4) =  2 - 6 * x + 6 * y;
   divshape(5) = -2 * (-4 + 6 * x + 3 * y);
   divshape(6) = -9 * (-1 + 2 * x + y);
   divshape(7) = -9 * (-1 + x + 2 * y);
}

const real_t RT1TriangleFiniteElement::nk[8][2] =
{
   { 0,-1}, { 0,-1},
   { 1, 1}, { 1, 1},
   {-1, 0}, {-1, 0},
   { 1, 0}, { 0, 1}
};

void RT1TriangleFiniteElement::GetLocalInterpolation (
   ElementTransformation &Trans, DenseMatrix &I) const
{
   int k, j;
#ifdef MFEM_THREAD_SAFE
   DenseMatrix vshape(dof, dim);
#endif

#ifdef MFEM_DEBUG
   for (k = 0; k < 8; k++)
   {
      CalcVShape (Nodes.IntPoint(k), vshape);
      for (j = 0; j < 8; j++)
      {
         real_t d = vshape(j,0)*nk[k][0]+vshape(j,1)*nk[k][1];
         if (j == k) { d -= 1.0; }
         if (fabs(d) > 1.0e-12)
         {
            mfem::err << "RT1QuadFiniteElement::GetLocalInterpolation (...)\n"
                      " k = " << k << ", j = " << j << ", d = " << d << endl;
            mfem_error();
         }
      }
   }
#endif

   IntegrationPoint ip;
   ip.x = ip.y = 0.0;
   Trans.SetIntPoint (&ip);
   // Trans must be linear (more to have embedding?)
   // set Jinv = |J| J^{-t} = adj(J)^t
   const DenseMatrix &Jinv = Trans.TransposeAdjugateJacobian();

   real_t vk[2];
   Vector xk (vk, 2);

   for (k = 0; k < 8; k++)
   {
      Trans.Transform (Nodes.IntPoint (k), xk);
      ip.x = vk[0]; ip.y = vk[1];
      CalcVShape (ip, vshape);
      //  vk = |J| J^{-t} nk
      vk[0] = Jinv(0,0)*nk[k][0]+Jinv(0,1)*nk[k][1];
      vk[1] = Jinv(1,0)*nk[k][0]+Jinv(1,1)*nk[k][1];
      for (j = 0; j < 8; j++)
         if (fabs (I(k,j) = vshape(j,0)*vk[0]+vshape(j,1)*vk[1]) < 1.0e-12)
         {
            I(k,j) = 0.0;
         }
   }
}

void RT1TriangleFiniteElement::Project (
   VectorCoefficient &vc, ElementTransformation &Trans, Vector &dofs) const
{
   real_t vk[2];
   Vector xk (vk, 2);

   for (int k = 0; k < 8; k++)
   {
      Trans.SetIntPoint (&Nodes.IntPoint (k));
      // set Jinv = |J| J^{-t} = adj(J)^t
      const DenseMatrix &Jinv = Trans.TransposeAdjugateJacobian();

      vc.Eval (xk, Trans, Nodes.IntPoint (k));
      //  xk^t |J| J^{-t} nk
      dofs(k) = (vk[0] * ( Jinv(0,0)*nk[k][0]+Jinv(0,1)*nk[k][1] ) +
                 vk[1] * ( Jinv(1,0)*nk[k][0]+Jinv(1,1)*nk[k][1] ));
      dofs(k) *= 0.5;
   }
}

RT1QuadFiniteElement::RT1QuadFiniteElement()
   : VectorFiniteElement(2, Geometry::SQUARE, 12, 2, H_DIV,
                         FunctionSpace::Qk)
{
   // y = 0
   Nodes.IntPoint(0).x  = 1./3.;
   Nodes.IntPoint(0).y  = 0.0;
   Nodes.IntPoint(1).x  = 2./3.;
   Nodes.IntPoint(1).y  = 0.0;
   // x = 1
   Nodes.IntPoint(2).x  = 1.0;
   Nodes.IntPoint(2).y  = 1./3.;
   Nodes.IntPoint(3).x  = 1.0;
   Nodes.IntPoint(3).y  = 2./3.;
   // y = 1
   Nodes.IntPoint(4).x  = 2./3.;
   Nodes.IntPoint(4).y  = 1.0;
   Nodes.IntPoint(5).x  = 1./3.;
   Nodes.IntPoint(5).y  = 1.0;
   // x = 0
   Nodes.IntPoint(6).x  = 0.0;
   Nodes.IntPoint(6).y  = 2./3.;
   Nodes.IntPoint(7).x  = 0.0;
   Nodes.IntPoint(7).y  = 1./3.;
   // x = 0.5 (interior)
   Nodes.IntPoint(8).x  = 0.5;
   Nodes.IntPoint(8).y  = 1./3.;
   Nodes.IntPoint(9).x  = 0.5;
   Nodes.IntPoint(9).y  = 2./3.;
   // y = 0.5 (interior)
   Nodes.IntPoint(10).x = 1./3.;
   Nodes.IntPoint(10).y = 0.5;
   Nodes.IntPoint(11).x = 2./3.;
   Nodes.IntPoint(11).y = 0.5;
}

void RT1QuadFiniteElement::CalcVShape(const IntegrationPoint &ip,
                                      DenseMatrix &shape) const
{
   real_t x = ip.x, y = ip.y;

   // y = 0
   shape(0,0)  = 0;
   shape(0,1)  = -( 1. - 3.*y + 2.*y*y)*( 2. - 3.*x);
   shape(1,0)  = 0;
   shape(1,1)  = -( 1. - 3.*y + 2.*y*y)*(-1. + 3.*x);
   // x = 1
   shape(2,0)  = (-x + 2.*x*x)*( 2. - 3.*y);
   shape(2,1)  = 0;
   shape(3,0)  = (-x + 2.*x*x)*(-1. + 3.*y);
   shape(3,1)  = 0;
   // y = 1
   shape(4,0)  = 0;
   shape(4,1)  = (-y + 2.*y*y)*(-1. + 3.*x);
   shape(5,0)  = 0;
   shape(5,1)  = (-y + 2.*y*y)*( 2. - 3.*x);
   // x = 0
   shape(6,0)  = -(1. - 3.*x + 2.*x*x)*(-1. + 3.*y);
   shape(6,1)  = 0;
   shape(7,0)  = -(1. - 3.*x + 2.*x*x)*( 2. - 3.*y);
   shape(7,1)  = 0;
   // x = 0.5 (interior)
   shape(8,0)  = (4.*x - 4.*x*x)*( 2. - 3.*y);
   shape(8,1)  = 0;
   shape(9,0)  = (4.*x - 4.*x*x)*(-1. + 3.*y);
   shape(9,1)  = 0;
   // y = 0.5 (interior)
   shape(10,0) = 0;
   shape(10,1) = (4.*y - 4.*y*y)*( 2. - 3.*x);
   shape(11,0) = 0;
   shape(11,1) = (4.*y - 4.*y*y)*(-1. + 3.*x);
}

void RT1QuadFiniteElement::CalcDivShape(const IntegrationPoint &ip,
                                        Vector &divshape) const
{
   real_t x = ip.x, y = ip.y;

   divshape(0)  = -(-3. + 4.*y)*( 2. - 3.*x);
   divshape(1)  = -(-3. + 4.*y)*(-1. + 3.*x);
   divshape(2)  = (-1. + 4.*x)*( 2. - 3.*y);
   divshape(3)  = (-1. + 4.*x)*(-1. + 3.*y);
   divshape(4)  = (-1. + 4.*y)*(-1. + 3.*x);
   divshape(5)  = (-1. + 4.*y)*( 2. - 3.*x);
   divshape(6)  = -(-3. + 4.*x)*(-1. + 3.*y);
   divshape(7)  = -(-3. + 4.*x)*( 2. - 3.*y);
   divshape(8)  = ( 4. - 8.*x)*( 2. - 3.*y);
   divshape(9)  = ( 4. - 8.*x)*(-1. + 3.*y);
   divshape(10) = ( 4. - 8.*y)*( 2. - 3.*x);
   divshape(11) = ( 4. - 8.*y)*(-1. + 3.*x);
}

const real_t RT1QuadFiniteElement::nk[12][2] =
{
   // y = 0
   {0,-1}, {0,-1},
   // X = 1
   {1, 0}, {1, 0},
   // y = 1
   {0, 1}, {0, 1},
   // x = 0
   {-1,0}, {-1,0},
   // x = 0.5 (interior)
   {1, 0}, {1, 0},
   // y = 0.5 (interior)
   {0, 1}, {0, 1}
};

void RT1QuadFiniteElement::GetLocalInterpolation (
   ElementTransformation &Trans, DenseMatrix &I) const
{
   int k, j;
#ifdef MFEM_THREAD_SAFE
   DenseMatrix vshape(dof, dim);
#endif

#ifdef MFEM_DEBUG
   for (k = 0; k < 12; k++)
   {
      CalcVShape (Nodes.IntPoint(k), vshape);
      for (j = 0; j < 12; j++)
      {
         real_t d = vshape(j,0)*nk[k][0]+vshape(j,1)*nk[k][1];
         if (j == k) { d -= 1.0; }
         if (fabs(d) > 1.0e-12)
         {
            mfem::err << "RT1QuadFiniteElement::GetLocalInterpolation (...)\n"
                      " k = " << k << ", j = " << j << ", d = " << d << endl;
            mfem_error();
         }
      }
   }
#endif

   IntegrationPoint ip;
   ip.x = ip.y = 0.0;
   Trans.SetIntPoint (&ip);
   // Trans must be linear (more to have embedding?)
   // set Jinv = |J| J^{-t} = adj(J)^t
   const DenseMatrix &Jinv = Trans.TransposeAdjugateJacobian();

   real_t vk[2];
   Vector xk (vk, 2);

   for (k = 0; k < 12; k++)
   {
      Trans.Transform (Nodes.IntPoint (k), xk);
      ip.x = vk[0]; ip.y = vk[1];
      CalcVShape (ip, vshape);
      //  vk = |J| J^{-t} nk
      vk[0] = Jinv(0,0)*nk[k][0]+Jinv(0,1)*nk[k][1];
      vk[1] = Jinv(1,0)*nk[k][0]+Jinv(1,1)*nk[k][1];
      for (j = 0; j < 12; j++)
         if (fabs (I(k,j) = vshape(j,0)*vk[0]+vshape(j,1)*vk[1]) < 1.0e-12)
         {
            I(k,j) = 0.0;
         }
   }
}

void RT1QuadFiniteElement::Project (
   VectorCoefficient &vc, ElementTransformation &Trans, Vector &dofs) const
{
   real_t vk[2];
   Vector xk (vk, 2);

   for (int k = 0; k < 12; k++)
   {
      Trans.SetIntPoint (&Nodes.IntPoint (k));
      // set Jinv = |J| J^{-t} = adj(J)^t
      const DenseMatrix &Jinv = Trans.TransposeAdjugateJacobian();

      vc.Eval (xk, Trans, Nodes.IntPoint (k));
      //  xk^t |J| J^{-t} nk
      dofs(k) = (vk[0] * ( Jinv(0,0)*nk[k][0]+Jinv(0,1)*nk[k][1] ) +
                 vk[1] * ( Jinv(1,0)*nk[k][0]+Jinv(1,1)*nk[k][1] ));
   }
}

const real_t RT2TriangleFiniteElement::M[15][15] =
{
   // *INDENT-OFF*
   {
      0, -5.3237900077244501311, 5.3237900077244501311, 16.647580015448900262,
      0, 24.442740046346700787, -16.647580015448900262, -12.,
      -19.118950038622250656, -47.237900077244501311, 0, -34.414110069520051180,
      12., 30.590320061795601049, 15.295160030897800524
   },
   {
      0, 1.5, -1.5, -15., 0, 2.625, 15., 15., -4.125, 30., 0, -14.625, -15.,
      -15., 10.5
   },
   {
      0, -0.67620999227554986889, 0.67620999227554986889, 7.3524199845510997378,
      0, -3.4427400463467007866, -7.3524199845510997378, -12.,
      4.1189500386222506555, -0.76209992275549868892, 0, 7.4141100695200511800,
      12., -6.5903200617956010489, -3.2951600308978005244
   },
   {
      0, 0, 1.5, 0, 0, 1.5, -11.471370023173350393, 0, 2.4713700231733503933,
      -11.471370023173350393, 0, 2.4713700231733503933, 15.295160030897800524,
      0, -3.2951600308978005244
   },
   {
      0, 0, 4.875, 0, 0, 4.875, -16.875, 0, -16.875, -16.875, 0, -16.875, 10.5,
      36., 10.5
   },
   {
      0, 0, 1.5, 0, 0, 1.5, 2.4713700231733503933, 0, -11.471370023173350393,
      2.4713700231733503933, 0, -11.471370023173350393, -3.2951600308978005244,
      0, 15.295160030897800524
   },
   {
      -0.67620999227554986889, 0, -3.4427400463467007866, 0,
      7.3524199845510997378, 0.67620999227554986889, 7.4141100695200511800, 0,
      -0.76209992275549868892, 4.1189500386222506555, -12.,
      -7.3524199845510997378, -3.2951600308978005244, -6.5903200617956010489,
      12.
   },
   {
      1.5, 0, 2.625, 0, -15., -1.5, -14.625, 0, 30., -4.125, 15., 15., 10.5,
      -15., -15.
   },
   {
      -5.3237900077244501311, 0, 24.442740046346700787, 0, 16.647580015448900262,
      5.3237900077244501311, -34.414110069520051180, 0, -47.237900077244501311,
      -19.118950038622250656, -12., -16.647580015448900262, 15.295160030897800524,
      30.590320061795601049, 12.
   },
   { 0, 0, 18., 0, 0, 6., -42., 0, -30., -26., 0, -14., 24., 32., 8.},
   { 0, 0, 6., 0, 0, 18., -14., 0, -26., -30., 0, -42., 8., 32., 24.},
   { 0, 0, -6., 0, 0, -4., 30., 0, 4., 22., 0, 4., -24., -16., 0},
   { 0, 0, -4., 0, 0, -8., 20., 0, 8., 36., 0, 8., -16., -32., 0},
   { 0, 0, -8., 0, 0, -4., 8., 0, 36., 8., 0, 20., 0, -32., -16.},
   { 0, 0, -4., 0, 0, -6., 4., 0, 22., 4., 0, 30., 0, -16., -24.}
   // *INDENT-ON*
};

RT2TriangleFiniteElement::RT2TriangleFiniteElement()
   : VectorFiniteElement(2, Geometry::TRIANGLE, 15, 3, H_DIV)
{
   const real_t p = 0.11270166537925831148;

   Nodes.IntPoint(0).x = p;
   Nodes.IntPoint(0).y = 0.0;
   Nodes.IntPoint(1).x = 0.5;
   Nodes.IntPoint(1).y = 0.0;
   Nodes.IntPoint(2).x = 1.-p;
   Nodes.IntPoint(2).y = 0.0;
   Nodes.IntPoint(3).x = 1.-p;
   Nodes.IntPoint(3).y = p;
   Nodes.IntPoint(4).x = 0.5;
   Nodes.IntPoint(4).y = 0.5;
   Nodes.IntPoint(5).x = p;
   Nodes.IntPoint(5).y = 1.-p;
   Nodes.IntPoint(6).x = 0.0;
   Nodes.IntPoint(6).y = 1.-p;
   Nodes.IntPoint(7).x = 0.0;
   Nodes.IntPoint(7).y = 0.5;
   Nodes.IntPoint(8).x = 0.0;
   Nodes.IntPoint(8).y = p;
   Nodes.IntPoint(9).x  = 0.25;
   Nodes.IntPoint(9).y  = 0.25;
   Nodes.IntPoint(10).x = 0.25;
   Nodes.IntPoint(10).y = 0.25;
   Nodes.IntPoint(11).x = 0.5;
   Nodes.IntPoint(11).y = 0.25;
   Nodes.IntPoint(12).x = 0.5;
   Nodes.IntPoint(12).y = 0.25;
   Nodes.IntPoint(13).x = 0.25;
   Nodes.IntPoint(13).y = 0.5;
   Nodes.IntPoint(14).x = 0.25;
   Nodes.IntPoint(14).y = 0.5;
}

void RT2TriangleFiniteElement::CalcVShape(const IntegrationPoint &ip,
                                          DenseMatrix &shape) const
{
   real_t x = ip.x, y = ip.y;

   real_t Bx[15] = {1., 0., x, 0., y, 0., x*x, 0., x*y, 0., y*y, 0., x*x*x,
                    x*x*y, x*y*y
                   };
   real_t By[15] = {0., 1., 0., x, 0., y, 0., x*x, 0., x*y, 0., y*y,
                    x*x*y, x*y*y, y*y*y
                   };

   for (int i = 0; i < 15; i++)
   {
      real_t cx = 0.0, cy = 0.0;
      for (int j = 0; j < 15; j++)
      {
         cx += M[i][j] * Bx[j];
         cy += M[i][j] * By[j];
      }
      shape(i,0) = cx;
      shape(i,1) = cy;
   }
}

void RT2TriangleFiniteElement::CalcDivShape(const IntegrationPoint &ip,
                                            Vector &divshape) const
{
   real_t x = ip.x, y = ip.y;
   constexpr real_t f2 = 2.0;
   constexpr real_t f4 = 4.0;

   real_t DivB[15] = {0., 0., 1., 0., 0., 1., f2*x, 0., y, x, 0., f2*y,
                      f4*x*x, f4*x*y, f4*y*y
                     };

   for (int i = 0; i < 15; i++)
   {
      real_t div = 0.0;
      for (int j = 0; j < 15; j++)
      {
         div += M[i][j] * DivB[j];
      }
      divshape(i) = div;
   }
}

const real_t RT2QuadFiniteElement::pt[4] = {0.,1./3.,2./3.,1.};

const real_t RT2QuadFiniteElement::dpt[3] = {0.25,0.5,0.75};

RT2QuadFiniteElement::RT2QuadFiniteElement()
   : VectorFiniteElement(2, Geometry::SQUARE, 24, 3, H_DIV,
                         FunctionSpace::Qk)
{
   // y = 0 (pt[0])
   Nodes.IntPoint(0).x  = dpt[0];  Nodes.IntPoint(0).y  =  pt[0];
   Nodes.IntPoint(1).x  = dpt[1];  Nodes.IntPoint(1).y  =  pt[0];
   Nodes.IntPoint(2).x  = dpt[2];  Nodes.IntPoint(2).y  =  pt[0];
   // x = 1 (pt[3])
   Nodes.IntPoint(3).x  =  pt[3];  Nodes.IntPoint(3).y  = dpt[0];
   Nodes.IntPoint(4).x  =  pt[3];  Nodes.IntPoint(4).y  = dpt[1];
   Nodes.IntPoint(5).x  =  pt[3];  Nodes.IntPoint(5).y  = dpt[2];
   // y = 1 (pt[3])
   Nodes.IntPoint(6).x  = dpt[2];  Nodes.IntPoint(6).y  =  pt[3];
   Nodes.IntPoint(7).x  = dpt[1];  Nodes.IntPoint(7).y  =  pt[3];
   Nodes.IntPoint(8).x  = dpt[0];  Nodes.IntPoint(8).y  =  pt[3];
   // x = 0 (pt[0])
   Nodes.IntPoint(9).x  =  pt[0];  Nodes.IntPoint(9).y  = dpt[2];
   Nodes.IntPoint(10).x =  pt[0];  Nodes.IntPoint(10).y = dpt[1];
   Nodes.IntPoint(11).x =  pt[0];  Nodes.IntPoint(11).y = dpt[0];
   // x = pt[1] (interior)
   Nodes.IntPoint(12).x =  pt[1];  Nodes.IntPoint(12).y = dpt[0];
   Nodes.IntPoint(13).x =  pt[1];  Nodes.IntPoint(13).y = dpt[1];
   Nodes.IntPoint(14).x =  pt[1];  Nodes.IntPoint(14).y = dpt[2];
   // x = pt[2] (interior)
   Nodes.IntPoint(15).x =  pt[2];  Nodes.IntPoint(15).y = dpt[0];
   Nodes.IntPoint(16).x =  pt[2];  Nodes.IntPoint(16).y = dpt[1];
   Nodes.IntPoint(17).x =  pt[2];  Nodes.IntPoint(17).y = dpt[2];
   // y = pt[1] (interior)
   Nodes.IntPoint(18).x = dpt[0];  Nodes.IntPoint(18).y =  pt[1];
   Nodes.IntPoint(19).x = dpt[1];  Nodes.IntPoint(19).y =  pt[1];
   Nodes.IntPoint(20).x = dpt[2];  Nodes.IntPoint(20).y =  pt[1];
   // y = pt[2] (interior)
   Nodes.IntPoint(21).x = dpt[0];  Nodes.IntPoint(21).y =  pt[2];
   Nodes.IntPoint(22).x = dpt[1];  Nodes.IntPoint(22).y =  pt[2];
   Nodes.IntPoint(23).x = dpt[2];  Nodes.IntPoint(23).y =  pt[2];
}

void RT2QuadFiniteElement::CalcVShape(const IntegrationPoint &ip,
                                      DenseMatrix &shape) const
{
   real_t x = ip.x, y = ip.y;

   real_t ax0 =  pt[0] - x;
   real_t ax1 =  pt[1] - x;
   real_t ax2 =  pt[2] - x;
   real_t ax3 =  pt[3] - x;

   real_t by0 = dpt[0] - y;
   real_t by1 = dpt[1] - y;
   real_t by2 = dpt[2] - y;

   real_t ay0 =  pt[0] - y;
   real_t ay1 =  pt[1] - y;
   real_t ay2 =  pt[2] - y;
   real_t ay3 =  pt[3] - y;

   real_t bx0 = dpt[0] - x;
   real_t bx1 = dpt[1] - x;
   real_t bx2 = dpt[2] - x;

   real_t A01 =  pt[0] -  pt[1];
   real_t A02 =  pt[0] -  pt[2];
   real_t A12 =  pt[1] -  pt[2];
   real_t A03 =  pt[0] -  pt[3];
   real_t A13 =  pt[1] -  pt[3];
   real_t A23 =  pt[2] -  pt[3];

   real_t B01 = dpt[0] - dpt[1];
   real_t B02 = dpt[0] - dpt[2];
   real_t B12 = dpt[1] - dpt[2];

   real_t tx0 =  (bx1*bx2)/(B01*B02);
   real_t tx1 = -(bx0*bx2)/(B01*B12);
   real_t tx2 =  (bx0*bx1)/(B02*B12);

   real_t ty0 =  (by1*by2)/(B01*B02);
   real_t ty1 = -(by0*by2)/(B01*B12);
   real_t ty2 =  (by0*by1)/(B02*B12);

   // y = 0 (p[0])
   shape(0,  0) =  0;
   shape(0,  1) =  (ay1*ay2*ay3)/(A01*A02*A03)*tx0;
   shape(1,  0) =  0;
   shape(1,  1) =  (ay1*ay2*ay3)/(A01*A02*A03)*tx1;
   shape(2,  0) =  0;
   shape(2,  1) =  (ay1*ay2*ay3)/(A01*A02*A03)*tx2;
   // x = 1 (p[3])
   shape(3,  0) =  (ax0*ax1*ax2)/(A03*A13*A23)*ty0;
   shape(3,  1) =  0;
   shape(4,  0) =  (ax0*ax1*ax2)/(A03*A13*A23)*ty1;
   shape(4,  1) =  0;
   shape(5,  0) =  (ax0*ax1*ax2)/(A03*A13*A23)*ty2;
   shape(5,  1) =  0;
   // y = 1 (p[3])
   shape(6,  0) =  0;
   shape(6,  1) =  (ay0*ay1*ay2)/(A03*A13*A23)*tx2;
   shape(7,  0) =  0;
   shape(7,  1) =  (ay0*ay1*ay2)/(A03*A13*A23)*tx1;
   shape(8,  0) =  0;
   shape(8,  1) =  (ay0*ay1*ay2)/(A03*A13*A23)*tx0;
   // x = 0 (p[0])
   shape(9,  0) =  (ax1*ax2*ax3)/(A01*A02*A03)*ty2;
   shape(9,  1) =  0;
   shape(10, 0) =  (ax1*ax2*ax3)/(A01*A02*A03)*ty1;
   shape(10, 1) =  0;
   shape(11, 0) =  (ax1*ax2*ax3)/(A01*A02*A03)*ty0;
   shape(11, 1) =  0;
   // x = p[1] (interior)
   shape(12, 0) =  (ax0*ax2*ax3)/(A01*A12*A13)*ty0;
   shape(12, 1) =  0;
   shape(13, 0) =  (ax0*ax2*ax3)/(A01*A12*A13)*ty1;
   shape(13, 1) =  0;
   shape(14, 0) =  (ax0*ax2*ax3)/(A01*A12*A13)*ty2;
   shape(14, 1) =  0;
   // x = p[2] (interior)
   shape(15, 0) = -(ax0*ax1*ax3)/(A02*A12*A23)*ty0;
   shape(15, 1) =  0;
   shape(16, 0) = -(ax0*ax1*ax3)/(A02*A12*A23)*ty1;
   shape(16, 1) =  0;
   shape(17, 0) = -(ax0*ax1*ax3)/(A02*A12*A23)*ty2;
   shape(17, 1) =  0;
   // y = p[1] (interior)
   shape(18, 0) =  0;
   shape(18, 1) =  (ay0*ay2*ay3)/(A01*A12*A13)*tx0;
   shape(19, 0) =  0;
   shape(19, 1) =  (ay0*ay2*ay3)/(A01*A12*A13)*tx1;
   shape(20, 0) =  0;
   shape(20, 1) =  (ay0*ay2*ay3)/(A01*A12*A13)*tx2;
   // y = p[2] (interior)
   shape(21, 0) =  0;
   shape(21, 1) = -(ay0*ay1*ay3)/(A02*A12*A23)*tx0;
   shape(22, 0) =  0;
   shape(22, 1) = -(ay0*ay1*ay3)/(A02*A12*A23)*tx1;
   shape(23, 0) =  0;
   shape(23, 1) = -(ay0*ay1*ay3)/(A02*A12*A23)*tx2;
}

void RT2QuadFiniteElement::CalcDivShape(const IntegrationPoint &ip,
                                        Vector &divshape) const
{
   real_t x = ip.x, y = ip.y;

   real_t a01 =  pt[0]*pt[1];
   real_t a02 =  pt[0]*pt[2];
   real_t a12 =  pt[1]*pt[2];
   real_t a03 =  pt[0]*pt[3];
   real_t a13 =  pt[1]*pt[3];
   real_t a23 =  pt[2]*pt[3];

   real_t bx0 = dpt[0] - x;
   real_t bx1 = dpt[1] - x;
   real_t bx2 = dpt[2] - x;

   real_t by0 = dpt[0] - y;
   real_t by1 = dpt[1] - y;
   real_t by2 = dpt[2] - y;

   real_t A01 =  pt[0] -  pt[1];
   real_t A02 =  pt[0] -  pt[2];
   real_t A12 =  pt[1] -  pt[2];
   real_t A03 =  pt[0] -  pt[3];
   real_t A13 =  pt[1] -  pt[3];
   real_t A23 =  pt[2] -  pt[3];

   real_t A012 = pt[0] + pt[1] + pt[2];
   real_t A013 = pt[0] + pt[1] + pt[3];
   real_t A023 = pt[0] + pt[2] + pt[3];
   real_t A123 = pt[1] + pt[2] + pt[3];

   real_t B01 = dpt[0] - dpt[1];
   real_t B02 = dpt[0] - dpt[2];
   real_t B12 = dpt[1] - dpt[2];

   real_t tx0 =  (bx1*bx2)/(B01*B02);
   real_t tx1 = -(bx0*bx2)/(B01*B12);
   real_t tx2 =  (bx0*bx1)/(B02*B12);

   real_t ty0 =  (by1*by2)/(B01*B02);
   real_t ty1 = -(by0*by2)/(B01*B12);
   real_t ty2 =  (by0*by1)/(B02*B12);

   // y = 0 (p[0])
   divshape(0)  = -(a12 + a13 + a23 - 2.*A123*y + 3.*y*y)/(A01*A02*A03)*tx0;
   divshape(1)  = -(a12 + a13 + a23 - 2.*A123*y + 3.*y*y)/(A01*A02*A03)*tx1;
   divshape(2)  = -(a12 + a13 + a23 - 2.*A123*y + 3.*y*y)/(A01*A02*A03)*tx2;
   // x = 1 (p[3])
   divshape(3)  = -(a01 + a02 + a12 - 2.*A012*x + 3.*x*x)/(A03*A13*A23)*ty0;
   divshape(4)  = -(a01 + a02 + a12 - 2.*A012*x + 3.*x*x)/(A03*A13*A23)*ty1;
   divshape(5)  = -(a01 + a02 + a12 - 2.*A012*x + 3.*x*x)/(A03*A13*A23)*ty2;
   // y = 1 (p[3])
   divshape(6)  = -(a01 + a02 + a12 - 2.*A012*y + 3.*y*y)/(A03*A13*A23)*tx2;
   divshape(7)  = -(a01 + a02 + a12 - 2.*A012*y + 3.*y*y)/(A03*A13*A23)*tx1;
   divshape(8)  = -(a01 + a02 + a12 - 2.*A012*y + 3.*y*y)/(A03*A13*A23)*tx0;
   // x = 0 (p[0])
   divshape(9)  = -(a12 + a13 + a23 - 2.*A123*x + 3.*x*x)/(A01*A02*A03)*ty2;
   divshape(10) = -(a12 + a13 + a23 - 2.*A123*x + 3.*x*x)/(A01*A02*A03)*ty1;
   divshape(11) = -(a12 + a13 + a23 - 2.*A123*x + 3.*x*x)/(A01*A02*A03)*ty0;
   // x = p[1] (interior)
   divshape(12) = -(a02 + a03 + a23 - 2.*A023*x + 3.*x*x)/(A01*A12*A13)*ty0;
   divshape(13) = -(a02 + a03 + a23 - 2.*A023*x + 3.*x*x)/(A01*A12*A13)*ty1;
   divshape(14) = -(a02 + a03 + a23 - 2.*A023*x + 3.*x*x)/(A01*A12*A13)*ty2;
   // x = p[2] (interior)
   divshape(15) =  (a01 + a03 + a13 - 2.*A013*x + 3.*x*x)/(A02*A12*A23)*ty0;
   divshape(16) =  (a01 + a03 + a13 - 2.*A013*x + 3.*x*x)/(A02*A12*A23)*ty1;
   divshape(17) =  (a01 + a03 + a13 - 2.*A013*x + 3.*x*x)/(A02*A12*A23)*ty2;
   // y = p[1] (interior)
   divshape(18) = -(a02 + a03 + a23 - 2.*A023*y + 3.*y*y)/(A01*A12*A13)*tx0;
   divshape(19) = -(a02 + a03 + a23 - 2.*A023*y + 3.*y*y)/(A01*A12*A13)*tx1;
   divshape(20) = -(a02 + a03 + a23 - 2.*A023*y + 3.*y*y)/(A01*A12*A13)*tx2;
   // y = p[2] (interior)
   divshape(21) =  (a01 + a03 + a13 - 2.*A013*y + 3.*y*y)/(A02*A12*A23)*tx0;
   divshape(22) =  (a01 + a03 + a13 - 2.*A013*y + 3.*y*y)/(A02*A12*A23)*tx1;
   divshape(23) =  (a01 + a03 + a13 - 2.*A013*y + 3.*y*y)/(A02*A12*A23)*tx2;
}

const real_t RT2QuadFiniteElement::nk[24][2] =
{
   // y = 0
   {0,-1}, {0,-1}, {0,-1},
   // x = 1
   {1, 0}, {1, 0}, {1, 0},
   // y = 1
   {0, 1}, {0, 1}, {0, 1},
   // x = 0
   {-1,0}, {-1,0}, {-1,0},
   // x = p[1] (interior)
   {1, 0}, {1, 0}, {1, 0},
   // x = p[2] (interior)
   {1, 0}, {1, 0}, {1, 0},
   // y = p[1] (interior)
   {0, 1}, {0, 1}, {0, 1},
   // y = p[1] (interior)
   {0, 1}, {0, 1}, {0, 1}
};

void RT2QuadFiniteElement::GetLocalInterpolation (
   ElementTransformation &Trans, DenseMatrix &I) const
{
   int k, j;
#ifdef MFEM_THREAD_SAFE
   DenseMatrix vshape(dof, dim);
#endif

#ifdef MFEM_DEBUG
   for (k = 0; k < 24; k++)
   {
      CalcVShape (Nodes.IntPoint(k), vshape);
      for (j = 0; j < 24; j++)
      {
         real_t d = vshape(j,0)*nk[k][0]+vshape(j,1)*nk[k][1];
         if (j == k) { d -= 1.0; }
         if (fabs(d) > 1.0e-12)
         {
            mfem::err << "RT2QuadFiniteElement::GetLocalInterpolation (...)\n"
                      " k = " << k << ", j = " << j << ", d = " << d << endl;
            mfem_error();
         }
      }
   }
#endif

   IntegrationPoint ip;
   ip.x = ip.y = 0.0;
   Trans.SetIntPoint (&ip);
   // Trans must be linear (more to have embedding?)
   // set Jinv = |J| J^{-t} = adj(J)^t
   const DenseMatrix &Jinv = Trans.TransposeAdjugateJacobian();

   real_t vk[2];
   Vector xk (vk, 2);

   for (k = 0; k < 24; k++)
   {
      Trans.Transform (Nodes.IntPoint (k), xk);
      ip.x = vk[0]; ip.y = vk[1];
      CalcVShape (ip, vshape);
      //  vk = |J| J^{-t} nk
      vk[0] = Jinv(0,0)*nk[k][0]+Jinv(0,1)*nk[k][1];
      vk[1] = Jinv(1,0)*nk[k][0]+Jinv(1,1)*nk[k][1];
      for (j = 0; j < 24; j++)
         if (fabs (I(k,j) = vshape(j,0)*vk[0]+vshape(j,1)*vk[1]) < 1.0e-12)
         {
            I(k,j) = 0.0;
         }
   }
}

void RT2QuadFiniteElement::Project (
   VectorCoefficient &vc, ElementTransformation &Trans, Vector &dofs) const
{
   real_t vk[2];
   Vector xk (vk, 2);

   for (int k = 0; k < 24; k++)
   {
      Trans.SetIntPoint (&Nodes.IntPoint (k));
      // set Jinv = |J| J^{-t} = adj(J)^t
      const DenseMatrix &Jinv = Trans.TransposeAdjugateJacobian();

      vc.Eval (xk, Trans, Nodes.IntPoint (k));
      //  xk^t |J| J^{-t} nk
      dofs(k) = (vk[0] * ( Jinv(0,0)*nk[k][0]+Jinv(0,1)*nk[k][1] ) +
                 vk[1] * ( Jinv(1,0)*nk[k][0]+Jinv(1,1)*nk[k][1] ));
   }
}

P1SegmentFiniteElement::P1SegmentFiniteElement()
   : NodalFiniteElement(1, Geometry::SEGMENT, 2, 1)
{
   Nodes.IntPoint(0).x = 0.33333333333333333333;
   Nodes.IntPoint(1).x = 0.66666666666666666667;
}

void P1SegmentFiniteElement::CalcShape(const IntegrationPoint &ip,
                                       Vector &shape) const
{
   real_t x = ip.x;

   shape(0) = 2. - 3. * x;
   shape(1) = 3. * x - 1.;
}

void P1SegmentFiniteElement::CalcDShape(const IntegrationPoint &ip,
                                        DenseMatrix &dshape) const
{
   dshape(0,0) = -3.;
   dshape(1,0) =  3.;
}


P2SegmentFiniteElement::P2SegmentFiniteElement()
   : NodalFiniteElement(1, Geometry::SEGMENT, 3, 2)
{
   const real_t p = 0.11270166537925831148;

   Nodes.IntPoint(0).x = p;
   Nodes.IntPoint(1).x = 0.5;
   Nodes.IntPoint(2).x = 1.-p;
}

void P2SegmentFiniteElement::CalcShape(const IntegrationPoint &ip,
                                       Vector &shape) const
{
   const real_t p = 0.11270166537925831148;
   const real_t w = 1./((1-2*p)*(1-2*p));
   real_t x = ip.x;

   shape(0) = (2*x-1)*(x-1+p)*w;
   shape(1) = 4*(x-1+p)*(p-x)*w;
   shape(2) = (2*x-1)*(x-p)*w;
}

void P2SegmentFiniteElement::CalcDShape(const IntegrationPoint &ip,
                                        DenseMatrix &dshape) const
{
   const real_t p = 0.11270166537925831148;
   const real_t w = 1./((1-2*p)*(1-2*p));
   real_t x = ip.x;

   dshape(0,0) = (-3+4*x+2*p)*w;
   dshape(1,0) = (4-8*x)*w;
   dshape(2,0) = (-1+4*x-2*p)*w;
}


Lagrange1DFiniteElement::Lagrange1DFiniteElement(int degree)
   : NodalFiniteElement(1, Geometry::SEGMENT, degree+1, degree)
{
   int i, m = degree;

   Nodes.IntPoint(0).x = 0.0;
   Nodes.IntPoint(1).x = 1.0;
   for (i = 1; i < m; i++)
   {
      Nodes.IntPoint(i+1).x = real_t(i) / m;
   }

   rwk.SetSize(degree+1);
#ifndef MFEM_THREAD_SAFE
   rxxk.SetSize(degree+1);
#endif

   rwk(0) = 1.0;
   for (i = 1; i <= m; i++)
   {
      rwk(i) = rwk(i-1) * ( (real_t)(m) / (real_t)(i) );
   }
   for (i = 0; i < m/2+1; i++)
   {
      rwk(m-i) = ( rwk(i) *= rwk(m-i) );
   }
   for (i = m-1; i >= 0; i -= 2)
   {
      rwk(i) = -rwk(i);
   }
}

void Lagrange1DFiniteElement::CalcShape(const IntegrationPoint &ip,
                                        Vector &shape) const
{
   real_t w, wk, x = ip.x;
   int i, k, m = GetOrder();

#ifdef MFEM_THREAD_SAFE
   Vector rxxk(m+1);
#endif

   k = (int) floor ( m * x + 0.5 );
   k = k > m ? m : k < 0 ? 0 : k; // clamp k to [0,m]

   wk = 1.0;
   for (i = 0; i <= m; i++)
      if (i != k)
      {
         wk *= ( rxxk(i) = x - (real_t)(i) / m );
      }
   w = wk * ( rxxk(k) = x - (real_t)(k) / m );

   if (k != 0)
   {
      shape(0) = w * rwk(0) / rxxk(0);
   }
   else
   {
      shape(0) = wk * rwk(0);
   }
   if (k != m)
   {
      shape(1) = w * rwk(m) / rxxk(m);
   }
   else
   {
      shape(1) = wk * rwk(k);
   }
   for (i = 1; i < m; i++)
      if (i != k)
      {
         shape(i+1) = w * rwk(i) / rxxk(i);
      }
      else
      {
         shape(k+1) = wk * rwk(k);
      }
}

void Lagrange1DFiniteElement::CalcDShape(const IntegrationPoint &ip,
                                         DenseMatrix &dshape) const
{
   real_t s, srx, w, wk, x = ip.x;
   int i, k, m = GetOrder();

#ifdef MFEM_THREAD_SAFE
   Vector rxxk(m+1);
#endif

   k = (int) floor ( m * x + 0.5 );
   k = k > m ? m : k < 0 ? 0 : k; // clamp k to [0,m]

   wk = 1.0;
   for (i = 0; i <= m; i++)
      if (i != k)
      {
         wk *= ( rxxk(i) = x - (real_t)(i) / m );
      }
   w = wk * ( rxxk(k) = x - (real_t)(k) / m );

   for (i = 0; i <= m; i++)
   {
      rxxk(i) = 1.0 / rxxk(i);
   }
   srx = 0.0;
   for (i = 0; i <= m; i++)
      if (i != k)
      {
         srx += rxxk(i);
      }
   s = w * srx + wk;

   if (k != 0)
   {
      dshape(0,0) = (s - w * rxxk(0)) * rwk(0) * rxxk(0);
   }
   else
   {
      dshape(0,0) = wk * srx * rwk(0);
   }
   if (k != m)
   {
      dshape(1,0) = (s - w * rxxk(m)) * rwk(m) * rxxk(m);
   }
   else
   {
      dshape(1,0) = wk * srx * rwk(k);
   }
   for (i = 1; i < m; i++)
      if (i != k)
      {
         dshape(i+1,0) = (s - w * rxxk(i)) * rwk(i) * rxxk(i);
      }
      else
      {
         dshape(k+1,0) = wk * srx * rwk(k);
      }
}


P1TetNonConfFiniteElement::P1TetNonConfFiniteElement()
   : NodalFiniteElement(3, Geometry::TETRAHEDRON, 4, 1)
{
   Nodes.IntPoint(0).x = 0.33333333333333333333;
   Nodes.IntPoint(0).y = 0.33333333333333333333;
   Nodes.IntPoint(0).z = 0.33333333333333333333;

   Nodes.IntPoint(1).x = 0.0;
   Nodes.IntPoint(1).y = 0.33333333333333333333;
   Nodes.IntPoint(1).z = 0.33333333333333333333;

   Nodes.IntPoint(2).x = 0.33333333333333333333;
   Nodes.IntPoint(2).y = 0.0;
   Nodes.IntPoint(2).z = 0.33333333333333333333;

   Nodes.IntPoint(3).x = 0.33333333333333333333;
   Nodes.IntPoint(3).y = 0.33333333333333333333;
   Nodes.IntPoint(3).z = 0.0;

}

void P1TetNonConfFiniteElement::CalcShape(const IntegrationPoint &ip,
                                          Vector &shape) const
{
   real_t L0, L1, L2, L3;

   L1 = ip.x;  L2 = ip.y;  L3 = ip.z;  L0 = 1.0 - L1 - L2 - L3;
   shape(0) = 1.0 - 3.0 * L0;
   shape(1) = 1.0 - 3.0 * L1;
   shape(2) = 1.0 - 3.0 * L2;
   shape(3) = 1.0 - 3.0 * L3;
}

void P1TetNonConfFiniteElement::CalcDShape(const IntegrationPoint &ip,
                                           DenseMatrix &dshape) const
{
   dshape(0,0) =  3.0; dshape(0,1) =  3.0; dshape(0,2) =  3.0;
   dshape(1,0) = -3.0; dshape(1,1) =  0.0; dshape(1,2) =  0.0;
   dshape(2,0) =  0.0; dshape(2,1) = -3.0; dshape(2,2) =  0.0;
   dshape(3,0) =  0.0; dshape(3,1) =  0.0; dshape(3,2) = -3.0;
}


P0TetFiniteElement::P0TetFiniteElement()
   : NodalFiniteElement(3, Geometry::TETRAHEDRON, 1, 0)
{
   Nodes.IntPoint(0).x = 0.25;
   Nodes.IntPoint(0).y = 0.25;
   Nodes.IntPoint(0).z = 0.25;
}

void P0TetFiniteElement::CalcShape(const IntegrationPoint &ip,
                                   Vector &shape) const
{
   shape(0) = 1.0;
}

void P0TetFiniteElement::CalcDShape(const IntegrationPoint &ip,
                                    DenseMatrix &dshape) const
{
   dshape(0,0) =  0.0; dshape(0,1) =  0.0; dshape(0,2) = 0.0;
}


P0HexFiniteElement::P0HexFiniteElement()
   : NodalFiniteElement(3, Geometry::CUBE, 1, 0, FunctionSpace::Qk)
{
   Nodes.IntPoint(0).x = 0.5;
   Nodes.IntPoint(0).y = 0.5;
   Nodes.IntPoint(0).z = 0.5;
}

void P0HexFiniteElement::CalcShape(const IntegrationPoint &ip,
                                   Vector &shape) const
{
   shape(0) = 1.0;
}

void P0HexFiniteElement::CalcDShape(const IntegrationPoint &ip,
                                    DenseMatrix &dshape) const
{
   dshape(0,0) =  0.0; dshape(0,1) =  0.0; dshape(0,2) = 0.0;
}


P0WdgFiniteElement::P0WdgFiniteElement()
   : NodalFiniteElement(3, Geometry::PRISM, 1, 0, FunctionSpace::Qk)
{
   Nodes.IntPoint(0).x = 1.0 / 3.0;
   Nodes.IntPoint(0).y = 1.0 / 3.0;
   Nodes.IntPoint(0).z = 0.5;
}

void P0WdgFiniteElement::CalcShape(const IntegrationPoint &ip,
                                   Vector &shape) const
{
   shape(0) = 1.0;
}

void P0WdgFiniteElement::CalcDShape(const IntegrationPoint &ip,
                                    DenseMatrix &dshape) const
{
   dshape(0,0) =  0.0; dshape(0,1) =  0.0; dshape(0,2) = 0.0;
}


P0PyrFiniteElement::P0PyrFiniteElement()
   : NodalFiniteElement(3, Geometry::PYRAMID, 1, 0, FunctionSpace::Uk)
{
   Nodes.IntPoint(0).x = 0.375;
   Nodes.IntPoint(0).y = 0.375;
   Nodes.IntPoint(0).z = 0.25;
}

void P0PyrFiniteElement::CalcShape(const IntegrationPoint &ip,
                                   Vector &shape) const
{
   shape(0) = 1.0;
}

void P0PyrFiniteElement::CalcDShape(const IntegrationPoint &ip,
                                    DenseMatrix &dshape) const
{
   dshape(0,0) =  0.0; dshape(0,1) =  0.0; dshape(0,2) = 0.0;
}


LagrangeHexFiniteElement::LagrangeHexFiniteElement (int degree)
   : NodalFiniteElement(3, Geometry::CUBE, (degree+1)*(degree+1)*(degree+1),
                        degree, FunctionSpace::Qk)
{
   if (degree == 2)
   {
      I = new int[dof];
      J = new int[dof];
      K = new int[dof];
      // nodes
      I[ 0] = 0; J[ 0] = 0; K[ 0] = 0;
      I[ 1] = 1; J[ 1] = 0; K[ 1] = 0;
      I[ 2] = 1; J[ 2] = 1; K[ 2] = 0;
      I[ 3] = 0; J[ 3] = 1; K[ 3] = 0;
      I[ 4] = 0; J[ 4] = 0; K[ 4] = 1;
      I[ 5] = 1; J[ 5] = 0; K[ 5] = 1;
      I[ 6] = 1; J[ 6] = 1; K[ 6] = 1;
      I[ 7] = 0; J[ 7] = 1; K[ 7] = 1;
      // edges
      I[ 8] = 2; J[ 8] = 0; K[ 8] = 0;
      I[ 9] = 1; J[ 9] = 2; K[ 9] = 0;
      I[10] = 2; J[10] = 1; K[10] = 0;
      I[11] = 0; J[11] = 2; K[11] = 0;
      I[12] = 2; J[12] = 0; K[12] = 1;
      I[13] = 1; J[13] = 2; K[13] = 1;
      I[14] = 2; J[14] = 1; K[14] = 1;
      I[15] = 0; J[15] = 2; K[15] = 1;
      I[16] = 0; J[16] = 0; K[16] = 2;
      I[17] = 1; J[17] = 0; K[17] = 2;
      I[18] = 1; J[18] = 1; K[18] = 2;
      I[19] = 0; J[19] = 1; K[19] = 2;
      // faces
      I[20] = 2; J[20] = 2; K[20] = 0;
      I[21] = 2; J[21] = 0; K[21] = 2;
      I[22] = 1; J[22] = 2; K[22] = 2;
      I[23] = 2; J[23] = 1; K[23] = 2;
      I[24] = 0; J[24] = 2; K[24] = 2;
      I[25] = 2; J[25] = 2; K[25] = 1;
      // element
      I[26] = 2; J[26] = 2; K[26] = 2;
   }
   else if (degree == 3)
   {
      I = new int[dof];
      J = new int[dof];
      K = new int[dof];
      // nodes
      I[ 0] = 0; J[ 0] = 0; K[ 0] = 0;
      I[ 1] = 1; J[ 1] = 0; K[ 1] = 0;
      I[ 2] = 1; J[ 2] = 1; K[ 2] = 0;
      I[ 3] = 0; J[ 3] = 1; K[ 3] = 0;
      I[ 4] = 0; J[ 4] = 0; K[ 4] = 1;
      I[ 5] = 1; J[ 5] = 0; K[ 5] = 1;
      I[ 6] = 1; J[ 6] = 1; K[ 6] = 1;
      I[ 7] = 0; J[ 7] = 1; K[ 7] = 1;
      // edges
      I[ 8] = 2; J[ 8] = 0; K[ 8] = 0;
      I[ 9] = 3; J[ 9] = 0; K[ 9] = 0;
      I[10] = 1; J[10] = 2; K[10] = 0;
      I[11] = 1; J[11] = 3; K[11] = 0;
      I[12] = 2; J[12] = 1; K[12] = 0;
      I[13] = 3; J[13] = 1; K[13] = 0;
      I[14] = 0; J[14] = 2; K[14] = 0;
      I[15] = 0; J[15] = 3; K[15] = 0;
      I[16] = 2; J[16] = 0; K[16] = 1;
      I[17] = 3; J[17] = 0; K[17] = 1;
      I[18] = 1; J[18] = 2; K[18] = 1;
      I[19] = 1; J[19] = 3; K[19] = 1;
      I[20] = 2; J[20] = 1; K[20] = 1;
      I[21] = 3; J[21] = 1; K[21] = 1;
      I[22] = 0; J[22] = 2; K[22] = 1;
      I[23] = 0; J[23] = 3; K[23] = 1;
      I[24] = 0; J[24] = 0; K[24] = 2;
      I[25] = 0; J[25] = 0; K[25] = 3;
      I[26] = 1; J[26] = 0; K[26] = 2;
      I[27] = 1; J[27] = 0; K[27] = 3;
      I[28] = 1; J[28] = 1; K[28] = 2;
      I[29] = 1; J[29] = 1; K[29] = 3;
      I[30] = 0; J[30] = 1; K[30] = 2;
      I[31] = 0; J[31] = 1; K[31] = 3;
      // faces
      I[32] = 2; J[32] = 3; K[32] = 0;
      I[33] = 3; J[33] = 3; K[33] = 0;
      I[34] = 2; J[34] = 2; K[34] = 0;
      I[35] = 3; J[35] = 2; K[35] = 0;
      I[36] = 2; J[36] = 0; K[36] = 2;
      I[37] = 3; J[37] = 0; K[37] = 2;
      I[38] = 2; J[38] = 0; K[38] = 3;
      I[39] = 3; J[39] = 0; K[39] = 3;
      I[40] = 1; J[40] = 2; K[40] = 2;
      I[41] = 1; J[41] = 3; K[41] = 2;
      I[42] = 1; J[42] = 2; K[42] = 3;
      I[43] = 1; J[43] = 3; K[43] = 3;
      I[44] = 3; J[44] = 1; K[44] = 2;
      I[45] = 2; J[45] = 1; K[45] = 2;
      I[46] = 3; J[46] = 1; K[46] = 3;
      I[47] = 2; J[47] = 1; K[47] = 3;
      I[48] = 0; J[48] = 3; K[48] = 2;
      I[49] = 0; J[49] = 2; K[49] = 2;
      I[50] = 0; J[50] = 3; K[50] = 3;
      I[51] = 0; J[51] = 2; K[51] = 3;
      I[52] = 2; J[52] = 2; K[52] = 1;
      I[53] = 3; J[53] = 2; K[53] = 1;
      I[54] = 2; J[54] = 3; K[54] = 1;
      I[55] = 3; J[55] = 3; K[55] = 1;
      // element
      I[56] = 2; J[56] = 2; K[56] = 2;
      I[57] = 3; J[57] = 2; K[57] = 2;
      I[58] = 3; J[58] = 3; K[58] = 2;
      I[59] = 2; J[59] = 3; K[59] = 2;
      I[60] = 2; J[60] = 2; K[60] = 3;
      I[61] = 3; J[61] = 2; K[61] = 3;
      I[62] = 3; J[62] = 3; K[62] = 3;
      I[63] = 2; J[63] = 3; K[63] = 3;
   }
   else
   {
      mfem_error ("LagrangeHexFiniteElement::LagrangeHexFiniteElement");
   }

   fe1d = new Lagrange1DFiniteElement(degree);
   dof1d = fe1d -> GetDof();

#ifndef MFEM_THREAD_SAFE
   shape1dx.SetSize(dof1d);
   shape1dy.SetSize(dof1d);
   shape1dz.SetSize(dof1d);

   dshape1dx.SetSize(dof1d,1);
   dshape1dy.SetSize(dof1d,1);
   dshape1dz.SetSize(dof1d,1);
#endif

   for (int n = 0; n < dof; n++)
   {
      Nodes.IntPoint(n).x = fe1d -> GetNodes().IntPoint(I[n]).x;
      Nodes.IntPoint(n).y = fe1d -> GetNodes().IntPoint(J[n]).x;
      Nodes.IntPoint(n).z = fe1d -> GetNodes().IntPoint(K[n]).x;
   }
}

void LagrangeHexFiniteElement::CalcShape(const IntegrationPoint &ip,
                                         Vector &shape) const
{
   IntegrationPoint ipy, ipz;
   ipy.x = ip.y;
   ipz.x = ip.z;

#ifdef MFEM_THREAD_SAFE
   Vector shape1dx(dof1d), shape1dy(dof1d), shape1dz(dof1d);
#endif

   fe1d -> CalcShape(ip,  shape1dx);
   fe1d -> CalcShape(ipy, shape1dy);
   fe1d -> CalcShape(ipz, shape1dz);

   for (int n = 0; n < dof; n++)
   {
      shape(n) = shape1dx(I[n]) *  shape1dy(J[n]) * shape1dz(K[n]);
   }
}

void LagrangeHexFiniteElement::CalcDShape(const IntegrationPoint &ip,
                                          DenseMatrix &dshape) const
{
   IntegrationPoint ipy, ipz;
   ipy.x = ip.y;
   ipz.x = ip.z;

#ifdef MFEM_THREAD_SAFE
   Vector shape1dx(dof1d), shape1dy(dof1d), shape1dz(dof1d);
   DenseMatrix dshape1dx(dof1d,1), dshape1dy(dof1d,1), dshape1dz(dof1d,1);
#endif

   fe1d -> CalcShape(ip,  shape1dx);
   fe1d -> CalcShape(ipy, shape1dy);
   fe1d -> CalcShape(ipz, shape1dz);

   fe1d -> CalcDShape(ip,  dshape1dx);
   fe1d -> CalcDShape(ipy, dshape1dy);
   fe1d -> CalcDShape(ipz, dshape1dz);

   for (int n = 0; n < dof; n++)
   {
      dshape(n,0) = dshape1dx(I[n],0) * shape1dy(J[n])    * shape1dz(K[n]);
      dshape(n,1) = shape1dx(I[n])    * dshape1dy(J[n],0) * shape1dz(K[n]);
      dshape(n,2) = shape1dx(I[n])    * shape1dy(J[n])    * dshape1dz(K[n],0);
   }
}

LagrangeHexFiniteElement::~LagrangeHexFiniteElement ()
{
   delete fe1d;

   delete [] I;
   delete [] J;
   delete [] K;
}


RefinedLinear1DFiniteElement::RefinedLinear1DFiniteElement()
   : NodalFiniteElement(1, Geometry::SEGMENT, 3, 4)
{
   Nodes.IntPoint(0).x = 0.0;
   Nodes.IntPoint(1).x = 1.0;
   Nodes.IntPoint(2).x = 0.5;
}

void RefinedLinear1DFiniteElement::CalcShape(const IntegrationPoint &ip,
                                             Vector &shape) const
{
   real_t x = ip.x;

   if (x <= 0.5)
   {
      shape(0) = 1.0 - 2.0 * x;
      shape(1) = 0.0;
      shape(2) = 2.0 * x;
   }
   else
   {
      shape(0) = 0.0;
      shape(1) = 2.0 * x - 1.0;
      shape(2) = 2.0 - 2.0 * x;
   }
}

void RefinedLinear1DFiniteElement::CalcDShape(const IntegrationPoint &ip,
                                              DenseMatrix &dshape) const
{
   real_t x = ip.x;

   if (x <= 0.5)
   {
      dshape(0,0) = - 2.0;
      dshape(1,0) =   0.0;
      dshape(2,0) =   2.0;
   }
   else
   {
      dshape(0,0) =   0.0;
      dshape(1,0) =   2.0;
      dshape(2,0) = - 2.0;
   }
}

RefinedLinear2DFiniteElement::RefinedLinear2DFiniteElement()
   : NodalFiniteElement(2, Geometry::TRIANGLE, 6, 5)
{
   Nodes.IntPoint(0).x = 0.0;
   Nodes.IntPoint(0).y = 0.0;
   Nodes.IntPoint(1).x = 1.0;
   Nodes.IntPoint(1).y = 0.0;
   Nodes.IntPoint(2).x = 0.0;
   Nodes.IntPoint(2).y = 1.0;
   Nodes.IntPoint(3).x = 0.5;
   Nodes.IntPoint(3).y = 0.0;
   Nodes.IntPoint(4).x = 0.5;
   Nodes.IntPoint(4).y = 0.5;
   Nodes.IntPoint(5).x = 0.0;
   Nodes.IntPoint(5).y = 0.5;
}

void RefinedLinear2DFiniteElement::CalcShape(const IntegrationPoint &ip,
                                             Vector &shape) const
{
   int i;

   real_t L0, L1, L2;
   L0 = 2.0 * ( 1. - ip.x - ip.y );
   L1 = 2.0 * ( ip.x );
   L2 = 2.0 * ( ip.y );

   // The reference triangle is split in 4 triangles as follows:
   //
   // T0 - 0,3,5
   // T1 - 1,3,4
   // T2 - 2,4,5
   // T3 - 3,4,5

   for (i = 0; i < 6; i++)
   {
      shape(i) = 0.0;
   }

   if (L0 >= 1.0)   // T0
   {
      shape(0) = L0 - 1.0;
      shape(3) =       L1;
      shape(5) =       L2;
   }
   else if (L1 >= 1.0)   // T1
   {
      shape(3) =       L0;
      shape(1) = L1 - 1.0;
      shape(4) =       L2;
   }
   else if (L2 >= 1.0)   // T2
   {
      shape(5) =       L0;
      shape(4) =       L1;
      shape(2) = L2 - 1.0;
   }
   else   // T3
   {
      shape(3) = 1.0 - L2;
      shape(4) = 1.0 - L0;
      shape(5) = 1.0 - L1;
   }
}

void RefinedLinear2DFiniteElement::CalcDShape(const IntegrationPoint &ip,
                                              DenseMatrix &dshape) const
{
   int i,j;

   real_t L0, L1, L2;
   L0 = 2.0 * ( 1. - ip.x - ip.y );
   L1 = 2.0 * ( ip.x );
   L2 = 2.0 * ( ip.y );

   real_t DL0[2], DL1[2], DL2[2];
   DL0[0] = -2.0; DL0[1] = -2.0;
   DL1[0] =  2.0; DL1[1] =  0.0;
   DL2[0] =  0.0; DL2[1] =  2.0;

   for (i = 0; i < 6; i++)
      for (j = 0; j < 2; j++)
      {
         dshape(i,j) = 0.0;
      }

   if (L0 >= 1.0)   // T0
   {
      for (j = 0; j < 2; j++)
      {
         dshape(0,j) = DL0[j];
         dshape(3,j) = DL1[j];
         dshape(5,j) = DL2[j];
      }
   }
   else if (L1 >= 1.0)   // T1
   {
      for (j = 0; j < 2; j++)
      {
         dshape(3,j) = DL0[j];
         dshape(1,j) = DL1[j];
         dshape(4,j) = DL2[j];
      }
   }
   else if (L2 >= 1.0)   // T2
   {
      for (j = 0; j < 2; j++)
      {
         dshape(5,j) = DL0[j];
         dshape(4,j) = DL1[j];
         dshape(2,j) = DL2[j];
      }
   }
   else   // T3
   {
      for (j = 0; j < 2; j++)
      {
         dshape(3,j) = - DL2[j];
         dshape(4,j) = - DL0[j];
         dshape(5,j) = - DL1[j];
      }
   }
}

RefinedLinear3DFiniteElement::RefinedLinear3DFiniteElement()
   : NodalFiniteElement(3, Geometry::TETRAHEDRON, 10, 4)
{
   Nodes.IntPoint(0).x = 0.0;
   Nodes.IntPoint(0).y = 0.0;
   Nodes.IntPoint(0).z = 0.0;
   Nodes.IntPoint(1).x = 1.0;
   Nodes.IntPoint(1).y = 0.0;
   Nodes.IntPoint(1).z = 0.0;
   Nodes.IntPoint(2).x = 0.0;
   Nodes.IntPoint(2).y = 1.0;
   Nodes.IntPoint(2).z = 0.0;
   Nodes.IntPoint(3).x = 0.0;
   Nodes.IntPoint(3).y = 0.0;
   Nodes.IntPoint(3).z = 1.0;
   Nodes.IntPoint(4).x = 0.5;
   Nodes.IntPoint(4).y = 0.0;
   Nodes.IntPoint(4).z = 0.0;
   Nodes.IntPoint(5).x = 0.0;
   Nodes.IntPoint(5).y = 0.5;
   Nodes.IntPoint(5).z = 0.0;
   Nodes.IntPoint(6).x = 0.0;
   Nodes.IntPoint(6).y = 0.0;
   Nodes.IntPoint(6).z = 0.5;
   Nodes.IntPoint(7).x = 0.5;
   Nodes.IntPoint(7).y = 0.5;
   Nodes.IntPoint(7).z = 0.0;
   Nodes.IntPoint(8).x = 0.5;
   Nodes.IntPoint(8).y = 0.0;
   Nodes.IntPoint(8).z = 0.5;
   Nodes.IntPoint(9).x = 0.0;
   Nodes.IntPoint(9).y = 0.5;
   Nodes.IntPoint(9).z = 0.5;
}

void RefinedLinear3DFiniteElement::CalcShape(const IntegrationPoint &ip,
                                             Vector &shape) const
{
   int i;

   real_t L0, L1, L2, L3, L4, L5;
   L0 = 2.0 * ( 1. - ip.x - ip.y - ip.z );
   L1 = 2.0 * ( ip.x );
   L2 = 2.0 * ( ip.y );
   L3 = 2.0 * ( ip.z );
   L4 = 2.0 * ( ip.x + ip.y );
   L5 = 2.0 * ( ip.y + ip.z );

   // The reference tetrahedron is split in 8 tetrahedra as follows:
   //
   // T0 - 0,4,5,6
   // T1 - 1,4,7,8
   // T2 - 2,5,7,9
   // T3 - 3,6,8,9
   // T4 - 4,5,6,8
   // T5 - 4,5,7,8
   // T6 - 5,6,8,9
   // T7 - 5,7,8,9

   for (i = 0; i < 10; i++)
   {
      shape(i) = 0.0;
   }

   if (L0 >= 1.0)   // T0
   {
      shape(0) = L0 - 1.0;
      shape(4) =       L1;
      shape(5) =       L2;
      shape(6) =       L3;
   }
   else if (L1 >= 1.0)   // T1
   {
      shape(4) =       L0;
      shape(1) = L1 - 1.0;
      shape(7) =       L2;
      shape(8) =       L3;
   }
   else if (L2 >= 1.0)   // T2
   {
      shape(5) =       L0;
      shape(7) =       L1;
      shape(2) = L2 - 1.0;
      shape(9) =       L3;
   }
   else if (L3 >= 1.0)   // T3
   {
      shape(6) =       L0;
      shape(8) =       L1;
      shape(9) =       L2;
      shape(3) = L3 - 1.0;
   }
   else if ((L4 <= 1.0) && (L5 <= 1.0))   // T4
   {
      shape(4) = 1.0 - L5;
      shape(5) =       L2;
      shape(6) = 1.0 - L4;
      shape(8) = 1.0 - L0;
   }
   else if ((L4 >= 1.0) && (L5 <= 1.0))   // T5
   {
      shape(4) = 1.0 - L5;
      shape(5) = 1.0 - L1;
      shape(7) = L4 - 1.0;
      shape(8) =       L3;
   }
   else if ((L4 <= 1.0) && (L5 >= 1.0))   // T6
   {
      shape(5) = 1.0 - L3;
      shape(6) = 1.0 - L4;
      shape(8) =       L1;
      shape(9) = L5 - 1.0;
   }
   else if ((L4 >= 1.0) && (L5 >= 1.0))   // T7
   {
      shape(5) =       L0;
      shape(7) = L4 - 1.0;
      shape(8) = 1.0 - L2;
      shape(9) = L5 - 1.0;
   }
}

void RefinedLinear3DFiniteElement::CalcDShape(const IntegrationPoint &ip,
                                              DenseMatrix &dshape) const
{
   int i,j;

   real_t L0, L1, L2, L3, L4, L5;
   L0 = 2.0 * ( 1. - ip.x - ip.y - ip.z );
   L1 = 2.0 * ( ip.x );
   L2 = 2.0 * ( ip.y );
   L3 = 2.0 * ( ip.z );
   L4 = 2.0 * ( ip.x + ip.y );
   L5 = 2.0 * ( ip.y + ip.z );

   real_t DL0[3], DL1[3], DL2[3], DL3[3], DL4[3], DL5[3];
   DL0[0] = -2.0; DL0[1] = -2.0; DL0[2] = -2.0;
   DL1[0] =  2.0; DL1[1] =  0.0; DL1[2] =  0.0;
   DL2[0] =  0.0; DL2[1] =  2.0; DL2[2] =  0.0;
   DL3[0] =  0.0; DL3[1] =  0.0; DL3[2] =  2.0;
   DL4[0] =  2.0; DL4[1] =  2.0; DL4[2] =  0.0;
   DL5[0] =  0.0; DL5[1] =  2.0; DL5[2] =  2.0;

   for (i = 0; i < 10; i++)
      for (j = 0; j < 3; j++)
      {
         dshape(i,j) = 0.0;
      }

   if (L0 >= 1.0)   // T0
   {
      for (j = 0; j < 3; j++)
      {
         dshape(0,j) = DL0[j];
         dshape(4,j) = DL1[j];
         dshape(5,j) = DL2[j];
         dshape(6,j) = DL3[j];
      }
   }
   else if (L1 >= 1.0)   // T1
   {
      for (j = 0; j < 3; j++)
      {
         dshape(4,j) = DL0[j];
         dshape(1,j) = DL1[j];
         dshape(7,j) = DL2[j];
         dshape(8,j) = DL3[j];
      }
   }
   else if (L2 >= 1.0)   // T2
   {
      for (j = 0; j < 3; j++)
      {
         dshape(5,j) = DL0[j];
         dshape(7,j) = DL1[j];
         dshape(2,j) = DL2[j];
         dshape(9,j) = DL3[j];
      }
   }
   else if (L3 >= 1.0)   // T3
   {
      for (j = 0; j < 3; j++)
      {
         dshape(6,j) = DL0[j];
         dshape(8,j) = DL1[j];
         dshape(9,j) = DL2[j];
         dshape(3,j) = DL3[j];
      }
   }
   else if ((L4 <= 1.0) && (L5 <= 1.0))   // T4
   {
      for (j = 0; j < 3; j++)
      {
         dshape(4,j) = - DL5[j];
         dshape(5,j) =   DL2[j];
         dshape(6,j) = - DL4[j];
         dshape(8,j) = - DL0[j];
      }
   }
   else if ((L4 >= 1.0) && (L5 <= 1.0))   // T5
   {
      for (j = 0; j < 3; j++)
      {
         dshape(4,j) = - DL5[j];
         dshape(5,j) = - DL1[j];
         dshape(7,j) =   DL4[j];
         dshape(8,j) =   DL3[j];
      }
   }
   else if ((L4 <= 1.0) && (L5 >= 1.0))   // T6
   {
      for (j = 0; j < 3; j++)
      {
         dshape(5,j) = - DL3[j];
         dshape(6,j) = - DL4[j];
         dshape(8,j) =   DL1[j];
         dshape(9,j) =   DL5[j];
      }
   }
   else if ((L4 >= 1.0) && (L5 >= 1.0))   // T7
   {
      for (j = 0; j < 3; j++)
      {
         dshape(5,j) =   DL0[j];
         dshape(7,j) =   DL4[j];
         dshape(8,j) = - DL2[j];
         dshape(9,j) =   DL5[j];
      }
   }
}


RefinedBiLinear2DFiniteElement::RefinedBiLinear2DFiniteElement()
   : NodalFiniteElement(2, Geometry::SQUARE, 9, 1, FunctionSpace::rQk)
{
   Nodes.IntPoint(0).x = 0.0;
   Nodes.IntPoint(0).y = 0.0;
   Nodes.IntPoint(1).x = 1.0;
   Nodes.IntPoint(1).y = 0.0;
   Nodes.IntPoint(2).x = 1.0;
   Nodes.IntPoint(2).y = 1.0;
   Nodes.IntPoint(3).x = 0.0;
   Nodes.IntPoint(3).y = 1.0;
   Nodes.IntPoint(4).x = 0.5;
   Nodes.IntPoint(4).y = 0.0;
   Nodes.IntPoint(5).x = 1.0;
   Nodes.IntPoint(5).y = 0.5;
   Nodes.IntPoint(6).x = 0.5;
   Nodes.IntPoint(6).y = 1.0;
   Nodes.IntPoint(7).x = 0.0;
   Nodes.IntPoint(7).y = 0.5;
   Nodes.IntPoint(8).x = 0.5;
   Nodes.IntPoint(8).y = 0.5;
}

void RefinedBiLinear2DFiniteElement::CalcShape(const IntegrationPoint &ip,
                                               Vector &shape) const
{
   int i;
   real_t x = ip.x, y = ip.y;
   real_t Lx, Ly;
   Lx = 2.0 * ( 1. - x );
   Ly = 2.0 * ( 1. - y );

   // The reference square is split in 4 squares as follows:
   //
   // T0 - 0,4,7,8
   // T1 - 1,4,5,8
   // T2 - 2,5,6,8
   // T3 - 3,6,7,8

   for (i = 0; i < 9; i++)
   {
      shape(i) = 0.0;
   }

   if ((x <= 0.5) && (y <= 0.5))   // T0
   {
      shape(0) = (Lx - 1.0) * (Ly - 1.0);
      shape(4) = (2.0 - Lx) * (Ly - 1.0);
      shape(8) = (2.0 - Lx) * (2.0 - Ly);
      shape(7) = (Lx - 1.0) * (2.0 - Ly);
   }
   else if ((x >= 0.5) && (y <= 0.5))   // T1
   {
      shape(4) =        Lx  * (Ly - 1.0);
      shape(1) = (1.0 - Lx) * (Ly - 1.0);
      shape(5) = (1.0 - Lx) * (2.0 - Ly);
      shape(8) =        Lx  * (2.0 - Ly);
   }
   else if ((x >= 0.5) && (y >= 0.5))   // T2
   {
      shape(8) =        Lx  *        Ly ;
      shape(5) = (1.0 - Lx) *        Ly ;
      shape(2) = (1.0 - Lx) * (1.0 - Ly);
      shape(6) =        Lx  * (1.0 - Ly);
   }
   else if ((x <= 0.5) && (y >= 0.5))   // T3
   {
      shape(7) = (Lx - 1.0) *        Ly ;
      shape(8) = (2.0 - Lx) *        Ly ;
      shape(6) = (2.0 - Lx) * (1.0 - Ly);
      shape(3) = (Lx - 1.0) * (1.0 - Ly);
   }
}

void RefinedBiLinear2DFiniteElement::CalcDShape(const IntegrationPoint &ip,
                                                DenseMatrix &dshape) const
{
   int i,j;
   real_t x = ip.x, y = ip.y;
   real_t Lx, Ly;
   Lx = 2.0 * ( 1. - x );
   Ly = 2.0 * ( 1. - y );

   for (i = 0; i < 9; i++)
      for (j = 0; j < 2; j++)
      {
         dshape(i,j) = 0.0;
      }

   if ((x <= 0.5) && (y <= 0.5))   // T0
   {
      dshape(0,0) =  2.0 * (1.0 - Ly);
      dshape(0,1) =  2.0 * (1.0 - Lx);

      dshape(4,0) =  2.0 * (Ly - 1.0);
      dshape(4,1) = -2.0 * (2.0 - Lx);

      dshape(8,0) =  2.0 * (2.0 - Ly);
      dshape(8,1) =  2.0 * (2.0 - Lx);

      dshape(7,0) = -2.0 * (2.0 - Ly);
      dshape(7,0) =  2.0 * (Lx - 1.0);
   }
   else if ((x >= 0.5) && (y <= 0.5))   // T1
   {
      dshape(4,0) = -2.0 * (Ly - 1.0);
      dshape(4,1) = -2.0 * Lx;

      dshape(1,0) =  2.0 * (Ly - 1.0);
      dshape(1,1) = -2.0 * (1.0 - Lx);

      dshape(5,0) =  2.0 * (2.0 - Ly);
      dshape(5,1) =  2.0 * (1.0 - Lx);

      dshape(8,0) = -2.0 * (2.0 - Ly);
      dshape(8,1) =  2.0 * Lx;
   }
   else if ((x >= 0.5) && (y >= 0.5))   // T2
   {
      dshape(8,0) = -2.0 * Ly;
      dshape(8,1) = -2.0 * Lx;

      dshape(5,0) =  2.0 * Ly;
      dshape(5,1) = -2.0 * (1.0 - Lx);

      dshape(2,0) =  2.0 * (1.0 - Ly);
      dshape(2,1) =  2.0 * (1.0 - Lx);

      dshape(6,0) = -2.0 * (1.0 - Ly);
      dshape(6,1) =  2.0 * Lx;
   }
   else if ((x <= 0.5) && (y >= 0.5))   // T3
   {
      dshape(7,0) = -2.0 * Ly;
      dshape(7,1) = -2.0 * (Lx - 1.0);

      dshape(8,0) =  2.0 * Ly ;
      dshape(8,1) = -2.0 * (2.0 - Lx);

      dshape(6,0) = 2.0 * (1.0 - Ly);
      dshape(6,1) = 2.0 * (2.0 - Lx);

      dshape(3,0) = -2.0 * (1.0 - Ly);
      dshape(3,1) =  2.0 * (Lx - 1.0);
   }
}

RefinedTriLinear3DFiniteElement::RefinedTriLinear3DFiniteElement()
   : NodalFiniteElement(3, Geometry::CUBE, 27, 2, FunctionSpace::rQk)
{
   real_t I[27];
   real_t J[27];
   real_t K[27];
   // nodes
   I[ 0] = 0.0; J[ 0] = 0.0; K[ 0] = 0.0;
   I[ 1] = 1.0; J[ 1] = 0.0; K[ 1] = 0.0;
   I[ 2] = 1.0; J[ 2] = 1.0; K[ 2] = 0.0;
   I[ 3] = 0.0; J[ 3] = 1.0; K[ 3] = 0.0;
   I[ 4] = 0.0; J[ 4] = 0.0; K[ 4] = 1.0;
   I[ 5] = 1.0; J[ 5] = 0.0; K[ 5] = 1.0;
   I[ 6] = 1.0; J[ 6] = 1.0; K[ 6] = 1.0;
   I[ 7] = 0.0; J[ 7] = 1.0; K[ 7] = 1.0;
   // edges
   I[ 8] = 0.5; J[ 8] = 0.0; K[ 8] = 0.0;
   I[ 9] = 1.0; J[ 9] = 0.5; K[ 9] = 0.0;
   I[10] = 0.5; J[10] = 1.0; K[10] = 0.0;
   I[11] = 0.0; J[11] = 0.5; K[11] = 0.0;
   I[12] = 0.5; J[12] = 0.0; K[12] = 1.0;
   I[13] = 1.0; J[13] = 0.5; K[13] = 1.0;
   I[14] = 0.5; J[14] = 1.0; K[14] = 1.0;
   I[15] = 0.0; J[15] = 0.5; K[15] = 1.0;
   I[16] = 0.0; J[16] = 0.0; K[16] = 0.5;
   I[17] = 1.0; J[17] = 0.0; K[17] = 0.5;
   I[18] = 1.0; J[18] = 1.0; K[18] = 0.5;
   I[19] = 0.0; J[19] = 1.0; K[19] = 0.5;
   // faces
   I[20] = 0.5; J[20] = 0.5; K[20] = 0.0;
   I[21] = 0.5; J[21] = 0.0; K[21] = 0.5;
   I[22] = 1.0; J[22] = 0.5; K[22] = 0.5;
   I[23] = 0.5; J[23] = 1.0; K[23] = 0.5;
   I[24] = 0.0; J[24] = 0.5; K[24] = 0.5;
   I[25] = 0.5; J[25] = 0.5; K[25] = 1.0;
   // element
   I[26] = 0.5; J[26] = 0.5; K[26] = 0.5;

   for (int n = 0; n < 27; n++)
   {
      Nodes.IntPoint(n).x = I[n];
      Nodes.IntPoint(n).y = J[n];
      Nodes.IntPoint(n).z = K[n];
   }
}

void RefinedTriLinear3DFiniteElement::CalcShape(const IntegrationPoint &ip,
                                                Vector &shape) const
{
   int i, N[8];
   real_t Lx, Ly, Lz;
   real_t x = ip.x, y = ip.y, z = ip.z;

   for (i = 0; i < 27; i++)
   {
      shape(i) = 0.0;
   }

   if ((x <= 0.5) && (y <= 0.5) && (z <= 0.5))   // T0
   {
      Lx = 1.0 - 2.0 * x;
      Ly = 1.0 - 2.0 * y;
      Lz = 1.0 - 2.0 * z;

      N[0] =  0;
      N[1] =  8;
      N[2] = 20;
      N[3] = 11;
      N[4] = 16;
      N[5] = 21;
      N[6] = 26;
      N[7] = 24;
   }
   else if ((x >= 0.5) && (y <= 0.5) && (z <= 0.5))   // T1
   {
      Lx = 2.0 - 2.0 * x;
      Ly = 1.0 - 2.0 * y;
      Lz = 1.0 - 2.0 * z;

      N[0] =  8;
      N[1] =  1;
      N[2] =  9;
      N[3] = 20;
      N[4] = 21;
      N[5] = 17;
      N[6] = 22;
      N[7] = 26;
   }
   else if ((x <= 0.5) && (y >= 0.5) && (z <= 0.5))   // T2
   {
      Lx = 2.0 - 2.0 * x;
      Ly = 2.0 - 2.0 * y;
      Lz = 1.0 - 2.0 * z;

      N[0] = 20;
      N[1] =  9;
      N[2] =  2;
      N[3] = 10;
      N[4] = 26;
      N[5] = 22;
      N[6] = 18;
      N[7] = 23;
   }
   else if ((x >= 0.5) && (y >= 0.5) && (z <= 0.5))   // T3
   {
      Lx = 1.0 - 2.0 * x;
      Ly = 2.0 - 2.0 * y;
      Lz = 1.0 - 2.0 * z;

      N[0] = 11;
      N[1] = 20;
      N[2] = 10;
      N[3] =  3;
      N[4] = 24;
      N[5] = 26;
      N[6] = 23;
      N[7] = 19;
   }
   else if ((x <= 0.5) && (y <= 0.5) && (z >= 0.5))   // T4
   {
      Lx = 1.0 - 2.0 * x;
      Ly = 1.0 - 2.0 * y;
      Lz = 2.0 - 2.0 * z;

      N[0] = 16;
      N[1] = 21;
      N[2] = 26;
      N[3] = 24;
      N[4] =  4;
      N[5] = 12;
      N[6] = 25;
      N[7] = 15;
   }
   else if ((x >= 0.5) && (y <= 0.5) && (z >= 0.5))   // T5
   {
      Lx = 2.0 - 2.0 * x;
      Ly = 1.0 - 2.0 * y;
      Lz = 2.0 - 2.0 * z;

      N[0] = 21;
      N[1] = 17;
      N[2] = 22;
      N[3] = 26;
      N[4] = 12;
      N[5] =  5;
      N[6] = 13;
      N[7] = 25;
   }
   else if ((x <= 0.5) && (y >= 0.5) && (z >= 0.5))   // T6
   {
      Lx = 2.0 - 2.0 * x;
      Ly = 2.0 - 2.0 * y;
      Lz = 2.0 - 2.0 * z;

      N[0] = 26;
      N[1] = 22;
      N[2] = 18;
      N[3] = 23;
      N[4] = 25;
      N[5] = 13;
      N[6] =  6;
      N[7] = 14;
   }
   else   // T7
   {
      Lx = 1.0 - 2.0 * x;
      Ly = 2.0 - 2.0 * y;
      Lz = 2.0 - 2.0 * z;

      N[0] = 24;
      N[1] = 26;
      N[2] = 23;
      N[3] = 19;
      N[4] = 15;
      N[5] = 25;
      N[6] = 14;
      N[7] =  7;
   }

   shape(N[0]) = Lx       * Ly       * Lz;
   shape(N[1]) = (1 - Lx) * Ly       * Lz;
   shape(N[2]) = (1 - Lx) * (1 - Ly) * Lz;
   shape(N[3]) = Lx       * (1 - Ly) * Lz;
   shape(N[4]) = Lx       * Ly       * (1 - Lz);
   shape(N[5]) = (1 - Lx) * Ly       * (1 - Lz);
   shape(N[6]) = (1 - Lx) * (1 - Ly) * (1 - Lz);
   shape(N[7]) = Lx       * (1 - Ly) * (1 - Lz);
}

void RefinedTriLinear3DFiniteElement::CalcDShape(const IntegrationPoint &ip,
                                                 DenseMatrix &dshape) const
{
   int i, j, N[8];
   real_t Lx, Ly, Lz;
   real_t x = ip.x, y = ip.y, z = ip.z;

   for (i = 0; i < 27; i++)
      for (j = 0; j < 3; j++)
      {
         dshape(i,j) = 0.0;
      }

   if ((x <= 0.5) && (y <= 0.5) && (z <= 0.5))   // T0
   {
      Lx = 1.0 - 2.0 * x;
      Ly = 1.0 - 2.0 * y;
      Lz = 1.0 - 2.0 * z;

      N[0] =  0;
      N[1] =  8;
      N[2] = 20;
      N[3] = 11;
      N[4] = 16;
      N[5] = 21;
      N[6] = 26;
      N[7] = 24;
   }
   else if ((x >= 0.5) && (y <= 0.5) && (z <= 0.5))   // T1
   {
      Lx = 2.0 - 2.0 * x;
      Ly = 1.0 - 2.0 * y;
      Lz = 1.0 - 2.0 * z;

      N[0] =  8;
      N[1] =  1;
      N[2] =  9;
      N[3] = 20;
      N[4] = 21;
      N[5] = 17;
      N[6] = 22;
      N[7] = 26;
   }
   else if ((x <= 0.5) && (y >= 0.5) && (z <= 0.5))   // T2
   {
      Lx = 2.0 - 2.0 * x;
      Ly = 2.0 - 2.0 * y;
      Lz = 1.0 - 2.0 * z;

      N[0] = 20;
      N[1] =  9;
      N[2] =  2;
      N[3] = 10;
      N[4] = 26;
      N[5] = 22;
      N[6] = 18;
      N[7] = 23;
   }
   else if ((x >= 0.5) && (y >= 0.5) && (z <= 0.5))   // T3
   {
      Lx = 1.0 - 2.0 * x;
      Ly = 2.0 - 2.0 * y;
      Lz = 1.0 - 2.0 * z;

      N[0] = 11;
      N[1] = 20;
      N[2] = 10;
      N[3] =  3;
      N[4] = 24;
      N[5] = 26;
      N[6] = 23;
      N[7] = 19;
   }
   else if ((x <= 0.5) && (y <= 0.5) && (z >= 0.5))   // T4
   {
      Lx = 1.0 - 2.0 * x;
      Ly = 1.0 - 2.0 * y;
      Lz = 2.0 - 2.0 * z;

      N[0] = 16;
      N[1] = 21;
      N[2] = 26;
      N[3] = 24;
      N[4] =  4;
      N[5] = 12;
      N[6] = 25;
      N[7] = 15;
   }
   else if ((x >= 0.5) && (y <= 0.5) && (z >= 0.5))   // T5
   {
      Lx = 2.0 - 2.0 * x;
      Ly = 1.0 - 2.0 * y;
      Lz = 2.0 - 2.0 * z;

      N[0] = 21;
      N[1] = 17;
      N[2] = 22;
      N[3] = 26;
      N[4] = 12;
      N[5] =  5;
      N[6] = 13;
      N[7] = 25;
   }
   else if ((x <= 0.5) && (y >= 0.5) && (z >= 0.5))   // T6
   {
      Lx = 2.0 - 2.0 * x;
      Ly = 2.0 - 2.0 * y;
      Lz = 2.0 - 2.0 * z;

      N[0] = 26;
      N[1] = 22;
      N[2] = 18;
      N[3] = 23;
      N[4] = 25;
      N[5] = 13;
      N[6] =  6;
      N[7] = 14;
   }
   else   // T7
   {
      Lx = 1.0 - 2.0 * x;
      Ly = 2.0 - 2.0 * y;
      Lz = 2.0 - 2.0 * z;

      N[0] = 24;
      N[1] = 26;
      N[2] = 23;
      N[3] = 19;
      N[4] = 15;
      N[5] = 25;
      N[6] = 14;
      N[7] =  7;
   }

   dshape(N[0],0) = -2.0 * Ly       * Lz      ;
   dshape(N[0],1) = -2.0 * Lx       * Lz      ;
   dshape(N[0],2) = -2.0 * Lx       * Ly      ;

   dshape(N[1],0) =  2.0 * Ly       * Lz      ;
   dshape(N[1],1) = -2.0 * (1 - Lx) * Lz      ;
   dshape(N[1],2) = -2.0 * (1 - Lx) * Ly      ;

   dshape(N[2],0) =  2.0 * (1 - Ly) * Lz      ;
   dshape(N[2],1) =  2.0 * (1 - Lx) * Lz      ;
   dshape(N[2],2) = -2.0 * (1 - Lx) * (1 - Ly);

   dshape(N[3],0) = -2.0 * (1 - Ly) * Lz      ;
   dshape(N[3],1) =  2.0 * Lx       * Lz      ;
   dshape(N[3],2) = -2.0 * Lx       * (1 - Ly);

   dshape(N[4],0) = -2.0 * Ly       * (1 - Lz);
   dshape(N[4],1) = -2.0 * Lx       * (1 - Lz);
   dshape(N[4],2) =  2.0 * Lx       * Ly      ;

   dshape(N[5],0) =  2.0 * Ly       * (1 - Lz);
   dshape(N[5],1) = -2.0 * (1 - Lx) * (1 - Lz);
   dshape(N[5],2) =  2.0 * (1 - Lx) * Ly      ;

   dshape(N[6],0) =  2.0 * (1 - Ly) * (1 - Lz);
   dshape(N[6],1) =  2.0 * (1 - Lx) * (1 - Lz);
   dshape(N[6],2) =  2.0 * (1 - Lx) * (1 - Ly);

   dshape(N[7],0) = -2.0 * (1 - Ly) * (1 - Lz);
   dshape(N[7],1) =  2.0 * Lx       * (1 - Lz);
   dshape(N[7],2) =  2.0 * Lx       * (1 - Ly);
}


Nedelec1HexFiniteElement::Nedelec1HexFiniteElement()
   : VectorFiniteElement(3, Geometry::CUBE, 12, 1, H_CURL,
                         FunctionSpace::Qk)
{
   // not real nodes ...
   Nodes.IntPoint(0).x = 0.5;
   Nodes.IntPoint(0).y = 0.0;
   Nodes.IntPoint(0).z = 0.0;

   Nodes.IntPoint(1).x = 1.0;
   Nodes.IntPoint(1).y = 0.5;
   Nodes.IntPoint(1).z = 0.0;

   Nodes.IntPoint(2).x = 0.5;
   Nodes.IntPoint(2).y = 1.0;
   Nodes.IntPoint(2).z = 0.0;

   Nodes.IntPoint(3).x = 0.0;
   Nodes.IntPoint(3).y = 0.5;
   Nodes.IntPoint(3).z = 0.0;

   Nodes.IntPoint(4).x = 0.5;
   Nodes.IntPoint(4).y = 0.0;
   Nodes.IntPoint(4).z = 1.0;

   Nodes.IntPoint(5).x = 1.0;
   Nodes.IntPoint(5).y = 0.5;
   Nodes.IntPoint(5).z = 1.0;

   Nodes.IntPoint(6).x = 0.5;
   Nodes.IntPoint(6).y = 1.0;
   Nodes.IntPoint(6).z = 1.0;

   Nodes.IntPoint(7).x = 0.0;
   Nodes.IntPoint(7).y = 0.5;
   Nodes.IntPoint(7).z = 1.0;

   Nodes.IntPoint(8).x = 0.0;
   Nodes.IntPoint(8).y = 0.0;
   Nodes.IntPoint(8).z = 0.5;

   Nodes.IntPoint(9).x = 1.0;
   Nodes.IntPoint(9).y = 0.0;
   Nodes.IntPoint(9).z = 0.5;

   Nodes.IntPoint(10).x= 1.0;
   Nodes.IntPoint(10).y= 1.0;
   Nodes.IntPoint(10).z= 0.5;

   Nodes.IntPoint(11).x= 0.0;
   Nodes.IntPoint(11).y= 1.0;
   Nodes.IntPoint(11).z= 0.5;
}

void Nedelec1HexFiniteElement::CalcVShape(const IntegrationPoint &ip,
                                          DenseMatrix &shape) const
{
   real_t x = ip.x, y = ip.y, z = ip.z;

   shape(0,0) = (1. - y) * (1. - z);
   shape(0,1) = 0.;
   shape(0,2) = 0.;

   shape(2,0) = y * (1. - z);
   shape(2,1) = 0.;
   shape(2,2) = 0.;

   shape(4,0) = z * (1. - y);
   shape(4,1) = 0.;
   shape(4,2) = 0.;

   shape(6,0) = y * z;
   shape(6,1) = 0.;
   shape(6,2) = 0.;

   shape(1,0) = 0.;
   shape(1,1) = x * (1. - z);
   shape(1,2) = 0.;

   shape(3,0) = 0.;
   shape(3,1) = (1. - x) * (1. - z);
   shape(3,2) = 0.;

   shape(5,0) = 0.;
   shape(5,1) = x * z;
   shape(5,2) = 0.;

   shape(7,0) = 0.;
   shape(7,1) = (1. - x) * z;
   shape(7,2) = 0.;

   shape(8,0) = 0.;
   shape(8,1) = 0.;
   shape(8,2) = (1. - x) * (1. - y);

   shape(9,0) = 0.;
   shape(9,1) = 0.;
   shape(9,2) = x * (1. - y);

   shape(10,0) = 0.;
   shape(10,1) = 0.;
   shape(10,2) = x * y;

   shape(11,0) = 0.;
   shape(11,1) = 0.;
   shape(11,2) = y * (1. - x);

}

void Nedelec1HexFiniteElement::CalcCurlShape(const IntegrationPoint &ip,
                                             DenseMatrix &curl_shape)
const
{
   real_t x = ip.x, y = ip.y, z = ip.z;

   curl_shape(0,0) = 0.;
   curl_shape(0,1) = y - 1.;
   curl_shape(0,2) = 1. - z;

   curl_shape(2,0) = 0.;
   curl_shape(2,1) = -y;
   curl_shape(2,2) = z - 1.;

   curl_shape(4,0) = 0;
   curl_shape(4,1) = 1. - y;
   curl_shape(4,2) = z;

   curl_shape(6,0) = 0.;
   curl_shape(6,1) = y;
   curl_shape(6,2) = -z;

   curl_shape(1,0) = x;
   curl_shape(1,1) = 0.;
   curl_shape(1,2) = 1. - z;

   curl_shape(3,0) = 1. - x;
   curl_shape(3,1) = 0.;
   curl_shape(3,2) = z - 1.;

   curl_shape(5,0) = -x;
   curl_shape(5,1) = 0.;
   curl_shape(5,2) = z;

   curl_shape(7,0) = x - 1.;
   curl_shape(7,1) = 0.;
   curl_shape(7,2) = -z;

   curl_shape(8,0) = x - 1.;
   curl_shape(8,1) = 1. - y;
   curl_shape(8,2) = 0.;

   curl_shape(9,0) = -x;
   curl_shape(9,1) = y - 1.;
   curl_shape(9,2) = 0;

   curl_shape(10,0) = x;
   curl_shape(10,1) = -y;
   curl_shape(10,2) = 0.;

   curl_shape(11,0) = 1. - x;
   curl_shape(11,1) = y;
   curl_shape(11,2) = 0.;
}

const real_t Nedelec1HexFiniteElement::tk[12][3] =
{
   {1,0,0}, {0,1,0}, {1,0,0}, {0,1,0},
   {1,0,0}, {0,1,0}, {1,0,0}, {0,1,0},
   {0,0,1}, {0,0,1}, {0,0,1}, {0,0,1}
};

void Nedelec1HexFiniteElement::GetLocalInterpolation (
   ElementTransformation &Trans, DenseMatrix &I) const
{
   int k, j;
#ifdef MFEM_THREAD_SAFE
   DenseMatrix vshape(dof, dim);
#endif

#ifdef MFEM_DEBUG
   for (k = 0; k < dof; k++)
   {
      CalcVShape (Nodes.IntPoint(k), vshape);
      for (j = 0; j < dof; j++)
      {
         real_t d = ( vshape(j,0)*tk[k][0] + vshape(j,1)*tk[k][1] +
                      vshape(j,2)*tk[k][2] );
         if (j == k) { d -= 1.0; }
         if (fabs(d) > 1.0e-12)
         {
            mfem::err << "Nedelec1HexFiniteElement::GetLocalInterpolation (...)\n"
                      " k = " << k << ", j = " << j << ", d = " << d << endl;
            mfem_error();
         }
      }
   }
#endif

   IntegrationPoint ip;
   ip.x = ip.y = ip.z = 0.0;
   Trans.SetIntPoint (&ip);
   // Trans must be linear (more to have embedding?)
   const DenseMatrix &J = Trans.Jacobian();
   real_t vk[3];
   Vector xk (vk, 3);

   for (k = 0; k < dof; k++)
   {
      Trans.Transform (Nodes.IntPoint (k), xk);
      ip.x = vk[0]; ip.y = vk[1]; ip.z = vk[2];
      CalcVShape (ip, vshape);
      //  vk = J tk
      vk[0] = J(0,0)*tk[k][0]+J(0,1)*tk[k][1]+J(0,2)*tk[k][2];
      vk[1] = J(1,0)*tk[k][0]+J(1,1)*tk[k][1]+J(1,2)*tk[k][2];
      vk[2] = J(2,0)*tk[k][0]+J(2,1)*tk[k][1]+J(2,2)*tk[k][2];
      for (j = 0; j < dof; j++)
         if (fabs (I(k,j) = (vshape(j,0)*vk[0]+vshape(j,1)*vk[1]+
                             vshape(j,2)*vk[2])) < 1.0e-12)
         {
            I(k,j) = 0.0;
         }
   }
}

void Nedelec1HexFiniteElement::Project (
   VectorCoefficient &vc, ElementTransformation &Trans,
   Vector &dofs) const
{
   real_t vk[3];
   Vector xk (vk, 3);

   for (int k = 0; k < dof; k++)
   {
      Trans.SetIntPoint (&Nodes.IntPoint (k));
      const DenseMatrix &J = Trans.Jacobian();

      vc.Eval (xk, Trans, Nodes.IntPoint (k));
      //  xk^t J tk
      dofs(k) =
         vk[0] * ( J(0,0)*tk[k][0]+J(0,1)*tk[k][1]+J(0,2)*tk[k][2] ) +
         vk[1] * ( J(1,0)*tk[k][0]+J(1,1)*tk[k][1]+J(1,2)*tk[k][2] ) +
         vk[2] * ( J(2,0)*tk[k][0]+J(2,1)*tk[k][1]+J(2,2)*tk[k][2] );
   }
}

void Nedelec1HexFiniteElement::ProjectGrad(const FiniteElement &fe,
                                           ElementTransformation &Trans,
                                           DenseMatrix &grad) const
{
   DenseMatrix dshape(fe.GetDof(), 3);
   Vector grad_k(fe.GetDof());

   grad.SetSize(dof, fe.GetDof());
   for (int k = 0; k < dof; k++)
   {
      fe.CalcDShape(Nodes.IntPoint(k), dshape);
      dshape.Mult(tk[k], grad_k);
      for (int j = 0; j < grad_k.Size(); j++)
      {
         grad(k,j) = (fabs(grad_k(j)) < 1e-12) ? 0.0 : grad_k(j);
      }
   }
}


Nedelec1TetFiniteElement::Nedelec1TetFiniteElement()
   : VectorFiniteElement(3, Geometry::TETRAHEDRON, 6, 1, H_CURL)
{
   // not real nodes ...
   Nodes.IntPoint(0).x = 0.5;
   Nodes.IntPoint(0).y = 0.0;
   Nodes.IntPoint(0).z = 0.0;

   Nodes.IntPoint(1).x = 0.0;
   Nodes.IntPoint(1).y = 0.5;
   Nodes.IntPoint(1).z = 0.0;

   Nodes.IntPoint(2).x = 0.0;
   Nodes.IntPoint(2).y = 0.0;
   Nodes.IntPoint(2).z = 0.5;

   Nodes.IntPoint(3).x = 0.5;
   Nodes.IntPoint(3).y = 0.5;
   Nodes.IntPoint(3).z = 0.0;

   Nodes.IntPoint(4).x = 0.5;
   Nodes.IntPoint(4).y = 0.0;
   Nodes.IntPoint(4).z = 0.5;

   Nodes.IntPoint(5).x = 0.0;
   Nodes.IntPoint(5).y = 0.5;
   Nodes.IntPoint(5).z = 0.5;
}

void Nedelec1TetFiniteElement::CalcVShape(const IntegrationPoint &ip,
                                          DenseMatrix &shape) const
{
   real_t x = ip.x, y = ip.y, z = ip.z;

   shape(0,0) = 1. - y - z;
   shape(0,1) = x;
   shape(0,2) = x;

   shape(1,0) = y;
   shape(1,1) = 1. - x - z;
   shape(1,2) = y;

   shape(2,0) = z;
   shape(2,1) = z;
   shape(2,2) = 1. - x - y;

   shape(3,0) = -y;
   shape(3,1) = x;
   shape(3,2) = 0.;

   shape(4,0) = -z;
   shape(4,1) = 0.;
   shape(4,2) = x;

   shape(5,0) = 0.;
   shape(5,1) = -z;
   shape(5,2) = y;
}

void Nedelec1TetFiniteElement::CalcCurlShape(const IntegrationPoint &ip,
                                             DenseMatrix &curl_shape)
const
{
   curl_shape(0,0) =  0.;
   curl_shape(0,1) = -2.;
   curl_shape(0,2) =  2.;

   curl_shape(1,0) =  2.;
   curl_shape(1,1) =  0.;
   curl_shape(1,2) = -2.;

   curl_shape(2,0) = -2.;
   curl_shape(2,1) =  2.;
   curl_shape(2,2) =  0.;

   curl_shape(3,0) = 0.;
   curl_shape(3,1) = 0.;
   curl_shape(3,2) = 2.;

   curl_shape(4,0) =  0.;
   curl_shape(4,1) = -2.;
   curl_shape(4,2) =  0.;

   curl_shape(5,0) = 2.;
   curl_shape(5,1) = 0.;
   curl_shape(5,2) = 0.;
}

const real_t Nedelec1TetFiniteElement::tk[6][3] =
{{1,0,0}, {0,1,0}, {0,0,1}, {-1,1,0}, {-1,0,1}, {0,-1,1}};

void Nedelec1TetFiniteElement::GetLocalInterpolation (
   ElementTransformation &Trans, DenseMatrix &I) const
{
   int k, j;
#ifdef MFEM_THREAD_SAFE
   DenseMatrix vshape(dof, dim);
#endif

#ifdef MFEM_DEBUG
   for (k = 0; k < dof; k++)
   {
      CalcVShape (Nodes.IntPoint(k), vshape);
      for (j = 0; j < dof; j++)
      {
         real_t d = ( vshape(j,0)*tk[k][0] + vshape(j,1)*tk[k][1] +
                      vshape(j,2)*tk[k][2] );
         if (j == k) { d -= 1.0; }
         if (fabs(d) > 1.0e-12)
         {
            mfem::err << "Nedelec1TetFiniteElement::GetLocalInterpolation (...)\n"
                      " k = " << k << ", j = " << j << ", d = " << d << endl;
            mfem_error();
         }
      }
   }
#endif

   IntegrationPoint ip;
   ip.x = ip.y = ip.z = 0.0;
   Trans.SetIntPoint (&ip);
   // Trans must be linear
   const DenseMatrix &J = Trans.Jacobian();
   real_t vk[3];
   Vector xk (vk, 3);

   for (k = 0; k < dof; k++)
   {
      Trans.Transform (Nodes.IntPoint (k), xk);
      ip.x = vk[0]; ip.y = vk[1]; ip.z = vk[2];
      CalcVShape (ip, vshape);
      //  vk = J tk
      vk[0] = J(0,0)*tk[k][0]+J(0,1)*tk[k][1]+J(0,2)*tk[k][2];
      vk[1] = J(1,0)*tk[k][0]+J(1,1)*tk[k][1]+J(1,2)*tk[k][2];
      vk[2] = J(2,0)*tk[k][0]+J(2,1)*tk[k][1]+J(2,2)*tk[k][2];
      for (j = 0; j < dof; j++)
         if (fabs (I(k,j) = (vshape(j,0)*vk[0]+vshape(j,1)*vk[1]+
                             vshape(j,2)*vk[2])) < 1.0e-12)
         {
            I(k,j) = 0.0;
         }
   }
}

void Nedelec1TetFiniteElement::Project (
   VectorCoefficient &vc, ElementTransformation &Trans,
   Vector &dofs) const
{
   real_t vk[3];
   Vector xk (vk, 3);

   for (int k = 0; k < dof; k++)
   {
      Trans.SetIntPoint (&Nodes.IntPoint (k));
      const DenseMatrix &J = Trans.Jacobian();

      vc.Eval (xk, Trans, Nodes.IntPoint (k));
      //  xk^t J tk
      dofs(k) =
         vk[0] * ( J(0,0)*tk[k][0]+J(0,1)*tk[k][1]+J(0,2)*tk[k][2] ) +
         vk[1] * ( J(1,0)*tk[k][0]+J(1,1)*tk[k][1]+J(1,2)*tk[k][2] ) +
         vk[2] * ( J(2,0)*tk[k][0]+J(2,1)*tk[k][1]+J(2,2)*tk[k][2] );
   }
}

void Nedelec1TetFiniteElement::ProjectGrad(const FiniteElement &fe,
                                           ElementTransformation &Trans,
                                           DenseMatrix &grad) const
{
   DenseMatrix dshape(fe.GetDof(), 3);
   Vector grad_k(fe.GetDof());

   grad.SetSize(dof, fe.GetDof());
   for (int k = 0; k < dof; k++)
   {
      fe.CalcDShape(Nodes.IntPoint(k), dshape);
      dshape.Mult(tk[k], grad_k);
      for (int j = 0; j < grad_k.Size(); j++)
      {
         grad(k,j) = (fabs(grad_k(j)) < 1e-12) ? 0.0 : grad_k(j);
      }
   }
}


Nedelec1WdgFiniteElement::Nedelec1WdgFiniteElement()
   : VectorFiniteElement(3, Geometry::PRISM, 9, 1, H_CURL, FunctionSpace::Qk)
{
   // not real nodes ...
   Nodes.IntPoint(0).x = 0.5;
   Nodes.IntPoint(0).y = 0.0;
   Nodes.IntPoint(0).z = 0.0;

   Nodes.IntPoint(1).x = 0.5;
   Nodes.IntPoint(1).y = 0.5;
   Nodes.IntPoint(1).z = 0.0;

   Nodes.IntPoint(2).x = 0.0;
   Nodes.IntPoint(2).y = 0.5;
   Nodes.IntPoint(2).z = 0.0;

   Nodes.IntPoint(3).x = 0.5;
   Nodes.IntPoint(3).y = 0.0;
   Nodes.IntPoint(3).z = 1.0;

   Nodes.IntPoint(4).x = 0.5;
   Nodes.IntPoint(4).y = 0.5;
   Nodes.IntPoint(4).z = 1.0;

   Nodes.IntPoint(5).x = 0.0;
   Nodes.IntPoint(5).y = 0.5;
   Nodes.IntPoint(5).z = 1.0;

   Nodes.IntPoint(6).x = 0.0;
   Nodes.IntPoint(6).y = 0.0;
   Nodes.IntPoint(6).z = 0.5;

   Nodes.IntPoint(7).x = 1.0;
   Nodes.IntPoint(7).y = 0.0;
   Nodes.IntPoint(7).z = 0.5;

   Nodes.IntPoint(8).x = 0.0;
   Nodes.IntPoint(8).y = 1.0;
   Nodes.IntPoint(8).z = 0.5;
}

void Nedelec1WdgFiniteElement::CalcVShape(const IntegrationPoint &ip,
                                          DenseMatrix &shape) const
{
   real_t x = ip.x, y = ip.y, z = ip.z;

   shape(0,0) = (1. - y) * (1. - z);
   shape(0,1) = x * (1. - z);
   shape(0,2) = 0.;

   shape(1,0) = - y * (1. - z);
   shape(1,1) = x * (1. - z);
   shape(1,2) = 0.;

   shape(2,0) = - y * (1. - z);
   shape(2,1) = - (1. - x) * (1. - z);
   shape(2,2) = 0.;

   shape(3,0) = (1. - y) * z;
   shape(3,1) = x * z;
   shape(3,2) = 0.;

   shape(4,0) = - y * z;
   shape(4,1) = x * z;
   shape(4,2) = 0.;

   shape(5,0) = - y * z;
   shape(5,1) = - (1. - x) * z;
   shape(5,2) = 0.;

   shape(6,0) = 0.;
   shape(6,1) = 0.;
   shape(6,2) = 1. - x - y;

   shape(7,0) = 0.;
   shape(7,1) = 0.;
   shape(7,2) = x;

   shape(8,0) = 0.;
   shape(8,1) = 0.;
   shape(8,2) = y;
}

void Nedelec1WdgFiniteElement::CalcCurlShape(const IntegrationPoint &ip,
                                             DenseMatrix &curl_shape)
const
{
   real_t x = ip.x, y = ip.y, z2 = 2. * ip.z;

   curl_shape(0,0) =   x;
   curl_shape(0,1) = - 1. + y;
   curl_shape(0,2) =   2. - z2;

   curl_shape(1,0) =   x;
   curl_shape(1,1) =   y;
   curl_shape(1,2) =   2. - z2;

   curl_shape(2,0) = - 1. + x;
   curl_shape(2,1) =   y;
   curl_shape(2,2) =   2. - z2;

   curl_shape(3,0) = - x;
   curl_shape(3,1) =   1. - y;
   curl_shape(3,2) =   z2;

   curl_shape(4,0) = - x;
   curl_shape(4,1) = - y;
   curl_shape(4,2) =   z2;

   curl_shape(5,0) =   1. - x;
   curl_shape(5,1) = - y;
   curl_shape(5,2) =   z2;

   curl_shape(6,0) = - 1.;
   curl_shape(6,1) =   1.;
   curl_shape(6,2) =   0.;

   curl_shape(7,0) =   0.;
   curl_shape(7,1) = - 1.;
   curl_shape(7,2) =   0.;

   curl_shape(8,0) =   1.;
   curl_shape(8,1) =   0.;
   curl_shape(8,2) =   0.;
}

const real_t Nedelec1WdgFiniteElement::tk[9][3] =
{
   {1,0,0}, {-1,1,0}, {0,-1,0}, {1,0,0}, {-1,1,0}, {0,-1,0},
   {0,0,1}, {0,0,1}, {0,0,1}
};

void Nedelec1WdgFiniteElement::GetLocalInterpolation (
   ElementTransformation &Trans, DenseMatrix &I) const
{
   int k, j;
#ifdef MFEM_THREAD_SAFE
   DenseMatrix vshape(dof, dim);
#endif

#ifdef MFEM_DEBUG
   for (k = 0; k < dof; k++)
   {
      CalcVShape (Nodes.IntPoint(k), vshape);
      for (j = 0; j < dof; j++)
      {
         real_t d = ( vshape(j,0)*tk[k][0] + vshape(j,1)*tk[k][1] +
                      vshape(j,2)*tk[k][2] );
         if (j == k) { d -= 1.0; }
         if (fabs(d) > 1.0e-12)
         {
            mfem::err << "Nedelec1WdgFiniteElement::GetLocalInterpolation (...)\n"
                      " k = " << k << ", j = " << j << ", d = " << d << endl;
            mfem_error();
         }
      }
   }
#endif

   IntegrationPoint ip;
   ip.x = ip.y = ip.z = 0.0;
   Trans.SetIntPoint (&ip);
   // Trans must be linear
   const DenseMatrix &J = Trans.Jacobian();
   real_t vk[3];
   Vector xk (vk, 3);

   for (k = 0; k < dof; k++)
   {
      Trans.Transform (Nodes.IntPoint (k), xk);
      ip.x = vk[0]; ip.y = vk[1]; ip.z = vk[2];
      CalcVShape (ip, vshape);
      //  vk = J tk
      vk[0] = J(0,0)*tk[k][0]+J(0,1)*tk[k][1]+J(0,2)*tk[k][2];
      vk[1] = J(1,0)*tk[k][0]+J(1,1)*tk[k][1]+J(1,2)*tk[k][2];
      vk[2] = J(2,0)*tk[k][0]+J(2,1)*tk[k][1]+J(2,2)*tk[k][2];
      for (j = 0; j < dof; j++)
         if (fabs (I(k,j) = (vshape(j,0)*vk[0]+vshape(j,1)*vk[1]+
                             vshape(j,2)*vk[2])) < 1.0e-12)
         {
            I(k,j) = 0.0;
         }
   }
}

void Nedelec1WdgFiniteElement::Project (
   VectorCoefficient &vc, ElementTransformation &Trans,
   Vector &dofs) const
{
   real_t vk[3];
   Vector xk (vk, 3);

   for (int k = 0; k < dof; k++)
   {
      Trans.SetIntPoint (&Nodes.IntPoint (k));
      const DenseMatrix &J = Trans.Jacobian();

      vc.Eval (xk, Trans, Nodes.IntPoint (k));
      //  xk^t J tk
      dofs(k) =
         vk[0] * ( J(0,0)*tk[k][0]+J(0,1)*tk[k][1]+J(0,2)*tk[k][2] ) +
         vk[1] * ( J(1,0)*tk[k][0]+J(1,1)*tk[k][1]+J(1,2)*tk[k][2] ) +
         vk[2] * ( J(2,0)*tk[k][0]+J(2,1)*tk[k][1]+J(2,2)*tk[k][2] );
   }
}

void Nedelec1WdgFiniteElement::ProjectGrad(const FiniteElement &fe,
                                           ElementTransformation &Trans,
                                           DenseMatrix &grad) const
{
   DenseMatrix dshape(fe.GetDof(), 3);
   Vector grad_k(fe.GetDof());

   grad.SetSize(dof, fe.GetDof());
   for (int k = 0; k < dof; k++)
   {
      fe.CalcDShape(Nodes.IntPoint(k), dshape);
      dshape.Mult(tk[k], grad_k);
      for (int j = 0; j < grad_k.Size(); j++)
      {
         grad(k,j) = (fabs(grad_k(j)) < 1e-12) ? 0.0 : grad_k(j);
      }
   }
}


Nedelec1PyrFiniteElement::Nedelec1PyrFiniteElement()
   : VectorFiniteElement(3, Geometry::PYRAMID, 8, 1, H_CURL, FunctionSpace::Uk)
{
   // not real nodes ...
   Nodes.IntPoint(0).x = 0.5;
   Nodes.IntPoint(0).y = 0.0;
   Nodes.IntPoint(0).z = 0.0;

   Nodes.IntPoint(1).x = 1.0;
   Nodes.IntPoint(1).y = 0.5;
   Nodes.IntPoint(1).z = 0.0;

   Nodes.IntPoint(2).x = 0.5;
   Nodes.IntPoint(2).y = 1.0;
   Nodes.IntPoint(2).z = 0.0;

   Nodes.IntPoint(3).x = 0.0;
   Nodes.IntPoint(3).y = 0.5;
   Nodes.IntPoint(3).z = 0.0;

   Nodes.IntPoint(4).x = 0.0;
   Nodes.IntPoint(4).y = 0.0;
   Nodes.IntPoint(4).z = 0.5;

   Nodes.IntPoint(5).x = 0.5;
   Nodes.IntPoint(5).y = 0.0;
   Nodes.IntPoint(5).z = 0.5;

   Nodes.IntPoint(6).x = 0.5;
   Nodes.IntPoint(6).y = 0.5;
   Nodes.IntPoint(6).z = 0.5;

   Nodes.IntPoint(7).x = 0.0;
   Nodes.IntPoint(7).y = 0.5;
   Nodes.IntPoint(7).z = 0.5;
}

void Nedelec1PyrFiniteElement::CalcVShape(const IntegrationPoint &ip,
                                          DenseMatrix &shape) const
{
   real_t x = ip.x, y = ip.y, z = ip.z, z2 = 2. * ip.z;
   real_t ox = 1. - x - z, oy = 1. - y - z, oz = 1. - z;

   real_t tol = 1e-6;

   if (oz <= tol)
   {
      // We must return the limit of the basis functions as z->1.  In order to
      // remain inside the pyramid in this limit the x and y coordinates must
      // be approaching 0. Unfortunately we obtain different limits if we
      // approach (0,0,1) from different directions. The values provided below
      // are the limits as x->(1-z)/2 and y->(1-z)/2 i.e. along the line from
      // the center of the base of the pyramid towards the apex. The resulting
      // limiting basis function values are:
      shape(0,0) =   0.;
      shape(0,1) =   0.;
      shape(0,2) =   0.;

      shape(1,0) =   0.;
      shape(1,1) =   0.;
      shape(1,2) =   0.;

      shape(2,0) =   0.;
      shape(2,1) =   0.;
      shape(2,2) =   0.;

      shape(3,0) =   0.;
      shape(3,1) =   0.;
      shape(3,2) =   0.;

      shape(4,0) =   0.5;
      shape(4,1) =   0.5;
      shape(4,2) =   0.75;

      shape(5,0) = - 0.5;
      shape(5,1) =   0.5;
      shape(5,2) =   0.25;

      shape(6,0) = - 0.5;
      shape(6,1) = - 0.5;
      shape(6,2) = - 0.25;

      shape(7,0) =   0.5;
      shape(7,1) = - 0.5;
      shape(7,2) =   0.25;

      return;
   }

   real_t ozi = 1.0 / oz;

   shape(0,0) =   oy;
   shape(0,1) =   0.;
   shape(0,2) =   x * oy * ozi;

   shape(1,0) =   0.;
   shape(1,1) =   x;
   shape(1,2) =   x * y * ozi;

   shape(2,0) =   y;
   shape(2,1) =   0.;
   shape(2,2) =   x * y * ozi;

   shape(3,0) =   0.;
   shape(3,1) =   ox;
   shape(3,2) =   ox * y * ozi;

   shape(4,0) =   oy * z * ozi;
   shape(4,1) =   ox * z * ozi;
   shape(4,2) =   1. - x - y + x * y * (1. - z2) * ozi * ozi;

   shape(5,0) = - oy * z * ozi;
   shape(5,1) =   x * z * ozi;
   shape(5,2) =   x * (1. - y * (1. - z2) * ozi * ozi);

   shape(6,0) = - y * z * ozi;
   shape(6,1) = - x * z * ozi;
   shape(6,2) =   x * y * (1. - z2) * ozi * ozi;

   shape(7,0) =   y * z * ozi;
   shape(7,1) = - ox * z * ozi;
   shape(7,2) =   y * (1. - x * (1. - z2) * ozi * ozi);
}

void Nedelec1PyrFiniteElement::CalcCurlShape(const IntegrationPoint &ip,
                                             DenseMatrix &curl_shape)
const
{
   real_t x = ip.x, y = ip.y, z = ip.z, z2 = 2. * z;
   real_t ox = 1. - x - z, oy = 1. - y - z, oz = 1. - z;

   real_t tol = 1e-6;

   if (oz <= tol)
   {
      // We must return the limit of the basis function derivatives as z->1.
      // In order to remain inside the pyramid in this limit the x and y
      // coordinates must be approaching 0. The resulting limiting basis
      // function values are:
      curl_shape(0,0) = - 0.5;
      curl_shape(0,1) = - 1.5;
      curl_shape(0,2) =   1.;

      curl_shape(1,0) =   0.5;
      curl_shape(1,1) = - 0.5;
      curl_shape(1,2) =   1.;

      curl_shape(2,0) =   0.5;
      curl_shape(2,1) = - 0.5;
      curl_shape(2,2) = - 1.;

      curl_shape(3,0) =   1.5;
      curl_shape(3,1) =   0.5;
      curl_shape(3,2) = - 1.;

      curl_shape(4,0) = - 1.;
      curl_shape(4,1) =   1.;
      curl_shape(4,2) =   0.;

      curl_shape(5,0) = - 1.;
      curl_shape(5,1) = - 1.;
      curl_shape(5,2) =   0.;

      curl_shape(6,0) =   1.;
      curl_shape(6,1) = - 1.;
      curl_shape(6,2) =   0.;

      curl_shape(7,0) =   1.;
      curl_shape(7,1) =   1.;
      curl_shape(7,2) =   0.;

      return;
   }

   real_t ozi = 1. / oz;

   curl_shape(0,0) = - x * ozi;
   curl_shape(0,1) = - 2. + y * ozi;
   curl_shape(0,2) =   1.;

   curl_shape(1,0) =   x * ozi;
   curl_shape(1,1) = - y * ozi;
   curl_shape(1,2) =   1.;

   curl_shape(2,0) =   x * ozi;
   curl_shape(2,1) = - y * ozi;
   curl_shape(2,2) = - 1.;

   curl_shape(3,0) =   (2. - x  - z2) * ozi;
   curl_shape(3,1) =   y * ozi;
   curl_shape(3,2) = - 1.;

   curl_shape(4,0) = - 2. * ox * ozi;
   curl_shape(4,1) =   2. * oy * ozi;
   curl_shape(4,2) =   0.;

   curl_shape(5,0) = - 2. * x * ozi;
   curl_shape(5,1) = - 2. * oy * ozi;
   curl_shape(5,2) =   0.;

   curl_shape(6,0) =   2. * x * ozi;
   curl_shape(6,1) = - 2. * y * ozi;
   curl_shape(6,2) =   0.;

   curl_shape(7,0) =   2. * ox * ozi;
   curl_shape(7,1) =   2. * y * ozi;
   curl_shape(7,2) =   0.;
}

const real_t Nedelec1PyrFiniteElement::tk[8][3] =
{{1,0,0}, {0,1,0}, {1,0,0}, {0,1,0}, {0,0,1}, {-1,0,1}, {-1,-1,1}, {0,-1,1}};

void Nedelec1PyrFiniteElement::GetLocalInterpolation (
   ElementTransformation &Trans, DenseMatrix &I) const
{
   int k, j;
#ifdef MFEM_THREAD_SAFE
   DenseMatrix vshape(dof, dim);
#endif

#ifdef MFEM_DEBUG
   for (k = 0; k < dof; k++)
   {
      CalcVShape (Nodes.IntPoint(k), vshape);
      for (j = 0; j < dof; j++)
      {
         real_t d = ( vshape(j,0)*tk[k][0] + vshape(j,1)*tk[k][1] +
                      vshape(j,2)*tk[k][2] );
         if (j == k) { d -= 1.0; }
         if (fabs(d) > 1.0e-12)
         {
            mfem::err << "Nedelec1PyrFiniteElement::GetLocalInterpolation (...)\n"
                      " k = " << k << ", j = " << j << ", d = " << d << endl;
            mfem_error();
         }
      }
   }
#endif

   IntegrationPoint ip;
   ip.x = ip.y = ip.z = 0.0;
   Trans.SetIntPoint (&ip);
   // Trans must be linear
   const DenseMatrix &J = Trans.Jacobian();
   real_t vk[3];
   Vector xk (vk, 3);

   for (k = 0; k < dof; k++)
   {
      Trans.Transform (Nodes.IntPoint (k), xk);
      ip.x = vk[0]; ip.y = vk[1]; ip.z = vk[2];
      CalcVShape (ip, vshape);
      //  vk = J tk
      vk[0] = J(0,0)*tk[k][0]+J(0,1)*tk[k][1]+J(0,2)*tk[k][2];
      vk[1] = J(1,0)*tk[k][0]+J(1,1)*tk[k][1]+J(1,2)*tk[k][2];
      vk[2] = J(2,0)*tk[k][0]+J(2,1)*tk[k][1]+J(2,2)*tk[k][2];
      for (j = 0; j < dof; j++)
         if (fabs (I(k,j) = (vshape(j,0)*vk[0]+vshape(j,1)*vk[1]+
                             vshape(j,2)*vk[2])) < 1.0e-12)
         {
            I(k,j) = 0.0;
         }
   }
}

void Nedelec1PyrFiniteElement::Project (
   VectorCoefficient &vc, ElementTransformation &Trans,
   Vector &dofs) const
{
   real_t vk[3];
   Vector xk (vk, 3);

   for (int k = 0; k < dof; k++)
   {
      Trans.SetIntPoint (&Nodes.IntPoint (k));
      const DenseMatrix &J = Trans.Jacobian();

      vc.Eval (xk, Trans, Nodes.IntPoint (k));
      //  xk^t J tk
      dofs(k) =
         vk[0] * ( J(0,0)*tk[k][0]+J(0,1)*tk[k][1]+J(0,2)*tk[k][2] ) +
         vk[1] * ( J(1,0)*tk[k][0]+J(1,1)*tk[k][1]+J(1,2)*tk[k][2] ) +
         vk[2] * ( J(2,0)*tk[k][0]+J(2,1)*tk[k][1]+J(2,2)*tk[k][2] );
   }
}

void Nedelec1PyrFiniteElement::ProjectGrad(const FiniteElement &fe,
                                           ElementTransformation &Trans,
                                           DenseMatrix &grad) const
{
   DenseMatrix dshape(fe.GetDof(), 3);
   Vector grad_k(fe.GetDof());

   grad.SetSize(dof, fe.GetDof());
   for (int k = 0; k < dof; k++)
   {
      fe.CalcDShape(Nodes.IntPoint(k), dshape);
      dshape.Mult(tk[k], grad_k);
      for (int j = 0; j < grad_k.Size(); j++)
      {
         grad(k,j) = (fabs(grad_k(j)) < 1e-12) ? 0.0 : grad_k(j);
      }
   }
}


Nedelec2PyrFiniteElement::Nedelec2PyrFiniteElement()
   : VectorFiniteElement(3, Geometry::PYRAMID, 28, 2, H_CURL, FunctionSpace::Uk)
{
   const real_t *eop = poly1d.OpenPoints(2 - 1);
   const real_t  fop = 1. / 3.;

   // not real nodes ...
   Nodes.IntPoint(0).Set3(eop[0], 0., 0.);
   Nodes.IntPoint(1).Set3(eop[1], 0., 0.);

   Nodes.IntPoint(2).Set3(1.0, eop[0], 0.);
   Nodes.IntPoint(3).Set3(1.0, eop[1], 0.);

   Nodes.IntPoint(4).Set3(eop[0], 1.0, 0.);
   Nodes.IntPoint(5).Set3(eop[1], 1.0, 0.);

   Nodes.IntPoint(6).Set3(0., eop[0], 0.);
   Nodes.IntPoint(7).Set3(0., eop[1], 0.);

   Nodes.IntPoint(8).Set3(0., 0., eop[0]);
   Nodes.IntPoint(9).Set3(0., 0., eop[1]);

   Nodes.IntPoint(10).Set3(eop[1], 0., eop[0]);
   Nodes.IntPoint(11).Set3(eop[0], 0., eop[1]);

   Nodes.IntPoint(12).Set3(eop[1], eop[1], eop[0]);
   Nodes.IntPoint(13).Set3(eop[0], eop[0], eop[1]);

   Nodes.IntPoint(14).Set3(0., eop[1], eop[0]);
   Nodes.IntPoint(15).Set3(0., eop[0], eop[1]);

   Nodes.IntPoint(16).Set3(eop[0], 0.5, 0.);
   Nodes.IntPoint(17).Set3(eop[1], 0.5, 0.);

   Nodes.IntPoint(18).Set3(0.5, eop[0], 0.);
   Nodes.IntPoint(19).Set3(0.5, eop[1], 0.);

   Nodes.IntPoint(20).Set3(fop, 0., fop);
   Nodes.IntPoint(21).Set3(fop, 0., fop);

   Nodes.IntPoint(22).Set3(2.*fop, fop, fop);
   Nodes.IntPoint(23).Set3(2.*fop, fop, fop);

   Nodes.IntPoint(24).Set3(fop, 2.*fop, fop);
   Nodes.IntPoint(25).Set3(fop, 2.*fop, fop);

   Nodes.IntPoint(26).Set3(0., fop, fop);
   Nodes.IntPoint(27).Set3(0., fop, fop);

   {
      int n = 28;
      DenseMatrix I(n,n);
      DenseMatrix vecs(n,3);
      I = 0.0;

      for (int i=0; i<n; i++)
      {
         CalcVShape(Nodes.IntPoint(i), vecs);
         for (int j=0; j<n; j++)
         {
            I(j,i) = vecs(j,0)*tk[i][0]+vecs(j,1)*tk[i][1]+vecs(j,2)*tk[i][2];
         }
      }
   }
}

void Nedelec2PyrFiniteElement::CalcVShape(const IntegrationPoint &ip,
                                          DenseMatrix &shape) const
{
   shape = 0.0;

   const real_t one = 1.0;
   const real_t x = ip.x, y = ip.y, z = ip.z;
   const real_t ox = one - x - z, oy = one - y - z, oz = one - z;
   const real_t sq3 = sqrt(3.0);
   const real_t tol = 1e-6;

   if (oz <= tol)
   {
      // We must return the limit of the basis functions as z->1.  In order to
      // remain inside the pyramid in this limit the x and y coordinates must
      // be approaching 0. The resulting limiting basis function values are:
      shape(0,0) =   0.;
      shape(0,1) =   0.;
      shape(0,2) =   0.;

      shape(1,0) =   0.;
      shape(1,1) =   0.;
      shape(1,2) =   0.;

      shape(2,0) =   0.;
      shape(2,1) =   0.;
      shape(2,2) =   0.;

      shape(3,0) =   0.;
      shape(3,1) =   0.;
      shape(3,2) =   0.;

      shape(4,0) =   0.;
      shape(4,1) =   0.;
      shape(4,2) =   0.;

      shape(5,0) =   0.;
      shape(5,1) =   0.;
      shape(5,2) =   0.;

      shape(6,0) =   0.;
      shape(6,1) =   0.;
      shape(6,2) =   0.;

      shape(7,0) =   0.;
      shape(7,1) =   0.;
      shape(7,2) =   0.;

      return;
   }

   const real_t ozi = one / oz;

   const real_t me0120[3] = {oy, 0., x * oy * ozi};
   const real_t me1120[3] = {(x - ox) * oy, 0., (x - ox) * x * oy * ozi};

   const real_t me0121[3] = {y, 0., x * y * ozi};
   const real_t me1121[3] = {(x - ox) * y, 0., (x - ox) * x * y * ozi};

   const real_t me0210[3] = {0., ox, ox * y * ozi};
   const real_t me1210[3] = {0., ox * (y - oy), ox * y * (y - oy) * ozi};

   const real_t me0211[3] = {0., x, x * y * ozi};
   const real_t me1211[3] = {0., x * (y - oy), x * y * (y - oy) * ozi};

   const real_t te01[3] = {oy * z * ozi, ox * z * ozi,
                           (ox * oy + (x * oy + ox * y) * z) * ozi * ozi
                          };
   const real_t te11[3] = {oy * z * (z * oz - ox * oy) * ozi * ozi,
                           ox * z * (z * oz - ox * oy) * ozi * ozi,
                           (ox * oy + z * (x * oy + ox * y)) *
                           (z * oz - ox * oy) * ozi * ozi * ozi
                          };

   const real_t te02[3] = {-oy * z * ozi, x * z * ozi,
                           x * (y * z + oy * oz) * ozi * ozi
                          };
   const real_t te12[3] = {oy * z * (x * oy - z * oz) * ozi * ozi,
                           -x * z * (x * oy - z * oz) * ozi * ozi,
                           -x * (y * z + oy * oz) * (x * oy - z * oz)
                           * ozi * ozi * ozi
                          };

   const real_t te03[3] = {-y * z * ozi, -x * z * ozi,
                           x * y * (one - 2_r * z) * ozi * ozi
                          };
   const real_t te13[3] = {y * z * (x * y - z * oz) * ozi * ozi,
                           x * z * (x * y - z * oz) * ozi * ozi,
                           -x * y * (one - 2_r * z) * (x * y - z * oz)
                           * ozi * ozi * ozi
                          };

   const real_t te04[3] = {y * z * ozi, -ox * z * ozi,
                           y * (x * z + ox * oz) * ozi * ozi
                          };
   const real_t te14[3] = {-y * z * (ox * y - z * oz) * ozi * ozi,
                           ox * z * (ox * y - z * oz) * ozi * ozi,
                           -y * (x * z + ox * oz) * (ox * y - z * oz)
                           * ozi * ozi * ozi
                          };

   const real_t qI02[3] = {-y * oy * ozi, 0., -x * y * oy * ozi * ozi};
   const real_t qI12[3] = {-(x - ox) * y * oy * ozi * ozi, 0.,
                           -(x - ox) * x * y * oy * ozi * ozi * ozi
                          };

   const real_t qII02[3] = {0., -x * ox * ozi, -x * y * ox * ozi * ozi};
   const real_t qII12[3] = {0., -x * ox * (y - oy) * ozi * ozi,
                            -x * ox * y * (y - oy) * ozi * ozi * ozi
                           };

   const real_t tI120[3] = {oy * z, 0., x * oy * z * ozi};
   const real_t tI121[3] = {y * z, 0., x * y * z * ozi};
   const real_t tI210[3] = {0., ox * z, ox * y * z * ozi};
   const real_t tI211[3] = {0., x * z, x * y * z * ozi};

   const real_t tII120[3] = {-ox * oy * z * ozi, 0., x * ox * oy * ozi};
   const real_t tII121[3] = {-ox * y * z * ozi, 0., x * ox * y * ozi};
   const real_t tII210[3] = {0., -ox * oy * z * ozi, ox * y * oy * ozi};
   const real_t tII211[3] = {0., -x * oy * z * ozi, x * y * oy * ozi};

   // Edge 0,1
   for (int d=0; d<3; d++)
   {
      shape(0,d) = 0.5 * me0120[d] + qI02[d]
                   - sq3 * (0.5 * me1120[d] + qI12[d]) - 1.5 * tI120[d];
   }
   for (int d=0; d<3; d++)
   {
      shape(1,d) = 0.5 * me0120[d] + qI02[d]
                   + sq3 * (0.5 * me1120[d] + qI12[d]) - 1.5 * tI120[d];
   }

   // Edge 1,2
   for (int d=0; d<3; d++)
   {
      shape(2,d) = 0.5 * me0211[d] + qII02[d]
                   - sq3 * (0.5 * me1211[d] + qII12[d]) - 1.5 * tI211[d];
   }
   for (int d=0; d<3; d++)
   {
      shape(3,d) = 0.5 * me0211[d] + qII02[d]
                   + sq3 * (0.5 * me1211[d] + qII12[d]) - 1.5 * tI211[d];
   }

   // Edge 3,2
   for (int d=0; d<3; d++)
   {
      shape(4,d) = 0.5 * me0121[d] + qI02[d]
                   - sq3 * (0.5 * me1121[d] + qI12[d]) - 1.5 * tI121[d];
   }
   for (int d=0; d<3; d++)
   {
      shape(5,d) = 0.5 * me0121[d] + qI02[d]
                   + sq3 * (0.5 * me1121[d] + qI12[d]) - 1.5 * tI121[d];
   }

   // Edge 0,3
   for (int d=0; d<3; d++)
   {
      shape(6,d) = 0.5 * me0210[d] + qII02[d]
                   - sq3 * (0.5 * me1210[d] + qII12[d]) - 1.5 * tI210[d];
   }
   for (int d=0; d<3; d++)
   {
      shape(7,d) = 0.5 * me0210[d] + qII02[d]
                   + sq3 * (0.5 * me1210[d] + qII12[d]) - 1.5 * tI210[d];
   }

   // Edge 0,4
   for (int d=0; d<3; d++)
   {
      shape(8,d) = 0.5 * te01[d] - sq3 * 0.5 * te11[d]
                   - 1.5 * (tI120[d] + tII120[d] + tI210[d] + tII210[d]);
   }
   for (int d=0; d<3; d++)
   {
      shape(9,d) = 0.5 * te01[d] + sq3 * 0.5 * te11[d]
                   - 1.5 * (tI120[d] + tII120[d] + tI210[d] + tII210[d]);
   }

   // Edge 1,4
   for (int d=0; d<3; d++)
   {
      shape(10,d) = 0.5 * te02[d] - sq3 * 0.5 * te12[d]
                    - 1.5 * (tII120[d] + tI211[d] + tII211[d]);
   }
   for (int d=0; d<3; d++)
   {
      shape(11,d) = 0.5 * te02[d] + sq3 * 0.5 * te12[d]
                    - 1.5 * (tII120[d] + tI211[d] + tII211[d]);
   }

   // Edge 2,4
   for (int d=0; d<3; d++)
   {
      shape(12,d) = 0.5 * te03[d] - sq3 * 0.5 * te13[d]
                    - 1.5 * (tII211[d] + tII121[d]);
   }
   for (int d=0; d<3; d++)
   {
      shape(13,d) = 0.5 * te03[d] + sq3 * 0.5 * te13[d]
                    - 1.5 * (tII211[d] + tII121[d]);
   }

   // Edge 3,4
   for (int d=0; d<3; d++)
   {
      shape(14,d) = 0.5 * te04[d] - sq3 * 0.5 * te14[d]
                    - 1.5 * (tI121[d] + tII121[d] + tII210[d]);
   }
   for (int d=0; d<3; d++)
   {
      shape(15,d) = 0.5 * te04[d] + sq3 * 0.5 * te14[d]
                    - 1.5 * (tI121[d] + tII121[d] + tII210[d]);
   }

   // Quadrilateral face
   for (int d=0; d<3; d++)
   {
      shape(16,d) = -2. * qI02[d] + 2. * sq3 * qI12[d];
   }
   for (int d=0; d<3; d++)
   {
      shape(17,d) = -2. * qI02[d] - 2. * sq3 * qI12[d];
   }

   for (int d=0; d<3; d++)
   {
      shape(18,d) = 2. * qII02[d] - 2. * sq3 * qII12[d];
   }
   for (int d=0; d<3; d++)
   {
      shape(19,d) = 2. * qII02[d] + 2. * sq3 * qII12[d];
   }

   // Triangular face 0,1,4
   for (int d=0; d<3; d++)
   {
      shape(20,d) = 3. * tI120[d] - 3. * tII120[d];
   }
   for (int d=0; d<3; d++)
   {
      shape(21,d) = 3. * tI120[d] + 6. * tII120[d];
   }

   // Triangular face 1,2,4
   for (int d=0; d<3; d++)
   {
      shape(22,d) = 3. * tI211[d] - 3. * tII211[d];
   }
   for (int d=0; d<3; d++)
   {
      shape(23,d) = 3. * tI211[d] + 6. * tII211[d];
   }

   // Triangular face 2,3,4
   for (int d=0; d<3; d++)
   {
      shape(24,d) = -6. * tI121[d] - 3. * tII121[d];
   }
   for (int d=0; d<3; d++)
   {
      shape(25,d) = 3. * tI121[d] + 6. * tII121[d];
   }

   // Triangular face 3,0,4
   for (int d=0; d<3; d++)
   {
      shape(26,d) = -6. * tI210[d] - 3. * tII210[d];
   }
   for (int d=0; d<3; d++)
   {
      shape(27,d) = 3. * tI210[d] + 6. * tII210[d];
   }
}

void Nedelec2PyrFiniteElement::CalcCurlShape(const IntegrationPoint &ip,
                                             DenseMatrix &curl_shape)
const
{
   const real_t one = 1.0;
   const real_t x = ip.x, y = ip.y, z = ip.z, z2 = 2. * z;
   const real_t ox = one - x - z, oy = one - y - z, oz = one - z;

   const real_t tol = 1e-6;

   if (oz <= tol)
   {
      // We must return the limit of the basis function derivatives as z->1.
      // In order to remain inside the pyramid in this limit the x and y
      // coordinates must be approaching 0. The resulting limiting basis
      // function values are:
      curl_shape(0,0) =   0.;
      curl_shape(0,1) = - 2.;
      curl_shape(0,2) =   1.;

      curl_shape(1,0) =   0.;
      curl_shape(1,1) =   0.;
      curl_shape(1,2) =   1.;

      curl_shape(2,0) =   0.;
      curl_shape(2,1) =   0.;
      curl_shape(2,2) = - 1.;

      curl_shape(3,0) =   2.;
      curl_shape(3,1) =   0.;
      curl_shape(3,2) = - 1.;

      curl_shape(4,0) = - 2.;
      curl_shape(4,1) =   2.;
      curl_shape(4,2) =   0.;

      curl_shape(5,0) =   0.;
      curl_shape(5,1) = - 2.;
      curl_shape(5,2) =   0.;

      curl_shape(6,0) =   0.;
      curl_shape(6,1) =   0.;
      curl_shape(6,2) =   0.;

      curl_shape(7,0) =   2.;
      curl_shape(7,1) =   0.;
      curl_shape(7,2) =   0.;

      return;
   }

   real_t ozi = one / oz;

   curl_shape(0,0) = - x * ozi;
   curl_shape(0,1) = - 2. + y * ozi;
   curl_shape(0,2) =   1.;

   curl_shape(1,0) =   x * ozi;
   curl_shape(1,1) = - y * ozi;
   curl_shape(1,2) =   1.;

   curl_shape(2,0) =   x * ozi;
   curl_shape(2,1) = - y * ozi;
   curl_shape(2,2) = - 1.;

   curl_shape(3,0) =   (2. - x  - z2) * ozi;
   curl_shape(3,1) =   y * ozi;
   curl_shape(3,2) = - 1.;

   curl_shape(4,0) = - 2. * ox * ozi;
   curl_shape(4,1) =   2. * oy * ozi;
   curl_shape(4,2) =   0.;

   curl_shape(5,0) = - 2. * x * ozi;
   curl_shape(5,1) = - 2. * oy * ozi;
   curl_shape(5,2) =   0.;

   curl_shape(6,0) =   2. * x * ozi;
   curl_shape(6,1) = - 2. * y * ozi;
   curl_shape(6,2) =   0.;

   curl_shape(7,0) =   2. * ox * ozi;
   curl_shape(7,1) =   2. * y * ozi;
   curl_shape(7,2) =   0.;
}

const real_t Nedelec2PyrFiniteElement::tk[28][3] =
{
   {1,0,0}, {1,0,0}, {0,1,0}, {0,1,0},
   {1,0,0}, {1,0,0}, {0,1,0}, {0,1,0},
   {0,0,1}, {0,0,1}, {-1,0,1}, {-1,0,1},
   {-1,-1,1}, {-1,-1,1}, {0,-1,1}, {0,-1,1},
   {1,0,0}, {1,0,0}, {0,-1,0}, {0,-1,0},
   {1,0,0}, {0,0,1}, {0,1,0}, {-1,0,1},
   {-1,0,0}, {-1,-1,1}, {0,-1,0}, {0,-1,1}
};

void Nedelec2PyrFiniteElement::GetLocalInterpolation (
   ElementTransformation &Trans, DenseMatrix &I) const
{
   int k, j;
#ifdef MFEM_THREAD_SAFE
   DenseMatrix vshape(dof, dim);
#endif

#ifdef MFEM_DEBUG
   for (k = 0; k < dof; k++)
   {
      CalcVShape (Nodes.IntPoint(k), vshape);
      for (j = 0; j < dof; j++)
      {
         real_t d = ( vshape(j,0)*tk[k][0] + vshape(j,1)*tk[k][1] +
                      vshape(j,2)*tk[k][2] );
         if (j == k) { d -= 1.0; }
         if (fabs(d) > 1.0e-12)
         {
            mfem::err << "Nedelec1PyrFiniteElement::GetLocalInterpolation (...)\n"
                      " k = " << k << ", j = " << j << ", d = " << d << endl;
            mfem_error();
         }
      }
   }
#endif

   IntegrationPoint ip;
   ip.x = ip.y = ip.z = 0.0;
   Trans.SetIntPoint (&ip);
   // Trans must be linear
   const DenseMatrix &J = Trans.Jacobian();
   real_t vk[3];
   Vector xk (vk, 3);

   for (k = 0; k < dof; k++)
   {
      Trans.Transform (Nodes.IntPoint (k), xk);
      ip.x = vk[0]; ip.y = vk[1]; ip.z = vk[2];
      CalcVShape (ip, vshape);
      //  vk = J tk
      vk[0] = J(0,0)*tk[k][0]+J(0,1)*tk[k][1]+J(0,2)*tk[k][2];
      vk[1] = J(1,0)*tk[k][0]+J(1,1)*tk[k][1]+J(1,2)*tk[k][2];
      vk[2] = J(2,0)*tk[k][0]+J(2,1)*tk[k][1]+J(2,2)*tk[k][2];
      for (j = 0; j < dof; j++)
         if (fabs (I(k,j) = (vshape(j,0)*vk[0]+vshape(j,1)*vk[1]+
                             vshape(j,2)*vk[2])) < 1.0e-12)
         {
            I(k,j) = 0.0;
         }
   }
}

void Nedelec2PyrFiniteElement::Project (
   VectorCoefficient &vc, ElementTransformation &Trans,
   Vector &dofs) const
{
   real_t vk[3];
   Vector xk (vk, 3);

   for (int k = 0; k < dof; k++)
   {
      Trans.SetIntPoint (&Nodes.IntPoint (k));
      const DenseMatrix &J = Trans.Jacobian();

      vc.Eval (xk, Trans, Nodes.IntPoint (k));
      //  xk^t J tk
      dofs(k) =
         vk[0] * ( J(0,0)*tk[k][0]+J(0,1)*tk[k][1]+J(0,2)*tk[k][2] ) +
         vk[1] * ( J(1,0)*tk[k][0]+J(1,1)*tk[k][1]+J(1,2)*tk[k][2] ) +
         vk[2] * ( J(2,0)*tk[k][0]+J(2,1)*tk[k][1]+J(2,2)*tk[k][2] );
   }
}

void Nedelec2PyrFiniteElement::ProjectGrad(const FiniteElement &fe,
                                           ElementTransformation &Trans,
                                           DenseMatrix &grad) const
{
   DenseMatrix dshape(fe.GetDof(), 3);
   Vector grad_k(fe.GetDof());

   grad.SetSize(dof, fe.GetDof());
   for (int k = 0; k < dof; k++)
   {
      fe.CalcDShape(Nodes.IntPoint(k), dshape);
      dshape.Mult(tk[k], grad_k);
      for (int j = 0; j < grad_k.Size(); j++)
      {
         grad(k,j) = (fabs(grad_k(j)) < 1e-12) ? 0.0 : grad_k(j);
      }
   }
}


RT0HexFiniteElement::RT0HexFiniteElement()
   : VectorFiniteElement(3, Geometry::CUBE, 6, 1, H_DIV, FunctionSpace::Qk)
{
   // not real nodes ...
   // z = 0, y = 0, x = 1, y = 1, x = 0, z = 1
   Nodes.IntPoint(0).x = 0.5;
   Nodes.IntPoint(0).y = 0.5;
   Nodes.IntPoint(0).z = 0.0;

   Nodes.IntPoint(1).x = 0.5;
   Nodes.IntPoint(1).y = 0.0;
   Nodes.IntPoint(1).z = 0.5;

   Nodes.IntPoint(2).x = 1.0;
   Nodes.IntPoint(2).y = 0.5;
   Nodes.IntPoint(2).z = 0.5;

   Nodes.IntPoint(3).x = 0.5;
   Nodes.IntPoint(3).y = 1.0;
   Nodes.IntPoint(3).z = 0.5;

   Nodes.IntPoint(4).x = 0.0;
   Nodes.IntPoint(4).y = 0.5;
   Nodes.IntPoint(4).z = 0.5;

   Nodes.IntPoint(5).x = 0.5;
   Nodes.IntPoint(5).y = 0.5;
   Nodes.IntPoint(5).z = 1.0;
}

void RT0HexFiniteElement::CalcVShape(const IntegrationPoint &ip,
                                     DenseMatrix &shape) const
{
   real_t x = ip.x, y = ip.y, z = ip.z;
   // z = 0
   shape(0,0) = 0.;
   shape(0,1) = 0.;
   shape(0,2) = z - 1.;
   // y = 0
   shape(1,0) = 0.;
   shape(1,1) = y - 1.;
   shape(1,2) = 0.;
   // x = 1
   shape(2,0) = x;
   shape(2,1) = 0.;
   shape(2,2) = 0.;
   // y = 1
   shape(3,0) = 0.;
   shape(3,1) = y;
   shape(3,2) = 0.;
   // x = 0
   shape(4,0) = x - 1.;
   shape(4,1) = 0.;
   shape(4,2) = 0.;
   // z = 1
   shape(5,0) = 0.;
   shape(5,1) = 0.;
   shape(5,2) = z;
}

void RT0HexFiniteElement::CalcDivShape(const IntegrationPoint &ip,
                                       Vector &divshape) const
{
   divshape(0) = 1.;
   divshape(1) = 1.;
   divshape(2) = 1.;
   divshape(3) = 1.;
   divshape(4) = 1.;
   divshape(5) = 1.;
}

const real_t RT0HexFiniteElement::nk[6][3] =
{{0,0,-1}, {0,-1,0}, {1,0,0}, {0,1,0}, {-1,0,0}, {0,0,1}};

void RT0HexFiniteElement::GetLocalInterpolation (
   ElementTransformation &Trans, DenseMatrix &I) const
{
   int k, j;
#ifdef MFEM_THREAD_SAFE
   DenseMatrix vshape(dof, dim);
#endif

#ifdef MFEM_DEBUG
   for (k = 0; k < 6; k++)
   {
      CalcVShape (Nodes.IntPoint(k), vshape);
      for (j = 0; j < 6; j++)
      {
         real_t d = ( vshape(j,0)*nk[k][0] + vshape(j,1)*nk[k][1] +
                      vshape(j,2)*nk[k][2] );
         if (j == k) { d -= 1.0; }
         if (fabs(d) > 1.0e-12)
         {
            mfem::err << "RT0HexFiniteElement::GetLocalInterpolation (...)\n"
                      " k = " << k << ", j = " << j << ", d = " << d << endl;
            mfem_error();
         }
      }
   }
#endif

   IntegrationPoint ip;
   ip.x = ip.y = ip.z = 0.0;
   Trans.SetIntPoint (&ip);
   // Trans must be linear
   // set Jinv = |J| J^{-t} = adj(J)^t
   const DenseMatrix &Jinv = Trans.TransposeAdjugateJacobian();

   real_t vk[3];
   Vector xk (vk, 3);

   for (k = 0; k < 6; k++)
   {
      Trans.Transform (Nodes.IntPoint (k), xk);
      ip.x = vk[0]; ip.y = vk[1]; ip.z = vk[2];
      CalcVShape (ip, vshape);
      //  vk = |J| J^{-t} nk
      vk[0] = Jinv(0,0)*nk[k][0]+Jinv(0,1)*nk[k][1]+Jinv(0,2)*nk[k][2];
      vk[1] = Jinv(1,0)*nk[k][0]+Jinv(1,1)*nk[k][1]+Jinv(1,2)*nk[k][2];
      vk[2] = Jinv(2,0)*nk[k][0]+Jinv(2,1)*nk[k][1]+Jinv(2,2)*nk[k][2];
      for (j = 0; j < 6; j++)
         if (fabs (I(k,j) = (vshape(j,0)*vk[0]+vshape(j,1)*vk[1]+
                             vshape(j,2)*vk[2])) < 1.0e-12)
         {
            I(k,j) = 0.0;
         }
   }
}

void RT0HexFiniteElement::Project (
   VectorCoefficient &vc, ElementTransformation &Trans,
   Vector &dofs) const
{
   real_t vk[3];
   Vector xk (vk, 3);

   for (int k = 0; k < 6; k++)
   {
      Trans.SetIntPoint (&Nodes.IntPoint (k));
      // set Jinv = |J| J^{-t} = adj(J)^t
      const DenseMatrix &Jinv = Trans.TransposeAdjugateJacobian();

      vc.Eval (xk, Trans, Nodes.IntPoint (k));
      //  xk^t |J| J^{-t} nk
      dofs(k) =
         vk[0] * ( Jinv(0,0)*nk[k][0]+Jinv(0,1)*nk[k][1]+Jinv(0,2)*nk[k][2] ) +
         vk[1] * ( Jinv(1,0)*nk[k][0]+Jinv(1,1)*nk[k][1]+Jinv(1,2)*nk[k][2] ) +
         vk[2] * ( Jinv(2,0)*nk[k][0]+Jinv(2,1)*nk[k][1]+Jinv(2,2)*nk[k][2] );
   }
}

RT1HexFiniteElement::RT1HexFiniteElement()
   : VectorFiniteElement(3, Geometry::CUBE, 36, 2, H_DIV,
                         FunctionSpace::Qk)
{
   // z = 0
   Nodes.IntPoint(2).x  = 1./3.;
   Nodes.IntPoint(2).y  = 1./3.;
   Nodes.IntPoint(2).z  = 0.0;
   Nodes.IntPoint(3).x  = 2./3.;
   Nodes.IntPoint(3).y  = 1./3.;
   Nodes.IntPoint(3).z  = 0.0;
   Nodes.IntPoint(0).x  = 1./3.;
   Nodes.IntPoint(0).y  = 2./3.;
   Nodes.IntPoint(0).z  = 0.0;
   Nodes.IntPoint(1).x  = 2./3.;
   Nodes.IntPoint(1).y  = 2./3.;
   Nodes.IntPoint(1).z  = 0.0;
   // y = 0
   Nodes.IntPoint(4).x  = 1./3.;
   Nodes.IntPoint(4).y  = 0.0;
   Nodes.IntPoint(4).z  = 1./3.;
   Nodes.IntPoint(5).x  = 2./3.;
   Nodes.IntPoint(5).y  = 0.0;
   Nodes.IntPoint(5).z  = 1./3.;
   Nodes.IntPoint(6).x  = 1./3.;
   Nodes.IntPoint(6).y  = 0.0;
   Nodes.IntPoint(6).z  = 2./3.;
   Nodes.IntPoint(7).x  = 2./3.;
   Nodes.IntPoint(7).y  = 0.0;
   Nodes.IntPoint(7).z  = 2./3.;
   // x = 1
   Nodes.IntPoint(8).x  = 1.0;
   Nodes.IntPoint(8).y  = 1./3.;
   Nodes.IntPoint(8).z  = 1./3.;
   Nodes.IntPoint(9).x  = 1.0;
   Nodes.IntPoint(9).y  = 2./3.;
   Nodes.IntPoint(9).z  = 1./3.;
   Nodes.IntPoint(10).x = 1.0;
   Nodes.IntPoint(10).y = 1./3.;
   Nodes.IntPoint(10).z = 2./3.;
   Nodes.IntPoint(11).x = 1.0;
   Nodes.IntPoint(11).y = 2./3.;
   Nodes.IntPoint(11).z = 2./3.;
   // y = 1
   Nodes.IntPoint(13).x = 1./3.;
   Nodes.IntPoint(13).y = 1.0;
   Nodes.IntPoint(13).z = 1./3.;
   Nodes.IntPoint(12).x = 2./3.;
   Nodes.IntPoint(12).y = 1.0;
   Nodes.IntPoint(12).z = 1./3.;
   Nodes.IntPoint(15).x = 1./3.;
   Nodes.IntPoint(15).y = 1.0;
   Nodes.IntPoint(15).z = 2./3.;
   Nodes.IntPoint(14).x = 2./3.;
   Nodes.IntPoint(14).y = 1.0;
   Nodes.IntPoint(14).z = 2./3.;
   // x = 0
   Nodes.IntPoint(17).x = 0.0;
   Nodes.IntPoint(17).y = 1./3.;
   Nodes.IntPoint(17).z = 1./3.;
   Nodes.IntPoint(16).x = 0.0;
   Nodes.IntPoint(16).y = 2./3.;
   Nodes.IntPoint(16).z = 1./3.;
   Nodes.IntPoint(19).x = 0.0;
   Nodes.IntPoint(19).y = 1./3.;
   Nodes.IntPoint(19).z = 2./3.;
   Nodes.IntPoint(18).x = 0.0;
   Nodes.IntPoint(18).y = 2./3.;
   Nodes.IntPoint(18).z = 2./3.;
   // z = 1
   Nodes.IntPoint(20).x = 1./3.;
   Nodes.IntPoint(20).y = 1./3.;
   Nodes.IntPoint(20).z = 1.0;
   Nodes.IntPoint(21).x = 2./3.;
   Nodes.IntPoint(21).y = 1./3.;
   Nodes.IntPoint(21).z = 1.0;
   Nodes.IntPoint(22).x = 1./3.;
   Nodes.IntPoint(22).y = 2./3.;
   Nodes.IntPoint(22).z = 1.0;
   Nodes.IntPoint(23).x = 2./3.;
   Nodes.IntPoint(23).y = 2./3.;
   Nodes.IntPoint(23).z = 1.0;
   // x = 0.5 (interior)
   Nodes.IntPoint(24).x = 0.5;
   Nodes.IntPoint(24).y = 1./3.;
   Nodes.IntPoint(24).z = 1./3.;
   Nodes.IntPoint(25).x = 0.5;
   Nodes.IntPoint(25).y = 1./3.;
   Nodes.IntPoint(25).z = 2./3.;
   Nodes.IntPoint(26).x = 0.5;
   Nodes.IntPoint(26).y = 2./3.;
   Nodes.IntPoint(26).z = 1./3.;
   Nodes.IntPoint(27).x = 0.5;
   Nodes.IntPoint(27).y = 2./3.;
   Nodes.IntPoint(27).z = 2./3.;
   // y = 0.5 (interior)
   Nodes.IntPoint(28).x = 1./3.;
   Nodes.IntPoint(28).y = 0.5;
   Nodes.IntPoint(28).z = 1./3.;
   Nodes.IntPoint(29).x = 1./3.;
   Nodes.IntPoint(29).y = 0.5;
   Nodes.IntPoint(29).z = 2./3.;
   Nodes.IntPoint(30).x = 2./3.;
   Nodes.IntPoint(30).y = 0.5;
   Nodes.IntPoint(30).z = 1./3.;
   Nodes.IntPoint(31).x = 2./3.;
   Nodes.IntPoint(31).y = 0.5;
   Nodes.IntPoint(31).z = 2./3.;
   // z = 0.5 (interior)
   Nodes.IntPoint(32).x = 1./3.;
   Nodes.IntPoint(32).y = 1./3.;
   Nodes.IntPoint(32).z = 0.5;
   Nodes.IntPoint(33).x = 1./3.;
   Nodes.IntPoint(33).y = 2./3.;
   Nodes.IntPoint(33).z = 0.5;
   Nodes.IntPoint(34).x = 2./3.;
   Nodes.IntPoint(34).y = 1./3.;
   Nodes.IntPoint(34).z = 0.5;
   Nodes.IntPoint(35).x = 2./3.;
   Nodes.IntPoint(35).y = 2./3.;
   Nodes.IntPoint(35).z = 0.5;
}

void RT1HexFiniteElement::CalcVShape(const IntegrationPoint &ip,
                                     DenseMatrix &shape) const
{
   real_t x = ip.x, y = ip.y, z = ip.z;
   // z = 0
   shape(2,0)  = 0.;
   shape(2,1)  = 0.;
   shape(2,2)  = -(1. - 3.*z + 2.*z*z)*( 2. - 3.*x)*( 2. - 3.*y);
   shape(3,0)  = 0.;
   shape(3,1)  = 0.;
   shape(3,2)  = -(1. - 3.*z + 2.*z*z)*(-1. + 3.*x)*( 2. - 3.*y);
   shape(0,0)  = 0.;
   shape(0,1)  = 0.;
   shape(0,2)  = -(1. - 3.*z + 2.*z*z)*( 2. - 3.*x)*(-1. + 3.*y);
   shape(1,0)  = 0.;
   shape(1,1)  = 0.;
   shape(1,2)  = -(1. - 3.*z + 2.*z*z)*(-1. + 3.*x)*(-1. + 3.*y);
   // y = 0
   shape(4,0)  = 0.;
   shape(4,1)  = -(1. - 3.*y + 2.*y*y)*( 2. - 3.*x)*( 2. - 3.*z);
   shape(4,2)  = 0.;
   shape(5,0)  = 0.;
   shape(5,1)  = -(1. - 3.*y + 2.*y*y)*(-1. + 3.*x)*( 2. - 3.*z);
   shape(5,2)  = 0.;
   shape(6,0)  = 0.;
   shape(6,1)  = -(1. - 3.*y + 2.*y*y)*( 2. - 3.*x)*(-1. + 3.*z);
   shape(6,2)  = 0.;
   shape(7,0)  = 0.;
   shape(7,1)  = -(1. - 3.*y + 2.*y*y)*(-1. + 3.*x)*(-1. + 3.*z);
   shape(7,2)  = 0.;
   // x = 1
   shape(8,0)  = (-x + 2.*x*x)*( 2. - 3.*y)*( 2. - 3.*z);
   shape(8,1)  = 0.;
   shape(8,2)  = 0.;
   shape(9,0)  = (-x + 2.*x*x)*(-1. + 3.*y)*( 2. - 3.*z);
   shape(9,1)  = 0.;
   shape(9,2)  = 0.;
   shape(10,0) = (-x + 2.*x*x)*( 2. - 3.*y)*(-1. + 3.*z);
   shape(10,1) = 0.;
   shape(10,2) = 0.;
   shape(11,0) = (-x + 2.*x*x)*(-1. + 3.*y)*(-1. + 3.*z);
   shape(11,1) = 0.;
   shape(11,2) = 0.;
   // y = 1
   shape(13,0) = 0.;
   shape(13,1) = (-y + 2.*y*y)*( 2. - 3.*x)*( 2. - 3.*z);
   shape(13,2) = 0.;
   shape(12,0) = 0.;
   shape(12,1) = (-y + 2.*y*y)*(-1. + 3.*x)*( 2. - 3.*z);
   shape(12,2) = 0.;
   shape(15,0) = 0.;
   shape(15,1) = (-y + 2.*y*y)*( 2. - 3.*x)*(-1. + 3.*z);
   shape(15,2) = 0.;
   shape(14,0) = 0.;
   shape(14,1) = (-y + 2.*y*y)*(-1. + 3.*x)*(-1. + 3.*z);
   shape(14,2) = 0.;
   // x = 0
   shape(17,0) = -(1. - 3.*x + 2.*x*x)*( 2. - 3.*y)*( 2. - 3.*z);
   shape(17,1) = 0.;
   shape(17,2) = 0.;
   shape(16,0) = -(1. - 3.*x + 2.*x*x)*(-1. + 3.*y)*( 2. - 3.*z);
   shape(16,1) = 0.;
   shape(16,2) = 0.;
   shape(19,0) = -(1. - 3.*x + 2.*x*x)*( 2. - 3.*y)*(-1. + 3.*z);
   shape(19,1) = 0.;
   shape(19,2) = 0.;
   shape(18,0) = -(1. - 3.*x + 2.*x*x)*(-1. + 3.*y)*(-1. + 3.*z);
   shape(18,1) = 0.;
   shape(18,2) = 0.;
   // z = 1
   shape(20,0) = 0.;
   shape(20,1) = 0.;
   shape(20,2) = (-z + 2.*z*z)*( 2. - 3.*x)*( 2. - 3.*y);
   shape(21,0) = 0.;
   shape(21,1) = 0.;
   shape(21,2) = (-z + 2.*z*z)*(-1. + 3.*x)*( 2. - 3.*y);
   shape(22,0) = 0.;
   shape(22,1) = 0.;
   shape(22,2) = (-z + 2.*z*z)*( 2. - 3.*x)*(-1. + 3.*y);
   shape(23,0) = 0.;
   shape(23,1) = 0.;
   shape(23,2) = (-z + 2.*z*z)*(-1. + 3.*x)*(-1. + 3.*y);
   // x = 0.5 (interior)
   shape(24,0) = (4.*x - 4.*x*x)*( 2. - 3.*y)*( 2. - 3.*z);
   shape(24,1) = 0.;
   shape(24,2) = 0.;
   shape(25,0) = (4.*x - 4.*x*x)*( 2. - 3.*y)*(-1. + 3.*z);
   shape(25,1) = 0.;
   shape(25,2) = 0.;
   shape(26,0) = (4.*x - 4.*x*x)*(-1. + 3.*y)*( 2. - 3.*z);
   shape(26,1) = 0.;
   shape(26,2) = 0.;
   shape(27,0) = (4.*x - 4.*x*x)*(-1. + 3.*y)*(-1. + 3.*z);
   shape(27,1) = 0.;
   shape(27,2) = 0.;
   // y = 0.5 (interior)
   shape(28,0) = 0.;
   shape(28,1) = (4.*y - 4.*y*y)*( 2. - 3.*x)*( 2. - 3.*z);
   shape(28,2) = 0.;
   shape(29,0) = 0.;
   shape(29,1) = (4.*y - 4.*y*y)*( 2. - 3.*x)*(-1. + 3.*z);
   shape(29,2) = 0.;
   shape(30,0) = 0.;
   shape(30,1) = (4.*y - 4.*y*y)*(-1. + 3.*x)*( 2. - 3.*z);
   shape(30,2) = 0.;
   shape(31,0) = 0.;
   shape(31,1) = (4.*y - 4.*y*y)*(-1. + 3.*x)*(-1. + 3.*z);
   shape(31,2) = 0.;
   // z = 0.5 (interior)
   shape(32,0) = 0.;
   shape(32,1) = 0.;
   shape(32,2) = (4.*z - 4.*z*z)*( 2. - 3.*x)*( 2. - 3.*y);
   shape(33,0) = 0.;
   shape(33,1) = 0.;
   shape(33,2) = (4.*z - 4.*z*z)*( 2. - 3.*x)*(-1. + 3.*y);
   shape(34,0) = 0.;
   shape(34,1) = 0.;
   shape(34,2) = (4.*z - 4.*z*z)*(-1. + 3.*x)*( 2. - 3.*y);
   shape(35,0) = 0.;
   shape(35,1) = 0.;
   shape(35,2) = (4.*z - 4.*z*z)*(-1. + 3.*x)*(-1. + 3.*y);
}

void RT1HexFiniteElement::CalcDivShape(const IntegrationPoint &ip,
                                       Vector &divshape) const
{
   real_t x = ip.x, y = ip.y, z = ip.z;
   // z = 0
   divshape(2)  = -(-3. + 4.*z)*( 2. - 3.*x)*( 2. - 3.*y);
   divshape(3)  = -(-3. + 4.*z)*(-1. + 3.*x)*( 2. - 3.*y);
   divshape(0)  = -(-3. + 4.*z)*( 2. - 3.*x)*(-1. + 3.*y);
   divshape(1)  = -(-3. + 4.*z)*(-1. + 3.*x)*(-1. + 3.*y);
   // y = 0
   divshape(4)  = -(-3. + 4.*y)*( 2. - 3.*x)*( 2. - 3.*z);
   divshape(5)  = -(-3. + 4.*y)*(-1. + 3.*x)*( 2. - 3.*z);
   divshape(6)  = -(-3. + 4.*y)*( 2. - 3.*x)*(-1. + 3.*z);
   divshape(7)  = -(-3. + 4.*y)*(-1. + 3.*x)*(-1. + 3.*z);
   // x = 1
   divshape(8)  = (-1. + 4.*x)*( 2. - 3.*y)*( 2. - 3.*z);
   divshape(9)  = (-1. + 4.*x)*(-1. + 3.*y)*( 2. - 3.*z);
   divshape(10) = (-1. + 4.*x)*( 2. - 3.*y)*(-1. + 3.*z);
   divshape(11) = (-1. + 4.*x)*(-1. + 3.*y)*(-1. + 3.*z);
   // y = 1
   divshape(13) = (-1. + 4.*y)*( 2. - 3.*x)*( 2. - 3.*z);
   divshape(12) = (-1. + 4.*y)*(-1. + 3.*x)*( 2. - 3.*z);
   divshape(15) = (-1. + 4.*y)*( 2. - 3.*x)*(-1. + 3.*z);
   divshape(14) = (-1. + 4.*y)*(-1. + 3.*x)*(-1. + 3.*z);
   // x = 0
   divshape(17) = -(-3. + 4.*x)*( 2. - 3.*y)*( 2. - 3.*z);
   divshape(16) = -(-3. + 4.*x)*(-1. + 3.*y)*( 2. - 3.*z);
   divshape(19) = -(-3. + 4.*x)*( 2. - 3.*y)*(-1. + 3.*z);
   divshape(18) = -(-3. + 4.*x)*(-1. + 3.*y)*(-1. + 3.*z);
   // z = 1
   divshape(20) = (-1. + 4.*z)*( 2. - 3.*x)*( 2. - 3.*y);
   divshape(21) = (-1. + 4.*z)*(-1. + 3.*x)*( 2. - 3.*y);
   divshape(22) = (-1. + 4.*z)*( 2. - 3.*x)*(-1. + 3.*y);
   divshape(23) = (-1. + 4.*z)*(-1. + 3.*x)*(-1. + 3.*y);
   // x = 0.5 (interior)
   divshape(24) = ( 4. - 8.*x)*( 2. - 3.*y)*( 2. - 3.*z);
   divshape(25) = ( 4. - 8.*x)*( 2. - 3.*y)*(-1. + 3.*z);
   divshape(26) = ( 4. - 8.*x)*(-1. + 3.*y)*( 2. - 3.*z);
   divshape(27) = ( 4. - 8.*x)*(-1. + 3.*y)*(-1. + 3.*z);
   // y = 0.5 (interior)
   divshape(28) = ( 4. - 8.*y)*( 2. - 3.*x)*( 2. - 3.*z);
   divshape(29) = ( 4. - 8.*y)*( 2. - 3.*x)*(-1. + 3.*z);
   divshape(30) = ( 4. - 8.*y)*(-1. + 3.*x)*( 2. - 3.*z);
   divshape(31) = ( 4. - 8.*y)*(-1. + 3.*x)*(-1. + 3.*z);
   // z = 0.5 (interior)
   divshape(32) = ( 4. - 8.*z)*( 2. - 3.*x)*( 2. - 3.*y);
   divshape(33) = ( 4. - 8.*z)*( 2. - 3.*x)*(-1. + 3.*y);
   divshape(34) = ( 4. - 8.*z)*(-1. + 3.*x)*( 2. - 3.*y);
   divshape(35) = ( 4. - 8.*z)*(-1. + 3.*x)*(-1. + 3.*y);
}

const real_t RT1HexFiniteElement::nk[36][3] =
{
   {0, 0,-1}, {0, 0,-1}, {0, 0,-1}, {0, 0,-1},
   {0,-1, 0}, {0,-1, 0}, {0,-1, 0}, {0,-1, 0},
   {1, 0, 0}, {1, 0, 0}, {1, 0, 0}, {1, 0, 0},
   {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0},
   {-1,0, 0}, {-1,0, 0}, {-1,0, 0}, {-1,0, 0},
   {0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1},
   {1, 0, 0}, {1, 0, 0}, {1, 0, 0}, {1, 0, 0},
   {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0},
   {0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}
};

void RT1HexFiniteElement::GetLocalInterpolation (
   ElementTransformation &Trans, DenseMatrix &I) const
{
   int k, j;
#ifdef MFEM_THREAD_SAFE
   DenseMatrix vshape(dof, dim);
#endif

#ifdef MFEM_DEBUG
   for (k = 0; k < 36; k++)
   {
      CalcVShape (Nodes.IntPoint(k), vshape);
      for (j = 0; j < 36; j++)
      {
         real_t d = ( vshape(j,0)*nk[k][0] + vshape(j,1)*nk[k][1] +
                      vshape(j,2)*nk[k][2] );
         if (j == k) { d -= 1.0; }
         if (fabs(d) > 1.0e-12)
         {
            mfem::err << "RT0HexFiniteElement::GetLocalInterpolation (...)\n"
                      " k = " << k << ", j = " << j << ", d = " << d << endl;
            mfem_error();
         }
      }
   }
#endif

   IntegrationPoint ip;
   ip.x = ip.y = ip.z = 0.0;
   Trans.SetIntPoint (&ip);
   // Trans must be linear
   // set Jinv = |J| J^{-t} = adj(J)^t
   const DenseMatrix &Jinv = Trans.TransposeAdjugateJacobian();

   real_t vk[3];
   Vector xk (vk, 3);

   for (k = 0; k < 36; k++)
   {
      Trans.Transform (Nodes.IntPoint (k), xk);
      ip.x = vk[0]; ip.y = vk[1]; ip.z = vk[2];
      CalcVShape (ip, vshape);
      //  vk = |J| J^{-t} nk
      vk[0] = Jinv(0,0)*nk[k][0]+Jinv(0,1)*nk[k][1]+Jinv(0,2)*nk[k][2];
      vk[1] = Jinv(1,0)*nk[k][0]+Jinv(1,1)*nk[k][1]+Jinv(1,2)*nk[k][2];
      vk[2] = Jinv(2,0)*nk[k][0]+Jinv(2,1)*nk[k][1]+Jinv(2,2)*nk[k][2];
      for (j = 0; j < 36; j++)
         if (fabs (I(k,j) = (vshape(j,0)*vk[0]+vshape(j,1)*vk[1]+
                             vshape(j,2)*vk[2])) < 1.0e-12)
         {
            I(k,j) = 0.0;
         }
   }
}

void RT1HexFiniteElement::Project (
   VectorCoefficient &vc, ElementTransformation &Trans,
   Vector &dofs) const
{
   real_t vk[3];
   Vector xk (vk, 3);

   for (int k = 0; k < 36; k++)
   {
      Trans.SetIntPoint (&Nodes.IntPoint (k));
      // set Jinv = |J| J^{-t} = adj(J)^t
      const DenseMatrix &Jinv = Trans.TransposeAdjugateJacobian();

      vc.Eval (xk, Trans, Nodes.IntPoint (k));
      //  xk^t |J| J^{-t} nk
      dofs(k) =
         vk[0] * ( Jinv(0,0)*nk[k][0]+Jinv(0,1)*nk[k][1]+Jinv(0,2)*nk[k][2] ) +
         vk[1] * ( Jinv(1,0)*nk[k][0]+Jinv(1,1)*nk[k][1]+Jinv(1,2)*nk[k][2] ) +
         vk[2] * ( Jinv(2,0)*nk[k][0]+Jinv(2,1)*nk[k][1]+Jinv(2,2)*nk[k][2] );
   }
}

RT0TetFiniteElement::RT0TetFiniteElement()
   : VectorFiniteElement(3, Geometry::TETRAHEDRON, 4, 1, H_DIV)
{
   // not real nodes ...
   Nodes.IntPoint(0).x = 0.33333333333333333333;
   Nodes.IntPoint(0).y = 0.33333333333333333333;
   Nodes.IntPoint(0).z = 0.33333333333333333333;

   Nodes.IntPoint(1).x = 0.0;
   Nodes.IntPoint(1).y = 0.33333333333333333333;
   Nodes.IntPoint(1).z = 0.33333333333333333333;

   Nodes.IntPoint(2).x = 0.33333333333333333333;
   Nodes.IntPoint(2).y = 0.0;
   Nodes.IntPoint(2).z = 0.33333333333333333333;

   Nodes.IntPoint(3).x = 0.33333333333333333333;
   Nodes.IntPoint(3).y = 0.33333333333333333333;
   Nodes.IntPoint(3).z = 0.0;
}

void RT0TetFiniteElement::CalcVShape(const IntegrationPoint &ip,
                                     DenseMatrix &shape) const
{
   real_t x2 = 2.0*ip.x, y2 = 2.0*ip.y, z2 = 2.0*ip.z;

   shape(0,0) = x2;
   shape(0,1) = y2;
   shape(0,2) = z2;

   shape(1,0) = x2 - 2.0;
   shape(1,1) = y2;
   shape(1,2) = z2;

   shape(2,0) = x2;
   shape(2,1) = y2 - 2.0;
   shape(2,2) = z2;

   shape(3,0) = x2;
   shape(3,1) = y2;
   shape(3,2) = z2 - 2.0;
}

void RT0TetFiniteElement::CalcDivShape(const IntegrationPoint &ip,
                                       Vector &divshape) const
{
   divshape(0) = 6.0;
   divshape(1) = 6.0;
   divshape(2) = 6.0;
   divshape(3) = 6.0;
}

const real_t RT0TetFiniteElement::nk[4][3] =
{{.5,.5,.5}, {-.5,0,0}, {0,-.5,0}, {0,0,-.5}};

void RT0TetFiniteElement::GetLocalInterpolation (
   ElementTransformation &Trans, DenseMatrix &I) const
{
   int k, j;
#ifdef MFEM_THREAD_SAFE
   DenseMatrix vshape(dof, dim);
#endif

#ifdef MFEM_DEBUG
   for (k = 0; k < 4; k++)
   {
      CalcVShape (Nodes.IntPoint(k), vshape);
      for (j = 0; j < 4; j++)
      {
         real_t d = ( vshape(j,0)*nk[k][0] + vshape(j,1)*nk[k][1] +
                      vshape(j,2)*nk[k][2] );
         if (j == k) { d -= 1.0; }
         if (fabs(d) > 1.0e-12)
         {
            mfem::err << "RT0TetFiniteElement::GetLocalInterpolation (...)\n"
                      " k = " << k << ", j = " << j << ", d = " << d << endl;
            mfem_error();
         }
      }
   }
#endif

   IntegrationPoint ip;
   ip.x = ip.y = ip.z = 0.0;
   Trans.SetIntPoint (&ip);
   // Trans must be linear
   // set Jinv = |J| J^{-t} = adj(J)^t
   const DenseMatrix &Jinv = Trans.TransposeAdjugateJacobian();

   real_t vk[3];
   Vector xk (vk, 3);

   for (k = 0; k < 4; k++)
   {
      Trans.Transform (Nodes.IntPoint (k), xk);
      ip.x = vk[0]; ip.y = vk[1]; ip.z = vk[2];
      CalcVShape (ip, vshape);
      //  vk = |J| J^{-t} nk
      vk[0] = Jinv(0,0)*nk[k][0]+Jinv(0,1)*nk[k][1]+Jinv(0,2)*nk[k][2];
      vk[1] = Jinv(1,0)*nk[k][0]+Jinv(1,1)*nk[k][1]+Jinv(1,2)*nk[k][2];
      vk[2] = Jinv(2,0)*nk[k][0]+Jinv(2,1)*nk[k][1]+Jinv(2,2)*nk[k][2];
      for (j = 0; j < 4; j++)
         if (fabs (I(k,j) = (vshape(j,0)*vk[0]+vshape(j,1)*vk[1]+
                             vshape(j,2)*vk[2])) < 1.0e-12)
         {
            I(k,j) = 0.0;
         }
   }
}

void RT0TetFiniteElement::Project (
   VectorCoefficient &vc, ElementTransformation &Trans,
   Vector &dofs) const
{
   real_t vk[3];
   Vector xk (vk, 3);

   for (int k = 0; k < 4; k++)
   {
      Trans.SetIntPoint (&Nodes.IntPoint (k));
      // set Jinv = |J| J^{-t} = adj(J)^t
      const DenseMatrix &Jinv = Trans.TransposeAdjugateJacobian();

      vc.Eval (xk, Trans, Nodes.IntPoint (k));
      //  xk^t |J| J^{-t} nk
      dofs(k) =
         vk[0] * ( Jinv(0,0)*nk[k][0]+Jinv(0,1)*nk[k][1]+Jinv(0,2)*nk[k][2] ) +
         vk[1] * ( Jinv(1,0)*nk[k][0]+Jinv(1,1)*nk[k][1]+Jinv(1,2)*nk[k][2] ) +
         vk[2] * ( Jinv(2,0)*nk[k][0]+Jinv(2,1)*nk[k][1]+Jinv(2,2)*nk[k][2] );
   }
}

RT0WdgFiniteElement::RT0WdgFiniteElement()
   : VectorFiniteElement(3, Geometry::PRISM, 5, 1, H_DIV, FunctionSpace::Pk)
{
   // not real nodes ...
   Nodes.IntPoint(0).x = 0.33333333333333333333;
   Nodes.IntPoint(0).y = 0.33333333333333333333;
   Nodes.IntPoint(0).z = 0.0;

   Nodes.IntPoint(1).x = 0.33333333333333333333;
   Nodes.IntPoint(1).y = 0.33333333333333333333;
   Nodes.IntPoint(1).z = 1.0;

   Nodes.IntPoint(2).x = 0.5;
   Nodes.IntPoint(2).y = 0.0;
   Nodes.IntPoint(2).z = 0.5;

   Nodes.IntPoint(3).x = 0.5;
   Nodes.IntPoint(3).y = 0.5;
   Nodes.IntPoint(3).z = 0.5;

   Nodes.IntPoint(4).x = 0.0;
   Nodes.IntPoint(4).y = 0.5;
   Nodes.IntPoint(4).z = 0.5;
}

void RT0WdgFiniteElement::CalcVShape(const IntegrationPoint &ip,
                                     DenseMatrix &shape) const
{
   real_t x = ip.x, y = ip.y, z2 = 2.0*ip.z;

   shape(0,0) = 0.0;
   shape(0,1) = 0.0;
   shape(0,2) = z2 - 2.0;

   shape(1,0) = 0.0;
   shape(1,1) = 0.0;
   shape(1,2) = z2;

   shape(2,0) = x;
   shape(2,1) = y - 1.0;
   shape(2,2) = 0.0;

   shape(3,0) = x;
   shape(3,1) = y;
   shape(3,2) = 0.0;

   shape(4,0) = x - 1.0;
   shape(4,1) = y;
   shape(4,2) = 0.0;
}

void RT0WdgFiniteElement::CalcDivShape(const IntegrationPoint &ip,
                                       Vector &divshape) const
{
   divshape(0) = 2.0;
   divshape(1) = 2.0;
   divshape(2) = 2.0;
   divshape(3) = 2.0;
   divshape(4) = 2.0;
}

const real_t RT0WdgFiniteElement::nk[5][3] =
{{0.,0.,-.5}, {0.,0.,.5}, {0,-1.,0}, {1.,1.,0}, {-1.,0,0}};

void RT0WdgFiniteElement::GetLocalInterpolation (
   ElementTransformation &Trans, DenseMatrix &I) const
{
   int k, j;
#ifdef MFEM_THREAD_SAFE
   DenseMatrix vshape(dof, dim);
#endif

#ifdef MFEM_DEBUG
   for (k = 0; k < dof; k++)
   {
      CalcVShape (Nodes.IntPoint(k), vshape);
      for (j = 0; j < dof; j++)
      {
         real_t d = ( vshape(j,0)*nk[k][0] + vshape(j,1)*nk[k][1] +
                      vshape(j,2)*nk[k][2] );
         if (j == k) { d -= 1.0; }
         if (fabs(d) > 1.0e-12)
         {
            mfem::err << "RT0WdgFiniteElement::GetLocalInterpolation (...)\n"
                      " k = " << k << ", j = " << j << ", d = " << d << endl;
            mfem_error();
         }
      }
   }
#endif

   IntegrationPoint ip;
   ip.x = ip.y = ip.z = 0.0;
   Trans.SetIntPoint (&ip);
   // Trans must be linear
   // set Jinv = |J| J^{-t} = adj(J)^t
   const DenseMatrix &Jinv = Trans.TransposeAdjugateJacobian();

   real_t vk[3];
   Vector xk (vk, 3);

   for (k = 0; k < dof; k++)
   {
      Trans.Transform (Nodes.IntPoint (k), xk);
      ip.x = vk[0]; ip.y = vk[1]; ip.z = vk[2];
      CalcVShape (ip, vshape);
      //  vk = |J| J^{-t} nk
      vk[0] = Jinv(0,0)*nk[k][0]+Jinv(0,1)*nk[k][1]+Jinv(0,2)*nk[k][2];
      vk[1] = Jinv(1,0)*nk[k][0]+Jinv(1,1)*nk[k][1]+Jinv(1,2)*nk[k][2];
      vk[2] = Jinv(2,0)*nk[k][0]+Jinv(2,1)*nk[k][1]+Jinv(2,2)*nk[k][2];
      for (j = 0; j < dof; j++)
         if (fabs (I(k,j) = (vshape(j,0)*vk[0]+vshape(j,1)*vk[1]+
                             vshape(j,2)*vk[2])) < 1.0e-12)
         {
            I(k,j) = 0.0;
         }
   }
}

void RT0WdgFiniteElement::Project (
   VectorCoefficient &vc, ElementTransformation &Trans,
   Vector &dofs) const
{
   real_t vk[3];
   Vector xk (vk, 3);

   for (int k = 0; k < 5; k++)
   {
      Trans.SetIntPoint (&Nodes.IntPoint (k));
      // set Jinv = |J| J^{-t} = adj(J)^t
      const DenseMatrix &Jinv = Trans.TransposeAdjugateJacobian();

      vc.Eval (xk, Trans, Nodes.IntPoint (k));
      //  xk^t |J| J^{-t} nk
      dofs(k) =
         vk[0] * ( Jinv(0,0)*nk[k][0]+Jinv(0,1)*nk[k][1]+Jinv(0,2)*nk[k][2] ) +
         vk[1] * ( Jinv(1,0)*nk[k][0]+Jinv(1,1)*nk[k][1]+Jinv(1,2)*nk[k][2] ) +
         vk[2] * ( Jinv(2,0)*nk[k][0]+Jinv(2,1)*nk[k][1]+Jinv(2,2)*nk[k][2] );
   }
}

void RT0WdgFiniteElement::ProjectCurl(const FiniteElement &fe,
                                      ElementTransformation &Trans,
                                      DenseMatrix &curl) const
{
   DenseMatrix curl_shape(fe.GetDof(), dim);
   Vector curl_k(fe.GetDof());

   curl.SetSize(dof, fe.GetDof());
   for (int k = 0; k < dof; k++)
   {
      fe.CalcCurlShape(Nodes.IntPoint(k), curl_shape);
      curl_shape.Mult(nk[k], curl_k);
      for (int j = 0; j < curl_k.Size(); j++)
      {
         curl(k,j) = (fabs(curl_k(j)) < 1e-12) ? 0.0 : curl_k(j);
      }
   }
}

RT0PyrFiniteElement::RT0PyrFiniteElement(bool rt0tets)
   : VectorFiniteElement(3, Geometry::PYRAMID, 5, 1, H_DIV, FunctionSpace::Uk),
     rt0(rt0tets)
{
   // not real nodes ...
   Nodes.IntPoint(0).x = 0.5;
   Nodes.IntPoint(0).y = 0.5;
   Nodes.IntPoint(0).z = 0.0;

   Nodes.IntPoint(1).x = 0.33333333333333333333;
   Nodes.IntPoint(1).y = 0.0;
   Nodes.IntPoint(1).z = 0.33333333333333333333;

   Nodes.IntPoint(2).x = 0.66666666666666666667;
   Nodes.IntPoint(2).y = 0.33333333333333333333;
   Nodes.IntPoint(2).z = 0.33333333333333333333;

   Nodes.IntPoint(3).x = 0.33333333333333333333;
   Nodes.IntPoint(3).y = 0.66666666666666666667;
   Nodes.IntPoint(3).z = 0.33333333333333333333;

   Nodes.IntPoint(4).x = 0.0;
   Nodes.IntPoint(4).y = 0.33333333333333333333;
   Nodes.IntPoint(4).z = 0.33333333333333333333;
}

void RT0PyrFiniteElement::CalcVShape(const IntegrationPoint &ip,
                                     DenseMatrix &shape) const
{
   real_t x = ip.x, y = ip.y, z = ip.z, oz = 1.0 - z;
   real_t x2 = 2.0*ip.x, y2 = 2.0*ip.y, z2 = 2.0*ip.z;

   real_t tol = 1e-6;

   if (oz <= tol)
   {
      // We must return the limit of the basis functions as z->1.  In order to
      // remain inside the pyramid in this limit the x and y coordinates must
      // be approaching 0. Unfortunately we obtain different limits if we
      // approach (0,0,1) from different directions. The values provided below
      // are the limits of the average over the square cross section
      // [0,epsilon]x[0,epsilon] at the height z = 1-epsilon as epsilon -> 0.
      shape(0,0) =   0.0;
      shape(0,1) =   0.0;
      shape(0,2) =   0.0;

      shape(1,0) = - 0.5;
      shape(1,1) = - 1.5;
      shape(1,2) =   1.;

      shape(2,0) =   0.5;
      shape(2,1) = - 0.5;
      shape(2,2) =   1.0;

      shape(3,0) = - 0.5;
      shape(3,1) =   0.5;
      shape(3,2) =   1.0;

      shape(4,0) = - 1.5;
      shape(4,1) = - 0.5;
      shape(4,2) =   1.0;

      if (!rt0)
      {
         for (int i=1; i<5; i++)
            for (int j=0; j<3; j++)
            {
               shape(i, j) *= 0.5;
            }
      }

      return;
   }

   real_t ozi = 1.0 / oz;

   shape(0,0) = x;
   shape(0,1) = y;
   shape(0,2) = z - 1.;

   shape(1,0) = - x * z * ozi;
   shape(1,1) = (y2 + z2 - y * z - 2.0) * ozi;
   shape(1,2) = z;

   shape(2,0) = x * (2.0 - z) * ozi;
   shape(2,1) = - y * z * ozi;
   shape(2,2) = z;

   shape(3,0) = - x * z * ozi;
   shape(3,1) = y * (2.0 - z) * ozi;
   shape(3,2) = z;

   shape(4,0) = (x2 + z2 - x * z - 2.0) * ozi;
   shape(4,1) = - y * z * ozi;
   shape(4,2) = z;

   if (!rt0)
   {
      for (int i=1; i<5; i++)
         for (int j=0; j<3; j++)
         {
            shape(i, j) *= 0.5;
         }
   }
}

void RT0PyrFiniteElement::CalcDivShape(const IntegrationPoint &ip,
                                       Vector &divshape) const
{
   divshape(0) = 3.0;
   divshape(1) = 3.0;
   divshape(2) = 3.0;
   divshape(3) = 3.0;
   divshape(4) = 3.0;

   if (!rt0)
   {
      for (int i=1; i<5; i++)
      {
         divshape(i) *= 0.5;
      }
   }
}

const real_t RT0PyrFiniteElement::nk[5][3] =
{{0.,0.,-1}, {0,-1,0}, {1,0,1}, {0,1,1}, {-1,0,0}};

void RT0PyrFiniteElement::GetLocalInterpolation (
   ElementTransformation &Trans, DenseMatrix &I) const
{
   int k, j;
#ifdef MFEM_THREAD_SAFE
   DenseMatrix vshape(dof, dim);
#endif

#ifdef MFEM_DEBUG
   for (k = 0; k < dof; k++)
   {
      CalcVShape (Nodes.IntPoint(k), vshape);
      for (j = 0; j < dof; j++)
      {
         real_t d = ( vshape(j,0)*nk[k][0] + vshape(j,1)*nk[k][1] +
                      vshape(j,2)*nk[k][2] );
         if (j == k) { d -= 1.0; }
         if (fabs(d) > 1.0e-12)
         {
            mfem::err << "RT0PyrFiniteElement::GetLocalInterpolation (...)\n"
                      " k = " << k << ", j = " << j << ", d = " << d << endl;
            mfem_error();
         }
      }
   }
#endif

   IntegrationPoint ip;
   ip.x = ip.y = ip.z = 0.0;
   Trans.SetIntPoint (&ip);
   // Trans must be linear
   // set Jinv = |J| J^{-t} = adj(J)^t
   const DenseMatrix &Jinv = Trans.TransposeAdjugateJacobian();

   real_t vk[3];
   Vector xk (vk, 3);

   for (k = 0; k < dof; k++)
   {
      Trans.Transform (Nodes.IntPoint (k), xk);
      ip.x = vk[0]; ip.y = vk[1]; ip.z = vk[2];
      CalcVShape (ip, vshape);
      //  vk = |J| J^{-t} nk
      vk[0] = Jinv(0,0)*nk[k][0]+Jinv(0,1)*nk[k][1]+Jinv(0,2)*nk[k][2];
      vk[1] = Jinv(1,0)*nk[k][0]+Jinv(1,1)*nk[k][1]+Jinv(1,2)*nk[k][2];
      vk[2] = Jinv(2,0)*nk[k][0]+Jinv(2,1)*nk[k][1]+Jinv(2,2)*nk[k][2];
      for (j = 0; j < dof; j++)
         if (fabs (I(k,j) = (vshape(j,0)*vk[0]+vshape(j,1)*vk[1]+
                             vshape(j,2)*vk[2])) < 1.0e-12)
         {
            I(k,j) = 0.0;
         }
   }
}

void RT0PyrFiniteElement::Project (
   VectorCoefficient &vc, ElementTransformation &Trans,
   Vector &dofs) const
{
   real_t vk[3];
   Vector xk (vk, 3);

   for (int k = 0; k < dof; k++)
   {
      Trans.SetIntPoint (&Nodes.IntPoint (k));
      // set Jinv = |J| J^{-t} = adj(J)^t
      const DenseMatrix &Jinv = Trans.TransposeAdjugateJacobian();

      vc.Eval (xk, Trans, Nodes.IntPoint (k));
      //  xk^t |J| J^{-t} nk
      dofs(k) =
         vk[0] * ( Jinv(0,0)*nk[k][0]+Jinv(0,1)*nk[k][1]+Jinv(0,2)*nk[k][2] ) +
         vk[1] * ( Jinv(1,0)*nk[k][0]+Jinv(1,1)*nk[k][1]+Jinv(1,2)*nk[k][2] ) +
         vk[2] * ( Jinv(2,0)*nk[k][0]+Jinv(2,1)*nk[k][1]+Jinv(2,2)*nk[k][2] );
      if (!rt0 && k > 0) { dofs[k] *= 2.0; }
   }
}

void RT0PyrFiniteElement::ProjectCurl(const FiniteElement &fe,
                                      ElementTransformation &Trans,
                                      DenseMatrix &curl) const
{
   DenseMatrix curl_shape(fe.GetDof(), dim);
   Vector curl_k(fe.GetDof());

   curl.SetSize(dof, fe.GetDof());
   for (int k = 0; k < dof; k++)
   {
      fe.CalcCurlShape(Nodes.IntPoint(k), curl_shape);
      curl_shape.Mult(nk[k], curl_k);
      if (!rt0 && k > 0) { curl_k *= 2.0; }
      for (int j = 0; j < curl_k.Size(); j++)
      {
         curl(k,j) = (fabs(curl_k(j)) < 1e-12) ? 0.0 : curl_k(j);
      }
   }
}

RotTriLinearHexFiniteElement::RotTriLinearHexFiniteElement()
   : NodalFiniteElement(3, Geometry::CUBE, 6, 2, FunctionSpace::Qk)
{
   Nodes.IntPoint(0).x = 0.5;
   Nodes.IntPoint(0).y = 0.5;
   Nodes.IntPoint(0).z = 0.0;

   Nodes.IntPoint(1).x = 0.5;
   Nodes.IntPoint(1).y = 0.0;
   Nodes.IntPoint(1).z = 0.5;

   Nodes.IntPoint(2).x = 1.0;
   Nodes.IntPoint(2).y = 0.5;
   Nodes.IntPoint(2).z = 0.5;

   Nodes.IntPoint(3).x = 0.5;
   Nodes.IntPoint(3).y = 1.0;
   Nodes.IntPoint(3).z = 0.5;

   Nodes.IntPoint(4).x = 0.0;
   Nodes.IntPoint(4).y = 0.5;
   Nodes.IntPoint(4).z = 0.5;

   Nodes.IntPoint(5).x = 0.5;
   Nodes.IntPoint(5).y = 0.5;
   Nodes.IntPoint(5).z = 1.0;
}

void RotTriLinearHexFiniteElement::CalcShape(const IntegrationPoint &ip,
                                             Vector &shape) const
{
   real_t x = 2. * ip.x - 1.;
   real_t y = 2. * ip.y - 1.;
   real_t z = 2. * ip.z - 1.;
   real_t f5 = x * x - y * y;
   real_t f6 = y * y - z * z;

   shape(0) = (1./6.) * (1. - 3. * z -      f5 - 2. * f6);
   shape(1) = (1./6.) * (1. - 3. * y -      f5 +      f6);
   shape(2) = (1./6.) * (1. + 3. * x + 2. * f5 +      f6);
   shape(3) = (1./6.) * (1. + 3. * y -      f5 +      f6);
   shape(4) = (1./6.) * (1. - 3. * x + 2. * f5 +      f6);
   shape(5) = (1./6.) * (1. + 3. * z -      f5 - 2. * f6);
}

void RotTriLinearHexFiniteElement::CalcDShape(const IntegrationPoint &ip,
                                              DenseMatrix &dshape) const
{
   const real_t a = 2./3.;

   real_t xt = a * (1. - 2. * ip.x);
   real_t yt = a * (1. - 2. * ip.y);
   real_t zt = a * (1. - 2. * ip.z);

   dshape(0,0) = xt;
   dshape(0,1) = yt;
   dshape(0,2) = -1. - 2. * zt;

   dshape(1,0) = xt;
   dshape(1,1) = -1. - 2. * yt;
   dshape(1,2) = zt;

   dshape(2,0) = 1. - 2. * xt;
   dshape(2,1) = yt;
   dshape(2,2) = zt;

   dshape(3,0) = xt;
   dshape(3,1) = 1. - 2. * yt;
   dshape(3,2) = zt;

   dshape(4,0) = -1. - 2. * xt;
   dshape(4,1) = yt;
   dshape(4,2) = zt;

   dshape(5,0) = xt;
   dshape(5,1) = yt;
   dshape(5,2) = 1. - 2. * zt;
}

}
