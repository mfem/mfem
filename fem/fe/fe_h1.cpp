// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

// H1 Finite Element classes

#include "fe_h1.hpp"

namespace mfem
{

using namespace std;

H1_SegmentElement::H1_SegmentElement(const int p, const int btype)
   : NodalTensorFiniteElement(1, p, VerifyClosed(btype), H1_DOF_MAP)
{
   const double *cp = poly1d.ClosedPoints(p, b_type);

#ifndef MFEM_THREAD_SAFE
   shape_x.SetSize(p+1);
   dshape_x.SetSize(p+1);
   d2shape_x.SetSize(p+1);
#endif

   Nodes.IntPoint(0).x = cp[0];
   Nodes.IntPoint(1).x = cp[p];
   for (int i = 1; i < p; i++)
   {
      Nodes.IntPoint(i+1).x = cp[i];
   }
}

void H1_SegmentElement::CalcShape(const IntegrationPoint &ip,
                                  Vector &shape) const
{
   const int p = order;

#ifdef MFEM_THREAD_SAFE
   Vector shape_x(p+1);
#endif

   basis1d.Eval(ip.x, shape_x);

   shape(0) = shape_x(0);
   shape(1) = shape_x(p);
   for (int i = 1; i < p; i++)
   {
      shape(i+1) = shape_x(i);
   }
}

void H1_SegmentElement::CalcDShape(const IntegrationPoint &ip,
                                   DenseMatrix &dshape) const
{
   const int p = order;

#ifdef MFEM_THREAD_SAFE
   Vector shape_x(p+1), dshape_x(p+1);
#endif

   basis1d.Eval(ip.x, shape_x, dshape_x);

   dshape(0,0) = dshape_x(0);
   dshape(1,0) = dshape_x(p);
   for (int i = 1; i < p; i++)
   {
      dshape(i+1,0) = dshape_x(i);
   }
}

void H1_SegmentElement::CalcHessian(const IntegrationPoint &ip,
                                    DenseMatrix &Hessian) const
{
   const int p = order;

#ifdef MFEM_THREAD_SAFE
   Vector shape_x(p+1), dshape_x(p+1), d2shape_x(p+1);
#endif

   basis1d.Eval(ip.x, shape_x, dshape_x, d2shape_x);

   Hessian(0,0) = d2shape_x(0);
   Hessian(1,0) = d2shape_x(p);
   for (int i = 1; i < p; i++)
   {
      Hessian(i+1,0) = d2shape_x(i);
   }
}

void H1_SegmentElement::ProjectDelta(int vertex, Vector &dofs) const
{
   const int p = order;
   const double *cp = poly1d.ClosedPoints(p, b_type);

   switch (vertex)
   {
      case 0:
         dofs(0) = poly1d.CalcDelta(p, (1.0 - cp[0]));
         dofs(1) = poly1d.CalcDelta(p, (1.0 - cp[p]));
         for (int i = 1; i < p; i++)
         {
            dofs(i+1) = poly1d.CalcDelta(p, (1.0 - cp[i]));
         }
         break;

      case 1:
         dofs(0) = poly1d.CalcDelta(p, cp[0]);
         dofs(1) = poly1d.CalcDelta(p, cp[p]);
         for (int i = 1; i < p; i++)
         {
            dofs(i+1) = poly1d.CalcDelta(p, cp[i]);
         }
         break;
   }
}


H1_QuadrilateralElement::H1_QuadrilateralElement(const int p, const int btype)
   : NodalTensorFiniteElement(2, p, VerifyClosed(btype), H1_DOF_MAP)
{
   const double *cp = poly1d.ClosedPoints(p, b_type);

#ifndef MFEM_THREAD_SAFE
   const int p1 = p + 1;

   shape_x.SetSize(p1);
   shape_y.SetSize(p1);
   dshape_x.SetSize(p1);
   dshape_y.SetSize(p1);
   d2shape_x.SetSize(p1);
   d2shape_y.SetSize(p1);
#endif

   int o = 0;
   for (int j = 0; j <= p; j++)
   {
      for (int i = 0; i <= p; i++)
      {
         Nodes.IntPoint(dof_map[o++]).Set2(cp[i], cp[j]);
      }
   }
}

void H1_QuadrilateralElement::CalcShape(const IntegrationPoint &ip,
                                        Vector &shape) const
{
   const int p = order;

#ifdef MFEM_THREAD_SAFE
   Vector shape_x(p+1), shape_y(p+1);
#endif

   basis1d.Eval(ip.x, shape_x);
   basis1d.Eval(ip.y, shape_y);

   for (int o = 0, j = 0; j <= p; j++)
      for (int i = 0; i <= p; i++)
      {
         shape(dof_map[o++]) = shape_x(i)*shape_y(j);
      }
}

void H1_QuadrilateralElement::CalcDShape(const IntegrationPoint &ip,
                                         DenseMatrix &dshape) const
{
   const int p = order;

#ifdef MFEM_THREAD_SAFE
   Vector shape_x(p+1), shape_y(p+1), dshape_x(p+1), dshape_y(p+1);
#endif

   basis1d.Eval(ip.x, shape_x, dshape_x);
   basis1d.Eval(ip.y, shape_y, dshape_y);

   for (int o = 0, j = 0; j <= p; j++)
   {
      for (int i = 0; i <= p; i++)
      {
         dshape(dof_map[o],0) = dshape_x(i)* shape_y(j);
         dshape(dof_map[o],1) =  shape_x(i)*dshape_y(j);  o++;
      }
   }
}

void H1_QuadrilateralElement::CalcHessian(const IntegrationPoint &ip,
                                          DenseMatrix &Hessian) const
{
   const int p = order;

#ifdef MFEM_THREAD_SAFE
   Vector shape_x(p+1), shape_y(p+1), dshape_x(p+1), dshape_y(p+1),
          d2shape_x(p+1), d2shape_y(p+1);
#endif

   basis1d.Eval(ip.x, shape_x, dshape_x, d2shape_x);
   basis1d.Eval(ip.y, shape_y, dshape_y, d2shape_y);

   for (int o = 0, j = 0; j <= p; j++)
   {
      for (int i = 0; i <= p; i++)
      {
         Hessian(dof_map[o],0) = d2shape_x(i)*  shape_y(j);
         Hessian(dof_map[o],1) =  dshape_x(i)* dshape_y(j);
         Hessian(dof_map[o],2) =   shape_x(i)*d2shape_y(j);  o++;
      }
   }
}

void H1_QuadrilateralElement::ProjectDelta(int vertex, Vector &dofs) const
{
   const int p = order;
   const double *cp = poly1d.ClosedPoints(p, b_type);

#ifdef MFEM_THREAD_SAFE
   Vector shape_x(p+1), shape_y(p+1);
#endif

   for (int i = 0; i <= p; i++)
   {
      shape_x(i) = poly1d.CalcDelta(p, (1.0 - cp[i]));
      shape_y(i) = poly1d.CalcDelta(p, cp[i]);
   }

   switch (vertex)
   {
      case 0:
         for (int o = 0, j = 0; j <= p; j++)
            for (int i = 0; i <= p; i++)
            {
               dofs(dof_map[o++]) = shape_x(i)*shape_x(j);
            }
         break;
      case 1:
         for (int o = 0, j = 0; j <= p; j++)
            for (int i = 0; i <= p; i++)
            {
               dofs(dof_map[o++]) = shape_y(i)*shape_x(j);
            }
         break;
      case 2:
         for (int o = 0, j = 0; j <= p; j++)
            for (int i = 0; i <= p; i++)
            {
               dofs(dof_map[o++]) = shape_y(i)*shape_y(j);
            }
         break;
      case 3:
         for (int o = 0, j = 0; j <= p; j++)
            for (int i = 0; i <= p; i++)
            {
               dofs(dof_map[o++]) = shape_x(i)*shape_y(j);
            }
         break;
   }
}


H1_HexahedronElement::H1_HexahedronElement(const int p, const int btype)
   : NodalTensorFiniteElement(3, p, VerifyClosed(btype), H1_DOF_MAP)
{
   const double *cp = poly1d.ClosedPoints(p, b_type);

#ifndef MFEM_THREAD_SAFE
   const int p1 = p + 1;

   shape_x.SetSize(p1);
   shape_y.SetSize(p1);
   shape_z.SetSize(p1);
   dshape_x.SetSize(p1);
   dshape_y.SetSize(p1);
   dshape_z.SetSize(p1);
   d2shape_x.SetSize(p1);
   d2shape_y.SetSize(p1);
   d2shape_z.SetSize(p1);
#endif

   int o = 0;
   for (int k = 0; k <= p; k++)
      for (int j = 0; j <= p; j++)
         for (int i = 0; i <= p; i++)
         {
            Nodes.IntPoint(dof_map[o++]).Set3(cp[i], cp[j], cp[k]);
         }
}

void H1_HexahedronElement::CalcShape(const IntegrationPoint &ip,
                                     Vector &shape) const
{
   const int p = order;

#ifdef MFEM_THREAD_SAFE
   Vector shape_x(p+1), shape_y(p+1), shape_z(p+1);
#endif

   basis1d.Eval(ip.x, shape_x);
   basis1d.Eval(ip.y, shape_y);
   basis1d.Eval(ip.z, shape_z);

   for (int o = 0, k = 0; k <= p; k++)
      for (int j = 0; j <= p; j++)
         for (int i = 0; i <= p; i++)
         {
            shape(dof_map[o++]) = shape_x(i)*shape_y(j)*shape_z(k);
         }
}

void H1_HexahedronElement::CalcDShape(const IntegrationPoint &ip,
                                      DenseMatrix &dshape) const
{
   const int p = order;

#ifdef MFEM_THREAD_SAFE
   Vector shape_x(p+1),  shape_y(p+1),  shape_z(p+1);
   Vector dshape_x(p+1), dshape_y(p+1), dshape_z(p+1);
#endif

   basis1d.Eval(ip.x, shape_x, dshape_x);
   basis1d.Eval(ip.y, shape_y, dshape_y);
   basis1d.Eval(ip.z, shape_z, dshape_z);

   for (int o = 0, k = 0; k <= p; k++)
      for (int j = 0; j <= p; j++)
         for (int i = 0; i <= p; i++)
         {
            dshape(dof_map[o],0) = dshape_x(i)* shape_y(j)* shape_z(k);
            dshape(dof_map[o],1) =  shape_x(i)*dshape_y(j)* shape_z(k);
            dshape(dof_map[o],2) =  shape_x(i)* shape_y(j)*dshape_z(k);  o++;
         }
}

void H1_HexahedronElement::CalcHessian(const IntegrationPoint &ip,
                                       DenseMatrix &Hessian) const
{
   const int p = order;

#ifdef MFEM_THREAD_SAFE
   Vector shape_x(p+1),  shape_y(p+1),  shape_z(p+1);
   Vector dshape_x(p+1), dshape_y(p+1), dshape_z(p+1);
   Vector d2shape_x(p+1), d2shape_y(p+1), d2shape_z(p+1);
#endif

   basis1d.Eval(ip.x, shape_x, dshape_x, d2shape_x);
   basis1d.Eval(ip.y, shape_y, dshape_y, d2shape_y);
   basis1d.Eval(ip.z, shape_z, dshape_z, d2shape_z);

   for (int o = 0, k = 0; k <= p; k++)
      for (int j = 0; j <= p; j++)
         for (int i = 0; i <= p; i++)
         {
            Hessian(dof_map[o],0) = d2shape_x(i)*  shape_y(j)*  shape_z(k);
            Hessian(dof_map[o],1) =  dshape_x(i)* dshape_y(j)*  shape_z(k);
            Hessian(dof_map[o],2) =  dshape_x(i)*  shape_y(j)* dshape_z(k);
            Hessian(dof_map[o],3) =   shape_x(i)*d2shape_y(j)*  shape_z(k);
            Hessian(dof_map[o],4) =   shape_x(i)* dshape_y(j)* dshape_z(k);
            Hessian(dof_map[o],5) =   shape_x(i)*  shape_y(j)*d2shape_z(k);
            o++;
         }
}

void H1_HexahedronElement::ProjectDelta(int vertex, Vector &dofs) const
{
   const int p = order;
   const double *cp = poly1d.ClosedPoints(p,b_type);

#ifdef MFEM_THREAD_SAFE
   Vector shape_x(p+1), shape_y(p+1);
#endif

   for (int i = 0; i <= p; i++)
   {
      shape_x(i) = poly1d.CalcDelta(p, (1.0 - cp[i]));
      shape_y(i) = poly1d.CalcDelta(p, cp[i]);
   }

   switch (vertex)
   {
      case 0:
         for (int o = 0, k = 0; k <= p; k++)
            for (int j = 0; j <= p; j++)
               for (int i = 0; i <= p; i++)
               {
                  dofs(dof_map[o++]) = shape_x(i)*shape_x(j)*shape_x(k);
               }
         break;
      case 1:
         for (int o = 0, k = 0; k <= p; k++)
            for (int j = 0; j <= p; j++)
               for (int i = 0; i <= p; i++)
               {
                  dofs(dof_map[o++]) = shape_y(i)*shape_x(j)*shape_x(k);
               }
         break;
      case 2:
         for (int o = 0, k = 0; k <= p; k++)
            for (int j = 0; j <= p; j++)
               for (int i = 0; i <= p; i++)
               {
                  dofs(dof_map[o++]) = shape_y(i)*shape_y(j)*shape_x(k);
               }
         break;
      case 3:
         for (int o = 0, k = 0; k <= p; k++)
            for (int j = 0; j <= p; j++)
               for (int i = 0; i <= p; i++)
               {
                  dofs(dof_map[o++]) = shape_x(i)*shape_y(j)*shape_x(k);
               }
         break;
      case 4:
         for (int o = 0, k = 0; k <= p; k++)
            for (int j = 0; j <= p; j++)
               for (int i = 0; i <= p; i++)
               {
                  dofs(dof_map[o++]) = shape_x(i)*shape_x(j)*shape_y(k);
               }
         break;
      case 5:
         for (int o = 0, k = 0; k <= p; k++)
            for (int j = 0; j <= p; j++)
               for (int i = 0; i <= p; i++)
               {
                  dofs(dof_map[o++]) = shape_y(i)*shape_x(j)*shape_y(k);
               }
         break;
      case 6:
         for (int o = 0, k = 0; k <= p; k++)
            for (int j = 0; j <= p; j++)
               for (int i = 0; i <= p; i++)
               {
                  dofs(dof_map[o++]) = shape_y(i)*shape_y(j)*shape_y(k);
               }
         break;
      case 7:
         for (int o = 0, k = 0; k <= p; k++)
            for (int j = 0; j <= p; j++)
               for (int i = 0; i <= p; i++)
               {
                  dofs(dof_map[o++]) = shape_x(i)*shape_y(j)*shape_y(k);
               }
         break;
   }
}

H1_TriangleElement::H1_TriangleElement(const int p, const int btype)
   : NodalFiniteElement(2, Geometry::TRIANGLE, ((p + 1)*(p + 2))/2, p,
                        FunctionSpace::Pk)
{
   const double *cp = poly1d.ClosedPoints(p, VerifyNodal(VerifyClosed(btype)));

#ifndef MFEM_THREAD_SAFE
   shape_x.SetSize(p + 1);
   shape_y.SetSize(p + 1);
   shape_l.SetSize(p + 1);
   dshape_x.SetSize(p + 1);
   dshape_y.SetSize(p + 1);
   dshape_l.SetSize(p + 1);
   ddshape_x.SetSize(p + 1);
   ddshape_y.SetSize(p + 1);
   ddshape_l.SetSize(p + 1);
   u.SetSize(dof);
   du.SetSize(dof, dim);
   ddu.SetSize(dof, (dim * (dim + 1)) / 2 );
#else
   Vector shape_x(p + 1), shape_y(p + 1), shape_l(p + 1);
#endif

   int p2p3 = 2*p + 3;
   auto idx = [p2p3](int i, int j) { return ((p2p3-j)*j)/2+i; };
   lex_ordering.SetSize(dof);

   // vertices
   lex_ordering[idx(0,0)] = 0;
   Nodes.IntPoint(0).Set2(cp[0], cp[0]);
   lex_ordering[idx(p,0)] = 1;
   Nodes.IntPoint(1).Set2(cp[p], cp[0]);
   lex_ordering[idx(0,p)] = 2;
   Nodes.IntPoint(2).Set2(cp[0], cp[p]);

   // edges
   int o = 3;
   for (int i = 1; i < p; i++)
   {
      lex_ordering[idx(i,0)] = o;
      Nodes.IntPoint(o++).Set2(cp[i], cp[0]);
   }
   for (int i = 1; i < p; i++)
   {
      lex_ordering[idx(p-i,i)] = o;
      Nodes.IntPoint(o++).Set2(cp[p-i], cp[i]);
   }
   for (int i = 1; i < p; i++)
   {
      lex_ordering[idx(0,p-i)] = o;
      Nodes.IntPoint(o++).Set2(cp[0], cp[p-i]);
   }

   // interior
   for (int j = 1; j < p; j++)
      for (int i = 1; i + j < p; i++)
      {
         const double w = cp[i] + cp[j] + cp[p-i-j];
         lex_ordering[idx(i,j)] = o;
         Nodes.IntPoint(o++).Set2(cp[i]/w, cp[j]/w);
      }

   DenseMatrix T(dof);
   for (int k = 0; k < dof; k++)
   {
      IntegrationPoint &ip = Nodes.IntPoint(k);
      poly1d.CalcBasis(p, ip.x, shape_x);
      poly1d.CalcBasis(p, ip.y, shape_y);
      poly1d.CalcBasis(p, 1. - ip.x - ip.y, shape_l);

      o = 0;
      for (int j = 0; j <= p; j++)
         for (int i = 0; i + j <= p; i++)
         {
            T(o++, k) = shape_x(i)*shape_y(j)*shape_l(p-i-j);
         }
   }

   Ti.Factor(T);
   // mfem::out << "H1_TriangleElement(" << p << ") : "; Ti.TestInversion();
}

void H1_TriangleElement::CalcShape(const IntegrationPoint &ip,
                                   Vector &shape) const
{
   const int p = order;

#ifdef MFEM_THREAD_SAFE
   Vector shape_x(p + 1), shape_y(p + 1), shape_l(p + 1), u(dof);
#endif

   poly1d.CalcBasis(p, ip.x, shape_x);
   poly1d.CalcBasis(p, ip.y, shape_y);
   poly1d.CalcBasis(p, 1. - ip.x - ip.y, shape_l);

   for (int o = 0, j = 0; j <= p; j++)
      for (int i = 0; i + j <= p; i++)
      {
         u(o++) = shape_x(i)*shape_y(j)*shape_l(p-i-j);
      }

   Ti.Mult(u, shape);
}

void H1_TriangleElement::CalcDShape(const IntegrationPoint &ip,
                                    DenseMatrix &dshape) const
{
   const int p = order;

#ifdef MFEM_THREAD_SAFE
   Vector  shape_x(p + 1),  shape_y(p + 1),  shape_l(p + 1);
   Vector dshape_x(p + 1), dshape_y(p + 1), dshape_l(p + 1);
   DenseMatrix du(dof, dim);
#endif

   poly1d.CalcBasis(p, ip.x, shape_x, dshape_x);
   poly1d.CalcBasis(p, ip.y, shape_y, dshape_y);
   poly1d.CalcBasis(p, 1. - ip.x - ip.y, shape_l, dshape_l);

   for (int o = 0, j = 0; j <= p; j++)
      for (int i = 0; i + j <= p; i++)
      {
         int k = p - i - j;
         du(o,0) = ((dshape_x(i)* shape_l(k)) -
                    ( shape_x(i)*dshape_l(k)))*shape_y(j);
         du(o,1) = ((dshape_y(j)* shape_l(k)) -
                    ( shape_y(j)*dshape_l(k)))*shape_x(i);
         o++;
      }

   Ti.Mult(du, dshape);
}

void H1_TriangleElement::CalcHessian(const IntegrationPoint &ip,
                                     DenseMatrix &ddshape) const
{
   const int p = order;
#ifdef MFEM_THREAD_SAFE
   Vector   shape_x(p + 1),   shape_y(p + 1),   shape_l(p + 1);
   Vector  dshape_x(p + 1),  dshape_y(p + 1),  dshape_l(p + 1);
   Vector ddshape_x(p + 1), ddshape_y(p + 1), ddshape_l(p + 1);
   DenseMatrix ddu(dof, dim);
#endif

   poly1d.CalcBasis(p, ip.x, shape_x, dshape_x, ddshape_x);
   poly1d.CalcBasis(p, ip.y, shape_y, dshape_y, ddshape_y);
   poly1d.CalcBasis(p, 1. - ip.x - ip.y, shape_l, dshape_l, ddshape_l);

   for (int o = 0, j = 0; j <= p; j++)
      for (int i = 0; i + j <= p; i++)
      {
         int k = p - i - j;
         // u_xx, u_xy, u_yy
         ddu(o,0) = ((ddshape_x(i) * shape_l(k)) - 2. * (dshape_x(i) * dshape_l(k)) +
                     (shape_x(i) * ddshape_l(k))) * shape_y(j);
         ddu(o,1) = (((shape_x(i) * ddshape_l(k)) - dshape_x(i) * dshape_l(k)) * shape_y(
                        j)) + (((dshape_x(i) * shape_l(k)) - (shape_x(i) * dshape_l(k))) * dshape_y(j));
         ddu(o,2) = ((ddshape_y(j) * shape_l(k)) - 2. * (dshape_y(j) * dshape_l(k)) +
                     (shape_y(j) * ddshape_l(k))) * shape_x(i);
         o++;
      }

   Ti.Mult(ddu, ddshape);
}


H1_TetrahedronElement::H1_TetrahedronElement(const int p, const int btype)
   : NodalFiniteElement(3, Geometry::TETRAHEDRON, ((p + 1)*(p + 2)*(p + 3))/6,
                        p, FunctionSpace::Pk)
{
   const double *cp = poly1d.ClosedPoints(p, VerifyNodal(VerifyClosed(btype)));

#ifndef MFEM_THREAD_SAFE
   shape_x.SetSize(p + 1);
   shape_y.SetSize(p + 1);
   shape_z.SetSize(p + 1);
   shape_l.SetSize(p + 1);
   dshape_x.SetSize(p + 1);
   dshape_y.SetSize(p + 1);
   dshape_z.SetSize(p + 1);
   dshape_l.SetSize(p + 1);
   ddshape_x.SetSize(p + 1);
   ddshape_y.SetSize(p + 1);
   ddshape_z.SetSize(p + 1);
   ddshape_l.SetSize(p + 1);
   u.SetSize(dof);
   du.SetSize(dof, dim);
   ddu.SetSize(dof, (dim * (dim + 1)) / 2);
#else
   Vector shape_x(p + 1), shape_y(p + 1), shape_z(p + 1), shape_l(p + 1);
#endif

   auto tri = [](int k) { return (k*(k + 1))/2; };
   auto tet = [](int k) { return (k*(k + 1)*(k + 2))/6; };
   int ndof = tet(p+1);
   auto idx = [tri, tet, p, ndof](int i, int j, int k)
   {
      return ndof - tet(p - k) - tri(p + 1 - k - j) + i;
   };

   lex_ordering.SetSize(dof);

   // vertices
   lex_ordering[idx(0,0,0)] = 0;
   Nodes.IntPoint(0).Set3(cp[0], cp[0], cp[0]);
   lex_ordering[idx(p,0,0)] = 1;
   Nodes.IntPoint(1).Set3(cp[p], cp[0], cp[0]);
   lex_ordering[idx(0,p,0)] = 2;
   Nodes.IntPoint(2).Set3(cp[0], cp[p], cp[0]);
   lex_ordering[idx(0,0,p)] = 3;
   Nodes.IntPoint(3).Set3(cp[0], cp[0], cp[p]);

   // edges (see Tetrahedron::edges in mesh/tetrahedron.cpp)
   int o = 4;
   for (int i = 1; i < p; i++)  // (0,1)
   {
      lex_ordering[idx(i,0,0)] = o;
      Nodes.IntPoint(o++).Set3(cp[i], cp[0], cp[0]);
   }
   for (int i = 1; i < p; i++)  // (0,2)
   {
      lex_ordering[idx(0,i,0)] = o;
      Nodes.IntPoint(o++).Set3(cp[0], cp[i], cp[0]);
   }
   for (int i = 1; i < p; i++)  // (0,3)
   {
      lex_ordering[idx(0,0,i)] = o;
      Nodes.IntPoint(o++).Set3(cp[0], cp[0], cp[i]);
   }
   for (int i = 1; i < p; i++)  // (1,2)
   {
      lex_ordering[idx(p-i,i,0)] = o;
      Nodes.IntPoint(o++).Set3(cp[p-i], cp[i], cp[0]);
   }
   for (int i = 1; i < p; i++)  // (1,3)
   {
      lex_ordering[idx(p-i,0,i)] = o;
      Nodes.IntPoint(o++).Set3(cp[p-i], cp[0], cp[i]);
   }
   for (int i = 1; i < p; i++)  // (2,3)
   {
      lex_ordering[idx(0,p-i,i)] = o;
      Nodes.IntPoint(o++).Set3(cp[0], cp[p-i], cp[i]);
   }

   // faces (see Mesh::GenerateFaces in mesh/mesh.cpp)
   for (int j = 1; j < p; j++)
      for (int i = 1; i + j < p; i++)  // (1,2,3)
      {
         lex_ordering[idx(p-i-j,i,j)] = o;
         double w = cp[i] + cp[j] + cp[p-i-j];
         Nodes.IntPoint(o++).Set3(cp[p-i-j]/w, cp[i]/w, cp[j]/w);
      }
   for (int j = 1; j < p; j++)
      for (int i = 1; i + j < p; i++)  // (0,3,2)
      {
         lex_ordering[idx(0,j,i)] = o;
         double w = cp[i] + cp[j] + cp[p-i-j];
         Nodes.IntPoint(o++).Set3(cp[0], cp[j]/w, cp[i]/w);
      }
   for (int j = 1; j < p; j++)
      for (int i = 1; i + j < p; i++)  // (0,1,3)
      {
         lex_ordering[idx(i,0,j)] = o;
         double w = cp[i] + cp[j] + cp[p-i-j];
         Nodes.IntPoint(o++).Set3(cp[i]/w, cp[0], cp[j]/w);
      }
   for (int j = 1; j < p; j++)
      for (int i = 1; i + j < p; i++)  // (0,2,1)
      {
         lex_ordering[idx(j,i,0)] = o;
         double w = cp[i] + cp[j] + cp[p-i-j];
         Nodes.IntPoint(o++).Set3(cp[j]/w, cp[i]/w, cp[0]);
      }

   // interior
   for (int k = 1; k < p; k++)
      for (int j = 1; j + k < p; j++)
         for (int i = 1; i + j + k < p; i++)
         {
            lex_ordering[idx(i,j,k)] = o;
            double w = cp[i] + cp[j] + cp[k] + cp[p-i-j-k];
            Nodes.IntPoint(o++).Set3(cp[i]/w, cp[j]/w, cp[k]/w);
         }

   DenseMatrix T(dof);
   for (int m = 0; m < dof; m++)
   {
      IntegrationPoint &ip = Nodes.IntPoint(m);
      poly1d.CalcBasis(p, ip.x, shape_x);
      poly1d.CalcBasis(p, ip.y, shape_y);
      poly1d.CalcBasis(p, ip.z, shape_z);
      poly1d.CalcBasis(p, 1. - ip.x - ip.y - ip.z, shape_l);

      o = 0;
      for (int k = 0; k <= p; k++)
         for (int j = 0; j + k <= p; j++)
            for (int i = 0; i + j + k <= p; i++)
            {
               T(o++, m) = shape_x(i)*shape_y(j)*shape_z(k)*shape_l(p-i-j-k);
            }
   }

   Ti.Factor(T);
   // mfem::out << "H1_TetrahedronElement(" << p << ") : "; Ti.TestInversion();
}

void H1_TetrahedronElement::CalcShape(const IntegrationPoint &ip,
                                      Vector &shape) const
{
   const int p = order;

#ifdef MFEM_THREAD_SAFE
   Vector shape_x(p + 1), shape_y(p + 1), shape_z(p + 1), shape_l(p + 1);
   Vector u(dof);
#endif

   poly1d.CalcBasis(p, ip.x, shape_x);
   poly1d.CalcBasis(p, ip.y, shape_y);
   poly1d.CalcBasis(p, ip.z, shape_z);
   poly1d.CalcBasis(p, 1. - ip.x - ip.y - ip.z, shape_l);

   for (int o = 0, k = 0; k <= p; k++)
      for (int j = 0; j + k <= p; j++)
         for (int i = 0; i + j + k <= p; i++)
         {
            u(o++) = shape_x(i)*shape_y(j)*shape_z(k)*shape_l(p-i-j-k);
         }

   Ti.Mult(u, shape);
}

void H1_TetrahedronElement::CalcDShape(const IntegrationPoint &ip,
                                       DenseMatrix &dshape) const
{
   const int p = order;

#ifdef MFEM_THREAD_SAFE
   Vector  shape_x(p + 1),  shape_y(p + 1),  shape_z(p + 1),  shape_l(p + 1);
   Vector dshape_x(p + 1), dshape_y(p + 1), dshape_z(p + 1), dshape_l(p + 1);
   DenseMatrix du(dof, dim);
#endif

   poly1d.CalcBasis(p, ip.x, shape_x, dshape_x);
   poly1d.CalcBasis(p, ip.y, shape_y, dshape_y);
   poly1d.CalcBasis(p, ip.z, shape_z, dshape_z);
   poly1d.CalcBasis(p, 1. - ip.x - ip.y - ip.z, shape_l, dshape_l);

   for (int o = 0, k = 0; k <= p; k++)
      for (int j = 0; j + k <= p; j++)
         for (int i = 0; i + j + k <= p; i++)
         {
            int l = p - i - j - k;
            du(o,0) = ((dshape_x(i)* shape_l(l)) -
                       ( shape_x(i)*dshape_l(l)))*shape_y(j)*shape_z(k);
            du(o,1) = ((dshape_y(j)* shape_l(l)) -
                       ( shape_y(j)*dshape_l(l)))*shape_x(i)*shape_z(k);
            du(o,2) = ((dshape_z(k)* shape_l(l)) -
                       ( shape_z(k)*dshape_l(l)))*shape_x(i)*shape_y(j);
            o++;
         }

   Ti.Mult(du, dshape);
}

void H1_TetrahedronElement::CalcHessian(const IntegrationPoint &ip,
                                        DenseMatrix &ddshape) const
{
   const int p = order;

#ifdef MFEM_THREAD_SAFE
   Vector   shape_x(p + 1),   shape_y(p + 1),   shape_z(p + 1),   shape_l(p + 1);
   Vector  dshape_x(p + 1),  dshape_y(p + 1),  dshape_z(p + 1),  dshape_l(p + 1);
   Vector ddshape_x(p + 1), ddshape_y(p + 1), ddshape_z(p + 1), ddshape_l(p + 1);
   DenseMatrix ddu(dof, ((dim + 1) * dim) / 2);
#endif

   poly1d.CalcBasis(p, ip.x, shape_x, dshape_x, ddshape_x);
   poly1d.CalcBasis(p, ip.y, shape_y, dshape_y, ddshape_y);
   poly1d.CalcBasis(p, ip.z, shape_z, dshape_z, ddshape_z);
   poly1d.CalcBasis(p, 1. - ip.x - ip.y - ip.z, shape_l, dshape_l, ddshape_l);

   for (int o = 0, k = 0; k <= p; k++)
      for (int j = 0; j + k <= p; j++)
         for (int i = 0; i + j + k <= p; i++)
         {
            // u_xx, u_xy, u_xz, u_yy, u_yz, u_zz
            int l = p - i - j - k;
            ddu(o,0) = ((ddshape_x(i) * shape_l(l)) - 2. * (dshape_x(i) * dshape_l(l)) +
                        (shape_x(i) * ddshape_l(l))) * shape_y(j) * shape_z(k);
            ddu(o,1) = ((dshape_y(j) * ((dshape_x(i) * shape_l(l)) -
                                        (shape_x(i) * dshape_l(l)))) +
                        (shape_y(j) * ((ddshape_l(l) * shape_x(i)) -
                                       (dshape_x(i) * dshape_l(l)))))* shape_z(k);
            ddu(o,2) = ((dshape_z(k) * ((dshape_x(i) * shape_l(l)) -
                                        (shape_x(i) * dshape_l(l)))) +
                        (shape_z(k) * ((ddshape_l(l) * shape_x(i)) -
                                       (dshape_x(i) * dshape_l(l)))))* shape_y(j);
            ddu(o,3) = ((ddshape_y(j) * shape_l(l)) - 2. * (dshape_y(j) * dshape_l(l)) +
                        (shape_y(j) * ddshape_l(l))) * shape_x(i) * shape_z(k);
            ddu(o,4) = ((dshape_z(k) * ((dshape_y(j) * shape_l(l)) -
                                        (shape_y(j)*dshape_l(l))) ) +
                        (shape_z(k)* ((ddshape_l(l)*shape_y(j)) -
                                      (dshape_y(j) * dshape_l(l)) ) ) )* shape_x(i);
            ddu(o,5) = ((ddshape_z(k) * shape_l(l)) - 2. * (dshape_z(k) * dshape_l(l)) +
                        (shape_z(k) * ddshape_l(l))) * shape_y(j) * shape_x(i);
            o++;
         }
   Ti.Mult(ddu, ddshape);
}

// TODO: use a FunctionSpace specific to wedges instead of Qk.
H1_WedgeElement::H1_WedgeElement(const int p,
                                 const int btype)
   : NodalFiniteElement(3, Geometry::PRISM, ((p + 1)*(p + 1)*(p + 2))/2,
                        p, FunctionSpace::Qk),
     TriangleFE(p, btype),
     SegmentFE(p, btype)
{
#ifndef MFEM_THREAD_SAFE
   t_shape.SetSize(TriangleFE.GetDof());
   s_shape.SetSize(SegmentFE.GetDof());
   t_dshape.SetSize(TriangleFE.GetDof(), 2);
   s_dshape.SetSize(SegmentFE.GetDof(), 1);
#endif

   t_dof.SetSize(dof);
   s_dof.SetSize(dof);

   int p2p3 = 2*p + 3, ntri = ((p + 1)*(p + 2))/2;
   auto idx = [p2p3,ntri](int i, int j, int k)
   {
      return k*ntri + ((p2p3-j)*j)/2+i;
   };

   lex_ordering.SetSize(dof);
   int o = 0;

   // Nodal DoFs
   lex_ordering[idx(0,0,0)] = o++;
   lex_ordering[idx(p,0,0)] = o++;
   lex_ordering[idx(0,p,0)] = o++;
   lex_ordering[idx(0,0,p)] = o++;
   lex_ordering[idx(p,0,p)] = o++;
   lex_ordering[idx(0,p,p)] = o++;
   t_dof[0] = 0; s_dof[0] = 0;
   t_dof[1] = 1; s_dof[1] = 0;
   t_dof[2] = 2; s_dof[2] = 0;
   t_dof[3] = 0; s_dof[3] = 1;
   t_dof[4] = 1; s_dof[4] = 1;
   t_dof[5] = 2; s_dof[5] = 1;

   // Edge DoFs
   int k = 0;
   int ne = p-1;
   for (int i=1; i<p; i++)
   {
      lex_ordering[idx(i,0,0)] = o + 0*ne + k;
      lex_ordering[idx(p-i,i,0)] = o + 1*ne + k;
      lex_ordering[idx(0,p-i,0)] = o + 2*ne + k;
      lex_ordering[idx(i,0,p)] = o + 3*ne + k;
      lex_ordering[idx(p-i,i,p)] = o + 4*ne + k;
      lex_ordering[idx(0,p-i,p)] = o + 5*ne + k;
      lex_ordering[idx(0,0,i)] = o + 6*ne + k;
      lex_ordering[idx(p,0,i)] = o + 7*ne + k;
      lex_ordering[idx(0,p,i)] = o + 8*ne + k;
      t_dof[5 + 0 * ne + i] = 2 + 0 * ne + i; s_dof[5 + 0 * ne + i] = 0;
      t_dof[5 + 1 * ne + i] = 2 + 1 * ne + i; s_dof[5 + 1 * ne + i] = 0;
      t_dof[5 + 2 * ne + i] = 2 + 2 * ne + i; s_dof[5 + 2 * ne + i] = 0;
      t_dof[5 + 3 * ne + i] = 2 + 0 * ne + i; s_dof[5 + 3 * ne + i] = 1;
      t_dof[5 + 4 * ne + i] = 2 + 1 * ne + i; s_dof[5 + 4 * ne + i] = 1;
      t_dof[5 + 5 * ne + i] = 2 + 2 * ne + i; s_dof[5 + 5 * ne + i] = 1;
      t_dof[5 + 6 * ne + i] = 0;              s_dof[5 + 6 * ne + i] = i + 1;
      t_dof[5 + 7 * ne + i] = 1;              s_dof[5 + 7 * ne + i] = i + 1;
      t_dof[5 + 8 * ne + i] = 2;              s_dof[5 + 8 * ne + i] = i + 1;
      ++k;
   }
   o += 9*ne;

   // Triangular Face DoFs
   k=0;
   int nt = (p-1)*(p-2)/2;
   for (int j=1; j<p; j++)
   {
      for (int i=1; i<p-j; i++)
      {
         int l = j - p + (((2 * p - 1) - i) * i) / 2;
         lex_ordering[idx(i,j,0)] = o+l;
         lex_ordering[idx(i,j,p)] = o+nt+k;
         t_dof[6 + 9 * ne + k]      = 3 * p + l; s_dof[6 + 9 * ne + k]      = 0;
         t_dof[6 + 9 * ne + nt + k] = 3 * p + k; s_dof[6 + 9 * ne + nt + k] = 1;
         k++;
      }
   }
   o += 2*nt;

   // Quadrilateral Face DoFs
   k=0;
   int nq = (p-1)*(p-1);
   for (int j=1; j<p; j++)
   {
      for (int i=1; i<p; i++)
      {
         lex_ordering[idx(i,0,j)] = o+k;
         lex_ordering[idx(p-i,i,j)] = o+nq+k;
         lex_ordering[idx(0,p-i,j)] = o+2*nq+k;

         t_dof[6 + 9 * ne + 2 * nt + 0 * nq + k] = 2 + 0 * ne + i;
         t_dof[6 + 9 * ne + 2 * nt + 1 * nq + k] = 2 + 1 * ne + i;
         t_dof[6 + 9 * ne + 2 * nt + 2 * nq + k] = 2 + 2 * ne + i;

         s_dof[6 + 9 * ne + 2 * nt + 0 * nq + k] = 1 + j;
         s_dof[6 + 9 * ne + 2 * nt + 1 * nq + k] = 1 + j;
         s_dof[6 + 9 * ne + 2 * nt + 2 * nq + k] = 1 + j;

         k++;
      }
   }
   o += 3*nq;

   // Interior DoFs
   int m=0;
   for (k=1; k<p; k++)
   {
      int l=0;
      for (int j=1; j<p; j++)
      {
         for (int i=1; i+j<p; i++)
         {
            lex_ordering[idx(i,j,k)] = o++;
            t_dof[6 + 9 * ne + 2 * nt + 3 * nq + m] = 3 * p + l;
            s_dof[6 + 9 * ne + 2 * nt + 3 * nq + m] = 1 + k;
            l++; m++;
         }
      }
   }

   // Define Nodes
   const IntegrationRule & t_Nodes = TriangleFE.GetNodes();
   const IntegrationRule & s_Nodes = SegmentFE.GetNodes();
   for (int i=0; i<dof; i++)
   {
      Nodes.IntPoint(i).x = t_Nodes.IntPoint(t_dof[i]).x;
      Nodes.IntPoint(i).y = t_Nodes.IntPoint(t_dof[i]).y;
      Nodes.IntPoint(i).z = s_Nodes.IntPoint(s_dof[i]).x;
   }
}

void H1_WedgeElement::CalcShape(const IntegrationPoint &ip,
                                Vector &shape) const
{
#ifdef MFEM_THREAD_SAFE
   Vector t_shape(TriangleFE.GetDof());
   Vector s_shape(SegmentFE.GetDof());
#endif

   IntegrationPoint ipz; ipz.x = ip.z; ipz.y = 0.0; ipz.z = 0.0;

   TriangleFE.CalcShape(ip, t_shape);
   SegmentFE.CalcShape(ipz, s_shape);

   for (int i=0; i<dof; i++)
   {
      shape[i] = t_shape[t_dof[i]] * s_shape[s_dof[i]];
   }
}

void H1_WedgeElement::CalcDShape(const IntegrationPoint &ip,
                                 DenseMatrix &dshape) const
{
#ifdef MFEM_THREAD_SAFE
   Vector      t_shape(TriangleFE.GetDof());
   DenseMatrix t_dshape(TriangleFE.GetDof(), 2);
   Vector      s_shape(SegmentFE.GetDof());
   DenseMatrix s_dshape(SegmentFE.GetDof(), 1);
#endif

   IntegrationPoint ipz; ipz.x = ip.z; ipz.y = 0.0; ipz.z = 0.0;

   TriangleFE.CalcShape(ip, t_shape);
   TriangleFE.CalcDShape(ip, t_dshape);
   SegmentFE.CalcShape(ipz, s_shape);
   SegmentFE.CalcDShape(ipz, s_dshape);

   for (int i=0; i<dof; i++)
   {
      dshape(i, 0) = t_dshape(t_dof[i],0) * s_shape[s_dof[i]];
      dshape(i, 1) = t_dshape(t_dof[i],1) * s_shape[s_dof[i]];
      dshape(i, 2) = t_shape[t_dof[i]] * s_dshape(s_dof[i],0);
   }
}

H1_FuentesPyramidElement::H1_FuentesPyramidElement(const int p, const int btype)
   : NodalFiniteElement(3, Geometry::PYRAMID,
			p * (p * p + 3) + 1, // Fuentes et al
                        p, FunctionSpace::Qk)
{
   const double *cp = poly1d.ClosedPoints(p, VerifyNodal(VerifyClosed(btype)));

#ifndef MFEM_THREAD_SAFE
   /*
   shape_x.SetSize(p + 1);
   shape_y.SetSize(p + 1);
   shape_z.SetSize(p + 1);
   dshape_x.SetSize(p + 1);
   dshape_y.SetSize(p + 1);
   dshape_z.SetSize(p + 1);
   ddshape_x.SetSize(p + 1);
   ddshape_y.SetSize(p + 1);
   ddshape_z.SetSize(p + 1);
   u.SetSize(dof);
   du.SetSize(dof, dim);
   ddu.SetSize(dof, (dim * (dim + 1)) / 2);
   */
   shape_0.SetSize(p + 1);
   shape_1.SetSize(p + 1);
   shape_2.SetSize(p + 1);
   dshape_0_0.SetSize(p + 1);
   dshape_1_0.SetSize(p + 1);
   dshape_2_0.SetSize(p + 1);
   dshape_0_1.SetSize(p + 1);
   dshape_1_1.SetSize(p + 1);
   dshape_2_1.SetSize(p + 1);
   u.SetSize(dof);
   du.SetSize(dof, dim);
#else
   Vector shape_x(p + 1), shape_y(p + 1), shape_z(p + 1);
   /*
   Vector shape_0(p + 1);
   Vector shape_1(p + 1);
   Vector shape_2(p + 1);
   Vector dshape_0_0(p + 1);
   Vector dshape_1_0(p + 1);
   Vector dshape_2_0(p + 1);
   Vector dshape_0_1(p + 1);
   Vector dshape_1_1(p + 1);
   Vector dshape_2_1(p + 1);
   */
#endif

   // vertices
   Nodes.IntPoint(0).Set3(cp[0], cp[0], cp[0]);
   Nodes.IntPoint(1).Set3(cp[p], cp[0], cp[0]);
   Nodes.IntPoint(2).Set3(cp[p], cp[p], cp[0]);
   Nodes.IntPoint(3).Set3(cp[0], cp[p], cp[0]);
   Nodes.IntPoint(4).Set3(cp[0], cp[0], cp[p]);

   // edges
   int o = 5;
   for (int i = 1; i < p; i++)  // (0,1)
   {
      Nodes.IntPoint(o++).Set3(cp[i], cp[0], cp[0]);
   }
   for (int i = 1; i < p; i++)  // (1,2)
   {
      Nodes.IntPoint(o++).Set3(cp[p], cp[i], cp[0]);
   }
   for (int i = 1; i < p; i++)  // (3,2)
   {
      Nodes.IntPoint(o++).Set3(cp[i], cp[p], cp[0]);
   }
   for (int i = 1; i < p; i++)  // (0,3)
   {
      Nodes.IntPoint(o++).Set3(cp[0], cp[i], cp[0]);
   }
   for (int i = 1; i < p; i++)  // (0,4)
   {
      Nodes.IntPoint(o++).Set3(cp[0], cp[0], cp[i]);
   }
   for (int i = 1; i < p; i++)  // (1,4)
   {
      Nodes.IntPoint(o++).Set3(cp[p-i], cp[0], cp[i]);
   }
   for (int i = 1; i < p; i++)  // (2,4)
   {
      Nodes.IntPoint(o++).Set3(cp[p-i], cp[p-i], cp[i]);
   }
   for (int i = 1; i < p; i++)  // (3,4)
   {
      Nodes.IntPoint(o++).Set3(cp[0], cp[p-i], cp[i]);
   }

   // quadrilateral face
   for (int j = 1; j < p; j++)
   {
      for (int i = 1; i < p; i++)
      {
         Nodes.IntPoint(o++).Set3(cp[i], cp[j], cp[0]);
      }
   }

   // triangular faces
   for (int j = 1; j < p; j++)
      for (int i = 1; i + j < p; i++)  // (0,1,4)
      {
         double w = cp[i] + cp[j] + cp[p-i-j];
         Nodes.IntPoint(o++).Set3(cp[i]/w, cp[0], cp[j]/w);
         // mfem::out << i << " " << j << "\t" << cp[i]/w << " " << cp[0] << " " << cp[j]/w << std::endl;
      }
   for (int j = 1; j < p; j++)
      for (int i = 1; i + j < p; i++)  // (1,2,4)
      {
         double w = cp[i] + cp[j] + cp[p-i-j];
         Nodes.IntPoint(o++).Set3(1.0 - cp[j]/w, cp[i]/w, cp[j]/w);
         // mfem::out << i << " " << j << "\t" << 1.0 - cp[j]/w << " " << cp[i]/w << " " << cp[j]/w << std::endl;
      }
   for (int j = 1; j < p; j++)
      for (int i = 1; i + j < p; i++)  // (3,4,2)
      {
         double w = cp[i] + cp[j] + cp[p-i-j];
         Nodes.IntPoint(o++).Set3(cp[j]/w, 1.0 - cp[i]/w, cp[i]/w);
         // mfem::out << i << " " << j << "\t" << cp[j]/w << " " << 1.0 - cp[i]/w << " " << cp[i]/w << std::endl;
      }
   for (int j = 1; j < p; j++)
      for (int i = 1; i + j < p; i++)  // (0,4,3)
      {
         double w = cp[i] + cp[j] + cp[p-i-j];
         Nodes.IntPoint(o++).Set3(cp[0], cp[j]/w, cp[i]/w);
      }
   /*
   mfem::out << "pts342 = {";
   for (int j = 0; j <= p; j++)
   {
     for (int i = 0; i + j <= p; i++)  // (3,4,2)
     {
       double w = cp[i] + cp[j] + cp[p-i-j];
       mfem::out << "{" << cp[j] / w
       << "," << 1.0 - cp[i] / w
       << "," << cp[i] / w << "}";
       if (i + j != p) mfem::out << ",";
     }
     if (j != p) mfem::out << ",";
     mfem::out << std::endl;
   }
   mfem::out << "};\npts043 = {";
   for (int j = 0; j <= p; j++)
   {
     for (int i = 0; i + j <= p; i++)  // (0,4,3)
     {
       double w = cp[i] + cp[j] + cp[p-i-j];
       mfem::out << "{" << cp[0]
       << "," << cp[j] / w
       << "," << cp[i] / w << "}";
       if (i + j != p) mfem::out << ",";
     }
     if (j != p) mfem::out << ",";
     mfem::out << std::endl;
   }
   mfem::out << "};\n";
   */
   
   // Points based on Fuentes' interior bubbles
   // mfem::out << "pts = {";
   for (int k = 1; k < p; k++)
   {
      for (int j = 1; j < p; j++)
      {
         // double wjk = cp[j] + cp[k] + cp[p-j-k];
         for (int i = 1; i < p; i++)
         {
            // double wik = cp[i] + cp[k] + cp[p-i-k];
            // double w = wik * wjk;
            // mfem::out << "{" << cp[i] * (1.0 - cp[k])
            //         << "," << cp[j] * (1.0 - cp[k])
            //         << "," << cp[k] << "}";
            // if (i != p - 1) mfem::out << ",";
            Nodes.IntPoint(o++).Set3(cp[i] * (1.0 - cp[k]),
                                     cp[j] * (1.0 - cp[k]),
                                     cp[k]);
         }
         // if (j != p - 1) mfem::out << ",";
         // mfem::out << std::endl;
      }
      // if (k != p - 1) mfem::out << ",";
   }
   // mfem::out << "};\n";

   MFEM_ASSERT(o == dof,
               "Number of nodes does not match the "
               "number of degrees of freedom");
   DenseMatrix T(dof);

   for (int m = 0; m < dof; m++)
   {
      const IntegrationPoint &ip = Nodes.IntPoint(m);
      calcBasis(order, ip, shape_0, shape_1, shape_2, T.GetColumn(m));
   }

   Ti.Factor(T);

   if (false)
   {
      calcScaledLegendre(6, 0.3, 0.7, shape_0, dshape_0_0, dshape_0_1);
      mfem::out << "Scaled Legendre: ";
      for (int i=0; i<6; i++) { mfem::out << '\t' << shape_0[i]; }
      mfem::out << '\n';

      mfem::out << "Scaled Legendre du/dx:\n";
      for (int i=0; i<6; i++) { mfem::out << '\t' << dshape_0_0[i]; }
      mfem::out << '\n';

      double dx = 1e-8;
      calcScaledLegendre(6, 0.3+dx, 0.7, shape_2);
      mfem::out << "Scaled Legendre (x+dx): ";
      for (int i=0; i<6; i++) { mfem::out << '\t' << shape_2[i]; }
      mfem::out << '\n';
      mfem::out << "Scaled Legendre (u(x+dx) - u(x)) / dx:\n";
      for (int i=0; i<6; i++) { mfem::out << '\t' << (shape_2[i] - shape_0[i]) / dx; }
      mfem::out << '\n';

      mfem::out << "Scaled Legendre du/dt:\n";
      for (int i=0; i<6; i++) { mfem::out << '\t' << dshape_0_1[i]; }
      mfem::out << '\n';

      double dt = 1e-8;
      calcScaledLegendre(6, 0.3, 0.7+dt, shape_2);
      mfem::out << "Scaled Legendre (t+dt): ";
      for (int i=0; i<6; i++) { mfem::out << '\t' << shape_2[i]; }
      mfem::out << '\n';
      mfem::out << "Scaled Legendre (u(t+dt) - u(t)) / dt:\n";
      for (int i=0; i<6; i++) { mfem::out << '\t' << (shape_2[i] - shape_0[i]) / dt; }
      mfem::out << '\n';
   }
}

void H1_FuentesPyramidElement::CalcShape(const IntegrationPoint &ip,
                                  Vector &shape) const
{
   const int p = order;

#ifdef MFEM_THREAD_SAFE
   Vector shape_0(order+1);
   Vector shape_1(order+1);
   Vector shape_2(order+1);
   /*
   Vector shape_x(order+1);
   Vector shape_y(order+1);
   Vector shape_z(order+1);
   */
   Vector u(dof);
#endif

   calcBasis(p, ip, shape_0, shape_1, shape_2, u);

   Ti.Mult(u, shape);
}

void H1_FuentesPyramidElement::CalcDShape(const IntegrationPoint &ip,
                                   DenseMatrix &dshape) const
{
   const int p = order;

#ifdef MFEM_THREAD_SAFE
   Vector  shape_0(p + 1), shape_1(p + 1), shape_2(p + 1);
   Vector dshape_0_0(p + 1), dshape_1_0(p + 1), dshape_2_0(p + 1);
   Vector dshape_0_1(p + 1), dshape_1_1(p + 1), dshape_2_1(p + 1);
   DenseMatrix du(dof, dim);
#endif

   double dlam[3];
   double dlam4[3];

   double dnu0[2];
   double dnu1[2];
   double dnu2[2];

   double x = ip.x;
   double y = ip.y;
   double z = ip.z;

   int o = 0;

   // Vertices
   grad_lam0(x, y, z, dlam);
   for (int i = 0; i < 3; i++) { du(0, i) = dlam[i]; }
   grad_lam1(x, y, z, dlam);
   for (int i = 0; i < 3; i++) { du(1, i) = dlam[i]; }
   grad_lam2(x, y, z, dlam);
   for (int i = 0; i < 3; i++) { du(2, i) = dlam[i]; }
   grad_lam3(x, y, z, dlam);
   for (int i = 0; i < 3; i++) { du(3, i) = dlam[i]; }
   grad_lam4(x, y, z, dlam);
   for (int i = 0; i < 3; i++) { du(4, i) = dlam[i]; }
   o += 5;

   // Mixed edges (base edges)
   phi_E(p, nu0(x, z), nu1(x, z), shape_0, dshape_0_0, dshape_0_1);
   grad_nu0(x, z, dnu0);
   grad_nu1(x, z, dnu1);
   for (int i = 2; i <= p; i++, o++)
   {
      // Grad(mu0(y / (1.0 - z)) phi_E(nu0(x, z), nu1(x, z)))
      du(o, 0) = mu0(y / (1.0 - z)) *
                 (dshape_0_0[i] * dnu0[0] + dshape_0_1[i] * dnu1[0]);
      du(o, 1) = dmu0(y / (1.0 - z)) * shape_0[i] / (1.0 - z);
      du(o, 2) = dmu0(y / (1.0 - z)) * shape_0[i] / pow(1.0 - z, 2) +
                 mu0(y / (1.0 - z)) *
                 (dshape_0_0[i] * dnu0[1] + dshape_0_1[i] * dnu1[1]);
   }
   phi_E(p, nu0(y, z), nu1(y, z), shape_0, dshape_0_0, dshape_0_1);
   grad_nu0(y, z, dnu0);
   grad_nu1(y, z, dnu1);
   for (int i = 2; i <= p; i++, o++)
   {
      // Grad(mu1(x / (1.0 - z)) * phi_E(nu0(y, z), nu1(y, z)))
      du(o, 0) = dmu1(x / (1.0 - z)) * shape_0[i] / (1.0 - z);
      du(o, 1) = mu1(x / (1.0 - z)) *
                 (dshape_0_0[i] * dnu0[0] + dshape_0_1[i] * dnu1[0]);
      du(o, 2) = dmu1(x / (1.0 - z)) * shape_0[i] / pow(1.0 - z, 2) +
                 mu1(x / (1.0 - z)) *
                 (dshape_0_0[i] * dnu0[1] + dshape_0_1[i] * dnu1[1]);
   }
   phi_E(p, nu0(x, z), nu1(x, z), shape_0, dshape_0_0, dshape_0_1);
   grad_nu0(x, z, dnu0);
   grad_nu1(x, z, dnu1);
   for (int i = 2; i <= p; i++, o++)
   {
      // Grad(mu1(y / (1.0 - z)) * phi_E(nu0(x, z), nu1(x, z)))
      du(o, 0) = mu1(y / (1.0 - z)) *
                 (dshape_0_0[i] * dnu0[0] + dshape_0_1[i] * dnu1[0]);
      du(o, 1) = dmu1(y / (1.0 - z)) * shape_0[i] / (1.0 - z);
      du(o, 2) = dmu1(y / (1.0 - z)) * shape_0[i] / pow(1.0 - z, 2) +
                 mu1(y / (1.0 - z)) *
                 (dshape_0_0[i] * dnu0[1] + dshape_0_1[i] * dnu1[1]);
   }
   phi_E(p, nu0(y, z), nu1(y, z), shape_0, dshape_0_0, dshape_0_1);
   grad_nu0(y, z, dnu0);
   grad_nu1(y, z, dnu1);
   for (int i = 2; i <= p; i++, o++)
   {
      // Grad(mu0(x / (1.0 - z)) * phi_E(nu0(y, z), nu1(y, z)))
      du(o, 0) = dmu0(x / (1.0 - z)) * shape_0[i] / (1.0 - z);
      du(o, 1) = mu0(x / (1.0 - z)) *
                 (dshape_0_0[i] * dnu0[0] + dshape_0_1[i] * dnu1[0]);
      du(o, 2) = dmu0(x / (1.0 - z)) * shape_0[i] / pow(1.0 - z, 2) +
                 mu0(x / (1.0 - z)) *
                 (dshape_0_0[i] * dnu0[1] + dshape_0_1[i] * dnu1[1]);
   }

   // Triangle edges (upright edges)
   grad_lam4(x, y, z, dlam4);
   grad_lam0(x, y, z, dlam);
   phi_E(p, lam0(x, y, z), lam4(x, y, z), shape_0, dshape_0_0, dshape_0_1);
   for (int i = 2; i<= p; i++, o++)
   {
      // Grad(phi_E(lam0(x,y,z), lam4(x,y,z)))
      for (int j = 0; j < 3; j++)
      {
         du(o, j) = dshape_0_0[i] * dlam[j] + dshape_0_1[i] * dlam4[j];
      }
   }
   grad_lam1(x, y, z, dlam);
   phi_E(p, lam1(x, y, z), lam4(x, y, z), shape_0, dshape_0_0, dshape_0_1);
   for (int i = 2; i<= p; i++, o++)
   {
      // Grad(phi_E(lam1(x,y,z), lam4(x,y,z)))
      for (int j = 0; j < 3; j++)
      {
         du(o, j) = dshape_0_0[i] * dlam[j] + dshape_0_1[i] * dlam4[j];
      }
   }
   grad_lam2(x, y, z, dlam);
   phi_E(p, lam2(x, y, z), lam4(x, y, z), shape_0, dshape_0_0, dshape_0_1);
   for (int i = 2; i<= p; i++, o++)
   {
      // Grad(phi_E(lam2(x,y,z), lam4(x,y,z)))
      for (int j = 0; j < 3; j++)
      {
         du(o, j) = dshape_0_0[i] * dlam[j] + dshape_0_1[i] * dlam4[j];
      }
   }
   grad_lam3(x, y, z, dlam);
   phi_E(p, lam3(x, y, z), lam4(x, y, z), shape_0, dshape_0_0, dshape_0_1);
   for (int i = 2; i<= p; i++, o++)
   {
      // Grad(phi_E(lam3(x,y,z), lam4(x,y,z)))
      for (int j = 0; j < 3; j++)
      {
         du(o, j) = dshape_0_0[i] * dlam[j] + dshape_0_1[i] * dlam4[j];
      }
   }

   // Quadrilateral face
   phi_E(p, mu0(x / (1.0 - z)), mu1(x / (1.0 - z)), shape_0,
         dshape_0_0, dshape_0_1);
   phi_E(p, mu0(y / (1.0 - z)), mu1(y / (1.0 - z)), shape_1,
         dshape_1_0, dshape_1_1);
   for (int j = 2; j <= p; j++)
   {
      for (int i = 2; i <= p; i++, o++)
      {
         // Grad(mu0(z) * phi_E(mu0(x / (1.0 - z)), mu1(x / (1.0 - z)))
         //             * phi_E(mu0(y / (1.0 - z)), mu1(y / (1.0 - z))))
         du(o, 0) = mu0(z) * (dshape_0_0[i] * dmu0(x / (1.0 - z)) +
                              dshape_0_1[i] * dmu1(x / (1.0 - z))) * shape_1[j]
                    / (1.0 - z);
         du(o, 1) = mu0(z) * shape_0[i] * (dshape_1_0[i] * dmu0(y / (1.0 - z)) +
                                           dshape_1_1[i] * dmu1(y / (1.0 - z)))
                    / (1.0 - z);
         du(o, 2) = dmu0(z) * shape_0[i] * shape_1[j] +
                    mu0(z) * ((dshape_0_0[i] * dmu0(x / (1.0 - z)) +
                               dshape_0_1[i] * dmu1(x / (1.0 - z))) * x * shape_1[j] +
                              shape_0[i] * (dshape_1_0[i] * dmu0(y / (1.0 - z)) +
                                            dshape_1_1[i] * dmu1(y / (1.0 - z))) * y)
                    / pow(1.0 - z, 2);
      }
   }

   // Triangular faces
   phi_E(p, nu0(x, z), nu1(x, z), shape_0, dshape_0_0, dshape_0_1);
   grad_nu0(x, z, dnu0);
   grad_nu1(x, z, dnu1);
   grad_nu2(x, z, dnu2);
   for (int i = 2; i <= p; i++)
   {
      calcIntegratedJacobi(p, 2.0 * i, nu2(x, z), 1.0, shape_1,
                           dshape_1_0, dshape_1_1);
      for (int j = 1; j <= p - i; j++, o++)
      {
         // u[o] = mu0(y / (1.0 - z)) * tmp_x[i] * tmp_y[j];
         du(o, 0) = mu0(y / (1.0 - z)) *
                    ((dshape_0_0[i] * dnu0[0] + dshape_0_1[i] * dnu1[0]) * shape_1[j] +
                     shape_0[i] * dshape_1_0[j] * dnu2[0]);
         du(o, 1) = dmu0(y / (1.0 - z)) * shape_0[i] * shape_1[j] / (1.0 - z);
         du(o, 2) = dmu0(y / (1.0 - z)) * shape_0[i] * shape_1[j]
                    / pow(1.0 - z, 2) +
                    mu0(y / (1.0 - z)) *
                    ((dshape_0_0[i] * dnu0[1] + dshape_0_1[i] * dnu1[1]) * shape_1[j] +
                     shape_0[i] * dshape_1_0[i] * dnu2[1]);
      }
   }
   phi_E(p, nu0(y, z), nu1(y, z), shape_0, dshape_0_0, dshape_0_1);
   grad_nu0(y, z, dnu0);
   grad_nu1(y, z, dnu1);
   grad_nu2(y, z, dnu2);
   for (int i = 2; i <= p; i++)
   {
      calcIntegratedJacobi(p, 2.0 * i, nu2(y, z), 1.0, shape_1,
                           dshape_1_0, dshape_1_1);
      for (int j = 1; j <= p - i; j++, o++)
      {
         // u[o] = mu1(x / (1.0 - z)) * tmp_x[i] * tmp_y[j];
         du(o, 0) = dmu1(x / (1.0 - z)) * shape_0[i] * shape_1[j] / (1.0 - z);
         du(o, 1) = mu1(x / (1.0 - z)) *
                    ((dshape_0_0[i] * dnu0[0] + dshape_0_1[i] * dnu1[0]) * shape_1[j] +
                     shape_0[i] * dshape_1_0[j] * dnu2[0]);
         du(o, 2) = dmu1(x / (1.0 - z)) * shape_0[i] * shape_1[j]
                    / pow(1.0 - z, 2) +
                    mu1(x / (1.0 - z)) *
                    ((dshape_0_0[i] * dnu0[1] + dshape_0_1[i] * dnu1[1]) * shape_1[j] +
                     shape_0[i] * dshape_1_0[i] * dnu2[1]);
      }
   }
   phi_E(p, nu0(x, z), nu1(x, z), shape_0, dshape_0_0, dshape_0_1);
   grad_nu0(x, z, dnu0);
   grad_nu1(x, z, dnu1);
   grad_nu2(x, z, dnu2);
   for (int i = 2; i <= p; i++)
   {
      calcIntegratedJacobi(p, 2.0 * i, nu2(x, z), 1.0, shape_1,
                           dshape_1_0, dshape_1_1);
      for (int j = 1; j <= p - i; j++, o++)
      {
         // u[o] = mu1(y / (1.0 - z)) * tmp_x[i] * tmp_y[j];
         du(o, 0) = mu1(y / (1.0 - z)) *
                    ((dshape_0_0[i] * dnu0[0] + dshape_0_1[i] * dnu1[0]) * shape_1[j] +
                     shape_0[i] * dshape_1_0[j] * dnu2[0]);
         du(o, 1) = dmu1(y / (1.0 - z)) * shape_0[i] * shape_1[j] / (1.0 - z);
         du(o, 2) = dmu1(y / (1.0 - z)) * shape_0[i] * shape_1[j]
                    / pow(1.0 - z, 2) +
                    mu1(y / (1.0 - z)) *
                    ((dshape_0_0[i] * dnu0[1] + dshape_0_1[i] * dnu1[1]) * shape_1[j] +
                     shape_0[i] * dshape_1_0[i] * dnu2[1]);
      }
   }
   phi_E(p, nu0(y, z), nu1(y, z), shape_0, dshape_0_0, dshape_0_1);
   grad_nu0(y, z, dnu0);
   grad_nu1(y, z, dnu1);
   grad_nu2(y, z, dnu2);
   for (int i = 2; i <= p; i++)
   {
      calcIntegratedJacobi(p, 2.0 * i, nu2(y, z), 1.0, shape_1,
                           dshape_1_0, dshape_1_1);
      for (int j = 1; j <= p - i; j++, o++)
      {
         // u[o] = mu0(x / (1.0 - z)) * tmp_x[i] * tmp_y[j];
         du(o, 0) = dmu0(x / (1.0 - z)) * shape_0[i] * shape_1[j] / (1.0 - z);
         du(o, 1) = mu0(x / (1.0 - z)) *
                    ((dshape_0_0[i] * dnu0[0] + dshape_0_1[i] * dnu1[0]) * shape_1[j] +
                     shape_0[i] * dshape_1_0[j] * dnu2[0]);
         du(o, 2) = dmu0(x / (1.0 - z)) * shape_0[i] * shape_1[j]
                    / pow(1.0 - z, 2) +
                    mu0(x / (1.0 - z)) *
                    ((dshape_0_0[i] * dnu0[1] + dshape_0_1[i] * dnu1[1]) * shape_1[j] +
                     shape_0[i] * dshape_1_0[i] * dnu2[1]);
      }
   }

   // Interior
   phi_E(p, mu0(x / (1.0 - z)), mu1(x / (1.0 - z)), shape_0,
         dshape_0_0, dshape_0_1);
   phi_E(p, mu0(y / (1.0 - z)), mu1(y / (1.0 - z)), shape_1,
         dshape_1_0, dshape_1_1);
   phi_E(p, mu0(z), mu1(z), shape_2, dshape_2_0, dshape_2_1);
   for (int k = 2; k <= p; k++)
   {
      for (int j = 2; j <= p; j++)
      {
         for (int i = 2; i <= p; i++, o++)
         {
            du(o, 0) = (dshape_0_0[i] * dmu0(x / (1.0 - z)) +
                        dshape_0_1[i] * dmu1(x / (1.0 - z))) *
                       shape_1[j] * shape_2[k] / (1.0 - z);
            du(o, 1) = shape_0[i] * (dshape_1_0[j] * dmu0(y / (1.0 - z)) +
                                     dshape_1_1[j] * dmu1(y / (1.0 - z))) *
                       shape_2[k] / (1.0 - z);
            du(o, 2) = ((dshape_0_0[i] * dmu0(x / (1.0 - z)) +
                         dshape_0_1[i] * dmu1(x / (1.0 - z))) * shape_1[j] +
                        shape_0[i] * (dshape_1_0[j] * dmu0(y / (1.0 - z)) +
                                      dshape_1_1[j] * dmu1(y / (1.0 - z)))) *
                       shape_2[k] / pow(1.0 - z, 2) +
                       shape_0[i] * shape_1[j] * (dshape_2_0[k] * dmu0(z) +
                                                  dshape_2_1[k] * dmu1(z));
         }
      }
   }
   /*
   double x = ip.x;
   double y = ip.y;
   double z = ip.z;

   dshape.SetSize(5, 3);
   dshape(0,0) = -1.0 + ((z<1.0) ? (y / (1.0 - z)) : 0.0);
   dshape(0,1) = -1.0 + ((z<1.0) ? (x / (1.0 - z)) : 0.0);
   dshape(0,2) = -1.0 + ((z<1.0) ? (x * y / pow(1.0 - z, 2)) : 0.0);

   dshape(1,0) = 1.0 - ((z<1.0) ? (y / (1.0 - z)) : 0.0);
   dshape(1,1) = (z<1.0) ? (-x / (1.0 - z)) : 0.0;
   dshape(1,2) = (z<1.0) ? (-x * y / pow(1.0 - z, 2)) : 0.0;

   dshape(2,0) = (z<1.0) ? (y / (1.0 - z)) : 0.0;
   dshape(2,1) = (z<1.0) ? (x / (1.0 - z)) : 0.0;
   dshape(2,2) = (z<1.0) ? (x * y / pow(1.0 - z, 2)) : 0.0;

   dshape(3,0) = (z<1.0) ? (-y / (1.0 - z)) : 0.0;
   dshape(3,1) = 1.0 - ((z<1.0) ? (x / (1.0 - z)) : 0.0);
   dshape(3,2) = (z<1.0) ? (-x * y / pow(1.0 - z, 2)) : 0.0;

   dshape(4,0) = 0.0;
   dshape(4,1) = 0.0;
   dshape(4,2) = 1.0;
   */
   /*
   double x = (ip.z < 1.0) ? (ip.x / (1.0 - ip.z)) : 0.0;
   double y = (ip.z < 1.0) ? (ip.y / (1.0 - ip.z)) : 0.0;
   double z = ip.z;

   int o = 0;
   for (int i = 0; i <= p; i++)
   {
      poly1d.CalcLegendre(i, x, shape_x, dshape_x);
      for (int j = 0; j <= p; j++)
      {
    poly1d.CalcLegendre(j, y, shape_y, dshape_y);
    int maxij = std::max(i, j);
    for (int k = 0; k <= p - maxij; k++)
    {
      poly1d.CalcJacobi(k, 2.0 * (maxij + 1.0), 0.0, z,
                shape_z, dshape_z);
      du(o,0) = dshape_x(i) * shape_y(j) * shape_z(k) *
        pow(1.0 - ip.z, maxij - 1);
      du(o,1) = shape_x(i) * dshape_y(j) * shape_z(k) *
        pow(1.0 - ip.z, maxij - 1);
      du(o,2) = shape_x(i) * shape_y(j) * dshape_z(k) *
        pow(1.0 - ip.z, maxij) +
        (dshape_x(i) * shape_y(j) + shape_x(i) * dshape_y(j)) *
        shape_z(k) * pow(1.0 - ip.z, maxij - 2) -
        ((maxij > 0) ? (maxij * shape_x(i) * shape_y(j) * shape_z(k) *
              pow(1.0 - ip.z, maxij - 1)) : 0.0);
      o++;
    }
      }
   }
   */
   // calcDBasis(order, ip, du);
   Ti.Mult(du, dshape);
}

void H1_FuentesPyramidElement::calcBasis(const int p, const IntegrationPoint &ip,
                                  double * tmp_x, double * tmp_y,
                                  double * tmp_z, double *u)
{
   double x = ip.x;
   double y = ip.y;
   double z = ip.z;

   int o = 0;

   // Vertices
   u[0] = lam0(x, y, z);
   u[1] = lam1(x, y, z);
   u[2] = lam2(x, y, z);
   u[3] = lam3(x, y, z);
   u[4] = lam4(x, y, z);
   o += 5;

   // Mixed edges (base edges)
   if (z < 1.0)
   {
      phi_E(p, nu0(x, z), nu1(x, z), tmp_x);
      for (int i = 2; i <= p; i++, o++)
      {
         u[o] = mu0(y / (1.0 - z)) * tmp_x[i];
      }
      phi_E(p, nu0(y, z), nu1(y, z), tmp_x);
      for (int i = 2; i <= p; i++, o++)
      {
         u[o] = mu1(x / (1.0 - z)) * tmp_x[i];
      }
      phi_E(p, nu0(x, z), nu1(x, z), tmp_x);
      for (int i = 2; i <= p; i++, o++)
      {
         u[o] = mu1(y / (1.0 - z)) * tmp_x[i];
      }
      phi_E(p, nu0(y, z), nu1(y, z), tmp_x);
      for (int i = 2; i <= p; i++, o++)
      {
         u[o] = mu0(x / (1.0 - z)) * tmp_x[i];
      }
   }
   else
   {
      for (int i = 0; i < 4 * (p - 1); i++, o++)
      {
         u[o] = 0.0;
      }
   }

   // Triangle edges (upright edges)
   phi_E(p, lam0(x, y, z), lam4(x, y, z), tmp_x);
   for (int i = 2; i<= p; i++, o++)
   {
      u[o] = tmp_x[i];
   }
   phi_E(p, lam1(x, y, z), lam4(x, y, z), tmp_x);
   for (int i = 2; i<= p; i++, o++)
   {
      u[o] = tmp_x[i];
   }
   phi_E(p, lam2(x, y, z), lam4(x, y, z), tmp_x);
   for (int i = 2; i<= p; i++, o++)
   {
      u[o] = tmp_x[i];
   }
   phi_E(p, lam3(x, y, z), lam4(x, y, z), tmp_x);
   for (int i = 2; i<= p; i++, o++)
   {
      u[o] = tmp_x[i];
   }

   // Quadrilateral face
   if (z < 1.0)
   {
      phi_E(p, mu0(x / (1.0 - z)), mu1(x / (1.0 - z)), tmp_x);
      phi_E(p, mu0(y / (1.0 - z)), mu1(y / (1.0 - z)), tmp_y);
      for (int j = 2; j <= p; j++)
      {
         for (int i = 2; i <= p; i++, o++)
         {
            u[o] = mu0(z) * tmp_x[i] * tmp_y[j];
         }
      }
   }
   else
   {
      for (int j = 2; j <= p; j++)
      {
         for (int i = 2; i <= p; i++, o++)
         {
            u[o] = 0.0;
         }
      }
   }

   // Triangular faces
   if (z < 1.0)
   {
      phi_E(p, nu0(x, z), nu1(x, z), tmp_x);
      for (int i = 2; i <= p; i++)
      {
         calcIntegratedJacobi(p, 2.0 * i, nu2(x, z), 1.0, tmp_y);
         for (int j = 1; j <= p - i; j++, o++)
         {
            u[o] = mu0(y / (1.0 - z)) * tmp_x[i] * tmp_y[j];
         }
      }
      phi_E(p, nu0(y, z), nu1(y, z), tmp_x);
      for (int i = 2; i <= p; i++)
      {
         calcIntegratedJacobi(p, 2.0 * i, nu2(y, z), 1.0, tmp_y);
         for (int j = 1; j <= p - i; j++, o++)
         {
            u[o] = mu1(x / (1.0 - z)) * tmp_x[i] * tmp_y[j];
         }
      }
      phi_E(p, nu0(x, z), nu1(x, z), tmp_x);
      for (int i = 2; i <= p; i++)
      {
         calcIntegratedJacobi(p, 2.0 * i, nu2(x, z), 1.0, tmp_y);
         for (int j = 1; j <= p - i; j++, o++)
         {
            u[o] = mu1(y / (1.0 - z)) * tmp_x[i] * tmp_y[j];
         }
      }
      phi_E(p, nu0(y, z), nu1(y, z), tmp_x);
      for (int i = 2; i <= p; i++)
      {
         calcIntegratedJacobi(p, 2.0 * i, nu2(y, z), 1.0, tmp_y);
         for (int j = 1; j <= p - i; j++, o++)
         {
            u[o] = mu0(x / (1.0 - z)) * tmp_x[i] * tmp_y[j];
         }
      }
   }
   else
   {
      for (int i = 0; i < 2 * (p - 1) * (p - 2); i++, o++)
      {
         u[o] = 0.0;
      }
   }

   // Interior
   if (z < 1.0)
   {
      phi_E(p, mu0(x / (1.0 - z)), mu1(x / (1.0 - z)), tmp_x);
      phi_E(p, mu0(y / (1.0 - z)), mu1(y / (1.0 - z)), tmp_y);
      phi_E(p, mu0(z), mu1(z), tmp_z);
      for (int k = 2; k <= p; k++)
      {
         for (int j = 2; j <= p; j++)
         {
            for (int i = 2; i <= p; i++, o++)
            {
               u[o] = tmp_x[i] * tmp_y[j] * tmp_z[k];
            }
         }
      }
   }
   else
   {
      for (int i = 0; i < (p - 1) * (p - 1) * (p - 1); i++, o++)
      {
         u[o]= 0.0;
      }
   }
   /*
   if (p == 3)
   {
   if (z < 1.0)
   {
      phi_E(p-1, mu0(x / (1.0 - z)), mu1(x / (1.0 - z)), tmp_x);
      phi_E(p-1, mu0(y / (1.0 - z)), mu1(y / (1.0 - z)), tmp_y);
      phi_E(p-1, mu0(z), mu1(z), tmp_z);
      for (int k = 2; k <= p-1; k++)
      {
         for (int j = 2; j <= p-1; j++)
         {
            for (int i = 2; i <= p-1; i++, o++)
            {
               u[o] = tmp_x[i] * tmp_y[j] * tmp_z[k];
            }
         }
      }
   }
   else
   {
      for (int i = 0; i < (p - 2) * (p - 2) * (p - 2); i++, o++)
      {
         u[o]= 0.0;
      }
   }
   }
   */
}
/*
void H1_FuentesPyramidElement::calcDBasis(const int p, const IntegrationPoint &ip,
                                   DenseMatrix &du)
{
   du = 0.0;
}
*/
void H1_FuentesPyramidElement::grad_lam0(const double x, const double y,
                                  const double z, double du[])
{
   du[0] = (z < 1.0) ? - (1.0 - y - z) / (1.0 - z) : 0.0;
   du[1] = (z < 1.0) ? - (1.0 - x - z) / (1.0 - z) : 0.0;
   du[2] = (z < 1.0) ? x * y / ((1.0 - z) * (1.0 - z)) - 1.0 : 0.0;
}

void H1_FuentesPyramidElement::grad_lam1(const double x, const double y,
                                  const double z, double du[])
{
   du[0] = (z < 1.0) ? (1.0 - y - z) / (1.0 - z) : 0.0;
   du[1] = (z < 1.0) ? - x / (1.0 - z) : 0.0;
   du[2] = (z < 1.0) ? - x * y / ((1.0 - z) * (1.0 - z)) : 0.0;
}

void H1_FuentesPyramidElement::grad_lam2(const double x, const double y,
                                  const double z, double du[])
{
   du[0] = (z < 1.0) ? y / (1.0 - z) : 0.0;
   du[1] = (z < 1.0) ? x / (1.0 - z) : 0.0;
   du[2] = (z < 1.0) ? x * y / ((1.0 - z) * (1.0 - z)) : 0.0;
}

void H1_FuentesPyramidElement::grad_lam3(const double x, const double y,
                                  const double z, double du[])
{
   du[0] = (z < 1.0) ? - y / (1.0 - z) : 0.0;
   du[1] = (z < 1.0) ? (1.0 - x - z) / (1.0 - z) : 0.0;
   du[2] = (z < 1.0) ? - x * y / ((1.0 - z) * (1.0 - z)) : 0.0;
}

void H1_FuentesPyramidElement::grad_lam4(const double x, const double y,
                                  const double z, double du[])
{
   du[0] = 0.0;
   du[1] = 0.0;
   du[2] = 1.0;
}

void H1_FuentesPyramidElement::phi_E(const int p, const double s0, double s1,
                              double *u)
{
   calcIntegratedLegendre(p, s1, s0 + s1, u);
}

void H1_FuentesPyramidElement::phi_E(const int p, const double s0, double s1,
                              double *u, double *duds0, double *duds1)
{
   calcIntegratedLegendre(p, s1, s0 + s1, u, duds1, duds0);
   for (int i = 0; i <= p; i++) { duds1[i] += duds0[i]; }
}

void H1_FuentesPyramidElement::calcIntegratedLegendre(const int p, const double x,
                                               const double t,
                                               double *u)
{
   if (t > 0.0)
   {
      calcScaledLegendre(p, x, t, u);
      for (int i = p; i >= 2; i--)
      {
         u[i] = (u[i] - t * t * u[i-2]) / (4.0 * i - 2.0);
      }
      if (p >= 1)
      {
         u[1] = x;
      }
      u[0] = 0.0;
   }
   else
   {
      for (int i = 0; i <= p; i++)
      {
         u[i] = 0.0;
      }
   }
}

void H1_FuentesPyramidElement::calcIntegratedLegendre(const int p, const double x,
                                               const double t,
                                               double *u,
                                               double *dudx, double *dudt)
{
   if (t > 0.0)
   {
      calcScaledLegendre(p, x, t, u, dudx, dudt);
      for (int i = p; i >= 2; i--)
      {
         u[i] = (u[i] - t * t * u[i-2]) / (4.0 * i - 2.0);
         dudx[i] = (dudx[i] - t * t * dudx[i-2]) / (4.0 * i - 2.0);
         dudt[i] = (dudt[i] - t * t * dudt[i-2] - 2.0 * t * u[i-2]) /
                   (4.0 * i - 2.0);
      }
      if (p >= 1)
      {
         u[1] = x; dudx[1] = 1.0; dudt[1] = 0.0;
      }
      u[0] = 0.0; dudx[0] = 0.0; dudt[0] = 0.0;
   }
   else
   {
      for (int i = 0; i <= p; i++)
      {
         u[i] = 0.0;
         dudx[i] = 0.0;
         dudt[i] = 0.0;
      }
   }
}

/** Implements a scaled and shifted set of Legendre polynomials

      P_i(x / t) * t^i

   where t >= 0.0, x \in [0,t], and P_i is the shifted Legendre
   polynomial defined on [0,1] rather than the usual [-1,1].
*/
void H1_FuentesPyramidElement::calcScaledLegendre(const int p, const double x,
                                           const double t,
                                           double *u)
{
   if (t > 0.0)
   {
      Poly_1D::CalcLegendre(p, x / t, u);
      for (int i = 1; i <= p; i++)
      {
         u[i] *= pow(t, i);
      }
   }
   else
   {
      // This assumes x = 0 as well as t = 0 since x \in [0,t]
      u[0] = 1.0;
      for (int i = 1; i <= p; i++) { u[i] = 0.0; }
   }
}

void H1_FuentesPyramidElement::calcScaledLegendre(const int p, const double x,
                                           const double t,
                                           double *u,
                                           double *dudx, double *dudt)
{
   if (t > 0.0)
   {
      Poly_1D::CalcLegendre(p, x / t, u, dudx);
      dudx[0] = 0.0;
      dudt[0] = - dudx[0] * x / t;
      for (int i = 1; i <= p; i++)
      {
         u[i]    *= pow(t, i);
         dudx[i] *= pow(t, i - 1);
         dudt[i]  = (u[i] * i - dudx[i] * x) / t;
      }
   }
   else
   {
      // This assumes x = 0 as well as t = 0 since x \in [0,t]
      u[0]    = 1.0;
      dudx[0] = 0.0;
      dudt[0] = 0.0;
      if (p >=1)
      {
         u[1]    =  0.0;
         dudx[1] =  2.0;
         dudt[1] = -1.0;
      }
      for (int i = 2; i <= p; i++)
      {
         u[i] = 0.0;
         dudx[i] = 0.0;
         dudt[i] = 0.0;
      }
   }
}

void H1_FuentesPyramidElement::calcIntegratedJacobi(const int p,
                                             const double alpha,
                                             const double x,
                                             const double t,
                                             double *u)
{
   if (t > 0.0)
   {
      calcScaledJacobi(p, alpha, x, t, u);
      for (int i = p; i >= 2; i--)
      {
         double d0 = 2.0 * i + alpha;
         double d1 = d0 - 1.0;
         double d2 = d0 - 2.0;
         double a = (alpha + i) / (d0 * d1);
         double b = alpha / (d0 * d2);
         double c = (double)(i - 1) / (d1 * d2);
         u[i] = a * u[i] + b * t * u[i - 1] - c * t * t * u[i - 2];
      }
      if (p >= 1)
      {
         u[1] = x;
      }
      u[0] = 0.0;
   }
   else
   {
      u[0] = 1.0;
      for (int i = 1; i <= p; i++)
      {
         u[i] = 0.0;
      }
   }
}

void H1_FuentesPyramidElement::calcIntegratedJacobi(const int p,
                                             const double alpha,
                                             const double x,
                                             const double t,
                                             double *u,
                                             double *dudx,
                                             double *dudt)
{
   calcScaledJacobi(p, alpha, x, t, u, dudx, dudt);
   for (int i = p; i >= 2; i--)
   {
      double d0 = 2.0 * i + alpha;
      double d1 = d0 - 1.0;
      double d2 = d0 - 2.0;
      double a = (alpha + i) / (d0 * d1);
      double b = alpha / (d0 * d2);
      double c = (double)(i - 1) / (d1 * d2);
      u[i]    = a * u[i] + b * t * u[i - 1] - c * t * t * u[i - 2];
      dudx[i] = a * dudx[i] + b * t * dudx[i - 1] - c * t * t * dudx[i - 2];
      dudt[i] = a * dudt[i] + b * t * dudt[i - 1] + b * u[i - 1]
                - c * t * t * dudt[i - 2] - 2.0 * c * t * u[i - 2];
   }
   if (p >= 1)
   {
      u[1]    = x;
      dudx[1] = 1.0;
      dudt[1] = 0.0;
   }
   u[0]    = 0.0;
   dudx[0] = 0.0;
   dudt[0] = 0.0;
}

/** Implements a set of scaled and shifted subset of Jacobi polynomials

      P_i^{\alpha, 0}(x / t) * t^i

   where t >= 0.0, x \in [0,t], and P_i^{\alpha, \beta} is the shifted Jacobi
   polynomial defined on [0,1] rather than the usual [-1,1]. Note that we only
   consider the special case when \beta = 0.
*/
void H1_FuentesPyramidElement::calcScaledJacobi(const int p, const double alpha,
                                         const double x,
                                         const double t,
                                         double *u)
{
   u[0] = 1.0;
   if (p >= 1)
   {
      u[1] = (2.0 + alpha) * x - t;
   }
   for (int i = 2; i <= p; i++)
   {
      double a = 2.0 * i * (alpha + i) * (2.0 * i + alpha - 2.0);
      double b = 2.0 * i + alpha - 1.0;
      double c = (2.0 * i + alpha) * (2.0 * i + alpha - 2.0);
      double d = 2.0 * (alpha + i - 1.0) * (i - 1) * (2.0 * i - alpha);
      u[i] = (b * (c * (2.0 * x - t) + alpha * alpha * t) * u[i - 1]
              - d * t * t * u[i - 2]) / a;
   }
}

void H1_FuentesPyramidElement::calcScaledJacobi(const int p, const double alpha,
                                         const double x,
                                         const double t,
                                         double *u, double *dudx, double *dudt)
{
   u[0]    = 1.0;
   dudx[0] = 0.0;
   dudt[0] = 0.0;
   if (p >= 1)
   {
      u[1]    = (2.0 + alpha) * x - t;
      dudx[1] =  2.0 + alpha;
      dudt[1] = -1.0;
   }
   for (int i = 2; i <= p; i++)
   {
      double a = 2.0 * i * (alpha + i) * (2.0 * i + alpha - 2.0);
      double b = 2.0 * i + alpha - 1.0;
      double c = (2.0 * i + alpha) * (2.0 * i + alpha - 2.0);
      double d = 2.0 * (alpha + i - 1.0) * (i - 1) * (2.0 * i - alpha);
      u[i] = (b * (c * (2.0 * x - t) + alpha * alpha * t) * u[i - 1]
              - d * t * t * u[i - 2]) / a;
      dudx[i] = (b * ((c * (2.0 * x - t) + alpha * alpha * t) * dudx[i - 1] +
                      2.0 * c * u[i - 1])
                 - d * t * t * dudx[i - 2]) / a;
      dudt[i] = (b * ((c * (2.0 * x - t) + alpha * alpha * t) * dudt[i - 1] +
                      (alpha * alpha - c) * u[i - 1])
                 - d * t * t * dudt[i - 2] - 2.0 * d * t * u[i - 2]) / a;
   }
}

H1_BergotPyramidElement::H1_BergotPyramidElement(const int p, const int btype)
   : NodalFiniteElement(3, Geometry::PYRAMID,
                        (p + 1) * (p + 2) * (2 * p + 3) / 6, // Bergot (JSC)
                        p, FunctionSpace::Qk)
{
   const double *cp = poly1d.ClosedPoints(p, VerifyNodal(VerifyClosed(btype)));

#ifndef MFEM_THREAD_SAFE
   shape_x.SetSize(p + 1);
   shape_y.SetSize(p + 1);
   shape_z.SetSize(p + 1);
   dshape_x.SetSize(p + 1);
   dshape_y.SetSize(p + 1);
   dshape_z.SetSize(p + 1);
   ddshape_x.SetSize(p + 1);
   ddshape_y.SetSize(p + 1);
   ddshape_z.SetSize(p + 1);
   u.SetSize(dof);
   du.SetSize(dof, dim);
   ddu.SetSize(dof, (dim * (dim + 1)) / 2);
   /*
   shape_0.SetSize(p + 1);
   shape_1.SetSize(p + 1);
   shape_2.SetSize(p + 1);
   dshape_0_0.SetSize(p + 1);
   dshape_1_0.SetSize(p + 1);
   dshape_2_0.SetSize(p + 1);
   dshape_0_1.SetSize(p + 1);
   dshape_1_1.SetSize(p + 1);
   dshape_2_1.SetSize(p + 1);
   u.SetSize(dof);
   du.SetSize(dof, dim);
   */
#else
   Vector shape_x(p + 1), shape_y(p + 1), shape_z(p + 1);
   /*
   Vector shape_0(p + 1);
   Vector shape_1(p + 1);
   Vector shape_2(p + 1);
   Vector dshape_0_0(p + 1);
   Vector dshape_1_0(p + 1);
   Vector dshape_2_0(p + 1);
   Vector dshape_0_1(p + 1);
   Vector dshape_1_1(p + 1);
   Vector dshape_2_1(p + 1);
   */
#endif

   // vertices
   Nodes.IntPoint(0).Set3(cp[0], cp[0], cp[0]);
   Nodes.IntPoint(1).Set3(cp[p], cp[0], cp[0]);
   Nodes.IntPoint(2).Set3(cp[p], cp[p], cp[0]);
   Nodes.IntPoint(3).Set3(cp[0], cp[p], cp[0]);
   Nodes.IntPoint(4).Set3(cp[0], cp[0], cp[p]);

   // edges
   int o = 5;
   for (int i = 1; i < p; i++)  // (0,1)
   {
      Nodes.IntPoint(o++).Set3(cp[i], cp[0], cp[0]);
   }
   for (int i = 1; i < p; i++)  // (1,2)
   {
      Nodes.IntPoint(o++).Set3(cp[p], cp[i], cp[0]);
   }
   for (int i = 1; i < p; i++)  // (3,2)
   {
      Nodes.IntPoint(o++).Set3(cp[i], cp[p], cp[0]);
   }
   for (int i = 1; i < p; i++)  // (0,3)
   {
      Nodes.IntPoint(o++).Set3(cp[0], cp[i], cp[0]);
   }
   for (int i = 1; i < p; i++)  // (0,4)
   {
      Nodes.IntPoint(o++).Set3(cp[0], cp[0], cp[i]);
   }
   for (int i = 1; i < p; i++)  // (1,4)
   {
      Nodes.IntPoint(o++).Set3(cp[p-i], cp[0], cp[i]);
   }
   for (int i = 1; i < p; i++)  // (2,4)
   {
      Nodes.IntPoint(o++).Set3(cp[p-i], cp[p-i], cp[i]);
   }
   for (int i = 1; i < p; i++)  // (3,4)
   {
      Nodes.IntPoint(o++).Set3(cp[0], cp[p-i], cp[i]);
   }

   // quadrilateral face
   for (int j = 1; j < p; j++)
   {
      for (int i = 1; i < p; i++)
      {
         Nodes.IntPoint(o++).Set3(cp[i], cp[j], cp[0]);
      }
   }

   // triangular faces
   for (int j = 1; j < p; j++)
      for (int i = 1; i + j < p; i++)  // (0,1,4)
      {
         double w = cp[i] + cp[j] + cp[p-i-j];
         Nodes.IntPoint(o++).Set3(cp[i]/w, cp[0], cp[j]/w);
         // mfem::out << i << " " << j << "\t" << cp[i]/w << " " << cp[0] << " " << cp[j]/w << std::endl;
      }
   for (int j = 1; j < p; j++)
      for (int i = 1; i + j < p; i++)  // (1,2,4)
      {
         double w = cp[i] + cp[j] + cp[p-i-j];
         Nodes.IntPoint(o++).Set3(1.0 - cp[j]/w, cp[i]/w, cp[j]/w);
         // mfem::out << i << " " << j << "\t" << 1.0 - cp[j]/w << " " << cp[i]/w << " " << cp[j]/w << std::endl;
      }
   for (int j = 1; j < p; j++)
      for (int i = 1; i + j < p; i++)  // (3,4,2)
      {
         double w = cp[i] + cp[j] + cp[p-i-j];
         Nodes.IntPoint(o++).Set3(cp[j]/w, 1.0 - cp[i]/w, cp[i]/w);
         // mfem::out << i << " " << j << "\t" << cp[j]/w << " " << 1.0 - cp[i]/w << " " << cp[i]/w << std::endl;
      }
   for (int j = 1; j < p; j++)
      for (int i = 1; i + j < p; i++)  // (0,4,3)
      {
         double w = cp[i] + cp[j] + cp[p-i-j];
         Nodes.IntPoint(o++).Set3(cp[0], cp[j]/w, cp[i]/w);
      }
   /*
   mfem::out << "pts342 = {";
   for (int j = 0; j <= p; j++)
   {
     for (int i = 0; i + j <= p; i++)  // (3,4,2)
     {
       double w = cp[i] + cp[j] + cp[p-i-j];
       mfem::out << "{" << cp[j] / w
       << "," << 1.0 - cp[i] / w
       << "," << cp[i] / w << "}";
       if (i + j != p) mfem::out << ",";
     }
     if (j != p) mfem::out << ",";
     mfem::out << std::endl;
   }
   mfem::out << "};\npts043 = {";
   for (int j = 0; j <= p; j++)
   {
     for (int i = 0; i + j <= p; i++)  // (0,4,3)
     {
       double w = cp[i] + cp[j] + cp[p-i-j];
       mfem::out << "{" << cp[0]
       << "," << cp[j] / w
       << "," << cp[i] / w << "}";
       if (i + j != p) mfem::out << ",";
     }
     if (j != p) mfem::out << ",";
     mfem::out << std::endl;
   }
   mfem::out << "};\n";
   */
   // interior
   for (int k = 1; k < p - 1; k++)
   {
      for (int j = 1; j < p - k; j++)
      {
	double wjk = cp[j] + cp[k] + cp[p-j-k];
	for (int i = 1; i < p - k; i++)
	{
	  double wik = cp[i] + cp[k] + cp[p-i-k];
	  double w = wik * wjk * cp[p-k];
	  Nodes.IntPoint(o++).Set3(cp[i] * (cp[j] + cp[p-j-k]) / w,
				   cp[j] * (cp[i] + cp[p-i-k]) / w,
				   cp[k] * cp[p-k] / w);
	}
      }
   }
   
   /*
   // Points based on Bergot (JSC)'s interior bubbles
   // mfem::out << "pts = {";
   for (int k = 1; k < p - 1; k++)
   {
      for (int j = 0; j <= p - k; j++)
      {
    double wjk = cp[j] + cp[k] + cp[p-j-k];
    for (int i = 0; i <= p - k; i++)
    {
       double wik = cp[i] + cp[k] + cp[p-i-k];
       // double w = (i >= j) ? wik : wjk;
       double w = wik * wjk;
       // mfem::out << wik;
       mfem::out << "{" << cp[i] * (cp[j] + cp[p-j-k]) / (w * cp[p-k])
            << "," << cp[j] * (cp[i] + cp[p-i-k]) / (w * cp[p-k])
            << "," << cp[k] / w << "}";
       // mfem::out << wjk;
       if (i != p - k) mfem::out << ",";
    }
    if (j != p - k) mfem::out << ",";
    mfem::out << std::endl;
      }
      if (k != p - 2) mfem::out << ",";
   }
   mfem::out << "};\n";
   */
   MFEM_ASSERT(o == dof,
               "Number of nodes does not match the "
               "number of degrees of freedom");
   DenseMatrix T(dof);

   for (int m = 0; m < dof; m++)
   {
      const IntegrationPoint &ip = Nodes.IntPoint(m);

      double x = (ip.z < 1.0) ? (ip.x / (1.0 - ip.z)) : 0.0;
      double y = (ip.z < 1.0) ? (ip.y / (1.0 - ip.z)) : 0.0;
      double z = ip.z;
      
      o = 0;
      for (int i = 0; i <= p; i++)
      {
	 poly1d.CalcLegendre(i, x, shape_x);
	 for (int j = 0; j <= p; j++)
         {
	    poly1d.CalcLegendre(j, y, shape_y);
	    int maxij = std::max(i, j);
	    for (int k = 0; k <= p - maxij; k++)
            {
	       poly1d.CalcJacobi(k, 2.0 * (maxij + 1.0), 0.0, z, shape_z);
               T(o++, m) = shape_x(i) * shape_y(j) * shape_z(k) *
		 pow(1.0 - ip.z, maxij);
            }
	 }
      }
   }

   Ti.Factor(T);
   /*
   if (false)
   {
      calcScaledLegendre(6, 0.3, 0.7, shape_0, dshape_0_0, dshape_0_1);
      mfem::out << "Scaled Legendre: ";
      for (int i=0; i<6; i++) { mfem::out << '\t' << shape_0[i]; }
      mfem::out << '\n';

      mfem::out << "Scaled Legendre du/dx:\n";
      for (int i=0; i<6; i++) { mfem::out << '\t' << dshape_0_0[i]; }
      mfem::out << '\n';

      double dx = 1e-8;
      calcScaledLegendre(6, 0.3+dx, 0.7, shape_2);
      mfem::out << "Scaled Legendre (x+dx): ";
      for (int i=0; i<6; i++) { mfem::out << '\t' << shape_2[i]; }
      mfem::out << '\n';
      mfem::out << "Scaled Legendre (u(x+dx) - u(x)) / dx:\n";
      for (int i=0; i<6; i++) { mfem::out << '\t' << (shape_2[i] - shape_0[i]) / dx; }
      mfem::out << '\n';

      mfem::out << "Scaled Legendre du/dt:\n";
      for (int i=0; i<6; i++) { mfem::out << '\t' << dshape_0_1[i]; }
      mfem::out << '\n';

      double dt = 1e-8;
      calcScaledLegendre(6, 0.3, 0.7+dt, shape_2);
      mfem::out << "Scaled Legendre (t+dt): ";
      for (int i=0; i<6; i++) { mfem::out << '\t' << shape_2[i]; }
      mfem::out << '\n';
      mfem::out << "Scaled Legendre (u(t+dt) - u(t)) / dt:\n";
      for (int i=0; i<6; i++) { mfem::out << '\t' << (shape_2[i] - shape_0[i]) / dt; }
      mfem::out << '\n';
   }
   */
}

void H1_BergotPyramidElement::CalcShape(const IntegrationPoint &ip,
                                  Vector &shape) const
{
   const int p = order;

#ifdef MFEM_THREAD_SAFE
   Vector shape_x(order+1);
   Vector shape_y(order+1);
   Vector shape_z(order+1);
   Vector u(dof);
#endif

   double x = (ip.z < 1.0) ? (ip.x / (1.0 - ip.z)) : 0.0;
   double y = (ip.z < 1.0) ? (ip.y / (1.0 - ip.z)) : 0.0;
   double z = ip.z;

   int o = 0;

   for (int i = 0; i <= p; i++)
   {
      poly1d.CalcLegendre(i, x, shape_x);
      for (int j = 0; j <= p; j++)
      {
	 poly1d.CalcLegendre(j, y, shape_y);
	 int maxij = std::max(i, j);
	 for (int k = 0; k <= p - maxij; k++)
	 {
	    poly1d.CalcJacobi(k, 2.0 * (maxij + 1.0), 0.0, z, shape_z);
	    u[o++] = shape_x(i) * shape_y(j) * shape_z(k) *
	       pow(1.0 - ip.z, maxij);
	 }
      }
   }
      
   Ti.Mult(u, shape);
}

void H1_BergotPyramidElement::CalcDShape(const IntegrationPoint &ip,
                                   DenseMatrix &dshape) const
{
  // const int p = order;

#ifdef MFEM_THREAD_SAFE
   Vector  shape_0(p + 1), shape_1(p + 1), shape_2(p + 1);
   Vector dshape_0_0(p + 1), dshape_1_0(p + 1), dshape_2_0(p + 1);
   Vector dshape_0_1(p + 1), dshape_1_1(p + 1), dshape_2_1(p + 1);
   DenseMatrix du(dof, dim);
#endif
   /*
   double dlam[3];
   double dlam4[3];

   double dnu0[2];
   double dnu1[2];
   double dnu2[2];

   double x = ip.x;
   double y = ip.y;
   double z = ip.z;

   int o = 0;

   // Vertices
   grad_lam0(x, y, z, dlam);
   for (int i = 0; i < 3; i++) { du(0, i) = dlam[i]; }
   grad_lam1(x, y, z, dlam);
   for (int i = 0; i < 3; i++) { du(1, i) = dlam[i]; }
   grad_lam2(x, y, z, dlam);
   for (int i = 0; i < 3; i++) { du(2, i) = dlam[i]; }
   grad_lam3(x, y, z, dlam);
   for (int i = 0; i < 3; i++) { du(3, i) = dlam[i]; }
   grad_lam4(x, y, z, dlam);
   for (int i = 0; i < 3; i++) { du(4, i) = dlam[i]; }
   o += 5;

   // Mixed edges (base edges)
   phi_E(p, nu0(x, z), nu1(x, z), shape_0, dshape_0_0, dshape_0_1);
   grad_nu0(x, z, dnu0);
   grad_nu1(x, z, dnu1);
   for (int i = 2; i <= p; i++, o++)
   {
      // Grad(mu0(y / (1.0 - z)) phi_E(nu0(x, z), nu1(x, z)))
      du(o, 0) = mu0(y / (1.0 - z)) *
                 (dshape_0_0[i] * dnu0[0] + dshape_0_1[i] * dnu1[0]);
      du(o, 1) = dmu0(y / (1.0 - z)) * shape_0[i] / (1.0 - z);
      du(o, 2) = dmu0(y / (1.0 - z)) * shape_0[i] / pow(1.0 - z, 2) +
                 mu0(y / (1.0 - z)) *
                 (dshape_0_0[i] * dnu0[1] + dshape_0_1[i] * dnu1[1]);
   }
   phi_E(p, nu0(y, z), nu1(y, z), shape_0, dshape_0_0, dshape_0_1);
   grad_nu0(y, z, dnu0);
   grad_nu1(y, z, dnu1);
   for (int i = 2; i <= p; i++, o++)
   {
      // Grad(mu1(x / (1.0 - z)) * phi_E(nu0(y, z), nu1(y, z)))
      du(o, 0) = dmu1(x / (1.0 - z)) * shape_0[i] / (1.0 - z);
      du(o, 1) = mu1(x / (1.0 - z)) *
                 (dshape_0_0[i] * dnu0[0] + dshape_0_1[i] * dnu1[0]);
      du(o, 2) = dmu1(x / (1.0 - z)) * shape_0[i] / pow(1.0 - z, 2) +
                 mu1(x / (1.0 - z)) *
                 (dshape_0_0[i] * dnu0[1] + dshape_0_1[i] * dnu1[1]);
   }
   phi_E(p, nu0(x, z), nu1(x, z), shape_0, dshape_0_0, dshape_0_1);
   grad_nu0(x, z, dnu0);
   grad_nu1(x, z, dnu1);
   for (int i = 2; i <= p; i++, o++)
   {
      // Grad(mu1(y / (1.0 - z)) * phi_E(nu0(x, z), nu1(x, z)))
      du(o, 0) = mu1(y / (1.0 - z)) *
                 (dshape_0_0[i] * dnu0[0] + dshape_0_1[i] * dnu1[0]);
      du(o, 1) = dmu1(y / (1.0 - z)) * shape_0[i] / (1.0 - z);
      du(o, 2) = dmu1(y / (1.0 - z)) * shape_0[i] / pow(1.0 - z, 2) +
                 mu1(y / (1.0 - z)) *
                 (dshape_0_0[i] * dnu0[1] + dshape_0_1[i] * dnu1[1]);
   }
   phi_E(p, nu0(y, z), nu1(y, z), shape_0, dshape_0_0, dshape_0_1);
   grad_nu0(y, z, dnu0);
   grad_nu1(y, z, dnu1);
   for (int i = 2; i <= p; i++, o++)
   {
      // Grad(mu0(x / (1.0 - z)) * phi_E(nu0(y, z), nu1(y, z)))
      du(o, 0) = dmu0(x / (1.0 - z)) * shape_0[i] / (1.0 - z);
      du(o, 1) = mu0(x / (1.0 - z)) *
                 (dshape_0_0[i] * dnu0[0] + dshape_0_1[i] * dnu1[0]);
      du(o, 2) = dmu0(x / (1.0 - z)) * shape_0[i] / pow(1.0 - z, 2) +
                 mu0(x / (1.0 - z)) *
                 (dshape_0_0[i] * dnu0[1] + dshape_0_1[i] * dnu1[1]);
   }

   // Triangle edges (upright edges)
   grad_lam4(x, y, z, dlam4);
   grad_lam0(x, y, z, dlam);
   phi_E(p, lam0(x, y, z), lam4(x, y, z), shape_0, dshape_0_0, dshape_0_1);
   for (int i = 2; i<= p; i++, o++)
   {
      // Grad(phi_E(lam0(x,y,z), lam4(x,y,z)))
      for (int j = 0; j < 3; j++)
      {
         du(o, j) = dshape_0_0[i] * dlam[j] + dshape_0_1[i] * dlam4[j];
      }
   }
   grad_lam1(x, y, z, dlam);
   phi_E(p, lam1(x, y, z), lam4(x, y, z), shape_0, dshape_0_0, dshape_0_1);
   for (int i = 2; i<= p; i++, o++)
   {
      // Grad(phi_E(lam1(x,y,z), lam4(x,y,z)))
      for (int j = 0; j < 3; j++)
      {
         du(o, j) = dshape_0_0[i] * dlam[j] + dshape_0_1[i] * dlam4[j];
      }
   }
   grad_lam2(x, y, z, dlam);
   phi_E(p, lam2(x, y, z), lam4(x, y, z), shape_0, dshape_0_0, dshape_0_1);
   for (int i = 2; i<= p; i++, o++)
   {
      // Grad(phi_E(lam2(x,y,z), lam4(x,y,z)))
      for (int j = 0; j < 3; j++)
      {
         du(o, j) = dshape_0_0[i] * dlam[j] + dshape_0_1[i] * dlam4[j];
      }
   }
   grad_lam3(x, y, z, dlam);
   phi_E(p, lam3(x, y, z), lam4(x, y, z), shape_0, dshape_0_0, dshape_0_1);
   for (int i = 2; i<= p; i++, o++)
   {
      // Grad(phi_E(lam3(x,y,z), lam4(x,y,z)))
      for (int j = 0; j < 3; j++)
      {
         du(o, j) = dshape_0_0[i] * dlam[j] + dshape_0_1[i] * dlam4[j];
      }
   }

   // Quadrilateral face
   phi_E(p, mu0(x / (1.0 - z)), mu1(x / (1.0 - z)), shape_0,
         dshape_0_0, dshape_0_1);
   phi_E(p, mu0(y / (1.0 - z)), mu1(y / (1.0 - z)), shape_1,
         dshape_1_0, dshape_1_1);
   for (int j = 2; j <= p; j++)
   {
      for (int i = 2; i <= p; i++, o++)
      {
         // Grad(mu0(z) * phi_E(mu0(x / (1.0 - z)), mu1(x / (1.0 - z)))
         //             * phi_E(mu0(y / (1.0 - z)), mu1(y / (1.0 - z))))
         du(o, 0) = mu0(z) * (dshape_0_0[i] * dmu0(x / (1.0 - z)) +
                              dshape_0_1[i] * dmu1(x / (1.0 - z))) * shape_1[j]
                    / (1.0 - z);
         du(o, 1) = mu0(z) * shape_0[i] * (dshape_1_0[i] * dmu0(y / (1.0 - z)) +
                                           dshape_1_1[i] * dmu1(y / (1.0 - z)))
                    / (1.0 - z);
         du(o, 2) = dmu0(z) * shape_0[i] * shape_1[j] +
                    mu0(z) * ((dshape_0_0[i] * dmu0(x / (1.0 - z)) +
                               dshape_0_1[i] * dmu1(x / (1.0 - z))) * x * shape_1[j] +
                              shape_0[i] * (dshape_1_0[i] * dmu0(y / (1.0 - z)) +
                                            dshape_1_1[i] * dmu1(y / (1.0 - z))) * y)
                    / pow(1.0 - z, 2);
      }
   }

   // Triangular faces
   phi_E(p, nu0(x, z), nu1(x, z), shape_0, dshape_0_0, dshape_0_1);
   grad_nu0(x, z, dnu0);
   grad_nu1(x, z, dnu1);
   grad_nu2(x, z, dnu2);
   for (int i = 2; i <= p; i++)
   {
      calcIntegratedJacobi(p, 2.0 * i, nu2(x, z), 1.0, shape_1,
                           dshape_1_0, dshape_1_1);
      for (int j = 1; j <= p - i; j++, o++)
      {
         // u[o] = mu0(y / (1.0 - z)) * tmp_x[i] * tmp_y[j];
         du(o, 0) = mu0(y / (1.0 - z)) *
                    ((dshape_0_0[i] * dnu0[0] + dshape_0_1[i] * dnu1[0]) * shape_1[j] +
                     shape_0[i] * dshape_1_0[j] * dnu2[0]);
         du(o, 1) = dmu0(y / (1.0 - z)) * shape_0[i] * shape_1[j] / (1.0 - z);
         du(o, 2) = dmu0(y / (1.0 - z)) * shape_0[i] * shape_1[j]
                    / pow(1.0 - z, 2) +
                    mu0(y / (1.0 - z)) *
                    ((dshape_0_0[i] * dnu0[1] + dshape_0_1[i] * dnu1[1]) * shape_1[j] +
                     shape_0[i] * dshape_1_0[i] * dnu2[1]);
      }
   }
   phi_E(p, nu0(y, z), nu1(y, z), shape_0, dshape_0_0, dshape_0_1);
   grad_nu0(y, z, dnu0);
   grad_nu1(y, z, dnu1);
   grad_nu2(y, z, dnu2);
   for (int i = 2; i <= p; i++)
   {
      calcIntegratedJacobi(p, 2.0 * i, nu2(y, z), 1.0, shape_1,
                           dshape_1_0, dshape_1_1);
      for (int j = 1; j <= p - i; j++, o++)
      {
         // u[o] = mu1(x / (1.0 - z)) * tmp_x[i] * tmp_y[j];
         du(o, 0) = dmu1(x / (1.0 - z)) * shape_0[i] * shape_1[j] / (1.0 - z);
         du(o, 1) = mu1(x / (1.0 - z)) *
                    ((dshape_0_0[i] * dnu0[0] + dshape_0_1[i] * dnu1[0]) * shape_1[j] +
                     shape_0[i] * dshape_1_0[j] * dnu2[0]);
         du(o, 2) = dmu1(x / (1.0 - z)) * shape_0[i] * shape_1[j]
                    / pow(1.0 - z, 2) +
                    mu1(x / (1.0 - z)) *
                    ((dshape_0_0[i] * dnu0[1] + dshape_0_1[i] * dnu1[1]) * shape_1[j] +
                     shape_0[i] * dshape_1_0[i] * dnu2[1]);
      }
   }
   phi_E(p, nu0(x, z), nu1(x, z), shape_0, dshape_0_0, dshape_0_1);
   grad_nu0(x, z, dnu0);
   grad_nu1(x, z, dnu1);
   grad_nu2(x, z, dnu2);
   for (int i = 2; i <= p; i++)
   {
      calcIntegratedJacobi(p, 2.0 * i, nu2(x, z), 1.0, shape_1,
                           dshape_1_0, dshape_1_1);
      for (int j = 1; j <= p - i; j++, o++)
      {
         // u[o] = mu1(y / (1.0 - z)) * tmp_x[i] * tmp_y[j];
         du(o, 0) = mu1(y / (1.0 - z)) *
                    ((dshape_0_0[i] * dnu0[0] + dshape_0_1[i] * dnu1[0]) * shape_1[j] +
                     shape_0[i] * dshape_1_0[j] * dnu2[0]);
         du(o, 1) = dmu1(y / (1.0 - z)) * shape_0[i] * shape_1[j] / (1.0 - z);
         du(o, 2) = dmu1(y / (1.0 - z)) * shape_0[i] * shape_1[j]
                    / pow(1.0 - z, 2) +
                    mu1(y / (1.0 - z)) *
                    ((dshape_0_0[i] * dnu0[1] + dshape_0_1[i] * dnu1[1]) * shape_1[j] +
                     shape_0[i] * dshape_1_0[i] * dnu2[1]);
      }
   }
   phi_E(p, nu0(y, z), nu1(y, z), shape_0, dshape_0_0, dshape_0_1);
   grad_nu0(y, z, dnu0);
   grad_nu1(y, z, dnu1);
   grad_nu2(y, z, dnu2);
   for (int i = 2; i <= p; i++)
   {
      calcIntegratedJacobi(p, 2.0 * i, nu2(y, z), 1.0, shape_1,
                           dshape_1_0, dshape_1_1);
      for (int j = 1; j <= p - i; j++, o++)
      {
         // u[o] = mu0(x / (1.0 - z)) * tmp_x[i] * tmp_y[j];
         du(o, 0) = dmu0(x / (1.0 - z)) * shape_0[i] * shape_1[j] / (1.0 - z);
         du(o, 1) = mu0(x / (1.0 - z)) *
                    ((dshape_0_0[i] * dnu0[0] + dshape_0_1[i] * dnu1[0]) * shape_1[j] +
                     shape_0[i] * dshape_1_0[j] * dnu2[0]);
         du(o, 2) = dmu0(x / (1.0 - z)) * shape_0[i] * shape_1[j]
                    / pow(1.0 - z, 2) +
                    mu0(x / (1.0 - z)) *
                    ((dshape_0_0[i] * dnu0[1] + dshape_0_1[i] * dnu1[1]) * shape_1[j] +
                     shape_0[i] * dshape_1_0[i] * dnu2[1]);
      }
   }

   // Interior
   phi_E(p, mu0(x / (1.0 - z)), mu1(x / (1.0 - z)), shape_0,
         dshape_0_0, dshape_0_1);
   phi_E(p, mu0(y / (1.0 - z)), mu1(y / (1.0 - z)), shape_1,
         dshape_1_0, dshape_1_1);
   phi_E(p, mu0(z), mu1(z), shape_2, dshape_2_0, dshape_2_1);
   for (int k = 2; k <= p; k++)
   {
      for (int j = 2; j <= p; j++)
      {
         for (int i = 2; i <= p; i++, o++)
         {
            du(o, 0) = (dshape_0_0[i] * dmu0(x / (1.0 - z)) +
                        dshape_0_1[i] * dmu1(x / (1.0 - z))) *
                       shape_1[j] * shape_2[k] / (1.0 - z);
            du(o, 1) = shape_0[i] * (dshape_1_0[j] * dmu0(y / (1.0 - z)) +
                                     dshape_1_1[j] * dmu1(y / (1.0 - z))) *
                       shape_2[k] / (1.0 - z);
            du(o, 2) = ((dshape_0_0[i] * dmu0(x / (1.0 - z)) +
                         dshape_0_1[i] * dmu1(x / (1.0 - z))) * shape_1[j] +
                        shape_0[i] * (dshape_1_0[j] * dmu0(y / (1.0 - z)) +
                                      dshape_1_1[j] * dmu1(y / (1.0 - z)))) *
                       shape_2[k] / pow(1.0 - z, 2) +
                       shape_0[i] * shape_1[j] * (dshape_2_0[k] * dmu0(z) +
                                                  dshape_2_1[k] * dmu1(z));
         }
      }
   }
   */
   /*
   double x = ip.x;
   double y = ip.y;
   double z = ip.z;

   dshape.SetSize(5, 3);
   dshape(0,0) = -1.0 + ((z<1.0) ? (y / (1.0 - z)) : 0.0);
   dshape(0,1) = -1.0 + ((z<1.0) ? (x / (1.0 - z)) : 0.0);
   dshape(0,2) = -1.0 + ((z<1.0) ? (x * y / pow(1.0 - z, 2)) : 0.0);

   dshape(1,0) = 1.0 - ((z<1.0) ? (y / (1.0 - z)) : 0.0);
   dshape(1,1) = (z<1.0) ? (-x / (1.0 - z)) : 0.0;
   dshape(1,2) = (z<1.0) ? (-x * y / pow(1.0 - z, 2)) : 0.0;

   dshape(2,0) = (z<1.0) ? (y / (1.0 - z)) : 0.0;
   dshape(2,1) = (z<1.0) ? (x / (1.0 - z)) : 0.0;
   dshape(2,2) = (z<1.0) ? (x * y / pow(1.0 - z, 2)) : 0.0;

   dshape(3,0) = (z<1.0) ? (-y / (1.0 - z)) : 0.0;
   dshape(3,1) = 1.0 - ((z<1.0) ? (x / (1.0 - z)) : 0.0);
   dshape(3,2) = (z<1.0) ? (-x * y / pow(1.0 - z, 2)) : 0.0;

   dshape(4,0) = 0.0;
   dshape(4,1) = 0.0;
   dshape(4,2) = 1.0;
   */
   /*
   double x = (ip.z < 1.0) ? (ip.x / (1.0 - ip.z)) : 0.0;
   double y = (ip.z < 1.0) ? (ip.y / (1.0 - ip.z)) : 0.0;
   double z = ip.z;

   int o = 0;
   for (int i = 0; i <= p; i++)
   {
      poly1d.CalcLegendre(i, x, shape_x, dshape_x);
      for (int j = 0; j <= p; j++)
      {
    poly1d.CalcLegendre(j, y, shape_y, dshape_y);
    int maxij = std::max(i, j);
    for (int k = 0; k <= p - maxij; k++)
    {
      poly1d.CalcJacobi(k, 2.0 * (maxij + 1.0), 0.0, z,
                shape_z, dshape_z);
      du(o,0) = dshape_x(i) * shape_y(j) * shape_z(k) *
        pow(1.0 - ip.z, maxij - 1);
      du(o,1) = shape_x(i) * dshape_y(j) * shape_z(k) *
        pow(1.0 - ip.z, maxij - 1);
      du(o,2) = shape_x(i) * shape_y(j) * dshape_z(k) *
        pow(1.0 - ip.z, maxij) +
        (dshape_x(i) * shape_y(j) + shape_x(i) * dshape_y(j)) *
        shape_z(k) * pow(1.0 - ip.z, maxij - 2) -
        ((maxij > 0) ? (maxij * shape_x(i) * shape_y(j) * shape_z(k) *
              pow(1.0 - ip.z, maxij - 1)) : 0.0);
      o++;
    }
      }
   }
   */
   // calcDBasis(order, ip, du);
   Ti.Mult(du, dshape);
}


}
