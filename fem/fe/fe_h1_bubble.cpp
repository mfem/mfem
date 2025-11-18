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

// H1 Finite Element classes

#include "fe_h1_bubble.hpp"

namespace mfem
{

using namespace std;

H1Bubble_TriangleElement::H1Bubble_TriangleElement(int p, int q, int btype)
   : NodalFiniteElement(2, Geometry::TRIANGLE, 3*p + ((q+1)*(q+2))/2, p,
                        FunctionSpace::Pk),
     bubble_order(q)
{
   const real_t *cp = poly1d.ClosedPoints(p, VerifyNodal(VerifyClosed(btype)));
   const real_t *cp2 = poly1d.ClosedPoints(
                          q + 3, VerifyNodal(VerifyClosed(btype)));

   const int n1d = std::max(p + 1, q + 1);
   const int npq = ((p+1)*(p+2))/2 + ((q+1)*(q+2))/2;

#ifndef MFEM_THREAD_SAFE
   shape_x.SetSize(n1d);
   shape_y.SetSize(n1d);
   shape_l.SetSize(n1d);
   dshape_x.SetSize(n1d);
   dshape_y.SetSize(n1d);
   dshape_l.SetSize(n1d);
   u.SetSize(npq);
   du.SetSize(npq, dim);
#endif

   // vertices
   Nodes.IntPoint(0).Set2(cp[0], cp[0]);
   Nodes.IntPoint(1).Set2(cp[p], cp[0]);
   Nodes.IntPoint(2).Set2(cp[0], cp[p]);

   // edges
   int o = 3;
   for (int i = 1; i < p; i++)
   {
      Nodes.IntPoint(o++).Set2(cp[i], cp[0]);
   }
   for (int i = 1; i < p; i++)
   {
      Nodes.IntPoint(o++).Set2(cp[p-i], cp[i]);
   }
   for (int i = 1; i < p; i++)
   {
      Nodes.IntPoint(o++).Set2(cp[0], cp[p-i]);
   }

   // Interior P_{q-3} nodes
   for (int j = 1; j < q + 3; j++)
   {
      for (int i = 1; i + j < q + 3; i++)
      {
         const real_t w = cp2[i] + cp2[j] + cp2[q+3-i-j];
         Nodes.IntPoint(o++).Set2(cp2[i]/w, cp2[j]/w);
      }
   }

#ifdef MFEM_THREAD_SAFE
   Vector shape_x(n1d), shape_y(n1d), shape_l(n1d);
#endif

   DenseMatrix Tt(dof, npq);
   for (int k = 0; k < dof; ++k)
   {
      const IntegrationPoint &ip = Nodes.IntPoint(k);
      poly1d.CalcBasis(p, ip.x, shape_x);
      poly1d.CalcBasis(p, ip.y, shape_y);
      poly1d.CalcBasis(p, 1. - ip.x - ip.y, shape_l);
      o = 0;
      for (int j = 0; j <= p; j++)
      {
         for (int i = 0; i + j <= p; i++)
         {
            Tt(k, o++) = shape_x[i]*shape_y[j]*shape_l[p-i-j];
         }
      }

      poly1d.CalcBasis(q, ip.x, shape_x);
      poly1d.CalcBasis(q, ip.y, shape_y);
      poly1d.CalcBasis(q, 1. - ip.x - ip.y, shape_l);
      const real_t b_T = ip.x * ip.y * (1 - ip.x - ip.y);
      for (int j = 0; j <= q; j++)
      {
         for (int i = 0; i + j <= q; i++)
         {
            Tt(k, o++) = b_T*shape_x[i]*shape_y[j]*shape_l[q-i-j];
         }
      }
   }

   // Compute left inverse of T (given Tt = T^T).
   DenseMatrix TtT(dof, dof);
   MultAAt(Tt, TtT);

   DenseMatrixInverse TtT_inv(TtT);
   T_pinv.SetSize(dof, dof);
   TtT_inv.Mult(Tt, T_pinv);
}

void H1Bubble_TriangleElement::CalcShape(const IntegrationPoint &ip,
                                         Vector &shape) const
{
   const int p = order;
   const int q = bubble_order;

#ifdef MFEM_THREAD_SAFE
   const int n1d = std::max(p + 1, q + 2);
   const int npq = ((p+1)*(p+2))/2 + ((q+1)*(q+2))/2;
   Vector shape_x(n1d), shape_y(n1d), shape_l(n1d), u(npq);
#endif

   poly1d.CalcBasis(p, ip.x, shape_x);
   poly1d.CalcBasis(p, ip.y, shape_y);
   poly1d.CalcBasis(p, 1. - ip.x - ip.y, shape_l);

   int o = 0;
   for (int j = 0; j <= p; j++)
   {
      for (int i = 0; i + j <= p; i++)
      {
         u(o++) = shape_x[i]*shape_y[j]*shape_l[p-i-j];
      }
   }

   poly1d.CalcBasis(q, ip.x, shape_x);
   poly1d.CalcBasis(q, ip.y, shape_y);
   poly1d.CalcBasis(q, 1. - ip.x - ip.y, shape_l);
   const real_t b_T = ip.x * ip.y * (1 - ip.x - ip.y);

   for (int j = 0; j <= q; j++)
   {
      for (int i = 0; i + j <= q; i++)
      {
         u(o++) = b_T*shape_x[i]*shape_y[j]*shape_l[q-i-j];
      }
   }

   T_pinv.Mult(u, shape);
}

void H1Bubble_TriangleElement::CalcDShape(const IntegrationPoint &ip,
                                          DenseMatrix &dshape) const
{
   const int p = order;
   const int q = bubble_order;

#ifdef MFEM_THREAD_SAFE
   const int n1d = std::max(p + 1, q + 2);
   const int npq = ((p+1)*(p+2))/2 + ((q+1)*(q+2))/2;
   Vector  shape_x(n1d),  shape_y(n1d),  shape_l(n1d);
   Vector dshape_x(n1d), dshape_y(n1d), dshape_l(n1d);
   DenseMatrix du(npq, dim);
#endif

   const real_t lambda = 1.0 - ip.x - ip.y;

   poly1d.CalcBasis(p, ip.x, shape_x, dshape_x);
   poly1d.CalcBasis(p, ip.y, shape_y, dshape_y);
   poly1d.CalcBasis(p, lambda, shape_l, dshape_l);

   int o = 0;
   for (int j = 0; j <= p; j++)
   {
      for (int i = 0; i + j <= p; i++)
      {
         int k = p - i - j;
         du(o,0) = (dshape_x[i]*shape_l[k] - shape_x[i]*dshape_l[k])*shape_y[j];
         du(o,1) = (dshape_y[j]* shape_l[k] - shape_y[j]*dshape_l[k])*shape_x[i];
         o++;
      }
   }

   poly1d.CalcBasis(q, ip.x, shape_x, dshape_x);
   poly1d.CalcBasis(q, ip.y, shape_y, dshape_y);
   poly1d.CalcBasis(q, lambda, shape_l, dshape_l);
   const real_t b_T = ip.x * ip.y * lambda;
   const real_t dxb_T = ip.y * (lambda - ip.x);
   const real_t dyb_T = ip.x * (lambda - ip.y);

   for (int j = 0; j <= q; j++)
   {
      for (int i = 0; i + j <= q; i++)
      {
         int k = q - i - j;
         du(o,0) = shape_y[j]*(dxb_T*shape_x[i]*shape_l[k]
                               + b_T*dshape_x[i]*shape_l[k]
                               - b_T*shape_x[i]*dshape_l[k]);
         du(o,1) = shape_x[i]*(dyb_T*shape_y[i]*shape_l[k]
                               + b_T*dshape_y[i]*shape_l[k]
                               - b_T*shape_y[i]*dshape_l[k]);
         o++;
      }
   }

   Mult(T_pinv, du, dshape);
}

H1Bubble_QuadrilateralElement::H1Bubble_QuadrilateralElement(
   int p, int q, int btype)
   : NodalFiniteElement(2, Geometry::SQUARE, 4*p + (q+1)*(q+1), p,
                        FunctionSpace::Qk)
{
   const real_t *cp = poly1d.ClosedPoints(p, VerifyNodal(VerifyClosed(btype)));
   const real_t *cp2 = poly1d.ClosedPoints(
                          q + 2, VerifyNodal(VerifyClosed(btype)));

   const int n1d = std::max(p + 1, q + 1);
   const int npq = (p+1)*(p+1) + (q+1)*(q+1);

#ifndef MFEM_THREAD_SAFE
   shape_x.SetSize(n1d);
   shape_y.SetSize(n1d);
   dshape_x.SetSize(n1d);
   dshape_y.SetSize(n1d);

   u.SetSize(npq);
   du.SetSize(npq, dim);
#endif

   // vertices
   Nodes.IntPoint(0).Set2(cp[0], cp[0]);
   Nodes.IntPoint(1).Set2(cp[p], cp[0]);
   Nodes.IntPoint(2).Set2(cp[p], cp[p]);
   Nodes.IntPoint(3).Set2(cp[0], cp[p]);

   // edges
   int o = 4;
   for (int i = 1; i < p; i++)
   {
      Nodes.IntPoint(o++).Set2(cp[i], cp[0]);
   }
   for (int i = 1; i < p; i++)
   {
      Nodes.IntPoint(o++).Set2(cp[p], cp[i]);
   }
   for (int i = 1; i < p; i++)
   {
      Nodes.IntPoint(o++).Set2(cp[p-i], cp[p]);
   }
   for (int i = 1; i < p; i++)
   {
      Nodes.IntPoint(o++).Set2(cp[0], cp[p-i]);
   }

   // interior P_{q+2} nodes
   for (int j = 1; j < q+2; j++)
   {
      for (int i = 1; i < q+2; i++)
      {
         Nodes.IntPoint(o++).Set2(cp2[i], cp2[j]);
      }
   }

#ifdef MFEM_THREAD_SAFE
   Vector shape_x(n1d), shape_y(n1d);
#endif

   DenseMatrix Tt(dof, npq);
   for (int k = 0; k < dof; ++k)
   {
      const IntegrationPoint &ip = Nodes.IntPoint(k);
      poly1d.CalcBasis(p, ip.x, shape_x);
      poly1d.CalcBasis(p, ip.y, shape_y);

      o = 0;
      for (int j = 0; j <= p; j++)
      {
         for (int i = 0; i <= p; i++)
         {
            Tt(k, o++) = shape_x[i]*shape_y[j];
         }
      }

      poly1d.CalcBasis(q, ip.x, shape_x);
      poly1d.CalcBasis(q, ip.y, shape_y);
      const real_t b_T = ip.x*(1.0 - ip.x)*ip.y*(1.0 - ip.y);
      for (int j = 0; j <= q; j++)
      {
         for (int i = 0; i <= q; i++)
         {
            Tt(k, o++) = b_T*shape_x[i]*shape_y[j];
         }
      }
   }

   // Compute left inverse of T (given Tt = T^T).
   DenseMatrix TtT(dof, dof);
   MultAAt(Tt, TtT);

   DenseMatrixInverse TtT_inv(TtT);
   T_pinv.SetSize(dof, dof);
   TtT_inv.Mult(Tt, T_pinv);
}

void H1Bubble_QuadrilateralElement::CalcShape(const IntegrationPoint &ip,
                                              Vector &shape) const
{
   const int p = order;
   const int q = bubble_order;

#ifdef MFEM_THREAD_SAFE
   const int n1d = std::max(p + 1, q + 1);
   const int npq = (p+1)*(p+1) + (q+1)*(q+1);
   Vector shape_x(n1d), shape_y(n1d), u(npq);
#endif

   poly1d.CalcBasis(p, ip.x, shape_x);
   poly1d.CalcBasis(p, ip.y, shape_y);

   int o = 0;
   for (int j = 0; j <= p; j++)
   {
      for (int i = 0; i <= p; i++)
      {
         u(o++) = shape_x[i]*shape_y[j];
      }
   }

   poly1d.CalcBasis(q, ip.x, shape_x);
   poly1d.CalcBasis(q, ip.y, shape_y);
   const real_t b_T = ip.x*(1.0 - ip.x)*ip.y*(1.0 - ip.y);

   for (int j = 0; j <= q; j++)
   {
      for (int i = 0; i <= q; i++)
      {
         u(o++) = b_T*shape_x[i]*shape_y[j];
      }
   }

   T_pinv.Mult(u, shape);
}

void H1Bubble_QuadrilateralElement::CalcDShape(const IntegrationPoint &ip,
                                               DenseMatrix &dshape) const
{
   const int p = order;
   const int q = bubble_order;

#ifdef MFEM_THREAD_SAFE
   const int n1d = std::max(p + 1, q + 1);
   const int npq = (p+1)*(p+1) + (q+1)*(q+1);
   Vector shape_x(n1d), shape_y(n1d), dshape_x(n1d), dshape_y(n1d);
   DenseMatrix du(npq, dim);
#endif

   poly1d.CalcBasis(p, ip.x, shape_x, dshape_x);
   poly1d.CalcBasis(p, ip.y, shape_y, dshape_y);

   int o = 0;
   for (int j = 0; j <= p; j++)
   {
      for (int i = 0; i <= p; i++)
      {
         du(o,0) = dshape_x[i]*shape_y[j];
         du(o,1) = shape_x[i]*dshape_y[j];
         o += 1;
      }
   }

   poly1d.CalcBasis(q, ip.x, shape_x, dshape_x);
   poly1d.CalcBasis(q, ip.y, shape_y, dshape_y);
   const real_t b_T = ip.x*(1.0 - ip.x)*ip.y*(1.0 - ip.y);
   const real_t dxb_T = (1.0 - 2*ip.x)*ip.y*(1.0 - ip.y);
   const real_t dyb_T = ip.x*(1.0 - ip.x)*(1.0 - 2*ip.y);

   for (int j = 0; j <= q; j++)
   {
      for (int i = 0; i <= q; i++)
      {
         du(o,0) = (dxb_T*shape_x[i] + b_T*dshape_x[i])*shape_y[j];
         du(o,1) = (dyb_T*shape_y[j] + b_T*dshape_y[j])*shape_x[i];
         o += 1;
      }
   }

   Mult(T_pinv, du, dshape);
}

}
