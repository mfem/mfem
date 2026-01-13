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
   : NodalFiniteElement(2, Geometry::TRIANGLE, 3*p + ((q+1)*(q+2))/2,
                        max(p, 3 + q), FunctionSpace::Pk),
     base_order(p), bubble_order(q)
{
   const real_t *cp = poly1d.ClosedPoints(p, VerifyNodal(VerifyClosed(btype)));
   const real_t *cp2 = poly1d.ClosedPoints(
                          q + 3, VerifyNodal(VerifyClosed(btype)));

   const int n1d = max(p + 1, q + 1);
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

   // Interior P_{q+3} nodes
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
   const int p = base_order;
   const int q = bubble_order;

#ifdef MFEM_THREAD_SAFE
   const int n1d = max(p + 1, q + 1);
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
   const int p = base_order;
   const int q = bubble_order;

#ifdef MFEM_THREAD_SAFE
   const int n1d = max(p + 1, q + 1);
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
         du(o,1) = shape_x[i]*(dyb_T*shape_y[j]*shape_l[k]
                               + b_T*dshape_y[j]*shape_l[k]
                               - b_T*shape_y[j]*dshape_l[k]);
         o++;
      }
   }

   Mult(T_pinv, du, dshape);
}

H1Bubble_QuadrilateralElement::H1Bubble_QuadrilateralElement(
   int p, int q, int btype)
   : NodalFiniteElement(2, Geometry::SQUARE, 4*p + (q+1)*(q+1),
                        max(p, 2 + q), FunctionSpace::Qk),
     base_order(p), bubble_order(q)
{
   const real_t *cp = poly1d.ClosedPoints(p, VerifyNodal(VerifyClosed(btype)));
   const real_t *cp2 = poly1d.ClosedPoints(
                          q + 2, VerifyNodal(VerifyClosed(btype)));

   const int n1d = max(p + 1, q + 1);
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
   const int p = base_order;
   const int q = bubble_order;

#ifdef MFEM_THREAD_SAFE
   const int n1d = max(p + 1, q + 1);
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
   const int p = base_order;
   const int q = bubble_order;

#ifdef MFEM_THREAD_SAFE
   const int n1d = max(p + 1, q + 1);
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

H1Bubble_TetrahedronElement::H1Bubble_TetrahedronElement(
   int p, int q, int btype)
   : NodalFiniteElement(3, Geometry::TETRAHEDRON,
                        2*(p*p + 1) + ((q+1)*(q+2)*(q+3))/6,
                        max(p, 4 + q), FunctionSpace::Pk),
     base_order(p), bubble_order(q)
{
   const real_t *cp = poly1d.ClosedPoints(p, VerifyNodal(VerifyClosed(btype)));
   const real_t *cp2 = poly1d.ClosedPoints(
                          q + 4, VerifyNodal(VerifyClosed(btype)));

   const int n1d = max(p+1, q+1);
   const int npq = ((p+1)*(p+2)*(p+3))/6 + ((q+1)*(q+2)*(q+3))/6;

#ifndef MFEM_THREAD_SAFE
   shape_x.SetSize(n1d);
   shape_y.SetSize(n1d);
   shape_z.SetSize(n1d);
   shape_l.SetSize(n1d);
   dshape_x.SetSize(n1d);
   dshape_y.SetSize(n1d);
   dshape_z.SetSize(n1d);
   dshape_l.SetSize(n1d);
   u.SetSize(npq);
   du.SetSize(npq, dim);
#else
   Vector shape_x(n1d), shape_y(n1d), shape_z(n1d), shape_l(n1d);
#endif

   // vertices
   Nodes.IntPoint(0).Set3(cp[0], cp[0], cp[0]);
   Nodes.IntPoint(1).Set3(cp[p], cp[0], cp[0]);
   Nodes.IntPoint(2).Set3(cp[0], cp[p], cp[0]);
   Nodes.IntPoint(3).Set3(cp[0], cp[0], cp[p]);

   // edges (see Tetrahedron::edges in mesh/tetrahedron.cpp)
   int o = 4;
   for (int i = 1; i < p; i++)  // (0,1)
   {
      Nodes.IntPoint(o++).Set3(cp[i], cp[0], cp[0]);
   }
   for (int i = 1; i < p; i++)  // (0,2)
   {
      Nodes.IntPoint(o++).Set3(cp[0], cp[i], cp[0]);
   }
   for (int i = 1; i < p; i++)  // (0,3)
   {
      Nodes.IntPoint(o++).Set3(cp[0], cp[0], cp[i]);
   }
   for (int i = 1; i < p; i++)  // (1,2)
   {
      Nodes.IntPoint(o++).Set3(cp[p-i], cp[i], cp[0]);
   }
   for (int i = 1; i < p; i++)  // (1,3)
   {
      Nodes.IntPoint(o++).Set3(cp[p-i], cp[0], cp[i]);
   }
   for (int i = 1; i < p; i++)  // (2,3)
   {
      Nodes.IntPoint(o++).Set3(cp[0], cp[p-i], cp[i]);
   }

   // faces (see Mesh::GenerateFaces in mesh/mesh.cpp)
   for (int j = 1; j < p; j++)
   {
      for (int i = 1; i + j < p; i++)  // (1,2,3)
      {
         real_t w = cp[i] + cp[j] + cp[p-i-j];
         Nodes.IntPoint(o++).Set3(cp[p-i-j]/w, cp[i]/w, cp[j]/w);
      }
   }
   for (int j = 1; j < p; j++)
   {
      for (int i = 1; i + j < p; i++)  // (0,3,2)
      {
         real_t w = cp[i] + cp[j] + cp[p-i-j];
         Nodes.IntPoint(o++).Set3(cp[0], cp[j]/w, cp[i]/w);
      }
   }
   for (int j = 1; j < p; j++)
   {
      for (int i = 1; i + j < p; i++)  // (0,1,3)
      {
         real_t w = cp[i] + cp[j] + cp[p-i-j];
         Nodes.IntPoint(o++).Set3(cp[i]/w, cp[0], cp[j]/w);
      }
   }
   for (int j = 1; j < p; j++)
   {
      for (int i = 1; i + j < p; i++)  // (0,2,1)
      {
         real_t w = cp[i] + cp[j] + cp[p-i-j];
         Nodes.IntPoint(o++).Set3(cp[j]/w, cp[i]/w, cp[0]);
      }
   }

   // Interior P_{q+4} nodes
   for (int k = 1; k < q + 4; k++)
   {
      for (int j = 1; j + k < q + 4; j++)
      {
         for (int i = 1; i + j + k < q + 4; i++)
         {
            real_t w = cp2[i] + cp2[j] + cp2[k] + cp2[q+4-i-j-k];
            Nodes.IntPoint(o++).Set3(cp2[i]/w, cp2[j]/w, cp2[k]/w);
         }
      }
   }

   DenseMatrix Tt(dof, npq);
   for (int m = 0; m < dof; ++m)
   {
      const IntegrationPoint &ip = Nodes.IntPoint(m);
      poly1d.CalcBasis(p, ip.x, shape_x);
      poly1d.CalcBasis(p, ip.y, shape_y);
      poly1d.CalcBasis(p, ip.z, shape_z);
      poly1d.CalcBasis(p, 1. - ip.x - ip.y - ip.z, shape_l);

      o = 0;
      for (int k = 0; k <= p; k++)
      {
         for (int j = 0; j + k <= p; j++)
         {
            for (int i = 0; i + j + k <= p; i++)
            {
               Tt(m, o++) = shape_x[i]*shape_y[j]*shape_z[k]*shape_l[p-i-j-k];
            }
         }
      }

      poly1d.CalcBasis(q, ip.x, shape_x);
      poly1d.CalcBasis(q, ip.y, shape_y);
      poly1d.CalcBasis(q, ip.z, shape_z);
      poly1d.CalcBasis(q, 1. - ip.x - ip.y - ip.z, shape_l);
      const real_t b_T = ip.x * ip.y * ip.z * (1 - ip.x - ip.y - ip.z);

      for (int k = 0; k <= q; k++)
      {
         for (int j = 0; j + k <= q; j++)
         {
            for (int i = 0; i + j + k <= q; i++)
            {
               Tt(m, o++) = b_T*shape_x[i]*shape_y[j]*shape_z[k]*shape_l[q-i-j-k];
            }
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

void H1Bubble_TetrahedronElement::CalcShape(const IntegrationPoint &ip,
                                            Vector &shape) const
{
   const int p = base_order;
   const int q = bubble_order;

#ifdef MFEM_THREAD_SAFE
   const int n1d = max(p + 1, q + 1);
   const int npq = ((p+1)*(p+2)*(p+3))/6 + ((q+1)*(q+2)*(q+3))/6;
   Vector shape_x(n1d), shape_y(n1d), shape_z(n1d), shape_l(n1d), u(npq);
#endif

   poly1d.CalcBasis(p, ip.x, shape_x);
   poly1d.CalcBasis(p, ip.y, shape_y);
   poly1d.CalcBasis(p, ip.z, shape_z);
   poly1d.CalcBasis(p, 1. - ip.x - ip.y - ip.z, shape_l);

   int o = 0;
   for (int k = 0; k <= p; k++)
   {
      for (int j = 0; j + k <= p; j++)
      {
         for (int i = 0; i + j + k <= p; i++)
         {
            u[o++] = shape_x[i]*shape_y[j]*shape_z[k]*shape_l[p-i-j-k];
         }
      }
   }

   poly1d.CalcBasis(q, ip.x, shape_x);
   poly1d.CalcBasis(q, ip.y, shape_y);
   poly1d.CalcBasis(q, ip.z, shape_z);
   poly1d.CalcBasis(q, 1. - ip.x - ip.y - ip.z, shape_l);
   const real_t b_T = ip.x * ip.y * ip.z * (1 - ip.x - ip.y - ip.z);

   for (int k = 0; k <= q; k++)
   {
      for (int j = 0; j + k <= q; j++)
      {
         for (int i = 0; i + j + k <= q; i++)
         {
            u(o++) = b_T*shape_x[i]*shape_y[j]*shape_z[k]*shape_l[q-i-j-k];
         }
      }
   }

   T_pinv.Mult(u, shape);
}

void H1Bubble_TetrahedronElement::CalcDShape(const IntegrationPoint &ip,
                                             DenseMatrix &dshape) const
{
   const int p = base_order;
   const int q = bubble_order;

#ifdef MFEM_THREAD_SAFE
   const int n1d = max(p+1, q+1);
   const int npq = ((p+1)*(p+2)*(p+3))/6 + ((q+1)*(q+2)*(q+3))/6;

   Vector shape_x(n1d), shape_y(n1d), shape_z(n1d), shape_l(n1d);
   Vector dshape_x(n1d), dshape_y(n1d), dshape_z(n1d), dshape_l(n1d);
   DenseMatrix du(npq, dim);
#endif

   const real_t lambda = 1.0 - ip.x - ip.y - ip.z;

   poly1d.CalcBasis(p, ip.x, shape_x, dshape_x);
   poly1d.CalcBasis(p, ip.y, shape_y, dshape_y);
   poly1d.CalcBasis(p, ip.z, shape_z, dshape_z);
   poly1d.CalcBasis(p, lambda, shape_l, dshape_l);

   int o = 0;
   for (int k = 0; k <= p; k++)
   {
      for (int j = 0; j + k <= p; j++)
      {
         for (int i = 0; i + j + k <= p; i++)
         {
            int l = p - i - j - k;
            du(o,0) = (dshape_x[i]*shape_l[l] - shape_x[i]*dshape_l[l])
                      *shape_y[j]*shape_z[k];
            du(o,1) = (dshape_y[j]*shape_l[l] - shape_y[j]*dshape_l[l])
                      *shape_x[i]*shape_z[k];
            du(o,2) = (dshape_z[k]*shape_l[l] - shape_z[k]*dshape_l[l])
                      *shape_x[i]*shape_y[j];
            o++;
         }
      }
   }

   poly1d.CalcBasis(q, ip.x, shape_x, dshape_x);
   poly1d.CalcBasis(q, ip.y, shape_y, dshape_y);
   poly1d.CalcBasis(q, ip.z, shape_z, dshape_z);
   poly1d.CalcBasis(q, lambda, shape_l, dshape_l);
   const real_t b_T = ip.x * ip.y * ip.z * (1 - ip.x - ip.y - ip.z);
   const real_t dxb_T = ip.y * ip.z * (lambda - ip.x);
   const real_t dyb_T = ip.x * ip.z * (lambda - ip.y);
   const real_t dzb_T = ip.x * ip.y * (lambda - ip.z);

   for (int k = 0; k <= q; k++)
   {
      for (int j = 0; j + k <= q; j++)
      {
         for (int i = 0; i + j + k <= q; i++)
         {
            int l = q - i - j - k;
            du(o,0) = shape_y[j]*shape_z[k]*(dxb_T*shape_x[i]*shape_l[l]
                                             + b_T*dshape_x[i]*shape_l[l]
                                             - b_T*shape_x[i]*dshape_l[l]);
            du(o,1) = shape_x[i]*shape_z[k]*(dyb_T*shape_y[j]*shape_l[l]
                                             + b_T*dshape_y[j]*shape_l[l]
                                             - b_T*shape_y[j]*dshape_l[l]);
            du(o,2) = shape_x[i]*shape_y[j]*(dzb_T*shape_z[k]*shape_l[l]
                                             + b_T*dshape_z[k]*shape_l[l]
                                             - b_T*shape_z[k]*dshape_l[l]);
            o++;
         }
      }
   }

   Mult(T_pinv, du, dshape);
}

H1Bubble_HexahedronElement::H1Bubble_HexahedronElement(
   int p, int q, int btype)
   : NodalFiniteElement(3, Geometry::CUBE, (2 + 6*p*p) + (q+1)*(q+1)*(q+1),
                        max(p, 2 + q), FunctionSpace::Qk),
     base_order(p), bubble_order(q)
{
   const real_t *cp = poly1d.ClosedPoints(p, VerifyNodal(VerifyClosed(btype)));
   const real_t *cp2 = poly1d.ClosedPoints(
                          q + 2, VerifyNodal(VerifyClosed(btype)));

   const int n1d = max(p + 1, q + 1);
   const int npq = (p+1)*(p+1)*(p+1) + (q+1)*(q+1)*(q+1);

#ifndef MFEM_THREAD_SAFE
   shape_x.SetSize(n1d);
   shape_y.SetSize(n1d);
   shape_z.SetSize(n1d);
   dshape_x.SetSize(n1d);
   dshape_y.SetSize(n1d);
   dshape_z.SetSize(n1d);

   u.SetSize(npq);
   du.SetSize(npq, dim);
#endif

   // vertices
   Nodes.IntPoint(0).Set3(cp[0], cp[0], cp[0]);
   Nodes.IntPoint(1).Set3(cp[p], cp[0], cp[0]);
   Nodes.IntPoint(2).Set3(cp[p], cp[p], cp[0]);
   Nodes.IntPoint(3).Set3(cp[0], cp[p], cp[0]);

   Nodes.IntPoint(4).Set3(cp[0], cp[0], cp[p]);
   Nodes.IntPoint(5).Set3(cp[p], cp[0], cp[p]);
   Nodes.IntPoint(6).Set3(cp[p], cp[p], cp[p]);
   Nodes.IntPoint(7).Set3(cp[0], cp[p], cp[p]);

   int o = 8;
   for (int i = 1; i < p; i++)
   {
      Nodes.IntPoint(o++).Set3(cp[i], cp[0], cp[0]);   // (0,1)
   }
   for (int i = 1; i < p; i++)
   {
      Nodes.IntPoint(o++).Set3(cp[p], cp[i], cp[0]);   // (1,2)
   }
   for (int i = 1; i < p; i++)
   {
      Nodes.IntPoint(o++).Set3(cp[i], cp[p], cp[0]);   // (3,2)
   }
   for (int i = 1; i < p; i++)
   {
      Nodes.IntPoint(o++).Set3(cp[0], cp[i], cp[0]);   // (0,3)
   }
   for (int i = 1; i < p; i++)
   {
      Nodes.IntPoint(o++).Set3(cp[i], cp[0], cp[p]);   // (4,5)
   }
   for (int i = 1; i < p; i++)
   {
      Nodes.IntPoint(o++).Set3(cp[p], cp[i], cp[p]);   // (5,6)
   }
   for (int i = 1; i < p; i++)
   {
      Nodes.IntPoint(o++).Set3(cp[i], cp[p], cp[p]);   // (7,6)
   }
   for (int i = 1; i < p; i++)
   {
      Nodes.IntPoint(o++).Set3(cp[0], cp[i], cp[p]);   // (4,7)
   }
   for (int i = 1; i < p; i++)
   {
      Nodes.IntPoint(o++).Set3(cp[0], cp[0], cp[i]);   // (0,4)
   }
   for (int i = 1; i < p; i++)
   {
      Nodes.IntPoint(o++).Set3(cp[p], cp[0], cp[i]);   // (1,5)
   }
   for (int i = 1; i < p; i++)
   {
      Nodes.IntPoint(o++).Set3(cp[p], cp[p], cp[i]);   // (2,6)
   }
   for (int i = 1; i < p; i++)
   {
      Nodes.IntPoint(o++).Set3(cp[0], cp[p], cp[i]);   // (3,7)
   }

   // faces
   for (int j = 1; j < p; j++)
   {
      for (int i = 1; i < p; i++)
      {
         Nodes.IntPoint(o++).Set3(cp[i], cp[p-j], cp[0]);   // (3,2,1,0)
      }
   }
   for (int j = 1; j < p; j++)
   {
      for (int i = 1; i < p; i++)
      {
         Nodes.IntPoint(o++).Set3(cp[i], cp[0], cp[j]);   // (0,1,5,4)
      }
   }
   for (int j = 1; j < p; j++)
   {
      for (int i = 1; i < p; i++)
      {
         Nodes.IntPoint(o++).Set3(cp[p], cp[i], cp[j]);   // (1,2,6,5)
      }
   }
   for (int j = 1; j < p; j++)
   {
      for (int i = 1; i < p; i++)
      {
         Nodes.IntPoint(o++).Set3(cp[p-i], cp[p], cp[j]);   // (2,3,7,6)
      }
   }
   for (int j = 1; j < p; j++)
   {
      for (int i = 1; i < p; i++)
      {
         Nodes.IntPoint(o++).Set3(cp[0], cp[p-i], cp[j]);   // (3,0,4,7)
      }
   }
   for (int j = 1; j < p; j++)
   {
      for (int i = 1; i < p; i++)
      {
         Nodes.IntPoint(o++).Set3(cp[i], cp[j], cp[p]);   // (4,5,6,7)
      }
   }

   // interior P_{q+2} nodes
   for (int k = 1; k < q+2; k++)
   {
      for (int j = 1; j < q+2; j++)
      {
         for (int i = 1; i < q+2; i++)
         {
            Nodes.IntPoint(o++).Set3(cp2[i], cp2[j], cp2[k]);
         }
      }
   }

#ifdef MFEM_THREAD_SAFE
   Vector shape_x(n1d), shape_y(n1d), shape_z(n1d);
#endif

   DenseMatrix Tt(dof, npq);
   for (int m = 0; m < dof; ++m)
   {
      const IntegrationPoint &ip = Nodes.IntPoint(m);
      poly1d.CalcBasis(p, ip.x, shape_x);
      poly1d.CalcBasis(p, ip.y, shape_y);
      poly1d.CalcBasis(p, ip.z, shape_z);

      o = 0;
      for (int k = 0; k <= p; k++)
      {
         for (int j = 0; j <= p; j++)
         {
            for (int i = 0; i <= p; i++)
            {
               Tt(m, o++) = shape_x[i]*shape_y[j]*shape_z[k];
            }
         }
      }

      poly1d.CalcBasis(q, ip.x, shape_x);
      poly1d.CalcBasis(q, ip.y, shape_y);
      poly1d.CalcBasis(q, ip.z, shape_z);
      const real_t b_T = ip.x*(1.0 - ip.x)*ip.y*(1.0 - ip.y)*ip.z*(1.0 - ip.z);
      for (int k = 0; k <= q; k++)
      {
         for (int j = 0; j <= q; j++)
         {
            for (int i = 0; i <= q; i++)
            {
               Tt(m, o++) = b_T*shape_x[i]*shape_y[j]*shape_z[k];
            }
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

void H1Bubble_HexahedronElement::CalcShape(const IntegrationPoint &ip,
                                           Vector &shape) const
{
   const int p = base_order;
   const int q = bubble_order;

#ifdef MFEM_THREAD_SAFE
   const int n1d = max(p + 1, q + 1);
   const int npq = (p+1)*(p+1)*(p+1) + (q+1)*(q+1)*(q+1);
   Vector shape_x(n1d), shape_y(n1d), shape_z(n1d), u(npq);
#endif

   poly1d.CalcBasis(p, ip.x, shape_x);
   poly1d.CalcBasis(p, ip.y, shape_y);
   poly1d.CalcBasis(p, ip.z, shape_z);

   int o = 0;
   for (int k = 0; k <= p; k++)
   {
      for (int j = 0; j <= p; j++)
      {
         for (int i = 0; i <= p; i++)
         {
            u(o++) = shape_x[i]*shape_y[j]*shape_z[k];
         }
      }
   }

   poly1d.CalcBasis(q, ip.x, shape_x);
   poly1d.CalcBasis(q, ip.y, shape_y);
   poly1d.CalcBasis(q, ip.z, shape_z);
   const real_t b_T = ip.x*(1.0 - ip.x)*ip.y*(1.0 - ip.y)*ip.z*(1.0 - ip.z);

   for (int k = 0; k <= q; k++)
   {
      for (int j = 0; j <= q; j++)
      {
         for (int i = 0; i <= q; i++)
         {
            u(o++) = b_T*shape_x[i]*shape_y[j]*shape_z[k];
         }
      }
   }

   T_pinv.Mult(u, shape);
}

void H1Bubble_HexahedronElement::CalcDShape(const IntegrationPoint &ip,
                                            DenseMatrix &dshape) const
{
   const int p = base_order;
   const int q = bubble_order;

#ifdef MFEM_THREAD_SAFE
   const int n1d = max(p + 1, q + 1);
   const int npq = (p+1)*(p+1)*(p+1) + (q+1)*(q+1)*(q+1);
   Vector shape_x(n1d), shape_y(n1d), shape_z(n1d), dshape_x(n1d),
          dshape_y(n1d), dshape_z(n1d);
   DenseMatrix du(npq, dim);
#endif

   poly1d.CalcBasis(p, ip.x, shape_x, dshape_x);
   poly1d.CalcBasis(p, ip.y, shape_y, dshape_y);
   poly1d.CalcBasis(p, ip.z, shape_z, dshape_z);

   int o = 0;
   for (int k = 0; k <= p; k++)
   {
      for (int j = 0; j <= p; j++)
      {
         for (int i = 0; i <= p; i++)
         {
            du(o,0) = dshape_x[i]*shape_y[j]*shape_z[k];
            du(o,1) = shape_x[i]*dshape_y[j]*shape_z[k];
            du(o,2) = shape_x[i]*shape_y[j]*dshape_z[k];
            o += 1;
         }
      }
   }

   poly1d.CalcBasis(q, ip.x, shape_x, dshape_x);
   poly1d.CalcBasis(q, ip.y, shape_y, dshape_y);
   poly1d.CalcBasis(q, ip.z, shape_z, dshape_z);
   const real_t b_T = ip.x*(1.0 - ip.x)*ip.y*(1.0 - ip.y)*ip.z*(1.0 - ip.z);
   const real_t dxb_T = (1.0 - 2*ip.x)*ip.y*(1.0 - ip.y)*ip.z*(1.0 - ip.z);
   const real_t dyb_T = ip.x*(1.0 - ip.x)*(1.0 - 2*ip.y)*ip.z*(1.0 - ip.z);
   const real_t dzb_T = ip.x*(1.0 - ip.x)*ip.y*(1.0 - ip.y)*(1.0 - 2*ip.z);

   for (int k = 0; k <= q; k++)
   {
      for (int j = 0; j <= q; j++)
      {
         for (int i = 0; i <= q; i++)
         {
            du(o,0) = (dxb_T*shape_x[i] + b_T*dshape_x[i])*shape_y[j]*shape_z[k];
            du(o,1) = (dyb_T*shape_y[j] + b_T*dshape_y[j])*shape_x[i]*shape_z[k];
            du(o,2) = (dzb_T*shape_z[k] + b_T*dshape_z[k])*shape_x[i]*shape_y[j];
            o += 1;
         }
      }
   }

   Mult(T_pinv, du, dshape);
}

}
