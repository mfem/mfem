// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
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
#include <fstream>


namespace mfem
{

using namespace std;

H1_SegmentElement::H1_SegmentElement(const int p, const int btype)
   : NodalTensorFiniteElement(1, p, VerifyClosed(btype), H1_DOF_MAP)
{
   const real_t *cp = poly1d.ClosedPoints(p, b_type);

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
   const real_t *cp = poly1d.ClosedPoints(p, b_type);

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
   const real_t *cp = poly1d.ClosedPoints(p, b_type);

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
   const real_t *cp = poly1d.ClosedPoints(p, b_type);

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
   const real_t *cp = poly1d.ClosedPoints(p, b_type);

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
   const real_t *cp = poly1d.ClosedPoints(p,b_type);

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
   const real_t *cp = poly1d.ClosedPoints(p, VerifyNodal(VerifyClosed(btype)));

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
         const real_t w = cp[i] + cp[j] + cp[p-i-j];
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
    
        for (int i =0; i<dof; i++) {
            //for (int j = 0; j<dim; j++) {
                std::cout << shape(i) << ",";
            //}
            std::cout << std::endl;
        }
    std::cout << "end" << std::endl;
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
   DenseMatrix ddu(dof, (dim*(dim+1))/2);
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
   const real_t *cp = poly1d.ClosedPoints(p, VerifyNodal(VerifyClosed(btype)));

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
         real_t w = cp[i] + cp[j] + cp[p-i-j];
         Nodes.IntPoint(o++).Set3(cp[p-i-j]/w, cp[i]/w, cp[j]/w);
      }
   for (int j = 1; j < p; j++)
      for (int i = 1; i + j < p; i++)  // (0,3,2)
      {
         lex_ordering[idx(0,j,i)] = o;
         real_t w = cp[i] + cp[j] + cp[p-i-j];
         Nodes.IntPoint(o++).Set3(cp[0], cp[j]/w, cp[i]/w);
      }
   for (int j = 1; j < p; j++)
      for (int i = 1; i + j < p; i++)  // (0,1,3)
      {
         lex_ordering[idx(i,0,j)] = o;
         real_t w = cp[i] + cp[j] + cp[p-i-j];
         Nodes.IntPoint(o++).Set3(cp[i]/w, cp[0], cp[j]/w);
      }
   for (int j = 1; j < p; j++)
      for (int i = 1; i + j < p; i++)  // (0,2,1)
      {
         lex_ordering[idx(j,i,0)] = o;
         real_t w = cp[i] + cp[j] + cp[p-i-j];
         Nodes.IntPoint(o++).Set3(cp[j]/w, cp[i]/w, cp[0]);
      }

   // interior
   for (int k = 1; k < p; k++)
      for (int j = 1; j + k < p; j++)
         for (int i = 1; i + j + k < p; i++)
         {
            lex_ordering[idx(i,j,k)] = o;
            real_t w = cp[i] + cp[j] + cp[k] + cp[p-i-j-k];
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
    
        for (int i =0; i<dof; i++) {
            //for (int j = 0; j<dim; j++) {
                std::cout << shape(i) << ",";
            //}
            std::cout << std::endl;
        }
    std::cout << "end" << std::endl;
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

H1_PentatopeElement::H1_PentatopeElement(const int p, const int type)
   : NodalFiniteElement(4, Geometry::PENTATOPE,
                        ((p + 1)*(p + 2)*(p + 3)*(p + 4))/24,
                        p, FunctionSpace::Pk)
{
   const double *cp = poly1d.ClosedPoints(p, VerifyClosed(type));

#ifndef MFEM_THREAD_SAFE
   shape_x.SetSize(p + 1);
   shape_y.SetSize(p + 1);
   shape_z.SetSize(p + 1);
   shape_t.SetSize(p + 1);
   shape_l.SetSize(p + 1);
   dshape_x.SetSize(p + 1);
   dshape_y.SetSize(p + 1);
   dshape_z.SetSize(p + 1);
   dshape_t.SetSize(p + 1);
   dshape_l.SetSize(p + 1);
   ddshape_x.SetSize(p + 1);
   ddshape_y.SetSize(p + 1);
   ddshape_z.SetSize(p + 1);
   ddshape_t.SetSize(p + 1);
   ddshape_l.SetSize(p + 1);
   u.SetSize(dof);
   du.SetSize(dof, dim);
   ddu.SetSize(dof,dim*(dim+1)/2 );
#else
   Vector shape_x(p + 1), shape_y(p + 1), shape_z(p + 1), shape_t(p+1),
          shape_l(p + 1);
#endif

   // vertices
   Nodes.IntPoint(0).Set4(cp[0], cp[0], cp[0], cp[0]);
   Nodes.IntPoint(1).Set4(cp[p], cp[0], cp[0], cp[0]);
   Nodes.IntPoint(2).Set4(cp[0], cp[p], cp[0], cp[0]);
   Nodes.IntPoint(3).Set4(cp[0], cp[0], cp[p], cp[0]);
   Nodes.IntPoint(4).Set4(cp[0], cp[0], cp[0], cp[p]);

   // edges (see Tetrahedron::edges in mesh/tetrahedron.cpp)
   int o = 5;
   for (int i = 1; i < p; i++)  // (0,1)
   {
      Nodes.IntPoint(o++).Set4(cp[i], cp[0], cp[0], cp[0]);
   }
   for (int i = 1; i < p; i++)  // (0,2)
   {
      Nodes.IntPoint(o++).Set4(cp[0], cp[i], cp[0], cp[0]);
   }
   for (int i = 1; i < p; i++)  // (0,3)
   {
      Nodes.IntPoint(o++).Set4(cp[0], cp[0], cp[i], cp[0]);
   }
   for (int i = 1; i < p; i++)  // (0,4)
   {
      Nodes.IntPoint(o++).Set4(cp[0], cp[0], cp[0], cp[i]);
   }
   for (int i = 1; i < p; i++)  // (1,2)
   {
      Nodes.IntPoint(o++).Set4(cp[p-i], cp[i], cp[0], cp[0]);
   }
   for (int i = 1; i < p; i++)  // (1,3)
   {
      Nodes.IntPoint(o++).Set4(cp[p-i], cp[0], cp[i], cp[0]);
   }
   for (int i = 1; i < p; i++)  // (1,4)
   {
      Nodes.IntPoint(o++).Set4(cp[p-i], cp[0], cp[0], cp[i]);
   }
   for (int i = 1; i < p; i++)  // (2,3)
   {
      Nodes.IntPoint(o++).Set4(cp[0], cp[p-i], cp[i], cp[0]);
   }
   for (int i = 1; i < p; i++)  // (2,4)
   {
      Nodes.IntPoint(o++).Set4(cp[0], cp[p-i], cp[0], cp[i]);
   }
   for (int i = 1; i < p; i++)  // (3,4)
   {
      Nodes.IntPoint(o++).Set4(cp[0], cp[0], cp[p-i], cp[i]);
   }

   // planars (see Mesh::GeneratePlanars in mesh/mesh.cpp)
   for (int j = 1; j < p; j++)
      for (int i=1; i + j < p; i++) // (0,1,2)
      {
         double w = cp[i] + cp[j] + cp[p-i-j];
         Nodes.IntPoint(o++).Set4(cp[i]/w, cp[j]/w, cp[0], cp[0]);
      }
   for (int j = 1; j < p; j++)
      for (int i=1; i + j < p; i++) // (0,1,3)
      {
         double w = cp[i] + cp[j] + cp[p-i-j];
         Nodes.IntPoint(o++).Set4(cp[i]/w, cp[0], cp[j]/w, cp[0]);
      }
   for (int j = 1; j < p; j++)
      for (int i=1; i + j < p; i++) // (0,1,4)
      {
         double w = cp[i] + cp[j] + cp[p-i-j];
         Nodes.IntPoint(o++).Set4(cp[i]/w, cp[0], cp[0], cp[j]/w);
      }
   for (int j = 1; j < p; j++)
      for (int i=1; i + j < p; i++) // (0,2,3)
      {
         double w = cp[i] + cp[j] + cp[p-i-j];
         Nodes.IntPoint(o++).Set4(cp[0], cp[i]/w, cp[j]/w, cp[0]);
      }
   for (int j = 1; j < p; j++)
      for (int i=1; i + j < p; i++) // (0,2,4)
      {
         double w = cp[i] + cp[j] + cp[p-i-j];
         Nodes.IntPoint(o++).Set4(cp[0], cp[i]/w, cp[0], cp[j]/w);
      }
   for (int j = 1; j < p; j++)
      for (int i=1; i + j < p; i++) // (0,3,4)
      {
         double w = cp[i] + cp[j] + cp[p-i-j];
         Nodes.IntPoint(o++).Set4(cp[0], cp[0], cp[i]/w, cp[j]/w);
      }
   for (int j = 1; j < p; j++)
      for (int i=1; i + j < p; i++) // (1,2,3)
      {
         double w = cp[i] + cp[j] + cp[p-i-j];
         Nodes.IntPoint(o++).Set4(cp[p-i-j]/w, cp[i]/w, cp[j]/w, cp[0]);
      }
   for (int j = 1; j < p; j++)
      for (int i=1; i + j < p; i++) // (1,2,4)
      {
         double w = cp[i] + cp[j] + cp[p-i-j];
         Nodes.IntPoint(o++).Set4(cp[p-i-j]/w, cp[i]/w, cp[0], cp[j]/w);
      }
   for (int j = 1; j < p; j++)
      for (int i=1; i + j < p; i++) // (1,3,4)
      {
         double w = cp[i] + cp[j] + cp[p-i-j];
         Nodes.IntPoint(o++).Set4(cp[p-i-j]/w, cp[0], cp[i]/w, cp[j]/w);
      }
   for (int j = 1; j < p; j++)
      for (int i=1; i + j < p; i++) // (2,3,4)
      {
         double w = cp[i] + cp[j] + cp[p-i-j];
         Nodes.IntPoint(o++).Set4(cp[0], cp[p-i-j]/w, cp[i]/w, cp[j]/w);
      }

   // face(volumes)s (see Mesh::GenerateFaces in mesh/mesh.cpp)
   for (int k = 1; k < p; k++)
      for (int j = 1; j + k < p; j++)
         for (int i = 1; i + j + k < p; i++)  // (0,1,2,3)
         {
            double w = cp[i] + cp[j] + cp[k] + cp[p-i-j-k];
            Nodes.IntPoint(o++).Set4(cp[i]/w, cp[j]/w, cp[k]/w, cp[0]);
         }
   for (int k = 1; k < p; k++)
      for (int j = 1; j + k < p; j++)
         for (int i = 1; i + j + k < p; i++)  // (0,2,1,4)
         {
            double w = cp[i] + cp[j] + cp[k] + cp[p-i-j-k];
            Nodes.IntPoint(o++).Set4(cp[j]/w, cp[i]/w, cp[0], cp[k]/w);
         }
   for (int k = 1; k < p; k++)
      for (int j = 1; j + k < p; j++)
         for (int i = 1; i + j + k < p; i++)  // (0,1,3,4)
         {
            double w = cp[i] + cp[j] + cp[k] + cp[p-i-j-k];
            Nodes.IntPoint(o++).Set4(cp[i]/w, cp[0], cp[j]/w, cp[k]/w);
         }
   for (int k = 1; k < p; k++)
      for (int j = 1; j + k < p; j++)
         for (int i = 1; i + j + k < p; i++)  // (0,3,2,4)
         {
            double w = cp[i] + cp[j] + cp[k] + cp[p-i-j-k];
            Nodes.IntPoint(o++).Set4(cp[0], cp[j]/w, cp[i]/w, cp[k]/w);
         }
   for (int k = 1; k < p; k++)
      for (int j = 1; j + k < p; j++)
         for (int i = 1; i + j + k < p; i++)  // (1,2,3,4)
         {
            double w = cp[i] + cp[j] + cp[k] + cp[p-i-j-k];
            Nodes.IntPoint(o++).Set4(cp[p-i-j-k]/w, cp[i]/w, cp[j]/w, cp[k]/w);
         }

   // interior
   for (int l = 1; l < p; l++)
      for (int k = 1; k + l < p; k++)
         for (int j = 1; j + k + l < p; j++)
            for (int i = 1; i + j + k + l < p; i++)
            {
               double w = cp[i] + cp[j] + cp[k] + cp[l] + cp[p-i-j-k-l];
               Nodes.IntPoint(o++).Set4(cp[i]/w, cp[j]/w, cp[k]/w, cp[l]/w);
            }

   DenseMatrix T(dof);
   for (int m = 0; m < dof; m++)
   {
      IntegrationPoint &ip = Nodes.IntPoint(m);
      poly1d.CalcBasis(p, ip.x, shape_x);
      poly1d.CalcBasis(p, ip.y, shape_y);
      poly1d.CalcBasis(p, ip.z, shape_z);
      poly1d.CalcBasis(p, ip.t, shape_t);
      poly1d.CalcBasis(p, 1. - ip.x - ip.y - ip.z - ip.t, shape_l);

      o = 0;
      for (int l = 0; l <= p; l++)
         for (int k = 0; k + l <= p; k++)
            for (int j = 0; j + k +l <= p; j++)
               for (int i = 0; i + j + k + l <= p; i++)
               {
                  T(o++, m) = shape_x(i)*shape_y(j)*shape_z(k)*shape_t(l)*shape_l(p-i-j-k-l);
               }
   }

   Ti.Factor(T);
   // cout << "H1_PentatopeElement(" << p << ") : "; Ti.TestInversion();
}

void H1_PentatopeElement::CalcShape(const IntegrationPoint &ip,
                                    Vector &shape) const
{
   const int p = order;

#ifdef MFEM_THREAD_SAFE
   Vector shape_x(p + 1), shape_y(p + 1), shape_z(p + 1), shape_t(p+1),
          shape_l(p + 1);
   Vector u(Dof);
#endif

   poly1d.CalcBasis(p, ip.x, shape_x);
   poly1d.CalcBasis(p, ip.y, shape_y);
   poly1d.CalcBasis(p, ip.z, shape_z);
   poly1d.CalcBasis(p, ip.t, shape_t);
   poly1d.CalcBasis(p, 1. - ip.x - ip.y - ip.z - ip.t, shape_l);

   for (int o = 0, l = 0; l <= p; l++)
      for (int k = 0; k + l <= p; k++)
         for (int j = 0; j + k + l <= p; j++)
            for (int i = 0; i + j + k + l <= p; i++)
            {
               u(o++) = shape_x(i)*shape_y(j)*shape_z(k)*shape_t(l)*shape_l(p-i-j-k-l);
            }

   Ti.Mult(u, shape);
}

void H1_PentatopeElement::CalcDShape(const IntegrationPoint &ip,
                                     DenseMatrix &dshape) const
{
   const int p = order;

#ifdef MFEM_THREAD_SAFE
   Vector  shape_x(p + 1),  shape_y(p + 1),  shape_z(p + 1),  shape_t(p+1),
           shape_l(p + 1);
   Vector dshape_x(p + 1), dshape_y(p + 1), dshape_z(p + 1), dshape_t(p+1),
          dshape_l(p + 1);
   DenseMatrix du(Dof, Dim);
#endif

   poly1d.CalcBasis(p, ip.x, shape_x, dshape_x);
   poly1d.CalcBasis(p, ip.y, shape_y, dshape_y);
   poly1d.CalcBasis(p, ip.z, shape_z, dshape_z);
   poly1d.CalcBasis(p, ip.t, shape_t, dshape_t);
   poly1d.CalcBasis(p, 1. - ip.x - ip.y - ip.z - ip.t, shape_l, dshape_l);

   for (int o = 0, l = 0; l <= p; l++)
      for (int k = 0; k + l <= p; k++)
         for (int j = 0; j + k + l <= p; j++)
            for (int i = 0; i + j + k + l <= p; i++)
            {
               int m = p - i - j - k - l;
               du(o,0) = ((dshape_x(i)* shape_l(m)) -
                          ( shape_x(i)*dshape_l(m)))*shape_y(j)*shape_z(k)*shape_t(l);
               du(o,1) = ((dshape_y(j)* shape_l(m)) -
                          ( shape_y(j)*dshape_l(m)))*shape_x(i)*shape_z(k)*shape_t(l);
               du(o,2) = ((dshape_z(k)* shape_l(m)) -
                          ( shape_z(k)*dshape_l(m)))*shape_x(i)*shape_y(j)*shape_t(l);
               du(o,3) = ((dshape_t(l)* shape_l(m)) -
                          ( shape_t(l)*dshape_l(m)))*shape_x(i)*shape_y(j)*shape_z(k);
               o++;
            }

   Ti.Mult(du, dshape);
}

void H1_PentatopeElement::CalcHessian(const IntegrationPoint &ip,
                                      DenseMatrix &ddshape) const
{
   const int p = order;
#ifdef MFEM_THREAD_SAFE
   Vector   shape_x(p + 1),   shape_y(p + 1),   shape_z(p + 1),   shape_t(p+1),
            shape_l(p + 1);
   Vector  dshape_x(p + 1),  dshape_y(p + 1),  dshape_z(p + 1),  dshape_t(p+1),
           dshape_l(p + 1);
   Vector ddshape_x(p + 1), ddshape_y(p + 1), ddshape_z(p + 1), ddshape_t(p+1),
          ddshape_l(p + 1);
   DenseMatrix ddu(Dof, ((Dim+1)*Dim)/2);
#endif

   poly1d.CalcBasis(p, ip.x, shape_x, dshape_x, ddshape_x);
   poly1d.CalcBasis(p, ip.y, shape_y, dshape_y, ddshape_y);
   poly1d.CalcBasis(p, ip.z, shape_z, dshape_z, ddshape_z);
   poly1d.CalcBasis(p, ip.t, shape_t, dshape_t, ddshape_t);
   poly1d.CalcBasis(p, 1. - ip.x - ip.y - ip.z - ip.t, shape_l, dshape_l,
                    ddshape_l);

   for (int o = 0, l = 0; l <= p; l++)
      for (int k = 0; k + l <= p; k++)
         for (int j = 0; j + k + l <= p; j++)
            for (int i = 0; i + j + k + l <= p; i++)
            {
               // u_xx, u_xy, u_xz, u_xt, u_yy, u_yz, u_yt, u_zz, u_zt, u_tt
               int m = p - i - j - k - l;
               ddu(o,0) = ((ddshape_x(i)*shape_l(m)) - 2.* (dshape_x(i)*dshape_l(m)) +
                           (shape_x(i)*ddshape_l(m))) * shape_y(j) * shape_z(k) * shape_t(l);
               ddu(o,1) = ((dshape_y(j)* ( (dshape_x(i)*shape_l(m)) - (shape_x(i)*dshape_l(
                                                                          m))) ) + (shape_y(j)* ((ddshape_l(m)*shape_x(i)) - (dshape_x(i) * dshape_l(
                                                                                       m)) ) ) )* shape_z(k) * shape_t(l);
               ddu(o,2) = ((dshape_z(k)* ( (dshape_x(i)*shape_l(m)) - (shape_x(i)*dshape_l(
                                                                          m))) ) + (shape_z(k)* ((ddshape_l(m)*shape_x(i)) - (dshape_x(i) * dshape_l(
                                                                                       m)) ) ) )* shape_y(j) * shape_t(l);
               ddu(o,3) = ((dshape_t(l)* ( (dshape_x(i)*shape_l(m)) - (shape_x(i)*dshape_l(
                                                                          m))) ) + (shape_t(l)* ((ddshape_l(m)*shape_x(i)) - (dshape_x(i) * dshape_l(
                                                                                       m)) ) ) )* shape_y(j) * shape_z(k);
               ddu(o,4) = ((ddshape_y(j)*shape_l(m)) - 2.* (dshape_y(j)*dshape_l(m)) +
                           (shape_y(j)*ddshape_l(m))) * shape_x(i) * shape_z(k) * shape_t(l);
               ddu(o,5) = ((dshape_z(k)* ( (dshape_y(j)*shape_l(m)) - (shape_y(j)*dshape_l(
                                                                          m))) ) + (shape_z(k)* ((ddshape_l(m)*shape_y(j)) - (dshape_y(j) * dshape_l(
                                                                                       m)) ) ) )* shape_x(i) * shape_t(l);
               ddu(o,6) = ((dshape_t(l)* ( (dshape_y(j)*shape_l(m)) - (shape_y(j)*dshape_l(
                                                                          m))) ) + (shape_t(l)* ((ddshape_l(m)*shape_y(j)) - (dshape_y(j) * dshape_l(
                                                                                       m)) ) ) )* shape_x(i) * shape_z(k);
               ddu(o,7) = ((ddshape_z(k)*shape_l(m)) - 2.* (dshape_z(k)*dshape_l(m)) +
                           (shape_z(k)*ddshape_l(m))) * shape_y(j) * shape_x(i) * shape_t(l);
               ddu(o,8) = ((dshape_t(l)* ( (dshape_z(k)*shape_l(m)) - (shape_z(k)*dshape_l(
                                                                          m))) ) + (shape_t(l)* ((ddshape_l(m)*shape_z(k)) - (dshape_z(k) * dshape_l(
                                                                                       m)) ) ) )* shape_x(i) * shape_y(j);
               ddu(o,9) = ((ddshape_t(l)*shape_l(m)) - 2.* (dshape_t(l)*dshape_l(m)) +
                           (shape_t(l)*ddshape_l(m))) * shape_y(j) * shape_x(i) * shape_z(k);
               o++;
            }
   Ti.Mult(ddu, ddshape);
}

H1_PentatopeElement_Barycentric::H1_PentatopeElement_Barycentric(const int p, const int type)
   : NodalFiniteElement(4, Geometry::PENTATOPE,
                        ((p + 1)*(p + 2)*(p + 3)*(p + 4))/24,
                        p, FunctionSpace::Pk)
{
   const double *cp = poly1d.ClosedPoints(p, VerifyClosed(type));

#ifndef MFEM_THREAD_SAFE
   shape_x.SetSize(p + 1);
   shape_y.SetSize(p + 1);
   shape_z.SetSize(p + 1);
   shape_t.SetSize(p + 1);
   shape_l.SetSize(p + 1);
   dshape_x.SetSize(p + 1);
   dshape_y.SetSize(p + 1);
   dshape_z.SetSize(p + 1);
   dshape_t.SetSize(p + 1);
   dshape_l.SetSize(p + 1);
   ddshape_x.SetSize(p + 1);
   ddshape_y.SetSize(p + 1);
   ddshape_z.SetSize(p + 1);
   ddshape_t.SetSize(p + 1);
   ddshape_l.SetSize(p + 1);
   u.SetSize(dof);
   du.SetSize(dof, dim);
   ddu.SetSize(dof,dim*(dim+1)/2 );
#else
   Vector shape_x(p + 1), shape_y(p + 1), shape_z(p + 1), shape_t(p+1),
          shape_l(p + 1);
#endif

   // vertices
   Nodes.IntPoint(0).Set4(cp[0], cp[0], cp[0], cp[0]);
   Nodes.IntPoint(1).Set4(cp[p], cp[0], cp[0], cp[0]);
   Nodes.IntPoint(2).Set4(cp[0], cp[p], cp[0], cp[0]);
   Nodes.IntPoint(3).Set4(cp[0], cp[0], cp[p], cp[0]);
   Nodes.IntPoint(4).Set4(cp[0], cp[0], cp[0], cp[p]);

   // edges (see Tetrahedron::edges in mesh/tetrahedron.cpp)
   int o = 5;
   for (int i = 1; i < p; i++)  // (0,1)
   {
      Nodes.IntPoint(o++).Set4(cp[i], cp[0], cp[0], cp[0]);
   }
   for (int i = 1; i < p; i++)  // (0,2)
   {
      Nodes.IntPoint(o++).Set4(cp[0], cp[i], cp[0], cp[0]);
   }
   for (int i = 1; i < p; i++)  // (0,3)
   {
      Nodes.IntPoint(o++).Set4(cp[0], cp[0], cp[i], cp[0]);
   }
   for (int i = 1; i < p; i++)  // (0,4)
   {
      Nodes.IntPoint(o++).Set4(cp[0], cp[0], cp[0], cp[i]);
   }
   for (int i = 1; i < p; i++)  // (1,2)
   {
      Nodes.IntPoint(o++).Set4(cp[p-i], cp[i], cp[0], cp[0]);
   }
   for (int i = 1; i < p; i++)  // (1,3)
   {
      Nodes.IntPoint(o++).Set4(cp[p-i], cp[0], cp[i], cp[0]);
   }
   for (int i = 1; i < p; i++)  // (1,4)
   {
      Nodes.IntPoint(o++).Set4(cp[p-i], cp[0], cp[0], cp[i]);
   }
   for (int i = 1; i < p; i++)  // (2,3)
   {
      Nodes.IntPoint(o++).Set4(cp[0], cp[p-i], cp[i], cp[0]);
   }
   for (int i = 1; i < p; i++)  // (2,4)
   {
      Nodes.IntPoint(o++).Set4(cp[0], cp[p-i], cp[0], cp[i]);
   }
   for (int i = 1; i < p; i++)  // (3,4)
   {
      Nodes.IntPoint(o++).Set4(cp[0], cp[0], cp[p-i], cp[i]);
   }

   // faces (see Mesh::GeneratePlanars in mesh/mesh.cpp)
   for (int j = 1; j < p; j++)
      for (int i=1; i + j < p; i++) // (0,1,2)
      {
         double w = cp[i] + cp[j] + cp[p-i-j];
         Nodes.IntPoint(o++).Set4(cp[i]/w, cp[j]/w, cp[0], cp[0]);
      }
   for (int j = 1; j < p; j++)
      for (int i=1; i + j < p; i++) // (0,1,3)
      {
         double w = cp[i] + cp[j] + cp[p-i-j];
         Nodes.IntPoint(o++).Set4(cp[i]/w, cp[0], cp[j]/w, cp[0]);
      }
   for (int j = 1; j < p; j++)
      for (int i=1; i + j < p; i++) // (0,1,4)
      {
         double w = cp[i] + cp[j] + cp[p-i-j];
         Nodes.IntPoint(o++).Set4(cp[i]/w, cp[0], cp[0], cp[j]/w);
      }
   for (int j = 1; j < p; j++)
      for (int i=1; i + j < p; i++) // (0,2,3)
      {
         double w = cp[i] + cp[j] + cp[p-i-j];
         Nodes.IntPoint(o++).Set4(cp[0], cp[i]/w, cp[j]/w, cp[0]);
      }
   for (int j = 1; j < p; j++)
      for (int i=1; i + j < p; i++) // (0,2,4)
      {
         double w = cp[i] + cp[j] + cp[p-i-j];
         Nodes.IntPoint(o++).Set4(cp[0], cp[i]/w, cp[0], cp[j]/w);
      }
   for (int j = 1; j < p; j++)
      for (int i=1; i + j < p; i++) // (0,3,4)
      {
         double w = cp[i] + cp[j] + cp[p-i-j];
         Nodes.IntPoint(o++).Set4(cp[0], cp[0], cp[i]/w, cp[j]/w);
      }
   for (int j = 1; j < p; j++)
      for (int i=1; i + j < p; i++) // (1,2,3)
      {
         double w = cp[i] + cp[j] + cp[p-i-j];
         Nodes.IntPoint(o++).Set4(cp[p-i-j]/w, cp[i]/w, cp[j]/w, cp[0]);
      }
   for (int j = 1; j < p; j++)
      for (int i=1; i + j < p; i++) // (1,2,4)
      {
         double w = cp[i] + cp[j] + cp[p-i-j];
         Nodes.IntPoint(o++).Set4(cp[p-i-j]/w, cp[i]/w, cp[0], cp[j]/w);
      }
   for (int j = 1; j < p; j++)
      for (int i=1; i + j < p; i++) // (1,3,4)
      {
         double w = cp[i] + cp[j] + cp[p-i-j];
         Nodes.IntPoint(o++).Set4(cp[p-i-j]/w, cp[0], cp[i]/w, cp[j]/w);
      }
   for (int j = 1; j < p; j++)
      for (int i=1; i + j < p; i++) // (2,3,4)
      {
         double w = cp[i] + cp[j] + cp[p-i-j];
         Nodes.IntPoint(o++).Set4(cp[0], cp[p-i-j]/w, cp[i]/w, cp[j]/w);
      }

   // facets (see Mesh::GenerateFaces in mesh/mesh.cpp)
   for (int k = 1; k < p; k++)
      for (int j = 1; j + k < p; j++)
         for (int i = 1; i + j + k < p; i++)  // (0,1,2,3)
         {
            double w = cp[i] + cp[j] + cp[k] + cp[p-i-j-k];
            Nodes.IntPoint(o++).Set4(cp[i]/w, cp[j]/w, cp[k]/w, cp[0]);
         }
   for (int k = 1; k < p; k++)
      for (int j = 1; j + k < p; j++)
         for (int i = 1; i + j + k < p; i++)  // (0,2,1,4)
         {
            double w = cp[i] + cp[j] + cp[k] + cp[p-i-j-k];
            Nodes.IntPoint(o++).Set4(cp[j]/w, cp[i]/w, cp[0], cp[k]/w);
         }
   for (int k = 1; k < p; k++)
      for (int j = 1; j + k < p; j++)
         for (int i = 1; i + j + k < p; i++)  // (0,1,3,4)
         {
            double w = cp[i] + cp[j] + cp[k] + cp[p-i-j-k];
            Nodes.IntPoint(o++).Set4(cp[i]/w, cp[0], cp[j]/w, cp[k]/w);
         }
   for (int k = 1; k < p; k++)
      for (int j = 1; j + k < p; j++)
         for (int i = 1; i + j + k < p; i++)  // (0,3,2,4)
         {
            double w = cp[i] + cp[j] + cp[k] + cp[p-i-j-k];
            Nodes.IntPoint(o++).Set4(cp[0], cp[j]/w, cp[i]/w, cp[k]/w);
         }
   for (int k = 1; k < p; k++)
      for (int j = 1; j + k < p; j++)
         for (int i = 1; i + j + k < p; i++)  // (1,2,3,4)
         {
            double w = cp[i] + cp[j] + cp[k] + cp[p-i-j-k];
            Nodes.IntPoint(o++).Set4(cp[p-i-j-k]/w, cp[i]/w, cp[j]/w, cp[k]/w);
         }

   // interior bubbles
   for (int l = 1; l < p; l++)
      for (int k = 1; k + l < p; k++)
         for (int j = 1; j + k + l < p; j++)
            for (int i = 1; i + j + k + l < p; i++)
            {
               double w = cp[i] + cp[j] + cp[k] + cp[l] + cp[p-i-j-k-l];
               Nodes.IntPoint(o++).Set4(cp[i]/w, cp[j]/w, cp[k]/w, cp[l]/w);
            }

   DenseMatrix T(dof);
   for (int m = 0; m < dof; m++)
   {
       IntegrationPoint &ip = Nodes.IntPoint(m);
       std::vector<double> bary_vector{(1.0 - ip.x - ip.y - ip.z - ip.t), ip.x, ip.y, ip.z, ip.t};
       
       int o = 0;
       
       double La, Lb, Lc, Ld, Le;
       
       //Vertices
       for (int a=0; a<5; a++)
       {
           //compute barycentric function
           T(o++,m) = bary_vector[a];
       }
       
       //Edges
       for(int i=2; i<=p; i++)
       {
           for(int a=0; a<5; a++)
           {
               for(int b=0; b<5; b++)
               {
                   if(a<b)
                   {
                       La = bary_vector[a];
                       Lb = bary_vector[b];
                       
                       // compute polynomials
                       std::vector<double> int_Legendre_i;
                       double x = Lb;
                       double y = La + Lb;
                       poly1d.CalcIntLegendre(i, x, y, int_Legendre_i);
                       // Add Basis Funcitons
                       T(o++, m) = int_Legendre_i[int_Legendre_i.size()-1];
                       
                   }
               }
           }
       }// end of edges
       
       //Faces
       for(int i=2; i<=p;i++)
       {
           for(int j=1; j<=p;j++)
           {
               for(int a=0; a<5;a++)
               {
                   for(int b=0; b<5;b++)
                   {
                       for(int c=0; c<5;c++)
                       {
                           if((a<b)&&(b<c)&&((i+j)<=p))
                           {
                               La = bary_vector[a];
                               Lb = bary_vector[b];
                               Lc = bary_vector[c];
                               
                               // compute polynomials
                               double x = Lb;
                               double y = La + Lb;
                               std::vector<double> int_Legendre_i;
                               poly1d.CalcIntLegendre(i, x, y, int_Legendre_i);
                               
                               double alpha = 2.0*i;
                               x = Lc;
                               y = La + Lb + Lc;
                               std::vector<double> int_Jacobi_j;
                               poly1d.CalcIntJacobi(j, x, y, alpha, int_Jacobi_j);
                               
                               // Add Basis functions
                               T(o++, m) = int_Legendre_i[int_Legendre_i.size()-1] * int_Jacobi_j[int_Jacobi_j.size()-1];
                           }
                       }
                   }
               }
           }
       }
       // end of faces
       
      //Facets
      int a;
      int b;
      int c;
      int d;

      for (int f=0; f<5; f++)
      {
          // Define each facet
          if (f==0)
          {
              a = 0;
              b = 1;
              c = 3;
              d = 2;
          }
          else if(f==1)
          {
              // Define Facet
              a = 0;
              b = 1;
              c = 4;
              d = 2;
          }
          // Define each facet
          else if (f==2)
          {
              // Define Facet
              a = 0;
              b = 1;
              c = 3;
              d = 4;
          }
          // Define each facet
          else if (f==3)
          {
              // Define Facet
              a = 0;
              b = 2;
              c = 3;
              d = 4;
          }
          // Define each facet
          else if (f==4)
          {
              // Define Facet
              a = 1;
              b = 2;
              c = 3;
              d = 4;
          }
          else
          {
              mfem_error("Invaild facet");
          }

          // Define Barycentric Coordinates
          La = bary_vector[a];
          Lb = bary_vector[b];
          Lc = bary_vector[c];
          Ld = bary_vector[d];

          for(int i=2; i<=p;i++)
          {
              for(int j=1; j<=p;j++)
              {
                  for(int l=1; l<=p;l++)
                  {
                      if((i+j+l)<=p)
                      {

                          // compute polynomials
                          double x = Lb;
                          double y = La + Lb;
                          std::vector<double> int_Legendre_i;
                          poly1d.CalcIntLegendre(i, x, y, int_Legendre_i);

                          double alpha = 2.0*i;
                          x = Lc;
                          y = La + Lb + Lc;
                          std::vector<double> int_Jacobi_j;
                          poly1d.CalcIntJacobi(j, x, y, alpha, int_Jacobi_j);

                          alpha = 2.0*(i+j);
                          x = Ld;
                          y = La + Lb + Lc + Ld;
                          std::vector<double> int_Jacobi_l;
                          poly1d.CalcIntJacobi(l, x, y, alpha, int_Jacobi_l);

                          // Add Basis Functions
                          T(o++, m) = int_Legendre_i[int_Legendre_i.size()-1] * int_Jacobi_j[int_Jacobi_j.size()-1] * int_Jacobi_l[int_Jacobi_l.size()-1];

                      }
                  }
              }
          }
      }// end of Facets

       // define lamda
       La = bary_vector[0];
       Lb = bary_vector[1];
       Lc = bary_vector[2];
       Ld = bary_vector[3];
       Le = bary_vector[4];
       
      //Interiors
      for(int i=2; i<=p;i++)
      {
          for(int j=1; j<=p;j++)
          {
              for(int l=1; l<=p;l++)
              {
                  for(int q=1; q<=p;q++)
                  {
                      if ((i+j+l+q)<=p)
                      {

                          // compute polynomials
                          double x = Lb;
                          double y = La + Lb;
                          std::vector<double> int_Legendre_i;
                          poly1d.CalcIntLegendre(i, x, y, int_Legendre_i);
                          
                          double alpha = 2.0*i;
                          x = Lc;
                          y = La + Lb + Lc;
                          std::vector<double> int_Jacobi_j;
                          poly1d.CalcIntJacobi(j, x, y, alpha, int_Jacobi_j);
                          
                          alpha = 2.0*(i+j);
                          x = Ld;
                          y = La + Lb + Lc + Ld;
                          std::vector<double> int_Jacobi_l;
                          poly1d.CalcIntJacobi(l, x, y, alpha, int_Jacobi_l);
                          
                          alpha = 2.0*(i + j + l);
                          x = Le;
                          y = 1.0;
                          std::vector<double> int_Jacobi_q;
                          poly1d.CalcIntJacobi(q, x, y, alpha, int_Jacobi_q);
                          
                          //Add Basis Functions
                          T(o++, m) = (int_Legendre_i[int_Legendre_i.size()-1] * int_Jacobi_j[int_Jacobi_j.size()-1] * int_Jacobi_l[int_Jacobi_l.size()-1] * int_Jacobi_q[int_Jacobi_q.size()-1]);

                      }
                  }
              }
          }
      } //end of Interiors
       
   }

   Ti.Factor(T);
   cout << "H1_PentatopeElement(" << p << ") : "; Ti.TestInversion();
}


void H1_PentatopeElement_Barycentric::CalcShape(const IntegrationPoint &ip,
                                                Vector &shape) const
{
   const int p = order;
    
   int num_dof = dof;

#ifdef MFEM_THREAD_SAFE
   Vector shape_x(p + 1), shape_y(p + 1), shape_z(p + 1), shape_t(p+1),
          shape_l(p + 1);
   Vector u(Dof);
#endif
    
    double L1, L2, L3, L4, L5;
    //compute barycentric coordinates as function of ip
    double L[4];
    ip.Get(L,4);
    
    std::vector<double> bary_vector{(1.0 - L[0]-L[1]-L[2]-L[3]), L[0], L[1], L[2], L[3]};
    
    int o = 0;
    
    double La, Lb, Lc, Ld, Le;
    
    int local_dof = 0;
    
   //Vertices
    for (int a=0; a<5; a++)
    {
        //compute barycentric function
        u(o++) = bary_vector[a];
    }
    
   //Edges
    for(int i=2; i<=p; i++)
    {
        for(int a=0; a<5; a++)
        {
            for(int b=0; b<5; b++)
            {
                if(a<b)
                {
                    La = bary_vector[a];
                    Lb = bary_vector[b];
                    
                    // compute polynomials
                    std::vector<double> int_Legendre_i;
                    double x = Lb;
                    double y = La + Lb;
                    poly1d.CalcIntLegendre(i, x, y, int_Legendre_i);
                                        
                    // Add Basis Funcitons
                    u(o++) = int_Legendre_i[int_Legendre_i.size()-1];
                }
            }
        }
    } // end of edges

    
   //Faces
   for(int i=2; i<=p;i++)
   {
       for(int j=1; j<=p;j++)
       {
           for(int a=0; a<5;a++)
           {
               for(int b=0; b<5;b++)
               {
                   for(int c=0; c<5;c++)
                   {
                       if((a<b)&&(b<c)&&((i+j)<=p))
                       {
                           La = bary_vector[a];
                           Lb = bary_vector[b];
                           Lc = bary_vector[c];
                           
                           // compute polynomials
                           double x = Lb;
                           double y = La + Lb;
                           std::vector<double> int_Legendre_i;
                           poly1d.CalcIntLegendre(i, x, y, int_Legendre_i);
                           
                           double alpha = 2.0*i;
                           x = Lc;
                           y = La + Lb + Lc;
                           std::vector<double> int_Jacobi_j;
                           poly1d.CalcIntJacobi(j, x, y, alpha, int_Jacobi_j);
                           
                           // Add Basis functions
                           u(o++) = int_Legendre_i[int_Legendre_i.size()-1] * int_Jacobi_j[int_Jacobi_j.size()-1];
                       }
                   }
               }
           }
       }
   } // end of faces
   
    
    //Facets
    int a;
    int b;
    int c;
    int d;
      
    for (int f=0; f<5; f++)
    {
        // Define each facet
        if (f==0)
        {
            a = 0;
            b = 1;
            c = 3;
            d = 2;
        }
        else if(f==1)
        {
            // Define Facet
            a = 0;
            b = 1;
            c = 4;
            d = 2;
        }
        // Define each facet
        else if (f==2)
        {
            // Define Facet
            a = 0;
            b = 1;
            c = 3;
            d = 4;
        }
        // Define each facet
        else if (f==3)
        {
            // Define Facet
            a = 0;
            b = 2;
            c = 3;
            d = 4;
        }
        // Define each facet
        else if (f==4)
        {
            // Define Facet
            a = 1;
            b = 2;
            c = 3;
            d = 4;
        }
        else
        {
            mfem_error("Invaild facet");
        }
        
        // Define Barycentric Coordinates
        La = bary_vector[a];
        Lb = bary_vector[b];
        Lc = bary_vector[c];
        Ld = bary_vector[d];
        
        for(int i=2; i<=p;i++)
        {
            for(int j=1; j<=p;j++)
            {
                for(int l=1; l<=p;l++)
                {
                    if((i+j+l)<=p)
                    {
                        
                        // compute polynomials
                        double x = Lb;
                        double y = La + Lb;
                        std::vector<double> int_Legendre_i;
                        poly1d.CalcIntLegendre(i, x, y, int_Legendre_i);
                        
                        double alpha = 2.0*i;
                        x = Lc;
                        y = La + Lb + Lc;
                        std::vector<double> int_Jacobi_j;
                        poly1d.CalcIntJacobi(j, x, y, alpha, int_Jacobi_j);
                        
                        alpha = 2.0*(i+j);
                        x = Ld;
                        y = La + Lb + Lc + Ld;
                        std::vector<double> int_Jacobi_l;
                        poly1d.CalcIntJacobi(l, x, y, alpha, int_Jacobi_l);
                        
                        // Add Basis Functions
                        u(o++) = int_Legendre_i[int_Legendre_i.size()-1] * int_Jacobi_j[int_Jacobi_j.size()-1] * int_Jacobi_l[int_Jacobi_l.size()-1];
                        local_dof = o;


                    }
                }
            }
        }
    }// end of Facets

    // define lamda
    La = bary_vector[0];
    Lb = bary_vector[1];
    Lc = bary_vector[2];
    Ld = bary_vector[3];
    Le = bary_vector[4];
           
    //Interiors
   for(int i=2; i<=p;i++)
   {
       for(int j=1; j<=p;j++)
       {
           for(int l=1; l<=p;l++)
           {
               for(int q=1; q<=p;q++)
               {
                   if ((i+j+l+q)<=p)
                   {
                                              
                        //compute polynomials
                       double x = Lb;
                       double y = La + Lb;
                       std::vector<double> int_Legendre_i;
                       poly1d.CalcIntLegendre(i, x, y, int_Legendre_i);
                       
                       double alpha = 2.0*i;
                       x = Lc;
                       y = La + Lb + Lc;
                       std::vector<double> int_Jacobi_j;
                       poly1d.CalcIntJacobi(j, x, y, alpha, int_Jacobi_j);
                       
                       alpha = 2.0*(i+j);
                       x = Ld;
                       y = La + Lb + Lc + Ld;
                       std::vector<double> int_Jacobi_l;
                       poly1d.CalcIntJacobi(l, x, y, alpha, int_Jacobi_l);
                       
                       alpha = 2.0*(i + j + l);
                       x = Le;
                       y = 1.0;
                       std::vector<double> int_Jacobi_q;
                       poly1d.CalcIntJacobi(q, x, y, alpha, int_Jacobi_q);

                        //Add Basis Functions
                       u(o++) = (int_Legendre_i[int_Legendre_i.size()-1] * int_Jacobi_j[int_Jacobi_j.size()-1] * int_Jacobi_l[int_Jacobi_l.size()-1] * int_Jacobi_q[int_Jacobi_q.size()-1]);

                   }
               }
           }
       }
   } //end of Interiors */
    
   Ti.Mult(u, shape);

}

void H1_PentatopeElement_Barycentric::CalcDShape(const IntegrationPoint &ip,
                                     DenseMatrix &dshape) const
{
   const int p = order;

#ifdef MFEM_THREAD_SAFE
   Vector  shape_x(p + 1),  shape_y(p + 1),  shape_z(p + 1),  shape_t(p+1),
           shape_l(p + 1);
   Vector dshape_x(p + 1), dshape_y(p + 1), dshape_z(p + 1), dshape_t(p+1),
          dshape_l(p + 1);
   DenseMatrix du(Dof, Dim);
#endif
    int o = 0;

    double L1, L2, L3, L4, L5;
    //compute barycentric coordinates as function of ip
    double L[4];
    ip.Get(L,4);
    std::vector<double> bary_vector{(1.0-L[0]-L[1]-L[2]-L[3]), L[0], L[1], L[2], L[3]};
    double La, Lb, Lc, Ld, Le;
    
    // compute grad of barycentric coordinates
    std::vector<std::vector<double>> grad_bary_vectors{{-1.0,-1.0,-1.0,-1.0},{1.0,0,0,0},{0,1.0,0,0},{0,0,1.0,0},{0,0,0,1.0}};
    std::vector<double> grad_La, grad_Lb, grad_Lc, grad_Ld, grad_Le;
    
   //Vertices
    for (int a=0; a<5; a++)
    {
        //compute vertex
        du(o,0) = grad_bary_vectors[a][0];
        du(o,1) = grad_bary_vectors[a][1];
        du(o,2) = grad_bary_vectors[a][2];
        du(o,3) = grad_bary_vectors[a][3];
        o++;
    }
    
   //Edges
    for(int i=2; i<=p; i++)
    {
        for(int a=0; a<5; a++)
        {
            for(int b=0; b<5; b++)
            {
                if(a<b)
                {
                    // Define Barycentric coordinates
                    La = bary_vector[a];
                    Lb = bary_vector[b];
                    
                    // compute polynomials
                    std::vector<double> Legendre_i; // Use i-1 polynomial
                    double x = Lb;
                    double y = La + Lb;
                
                    poly1d.CalcLegendreShifted(i, x, y, Legendre_i);
                    
                    std::vector<double> R_i;
                    x = Lb;
                    y = La + Lb;
                    poly1d.CalcRLegendre(i, x, y, R_i);
                    
                    
                    // Add Basis Funcitons
                    du(o,0) = Legendre_i[Legendre_i.size()-2] * grad_bary_vectors[b][0] + R_i[R_i.size()-2] * (grad_bary_vectors[a][0] + grad_bary_vectors[b][0]);
                    
                    du(o,1) = Legendre_i[Legendre_i.size()-2] * grad_bary_vectors[b][1] + R_i[R_i.size()-2] * (grad_bary_vectors[a][1] + grad_bary_vectors[b][1]);
                    
                    du(o,2) = Legendre_i[Legendre_i.size()-2] * grad_bary_vectors[b][2] + R_i[R_i.size()-2] * (grad_bary_vectors[a][2] + grad_bary_vectors[b][2]);
                    
                    du(o,3) = Legendre_i[Legendre_i.size()-2] * grad_bary_vectors[b][3] + R_i[R_i.size()-2] * (grad_bary_vectors[a][3] + grad_bary_vectors[b][3]);
                
                    o++;
                }
            }
        }
    } // end of edges
    
   //Faces
   for(int i=2; i<=p;i++)
   {
       for(int j=1; j<=p;j++)
       {
           for(int a=0; a<5;a++)
           {
               for(int b=0; b<5;b++)
               {
                   for(int c=0; c<5;c++)
                   {
                       if((a<b)&&(b<c)&&((i+j)<=p))
                       {
                           // Define Barycentric coordinates
                           La = bary_vector[a];
                           Lb = bary_vector[b];
                           Lc = bary_vector[c];
                           
                           // compute polynomials
                           double x = Lb;
                           double y = La + Lb;
                           std::vector<double> int_Legendre_i;
                           poly1d.CalcIntLegendre(i, x, y, int_Legendre_i);
                           
                           double alpha = 2.0*i;
                           x = Lc;
                           y = La + Lb + Lc;
                           std::vector<double> int_Jacobi_j;
                           poly1d.CalcIntJacobi(j, x, y, alpha, int_Jacobi_j);
                           
                           // compute polynomials for grad(L_i)
                           std::vector<double> Legendre_i; // Use i-1 polynomial
                           x = Lb;
                           y = La + Lb;
                           poly1d.CalcLegendreShifted(i, x, y, Legendre_i);
                           
                           std::vector<double> R_i; // Use i-1 polynomial
                           x = Lb;
                           y = La + Lb;
                           poly1d.CalcRLegendre(i, x, y, R_i);
                           
                           // compute polynimials for grad(L_j^2i)
                           std::vector<double> Jacobi_j; // Use i-1 polynomial
                           x = Lc;
                           y = La + Lb + Lc;
                           alpha = 2.0*i;
                           poly1d.CalcJacobi(j, x, y, alpha, Jacobi_j);
                           
                           std::vector<double> R_j; // Use i-1 polynomial
                           x = Lc;
                           y = La + Lb + Lc;
                           poly1d.CalcRJacobi(j, x, y, alpha, R_j);

                           // Add Basis Funcitons
                           du(o,0) = (Legendre_i[Legendre_i.size()-2] * grad_bary_vectors[b][0] +
                                      R_i[R_i.size()-2] * (grad_bary_vectors[a][0] +grad_bary_vectors[b][0]))
                                      * int_Jacobi_j[int_Jacobi_j.size()-1]
                           + int_Legendre_i[int_Legendre_i.size()-1] * (Jacobi_j[Jacobi_j.size()-2] * grad_bary_vectors[c][0] + R_j[R_j.size()-2] * (grad_bary_vectors[a][0] + grad_bary_vectors[b][0] + grad_bary_vectors[c][0]));
                           
                           du(o,1) = (Legendre_i[Legendre_i.size()-2] * grad_bary_vectors[b][1] +
                                      R_i[R_i.size()-2] * (grad_bary_vectors[a][1] +grad_bary_vectors[b][1]))
                                      * int_Jacobi_j[int_Jacobi_j.size()-1]
                           + int_Legendre_i[int_Legendre_i.size()-1] * (Jacobi_j[Jacobi_j.size()-2] * grad_bary_vectors[c][1] + R_j[R_j.size()-2] * (grad_bary_vectors[a][1] + grad_bary_vectors[b][1] + grad_bary_vectors[c][1]));
                           
                           du(o,2) = (Legendre_i[Legendre_i.size()-2] * grad_bary_vectors[b][2] +
                                      R_i[R_i.size()-2] * (grad_bary_vectors[a][2] +grad_bary_vectors[b][2]))
                                      * int_Jacobi_j[int_Jacobi_j.size()-1]
                           + int_Legendre_i[int_Legendre_i.size()-1] * (Jacobi_j[Jacobi_j.size()-2] * grad_bary_vectors[c][2] + R_j[R_j.size()-2] * (grad_bary_vectors[a][2] + grad_bary_vectors[b][2] + grad_bary_vectors[c][2]));
                           
                           du(o,3) = (Legendre_i[Legendre_i.size()-2] * grad_bary_vectors[b][3] +
                                      R_i[R_i.size()-2] * (grad_bary_vectors[a][3] +grad_bary_vectors[b][3]))
                                      * int_Jacobi_j[int_Jacobi_j.size()-1]
                           + int_Legendre_i[int_Legendre_i.size()-1] * (Jacobi_j[Jacobi_j.size()-2] * grad_bary_vectors[c][3] + R_j[R_j.size()-2] * (grad_bary_vectors[a][3] + grad_bary_vectors[b][3] + grad_bary_vectors[c][3]));
                           
                           o++;
                       }
                   }
               }
           }
       }
   } // end of faces
    
    
  //Facets
  int a;
  int b;
  int c;
  int d;
    
  for (int f=0; f<5; f++)
  {
      // Define each facet
      if (f==0)
      {
          a = 0;
          b = 1;
          c = 3;
          d = 2;
      }
      else if(f==1)
      {
          // Define Facet
          a = 0;
          b = 1;
          c = 4;
          d = 2;
      }
      // Define each facet
      else if (f==2)
      {
          // Define Facet
          a = 0;
          b = 1;
          c = 3;
          d = 4;
      }
      // Define each facet
      else if (f==3)
      {
          // Define Facet
          a = 0;
          b = 2;
          c = 3;
          d = 4;
      }
      // Define each facet
      else if (f==4)
      {
          // Define Facet
          a = 1;
          b = 2;
          c = 3;
          d = 4;
      }
      else
      {
          mfem_error("Invaild facet");
      }
      
      // Define Barycentric Coordinates
      La = bary_vector[a];
      Lb = bary_vector[b];
      Lc = bary_vector[c];
      Ld = bary_vector[d];
      
      for(int i=2; i<=p;i++)
      {
          for(int j=1; j<=p;j++)
          {
              for(int l=1; l<=p;l++)
              {
                  if((i+j+l)<=p)
                  {
                      
                      // compute polynomials
                      double x = Lb;
                      double y = La + Lb;
                      std::vector<double> int_Legendre_i;
                      poly1d.CalcIntLegendre(i, x, y, int_Legendre_i);
                      
                      double alpha = 2.0*i;
                      x = Lc;
                      y = La + Lb + Lc;
                      std::vector<double> int_Jacobi_j;
                      poly1d.CalcIntJacobi(j, x, y, alpha, int_Jacobi_j);
                      
                      alpha = 2.0*(i+j);
                      x = Ld;
                      y = La + Lb + Lc + Ld;
                      std::vector<double> int_Jacobi_l;
                      poly1d.CalcIntJacobi(l, x, y, alpha, int_Jacobi_l);
                      
                      // compute polynomials for grad(L_i)
                      std::vector<double> Legendre_i; // Use i-1 polynomial
                      x = Lb;
                      y = La + Lb;
                      poly1d.CalcLegendreShifted(i, x, y, Legendre_i);
                      
                      std::vector<double> R_i; // Use i-1 polynomial
                      x = Lb;
                      y = La + Lb;
                      poly1d.CalcRLegendre(i, x, y, R_i);
                      
                      // compute polynimials for grad(L_j^2i)
                      std::vector<double> Jacobi_j; // Use i-1 polynomial
                      x = Lc;
                      y = La + Lb + Lc;
                      alpha = 2.0*i;
                      poly1d.CalcJacobi(j, x, y, alpha, Jacobi_j);
                      
                      std::vector<double> R_j; // Use i-1 polynomial
                      x = Lc;
                      y = La + Lb + Lc;
                      poly1d.CalcRJacobi(j, x, y, alpha, R_j);
                      
                      
                      // compute polynimials for grad(L_l^2(i+j))
                      std::vector<double> Jacobi_l; // Use i-1 polynomial
                      x = Ld;
                      y = La + Lb + Lc + Ld;
                      alpha = 2.0*(i+j);
                      poly1d.CalcJacobi(l, x, y, alpha, Jacobi_l);
                      
                      std::vector<double> R_l; // Use i-1 polynomial
                      x = Ld;
                      y = La + Lb + Lc + Ld;
                      poly1d.CalcRJacobi(l, x, y, alpha, R_l);
                      
                      // Add Basis Funcitons
                      du(o,0) = (Legendre_i[Legendre_i.size()-2] * grad_bary_vectors[b][0] +
                                 R_i[R_i.size()-2] * (grad_bary_vectors[a][0] +grad_bary_vectors[b][0]))
                                 * int_Jacobi_j[int_Jacobi_j.size()-1] * int_Jacobi_l[int_Jacobi_l.size()-1]
                      + (int_Legendre_i[int_Legendre_i.size()-1] * (Jacobi_j[Jacobi_j.size()-2] * grad_bary_vectors[c][0] + R_j[R_j.size()-2] * (grad_bary_vectors[a][0] + grad_bary_vectors[b][0] + grad_bary_vectors[c][0])) * int_Jacobi_l[int_Jacobi_l.size()-1])
                      + (int_Legendre_i[int_Legendre_i.size()-1] * int_Jacobi_j[int_Jacobi_j.size()-1] * (Jacobi_l[Jacobi_l.size()-2] * grad_bary_vectors[d][0] + R_l[R_l.size()-2] * (grad_bary_vectors[a][0] + grad_bary_vectors[b][0] + grad_bary_vectors[c][0] + grad_bary_vectors[d][0])));
                      
                      du(o,1) = (Legendre_i[Legendre_i.size()-2] * grad_bary_vectors[b][1] +
                                 R_i[R_i.size()-2] * (grad_bary_vectors[a][1] +grad_bary_vectors[b][1]))
                                 * int_Jacobi_j[int_Jacobi_j.size()-1] * int_Jacobi_l[int_Jacobi_l.size()-1]
                      + (int_Legendre_i[int_Legendre_i.size()-1] * (Jacobi_j[Jacobi_j.size()-2] * grad_bary_vectors[c][1] + R_j[R_j.size()-2] * (grad_bary_vectors[a][1] + grad_bary_vectors[b][1] + grad_bary_vectors[c][1])) * int_Jacobi_l[int_Jacobi_l.size()-1])
                      + (int_Legendre_i[int_Legendre_i.size()-1] * int_Jacobi_j[int_Jacobi_j.size()-1] * (Jacobi_l[Jacobi_l.size()-2] * grad_bary_vectors[d][1] + R_l[R_l.size()-2] * (grad_bary_vectors[a][1] + grad_bary_vectors[b][1] + grad_bary_vectors[c][1] + grad_bary_vectors[d][1])));
                      
                      du(o,2) = (Legendre_i[Legendre_i.size()-2] * grad_bary_vectors[b][2] +
                                 R_i[R_i.size()-2] * (grad_bary_vectors[a][2] +grad_bary_vectors[b][2]))
                                 * int_Jacobi_j[int_Jacobi_j.size()-1] * int_Jacobi_l[int_Jacobi_l.size()-1]
                      + (int_Legendre_i[int_Legendre_i.size()-1] * (Jacobi_j[Jacobi_j.size()-2] * grad_bary_vectors[c][2] + R_j[R_j.size()-2] * (grad_bary_vectors[a][2] + grad_bary_vectors[b][2] + grad_bary_vectors[c][2])) * int_Jacobi_l[int_Jacobi_l.size()-1])
                      + (int_Legendre_i[int_Legendre_i.size()-1] * int_Jacobi_j[int_Jacobi_j.size()-1] * (Jacobi_l[Jacobi_l.size()-2] * grad_bary_vectors[d][2] + R_l[R_l.size()-2] * (grad_bary_vectors[a][2] + grad_bary_vectors[b][2] + grad_bary_vectors[c][2] + grad_bary_vectors[d][2])));
                      
                      du(o,3) = (Legendre_i[Legendre_i.size()-2] * grad_bary_vectors[b][3] +
                                 R_i[R_i.size()-2] * (grad_bary_vectors[a][3] +grad_bary_vectors[b][3]))
                                 * int_Jacobi_j[int_Jacobi_j.size()-1] * int_Jacobi_l[int_Jacobi_l.size()-1]
                      + (int_Legendre_i[int_Legendre_i.size()-1] * (Jacobi_j[Jacobi_j.size()-2] * grad_bary_vectors[c][3] + R_j[R_j.size()-2] * (grad_bary_vectors[a][3] + grad_bary_vectors[b][3] + grad_bary_vectors[c][3])) * int_Jacobi_l[int_Jacobi_l.size()-1])
                      + (int_Legendre_i[int_Legendre_i.size()-1] * int_Jacobi_j[int_Jacobi_j.size()-1] * (Jacobi_l[Jacobi_l.size()-2] * grad_bary_vectors[d][3] + R_l[R_l.size()-2] * (grad_bary_vectors[a][3] + grad_bary_vectors[b][3] + grad_bary_vectors[c][3] + grad_bary_vectors[d][3])));
                      o++;
                          
                  }
              }
          }
      }
  }// end of Facets

           
    //Interiors
   for(int i=2; i<=p;i++)
   {
       for(int j=1; j<=p;j++)
       {
           for(int l=1; l<=p;l++)
           {
               for(int m=1; m<=p;m++)
               {
                   if ((i+j+l+m)<=p)
                   {
                       // define lamda
                       La = bary_vector[0];
                       Lb = bary_vector[1];
                       Lc = bary_vector[2];
                       Ld = bary_vector[3];
                       Le = bary_vector[4];
                       
                       // compute polynomials
                       double x = Lb;
                       double y = La + Lb;
                       std::vector<double> int_Legendre_i;
                       poly1d.CalcIntLegendre(i, x, y, int_Legendre_i);
                       
                       double alpha = 2.0*i;
                       x = Lc;
                       y = La + Lb + Lc;
                       std::vector<double> int_Jacobi_j;
                       poly1d.CalcIntJacobi(j, x, y, alpha, int_Jacobi_j);
                       
                       alpha = 2.0*(i+j);
                       x = Ld;
                       y = La + Lb + Lc + Ld;
                       std::vector<double> int_Jacobi_l;
                       poly1d.CalcIntJacobi(l, x, y, alpha, int_Jacobi_l);
                       
                       alpha = 2.0*(i + j + l);
                       x = Le;
                       y = 1.0;
                       std::vector<double> int_Jacobi_m;
                       poly1d.CalcIntJacobi(m, x, y, alpha, int_Jacobi_m);
                       
                       // compute polynomials for grad(L_i)
                       std::vector<double> Legendre_i; // Use i-1 polynomial
                       x = Lb;
                       y = La + Lb;
                       poly1d.CalcLegendreShifted(i, x, y, Legendre_i);
                       
                       std::vector<double> R_i; // Use i-1 polynomial
                       x = Lb;
                       y = La + Lb;
                       poly1d.CalcRLegendre(i, x, y, R_i);
                       
                       // compute polynimials for grad(L_j^2i)
                       std::vector<double> Jacobi_j; // Use i-1 polynomial
                       x = Lc;
                       y = La + Lb + Lc;
                       alpha = 2.0*i;
                       poly1d.CalcJacobi(j, x, y, alpha, Jacobi_j);
                       
                       std::vector<double> R_j; // Use i-1 polynomial
                       x = Lc;
                       y = La + Lb + Lc;
                       poly1d.CalcRJacobi(j, x, y, alpha, R_j);
                       
                       // compute polynimials for grad(L_l^2(i+j))
                       std::vector<double> Jacobi_l; // Use i-1 polynomial
                       x = Ld;
                       y = La + Lb + Lc + Ld;
                       alpha = 2.0*(i+j);
                       poly1d.CalcJacobi(l, x, y, alpha, Jacobi_l);
                       
                       std::vector<double> R_l; // Use i-1 polynomial
                       x = Ld;
                       y = La + Lb + Lc + Ld;
                       poly1d.CalcRJacobi(l, x, y, alpha, R_l);
                       
                       // compute polynimials for grad(L_m^2(i+j+l))
                       std::vector<double> Jacobi_m; // Use i-1 polynomial
                       x = Le;
                       y = 1.0;
                       alpha = 2.0*(i+j+l);
                       poly1d.CalcJacobi(m, x, y, alpha, Jacobi_m);
                    
 
                       // Add Basis Funcitons
                       du(o, 0) = ((Legendre_i[Legendre_i.size()-2] * grad_bary_vectors[1][0] + R_i[R_i.size()-2] * (grad_bary_vectors[0][0] + grad_bary_vectors[1][0]))
                                   * int_Jacobi_j[int_Jacobi_j.size()-1] * int_Jacobi_l[int_Jacobi_l.size()-1] * int_Jacobi_m[int_Jacobi_m.size()-1]
                                     
                        +  int_Legendre_i[int_Legendre_i.size()-1] * (Jacobi_j[Jacobi_j.size()-2] * grad_bary_vectors[2][0] + R_j[R_j.size()-2] * (grad_bary_vectors[0][0] + grad_bary_vectors[1][0] + grad_bary_vectors[2][0])) * int_Jacobi_l[int_Jacobi_l.size()-1] * int_Jacobi_m[int_Jacobi_m.size()-1]
                                     
                        + int_Legendre_i[int_Legendre_i.size()-1] * int_Jacobi_j[int_Jacobi_j.size()-1] * (Jacobi_l[Jacobi_l.size()-2] * grad_bary_vectors[3][0] + R_l[R_l.size()-2] * (grad_bary_vectors[0][0] + grad_bary_vectors[1][0] + grad_bary_vectors[2][0] + grad_bary_vectors[3][0])) *  int_Jacobi_m[int_Jacobi_m.size()-1]
                                     
                        + int_Legendre_i[int_Legendre_i.size()-1] * int_Jacobi_j[int_Jacobi_j.size()-1] * int_Jacobi_l[int_Jacobi_l.size()-1] * (Jacobi_m[Jacobi_m.size()-2] * grad_bary_vectors[4][0]));
                       
                       
                       du(o, 1) = ((Legendre_i[Legendre_i.size()-2] * grad_bary_vectors[1][1] + R_i[R_i.size()-2] * (grad_bary_vectors[0][1] +grad_bary_vectors[1][1]))
                                   * int_Jacobi_j[int_Jacobi_j.size()-1] * int_Jacobi_l[int_Jacobi_l.size()-1] * int_Jacobi_m[int_Jacobi_m.size()-1]
                                     
                        +  int_Legendre_i[int_Legendre_i.size()-1] * (Jacobi_j[Jacobi_j.size()-2] * grad_bary_vectors[2][1] + R_j[R_j.size()-2] * (grad_bary_vectors[0][1] + grad_bary_vectors[1][1] + grad_bary_vectors[2][1])) * int_Jacobi_l[int_Jacobi_l.size()-1] * int_Jacobi_m[int_Jacobi_m.size()-1]
                                     
                        + int_Legendre_i[int_Legendre_i.size()-1] * int_Jacobi_j[int_Jacobi_j.size()-1] * (Jacobi_l[Jacobi_l.size()-2] * grad_bary_vectors[3][1] + R_l[R_l.size()-2] * (grad_bary_vectors[0][1] + grad_bary_vectors[1][1] + grad_bary_vectors[2][1] + grad_bary_vectors[3][1])) *  int_Jacobi_m[int_Jacobi_m.size()-1]
                                     
                        + int_Legendre_i[int_Legendre_i.size()-1] * int_Jacobi_j[int_Jacobi_j.size()-1] * int_Jacobi_l[int_Jacobi_l.size()-1] * (Jacobi_m[Jacobi_m.size()-2] * grad_bary_vectors[4][1]));

                                              
                       du(o, 2) = ((Legendre_i[Legendre_i.size()-2] * grad_bary_vectors[1][2] + R_i[R_i.size()-2] * (grad_bary_vectors[0][2] +grad_bary_vectors[1][2]))
                                   * int_Jacobi_j[int_Jacobi_j.size()-1] * int_Jacobi_l[int_Jacobi_l.size()-1] * int_Jacobi_m[int_Jacobi_m.size()-1]
                                     
                        +  int_Legendre_i[int_Legendre_i.size()-1] * (Jacobi_j[Jacobi_j.size()-2] * grad_bary_vectors[2][2] + R_j[R_j.size()-2] * (grad_bary_vectors[0][2] + grad_bary_vectors[1][2] + grad_bary_vectors[2][2])) * int_Jacobi_l[int_Jacobi_l.size()-1] * int_Jacobi_m[int_Jacobi_m.size()-1]
                                     
                        + int_Legendre_i[int_Legendre_i.size()-1] * int_Jacobi_j[int_Jacobi_j.size()-1] * (Jacobi_l[Jacobi_l.size()-2] * grad_bary_vectors[3][2] + R_l[R_l.size()-2] * (grad_bary_vectors[0][2] + grad_bary_vectors[1][2] + grad_bary_vectors[2][2] + grad_bary_vectors[3][2])) *  int_Jacobi_m[int_Jacobi_m.size()-1]
                                     
                        + int_Legendre_i[int_Legendre_i.size()-1] * int_Jacobi_j[int_Jacobi_j.size()-1] * int_Jacobi_l[int_Jacobi_l.size()-1] * (Jacobi_m[Jacobi_m.size()-2] * grad_bary_vectors[4][2]));

                       
                       
                       du(o, 3) = ((Legendre_i[Legendre_i.size()-2] * grad_bary_vectors[1][3] + R_i[R_i.size()-2] * (grad_bary_vectors[0][3] +grad_bary_vectors[1][3]))
                                   * int_Jacobi_j[int_Jacobi_j.size()-1] * int_Jacobi_l[int_Jacobi_l.size()-1] * int_Jacobi_m[int_Jacobi_m.size()-1]
                                     
                        +  int_Legendre_i[int_Legendre_i.size()-1] * (Jacobi_j[Jacobi_j.size()-2] * grad_bary_vectors[2][3] + R_j[R_j.size()-2] * (grad_bary_vectors[0][3] + grad_bary_vectors[1][3] + grad_bary_vectors[2][3])) * int_Jacobi_l[int_Jacobi_l.size()-1] * int_Jacobi_m[int_Jacobi_m.size()-1]
                                     
                        + int_Legendre_i[int_Legendre_i.size()-1] * int_Jacobi_j[int_Jacobi_j.size()-1] * (Jacobi_l[Jacobi_l.size()-2] * grad_bary_vectors[3][3] + R_l[R_l.size()-2] * (grad_bary_vectors[0][3] + grad_bary_vectors[1][3] + grad_bary_vectors[2][3] + grad_bary_vectors[3][3])) *  int_Jacobi_m[int_Jacobi_m.size()-1]
                                     
                        + int_Legendre_i[int_Legendre_i.size()-1] * int_Jacobi_j[int_Jacobi_j.size()-1] * int_Jacobi_l[int_Jacobi_l.size()-1] * (Jacobi_m[Jacobi_m.size()-2] * grad_bary_vectors[4][3]));

                       o++;
                       
                   }
               }
           }
       }
   } //end of Interiors
    
   Ti.Mult(du, dshape);
}

void H1_PentatopeElement_Barycentric::CalcHessian(const IntegrationPoint &ip,
                                      DenseMatrix &ddshape) const
{
    //Do Nothing
    mfem_error("H1_PentatopeElement_Barycentric::CalcHessian not implemented");
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
                        p, FunctionSpace::Uk)
{
   zmax = 0.0;

   const real_t *cp = poly1d.ClosedPoints(p, VerifyNodal(VerifyClosed(btype)));

#ifndef MFEM_THREAD_SAFE
   tmp_i.SetSize(p + 1);
   tmp1_ij.SetSize(p + 1, p + 1);
   tmp2_ij.SetSize(p + 1, dim);
   tmp_ijk.SetSize(p + 1, p + 1, dim);
   tmp_u.SetSize(dof);
   tmp_du.SetSize(dof, dim);
#else
   Vector tmp_i(p + 1);
   DenseMatrix tmp1_ij(p + 1, p + 1);
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
         Nodes.IntPoint(o++).Set3(cp[i], cp[p-j], cp[0]);
      }
   }

   // triangular faces
   for (int j = 1; j < p; j++)
      for (int i = 1; i + j < p; i++)  // (0,1,4)
      {
         real_t w = cp[i] + cp[j] + cp[p-i-j];
         Nodes.IntPoint(o++).Set3(cp[i]/w, cp[0], cp[j]/w);
      }
   for (int j = 1; j < p; j++)
      for (int i = 1; i + j < p; i++)  // (1,2,4)
      {
         real_t w = cp[i] + cp[j] + cp[p-i-j];
         Nodes.IntPoint(o++).Set3((cp[i] + cp[p-i-j])/w, cp[i]/w, cp[j]/w);
      }
   for (int j = 1; j < p; j++)
      for (int i = 1; i + j < p; i++)  // (2,3,4)
      {
         real_t w = cp[i] + cp[j] + cp[p-i-j];
         Nodes.IntPoint(o++).Set3(cp[p-i-j]/w, (cp[i] + cp[p-i-j])/w, cp[j]/w);
      }
   for (int j = 1; j < p; j++)
      for (int i = 1; i + j < p; i++)  // (3,0,4)
      {
         real_t w = cp[i] + cp[j] + cp[p-i-j];
         Nodes.IntPoint(o++).Set3(cp[0], cp[p-i-j]/w, cp[j]/w);
      }

   // Points based on Fuentes' interior bubbles
   for (int k = 1; k < p; k++)
   {
      for (int j = 1; j < p; j++)
      {
         for (int i = 1; i < p; i++)
         {
            Nodes.IntPoint(o++).Set3(cp[i] * (1.0 - cp[k]),
                                     cp[j] * (1.0 - cp[k]),
                                     cp[k]);
         }
      }
   }

   MFEM_ASSERT(o == dof,
               "Number of nodes does not match the "
               "number of degrees of freedom");
   DenseMatrix T(dof);

   for (int m = 0; m < dof; m++)
   {
      const IntegrationPoint &ip = Nodes.IntPoint(m);
      Vector col(T.GetColumn(m), dof);
      calcBasis(order, ip, tmp_i, tmp1_ij, col);
   }

   Ti.Factor(T);
}

void H1_FuentesPyramidElement::CalcShape(const IntegrationPoint &ip,
                                         Vector &shape) const
{
   const int p = order;

#ifdef MFEM_THREAD_SAFE
   Vector tmp_i(p + 1);
   Vector tmp_u(dof);
   DenseMatrix tmp1_ij(p + 1, p + 1);
#endif

   calcBasis(p, ip, tmp_i, tmp1_ij, tmp_u);

   Ti.Mult(tmp_u, shape);
}

void H1_FuentesPyramidElement::CalcDShape(const IntegrationPoint &ip,
                                          DenseMatrix &dshape) const
{
   const int p = order;

#ifdef MFEM_THREAD_SAFE
   Vector tmp_i(p + 1);
   DenseMatrix tmp1_ij(p + 1, p + 1);
   DenseMatrix tmp2_ij(p + 1, dim);
   DenseTensor tmp_ijk(p + 1, p + 1, dim);
   DenseMatrix tmp_du(dof, dim);
#endif

   calcGradBasis(p, ip, tmp_i, tmp2_ij, tmp1_ij, tmp_ijk, tmp_du);
   Ti.Mult(tmp_du, dshape);
}

void H1_FuentesPyramidElement::CalcRawShape(const IntegrationPoint &ip,
                                            Vector &shape) const
{
   const int p = order;

#ifdef MFEM_THREAD_SAFE
   Vector tmp_i(p + 1);
   DenseMatrix tmp1_ij(p + 1, p + 1);
#endif

   calcBasis(p, ip, tmp_i, tmp1_ij, shape);
}

void H1_FuentesPyramidElement::CalcRawDShape(const IntegrationPoint &ip,
                                             DenseMatrix &dshape) const
{
   const int p = order;

#ifdef MFEM_THREAD_SAFE
   Vector tmp_i(p + 1);
   DenseMatrix tmp1_ij(p + 1, p + 1);
   DenseMatrix tmp2_ij(p + 1, dim);
   DenseTensor tmp_ijk(p + 1, p + 1, dim);
#endif

   calcGradBasis(p, ip, tmp_i, tmp2_ij, tmp1_ij, tmp_ijk, dshape);
}

void H1_FuentesPyramidElement::calcBasis(const int p,
                                         const IntegrationPoint &ip,
                                         Vector &phi_i, DenseMatrix &phi_ij,
                                         Vector &u) const
{
   real_t x = ip.x;
   real_t y = ip.y;
   real_t z = ip.z;
   Vector xy({x,y});

   zmax = std::max(z, zmax);

   real_t mu;

   int o = 0;

   // Vertices
   u[0] = lam1(x, y, z);
   u[1] = lam2(x, y, z);
   u[2] = lam3(x, y, z);
   u[3] = lam4(x, y, z);
   u[4] = lam5(x, y, z);

   o += 5;

   // Mixed edges (base edges)
   if (CheckZ(z) && p >= 2)
   {
      // (a,b) = (1,2), c = 0
      phi_E(p, nu01(z, xy, 1), phi_i);
      mu = mu0(z, xy, 2);
      for (int i = 2; i <= p; i++, o++)
      {
         u[o] = mu * phi_i[i];
      }
      // (a,b) = (1,2), c = 1
      mu = mu1(z, xy, 2);
      for (int i = 2; i <= p; i++, o++)
      {
         u[o] = mu * phi_i[i];
      }
      // (a,b) = (2,1), c = 0
      phi_E(p, nu01(z, xy, 2), phi_i);
      mu = mu0(z, xy, 1);
      for (int i = 2; i <= p; i++, o++)
      {
         u[o] = mu * phi_i[i];
      }
      // (a,b) = (2,1), c = 1
      mu = mu1(z, xy, 1);
      for (int i = 2; i <= p; i++, o++)
      {
         u[o] = mu * phi_i[i];
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
   if (p >= 2)
   {
      phi_E(p, lam15(x, y, z), phi_i);
      for (int i = 2; i<= p; i++, o++)
      {
         u[o] = phi_i[i];
      }
      phi_E(p, lam25(x, y, z), phi_i);
      for (int i = 2; i<= p; i++, o++)
      {
         u[o] = phi_i[i];
      }
      phi_E(p, lam35(x, y, z), phi_i);
      for (int i = 2; i<= p; i++, o++)
      {
         u[o] = phi_i[i];
      }
      phi_E(p, lam45(x, y, z), phi_i);
      for (int i = 2; i<= p; i++, o++)
      {
         u[o] = phi_i[i];
      }
   }

   // Quadrilateral face
   if (CheckZ(z) && p >= 2)
   {
      phi_Q(p, mu01(z, xy, 1), mu01(z, xy, 2), phi_ij);
      mu = mu0(z);
      for (int j = 2; j <= p; j++)
      {
         for (int i = 2; i <= p; i++, o++)
         {
            u[o] = mu * phi_ij(i,j);
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
   if (CheckZ(z) && p >= 3)
   {
      // (a,b) = (1,2), c = 0
      phi_T(p, nu012(z, xy, 1), phi_ij);
      mu = mu0(z, xy, 2);
      for (int i = 2; i < p; i++)
         for (int j = 1; i + j <= p; j++, o++)
         {
            u[o] = mu * phi_ij(i,j);
         }
      // (a,b) = (1,2), c = 1
      mu = mu1(z, xy, 2);
      for (int i = 2; i < p; i++)
         for (int j = 1; i + j <= p; j++, o++)
         {
            u[o] = mu * phi_ij(i,j);
         }
      // (a,b) = (2,1), c = 0
      phi_T(p, nu012(z, xy, 2), phi_ij);
      mu = mu0(z, xy, 1);
      for (int i = 2; i < p; i++)
         for (int j = 1; i + j <= p; j++, o++)
         {
            u[o] = mu * phi_ij(i,j);
         }
      // (a,b) = (2,1), c = 1
      mu = mu1(z, xy, 1);
      for (int i = 2; i < p; i++)
         for (int j = 1; i + j <= p; j++, o++)
         {
            u[o] = mu * phi_ij(i,j);
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
   if (CheckZ(z) && p >= 2)
   {
      phi_Q(p, mu01(z, xy, 1), mu01(z, xy, 2), phi_ij);
      phi_E(p, mu01(z), phi_i);
      for (int k = 2; k <= p; k++)
      {
         for (int j = 2; j <= p; j++)
         {
            for (int i = 2; i <= p; i++, o++)
            {
               u[o] = phi_ij(i,j) * phi_i(k);
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
}

void H1_FuentesPyramidElement::calcGradBasis(const int p,
                                             const IntegrationPoint &ip,
                                             Vector &phi_i,
                                             DenseMatrix &dphi_i,
                                             DenseMatrix &phi_ij,
                                             DenseTensor &dphi_ij,
                                             DenseMatrix &du) const
{
   real_t x = ip.x;
   real_t y = ip.y;
   real_t z = ip.z;
   Vector xy({x,y});

   zmax = std::max(z, zmax);

   real_t mu;
   Vector dmu(3);
   Vector dlam(3);

   int o = 0;

   // Vertices
   dlam = grad_lam1(x, y, z);
   for (int d=0; d<3; d++) { du(0, d) = dlam(d); }
   dlam = grad_lam2(x, y, z);
   for (int d=0; d<3; d++) { du(1, d) = dlam(d); }
   dlam = grad_lam3(x, y, z);
   for (int d=0; d<3; d++) { du(2, d) = dlam(d); }
   dlam = grad_lam4(x, y, z);
   for (int d=0; d<3; d++) { du(3, d) = dlam(d); }
   dlam = grad_lam5(x, y, z);
   for (int d=0; d<3; d++) { du(4, d) = dlam(d); }

   o += 5;

   // Mixed edges (base edges)
   if (CheckZ(z) && p >= 2)
   {
      // (a,b) = (1,2), c = 0
      phi_E(p, nu01(z, xy, 1), grad_nu01(z, xy, 1), phi_i, dphi_i);
      mu = mu0(z, xy, 2);
      dmu = grad_mu0(z, xy, 2);;
      for (int i = 2; i <= p; i++, o++)
         for (int d=0; d<3; d++)
         {
            du(o, d) = dmu(d) * phi_i[i] + mu * dphi_i(i, d);
         }

      // (a,b) = (1,2), c = 1
      mu = mu1(z, xy, 2);
      dmu = grad_mu1(z, xy, 2);;
      for (int i = 2; i <= p; i++, o++)
         for (int d=0; d<3; d++)
         {
            du(o, d) = dmu(d) * phi_i[i] + mu * dphi_i(i, d);
         }

      // (a,b) = (2,1), c = 0
      phi_E(p, nu01(z, xy, 2), grad_nu01(z, xy, 2), phi_i, dphi_i);
      mu = mu0(z, xy, 1);
      dmu = grad_mu0(z, xy, 1);;
      for (int i = 2; i <= p; i++, o++)
         for (int d=0; d<3; d++)
         {
            du(o, d) = dmu(d) * phi_i[i] + mu * dphi_i(i, d);
         }

      // (a,b) = (2,1), c = 1
      mu = mu1(z, xy, 1);
      dmu = grad_mu1(z, xy, 1);;
      for (int i = 2; i <= p; i++, o++)
         for (int d=0; d<3; d++)
         {
            du(o, d) = dmu(d) * phi_i[i] + mu * dphi_i(i, d);
         }
   }
   else
   {
      for (int i = 0; i < 4 * (p - 1); i++, o++)
         for (int d=0; d<3; d++)
         {
            du(o, d) = 0.0;
         }
   }

   // Triangle edges (upright edges)
   if (p >= 2)
   {
      phi_E(p, lam15(x, y, z), grad_lam15(x,y,z), phi_i, dphi_i);
      for (int i = 2; i<= p; i++, o++)
         for (int d=0; d<3; d++)
         {
            du(o, d) = dphi_i(i, d);
         }

      phi_E(p, lam25(x, y, z), grad_lam25(x, y, z), phi_i, dphi_i);
      for (int i = 2; i<= p; i++, o++)
         for (int d=0; d<3; d++)
         {
            du(o, d) = dphi_i(i, d);
         }

      phi_E(p, lam35(x, y, z), grad_lam35(x, y, z), phi_i, dphi_i);
      for (int i = 2; i<= p; i++, o++)
         for (int d=0; d<3; d++)
         {
            du(o, d) = dphi_i(i, d);
         }

      phi_E(p, lam45(x, y, z), grad_lam45(x, y, z), phi_i, dphi_i);
      for (int i = 2; i<= p; i++, o++)
         for (int d=0; d<3; d++)
         {
            du(o, d) = dphi_i(i, d);
         }
   }

   // Quadrilateral face
   if (CheckZ(z) && p >= 2)
   {
      phi_Q(p, mu01(z, xy, 1), grad_mu01(z, xy, 1),
            mu01(z, xy, 2), grad_mu01(z, xy, 2), phi_ij, dphi_ij);
      mu = mu0(z);
      dmu = grad_mu0(z);
      for (int j = 2; j <= p; j++)
         for (int i = 2; i <= p; i++, o++)
            for (int d=0; d<3; d++)
            {
               du(o, d) = dmu(d) * phi_ij(i, j) + mu * dphi_ij(i, j, d);
            }
   }
   else
   {
      for (int j = 2; j <= p; j++)
         for (int i = 2; i <= p; i++, o++)
            for (int d=0; d<3; d++)
            {
               du(o, d) = 0.0;
            }
   }

   // Triangular faces
   if (CheckZ(z) && p >= 3)
   {
      // (a,b) = (1,2), c = 0
      phi_T(p, nu012(z, xy, 1), grad_nu012(z, xy, 1), phi_ij, dphi_ij);
      mu = mu0(z, xy, 2);
      dmu = grad_mu0(z, xy, 2);
      for (int i = 2; i < p; i++)
         for (int j = 1; i + j <= p; j++, o++)
            for (int d=0; d<3; d++)
            {
               du(o, d) = dmu(d) * phi_ij(i, j) + mu * dphi_ij(i, j, d);
            }

      // (a,b) = (1,2), c = 1
      mu = mu1(z, xy, 2);
      dmu = grad_mu1(z, xy, 2);
      for (int i = 2; i < p; i++)
         for (int j = 1; i + j <= p; j++, o++)
            for (int d=0; d<3; d++)
            {
               du(o, d) = dmu(d) * phi_ij(i, j) + mu * dphi_ij(i, j, d);
            }

      // (a,b) = (2,1), c = 0
      phi_T(p, nu012(z, xy, 2), grad_nu012(z, xy, 2), phi_ij, dphi_ij);
      mu = mu0(z, xy, 1);
      dmu = grad_mu0(z, xy, 1);
      for (int i = 2; i < p; i++)
         for (int j = 1; i + j <= p; j++, o++)
            for (int d=0; d<3; d++)
            {
               du(o, d) = dmu(d) * phi_ij(i, j) + mu * dphi_ij(i, j, d);
            }

      // (a,b) = (2,1), c = 1
      mu = mu1(z, xy, 1);
      dmu = grad_mu1(z, xy, 1);
      for (int i = 2; i < p; i++)
         for (int j = 1; i + j <= p; j++, o++)
            for (int d=0; d<3; d++)
            {
               du(o, d) = dmu(d) * phi_ij(i, j) + mu * dphi_ij(i, j, d);
            }
   }
   else
   {
      for (int i = 0; i < 2 * (p - 1) * (p - 2); i++, o++)
         for (int d=0; d<3; d++)
         {
            du(o, d) = 0.0;
         }
   }

   // Interior
   if (CheckZ(z) && p >= 2)
   {
      phi_Q(p, mu01(z, xy, 1), grad_mu01(z, xy, 1),
            mu01(z, xy, 2), grad_mu01(z, xy, 2), phi_ij, dphi_ij);
      phi_E(p, mu01(z), grad_mu01(z), phi_i, dphi_i);
      for (int k = 2; k <= p; k++)
         for (int j = 2; j <= p; j++)
            for (int i = 2; i <= p; i++, o++)
               for (int d=0; d<3; d++)
                  du(o, d) = dphi_ij(i, j, d) * phi_i(k) +
                             phi_ij(i, j) * dphi_i(k, d);
   }
   else
   {
      for (int i = 0; i < (p - 1) * (p - 1) * (p - 1); i++, o++)
         for (int d=0; d<3; d++)
         {
            du(o, d) = 0.0;
         }
   }
}

H1_BergotPyramidElement::H1_BergotPyramidElement(const int p, const int btype)
   : NodalFiniteElement(3, Geometry::PYRAMID,
                        (p + 1) * (p + 2) * (2 * p + 3) / 6, // Bergot (JSC)
                        p, FunctionSpace::Uk)
{
   const real_t *cp = poly1d.ClosedPoints(p, VerifyNodal(VerifyClosed(btype)));

#ifndef MFEM_THREAD_SAFE
   shape_x.SetSize(p + 1);
   shape_y.SetSize(p + 1);
   shape_z.SetSize(p + 1);
   dshape_x.SetSize(p + 1);
   dshape_y.SetSize(p + 1);
   dshape_z.SetSize(p + 1);
   dshape_z_dt.SetSize(p + 1);
   ddshape_x.SetSize(p + 1);
   ddshape_y.SetSize(p + 1);
   ddshape_z.SetSize(p + 1);
   u.SetSize(dof);
   du.SetSize(dof, dim);
   ddu.SetSize(dof, (dim * (dim + 1)) / 2);
#else
   Vector shape_x(p + 1), shape_y(p + 1), shape_z(p + 1);
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
         real_t w = cp[i] + cp[j] + cp[p-i-j];
         Nodes.IntPoint(o++).Set3(cp[i]/w, cp[0], cp[j]/w);
      }
   for (int j = 1; j < p; j++)
      for (int i = 1; i + j < p; i++)  // (1,2,4)
      {
         real_t w = cp[i] + cp[j] + cp[p-i-j];
         Nodes.IntPoint(o++).Set3(1.0 - cp[j]/w, cp[i]/w, cp[j]/w);
      }
   for (int j = 1; j < p; j++)
      for (int i = 1; i + j < p; i++)  // (3,4,2)
      {
         real_t w = cp[i] + cp[j] + cp[p-i-j];
         Nodes.IntPoint(o++).Set3(cp[j]/w, 1.0 - cp[i]/w, cp[i]/w);
      }
   for (int j = 1; j < p; j++)
      for (int i = 1; i + j < p; i++)  // (0,4,3)
      {
         real_t w = cp[i] + cp[j] + cp[p-i-j];
         Nodes.IntPoint(o++).Set3(cp[0], cp[j]/w, cp[i]/w);
      }

   // interior
   for (int k = 1; k < p - 1; k++)
   {
      for (int j = 1; j < p - k; j++)
      {
         real_t wjk = cp[j] + cp[k] + cp[p-j-k];
         for (int i = 1; i < p - k; i++)
         {
            real_t wik = cp[i] + cp[k] + cp[p-i-k];
            real_t w = wik * wjk * cp[p-k];
            Nodes.IntPoint(o++).Set3(cp[i] * (cp[j] + cp[p-j-k]) / w,
                                     cp[j] * (cp[i] + cp[p-i-k]) / w,
                                     cp[k] * cp[p-k] / w);
         }
      }
   }

   MFEM_ASSERT(o == dof,
               "Number of nodes does not match the "
               "number of degrees of freedom");
   DenseMatrix T(dof);

   for (int m = 0; m < dof; m++)
   {
      const IntegrationPoint &ip = Nodes.IntPoint(m);

      real_t x = (ip.z < 1.0) ? (ip.x / (1.0 - ip.z)) : 0.0;
      real_t y = (ip.z < 1.0) ? (ip.y / (1.0 - ip.z)) : 0.0;
      real_t z = ip.z;

      if (std::abs(z - 1.0) < apex_tol)
      {
         // Compute the limit of the basis functions as z->1 with x and y on the
         // line between the center of the base and the apex
         o = 0;
         for (int i = 0; i <= p; i++)
            for (int j = 0; j <= p; j++)
            {
               int maxij = std::max(i, j);
               for (int k = 0; k <= p - maxij; k++)
                  if (i == 0 && j == 0)
                  {
                     T(o++, m) = ((k + 3.) * k + 2.) / 2.;
                  }
                  else
                  {
                     T(o++, m) = 0.;
                  }
            }
      }
      else
      {
         poly1d.CalcLegendre(p, x, shape_x.GetData());
         poly1d.CalcLegendre(p, y, shape_y.GetData());

         o = 0;
         for (int i = 0; i <= p; i++)
         {
            for (int j = 0; j <= p; j++)
            {
               int maxij = std::max(i, j);
               FuentesPyramid::CalcScaledJacobi(p-maxij, 2.0 * (maxij + 1.0),
                                                z, 1.0, shape_z);

               for (int k = 0; k <= p - maxij; k++)
               {
                  T(o++, m) = shape_x(i) * shape_y(j) * shape_z(k) *
                              pow(1.0 - ip.z, maxij);
               }
            }
         }
      }
   }

   Ti.Factor(T);
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

   const real_t x = (ip.z < 1.0) ? (ip.x / (1.0 - ip.z)) : 0.0;
   const real_t y = (ip.z < 1.0) ? (ip.y / (1.0 - ip.z)) : 0.0;
   const real_t z = ip.z;

   if (std::abs(z - 1.0) < apex_tol)
   {
      // Compute the limit of the basis functions as z->1 with x and y on the
      // line between the center of the base and the apex
      u = 0.;
      int o = 0;
      for (int i = 0; i <= p; i++)
         for (int j = 0; j <= p; j++)
         {
            int maxij = std::max(i, j);
            for (int k = 0; k <= p - maxij; k++, o++)
               if (i == 0 && j == 0)
               {
                  u(o) = ((k + 3.) * k + 2.) / 2.;
               }
         }
   }
   else
   {
      poly1d.CalcLegendre(p, x, shape_x.GetData());
      poly1d.CalcLegendre(p, y, shape_y.GetData());

      int o = 0;
      for (int i = 0; i <= p; i++)
         for (int j = 0; j <= p; j++)
         {
            int maxij = std::max(i, j);
            FuentesPyramid::CalcScaledJacobi(p-maxij, 2.0 * (maxij + 1.0),
                                             z, 1.0, shape_z);
            for (int k = 0; k <= p - maxij; k++)
               u[o++] = shape_x(i) * shape_y(j) * shape_z(k) *
                        pow(1.0 - ip.z, maxij);
         }
   }
   Ti.Mult(u, shape);
}

void H1_BergotPyramidElement::CalcDShape(const IntegrationPoint &ip,
                                         DenseMatrix &dshape) const
{
   const int p = order;

#ifdef MFEM_THREAD_SAFE
   DenseMatrix du(dof, dim);
   Vector shape_x(order+1);
   Vector shape_y(order+1);
   Vector shape_z(order+1);
   Vector dshape_x(order+1);
   Vector dshape_y(order+1);
   Vector dshape_z(order+1);
   Vector dshape_z_dt(order+1);
#endif
   const real_t x = (ip.z < 1.0) ? (ip.x / (1.0 - ip.z)) : 0.0;
   const real_t y = (ip.z < 1.0) ? (ip.y / (1.0 - ip.z)) : 0.0;
   const real_t z = ip.z;

   if (std::abs(z - 1.0) < apex_tol)
   {
      // Compute the limit of the gradients of the basis functions as
      // z->1 with x and y on the line between the center of the base and the
      // apex
      du = 0.;
      int o = 0;
      for (int i = 0; i <= p; i++)
         for (int j = 0; j <= p; j++)
         {
            int maxij = std::max(i, j);
            for (int k = 0; k <= p - maxij; k++, o++)
            {
               if (i == 0 && j == 0)
               {
                  du(o,2) = (((k + 6.) * k + 11.) * k + 6.) * k / 6.;
               }
               else if (i == 1 && j == 0)
               {
                  du(o,0) = ((((k + 10.) * k + 35.) * k + 50.) * k + 24.) / 24.;
               }
               else if (i == 0 && j == 1)
               {
                  du(o,1) = ((((k + 10.) * k + 35.) * k + 50.) * k + 24.) / 24.;
               }
            }
         }
   }
   else
   {
      poly1d.CalcLegendre(p, x, shape_x.GetData(), dshape_x.GetData());
      poly1d.CalcLegendre(p, y, shape_y.GetData(), dshape_y.GetData());

      int o = 0;
      for (int i = 0; i <= p; i++)
         for (int j = 0; j <= p; j++)
         {
            int maxij = std::max(i, j);
            FuentesPyramid::CalcScaledJacobi(p-maxij, 2.0 * (maxij + 1.0),
                                             z, 1.0,
                                             shape_z, dshape_z, dshape_z_dt);

            for (int k = 0; k <= p - maxij; k++, o++)
            {
               du(o,0) = dshape_x(i) * shape_y(j) * shape_z(k) *
                         pow(1.0 - ip.z, maxij - 1);
               du(o,1) = shape_x(i) * dshape_y(j) * shape_z(k) *
                         pow(1.0 - ip.z, maxij - 1);
               du(o,2) = shape_x(i) * shape_y(j) * dshape_z(k) *
                         pow(1.0 - ip.z, maxij) +
                         (ip.x * dshape_x(i) * shape_y(j) +
                          ip.y * shape_x(i) * dshape_y(j)) *
                         shape_z(k) * pow(1.0 - ip.z, maxij - 2) -
                         maxij * shape_x(i) * shape_y(j) * shape_z(k) *
                         (maxij > 0 ? pow(1.0 - ip.z, maxij - 1) : 0.0);
            }
         }
   }
   Ti.Mult(du, dshape);
}


}
