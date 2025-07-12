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

// L2 Finite Element classes

#include "fe_l2.hpp"
#include "fe_h1.hpp"
#include "../coefficient.hpp"

namespace mfem
{

using namespace std;

L2_SegmentElement::L2_SegmentElement(const int p, const int btype)
   : NodalTensorFiniteElement(1, p, VerifyOpen(btype), L2_DOF_MAP)
{
   const real_t *op = poly1d.OpenPoints(p, btype);

#ifndef MFEM_THREAD_SAFE
   shape_x.SetSize(p + 1);
   dshape_x.SetDataAndSize(NULL, p + 1);
#endif

   for (int i = 0; i <= p; i++)
   {
      Nodes.IntPoint(i).x = op[i];
   }
}

void L2_SegmentElement::CalcShape(const IntegrationPoint &ip,
                                  Vector &shape) const
{
   basis1d.ScaleIntegrated(map_type == VALUE);
   basis1d.Eval(ip.x, shape);
}

void L2_SegmentElement::CalcDShape(const IntegrationPoint &ip,
                                   DenseMatrix &dshape) const
{
#ifdef MFEM_THREAD_SAFE
   Vector shape_x(dof), dshape_x(dshape.Data(), dof);
#else
   dshape_x.SetData(dshape.Data());
#endif
   basis1d.ScaleIntegrated(map_type == VALUE);
   basis1d.Eval(ip.x, shape_x, dshape_x);
}

void L2_SegmentElement::ProjectDelta(int vertex, Vector &dofs) const
{
   const int p = order;
   const real_t *op = poly1d.OpenPoints(p, b_type);

   switch (vertex)
   {
      case 0:
         for (int i = 0; i <= p; i++)
         {
            dofs(i) = poly1d.CalcDelta(p,(1.0 - op[i]));
         }
         break;

      case 1:
         for (int i = 0; i <= p; i++)
         {
            dofs(i) = poly1d.CalcDelta(p,op[i]);
         }
         break;
   }
}


L2_QuadrilateralElement::L2_QuadrilateralElement(const int p, const int btype)
   : NodalTensorFiniteElement(2, p, VerifyOpen(btype), L2_DOF_MAP)
{
   const real_t *op = poly1d.OpenPoints(p, b_type);

#ifndef MFEM_THREAD_SAFE
   shape_x.SetSize(p + 1);
   shape_y.SetSize(p + 1);
   dshape_x.SetSize(p + 1);
   dshape_y.SetSize(p + 1);
#endif

   for (int o = 0, j = 0; j <= p; j++)
      for (int i = 0; i <= p; i++)
      {
         Nodes.IntPoint(o++).Set2(op[i], op[j]);
      }
}

void L2_QuadrilateralElement::CalcShape(const IntegrationPoint &ip,
                                        Vector &shape) const
{
   const int p = order;

#ifdef MFEM_THREAD_SAFE
   Vector shape_x(p+1), shape_y(p+1);
#endif

   basis1d.ScaleIntegrated(map_type == VALUE);
   basis1d.Eval(ip.x, shape_x);
   basis1d.Eval(ip.y, shape_y);

   for (int o = 0, j = 0; j <= p; j++)
      for (int i = 0; i <= p; i++)
      {
         shape(o++) = shape_x(i)*shape_y(j);
      }
}

void L2_QuadrilateralElement::CalcDShape(const IntegrationPoint &ip,
                                         DenseMatrix &dshape) const
{
   const int p = order;

#ifdef MFEM_THREAD_SAFE
   Vector shape_x(p+1), shape_y(p+1), dshape_x(p+1), dshape_y(p+1);
#endif

   basis1d.ScaleIntegrated(map_type == VALUE);
   basis1d.Eval(ip.x, shape_x, dshape_x);
   basis1d.Eval(ip.y, shape_y, dshape_y);

   for (int o = 0, j = 0; j <= p; j++)
      for (int i = 0; i <= p; i++)
      {
         dshape(o,0) = dshape_x(i)* shape_y(j);
         dshape(o,1) =  shape_x(i)*dshape_y(j);  o++;
      }
}

void L2_QuadrilateralElement::ProjectDelta(int vertex, Vector &dofs) const
{
   const int p = order;
   const real_t *op = poly1d.OpenPoints(p, b_type);

#ifdef MFEM_THREAD_SAFE
   Vector shape_x(p+1), shape_y(p+1);
#endif

   for (int i = 0; i <= p; i++)
   {
      shape_x(i) = poly1d.CalcDelta(p,(1.0 - op[i]));
      shape_y(i) = poly1d.CalcDelta(p,op[i]);
   }

   switch (vertex)
   {
      case 0:
         for (int o = 0, j = 0; j <= p; j++)
            for (int i = 0; i <= p; i++)
            {
               dofs[o++] = shape_x(i)*shape_x(j);
            }
         break;
      case 1:
         for (int o = 0, j = 0; j <= p; j++)
            for (int i = 0; i <= p; i++)
            {
               dofs[o++] = shape_y(i)*shape_x(j);
            }
         break;
      case 2:
         for (int o = 0, j = 0; j <= p; j++)
            for (int i = 0; i <= p; i++)
            {
               dofs[o++] = shape_y(i)*shape_y(j);
            }
         break;
      case 3:
         for (int o = 0, j = 0; j <= p; j++)
            for (int i = 0; i <= p; i++)
            {
               dofs[o++] = shape_x(i)*shape_y(j);
            }
         break;
   }
}

void L2_QuadrilateralElement::ProjectDiv(const FiniteElement &fe,
                                         ElementTransformation &Trans,
                                         DenseMatrix &div) const
{
   if (basis1d.IsIntegratedType())
   {
      // Compute subcell integrals of the divergence
      const int fe_ndof = fe.GetDof();
      Vector div_shape(fe_ndof);
      div.SetSize(dof, fe_ndof);
      div = 0.0;

      const IntegrationRule &ir = IntRules.Get(geom_type, fe.GetOrder());
      const real_t *gll_pts = poly1d.GetPoints(order+1, BasisType::GaussLobatto);

      // Loop over subcells
      for (int iy = 0; iy < order+1; ++iy)
      {
         const real_t hy = gll_pts[iy+1] - gll_pts[iy];
         for (int ix = 0; ix < order+1; ++ix)
         {
            const int i = ix + iy*(order+1);
            const real_t hx = gll_pts[ix+1] - gll_pts[ix];
            // Loop over subcell quadrature points
            for (int iq = 0; iq < ir.Size(); ++iq)
            {
               IntegrationPoint ip = ir[iq];
               ip.x = gll_pts[ix] + hx*ip.x;
               ip.y = gll_pts[iy] + hy*ip.y;
               Trans.SetIntPoint(&ip);
               fe.CalcDivShape(ip, div_shape);
               real_t w = ip.weight;
               if (map_type == VALUE)
               {
                  const real_t detJ = Trans.Weight();
                  w /= detJ;
               }
               else if (map_type == INTEGRAL)
               {
                  w *= hx*hy;
               }
               for (int j = 0; j < fe_ndof; j++)
               {
                  const real_t div_j = div_shape(j);
                  div(i,j) += w*div_j;
               }
            }
         }
      }
      // Filter small entries
      for (int i = 0; i < dof; ++i)
      {
         for (int j = 0; j < fe_ndof; j++)
         {
            if (std::fabs(div(i,j)) < 1e-12) { div(i,j) = 0.0; }
         }
      }
   }
   else
   {
      // Fall back on standard nodal interpolation
      NodalFiniteElement::ProjectDiv(fe, Trans, div);
   }
}

void L2_QuadrilateralElement::Project(Coefficient &coeff,
                                      ElementTransformation &Trans,
                                      Vector &dofs) const
{
   if (basis1d.IsIntegratedType())
   {
      const IntegrationRule &ir = IntRules.Get(geom_type, order);
      const real_t *gll_pts = poly1d.GetPoints(order+1, BasisType::GaussLobatto);

      dofs = 0.0;
      // Loop over subcells
      for (int iy = 0; iy < order+1; ++iy)
      {
         const real_t hy = gll_pts[iy+1] - gll_pts[iy];
         for (int ix = 0; ix < order+1; ++ix)
         {
            const int i = ix + iy*(order+1);
            const real_t hx = gll_pts[ix+1] - gll_pts[ix];
            // Loop over subcell quadrature points
            for (int iq = 0; iq < ir.Size(); ++iq)
            {
               IntegrationPoint ip = ir[iq];
               ip.x = gll_pts[ix] + hx*ip.x;
               ip.y = gll_pts[iy] + hy*ip.y;
               Trans.SetIntPoint(&ip);
               const real_t val = coeff.Eval(Trans, ip);
               real_t w = ip.weight;
               if (map_type == INTEGRAL)
               {
                  w *= hx*hy*Trans.Weight();
               }
               dofs[i] += val*w;
            }
         }
      }
   }
   else
   {
      NodalFiniteElement::Project(coeff, Trans, dofs);
   }
}


L2_HexahedronElement::L2_HexahedronElement(const int p, const int btype)
   : NodalTensorFiniteElement(3, p, VerifyOpen(btype), L2_DOF_MAP)
{
   const real_t *op = poly1d.OpenPoints(p, btype);

#ifndef MFEM_THREAD_SAFE
   shape_x.SetSize(p + 1);
   shape_y.SetSize(p + 1);
   shape_z.SetSize(p + 1);
   dshape_x.SetSize(p + 1);
   dshape_y.SetSize(p + 1);
   dshape_z.SetSize(p + 1);
#endif

   for (int o = 0, k = 0; k <= p; k++)
      for (int j = 0; j <= p; j++)
         for (int i = 0; i <= p; i++)
         {
            Nodes.IntPoint(o++).Set3(op[i], op[j], op[k]);
         }
}

void L2_HexahedronElement::CalcShape(const IntegrationPoint &ip,
                                     Vector &shape) const
{
   const int p = order;

#ifdef MFEM_THREAD_SAFE
   Vector shape_x(p+1), shape_y(p+1), shape_z(p+1);
#endif

   basis1d.ScaleIntegrated(map_type == VALUE);
   basis1d.Eval(ip.x, shape_x);
   basis1d.Eval(ip.y, shape_y);
   basis1d.Eval(ip.z, shape_z);

   for (int o = 0, k = 0; k <= p; k++)
      for (int j = 0; j <= p; j++)
         for (int i = 0; i <= p; i++)
         {
            shape(o++) = shape_x(i)*shape_y(j)*shape_z(k);
         }
}

void L2_HexahedronElement::CalcDShape(const IntegrationPoint &ip,
                                      DenseMatrix &dshape) const
{
   const int p = order;

#ifdef MFEM_THREAD_SAFE
   Vector shape_x(p+1),  shape_y(p+1),  shape_z(p+1);
   Vector dshape_x(p+1), dshape_y(p+1), dshape_z(p+1);
#endif

   basis1d.ScaleIntegrated(map_type == VALUE);
   basis1d.Eval(ip.x, shape_x, dshape_x);
   basis1d.Eval(ip.y, shape_y, dshape_y);
   basis1d.Eval(ip.z, shape_z, dshape_z);

   for (int o = 0, k = 0; k <= p; k++)
      for (int j = 0; j <= p; j++)
         for (int i = 0; i <= p; i++)
         {
            dshape(o,0) = dshape_x(i)* shape_y(j)* shape_z(k);
            dshape(o,1) =  shape_x(i)*dshape_y(j)* shape_z(k);
            dshape(o,2) =  shape_x(i)* shape_y(j)*dshape_z(k);  o++;
         }
}

void L2_HexahedronElement::ProjectDelta(int vertex, Vector &dofs) const
{
   const int p = order;
   const real_t *op = poly1d.OpenPoints(p, b_type);

#ifdef MFEM_THREAD_SAFE
   Vector shape_x(p+1), shape_y(p+1);
#endif

   for (int i = 0; i <= p; i++)
   {
      shape_x(i) = poly1d.CalcDelta(p,(1.0 - op[i]));
      shape_y(i) = poly1d.CalcDelta(p,op[i]);
   }

   switch (vertex)
   {
      case 0:
         for (int o = 0, k = 0; k <= p; k++)
            for (int j = 0; j <= p; j++)
               for (int i = 0; i <= p; i++)
               {
                  dofs[o++] = shape_x(i)*shape_x(j)*shape_x(k);
               }
         break;
      case 1:
         for (int o = 0, k = 0; k <= p; k++)
            for (int j = 0; j <= p; j++)
               for (int i = 0; i <= p; i++)
               {
                  dofs[o++] = shape_y(i)*shape_x(j)*shape_x(k);
               }
         break;
      case 2:
         for (int o = 0, k = 0; k <= p; k++)
            for (int j = 0; j <= p; j++)
               for (int i = 0; i <= p; i++)
               {
                  dofs[o++] = shape_y(i)*shape_y(j)*shape_x(k);
               }
         break;
      case 3:
         for (int o = 0, k = 0; k <= p; k++)
            for (int j = 0; j <= p; j++)
               for (int i = 0; i <= p; i++)
               {
                  dofs[o++] = shape_x(i)*shape_y(j)*shape_x(k);
               }
         break;
      case 4:
         for (int o = 0, k = 0; k <= p; k++)
            for (int j = 0; j <= p; j++)
               for (int i = 0; i <= p; i++)
               {
                  dofs[o++] = shape_x(i)*shape_x(j)*shape_y(k);
               }
         break;
      case 5:
         for (int o = 0, k = 0; k <= p; k++)
            for (int j = 0; j <= p; j++)
               for (int i = 0; i <= p; i++)
               {
                  dofs[o++] = shape_y(i)*shape_x(j)*shape_y(k);
               }
         break;
      case 6:
         for (int o = 0, k = 0; k <= p; k++)
            for (int j = 0; j <= p; j++)
               for (int i = 0; i <= p; i++)
               {
                  dofs[o++] = shape_y(i)*shape_y(j)*shape_y(k);
               }
         break;
      case 7:
         for (int o = 0, k = 0; k <= p; k++)
            for (int j = 0; j <= p; j++)
               for (int i = 0; i <= p; i++)
               {
                  dofs[o++] = shape_x(i)*shape_y(j)*shape_y(k);
               }
         break;
   }
}

void L2_HexahedronElement::ProjectDiv(const FiniteElement &fe,
                                      ElementTransformation &Trans,
                                      DenseMatrix &div) const
{
   if (basis1d.IsIntegratedType())
   {
      // Compute subcell integrals of the divergence
      const int fe_ndof = fe.GetDof();
      Vector div_shape(fe_ndof);
      div.SetSize(dof, fe_ndof);
      div = 0.0;

      const IntegrationRule &ir = IntRules.Get(geom_type, fe.GetOrder());
      const real_t *gll_pts = poly1d.GetPoints(order+1, BasisType::GaussLobatto);

      // Loop over subcells
      for (int iz = 0; iz < order+1; ++iz)
      {
         const real_t hz = gll_pts[iz+1] - gll_pts[iz];
         for (int iy = 0; iy < order+1; ++iy)
         {
            const real_t hy = gll_pts[iy+1] - gll_pts[iy];
            for (int ix = 0; ix < order+1; ++ix)
            {
               const int i = ix + iy*(order+1) + iz*(order+1)*(order+1);
               const real_t hx = gll_pts[ix+1] - gll_pts[ix];
               // Loop over subcell quadrature points
               for (int iq = 0; iq < ir.Size(); ++iq)
               {
                  IntegrationPoint ip = ir[iq];
                  ip.x = gll_pts[ix] + hx*ip.x;
                  ip.y = gll_pts[iy] + hy*ip.y;
                  ip.z = gll_pts[iz] + hz*ip.z;
                  Trans.SetIntPoint(&ip);
                  fe.CalcDivShape(ip, div_shape);
                  real_t w = ip.weight;
                  if (map_type == VALUE)
                  {
                     const real_t detJ = Trans.Weight();
                     w /= detJ;
                  }
                  else if (map_type == INTEGRAL)
                  {
                     w *= hx*hy*hz;
                  }
                  for (int j = 0; j < fe_ndof; j++)
                  {
                     const real_t div_j = div_shape(j);
                     div(i,j) += w*div_j;
                  }
               }
            }
         }
      }
      // Filter small entries
      for (int i = 0; i < dof; ++i)
      {
         for (int j = 0; j < fe_ndof; j++)
         {
            if (std::fabs(div(i,j)) < 1e-12) { div(i,j) = 0.0; }
         }
      }
   }
   else
   {
      // Fall back on standard nodal interpolation
      NodalFiniteElement::ProjectDiv(fe, Trans, div);
   }
}

void L2_HexahedronElement::Project(Coefficient &coeff,
                                   ElementTransformation &Trans,
                                   Vector &dofs) const
{
   if (basis1d.IsIntegratedType())
   {
      const IntegrationRule &ir = IntRules.Get(geom_type, order);
      const real_t *gll_pts = poly1d.GetPoints(order+1, BasisType::GaussLobatto);

      dofs = 0.0;
      // Loop over subcells
      for (int iz = 0; iz < order+1; ++iz)
      {
         const real_t hz = gll_pts[iz+1] - gll_pts[iz];
         for (int iy = 0; iy < order+1; ++iy)
         {
            const real_t hy = gll_pts[iy+1] - gll_pts[iy];
            for (int ix = 0; ix < order+1; ++ix)
            {
               const real_t hx = gll_pts[ix+1] - gll_pts[ix];
               const int i = ix + iy*(order+1) + iz*(order+1)*(order+1);
               // Loop over subcell quadrature points
               for (int iq = 0; iq < ir.Size(); ++iq)
               {
                  IntegrationPoint ip = ir[iq];
                  ip.x = gll_pts[ix] + hx*ip.x;
                  ip.y = gll_pts[iy] + hy*ip.y;
                  ip.z = gll_pts[iz] + hz*ip.z;
                  Trans.SetIntPoint(&ip);
                  const real_t val = coeff.Eval(Trans, ip);
                  real_t w = ip.weight;
                  if (map_type == INTEGRAL)
                  {
                     const real_t detJ = Trans.Weight();
                     w *= detJ*hx*hy*hz;
                  }
                  dofs[i] += val*w;
               }
            }
         }
      }
   }
   else
   {
      NodalFiniteElement::Project(coeff, Trans, dofs);
   }
}


L2_TriangleElement::L2_TriangleElement(const int p, const int btype)
   : NodalFiniteElement(2, Geometry::TRIANGLE, ((p + 1)*(p + 2))/2, p,
                        FunctionSpace::Pk)
{
   const real_t *op = poly1d.OpenPoints(p, VerifyOpen(btype));

#ifndef MFEM_THREAD_SAFE
   shape_x.SetSize(p + 1);
   shape_y.SetSize(p + 1);
   shape_l.SetSize(p + 1);
   dshape_x.SetSize(p + 1);
   dshape_y.SetSize(p + 1);
   dshape_l.SetSize(p + 1);
   u.SetSize(dof);
   du.SetSize(dof, dim);
#else
   Vector shape_x(p + 1), shape_y(p + 1), shape_l(p + 1);
#endif

   for (int o = 0, j = 0; j <= p; j++)
      for (int i = 0; i + j <= p; i++)
      {
         real_t w = op[i] + op[j] + op[p-i-j];
         Nodes.IntPoint(o++).Set2(op[i]/w, op[j]/w);
      }

   DenseMatrix T(dof);
   for (int k = 0; k < dof; k++)
   {
      IntegrationPoint &ip = Nodes.IntPoint(k);
      poly1d.CalcBasis(p, ip.x, shape_x);
      poly1d.CalcBasis(p, ip.y, shape_y);
      poly1d.CalcBasis(p, 1. - ip.x - ip.y, shape_l);

      for (int o = 0, j = 0; j <= p; j++)
         for (int i = 0; i + j <= p; i++)
         {
            T(o++, k) = shape_x(i)*shape_y(j)*shape_l(p-i-j);
         }
   }

   Ti.Factor(T);
   // mfem::out << "L2_TriangleElement(" << p << ") : "; Ti.TestInversion();
}

void L2_TriangleElement::CalcShape(const IntegrationPoint &ip,
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

void L2_TriangleElement::CalcDShape(const IntegrationPoint &ip,
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

void L2_TriangleElement::ProjectDelta(int vertex, Vector &dofs) const
{
   switch (vertex)
   {
      case 0:
         for (int i = 0; i < dof; i++)
         {
            const IntegrationPoint &ip = Nodes.IntPoint(i);
            dofs[i] = pow(1.0 - ip.x - ip.y, order);
         }
         break;
      case 1:
         for (int i = 0; i < dof; i++)
         {
            const IntegrationPoint &ip = Nodes.IntPoint(i);
            dofs[i] = pow(ip.x, order);
         }
         break;
      case 2:
         for (int i = 0; i < dof; i++)
         {
            const IntegrationPoint &ip = Nodes.IntPoint(i);
            dofs[i] = pow(ip.y, order);
         }
         break;
   }
}


L2_TetrahedronElement::L2_TetrahedronElement(const int p, const int btype)
   : NodalFiniteElement(3, Geometry::TETRAHEDRON, ((p + 1)*(p + 2)*(p + 3))/6,
                        p, FunctionSpace::Pk)
{
   const real_t *op = poly1d.OpenPoints(p, VerifyOpen(btype));

#ifndef MFEM_THREAD_SAFE
   shape_x.SetSize(p + 1);
   shape_y.SetSize(p + 1);
   shape_z.SetSize(p + 1);
   shape_l.SetSize(p + 1);
   dshape_x.SetSize(p + 1);
   dshape_y.SetSize(p + 1);
   dshape_z.SetSize(p + 1);
   dshape_l.SetSize(p + 1);
   u.SetSize(dof);
   du.SetSize(dof, dim);
#else
   Vector shape_x(p + 1), shape_y(p + 1), shape_z(p + 1), shape_l(p + 1);
#endif

   for (int o = 0, k = 0; k <= p; k++)
      for (int j = 0; j + k <= p; j++)
         for (int i = 0; i + j + k <= p; i++)
         {
            real_t w = op[i] + op[j] + op[k] + op[p-i-j-k];
            Nodes.IntPoint(o++).Set3(op[i]/w, op[j]/w, op[k]/w);
         }

   DenseMatrix T(dof);
   for (int m = 0; m < dof; m++)
   {
      IntegrationPoint &ip = Nodes.IntPoint(m);
      poly1d.CalcBasis(p, ip.x, shape_x);
      poly1d.CalcBasis(p, ip.y, shape_y);
      poly1d.CalcBasis(p, ip.z, shape_z);
      poly1d.CalcBasis(p, 1. - ip.x - ip.y - ip.z, shape_l);

      for (int o = 0, k = 0; k <= p; k++)
         for (int j = 0; j + k <= p; j++)
            for (int i = 0; i + j + k <= p; i++)
            {
               T(o++, m) = shape_x(i)*shape_y(j)*shape_z(k)*shape_l(p-i-j-k);
            }
   }

   Ti.Factor(T);
   // mfem::out << "L2_TetrahedronElement(" << p << ") : "; Ti.TestInversion();
}

void L2_TetrahedronElement::CalcShape(const IntegrationPoint &ip,
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

void L2_TetrahedronElement::CalcDShape(const IntegrationPoint &ip,
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

void L2_TetrahedronElement::ProjectDelta(int vertex, Vector &dofs) const
{
   switch (vertex)
   {
      case 0:
         for (int i = 0; i < dof; i++)
         {
            const IntegrationPoint &ip = Nodes.IntPoint(i);
            dofs[i] = pow(1.0 - ip.x - ip.y - ip.z, order);
         }
         break;
      case 1:
         for (int i = 0; i < dof; i++)
         {
            const IntegrationPoint &ip = Nodes.IntPoint(i);
            dofs[i] = pow(ip.x, order);
         }
         break;
      case 2:
         for (int i = 0; i < dof; i++)
         {
            const IntegrationPoint &ip = Nodes.IntPoint(i);
            dofs[i] = pow(ip.y, order);
         }
         break;
      case 3:
         for (int i = 0; i < dof; i++)
         {
            const IntegrationPoint &ip = Nodes.IntPoint(i);
            dofs[i] = pow(ip.z, order);
         }
         break;
   }
}


L2_WedgeElement::L2_WedgeElement(const int p, const int btype)
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

   // Interior DoFs
   int m=0;
   for (int k=0; k<=p; k++)
   {
      int l=0;
      for (int j=0; j<=p; j++)
      {
         for (int i=0; i<=j; i++)
         {
            t_dof[m] = l;
            s_dof[m] = k;
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

void L2_WedgeElement::CalcShape(const IntegrationPoint &ip,
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

void L2_WedgeElement::CalcDShape(const IntegrationPoint &ip,
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

L2_FuentesPyramidElement::L2_FuentesPyramidElement(const int p, const int btype)
   : NodalFiniteElement(3, Geometry::PYRAMID, ((p + 1)*(p + 1)*(p + 1)),
                        p, FunctionSpace::Uk)
{
   const real_t *op = poly1d.OpenPoints(p, VerifyOpen(btype));

   // These basis functions are not independent on a closed set of
   // interpolation points when p >= 1. For this reason we force the points
   // to be open in the z direction whenever closed points are requested.
   // This should be regarded as a limitation of this choice of basis function.
   // If a truly closed set of points is needed consider using
   // L2_BergotPyramidElement instead.
   real_t a = 1.0;
   if (IsClosedType(btype) && p > 0)
   {
      a = (poly1d.GetPoints(p, BasisType::GaussLegendre))[p];
   }


#ifndef MFEM_THREAD_SAFE
   shape_x.SetSize(p + 1);
   shape_y.SetSize(p + 1);
   shape_z.SetSize(p + 1);
   dshape_x.SetSize(p + 1);
   dshape_y.SetSize(p + 1);
   dshape_z.SetSize(p + 1);
   u.SetSize(dof);
   du.SetSize(dof, dim);
#else
   Vector shape_x(p + 1);
   Vector shape_y(p + 1);
   Vector shape_z(p + 1);
#endif

   int o = 0;
   for (int k = 0; k <= p; k++)
      for (int j = 0; j <= p; j++)
         for (int i = 0; i <= p; i++)
         {
            Nodes.IntPoint(o++).Set3(op[i] * (1.0 - a * op[k]),
                                     op[j] * (1.0 - a * op[k]),
                                     a * op[k]);
         }

   MFEM_ASSERT(o == dof,
               "Number of nodes does not match the "
               "number of degrees of freedom");
   DenseMatrix T(dof);

   for (int m = 0; m < dof; m++)
   {
      const IntegrationPoint &ip = Nodes.IntPoint(m);
      real_t x = ip.x;
      real_t y = ip.y;
      real_t z = ip.z;
      Vector xy({x,y});
      CalcHomogenizedScaLegendre(p, mu0(z, xy, 1), mu1(z, xy, 1), shape_x);
      CalcHomogenizedScaLegendre(p, mu0(z, xy, 2), mu1(z, xy, 2), shape_y);
      CalcHomogenizedScaLegendre(p, mu0(z), mu1(z), shape_z);

      o = 0;
      for (int k = 0; k <= p; k++)
      {
         for (int j = 0; j <= p; j++)
         {
            for (int i = 0; i <= p; i++, o++)
            {
               T(o, m) = shape_x[i] * shape_y[j] * shape_z[k];
            }
         }
      }
   }

   Ti.Factor(T);
}

void L2_FuentesPyramidElement::CalcShape(const IntegrationPoint &ip,
                                         Vector &shape) const
{
   const int p = order;

#ifdef MFEM_THREAD_SAFE
   Vector shape_x(p + 1);
   Vector shape_y(p + 1);
   Vector shape_z(p + 1);
   Vector u(dof);
#endif
   real_t x = ip.x;
   real_t y = ip.y;
   real_t z = ip.z;
   Vector xy({x,y});

   if (z < 1.0)
   {
      CalcHomogenizedScaLegendre(p, mu0(z, xy, 1), mu1(z, xy, 1), shape_x);
      CalcHomogenizedScaLegendre(p, mu0(z, xy, 2), mu1(z, xy, 2), shape_y);
   }
   else
   {
      shape_x = 0.0; shape_x(0) = 1.0;
      shape_y = 0.0; shape_y(0) = 1.0;
   }
   CalcHomogenizedScaLegendre(p, mu0(z), mu1(z), shape_z);

   int o = 0;
   for (int k = 0; k <= p; k++)
      for (int j = 0; j <= p; j++)
         for (int i = 0; i <= p; i++, o++)
         {
            u[o] = shape_x[i] * shape_y[j] * shape_z[k];
         }

   Ti.Mult(u, shape);
}

void L2_FuentesPyramidElement::CalcDShape(const IntegrationPoint &ip,
                                          DenseMatrix &dshape) const
{
   const int p = order;

#ifdef MFEM_THREAD_SAFE
   Vector shape_x(p + 1);
   Vector shape_y(p + 1);
   Vector shape_z(p + 1);
   Vector dshape_x(p + 1);
   Vector dshape_y(p + 1);
   Vector dshape_z(p + 1);
   DenseMatrix du(dof, dim);
#endif

   Poly_1D::CalcLegendre(p, ip.x / (1.0 - ip.z), shape_x.GetData(),
                         dshape_x.GetData());
   Poly_1D::CalcLegendre(p, ip.y / (1.0 - ip.z), shape_y.GetData(),
                         dshape_y.GetData());
   Poly_1D::CalcLegendre(p, ip.z, shape_z.GetData(), dshape_z.GetData());

   int o = 0;
   for (int k = 0; k <= p; k++)
      for (int j = 0; j <= p; j++)
         for (int i = 0; i <= p; i++, o++)
         {
            du(o, 0) = dshape_x[i] * shape_y[j] * shape_z[k] / (1.0 - ip.z);
            du(o, 1) = shape_x[i] * dshape_y[j] * shape_z[k] / (1.0 - ip.z);
            du(o, 2) = shape_x[i] * shape_y[j] * dshape_z[k] +
                       (ip.x * dshape_x[i] * shape_y[j] +
                        ip.y * shape_x[i] * dshape_y[j]) *
                       shape_z[k] / pow(1.0 - ip.z, 2);
         }
   Ti.Mult(du, dshape);
}

L2_BergotPyramidElement::L2_BergotPyramidElement(const int p, const int btype)
   : NodalFiniteElement(3, Geometry::PYRAMID, (p + 1)*(p + 2)*(2*p + 3)/6,
                        p, FunctionSpace::Pk)
{
   const real_t *op = poly1d.OpenPoints(p, VerifyOpen(btype));

#ifndef MFEM_THREAD_SAFE
   shape_x.SetSize(p + 1);
   shape_y.SetSize(p + 1);
   shape_z.SetSize(p + 1);
   dshape_x.SetSize(p + 1);
   dshape_y.SetSize(p + 1);
   dshape_z.SetSize(p + 1);
   dshape_z_dt.SetSize(p + 1);
   u.SetSize(dof);
   du.SetSize(dof, dim);
#else
   Vector shape_x(p + 1);
   Vector shape_y(p + 1);
   Vector shape_z(p + 1);
   Vector dshape_z_dt(p + 1);
#endif

   int o = 0;
   for (int k = 0; k <= p; k++)
      for (int j = 0; j <= p - k; j++)
      {
         const real_t wjk = op[j] + op[k] + op[p-j-k];
         for (int i = 0; i <= p - k; i++)
         {
            const real_t wik = op[i] + op[k] + op[p-i-k];
            const real_t w = wik * wjk * op[p-k];
            Nodes.IntPoint(o++).Set3(op[i] * (op[j] + op[p-j-k]) / w,
                                     op[j] * (op[j] + op[p-j-k]) / w,
                                     op[k] * op[p-k] / w);
         }
      }

   MFEM_ASSERT(o == dof,
               "Number of nodes does not match the "
               "number of degrees of freedom");
   DenseMatrix T(dof);

   for (int m = 0; m < dof; m++)
   {
      const IntegrationPoint &ip = Nodes.IntPoint(m);

      const real_t x = (ip.z < 1.0) ? (ip.x / (1.0 - ip.z)) : 0.0;
      const real_t y = (ip.z < 1.0) ? (ip.y / (1.0 - ip.z)) : 0.0;
      const real_t z = ip.z;

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

   Ti.Factor(T);
}

void L2_BergotPyramidElement::CalcShape(const IntegrationPoint &ip,
                                        Vector &shape) const
{
   const int p = order;

#ifdef MFEM_THREAD_SAFE
   Vector shape_x(p + 1);
   Vector shape_y(p + 1);
   Vector shape_z(p + 1);
   Vector u(dof);
#endif

   const real_t x = (ip.z < 1.0) ? (ip.x / (1.0 - ip.z)) : 0.0;
   const real_t y = (ip.z < 1.0) ? (ip.y / (1.0 - ip.z)) : 0.0;
   const real_t z = ip.z;

   poly1d.CalcLegendre(p, x, shape_x.GetData());
   poly1d.CalcLegendre(p, y, shape_y.GetData());

   int o = 0;
   for (int i = 0; i <= p; i++)
   {
      for (int j = 0; j <= p; j++)
      {
         int maxij = std::max(i, j);
         FuentesPyramid::CalcScaledJacobi(p-maxij, 2.0 * (maxij + 1.0), z, 1.0,
                                          shape_z);

         for (int k = 0; k <= p - maxij; k++)
         {
            u[o++] = shape_x(i) * shape_y(j) * shape_z(k) *
                     pow(1.0 - ip.z, maxij);
         }
      }
   }

   Ti.Mult(u, shape);
}

void L2_BergotPyramidElement::CalcDShape(const IntegrationPoint &ip,
                                         DenseMatrix &dshape) const
{
   const int p = order;

#ifdef MFEM_THREAD_SAFE
   Vector shape_x(p + 1);
   Vector shape_y(p + 1);
   Vector shape_z(p + 1);
   Vector dshape_x(p + 1);
   Vector dshape_y(p + 1);
   Vector dshape_z(p + 1);
   Vector dshape_z_dt(p + 1);
   DenseMatrix du(dof, dim);
#endif

   const real_t x = (ip.z < 1.0) ? (ip.x / (1.0 - ip.z)) : 0.0;
   const real_t y = (ip.z < 1.0) ? (ip.y / (1.0 - ip.z)) : 0.0;
   const real_t z = ip.z;

   Poly_1D::CalcLegendre(p, x, shape_x.GetData(), dshape_x.GetData());
   Poly_1D::CalcLegendre(p, y, shape_y.GetData(), dshape_y.GetData());

   int o = 0;
   for (int i = 0; i <= p; i++)
   {
      for (int j = 0; j <= p; j++)
      {
         int maxij = std::max(i, j);
         FuentesPyramid::CalcScaledJacobi(p-maxij, 2.0 * (maxij + 1.0), z, 1.0,
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
                      ((maxij > 0) ? (maxij * shape_x(i) * shape_y(j) * shape_z(k) *
                                      pow(1.0 - ip.z, maxij - 1)) : 0.0);
         }
      }
   }

   Ti.Mult(du, dshape);
}

}
