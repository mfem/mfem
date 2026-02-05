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

// Raviart-Thomas Finite Element classes

#include "fe_rt.hpp"
#include "face_map_utils.hpp"
#include "../coefficient.hpp"

namespace mfem
{

using namespace std;

const real_t RT_QuadrilateralElement::nk[8] =
{ 0., -1.,  1., 0.,  0., 1.,  -1., 0. };

RT_QuadrilateralElement::RT_QuadrilateralElement(const int p,
                                                 const int cb_type,
                                                 const int ob_type)
   : VectorTensorFiniteElement(2, 2*(p + 1)*(p + 2), p + 1, cb_type, ob_type,
                               H_DIV, DofMapType::L2_DOF_MAP),
     dof2nk(dof),
     cp(poly1d.ClosedPoints(p + 1, cb_type))
{
   if (obasis1d.IsIntegratedType()) { is_nodal = false; }

   dof_map.SetSize(dof);

   const real_t *op = poly1d.OpenPoints(p, ob_type);
   const int dof2 = dof/2;

#ifndef MFEM_THREAD_SAFE
   shape_cx.SetSize(p + 2);
   shape_ox.SetSize(p + 1);
   shape_cy.SetSize(p + 2);
   shape_oy.SetSize(p + 1);
   dshape_cx.SetSize(p + 2);
   dshape_cy.SetSize(p + 2);
#endif

   // edges
   int o = 0;
   for (int i = 0; i <= p; i++)  // (0,1)
   {
      dof_map[1*dof2 + i + 0*(p + 1)] = o++;
   }
   for (int i = 0; i <= p; i++)  // (1,2)
   {
      dof_map[0*dof2 + (p + 1) + i*(p + 2)] = o++;
   }
   for (int i = 0; i <= p; i++)  // (2,3)
   {
      dof_map[1*dof2 + (p - i) + (p + 1)*(p + 1)] = o++;
   }
   for (int i = 0; i <= p; i++)  // (3,0)
   {
      dof_map[0*dof2 + 0 + (p - i)*(p + 2)] = o++;
   }

   // interior
   for (int j = 0; j <= p; j++)  // x-components
      for (int i = 1; i <= p; i++)
      {
         dof_map[0*dof2 + i + j*(p + 2)] = o++;
      }
   for (int j = 1; j <= p; j++)  // y-components
      for (int i = 0; i <= p; i++)
      {
         dof_map[1*dof2 + i + j*(p + 1)] = o++;
      }

   // dof orientations
   // x-components
   for (int j = 0; j <= p; j++)
      for (int i = 0; i <= p/2; i++)
      {
         int idx = 0*dof2 + i + j*(p + 2);
         dof_map[idx] = -1 - dof_map[idx];
      }
   if (p%2 == 1)
      for (int j = p/2 + 1; j <= p; j++)
      {
         int idx = 0*dof2 + (p/2 + 1) + j*(p + 2);
         dof_map[idx] = -1 - dof_map[idx];
      }
   // y-components
   for (int j = 0; j <= p/2; j++)
      for (int i = 0; i <= p; i++)
      {
         int idx = 1*dof2 + i + j*(p + 1);
         dof_map[idx] = -1 - dof_map[idx];
      }
   if (p%2 == 1)
      for (int i = 0; i <= p/2; i++)
      {
         int idx = 1*dof2 + i + (p/2 + 1)*(p + 1);
         dof_map[idx] = -1 - dof_map[idx];
      }

   o = 0;
   for (int j = 0; j <= p; j++)
      for (int i = 0; i <= p + 1; i++)
      {
         int idx;
         if ((idx = dof_map[o++]) < 0)
         {
            idx = -1 - idx;
            dof2nk[idx] = 3;
         }
         else
         {
            dof2nk[idx] = 1;
         }
         Nodes.IntPoint(idx).Set2(cp[i], op[j]);
      }
   for (int j = 0; j <= p + 1; j++)
      for (int i = 0; i <= p; i++)
      {
         int idx;
         if ((idx = dof_map[o++]) < 0)
         {
            idx = -1 - idx;
            dof2nk[idx] = 0;
         }
         else
         {
            dof2nk[idx] = 2;
         }
         Nodes.IntPoint(idx).Set2(op[i], cp[j]);
      }
}

void RT_QuadrilateralElement::CalcVShape(const IntegrationPoint &ip,
                                         DenseMatrix &shape) const
{
   const int pp1 = order;

#ifdef MFEM_THREAD_SAFE
   Vector shape_cx(pp1 + 1), shape_ox(pp1), shape_cy(pp1 + 1), shape_oy(pp1);
#endif

   if (obasis1d.IsIntegratedType())
   {
#ifdef MFEM_THREAD_SAFE
      Vector dshape_cx(pp1 + 1), dshape_cy(pp1 + 1);
#endif
      basis1d.Eval(ip.x, shape_cx, dshape_cx);
      basis1d.Eval(ip.y, shape_cy, dshape_cy);
      obasis1d.ScaleIntegrated(false);
      obasis1d.EvalIntegrated(dshape_cx, shape_ox);
      obasis1d.EvalIntegrated(dshape_cy, shape_oy);
   }
   else
   {
      basis1d.Eval(ip.x, shape_cx);
      basis1d.Eval(ip.y, shape_cy);
      obasis1d.Eval(ip.x, shape_ox);
      obasis1d.Eval(ip.y, shape_oy);
   }

   int o = 0;
   for (int j = 0; j < pp1; j++)
      for (int i = 0; i <= pp1; i++)
      {
         int idx, s;
         if ((idx = dof_map[o++]) < 0)
         {
            idx = -1 - idx, s = -1;
         }
         else
         {
            s = +1;
         }
         shape(idx,0) = s*shape_cx(i)*shape_oy(j);
         shape(idx,1) = 0.;
      }
   for (int j = 0; j <= pp1; j++)
      for (int i = 0; i < pp1; i++)
      {
         int idx, s;
         if ((idx = dof_map[o++]) < 0)
         {
            idx = -1 - idx, s = -1;
         }
         else
         {
            s = +1;
         }
         shape(idx,0) = 0.;
         shape(idx,1) = s*shape_ox(i)*shape_cy(j);
      }
}

void RT_QuadrilateralElement::CalcDivShape(const IntegrationPoint &ip,
                                           Vector &divshape) const
{
   const int pp1 = order;

#ifdef MFEM_THREAD_SAFE
   Vector shape_cx(pp1 + 1), shape_ox(pp1), shape_cy(pp1 + 1), shape_oy(pp1);
   Vector dshape_cx(pp1 + 1), dshape_cy(pp1 + 1);
#endif

   basis1d.Eval(ip.x, shape_cx, dshape_cx);
   basis1d.Eval(ip.y, shape_cy, dshape_cy);
   if (obasis1d.IsIntegratedType())
   {
      obasis1d.ScaleIntegrated(false);
      obasis1d.EvalIntegrated(dshape_cx, shape_ox);
      obasis1d.EvalIntegrated(dshape_cy, shape_oy);
   }
   else
   {
      obasis1d.Eval(ip.x, shape_ox);
      obasis1d.Eval(ip.y, shape_oy);
   }

   int o = 0;
   for (int j = 0; j < pp1; j++)
      for (int i = 0; i <= pp1; i++)
      {
         int idx, s;
         if ((idx = dof_map[o++]) < 0)
         {
            idx = -1 - idx, s = -1;
         }
         else
         {
            s = +1;
         }
         divshape(idx) = s*dshape_cx(i)*shape_oy(j);
      }
   for (int j = 0; j <= pp1; j++)
      for (int i = 0; i < pp1; i++)
      {
         int idx, s;
         if ((idx = dof_map[o++]) < 0)
         {
            idx = -1 - idx, s = -1;
         }
         else
         {
            s = +1;
         }
         divshape(idx) = s*shape_ox(i)*dshape_cy(j);
      }
}

void RT_QuadrilateralElement::ProjectIntegrated(VectorCoefficient &vc,
                                                ElementTransformation &Trans,
                                                Vector &dofs) const
{
   MFEM_ASSERT(obasis1d.IsIntegratedType(), "Not integrated type");
   real_t vk[Geometry::MaxDim];
   Vector xk(vk, vc.GetVDim());

   const IntegrationRule &ir = IntRules.Get(Geometry::SEGMENT, order);
   const int nqpt = ir.GetNPoints();

   IntegrationPoint ip2d;

   int o = 0;
   for (int c = 0; c < 2; c++)
   {
      int im = (c == 0) ? order + 1 : order;
      int jm = (c == 1) ? order + 1 : order;
      for (int j = 0; j < jm; j++)
         for (int i = 0; i < im; i++)
         {
            int idx = dof_map[o++];
            if (idx < 0) { idx = -1 - idx; }
            int ic = (c == 0) ? j : i;
            const real_t h = cp[ic+1] - cp[ic];
            real_t val = 0.0;
            for (int k = 0; k < nqpt; k++)
            {
               const IntegrationPoint &ip1d = ir.IntPoint(k);
               if (c == 0) { ip2d.Set2(cp[i], cp[j] + (h*ip1d.x)); }
               else { ip2d.Set2(cp[i] + (h*ip1d.x), cp[j]); }
               Trans.SetIntPoint(&ip2d);
               vc.Eval(xk, Trans, ip2d);
               // nk^t adj(J) xk
               const real_t ipval = Trans.AdjugateJacobian().InnerProduct(vk,
                                                                          nk + dof2nk[idx]*dim);
               val += ip1d.weight*ipval;
            }
            dofs(idx) = val*h;
         }
   }
}

void RT_QuadrilateralElement::GetFaceMap(const int face_id,
                                         Array<int> &face_map) const
{
   const int p = order;
   const int pp1 = p + 1;
   const int n_face_dofs = p;

   std::vector<int> offsets;
   std::vector<int> strides = {(face_id == 0 || face_id == 2) ? 1 : pp1};
   switch (face_id)
   {
      case 0: offsets = {p*pp1}; break; // y = 0
      case 1: offsets = {pp1 - 1}; break; // x = 1
      case 2: offsets = {p*pp1 + p*(pp1 - 1)}; break; // y = 1
      case 3: offsets = {0}; break; // x = 0
   }

   std::vector<int> n_dofs(dim - 1, p);
   internal::FillFaceMap(n_face_dofs, offsets, strides, n_dofs, face_map);
}


const real_t RT_HexahedronElement::nk[18] =
{ 0.,0.,-1.,  0.,-1.,0.,  1.,0.,0.,  0.,1.,0.,  -1.,0.,0.,  0.,0.,1. };

RT_HexahedronElement::RT_HexahedronElement(const int p,
                                           const int cb_type,
                                           const int ob_type)
   : VectorTensorFiniteElement(3, 3*(p + 1)*(p + 1)*(p + 2), p + 1, cb_type,
                               ob_type, H_DIV, DofMapType::L2_DOF_MAP),
     dof2nk(dof),
     cp(poly1d.ClosedPoints(p + 1, cb_type))
{
   if (obasis1d.IsIntegratedType()) { is_nodal = false; }

   dof_map.SetSize(dof);

   const real_t *op = poly1d.OpenPoints(p, ob_type);
   const int dof3 = dof/3;

#ifndef MFEM_THREAD_SAFE
   shape_cx.SetSize(p + 2);
   shape_ox.SetSize(p + 1);
   shape_cy.SetSize(p + 2);
   shape_oy.SetSize(p + 1);
   shape_cz.SetSize(p + 2);
   shape_oz.SetSize(p + 1);
   dshape_cx.SetSize(p + 2);
   dshape_cy.SetSize(p + 2);
   dshape_cz.SetSize(p + 2);
#endif

   // faces
   int o = 0;
   for (int j = 0; j <= p; j++)  // (3,2,1,0) -- bottom
      for (int i = 0; i <= p; i++)
      {
         dof_map[2*dof3 + i + ((p - j) + 0*(p + 1))*(p + 1)] = o++;
      }
   for (int j = 0; j <= p; j++)  // (0,1,5,4) -- front
      for (int i = 0; i <= p; i++)
      {
         dof_map[1*dof3 + i + (0 + j*(p + 2))*(p + 1)] = o++;
      }
   for (int j = 0; j <= p; j++)  // (1,2,6,5) -- right
      for (int i = 0; i <= p; i++)
      {
         dof_map[0*dof3 + (p + 1) + (i + j*(p + 1))*(p + 2)] = o++;
      }
   for (int j = 0; j <= p; j++)  // (2,3,7,6) -- back
      for (int i = 0; i <= p; i++)
      {
         dof_map[1*dof3 + (p - i) + ((p + 1) + j*(p + 2))*(p + 1)] = o++;
      }
   for (int j = 0; j <= p; j++)  // (3,0,4,7) -- left
      for (int i = 0; i <= p; i++)
      {
         dof_map[0*dof3 + 0 + ((p - i) + j*(p + 1))*(p + 2)] = o++;
      }
   for (int j = 0; j <= p; j++)  // (4,5,6,7) -- top
      for (int i = 0; i <= p; i++)
      {
         dof_map[2*dof3 + i + (j + (p + 1)*(p + 1))*(p + 1)] = o++;
      }

   // interior
   // x-components
   for (int k = 0; k <= p; k++)
      for (int j = 0; j <= p; j++)
         for (int i = 1; i <= p; i++)
         {
            dof_map[0*dof3 + i + (j + k*(p + 1))*(p + 2)] = o++;
         }
   // y-components
   for (int k = 0; k <= p; k++)
      for (int j = 1; j <= p; j++)
         for (int i = 0; i <= p; i++)
         {
            dof_map[1*dof3 + i + (j + k*(p + 2))*(p + 1)] = o++;
         }
   // z-components
   for (int k = 1; k <= p; k++)
      for (int j = 0; j <= p; j++)
         for (int i = 0; i <= p; i++)
         {
            dof_map[2*dof3 + i + (j + k*(p + 1))*(p + 1)] = o++;
         }

   // dof orientations
   // for odd p, do not change the orientations in the mid-planes
   // {i = p/2 + 1}, {j = p/2 + 1}, {k = p/2 + 1} in the x, y, z-components
   // respectively.
   // x-components
   for (int k = 0; k <= p; k++)
      for (int j = 0; j <= p; j++)
         for (int i = 0; i <= p/2; i++)
         {
            int idx = 0*dof3 + i + (j + k*(p + 1))*(p + 2);
            dof_map[idx] = -1 - dof_map[idx];
         }
   // y-components
   for (int k = 0; k <= p; k++)
      for (int j = 0; j <= p/2; j++)
         for (int i = 0; i <= p; i++)
         {
            int idx = 1*dof3 + i + (j + k*(p + 2))*(p + 1);
            dof_map[idx] = -1 - dof_map[idx];
         }
   // z-components
   for (int k = 0; k <= p/2; k++)
      for (int j = 0; j <= p; j++)
         for (int i = 0; i <= p; i++)
         {
            int idx = 2*dof3 + i + (j + k*(p + 1))*(p + 1);
            dof_map[idx] = -1 - dof_map[idx];
         }

   o = 0;
   // x-components
   for (int k = 0; k <= p; k++)
      for (int j = 0; j <= p; j++)
         for (int i = 0; i <= p + 1; i++)
         {
            int idx;
            if ((idx = dof_map[o++]) < 0)
            {
               idx = -1 - idx;
               dof2nk[idx] = 4;
            }
            else
            {
               dof2nk[idx] = 2;
            }
            Nodes.IntPoint(idx).Set3(cp[i], op[j], op[k]);
         }
   // y-components
   for (int k = 0; k <= p; k++)
      for (int j = 0; j <= p + 1; j++)
         for (int i = 0; i <= p; i++)
         {
            int idx;
            if ((idx = dof_map[o++]) < 0)
            {
               idx = -1 - idx;
               dof2nk[idx] = 1;
            }
            else
            {
               dof2nk[idx] = 3;
            }
            Nodes.IntPoint(idx).Set3(op[i], cp[j], op[k]);
         }
   // z-components
   for (int k = 0; k <= p + 1; k++)
      for (int j = 0; j <= p; j++)
         for (int i = 0; i <= p; i++)
         {
            int idx;
            if ((idx = dof_map[o++]) < 0)
            {
               idx = -1 - idx;
               dof2nk[idx] = 0;
            }
            else
            {
               dof2nk[idx] = 5;
            }
            Nodes.IntPoint(idx).Set3(op[i], op[j], cp[k]);
         }
}

void RT_HexahedronElement::CalcVShape(const IntegrationPoint &ip,
                                      DenseMatrix &shape) const
{
   const int pp1 = order;

#ifdef MFEM_THREAD_SAFE
   Vector shape_cx(pp1 + 1), shape_ox(pp1), shape_cy(pp1 + 1), shape_oy(pp1);
   Vector shape_cz(pp1 + 1), shape_oz(pp1);
#endif

   if (obasis1d.IsIntegratedType())
   {
#ifdef MFEM_THREAD_SAFE
      Vector dshape_cx(pp1 + 1), dshape_cy(pp1 + 1), dshape_cz(pp1 + 1);
#endif
      basis1d.Eval(ip.x, shape_cx, dshape_cx);
      basis1d.Eval(ip.y, shape_cy, dshape_cy);
      basis1d.Eval(ip.z, shape_cz, dshape_cz);
      obasis1d.ScaleIntegrated(false);
      obasis1d.EvalIntegrated(dshape_cx, shape_ox);
      obasis1d.EvalIntegrated(dshape_cy, shape_oy);
      obasis1d.EvalIntegrated(dshape_cz, shape_oz);
   }
   else
   {
      basis1d.Eval(ip.x, shape_cx);
      basis1d.Eval(ip.y, shape_cy);
      basis1d.Eval(ip.z, shape_cz);
      obasis1d.Eval(ip.x, shape_ox);
      obasis1d.Eval(ip.y, shape_oy);
      obasis1d.Eval(ip.z, shape_oz);
   }

   int o = 0;
   // x-components
   for (int k = 0; k < pp1; k++)
      for (int j = 0; j < pp1; j++)
         for (int i = 0; i <= pp1; i++)
         {
            int idx, s;
            if ((idx = dof_map[o++]) < 0)
            {
               idx = -1 - idx, s = -1;
            }
            else
            {
               s = +1;
            }
            shape(idx,0) = s*shape_cx(i)*shape_oy(j)*shape_oz(k);
            shape(idx,1) = 0.;
            shape(idx,2) = 0.;
         }
   // y-components
   for (int k = 0; k < pp1; k++)
      for (int j = 0; j <= pp1; j++)
         for (int i = 0; i < pp1; i++)
         {
            int idx, s;
            if ((idx = dof_map[o++]) < 0)
            {
               idx = -1 - idx, s = -1;
            }
            else
            {
               s = +1;
            }
            shape(idx,0) = 0.;
            shape(idx,1) = s*shape_ox(i)*shape_cy(j)*shape_oz(k);
            shape(idx,2) = 0.;
         }
   // z-components
   for (int k = 0; k <= pp1; k++)
      for (int j = 0; j < pp1; j++)
         for (int i = 0; i < pp1; i++)
         {
            int idx, s;
            if ((idx = dof_map[o++]) < 0)
            {
               idx = -1 - idx, s = -1;
            }
            else
            {
               s = +1;
            }
            shape(idx,0) = 0.;
            shape(idx,1) = 0.;
            shape(idx,2) = s*shape_ox(i)*shape_oy(j)*shape_cz(k);
         }
}

void RT_HexahedronElement::CalcDivShape(const IntegrationPoint &ip,
                                        Vector &divshape) const
{
   const int pp1 = order;

#ifdef MFEM_THREAD_SAFE
   Vector shape_cx(pp1 + 1), shape_ox(pp1), shape_cy(pp1 + 1), shape_oy(pp1);
   Vector shape_cz(pp1 + 1), shape_oz(pp1);
   Vector dshape_cx(pp1 + 1), dshape_cy(pp1 + 1), dshape_cz(pp1 + 1);
#endif

   basis1d.Eval(ip.x, shape_cx, dshape_cx);
   basis1d.Eval(ip.y, shape_cy, dshape_cy);
   basis1d.Eval(ip.z, shape_cz, dshape_cz);
   if (obasis1d.IsIntegratedType())
   {
      obasis1d.ScaleIntegrated(false);
      obasis1d.EvalIntegrated(dshape_cx, shape_ox);
      obasis1d.EvalIntegrated(dshape_cy, shape_oy);
      obasis1d.EvalIntegrated(dshape_cz, shape_oz);
   }
   else
   {
      obasis1d.Eval(ip.x, shape_ox);
      obasis1d.Eval(ip.y, shape_oy);
      obasis1d.Eval(ip.z, shape_oz);
   }

   int o = 0;
   // x-components
   for (int k = 0; k < pp1; k++)
      for (int j = 0; j < pp1; j++)
         for (int i = 0; i <= pp1; i++)
         {
            int idx, s;
            if ((idx = dof_map[o++]) < 0)
            {
               idx = -1 - idx, s = -1;
            }
            else
            {
               s = +1;
            }
            divshape(idx) = s*dshape_cx(i)*shape_oy(j)*shape_oz(k);
         }
   // y-components
   for (int k = 0; k < pp1; k++)
      for (int j = 0; j <= pp1; j++)
         for (int i = 0; i < pp1; i++)
         {
            int idx, s;
            if ((idx = dof_map[o++]) < 0)
            {
               idx = -1 - idx, s = -1;
            }
            else
            {
               s = +1;
            }
            divshape(idx) = s*shape_ox(i)*dshape_cy(j)*shape_oz(k);
         }
   // z-components
   for (int k = 0; k <= pp1; k++)
      for (int j = 0; j < pp1; j++)
         for (int i = 0; i < pp1; i++)
         {
            int idx, s;
            if ((idx = dof_map[o++]) < 0)
            {
               idx = -1 - idx, s = -1;
            }
            else
            {
               s = +1;
            }
            divshape(idx) = s*shape_ox(i)*shape_oy(j)*dshape_cz(k);
         }
}

void RT_HexahedronElement::ProjectIntegrated(VectorCoefficient &vc,
                                             ElementTransformation &Trans,
                                             Vector &dofs) const
{
   MFEM_ASSERT(obasis1d.IsIntegratedType(), "Not integrated type");
   real_t vq[Geometry::MaxDim];
   Vector xq(vq, vc.GetVDim());

   const IntegrationRule &ir2d = IntRules.Get(Geometry::SQUARE, order);
   const int nqpt = ir2d.GetNPoints();

   IntegrationPoint ip3d;

   int o = 0;
   for (int c = 0; c < 3; c++)
   {
      int im = (c == 0) ? order + 1 : order;
      int jm = (c == 1) ? order + 1 : order;
      int km = (c == 2) ? order + 1 : order;
      for (int k = 0; k < km; k++)
         for (int j = 0; j < jm; j++)
            for (int i = 0; i < im; i++)
            {
               int idx = dof_map[o++];
               if (idx < 0) { idx = -1 - idx; }
               int ic1, ic2;
               if (c == 0) { ic1 = j; ic2 = k; }
               else if (c == 1) { ic1 = i; ic2 = k; }
               else { ic1 = i; ic2 = j; }
               const real_t h1 = cp[ic1+1] - cp[ic1];
               const real_t h2 = cp[ic2+1] - cp[ic2];
               real_t val = 0.0;
               for (int q = 0; q < nqpt; q++)
               {
                  const IntegrationPoint &ip2d = ir2d.IntPoint(q);
                  if (c == 0) { ip3d.Set3(cp[i], cp[j] + h1*ip2d.x, cp[k] + h2*ip2d.y); }
                  else if (c == 1) { ip3d.Set3(cp[i] + h1*ip2d.x, cp[j], cp[k] + h2*ip2d.y); }
                  else { ip3d.Set3(cp[i] + h1*ip2d.x, cp[j] + h2*ip2d.y, cp[k]); }
                  Trans.SetIntPoint(&ip3d);
                  vc.Eval(xq, Trans, ip3d);
                  // nk^t adj(J) xq
                  const real_t ipval
                     = Trans.AdjugateJacobian().InnerProduct(vq, nk + dof2nk[idx]*dim);
                  val += ip2d.weight*ipval;
               }
               dofs(idx) = val*h1*h2;
            }
   }
}

void RT_HexahedronElement::GetFaceMap(const int face_id,
                                      Array<int> &face_map) const
{
   const int p = order;
   const int pp1 = p + 1;
   int n_face_dofs = p*p;
   std::vector<int> strides, offsets;
   const int n_dof_per_dim = p*p*pp1;
   const auto f = internal::GetFaceNormal3D(face_id);
   const int face_normal = f.first, level = f.second;
   if (face_normal == 0) // x-normal
   {
      offsets = {level ? pp1 - 1 : 0};
      strides = {pp1, p*pp1};
   }
   else if (face_normal == 1) // y-normal
   {
      offsets = {n_dof_per_dim + (level ? p*(pp1 - 1) : 0)};
      strides = {1, p*pp1};
   }
   else if (face_normal == 2) // z-normal
   {
      offsets = {2*n_dof_per_dim + (level ? p*p*(pp1 - 1) : 0)};
      strides = {1, p};
   }
   std::vector<int> n_dofs = {p, p};
   internal::FillFaceMap(n_face_dofs, offsets, strides, n_dofs, face_map);
}


const real_t RT_TriangleElement::nk[6] =
{ 0., -1., 1., 1., -1., 0. };

const real_t RT_TriangleElement::c = 1./3.;

RT_TriangleElement::RT_TriangleElement(const int p)
   : VectorFiniteElement(2, Geometry::TRIANGLE, (p + 1)*(p + 3), p + 1,
                         H_DIV, FunctionSpace::Pk),
     dof2nk(dof)
{
   const real_t *iop = (p > 0) ? poly1d.OpenPoints(p - 1) : NULL;
   const real_t *bop = poly1d.OpenPoints(p);

#ifndef MFEM_THREAD_SAFE
   shape_x.SetSize(p + 1);
   shape_y.SetSize(p + 1);
   shape_l.SetSize(p + 1);
   dshape_x.SetSize(p + 1);
   dshape_y.SetSize(p + 1);
   dshape_l.SetSize(p + 1);
   u.SetSize(dof, dim);
   divu.SetSize(dof);
#else
   Vector shape_x(p + 1), shape_y(p + 1), shape_l(p + 1);
#endif

   // edges
   int o = 0;
   for (int i = 0; i <= p; i++)  // (0,1)
   {
      Nodes.IntPoint(o).Set2(bop[i], 0.);
      dof2nk[o++] = 0;
   }
   for (int i = 0; i <= p; i++)  // (1,2)
   {
      Nodes.IntPoint(o).Set2(bop[p-i], bop[i]);
      dof2nk[o++] = 1;
   }
   for (int i = 0; i <= p; i++)  // (2,0)
   {
      Nodes.IntPoint(o).Set2(0., bop[p-i]);
      dof2nk[o++] = 2;
   }

   // interior
   for (int j = 0; j < p; j++)
      for (int i = 0; i + j < p; i++)
      {
         real_t w = iop[i] + iop[j] + iop[p-1-i-j];
         Nodes.IntPoint(o).Set2(iop[i]/w, iop[j]/w);
         dof2nk[o++] = 0;
         Nodes.IntPoint(o).Set2(iop[i]/w, iop[j]/w);
         dof2nk[o++] = 2;
      }

   DenseMatrix T(dof);
   for (int k = 0; k < dof; k++)
   {
      const IntegrationPoint &ip = Nodes.IntPoint(k);
      poly1d.CalcBasis(p, ip.x, shape_x);
      poly1d.CalcBasis(p, ip.y, shape_y);
      poly1d.CalcBasis(p, 1. - ip.x - ip.y, shape_l);
      const real_t *n_k = nk + 2*dof2nk[k];

      o = 0;
      for (int j = 0; j <= p; j++)
         for (int i = 0; i + j <= p; i++)
         {
            real_t s = shape_x(i)*shape_y(j)*shape_l(p-i-j);
            T(o++, k) = s*n_k[0];
            T(o++, k) = s*n_k[1];
         }
      for (int i = 0; i <= p; i++)
      {
         real_t s = shape_x(i)*shape_y(p-i);
         T(o++, k) = s*((ip.x - c)*n_k[0] + (ip.y - c)*n_k[1]);
      }
   }

   Ti.Factor(T);
   // mfem::out << "RT_TriangleElement(" << p << ") : "; Ti.TestInversion();
}

void RT_TriangleElement::CalcVShape(const IntegrationPoint &ip,
                                    DenseMatrix &shape) const
{
   const int p = order - 1;

#ifdef MFEM_THREAD_SAFE
   Vector shape_x(p + 1), shape_y(p + 1), shape_l(p + 1);
   DenseMatrix u(dof, dim);
#endif

   poly1d.CalcBasis(p, ip.x, shape_x);
   poly1d.CalcBasis(p, ip.y, shape_y);
   poly1d.CalcBasis(p, 1. - ip.x - ip.y, shape_l);

   int o = 0;
   for (int j = 0; j <= p; j++)
      for (int i = 0; i + j <= p; i++)
      {
         real_t s = shape_x(i)*shape_y(j)*shape_l(p-i-j);
         u(o,0) = s;  u(o,1) = 0;  o++;
         u(o,0) = 0;  u(o,1) = s;  o++;
      }
   for (int i = 0; i <= p; i++)
   {
      real_t s = shape_x(i)*shape_y(p-i);
      u(o,0) = (ip.x - c)*s;
      u(o,1) = (ip.y - c)*s;
      o++;
   }

   Ti.Mult(u, shape);
}

void RT_TriangleElement::CalcDivShape(const IntegrationPoint &ip,
                                      Vector &divshape) const
{
   const int p = order - 1;

#ifdef MFEM_THREAD_SAFE
   Vector shape_x(p + 1),  shape_y(p + 1),  shape_l(p + 1);
   Vector dshape_x(p + 1), dshape_y(p + 1), dshape_l(p + 1);
   Vector divu(dof);
#endif

   poly1d.CalcBasis(p, ip.x, shape_x, dshape_x);
   poly1d.CalcBasis(p, ip.y, shape_y, dshape_y);
   poly1d.CalcBasis(p, 1. - ip.x - ip.y, shape_l, dshape_l);

   int o = 0;
   for (int j = 0; j <= p; j++)
      for (int i = 0; i + j <= p; i++)
      {
         int k = p - i - j;
         divu(o++) = (dshape_x(i)*shape_l(k) -
                      shape_x(i)*dshape_l(k))*shape_y(j);
         divu(o++) = (dshape_y(j)*shape_l(k) -
                      shape_y(j)*dshape_l(k))*shape_x(i);
      }
   for (int i = 0; i <= p; i++)
   {
      int j = p - i;
      divu(o++) = ((shape_x(i) + (ip.x - c)*dshape_x(i))*shape_y(j) +
                   (shape_y(j) + (ip.y - c)*dshape_y(j))*shape_x(i));
   }

   Ti.Mult(divu, divshape);
}


const real_t RT_TetrahedronElement::nk[12] =
{ 1,1,1,  -1,0,0,  0,-1,0,  0,0,-1 };
// { .5,.5,.5, -.5,0,0, 0,-.5,0, 0,0,-.5}; // n_F |F|

const real_t RT_TetrahedronElement::c = 1./4.;

RT_TetrahedronElement::RT_TetrahedronElement(const int p)
   : VectorFiniteElement(3, Geometry::TETRAHEDRON, (p + 1)*(p + 2)*(p + 4)/2,
                         p + 1, H_DIV, FunctionSpace::Pk),
     dof2nk(dof)
{
   const real_t *iop = (p > 0) ? poly1d.OpenPoints(p - 1) : NULL;
   const real_t *bop = poly1d.OpenPoints(p);

#ifndef MFEM_THREAD_SAFE
   shape_x.SetSize(p + 1);
   shape_y.SetSize(p + 1);
   shape_z.SetSize(p + 1);
   shape_l.SetSize(p + 1);
   dshape_x.SetSize(p + 1);
   dshape_y.SetSize(p + 1);
   dshape_z.SetSize(p + 1);
   dshape_l.SetSize(p + 1);
   u.SetSize(dof, dim);
   divu.SetSize(dof);
#else
   Vector shape_x(p + 1), shape_y(p + 1), shape_z(p + 1), shape_l(p + 1);
#endif

   int o = 0;
   // faces (see Mesh::GenerateFaces in mesh/mesh.cpp,
   //        the constructor of H1_TetrahedronElement)
   for (int j = 0; j <= p; j++)
      for (int i = 0; i + j <= p; i++)  // (1,2,3)
      {
         real_t w = bop[i] + bop[j] + bop[p-i-j];
         Nodes.IntPoint(o).Set3(bop[p-i-j]/w, bop[i]/w, bop[j]/w);
         dof2nk[o++] = 0;
      }
   for (int j = 0; j <= p; j++)
      for (int i = 0; i + j <= p; i++)  // (0,3,2)
      {
         real_t w = bop[i] + bop[j] + bop[p-i-j];
         Nodes.IntPoint(o).Set3(0., bop[j]/w, bop[i]/w);
         dof2nk[o++] = 1;
      }
   for (int j = 0; j <= p; j++)
      for (int i = 0; i + j <= p; i++)  // (0,1,3)
      {
         real_t w = bop[i] + bop[j] + bop[p-i-j];
         Nodes.IntPoint(o).Set3(bop[i]/w, 0., bop[j]/w);
         dof2nk[o++] = 2;
      }
   for (int j = 0; j <= p; j++)
      for (int i = 0; i + j <= p; i++)  // (0,2,1)
      {
         real_t w = bop[i] + bop[j] + bop[p-i-j];
         Nodes.IntPoint(o).Set3(bop[j]/w, bop[i]/w, 0.);
         dof2nk[o++] = 3;
      }

   // interior
   for (int k = 0; k < p; k++)
      for (int j = 0; j + k < p; j++)
         for (int i = 0; i + j + k < p; i++)
         {
            real_t w = iop[i] + iop[j] + iop[k] + iop[p-1-i-j-k];
            Nodes.IntPoint(o).Set3(iop[i]/w, iop[j]/w, iop[k]/w);
            dof2nk[o++] = 1;
            Nodes.IntPoint(o).Set3(iop[i]/w, iop[j]/w, iop[k]/w);
            dof2nk[o++] = 2;
            Nodes.IntPoint(o).Set3(iop[i]/w, iop[j]/w, iop[k]/w);
            dof2nk[o++] = 3;
         }

   DenseMatrix T(dof);
   for (int m = 0; m < dof; m++)
   {
      const IntegrationPoint &ip = Nodes.IntPoint(m);
      poly1d.CalcBasis(p, ip.x, shape_x);
      poly1d.CalcBasis(p, ip.y, shape_y);
      poly1d.CalcBasis(p, ip.z, shape_z);
      poly1d.CalcBasis(p, 1. - ip.x - ip.y - ip.z, shape_l);
      const real_t *nm = nk + 3*dof2nk[m];

      o = 0;
      for (int k = 0; k <= p; k++)
         for (int j = 0; j + k <= p; j++)
            for (int i = 0; i + j + k <= p; i++)
            {
               real_t s = shape_x(i)*shape_y(j)*shape_z(k)*shape_l(p-i-j-k);
               T(o++, m) = s * nm[0];
               T(o++, m) = s * nm[1];
               T(o++, m) = s * nm[2];
            }
      for (int j = 0; j <= p; j++)
         for (int i = 0; i + j <= p; i++)
         {
            real_t s = shape_x(i)*shape_y(j)*shape_z(p-i-j);
            T(o++, m) = s*((ip.x - c)*nm[0] + (ip.y - c)*nm[1] +
                           (ip.z - c)*nm[2]);
         }
   }

   Ti.Factor(T);
   // mfem::out << "RT_TetrahedronElement(" << p << ") : "; Ti.TestInversion();
}

void RT_TetrahedronElement::CalcVShape(const IntegrationPoint &ip,
                                       DenseMatrix &shape) const
{
   const int p = order - 1;

#ifdef MFEM_THREAD_SAFE
   Vector shape_x(p + 1), shape_y(p + 1), shape_z(p + 1), shape_l(p + 1);
   DenseMatrix u(dof, dim);
#endif

   poly1d.CalcBasis(p, ip.x, shape_x);
   poly1d.CalcBasis(p, ip.y, shape_y);
   poly1d.CalcBasis(p, ip.z, shape_z);
   poly1d.CalcBasis(p, 1. - ip.x - ip.y - ip.z, shape_l);

   int o = 0;
   for (int k = 0; k <= p; k++)
      for (int j = 0; j + k <= p; j++)
         for (int i = 0; i + j + k <= p; i++)
         {
            real_t s = shape_x(i)*shape_y(j)*shape_z(k)*shape_l(p-i-j-k);
            u(o,0) = s;  u(o,1) = 0;  u(o,2) = 0;  o++;
            u(o,0) = 0;  u(o,1) = s;  u(o,2) = 0;  o++;
            u(o,0) = 0;  u(o,1) = 0;  u(o,2) = s;  o++;
         }
   for (int j = 0; j <= p; j++)
      for (int i = 0; i + j <= p; i++)
      {
         real_t s = shape_x(i)*shape_y(j)*shape_z(p-i-j);
         u(o,0) = (ip.x - c)*s;  u(o,1) = (ip.y - c)*s;  u(o,2) = (ip.z - c)*s;
         o++;
      }

   Ti.Mult(u, shape);
}

void RT_TetrahedronElement::CalcDivShape(const IntegrationPoint &ip,
                                         Vector &divshape) const
{
   const int p = order - 1;

#ifdef MFEM_THREAD_SAFE
   Vector shape_x(p + 1),  shape_y(p + 1),  shape_z(p + 1),  shape_l(p + 1);
   Vector dshape_x(p + 1), dshape_y(p + 1), dshape_z(p + 1), dshape_l(p + 1);
   Vector divu(dof);
#endif

   poly1d.CalcBasis(p, ip.x, shape_x, dshape_x);
   poly1d.CalcBasis(p, ip.y, shape_y, dshape_y);
   poly1d.CalcBasis(p, ip.z, shape_z, dshape_z);
   poly1d.CalcBasis(p, 1. - ip.x - ip.y - ip.z, shape_l, dshape_l);

   int o = 0;
   for (int k = 0; k <= p; k++)
      for (int j = 0; j + k <= p; j++)
         for (int i = 0; i + j + k <= p; i++)
         {
            int l = p - i - j - k;
            divu(o++) = (dshape_x(i)*shape_l(l) -
                         shape_x(i)*dshape_l(l))*shape_y(j)*shape_z(k);
            divu(o++) = (dshape_y(j)*shape_l(l) -
                         shape_y(j)*dshape_l(l))*shape_x(i)*shape_z(k);
            divu(o++) = (dshape_z(k)*shape_l(l) -
                         shape_z(k)*dshape_l(l))*shape_x(i)*shape_y(j);
         }
   for (int j = 0; j <= p; j++)
      for (int i = 0; i + j <= p; i++)
      {
         int k = p - i - j;
         divu(o++) =
            (shape_x(i) + (ip.x - c)*dshape_x(i))*shape_y(j)*shape_z(k) +
            (shape_y(j) + (ip.y - c)*dshape_y(j))*shape_x(i)*shape_z(k) +
            (shape_z(k) + (ip.z - c)*dshape_z(k))*shape_x(i)*shape_y(j);
      }

   Ti.Mult(divu, divshape);
}

const real_t RT_WedgeElement::nk[15] =
{ 0,0,-1, 0,0,1, 0,-1,0, 1,1,0, -1,0,0};

RT_WedgeElement::RT_WedgeElement(const int p)
   : VectorFiniteElement(3, Geometry::PRISM,
                         (p + 2) * ((p + 1) * (p + 2)) / 2 +
                         (p + 1) * (p + 1) * (p + 3), p + 1,
                         H_DIV, FunctionSpace::Qk),
     dof2nk(dof),
     t_dof(dof),
     s_dof(dof),
     L2TriangleFE(p),
     RTTriangleFE(p),
     H1SegmentFE(p + 1),
     L2SegmentFE(p)
{
   MFEM_ASSERT(L2TriangleFE.GetDof() * H1SegmentFE.GetDof() +
               RTTriangleFE.GetDof() * L2SegmentFE.GetDof() == dof,
               "Mismatch in number of degrees of freedom "
               "when building RT_WedgeElement!");

   const int pm1 = p - 1;

#ifndef MFEM_THREAD_SAFE
   tl2_shape.SetSize(L2TriangleFE.GetDof());
   sh1_shape.SetSize(H1SegmentFE.GetDof());
   trt_shape.SetSize(RTTriangleFE.GetDof(), 2);
   sl2_shape.SetSize(L2SegmentFE.GetDof());
   sh1_dshape.SetSize(H1SegmentFE.GetDof(), 1);
   trt_dshape.SetSize(RTTriangleFE.GetDof());
#endif

   const IntegrationRule &tl2_n = L2TriangleFE.GetNodes();
   const IntegrationRule &trt_n = RTTriangleFE.GetNodes();
   const IntegrationRule &sh1_n = H1SegmentFE.GetNodes();
   const IntegrationRule &sl2_n = L2SegmentFE.GetNodes();

   // faces
   int o = 0;
   int l = 0;
   // (0,2,1) -- bottom
   for (int j = 0; j <= p; j++)
      for (int i = 0; i + j <= p; i++)
      {
         l = j + i * (2 * p + 3 - i) / 2;
         t_dof[o] = l; s_dof[o] = 0; dof2nk[o] = 0;
         const IntegrationPoint & t_ip = tl2_n.IntPoint(t_dof[o]);
         Nodes.IntPoint(o).Set3(t_ip.x, t_ip.y, sh1_n.IntPoint(s_dof[o]).x);
         o++;
      }
   // (3,4,5) -- top
   l = 0;
   for (int j = 0; j <= p; j++)
      for (int i = 0; i + j <= p; i++)
      {
         t_dof[o] = l; s_dof[o] = 1; dof2nk[o] = 1; l++;
         const IntegrationPoint & t_ip = tl2_n.IntPoint(t_dof[o]);
         Nodes.IntPoint(o).Set3(t_ip.x, t_ip.y, sh1_n.IntPoint(s_dof[o]).x);
         o++;
      }
   // (0, 1, 4, 3) -- xz plane
   for (int j = 0; j <= p; j++)
      for (int i = 0; i <= p; i++)
      {
         t_dof[o] = i; s_dof[o] = j; dof2nk[o] = 2;
         const IntegrationPoint & t_ip = trt_n.IntPoint(t_dof[o]);
         Nodes.IntPoint(o).Set3(t_ip.x, t_ip.y, sl2_n.IntPoint(s_dof[o]).x);
         o++;
      }
   // (1, 2, 5, 4) -- (y-x)z plane
   for (int j = 0; j <= p; j++)
      for (int i = 0; i <= p; i++)
      {
         t_dof[o] = p + 1 + i; s_dof[o] = j; dof2nk[o] = 3;
         const IntegrationPoint & t_ip = trt_n.IntPoint(t_dof[o]);
         Nodes.IntPoint(o).Set3(t_ip.x, t_ip.y, sl2_n.IntPoint(s_dof[o]).x);
         o++;
      }
   // (2, 0, 3, 5) -- yz plane
   for (int j = 0; j <= p; j++)
      for (int i = 0; i <= p; i++)
      {
         t_dof[o] = 2 * p + 2 + i; s_dof[o] = j; dof2nk[o] = 4;
         const IntegrationPoint & t_ip = trt_n.IntPoint(t_dof[o]);
         Nodes.IntPoint(o).Set3(t_ip.x, t_ip.y, sl2_n.IntPoint(s_dof[o]).x);
         o++;
      }

   // interior
   for (int k = 0; k < L2SegmentFE.GetDof(); k++)
   {
      l = 0;
      for (int j = 0; j <= pm1; j++)
         for (int i = 0; i + j <= pm1; i++)
         {
            t_dof[o] = 3 * (p + 1) + 2 * l;     s_dof[o] = k;
            dof2nk[o] = 2;
            const IntegrationPoint & t_ip0 = trt_n.IntPoint(t_dof[o]);
            const IntegrationPoint & s_ip0 = sl2_n.IntPoint(s_dof[o]);
            Nodes.IntPoint(o).Set3(t_ip0.x, t_ip0.y, s_ip0.x);
            o++;
            t_dof[o] = 3 * (p + 1) + 2 * l + 1; s_dof[o] = k;
            dof2nk[o] = 4; l++;
            const IntegrationPoint & t_ip1 = trt_n.IntPoint(t_dof[o]);
            const IntegrationPoint & s_ip1 = sl2_n.IntPoint(s_dof[o]);
            Nodes.IntPoint(o).Set3(t_ip1.x, t_ip1.y, s_ip1.x);
            o++;
         }
   }
   for (int k = 2; k < H1SegmentFE.GetDof(); k++)
   {
      for (l = 0; l < L2TriangleFE.GetDof(); l++)
      {
         t_dof[o] = l; s_dof[o] = k; dof2nk[o] = 1;
         const IntegrationPoint & t_ip = tl2_n.IntPoint(t_dof[o]);
         Nodes.IntPoint(o).Set3(t_ip.x, t_ip.y, sh1_n.IntPoint(s_dof[o]).x);
         o++;
      }
   }
}

void RT_WedgeElement::CalcVShape(const IntegrationPoint &ip,
                                 DenseMatrix &shape) const
{
#ifdef MFEM_THREAD_SAFE
   DenseMatrix trt_shape(RTTriangleFE.GetDof(), 2);
   Vector tl2_shape(L2TriangleFE.GetDof());
   Vector sh1_shape(H1SegmentFE.GetDof());
   Vector sl2_shape(L2SegmentFE.GetDof());
#endif

   IntegrationPoint ipz; ipz.x = ip.z; ipz.y = 0.0; ipz.z = 0.0;

   L2TriangleFE.CalcShape(ip, tl2_shape);
   RTTriangleFE.CalcVShape(ip, trt_shape);
   H1SegmentFE.CalcShape(ipz, sh1_shape);
   L2SegmentFE.CalcShape(ipz, sl2_shape);

   for (int i=0; i<dof; i++)
   {
      if ( dof2nk[i] >= 2 )
      {
         shape(i, 0) = trt_shape(t_dof[i], 0) * sl2_shape[s_dof[i]];
         shape(i, 1) = trt_shape(t_dof[i], 1) * sl2_shape[s_dof[i]];
         shape(i, 2) = 0.0;
      }
      else
      {
         real_t s = (dof2nk[i] == 0) ? -1.0 : 1.0;
         shape(i, 0) = 0.0;
         shape(i, 1) = 0.0;
         shape(i, 2) = s * tl2_shape[t_dof[i]] * sh1_shape(s_dof[i]);
      }
   }
}

void RT_WedgeElement::CalcDivShape(const IntegrationPoint &ip,
                                   Vector &divshape) const
{
#ifdef MFEM_THREAD_SAFE
   Vector      trt_dshape(RTTriangleFE.GetDof());
   Vector      tl2_shape(L2TriangleFE.GetDof());
   Vector      sl2_shape(L2SegmentFE.GetDof());
   DenseMatrix sh1_dshape(H1SegmentFE.GetDof(), 1);
#endif

   IntegrationPoint ipz; ipz.x = ip.z; ipz.y = 0.0; ipz.z = 0.0;

   RTTriangleFE.CalcDivShape(ip, trt_dshape);
   L2TriangleFE.CalcShape(ip, tl2_shape);

   L2SegmentFE.CalcShape(ipz, sl2_shape);
   H1SegmentFE.CalcDShape(ipz, sh1_dshape);

   for (int i=0; i<dof; i++)
   {
      if ( dof2nk[i] >= 2 )
      {
         divshape(i) = trt_dshape(t_dof[i]) * sl2_shape(s_dof[i]);
      }
      else
      {
         real_t s = (dof2nk[i] == 0) ? -1.0 : 1.0;
         divshape(i) = s * tl2_shape(t_dof[i]) * sh1_dshape(s_dof[i], 0);
      }
   }
}

const real_t RT_FuentesPyramidElement::nk[24] =
{
   0,0,-1,  0,-1,0,  1,0,1,  0,1,1,  -1,0,0,
   M_SQRT2,0,M_SQRT1_2, 0,M_SQRT2,M_SQRT1_2, 0,0,1
};

RT_FuentesPyramidElement::RT_FuentesPyramidElement(const int p)
   : VectorFiniteElement(3, Geometry::PYRAMID, (p + 1)*(3*p*(p + 2) + 5),
                         p + 1, H_DIV, FunctionSpace::Uk),
     dof2nk(dof)
{
   zmax = 0.0;

   const real_t *iop = poly1d.OpenPoints(p);
   const real_t *icp = poly1d.ClosedPoints(p + 1);
   const real_t *bop = poly1d.OpenPoints(p);

#ifndef MFEM_THREAD_SAFE
   tmp1_i.SetSize(p + 2);
   tmp1_ij.SetSize(p + 2, p + 2);
   tmp2_ij.SetSize(p + 2, dim);
   tmp3_ij.SetSize(p + 2, dim);
   tmp4_ij.SetSize(p + 1, p + 1);
   tmp1_ijk.SetSize(p + 1, p + 1, dim);
   tmp2_ijk.SetSize(p + 1, p + 1, dim);
   tmp3_ijk.SetSize(p + 1, p + 1, dim);
   tmp4_ijk.SetSize(p + 1, p + 2, dim);
   tmp5_ijk.SetSize(p + 1, p + 2, dim);
   tmp6_ijk.SetSize(p + 2, p + 2, dim);
   tmp7_ijk.SetSize(p + 2, p + 2, dim);
   u.SetSize(dof, dim);
   divu.SetSize(dof);
#else
   Vector      tmp1_i(p + 2);
   DenseMatrix tmp1_ij(p + 2, p + 2);
   DenseMatrix tmp2_ij(p + 2, dim);
   DenseMatrix tmp3_ij(p + 2, dim);
   DenseTensor tmp1_ijk(p + 1, p + 1, dim);
   DenseTensor tmp2_ijk(p + 1, p + 1, dim);
   DenseTensor tmp3_ijk(p + 1, p + 1, dim);
   DenseTensor tmp4_ijk(p + 1, p + 2, dim);
   DenseTensor tmp5_ijk(p + 1, p + 2, dim);
   DenseTensor tmp6_ijk(p + 2, p + 2, dim);
   DenseTensor tmp7_ijk(p + 2, p + 2, dim);
   DenseMatrix u(dof, dim);
#endif

   int o = 0;

   // quadrilateral face
   for (int j = 0; j <= p; j++)
      for (int i = 0; i <= p; i++)  // (3,2,1,0)
      {
         Nodes.IntPoint(o).Set3(bop[i], bop[p-j], 0.);
         dof2nk[o++] = 0;
      }
   // triangular faces
   for (int j = 0; j <= p; j++)
      for (int i = 0; i + j <= p; i++)  // (0,1,4)
      {
         real_t w = bop[i] + bop[j] + bop[p-i-j];
         Nodes.IntPoint(o).Set3(bop[i]/w, 0., bop[j]/w);
         dof2nk[o++] = 1;
      }
   for (int j = 0; j <= p; j++)
      for (int i = 0; i + j <= p; i++)  // (1,2,4)
      {
         real_t w = bop[i] + bop[j] + bop[p-i-j];
         Nodes.IntPoint(o).Set3(1.-bop[j]/w, bop[i]/w, bop[j]/w);
         dof2nk[o++] = 2;
      }
   for (int j = 0; j <= p; j++)
      for (int i = p - j; i >= 0; i--)  // (2,3,4)
      {
         real_t w = bop[i] + bop[j] + bop[p-i-j];
         Nodes.IntPoint(o).Set3(bop[i]/w, 1.0-bop[j]/w, bop[j]/w);
         dof2nk[o++] = 3;
      }
   for (int j = 0; j <= p; j++)
      for (int i = p - j; i >= 0; i--)  // (3,0,4)
      {
         real_t w = bop[i] + bop[j] + bop[p-i-j];
         Nodes.IntPoint(o).Set3(0., bop[i]/w, bop[j]/w);
         dof2nk[o++] = 4;
      }

   // interior
   // x-components
   for (int k = 0; k <= p; k++)
      for (int j = 0; j <= p; j++)
         for (int i = 1; i <= p; i++)
         {
            real_t w = 1.0 - iop[k];
            Nodes.IntPoint(o).Set3(icp[i]*w, iop[j]*w, iop[k]);
            dof2nk[o++] = 5;
         }
   // y-components
   for (int k = 0; k <= p; k++)
      for (int j = 1; j <= p; j++)
         for (int i = 0; i <= p; i++)
         {
            real_t w = 1.0 - iop[k];
            Nodes.IntPoint(o).Set3(iop[i]*w, icp[j]*w, iop[k]);
            dof2nk[o++] = 6;
         }
   // z-components
   for (int k = 1; k <= p; k++)
      for (int j = 0; j <= p; j++)
         for (int i = 0; i <= p; i++)
         {
            real_t w = 1.0 - icp[k];
            Nodes.IntPoint(o).Set3(iop[i]*w, iop[j]*w, icp[k]);
            dof2nk[o++] = 7;
         }

   DenseMatrix T(dof);

   for (int m = 0; m < dof; m++)
   {
      const IntegrationPoint &ip = Nodes.IntPoint(m);
      const Vector nm({nk[3*dof2nk[m]], nk[3*dof2nk[m]+1], nk[3*dof2nk[m]+2]});
      calcBasis(order, ip, tmp1_i, tmp1_ij, tmp2_ij,
                tmp1_ijk, tmp2_ijk, tmp3_ijk, tmp4_ijk, tmp5_ijk, tmp6_ijk,
                tmp7_ijk,
                tmp3_ij, u);
      u.Mult(nm, T.GetColumn(m));
   }

   Ti.Factor(T);
}

void RT_FuentesPyramidElement::CalcVShape(const IntegrationPoint &ip,
                                          DenseMatrix &shape) const
{
#ifdef MFEM_THREAD_SAFE
   const int p = order - 1;

   Vector      tmp1_i(p + 2);
   DenseMatrix tmp1_ij(p + 2, p + 2);
   DenseMatrix tmp2_ij(p + 2, dim);
   DenseMatrix tmp3_ij(p + 2, dim);
   DenseTensor tmp1_ijk(p + 1, p + 1, dim);
   DenseTensor tmp2_ijk(p + 1, p + 1, dim);
   DenseTensor tmp3_ijk(p + 1, p + 1, dim);
   DenseTensor tmp4_ijk(p + 1, p + 2, dim);
   DenseTensor tmp5_ijk(p + 1, p + 2, dim);
   DenseTensor tmp6_ijk(p + 2, p + 2, dim);
   DenseTensor tmp7_ijk(p + 2, p + 2, dim);
   DenseMatrix u(dof, dim);
#endif

   calcBasis(order, ip, tmp1_i, tmp1_ij, tmp2_ij,
             tmp1_ijk, tmp2_ijk, tmp3_ijk, tmp4_ijk, tmp5_ijk, tmp6_ijk,
             tmp7_ijk, tmp3_ij, u);

   Ti.Mult(u, shape);
}

void RT_FuentesPyramidElement::CalcRawVShape(const IntegrationPoint &ip,
                                             DenseMatrix &shape) const
{
#ifdef MFEM_THREAD_SAFE
   const int p = order - 1;

   Vector      tmp1_i(p + 2);
   DenseMatrix tmp1_ij(p + 2, p + 2);
   DenseMatrix tmp2_ij(p + 2, dim);
   DenseMatrix tmp3_ij(p + 2, dim);
   DenseTensor tmp1_ijk(p + 1, p + 1, dim);
   DenseTensor tmp2_ijk(p + 1, p + 1, dim);
   DenseTensor tmp3_ijk(p + 1, p + 1, dim);
   DenseTensor tmp4_ijk(p + 1, p + 2, dim);
   DenseTensor tmp5_ijk(p + 1, p + 2, dim);
   DenseTensor tmp6_ijk(p + 2, p + 2, dim);
   DenseTensor tmp7_ijk(p + 2, p + 2, dim);
#endif

   calcBasis(order, ip, tmp1_i, tmp1_ij, tmp2_ij,
             tmp1_ijk, tmp2_ijk, tmp3_ijk, tmp4_ijk, tmp5_ijk, tmp6_ijk,
             tmp7_ijk, tmp3_ij, shape);
}

void RT_FuentesPyramidElement::CalcDivShape(const IntegrationPoint &ip,
                                            Vector &divshape) const
{
#ifdef MFEM_THREAD_SAFE
   const int p = order - 1;

   Vector      tmp1_i(p + 2);
   DenseMatrix tmp1_ij(p + 2, p + 2);
   DenseMatrix tmp2_ij(p + 2, dim);
   DenseMatrix tmp3_ij(p + 2, dim);
   DenseMatrix tmp4_ij(p + 1, p + 1);
   DenseTensor tmp1_ijk(p + 1, p + 1, dim);
   DenseTensor tmp2_ijk(p + 1, p + 1, dim);
   DenseTensor tmp3_ijk(p + 1, p + 1, dim);
   DenseTensor tmp4_ijk(p + 1, p + 2, dim);
   DenseTensor tmp5_ijk(p + 1, p + 2, dim);
   DenseTensor tmp6_ijk(p + 2, p + 2, dim);
   DenseTensor tmp7_ijk(p + 2, p + 2, dim);
   Vector divu(dof);
#endif
   divu = 0.0;

   calcDivBasis(order, ip, tmp1_i, tmp1_ij, tmp2_ij,
                tmp1_ijk, tmp2_ijk, tmp3_ijk, tmp4_ij, tmp4_ijk, tmp5_ijk,
                tmp6_ijk, tmp7_ijk, tmp3_ij, divu);

   Ti.Mult(divu, divshape);
}

void RT_FuentesPyramidElement::CalcRawDivShape(const IntegrationPoint &ip,
                                               Vector &dshape) const
{
#ifdef MFEM_THREAD_SAFE
   const int p = order - 1;

   Vector      tmp1_i(p + 2);
   DenseMatrix tmp1_ij(p + 2, p + 2);
   DenseMatrix tmp2_ij(p + 2, dim);
   DenseMatrix tmp3_ij(p + 2, dim);
   DenseMatrix tmp4_ij(p + 1, p + 1);
   DenseTensor tmp1_ijk(p + 1, p + 1, dim);
   DenseTensor tmp2_ijk(p + 1, p + 1, dim);
   DenseTensor tmp3_ijk(p + 1, p + 1, dim);
   DenseTensor tmp4_ijk(p + 1, p + 2, dim);
   DenseTensor tmp5_ijk(p + 1, p + 2, dim);
   DenseTensor tmp6_ijk(p + 2, p + 2, dim);
   DenseTensor tmp7_ijk(p + 2, p + 2, dim);
#endif

   calcDivBasis(order, ip, tmp1_i, tmp1_ij, tmp2_ij,
                tmp1_ijk, tmp2_ijk, tmp3_ijk, tmp4_ij, tmp4_ijk, tmp5_ijk,
                tmp6_ijk, tmp7_ijk, tmp3_ij, dshape);
}

void RT_FuentesPyramidElement::calcBasis(const int p,
                                         const IntegrationPoint &ip,
                                         Vector &phi_k,
                                         DenseMatrix &phi_ij,
                                         DenseMatrix &dphi_k,
                                         DenseTensor &VQ_ijk,
                                         DenseTensor &VT_ijk,
                                         DenseTensor &VTT_ijk,
                                         DenseTensor &E_ijk,
                                         DenseTensor &dE_ijk,
                                         DenseTensor &dphi_ijk,
                                         DenseTensor &VL_ijk,
                                         DenseMatrix &VR_ij,
                                         DenseMatrix &F) const
{
   real_t x = ip.x;
   real_t y = ip.y;
   real_t z = ip.z;
   Vector xy({x,y});
   real_t mu;

   if (std::fabs(1.0 - z) < apex_tol)
   {
      z = 1.0 - apex_tol;
      y = 0.5 * (1.0 - z);
      x = 0.5 * (1.0 - z);
      xy(0) = x; xy(1) = y;
   }
   zmax = std::max(z, zmax);

   F = 0.0;

   int o = 0;

   // Quadrilateral face
   if (z < 1.0)
   {
      V_Q(p, mu01(z, xy, 1), mu01_grad_mu01(z, xy, 1),
          mu01(z, xy, 2), mu01_grad_mu01(z, xy, 2),
          VQ_ijk);

      const real_t muz3 = pow(mu0(z), 3);

      for (int j=0; j<p; j++)
         for (int i=0; i<p; i++, o++)
            for (int k=0; k<3; k++)
            {
               F(o, k) = muz3 * VQ_ijk(i, j, k);
            }
   }

   // Triangular faces
   if (z < 1.0)
   {
      Vector dmuz;

      // (a,b) = (1,2), c = 0
      V_T(p, nu012(z, xy, 1), nu012_grad_nu012(z, xy, 1), VT_ijk);
      mu = mu0(z, xy, 2);
      dmuz.Destroy(); dmuz = grad_mu0(z, xy, 2);
      VT_T(p, nu012(z, xy, 1), nu01_grad_nu01(z, xy, 1),
           nu012_grad_nu012(z, xy, 1), mu, dmuz, VTT_ijk);
      for (int j=0; j<p; j++)
         for (int i=0; i+j<p; i++, o++)
            for (int k=0; k<3; k++)
            {
               F(o, k) = 0.5 * (mu * VT_ijk(i, j, k) + VTT_ijk(i, j, k));
            }

      // (a,b) = (1,2), c = 1
      mu = mu1(z, xy, 2);
      dmuz.Destroy(); dmuz = grad_mu1(z, xy, 2);
      VT_T(p, nu012(z, xy, 1), nu01_grad_nu01(z, xy, 1),
           nu012_grad_nu012(z, xy, 1), mu, dmuz, VTT_ijk);
      for (int j=0; j<p; j++)
         for (int i=0; i+j<p; i++, o++)
            for (int k=0; k<3; k++)
            {
               F(o, k) = 0.5 * (mu * VT_ijk(i, j, k) + VTT_ijk(i, j, k));
            }

      // (a,b) = (2,1), c = 0
      V_T(p, nu012(z, xy, 2), nu012_grad_nu012(z, xy, 2), VT_ijk);
      mu = mu0(z, xy, 1);
      dmuz.Destroy(); dmuz = grad_mu0(z, xy, 1);
      VT_T(p, nu012(z, xy, 2), nu01_grad_nu01(z, xy, 2),
           nu012_grad_nu012(z, xy, 2), mu, dmuz, VTT_ijk);
      for (int j=0; j<p; j++)
         for (int i=0; i+j<p; i++, o++)
            for (int k=0; k<3; k++)
            {
               F(o, k) = 0.5 * (mu * VT_ijk(i, j, k) + VTT_ijk(i, j, k));
            }

      // (a,b) = (2,1), c = 1
      mu = mu1(z, xy, 1);
      dmuz.Destroy(); dmuz = grad_mu1(z, xy, 1);
      VT_T(p, nu012(z, xy, 2), nu01_grad_nu01(z, xy, 2),
           nu012_grad_nu012(z, xy, 2), mu, dmuz, VTT_ijk);
      for (int j=0; j<p; j++)
         for (int i=0; i+j<p; i++, o++)
            for (int k=0; k<3; k++)
            {
               F(o, k) = 0.5 * (mu * VT_ijk(i, j, k) + VTT_ijk(i, j, k));
            }
   }

   // Interior
   // Family I
   if (z < 1.0 && p >= 2)
   {
      E_Q(p, mu01(z, xy, 1), grad_mu01(z, xy, 1),
          mu01(z, xy, 2), grad_mu01(z, xy, 2), E_ijk, dE_ijk);
      phi_E(p, mu01(z), grad_mu01(z), phi_k, dphi_k);
      const real_t muz = mu0(z);
      const Vector dmuz(grad_mu0(z));

      Vector dmuphi(3), E(3), v(3);

      for (int k=2; k<=p; k++)
      {
         dmuphi(0) = muz * dphi_k(k,0) + dmuz(0) * phi_k(k);
         dmuphi(1) = muz * dphi_k(k,1) + dmuz(1) * phi_k(k);
         dmuphi(2) = muz * dphi_k(k,2) + dmuz(2) * phi_k(k);
         for (int j=2; j<=p; j++)
            for (int i=0; i<p; i++, o++)
            {
               E(0) = E_ijk(i,j,0); E(1) = E_ijk(i,j,1); E(2) = E_ijk(i,j,2);
               dmuphi.cross3D(E, v);
               for (int l=0; l<3; l++)
               {
                  F(o, l) = muz * phi_k(k) * dE_ijk(i,j,l) + v(l);
               }
            }
      }
   }

   // Family II
   if (z < 1.0 && p >= 2)
   {
      E_Q(p, mu01(z, xy, 2), grad_mu01(z, xy, 2),
          mu01(z, xy, 1), grad_mu01(z, xy, 1), E_ijk, dE_ijk);
      // Re-using phi_E from Family I
      const real_t muz = mu0(z);
      const Vector dmuz(grad_mu0(z));

      Vector dmuphi(3), E(3), v(3);

      for (int k=2; k<=p; k++)
      {
         dmuphi(0) = muz * dphi_k(k,0) + dmuz(0) * phi_k(k);
         dmuphi(1) = muz * dphi_k(k,1) + dmuz(1) * phi_k(k);
         dmuphi(2) = muz * dphi_k(k,2) + dmuz(2) * phi_k(k);
         for (int j=2; j<=p; j++)
            for (int i=0; i<p; i++, o++)
            {
               E(0) = E_ijk(i,j,0); E(1) = E_ijk(i,j,1); E(2) = E_ijk(i,j,2);
               dmuphi.cross3D(E, v);
               for (int l=0; l<3; l++)
               {
                  F(o, l) = muz * phi_k(k) * dE_ijk(i,j,l) + v(l);
               }
            }
      }
   }
   // Family III
   if (z < 1.0 && p >= 2)
   {
      phi_Q(p, mu01(z, xy, 2), grad_mu01(z, xy, 2),
            mu01(z, xy, 1), grad_mu01(z, xy, 1), phi_ij, dphi_ijk);
      const real_t muz = mu0(z);
      const Vector dmuz(grad_mu0(z));

      for (int j=2; j<=p; j++)
         for (int i=2; i<=p; i++, o++)
         {
            const int n = std::max(i,j);
            const real_t nmu = n * pow(muz, n-1);
            F(o, 0) = nmu * (dphi_ijk(i,j,1) * dmuz(2) -
                             dphi_ijk(i,j,2) * dmuz(1));
            F(o, 1) = nmu * (dphi_ijk(i,j,2) * dmuz(0) -
                             dphi_ijk(i,j,0) * dmuz(2));
            F(o, 2) = nmu * (dphi_ijk(i,j,0) * dmuz(1) -
                             dphi_ijk(i,j,1) * dmuz(0));
         }
   }
   // Family IV
   if (z < 1.0 && p >= 2)
   {
      // Re-using V_Q from Quadrilateral Face
      phi_E(p, mu01(z), phi_k);

      const real_t muz2 = pow(mu0(z), 2);

      for (int k=2; k<=p; k++)
         for (int j=0; j<p; j++)
            for (int i=0; i<p; i++, o++)
               for (int l=0; l<3; l++)
               {
                  F(o, l) = muz2 * VQ_ijk(i, j, l) * phi_k(k);
               }

   }
   // Family V
   if (z < 1.0 && p >= 2)
   {
      V_L(p, mu01(z, xy, 1), grad_mu01(z, xy, 1),
          mu01(z, xy, 2), grad_mu01(z, xy, 2), mu0(z), grad_mu0(z), VL_ijk);

      const real_t muz = mu1(z);

      for (int j=2; j<=p; j++)
         for (int i=2; i<=p; i++, o++)
         {
            const int n = std::max(i, j);
            const real_t muzi = pow(muz, n-1);
            for (int l=0; l<3; l++)
            {
               F(o, l) = muzi * VL_ijk(i, j, l);
            }
         }
   }
   // Family VI
   if (z < 1.0 && p >= 2)
   {
      V_R(p, mu01(z, xy, 1), grad_mu01(z, xy, 1),
          mu1(z, xy, 2), grad_mu1(z, xy, 2), mu0(z), grad_mu0(z), VR_ij);

      const real_t muz = mu1(z);

      for (int i=2; i<=p; i++, o++)
      {
         const real_t muzi = pow(muz, i-1);
         for (int l=0; l<3; l++)
         {
            F(o, l) = muzi * VR_ij(i, l);
         }
      }
   }
   // Family VII
   if (z < 1.0 && p >= 2)
   {
      V_R(p, mu01(z, xy, 2), grad_mu01(z, xy, 2),
          mu1(z, xy, 1), grad_mu1(z,xy,1), mu0(z), grad_mu0(z), VR_ij);

      const real_t muz = mu1(z);

      for (int i=2; i<=p; i++, o++)
      {
         const real_t muzi = pow(muz, i-1);
         for (int l=0; l<3; l++)
         {
            F(o, l) = muzi * VR_ij(i, l);
         }
      }
   }
}

void RT_FuentesPyramidElement::calcDivBasis(const int p,
                                            const IntegrationPoint &ip,
                                            Vector &phi_k,
                                            DenseMatrix &phi_ij,
                                            DenseMatrix &dphi_k,
                                            DenseTensor &VQ_ijk,
                                            DenseTensor &VT_ijk,
                                            DenseTensor &VTT_ijk,
                                            DenseMatrix &dVTT_ij,
                                            DenseTensor &E_ijk,
                                            DenseTensor &dE_ijk,
                                            DenseTensor &dphi_ijk,
                                            DenseTensor &VL_ijk,
                                            DenseMatrix &VR_ij,
                                            Vector &dF) const
{
   real_t x = ip.x;
   real_t y = ip.y;
   real_t z = ip.z;
   Vector xy({x,y});
   real_t mu;

   bool limz1 = false;
   if (std::fabs(1.0 - z) < apex_tol)
   {
      limz1 = true;
      z = 1.0 - apex_tol;
      y = 0.5 * (1.0 - z);
      x = 0.5 * (1.0 - z);
      xy(0) = x; xy(1) = y;
   }
   zmax = std::max(z, zmax);

   dF = 0.0;

   int o = 0;

   // Quadrilateral face
   {
      V_Q(p, mu01(z, xy, 1), mu01_grad_mu01(z, xy, 1),
          mu01(z, xy, 2), mu01_grad_mu01(z, xy, 2),
          VQ_ijk);

      const real_t muz2 = pow(mu0(z), 2);
      const Vector dmuz = grad_mu0(z);

      const int o0 = o;
      for (int j=0; j<p; j++)
         for (int i=0; i<p; i++, o++)
            for (int k=0; k<3; k++)
            {
               dF(o) += 3.0 * muz2 * dmuz(k) * VQ_ijk(i, j, k);
            }

      // Overwrite lowest order quadrilateral face DoF with known limiting
      // value
      if (limz1)
      {
         dF(o0) = -3.0;
      }
   }

   // Triangular faces
   {
      Vector dmuz;

      // (a,b) = (1,2), c = 0
      V_T(p, nu012(z, xy, 1), nu012_grad_nu012(z, xy, 1), VT_ijk);
      mu = mu0(z, xy, 2);
      dmuz.Destroy(); dmuz = grad_mu0(z, xy, 2);
      VT_T(p, nu012(z, xy, 1), nu01_grad_nu01(z, xy, 1),
           nu012_grad_nu012(z, xy, 1), grad_nu2(z, xy, 1), mu, dmuz,
           VTT_ijk, dVTT_ij);
      const int o1 = o;
      for (int j=0; j<p; j++)
         for (int i=0; i+j<p; i++, o++)
         {
            dF(o) = 0.5 * dVTT_ij(i, j);
            for (int k=0; k<3; k++)
            {
               dF(o) += 0.5 * dmuz(k) * VT_ijk(i, j, k);
            }
         }

      // (a,b) = (1,2), c = 1
      mu = mu1(z, xy, 2);
      dmuz.Destroy(); dmuz = grad_mu1(z, xy, 2);
      VT_T(p, nu012(z, xy, 1), nu01_grad_nu01(z, xy, 1),
           nu012_grad_nu012(z, xy, 1), grad_nu2(z, xy, 1), mu, dmuz,
           VTT_ijk, dVTT_ij);
      const int o2 = o;
      for (int j=0; j<p; j++)
         for (int i=0; i+j<p; i++, o++)
         {
            dF(o) = 0.5 * dVTT_ij(i, j);
            for (int k=0; k<3; k++)
            {
               dF(o) += 0.5 * dmuz(k) * VT_ijk(i, j, k);
            }
         }

      // (a,b) = (2,1), c = 0
      V_T(p, nu012(z, xy, 2), nu012_grad_nu012(z, xy, 2), VT_ijk);
      mu = mu0(z, xy, 1);
      dmuz.Destroy(); dmuz = grad_mu0(z, xy, 1);
      VT_T(p, nu012(z, xy, 2), nu01_grad_nu01(z, xy, 2),
           nu012_grad_nu012(z, xy, 2), grad_nu2(z, xy, 2), mu, dmuz,
           VTT_ijk, dVTT_ij);
      const int o3 = o;
      for (int j=0; j<p; j++)
         for (int i=0; i+j<p; i++, o++)
         {
            dF(o) = 0.5 * dVTT_ij(i, j);
            for (int k=0; k<3; k++)
            {
               dF(o) += 0.5 * dmuz(k) * VT_ijk(i, j, k);
            }
         }

      // (a,b) = (2,1), c = 1
      mu = mu1(z, xy, 1);
      dmuz.Destroy(); dmuz = grad_mu1(z, xy, 1);
      VT_T(p, nu012(z, xy, 2), nu01_grad_nu01(z, xy, 2),
           nu012_grad_nu012(z, xy, 2), grad_nu2(z, xy, 2), mu, dmuz,
           VTT_ijk, dVTT_ij);
      const int o4 = o;
      for (int j=0; j<p; j++)
         for (int i=0; i+j<p; i++, o++)
         {
            dF(o) = 0.5 * dVTT_ij(i, j);
            for (int k=0; k<3; k++)
            {
               dF(o) += 0.5 * dmuz(k) * VT_ijk(i, j, k);
            }
         }

      // Overwrite lowest order triangular face DoFs with known limiting values
      if (limz1)
      {
         dF(o1) =  1.5;
         dF(o2) = -1.5;
         dF(o3) = -1.5;
         dF(o4) =  1.5;
      }
   }

   // Interior
   // Family I
   if (p >= 2)
   {
      // Divergence is zero so skip ahead
      o += (p-1) * (p-1) * p;
   }

   // Family II
   if (p >= 2)
   {
      // Divergence is zero so skip ahead
      o += (p-1) * (p-1) * p;
   }
   // Family III
   if (p >= 2)
   {
      // Divergence is zero so skip ahead
      o += (p-1) * (p-1);
   }
   // Family IV
   if (p >= 2)
   {
      // Re-using V_Q from Quadrilateral Face
      phi_E(p, mu01(z), grad_mu01(z), phi_k, dphi_k);

      const real_t muz2 = pow(mu0(z), 2);
      const Vector dmuz = grad_mu0(z);

      for (int k=2; k<=p; k++)
         for (int j=0; j<p; j++)
            for (int i=0; i<p; i++, o++)
               for (int l=0; l<3; l++)
               {
                  dF(o) += (muz2 * dphi_k(k, l) +
                            2.0 * mu0(z) * phi_k(k) * dmuz(l)) * VQ_ijk(i, j, l);
               }
   }
   // Family V
   if (p >= 2)
   {
      V_L(p, mu01(z, xy, 1), grad_mu01(z, xy, 1),
          mu01(z, xy, 2), grad_mu01(z, xy, 2), mu0(z), grad_mu0(z), VL_ijk);

      const real_t muz = mu1(z);
      const Vector dmuz = grad_mu1(z);

      for (int j=2; j<=p; j++)
         for (int i=2; i<=p; i++, o++)
         {
            const int n = std::max(i, j);
            const real_t muzi = pow(muz, n-2);
            for (int l=0; l<3; l++)
            {
               dF(o) += (n-1) * muzi * dmuz(l) * VL_ijk(i, j, l);
            }
         }
   }
   // Family VI
   if (p >= 2)
   {
      V_R(p, mu01(z, xy, 1), grad_mu01(z, xy, 1),
          mu1(z, xy, 2), grad_mu1(z, xy, 2), mu0(z), grad_mu0(z), VR_ij);

      const real_t muz = mu1(z);
      const Vector dmuz = grad_mu1(z);

      for (int i=2; i<=p; i++, o++)
      {
         const real_t muzi = pow(muz, i-2);
         for (int l=0; l<3; l++)
         {
            dF(o) += (i-1) * muzi * dmuz(l) * VR_ij(i, l);
         }
      }
   }
   // Family VII
   if (p >= 2)
   {
      V_R(p, mu01(z, xy, 2), grad_mu01(z, xy, 2),
          mu1(z, xy, 1), grad_mu1(z,xy,1), mu0(z), grad_mu0(z), VR_ij);

      const real_t muz = mu1(z);
      const Vector dmuz = grad_mu1(z);

      for (int i=2; i<=p; i++, o++)
      {
         const real_t muzi = pow(muz, i-2);
         for (int l=0; l<3; l++)
         {
            dF(o) += (i-1) * muzi * dmuz(l) * VR_ij(i, l);
         }
      }
   }
}

const real_t RT_R1D_SegmentElement::nk[9] = { 1.,0.,0., 0.,1.,0., 0.,0.,1. };

RT_R1D_SegmentElement::RT_R1D_SegmentElement(const int p,
                                             const int cb_type,
                                             const int ob_type)
   : VectorFiniteElement(1, Geometry::SEGMENT, 3 * p + 4, p + 1,
                         H_DIV, FunctionSpace::Pk),
     dof2nk(dof),
     cbasis1d(poly1d.GetBasis(p + 1, VerifyClosed(cb_type))),
     obasis1d(poly1d.GetBasis(p, VerifyOpen(ob_type)))
{
   // Override default dimension for VectorFiniteElements
   vdim = 3;

   const real_t *cp = poly1d.ClosedPoints(p + 1, cb_type);
   const real_t *op = poly1d.OpenPoints(p, ob_type);

#ifndef MFEM_THREAD_SAFE
   shape_cx.SetSize(p + 2);
   shape_ox.SetSize(p + 1);
   dshape_cx.SetSize(p + 2);
#endif

   dof_map.SetSize(dof);

   int o = 0;
   // nodes
   // (0)
   Nodes.IntPoint(o).x = cp[0]; // x-directed
   dof_map[0] = o; dof2nk[o++] = 0;

   // (1)
   Nodes.IntPoint(o).x = cp[p+1]; // x-directed
   dof_map[p+1] = o; dof2nk[o++] = 0;

   // interior
   // x-components
   for (int i = 1; i <= p; i++)
   {
      Nodes.IntPoint(o).x = cp[i];
      dof_map[i] = o; dof2nk[o++] = 0;
   }
   // y-components
   for (int i = 0; i <= p; i++)
   {
      Nodes.IntPoint(o).x = op[i];
      dof_map[p+i+2] = o; dof2nk[o++] = 1;
   }
   // z-components
   for (int i = 0; i <= p; i++)
   {
      Nodes.IntPoint(o).x = op[i];
      dof_map[2*p+3+i] = o; dof2nk[o++] = 2;
   }
}

void RT_R1D_SegmentElement::CalcVShape(const IntegrationPoint &ip,
                                       DenseMatrix &shape) const
{
   const int p = order;

#ifdef MFEM_THREAD_SAFE
   Vector shape_cx(p + 1), shape_ox(p);
#endif

   cbasis1d.Eval(ip.x, shape_cx);
   obasis1d.Eval(ip.x, shape_ox);

   int o = 0;
   // x-components
   for (int i = 0; i <= p; i++)
   {
      int idx = dof_map[o++];
      shape(idx,0) = shape_cx(i);
      shape(idx,1) = 0.;
      shape(idx,2) = 0.;
   }
   // y-components
   for (int i = 0; i < p; i++)
   {
      int idx = dof_map[o++];
      shape(idx,0) = 0.;
      shape(idx,1) = shape_ox(i);
      shape(idx,2) = 0.;
   }
   // z-components
   for (int i = 0; i < p; i++)
   {
      int idx = dof_map[o++];
      shape(idx,0) = 0.;
      shape(idx,1) = 0.;
      shape(idx,2) = shape_ox(i);
   }
}

void RT_R1D_SegmentElement::CalcVShape(ElementTransformation &Trans,
                                       DenseMatrix &shape) const
{
   CalcVShape(Trans.GetIntPoint(), shape);
   const DenseMatrix & J = Trans.Jacobian();
   MFEM_ASSERT(J.Width() == 1 && J.Height() == 1,
               "RT_R1D_SegmentElement cannot be embedded in "
               "2 or 3 dimensional spaces");
   for (int i=0; i<dof; i++)
   {
      shape(i, 0) *= J(0,0);
   }
   shape *= (1.0 / Trans.Weight());
}

void RT_R1D_SegmentElement::CalcDivShape(const IntegrationPoint &ip,
                                         Vector &divshape) const
{
   const int p = order;

#ifdef MFEM_THREAD_SAFE
   Vector shape_cx(p + 1);
   Vector dshape_cx(p + 1);
#endif

   cbasis1d.Eval(ip.x, shape_cx, dshape_cx);

   int o = 0;
   // x-components
   for (int i = 0; i <= p; i++)
   {
      int idx = dof_map[o++];
      divshape(idx) = dshape_cx(i);
   }
   // y-components
   for (int i = 0; i < p; i++)
   {
      int idx = dof_map[o++];
      divshape(idx) = 0.;
   }
   // z-components
   for (int i = 0; i < p; i++)
   {
      int idx = dof_map[o++];
      divshape(idx) = 0.;
   }
}

void RT_R1D_SegmentElement::Project(VectorCoefficient &vc,
                                    ElementTransformation &Trans,
                                    Vector &dofs) const
{
   real_t data[3];
   Vector vk1(data, 1);
   Vector vk3(data, 3);

   real_t * nk_ptr = const_cast<real_t*>(nk);

   for (int k = 0; k < dof; k++)
   {
      Trans.SetIntPoint(&Nodes.IntPoint(k));

      vc.Eval(vk3, Trans, Nodes.IntPoint(k));
      // dof_k = nk^t adj(J) vk
      Vector n1(&nk_ptr[dof2nk[k] * 3], 1);
      Vector n3(&nk_ptr[dof2nk[k] * 3], 3);

      dofs(k) = Trans.AdjugateJacobian().InnerProduct(vk1, n1) +
                Trans.Weight() * vk3(1) * n3(1) +
                Trans.Weight() * vk3(2) * n3(2);
   }
}

void RT_R1D_SegmentElement::Project(const FiniteElement &fe,
                                    ElementTransformation &Trans,
                                    DenseMatrix &I) const
{
   if (fe.GetRangeType() == SCALAR)
   {
      real_t vk[Geometry::MaxDim];
      Vector shape(fe.GetDof());

      real_t * nk_ptr = const_cast<real_t*>(nk);

      I.SetSize(dof, vdim*fe.GetDof());
      for (int k = 0; k < dof; k++)
      {
         const IntegrationPoint &ip = Nodes.IntPoint(k);

         Vector n1(&nk_ptr[dof2nk[k] * 3], 1);
         Vector n3(&nk_ptr[dof2nk[k] * 3], 3);

         fe.CalcShape(ip, shape);
         Trans.SetIntPoint(&ip);
         // Transform RT face normals from reference to physical space
         // vk = adj(J)^T nk
         Trans.AdjugateJacobian().MultTranspose(n1, vk);
         vk[1] = n3[1] * Trans.Weight();
         vk[2] = n3[2] * Trans.Weight();
         if (fe.GetMapType() == INTEGRAL)
         {
            real_t w = 1.0/Trans.Weight();
            for (int d = 0; d < 1; d++)
            {
               vk[d] *= w;
            }
         }

         for (int j = 0; j < shape.Size(); j++)
         {
            real_t s = shape(j);
            if (fabs(s) < 1e-12)
            {
               s = 0.0;
            }
            // Project scalar basis function multiplied by each coordinate
            // direction onto the transformed face normals
            for (int d = 0; d < vdim; d++)
            {
               I(k,j+d*shape.Size()) = s*vk[d];
            }
         }
      }
   }
   else
   {
      real_t vk[Geometry::MaxDim];
      DenseMatrix vshape(fe.GetDof(), fe.GetRangeDim());

      real_t * nk_ptr = const_cast<real_t*>(nk);

      I.SetSize(dof, fe.GetDof());
      for (int k = 0; k < dof; k++)
      {
         const IntegrationPoint &ip = Nodes.IntPoint(k);

         Vector n1(&nk_ptr[dof2nk[k] * 3], 1);
         Vector n3(&nk_ptr[dof2nk[k] * 3], 3);

         Trans.SetIntPoint(&ip);
         // Transform RT face normals from reference to physical space
         // vk = adj(J)^T nk
         Trans.AdjugateJacobian().MultTranspose(n1, vk);
         // Compute fe basis functions in physical space
         fe.CalcVShape(Trans, vshape);
         // Project fe basis functions onto transformed face normals
         for (int j=0; j<vshape.Height(); j++)
         {
            I(k, j) = 0.0;
            I(k, j) += vshape(j, 0) * vk[0];
            if (vshape.Width() == 3)
            {
               I(k, j) += Trans.Weight() * vshape(j, 1) * n3(1);
               I(k, j) += Trans.Weight() * vshape(j, 2) * n3(2);
            }
         }
      }
   }
}

void RT_R1D_SegmentElement::ProjectCurl(const FiniteElement &fe,
                                        ElementTransformation &Trans,
                                        DenseMatrix &curl) const
{
   DenseMatrix curl_shape(fe.GetDof(), fe.GetRangeDim());
   Vector curl_k(fe.GetDof());

   real_t * nk_ptr = const_cast<real_t*>(nk);

   curl.SetSize(dof, fe.GetDof());
   for (int k = 0; k < dof; k++)
   {
      fe.CalcCurlShape(Nodes.IntPoint(k), curl_shape);
      curl_shape.Mult(nk_ptr + dof2nk[k] * 3, curl_k);
      for (int j = 0; j < curl_k.Size(); j++)
      {
         curl(k,j) = (fabs(curl_k(j)) < 1e-12) ? 0.0 : curl_k(j);
      }
   }
}

const real_t RT_R2D_SegmentElement::nk[2] = { 0.,1.};

RT_R2D_SegmentElement::RT_R2D_SegmentElement(const int p,
                                             const int ob_type)
   : VectorFiniteElement(1, Geometry::SEGMENT, p + 1, p + 1,
                         H_DIV, FunctionSpace::Pk),
     dof2nk(dof),
     obasis1d(poly1d.GetBasis(p, VerifyOpen(ob_type)))
{
   // Override default dimension for VectorFiniteElements
   vdim = 2;

   const real_t *op = poly1d.OpenPoints(p, ob_type);

#ifndef MFEM_THREAD_SAFE
   shape_ox.SetSize(p+1);
#endif

   dof_map.SetSize(dof);

   int o = 0;
   // interior
   // z-components
   for (int i = 0; i <= p; i++)
   {
      Nodes.IntPoint(o).x = op[i];
      dof_map[i] = o; dof2nk[o++] = 0;
   }
}

void RT_R2D_SegmentElement::CalcVShape(const IntegrationPoint &ip,
                                       DenseMatrix &shape) const
{
   const int p = order;

#ifdef MFEM_THREAD_SAFE
   Vector shape_ox(p);
#endif

   obasis1d.Eval(ip.x, shape_ox);

   int o = 0;
   // z-components
   for (int i = 0; i <= p; i++)
   {
      int idx = dof_map[o++];
      shape(idx,0) = shape_ox(i);
      shape(idx,1) = 0.;
   }
}

void RT_R2D_SegmentElement::CalcVShape(ElementTransformation &Trans,
                                       DenseMatrix &shape) const
{
   CalcVShape(Trans.GetIntPoint(), shape);
   const DenseMatrix & J = Trans.Jacobian();
   MFEM_ASSERT(J.Width() == 1 && J.Height() == 1,
               "RT_R2D_SegmentElement cannot be embedded in "
               "2 or 3 dimensional spaces");
   for (int i=0; i<dof; i++)
   {
      shape(i, 0) *= J(0,0);
   }
   shape *= (1.0 / Trans.Weight());
}

void RT_R2D_SegmentElement::CalcDivShape(const IntegrationPoint &ip,
                                         Vector &div_shape) const
{
   div_shape = 0.0;
}

void RT_R2D_SegmentElement::LocalInterpolation(const VectorFiniteElement &cfe,
                                               ElementTransformation &Trans,
                                               DenseMatrix &I) const
{
   real_t vk[Geometry::MaxDim]; vk[1] = 0.0; vk[2] = 0.0;
   Vector xk(vk, dim);
   IntegrationPoint ip;
   DenseMatrix vshape(cfe.GetDof(), vdim);

   real_t * nk_ptr = const_cast<real_t*>(nk);

   I.SetSize(dof, vshape.Height());

   // assuming Trans is linear; this should be ok for all refinement types
   Trans.SetIntPoint(&Geometries.GetCenter(geom_type));
   const DenseMatrix &adjJ = Trans.AdjugateJacobian();
   for (int k = 0; k < dof; k++)
   {
      Vector n2(&nk_ptr[dof2nk[k] * 2], 2);

      Trans.Transform(Nodes.IntPoint(k), xk);
      ip.Set3(vk);
      cfe.CalcVShape(ip, vshape);
      // xk = |J| J^{-t} n_k
      adjJ.MultTranspose(n2, vk);
      // I_k = vshape_k.adj(J)^t.n_k, k=1,...,dof
      for (int j = 0; j < vshape.Height(); j++)
      {
         real_t Ikj = 0.;
         /*
              for (int i = 0; i < dim; i++)
              {
                 Ikj += vshape(j, i) * vk[i];
              }
         */
         Ikj += Trans.Weight() * vshape(j, 1) * n2(1);
         I(k, j) = (fabs(Ikj) < 1e-12) ? 0.0 : Ikj;
      }
   }
}

RT_R2D_FiniteElement::RT_R2D_FiniteElement(int p, Geometry::Type G, int Do,
                                           const real_t *nk_fe)
   : VectorFiniteElement(2, G, Do, p + 1,
                         H_DIV, FunctionSpace::Pk),
     nk(nk_fe),
     dof_map(dof),
     dof2nk(dof)
{
   // Override default dimension for VectorFiniteElements
   vdim = 3;
}

void RT_R2D_FiniteElement::CalcVShape(ElementTransformation &Trans,
                                      DenseMatrix &shape) const
{
   CalcVShape(Trans.GetIntPoint(), shape);
   const DenseMatrix & J = Trans.Jacobian();
   MFEM_ASSERT(J.Width() == 2 && J.Height() == 2,
               "RT_R2D_FiniteElement cannot be embedded in "
               "3 dimensional spaces");
   for (int i=0; i<dof; i++)
   {
      real_t sx = shape(i, 0);
      real_t sy = shape(i, 1);
      shape(i, 0) = sx * J(0, 0) + sy * J(0, 1);
      shape(i, 1) = sx * J(1, 0) + sy * J(1, 1);
   }
   shape *= (1.0 / Trans.Weight());
}

void
RT_R2D_FiniteElement::LocalInterpolation(const VectorFiniteElement &cfe,
                                         ElementTransformation &Trans,
                                         DenseMatrix &I) const
{
   real_t vk[Geometry::MaxDim]; vk[2] = 0.0;
   Vector xk(vk, dim);
   IntegrationPoint ip;
   DenseMatrix vshape(cfe.GetDof(), vdim);

   real_t * nk_ptr = const_cast<real_t*>(nk);

   I.SetSize(dof, vshape.Height());

   // assuming Trans is linear; this should be ok for all refinement types
   Trans.SetIntPoint(&Geometries.GetCenter(geom_type));
   const DenseMatrix &adjJ = Trans.AdjugateJacobian();
   for (int k = 0; k < dof; k++)
   {
      Vector n2(&nk_ptr[dof2nk[k] * 3], 2);
      Vector n3(&nk_ptr[dof2nk[k] * 3], 3);

      Trans.Transform(Nodes.IntPoint(k), xk);
      ip.Set3(vk);
      cfe.CalcVShape(ip, vshape);
      // xk = |J| J^{-t} n_k
      adjJ.MultTranspose(n2, vk);
      // I_k = vshape_k.adj(J)^t.n_k, k=1,...,dof
      for (int j = 0; j < vshape.Height(); j++)
      {
         real_t Ikj = 0.;
         for (int i = 0; i < dim; i++)
         {
            Ikj += vshape(j, i) * vk[i];
         }
         Ikj += Trans.Weight() * vshape(j, 2) * n3(2);
         I(k, j) = (fabs(Ikj) < 1e-12) ? 0.0 : Ikj;
      }
   }
}

void RT_R2D_FiniteElement::GetLocalRestriction(ElementTransformation &Trans,
                                               DenseMatrix &R) const
{
   real_t pt_data[Geometry::MaxDim];
   IntegrationPoint ip;
   Vector pt(pt_data, dim);

#ifdef MFEM_THREAD_SAFE
   DenseMatrix vshape(dof, vdim);
#endif

   real_t * nk_ptr = const_cast<real_t*>(nk);

   Trans.SetIntPoint(&Geometries.GetCenter(geom_type));
   const DenseMatrix &J = Trans.Jacobian();
   const real_t weight = Trans.Weight();
   for (int j = 0; j < dof; j++)
   {
      Vector n2(&nk_ptr[dof2nk[j] * 3], 2);
      Vector n3(&nk_ptr[dof2nk[j] * 3], 3);

      InvertLinearTrans(Trans, Nodes.IntPoint(j), pt);
      ip.Set(pt_data, dim);
      if (Geometries.CheckPoint(geom_type, ip)) // do we need an epsilon here?
      {
         CalcVShape(ip, vshape);
         J.MultTranspose(n2, pt_data);
         pt /= weight;
         for (int k = 0; k < dof; k++)
         {
            real_t R_jk = 0.0;
            for (int d = 0; d < dim; d++)
            {
               R_jk += vshape(k,d)*pt_data[d];
            }
            R_jk += vshape(k,2) * n3(2);
            R(j,k) = R_jk;
         }
      }
      else
      {
         // Set the whole row to avoid valgrind warnings in R.Threshold().
         R.SetRow(j, infinity());
      }
   }
   R.Threshold(1e-12);
}

void RT_R2D_FiniteElement::Project(VectorCoefficient &vc,
                                   ElementTransformation &Trans,
                                   Vector &dofs) const
{
   real_t data[3];
   Vector vk2(data, 2);
   Vector vk3(data, 3);

   real_t * nk_ptr = const_cast<real_t*>(nk);

   for (int k = 0; k < dof; k++)
   {
      Trans.SetIntPoint(&Nodes.IntPoint(k));

      vc.Eval(vk3, Trans, Nodes.IntPoint(k));
      // dof_k = nk^t adj(J) vk
      Vector n2(&nk_ptr[dof2nk[k] * 3], 2);
      Vector n3(&nk_ptr[dof2nk[k] * 3], 3);

      dofs(k) = Trans.AdjugateJacobian().InnerProduct(vk2, n2) +
                Trans.Weight() * vk3(2) * n3(2);
   }
}

void RT_R2D_FiniteElement::Project(const FiniteElement &fe,
                                   ElementTransformation &Trans,
                                   DenseMatrix &I) const
{
   if (fe.GetRangeType() == SCALAR)
   {
      real_t vk[Geometry::MaxDim];
      Vector shape(fe.GetDof());

      real_t * nk_ptr = const_cast<real_t*>(nk);

      I.SetSize(dof, vdim*fe.GetDof());
      for (int k = 0; k < dof; k++)
      {
         const IntegrationPoint &ip = Nodes.IntPoint(k);

         Vector n2(&nk_ptr[dof2nk[k] * 3], 2);
         Vector n3(&nk_ptr[dof2nk[k] * 3], 3);

         fe.CalcShape(ip, shape);
         Trans.SetIntPoint(&ip);
         // Transform RT face normals from reference to physical space
         // vk = adj(J)^T nk
         Trans.AdjugateJacobian().MultTranspose(n2, vk);
         vk[2] = n3[2] * Trans.Weight();
         if (fe.GetMapType() == INTEGRAL)
         {
            real_t w = 1.0/Trans.Weight();
            for (int d = 0; d < 2; d++)
            {
               vk[d] *= w;
            }
         }

         for (int j = 0; j < shape.Size(); j++)
         {
            real_t s = shape(j);
            if (fabs(s) < 1e-12)
            {
               s = 0.0;
            }
            // Project scalar basis function multiplied by each coordinate
            // direction onto the transformed face normals
            for (int d = 0; d < vdim; d++)
            {
               I(k,j+d*shape.Size()) = s*vk[d];
            }
         }
      }
   }
   else
   {
      real_t vk[Geometry::MaxDim];
      DenseMatrix vshape(fe.GetDof(), fe.GetRangeDim());

      real_t * nk_ptr = const_cast<real_t*>(nk);

      I.SetSize(dof, fe.GetDof());
      for (int k = 0; k < dof; k++)
      {
         const IntegrationPoint &ip = Nodes.IntPoint(k);

         Vector n2(&nk_ptr[dof2nk[k] * 3], 2);
         Vector n3(&nk_ptr[dof2nk[k] * 3], 3);

         Trans.SetIntPoint(&ip);
         // Transform RT face normals from reference to physical space
         // vk = adj(J)^T nk
         Trans.AdjugateJacobian().MultTranspose(n2, vk);
         // Compute fe basis functions in physical space
         fe.CalcVShape(Trans, vshape);
         // Project fe basis functions onto transformed face normals
         for (int j=0; j<vshape.Height(); j++)
         {
            I(k, j) = 0.0;
            for (int i=0; i<2; i++)
            {
               I(k, j) += vshape(j, i) * vk[i];
            }
            if (vshape.Width() == 3)
            {
               I(k, j) += Trans.Weight() * vshape(j, 2) * n3(2);
            }
         }
      }
   }
}

void RT_R2D_FiniteElement::ProjectCurl(const FiniteElement &fe,
                                       ElementTransformation &Trans,
                                       DenseMatrix &curl) const
{
   DenseMatrix curl_shape(fe.GetDof(), fe.GetRangeDim());
   Vector curl_k(fe.GetDof());

   real_t * nk_ptr = const_cast<real_t*>(nk);

   curl.SetSize(dof, fe.GetDof());
   for (int k = 0; k < dof; k++)
   {
      fe.CalcCurlShape(Nodes.IntPoint(k), curl_shape);
      curl_shape.Mult(nk_ptr + dof2nk[k] * 3, curl_k);
      for (int j = 0; j < curl_k.Size(); j++)
      {
         curl(k,j) = (fabs(curl_k(j)) < 1e-12) ? 0.0 : curl_k(j);
      }
   }
}

const real_t RT_R2D_TriangleElement::nk_t[12] =
{ 0.,-1.,0.,  1.,1.,0.,  -1.,0.,0., 0.,0.,1. };

RT_R2D_TriangleElement::RT_R2D_TriangleElement(const int p)
   : RT_R2D_FiniteElement(p, Geometry::TRIANGLE, ((p + 1)*(3 * p + 8))/2, nk_t),
     RT_FE(p),
     L2_FE(p)
{
   L2_FE.SetMapType(INTEGRAL);

#ifndef MFEM_THREAD_SAFE
   rt_shape.SetSize(RT_FE.GetDof(), 2);
   l2_shape.SetSize(L2_FE.GetDof());
   rt_dshape.SetSize(RT_FE.GetDof());
#endif

   int o = 0;
   int r = 0;
   int l = 0;

   // Three edges
   for (int e=0; e<3; e++)
   {
      // Dofs in the plane
      for (int i=0; i<=p; i++)
      {
         dof_map[o] = r++; dof2nk[o++] = e;
      }
   }

   // Interior dofs in the plane
   for (int j = 0; j < p; j++)
      for (int i = 0; i + j < p; i++)
      {
         dof_map[o] = r++; dof2nk[o++] = 0;
         dof_map[o] = r++; dof2nk[o++] = 2;
      }

   // Interior z-directed dofs
   for (int j = 0; j <= p; j++)
      for (int i = 0; i + j <= p; i++)
      {
         dof_map[o] = -1 - l++; dof2nk[o++] = 3;
      }

   MFEM_VERIFY(r == RT_FE.GetDof(),
               "RT_R2D_Triangle incorrect number of RT dofs.");
   MFEM_VERIFY(l == L2_FE.GetDof(),
               "RT_R2D_Triangle incorrect number of L2 dofs.");
   MFEM_VERIFY(o == GetDof(),
               "RT_R2D_Triangle incorrect number of dofs.");

   const IntegrationRule & rt_Nodes = RT_FE.GetNodes();
   const IntegrationRule & l2_Nodes = L2_FE.GetNodes();

   for (int i=0; i<dof; i++)
   {
      int idx = dof_map[i];
      if (idx >= 0)
      {
         const IntegrationPoint & ip = rt_Nodes.IntPoint(idx);
         Nodes.IntPoint(i).Set2(ip.x, ip.y);
      }
      else
      {
         const IntegrationPoint & ip = l2_Nodes.IntPoint(-idx-1);
         Nodes.IntPoint(i).Set2(ip.x, ip.y);
      }
   }
}

void RT_R2D_TriangleElement::CalcVShape(const IntegrationPoint &ip,
                                        DenseMatrix &shape) const
{
#ifdef MFEM_THREAD_SAFE
   DenseMatrix rt_shape(RT_FE.GetDof(), 2);
   Vector      l2_shape(L2_FE.GetDof());
#endif

   RT_FE.CalcVShape(ip, rt_shape);
   L2_FE.CalcShape(ip, l2_shape);

   for (int i=0; i<dof; i++)
   {
      int idx = dof_map[i];
      if (idx >= 0)
      {
         shape(i, 0) = rt_shape(idx, 0);
         shape(i, 1) = rt_shape(idx, 1);
         shape(i, 2) = 0.0;
      }
      else
      {
         shape(i, 0) = 0.0;
         shape(i, 1) = 0.0;
         shape(i, 2) = l2_shape(-idx-1);
      }
   }
}

void RT_R2D_TriangleElement::CalcDivShape(const IntegrationPoint &ip,
                                          Vector &div_shape) const
{
#ifdef MFEM_THREAD_SAFE
   Vector rt_dshape(RT_FE.GetDof());
#endif

   RT_FE.CalcDivShape(ip, rt_dshape);

   for (int i=0; i<dof; i++)
   {
      int idx = dof_map[i];
      if (idx >= 0)
      {
         div_shape(i) = rt_dshape(idx);
      }
      else
      {
         div_shape(i) = 0.0;
      }
   }
}

const real_t RT_R2D_QuadrilateralElement::nk_q[15] =
{ 0., -1., 0.,  1., 0., 0.,  0., 1., 0.,  -1., 0., 0.,  0., 0., 1. };

RT_R2D_QuadrilateralElement::RT_R2D_QuadrilateralElement(const int p,
                                                         const int cb_type,
                                                         const int ob_type)
   : RT_R2D_FiniteElement(p, Geometry::SQUARE, (3*p + 5)*(p + 1), nk_q),
     cbasis1d(poly1d.GetBasis(p + 1, VerifyClosed(cb_type))),
     obasis1d(poly1d.GetBasis(p, VerifyOpen(ob_type)))
{
   const real_t *cp = poly1d.ClosedPoints(p + 1, cb_type);
   const real_t *op = poly1d.OpenPoints(p, ob_type);
   const int dofx = (p + 1)*(p + 2);
   const int dofy = (p + 1)*(p + 2);
   const int dofxy = dofx + dofy;

#ifndef MFEM_THREAD_SAFE
   shape_cx.SetSize(p + 2);
   shape_ox.SetSize(p + 1);
   shape_cy.SetSize(p + 2);
   shape_oy.SetSize(p + 1);
   dshape_cx.SetSize(p + 2);
   dshape_cy.SetSize(p + 2);
#endif

   // edges
   int o = 0;
   for (int i = 0; i <= p; i++)  // (0,1)
   {
      dof_map[dofx + i + 0*(p + 1)] = o++;
   }
   for (int i = 0; i <= p; i++)  // (1,2)
   {
      dof_map[(p + 1) + i*(p + 2)] = o++;
   }
   for (int i = 0; i <= p; i++)  // (2,3)
   {
      dof_map[dofx + (p - i) + (p + 1)*(p + 1)] = o++;
   }
   for (int i = 0; i <= p; i++)  // (3,0)
   {
      dof_map[0 + (p - i)*(p + 2)] = o++;
   }

   // interior
   for (int j = 0; j <= p; j++)  // x-components
      for (int i = 1; i <= p; i++)
      {
         dof_map[i + j*(p + 2)] = o++;
      }
   for (int j = 1; j <= p; j++)  // y-components
      for (int i = 0; i <= p; i++)
      {
         dof_map[dofx + i + j*(p + 1)] = o++;
      }
   for (int j = 0; j <= p; j++)  // z-components
      for (int i = 0; i <= p; i++)
      {
         dof_map[dofxy + i + j*(p + 1)] = o++;
      }

   // dof orientations
   // x-components
   for (int j = 0; j <= p; j++)
      for (int i = 0; i <= p/2; i++)
      {
         int idx = i + j*(p + 2);
         dof_map[idx] = -1 - dof_map[idx];
      }
   if (p%2 == 1)
      for (int j = p/2 + 1; j <= p; j++)
      {
         int idx = (p/2 + 1) + j*(p + 2);
         dof_map[idx] = -1 - dof_map[idx];
      }
   // y-components
   for (int j = 0; j <= p/2; j++)
      for (int i = 0; i <= p; i++)
      {
         int idx = dofx + i + j*(p + 1);
         dof_map[idx] = -1 - dof_map[idx];
      }
   if (p%2 == 1)
      for (int i = 0; i <= p/2; i++)
      {
         int idx = dofx + i + (p/2 + 1)*(p + 1);
         dof_map[idx] = -1 - dof_map[idx];
      }

   o = 0;
   for (int j = 0; j <= p; j++)
      for (int i = 0; i <= p + 1; i++)
      {
         int idx;
         if ((idx = dof_map[o++]) < 0)
         {
            idx = -1 - idx;
            dof2nk[idx] = 3;
         }
         else
         {
            dof2nk[idx] = 1;
         }
         Nodes.IntPoint(idx).Set2(cp[i], op[j]);
      }
   for (int j = 0; j <= p + 1; j++)
      for (int i = 0; i <= p; i++)
      {
         int idx;
         if ((idx = dof_map[o++]) < 0)
         {
            idx = -1 - idx;
            dof2nk[idx] = 0;
         }
         else
         {
            dof2nk[idx] = 2;
         }
         Nodes.IntPoint(idx).Set2(op[i], cp[j]);
      }
   for (int j = 0; j <= p; j++)
      for (int i = 0; i <= p; i++)
      {
         int idx = dof_map[o++];
         dof2nk[idx] = 4;
         Nodes.IntPoint(idx).Set2(op[i], op[j]);
      }
}

void RT_R2D_QuadrilateralElement::CalcVShape(const IntegrationPoint &ip,
                                             DenseMatrix &shape) const
{
   const int pp1 = order;

#ifdef MFEM_THREAD_SAFE
   Vector shape_cx(pp1 + 1), shape_ox(pp1), shape_cy(pp1 + 1), shape_oy(pp1);
#endif

   cbasis1d.Eval(ip.x, shape_cx);
   obasis1d.Eval(ip.x, shape_ox);
   cbasis1d.Eval(ip.y, shape_cy);
   obasis1d.Eval(ip.y, shape_oy);

   int o = 0;
   for (int j = 0; j < pp1; j++)
      for (int i = 0; i <= pp1; i++)
      {
         int idx, s;
         if ((idx = dof_map[o++]) < 0)
         {
            idx = -1 - idx, s = -1;
         }
         else
         {
            s = +1;
         }
         shape(idx,0) = s*shape_cx(i)*shape_oy(j);
         shape(idx,1) = 0.;
         shape(idx,2) = 0.;
      }
   for (int j = 0; j <= pp1; j++)
      for (int i = 0; i < pp1; i++)
      {
         int idx, s;
         if ((idx = dof_map[o++]) < 0)
         {
            idx = -1 - idx, s = -1;
         }
         else
         {
            s = +1;
         }
         shape(idx,0) = 0.;
         shape(idx,1) = s*shape_ox(i)*shape_cy(j);
         shape(idx,2) = 0.;
      }
   for (int j = 0; j < pp1; j++)
      for (int i = 0; i < pp1; i++)
      {
         int idx = dof_map[o++];
         shape(idx,0) = 0.;
         shape(idx,1) = 0.;
         shape(idx,2) = shape_ox(i)*shape_oy(j);
      }
}

void RT_R2D_QuadrilateralElement::CalcDivShape(const IntegrationPoint &ip,
                                               Vector &divshape) const
{
   const int pp1 = order;

#ifdef MFEM_THREAD_SAFE
   Vector shape_cx(pp1 + 1), shape_ox(pp1), shape_cy(pp1 + 1), shape_oy(pp1);
   Vector dshape_cx(pp1 + 1), dshape_cy(pp1 + 1);
#endif

   cbasis1d.Eval(ip.x, shape_cx, dshape_cx);
   obasis1d.Eval(ip.x, shape_ox);
   cbasis1d.Eval(ip.y, shape_cy, dshape_cy);
   obasis1d.Eval(ip.y, shape_oy);

   int o = 0;
   for (int j = 0; j < pp1; j++)
      for (int i = 0; i <= pp1; i++)
      {
         int idx, s;
         if ((idx = dof_map[o++]) < 0)
         {
            idx = -1 - idx, s = -1;
         }
         else
         {
            s = +1;
         }
         divshape(idx) = s*dshape_cx(i)*shape_oy(j);
      }
   for (int j = 0; j <= pp1; j++)
      for (int i = 0; i < pp1; i++)
      {
         int idx, s;
         if ((idx = dof_map[o++]) < 0)
         {
            idx = -1 - idx, s = -1;
         }
         else
         {
            s = +1;
         }
         divshape(idx) = s*shape_ox(i)*dshape_cy(j);
      }
   for (int j = 0; j < pp1; j++)
      for (int i = 0; i < pp1; i++)
      {
         int idx = dof_map[o++];
         divshape(idx) = 0.;
      }
}

}
