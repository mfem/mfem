// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

// Nedelec Finite Element classes

#include "fe_nd.hpp"
#include "../coefficient.hpp"

namespace mfem
{

using namespace std;

const double ND_HexahedronElement::tk[18] =
{ 1.,0.,0.,  0.,1.,0.,  0.,0.,1., -1.,0.,0.,  0.,-1.,0.,  0.,0.,-1. };

ND_HexahedronElement::ND_HexahedronElement(const int p,
                                           const int cb_type, const int ob_type)
   : VectorTensorFiniteElement(3, 3*p*(p + 1)*(p + 1), p, cb_type, ob_type,
                               H_CURL, DofMapType::L2_DOF_MAP),
     dof2tk(dof), cp(poly1d.ClosedPoints(p, cb_type))
{
   if (obasis1d.IsIntegratedType()) { is_nodal = false; }

   dof_map.SetSize(dof);

   const double *op = poly1d.OpenPoints(p - 1, ob_type);
   const int dof3 = dof/3;

#ifndef MFEM_THREAD_SAFE
   shape_cx.SetSize(p + 1);
   shape_ox.SetSize(p);
   shape_cy.SetSize(p + 1);
   shape_oy.SetSize(p);
   shape_cz.SetSize(p + 1);
   shape_oz.SetSize(p);
   dshape_cx.SetSize(p + 1);
   dshape_cy.SetSize(p + 1);
   dshape_cz.SetSize(p + 1);
#endif

   // edges
   int o = 0;
   for (int i = 0; i < p; i++)  // (0,1)
   {
      dof_map[0*dof3 + i + (0 + 0*(p + 1))*p] = o++;
   }
   for (int i = 0; i < p; i++)  // (1,2)
   {
      dof_map[1*dof3 + p + (i + 0*p)*(p + 1)] = o++;
   }
   for (int i = 0; i < p; i++)  // (3,2)
   {
      dof_map[0*dof3 + i + (p + 0*(p + 1))*p] = o++;
   }
   for (int i = 0; i < p; i++)  // (0,3)
   {
      dof_map[1*dof3 + 0 + (i + 0*p)*(p + 1)] = o++;
   }
   for (int i = 0; i < p; i++)  // (4,5)
   {
      dof_map[0*dof3 + i + (0 + p*(p + 1))*p] = o++;
   }
   for (int i = 0; i < p; i++)  // (5,6)
   {
      dof_map[1*dof3 + p + (i + p*p)*(p + 1)] = o++;
   }
   for (int i = 0; i < p; i++)  // (7,6)
   {
      dof_map[0*dof3 + i + (p + p*(p + 1))*p] = o++;
   }
   for (int i = 0; i < p; i++)  // (4,7)
   {
      dof_map[1*dof3 + 0 + (i + p*p)*(p + 1)] = o++;
   }
   for (int i = 0; i < p; i++)  // (0,4)
   {
      dof_map[2*dof3 + 0 + (0 + i*(p + 1))*(p + 1)] = o++;
   }
   for (int i = 0; i < p; i++)  // (1,5)
   {
      dof_map[2*dof3 + p + (0 + i*(p + 1))*(p + 1)] = o++;
   }
   for (int i = 0; i < p; i++)  // (2,6)
   {
      dof_map[2*dof3 + p + (p + i*(p + 1))*(p + 1)] = o++;
   }
   for (int i = 0; i < p; i++)  // (3,7)
   {
      dof_map[2*dof3 + 0 + (p + i*(p + 1))*(p + 1)] = o++;
   }

   // faces
   // (3,2,1,0) -- bottom
   for (int j = 1; j < p; j++) // x - components
      for (int i = 0; i < p; i++)
      {
         dof_map[0*dof3 + i + ((p - j) + 0*(p + 1))*p] = o++;
      }
   for (int j = 0; j < p; j++) // y - components
      for (int i = 1; i < p; i++)
      {
         dof_map[1*dof3 + i + ((p - 1 - j) + 0*p)*(p + 1)] = -1 - (o++);
      }
   // (0,1,5,4) -- front
   for (int k = 1; k < p; k++) // x - components
      for (int i = 0; i < p; i++)
      {
         dof_map[0*dof3 + i + (0 + k*(p + 1))*p] = o++;
      }
   for (int k = 0; k < p; k++) // z - components
      for (int i = 1; i < p; i++ )
      {
         dof_map[2*dof3 + i + (0 + k*(p + 1))*(p + 1)] = o++;
      }
   // (1,2,6,5) -- right
   for (int k = 1; k < p; k++) // y - components
      for (int j = 0; j < p; j++)
      {
         dof_map[1*dof3 + p + (j + k*p)*(p + 1)] = o++;
      }
   for (int k = 0; k < p; k++) // z - components
      for (int j = 1; j < p; j++)
      {
         dof_map[2*dof3 + p + (j + k*(p + 1))*(p + 1)] = o++;
      }
   // (2,3,7,6) -- back
   for (int k = 1; k < p; k++) // x - components
      for (int i = 0; i < p; i++)
      {
         dof_map[0*dof3 + (p - 1 - i) + (p + k*(p + 1))*p] = -1 - (o++);
      }
   for (int k = 0; k < p; k++) // z - components
      for (int i = 1; i < p; i++)
      {
         dof_map[2*dof3 + (p - i) + (p + k*(p + 1))*(p + 1)] = o++;
      }
   // (3,0,4,7) -- left
   for (int k = 1; k < p; k++) // y - components
      for (int j = 0; j < p; j++)
      {
         dof_map[1*dof3 + 0 + ((p - 1 - j) + k*p)*(p + 1)] = -1 - (o++);
      }
   for (int k = 0; k < p; k++) // z - components
      for (int j = 1; j < p; j++)
      {
         dof_map[2*dof3 + 0 + ((p - j) + k*(p + 1))*(p + 1)] = o++;
      }
   // (4,5,6,7) -- top
   for (int j = 1; j < p; j++) // x - components
      for (int i = 0; i < p; i++)
      {
         dof_map[0*dof3 + i + (j + p*(p + 1))*p] = o++;
      }
   for (int j = 0; j < p; j++) // y - components
      for (int i = 1; i < p; i++)
      {
         dof_map[1*dof3 + i + (j + p*p)*(p + 1)] = o++;
      }

   // interior
   // x-components
   for (int k = 1; k < p; k++)
      for (int j = 1; j < p; j++)
         for (int i = 0; i < p; i++)
         {
            dof_map[0*dof3 + i + (j + k*(p + 1))*p] = o++;
         }
   // y-components
   for (int k = 1; k < p; k++)
      for (int j = 0; j < p; j++)
         for (int i = 1; i < p; i++)
         {
            dof_map[1*dof3 + i + (j + k*p)*(p + 1)] = o++;
         }
   // z-components
   for (int k = 0; k < p; k++)
      for (int j = 1; j < p; j++)
         for (int i = 1; i < p; i++)
         {
            dof_map[2*dof3 + i + (j + k*(p + 1))*(p + 1)] = o++;
         }

   // set dof2tk and Nodes
   o = 0;
   // x-components
   for (int k = 0; k <= p; k++)
      for (int j = 0; j <= p; j++)
         for (int i = 0; i < p; i++)
         {
            int idx;
            if ((idx = dof_map[o++]) < 0)
            {
               dof2tk[idx = -1 - idx] = 3;
            }
            else
            {
               dof2tk[idx] = 0;
            }
            Nodes.IntPoint(idx).Set3(op[i], cp[j], cp[k]);
         }
   // y-components
   for (int k = 0; k <= p; k++)
      for (int j = 0; j < p; j++)
         for (int i = 0; i <= p; i++)
         {
            int idx;
            if ((idx = dof_map[o++]) < 0)
            {
               dof2tk[idx = -1 - idx] = 4;
            }
            else
            {
               dof2tk[idx] = 1;
            }
            Nodes.IntPoint(idx).Set3(cp[i], op[j], cp[k]);
         }
   // z-components
   for (int k = 0; k < p; k++)
      for (int j = 0; j <= p; j++)
         for (int i = 0; i <= p; i++)
         {
            int idx;
            if ((idx = dof_map[o++]) < 0)
            {
               dof2tk[idx = -1 - idx] = 5;
            }
            else
            {
               dof2tk[idx] = 2;
            }
            Nodes.IntPoint(idx).Set3(cp[i], cp[j], op[k]);
         }
}

void ND_HexahedronElement::ProjectIntegrated(VectorCoefficient &vc,
                                             ElementTransformation &Trans,
                                             Vector &dofs) const
{
   MFEM_ASSERT(obasis1d.IsIntegratedType(), "Not integrated type");
   double vk[Geometry::MaxDim];
   Vector xk(vk, vc.GetVDim());

   const IntegrationRule &ir = IntRules.Get(Geometry::SEGMENT, order);
   const int nqpt = ir.GetNPoints();

   IntegrationPoint ip3d;

   int o = 0;
   for (int c = 0; c < 3; ++c)  // loop over x, y, z components
   {
      const int im = c == 0 ? order - 1 : order;
      const int jm = c == 1 ? order - 1 : order;
      const int km = c == 2 ? order - 1 : order;

      for (int k = 0; k <= km; k++)
         for (int j = 0; j <= jm; j++)
            for (int i = 0; i <= im; i++)
            {
               int idx;
               if ((idx = dof_map[o++]) < 0)
               {
                  idx = -1 - idx;
               }

               const int id1 = c == 0 ? i : (c == 1 ? j : k);
               const double h = cp[id1+1] - cp[id1];

               double val = 0.0;

               for (int q = 0; q < nqpt; q++)
               {
                  const IntegrationPoint &ip1d = ir.IntPoint(q);

                  if (c == 0)
                  {
                     ip3d.Set3(cp[i] + (h*ip1d.x), cp[j], cp[k]);
                  }
                  else if (c == 1)
                  {
                     ip3d.Set3(cp[i], cp[j] + (h*ip1d.x), cp[k]);
                  }
                  else
                  {
                     ip3d.Set3(cp[i], cp[j], cp[k] + (h*ip1d.x));
                  }

                  Trans.SetIntPoint(&ip3d);
                  vc.Eval(xk, Trans, ip3d);

                  // xk^t J tk
                  const double ipval = Trans.Jacobian().InnerProduct(tk + dof2tk[idx]*dim, vk);
                  val += ip1d.weight * ipval;
               }

               dofs(idx) = val*h;
            }
   }
}

void ND_HexahedronElement::CalcVShape(const IntegrationPoint &ip,
                                      DenseMatrix &shape) const
{
   const int p = order;

#ifdef MFEM_THREAD_SAFE
   Vector shape_cx(p + 1), shape_ox(p), shape_cy(p + 1), shape_oy(p);
   Vector shape_cz(p + 1), shape_oz(p);
   Vector dshape_cx(p + 1), dshape_cy(p + 1), dshape_cz(p + 1);
#endif

   if (obasis1d.IsIntegratedType())
   {
      cbasis1d.Eval(ip.x, shape_cx, dshape_cx);
      cbasis1d.Eval(ip.y, shape_cy, dshape_cy);
      cbasis1d.Eval(ip.z, shape_cz, dshape_cz);
      obasis1d.ScaleIntegrated(false);
      obasis1d.EvalIntegrated(dshape_cx, shape_ox);
      obasis1d.EvalIntegrated(dshape_cy, shape_oy);
      obasis1d.EvalIntegrated(dshape_cz, shape_oz);
   }
   else
   {
      cbasis1d.Eval(ip.x, shape_cx);
      cbasis1d.Eval(ip.y, shape_cy);
      cbasis1d.Eval(ip.z, shape_cz);
      obasis1d.Eval(ip.x, shape_ox);
      obasis1d.Eval(ip.y, shape_oy);
      obasis1d.Eval(ip.z, shape_oz);
   }

   int o = 0;
   // x-components
   for (int k = 0; k <= p; k++)
      for (int j = 0; j <= p; j++)
         for (int i = 0; i < p; i++)
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
            shape(idx,0) = s*shape_ox(i)*shape_cy(j)*shape_cz(k);
            shape(idx,1) = 0.;
            shape(idx,2) = 0.;
         }
   // y-components
   for (int k = 0; k <= p; k++)
      for (int j = 0; j < p; j++)
         for (int i = 0; i <= p; i++)
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
            shape(idx,1) = s*shape_cx(i)*shape_oy(j)*shape_cz(k);
            shape(idx,2) = 0.;
         }
   // z-components
   for (int k = 0; k < p; k++)
      for (int j = 0; j <= p; j++)
         for (int i = 0; i <= p; i++)
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
            shape(idx,2) = s*shape_cx(i)*shape_cy(j)*shape_oz(k);
         }
}

void ND_HexahedronElement::CalcCurlShape(const IntegrationPoint &ip,
                                         DenseMatrix &curl_shape) const
{
   const int p = order;

#ifdef MFEM_THREAD_SAFE
   Vector shape_cx(p + 1), shape_ox(p), shape_cy(p + 1), shape_oy(p);
   Vector shape_cz(p + 1), shape_oz(p);
   Vector dshape_cx(p + 1), dshape_cy(p + 1), dshape_cz(p + 1);
#endif

   cbasis1d.Eval(ip.x, shape_cx, dshape_cx);
   cbasis1d.Eval(ip.y, shape_cy, dshape_cy);
   cbasis1d.Eval(ip.z, shape_cz, dshape_cz);
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
   for (int k = 0; k <= p; k++)
      for (int j = 0; j <= p; j++)
         for (int i = 0; i < p; i++)
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
            curl_shape(idx,0) = 0.;
            curl_shape(idx,1) =  s*shape_ox(i)* shape_cy(j)*dshape_cz(k);
            curl_shape(idx,2) = -s*shape_ox(i)*dshape_cy(j)* shape_cz(k);
         }
   // y-components
   for (int k = 0; k <= p; k++)
      for (int j = 0; j < p; j++)
         for (int i = 0; i <= p; i++)
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
            curl_shape(idx,0) = -s* shape_cx(i)*shape_oy(j)*dshape_cz(k);
            curl_shape(idx,1) = 0.;
            curl_shape(idx,2) =  s*dshape_cx(i)*shape_oy(j)* shape_cz(k);
         }
   // z-components
   for (int k = 0; k < p; k++)
      for (int j = 0; j <= p; j++)
         for (int i = 0; i <= p; i++)
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
            curl_shape(idx,0) =   s* shape_cx(i)*dshape_cy(j)*shape_oz(k);
            curl_shape(idx,1) =  -s*dshape_cx(i)* shape_cy(j)*shape_oz(k);
            curl_shape(idx,2) = 0.;
         }
}

const double ND_QuadrilateralElement::tk[8] =
{ 1.,0.,  0.,1., -1.,0., 0.,-1. };

ND_QuadrilateralElement::ND_QuadrilateralElement(const int p,
                                                 const int cb_type,
                                                 const int ob_type)
   : VectorTensorFiniteElement(2, 2*p*(p + 1), p, cb_type, ob_type,
                               H_CURL, DofMapType::L2_DOF_MAP),
     dof2tk(dof),
     cp(poly1d.ClosedPoints(p, cb_type))
{
   if (obasis1d.IsIntegratedType()) { is_nodal = false; }

   dof_map.SetSize(dof);

   const double *op = poly1d.OpenPoints(p - 1, ob_type);
   const int dof2 = dof/2;

#ifndef MFEM_THREAD_SAFE
   shape_cx.SetSize(p + 1);
   shape_ox.SetSize(p);
   shape_cy.SetSize(p + 1);
   shape_oy.SetSize(p);
   dshape_cx.SetSize(p + 1);
   dshape_cy.SetSize(p + 1);
#endif

   // edges
   int o = 0;
   for (int i = 0; i < p; i++)  // (0,1)
   {
      dof_map[0*dof2 + i + 0*p] = o++;
   }
   for (int j = 0; j < p; j++)  // (1,2)
   {
      dof_map[1*dof2 + p + j*(p + 1)] = o++;
   }
   for (int i = 0; i < p; i++)  // (2,3)
   {
      dof_map[0*dof2 + (p - 1 - i) + p*p] = -1 - (o++);
   }
   for (int j = 0; j < p; j++)  // (3,0)
   {
      dof_map[1*dof2 + 0 + (p - 1 - j)*(p + 1)] = -1 - (o++);
   }

   // interior
   // x-components
   for (int j = 1; j < p; j++)
      for (int i = 0; i < p; i++)
      {
         dof_map[0*dof2 + i + j*p] = o++;
      }
   // y-components
   for (int j = 0; j < p; j++)
      for (int i = 1; i < p; i++)
      {
         dof_map[1*dof2 + i + j*(p + 1)] = o++;
      }

   // set dof2tk and Nodes
   o = 0;
   // x-components
   for (int j = 0; j <= p; j++)
      for (int i = 0; i < p; i++)
      {
         int idx;
         if ((idx = dof_map[o++]) < 0)
         {
            dof2tk[idx = -1 - idx] = 2;
         }
         else
         {
            dof2tk[idx] = 0;
         }
         Nodes.IntPoint(idx).Set2(op[i], cp[j]);
      }
   // y-components
   for (int j = 0; j < p; j++)
      for (int i = 0; i <= p; i++)
      {
         int idx;
         if ((idx = dof_map[o++]) < 0)
         {
            dof2tk[idx = -1 - idx] = 3;
         }
         else
         {
            dof2tk[idx] = 1;
         }
         Nodes.IntPoint(idx).Set2(cp[i], op[j]);
      }
}

void ND_QuadrilateralElement::ProjectIntegrated(VectorCoefficient &vc,
                                                ElementTransformation &Trans,
                                                Vector &dofs) const
{
   MFEM_ASSERT(obasis1d.IsIntegratedType(), "Not integrated type");
   double vk[Geometry::MaxDim];
   Vector xk(vk, vc.GetVDim());

   const IntegrationRule &ir = IntRules.Get(Geometry::SEGMENT, order);
   const int nqpt = ir.GetNPoints();

   IntegrationPoint ip2d;

   int o = 0;
   // x-components
   for (int j = 0; j <= order; j++)
      for (int i = 0; i < order; i++)
      {
         int idx;
         if ((idx = dof_map[o++]) < 0)
         {
            idx = -1 - idx;
         }

         const double h = cp[i+1] - cp[i];

         double val = 0.0;

         for (int k = 0; k < nqpt; k++)
         {
            const IntegrationPoint &ip1d = ir.IntPoint(k);

            ip2d.Set2(cp[i] + (h*ip1d.x), cp[j]);

            Trans.SetIntPoint(&ip2d);
            vc.Eval(xk, Trans, ip2d);

            // xk^t J tk
            const double ipval = Trans.Jacobian().InnerProduct(tk + dof2tk[idx]*dim, vk);
            val += ip1d.weight * ipval;
         }

         dofs(idx) = val*h;
      }
   // y-components
   for (int j = 0; j < order; j++)
      for (int i = 0; i <= order; i++)
      {
         int idx;
         if ((idx = dof_map[o++]) < 0)
         {
            idx = -1 - idx;
         }

         const double h = cp[j+1] - cp[j];

         double val = 0.0;

         for (int k = 0; k < nqpt; k++)
         {
            const IntegrationPoint &ip1d = ir.IntPoint(k);

            ip2d.Set2(cp[i], cp[j] + (h*ip1d.x));

            Trans.SetIntPoint(&ip2d);
            vc.Eval(xk, Trans, ip2d);

            // xk^t J tk
            const double ipval = Trans.Jacobian().InnerProduct(tk + dof2tk[idx]*dim, vk);
            val += ip1d.weight * ipval;
         }

         dofs(idx) = val*h;
      }
}

void ND_QuadrilateralElement::CalcVShape(const IntegrationPoint &ip,
                                         DenseMatrix &shape) const
{
   const int p = order;

#ifdef MFEM_THREAD_SAFE
   Vector shape_cx(p + 1), shape_ox(p), shape_cy(p + 1), shape_oy(p);
   Vector dshape_cx(p + 1), dshape_cy(p + 1);
#endif

   if (obasis1d.IsIntegratedType())
   {
      cbasis1d.Eval(ip.x, shape_cx, dshape_cx);
      cbasis1d.Eval(ip.y, shape_cy, dshape_cy);
      obasis1d.ScaleIntegrated(false);
      obasis1d.EvalIntegrated(dshape_cx, shape_ox);
      obasis1d.EvalIntegrated(dshape_cy, shape_oy);
   }
   else
   {
      cbasis1d.Eval(ip.x, shape_cx);
      cbasis1d.Eval(ip.y, shape_cy);
      obasis1d.Eval(ip.x, shape_ox);
      obasis1d.Eval(ip.y, shape_oy);
   }

   int o = 0;
   // x-components
   for (int j = 0; j <= p; j++)
      for (int i = 0; i < p; i++)
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
         shape(idx,0) = s*shape_ox(i)*shape_cy(j);
         shape(idx,1) = 0.;
      }
   // y-components
   for (int j = 0; j < p; j++)
      for (int i = 0; i <= p; i++)
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
         shape(idx,1) = s*shape_cx(i)*shape_oy(j);
      }
}

void ND_QuadrilateralElement::CalcCurlShape(const IntegrationPoint &ip,
                                            DenseMatrix &curl_shape) const
{
   const int p = order;

#ifdef MFEM_THREAD_SAFE
   Vector shape_cx(p + 1), shape_ox(p), shape_cy(p + 1), shape_oy(p);
   Vector dshape_cx(p + 1), dshape_cy(p + 1);
#endif

   cbasis1d.Eval(ip.x, shape_cx, dshape_cx);
   cbasis1d.Eval(ip.y, shape_cy, dshape_cy);
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
   // x-components
   for (int j = 0; j <= p; j++)
      for (int i = 0; i < p; i++)
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
         curl_shape(idx,0) = -s*shape_ox(i)*dshape_cy(j);
      }
   // y-components
   for (int j = 0; j < p; j++)
      for (int i = 0; i <= p; i++)
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
         curl_shape(idx,0) =  s*dshape_cx(i)*shape_oy(j);
      }
}


const double ND_TetrahedronElement::tk[18] =
{ 1.,0.,0.,  0.,1.,0.,  0.,0.,1.,  -1.,1.,0.,  -1.,0.,1.,  0.,-1.,1. };

const double ND_TetrahedronElement::c = 1./4.;

ND_TetrahedronElement::ND_TetrahedronElement(const int p)
   : VectorFiniteElement(3, Geometry::TETRAHEDRON, p*(p + 2)*(p + 3)/2, p,
                         H_CURL, FunctionSpace::Pk), dof2tk(dof)
{
   const double *eop = poly1d.OpenPoints(p - 1);
   const double *fop = (p > 1) ? poly1d.OpenPoints(p - 2) : NULL;
   const double *iop = (p > 2) ? poly1d.OpenPoints(p - 3) : NULL;

   const int pm1 = p - 1, pm2 = p - 2, pm3 = p - 3;

#ifndef MFEM_THREAD_SAFE
   shape_x.SetSize(p);
   shape_y.SetSize(p);
   shape_z.SetSize(p);
   shape_l.SetSize(p);
   dshape_x.SetSize(p);
   dshape_y.SetSize(p);
   dshape_z.SetSize(p);
   dshape_l.SetSize(p);
   u.SetSize(dof, dim);
#else
   Vector shape_x(p), shape_y(p), shape_z(p), shape_l(p);
#endif

   int o = 0;
   // edges
   for (int i = 0; i < p; i++) // (0,1)
   {
      Nodes.IntPoint(o).Set3(eop[i], 0., 0.);
      dof2tk[o++] = 0;
   }
   for (int i = 0; i < p; i++) // (0,2)
   {
      Nodes.IntPoint(o).Set3(0., eop[i], 0.);
      dof2tk[o++] = 1;
   }
   for (int i = 0; i < p; i++) // (0,3)
   {
      Nodes.IntPoint(o).Set3(0., 0., eop[i]);
      dof2tk[o++] = 2;
   }
   for (int i = 0; i < p; i++) // (1,2)
   {
      Nodes.IntPoint(o).Set3(eop[pm1-i], eop[i], 0.);
      dof2tk[o++] = 3;
   }
   for (int i = 0; i < p; i++) // (1,3)
   {
      Nodes.IntPoint(o).Set3(eop[pm1-i], 0., eop[i]);
      dof2tk[o++] = 4;
   }
   for (int i = 0; i < p; i++) // (2,3)
   {
      Nodes.IntPoint(o).Set3(0., eop[pm1-i], eop[i]);
      dof2tk[o++] = 5;
   }

   // faces
   for (int j = 0; j <= pm2; j++)  // (1,2,3)
      for (int i = 0; i + j <= pm2; i++)
      {
         double w = fop[i] + fop[j] + fop[pm2-i-j];
         Nodes.IntPoint(o).Set3(fop[pm2-i-j]/w, fop[i]/w, fop[j]/w);
         dof2tk[o++] = 3;
         Nodes.IntPoint(o).Set3(fop[pm2-i-j]/w, fop[i]/w, fop[j]/w);
         dof2tk[o++] = 4;
      }
   for (int j = 0; j <= pm2; j++)  // (0,3,2)
      for (int i = 0; i + j <= pm2; i++)
      {
         double w = fop[i] + fop[j] + fop[pm2-i-j];
         Nodes.IntPoint(o).Set3(0., fop[j]/w, fop[i]/w);
         dof2tk[o++] = 2;
         Nodes.IntPoint(o).Set3(0., fop[j]/w, fop[i]/w);
         dof2tk[o++] = 1;
      }
   for (int j = 0; j <= pm2; j++)  // (0,1,3)
      for (int i = 0; i + j <= pm2; i++)
      {
         double w = fop[i] + fop[j] + fop[pm2-i-j];
         Nodes.IntPoint(o).Set3(fop[i]/w, 0., fop[j]/w);
         dof2tk[o++] = 0;
         Nodes.IntPoint(o).Set3(fop[i]/w, 0., fop[j]/w);
         dof2tk[o++] = 2;
      }
   for (int j = 0; j <= pm2; j++)  // (0,2,1)
      for (int i = 0; i + j <= pm2; i++)
      {
         double w = fop[i] + fop[j] + fop[pm2-i-j];
         Nodes.IntPoint(o).Set3(fop[j]/w, fop[i]/w, 0.);
         dof2tk[o++] = 1;
         Nodes.IntPoint(o).Set3(fop[j]/w, fop[i]/w, 0.);
         dof2tk[o++] = 0;
      }

   // interior
   for (int k = 0; k <= pm3; k++)
      for (int j = 0; j + k <= pm3; j++)
         for (int i = 0; i + j + k <= pm3; i++)
         {
            double w = iop[i] + iop[j] + iop[k] + iop[pm3-i-j-k];
            Nodes.IntPoint(o).Set3(iop[i]/w, iop[j]/w, iop[k]/w);
            dof2tk[o++] = 0;
            Nodes.IntPoint(o).Set3(iop[i]/w, iop[j]/w, iop[k]/w);
            dof2tk[o++] = 1;
            Nodes.IntPoint(o).Set3(iop[i]/w, iop[j]/w, iop[k]/w);
            dof2tk[o++] = 2;
         }

   DenseMatrix T(dof);
   for (int m = 0; m < dof; m++)
   {
      const IntegrationPoint &ip = Nodes.IntPoint(m);
      const double *tm = tk + 3*dof2tk[m];
      o = 0;

      poly1d.CalcBasis(pm1, ip.x, shape_x);
      poly1d.CalcBasis(pm1, ip.y, shape_y);
      poly1d.CalcBasis(pm1, ip.z, shape_z);
      poly1d.CalcBasis(pm1, 1. - ip.x - ip.y - ip.z, shape_l);

      for (int k = 0; k <= pm1; k++)
         for (int j = 0; j + k <= pm1; j++)
            for (int i = 0; i + j + k <= pm1; i++)
            {
               double s = shape_x(i)*shape_y(j)*shape_z(k)*shape_l(pm1-i-j-k);
               T(o++, m) = s * tm[0];
               T(o++, m) = s * tm[1];
               T(o++, m) = s * tm[2];
            }
      for (int k = 0; k <= pm1; k++)
         for (int j = 0; j + k <= pm1; j++)
         {
            double s = shape_x(pm1-j-k)*shape_y(j)*shape_z(k);
            T(o++, m) = s*((ip.y - c)*tm[0] - (ip.x - c)*tm[1]);
            T(o++, m) = s*((ip.z - c)*tm[0] - (ip.x - c)*tm[2]);
         }
      for (int k = 0; k <= pm1; k++)
      {
         T(o++, m) =
            shape_y(pm1-k)*shape_z(k)*((ip.z - c)*tm[1] - (ip.y - c)*tm[2]);
      }
   }

   Ti.Factor(T);
   // mfem::out << "ND_TetrahedronElement(" << p << ") : "; Ti.TestInversion();
}

void ND_TetrahedronElement::CalcVShape(const IntegrationPoint &ip,
                                       DenseMatrix &shape) const
{
   const int pm1 = order - 1;

#ifdef MFEM_THREAD_SAFE
   const int p = order;
   Vector shape_x(p), shape_y(p), shape_z(p), shape_l(p);
   DenseMatrix u(dof, dim);
#endif

   poly1d.CalcBasis(pm1, ip.x, shape_x);
   poly1d.CalcBasis(pm1, ip.y, shape_y);
   poly1d.CalcBasis(pm1, ip.z, shape_z);
   poly1d.CalcBasis(pm1, 1. - ip.x - ip.y - ip.z, shape_l);

   int n = 0;
   for (int k = 0; k <= pm1; k++)
      for (int j = 0; j + k <= pm1; j++)
         for (int i = 0; i + j + k <= pm1; i++)
         {
            double s = shape_x(i)*shape_y(j)*shape_z(k)*shape_l(pm1-i-j-k);
            u(n,0) =  s;  u(n,1) = 0.;  u(n,2) = 0.;  n++;
            u(n,0) = 0.;  u(n,1) =  s;  u(n,2) = 0.;  n++;
            u(n,0) = 0.;  u(n,1) = 0.;  u(n,2) =  s;  n++;
         }
   for (int k = 0; k <= pm1; k++)
      for (int j = 0; j + k <= pm1; j++)
      {
         double s = shape_x(pm1-j-k)*shape_y(j)*shape_z(k);
         u(n,0) = s*(ip.y - c);  u(n,1) = -s*(ip.x - c);  u(n,2) =  0.;  n++;
         u(n,0) = s*(ip.z - c);  u(n,1) =  0.;  u(n,2) = -s*(ip.x - c);  n++;
      }
   for (int k = 0; k <= pm1; k++)
   {
      double s = shape_y(pm1-k)*shape_z(k);
      u(n,0) = 0.;  u(n,1) = s*(ip.z - c);  u(n,2) = -s*(ip.y - c);  n++;
   }

   Ti.Mult(u, shape);
}

void ND_TetrahedronElement::CalcCurlShape(const IntegrationPoint &ip,
                                          DenseMatrix &curl_shape) const
{
   const int pm1 = order - 1;

#ifdef MFEM_THREAD_SAFE
   const int p = order;
   Vector shape_x(p), shape_y(p), shape_z(p), shape_l(p);
   Vector dshape_x(p), dshape_y(p), dshape_z(p), dshape_l(p);
   DenseMatrix u(dof, dim);
#endif

   poly1d.CalcBasis(pm1, ip.x, shape_x, dshape_x);
   poly1d.CalcBasis(pm1, ip.y, shape_y, dshape_y);
   poly1d.CalcBasis(pm1, ip.z, shape_z, dshape_z);
   poly1d.CalcBasis(pm1, 1. - ip.x - ip.y - ip.z, shape_l, dshape_l);

   int n = 0;
   for (int k = 0; k <= pm1; k++)
      for (int j = 0; j + k <= pm1; j++)
         for (int i = 0; i + j + k <= pm1; i++)
         {
            int l = pm1-i-j-k;
            const double dx = (dshape_x(i)*shape_l(l) -
                               shape_x(i)*dshape_l(l))*shape_y(j)*shape_z(k);
            const double dy = (dshape_y(j)*shape_l(l) -
                               shape_y(j)*dshape_l(l))*shape_x(i)*shape_z(k);
            const double dz = (dshape_z(k)*shape_l(l) -
                               shape_z(k)*dshape_l(l))*shape_x(i)*shape_y(j);

            u(n,0) =  0.;  u(n,1) =  dz;  u(n,2) = -dy;  n++;
            u(n,0) = -dz;  u(n,1) =  0.;  u(n,2) =  dx;  n++;
            u(n,0) =  dy;  u(n,1) = -dx;  u(n,2) =  0.;  n++;
         }
   for (int k = 0; k <= pm1; k++)
      for (int j = 0; j + k <= pm1; j++)
      {
         int i = pm1 - j - k;
         // s = shape_x(i)*shape_y(j)*shape_z(k);
         // curl of s*(ip.y - c, -(ip.x - c), 0):
         u(n,0) =  shape_x(i)*(ip.x - c)*shape_y(j)*dshape_z(k);
         u(n,1) =  shape_x(i)*shape_y(j)*(ip.y - c)*dshape_z(k);
         u(n,2) =
            -((dshape_x(i)*(ip.x - c) + shape_x(i))*shape_y(j)*shape_z(k) +
              (dshape_y(j)*(ip.y - c) + shape_y(j))*shape_x(i)*shape_z(k));
         n++;
         // curl of s*(ip.z - c, 0, -(ip.x - c)):
         u(n,0) = -shape_x(i)*(ip.x - c)*dshape_y(j)*shape_z(k);
         u(n,1) = (shape_x(i)*shape_y(j)*(dshape_z(k)*(ip.z - c) + shape_z(k)) +
                   (dshape_x(i)*(ip.x - c) + shape_x(i))*shape_y(j)*shape_z(k));
         u(n,2) = -shape_x(i)*dshape_y(j)*shape_z(k)*(ip.z - c);
         n++;
      }
   for (int k = 0; k <= pm1; k++)
   {
      int j = pm1 - k;
      // curl of shape_y(j)*shape_z(k)*(0, ip.z - c, -(ip.y - c)):
      u(n,0) = -((dshape_y(j)*(ip.y - c) + shape_y(j))*shape_z(k) +
                 shape_y(j)*(dshape_z(k)*(ip.z - c) + shape_z(k)));
      u(n,1) = 0.;
      u(n,2) = 0.;  n++;
   }

   Ti.Mult(u, curl_shape);
}


const double ND_TriangleElement::tk[8] =
{ 1.,0.,  -1.,1.,  0.,-1.,  0.,1. };

const double ND_TriangleElement::c = 1./3.;

ND_TriangleElement::ND_TriangleElement(const int p)
   : VectorFiniteElement(2, Geometry::TRIANGLE, p*(p + 2), p,
                         H_CURL, FunctionSpace::Pk),
     dof2tk(dof)
{
   const double *eop = poly1d.OpenPoints(p - 1);
   const double *iop = (p > 1) ? poly1d.OpenPoints(p - 2) : NULL;

   const int pm1 = p - 1, pm2 = p - 2;

#ifndef MFEM_THREAD_SAFE
   shape_x.SetSize(p);
   shape_y.SetSize(p);
   shape_l.SetSize(p);
   dshape_x.SetSize(p);
   dshape_y.SetSize(p);
   dshape_l.SetSize(p);
   u.SetSize(dof, dim);
   curlu.SetSize(dof);
#else
   Vector shape_x(p), shape_y(p), shape_l(p);
#endif

   int n = 0;
   // edges
   for (int i = 0; i < p; i++) // (0,1)
   {
      Nodes.IntPoint(n).Set2(eop[i], 0.);
      dof2tk[n++] = 0;
   }
   for (int i = 0; i < p; i++) // (1,2)
   {
      Nodes.IntPoint(n).Set2(eop[pm1-i], eop[i]);
      dof2tk[n++] = 1;
   }
   for (int i = 0; i < p; i++) // (2,0)
   {
      Nodes.IntPoint(n).Set2(0., eop[pm1-i]);
      dof2tk[n++] = 2;
   }

   // interior
   for (int j = 0; j <= pm2; j++)
      for (int i = 0; i + j <= pm2; i++)
      {
         double w = iop[i] + iop[j] + iop[pm2-i-j];
         Nodes.IntPoint(n).Set2(iop[i]/w, iop[j]/w);
         dof2tk[n++] = 0;
         Nodes.IntPoint(n).Set2(iop[i]/w, iop[j]/w);
         dof2tk[n++] = 3;
      }

   DenseMatrix T(dof);
   for (int m = 0; m < dof; m++)
   {
      const IntegrationPoint &ip = Nodes.IntPoint(m);
      const double *tm = tk + 2*dof2tk[m];
      n = 0;

      poly1d.CalcBasis(pm1, ip.x, shape_x);
      poly1d.CalcBasis(pm1, ip.y, shape_y);
      poly1d.CalcBasis(pm1, 1. - ip.x - ip.y, shape_l);

      for (int j = 0; j <= pm1; j++)
         for (int i = 0; i + j <= pm1; i++)
         {
            double s = shape_x(i)*shape_y(j)*shape_l(pm1-i-j);
            T(n++, m) = s * tm[0];
            T(n++, m) = s * tm[1];
         }
      for (int j = 0; j <= pm1; j++)
      {
         T(n++, m) =
            shape_x(pm1-j)*shape_y(j)*((ip.y - c)*tm[0] - (ip.x - c)*tm[1]);
      }
   }

   Ti.Factor(T);
   // mfem::out << "ND_TriangleElement(" << p << ") : "; Ti.TestInversion();
}

void ND_TriangleElement::CalcVShape(const IntegrationPoint &ip,
                                    DenseMatrix &shape) const
{
   const int pm1 = order - 1;

#ifdef MFEM_THREAD_SAFE
   const int p = order;
   Vector shape_x(p), shape_y(p), shape_l(p);
   DenseMatrix u(dof, dim);
#endif

   poly1d.CalcBasis(pm1, ip.x, shape_x);
   poly1d.CalcBasis(pm1, ip.y, shape_y);
   poly1d.CalcBasis(pm1, 1. - ip.x - ip.y, shape_l);

   int n = 0;
   for (int j = 0; j <= pm1; j++)
      for (int i = 0; i + j <= pm1; i++)
      {
         double s = shape_x(i)*shape_y(j)*shape_l(pm1-i-j);
         u(n,0) = s;  u(n,1) = 0;  n++;
         u(n,0) = 0;  u(n,1) = s;  n++;
      }
   for (int j = 0; j <= pm1; j++)
   {
      double s = shape_x(pm1-j)*shape_y(j);
      u(n,0) =  s*(ip.y - c);
      u(n,1) = -s*(ip.x - c);
      n++;
   }

   Ti.Mult(u, shape);
}

void ND_TriangleElement::CalcCurlShape(const IntegrationPoint &ip,
                                       DenseMatrix &curl_shape) const
{
   const int pm1 = order - 1;

#ifdef MFEM_THREAD_SAFE
   const int p = order;
   Vector shape_x(p), shape_y(p), shape_l(p);
   Vector dshape_x(p), dshape_y(p), dshape_l(p);
   Vector curlu(dof);
#endif

   poly1d.CalcBasis(pm1, ip.x, shape_x, dshape_x);
   poly1d.CalcBasis(pm1, ip.y, shape_y, dshape_y);
   poly1d.CalcBasis(pm1, 1. - ip.x - ip.y, shape_l, dshape_l);

   int n = 0;
   for (int j = 0; j <= pm1; j++)
      for (int i = 0; i + j <= pm1; i++)
      {
         int l = pm1-i-j;
         const double dx = (dshape_x(i)*shape_l(l) -
                            shape_x(i)*dshape_l(l)) * shape_y(j);
         const double dy = (dshape_y(j)*shape_l(l) -
                            shape_y(j)*dshape_l(l)) * shape_x(i);

         curlu(n++) = -dy;
         curlu(n++) =  dx;
      }

   for (int j = 0; j <= pm1; j++)
   {
      int i = pm1 - j;
      // curl of shape_x(i)*shape_y(j) * (ip.y - c, -(ip.x - c), 0):
      curlu(n++) = -((dshape_x(i)*(ip.x - c) + shape_x(i)) * shape_y(j) +
                     (dshape_y(j)*(ip.y - c) + shape_y(j)) * shape_x(i));
   }

   Vector curl2d(curl_shape.Data(),dof);
   Ti.Mult(curlu, curl2d);
}


const double ND_SegmentElement::tk[1] = { 1. };

ND_SegmentElement::ND_SegmentElement(const int p, const int ob_type)
   : VectorTensorFiniteElement(1, p, p - 1, ob_type, H_CURL,
                               DofMapType::L2_DOF_MAP),
     dof2tk(dof)
{
   if (obasis1d.IsIntegratedType()) { is_nodal = false; }

   const double *op = poly1d.OpenPoints(p - 1, ob_type);

   // set dof2tk and Nodes
   for (int i = 0; i < p; i++)
   {
      dof2tk[i] = 0;
      Nodes.IntPoint(i).x = op[i];
   }
}

void ND_SegmentElement::CalcVShape(const IntegrationPoint &ip,
                                   DenseMatrix &shape) const
{
   Vector vshape(shape.Data(), dof);

   obasis1d.Eval(ip.x, vshape);
}

const double ND_WedgeElement::tk[15] =
{ 1.,0.,0., -1.,1.,0., 0.,-1.,0., 0.,0.,1., 0.,1.,0. };

ND_WedgeElement::ND_WedgeElement(const int p,
                                 const int cb_type,
                                 const int ob_type)
   : VectorFiniteElement(3, Geometry::PRISM,
                         3 * p * ((p + 1) * (p + 2))/2, p,
                         H_CURL, FunctionSpace::Qk),
     dof2tk(dof),
     t_dof(dof),
     s_dof(dof),
     H1TriangleFE(p, cb_type),
     NDTriangleFE(p),
     H1SegmentFE(p, cb_type),
     NDSegmentFE(p, ob_type)
{
   MFEM_ASSERT(H1TriangleFE.GetDof() * NDSegmentFE.GetDof() +
               NDTriangleFE.GetDof() * H1SegmentFE.GetDof() == dof,
               "Mismatch in number of degrees of freedom "
               "when building ND_WedgeElement!");

#ifndef MFEM_THREAD_SAFE
   t1_shape.SetSize(H1TriangleFE.GetDof());
   s1_shape.SetSize(H1SegmentFE.GetDof());
   tn_shape.SetSize(NDTriangleFE.GetDof(), 2);
   sn_shape.SetSize(NDSegmentFE.GetDof(), 1);
   t1_dshape.SetSize(H1TriangleFE.GetDof(), 2);
   s1_dshape.SetSize(H1SegmentFE.GetDof(), 1);
   tn_dshape.SetSize(NDTriangleFE.GetDof(), 1);
#endif

   const int pm1 = p - 1, pm2 = p - 2;

   const IntegrationRule &t1_n = H1TriangleFE.GetNodes();
   const IntegrationRule &tn_n = NDTriangleFE.GetNodes();
   const IntegrationRule &s1_n = H1SegmentFE.GetNodes();
   const IntegrationRule &sn_n = NDSegmentFE.GetNodes();

   // edges
   int o = 0;
   for (int i = 0; i < p; i++)  // (0,1)
   {
      t_dof[o] = i; s_dof[o] = 0; dof2tk[o] = 0;
      const IntegrationPoint & t_ip = tn_n.IntPoint(t_dof[o]);
      Nodes.IntPoint(o).Set3(t_ip.x, t_ip.y, s1_n.IntPoint(s_dof[o]).x);
      o++;
   }
   for (int i = 0; i < p; i++)  // (1,2)
   {
      t_dof[o] = p + i; s_dof[o] = 0; dof2tk[o] = 1;
      const IntegrationPoint & t_ip = tn_n.IntPoint(t_dof[o]);
      Nodes.IntPoint(o).Set3(t_ip.x, t_ip.y, s1_n.IntPoint(s_dof[o]).x);
      o++;
   }
   for (int i = 0; i < p; i++)  // (2,0)
   {
      t_dof[o] = 2 * p + i; s_dof[o] = 0; dof2tk[o] = 2;
      const IntegrationPoint & t_ip = tn_n.IntPoint(t_dof[o]);
      Nodes.IntPoint(o).Set3(t_ip.x, t_ip.y, s1_n.IntPoint(s_dof[o]).x);
      o++;
   }
   for (int i = 0; i < p; i++)  // (3,4)
   {
      t_dof[o] = i; s_dof[o] = 1; dof2tk[o] = 0;
      const IntegrationPoint & t_ip = tn_n.IntPoint(t_dof[o]);
      Nodes.IntPoint(o).Set3(t_ip.x, t_ip.y, s1_n.IntPoint(s_dof[o]).x);
      o++;
   }
   for (int i = 0; i < p; i++)  // (4,5)
   {
      t_dof[o] = p + i; s_dof[o] = 1; dof2tk[o] = 1;
      const IntegrationPoint & t_ip = tn_n.IntPoint(t_dof[o]);
      Nodes.IntPoint(o).Set3(t_ip.x, t_ip.y, s1_n.IntPoint(s_dof[o]).x);
      o++;
   }
   for (int i = 0; i < p; i++)  // (5,3)
   {
      t_dof[o] = 2 * p + i; s_dof[o] = 1; dof2tk[o] = 2;
      const IntegrationPoint & t_ip = tn_n.IntPoint(t_dof[o]);
      Nodes.IntPoint(o).Set3(t_ip.x, t_ip.y, s1_n.IntPoint(s_dof[o]).x);
      o++;
   }
   for (int i = 0; i < p; i++)  // (0,3)
   {
      t_dof[o] = 0; s_dof[o] = i; dof2tk[o] = 3;
      const IntegrationPoint & t_ip = t1_n.IntPoint(t_dof[o]);
      Nodes.IntPoint(o).Set3(t_ip.x, t_ip.y, sn_n.IntPoint(s_dof[o]).x);
      o++;
   }
   for (int i = 0; i < p; i++)  // (1,4)
   {
      t_dof[o] = 1; s_dof[o] = i; dof2tk[o] = 3;
      const IntegrationPoint & t_ip = t1_n.IntPoint(t_dof[o]);
      Nodes.IntPoint(o).Set3(t_ip.x, t_ip.y, sn_n.IntPoint(s_dof[o]).x);
      o++;
   }
   for (int i = 0; i < p; i++)  // (2,5)
   {
      t_dof[o] = 2; s_dof[o] = i; dof2tk[o] = 3;
      const IntegrationPoint & t_ip = t1_n.IntPoint(t_dof[o]);
      Nodes.IntPoint(o).Set3(t_ip.x, t_ip.y, sn_n.IntPoint(s_dof[o]).x);
      o++;
   }

   // faces
   // (0,2,1) -- bottom
   int l = 0;
   for (int j = 0; j <= pm2; j++)
      for (int i = 0; i + j <= pm2; i++)
      {
         l = j + ( 2 * p - 1 - i) * i / 2;
         t_dof[o] = 3 * p + 2*l+1; s_dof[o] = 0; dof2tk[o] = 4;
         const IntegrationPoint & t_ip0 = tn_n.IntPoint(t_dof[o]);
         Nodes.IntPoint(o).Set3(t_ip0.x, t_ip0.y, s1_n.IntPoint(s_dof[o]).x);
         o++;
         t_dof[o] = 3 * p + 2*l;   s_dof[o] = 0; dof2tk[o] = 0;
         const IntegrationPoint & t_ip1 = tn_n.IntPoint(t_dof[o]);
         Nodes.IntPoint(o).Set3(t_ip1.x, t_ip1.y, s1_n.IntPoint(s_dof[o]).x);
         o++;
      }
   // (3,4,5) -- top
   int m = 0;
   for (int j = 0; j <= pm2; j++)
      for (int i = 0; i + j <= pm2; i++)
      {
         t_dof[o] = 3 * p + m; s_dof[o] = 1; dof2tk[o] = 0; m++;
         const IntegrationPoint & t_ip0 = tn_n.IntPoint(t_dof[o]);
         Nodes.IntPoint(o).Set3(t_ip0.x, t_ip0.y, s1_n.IntPoint(s_dof[o]).x);
         o++;
         t_dof[o] = 3 * p + m; s_dof[o] = 1; dof2tk[o] = 4; m++;
         const IntegrationPoint & t_ip1 = tn_n.IntPoint(t_dof[o]);
         Nodes.IntPoint(o).Set3(t_ip1.x, t_ip1.y, s1_n.IntPoint(s_dof[o]).x);
         o++;
      }
   // (0, 1, 4, 3) -- xz plane
   for (int j = 2; j <= p; j++)
      for (int i = 0; i < p; i++)
      {
         t_dof[o] = i; s_dof[o] = j; dof2tk[o] = 0;
         const IntegrationPoint & t_ip = tn_n.IntPoint(t_dof[o]);
         Nodes.IntPoint(o).Set3(t_ip.x, t_ip.y, s1_n.IntPoint(s_dof[o]).x);
         o++;
      }
   for (int j = 0; j < p; j++)
      for (int i = 0; i < pm1; i++)
      {
         t_dof[o] = 3 + i; s_dof[o] = j; dof2tk[o] = 3;
         const IntegrationPoint & t_ip = t1_n.IntPoint(t_dof[o]);
         Nodes.IntPoint(o).Set3(t_ip.x, t_ip.y, sn_n.IntPoint(s_dof[o]).x);
         o++;
      }
   // (1, 2, 5, 4) -- (y-x)z plane
   for (int j = 2; j <= p; j++)
      for (int i = 0; i < p; i++)
      {
         t_dof[o] = p + i; s_dof[o] = j; dof2tk[o] = 1;
         const IntegrationPoint & t_ip = tn_n.IntPoint(t_dof[o]);
         Nodes.IntPoint(o).Set3(t_ip.x, t_ip.y, s1_n.IntPoint(s_dof[o]).x);
         o++;
      }
   for (int j = 0; j < p; j++)
      for (int i = 0; i < pm1; i++)
      {
         t_dof[o] = p + 2 + i; s_dof[o] = j; dof2tk[o] = 3;
         const IntegrationPoint & t_ip = t1_n.IntPoint(t_dof[o]);
         Nodes.IntPoint(o).Set3(t_ip.x, t_ip.y, sn_n.IntPoint(s_dof[o]).x);
         o++;
      }
   // (2, 0, 3, 5) -- yz plane
   for (int j = 2; j <= p; j++)
      for (int i = 0; i < p; i++)
      {
         t_dof[o] = 2 * p + i; s_dof[o] = j; dof2tk[o] = 2;
         const IntegrationPoint & t_ip = tn_n.IntPoint(t_dof[o]);
         Nodes.IntPoint(o).Set3(t_ip.x, t_ip.y, s1_n.IntPoint(s_dof[o]).x);
         o++;
      }
   for (int j = 0; j < p; j++)
      for (int i = 0; i < pm1; i++)
      {
         t_dof[o] = 2 * p + 1 + i; s_dof[o] = j; dof2tk[o] = 3;
         const IntegrationPoint & t_ip = t1_n.IntPoint(t_dof[o]);
         Nodes.IntPoint(o).Set3(t_ip.x, t_ip.y, sn_n.IntPoint(s_dof[o]).x);
         o++;
      }

   // interior
   for (int k = 2; k <= p; k++)
   {
      l = 0;
      for (int j = 0; j <= pm2; j++)
         for (int i = 0; i + j <= pm2; i++)
         {
            t_dof[o] = 3 * p + l; s_dof[o] = k; dof2tk[o] = 0; l++;
            const IntegrationPoint & t_ip0 = tn_n.IntPoint(t_dof[o]);
            Nodes.IntPoint(o).Set3(t_ip0.x, t_ip0.y, s1_n.IntPoint(s_dof[o]).x);
            o++;
            t_dof[o] = 3 * p + l; s_dof[o] = k; dof2tk[o] = 4; l++;
            const IntegrationPoint & t_ip1 = tn_n.IntPoint(t_dof[o]);
            Nodes.IntPoint(o).Set3(t_ip1.x, t_ip1.y, s1_n.IntPoint(s_dof[o]).x);
            o++;
         }
   }
   for (int k = 0; k < p; k++)
   {
      l = 0;
      for (int j = 0; j < pm2; j++)
         for (int i = 0; i + j < pm2; i++)
         {
            t_dof[o] = 3 * p + l; s_dof[o] = k; dof2tk[o] = 3; l++;
            const IntegrationPoint & t_ip = t1_n.IntPoint(t_dof[o]);
            Nodes.IntPoint(o).Set3(t_ip.x, t_ip.y, sn_n.IntPoint(s_dof[o]).x);
            o++;
         }
   }
}

void ND_WedgeElement::CalcVShape(const IntegrationPoint &ip,
                                 DenseMatrix &shape) const
{
#ifdef MFEM_THREAD_SAFE
   Vector t1_shape(H1TriangleFE.GetDof());
   Vector s1_shape(H1SegmentFE.GetDof());
   DenseMatrix tn_shape(NDTriangleFE.GetDof(), 2);
   DenseMatrix sn_shape(NDSegmentFE.GetDof(), 1);
#endif

   IntegrationPoint ipz; ipz.x = ip.z; ipz.y = 0.0; ipz.z = 0.0;

   H1TriangleFE.CalcShape(ip, t1_shape);
   NDTriangleFE.CalcVShape(ip, tn_shape);
   H1SegmentFE.CalcShape(ipz, s1_shape);
   NDSegmentFE.CalcVShape(ipz, sn_shape);

   for (int i=0; i<dof; i++)
   {
      if ( dof2tk[i] != 3 )
      {
         shape(i, 0) = tn_shape(t_dof[i], 0) * s1_shape[s_dof[i]];
         shape(i, 1) = tn_shape(t_dof[i], 1) * s1_shape[s_dof[i]];
         shape(i, 2) = 0.0;
      }
      else
      {
         shape(i, 0) = 0.0;
         shape(i, 1) = 0.0;
         shape(i, 2) = t1_shape[t_dof[i]] * sn_shape(s_dof[i], 0);
      }
   }
}

void ND_WedgeElement::CalcCurlShape(const IntegrationPoint &ip,
                                    DenseMatrix &curl_shape) const
{
#ifdef MFEM_THREAD_SAFE
   Vector s1_shape(H1SegmentFE.GetDof());
   DenseMatrix t1_dshape(H1TriangleFE.GetDof(), 2);
   DenseMatrix s1_dshape(H1SegmentFE.GetDof(), 1);
   DenseMatrix tn_shape(NDTriangleFE.GetDof(), 2);
   DenseMatrix sn_shape(NDSegmentFE.GetDof(), 1);
   DenseMatrix tn_dshape(NDTriangleFE.GetDof(), 1);
#endif

   IntegrationPoint ipz; ipz.x = ip.z; ipz.y = 0.0; ipz.z = 0.0;

   H1TriangleFE.CalcDShape(ip, t1_dshape);
   H1SegmentFE.CalcShape(ipz, s1_shape);
   H1SegmentFE.CalcDShape(ipz, s1_dshape);
   NDTriangleFE.CalcVShape(ip, tn_shape);
   NDTriangleFE.CalcCurlShape(ip, tn_dshape);
   NDSegmentFE.CalcVShape(ipz, sn_shape);

   for (int i=0; i<dof; i++)
   {
      if ( dof2tk[i] != 3 )
      {
         curl_shape(i, 0) = -tn_shape(t_dof[i], 1) * s1_dshape(s_dof[i], 0);
         curl_shape(i, 1) =  tn_shape(t_dof[i], 0) * s1_dshape(s_dof[i], 0);
         curl_shape(i, 2) =  tn_dshape(t_dof[i], 0) * s1_shape[s_dof[i]];
      }
      else
      {
         curl_shape(i, 0) =  t1_dshape(t_dof[i], 1) * sn_shape(s_dof[i], 0);
         curl_shape(i, 1) = -t1_dshape(t_dof[i], 0) * sn_shape(s_dof[i], 0);
         curl_shape(i, 2) =  0.0;
      }
   }
}

ND_R1D_PointElement::ND_R1D_PointElement(int p)
   : VectorFiniteElement(1, Geometry::POINT, 2, p,
                         H_CURL, FunctionSpace::Pk)
{
   // VectorFiniteElement::SetDerivMembers doesn't support 0D H_CURL elements
   // so we mimic a 1D element and then correct the dimension here.
   dim = 0;
   vdim = 2;
   cdim = 0;
}

void ND_R1D_PointElement::CalcVShape(const IntegrationPoint &ip,
                                     DenseMatrix &shape) const
{
   shape(0,0) = 1.0;
   shape(0,1) = 0.0;

   shape(1,0) = 0.0;
   shape(1,1) = 1.0;
}

void ND_R1D_PointElement::CalcVShape(ElementTransformation &Trans,
                                     DenseMatrix &shape) const
{
   CalcVShape(Trans.GetIntPoint(), shape);
}

const double ND_R1D_SegmentElement::tk[9] = { 1.,0.,0., 0.,1.,0., 0.,0.,1. };

ND_R1D_SegmentElement::ND_R1D_SegmentElement(const int p,
                                             const int cb_type,
                                             const int ob_type)
   : VectorFiniteElement(1, Geometry::SEGMENT, 3 * p + 2, p,
                         H_CURL, FunctionSpace::Pk),
     dof2tk(dof),
     cbasis1d(poly1d.GetBasis(p, VerifyClosed(cb_type))),
     obasis1d(poly1d.GetBasis(p - 1, VerifyOpen(ob_type)))
{
   // Override default types for VectorFiniteElements
   deriv_type = CURL;
   deriv_range_type = VECTOR;
   deriv_map_type = H_DIV;

   // Override default dimensions for VectorFiniteElements
   vdim = 3;
   cdim = 3;

   const double *cp = poly1d.ClosedPoints(p, cb_type);
   const double *op = poly1d.OpenPoints(p - 1, ob_type);

#ifndef MFEM_THREAD_SAFE
   shape_cx.SetSize(p + 1);
   shape_ox.SetSize(p);
   dshape_cx.SetSize(p + 1);
#endif

   dof_map.SetSize(dof);

   int o = 0;
   // nodes
   // (0)
   Nodes.IntPoint(o).x = cp[0]; // y-directed
   dof_map[p] = o; dof2tk[o++] = 1;
   Nodes.IntPoint(o).x = cp[0]; // z-directed
   dof_map[2*p+1] = o; dof2tk[o++] = 2;

   // (1)
   Nodes.IntPoint(o).x = cp[p]; // y-directed
   dof_map[2*p] = o; dof2tk[o++] = 1;
   Nodes.IntPoint(o).x = cp[p]; // z-directed
   dof_map[3*p+1] = o; dof2tk[o++] = 2;

   // interior
   // x-components
   for (int i = 0; i < p; i++)
   {
      Nodes.IntPoint(o).x = op[i];
      dof_map[i] = o; dof2tk[o++] = 0;
   }
   // y-components
   for (int i = 1; i < p; i++)
   {
      Nodes.IntPoint(o).x = cp[i];
      dof_map[p+i] = o; dof2tk[o++] = 1;
   }
   // z-components
   for (int i = 1; i < p; i++)
   {
      Nodes.IntPoint(o).x = cp[i];
      dof_map[2*p+1+i] = o; dof2tk[o++] = 2;
   }
}

void ND_R1D_SegmentElement::CalcVShape(const IntegrationPoint &ip,
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
   for (int i = 0; i < p; i++)
   {
      int idx = dof_map[o++];
      shape(idx,0) = shape_ox(i);
      shape(idx,1) = 0.;
      shape(idx,2) = 0.;
   }
   // y-components
   for (int i = 0; i <= p; i++)
   {
      int idx = dof_map[o++];
      shape(idx,0) = 0.;
      shape(idx,1) = shape_cx(i);
      shape(idx,2) = 0.;
   }
   // z-components
   for (int i = 0; i <= p; i++)
   {
      int idx = dof_map[o++];
      shape(idx,0) = 0.;
      shape(idx,1) = 0.;
      shape(idx,2) = shape_cx(i);
   }
}

void ND_R1D_SegmentElement::CalcVShape(ElementTransformation &Trans,
                                       DenseMatrix &shape) const
{
   CalcVShape(Trans.GetIntPoint(), shape);
   const DenseMatrix & JI = Trans.InverseJacobian();
   MFEM_ASSERT(JI.Width() == 1 && JI.Height() == 1,
               "ND_R1D_SegmentElement cannot be embedded in "
               "2 or 3 dimensional spaces");
   for (int i=0; i<dof; i++)
   {
      shape(i, 0) *= JI(0,0);
   }
}

void ND_R1D_SegmentElement::CalcCurlShape(const IntegrationPoint &ip,
                                          DenseMatrix &curl_shape) const
{
   const int p = order;

#ifdef MFEM_THREAD_SAFE
   Vector shape_cx(p + 1), shape_ox(p);
   Vector dshape_cx(p + 1);
#endif

   cbasis1d.Eval(ip.x, shape_cx, dshape_cx);
   obasis1d.Eval(ip.x, shape_ox);

   int o = 0;
   // x-components
   for (int i = 0; i < p; i++)
   {
      int idx = dof_map[o++];
      curl_shape(idx,0) = 0.;
      curl_shape(idx,1) = 0.;
      curl_shape(idx,2) = 0.;
   }
   // y-components
   for (int i = 0; i <= p; i++)
   {
      int idx = dof_map[o++];
      curl_shape(idx,0) = 0.;
      curl_shape(idx,1) = 0.;
      curl_shape(idx,2) = dshape_cx(i);
   }
   // z-components
   for (int i = 0; i <= p; i++)
   {
      int idx = dof_map[o++];
      curl_shape(idx,0) = 0.;
      curl_shape(idx,1) = -dshape_cx(i);
      curl_shape(idx,2) = 0.;
   }
}

void ND_R1D_SegmentElement::CalcPhysCurlShape(ElementTransformation &Trans,
                                              DenseMatrix &curl_shape) const
{
   CalcCurlShape(Trans.GetIntPoint(), curl_shape);
   const DenseMatrix & J = Trans.Jacobian();
   MFEM_ASSERT(J.Width() == 1 && J.Height() == 1,
               "ND_R1D_SegmentElement cannot be embedded in "
               "2 or 3 dimensional spaces");
   curl_shape *= (1.0 / J.Weight());
}

void ND_R1D_SegmentElement::Project(VectorCoefficient &vc,
                                    ElementTransformation &Trans,
                                    Vector &dofs) const
{
   double data[3];
   Vector vk(data, 3);

   for (int k = 0; k < dof; k++)
   {
      Trans.SetIntPoint(&Nodes.IntPoint(k));

      vc.Eval(vk, Trans, Nodes.IntPoint(k));
      // dof_k = vk^t J tk
      Vector t(const_cast<double*>(&tk[dof2tk[k] * 3]), 3);
      dofs(k) = Trans.Jacobian()(0,0) * t(0) * vk(0) +
                t(1) * vk(1) + t(2) * vk(2);
   }

}

void ND_R1D_SegmentElement::Project(const FiniteElement &fe,
                                    ElementTransformation &Trans,
                                    DenseMatrix &I) const
{
   if (fe.GetRangeType() == SCALAR)
   {
      double vk[Geometry::MaxDim];
      Vector shape(fe.GetDof());

      double * tk_ptr = const_cast<double*>(tk);

      I.SetSize(dof, vdim*fe.GetDof());
      for (int k = 0; k < dof; k++)
      {
         const IntegrationPoint &ip = Nodes.IntPoint(k);

         Vector t1(&tk_ptr[dof2tk[k] * 3], 1);
         Vector t3(&tk_ptr[dof2tk[k] * 3], 3);

         fe.CalcShape(ip, shape);
         Trans.SetIntPoint(&ip);
         // Transform ND edge tengents from reference to physical space
         // vk = J tk
         Trans.Jacobian().Mult(t1, vk);
         vk[1] = t3[1];
         vk[2] = t3[2];
         if (fe.GetMapType() == INTEGRAL)
         {
            double w = 1.0/Trans.Weight();
            for (int d = 0; d < vdim; d++)
            {
               vk[d] *= w;
            }
         }

         for (int j = 0; j < shape.Size(); j++)
         {
            double s = shape(j);
            if (fabs(s) < 1e-12)
            {
               s = 0.0;
            }
            // Project scalar basis function multiplied by each coordinate
            // direction onto the transformed edge tangents
            for (int d = 0; d < vdim; d++)
            {
               I(k, j + d*shape.Size()) = s*vk[d];
            }
         }
      }
   }
   else
   {
      double vk[Geometry::MaxDim];
      DenseMatrix vshape(fe.GetDof(), fe.GetVDim());

      double * tk_ptr = const_cast<double*>(tk);

      I.SetSize(dof, fe.GetDof());
      for (int k = 0; k < dof; k++)
      {
         const IntegrationPoint &ip = Nodes.IntPoint(k);

         Vector t1(&tk_ptr[dof2tk[k] * 3], 1);
         Vector t3(&tk_ptr[dof2tk[k] * 3], 3);

         Trans.SetIntPoint(&ip);
         // Transform ND edge tangents from reference to physical space
         // vk = J tk
         Trans.Jacobian().Mult(t1, vk);
         // Compute fe basis functions in physical space
         fe.CalcVShape(Trans, vshape);
         // Project fe basis functions onto transformed edge tangents
         for (int j=0; j<vshape.Height(); j++)
         {
            I(k, j) = 0.0;
            I(k, j) += vshape(j, 0) * vk[0];
            if (vshape.Width() == 3)
            {
               I(k, j) += vshape(j, 1) * t3(1);
               I(k, j) += vshape(j, 2) * t3(2);
            }
         }
      }
   }
}

const double ND_R2D_SegmentElement::tk[4] = { 1.,0., 0.,1. };

ND_R2D_SegmentElement::ND_R2D_SegmentElement(const int p,
                                             const int cb_type,
                                             const int ob_type)
   : VectorFiniteElement(1, Geometry::SEGMENT, 2 * p + 1, p,
                         H_CURL, FunctionSpace::Pk),
     dof2tk(dof),
     cbasis1d(poly1d.GetBasis(p, VerifyClosed(cb_type))),
     obasis1d(poly1d.GetBasis(p - 1, VerifyOpen(ob_type)))
{
   // Override default dimensions for VectorFiniteElements
   vdim = 2;
   cdim = 1;

   const double *cp = poly1d.ClosedPoints(p, cb_type);
   const double *op = poly1d.OpenPoints(p - 1, ob_type);

#ifndef MFEM_THREAD_SAFE
   shape_cx.SetSize(p + 1);
   shape_ox.SetSize(p);
   dshape_cx.SetSize(p + 1);
#endif

   dof_map.SetSize(dof);

   int o = 0;
   // nodes
   // (0)
   Nodes.IntPoint(o).x = cp[0]; // z-directed
   dof_map[p] = o; dof2tk[o++] = 1;

   // (1)
   Nodes.IntPoint(o).x = cp[p]; // z-directed
   dof_map[2*p] = o; dof2tk[o++] = 1;

   // interior
   // x-components
   for (int i = 0; i < p; i++)
   {
      Nodes.IntPoint(o).x = op[i];
      dof_map[i] = o; dof2tk[o++] = 0;
   }
   // z-components
   for (int i = 1; i < p; i++)
   {
      Nodes.IntPoint(o).x = cp[i];
      dof_map[p+i] = o; dof2tk[o++] = 1;
   }
}

void ND_R2D_SegmentElement::CalcVShape(const IntegrationPoint &ip,
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
   for (int i = 0; i < p; i++)
   {
      int idx = dof_map[o++];
      shape(idx,0) = shape_ox(i);
      shape(idx,1) = 0.;
   }
   // z-components
   for (int i = 0; i <= p; i++)
   {
      int idx = dof_map[o++];
      shape(idx,0) = 0.;
      shape(idx,1) = shape_cx(i);
   }
}

void ND_R2D_SegmentElement::CalcVShape(ElementTransformation &Trans,
                                       DenseMatrix &shape) const
{
   CalcVShape(Trans.GetIntPoint(), shape);
   const DenseMatrix & JI = Trans.InverseJacobian();
   MFEM_ASSERT(JI.Width() == 1 && JI.Height() == 1,
               "ND_R2D_SegmentElement cannot be embedded in "
               "2 or 3 dimensional spaces");
   for (int i=0; i<dof; i++)
   {
      shape(i, 0) *= JI(0,0);
   }
}

void ND_R2D_SegmentElement::CalcCurlShape(const IntegrationPoint &ip,
                                          DenseMatrix &curl_shape) const
{
   const int p = order;

#ifdef MFEM_THREAD_SAFE
   Vector shape_cx(p + 1), shape_ox(p);
   Vector dshape_cx(p + 1);
#endif

   cbasis1d.Eval(ip.x, shape_cx, dshape_cx);
   obasis1d.Eval(ip.x, shape_ox);

   int o = 0;
   // x-components
   for (int i = 0; i < p; i++)
   {
      int idx = dof_map[o++];
      curl_shape(idx,0) = 0.;
   }
   // z-components
   for (int i = 0; i <= p; i++)
   {
      int idx = dof_map[o++];
      curl_shape(idx,0) = -dshape_cx(i);
   }
}

void ND_R2D_SegmentElement::LocalInterpolation(const VectorFiniteElement &cfe,
                                               ElementTransformation &Trans,
                                               DenseMatrix &I) const
{
   double vk[Geometry::MaxDim]; vk[1] = 0.0; vk[2] = 0.0;
   Vector xk(vk, dim);
   IntegrationPoint ip;
   DenseMatrix vshape(cfe.GetDof(), vdim);

   double * tk_ptr = const_cast<double*>(tk);

   I.SetSize(dof, vshape.Height());

   // assuming Trans is linear; this should be ok for all refinement types
   Trans.SetIntPoint(&Geometries.GetCenter(geom_type));
   const DenseMatrix &J = Trans.Jacobian();
   for (int k = 0; k < dof; k++)
   {
      Vector t1(&tk_ptr[dof2tk[k] * 2], 1);
      Vector t2(&tk_ptr[dof2tk[k] * 2], 2);

      Trans.Transform(Nodes.IntPoint(k), xk);
      ip.Set3(vk);
      cfe.CalcVShape(ip, vshape);
      // xk = J t_k
      J.Mult(t1, vk);
      // I_k = vshape_k.J.t_k, k=1,...,Dof
      for (int j = 0; j < vshape.Height(); j++)
      {
         double Ikj = 0.;
         for (int i = 0; i < dim; i++)
         {
            Ikj += vshape(j, i) * vk[i];
         }
         Ikj += vshape(j, 1) * t2(1);
         I(k, j) = (fabs(Ikj) < 1e-12) ? 0.0 : Ikj;
      }
   }
}

void ND_R2D_SegmentElement::Project(VectorCoefficient &vc,
                                    ElementTransformation &Trans,
                                    Vector &dofs) const
{
   double data[3];
   Vector vk1(data, 1);
   Vector vk2(data, 2);
   Vector vk3(data, 3);

   double * tk_ptr = const_cast<double*>(tk);

   for (int k = 0; k < dof; k++)
   {
      Trans.SetIntPoint(&Nodes.IntPoint(k));

      vc.Eval(vk3, Trans, Nodes.IntPoint(k));
      // dof_k = vk^t J tk
      Vector t1(&tk_ptr[dof2tk[k] * 2], 1);
      Vector t2(&tk_ptr[dof2tk[k] * 2], 2);

      dofs(k) = Trans.Jacobian().InnerProduct(t1, vk2) + t2(1) * vk3(2);
   }

}

ND_R2D_FiniteElement::ND_R2D_FiniteElement(int p, Geometry::Type G, int Do,
                                           const double *tk_fe)
   : VectorFiniteElement(2, G, Do, p,
                         H_CURL, FunctionSpace::Pk),
     tk(tk_fe),
     dof_map(dof),
     dof2tk(dof)
{
   // Override default types for VectorFiniteElements
   deriv_type = CURL;
   deriv_range_type = VECTOR;
   deriv_map_type = H_DIV;

   // Override default dimensions for VectorFiniteElements
   vdim = 3;
   cdim = 3;
}

void ND_R2D_FiniteElement::CalcVShape(ElementTransformation &Trans,
                                      DenseMatrix &shape) const
{
   CalcVShape(Trans.GetIntPoint(), shape);
   const DenseMatrix & JI = Trans.InverseJacobian();
   MFEM_ASSERT(JI.Width() == 2 && JI.Height() == 2,
               "ND_R2D_FiniteElement cannot be embedded in "
               "3 dimensional spaces");
   for (int i=0; i<dof; i++)
   {
      double sx = shape(i, 0);
      double sy = shape(i, 1);
      shape(i, 0) = sx * JI(0, 0) + sy * JI(1, 0);
      shape(i, 1) = sx * JI(0, 1) + sy * JI(1, 1);
   }
}

void ND_R2D_FiniteElement::CalcPhysCurlShape(ElementTransformation &Trans,
                                             DenseMatrix &curl_shape) const
{
   CalcCurlShape(Trans.GetIntPoint(), curl_shape);
   const DenseMatrix & J = Trans.Jacobian();
   MFEM_ASSERT(J.Width() == 2 && J.Height() == 2,
               "ND_R2D_FiniteElement cannot be embedded in "
               "3 dimensional spaces");
   for (int i=0; i<dof; i++)
   {
      double sx = curl_shape(i, 0);
      double sy = curl_shape(i, 1);
      curl_shape(i, 0) = sx * J(0, 0) + sy * J(0, 1);
      curl_shape(i, 1) = sx * J(1, 0) + sy * J(1, 1);
   }
   curl_shape *= (1.0 / Trans.Weight());
}

void ND_R2D_FiniteElement::LocalInterpolation(
   const VectorFiniteElement &cfe,
   ElementTransformation &Trans,
   DenseMatrix &I) const
{
   double vk[Geometry::MaxDim]; vk[2] = 0.0;
   Vector xk(vk, dim);
   IntegrationPoint ip;
#ifdef MFEM_THREAD_SAFE
   DenseMatrix vshape(cfe.GetDof(), vdim);
#else
   vshape.SetSize(cfe.GetDof(), vdim);
#endif

   double * tk_ptr = const_cast<double*>(tk);

   I.SetSize(dof, vshape.Height());

   // assuming Trans is linear; this should be ok for all refinement types
   Trans.SetIntPoint(&Geometries.GetCenter(geom_type));
   const DenseMatrix &J = Trans.Jacobian();
   for (int k = 0; k < dof; k++)
   {
      Vector t2(&tk_ptr[dof2tk[k] * 3], 2);
      Vector t3(&tk_ptr[dof2tk[k] * 3], 3);

      Trans.Transform(Nodes.IntPoint(k), xk);
      ip.Set3(vk);
      cfe.CalcVShape(ip, vshape);
      // xk = J t_k
      J.Mult(t2, vk);
      // I_k = vshape_k.J.t_k, k=1,...,Dof
      for (int j = 0; j < vshape.Height(); j++)
      {
         double Ikj = 0.;
         for (int i = 0; i < dim; i++)
         {
            Ikj += vshape(j, i) * vk[i];
         }
         Ikj += vshape(j, 2) * t3(2);
         I(k, j) = (fabs(Ikj) < 1e-12) ? 0.0 : Ikj;
      }
   }
}

void ND_R2D_FiniteElement::GetLocalRestriction(ElementTransformation &Trans,
                                               DenseMatrix &R) const
{
   double pt_data[Geometry::MaxDim];
   IntegrationPoint ip;
   Vector pt(pt_data, dim);

#ifdef MFEM_THREAD_SAFE
   DenseMatrix vshape(dof, vdim);
#endif

   double * tk_ptr = const_cast<double*>(tk);

   Trans.SetIntPoint(&Geometries.GetCenter(geom_type));
   const DenseMatrix &Jinv = Trans.InverseJacobian();
   for (int j = 0; j < dof; j++)
   {
      Vector t2(&tk_ptr[dof2tk[j] * 3], 2);
      Vector t3(&tk_ptr[dof2tk[j] * 3], 3);

      InvertLinearTrans(Trans, Nodes.IntPoint(j), pt);
      ip.Set(pt_data, dim);
      if (Geometries.CheckPoint(geom_type, ip)) // do we need an epsilon here?
      {
         CalcVShape(ip, vshape);
         Jinv.Mult(t2, pt_data);
         for (int k = 0; k < dof; k++)
         {
            double R_jk = 0.0;
            for (int d = 0; d < dim; d++)
            {
               R_jk += vshape(k,d)*pt_data[d];
            }
            R_jk += vshape(k, 2) * t3(2);
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

void ND_R2D_FiniteElement::Project(VectorCoefficient &vc,
                                   ElementTransformation &Trans,
                                   Vector &dofs) const
{
   double data[3];
   Vector vk2(data, 2);
   Vector vk3(data, 3);

   double * tk_ptr = const_cast<double*>(tk);

   for (int k = 0; k < dof; k++)
   {
      Trans.SetIntPoint(&Nodes.IntPoint(k));

      vc.Eval(vk3, Trans, Nodes.IntPoint(k));
      // dof_k = vk^t J tk
      Vector t2(&tk_ptr[dof2tk[k] * 3], 2);
      Vector t3(&tk_ptr[dof2tk[k] * 3], 3);

      dofs(k) = Trans.Jacobian().InnerProduct(t2, vk2) + t3(2) * vk3(2);
   }

}

void ND_R2D_FiniteElement::Project(const FiniteElement &fe,
                                   ElementTransformation &Trans,
                                   DenseMatrix &I) const
{
   if (fe.GetRangeType() == SCALAR)
   {
      double vk[Geometry::MaxDim];
      Vector shape(fe.GetDof());

      double * tk_ptr = const_cast<double*>(tk);

      I.SetSize(dof, vdim*fe.GetDof());
      for (int k = 0; k < dof; k++)
      {
         const IntegrationPoint &ip = Nodes.IntPoint(k);

         Vector t2(&tk_ptr[dof2tk[k] * 3], 2);
         Vector t3(&tk_ptr[dof2tk[k] * 3], 3);

         fe.CalcShape(ip, shape);
         Trans.SetIntPoint(&ip);
         // Transform ND edge tengents from reference to physical space
         // vk = J tk
         Trans.Jacobian().Mult(t2, vk);
         vk[2] = t3[2];
         if (fe.GetMapType() == INTEGRAL)
         {
            double w = 1.0/Trans.Weight();
            for (int d = 0; d < vdim; d++)
            {
               vk[d] *= w;
            }
         }

         for (int j = 0; j < shape.Size(); j++)
         {
            double s = shape(j);
            if (fabs(s) < 1e-12)
            {
               s = 0.0;
            }
            // Project scalar basis function multiplied by each coordinate
            // direction onto the transformed edge tangents
            for (int d = 0; d < vdim; d++)
            {
               I(k, j + d*shape.Size()) = s*vk[d];
            }
         }
      }
   }
   else
   {
      double vk[Geometry::MaxDim];
      DenseMatrix vshape(fe.GetDof(), fe.GetVDim());

      double * tk_ptr = const_cast<double*>(tk);

      I.SetSize(dof, fe.GetDof());
      for (int k = 0; k < dof; k++)
      {
         const IntegrationPoint &ip = Nodes.IntPoint(k);

         Vector t2(&tk_ptr[dof2tk[k] * 3], 2);
         Vector t3(&tk_ptr[dof2tk[k] * 3], 3);

         Trans.SetIntPoint(&ip);
         // Transform ND edge tangents from reference to physical space
         // vk = J tk
         Trans.Jacobian().Mult(t2, vk);
         // Compute fe basis functions in physical space
         fe.CalcVShape(Trans, vshape);
         // Project fe basis functions onto transformed edge tangents
         for (int j=0; j<vshape.Height(); j++)
         {
            I(k, j) = 0.0;
            for (int i=0; i<2; i++)
            {
               I(k, j) += vshape(j, i) * vk[i];
            }
            if (vshape.Width() == 3)
            {
               I(k, j) += vshape(j, 2) * t3(2);
            }
         }
      }
   }
}

void ND_R2D_FiniteElement::ProjectGrad(const FiniteElement &fe,
                                       ElementTransformation &Trans,
                                       DenseMatrix &grad) const
{
   MFEM_ASSERT(fe.GetMapType() == VALUE, "");

   DenseMatrix dshape(fe.GetDof(), fe.GetDim());
   Vector grad_k(fe.GetDof());

   grad.SetSize(dof, fe.GetDof());
   for (int k = 0; k < dof; k++)
   {
      fe.CalcDShape(Nodes.IntPoint(k), dshape);
      dshape.Mult(tk + dof2tk[k]*vdim, grad_k);
      for (int j = 0; j < grad_k.Size(); j++)
      {
         grad(k,j) = (fabs(grad_k(j)) < 1e-12) ? 0.0 : grad_k(j);
      }
   }
}

const double ND_R2D_TriangleElement::tk_t[15] =
{ 1.,0.,0.,  -1.,1.,0.,  0.,-1.,0.,  0.,1.,0., 0.,0.,1. };

ND_R2D_TriangleElement::ND_R2D_TriangleElement(const int p,
                                               const int cb_type)
   : ND_R2D_FiniteElement(p, Geometry::TRIANGLE, ((3*p + 1)*(p + 2))/2, tk_t),
     ND_FE(p),
     H1_FE(p, cb_type)
{
   int pm1 = p - 1, pm2 = p - 2;

#ifndef MFEM_THREAD_SAFE
   nd_shape.SetSize(ND_FE.GetDof(), 2);
   h1_shape.SetSize(H1_FE.GetDof());
   nd_dshape.SetSize(ND_FE.GetDof(), 1);
   h1_dshape.SetSize(H1_FE.GetDof(), 2);
#endif

   int o = 0;
   int n = 0;
   int h = 0;
   // Three nodes
   dof_map[o] = -1 - h++; dof2tk[o++] = 4;
   dof_map[o] = -1 - h++; dof2tk[o++] = 4;
   dof_map[o] = -1 - h++; dof2tk[o++] = 4;

   // Three edges
   for (int e=0; e<3; e++)
   {
      // Dofs in the plane
      for (int i=0; i<p; i++)
      {
         dof_map[o] = n++; dof2tk[o++] = e;
      }
      // z-directed dofs
      for (int i=0; i<pm1; i++)
      {
         dof_map[o] = -1 - h++; dof2tk[o++] = 4;
      }
   }

   // Interior dofs in the plane
   for (int j = 0; j <= pm2; j++)
      for (int i = 0; i + j <= pm2; i++)
      {
         dof_map[o] = n++; dof2tk[o++] = 0;
         dof_map[o] = n++; dof2tk[o++] = 3;
      }

   // Interior z-directed dofs
   for (int j = 0; j < pm1; j++)
      for (int i = 0; i + j < pm2; i++)
      {
         dof_map[o] = -1 - h++; dof2tk[o++] = 4;
      }

   MFEM_VERIFY(n == ND_FE.GetDof(),
               "ND_R2D_Triangle incorrect number of ND dofs.");
   MFEM_VERIFY(h == H1_FE.GetDof(),
               "ND_R2D_Triangle incorrect number of H1 dofs.");
   MFEM_VERIFY(o == GetDof(),
               "ND_R2D_Triangle incorrect number of dofs.");

   const IntegrationRule & nd_Nodes = ND_FE.GetNodes();
   const IntegrationRule & h1_Nodes = H1_FE.GetNodes();

   for (int i=0; i<dof; i++)
   {
      int idx = dof_map[i];
      if (idx >= 0)
      {
         const IntegrationPoint & ip = nd_Nodes.IntPoint(idx);
         Nodes.IntPoint(i).Set2(ip.x, ip.y);
      }
      else
      {
         const IntegrationPoint & ip = h1_Nodes.IntPoint(-idx-1);
         Nodes.IntPoint(i).Set2(ip.x, ip.y);
      }
   }
}

void ND_R2D_TriangleElement::CalcVShape(const IntegrationPoint &ip,
                                        DenseMatrix &shape) const
{
#ifdef MFEM_THREAD_SAFE
   DenseMatrix nd_shape(ND_FE.GetDof(), 2);
   Vector      h1_shape(H1_FE.GetDof());
#endif

   ND_FE.CalcVShape(ip, nd_shape);
   H1_FE.CalcShape(ip, h1_shape);

   for (int i=0; i<dof; i++)
   {
      int idx = dof_map[i];
      if (idx >= 0)
      {
         shape(i, 0) = nd_shape(idx, 0);
         shape(i, 1) = nd_shape(idx, 1);
         shape(i, 2) = 0.0;
      }
      else
      {
         shape(i, 0) = 0.0;
         shape(i, 1) = 0.0;
         shape(i, 2) = h1_shape(-idx-1);
      }
   }
}

void ND_R2D_TriangleElement::CalcCurlShape(const IntegrationPoint &ip,
                                           DenseMatrix &curl_shape) const
{
#ifdef MFEM_THREAD_SAFE
   DenseMatrix nd_dshape(ND_FE.GetDof(), 1);
   DenseMatrix h1_dshape(H1_FE.GetDof(), 2);
#endif

   ND_FE.CalcCurlShape(ip, nd_dshape);
   H1_FE.CalcDShape(ip, h1_dshape);

   for (int i=0; i<dof; i++)
   {
      int idx = dof_map[i];
      if (idx >= 0)
      {
         curl_shape(i, 0) = 0.0;
         curl_shape(i, 1) = 0.0;
         curl_shape(i, 2) = nd_dshape(idx, 0);
      }
      else
      {
         curl_shape(i, 0) = h1_dshape(-idx-1, 1);
         curl_shape(i, 1) = -h1_dshape(-idx-1, 0);
         curl_shape(i, 2) = 0.0;
      }
   }
}


const double ND_R2D_QuadrilateralElement::tk_q[15] =
{ 1.,0.,0.,  0.,1.,0., -1.,0.,0., 0.,-1.,0., 0.,0.,1. };

ND_R2D_QuadrilateralElement::ND_R2D_QuadrilateralElement(const int p,
                                                         const int cb_type,
                                                         const int ob_type)
   : ND_R2D_FiniteElement(p, Geometry::SQUARE, ((3*p + 1)*(p + 1)), tk_q),
     cbasis1d(poly1d.GetBasis(p, VerifyClosed(cb_type))),
     obasis1d(poly1d.GetBasis(p - 1, VerifyOpen(ob_type)))
{
   const double *cp = poly1d.ClosedPoints(p, cb_type);
   const double *op = poly1d.OpenPoints(p - 1, ob_type);
   const int dofx = p*(p+1);
   const int dofy = p*(p+1);
   const int dofxy = dofx+dofy;

#ifndef MFEM_THREAD_SAFE
   shape_cx.SetSize(p + 1);
   shape_ox.SetSize(p);
   shape_cy.SetSize(p + 1);
   shape_oy.SetSize(p);
   dshape_cx.SetSize(p + 1);
   dshape_cy.SetSize(p + 1);
#endif

   dof_map.SetSize(dof);

   int o = 0;
   // nodes
   dof_map[dofxy] = o++;   // (0)
   dof_map[dofxy+p] = o++; // (1)
   dof_map[dof-1] = o++;   // (2)
   dof_map[dof-p-1] = o++; // (3)

   // edges
   for (int i = 0; i < p; i++)  // (0,1) - x-directed
   {
      dof_map[i + 0*p] = o++;
   }
   for (int i = 1; i < p; i++)  // (0,1) - z-directed
   {
      dof_map[dofxy + i + 0*(p+1)] = o++;
   }
   for (int j = 0; j < p; j++)  // (1,2) - y-directed
   {
      dof_map[dofx + p + j*(p + 1)] = o++;
   }
   for (int j = 1; j < p; j++)  // (1,2) - z-directed
   {
      dof_map[dofxy + p + j*(p + 1)] = o++;
   }
   for (int i = 0; i < p; i++)  // (2,3) - x-directed
   {
      dof_map[(p - 1 - i) + p*p] = -1 - (o++);
   }
   for (int i = 1; i < p; i++)  // (2,3) - z-directed
   {
      dof_map[dofxy + (p - i) + p*(p + 1)] = o++;
   }
   for (int j = 0; j < p; j++)  // (3,0) - y-directed
   {
      dof_map[dofx + 0 + (p - 1 - j)*(p + 1)] = -1 - (o++);
   }
   for (int j = 1; j < p; j++)  // (3,0) - z-directed
   {
      dof_map[dofxy + (p - j)*(p + 1)] = o++;
   }

   // interior
   // x-components
   for (int j = 1; j < p; j++)
      for (int i = 0; i < p; i++)
      {
         dof_map[i + j*p] = o++;
      }
   // y-components
   for (int j = 0; j < p; j++)
      for (int i = 1; i < p; i++)
      {
         dof_map[dofx + i + j*(p + 1)] = o++;
      }
   // z-components
   for (int j = 1; j < p; j++)
      for (int i = 1; i < p; i++)
      {
         dof_map[dofxy + i + j*(p + 1)] = o++;
      }

   // set dof2tk and Nodes
   o = 0;
   // x-components
   for (int j = 0; j <= p; j++)
      for (int i = 0; i < p; i++)
      {
         int idx;
         if ((idx = dof_map[o++]) < 0)
         {
            dof2tk[idx = -1 - idx] = 2;
         }
         else
         {
            dof2tk[idx] = 0;
         }
         Nodes.IntPoint(idx).Set2(op[i], cp[j]);
      }
   // y-components
   for (int j = 0; j < p; j++)
      for (int i = 0; i <= p; i++)
      {
         int idx;
         if ((idx = dof_map[o++]) < 0)
         {
            dof2tk[idx = -1 - idx] = 3;
         }
         else
         {
            dof2tk[idx] = 1;
         }
         Nodes.IntPoint(idx).Set2(cp[i], op[j]);
      }
   // z-components
   for (int j = 0; j <= p; j++)
      for (int i = 0; i <= p; i++)
      {
         int idx = dof_map[o++];
         dof2tk[idx] = 4;
         Nodes.IntPoint(idx).Set2(cp[i], cp[j]);
      }
}

void ND_R2D_QuadrilateralElement::CalcVShape(const IntegrationPoint &ip,
                                             DenseMatrix &shape) const
{
   const int p = order;

#ifdef MFEM_THREAD_SAFE
   Vector shape_cx(p + 1), shape_ox(p), shape_cy(p + 1), shape_oy(p);
#endif

   cbasis1d.Eval(ip.x, shape_cx);
   obasis1d.Eval(ip.x, shape_ox);
   cbasis1d.Eval(ip.y, shape_cy);
   obasis1d.Eval(ip.y, shape_oy);

   int o = 0;
   // x-components
   for (int j = 0; j <= p; j++)
      for (int i = 0; i < p; i++)
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
         shape(idx,0) = s*shape_ox(i)*shape_cy(j);
         shape(idx,1) = 0.;
         shape(idx,2) = 0.;
      }
   // y-components
   for (int j = 0; j < p; j++)
      for (int i = 0; i <= p; i++)
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
         shape(idx,1) = s*shape_cx(i)*shape_oy(j);
         shape(idx,2) = 0.;
      }
   // z-components
   for (int j = 0; j <= p; j++)
      for (int i = 0; i <= p; i++)
      {
         int idx = dof_map[o++];
         shape(idx,0) = 0.;
         shape(idx,1) = 0.;
         shape(idx,2) = shape_cx(i)*shape_cy(j);
      }
}

void ND_R2D_QuadrilateralElement::CalcCurlShape(const IntegrationPoint &ip,
                                                DenseMatrix &curl_shape) const
{
   const int p = order;

#ifdef MFEM_THREAD_SAFE
   Vector shape_cx(p + 1), shape_ox(p), shape_cy(p + 1), shape_oy(p);
   Vector dshape_cx(p + 1), dshape_cy(p + 1);
#endif

   cbasis1d.Eval(ip.x, shape_cx, dshape_cx);
   obasis1d.Eval(ip.x, shape_ox);
   cbasis1d.Eval(ip.y, shape_cy, dshape_cy);
   obasis1d.Eval(ip.y, shape_oy);

   int o = 0;
   // x-components
   for (int j = 0; j <= p; j++)
      for (int i = 0; i < p; i++)
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
         curl_shape(idx,0) = 0.;
         curl_shape(idx,1) = 0.;
         curl_shape(idx,2) = -s*shape_ox(i)*dshape_cy(j);
      }
   // y-components
   for (int j = 0; j < p; j++)
      for (int i = 0; i <= p; i++)
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
         curl_shape(idx,0) = 0.;
         curl_shape(idx,1) = 0.;
         curl_shape(idx,2) =  s*dshape_cx(i)*shape_oy(j);
      }
   // z-components
   for (int j = 0; j <= p; j++)
      for (int i = 0; i <= p; i++)
      {
         int idx = dof_map[o++];
         curl_shape(idx,0) =  shape_cx(i)*dshape_cy(j);
         curl_shape(idx,1) = -dshape_cx(i)*shape_cy(j);
         curl_shape(idx,2) = 0.;
      }
}

}
