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

// Nedelec Finite Element classes

#include "fe_nd.hpp"
#include "face_map_utils.hpp"
#include "../coefficient.hpp"

namespace mfem
{

using namespace std;

const real_t ND_HexahedronElement::tk[18] =
{ 1.,0.,0.,  0.,1.,0.,  0.,0.,1., -1.,0.,0.,  0.,-1.,0.,  0.,0.,-1. };

ND_HexahedronElement::ND_HexahedronElement(const int p,
                                           const int cb_type, const int ob_type)
   : VectorTensorFiniteElement(3, 3*p*(p + 1)*(p + 1), p, cb_type, ob_type,
                               H_CURL, DofMapType::L2_DOF_MAP),
     dof2tk(dof), cp(poly1d.ClosedPoints(p, cb_type))
{
   if (obasis1d.IsIntegratedType()) { is_nodal = false; }

   dof_map.SetSize(dof);

   const real_t *op = poly1d.OpenPoints(p - 1, ob_type);
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
   real_t vk[Geometry::MaxDim];
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
               const real_t h = cp[id1+1] - cp[id1];

               real_t val = 0.0;

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
                  const real_t ipval = Trans.Jacobian().InnerProduct(tk + dof2tk[idx]*dim, vk);
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
#endif

   if (obasis1d.IsIntegratedType())
   {
#ifdef MFEM_THREAD_SAFE
      Vector dshape_cx(p + 1), dshape_cy(p + 1), dshape_cz(p + 1);
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

void ND_HexahedronElement::GetFaceMap(const int face_id,
                                      Array<int> &face_map) const
{
   const int p = order;
   const int pp1 = p + 1;
   const int n_face_dofs_per_component = p*pp1;
   const int n_dof_per_dim = p*pp1*pp1;

   std::vector<int> n_dofs = {p, pp1, pp1, p};
   std::vector<int> offsets, strides;

   const auto f = internal::GetFaceNormal3D(face_id);
   const int face_normal = f.first, level = f.second;
   if (face_normal == 0) // x-normal
   {
      offsets =
      {
         n_dof_per_dim + (level ? pp1 - 1 : 0),
         2*n_dof_per_dim + (level ? pp1 - 1 : 0)
      };
      strides = {pp1, p*pp1, pp1, pp1*pp1};
   }
   else if (face_normal == 1) // y-normal
   {
      offsets =
      {
         level ? p*(pp1 - 1) : 0,
         2*n_dof_per_dim + (level ? pp1*(pp1 - 1) : 0)
      };
      strides = {1, p*pp1, 1, pp1*pp1};
   }
   else if (face_normal == 2) // z-normal
   {
      offsets =
      {
         level ? p*pp1*(pp1 - 1) : 0,
         n_dof_per_dim + (level ? p*pp1*(pp1 - 1) : 0)
      };
      strides = {1, p, 1, pp1};
   }

   internal::FillFaceMap(n_face_dofs_per_component, offsets, strides, n_dofs,
                         face_map);
}

const real_t ND_QuadrilateralElement::tk[8] =
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

   const real_t *op = poly1d.OpenPoints(p - 1, ob_type);
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
   real_t vk[Geometry::MaxDim];
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

         const real_t h = cp[i+1] - cp[i];

         real_t val = 0.0;

         for (int k = 0; k < nqpt; k++)
         {
            const IntegrationPoint &ip1d = ir.IntPoint(k);

            ip2d.Set2(cp[i] + (h*ip1d.x), cp[j]);

            Trans.SetIntPoint(&ip2d);
            vc.Eval(xk, Trans, ip2d);

            // xk^t J tk
            const real_t ipval = Trans.Jacobian().InnerProduct(tk + dof2tk[idx]*dim, vk);
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

         const real_t h = cp[j+1] - cp[j];

         real_t val = 0.0;

         for (int k = 0; k < nqpt; k++)
         {
            const IntegrationPoint &ip1d = ir.IntPoint(k);

            ip2d.Set2(cp[i], cp[j] + (h*ip1d.x));

            Trans.SetIntPoint(&ip2d);
            vc.Eval(xk, Trans, ip2d);

            // xk^t J tk
            const real_t ipval = Trans.Jacobian().InnerProduct(tk + dof2tk[idx]*dim, vk);
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
#endif

   if (obasis1d.IsIntegratedType())
   {
#ifdef MFEM_THREAD_SAFE
      Vector dshape_cx(p + 1), dshape_cy(p + 1);
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

void ND_QuadrilateralElement::GetFaceMap(const int face_id,
                                         Array<int> &face_map) const
{
   const int p = order;
   const int pp1 = order + 1;
   const int n_face_dofs_per_component = p;
   std::vector<int> strides = {(face_id == 0 || face_id == 2) ? 1 : pp1};
   std::vector<int> n_dofs = {p};
   std::vector<int> offsets;
   switch (face_id)
   {
      case 0: offsets = {0}; break; // y = 0
      case 1: offsets = {p*pp1 + pp1 - 1}; break; // x = 1
      case 2: offsets = {p*(pp1 - 1)}; break; // y = 1
      case 3: offsets = {p*pp1}; break; // x = 0
   }
   internal::FillFaceMap(n_face_dofs_per_component, offsets, strides, n_dofs,
                         face_map);
}


const real_t ND_TetrahedronElement::tk[18] =
{ 1.,0.,0.,  0.,1.,0.,  0.,0.,1.,  -1.,1.,0.,  -1.,0.,1.,  0.,-1.,1. };

const real_t ND_TetrahedronElement::c = 1./4.;

ND_TetrahedronElement::ND_TetrahedronElement(const int p)
   : VectorFiniteElement(3, Geometry::TETRAHEDRON, p*(p + 2)*(p + 3)/2, p,
                         H_CURL, FunctionSpace::Pk), dof2tk(dof), doftrans(p)
{
   const real_t *eop = poly1d.OpenPoints(p - 1);
   const real_t *fop = (p > 1) ? poly1d.OpenPoints(p - 2) : NULL;
   const real_t *iop = (p > 2) ? poly1d.OpenPoints(p - 3) : NULL;

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
         real_t w = fop[i] + fop[j] + fop[pm2-i-j];
         Nodes.IntPoint(o).Set3(fop[pm2-i-j]/w, fop[i]/w, fop[j]/w);
         dof2tk[o++] = 3;
         Nodes.IntPoint(o).Set3(fop[pm2-i-j]/w, fop[i]/w, fop[j]/w);
         dof2tk[o++] = 4;
      }
   for (int j = 0; j <= pm2; j++)  // (0,3,2)
      for (int i = 0; i + j <= pm2; i++)
      {
         real_t w = fop[i] + fop[j] + fop[pm2-i-j];
         Nodes.IntPoint(o).Set3(0., fop[j]/w, fop[i]/w);
         dof2tk[o++] = 2;
         Nodes.IntPoint(o).Set3(0., fop[j]/w, fop[i]/w);
         dof2tk[o++] = 1;
      }
   for (int j = 0; j <= pm2; j++)  // (0,1,3)
      for (int i = 0; i + j <= pm2; i++)
      {
         real_t w = fop[i] + fop[j] + fop[pm2-i-j];
         Nodes.IntPoint(o).Set3(fop[i]/w, 0., fop[j]/w);
         dof2tk[o++] = 0;
         Nodes.IntPoint(o).Set3(fop[i]/w, 0., fop[j]/w);
         dof2tk[o++] = 2;
      }
   for (int j = 0; j <= pm2; j++)  // (0,2,1)
      for (int i = 0; i + j <= pm2; i++)
      {
         real_t w = fop[i] + fop[j] + fop[pm2-i-j];
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
            real_t w = iop[i] + iop[j] + iop[k] + iop[pm3-i-j-k];
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
      const real_t *tm = tk + 3*dof2tk[m];
      o = 0;

      poly1d.CalcBasis(pm1, ip.x, shape_x);
      poly1d.CalcBasis(pm1, ip.y, shape_y);
      poly1d.CalcBasis(pm1, ip.z, shape_z);
      poly1d.CalcBasis(pm1, 1. - ip.x - ip.y - ip.z, shape_l);

      for (int k = 0; k <= pm1; k++)
         for (int j = 0; j + k <= pm1; j++)
            for (int i = 0; i + j + k <= pm1; i++)
            {
               real_t s = shape_x(i)*shape_y(j)*shape_z(k)*shape_l(pm1-i-j-k);
               T(o++, m) = s * tm[0];
               T(o++, m) = s * tm[1];
               T(o++, m) = s * tm[2];
            }
      for (int k = 0; k <= pm1; k++)
         for (int j = 0; j + k <= pm1; j++)
         {
            real_t s = shape_x(pm1-j-k)*shape_y(j)*shape_z(k);
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
            real_t s = shape_x(i)*shape_y(j)*shape_z(k)*shape_l(pm1-i-j-k);
            u(n,0) =  s;  u(n,1) = 0.;  u(n,2) = 0.;  n++;
            u(n,0) = 0.;  u(n,1) =  s;  u(n,2) = 0.;  n++;
            u(n,0) = 0.;  u(n,1) = 0.;  u(n,2) =  s;  n++;
         }
   for (int k = 0; k <= pm1; k++)
      for (int j = 0; j + k <= pm1; j++)
      {
         real_t s = shape_x(pm1-j-k)*shape_y(j)*shape_z(k);
         u(n,0) = s*(ip.y - c);  u(n,1) = -s*(ip.x - c);  u(n,2) =  0.;  n++;
         u(n,0) = s*(ip.z - c);  u(n,1) =  0.;  u(n,2) = -s*(ip.x - c);  n++;
      }
   for (int k = 0; k <= pm1; k++)
   {
      real_t s = shape_y(pm1-k)*shape_z(k);
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
            const real_t dx = (dshape_x(i)*shape_l(l) -
                               shape_x(i)*dshape_l(l))*shape_y(j)*shape_z(k);
            const real_t dy = (dshape_y(j)*shape_l(l) -
                               shape_y(j)*dshape_l(l))*shape_x(i)*shape_z(k);
            const real_t dz = (dshape_z(k)*shape_l(l) -
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


const real_t ND_TriangleElement::tk[8] =
{ 1.,0.,  -1.,1.,  0.,-1.,  0.,1. };

const real_t ND_TriangleElement::c = 1./3.;

ND_TriangleElement::ND_TriangleElement(const int p)
   : VectorFiniteElement(2, Geometry::TRIANGLE, p*(p + 2), p,
                         H_CURL, FunctionSpace::Pk),
     dof2tk(dof), doftrans(p)
{
   const real_t *eop = poly1d.OpenPoints(p - 1);
   const real_t *iop = (p > 1) ? poly1d.OpenPoints(p - 2) : NULL;

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
         real_t w = iop[i] + iop[j] + iop[pm2-i-j];
         Nodes.IntPoint(n).Set2(iop[i]/w, iop[j]/w);
         dof2tk[n++] = 0;
         Nodes.IntPoint(n).Set2(iop[i]/w, iop[j]/w);
         dof2tk[n++] = 3;
      }

   DenseMatrix T(dof);
   for (int m = 0; m < dof; m++)
   {
      const IntegrationPoint &ip = Nodes.IntPoint(m);
      const real_t *tm = tk + 2*dof2tk[m];
      n = 0;

      poly1d.CalcBasis(pm1, ip.x, shape_x);
      poly1d.CalcBasis(pm1, ip.y, shape_y);
      poly1d.CalcBasis(pm1, 1. - ip.x - ip.y, shape_l);

      for (int j = 0; j <= pm1; j++)
         for (int i = 0; i + j <= pm1; i++)
         {
            real_t s = shape_x(i)*shape_y(j)*shape_l(pm1-i-j);
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
         real_t s = shape_x(i)*shape_y(j)*shape_l(pm1-i-j);
         u(n,0) = s;  u(n,1) = 0;  n++;
         u(n,0) = 0;  u(n,1) = s;  n++;
      }
   for (int j = 0; j <= pm1; j++)
   {
      real_t s = shape_x(pm1-j)*shape_y(j);
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
         const real_t dx = (dshape_x(i)*shape_l(l) -
                            shape_x(i)*dshape_l(l)) * shape_y(j);
         const real_t dy = (dshape_y(j)*shape_l(l) -
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


const real_t ND_SegmentElement::tk[1] = { 1. };

ND_SegmentElement::ND_SegmentElement(const int p, const int ob_type)
   : VectorTensorFiniteElement(1, p, p - 1, ob_type, H_CURL,
                               DofMapType::L2_DOF_MAP),
     dof2tk(dof)
{
   if (obasis1d.IsIntegratedType()) { is_nodal = false; }

   const real_t *op = poly1d.OpenPoints(p - 1, ob_type);

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

const real_t ND_WedgeElement::tk[15] =
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
     doftrans(p),
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

const real_t ND_FuentesPyramidElement::tk[27] =
{
   1., 0., 0.,   0., 1., 0.,  0., 0., 1.,
   -1., 0., 1.,  -1.,-1., 1.,  0.,-1., 1.,
   -1., 0., 0.,   0.,-1., 0., -M_SQRT1_2,-M_SQRT1_2,M_SQRT2
};

ND_FuentesPyramidElement::ND_FuentesPyramidElement(const int p,
                                                   const int cb_type,
                                                   const int ob_type)
   : VectorFiniteElement(3, Geometry::PYRAMID, p * (3 * p * p + 5), p,
                         H_CURL, FunctionSpace::Uk),
     dof2tk(dof), doftrans(p)
{
   zmax = 0.0;

   const real_t *eop = poly1d.OpenPoints(p - 1);
   const real_t *top = (p > 1) ? poly1d.OpenPoints(p - 2) : NULL;
   const real_t *qop = poly1d.OpenPoints(p - 1, ob_type);
   const real_t *qcp = poly1d.ClosedPoints(p, cb_type);

   const int pm2 = p - 2;

#ifndef MFEM_THREAD_SAFE
   tmp_E_E_ij.SetSize(p, dim);
   tmp_dE_E_ij.SetSize(p, dim);
   tmp_E_Q1_ijk.SetSize(p, p + 1, dim);
   tmp_dE_Q1_ijk.SetSize(p, p + 1, dim);
   tmp_E_Q2_ijk.SetSize(p, p + 1, dim);
   tmp_dE_Q2_ijk.SetSize(p, p + 1, dim);
   tmp_E_T_ijk.SetSize(p - 1, p, dim);
   tmp_dE_T_ijk.SetSize(p - 1, p, dim);
   tmp_phi_Q1_ij.SetSize(p + 1, p + 1);
   tmp_dphi_Q1_ij.SetSize(p + 1, p + 1, dim);
   tmp_phi_Q2_ij.SetSize(p + 1, p + 1);
   tmp_dphi_Q2_ij.SetSize(p + 1, p + 1, dim);
   tmp_phi_E_i.SetSize(p + 1);
   tmp_dphi_E_i.SetSize(p + 1, dim);
   u.SetSize(dof, dim);
   curlu.SetSize(dof, dim);
#else
   DenseMatrix tmp_E_E_ij(p, dim);
   DenseTensor tmp_E_Q1_ijk(p, p + 1, dim);
   DenseTensor tmp_dE_Q1_ijk(p, p + 1, dim);
   DenseTensor tmp_E_Q2_ijk(p, p + 1, dim);
   DenseTensor tmp_dE_Q2_ijk(p, p + 1, dim);
   DenseTensor tmp_E_T_ijk(p - 1, p, dim);
   DenseTensor tmp_dE_T_ijk(p - 1, p, dim);
   DenseMatrix tmp_phi_Q1_ij(p + 1, p + 1);
   DenseTensor tmp_dphi_Q1_ij(p + 1, p + 1, dim);
   DenseMatrix tmp_phi_Q2_ij(p + 1, p + 1);
   Vector      tmp_phi_E_i(p + 1);
   DenseMatrix tmp_dphi_E_i(p + 1, dim);
   DenseMatrix u(dof, dim);
#endif

   int o = 0;

   // edges
   for (int i = 0; i < p; i++) // (0, 1)
   {
      Nodes.IntPoint(o).Set3(eop[i], 0., 0.);
      dof2tk[o++] = 0;
   }
   for (int i = 0; i < p; i++) // (1, 2)
   {
      Nodes.IntPoint(o).Set3(1., eop[i], 0.);
      dof2tk[o++] = 1;
   }
   for (int i = 0; i < p; i++) // (3, 2)
   {
      Nodes.IntPoint(o).Set3(eop[i], 1., 0.);
      dof2tk[o++] = 0;
   }
   for (int i = 0; i < p; i++) // (0, 3)
   {
      Nodes.IntPoint(o).Set3(0., eop[i], 0.);
      dof2tk[o++] = 1;
   }
   for (int i = 0; i < p; i++) // (0, 4)
   {
      Nodes.IntPoint(o).Set3(0., 0., eop[i]);
      dof2tk[o++] = 2;
   }
   for (int i = 0; i < p; i++) // (1, 4)
   {
      Nodes.IntPoint(o).Set3(1. - eop[i], 0., eop[i]);
      dof2tk[o++] = 3;
   }
   for (int i = 0; i < p; i++) // (2, 4)
   {
      Nodes.IntPoint(o).Set3(1. - eop[i], 1. - eop[i], eop[i]);
      dof2tk[o++] = 4;
   }
   for (int i = 0; i < p; i++) // (3, 4)
   {
      Nodes.IntPoint(o).Set3(0., 1. - eop[i], eop[i]);
      dof2tk[o++] = 5;
   }

   // quadrilateral face (3, 2, 1, 0)
   // x-components
   for (int j = 1; j < p; j++)
      for (int i = 0; i < p; i++)
      {
         Nodes.IntPoint(o).Set3(qop[i], qcp[p-j], 0.);
         dof2tk[o++] = 0; // (1 0 0)
      }

   // y-components
   for (int j = 0; j < p; j++)
      for (int i = 1; i < p; i++)
      {
         Nodes.IntPoint(o).Set3(qcp[i], qop[p-1-j], 0.);
         dof2tk[o++] = 7; // (0 -1 0)
      }

   // triangular faces
   for (int j = 0; j <= pm2; j++)  // (0, 1, 4)
      for (int i = 0; i + j <= pm2; i++)
      {
         real_t w = top[i] + top[j] + top[pm2-i-j];
         Nodes.IntPoint(o).Set3(top[i]/w, 0., top[j]/w);
         dof2tk[o++] = 0;
         Nodes.IntPoint(o).Set3(top[i]/w, 0., top[j]/w);
         dof2tk[o++] = 2;
      }
   for (int j = 0; j <= pm2; j++)  // (1, 2, 4)
      for (int i = 0; i + j <= pm2; i++)
      {
         real_t w = top[i] + top[j] + top[pm2-i-j];
         Nodes.IntPoint(o).Set3((top[i] + top[pm2-i-j])/w, top[i]/w, top[j]/w);
         dof2tk[o++] = 1;
         Nodes.IntPoint(o).Set3((top[i] + top[pm2-i-j])/w, top[i]/w, top[j]/w);
         dof2tk[o++] = 3;
      }
   for (int j = 0; j <= pm2; j++)  // (2, 3, 4)
      for (int i = 0; i + j <= pm2; i++)
      {
         real_t w = top[i] + top[j] + top[pm2-i-j];
         Nodes.IntPoint(o).Set3(top[pm2-i-j]/w, (top[i] + top[pm2-i-j])/w,
                                top[j]/w);
         dof2tk[o++] = 6;
         Nodes.IntPoint(o).Set3(top[pm2-i-j]/w, (top[i] + top[pm2-i-j])/w,
                                top[j]/w);
         dof2tk[o++] = 4;
      }
   for (int j = 0; j <= pm2; j++)  // (3, 0, 4)
      for (int i = 0; i + j <= pm2; i++)
      {
         real_t w = top[i] + top[j] + top[pm2-i-j];
         Nodes.IntPoint(o).Set3(0., top[pm2-i-j]/w, top[j]/w);
         dof2tk[o++] = 7;
         Nodes.IntPoint(o).Set3(0., top[pm2-i-j]/w, top[j]/w);
         dof2tk[o++] = 5;
      }

   // interior
   // x-components
   for (int k = 1; k < p; k++)
      for (int j = 1; j < p; j++)
         for (int i = 0; i < p; i++)
         {
            real_t w = 1.0 - qcp[k];
            Nodes.IntPoint(o).Set3(qop[i]*w, qcp[j]*w, qcp[k]);
            dof2tk[o++] = 0;
         }
   // y-components
   for (int k = 1; k < p; k++)
      for (int j = 0; j < p; j++)
         for (int i = 1; i < p; i++)
         {
            real_t w = 1.0 - qcp[k];
            Nodes.IntPoint(o).Set3(qcp[i]*w, qop[j]*w, qcp[k]);
            dof2tk[o++] = 1;
         }
   // z-components
   for (int k = 0; k < p; k++)
      for (int j = 1; j < p; j++)
         for (int i = 1; i < p; i++)
         {
            real_t w = 1.0 - qop[k];
            Nodes.IntPoint(o).Set3(qcp[i]*w, qcp[j]*w, qop[k]);
            dof2tk[o++] = 8;
         }

   DenseMatrix T(dof);

   for (int m = 0; m < dof; m++)
   {
      const IntegrationPoint &ip = Nodes.IntPoint(m);
      calcBasis(p, ip, tmp_E_E_ij, tmp_E_Q1_ijk, tmp_E_Q2_ijk, tmp_E_T_ijk,
                tmp_phi_Q1_ij, tmp_dphi_Q1_ij, tmp_phi_Q2_ij,
                tmp_phi_E_i, tmp_dphi_E_i, u);

      const Vector tm({tk[3*dof2tk[m]], tk[3*dof2tk[m]+1], tk[3*dof2tk[m]+2]});
      u.Mult(tm, T.GetColumn(m));
   }

   Ti.Factor(T);
}

void ND_FuentesPyramidElement::CalcVShape(const IntegrationPoint &ip,
                                          DenseMatrix &shape) const
{
   const int p = order;

#ifdef MFEM_THREAD_SAFE
   DenseMatrix tmp_E_E_ij(p, dim);
   DenseTensor tmp_E_Q1_ijk(p, p + 1, dim);
   DenseTensor tmp_E_Q2_ijk(p, p + 1, dim);
   DenseTensor tmp_E_T_ijk(p - 1, p, dim);
   DenseMatrix tmp_phi_Q1_ij(p + 1, p + 1);
   DenseTensor tmp_dphi_Q1_ij(p + 1, p + 1, dim);
   DenseMatrix tmp_phi_Q2_ij(p + 1, p + 1);
   Vector      tmp_phi_E_i(p + 1);
   DenseMatrix tmp_dphi_E_i(p + 1, dim);
   DenseMatrix u(dof, dim);
#endif

   calcBasis(p, ip, tmp_E_E_ij, tmp_E_Q1_ijk, tmp_E_Q2_ijk, tmp_E_T_ijk,
             tmp_phi_Q1_ij, tmp_dphi_Q1_ij, tmp_phi_Q2_ij,
             tmp_phi_E_i, tmp_dphi_E_i, u);

   Ti.Mult(u, shape);
}

void ND_FuentesPyramidElement::CalcCurlShape(const IntegrationPoint &ip,
                                             DenseMatrix &curl_shape) const
{
   const int p = order;

#ifdef MFEM_THREAD_SAFE
   DenseMatrix tmp_E_E_ij(p, dim);
   DenseMatrix tmp_dE_E_ij(p, dim);
   DenseTensor tmp_E_Q1_ijk(p, p + 1, dim);
   DenseTensor tmp_dE_Q1_ijk(p, p + 1, dim);
   DenseTensor tmp_E_Q2_ijk(p, p + 1, dim);
   DenseTensor tmp_dE_Q2_ijk(p, p + 1, dim);
   DenseTensor tmp_E_T_ijk(p - 1, p, dim);
   DenseTensor tmp_dE_T_ijk(p - 1, p, dim);
   DenseMatrix tmp_phi_Q2_ij(p + 1, p + 1);
   DenseTensor tmp_dphi_Q2_ij(p + 1, p + 1, dim);
   Vector      tmp_phi_E_i(p + 1);
   DenseMatrix tmp_dphi_E_i(p + 1, dim);
   DenseMatrix curlu(dof, dim);
#endif

   calcCurlBasis(p, ip, tmp_E_E_ij, tmp_dE_E_ij, tmp_E_Q1_ijk, tmp_dE_Q1_ijk,
                 tmp_E_Q2_ijk, tmp_dE_Q2_ijk, tmp_E_T_ijk, tmp_dE_T_ijk,
                 tmp_phi_Q2_ij, tmp_dphi_Q2_ij, tmp_phi_E_i, tmp_dphi_E_i,
                 curlu);

   Ti.Mult(curlu, curl_shape);
}

void ND_FuentesPyramidElement::CalcRawVShape(const IntegrationPoint &ip,
                                             DenseMatrix &shape) const
{
   const int p = order;

#ifdef MFEM_THREAD_SAFE
   DenseMatrix tmp_E_E_ij(p, dim);
   DenseTensor tmp_E_Q1_ijk(p, p + 1, dim);
   DenseTensor tmp_E_Q2_ijk(p, p + 1, dim);
   DenseTensor tmp_E_T_ijk(p - 1, p, dim);
   DenseMatrix tmp_phi_Q1_ij(p + 1, p + 1);
   DenseTensor tmp_dphi_Q1_ij(p + 1, p + 1, dim);
   DenseMatrix tmp_phi_Q2_ij(p + 1, p + 1);
   Vector      tmp_phi_E_i(p + 1);
   DenseMatrix tmp_dphi_E_i(p + 1, dim);
#endif

   calcBasis(p, ip, tmp_E_E_ij, tmp_E_Q1_ijk, tmp_E_Q2_ijk, tmp_E_T_ijk,
             tmp_phi_Q1_ij, tmp_dphi_Q1_ij, tmp_phi_Q2_ij,
             tmp_phi_E_i, tmp_dphi_E_i, shape);
}

void ND_FuentesPyramidElement::CalcRawCurlShape(const IntegrationPoint &ip,
                                                DenseMatrix &dshape) const
{
   const int p = order;

#ifdef MFEM_THREAD_SAFE
   DenseMatrix tmp_E_E_ij(p, dim);
   DenseMatrix tmp_dE_E_ij(p, dim);
   DenseTensor tmp_E_Q1_ijk(p, p + 1, dim);
   DenseTensor tmp_dE_Q1_ijk(p, p + 1, dim);
   DenseTensor tmp_E_Q2_ijk(p, p + 1, dim);
   DenseTensor tmp_dE_Q2_ijk(p, p + 1, dim);
   DenseTensor tmp_E_T_ijk(p - 1, p, dim);
   DenseTensor tmp_dE_T_ijk(p - 1, p, dim);
   DenseMatrix tmp_phi_Q2_ij(p + 1, p + 1);
   DenseTensor tmp_dphi_Q2_ij(p + 1, p + 1, dim);
   Vector      tmp_phi_E_i(p + 1);
   DenseMatrix tmp_dphi_E_i(p + 1, dim);
#endif

   calcCurlBasis(p, ip, tmp_E_E_ij, tmp_dE_E_ij, tmp_E_Q1_ijk, tmp_dE_Q1_ijk,
                 tmp_E_Q2_ijk, tmp_dE_Q2_ijk, tmp_E_T_ijk, tmp_dE_T_ijk,
                 tmp_phi_Q2_ij, tmp_dphi_Q2_ij, tmp_phi_E_i, tmp_dphi_E_i,
                 dshape);
}

void ND_FuentesPyramidElement::calcBasis(const int p,
                                         const IntegrationPoint &ip,
                                         DenseMatrix & E_E_ik,
                                         DenseTensor & E_Q1_ijk,
                                         DenseTensor & E_Q2_ijk,
                                         DenseTensor & E_T_ijk,
                                         DenseMatrix & phi_Q1_ij,
                                         DenseTensor & dphi_Q1_ij,
                                         DenseMatrix & phi_Q2_ij,
                                         Vector      & phi_E_k,
                                         DenseMatrix & dphi_E_k,
                                         DenseMatrix &W) const
{
   real_t x = ip.x;
   real_t y = ip.y;
   real_t z = ip.z;
   Vector xy({x,y}), dmu(3);
   real_t mu, mu2;

   if (std::fabs(1.0 - z) < apex_tol)
   {
      z = 1.0 - apex_tol;
      y = 0.5 * (1.0 - z);
      x = 0.5 * (1.0 - z);
      xy(0) = x; xy(1) = y;
   }
   zmax = std::max(z, zmax);

   W = 0.0;

   int o = 0;

   // Mixed Edges
   if (z < 1.0)
   {
      // (a, b) = (1, 2), c = 0
      mu = mu0(z, xy, 2);
      E_E(p, nu01(z, xy, 1), nu01_grad_nu01(z, xy, 1), E_E_ik);
      for (int i=0; i<p; i++, o++)
         for (int k=0; k<3; k++)
         {
            W(o, k) = mu * E_E_ik(i, k);
         }

      // (a, b) = (1, 2), c = 1
      mu = mu1(z, xy, 2);
      for (int i=0; i<p; i++, o++)
         for (int k=0; k<3; k++)
         {
            W(o, k) = mu * E_E_ik(i, k);
         }

      // (a, b) = (2, 1), c = 0
      mu = mu0(z, xy, 1);
      E_E(p, nu01(z, xy, 2), nu01_grad_nu01(z, xy, 2), E_E_ik);
      for (int i=0; i<p; i++, o++)
         for (int k=0; k<3; k++)
         {
            W(o, k) = mu * E_E_ik(i, k);
         }

      // (a, b) = (2, 1), c = 1
      mu = mu1(z, xy, 1);
      for (int i=0; i<p; i++, o++)
         for (int k=0; k<3; k++)
         {
            W(o, k) = mu * E_E_ik(i, k);
         }
   }

   // Triangle Edges
   if (z < 1.0)
   {
      E_E(p, lam15(x, y, z), lam15_grad_lam15(x, y, z), E_E_ik);
      for (int i=0; i<p; i++, o++)
         for (int k=0; k<3; k++)
         {
            W(o, k) = E_E_ik(i, k);
         }

      E_E(p, lam25(x, y, z), lam25_grad_lam25(x, y, z), E_E_ik);
      for (int i=0; i<p; i++, o++)
         for (int k=0; k<3; k++)
         {
            W(o, k) = E_E_ik(i, k);
         }

      E_E(p, lam35(x, y, z), lam35_grad_lam35(x, y, z), E_E_ik);
      for (int i=0; i<p; i++, o++)
         for (int k=0; k<3; k++)
         {
            W(o, k) = E_E_ik(i, k);
         }

      E_E(p, lam45(x, y, z), lam45_grad_lam45(x, y, z), E_E_ik);
      for (int i=0; i<p; i++, o++)
         for (int k=0; k<3; k++)
         {
            W(o, k) = E_E_ik(i, k);
         }
   }

   // Quadrilateral Face
   if (z < 1.0 && p >= 2)
   {
      mu = mu0(z);
      mu2 = mu * mu;

      // Family I
      E_Q(p, mu01(z, xy, 1), mu01_grad_mu01(z, xy, 1), mu01(z, xy, 2),
          E_Q1_ijk);
      for (int j=2; j<=p; j++)
         for (int i=0; i<p; i++, o++)
            for (int k=0; k<3; k++)
            {
               W(o, k) = mu2 * E_Q1_ijk(i, j, k);
            }

      // Family II
      E_Q(p, mu01(z, xy, 2), mu01_grad_mu01(z, xy, 2), mu01(z, xy, 1),
          E_Q2_ijk);
      for (int j=2; j<=p; j++)
         for (int i=0; i<p; i++, o++)
            for (int k=0; k<3; k++)
            {
               W(o, k) = mu2 * E_Q2_ijk(i, j, k);
            }
   }

   // Triangular Faces
   if (z < 1.0 && p >= 2)
   {
      // Family I
      // (a, b) = (1, 2), c = 0
      mu = mu0(z, xy, 2);
      E_T(p, nu012(z, xy, 1), nu01_grad_nu01(z, xy, 1), E_T_ijk);
      for (int j=1; j<p; j++)
         for (int i=0; i+j<p; i++, o++)
            for (int k=0; k<3; k++)
            {
               W(o, k) = mu * E_T_ijk(i, j, k);
            }

      // (a, b) = (1, 2), c = 1
      mu = mu1(z, xy, 2);
      for (int j=1; j<p; j++)
         for (int i=0; i+j<p; i++, o++)
            for (int k=0; k<3; k++)
            {
               W(o, k) = mu * E_T_ijk(i, j, k);
            }

      // (a, b) = (2, 1), c = 0
      mu = mu0(z, xy, 1);
      E_T(p, nu012(z, xy, 2), nu01_grad_nu01(z, xy, 2), E_T_ijk);
      for (int j=1; j<p; j++)
         for (int i=0; i+j<p; i++, o++)
            for (int k=0; k<3; k++)
            {
               W(o, k) = mu * E_T_ijk(i, j, k);
            }

      // (a, b) = (2, 1), c = 1
      mu = mu1(z, xy, 1);
      for (int j=1; j<p; j++)
         for (int i=0; i+j<p; i++, o++)
            for (int k=0; k<3; k++)
            {
               W(o, k) = mu * E_T_ijk(i, j, k);
            }

      // Family II
      // (a, b) = (1, 2), c = 0
      mu = mu0(z, xy, 2);
      E_T(p, nu120(z, xy, 1), nu12_grad_nu12(z, xy, 1), E_T_ijk);
      for (int j=1; j<p; j++)
         for (int i=0; i+j<p; i++, o++)
            for (int k=0; k<3; k++)
            {
               W(o, k) = mu * E_T_ijk(i, j, k);
            }

      // (a, b) = (1, 2), c = 1
      mu = mu1(z, xy, 2);
      for (int j=1; j<p; j++)
         for (int i=0; i+j<p; i++, o++)
            for (int k=0; k<3; k++)
            {
               W(o, k) = mu * E_T_ijk(i, j, k);
            }

      // (a, b) = (2, 1), c = 0
      mu = mu0(z, xy, 1);
      E_T(p, nu120(z, xy, 2), nu12_grad_nu12(z, xy, 2), E_T_ijk);
      for (int j=1; j<p; j++)
         for (int i=0; i+j<p; i++, o++)
            for (int k=0; k<3; k++)
            {
               W(o, k) = mu * E_T_ijk(i, j, k);
            }

      // (a, b) = (2, 1), c = 1
      mu = mu1(z, xy, 1);
      for (int j=1; j<p; j++)
         for (int i=0; i+j<p; i++, o++)
            for (int k=0; k<3; k++)
            {
               W(o, k) = mu * E_T_ijk(i, j, k);
            }
   }

   // Interior
   if (z < 1.0 && p >= 2)
   {
      // Family I
      phi_Q(p, mu01(z, xy, 1), grad_mu01(z, xy, 1), mu01(z, xy, 2),
            grad_mu01(z, xy, 2), phi_Q1_ij, dphi_Q1_ij);
      phi_E(p, mu01(z), grad_mu01(z), phi_E_k, dphi_E_k);
      for (int k=2; k<=p; k++)
         for (int j=2; j<=p; j++)
            for (int i=2; i<=p; i++, o++)
               for (int l=0; l<3; l++)
                  W(o, l) = dphi_Q1_ij(i, j, l) * phi_E_k(k) +
                            phi_Q1_ij(i, j) * dphi_E_k(k, l);

      // Family II
      mu = mu0(z);
      for (int k=2; k<=p; k++)
         for (int j=2; j<=p; j++)
            for (int i=0; i<p; i++, o++)
               for (int l=0; l<3; l++)
               {
                  W(o, l) = mu * E_Q1_ijk(i, j, l) * phi_E_k(k);
               }

      // Family III
      for (int k=2; k<=p; k++)
         for (int j=2; j<=p; j++)
            for (int i=0; i<p; i++, o++)
               for (int l=0; l<3; l++)
               {
                  W(o, l) = mu * E_Q2_ijk(i, j, l) * phi_E_k(k);
               }

      // Family IV
      // Re-using mu from Family II
      dmu = grad_mu0(z);
      phi_Q(p, mu01(z, xy, 2), mu01(z, xy, 1), phi_Q2_ij);
      for (int j=2; j<=p; j++)
         for (int i=2; i<=p; i++, o++)
         {
            const int n = std::max(i,j);
            const real_t nmu = n * pow(mu, n-1);
            for (int l=0; l<3; l++)
            {
               W(o, l) = nmu * phi_Q2_ij(i, j) * dmu(l);
            }
         }
   }
}

void ND_FuentesPyramidElement::calcCurlBasis(const int p,
                                             const IntegrationPoint &ip,
                                             DenseMatrix & E_E_ik,
                                             DenseMatrix & dE_E_ik,
                                             DenseTensor & E_Q1_ijk,
                                             DenseTensor & dE_Q1_ijk,
                                             DenseTensor & E_Q2_ijk,
                                             DenseTensor & dE_Q2_ijk,
                                             DenseTensor & E_T_ijk,
                                             DenseTensor & dE_T_ijk,
                                             DenseMatrix & phi_Q2_ij,
                                             DenseTensor & dphi_Q2_ij,
                                             Vector      & phi_E_k,
                                             DenseMatrix & dphi_E_k,
                                             DenseMatrix & dW) const
{
   real_t x = ip.x;
   real_t y = ip.y;
   real_t z = ip.z;
   Vector xy({x,y}), dmu(3);
   Vector dmuxE(3), E(3), dphi(3), muphi(3);

   real_t mu, mu2;

   if (std::fabs(1.0 - z) < apex_tol)
   {
      z = 1.0 - apex_tol;
      y = 0.5 * (1.0 - z);
      x = 0.5 * (1.0 - z);
      xy(0) = x; xy(1) = y;
   }
   zmax = std::max(z, zmax);

   dW = 0.0;

   int o = 0;

   // Mixed Edges
   if (z < 1.0)
   {
      // (a, b) = (1, 2), c = 0
      mu = mu0(z, xy, 2);
      dmu = grad_mu0(z, xy, 2);
      E_E(p, nu01(z, xy, 1), grad_nu01(z, xy, 1), E_E_ik, dE_E_ik);
      for (int i=0; i<p; i++, o++)
      {
         E(0) = E_E_ik(i, 0); E(1) = E_E_ik(i, 1); E(2) = E_E_ik(i, 2);
         dmu.cross3D(E, dmuxE);
         for (int k=0; k<3; k++)
         {
            dW(o, k) = mu * dE_E_ik(i, k) + dmuxE(k);
         }
      }

      // (a, b) = (1, 2), c = 1
      mu = mu1(z, xy, 2);
      dmu = grad_mu1(z, xy, 2);
      for (int i=0; i<p; i++, o++)
      {
         E(0) = E_E_ik(i, 0); E(1) = E_E_ik(i, 1); E(2) = E_E_ik(i, 2);
         dmu.cross3D(E, dmuxE);
         for (int k=0; k<3; k++)
         {
            dW(o, k) = mu * dE_E_ik(i, k) + dmuxE(k);
         }
      }

      // (a, b) = (2, 1), c = 0
      mu = mu0(z, xy, 1);
      dmu = grad_mu0(z, xy, 1);
      E_E(p, nu01(z, xy, 2), grad_nu01(z, xy, 2), E_E_ik, dE_E_ik);
      for (int i=0; i<p; i++, o++)
      {
         E(0) = E_E_ik(i, 0); E(1) = E_E_ik(i, 1); E(2) = E_E_ik(i, 2);
         dmu.cross3D(E, dmuxE);
         for (int k=0; k<3; k++)
         {
            dW(o, k) = mu * dE_E_ik(i, k) + dmuxE(k);
         }
      }

      // (a, b) = (2, 1), c = 1
      mu = mu1(z, xy, 1);
      dmu = grad_mu1(z, xy, 1);
      for (int i=0; i<p; i++, o++)
      {
         E(0) = E_E_ik(i, 0); E(1) = E_E_ik(i, 1); E(2) = E_E_ik(i, 2);
         dmu.cross3D(E, dmuxE);
         for (int k=0; k<3; k++)
         {
            dW(o, k) = mu * dE_E_ik(i, k) + dmuxE(k);
         }
      }
   }

   // Triangle Edges
   if (z < 1.0)
   {
      E_E(p, lam15(x, y, z), grad_lam15(x, y, z), E_E_ik, dE_E_ik);
      for (int i=0; i<p; i++, o++)
         for (int k=0; k<3; k++)
         {
            dW(o, k) = dE_E_ik(i, k);
         }

      E_E(p, lam25(x, y, z), grad_lam25(x, y, z), E_E_ik, dE_E_ik);
      for (int i=0; i<p; i++, o++)
         for (int k=0; k<3; k++)
         {
            dW(o, k) = dE_E_ik(i, k);
         }

      E_E(p, lam35(x, y, z), grad_lam35(x, y, z), E_E_ik, dE_E_ik);
      for (int i=0; i<p; i++, o++)
         for (int k=0; k<3; k++)
         {
            dW(o, k) = dE_E_ik(i, k);
         }

      E_E(p, lam45(x, y, z), grad_lam45(x, y, z), E_E_ik, dE_E_ik);
      for (int i=0; i<p; i++, o++)
         for (int k=0; k<3; k++)
         {
            dW(o, k) = dE_E_ik(i, k);
         }
   }

   // Quadrilateral Face
   if (z < 1.0 && p >= 2)
   {
      mu = mu0(z);
      mu2 = mu * mu;
      dmu = grad_mu0(z);

      // Family I
      E_Q(p, mu01(z, xy, 1), grad_mu01(z, xy, 1),
          mu01(z, xy, 2), grad_mu01(z, xy, 2), E_Q1_ijk, dE_Q1_ijk);
      for (int j=2; j<=p; j++)
         for (int i=0; i<p; i++, o++)
         {
            E(0) = E_Q1_ijk(i, j, 0);
            E(1) = E_Q1_ijk(i, j, 1);
            E(2) = E_Q1_ijk(i, j, 2);
            dmu.cross3D(E, dmuxE);
            for (int k=0; k<3; k++)
            {
               dW(o, k) = mu2 * dE_Q1_ijk(i, j, k) + 2.0 * mu * dmuxE(k);
            }
         }

      // Family II
      E_Q(p, mu01(z, xy, 2), grad_mu01(z, xy, 2),
          mu01(z, xy, 1), grad_mu01(z, xy, 1), E_Q2_ijk, dE_Q2_ijk);
      for (int j=2; j<=p; j++)
         for (int i=0; i<p; i++, o++)
         {
            E(0) = E_Q2_ijk(i, j, 0);
            E(1) = E_Q2_ijk(i, j, 1);
            E(2) = E_Q2_ijk(i, j, 2);
            dmu.cross3D(E, dmuxE);
            for (int k=0; k<3; k++)
            {
               dW(o, k) = mu2 * dE_Q2_ijk(i, j, k) + 2.0 * mu * dmuxE(k);
            }
         }
   }

   // Triangular Faces
   if (z < 1.0 && p >= 2)
   {
      // Family I
      // (a, b) = (1, 2), c = 0
      mu = mu0(z, xy, 2);
      dmu = grad_mu0(z, xy, 2);
      E_T(p, nu012(z, xy, 1), grad_nu012(z, xy, 1), E_T_ijk, dE_T_ijk);
      for (int j=1; j<p; j++)
         for (int i=0; i+j<p; i++, o++)
         {
            E(0) = E_T_ijk(i, j, 0);
            E(1) = E_T_ijk(i, j, 1);
            E(2) = E_T_ijk(i, j, 2);
            dmu.cross3D(E, dmuxE);
            for (int k=0; k<3; k++)
            {
               dW(o, k) = mu * dE_T_ijk(i, j, k) + dmuxE(k);
            }
         }

      // (a, b) = (1, 2), c = 1
      mu = mu1(z, xy, 2);
      dmu = grad_mu1(z, xy, 2);
      for (int j=1; j<p; j++)
         for (int i=0; i+j<p; i++, o++)
         {
            E(0) = E_T_ijk(i, j, 0);
            E(1) = E_T_ijk(i, j, 1);
            E(2) = E_T_ijk(i, j, 2);
            dmu.cross3D(E, dmuxE);
            for (int k=0; k<3; k++)
            {
               dW(o, k) = mu * dE_T_ijk(i, j, k) + dmuxE(k);
            }
         }

      // (a, b) = (2, 1), c = 0
      mu = mu0(z, xy, 1);
      dmu = grad_mu0(z, xy, 1);
      E_T(p, nu012(z, xy, 2), grad_nu012(z, xy, 2), E_T_ijk, dE_T_ijk);
      for (int j=1; j<p; j++)
         for (int i=0; i+j<p; i++, o++)
         {
            E(0) = E_T_ijk(i, j, 0);
            E(1) = E_T_ijk(i, j, 1);
            E(2) = E_T_ijk(i, j, 2);
            dmu.cross3D(E, dmuxE);
            for (int k=0; k<3; k++)
            {
               dW(o, k) = mu * dE_T_ijk(i, j, k) + dmuxE(k);
            }
         }

      // (a, b) = (2, 1), c = 1
      mu = mu1(z, xy, 1);
      dmu = grad_mu1(z, xy, 1);
      for (int j=1; j<p; j++)
         for (int i=0; i+j<p; i++, o++)
         {
            E(0) = E_T_ijk(i, j, 0);
            E(1) = E_T_ijk(i, j, 1);
            E(2) = E_T_ijk(i, j, 2);
            dmu.cross3D(E, dmuxE);
            for (int k=0; k<3; k++)
            {
               dW(o, k) = mu * dE_T_ijk(i, j, k) + dmuxE(k);
            }
         }

      // Family II
      // (a, b) = (1, 2), c = 0
      mu = mu0(z, xy, 2);
      dmu = grad_mu0(z, xy, 2);
      E_T(p, nu120(z, xy, 1), grad_nu120(z, xy, 1), E_T_ijk, dE_T_ijk);
      for (int j=1; j<p; j++)
         for (int i=0; i+j<p; i++, o++)
         {
            E(0) = E_T_ijk(i, j, 0);
            E(1) = E_T_ijk(i, j, 1);
            E(2) = E_T_ijk(i, j, 2);
            dmu.cross3D(E, dmuxE);
            for (int k=0; k<3; k++)
            {
               dW(o, k) = mu * dE_T_ijk(i, j, k) + dmuxE(k);
            }
         }

      // (a, b) = (1, 2), c = 1
      mu = mu1(z, xy, 2);
      dmu = grad_mu1(z, xy, 2);
      for (int j=1; j<p; j++)
         for (int i=0; i+j<p; i++, o++)
         {
            E(0) = E_T_ijk(i, j, 0);
            E(1) = E_T_ijk(i, j, 1);
            E(2) = E_T_ijk(i, j, 2);
            dmu.cross3D(E, dmuxE);
            for (int k=0; k<3; k++)
            {
               dW(o, k) = mu * dE_T_ijk(i, j, k) + dmuxE(k);
            }
         }

      // (a, b) = (2, 1), c = 0
      mu = mu0(z, xy, 1);
      dmu = grad_mu0(z, xy, 1);
      E_T(p, nu120(z, xy, 2), grad_nu120(z, xy, 2), E_T_ijk, dE_T_ijk);
      for (int j=1; j<p; j++)
         for (int i=0; i+j<p; i++, o++)
         {
            E(0) = E_T_ijk(i, j, 0);
            E(1) = E_T_ijk(i, j, 1);
            E(2) = E_T_ijk(i, j, 2);
            dmu.cross3D(E, dmuxE);
            for (int k=0; k<3; k++)
            {
               dW(o, k) = mu * dE_T_ijk(i, j, k) + dmuxE(k);
            }
         }

      // (a, b) = (2, 1), c = 1
      mu = mu1(z, xy, 1);
      dmu = grad_mu1(z, xy, 1);
      for (int j=1; j<p; j++)
         for (int i=0; i+j<p; i++, o++)
         {
            E(0) = E_T_ijk(i, j, 0);
            E(1) = E_T_ijk(i, j, 1);
            E(2) = E_T_ijk(i, j, 2);
            dmu.cross3D(E, dmuxE);
            for (int k=0; k<3; k++)
            {
               dW(o, k) = mu * dE_T_ijk(i, j, k) + dmuxE(k);
            }
         }
   }

   // Interior
   if (z < 1.0 && p >= 2)
   {
      // Family I
      // Curl is zero so skip these functions
      o += (p - 1) * (p - 1) * (p - 1);

      // Family II
      mu = mu0(z);
      dmu = grad_mu0(z);
      phi_E(p, mu01(z), grad_mu01(z), phi_E_k, dphi_E_k);
      for (int k=2; k<=p; k++)
      {
         dphi(0) = dphi_E_k(k, 0);
         dphi(1) = dphi_E_k(k, 1);
         dphi(2) = dphi_E_k(k, 2);
         add(mu, dphi, phi_E_k(k), dmu, muphi);

         for (int j=2; j<=p; j++)
            for (int i=0; i<p; i++, o++)
            {
               E(0) = E_Q1_ijk(i, j, 0);
               E(1) = E_Q1_ijk(i, j, 1);
               E(2) = E_Q1_ijk(i, j, 2);
               muphi.cross3D(E, dmuxE);
               for (int l=0; l<3; l++)
               {
                  dW(o, l) = mu * dE_Q1_ijk(i, j, l) * phi_E_k(k) + dmuxE(l);
               }
            }
      }

      // Family III
      for (int k=2; k<=p; k++)
      {
         dphi(0) = dphi_E_k(k, 0);
         dphi(1) = dphi_E_k(k, 1);
         dphi(2) = dphi_E_k(k, 2);
         add(mu, dphi, phi_E_k(k), dmu, muphi);

         for (int j=2; j<=p; j++)
            for (int i=0; i<p; i++, o++)
            {
               E(0) = E_Q2_ijk(i, j, 0);
               E(1) = E_Q2_ijk(i, j, 1);
               E(2) = E_Q2_ijk(i, j, 2);
               muphi.cross3D(E, dmuxE);
               for (int l=0; l<3; l++)
               {
                  dW(o, l) = mu * dE_Q2_ijk(i, j, l) * phi_E_k(k) + dmuxE(l);
               }
            }
      }

      // Family IV
      // Re-using mu from Family II
      dmu = grad_mu0(z);
      phi_Q(p, mu01(z, xy, 2), grad_mu01(z, xy, 2), mu01(z, xy, 1),
            grad_mu01(z, xy, 1), phi_Q2_ij, dphi_Q2_ij);
      for (int j=2; j<=p; j++)
         for (int i=2; i<=p; i++, o++)
         {
            const int n = std::max(i,j);
            const real_t nmu = n * pow(mu, n-1);

            dphi(0) = dphi_Q2_ij(i, j, 0);
            dphi(1) = dphi_Q2_ij(i, j, 1);
            dphi(2) = dphi_Q2_ij(i, j, 2);
            dphi.cross3D(dmu, muphi);

            for (int l=0; l<3; l++)
            {
               dW(o, l) = nmu * muphi(l);
            }
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

const real_t ND_R1D_SegmentElement::tk[9] = { 1.,0.,0., 0.,1.,0., 0.,0.,1. };

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

   const real_t *cp = poly1d.ClosedPoints(p, cb_type);
   const real_t *op = poly1d.OpenPoints(p - 1, ob_type);

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
   real_t data[3];
   Vector vk(data, 3);

   for (int k = 0; k < dof; k++)
   {
      Trans.SetIntPoint(&Nodes.IntPoint(k));

      vc.Eval(vk, Trans, Nodes.IntPoint(k));
      // dof_k = vk^t J tk
      Vector t(const_cast<real_t*>(&tk[dof2tk[k] * 3]), 3);
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
      real_t vk[Geometry::MaxDim];
      Vector shape(fe.GetDof());

      real_t * tk_ptr = const_cast<real_t*>(tk);

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
            real_t w = 1.0/Trans.Weight();
            for (int d = 0; d < vdim; d++)
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
      real_t vk[Geometry::MaxDim];
      DenseMatrix vshape(fe.GetDof(), fe.GetRangeDim());

      real_t * tk_ptr = const_cast<real_t*>(tk);

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

const real_t ND_R2D_SegmentElement::tk[4] = { 1.,0., 0.,1. };

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

   const real_t *cp = poly1d.ClosedPoints(p, cb_type);
   const real_t *op = poly1d.OpenPoints(p - 1, ob_type);

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
   real_t vk[Geometry::MaxDim]; vk[1] = 0.0; vk[2] = 0.0;
   Vector xk(vk, dim);
   IntegrationPoint ip;
   DenseMatrix vshape(cfe.GetDof(), vdim);

   real_t * tk_ptr = const_cast<real_t*>(tk);

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
         real_t Ikj = 0.;
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
   real_t data[3];
   Vector vk1(data, 1);
   Vector vk2(data, 2);
   Vector vk3(data, 3);

   real_t * tk_ptr = const_cast<real_t*>(tk);

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
                                           const real_t *tk_fe)
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
      real_t sx = shape(i, 0);
      real_t sy = shape(i, 1);
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
      real_t sx = curl_shape(i, 0);
      real_t sy = curl_shape(i, 1);
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
   real_t vk[Geometry::MaxDim]; vk[2] = 0.0;
   Vector xk(vk, dim);
   IntegrationPoint ip;
#ifdef MFEM_THREAD_SAFE
   DenseMatrix vshape(cfe.GetDof(), vdim);
#else
   vshape.SetSize(cfe.GetDof(), vdim);
#endif

   real_t * tk_ptr = const_cast<real_t*>(tk);

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
         real_t Ikj = 0.;
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
   real_t pt_data[Geometry::MaxDim];
   IntegrationPoint ip;
   Vector pt(pt_data, dim);

#ifdef MFEM_THREAD_SAFE
   DenseMatrix vshape(dof, vdim);
#endif

   real_t * tk_ptr = const_cast<real_t*>(tk);

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
            real_t R_jk = 0.0;
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
   real_t data[3];
   Vector vk2(data, 2);
   Vector vk3(data, 3);

   real_t * tk_ptr = const_cast<real_t*>(tk);

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
      real_t vk[Geometry::MaxDim];
      Vector shape(fe.GetDof());

      real_t * tk_ptr = const_cast<real_t*>(tk);

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
            real_t w = 1.0/Trans.Weight();
            for (int d = 0; d < vdim; d++)
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
      real_t vk[Geometry::MaxDim];
      DenseMatrix vshape(fe.GetDof(), fe.GetRangeDim());

      real_t * tk_ptr = const_cast<real_t*>(tk);

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

const real_t ND_R2D_TriangleElement::tk_t[15] =
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


const real_t ND_R2D_QuadrilateralElement::tk_q[15] =
{ 1.,0.,0.,  0.,1.,0., -1.,0.,0., 0.,-1.,0., 0.,0.,1. };

ND_R2D_QuadrilateralElement::ND_R2D_QuadrilateralElement(const int p,
                                                         const int cb_type,
                                                         const int ob_type)
   : ND_R2D_FiniteElement(p, Geometry::SQUARE, ((3*p + 1)*(p + 1)), tk_q),
     cbasis1d(poly1d.GetBasis(p, VerifyClosed(cb_type))),
     obasis1d(poly1d.GetBasis(p - 1, VerifyOpen(ob_type)))
{
   const real_t *cp = poly1d.ClosedPoints(p, cb_type);
   const real_t *op = poly1d.OpenPoints(p - 1, ob_type);
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
