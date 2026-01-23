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

// H1 Finite Element classes utilizing the Bernstein basis

#include "fe_nurbs.hpp"
#include "../../mesh/nurbs.hpp"

namespace mfem
{

using namespace std;

void NURBS1DFiniteElement::SetOrder() const
{
   order = kv[0]->GetOrder();
   dof = order + 1;

   weights.SetSize(dof);
   shape_x.SetSize(dof);
}

void NURBS1DFiniteElement::CalcShape(const IntegrationPoint &ip,
                                     Vector &shape) const
{
   kv[0]->CalcShape(shape, ijk[0], ip.x);

   real_t sum = 0.0;
   for (int i = 0; i <= order; i++)
   {
      sum += (shape(i) *= weights(i));
   }

   shape /= sum;
}

void NURBS1DFiniteElement::CalcDShape(const IntegrationPoint &ip,
                                      DenseMatrix &dshape) const
{
   Vector grad(dshape.Data(), dof);

   kv[0]->CalcShape (shape_x, ijk[0], ip.x);
   kv[0]->CalcDShape(grad,    ijk[0], ip.x);

   real_t sum = 0.0, dsum = 0.0;
   for (int i = 0; i <= order; i++)
   {
      sum  += (shape_x(i) *= weights(i));
      dsum += (   grad(i) *= weights(i));
   }

   sum = 1.0/sum;
   add(sum, grad, -dsum*sum*sum, shape_x, grad);
}

void NURBS1DFiniteElement::CalcHessian (const IntegrationPoint &ip,
                                        DenseMatrix &hessian) const
{
   Vector grad(dof);
   Vector hess(hessian.Data(), dof);

   kv[0]->CalcShape (shape_x,  ijk[0], ip.x);
   kv[0]->CalcDShape(grad,     ijk[0], ip.x);
   kv[0]->CalcD2Shape(hess,    ijk[0], ip.x);

   real_t sum = 0.0, dsum = 0.0, d2sum = 0.0;
   for (int i = 0; i <= order; i++)
   {
      sum   += (shape_x(i) *= weights(i));
      dsum  += (   grad(i) *= weights(i));
      d2sum += (   hess(i) *= weights(i));
   }

   sum = 1.0/sum;
   add(sum, hess, -2*dsum*sum*sum, grad, hess);
   add(1.0, hess, (-d2sum + 2*dsum*dsum*sum)*sum*sum, shape_x, hess);
}

void NURBS1DFiniteElement::Project(Coefficient &coeff,
                                   ElementTransformation &Trans,
                                   Vector &dofs) const
{
   IntegrationPoint ip;

   for (int i = 0; i <= order; i++)
   {
      real_t kx = kv[0]->GetBotella(ijk[0] + i);
      if (!kv[0]->inSpan(kx, ijk[0]+order)) { continue; }
      ip.x = kv[0]->GetRefPoint(kx, ijk[0]+order);

      Trans.SetIntPoint(&ip);
      dofs(i) = coeff.Eval(Trans, ip);
   }
}

void NURBS1DFiniteElement::Project(VectorCoefficient &vc,
                                   ElementTransformation &Trans,
                                   Vector &dofs) const
{
   MFEM_ASSERT(dofs.Size() == vc.GetVDim()*dof, "");
   Vector x(vc.GetVDim());
   IntegrationPoint ip;

   for (int i = 0; i <= order; i++)
   {
      real_t kx = kv[0]->GetBotella(ijk[0] + i);
      if (!kv[0]->inSpan(kx, ijk[0]+order)) { continue; }
      ip.x = kv[0]->GetRefPoint(kx, ijk[0]+order);

      Trans.SetIntPoint(&ip);
      vc.Eval(x, Trans, ip);
      for (int j = 0; j < x.Size(); j++)
      {
         dofs(dof*j+i) = x(j);
      }
   }
}


void NURBS2DFiniteElement::SetOrder() const
{
   orders[0] = kv[0]->GetOrder();
   orders[1] = kv[1]->GetOrder();
   shape_x.SetSize(orders[0]+1);
   shape_y.SetSize(orders[1]+1);
   dshape_x.SetSize(orders[0]+1);
   dshape_y.SetSize(orders[1]+1);
   d2shape_x.SetSize(orders[0]+1);
   d2shape_y.SetSize(orders[1]+1);

   order = max(orders[0], orders[1]);
   dof = (orders[0] + 1)*(orders[1] + 1);
   u.SetSize(dof);
   du.SetSize(dof);
   weights.SetSize(dof);
}

void NURBS2DFiniteElement::CalcShape(const IntegrationPoint &ip,
                                     Vector &shape) const
{
   kv[0]->CalcShape(shape_x, ijk[0], ip.x);
   kv[1]->CalcShape(shape_y, ijk[1], ip.y);

   real_t sum = 0.0;
   for (int o = 0, j = 0; j <= orders[1]; j++)
   {
      const real_t sy = shape_y(j);
      for (int i = 0; i <= orders[0]; i++, o++)
      {
         sum += ( shape(o) = shape_x(i)*sy*weights(o) );
      }
   }

   shape /= sum;
}

void NURBS2DFiniteElement::CalcDShape(const IntegrationPoint &ip,
                                      DenseMatrix &dshape) const
{
   real_t sum, dsum[2];

   kv[0]->CalcShape ( shape_x, ijk[0], ip.x);
   kv[1]->CalcShape ( shape_y, ijk[1], ip.y);

   kv[0]->CalcDShape(dshape_x, ijk[0], ip.x);
   kv[1]->CalcDShape(dshape_y, ijk[1], ip.y);

   sum = dsum[0] = dsum[1] = 0.0;
   for (int o = 0, j = 0; j <= orders[1]; j++)
   {
      const real_t sy = shape_y(j), dsy = dshape_y(j);
      for (int i = 0; i <= orders[0]; i++, o++)
      {
         sum += ( u(o) = shape_x(i)*sy*weights(o) );

         dsum[0] += ( dshape(o,0) = dshape_x(i)*sy *weights(o) );
         dsum[1] += ( dshape(o,1) =  shape_x(i)*dsy*weights(o) );
      }
   }

   sum = 1.0/sum;
   dsum[0] *= sum*sum;
   dsum[1] *= sum*sum;

   for (int o = 0; o < dof; o++)
   {
      dshape(o,0) = dshape(o,0)*sum - u(o)*dsum[0];
      dshape(o,1) = dshape(o,1)*sum - u(o)*dsum[1];
   }
}

void NURBS2DFiniteElement::CalcHessian (const IntegrationPoint &ip,
                                        DenseMatrix &hessian) const
{
   real_t sum, dsum[2], d2sum[3];

   kv[0]->CalcShape ( shape_x, ijk[0], ip.x);
   kv[1]->CalcShape ( shape_y, ijk[1], ip.y);

   kv[0]->CalcDShape(dshape_x, ijk[0], ip.x);
   kv[1]->CalcDShape(dshape_y, ijk[1], ip.y);

   kv[0]->CalcD2Shape(d2shape_x, ijk[0], ip.x);
   kv[1]->CalcD2Shape(d2shape_y, ijk[1], ip.y);

   sum = dsum[0] = dsum[1] = 0.0;
   d2sum[0] = d2sum[1] = d2sum[2] = 0.0;
   for (int o = 0, j = 0; j <= orders[1]; j++)
   {
      const real_t sy = shape_y(j), dsy = dshape_y(j), d2sy = d2shape_y(j);
      for (int i = 0; i <= orders[0]; i++, o++)
      {
         const real_t sx = shape_x(i), dsx = dshape_x(i), d2sx = d2shape_x(i);
         sum += ( u(o) = sx*sy*weights(o) );

         dsum[0] += ( du(o,0) = dsx*sy*weights(o) );
         dsum[1] += ( du(o,1) = sx*dsy*weights(o) );

         d2sum[0] += ( hessian(o,0) = d2sx*sy*weights(o) );
         d2sum[1] += ( hessian(o,1) = dsx*dsy*weights(o) );
         d2sum[2] += ( hessian(o,2) = sx*d2sy*weights(o) );
      }
   }

   sum = 1.0/sum;
   dsum[0] *= sum;
   dsum[1] *= sum;

   d2sum[0] *= sum;
   d2sum[1] *= sum;
   d2sum[2] *= sum;

   for (int o = 0; o < dof; o++)
   {
      hessian(o,0) = hessian(o,0)*sum
                     - 2*du(o,0)*sum*dsum[0]
                     + u[o]*sum*(2*dsum[0]*dsum[0] - d2sum[0]);

      hessian(o,1) = hessian(o,1)*sum
                     - du(o,0)*sum*dsum[1]
                     - du(o,1)*sum*dsum[0]
                     + u[o]*sum*(2*dsum[0]*dsum[1] - d2sum[1]);

      hessian(o,2) = hessian(o,2)*sum
                     - 2*du(o,1)*sum*dsum[1]
                     + u[o]*sum*(2*dsum[1]*dsum[1] - d2sum[2]);
   }
}

void NURBS2DFiniteElement::Project(Coefficient &coeff,
                                   ElementTransformation &Trans,
                                   Vector &dofs) const
{
   IntegrationPoint ip;
   for (int o = 0, j = 0; j <= orders[1]; j++)
   {
      real_t ky = kv[1]->GetBotella(ijk[1] + j);
      if (!kv[1]->inSpan(ky, ijk[1]+orders[1]))
      {
         o += orders[0] + 1;
         continue;
      }
      ip.y = kv[1]->GetRefPoint(ky, ijk[1]+orders[1]);
      for (int i = 0; i <= orders[0]; i++, o++)
      {
         real_t kx = kv[0]->GetBotella(ijk[0] + i);
         if (!kv[0]->inSpan(kx, ijk[0]+orders[0])) { continue; }
         ip.x = kv[0]->GetRefPoint(kx, ijk[0]+orders[0]);

         Trans.SetIntPoint(&ip);
         dofs(o) = coeff.Eval(Trans, ip);
      }
   }
}

void NURBS2DFiniteElement::Project(VectorCoefficient &vc,
                                   ElementTransformation &Trans,
                                   Vector &dofs) const
{
   MFEM_ASSERT(dofs.Size() == vc.GetVDim()*dof, "");
   Vector x(vc.GetVDim());
   IntegrationPoint ip;
   for (int o = 0, j = 0; j <= orders[1]; j++)
   {
      real_t ky = kv[1]->GetBotella(ijk[1] + j);
      if (!kv[1]->inSpan(ky, ijk[1]+orders[1]))
      {
         o += orders[0] + 1;
         continue;
      }
      ip.y = kv[1]->GetRefPoint(ky, ijk[1]+orders[1]);
      for (int i = 0; i <= orders[0]; i++, o++)
      {
         real_t kx = kv[0]->GetBotella(ijk[0] + i);
         if (!kv[0]->inSpan(kx, ijk[0]+orders[0])) { continue; }
         ip.x = kv[0]->GetRefPoint(kx, ijk[0]+orders[0]);

         Trans.SetIntPoint(&ip);
         vc.Eval(x, Trans, ip);
         for (int v = 0; v < x.Size(); v++)
         {
            dofs(dof*v+o) = x(v);
         }
      }
   }
}

void NURBS3DFiniteElement::SetOrder() const
{
   orders[0] = kv[0]->GetOrder();
   orders[1] = kv[1]->GetOrder();
   orders[2] = kv[2]->GetOrder();
   shape_x.SetSize(orders[0]+1);
   shape_y.SetSize(orders[1]+1);
   shape_z.SetSize(orders[2]+1);

   dshape_x.SetSize(orders[0]+1);
   dshape_y.SetSize(orders[1]+1);
   dshape_z.SetSize(orders[2]+1);

   d2shape_x.SetSize(orders[0]+1);
   d2shape_y.SetSize(orders[1]+1);
   d2shape_z.SetSize(orders[2]+1);

   order = max(max(orders[0], orders[1]), orders[2]);
   dof = (orders[0] + 1)*(orders[1] + 1)*(orders[2] + 1);
   u.SetSize(dof);
   du.SetSize(dof);
   weights.SetSize(dof);
}

void NURBS3DFiniteElement::CalcShape(const IntegrationPoint &ip,
                                     Vector &shape) const
{
   kv[0]->CalcShape(shape_x, ijk[0], ip.x);
   kv[1]->CalcShape(shape_y, ijk[1], ip.y);
   kv[2]->CalcShape(shape_z, ijk[2], ip.z);

   real_t sum = 0.0;
   for (int o = 0, k = 0; k <= orders[2]; k++)
   {
      const real_t sz = shape_z(k);
      for (int j = 0; j <= orders[1]; j++)
      {
         const real_t sy_sz = shape_y(j)*sz;
         for (int i = 0; i <= orders[0]; i++, o++)
         {
            sum += ( shape(o) = shape_x(i)*sy_sz*weights(o) );
         }
      }
   }

   shape /= sum;
}

void NURBS3DFiniteElement::CalcDShape(const IntegrationPoint &ip,
                                      DenseMatrix &dshape) const
{
   real_t sum, dsum[3];

   kv[0]->CalcShape ( shape_x, ijk[0], ip.x);
   kv[1]->CalcShape ( shape_y, ijk[1], ip.y);
   kv[2]->CalcShape ( shape_z, ijk[2], ip.z);

   kv[0]->CalcDShape(dshape_x, ijk[0], ip.x);
   kv[1]->CalcDShape(dshape_y, ijk[1], ip.y);
   kv[2]->CalcDShape(dshape_z, ijk[2], ip.z);

   sum = dsum[0] = dsum[1] = dsum[2] = 0.0;
   for (int o = 0, k = 0; k <= orders[2]; k++)
   {
      const real_t sz = shape_z(k), dsz = dshape_z(k);
      for (int j = 0; j <= orders[1]; j++)
      {
         const real_t  sy_sz  =  shape_y(j)* sz;
         const real_t dsy_sz  = dshape_y(j)* sz;
         const real_t  sy_dsz =  shape_y(j)*dsz;
         for (int i = 0; i <= orders[0]; i++, o++)
         {
            sum += ( u(o) = shape_x(i)*sy_sz*weights(o) );

            dsum[0] += ( dshape(o,0) = dshape_x(i)* sy_sz *weights(o) );
            dsum[1] += ( dshape(o,1) =  shape_x(i)*dsy_sz *weights(o) );
            dsum[2] += ( dshape(o,2) =  shape_x(i)* sy_dsz*weights(o) );
         }
      }
   }

   sum = 1.0/sum;
   dsum[0] *= sum*sum;
   dsum[1] *= sum*sum;
   dsum[2] *= sum*sum;

   for (int o = 0; o < dof; o++)
   {
      dshape(o,0) = dshape(o,0)*sum - u(o)*dsum[0];
      dshape(o,1) = dshape(o,1)*sum - u(o)*dsum[1];
      dshape(o,2) = dshape(o,2)*sum - u(o)*dsum[2];
   }
}

void NURBS3DFiniteElement::CalcHessian (const IntegrationPoint &ip,
                                        DenseMatrix &hessian) const
{
   real_t sum, dsum[3], d2sum[6];

   kv[0]->CalcShape ( shape_x, ijk[0], ip.x);
   kv[1]->CalcShape ( shape_y, ijk[1], ip.y);
   kv[2]->CalcShape ( shape_z, ijk[2], ip.z);

   kv[0]->CalcDShape(dshape_x, ijk[0], ip.x);
   kv[1]->CalcDShape(dshape_y, ijk[1], ip.y);
   kv[2]->CalcDShape(dshape_z, ijk[2], ip.z);

   kv[0]->CalcD2Shape(d2shape_x, ijk[0], ip.x);
   kv[1]->CalcD2Shape(d2shape_y, ijk[1], ip.y);
   kv[2]->CalcD2Shape(d2shape_z, ijk[2], ip.z);

   sum = dsum[0] = dsum[1] = dsum[2] = 0.0;
   d2sum[0] = d2sum[1] = d2sum[2] = d2sum[3] = d2sum[4] = d2sum[5] = 0.0;

   for (int o = 0, k = 0; k <= orders[2]; k++)
   {
      const real_t sz = shape_z(k), dsz = dshape_z(k), d2sz = d2shape_z(k);
      for (int j = 0; j <= orders[1]; j++)
      {
         const real_t sy = shape_y(j), dsy = dshape_y(j), d2sy = d2shape_y(j);
         for (int i = 0; i <= orders[0]; i++, o++)
         {
            const real_t sx = shape_x(i), dsx = dshape_x(i), d2sx = d2shape_x(i);
            sum += ( u(o) = sx*sy*sz*weights(o) );

            dsum[0] += ( du(o,0) = dsx*sy*sz*weights(o) );
            dsum[1] += ( du(o,1) = sx*dsy*sz*weights(o) );
            dsum[2] += ( du(o,2) = sx*sy*dsz*weights(o) );

            d2sum[0] += ( hessian(o,0) = d2sx*sy*sz*weights(o) );
            d2sum[1] += ( hessian(o,1) = dsx*dsy*sz*weights(o) );
            d2sum[2] += ( hessian(o,2) = dsx*sy*dsz*weights(o) );
            d2sum[3] += ( hessian(o,3) = sx*d2sy*sz*weights(o) );
            d2sum[4] += ( hessian(o,4) = sx*dsy*dsz*weights(o) );
            d2sum[5] += ( hessian(o,5) = sx*sy*d2sz*weights(o) );

         }
      }
   }

   sum = 1.0/sum;
   dsum[0] *= sum;
   dsum[1] *= sum;
   dsum[2] *= sum;

   d2sum[0] *= sum;
   d2sum[1] *= sum;
   d2sum[2] *= sum;

   d2sum[3] *= sum;
   d2sum[4] *= sum;
   d2sum[5] *= sum;

   for (int o = 0; o < dof; o++)
   {
      hessian(o,0) = hessian(o,0)*sum
                     - 2*du(o,0)*sum*dsum[0]
                     + u[o]*sum*(2*dsum[0]*dsum[0] - d2sum[0]);

      hessian(o,1) = hessian(o,1)*sum
                     - du(o,0)*sum*dsum[1]
                     - du(o,1)*sum*dsum[0]
                     + u[o]*sum*(2*dsum[0]*dsum[1] - d2sum[1]);

      hessian(o,2) = hessian(o,2)*sum
                     - du(o,0)*sum*dsum[2]
                     - du(o,2)*sum*dsum[0]
                     + u[o]*sum*(2*dsum[0]*dsum[2] - d2sum[2]);

      hessian(o,3) = hessian(o,3)*sum
                     - du(o,1)*sum*dsum[2]
                     - du(o,2)*sum*dsum[1]
                     + u[o]*sum*(2*dsum[1]*dsum[2] - d2sum[3]);

      hessian(o,4) = hessian(o,4)*sum
                     - 2*du(o,2)*sum*dsum[2]
                     + u[o]*sum*(2*dsum[2]*dsum[2] - d2sum[4]);

      hessian(o,5) = hessian(o,5)*sum
                     - 2*du(o,1)*sum*dsum[1]
                     + u[o]*sum*(2*dsum[1]*dsum[1] - d2sum[5]);
   }
}

void NURBS3DFiniteElement::Project(Coefficient &coeff,
                                   ElementTransformation &Trans,
                                   Vector &dofs) const
{
   IntegrationPoint ip;

   for (int o = 0, k = 0; k <= orders[2]; k++)
   {
      real_t kz = kv[2]->GetBotella(ijk[2] + k);
      if (!kv[2]->inSpan(kz, ijk[2]+orders[2]))
      {
         o += (orders[0] + 1)*(orders[1] + 1);
         continue;
      }
      ip.z = kv[2]->GetRefPoint(kz, ijk[2]+orders[2]);
      for (int j = 0; j <= orders[1]; j++)
      {
         real_t ky = kv[1]->GetBotella(ijk[1] + j);
         if (!kv[1]->inSpan(ky, ijk[1]+orders[1]))
         {
            o += orders[0] + 1;
            continue;
         }
         ip.y = kv[1]->GetRefPoint(ky, ijk[1]+orders[1]);
         for (int i = 0; i <= orders[0]; i++, o++)
         {
            real_t kx = kv[0]->GetBotella(ijk[0] + i);
            if (!kv[0]->inSpan(kx, ijk[0]+orders[0])) { continue; }
            ip.x = kv[0]->GetRefPoint(kx, ijk[0]+orders[0]);

            Trans.SetIntPoint(&ip);
            dofs(o) = coeff.Eval(Trans, ip);
         }
      }
   }
}

void NURBS3DFiniteElement::Project(VectorCoefficient &vc,
                                   ElementTransformation &Trans,
                                   Vector &dofs) const
{
   MFEM_ASSERT(dofs.Size() == vc.GetVDim()*dof, "");
   Vector x(vc.GetVDim());
   IntegrationPoint ip;

   for (int o = 0, k = 0; k <= orders[2]; k++)
   {
      real_t kz = kv[2]->GetBotella(ijk[2] + k);
      if (!kv[2]->inSpan(kz, ijk[2]+orders[2]))
      {
         o += (orders[0] + 1)*(orders[1] + 1);
         continue;
      }
      ip.z = kv[2]->GetRefPoint(kz, ijk[2]+orders[2]);
      for (int j = 0; j <= orders[1]; j++)
      {
         real_t ky = kv[1]->GetBotella(ijk[1] + j);
         if (!kv[1]->inSpan(ky, ijk[1]+orders[1]))
         {
            o += orders[0] + 1;
            continue;
         }
         ip.y = kv[1]->GetRefPoint(ky, ijk[1]+orders[1]);
         for (int i = 0; i <= orders[0]; i++, o++)
         {
            real_t kx = kv[0]->GetBotella(ijk[0] + i);
            if (!kv[0]->inSpan(kx, ijk[0]+orders[0])) { continue; }
            ip.x = kv[0]->GetRefPoint(kx, ijk[0]+orders[0]);

            Trans.SetIntPoint(&ip);
            vc.Eval(x, Trans, ip);
            for (int v = 0; v < x.Size(); v++)
            {
               dofs(dof*v+o) = x(v);
            }
         }
      }
   }
}

void NURBS_HDiv2DFiniteElement::SetOrder() const
{
   orders[0] = kv[0]->GetOrder();
   orders[1] = kv[1]->GetOrder();

   if (kv1[0]) { delete kv1[0]; }
   if (kv1[1]) { delete kv1[1]; }

   kv1[0] = kv[0]->DegreeElevate(1);
   kv1[1] = kv[1]->DegreeElevate(1);

   shape_x.SetSize(orders[0]+1);
   shape_y.SetSize(orders[1]+1);

   dshape_x.SetSize(orders[0]+1);
   dshape_y.SetSize(orders[1]+1);

   d2shape_x.SetSize(orders[0]+1);
   d2shape_y.SetSize(orders[1]+1);

   shape1_x.SetSize(orders[0]+2);
   shape1_y.SetSize(orders[1]+2);

   dshape1_x.SetSize(orders[0]+2);
   dshape1_y.SetSize(orders[1]+2);

   d2shape1_x.SetSize(orders[0]+2);
   d2shape1_y.SetSize(orders[1]+2);

   order = max(orders[0]+1, orders[1]+1);
   dof = (orders[0] + 2)*(orders[1] + 1)
         + (orders[1] + 1)*(orders[1] + 2);
   u.SetSize(dof);
   du.SetSize(dof);
   weights.SetSize(dof);
}

void NURBS_HDiv2DFiniteElement::CalcVShape(const IntegrationPoint &ip,
                                           DenseMatrix &shape) const
{
   kv[0]->CalcShape(shape_x, ijk[0], ip.x);
   kv[1]->CalcShape(shape_y, ijk[1], ip.y);

   kv1[0]->CalcShape(shape1_x, ijk[0], ip.x);
   kv1[1]->CalcShape(shape1_y, ijk[1], ip.y);

   int o = 0;
   for (int j = 0; j <= orders[1]; j++)
   {
      const real_t sy = shape_y(j);
      for (int i = 0; i <= orders[0]+1; i++, o++)
      {
         shape(o,0) = shape1_x(i)*sy;
         shape(o,1) = 0.0;
      }
   }

   for (int j = 0; j <= orders[1]+1; j++)
   {
      const real_t sy1 = shape1_y(j);
      for (int i = 0; i <= orders[0]; i++, o++)
      {
         shape(o,0) = 0.0;
         shape(o,1) = shape_x(i)*sy1;
      }
   }
}

void NURBS_HDiv2DFiniteElement::CalcVShape(ElementTransformation &Trans,
                                           DenseMatrix &shape) const
{
   CalcVShape(Trans.GetIntPoint(), shape);
   const DenseMatrix & J = Trans.Jacobian();
   MFEM_ASSERT(J.Width() == 2 && J.Height() == 2,
               "NURBS_HDiv2DFiniteElement cannot be embedded in "
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

void NURBS_HDiv2DFiniteElement::CalcDivShape(const IntegrationPoint &ip,
                                             Vector &divshape) const
{
   kv[0]->CalcShape ( shape_x, ijk[0], ip.x);
   kv[1]->CalcShape ( shape_y, ijk[1], ip.y);

   kv1[0]->CalcDShape(dshape1_x, ijk[0], ip.x);
   kv1[1]->CalcDShape(dshape1_y, ijk[1], ip.y);

   int o = 0;
   for (int j = 0; j <= orders[1]; j++)
   {
      const real_t sy = shape_y(j);
      for (int i = 0; i <= orders[0]+1; i++, o++)
      {
         divshape(o) = dshape1_x(i)*sy;
      }
   }

   for (int j = 0; j <= orders[1]+1; j++)
   {
      const real_t dsy1 = dshape1_y(j);
      for (int i = 0; i <= orders[0]; i++, o++)
      {
         divshape(o) = shape_x(i)*dsy1;
      }
   }
}

void NURBS_HDiv2DFiniteElement::Project(VectorCoefficient &vc,
                                        ElementTransformation &Trans,
                                        Vector &dofs) const
{
   MFEM_ASSERT(dofs.Size() == dof, "");
   MFEM_ASSERT(vc.GetVDim() == 2, "");
   Vector x(2), mx(2);
   IntegrationPoint ip;
   int o = 0;

   for (int j = 0; j <= orders[1]; j++)
   {
      real_t ky = kv[1]->GetBotella(ijk[1] + j);
      if (!kv[1]->inSpan(ky, ijk[1]+orders[1]))
      {
         o += orders[0] + 2;
         continue;
      }
      ip.y = kv[1]->GetRefPoint(ky, ijk[1]+orders[1]);
      for (int i = 0; i <= orders[0]+1; i++, o++)
      {
         real_t kx = kv1[0]->GetBotella(ijk[0] + i);
         if (!kv1[0]->inSpan(kx, ijk[0]+orders[0]+1)) { continue; }
         ip.x = kv1[0]->GetRefPoint(kx, ijk[0]+orders[0]+1);

         Trans.SetIntPoint(&ip);
         vc.Eval(x, Trans, ip);

         Trans.AdjugateJacobian().Mult(x,mx);
         dofs(o) = mx(0);
      }
   }

   for (int j = 0; j <= orders[1]+1; j++)
   {
      real_t ky = kv1[1]->GetBotella(ijk[1] + j);
      if (!kv1[1]->inSpan(ky, ijk[1]+orders[1]+1))
      {
         o += orders[0] + 1;
         continue;
      }
      ip.y = kv1[1]->GetRefPoint(ky, ijk[1]+orders[1]+1);
      for (int i = 0; i <= orders[0]; i++, o++)
      {
         real_t kx = kv[0]->GetBotella(ijk[0] + i);
         if (!kv[0]->inSpan(kx, ijk[0]+orders[0])) { continue; }
         ip.x = kv[0]->GetRefPoint(kx, ijk[0]+orders[0]);

         Trans.SetIntPoint(&ip);
         vc.Eval(x, Trans, ip);

         Trans.AdjugateJacobian().Mult(x,mx);
         dofs(o) = mx(1);
      }
   }
}

NURBS_HDiv2DFiniteElement::~NURBS_HDiv2DFiniteElement()
{
   if (kv1[0]) { delete kv1[0]; }
   if (kv1[1]) { delete kv1[1]; }
}


void NURBS_HDiv3DFiniteElement::SetOrder() const
{
   orders[0] = kv[0]->GetOrder();
   orders[1] = kv[1]->GetOrder();
   orders[2] = kv[2]->GetOrder();

   if (kv1[0]) { delete kv1[0]; }
   if (kv1[1]) { delete kv1[1]; }
   if (kv1[2]) { delete kv1[2]; }

   kv1[0] = kv[0]->DegreeElevate(1);
   kv1[1] = kv[1]->DegreeElevate(1);
   kv1[2] = kv[2]->DegreeElevate(1);

   shape_x.SetSize(orders[0]+1);
   shape_y.SetSize(orders[1]+1);
   shape_z.SetSize(orders[2]+1);

   dshape_x.SetSize(orders[0]+1);
   dshape_y.SetSize(orders[1]+1);
   dshape_z.SetSize(orders[2]+1);

   d2shape_x.SetSize(orders[0]+1);
   d2shape_y.SetSize(orders[1]+1);
   d2shape_z.SetSize(orders[2]+1);

   shape1_x.SetSize(orders[0]+2);
   shape1_y.SetSize(orders[1]+2);
   shape1_z.SetSize(orders[2]+2);

   dshape1_x.SetSize(orders[0]+2);
   dshape1_y.SetSize(orders[1]+2);
   dshape1_z.SetSize(orders[2]+2);

   d2shape1_x.SetSize(orders[0]+2);
   d2shape1_y.SetSize(orders[1]+2);
   d2shape1_z.SetSize(orders[2]+2);

   order = max(orders[0]+1, max( orders[1]+1, orders[2]+1));
   dof = (orders[0] + 2)*(orders[1] + 1)*(orders[2] + 1) +
         (orders[0] + 1)*(orders[1] + 2)*(orders[2] + 1) +
         (orders[0] + 1)*(orders[1] + 1)*(orders[2] + 2);
   u.SetSize(dof);
   du.SetSize(dof);
   weights.SetSize(dof);
}

void NURBS_HDiv3DFiniteElement::CalcVShape(const IntegrationPoint &ip,
                                           DenseMatrix &shape) const
{
   kv[0]->CalcShape(shape_x, ijk[0], ip.x);
   kv[1]->CalcShape(shape_y, ijk[1], ip.y);
   kv[2]->CalcShape(shape_z, ijk[2], ip.z);

   kv1[0]->CalcShape(shape1_x, ijk[0], ip.x);
   kv1[1]->CalcShape(shape1_y, ijk[1], ip.y);
   kv1[2]->CalcShape(shape1_z, ijk[2], ip.z);

   shape = 0.0;
   int o = 0;
   for (int k = 0; k <= orders[2]; k++)
   {
      const real_t sz = shape_z(k);
      for (int j = 0; j <= orders[1]; j++)
      {
         const real_t sy_sz = shape_y(j)*sz;
         for (int i = 0; i <= orders[0]+1; i++, o++)
         {
            shape(o,0) = shape1_x(i)*sy_sz;
         }
      }
   }

   for (int  k = 0; k <= orders[2]; k++)
   {
      const real_t sz = shape_z(k);
      for (int j = 0; j <= orders[1]+1; j++)
      {
         const real_t sy1_sz = shape1_y(j)*sz;
         for (int i = 0; i <= orders[0]; i++, o++)
         {
            shape(o,1) = shape_x(i)*sy1_sz;
         }
      }
   }

   for (int  k = 0; k <= orders[2]+1; k++)
   {
      const real_t sz1 = shape1_z(k);
      for (int j = 0; j <= orders[1]; j++)
      {
         const real_t sy_sz1 = shape_y(j)*sz1;
         for (int i = 0; i <= orders[0]; i++, o++)
         {
            shape(o,2) = shape_x(i)*sy_sz1;
         }
      }
   }
}

void NURBS_HDiv3DFiniteElement::CalcVShape(ElementTransformation &Trans,
                                           DenseMatrix &shape) const
{
   CalcVShape(Trans.GetIntPoint(), shape);
   const DenseMatrix & J = Trans.Jacobian();
   MFEM_ASSERT(J.Width() == 3 && J.Height() == 3,
               "RT_R2D_FiniteElement cannot be embedded in "
               "3 dimensional spaces");
   for (int i=0; i<dof; i++)
   {
      real_t sx = shape(i, 0);
      real_t sy = shape(i, 1);
      real_t sz = shape(i, 2);
      shape(i, 0) = sx * J(0, 0) + sy * J(0, 1) + sz * J(0, 2);
      shape(i, 1) = sx * J(1, 0) + sy * J(1, 1) + sz * J(1, 2);
      shape(i, 2) = sx * J(2, 0) + sy * J(2, 1) + sz * J(2, 2);
   }
   shape *= (1.0 / Trans.Weight());
}

void NURBS_HDiv3DFiniteElement::CalcDivShape(const IntegrationPoint &ip,
                                             Vector &divshape) const
{
   kv[0]->CalcShape ( shape_x, ijk[0], ip.x);
   kv[1]->CalcShape ( shape_y, ijk[1], ip.y);
   kv[2]->CalcShape ( shape_z, ijk[2], ip.z);

   kv1[0]->CalcDShape(dshape1_x, ijk[0], ip.x);
   kv1[1]->CalcDShape(dshape1_y, ijk[1], ip.y);
   kv1[2]->CalcDShape(dshape1_z, ijk[2], ip.z);

   int o = 0;
   for (int  k = 0; k <= orders[2]; k++)
   {
      const real_t sz = shape_z(k);
      for (int j = 0; j <= orders[1]; j++)
      {
         const real_t sy_sz = shape_y(j)*sz;
         for (int i = 0; i <= orders[0]+1; i++, o++)
         {
            divshape(o) = dshape1_x(i)*sy_sz;
         }
      }
   }

   for (int  k = 0; k <= orders[2]; k++)
   {
      const real_t sz = shape_z(k);
      for (int j = 0; j <= orders[1]+1; j++)
      {
         const real_t dy1_sz = dshape1_y(j)*sz;
         for (int i = 0; i <= orders[0]; i++, o++)
         {
            divshape(o) = shape_x(i)*dy1_sz;
         }
      }
   }

   for (int  k = 0; k <= orders[2]+1; k++)
   {
      const real_t dz1 = dshape1_z(k);
      for (int j = 0; j <= orders[1]; j++)
      {
         const real_t sy_dz1 = shape_y(j)*dz1;
         for (int i = 0; i <= orders[0]; i++, o++)
         {
            divshape(o) = shape_x(i)*sy_dz1;
         }
      }
   }
}


void NURBS_HDiv3DFiniteElement::Project(VectorCoefficient &vc,
                                        ElementTransformation &Trans,
                                        Vector &dofs) const
{
   MFEM_ASSERT(dofs.Size() == dof, "");
   MFEM_ASSERT(vc.GetVDim() == 3, "");
   Vector x(2), mx(3);
   IntegrationPoint ip;

   int o = 0;

   for (int k = 0; k <= orders[2]; k++)
   {
      real_t kz = kv[2]->GetBotella(ijk[2] + k);
      if (!kv[2]->inSpan(kz, ijk[2]+orders[2]))
      {
         o += (orders[0] + 2)*(orders[1] + 1);
         continue;
      }
      ip.z = kv[2]->GetRefPoint(kz, ijk[2]+orders[2]);
      for (int j = 0; j <= orders[1]; j++)
      {
         real_t ky = kv[1]->GetBotella(ijk[1] + j);
         if (!kv[1]->inSpan(ky, ijk[1]+orders[1]))
         {
            o += orders[0] + 2;
            continue;
         }
         ip.y = kv[1]->GetRefPoint(ky, ijk[1]+orders[1]);
         for (int i = 0; i <= orders[0]+1; i++, o++)
         {
            real_t kx = kv1[0]->GetBotella(ijk[0] + i);
            if (!kv1[0]->inSpan(kx, ijk[0]+orders[0]+1)) { continue; }
            ip.x = kv1[0]->GetRefPoint(kx, ijk[0]+orders[0]+1);

            Trans.SetIntPoint(&ip);
            vc.Eval(x, Trans, ip);

            Trans.AdjugateJacobian().Mult(x,mx);
            dofs(o) = mx(0);
         }
      }
   }

   for (int k = 0; k <= orders[2]; k++)
   {
      real_t kz = kv[2]->GetBotella(ijk[2] + k);
      if (!kv[2]->inSpan(kz, ijk[2]+orders[2]))
      {
         o += (orders[0] + 1)*(orders[1] + 2);
         continue;
      }
      ip.z = kv[2]->GetRefPoint(kz, ijk[2]+orders[2]);
      for (int j = 0; j <= orders[1]+1; j++)
      {
         real_t ky = kv1[1]->GetBotella(ijk[1] + j);
         if (!kv1[1]->inSpan(ky, ijk[1]+orders[1]+1))
         {
            o += orders[0] + 1;
            continue;
         }
         ip.y = kv1[1]->GetRefPoint(ky, ijk[1]+orders[1]+1);
         for (int i = 0; i <= orders[0]; i++, o++)
         {
            real_t kx = kv[0]->GetBotella(ijk[0] + i);
            if (!kv[0]->inSpan(kx, ijk[0]+orders[0])) { continue; }
            ip.x = kv[0]->GetRefPoint(kx, ijk[0]+orders[0]);

            Trans.SetIntPoint(&ip);
            vc.Eval(x, Trans, ip);

            Trans.AdjugateJacobian().Mult(x,mx);
            dofs(o) = mx(1);
         }
      }
   }

   for (int k = 0; k <= orders[2]+1; k++)
   {
      real_t kz = kv1[2]->GetBotella(ijk[2] + k);
      if (!kv1[2]->inSpan(kz, ijk[2]+orders[2]+1))
      {
         o += (orders[0] + 1)*(orders[1] + 1);
         continue;
      }
      ip.z = kv1[2]->GetRefPoint(kz, ijk[2]+orders[2]+1);
      for (int j = 0; j <= orders[1]; j++)
      {
         real_t ky = kv[1]->GetBotella(ijk[1] + j);
         if (!kv[1]->inSpan(ky, ijk[1]+orders[1]))
         {
            o += orders[0] + 1;
            continue;
         }
         ip.y = kv[1]->GetRefPoint(ky, ijk[1]+orders[1]);
         for (int i = 0; i <= orders[0]; i++, o++)
         {
            real_t kx = kv[0]->GetBotella(ijk[0] + i);
            if (!kv[0]->inSpan(kx, ijk[0]+orders[0])) { continue; }
            ip.x = kv[0]->GetRefPoint(kx, ijk[0]+orders[0]);

            Trans.SetIntPoint(&ip);
            vc.Eval(x, Trans, ip);

            Trans.AdjugateJacobian().Mult(x,mx);
            dofs(o) = mx(2);
         }
      }
   }

}


NURBS_HDiv3DFiniteElement::~NURBS_HDiv3DFiniteElement()
{
   if (kv1[0]) { delete kv1[0]; }
   if (kv1[1]) { delete kv1[1]; }
   if (kv1[2]) { delete kv1[2]; }
}

void NURBS_HCurl2DFiniteElement::SetOrder() const
{
   orders[0] = kv[0]->GetOrder();
   orders[1] = kv[1]->GetOrder();

   if (kv1[0]) { delete kv1[0]; }
   if (kv1[1]) { delete kv1[1]; }

   kv1[0] = kv[0]->DegreeElevate(1);
   kv1[1] = kv[1]->DegreeElevate(1);

   shape_x.SetSize(orders[0]+1);
   shape_y.SetSize(orders[1]+1);

   dshape_x.SetSize(orders[0]+1);
   dshape_y.SetSize(orders[1]+1);

   d2shape_x.SetSize(orders[0]+1);
   d2shape_y.SetSize(orders[1]+1);

   shape1_x.SetSize(orders[0]+2);
   shape1_y.SetSize(orders[1]+2);

   dshape1_x.SetSize(orders[0]+2);
   dshape1_y.SetSize(orders[1]+2);

   d2shape1_x.SetSize(orders[0]+2);
   d2shape1_y.SetSize(orders[1]+2);

   order = max(orders[0]+1, orders[1]+1);
   dof = (orders[0] + 1)*(orders[1] + 2)
         + (orders[1] + 2)*(orders[1] + 1);
   u.SetSize(dof);
   du.SetSize(dof);
   weights.SetSize(dof);
}

void NURBS_HCurl2DFiniteElement::CalcVShape(const IntegrationPoint &ip,
                                            DenseMatrix &shape) const
{
   kv[0]->CalcShape(shape_x, ijk[0], ip.x);
   kv[1]->CalcShape(shape_y, ijk[1], ip.y);

   kv1[0]->CalcShape(shape1_x, ijk[0], ip.x);
   kv1[1]->CalcShape(shape1_y, ijk[1], ip.y);

   int o = 0;
   for (int j = 0; j <= orders[1]+1; j++)
   {
      const real_t sy1 = shape1_y(j);
      for (int i = 0; i <= orders[0]; i++, o++)
      {
         shape(o,0) = shape_x(i)*sy1;
         shape(o,1) = 0.0;
      }
   }

   for (int j = 0; j <= orders[1]; j++)
   {
      const real_t sy = shape_y(j);
      for (int i = 0; i <= orders[0]+1; i++, o++)
      {
         shape(o,0) = 0.0;
         shape(o,1) = shape1_x(i)*sy;
      }
   }
}

void NURBS_HCurl2DFiniteElement::CalcVShape(ElementTransformation &Trans,
                                            DenseMatrix &shape) const
{
   CalcVShape(Trans.GetIntPoint(), shape);
   const DenseMatrix & JI = Trans.InverseJacobian();
   MFEM_ASSERT(JI.Width() == 2 && JI.Height() == 2,
               "NURBS_HCurl2DFiniteElement cannot be embedded in "
               "3 dimensional spaces");
   for (int i=0; i<dof; i++)
   {
      real_t sx = shape(i, 0);
      real_t sy = shape(i, 1);
      shape(i, 0) = sx * JI(0, 0) + sy * JI(1, 0);
      shape(i, 1) = sx * JI(0, 1) + sy * JI(1, 1);
   }
}

void NURBS_HCurl2DFiniteElement::CalcCurlShape(const IntegrationPoint &ip,
                                               DenseMatrix &curl_shape) const
{
   kv[0]->CalcShape ( shape_x, ijk[0], ip.x);
   kv[1]->CalcShape ( shape_y, ijk[1], ip.y);

   kv1[0]->CalcDShape(dshape1_x, ijk[0], ip.x);
   kv1[1]->CalcDShape(dshape1_y, ijk[1], ip.y);

   int o = 0;
   for (int j = 0; j <= orders[1]+1; j++)
   {
      const real_t dsy1 = dshape1_y(j);
      for (int i = 0; i <= orders[0]; i++, o++)
      {
         curl_shape(o,0) = -shape_x(i)*dsy1;
      }
   }

   for (int j = 0; j <= orders[1]; j++)
   {
      const real_t sy = shape_y(j);
      for (int i = 0; i <= orders[0]+1; i++, o++)
      {
         curl_shape(o,0) = dshape1_x(i)*sy;
      }
   }
}

void NURBS_HCurl2DFiniteElement::Project(VectorCoefficient &vc,
                                         ElementTransformation &Trans,
                                         Vector &dofs) const
{
   MFEM_ASSERT(dofs.Size() == dof, "");
   MFEM_ASSERT(vc.GetVDim() == 2, "");
   Vector x(2), xm(2);
   IntegrationPoint ip;
   int i, j, o;
   for (o = 0, j = 0; j <= orders[1]+1; j++)
   {
      real_t ky = kv1[1]->GetBotella(ijk[1] + j);
      if (!kv1[1]->inSpan(ky, ijk[1]+orders[1]+1))
      {
         o += orders[0] + 1;
         continue;
      }
      ip.y = kv1[1]->GetRefPoint(ky, ijk[1]+orders[1]+1);
      for (i = 0; i <= orders[0]; i++, o++)
      {
         real_t kx = kv[0]->GetBotella(ijk[0] + i);
         if (!kv[0]->inSpan(kx, ijk[0]+orders[0])) { continue; }
         ip.x = kv[0]->GetRefPoint(kx, ijk[0]+orders[0]);

         Trans.SetIntPoint(&ip);
         vc.Eval(x, Trans, ip);

         Trans.Jacobian().MultTranspose(x,xm);
         dofs(o) = xm(0);
      }
   }

   for (j = 0; j <= orders[1]; j++)
   {
      real_t ky = kv[1]->GetBotella(ijk[1] + j);
      if (!kv[1]->inSpan(ky, ijk[1]+orders[1]))
      {
         o += orders[0] + 2;
         continue;
      }
      ip.y = kv[1]->GetRefPoint(ky, ijk[1]+orders[1]);
      for (i = 0; i <= orders[0]+1; i++, o++)
      {
         real_t kx = kv1[0]->GetBotella(ijk[0] + i);
         if (!kv1[0]->inSpan(kx, ijk[0]+orders[0]+1)) { continue; }
         ip.x = kv1[0]->GetRefPoint(kx, ijk[0]+orders[0]+1);

         Trans.SetIntPoint(&ip);
         vc.Eval(x, Trans, ip);

         Trans.Jacobian().MultTranspose(x,xm);
         dofs(o) = xm(1);
      }
   }
}

NURBS_HCurl2DFiniteElement::~NURBS_HCurl2DFiniteElement()
{
   if (kv1[0]) { delete kv1[0]; }
   if (kv1[1]) { delete kv1[1]; }
}

void NURBS_HCurl3DFiniteElement::SetOrder() const
{
   orders[0] = kv[0]->GetOrder();
   orders[1] = kv[1]->GetOrder();
   orders[2] = kv[2]->GetOrder();

   if (kv1[0]) { delete kv1[0]; }
   if (kv1[1]) { delete kv1[1]; }
   if (kv1[2]) { delete kv1[2]; }

   kv1[0] = kv[0]->DegreeElevate(1);
   kv1[1] = kv[1]->DegreeElevate(1);
   kv1[2] = kv[2]->DegreeElevate(1);

   shape_x.SetSize(orders[0]+1);
   shape_y.SetSize(orders[1]+1);
   shape_z.SetSize(orders[2]+1);

   dshape_x.SetSize(orders[0]+1);
   dshape_y.SetSize(orders[1]+1);
   dshape_z.SetSize(orders[2]+1);

   d2shape_x.SetSize(orders[0]+1);
   d2shape_y.SetSize(orders[1]+1);
   d2shape_z.SetSize(orders[2]+1);

   shape1_x.SetSize(orders[0]+2);
   shape1_y.SetSize(orders[1]+2);
   shape1_z.SetSize(orders[2]+2);

   dshape1_x.SetSize(orders[0]+2);
   dshape1_y.SetSize(orders[1]+2);
   dshape1_z.SetSize(orders[2]+2);

   d2shape1_x.SetSize(orders[0]+2);
   d2shape1_y.SetSize(orders[1]+2);
   d2shape1_z.SetSize(orders[2]+2);

   order = max(orders[0]+1, max( orders[1]+1, orders[2]+1));
   dof = (orders[0] + 1)*(orders[1] + 2)*(orders[2] + 2) +
         (orders[0] + 2)*(orders[1] + 1)*(orders[2] + 2) +
         (orders[0] + 2)*(orders[1] + 2)*(orders[2] + 1);
   u.SetSize(dof);
   du.SetSize(dof);
   weights.SetSize(dof);
}

void NURBS_HCurl3DFiniteElement::CalcVShape(const IntegrationPoint &ip,
                                            DenseMatrix &shape) const
{
   kv[0]->CalcShape(shape_x, ijk[0], ip.x);
   kv[1]->CalcShape(shape_y, ijk[1], ip.y);
   kv[2]->CalcShape(shape_z, ijk[2], ip.z);

   kv1[0]->CalcShape(shape1_x, ijk[0], ip.x);
   kv1[1]->CalcShape(shape1_y, ijk[1], ip.y);
   kv1[2]->CalcShape(shape1_z, ijk[2], ip.z);

   shape = 0.0;
   int o = 0;
   for (int  k = 0; k <= orders[2]+1; k++)
   {
      const real_t sz1 = shape1_z(k);
      for (int j = 0; j <= orders[1]+1; j++)
      {
         const real_t sy1_sz1 = shape1_y(j)*sz1;
         for (int i = 0; i <= orders[0]; i++, o++)
         {
            shape(o,0) = shape_x(i)*sy1_sz1;
         }
      }
   }

   for (int  k = 0; k <= orders[2]+1; k++)
   {
      const real_t sz1 = shape1_z(k);
      for (int j = 0; j <= orders[1]; j++)
      {
         const real_t sy_sz1 = shape_y(j)*sz1;
         for (int i = 0; i <= orders[0]+1; i++, o++)
         {
            shape(o,1) = shape1_x(i)*sy_sz1;
         }
      }
   }

   for (int  k = 0; k <= orders[2]; k++)
   {
      const real_t sz = shape_z(k);
      for (int j = 0; j <= orders[1]+1; j++)
      {
         const real_t sy1_sz = shape1_y(j)*sz;
         for (int i = 0; i <= orders[0]+1; i++, o++)
         {
            shape(o,2) = shape1_x(i)*sy1_sz;
         }
      }
   }
}

void NURBS_HCurl3DFiniteElement::CalcVShape(ElementTransformation &Trans,
                                            DenseMatrix &shape) const
{
   CalcVShape(Trans.GetIntPoint(), shape);
   const DenseMatrix & JI = Trans.InverseJacobian();
   MFEM_ASSERT(JI.Width() == 3 && JI.Height() == 3,
               "NURBS_HCurl3DFiniteElement must be in a"
               "3 dimensional spaces");
   for (int i=0; i<dof; i++)
   {
      real_t sx = shape(i, 0);
      real_t sy = shape(i, 1);
      real_t sz = shape(i, 2);
      shape(i, 0) = sx * JI(0, 0) + sy * JI(1, 0) + sz * JI(2, 0);
      shape(i, 1) = sx * JI(0, 1) + sy * JI(1, 1) + sz * JI(2, 1);
      shape(i, 2) = sx * JI(0, 2) + sy * JI(1, 2) + sz * JI(2, 2);
   }
}

void NURBS_HCurl3DFiniteElement::CalcCurlShape(const IntegrationPoint &ip,
                                               DenseMatrix &curl_shape) const
{
   kv[0]->CalcShape ( shape_x, ijk[0], ip.x);
   kv[1]->CalcShape ( shape_y, ijk[1], ip.y);
   kv[2]->CalcShape ( shape_z, ijk[2], ip.z);

   kv1[0]->CalcShape(shape1_x, ijk[0], ip.x);
   kv1[1]->CalcShape(shape1_y, ijk[1], ip.y);
   kv1[2]->CalcShape(shape1_z, ijk[2], ip.z);

   kv1[0]->CalcDShape(dshape1_x, ijk[0], ip.x);
   kv1[1]->CalcDShape(dshape1_y, ijk[1], ip.y);
   kv1[2]->CalcDShape(dshape1_z, ijk[2], ip.z);

   int o = 0;
   for (int  k = 0; k <= orders[2]+1; k++)
   {
      const real_t sz1 = shape1_z(k), dsz1 = dshape1_z(k);
      for (int j = 0; j <= orders[1]+1; j++)
      {
         const real_t sy1_dsz1 = shape1_y(j)*dsz1,
                      dsy1_sz1 = dshape1_y(j)*sz1;
         for (int i = 0; i <= orders[0]; i++, o++)
         {
            curl_shape(o,0) = 0.0;
            curl_shape(o,1) =  shape_x(i)*sy1_dsz1;
            curl_shape(o,2) = -shape_x(i)*dsy1_sz1;
         }
      }
   }

   for (int  k = 0; k <= orders[2]+1; k++)
   {
      const real_t sz1 = shape1_z(k), dsz1 = dshape1_z(k);
      for (int j = 0; j <= orders[1]; j++)
      {
         const real_t sy_dsz1 = shape_y(j)*dsz1,
                      sy_sz1 = shape_y(j)*sz1;
         for (int i = 0; i <= orders[0]+1; i++, o++)
         {
            curl_shape(o,0) = -shape1_x(i)*sy_dsz1;
            curl_shape(o,1) = 0.0;
            curl_shape(o,2) = dshape1_x(i)*sy_sz1;
         }
      }
   }

   for (int  k = 0; k <= orders[2]; k++)
   {
      const real_t sz = shape_z(k);
      for (int j = 0; j <= orders[1]+1; j++)
      {
         const real_t sy1_sz = shape1_y(j)*sz,
                      dsy1_sz = dshape1_y(j)*sz;
         for (int i = 0; i <= orders[0]+1; i++, o++)
         {
            curl_shape(o,0) = shape1_x(i)*dsy1_sz;
            curl_shape(o,1) = -dshape1_x(i)*sy1_sz;
            curl_shape(o,2) = 0.0;

         }
      }
   }
}

void NURBS_HCurl3DFiniteElement::Project(VectorCoefficient &vc,
                                         ElementTransformation &Trans,
                                         Vector &dofs) const
{
   MFEM_ASSERT(dofs.Size() == dof, "");
   MFEM_ASSERT(vc.GetVDim() == 3, "");
   Vector x(3), xm(3);
   IntegrationPoint ip;

   int o = 0;
   for (int k = 0; k <= orders[2]+1; k++)
   {
      real_t kz = kv1[2]->GetBotella(ijk[2] + k);
      if (!kv1[2]->inSpan(kz, ijk[2]+orders[2]+1))
      {
         o += (orders[0] + 1)*(orders[1] + 2);
         continue;
      }
      ip.z = kv1[2]->GetRefPoint(kz, ijk[2]+orders[2]+1);
      for (int j = 0; j <= orders[1]+1; j++)
      {
         real_t ky = kv1[1]->GetBotella(ijk[1] + j);
         if (!kv1[1]->inSpan(ky, ijk[1]+orders[1]+1))
         {
            o += orders[0] + 1;
            continue;
         }
         ip.y = kv1[1]->GetRefPoint(ky, ijk[1]+orders[1]+1);
         for (int i = 0; i <= orders[0]; i++, o++)
         {
            real_t kx = kv[0]->GetBotella(ijk[0] + i);
            if (!kv[0]->inSpan(kx, ijk[0]+orders[0])) { continue; }
            ip.x = kv[0]->GetRefPoint(kx, ijk[0]+orders[0]);

            Trans.SetIntPoint(&ip);
            vc.Eval(x, Trans, ip);

            Trans.Jacobian().MultTranspose(x,xm);
            dofs(o) = xm(0);
         }
      }
   }

   for (int k = 0; k <= orders[2]+1; k++)
   {
      real_t kz = kv1[2]->GetBotella(ijk[2] + k);
      if (!kv1[2]->inSpan(kz, ijk[2]+orders[2]+1))
      {
         o += (orders[0] + 2)*(orders[1] + 1);
         continue;
      }
      ip.z = kv1[2]->GetRefPoint(kz, ijk[2]+orders[2]+1);
      for (int j = 0; j <= orders[1]; j++)
      {
         real_t ky = kv[1]->GetBotella(ijk[1] + j);
         if (!kv[1]->inSpan(ky, ijk[1]+orders[1]))
         {
            o += orders[0] + 2;
            continue;
         }
         ip.y = kv[1]->GetRefPoint(ky, ijk[1]+orders[1]);
         for (int i = 0; i <= orders[0]+1; i++, o++)
         {
            real_t kx = kv1[0]->GetBotella(ijk[0] + i);
            if (!kv1[0]->inSpan(kx, ijk[0]+orders[0]+1)) { continue; }
            ip.x = kv1[0]->GetRefPoint(kx, ijk[0]+orders[0]+1);

            Trans.SetIntPoint(&ip);
            vc.Eval(x, Trans, ip);

            Trans.Jacobian().MultTranspose(x,xm);
            dofs(o) = xm(1);
         }
      }
   }

   for (int k = 0; k <= orders[2]; k++)
   {
      real_t kz = kv[2]->GetBotella(ijk[2] + k);
      if (!kv[2]->inSpan(kz, ijk[2]+orders[2]))
      {
         o += (orders[0] + 2)*(orders[1] + 2);
         continue;
      }
      ip.z = kv[2]->GetRefPoint(kz, ijk[2]+orders[2]);
      for (int j = 0; j <= orders[1]+1; j++)
      {
         real_t ky = kv1[1]->GetBotella(ijk[1] + j);
         if (!kv1[1]->inSpan(ky, ijk[1]+orders[1]+1))
         {
            o += orders[0] + 2;
            continue;
         }
         ip.y = kv1[1]->GetRefPoint(ky, ijk[1]+orders[1]+1);
         for (int i = 0; i <= orders[0]+1; i++, o++)
         {
            real_t kx = kv1[0]->GetBotella(ijk[0] + i);
            if (!kv1[0]->inSpan(kx, ijk[0]+orders[0]+1)) { continue; }
            ip.x = kv1[0]->GetRefPoint(kx, ijk[0]+orders[0]+1);

            Trans.SetIntPoint(&ip);
            vc.Eval(x, Trans, ip);

            Trans.Jacobian().MultTranspose(x,xm);
            dofs(o) = xm(2);
         }
      }
   }

}


NURBS_HCurl3DFiniteElement::~NURBS_HCurl3DFiniteElement()
{
   if (kv1[0]) { delete kv1[0]; }
   if (kv1[1]) { delete kv1[1]; }
   if (kv1[2]) { delete kv1[2]; }
}

}
