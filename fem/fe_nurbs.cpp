// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
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
#include "../mesh/nurbs.hpp"

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

   double sum = 0.0;
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

   double sum = 0.0, dsum = 0.0;
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

   double sum = 0.0, dsum = 0.0, d2sum = 0.0;
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

   double sum = 0.0;
   for (int o = 0, j = 0; j <= orders[1]; j++)
   {
      const double sy = shape_y(j);
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
   double sum, dsum[2];

   kv[0]->CalcShape ( shape_x, ijk[0], ip.x);
   kv[1]->CalcShape ( shape_y, ijk[1], ip.y);

   kv[0]->CalcDShape(dshape_x, ijk[0], ip.x);
   kv[1]->CalcDShape(dshape_y, ijk[1], ip.y);

   sum = dsum[0] = dsum[1] = 0.0;
   for (int o = 0, j = 0; j <= orders[1]; j++)
   {
      const double sy = shape_y(j), dsy = dshape_y(j);
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
   double sum, dsum[2], d2sum[3];

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
      const double sy = shape_y(j), dsy = dshape_y(j), d2sy = d2shape_y(j);
      for (int i = 0; i <= orders[0]; i++, o++)
      {
         const double sx = shape_x(i), dsx = dshape_x(i), d2sx = d2shape_x(i);
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

   double sum = 0.0;
   for (int o = 0, k = 0; k <= orders[2]; k++)
   {
      const double sz = shape_z(k);
      for (int j = 0; j <= orders[1]; j++)
      {
         const double sy_sz = shape_y(j)*sz;
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
   double sum, dsum[3];

   kv[0]->CalcShape ( shape_x, ijk[0], ip.x);
   kv[1]->CalcShape ( shape_y, ijk[1], ip.y);
   kv[2]->CalcShape ( shape_z, ijk[2], ip.z);

   kv[0]->CalcDShape(dshape_x, ijk[0], ip.x);
   kv[1]->CalcDShape(dshape_y, ijk[1], ip.y);
   kv[2]->CalcDShape(dshape_z, ijk[2], ip.z);

   sum = dsum[0] = dsum[1] = dsum[2] = 0.0;
   for (int o = 0, k = 0; k <= orders[2]; k++)
   {
      const double sz = shape_z(k), dsz = dshape_z(k);
      for (int j = 0; j <= orders[1]; j++)
      {
         const double  sy_sz  =  shape_y(j)* sz;
         const double dsy_sz  = dshape_y(j)* sz;
         const double  sy_dsz =  shape_y(j)*dsz;
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
   double sum, dsum[3], d2sum[6];

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
      const double sz = shape_z(k), dsz = dshape_z(k), d2sz = d2shape_z(k);
      for (int j = 0; j <= orders[1]; j++)
      {
         const double sy = shape_y(j), dsy = dshape_y(j), d2sy = d2shape_y(j);
         for (int i = 0; i <= orders[0]; i++, o++)
         {
            const double sx = shape_x(i), dsx = dshape_x(i), d2sx = d2shape_x(i);
            sum += ( u(o) = sx*sy*sz*weights(o) );

            dsum[0] += ( du(o,0) = dsx*sy*sz*weights(o) );
            dsum[1] += ( du(o,1) = sx*dsy*sz*weights(o) );
            dsum[2] += ( du(o,2) = sx*sy*dsz*weights(o) );

            d2sum[0] += ( hessian(o,0) = d2sx*sy*sz*weights(o) );
            d2sum[1] += ( hessian(o,1) = dsx*dsy*sz*weights(o) );
            d2sum[2] += ( hessian(o,2) = dsx*sy*dsz*weights(o) );

            d2sum[3] += ( hessian(o,3) = sx*dsy*dsz*weights(o) );

            d2sum[4] += ( hessian(o,4) = sx*sy*d2sz*weights(o) );
            d2sum[5] += ( hessian(o,5) = sx*d2sy*sz*weights(o) );
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

}
