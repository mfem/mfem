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







void NURBS_HDiv1DFiniteElement::SetOrder() const
{
   orders[0] = kv[0]->GetOrder();

   if (kv1[0]) { delete kv1[0]; }

   kv1[0] = kv[0]->DegreeElevate(1);

   shape_x.SetSize(orders[0]+1);
   dshape_x.SetSize(orders[0]+1);
   d2shape_x.SetSize(orders[0]+1);

   shape1_x.SetSize(orders[0]+2);
   dshape1_x.SetSize(orders[0]+2);
   d2shape1_x.SetSize(orders[0]+2);

   order = orders[0];
   dof = (orders[0] + 2) + (orders[1] + 1);
   u.SetSize(dof);
   du.SetSize(dof);
   weights.SetSize(dof);
}

void NURBS_HDiv1DFiniteElement::CalcVShape(const IntegrationPoint &ip,
                                           DenseMatrix &shape) const
{
   kv[0]->CalcShape(shape_x, ijk[0], ip.x);
   kv1[0]->CalcShape(shape1_x, ijk[0], ip.x);

   int o = 0;
   for (int i = 0; i <= orders[0]+1; i++, o++)
   {
      shape(o,0) = shape1_x(i);
      shape(o,1) = 0.0;
   }

   for (int i = 0; i <= orders[0]; i++, o++)
   {
      shape(o,0) = 0.0;
      shape(o,1) = shape_x(i);
   }
}

void NURBS_HDiv1DFiniteElement::CalcVShape(ElementTransformation &Trans,
                                           DenseMatrix &shape) const
{
   CalcVShape(Trans.GetIntPoint(), shape);
   const DenseMatrix & J = Trans.Jacobian();
   MFEM_ASSERT(J.Width() == 2 && J.Height() == 2,
               "RT_R2D_FiniteElement cannot be embedded in "
               "3 dimensional spaces");
   /*  for (int i=0; i<dof; i++)
     {
        double sx = shape(i, 0);
        double sy = shape(i, 1);
        shape(i, 0) = sx * J(0, 0) + sy * J(0, 1);
        shape(i, 1) = sx * J(1, 0) + sy * J(1, 1);
     }
     shape *= (1.0 / Trans.Weight());*/
}
/*
void NURBS_HDiv1DFiniteElement::CalcDivShape(const IntegrationPoint &ip,
                                             Vector &divshape) const
{
   kv[0]->CalcShape ( shape_x, ijk[0], ip.x);
   kv[1]->CalcShape ( shape_y, ijk[1], ip.y);
   kv1[0]->CalcShape ( shape1_x, ijk[0], ip.x);
   kv1[1]->CalcShape ( shape1_y, ijk[1], ip.y);

   kv[0]->CalcDShape(dshape_x, ijk[0], ip.x);
   kv[1]->CalcDShape(dshape_y, ijk[1], ip.y);
   kv1[0]->CalcDShape(dshape1_x, ijk[0], ip.x);
   kv1[1]->CalcDShape(dshape1_y, ijk[1], ip.y);

   int o = 0;
   for (int j = 0; j <= orders[1]; j++)
   {
      const double sy = shape_y(j), dsy = dshape_y(j);
      for (int i = 0; i <= orders[0]+1; i++, o++)
      {
         divshape(o) = dshape1_x(i)*sy + shape1_x(i)*dsy;
      }
   }

   for (int j = 0; j <= orders[1]+1; j++)
   {
      const double sy1 = shape1_y(j), dsy1 = dshape1_y(j);
      for (int i = 0; i <= orders[0]; i++, o++)
      {
         divshape(o) = dshape_x(i)*sy1 + shape_x(i)*dsy1;
      }
   }

   if (divshape.Norml2() != divshape.Norml2())
   {
      mfem::out<<" shape_x   = ";shape_x.Print();
      mfem::out<<" shape_y   = ";shape_y.Print();
      mfem::out<<" shape1_x  = ";shape1_x.Print();
      mfem::out<<" shape1_y  = ";shape1_y.Print();

      mfem::out<<" dshape_x  = ";dshape_x.Print();
      mfem::out<<" dshape_y  = ";dshape_y.Print();
      mfem::out<<" dshape1_x = ";dshape1_x.Print();
      mfem::out<<" dshape1_y = ";dshape1_y.Print();

      mfem_error("");
   }
}

void NURBS_HDiv1DFiniteElement::CalcVDShape(const IntegrationPoint &ip,
                                            DenseTensor &dshape) const
{
   double sum, dsum[2];

   kv[0]->CalcShape ( shape_x, ijk[0], ip.x);
   kv[1]->CalcShape ( shape_y, ijk[1], ip.y);
   kv1[0]->CalcShape ( shape1_x, ijk[0]+1, ip.x);
   kv1[1]->CalcShape ( shape1_y, ijk[1]+1, ip.y);

   kv[0]->CalcDShape(dshape_x, ijk[0], ip.x);
   kv[1]->CalcDShape(dshape_y, ijk[1], ip.y);
   kv1[0]->CalcDShape(dshape1_x, ijk[0]+1, ip.x);
   kv1[1]->CalcDShape(dshape1_y, ijk[1]+1, ip.y);

   sum = dsum[0] = dsum[1] = 0.0;
   for (int o = 0, j = 0; j <= orders[1]; j++)
   {
      const double sy = shape_y(j), dsy = dshape_y(j);
      for (int i = 0; i <= orders[0]+1; i++, o++)
      {
     //    sum += ( u(o) = shape_x(i)*sy*weights(o) );
        // dsum[0] += ( dshape(o,0) = dshape_x(i)*sy *weights(o) );
       //  dsum[1] += ( dshape(o,1) =  shape_x(i)*dsy*weights(o) );
       dshape(o,0,0) = dshape1_x(i)*sy;
       dshape(o,0,1) = 0.0;
       dshape(o,1,0) = shape1_x(i)*dsy;
       dshape(o,1,1) = 0.0;
      }
   }

   for (int o = 0, j = 0; j <= orders[1]+1; j++)
   {
      const double sy = shape1_y(j), dsy = dshape1_y(j);
      for (int i = 0; i <= orders[0]; i++, o++)
      {
     //    sum += ( u(o) = shape_x(i)*sy*weights(o) );
        // dsum[0] += ( dshape(o,0) = dshape_x(i)*sy *weights(o) );
       //  dsum[1] += ( dshape(o,1) =  shape_x(i)*dsy*weights(o) );
       dshape(o,0,0) =  0.0;
       dshape(o,0,1) = dshape_x(i)*sy;
       dshape(o,1,0) =  0.0;
       dshape(o,1,1) = shape_x(i)*dsy;
      }
   }

}

void NURBS_HDiv1DFiniteElement::CalcVHessian(const IntegrationPoint &ip,
                                             DenseTensor &Hessian) const
{
   kv[0]->CalcShape ( shape_x, ijk[0], ip.x);
   kv[1]->CalcShape ( shape_y, ijk[1], ip.y);

   kv[0]->CalcDShape(dshape_x, ijk[0], ip.x);
   kv[1]->CalcDShape(dshape_y, ijk[1], ip.y);

   kv[0]->CalcD2Shape(d2shape_x, ijk[0], ip.x);
   kv[1]->CalcD2Shape(d2shape_y, ijk[1], ip.y);

   kv1[0]->CalcShape ( shape1_x, ijk[0]+1, ip.x);
   kv1[1]->CalcShape ( shape1_y, ijk[1]+1, ip.y);

   kv1[0]->CalcDShape(dshape1_x, ijk[0]+1, ip.x);
   kv1[1]->CalcDShape(dshape1_y, ijk[1]+1, ip.y);

   kv1[0]->CalcD2Shape(d2shape1_x, ijk[0]+1, ip.x);
   kv1[1]->CalcD2Shape(d2shape1_y, ijk[1]+1, ip.y);

   for (int o = 0, j = 0; j <= orders[1]; j++)
   {
      const double sy = shape_y(j), dsy = dshape_y(j), d2sy = d2shape_y(j);
      for (int i = 0; i <= orders[0]+1; i++, o++)
      {
         const double sx1 = shape1_x(i), dsx1 = dshape1_x(i), d2sx1 = d2shape1_x(i);
         Hessian(o,0,0) = d2sx1*sy;
         Hessian(o,1,0) = dsx1*dsy;
         Hessian(o,2,0) = sx1*d2sy;

         Hessian(o,0,1) = 0.0;
         Hessian(o,1,1) = 0.0;
         Hessian(o,2,1) = 0.0;
      }
   }
   for (int o = 0, j = 0; j <= orders[1]+1; j++)
   {
      const double sy1 = shape1_y(j), dsy1 = dshape1_y(j), d2sy1 = d2shape1_y(j);
      for (int i = 0; i <= orders[0]; i++, o++)
      {
         const double sx = shape_x(i), dsx = dshape_x(i), d2sx = d2shape_x(i);
         Hessian(o,0,0) = 0.0;
         Hessian(o,1,0) = 0.0;
         Hessian(o,2,0) = 0.0;

         Hessian(o,0,1) = d2sx*sy1;
         Hessian(o,1,1) = dsx*dsy1;
         Hessian(o,2,1) = sx*d2sy1;
      }
   }
}

*/


























void NURBS_HDiv2DFiniteElement::SetOrder() const
{
   orders[0] = kv[0]->GetOrder();
   orders[1] = kv[1]->GetOrder();

   if (kv1[0]) { delete kv1[0]; }
   if (kv1[1]) { delete kv1[1]; }

   kv1[0] = kv[0]->DegreeElevate(1);
   kv1[1] = kv[0]->DegreeElevate(1);

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
   /*
         double x,y;
         x = 0.0;
         for (int i = 0; i <= 100; i++)
         {
            kv[0]->CalcShape(shape_x, ijk[0], x);
            kv1[0]->CalcShape(shape1_x, ijk[0], x);
            cout<<"x = "<<x<<" "
                << shape_x[0]<<" "
                << shape_x[1]<<" "
                << shape1_x[0]<<" "
                << shape1_x[1]<<" "
                << shape1_x[2]<<std::endl;
            x += 0.01;
         }


         y = 0.0;
         for (int i = 0; i <= 100; i++)
         {
            kv[1]->CalcShape(shape_y, ijk[1], y);
            kv1[1]->CalcShape(shape1_y, ijk[1], y);
             cout<<"y = "<<y<<" "
                << shape_y[0]<<" "
                << shape_y[1]<<" "
                << shape1_y[0]<<" "
                << shape1_y[1]<<" "
                << shape1_y[2]<<std::endl;
            y += 0.01;
         }
   */











   kv[0]->CalcShape(shape_x, ijk[0], ip.x);
   kv[1]->CalcShape(shape_y, ijk[1], ip.y);

   kv1[0]->CalcShape(shape1_x, ijk[0], ip.x);
   kv1[1]->CalcShape(shape1_y, ijk[1], ip.y);


   int o = 0;
   for (int j = 0; j <= orders[1]; j++)
   {
      const double sy = shape_y(j);
      for (int i = 0; i <= orders[0]+1; i++, o++)
      {
         shape(o,0) = shape1_x(i)*sy;
         shape(o,1) = 0.0;
      }
   }

   for (int j = 0; j <= orders[1]+1; j++)
   {
      const double sy1 = shape1_y(j);
      for (int i = 0; i <= orders[0]; i++, o++)
      {
         shape(o,0) = 0.0;
         shape(o,1) = shape_x(i)*sy1;
      }
   }
   //std::cout<<ip.x <<" "<<ip.y<<"\n"; shape.Print();
   ///shape/= sum;*/
}

void NURBS_HDiv2DFiniteElement::CalcVShape(ElementTransformation &Trans,
                                           DenseMatrix &shape) const
{
   CalcVShape(Trans.GetIntPoint(), shape);
   const DenseMatrix & J = Trans.Jacobian();
   MFEM_ASSERT(J.Width() == 2 && J.Height() == 2,
               "RT_R2D_FiniteElement cannot be embedded in "
               "3 dimensional spaces");
   /* for (int i=0; i<dof; i++)
    {
       double sx = shape(i, 0);
       double sy = shape(i, 1);
       shape(i, 0) = sx * J(0, 0) + sy * J(0, 1);
       shape(i, 1) = sx * J(1, 0) + sy * J(1, 1);
    }
    shape *= (1.0 / Trans.Weight());*/
}

void NURBS_HDiv2DFiniteElement::CalcDivShape(const IntegrationPoint &ip,
                                             Vector &divshape) const
{
   kv[0]->CalcShape ( shape_x, ijk[0], ip.x);
   kv[1]->CalcShape ( shape_y, ijk[1], ip.y);
   kv1[0]->CalcShape ( shape1_x, ijk[0], ip.x);
   kv1[1]->CalcShape ( shape1_y, ijk[1], ip.y);

   kv[0]->CalcDShape(dshape_x, ijk[0], ip.x);
   kv[1]->CalcDShape(dshape_y, ijk[1], ip.y);
   kv1[0]->CalcDShape(dshape1_x, ijk[0], ip.x);
   kv1[1]->CalcDShape(dshape1_y, ijk[1], ip.y);

   int o = 0;
   for (int j = 0; j <= orders[1]; j++)
   {
      const double sy = shape_y(j), dsy = dshape_y(j);
      for (int i = 0; i <= orders[0]+1; i++, o++)
      {
         divshape(o) = dshape1_x(i)*sy;
      }
   }

   for (int j = 0; j <= orders[1]+1; j++)
   {
      const double sy1 = shape1_y(j), dsy1 = dshape1_y(j);
      for (int i = 0; i <= orders[0]; i++, o++)
      {
         divshape(o) = shape_x(i)*dsy1;
         
      }
   }

}

void NURBS_HDiv2DFiniteElement::CalcVDShape(const IntegrationPoint &ip,
                                            DenseTensor &dshape) const
{
   double sum, dsum[2];

   kv[0]->CalcShape ( shape_x, ijk[0], ip.x);
   kv[1]->CalcShape ( shape_y, ijk[1], ip.y);
   kv1[0]->CalcShape ( shape1_x, ijk[0]+1, ip.x);
   kv1[1]->CalcShape ( shape1_y, ijk[1]+1, ip.y);

   kv[0]->CalcDShape(dshape_x, ijk[0], ip.x);
   kv[1]->CalcDShape(dshape_y, ijk[1], ip.y);
   kv1[0]->CalcDShape(dshape1_x, ijk[0]+1, ip.x);
   kv1[1]->CalcDShape(dshape1_y, ijk[1]+1, ip.y);

   sum = dsum[0] = dsum[1] = 0.0;
   int o = 0 ;
   for (int j = 0; j <= orders[1]; j++)
   {
      const double sy = shape_y(j), dsy = dshape_y(j);
      for (int i = 0; i <= orders[0]+1; i++, o++)
      {
         //    sum += ( u(o) = shape_x(i)*sy*weights(o) );
         // dsum[0] += ( dshape(o,0) = dshape_x(i)*sy *weights(o) );
         //  dsum[1] += ( dshape(o,1) =  shape_x(i)*dsy*weights(o) );
         dshape(o,0,0) = dshape1_x(i)*sy;
         dshape(o,0,1) = 0.0;
         dshape(o,1,0) = shape1_x(i)*dsy;
         dshape(o,1,1) = 0.0;
      }
   }

   for (int j = 0; j <= orders[1]+1; j++)
   {
      const double sy = shape1_y(j), dsy = dshape1_y(j);
      for (int i = 0; i <= orders[0]; i++, o++)
      {
         //    sum += ( u(o) = shape_x(i)*sy*weights(o) );
         // dsum[0] += ( dshape(o,0) = dshape_x(i)*sy *weights(o) );
         //  dsum[1] += ( dshape(o,1) =  shape_x(i)*dsy*weights(o) );
         dshape(o,0,0) =  0.0;
         dshape(o,0,1) = dshape_x(i)*sy;
         dshape(o,1,0) =  0.0;
         dshape(o,1,1) = shape_x(i)*dsy;
      }
   }

   /*sum = 1.0/sum;
   dsum[0] *= sum*sum;
   dsum[1] *= sum*sum;

   for (int o = 0; o < dof; o++)
   {
      dshape(o,0) = dshape(o,0)*sum - u(o)*dsum[0];
      dshape(o,1) = dshape(o,1)*sum - u(o)*dsum[1];
   }*/
}

void NURBS_HDiv2DFiniteElement::CalcVHessian(const IntegrationPoint &ip,
                                             DenseTensor &Hessian) const
{
   kv[0]->CalcShape ( shape_x, ijk[0], ip.x);
   kv[1]->CalcShape ( shape_y, ijk[1], ip.y);

   kv[0]->CalcDShape(dshape_x, ijk[0], ip.x);
   kv[1]->CalcDShape(dshape_y, ijk[1], ip.y);

   kv[0]->CalcD2Shape(d2shape_x, ijk[0], ip.x);
   kv[1]->CalcD2Shape(d2shape_y, ijk[1], ip.y);

   kv1[0]->CalcShape ( shape1_x, ijk[0]+1, ip.x);
   kv1[1]->CalcShape ( shape1_y, ijk[1]+1, ip.y);

   kv1[0]->CalcDShape(dshape1_x, ijk[0]+1, ip.x);
   kv1[1]->CalcDShape(dshape1_y, ijk[1]+1, ip.y);

   kv1[0]->CalcD2Shape(d2shape1_x, ijk[0]+1, ip.x);
   kv1[1]->CalcD2Shape(d2shape1_y, ijk[1]+1, ip.y);

   int o = 0;
   for (int j = 0; j <= orders[1]; j++)
   {
      const double sy = shape_y(j), dsy = dshape_y(j), d2sy = d2shape_y(j);
      for (int i = 0; i <= orders[0]+1; i++, o++)
      {
         const double sx1 = shape1_x(i), dsx1 = dshape1_x(i), d2sx1 = d2shape1_x(i);
         Hessian(o,0,0) = d2sx1*sy;
         Hessian(o,1,0) = dsx1*dsy;
         Hessian(o,2,0) = sx1*d2sy;

         Hessian(o,0,1) = 0.0;
         Hessian(o,1,1) = 0.0;
         Hessian(o,2,1) = 0.0;
      }
   }
   for (int j = 0; j <= orders[1]+1; j++)
   {
      const double sy1 = shape1_y(j), dsy1 = dshape1_y(j), d2sy1 = d2shape1_y(j);
      for (int i = 0; i <= orders[0]; i++, o++)
      {
         const double sx = shape_x(i), dsx = dshape_x(i), d2sx = d2shape_x(i);
         Hessian(o,0,0) = 0.0;
         Hessian(o,1,0) = 0.0;
         Hessian(o,2,0) = 0.0;

         Hessian(o,0,1) = d2sx*sy1;
         Hessian(o,1,1) = dsx*dsy1;
         Hessian(o,2,1) = sx*d2sy1;
      }
   }
}

NURBS_HDiv2DFiniteElement::~NURBS_HDiv2DFiniteElement()
{
   if (kv1[0]) { delete kv1[0]; }
   if (kv1[1]) { delete kv1[1]; }
}


}
