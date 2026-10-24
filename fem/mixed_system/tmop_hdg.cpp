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

#include "tmop_hdg.hpp"
#include "../gridfunc.hpp"

namespace mfem
{

real_t HDG_TMOP_Integrator::GetFaceEnergy(
   const FiniteElement &el1, const FiniteElement &el2,
   FaceElementTransformations &Tr, const Vector &elfun)
{
   const int dim = el1.GetDim();
   MFEM_ASSERT(dim == 2, "Not implemented");

   const int ndof1 = el1.GetDof();
   const int ndof2 = (Tr.Elem2No >= 0)?(el2.GetDof()):(0);

   DenseMatrix elfun1, elfun2, mq;
   Vector grad(dim), nor(dim), nh(dim), ni(dim);
   shape1.SetSize(ndof1);
   dshape1.SetSize(ndof1, dim);

   if (x_0)
   {
      elfun1.SetSize(ndof1, dim);
      Vector elfun1v(elfun1.GetData(), ndof1 * dim);
      x_0->GetElementDofValues(Tr.Elem1No, elfun1v);
      Vector elfun_1(const_cast<Vector&>(elfun), 0, ndof1 * dim);
      elfun1v += elfun_1;
   }
   else
   {
      elfun1.MakeRef(const_cast<Memory<real_t>&>(elfun.GetMemory()), 0, ndof1, dim);
   }

   if (ndof2)
   {
      shape2.SetSize(ndof2);
      dshape2.SetSize(ndof2, dim);
      if (x_0)
      {
         elfun2.SetSize(ndof2, dim);
         Vector elfun2v(elfun2.GetData(), ndof2 * dim);
         x_0->GetElementDofValues(Tr.Elem2No, elfun2v);
         Vector elfun_2(const_cast<Vector&>(elfun), ndof1 * dim, ndof2 * dim);
         elfun2v += elfun_2;
      }
      else
      {
         elfun2.MakeRef(const_cast<Memory<real_t>&>(elfun.GetMemory()), ndof1 * dim,
                        ndof2, dim);
      }
   }

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int order;
      // Assuming order(u)==order(mesh)
      if (Tr.Elem2No >= 0)
         order = (std::min(Tr.Elem1->OrderW(), Tr.Elem2->OrderW()) +
                  2*std::max(el1.GetOrder(), el2.GetOrder()));
      else
      {
         order = Tr.Elem1->OrderW() + 2*el1.GetOrder();
      }
      if (el1.Space() == FunctionSpace::Pk)
      {
         order++;
      }
      ir = &IntRules.Get(Tr.GetGeometryType(), order);
   }

   real_t energy = 0.;

   for (int p = 0; p < ir->GetNPoints(); p++)
   {
      const IntegrationPoint &ip = ir->IntPoint(p);

      // Set the integration point in the face and the neighboring elements
      Tr.SetAllIntPoints(&ip);

      const IntegrationPoint &eip1 = Tr.GetElement1IntPoint();
      Tr.Loc1.Transf.SetIntPoint(&ip);
      const DenseMatrix &Jloc1 = Tr.Loc1.Transf.Jacobian();
      const Vector JLoc1v(Jloc1.GetData(), dim);
      el1.CalcDShape(eip1, dshape1);
      dshape1.Mult(JLoc1v, shape1);
      elfun1.MultTranspose(shape1, grad);
      nor(0) = +grad(1);
      nor(1) = -grad(0);

      real_t wn = ip.weight/Tr.Elem1->Weight();

      if (!MQ)
      {
         if (Q)
         {
            wn *= Q->Eval(*Tr.Elem1, eip1);
         }
         ni.Set(wn, nor);
      }
      else
      {
         nh.Set(wn, nor);
         MQ->Eval(mq, *Tr.Elem1, eip1);
         mq.MultTranspose(nh, ni);
      }

      const real_t u_p = u.GetValue(*Tr.Elem1, eip1);
      const real_t uhat_p = uhat.GetValue(Tr, ip);
      const real_t d_p = u_p - uhat_p;
      const real_t w = d_p*d_p * td;

      energy += w *  (ni * nor);

      if (ndof2)
      {
         const IntegrationPoint &eip2 = Tr.GetElement2IntPoint();
         Tr.Loc2.Transf.SetIntPoint(&ip);
         const DenseMatrix &Jloc2 = Tr.Loc2.Transf.Jacobian();
         const Vector JLoc2v(Jloc2.GetData(), dim);
         el2.CalcDShape(eip2, dshape2);
         dshape2.Mult(JLoc2v, shape2);
         elfun2.MultTranspose(shape2, grad);
         nor(0) = +grad(1);
         nor(1) = -grad(0);

         real_t wn = ip.weight/Tr.Elem2->Weight();

         if (!MQ)
         {
            if (Q)
            {
               wn *= Q->Eval(*Tr.Elem2, eip2);
            }
            ni.Set(wn, nor);
         }
         else
         {
            nh.Set(wn, nor);
            MQ->Eval(mq, *Tr.Elem2, eip2);
            mq.MultTranspose(nh, ni);
         }

         const real_t u_p = u.GetValue(*Tr.Elem2, eip2);
         const real_t uhat_p = uhat.GetValue(Tr, ip);
         const real_t d_p = u_p - uhat_p;
         const real_t w = d_p*d_p * td;

         energy += w *  (ni * nor);
      }
   }

   return energy;
}

void HDG_TMOP_Integrator::AssembleFaceVector(
   const FiniteElement &el1, const FiniteElement &el2,
   FaceElementTransformations &Tr, const Vector &elfun, Vector &elvect)
{
   const int dim = el1.GetDim();
   MFEM_ASSERT(dim == 2, "Not implemented");

   const int ndof1 = el1.GetDof();
   const int ndof2 = (Tr.Elem2No >= 0)?(el2.GetDof()):(0);

   DenseMatrix elfun1, elfun2, mq;
   Vector grad(dim), nor(dim), nh(dim), ni(dim);
   shape1.SetSize(ndof1);
   dshape1.SetSize(ndof1, dim);
   if (x_0)
   {
      elfun1.SetSize(ndof1, dim);
      Vector elfun1v(elfun1.GetData(), ndof1 * dim);
      x_0->GetElementDofValues(Tr.Elem1No, elfun1v);
      Vector elfun_1(const_cast<Vector&>(elfun), 0, ndof1 * dim);
      elfun1v += elfun_1;
   }
   else
   {
      elfun1.MakeRef(const_cast<Memory<real_t>&>(elfun.GetMemory()), 0, ndof1, dim);
   }

   if (ndof2)
   {
      shape2.SetSize(ndof2);
      dshape2.SetSize(ndof2, dim);
      if (x_0)
      {
         elfun2.SetSize(ndof2, dim);
         Vector elfun2v(elfun2.GetData(), ndof2 * dim);
         x_0->GetElementDofValues(Tr.Elem2No, elfun2v);
         Vector elfun_2(const_cast<Vector&>(elfun), ndof1 * dim, ndof2 * dim);
         elfun2v += elfun_2;
      }
      else
      {
         elfun2.MakeRef(const_cast<Memory<real_t>&>(elfun.GetMemory()), ndof1 * dim,
                        ndof2, dim);
      }
   }

   elvect.SetSize((ndof1 + ndof2) * dim);
   elvect = 0.;

   Vector elvec1, elvec2;
   elvec1.MakeRef(elvect, 0, ndof1 * dim);
   if (ndof2)
   {
      elvec2.MakeRef(elvect, ndof1 * dim, ndof2 * dim);
   }

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int order;
      // Assuming order(u)==order(mesh)
      if (Tr.Elem2No >= 0)
         order = (std::min(Tr.Elem1->OrderW(), Tr.Elem2->OrderW()) +
                  2*std::max(el1.GetOrder(), el2.GetOrder()));
      else
      {
         order = Tr.Elem1->OrderW() + 2*el1.GetOrder();
      }
      if (el1.Space() == FunctionSpace::Pk)
      {
         order++;
      }
      ir = &IntRules.Get(Tr.GetGeometryType(), order);
   }

   for (int p = 0; p < ir->GetNPoints(); p++)
   {
      const IntegrationPoint &ip = ir->IntPoint(p);

      // Set the integration point in the face and the neighboring elements
      Tr.SetAllIntPoints(&ip);

      const IntegrationPoint &eip1 = Tr.GetElement1IntPoint();
      Tr.Loc1.Transf.SetIntPoint(&ip);
      const DenseMatrix &Jloc1 = Tr.Loc1.Transf.Jacobian();
      const Vector JLoc1v(Jloc1.GetData(), dim);
      el1.CalcDShape(eip1, dshape1);
      dshape1.Mult(JLoc1v, shape1);
      elfun1.MultTranspose(shape1, grad);
      nor(0) = +grad(1);
      nor(1) = -grad(0);

      real_t wn = ip.weight/Tr.Elem1->Weight();
      if (!MQ)
      {
         if (Q)
         {
            wn *= Q->Eval(*Tr.Elem1, eip1);
         }
         ni.Set(wn, nor);
      }
      else
      {
         nh.Set(wn, nor);
         MQ->Eval(mq, *Tr.Elem1, eip1);
         mq.MultTranspose(nh, ni);
      }

      const real_t u_p = u.GetValue(*Tr.Elem1, eip1);
      const real_t uhat_p = uhat.GetValue(Tr, ip);
      const real_t d_p = u_p - uhat_p;
      const real_t w = 2. * d_p*d_p * td;

      for (int i = 0; i < ndof1; i++)
      {
         elvec1(ndof1 * 0 + i) -= w * ni(1) * shape1(i);
      }
      for (int i = 0; i < ndof1; i++)
      {
         elvec1(ndof1 * 1 + i) += w * ni(0) * shape1(i);
      }

      if (ndof2)
      {
         const IntegrationPoint &eip2 = Tr.GetElement2IntPoint();
         Tr.Loc2.Transf.SetIntPoint(&ip);
         const DenseMatrix &Jloc2 = Tr.Loc2.Transf.Jacobian();
         const Vector JLoc2v(Jloc2.GetData(), dim);
         el2.CalcDShape(eip2, dshape2);
         dshape2.Mult(JLoc2v, shape2);
         elfun2.MultTranspose(shape2, grad);
         nor(0) = +grad(1);
         nor(1) = -grad(0);

         real_t wn = ip.weight/Tr.Elem2->Weight();
         if (!MQ)
         {
            if (Q)
            {
               wn *= Q->Eval(*Tr.Elem2, eip2);
            }
            ni.Set(wn, nor);
         }
         else
         {
            nh.Set(wn, nor);
            MQ->Eval(mq, *Tr.Elem2, eip2);
            mq.MultTranspose(nh, ni);
         }

         const real_t u_p = u.GetValue(*Tr.Elem2, eip2);
         const real_t uhat_p = uhat.GetValue(Tr, ip);
         const real_t d_p = u_p - uhat_p;
         const real_t w = 2. * d_p*d_p * td;

         for (int i = 0; i < ndof2; i++)
         {
            elvec2(ndof2 * 0 + i) -= w * ni(1) * shape2(i);
         }
         for (int i = 0; i < ndof2; i++)
         {
            elvec2(ndof2 * 1 + i) += w * ni(0) * shape2(i);
         }
      }
   }
}

void HDG_TMOP_Integrator::AssembleFaceGrad(
   const FiniteElement &el1, const FiniteElement &el2,
   FaceElementTransformations &Tr, const Vector &elfun, DenseMatrix &elmat)
{
   const int dim = el1.GetDim();
   MFEM_ASSERT(dim == 2, "Not implemented");

   const int ndof1 = el1.GetDof();
   const int ndof2 = (Tr.Elem2No >= 0)?(el2.GetDof()):(0);

   DenseMatrix mq;
   shape1.SetSize(ndof1);
   dshape1.SetSize(ndof1, dim);
   if (ndof2)
   {
      shape2.SetSize(ndof2);
      dshape2.SetSize(ndof2, dim);
   }

   elmat.SetSize((ndof1 + ndof2) * dim);
   elmat = 0.;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int order;
      // Assuming order(u)==order(mesh)
      if (Tr.Elem2No >= 0)
         order = (std::min(Tr.Elem1->OrderW(), Tr.Elem2->OrderW()) +
                  2*std::max(el1.GetOrder(), el2.GetOrder()));
      else
      {
         order = Tr.Elem1->OrderW() + 2*el1.GetOrder();
      }
      if (el1.Space() == FunctionSpace::Pk)
      {
         order++;
      }
      ir = &IntRules.Get(Tr.GetGeometryType(), order);
   }

   for (int p = 0; p < ir->GetNPoints(); p++)
   {
      const IntegrationPoint &ip = ir->IntPoint(p);

      // Set the integration point in the face and the neighboring elements
      Tr.SetAllIntPoints(&ip);

      const IntegrationPoint &eip1 = Tr.GetElement1IntPoint();
      Tr.Loc1.Transf.SetIntPoint(&ip);
      const DenseMatrix &Jloc1 = Tr.Loc1.Transf.Jacobian();
      const Vector JLoc1v(Jloc1.GetData(), dim);
      el1.CalcDShape(eip1, dshape1);
      dshape1.Mult(JLoc1v, shape1);

      real_t wn = ip.weight/Tr.Elem1->Weight();

      if (!MQ)
      {
         if (Q)
         {
            wn *= Q->Eval(*Tr.Elem1, eip1);
         }
      }
      else
      {
         MQ->Eval(mq, *Tr.Elem1, eip1);
      }

      const real_t u_p = u.GetValue(*Tr.Elem1, eip1);
      const real_t uhat_p = uhat.GetValue(Tr, ip);
      const real_t d_p = u_p - uhat_p;
      const real_t w = 2. * d_p*d_p * td * wn;

      for (int d_tr = 0; d_tr < dim; d_tr++)
         for (int d_te = 0; d_te < dim; d_te++)
         {
            const real_t s = ((d_tr + d_te) % dim)?(-1.):(+1.);
            const real_t k = (MQ)?(mq((d_te+1) % dim, (d_tr+1) % dim)):(1.);
            const real_t skw = s * k * w;
            for (int j = 0; j < ndof1; j++)
               for (int i = 0; i < ndof1; i++)
               {
                  elmat(ndof1 * d_te + i, ndof1 * d_tr + j) += skw * shape1(i) * shape1(j);
               }
         }

      if (ndof2)
      {
         const IntegrationPoint &eip2 = Tr.GetElement2IntPoint();
         Tr.Loc2.Transf.SetIntPoint(&ip);
         const DenseMatrix &Jloc2 = Tr.Loc2.Transf.Jacobian();
         const Vector JLoc2v(Jloc2.GetData(), dim);
         el2.CalcDShape(eip2, dshape2);
         dshape2.Mult(JLoc2v, shape2);

         real_t wn = ip.weight/Tr.Elem2->Weight();

         if (!MQ)
         {
            if (Q)
            {
               wn *= Q->Eval(*Tr.Elem1, eip1);
            }
         }
         else
         {
            MQ->Eval(mq, *Tr.Elem1, eip1);
         }

         const real_t u_p = u.GetValue(*Tr.Elem2, eip2);
         const real_t uhat_p = uhat.GetValue(Tr, ip);
         const real_t d_p = u_p - uhat_p;
         const real_t w = 2. * d_p*d_p * td * wn;

         for (int d_tr = 0; d_tr < dim; d_tr++)
            for (int d_te = 0; d_te < dim; d_te++)
            {
               const real_t s = ((d_tr + d_te) % dim)?(-1.):(+1.);
               const real_t k = (MQ)?(mq((d_te+1) % dim, (d_tr+1) % dim)):(1.);
               const real_t skw = s * k * w;
               for (int j = 0; j < ndof2; j++)
                  for (int i = 0; i < ndof2; i++)
                  {
                     elmat(ndof1 * dim + ndof2 * d_te + i,
                           ndof1 * dim + ndof2 * d_tr + j) += skw * shape2(i) * shape2(j);
                  }
            }
      }

   }
}

} // namespace mfem
