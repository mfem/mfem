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

#include "bilininteg_hdg.hpp"

using std::min;
using std::max;

namespace mfem
{

void HDGConvectionCenteredIntegrator::AssembleHDGFaceMatrix(
   const FiniteElement &trace_el, const FiniteElement &el1,
   const FiniteElement &el2, FaceElementTransformations &Trans,
   DenseMatrix &elmat)
{
   MFEM_VERIFY(trace_el.GetMapType() == FiniteElement::VALUE, "");

   const int dim = el1.GetDim();
   const int tr_ndof = trace_el.GetDof();
   const int ndof1 = el1.GetDof();
   const int ndof2 = (Trans.Elem2No >= 0)?(el2.GetDof()):(0);
   const int el_ndof = ndof1 + ndof2;

   Vector vu(dim), nor(dim);

   tr_shape.SetSize(tr_ndof);
   shape1.SetSize(ndof1);
   shape2.SetSize(ndof2);

   elmat.SetSize(el_ndof + tr_ndof);
   elmat = 0.0;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int order;
      // Assuming order(u)==order(mesh)
      if (Trans.Elem2No >= 0)
         order = (min(Trans.Elem1->OrderW(), Trans.Elem2->OrderW()) +
                  2*max(el1.GetOrder(), el2.GetOrder()));
      else
      {
         order = Trans.Elem1->OrderW() + 2*el1.GetOrder();
      }
      if (el1.Space() == FunctionSpace::Pk)
      {
         order++;
      }
      ir = &IntRules.Get(Trans.GetGeometryType(), order);
   }

   for (int p = 0; p < ir->GetNPoints(); p++)
   {
      const IntegrationPoint &ip = ir->IntPoint(p);

      // Set the integration point in the face and the neighboring elements
      Trans.SetAllIntPoints(&ip);

      // Access the neighboring elements' integration point
      const IntegrationPoint &eip1 = Trans.GetElement1IntPoint();

      trace_el.CalcShape(ip, tr_shape);
      el1.CalcPhysShape(*Trans.Elem1, shape1);
      if (ndof2)
      {
         el2.CalcPhysShape(*Trans.Elem2, shape2);
      }

      u->Eval(vu, *Trans.Elem1, eip1);

      if (dim == 1)
      {
         nor(0) = 2*eip1.x - 1.0;
      }
      else
      {
         CalcOrtho(Trans.Jacobian(), nor);
      }

      const real_t un = vu * nor;
      const real_t a = alpha * un;
      const real_t b = fabs(alpha * un);
      // note: if |alpha/2|==|beta| then |a|==|b|, i.e. (a==b) or (a==-b)
      //       and therefore two blocks in the element matrix contribution
      //       (from the current quadrature point) are 0

      real_t w = ip.weight * b;
      if (w != 0.0)
      {
         // assemble the element matrix
         for (int i = 0; i < ndof1; i++)
            for (int j = 0; j < ndof1; j++)
            {
               elmat(i, j) += w * shape1(i) * shape1(j);
            }
         for (int i = 0; i < ndof2; i++)
            for (int j = 0; j < ndof2; j++)
            {
               elmat(ndof1+i, ndof1+j) += w * shape2(i) * shape2(j);
            }

         // assemble the constraint matrix
         for (int i = 0; i < tr_ndof; i++)
            for (int j = 0; j < ndof1; j++)
            {
               elmat(el_ndof+i, j) += w * tr_shape(i) * shape1(j);
            }
         for (int i = 0; i < tr_ndof; i++)
            for (int j = 0; j < ndof2; j++)
            {
               elmat(el_ndof+i, ndof1+j) += w * tr_shape(i) * shape2(j);
            }

      }

      // assemble the face matrix
      // note: that this term must be non-zero at the boundary for stability
      //       reasons, so the advective part is intentionally dropped here
      //       and must be compensated elsewhere
      w = ip.weight * ((ndof2)?(2.*b):(b));//<-- single face integration
      if (w != 0.0)
      {
         for (int i = 0; i < tr_ndof; i++)
            for (int j = 0; j < tr_ndof; j++)
            {
               elmat(el_ndof+i, el_ndof+j) -= w * tr_shape(i) * tr_shape(j);
            }
      }

      w = ip.weight * (b-a);
      if (w != 0.0)
      {
         // assemble the trace matrix (elem1)
         for (int i = 0; i < ndof1; i++)
            for (int j = 0; j < tr_ndof; j++)
            {
               elmat(i, el_ndof+j) -= w * shape1(i) * tr_shape(j);
            }
      }

      w = ip.weight * (b+a);
      if (w != 0.0)
      {
         // assemble the trace matrix (elem2)
         for (int i = 0; i < ndof2; i++)
            for (int j = 0; j < tr_ndof; j++)
            {
               elmat(ndof1+i, el_ndof+j) -= w * shape2(i) * tr_shape(j);
            }
      }
   }
}

void HDGConvectionCenteredIntegrator::AssembleHDGFaceMatrix(
   int side, const FiniteElement &trace_el, const FiniteElement &el,
   FaceElementTransformations &Trans, DenseMatrix &elmat)
{
   MFEM_VERIFY(trace_el.GetMapType() == FiniteElement::VALUE, "");

   const int dim = el.GetDim();
   const int tr_ndof = trace_el.GetDof();
   const int el_ndof = el.GetDof();

   Vector vu(dim), nor(dim);
   Vector &el_shape = shape1;

   tr_shape.SetSize(tr_ndof);
   el_shape.SetSize(el_ndof);

   elmat.SetSize(el_ndof + tr_ndof);
   elmat = 0.0;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int order;
      // Assuming order(u)==order(mesh)
      if (side != 0)
      {
         order = Trans.Elem2->OrderW();
      }
      else
      {
         order = Trans.Elem1->OrderW();
      }
      order += 2*el.GetOrder();
      if (el.Space() == FunctionSpace::Pk)
      {
         order++;
      }
      ir = &IntRules.Get(Trans.GetGeometryType(), order);
   }

   for (int p = 0; p < ir->GetNPoints(); p++)
   {
      const IntegrationPoint &ip = ir->IntPoint(p);

      // Set the integration point in the face and the neighboring elements
      Trans.SetAllIntPoints(&ip);

      // Access the neighboring elements' integration point
      const IntegrationPoint &eip1 = Trans.GetElement1IntPoint();

      trace_el.CalcShape(ip, tr_shape);

      if (side != 0)
      {
         el.CalcPhysShape(*Trans.Elem2, el_shape);
      }
      else
      {
         el.CalcPhysShape(*Trans.Elem1, el_shape);
      }

      u->Eval(vu, *Trans.Elem1, eip1);

      if (dim == 1)
      {
         nor(0) = 2*eip1.x - 1.0;
      }
      else
      {
         CalcOrtho(Trans.Jacobian(), nor);
      }

      if (side != 0) { nor.Neg(); }

      const real_t un = vu * nor;
      const real_t a = alpha * un;
      const real_t b = fabs(alpha * un);
      // note: if |alpha/2|==|beta| then |a|==|b|, i.e. (a==b) or (a==-b)
      //       and therefore two blocks in the element matrix contribution
      //       (from the current quadrature point) are 0

      real_t w = ip.weight * b;
      if (w != 0.0)
      {
         // assemble the element matrix
         for (int i = 0; i < el_ndof; i++)
            for (int j = 0; j < el_ndof; j++)
            {
               elmat(i, j) += w * el_shape(i) * el_shape(j);
            }

         // assemble the constraint matrix
         for (int i = 0; i < tr_ndof; i++)
            for (int j = 0; j < el_ndof; j++)
            {
               elmat(el_ndof+i, j) += w * tr_shape(i) * el_shape(j);
            }
      }

      // assemble the face matrix
      // note: that this term must be non-zero at the boundary for stability
      //       reasons, so the advective part is intentionally dropped here
      //       and must be compensated elsewhere
      w = ip.weight * ((Trans.Elem2No >= 0)?(b-a):(b));
      if (w != 0.0)
      {
         for (int i = 0; i < tr_ndof; i++)
            for (int j = 0; j < tr_ndof; j++)
            {
               elmat(el_ndof+i, el_ndof+j) -= w * tr_shape(i) * tr_shape(j);
            }
      }

      w = ip.weight * (b-a);
      if (w != 0.0)
      {
         // assemble the trace matrix
         for (int i = 0; i < el_ndof; i++)
            for (int j = 0; j < tr_ndof; j++)
            {
               elmat(i, el_ndof+j) -= w * el_shape(i) * tr_shape(j);
            }
      }
   }
}

void HDGConvectionCenteredIntegrator::AssembleHDGFaceVector(
   int type, const FiniteElement &trace_el, const FiniteElement &el,
   FaceElementTransformations &Trans, const Vector &trfun, const Vector &elfun,
   Vector &elvec)
{
   MFEM_VERIFY(trace_el.GetMapType() == FiniteElement::VALUE, "");

   if (Trans.Elem2No < 0) { type &= ~1; }

   const int dim = el.GetDim();
   const int tr_ndof = trace_el.GetDof();
   const int el_ndof = el.GetDof();
   const int ioff = (type & (HDGFaceType::ELEM | HDGFaceType::TRACE))?
                    (el_ndof):(0);

   Vector vu(dim), nor(dim);
   Vector &el_shape = shape1;

   tr_shape.SetSize(tr_ndof);
   el_shape.SetSize(el_ndof);

   int ndofs = 0;
   if (type & (HDGFaceType::ELEM | HDGFaceType::TRACE)) { ndofs += el_ndof; }
   if (type & (HDGFaceType::CONSTR | HDGFaceType::FACE)) { ndofs += tr_ndof; }
   elvec.SetSize(ndofs);
   elvec = 0.0;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int order;
      // Assuming order(u)==order(mesh)
      if (type & 1)
      {
         order = Trans.Elem2->OrderW();
      }
      else
      {
         order = Trans.Elem1->OrderW();
      }
      order += 2*el.GetOrder();
      if (el.Space() == FunctionSpace::Pk)
      {
         order++;
      }
      ir = &IntRules.Get(Trans.GetGeometryType(), order);
   }

   for (int p = 0; p < ir->GetNPoints(); p++)
   {
      const IntegrationPoint &ip = ir->IntPoint(p);

      // Set the integration point in the face and the neighboring elements
      Trans.SetAllIntPoints(&ip);

      // Access the neighboring elements' integration point
      const IntegrationPoint &eip = Trans.GetElement1IntPoint();

      trace_el.CalcShape(ip, tr_shape);
      if (type & 1)
      {
         el.CalcPhysShape(*Trans.Elem2, el_shape);
      }
      else
      {
         el.CalcPhysShape(*Trans.Elem1, el_shape);
      }

      u->Eval(vu, *Trans.Elem1, eip);

      if (dim == 1)
      {
         nor(0) = 2*eip.x - 1.0;
      }
      else
      {
         CalcOrtho(Trans.Jacobian(), nor);
      }

      if (type & 1) { nor.Neg(); }

      const real_t un = vu * nor;
      const real_t a = alpha * un;
      const real_t b = fabs(alpha * un);
      // note: if |alpha/2|==|beta| then |a|==|b|, i.e. (a==b) or (a==-b)
      //       and therefore two blocks in the element matrix contribution
      //       (from the current quadrature point) are 0

      if (type & (HDGFaceType::ELEM | HDGFaceType::CONSTR))
      {
         const real_t v_q = el_shape * elfun;
         const real_t w = ip.weight * b * v_q;

         if (w != 0.0)
         {
            if (type & HDGFaceType::ELEM)
            {
               // assemble the element term
               for (int i = 0; i < el_ndof; i++)
               {
                  elvec(i) += w * el_shape(i);
               }
            }

            if (type & HDGFaceType::CONSTR)
            {
               // assemble the constraint term
               for (int i = 0; i < tr_ndof; i++)
               {
                  elvec(ioff+i) += w * tr_shape(i);
               }
            }
         }
      }

      if (type & (HDGFaceType::TRACE | HDGFaceType::FACE))
      {
         const real_t tr_q = tr_shape * trfun;

         if (type & HDGFaceType::TRACE)
         {
            const real_t w = ip.weight * (b-a) * tr_q;
            if (w != 0.0)
            {
               // assemble the trace term
               for (int i = 0; i < el_ndof; i++)
               {
                  elvec(i) -= w * el_shape(i);
               }
            }
         }

         if (type & HDGFaceType::FACE)
         {
            // assemble the face term
            // note: that this term must be non-zero at the boundary for stability
            //       reasons, so the advective part is intentionally dropped here
            //       and must be compensated elsewhere
            const real_t w = ip.weight * tr_q * ((Trans.Elem2No >= 0)?(b-a):(b));
            if (w != 0.0)
            {
               for (int i = 0; i < tr_ndof; i++)
               {
                  elvec(ioff+i) -= w * tr_shape(i);
               }
            }
         }
      }
   }
}

void HDGConvectionUpwindedIntegrator::AssembleHDGFaceMatrix(
   const FiniteElement &trace_el, const FiniteElement &el1,
   const FiniteElement &el2, FaceElementTransformations &Trans,
   DenseMatrix &elmat)
{
   MFEM_VERIFY(trace_el.GetMapType() == FiniteElement::VALUE, "");

   const int dim = el1.GetDim();
   const int tr_ndof = trace_el.GetDof();
   const int ndof1 = el1.GetDof();
   const int ndof2 = (Trans.Elem2No >= 0)?(el2.GetDof()):(0);
   const int el_ndof = ndof1 + ndof2;

   Vector vu(dim), nor(dim);

   tr_shape.SetSize(tr_ndof);
   shape1.SetSize(ndof1);
   shape2.SetSize(ndof2);

   elmat.SetSize(el_ndof + tr_ndof);
   elmat = 0.0;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int order;
      // Assuming order(u)==order(mesh)
      if (Trans.Elem2No >= 0)
         order = (min(Trans.Elem1->OrderW(), Trans.Elem2->OrderW()) +
                  2*max(el1.GetOrder(), el2.GetOrder()));
      else
      {
         order = Trans.Elem1->OrderW() + 2*el1.GetOrder();
      }
      if (el1.Space() == FunctionSpace::Pk)
      {
         order++;
      }
      ir = &IntRules.Get(Trans.GetGeometryType(), order);
   }

   for (int p = 0; p < ir->GetNPoints(); p++)
   {
      const IntegrationPoint &ip = ir->IntPoint(p);

      // Set the integration point in the face and the neighboring elements
      Trans.SetAllIntPoints(&ip);

      // Access the neighboring elements' integration point
      const IntegrationPoint &eip1 = Trans.GetElement1IntPoint();

      trace_el.CalcShape(ip, tr_shape);
      el1.CalcPhysShape(*Trans.Elem1, shape1);
      if (ndof2)
      {
         el2.CalcPhysShape(*Trans.Elem2, shape2);
      }

      u->Eval(vu, *Trans.Elem1, eip1);

      if (dim == 1)
      {
         nor(0) = 2*eip1.x - 1.0;
      }
      else
      {
         CalcOrtho(Trans.Jacobian(), nor);
      }

      const real_t un = vu * nor;
      const real_t a = 0.5 * alpha * un;
      const real_t b = beta * fabs(un);
      // note: if |alpha/2|==|beta| then |a|==|b|, i.e. (a==b) or (a==-b)
      //       and therefore two blocks in the element matrix contribution
      //       (from the current quadrature point) are 0

      real_t w = ip.weight * (b+a);
      if (w != 0.0)
      {
         // assemble the element matrix (elem1)
         for (int i = 0; i < ndof1; i++)
            for (int j = 0; j < ndof1; j++)
            {
               elmat(i, j) += w * shape1(i) * shape1(j);
            }
         // assemble the constraint matrix (elem1)
         for (int i = 0; i < tr_ndof; i++)
            for (int j = 0; j < ndof1; j++)
            {
               elmat(el_ndof+i, j) += w * tr_shape(i) * shape1(j);
            }
         // assemble the trace matrix (elem2)
         for (int i = 0; i < ndof2; i++)
            for (int j = 0; j < tr_ndof; j++)
            {
               elmat(ndof1+i, el_ndof+j) -= w * shape2(i) * tr_shape(j);
            }
      }

      w = ip.weight * (b-a);
      if (w != 0.0)
      {
         // assemble the element matrix (elem2)
         for (int i = 0; i < ndof2; i++)
            for (int j = 0; j < ndof2; j++)
            {
               elmat(ndof1+i, ndof1+j) += w * shape2(i) * shape2(j);
            }
         // assemble the constraint matrix (elem2)
         for (int i = 0; i < tr_ndof; i++)
            for (int j = 0; j < ndof2; j++)
            {
               elmat(el_ndof+i, ndof1+j) += w * tr_shape(i) * shape2(j);
            }
         // assemble the trace matrix (elem1)
         for (int i = 0; i < ndof1; i++)
            for (int j = 0; j < tr_ndof; j++)
            {
               elmat(i, el_ndof+j) -= w * shape1(i) * tr_shape(j);
            }
      }

      // assemble the face matrix
      // note: that this term must be non-zero at the boundary for stability
      //       reasons, so the advective part is intentionally dropped here
      //       and must be compensated elsewhere
      w = ip.weight * 2.*b;//<-- single face integration
      if (w != 0.0)
      {
         for (int i = 0; i < tr_ndof; i++)
            for (int j = 0; j < tr_ndof; j++)
            {
               elmat(el_ndof+i, el_ndof+j) -= w * tr_shape(i) * tr_shape(j);
            }
      }
   }
}

void HDGConvectionUpwindedIntegrator::AssembleHDGFaceMatrix(
   int side, const FiniteElement &trace_el, const FiniteElement &el,
   FaceElementTransformations &Trans, DenseMatrix &elmat)
{
   MFEM_VERIFY(trace_el.GetMapType() == FiniteElement::VALUE, "");

   const int dim = el.GetDim();
   const int tr_ndof = trace_el.GetDof();
   const int el_ndof = el.GetDof();

   Vector vu(dim), nor(dim);
   Vector &el_shape = shape1;

   tr_shape.SetSize(tr_ndof);
   el_shape.SetSize(el_ndof);

   elmat.SetSize(el_ndof + tr_ndof);
   elmat = 0.0;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int order;
      // Assuming order(u)==order(mesh)
      if (side != 0)
      {
         order = Trans.Elem2->OrderW();
      }
      else
      {
         order = Trans.Elem1->OrderW();
      }
      order += 2*el.GetOrder();
      if (el.Space() == FunctionSpace::Pk)
      {
         order++;
      }
      ir = &IntRules.Get(Trans.GetGeometryType(), order);
   }

   for (int p = 0; p < ir->GetNPoints(); p++)
   {
      const IntegrationPoint &ip = ir->IntPoint(p);

      // Set the integration point in the face and the neighboring elements
      Trans.SetAllIntPoints(&ip);

      // Access the neighboring elements' integration point
      const IntegrationPoint &eip1 = Trans.GetElement1IntPoint();

      trace_el.CalcShape(ip, tr_shape);

      if (side != 0)
      {
         el.CalcPhysShape(*Trans.Elem2, el_shape);
      }
      else
      {
         el.CalcPhysShape(*Trans.Elem1, el_shape);
      }

      u->Eval(vu, *Trans.Elem1, eip1);

      if (dim == 1)
      {
         nor(0) = 2*eip1.x - 1.0;
      }
      else
      {
         CalcOrtho(Trans.Jacobian(), nor);
      }

      if (side != 0) { nor.Neg(); }

      const real_t un = vu * nor;
      const real_t a = 0.5 * alpha * un;
      const real_t b = beta * fabs(un);
      // note: if |alpha/2|==|beta| then |a|==|b|, i.e. (a==b) or (a==-b)
      //       and therefore two blocks in the element matrix contribution
      //       (from the current quadrature point) are 0

      real_t w = ip.weight * (b+a);
      if (w != 0.0)
      {
         // assemble the element matrix
         for (int i = 0; i < el_ndof; i++)
            for (int j = 0; j < el_ndof; j++)
            {
               elmat(i, j) += w * el_shape(i) * el_shape(j);
            }
         // assemble the constraint matrix
         for (int i = 0; i < tr_ndof; i++)
            for (int j = 0; j < el_ndof; j++)
            {
               elmat(el_ndof+i, j) += w * tr_shape(i) * el_shape(j);
            }
      }

      w = ip.weight * (b-a);
      if (w != 0.0)
      {
         // assemble the trace matrix (elem1)
         for (int i = 0; i < el_ndof; i++)
            for (int j = 0; j < tr_ndof; j++)
            {
               elmat(i, el_ndof+j) -= w * el_shape(i) * tr_shape(j);
            }
      }

      // assemble the face matrix
      // note: that this term must be non-zero at the boundary for stability
      //       reasons, so the advective part is intentionally dropped here
      //       and must be compensated elsewhere
      w = ip.weight * ((Trans.Elem2No >= 0)?(b-a):(2.*b));
      if (w != 0.0)
      {
         for (int i = 0; i < tr_ndof; i++)
            for (int j = 0; j < tr_ndof; j++)
            {
               elmat(el_ndof+i, el_ndof+j) -= w * tr_shape(i) * tr_shape(j);
            }
      }
   }
}

void HDGConvectionUpwindedIntegrator::AssembleHDGFaceVector(
   int type, const FiniteElement &trace_el, const FiniteElement &el,
   FaceElementTransformations &Trans, const Vector &trfun, const Vector &elfun,
   Vector &elvec)
{
   MFEM_VERIFY(trace_el.GetMapType() == FiniteElement::VALUE, "");

   if (Trans.Elem2No < 0) { type &= ~1; }

   int tr_ndof, el_ndof;

   dim = el.GetDim();
   tr_ndof = trace_el.GetDof();
   el_ndof = el.GetDof();
   const int ioff = (type & (HDGFaceType::ELEM | HDGFaceType::TRACE))?
                    (el_ndof):(0);
   Vector vu(dim), nor(dim);

   tr_shape.SetSize(tr_ndof);
   Vector &el_shape = shape1;
   el_shape.SetSize(el_ndof);

   int ndofs = 0;
   if (type & (HDGFaceType::ELEM | HDGFaceType::TRACE)) { ndofs += el_ndof; }
   if (type & (HDGFaceType::CONSTR | HDGFaceType::FACE)) { ndofs += tr_ndof; }
   elvec.SetSize(ndofs);
   elvec = 0.0;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int order;
      // Assuming order(u)==order(mesh)
      if (type & 1)
      {
         order = Trans.Elem2->OrderW();
      }
      else
      {
         order = Trans.Elem1->OrderW();
      }
      order += 2*el.GetOrder();
      if (el.Space() == FunctionSpace::Pk)
      {
         order++;
      }
      ir = &IntRules.Get(Trans.GetGeometryType(), order);
   }

   for (int p = 0; p < ir->GetNPoints(); p++)
   {
      const IntegrationPoint &ip = ir->IntPoint(p);

      // Set the integration point in the face and the neighboring elements
      Trans.SetAllIntPoints(&ip);

      // Access the neighboring elements' integration point
      const IntegrationPoint &eip = Trans.GetElement1IntPoint();

      trace_el.CalcShape(ip, tr_shape);
      if (type & 1)
      {
         el.CalcPhysShape(*Trans.Elem2, el_shape);
      }
      else
      {
         el.CalcPhysShape(*Trans.Elem1, el_shape);
      }

      u->Eval(vu, *Trans.Elem1, eip);

      if (dim == 1)
      {
         nor(0) = 2*eip.x - 1.0;
      }
      else
      {
         CalcOrtho(Trans.Jacobian(), nor);
      }

      if (type & 1) { nor.Neg(); }

      const real_t un = vu * nor;
      const real_t a = 0.5 * alpha * un;
      const real_t b = beta * fabs(un);
      // note: if |alpha/2|==|beta| then |a|==|b|, i.e. (a==b) or (a==-b)
      //       and therefore two blocks in the element matrix contribution
      //       (from the current quadrature point) are 0

      if (type & (HDGFaceType::ELEM | HDGFaceType::CONSTR))
      {
         const real_t v_q = el_shape * elfun;
         const real_t w = ip.weight * (b+a) * v_q;

         if (w != 0.0)
         {
            if (type & HDGFaceType::ELEM)
            {
               // assemble the element term
               for (int i = 0; i < el_ndof; i++)
               {
                  elvec(i) += w * el_shape(i);
               }
            }

            if (type & HDGFaceType::CONSTR)
            {
               // assemble the constraint term
               for (int i = 0; i < tr_ndof; i++)
               {
                  elvec(ioff+i) += w * tr_shape(i);
               }
            }
         }
      }

      if (type & (HDGFaceType::TRACE | HDGFaceType::FACE))
      {
         const real_t tr_q = tr_shape * trfun;
         if (type & HDGFaceType::TRACE)
         {
            // assemble the trace term
            const real_t w = ip.weight * (b-a) * tr_q;
            if (w != 0.0)
            {
               for (int i = 0; i < el_ndof; i++)
               {
                  elvec(i) -= w * el_shape(i);
               }
            }
         }

         if (type & HDGFaceType::FACE)
         {
            // assemble the face term
            // note: that this term must be non-zero at the boundary for stability
            //       reasons, so the advective part is intentionally dropped here
            //       and must be compensated elsewhere
            const real_t w = ip.weight * tr_q * (((Trans.Elem2No >= 0)?(b-a):(2.*b)));
            if (w != 0.0)
            {
               for (int i = 0; i < tr_ndof; i++)
               {
                  elvec(ioff+i) -= w * tr_shape(i);
               }
            }
         }
      }
   }
}

void HDGDiffusionIntegrator::AssembleFaceMatrix(
   const FiniteElement &el1, const FiniteElement &el2,
   FaceElementTransformations &Trans, DenseMatrix &elmat)
{
   int dim, ndof1, ndof2, ndofs;

   dim = el1.GetDim();
   ndof1 = el1.GetDof();

   nor.SetSize(dim);
   nh.SetSize(dim);
   ni.SetSize(dim);
   if (MQ)
   {
      mq.SetSize(dim);
   }

   shape1.SetSize(ndof1);
   if (Trans.Elem2No >= 0)
   {
      ndof2 = el2.GetDof();
      shape2.SetSize(ndof2);
   }
   else
   {
      ndof2 = 0;
   }

   ndofs = ndof1 + ndof2;
   elmat.SetSize(ndofs);
   elmat = 0.0;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      // a simple choice for the integration order; is this OK?
      int order;
      if (ndof2)
      {
         order = 2*max(el1.GetOrder(), el2.GetOrder());
      }
      else
      {
         order = 2*el1.GetOrder();
      }
      ir = &IntRules.Get(Trans.GetGeometryType(), order);
   }

   // assemble: alpha < {h^{-1} Q} [u],[v] >
   for (int p = 0; p < ir->GetNPoints(); p++)
   {
      const IntegrationPoint &ip = ir->IntPoint(p);

      // Set the integration point in the face and the neighboring elements
      Trans.SetAllIntPoints(&ip);

      // Access the neighboring elements' integration points
      // Note: eip2 will only contain valid data if Elem2 exists
      const IntegrationPoint &eip1 = Trans.GetElement1IntPoint();
      const IntegrationPoint &eip2 = Trans.GetElement2IntPoint();

      if (dim == 1)
      {
         nor(0) = 2*eip1.x - 1.0;
      }
      else
      {
         CalcOrtho(Trans.Jacobian(), nor);
      }

      el1.CalcPhysShape(*Trans.Elem1, shape1);
      {
         real_t wn = ip.weight/Trans.Elem1->Weight();
         if (ndof2)
         {
            wn /= 2;
         }
         if (!MQ)
         {
            if (Q)
            {
               wn *= Q->Eval(*Trans.Elem1, eip1);
            }
            ni.Set(wn, nor);
         }
         else
         {
            nh.Set(wn, nor);
            MQ->Eval(mq, *Trans.Elem1, eip1);
            mq.MultTranspose(nh, ni);
         }
      }
      real_t wq = ni * nor;
      // Note: in the jump term, we use 1/h1 = |nor|/det(J1) which is
      // independent of Loc1 and always gives the size of element 1 in
      // direction perpendicular to the face. Indeed, for linear transformation
      //     |nor|=measure(face)/measure(ref. face),
      //   det(J1)=measure(element)/measure(ref. element),
      // and the ratios measure(ref. element)/measure(ref. face) are
      // compatible for all element/face pairs.
      // For example: meas(ref. tetrahedron)/meas(ref. triangle) = 1/3, and
      // for any tetrahedron vol(tet)=(1/3)*height*area(base).
      // For interior faces: q_e/h_e=(q1/h1+q2/h2)/2.

      if (ndof2)
      {
         el2.CalcPhysShape(*Trans.Elem2, shape2);
         real_t wn = ip.weight/2/Trans.Elem2->Weight();
         if (!MQ)
         {
            if (Q)
            {
               wn *= Q->Eval(*Trans.Elem2, eip2);
            }
            ni.Set(wn, nor);
         }
         else
         {
            nh.Set(wn, nor);
            MQ->Eval(mq, *Trans.Elem2, eip2);
            mq.MultTranspose(nh, ni);
         }
         wq += ni * nor;
      }

      wq *= 0.5 * beta;

      // only assemble the lower triangular part
      for (int i = 0; i < ndof1; i++)
      {
         const real_t wsi = wq*shape1(i);
         for (int j = 0; j <= i; j++)
         {
            elmat(i, j) += wsi * shape1(j);
         }
      }
      if (ndof2)
      {
         for (int i = 0; i < ndof2; i++)
         {
            const int i2 = ndof1 + i;
            const real_t wsi = wq*shape2(i);
            for (int j = 0; j < ndof1; j++)
            {
               elmat(i2, j) -= wsi * shape1(j);
            }
            for (int j = 0; j <= i; j++)
            {
               elmat(i2, ndof1 + j) += wsi * shape2(j);
            }
         }
      }
   }

   // complete the upper triangular part
   for (int i = 0; i < ndofs; i++)
      for (int j = 0; j < i; j++)
      {
         elmat(j,i) = elmat(i,j);
      }
}

void HDGDiffusionIntegrator::AssembleHDGFaceMatrix(
   const FiniteElement &trace_el, const FiniteElement &el1,
   const FiniteElement &el2, FaceElementTransformations &Trans,
   DenseMatrix &elmat)
{
   MFEM_VERIFY(trace_el.GetMapType() == FiniteElement::VALUE, "");

   int dim, tr_ndof, ndof1, ndof2, el_ndof;
   real_t w, wq = 0.0;
   real_t un, a, b;

   dim = el1.GetDim();
   tr_ndof = trace_el.GetDof();
   ndof1 = el1.GetDof();

   vu.SetSize(dim);
   nor.SetSize(dim);
   nh.SetSize(dim);
   ni.SetSize(dim);
   if (MQ)
   {
      mq.SetSize(dim);
   }

   tr_shape.SetSize(tr_ndof);
   shape1.SetSize(ndof1);
   if (Trans.Elem2No >= 0)
   {
      ndof2 = el2.GetDof();
      shape2.SetSize(ndof2);
   }
   else
   {
      ndof2 = 0;
   }
   el_ndof = ndof1 + ndof2;

   elmat.SetSize(el_ndof + tr_ndof);
   elmat = 0.0;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      // a simple choice for the integration order; is this OK?
      int order;
      if (ndof2)
      {
         order = 2*max(el1.GetOrder(), el2.GetOrder());
      }
      else
      {
         order = 2*el1.GetOrder();
      }
      ir = &IntRules.Get(Trans.GetGeometryType(), order);
   }

   // assemble: alpha < {h^{-1} Q} [u],[v] >
   for (int p = 0; p < ir->GetNPoints(); p++)
   {
      const IntegrationPoint &ip = ir->IntPoint(p);

      // Set the integration point in the face and the neighboring elements
      Trans.SetAllIntPoints(&ip);

      // Access the neighboring elements' integration points
      // Note: eip2 will only contain valid data if Elem2 exists
      const IntegrationPoint &eip1 = Trans.GetElement1IntPoint();
      const IntegrationPoint &eip2 = Trans.GetElement2IntPoint();

      if (dim == 1)
      {
         nor(0) = 2*eip1.x - 1.0;
      }
      else
      {
         CalcOrtho(Trans.Jacobian(), nor);
      }

      trace_el.CalcShape(ip, tr_shape);

      if (v)
      {
         v->Eval(vu, *Trans.Elem1, eip1);
         un = vu * nor;
      }
      else
      {
         un = 0.0;
      }

      el1.CalcPhysShape(*Trans.Elem1, shape1);
      w = ip.weight/Trans.Elem1->Weight();
      if (ndof2)
      {
         w /= 2;
      }
      if (!MQ)
      {
         if (Q)
         {
            w *= Q->Eval(*Trans.Elem1, eip1);
         }
         ni.Set(w, nor);
      }
      else
      {
         nh.Set(w, nor);
         MQ->Eval(mq, *Trans.Elem1, eip1);
         mq.MultTranspose(nh, ni);
      }
      wq = ni * nor;
      // Note: in the jump term, we use 1/h1 = |nor|/det(J1) which is
      // independent of Loc1 and always gives the size of element 1 in
      // direction perpendicular to the face. Indeed, for linear transformation
      //     |nor|=measure(face)/measure(ref. face),
      //   det(J1)=measure(element)/measure(ref. element),
      // and the ratios measure(ref. element)/measure(ref. face) are
      // compatible for all element/face pairs.
      // For example: meas(ref. tetrahedron)/meas(ref. triangle) = 1/3, and
      // for any tetrahedron vol(tet)=(1/3)*height*area(base).
      // For interior faces: q_e/h_e=(q1/h1+q2/h2)/2.

      if (ndof2)
      {
         el2.CalcPhysShape(*Trans.Elem2, shape2);
         w = ip.weight/2/Trans.Elem2->Weight();
         if (!MQ)
         {
            if (Q)
            {
               w *= Q->Eval(*Trans.Elem2, eip2);
            }
            ni.Set(w, nor);
         }
         else
         {
            nh.Set(w, nor);
            MQ->Eval(mq, *Trans.Elem2, eip2);
            mq.MultTranspose(nh, ni);
         }
         wq += ni * nor;
      }

      if (un != 0.)
      {
         un /= fabs(un);
         a = 0.5 * alpha * un;
         b = beta * fabs(un);
      }
      else
      {
         a = 0.0;
         b = beta;
      }

      w = wq * (b+a);
      if (w != 0.0)
      {
         // assemble the element matrix
         // (only the lower triangular part)
         for (int i = 0; i < ndof1; i++)
         {
            const real_t wsi = w*shape1(i);
            for (int j = 0; j <= i; j++)
            {
               elmat(i, j) += wsi * shape1(j);
            }
         }

         // assemble the constraint matrix
         for (int i = 0; i < ndof1; i++)
         {
            const real_t wsi = w*shape1(i);
            for (int j = 0; j < tr_ndof; j++)
            {
               elmat(i, el_ndof+j) -= wsi * tr_shape(j);
            }
         }
      }

      w = wq * (b-a);
      if (w != 0.0)
      {
         // assemble the element matrix
         // (only the lower triangular part)
         for (int i = 0; i < ndof2; i++)
         {
            const real_t wsi = w*shape2(i);
            for (int j = 0; j <= i; j++)
            {
               elmat(ndof1+i, ndof1+j) += wsi * shape2(j);
            }
         }

         // assemble the constraint matrix
         for (int i = 0; i < ndof2; i++)
         {
            const real_t wsi = w*shape2(i);
            for (int j = 0; j < tr_ndof; j++)
            {
               elmat(i+ndof1, el_ndof+j) -= wsi * tr_shape(j);
            }
         }
      }

      w = wq * ((ndof2)?(2.*b):(b+a));//<-- single face integration
      if (w != 0.0)
      {
         // assemble the trace matrix
         for (int i = 0; i < tr_ndof; i++)
         {
            const real_t wsi = w*tr_shape(i);
            for (int j = 0; j <= i; j++)
            {
               elmat(el_ndof+i, el_ndof+j) -= wsi * tr_shape(j);
            }
         }
      }
   }

   // complete the element matrices
   // (the upper triangular part)
   for (int i = 0; i < ndof1; i++)
      for (int j = 0; j < i; j++)
      {
         elmat(j, i) = elmat(i, j);
      }

   for (int i = 0; i < ndof2; i++)
      for (int j = 0; j < i; j++)
      {
         elmat(ndof1+j, ndof1+i) = elmat(ndof1+i, ndof1+j);
      }

   // complete the constraint matrix
   for (int i = 0; i < el_ndof; i++)
      for (int j = 0; j < tr_ndof; j++)
      {
         elmat(el_ndof+j, i) = -elmat(i, el_ndof+j);
      }

   // complete the trace matrix
   for (int i = 0; i < tr_ndof; i++)
      for (int j = 0; j < i; j++)
      {
         elmat(el_ndof+j, el_ndof+i) = elmat(el_ndof+i, el_ndof+j);
      }
}

void HDGDiffusionIntegrator::AssembleHDGFaceVector(
   int type, const FiniteElement &trace_el, const FiniteElement &el,
   FaceElementTransformations &Trans, const Vector &trfun, const Vector &elfun,
   Vector &elvec)
{
   MFEM_VERIFY(trace_el.GetMapType() == FiniteElement::VALUE, "");

   if (Trans.Elem2No < 0) { type &= ~1; }

   int dim, tr_ndof, el_ndof;

   dim = el.GetDim();
   tr_ndof = trace_el.GetDof();
   el_ndof = el.GetDof();
   const int ioff = (type & (HDGFaceType::ELEM | HDGFaceType::TRACE))?
                    (el_ndof):(0);

   vu.SetSize(dim);
   nor.SetSize(dim);
   nh.SetSize(dim);
   ni.SetSize(dim);
   if (MQ)
   {
      mq.SetSize(dim);
   }

   tr_shape.SetSize(tr_ndof);
   Vector &el_shape = shape1;
   el_shape.SetSize(el_ndof);

   int ndofs = 0;
   if (type & (HDGFaceType::ELEM | HDGFaceType::TRACE)) { ndofs += el_ndof; }
   if (type & (HDGFaceType::CONSTR | HDGFaceType::FACE)) { ndofs += tr_ndof; }
   elvec.SetSize(ndofs);
   elvec = 0.0;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      // a simple choice for the integration order; is this OK?
      int order;
      order = 2*el.GetOrder();
      ir = &IntRules.Get(Trans.GetGeometryType(), order);
   }

   // assemble: alpha < {h^{-1} Q} [u],[v] >
   for (int p = 0; p < ir->GetNPoints(); p++)
   {
      const IntegrationPoint &ip = ir->IntPoint(p);

      // Set the integration point in the face and the neighboring elements
      Trans.SetAllIntPoints(&ip);

      // Access the neighboring elements' integration points
      // Note: eip2 will only contain valid data if Elem2 exists
      const IntegrationPoint &eip1 = Trans.GetElement1IntPoint();
      const IntegrationPoint &eip2 = Trans.GetElement2IntPoint();

      if (dim == 1)
      {
         nor(0) = 2*eip1.x - 1.0;
      }
      else
      {
         CalcOrtho(Trans.Jacobian(), nor);
      }

      if (type & 1) { nor.Neg(); }

      trace_el.CalcShape(ip, tr_shape);

      real_t un;
      if (v)
      {
         v->Eval(vu, *Trans.Elem1, eip1);
         un = vu * nor;
      }
      else
      {
         un = 0.0;
      }

      if (type & 1)
      {
         el.CalcPhysShape(*Trans.Elem2, el_shape);
      }
      else
      {
         el.CalcPhysShape(*Trans.Elem1, el_shape);
      }

      {
         real_t wn = ip.weight/Trans.Elem1->Weight();
         if (Trans.Elem2No >= 0)
         {
            wn /= 2;
         }
         if (!MQ)
         {
            if (Q)
            {
               wn *= Q->Eval(*Trans.Elem1, eip1);
            }
            ni.Set(wn, nor);
         }
         else
         {
            nh.Set(wn, nor);
            MQ->Eval(mq, *Trans.Elem1, eip1);
            mq.MultTranspose(nh, ni);
         }
      }
      real_t wq = ni * nor;
      // Note: in the jump term, we use 1/h1 = |nor|/det(J1) which is
      // independent of Loc1 and always gives the size of element 1 in
      // direction perpendicular to the face. Indeed, for linear transformation
      //     |nor|=measure(face)/measure(ref. face),
      //   det(J1)=measure(element)/measure(ref. element),
      // and the ratios measure(ref. element)/measure(ref. face) are
      // compatible for all element/face pairs.
      // For example: meas(ref. tetrahedron)/meas(ref. triangle) = 1/3, and
      // for any tetrahedron vol(tet)=(1/3)*height*area(base).
      // For interior faces: q_e/h_e=(q1/h1+q2/h2)/2.

      if (Trans.Elem2No >= 0)
      {
         real_t wn = ip.weight/2/Trans.Elem2->Weight();
         if (!MQ)
         {
            if (Q)
            {
               wn *= Q->Eval(*Trans.Elem2, eip2);
            }
            ni.Set(wn, nor);
         }
         else
         {
            nh.Set(wn, nor);
            MQ->Eval(mq, *Trans.Elem2, eip2);
            mq.MultTranspose(nh, ni);
         }
         wq += ni * nor;
      }

      real_t a, b;
      if (un != 0.)
      {
         un /= fabs(un);
         a = 0.5 * alpha * un;
         b = beta * fabs(un);
      }
      else
      {
         a = 0.0;
         b = beta;
      }

      const real_t w = wq * (b+a);
      if (w == 0.) { continue; }

      if (type & (HDGFaceType::ELEM | HDGFaceType::CONSTR))
      {
         const real_t v_q = el_shape * elfun;
         const real_t wv = w * v_q;
         if (wv != 0.0)
         {
            if (type & HDGFaceType::ELEM)
            {
               // assemble the element term
               for (int i = 0; i < el_ndof; i++)
               {
                  elvec(i) += wv * el_shape(i);
               }
            }

            if (type & HDGFaceType::CONSTR)
            {
               // assemble the constraint term
               for (int i = 0; i < tr_ndof; i++)
               {
                  elvec(ioff+i) += wv * tr_shape(i);
               }
            }
         }
      }

      if (type & (HDGFaceType::TRACE | HDGFaceType::FACE))
      {
         const real_t tr_q = tr_shape * trfun;
         const real_t wt = w * tr_q;
         if (wt != 0.0)
         {
            if (type & HDGFaceType::TRACE)
            {
               // assemble the trace term
               for (int i = 0; i < el_ndof; i++)
               {
                  elvec(i) -= wt * el_shape(i);
               }
            }

            if (type & HDGFaceType::FACE)
            {
               // assemble the face term
               for (int i = 0; i < tr_ndof; i++)
               {
                  elvec(ioff+i) -= wt * tr_shape(i);
               }
            }
         }
      }
   }
}

real_t HDGDiffusionIntegrator::ComputeHDGFaceEnergy(
   int side, const FiniteElement &trace_el, const FiniteElement &el,
   FaceElementTransformations &Trans, const Vector &trfun, const Vector &elfun,
   Vector *d_energy)
{
   MFEM_VERIFY(trace_el.GetMapType() == FiniteElement::VALUE, "");

   if (Trans.Elem2No < 0) { side = 0; }

   int dim, tr_ndof, el_ndof;

   dim = el.GetDim();
   tr_ndof = trace_el.GetDof();
   el_ndof = el.GetDof();

   vu.SetSize(dim);
   nor.SetSize(dim);
   nh.SetSize(dim);
   ni.SetSize(dim);
   if (MQ)
   {
      mq.SetSize(dim);
   }

   tr_shape.SetSize(tr_ndof);
   Vector &el_shape = shape1;
   el_shape.SetSize(el_ndof);

   if (d_energy)
   {
      d_energy->SetSize(dim);
      *d_energy = 0.;
   }
   real_t energy = 0.;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      // a simple choice for the integration order; is this OK?
      int order;
      order = 2*el.GetOrder();
      ir = &IntRules.Get(Trans.GetGeometryType(), order);
   }

   // assemble: alpha < {h^{-1} Q} [u],[v] >
   for (int p = 0; p < ir->GetNPoints(); p++)
   {
      const IntegrationPoint &ip = ir->IntPoint(p);

      // Set the integration point in the face and the neighboring elements
      Trans.SetAllIntPoints(&ip);

      // Access the neighboring elements' integration points
      // Note: eip2 will only contain valid data if Elem2 exists
      const IntegrationPoint &eip1 = Trans.GetElement1IntPoint();
      const IntegrationPoint &eip2 = Trans.GetElement2IntPoint();

      if (dim == 1)
      {
         nor(0) = 2*eip1.x - 1.0;
      }
      else
      {
         CalcOrtho(Trans.Jacobian(), nor);
      }

      if (side != 0) { nor.Neg(); }

      trace_el.CalcShape(ip, tr_shape);

      real_t un;
      if (v)
      {
         v->Eval(vu, *Trans.Elem1, eip1);
         un = vu * nor;
      }
      else
      {
         un = 0.0;
      }

      if (side != 0)
      {
         el.CalcPhysShape(*Trans.Elem2, el_shape);
      }
      else
      {
         el.CalcPhysShape(*Trans.Elem1, el_shape);
      }

      {
         real_t wn = ip.weight/Trans.Elem1->Weight();
         if (Trans.Elem2No >= 0)
         {
            wn /= 2;
         }
         if (!MQ)
         {
            if (Q)
            {
               wn *= Q->Eval(*Trans.Elem1, eip1);
            }
            ni.Set(wn, nor);
         }
         else
         {
            nh.Set(wn, nor);
            MQ->Eval(mq, *Trans.Elem1, eip1);
            mq.MultTranspose(nh, ni);
         }
      }
      // Note: in the jump term, we use 1/h1 = |nor|/det(J1) which is
      // independent of Loc1 and always gives the size of element 1 in
      // direction perpendicular to the face. Indeed, for linear transformation
      //     |nor|=measure(face)/measure(ref. face),
      //   det(J1)=measure(element)/measure(ref. element),
      // and the ratios measure(ref. element)/measure(ref. face) are
      // compatible for all element/face pairs.
      // For example: meas(ref. tetrahedron)/meas(ref. triangle) = 1/3, and
      // for any tetrahedron vol(tet)=(1/3)*height*area(base).
      // For interior faces: q_e/h_e=(q1/h1+q2/h2)/2.

      if (Trans.Elem2No >= 0)
      {
         real_t wn = ip.weight/2/Trans.Elem2->Weight();
         if (!MQ)
         {
            if (Q)
            {
               wn *= Q->Eval(*Trans.Elem2, eip2);
            }
            ni.Add(wn, nor);
         }
         else
         {
            nh.Set(wn, nor);
            MQ->Eval(mq, *Trans.Elem2, eip2);
            mq.AddMultTranspose(nh, ni);
         }
      }

      real_t a, b;
      if (un != 0.)
      {
         un /= fabs(un);
         a = 0.5 * alpha * un;
         b = beta * fabs(un);
      }
      else
      {
         a = 0.0;
         b = beta;
      }

      const real_t v_q = el_shape * elfun;
      const real_t tr_q = tr_shape * trfun;
      const real_t d_q = v_q - tr_q;

      const real_t w = d_q*d_q * fabs(b+a);
      if (w == 0.) { continue; }

      for (int d = 0; d < dim; d++)
      {
         const real_t energy_d = w * ni(d) * nor(d);
         if (d_energy) { (*d_energy)(d) += energy_d; }
         energy += energy_d;
      }
   }

   return energy;
}
}
