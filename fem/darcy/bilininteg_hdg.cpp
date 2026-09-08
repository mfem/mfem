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
#include "../../general/forall.hpp"
#include "../../linalg/dtensor.hpp"

using std::min;
using std::max;

/** @file
    Quadrature for the HDG face terms, and why the trace element's order is in
    every rule below.

    Each routine here that takes a @a trace_el assembles three kinds of block on
    one face: element-element, element-trace, and trace-trace. The last is a
    mass-like matrix in the trace basis alone -- it is where the stabilization
    `<tau*uhat, mu>` lands -- so it is of degree `2*p_trace`, and a rule chosen
    from the element orders alone under-integrates it whenever the trace outruns
    its elements.

    That is not a lost digit, it is a rank failure. A Gauss rule of degree
    `2*p_el` on a face carries `p_el + 1` points, and a Gram matrix built from
    `n` points has rank at most `n`; the trace-trace block is
    `(p_trace + 1) x (p_trace + 1)`. At `p_trace = p_el + 1` it is therefore
    rank-deficient by exactly one per face and the reduced trace system is
    singular. Measured before the fix: `convdiff -p 1 -dg -hb -o 2` with an
    order-3 trace diverges (GMRES residual 1e21, relative errors of 1e4);
    raising the rule to degree `2*p_trace` -- and nothing else -- converges it.
    The threshold is exactly `2*p_trace`, degree `2*p_el` still fails.

    So the max below spans the trace element too, which is what
    `fem/hyperbolic.cpp`'s HDG routines and `NormalTraceJumpIntegrator` in
    `fem/bilininteg.cpp` already do. Where `p_trace <= p_el` the expression is
    arithmetically the old one, so no existing configuration moves; the one
    committed case that does is `--order 0 --trace-H1`, whose trace collection is
    `max(order, 1)` and so outruns its elements (errors move by 4-5% and no
    regression reference covers it).

    This matters beyond tidiness: a trace order that may exceed an element order
    is exactly what `p`-adaptivity's max rule needs on a face between elements of
    different degree, and this was what blocked it.

    `AssembleFaceMatrix` below takes no trace element and is left alone. */

namespace mfem
{


const IntegrationRule &HDGConvectionCenteredIntegrator::GetHDGFaceIntRule(
   const FiniteElement &trace_el, const FiniteElement &el1,
   const FiniteElement &el2, FaceElementTransformations &Trans) const
{
   int order;
   // Assuming order(u)==order(mesh)
   if (Trans.Elem2No >= 0)
      order = (min(Trans.Elem1->OrderW(), Trans.Elem2->OrderW()) +
               2*max(max(el1.GetOrder(), el2.GetOrder()), trace_el.GetOrder()));
   else
   {
      order = Trans.Elem1->OrderW() + 2*max(el1.GetOrder(), trace_el.GetOrder());
   }
   if (el1.Space() == FunctionSpace::Pk)
   {
      order++;
   }
   return IntRules.Get(Trans.GetGeometryType(), order);
}

const IntegrationRule &HDGConvectionUpwindedIntegrator::GetHDGFaceIntRule(
   const FiniteElement &trace_el, const FiniteElement &el1,
   const FiniteElement &el2, FaceElementTransformations &Trans) const
{
   // The same rule as the centred form; they differ in the weights, not in
   // where they are sampled.
   int order;
   if (Trans.Elem2No >= 0)
      order = (min(Trans.Elem1->OrderW(), Trans.Elem2->OrderW()) +
               2*max(max(el1.GetOrder(), el2.GetOrder()), trace_el.GetOrder()));
   else
   {
      order = Trans.Elem1->OrderW() + 2*max(el1.GetOrder(), trace_el.GetOrder());
   }
   if (el1.Space() == FunctionSpace::Pk)
   {
      order++;
   }
   return IntRules.Get(Trans.GetGeometryType(), order);
}

const IntegrationRule &HDGDiffusionIntegrator::GetHDGFaceIntRule(
   const FiniteElement &trace_el, const FiniteElement &el1,
   const FiniteElement &el2, FaceElementTransformations &Trans) const
{
   // Degree 2*max(element, trace): see the note at the top of this file for
   // why the trace element has to be in the max.
   const int order = (Trans.Elem2No >= 0)
                     ? 2*max(max(el1.GetOrder(), el2.GetOrder()),
                             trace_el.GetOrder())
                     : 2*max(el1.GetOrder(), trace_el.GetOrder());
   return IntRules.Get(Trans.GetGeometryType(), order);
}

void HDGConvectionCenteredIntegrator::AssembleHDGFaceMatrix(
   const FiniteElement &trace_el, const FiniteElement &el1,
   const FiniteElement &el2, FaceElementTransformations &Trans,
   DenseMatrix &elmat)
{
#ifdef MFEM_THREAD_SAFE
   Vector tr_shape, shape1, shape2;
#endif
   MFEM_VERIFY(trace_el.GetMapType() == FiniteElement::VALUE, "");

   const int el_dim = el1.GetDim();
   const int tr_ndof = trace_el.GetDof();
   const int ndof1 = el1.GetDof();
   const int ndof2 = (Trans.Elem2No >= 0)?(el2.GetDof()):(0);
   const int el_ndof = ndof1 + ndof2;

   Vector vu(el_dim), nor(el_dim);

   tr_shape.SetSize(tr_ndof);
   shape1.SetSize(ndof1);
   shape2.SetSize(ndof2);

   elmat.SetSize(el_ndof + tr_ndof);
   elmat = 0.0;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      ir = &GetHDGFaceIntRule(trace_el, el1, el2, Trans);
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

      if (el_dim == 1)
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
#ifdef MFEM_THREAD_SAFE
   Vector tr_shape, shape1;
#endif
   MFEM_VERIFY(trace_el.GetMapType() == FiniteElement::VALUE, "");

   const int el_dim = el.GetDim();
   const int tr_ndof = trace_el.GetDof();
   const int el_ndof = el.GetDof();

   Vector vu(el_dim), nor(el_dim);
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
      order += 2*max(el.GetOrder(), trace_el.GetOrder());
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

      if (el_dim == 1)
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
#ifdef MFEM_THREAD_SAFE
   Vector tr_shape, shape1;
#endif
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
      order += 2*max(el.GetOrder(), trace_el.GetOrder());
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
#ifdef MFEM_THREAD_SAFE
   Vector tr_shape, shape1, shape2;
#endif
   MFEM_VERIFY(trace_el.GetMapType() == FiniteElement::VALUE, "");

   const int el_dim = el1.GetDim();
   const int tr_ndof = trace_el.GetDof();
   const int ndof1 = el1.GetDof();
   const int ndof2 = (Trans.Elem2No >= 0)?(el2.GetDof()):(0);
   const int el_ndof = ndof1 + ndof2;

   Vector vu(el_dim), nor(el_dim);

   tr_shape.SetSize(tr_ndof);
   shape1.SetSize(ndof1);
   shape2.SetSize(ndof2);

   elmat.SetSize(el_ndof + tr_ndof);
   elmat = 0.0;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      ir = &GetHDGFaceIntRule(trace_el, el1, el2, Trans);
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

      if (el_dim == 1)
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
#ifdef MFEM_THREAD_SAFE
   Vector tr_shape, shape1;
#endif
   MFEM_VERIFY(trace_el.GetMapType() == FiniteElement::VALUE, "");

   const int el_dim = el.GetDim();
   const int tr_ndof = trace_el.GetDof();
   const int el_ndof = el.GetDof();

   Vector vu(el_dim), nor(el_dim);
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
      order += 2*max(el.GetOrder(), trace_el.GetOrder());
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

      if (el_dim == 1)
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
#ifdef MFEM_THREAD_SAFE
   Vector tr_shape, shape1;
#endif
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
      order += 2*max(el.GetOrder(), trace_el.GetOrder());
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
#ifdef MFEM_THREAD_SAFE
   Vector shape1, shape2, nor, nh, ni;
   DenseMatrix mq;
#endif
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
      // No trace element on this route, so the element orders are the whole rule.
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
#ifdef MFEM_THREAD_SAFE
   Vector tr_shape, shape1, shape2, vu, nor, nh, ni;
   DenseMatrix mq;
#endif
   MFEM_VERIFY(trace_el.GetMapType() == FiniteElement::VALUE, "");
   MFEM_VERIFY(!stab || stab->IsConstant(),
               "A state dependent stabilization makes the face term nonlinear; "
               "assemble it through AssembleHDGFaceVector/Grad instead of as a "
               "bilinear form.");

   const int dim = el1.GetDim();
   const int tr_ndof = trace_el.GetDof();
   const int ndof1 = el1.GetDof();
   const int ndof2 = (Trans.Elem2No >= 0)?(el2.GetDof()):(0);

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
   if (ndof2)
   {
      shape2.SetSize(ndof2);
   }
   const int el_ndof = ndof1 + ndof2;

   elmat.SetSize(el_ndof + tr_ndof);
   elmat = 0.0;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      ir = &GetHDGFaceIntRule(trace_el, el1, el2, Trans);
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

      // evaluate the normal velocity
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

      // calculate the stabilization coefficient on side 1
      {
         el1.CalcPhysShape(*Trans.Elem1, shape1);
         real_t wn = ip.weight/Trans.Elem1->Weight();
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
      const real_t wq1 = ni * nor;
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

      // calculate the stabilization coefficient on side 2
      real_t wq2;
      if (ndof2)
      {
         el2.CalcPhysShape(*Trans.Elem2, shape2);
         real_t wn = ip.weight/Trans.Elem2->Weight();
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
         wq2 = ni * nor;
      }

      const real_t un_raw = un;
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
      const real_t face_w = ip.weight * nor.Norml2();

      // assemble side 1
      real_t w = StabValue(wq1, b+a, un_raw, face_w, 0., 0., *Trans.Elem1);
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

      // assemble side 2
      if (ndof2)
      {
         w = StabValue(wq2, b-a, un_raw, face_w, 0., 0., *Trans.Elem2);
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
      }

      w = StabValue(wq1, b+a, un_raw, face_w, 0., 0., *Trans.Elem1);
      //<-- single face integration
      if (ndof2)
      {
         w += StabValue(wq2, b-a, un_raw, face_w, 0., 0., *Trans.Elem2);
      }
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

void HDGDiffusionIntegrator::AssembleHDGFaceMatrix(
   int side, const FiniteElement &trace_el, const FiniteElement &el,
   FaceElementTransformations &Trans, DenseMatrix &elmat)
{
#ifdef MFEM_THREAD_SAFE
   Vector tr_shape, shape1, vu, nor, nh, ni;
   DenseMatrix mq;
#endif
   MFEM_VERIFY(trace_el.GetMapType() == FiniteElement::VALUE, "");
   MFEM_VERIFY(!stab || stab->IsConstant(),
               "A state dependent stabilization makes the face term nonlinear; "
               "assemble it through AssembleHDGFaceVector/Grad instead of as a "
               "bilinear form.");

   const int dim = el.GetDim();
   const int tr_ndof = trace_el.GetDof();
   const int el_ndof = el.GetDof();

   vu.SetSize(dim);
   nor.SetSize(dim);
   nh.SetSize(dim);
   ni.SetSize(dim);
   if (MQ)
   {
      mq.SetSize(dim);
   }

   Vector &el_shape = shape1;
   tr_shape.SetSize(tr_ndof);
   el_shape.SetSize(el_ndof);

   elmat.SetSize(el_ndof + tr_ndof);
   elmat = 0.0;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      // Degree 2*max(element, trace): see the note at the top of this file
      // for why the trace element has to be in the max.
      int order = 2*max(el.GetOrder(), trace_el.GetOrder());
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

      // evaluate the normal velocity
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

      // calculate the stabilization coefficient
      ElementTransformation *ElTr = (side != 0)?(Trans.Elem2):(Trans.Elem1);
      const IntegrationPoint &eip = (side != 0)?(eip2):(eip1);

      el.CalcPhysShape(*ElTr, el_shape);
      real_t wn = ip.weight/ElTr->Weight();
      if (!MQ)
      {
         if (Q)
         {
            wn *= Q->Eval(*ElTr, eip);
         }
         ni.Set(wn, nor);
      }
      else
      {
         nh.Set(wn, nor);
         MQ->Eval(mq, *ElTr, eip);
         mq.MultTranspose(nh, ni);
      }
      const real_t wq = ni * nor;
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

      const real_t un_raw = un;
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

      const real_t face_w = ip.weight * nor.Norml2();
      real_t w = StabValue(wq, b+a, un_raw, face_w, 0., 0., *ElTr);
      if (w != 0.0)
      {
         // assemble the element matrix
         // (only the lower triangular part)
         for (int i = 0; i < el_ndof; i++)
         {
            const real_t wsi = w*el_shape(i);
            for (int j = 0; j <= i; j++)
            {
               elmat(i, j) += wsi * el_shape(j);
            }
         }

         // assemble the constraint matrix
         for (int i = 0; i < el_ndof; i++)
         {
            const real_t wsi = w*el_shape(i);
            for (int j = 0; j < tr_ndof; j++)
            {
               elmat(i, el_ndof+j) -= wsi * tr_shape(j);
            }
         }

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

   // complete the element matrix
   // (the upper triangular part)
   for (int i = 0; i < el_ndof; i++)
      for (int j = 0; j < i; j++)
      {
         elmat(j, i) = elmat(i, j);
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
#ifdef MFEM_THREAD_SAFE
   Vector tr_shape, shape1, vu, nor, nh, ni;
   DenseMatrix mq;
#endif
   MFEM_VERIFY(trace_el.GetMapType() == FiniteElement::VALUE, "");

   if (Trans.Elem2No < 0) { type &= ~1; }

   const int dim = el.GetDim();
   const int tr_ndof = trace_el.GetDof();
   const int el_ndof = el.GetDof();
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
      // Degree 2*max(element, trace): see the note at the top of this file
      // for why the trace element has to be in the max.
      int order;
      order = 2*max(el.GetOrder(), trace_el.GetOrder());
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

      // evaluate the normal velocity
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

      // calculate the stabilization coefficient
      ElementTransformation *ElTr = (type & 1)?(Trans.Elem2):(Trans.Elem1);
      const IntegrationPoint &eip = (type & 1)?(eip2):(eip1);

      el.CalcPhysShape(*ElTr, el_shape);
      real_t wn = ip.weight/ElTr->Weight();
      if (!MQ)
      {
         if (Q)
         {
            wn *= Q->Eval(*ElTr, eip);
         }
         ni.Set(wn, nor);
      }
      else
      {
         nh.Set(wn, nor);
         MQ->Eval(mq, *ElTr, eip);
         mq.MultTranspose(nh, ni);
      }
      const real_t wq = ni * nor;
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

      const real_t un_raw = un;
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

      // Here wq = (ip.weight |nor|)(Q/h), so face_w below is the physical
      // measure at the point and the rest is the stabilization itself.
      real_t w;
      if (!stab)
      {
         w = wq * (b+a);
      }
      else
      {
         const real_t face_w = ip.weight * nor.Norml2();
         w = StabValue(wq, b+a, un_raw, face_w, el_shape * elfun,
                       tr_shape * trfun, *ElTr);
      }
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

void HDGDiffusionIntegrator::AssembleHDGFaceGrad(
   int type, const FiniteElement &trace_el, const FiniteElement &el,
   FaceElementTransformations &Trans, const Vector &trfun, const Vector &elfun,
   DenseMatrix &grad)
{
#ifdef MFEM_THREAD_SAFE
   Vector tr_shape, shape1, vu, nor, nh, ni;
   DenseMatrix mq;
#endif
   if (!stab || stab->IsConstant())
   {
      // Nothing state dependent to carry, so the base class building the
      // gradient out of the face matrix is exactly right. This is the path
      // every existing caller takes and it is left untouched.
      BilinearFormIntegrator::AssembleHDGFaceGrad(type, trace_el, el, Trans,
                                                  trfun, elfun, grad);
      return;
   }

   MFEM_VERIFY(trace_el.GetMapType() == FiniteElement::VALUE, "");

   if (Trans.Elem2No < 0) { type &= ~1; }

   const int dim = el.GetDim();
   const int tr_ndof = trace_el.GetDof();
   const int el_ndof = el.GetDof();

   MFEM_VERIFY(elfun.Size() == el_ndof && trfun.Size() == tr_ndof,
               "A state dependent stabilization is implemented for a single "
               "equation; for a system s becomes a matrix over the equations, "
               "which is a question about the formulation.");

   vu.SetSize(dim);
   nor.SetSize(dim);
   nh.SetSize(dim);
   ni.SetSize(dim);
   if (MQ) { mq.SetSize(dim); }

   tr_shape.SetSize(tr_ndof);
   Vector &el_shape = shape1;
   el_shape.SetSize(el_ndof);

   // Accumulate the full one-sided matrix in the [element | trace] layout that
   // AssembleHDGFaceMatrix() produces, so that the extraction at the end can
   // follow the base class exactly.
   DenseMatrix full(el_ndof + tr_ndof);
   full = 0.0;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      const int order = 2*max(el.GetOrder(), trace_el.GetOrder());
      ir = &IntRules.Get(Trans.GetGeometryType(), order);
   }

   for (int p = 0; p < ir->GetNPoints(); p++)
   {
      const IntegrationPoint &ip = ir->IntPoint(p);
      Trans.SetAllIntPoints(&ip);

      const IntegrationPoint &eip1 = Trans.GetElement1IntPoint();
      const IntegrationPoint &eip2 = Trans.GetElement2IntPoint();

      if (dim == 1) { nor(0) = 2*eip1.x - 1.0; }
      else { CalcOrtho(Trans.Jacobian(), nor); }

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

      ElementTransformation *ElTr = (type & 1)?(Trans.Elem2):(Trans.Elem1);
      const IntegrationPoint &eip = (type & 1)?(eip2):(eip1);

      el.CalcPhysShape(*ElTr, el_shape);
      real_t wn = ip.weight/ElTr->Weight();
      if (!MQ)
      {
         if (Q) { wn *= Q->Eval(*ElTr, eip); }
         ni.Set(wn, nor);
      }
      else
      {
         nh.Set(wn, nor);
         MQ->Eval(mq, *ElTr, eip);
         mq.MultTranspose(nh, ni);
      }
      const real_t wq = ni * nor;

      const real_t un_raw = un;
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

      const real_t face_w = ip.weight * nor.Norml2();
      if (face_w == 0.) { continue; }
      const real_t s_diff = wq * (b+a) / face_w;

      const real_t u_q = el_shape * elfun;
      const real_t t_q = tr_shape * trfun;

      const real_t s = stab->Eval(s_diff, un_raw, u_q, t_q, *ElTr);
      real_t d1s, d2s;
      stab->EvalGrad(s_diff, un_raw, u_q, t_q, *ElTr, d1s, d2s);

      // The residual contributes s(u,uhat)(u - uhat) against both test spaces,
      // so its derivatives are s + d1s (u - uhat) with respect to the potential
      // and -s + d2s (u - uhat) with respect to the trace. Those are the
      // coefficients of the G, and of the E and H, blocks of Eq. (15) of
      // Nguyen, Peraire and Cockburn, once the convective term F'(uhat).n
      // carried by the convection integrators is set aside.
      const real_t jump = u_q - t_q;
      const real_t cu = face_w * ( s + d1s * jump);
      const real_t ct = face_w * (-s + d2s * jump);

      for (int i = 0; i < el_ndof; i++)
      {
         const real_t si = el_shape(i);
         for (int j = 0; j < el_ndof; j++)
         {
            full(i, j) += cu * si * el_shape(j);
         }
         for (int j = 0; j < tr_ndof; j++)
         {
            full(i, el_ndof + j) += ct * si * tr_shape(j);
         }
      }
      for (int i = 0; i < tr_ndof; i++)
      {
         const real_t si = tr_shape(i);
         for (int j = 0; j < el_ndof; j++)
         {
            full(el_ndof + i, j) += cu * si * el_shape(j);
         }
         for (int j = 0; j < tr_ndof; j++)
         {
            full(el_ndof + i, el_ndof + j) += ct * si * tr_shape(j);
         }
      }
   }

   int h = 0, w = 0;
   if (type & (HDGFaceType::ELEM | HDGFaceType::TRACE))  { h += el_ndof; }
   if (type & (HDGFaceType::CONSTR | HDGFaceType::FACE)) { h += tr_ndof; }
   if (type & (HDGFaceType::ELEM | HDGFaceType::CONSTR)) { w += el_ndof; }
   if (type & (HDGFaceType::TRACE | HDGFaceType::FACE))  { w += tr_ndof; }

   grad.SetSize(h, w);
   grad = 0.;

   int ioff = 0, joff = 0;
   if (type & HDGFaceType::ELEM)
   {
      grad.CopyMN(full, el_ndof, el_ndof, 0, 0, ioff, joff);
   }
   if (type & (HDGFaceType::ELEM | HDGFaceType::CONSTR)) { joff += el_ndof; }
   if (type & HDGFaceType::TRACE)
   {
      grad.CopyMN(full, el_ndof, tr_ndof, 0, el_ndof, ioff, joff);
   }
   if (type & (HDGFaceType::ELEM | HDGFaceType::TRACE)) { ioff += el_ndof; }
   joff = 0;
   if (type & HDGFaceType::CONSTR)
   {
      grad.CopyMN(full, tr_ndof, el_ndof, el_ndof, 0, ioff, joff);
   }
   if (type & (HDGFaceType::ELEM | HDGFaceType::CONSTR)) { joff += el_ndof; }
   if (type & HDGFaceType::FACE)
   {
      grad.CopyMN(full, tr_ndof, tr_ndof, el_ndof, el_ndof, ioff, joff);
   }
}

real_t HDGDiffusionIntegrator::ComputeHDGFaceEnergy(
   int side, const FiniteElement &trace_el, const FiniteElement &el,
   FaceElementTransformations &Trans, const Vector &trfun, const Vector &elfun,
   Vector *d_energy)
{
#ifdef MFEM_THREAD_SAFE
   Vector tr_shape, shape1, vu, nor, nh, ni;
   Vector nor_Jt, nor_Ji, ni_Jt, ni_Ji;
   DenseMatrix mq;
#endif
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

      ni_Ji.SetSize(dim);
      nor_Jt.SetSize(dim);
      if (MQ)
      {
         nor_Ji.SetSize(dim);
         ni_Jt.SetSize(dim);
      }
   }
   real_t energy = 0.;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      // Degree 2*max(element, trace): see the note at the top of this file
      // for why the trace element has to be in the max.
      int order;
      order = 2*max(el.GetOrder(), trace_el.GetOrder());
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

      // evaluate the normal velocity
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

      // calculate the stabilization coefficient
      ElementTransformation *ElTr = (side != 0)?(Trans.Elem2):(Trans.Elem1);
      const IntegrationPoint &eip = (side != 0)?(eip2):(eip1);

      el.CalcPhysShape(*ElTr, el_shape);
      real_t wn = ip.weight/ElTr->Weight();
      if (!MQ)
      {
         if (Q)
         {
            wn *= Q->Eval(*ElTr, eip);
         }
         ni.Set(wn, nor);
      }
      else
      {
         nh.Set(wn, nor);
         MQ->Eval(mq, *ElTr, eip);
         mq.MultTranspose(nh, ni);
      }
      const real_t wq = ni * nor;
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

      const real_t un_raw = un;
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

      // The weighted stabilization, taken exactly as the four assembly paths
      // take it. With no hook StabValue() returns wq*(b+a), so the energy is
      // the same number as before to the last bit; with one installed this is
      // the only place the hook's value can enter, and the estimator reported
      // the built-in expression instead until it did.
      const real_t face_w = ip.weight * nor.Norml2();
      const real_t s_w = StabValue(wq, b+a, un_raw, face_w, v_q, tr_q, *ElTr);

      const real_t w = d_q*d_q * fabs(s_w);
      if (w == 0.) { continue; }

      energy += w;

      if (d_energy)
      {
         // The split is geometry: the weights below sum to ni.nor over the
         // directions, so dividing by it distributes exactly the energy just
         // added, whatever the stabilization was. A face where the diffusion
         // vanishes has no directions to split along and gets none of it.
         const real_t wg = (wq != 0.) ? (w / wq) : 0.;

         ElTr->InverseJacobian().Mult(ni, ni_Ji);
         ElTr->Jacobian().MultTranspose(nor, nor_Jt);
         if (!MQ)
         {
            for (int d = 0; d < dim; d++)
            {
               (*d_energy)(d) += wg * ni_Ji(d) * nor_Jt(d);
            }
         }
         else
         {
            // symmetrize the product as:
            // 1/2 * (n J J^-1 MQ J^-T J^T n + n J^-T J^T MQ J^-1 J n) =
            // 1/2 * (J^T n . J^-1 MQ n + J^-1 n . J^T MQ n)
            ElTr->Jacobian().MultTranspose(ni, ni_Jt);
            ElTr->InverseJacobian().Mult(nor, nor_Ji);

            for (int d = 0; d < dim; d++)
            {
               (*d_energy)(d) += 0.5 * wg * (ni_Jt(d) * nor_Ji(d) + ni_Ji(d) * nor_Jt(d));
            }
         }
      }
   }

   return energy;
}

bool HDGDiffusionFaceMatricesCanBatch(const FiniteElementSpace &tr_fes,
                                      const FiniteElementSpace &el_fes)
{
   const Mesh *mesh = el_fes.GetMesh();
   if (!mesh || mesh->GetNE() == 0) { return false; }
   if (mesh->Nonconforming()) { return false; }
   if (tr_fes.GetVDim() != 1 || el_fes.GetVDim() != 1) { return false; }
   if (!dynamic_cast<const DG_Interface_FECollection*>(tr_fes.FEColl()))
   { return false; }

   // One element geometry and one order, so every face matrix is the same
   // size and the reference trace shapes are one table.
   const Geometry::Type g = mesh->GetElementBaseGeometry(0);
   const int nd = el_fes.GetFE(0)->GetDof();
   for (int e = 1; e < mesh->GetNE(); e++)
   {
      if (mesh->GetElementBaseGeometry(e) != g) { return false; }
      if (el_fes.GetFE(e)->GetDof() != nd) { return false; }
   }
   return true;
}

void HDGDiffusionFaceMatricesBatched(const FiniteElementSpace &tr_fes,
                                     const FiniteElementSpace &el_fes,
                                     Coefficient *Q, real_t beta,
                                     const HDGStabilization *stab,
                                     DenseTensor &elmats)
{
   MFEM_VERIFY(HDGDiffusionFaceMatricesCanBatch(tr_fes, el_fes),
               "the spaces do not admit the batched face assembly");
   MFEM_VERIFY(!stab || stab->IsConstant(),
               "a state dependent stabilization makes the face term nonlinear");

   Mesh *mesh = el_fes.GetMesh();
   const int nfaces = mesh->GetNumFaces();

   Array<int> flist;
   for (int f = 0; f < nfaces; f++)
   {
      if (mesh->FaceIsInterior(f)) { flist.Append(f); }
   }
   const int NF = flist.Size();

   const FiniteElement *tr_fe0 = tr_fes.GetFaceElement(flist.Size() ? flist[0] :
                                                       0);
   const int TRD = tr_fe0->GetDof();
   const int ND = el_fes.GetFE(0)->GetDof();
   const int SZ = 2 * ND + TRD;

   elmats.SetSize(SZ, SZ, NF);
   // Without this the kernel below runs on the HOST whatever the Device is
   // configured as -- Read() hands back a host pointer and mfem::forall
   // degrades to a plain loop. It is the same trap as the raw-pointer
   // DenseTensor in DarcyHybridization::InvertA(), measured there and walked
   // into again here: the first version of this kernel was silently host-only
   // and 1.5x SLOWER under -d cuda than under -d cpu.
   elmats.GetMemory().UseDevice(true);
   if (NF == 0) { return; }

   // The rule the per-face path would pick, so the two agree by construction.
   const int order = 2 * std::max(el_fes.GetFE(0)->GetOrder(),
                                  tr_fe0->GetOrder());
   FaceElementTransformations *ftr0 = mesh->GetInteriorFaceTransformations(
                                         flist[0]);
   MFEM_VERIFY(ftr0, "no interior face transformation");
   const IntegrationRule &ir = IntRules.Get(ftr0->GetGeometryType(), order);
   const int NQ = ir.GetNPoints();

   // ---- host precompute: two weights and the shapes per quadrature point ----
   Vector w1(NQ * NF), w2(NQ * NF);
   Vector sh1(NQ * ND * NF), sh2(NQ * ND * NF);
   Vector trs(NQ * TRD);
   w1.UseDevice(true);
   w2.UseDevice(true);
   sh1.UseDevice(true);
   sh2.UseDevice(true);
   trs.UseDevice(true);

   {
      Vector t(TRD);
      for (int q = 0; q < NQ; q++)
      {
         tr_fe0->CalcShape(ir.IntPoint(q), t);
         for (int i = 0; i < TRD; i++) { trs(q + i * NQ) = t(i); }
      }
   }

   {
      const int dim = mesh->Dimension();
      Vector nor(dim), s1(ND), s2(ND);
      real_t *pw1 = w1.HostWrite(), *pw2 = w2.HostWrite();
      real_t *ps1 = sh1.HostWrite(), *ps2 = sh2.HostWrite();

      for (int fi = 0; fi < NF; fi++)
      {
         FaceElementTransformations *ftr =
            mesh->GetInteriorFaceTransformations(flist[fi]);
         const FiniteElement &e1 = *el_fes.GetFE(ftr->Elem1No);
         const FiniteElement &e2 = *el_fes.GetFE(ftr->Elem2No);

         for (int q = 0; q < NQ; q++)
         {
            const IntegrationPoint &ip = ir.IntPoint(q);
            ftr->SetAllIntPoints(&ip);
            const IntegrationPoint &eip1 = ftr->GetElement1IntPoint();
            const IntegrationPoint &eip2 = ftr->GetElement2IntPoint();

            if (dim == 1) { nor(0) = 2 * eip1.x - 1.0; }
            else { CalcOrtho(ftr->Jacobian(), nor); }

            e1.CalcPhysShape(*ftr->Elem1, s1);
            e2.CalcPhysShape(*ftr->Elem2, s2);

            const real_t nn = nor * nor;
            const real_t face_w = ip.weight * nor.Norml2();

            real_t wn1 = ip.weight / ftr->Elem1->Weight();
            if (Q) { wn1 *= Q->Eval(*ftr->Elem1, eip1); }
            real_t wn2 = ip.weight / ftr->Elem2->Weight();
            if (Q) { wn2 *= Q->Eval(*ftr->Elem2, eip2); }

            // v is null on this path, so un == 0 and a == 0, b == beta.
            const real_t wq1 = wn1 * nn, wq2 = wn2 * nn;
            const real_t v1 = stab
                              ? face_w * stab->Eval((face_w != 0.) ? (wq1 * beta / face_w) : 0.,
                                                    0., 0., 0., *ftr->Elem1)
                              : wq1 * beta;
            const real_t v2 = stab
                              ? face_w * stab->Eval((face_w != 0.) ? (wq2 * beta / face_w) : 0.,
                                                    0., 0., 0., *ftr->Elem2)
                              : wq2 * beta;
            pw1[q + fi * NQ] = v1;
            pw2[q + fi * NQ] = v2;
            for (int i = 0; i < ND; i++)
            {
               ps1[q + NQ * (i + ND * fi)] = s1(i);
               ps2[q + NQ * (i + ND * fi)] = s2(i);
            }
         }
      }
   }

   // ---- the kernel ----
   // Measured, 2-D quads, steady state (first call is 100x on either backend,
   // allocation and first touch -- do not time one call):
   //
   //   n=64 order 3   precompute   kernel   copy-back
   //     -d cpu         16.8 ms    26.3 ms     0.0 ms
   //     -d cuda        16.9 ms    32.7 ms    10.9 ms
   //
   // THE COPY-BACK IS THE FINDING, not the kernel time. This routine's output
   // is consumed by host code (ComputeAndAssemblePotFaceMatrix splitting it
   // into E/G/H/D), so every face matrix has to come back -- at n=96 order 3
   // that is 193 MB and 183 ms against an 85 ms kernel. An isolated device
   // kernel whose consumer is on the host pays more in transfer than it saves
   // in arithmetic, which is the plan's own gate argument arriving from the
   // integrator side. The kernel is worth having; it is not worth switching on
   // until the consumers are device-resident too.
   //
   // The kernel share is what makes it worth having at all: 47% of this
   // routine at order 1, 72% at order 2, 83% at order 3, because the outer
   // products are O(NQ*SZ^2) against O(NQ*ND) for the shapes.
   //
   // GPU timings here are from a consumer card shared with a desktop under
   // WSL2 and are not a verdict on the approach.
   // ---- one face per thread ----
   const auto d_w1 = Reshape(w1.Read(), NQ, NF);
   const auto d_w2 = Reshape(w2.Read(), NQ, NF);
   const auto d_s1 = Reshape(sh1.Read(), NQ, ND, NF);
   const auto d_s2 = Reshape(sh2.Read(), NQ, ND, NF);
   const auto d_tr = Reshape(trs.Read(), NQ, TRD);
   auto d_M = Reshape(elmats.Write(), SZ, SZ, NF);

   const int nd = ND, trd = TRD, nq = NQ, sz = SZ;

   mfem::forall(NF, [=] MFEM_HOST_DEVICE (int f)
   {
      for (int j = 0; j < sz; j++)
         for (int i = 0; i < sz; i++)
         {
            d_M(i, j, f) = 0.0;
         }

      for (int q = 0; q < nq; q++)
      {
         const real_t a1 = d_w1(q, f), a2 = d_w2(q, f);

         // D blocks, both sides
         for (int i = 0; i < nd; i++)
         {
            const real_t b1 = a1 * d_s1(q, i, f);
            const real_t b2 = a2 * d_s2(q, i, f);
            for (int j = 0; j < nd; j++)
            {
               d_M(i, j, f)           += b1 * d_s1(q, j, f);
               d_M(nd + i, nd + j, f) += b2 * d_s2(q, j, f);
            }
            // E blocks, and G = -E^T
            for (int j = 0; j < trd; j++)
            {
               const real_t e1 = b1 * d_tr(q, j);
               const real_t e2 = b2 * d_tr(q, j);
               d_M(i, 2 * nd + j, f)      -= e1;
               d_M(nd + i, 2 * nd + j, f) -= e2;
               d_M(2 * nd + j, i, f)      += e1;
               d_M(2 * nd + j, nd + i, f) += e2;
            }
         }

         // the face block, once, with both sides' weights
         const real_t aa = a1 + a2;
         for (int i = 0; i < trd; i++)
         {
            const real_t t = aa * d_tr(q, i);
            for (int j = 0; j < trd; j++)
            {
               d_M(2 * nd + i, 2 * nd + j, f) -= t * d_tr(q, j);
            }
         }
      }
   });

}


namespace
{

/// Which of the three face terms the batched kernel implements @a bfi is.
enum class HDGBatchKind { Diffusion, ConvCentered, ConvUpwinded, Unsupported };

HDGBatchKind HDGBatchKindOf(const BilinearFormIntegrator *bfi)
{
   // Order matters only in that the convection pair are siblings; a
   // dynamic_cast to either cannot match the other.
   if (dynamic_cast<const HDGDiffusionIntegrator*>(bfi))
   { return HDGBatchKind::Diffusion; }
   if (dynamic_cast<const HDGConvectionCenteredIntegrator*>(bfi))
   { return HDGBatchKind::ConvCentered; }
   if (dynamic_cast<const HDGConvectionUpwindedIntegrator*>(bfi))
   { return HDGBatchKind::ConvUpwinded; }
   return HDGBatchKind::Unsupported;
}

/** @brief The rule @a bfi integrates this face at, asked of the integrator
    rather than reconstructed. A rule the caller set explicitly wins, exactly
    as it does in AssembleHDGFaceMatrix(). */
const IntegrationRule *HDGBatchRule(const BilinearFormIntegrator *bfi,
                                    const FiniteElement &tr_fe,
                                    const FiniteElement &e1,
                                    const FiniteElement &e2,
                                    FaceElementTransformations &ftr)
{
   if (const IntegrationRule *ir = bfi->GetIntRule()) { return ir; }
   if (auto *d = dynamic_cast<const HDGDiffusionIntegrator*>(bfi))
   { return &d->GetHDGFaceIntRule(tr_fe, e1, e2, ftr); }
   if (auto *c = dynamic_cast<const HDGConvectionCenteredIntegrator*>(bfi))
   { return &c->GetHDGFaceIntRule(tr_fe, e1, e2, ftr); }
   if (auto *u = dynamic_cast<const HDGConvectionUpwindedIntegrator*>(bfi))
   { return &u->GetHDGFaceIntRule(tr_fe, e1, e2, ftr); }
   return NULL;
}

/// The seven per-(point, face) weight streams of HDGFaceScatterBatched().
struct HDGFaceWeights
{
   Vector d1, d2, e1, e2, g1, g2, h;

   void Init(int n)
   {
      Vector *all[7] = { &d1, &d2, &e1, &e2, &g1, &g2, &h };
      for (Vector *v : all) { v->SetSize(n); *v = 0.; v->UseDevice(true); }
   }
};

} // namespace

bool HDGFaceScatterCanBatch(const FiniteElementSpace &tr_fes,
                            const FiniteElementSpace &el_fes,
                            const Array<BilinearFormIntegrator*> &integs,
                            const Array<int> &face_list)
{
   if (integs.Size() == 0) { return false; }
   if (!HDGDiffusionFaceMatricesCanBatch(tr_fes, el_fes)) { return false; }
   if (face_list.Size() == 0) { return true; }

   Mesh *mesh = el_fes.GetMesh();
   for (BilinearFormIntegrator *bfi : integs)
   {
      if (HDGBatchKindOf(bfi) == HDGBatchKind::Unsupported) { return false; }
      // A state dependent stabilization is not a bilinear form at all; the
      // per-face routine refuses it too.
      if (auto *d = dynamic_cast<const HDGDiffusionIntegrator*>(bfi))
      {
         const HDGStabilization *st = d->GetStabilization();
         if (st && !st->IsConstant()) { return false; }
      }

      // ONE rule for the whole face list. The convection forms take
      // ElementTransformation::OrderW() into their rule, so a mesh whose
      // elements do not all report the same one would want a different number
      // of points on different faces -- which the kernel, sampling every face
      // at one rule, cannot do. Asked of the faces because that is where the
      // answer is; HDGDiffusionFaceMatricesCanBatch() checks the geometry and
      // the dof counts and could not see this.
      const IntegrationRule *ir0 = NULL;
      for (int fi = 0; fi < face_list.Size(); fi++)
      {
         FaceElementTransformations *ftr =
            mesh->GetInteriorFaceTransformations(face_list[fi]);
         if (!ftr) { return false; }
         const IntegrationRule *ir =
            HDGBatchRule(bfi, *tr_fes.GetFaceElement(face_list[fi]),
                         *el_fes.GetFE(ftr->Elem1No),
                         *el_fes.GetFE(ftr->Elem2No), *ftr);
         if (!ir) { return false; }
         if (!ir0) { ir0 = ir; }
         else if (ir != ir0) { return false; }
      }
   }
   return true;
}

void HDGFaceScatterBatched(const FiniteElementSpace &tr_fes,
                           const FiniteElementSpace &el_fes,
                           const Array<BilinearFormIntegrator*> &integs,
                           const Array<int> &face_list,
                           const Array<int> &E_offsets,
                           const Array<int> &H_offsets,
                           const Array<int> &Df_offsets,
                           Vector &E_data, Vector &G_data,
                           Vector &H_data, Vector &Df_data)
{
   MFEM_VERIFY(HDGFaceScatterCanBatch(tr_fes, el_fes, integs, face_list),
               "these integrators do not admit the batched face assembly");

   Mesh *mesh = el_fes.GetMesh();
   const int NF = face_list.Size();
   if (NF == 0) { return; }

   const FiniteElement *tr_fe0 = tr_fes.GetFaceElement(face_list[0]);
   const int TRD = tr_fe0->GetDof();
   const int ND = el_fes.GetFE(0)->GetDof();

   // Face-indexed offsets, shared by every pass.
   Array<int> eo(NF), ho(NF), d1o(NF), d2o(NF);
   for (int fi = 0; fi < NF; fi++)
   {
      const int face = face_list[fi];
      FaceElementTransformations *ftr =
         mesh->GetInteriorFaceTransformations(face);
      eo[fi] = E_offsets[face];
      ho[fi] = H_offsets[face];
      d1o[fi] = Df_offsets[ftr->Elem1No];
      d2o[fi] = Df_offsets[ftr->Elem2No];
   }

   {
      // E, G and H belong to a face alone, so they are zeroed once here and
      // every integrator's pass then accumulates. D is NOT zeroed: it
      // accumulates across the faces of an element and across the potential
      // mass form as well, so whoever owns it zeroes it.
      const int *d_eo = eo.Read(), *d_ho = ho.Read();
      real_t *d_E = E_data.ReadWrite();
      real_t *d_G = G_data.ReadWrite();
      real_t *d_H = H_data.ReadWrite();
      const int nd = ND, trd = TRD;
      mfem::forall(NF, [=] MFEM_HOST_DEVICE (int f)
      {
         const int e0 = d_eo[f], h0 = d_ho[f];
         const int e2off = e0 + trd * nd;      // side 2 follows side 1
         for (int j = 0; j < trd; j++)
            for (int i = 0; i < nd; i++)
            {
               d_E[e0 + i + nd * j] = 0.0;
               d_E[e2off + i + nd * j] = 0.0;
               d_G[e0 + j + trd * i] = 0.0;
               d_G[e2off + j + trd * i] = 0.0;
            }
         for (int j = 0; j < trd; j++)
            for (int i = 0; i < trd; i++)
            {
               d_H[h0 + i + trd * j] = 0.0;
            }
      });
   }

   const int dim = mesh->Dimension();

   for (BilinearFormIntegrator *bfi : integs)
   {
      const HDGBatchKind kind = HDGBatchKindOf(bfi);
      auto *dif = dynamic_cast<const HDGDiffusionIntegrator*>(bfi);
      auto *tr_bfi = dynamic_cast<const DGTraceIntegrator*>(bfi);

      FaceElementTransformations *ftr0 =
         mesh->GetInteriorFaceTransformations(face_list[0]);
      const IntegrationRule &ir =
         *HDGBatchRule(bfi, *tr_fe0, *el_fes.GetFE(ftr0->Elem1No),
                       *el_fes.GetFE(ftr0->Elem2No), *ftr0);
      const int NQ = ir.GetNPoints();

      HDGFaceWeights w;
      w.Init(NQ * NF);
      Vector sh1(NQ * ND * NF), sh2(NQ * ND * NF), trs(NQ * TRD);
      sh1.UseDevice(true);
      sh2.UseDevice(true);
      trs.UseDevice(true);

      {
         Vector t(TRD);
         for (int q = 0; q < NQ; q++)
         {
            tr_fe0->CalcShape(ir.IntPoint(q), t);
            for (int i = 0; i < TRD; i++) { trs(q + i * NQ) = t(i); }
         }
      }

      {
         Vector nor(dim), vu(dim), nh(dim), ni(dim), s1(ND), s2(ND);
         DenseMatrix mq(dim);
         real_t *pd1 = w.d1.HostWrite(), *pd2 = w.d2.HostWrite();
         real_t *pe1 = w.e1.HostWrite(), *pe2 = w.e2.HostWrite();
         real_t *pg1 = w.g1.HostWrite(), *pg2 = w.g2.HostWrite();
         real_t *ph = w.h.HostWrite();
         real_t *ps1 = sh1.HostWrite(), *ps2 = sh2.HostWrite();

         for (int fi = 0; fi < NF; fi++)
         {
            FaceElementTransformations *ftr =
               mesh->GetInteriorFaceTransformations(face_list[fi]);
            const FiniteElement &e1 = *el_fes.GetFE(ftr->Elem1No);
            const FiniteElement &e2 = *el_fes.GetFE(ftr->Elem2No);

            for (int q = 0; q < NQ; q++)
            {
               const IntegrationPoint &ip = ir.IntPoint(q);
               ftr->SetAllIntPoints(&ip);
               const IntegrationPoint &eip1 = ftr->GetElement1IntPoint();
               const IntegrationPoint &eip2 = ftr->GetElement2IntPoint();

               if (dim == 1) { nor(0) = 2 * eip1.x - 1.0; }
               else { CalcOrtho(ftr->Jacobian(), nor); }

               e1.CalcPhysShape(*ftr->Elem1, s1);
               e2.CalcPhysShape(*ftr->Elem2, s2);
               for (int i = 0; i < ND; i++)
               {
                  ps1[q + NQ * (i + ND * fi)] = s1(i);
                  ps2[q + NQ * (i + ND * fi)] = s2(i);
               }

               const int o = q + fi * NQ;
               if (kind == HDGBatchKind::Diffusion)
               {
                  // The velocity enters only through sign(u.n) and |u.n|; see
                  // HDGDiffusionIntegrator::AssembleHDGFaceMatrix().
                  real_t un = 0.;
                  if (VectorCoefficient *v = dif->GetVelocity())
                  {
                     v->Eval(vu, *ftr->Elem1, eip1);
                     un = vu * nor;
                  }
                  const real_t un_raw = un;
                  real_t a, b;
                  if (un != 0.)
                  {
                     un /= std::fabs(un);
                     a = 0.5 * dif->GetAlpha() * un;
                     b = dif->GetBeta() * std::fabs(un);
                  }
                  else { a = 0.; b = dif->GetBeta(); }

                  Coefficient *Q = dif->GetCoefficient();
                  MatrixCoefficient *MQ = dif->GetMatrixCoefficient();
                  real_t wq[2];
                  ElementTransformation *el[2] = { ftr->Elem1, ftr->Elem2 };
                  const IntegrationPoint *eip[2] = { &eip1, &eip2 };
                  for (int side = 0; side < 2; side++)
                  {
                     const real_t wn = ip.weight / el[side]->Weight();
                     if (!MQ)
                     {
                        ni.Set(Q ? (wn * Q->Eval(*el[side], *eip[side])) : wn,
                               nor);
                     }
                     else
                     {
                        nh.Set(wn, nor);
                        MQ->Eval(mq, *el[side], *eip[side]);
                        mq.MultTranspose(nh, ni);
                     }
                     wq[side] = ni * nor;
                  }

                  const real_t face_w = ip.weight * nor.Norml2();
                  const real_t w1 = dif->EvalStabilization(
                                       wq[0], b + a, un_raw, face_w, 0., 0., *ftr->Elem1);
                  const real_t w2 = dif->EvalStabilization(
                                       wq[1], b - a, un_raw, face_w, 0., 0., *ftr->Elem2);
                  pd1[o] += w1;  pe1[o] += w1;  pg1[o] += w1;
                  pd2[o] += w2;  pe2[o] += w2;  pg2[o] += w2;
                  ph[o] += w1 + w2;
               }
               else
               {
                  tr_bfi->GetVelocity()->Eval(vu, *ftr->Elem1, eip1);
                  const real_t un = vu * nor;
                  const real_t alpha = tr_bfi->GetAlpha();
                  real_t wp, wm, wd;
                  if (kind == HDGBatchKind::ConvCentered)
                  {
                     // D and G take |a u.n| on both sides; E takes b-a on
                     // side 1 and b+a on side 2.
                     const real_t a = alpha * un, b = std::fabs(alpha * un);
                     wd = ip.weight * b;
                     wm = ip.weight * (b - a);
                     wp = ip.weight * (b + a);
                     pd1[o] += wd;  pg1[o] += wd;  pe1[o] += wm;
                     pd2[o] += wd;  pg2[o] += wd;  pe2[o] += wp;
                     ph[o] += 2. * wd;
                  }
                  else
                  {
                     // Upwinded: D and G take their own side's weight and E
                     // takes the OTHER side's. That crossing is the whole
                     // difference from the centred form.
                     const real_t a = 0.5 * alpha * un;
                     const real_t b = tr_bfi->GetBeta() * std::fabs(un);
                     wp = ip.weight * (b + a);
                     wm = ip.weight * (b - a);
                     pd1[o] += wp;  pg1[o] += wp;  pe1[o] += wm;
                     pd2[o] += wm;  pg2[o] += wm;  pe2[o] += wp;
                     ph[o] += wp + wm;
                  }
               }
            }
         }
      }

      const auto d_d1 = Reshape(w.d1.Read(), NQ, NF);
      const auto d_d2 = Reshape(w.d2.Read(), NQ, NF);
      const auto d_e1 = Reshape(w.e1.Read(), NQ, NF);
      const auto d_e2 = Reshape(w.e2.Read(), NQ, NF);
      const auto d_g1 = Reshape(w.g1.Read(), NQ, NF);
      const auto d_g2 = Reshape(w.g2.Read(), NQ, NF);
      const auto d_h  = Reshape(w.h.Read(),  NQ, NF);
      const auto d_s1 = Reshape(sh1.Read(), NQ, ND, NF);
      const auto d_s2 = Reshape(sh2.Read(), NQ, ND, NF);
      const auto d_tr = Reshape(trs.Read(), NQ, TRD);
      const int *d_eo = eo.Read(), *d_ho = ho.Read();
      const int *d_o1 = d1o.Read(), *d_o2 = d2o.Read();

      real_t *d_E = E_data.ReadWrite();
      real_t *d_G = G_data.ReadWrite();
      real_t *d_H = H_data.ReadWrite();
      real_t *d_D = Df_data.ReadWrite();

      const int nd = ND, trd = TRD, nq = NQ;

      mfem::forall(NF, [=] MFEM_HOST_DEVICE (int f)
      {
         const int e0 = d_eo[f], h0 = d_ho[f];
         const int o1 = d_o1[f], o2 = d_o2[f];
         const int e2off = e0 + trd * nd;

         for (int q = 0; q < nq; q++)
         {
            const real_t ad1 = d_d1(q, f), ad2 = d_d2(q, f);
            const real_t ae1 = d_e1(q, f), ae2 = d_e2(q, f);
            const real_t ag1 = d_g1(q, f), ag2 = d_g2(q, f);

            for (int i = 0; i < nd; i++)
            {
               const real_t s1i = d_s1(q, i, f), s2i = d_s2(q, i, f);

               // D accumulates per ELEMENT, and two faces of one element
               // collide, so it is the one block that needs atomics.
               for (int j = 0; j < nd; j++)
               {
                  AtomicAdd(d_D[o1 + i + nd * j], ad1 * s1i * d_s1(q, j, f));
                  AtomicAdd(d_D[o2 + i + nd * j], ad2 * s2i * d_s2(q, j, f));
               }
               for (int j = 0; j < trd; j++)
               {
                  const real_t t = d_tr(q, j);
                  d_E[e0 + i + nd * j]    -= ae1 * s1i * t;
                  d_E[e2off + i + nd * j] -= ae2 * s2i * t;
                  d_G[e0 + j + trd * i]    += ag1 * t * s1i;
                  d_G[e2off + j + trd * i] += ag2 * t * s2i;
               }
            }

            const real_t ah = d_h(q, f);
            for (int i = 0; i < trd; i++)
            {
               const real_t t = ah * d_tr(q, i);
               for (int j = 0; j < trd; j++)
               {
                  d_H[h0 + i + trd * j] -= t * d_tr(q, j);
               }
            }
         }
      });
   }
}



namespace
{

/// The rule @a bfi integrates @a el at, asked of the integrator.
const IntegrationRule *HDGMassRule(const BilinearFormIntegrator *bfi,
                                   const FiniteElement &el,
                                   ElementTransformation &Tr)
{
   if (auto *m = dynamic_cast<const MassIntegrator*>(bfi))
   { return &m->GetElementIntRule(el, Tr); }
   if (auto *v = dynamic_cast<const VectorMassIntegrator*>(bfi))
   { return &v->GetElementIntRule(el, Tr); }
   return NULL;
}

} // namespace

bool HDGElementMassCanBatch(const FiniteElementSpace &fes,
                            const Array<BilinearFormIntegrator*> &integs)
{
   if (integs.Size() == 0) { return false; }

   Mesh *mesh = fes.GetMesh();
   const int NE = fes.GetNE();
   if (!mesh || NE == 0) { return false; }

   // One element geometry and one dof count, so every block is the same size
   // and one shape table serves the mesh.
   const Geometry::Type g = mesh->GetElementBaseGeometry(0);
   const int nd = fes.GetFE(0)->GetDof();
   for (int e = 1; e < NE; e++)
   {
      if (mesh->GetElementBaseGeometry(e) != g) { return false; }
      if (fes.GetFE(e)->GetDof() != nd) { return false; }
   }

   const int vd = fes.GetVDim();
   for (BilinearFormIntegrator *bfi : integs)
   {
      auto *m = dynamic_cast<const MassIntegrator*>(bfi);
      auto *v = dynamic_cast<const VectorMassIntegrator*>(bfi);
      if (!m && !v) { return false; }
      // A scalar MassIntegrator on a vector space would assemble one block
      // where the space wants vd of them; the per-element route would too, so
      // this is a configuration nobody has, not a case to support.
      if (m && vd != 1) { return false; }
      if (v)
      {
         // vdim == -1 is VectorMassIntegrator's LAZY DEFAULT, not an error:
         // AssembleElementMatrix() resolves it to Trans.GetSpaceDim() on
         // first use, so before any assembly the accessor reports -1 on every
         // integrator built from a scalar Coefficient. Refusing on that would
         // refuse the flux mass of every miniapp in the tree -- which it did,
         // silently, until the report said "flux n" on a case that plainly
         // qualified.
         const int ivd = (v->GetVDim() == -1)
                         ? mesh->SpaceDimension() : v->GetVDim();
         if (ivd != vd) { return false; }
      }

      const IntegrationRule *ir0 = NULL;
      for (int e = 0; e < NE; e++)
      {
         ElementTransformation *Tr = mesh->GetElementTransformation(e);
         const IntegrationRule *ir = HDGMassRule(bfi, *fes.GetFE(e), *Tr);
         if (!ir) { return false; }
         if (!ir0) { ir0 = ir; }
         else if (ir != ir0) { return false; }
      }
   }
   return true;
}

void HDGElementMassBatched(const FiniteElementSpace &fes,
                           const Array<BilinearFormIntegrator*> &integs,
                           Vector &emat)
{
   MFEM_VERIFY(HDGElementMassCanBatch(fes, integs),
               "these integrators do not admit the batched element assembly");

   Mesh *mesh = fes.GetMesh();
   const int NE = fes.GetNE();
   const int ND = fes.GetFE(0)->GetDof();
   const int VD = fes.GetVDim();
   const int N = ND * VD;

   emat.SetSize(N * N * NE);
   emat.UseDevice(true);
   emat = 0.;
   if (NE == 0) { return; }

   for (BilinearFormIntegrator *bfi : integs)
   {
      auto *vm = dynamic_cast<const VectorMassIntegrator*>(bfi);
      auto *sm = dynamic_cast<const MassIntegrator*>(bfi);

      ElementTransformation *Tr0 = mesh->GetElementTransformation(0);
      const IntegrationRule &ir = *HDGMassRule(bfi, *fes.GetFE(0), *Tr0);
      const int NQ = ir.GetNPoints();

      // One weight per (point, element, field pair). The coupled block is
      // what a MatrixCoefficient needs and it is the only thing that costs
      // here: VD*VD weights against VD for a diagonal one, on a vector of
      // NQ*NE. The kernel skips a zero block rather than the caller having to
      // say which shape it is.
      Vector wt(NQ * NE * VD * VD), sh(NQ * ND * NE);
      wt = 0.;
      wt.UseDevice(true);
      sh.UseDevice(true);

      {
         Vector shape(ND), vec(VD);
         DenseMatrix mc(VD);
         real_t *pw = wt.HostWrite(), *ps = sh.HostWrite();
         // MassIntegrator::GetCoefficient() is const-qualified on its
         // return and VectorMassIntegrator's is not, so the two cannot meet
         // in a ternary; and Coefficient::Eval() is non-const, so the const
         // one has to be cast. Upstream's asymmetry, not ours.
         const int ivd = vm ? ((vm->GetVDim() == -1) ? mesh->SpaceDimension()
                               : vm->GetVDim()) : 1;
         MFEM_VERIFY(ivd == VD, "vdim mismatch");
         Coefficient *Q = vm ? vm->GetCoefficient()
                          : const_cast<Coefficient*>(sm->GetCoefficient());
         VectorCoefficient *VQ = vm ? vm->GetVectorCoefficient() : NULL;
         MatrixCoefficient *MQ = vm ? vm->GetMatrixCoefficient() : NULL;

         for (int e = 0; e < NE; e++)
         {
            const FiniteElement &el = *fes.GetFE(e);
            ElementTransformation *Tr = mesh->GetElementTransformation(e);
            for (int q = 0; q < NQ; q++)
            {
               const IntegrationPoint &ip = ir.IntPoint(q);
               Tr->SetIntPoint(&ip);
               el.CalcPhysShape(*Tr, shape);
               for (int i = 0; i < ND; i++)
               {
                  ps[q + NQ * (i + ND * e)] = shape(i);
               }

               const real_t norm = ip.weight * Tr->Weight();
               const int o = (q + NQ * e) * VD * VD;
               if (MQ)
               {
                  MQ->Eval(mc, *Tr, ip);
                  for (int k = 0; k < VD; k++)
                     for (int l = 0; l < VD; l++)
                     {
                        pw[o + k + VD * l] += norm * mc(k, l);
                     }
               }
               else if (VQ)
               {
                  VQ->Eval(vec, *Tr, ip);
                  for (int k = 0; k < VD; k++)
                  {
                     pw[o + k + VD * k] += norm * vec(k);
                  }
               }
               else
               {
                  const real_t w = Q ? (norm * Q->Eval(*Tr, ip)) : norm;
                  for (int k = 0; k < VD; k++) { pw[o + k + VD * k] += w; }
               }
            }
         }
      }

      const auto d_w = Reshape(wt.Read(), VD, VD, NQ, NE);
      const auto d_s = Reshape(sh.Read(), NQ, ND, NE);
      real_t *d_M = emat.ReadWrite();

      const int nd = ND, vd = VD, nq = NQ, n = N;

      mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
      {
         real_t *M = d_M + n * n * e;
         for (int q = 0; q < nq; q++)
         {
            for (int k = 0; k < vd; k++)
               for (int l = 0; l < vd; l++)
               {
                  const real_t w = d_w(k, l, q, e);
                  if (w == 0.0) { continue; }
                  for (int b = 0; b < nd; b++)
                  {
                     const real_t wsb = w * d_s(q, b, e);
                     for (int a = 0; a < nd; a++)
                     {
                        // Field-outermost, native dof order: block (k,l) at
                        // rows nd*k and columns nd*l, which is what
                        // VectorMassIntegrator::AssembleElementMatrix() writes
                        // and what Ordering::byNODES indexes.
                        M[(nd * k + a) + n * (nd * l + b)] +=
                           wsb * d_s(q, a, e);
                     }
                  }
               }
         }
      });
   }
}


bool HDGElementDivCanBatch(const FiniteElementSpace &trial_fes,
                           const FiniteElementSpace &test_fes,
                           const Array<BilinearFormIntegrator*> &integs)
{
   if (integs.Size() == 0) { return false; }

   Mesh *mesh = trial_fes.GetMesh();
   const int NE = trial_fes.GetNE();
   if (!mesh || NE == 0) { return false; }
   if (mesh != test_fes.GetMesh()) { return false; }

   // The trial space carries one scalar basis per space dimension, which is
   // what makes the block (test dofs) x (sdim * trial dofs) and what the hat
   // dof mask assumes. An H(div) flux is a different element entirely and
   // takes VectorFEDivergenceIntegrator, which this does not implement.
   if (trial_fes.GetVDim() != mesh->SpaceDimension()) { return false; }
   if (test_fes.GetVDim() != 1) { return false; }

   const Geometry::Type g = mesh->GetElementBaseGeometry(0);
   const int ndu = trial_fes.GetFE(0)->GetDof();
   const int ndp = test_fes.GetFE(0)->GetDof();
   for (int e = 1; e < NE; e++)
   {
      if (mesh->GetElementBaseGeometry(e) != g) { return false; }
      if (trial_fes.GetFE(e)->GetDof() != ndu) { return false; }
      if (test_fes.GetFE(e)->GetDof() != ndp) { return false; }
   }

   for (BilinearFormIntegrator *bfi : integs)
   {
      auto *d = dynamic_cast<const VectorDivergenceIntegrator*>(bfi);
      if (!d) { return false; }

      const IntegrationRule *ir0 = NULL;
      for (int e = 0; e < NE; e++)
      {
         ElementTransformation *Tr = mesh->GetElementTransformation(e);
         const IntegrationRule *ir =
            &d->GetElementIntRule(*trial_fes.GetFE(e), *test_fes.GetFE(e), *Tr);
         if (!ir0) { ir0 = ir; }
         else if (ir != ir0) { return false; }
      }
   }
   return true;
}

void HDGElementDivBatched(const FiniteElementSpace &trial_fes,
                          const FiniteElementSpace &test_fes,
                          const Array<BilinearFormIntegrator*> &integs,
                          Vector &emat)
{
   MFEM_VERIFY(HDGElementDivCanBatch(trial_fes, test_fes, integs),
               "these integrators do not admit the batched divergence "
               "assembly");

   Mesh *mesh = trial_fes.GetMesh();
   const int NE = trial_fes.GetNE();
   const int NDU = trial_fes.GetFE(0)->GetDof();
   const int NDP = test_fes.GetFE(0)->GetDof();
   const int SDIM = mesh->SpaceDimension();
   const int DIM = mesh->Dimension();
   const int W = NDU * SDIM;

   emat.SetSize(NDP * W * NE);
   emat.UseDevice(true);
   emat = 0.;
   if (NE == 0) { return; }

   for (BilinearFormIntegrator *bfi : integs)
   {
      auto *dv = dynamic_cast<const VectorDivergenceIntegrator*>(bfi);

      ElementTransformation *Tr0 = mesh->GetElementTransformation(0);
      const IntegrationRule &ir =
         dv->GetElementIntRule(*trial_fes.GetFE(0), *test_fes.GetFE(0), *Tr0);
      const int NQ = ir.GetNPoints();

      // The physical gradient of every trial basis function, and the test
      // shape, at every point of every element -- plus one scalar weight.
      // That is all the kernel needs: the divergence of the vector basis
      // function (a, k) is d_k phi_a, so gshape IS the divergence table,
      // which is what DenseMatrix::GradToDiv() says by copying it verbatim.
      Vector gsh(NQ * NDU * SDIM * NE), psh(NQ * NDP * NE), wt(NQ * NE);
      gsh.UseDevice(true);
      psh.UseDevice(true);
      wt = 0.;
      wt.UseDevice(true);

      {
         DenseMatrix dshape(NDU, DIM), gshape(NDU, SDIM), Jadj(DIM, SDIM);
         Vector shape(NDP);
         real_t *pg = gsh.HostWrite(), *pp = psh.HostWrite();
         real_t *pw = wt.HostWrite();
         Coefficient *Q = dv->GetCoefficient();

         for (int e = 0; e < NE; e++)
         {
            const FiniteElement &fu = *trial_fes.GetFE(e);
            const FiniteElement &fp = *test_fes.GetFE(e);
            ElementTransformation *Tr = mesh->GetElementTransformation(e);

            for (int q = 0; q < NQ; q++)
            {
               const IntegrationPoint &ip = ir.IntPoint(q);
               Tr->SetIntPoint(&ip);

               fu.CalcDShape(ip, dshape);
               fp.CalcPhysShape(*Tr, shape);
               CalcAdjugate(Tr->Jacobian(), Jadj);
               Mult(dshape, Jadj, gshape);

               real_t c = ip.weight;
               if (DIM != SDIM) { c /= Tr->Weight(); }
               if (Q) { c *= Q->Eval(*Tr, ip); }
               pw[q + NQ * e] += c;

               for (int k = 0; k < SDIM; k++)
                  for (int a = 0; a < NDU; a++)
                  {
                     pg[q + NQ * (a + NDU * (k + SDIM * e))] = gshape(a, k);
                  }
               for (int i = 0; i < NDP; i++)
               {
                  pp[q + NQ * (i + NDP * e)] = shape(i);
               }
            }
         }
      }

      const auto d_g = Reshape(gsh.Read(), NQ, NDU, SDIM, NE);
      const auto d_p = Reshape(psh.Read(), NQ, NDP, NE);
      const auto d_w = Reshape(wt.Read(), NQ, NE);
      real_t *d_B = emat.ReadWrite();

      const int ndu = NDU, ndp = NDP, sdim = SDIM, nq = NQ, w = W;

      mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
      {
         real_t *B = d_B + ndp * w * e;
         for (int q = 0; q < nq; q++)
         {
            const real_t c = d_w(q, e);
            if (c == 0.0) { continue; }
            for (int k = 0; k < sdim; k++)
               for (int a = 0; a < ndu; a++)
               {
                  const real_t cg = c * d_g(q, a, k, e);
                  if (cg == 0.0) { continue; }
                  const int col = k * ndu + a;
                  for (int i = 0; i < ndp; i++)
                  {
                     B[i + ndp * col] += cg * d_p(q, i, e);
                  }
               }
         }
      });
   }
}

bool HDGBdrFaceScatterCanBatch(const FiniteElementSpace &tr_fes,
                               const FiniteElementSpace &el_fes,
                               const Array<BilinearFormIntegrator*> &integs,
                               const std::vector<Array<int>> &face_lists)
{
   if (integs.Size() == 0) { return false; }
   MFEM_ASSERT((size_t)integs.Size() == face_lists.size(), "one list each");
   if (!HDGDiffusionFaceMatricesCanBatch(tr_fes, el_fes)) { return false; }

   Mesh *mesh = el_fes.GetMesh();
   for (int k = 0; k < integs.Size(); k++)
   {
      BilinearFormIntegrator *bfi = integs[k];
      if (HDGBatchKindOf(bfi) == HDGBatchKind::Unsupported) { return false; }
      if (auto *d = dynamic_cast<const HDGDiffusionIntegrator*>(bfi))
      {
         const HDGStabilization *st = d->GetStabilization();
         if (st && !st->IsConstant()) { return false; }
      }

      // One rule across this integrator's own face list; see
      // HDGFaceScatterCanBatch(). Elem2 is absent, so the rule takes only
      // Elem1's OrderW() -- but that can still vary element to element.
      const IntegrationRule *ir0 = NULL;
      for (int fi = 0; fi < face_lists[k].Size(); fi++)
      {
         const int face = face_lists[k][fi];
         FaceElementTransformations *ftr =
            mesh->GetFaceElementTransformations(face);
         if (!ftr || ftr->Elem2No >= 0) { return false; }
         const FiniteElement &fe = *el_fes.GetFE(ftr->Elem1No);
         const IntegrationRule *ir =
            HDGBatchRule(bfi, *tr_fes.GetFaceElement(face), fe, fe, *ftr);
         if (!ir) { return false; }
         if (!ir0) { ir0 = ir; }
         else if (ir != ir0) { return false; }
      }
   }
   return true;
}

void HDGBdrFaceScatterBatched(const FiniteElementSpace &tr_fes,
                              const FiniteElementSpace &el_fes,
                              const Array<BilinearFormIntegrator*> &integs,
                              const std::vector<Array<int>> &face_lists,
                              const Array<int> &all_faces,
                              const Array<int> &E_offsets,
                              const Array<int> &H_offsets,
                              const Array<int> &Df_offsets,
                              Vector &E_data, Vector &G_data,
                              Vector &H_data, Vector &Df_data)
{
   MFEM_VERIFY(HDGBdrFaceScatterCanBatch(tr_fes, el_fes, integs, face_lists),
               "these integrators do not admit the batched boundary assembly");

   Mesh *mesh = el_fes.GetMesh();
   if (all_faces.Size() == 0) { return; }

   const int TRD = tr_fes.GetFaceElement(all_faces[0])->GetDof();
   const int ND = el_fes.GetFE(0)->GetDof();
   const int dim = mesh->Dimension();

   {
      // E, G and H over the union of the lists. The per-face route ASSIGNS
      // them with CopyMN, so zeroing here and accumulating below reproduces
      // it; D accumulates in both.
      Array<int> eo(all_faces.Size()), ho(all_faces.Size());
      for (int fi = 0; fi < all_faces.Size(); fi++)
      {
         eo[fi] = E_offsets[all_faces[fi]];
         ho[fi] = H_offsets[all_faces[fi]];
      }
      const int *d_eo = eo.Read(), *d_ho = ho.Read();
      real_t *d_E = E_data.ReadWrite();
      real_t *d_G = G_data.ReadWrite();
      real_t *d_H = H_data.ReadWrite();
      const int nd = ND, trd = TRD;
      mfem::forall(all_faces.Size(), [=] MFEM_HOST_DEVICE (int f)
      {
         const int e0 = d_eo[f], h0 = d_ho[f];
         for (int j = 0; j < trd; j++)
            for (int i = 0; i < nd; i++)
            {
               d_E[e0 + i + nd * j] = 0.0;
               d_G[e0 + j + trd * i] = 0.0;
            }
         for (int j = 0; j < trd; j++)
            for (int i = 0; i < trd; i++)
            {
               d_H[h0 + i + trd * j] = 0.0;
            }
      });
   }

   for (int k = 0; k < integs.Size(); k++)
   {
      BilinearFormIntegrator *bfi = integs[k];
      const Array<int> &flist = face_lists[k];
      const int NF = flist.Size();
      if (NF == 0) { continue; }

      const HDGBatchKind kind = HDGBatchKindOf(bfi);
      auto *dif = dynamic_cast<const HDGDiffusionIntegrator*>(bfi);
      auto *tr_bfi = dynamic_cast<const DGTraceIntegrator*>(bfi);

      const FiniteElement *tr_fe0 = tr_fes.GetFaceElement(flist[0]);
      FaceElementTransformations *ftr0 =
         mesh->GetFaceElementTransformations(flist[0]);
      const FiniteElement &fe0 = *el_fes.GetFE(ftr0->Elem1No);
      const IntegrationRule &ir =
         *HDGBatchRule(bfi, *tr_fe0, fe0, fe0, *ftr0);
      const int NQ = ir.GetNPoints();

      // ZEROED, and not merely sized. Vector(int) does not initialise, the
      // loop below only accumulates into these, and an uninitialised weight
      // is not a crash -- it is a plausible-looking face integral. Measured:
      // the E block of a face whose true weight is exactly zero came back
      // carrying another face's upwinded values, and the answer was 46% out.
      // HDGFaceWeights::Init() does this for the interior kernel; this loop
      // was open-coded and dropped it.
      Vector wd(NQ * NF), we(NQ * NF), wg(NQ * NF), wh(NQ * NF);
      Vector sh(NQ * ND * NF), trs(NQ * TRD);
      Array<int> eo(NF), ho(NF), dof(NF);
      wd = 0.;
      we = 0.;
      wg = 0.;
      wh = 0.;
      wd.UseDevice(true);
      we.UseDevice(true);
      wg.UseDevice(true);
      wh.UseDevice(true);
      sh.UseDevice(true);
      trs.UseDevice(true);

      {
         Vector t(TRD);
         for (int q = 0; q < NQ; q++)
         {
            tr_fe0->CalcShape(ir.IntPoint(q), t);
            for (int i = 0; i < TRD; i++) { trs(q + i * NQ) = t(i); }
         }
      }

      {
         Vector nor(dim), vu(dim), nh(dim), ni(dim), sv(ND);
         DenseMatrix mq(dim);
         real_t *pd = wd.HostWrite(), *pe = we.HostWrite();
         real_t *pg = wg.HostWrite(), *ph = wh.HostWrite();
         real_t *ps = sh.HostWrite();

         for (int fi = 0; fi < NF; fi++)
         {
            const int face = flist[fi];
            FaceElementTransformations *ftr =
               mesh->GetFaceElementTransformations(face);
            const FiniteElement &fe = *el_fes.GetFE(ftr->Elem1No);
            eo[fi] = E_offsets[face];
            ho[fi] = H_offsets[face];
            dof[fi] = Df_offsets[ftr->Elem1No];

            for (int q = 0; q < NQ; q++)
            {
               const IntegrationPoint &ip = ir.IntPoint(q);
               ftr->SetAllIntPoints(&ip);
               const IntegrationPoint &eip1 = ftr->GetElement1IntPoint();

               if (dim == 1) { nor(0) = 2 * eip1.x - 1.0; }
               else { CalcOrtho(ftr->Jacobian(), nor); }

               fe.CalcPhysShape(*ftr->Elem1, sv);
               for (int i = 0; i < ND; i++)
               {
                  ps[q + NQ * (i + ND * fi)] = sv(i);
               }

               const int o = q + fi * NQ;
               if (kind == HDGBatchKind::Diffusion)
               {
                  real_t un = 0.;
                  if (VectorCoefficient *v = dif->GetVelocity())
                  {
                     v->Eval(vu, *ftr->Elem1, eip1);
                     un = vu * nor;
                  }
                  const real_t un_raw = un;
                  real_t a, b;
                  if (un != 0.)
                  {
                     un /= std::fabs(un);
                     a = 0.5 * dif->GetAlpha() * un;
                     b = dif->GetBeta() * std::fabs(un);
                  }
                  else { a = 0.; b = dif->GetBeta(); }

                  Coefficient *Q = dif->GetCoefficient();
                  MatrixCoefficient *MQ = dif->GetMatrixCoefficient();
                  const real_t wn = ip.weight / ftr->Elem1->Weight();
                  if (!MQ)
                  {
                     ni.Set(Q ? (wn * Q->Eval(*ftr->Elem1, eip1)) : wn, nor);
                  }
                  else
                  {
                     nh.Set(wn, nor);
                     MQ->Eval(mq, *ftr->Elem1, eip1);
                     mq.MultTranspose(nh, ni);
                  }
                  const real_t wq1 = ni * nor;
                  const real_t face_w = ip.weight * nor.Norml2();
                  const real_t w1 = dif->EvalStabilization(
                                       wq1, b + a, un_raw, face_w, 0., 0., *ftr->Elem1);
                  // One weight for all four: the trace block takes side 1's
                  // alone, side 2 being absent.
                  pd[o] += w1;  pe[o] += w1;  pg[o] += w1;  ph[o] += w1;
               }
               else
               {
                  tr_bfi->GetVelocity()->Eval(vu, *ftr->Elem1, eip1);
                  const real_t un = vu * nor;
                  const real_t alpha = tr_bfi->GetAlpha();
                  if (kind == HDGBatchKind::ConvCentered)
                  {
                     const real_t a = alpha * un, b = std::fabs(alpha * un);
                     pd[o] += ip.weight * b;
                     pg[o] += ip.weight * b;
                     pe[o] += ip.weight * (b - a);
                     // b and not 2b: the centred form halves its trace term
                     // at a boundary, where the upwinded one does not.
                     ph[o] += ip.weight * b;
                  }
                  else
                  {
                     const real_t a = 0.5 * alpha * un;
                     const real_t b = tr_bfi->GetBeta() * std::fabs(un);
                     pd[o] += ip.weight * (b + a);
                     pg[o] += ip.weight * (b + a);
                     pe[o] += ip.weight * (b - a);
                     // 2b even here -- "this term must be non-zero at the
                     // boundary for stability reasons, so the advective part
                     // is intentionally dropped", says the per-face routine.
                     ph[o] += ip.weight * 2. * b;
                  }
               }
            }
         }
      }

      const auto d_wd = Reshape(wd.Read(), NQ, NF);
      const auto d_we = Reshape(we.Read(), NQ, NF);
      const auto d_wg = Reshape(wg.Read(), NQ, NF);
      const auto d_wh = Reshape(wh.Read(), NQ, NF);
      const auto d_s = Reshape(sh.Read(), NQ, ND, NF);
      const auto d_tr = Reshape(trs.Read(), NQ, TRD);
      const int *d_eo = eo.Read(), *d_ho = ho.Read(), *d_do = dof.Read();

      real_t *d_E = E_data.ReadWrite();
      real_t *d_G = G_data.ReadWrite();
      real_t *d_H = H_data.ReadWrite();
      real_t *d_D = Df_data.ReadWrite();

      const int nd = ND, trd = TRD, nq = NQ;

      mfem::forall(NF, [=] MFEM_HOST_DEVICE (int f)
      {
         const int e0 = d_eo[f], h0 = d_ho[f], o1 = d_do[f];

         for (int q = 0; q < nq; q++)
         {
            const real_t ad = d_wd(q, f), ae = d_we(q, f);
            const real_t ag = d_wg(q, f), ah = d_wh(q, f);

            for (int i = 0; i < nd; i++)
            {
               const real_t si = d_s(q, i, f);
               // Two boundary faces of one element collide in D -- any
               // corner element has them -- so the atomics are needed here
               // for the same reason as on the interior.
               for (int j = 0; j < nd; j++)
               {
                  AtomicAdd(d_D[o1 + i + nd * j], ad * si * d_s(q, j, f));
               }
               for (int j = 0; j < trd; j++)
               {
                  const real_t t = d_tr(q, j);
                  d_E[e0 + i + nd * j] -= ae * si * t;
                  d_G[e0 + j + trd * i] += ag * t * si;
               }
            }

            for (int i = 0; i < trd; i++)
            {
               const real_t t = ah * d_tr(q, i);
               for (int j = 0; j < trd; j++)
               {
                  d_H[h0 + i + trd * j] -= t * d_tr(q, j);
               }
            }
         }
      });
   }
}

void HDGDiffusionFaceScatterBatched(const FiniteElementSpace &tr_fes,
                                    const FiniteElementSpace &el_fes,
                                    Coefficient *Q, real_t beta,
                                    const HDGStabilization *stab,
                                    const Array<int> &face_list,
                                    const Array<int> &E_offsets,
                                    const Array<int> &H_offsets,
                                    const Array<int> &Df_offsets,
                                    Vector &E_data, Vector &G_data,
                                    Vector &H_data, Vector &Df_data)
{
   MFEM_VERIFY(HDGDiffusionFaceMatricesCanBatch(tr_fes, el_fes),
               "the spaces do not admit the batched face assembly");
   MFEM_VERIFY(!stab || stab->IsConstant(),
               "a state dependent stabilization makes the face term nonlinear");

   Mesh *mesh = el_fes.GetMesh();
   const int NF = face_list.Size();
   if (NF == 0) { return; }

   const FiniteElement *tr_fe0 = tr_fes.GetFaceElement(face_list[0]);
   const int TRD = tr_fe0->GetDof();
   const int ND = el_fes.GetFE(0)->GetDof();

   const int order = 2 * std::max(el_fes.GetFE(0)->GetOrder(),
                                  tr_fe0->GetOrder());
   FaceElementTransformations *ftr0 =
      mesh->GetInteriorFaceTransformations(face_list[0]);
   MFEM_VERIFY(ftr0, "no interior face transformation");
   const IntegrationRule &ir = IntRules.Get(ftr0->GetGeometryType(), order);
   const int NQ = ir.GetNPoints();

   // Same host precompute as the dense form; see the note there on what it
   // would take to remove it.
   Vector w1(NQ * NF), w2(NQ * NF);
   Vector sh1(NQ * ND * NF), sh2(NQ * ND * NF);
   Vector trs(NQ * TRD);
   Array<int> eo(NF), ho(NF), d1(NF), d2(NF);
   w1.UseDevice(true);
   w2.UseDevice(true);
   sh1.UseDevice(true);
   sh2.UseDevice(true);
   trs.UseDevice(true);

   {
      Vector t(TRD);
      for (int q = 0; q < NQ; q++)
      {
         tr_fe0->CalcShape(ir.IntPoint(q), t);
         for (int i = 0; i < TRD; i++) { trs(q + i * NQ) = t(i); }
      }
   }

   {
      const int dim = mesh->Dimension();
      Vector nor(dim), s1(ND), s2(ND);
      real_t *pw1 = w1.HostWrite(), *pw2 = w2.HostWrite();
      real_t *ps1 = sh1.HostWrite(), *ps2 = sh2.HostWrite();

      for (int fi = 0; fi < NF; fi++)
      {
         const int face = face_list[fi];
         FaceElementTransformations *ftr =
            mesh->GetInteriorFaceTransformations(face);
         const FiniteElement &e1 = *el_fes.GetFE(ftr->Elem1No);
         const FiniteElement &e2 = *el_fes.GetFE(ftr->Elem2No);
         eo[fi] = E_offsets[face];
         ho[fi] = H_offsets[face];
         d1[fi] = Df_offsets[ftr->Elem1No];
         d2[fi] = Df_offsets[ftr->Elem2No];

         for (int q = 0; q < NQ; q++)
         {
            const IntegrationPoint &ip = ir.IntPoint(q);
            ftr->SetAllIntPoints(&ip);
            const IntegrationPoint &eip1 = ftr->GetElement1IntPoint();
            const IntegrationPoint &eip2 = ftr->GetElement2IntPoint();

            if (dim == 1) { nor(0) = 2 * eip1.x - 1.0; }
            else { CalcOrtho(ftr->Jacobian(), nor); }

            e1.CalcPhysShape(*ftr->Elem1, s1);
            e2.CalcPhysShape(*ftr->Elem2, s2);

            const real_t nn = nor * nor;
            const real_t face_w = ip.weight * nor.Norml2();
            real_t wn1 = ip.weight / ftr->Elem1->Weight();
            if (Q) { wn1 *= Q->Eval(*ftr->Elem1, eip1); }
            real_t wn2 = ip.weight / ftr->Elem2->Weight();
            if (Q) { wn2 *= Q->Eval(*ftr->Elem2, eip2); }
            const real_t wq1 = wn1 * nn, wq2 = wn2 * nn;
            pw1[q + fi * NQ] = stab
                               ? face_w * stab->Eval((face_w != 0.) ? (wq1 * beta / face_w) : 0.,
                                                     0., 0., 0., *ftr->Elem1)
                               : wq1 * beta;
            pw2[q + fi * NQ] = stab
                               ? face_w * stab->Eval((face_w != 0.) ? (wq2 * beta / face_w) : 0.,
                                                     0., 0., 0., *ftr->Elem2)
                               : wq2 * beta;
            for (int i = 0; i < ND; i++)
            {
               ps1[q + NQ * (i + ND * fi)] = s1(i);
               ps2[q + NQ * (i + ND * fi)] = s2(i);
            }
         }
      }
   }

   const auto d_w1 = Reshape(w1.Read(), NQ, NF);
   const auto d_w2 = Reshape(w2.Read(), NQ, NF);
   const auto d_s1 = Reshape(sh1.Read(), NQ, ND, NF);
   const auto d_s2 = Reshape(sh2.Read(), NQ, ND, NF);
   const auto d_tr = Reshape(trs.Read(), NQ, TRD);
   const int *d_eo = eo.Read(), *d_ho = ho.Read();
   const int *d_d1 = d1.Read(), *d_d2 = d2.Read();

   real_t *d_E = E_data.ReadWrite();
   real_t *d_G = G_data.ReadWrite();
   real_t *d_H = H_data.ReadWrite();
   real_t *d_D = Df_data.ReadWrite();

   const int nd = ND, trd = TRD, nq = NQ;

   mfem::forall(NF, [=] MFEM_HOST_DEVICE (int f)
   {
      const int e0 = d_eo[f], h0 = d_ho[f];
      const int o1 = d_d1[f], o2 = d_d2[f];
      const int e2off = e0 + trd * nd;      // side 2 follows side 1

      // E, G and H belong to this face alone: overwrite, no atomics.
      for (int j = 0; j < trd; j++)
         for (int i = 0; i < nd; i++)
         {
            d_E[e0 + i + nd * j] = 0.0;
            d_E[e2off + i + nd * j] = 0.0;
            d_G[e0 + j + trd * i] = 0.0;
            d_G[e2off + j + trd * i] = 0.0;
         }
      for (int j = 0; j < trd; j++)
         for (int i = 0; i < trd; i++)
         {
            d_H[h0 + i + trd * j] = 0.0;
         }

      for (int q = 0; q < nq; q++)
      {
         const real_t a1 = d_w1(q, f), a2 = d_w2(q, f);

         for (int i = 0; i < nd; i++)
         {
            const real_t b1 = a1 * d_s1(q, i, f);
            const real_t b2 = a2 * d_s2(q, i, f);

            // D accumulates per ELEMENT, and two faces of one element
            // collide, so it is the one block that needs atomics.
            for (int j = 0; j < nd; j++)
            {
               AtomicAdd(d_D[o1 + i + nd * j], b1 * d_s1(q, j, f));
               AtomicAdd(d_D[o2 + i + nd * j], b2 * d_s2(q, j, f));
            }
            for (int j = 0; j < trd; j++)
            {
               const real_t e1v = b1 * d_tr(q, j);
               const real_t e2v = b2 * d_tr(q, j);
               d_E[e0 + i + nd * j]    -= e1v;
               d_E[e2off + i + nd * j] -= e2v;
               d_G[e0 + j + trd * i]    += e1v;
               d_G[e2off + j + trd * i] += e2v;
            }
         }

         const real_t aa = a1 + a2;
         for (int i = 0; i < trd; i++)
         {
            const real_t t = aa * d_tr(q, i);
            for (int j = 0; j < trd; j++)
            {
               d_H[h0 + i + trd * j] -= t * d_tr(q, j);
            }
         }
      }
   });
}

}
