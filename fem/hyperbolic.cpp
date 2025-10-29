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

// Implementation of hyperbolic conservation laws

#include "hyperbolic.hpp"
#include "nonlinearform.hpp"
#include "pnonlinearform.hpp"

namespace mfem
{

HyperbolicFormIntegrator::HyperbolicFormIntegrator(
   const NumericalFlux &numFlux,
   const int IntOrderOffset,
   real_t sign)
   : NonlinearFormIntegrator(),
     numFlux(numFlux),
     fluxFunction(numFlux.GetFluxFunction()),
     IntOrderOffset(IntOrderOffset),
     sign(sign),
     num_equations(fluxFunction.num_equations)
{
#ifndef MFEM_THREAD_SAFE
   state.SetSize(num_equations);
   flux.SetSize(num_equations, fluxFunction.dim);
   state1.SetSize(num_equations);
   state2.SetSize(num_equations);
   fluxN.SetSize(num_equations);
   JDotN.SetSize(num_equations);
   nor.SetSize(fluxFunction.dim);
#endif
   ResetMaxCharSpeed();
}

void HyperbolicFormIntegrator::AssembleElementVector(const FiniteElement &el,
                                                     ElementTransformation &Tr,
                                                     const Vector &elfun,
                                                     Vector &elvect)
{
   // current element's the number of degrees of freedom
   // does not consider the number of equations
   const int dof = el.GetDof();

#ifdef MFEM_THREAD_SAFE
   // Local storage for element integration

   // shape function value at an integration point
   Vector shape(dof);
   // derivative of shape function at an integration point
   DenseMatrix dshape(dof, Tr.GetSpaceDim());
   // state value at an integration point
   Vector state(num_equations);
   // flux value at an integration point
   DenseMatrix flux(num_equations, el.GetDim());
#else
   // resize shape and gradient shape storage
   shape.SetSize(dof);
   dshape.SetSize(dof, Tr.GetSpaceDim());
#endif

   // setDegree-up output vector
   elvect.SetSize(dof * num_equations);
   elvect = 0.0;

   // make state variable and output dual vector matrix form.
   const DenseMatrix elfun_mat(elfun.GetData(), dof, num_equations);
   DenseMatrix elvect_mat(elvect.GetData(), dof, num_equations);

   // obtain integration rule. If integration is rule is given, then use it.
   // Otherwise, get (2*p + IntOrderOffset) order integration rule
   const IntegrationRule *ir = IntRule;
   if (!ir)
   {
      const int order = el.GetOrder()*2 + IntOrderOffset;
      ir = &IntRules.Get(Tr.GetGeometryType(), order);
   }

   // loop over integration points
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      Tr.SetIntPoint(&ip);

      el.CalcShape(ip, shape);
      el.CalcPhysDShape(Tr, dshape);
      // compute current state value with given shape function values
      elfun_mat.MultTranspose(shape, state);
      // compute F(u,x) and point maximum characteristic speed

      const real_t mcs = fluxFunction.ComputeFlux(state, Tr, flux);
      // update maximum characteristic speed
      max_char_speed = std::max(mcs, max_char_speed);
      // integrate (F(u,x), grad v)
      AddMult_a_ABt(ip.weight * Tr.Weight() * sign, dshape, flux, elvect_mat);
   }
}

void HyperbolicFormIntegrator::AssembleElementGrad(
   const FiniteElement &el, ElementTransformation &Tr, const Vector &elfun,
   DenseMatrix &grad)
{
   // current element's the number of degrees of freedom
   // does not consider the number of equations
   const int dof = el.GetDof();

#ifdef MFEM_THREAD_SAFE
   // Local storage for element integration

   // shape function value at an integration point
   Vector shape(dof);
   // derivative of shape function at an integration point
   DenseMatrix dshape(dof, Tr.GetSpaceDim());
   // state value at an integration point
   Vector state(num_equations);
   // Jacobian value at an integration point
   DenseTensor J(num_equations, num_equations, fluxFunction.dim);
#else
   // resize shape, gradient shape and Jacobian storage
   shape.SetSize(dof);
   dshape.SetSize(dof, Tr.GetSpaceDim());
   J.SetSize(num_equations, num_equations, fluxFunction.dim);
#endif

   // setup output gradient matrix
   grad.SetSize(dof * num_equations);
   grad = 0.0;

   // make state variable and output dual vector matrix form.
   const DenseMatrix elfun_mat(elfun.GetData(), dof, num_equations);
   //DenseMatrix elvect_mat(elvect.GetData(), dof, num_equations);

   // obtain integration rule. If integration is rule is given, then use it.
   // Otherwise, get (2*p + IntOrderOffset) order integration rule
   const IntegrationRule *ir = IntRule;
   if (!ir)
   {
      const int order = el.GetOrder()*2 + IntOrderOffset;
      ir = &IntRules.Get(Tr.GetGeometryType(), order);
   }

   // loop over integration points
   for (int q = 0; q < ir->GetNPoints(); q++)
   {
      const IntegrationPoint &ip = ir->IntPoint(q);
      Tr.SetIntPoint(&ip);

      el.CalcShape(ip, shape);
      el.CalcPhysDShape(Tr, dshape);
      // compute current state value with given shape function values
      elfun_mat.MultTranspose(shape, state);

      // compute J(u,x)
      fluxFunction.ComputeFluxJacobian(state, Tr, J);

      // integrate (J(u,x), grad v)
      const real_t w = ip.weight * Tr.Weight() * sign;
      for (int di = 0; di < num_equations; di++)
         for (int dj = 0; dj < num_equations; dj++)
            for (int i = 0; i < dof; i++)
               for (int j = 0; j < dof; j++)
                  for (int d = 0; d < fluxFunction.dim; d++)
                  {
                     grad(di*dof+i, dj*dof+j) += w * dshape(i,d) * shape(j) * J(di,dj,d);
                  }
   }
}

void HyperbolicFormIntegrator::AssembleFaceVector(
   const FiniteElement &el1, const FiniteElement &el2,
   FaceElementTransformations &Tr, const Vector &elfun, Vector &elvect)
{
   // current elements' the number of degrees of freedom
   // does not consider the number of equations
   const int dof1 = el1.GetDof();
   const int dof2 = (Tr.Elem2No >= 0)?(el2.GetDof()):(0);

#ifdef MFEM_THREAD_SAFE
   // Local storage for element integration

   // shape function value at an integration point - first elem
   Vector shape1(dof1);
   // shape function value at an integration point - second elem
   Vector shape2(dof2);
   // normal vector (usually not a unit vector)
   Vector nor(Tr.GetSpaceDim());
   // state value at an integration point - first elem
   Vector state1(num_equations);
   // state value at an integration point - second elem
   Vector state2(num_equations);
   // hat(F)(u,x)
   Vector fluxN(num_equations);
#else
   shape1.SetSize(dof1);
   shape2.SetSize(dof2);
#endif

   elvect.SetSize((dof1 + dof2) * num_equations);
   elvect = 0.0;

   const DenseMatrix elfun1_mat(elfun.GetData(), dof1, num_equations);
   const DenseMatrix elfun2_mat(elfun.GetData() + dof1 * num_equations, dof2,
                                num_equations);

   DenseMatrix elvect1_mat(elvect.GetData(), dof1, num_equations);
   DenseMatrix elvect2_mat(elvect.GetData() + dof1 * num_equations, dof2,
                           num_equations);

   // Obtain integration rule. If integration is rule is given, then use it.
   // Otherwise, get (2*p + IntOrderOffset) order integration rule
   const IntegrationRule *ir = IntRule;
   if (!ir)
   {
      const int max_el_order = dof2 ? std::max(el1.GetOrder(),
                                               el2.GetOrder()) : el1.GetOrder();
      const int order = 2*max_el_order + IntOrderOffset;
      ir = &IntRules.Get(Tr.GetGeometryType(), order);
   }
   // loop over integration points
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

      Tr.SetAllIntPoints(&ip); // set face and element int. points

      // Calculate basis functions on both elements at the face
      el1.CalcShape(Tr.GetElement1IntPoint(), shape1);

      // Interpolate elfun at the point
      elfun1_mat.MultTranspose(shape1, state1);

      if (dof2)
      {
         // Calculate basis functions on both elements at the face
         el2.CalcShape(Tr.GetElement2IntPoint(), shape2);
         // Interpolate elfun at the point
         elfun2_mat.MultTranspose(shape2, state2);
      }

      // Get the normal vector and the flux on the face
      if (nor.Size() == 1)  // if 1D, use 1 or -1.
      {
         nor(0) = 2*Tr.GetElement1IntPoint().x - 1.;
      }
      else
      {
         CalcOrtho(Tr.Jacobian(), nor);
      }
      // Compute F(u+, x) and F(u-, x) with maximum characteristic speed
      // Compute hat(F) using evaluated quantities
      const real_t speed = (dof2) ? numFlux.Eval(state1, state2, nor, Tr, fluxN):
                           fluxFunction.ComputeFluxDotN(state1, nor, Tr, fluxN);

      // Update the global max char speed
      max_char_speed = std::max(speed, max_char_speed);

      // pre-multiply integration weight to flux
      AddMult_a_VWt(-ip.weight*sign, shape1, fluxN, elvect1_mat);
      if (dof2)
      {
         AddMult_a_VWt(+ip.weight*sign, shape2, fluxN, elvect2_mat);
      }
   }
}

void HyperbolicFormIntegrator::AssembleFaceGrad(
   const FiniteElement &el1, const FiniteElement &el2,
   FaceElementTransformations &Tr, const Vector &elfun, DenseMatrix &elmat)
{
   // current elements' the number of degrees of freedom
   // does not consider the number of equations
   const int dof1 = el1.GetDof();
   const int dof2 = (Tr.Elem2No >= 0)?(el2.GetDof()):(0);

#ifdef MFEM_THREAD_SAFE
   // Local storage for element integration

   // shape function value at an integration point - first elem
   Vector shape1(dof1);
   // shape function value at an integration point - second elem
   Vector shape2(dof2);
   // normal vector (usually not a unit vector)
   Vector nor(Tr.GetSpaceDim());
   // state value at an integration point - first elem
   Vector state1(num_equations);
   // state value at an integration point - second elem
   Vector state2(num_equations);
   // hat(J)(u,x)
   DenseMatrix JDotN(num_equations);
#else
   shape1.SetSize(dof1);
   shape2.SetSize(dof2);
#endif

   elmat.SetSize((dof1 + dof2) * num_equations);
   elmat = 0.0;

   const DenseMatrix elfun1_mat(elfun.GetData(), dof1, num_equations);
   const DenseMatrix elfun2_mat(elfun.GetData() + dof1 * num_equations, dof2,
                                num_equations);

   // Obtain integration rule. If integration is rule is given, then use it.
   // Otherwise, get (2*p + IntOrderOffset) order integration rule
   const IntegrationRule *ir = IntRule;
   if (!ir)
   {
      const int max_el_order = dof2 ? std::max(el1.GetOrder(),
                                               el2.GetOrder()) : el1.GetOrder();
      const int order = 2*max_el_order + IntOrderOffset;
      ir = &IntRules.Get(Tr.GetGeometryType(), order);
   }
   // loop over integration points
   for (int q = 0; q < ir->GetNPoints(); q++)
   {
      const IntegrationPoint &ip = ir->IntPoint(q);

      Tr.SetAllIntPoints(&ip); // set face and element int. points

      // Calculate basis functions of the first element at the face
      el1.CalcShape(Tr.GetElement1IntPoint(), shape1);

      // Interpolate elfun at the point
      elfun1_mat.MultTranspose(shape1, state1);

      if (dof2)
      {
         // Calculate basis function of the second element at the face
         el2.CalcShape(Tr.GetElement2IntPoint(), shape2);

         // Interpolate elfun at the point
         elfun2_mat.MultTranspose(shape2, state2);
      }

      // Get the normal vector and the flux on the face
      if (nor.Size() == 1)  // if 1D, use 1 or -1.
      {
         nor(0) = 2*Tr.GetElement1IntPoint().x - 1.;
      }
      else
      {
         CalcOrtho(Tr.Jacobian(), nor);
      }

      // Trial side 1

      // Compute hat(J) using evaluated quantities
      if (dof2)
      {
         numFlux.Grad(1, state1, state2, nor, Tr, JDotN);
      }
      else
      {
         fluxFunction.ComputeFluxJacobianDotN(state1, nor, Tr, JDotN);
      }

      const int ioff = fluxFunction.num_equations * dof1;

      for (int di = 0; di < fluxFunction.num_equations; di++)
         for (int dj = 0; dj < fluxFunction.num_equations; dj++)
         {
            // pre-multiply integration weight to Jacobian
            const real_t w = -ip.weight * sign * JDotN(di,dj);
            for (int j = 0; j < dof1; j++)
            {
               // Test side 1
               for (int i = 0; i < dof1; i++)
               {
                  elmat(i+dof1*di, j+dof1*dj) += w * shape1(i) * shape1(j);
               }

               // Test side 2
               for (int i = 0; i < dof2; i++)
               {
                  elmat(ioff+i+dof2*di, j+dof1*dj) -= w * shape2(i) * shape1(j);
               }
            }
         }

      if (dof2)
      {
         // Trial side 2

         // Compute hat(J) using evaluated quantities
         numFlux.Grad(2, state1, state2, nor, Tr, JDotN);

         const int joff = ioff;

         for (int di = 0; di < fluxFunction.num_equations; di++)
            for (int dj = 0; dj < fluxFunction.num_equations; dj++)
            {
               // pre-multiply integration weight to Jacobian
               const real_t w = +ip.weight * sign * JDotN(di,dj);
               for (int j = 0; j < dof2; j++)
               {
                  // Test side 1
                  for (int i = 0; i < dof1; i++)
                  {
                     elmat(i+dof1*di, joff+j+dof2*dj) += w * shape1(i) * shape2(j);
                  }

                  // Test side 2
                  for (int i = 0; i < dof2; i++)
                  {
                     elmat(ioff+i+dof2*di, joff+j+dof2*dj) -= w * shape2(i) * shape2(j);
                  }
               }
            }
      }
   }
}

void HyperbolicFormIntegrator::AssembleHDGFaceVector(
   int type, const FiniteElement &trace_face_fe, const FiniteElement &fe,
   FaceElementTransformations &Tr, const Vector &trfun, const Vector &elfun,
   Vector &elvect)
{
   MFEM_ASSERT((type & HDGFaceType::ELEM && type & HDGFaceType::TRACE) ||
               (type & HDGFaceType::CONSTR && type & HDGFaceType::FACE),
               "Not allowed combination of types");

   const int dof_el = fe.GetDof();
   const int dof_tr = trace_face_fe.GetDof();

   const int dof_dual_el = (type & (HDGFaceType::ELEM | HDGFaceType::TRACE))?
                           (dof_el):(0);
   const int dof_dual_tr = (type & (HDGFaceType::CONSTR | HDGFaceType::FACE))?
                           (dof_tr):(0);

#ifdef MFEM_THREAD_SAFE
   // Local storage for element integration

   // normal vector (usually not a unit vector)
   Vector nor(Tr.GetSpaceDim());
   // shape function value at an integration point - elem
   Vector shape_el(dof_el);
   // shape function value at an integration point - trace
   Vector shape_tr(dof_tr);
   // state value at an integration point - elem
   Vector state_el(num_equations);
   // state value at an integration point - trace
   Vector state_tr(num_equations);
   // hat(F)(u,x)
   Vector fluxN(num_equations);
#else
   Vector &shape_el(shape1);
   Vector &shape_tr(shape2);
   Vector &state_el(state1);
   Vector &state_tr(state2);
   shape_el.SetSize(dof_el);
   shape_tr.SetSize(dof_tr);
#endif

   elvect.SetSize((dof_dual_el + dof_dual_tr) * num_equations);
   elvect = 0.0;

   const DenseMatrix elfun_mat(elfun.GetData(), dof_el, num_equations);
   const DenseMatrix trfun_mat(trfun.GetData(), dof_tr, num_equations);

   DenseMatrix elvect_mat(elvect.GetData(), dof_dual_el, num_equations);
   DenseMatrix trvect_mat(elvect.GetData() + dof_dual_el * num_equations,
                          dof_dual_tr, num_equations);

   const IntegrationRule *ir = IntRule;
   if (!ir)
   {
      const int order = 2*std::max(fe.GetOrder(),
                                   trace_face_fe.GetOrder()) + IntOrderOffset;
      ir = &IntRules.Get(Tr.GetGeometryType(), order);
   }

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

      Tr.SetAllIntPoints(&ip); // set face and element int. points
      const IntegrationPoint &eip = (type & 1)?(Tr.GetElement2IntPoint()):
                                    (Tr.GetElement1IntPoint());

      // Calculate basis functions on both elements at the face
      fe.CalcShape(eip, shape_el);
      trace_face_fe.CalcShape(ip, shape_tr);

      // Interpolate elfun and trfun at the point
      elfun_mat.MultTranspose(shape_el, state_el);
      trfun_mat.MultTranspose(shape_tr, state_tr);

      // Get the normal vector and the flux on the face
      if (nor.Size() == 1)  // if 1D, use 1 or -1.
      {
         // This assume the 1D integration point is in (0,1). This may not work
         // if this changes.
         nor(0) = (Tr.GetElement1IntPoint().x - 0.5) * 2.0;
      }
      else
      {
         CalcOrtho(Tr.Jacobian(), nor);
      }
      if (type & 1) { nor.Neg(); }

      // Compute average flux hat(F)(û,u) with maximum characteristic speed
      numFlux.Average(state_tr, state_el, nor, Tr, fluxN);

      // pre-multiply integration weight to flux
      if (type & (HDGFaceType::ELEM | HDGFaceType::TRACE))
      {
         AddMult_a_VWt(-ip.weight*sign, shape_el, fluxN, elvect_mat);
      }
      if (type & (HDGFaceType::CONSTR | HDGFaceType::FACE))
      {
         AddMult_a_VWt(-ip.weight*sign, shape_tr, fluxN, trvect_mat);
      }
   }
}

void HyperbolicFormIntegrator::AssembleHDGFaceGrad(
   int type, const FiniteElement &trace_face_fe, const FiniteElement &fe,
   FaceElementTransformations &Tr, const Vector &trfun, const Vector &elfun,
   DenseMatrix &elmat)
{
   const int dof_el = fe.GetDof();
   const int dof_tr = trace_face_fe.GetDof();

   const int dof_prim_el = (type & (HDGFaceType::ELEM | HDGFaceType::CONSTR))?
                           (dof_el):(0);
   const int dof_prim_tr = (type & (HDGFaceType::TRACE | HDGFaceType::FACE))?
                           (dof_tr):(0);
   const int dof_dual_el = (type & (HDGFaceType::ELEM | HDGFaceType::TRACE))?
                           (dof_el):(0);
   const int dof_dual_tr = (type & (HDGFaceType::CONSTR | HDGFaceType::FACE))?
                           (dof_tr):(0);
   const int dof_prim = dof_prim_el + dof_prim_tr;
   const int dof_dual = dof_dual_el + dof_dual_tr;

#ifdef MFEM_THREAD_SAFE
   // Local storage for element integration

   // normal vector (usually not a unit vector)
   Vector nor(Tr.GetSpaceDim());
   // shape function value at an integration point - elem
   Vector shape_el(dof_el);
   // shape function value at an integration point - trace
   Vector shape_tr(dof_tr);
   // state value at an integration point - elem
   Vector state_el(num_equations);
   // state value at an integration point - trace
   Vector state_tr(num_equations);
   // J(F)(u,x)
   DenseMatrix JDotN(num_equations);
#else
   Vector &shape_el(shape1);
   Vector &shape_tr(shape2);
   Vector &state_el(state1);
   Vector &state_tr(state2);
   shape_el.SetSize(dof_el);
   shape_tr.SetSize(dof_tr);
#endif

   elmat.SetSize(dof_dual * num_equations,
                 dof_prim * num_equations);
   elmat = 0.0;

   const DenseMatrix elfun_mat(elfun.GetData(), dof_el, num_equations);
   const DenseMatrix trfun_mat(trfun.GetData(), dof_tr, num_equations);

   const IntegrationRule *ir = IntRule;
   if (!ir)
   {
      const int order = 2*std::max(fe.GetOrder(),
                                   trace_face_fe.GetOrder()) + IntOrderOffset;
      ir = &IntRules.Get(Tr.GetGeometryType(), order);
   }

   for (int p = 0; p < ir->GetNPoints(); p++)
   {
      const IntegrationPoint &ip = ir->IntPoint(p);

      Tr.SetAllIntPoints(&ip); // set face and element int. points
      const IntegrationPoint &eip = (type & 1)?(Tr.GetElement2IntPoint()):
                                    (Tr.GetElement1IntPoint());

      // Calculate basis functions on both elements at the face
      fe.CalcShape(eip, shape_el);
      trace_face_fe.CalcShape(ip, shape_tr);

      // Interpolate elfun and trfun at the point
      elfun_mat.MultTranspose(shape_el, state_el);
      trfun_mat.MultTranspose(shape_tr, state_tr);

      // Get the normal vector and the flux on the face
      if (nor.Size() == 1)  // if 1D, use 1 or -1.
      {
         // This assume the 1D integration point is in (0,1). This may not work
         // if this changes.
         nor(0) = (Tr.GetElement1IntPoint().x - 0.5) * 2.0;
      }
      else
      {
         CalcOrtho(Tr.Jacobian(), nor);
      }
      if (type & 1) { nor.Neg(); }

      // Compute average J(û, u)
      int joff = 0;
      if (type & (HDGFaceType::ELEM | HDGFaceType::CONSTR))
      {
         numFlux.AverageGrad(2, state_tr, state_el, nor, Tr, JDotN);

         // pre-multiply integration weight to Jacobians
         const real_t w = -ip.weight*sign;
         int ioff = 0;
         if (type & HDGFaceType::ELEM)
         {
            for (int di = 0; di < num_equations; di++)
               for (int dj = 0; dj < num_equations; dj++)
                  for (int i = 0; i < dof_el; i++)
                     for (int j = 0; j < dof_el; j++)
                     {
                        elmat(di*dof_dual+ioff+i, dj*dof_prim+joff+j) +=
                           w * JDotN(di,dj) * shape_el(i) * shape_el(j);
                     }
            ioff += dof_el;
         }
         if (type & HDGFaceType::CONSTR)
         {
            for (int di = 0; di < num_equations; di++)
               for (int dj = 0; dj < num_equations; dj++)
                  for (int i = 0; i < dof_tr; i++)
                     for (int j = 0; j < dof_el; j++)
                     {
                        elmat(di*dof_dual+ioff+i, dj*dof_prim+joff+j) +=
                           w * JDotN(di,dj) * shape_tr(i) * shape_el(j);
                     }
         }
         joff += dof_el;
      }
      if (type & (HDGFaceType::TRACE | HDGFaceType::FACE))
      {
         numFlux.AverageGrad(1, state_tr, state_el, nor, Tr, JDotN);

         // pre-multiply integration weight to Jacobians
         const real_t w = -ip.weight*sign;
         int ioff = 0;
         if (type & HDGFaceType::TRACE)
         {
            for (int di = 0; di < num_equations; di++)
               for (int dj = 0; dj < num_equations; dj++)
                  for (int i = 0; i < dof_el; i++)
                     for (int j = 0; j < dof_tr; j++)
                     {
                        elmat(di*dof_dual+ioff+i, dj*dof_prim+joff+j) +=
                           w * JDotN(di,dj) * shape_el(i) * shape_tr(j);
                     }
            ioff += dof_el;
         }
         if (type & HDGFaceType::FACE)
         {
            for (int di = 0; di < num_equations; di++)
               for (int dj = 0; dj < num_equations; dj++)
                  for (int i = 0; i < dof_tr; i++)
                     for (int j = 0; j < dof_tr; j++)
                     {
                        elmat(di*dof_dual+ioff+i, dj*dof_prim+joff+j) +=
                           w * JDotN(di,dj) * shape_tr(i) * shape_tr(j);
                     }
         }
      }
   }
}

BdrHyperbolicDirichletIntegrator::BdrHyperbolicDirichletIntegrator(
   const NumericalFlux &numFlux,
   VectorCoefficient &bdrState,
   const int IntOrderOffset,
   real_t sign)
   : NonlinearFormIntegrator(),
     numFlux(numFlux),
     fluxFunction(numFlux.GetFluxFunction()),
     u_vcoeff(bdrState),
     IntOrderOffset(IntOrderOffset),
     sign(sign),
     num_equations(fluxFunction.num_equations)
{
   MFEM_VERIFY(fluxFunction.num_equations == bdrState.GetVDim(),
               "Flux function does not match the vector dimension of the coefficient!");
#ifndef MFEM_THREAD_SAFE
   state_in.SetSize(num_equations);
   state_out.SetSize(num_equations);
   fluxN.SetSize(num_equations);
   JDotN.SetSize(num_equations);
   nor.SetSize(fluxFunction.dim);
#endif
   ResetMaxCharSpeed();
}

void BdrHyperbolicDirichletIntegrator::AssembleFaceVector(
   const FiniteElement &el, const FiniteElement &,
   FaceElementTransformations &Tr, const Vector &elfun, Vector &elvect)
{
   MFEM_ASSERT(Tr.Elem2No < 0, "Not a boundary face!");

   // current elements' the number of degrees of freedom
   // does not consider the number of equations
   const int dof = el.GetDof();

#ifdef MFEM_THREAD_SAFE
   // Local storage for element integration

   // shape function value at an integration point
   Vector shape(dof);
   // normal vector (usually not a unit vector)
   Vector nor(Tr.GetSpaceDim());
   // state value at an integration point - interior
   Vector state_in(num_equations);
   // state value at an integration point - boundary
   Vector state_out(num_equations);
   // hat(F)(u,x)
   Vector fluxN(num_equations);
#else
   shape.SetSize(dof);
#endif

   elvect.SetSize(dof * num_equations);
   elvect = 0.0;

   const DenseMatrix elfun_mat(elfun.GetData(), dof, num_equations);

   DenseMatrix elvect_mat(elvect.GetData(), dof, num_equations);

   // Obtain integration rule. If integration is rule is given, then use it.
   // Otherwise, get (2*p + IntOrderOffset) order integration rule
   const IntegrationRule *ir = IntRule;
   if (!ir)
   {
      const int order = 2*el.GetOrder() + IntOrderOffset;
      ir = &IntRules.Get(Tr.GetGeometryType(), order);
   }
   // loop over integration points
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

      Tr.SetAllIntPoints(&ip); // set face and element int. points

      // Calculate basis functions at the face
      el.CalcShape(Tr.GetElement1IntPoint(), shape);

      // Interpolate elfun at the point
      elfun_mat.MultTranspose(shape, state_in);

      // Evaluate boundary state at the point
      u_vcoeff.Eval(state_out, Tr, ip);

      // Get the normal vector and the flux on the face
      if (nor.Size() == 1)  // if 1D, use 1 or -1.
      {
         nor(0) = 2*Tr.GetElement1IntPoint().x - 1.;
      }
      else
      {
         CalcOrtho(Tr.Jacobian(), nor);
      }
      // Compute F(u+, x) and F(u_b, x) with maximum characteristic speed
      // Compute hat(F) using evaluated quantities
      const real_t speed = numFlux.Eval(state_in, state_out, nor, Tr, fluxN);

      // Update the global max char speed
      max_char_speed = std::max(speed, max_char_speed);

      // pre-multiply integration weight to flux
      AddMult_a_VWt(-ip.weight*sign, shape, fluxN, elvect_mat);
   }
}

void BdrHyperbolicDirichletIntegrator::AssembleFaceGrad(
   const FiniteElement &el, const FiniteElement &,
   FaceElementTransformations &Tr, const Vector &elfun, DenseMatrix &elmat)
{
   // current elements' the number of degrees of freedom
   // does not consider the number of equations
   const int dof = el.GetDof();

#ifdef MFEM_THREAD_SAFE
   // Local storage for element integration

   // shape function value at an integration point
   Vector shape(dof);
   // normal vector (usually not a unit vector)
   Vector nor(Tr.GetSpaceDim());
   // state value at an integration point - interior
   Vector state_in(num_equations);
   // state value at an integration point - boundary
   Vector state_out(num_equations);
   // hat(J)(u,x)
   DenseMatrix JDotN(num_equations);
#else
   shape.SetSize(dof);
#endif

   elmat.SetSize(dof * num_equations);
   elmat = 0.0;

   const DenseMatrix elfun_mat(elfun.GetData(), dof, num_equations);

   // Obtain integration rule. If integration is rule is given, then use it.
   // Otherwise, get (2*p + IntOrderOffset) order integration rule
   const IntegrationRule *ir = IntRule;
   if (!ir)
   {
      const int order = 2*el.GetOrder() + IntOrderOffset;
      ir = &IntRules.Get(Tr.GetGeometryType(), order);
   }
   // loop over integration points
   for (int q = 0; q < ir->GetNPoints(); q++)
   {
      const IntegrationPoint &ip = ir->IntPoint(q);

      Tr.SetAllIntPoints(&ip); // set face and element int. points

      // Calculate basis functions at the face
      el.CalcShape(Tr.GetElement1IntPoint(), shape);

      // Interpolate elfun at the point
      elfun_mat.MultTranspose(shape, state_in);

      // Evaluate boundary state at the point
      u_vcoeff.Eval(state_out, Tr, ip);

      // Get the normal vector and the flux on the face
      if (nor.Size() == 1)  // if 1D, use 1 or -1.
      {
         nor(0) = 2*Tr.GetElement1IntPoint().x - 1.;
      }
      else
      {
         CalcOrtho(Tr.Jacobian(), nor);
      }

      // Compute hat(J) using evaluated quantities
      numFlux.Grad(1, state_in, state_out, nor, Tr, JDotN);

      for (int di = 0; di < fluxFunction.num_equations; di++)
         for (int dj = 0; dj < fluxFunction.num_equations; dj++)
         {
            // pre-multiply integration weight to Jacobian
            const real_t w = -ip.weight * sign * JDotN(di,dj);
            for (int j = 0; j < dof; j++)
               for (int i = 0; i < dof; i++)
               {
                  elmat(i+dof*di, j+dof*dj) += w * shape(i) * shape(j);
               }
         }
   }
}

BoundaryHyperbolicFlowIntegrator::BoundaryHyperbolicFlowIntegrator(
   const FluxFunction &flux, VectorCoefficient &u, real_t alpha_, real_t beta_,
   const int IntOrderOffset_)
   : fluxFunction(flux), u_vcoeff(u), alpha(alpha_), beta(beta_),
     IntOrderOffset(IntOrderOffset_)
{
   MFEM_VERIFY(fluxFunction.num_equations == u_vcoeff.GetVDim(),
               "Flux function does not match the vector dimension of the coefficient!");
#ifndef MFEM_THREAD_SAFE
   state.SetSize(fluxFunction.num_equations);
   nor.SetSize(fluxFunction.dim);
   fluxN.SetSize(fluxFunction.num_equations);
#endif
   ResetMaxCharSpeed();
}

void BoundaryHyperbolicFlowIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)
{
   mfem_error("BoundaryHyperbolicFlowIntegrator::AssembleRHSElementVect\n"
              "  is not implemented as boundary integrator!\n"
              "  Use LinearForm::AddBdrFaceIntegrator instead of\n"
              "  LinearForm::AddBoundaryIntegrator.");
}

void BoundaryHyperbolicFlowIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, FaceElementTransformations &Tr, Vector &elvect)
{
   // current elements' the number of degrees of freedom
   // does not consider the number of equations
   const int dof = el.GetDof();

#ifdef MFEM_THREAD_SAFE
   // Local storage for element integration

   // shape function value at an integration point
   Vector shape(dof);
   // state value at an integration point
   Vector state(fluxFunction.num_equations);
   // normal vector (usually not a unit vector)
   Vector nor(Tr.GetSpaceDim());
   // hat(F)(u,x)
   Vector fluxN(fluxFunction.num_equations);
#else
   shape.SetSize(dof);
#endif

   elvect.SetSize(dof * fluxFunction.num_equations);
   elvect = 0.0;

   DenseMatrix elvect_mat(elvect.GetData(), dof, fluxFunction.num_equations);

   // Obtain integration rule. If integration is rule is given, then use it.
   // Otherwise, get (2*p + IntOrderOffset) order integration rule
   const IntegrationRule *ir = IntRule;
   if (!ir)
   {
      const int order = 2*el.GetOrder() + IntOrderOffset;
      ir = &IntRules.Get(Tr.GetGeometryType(), order);
   }
   // loop over integration points
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

      Tr.SetAllIntPoints(&ip); // set face and element int. points

      // Calculate basis functions on both elements at the face
      el.CalcShape(Tr.GetElement1IntPoint(), shape);

      // Evaluate the coefficient at the point
      u_vcoeff.Eval(state, Tr, ip);

      // Get the normal vector and the flux on the face
      if (nor.Size() == 1)  // if 1D, use 1 or -1.
      {
         nor(0) = 2*Tr.GetElement1IntPoint().x - 1.;
      }
      else
      {
         CalcOrtho(Tr.Jacobian(), nor);
      }
      // Compute F(u, x) with maximum characteristic speed
      const real_t speed = fluxFunction.ComputeFluxDotN(state, nor, Tr, fluxN);

      // Update the global max char speed
      max_char_speed = std::max(speed, max_char_speed);

      // pre-multiply integration weight to flux
      const real_t a = 0.5 * alpha  * ip.weight;
      const real_t b = beta * ip.weight;

      for (int n = 0; n < fluxFunction.num_equations; n++)
      {
         fluxN(n) = a * fluxN(n) - b * fabs(fluxN(n));
      }

      AddMultVWt(shape, fluxN, elvect_mat);
   }
}

real_t FluxFunction::ComputeFluxDotN(const Vector &U,
                                     const Vector &normal,
                                     FaceElementTransformations &Tr,
                                     Vector &FUdotN) const
{
#ifdef MFEM_THREAD_SAFE
   DenseMatrix flux(num_equations, dim);
#else
   flux.SetSize(num_equations, dim);
#endif
   real_t val = ComputeFlux(U, Tr, flux);
   flux.Mult(normal, FUdotN);
   return val;
}

real_t FluxFunction::ComputeAvgFluxDotN(const Vector &U1, const Vector &U2,
                                        const Vector &normal,
                                        FaceElementTransformations &Tr,
                                        Vector &fluxDotN) const
{
#ifdef MFEM_THREAD_SAFE
   DenseMatrix flux(num_equations, dim);
#else
   flux.SetSize(num_equations, dim);
#endif
   real_t val = ComputeAvgFlux(U1, U2, Tr, flux);
   flux.Mult(normal, fluxDotN);
   return val;
}

void FluxFunction::ComputeFluxJacobianDotN(const Vector &U,
                                           const Vector &normal,
                                           ElementTransformation &Tr,
                                           DenseMatrix &JDotN) const
{
#ifdef MFEM_THREAD_SAFE
   DenseTensor J(num_equations, num_equations, dim);
#else
   J.SetSize(num_equations, num_equations, dim);
#endif
   ComputeFluxJacobian(U, Tr, J);
   JDotN.Set(normal(0), J(0));
   for (int d = 1; d < dim; d++)
   {
      JDotN.AddMatrix(normal(d), J(d), 0, 0);
   }
}

RusanovFlux::RusanovFlux(const FluxFunction &fluxFunction)
   : NumericalFlux(fluxFunction)
{
#ifndef MFEM_THREAD_SAFE
   fluxN1.SetSize(fluxFunction.num_equations);
   fluxN2.SetSize(fluxFunction.num_equations);
#endif
}

real_t RusanovFlux::Eval(const Vector &state1, const Vector &state2,
                         const Vector &nor, FaceElementTransformations &Tr,
                         Vector &flux) const
{
#ifdef MFEM_THREAD_SAFE
   Vector fluxN1(fluxFunction.num_equations), fluxN2(fluxFunction.num_equations);
#endif
   const real_t speed1 = fluxFunction.ComputeFluxDotN(state1, nor, Tr, fluxN1);
   const real_t speed2 = fluxFunction.ComputeFluxDotN(state2, nor, Tr, fluxN2);
   // NOTE: nor in general is not a unit normal
   const real_t maxE = std::max(speed1, speed2);
   // here, nor.Norml2() is multiplied to match the scale with fluxN
   const real_t scaledMaxE = maxE * nor.Norml2();
   for (int i = 0; i < fluxFunction.num_equations; i++)
   {
      flux(i) = 0.5*(scaledMaxE*(state1(i) - state2(i)) + (fluxN1(i) + fluxN2(i)));
   }
   return maxE;
}

void RusanovFlux::Grad(int side, const Vector &state1, const Vector &state2,
                       const Vector &nor, FaceElementTransformations &Tr,
                       DenseMatrix &grad) const
{
#ifdef MFEM_THREAD_SAFE
   Vector fluxN1(fluxFunction.num_equations), fluxN2(fluxFunction.num_equations);
#endif

   const real_t speed1 = fluxFunction.ComputeFluxDotN(state1, nor, Tr, fluxN1);
   const real_t speed2 = fluxFunction.ComputeFluxDotN(state2, nor, Tr, fluxN2);

   // NOTE: nor in general is not a unit normal
   const real_t maxE = std::max(speed1, speed2);
   // here, nor.Norml2() is multiplied to match the scale with fluxN
   const real_t scaledMaxE = maxE * nor.Norml2();

   if (side == 1)
   {
      fluxFunction.ComputeFluxJacobianDotN(state1, nor, Tr, grad);

      for (int i = 0; i < fluxFunction.num_equations; i++)
      {
         grad(i,i) += 0.5 * scaledMaxE;
      }
   }
   else
   {
      fluxFunction.ComputeFluxJacobianDotN(state2, nor, Tr, grad);

      for (int i = 0; i < fluxFunction.num_equations; i++)
      {
         grad(i,i) -= 0.5 * scaledMaxE;
      }
   }
}

real_t RusanovFlux::Average(const Vector &state1, const Vector &state2,
                            const Vector &nor, FaceElementTransformations &Tr,
                            Vector &flux) const
{
#ifdef MFEM_THREAD_SAFE
   Vector fluxN1(fluxFunction.num_equations), fluxN2(fluxFunction.num_equations);
#endif
   const real_t speed1 = fluxFunction.ComputeFluxDotN(state1, nor, Tr, fluxN1);
   const real_t speed2 = fluxFunction.ComputeAvgFluxDotN(state1, state2, nor, Tr,
                                                         fluxN2);
   // NOTE: nor in general is not a unit normal
   const real_t maxE = std::max(speed1, speed2);
   // here, nor.Norml2() is multiplied to match the scale with fluxN
   const real_t scaledMaxE = maxE * nor.Norml2() * 0.5;
   for (int i = 0; i < fluxFunction.num_equations; i++)
   {
      flux(i) = 0.5*(scaledMaxE*(state1(i) - state2(i)) + (fluxN1(i) + fluxN2(i)));
   }
   return maxE;
}

void RusanovFlux::AverageGrad(int side, const Vector &state1,
                              const Vector &state2,
                              const Vector &nor, FaceElementTransformations &Tr,
                              DenseMatrix &grad) const
{
#ifdef MFEM_THREAD_SAFE
   Vector fluxN1(fluxFunction.num_equations), fluxN2(fluxFunction.num_equations);
#endif

#if defined(MFEM_USE_DOUBLE)
   constexpr real_t tol = 1e-12;
#elif defined(MFEM_USE_SINGLE)
   constexpr real_t tol = 4e-6;
#else
#error "Only single and double precision are supported!"
   constexpr real_t tol = 1.;
#endif

   auto equal_check = [=](real_t a, real_t b) -> bool { return std::abs(a - b) <= tol * std::abs(a + b); };

   if (side == 1)
   {
#ifdef MFEM_THREAD_SAFE
      DenseMatrix JDotN(fluxFunction.num_equations);
#else
      JDotN.SetSize(fluxFunction.num_equations);
#endif
      const real_t speed1 = fluxFunction.ComputeFluxDotN(state1, nor, Tr, fluxN1);
      const real_t speed2 = fluxFunction.ComputeAvgFluxDotN(state1, state2, nor, Tr,
                                                            fluxN2);
      fluxFunction.ComputeFluxJacobianDotN(state1, nor, Tr, JDotN);

      // NOTE: nor in general is not a unit normal
      const real_t maxE = std::max(speed1, speed2);
      // here, nor.Norml2() is multiplied to match the scale with fluxN
      const real_t scaledMaxE = maxE * nor.Norml2() * 0.5;

      grad = 0.;

      for (int i = 0; i < fluxFunction.num_equations; i++)
      {
         // Only diagonal terms of J are considered
         // lim_{u → u⁻} (F̄(u⁻,u)n - F(u⁻)n) / (u - u⁻) = ½λ
         if (equal_check(state1(i), state2(i))) { continue; }
         grad(i,i) = 0.5 * ((fluxN2(i) - fluxN1(i)) / (state2(i) - state1(i))
                            - JDotN(i,i) + scaledMaxE);
      }
   }
   else
   {
      const real_t speed1 = fluxFunction.ComputeAvgFluxDotN(state1, state2, nor, Tr,
                                                            fluxN1);
      const real_t speed2 = fluxFunction.ComputeFluxDotN(state2, nor, Tr, fluxN2);

      // NOTE: nor in general is not a unit normal
      const real_t maxE = std::max(speed1, speed2);
      // here, nor.Norml2() is multiplied to match the scale with fluxN
      const real_t scaledMaxE = maxE * nor.Norml2() * 0.5;

      grad = 0.;

      for (int i = 0; i < fluxFunction.num_equations; i++)
      {
         // lim_{u → u⁻} (F(u)n - F̄(u⁻,u)n) / (u - u⁻) = ½λ
         if (equal_check(state1(i), state2(i))) { continue; }
         grad(i,i) = 0.5 * ((fluxN2(i) - fluxN1(i)) / (state2(i) - state1(i))
                            - scaledMaxE);
      }
   }
}

ComponentwiseUpwindFlux::ComponentwiseUpwindFlux(
   const FluxFunction &fluxFunction)
   : NumericalFlux(fluxFunction)
{
#ifndef MFEM_THREAD_SAFE
   fluxN1.SetSize(fluxFunction.num_equations);
   fluxN2.SetSize(fluxFunction.num_equations);
#endif
   if (fluxFunction.dim > 1)
      MFEM_WARNING("Upwinded flux is implemented only component-wise.")
   }

real_t ComponentwiseUpwindFlux::Eval(const Vector &state1, const Vector &state2,
                                     const Vector &nor, FaceElementTransformations &Tr,
                                     Vector &flux) const
{
#ifdef MFEM_THREAD_SAFE
   Vector fluxN1(fluxFunction.num_equations), fluxN2(fluxFunction.num_equations);
#endif
   const real_t speed1 = fluxFunction.ComputeFluxDotN(state1, nor, Tr, fluxN1);
   const real_t speed2 = fluxFunction.ComputeFluxDotN(state2, nor, Tr, fluxN2);

   for (int i = 0; i < fluxFunction.num_equations; i++)
   {
      if (state1(i) <= state2(i))
      {
         flux(i) = std::min(fluxN1(i), fluxN2(i));
      }
      else
      {
         flux(i) = std::max(fluxN1(i), fluxN2(i));
      }
   }

   return std::max(speed1, speed2);
}

void ComponentwiseUpwindFlux::Grad(int side, const Vector &state1,
                                   const Vector &state2,
                                   const Vector &nor, FaceElementTransformations &Tr,
                                   DenseMatrix &grad) const
{
#ifdef MFEM_THREAD_SAFE
   DenseMatrix JDotN(fluxFunction.num_equations);
#else
   JDotN.SetSize(fluxFunction.num_equations);
#endif

   grad = 0.;

   if (side == 1)
   {
      fluxFunction.ComputeFluxJacobianDotN(state1, nor, Tr, JDotN);

      for (int i = 0; i < fluxFunction.num_equations; i++)
      {
         // Only diagonal terms of J are considered
         grad(i,i) = std::max(JDotN(i,i), 0_r);
      }
   }
   else
   {
      fluxFunction.ComputeFluxJacobianDotN(state2, nor, Tr, JDotN);

      for (int i = 0; i < fluxFunction.num_equations; i++)
      {
         // Only diagonal terms of J are considered
         grad(i,i) = std::min(JDotN(i,i), 0_r);
      }
   }
}

real_t ComponentwiseUpwindFlux::Average(const Vector &state1,
                                        const Vector &state2,
                                        const Vector &nor, FaceElementTransformations &Tr,
                                        Vector &flux) const
{
#ifdef MFEM_THREAD_SAFE
   Vector fluxN1(fluxFunction.num_equations), fluxN2(fluxFunction.num_equations);
#endif
   const real_t speed1 = fluxFunction.ComputeFluxDotN(state1, nor, Tr, fluxN1);
   const real_t speed2 = fluxFunction.ComputeAvgFluxDotN(state1, state2, nor, Tr,
                                                         fluxN2);

   for (int i = 0; i < fluxFunction.num_equations; i++)
   {
      if (state1(i) <= state2(i))
      {
         flux(i) = std::min(fluxN1(i), fluxN2(i));
      }
      else
      {
         flux(i) = std::max(fluxN1(i), fluxN2(i));
      }
   }

   return std::max(speed1, speed2);
}

void ComponentwiseUpwindFlux::AverageGrad(int side, const Vector &state1,
                                          const Vector &state2,
                                          const Vector &nor, FaceElementTransformations &Tr,
                                          DenseMatrix &grad) const
{
#ifdef MFEM_THREAD_SAFE
   Vector fluxN1(fluxFunction.num_equations), fluxN2(fluxFunction.num_equations);
#endif

#if defined(MFEM_USE_DOUBLE)
   constexpr real_t tol = 1e-12;
#elif defined(MFEM_USE_SINGLE)
   constexpr real_t tol = 4e-6;
#else
#error "Only single and double precision are supported!"
   constexpr real_t tol = 1.;
#endif

   auto equal_check = [=](real_t a, real_t b) -> bool { return std::abs(a - b) <= tol * std::abs(a + b); };

   if (side == 1)
   {
#ifdef MFEM_THREAD_SAFE
      DenseMatrix JDotN(fluxFunction.num_equations);
#else
      JDotN.SetSize(fluxFunction.num_equations);
#endif
      fluxFunction.ComputeFluxDotN(state1, nor, Tr, fluxN1);
      fluxFunction.ComputeAvgFluxDotN(state1, state2, nor, Tr, fluxN2);
      fluxFunction.ComputeFluxJacobianDotN(state1, nor, Tr, JDotN);

      grad = 0.;

      for (int i = 0; i < fluxFunction.num_equations; i++)
      {
         // Only diagonal terms of J are considered
         // lim_{u → u⁻} (F̄(u⁻,u)n - F(u⁻)n) / (u - u⁻) = ½J(u⁻)n
         const real_t gr12 = (!equal_check(state1(i), state2(i)))?
                             (fluxN2(i) - fluxN1(i)) / (state2(i) - state1(i))
                             :(0.5 * JDotN(i,i));
         grad(i,i) = (gr12 >= 0.)?(JDotN(i,i)):(gr12);
      }
   }
   else
   {
#ifdef MFEM_THREAD_SAFE
      DenseMatrix JDotN;
#endif
      fluxFunction.ComputeAvgFluxDotN(state1, state2, nor, Tr, fluxN1);
      fluxFunction.ComputeFluxDotN(state2, nor, Tr, fluxN2);

      // Jacobian is not needed except the limit case when u⁺=u⁻
      bool J_needed = false;
      for (int i = 0; i < fluxFunction.num_equations; i++)
         if (equal_check(state1(i), state2(i)))
         {
            J_needed = true;
            break;
         }

      if (J_needed)
      {
         JDotN.SetSize(fluxFunction.num_equations);
         fluxFunction.ComputeFluxJacobianDotN(state1, nor, Tr, JDotN);
      }

      grad = 0.;

      for (int i = 0; i < fluxFunction.num_equations; i++)
      {
         // Only diagonal terms of J are considered
         // lim_{u → u⁻} (F(u)n - F̄(u⁻,u)n) / (u - u⁻) = ½J(u⁻)n
         const real_t gr12 = (!equal_check(state1(i), state2(i)))?
                             (fluxN2(i) - fluxN1(i)) / (state2(i) - state1(i))
                             :(0.5 * JDotN(i,i));
         grad(i,i) = std::min(gr12, 0_r);
      }
   }
}

real_t HDGFlux::Average(const Vector &state1, const Vector &state2,
                        const Vector &nor, FaceElementTransformations &Tr,
                        Vector &flux) const
{
#ifdef MFEM_THREAD_SAFE
   Vector fluxN1(fluxFunction.num_equations), fluxN2(fluxFunction.num_equations);
#endif

   // HDG-I/II schemes

   const real_t speed1 = fluxFunction.ComputeFluxDotN(state1, nor, Tr, fluxN1);
   const real_t speed2 = fluxFunction.ComputeFluxDotN(state2, nor, Tr, fluxN2);
   switch (scheme)
   {
      case HDGScheme::HDG_1:
         flux = fluxN1;
         break;
      case HDGScheme::HDG_2:
         flux = fluxN2;
         break;
   }
   for (int i = 0; i < fluxFunction.num_equations; i++)
   {
      flux(i) += Ctau * (state2(i) - state1(i)) * nor.Norml2();
   }
   return std::max(speed1, speed2);
}

void HDGFlux::AverageGrad(int side, const Vector &state1, const Vector &state2,
                          const Vector &nor, FaceElementTransformations &Tr,
                          DenseMatrix &grad) const
{
   MFEM_ASSERT(side == 1 || side == 2, "Unknown side");

   // HDG-I/II schemes

   switch (scheme)
   {
      case HDGScheme::HDG_1:
         if (side == 1)
         { fluxFunction.ComputeFluxJacobianDotN(state1, nor, Tr, grad); }
         else
         { grad = 0.; }
         break;
      case HDGScheme::HDG_2:
         if (side == 1)
         { grad = 0.; }
         else
         { fluxFunction.ComputeFluxJacobianDotN(state2, nor, Tr, grad); }
         break;
   }

   if (side == 1)
      for (int i = 0; i < fluxFunction.num_equations; i++)
      {
         grad(i,i) -= Ctau * nor.Norml2();
      }
   else
      for (int i = 0; i < fluxFunction.num_equations; i++)
      {
         grad(i,i) += Ctau * nor.Norml2();
      }
}

real_t AdvectionFlux::ComputeFlux(const Vector &U,
                                  ElementTransformation &Tr,
                                  DenseMatrix &FU) const
{
#ifdef MFEM_THREAD_SAFE
   Vector bval(b.GetVDim());
#endif
   b.Eval(bval, Tr, Tr.GetIntPoint());
   MultVWt(U, bval, FU);
   return bval.Norml2();
}

real_t AdvectionFlux::ComputeFluxDotN(const Vector &U,
                                      const Vector &normal,
                                      FaceElementTransformations &Tr,
                                      Vector &FDotN) const
{
#ifdef MFEM_THREAD_SAFE
   Vector bval(b.GetVDim());
#endif
   b.Eval(bval, Tr, Tr.GetIntPoint());
   FDotN(0) = U(0) * (bval * normal);
   return bval.Norml2();
}

real_t AdvectionFlux::ComputeAvgFlux(const Vector &U1, const Vector &U2,
                                     ElementTransformation &Tr,
                                     DenseMatrix &FU) const
{
#ifdef MFEM_THREAD_SAFE
   Vector bval(b.GetVDim());
#endif
   b.Eval(bval, Tr, Tr.GetIntPoint());
   Vector Uavg(1);
   Uavg(0) = (U1(0) + U2(0)) * 0.5;
   MultVWt(Uavg, bval, FU);
   return bval.Norml2();
}

real_t AdvectionFlux::ComputeAvgFluxDotN(const Vector &U1, const Vector &U2,
                                         const Vector &normal,
                                         FaceElementTransformations &Tr,
                                         Vector &FDotN) const
{
#ifdef MFEM_THREAD_SAFE
   Vector bval(b.GetVDim());
#endif
   b.Eval(bval, Tr, Tr.GetIntPoint());
   FDotN(0) = (U1(0) + U2(0)) * 0.5 * (bval * normal);
   return bval.Norml2();
}

void AdvectionFlux::ComputeFluxJacobian(const Vector &state,
                                        ElementTransformation &Tr,
                                        DenseTensor &J) const
{
#ifdef MFEM_THREAD_SAFE
   Vector bval(b.GetVDim());
#endif
   b.Eval(bval, Tr, Tr.GetIntPoint());
   J = 0.;
   for (int d = 0; d < dim; d++)
   {
      J(0,0,d) = bval(d);
   }
}

void AdvectionFlux::ComputeFluxJacobianDotN(const Vector &state,
                                            const Vector &normal,
                                            ElementTransformation &Tr,
                                            DenseMatrix &JDotN) const
{
#ifdef MFEM_THREAD_SAFE
   Vector bval(b.GetVDim());
#endif
   b.Eval(bval, Tr, Tr.GetIntPoint());
   JDotN(0,0) = bval * normal;
}

real_t BurgersFlux::ComputeFlux(const Vector &U,
                                ElementTransformation &Tr,
                                DenseMatrix &FU) const
{
   FU = U(0) * U(0) * 0.5;
   return std::fabs(U(0));
}

real_t BurgersFlux::ComputeFluxDotN(const Vector &U,
                                    const Vector &normal,
                                    FaceElementTransformations &Tr,
                                    Vector &FDotN) const
{
   FDotN(0) = U(0) * U(0) * 0.5 * normal.Sum();
   return std::fabs(U(0));
}

real_t BurgersFlux::ComputeAvgFlux(const Vector &U1,
                                   const Vector &U2,
                                   ElementTransformation &Tr,
                                   DenseMatrix &FU) const
{
   FU = (U1(0)*U1(0) + U1(0)*U2(0) + U2(0)*U2(0)) / 6.;
   return std::max(std::fabs(U1(0)), std::fabs(U2(0)));
}

real_t BurgersFlux::ComputeAvgFluxDotN(const Vector &U1,
                                       const Vector &U2,
                                       const Vector &normal,
                                       FaceElementTransformations &Tr,
                                       Vector &FDotN) const
{
   FDotN(0) = (U1(0)*U1(0) + U1(0)*U2(0) + U2(0)*U2(0)) / 6. * normal.Sum();
   return std::max(std::fabs(U1(0)), std::fabs(U2(0)));
}

void BurgersFlux::ComputeFluxJacobian(const Vector &U,
                                      ElementTransformation &Tr,
                                      DenseTensor &J) const
{
   J = 0.;
   for (int d = 0; d < dim; d++)
   {
      J(0,0,d) = U(0);
   }
}

void BurgersFlux::ComputeFluxJacobianDotN(const Vector &U,
                                          const Vector &normal,
                                          ElementTransformation &Tr,
                                          DenseMatrix &JDotN) const
{
   JDotN(0,0) = U(0) * normal.Sum();
}

real_t ShallowWaterFlux::ComputeFlux(const Vector &U,
                                     ElementTransformation &Tr,
                                     DenseMatrix &FU) const
{
   const real_t height = U(0);
   const Vector h_vel(U.GetData() + 1, dim);

   const real_t energy = 0.5 * g * (height * height);

   MFEM_ASSERT(height >= 0, "Negative Height");

   for (int d = 0; d < dim; d++)
   {
      FU(0, d) = h_vel(d);
      for (int i = 0; i < dim; i++)
      {
         FU(1 + i, d) = h_vel(i) * h_vel(d) / height;
      }
      FU(1 + d, d) += energy;
   }

   const real_t sound = std::sqrt(g * height);
   const real_t vel = std::sqrt(h_vel * h_vel) / height;

   return vel + sound;
}


real_t ShallowWaterFlux::ComputeFluxDotN(const Vector &U,
                                         const Vector &normal,
                                         FaceElementTransformations &Tr,
                                         Vector &FUdotN) const
{
   const real_t height = U(0);
   const Vector h_vel(U.GetData() + 1, dim);

   const real_t energy = 0.5 * g * (height * height);

   MFEM_ASSERT(height >= 0, "Negative Height");
   FUdotN(0) = h_vel * normal;
   const real_t normal_vel = FUdotN(0) / height;
   for (int i = 0; i < dim; i++)
   {
      FUdotN(1 + i) = normal_vel * h_vel(i) + energy * normal(i);
   }

   const real_t sound = std::sqrt(g * height);
   const real_t vel = std::fabs(normal_vel) / std::sqrt(normal*normal);

   return vel + sound;
}


real_t EulerFlux::ComputeFlux(const Vector &U,
                              ElementTransformation &Tr,
                              DenseMatrix &FU) const
{
   // 1. Get states
   const real_t density = U(0);                  // ρ
   const Vector momentum(U.GetData() + 1, dim);  // ρu
   const real_t energy = U(1 + dim);             // E, internal energy ρe
   const real_t kinetic_energy = (density > 0.)?(0.5 * (momentum*momentum) /
                                                 density):(0.);
   // pressure, p = (γ-1)*(E - ½ρ|u|^2)
   const real_t pressure = (specific_heat_ratio - 1.0) *
                           (energy - kinetic_energy);

   // Check whether the solution is physical only in debug mode
   MFEM_ASSERT(density >= 0, "Negative Density");
   MFEM_ASSERT(pressure >= 0, "Negative Pressure");
   MFEM_ASSERT(energy >= 0, "Negative Energy");

   // Detect vacuum state
   if (density == 0.)
   {
      FU = 0.;
      return 0.;
   }

   // 2. Compute Flux
   for (int d = 0; d < dim; d++)
   {
      FU(0, d) = momentum(d);  // ρu
      for (int i = 0; i < dim; i++)
      {
         // ρuuᵀ
         FU(1 + i, d) = momentum(i) * momentum(d) / density;
      }
      // (ρuuᵀ) + p
      FU(1 + d, d) += pressure;
   }
   // enthalpy H = e + p/ρ = (E + p)/ρ
   const real_t H = (energy + pressure) / density;
   for (int d = 0; d < dim; d++)
   {
      // u(E+p) = ρu*(E + p)/ρ = ρu*H
      FU(1 + dim, d) = momentum(d) * H;
   }

   // 3. Compute maximum characteristic speed

   // sound speed, √(γ p / ρ)
   const real_t sound = std::sqrt(specific_heat_ratio * pressure / density);
   // fluid speed |u|
   const real_t speed = std::sqrt(2.0 * kinetic_energy / density);
   // max characteristic speed = fluid speed + sound speed
   return speed + sound;
}


real_t EulerFlux::ComputeFluxDotN(const Vector &x,
                                  const Vector &normal,
                                  FaceElementTransformations &Tr,
                                  Vector &FUdotN) const
{
   // 1. Get states
   const real_t density = x(0);                  // ρ
   const Vector momentum(x.GetData() + 1, dim);  // ρu
   const real_t energy = x(1 + dim);             // E, internal energy ρe
   const real_t kinetic_energy = (density > 0.)?(0.5 * (momentum*momentum) /
                                                 density):(0.);
   // pressure, p = (γ-1)*(E - ½ρ|u|^2)
   const real_t pressure = (specific_heat_ratio - 1.0) *
                           (energy - kinetic_energy);

   // Check whether the solution is physical only in debug mode
   MFEM_ASSERT(density >= 0, "Negative Density");
   MFEM_ASSERT(pressure >= 0, "Negative Pressure");
   MFEM_ASSERT(energy >= 0, "Negative Energy");

   // Detect vacuum state
   if (density == 0.)
   {
      FUdotN = 0.;
      return 0.;
   }

   // 2. Compute normal flux

   FUdotN(0) = momentum * normal;  // ρu⋅n
   // u⋅n
   const real_t normal_velocity = FUdotN(0) / density;
   for (int d = 0; d < dim; d++)
   {
      // (ρuuᵀ + pI)n = ρu*(u⋅n) + pn
      FUdotN(1 + d) = normal_velocity * momentum(d) + pressure * normal(d);
   }
   // (u⋅n)(E + p)
   FUdotN(1 + dim) = normal_velocity * (energy + pressure);

   // 3. Compute maximum characteristic speed

   // sound speed, √(γ p / ρ)
   const real_t sound = std::sqrt(specific_heat_ratio * pressure / density);
   // fluid speed |u|
   const real_t speed = std::fabs(normal_velocity) / std::sqrt(normal*normal);
   // max characteristic speed = fluid speed + sound speed
   return speed + sound;
}

real_t EulerFlux::CalcAvgKineticEnergy(real_t density1, real_t density2,
                                       const Vector &momentum1, const Vector &momentum2)
{
   real_t kinetic_energy = 0.;

   const real_t density = 0.5 * (density1 + density2);
   const real_t dden = (density2 - density1) / density;
   constexpr real_t den_tol = 5e-6;
   const int dim = momentum1.Size();

   if (fabs(dden) > den_tol)
   {
      const real_t dlnden = (-log(1. - dden/2.) + log(1. + dden/2.)) / dden;
      for (int d = 0; d < dim; d++)
      {
         const real_t &mom1 = momentum1(d);
         const real_t &mom2 = momentum2(d);
         const real_t dmom = mom1 - mom2;
         const real_t amom = 0.5 * (mom1 + mom2);
         kinetic_energy += 0.5 * ((dlnden - 1.) / (dden*dden) * dmom *
                                  (dmom + amom * 2. * dden)
                                  + amom*amom * dlnden) / density;
      }
   }
   else
   {
      for (int d = 0; d < dim; d++)
      {
         const real_t &mom1 = momentum1(d);
         const real_t &mom2 = momentum2(d);
         const real_t dmom = mom1 - mom2;
         const real_t amom = 0.5 * (mom1 + mom2);
         kinetic_energy += (mom1*mom1 + mom1*mom2 + mom2*mom2 + amom * dmom * dden /
                            2.) / (6. * density);
      }
   }

   return kinetic_energy;
}

void EulerFlux::CalcAvgFlux(real_t density1, real_t density2,
                            const Vector &momentum1, const Vector &momentum2, real_t vel1, real_t vel2,
                            Vector &flux)
{
   const real_t density = 0.5 * (density1 + density2);
   const real_t dden = (density2 - density1) / density;
   constexpr real_t den_tol = 5e-6;
   const int dim = momentum1.Size();
   const real_t avel = 0.5 * (vel1 + vel2);
   const real_t dvel = vel1 - vel2;

   if (fabs(dden) > den_tol)
   {
      const real_t dlnden = (-log(1. - dden/2.) + log(1. + dden/2.)) / dden;
      for (int d = 0; d < dim; d++)
      {
         const real_t &mom1 = momentum1(d);
         const real_t &mom2 = momentum2(d);
         const real_t dmom = mom1 - mom2;
         const real_t amom = 0.5 * (mom1 + mom2);
         flux(d) = (dlnden - 1.) / (dden*dden) *
                   (dmom * dvel + (dmom * avel + amom * dvel) * dden)
                   + amom*avel * dlnden;
      }
   }
   else
   {
      for (int d = 0; d < dim; d++)
      {
         const real_t &mom1 = momentum1(d);
         const real_t &mom2 = momentum2(d);
         const real_t dmom = mom1 - mom2;
         const real_t amom = 0.5 * (mom1 + mom2);
         flux(d) = (2.*mom1*vel1 + mom1*vel2 + mom2*vel1 + 2.*mom2*vel2) / 6. +
                   (amom * dvel + dmom * avel) * dden / 12.;
      }
   }
}

real_t EulerFlux::CalcAvgFlux(real_t den1, real_t den2, real_t mom1,
                              real_t mom2, real_t vel1, real_t vel2)
{
   Vector vmom1{mom1}, vmom2{mom2}, flux(1);
   CalcAvgFlux(den1, den2, vmom1, vmom2, vel1, vel2, flux);
   return flux(0);
}

real_t EulerFlux::ComputeAvgFlux(const Vector &U1, const Vector &U2,
                                 ElementTransformation &Tr, DenseMatrix &FAvgU) const
{
   // 1. Get states
   const real_t density1 = U1(0);                 // ρ1
   const real_t density2 = U2(0);                 // ρ2
   const Vector momentum1(U1.GetData() + 1, dim); // ρu1
   const Vector momentum2(U2.GetData() + 1, dim); // ρu2
   const real_t energy1 = U1(1 + dim);            // E1, internal energy ρe1
   const real_t energy2 = U2(1 + dim);            // E2, internal energy ρe2
   const real_t kinetic_energy1 = (density1 > 0.)?(0.5 * (momentum1*momentum1) /
                                                   density1):(0.);
   const real_t kinetic_energy2 = (density2 > 0.)?(0.5 * (momentum2*momentum2) /
                                                   density2):(0.);
   // pressure, p = (γ-1)*(E - ½ρ|u|^2)
   const real_t pressure1 = (specific_heat_ratio - 1.0) *
                            (energy1 - kinetic_energy1);
   const real_t pressure2 = (specific_heat_ratio - 1.0) *
                            (energy2 - kinetic_energy2);

   // Check whether the solution is physical only in debug mode
   MFEM_ASSERT(density1 >= 0 && density2 >= 0, "Negative Density");
   MFEM_ASSERT(pressure1 >= 0 && pressure2 >= 0, "Negative Pressure");
   MFEM_ASSERT(energy1 >= 0 && energy2 >= 0, "Negative Energy");

   // Detect vacuum state
   if (density1 == 0.)
   {
      return ComputeFlux(U2, Tr, FAvgU);
   }
   else if (density2 == 0.)
   {
      return ComputeFlux(U1, Tr, FAvgU);
   }

   // density
   const real_t density = 0.5 * (density1 + density2);

   // energy
   const real_t energy = 0.5 * (energy1 + energy2);

   // kinetic energy, ½ρ|u|^2
   const real_t kinetic_energy = CalcAvgKineticEnergy(density1, density2,
                                                      momentum1, momentum2);

   // pressure, p = (γ-1)*(E - ½ρ|u|^2)
   const real_t pressure = (specific_heat_ratio - 1.0) * (energy - kinetic_energy);
   MFEM_ASSERT(pressure >= 0, "Negative Pressure");

   // 2. Compute Flux
   for (int d = 0; d < dim; d++)
   {
      const real_t &mom1_d = momentum1(d);
      const real_t &mom2_d = momentum2(d);
      FAvgU(0, d) = 0.5 * (mom1_d + mom2_d); // ρu

      // ρuuᵀ
      Vector FAvgU_mom(FAvgU.GetData() + 1 + d*(2+dim), dim);
      const real_t vel1_d = mom1_d / density;
      const real_t vel2_d = mom2_d / density;
      CalcAvgFlux(density1, density2, momentum1, momentum2, vel1_d, vel2_d,
                  FAvgU_mom);
      // (ρuuᵀ) + p
      FAvgU(1 + d, d) += pressure;
   }
   // enthalpy H = e + p/ρ = (E + p)/ρ
   // note we take Hρ = E + p as a primary variable and average it, neglecting
   // higher order terms in averaging of the kinetic energy in the definition
   // of pressure
   // u(E+p) = ρu*(E + p)/ρ = ρu*H
   const real_t H1 = (energy1 + pressure1) / density;
   const real_t H2 = (energy2 + pressure2) / density;
   Vector FAvgU_en(dim);
   CalcAvgFlux(density1, density2, momentum1, momentum2, H1, H2, FAvgU_en);
   FAvgU.SetRow(1 + dim, FAvgU_en);

   // 3. Compute maximum characteristic speed

   // sound speed, √(γ p / ρ)
   const real_t sound = std::sqrt(specific_heat_ratio * pressure / density);
   // fluid speed |u|
   const real_t speed = std::sqrt(2.0 * kinetic_energy / density);
   // max characteristic speed = fluid speed + sound speed
   return speed + sound;
}

real_t EulerFlux::ComputeAvgFluxDotN(const Vector &U1, const Vector &U2,
                                     const Vector &normal, FaceElementTransformations &Tr,
                                     Vector &FAvgUDotN) const
{
   // 1. Get states
   const real_t density1 = U1(0);                 // ρ1
   const real_t density2 = U2(0);                 // ρ2
   const Vector momentum1(U1.GetData() + 1, dim); // ρu1
   const Vector momentum2(U2.GetData() + 1, dim); // ρu2
   const real_t energy1 = U1(1 + dim);            // E1, internal energy ρe1
   const real_t energy2 = U2(1 + dim);            // E2, internal energy ρe2
   const real_t kinetic_energy1 = (density1 > 0.)?(0.5 * (momentum1*momentum1) /
                                                   density1):(0.);
   const real_t kinetic_energy2 = (density2 > 0.)?(0.5 * (momentum2*momentum2) /
                                                   density2):(0.);
   // pressure, p = (γ-1)*(E - ½ρ|u|^2)
   const real_t pressure1 = (specific_heat_ratio - 1.0) *
                            (energy1 - kinetic_energy1);
   const real_t pressure2 = (specific_heat_ratio - 1.0) *
                            (energy2 - kinetic_energy2);

   // Check whether the solution is physical only in debug mode
   MFEM_ASSERT(density1 >= 0 && density2 >= 0, "Negative Density");
   MFEM_ASSERT(pressure1 >= 0 && pressure2 >= 0, "Negative Pressure");
   MFEM_ASSERT(energy1 >= 0 && energy2 >= 0, "Negative Energy");

   // Detect vacuum state
   if (density1 == 0.)
   {
      return ComputeFluxDotN(U2, normal, Tr, FAvgUDotN);
   }
   else if (density2 == 0.)
   {
      return ComputeFluxDotN(U1, normal, Tr, FAvgUDotN);
   }

   // density
   const real_t density = 0.5 * (density1 + density2);

   // energy
   const real_t energy = 0.5 * (energy1 + energy2);

   // kinetic energy, ½ρ|u|^2
   const real_t kinetic_energy = CalcAvgKineticEnergy(density1, density2,
                                                      momentum1, momentum2);

   // pressure, p = (γ-1)*(E - ½ρ|u|^2)
   const real_t pressure = (specific_heat_ratio - 1.0) * (energy - kinetic_energy);
   MFEM_ASSERT(pressure >= 0, "Negative Pressure");

   // 2. Compute normal flux

   FAvgUDotN(0) = 0.5 * (momentum1 * normal + momentum2 * normal); // ρu⋅n

   // ρuu⋅n + pn
   Vector FAvgUDotN_mom(FAvgUDotN, 1, dim);
   const real_t nvel1 = momentum1 * normal / density;
   const real_t nvel2 = momentum2 * normal / density;
   CalcAvgFlux(density1, density2, momentum1, momentum2, nvel1, nvel2,
               FAvgUDotN_mom);
   FAvgUDotN_mom.Add(pressure, normal);

   // enthalpy H = e + p/ρ = (E + p)/ρ
   // note we take Hρ = E + p as a primary variable and average it, neglecting
   // higher order terms in averaging of the kinetic energy in the definition
   // of pressure
   const real_t Hrho1 = energy1 + pressure1;
   const real_t Hrho2 = energy2 + pressure2;
   // u⋅n(E+p) = ρu⋅n*(E + p)/ρ = ρu⋅n*H
   FAvgUDotN(1 + dim) = CalcAvgFlux(density1, density2, Hrho1, Hrho2, nvel1,
                                    nvel2);

   // 3. Compute maximum characteristic speed

   // sound speed, √(γ p / ρ)
   const real_t sound = std::sqrt(specific_heat_ratio * pressure / density);
   // fluid speed |u|
   const real_t speed = std::sqrt(2.0 * kinetic_energy / density);
   // max characteristic speed = fluid speed + sound speed
   return speed + sound;
}

void EulerFlux::ComputeFluxJacobian(const Vector &U, ElementTransformation &Tr,
                                    DenseTensor &JU) const
{
   // 1. Get states
   const real_t density = U(0);                  // ρ
   const Vector momentum(U.GetData() + 1, dim);  // ρu
   const real_t energy = U(1 + dim);             // E, internal energy ρe
   const real_t kinetic_energy = (density > 0.)?(0.5 * (momentum*momentum) /
                                                 density):(0.);
   // pressure, p = (γ-1)*(E - ½ρ|u|^2)
   const real_t pressure = (specific_heat_ratio - 1.0) *
                           (energy - kinetic_energy);

   // Check whether the solution is physical only in debug mode
   MFEM_ASSERT(density >= 0, "Negative Density");
   MFEM_ASSERT(pressure >= 0, "Negative Pressure");
   MFEM_ASSERT(energy >= 0, "Negative Energy");

   // Detect vacuum state
   if (density == 0.)
   {
      JU = 0.;
      return;
   }

   // 2. Compute Jacobian
   JU = 0.;
   for (int d = 0; d < dim; d++)
   {
      const real_t velocity_d = momentum(d) / density;
      // ρu
      JU(0, 1 + d, d) = 1.;
      // ρuuᵀ
      JU(1 + d, 1 + d, d) = 2. * velocity_d;
      for (int i = 0; i < dim; i++)
      {
         const real_t velocity_i = momentum(i) / density;
         // ρuuᵀ
         JU(1 + i, 0, d) = -velocity_i * velocity_d;
         if (i != d)
         {
            JU(1 + i, 1 + i, d) = velocity_d;
            JU(1 + i, 1 + d, d) = velocity_i;
         }
      }
      // (ρuuᵀ) + p
      JU(1 + d, 0, d) += (specific_heat_ratio - 1.0) * kinetic_energy / density;
      for (int i = 0; i < dim; i++)
      {
         JU(1 + d, 1 + i, d) -= (specific_heat_ratio - 1.0) * momentum(i) / density;
      }
      JU(1 + d, 1 + dim, d) = specific_heat_ratio - 1.0;
   }
   // enthalpy H = e + p/ρ = (E + p)/ρ
   const real_t H = (energy + pressure) / density;
   // u(E+p) = ρu*(E + p)/ρ = ρu*H
   for (int d = 0; d < dim; d++)
   {
      const real_t velocity_d = momentum(d) / density;
      JU(1 + dim, 0, d) = velocity_d * (-H + (specific_heat_ratio - 1.0) *
                                        kinetic_energy);
      for (int i = 0; i < dim; i++)
      {
         JU(1 + dim, 1 + i, d) = -(specific_heat_ratio - 1.0) * velocity_d
                                 * momentum(i) / density;
      }
      JU(1 + dim, 1 + d, d) += H;
      JU(1 + dim, 1 + dim, d) = specific_heat_ratio * velocity_d;
   }
}

real_t CompoundFlux::ComputeFlux(const Vector &Uv, ElementTransformation &Tr,
                                 DenseMatrix &Fv) const
{
   Vector U(1);
   DenseMatrix F(1, dim);
   real_t max_char_speed = 0.;
   for (int i = 0; i < num_equations; i++)
   {
      U(0) = Uv(i);
      const real_t speed = flux.ComputeFlux(U, Tr, F);
      max_char_speed = std::max(speed, max_char_speed);
      Fv.SetRow(i, F.GetData());
   }
   return max_char_speed;
}

real_t CompoundFlux::ComputeFluxDotN(const Vector &Uv, const Vector &normal,
                                     FaceElementTransformations &Tr, Vector &FvDotN) const
{
   Vector U(1);
   Vector FDotN(1);
   real_t max_char_speed = 0.;
   for (int i = 0; i < num_equations; i++)
   {
      U(0) = Uv(i);
      const real_t speed = flux.ComputeFluxDotN(U, normal, Tr, FDotN);
      max_char_speed = std::max(speed, max_char_speed);
      FvDotN(i) = FDotN(0);
   }
   return max_char_speed;
}

real_t CompoundFlux::ComputeAvgFlux(const Vector &Uv1, const Vector &Uv2,
                                    ElementTransformation &Tr, DenseMatrix &Fv) const
{
   Vector U1(1), U2(1);
   DenseMatrix F(1, dim);
   real_t max_char_speed = 0.;
   for (int i = 0; i < num_equations; i++)
   {
      U1(0) = Uv1(i);
      U2(0) = Uv2(i);
      const real_t speed = flux.ComputeAvgFlux(U1, U2, Tr, F);
      max_char_speed = std::max(speed, max_char_speed);
      Fv.SetRow(i, F.GetData());
   }
   return max_char_speed;
}

real_t CompoundFlux::ComputeAvgFluxDotN(const Vector &Uv1, const Vector &Uv2,
                                        const Vector &normal, FaceElementTransformations &Tr, Vector &FvDotN) const
{
   Vector U1(1), U2(1);
   Vector FDotN(1);
   real_t max_char_speed = 0.;
   for (int i = 0; i < num_equations; i++)
   {
      U1(0) = Uv1(i);
      U2(0) = Uv2(i);
      const real_t speed = flux.ComputeAvgFluxDotN(U1, U2, normal, Tr, FDotN);
      max_char_speed = std::max(speed, max_char_speed);
      FvDotN(i) = FDotN(0);
   }
   return max_char_speed;
}

void CompoundFlux::ComputeFluxJacobian(const Vector &Uv,
                                       ElementTransformation &Tr, DenseTensor &Jv) const
{
   Vector U(1);
   DenseTensor J(1, 1, dim);
   for (int i = 0; i < num_equations; i++)
   {
      U(0) = Uv(i);
      flux.ComputeFluxJacobian(U, Tr, J);
      for (int d = 0; d < dim; d++)
      {
         Jv(i, i, d) = J(0, 0, d);
      }
   }
}

void CompoundFlux::ComputeFluxJacobianDotN(const Vector &Uv,
                                           const Vector &normal, ElementTransformation &Tr, DenseMatrix &JvDotN) const
{
   Vector U(1);
   DenseMatrix JDotN(1, 1);
   for (int i = 0; i < num_equations; i++)
   {
      U(0) = Uv(i);
      flux.ComputeFluxJacobianDotN(U, normal, Tr, JDotN);
      JvDotN(i, i) = JDotN(0, 0);
   }
}

} // namespace mfem
