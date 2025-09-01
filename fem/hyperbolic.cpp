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
   const real_t kinetic_energy = 0.5 * (momentum*momentum) / density;
   // pressure, p = (γ-1)*(E - ½ρ|u|^2)
   const real_t pressure = (specific_heat_ratio - 1.0) *
                           (energy - kinetic_energy);

   // Check whether the solution is physical only in debug mode
   MFEM_ASSERT(density >= 0, "Negative Density");
   MFEM_ASSERT(pressure >= 0, "Negative Pressure");
   MFEM_ASSERT(energy >= 0, "Negative Energy");

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
   const real_t kinetic_energy = 0.5 * (momentum*momentum) / density;
   // pressure, p = (γ-1)*(E - ½ρ|u|^2)
   const real_t pressure = (specific_heat_ratio - 1.0) *
                           (energy - kinetic_energy);

   // Check whether the solution is physical only in debug mode
   MFEM_ASSERT(density >= 0, "Negative Density");
   MFEM_ASSERT(pressure >= 0, "Negative Pressure");
   MFEM_ASSERT(energy >= 0, "Negative Energy");

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

} // namespace mfem
