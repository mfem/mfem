// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
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
   const RiemannSolver &rsolver,
   const int IntOrderOffset,
   real_t sign)
   : NonlinearFormIntegrator(),
     rsolver(rsolver),
     fluxFunction(rsolver.GetFluxFunction()),
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
   nor.SetSize(fluxFunction.dim);
#endif
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
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
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
   const int dof2 = el2.GetDof();

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
      const int order = 2*std::max(el1.GetOrder(), el2.GetOrder()) + IntOrderOffset;
      ir = &IntRules.Get(Tr.GetGeometryType(), order);
   }
   // loop over integration points
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

      Tr.SetAllIntPoints(&ip); // set face and element int. points

      // Calculate basis functions on both elements at the face
      el1.CalcShape(Tr.GetElement1IntPoint(), shape1);
      el2.CalcShape(Tr.GetElement2IntPoint(), shape2);

      // Interpolate elfun at the point
      elfun1_mat.MultTranspose(shape1, state1);
      elfun2_mat.MultTranspose(shape2, state2);

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
      // Compute F(u+, x) and F(u-, x) with maximum characteristic speed
      // Compute hat(F) using evaluated quantities
      const real_t speed = rsolver.Eval(state1, state2, nor, Tr, fluxN);

      // Update the global max char speed
      max_char_speed = std::max(speed, max_char_speed);

      // pre-multiply integration weight to flux
      AddMult_a_VWt(-ip.weight*sign, shape1, fluxN, elvect1_mat);
      AddMult_a_VWt(+ip.weight*sign, shape2, fluxN, elvect2_mat);
   }
}

HDGHyperbolicFormIntegrator::HDGHyperbolicFormIntegrator(
   HDGScheme scheme, const RiemannSolver &rsolver, real_t Ctau,
   const int IntOrderOffset, const real_t sign)
   : HyperbolicFormIntegrator(rsolver, IntOrderOffset, sign),
     scheme(scheme), Ctau(Ctau)
{
#ifndef MFEM_THREAD_SAFE
   JDotN.SetSize(num_equations);
#endif
}

void HDGHyperbolicFormIntegrator::AssembleHDGFaceVector(
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

      // Compute F(û, x) and F(u, x) with maximum characteristic speed
      // Compute hat(F) using evaluated quantities
      switch (scheme)
      {
         case HDGScheme::HDG_1:
            fluxFunction.ComputeFluxDotN(state_tr, nor, Tr, fluxN);
            break;
         case HDGScheme::HDG_2:
            fluxFunction.ComputeFluxDotN(state_el, nor, Tr, fluxN);
            break;
      }

      // Compute stabilization
      for (int d = 0; d < num_equations; d++)
      {
         fluxN(d) += Ctau * (state_el(d) - state_tr(d)) * nor.Norml2();
      }

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

void HDGHyperbolicFormIntegrator::AssembleHDGFaceGrad(
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

      // Compute J(û, x) and J(u, x)
      int joff = 0;
      if (type & (HDGFaceType::ELEM | HDGFaceType::CONSTR))
      {
         switch (scheme)
         {
            case HDGScheme::HDG_1:
               JDotN = 0.;
               break;
            case HDGScheme::HDG_2:
               fluxFunction.ComputeFluxJacobianDotN(state_el, nor, Tr, JDotN);
               break;
         }
         // Compute stabilization
         for (int d = 0; d < num_equations; d++)
         {
            JDotN(d,d) += Ctau * state_el(d) * nor.Norml2();
         }


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
         switch (scheme)
         {
            case HDGScheme::HDG_1:
               fluxFunction.ComputeFluxJacobianDotN(state_tr, nor, Tr, JDotN);
               break;
            case HDGScheme::HDG_2:
               JDotN = 0.;
               break;
         }
         // Compute stabilization
         for (int d = 0; d < num_equations; d++)
         {
            JDotN(d,d) -= Ctau * state_tr(d) * nor.Norml2();
         }

         // pre-multiply integration weight to Jacobians
         const real_t w = -ip.weight*sign;
         int ioff = 0;
         if (type & HDGFaceType::TRACE)
         {
            for (int di = 0; di < num_equations; di++)
               for (int dj = 0; dj < num_equations; dj++)
                  for (int i = 0; i < dof_el; i++)
                     for (int j = 0; j < dof_el; j++)
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
                     for (int j = 0; j < dof_el; j++)
                     {
                        elmat(di*dof_dual+ioff+i, dj*dof_prim+joff+j) +=
                           w * JDotN(di,dj) * shape_tr(i) * shape_tr(j);
                     }
         }
      }
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
   // here, std::sqrt(nor*nor) is multiplied to match the scale with fluxN
   const real_t scaledMaxE = maxE*std::sqrt(nor*nor);
   for (int i=0; i<state1.Size(); i++)
   {
      flux[i] = 0.5*(scaledMaxE*(state1[i] - state2[i]) + (fluxN1[i] + fluxN2[i]));
   }
   return std::max(speed1, speed2);
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


real_t BurgersFlux::ComputeFlux(const Vector &U,
                                ElementTransformation &Tr,
                                DenseMatrix &FU) const
{
   FU = U * U * 0.5;
   return std::fabs(U(0));
}

void BurgersFlux::ComputeFluxJacobian(const Vector &U,
                                      ElementTransformation &Tr,
                                      DenseTensor &J) const
{
   J = 0.;
   for (int i = 0; i < num_equations; i++)
      for (int d = 0; d < dim; d++)
      {
         J(i, i, d) = U(i);
      }
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
