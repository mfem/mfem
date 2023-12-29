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

// Implementation of hyperbolic conservation laws

#include "hyperbolic_conservation_laws.hpp"
#ifdef MFEM_USE_MPI
#include "pnonlinearform.hpp"
#endif

namespace mfem
{
//////////////////////////////////////////////////////////////////
///        HYPERBOLIC CONSERVATION LAWS IMPLEMENTATION         ///
//////////////////////////////////////////////////////////////////

// Implementation of class DGHyperbolicConservationLaws
DGHyperbolicConservationLaws::DGHyperbolicConservationLaws(
   FiniteElementSpace *vfes_, HyperbolicFormIntegrator *formIntegrator_,
   const int num_equations_)
   : TimeDependentOperator(vfes_->GetNDofs() * num_equations_),
     dim(vfes_->GetFE(0)->GetDim()),
     num_equations(num_equations_),
     vfes(vfes_),
     formIntegrator(formIntegrator_),
     Me_inv(0),
     z(vfes_->GetNDofs() * num_equations_)
{
   // Standard local assembly and inversion for energy mass matrices.
   ComputeInvMass();
#ifndef MFEM_USE_MPI
   nonlinearForm.reset(new NonlinearForm(vfes));
#else
   ParFiniteElementSpace *pvfes = dynamic_cast<ParFiniteElementSpace *>(vfes);
   if (pvfes)
   {
      nonlinearForm.reset(new ParNonlinearForm(pvfes));
   }
   else
   {
      nonlinearForm.reset(new NonlinearForm(vfes));
   }
#endif
   formIntegrator->resetMaxCharSpeed();

   nonlinearForm->AddDomainIntegrator(formIntegrator);
   nonlinearForm->AddInteriorFaceIntegrator(formIntegrator);

   height = z.Size();
   width = z.Size();
}

void DGHyperbolicConservationLaws::ComputeInvMass()
{
   DenseMatrix Me;     // auxiliary local mass matrix
   MassIntegrator mi;  // mass integrator
   // resize it to the current number of elements
   Me_inv.resize(vfes->GetNE());
   for (int i = 0; i < vfes->GetNE(); i++)
   {
      Me.SetSize(vfes->GetFE(i)->GetDof());
      mi.AssembleElementMatrix(*vfes->GetFE(i),
                               *vfes->GetElementTransformation(i), Me);
      DenseMatrixInverse inv(&Me);
      inv.Factor();
      inv.GetInverseMatrix(Me_inv[i]);
   }
}

void DGHyperbolicConservationLaws::Mult(const Vector &x, Vector &y) const
{
   // 0. Reset wavespeed computation before operator application.
   formIntegrator->resetMaxCharSpeed();
   // 1. Create the vector z with the face terms (F(u), grad v) - <F.n(u), [w]>.
   nonlinearForm->Mult(x, z);
   max_char_speed = formIntegrator->getMaxCharSpeed();

   // 2. Multiply element-wise by the inverse mass matrices.
   Vector zval;             // local dual vector storage
   Array<int> vdofs;        // local degrees of freedom storage
   DenseMatrix zmat, ymat;  // local dual vector storage

   for (int i = 0; i < vfes->GetNE(); i++)
   {
      // Return the vdofs ordered byNODES
      vfes->GetElementVDofs(i, vdofs);
      // get local dual vector
      z.GetSubVector(vdofs, zval);
      zmat.UseExternalData(zval.GetData(), vfes->GetFE(i)->GetDof(),
                           num_equations);
      ymat.SetSize(Me_inv[i].Height(), num_equations);
      // mass matrix inversion and pass it to global vector
      mfem::Mult(Me_inv[i], zmat, ymat);
      y.SetSubVector(vdofs, ymat.GetData());
   }
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
   // Local storages for element integration

   Vector shape(dof); // shape function value at an integration point
   DenseMatrix dshape(dof,
                      el.GetDim()); // derivative of shape function at an integration point
   Vector state(num_equations); // state value at an integration point
   DenseMatrix flux(num_equations,
                    el.GetDim()); // flux value at an integration point
#else
   // resize shape and gradient shape storage
   shape.SetSize(dof);
   dshape.SetSize(dof, el.GetDim());
#endif

   // setDegree-up output vector
   elvect.SetSize(dof * num_equations);
   elvect = 0.0;

   // make state variable and output dual vector matrix form.
   const DenseMatrix elfun_mat(elfun.GetData(), dof, num_equations);
   DenseMatrix elvect_mat(elvect.GetData(), dof, num_equations);

   // obtain integration rule. If integration is rule is given, then use it.
   // Otherwise, get (2*p + IntOrderOffset) order integration rule
   const IntegrationRule *ir = IntRule ? IntRule : &GetRule(el, Tr);
   // loop over interation points
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      Tr.SetIntPoint(&ip);

      el.CalcShape(ip, shape);
      el.CalcPhysDShape(Tr, dshape);
      // compute current state value with given shape function values
      elfun_mat.MultTranspose(shape, state);
      // compute F(u,x) and point maximum characteristic speed

      const double mcs = ComputeFlux(state, Tr, flux);
      // update maximum characteristic speed
      max_char_speed = std::max(mcs, max_char_speed);
      // integrate (F(u,x), grad v)
      AddMult_a_ABt(ip.weight * Tr.Weight(), dshape, flux, elvect_mat);
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
   // Local storages for element integration

   Vector shape1(
      dof1);  // shape function value at an integration point - first elem
   Vector shape2(
      dof2);  // shape function value at an integration point - second elem
   Vector nor(el1.GetDim());     // normal vector (usually not a unit vector)
   Vector state1(
      num_equations);  // state value at an integration point - first elem
   Vector state2(
      num_equations);  // state value at an integration point - second elem
   Vector fluxN1(
      num_equations);  // flux dot n value at an integration point - first elem
   Vector fluxN2(
      num_equations);  // flux dot n value at an integration point - second elem
   Vector fluxN(num_equations);   // hat(F)(u,x)
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

   // obtain integration rule. If integration is rule is given, then use it.
   // Otherwise, get (2*p + IntOrderOffset) order integration rule
   const IntegrationRule *ir = IntRule ? IntRule : &GetRule(el1, el2, Tr);
   // loop over integration points
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

      Tr.SetAllIntPoints(&ip);  // setDegree face and element int. points

      // Calculate basis functions on both elements at the face
      el1.CalcShape(Tr.GetElement1IntPoint(), shape1);
      el2.CalcShape(Tr.GetElement2IntPoint(), shape2);

      // Interpolate elfun at the point
      elfun1_mat.MultTranspose(shape1, state1);
      elfun2_mat.MultTranspose(shape2, state2);

      // Get the normal vector and the flux on the face
      if (nor.Size() == 1)  // if 1D, use 1 or -1.
      {
         // This assume the 1D integration point is in (0,1). This may not work if
         // this chages.
         nor(0) = (Tr.GetElement1IntPoint().x - 0.5) * 2.0;
      }
      else
      {
         CalcOrtho(Tr.Jacobian(), nor);
      }
      // Compute F(u+, x) and F(u-, x) with maximum characteristic speed
      const double speed1 = ComputeFluxDotN(state1, nor,
                                            Tr.GetElement1Transformation(), fluxN1);
      const double speed2 = ComputeFluxDotN(state2, nor,
                                            Tr.GetElement2Transformation(), fluxN2);
      // Compute hat(F) using evaluated quantities
      rsolver.Eval(state1, state2, fluxN1, fluxN2, speed1, speed2, nor, fluxN);

      // Update the global max char speed
      max_char_speed = std::max(std::max(speed1, speed2), max_char_speed);

      // pre-multiply integration weight to flux
      fluxN *= ip.weight;
      for (int k = 0; k < num_equations; k++)
      {
         // this loop structure can increase cache hit because
         for (int s = 0; s < dof1; s++)
         {
            elvect1_mat(s, k) -= fluxN(k) * shape1(s);
         }
         for (int s = 0; s < dof2; s++)
         {
            elvect2_mat(s, k) += fluxN(k) * shape2(s);
         }
      }
   }
}

double HyperbolicFormIntegrator::ComputeFluxDotN(const Vector &U,
                                                         const Vector &normal,
                                                         ElementTransformation &Tr, Vector &FUdotN)
{
   double val = ComputeFlux(U, Tr, flux);
   flux.Mult(normal, FUdotN);
   return val;
}

}