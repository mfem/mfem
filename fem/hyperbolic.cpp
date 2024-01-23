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

#include "hyperbolic.hpp"
#include "nonlinearform.hpp"
#include "pnonlinearform.hpp"

namespace mfem
{


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
   const IntegrationRule *ir = IntRule ? IntRule : &GetRule(el, Tr,
                                                            IntOrderOffset);
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

      const double mcs = fluxFunction.ComputeFlux(state, Tr, flux);
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
   const IntegrationRule *ir = IntRule ? IntRule : &GetRule(el1, el2, Tr,
                                                            IntOrderOffset);
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
      // Compute hat(F) using evaluated quantities
      const double speed = rsolver.Eval(fluxFunction, state1, state2, nor, Tr, fluxN);

      // Update the global max char speed
      max_char_speed = std::max(speed, max_char_speed);

      // pre-multiply integration weight to flux
      fluxN *= ip.weight;
      for (int k = 0; k < num_equations; k++)
      {
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

HyperbolicFormIntegrator::HyperbolicFormIntegrator(
   const FluxFunction &fluxFunction, const RiemannSolver &rsolver,
   const int IntOrderOffset)
   : NonlinearFormIntegrator(),
     fluxFunction(fluxFunction),
     IntOrderOffset(IntOrderOffset),
     rsolver(rsolver),
     num_equations(fluxFunction.num_equations)
{
#ifndef MFEM_THREAD_SAFE
   state.SetSize(num_equations);
   flux.SetSize(num_equations, fluxFunction.dim);
   state1.SetSize(num_equations);
   state2.SetSize(num_equations);
   fluxN1.SetSize(num_equations);
   fluxN2.SetSize(num_equations);
   fluxN.SetSize(num_equations);
   nor.SetSize(fluxFunction.dim);
#endif
}

double FluxFunction::ComputeFluxDotN(const Vector &U,
                                     const Vector &normal,
                                     FaceElementTransformations &Tr, Vector &FUdotN) const
{
   double val = ComputeFlux(U, Tr, flux);
   flux.Mult(normal, FUdotN);
   return val;
}


double RusanovFlux::Eval(const FluxFunction &fluxFunction,
                         const Vector &state1, const Vector &state2,
                         const Vector &nor, FaceElementTransformations &Tr,
                         Vector &flux) const
{
#ifdef MFEM_THREAD_SAFE
   Vector fluxN1(fluxFunction.num_equations), fluxN2(fluxFunction.num_equations);
#else
   fluxN1.SetSize(fluxFunction.num_equations);
   fluxN2.SetSize(fluxFunction.num_equations);
#endif
   const double speed1 = fluxFunction.ComputeFluxDotN(state1, nor, Tr, fluxN1);
   const double speed2 = fluxFunction.ComputeFluxDotN(state2, nor, Tr, fluxN2);
   // NOTE: nor in general is not a unit normal
   const double maxE = std::max(speed1, speed2);
   // here, std::sqrt(nor*nor) is multiplied to match the scale with fluxN
   const double scaledMaxE = maxE*sqrt(nor*nor);
   for (int i=0; i<state1.Size(); i++)
   {
      flux[i] = 0.5*(scaledMaxE*(state1[i] - state2[i]) + (fluxN1[i] + fluxN2[i]));
   }
   return std::max(speed1, speed2);
}


double AdvectionFlux::ComputeFlux(const Vector &U,
                                  ElementTransformation &Tr,
                                  DenseMatrix &FU) const
{
   b.Eval(bval, Tr, Tr.GetIntPoint());
   MultVWt(U, bval, FU);
   return bval.Norml2();
}


double BurgersFlux::ComputeFlux(const Vector &U,
                                ElementTransformation &Tr,
                                DenseMatrix &FU) const
{
   FU = U * U * 0.5;
   return std::fabs(U(0));
}


double ShallowWaterFlux::ComputeFlux(const Vector &U,
                                     ElementTransformation &Tr,
                                     DenseMatrix &FU) const
{
   const int dim = U.Size() - 1;
   const double height = U(0);
   const Vector h_vel(U.GetData() + 1, dim);

   const double energy = 0.5 * g * (height * height);

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

   const double sound = std::sqrt(g * height);
   const double vel = std::sqrt(h_vel * h_vel) / height;

   return vel + sound;
}


double ShallowWaterFlux::ComputeFluxDotN(const Vector &U,
                                         const Vector &normal,
                                         FaceElementTransformations &Tr, Vector &FUdotN) const
{
   const int dim = normal.Size();
   const double height = U(0);
   const Vector h_vel(U.GetData() + 1, dim);

   const double energy = 0.5 * g * (height * height);

   MFEM_ASSERT(height >= 0, "Negative Height");
   FUdotN(0) = h_vel * normal;
   const double normal_vel = FUdotN(0) / height;
   for (int i = 0; i < dim; i++)
   {
      FUdotN(1 + i) = normal_vel * h_vel(i) + energy * normal(i);
   }

   const double sound = std::sqrt(g * height);
   const double vel = std::fabs(h_vel * normal / normal.Norml2()) / height;

   return vel + sound;
}


double EulerFlux::ComputeFlux(const Vector &U,
                              ElementTransformation &Tr,
                              DenseMatrix &FU) const
{
   const int dim = U.Size() - 2;

   // 1. Get states
   const double density = U(0);                  // ρ
   const Vector momentum(U.GetData() + 1, dim);  // ρu
   const double energy = U(1 + dim);             // E, internal energy ρe
   // pressure, p = (γ-1)*(ρu - ½ρ|u|²)
   const double pressure = (specific_heat_ratio - 1.0) *
                           (energy - 0.5 * (momentum * momentum) / density);

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
   const double H = (energy + pressure) / density;
   for (int d = 0; d < dim; d++)
   {
      // u(E+p) = ρu*(E + p)/ρ = ρu*H
      FU(1 + dim, d) = momentum(d) * H;
   }

   // 3. Compute maximum characteristic speed

   // sound speed, √(γ p / ρ)
   const double sound = std::sqrt(specific_heat_ratio * pressure / density);
   // fluid speed |u|
   const double speed = std::sqrt(momentum * momentum) / density;
   // max characteristic speed = fluid speed + sound speed
   return speed + sound;
}


double EulerFlux::ComputeFluxDotN(const Vector &x,
                                  const Vector &normal,
                                  FaceElementTransformations &Tr, Vector &FUdotN) const
{
   const int dim = normal.Size();

   // 1. Get states
   const double density = x(0);                  // ρ
   const Vector momentum(x.GetData() + 1, dim);  // ρu
   const double energy = x(1 + dim);             // E, internal energy ρe
   // pressure, p = (γ-1)*(E - ½ρ|u|^2)
   const double pressure = (specific_heat_ratio - 1.0) *
                           (energy - 0.5 * (momentum * momentum) / density);

   // Check whether the solution is physical only in debug mode
   MFEM_ASSERT(density >= 0, "Negative Density");
   MFEM_ASSERT(pressure >= 0, "Negative Pressure");
   MFEM_ASSERT(energy >= 0, "Negative Energy");

   // 2. Compute normal flux

   FUdotN(0) = momentum * normal;  // ρu⋅n
   // u⋅n
   const double normal_velocity = FUdotN(0) / density;
   for (int d = 0; d < dim; d++)
   {
      // (ρuuᵀ + pI)n = ρu*(u⋅n) + pn
      FUdotN(1 + d) = normal_velocity * momentum(d) + pressure * normal(d);
   }
   // (u⋅n)(E + p)
   FUdotN(1 + dim) = normal_velocity * (energy + pressure);

   // 3. Compute maximum characteristic speed

   // sound speed, √(γ p / ρ)
   const double sound = std::sqrt(specific_heat_ratio * pressure / density);
   // fluid speed |u|
   const double speed = std::fabs(momentum * normal / normal.Norml2()) / density;
   // max characteristic speed = fluid speed + sound speed
   return speed + sound;
}

}
