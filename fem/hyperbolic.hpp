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


#ifndef MFEM_HYPERBOLIC
#define MFEM_HYPERBOLIC

#include "nonlinearform.hpp"

namespace mfem
{

//                     MFEM Hyperbolic Conservation Laws
//
// Description:
//
//    This file contains general hyperbolic conservation element/face form
//    integrators.
//
//    HyperbolicFormIntegrator and RiemannSolver are defined.
//    HyperbolicFormIntegrator is a NonlinearFormIntegrator that implements
//    element weak divergence and interface flux
//
//       ∫_T F(u):∇v,   ∫_e F̂(u)⋅[[v]]
//
//    Here, T is an element, e is an edge, and [[⋅]] is jump. This form
//    integrator is coupled with RiemannSolver that implements the numerical
//    flux F̂. For RiemannSolver, the Rusanov flux, also known as local
//    Lax-Friedrichs flux, is provided.
//
//    To implement a specific hyperbolic conservation laws, users can create
//    derived classes from FluxFunction with overloaded ComputeFlux. One can
//    optionally overload ComputeFluxDotN to avoid creating dense matrix when
//    computing normal flux. Several example equations are also defined
//    including: advection, Burgers', shallow water, and Euler equations. Users
//    can control the quadrature rule by either providing the integration rule,
//    or integration order offset. See, HyperbolicFormIntegrator::GetRule.
//
//    At each call of HyperbolicFormIntegrator::AssembleElementVector
//    HyperbolicFormIntegrator::AssembleFaceVector, the maximum characteristic
//    speed will be updated. This will not be reinitialized automatically.
//    To reinitialize, use HyperbolicFormIntegrator::ResetMaxCharSpeed. See,
//    ex18.hpp.
//
//    Note: To avoid communication overhead, we update the maximum
//    characteristic speed within each process. Use a proper MPI routine to
//    gather the information.
//

/**
 * @brief Abstract class for hyperbolic flux for a system of hyperbolic
 * conservation laws
 *
 */
class FluxFunction
{
public:
   const int num_equations;
   const int dim;

   FluxFunction(const int num_equations,
                const int dim):num_equations(num_equations), dim(dim)
   {
#ifndef MFEM_THREAD_SAFE
      flux.SetSize(num_equations, dim);
#endif
   };

   /**
    * @brief Compute flux F(u, x) for given state u and physical point x
    *
    * @param[in] state value of state at the current integration point
    * @param[in] Tr element information
    * @param[out] flux F(u, x)
    * @return double maximum characteristic speed
    *
    * @note One can put assertion in here to detect non-physical solution
    */
   virtual double ComputeFlux(const Vector &state, ElementTransformation &Tr,
                              DenseMatrix &flux) const = 0;
   /**
    * @brief Compute normal flux. Optionally overloadded in the
    * derived class to avoid creating full dense matrix for flux.
    *
    * @param[in] state state at the current integration point
    * @param[in] normal normal vector, @see CalcOrtho
    * @param[in] Tr face information
    * @param[out] fluxDotN normal flux from the given element at the current
    * integration point
    * @return double maximum (normal) characteristic velocity
    */
   virtual double ComputeFluxDotN(const Vector &state, const Vector &normal,
                                  FaceElementTransformations &Tr,
                                  Vector &fluxDotN) const;

   /**
    * @brief Compute flux Jacobian. Optionally overloaded in the derived class
    * when Jacobian is necessary (e.g. Newton iteration, flux limiter)
    *
    * @param state state at the current integration point
    * @param Tr element information
    * @param J flux Jacobian, J(i,j,d) = dF_{id} / u_j
    */
   virtual void ComputeFluxJacobian(const Vector &state,
                                    ElementTransformation &Tr,
                                    DenseTensor &J) const
   {
      MFEM_ABORT("Not Implemented.");
   }
private:
#ifndef MFEM_THREAD_SAFE
   mutable DenseMatrix flux;
#endif
};


/**
 * @brief Abstract class for numerical flux for a system of hyperbolic
 * conservation laws on a face with states, fluxes and characteristic speed
 *
 */
class RiemannSolver
{
public:
   RiemannSolver(const FluxFunction &fluxFunction):fluxFunction(fluxFunction) {}
   /**
    * @brief Evaluates numerical flux for given states and fluxes. Must be
    * overloaded in a derived class
    *
    * @param[in] state1 state value at a point from the first element
    * (num_equations)
    * @param[in] state2 state value at a point from the second element
    * (num_equations)
    * @param[in] nor scaled normal vector, @see mfem::CalcOrtho (dim)
    * @param[in] Tr face information
    * @param[out] flux numerical flux (num_equations)
    */
   virtual double Eval(const Vector &state1, const Vector &state2,
                       const Vector &nor, FaceElementTransformations &Tr,
                       Vector &flux) const = 0;
   virtual ~RiemannSolver() = default;

   /// @brief Get flux function F
   /// @return constant reference to the flux function.
   const FluxFunction &GetFluxFunction() const {return fluxFunction;}
protected:
   const FluxFunction &fluxFunction;
};

/**
 * @brief Abstract hyperbolic form integrator, (F(u, x), ∇v) and (F̂(u±, x, n))
 *
 */
class HyperbolicFormIntegrator : public NonlinearFormIntegrator
{
private:
   // The maximum characterstic speed, updated during element/face vector assembly
   double max_char_speed;
   const RiemannSolver &rsolver;   // Numerical flux that maps F(u±,x) to hat(F)
   const FluxFunction &fluxFunction;
   const int IntOrderOffset; // 2*p + IntOrderOffset will be used for quadrature
#ifndef MFEM_THREAD_SAFE
   // Local storages for element integration
   Vector shape;              // shape function value at an integration point
   Vector state;              // state value at an integration point
   DenseMatrix flux;          // flux value at an integration point
   DenseMatrix dshape;  // derivative of shape function at an integration point

   Vector shape1;  // shape function value at an integration point - first elem
   Vector shape2;  // shape function value at an integration point - second elem
   Vector state1;  // state value at an integration point - first elem
   Vector state2;  // state value at an integration point - second elem
   Vector fluxN1;  // flux dot n value at an integration point - first elem
   Vector fluxN2;  // flux dot n value at an integration point - second elem
   Vector nor;     // normal vector, @see CalcOrtho
   Vector fluxN;   // hat(F)(u,x)
#endif

public:
   const int num_equations;  // the number of equations
   /**
    * @brief Construct a new Hyperbolic Form Integrator object
    *
    * @param[in] rsolver numerical flux
    * @param[in] IntOrderOffset 2*p+IntOrderOffset order Gaussian quadrature
    * will be used
    */
   HyperbolicFormIntegrator(
      const RiemannSolver &rsolver,
      const int IntOrderOffset=0);

   /**
    * @brief Get the element integration rule based on IntOrderOffset, @see
    * AssembleElementVector. Used only when ir is not provided
    *
    * @param[in] el given finite element space
    * @param[in] Tr Element transformation for Jacobian order
    * @param[in] IntOrderOffset_ integration order offset
    * @return const IntegrationRule& with order 2*p*Tr.OrderJ() + IntOrderOffset
    */
   static const IntegrationRule &GetRule(const FiniteElement &el,
                                         const ElementTransformation &Tr,
                                         const int IntOrderOffset_)
   {
      const int order = 2 * el.GetOrder() + Tr.OrderJ() + IntOrderOffset_;
      return IntRules.Get(el.GetGeomType(), order);
   }

   /**
    * @brief Get the face integration rule based on IntOrderOffset, @see
    * AssembleFaceVector. Used only when ir is not provided
    *
    * @param[in] trial_fe trial finite element space
    * @param[in] test_fe test finite element space
    * @param[in] Tr Face element trasnformation for Jacobian order
    * @param[in] IntOrderOffset_ integration order offset
    * @return const IntegrationRule& with order (p1 + p2)*Tr.OrderJ() +
    *    IntOrderOffset
    */
   static const IntegrationRule &GetRule(const FiniteElement &trial_fe,
                                         const FiniteElement &test_fe,
                                         const FaceElementTransformations &Tr,
                                         const int IntOrderOffset_)
   {
      const int order = trial_fe.GetOrder() + test_fe.GetOrder() + Tr.OrderJ() +
                        IntOrderOffset_;
      return IntRules.Get(trial_fe.GetGeomType(), order);
   }

   /**
    * @brief Reset the Max Char Speed 0
    *
    */
   void ResetMaxCharSpeed()
   {
      max_char_speed = 0.0;
   }

   double GetMaxCharSpeed()
   {
      return max_char_speed;
   }

   /**
    * @brief implement (F(u), grad v) with abstract F computed by ComputeFlux
    *
    * @param[in] el local finite element
    * @param[in] Tr element transformation
    * @param[in] elfun local coefficient of basis
    * @param[out] elvect evaluated dual vector
    */
   void AssembleElementVector(const FiniteElement &el,
                              ElementTransformation &Tr,
                              const Vector &elfun, Vector &elvect) override;

   /**
    * @brief implement <-hat(F)(u,x) n, [[v]]> with abstract hat(F) computed by
    * ComputeFluxDotN and numerical flux object
    *
    * @param[in] el1 finite element of the first element
    * @param[in] el2 finite element of the second element
    * @param[in] Tr face element transformations
    * @param[in] elfun local coefficient of basis from both elements
    * @param[out] elvect evaluated dual vector <-hat(F)(u,x) n, [[v]]>
    */
   void AssembleFaceVector(const FiniteElement &el1,
                           const FiniteElement &el2,
                           FaceElementTransformations &Tr,
                           const Vector &elfun, Vector &elvect) override;

};


//////////////////////////////////////////////////////////////////
///                      NUMERICAL FLUXES                      ///
//////////////////////////////////////////////////////////////////

/**
 * @brief Rusanov flux, also known as local Lax-Friedrichs,
 *    F̂ n = ½(F(u⁺,x)n + F(u⁻,x)n) - ½λ(u⁺ - u⁻)
 * where λ is the maximum characteristic velocity
 *
 */
class RusanovFlux : public RiemannSolver
{
public:
   RusanovFlux(const FluxFunction &fluxFunction): RiemannSolver(fluxFunction) {}
   /**
    * @brief  hat(F)n = ½(F(u⁺,x)n + F(u⁻,x)n) - ½λ(u⁺ - u⁻)
    *
    * @param[in] state1 state value at a point from the first element
    * (num_equations)
    * @param[in] state2 state value at a point from the second element
    * (num_equations)
    * @param[in] nor normal vector (not a unit vector) (dim)
    * @param[in] Tr face element transformation
    * @param[out] flux ½(F(u⁺,x)n + F(u⁻,x)n) - ½λ(u⁺ - u⁻)
    */
   double Eval(const Vector &state1, const Vector &state2,
               const Vector &nor, FaceElementTransformations &Tr,
               Vector &flux) const override;
protected:
#ifndef MFEM_THREAD_SAFE
   mutable Vector fluxN1, fluxN2;
#endif
};

class AdvectionFlux : public FluxFunction
{
private:
   VectorCoefficient &b;  // velocity coefficient
#ifndef MFEM_THREAD_SAFE
   mutable Vector bval;           // velocity value storage
#endif

public:

   /**
    * @brief Construct a new Advection Flux Function with given
    * spatial dimension
    *
    * @param b velocity coefficient, possibly depends on space
    */
   AdvectionFlux(VectorCoefficient &b)
      : FluxFunction(1, b.GetVDim()), b(b)
   {
#ifndef MFEM_THREAD_SAFE
      bval.SetSize(b.GetVDim());
#endif
   }
   /**
    * @brief Compute F(u)
    *
    * @param state state (u) at current integration point
    * @param Tr current element transformation with integration point
    * @param flux F(u) = ubᵀ
    * @return double maximum characteristic speed, |b|
    */
   double ComputeFlux(const Vector &state, ElementTransformation &Tr,
                      DenseMatrix &flux) const override;
};

class BurgersFlux : public FluxFunction
{
public:
   /**
    * @brief Construct a new Burgers Flux Function with given
    * spatial dimension
    *
    * @param dim spatial dimension
    */
   BurgersFlux(const int dim)
      : FluxFunction(1, dim) {}

   /**
    * @brief Compute F(u)
    *
    * @param state state (u) at current integration point
    * @param Tr current element transformation with integration point
    * @param flux F(u) = ½u²*1ᵀ where 1 is (dim) vector
    * @return double maximum characteristic speed, |u|
    */
   double ComputeFlux(const Vector &state, ElementTransformation &Tr,
                      DenseMatrix &flux) const override;
};

class ShallowWaterFlux : public FluxFunction
{
private:
   const double g;  // gravity constant

public:
   /**
    * @brief Construct a new Shallow Water Flux Function with
    * given spatial dimension
    *
    * @param dim spatial dimension
    * @param g gravity constant
    */
   ShallowWaterFlux(const int dim, const double g=9.8)
      : FluxFunction(dim + 1, dim), g(g) {}

   /**
    * @brief Compute F(h, hu)
    *
    * @param state state (h, hu) at current integration point
    * @param Tr current element transformation with integration point
    * @param flux F(h, hu) = [huᵀ; huuᵀ + ½gh²I]
    * @return double maximum characteristic speed, |u| + √(gh)
    */
   double ComputeFlux(const Vector &state, ElementTransformation &Tr,
                      DenseMatrix &flux) const override;
   /**
    * @brief Compute normal flux, F(h, hu)
    *
    * @param state state (h, hu) at current integration point
    * @param normal normal vector, usually not a unit vector
    * @param Tr current element transformation with integration point
    * @param fluxN F(ρ, ρu, E)n = [ρu⋅n; ρu(u⋅n) + pn; (u⋅n)(E + p)]
    * @return double maximum characteristic speed, |u| + √(γp/ρ)
    */
   double ComputeFluxDotN(const Vector &state, const Vector &normal,
                          FaceElementTransformations &Tr,
                          Vector &fluxN) const override;
};

class EulerFlux : public FluxFunction
{
private:
   const double specific_heat_ratio;  // specific heat ratio, γ
   // const double gas_constant;         // gas constant

public:
   /**
    * @brief Construct a new Euler Flux Function with given
    * spatial dimension
    *
    * @param dim spatial dimension
    * @param specific_heat_ratio specific heat ratio, γ
    */
   EulerFlux(const int dim, const double specific_heat_ratio)
      : FluxFunction(dim + 2, dim),
        specific_heat_ratio(specific_heat_ratio) {}

   /**
    * @brief Compute F(ρ, ρu, E)
    *
    * @param state state (ρ, ρu, E) at current integration point
    * @param Tr current element transformation with integration point
    * @param flux F(ρ, ρu, E) = [ρuᵀ; ρuuᵀ + pI; uᵀ(E + p)]
    * @return double maximum characteristic speed, |u| + √(γp/ρ)
    */
   double ComputeFlux(const Vector &state, ElementTransformation &Tr,
                      DenseMatrix &flux) const override;

   /**
    * @brief Compute normal flux, F(ρ, ρu, E)n
    *
    * @param x x (ρ, ρu, E) at current integration point
    * @param normal normal vector, usually not a unit vector
    * @param Tr current element transformation with integration point
    * @param fluxN F(ρ, ρu, E)n = [ρu⋅n; ρu(u⋅n) + pn; (u⋅n)(E + p)]
    * @return double maximum characteristic speed, |u| + √(γp/ρ)
    */
   double ComputeFluxDotN(const Vector &x, const Vector &normal,
                          FaceElementTransformations &Tr,
                          Vector &fluxN) const override;
};

} // namespace mfem

#endif // MFEM_HYPERBOLIC
