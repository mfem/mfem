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


#ifndef MFEM_HYPERBOLIC
#define MFEM_HYPERBOLIC

#include "nonlinearform.hpp"

namespace mfem
{

// This file contains general hyperbolic conservation element/face form
// integrators. HyperbolicFormIntegrator and NumericalFlux are defined.
//
// HyperbolicFormIntegrator is a NonlinearFormIntegrator that implements
// element weak divergence and interface flux
//
//     ∫_K F(u):∇v,   -∫_f F̂(u)⋅n[v]
//
// Here, K is an element, f is a face, n normal and [⋅] is jump. This form
// integrator is coupled with NumericalFlux that implements the numerical flux
// F̂. For NumericalFlux, the Rusanov flux, also known as local Lax-Friedrichs
// flux, or component-wise upwinded flux are provided.
//
// To implement a specific hyperbolic conservation laws, users can create
// derived classes from FluxFunction with overloaded ComputeFlux. One can
// optionally overload ComputeFluxDotN to avoid creating dense matrix when
// computing normal flux. Several example equations are also defined including:
// advection, Burgers', shallow water, and Euler equations. Users can control
// the quadrature rule by either providing the integration rule, or integration
// order offset. Integration will use 2*p + IntOrderOffset order quadrature
// rule.
//
// At each call of HyperbolicFormIntegrator::AssembleElementVector
// HyperbolicFormIntegrator::AssembleFaceVector, the maximum characteristic
// speed will be updated. This will not be reinitialized automatically. To
// reinitialize, use HyperbolicFormIntegrator::ResetMaxCharSpeed. See, ex18.hpp.
//
// Note: To avoid communication overhead, we update the maximum characteristic
// speed within each MPI process only. Use the appropriate MPI routine to gather
// the information.

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

   FluxFunction(const int num_equations, const int dim)
      : num_equations(num_equations), dim(dim) { }

   virtual ~FluxFunction() {}

   /**
    * @brief Compute flux F(u, x). Must be implemented in a derived class.
    *
    * Used in HyperbolicFormIntegrator::AssembleElementVector() for evaluation
    * of (F(u), ∇v) and in the default implementation of ComputeFluxDotN()
    * for evaluation of F(u)⋅n.
    * @param[in] state state at the current integration point (num_equations)
    * @param[in] Tr element transformation
    * @param[out] flux flux from the given element at the current
    * integration point (num_equations, dim)
    * @return real_t maximum characteristic speed |dF(u,x)/du|
    *
    * @note One can put assertion in here to detect non-physical solution
    */
   virtual real_t ComputeFlux(const Vector &state, ElementTransformation &Tr,
                              DenseMatrix &flux) const = 0;
   /**
    * @brief Compute normal flux F(u, x)⋅n. Optionally overloaded in a derived
    * class to avoid creating a full dense matrix for flux.
    *
    * Used in NumericalFlux for evaluation of the normal flux on a face.
    * @param[in] state state at the current integration point (num_equations)
    * @param[in] normal normal vector, see mfem::CalcOrtho() (dim)
    * @param[in] Tr face transformation
    * @param[out] fluxDotN normal flux from the given element at the current
    * integration point (num_equations)
    * @return real_t maximum (normal) characteristic speed |dF(u,x)/du⋅n|
    */
   virtual real_t ComputeFluxDotN(const Vector &state, const Vector &normal,
                                  FaceElementTransformations &Tr,
                                  Vector &fluxDotN) const;

   /**
    * @brief Compute average flux over the given interval of states.
    * Optionally overloaded in a derived class.
    *
    * The average flux is defined as F̄(u1,u2) = ∫ F(u) du / (u2 - u1) for
    * u ∈ [u1,u2], where u1 is the first state (@a state1) and the u2 the
    * second state (@a state2), while F(u) is the flux as defined in
    * ComputeFlux().
    *
    * Used in the default implementation of ComputeAvgFluxDotN().
    * @param[in] state1 state of the beginning of the interval (num_equations)
    * @param[in] state2 state of the end of the interval (num_equations)
    * @param[in] Tr element transformation
    * @param[out] flux_ average flux from the given element at the current
    * integration point (num_equations, dim)
    * @return real_t maximum characteristic speed |dF(u,x)/du| over
    * the interval [u1,u2]
    */
   virtual real_t ComputeAvgFlux(const Vector &state1, const Vector &state2,
                                 ElementTransformation &Tr,
                                 DenseMatrix &flux_) const
   { MFEM_ABORT("Not Implemented."); }

   /**
    * @brief Compute average normal flux over the given interval of states.
    * Optionally overloaded in a derived class.
    *
    * The average normal flux is defined as F̄(u1,u2)n = ∫ F(u)n du / (u2 - u1)
    * for u ∈ [u1,u2], where u1 is the first state (@a state1) and the u2 the
    * second state (@a state2), while n is the normal and F(u) is the flux as
    * defined in ComputeFlux().
    *
    * Used in NumericalFlux::Average() and NumericalFlux::AverageGrad() for
    * evaluation of the average normal flux on a face.
    * @param[in] state1 state of the beginning of the interval (num_equations)
    * @param[in] state2 state of the end of the interval (num_equations)
    * @param[in] normal normal vector, see mfem::CalcOrtho() (dim)
    * @param[in] Tr face transformation
    * @param[out] fluxDotN average normal flux from the given element at the
    * current integration point (num_equations)
    * @return real_t maximum (normal) characteristic speed |dF(u,x)/du⋅n|
    * over the interval [u1,u2]
    */
   virtual real_t ComputeAvgFluxDotN(const Vector &state1, const Vector &state2,
                                     const Vector &normal,
                                     FaceElementTransformations &Tr,
                                     Vector &fluxDotN) const;

   /**
    * @brief Compute flux Jacobian J(u, x). Optionally overloaded in a derived
    * class when Jacobian is necessary (e.g. Newton iteration, flux limiter)
    *
    * Used in HyperbolicFormIntegrator::AssembleElementGrad() for evaluation of
    * Jacobian of the flux in an element and in the default implementation of
    * ComputeFluxJacobianDotN().
    * @param[in] state state at the current integration point (num_equations)
    * @param[in] Tr element transformation
    * @param[out] J_ flux Jacobian, $ J(i,j,d) = dF_{id} / du_j $
    */
   virtual void ComputeFluxJacobian(const Vector &state,
                                    ElementTransformation &Tr,
                                    DenseTensor &J_) const
   { MFEM_ABORT("Not Implemented."); }

   /**
    * @brief Compute normal flux Jacobian J(u, x)⋅n. Optionally overloaded in
    * a derived class to avoid creating a full dense tensor for Jacobian.
    *
    * Used in NumericalFlux for evaluation of Jacobian of the normal flux on
    * a face.
    * @param[in] state state at the current integration point (num_equations)
    * @param[in] normal normal vector, see mfem::CalcOrtho() (dim)
    * @param[in] Tr element transformation
    * @param[out] JDotN normal flux Jacobian, $ JDotN(i,j) = d(F_{id} n_d) / du_j $
    */
   virtual void ComputeFluxJacobianDotN(const Vector &state,
                                        const Vector &normal,
                                        ElementTransformation &Tr,
                                        DenseMatrix &JDotN) const;

private:
#ifndef MFEM_THREAD_SAFE
   mutable DenseMatrix flux;
   mutable DenseTensor J;
#endif
};


/**
 * @brief Abstract class for numerical flux for a system of hyperbolic
 * conservation laws on a face with states, fluxes and characteristic speed
 *
 */
class NumericalFlux
{
public:
   /**
    * @brief Constructor for a flux function
    * @param fluxFunction flux function F(u,x)
    */
   NumericalFlux(const FluxFunction &fluxFunction)
      : fluxFunction(fluxFunction) { }

   /**
    * @brief Evaluates normal numerical flux for the given states and normal.
    * Must be implemented in a derived class.
    *
    * Used in HyperbolicFormIntegrator::AssembleFaceVector() for evaluation of
    * <F̂(u⁻,u⁺,x) n, [v]> term at the face.
    * @param[in] state1 state value at a point from the first element
    * (num_equations)
    * @param[in] state2 state value at a point from the second element
    * (num_equations)
    * @param[in] nor scaled normal vector, see mfem::CalcOrtho() (dim)
    * @param[in] Tr face transformation
    * @param[out] flux numerical flux (num_equations)
    * @return real_t maximum characteristic speed |dF(u,x)/du⋅n|
    */
   virtual real_t Eval(const Vector &state1, const Vector &state2,
                       const Vector &nor, FaceElementTransformations &Tr,
                       Vector &flux) const = 0;

   /**
    * @brief Evaluates Jacobian of the normal numerical flux for the given
    * states and normal. Optionally overloaded in a derived class.
    *
    * Used in HyperbolicFormIntegrator::AssembleFaceGrad() for Jacobian
    * of the term <F̂(u⁻,u⁺,x) n, [v]> at the face.
    * @param[in] side indicates gradient w.r.t. the first (side = 1)
    * or second (side = 2) state
    * @param[in] state1 state value of the beginning of the interval
    * (num_equations)
    * @param[in] state2 state value of the end of the interval
    * (num_equations)
    * @param[in] nor scaled normal vector, see mfem::CalcOrtho() (dim)
    * @param[in] Tr face transformation
    * @param[out] grad Jacobian of normal numerical flux (num_equations, dim)
    */
   virtual void Grad(int side, const Vector &state1, const Vector &state2,
                     const Vector &nor, FaceElementTransformations &Tr,
                     DenseMatrix &grad) const
   { MFEM_ABORT("Not implemented."); }

   /**
    * @brief Evaluates average normal numerical flux over the interval between
    * the given end states in the second argument and for the given normal.
    * Optionally overloaded in a derived class.
    *
    * Presently, not used. Reserved for future use.
    * @param[in] state1 state value of the beginning of the interval
    * (num_equations)
    * @param[in] state2 state value of the end of the interval
    * (num_equations)
    * @param[in] nor scaled normal vector, see mfem::CalcOrtho() (dim)
    * @param[in] Tr face transformation
    * @param[out] flux numerical flux (num_equations)
    * @return real_t maximum characteristic speed |dF(u,x)/du⋅n|
    */
   virtual real_t Average(const Vector &state1, const Vector &state2,
                          const Vector &nor, FaceElementTransformations &Tr,
                          Vector &flux) const
   { MFEM_ABORT("Not implemented."); }

   /**
    * @brief Evaluates Jacobian of the average normal numerical flux over the
    * interval between the given end states in the second argument and for the
    * given normal. Optionally overloaded in a derived class.
    *
    * Presently, not used. Reserved for future use.
    * @param[in] side indicates gradient w.r.t. the first (side = 1)
    * or second (side = 2) state
    * @param[in] state1 state value of the beginning of the interval
    * (num_equations)
    * @param[in] state2 state value of the end of the interval
    * (num_equations)
    * @param[in] nor scaled normal vector, see mfem::CalcOrtho() (dim)
    * @param[in] Tr face transformation
    * @param[out] grad Jacobian of the average normal numerical flux
    * (num_equations, dim)
    */
   virtual void AverageGrad(int side, const Vector &state1, const Vector &state2,
                            const Vector &nor, FaceElementTransformations &Tr,
                            DenseMatrix &grad) const
   { MFEM_ABORT("Not implemented."); }

   virtual ~NumericalFlux() = default;

   /// @brief Get flux function F
   /// @return constant reference to the flux function.
   const FluxFunction &GetFluxFunction() const { return fluxFunction; }

protected:
   const FluxFunction &fluxFunction;
};

/// @deprecated Use NumericalFlux instead.
MFEM_DEPRECATED typedef NumericalFlux RiemannSolver;

/**
 * @brief Abstract hyperbolic form integrator, assembling (F(u, x), ∇v) and
 * <F̂(u⁻,u⁺,x) n, [v]> terms for scalar finite elements.
 *
 * This form integrator is coupled with a NumericalFlux that implements the
 * numerical flux F̂ at the faces. The flux F is obtained from the FluxFunction
 * assigned to the aforementioned NumericalFlux.
 */
class HyperbolicFormIntegrator : public NonlinearFormIntegrator
{
private:
   const NumericalFlux &numFlux;   // Numerical flux that maps F(u±,x) to F̂
   const FluxFunction &fluxFunction;
   const int IntOrderOffset; // integration order offset, 2*p + IntOrderOffset.
   const real_t sign;

   // The maximum characteristic speed, updated during element/face vector assembly
   real_t max_char_speed;

#ifndef MFEM_THREAD_SAFE
   // Local storage for element integration
   Vector shape;              // shape function value at an integration point
   Vector state;              // state value at an integration point
   DenseMatrix flux;          // flux value at an integration point
   DenseTensor J;             // Jacobian matrix at an integration point
   DenseMatrix dshape;  // derivative of shape function at an integration point

   Vector shape1;  // shape function value at an integration point - first elem
   Vector shape2;  // shape function value at an integration point - second elem
   Vector state1;  // state value at an integration point - first elem
   Vector state2;  // state value at an integration point - second elem
   Vector nor;     // normal vector, see mfem::CalcOrtho()
   Vector fluxN;   // F̂(u±,x) n
   DenseMatrix JDotN;   // Ĵ(u±,x) n
#endif

public:
   const int num_equations;  // the number of equations

   /**
    * @brief Construct a new HyperbolicFormIntegrator object
    *
    * @param[in] numFlux numerical flux
    * @param[in] IntOrderOffset integration order offset
    * @param[in] sign sign of the convection term
    */
   HyperbolicFormIntegrator(
      const NumericalFlux &numFlux,
      const int IntOrderOffset = 0,
      const real_t sign = 1.);

   /// Reset the maximum characteristic speed to zero
   void ResetMaxCharSpeed() { max_char_speed = 0.0; }

   /// Get the maximum characteristic speed
   real_t GetMaxCharSpeed() const { return max_char_speed; }

   /// Get the associated flux function
   const FluxFunction &GetFluxFunction() const { return fluxFunction; }

   /**
    * @brief Implements (F(u), ∇v) with abstract F computed by
    * FluxFunction::ComputeFlux()
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
    * @brief Implements (J(u), ∇v) with abstract J computed by
    * FluxFunction::ComputeFluxJacobian()
    *
    * @param[in] el local finite element
    * @param[in] Tr element transformation
    * @param[in] elfun local coefficient of basis
    * @param[out] grad evaluated Jacobian
    */
   void AssembleElementGrad(const FiniteElement &el,
                            ElementTransformation &Tr,
                            const Vector &elfun, DenseMatrix &grad) override;

   /**
    * @brief Implements <-F̂(u⁻,u⁺,x) n, [v]> with abstract F̂ computed by
    * NumericalFlux::Eval() of the numerical flux object
    *
    * @param[in] el1 finite element of the first element
    * @param[in] el2 finite element of the second element
    * @param[in] Tr face element transformations
    * @param[in] elfun local coefficient of basis from both elements
    * @param[out] elvect evaluated dual vector <-F̂(u⁻,u⁺,x) n, [v]>
    */
   void AssembleFaceVector(const FiniteElement &el1,
                           const FiniteElement &el2,
                           FaceElementTransformations &Tr,
                           const Vector &elfun, Vector &elvect) override;

   /**
    * @brief Implements <-Ĵ(u⁻,u⁺,x) n, [v]> with abstract Ĵ computed by
    * NumericalFlux::Grad() of the numerical flux object
    *
    * @param[in] el1 finite element of the first element
    * @param[in] el2 finite element of the second element
    * @param[in] Tr face element transformations
    * @param[in] elfun local coefficient of basis from both elements
    * @param[out] elmat evaluated Jacobian matrix <-Ĵ(u⁻,u⁺,x) n, [v]>
    */
   void AssembleFaceGrad(const FiniteElement &el1,
                         const FiniteElement &el2,
                         FaceElementTransformations &Tr,
                         const Vector &elfun, DenseMatrix &elmat) override;
};

/**
 * @brief Abstract boundary hyperbolic form integrator, assembling
 * <F̂(u⁻,u_b,x) n, [v]> term for scalar finite elements at the boundary.
 *
 * This form integrator is coupled with a NumericalFlux that implements the
 * numerical flux F̂ at the boundary faces. The flux F is obtained from the
 * FluxFunction assigned to the aforementioned NumericalFlux with the given
 * boundary coefficient for the state u_b.
 *
 * Note the class can be used for imposing conditions on interior interfaces.
 */
class BdrHyperbolicDirichletIntegrator : public NonlinearFormIntegrator
{
private:
   const NumericalFlux &numFlux;    // Numerical flux that maps F to F̂
   const FluxFunction &fluxFunction;
   VectorCoefficient &u_vcoeff;     // Boundary state vector coefficient
   const int IntOrderOffset; // integration order offset, 2*p + IntOrderOffset.
   const real_t sign;

   // The maximum characteristic speed, updated during element/face vector assembly
   real_t max_char_speed;

#ifndef MFEM_THREAD_SAFE
   // Local storage for element integration
   Vector shape;  // shape function value at an integration point
   Vector state_in;  // state value at an integration point - interior
   Vector state_out;  // state value at an integration point - boundary
   Vector nor;     // normal vector, see mfem::CalcOrtho()
   Vector fluxN;   // F̂(u⁻,u_b,x) n
   DenseMatrix JDotN;   // Ĵ(u⁻,u_b,x) n
#endif

public:
   const int num_equations;  // the number of equations

   /**
    * @brief Construct a new BdrHyperbolicDirichletIntegrator object
    *
    * @param[in] numFlux numerical flux
    * @param[in] bdrState boundary state coefficient
    * @param[in] IntOrderOffset integration order offset
    * @param[in] sign sign of the convection term
    */
   BdrHyperbolicDirichletIntegrator(
      const NumericalFlux &numFlux,
      VectorCoefficient &bdrState,
      const int IntOrderOffset = 0,
      const real_t sign = 1.);

   /// Reset the maximum characteristic speed to zero
   void ResetMaxCharSpeed() { max_char_speed = 0.0; }

   /// Get the maximum characteristic speed
   real_t GetMaxCharSpeed() const { return max_char_speed; }

   /// Get the associated flux function
   const FluxFunction &GetFluxFunction() const { return fluxFunction; }

   /**
    * @brief Implements <-F̂(u⁻,u_b,x) n, [v]> with abstract F̂ computed by
    * NumericalFlux::Eval() of the numerical flux object
    *
    * @param[in] el1 finite element of the interior element
    * @param[in] el2 not used
    * @param[in] Tr face element transformations
    * @param[in] elfun local coefficient of basis for the interior element
    * @param[out] elvect evaluated dual vector <-F̂(u⁻,u_b,x) n, [v]>
    */
   void AssembleFaceVector(const FiniteElement &el1,
                           const FiniteElement &el2,
                           FaceElementTransformations &Tr,
                           const Vector &elfun, Vector &elvect) override;

   /**
    * @brief Implements <-Ĵ(u⁻,u_b,x) n, [v]> with abstract Ĵ computed by
    * NumericalFlux::Grad() of the numerical flux object
    *
    * @param[in] el1 finite element of the interior element
    * @param[in] el2 not used
    * @param[in] Tr face element transformations
    * @param[in] elfun local coefficient of basis for the interior element
    * @param[out] elmat evaluated Jacobian matrix <-Ĵ(u⁻,u_b,x) n, [v]>
    */
   void AssembleFaceGrad(const FiniteElement &el1,
                         const FiniteElement &el2,
                         FaceElementTransformations &Tr,
                         const Vector &elfun, DenseMatrix &elmat) override;
};

/**
 * @brief Abstract boundary hyperbolic linear form integrator, assembling
 * <ɑ/2 F(u,x) n - β |F(u,x) n|, v> terms for scalar finite elements.
 *
 * This form integrator is coupled with a FluxFunction that evaluates the
 * flux F at the boundary.
 *
 * Note the upwinding is performed component-wise. For general boundary
 * integration with a numerical flux, see BdrHyperbolicDirichletIntegrator.
 */
class BoundaryHyperbolicFlowIntegrator : public LinearFormIntegrator
{
   const FluxFunction &fluxFunction;
   VectorCoefficient &u_vcoeff;
   const real_t alpha, beta;
   const int IntOrderOffset; // integration order offset, 2*p + IntOrderOffset.

   // The maximum characteristic speed, updated during face vector assembly
   real_t max_char_speed;

#ifndef MFEM_THREAD_SAFE
   // Local storage for element integration
   Vector shape;   // shape function value at an integration point
   Vector state;   // state value at an integration point
   Vector nor;     // normal vector, see mfem::CalcOrtho()
   Vector fluxN;   // F(u,x) n
#endif

public:
   /**
    * @brief Construct a new BoundaryHyperbolicFlowIntegrator object
    *
    * @param[in] flux flux function
    * @param[in] u vector state coefficient
    * @param[in] alpha ɑ coefficient (β = ɑ/2)
    * @param[in] IntOrderOffset integration order offset
    */
   BoundaryHyperbolicFlowIntegrator(const FluxFunction &flux, VectorCoefficient &u,
                                    real_t alpha = -1., int IntOrderOffset = 0)
      : BoundaryHyperbolicFlowIntegrator(flux, u, alpha, alpha/2., IntOrderOffset) { }

   /**
    * @brief Construct a new BoundaryHyperbolicFlowIntegrator object
    *
    * @param[in] flux flux function
    * @param[in] u vector state coefficient
    * @param[in] alpha ɑ coefficient
    * @param[in] beta β coefficient
    * @param[in] IntOrderOffset integration order offset
    */
   BoundaryHyperbolicFlowIntegrator(const FluxFunction &flux, VectorCoefficient &u,
                                    real_t alpha, real_t beta, int IntOrderOffset = 0);

   /// Reset the maximum characteristic speed to zero
   void ResetMaxCharSpeed() { max_char_speed = 0.0; }

   /// Get the maximum characteristic speed
   real_t GetMaxCharSpeed() const { return max_char_speed; }

   /// Get the associated flux function
   const FluxFunction &GetFluxFunction() const { return fluxFunction; }

   using LinearFormIntegrator::AssembleRHSElementVect;

   /**
    * @warning Boundary element integration not implemented, use
    * AssembleRHSElementVect(const FiniteElement&,
    * FaceElementTransformations &, Vector &) instead
    */
   void AssembleRHSElementVect(const FiniteElement &el,
                               ElementTransformation &Tr,
                               Vector &elvect) override;

   /**
    * @brief Implements <-F(u,x) n, v> with abstract F computed by
    * FluxFunction::ComputeFluxDotN() of the flux function object
    *
    * @param[in] el finite element
    * @param[in] Tr face element transformations
    * @param[out] elvect evaluated dual vector <F(u,x) n, v>
    */
   void AssembleRHSElementVect(const FiniteElement &el,
                               FaceElementTransformations &Tr,
                               Vector &elvect) override;
};


/**
 * @brief Rusanov flux, also known as local Lax-Friedrichs,
 *    F̂ n = ½(F(u⁺,x)n + F(u⁻,x)n) - ½λ(u⁺ - u⁻)
 * where λ is the maximum characteristic speed.
 * @note The implementation assumes monotonous |dF(u,x)/du⋅n| in u, so the
 * maximum characteristic speed λ for any interval [u⁻, u⁺] is given by
 * max(|dF(u⁺,x)/du⁺⋅n|, |dF(u⁻,x)/du⁻⋅n|).
 */
class RusanovFlux : public NumericalFlux
{
public:
   /**
    * @brief Constructor for a flux function
    * @param fluxFunction flux function F(u,x)
    */
   RusanovFlux(const FluxFunction &fluxFunction);

   /**
    * @brief  Normal numerical flux F̂(u⁻,u⁺,x) n
    * @note Systems of equations are treated component-wise
    *
    * @param[in] state1 state value (u⁻) at a point from the first element
    * (num_equations)
    * @param[in] state2 state value (u⁺) at a point from the second element
    * (num_equations)
    * @param[in] nor normal vector (not a unit vector) (dim)
    * @param[in] Tr face element transformation
    * @param[out] flux F̂ n = ½(F(u⁺,x)n + F(u⁻,x)n) - ½λ(u⁺ - u⁻)
    * @return max(|dF(u⁺,x)/du⁺⋅n|, |dF(u⁻,x)/du⁻⋅n|)
    */
   real_t Eval(const Vector &state1, const Vector &state2,
               const Vector &nor, FaceElementTransformations &Tr,
               Vector &flux) const override;

   /**
    * @brief  Jacobian of normal numerical flux F̂(u⁻,u⁺,x) n
    * @note The Jacobian of flux J n is required to be implemented in
    * FluxFunction::ComputeFluxJacobianDotN()
    *
    * @param[in] side gradient w.r.t the first (u⁻) or second argument (u⁺)
    * @param[in] state1 state value (u⁻) of the beginning of the interval
    * (num_equations)
    * @param[in] state2 state value (u⁺) of the end of the interval
    * (num_equations)
    * @param[in] nor normal vector (not a unit vector) (dim)
    * @param[in] Tr face element transformation
    * @param[out] grad Jacobian of F(u⁻,u⁺,x) n
    * side = 1:
    *    ½J(u⁻,x)n + ½λ
    * side = 2:
    *    ½J(u⁺,x)n - ½λ
    */
   void Grad(int side, const Vector &state1, const Vector &state2,
             const Vector &nor, FaceElementTransformations &Tr,
             DenseMatrix &grad) const override;

   /**
    * @brief  Average normal numerical flux over the interval [u⁻, u⁺] in the
    * second argument of the flux F̂(u⁻,u,x) n
    * @note The average normal flux F̄ n is required to be implemented in
    * FluxFunction::ComputeAvgFluxDotN()
    * @note Systems of equations are treated component-wise
    *
    * @param[in] state1 state value (u⁻) of the beginning of the interval
    * (num_equations)
    * @param[in] state2 state value (u⁺) of the end of the interval
    * (num_equations)
    * @param[in] nor normal vector (not a unit vector) (dim)
    * @param[in] Tr face element transformation
    * @param[out] flux ½(F̄(u⁻,u⁺,x)n + F(u⁻,x)n) - ¼λ(u⁺ - u⁻)
    * @return max(|dF(u⁺,x)/du⁺⋅n|, |dF(u⁻,x)/du⁻⋅n|)
    */
   real_t Average(const Vector &state1, const Vector &state2,
                  const Vector &nor, FaceElementTransformations &Tr,
                  Vector &flux) const override;

   /**
    * @brief  Jacobian of average normal numerical flux over the interval
    * [u⁻, u⁺] in the second argument of the flux F̂(u⁻,u,x) n
    * @note The average normal flux F̄ n is required to be implemented in
    * FluxFunction::ComputeAvgFluxDotN() and the Jacobian of flux J n in
    * FluxFunction::ComputeFluxJacobianDotN()
    * @note Only the diagonal terms of the J n are considered, i.e., systems
    * are treated as a set of independent equations
    *
    * @param[in] side gradient w.r.t the first (u⁻) or second argument (u⁺)
    * @param[in] state1 state value (u⁻) of the beginning of the interval
    * (num_equations)
    * @param[in] state2 state value (u⁺) of the end of the interval
    * (num_equations)
    * @param[in] nor normal vector (not a unit vector) (dim)
    * @param[in] Tr face element transformation
    * @param[out] grad Jacobian of F̄(u⁻,u⁺,x) n
    * side = 1:
    *    ½(F̄(u⁻,u⁺,x)n - F(u⁻,x)n) / (u⁺ - u⁻) - ½J(u⁻,x)n + ¼λ
    * side = 2:
    *    ½(F(u⁺,x)n - F̄(u⁻,u⁺,x)n) / (u⁺ - u⁻) - ¼λ
    */
   void AverageGrad(int side, const Vector &state1, const Vector &state2,
                    const Vector &nor, FaceElementTransformations &Tr,
                    DenseMatrix &grad) const override;

protected:
#ifndef MFEM_THREAD_SAFE
   mutable Vector fluxN1, fluxN2;
   mutable DenseMatrix JDotN;
#endif
};

/**
 * @brief Component-wise upwinded flux
 *
 * Upwinded flux for scalar equations, a special case of Godunov or
 * Engquist-Osher flux, is defined as follows:
 *    F̂ n = F(u⁺)n    for dF(u)/du < 0 on [u⁻,u⁺]
 *    F̂ n = F(u⁻)n    for dF(u)/du > 0 on [u⁻,u⁺]
 * @note This construction assumes monotonous F(u,x) in u
 * @note Systems of equations are treated component-wise
 */
class ComponentwiseUpwindFlux : public NumericalFlux
{
public:
   /**
    * @brief Constructor for a flux function
    * @param fluxFunction flux function F(u,x)
    */
   ComponentwiseUpwindFlux(const FluxFunction &fluxFunction);

   /**
    * @brief  Normal numerical flux F̂(u⁻,u⁺,x) n
    *
    * @param[in] state1 state value (u⁻) at a point from the first element
    * (num_equations)
    * @param[in] state2 state value (u⁺) at a point from the second element
    * (num_equations)
    * @param[in] nor normal vector (not a unit vector) (dim)
    * @param[in] Tr face element transformation
    * @param[out] flux F̂ n = min(F(u⁻,x)n, F(u⁺,x)n)    for u⁻ ≤ u⁺
    *               or F̂ n = max(F(u⁻,x)n, F(u⁺,x)n)    for u⁻ > u⁺
    * @return max(|dF(u⁺,x)/du⁺⋅n|, |dF(u⁻,x)/du⁻⋅n|)
    */
   real_t Eval(const Vector &state1, const Vector &state2,
               const Vector &nor, FaceElementTransformations &Tr,
               Vector &flux) const override;

   /**
    * @brief  Jacobian of normal numerical flux F̂(u⁻,u⁺,x) n
    * @note The Jacobian of flux J n is required to be implemented in
    * FluxFunction::ComputeFluxJacobianDotN()
    *
    * @param[in] side gradient w.r.t the first (u⁻) or second argument (u⁺)
    * @param[in] state1 state value (u⁻) of the beginning of the interval
    * (num_equations)
    * @param[in] state2 state value (u⁺) of the end of the interval
    * (num_equations)
    * @param[in] nor normal vector (not a unit vector) (dim)
    * @param[in] Tr face element transformation
    * @param[out] grad Jacobian of F(u⁻,u⁺,x) n
    * side = 1:
    *    max(J(u⁻,x)n, 0)
    * side = 2:
    *    min(J(u⁺,x)n, 0)
    */
   void Grad(int side, const Vector &state1, const Vector &state2,
             const Vector &nor, FaceElementTransformations &Tr,
             DenseMatrix &grad) const override;

   /**
    * @brief  Average normal numerical flux over the interval [u⁻, u⁺] in the
    * second argument of the flux F̂(u⁻,u,x) n
    * @note The average normal flux F̄ n is required to be implemented in
    * FluxFunction::ComputeAvgFluxDotN()
    *
    * @param[in] state1 state value (u⁻) of the beginning of the interval
    * (num_equations)
    * @param[in] state2 state value (u⁺) of the end of the interval
    * (num_equations)
    * @param[in] nor normal vector (not a unit vector) (dim)
    * @param[in] Tr face element transformation
    * @param[out] flux F̂ n = min(F(u⁻)n, F̄(u⁺,x)n)    for u⁻ ≤ u⁺
    *               or F̂ n = max(F(u⁻)n, F̄(u⁺,x)n)    for u⁻ > u⁺
    * @return max(|dF(u⁺,x)/du⁺⋅n|, |dF(u⁻,x)/du⁻⋅n|)
    */
   real_t Average(const Vector &state1, const Vector &state2,
                  const Vector &nor, FaceElementTransformations &Tr,
                  Vector &flux) const override;

   /**
    * @brief  Jacobian of average normal numerical flux over the interval
    * [u⁻, u⁺] in the second argument of the flux F̂(u⁻,u,x) n
    * @note The average normal flux F̄ n is required to be implemented in
    * FluxFunction::ComputeAvgFluxDotN() and the Jacobian of flux J n in
    * FluxFunction::ComputeFluxJacobianDotN()
    *
    * @param[in] side gradient w.r.t the first (u⁻) or second argument (u⁺)
    * @param[in] state1 state value (u⁻) of the beginning of the interval
    * (num_equations)
    * @param[in] state2 state value (u⁺) of the end of the interval
    * (num_equations)
    * @param[in] nor normal vector (not a unit vector) (dim)
    * @param[in] Tr face element transformation
    * @param[out] grad Jacobian of F̄(u⁻,u⁺,x) n
    * side = 1:
    *    (F(u⁺) - F̄(u⁻,u⁺))n / (u⁺ - u⁻)      when negative
    *    J(u⁻,x) n                            otherwise
    * side = 2:
    *    min((F(u⁺) - F̄(u⁻,u⁺))n / (u⁺ - u⁻), 0)
    */
   void AverageGrad(int side, const Vector &state1, const Vector &state2,
                    const Vector &nor, FaceElementTransformations &Tr,
                    DenseMatrix &grad) const override;

protected:
#ifndef MFEM_THREAD_SAFE
   mutable Vector fluxN1, fluxN2;
   mutable DenseMatrix JDotN;
#endif
};

/// Advection flux
class AdvectionFlux : public FluxFunction
{
private:
   VectorCoefficient &b;  // velocity coefficient
#ifndef MFEM_THREAD_SAFE
   mutable Vector bval;           // velocity value storage
#endif

public:

   /**
    * @brief Construct AdvectionFlux FluxFunction with given velocity
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
    * @param Tr current element transformation with the integration point
    * @param flux F(u) = ubᵀ
    * @return real_t maximum characteristic speed, |b|
    */
   real_t ComputeFlux(const Vector &state, ElementTransformation &Tr,
                      DenseMatrix &flux) const override;

   /**
    * @brief Compute F(u) n
    *
    * @param state state (u) at current integration point
    * @param normal normal vector, usually not a unit vector
    * @param Tr current element transformation with the integration point
    * @param fluxDotN F(u) n = u (bᵀn)
    * @return real_t maximum characteristic speed, |b|
    */
   real_t ComputeFluxDotN(const Vector &state,
                          const Vector &normal, FaceElementTransformations &Tr,
                          Vector &fluxDotN) const override;

   /**
    * @brief Compute average flux F̄(u)
    *
    * @param state1 state value (u⁻) of the beginning of the interval
    * @param state2 state value (u⁺) of the end of the interval
    * @param Tr current element transformation with the integration point
    * @param flux F̄(u) = (u⁻+u⁺)/2*bᵀ
    * @return real_t maximum characteristic speed, |b|
    */
   real_t ComputeAvgFlux(const Vector &state1, const Vector &state2,
                         ElementTransformation &Tr, DenseMatrix &flux) const override;

   /**
    * @brief Compute average flux F̄(u) n
    *
    * @param state1 state value (u⁻) of the beginning of the interval
    * @param state2 state value (u⁺) of the end of the interval
    * @param normal normal vector, usually not a unit vector
    * @param Tr current element transformation with the integration point
    * @param fluxDotN F̄(u) n = (u⁻+u⁺)/2*(bᵀn)
    * @return real_t maximum characteristic speed, |b|
    */
   real_t ComputeAvgFluxDotN(const Vector &state1, const Vector &state2,
                             const Vector &normal, FaceElementTransformations &Tr,
                             Vector &fluxDotN) const override;

   /**
    * @brief Compute J(u)
    *
    * @param state state (u) at current integration point
    * @param Tr current element transformation with the integration point
    * @param J J(u) = diag(b)
    */
   void ComputeFluxJacobian(const Vector &state,
                            ElementTransformation &Tr,
                            DenseTensor &J) const override;

   /**
    * @brief Compute J(u) n
    *
    * @param state state (u) at current integration point
    * @param normal normal vector, usually not a unit vector
    * @param Tr current element transformation with the integration point
    * @param JDotN J(u) n = bᵀn
    */
   void ComputeFluxJacobianDotN(const Vector &state,
                                const Vector &normal,
                                ElementTransformation &Tr,
                                DenseMatrix &JDotN) const override;
};

/// Burgers flux
class BurgersFlux : public FluxFunction
{
public:
   /**
    * @brief Construct BurgersFlux FluxFunction with given spatial dimension
    *
    * @param dim spatial dimension
    */
   BurgersFlux(const int dim)
      : FluxFunction(1, dim) {}

   /**
    * @brief Compute F(u)
    *
    * @param state state (u) at current integration point
    * @param Tr current element transformation with the integration point
    * @param flux F(u) = ½u²*1ᵀ where 1 is (dim) vector
    * @return real_t maximum characteristic speed, |u|
    */
   real_t ComputeFlux(const Vector &state, ElementTransformation &Tr,
                      DenseMatrix &flux) const override;

   /**
    * @brief Compute F(u) n
    *
    * @param state state (u) at current integration point
    * @param normal normal vector, usually not a unit vector
    * @param Tr current element transformation with the integration point
    * @param fluxDotN F(u) n = ½u²*(1ᵀn) where 1 is (dim) vector
    * @return real_t maximum characteristic speed, |u|
    */
   real_t ComputeFluxDotN(const Vector &state,
                          const Vector &normal,
                          FaceElementTransformations &Tr,
                          Vector &fluxDotN) const override;

   /**
    * @brief Compute average flux F̄(u)
    *
    * @param state1 state value (u⁻) of the beginning of the interval
    * @param state2 state value (u⁺) of the end of the interval
    * @param Tr current element transformation with the integration point
    * @param flux F̄(u) = (u⁻²+u⁻*u⁺+u⁺²)/6*1ᵀ where 1 is (dim) vector
    * @return real_t maximum characteristic speed, |u|
    */
   real_t ComputeAvgFlux(const Vector &state1,
                         const Vector &state2,
                         ElementTransformation &Tr,
                         DenseMatrix &flux) const override;

   /**
    * @brief Compute average flux F̄(u) n
    *
    * @param state1 state value (u⁻) of the beginning of the interval
    * @param state2 state value (u⁺) of the end of the interval
    * @param normal normal vector, usually not a unit vector
    * @param Tr current element transformation with the integration point
    * @param fluxDotN F̄(u) n = (u⁻²+u⁻*u⁺+u⁺²)/6*(1ᵀn) where 1 is (dim) vector
    * @return real_t maximum characteristic speed, |u|
    */
   real_t ComputeAvgFluxDotN(const Vector &state1,
                             const Vector &state2,
                             const Vector &normal,
                             FaceElementTransformations &Tr,
                             Vector &fluxDotN) const override;

   /**
    * @brief Compute J(u)
    *
    * @param state state (u) at current integration point
    * @param Tr current element transformation with the integration point
    * @param J J(u) = diag(u*1) where 1 is (dim) vector
    */
   void ComputeFluxJacobian(const Vector &state,
                            ElementTransformation &Tr,
                            DenseTensor &J) const override;

   /**
    * @brief Compute J(u) n
    *
    * @param state state (u) at current integration point
    * @param normal normal vector, usually not a unit vector
    * @param Tr current element transformation with the integration point
    * @param JDotN J(u) n = u*(1ᵀn) where 1 is (dim) vector
    */
   void ComputeFluxJacobianDotN(const Vector &state,
                                const Vector &normal,
                                ElementTransformation &Tr,
                                DenseMatrix &JDotN) const override;
};

/// Shallow water flux
class ShallowWaterFlux : public FluxFunction
{
private:
   const real_t g;  // gravity constant

public:
   /**
    * @brief Construct a new ShallowWaterFlux FluxFunction with given spatial
    * dimension and gravity constant
    *
    * @param dim spatial dimension
    * @param g gravity constant
    */
   ShallowWaterFlux(const int dim, const real_t g=9.8)
      : FluxFunction(dim + 1, dim), g(g) {}

   /**
    * @brief Compute F(h, hu)
    *
    * @param state state (h, hu) at current integration point
    * @param Tr current element transformation with the integration point
    * @param flux F(h, hu) = [huᵀ; huuᵀ + ½gh²I]
    * @return real_t maximum characteristic speed, |u| + √(gh)
    */
   real_t ComputeFlux(const Vector &state, ElementTransformation &Tr,
                      DenseMatrix &flux) const override;

   /**
    * @brief Compute normal flux, F(h, hu)
    *
    * @param state state (h, hu) at current integration point
    * @param normal normal vector, usually not a unit vector
    * @param Tr current element transformation with the integration point
    * @param fluxN F(ρ, ρu, E)n = [ρu⋅n; ρu(u⋅n) + pn; (u⋅n)(E + p)]
    * @return real_t maximum characteristic speed, |u| + √(γp/ρ)
    */
   real_t ComputeFluxDotN(const Vector &state, const Vector &normal,
                          FaceElementTransformations &Tr,
                          Vector &fluxN) const override;
};

/// Euler flux
class EulerFlux : public FluxFunction
{
private:
   const real_t specific_heat_ratio;  // specific heat ratio, γ
   // const real_t gas_constant;         // gas constant

public:
   /**
    * @brief Construct a new EulerFlux FluxFunction with given spatial
    * dimension and specific heat ratio
    *
    * @param dim spatial dimension
    * @param specific_heat_ratio specific heat ratio, γ
    */
   EulerFlux(const int dim, const real_t specific_heat_ratio)
      : FluxFunction(dim + 2, dim),
        specific_heat_ratio(specific_heat_ratio) {}

   /**
    * @brief Compute F(ρ, ρu, E)
    *
    * @param state state (ρ, ρu, E) at current integration point
    * @param Tr current element transformation with the integration point
    * @param flux F(ρ, ρu, E) = [ρuᵀ; ρuuᵀ + pI; uᵀ(E + p)]
    * @return real_t maximum characteristic speed, |u| + √(γp/ρ)
    */
   real_t ComputeFlux(const Vector &state, ElementTransformation &Tr,
                      DenseMatrix &flux) const override;

   /**
    * @brief Compute normal flux, F(ρ, ρu, E)n
    *
    * @param x x (ρ, ρu, E) at current integration point
    * @param normal normal vector, usually not a unit vector
    * @param Tr current element transformation with the integration point
    * @param fluxN F(ρ, ρu, E)n = [ρu⋅n; ρu(u⋅n) + pn; (u⋅n)(E + p)]
    * @return real_t maximum characteristic speed, |u| + √(γp/ρ)
    */
   real_t ComputeFluxDotN(const Vector &x, const Vector &normal,
                          FaceElementTransformations &Tr,
                          Vector &fluxN) const override;
};

} // namespace mfem

#endif // MFEM_HYPERBOLIC
