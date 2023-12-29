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


#ifndef MFEM_HYPERBOLIC_CONSERVATION_LAWS
#define MFEM_HYPERBOLIC_CONSERVATION_LAWS

#include "nonlinearform.hpp"
#ifdef MFEM_USE_MPI
#include "pnonlinearform.hpp"
#endif
namespace mfem
{
//           MFEM Hyperbolic Conservation Laws Serial/Parallel Shared Class
//
// Description:  This file contains general hyperbolic conservation element/face
//               form integrators.
//
//               Abstract RiemannSolver, HyperbolicFormIntegrator, and
//               DGHyperbolicConservationLaws are defined. Also, several example
//               hyperbolic equations are defined; including advection, burgers,
//               shallow water, and euler equations.
//
//               To implement a specific hyperbolic conservation laws, users can
//               create derived classes from HyperbolicFormIntegrator with overloaded
//               ComputeFlux. One can optionally overload ComputeFluxDotN to avoid
//               creating dense matrix when computing normal flux.
//
//               FormIntegrator use high-order quadrature points to implement
//               the given form to handle nonlinear flux function. During the
//               implementation of forms, it updates maximum characteristic
//               speed and collected by DGHyperbolicConservationLaws. The global
//               maximum characteristic speed can be obtained by public method
//               getMaxCharSpeed so that the time step can be determined by CFL
//               condition. Also, resetMaxCharSpeed should be called after each Mult.
//
//               @note For parallel version, users should reduce all maximum
//               characteristic speed from all processes using MPI_Allreduce.
//

/**
 * @brief Abstract class for numerical flux for a system of hyperbolic conservation laws
 * on a face with states, fluxes and characteristic speed
 *
 */
class RiemannSolver
{
public:
   /**
    * @brief Evaluates numerical flux for given states and fluxes. Must be
    * overloaded in a derived class
    *
    * @param[in] state1 state value at a point from the first element
    * (num_equations)
    * @param[in] state2 state value at a point from the second element
    * (num_equations)
    * @param[in] fluxN1 normal flux value at a point from the first element
    * (num_equations x dim)
    * @param[in] fluxN2 normal flux value at a point from the second element
    * (num_equations x dim)
    * @param[in] speed1 characteristic speed from the first element
    * @param[in] speed2 characteristic speed from the second element
    * @param[in] nor scaled normal vector, @see mfem::CalcOrtho (dim)
    * @param[out] flux numerical flux (num_equations)
    */
   virtual void Eval(const Vector &state1, const Vector &state2,
                     const Vector &fluxN1, const Vector &fluxN2,
                     const double speed1, const double speed2,
                     const Vector &nor, Vector &flux) const = 0;
   virtual ~RiemannSolver() = default;
};

/**
 * @brief Abstract hyperbolic form integrator, (F(u, x), ∇v) and (F̂(u±, x, n))
 *
 */
class HyperbolicFormIntegrator : public NonlinearFormIntegrator
{
private:
   const int num_equations;  // the number of equations
   // The maximum characterstic speed, updated during element/face vector assembly
   double max_char_speed;
   const int IntOrderOffset;  // 2*p + IntOrderOffset will be used for quadrature
   const RiemannSolver &rsolver;    // Numerical flux that maps F(u±,x) to hat(F)
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
   Vector nor;     // normal vector (usually not a unit vector)
   Vector fluxN;   // hat(F)(u,x)
#endif

public:
   /**
    * @brief Compute flux F(u, x) for given state u and physical point x
    *
    * @param[in] state value of state at the current integration point
    * @param[in] Tr Transformation to find physical location x of the current
    * integration point
    * @param[out] flux F(u, x)
    * @return double maximum characteristic speed
    *
    * @note One can put assertion in here to detect non-physical solution
    */
   virtual double ComputeFlux(const Vector &state, ElementTransformation &Tr,
                              DenseMatrix &flux) = 0;
   /**
    * @brief Compute normal flux. Optionally overloadded in the
    * derived class to avoid creating full dense matrix for flux.
    *
    * @param[in] U U value at the current integration point
    * @param[in] normal normal vector (usually not a unit vector)
    * @param[in] Tr element transformation
    * @param[out] FUdotN normal flux from the given element at the current
    * integration point
    * @return double maximum characteristic velocity
    */
   virtual double ComputeFluxDotN(const Vector &U, const Vector &normal,
                                  ElementTransformation &Tr, Vector &FUdotN)
   {
      double val = ComputeFlux(U, Tr, flux);
      flux.Mult(normal, FUdotN);
      return val;
   }

   /**
    * @brief Construct a new Hyperbolic Form Integrator object
    *
    * @param[in] rsolver_ numerical flux
    * @param[in] dim physical dimension
    * @param[in] num_equations_ the number of equations
    * @param[in] IntOrderOffset_ 2*p+IntOrderOffset order Gaussian quadrature
    * will be used
    */
   HyperbolicFormIntegrator(const RiemannSolver &rsolver_, const int dim,
                            const int num_equations_,
                            const int IntOrderOffset_ = 3)
      : NonlinearFormIntegrator(),
        num_equations(num_equations_),
        IntOrderOffset(IntOrderOffset_),
        rsolver(rsolver_)
   {
#ifndef MFEM_THREAD_SAFE
      state.SetSize(num_equations);
      flux.SetSize(num_equations, dim);
      state1.SetSize(num_equations);
      state2.SetSize(num_equations);
      fluxN1.SetSize(num_equations);
      fluxN2.SetSize(num_equations);
      fluxN.SetSize(num_equations);
      nor.SetSize(dim);
#endif
   }
   /**
    * @brief Construct an object with a fixed integration rule
    *
    * @param[in] rsolver_ numerical flux
    * @param[in] dim physical dimension
    * @param[in] num_equations_ the number of equations
    * @param[in] ir integration rule to be used
    */
   HyperbolicFormIntegrator(const RiemannSolver &rsolver_, const int dim,
                            const int num_equations_,
                            const IntegrationRule *ir)
      : NonlinearFormIntegrator(ir),
        num_equations(num_equations_),
        IntOrderOffset(0),
        rsolver(rsolver_)
   {
#ifndef MFEM_THREAD_SAFE
      state.SetSize(num_equations);
      flux.SetSize(num_equations, dim);
      state1.SetSize(num_equations);
      state2.SetSize(num_equations);
      fluxN1.SetSize(num_equations);
      fluxN2.SetSize(num_equations);
      fluxN.SetSize(num_equations);
      nor.SetSize(dim);
#endif
   }

   /**
    * @brief Get the element integration rule based on IntOrderOffset, @see
    * AssembleElementVector. Used only when ir is not provided
    *
    * @param[in] el given finite element space
    * @return const IntegrationRule& with order 2*p + IntOrderOffset
    */
   const IntegrationRule &GetRule(const FiniteElement &el)
   {
      int order;
      order = 2 * el.GetOrder() + IntOrderOffset;
      return IntRules.Get(el.GetGeomType(), order);
   }

   /**
    * @brief Get the face integration rule based on IntOrderOffset, @see
    * AssembleFaceVector. Used only when ir is not provided
    *
    * @param[in] trial_fe trial finite element space
    * @param[in] test_fe test finite element space
    * @return const IntegrationRule& with order 2*p + IntOrderOffset
    */
   const IntegrationRule &GetRule(const FiniteElement &trial_fe,
                                  const FiniteElement &test_fe)
   {
      int order;
      order = std::max(trial_fe.GetOrder(), test_fe.GetOrder()) * 2 + IntOrderOffset;
      return IntRules.Get(trial_fe.GetGeomType(), order);
   }

   /**
    * @brief Reset the Max Char Speed 0
    *
    */
   inline void resetMaxCharSpeed()
   {
      max_char_speed = 0.0;
   }

   inline double getMaxCharSpeed()
   {
      return max_char_speed;
   }

   /**
    * @brief Compute element flux F(u,x) to be used in ZZ error estimator
    */
   virtual void ComputeElementFlux(const FiniteElement &el,
                                   ElementTransformation &Trans, Vector &u,
                                   const FiniteElement &fluxelem, Vector &FU,
                                   bool with_coef = true,
                                   const IntegrationRule *ir = NULL)
   {
      mfem_error("NOT IMPLEMENTED");
   }

   /**
    * @brief implement (F(u), grad v) with abstract F computed by ComputeFlux
    *
    * @param[in] el local finite element
    * @param[in] Tr element transformation
    * @param[in] elfun local coefficient of basis
    * @param[out] elvect evaluated dual vector
    */
   virtual void AssembleElementVector(const FiniteElement &el,
                                      ElementTransformation &Tr,
                                      const Vector &elfun, Vector &elvect);

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
   virtual void AssembleFaceVector(const FiniteElement &el1,
                                   const FiniteElement &el2,
                                   FaceElementTransformations &Tr,
                                   const Vector &elfun, Vector &elvect);

   virtual ~HyperbolicFormIntegrator() {}
};


// Base Hyperbolic conservation law class.
// This contains all methods needed except the flux function.
class DGHyperbolicConservationLaws : public TimeDependentOperator
{
private:
   const int dim;
   const int num_equations;
   // Vector finite element space containing conserved variables
   FiniteElementSpace *vfes;
   // Element integration form. Should contain ComputeFlux
   HyperbolicFormIntegrator *formIntegrator;
   HyperbolicFormIntegrator *faceFormIntegrator;
   // Base Nonlinear Form
   NonlinearForm *nonlinearForm;
   // element-wise inverse mass matrix
   std::vector<DenseMatrix> Me_inv;
   // global maximum characteristic speed. Updated by form integrators
   mutable double max_char_speed;
   // auxiliary variable used in Mult
   mutable Vector z;

   // Compute element-wise inverse mass matrix
   void ComputeInvMass();

   void Update()
   {
      nonlinearForm->Update();
      height = nonlinearForm->Height();
      width = height;
      z.SetSize(height);

      ComputeInvMass();
   }

public:
   /**
    * @brief Construct a new DGHyperbolicConservationLaws object
    *
    * @param vfes_ vector finite element space. Only tested for DG [Pₚ]ⁿ
    * @param formIntegrator_ integrator (F(u,x), grad v)
    * @param faceFormIntegrator_ integrator (F̂(u±, x, n), [[v]])
    * @param num_equations_ the number of equations
    */
   DGHyperbolicConservationLaws(
      FiniteElementSpace *vfes_,
      HyperbolicFormIntegrator *formIntegrator_,
      HyperbolicFormIntegrator *faceFormIntegrator_,
      const int num_equations_);
   /**
    * @brief Apply nonlinear form to obtain M⁻¹(DIVF + JUMP HAT(F))
    *
    * @param x current solution vector
    * @param y resulting dual vector to be used in an EXPLICIT solver
    */
   virtual void Mult(const Vector &x, Vector &y) const;
   // get global maximum characteristic speed to be used in CFL condition
   // where max_char_speed is updated during Mult.
   inline double getMaxCharSpeed()
   {
      return max_char_speed;
   }

   virtual ~DGHyperbolicConservationLaws()
   {
      delete nonlinearForm;
   }
};


//////////////////////////////////////////////////////////////////
///                      NUMERICAL FLUXES                      ///
//////////////////////////////////////////////////////////////////

/**
 * @brief Rusanov flux hat(F)n = ½(F(u⁺,x)n + F(u⁻,x)n) - ½λ(u⁺ - u⁻)
 * where λ is the maximum characteristic velocity
 *
 */
class RusanovFlux : public RiemannSolver
{
public:
   /**
    * @brief  hat(F)n = ½(F(u⁺,x)n + F(u⁻,x)n) - ½λ(u⁺ - u⁻)
    *
    * @param[in] state1 state value at a point from the first element
    * (num_equations x 1)
    * @param[in] state2 state value at a point from the second element
    * (num_equations x 1)
    * @param[in] fluxN1 normal flux value at a point from the first element
    * (num_equations x dim)
    * @param[in] fluxN2 normal flux value at a point from the second element
    * (num_equations x dim)
    * @param[in] maxE maximum characteristic speed (eigenvalue of flux jacobian)
    * @param[in] nor normal vector (not a unit vector) (dim x 1)
    * @param[out] flux ½(F(u⁺,x)n + F(u⁻,x)n) - ½λ(u⁺ - u⁻)
    */
   void Eval(const Vector &state1, const Vector &state2, const Vector &fluxN1,
             const Vector &fluxN2, const double speed1, const double speed2, const Vector &nor,
             Vector &flux) const
   {
      // NOTE: nor in general is not a unit normal
      const double maxE = std::max(speed1, speed2);
      flux = state1;
      flux -= state2;
      // here, sqrt(nor*nor) is multiplied to match the scale with fluxN
      flux *= maxE * sqrt(nor * nor);
      flux += fluxN1;
      flux += fluxN2;
      flux *= 0.5;
   }
};

class AdvectionFormIntegrator : public HyperbolicFormIntegrator
{
private:
   VectorCoefficient &b;  // velocity coefficient
   Vector bval;           // velocity value storage

public:
   /**
    * @brief Compute F(u)
    *
    * @param U U (u) at current integration point
    * @param Tr current element transformation with integration point
    * @param FU F(u) = ubᵀ
    * @return double maximum characteristic speed, |b|
    */
   double ComputeFlux(const Vector &U, ElementTransformation &Tr,
                      DenseMatrix &FU)
   {
      b.Eval(bval, Tr, Tr.GetIntPoint());
      MultVWt(U, bval, FU);
      return bval.Norml2();
   }
   /**
    * @brief Compute normal flux, F(u)n
    *
    * @param U U (u) at current integration point
    * @param normal normal vector, usually not a unit vector
    * @param Tr current element transformation with integration point
    * @param FUdotN F(u)n = u(b⋅n)
    * @return double maximum characteristic speed, |b|
    */
   double ComputeFluxDotN(const Vector &U, const Vector &normal,
                          ElementTransformation &Tr, Vector &FUdotN)
   {
      b.Eval(bval, Tr, Tr.GetIntPoint());
      const double bN = bval * normal;
      FUdotN = U;
      FUdotN *= bN;
      return bval.Norml2();
   }

   /**
    * @brief Construct a new Advection Element Form Integrator object with given
    * integral order offset
    *
    * @param[in] rsolver_ numerical flux
    * @param dim spatial dimension
    * @param b_ velocity coefficient, possibly depends on space
    * @param IntOrderOffset_ 2*p + IntOrderOffset will be used for quadrature
    */
   AdvectionFormIntegrator(const RiemannSolver &rsolver_, const int dim,
                           VectorCoefficient &b_,
                           const int IntOrderOffset_ = 3)
      : HyperbolicFormIntegrator(rsolver_, dim, 1, IntOrderOffset_), b(b_),
        bval(dim) {}
   /**
    * @brief Construct a new Advection Element Form Integrator object with given
    * integral rule
    *
    * @param[in] rsolver_ numerical flux
    * @param dim spatial dimension
    * @param b_ velocity coefficient, possibly depends on space
    * @param ir this integral rule will be used for the Gauss quadrature
    */
   AdvectionFormIntegrator(const RiemannSolver &rsolver_, const int dim,
                           VectorCoefficient &b_,
                           const IntegrationRule *ir)
      : HyperbolicFormIntegrator(rsolver_, dim, 1, ir), b(b_), bval(dim) {}
};
class BurgersFormIntegrator : public HyperbolicFormIntegrator
{
public:
   /**
    * @brief Compute F(u)
    *
    * @param U U (u) at current integration point
    * @param Tr current element transformation with integration point
    * @param FU F(u) = ½u²*1ᵀ where 1 is (dim x 1) vector
    * @return double maximum characteristic speed, |u|
    */
   double ComputeFlux(const Vector &U, ElementTransformation &Tr,
                      DenseMatrix &FU)
   {
      FU = U * U * 0.5;
      return abs(U(0));
   }
   /**
    * @brief Compute normal flux, F(u)n
    *
    * @param U U (u) at current integration point
    * @param normal normal vector, usually not a unit vector
    * @param Tr current element transformation with integration point
    * @param FUdotN F(u)n = ½u² 1⋅n
    * @return double maximum characteristic speed, |u| + √(γp/ρ)
    */
   double ComputeFluxDotN(const Vector &U, const Vector &normal,
                          ElementTransformation &Tr, Vector &FUdotN)
   {
      FUdotN = normal.Sum() * (U * U) * 0.5;
      return abs(U(0));
   }

   /**
    * @brief Construct a new Burgers Element Form Integrator object with given
    * integral order offset
    *
    * @param[in] rsolver_ numerical flux
    * @param dim spatial dimension
    * @param IntOrderOffset_ 2*p + IntOrderOffset will be used for quadrature
    */
   BurgersFormIntegrator(const RiemannSolver &rsolver_, const int dim,
                         const int IntOrderOffset_ = 3)
      : HyperbolicFormIntegrator(rsolver_, dim, 1, IntOrderOffset_) {}
   /**
    * @brief Construct a new Burgers Element Form Integrator object with given
    * integral rule
    *
    * @param[in] rsolver_ numerical flux
    * @param dim spatial dimension
    * @param ir this integral rule will be used for the Gauss quadrature
    */
   BurgersFormIntegrator(const RiemannSolver &rsolver_, const int dim,
                         const IntegrationRule *ir)
      : HyperbolicFormIntegrator(rsolver_, dim, 1, ir) {}
};

class ShallowWaterFormIntegrator : public HyperbolicFormIntegrator
{
private:
   const double g;  // gravity constant

public:
   /**
    * @brief Compute F(h, hu)
    *
    * @param U U (h, hu) at current integration point
    * @param Tr current element transformation with integration point
    * @param FU F(h, hu) = [huᵀ; huuᵀ + ½gh²I]
    * @return double maximum characteristic speed, |u| + √(gh)
    */
   double ComputeFlux(const Vector &U, ElementTransformation &Tr,
                      DenseMatrix &FU)
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

      const double sound = sqrt(g * height);
      const double vel = sqrt(h_vel * h_vel) / height;

      return vel + sound;
   }
   /**
    * @brief Compute normal flux, F(h, hu)
    *
    * @param U U (h, hu) at current integration point
    * @param normal normal vector, usually not a unit vector
    * @param Tr current element transformation with integration point
    * @param FUdotN F(ρ, ρu, E)n = [ρu⋅n; ρu(u⋅n) + pn; (u⋅n)(E + p)]
    * @return double maximum characteristic speed, |u| + √(γp/ρ)
    */
   double ComputeFluxDotN(const Vector &U, const Vector &normal,
                          ElementTransformation &Tr, Vector &FUdotN)
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

      const double sound = sqrt(g * height);
      const double vel = sqrt(h_vel * h_vel) / height;

      return vel + sound;
   }

   /**
    * @brief Construct a new Shallow Water Element Form Integrator object with
    * given integral order offset
    *
    * @param[in] rsolver_ numerical flux
    * @param dim spatial dimension
    * @param g_ gravity constant
    * @param IntOrderOffset_ 2*p + IntOrderOffset will be used for quadrature
    */
   ShallowWaterFormIntegrator(const RiemannSolver &rsolver_, const int dim,
                              const double g_,
                              const int IntOrderOffset_ = 3)
      : HyperbolicFormIntegrator(rsolver_, dim, dim + 1, IntOrderOffset_), g(g_) {}
   /**
    * @brief Construct a new Shallow Water Element Form Integrator object with
    * given integral rule
    *
    * @param[in] rsolver_ numerical flux
    * @param dim spatial dimension
    * @param g_ gravity constant
    * @param ir this integral rule will be used for the Gauss quadrature
    */
   ShallowWaterFormIntegrator(const RiemannSolver &rsolver_, const int dim,
                              const double g_,
                              const IntegrationRule *ir)
      : HyperbolicFormIntegrator(rsolver_, dim, dim + 1, ir), g(g_) {}
};
class EulerFormIntegrator : public HyperbolicFormIntegrator
{
private:
   const double specific_heat_ratio;  // specific heat ratio, γ
   // const double gas_constant;         // gas constant

public:
   /**
    * @brief Compute F(ρ, ρu, E)
    *
    * @param U U (ρ, ρu, E) at current integration point
    * @param Tr current element transformation with integration point
    * @param FU F(ρ, ρu, E) = [ρuᵀ; ρuuᵀ + pI; uᵀ(E + p)]
    * @return double maximum characteristic speed, |u| + √(γp/ρ)
    */
   double ComputeFlux(const Vector &U, ElementTransformation &Tr,
                      DenseMatrix &FU)
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
      const double sound = sqrt(specific_heat_ratio * pressure / density);
      // fluid speed |u|
      const double speed = sqrt(momentum * momentum) / density;
      // max characteristic speed = fluid speed + sound speed
      return speed + sound;
   }
   /**
    * @brief Compute normal flux, F(ρ, ρu, E)n
    *
    * @param x x (ρ, ρu, E) at current integration point
    * @param normal normal vector, usually not a unit vector
    * @param Tr current element transformation with integration point
    * @param FUdotN F(ρ, ρu, E)n = [ρu⋅n; ρu(u⋅n) + pn; (u⋅n)(E + p)]
    * @return double maximum characteristic speed, |u| + √(γp/ρ)
    */
   double ComputeFluxDotN(const Vector &x, const Vector &normal,
                          ElementTransformation &Tr, Vector &FUdotN)
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
      const double sound = sqrt(specific_heat_ratio * pressure / density);
      // fluid speed |u|
      const double speed = sqrt(momentum * momentum) / density;
      // max characteristic speed = fluid speed + sound speed
      return speed + sound;
   }

   /**
    * @brief Construct a new Euler Element Form Integrator object with given
    * integral order offset
    *
    * @param[in] rsolver_ numerical flux
    * @param dim spatial dimension
    * @param specific_heat_ratio_ specific heat ratio, γ
    * @param IntOrderOffset_ 2*p + IntOrderOffset will be used for quadrature
    */
   EulerFormIntegrator(const RiemannSolver &rsolver_, const int dim,
                       const double specific_heat_ratio_,
                       const int IntOrderOffset_)
      : HyperbolicFormIntegrator(rsolver_, dim, dim + 2, IntOrderOffset_),
        specific_heat_ratio(specific_heat_ratio_) {}

   /**
    * @brief Construct a new Euler Element Form Integrator object with given
    * integral rule
    *
    * @param[in] rsolver_ numerical flux
    * @param dim spatial dimension
    * @param specific_heat_ratio_ specific heat ratio, γ
    * @param ir this integral rule will be used for the Gauss quadrature
    */
   EulerFormIntegrator(const RiemannSolver &rsolver_, const int dim,
                       const double specific_heat_ratio_,
                       const IntegrationRule *ir)
      : HyperbolicFormIntegrator(rsolver_, dim, dim + 2, ir),
        specific_heat_ratio(specific_heat_ratio_) {}
};
}

#endif