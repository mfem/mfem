// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_NAVIER_SOLVER_HPP
#define MFEM_NAVIER_SOLVER_HPP

#define NAVIER_VERSION 0.2

#include "mfem.hpp"
#include "kernels/curl_evaluator.hpp"
#include "kernels/mean_evaluator.hpp"
#include "kernels/shear_stress_evaluator.hpp"
#include "kernels/custom_navier_coefficients.hpp"
#include "kernels/vector_boundary_AMP_LFIntegrator.hpp"

#include "fsi_utils.hpp"

namespace mfem
{
namespace navier
{
using VecFuncT = mfem::common_fsi_utils::VecFuncT;
using ScalarFuncT = mfem::common_fsi_utils::ScalarFuncT;
using namespace mfem::common_fsi_utils;

/// Container for a Dirichlet boundary condition of the velocity field.
class VelDirichletBC_T
{
public:
   VelDirichletBC_T(Array<int> attr, VectorCoefficient *coeff, bool own = true)
      : attr(attr), coeff(coeff), own(own)
   {}

   VelDirichletBC_T(VelDirichletBC_T &&obj)
   {
      // Deep copy the attribute array
      this->attr = obj.attr;
      this->own = obj.own;
      this->component = obj.component;

      // Move the coefficient pointer
      this->coeff = obj.coeff;
      obj.coeff = nullptr;
   }

   ~VelDirichletBC_T()
   {
      if (own) { delete coeff; }
   }

   Array<int> attr;
   VectorCoefficient *coeff;
   bool own;
   int component; // -1 for all components, otherwise specifies the component to which the BC applies
};

/// Container for a Dirichlet boundary condition of the pressure field.
class PresDirichletBC_T
{
public:
   PresDirichletBC_T(Array<int> attr, Coefficient *coeff, bool own = true)
       : own(own), attr(attr), coeff(coeff)
   {}

   PresDirichletBC_T(PresDirichletBC_T &&obj)
   {
      // Deep copy the attribute array
      this->attr = obj.attr;
      this->own = obj.own;

      // Move the coefficient pointer
      this->coeff = obj.coeff;
      obj.coeff = nullptr;
   }

   ~PresDirichletBC_T()
   {
      if (own) { delete coeff; }
   }

   Array<int> attr;
   Coefficient *coeff;
   bool own;
};

class RobinBC_T : public MultiVectorCoeffContainer
{
public:
   // Constructor for multiple coefficients with individual scaling and alpha
   RobinBC_T(const Array<int> &attr_,
             real_t alpha_,
             real_t density_,
             const std::vector<VectorCoefficient *> &coeffs_,
             const std::vector<real_t> &scalings_,
             bool own_ = true)
       : MultiVectorCoeffContainer(attr_, coeffs_, scalings_, own_),
         density(density_)
   {
      alphaP = new ConstantCoefficient(alpha_);         // initialize with alpha
      beta = new ConstantCoefficient(alpha_);             // initialize with alpha
      beta_g = new ScalarVectorProductCoefficient(*beta, *(this->GetCoeff(1))); // beta * g
      rho_beta = new ProductCoefficient(density_, *beta);
      alphaP_u = new ScalarVectorProductCoefficient(*alphaP, *(this->GetCoeff(0))); // alpha * u_s
      alphaP_over_rho = new ProductCoefficient(1.0/density, *alphaP); // alpha / rho
   }

   // Constructor for multiple coefficients with uniform scaling and alpha
   RobinBC_T(const Array<int> &attr_,
             real_t alpha_,
             real_t density_,
             const std::vector<VectorCoefficient *> &coeffs_,
             real_t scaling_ = 1.0,
             bool own_ = true)
       : MultiVectorCoeffContainer(attr_, coeffs_, scaling_, own_),
         density(density_)
   {
      alphaP = new ConstantCoefficient(alpha_);
      beta = new ConstantCoefficient(alpha_);         // initialize with alpha
      rho_beta = new ProductCoefficient(density_, *beta); // rho * beta
      beta_g = new ScalarVectorProductCoefficient(*beta, *(this->GetCoeff(1)));
      alphaP_u = new ScalarVectorProductCoefficient(*alphaP, *(this->GetCoeff(0))); // alpha * u_s
      alphaP_over_rho = new ProductCoefficient(1.0/density, *alphaP); // alpha / rho
   }

   // Optimized constructor for multiple coefficients with no scaling and alpha
   RobinBC_T(const Array<int> &attr_,
             real_t alpha_,
             real_t density_,
             const std::vector<VectorCoefficient *> &coeffs_,
             bool own_ = true)
       : MultiVectorCoeffContainer(attr_, coeffs_, own_),
         density(density_)
   {
      alphaP = new ConstantCoefficient(alpha_);         // initialize with alpha
      beta = new ConstantCoefficient(alpha_);             // initialize with alpha
      rho_beta = new ProductCoefficient(density_, *beta); // rho * beta
      beta_g = new ScalarVectorProductCoefficient(*beta, *(this->GetCoeff(1)));
      alphaP_u = new ScalarVectorProductCoefficient(*alphaP, *(this->GetCoeff(0))); // alpha * u_s
      alphaP_over_rho = new ProductCoefficient(1.0/density, *alphaP);
   }

   // Destructor
   ~RobinBC_T()
   {
      delete alphaP;
      delete rho_beta;
      delete beta;
      delete beta_g;
      delete alphaP_u;
      delete beta_tau; 
      delete alphaP_over_rho; 
   }

   // Access method for alpha
   ConstantCoefficient *GetAlphaP() const
   {
      return alphaP;
   }

   // Access method for beta coefficient
   ConstantCoefficient *GetBeta() const
   {
      return beta;
   }

   // Access method for rho_beta coefficient
   ProductCoefficient *GetRhoBeta() const
   {
      return rho_beta;
   }

   // Access method for beta_g coefficient
   ScalarVectorProductCoefficient *GetBetaG() const
   {
      return beta_g;
   }

   // Method to update beta value (e.g., when timestep changes)
   void SetBeta(real_t beta_value)
   {
      beta->constant = beta_value;
      // This will also update rho_beta since it takes a reference to beta
   }

   // Method to update alpha value
   void SetAlphaP(real_t alpha_value)
   {
      alphaP->constant = alpha_value;
   }

   // Method to set the shear stress coefficient
   void SetShearStressCoefficient(ShearStressNormalTractionCoefficient *tau_coeff_)
   {
      tau_coeff = tau_coeff_;
      if (beta_tau == nullptr)
         beta_tau = new ScalarVectorProductCoefficient(*beta, *tau_coeff);
      else
         beta_tau->SetBCoef(*tau_coeff);
   }

   // Method to update density value
   void SetDensity(real_t density_value)
   {
      density = density_value;
      // Update rho_beta by setting the scaling coefficient
      rho_beta->SetAConst(density);
      alphaP_over_rho->SetAConst(1.0/density);
   }

   // Method to get current density value
   real_t GetDensity() const
   {
      return density;
   }

   void SetTime(real_t t) override
   {
      MultiVectorCoeffContainer::SetTime(t);
      //beta->SetTime(t);
      //rho_beta->SetTime(t); --> Not needed as rho is constant
      beta_g->SetTime(t);  // --> beta_g will also update beta
      //alpha->SetTime(t);
      alphaP_u->SetTime(t); // --> alpha_u will also update alpha
   }

   real_t density;                                     ///< Density value.
   ConstantCoefficient *alphaP = nullptr;               ///< Scalar value for the Robin BC.
   ConstantCoefficient *beta = nullptr;                ///< beta = 1/alpha b0/dt
   ProductCoefficient *alphaP_over_rho = nullptr;       ///< alpha / rho
   ProductCoefficient *rho_beta = nullptr;             ///< rho * beta
   ShearStressNormalTractionCoefficient *tau_coeff = nullptr;  ///< Coefficient for the shear stress term.
   ScalarVectorProductCoefficient *alphaP_u = nullptr;  ///< Coefficient for the term alpha * u_s.
   ScalarVectorProductCoefficient *beta_g = nullptr;   ///< Coefficient for the term beta * g, where g is the normal traction.
   ScalarVectorProductCoefficient *beta_tau = nullptr; ///< Coefficient for the term beta * tau, where tau is the shear stress.
};


class AMP_RobinBC_T : public RobinBC_T
{
public:
   // Constructor for multiple coefficients with individual scaling and alpha
   AMP_RobinBC_T(const Array<int> &attr_,
                 real_t alpha_s_,
                 real_t alpha_p_,
                 real_t density_,
                 const std::vector<VectorCoefficient *> &coeffs_,
                 const std::vector<real_t> &scalings_,
                 bool own_ = true)
       : RobinBC_T(attr_, alpha_p_, density_, coeffs_, scalings_, own_)
   {
      alphaS = new ConstantCoefficient(alpha_s_);
      alphaS_over_rho = new ProductCoefficient(1.0/density, *alphaS); // alphaS / rho
      int dim = coeffs[0]->GetVDim();
      normal_outer_product = new NormalOuterProductCoefficient(dim, alphaP_over_rho);
      tangent_outer_product = new TangentOuterProductCoefficient(dim, alphaS_over_rho);
   }

   // Constructor for multiple coefficients with uniform scaling and alpha
   AMP_RobinBC_T(const Array<int> &attr_,
                 real_t alpha_s_,
                 real_t alpha_p_,
                 real_t density_,
                 const std::vector<VectorCoefficient *> &coeffs_,
                 real_t scaling_ = 1.0,
                 bool own_ = true)
       : RobinBC_T(attr_, alpha_p_, density_, coeffs_, scaling_, own_)
   {
      alphaS = new ConstantCoefficient(alpha_s_);
      alphaS_over_rho = new ProductCoefficient(1.0/density, *alphaS); // alphaS / rho
      int dim = coeffs[0]->GetVDim();
      normal_outer_product = new NormalOuterProductCoefficient(dim, alphaP_over_rho);
      tangent_outer_product = new TangentOuterProductCoefficient(dim, alphaS_over_rho);
   }

   // Optimized constructor for multiple coefficients with no scaling and alpha
   AMP_RobinBC_T(const Array<int> &attr_,
                 real_t alpha_s_,
                 real_t alpha_p_,
                 real_t density_,
                 const std::vector<VectorCoefficient *> &coeffs_,
                 bool own_ = true)
       : RobinBC_T(attr_, alpha_p_, density_, coeffs_, own_)
   {
      alphaS = new ConstantCoefficient(alpha_s_);
      alphaS_over_rho = new ProductCoefficient(1.0/density, *alphaS); // alphaS / rho
      int dim = coeffs[0]->GetVDim();
      normal_outer_product = new NormalOuterProductCoefficient(dim, alphaP_over_rho);
      tangent_outer_product = new TangentOuterProductCoefficient(dim, alphaS_over_rho);
   }

   // Destructor
   ~AMP_RobinBC_T()
   {
      delete alphaS;
      delete alphaS_over_rho; 
      delete tangent_outer_product;
      delete normal_outer_product;
   }

   // Access method for alphaS
   ConstantCoefficient *GetAlphaS() const
   {
      return alphaS;
   }

   // Method to update alphaS value
   void SetAlphaS(real_t alpha_value)
   {
      alphaS->constant = alpha_value;
   }

      void SetDensity(real_t density_value)
   {
      RobinBC_T::SetDensity(density_value);
      alphaS_over_rho->SetAConst(1.0/density);
   }

   void SetTime(real_t t) override
   {
      RobinBC_T::SetTime(t);
      alphaS->SetTime(t); // Update alphaS time
   }

   ConstantCoefficient *alphaS = nullptr;  ///< Additional scalar coefficient alphaS.
   ProductCoefficient *alphaS_over_rho = nullptr; ///< Coefficient for the term alphaS / rho.
   NormalOuterProductCoefficient *normal_outer_product = nullptr; ///< Coefficient for the term a * n n^T.
   TangentOuterProductCoefficient *tangent_outer_product = nullptr; ///< Coefficient for the term a * t t^T.
};


/// Scalar coefficient that evaluates the derived pressure Dirichlet value on a
/// Neumann boundary: p = (τ·n)·n̂ - g·n̂ - α_div div(u), where τ = 2ν ε(u_ext)
/// and g is the prescribed traction. Used with ProjectBdrCoefficient in the PPE
/// solve.
class NeumannPressureBCCoefficient : public Coefficient
{
public:
   NeumannPressureBCCoefficient(ShearStressNormalTractionCoefficient *tau_n,
                                VectorCoefficient *g, int dim_,
                                bool use_shear_stress = true,
                                ParGridFunction *velocity_gf_ = nullptr,
                                Coefficient *alpha_div_ = nullptr)
      : tau_n_coeff(tau_n), g_coeff(g), dim(dim_),
        use_shear_stress(use_shear_stress), alpha_div_coeff(alpha_div_)
   {
      if (velocity_gf_)
      {
         div_u_coeff = new DivergenceGridFunctionCoefficient(velocity_gf_);
      }
   }

   ~NeumannPressureBCCoefficient() { delete div_u_coeff; }

   real_t Eval(ElementTransformation &T, const IntegrationPoint &ip) override
   {
      Vector nor(dim), tau_n_val(dim), g_val(dim);
      CalcOrtho(T.Jacobian(), nor);
      nor /= nor.Norml2(); // unit outward normal
      tau_n_coeff->Eval(tau_n_val, T, ip);
      g_coeff->Eval(g_val, T, ip);
      real_t val = use_shear_stress ? (tau_n_val * nor) - (g_val * nor) : -(g_val * nor);
      if (div_u_coeff)
      {
         real_t div_u = div_u_coeff->Eval(T, ip);
         real_t a_div = alpha_div_coeff ? alpha_div_coeff->Eval(T, ip) : 1.0;
         val -= a_div * div_u;
      }
      return val;
   }

   void SetTime(real_t t) override { g_coeff->SetTime(t); }

private:
   ShearStressNormalTractionCoefficient *tau_n_coeff; // not owned
   VectorCoefficient *g_coeff;                        // not owned
   int dim;
   Coefficient *alpha_div_coeff = nullptr;             // not owned (kin_vis)
   DivergenceGridFunctionCoefficient *div_u_coeff = nullptr; // owned
   bool use_shear_stress;
};

/// Container for a Neumann boundary condition.
/// Holds the prescribed traction g; pres_coeff is null until
/// SetCoefficients() is called in Setup().
class NeumannBC_T
{
public:
   NeumannBC_T(Array<int> attr, VectorCoefficient *coeff, bool own = true, bool use_shear_stress_ = true,
               bool use_div_ = false, Coefficient *alpha_div_ = nullptr,
               bool own_alpha_div_ = false)
      : attr(attr), coeff(coeff), own(own),
        use_div(use_div_), alpha_div(alpha_div_),
        own_alpha_div(own_alpha_div_), use_shear_stress(use_shear_stress_) {}

   NeumannBC_T(NeumannBC_T &&obj)
   {
      this->attr       = obj.attr;
      this->own        = obj.own;
      this->coeff      = obj.coeff;      obj.coeff      = nullptr;
      this->pres_coeff = obj.pres_coeff; obj.pres_coeff = nullptr;
      this->use_div    = obj.use_div;
      this->alpha_div  = obj.alpha_div;  obj.alpha_div  = nullptr;
      this->own_alpha_div = obj.own_alpha_div;
      obj.own_alpha_div = false;
   }

   ~NeumannBC_T()
   {
      if (own) { delete coeff; }
      delete pres_coeff;
      if (own_alpha_div) { delete alpha_div; }
   }

   /// Build the derived pressure Dirichlet coefficient for this Neumann BC.
   /// Called in Setup() once the shear-stress and velocity data are available.
   void SetCoefficients(ShearStressNormalTractionCoefficient *tau_n,
                        int dim,
                        ParGridFunction *velocity_gf = nullptr)
   {
      if (pres_coeff == nullptr)
      {
         ParGridFunction *vel = use_div ? velocity_gf : nullptr;
         pres_coeff = new NeumannPressureBCCoefficient(tau_n, coeff, dim, use_shear_stress,
                                                       vel, alpha_div);
      }
   }

   void SetTime(real_t t) { coeff->SetTime(t); if (pres_coeff) { pres_coeff->SetTime(t); } }

   Array<int>                     attr;
   VectorCoefficient             *coeff      = nullptr; ///< prescribed traction g
   NeumannPressureBCCoefficient  *pres_coeff = nullptr; ///< (τ·n)·n̂ - g·n̂ - α_div div(u), set in Setup()
   bool                           own;
   bool                           use_div = false;       ///< whether to include the div term
   bool                          use_shear_stress = true; ///< whether to include the shear stress term
   Coefficient                   *alpha_div = nullptr;   ///< coefficient for div penalty (e.g. kin_vis)
   bool                           own_alpha_div = false;
};

/// Container for an acceleration term.
class AccelTerm_T
{
public:
   AccelTerm_T(Array<int> attr, VectorCoefficient *coeff)
      : attr(attr), coeff(coeff)
   {}

   AccelTerm_T(AccelTerm_T &&obj)
   {
      // Deep copy the attribute array
      this->attr = obj.attr;

      // Move the coefficient pointer
      this->coeff = obj.coeff;
      obj.coeff = nullptr;
   }

   ~AccelTerm_T() { delete coeff; }

   Array<int> attr;
   VectorCoefficient *coeff;
};

/// Transient incompressible Navier Stokes solver in a split scheme formulation.
/**
 * This implementation of a transient incompressible Navier Stokes solver uses
 * the non-dimensionalized formulation. The coupled momentum and
 * incompressibility equations are decoupled using the split scheme described in
 * [1]. This leads to three solving steps.
 *
 * 1. An extrapolation step for all nonlinear terms which are treated
 *    explicitly. This step avoids a fully coupled nonlinear solve and only
 *    requires a solve of the mass matrix in velocity space $M_v^{-1}$. On
 *    the other hand this introduces a CFL stability condition on the maximum
 *    timestep.
 *
 * 2. A Poisson solve $S_p^{-1}$.
 *
 * 3. A Helmholtz like solve $(M_v - \partial t K_v)^{-1}$.
 *
 * The numerical solver setup for each step are as follows.
 *
 * $M_v^{-1}$ is solved using CG with Jacobi as preconditioner.
 *
 * $S_p^{-1}$ is solved using CG with AMG applied to the low order refined
 * (LOR) assembled pressure Poisson matrix. To avoid assembling a matrix for
 * preconditioning, one can use p-MG as an alternative (NYI).
 *
 * $(M_v - \partial t K_v)^{-1}$ due to the CFL condition we expect the time
 * step to be small. Therefore this is solved using CG with Jacobi as
 * preconditioner. For large time steps a preconditioner like AMG or p-MG should
 * be used (NYI).
 *
 * Statements marked with NYI mean this feature is planned but Not Yet
 * Implemented.
 *
 * A detailed description is available in [1] section 4.2. The algorithm is
 * originated from [2].
 *
 * [1] Michael Franco, Jean-Sylvain Camier, Julian Andrej, Will Pazner (2020)
 * High-order matrix-free incompressible flow solvers with GPU acceleration and
 * low-order refined preconditioners (https://arxiv.org/abs/1910.03032)
 *
 * [2] A. G. Tomboulides, J. C. Y. Lee & S. A. Orszag (1997) Numerical
 * Simulation of Low Mach Number Reactive Flows
 */
class NavierSolver
{
public:
   // Filter-based stabilization
   enum FilterMethod
   {
      NONE,
      // High-pass filter relaxation term using a lower order space to damp high
      // frequency modes.
      HPFRT_LOS,
      // High-pass filter relaxation term using a projection to a Legendre
      // polynomial space to damp high frequency modes.
      HPFRT_LPOLY
   };

   /// Initialize data structures, set FE space order and kinematic viscosity.
   /**
    * The ParMesh @a mesh can be a linear or curved parallel mesh. The @a order
    * of the finite element spaces in this algorithm is of equal order
    * $(P_N)^d P_N$ for velocity and pressure respectively. This means the
    * pressure is discretized in the same space (just scalar instead of a
    * vector space) as the velocity.
    *
    * Kinematic viscosity (dimensionless) is set using @a kin_vis and
    * automatically converted to the Reynolds number. If you want to set the
    * Reynolds number directly, you can provide the inverse.
    */
   NavierSolver(ParMesh *mesh, int order, real_t kin_vis, const real_t density = 1.0, bool verbose_ = true, bool variable_viscosity_ = true);

   /// Initialize forms, solvers and preconditioners.
   void Setup(real_t dt);

   void SetMeshVelocity(ParGridFunction *mesh_velocity_gf, bool own = false);

   void UpdateSpaces();

   // Update Forms, Solvers and Assemble operators
   void UpdateForms(bool full_update = true);
   void UpdateSolvers();
   void Assemble(bool full_update = true);

   /// Compute solution at the next time step t+dt.
   /**
    * This method can be called with the default value @a provisional which
    * always accepts the computed time step by automatically calling
    * UpdateTimestepHistory.
    *
    * If @a provisional is set to true, the solution at t+dt is not accepted
    * automatically and the application code has to call UpdateTimestepHistory
    * and update the @a time variable accordingly.
    *
    * The application code can check the provisional step by retrieving the
    * GridFunction with the method GetProvisionalVelocity. If the check fails,
    * it is possible to retry the step with a different time step by not
    * calling UpdateTimestepHistory and calling this method with the previous
    * @a time and @a cur_step.
    *
    * The method and parameter choices are based on [1].
    *
    * [1] D. Wang, S.J. Ruuth (2008) Variable step-size implicit-explicit
    * linear multistep methods for time-dependent partial differential
    * equations
    */
   void Step(real_t &time, real_t dt, int cur_step, bool provisional = false);

   /// Return a pointer to the provisional velocity ParGridFunction.
   ParGridFunction *GetProvisionalVelocity() { return &un_next_gf; }

   /// Return a pointer to the current velocity ParGridFunction.
   ParGridFunction *GetCurrentVelocity() { return &un_gf; }

   /// Return a pointer to the current pressure ParGridFunction.
   ParGridFunction *GetCurrentPressure() { return &pn_gf; }

   /// Return a pointer to the current vorticity ParGridFunction.
   /** The vorticity is only computed (in Step()) after the first call to this
       method, so it should be called before the time loop. */
   ParGridFunction *GetCurrentVorticity()
   {
      compute_vorticity = true;
      return &curlu_gf;
   }

   BilinearFormIntegrator *GetPressureEquationBLFI() { return *(Sp_form->GetDBFI()[0]); }

   /// Add a Dirichlet boundary condition to the velocity field.
   void AddVelDirichletBC(VectorCoefficient *coeff, Array<int> &attrs, bool own = true);
   void AddVelDirichletBC(VecFuncT *f, Array<int> &attrs);

   // Add a Dirichlet boundary condition to the velocity field for a specific component.
   void AddVelDirichletBC(Coefficient *coeff, Array<int> &attrs, int component, bool own = true);
   void AddVelDirichletBC(Coefficient *coeff, int attr, int component, bool own = true);

   /// Add a Dirichlet boundary condition to the pressure field.
   void AddPresDirichletBC(Coefficient *coeff, Array<int> &attrs, bool own = true);
   void AddPresDirichletBC(ScalarFuncT *f, Array<int> &attrs);

   // Add a Neumann boundary condition (turns into a Neumann for momentum and Dirichlet for PPE)
   void AddNeumannBC(VectorCoefficient *traction_coeff, Array<int> &attrs, bool own = true, bool use_shear_stress = true,
                     bool use_div = false, Coefficient *alpha_div = nullptr,
                     bool own_alpha_div = false);
   void AddNeumannBC(VecFuncT *traction_f, Array<int> &attrs);
   void AddNeumannBC(VectorCoefficient *traction_coeff, int attr, bool own = true, bool use_shear_stress = true,
                     bool use_div = false, Coefficient *alpha_div = nullptr,
                     bool own_alpha_div = false);
   void AddNeumannBC(VecFuncT *traction_f, int attr);

   /// Add a Robin boundary condition to the velocity field.
   void AddRobinBC(real_t alpha, VectorCoefficient *vel_coeff, VectorCoefficient *traction_coeff, Array<int> &attrs, bool own = true);
   void AddRobinBC(real_t alpha, VecFuncT *vel_f, VecFuncT *traction_f, Array<int> &attrs);
   void AddRobinBC(real_t alpha, VectorCoefficient *vel_coeff, VectorCoefficient *traction_coeff, int attr, bool own = true);
   void AddRobinBC(real_t alpha, VecFuncT *vel_f, VecFuncT *traction_f, int attr);
   void SetRobinBCAlpha(real_t alpha);

   /// Add a Robin boundary condition to the velocity field.
   void AddAMPRobinBC(real_t alphaS, real_t alpha_p, VectorCoefficient *vel_coeff, VectorCoefficient *traction_coeff, Array<int> &attrs, bool own = true);
   void AddAMPRobinBC(real_t alphaS, real_t alpha_p, VecFuncT *vel_f, VecFuncT *traction_f, Array<int> &attrs);
   void AddAMPRobinBC(real_t alphaS, real_t alpha_p, VectorCoefficient *vel_coeff, VectorCoefficient *traction_coeff, int attr, bool own = true);
   void AddAMPRobinBC(real_t alphaS, real_t alpha_p, VecFuncT *vel_f, VecFuncT *traction_f, int attr);
   void SetAMPRobinBCAlphaP(real_t alpha_p);
   void SetAMPRobinBCAlphaS(real_t alpha_s);

   /// Add an acceleration term to the RHS of the equation.
   /**
    * The VecFuncT @a f is evaluated at the current time t and extrapolated
    * together with the nonlinear parts of the Navier Stokes equation.
    */
   void AddAccelTerm(VectorCoefficient *coeff, Array<int> &attr);

   void AddAccelTerm(VecFuncT *f, Array<int> &attr);

   /// Print timing summary of the solving routine.
   /**
    * The summary shows the timing in seconds in the first row of
    *
    * 1. SETUP: Time spent for the setup of all forms, solvers and
    *    preconditioners.
    * 2. STEP: Time spent computing a full time step. It includes all three
    *    solves.
    * 3. EXTRAP: Time spent for extrapolation of all forcing and nonlinear
    *    terms.
    * 4. CURLCURL: Time spent for computing the curl curl term in the pressure
    *    Poisson equation (see references for detailed explanation).
    * 5. PSOLVE: Time spent in the pressure Poisson solve.
    * 6. HSOLVE: Time spent in the Helmholtz solve.
    *
    * The second row shows a proportion of a column relative to the whole
    * time step.
    */
   void PrintTimingData();

   ~NavierSolver();

   /// Update forms and solvers (e.g. after a mesh deformation).
   void Update();

   /// Compute $\nabla \times u$ for $u \in (H^1)^2$.
   void ComputeCurl2D(ParGridFunction &u,
                      ParGridFunction &cu,
                      bool assume_scalar = false);

   /// Compute $\nabla \times u$ for $u \in (H^1)^3$.
   void ComputeCurl3D(ParGridFunction &u, ParGridFunction &cu);

   /// Remove mean from a Vector.
   /**
    * Modify the Vector @a v by subtracting its mean using
    * $v = v - \frac{\sum_i^N v_i}{N} $
    */
   void Orthogonalize(Vector &v);

   /// Remove the mean from a ParGridFunction.
   /**
    * Modify the ParGridFunction @a v by subtracting its mean using
    * $ v = v - \int_\Omega \frac{v}{vol(\Omega)} dx $.
    */
   void MeanZero(ParGridFunction &v);

   /// Rotate entries in the time step and solution history arrays.
   void UpdateTimestepHistory(real_t dt);

   /// Set the maximum order to use for the BDF method.
   void SetMaxBDFOrder(int maxbdforder) { max_bdf_order = maxbdforder; };

   /// Compute CFL
   real_t ComputeCFL(ParGridFunction &u, real_t dt);

   int PredictTimestep(const real_t dt_min, const real_t dt_max,
                       const real_t cfl_target, real_t &dt);

   void EnableFilter(FilterMethod f);

   /// Set the number of modes to cut off in the interpolation filter
   void SetCutoffModes(int c) { filter_cutoff_modes = c; }

   /// Set the interpolation filter parameter @a a
   /**
    * If @a a is > 0, the filtering algorithm for the velocity field after every
    * time step from [1] is used. The parameter should satisfy 0 < @a a <= 1.
    *
    * [1] Paul Fischer, Julia Mullen (2001) Filter-based stabilization of
    * spectral element methods
    */
   void SetFilterAlpha(real_t a) { filter_alpha = a; }

   /// Return the (variable) kinematic viscosity. If it is modified, call RequestTDOperatorsUpdate().
   ParGridFunction *GetVariableViscosity()
   {
      return &kin_vis_gf;
   }

   /// Force the re-assembly of the time-dependent operators (and solvers) in the next Step(), e.g. after
   /// changing the viscosity or other coefficients the operators depend on. Mesh changes use Update().
   void RequestTDOperatorsUpdate() { update_td_operators = true; }

   ParGridFunction *GetCurrentMeshVelocity() { return wg_gf; }


   real_t NekNorm(ParGridFunction &u, int type, bool is_vector);

   VectorCoefficient *GetShearStressNormalTractionCoefficient() { return shear_stress_normal_traction_coeff; }

protected:
   /// Print information about the Navier version.
   void PrintInfo();

   /// Update the EXTk/BDF time integration coefficient.
   /**
    * Depending on which time step the computation is in, the EXTk/BDF time
    * integration coefficients have to be set accordingly. This allows
    * bootstrapping with a BDF scheme of order 1 and increasing the order each
    * following time step, up to order 3 (or whichever order is set in
    * SetMaxBDFOrder).
    */
   void SetTimeIntegrationCoefficients(int step);

   // Internal methods to assemble operators and forms.
   void AssembleTDOperators();
   void AssembleConstantOperators();

   /// Eliminate essential BCs in an Operator and apply to RHS.
   void EliminateRHS(ParBilinearForm &A_form,
                     ConstrainedOperator &constrainedA,
                     const Array<int> &ess_tdof_list,
                     Vector &x,
                     Vector &b,
                     Vector &X,
                     Vector &B,
                     int copy_interior = 0);

   void BuildHPFForcing(ParGridFunction &vel_gf);

   // Evaluate Legendre Polynomials of order N at point x
   void EvaluateLegendrePolynomial(const int poly_order, const real_t x,
                                   Vector &L);

   // Evaluate Legendre Polynomials of order N at point x
   void EvaluateLegendrePolynomialShifted(const int poly_order, const real_t x,
                                          Vector &L);

   // Evaluate Monomial Basis of order N at point x
   void EvaluateMonomialBasis(const int poly_order, const real_t x, Vector &L);

   /// Enable/disable debug output.
   bool debug = false;

   /// Enable/disable verbose output.
   bool verbose;

   /// Enable/disable variable viscosity (stress formulation)
   bool variable_viscosity = true;

   /// ALE formulation
   // bool ale_formulation = false;

   /// The parallel mesh.
   ParMesh *pmesh = nullptr;

   /// The order of the velocity and pressure space.
   int order;

   /// Kinematic viscosity (dimensionless).
   real_t kin_vis;

   /// Fluid density
   real_t density;

   ///
   ParGridFunction kin_vis_gf;

   GridFunctionCoefficient kin_vis_gf_coeff;

   int CheckBCAttr(int i, Array<int> &attrs);

public:
   IntegrationRules gll_rules;
   IntegrationRule gll_ir;
   IntegrationRule gll_ir_face;
   IntegrationRule gll_ir_nl;
protected:

   /// Velocity $H^1$ finite element collection.
   FiniteElementCollection *vfec = nullptr;

   /// Pressure $H^1$ finite element collection.
   FiniteElementCollection *vfecL2 = nullptr;

   /// Pressure \f$H^1\f$ finite element collection.
   FiniteElementCollection *pfec = nullptr;

   /// Velocity $(H^1)^d$ finite element space.
   ParFiniteElementSpace *vfes = nullptr;

   /// Pressure $H^1$ finite element space.
   ParFiniteElementSpace *vfesL2 = nullptr;

   /// Pressure \f$H^1\f$ finite element space.
   ParFiniteElementSpace *pfes = nullptr;

   ParNonlinearForm *N = nullptr;

   ParBilinearForm *Mv_form = nullptr;

   ParBilinearForm *Sp_form = nullptr;

   ParMixedBilinearForm *D_form = nullptr;

   ParMixedBilinearForm *G_form = nullptr;

   ParBilinearForm *H_form = nullptr;
   ParBilinearForm *HMv_form = nullptr;

   VectorGridFunctionCoefficient *FText_gfcoeff = nullptr;

   ProductCoefficient *dyn_vis_coeff = nullptr;  ///< mu = rho nu, for the dynamic tau in traction-type BC data
   ShearStressNormalTractionCoefficient *shear_stress_normal_traction_coeff = nullptr;

   /// Pressure p^{n+1} for the (p n, v) boundary terms on traction-type boundaries.
   /// Needed because the momentum uses the strong gradient (grad p, v), whose natural BC is tau·n.
   GridFunctionCoefficient *pn_bdr_coeff = nullptr;

   ParLinearForm *FText_bdr_form = nullptr;

   ParLinearForm *f_form = nullptr;

   ParLinearForm *g_bdr_form = nullptr;

   // Robin boundary terms
   ParLinearForm *g_pres_robin_bdr_form = nullptr;
   ParLinearForm *t_pres_robin_bdr_form = nullptr;
   ParLinearForm *tau_pres_robin_bdr_form = nullptr;
   ParLinearForm *t_vel_robin_bdr_form = nullptr;
   ParLinearForm *g_vel_robin_bdr_form = nullptr;
   ParLinearForm *g_vel_tangent_amp_robin_bdr_form = nullptr;
   ParLinearForm *g_vel_norm_amp_robin_bdr_form = nullptr;

   // Neumann boundary terms
   ParLinearForm *g_vel_neumann_bdr_form = nullptr; ///< (g, v)_Γ for the Helmholtz RHS

   /// Linear form to compute the mass matrix in various subroutines.
   ParLinearForm *mass_lf = nullptr;
   ParLinearForm *component_mass_lf = nullptr;
   ConstantCoefficient onecoeff;

public:
   real_t volume = 0.0;
protected:

   ConstantCoefficient nlcoeff;
   ConstantCoefficient Sp_coeff;
   ConstantCoefficient H_lincoeff;
   ConstantCoefficient H_bdfcoeff;

   OperatorHandle Sp;
   OperatorHandle D;
   OperatorHandle G;
   OperatorHandle H;
   OperatorHandle Mv;

   HypreBoomerAMG *SpInvPC = nullptr;
   OrthoSolver *SpInvOrthoPC = nullptr;
   CGSolver *SpInv = nullptr;

   Solver *HInvPC = nullptr;
   CGSolver *HInv = nullptr;

   std::unique_ptr<Solver>MvInvPC;
   std::unique_ptr<CGSolver> MvInv;

   Vector fn, un, un_next, unm1, unm2, Nun, Nunm1, Nunm2, u_ext, Fext, FText, Lext,
          resu;
   Vector tmp1, tmp2, hpfrt_tdofs;

   Vector pn, resp, FText_bdr, g_bdr;

   ParGridFunction un_gf, un_next_gf, u_ext_gf, curlu_gf, curlcurlu_gf, Lext_gf, FText_gf,
                   resu_gf;

   ParGridFunction pn_gf, resp_gf;

   /// Compute the vorticity curlu_gf in Step(), see GetCurrentVorticity().
   bool compute_vorticity = false;

   bool full_update = true; // Tracks whether to perform the assembly of all operators (including constants ones like G, D)

   /// The time-dependent operators must be re-assembled in the next Step(), see RequestTDOperatorsUpdate().
   bool update_td_operators = true;
   /// Value of bd0/dt the time-dependent operators were last assembled with.
   real_t assembled_H_bdfcoeff = -1.0;

   // All essential attributes.
   Array<int> vel_ess_attr;
   Array<int> vel_ess_attr_x; // Essential mesh attributes (x component applied).
   Array<int> vel_ess_attr_y; // Essential mesh attributes (y component applied).
   Array<int> vel_ess_attr_z; // Essential mesh attributes (z component applied).
   Array<int> pres_ess_attr;
   Array<int> neumann_ess_attr;
   Array<int> robin_ess_attr;
   Array<int> amp_robin_ess_attr;

   // All essential true dofs.
   Array<int> vel_ess_tdof;
   Array<int> vel_ess_tdof_x;        // All essential true dofs x component.
   Array<int> vel_ess_tdof_y;        // All essential true dofs y component.
   Array<int> vel_ess_tdof_z;        // All essential true dofs z component.
   Array<int> pres_ess_tdof;


   int max_bdr_attributes = 0;

   bool well_posed = false; // True if pressure Dirichlet or Robin BCs are set for the pressure equation.

   // Bookkeeping for velocity dirichlet bcs.
   std::vector<VelDirichletBC_T> vel_dbcs;

   // Bookkeeping for componentwise velocity dirichlet bcs.
   std::vector<CompCoeffContainer> vel_dbcs_xyz;

   // Bookkeeping for pressure dirichlet bcs.
   std::vector<PresDirichletBC_T> pres_dbcs;

   // Bookkeeping for neumann bcs.
   std::vector<NeumannBC_T> neumann_dbcs;

   // Bookkeeping for robin bcs.
   std::vector<RobinBC_T> robin_dbcs;
   std::vector<AMP_RobinBC_T> amp_robin_dbcs;

   // Bookkeeping for acceleration (forcing) terms.
   std::vector<AccelTerm_T> accel_terms;

   int max_bdf_order = 3;
   int cur_step = 0;
   std::vector<real_t> dthist = {0.0, 0.0, 0.0};

   // BDFk/EXTk coefficients.
   real_t bd0 = 0.0;
   real_t bd1 = 0.0;
   real_t bd2 = 0.0;
   real_t bd3 = 0.0;
   real_t ab1 = 0.0;
   real_t ab2 = 0.0;
   real_t ab3 = 0.0;

   real_t vel_cfl = 0.0, cfl_old = 0.0;

   // Timers.
   StopWatch sw_setup, sw_step, sw_extrap, sw_curlcurl, sw_spsolve, sw_hsolve;

   // Print levels.
   int pl_mvsolve = 0;
   int pl_spsolve = 0;
   int pl_hsolve = 0;
   int pl_amg = 0;

   // Relative tolerances.
#if defined(MFEM_USE_DOUBLE)
   real_t rtol_mvsolve = 1e-12;
   real_t rtol_spsolve = 1e-6;
   real_t rtol_hsolve = 1e-8;
#elif defined(MFEM_USE_SINGLE)
   real_t rtol_mvsolve = 1e-9;
   real_t rtol_spsolve = 1e-5;
   real_t rtol_hsolve = 1e-7;
#else
#error "Only single and double precision are supported!"
   real_t rtol_mvsolve = 1e-12;
   real_t rtol_spsolve = 1e-6;
   real_t rtol_hsolve = 1e-8;
#endif

   // Iteration counts.
   int iter_mvsolve = 0, iter_spsolve = 0, iter_hsolve = 0;

   // Residuals.
   real_t res_mvsolve = 0.0, res_spsolve = 0.0, res_hsolve = 0.0;

   // LOR related.
   ParLORDiscretization *lor = nullptr;

   FilterMethod filter_method = FilterMethod::NONE;
   int filter_cutoff_modes = 1;
   real_t filter_alpha = 10.0;
   DenseMatrix FHPF, FHPFt;
   FiniteElementCollection *vfec_filter = nullptr;
   ParFiniteElementSpace *vfes_filter = nullptr;
   ParGridFunction u_filter_basis_gf, u_low_modes_gf, hpfrt_gf;

   ShearStressEvaluator *stress_evaluator = nullptr;
   std::unique_ptr<CurlEvaluator> curl_evaluator;
   std::unique_ptr<MeanEvaluator> mean_evaluator;

   Vector grad_nu_sym_grad_uext, kv;

   bool  own_mesh_velocity = false; 
   ParGridFunction *wg_gf = nullptr;       //< OWNED depending on own_mesh_velocity
   VectorCoefficient *wg_coef = nullptr;   //< OWNED
};

} // namespace navier

} // namespace mfem

#endif
