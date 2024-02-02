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

#ifndef MFEM_NAVIER_SOLVER_HPP
#define MFEM_NAVIER_SOLVER_HPP

#define NAVIER_VERSION 0.2

#include "mfem.hpp"
#include "kernels/curl_evaluator.hpp"
#include "kernels/mean_evaluator.hpp"
#include "kernels/shear_stress_evaluator.hpp"

namespace mfem
{
namespace navier
{
using VecFuncT = void(const Vector &x, double t, Vector &u);
using ScalarFuncT = double(const Vector &x, double t);

/// Container for a Dirichlet boundary condition of the velocity field.
class VelDirichletBC_T
{
public:
   VelDirichletBC_T(Array<int> attr, VectorCoefficient *coeff)
      : attr(attr), coeff(coeff)
   {}

   VelDirichletBC_T(VelDirichletBC_T &&obj)
   {
      // Deep copy the attribute array
      this->attr = obj.attr;

      // Move the coefficient pointer
      this->coeff = obj.coeff;
      obj.coeff = nullptr;
   }

   ~VelDirichletBC_T() { delete coeff; }

   Array<int> attr;
   VectorCoefficient *coeff;
};

/// Container for a Dirichlet boundary condition of the pressure field.
class PresDirichletBC_T
{
public:
   PresDirichletBC_T(Array<int> attr, Coefficient *coeff)
      : attr(attr), coeff(coeff)
   {}

   PresDirichletBC_T(PresDirichletBC_T &&obj)
   {
      // Deep copy the attribute array
      this->attr = obj.attr;

      // Move the coefficient pointer
      this->coeff = obj.coeff;
      obj.coeff = nullptr;
   }

   ~PresDirichletBC_T() { delete coeff; }

   Array<int> attr;
   Coefficient *coeff;
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
 *    requires a solve of the mass matrix in velocity space \f$M_v^{-1}\f$. On
 *    the other hand this introduces a CFL stability condition on the maximum
 *    timestep.
 *
 * 2. A Poisson solve \f$S_p^{-1}\f$.
 *
 * 3. A Helmholtz like solve \f$(M_v - \partial t K_v)^{-1}\f$.
 *
 * The numerical solver setup for each step are as follows.
 *
 * \f$M_v^{-1}\f$ is solved using CG with Jacobi as preconditioner.
 *
 * \f$S_p^{-1}\f$ is solved using CG with AMG applied to the low order refined
 * (LOR) assembled pressure Poisson matrix. To avoid assembling a matrix for
 * preconditioning, one can use p-MG as an alternative (NYI).
 *
 * \f$(M_v - \partial t K_v)^{-1}\f$ due to the CFL condition we expect the time
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
    * of the finite element spaces is this algorithm is of equal order
    * \f$(P_N)^d P_N\f$ for velocity and pressure respectively. This means the
    * pressure is in discretized in the same space (just scalar instead of a
    * vector space) as the velocity.
    *
    * Kinematic viscosity (dimensionless) is set using @a kin_vis and
    * automatically converted to the Reynolds number. If you want to set the
    * Reynolds number directly, you can provide the inverse.
    */
   NavierSolver(ParMesh *mesh, int order, double kin_vis, const double density = 1.0);

   /// Initialize forms, solvers and preconditioners.
   void Setup(double dt);

   void UpdateSpaces();

   void UpdateForms();

   void UpdateSolvers();

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
   void Step(double &time, double dt, int cur_step, bool provisional = false);

   /// Return a pointer to the provisional velocity ParGridFunction.
   ParGridFunction *GetProvisionalVelocity() { return &un_next_gf; }

   /// Return a pointer to the current velocity ParGridFunction.
   ParGridFunction *GetCurrentVelocity() { return &un_gf; }

   /// Return a pointer to the current pressure ParGridFunction.
   ParGridFunction *GetCurrentPressure() { return &pn_gf; }

   BilinearFormIntegrator *GetPressureEquationBLFI() { return *(Sp_form->GetDBFI()[0]); }

   /// Add a Dirichlet boundary condition to the velocity field.
   void AddVelDirichletBC(VectorCoefficient *coeff, Array<int> &attr);

   void AddVelDirichletBC(VecFuncT *f, Array<int> &attr);

   /// Add a Dirichlet boundary condition to the pressure field.
   void AddPresDirichletBC(Coefficient *coeff, Array<int> &attr);

   void AddPresDirichletBC(ScalarFuncT *f, Array<int> &attr);

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

   /// Remove mean from a Vector.
   /**
    * Modify the Vector @a v by subtracting its mean using
    * \f$v = v - \frac{\sum_i^N v_i}{N} \f$
    */
   void Orthogonalize(Vector &v);

   /// Remove the mean from a ParGridFunction.
   /**
    * Modify the ParGridFunction @a v by subtracting its mean using
    * \f$ v = v - \int_\Omega \frac{v}{vol(\Omega)} dx \f$.
    */
   void MeanZero(ParGridFunction &v);

   /// Rotate entries in the time step and solution history arrays.
   void UpdateTimestepHistory(double dt);

   /// Set the maximum order to use for the BDF method.
   void SetMaxBDFOrder(int maxbdforder) { max_bdf_order = maxbdforder; };

   /// Compute CFL
   double ComputeCFL(ParGridFunction &u, double dt);

   int PredictTimestep(const double dt_min, const double dt_max,
                       const double cfl_target, double &dt);

   void EnableFilter(FilterMethod f);

   /// Set the number of modes to cut off in the interpolation filter
   void SetCutoffModes(int c) { filter_cutoff_modes = c; }

   /// Set the interpolation filter parameter @a a
   /**
    * If @a a is > 0, the filtering algorithm for the velocity field after every
    * time step from [1] is used. The parameter should be 0 > @a >= 1.
    *
    * [1] Paul Fischer, Julia Mullen (2001) Filter-based stabilization of
    * spectral element methods
    */
   void SetFilterAlpha(double a) { filter_alpha = a; }

   ParGridFunction *GetVariableViscosity()
   {
      return &kin_vis_gf;
   }

   ParGridFunction *GetCurrentMeshVelocity() { return &wgn_gf; }

   void TransformMesh(VectorCoefficient &dx);

   double NekNorm(ParGridFunction &u, int type, bool is_vector);

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

   /// Eliminate essential BCs in an Operator and apply to RHS.
   void EliminateRHS(Operator &A,
                     ConstrainedOperator &constrainedA,
                     const Array<int> &ess_tdof_list,
                     Vector &x,
                     Vector &b,
                     Vector &X,
                     Vector &B,
                     int copy_interior = 0);

   void BuildHPFForcing(ParGridFunction &vel_gf);

   // Evaluate Legendre Polynomials of order N at point x
   void EvaluateLegendrePolynomial(const int poly_order, const double x,
                                   Vector &L);

   // Evaluate Legendre Polynomials of order N at point x
   void EvaluateLegendrePolynomialShifted(const int poly_order, const double x,
                                          Vector &L);

   // Evaluate Monomial Basis of order N at point x
   void EvaluateMonomialBasis(const int poly_order, const double x, Vector &L);

   /// Enable/disable debug output.
   bool debug = false;

   /// Enable/disable verbose output.
   bool verbose = true;

   /// Enable/disable variable viscosity (stress formulation)
   bool variable_viscosity = true;

   /// The parallel mesh.
   ParMesh *pmesh = nullptr;

   /// The order of the velocity and pressure space.
   int order;

   /// Kinematic viscosity (dimensionless).
   double kin_vis;

   /// Fluid density
   double density;

   ///
   ParGridFunction kin_vis_gf;

   GridFunctionCoefficient kin_vis_gf_coeff;

public:
   IntegrationRules gll_rules;
   IntegrationRule gll_ir;
   IntegrationRule gll_ir_face;
   IntegrationRule gll_ir_nl;
protected:

   /// Velocity \f$H^1\f$ finite element collection.
   FiniteElementCollection *vfec = nullptr;

   FiniteElementCollection *vfecL2 = nullptr;

   /// Pressure \f$H^1\f$ finite element collection.
   FiniteElementCollection *pfec = nullptr;

   /// Velocity \f$(H^1)^d\f$ finite element space.
   ParFiniteElementSpace *vfes = nullptr;

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

   ParLinearForm *FText_bdr_form = nullptr;

   ParLinearForm *f_form = nullptr;

   ParLinearForm *g_bdr_form = nullptr;

   /// Linear form to compute the mass matrix in various subroutines.
   ParLinearForm *mass_lf = nullptr;
   ParLinearForm *component_mass_lf = nullptr;
   ConstantCoefficient onecoeff;

public:
   double volume = 0.0;
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

   ParGridFunction un_gf, un_next_gf, curlu_gf, curlcurlu_gf, Lext_gf, FText_gf,
                   resu_gf;

   ParGridFunction pn_gf, resp_gf;

   // All essential attributes.
   Array<int> vel_ess_attr;
   Array<int> pres_ess_attr;

   // All essential true dofs.
   Array<int> vel_ess_tdof;
   Array<int> pres_ess_tdof;

   // Bookkeeping for velocity dirichlet bcs.
   std::vector<VelDirichletBC_T> vel_dbcs;

   // Bookkeeping for pressure dirichlet bcs.
   std::vector<PresDirichletBC_T> pres_dbcs;

   // Bookkeeping for acceleration (forcing) terms.
   std::vector<AccelTerm_T> accel_terms;

   int max_bdf_order = 3;
   int cur_step = 0;
   std::vector<double> dthist = {0.0, 0.0, 0.0};

   // BDFk/EXTk coefficients.
   double bd0 = 0.0;
   double bd1 = 0.0;
   double bd2 = 0.0;
   double bd3 = 0.0;
   double ab1 = 0.0;
   double ab2 = 0.0;
   double ab3 = 0.0;

   double vel_cfl = 0.0, cfl_old = 0.0;

   // Timers.
   StopWatch sw_setup, sw_step, sw_extrap, sw_curlcurl, sw_spsolve, sw_hsolve;

   // Print levels.
   int pl_mvsolve = 0;
   int pl_spsolve = 0;
   int pl_hsolve = 0;
   int pl_amg = 0;

   // Relative tolerances.
   double rtol_spsolve = 1e-4;
   double rtol_hsolve = 1e-6;

   // Iteration counts.
   int iter_mvsolve = 0, iter_spsolve = 0, iter_hsolve = 0;

   // Residuals.
   double res_mvsolve = 0.0, res_spsolve = 0.0, res_hsolve = 0.0;

   // LOR related.
   ParLORDiscretization *lor = nullptr;

   FilterMethod filter_method = FilterMethod::NONE;
   int filter_cutoff_modes = 1;
   double filter_alpha = 10.0;
   DenseMatrix FHPF, FHPFt;
   FiniteElementCollection *vfec_filter = nullptr;
   ParFiniteElementSpace *vfes_filter = nullptr;
   ParGridFunction u_filter_basis_gf, u_low_modes_gf, hpfrt_gf;

   ShearStressEvaluator *stress_evaluator = nullptr;
   std::unique_ptr<CurlEvaluator> curl_evaluator;
   std::unique_ptr<MeanEvaluator> mean_evaluator;

   Vector grad_nu_sym_grad_uext, kv;

   ParGridFunction wgn_gf;
   VectorGridFunctionCoefficient wg_coef;
};

} // namespace navier

} // namespace mfem

#endif
