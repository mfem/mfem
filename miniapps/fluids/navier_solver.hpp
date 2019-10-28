#pragma once

#include "mfem.hpp"
#include "ortho_solver.hpp"

namespace mfem
{
namespace navier
{
typedef void(VecFuncT)(const Vector &x, double t, Vector &u);
typedef double(ScalarFuncT)(const Vector &x, double t);

/// Container for a Dirichlet boundary condition of the velocity field.
class VelDirichletBC_T
{
public:
   VelDirichletBC_T(VecFuncT *f,
                    Array<int> attr,
                    VectorFunctionCoefficient coeff)
      : f(f), attr(attr), coeff(coeff)
   {}

   VecFuncT *f;
   Array<int> attr;
   VectorFunctionCoefficient coeff;
};

/// Container for a Dirichlet boundary condition of the pressure field.
class PresDirichletBC_T
{
public:
   PresDirichletBC_T(ScalarFuncT *f,
                     Array<int> attr,
                     FunctionCoefficient coeff)
      : f(f), attr(attr), coeff(coeff)
   {}

   ScalarFuncT *f;
   Array<int> attr;
   FunctionCoefficient coeff;
};

/// Container for an acceleration term.
class AccelTerm_T
{
public:
   AccelTerm_T(VecFuncT *f,
               Array<int> attr,
               VectorFunctionCoefficient coeff)
      : f(f), attr(attr), coeff(coeff)
   {}

   VecFuncT *f;
   Array<int> attr;
   VectorFunctionCoefficient coeff;
};

/// Navier Stokes solver.
/**
 * Transient Navier Stokes solver in a split scheme formulation.
 */
class NavierSolver
{
public:
   NavierSolver(ParMesh *mesh, int order, double kin_vis);

   void Setup(double dt);
   
   /// Compute provisional solution at the next time step t+dt.
   /**
    * Compute provisional solution at the next time step t+dt without
    * automatically accepting the solution. The method should be used
    * in combination with UpdateTimestepHistory if the user decides that
    * the solution fulfills all a posteriori requirements.
    */
   void ProvisionalStep(double time, double dt, int cur_step);

   /// Compute solution at the next time step t+dt.
   /**
    * Compute solution at the next time step t+dt and automatically
    * accept the solution. This method should be used when using constant
    * time steps.
    */
   void Step(double time, double dt, int cur_step);

   /// Return a pointer to the current velocity ParGridFunction.
   ParGridFunction *GetCurrentVelocity() { return &un_gf; }

   /// Return a pointer to the provisional velocity ParGridFunction.
   ParGridFunction *GetProvisionalVelocity() { return &un_next_gf; }

   /// Return a pointer to the current pressure ParGridFunction.
   ParGridFunction *GetCurrentPressure() { return &pn_gf; }

   /// Add a Dirichlet boundary condition to the velocity field.
   void AddVelDirichletBC(VecFuncT *f, Array<int> &attr);

   /// Add a Dirichlet boundary condition to the pressure field.
   void AddPresDirichletBC(ScalarFuncT *f, Array<int> &attr);

   /// Add an accelaration term to the RHS of the equation.
   /**
    * The VectorFunction \p f is evaluated at the current time t
    * and extrapolated with the nonlinear parts of the Navier Stokes
    * equation.
    */
   void AddAccelTerm(VecFuncT *f, Array<int> &attr);

   /// Enable partial assembly for every operator.
   void EnablePA(bool pa = true) { partial_assembly = pa; }

   /// Enable numerical integration rules.
   void EnableNI(bool ni = true) { numerical_integ = ni; }

   void EnableDebug(bool d = true) { debug = d; }

   void EnableVerbose(bool v = true) { verbose = v; }

   /// Rotate entries in the time step and solution history arrays.
   void UpdateTimestepHistory(double dt);

   /// Set the maximum order to use for the BDF method.
   void SetMaxBDFOrder(int maxbdforder) { max_bdf_order = maxbdforder; };

   /// Compute $\nabla times \nabla times u$ for $u \in (H^1)^2$
   void ComputeCurl2D(ParGridFunction &u,
                      ParGridFunction &cu,
                      bool assume_scalar = false);

   /// Compute $\nabla times \nabla times u$ for $u \in (H^1)^3$
   void ComputeCurl3D(ParGridFunction &u, ParGridFunction &cu);

   /// Compute the global maximum cell wise CFL number.
   double ComputeCFL(ParGridFunction &u, double &dt);

   void PrintTimingData();

   ~NavierSolver();

protected:
   void PrintInfo();

   // Set time integration coefficient based on the time step
   // history. This works with variable and constant step size.
   // For details of computation of the coefficient
   // see [Wang and Ruuth, JSTOR, 2008].
   void SetTimeIntegrationCoefficients(int step);

   void Orthogonalize(Vector &v);

   void MeanZero(ParGridFunction &v);

   void EliminateRHS(Operator &A,
                     ConstrainedOperator &constrainedA,
                     const Array<int> &ess_tdof_list,
                     Vector &x,
                     Vector &b,
                     Vector &X,
                     Vector &B,
                     int copy_interior = 0);

   bool debug = false;
   bool verbose = false;
   bool partial_assembly = false;
   bool numerical_integ = false;

   ParMesh *pmesh;

   double order;
   double kin_vis;

   IntegrationRules rules_ni;

   FiniteElementCollection *vfec;
   FiniteElementCollection *pfec;
   ParFiniteElementSpace *vfes;
   ParFiniteElementSpace *pfes;

   ParNonlinearForm *N;
   ParBilinearForm *Mv_form;
   ParBilinearForm *Sp_form;
   ParMixedBilinearForm *D_form;
   ParMixedBilinearForm *G_form;
   ParBilinearForm *H_form;

   VectorGridFunctionCoefficient *FText_gfcoeff;
   ParLinearForm *FText_bdr_form;
   ParLinearForm *g_bdr_form;
   ParLinearForm *f_form;
   ParLinearForm *mass_lf = nullptr;

   ConstantCoefficient onecoeff;
   double volume = 0.0;

   ConstantCoefficient nlcoeff;
   ConstantCoefficient Sp_coeff;
   ConstantCoefficient H_lincoeff;
   ConstantCoefficient H_bdfcoeff;

   OperatorHandle Mv;
   OperatorHandle Sp;
   OperatorHandle D;
   OperatorHandle G;
   OperatorHandle H;

   Solver *MvInvPC;
   CGSolver *MvInv;

   Solver *SpInvPC;
   OrthoSolver *SpInvOrthoPC;
   CGSolver *SpInv;

   Solver *HInvPC;
   CGSolver *HInv;

   Vector fn, un, un_next, unm1, unm2, Nun, Nunm1, Nunm2, Fext, FText, Lext,
      resu;
   Vector tmp1;

   Vector pn, resp, FText_bdr, g_bdr;

   ParGridFunction un_gf, un_next_gf, curlu_gf, curlcurlu_gf, Lext_gf, FText_gf,
      resu_gf;

   ParGridFunction pn_gf, resp_gf;

   // All essential attributes
   Array<int> vel_ess_attr;
   Array<int> pres_ess_attr;

   // All essential true dofs
   Array<int> vel_ess_tdof;
   Array<int> pres_ess_tdof;

   // Bookkeeping for velocity dirichlet bcs
   std::vector<VelDirichletBC_T> vel_dbcs;

   // Bookkeeping for pressure dirichlet bcs
   std::vector<PresDirichletBC_T> pres_dbcs;

   // Bookkeeping for acceleration (forcing) terms
   std::vector<AccelTerm_T> accel_terms;

   int max_bdf_order = 3;
   int cur_step = 0;
   std::vector<double> dthist = {0.0, 0.0, 0.0};

   // BDFk/EXTk coefficients
   double bd0;
   double bd1;
   double bd2;
   double bd3;
   double ab1;
   double ab2;
   double ab3;

   // Timers
   StopWatch sw_setup, sw_step, sw_single_step, sw_extrap, sw_curlcurl,
      sw_spsolve, sw_hsolve;

   // Printlevels
   int pl_mvsolve = 0;
   int pl_spsolve = 0;
   int pl_hsolve = 0;
   int pl_amg = 0;

   // Tolerances
   double rtol_spsolve = 1e-12;
   double rtol_hsolve = 1e-12;

   // Iteration counts
   int iter_mvsolve, iter_spsolve, iter_hsolve;

   // Residuals
   double res_mvsolve, res_spsolve, res_hsolve;

   // LOR PC related
   ParMesh *pmesh_lor;
   FiniteElementCollection *pfec_lor;
   ParFiniteElementSpace *pfes_lor;
   ParBilinearForm *Sp_form_lor;
   OperatorHandle Sp_lor;
};
} // namespace navier
} // namespace mfem
