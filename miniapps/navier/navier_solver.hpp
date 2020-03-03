#pragma once

#include "mfem.hpp"
#include "ortho_solver.hpp"

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
   PresDirichletBC_T(ScalarFuncT *f, Array<int> attr, FunctionCoefficient coeff)
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
   AccelTerm_T(VecFuncT *f, Array<int> attr, VectorFunctionCoefficient coeff)
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

   /// Compute solution at the next time step t+dt.
   void Step(double &time, double dt, int cur_step);

   /// Return a pointer to the current velocity ParGridFunction.
   ParGridFunction *GetCurrentVelocity() { return &un_gf; }

   /// Return a pointer to the current pressure ParGridFunction.
   ParGridFunction *GetCurrentPressure() { return &pn_gf; }

   /// Add a Dirichlet boundary condition to the velocity field.
   void AddVelDirichletBC(VecFuncT *f, Array<int> &attr);

   /// Add a Dirichlet boundary condition to the pressure field.
   void AddPresDirichletBC(ScalarFuncT *f, Array<int> &attr);

   /// Add an accelaration term to the RHS of the equation.
   /**
    * The VecFuncT \p @f is evaluated at the current time t
    * and extrapolated with the nonlinear parts of the Navier Stokes
    * equation.
    */
   void AddAccelTerm(VecFuncT *f, Array<int> &attr);

   /// Enable partial assembly for every operator.
   void EnablePA(bool pa) { partial_assembly = pa; }

   /// Enable numerical integration rules.
   void EnableNI(bool ni) { numerical_integ = ni; }

   void PrintTimingData();

   ~NavierSolver();

   /// Compute $\nabla times \nabla times u$ for $u \in (H^1)^2$
   void ComputeCurl2D(ParGridFunction &u,
                      ParGridFunction &cu,
                      bool assume_scalar = false);

   /// Compute $\nabla times \nabla times u$ for $u \in (H^1)^3$
   void ComputeCurl3D(ParGridFunction &u, ParGridFunction &cu);

protected:
   void PrintInfo();
   
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

   bool debug = true;
   bool verbose = true;
   bool partial_assembly = false;
   bool numerical_integ = false;

   ParMesh *pmesh = nullptr;

   int order;
   double kin_vis;

   FiniteElementCollection *vfec = nullptr;
   FiniteElementCollection *pfec = nullptr;
   ParFiniteElementSpace *vfes = nullptr;
   ParFiniteElementSpace *pfes = nullptr;

   ParNonlinearForm *N = nullptr;
   ParBilinearForm *Mv_form = nullptr;
   ParBilinearForm *Sp_form = nullptr;
   ParMixedBilinearForm *D_form = nullptr;
   ParMixedBilinearForm *G_form = nullptr;
   ParBilinearForm *H_form = nullptr;

   VectorGridFunctionCoefficient *FText_gfcoeff = nullptr;
   ParLinearForm *FText_bdr_form = nullptr;
   ParLinearForm *f_form = nullptr;
   ParLinearForm *g_bdr_form = nullptr;

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

   Solver *MvInvPC = nullptr;
   CGSolver *MvInv = nullptr;

   HypreBoomerAMG *SpInvPC = nullptr;
   OrthoSolver *SpInvOrthoPC = nullptr;
   CGSolver *SpInv = nullptr;

   Solver *HInvPC = nullptr;
   CGSolver *HInv = nullptr;

   Vector fn, un, unm1, unm2, Nun, Nunm1, Nunm2, Fext, FText, Lext, resu;
   Vector tmp1;

   Vector pn, resp, FText_bdr, g_bdr;

   ParGridFunction un_gf, curlu_gf, curlcurlu_gf, Lext_gf, FText_gf, resu_gf;

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

   int cur_step = 0;

   // BDFk/EXTk coefficients
   double bd0 = 0.0;
   double bd1 = 0.0;
   double bd2 = 0.0;
   double bd3 = 0.0;
   double ab1 = 0.0;
   double ab2 = 0.0;
   double ab3 = 0.0;

   // Timers
   StopWatch sw_setup, sw_step, sw_extrap, sw_curlcurl, sw_spsolve, sw_hsolve;

   // Printlevels
   int pl_mvsolve = 0;
   int pl_spsolve = 0;
   int pl_hsolve = 0;
   int pl_amg = 0;

   // Tolerances
   double rtol_spsolve = 1e-6;
   double rtol_hsolve = 1e-8;

   // Iteration counts
   int iter_mvsolve = 0, iter_spsolve = 0, iter_hsolve = 0;

   // Residuals
   double res_mvsolve = 0.0, res_spsolve = 0.0, res_hsolve = 0.0;

   // LOR PC related
   ParMesh *pmesh_lor = nullptr;
   FiniteElementCollection *vfec_lor = nullptr;
   FiniteElementCollection *pfec_lor = nullptr;
   ParFiniteElementSpace *vfes_lor = nullptr;
   ParFiniteElementSpace *pfes_lor = nullptr;
   InterpolationGridTransfer *vgt = nullptr, *pgt = nullptr;

   ParBilinearForm *Mv_form_lor = nullptr;
   ParBilinearForm *Sp_form_lor = nullptr;
   ParBilinearForm *H_form_lor = nullptr;

   OperatorHandle Mv_lor;
   OperatorHandle Sp_lor;
   OperatorHandle H_lor;
};
} // namespace navier
} // namespace mfem
