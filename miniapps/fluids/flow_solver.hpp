#pragma once

#include "mfem.hpp"
#include "ortho_solver.hpp"

namespace mfem
{
namespace flow
{
typedef void(VecFuncT)(const Vector &x, double t, Vector &u);
typedef double(ScalarFuncT)(const Vector &x, double t);

class VelDirichletBC_T
{
public:
   VelDirichletBC_T(void (*f)(const Vector &x, double t, Vector &u),
                    Array<int> attr,
                    VectorFunctionCoefficient coeff)
      : f(f), attr(attr), coeff(coeff)
   {}

   void (*f)(const Vector &x, double t, Vector &u);
   Array<int> attr;
   VectorFunctionCoefficient coeff;
};

class PresDirichletBC_T
{
public:
   PresDirichletBC_T(double (*f)(const Vector &x, double t),
                    Array<int> attr,
                    FunctionCoefficient coeff)
      : f(f), attr(attr), coeff(coeff)
   {}

   double (*f)(const Vector &x, double t);
   Array<int> attr;
   FunctionCoefficient coeff;
};

class FlowSolver
{
public:
   FlowSolver(ParMesh *mesh, int order, double kin_vis);

   void Setup(double dt);
   void Step(double &time, double dt, int cur_step);

   ParGridFunction *GetCurrentVelocity() { return &un_gf; }

   ParGridFunction *GetCurrentPressure() { return &pn_gf; }

   void AddVelDirichletBC(void (*f)(const Vector &x, double t, Vector &u),
                         Array<int> &attr);

   void AddPresDirichletBC(double (*f)(const Vector &x, double t),
                          Array<int> &attr);

   void PrintTimingData();

   ~FlowSolver();

protected:
   void PrintInfo();
   void SetTimeIntegrationCoefficients(int step);
   void ComputeCurlCurl(ParGridFunction &u, ParGridFunction &ccu);
   void ComputeCurl2D(ParGridFunction &u,
                      ParGridFunction &cu,
                      bool assume_scalar = false);
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

   bool verbose = true;
   bool partial_assembly = true;
   bool numerical_integ = true;

   ParMesh *pmesh;

   double order;
   double kin_vis;

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

   ParLinearForm *mass_lf = nullptr;
   ConstantCoefficient onecoeff;
   double volume;

   ConstantCoefficient nlcoeff;
   ConstantCoefficient H_lincoeff;
   ConstantCoefficient H_bdfcoeff;

   OperatorHandle Mv;
   OperatorHandle Sp;
   OperatorHandle D;
   OperatorHandle G;
   OperatorHandle H;

   Solver *MvInvPC;
   CGSolver *MvInv;

   HypreBoomerAMG *SpInvPC;
   OrthoSolver *SpInvOrthoPC;
   CGSolver *SpInv;

   Solver *HInvPC;
   CGSolver *HInv;

   Vector fn, un, unm1, unm2, Nun, Nunm1, Nunm2, Fext, FText, Lext, resu;
   Vector tmp1;

   Vector pn, resp, FText_bdr, g_bdr;

   ParGridFunction un_gf, curlu_gf, curlcurlu_gf, Lext_gf, FText_gf, resu_gf;

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

   // Bookkeeping for forcing (acceleration) terms
   // std::vector<void (*)(const Vector &x, double t, Vector &u)> acc_terms;

   int cur_step;

   // BDFk/EXTk coefficients
   double bd0;
   double bd1;
   double bd2;
   double bd3;
   double ab1;
   double ab2;
   double ab3;

   // Timers
   StopWatch sw_setup, sw_step, sw_extrap, sw_curlcurl, sw_spsolve, sw_hsolve;

   // Printlevels
   int pl_mvsolve = 0;
   int pl_spsolve = 0;
   int pl_hsolve = 0;
   int pl_amg = 0;

   // Tolerances
   double rtol_spsolve = 1e-5;
   double rtol_hsolve = 1e-6;

   // Iteration counts
   int iter_spsolve, iter_hsolve;

   // Residuals
   double res_spsolve, res_hsolve;

   // LOR PC related
   ParMesh *pmesh_lor;
   FiniteElementCollection *vfec_lor;
   FiniteElementCollection *pfec_lor;
   ParFiniteElementSpace *vfes_lor;
   ParFiniteElementSpace *pfes_lor;
   InterpolationGridTransfer *vgt, *pgt;

   ParBilinearForm *Mv_form_lor;
   ParBilinearForm *Sp_form_lor;
   ParBilinearForm *H_form_lor;

   OperatorHandle Mv_lor;
   OperatorHandle Sp_lor;
   OperatorHandle H_lor;
};
} // namespace flow
} // namespace mfem
