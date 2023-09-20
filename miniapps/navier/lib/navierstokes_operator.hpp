#pragma once

#include <transient_newton_residual.hpp>
#include <mfem.hpp>

namespace mfem
{

using VelDirichletBC = std::pair<VectorCoefficient *, Array<int> *>;

/// Constructs an operator of the form
///
/// [ a_m * M + K(a_k)    a_d * D^t ] [ u ] = [ f - N(w) ]
/// [ a_d * D                   0   ] [ p ]   [ 0        ]
///
/// where a_m, a_k and a_d are functions that scale the mass matrix, the viscous
/// stress and the gradient/divergence matrix respectively.
/// N(w) is the convective term in the Navier-Stokes equations and w represents
/// a velocity value that is already known. That means there is no nonlinear
/// contribution to the implicit terms. The term can be enabled/disabled.
class NavierStokesOperator : public Operator
{
   friend class TransientNewtonResidual;
   friend class LinearizedTransientNewtonResidual;

public:
   /// @brief NavierStokesOperator
   /// @param vel_fes
   /// @param pres_fes
   /// @param vel_ess_bdr
   /// @param pres_ess_bdr
   NavierStokesOperator(ParFiniteElementSpace &vel_fes,
                        ParFiniteElementSpace &pres_fes,
                        std::vector<VelDirichletBC> velocity_dbcs,
                        const Array<int> pres_ess_bdr,
                        ParGridFunction &u_gf,
                        bool convection = true,
                        bool convection_explicit = false,
                        bool matrix_free = false);

   void Mult(const Vector &x, Vector &y) const override;

   void MultExplicit(const Vector &x, Vector &y) const;

   void MultImplicit(const Vector &x, Vector &y) const;

   void Step(BlockVector &X, double &t, const double dt);

   void SetForcing(VectorCoefficient *f);

   void Setup(const double dt);

   void Assemble();

   void SetTime(double t);

   const Array<int>& GetOffsets() const;

   void ProjectVelocityDirichletBC(Vector &v);

protected:
   /// @brief Set the parameters and prepare forms
   /// @param am_coeff Coefficient that scales mass matrix
   /// @param ak_coeff Coefficient that scales viscous stress matrix
   /// @param ad_coeff Coefficient that scales gradient/divergence matrix
   void SetParameters(Coefficient *am_coeff, Coefficient *ak_coeff,
                      Coefficient *ad_coeff);

   void RebuildPC(const Vector &x);

   ParFiniteElementSpace &vel_fes;
   ParFiniteElementSpace &pres_fes;

   ParGridFunction &kinematic_viscosity;

   std::unique_ptr<ParGridFunction> ak_gf;
   std::unique_ptr<ParGridFunction> vel_bc_gf;

   Array<int> vel_ess_bdr;
   Array<int> pres_ess_bdr;

   Array<int> vel_ess_tdofs;
   Array<int> pres_ess_tdofs;

   std::vector<VelDirichletBC> velocity_dbcs;

   bool convection;
   bool convection_explicit;
   bool matrix_free;

   Array<int> offsets;

   IntegrationRules intrules;
   IntegrationRule ir, ir_nl;

   double time = 0.0;

   Coefficient *am_coeff = nullptr;
   Coefficient *ak_coeff = nullptr;
   Coefficient *ad_coeff = nullptr;
   VectorCoefficient *forcing_coeff = nullptr;
   Coefficient *am_mono_coeff = nullptr;
   Coefficient *ak_mono_coeff = nullptr;

   ParBilinearForm *mv_form = nullptr;
   ParBilinearForm *mp_form = nullptr;
   ParBilinearForm *k_form = nullptr;
   ParMixedBilinearForm *d_form = nullptr;
   ParMixedBilinearForm *g_form = nullptr;
   ParMixedBilinearForm *dmono_form = nullptr;
   ParMixedBilinearForm *gmono_form = nullptr;
   ParNonlinearForm *mdta_form = nullptr;
   ParNonlinearForm *n_form = nullptr;
   OperatorHandle Mv;
   OperatorHandle Mp;
   OperatorHandle K;
   OperatorHandle Ke;
   OperatorHandle D;
   OperatorHandle De;
   OperatorHandle G;
   OperatorHandle MdtAe, Dmonoe, Gmonoe;
   TransposeOperator *Dte = nullptr;
   mutable Operator *Ne = nullptr;
   BlockOperator *A = nullptr;
   BlockOperator *Ae = nullptr;
   HypreParMatrix *Amonoe = nullptr;

   std::unique_ptr<TransientNewtonResidual> trans_newton_residual;
   std::unique_ptr<MUMPSSolver> mumps;

   STRUMPACKRowLocMatrix *Amonoe_rowloc = nullptr;
   std::unique_ptr<STRUMPACKSolver> strumpack;

   Solver *krylov = nullptr;
   Solver *pc = nullptr;
   Solver *solver = nullptr;
   Solver *MvInv = nullptr;
   Solver *MvInvPC = nullptr;
   Solver *MpInv = nullptr;
   Solver *MpInvPC = nullptr;

   ParLinearForm *forcing_form = nullptr;
   Vector fu_rhs;

   mutable BlockVector z, w, Y;

   Vector vel_bc_tdof;

   mutable double cached_dt = -1.0;
};

}