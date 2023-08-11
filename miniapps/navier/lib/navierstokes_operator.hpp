#pragma once

#include <transient_newton_residual.hpp>
#include <mfem.hpp>

namespace mfem
{

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
class NavierStokesOperator : public TimeDependentOperator
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
                        const Array<int> vel_ess_bdr,
                        const Array<int> pres_ess_bdr,
                        ParGridFunction &u_gf,
                        bool convection = true,
                        bool convection_explicit = false,
                        bool matrix_free = false);

   void Mult(const Vector &x, Vector &y) const override;

   void ImplicitSolve(const double dt, const Vector &x, Vector &k) override;

   void SetForcing(VectorCoefficient *f);

   void Setup(const double dt);

   void Assemble();

   void SetTime(double t) override;

   const Array<int>& GetOffsets() const;

protected:
   /// @brief Set the parameters and prepare forms
   /// @param am_coeff Coefficient that scales mass matrix
   /// @param ak_coeff Coefficient that scales viscous stress matrix
   /// @param ad_coeff Coefficient that scales gradient/divergence matrix
   void SetParameters(Coefficient *am_coeff, Coefficient *ak_coeff,
                      Coefficient *ad_coeff);

   ParFiniteElementSpace &vel_fes;
   ParFiniteElementSpace &pres_fes;

   ParGridFunction &kinematic_viscosity;

   std::unique_ptr<ParGridFunction> ak_gf;

   const Array<int> vel_ess_bdr;
   const Array<int> pres_ess_bdr;

   Array<int> vel_ess_tdofs;
   Array<int> pres_ess_tdofs;

   bool convection;
   bool convection_explicit;
   bool matrix_free;

   Array<int> offsets;

   double time = 0.0;

   Coefficient *am_coeff = nullptr;
   Coefficient *ak_coeff = nullptr;
   Coefficient *ad_coeff = nullptr;
   VectorCoefficient *forcing_coeff = nullptr;

   ParBilinearForm *mv_form = nullptr;
   ParBilinearForm *mp_form = nullptr;
   ParBilinearForm *k_form = nullptr;
   ParMixedBilinearForm *d_form = nullptr;
   ParMixedBilinearForm *g_form = nullptr;
   ParNonlinearForm *n_form = nullptr;
   OperatorHandle Mv;
   OperatorHandle Mp;
   OperatorHandle K;
   OperatorHandle Ke;
   OperatorHandle D;
   OperatorHandle De;
   OperatorHandle Dt;
   TransposeOperator *Dte = nullptr;
   mutable Operator *Ne = nullptr;
   BlockOperator *A = nullptr;
   BlockOperator *Ae = nullptr;

   std::unique_ptr<TransientNewtonResidual> trans_newton_residual;

   Solver *krylov = nullptr;
   Solver *pc = nullptr;
   Solver *solver = nullptr;
   Solver *MvInv = nullptr;
   Solver *MvInvPC = nullptr;
   Solver *MpInv = nullptr;
   Solver *MpInvPC = nullptr;

   /// Local matrices for LU decomposition. Only used for testing.
   SparseMatrix *K_local = nullptr;
   Solver *K_inv_LU = nullptr;

   ParLinearForm *forcing_form = nullptr;
   Vector fu_rhs;

   mutable BlockVector z, w;

   mutable double cached_dt = -1.0;
};

}