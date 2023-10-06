#pragma once

#include <transient_newton_residual.hpp>
#include <mfem.hpp>

namespace mfem
{

using VelDirichletBC = std::pair<VectorCoefficient *, Array<int> *>;
using PresDirichletBC = std::pair<Coefficient *, Array<int> *>;

struct CahouetChabardPC : Solver
{
   Solver &Mp_inv;
   Solver &Lp_inv;
   OperatorHandle Fp;
   mutable Vector z;
   double dt;
   Array<int> &pres_ess_tdofs;
   CahouetChabardPC(Solver &Mp_inv_, Solver &Lp_inv_, const double dt, Array<int> &pres_ess_tdofs) :
      Solver(Mp_inv_.Height()),
      Mp_inv(Mp_inv_),
      Lp_inv(Lp_inv_),
      dt(dt),
      z(Mp_inv_.Height()),
      pres_ess_tdofs(pres_ess_tdofs) { }
   void Mult(const Vector &x, Vector &y) const
   {
      z.SetSize(y.Size());

      Lp_inv.Mult(x, y);
      y /= dt;
      Mp_inv.Mult(x, z);
      y += z;

      for (int i = 0; i < pres_ess_tdofs.Size(); i++)
      {
         y[pres_ess_tdofs[i]] = x[pres_ess_tdofs[i]];
      }
   }
   void SetOperator(const Operator &op) { }
};

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
   friend class CahouetChabardPC;

public:
   /// @brief NavierStokesOperator
   /// @param vel_fes
   /// @param pres_fes
   /// @param vel_ess_bdr
   /// @param pres_ess_bdr
   NavierStokesOperator(ParFiniteElementSpace &vel_fes,
                        ParFiniteElementSpace &pres_fes,
                        std::vector<VelDirichletBC> velocity_dbcs,
                        std::vector<PresDirichletBC> pressure_dbcs,
                        ParGridFunction &u_gf,
                        bool convection = true,
                        bool convection_explicit = true,
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

   void ProjectPressureDirichletBC(Vector &p);

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
   std::unique_ptr<ParGridFunction> vel_bc_gf, pres_bc_gf;

   Array<int> vel_ess_bdr;
   Array<int> pres_ess_bdr;

   Array<int> vel_ess_tdofs;
   Array<int> pres_ess_tdofs;

   Array<int> schur_ess_tdofs;

   std::vector<VelDirichletBC> velocity_dbcs;
   std::vector<PresDirichletBC> pressure_dbcs;

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
   ConstantCoefficient zero_coeff;

   ParBilinearForm *mv_form = nullptr;
   ParBilinearForm *mp_form = nullptr;
   ParBilinearForm *lp_form = nullptr;
   ParBilinearForm *k_form = nullptr;
   ParMixedBilinearForm *d_form = nullptr;
   ParMixedBilinearForm *g_form = nullptr;
   ParMixedBilinearForm *dmono_form = nullptr;
   ParMixedBilinearForm *gmono_form = nullptr;
   ParBilinearForm *cmono_form = nullptr;   
   ParBilinearForm *mdta_form = nullptr;
   ParNonlinearForm *n_form = nullptr;
   OperatorHandle Mv;
   OperatorHandle Mp;
   OperatorHandle Lp;
   OperatorHandle K;
   OperatorHandle Ke;
   OperatorHandle D;
   OperatorHandle De;
   OperatorHandle G;
   OperatorHandle MdtAe, Dmonoe, Gmonoe, Cmonoe;
   TransposeOperator *Dte = nullptr;
   mutable Operator *Ne = nullptr;
   BlockOperator *A = nullptr;
   BlockOperator *Ae = nullptr;
   HypreParMatrix *Amonoe = nullptr;
   BlockOperator *Amonoe_matfree = nullptr;

   std::unique_ptr<TransientNewtonResidual> trans_newton_residual;
   std::unique_ptr<MUMPSSolver> mumps;
   std::unique_ptr<Solver> pc;
   std::unique_ptr<Solver> MdtAinv;
   std::unique_ptr<Solver> MdtAinvPC;
   std::unique_ptr<Solver> Mpinv;
   std::unique_ptr<Solver> MpinvPC;
   std::unique_ptr<Solver> Lpinv;
   std::unique_ptr<Solver> LpinvPC;
   std::unique_ptr<Operator> SchurInv;

   STRUMPACKRowLocMatrix *Amonoe_rowloc = nullptr;
   std::unique_ptr<STRUMPACKSolver> strumpack;

   Solver *krylov = nullptr;
   Solver *solver = nullptr;

   CahouetChabardPC *Sinv = nullptr;

   ParLinearForm *forcing_form = nullptr;
   Vector fu_rhs;

   mutable BlockVector z, w, Y;

   Vector vel_bc_tdof;

   mutable double cached_dt = -1.0;
};

}