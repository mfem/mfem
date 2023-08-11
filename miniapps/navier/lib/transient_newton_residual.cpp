#include "transient_newton_residual.hpp"
#include "navierstokes_operator.hpp"
#include "util.hpp"

using namespace mfem;

TransientNewtonResidual::TransientNewtonResidual(NavierStokesOperator &nav) :
   Operator(nav.offsets.Last()),
   nav(nav),
   linearized(nav),
   z(nav.offsets) {}

void TransientNewtonResidual::Mult(const Vector &xb, Vector &yb) const
{
   const BlockVector x(xb.GetData(), nav.offsets);
   BlockVector y(yb.GetData(), nav.offsets);

   const Vector &xu = x.GetBlock(0);
   Vector &zu = z.GetBlock(0);
   Vector &zp = z.GetBlock(1);
   Vector &yu = y.GetBlock(0);
   Vector &yp = y.GetBlock(1);

   nav.A->Mult(x, y);

   if (nav.convection && !nav.convection_explicit)
   {
      // ParNonlinearForm::Mult can be applied to a T-vector.
      nav.n_form->Mult(xu, zu);
      add(yu, zu, yu);
   }

   nav.Mv->Mult(xu, zu);

   subtract(-dt, yu, nav.fu_rhs, yu);
   add(yu, zu, yu);

   yu.SetSubVector(nav.vel_ess_tdofs, 0.0);
}

Operator& TransientNewtonResidual::GetGradient(const Vector &x) const
{
   fd_linearized.reset(new FDJacobian(*this, x));
   return *fd_linearized;
   // return linearized;
}

void TransientNewtonResidual::Setup(const double dt)
{
   this->dt = dt;
   linearized.Setup(dt);
}