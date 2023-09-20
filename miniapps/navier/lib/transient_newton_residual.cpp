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
   const Vector &xp = x.GetBlock(1);
   Vector &zu = z.GetBlock(0);
   Vector &zp = z.GetBlock(1);
   Vector &yu = y.GetBlock(0);
   Vector &yp = y.GetBlock(1);

   nav.K->Mult(xu, yu);
   yu *= dt;

   if (nav.convection && !nav.convection_explicit)
   {
      nav.n_form->Mult(xu, zu);
      zu *= dt;
      yu += zu;
   }

   nav.G->Mult(xp, zu);
   zu *= dt;
   yu += zu;
  
   nav.D->Mult(xu, yp);
      
   nav.Mv->Mult(xu, zu);
   yu += zu;

   zu = nav.fu_rhs;
   zu.Neg();
   zu *= dt;
   yu += zu;

   yu.SetSubVector(nav.vel_ess_tdofs, 0.0);
}

Operator& TransientNewtonResidual::GetGradient(const Vector &x) const
{
   // Build preconditioner

   if (nav.Amonoe == nullptr) {
      nav.RebuildPC(x);
   }
   return *nav.Amonoe;

   // fd_linearized.reset(new FDJacobian(*this, x));
   // return *fd_linearized;
}

void TransientNewtonResidual::Setup(const double dt)
{
   this->dt = dt;
   linearized.Setup(dt);
}