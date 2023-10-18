#include "transient_newton_residual.hpp"
#include "navierstokes_operator.hpp"
#include "util.hpp"
#include <caliper/cali.h>

using namespace mfem;

TransientNewtonResidual::TransientNewtonResidual(NavierStokesOperator &nav) :
   Operator(nav.offsets.Last()),
   nav(nav),
   z(nav.offsets) {}

void TransientNewtonResidual::Mult(const Vector &xb, Vector &yb) const
{
   CALI_MARK_BEGIN("TransientNewtonResidual::Mult");
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

   nav.G->Mult(xp, zu);
   zu *= dt;
   yu += zu;

   nav.D->Mult(xu, yp);

   nav.Mv->Mult(xu, zu);
   yu += zu;

   yu.SetSubVector(nav.vel_ess_tdofs, 0.0);
   yp.SetSubVector(nav.pres_ess_tdofs, 0.0);
   CALI_MARK_BEGIN("TransientNewtonResidual::Mult");
}

Operator& TransientNewtonResidual::GetGradient(const Vector &x) const
{
   // Build preconditioner

   if (nav.Amonoe_matfree == nullptr || rebuild_pc)
   {
      nav.RebuildPC(x);
      rebuild_pc = false;
   }
   return *nav.Amonoe_matfree;

   // fd_linearized.reset(new FDJacobian(*this, x));
   // return *fd_linearized;
}

void TransientNewtonResidual::Setup(const double dt)
{
   if (this->dt != dt)
   {
      rebuild_pc = true;
   }
   this->dt = dt;
}