#include "linearized_newton_residual.hpp"
#include "navierstokes_operator.hpp"

using namespace mfem;

LinearizedTransientNewtonResidual::LinearizedTransientNewtonResidual(
   NavierStokesOperator &nav) :
   Operator(nav.offsets.Last()),
   nav(nav),
   z(nav.offsets) {}

void LinearizedTransientNewtonResidual::Mult(const Vector &xb, Vector &yb) const
{
   const BlockVector x(xb.GetData(), nav.offsets);
   BlockVector y(yb.GetData(), nav.offsets);

   const Vector &xu = x.GetBlock(0);
   Vector &zu = z.GetBlock(0);
   Vector &yu = y.GetBlock(0);

   // z_u = M_v * x_u
   nav.Mv->Mult(xu, zu);

   // y = dA(x)/dx * x
   nav.Ae->Mult(x, y);

   // M_v * x_u + dt * dA(x)/dx x
   yu.Add(dt, zu);
}

void LinearizedTransientNewtonResidual::Setup(const double dt)
{
   this->dt = dt;
}