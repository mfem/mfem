#pragma once

#include "linearized_newton_residual.hpp"
#include "util.hpp"
#include <mfem.hpp>

namespace mfem
{

class NavierStokesOperator;

class TransientNewtonResidual : public Operator
{
public:
   TransientNewtonResidual(NavierStokesOperator &nav);

   /// Compute F(x) = Mx - dt * f(x)
   ///              = Mx + dt * R(x)
   void Mult(const Vector &x, Vector &y) const override;

   /// Return an Operator that represents
   /// M + dt*dA(x)/dx
   Operator &GetGradient(const Vector &x) const override;

   void Setup(const double dt);

   NavierStokesOperator &nav;
   mutable LinearizedTransientNewtonResidual linearized;
   mutable std::shared_ptr<FDJacobian> fd_linearized;
   mutable BlockVector z;
   double dt;
   mutable bool rebuild_pc = true;
};

}