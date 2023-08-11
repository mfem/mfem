#pragma once

#include <mfem.hpp>

namespace mfem
{

class NavierStokesOperator;

class LinearizedTransientNewtonResidual : public Operator
{
public:
   LinearizedTransientNewtonResidual(NavierStokesOperator &nav);

   /// Compute y = (M + dt*dA(x)/dx) * x
   void Mult(const Vector &x, Vector &y) const override;

   void Setup(const double dt);

   NavierStokesOperator &nav;
   double dt;
   mutable BlockVector z;
};

}