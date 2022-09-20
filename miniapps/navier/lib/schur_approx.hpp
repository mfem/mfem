#pragma once

#include <mfem.hpp>

namespace mfem
{
class SchurApproxInv : public Operator
{
public:
   SchurApproxInv(const double dt, Solver *Mpinv, Solver *Lpinv) :
      Operator(Mpinv->Height()),
      dt(dt),
      Mpinv(Mpinv),
      Lpinv(Lpinv),
      z(Mpinv->Height()) {}

   void Mult(const Vector &x, Vector &y) const
   {
      Mpinv->Mult(x, y);
      y *= dt;
      Lpinv->Mult(x, z);
      y += z;
   }

   const double dt;
   Solver *Mpinv;
   Solver *Lpinv;
   mutable Vector z;
};
}