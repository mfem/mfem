#pragma once

#include <mfem.hpp>

namespace mfem
{

class CahouetChabardPC : public Solver
{
public:
   CahouetChabardPC(Solver &Mp_inv, Solver &Lp_inv, const double dt,
                    Array<int> &pres_ess_tdofs);

   void Mult(const Vector &x, Vector &y) const override;

   void SetOperator(const Operator &op) override { }

private:
   Solver &Mp_inv;
   Solver &Lp_inv;
   OperatorHandle Fp;
   mutable Vector z;
   double dt;
   Array<int> &pres_ess_tdofs;
};

}