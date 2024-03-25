#pragma once

#include <mfem.hpp>
#include "util.hpp"

namespace mfem
{

// Convenience interface for BlockLowerTriangularPreconditioner with Schur
// complement approximation.
class BlockSchurPC : public Solver
{
public:
   BlockSchurPC(Array<int> &offsets, Solver &Finv, Operator &D,
                Solver &Sinv, Array<int> vel_ess_tdofs) :
      Solver(offsets.Last()),
      offsets(offsets),
      Finv(Finv),
      D(D),
      Sinv(Sinv),
      vel_ess_tdofs(vel_ess_tdofs)
   {
      P = new BlockLowerTriangularPreconditioner(offsets);
      P->SetBlock(0, 0, &Finv);
      P->SetBlock(1, 0, &D);
      P->SetBlock(1, 1, &Sinv);
   }

   void Mult(const Vector &x, Vector &y) const override
   {
      mfem::BlockVector yb(y.GetData(), offsets);

      P->Mult(x, y);
   }

   void SetOperator(const Operator &op) override {};

   Array<int> offsets;
   Solver &Finv;
   Operator &D;
   Solver &Sinv;
   Array<int> vel_ess_tdofs;
   BlockLowerTriangularPreconditioner *P = nullptr;
};

}