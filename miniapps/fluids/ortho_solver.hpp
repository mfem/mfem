#pragma once

#include "mfem.hpp"

namespace mfem
{
namespace flow
{
class OrthoSolver : public Solver
{
public:
   OrthoSolver();

   virtual void SetOperator(const Operator &op);

   void Mult(const Vector &b, Vector &x) const;

private:
   const Operator *oper;

   mutable Vector b_ortho;

   void Orthoganalize(const Vector &v, Vector &v_ortho) const;
};
} // namespace flow
} // namespace mfem