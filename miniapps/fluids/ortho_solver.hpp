#pragma once

#include "mfem.hpp"

namespace mfem
{
namespace flow
{
class OrthoSolver : public Solver
{
public:
   OrthoSolver(MPI_Comm comm);

   virtual void SetOperator(const Operator &op);

   void Mult(const Vector &b, Vector &x) const;

private:
   MPI_Comm comm;
   const Operator *oper;

   void Orthoganalize(Vector &v) const;
};
} // namespace flow
} // namespace mfem