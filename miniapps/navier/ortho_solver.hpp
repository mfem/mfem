#ifndef MFEM_NAVIER_ORTHO_SOLVER_HPP
#define MFEM_NAVIER_ORTHO_SOLVER_HPP

#include "mfem.hpp"

namespace mfem
{
namespace navier
{
class OrthoSolver : public Solver
{
public:
   OrthoSolver();

   virtual void SetOperator(const Operator &op);

   void Mult(const Vector &b, Vector &x) const;

private:
   const Operator *oper = nullptr;

   mutable Vector b_ortho;

   void Orthoganalize(const Vector &v, Vector &v_ortho) const;
};
} // namespace navier
} // namespace mfem
#endif