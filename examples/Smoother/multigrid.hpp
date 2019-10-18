#pragma once

#include "mfem.hpp"
#include "util.hpp"
#include "blkschwarzp.hpp"

namespace mfem {

class MGSolver : public Solver
{
private:
   /// The linear system matrix
   HypreParMatrix *Af;
   std::vector<HypreParMatrix *> A;
   std::vector<HypreParMatrix *> P;
   std::vector<ParSchwarzSmoother *> S;
   int NumGrids;
   PetscLinearSolver *petsc = nullptr;
   Solver *invAc = nullptr;
   double theta = 1.0;

public:
   MGSolver(HypreParMatrix *Af_, std::vector<HypreParMatrix *> P_, std::vector<ParFiniteElementSpace *> fespaces);

   virtual void SetOperator(const Operator &op) {}

   virtual void SetTheta(const double a) { theta = a; }

   virtual void Mult(const Vector &r, Vector &z) const;
   virtual ~MGSolver();
};

class BlockMGSolver : public Solver
{
private:
   /// The linear system matrix
   Array2D<HypreParMatrix *> Af;
   vector<Array<int>>Aoffsets;
   vector<Array<int>>Poffsets_i;
   vector<Array<int>>Poffsets_j;
   std::vector<Array2D<HypreParMatrix *>> A;
   std::vector<HypreParMatrix *> P;
   std::vector<BlockOperator *> BlkP;
   std::vector<BlockOperator *> BlkA;
   std::vector<BlkParSchwarzSmoother *> S;
   HypreParMatrix * Ac;
   int NumGrids;
   PetscLinearSolver *petsc = nullptr;
   Solver *invAc = nullptr;
   double theta = 1.0;

public:
   BlockMGSolver(Array2D<HypreParMatrix *> Af_, std::vector<HypreParMatrix *> P_, std::vector<ParFiniteElementSpace *> fespaces);

   virtual void SetOperator(const Operator &op) {}

   virtual void SetTheta(const double a) { theta = a; }

   virtual void Mult(const Vector &r, Vector &z) const;
   virtual ~BlockMGSolver();
};

} // namespace mfem
