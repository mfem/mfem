
   
#pragma once

#include "mfem.hpp"
#include "schwarz.hpp"

class MGSolver : public Solver
{
private:
   /// The linear system matrix
   HypreParMatrix *Af;
   std::vector<HypreParMatrix *> A;
   std::vector<HypreParMatrix *> P;
   std::vector<SchwarzSmoother *> S;
   int NumGrids;
   MUMPSSolver *mumps = nullptr;
   Solver *invAc = nullptr;
   double theta = 1.0;

public:
   MGSolver(HypreParMatrix *Af_, std::vector<HypreParMatrix *> P_, std::vector<ParFiniteElementSpace *> fespaces);

   virtual void SetOperator(const Operator &op) {}

   virtual void SetTheta(const double a) { theta = a; }

   virtual void Mult(const Vector &r, Vector &z) const;
   virtual ~MGSolver();
};

class ComplexMGSolver : public Solver
{
private:
   /// The linear system matrix
   ComplexHypreParMatrix *Af;
   std::vector<ComplexHypreParMatrix *> A;
   std::vector<HypreParMatrix *> P;
   std::vector<ComplexSchwarzSmoother *> S;
   int NumGrids;
   ComplexMUMPSSolver *mumps = nullptr;
   Solver *invAc = nullptr;
   double theta = 1.0;

public:
   ComplexMGSolver(ComplexHypreParMatrix *Af_, 
   std::vector<HypreParMatrix *> P_, std::vector<ParFiniteElementSpace *> fespaces);

   virtual void SetOperator(const Operator &op) {}

   virtual void SetTheta(const double a) { theta = a; }

   virtual void Mult(const Vector &r, Vector &z) const;
   virtual ~ComplexMGSolver();
};