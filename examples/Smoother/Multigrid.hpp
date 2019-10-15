#include "mfem.hpp"
#include "Schwarzp.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

class MGSolver : public Solver {
private:
   /// The linear system matrix
   HypreParMatrix * Af;
   std::vector<HypreParMatrix *> A;
   std::vector<HypreParMatrix *> P;
   std::vector<ParSchwarzSmoother  *> S;
   int NumGrids;
   PetscLinearSolver *petsc = nullptr;
   Solver * invAc=nullptr;
   double theta = 1.0;
public:

   MGSolver(HypreParMatrix * Af_, std::vector<HypreParMatrix *> P_, std::vector<ParFiniteElementSpace * > fespaces);

   virtual void SetOperator(const Operator &op) {}

   virtual void SetTheta(const double a) {theta = a;}

   virtual void Mult(const Vector &r, Vector &z) const;
   virtual ~MGSolver();
};



class BlockMGSolver : public Solver {
private:
   /// The linear system matrix
   Array2D<HypreParMatrix *> Af;
   std::vector<HypreParMatrix *> A;
   std::vector<HypreParMatrix *> P;
   std::vector<ParSchwarzSmoother  *> S;
   int NumGrids;
   PetscLinearSolver *petsc = nullptr;
   Solver * invAc=nullptr;
   double theta = 1.0;
public:

   BlockMGSolver(Array2D<HypreParMatrix *> Af_, std::vector<HypreParMatrix *> P_, std::vector<ParFiniteElementSpace * > fespaces);

   virtual void SetOperator(const Operator &op) {}

   virtual void SetTheta(const double a) {theta = a;}

   virtual void Mult(const Vector &r, Vector &z) const;
   virtual ~BlockMGSolver();
};