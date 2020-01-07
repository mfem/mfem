#ifndef MFEM_ADVECTION
#define MFEM_ADVECTION

#include "../lib/hypsys.hpp"

class Advection : public HyperbolicSystem
{
public:
   bool SolutionKnown = true;
   bool WriteErrors = false;

	FiniteElementSpace *fes;
   SparseMatrix K;
   Vector b;

   explicit Advection(FiniteElementSpace *fes_, Configuration &config_);
   ~Advection() { };

	virtual void EvaluateFlux(const Vector &u, DenseMatrix &f) const;
   virtual void PreprocessProblem(FiniteElementSpace *fes, GridFunction &u);
};

#endif
