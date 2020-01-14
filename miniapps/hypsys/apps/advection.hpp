#ifndef MFEM_ADVECTION
#define MFEM_ADVECTION

#include "../hypsys.hpp"

class Advection : public HyperbolicSystem
{
public:
   bool SolutionKnown = true; // TODO
   bool WriteErrors = false;

	FiniteElementSpace *fes;

   explicit Advection(FiniteElementSpace *fes_, Configuration &config_);
   ~Advection() { };

	virtual void EvaluateFlux(const Vector &u, DenseMatrix &f) const;
   virtual void PreprocessProblem(FiniteElementSpace *fes, GridFunction &u);
	virtual void PostprocessProblem(const GridFunction &u);
};

#endif
