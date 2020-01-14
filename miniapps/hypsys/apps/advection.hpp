#ifndef MFEM_ADVECTION
#define MFEM_ADVECTION

#include "../hypsys.hpp"

class Advection : public HyperbolicSystem
{
public:
   bool SolutionKnown = true;
	double InitialMass;

   explicit Advection(FiniteElementSpace *fes_, Configuration &config_);
   ~Advection();

	virtual void EvaluateFlux(const Vector &u, DenseMatrix &f) const;
};

#endif
