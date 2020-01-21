#ifndef MFEM_ADVECTION
#define MFEM_ADVECTION

#include "../hypsys.hpp"

class Advection : public HyperbolicSystem
{
public:
   explicit Advection(FiniteElementSpace *fes_, DofInfo &dofs_, const Vector &LumpedMassMat_, Configuration &config_);
   ~Advection() { };

	virtual void EvaluateFlux(const Vector &u, DenseMatrix &f) const;
	virtual void ComputeErrors(Array<double> &errors, double DomainSize) const;
	virtual void WriteErrors(const Array<double> &errors) const;
};

#endif
