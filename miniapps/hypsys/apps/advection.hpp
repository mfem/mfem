#ifndef HYPSYS_ADVECTION
#define HYPSYS_ADVECTION

#include "hypsys.hpp"

class Advection : public HyperbolicSystem
{
public:
   explicit Advection(FiniteElementSpace *fes_, BlockVector &u_block,
							 Configuration &config_);
   ~Advection() { };

   virtual void EvaluateFlux(const Vector &u, DenseMatrix &f) const;
   virtual void ComputeErrors(Array<double> &errors, double DomainSize,
                              const GridFunction &u) const;
   virtual void WriteErrors(const Array<double> &errors) const;
};

#endif
