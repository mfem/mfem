#ifndef HYPSYS_KPP
#define HYPSYS_KPP

#include "hyperbolic_system.hpp"

class KPP : public HyperbolicSystem
{
public:
   explicit KPP(FiniteElementSpace *fes_, BlockVector &u_block,
                Configuration &config_);
   ~KPP() { };

   virtual void EvaluateFlux(const Vector &u, DenseMatrix &FluxEval,
                             int e, int k, int i = -1) const;
   virtual double GetWaveSpeed(const Vector &u, const Vector n, int e, int k,
                               int i) const;
   virtual void ComputeErrors(Array<double> &errors, const GridFunction &u,
                              double DomainSize, double t) const;
};

#endif
