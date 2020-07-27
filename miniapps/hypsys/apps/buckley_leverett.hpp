#ifndef HYPSYS_BUCKLEYLEVERETT
#define HYPSYS_BUCKLEYLEVERETT

#include "hyperbolic_system.hpp"

class BuckleyLeverett : public HyperbolicSystem
{
public:
   explicit BuckleyLeverett(FiniteElementSpace *fes_, BlockVector &u_block,
                     Configuration &config_);
   ~BuckleyLeverett() { };

   virtual void EvaluateFlux(const Vector &u, DenseMatrix &FluxEval,
                             int e, int k, int i = -1) const;
   virtual double GetWaveSpeed(const Vector &u, const Vector n, int e, int k,
                               int i) const;
};

#endif
