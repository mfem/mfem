#ifndef HYPSYS_SHALLOWWATER
#define HYPSYS_SHALLOWWATER

#include "hypsys.hpp"

class ShallowWater : public HyperbolicSystem
{
public:
   explicit ShallowWater(FiniteElementSpace *fes_, BlockVector &u_block,
                         Configuration &config_);
   ~ShallowWater() { };

   virtual void EvaluateFlux(const Vector &u, DenseMatrix &f,
                             int e, int k, int i = -1) const;
   virtual double GetWaveSpeed(const Vector &u, const Vector n, int e, int k,
                               int i) const;
   virtual void ComputeErrors(Array<double> &errors, const GridFunction &u,
                              double DomainSize, double t) const;
};

#endif
