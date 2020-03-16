#ifndef HYPSYS_EULER
#define HYPSYS_EULER

#include "hyperbolic_system.hpp"

class Euler : public HyperbolicSystem
{
public:
   explicit Euler(FiniteElementSpace *fes_, BlockVector &u_block,
                  Configuration &config_);
   ~Euler() { };

   virtual double EvaluatePressure(const Vector &u) const;

   virtual void EvaluateFlux(const Vector &u, DenseMatrix &FluxEval,
                             int e, int k, int i = -1) const;
   virtual double GetWaveSpeed(const Vector &u, const Vector n, int e, int k,
                               int i) const;
   virtual void ComputeErrors(Array<double> &errors, const GridFunction &u,
                              double DomainSize, double t) const;
};

#endif
