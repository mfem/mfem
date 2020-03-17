#ifndef HYPSYS_BURGERS
#define HYPSYS_BURGERS

#include "hyperbolic_system.hpp"

class Burgers : public HyperbolicSystem
{
public:
   explicit Burgers(FiniteElementSpace *fes_, BlockVector &u_block,
                    Configuration &config_);
   ~Burgers() { };

   virtual void EvaluateFlux(const Vector &u, DenseMatrix &FluxEval,
                             int e, int k, int i = -1) const;
   virtual double GetWaveSpeed(const Vector &u, const Vector n, int e, int k, int i) const;
   virtual double EvaluateBdrCond(const Vector &inflow, const Vector &x, const Vector &normal,
                                  int n, int e, int i, int attr, int DofInd) const;
   virtual void ComputeErrors(Array<double> &errors, const GridFunction &u,
                              double DomainSize, double t) const;
};

#endif
