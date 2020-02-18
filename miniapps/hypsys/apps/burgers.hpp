#ifndef HYPSYS_BURGERS
#define HYPSYS_BURGERS

#include "hypsys.hpp"

class Burgers : public HyperbolicSystem
{
public:
   explicit Burgers(FiniteElementSpace *fes_, BlockVector &u_block,
                    Configuration &config_);
   ~Burgers(){};

   virtual void EvaluateFlux(const Vector &u, DenseMatrix &f,
                             int e, int k, int i = -1) const;
   virtual double GetWaveSpeed(const Vector &u, const Vector n, int e, int k,
                               int i) const;
   virtual void ComputeErrors(Array<double> &errors, const GridFunction &u,
                              double DomainSize, double t) const;
   // virtual void WriteErrors(const Array<double> &errors) const;
};

#endif
