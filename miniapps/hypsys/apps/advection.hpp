#ifndef HYPSYS_ADVECTION
#define HYPSYS_ADVECTION

#include "hyperbolic_system.hpp"
#include "../lib/dofs.hpp"

class Advection : public HyperbolicSystem
{
public:
   explicit Advection(FiniteElementSpace *fes_, BlockVector &u_block,
                      Configuration &config_, bool NodalQuadRule);
   ~Advection() { };

   virtual void EvaluateFlux(const Vector &u, DenseMatrix &FluxEval,
                             int e, int k, int i = -1) const;
   virtual double GetWaveSpeed(const Vector &u, const Vector n, int e, int k,
                               int i) const;
   virtual void ComputeErrors(Array<double> &errors, const GridFunction &u,
                              double DomainSize, double t) const override;

   int nqf;
   DenseTensor VelElem, VelFace;
   mutable Vector VelocityVector;
};

#endif
