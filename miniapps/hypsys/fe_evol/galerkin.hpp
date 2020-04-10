#ifndef HYPSYS_GALERKINEVOLUTION
#define HYPSYS_GALERKINEVOLUTION

#include "fe_evol.hpp"

using namespace std;
using namespace mfem;

class GalerkinEvolution : public FE_Evolution
{
public:
   explicit GalerkinEvolution(FiniteElementSpace *fes_,
                              HyperbolicSystem *hyp_, DofInfo &dofs_);

   virtual ~GalerkinEvolution() { }

   void Mult(const Vector&x, Vector &y) const override;

   void ComputeTimeDerivative(const Vector &x, Vector &y,
                              const Vector &xMPI = serial) const;
};

#endif