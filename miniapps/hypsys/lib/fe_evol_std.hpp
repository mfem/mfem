#ifndef HYPSYS_STANDARDEVOLUTION
#define HYPSYS_STANDARDEVOLUTION

#include "fe_evol.hpp"

using namespace std;
using namespace mfem;

static Vector serial;

class StandardEvolution : public FE_Evolution
{
public:
   explicit StandardEvolution(FiniteElementSpace *fes_,
                              HyperbolicSystem *hyp_, DofInfo &dofs_,
                              EvolutionScheme scheme_);

   virtual ~StandardEvolution() { }

   void Mult(const Vector&x, Vector &y) const override;
   void ComputeTimeDerivative(const Vector &x, Vector &y,
                              const Vector &xMPI = serial) const;
};

#endif