#ifndef HYPSYS_SCHEME
#define HYPSYS_SCHEME

#include "fe_evol.hpp"

using namespace std;
using namespace mfem;

static Vector serial;

class SCHEME : public FE_Evolution
{
public:
   explicit SCHEME(FiniteElementSpace *fes_, HyperbolicSystem *hyp_,
                   DofInfo &dofs_, EvolutionScheme scheme_);

   virtual ~SCHEME() { }

   void Mult(const Vector&x, Vector &y) const override;
   void ComputeTimeDerivative(const Vector &x, Vector &y,
                              const Vector &xMPI = serial) const;
};

#endif