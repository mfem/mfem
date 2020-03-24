#ifndef HYPSYS_PARSCHEME
#define HYPSYS_PARSCHEME

#include "pfe_evol.hpp"
#include "scheme.hpp"

using namespace std;
using namespace mfem;

class PARSCHEME : public ParFE_Evolution, public SCHEME
{
public:
   explicit PARSCHEME(ParFiniteElementSpace *fes_, HyperbolicSystem *hyp_,
                      DofInfo &dofs_, EvolutionScheme scheme_);

   virtual ~PARSCHEME() { }

   void Mult(const Vector&x, Vector &y) const override;
};

#endif