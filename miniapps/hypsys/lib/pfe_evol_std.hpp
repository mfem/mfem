#ifndef HYPSYS_PARSTANDARDEVOLUTION
#define HYPSYS_PARSTANDARDEVOLUTION

#include "pfe_evol.hpp"
#include "fe_evol_std.hpp"

using namespace std;
using namespace mfem;

class ParStandardEvolution : public ParFE_Evolution, public StandardEvolution
{
public:
   explicit ParStandardEvolution(ParFiniteElementSpace *fes_,
                                 HyperbolicSystem *hyp_, DofInfo &dofs_,
                                 EvolutionScheme scheme_);

   virtual ~ParStandardEvolution() { }

   void Mult(const Vector&x, Vector &y) const override;
};

#endif