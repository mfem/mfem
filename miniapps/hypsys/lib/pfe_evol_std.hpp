#ifndef HYPSYS_PARSTANDARDEVOLUTION
#define HYPSYS_PARSTANDARDEVOLUTION

#include "fe_evol_std.hpp"
#include "pdofs.hpp"
#include "ptools.hpp"

using namespace std;
using namespace mfem;

class ParStandardEvolution : public StandardEvolution
{
public:
   mutable ParGridFunction x_gf_MPI;

   explicit ParStandardEvolution(ParFiniteElementSpace *fes_,
                                 HyperbolicSystem *hyp_, DofInfo &dofs_,
                                 EvolutionScheme scheme_);

   virtual ~ParStandardEvolution() { }

   void Mult(const Vector&x, Vector &y) const override;
};

#endif