#ifndef HYPSYS_PFE_EVOL
#define HYPSYS_PFE_EVOL

#include "fe_evol.hpp"
#include "pdofs.hpp"
#include "ptools.hpp"

using namespace std;
using namespace mfem;


class ParFE_Evolution : public FE_Evolution
{
public:
   ParFiniteElementSpace *pfes;
   mutable ParGridFunction x_gf_MPI;

   ParFE_Evolution(ParFiniteElementSpace *pfes_, HyperbolicSystem *hyp_,
                   DofInfo &dofs_, EvolutionScheme scheme_);

   virtual ~ParFE_Evolution() { };

   void Mult(const Vector &x, Vector &y) const override;
   double ConvergenceCheck(double dt, double tol, const Vector &u) const override;
};

#endif
