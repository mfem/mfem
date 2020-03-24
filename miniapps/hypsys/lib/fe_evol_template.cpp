#include "fe_evol_template.hpp"

TEMPLATE::TEMPLATE(FiniteElementSpace *fes_, HyperbolicSystem *hyp_,
                   DofInfo &dofs_, EvolutionScheme scheme_)
   : FE_Evolution(fes_, hyp_, dofs_, scheme_)
{
   // TODO
}

void TEMPLATE::Mult(const Vector &x, Vector &y) const
{
   ComputeTimeDerivative(x, y);
}

void TEMPLATE::ComputeTimeDerivative(const Vector &x, Vector &y,
                                     const Vector &xMPI) const
{
   // TODO
}