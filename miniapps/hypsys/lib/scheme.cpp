#include "scheme.hpp"

SCHEME::SCHEME(FiniteElementSpace *fes_, HyperbolicSystem *hyp_,
               DofInfo &dofs_, EvolutionScheme scheme_)
   : FE_Evolution(fes_, hyp_, dofs_, scheme_)
{
   // TODO
}

void SCHEME::Mult(const Vector &x, Vector &y) const
{
   ComputeTimeDerivative(x, y);
}

void SCHEME::ComputeTimeDerivative(const Vector &x, Vector &y,
                                   const Vector &xMPI) const
{
   // TODO
}