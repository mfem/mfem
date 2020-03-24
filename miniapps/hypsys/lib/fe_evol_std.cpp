#include "fe_evol_std.hpp"

StandardEvolution::StandardEvolution(FiniteElementSpace *fes_,
                                     HyperbolicSystem *hyp_, DofInfo &dofs_,
                                     EvolutionScheme scheme_)
   : FE_Evolution(fes_, hyp_, dofs_, scheme_)
{
   // TODO
}

void StandardEvolution::Mult(const Vector &x, Vector &y) const
{
   ComputeTimeDerivative(x, y);
}

void StandardEvolution::ComputeTimeDerivative(const Vector &x, Vector &y,
                                              const Vector &xMPI) const
{
   // TODO
}