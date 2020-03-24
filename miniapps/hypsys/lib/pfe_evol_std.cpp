#include "pfe_evol_std.hpp"

ParStandardEvolution::ParStandardEvolution(ParFiniteElementSpace *pfes_,
                                           HyperbolicSystem *hyp_,
                                           DofInfo &dofs_,
                                           EvolutionScheme scheme_)
   : StandardEvolution(pfes_, hyp_, dofs_, scheme_), x_gf_MPI(pfes_) { }

void ParStandardEvolution::Mult(const Vector &x, Vector &y) const
{
   x_gf_MPI = x;
   x_gf_MPI.ExchangeFaceNbrData();
   Vector &xMPI = x_gf_MPI.FaceNbrData();
   ComputeTimeDerivative(x, y, xMPI);
}