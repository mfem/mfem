#include "pfe_evol_std.hpp"

ParStandardEvolution::ParStandardEvolution(ParFiniteElementSpace *fes_,
                                           HyperbolicSystem *hyp_,
                                           DofInfo &dofs_,
                                           EvolutionScheme scheme_)
   : ParFE_Evolution(fes_, hyp_, dofs_, scheme_),
     StandardEvolution(fes_, hyp_, dofs_, scheme_) { }

void ParStandardEvolution::Mult(const Vector &x, Vector &y) const
{
   x_gf_MPI = x;
   x_gf_MPI.ExchangeFaceNbrData();
   Vector &xMPI = x_gf_MPI.FaceNbrData();
   ComputeTimeDerivative(x, y, xMPI);
}