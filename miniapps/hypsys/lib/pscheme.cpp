#include "pscheme.hpp"

PARSCHEME::PARSCHEME(ParFiniteElementSpace *fes_, HyperbolicSystem *hyp_,
                     DofInfo &dofs_, EvolutionScheme scheme_)
   : ParFE_Evolution(fes_, hyp_, dofs_, scheme_),
     SCHEME(fes_, hyp_, dofs_, scheme_) { }

void PARSCHEME::Mult(const Vector &x, Vector &y) const
{
   x_gf_MPI = x;
   x_gf_MPI.ExchangeFaceNbrData();
   Vector &xMPI = x_gf_MPI.FaceNbrData();
   ComputeTimeDerivative(x, y, xMPI);
}