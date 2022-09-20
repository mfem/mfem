#include "prandtl_kolmogorov.hpp"

using namespace mfem;
using namespace navier;

PrandtlKolmogorov::PrandtlKolmogorov(ParMesh &mesh, const int order,
                                     ParGridFunction &velgf,
                                     ParGridFunction &nugf,
                                     ParGridFunction &wdgf) :
   ScalarEquation(mesh, order, velgf),
   velgf(velgf),
   nugf(nugf),
   wdgf(wdgf),
   viscosity_coeff(nugf, k_gf, wdgf, mu),
   reaction_coeff(k_gf, velgf, wdgf, mu),
   eddy_viscosity_coeff(k_gf, wdgf, mu)
{
   nugf.ExchangeFaceNbrData();
   wdgf.ExchangeFaceNbrData();

   // Disable diffusion for now. There currently is no PA version of DG
   // Diffusion and therefore significantly slows down the computation.
   auto zero_coeff = new ConstantCoefficient(0.0);
   SetViscosityCoefficient(*zero_coeff);
   // SetViscosityCoefficient(viscosity_coeff);

   AddReaction(reaction_coeff);
}

void PrandtlKolmogorov::ComputeEddyViscosity(ParGridFunction &nu)
{
   nu.ProjectCoefficient(eddy_viscosity_coeff);
}