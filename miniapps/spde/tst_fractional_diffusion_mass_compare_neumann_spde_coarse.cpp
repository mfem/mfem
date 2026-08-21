// Run the Neumann fractional diffusion+mass comparison with an SPDE solve on
// the coarsest level of the additive multilevel generator.

#define MFEM_FRACTIONAL_MG_GENERATOR FracRandomFieldGeneratorSPDE
#define MFEM_FRACTIONAL_MG_DESCRIPTION "additive MG+SPDE coarse"
#define MFEM_FRACTIONAL_MG_FIELD "additive_mg_spde_coarse"
#define MFEM_FRACTIONAL_OUTPUT_NAME \
   "fractional_diffusion_mass_compare_neumann_spde_coarse"

#include "tst_fractional_diffusion_mass_compare_neumann.cpp"
