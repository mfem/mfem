// =============================================================================
// Material Parameters for Transient Topology Optimization
// =============================================================================

#ifndef MATERIAL_PARAMS_HPP
#define MATERIAL_PARAMS_HPP

#include "mfem.hpp"

namespace mfem
{

struct MaterialParams
{
   real_t rho0 = 1.0;
   real_t lambda0 = 2.0;
   real_t mu0 = 1.0;
   real_t r_min = 1e-6;
   real_t r_max = 1.0;
   real_t simp_p = 3.0;
};

} // namespace mfem

#endif // MATERIAL_PARAMS_HPP
