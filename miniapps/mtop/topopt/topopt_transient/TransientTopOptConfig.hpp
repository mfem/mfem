// =============================================================================
// Passive Configuration for Transient Topology Optimization
// =============================================================================

#ifndef TRANSIENT_TOPOPT_CONFIG_HPP
#define TRANSIENT_TOPOPT_CONFIG_HPP

#include "mfem.hpp"
#include "BoundaryLoadSpec.hpp"
#include "MaterialParams.hpp"
#include <string>

namespace mfem
{

struct TransientTopOptConfig
{
   std::string mesh_file = "lamb-problem-damping-mesh-triangs.msh";
   int ref_levels = 0;
   int order = 1;

   real_t t_final = 0.006;
   real_t dt = 5e-5;
   real_t vol_frac = 0.5;
   real_t filter_radius = 0.05;
   int max_it = 20;
   real_t move = 0.2;
   real_t change_tol = 1e-3;

   real_t x_max = 1.5;
   real_t y_max = 0.75;
   real_t damping_thickness = 0.25;
   real_t damping_scale_length = 0.2136;
   real_t damping_reflection = 1e-4;
   real_t damping_beta = 2.0;
   int damping_exponent = 2;

   real_t protected_radius = 0.2;

   MaterialParams material;
   BoundaryLoadSpec boundary_load;

   Array<int> essential_bdr_attributes;
   Array<int> absorbing_bdr_attributes;

   TransientTopOptConfig()
   {
      essential_bdr_attributes.SetSize(0);

      absorbing_bdr_attributes.SetSize(3);
      absorbing_bdr_attributes[0] = 10;
      absorbing_bdr_attributes[1] = 11;
      absorbing_bdr_attributes[2] = 12;
   }
};

} // namespace mfem

#endif // TRANSIENT_TOPOPT_CONFIG_HPP
