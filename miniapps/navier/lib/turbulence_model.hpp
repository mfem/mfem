#ifndef MFEM_NAVIER_TURBULENCE_MODEL
#define MFEM_NAVIER_TURBULENCE_MODEL

#include <mfem.hpp>

namespace mfem
{
namespace navier
{
/**
 * @brief Abstract base class for a turbulence model.
 */
class TurbulenceModel
{
public:
   /// Compute the turbulent/eddy viscosity.
   virtual void ComputeEddyViscosity(ParGridFunction &nu) = 0;
};
}
}

#endif