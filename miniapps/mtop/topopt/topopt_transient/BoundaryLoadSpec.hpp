// =============================================================================
// Boundary Loading Specification for Transient Elastodynamics
// =============================================================================

#ifndef BOUNDARY_LOAD_SPEC_HPP
#define BOUNDARY_LOAD_SPEC_HPP

#include "mfem.hpp"

namespace mfem
{

enum class LoadTimeProfile
{
   CONSTANT,
   GAUSSIAN,
   HARMONIC
};

inline const char *LoadTimeProfileName(LoadTimeProfile profile)
{
   switch (profile)
   {
      case LoadTimeProfile::CONSTANT: return "Constant";
      case LoadTimeProfile::GAUSSIAN: return "Smooth Gaussian pulse";
      case LoadTimeProfile::HARMONIC: return "Harmonic";
   }
   return "Unknown";
}

struct BoundaryLoadSpec
{
   Array<int> bdr_attributes;
   Vector direction;
   LoadTimeProfile time_profile = LoadTimeProfile::GAUSSIAN;
   real_t amplitude = 30.0;
   real_t duration = 0.005;
   real_t phase = 0.0;
   real_t frequency = 1.0;

   BoundaryLoadSpec()
   {
      bdr_attributes.SetSize(6);
      for (int i = 0; i < bdr_attributes.Size(); i++)
      {
         bdr_attributes[i] = 21 + i;
      }

      direction.SetSize(2);
      direction = 0.0;
      direction[1] = -1.0;
   }
};

class DirectionalBoundaryLoadCoefficient : public VectorCoefficient
{
private:
   const Vector &direction;

public:
   explicit DirectionalBoundaryLoadCoefficient(const Vector &dir)
      : VectorCoefficient(dir.Size()), direction(dir) {}

   void Eval(Vector &V, ElementTransformation &T,
             const IntegrationPoint &ip) override
   {
      (void)T;
      (void)ip;
      V.SetSize(vdim);
      V = direction;
   }
};

} // namespace mfem

#endif // BOUNDARY_LOAD_SPEC_HPP
