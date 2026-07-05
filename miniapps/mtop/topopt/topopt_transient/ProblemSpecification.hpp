// =============================================================================
// Problem Specification Interface for Transient Topology Optimization
// =============================================================================

#ifndef PROBLEM_SPECIFICATION_HPP
#define PROBLEM_SPECIFICATION_HPP

#include "mfem.hpp"
#include "ObjectiveFunctional.hpp"
#include "TransientTopOptConfig.hpp"

#include <memory>
#include <ostream>
#include <string>

namespace mfem
{

struct DampingParameters
{
   real_t thickness = 0.25;
   real_t x_max = 1.5;
   real_t y_max = 0.75;
   real_t scale_length = 0.2136;
   real_t reflection = 1e-4;
   real_t beta = 2.0;
   int exponent = 2;
};

class TransientTopOptProblem
{
public:
   virtual ~TransientTopOptProblem() = default;

   virtual const TransientTopOptConfig &GetConfig() const = 0;

   virtual const std::string &GetMeshFile() const
   {
      return GetConfig().mesh_file;
   }

   virtual int GetRefinementLevel() const { return GetConfig().ref_levels; }
   virtual int GetOrder() const { return GetConfig().order; }

   virtual real_t GetFinalTime() const { return GetConfig().t_final; }
   virtual real_t GetTimeStep() const { return GetConfig().dt; }
   virtual real_t GetVolumeFraction() const { return GetConfig().vol_frac; }
   virtual real_t GetFilterRadius() const { return GetConfig().filter_radius; }
   virtual int GetMaxIterations() const { return GetConfig().max_it; }
   virtual real_t GetMoveLimit() const { return GetConfig().move; }
   virtual real_t GetChangeTolerance() const { return GetConfig().change_tol; }

   virtual const MaterialParams &GetMaterialParams() const
   {
      return GetConfig().material;
   }

   virtual const BoundaryLoadSpec &GetBoundaryLoad() const
   {
      return GetConfig().boundary_load;
   }

   virtual void GetReferenceDomainExtents(real_t &x_max,
                                          real_t &y_max) const
   {
      x_max = GetConfig().x_max;
      y_max = GetConfig().y_max;
   }

   virtual DampingParameters GetDampingParameters() const
   {
      const TransientTopOptConfig &cfg = GetConfig();
      DampingParameters damping;
      damping.thickness = cfg.damping_thickness;
      damping.x_max = cfg.x_max;
      damping.y_max = cfg.y_max;
      damping.scale_length = cfg.damping_scale_length;
      damping.reflection = cfg.damping_reflection;
      damping.beta = cfg.damping_beta;
      damping.exponent = cfg.damping_exponent;
      return damping;
   }

   virtual void GetEssentialBoundaryAttributes(Array<int> &attrs) const = 0;
   virtual void GetAbsorbingBoundaryAttributes(Array<int> &attrs) const = 0;

   virtual std::unique_ptr<VectorCoefficient>
   CreateBoundaryLoadCoefficient() const = 0;

   virtual std::unique_ptr<TimeIntegratedObjective>
   CreateObjective(ParFiniteElementSpace *state_fes, MPI_Comm comm) const = 0;

   virtual bool Validate(std::ostream &err) const
   {
      const TransientTopOptConfig &cfg = GetConfig();
      if (cfg.order < 1)
      {
         err << "Error: finite element order must be at least 1.\n";
         return false;
      }
      if (cfg.dt <= 0.0 || cfg.t_final <= 0.0)
      {
         err << "Error: time step and final time must both be positive.\n";
         return false;
      }
      if (cfg.vol_frac <= 0.0 || cfg.vol_frac > 1.0)
      {
         err << "Error: target volume fraction must be in (0, 1].\n";
         return false;
      }
      if (cfg.max_it < 1)
      {
         err << "Error: maximum MMA iterations must be at least 1.\n";
         return false;
      }
      if (cfg.damping_scale_length <= 0.0 || cfg.damping_reflection <= 0.0)
      {
         err << "Error: damping scale length and reflection must be positive.\n";
         return false;
      }
      if (cfg.boundary_load.bdr_attributes.Size() == 0)
      {
         err << "Error: boundary load has no boundary attributes.\n";
         return false;
      }
      if (cfg.boundary_load.direction.Size() == 0)
      {
         err << "Error: boundary load direction is empty.\n";
         return false;
      }
      return true;
   }
};

class WaveShieldingProblem final : public TransientTopOptProblem
{
private:
   TransientTopOptConfig cfg;

   static void CopyAttributes(const Array<int> &src, Array<int> &dst)
   {
      dst.SetSize(src.Size());
      for (int i = 0; i < src.Size(); i++)
      {
         dst[i] = src[i];
      }
   }

public:
   explicit WaveShieldingProblem(const TransientTopOptConfig &config)
      : cfg(config) {}

   const TransientTopOptConfig &GetConfig() const override { return cfg; }

   void GetEssentialBoundaryAttributes(Array<int> &attrs) const override
   {
      CopyAttributes(cfg.essential_bdr_attributes, attrs);
   }

   void GetAbsorbingBoundaryAttributes(Array<int> &attrs) const override
   {
      CopyAttributes(cfg.absorbing_bdr_attributes, attrs);
   }

   std::unique_ptr<VectorCoefficient>
   CreateBoundaryLoadCoefficient() const override
   {
      return std::make_unique<DirectionalBoundaryLoadCoefficient>(
         cfg.boundary_load.direction);
   }

   std::unique_ptr<TimeIntegratedObjective>
   CreateObjective(ParFiniteElementSpace *state_fes, MPI_Comm comm) const override
   {
      auto indicator = std::make_unique<SubdomainIndicator>(
         cfg.x_max/2.0, cfg.y_max/2.0, cfg.protected_radius);

      return std::make_unique<DisplacementL2Objective>(
         state_fes, std::move(indicator), comm);
   }
};

} // namespace mfem

#endif // PROBLEM_SPECIFICATION_HPP
