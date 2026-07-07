// =============================================================================
// Forward-Problem Specification for Transient Topology Optimization
// =============================================================================
//
// One header for the fixed forward experiment the design variable is optimized
// against. It gathers, in dependency order:
//
//   1. MaterialParams          - reference material + SIMP parametrization
//   2. BoundaryLoadSpec        - the applied boundary load (source term)
//   3. Damping*                - sponge-layer / absorbing-boundary damping
//   4. TransientTopOptConfig   - passive (POD) config bag: mesh/time/opt + above
//   5. TransientTopOptProblem  - the problem interface the driver talks to
//      WaveShieldingProblem    - the concrete wave-shielding instance
//      BandWaveguideProblem    - 2D lift of a 1D transient waveguide example
//
// The objective (what to measure, J and dJ/du) lives in ObjectiveFunctional.hpp;
// the forward + adjoint solver lives in ElastodynamicsSolver.hpp.
//
// =============================================================================

#ifndef PROBLEM_SPECIFICATION_HPP
#define PROBLEM_SPECIFICATION_HPP

#include "mfem.hpp"
#include "ObjectiveFunctional.hpp"

#include <cmath>
#include <fstream>
#include <memory>
#include <ostream>
#include <string>
#include <utility>

namespace mfem
{

// =============================================================================
// MATERIAL PARAMETERS: reference coefficients + SIMP parametrization
// =============================================================================
struct MaterialParams
{
   real_t rho0 = 1.0;
   real_t lambda0 = 2.0;
   real_t mu0 = 1.0;
   real_t r_min = 1e-6;
   real_t r_max = 1.0;
   real_t simp_p = 3.0;
};

// =============================================================================
// BOUNDARY LOAD: applied traction (source term) on the boundary
// =============================================================================
enum class LoadTimeProfile
{
   CONSTANT,
   GAUSSIAN,
   MODULATED_GAUSSIAN,
   HARMONIC
};

inline const char *LoadTimeProfileName(LoadTimeProfile profile)
{
   switch (profile)
   {
      case LoadTimeProfile::CONSTANT: return "Constant";
      case LoadTimeProfile::GAUSSIAN: return "Smooth Gaussian pulse";
      case LoadTimeProfile::MODULATED_GAUSSIAN:
         return "Modulated Gaussian pulse";
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
   // false: boundary traction on bdr_attributes; true: body force over the domain
   // (the load coefficient then supplies its own spatial support, e.g. a disc).
   bool domain_load = false;

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

// A concentrated body force: `direction` inside a disc of `radius` centered at
// (x_center, y_center), zero elsewhere (mirrors ElastTopOpt_static's bodyload).
// Owns its direction vector so it is safe to create on the fly.
class ConcentratedLoadCoefficient : public VectorCoefficient
{
private:
   real_t x_center, y_center, radius;
   Vector direction;

public:
   ConcentratedLoadCoefficient(real_t xc, real_t yc, real_t r,
                               const Vector &dir)
      : VectorCoefficient(dir.Size()), x_center(xc), y_center(yc), radius(r),
        direction(dir) {}

   void Eval(Vector &V, ElementTransformation &T,
             const IntegrationPoint &ip) override
   {
      V.SetSize(vdim);
      V = 0.0;

      Vector x(vdim);
      T.Transform(ip, x);
      const real_t dx = x(0) - x_center;
      const real_t dy = x(1) - y_center;
      if (std::sqrt(dx*dx + dy*dy) < radius) { V = direction; }
   }
};

// A rectangular body force: `direction` inside the box
// [x_min, x_max] x [y_min, y_max], zero elsewhere.
class RectangularLoadCoefficient : public VectorCoefficient
{
private:
   real_t x_min, x_max, y_min, y_max;
   Vector direction;

public:
   RectangularLoadCoefficient(real_t xmin, real_t xmax,
                              real_t ymin, real_t ymax,
                              const Vector &dir)
      : VectorCoefficient(dir.Size()), x_min(xmin), x_max(xmax),
        y_min(ymin), y_max(ymax), direction(dir) {}

   void Eval(Vector &V, ElementTransformation &T,
             const IntegrationPoint &ip) override
   {
      V.SetSize(vdim);
      V = 0.0;

      Vector x(vdim);
      T.Transform(ip, x);
      if (x(0) >= x_min && x(0) <= x_max &&
          x(1) >= y_min && x(1) <= y_max)
      {
         V = direction;
      }
   }
};

// =============================================================================
// DAMPING: sponge-layer profile, spatial coefficient, and owning field
// =============================================================================
struct DampingParameters
{
   real_t thickness = 0.25;
   real_t x_max = 1.5;
   real_t y_max = 0.75;
   real_t scale_length = 0.2136;
   real_t reflection = 1e-4;
   real_t beta = 2.0;
   int exponent = 2;
   real_t uniform = 0.0;   // uniform bulk damping alpha added on top of the sponge
   bool damp_left = true;
   bool damp_right = true;
   bool damp_bottom = true;
   bool damp_top = false;
};

class DampingProfile : public Coefficient
{
private:
   real_t thickness;
   real_t x_max, y_max;
   real_t phi_max;
   bool damp_left, damp_right, damp_bottom, damp_top;

public:
   DampingProfile(real_t thick, real_t xmax, real_t ymax,
                  bool left = true, bool right = true,
                  bool bottom = true, bool top = false)
      : thickness(thick), x_max(xmax), y_max(ymax),
        damp_left(left), damp_right(right),
        damp_bottom(bottom), damp_top(top)
   {
      phi_max = thickness * thickness / 2.0;
   }

   real_t GetPhiMax() const { return phi_max; }

   virtual real_t Eval(ElementTransformation &T, const IntegrationPoint &ip)
   {
      Vector x(2);
      T.Transform(ip, x);

      real_t phi = 0.0;
      real_t s = 0.0;

      // Left boundary layer
      if (damp_left && x(0) < thickness)
      {
         s = thickness - x(0);
         real_t phi_local = thickness * s - 0.5 * s * s;
         phi = std::max(phi, phi_local);
      }

      // Right boundary layer
      if (damp_right && x(0) > x_max - thickness)
      {
         s = x(0) - (x_max - thickness);
         real_t phi_local = thickness * s - 0.5 * s * s;
         phi = std::max(phi, phi_local);
      }

      // Bottom boundary layer
      if (damp_bottom && x(1) < thickness)
      {
         s = thickness - x(1);
         real_t phi_local = thickness * s - 0.5 * s * s;
         phi = std::max(phi, phi_local);
      }

      // Top boundary layer
      if (damp_top && x(1) > y_max - thickness)
      {
         s = x(1) - (y_max - thickness);
         real_t phi_local = thickness * s - 0.5 * s * s;
         phi = std::max(phi, phi_local);
      }

      return phi;
   }
};

class SpatialDampingCoefficient : public Coefficient
{
private:
   DampingProfile *phi_coef;
   real_t phi_max;
   real_t gamma_max;
   real_t rho;
   real_t beta;
   int m;

public:
   SpatialDampingCoefficient(DampingProfile *phi, real_t gmax,
                              real_t density, real_t b = 2.0, int mp = 2)
      : phi_coef(phi), gamma_max(gmax), rho(density), beta(b), m(mp)
   {
      phi_max = phi_coef->GetPhiMax();
   }

   virtual real_t Eval(ElementTransformation &T, const IntegrationPoint &ip)
   {
      real_t phi_val = phi_coef->Eval(T, ip);

      if (phi_max < 1e-12) return 0.0;

      real_t eta = phi_val / phi_max;
      eta = std::min(std::max(eta, 0.0), 1.0);

      real_t eta_pow = std::pow(eta, m);
      real_t F_eta = (std::exp(beta * eta_pow) - 1.0) / (std::exp(beta) - 1.0);

      return rho * gamma_max * F_eta;
   }
};

// Owns the damping field gamma(x) supplied to the operator, and derives the
// absorbing-boundary impedance rho0*c_p from the material. The field is the
// boundary sponge SpatialDampingCoefficient, optionally plus a uniform bulk term
// alpha (gamma = sponge + alpha) for dynamic relaxation toward a static solution.
// Owns every piece so the coefficient returned by GetCoefficient() stays valid.
class DampingField
{
private:
   std::unique_ptr<DampingProfile> profile;
   std::unique_ptr<SpatialDampingCoefficient> sponge;
   std::unique_ptr<ConstantCoefficient> uniform;
   std::unique_ptr<SumCoefficient> combined;
   Coefficient *effective;
   real_t p_wave_speed;
   real_t impedance_;

public:
   DampingField(const MaterialParams &mat, const DampingParameters &damping,
                bool enabled = true)
   {
      p_wave_speed = std::sqrt((mat.lambda0 + 2.0 * mat.mu0) / mat.rho0);

      if (!enabled)
      {
         // Damping off: gamma = 0 (no bulk / no sponge) AND impedance = 0, which
         // turns the absorbing (Robin, sigma.n = -z v) boundaries into free
         // (Neumann, sigma.n = 0) boundaries -> a fully conservative system.
         uniform = std::make_unique<ConstantCoefficient>(0.0);
         effective = uniform.get();
         impedance_ = 0.0;
         return;
      }

      const real_t gamma_max = (2.0 * p_wave_speed / damping.scale_length)
                               * std::log(1.0 / damping.reflection);

      profile = std::make_unique<DampingProfile>(damping.thickness,
                                                 damping.x_max, damping.y_max,
                                                 damping.damp_left,
                                                 damping.damp_right,
                                                 damping.damp_bottom,
                                                 damping.damp_top);
      sponge = std::make_unique<SpatialDampingCoefficient>(
                  profile.get(), gamma_max, mat.rho0, damping.beta,
                  damping.exponent);
      impedance_ = mat.rho0 * p_wave_speed;

      if (damping.uniform > 0.0)
      {
         uniform = std::make_unique<ConstantCoefficient>(damping.uniform);
         combined = std::make_unique<SumCoefficient>(*sponge, *uniform);
         effective = combined.get();
      }
      else
      {
         effective = sponge.get();
      }
   }

   Coefficient &GetCoefficient() { return *effective; }
   real_t GetImpedance() const { return impedance_; }
   real_t GetPWaveSpeed() const { return p_wave_speed; }
};

// =============================================================================
// PASSIVE CONFIG: mesh / time / optimization constants + the specs above
// =============================================================================
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
   real_t damping_uniform = 0.0;   // uniform bulk (mass-proportional) damping
   bool damping_left = true;
   bool damping_right = true;
   bool damping_bottom = true;
   bool damping_top = false;

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

// =============================================================================
// PROBLEM INTERFACE: the fixed forward experiment the driver optimizes against
// =============================================================================
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

   /// Build the (coarse) mesh for this problem. Default reads the mesh file;
   /// problems with generated geometry (e.g. a cantilever beam) override this.
   /// Uniform refinement is applied by the driver via GetRefinementLevel().
   virtual Mesh CreateMesh() const
   {
      std::ifstream imesh(GetMeshFile().c_str());
      MFEM_VERIFY(imesh.good(),
                  "Cannot open mesh file '" << GetMeshFile() << "'.");
      Mesh mesh(imesh, 1, 1);
      imesh.close();
      return mesh;
   }

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
      damping.uniform = cfg.damping_uniform;
      damping.damp_left = cfg.damping_left;
      damping.damp_right = cfg.damping_right;
      damping.damp_bottom = cfg.damping_bottom;
      damping.damp_top = cfg.damping_top;
      return damping;
   }

   /// Assemble this problem's damping field gamma(x) + absorbing-boundary
   /// impedance from the material and damping parameters. With enabled=false all
   /// dissipation is removed (gamma=0, impedance=0 -> Neumann boundaries). The
   /// returned object owns its coefficients, so keep it alive while in use.
   virtual std::unique_ptr<DampingField>
   CreateDampingField(bool enabled = true) const
   {
      return std::make_unique<DampingField>(GetMaterialParams(),
                                            GetDampingParameters(), enabled);
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
      if (!cfg.boundary_load.domain_load &&
          cfg.boundary_load.bdr_attributes.Size() == 0)
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

// =============================================================================
// WAVE-SHIELDING PROBLEM: minimize |u|^2 in a protected circular subdomain
// =============================================================================
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

// =============================================================================
// BAND-WAVEGUIDE PROBLEM: 2D lift of a 1D transient wave-filtering example
// =============================================================================
// A long, thin elastic band is forced by a narrow vertical body-force strip at
// its center. The force is axial, so it launches left- and right-traveling waves
// through the elastodynamic operator. Left/right sponge layers and absorbing
// boundary impedance remove outgoing waves. The objective minimizes the
// time-integrated displacement energy in symmetric rectangular receiver regions
// on both sides of the source.
class BandWaveguideProblem final : public TransientTopOptProblem
{
private:
   static constexpr real_t length_ = 8.0;
   static constexpr real_t height_ = 0.5;
   static constexpr int nx_ = 320;
   static constexpr int ny_ = 20;

   // Boundary attributes from Mesh::MakeCartesian2D:
   // bottom = 1, right = 2, top = 3, left = 4.
   static constexpr int right_attr_ = 2;
   static constexpr int left_attr_ = 4;

   static constexpr real_t source_width_ = 0.06;
   static constexpr real_t source_x_ = 0.5 * length_;

   static constexpr real_t left_receiver_x_min_ = 0.90;
   static constexpr real_t left_receiver_x_max_ = 2.10;
   static constexpr real_t right_receiver_x_min_ = 5.90;
   static constexpr real_t right_receiver_x_max_ = 7.10;

   TransientTopOptConfig cfg;
   std::string mesh_desc = "<generated 2D band waveguide>";

   static void CopyAttributes(const Array<int> &src, Array<int> &dst)
   {
      dst.SetSize(src.Size());
      for (int i = 0; i < src.Size(); i++) { dst[i] = src[i]; }
   }

   std::unique_ptr<VectorCoefficient> MakeLoadCoefficient() const
   {
      Vector dir(2);
      dir = 0.0;
      dir[0] = 1.0;   // axial polarization; propagation is both left and right

      return std::make_unique<RectangularLoadCoefficient>(
         source_x_ - 0.5 * source_width_,
         source_x_ + 0.5 * source_width_,
         0.0, height_, dir);
   }

public:
   explicit BandWaveguideProblem(const TransientTopOptConfig &base)
      : cfg(base)
   {
      cfg.x_max = length_;
      cfg.y_max = height_;

      // Keep SIMP but avoid a near-disconnecting void slit. The finite contrast
      // makes multiple impedance interfaces more useful for this band-gap-style
      // reference problem.
      cfg.material.r_min = 0.10;

      // Narrow center strip body force with a Gaussian-modulated carrier. With
      // c_p ~= 2, frequency 5 gives lambda_p ~= 0.4, so Bragg spacing
      // lambda/2 ~= 0.2 encourages more visible repeated interfaces along
      // each side of the longer waveguide.
      cfg.boundary_load.domain_load = true;
      cfg.boundary_load.time_profile = LoadTimeProfile::MODULATED_GAUSSIAN;
      cfg.boundary_load.amplitude = 30.0;
      cfg.boundary_load.duration = 0.80;
      cfg.boundary_load.frequency = 5.0;
      cfg.boundary_load.phase = 0.0;
      cfg.boundary_load.bdr_attributes.SetSize(0);
      cfg.boundary_load.direction.SetSize(2);
      cfg.boundary_load.direction = 0.0;
      cfg.boundary_load.direction[0] = 1.0;

      // No clamped boundaries; the waveguide is free except for absorbing ends.
      cfg.essential_bdr_attributes.SetSize(0);

      cfg.absorbing_bdr_attributes.SetSize(2);
      cfg.absorbing_bdr_attributes[0] = left_attr_;
      cfg.absorbing_bdr_attributes[1] = right_attr_;

      cfg.damping_thickness = 0.75;
      cfg.damping_scale_length = 0.25;
      cfg.damping_left = true;
      cfg.damping_right = true;
      cfg.damping_bottom = false;
      cfg.damping_top = false;
      cfg.damping_uniform = 0.0;
   }

   const TransientTopOptConfig &GetConfig() const override { return cfg; }

   const std::string &GetMeshFile() const override { return mesh_desc; }

   Mesh CreateMesh() const override
   {
      return Mesh::MakeCartesian2D(nx_, ny_, Element::QUADRILATERAL,
                                   /*generate_edges=*/true, length_, height_);
   }

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
      return MakeLoadCoefficient();
   }

   std::unique_ptr<TimeIntegratedObjective>
   CreateObjective(ParFiniteElementSpace *state_fes, MPI_Comm comm) const override
   {
      auto indicator = std::make_unique<DoubleRectangularIndicator>(
         left_receiver_x_min_, left_receiver_x_max_, 0.0, height_,
         right_receiver_x_min_, right_receiver_x_max_, 0.0, height_);

      return std::make_unique<DisplacementL2Objective>(
         state_fes, std::move(indicator), comm);
   }
};

// =============================================================================
// CANTILEVER COMPLIANCE PROBLEM: transient analog of ElastTopOpt_static.cpp
// =============================================================================
// A 3x1 beam clamped on the left, driven by a constant concentrated downward
// body force near the free tip (a disc at (2.85, 0.5), mirroring the static
// bodyload). Minimize the time-integrated compliance J = int_0^T int_Omega f.u,
// i.e. the *dynamic* work done by the load (pure dynamics, no damping - the beam
// keeps ringing, so this is a dynamic-stiffness objective, not the static one).
// Material matches the static example: E = 3, nu = 0.3, SIMP exponent 3. Run
// with lumped mass (exact Dirichlet projection).
class CantileverComplianceProblem final : public TransientTopOptProblem
{
private:
   static constexpr real_t length_ = 3.0;
   static constexpr real_t height_ = 1.0;
   static constexpr int nx_ = 48;
   static constexpr int ny_ = 16;
   static constexpr int clamped_attr_ = 4;   // left edge (MakeCartesian2D)
   // Concentrated tip load (disc radius sized to catch elements at this mesh).
   static constexpr real_t load_x_ = 2.85;
   static constexpr real_t load_y_ = 0.5;
   static constexpr real_t load_r_ = 0.15;

   TransientTopOptConfig cfg;
   std::string mesh_desc = "<generated cantilever beam (compliance)>";

   static void CopyAttributes(const Array<int> &src, Array<int> &dst)
   {
      dst.SetSize(src.Size());
      for (int i = 0; i < src.Size(); i++) { dst[i] = src[i]; }
   }

   std::unique_ptr<VectorCoefficient> MakeLoadCoefficient() const
   {
      Vector dir(2);
      dir = 0.0;
      dir[1] = -1.0;   // downward
      return std::make_unique<ConcentratedLoadCoefficient>(
         load_x_, load_y_, load_r_, dir);
   }

public:
   explicit CantileverComplianceProblem(const TransientTopOptConfig &base)
      : cfg(base)
   {
      cfg.x_max = length_;
      cfg.y_max = height_;

      // Material E = 3, nu = 0.3 (as in ElastTopOpt_static), rho0 = 1.
      const real_t E = 3.0, nu = 0.3;
      cfg.material.mu0 = E / (2.0 * (1.0 + nu));
      cfg.material.lambda0 = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
      cfg.material.rho0 = 1.0;

      // Raise the SIMP floor: with uniform (design-independent) damping C = alpha*M1,
      // the explicit-RK4 stability limit is M^-1 C ~ alpha/(SIMP*rho0) < ~2.78/dt, so
      // the mass factor must stay above ~ dt*alpha/2.78. r_min = 1e-2 keeps SIMP well
      // above that (M^-1 C <= alpha/r_min = 200) so a low-density pocket cannot
      // destabilize the damped forward sweep.
      cfg.material.r_min = 1e-2;

      // Damping is part of this problem's physics: no sponge / no absorbing
      // boundaries, but a uniform bulk (mass-proportional) term alpha so a
      // step-loaded beam relaxes toward the static equilibrium (dynamic
      // relaxation). Roughly near-critical for the fundamental mode; tune here.
      // The driver's -no-damp toggle zeroes it for the pure-dynamics variant.
      cfg.damping_thickness = 0.0;
      cfg.absorbing_bdr_attributes.SetSize(0);
      cfg.damping_uniform = 2.0;

      // Clamped left edge.
      cfg.essential_bdr_attributes.SetSize(1);
      cfg.essential_bdr_attributes[0] = clamped_attr_;

      // Constant concentrated downward body force near the tip. amplitude = 1
      // and a CONSTANT profile make the applied force equal the objective's raw
      // load, so J is exactly the dynamic compliance int f.u.
      cfg.boundary_load.domain_load = true;
      cfg.boundary_load.time_profile = LoadTimeProfile::CONSTANT;
      cfg.boundary_load.amplitude = 1.0;
      cfg.boundary_load.bdr_attributes.SetSize(0);
      cfg.boundary_load.direction.SetSize(2);
      cfg.boundary_load.direction = 0.0;
      cfg.boundary_load.direction[1] = -1.0;
   }

   const TransientTopOptConfig &GetConfig() const override { return cfg; }

   const std::string &GetMeshFile() const override { return mesh_desc; }

   Mesh CreateMesh() const override
   {
      return Mesh::MakeCartesian2D(nx_, ny_, Element::QUADRILATERAL,
                                   /*generate_edges=*/true, length_, height_);
   }

   // A body force carries no boundary attributes; skip the empty-load-attr check.
   bool Validate(std::ostream &err) const override
   {
      const TransientTopOptConfig &c = GetConfig();
      if (c.order < 1) { err << "Error: order must be >= 1.\n"; return false; }
      if (c.dt <= 0.0 || c.t_final <= 0.0)
      {
         err << "Error: dt and t_final must be positive.\n"; return false;
      }
      if (c.vol_frac <= 0.0 || c.vol_frac > 1.0)
      {
         err << "Error: vol_frac must be in (0, 1].\n"; return false;
      }
      if (c.max_it < 1) { err << "Error: max_it must be >= 1.\n"; return false; }
      return true;
   }

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
      return MakeLoadCoefficient();
   }

   std::unique_ptr<TimeIntegratedObjective>
   CreateObjective(ParFiniteElementSpace *state_fes, MPI_Comm comm) const override
   {
      // Compliance J = int_0^T int_Omega f.u: the objective owns its own copy of
      // the same body force that drives the forward problem.
      return std::make_unique<ComplianceObjective>(
         state_fes, MakeLoadCoefficient(), comm);
   }
};

} // namespace mfem

#endif // PROBLEM_SPECIFICATION_HPP
