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
//      BandModeConverterProblem- 2D optimizable spectral-separation pilot
//      ElasticInclusionIdentificationProblem
//                              - boundary-data inverse problem
//
// The objective (what to measure, J and dJ/du) lives in ObjectiveFunctional.hpp;
// the forward + adjoint solver lives in ElastodynamicsSolver.hpp.
//
// =============================================================================

#ifndef PROBLEM_SPECIFICATION_HPP
#define PROBLEM_SPECIFICATION_HPP

#include "mfem.hpp"
#include "ObjectiveFunctional.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iterator>
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

inline real_t EvaluateLoadTimeFactor(LoadTimeProfile profile,
                                     real_t time,
                                     real_t duration,
                                     real_t frequency,
                                     real_t phase)
{
   constexpr real_t two_pi =
      2.0 * 3.1415926535897932384626433832795;

   if (profile == LoadTimeProfile::CONSTANT) { return 1.0; }
   if (profile == LoadTimeProfile::HARMONIC)
   {
      // frequency is measured in cycles per unit time, consistently with
      // wavelength = c/f and the modulated carrier below.
      return std::sin(two_pi * frequency * time + phase);
   }

   MFEM_VERIFY(duration > 0.0,
               "Gaussian load duration must be strictly positive.");
   const real_t t_center = duration / 2.0;
   const real_t sigma = duration / 4.0;
   const real_t t_diff = time - t_center;
   const real_t envelope =
      std::exp(-t_diff * t_diff / (2.0 * sigma * sigma));

   if (profile == LoadTimeProfile::GAUSSIAN) { return envelope; }

   MFEM_VERIFY(profile == LoadTimeProfile::MODULATED_GAUSSIAN,
               "Unknown load time profile.");
   return envelope * std::cos(two_pi * frequency * t_diff + phase);
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

// Monopole source coefficient for 3D spherical wave generation.
// Radial body force f = amplitude * r_hat within source sphere,
// where r_hat = x / |x| is the outward unit radial vector.
class MonopoleSourceCoefficient : public VectorCoefficient
{
private:
   real_t radius;
   real_t amplitude;
   Vector center;

public:
   MonopoleSourceCoefficient(real_t r, real_t amp = 1.0,
                             const Vector *ctr = nullptr)
      : VectorCoefficient(3), radius(r), amplitude(amp)
   {
      center.SetSize(3);
      if (ctr && ctr->Size() == 3) { center = *ctr; }
      else { center = 0.0; }
   }

   void Eval(Vector &V, ElementTransformation &T,
             const IntegrationPoint &ip) override
   {
      V.SetSize(vdim);
      V = 0.0;

      Vector x(vdim);
      T.Transform(ip, x);
      x -= center;

      real_t r = x.Norml2();

      // Inside source sphere and not at singularity
      if (r < radius && r > 1e-12)
      {
         // Radial direction: f = amp * x / |x|
         real_t scale = amplitude / r;
         V = x;
         V *= scale;
      }
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

// Smooth scalar collar window: sin^2 in x and zero outside [x_min,x_max].
// It is useful as an objective-region weight because, unlike a hard indicator,
// it does not inject artificial high-frequency content at the collar edges.
class AxialSinSquaredWindow2D : public Coefficient
{
private:
   real_t x_min, x_max;

public:
   AxialSinSquaredWindow2D(real_t xmin, real_t xmax)
      : x_min(xmin), x_max(xmax)
   {
      MFEM_VERIFY(x_max > x_min, "Invalid axial window support.");
   }

   real_t Eval(ElementTransformation &T,
               const IntegrationPoint &ip) override
   {
      Vector x(2);
      T.Transform(ip, x);
      if (x(0) < x_min || x(0) > x_max) { return 0.0; }

      constexpr real_t pi =
         3.1415926535897932384626433832795;
      const real_t xi = (x(0) - x_min) / (x_max - x_min);
      return std::pow(std::sin(pi * xi), 2);
   }
};

// Smooth axial vector mode in a rectangular 2D waveguide collar.  The
// longitudinal sin^2 envelope avoids injecting an artificial element-scale jump.
// mode_y = 0 is the smooth fundamental cross-sectional mode; larger indices
// create controlled transverse spectral content through cos(mode_y*pi*y/H).
class WaveguideModeCoefficient2D : public VectorCoefficient
{
private:
   real_t x_min, x_max, height;
   real_t normalization;
   int mode_y;
   Vector polarization;

public:
   WaveguideModeCoefficient2D(real_t xmin, real_t xmax, real_t height_,
                              int mode_y_, const Vector &polarization_)
      : VectorCoefficient(2), x_min(xmin), x_max(xmax), height(height_),
        normalization(1.0), mode_y(mode_y_), polarization(polarization_)
   {
      MFEM_VERIFY(x_max > x_min && height > 0.0,
                  "Invalid 2D waveguide-mode support.");
      MFEM_VERIFY(polarization.Size() == 2,
                  "2D waveguide mode requires a two-component polarization.");
      MFEM_VERIFY(mode_y >= 0,
                  "Waveguide mode index must be nonnegative.");
      const real_t polarization_norm_sq = polarization * polarization;
      MFEM_VERIFY(polarization_norm_sq > 0.0 &&
                  std::isfinite(polarization_norm_sq),
                  "2D waveguide-mode polarization must be finite and nonzero.");

      // Integral sin^4(pi*xi) dx = 3*(xmax-xmin)/8.  The transverse
      // integral is H for mode zero and H/2 for every positive integer mode.
      // Unit L2 normalization makes source/target amplitudes independent of
      // collar width and modal index.
      const real_t axial_norm_sq = 3.0 * (x_max - x_min) / 8.0;
      const real_t transverse_norm_sq =
         (mode_y == 0) ? height : 0.5 * height;
      normalization = 1.0 / std::sqrt(
         axial_norm_sq * transverse_norm_sq * polarization_norm_sq);
   }

   void Eval(Vector &value, ElementTransformation &T,
             const IntegrationPoint &ip) override
   {
      value.SetSize(2);
      value = 0.0;

      Vector x(2);
      T.Transform(ip, x);
      if (x(0) < x_min || x(0) > x_max ||
          x(1) < 0.0 || x(1) > height)
      {
         return;
      }

      constexpr real_t pi =
         3.1415926535897932384626433832795;
      const real_t xi = (x(0) - x_min) / (x_max - x_min);
      const real_t axial_envelope =
         std::pow(std::sin(pi * xi), 2);
      const real_t transverse_mode =
         std::cos(mode_y * pi * x(1) / height);

      value = polarization;
      value *= normalization * axial_envelope * transverse_mode;
   }
};

// Smooth axial body-force mode in a rectangular 3D waveguide collar.  The
// longitudinal sin^2 envelope avoids injecting an artificial element-scale jump.
// mode_y = mode_z = 0 is the smooth fundamental cross-sectional mode; larger
// indices create controlled transverse spectral content.
class WaveguideModeCoefficient3D : public VectorCoefficient
{
private:
   real_t x_min, x_max, width, height;
   int mode_y, mode_z;
   Vector polarization;

public:
   WaveguideModeCoefficient3D(real_t xmin, real_t xmax,
                              real_t width_, real_t height_,
                              int mode_y_, int mode_z_,
                              const Vector &polarization_)
      : VectorCoefficient(3), x_min(xmin), x_max(xmax),
        width(width_), height(height_),
        mode_y(mode_y_), mode_z(mode_z_),
        polarization(polarization_)
   {
      MFEM_VERIFY(x_max > x_min && width > 0.0 && height > 0.0,
                  "Invalid 3D waveguide-mode support.");
      MFEM_VERIFY(polarization.Size() == 3,
                  "3D waveguide mode requires a three-component polarization.");
      MFEM_VERIFY(mode_y >= 0 && mode_z >= 0,
                  "Waveguide mode indices must be nonnegative.");
   }

   void Eval(Vector &value, ElementTransformation &T,
             const IntegrationPoint &ip) override
   {
      value.SetSize(3);
      value = 0.0;

      Vector x(3);
      T.Transform(ip, x);
      if (x(0) < x_min || x(0) > x_max ||
          x(1) < 0.0 || x(1) > width ||
          x(2) < 0.0 || x(2) > height)
      {
         return;
      }

      constexpr real_t pi =
         3.1415926535897932384626433832795;
      const real_t xi = (x(0) - x_min) / (x_max - x_min);
      const real_t axial_envelope =
         std::pow(std::sin(pi * xi), 2);
      const real_t transverse_mode =
         std::cos(mode_y * pi * x(1) / width) *
         std::cos(mode_z * pi * x(2) / height);

      value = polarization;
      value *= axial_envelope * transverse_mode;
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

// Base interface for damping profile coefficients
// Provides phi(x) spatial function and phi_max for normalization
class DampingProfileBase : public Coefficient
{
public:
   virtual ~DampingProfileBase() = default;
   virtual real_t GetPhiMax() const = 0;
};

class DampingProfile : public DampingProfileBase
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

   real_t GetPhiMax() const override { return phi_max; }

   real_t Eval(ElementTransformation &T, const IntegrationPoint &ip) override
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

// Spherical damping profile for 3D radial geometry.
//
// The normalized coordinate is the exact harmonic coordinate in a spherical
// shell: eta(r) = (1/r_inner - 1/r) / (1/r_inner - 1/r_outer).  This replaces
// the Cartesian parabolic profile previously evaluated at the radius, which
// was not a solution of the radial Laplace equation.
class SphericalDampingProfile : public DampingProfileBase
{
private:
   real_t r_inner;     // Inner radius where damping starts
   real_t r_outer;     // Outer radius (maximum damping)
   real_t phi_max;

public:
   SphericalDampingProfile(real_t r_in, real_t r_out)
      : r_inner(r_in), r_outer(r_out)
   {
      // The profile is used only through phi / phi_max. Retaining a positive
      // scale keeps the DampingProfileBase interface shared with the 2D path.
      phi_max = 1.0;
   }

   real_t GetPhiMax() const override { return phi_max; }

   real_t NormalizedCoordinate(real_t r) const
   {
      if (r <= r_inner) { return 0.0; }
      if (r >= r_outer) { return 1.0; }

      const real_t denominator = 1.0 / r_inner - 1.0 / r_outer;
      return (1.0 / r_inner - 1.0 / r) / denominator;
   }

   real_t Eval(ElementTransformation &T, const IntegrationPoint &ip) override
   {
      Vector x(3);
      T.Transform(ip, x);
      real_t r = x.Norml2();

      return phi_max * NormalizedCoordinate(r);
   }

   real_t GetInnerRadius() const { return r_inner; }
   real_t GetOuterRadius() const { return r_outer; }
};

class SpatialDampingCoefficient : public Coefficient
{
private:
   DampingProfileBase *phi_coef;
   real_t phi_max;
   real_t gamma_max;
   real_t rho;
   real_t beta;
   int m;

public:
   SpatialDampingCoefficient(DampingProfileBase *phi, real_t gmax,
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
      return rho * gamma_max * Ramp(eta, beta, m);
   }

   static real_t Ramp(real_t eta, real_t beta, int m)
   {
      eta = std::min(std::max(eta, 0.0), 1.0);
      const real_t eta_pow = std::pow(eta, m);
      return (std::exp(beta * eta_pow) - 1.0) / (std::exp(beta) - 1.0);
   }
};

// Integrate the damping ramp along the physical propagation coordinate.  The
// caller supplies the normalized profile coordinate eta(s) actually used by
// SpatialDampingCoefficient, so the attenuation normalization cannot silently
// drift away from the implemented spatial profile.
template <typename NormalizedCoordinate>
inline real_t IntegrateNormalizedDampingRamp(
   real_t coordinate_min, real_t coordinate_max,
   NormalizedCoordinate normalized_coordinate,
   real_t beta, int exponent, int num_points = 2048)
{
   MFEM_VERIFY(coordinate_max > coordinate_min,
               "Damping ramp integration requires a positive interval.");
   MFEM_VERIFY(num_points > 0,
               "Damping ramp integration requires positive quadrature size.");

   const real_t h = (coordinate_max - coordinate_min) / num_points;
   real_t integral = 0.0;
   for (int q = 0; q < num_points; q++)
   {
      const real_t coordinate = coordinate_min + (q + 0.5) * h;
      integral += SpatialDampingCoefficient::Ramp(
         normalized_coordinate(coordinate), beta, exponent);
   }
   return h * integral;
}

// DampingProfile uses phi/phi_max = 2 q - q^2, where q is the distance
// through the layer divided by its thickness.  Integrate that exact profile,
// rather than using a hand-tuned length unrelated to the implemented ramp.
inline real_t CartesianDampingRampIntegral(real_t thickness,
                                           real_t beta, int exponent)
{
   MFEM_VERIFY(thickness > 0.0,
               "Cartesian damping requires positive layer thickness.");
   return IntegrateNormalizedDampingRamp(
      0.0, thickness,
      [thickness](real_t layer_depth)
      {
         const real_t q = layer_depth / thickness;
         return 2.0 * q - q * q;
      },
      beta, exponent);
}

// Base interface for damping fields
// Provides gamma(x) coefficient and absorbing-boundary impedance
class DampingFieldBase
{
public:
   virtual ~DampingFieldBase() = default;
   virtual Coefficient &GetCoefficient() = 0;
   virtual real_t GetImpedance() const = 0;
   virtual real_t GetPWaveSpeed() const = 0;
   virtual void PrintSummary(std::ostream &) const {}
};

// Owns the damping field gamma(x) supplied to the operator. The sponge density
// and absorbing-boundary impedance use the same passive SIMP material scale as
// the mass operator at the outer boundary. The field is the boundary sponge
// SpatialDampingCoefficient, optionally plus a uniform bulk term alpha
// (gamma = sponge + alpha) for dynamic relaxation toward a static solution.
// Owns every piece so the coefficient returned by GetCoefficient() stays valid.
class DampingField : public DampingFieldBase
{
private:
   std::unique_ptr<DampingProfile> profile;
   std::unique_ptr<SpatialDampingCoefficient> sponge;
   std::unique_ptr<ConstantCoefficient> uniform;
   std::unique_ptr<SumCoefficient> combined;
   Coefficient *effective;
   real_t p_wave_speed;
   real_t impedance_;
   real_t passive_mass_density_;
   real_t shape_integral_;
   real_t gamma_rate_max_;
   real_t uniform_gamma_;
   bool enabled_;

public:
   DampingField(const MaterialParams &mat, const DampingParameters &damping,
                bool enabled = true, real_t passive_material_scale = 1.0)
      : effective(nullptr), p_wave_speed(0.0), impedance_(0.0),
        passive_mass_density_(mat.rho0 * passive_material_scale),
        shape_integral_(damping.scale_length), gamma_rate_max_(0.0),
        uniform_gamma_(damping.uniform), enabled_(enabled)
   {
      p_wave_speed = std::sqrt((mat.lambda0 + 2.0 * mat.mu0) / mat.rho0);
      MFEM_VERIFY(passive_mass_density_ > 0.0,
                  "Damping requires positive passive mass density.");

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

      MFEM_VERIFY(shape_integral_ > 0.0,
                  "Damping requires a positive ramp integral.");
      MFEM_VERIFY(damping.reflection > 0.0 && damping.reflection < 1.0,
                  "Damping reflection target must be in (0, 1).");
      gamma_rate_max_ = (2.0 * p_wave_speed / shape_integral_)
                        * std::log(1.0 / damping.reflection);

      profile = std::make_unique<DampingProfile>(damping.thickness,
                                                 damping.x_max, damping.y_max,
                                                 damping.damp_left,
                                                 damping.damp_right,
                                                 damping.damp_bottom,
                                                 damping.damp_top);
      sponge = std::make_unique<SpatialDampingCoefficient>(
                  profile.get(), gamma_rate_max_, passive_mass_density_,
                  damping.beta,
                  damping.exponent);
      impedance_ = passive_mass_density_ * p_wave_speed;

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

   Coefficient &GetCoefficient() override { return *effective; }
   real_t GetImpedance() const override { return impedance_; }
   real_t GetPWaveSpeed() const override { return p_wave_speed; }
   void PrintSummary(std::ostream &out) const override
   {
      if (!enabled_)
      {
         out << "Cartesian damping: disabled; ABC Zp = 0\n";
         return;
      }

      out << "Cartesian damping: normalized parabolic coordinate, target amplitude "
          << "reflection = " << std::scientific << std::setprecision(3)
          << std::exp(-gamma_rate_max_ * shape_integral_
                      / (2.0 * p_wave_speed))
          << ", passive mass density = " << passive_mass_density_
          << ", shape integral = " << shape_integral_
          << ", gamma_rate_max = " << gamma_rate_max_
          << ", gamma_max = " << passive_mass_density_ * gamma_rate_max_
          << ", uniform gamma = " << uniform_gamma_
          << ", ABC Zp = " << impedance_ << std::defaultfloat << "\n";
   }
};

// Spherical damping field - radial sponge layer for 3D spherical geometry
// Same ownership pattern as DampingField, but uses SphericalDampingProfile
class SphericalDampingField : public DampingFieldBase
{
private:
   std::unique_ptr<SphericalDampingProfile> profile;
   std::unique_ptr<SpatialDampingCoefficient> sponge;
   Coefficient *effective;
   real_t p_wave_speed;
   real_t impedance_;
   real_t passive_mass_density_;
   real_t shape_integral_;
   real_t gamma_rate_max_;

public:
   SphericalDampingField(const MaterialParams &mat, real_t r_inner, real_t r_outer,
                         real_t passive_material_scale, real_t reflection,
                         real_t beta = 2.0, int exponent = 2)
   {
      p_wave_speed = std::sqrt((mat.lambda0 + 2.0 * mat.mu0) / mat.rho0);
      profile = std::make_unique<SphericalDampingProfile>(r_inner, r_outer);
      passive_mass_density_ = mat.rho0 * passive_material_scale;
      MFEM_VERIFY(passive_mass_density_ > 0.0,
                  "Spherical damping requires positive passive mass density.");
      MFEM_VERIFY(reflection > 0.0 && reflection < 1.0,
                  "Spherical damping reflection target must be in (0, 1).");

      // For a locally outgoing P wave, amplitude decays as
      // exp[-integral(gamma/(2 rho c_p)) dr]. Choosing this rate makes the
      // requested reflection target exact for the implemented ramp.
      shape_integral_ = IntegrateNormalizedDampingRamp(
         profile->GetInnerRadius(), profile->GetOuterRadius(),
         [this](real_t radius)
         {
            return profile->NormalizedCoordinate(radius);
         },
         beta, exponent);
      MFEM_VERIFY(shape_integral_ > 0.0,
                  "Spherical damping profile has zero normalization integral.");
      gamma_rate_max_ = 2.0 * p_wave_speed * std::log(1.0 / reflection)
                        / shape_integral_;

      // The sponge and scalar P-wave ABC must use the same passive material
      // that appears in M. The old rho0 values over-damped this rho=0.5 SIMP
      // shell and mismatched the outer-boundary impedance.
      impedance_ = passive_mass_density_ * p_wave_speed;
      sponge = std::make_unique<SpatialDampingCoefficient>(
                  profile.get(), gamma_rate_max_, passive_mass_density_, beta, exponent);
      effective = sponge.get();
   }

   Coefficient &GetCoefficient() override { return *effective; }
   real_t GetImpedance() const override { return impedance_; }
   real_t GetPWaveSpeed() const override { return p_wave_speed; }
   void PrintSummary(std::ostream &out) const override
   {
      out << "Spherical damping: harmonic radial coordinate, target amplitude "
          << "reflection = " << std::scientific << std::setprecision(3)
          << std::exp(-gamma_rate_max_ * shape_integral_ / (2.0 * p_wave_speed))
          << ", passive mass density = " << passive_mass_density_
          << ", shape integral = " << shape_integral_
          << ", gamma_rate_max = " << gamma_rate_max_
          << ", gamma_max = " << passive_mass_density_ * gamma_rate_max_
          << ", ABC Zp = " << impedance_ << std::defaultfloat << "\n";
   }
};

// =============================================================================
// PASSIVE CONFIG: mesh / time / optimization constants + the specs above
// =============================================================================
struct TransientTopOptConfig
{
   std::string mesh_file = "lamb-problem-damping-mesh-triangs.msh";
   // Set by the driver when -mesh was given explicitly; problems with their own
   // default mesh (e.g. spherical-bandgap) only override mesh_file when false.
   bool mesh_file_is_user = false;
   // Same pattern for -freq / -dur: when true, the user's carrier frequency /
   // pulse duration win over the problem's defaults. Lets the same problem run
   // cheap (low f, coarse mesh) locally and rich (high f, fine mesh) on HPC.
   bool load_frequency_is_user = false;
   bool load_duration_is_user = false;
   // Material-law overrides follow the same convention: individual problems
   // provide physically useful defaults unless the command line supplied an
   // endpoint/exponent explicitly.
   bool simp_r_min_is_user = false;
   bool simp_r_max_is_user = false;
   bool simp_p_is_user = false;
   // Discretization/experiment defaults can also be problem-specific.  The
   // driver sets these flags when the corresponding command-line option was
   // supplied so a problem does not overwrite an explicit user choice.
   bool order_is_user = false;
   bool t_final_is_user = false;
   bool time_step_is_user = false;
   bool filter_radius_is_user = false;
   bool volume_fraction_is_user = false;
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

   // Spectral-band experiment controls.  They are ignored by problems that do
   // not define a transverse target mode.
   int mode_converter_target_mode = 8;
   real_t mode_converter_target_amplitude = 1.0;
   real_t mode_converter_energy_residual_weight = 0.25;
   real_t mode_converter_energy_window_start = 3.0;
   real_t mode_converter_energy_window_ramp = 0.5;

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

   /// Whether a generated 2D waveguide identifies y=0 with y=y_max.  Keeping
   /// this in the problem contract avoids hard-coding problem names in the
   /// driver and lets related spectral-role variants share the same geometry.
   virtual bool UsesPeriodicYBoundary() const { return false; }

   /// Optional active strip and transverse mode for a deterministic modal
   /// initial density. The driver uses this only for -init modal-seed.
   virtual bool GetModalSeedRegion(real_t &, real_t &, int &) const
   {
      return false;
   }

   /// Optional problem-specific configuration written to stdout/history.
   virtual void PrintSummary(std::ostream &) const {}

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
   virtual std::unique_ptr<DampingFieldBase>
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

   /// Optional inverse-problem truth.  A non-null coefficient is projected to
   /// the ordinary raw-control space, then passed through the same filter/SIMP
   /// pipeline as every reconstruction.  The default forward problems have no
   /// prescribed truth field.
   virtual bool HasReferenceTruth() const { return false; }

   virtual std::unique_ptr<Coefficient>
   CreateTruthDensityCoefficient() const
   {
      return nullptr;
   }

   /// Boundary attributes on which inverse-problem observations live.  The
   /// returned list contains attribute numbers, not a boundary marker.
   virtual void GetObservationBoundaryAttributes(Array<int> &attrs) const
   {
      attrs.SetSize(0);
   }

   /// Complete the mesh-dependent truth-volume calculation.  Only problems
   /// with a prescribed truth accept this lifecycle callback.
   virtual void SetComputedTruthVolumeFraction(real_t)
   {
      MFEM_ABORT("The selected problem has no computed truth volume.");
   }

   /// Whether CreateObjective() requires a finalized reference trace history.
   virtual bool RequiresReferenceBoundaryData() const { return false; }

   /// Attach immutable reference observations after their high-fidelity solve.
   virtual void SetBoundaryTraceHistory(
      std::shared_ptr<const BoundaryTraceHistory>)
   {
      MFEM_ABORT("The selected problem does not accept boundary trace data.");
   }

   /// Optional spatial mode used only to report a forward temporal-convergence
   /// observable. It must not change the optimization objective or adjoint RHS.
   virtual std::unique_ptr<VectorCoefficient>
   CreateForwardModalProbe() const
   {
      return nullptr;
   }

   /// Optional output mode paired with CreateForwardModalProbe().  The first
   /// probe follows the launched mode into the receiver collar; this second
   /// probe measures the desired converted mode there.  Both are diagnostics
   /// only and must not alter the objective or adjoint source.
   virtual std::unique_ptr<VectorCoefficient>
   CreateTargetModalProbe() const
   {
      return nullptr;
   }

   /// Create a coefficient identifying passive (non-designable) regions.
   /// Returns 1.0 in passive regions (fixed material), 0.0 in active design regions.
   /// Default: nullptr (entire domain is designable).
   virtual std::unique_ptr<Coefficient> CreatePassiveRegionCoefficient() const
   {
      return nullptr;
   }

   /// Density at which passive regions are frozen. Default keeps the
   /// historical behavior (the target volume fraction). Problems whose
   /// reference medium must stay fixed while -vf is swept should return a
   /// constant instead.
   virtual real_t GetPassiveDensity() const { return GetVolumeFraction(); }

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
      if (cfg.damping_scale_length <= 0.0 || cfg.damping_reflection <= 0.0 ||
          cfg.damping_reflection >= 1.0)
      {
         err << "Error: damping scale length must be positive and reflection "
             << "must lie in (0,1).\n";
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
// ELASTIC-INCLUSION TRUTH: background solid with three interior inclusions
// =============================================================================
class ElasticInclusionTruthCoefficient final : public Coefficient
{
private:
   static real_t OrientedAreaTwice(real_t ax, real_t ay,
                                   real_t bx, real_t by,
                                   real_t px, real_t py)
   {
      return (bx - ax) * (py - ay) - (by - ay) * (px - ax);
   }

   static bool InsideTriangle(real_t x, real_t y)
   {
      const real_t side_1 = OrientedAreaTwice(
         TriangleX1(), TriangleY1(), TriangleX2(), TriangleY2(), x, y);
      const real_t side_2 = OrientedAreaTwice(
         TriangleX2(), TriangleY2(), TriangleX3(), TriangleY3(), x, y);
      const real_t side_3 = OrientedAreaTwice(
         TriangleX3(), TriangleY3(), TriangleX1(), TriangleY1(), x, y);
      const bool has_negative = side_1 < 0.0 || side_2 < 0.0 || side_3 < 0.0;
      const bool has_positive = side_1 > 0.0 || side_2 > 0.0 || side_3 > 0.0;
      return !(has_negative && has_positive);
   }

public:
   static constexpr real_t BackgroundDensity() { return 1.0; }

   static constexpr real_t SquareXMin() { return 0.350; }
   static constexpr real_t SquareXMax() { return 0.525; }
   static constexpr real_t SquareYMin() { return 0.400; }
   static constexpr real_t SquareYMax() { return 0.575; }
   static constexpr real_t SquareDensity() { return 0.15; }

   static constexpr real_t TriangleX1() { return 0.650; }
   static constexpr real_t TriangleY1() { return 0.350; }
   static constexpr real_t TriangleX2() { return 0.850; }
   static constexpr real_t TriangleY2() { return 0.350; }
   static constexpr real_t TriangleX3() { return 0.750; }
   static constexpr real_t TriangleY3() { return 0.525; }
   static constexpr real_t TriangleDensity() { return 0.45; }

   static constexpr real_t DiskCenterX() { return 1.050; }
   static constexpr real_t DiskCenterY() { return 0.500; }
   static constexpr real_t DiskRadius() { return 0.080; }
   static constexpr real_t DiskDensity() { return 0.75; }

   static constexpr real_t SquareArea()
   {
      return (SquareXMax() - SquareXMin()) *
             (SquareYMax() - SquareYMin());
   }

   static constexpr real_t TriangleArea()
   {
      return 0.5 * (TriangleX2() - TriangleX1()) *
             (TriangleY3() - TriangleY1());
   }

   static constexpr real_t DiskArea()
   {
      return 3.1415926535897932384626433832795 *
             DiskRadius() * DiskRadius();
   }

   /// Integral of 1-rho_dagger over the three non-overlapping inclusions.
   static constexpr real_t DensityDeficitArea()
   {
      return (BackgroundDensity() - SquareDensity()) * SquareArea() +
             (BackgroundDensity() - TriangleDensity()) * TriangleArea() +
             (BackgroundDensity() - DiskDensity()) * DiskArea();
   }

   real_t Eval(ElementTransformation &T,
               const IntegrationPoint &ip) override
   {
      Vector x(2);
      T.Transform(ip, x);

      if (x[0] >= SquareXMin() && x[0] <= SquareXMax() &&
          x[1] >= SquareYMin() && x[1] <= SquareYMax())
      {
         return SquareDensity();
      }
      if (InsideTriangle(x[0], x[1]))
      {
         return TriangleDensity();
      }

      const real_t dx = x[0] - DiskCenterX();
      const real_t dy = x[1] - DiskCenterY();
      if (dx * dx + dy * dy <= DiskRadius() * DiskRadius())
      {
         return DiskDensity();
      }
      return BackgroundDensity();
   }
};

// =============================================================================
// ELASTIC INCLUSION IDENTIFICATION: recover the truth from boundary traces
// =============================================================================
class ElasticInclusionIdentificationProblem final
   : public TransientTopOptProblem
{
private:
   static constexpr real_t length_ = 1.5;
   static constexpr real_t height_ = 0.75;
   static constexpr int nx_ = 60;
   static constexpr int ny_ = 30;

   // Mesh::MakeCartesian2D starts with bottom/right/top/left = 1/2/3/4.
   // CreateMesh splits one contiguous part of the top edge into attribute 5.
   static constexpr int bottom_attr_ = 1;
   static constexpr int right_attr_ = 2;
   static constexpr int top_attr_ = 3;
   static constexpr int left_attr_ = 4;
   static constexpr int source_attr_ = 5;

   static constexpr real_t source_x_min_ = 0.65;
   static constexpr real_t source_x_max_ = 0.85;
   static constexpr real_t sponge_thickness_ = 0.25;
   static constexpr real_t top_passive_thickness_ = 0.10;
   static constexpr real_t active_x_min_ = sponge_thickness_;
   static constexpr real_t active_x_max_ = length_ - sponge_thickness_;
   static constexpr real_t active_y_min_ = sponge_thickness_;
   static constexpr real_t active_y_max_ = height_ - top_passive_thickness_;

   TransientTopOptConfig cfg_;
   std::string mesh_description_ =
      "<generated 2D elastic-inclusion-identification rectangle>";
   real_t truth_volume_fraction_;
   bool truth_volume_is_discrete_ = false;
   std::shared_ptr<const BoundaryTraceHistory> trace_history_;

   static void CopyAttributes(const Array<int> &source, Array<int> &target)
   {
      target.SetSize(source.Size());
      for (int i = 0; i < source.Size(); i++) { target[i] = source[i]; }
   }

   static constexpr real_t ActiveArea()
   {
      return (active_x_max_ - active_x_min_) *
             (active_y_max_ - active_y_min_);
   }

   static constexpr real_t AnalyticTruthVolumeFraction()
   {
      return 1.0 - ElasticInclusionTruthCoefficient::DensityDeficitArea() /
                   ActiveArea();
   }

public:
   explicit ElasticInclusionIdentificationProblem(
      const TransientTopOptConfig &base)
      : cfg_(base), truth_volume_fraction_(AnalyticTruthVolumeFraction())
   {
      cfg_.x_max = length_;
      cfg_.y_max = height_;

      // These are production-oriented defaults, while every explicit command
      // line choice remains authoritative through its *_is_user flag.
      if (!cfg_.order_is_user) { cfg_.order = 2; }
      if (!cfg_.t_final_is_user) { cfg_.t_final = 1.5; }
      if (!cfg_.time_step_is_user) { cfg_.dt = 1.0e-3; }
      if (!cfg_.filter_radius_is_user) { cfg_.filter_radius = 0.05; }

      // Keep a valid analytic value until the driver projects rho_dagger to its
      // actual control space and supplies the exact discrete weighted value.
      cfg_.vol_frac = truth_volume_fraction_;

      cfg_.boundary_load.domain_load = false;
      cfg_.boundary_load.time_profile = LoadTimeProfile::MODULATED_GAUSSIAN;
      cfg_.boundary_load.amplitude = 1.0;
      if (!cfg_.load_frequency_is_user)
      {
         cfg_.boundary_load.frequency = 6.0;
      }
      if (!cfg_.load_duration_is_user)
      {
         cfg_.boundary_load.duration = 0.5;
      }
      cfg_.boundary_load.phase = 0.0;
      cfg_.boundary_load.bdr_attributes.SetSize(1);
      cfg_.boundary_load.bdr_attributes[0] = source_attr_;
      cfg_.boundary_load.direction.SetSize(2);
      const real_t inverse_sqrt_two = 1.0 / std::sqrt(2.0);
      cfg_.boundary_load.direction[0] = inverse_sqrt_two;
      cfg_.boundary_load.direction[1] = -inverse_sqrt_two;

      // The body is free except for damping.  Robin conditions terminate the
      // three sponge-backed sides; the source/free-surface top is not absorbed.
      cfg_.essential_bdr_attributes.SetSize(0);
      cfg_.absorbing_bdr_attributes.SetSize(3);
      cfg_.absorbing_bdr_attributes[0] = bottom_attr_;
      cfg_.absorbing_bdr_attributes[1] = right_attr_;
      cfg_.absorbing_bdr_attributes[2] = left_attr_;

      cfg_.damping_thickness = sponge_thickness_;
      cfg_.damping_left = true;
      cfg_.damping_right = true;
      cfg_.damping_bottom = true;
      cfg_.damping_top = false;
      cfg_.damping_uniform = 0.0;
      cfg_.damping_scale_length = CartesianDampingRampIntegral(
         cfg_.damping_thickness, cfg_.damping_beta, cfg_.damping_exponent);
   }

   const TransientTopOptConfig &GetConfig() const override { return cfg_; }
   const std::string &GetMeshFile() const override
   {
      return mesh_description_;
   }

   real_t GetVolumeFraction() const override
   {
      return truth_volume_fraction_;
   }

   real_t GetAnalyticTruthVolumeFraction() const
   {
      return AnalyticTruthVolumeFraction();
   }

   bool HasComputedTruthVolumeFraction() const
   {
      return truth_volume_is_discrete_;
   }

   real_t GetActiveXMin() const { return active_x_min_; }
   real_t GetActiveXMax() const { return active_x_max_; }
   real_t GetActiveYMin() const { return active_y_min_; }
   real_t GetActiveYMax() const { return active_y_max_; }
   real_t GetSpongeThickness() const { return sponge_thickness_; }
   real_t GetTopPassiveThickness() const { return top_passive_thickness_; }
   real_t GetSourceXMin() const { return source_x_min_; }
   real_t GetSourceXMax() const { return source_x_max_; }
   int GetSourceBoundaryAttribute() const { return source_attr_; }
   int GetMeshNX() const { return nx_; }
   int GetMeshNY() const { return ny_; }

   Mesh CreateMesh() const override
   {
      Mesh mesh = Mesh::MakeCartesian2D(
         nx_, ny_, Element::QUADRILATERAL,
         /*generate_edges=*/true, length_, height_);

      Array<int> boundary_counts(source_attr_);
      boundary_counts = 0;
      const real_t tolerance = 1.0e-12 * std::max(length_, height_);
      for (int be_index = 0; be_index < mesh.GetNBE(); be_index++)
      {
         Element *boundary_element = mesh.GetBdrElement(be_index);
         Array<int> vertices;
         boundary_element->GetVertices(vertices);
         MFEM_VERIFY(vertices.Size() == 2,
                     "Elastic inclusion mesh expects segment boundaries.");

         const real_t *x_1 = mesh.GetVertex(vertices[0]);
         const real_t *x_2 = mesh.GetVertex(vertices[1]);
         const real_t x_mid = 0.5 * (x_1[0] + x_2[0]);
         const real_t y_mid = 0.5 * (x_1[1] + x_2[1]);

         int attribute = 0;
         if (std::abs(y_mid) <= tolerance)
         {
            attribute = bottom_attr_;
         }
         else if (std::abs(x_mid - length_) <= tolerance)
         {
            attribute = right_attr_;
         }
         else if (std::abs(y_mid - height_) <= tolerance)
         {
            attribute = (x_mid >= source_x_min_ && x_mid <= source_x_max_) ?
                        source_attr_ : top_attr_;
         }
         else if (std::abs(x_mid) <= tolerance)
         {
            attribute = left_attr_;
         }
         MFEM_VERIFY(attribute > 0,
                     "Could not classify an elastic-inclusion boundary edge.");
         boundary_element->SetAttribute(attribute);
         boundary_counts[attribute - 1]++;
      }
      mesh.SetAttributes();

      for (int attribute = 1; attribute <= source_attr_; attribute++)
      {
         MFEM_VERIFY(boundary_counts[attribute - 1] > 0,
                     "Elastic inclusion boundary attribute " << attribute
                     << " has no elements.");
      }
      return mesh;
   }

   void GetEssentialBoundaryAttributes(Array<int> &attrs) const override
   {
      CopyAttributes(cfg_.essential_bdr_attributes, attrs);
   }

   void GetAbsorbingBoundaryAttributes(Array<int> &attrs) const override
   {
      CopyAttributes(cfg_.absorbing_bdr_attributes, attrs);
   }

   void GetObservationBoundaryAttributes(Array<int> &attrs) const override
   {
      attrs.SetSize(4);
      attrs[0] = bottom_attr_;
      attrs[1] = right_attr_;
      attrs[2] = top_attr_;
      attrs[3] = left_attr_;
   }

   std::unique_ptr<VectorCoefficient>
   CreateBoundaryLoadCoefficient() const override
   {
      return std::make_unique<DirectionalBoundaryLoadCoefficient>(
         cfg_.boundary_load.direction);
   }

   std::unique_ptr<Coefficient>
   CreatePassiveRegionCoefficient() const override
   {
      auto bottom_and_top = std::make_unique<DoubleRectangularIndicator>(
         0.0, length_, 0.0, active_y_min_,
         0.0, length_, active_y_max_, height_);
      return std::make_unique<TripleRectangularIndicator>(
         std::make_unique<RectangularIndicator>(
            0.0, active_x_min_, 0.0, height_),
         std::make_unique<RectangularIndicator>(
            active_x_max_, length_, 0.0, height_),
         std::move(bottom_and_top));
   }

   real_t GetPassiveDensity() const override { return 1.0; }

   std::unique_ptr<DampingFieldBase>
   CreateDampingField(bool enabled = true) const override
   {
      // rho=1 maps to the upper SIMP scale r_max, including when that endpoint
      // was explicitly changed on the command line.
      return std::make_unique<DampingField>(
         GetMaterialParams(), GetDampingParameters(), enabled,
         GetMaterialParams().r_max);
   }

   bool HasReferenceTruth() const override { return true; }

   std::unique_ptr<Coefficient>
   CreateTruthDensityCoefficient() const override
   {
      return std::make_unique<ElasticInclusionTruthCoefficient>();
   }

   void SetComputedTruthVolumeFraction(real_t volume_fraction) override
   {
      MFEM_VERIFY(std::isfinite(volume_fraction) &&
                  volume_fraction > 0.0 && volume_fraction <= 1.0,
                  "Computed elastic-inclusion truth volume fraction must lie "
                  "in (0,1].");
      truth_volume_fraction_ = volume_fraction;
      cfg_.vol_frac = volume_fraction;
      truth_volume_is_discrete_ = true;
   }

   bool RequiresReferenceBoundaryData() const override { return true; }

   void SetBoundaryTraceHistory(
      std::shared_ptr<const BoundaryTraceHistory> history) override
   {
      MFEM_VERIFY(history,
                  "Elastic inclusion problem requires non-null trace data.");
      history->ValidateComplete();
      const Array<int> &marker = history->ObservationMarker();
      MFEM_VERIFY(marker.Size() >= source_attr_,
                  "Elastic inclusion trace marker does not cover all mesh "
                  "boundary attributes.");
      for (int attribute = bottom_attr_; attribute <= left_attr_; attribute++)
      {
         MFEM_VERIFY(marker[attribute - 1] == 1,
                     "Elastic inclusion trace marker is missing observation "
                     "attribute " << attribute << ".");
      }
      MFEM_VERIFY(marker[source_attr_ - 1] == 0,
                  "Elastic inclusion source boundary must not be observed.");
      trace_history_ = std::move(history);
   }

   const std::shared_ptr<const BoundaryTraceHistory> &
   GetBoundaryTraceHistory() const
   {
      return trace_history_;
   }

   std::unique_ptr<TimeIntegratedObjective>
   CreateObjective(ParFiniteElementSpace *state_fes,
                   MPI_Comm comm) const override
   {
      MFEM_VERIFY(trace_history_,
                  "Generate and attach elastic-inclusion reference boundary "
                  "data before creating its tracking objective.");
      return std::make_unique<BoundaryDisplacementTrackingObjective>(
         state_fes, trace_history_, comm);
   }

   bool Validate(std::ostream &err) const override
   {
      if (!TransientTopOptProblem::Validate(err)) { return false; }
      if (cfg_.mesh_file_is_user)
      {
         err << "Error: elastic-inclusion-identification uses its generated "
                "rectangular mesh; do not supply -mesh.\n";
         return false;
      }
      if (cfg_.volume_fraction_is_user)
      {
         err << "Error: elastic-inclusion-identification fixes the volume "
                "fraction from rho_dagger; do not supply -vf.\n";
         return false;
      }
      if (!std::isfinite(cfg_.filter_radius) || cfg_.filter_radius <= 0.0)
      {
         err << "Error: elastic inclusion filter radius must be finite and "
                "positive.\n";
         return false;
      }
      if (cfg_.boundary_load.domain_load ||
          cfg_.boundary_load.time_profile !=
             LoadTimeProfile::MODULATED_GAUSSIAN ||
          cfg_.boundary_load.bdr_attributes.Size() != 1 ||
          cfg_.boundary_load.bdr_attributes[0] != source_attr_)
      {
         err << "Error: elastic inclusion identification requires one "
                "modulated-Gaussian traction on boundary attribute 5.\n";
         return false;
      }
      if (cfg_.boundary_load.direction.Size() != 2 ||
          !std::isfinite(cfg_.boundary_load.direction[0]) ||
          !std::isfinite(cfg_.boundary_load.direction[1]) ||
          std::abs(cfg_.boundary_load.direction[0]) <= 0.0 ||
          std::abs(cfg_.boundary_load.direction[1]) <= 0.0)
      {
         err << "Error: elastic inclusion source must have nonzero normal "
                "and tangential components.\n";
         return false;
      }
      if (!std::isfinite(cfg_.boundary_load.frequency) ||
          !std::isfinite(cfg_.boundary_load.duration) ||
          cfg_.boundary_load.frequency <= 0.0 ||
          cfg_.boundary_load.duration <= 0.0)
      {
         err << "Error: elastic inclusion source frequency and duration must "
                "be finite and positive.\n";
         return false;
      }

      // The minimum listed distance is to a sponge interface, another shape,
      // or Gamma_src.  The separate 0.10 top collar keeps every exterior DOF
      // passive; it is not a damping sponge.
      const real_t clearances[] =
      {
         ElasticInclusionTruthCoefficient::SquareXMin() - active_x_min_,
         ElasticInclusionTruthCoefficient::SquareYMin() - active_y_min_,
         ElasticInclusionTruthCoefficient::TriangleY1() - active_y_min_,
         ElasticInclusionTruthCoefficient::DiskCenterX() -
            ElasticInclusionTruthCoefficient::DiskRadius() -
            ElasticInclusionTruthCoefficient::TriangleX2(),
         active_x_max_ - ElasticInclusionTruthCoefficient::DiskCenterX() -
            ElasticInclusionTruthCoefficient::DiskRadius(),
         ElasticInclusionTruthCoefficient::TriangleX1() -
            ElasticInclusionTruthCoefficient::SquareXMax(),
         height_ - ElasticInclusionTruthCoefficient::SquareYMax(),
         height_ - ElasticInclusionTruthCoefficient::TriangleY3(),
         height_ - ElasticInclusionTruthCoefficient::DiskCenterY() -
            ElasticInclusionTruthCoefficient::DiskRadius()
      };
      const real_t minimum_clearance =
         *std::min_element(std::begin(clearances), std::end(clearances));
      if (minimum_clearance + 1.0e-12 < 2.0 * cfg_.filter_radius)
      {
         err << "Error: elastic-inclusion geometry has minimum source/shape/"
                "sponge clearance " << minimum_clearance
             << ", smaller than two filter radii "
             << 2.0 * cfg_.filter_radius << ".\n";
         return false;
      }
      const bool square_inside =
         ElasticInclusionTruthCoefficient::SquareXMin() >= active_x_min_ &&
         ElasticInclusionTruthCoefficient::SquareXMax() <= active_x_max_ &&
         ElasticInclusionTruthCoefficient::SquareYMin() >= active_y_min_ &&
         ElasticInclusionTruthCoefficient::SquareYMax() <= active_y_max_;
      const bool triangle_inside =
         ElasticInclusionTruthCoefficient::TriangleX1() >= active_x_min_ &&
         ElasticInclusionTruthCoefficient::TriangleX1() <= active_x_max_ &&
         ElasticInclusionTruthCoefficient::TriangleX2() >= active_x_min_ &&
         ElasticInclusionTruthCoefficient::TriangleX2() <= active_x_max_ &&
         ElasticInclusionTruthCoefficient::TriangleX3() >= active_x_min_ &&
         ElasticInclusionTruthCoefficient::TriangleX3() <= active_x_max_ &&
         ElasticInclusionTruthCoefficient::TriangleY1() >= active_y_min_ &&
         ElasticInclusionTruthCoefficient::TriangleY1() <= active_y_max_ &&
         ElasticInclusionTruthCoefficient::TriangleY2() >= active_y_min_ &&
         ElasticInclusionTruthCoefficient::TriangleY2() <= active_y_max_ &&
         ElasticInclusionTruthCoefficient::TriangleY3() >= active_y_min_ &&
         ElasticInclusionTruthCoefficient::TriangleY3() <= active_y_max_;
      const bool disk_inside =
         ElasticInclusionTruthCoefficient::DiskCenterX() -
            ElasticInclusionTruthCoefficient::DiskRadius() >= active_x_min_ &&
         ElasticInclusionTruthCoefficient::DiskCenterX() +
            ElasticInclusionTruthCoefficient::DiskRadius() <= active_x_max_ &&
         ElasticInclusionTruthCoefficient::DiskCenterY() -
            ElasticInclusionTruthCoefficient::DiskRadius() >= active_y_min_ &&
         ElasticInclusionTruthCoefficient::DiskCenterY() +
            ElasticInclusionTruthCoefficient::DiskRadius() <= active_y_max_;
      if (!square_inside || !triangle_inside || !disk_inside)
      {
         err << "Error: an elastic inclusion lies outside the active interior.\n";
         return false;
      }
      return true;
   }

   void PrintSummary(std::ostream &out) const override
   {
      out << "Elastic inclusion identification: domain=[0," << length_
          << "]x[0," << height_ << "], mesh=" << nx_ << 'x' << ny_
          << " Q, active=[" << active_x_min_ << ',' << active_x_max_
          << "]x[" << active_y_min_ << ',' << active_y_max_
          << "], passive=left/right/bottom sponge collars plus top source "
             "collar, source_attr=" << source_attr_ << " source_x=["
          << source_x_min_ << ',' << source_x_max_
          << "], observation_attrs=[1,2,3,4], traction_direction=["
          << cfg_.boundary_load.direction[0] << ','
          << cfg_.boundary_load.direction[1] << "], square=["
          << ElasticInclusionTruthCoefficient::SquareXMin() << ','
          << ElasticInclusionTruthCoefficient::SquareXMax() << "]x["
          << ElasticInclusionTruthCoefficient::SquareYMin() << ','
          << ElasticInclusionTruthCoefficient::SquareYMax() << "] rho="
          << ElasticInclusionTruthCoefficient::SquareDensity()
          << ", triangle=[(" << ElasticInclusionTruthCoefficient::TriangleX1()
          << ',' << ElasticInclusionTruthCoefficient::TriangleY1() << "),("
          << ElasticInclusionTruthCoefficient::TriangleX2() << ','
          << ElasticInclusionTruthCoefficient::TriangleY2() << "),("
          << ElasticInclusionTruthCoefficient::TriangleX3() << ','
          << ElasticInclusionTruthCoefficient::TriangleY3() << ")] rho="
          << ElasticInclusionTruthCoefficient::TriangleDensity()
          << ", disk=center(" << ElasticInclusionTruthCoefficient::DiskCenterX()
          << ',' << ElasticInclusionTruthCoefficient::DiskCenterY() << ") r="
          << ElasticInclusionTruthCoefficient::DiskRadius() << " rho="
          << ElasticInclusionTruthCoefficient::DiskDensity()
          << ", truth_volume_fraction=" << truth_volume_fraction_
          << (truth_volume_is_discrete_ ? " (discrete)" : " (analytic provisional)")
          << "\n";
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

   // Measurement regions: ~2 wavelengths wide (wavelength ≈ 0.4, so 2λ ≈ 0.8)
   static constexpr real_t left_receiver_x_min_ = 0.90;
   static constexpr real_t left_receiver_x_max_ = 1.70;   // 0.8 units wide
   static constexpr real_t right_receiver_x_min_ = 6.30;  // 0.8 units wide
   static constexpr real_t right_receiver_x_max_ = 7.10;
   static constexpr real_t active_x_min_ = 2.10;
   static constexpr real_t active_x_max_ = 5.90;
   // The original striped f=5 study optimized directly between the receiver
   // collars.  Keep that historical layout selectable for initialization and
   // mesh-sensitivity studies without changing the paper experiment above.
   static constexpr real_t legacy_active_x_min_ = left_receiver_x_max_;
   static constexpr real_t legacy_active_x_max_ = right_receiver_x_min_;

   TransientTopOptConfig cfg;
   bool legacy_layout_ = false;
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
   explicit BandWaveguideProblem(const TransientTopOptConfig &base,
                                 bool legacy_layout = false)
      : cfg(base), legacy_layout_(legacy_layout)
   {
      if (legacy_layout_)
      {
         mesh_desc = "<generated 2D legacy-layout band waveguide>";
      }
      cfg.x_max = length_;
      cfg.y_max = height_;

      // Keep SIMP but avoid a near-disconnecting void slit. The finite contrast
      // makes multiple impedance interfaces more useful for this band-gap-style
      // reference problem.
      if (!cfg.simp_r_min_is_user) { cfg.material.r_min = 0.10; }
      if (!cfg.simp_r_max_is_user) { cfg.material.r_max = 1.0; }
      if (!cfg.simp_p_is_user) { cfg.material.simp_p = 3.0; }

      // Narrow center strip body force with a Gaussian-modulated carrier. With
      // c_p ~= 2, frequency 5 gives lambda_p ~= 0.4, so Bragg spacing
      // lambda/2 ~= 0.2 encourages more visible repeated interfaces along
      // each side of the longer waveguide. By default the Gaussian envelope
      // spans the complete simulated interval; -freq / -dur override these.
      cfg.boundary_load.domain_load = true;
      cfg.boundary_load.time_profile = LoadTimeProfile::MODULATED_GAUSSIAN;
      cfg.boundary_load.amplitude = 30.0;
      if (!cfg.load_duration_is_user)
      {
         cfg.boundary_load.duration = cfg.t_final;
      }
      if (!cfg.load_frequency_is_user) { cfg.boundary_load.frequency = 5.0; }
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
      cfg.damping_left = true;
      cfg.damping_right = true;
      cfg.damping_bottom = false;
      cfg.damping_top = false;
      cfg.damping_uniform = 0.0;
      cfg.damping_scale_length = legacy_layout_ ? real_t(0.25) :
         CartesianDampingRampIntegral(
            cfg.damping_thickness, cfg.damping_beta, cfg.damping_exponent);
   }

   const TransientTopOptConfig &GetConfig() const override { return cfg; }

   bool UsesPeriodicYBoundary() const override { return true; }

   const std::string &GetMeshFile() const override { return mesh_desc; }

   Mesh CreateMesh() const override
   {
      return Mesh::MakeCartesian2D(nx_, ny_, Element::QUADRILATERAL,
                                   /*generate_edges=*/true, length_, height_);
   }

   std::unique_ptr<DampingFieldBase>
   CreateDampingField(bool enabled = true) const override
   {
      const MaterialParams &mat = GetMaterialParams();
      const real_t passive_density = GetPassiveDensity();
      const real_t passive_material_scale =
         mat.r_min + std::pow(passive_density, mat.simp_p)
                     * (mat.r_max - mat.r_min);

      return std::make_unique<DampingField>(
         mat, GetDampingParameters(), enabled, passive_material_scale);
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

   std::unique_ptr<Coefficient> CreatePassiveRegionCoefficient() const override
   {
      // Active (designable) strip: [2.10,5.90] x [0,0.5].  The two
      // 0.4-wide collars between this strip and the receiver regions are
      // passive, as are the receiver and sponge regions.  The returned
      // coefficient is one in the passive part and zero in the active part.
      const real_t active_min = legacy_layout_ ?
                                legacy_active_x_min_ : active_x_min_;
      const real_t active_max = legacy_layout_ ?
                                legacy_active_x_max_ : active_x_max_;
      return std::make_unique<DoubleRectangularIndicator>(
         0.0, active_min, 0.0, height_,
         active_max, length_, 0.0, height_);
   }
};

// =============================================================================
// 2D BAND MODE CONVERTER: OPTIMIZABLE SPECTRAL-SEPARATION PILOT
// =============================================================================
// Layout along x:
//
//   [left sponge][smooth input collar][active design][output collar][right sponge]
//      0--0.75       0.75--4.25        4.25--8.00    8.00--11.25  11.25--12.0
//
// The standard variant launches the y-uniform axial mode from the input collar
// and tracks a higher transverse mode in the output collar.  Its forward RHS is
// deliberately smooth while its adjoint RHS is spectrally richer.  The reversed
// variant launches the high mode and maximizes harmonic correlation with the
// output fundamental; that state-independent low-mode adjoint source provides
// the complementary fine-forward/coarse-adjoint experiment.  Both variants use
// exactly the same mesh, passive collars, design region, and damping treatment.
// The separable axial-polarized Fourier patterns are manufactured spectral
// targets, not power-normalized elastic guided modes; propagation/phase checks
// are therefore required before interpreting this pilot as a physical converter.
class BandModeConverterProblem final : public TransientTopOptProblem
{
private:
   static constexpr real_t length_ = 12.0;
   static constexpr real_t height_ = 1.0;
   static constexpr int nx_ = 384;
   static constexpr int ny_ = 64;

   static constexpr int right_attr_ = 2;
   static constexpr int left_attr_ = 4;

   static constexpr real_t sponge_thickness_ = 0.75;
   static constexpr real_t input_collar_x_min_ = 0.75;
   static constexpr real_t input_collar_x_max_ = 4.25;
   static constexpr real_t source_x_min_ = 1.00;
   static constexpr real_t source_x_max_ = 4.00;
   static constexpr real_t active_x_min_ = 4.25;
   static constexpr real_t active_x_max_ = 8.00;
   static constexpr real_t output_collar_x_min_ = 8.00;
   static constexpr real_t output_collar_x_max_ = 11.25;
   static constexpr real_t target_x_min_ = 8.25;
   static constexpr real_t target_x_max_ = 10.75;

   TransientTopOptConfig cfg;
   bool reverse_spectral_roles_;
   bool modal_correlation_objective_;
   bool modal_energy_objective_;
   std::string mesh_desc = "<generated 2D band mode converter>";

   static void CopyAttributes(const Array<int> &src, Array<int> &dst)
   {
      dst.SetSize(src.Size());
      for (int i = 0; i < src.Size(); i++) { dst[i] = src[i]; }
   }

   std::unique_ptr<VectorCoefficient>
   MakeMode(real_t x_min, real_t x_max, int mode_y) const
   {
      Vector axial(2);
      axial = 0.0;
      axial[0] = 1.0;
      return std::make_unique<WaveguideModeCoefficient2D>(
         x_min, x_max, height_, mode_y, axial);
   }

   real_t CarrierPhase() const
   {
      constexpr real_t pi =
         3.1415926535897932384626433832795;
      // EvaluateLoadTimeFactor centers the modulated carrier at duration/2.
      // Express that same carrier in the objective's cos(2*pi*f*t+phase)
      // convention; propagation adds a separate phase that must be calibrated.
      return cfg.boundary_load.phase -
             pi * cfg.boundary_load.frequency *
             cfg.boundary_load.duration;
   }

public:
   explicit BandModeConverterProblem(const TransientTopOptConfig &base,
                                     bool reverse_spectral_roles = false,
                                     bool modal_correlation_objective = false,
                                     bool modal_energy_objective = false)
      : cfg(base), reverse_spectral_roles_(reverse_spectral_roles),
        modal_correlation_objective_(modal_correlation_objective),
        modal_energy_objective_(modal_energy_objective)
   {
      MFEM_VERIFY(!(modal_correlation_objective_ && modal_energy_objective_),
                  "Band converter cannot select correlation and energy "
                  "objectives simultaneously.");
      cfg.x_max = length_;
      cfg.y_max = height_;

      cfg.material.rho0 = 1.0;
      cfg.material.lambda0 = 2.0;
      cfg.material.mu0 = 1.0;
      cfg.material.r_min = 0.10;
      cfg.material.r_max = 1.0;
      cfg.material.simp_p = 3.0;

      // The spatial mode coefficient supplies all source localization.  A
      // Gaussian-modulated carrier localizes the temporal spectrum.  Five
      // carrier cycles keep the pulse duration independent of the observation
      // horizon; explicit -freq/-dur values override these defaults.
      cfg.boundary_load.domain_load = true;
      cfg.boundary_load.time_profile = LoadTimeProfile::MODULATED_GAUSSIAN;
      cfg.boundary_load.amplitude = 1.0;
      if (!cfg.load_frequency_is_user) { cfg.boundary_load.frequency = 5.0; }
      if (!cfg.load_duration_is_user)
      {
         cfg.boundary_load.duration = 1.0;
      }
      cfg.boundary_load.phase = 0.0;
      cfg.boundary_load.bdr_attributes.SetSize(0);
      cfg.boundary_load.direction.SetSize(2);
      cfg.boundary_load.direction = 0.0;
      cfg.boundary_load.direction[0] = 1.0;

      // Anchor the boundary behind the left sponge to remove rigid translation
      // from the K/M spectrum.  The right end keeps the outgoing-wave ABC; the
      // left sponge attenuates source leakage before it reaches the clamp.
      cfg.essential_bdr_attributes.SetSize(1);
      cfg.essential_bdr_attributes[0] = left_attr_;
      cfg.absorbing_bdr_attributes.SetSize(1);
      cfg.absorbing_bdr_attributes[0] = right_attr_;

      cfg.damping_thickness = sponge_thickness_;
      cfg.damping_left = true;
      cfg.damping_right = true;
      cfg.damping_bottom = false;
      cfg.damping_top = false;
      cfg.damping_uniform = 0.0;
      cfg.damping_scale_length = CartesianDampingRampIntegral(
         cfg.damping_thickness, cfg.damping_beta, cfg.damping_exponent);
   }

   const TransientTopOptConfig &GetConfig() const override { return cfg; }
   const std::string &GetMeshFile() const override { return mesh_desc; }
   bool UsesPeriodicYBoundary() const override { return true; }
   bool ReversesSpectralRoles() const { return reverse_spectral_roles_; }
   int GetTargetMode() const { return cfg.mode_converter_target_mode; }
   real_t GetTargetAmplitude() const
   {
      return cfg.mode_converter_target_amplitude;
   }

   bool GetModalSeedRegion(real_t &x_min, real_t &x_max,
                           int &transverse_mode) const override
   {
      x_min = active_x_min_;
      x_max = active_x_max_;
      transverse_mode = GetTargetMode();
      return true;
   }

   Mesh CreateMesh() const override
   {
      return Mesh::MakeCartesian2D(nx_, ny_, Element::QUADRILATERAL,
                                   /*generate_edges=*/true, length_, height_);
   }

   real_t GetPassiveDensity() const override { return 1.0; }

   std::unique_ptr<DampingFieldBase>
   CreateDampingField(bool enabled = true) const override
   {
      return std::make_unique<DampingField>(
         GetMaterialParams(), GetDampingParameters(), enabled,
         /*passive_material_scale=*/1.0);
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
      const int source_mode = reverse_spectral_roles_ ? GetTargetMode() : 0;
      return MakeMode(source_x_min_, source_x_max_, source_mode);
   }

   std::unique_ptr<VectorCoefficient> CreateForwardModalProbe() const override
   {
      const int source_mode = reverse_spectral_roles_ ? GetTargetMode() : 0;
      return MakeMode(target_x_min_, target_x_max_, source_mode);
   }

   std::unique_ptr<VectorCoefficient> CreateTargetModalProbe() const override
   {
      const int converted_mode = reverse_spectral_roles_ ? 0 : GetTargetMode();
      return MakeMode(target_x_min_, target_x_max_, converted_mode);
   }

   std::unique_ptr<TimeIntegratedObjective>
   CreateObjective(ParFiniteElementSpace *state_fes, MPI_Comm comm) const override
   {
      if (modal_energy_objective_)
      {
         return std::make_unique<WindowedModalEnergyObjective>(
            state_fes,
            MakeMode(target_x_min_, target_x_max_, GetTargetMode()),
            MakeMode(target_x_min_, target_x_max_, /*mode_y=*/0),
            /*converted_weight=*/1.0,
            cfg.mode_converter_energy_residual_weight,
            cfg.mode_converter_energy_window_start,
            cfg.mode_converter_energy_window_ramp, comm);
      }
      if (reverse_spectral_roles_ || modal_correlation_objective_)
      {
         const int objective_mode =
            reverse_spectral_roles_ ? 0 : GetTargetMode();
         return std::make_unique<HarmonicModalCorrelationObjective>(
            state_fes,
            MakeMode(target_x_min_, target_x_max_, objective_mode),
            GetTargetAmplitude(), cfg.boundary_load.frequency,
            CarrierPhase(), comm);
      }

      auto receiver = std::make_unique<AxialSinSquaredWindow2D>(
         target_x_min_, target_x_max_);
      auto target = MakeMode(target_x_min_, target_x_max_, GetTargetMode());
      return std::make_unique<HarmonicDisplacementTrackingObjective>(
         state_fes, std::move(receiver), std::move(target),
         GetTargetAmplitude(), cfg.boundary_load.frequency,
         CarrierPhase(), comm);
   }

   std::unique_ptr<Coefficient> CreatePassiveRegionCoefficient() const override
   {
      return std::make_unique<DoubleRectangularIndicator>(
         0.0, active_x_min_, 0.0, height_,
         active_x_max_, length_, 0.0, height_);
   }

   bool Validate(std::ostream &err) const override
   {
      if (!TransientTopOptProblem::Validate(err)) { return false; }
      if (GetTargetMode() <= 0)
      {
         err << "Error: band mode-converter target mode must be positive.\n";
         return false;
      }
      if (GetTargetMode() % 2 != 0)
      {
         err << "Error: periodic-y band mode-converter requires an even "
                "target mode in cos(n*pi*y/H).\n";
         return false;
      }
      if (!std::isfinite(GetTargetAmplitude()) || GetTargetAmplitude() <= 0.0)
      {
         err << "Error: band mode-converter target amplitude must be positive.\n";
         return false;
      }
      if (modal_energy_objective_ &&
          (!std::isfinite(cfg.mode_converter_energy_residual_weight) ||
           cfg.mode_converter_energy_residual_weight < 0.0 ||
           !std::isfinite(cfg.mode_converter_energy_window_start) ||
           !std::isfinite(cfg.mode_converter_energy_window_ramp) ||
           cfg.mode_converter_energy_window_start >= cfg.t_final ||
           cfg.mode_converter_energy_window_ramp < 0.0))
      {
         err << "Error: band modal-energy objective needs a nonnegative "
                "residual weight/ramp and a window start before t_final.\n";
         return false;
      }
      if (!std::isfinite(cfg.boundary_load.frequency) ||
          !std::isfinite(cfg.boundary_load.duration) ||
          cfg.boundary_load.frequency <= 0.0 ||
          cfg.boundary_load.duration <= 0.0)
      {
         err << "Error: band mode-converter carrier frequency and pulse "
                "duration must be positive.\n";
         return false;
      }

      const real_t shear_speed =
         std::sqrt(cfg.material.mu0 / cfg.material.rho0);
      const real_t shear_cutoff =
         GetTargetMode() * shear_speed / (2.0 * height_);
      real_t target_group_speed = 0.0;
      if (cfg.boundary_load.frequency <= shear_cutoff)
      {
         err << "WARNING: target transverse mode " << GetTargetMode()
             << " has shear cutoff frequency " << shear_cutoff
             << ", not below carrier " << cfg.boundary_load.frequency
             << "; verify propagation or lower the mode.\n";
      }
      else
      {
         const real_t cutoff_ratio =
            shear_cutoff / cfg.boundary_load.frequency;
         target_group_speed = shear_speed *
            std::sqrt(1.0 - cutoff_ratio * cutoff_ratio);
      }

      const real_t transverse_elements_per_wavelength =
         2.0 * ny_ / GetTargetMode();
      if (transverse_elements_per_wavelength < 12.0)
      {
         err << "WARNING: target transverse mode " << GetTargetMode()
             << " has only " << transverse_elements_per_wavelength
             << " order-1 elements per wavelength; verify with -r 1 or "
                "higher state order.\n";
      }

      const real_t p_speed = std::sqrt(
         (cfg.material.lambda0 + 2.0 * cfg.material.mu0) /
         cfg.material.rho0);
      const real_t source_group_speed =
         reverse_spectral_roles_ ? target_group_speed : p_speed;
      if (source_group_speed > 0.0)
      {
         // Both spatial fields have sin^2 envelopes, so their edge values are
         // zero.  Use the window centroids for a representative peak-to-peak
         // propagation estimate instead of the misleading support-edge gap.
         const real_t source_center =
            0.5 * (source_x_min_ + source_x_max_);
         const real_t target_center =
            0.5 * (target_x_min_ + target_x_max_);
         const real_t travel_time =
            (target_center - source_center) / source_group_speed;
         const real_t pulse_peak_arrival =
            0.5 * cfg.boundary_load.duration + travel_time;
         if (cfg.t_final <= pulse_peak_arrival)
         {
            err << "WARNING: t_final=" << cfg.t_final
                << " does not reach the estimated source-envelope peak at "
                << "the output (t~" << pulse_peak_arrival
                << "); use this setup only for wiring/spectrum work or extend "
                   "the run after a measured arrival test.\n";
         }
      }
      return true;
   }

   void PrintSummary(std::ostream &out) const override
   {
      out << "Band mode converter: roles="
          << (reverse_spectral_roles_ ? "high-forward/low-adjoint" :
              "low-forward/high-adjoint")
          << ", objective="
          << (modal_energy_objective_ ? "windowed-modal-energy" :
              ((reverse_spectral_roles_ || modal_correlation_objective_) ?
               "harmonic-modal-correlation" : "harmonic-tracking"))
          << ", target transverse mode=" << GetTargetMode()
          << ", target amplitude=" << GetTargetAmplitude()
          << ", target phase=" << CarrierPhase()
          << ", input collar=[" << input_collar_x_min_ << ','
          << input_collar_x_max_ << ']'
          << ", source=[" << source_x_min_ << ',' << source_x_max_ << ']'
          << ", active=[" << active_x_min_ << ',' << active_x_max_ << ']'
          << ", output collar=[" << output_collar_x_min_ << ','
          << output_collar_x_max_ << ']'
          << ", target=[" << target_x_min_ << ',' << target_x_max_ << "]\n";
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
   // The harmonic/L2 variant is a deliberately stage-sensitive elastic
   // experiment.  It keeps the same spatial finite-element model as the
   // dynamic-compliance cantilever, but drives it harmonically and measures a
   // receiver displacement energy instead of work-compliance.
   bool harmonic_l2_ = false;
   bool harmonic_tracking_ = false;
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
      // Put the harmonic actuator near the clamp and measure at the free end,
      // so the design sensitivity samples propagation through the full beam.
      const real_t x = (harmonic_l2_ || harmonic_tracking_) ? 0.35 : load_x_;
      return std::make_unique<ConcentratedLoadCoefficient>(
         x, load_y_, load_r_, dir);
   }

public:
   explicit CantileverComplianceProblem(const TransientTopOptConfig &base,
                                        bool harmonic_l2 = false,
                                        bool harmonic_tracking = false)
      : cfg(base), harmonic_l2_(harmonic_l2),
        harmonic_tracking_(harmonic_tracking)
   {
      MFEM_VERIFY(!(harmonic_l2_ && harmonic_tracking_),
                  "Cantilever experiment cannot select both harmonic objectives.");
      if (harmonic_l2_ || harmonic_tracking_)
      {
         mesh_desc = harmonic_tracking_
                     ? "<generated cantilever beam (harmonic tracking)>"
                     : "<generated cantilever beam (harmonic L2 receiver)>";
      }
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

      // The baseline uses the constant tip load whose work is the compliance
      // objective.  The harmonic/L2 variant instead applies a temporally fast,
      // spatially localized drive near the clamp and observes a free-end patch.
      // Its frequency is user-overridable with -freq.
      cfg.boundary_load.domain_load = true;
      cfg.boundary_load.time_profile = (harmonic_l2_ || harmonic_tracking_)
                                      ? LoadTimeProfile::HARMONIC
                                      : LoadTimeProfile::CONSTANT;
      cfg.boundary_load.amplitude = 1.0;
      if ((harmonic_l2_ || harmonic_tracking_) && !cfg.load_frequency_is_user)
      {
         cfg.boundary_load.frequency = 24.0;
      }
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
      if (harmonic_l2_)
      {
         auto receiver = std::make_unique<RectangularIndicator>(
            2.60, 2.98, 0.20, 0.80);
         return std::make_unique<DisplacementL2Objective>(
            state_fes, std::move(receiver), comm);
      }
      if (harmonic_tracking_)
      {
         Vector direction(2);
         direction = 0.0;
         direction[1] = -1.0;
         auto receiver = std::make_unique<RectangularIndicator>(
            2.60, 2.98, 0.20, 0.80);
         auto target = std::make_unique<ConcentratedLoadCoefficient>(
            2.80, 0.50, 0.20, direction);
         return std::make_unique<HarmonicDisplacementTrackingObjective>(
            state_fes, std::move(receiver), std::move(target),
            /*amplitude=*/1.0, cfg.boundary_load.frequency,
            /*phase=*/0.0, comm);
      }

      // Compliance J = int_0^T int_Omega f.u: the objective owns its own copy of
      // the same body force that drives the forward problem.
      return std::make_unique<ComplianceObjective>(
         state_fes, MakeLoadCoefficient(), comm);
   }
};

// =============================================================================
// 3D MODE-CONVERTER WAVEGUIDE: FIXED-DENSITY SPECTRAL PILOT
// =============================================================================
// A smooth full-length fundamental axial body mode is applied as a manufactured
// low-spectrum forward source.  The tracking objective asks for a
// fourth-by-fourth transverse axial mode in the output collar.  Thus F and dJ/du
// are constructed through the ordinary problem and objective interfaces but
// deliberately occupy well-separated spatial spectra.  The optional reversed
// variant instead drives the fourth-by-fourth mode and measures a full-length
// fundamental modal correlation.  Its state-independent adjoint source stays
// low-spectrum during the whole fixed-density sweep, making it a clean
// forward-fine/adjoint-coarse diagnostic.  Compact actuators are intentionally
// deferred: their support edges inject a broad spectral tail and obscure the
// eigenspace experiment this pilot is meant to isolate.
//
// This first instance is a homogeneous fixed-density diagnostic.  Run it with
// -rhs-spectrum to measure the two spectral distributions, or -forward-only to
// inspect the forward field.  A later optimization instance can make the middle
// section designable without changing either RHS definition.
class ModeConverterWaveguideProblem final : public TransientTopOptProblem
{
private:
   static constexpr real_t length_ = 6.0;
   static constexpr real_t width_ = 1.0;
   static constexpr real_t height_ = 1.0;
   static constexpr int nx_ = 24;
   static constexpr int ny_ = 8;
   static constexpr int nz_ = 8;

   static constexpr real_t output_x_min_ = 4.75;
   static constexpr real_t output_x_max_ = 5.75;
   static constexpr int target_mode_y_ = 4;
   static constexpr int target_mode_z_ = 4;

   // MakeCartesian3D boundary attributes: left x=0 -> 5, right x=L -> 3.
   static constexpr int left_attr_ = 5;
   static constexpr int right_attr_ = 3;

   TransientTopOptConfig cfg;
   bool reverse_spectral_roles_;
   std::string mesh_desc = "<generated 3D mode-converter waveguide>";

   std::unique_ptr<VectorCoefficient>
   MakeMode(real_t x_min, real_t x_max, int mode_y, int mode_z) const
   {
      Vector axial(3);
      axial = 0.0;
      axial[0] = 1.0;
      return std::make_unique<WaveguideModeCoefficient3D>(
         x_min, x_max, width_, height_, mode_y, mode_z, axial);
   }

public:
   explicit ModeConverterWaveguideProblem(const TransientTopOptConfig &base,
                                          bool reverse_spectral_roles = false)
      : cfg(base), reverse_spectral_roles_(reverse_spectral_roles)
   {
      cfg.x_max = length_;
      cfg.y_max = width_;

      cfg.material.rho0 = 1.0;
      cfg.material.lambda0 = 2.0;
      cfg.material.mu0 = 1.0;
      cfg.material.r_min = 0.10;
      cfg.material.r_max = 1.0;
      cfg.material.simp_p = 3.0;

      // The pilot is homogeneous by construction.  With uniform initialization,
      // vf=1 fixes both raw and filtered density at one.
      cfg.vol_frac = 1.0;
      cfg.filter_radius = 0.05;

      cfg.boundary_load.domain_load = true;
      cfg.boundary_load.time_profile = LoadTimeProfile::MODULATED_GAUSSIAN;
      cfg.boundary_load.amplitude = 1.0;
      if (!cfg.load_frequency_is_user) { cfg.boundary_load.frequency = 1.0; }
      if (!cfg.load_duration_is_user)
      {
         cfg.boundary_load.duration = cfg.t_final;
      }
      cfg.boundary_load.phase = 0.0;
      cfg.boundary_load.bdr_attributes.SetSize(0);
      cfg.boundary_load.direction.SetSize(3);
      cfg.boundary_load.direction = 0.0;
      cfg.boundary_load.direction[0] = 1.0;

      // Remove rigid-body modes for a clean K phi = lambda M phi spectrum.
      cfg.essential_bdr_attributes.SetSize(1);
      cfg.essential_bdr_attributes[0] = left_attr_;

      // A right sponge/ABC lets the same instance run meaningful forward waves.
      cfg.absorbing_bdr_attributes.SetSize(1);
      cfg.absorbing_bdr_attributes[0] = right_attr_;
      cfg.damping_thickness = 0.75;
      cfg.damping_left = false;
      cfg.damping_right = true;
      cfg.damping_bottom = false;
      cfg.damping_top = false;
      cfg.damping_uniform = 0.0;
      cfg.damping_scale_length = CartesianDampingRampIntegral(
         cfg.damping_thickness, cfg.damping_beta, cfg.damping_exponent);
   }

   const TransientTopOptConfig &GetConfig() const override { return cfg; }
   const std::string &GetMeshFile() const override { return mesh_desc; }

   Mesh CreateMesh() const override
   {
      return Mesh::MakeCartesian3D(nx_, ny_, nz_, Element::HEXAHEDRON,
                                   length_, width_, height_);
   }

   void GetEssentialBoundaryAttributes(Array<int> &attrs) const override
   {
      attrs = cfg.essential_bdr_attributes;
   }

   void GetAbsorbingBoundaryAttributes(Array<int> &attrs) const override
   {
      attrs = cfg.absorbing_bdr_attributes;
   }

   std::unique_ptr<VectorCoefficient>
   CreateBoundaryLoadCoefficient() const override
   {
      const int mode_y = reverse_spectral_roles_ ? target_mode_y_ : 0;
      const int mode_z = reverse_spectral_roles_ ? target_mode_z_ : 0;
      return MakeMode(0.0, length_, mode_y, mode_z);
   }

   std::unique_ptr<VectorCoefficient> CreateForwardModalProbe() const override
   {
      if (!reverse_spectral_roles_) { return nullptr; }
      return MakeMode(0.0, length_, target_mode_y_, target_mode_z_);
   }

   std::unique_ptr<TimeIntegratedObjective>
   CreateObjective(ParFiniteElementSpace *state_fes, MPI_Comm comm) const override
   {
      if (reverse_spectral_roles_)
      {
         // A full-length fundamental modal correlation has an adjoint source
         // that is spatially low mode and independent of u.  This avoids the
         // high-mode forward state re-entering dJ/du as it would for a
         // pointwise tracking residual.
         return std::make_unique<ComplianceObjective>(
            state_fes, MakeMode(0.0, length_, 0, 0), comm);
      }

      auto receiver = std::make_unique<BoxIndicator3D>(
         output_x_min_, output_x_max_,
         0.0, width_, 0.0, height_);
      auto target = MakeMode(output_x_min_, output_x_max_,
                             target_mode_y_, target_mode_z_);

      return std::make_unique<HarmonicDisplacementTrackingObjective>(
         state_fes, std::move(receiver), std::move(target),
         /*amplitude=*/1.0, cfg.boundary_load.frequency,
         /*phase=*/0.0, comm);
   }
};

// =============================================================================
// SPHERICAL BAND-GAP PROBLEM: 3D spherical wave shielding
// =============================================================================
// A radial (monopole) tone burst is emitted from a small central sphere. The
// design shell between source and receiver is optimized to minimize the
// time-integrated displacement energy in the receiver shell. An outer sponge
// shell plus absorbing boundary emulate an unbounded medium.
//
// Mesh: spherical_bandgap.msh, generated from spherical_bandgap.geo:
//   gmsh -3 -format msh2 spherical_bandgap.geo -o spherical_bandgap.msh
// Element attributes: 1 source, 2 design, 3 receiver, 4 gap, 5 damping.
// Boundary attribute 100 = outer r=10 sphere (absorbing).
//
// OPERATING POINT (kept consistent with the mesh resolution lc ~= 0.3):
//   c_p = sqrt((lambda0 + 2 mu0)/rho0) = 2, carrier f = 1.0 -> lambda_p = 2.0,
//   i.e. ~6-7 linear elements per P-wavelength. The design shell is 5.5 units
//   thick ~= 2.75 lambda_p. P-arrival at the receiver (r = 6) is t ~= 3.
//   With the default duration = t_final, the envelope peaks at t_final/2;
//   -tf 9 therefore lets the peak reach the receiver near t = 7.5. Source
//   energy emitted during the final ~3 time units cannot reach that shell
//   before the simulation ends. Recommended baseline: -tf 9 -dt 1e-3.
//   The current normalized spherical
//   sponge has max(gamma/mass) ~= 58.3, whose damping-only RK4 limit is about
//   0.048; in resolved high-order runs the spatial wave CFL is more restrictive
//   and is estimated from the assembled operator at startup.
class SphericalBandGapProblem final : public TransientTopOptProblem
{
private:
   static constexpr real_t r_source_outer_ = 0.5;
   static constexpr real_t r_design_inner_ = 0.5;
   static constexpr real_t r_design_outer_ = 6.0;
   static constexpr real_t r_receiver_inner_ = 6.0;
   static constexpr real_t r_receiver_outer_ = 7.0;
   static constexpr real_t r_gap_outer_ = 7.5;
   static constexpr real_t r_damping_inner_ = 7.5;
   static constexpr real_t r_damping_outer_ = 10.0;

   static constexpr int outer_boundary_attr_ = 100;  // gmsh Physical Surface

   TransientTopOptConfig cfg;

   static void CopyAttributes(const Array<int> &src, Array<int> &dst)
   {
      dst.SetSize(src.Size());
      for (int i = 0; i < src.Size(); i++) { dst[i] = src[i]; }
   }

public:
   explicit SphericalBandGapProblem(const TransientTopOptConfig &base)
      : cfg(base)
   {
      // Default mesh for this problem; an explicit -mesh (e.g. the coarse
      // variant spherical_bandgap_coarse.msh) wins.
      if (!cfg.mesh_file_is_user) { cfg.mesh_file = "spherical_bandgap.msh"; }

      // Material (same reference material as BandWaveguideProblem): c_p = 2.
      cfg.material.rho0 = 1.0;
      cfg.material.lambda0 = 2.0;
      cfg.material.mu0 = 1.0;
      cfg.material.r_min = 0.10;    // Finite contrast, avoid complete voids
      cfg.material.r_max = 1.0;
      cfg.material.simp_p = 3.0;

      // Source: narrowband radial tone burst (modulated Gaussian) in the
      // central sphere. The spatial monopole coefficient is unit-amplitude;
      // amplitude enters ONCE through the time profile.
      //
      // Carrier frequency vs mesh: resolving the carrier needs
      // lc <~ lambda_p/7 = c_p/(7 f). The shipped lc ~= 0.3 mesh supports
      // f = 1.0 (lambda_p = 2.0, ~6-7 elem/lambda) - the cheap local default.
      // Higher f packs more Bragg bands into the design shell (more
      // attractive band-gap physics) but requires a finer (HPC-scale) mesh:
      // e.g. -freq 5 needs lc ~= 0.06. Override with -freq. Unless -dur is
      // explicitly given, the Gaussian envelope spans the full simulation.
      cfg.boundary_load.domain_load = true;  // Body force, not boundary traction
      cfg.boundary_load.time_profile = LoadTimeProfile::MODULATED_GAUSSIAN;
      cfg.boundary_load.amplitude = 30.0;
      if (!cfg.load_frequency_is_user) { cfg.boundary_load.frequency = 1.0; }
      if (!cfg.load_duration_is_user)
      {
         cfg.boundary_load.duration = cfg.t_final;
      }
      cfg.boundary_load.phase = 0.0;
      cfg.boundary_load.bdr_attributes.SetSize(0);  // Not used (domain load)
      cfg.boundary_load.direction.SetSize(3);       // Placeholder (monopole overrides)
      cfg.boundary_load.direction = 0.0;

      // No clamped boundaries (free-floating sphere)
      cfg.essential_bdr_attributes.SetSize(0);

      // Absorbing boundary on outer surface
      cfg.absorbing_bdr_attributes.SetSize(1);
      cfg.absorbing_bdr_attributes[0] = outer_boundary_attr_;

      // Damping parameters consumed by the CreateDampingField() override below
      // (radial sponge in the outer shell; no Cartesian sides).
      cfg.damping_thickness = r_damping_outer_ - r_damping_inner_;  // 2.5 units
      // The spherical field normalizes its actual radial profile integral, so
      // it does not use the generic Cartesian scale-length parameter.
      cfg.damping_scale_length = 1.0;
      cfg.damping_reflection = 1e-4;         // Target reflection coefficient
      cfg.damping_beta = 2.0;
      cfg.damping_exponent = 2;
      cfg.damping_uniform = 0.0;
      cfg.damping_left = false;
      cfg.damping_right = false;
      cfg.damping_bottom = false;
      cfg.damping_top = false;
   }

   const TransientTopOptConfig &GetConfig() const override { return cfg; }

   // Use default CreateMesh() - reads cfg.mesh_file via ifstream

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

      // Guardrail for the bug class where J stays identically zero because the
      // simulation ends before the pulse can reach the receiver shell.
      const real_t c_p = std::sqrt((c.material.lambda0 + 2.0 * c.material.mu0)
                                   / c.material.rho0);
      const real_t t_needed = 0.5 * c.boundary_load.duration
                              + r_receiver_inner_ / c_p;
      if (c.t_final < t_needed)
      {
         err << "WARNING: -tf " << c.t_final << " is shorter than the pulse "
             << "travel time to the receiver shell (~" << t_needed << "). "
             << "The objective will be (near) zero.\n";
      }
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
      // Unit-amplitude monopole in the central sphere. The operator scales the
      // assembled base load by amplitude * time_profile(t); putting the
      // amplitude here as well would apply it twice.
      return std::make_unique<MonopoleSourceCoefficient>(
         r_source_outer_, /*amp=*/1.0);
   }

   std::unique_ptr<TimeIntegratedObjective>
   CreateObjective(ParFiniteElementSpace *state_fes, MPI_Comm comm) const override
   {
      // Minimize displacement energy in receiver shell (6.0 < r < 7.0)
      auto indicator = std::make_unique<SphericalShellIndicator>(
         r_receiver_inner_, r_receiver_outer_);

      return std::make_unique<DisplacementL2Objective>(
         state_fes, std::move(indicator), comm);
   }

   // The reference medium of the experiment (source/receiver/gap/damping
   // shells) is rho = 0.5 by definition, independent of the -vf budget for
   // the design shell.
   real_t GetPassiveDensity() const override { return 0.5; }

   std::unique_ptr<Coefficient> CreatePassiveRegionCoefficient() const override
   {
      // Passive (fixed ρ = 0.5): source + receiver + gap + damping
      // Active (designable): design region only (0.5 < r < 6.0)

      auto passive = std::make_unique<MultiSphericalShellIndicator>();
      passive->AddShell(0.0, r_source_outer_);              // Source sphere
      passive->AddShell(r_receiver_inner_, r_receiver_outer_);  // Receiver shell
      passive->AddShell(r_receiver_outer_, r_gap_outer_);       // Gap shell
      passive->AddShell(r_damping_inner_, r_damping_outer_);    // Damping shell

      return passive;
   }

   std::unique_ptr<DampingFieldBase> CreateDampingField(bool enabled = true) const override
   {
      // Override to use spherical damping instead of Cartesian damping
      // This demonstrates the modularity of the problem-agnostic framework:
      // by extending DampingFieldBase, we can plug in radial damping for spherical geometry

      if (!enabled)
      {
         // Return a standard DampingField with damping disabled
         return std::make_unique<DampingField>(GetMaterialParams(),
                                               GetDampingParameters(), false);
      }

      // Create spherical damping field for outer shell (7.5 < r < 10.0)
      // No damping in source, design, receiver, or gap regions
      const MaterialParams &mat = GetMaterialParams();
      const real_t passive_density = GetPassiveDensity();
      const real_t passive_material_scale =
         mat.r_min + std::pow(passive_density, mat.simp_p) * (mat.r_max - mat.r_min);

      return std::make_unique<SphericalDampingField>(
         GetMaterialParams(),
         r_damping_inner_,  // 7.5
         r_damping_outer_,  // 10.0
         passive_material_scale,
         cfg.damping_reflection,
         cfg.damping_beta,
         cfg.damping_exponent
      );
   }
};

} // namespace mfem

#endif // PROBLEM_SPECIFICATION_HPP
