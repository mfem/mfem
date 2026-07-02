// =============================================================================
// Problem Specification Interface for Transient Topology Optimization
// =============================================================================
//
// This header defines the abstraction layer that separates the **solver**
// (time integration, adjoint, optimization) from the **problem** (mesh,
// forcing, damping, objectives).
//
// ARCHITECTURE:
//   - Solver: ElastodynamicsSolver.hpp — agnostic to problem details
//   - Problem: This file — encodes mesh, BCs, loading, damping
//   - Objective: ObjectiveFunctional.hpp — loss functions + gradients
//
// USAGE:
//   1. Define a problem by implementing TransientTopOptProblem
//   2. Pass it to the solver via DesignObjectiveAdjointGradient
//   3. Solver queries problem for BCs, forcing, damping
//   4. Objective is pluggable via TimeIntegratedObjective interface
//
// =============================================================================

#ifndef PROBLEM_SPECIFICATION_HPP
#define PROBLEM_SPECIFICATION_HPP

#include "mfem.hpp"
#include "ObjectiveFunctional.hpp"
#include <string>

namespace mfem
{

// =============================================================================
// ABSTRACT DAMPING COEFFICIENT
// =============================================================================
// Base class for damping profiles. Subclasses define spatial variation.
// The solver uses this interface; problems provide implementations.
class DampingCoefficient : public Coefficient
{
public:
   virtual ~DampingCoefficient() = default;

   // Return damping coefficient at point (e.g., for Rayleigh damping: C = γ(x) M)
   virtual real_t Eval(ElementTransformation &T, const IntegrationPoint &ip) = 0;
};

// =============================================================================
// MESH-DRIVEN DAMPING (reads region from mesh attributes)
// =============================================================================
// Damping is 0 inside domain, ramps up in marked boundary layer.
// The ramp function is computed automatically from distance to the damping interface.
class MeshDrivenDampingCoefficient : public DampingCoefficient
{
private:
   ParMesh *mesh;
   Array<int> damping_interface_attrs;  // Boundary attributes marking damping region
   real_t gamma_max;     // Peak damping coefficient
   real_t thickness;     // Width of damping layer
   real_t rho;           // Material density (for Rayleigh damping)
   real_t beta;          // Exponential ramp parameter
   int m;                // Polynomial ramp exponent

   // Precomputed distance field (optional, for efficiency)
   mutable GridFunction *distance_to_interface;

public:
   /// Construct damping from mesh attributes
   /// @param pmesh Parallel mesh
   /// @param damping_attrs Array of boundary attributes marking the damping interface
   /// @param gmax Peak damping coefficient
   /// @param thick Thickness of damping layer (as fraction of domain size or absolute)
   /// @param density Material density
   /// @param b Exponential ramp parameter (default 2.0)
   /// @param mp Polynomial ramp exponent (default 2)
   MeshDrivenDampingCoefficient(ParMesh *pmesh,
                                 const Array<int> &damping_attrs,
                                 real_t gmax,
                                 real_t thick,
                                 real_t density,
                                 real_t b = 2.0,
                                 int mp = 2)
      : mesh(pmesh), damping_interface_attrs(damping_attrs),
        gamma_max(gmax), thickness(thick), rho(density), beta(b), m(mp),
        distance_to_interface(nullptr)
   {
      // TODO: Precompute distance field for efficiency
      // For now, compute on-the-fly in Eval()
   }

   virtual real_t Eval(ElementTransformation &T, const IntegrationPoint &ip) override
   {
      Vector x_point(mesh->SpaceDimension());
      T.Transform(ip, x_point);

      // Compute distance to damping interface
      // For now: simplified — assume damping interface bounds the domain
      // TODO: Implement proper signed-distance-field computation from mesh attributes

      Vector bbox_min(mesh->SpaceDimension()), bbox_max(mesh->SpaceDimension());
      mesh->GetBoundingBox(bbox_min, bbox_max);

      // Distance from boundary (simple per-axis version)
      real_t dist_from_boundary = thickness + 1.0;  // start outside
      for (int d = 0; d < mesh->SpaceDimension(); d++)
      {
         real_t dist_low = x_point(d) - bbox_min(d);
         real_t dist_high = bbox_max(d) - x_point(d);
         dist_from_boundary = std::min(dist_from_boundary,
                                       std::min(dist_low, dist_high));
      }

      // Ramp function: 0 inside, increases toward boundary
      if (dist_from_boundary >= thickness) return 0.0;

      real_t eta = (thickness - dist_from_boundary) / thickness;  // 0→1 toward boundary
      eta = std::min(std::max(eta, 0.0), 1.0);

      real_t eta_pow = std::pow(eta, m);
      real_t F_eta = (std::exp(beta * eta_pow) - 1.0) / (std::exp(beta) - 1.0);

      return rho * gamma_max * F_eta;
   }

   virtual ~MeshDrivenDampingCoefficient()
   {
      delete distance_to_interface;
   }
};

// =============================================================================
// CONSTANT DAMPING (simple uniform damping)
// =============================================================================
// For problems with constant damping coefficient everywhere.
class ConstantDampingCoefficient : public DampingCoefficient
{
private:
   real_t gamma_value;

public:
   ConstantDampingCoefficient(real_t gamma) : gamma_value(gamma) {}

   virtual real_t Eval(ElementTransformation &T, const IntegrationPoint &ip) override
   {
      return gamma_value;
   }
};

// =============================================================================
// FORCING SPECIFICATION (Neumann + Body Force)
// =============================================================================
// Explicit flag-based design: user says "yes/no" for each type,
// then provides the details only if "yes".
//
// USAGE EXAMPLES:
//
// 1. Boundary load only (wave shielding):
//    forcing.has_neumann = true;
//    forcing.neumann_boundaries = {21, 22, 23, 24, 25, 26};
//    forcing.neumann_spatial = new DownwardTractionCoef();
//    forcing.neumann_time_profile = GAUSSIAN;
//    forcing.neumann_amplitude = 30.0;
//    forcing.neumann_duration = 0.005;
//    forcing.has_body_force = false;
//
// 2. Body force only (gravity):
//    forcing.has_neumann = false;
//    forcing.has_body_force = true;
//    forcing.body_force_spatial = new GravityCoef(9.81);
//    forcing.body_force_time_profile = CONSTANT;
//
// 3. Both (seismic + self-weight):
//    forcing.has_neumann = true;
//    forcing.neumann_boundaries = {5};
//    forcing.neumann_spatial = new SeismicExcitationCoef();
//    forcing.neumann_time_profile = HARMONIC;
//    forcing.has_body_force = true;
//    forcing.body_force_spatial = new GravityCoef(9.81);
//    forcing.body_force_time_profile = CONSTANT;
//
struct ForcingSpec
{
   // ---------------------------------------------------------------------
   // NEUMANN (BOUNDARY) LOADING
   // ---------------------------------------------------------------------
   bool has_neumann;                   // YES/NO flag
   Array<int> neumann_boundaries;      // Which boundary attributes (if has_neumann=true)
   VectorCoefficient *neumann_spatial; // Spatial variation f(x) (if has_neumann=true)

   // Time modulation for Neumann (if has_neumann=true)
   enum TimeProfile { CONSTANT, GAUSSIAN, HARMONIC, CUSTOM };
   TimeProfile neumann_time_profile;
   real_t neumann_amplitude;           // Peak amplitude
   real_t neumann_duration;            // For GAUSSIAN: pulse duration; HARMONIC: period
   real_t neumann_phase;               // For HARMONIC: phase shift

   // ---------------------------------------------------------------------
   // BODY FORCE (DOMAIN RHS)
   // ---------------------------------------------------------------------
   bool has_body_force;                // YES/NO flag
   VectorCoefficient *body_force_spatial; // Spatial variation f(x) (if has_body_force=true)

   // Time modulation for body force (if has_body_force=true)
   TimeProfile body_force_time_profile;
   real_t body_force_amplitude;
   real_t body_force_duration;
   real_t body_force_phase;

   // ---------------------------------------------------------------------
   // CONSTRUCTOR (sensible defaults)
   // ---------------------------------------------------------------------
   ForcingSpec()
      : has_neumann(false), neumann_spatial(nullptr),
        neumann_time_profile(GAUSSIAN), neumann_amplitude(1.0),
        neumann_duration(1.0), neumann_phase(0.0),
        has_body_force(false), body_force_spatial(nullptr),
        body_force_time_profile(CONSTANT), body_force_amplitude(1.0),
        body_force_duration(1.0), body_force_phase(0.0) {}

   ~ForcingSpec()
   {
      // Coefficients owned by problem, not deleted here
   }

   // ---------------------------------------------------------------------
   // VALIDATION (call this in solver to catch user errors)
   // ---------------------------------------------------------------------
   bool Validate(std::ostream &err) const
   {
      if (has_neumann)
      {
         if (neumann_boundaries.Size() == 0)
         {
            err << "Error: has_neumann=true but neumann_boundaries is empty.\n";
            return false;
         }
         if (neumann_spatial == nullptr)
         {
            err << "Error: has_neumann=true but neumann_spatial is nullptr.\n";
            return false;
         }
      }
      if (has_body_force && body_force_spatial == nullptr)
      {
         err << "Error: has_body_force=true but body_force_spatial is nullptr.\n";
         return false;
      }
      if (!has_neumann && !has_body_force)
      {
         err << "Warning: No forcing specified (has_neumann=false, has_body_force=false).\n";
         // Not an error — free vibration / initial conditions only
      }
      return true;
   }
};

// =============================================================================
// PROBLEM SPECIFICATION INTERFACE
// =============================================================================
// Abstract interface for defining a transient topology optimization problem.
// Separates problem definition from solver implementation.
class TransientTopOptProblem
{
public:
   virtual ~TransientTopOptProblem() = default;

   // -------------------------------------------------------------------------
   // MESH & GEOMETRY
   // -------------------------------------------------------------------------
   /// Return the mesh file path
   virtual std::string GetMeshFile() const = 0;

   /// Return refinement level (0 = no refinement)
   virtual int GetRefinementLevel() const { return 0; }

   /// Return finite element order
   virtual int GetFEOrder() const { return 1; }

   // -------------------------------------------------------------------------
   // BOUNDARY CONDITIONS
   // -------------------------------------------------------------------------
   // Three types supported:
   //   1. Essential (Dirichlet): u = 0 (clamped)
   //   2. Absorbing (Robin/Impedance): σ·n = -Z u̇ (non-reflecting)
   //   3. Natural (Neumann): σ·n = f (traction, specified in ForcingSpec)
   //
   // USAGE:
   //   - Dirichlet: return boundary attribute list (e.g., {1} = clamped end)
   //   - Absorbing: return boundary attribute list (e.g., {10,11,12,13} = exterior)
   //   - Neumann: specified in GetForcing().neumann_boundaries
   //
   // Empty array = no BC of that type.

   /// Essential (Dirichlet) boundary attributes: u = 0
   /// Example: {1} for clamped end (Gmsh attr 1)
   virtual void GetEssentialBoundaryAttributes(Array<int> &ess_bdr_attrs) const = 0;

   /// Absorbing (impedance) boundary attributes: σ·n = -Z u̇
   /// Example: {10, 11, 12, 13} for exterior boundaries
   virtual void GetAbsorbingBoundaryAttributes(Array<int> &abc_bdr_attrs) const = 0;

   // -------------------------------------------------------------------------
   // FORCING
   // -------------------------------------------------------------------------
   /// Return forcing specification (Neumann + body force)
   virtual ForcingSpec GetForcing() const = 0;

   // -------------------------------------------------------------------------
   // DAMPING (OPTIONAL)
   // -------------------------------------------------------------------------
   // Rayleigh damping: C = γ(x) M, where γ(x) is the damping coefficient.
   //
   // USAGE:
   //   - No damping: return nullptr
   //   - Mesh-driven: return new MeshDrivenDampingCoefficient(mesh, attrs, ...)
   //   - Custom: return your own DampingCoefficient subclass
   //
   // SOLVER BEHAVIOR:
   //   - If nullptr: damping matrix C = 0 (no damping term in equations)
   //   - If not nullptr: C = γ(x) M assembled from the coefficient

   /// Return damping coefficient (nullptr = no damping)
   /// @param mesh Parallel mesh (needed for mesh-driven damping)
   virtual DampingCoefficient* GetDampingCoefficient(ParMesh *mesh) const = 0;

   // -------------------------------------------------------------------------
   // MATERIAL
   // -------------------------------------------------------------------------
   /// Material parameters (density, Lamé parameters, SIMP exponent)
   virtual void GetMaterialParams(real_t &rho0, real_t &lambda0, real_t &mu0,
                                  real_t &r_min, real_t &r_max, real_t &simp_p) const
   {
      rho0 = 1.0;
      lambda0 = 2.0;
      mu0 = 1.0;
      r_min = 1e-6;
      r_max = 1.0;
      simp_p = 3.0;
   }

   // -------------------------------------------------------------------------
   // OBJECTIVE FUNCTIONAL
   // -------------------------------------------------------------------------
   /// Create the objective functional for this problem
   virtual TimeIntegratedObjective* CreateObjective(
      ParFiniteElementSpace *state_fes, MPI_Comm comm) const = 0;

   // -------------------------------------------------------------------------
   // TIME INTEGRATION
   // -------------------------------------------------------------------------
   /// Final time
   virtual real_t GetFinalTime() const = 0;

   /// Time step
   virtual real_t GetTimeStep() const = 0;

   // -------------------------------------------------------------------------
   // OPTIMIZATION
   // -------------------------------------------------------------------------
   /// Target volume fraction
   virtual real_t GetVolumeFraction() const = 0;

   /// Filter radius
   virtual real_t GetFilterRadius() const = 0;

   /// Maximum MMA iterations
   virtual int GetMaxIterations() const { return 100; }

   /// Move limit
   virtual real_t GetMoveLimit() const { return 0.2; }

   /// Convergence tolerance
   virtual real_t GetTolerance() const { return 1e-3; }
};

// =============================================================================
// EXAMPLE: WAVE SHIELDING PROBLEM (current problem as reference)
// =============================================================================
class WaveShieldingProblem : public TransientTopOptProblem
{
private:
   std::string mesh_file;
   real_t t_final, dt, vol_frac, filter_radius;

public:
   WaveShieldingProblem(const char *mesh = "lamb-problem-damping-mesh-triangs.msh",
                        real_t tf = 0.3, real_t dt_ = 1e-4,
                        real_t vf = 0.5, real_t fr = 0.03)
      : mesh_file(mesh), t_final(tf), dt(dt_), vol_frac(vf), filter_radius(fr) {}

   std::string GetMeshFile() const override { return mesh_file; }

   void GetEssentialBoundaryAttributes(Array<int> &ess_bdr) const override
   {
      ess_bdr.SetSize(0);  // No clamped boundaries in wave shielding
   }

   void GetAbsorbingBoundaryAttributes(Array<int> &abc_bdr) const override
   {
      // Gmsh attributes 10-13: exterior boundaries
      abc_bdr.SetSize(4);
      abc_bdr[0] = 10;  // exterior_left
      abc_bdr[1] = 11;  // exterior_bottom
      abc_bdr[2] = 12;  // exterior_right
      abc_bdr[3] = 13;  // exterior_top
   }

   ForcingSpec GetForcing() const override
   {
      ForcingSpec forcing;

      // ----------------------------------------------------------------
      // NEUMANN (boundary traction on load strips)
      // ----------------------------------------------------------------
      forcing.has_neumann = true;

      // Which boundaries: attrs 21-26 from Gmsh (load_strip_1 ... load_strip_6)
      forcing.neumann_boundaries.SetSize(6);
      for (int i = 0; i < 6; i++) forcing.neumann_boundaries[i] = 21 + i;

      // Spatial variation: downward traction
      class DownwardTraction : public VectorCoefficient
      {
      public:
         DownwardTraction(int dim) : VectorCoefficient(dim) {}
         void Eval(Vector &V, ElementTransformation &T, const IntegrationPoint &ip) override
         {
            V.SetSize(vdim);
            V = 0.0;
            V(1) = -1.0;  // Unit traction in -y direction
         }
      };
      forcing.neumann_spatial = new DownwardTraction(2);

      // Time variation: Gaussian pulse
      forcing.neumann_time_profile = ForcingSpec::GAUSSIAN;
      forcing.neumann_amplitude = 30.0;
      forcing.neumann_duration = 0.005;
      forcing.neumann_phase = 0.0;

      // ----------------------------------------------------------------
      // BODY FORCE (none for wave shielding)
      // ----------------------------------------------------------------
      forcing.has_body_force = false;

      return forcing;
   }

   DampingCoefficient* GetDampingCoefficient(ParMesh *mesh) const override
   {
      // Use mesh-driven damping with attribute 30 (interior_damping_interface)
      Array<int> damping_attrs(1);
      damping_attrs[0] = 30;

      // Compute gamma_max from wave speed (as in original code)
      real_t rho0 = 1.0, lambda0 = 2.0, mu0 = 1.0;
      real_t c_p = sqrt((lambda0 + 2.0*mu0) / rho0);
      real_t gamma_max = (2.0 * c_p / 0.2136) * log(1.0 / 1e-4);

      return new MeshDrivenDampingCoefficient(mesh, damping_attrs, gamma_max,
                                               0.25, rho0, 2.0, 2);
   }

   TimeIntegratedObjective* CreateObjective(ParFiniteElementSpace *fes,
                                            MPI_Comm comm) const override
   {
      // Protected circular region at domain center
      Vector bbox_min(2), bbox_max(2);
      fes->GetParMesh()->GetBoundingBox(bbox_min, bbox_max);
      real_t x_center = 0.5 * (bbox_min(0) + bbox_max(0));
      real_t y_center = 0.5 * (bbox_min(1) + bbox_max(1));

      SubdomainIndicator *indicator = new SubdomainIndicator(x_center, y_center, 0.2);
      return new DisplacementL2Objective(fes, indicator, comm);
   }

   real_t GetFinalTime() const override { return t_final; }
   real_t GetTimeStep() const override { return dt; }
   real_t GetVolumeFraction() const override { return vol_frac; }
   real_t GetFilterRadius() const override { return filter_radius; }
};

} // namespace mfem

#endif // PROBLEM_SPECIFICATION_HPP
