
/** @file
    Single-element cut quadrature.

    Level sets describe one reference element. Backends construct reference
    rules; the generator supplies reference geometry data; a separate mapping
    operation supplies physical weights and data. There are no batch methods,
    mesh caches, version counters, or persistent level-set sources.

    This API replaces
    the legacy cut-rule API without compatibility wrappers or dependencies on
    its moment-fitting implementation. */

#ifndef MFEM_CUT_QUADRATURE
#define MFEM_CUT_QUADRATURE

#include "../config/config.hpp"
#include "../general/array.hpp"
#include "../linalg/densemat.hpp"
#include "intrules.hpp"
#include "eltrans.hpp"
#include "coefficient.hpp"
#include "gridfunc.hpp"

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace mfem
{

/** Computation outcome, independent of whether the domain is empty or full. */
enum class CutQuadratureStatus
{
   Success,
   InvalidInput,
   Unsupported,
   DegenerateGeometry,
   NumericalFailure
};

/** @brief Outcome and optional moment-fitting diagnostics.

    A successful call never silently lowers the requested nominal order.
    Unsupported orders return Unsupported. Moment-system rank, convergence,
    and tolerance failures return NumericalFailure. A valid empty domain is
    Success; a singular geometric mapping returns DegenerateGeometry.

    moment_degree is the total degree of the fitted reference polynomial
    space, or -1 when not applicable. moment_residual is
    ||A weights - moments||_inf / max(1, ||moments||_inf), or -1 if unavailable.
    It measures the fit, not the accuracy of the supplied moments, the level-set
    approximation, or integration of arbitrary functions. Backend-specific
    solver tolerances and refinement policies belong to concrete backends. */
struct CutQuadratureReport
{
   CutQuadratureStatus status = CutQuadratureStatus::Success;
   std::string message;
   int moment_degree = -1;
   real_t moment_residual = -1;

   bool Succeeded() const { return status == CutQuadratureStatus::Success; }
};

/** @brief A scalar level set in MFEM reference-element coordinates.

    Evaluation is required throughout the closed reference element.
    Gradients have GetDim() entries and are with respect to these coordinates.
    Implementations and their finite elements need not be reentrant; callers
    must not share mutable evaluation state across concurrent calls. */
class ReferenceLevelSet
{
public:
   virtual ~ReferenceLevelSet() = default;
   virtual Geometry::Type GetGeometry() const = 0;
   int GetDim() const;

   virtual real_t Eval(const IntegrationPoint &point) const = 0;
   virtual void EvalGradient(const IntegrationPoint &point,
                             Vector &gradient) const = 0;

   /** Return guaranteed bounds on the represented function, or false when
       unavailable. Bounds on an approximation do not bound its original
       physical coefficient. Outputs are unspecified on false. The default
       implementation returns false. */
   virtual bool GetBounds(real_t &lower, real_t &upper) const;

   /** Default implementation returns nullptr. */
   virtual const class PolynomialLevelSet *AsPolynomial() const;
};

/** @brief A supported scalar, VALUE-mapped polynomial finite element and dofs.

    Set and Project copy dofs, but borrow the finite element. The element must
    outlive this object and must not change while it is used. Rational bases,
    unsupported polynomial spaces, and INTEGRAL-mapped fields are rejected.
    A positive basis on a pyramid is not automatically a Cartesian polynomial.

    Default construction leaves the object uninitialized. Failed Set/Project
    calls clear it. GetGeometry returns INVALID and AsPolynomial returns nullptr
    while uninitialized. Evaluation and GetFE require successful initialization.
    There is no retained Bernstein conversion or element-dependent cache. */
class PolynomialLevelSet : public ReferenceLevelSet
{
public:
   PolynomialLevelSet() = default;

   CutQuadratureReport Set(const FiniteElement &element,
                           const Vector &values);

   /** Extract local scalar dofs, including the local dof transformation.
       The GridFunction and its finite-element space must remain valid and
       unchanged for the lifetime of the borrowed finite element. */
   CutQuadratureReport Set(const GridFunction &function, int element);

   /** Project coefficient(T(point)) into the supplied reference polynomial
       space. Projection is local; it does not construct a mesh-wide field.
       Interpolate in the matching Gauss-Legendre nodal tensor space, then
       convert to the supplied basis. Exactly representable polynomials are
       preserved, including for positive (Bernstein) elements. */
   CutQuadratureReport Project(Coefficient &coefficient,
                               ElementTransformation &transformation,
                               const FiniteElement &element);

   bool IsInitialized() const { return element != nullptr; }
   const FiniteElement &GetFE() const;
   const Vector &GetDofs() const { return dofs; }

   /** Convert exactly, up to roundoff, to a supplied supported Bernstein
       basis on the same reference geometry spanning this polynomial space.
       No interpolation or degree reduction is permitted. Output uses MFEM
       dof ordering and the geometry's native basis: barycentric on simplices,
       tensor-product on boxes. Exporting tensor-product data to ALGOIM needs
       a permutation; exporting simplex data to a box needs a separate basis
       conversion and domain restriction inside the backend. */
   CutQuadratureReport GetBernsteinCoefficients(
      const FiniteElement &bernstein_element, Vector &coefficients) const;

   Geometry::Type GetGeometry() const override;
   real_t Eval(const IntegrationPoint &point) const override;
   void EvalGradient(const IntegrationPoint &point,
                     Vector &gradient) const override;
   bool GetBounds(real_t &lower, real_t &upper) const override;
   const PolynomialLevelSet *AsPolynomial() const override;

private:
   const FiniteElement *element = nullptr;
   Vector dofs;
};

/** @brief Reference-coordinate callbacks with an explicitly supplied gradient.

    Both callbacks must be nonempty. There is no implicit finite-difference
    gradient. Real-valued callbacks do not satisfy ALGOIM's interval-arithmetic
    contract; a backend without suitable support must return Unsupported. */
class FunctionLevelSet : public ReferenceLevelSet
{
public:
   using ValueFunction = std::function<real_t(const IntegrationPoint &)>;
   using GradientFunction =
      std::function<void(const IntegrationPoint &, Vector &)>;

   FunctionLevelSet(Geometry::Type geometry, ValueFunction value,
                    GradientFunction gradient);

   Geometry::Type GetGeometry() const override { return geometry; }
   real_t Eval(const IntegrationPoint &point) const override;
   void EvalGradient(const IntegrationPoint &point,
                     Vector &result) const override;

private:
   Geometry::Type geometry;
   ValueFunction value;
   GradientFunction gradient;
};

enum class LevelSetSign : signed char { Negative = -1, Zero = 0, Positive = 1 };

/** @brief An intersection of sign constraints inside an element or local face.

    Negative and Positive mean strict inequalities; Zero means equality.
    The default-constructed domain is the whole element. OnFace restricts the
    carrier to a codimension-one facet: a vertex in 1D, an edge in 2D, or a
    face in 3D, using MFEM's corresponding local entity numbering. In particular,
    triangles use their three edges, not Element::GetNFaces(). All quadrature
    coordinates remain parent-element coordinates, including facet rules.

    Codimension counts Zero constraints plus a face restriction. It is only
    the nominal codimension: Zero gradients and the face conormal must be
    independent on the requested manifold. Coincident interfaces, tangential
    intersections, and identically zero equality constraints require explicit
    handling or a DegenerateGeometry result; they are not ordinary empty cuts.

    Validation rejects invalid indices, repeated constraints on the same level
    set, invalid face numbers, and codimension greater than element dimension.
    Regular zero-dimensional intersections use counting measure. */
class CutDomain
{
public:
   struct Constraint
   {
      int level_set;
      LevelSetSign sign;
   };

   static CutDomain Positive(int level_set = 0);
   static CutDomain Negative(int level_set = 0);
   static CutDomain Surface(int level_set = 0);
   static CutDomain Intersection(int first_level_set, int second_level_set);

   CutDomain &Add(int level_set, LevelSetSign sign);
   CutDomain &OnFace(int face);

   int NumConstraints() const;
   const Constraint &GetConstraint(int index) const;
   int GetFace() const { return face; }
   int Codimension() const;
   CutQuadratureReport Validate(Geometry::Type geometry,
                                int num_level_sets) const;

private:
   std::vector<Constraint> constraints;
   int face = -1;
};

/** @brief One integration request, independent of physical weight conventions.

    order is a nonnegative nominal MFEM integration order, not a certificate of
    exactness on curved domains. Backends document how it selects their fitting
    degree or underlying quadrature parameters. Successful output rules set
    IntegrationRule::GetOrder() to this nominal order.

    With fixed_points == nullptr, every nonzero-weight point must lie in the
    closure of the requested domain, within the backend's geometric tolerance.
    In particular, surface points lie on the interface, including for moment
    fitting. Signed weights are permitted; positivity is not guaranteed.

    Non-null fixed_points requests weight fitting on exactly those coordinates,
    in that order. Its input weights are ignored; points are borrowed only for
    this call and must be in the closed reference element. This initial API
    permits fixed points only for full-dimensional volume domains, with no
    face restriction. Unsupported backends return Unsupported rather than
    replacing the points. Full and empty cells preserve the same point layout;
    empty-cell weights are all zero. Output point indices are 0, ..., npoints-1.
    An empty prescribed point set is invalid. */
struct CutQuadratureRequest
{
   CutDomain domain = CutDomain::Positive();
   int order = 2;
   const IntegrationRule *fixed_points = nullptr;
};

/** Relation to the carrier: the whole element, or the requested face.

    Full means the domain covers the carrier up to a set of zero measure and
    applies only without equality constraints. Empty means zero measure in
    the requested manifold dimension, not necessarily an empty set.
    Unknown is the uninitialized/failed state, never a successful result. */
enum class CutRegion { Unknown, Empty, Full, Cut };

/** @brief A reference rule with the geometric data needed for later mapping.

    Weights always integrate reference measure. All coordinates are in the
    parent element. The generator fills this object using backend-produced
    points, weights, and region, then supplies geometry, domain, and conormals.

    conormals has dimensions dim x codimension x npoints. For each Zero
    constraint its column is grad(phi)/|grad(phi)|, toward phi > 0, in constraint
    order. The outward reference face conormal is last. Columns are unit but
    need not be orthogonal. They are intrinsic conormals, not ambient normals.
    Columns at zero-weight nodes are zero and must not be used for geometry.
    For codimension zero the tensor is empty.

    The object owns its data and retains no references to the input level sets.
    Clear resets geometry, domain, region, rule, and conormals. */
struct CutQuadrature
{
   Geometry::Type geometry = Geometry::INVALID;
   CutDomain domain;
   CutRegion region = CutRegion::Unknown;
   IntegrationRule rule;
   DenseTensor conormals;

   int GetNPoints() const { return rule.GetNPoints(); }
   void Clear();
};

/** @brief Backend for one reference element and one requested domain.

    Calls receive no mesh or physical transformation and retain no input
    references or element-dependent state. Temporary work is private to the
    call. Algorithm configuration may be immutable backend state.

    CheckSupport examines actual input types, geometry, polynomial spaces,
    nominal order, constraints, and prescribed points. It is not a geometric
    regularity or convergence certificate. Generate must also reject invalid
    or unsupported input when called directly, without CheckSupport.

    Generate handles empty, full, and cut domains, including fixed-point
    layouts. It fills rule and region; it never normalizes or maps weights.
    On failure it clears rule and sets region to Unknown. The implementation
    must distinguish degenerate geometry from an ordinary zero-measure domain. */
class CutQuadratureBackend
{
public:
   virtual ~CutQuadratureBackend() = default;
   virtual std::string Name() const = 0;

   virtual CutQuadratureReport CheckSupport(
      Geometry::Type geometry,
      const Array<const ReferenceLevelSet *> &level_sets,
      const CutQuadratureRequest &request) const = 0;

   virtual CutQuadratureReport Generate(
      Geometry::Type geometry,
      const Array<const ReferenceLevelSet *> &level_sets,
      const CutQuadratureRequest &request,
      IntegrationRule &rule, CutRegion &region) const = 0;
};

/** Construct a compiled-in backend with its default configuration.
    Returns nullptr for an unknown or unavailable backend. Concrete backend
    constructors provide algorithm-specific configuration. */
std::unique_ptr<CutQuadratureBackend> CreateCutQuadratureBackend(
   const std::string &name);

/** Names such as "algoim" and "moment-fitting" available in this build. */
std::vector<std::string> GetCutQuadratureBackends();

/** With measure ratio mu and volume factor volume = Tr.Weight():

    Reference:  weight = reference_weight
    Physical:   weight = reference_weight * mu
    Integrator: weight = reference_weight * mu / volume

    Integrator weights are for element integrators whose measure factor is
    that of this same parent transformation. They are not directly usable as
    face-coordinate rules in FaceElementTransformations-based integrators.
    Weight conversion alone does not turn volume operators into intrinsic
    surface differential operators. */
enum class CutWeightType { Reference, Physical, Integrator };

enum CutQuadratureData : int
{
   CQ_RULE = 0,
   CQ_PHYSICAL_POINTS = 1 << 0,
   CQ_PHYSICAL_CONORMALS = 1 << 1,
   CQ_MEASURE_SCALING = 1 << 2
};

struct CutMappingOptions
{
   CutWeightType weight_type = CutWeightType::Integrator;
   int data = CQ_RULE;
};

/** @brief A mapped copy; reference quadrature weights remain untouched.

    rule still uses parent reference coordinates, with weights in weight_type.
    Optional arrays are populated only when requested: physical_points is
    sdim x npoints, physical_conormals is sdim x codimension x npoints, and
    measure_scaling has npoints entries. Unrequested arrays are empty.
    Conormals and scaling at zero-weight nodes are zero and not geometric data.
    Point coordinates may still be mapped there. Clear resets all fields. */
struct MappedCutQuadrature
{
   CutRegion region = CutRegion::Unknown;
   CutWeightType weight_type = CutWeightType::Reference;
   IntegrationRule rule;
   DenseMatrix physical_points;
   DenseTensor physical_conormals;
   Vector measure_scaling;

   int GetNPoints() const { return rule.GetNPoints(); }
   void Clear();
};

/** @brief Map a reference rule without recomputing its level sets.

    The input is immutable and its type contains reference weights only, so
    repeated calls cannot accidentally transform already mapped weights.
    Geometry must match transformation.GetGeometryType(). The caller ensures
    that the reference rule describes the intended domain for this mapping;
    a moved mesh may require reprojection and rule regeneration first.

    For J = dT/dxi, G = J^T J, and reference conormal matrix N:
       mu = sqrt(det(G)) * sqrt(det(N^T G^{-1} N) / det(N^T N)).
    For codimension zero mu = Tr.Weight(), including embedded elements.
    Physical conormals are the normalized columns of J G^{-1} N.
    J and the active conormals must have full column rank; singular cases
    return DegenerateGeometry. There are no gradient evaluations at
    zero-weight nodes. On failure out is cleared, never partly usable.

    Mapping generally does not preserve reference polynomial exactness.
    The transformation's current integration point is restored on return. */
CutQuadratureReport MapCutQuadrature(
   ElementTransformation &transformation, const CutQuadrature &reference,
   const CutMappingOptions &options, MappedCutQuadrature &out);

/** @brief Single-element front end borrowing a backend, not a mesh or source.

    The backend must outlive this object. Each call validates inputs, invokes
    the backend, and evaluates reference conormals at active manifold nodes.
    There are no full/empty shortcuts that replace a prescribed node layout.
    Failures clear out. Every referenced level set must be non-null, initialized
    when applicable, and defined on the supplied reference geometry.

    Concurrent calls require distinct outputs, transformations, and any
    non-reentrant level-set/finite-element evaluation state. Const methods do
    not by themselves guarantee that user callbacks or MFEM elements are safe
    to share. The generator retains no per-element state. */
class CutQuadratureGenerator
{
public:
   explicit CutQuadratureGenerator(const CutQuadratureBackend &backend);
   const CutQuadratureBackend &GetBackend() const { return backend; }

   CutQuadratureReport Generate(
      Geometry::Type geometry,
      const Array<const ReferenceLevelSet *> &level_sets,
      const CutQuadratureRequest &request, CutQuadrature &out) const;

   /** Convenience composition of reference generation and MapCutQuadrature.
       Inputs are already pulled back/projected for this element. Geometry is
       taken from transformation, never from a duplicate request field.
       Successful reports preserve the reference-generation diagnostics. */
   CutQuadratureReport Generate(
      ElementTransformation &transformation,
      const Array<const ReferenceLevelSet *> &level_sets,
      const CutQuadratureRequest &request,
      const CutMappingOptions &mapping, MappedCutQuadrature &out) const;

private:
   const CutQuadratureBackend &backend;
};

}

#endif
