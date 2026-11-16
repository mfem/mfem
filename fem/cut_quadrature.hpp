
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
   int GetDim() const;

   /// Return the MFEM reference-element geometry on which this LS is defined.
   virtual Geometry::Type GetReferenceGeometry() const = 0;

   /// Value of the LS at an IntegrationPoint.
   virtual real_t Eval(const IntegrationPoint &point) const = 0;

   /// Gradient of the LS at an IntegrationPoint.
   virtual void EvalGradient(const IntegrationPoint &point,
                             Vector &gradient) const = 0;

   virtual ~ReferenceLevelSet() = default;
};

/** @brief A supported scalar, VALUE-mapped polynomial finite element and dofs.

    Set and Project copy dofs, but borrow the finite element. The element must
    outlive this object and must not change while it is used. Rational bases,
    unsupported polynomial spaces, and INTEGRAL-mapped fields are rejected.
    A positive basis on a pyramid is not automatically a Cartesian polynomial.

    Default construction leaves the object uninitialized. Failed Set/Project
    calls clear it. GetReferenceGeometry returns INVALID while uninitialized.
    Evaluation and GetFE require successful initialization.
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

   Geometry::Type GetReferenceGeometry() const override;
   real_t Eval(const IntegrationPoint &point) const override;
   void EvalGradient(const IntegrationPoint &point,
                     Vector &gradient) const override;
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
   using ValueFunction =
      std::function<real_t(const IntegrationPoint &)>;
   using GradientFunction =
      std::function<void(const IntegrationPoint &, Vector &)>;

   FunctionLevelSet(Geometry::Type geometry,
                    ValueFunction value, GradientFunction gradient);

   Geometry::Type GetReferenceGeometry() const override { return geometry; }
   real_t Eval(const IntegrationPoint &point) const override;
   void EvalGradient(const IntegrationPoint &point,
                     Vector &result) const override;

private:
   Geometry::Type geometry;
   ValueFunction value;
   GradientFunction gradient;
};

/** @brief An intersection of level-set sign constraints on one reference
    element.

    A default-constructed domain is the whole element. The domain borrows its
    level sets, which must outlive every operation using it. */
class CutDomain
{
public:
   /// Sign condition imposed on a level set; Positive and Negative are strict.
   enum class LevelSetSign: signed char {Negative = -1, Zero = 0, Positive = 1};

   /// One borrowed level set and the sign condition imposed on it.
   struct Constraint
   {
      std::reference_wrapper<const ReferenceLevelSet> level_set;
      LevelSetSign sign;
   };

   /// Construct the domain where @a level_set is strictly positive.
   static CutDomain Positive(const ReferenceLevelSet &level_set);

   /// Construct the domain where @a level_set is strictly negative.
   static CutDomain Negative(const ReferenceLevelSet &level_set);

   /// Construct the surface where @a level_set is zero.
   static CutDomain Surface(const ReferenceLevelSet &level_set);

   /// Intersect this domain with another sign constraint and return `*this`.
   CutDomain &Add(const ReferenceLevelSet &level_set, LevelSetSign sign);

   /// Return the number of sign constraints in this domain.
   int NumConstraints() const;

   /// Return constraint @a index, throwing if the index is out of range.
   const Constraint &GetConstraint(int index) const;

   /// Return the nominal codimension: the number of zero constraints.
   /// Regular zero-dimensional intersections use counting measure.
   int Codimension() const;

   /// Validate signs, reference geometries, uniqueness, and codimension.
   /// This doesn't test whether equality gradients are nonzero and independent;
   /// generation reports such cases as DegenerateGeometry.
   CutQuadratureReport Validate(Geometry::Type geometry) const;

private:
   std::vector<Constraint> constraints;
};

/** @brief Result of cut-quadrature generation on a reference element.

    This object owns its data and retains no references to the level sets. */
struct CutQuadrature
{
   /** Relation to the reference element.

       Full: the domain covers the reference element up to a set of zero measure
             and applies only when there are no equality constraints.
       Empty: zero measure in the requested manifold dimension,
              not necessarily an empty set.
       Unknown: uninitialized/failed state, never a successful result. */
   enum class Region { Unknown, Empty, Full, Cut };
   Region region = Region::Unknown;

   /// Geometry of the reference element containing the rule.
   Geometry::Type geometry = Geometry::INVALID;

   /// Codimension of the integration domain in the reference element.
   int codimension = 0;

   /// Rule points in reference coordinates, with reference-measure weights.
   IntegrationRule rule;

   /// Intrinsic unit conormals at the rule points, stored as a
   /// dim x codimension x npoints tensor. Each column corresponds to one
   /// equality constraint, in constraint order, and points toward phi > 0.
   /// Columns need not be orthogonal. The tensor is empty for codimension zero;
   /// columns at zero-weight points are zero and must not be used for geometry.
   DenseTensor conormals;

   /// Return the number of quadrature points.
   int GetNPoints() const { return rule.GetNPoints(); }

   /// Reset all fields to their uninitialized state.
   void Clear();
};

/** @brief Interface for cut-quadrature generation on a reference element.

    A backend retains no input references or element-dependent state. */
class CutQuadratureBackend
{
public:
   virtual ~CutQuadratureBackend() = default;

   /// Return the backend name.
   virtual std::string Name() const = 0;

   /// Check whether this backend supports the supplied input types, geometry,
   /// polynomial spaces, order, constraints, and prescribed points. Success
   /// does not certify geometric regularity, convergence, or exactness.
   virtual CutQuadratureReport CheckSupport(
      Geometry::Type geometry, const CutDomain &domain, int order,
      const IntegrationRule *fixed_points = nullptr) const = 0;

   /** Generate the rule and classify the domain as empty, full, or cut.

       This method validates its inputs independently of CheckSupport.
       Nonzero-weight nodes must lie in the domain closure; surface nodes must
       lie on the interface, and weights may be signed. A prescribed point
       layout keeps its coordinates and order, using zero weights for an empty
       domain. On failure the rule is cleared and the region is Unknown. */
   virtual CutQuadratureReport Generate(
      Geometry::Type geometry, const CutDomain &domain, int order,
      IntegrationRule &rule, CutQuadrature::Region &region,
      const IntegrationRule *fixed_points = nullptr) const = 0;
};

/** Construct a compiled-in backend with its default configuration.
    Returns nullptr for an unknown or unavailable backend. Concrete backend
    constructors provide algorithm-specific configuration.

    Polynomial ALGOIM simplifies same-degree constraints with exactly equal or
    opposite normalized Bernstein coefficients. Remaining nearly proportional
    constraints at coefficient-roundoff scale return NumericalFailure. Degree
    elevation and normalization are used only for this diagnostic; constraints
    are never merged using a tolerance. */
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
    The measure ratio mu is positive, while volume retains the sign of
    Tr.Weight(), including for orientation-reversing square mappings.
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
   CutQuadrature::Region region = CutQuadrature::Region::Unknown;
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
    For codimension zero mu = sqrt(det(G)), including embedded elements; for
    square mappings this is abs(Tr.Weight()), not the signed integrator factor.
    Physical conormals are the normalized columns of J G^{-1} N.
    J and the active conormals must have full column rank; singular cases
    return DegenerateGeometry. There are no gradient evaluations at
    zero-weight nodes. On failure out is cleared, never partly usable.
    Standard exceptions from mapping evaluation return NumericalFailure.

    Mapping generally does not preserve reference polynomial exactness.
    The transformation's current integration point is restored on return. */
CutQuadratureReport MapCutQuadrature(
   ElementTransformation &transformation, const CutQuadrature &reference,
   const CutMappingOptions &options, MappedCutQuadrature &out);

/** @brief Single-element front end borrowing a backend, not a mesh or source.

    The backend must outlive this object. Each call validates inputs, invokes
    the backend, and evaluates reference conormals at active manifold nodes.
    There are no full/empty shortcuts that replace a prescribed node layout.
    Failures clear out. Every referenced level set must be initialized when
    applicable and defined on the supplied reference geometry.

    Concurrent calls require distinct outputs, transformations, and any
    non-reentrant level-set/finite-element evaluation state. Const methods do
    not by themselves guarantee that user callbacks or MFEM elements are safe
    to share. The generator retains no per-element state. */
class CutQuadratureGenerator
{
public:
   explicit CutQuadratureGenerator(const CutQuadratureBackend &backend);
   const CutQuadratureBackend &GetBackend() const { return backend; }

   /** Generate a reference rule of nominal MFEM integration @a order; this is
       not an exactness guarantee on curved domains. Non-null @a fixed_points
       supplies borrowed coordinates for full-dimensional weight fitting; its
       weights are ignored, and backends may report this mode as unsupported. */
   CutQuadratureReport Generate(
      Geometry::Type geometry,
      const CutDomain &domain, int order, CutQuadrature &out,
      const IntegrationRule *fixed_points = nullptr) const;

   /** Convenience composition of reference generation and MapCutQuadrature.
       Inputs are already pulled back/projected for this element. Geometry is
       taken from the transformation. Successful reports preserve the
       reference-generation diagnostics. */
   CutQuadratureReport Generate(
      ElementTransformation &transformation,
      const CutDomain &domain, int order, const CutMappingOptions &mapping,
      MappedCutQuadrature &out,
      const IntegrationRule *fixed_points = nullptr) const;

private:
   const CutQuadratureBackend &backend;
};

}

#endif
