
/** @file
    Single-element cut quadrature.

    Level sets describe one reference element. Backends construct reference
    rules; the generator supplies reference geometry data; a separate mapping
    operation supplies physical geometric data. There are no batch methods,
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
    unsupported polynomial spaces, and INTEGRAL-mapped fields fail
    verification.
    A positive basis on a pyramid is not automatically a Cartesian polynomial.

    Default construction leaves the object uninitialized. GetReferenceGeometry
    returns INVALID while uninitialized. Evaluation and GetFE require
    successful initialization. */
class PolynomialLevelSet : public ReferenceLevelSet
{
public:
   PolynomialLevelSet() = default;

   void Set(const FiniteElement &element, const Vector &values);

   /// Extract local scalar dofs, including the local dof transformation.
   /// The GridFunction and its finite-element space must remain valid and
   /// unchanged for the lifetime of the borrowed finite element.
   void Set(const GridFunction &function, int element);

   /// Project coefficient into the supplied reference polynomial space.
   /// The projection is local; it does not construct a mesh-wide field.
   void Project(Coefficient &coefficient,
                ElementTransformation &transformation,
                const FiniteElement &element);

   bool IsInitialized() const { return element != nullptr; }
   const FiniteElement &GetFE() const;
   const Vector &GetDofs() const { return dofs; }

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

/// Result of a cut-quadrature related computation.
struct CutQuadratureReport
{
   /// Computation outcome, independent of whether the domain is empty or full.
   /// A valid empty domain is Success.
   /// Unsupported orders return Unsupported rather than being silently reduced.
   /// Singular mapping geometry -> DegenerateGeometry;
   /// moment-system rank, convergence, tolerance failures -> NumericalFailure.
   enum class Status
   {
      Success,
      InvalidInput,
      Unsupported,
      DegenerateGeometry,
      NumericalFailure
   };
   Status status = Status::Success;

   /// Optional detail about the outcome.
   std::string message;

   /// Total degree of the fitted reference polynomial space, or -1 when not
   /// applicable.
   int moment_degree = -1;

   /// Relative moment residual
   /// ||A weights - moments||_inf / max(1, ||moments||_inf), or -1 when
   /// unavailable. This measures the fit, not the accuracy of the supplied
   /// moments, level-set approximation, or integration of arbitrary functions.
   real_t moment_residual = -1;

   bool Succeeded() const { return status == Status::Success; }
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

/// Result of cut-quadrature generation on a reference element.
/// This object owns its data and retains no references to the level sets.
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

/// Interface for cut-quadrature generation on a reference element.
/// A backend retains no input references or element-dependent state.
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

   /// Generate the reference rule and region directly in @a out. This method
   /// validates its inputs independently of CheckSupport. Nonzero-weight nodes
   /// must lie in the domain closure; surface nodes must lie on the interface,
   /// and weights may be signed. A prescribed point layout keeps its
   /// coordinates and order, using zero weights for an empty domain.
   /// On failure @a out is cleared.
   virtual CutQuadratureReport Generate(
      Geometry::Type geometry, const CutDomain &domain, int order,
      CutQuadrature &out,
      const IntegrationRule *fixed_points = nullptr) const = 0;
};

/// Construct a compiled-in backend by name.
/// Returns nullptr if the name is unknown or the backend is unavailable.
std::unique_ptr<CutQuadratureBackend>
CreateCutQuadratureBackend(const std::string &name);

/// Physical geometric data evaluated at cut-quadrature points.
struct CutQuadratureMapping
{
   /// Optional data computed by MapCutQuadrature and stored in this struct.
   enum Data : int
   {
      NONE = 0,
      PHYSICAL_POINTS = 1 << 0,
      PHYSICAL_CONORMALS = 1 << 1,
      MEASURE_SCALING = 1 << 2
   };

   /// Physical point coordinates, stored as sdim x npoints.
   DenseMatrix physical_points;

   /// Physical unit conormals, stored as sdim x codimension x npoints.
   DenseTensor physical_conormals;

   /// Ratio mu of physical to reference cut-manifold measure at each point.
   /// For a full-dimensional domain mu is the element-volume scaling.
   /// For a lower-dimensional cut it is the tangential scaling and depends on
   /// the reference conormals; it is generally not det(J).
   Vector measure_scaling;

   /// Clear all computed data.
   void Clear();
};

/** @brief Compute physical geometric data without modifying the reference rule.

    CutQuadrature's geometry must match transformation.GetGeometryType().
    Note that a moved mesh may require reprojection and rule regeneration first.

    For J = dT/dxi, G = J^T J, and reference conormal matrix N:
       mu = sqrt(det(G)) * sqrt(det(N^T G^{-1} N) / det(N^T N)).
    For codimension zero mu = sqrt(det(G)), including embedded elements; for
    square mappings this is abs(det(J)).
    Physical conormals are the normalized columns of J G^{-1} N.
    J and the active conormals must have full column rank; singular cases
    return DegenerateGeometry. Conormals and scaling at zero-weight nodes are
    zero and are not geometric data. Physical positions may still be mapped
    there. On failure out is cleared, never partly usable.
    Standard exceptions from mapping evaluation return NumericalFailure. */
CutQuadratureReport MapCutQuadrature(ElementTransformation &transformation,
                                     const CutQuadrature &reference,
                                     int requested_data,
                                     CutQuadratureMapping &out);

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

   /// Generate a reference rule of nominal MFEM integration @a order; this is
   /// not an exactness guarantee on curved domains.
   /// Non-null @a fixed_points supplies borrowed coordinates for
   /// full-dimensional weight fitting; its weights are ignored, and backends
   /// may report this mode as unsupported.
   CutQuadratureReport Generate(
      Geometry::Type geometry,
      const CutDomain &domain, int order, CutQuadrature &out,
      const IntegrationRule *fixed_points = nullptr) const;

private:
   const CutQuadratureBackend &backend;
};

}

#endif
