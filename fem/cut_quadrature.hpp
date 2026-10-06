// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC.
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MFEM_CUT_QUADRATURE_HPP
#define MFEM_CUT_QUADRATURE_HPP

/** @file
    @brief Element-local quadrature for level-set regions and interfaces.

    Extractors extract a polynomial in reference coordinates; constructors produce
    reference rules for its selected sign region and/or zero set. Physical
    metric factors are applied separately by MapReferenceVolumeRule() and
    MapReferenceInterfaceRule(). Applications may retain reference rules and
    reuse them while the extractor revision and request remain unchanged. */

#include "../config/config.hpp"
#include "../general/array.hpp"
#include "../linalg/densemat.hpp"
#include "../linalg/vector.hpp"
#include "geom.hpp"
#include "intrules.hpp"

#include <atomic>
#include <cstdint>
#include <memory>

namespace mfem
{

class Coefficient;
class ElementTransformation;
class GridFunction;

/// Sign region of the level-set polynomial selected for volume integration.
enum class CutRegion { Negative, Positive };

/// Requested measures; combine Volume and Interface with operator|().
enum class CutMeasure : unsigned
{
   Volume = 1u,    ///< Integrate the selected sign region.
   Interface = 2u ///< Integrate the zero level set.
};

/** Combine requested measures, e.g. CutMeasure::Volume | CutMeasure::Interface.
    @param lhs Left-hand side operand (Volume in the example).
    @param rhs Right-hand side operand (Interface in the example). */
inline CutMeasure operator|(CutMeasure lhs, CutMeasure rhs)
{
   return static_cast<CutMeasure>(static_cast<unsigned>(lhs) |
                                  static_cast<unsigned>(rhs));
}

/** Intersect measure masks; a zero underlying value means no common measure.
    @param lhs Left-hand side operand of operator&.
    @param rhs Right-hand side operand of operator&. */
inline CutMeasure operator&(CutMeasure lhs, CutMeasure rhs)
{
   return static_cast<CutMeasure>(static_cast<unsigned>(lhs) &
                                  static_cast<unsigned>(rhs));
}

/// Whether generated weights may have either sign or must be nonnegative.
enum class QuadratureWeightPolicy { Unconstrained, Nonnegative };
/// Requested execution location; support is determined by the backend.
enum class CutExecutionMode { Host, Device };
/// Representation of the element-local level-set polynomial.
enum class PolynomialBasis { BernsteinTensor, BernsteinSimplex };

/** Classification relative to the requested sign region.
    Cut may be a conservative classification based on polynomial bounds.
    Full or Empty volume does not necessarily imply an empty boundary interface. */
enum class CutCellClass
{
   Unclassified, ///< No classification has been established (e.g. before validation).
   Empty,        ///< The selected sign region has no volume in the element.
   Full,         ///< The selected sign region fills the element up to its zero set.
   Cut,          ///< The element may contain both sign regions; bounds are inconclusive.
   Degenerate    ///< The level set does not define a regular cut (e.g. identically zero).
};

/// Outcome of extraction, request validation, or rule generation.
enum class CutQuadratureStatus
{
   Success,                    ///< The operation completed successfully.
   InvalidRequest,             ///< Invalid order, measure mask, or enum value.
   UnsupportedGeometry,        ///< The backend does not support this geometry.
   UnsupportedSourceBasis,     ///< The extractor cannot extract this source.
   UnsupportedPolynomialBasis, ///< The backend cannot use this representation.
   UnsupportedOrder,           ///< Target quadrature order is unsupported.
   UnsupportedExecutionMode,   ///< Requested host/device mode is unsupported.
   UnsupportedWeightPolicy,    ///< Requested weight policy is unsupported.
   InvalidBatch,               ///< Inconsistent batch sizes or metadata.
   HeterogeneousBatch,         ///< Successful entries have different descriptors.
   ExecutionFailure,           ///< Batch execution failed as a whole.
   InvalidLevelSet,            ///< Invalid polynomial data or source element.
   DegenerateVolume,           ///< Degenerate sign-region definition.
   DegenerateInterface,        ///< Singular or otherwise degenerate zero set.
   WeightConstraintInfeasible, ///< The requested weight constraint cannot be met.
   GenerationFailure          ///< Rule generation failed or produced invalid data.
};

/** Polynomial defining the cut in reference-element coordinates.
    For BernsteinTensor in dimension d, order p is the degree in each coordinate
    and coefficients has (p+1)^d entries. Indices are lexicographic with the first
    coordinate varying fastest: i + (p+1)*j in 2D and
    i + (p+1)*(j + (p+1)*k) in 3D. Backend support for bases and geometries must
    be queried through CutQuadratureCapabilities. */
struct ElementLevelSet
{
   /// Reference-element geometry.
   Geometry::Type geometry = Geometry::INVALID;
   /// Polynomial degree, distinct from the requested quadrature order.
   int order = -1;
   /// Basis in which coefficients are expressed.
   PolynomialBasis basis = PolynomialBasis::BernsteinTensor;
   /// Coefficient data in the ordering specified above.
   Vector coefficients;
};

/// Polynomial metadata used to compare entries and query backend support.
struct ElementLevelSetDescriptor
{
   // No member initializers: mfem::Array requires a trivial element type.
   Geometry::Type geometry;
   PolynomialBasis basis;
   int order;

   bool operator==(const ElementLevelSetDescriptor &other) const;
   bool operator!=(const ElementLevelSetDescriptor &other) const;
};

/** Input polynomials for a homogeneous batch.
    The coefficient matrix width is the batch size; both metadata arrays must
    have that size. Successfully extracted entries must match descriptor.
    Algoim also accepts extraction failures UnsupportedSourceBasis and
    InvalidLevelSet, propagating them to the corresponding output entries. */
struct ElementLevelSetBatch
{
   /// Shared geometry, basis, and polynomial degree.
   ElementLevelSetDescriptor descriptor =
   { Geometry::INVALID, PolynomialBasis::BernsteinTensor, -1 };
   /// One element per column (coefficient SoA), fixed by descriptor.
   DenseMatrix coefficients;
   /// Actual descriptor for each entry; failed extractions need not match.
   Array<ElementLevelSetDescriptor> element_descriptors;
   /// Extractor extraction outcome for each entry.
   Array<CutQuadratureStatus> extraction_status;
};

/// Parameters shared by all elements in a generation call.
struct CutQuadratureRequest
{
   /** Target integration order, not the level-set polynomial degree.
       Accuracy on curved cuts depends on the backend; this is not a guarantee
       of exactness for arbitrary curved interfaces. */
   int order = 4;
   /// Sign region for volume rules; interface normals retain their orientation.
   CutRegion region = CutRegion::Negative;
   /// Nonempty mask of Volume, Interface, or both.
   CutMeasure measures = CutMeasure::Volume;
   /// Constraint on the sign of generated weights.
   QuadratureWeightPolicy weight_policy =
      QuadratureWeightPolicy::Unconstrained;
   /// Host or device execution, subject to backend capabilities.
   CutExecutionMode execution = CutExecutionMode::Host;
   /// Request unit reference normals for interface points (needed for mapping).
   bool compute_reference_normals = false;

   /// Compare every request field, including the normal and execution options.
   bool operator==(const CutQuadratureRequest &other) const;
   bool operator!=(const CutQuadratureRequest &other) const;
};

/// Backend support advertised independently of any particular polynomial data.
struct CutQuadratureCapabilities
{
   /// Supported reference geometries.
   Array<Geometry::Type> geometries;
   /// Supported polynomial representations.
   Array<PolynomialBasis> bases;
   /// Bounds are in CutQuadratureRequest target-order units.
   int min_order = 0;
   /// Inclusive upper target-order bound; -1 means no nonnegative order fits.
   int max_order = -1;
   bool volume = false;         ///< Can generate volume rules.
   bool interface = false;      ///< Can generate interface rules.
   bool negative_phase = false; ///< Can select the negative region.
   bool positive_phase = false; ///< Can select the positive region.
   /// Advertised support for both phases; a request still selects one region.
   bool both_phases = false;
   bool unconstrained_weights = false; ///< Supports unrestricted weight signs.
   bool nonnegative_weights = false;   ///< Supports nonnegative weights.
   bool normals = false;              ///< Can return unit reference normals.
   bool host_scalar = false;          ///< Supports scalar host calls.
   bool host_batch = false;           ///< Supports batch host calls.
   bool device_batch = false;         ///< Supports batch device calls.

   /** Check advertised support for the request and polynomial metadata.
       This does not validate coefficient data or guarantee successful generation.
       @param request Quadrature order, measures, region, weight policy, execution
                      mode, and normal requirements to check.
       @param level_set Reference geometry, polynomial basis, and degree describing
                        the input level set.
       @param batch Select batch support instead of scalar support. */
   bool Supports(const CutQuadratureRequest &request,
                 const ElementLevelSetDescriptor &level_set,
                 bool batch = false) const;
};

/** Interface points and surface weights in reference coordinates.
    Unit normals point toward increasing level-set values, independently of
    CutRegion. Physical surface mapping requires these normals. */
struct ReferenceInterfaceRule
{
   IntegrationRule rule;
   /** Unit normals to the zero level set in reference-element coordinates.
       The matrix has dim rows and nq columns, where dim is the reference
       element dimension and nq = rule.GetNPoints(). Column q contains the
       normal at rule.IntPoint(q); entry (d, q) is its coordinate component d.
       For example, in 2D column q contains (n_x, n_y).

       SoA means "structure of arrays": components occupy separate matrix rows,
       while each column groups the components for one quadrature point. MFEM's
       DenseMatrix stores columns contiguously in memory.

       Normals are normalized level-set gradients and point toward increasing
       level-set values, independently of the requested CutRegion. They are
       populated when CutQuadratureRequest::compute_reference_normals is true.
       Set that option when using MapReferenceInterfaceRule() or
       CutQuadratureIntegrator::IntegrateInterface(), since physical surface
       weights require these normals even if physical normals are not requested. */
   DenseMatrix reference_normals;
};

/** Scalar generation result, with no physical metric factors in its weights.
    Consume rules only when status is Success. Unrequested measures may be empty;
    an empty rule can also be a valid result for a region or interface. */
struct ReferenceCutQuadrature
{
   CutQuadratureStatus status = CutQuadratureStatus::Success;
   CutCellClass classification = CutCellClass::Unclassified;
   IntegrationRule volume;
   ReferenceInterfaceRule interface;
};

/** Concatenated reference rules for a homogeneous batch of n elements.
    Each instance stores either volume rules or interface rules. Coordinates,
    weights, and optional normals use the same global point index q, with rules
    concatenated in input batch order. Coordinates and weights are in reference
    space; weights contain no physical element or surface metric factors.

    After successful batch execution, offsets has n+1 nondecreasing entries,
    offsets[0] = 0, and offsets[n] = weights.Size() = points.Width(). Entry e
    owns points in [offsets[e], offsets[e+1]), so its point count is
    offsets[e+1] - offsets[e]. Local point j has global index offsets[e] + j.
    For example, offsets = {0, 2, 2, 5} describes three entries containing
    two, zero, and three points, respectively.

    In the Algoim backend, failed entries, empty rules, and unrequested measures
    contribute no points. Equal adjacent offsets alone do not indicate failure:
    inspect BatchedReferenceCutQuadrature::status for each entry. A batch call
    returning Success may still contain failed entries; a call-level failure
    may leave the packed arrays empty rather than providing n+1 offsets. */
struct PackedReferenceRules
{
   /** Reference coordinates in a dim by total_nq matrix.
       dim is the reference-element dimension and total_nq = weights.Size().
       Entry (d, q) is coordinate component d of global point q (x, y, or z).
       Each column represents one point. DenseMatrix uses column-major storage,
       so a point's components are contiguous; coordinate rows are strided. */
   DenseMatrix points;
   /// Reference volume or surface weight at global point q is weights(q).
   Vector weights;
   /** Unit reference normals in a dim by total_nq matrix when requested for
       interface rules; otherwise a 0 by 0 matrix in the Algoim backend.
       Column q corresponds to points column q and points toward increasing
       level-set values, independently of CutRegion. Volume rules have no normals. */
   DenseMatrix normals;
   /// Prefix sums delimiting rules by batch entry, not by mesh element ID.
   Array<int> offsets;
};

/// Per-element outcomes and packed rules in the input batch order.
struct BatchedReferenceCutQuadrature
{
   /// Check each entry even when the batch call returns Success.
   Array<CutQuadratureStatus> status;
   Array<CutCellClass> classification;
   PackedReferenceRules volume;
   PackedReferenceRules interface;
};

/// Backend-owned scratch storage; use a separate instance for each concurrent call.
class CutQuadratureWorkspace
{
public:
   virtual ~CutQuadratureWorkspace() = default;
};

/** Backend-neutral cut-rule constructor.

    Const constructors are safe to share between threads. Workspaces are not:
    every concurrent caller must use a separate workspace. */
class CutQuadratureConstructor
{
public:
   virtual ~CutQuadratureConstructor() = default;
   /// Return this constructor's supported geometries, requests, and execution modes.
   virtual const CutQuadratureCapabilities &Capabilities() const = 0;
   /// Allocate scratch storage compatible with this backend.
   virtual std::unique_ptr<CutQuadratureWorkspace> CreateWorkspace() const = 0;

   /** Generate reference rules for one polynomial using backend workspace.
       The returned status is also stored in result.status. A failed result must
       not be integrated, even if it contains partially generated points. */
   virtual CutQuadratureStatus GenerateReference(
      const ElementLevelSet &level_set,
      const CutQuadratureRequest &request,
      ReferenceCutQuadrature &result,
      CutQuadratureWorkspace &workspace) const = 0;

   /** Generate packed rules for a batch using backend workspace.
       The return value describes the batch operation; Success does not imply
       that all result.status entries succeeded. Check each entry before use. */
   virtual CutQuadratureStatus GenerateReferenceBatch(
      const ElementLevelSetBatch &level_sets,
      const CutQuadratureRequest &request,
      BatchedReferenceCutQuadrature &result,
      CutQuadratureWorkspace &workspace) const = 0;
};

/// Application-managed version token for changes affecting an extracted polynomial.
using LevelSetRevision = std::uint64_t;

/** Extract an element-local polynomial from an application field.

    Extractors and their wrapped read-only sources may be shared by concurrent
    callers. The source must not be mutated concurrently. A caller that changes
    source values must also change Revision(); forgetting to do so can silently
    reuse stale application-owned rules. */
class ElementLevelSetExtractor
{
public:
   ElementLevelSetExtractor();
   virtual ~ElementLevelSetExtractor() = default;
   ElementLevelSetExtractor(const ElementLevelSetExtractor &) = delete;
   ElementLevelSetExtractor &operator=(const ElementLevelSetExtractor &) = delete;
   ElementLevelSetExtractor(ElementLevelSetExtractor &&) = delete;
   ElementLevelSetExtractor &operator=(ElementLevelSetExtractor &&) = delete;

   /// Stable identity of this extractor instance for retained-rule matching.
   std::uint64_t Id() const { return id_; }

   /** Extract element's polynomial using its matching transformation Tr.
       result is usable only on Success. Concurrent calls require separate
       transformations because sampling may change their current point. */
   virtual CutQuadratureStatus GetElementLevelSet(
      int element, ElementTransformation &Tr, ElementLevelSet &result) const = 0;
   /// Current source version; update it whenever extraction results may change.
   virtual LevelSetRevision Revision() const = 0;

private:
   std::uint64_t id_;
};

/** Extract tensor Bernstein polynomials from a scalar GridFunction.
    Supports scalar VALUE-mapped TensorBasisElement sources on squares and cubes
    with degree at least one. The source is sampled at tensor H1 nodes and
    converted to Bernstein coefficients. The GridFunction is borrowed and must
    outlive the extractor; source updates require a revision change. */
class GridFunctionLevelSetExtractor : public ElementLevelSetExtractor
{
public:
   explicit GridFunctionLevelSetExtractor(const GridFunction &level_set,
                                          LevelSetRevision revision = 0);
   CutQuadratureStatus GetElementLevelSet(
      int element, ElementTransformation &Tr,
      ElementLevelSet &result) const override;
   LevelSetRevision Revision() const override { return revision_.load(); }
   /// Publish a new source version; this does not modify the GridFunction.
   void SetRevision(LevelSetRevision revision) { revision_.store(revision); }
   /// Advance and return the version token after a source change.
   LevelSetRevision IncrementRevision() { return ++revision_; }

private:
   const GridFunction *level_set_;
   std::atomic<LevelSetRevision> revision_;
};

/** Element-local tensor-H1 interpolation of a general Coefficient.
    Supports squares and cubes with approximation_order at least one, returning
    tensor Bernstein coefficients for the interpolant. This approximates the
    source rather than preserving an arbitrary nonpolynomial zero set exactly.
    The Coefficient is borrowed and must outlive the extractor. Changes to source
    values, time, or geometry affecting sampling require a revision change.
    Concurrent sampling also requires a thread-safe Coefficient::Eval(). */
class CoefficientLevelSetExtractor : public ElementLevelSetExtractor
{
public:
   CoefficientLevelSetExtractor(Coefficient &level_set, int approximation_order,
                                LevelSetRevision revision = 0);
   /// Sample through Tr; the element argument is unused by this extractor.
   CutQuadratureStatus GetElementLevelSet(
      int element, ElementTransformation &Tr,
      ElementLevelSet &result) const override;
   LevelSetRevision Revision() const override { return revision_.load(); }
   /// Publish a new source version; this does not modify the Coefficient.
   void SetRevision(LevelSetRevision revision) { revision_.store(revision); }
   /// Advance and return the version token after a source change.
   LevelSetRevision IncrementRevision() { return ++revision_; }
   /// Degree of the tensor H1 interpolant used during extraction.
   int ApproximationOrder() const { return approximation_order_; }

private:
   Coefficient *level_set_;
   int approximation_order_;
   std::atomic<LevelSetRevision> revision_;
};

/** Application-owned reference rule and the key describing its source.
    The caller fills the key and result; this object does not generate rules or
    automatically track source changes. Physical metrics are applied at use time. */
struct RetainedCutQuadrature
{
   std::uint64_t extractor_id = 0;
   int element = -1;
   LevelSetRevision revision = 0;
   CutQuadratureRequest request;
   ReferenceCutQuadrature result;

   /** Match extractor identity, revision, element, and all request fields.
       This checks only the key, not result.status or the rule contents. */
   bool IsValid(const ElementLevelSetExtractor &extractor, int element_id,
                const CutQuadratureRequest &requested) const;
};

/// Application-owned batch rules and their extractor, revision, and request key.
struct RetainedBatchedCutQuadrature
{
   std::uint64_t extractor_id = 0;
   Array<int> elements;
   LevelSetRevision revision = 0;
   CutQuadratureRequest request;
   BatchedReferenceCutQuadrature result;

   /** Match the key including the exact ordered list of element IDs.
       This does not inspect per-element result.status values. */
   bool IsValid(const ElementLevelSetExtractor &extractor,
                const Array<int> &element_ids,
                const CutQuadratureRequest &requested) const;
};

/** Multiply reference volume weights by Tr.Weight() at each point.
    Point coordinates remain in the reference element; only weights are mapped.
    The input rule is unchanged when mapped is a distinct output object.
    Tr's current integration point is changed during mapping. */
void MapReferenceVolumeRule(ElementTransformation &Tr,
                            const IntegrationRule &reference,
                            IntegrationRule &mapped);

/** Apply the physical surface metric to reference interface weights.
    reference.reference_normals must have Tr.GetDimension() rows and one column
    per point, even when physical_normals is nullptr. Weights are multiplied by
    Tr.Weight() * ||J^{-T} n_ref||; coordinates remain in the reference element.
    If requested, physical_normals receives normalized transformed normals as
    Tr.GetSpaceDim() by nq columns. Tr's current integration point is changed. */
void MapReferenceInterfaceRule(ElementTransformation &Tr,
                               const ReferenceInterfaceRule &reference,
                               IntegrationRule &mapped,
                               DenseMatrix *physical_normals = nullptr);

/// Integrate a Coefficient using current physical metrics and successful reference rules.
class CutQuadratureIntegrator
{
public:
   /// Sum coefficient values times reference weights and Tr.Weight().
   static real_t IntegrateVolume(Coefficient &coefficient,
                                 ElementTransformation &Tr,
                                 const ReferenceCutQuadrature &quadrature);
   /** Integrate using MapReferenceInterfaceRule(); reference normals are required.
       Optionally return physical unit normals, one column per interface point. */
   static real_t IntegrateInterface(Coefficient &coefficient,
                                    ElementTransformation &Tr,
                                    const ReferenceCutQuadrature &quadrature,
                                    DenseMatrix *physical_normals = nullptr);
};

#ifdef MFEM_USE_ALGOIM
/** Host-only Algoim backend for tensor Bernstein polynomials on squares and cubes.
    Supports scalar calls and homogeneous batches, either sign region, volume
    and interface rules, optional reference normals, and target orders 0--19.
    Batches are processed element by element on the host. Both weight policies
    are advertised. The target order p selects (p+2)/2 Algoim quadrature nodes
    per one-dimensional rule (integer division).

    An identically zero polynomial returns DegenerateVolume. Classification uses
    Bernstein coefficient bounds, so Cut is conservative. Interface normals are
    normalized gradients of the original polynomial, even for the positive phase.
    Available only when MFEM is configured with MFEM_USE_ALGOIM. */
class AlgoimCutQuadratureConstructor : public CutQuadratureConstructor
{
public:
   AlgoimCutQuadratureConstructor();
   const CutQuadratureCapabilities &Capabilities() const override
   { return capabilities_; }
   std::unique_ptr<CutQuadratureWorkspace> CreateWorkspace() const override;
   CutQuadratureStatus GenerateReference(
      const ElementLevelSet &level_set, const CutQuadratureRequest &request,
      ReferenceCutQuadrature &result,
      CutQuadratureWorkspace &workspace) const override;
   CutQuadratureStatus GenerateReferenceBatch(
      const ElementLevelSetBatch &level_sets,
      const CutQuadratureRequest &request,
      BatchedReferenceCutQuadrature &result,
      CutQuadratureWorkspace &workspace) const override;

private:
   CutQuadratureCapabilities capabilities_;
};
#endif

} // namespace mfem

#endif // MFEM_CUT_QUADRATURE_HPP
