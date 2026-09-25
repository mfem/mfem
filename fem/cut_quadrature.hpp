// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC.
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MFEM_CUT_QUADRATURE_HPP
#define MFEM_CUT_QUADRATURE_HPP

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

enum class CutRegion { Negative, Positive };

enum class CutMeasure : unsigned
{
   Volume = 1u,
   Interface = 2u
};

inline CutMeasure operator|(CutMeasure lhs, CutMeasure rhs)
{
   return static_cast<CutMeasure>(static_cast<unsigned>(lhs) |
                                  static_cast<unsigned>(rhs));
}

inline CutMeasure operator&(CutMeasure lhs, CutMeasure rhs)
{
   return static_cast<CutMeasure>(static_cast<unsigned>(lhs) &
                                  static_cast<unsigned>(rhs));
}

enum class QuadratureWeightPolicy { Unconstrained, Nonnegative };
enum class CutExecutionMode { Host, Device };
enum class PolynomialBasis { BernsteinTensor, BernsteinSimplex };

enum class CutCellClass { Unclassified, Empty, Full, Cut, Degenerate };

enum class CutQuadratureStatus
{
   Success,
   InvalidRequest,
   UnsupportedGeometry,
   UnsupportedSourceBasis,
   UnsupportedPolynomialBasis,
   UnsupportedOrder,
   UnsupportedExecutionMode,
   UnsupportedWeightPolicy,
   InvalidBatch,
   HeterogeneousBatch,
   ExecutionFailure,
   InvalidLevelSet,
   DegenerateVolume,
   DegenerateInterface,
   WeightConstraintInfeasible,
   GenerationFailure
};

struct ElementLevelSet
{
   Geometry::Type geometry = Geometry::INVALID;
   int order = -1;
   PolynomialBasis basis = PolynomialBasis::BernsteinTensor;
   Vector coefficients;
};

struct ElementLevelSetDescriptor
{
   // No member initializers: mfem::Array requires a trivial element type.
   Geometry::Type geometry;
   PolynomialBasis basis;
   int order;

   bool operator==(const ElementLevelSetDescriptor &other) const;
   bool operator!=(const ElementLevelSetDescriptor &other) const;
};

struct ElementLevelSetBatch
{
   ElementLevelSetDescriptor descriptor =
   { Geometry::INVALID, PolynomialBasis::BernsteinTensor, -1 };
   /// One element per column (coefficient SoA), fixed by descriptor.
   DenseMatrix coefficients;
   Array<ElementLevelSetDescriptor> element_descriptors;
   Array<CutQuadratureStatus> extraction_status;
};

struct CutQuadratureRequest
{
   int order = 4;
   CutRegion region = CutRegion::Negative;
   CutMeasure measures = CutMeasure::Volume;
   QuadratureWeightPolicy weight_policy =
      QuadratureWeightPolicy::Unconstrained;
   CutExecutionMode execution = CutExecutionMode::Host;
   bool compute_reference_normals = false;

   bool operator==(const CutQuadratureRequest &other) const;
   bool operator!=(const CutQuadratureRequest &other) const;
};

struct CutQuadratureCapabilities
{
   Array<Geometry::Type> geometries;
   Array<PolynomialBasis> bases;
   /// Bounds are in CutQuadratureRequest target-order units.
   int min_order = 0;
   int max_order = -1;
   bool volume = false;
   bool interface = false;
   bool negative_phase = false;
   bool positive_phase = false;
   bool both_phases = false;
   bool unconstrained_weights = false;
   bool nonnegative_weights = false;
   bool normals = false;
   bool host_scalar = false;
   bool host_batch = false;
   bool device_batch = false;

   bool Supports(const CutQuadratureRequest &request,
                 const ElementLevelSetDescriptor &level_set,
                 bool batch = false) const;
};

struct ReferenceInterfaceRule
{
   IntegrationRule rule;
   /// Reference normals are stored as dim by nq (SoA).
   DenseMatrix reference_normals;
};

struct ReferenceCutQuadrature
{
   CutQuadratureStatus status = CutQuadratureStatus::Success;
   CutCellClass classification = CutCellClass::Unclassified;
   IntegrationRule volume;
   ReferenceInterfaceRule interface;
};

struct PackedReferenceRules
{
   /// Points and optional normals are dim by total_nq (SoA).
   DenseMatrix points;
   Vector weights;
   DenseMatrix normals;
   Array<int> offsets;
};

struct BatchedReferenceCutQuadrature
{
   Array<CutQuadratureStatus> status;
   Array<CutCellClass> classification;
   PackedReferenceRules volume;
   PackedReferenceRules interface;
};

class CutQuadratureWorkspace
{
public:
   virtual ~CutQuadratureWorkspace() = default;
};

/** Backend-neutral cut-rule generator.

    Const generators are safe to share between threads. Workspaces are not:
    every concurrent caller must use a separate workspace. */
class CutQuadratureGenerator
{
public:
   virtual ~CutQuadratureGenerator() = default;
   virtual const CutQuadratureCapabilities &Capabilities() const = 0;
   virtual std::unique_ptr<CutQuadratureWorkspace> CreateWorkspace() const = 0;

   virtual CutQuadratureStatus GenerateReference(
      const ElementLevelSet &level_set,
      const CutQuadratureRequest &request,
      ReferenceCutQuadrature &result,
      CutQuadratureWorkspace &workspace) const = 0;

   virtual CutQuadratureStatus GenerateReferenceBatch(
      const ElementLevelSetBatch &level_sets,
      const CutQuadratureRequest &request,
      BatchedReferenceCutQuadrature &result,
      CutQuadratureWorkspace &workspace) const = 0;
};

using LevelSetRevision = std::uint64_t;

/** Extract an element-local polynomial from an application field.

    Providers and their wrapped read-only sources may be shared by concurrent
    callers. The source must not be mutated concurrently. A caller that changes
    source values must also change Revision(); forgetting to do so can silently
    reuse stale application-owned rules. */
class ElementLevelSetProvider
{
public:
   ElementLevelSetProvider();
   virtual ~ElementLevelSetProvider() = default;
   ElementLevelSetProvider(const ElementLevelSetProvider &) = delete;
   ElementLevelSetProvider &operator=(const ElementLevelSetProvider &) = delete;
   ElementLevelSetProvider(ElementLevelSetProvider &&) = delete;
   ElementLevelSetProvider &operator=(ElementLevelSetProvider &&) = delete;

   std::uint64_t Id() const { return id_; }

   virtual CutQuadratureStatus GetElementLevelSet(
      int element, ElementTransformation &Tr, ElementLevelSet &result) const = 0;
   virtual LevelSetRevision Revision() const = 0;

private:
   std::uint64_t id_;
};

class GridFunctionLevelSetProvider : public ElementLevelSetProvider
{
public:
   explicit GridFunctionLevelSetProvider(const GridFunction &level_set,
                                         LevelSetRevision revision = 0);
   CutQuadratureStatus GetElementLevelSet(
      int element, ElementTransformation &Tr,
      ElementLevelSet &result) const override;
   LevelSetRevision Revision() const override { return revision_.load(); }
   void SetRevision(LevelSetRevision revision) { revision_.store(revision); }
   LevelSetRevision IncrementRevision() { return ++revision_; }

private:
   const GridFunction *level_set_;
   std::atomic<LevelSetRevision> revision_;
};

/** Element-local tensor-H1 interpolation of a general Coefficient. */
class CoefficientLevelSetProvider : public ElementLevelSetProvider
{
public:
   CoefficientLevelSetProvider(Coefficient &level_set, int approximation_order,
                               LevelSetRevision revision = 0);
   CutQuadratureStatus GetElementLevelSet(
      int element, ElementTransformation &Tr,
      ElementLevelSet &result) const override;
   LevelSetRevision Revision() const override { return revision_.load(); }
   void SetRevision(LevelSetRevision revision) { revision_.store(revision); }
   LevelSetRevision IncrementRevision() { return ++revision_; }
   int ApproximationOrder() const { return approximation_order_; }

private:
   Coefficient *level_set_;
   int approximation_order_;
   std::atomic<LevelSetRevision> revision_;
};

struct RetainedCutQuadrature
{
   std::uint64_t provider_id = 0;
   int element = -1;
   LevelSetRevision revision = 0;
   CutQuadratureRequest request;
   ReferenceCutQuadrature result;

   bool IsValid(const ElementLevelSetProvider &provider, int element_id,
                const CutQuadratureRequest &requested) const;
};

struct RetainedBatchedCutQuadrature
{
   std::uint64_t provider_id = 0;
   Array<int> elements;
   LevelSetRevision revision = 0;
   CutQuadratureRequest request;
   BatchedReferenceCutQuadrature result;

   bool IsValid(const ElementLevelSetProvider &provider,
                const Array<int> &element_ids,
                const CutQuadratureRequest &requested) const;
};

/** Map reference weights using current element metrics without modifying the
    retained reference rule. */
void MapReferenceVolumeRule(ElementTransformation &Tr,
                            const IntegrationRule &reference,
                            IntegrationRule &mapped);

void MapReferenceInterfaceRule(ElementTransformation &Tr,
                               const ReferenceInterfaceRule &reference,
                               IntegrationRule &mapped,
                               DenseMatrix *physical_normals = nullptr);

/** Small consumer illustrating the reference-weight convention. */
class CutQuadratureIntegrator
{
public:
   static real_t IntegrateVolume(Coefficient &coefficient,
                                 ElementTransformation &Tr,
                                 const ReferenceCutQuadrature &quadrature);
   static real_t IntegrateInterface(Coefficient &coefficient,
                                    ElementTransformation &Tr,
                                    const ReferenceCutQuadrature &quadrature,
                                    DenseMatrix *physical_normals = nullptr);
};

#ifdef MFEM_USE_ALGOIM
class AlgoimCutQuadratureGenerator : public CutQuadratureGenerator
{
public:
   AlgoimCutQuadratureGenerator();
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
