// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC.
// SPDX-License-Identifier: BSD-3-Clause

#include "cut_quadrature.hpp"

#include "coefficient.hpp"
#include "eltrans.hpp"
#include "fe/fe_h1.hpp"
#include "fe/fe_pos.hpp"
#include "fespace.hpp"
#include "gridfunc.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>

namespace mfem
{

namespace
{

std::atomic<std::uint64_t> next_provider_id(1);

bool HasMeasure(CutMeasure set, CutMeasure measure)
{
   return static_cast<unsigned>(set & measure) != 0u;
}

bool IsFinite(const Vector &values)
{
   for (int i = 0; i < values.Size(); i++)
   {
      if (!std::isfinite(values(i))) { return false; }
   }
   return true;
}

template <typename NodalElement, typename PositiveElement, typename Sampler>
CutQuadratureStatus InterpolateToBernstein(int order,
                                          ElementTransformation &Tr,
                                          Sampler sample,
                                          ElementLevelSet &result)
{
   NodalElement nodal(order);
   PositiveElement positive(order);
   const IntegrationRule &nodes = nodal.GetNodes();
   Vector nodal_values(nodes.GetNPoints());
   for (int i = 0; i < nodes.GetNPoints(); i++)
   {
      nodal_values(i) = sample(nodes.IntPoint(i));
   }
   if (!IsFinite(nodal_values)) { return CutQuadratureStatus::InvalidLevelSet; }

   DenseMatrix transform(positive.GetDof(), nodal.GetDof());
   positive.Project(nodal, Tr, transform);
   Vector positive_values(positive.GetDof());
   transform.Mult(nodal_values, positive_values);
   if (!IsFinite(positive_values))
   {
      return CutQuadratureStatus::InvalidLevelSet;
   }

   const Array<int> &dof_map = positive.GetDofMap();
   result.coefficients.SetSize(positive_values.Size());
   for (int i = 0; i < positive_values.Size(); i++)
   {
      result.coefficients(i) = positive_values(dof_map[i]);
   }
   return CutQuadratureStatus::Success;
}

template <typename Sampler>
CutQuadratureStatus ExtractTensorPolynomial(Geometry::Type geometry, int order,
                                            ElementTransformation &Tr,
                                            Sampler sample,
                                            ElementLevelSet &result)
{
   if (order < 1) { return CutQuadratureStatus::UnsupportedSourceBasis; }
   CutQuadratureStatus status;
   if (geometry == Geometry::SQUARE)
   {
      status = InterpolateToBernstein<H1_QuadrilateralElement,
             H1Pos_QuadrilateralElement>(order, Tr, sample, result);
   }
   else if (geometry == Geometry::CUBE)
   {
      status = InterpolateToBernstein<H1_HexahedronElement,
             H1Pos_HexahedronElement>(order, Tr, sample, result);
   }
   else
   {
      return CutQuadratureStatus::UnsupportedSourceBasis;
   }
   if (status == CutQuadratureStatus::Success)
   {
      result.geometry = geometry;
      result.order = order;
      result.basis = PolynomialBasis::BernsteinTensor;
   }
   return status;
}

} // namespace

bool ElementLevelSetDescriptor::operator==(
   const ElementLevelSetDescriptor &other) const
{
   return geometry == other.geometry && basis == other.basis &&
          order == other.order;
}

bool ElementLevelSetDescriptor::operator!=(
   const ElementLevelSetDescriptor &other) const
{
   return !(*this == other);
}

bool CutQuadratureRequest::operator==(
   const CutQuadratureRequest &other) const
{
   return order == other.order && region == other.region &&
          measures == other.measures && weight_policy == other.weight_policy &&
          execution == other.execution &&
          compute_reference_normals == other.compute_reference_normals;
}

bool CutQuadratureRequest::operator!=(
   const CutQuadratureRequest &other) const
{
   return !(*this == other);
}

bool CutQuadratureCapabilities::Supports(
   const CutQuadratureRequest &request,
   const ElementLevelSetDescriptor &level_set, bool batch) const
{
   const unsigned measures = static_cast<unsigned>(request.measures);
   const unsigned allowed = static_cast<unsigned>(CutMeasure::Volume) |
                            static_cast<unsigned>(CutMeasure::Interface);
   if (measures == 0u || (measures & ~allowed) != 0u) { return false; }
   if (request.order < min_order || request.order > max_order) { return false; }
   if (request.execution != CutExecutionMode::Host &&
       request.execution != CutExecutionMode::Device) { return false; }
   if ((request.execution == CutExecutionMode::Host &&
        (batch ? !host_batch : !host_scalar)) ||
       (request.execution == CutExecutionMode::Device &&
        (!batch || !device_batch)))
   {
      return false;
   }

   bool found = false;
   for (int i = 0; i < geometries.Size(); i++)
   {
      found = found || geometries[i] == level_set.geometry;
   }
   if (!found) { return false; }
   found = false;
   for (int i = 0; i < bases.Size(); i++)
   {
      found = found || bases[i] == level_set.basis;
   }
   if (!found) { return false; }

   if ((HasMeasure(request.measures, CutMeasure::Volume) && !volume) ||
       (HasMeasure(request.measures, CutMeasure::Interface) && !interface) ||
       (request.region == CutRegion::Negative && !negative_phase) ||
       (request.region == CutRegion::Positive && !positive_phase) ||
       (request.compute_reference_normals && !normals) ||
       (request.weight_policy == QuadratureWeightPolicy::Unconstrained &&
        !unconstrained_weights) ||
       (request.weight_policy == QuadratureWeightPolicy::Nonnegative &&
        !nonnegative_weights))
   {
      return false;
   }
   return true;
}

ElementLevelSetProvider::ElementLevelSetProvider()
   : id_(next_provider_id.fetch_add(1))
{
}

GridFunctionLevelSetProvider::GridFunctionLevelSetProvider(
   const GridFunction &level_set, LevelSetRevision revision)
   : level_set_(&level_set), revision_(revision)
{
}

CutQuadratureStatus GridFunctionLevelSetProvider::GetElementLevelSet(
   int element, ElementTransformation &Tr, ElementLevelSet &result) const
{
   const FiniteElementSpace *fes = level_set_->FESpace();
   if (!fes || element < 0 || element >= fes->GetNE())
   {
      return CutQuadratureStatus::InvalidLevelSet;
   }
   const FiniteElement *fe = fes->GetFE(element);
   if (fes->GetVDim() != 1 || fe->GetRangeType() != FiniteElement::SCALAR ||
       fe->GetMapType() != FiniteElement::VALUE ||
       dynamic_cast<const TensorBasisElement *>(fe) == nullptr)
   {
      return CutQuadratureStatus::UnsupportedSourceBasis;
   }
   const Geometry::Type geometry = fe->GetGeomType();
   const int order = fe->GetOrder();
   return ExtractTensorPolynomial(
             geometry, order, Tr,
             [this, element](const IntegrationPoint &ip)
   {
      return level_set_->GetValue(element, ip);
   }, result);
}

CoefficientLevelSetProvider::CoefficientLevelSetProvider(
   Coefficient &level_set, int approximation_order, LevelSetRevision revision)
   : level_set_(&level_set), approximation_order_(approximation_order),
     revision_(revision)
{
}

CutQuadratureStatus CoefficientLevelSetProvider::GetElementLevelSet(
   int, ElementTransformation &Tr, ElementLevelSet &result) const
{
   if (approximation_order_ < 1)
   {
      return CutQuadratureStatus::UnsupportedSourceBasis;
   }
   return ExtractTensorPolynomial(
             Tr.GetGeometryType(), approximation_order_, Tr,
             [this, &Tr](const IntegrationPoint &ip)
   {
      Tr.SetIntPoint(&ip);
      return level_set_->Eval(Tr, ip);
   }, result);
}

bool RetainedCutQuadrature::IsValid(
   const ElementLevelSetProvider &provider, int element_id,
   const CutQuadratureRequest &requested) const
{
   return provider_id == provider.Id() && element == element_id &&
          revision == provider.Revision() && request == requested;
}

bool RetainedBatchedCutQuadrature::IsValid(
   const ElementLevelSetProvider &provider, const Array<int> &element_ids,
   const CutQuadratureRequest &requested) const
{
   if (provider_id != provider.Id() || revision != provider.Revision() ||
       request != requested || elements.Size() != element_ids.Size())
   {
      return false;
   }
   for (int i = 0; i < elements.Size(); i++)
   {
      if (elements[i] != element_ids[i]) { return false; }
   }
   return true;
}

void MapReferenceVolumeRule(ElementTransformation &Tr,
                            const IntegrationRule &reference,
                            IntegrationRule &mapped)
{
   mapped.SetSize(reference.GetNPoints());
   mapped.SetOrder(reference.GetOrder());
   for (int i = 0; i < reference.GetNPoints(); i++)
   {
      const IntegrationPoint &source = reference.IntPoint(i);
      mapped.IntPoint(i) = source;
      Tr.SetIntPoint(&source);
      mapped.IntPoint(i).weight = source.weight * Tr.Weight();
   }
   mapped.SetPointIndices();
}

void MapReferenceInterfaceRule(ElementTransformation &Tr,
                               const ReferenceInterfaceRule &reference,
                               IntegrationRule &mapped,
                               DenseMatrix *physical_normals)
{
   const int nq = reference.rule.GetNPoints();
   const int rdim = Tr.GetDimension();
   const int sdim = Tr.GetSpaceDim();
   MFEM_VERIFY(reference.reference_normals.Height() == rdim &&
               reference.reference_normals.Width() == nq,
               "reference normals are required for surface mapping");
   mapped.SetSize(nq);
   mapped.SetOrder(reference.rule.GetOrder());
   if (physical_normals) { physical_normals->SetSize(sdim, nq); }
   Vector nref(rdim), transformed(sdim);
   for (int i = 0; i < nq; i++)
   {
      const IntegrationPoint &source = reference.rule.IntPoint(i);
      mapped.IntPoint(i) = source;
      Tr.SetIntPoint(&source);
      reference.reference_normals.GetColumn(i, nref);
      Tr.InverseJacobian().MultTranspose(nref, transformed);
      const real_t metric = transformed.Norml2();
      mapped.IntPoint(i).weight = source.weight * Tr.Weight() * metric;
      if (physical_normals)
      {
         MFEM_VERIFY(metric > 0.0, "degenerate physical interface normal");
         transformed /= metric;
         physical_normals->SetCol(i, transformed);
      }
   }
   mapped.SetPointIndices();
}

real_t CutQuadratureIntegrator::IntegrateVolume(
   Coefficient &coefficient, ElementTransformation &Tr,
   const ReferenceCutQuadrature &quadrature)
{
   MFEM_VERIFY(quadrature.status == CutQuadratureStatus::Success,
               "cannot consume a failed cut quadrature result");
   real_t integral = 0.0;
   for (int i = 0; i < quadrature.volume.GetNPoints(); i++)
   {
      const IntegrationPoint &ip = quadrature.volume.IntPoint(i);
      Tr.SetIntPoint(&ip);
      integral += ip.weight * Tr.Weight() * coefficient.Eval(Tr, ip);
   }
   return integral;
}

real_t CutQuadratureIntegrator::IntegrateInterface(
   Coefficient &coefficient, ElementTransformation &Tr,
   const ReferenceCutQuadrature &quadrature, DenseMatrix *physical_normals)
{
   MFEM_VERIFY(quadrature.status == CutQuadratureStatus::Success,
               "cannot consume a failed cut quadrature result");
   IntegrationRule mapped;
   MapReferenceInterfaceRule(Tr, quadrature.interface, mapped,
                             physical_normals);
   real_t integral = 0.0;
   for (int i = 0; i < mapped.GetNPoints(); i++)
   {
      const IntegrationPoint &ip = mapped.IntPoint(i);
      Tr.SetIntPoint(&ip);
      integral += ip.weight * coefficient.Eval(Tr, ip);
   }
   return integral;
}

} // namespace mfem
