#include "cut_quadrature_algoim.hpp"

#ifdef MFEM_USE_ALGOIM
#include "fe_coll.hpp"
#include <algoim/quadrature_general.hpp>
#ifdef MFEM_USE_LAPACKE
#include <algoim/quadrature_multipoly.hpp>
#endif
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace mfem
{
namespace
{
struct AlgoimPolynomial
{
   int dimension = 0;
   int extent = 0;
   std::vector<real_t> coefficients;
   std::array<std::vector<real_t>, 3> derivatives;

   CutQuadratureReport Set(const PolynomialLevelSet &level_set);
   CutQuadratureReport Classify(LevelSetSign sign, CutRegion &region) const;

   template <typename Scalar>
   Scalar Evaluate(const Scalar *point, int derivative = -1) const
   {
      if (derivative >= 0 && extent == 1) { return Scalar(0); }
      const auto &source = derivative < 0 ? coefficients : derivatives[derivative];
      std::vector<Scalar> values;
      values.reserve(source.size());
      for (real_t value : source) { values.emplace_back(value); }
      size_t count = values.size();
      for (int axis = 0; axis < dimension; ++axis)
      {
         const int width = extent - (axis == derivative ? 1 : 0);
         const size_t blocks = count / width;
         for (size_t block = 0; block < blocks; ++block)
         {
            const size_t start = block * width;
            for (int degree = width - 1; degree > 0; --degree)
               for (int index = 0; index < degree; ++index)
               {
                  values[start + index] =
                     (Scalar(1) - point[axis]) * values[start + index] +
                     point[axis] * values[start + index + 1];
               }
            values[block] = values[start];
         }
         count = blocks;
      }
      return values[0];
   }
};

CutQuadratureReport AlgoimPolynomial::Set(const PolynomialLevelSet &level_set)
{
   dimension = level_set.GetDim();
   extent = level_set.GetFE().GetOrder() + 1;
   L2_FECollection collection(extent - 1, dimension, BasisType::Positive);
   const auto &positive =
      *collection.FiniteElementForGeometry(level_set.GetGeometry());
   Vector values;
   auto report = level_set.GetBernsteinCoefficients(positive, values);
   if (!report.Succeeded()) { return report; }
   const real_t scale = values.Normlinf();
   if (scale > 0) { values /= scale; }
   const auto &mapping =
      dynamic_cast<const PositiveTensorFiniteElement &>(positive).GetDofMap();
   coefficients.resize(values.Size());
   for (int index = 0; index < values.Size(); ++index)
   { coefficients[index] = values(mapping.Size() ? mapping[index] : index); }
   int stride = 1;
   for (int axis = 0; axis < dimension; ++axis)
   {
      auto &derivative = derivatives[axis];
      derivative.clear();
      for (int index = 0; index < values.Size(); ++index)
      {
         if ((index / stride) % extent == extent - 1) { continue; }
         derivative.push_back((extent - 1) *
                              (coefficients[index + stride] - coefficients[index]));
      }
      stride *= extent;
   }
   return {};
}

CutQuadratureReport AlgoimPolynomial::Classify(
   LevelSetSign sign, CutRegion &region) const
{
   region = CutRegion::Cut;
   const auto bounds = std::minmax_element(coefficients.begin(), coefficients.end());
   const real_t lower = *bounds.first, upper = *bounds.second;
   const bool surface = sign == LevelSetSign::Zero;
   if (lower == 0 && upper == 0)
   {
      if (surface)
      { return {CutQuadratureStatus::DegenerateGeometry, "Identically zero equality constraint."}; }
      region = CutRegion::Empty;
   }
   else if (surface)
   {
      if (lower > 0 || upper < 0) { region = CutRegion::Empty; return {}; }
      int stride = 1;
      for (int axis = 0; axis < dimension; ++axis)
      {
         for (int side = 0; side < 2; ++side)
         {
            real_t maximum = 0;
            for (size_t index = 0; index < coefficients.size(); ++index)
            {
               if ((index / stride) % extent == size_t(side * (extent - 1)))
               { maximum = std::max(maximum, std::abs(coefficients[index])); }
            }
            if (maximum <= 64 * std::numeric_limits<real_t>::epsilon())
            { return {CutQuadratureStatus::DegenerateGeometry, "Interface coincides with an element facet."}; }
         }
         stride *= extent;
      }
   }
   else if ((sign == LevelSetSign::Positive && lower >= 0) ||
            (sign == LevelSetSign::Negative && upper <= 0))
   { region = CutRegion::Full; }
   else if ((sign == LevelSetSign::Positive && upper <= 0) ||
            (sign == LevelSetSign::Negative && lower >= 0))
   { region = CutRegion::Empty; }
   return {};
}

real_t Checked(real_t value)
{
   if (!std::isfinite(value))
   { throw std::runtime_error("Nonfinite ALGOIM function evaluation."); }
   return value;
}

template <int Dimension>
algoim::Interval<Dimension> Checked(const algoim::Interval<Dimension> &value)
{
   Checked(value.alpha);
   Checked(value.eps);
   if (value.eps < 0)
   { throw std::runtime_error("Negative ALGOIM interval remainder."); }
   for (int axis = 0; axis < Dimension; ++axis) { Checked(value.beta(axis)); }
   return value;
}

template <int Dimension>
class IntervalGuard
{
public:
   IntervalGuard() : saved(algoim::Interval<Dimension>::delta()) { }
   ~IntervalGuard() { algoim::Interval<Dimension>::delta() = saved; }
private:
   algoim::uvector<real_t, Dimension> saved;
};

template <int Dimension>
struct GeneralFunction
{
   const ReferenceLevelSet &level_set;
   const AlgoimPolynomial *polynomial;
   real_t sign;

   template <typename Scalar>
   Scalar operator()(const algoim::uvector<Scalar, Dimension> &point) const
   {
      if (polynomial)
      {
         std::array<Scalar, Dimension> coordinates;
         for (int axis = 0; axis < Dimension; ++axis)
         { coordinates[axis] = point(axis); }
         return Checked(polynomial->Evaluate(coordinates.data()) * sign);
      }
      if constexpr (std::is_same<Scalar, real_t>::value)
      {
         IntegrationPoint reference;
         reference.Init(0);
         reference.x = point(0); reference.y = point(1);
         if constexpr (Dimension == 3) { reference.z = point(2); }
         return Checked(level_set.Eval(reference) * sign);
      }
      else
      {
         return Checked(dynamic_cast<const AlgoimIntervalLevelSet<Dimension> &>(level_set)
                        .EvalInterval(point) * sign);
      }
   }

   template <typename Scalar>
   algoim::uvector<Scalar, Dimension> grad(
      const algoim::uvector<Scalar, Dimension> &point) const
   {
      algoim::uvector<Scalar, Dimension> result;
      if (polynomial)
      {
         std::array<Scalar, Dimension> coordinates;
         for (int axis = 0; axis < Dimension; ++axis)
         { coordinates[axis] = point(axis); }
         for (int axis = 0; axis < Dimension; ++axis)
         { result(axis) = polynomial->Evaluate(coordinates.data(), axis) * sign; }
      }
      else if constexpr (std::is_same<Scalar, real_t>::value)
      {
         IntegrationPoint reference;
         reference.Init(0);
         reference.x = point(0); reference.y = point(1);
         if constexpr (Dimension == 3) { reference.z = point(2); }
         Vector gradient;
         level_set.EvalGradient(reference, gradient);
         if (gradient.Size() != Dimension)
         { throw std::runtime_error("Invalid ALGOIM gradient dimension."); }
         for (int axis = 0; axis < Dimension; ++axis)
         { result(axis) = gradient(axis) * sign; }
      }
      else
      {
         result = dynamic_cast<const AlgoimIntervalLevelSet<Dimension> &>(level_set)
                  .EvalIntervalGradient(point);
         for (int axis = 0; axis < Dimension; ++axis)
         { result(axis) *= sign; }
      }
      for (int axis = 0; axis < Dimension; ++axis) { Checked(result(axis)); }
      return result;
   }
};

template <int Dimension>
CutQuadratureReport ComputeGeneral(
   const ReferenceLevelSet &level_set, LevelSetSign sign, int order,
   IntegrationRule &rule, CutRegion &region)
{
   IntervalGuard<Dimension> guard;
   AlgoimPolynomial polynomial;
   const auto *source = level_set.AsPolynomial();
   region = CutRegion::Cut;
   if (source)
   {
      auto report = polynomial.Set(*source);
      if (!report.Succeeded()) { return report; }
      report = polynomial.Classify(sign, region);
      if (!report.Succeeded()) { return report; }
   }
   GeneralFunction<Dimension> function{
      level_set, source ? &polynomial : nullptr,
      sign == LevelSetSign::Positive ? -1.0 : 1.0};
   if (!source)
   {
      using Interval = algoim::Interval<Dimension>;
      algoim::uvector<Interval, Dimension> box;
      Interval::delta() = 0.5;
      for (int axis = 0; axis < Dimension; ++axis)
      {
         algoim::uvector<real_t, Dimension> slope(0.0);
         slope(axis) = 1;
         box(axis) = Interval(0.5, slope);
      }
      const auto value = function(box);
      const real_t deviation = value.maxDeviation();
      if (!std::isfinite(value.alpha) || !std::isfinite(deviation) || deviation < 0)
      { return {CutQuadratureStatus::NumericalFailure, "Invalid interval bounds."}; }
      if (value.alpha == 0 && deviation == 0)
      {
         if (sign == LevelSetSign::Zero)
         { return {CutQuadratureStatus::DegenerateGeometry, "Identically zero equality constraint."}; }
         region = CutRegion::Empty;
      }
      else if (value.alpha - deviation > 0)
      { region = CutRegion::Empty; }
      else if (value.alpha + deviation < 0)
      { region = sign == LevelSetSign::Zero ? CutRegion::Empty : CutRegion::Full; }
      if (sign == LevelSetSign::Zero && region == CutRegion::Cut)
      {
         for (int axis = 0; axis < Dimension; ++axis)
         {
            const auto saved = box(axis);
            for (int side = 0; side < 2; ++side)
            {
               box(axis) = Interval(side);
               const auto face = function(box);
               if (face.alpha == 0 && face.maxDeviation() == 0)
               { return {CutQuadratureStatus::DegenerateGeometry, "Interface coincides with an element facet."}; }
            }
            box(axis) = saved;
         }
      }
   }
   if (region == CutRegion::Empty) { return {}; }
   if (region == CutRegion::Full)
   { rule = IntRules.Get(level_set.GetGeometry(), order); return {}; }
   const auto quadrature = algoim::quadGen<Dimension>(
      function, algoim::HyperRectangle<real_t, Dimension>(0.0, 1.0),
      sign == LevelSetSign::Zero ? Dimension : -1, -1, order / 2 + 1);
   if (quadrature.nodes.size() > size_t(std::numeric_limits<int>::max()))
   { return {CutQuadratureStatus::NumericalFailure, "Too many ALGOIM nodes."}; }
   rule.SetSize(static_cast<int>(quadrature.nodes.size()));
   for (int index = 0; index < rule.GetNPoints(); ++index)
   {
      const auto &node = quadrature.nodes[index];
      auto &point = rule.IntPoint(index);
      point.Init(index);
      point.x = node.x(0); point.y = node.x(1);
      if constexpr (Dimension == 3) { point.z = node.x(2); }
      point.weight = node.w;
   }
   region = rule.GetNPoints() ? CutRegion::Cut : CutRegion::Empty;
   return {};
}

bool SupportsAlgoimIntervals(const ReferenceLevelSet &level_set)
{
   return level_set.AsPolynomial() ||
          dynamic_cast<const AlgoimIntervalLevelSet<2> *>(&level_set) ||
          dynamic_cast<const AlgoimIntervalLevelSet<3> *>(&level_set);
}

CutQuadratureReport GenerateAlgoimGeneral(
   Geometry::Type geometry, const Array<const ReferenceLevelSet *> &level_sets,
   const CutQuadratureRequest &request, IntegrationRule &rule, CutRegion &region)
{
   const auto &constraint = request.domain.GetConstraint(0);
   const auto &level_set = *level_sets[constraint.level_set];
   return geometry == Geometry::SQUARE ?
          ComputeGeneral<2>(level_set, constraint.sign, request.order, rule, region) :
          ComputeGeneral<3>(level_set, constraint.sign, request.order, rule, region);
}

#ifdef MFEM_USE_LAPACKE
template <int Dimension>
CutQuadratureReport ComputePolynomialRule(
   const std::vector<AlgoimPolynomial> &data, const std::vector<LevelSetSign> &signs,
   const CutQuadratureRequest &request, IntegrationRule &rule, CutRegion &region)
{
   if constexpr (Dimension == 3)
   {
      for (int axis = 0; axis < Dimension; ++axis)
      {
         bool independent = true;
         for (const auto &polynomial : data)
         {
            for (real_t value : polynomial.derivatives[axis])
            {
               if (std::abs(value) > 64 * polynomial.extent *
                   std::numeric_limits<real_t>::epsilon())
               { independent = false; break; }
            }
         }
         if (!independent) { continue; }
         std::vector<AlgoimPolynomial> reduced;
         for (const auto &polynomial : data)
         {
            AlgoimPolynomial slice;
            slice.dimension = 2;
            slice.extent = polynomial.extent;
            int stride = 1;
            for (int previous = 0; previous < axis; ++previous)
            { stride *= polynomial.extent; }
            for (size_t index = 0; index < polynomial.coefficients.size(); ++index)
            {
               if ((index / stride) % polynomial.extent == 0)
               { slice.coefficients.push_back(polynomial.coefficients[index]); }
            }
            reduced.push_back(std::move(slice));
         }
         IntegrationRule plane;
         auto report = ComputePolynomialRule<2>(reduced, signs, request, plane, region);
         if (!report.Succeeded()) { return report; }
         const auto &line = IntRules.Get(Geometry::SEGMENT, request.order);
         if (plane.GetNPoints() > std::numeric_limits<int>::max() / line.GetNPoints())
         { return {CutQuadratureStatus::NumericalFailure, "Too many extruded ALGOIM nodes."}; }
         rule.SetSize(plane.GetNPoints() * line.GetNPoints());
         int output = 0;
         for (int index = 0; index < plane.GetNPoints(); ++index)
            for (int height = 0; height < line.GetNPoints(); ++height)
            {
               const auto &point = plane.IntPoint(index);
               const auto &extrusion = line.IntPoint(height);
               real_t coordinates[3];
               int coordinate = 0;
               for (int dimension = 0; dimension < 3; ++dimension)
               {
                  coordinates[dimension] = dimension == axis ? extrusion.x :
                                            (coordinate++ == 0 ? point.x : point.y);
               }
               rule.IntPoint(output++).Set3w(coordinates[0], coordinates[1],
                                             coordinates[2], point.weight * extrusion.weight);
            }
         return {};
      }
   }
   std::array<std::vector<real_t>, 2> storage;
   std::vector<std::unique_ptr<algoim::xarray<real_t, Dimension>>> polynomials;
   for (size_t index = 0; index < data.size(); ++index)
   {
      const auto &polynomial = data[index];
      storage[index].resize(polynomial.coefficients.size());
      polynomials.emplace_back(new algoim::xarray<real_t, Dimension>(
                                  storage[index].data(),
                                  algoim::uvector<int, Dimension>(polynomial.extent)));
      auto &target = *polynomials.back();
      for (auto indices = target.loop(); ~indices; ++indices)
      {
         int offset = 0, stride = 1;
         for (int axis = 0; axis < Dimension; ++axis)
         { offset += indices(axis) * stride; stride *= polynomial.extent; }
         target.l(indices) = polynomial.coefficients[offset];
      }
      std::vector<real_t> reduced_storage = storage[index];
      algoim::xarray<real_t, Dimension> reduced(reduced_storage.data(), target.ext());
      if (algoim::bernstein::autoReduction(reduced))
      {
         auto extents = reduced.ext();
         for (int axis = 0; axis < Dimension; ++axis)
         { extents(axis) = std::max(2, extents(axis)); }
         target.alterExtent(extents);
         algoim::bernstein::bernsteinElevate(reduced, target);
      }
   }
   std::unique_ptr<algoim::ImplicitPolyQuadrature<Dimension>> quadrature;
   if (data.size() == 1)
   { quadrature.reset(new algoim::ImplicitPolyQuadrature<Dimension>(*polynomials[0])); }
   else
   { quadrature.reset(new algoim::ImplicitPolyQuadrature<Dimension>(*polynomials[0], *polynomials[1])); }
   std::vector<IntegrationPoint> nodes;
   auto append = [&](const algoim::uvector<real_t, Dimension> &coordinates,
                     real_t weight)
   {
      IntegrationPoint point;
      point.Init(0);
      point.x = coordinates(0); point.y = coordinates(1);
      if constexpr (Dimension == 3) { point.z = coordinates(2); }
      point.weight = weight;
      nodes.push_back(point);
   };
   static_assert(algoim::GaussQuad::p_max >= 100 &&
                 algoim::TanhSinhQuadrature::n_max >= 100,
                 "Update the polynomial ALGOIM order limit.");
   if (request.domain.Codimension())
   {
      quadrature->integrate_surf(algoim::AutoMixed, request.order / 2 + 1,
         [&](const auto &coordinates, real_t weight, const auto &)
         { append(coordinates, weight); });
   }
   else
   {
      quadrature->integrate(algoim::AutoMixed, request.order / 2 + 1,
         [&](const auto &coordinates, real_t weight)
         {
            for (size_t index = 0; index < polynomials.size(); ++index)
            {
               const real_t value = algoim::bernstein::evalBernsteinPoly(
                                       *polynomials[index], coordinates);
               if ((signs[index] == LevelSetSign::Positive && value <= 0) ||
                   (signs[index] == LevelSetSign::Negative && value >= 0))
               { return; }
            }
            append(coordinates, weight);
         });
   }
   if (nodes.size() > size_t(std::numeric_limits<int>::max()))
   { return {CutQuadratureStatus::NumericalFailure, "Too many ALGOIM nodes."}; }
   rule.SetSize(static_cast<int>(nodes.size()));
   for (int index = 0; index < rule.GetNPoints(); ++index)
   { rule.IntPoint(index) = nodes[index]; }
   region = nodes.empty() ? CutRegion::Empty : CutRegion::Cut;
   return {};
}

CutQuadratureReport GenerateAlgoimPolynomial(
   Geometry::Type geometry, const Array<const ReferenceLevelSet *> &level_sets,
   const CutQuadratureRequest &request, IntegrationRule &rule, CutRegion &region)
{
   std::vector<AlgoimPolynomial> data;
   std::vector<LevelSetSign> signs;
   for (int index = 0; index < request.domain.NumConstraints(); ++index)
   {
      const auto &constraint = request.domain.GetConstraint(index);
      AlgoimPolynomial polynomial;
      auto report = polynomial.Set(*level_sets[constraint.level_set]->AsPolynomial());
      if (!report.Succeeded()) { return report; }
      CutRegion classification;
      report = polynomial.Classify(constraint.sign, classification);
      if (!report.Succeeded()) { return report; }
      if (classification == CutRegion::Empty)
      { region = CutRegion::Empty; return {}; }
      if (classification == CutRegion::Full) { continue; }
      data.push_back(std::move(polynomial));
      signs.push_back(constraint.sign);
   }
   if (data.empty())
   { rule = IntRules.Get(geometry, request.order); region = CutRegion::Full; return {}; }
   return geometry == Geometry::SQUARE ?
          ComputePolynomialRule<2>(data, signs, request, rule, region) :
          ComputePolynomialRule<3>(data, signs, request, rule, region);
}
#endif

enum class Algorithm { Automatic, General, Polynomial };

class AlgoimBackend : public CutQuadratureBackend
{
public:
   explicit AlgoimBackend(Algorithm algorithm) : algorithm(algorithm) { }

   std::string Name() const override
   {
      return algorithm == Algorithm::General ? "algoim-general" :
             (algorithm == Algorithm::Polynomial ? "algoim-polynomial" : "algoim");
   }

   CutQuadratureReport CheckSupport(
      Geometry::Type geometry, const Array<const ReferenceLevelSet *> &level_sets,
      const CutQuadratureRequest &request) const override
   {
      auto report = request.domain.Validate(geometry, level_sets.Size());
      if (!report.Succeeded()) { return report; }
      if (request.order < 0)
      { return {CutQuadratureStatus::InvalidInput, "Negative quadrature order."}; }
      for (const auto *level_set : level_sets)
      {
         if (!level_set || level_set->GetGeometry() != geometry)
         { return {CutQuadratureStatus::InvalidInput, "Invalid level-set geometry."}; }
      }
      if (geometry != Geometry::SQUARE && geometry != Geometry::CUBE)
      { return {CutQuadratureStatus::Unsupported, "ALGOIM supports quads and hexes."}; }
      if (request.domain.GetFace() >= 0 || request.fixed_points)
      { return {CutQuadratureStatus::Unsupported, "Unsupported ALGOIM facet or prescribed nodes."}; }
      const int count = request.domain.NumConstraints();
      const bool polynomial = UsePolynomial(level_sets, request);
      if (polynomial)
      {
#ifndef MFEM_USE_LAPACKE
         return {CutQuadratureStatus::Unsupported, "Polynomial ALGOIM requires LAPACKE."};
#else
         if (count > 2 || (count > 1 && request.domain.Codimension()))
         { return {CutQuadratureStatus::Unsupported, "Polynomial ALGOIM supports two signs or one surface."}; }
         for (int index = 0; index < count; ++index)
         {
            if (!level_sets[request.domain.GetConstraint(index).level_set]->AsPolynomial())
            { return {CutQuadratureStatus::Unsupported, "Polynomial ALGOIM requires polynomial inputs."}; }
         }
#endif
      }
      else
      {
         if (count > 1)
         { return {CutQuadratureStatus::Unsupported, "General ALGOIM supports one active level set."}; }
         if (count && !SupportsAlgoimIntervals(
                *level_sets[request.domain.GetConstraint(0).level_set]))
         { return {CutQuadratureStatus::Unsupported, "General ALGOIM requires interval-capable inputs."}; }
      }
      if (request.order / 2 + 1 > (polynomial ? 100 : 10))
      { return {CutQuadratureStatus::Unsupported, "Quadrature order exceeds the selected ALGOIM limit."}; }
      return {};
   }

   CutQuadratureReport Generate(
      Geometry::Type geometry, const Array<const ReferenceLevelSet *> &level_sets,
      const CutQuadratureRequest &request, IntegrationRule &rule,
      CutRegion &region) const override
   {
      rule.SetSize(0); rule.SetOrder(-1); region = CutRegion::Unknown;
      auto report = CheckSupport(geometry, level_sets, request);
      if (!report.Succeeded()) { return report; }
      try
      {
         if (!request.domain.NumConstraints())
         { rule = IntRules.Get(geometry, request.order); region = CutRegion::Full; }
#ifdef MFEM_USE_LAPACKE
         else if (UsePolynomial(level_sets, request))
         { report = GenerateAlgoimPolynomial(geometry, level_sets, request, rule, region); }
#endif
         else
         { report = GenerateAlgoimGeneral(geometry, level_sets, request, rule, region); }
      }
      catch (const std::exception &error)
      { report = {CutQuadratureStatus::NumericalFailure, error.what()}; }
      if (report.Succeeded())
      {
         for (int index = 0; index < rule.GetNPoints(); ++index)
         {
            const auto &point = rule.IntPoint(index);
            if (!std::isfinite(point.weight) || !std::isfinite(point.x) ||
                !std::isfinite(point.y) ||
                (geometry == Geometry::CUBE && !std::isfinite(point.z)) ||
                !Geometry::CheckPoint(geometry, point,
                                     128 * std::numeric_limits<real_t>::epsilon()))
            { report = {CutQuadratureStatus::NumericalFailure, "Invalid ALGOIM quadrature node."}; break; }
         }
      }
      if (!report.Succeeded())
      { rule.SetSize(0); rule.SetOrder(-1); region = CutRegion::Unknown; }
      else
      {
         rule.SetOrder(request.order);
         for (int index = 0; index < rule.GetNPoints(); ++index)
         { rule.IntPoint(index).index = index; }
      }
      return report;
   }

private:
   Algorithm algorithm;

   bool UsePolynomial(const Array<const ReferenceLevelSet *> &level_sets,
                      const CutQuadratureRequest &request) const
   {
      if (algorithm != Algorithm::Automatic)
      { return algorithm == Algorithm::Polynomial; }
#ifdef MFEM_USE_LAPACKE
      for (int index = 0; index < request.domain.NumConstraints(); ++index)
      {
         if (!level_sets[request.domain.GetConstraint(index).level_set]->AsPolynomial())
         { return false; }
      }
      return true;
#else
      return false;
#endif
   }
};
}

std::unique_ptr<CutQuadratureBackend> CreateCutQuadratureBackend(
   const std::string &name)
{
   if (name == "algoim")
   { return std::unique_ptr<CutQuadratureBackend>(new AlgoimBackend(Algorithm::Automatic)); }
   if (name == "algoim-general")
   { return std::unique_ptr<CutQuadratureBackend>(new AlgoimBackend(Algorithm::General)); }
#ifdef MFEM_USE_LAPACKE
   if (name == "algoim-polynomial")
   { return std::unique_ptr<CutQuadratureBackend>(new AlgoimBackend(Algorithm::Polynomial)); }
#endif
   return nullptr;
}

std::vector<std::string> GetCutQuadratureBackends()
{
   std::vector<std::string> result = {"algoim", "algoim-general"};
#ifdef MFEM_USE_LAPACKE
   result.push_back("algoim-polynomial");
#endif
   return result;
}
}
#endif
