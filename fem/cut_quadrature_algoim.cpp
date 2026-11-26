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
std::unique_ptr<CutQuadratureBackend> CreateMomentCutQuadratureBackend();

namespace
{
using Region = CutQuadrature::Region;

struct AlgoimPolynomial
{
   int dimension = 0;
   int extent = 0;
   std::vector<algoim::real> coefficients;
   std::array<std::vector<algoim::real>, 3> derivatives;

   void Set(const PolynomialLevelSet &level_set);
   CutQuadratureReport Classify(CutDomain::LevelSetSign sign,
                                Region &region) const;

   template <typename Scalar>
   Scalar Evaluate(const Scalar *point, int derivative = -1) const
   {
      if (derivative >= 0 && extent == 1)
      {
         return Scalar(0);
      }
      const auto &source = derivative < 0 ?
                           coefficients : derivatives[derivative];
      std::vector<Scalar> values;
      values.reserve(source.size());
      for (algoim::real value : source)
      {
         values.emplace_back(value);
      }
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

void BernsteinCoefficients(const PolynomialLevelSet &level_set,
                           const PositiveTensorFiniteElement &target,
                           Vector &coefficients)
{
   const FiniteElement &source = level_set.GetFE();
   const Vector &source_values = level_set.GetDofs();
   if (const auto *positive =
          dynamic_cast<const PositiveTensorFiniteElement *>(&source))
   {
      const auto &source_map = positive->GetDofMap();
      const auto &target_map = target.GetDofMap();
      coefficients.SetSize(source_values.Size());
      for (int index = 0; index < source_values.Size(); ++index)
      {
         coefficients(target_map.Size() ? target_map[index] : index) =
            source_values(source_map.Size() ? source_map[index] : index);
      }
      return;
   }

   DenseMatrix interpolation(source.GetDof(), target.GetDof());
   Vector shape(target.GetDof());
   for (int index = 0; index < source.GetDof(); ++index)
   {
      target.CalcShape(source.GetNodes().IntPoint(index), shape);
      interpolation.SetRow(index, shape);
   }
   DenseMatrixInverse inverse(interpolation);
   coefficients.SetSize(target.GetDof());
   inverse.Mult(source_values, coefficients);
   for (int index = 0; index < coefficients.Size(); ++index)
   {
      MFEM_VERIFY(std::isfinite(coefficients(index)),
                  "ALGOIM Bernstein conversion failed.");
   }
}

void AlgoimPolynomial::Set(const PolynomialLevelSet &level_set)
{
   dimension = level_set.GetDim();
   extent = level_set.GetFE().GetOrder() + 1;
   L2_FECollection collection(extent - 1, dimension, BasisType::Positive);
   const auto &positive = dynamic_cast<const PositiveTensorFiniteElement &>(
                             *collection.FiniteElementForGeometry(
                                level_set.GetReferenceGeometry()));
   Vector values;
   BernsteinCoefficients(level_set, positive, values);
   const real_t scale = values.Normlinf();
   if (scale > 0)
   {
      values /= scale;
   }
   const auto &mapping = positive.GetDofMap();
   coefficients.resize(values.Size());
   for (int index = 0; index < values.Size(); ++index)
   {
      coefficients[index] = values(mapping.Size() ? mapping[index] : index);
   }
   int stride = 1;
   for (int axis = 0; axis < dimension; ++axis)
   {
      auto &derivative = derivatives[axis];
      derivative.clear();
      for (int index = 0; index < values.Size(); ++index)
      {
         if ((index / stride) % extent == extent - 1)
         {
            continue;
         }
         derivative.push_back(
            (extent - 1) *
            (coefficients[index + stride] - coefficients[index]));
      }
      stride *= extent;
   }
}

CutQuadratureReport AlgoimPolynomial::Classify(
   CutDomain::LevelSetSign sign, Region &region) const
{
   region = Region::Cut;
   const auto bounds = std::minmax_element(coefficients.begin(),
                                           coefficients.end());
   const algoim::real lower = *bounds.first, upper = *bounds.second;
   const bool surface = sign == CutDomain::LevelSetSign::Zero;
   if (lower == 0 && upper == 0)
   {
      if (surface)
      {
         return {CutQuadratureReport::Status::DegenerateGeometry,
                 "Identically zero equality constraint."};
      }
      region = Region::Empty;
   }
   else if (surface)
   {
      if (lower > 0 || upper < 0)
      {
         region = Region::Empty;
         return {};
      }
      int stride = 1;
      for (int axis = 0; axis < dimension; ++axis)
      {
         for (int side = 0; side < 2; ++side)
         {
            algoim::real maximum = 0;
            for (size_t index = 0; index < coefficients.size(); ++index)
            {
               if ((index / stride) % extent == size_t(side * (extent - 1)))
               {
                  maximum = std::max(maximum, std::abs(coefficients[index]));
               }
            }
            if (maximum <= 64 * std::numeric_limits<real_t>::epsilon())
            {
               return {CutQuadratureReport::Status::DegenerateGeometry,
                       "Interface coincides with an element facet."};
            }
         }
         stride *= extent;
      }
   }
   else if ((sign == CutDomain::LevelSetSign::Positive && lower >= 0) ||
            (sign == CutDomain::LevelSetSign::Negative && upper <= 0))
   {
      region = Region::Full;
   }
   else if ((sign == CutDomain::LevelSetSign::Positive && upper <= 0) ||
            (sign == CutDomain::LevelSetSign::Negative && lower >= 0))
   {
      region = Region::Empty;
   }
   return {};
}

template <typename Scalar,
          typename = std::enable_if_t<std::is_floating_point<Scalar>::value>>
Scalar Checked(Scalar value)
{
   if (!std::isfinite(value))
   {
      throw std::runtime_error("Nonfinite ALGOIM function evaluation.");
   }
   return value;
}

template <int Dimension>
algoim::Interval<Dimension> Checked(const algoim::Interval<Dimension> &value)
{
   Checked(value.alpha);
   Checked(value.eps);
   if (value.eps < 0)
   {
      throw std::runtime_error("Negative ALGOIM interval remainder.");
   }
   for (int axis = 0; axis < Dimension; ++axis)
   {
      Checked(value.beta(axis));
   }
   return value;
}

template <int Dimension>
class IntervalGuard
{
public:
   IntervalGuard() : saved(algoim::Interval<Dimension>::delta())
   {
   }

   ~IntervalGuard()
   {
      algoim::Interval<Dimension>::delta() = saved;
   }

private:
   algoim::uvector<algoim::real, Dimension> saved;
};

template <int Dimension>
struct GeneralFunction
{
   const ReferenceLevelSet &level_set;
   const AlgoimPolynomial *polynomial;
   algoim::real sign;

   template <typename Scalar>
   Scalar operator()(const algoim::uvector<Scalar, Dimension> &point) const
   {
      if (polynomial)
      {
         std::array<Scalar, Dimension> coordinates;
         for (int axis = 0; axis < Dimension; ++axis)
         {
            coordinates[axis] = point(axis);
         }
         return Checked(polynomial->Evaluate(coordinates.data()) * sign);
      }
      if constexpr (std::is_same<Scalar, algoim::real>::value)
      {
         IntegrationPoint reference;
         reference.Init(0);
         reference.x = point(0);
         reference.y = point(1);
         if constexpr (Dimension == 3)
         {
            reference.z = point(2);
         }
         return Checked(level_set.Eval(reference) * sign);
      }
      else
      {
         return Checked(dynamic_cast<const AlgoimIntervalLevelSet<Dimension> &>
                        (level_set).EvalInterval(point) * sign);
      }
   }

   template <typename Scalar>
   algoim::uvector<Scalar, Dimension> grad(
      const algoim::uvector<Scalar, Dimension> &point) const
   {
      algoim::uvector<Scalar, Dimension> result;
      if (polynomial)
      {
         std::array<Scalar, Dimension> coords;
         for (int axis = 0; axis < Dimension; ++axis)
         {
            coords[axis] = point(axis);
         }
         for (int axis = 0; axis < Dimension; ++axis)
         {
            result(axis) = polynomial->Evaluate(coords.data(), axis) * sign;
         }
      }
      else if constexpr (std::is_same<Scalar, algoim::real>::value)
      {
         IntegrationPoint reference;
         reference.Init(0);
         reference.x = point(0);
         reference.y = point(1);
         if constexpr (Dimension == 3)
         {
            reference.z = point(2);
         }
         Vector gradient;
         level_set.EvalGradient(reference, gradient);
         if (gradient.Size() != Dimension)
         {
            throw std::runtime_error("Invalid ALGOIM gradient dimension.");
         }
         for (int axis = 0; axis < Dimension; ++axis)
         {
            result(axis) = gradient(axis) * sign;
         }
      }
      else
      {
         result = dynamic_cast<const AlgoimIntervalLevelSet<Dimension> &>
                  (level_set).EvalIntervalGradient(point);
         for (int axis = 0; axis < Dimension; ++axis)
         {
            result(axis) *= sign;
         }
      }
      for (int axis = 0; axis < Dimension; ++axis)
      {
         Checked(result(axis));
      }
      return result;
   }
};

template <int Dimension>
CutQuadratureReport ComputeGeneral(const ReferenceLevelSet &level_set,
                                   CutDomain::LevelSetSign sign, int order,
                                   IntegrationRule &rule, Region &region)
{
   IntervalGuard<Dimension> guard;
   AlgoimPolynomial polynomial;
   const auto *source = dynamic_cast<const PolynomialLevelSet *>(&level_set);
   region = Region::Cut;
   if (source)
   {
      polynomial.Set(*source);
      auto report = polynomial.Classify(sign, region);
      if (!report.Succeeded())
      {
         return report;
      }
   }
   GeneralFunction<Dimension> function
   {
      level_set, source ? &polynomial : nullptr,
      sign == CutDomain::LevelSetSign::Positive ? -1.0 : 1.0};
   if (!source)
   {
      using Interval = algoim::Interval<Dimension>;
      algoim::uvector<Interval, Dimension> box;
      Interval::delta() = 0.5;
      for (int axis = 0; axis < Dimension; ++axis)
      {
         algoim::uvector<algoim::real, Dimension> slope(0.0);
         slope(axis) = 1;
         box(axis) = Interval(0.5, slope);
      }
      const auto value = function(box);
      const algoim::real deviation = value.maxDeviation();
      if (!std::isfinite(value.alpha) ||
          !std::isfinite(deviation) || deviation < 0)
      {
         return {CutQuadratureReport::Status::NumericalFailure,
                 "Invalid interval bounds."};
      }
      if (value.alpha == 0 && deviation == 0)
      {
         if (sign == CutDomain::LevelSetSign::Zero)
         {
            return {CutQuadratureReport::Status::DegenerateGeometry,
                    "Identically zero equality constraint."};
         }
         region = Region::Empty;
      }
      else if (value.alpha - deviation > 0)
      {
         region = Region::Empty;
      }
      else if (value.alpha + deviation < 0)
      {
         region = sign == CutDomain::LevelSetSign::Zero ?
                  Region::Empty : Region::Full;
      }
      if (sign == CutDomain::LevelSetSign::Zero && region == Region::Cut)
      {
         for (int axis = 0; axis < Dimension; ++axis)
         {
            const auto saved = box(axis);
            for (int side = 0; side < 2; ++side)
            {
               box(axis) = Interval(side);
               const auto face = function(box);
               if (face.alpha == 0 && face.maxDeviation() == 0)
               {
                  return {CutQuadratureReport::Status::DegenerateGeometry,
                          "Interface coincides with an element facet."};
               }
            }
            box(axis) = saved;
         }
      }
   }
   if (region == Region::Empty)
   {
      return {};
   }
   if (region == Region::Full)
   {
      rule = IntRules.Get(level_set.GetReferenceGeometry(), order);
      return {};
   }
   const auto quadrature =
      algoim::quadGen<Dimension>(
         function,
         algoim::HyperRectangle<algoim::real, Dimension>(0.0, 1.0),
         sign == CutDomain::LevelSetSign::Zero ? Dimension : -1,
         -1, order / 2 + 1);
   if (quadrature.nodes.size() > size_t(std::numeric_limits<int>::max()))
   {
      return {CutQuadratureReport::Status::NumericalFailure,
              "Too many ALGOIM nodes."};
   }
   rule.SetSize(static_cast<int>(quadrature.nodes.size()));
   for (int index = 0; index < rule.GetNPoints(); ++index)
   {
      const auto &node = quadrature.nodes[index];
      auto &point = rule.IntPoint(index);
      point.Init(index);
      point.x = node.x(0);
      point.y = node.x(1);
      if constexpr (Dimension == 3)
      {
         point.z = node.x(2);
      }
      point.weight = node.w;
   }
   region = rule.GetNPoints() ? Region::Cut : Region::Empty;
   return {};
}

bool SupportsAlgoimIntervals(const ReferenceLevelSet &level_set)
{
   return dynamic_cast<const PolynomialLevelSet *>(&level_set) ||
          dynamic_cast<const AlgoimIntervalLevelSet<2> *>(&level_set) ||
          dynamic_cast<const AlgoimIntervalLevelSet<3> *>(&level_set);
}

CutQuadratureReport GenerateAlgoimGeneral(
   Geometry::Type geometry, const CutDomain &domain, int order,
   IntegrationRule &rule, Region &region)
{
   const auto &constraint = domain.GetConstraint(0);
   const auto &level_set = constraint.level_set.get();
   return geometry == Geometry::SQUARE ?
          ComputeGeneral<2>(level_set, constraint.sign, order, rule, region) :
          ComputeGeneral<3>(level_set, constraint.sign, order, rule, region);
}

#ifdef MFEM_USE_LAPACKE
template <int Dimension>
void CopyPolynomialCoefficients(const AlgoimPolynomial &polynomial,
                                algoim::xarray<algoim::real, Dimension> &target)
{
   for (auto indices = target.loop(); ~indices; ++indices)
   {
      int offset = 0, stride = 1;
      for (int axis = 0; axis < Dimension; ++axis)
      {
         offset += indices(axis) * stride;
         stride *= polynomial.extent;
      }
      target.l(indices) = polynomial.coefficients[offset];
   }
}

template <int Dimension>
bool NearlyProportional(const AlgoimPolynomial &first,
                        const AlgoimPolynomial &second)
{
   const int extent = std::max(first.extent, second.extent);
   const size_t count = std::max(first.coefficients.size(),
                                 second.coefficients.size());
   std::array<std::vector<algoim::real>, 2> coefficients;
   for (int index = 0; index < 2; ++index)
   {
      const auto &polynomial = index == 0 ? first : second;
      std::vector<algoim::real> storage(polynomial.coefficients.size());
      algoim::xarray<algoim::real, Dimension> source(
         storage.data(),
         algoim::uvector<int, Dimension>(polynomial.extent));
      CopyPolynomialCoefficients(polynomial, source);
      coefficients[index].resize(count);
      algoim::xarray<algoim::real, Dimension> elevated(
         coefficients[index].data(),
         algoim::uvector<int, Dimension>(extent));
      algoim::bernstein::bernsteinElevate(source, elevated);
      algoim::bernstein::normalise(elevated);
   }
   algoim::real same_residual = 0, opposite_residual = 0;
   for (size_t index = 0; index < count; ++index)
   {
      const algoim::real first_value = Checked(coefficients[0][index]);
      const algoim::real second_value = Checked(coefficients[1][index]);
      same_residual = std::max(same_residual,
                               std::abs(first_value - second_value));
      opposite_residual = std::max(opposite_residual,
                                   std::abs(first_value + second_value));
   }
   const algoim::real tolerance =
      64 * algoim::real(Dimension) * extent *
      std::numeric_limits<real_t>::epsilon();
   return std::min(same_residual, opposite_residual) <= tolerance;
}

template <int Dimension>
std::array<bool, Dimension> IndependentAxes(
   const AlgoimPolynomial &polynomial)
{
   std::vector<algoim::real> storage(polynomial.coefficients.size());
   algoim::xarray<algoim::real, Dimension> reduced(
      storage.data(),
      algoim::uvector<int, Dimension>(polynomial.extent));
   CopyPolynomialCoefficients(polynomial, reduced);
   algoim::bernstein::autoReduction(reduced);
   std::array<bool, Dimension> independent;
   for (int axis = 0; axis < Dimension; ++axis)
   {
      independent[axis] = reduced.ext(axis) == 1;
   }
   return independent;
}

template <int Dimension>
CutQuadratureReport ComputePolynomialRule(
   const std::vector<AlgoimPolynomial> &data,
   const std::vector<CutDomain::LevelSetSign> &signs,
   const CutDomain &domain, int order, IntegrationRule &rule, Region &region)
{
   if constexpr (Dimension == 3)
   {
      std::array<bool, Dimension> independent;
      independent.fill(true);
      for (const auto &polynomial : data)
      {
         const auto polynomial_independent =
            IndependentAxes<Dimension>(polynomial);
         for (int axis = 0; axis < Dimension; ++axis)
         {
            independent[axis] = independent[axis] &&
                                polynomial_independent[axis];
         }
      }
      for (int axis = 0; axis < Dimension; ++axis)
      {
         if (!independent[axis])
         {
            continue;
         }
         std::vector<AlgoimPolynomial> reduced;
         for (const auto &polynomial : data)
         {
            AlgoimPolynomial slice;
            slice.dimension = 2;
            slice.extent = polynomial.extent;
            int stride = 1;
            for (int previous = 0; previous < axis; ++previous)
            {
               stride *= polynomial.extent;
            }
            for (size_t index = 0;
                 index < polynomial.coefficients.size(); ++index)
            {
               if ((index / stride) % polynomial.extent == 0)
               {
                  slice.coefficients.push_back(polynomial.coefficients[index]);
               }
            }
            reduced.push_back(std::move(slice));
         }
         IntegrationRule plane;
         auto report = ComputePolynomialRule<2>(reduced, signs, domain, order,
                                                plane, region);
         if (!report.Succeeded())
         {
            return report;
         }
         const auto &line = IntRules.Get(Geometry::SEGMENT, order);
         if (plane.GetNPoints() >
             std::numeric_limits<int>::max() / line.GetNPoints())
         {
            return {CutQuadratureReport::Status::NumericalFailure,
                    "Too many extruded ALGOIM nodes."};
         }
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
                  coordinates[dimension] =
                     dimension == axis ? extrusion.x :
                     (coordinate++ == 0 ? point.x : point.y);
               }
               rule.IntPoint(output++).Set3w(
                  coordinates[0], coordinates[1], coordinates[2],
                  point.weight * extrusion.weight);
            }
         return {};
      }
   }
   std::array<std::vector<algoim::real>, 2> storage;
   std::vector<std::unique_ptr<
      algoim::xarray<algoim::real, Dimension>>> polynomials;
   for (size_t index = 0; index < data.size(); ++index)
   {
      const auto &polynomial = data[index];
      storage[index].resize(polynomial.coefficients.size());
      polynomials.emplace_back(
         new algoim::xarray<algoim::real, Dimension>(
            storage[index].data(),
            algoim::uvector<int, Dimension>(polynomial.extent)));
      auto &target = *polynomials.back();
      CopyPolynomialCoefficients(polynomial, target);
      std::vector<algoim::real> reduced_storage = storage[index];
      algoim::xarray<algoim::real, Dimension> reduced(
         reduced_storage.data(), target.ext());
      if (algoim::bernstein::autoReduction(reduced))
      {
         auto extents = reduced.ext();
         for (int axis = 0; axis < Dimension; ++axis)
         {
            extents(axis) = std::max(2, extents(axis));
         }
         target.alterExtent(extents);
         algoim::bernstein::bernsteinElevate(reduced, target);
      }
   }
   std::unique_ptr<algoim::ImplicitPolyQuadrature<Dimension>> quadrature;
   if (data.size() == 1)
   {
      quadrature.reset(new algoim::ImplicitPolyQuadrature<Dimension>
                       (*polynomials[0]));
   }
   else
   {
      quadrature.reset(
         new algoim::ImplicitPolyQuadrature<Dimension>(
            *polynomials[0], *polynomials[1]));
   }
   std::vector<IntegrationPoint> nodes;
   using AlgoimPoint = algoim::uvector<algoim::real, Dimension>;
   auto append = [&](const AlgoimPoint &coordinates, algoim::real weight)
   {
      IntegrationPoint point;
      point.Init(0);
      point.x = coordinates(0);
      point.y = coordinates(1);
      if constexpr (Dimension == 3)
      {
         point.z = coordinates(2);
      }
      point.weight = weight;
      nodes.push_back(point);
   };
   static_assert(algoim::GaussQuad::p_max >= 100 &&
                 algoim::TanhSinhQuadrature::n_max >= 100,
                 "Update the polynomial ALGOIM order limit.");
   if (domain.Codimension())
   {
      quadrature->integrate_surf(
         algoim::AutoMixed, order / 2 + 1,
         [&](const auto &coordinates, algoim::real weight, const auto &)
      {
         append(coordinates, weight);
      });
   }
   else
   {
      quadrature->integrate(algoim::AutoMixed, order / 2 + 1,
                            [&](const auto &coordinates, algoim::real weight)
      {
         for (size_t index = 0; index < polynomials.size(); ++index)
         {
            const algoim::real value =
               algoim::bernstein::evalBernsteinPoly(*polynomials[index],
                                                     coordinates);
            if ((signs[index] == CutDomain::LevelSetSign::Positive &&
                 value <= 0) ||
                (signs[index] == CutDomain::LevelSetSign::Negative &&
                 value >= 0))
            {
               return;
            }
         }
         append(coordinates, weight);
      });
   }
   if (nodes.size() > size_t(std::numeric_limits<int>::max()))
   {
      return {CutQuadratureReport::Status::NumericalFailure,
              "Too many ALGOIM nodes."};
   }
   rule.SetSize(static_cast<int>(nodes.size()));
   for (int index = 0; index < rule.GetNPoints(); ++index)
   {
      rule.IntPoint(index) = nodes[index];
   }
   region = nodes.empty() ? Region::Empty : Region::Cut;
   return {};
}

CutQuadratureReport GenerateAlgoimPolynomial(
   Geometry::Type geometry, const CutDomain &domain, int order,
   IntegrationRule &rule, Region &region)
{
   std::vector<AlgoimPolynomial> data;
   std::vector<CutDomain::LevelSetSign> signs;
   for (int index = 0; index < domain.NumConstraints(); ++index)
   {
      const auto &constraint = domain.GetConstraint(index);
      AlgoimPolynomial polynomial;
      const auto *source = dynamic_cast<const PolynomialLevelSet *>(
                              &constraint.level_set.get());
      if (!source)
      {
         return {CutQuadratureReport::Status::Unsupported,
                 "Polynomial ALGOIM requires polynomial inputs."};
      }
      polynomial.Set(*source);
      Region classification;
      auto report = polynomial.Classify(constraint.sign, classification);
      if (!report.Succeeded())
      {
         return report;
      }
      if (classification == Region::Empty)
      {
         region = Region::Empty;
         return {};
      }
      if (classification == Region::Full)
      {
         continue;
      }
      if (!data.empty() && polynomial.extent == data[0].extent)
      {
         bool same = true, opposite = true;
         for (size_t coefficient = 0;
              coefficient < polynomial.coefficients.size();
              ++coefficient)
         {
            const algoim::real first = data[0].coefficients[coefficient];
            const algoim::real second = polynomial.coefficients[coefficient];
            same = same && first == second;
            opposite = opposite && first == -second;
         }
         if (same || opposite)
         {
            if ((constraint.sign == signs[0]) != same)
            {
               region = Region::Empty;
               return {};
            }
            continue;
         }
      }
      if (!data.empty() && (geometry == Geometry::SQUARE ?
                            NearlyProportional<2>(data[0], polynomial) :
                            NearlyProportional<3>(data[0], polynomial)))
      {
         return {CutQuadratureReport::Status::NumericalFailure,
                 "Nearly proportional polynomial constraints cannot be "
                 "resolved reliably."};
      }
      data.push_back(std::move(polynomial));
      signs.push_back(constraint.sign);
   }
   if (data.empty())
   {
      rule = IntRules.Get(geometry, order);
      region = Region::Full;
      return {};
   }
   return geometry == Geometry::SQUARE ?
          ComputePolynomialRule<2>(data, signs, domain, order, rule, region) :
          ComputePolynomialRule<3>(data, signs, domain, order, rule, region);
}
#endif

enum class Algorithm
{
   Automatic,
   General,
   Polynomial
};

class AlgoimBackend : public CutQuadratureBackend
{
public:
   explicit AlgoimBackend(Algorithm algorithm) : algorithm(algorithm)
   {
   }

   std::string Name() const override
   {
      return algorithm == Algorithm::General ? "algoim-general" :
             (algorithm == Algorithm::Polynomial ? "algoim-polynomial" :
              "algoim-auto");
   }

   CutQuadratureReport CheckSupport(
      Geometry::Type geometry, const CutDomain &domain, int order,
      const IntegrationRule *fixed_points) const override
   {
      auto report = domain.Validate(geometry);
      if (!report.Succeeded())
      {
         return report;
      }
      if (order < 0)
      {
         return {CutQuadratureReport::Status::InvalidInput,
                 "Negative quadrature order."};
      }
      if (geometry != Geometry::SQUARE && geometry != Geometry::CUBE)
      {
         return {CutQuadratureReport::Status::Unsupported,
                 "ALGOIM supports quads and hexes."};
      }
      if (fixed_points)
      {
         return {CutQuadratureReport::Status::Unsupported,
                 "Unsupported ALGOIM prescribed nodes."};
      }
      const int count = domain.NumConstraints();
      const bool polynomial = UsePolynomial(domain);
      if (polynomial)
      {
#ifndef MFEM_USE_LAPACKE
         return {CutQuadratureReport::Status::Unsupported,
                 "Polynomial ALGOIM requires LAPACKE."};
#else
         if (count > 2 || (count > 1 && domain.Codimension()))
         {
            return {CutQuadratureReport::Status::Unsupported,
                    "Polynomial ALGOIM supports two signs or one surface."};
         }
         for (int index = 0; index < count; ++index)
         {
            if (!dynamic_cast<const PolynomialLevelSet *>(
                   &domain.GetConstraint(index).level_set.get()))
            {
               return {CutQuadratureReport::Status::Unsupported,
                       "Polynomial ALGOIM requires polynomial inputs."};
            }
         }
#endif
      }
      else
      {
         if (count > 1)
         {
            return {CutQuadratureReport::Status::Unsupported,
                    "General ALGOIM supports one active level set."};
         }
         if (count && !SupportsAlgoimIntervals(
                domain.GetConstraint(0).level_set.get()))
         {
            return {CutQuadratureReport::Status::Unsupported,
                    "General ALGOIM requires interval-capable inputs."};
         }
      }
      if (order / 2 + 1 > (polynomial ? 100 : 10))
      {
         return {CutQuadratureReport::Status::Unsupported,
                 "Quadrature order exceeds the selected ALGOIM limit."};
      }
      return {};
   }

   CutQuadratureReport Generate(
      Geometry::Type geometry, const CutDomain &domain, int order,
      CutQuadrature &output,
      const IntegrationRule *fixed_points) const override
   {
      output.Clear();
      IntegrationRule &rule = output.rule;
      Region &region = output.region;
      auto report = CheckSupport(geometry, domain, order, fixed_points);
      if (!report.Succeeded())
      {
         return report;
      }
      try
      {
         if (!domain.NumConstraints())
         {
            rule = IntRules.Get(geometry, order);
            region = Region::Full;
         }
#ifdef MFEM_USE_LAPACKE
         else if (UsePolynomial(domain))
         {
            report = GenerateAlgoimPolynomial(
                        geometry, domain, order, rule, region);
         }
#endif
         else
         {
            report = GenerateAlgoimGeneral(
                        geometry, domain, order, rule, region);
         }
      }
      catch (const std::exception &error)
      {
         report = {CutQuadratureReport::Status::NumericalFailure, error.what()};
      }
      if (report.Succeeded())
      {
         for (int index = 0; index < rule.GetNPoints(); ++index)
         {
            const auto &point = rule.IntPoint(index);
            if (!std::isfinite(point.weight) || !std::isfinite(point.x) ||
                !std::isfinite(point.y) ||
                (geometry == Geometry::CUBE && !std::isfinite(point.z)) ||
                !Geometry::CheckPoint(geometry, point,
                                      128 *
                                      std::numeric_limits<real_t>::epsilon()))
            {
               report = {CutQuadratureReport::Status::NumericalFailure,
                         "Invalid ALGOIM quadrature node."};
               break;
            }
         }
      }
      if (!report.Succeeded())
      {
         output.Clear();
      }
      else
      {
         rule.SetOrder(order);
         for (int index = 0; index < rule.GetNPoints(); ++index)
         {
            rule.IntPoint(index).index = index;
         }
      }
      return report;
   }

private:
   Algorithm algorithm;

   bool UsePolynomial(const CutDomain &domain) const
   {
      if (algorithm != Algorithm::Automatic)
      {
         return algorithm == Algorithm::Polynomial;
      }
#ifdef MFEM_USE_LAPACKE
      for (int index = 0; index < domain.NumConstraints(); ++index)
      {
         if (!dynamic_cast<const PolynomialLevelSet *>(
                &domain.GetConstraint(index).level_set.get()))
         {
            return false;
         }
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
   if (name == "moment")
   {
      return CreateMomentCutQuadratureBackend();
   }
   if (name == "algoim-auto")
   {
      return std::unique_ptr<CutQuadratureBackend>
             (new AlgoimBackend(Algorithm::Automatic));
   }
   if (name == "algoim-general")
   {
      return std::unique_ptr<CutQuadratureBackend>
             (new AlgoimBackend(Algorithm::General));
   }
#ifdef MFEM_USE_LAPACKE
   if (name == "algoim-polynomial")
   {
      return std::unique_ptr<CutQuadratureBackend>
             (new AlgoimBackend(Algorithm::Polynomial));
   }
#endif
   return nullptr;
}

}
#endif
