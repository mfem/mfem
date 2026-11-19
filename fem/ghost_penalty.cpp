#include "ghost_penalty.hpp"

#ifdef MFEM_USE_LAPACK

#include "gridfunc.hpp"
#ifdef MFEM_USE_MPI
#include "pgridfunc.hpp"
#endif
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <sstream>
#include <vector>

namespace mfem
{
namespace
{

void CheckOptions(const ProjectionGhostPenaltyOptions &options)
{
   const auto space = options.projection_space;
   MFEM_VERIFY(space == ProjectionGhostPenaltySpace::AUTO ||
                  space == ProjectionGhostPenaltySpace::TOTAL_DEGREE ||
                  space == ProjectionGhostPenaltySpace::TENSOR_PRODUCT,
               "Ghost penalty: invalid projection space.");
   MFEM_VERIFY(std::isfinite(options.gamma) && options.gamma >= 0.0,
               "Ghost penalty: gamma must be finite and nonnegative.");
   MFEM_VERIFY(options.projection_degree >= -1 && options.quadrature_extra >= 0,
               "Ghost penalty: invalid projection degree or quadrature extra.");
   MFEM_VERIFY(std::isfinite(options.svd_rtol) && options.svd_rtol >= 0.0 &&
                  options.svd_rtol < 1.0,
               "Ghost penalty: SVD relative tolerance must be in [0,1).");
}

struct CellGeometry
{
   std::vector<real_t> key;
   Vector center;
   real_t volume;
   real_t diameter;
   real_t tolerance;
   DenseMatrix jacobian;
   int order;
   int weight_order;
   bool affine;
   real_t minimum_jacobian = infinity();
};

void PositiveSimplexRule(int dimension, int order, IntegrationRule &rule);

int CheckedSize(long long size, const std::string &context)
{
   MFEM_VERIFY(size >= 0 && size <= std::numeric_limits<int>::max(),
               "Ghost penalty: order or allocation size overflow; " << context);
   return static_cast<int>(size);
}

void PositiveGeometryRule(Geometry::Type geometry, int order,
                          IntegrationRule &rule, const std::string &context)
{
   const int dimension = Geometry::Dimension[geometry];
   CheckedSize(static_cast<long long>(order) + dimension, context);
   const long long points = (static_cast<long long>(order) + dimension + 2) / 2;
   int count = 1;
   for (int axis = 0; axis < dimension; ++axis)
   { count = CheckedSize(count * points, context); }
   if (geometry == Geometry::SQUARE || geometry == Geometry::CUBE)
   {
      IntegrationRules rules(0, Quadrature1D::GaussLegendre);
      rule = rules.Get(geometry, order);
   }
   else { PositiveSimplexRule(dimension, order, rule); }
}

real_t CheckJacobian(ElementTransformation &trans,
                     const IntegrationPoint &point, const std::string &context)
{
   trans.SetIntPoint(&point);
   DenseMatrix normalized(trans.Jacobian());
   const real_t norm = normalized.FNorm();
   MFEM_VERIFY(std::isfinite(norm) && norm > 0.0,
               "Ghost penalty: degenerate element geometry; " << context);
   normalized *= 1.0 / norm;
   const real_t quality = normalized.Det();
   MFEM_VERIFY(quality != 0.0,
               "Ghost penalty: degenerate element geometry; " << context);
   const real_t tolerance = 64.0 * trans.GetDimension() *
                            std::numeric_limits<real_t>::epsilon();
   MFEM_VERIFY(std::isfinite(quality) && quality > tolerance,
               "Ghost penalty: invalid signed Jacobian; " << context
               << " reference=(" << point.x << ',' << point.y << ','
               << point.z << ") normalized_det=" << quality
               << " tolerance=" << tolerance);
   return quality;
}

bool IsTensorGeometry(Geometry::Type geometry)
{
   return geometry == Geometry::SQUARE || geometry == Geometry::CUBE;
}

const char *ProjectionSpaceName(ProjectionGhostPenaltySpace space)
{
   return space == ProjectionGhostPenaltySpace::TENSOR_PRODUCT ? "q" : "p";
}

void CenterTransformation(const IsoparametricTransformation &source,
                          const Vector &offset,
                          IsoparametricTransformation &centered,
                          const std::string &context)
{
   centered.SetFE(source.GetFE());
   centered.ElementNo = source.ElementNo;
   centered.GetPointMat() = source.GetPointMat();
   auto &points = centered.GetPointMat();
   for (int node = 0; node < points.Width(); ++node)
   {
      for (int axis = 0; axis < points.Height(); ++axis)
      {
         MFEM_VERIFY(std::isfinite(points(axis, node)) &&
                        std::isfinite(offset(axis)),
                     "Ghost penalty: nonfinite physical geometry node; "
                     << context);
         points(axis, node) -= offset(axis);
      }
   }
}

CellGeometry GetGeometry(ElementTransformation &original,
                         const std::string &context = "selection preparation")
{
   const auto geometry = original.GetGeometryType();
   const int dimension = Geometry::Dimension[geometry];
   const bool tensor = IsTensorGeometry(geometry);
   const auto *isoparametric =
      dynamic_cast<const IsoparametricTransformation *>(&original);
   MFEM_VERIFY(
      original.GetSpaceDim() == dimension &&
      isoparametric && original.Order() >= 1 &&
      !dynamic_cast<const NURBSFiniteElement *>(isoparametric->GetFE()) &&
      (((geometry == Geometry::TRIANGLE || geometry == Geometry::TETRAHEDRON) &&
        isoparametric->GetFE()->Space() == FunctionSpace::Pk) ||
       (tensor && isoparametric->GetFE()->Space() == FunctionSpace::Qk)),
      "Ghost penalty: require polynomial triangles/tetrahedra or "
      "quadrilaterals/hexahedra in 2D/3D (no rational or embedded cells); "
         << context);
   Vector offset(dimension);
   for (int axis = 0; axis < dimension; ++axis)
   { offset(axis) = isoparametric->GetPointMat()(axis, 0); }
   IsoparametricTransformation centered;
   CenterTransformation(*isoparametric, offset, centered, context);
   ElementTransformation &trans = centered;
   const IntegrationRule &vertices = *Geometries.GetVertices(geometry);
   CellGeometry result;
   result.order = trans.Order();
   result.weight_order = CheckedSize(
      tensor ? static_cast<long long>(dimension) * result.order - 1 :
      static_cast<long long>(dimension) * (result.order - 1), context);
   result.affine = result.order == 1;
   result.center.SetSize(dimension);
   result.center = 0.0;
   std::vector<std::vector<real_t>> positions;
   Vector position(dimension);
   for (int vertex = 0; vertex < vertices.GetNPoints(); ++vertex)
   {
      trans.Transform(vertices.IntPoint(vertex), position);
      std::vector<real_t> coordinates(dimension);
      for (int axis = 0; axis < dimension; ++axis)
      {
         MFEM_VERIFY(std::isfinite(position(axis)),
                     "Ghost penalty: nonfinite physical vertex; " << context);
         coordinates[axis] = position(axis);
         result.center(axis) += position(axis) / vertices.GetNPoints();
      }
      positions.push_back(coordinates);
   }
   std::sort(positions.begin(), positions.end());
   result.diameter = 0.0;
   for (const auto &first : positions)
   {
      for (int axis = 0; axis < dimension; ++axis)
      { result.key.push_back(first[axis] + offset(axis)); }
      for (const auto &second : positions)
      {
         real_t distance = 0.0;
         for (int axis = 0; axis < dimension; ++axis)
         {
            distance = std::hypot(distance, first[axis] - second[axis]);
         }
         result.diameter = std::max(result.diameter, distance);
      }
   }
   trans.SetIntPoint(&Geometries.GetCenter(geometry));
   result.jacobian = trans.Jacobian();
   result.minimum_jacobian =
      CheckJacobian(trans, Geometries.GetCenter(geometry), context);
   result.volume = trans.Weight() * Geometry::Volume[geometry];
   MFEM_VERIFY(std::isfinite(result.volume) && result.volume > 0.0 &&
                  std::isfinite(result.diameter) && result.diameter > 0.0,
               "Ghost penalty: degenerate element geometry; " << context);
   result.tolerance = 256.0 * std::numeric_limits<real_t>::epsilon() *
                      result.diameter;
   if (tensor && result.affine)
   {
      Vector reference(dimension), displacement(dimension);
      const auto &center = Geometries.GetCenter(geometry);
      for (int vertex = 0; vertex < vertices.GetNPoints(); ++vertex)
      {
         const auto &point = vertices.IntPoint(vertex);
         reference(0) = point.x - center.x;
         reference(1) = point.y - center.y;
         if (dimension == 3) { reference(2) = point.z - center.z; }
         result.jacobian.Mult(reference, displacement);
         trans.Transform(point, position);
         position -= result.center;
         position -= displacement;
         result.affine = result.affine &&
                         position.Normlinf() <= result.tolerance;
      }
   }
   if (result.affine)
   {
      result.weight_order = 0;
      result.center += offset;
      return result;
   }
   const int refinement = CheckedSize(2LL * result.order, context);
   int sample_count = 1;
   for (int axis = 0; axis < dimension; ++axis)
   {
      sample_count = CheckedSize(
         static_cast<long long>(sample_count) * (refinement + 1LL), context);
   }
   CheckedSize(static_cast<long long>(sample_count) * vertices.GetNPoints(),
               context);
   GeometryRefiner refiner;
   const auto &samples = refiner.Refine(geometry, refinement)->RefPts;
   positions.clear();
   result.diameter = 0.0;
   for (int sample = 0; sample < samples.GetNPoints(); ++sample)
   {
      const auto &point = samples.IntPoint(sample);
      result.minimum_jacobian = std::min(result.minimum_jacobian,
                                         CheckJacobian(trans, point, context));
      trans.Transform(point, position);
      std::vector<real_t> coordinates(dimension);
      for (int axis = 0; axis < dimension; ++axis)
      {
         MFEM_VERIFY(std::isfinite(position(axis)),
                     "Ghost penalty: nonfinite physical point; " << context);
         coordinates[axis] = position(axis);
      }
      for (const auto &previous : positions)
      {
         real_t distance = 0.0;
         for (int axis = 0; axis < dimension; ++axis)
         { distance = std::hypot(distance, coordinates[axis] - previous[axis]); }
         result.diameter = std::max(result.diameter, distance);
      }
      positions.push_back(coordinates);
   }
   IntegrationRule metric_rule;
   PositiveGeometryRule(geometry,
                        CheckedSize(static_cast<long long>(result.order) +
                                    result.weight_order, context),
                        metric_rule, context);
   Vector anchor(result.center);
   result.center = 0.0;
   result.volume = 0.0;
   for (int sample = 0; sample < metric_rule.GetNPoints(); ++sample)
   {
      const auto &point = metric_rule.IntPoint(sample);
      result.minimum_jacobian = std::min(result.minimum_jacobian,
                                         CheckJacobian(trans, point, context));
      const real_t weight = point.weight * trans.Weight();
      MFEM_VERIFY(std::isfinite(weight) && weight > 0.0,
                  "Ghost penalty: invalid geometry quadrature weight; "
                  << context);
      trans.Transform(point, position);
      position -= anchor;
      result.center.Add(weight, position);
      result.volume += weight;
   }
   MFEM_VERIFY(std::isfinite(result.volume) && result.volume > 0.0 &&
                  std::isfinite(result.diameter) && result.diameter > 0.0,
               "Ghost penalty: degenerate element geometry; " << context);
   result.center /= result.volume;
   result.center += anchor;
   for (int axis = 0; axis < dimension; ++axis)
   {
      MFEM_VERIFY(std::isfinite(result.center(axis)),
                  "Ghost penalty: nonfinite cell centroid; " << context);
   }
   result.center += offset;
   return result;
}

void TensorFrameInverse(const CellGeometry &canonical,
                        const CellGeometry &neighbor, DenseMatrix &inverse,
                        const std::string &context)
{
   const int dimension = canonical.center.Size();
   std::vector<std::vector<real_t>> directions;
   for (int column = 0; column < dimension; ++column)
   {
      std::vector<real_t> direction(dimension);
      real_t length = 0.0;
      int pivot = 0;
      for (int axis = 0; axis < dimension; ++axis)
      {
         direction[axis] = canonical.jacobian(axis, column);
         length = std::hypot(length, direction[axis]);
         if (std::abs(direction[axis]) > std::abs(direction[pivot]))
         { pivot = axis; }
      }
      MFEM_VERIFY(std::isfinite(length) && length > 0.0,
                  "Ghost penalty: degenerate tensor frame; " << context);
      const real_t signed_length = direction[pivot] < 0.0 ? -length : length;
      for (auto &value : direction) { value /= signed_length; }
      directions.push_back(direction);
   }
   std::sort(directions.begin(), directions.end());
   std::vector<bool> used(dimension, false);
   const real_t tolerance = 512.0 * std::numeric_limits<real_t>::epsilon();
   for (int column = 0; column < dimension; ++column)
   {
      real_t length = 0.0;
      for (int axis = 0; axis < dimension; ++axis)
      { length = std::hypot(length, neighbor.jacobian(axis, column)); }
      MFEM_VERIFY(std::isfinite(length) && length > 0.0,
                  "Ghost penalty: degenerate tensor frame; " << context);
      real_t best = infinity();
      int match = -1;
      for (int candidate = 0; candidate < dimension; ++candidate)
      {
         real_t dot = 0.0, residual = 0.0;
         for (int axis = 0; axis < dimension; ++axis)
         {
            dot += directions[candidate][axis] * neighbor.jacobian(axis, column);
         }
         const real_t sign = dot < 0.0 ? -1.0 : 1.0;
         for (int axis = 0; axis < dimension; ++axis)
         {
            residual = std::hypot(residual,
                                 neighbor.jacobian(axis, column) / length -
                                 sign * directions[candidate][axis]);
         }
         if (residual < best) { best = residual; match = candidate; }
      }
      MFEM_VERIFY(match >= 0 && !used[match] && best <= tolerance,
                  "Ghost penalty: incompatible tensor frames; " << context
                  << " residual=" << best << " tolerance=" << tolerance
                  << "; request TOTAL_DEGREE explicitly to use P projection.");
      used[match] = true;
   }
   DenseMatrix frame(dimension);
   for (int column = 0; column < dimension; ++column)
   {
      for (int axis = 0; axis < dimension; ++axis)
      { frame(axis, column) = directions[column][axis]; }
   }
   DenseMatrixInverse factor(frame);
   factor.GetInverseMatrix(inverse);
   MFEM_VERIFY(std::isfinite(inverse.FNorm()),
               "Ghost penalty: degenerate tensor frame; " << context);
}

void PositiveSimplexRule(int dimension, int order, IntegrationRule &rule)
{
   IntegrationRules rules;
   const IntegrationRule &segment =
      rules.Get(Geometry::SEGMENT, order + dimension - 1);
   const int count = segment.GetNPoints();
   rule.SetSize(dimension == 2 ? count * count : count * count * count);
   rule.SetOrder(order);
   int point = 0;
   for (int first = 0; first < count; ++first)
   {
      const auto &first_ip = segment.IntPoint(first);
      for (int second = 0; second < count; ++second)
      {
         const auto &second_ip = segment.IntPoint(second);
         const real_t remaining = 1.0 - first_ip.x;
         if (dimension == 2)
         {
            rule.IntPoint(point++).Set2w(first_ip.x, remaining * second_ip.x,
                                         first_ip.weight * second_ip.weight *
                                            remaining);
         }
         else
         {
            for (int third = 0; third < count; ++third)
            {
               const auto &third_ip = segment.IntPoint(third);
               rule.IntPoint(point++).Set3w(
                  first_ip.x, remaining * second_ip.x,
                  remaining * (1.0 - second_ip.x) * third_ip.x,
                  first_ip.weight * second_ip.weight * third_ip.weight *
                     remaining * remaining * (1.0 - second_ip.x));
            }
         }
      }
   }
}

std::pair<int, int> FaceKey(const FaceElementTransformations &trans)
{
   return std::minmax(trans.Elem1No, trans.Elem2No);
}

std::string FaceContext(const GhostFaceSelection &selection,
                        const FaceElementTransformations &trans)
{
   std::ostringstream context;
   context << "rank=" << selection.GetRank() << " face elements=("
           << trans.Elem1No << ',' << trans.Elem2No << ')';
   return context.str();
}

}

void ProjectionGhostPenaltyKernel::ComputeProjection(
   const DenseMatrix &weighted_polynomials, const DenseMatrix &weighted_shapes,
   real_t relative_tolerance, DenseMatrix &coefficients,
   ProjectionGhostPenaltyDiagnostics &diagnostics, const std::string &context)
{
   const int rows = weighted_polynomials.Height();
   const int columns = weighted_polynomials.Width();
   MFEM_VERIFY(columns > 0 && rows >= columns &&
                  weighted_shapes.Height() == rows &&
                  weighted_shapes.Width() > 0,
               "Ghost penalty: invalid projection dimensions; "
                  << context << " nq=" << rows << " m=" << columns);
   MFEM_VERIFY(std::isfinite(relative_tolerance) && relative_tolerance >= 0.0 &&
                  relative_tolerance < 1.0,
               "Ghost penalty: invalid SVD tolerance; " << context);
   for (const auto *matrix : {&weighted_polynomials, &weighted_shapes})
   {
      for (int entry = 0; entry < matrix->Height() * matrix->Width(); ++entry)
      {
         MFEM_VERIFY(std::isfinite(matrix->Data()[entry]),
                     "Ghost penalty: nonfinite projection input; " << context);
      }
   }
   DenseMatrix input(weighted_polynomials);
   DenseMatrixSVD decomposition(rows, columns, 'S', 'S');
   decomposition.Eval(input);
   diagnostics.singular_values = decomposition.Singularvalues();
   diagnostics.quadrature_points = rows;
   diagnostics.relative_threshold =
      relative_tolerance == 0.0 ? 32.0 * std::max(rows, columns) *
                                     std::numeric_limits<real_t>::epsilon()
                                : relative_tolerance;
   const real_t maximum = diagnostics.singular_values(0);
   diagnostics.rank = 0;
   for (int index = 0; index < columns; ++index)
   {
      const real_t singular = diagnostics.singular_values(index);
      if (std::isfinite(singular) &&
          singular > diagnostics.relative_threshold * maximum)
      {
         ++diagnostics.rank;
      }
   }
   std::ostringstream spectrum;
   spectrum.precision(std::numeric_limits<real_t>::max_digits10);
   for (int index = 0; index < columns; ++index)
   {
      spectrum << ' ' << diagnostics.singular_values(index);
   }
   MFEM_VERIFY(
      std::isfinite(maximum) && maximum > 0.0 && diagnostics.rank == columns,
      "Ghost penalty: rank-deficient projection; "
         << context << " nq=" << rows << " m=" << columns << " rank="
         << diagnostics.rank << " rtol=" << diagnostics.relative_threshold
         << " singular values:" << spectrum.str());
   diagnostics.condition_number =
      maximum / diagnostics.singular_values(columns - 1);
   DenseMatrix transformed(columns, weighted_shapes.Width());
   MultAtB(decomposition.LeftSingularvectors(), weighted_shapes, transformed);
   for (int column = 0; column < transformed.Width(); ++column)
   {
      for (int row = 0; row < columns; ++row)
      {
         transformed(row, column) /= diagnostics.singular_values(row);
      }
   }
   coefficients.SetSize(columns, weighted_shapes.Width());
   MultAtB(decomposition.RightSingularvectors(), transformed, coefficients);
   for (int entry = 0; entry < coefficients.Height() * coefficients.Width();
        ++entry)
   {
      MFEM_VERIFY(std::isfinite(coefficients.Data()[entry]),
                  "Ghost penalty: nonfinite projection coefficients; "
                     << context);
   }
}

void ProjectionGhostPenaltyKernel::Assemble(
   const FiniteElement &first_fe, const FiniteElement &second_fe,
   ElementTransformation &first_trans, ElementTransformation &second_trans,
   const ProjectionGhostPenaltyOptions &options, DenseMatrix &matrix,
   ProjectionGhostPenaltyDiagnostics *diagnostics,
   const IntegrationRule *first_rule, const IntegrationRule *second_rule,
   const std::string &context)
{
   CheckOptions(options);
   const int dimension = first_fe.GetDim();
   MFEM_VERIFY(first_fe.GetGeomType() == first_trans.GetGeometryType() &&
                  second_fe.GetGeomType() == second_trans.GetGeometryType() &&
                  first_fe.GetDim() == second_fe.GetDim() &&
                  first_fe.GetRangeType() == FiniteElement::SCALAR &&
                  second_fe.GetRangeType() == FiniteElement::SCALAR &&
                  first_fe.GetMapType() == FiniteElement::VALUE &&
                  second_fe.GetMapType() == FiniteElement::VALUE,
               "Ghost penalty: incompatible finite elements; " << context);
   const auto *first_map =
      dynamic_cast<const IsoparametricTransformation *>(&first_trans);
   const auto *second_map =
      dynamic_cast<const IsoparametricTransformation *>(&second_trans);
   MFEM_VERIFY(first_map && second_map &&
                  first_trans.GetSpaceDim() == dimension &&
                  second_trans.GetSpaceDim() == dimension,
               "Ghost penalty: require polynomial volume transformations; "
               << context);
   Vector translation(dimension);
   translation = infinity();
   for (const auto *mapping : {first_map, second_map})
   {
      const auto &points = mapping->GetPointMat();
      for (int node = 0; node < points.Width(); ++node)
      {
         for (int axis = 0; axis < dimension; ++axis)
         { translation(axis) = std::min(translation(axis), points(axis, node)); }
      }
   }
   IsoparametricTransformation first_centered, second_centered;
   CenterTransformation(*first_map, translation, first_centered, context);
   CenterTransformation(*second_map, translation, second_centered, context);
   const CellGeometry first_geometry = GetGeometry(first_centered, context);
   const CellGeometry second_geometry = GetGeometry(second_centered, context);
   MFEM_VERIFY(first_fe.GetGeomType() == second_fe.GetGeomType(),
               "Ghost penalty: mixed-type patches are unsupported; "
               << context);
   const bool tensor = IsTensorGeometry(first_fe.GetGeomType());
   auto projection_space = options.projection_space;
   MFEM_VERIFY((first_geometry.affine && second_geometry.affine) ||
                  projection_space == ProjectionGhostPenaltySpace::TOTAL_DEGREE,
               "Ghost penalty: non-affine or higher-order geometry requires "
               "explicit TOTAL_DEGREE projection; " << context);
   if (projection_space == ProjectionGhostPenaltySpace::AUTO)
   {
      projection_space = tensor ? ProjectionGhostPenaltySpace::TENSOR_PRODUCT
                         : ProjectionGhostPenaltySpace::TOTAL_DEGREE;
   }
   const bool tensor_projection =
      projection_space == ProjectionGhostPenaltySpace::TENSOR_PRODUCT;
   MFEM_VERIFY(!tensor_projection || tensor,
               "Ghost penalty: tensor projection requires quads/hexes; "
               << context);
   const bool reverse = second_geometry.key < first_geometry.key;
   DenseMatrix inverse_frame;
   if (tensor_projection)
   {
      TensorFrameInverse(reverse ? second_geometry : first_geometry,
                         reverse ? first_geometry : second_geometry,
                         inverse_frame, context);
   }
   const FiniteElement *elements[] = {&first_fe, &second_fe};
   ElementTransformation *transforms[] = {&first_centered, &second_centered};
   const CellGeometry *geometries[] = {&first_geometry, &second_geometry};
   const IntegrationRule *quadratures[] = {first_rule, second_rule};
   if (reverse)
   {
      std::swap(elements[0], elements[1]);
      std::swap(transforms[0], transforms[1]);
      std::swap(quadratures[0], quadratures[1]);
      std::swap(geometries[0], geometries[1]);
   }
   const int degree = options.projection_degree < 0
                         ? std::min(first_fe.GetOrder(), second_fe.GetOrder())
                         : options.projection_degree;
   std::vector<std::array<int, 3>> powers;
   int columns = 1;
   for (int axis = 1; axis <= dimension; ++axis)
   {
      columns = CheckedSize(tensor_projection ?
                            columns * (degree + 1LL) :
                            columns * (degree + static_cast<long long>(axis)) /
                            axis, context);
   }
   if (tensor_projection)
   {
      for (int first = 0; first <= degree; ++first)
      {
         for (int second = 0; second <= degree; ++second)
         {
            for (int third = 0; third <= (dimension == 3 ? degree : 0); ++third)
            { powers.push_back({first, second, third}); }
         }
      }
   }
   else for (int total = 0; total <= degree; ++total)
   {
      for (int first = 0; first <= total; ++first)
      {
         for (int second = 0; second <= total - first; ++second)
         {
            if (dimension == 3 || first + second == total)
            {
               powers.push_back({first, second, total - first - second});
            }
         }
      }
   }
   ProjectionGhostPenaltyDiagnostics local_diagnostics;
   local_diagnostics.projection_space = projection_space;
   local_diagnostics.face_length =
      std::max(first_geometry.diameter, second_geometry.diameter);
   local_diagnostics.basis_length =
      std::pow(first_geometry.volume + second_geometry.volume, 1.0 / dimension);
   Vector origin(dimension);
   add(0.5, first_geometry.center, 0.5, second_geometry.center, origin);
   IntegrationRule generated[2];
   for (int side = 0; side < 2; ++side)
   {
      const auto &geometry = *geometries[side];
      MFEM_VERIFY(geometry.affine ||
                     ((elements[side]->Space() == FunctionSpace::Pk ||
                       elements[side]->Space() == FunctionSpace::Qk) &&
                      !dynamic_cast<const NURBSFiniteElement *>(
                         elements[side])),
                  "Ghost penalty: curved projection requires polynomial FEs; "
                  << context);
      const int mapped_degree = CheckedSize(
         static_cast<long long>(geometry.order) * degree, context);
      const int required_order = CheckedSize(
         2LL * std::max(elements[side]->GetOrder(), mapped_degree) +
         geometry.weight_order, context);
      const int order = CheckedSize(static_cast<long long>(required_order) +
                                    options.quadrature_extra, context);
      if (!quadratures[side])
      {
         PositiveGeometryRule(elements[side]->GetGeomType(), order,
                              generated[side], context);
         quadratures[side] = &generated[side];
      }
      MFEM_VERIFY(quadratures[side]->GetOrder() >= required_order,
                  "Ghost penalty: insufficient quadrature order; " << context);
      const int original_side = reverse ? 1 - side : side;
      local_diagnostics.geometry_orders[original_side] = geometry.order;
      local_diagnostics.required_orders[original_side] = required_order;
      local_diagnostics.quadrature_orders[original_side] =
         quadratures[side]->GetOrder();
      local_diagnostics.volumes[original_side] = geometry.volume;
      local_diagnostics.centroids[original_side] = geometry.center;
      local_diagnostics.centroids[original_side] += translation;
      local_diagnostics.minimum_jacobians[original_side] =
         geometry.minimum_jacobian;
   }
   const int rows = CheckedSize(
      static_cast<long long>(quadratures[0]->GetNPoints()) +
      quadratures[1]->GetNPoints(), context);
   const int dofs = CheckedSize(static_cast<long long>(first_fe.GetDof()) +
                               second_fe.GetDof(), context);
   CheckedSize(static_cast<long long>(rows) * columns, context);
   CheckedSize(static_cast<long long>(rows) * dofs, context);
   CheckedSize(static_cast<long long>(dofs) * dofs, context);
   DenseMatrix weighted_polynomials(rows, powers.size());
   DenseMatrix weighted_shapes(rows, dofs);
   weighted_shapes = 0.0;
   int row_offset = 0;
   int column_offset = 0;
   for (int side = 0; side < 2; ++side)
   {
      Vector shape(elements[side]->GetDof()), physical(dimension),
             tensor_coordinates(dimension);
      for (int point = 0; point < quadratures[side]->GetNPoints(); ++point)
      {
         const auto &integration_point = quadratures[side]->IntPoint(point);
         MFEM_VERIFY(std::isfinite(integration_point.weight) &&
                        integration_point.weight > 0.0,
                     "Ghost penalty: quadrature weight must be positive and "
                     "finite; " << context);
         const int original_side = reverse ? 1 - side : side;
         auto &minimum = local_diagnostics.minimum_jacobians[original_side];
         minimum = std::min(minimum, CheckJacobian(*transforms[side],
                                                  integration_point, context));
         const real_t weight =
            integration_point.weight * transforms[side]->Weight();
         MFEM_VERIFY(
            std::isfinite(weight) && weight > 0.0,
            "Ghost penalty: quadrature weight must be positive and finite; "
               << context << " side=" << side << " point=" << point);
         const real_t root_weight = std::sqrt(weight);
         elements[side]->CalcShape(integration_point, shape);
         transforms[side]->Transform(integration_point, physical);
         physical -= origin;
         physical /= local_diagnostics.basis_length;
         if (tensor_projection)
         {
            inverse_frame.Mult(physical, tensor_coordinates);
            physical = tensor_coordinates;
         }
         for (int basis = 0; basis < static_cast<int>(powers.size()); ++basis)
         {
            real_t value = root_weight;
            for (int axis = 0; axis < dimension; ++axis)
            {
               value *= std::pow(physical(axis), powers[basis][axis]);
            }
            weighted_polynomials(row_offset + point, basis) = value;
         }
         for (int basis = 0; basis < shape.Size(); ++basis)
         {
            weighted_shapes(row_offset + point, column_offset + basis) =
               root_weight * shape(basis);
         }
      }
      row_offset += quadratures[side]->GetNPoints();
      column_offset += elements[side]->GetDof();
   }
   std::ostringstream description;
   description << context << " p1=" << first_fe.GetOrder()
               << " p2=" << second_fe.GetOrder() << " degree=" << degree
               << " projection_space=" << ProjectionSpaceName(projection_space)
               << " geometry=" << first_fe.GetGeomType() << ','
               << second_fe.GetGeomType()
               << " geometry_orders=" << first_geometry.order << ','
               << second_geometry.order
               << " quadrature_orders=" << local_diagnostics.quadrature_orders[0]
               << ',' << local_diagnostics.quadrature_orders[1];
   DenseMatrix coefficients, projected(rows, dofs);
   ComputeProjection(weighted_polynomials, weighted_shapes, options.svd_rtol,
                     coefficients, local_diagnostics, description.str());
   Mult(weighted_polynomials, coefficients, projected);
   weighted_shapes -= projected;
   const real_t alpha = options.gamma / local_diagnostics.face_length /
                        local_diagnostics.face_length;
   MFEM_VERIFY(std::isfinite(alpha),
               "Ghost penalty: nonfinite scaling; " << context);
   matrix.SetSize(dofs);
   auto original_index = [&](int index)
   {
      if (!reverse) { return index; }
      return index < second_fe.GetDof() ? first_fe.GetDof() + index
                                        : index - second_fe.GetDof();
   };
   for (int column = 0; column < dofs; ++column)
   {
      for (int row = 0; row <= column; ++row)
      {
         real_t value = 0.0;
         for (int point = 0; point < rows; ++point)
         {
            value +=
               weighted_shapes(point, row) * weighted_shapes(point, column);
         }
         value *= alpha;
         MFEM_VERIFY(std::isfinite(value),
                     "Ghost penalty: nonfinite matrix entry; " << context);
         matrix(original_index(row), original_index(column)) = value;
         matrix(original_index(column), original_index(row)) = value;
      }
   }
   if (options.verbose)
   {
      mfem::out << "Ghost penalty: " << description.str() << " nq=" << rows
                << " rank=" << local_diagnostics.rank
                << " condition=" << local_diagnostics.condition_number
                << " h_F=" << local_diagnostics.face_length << std::endl;
   }
   if (diagnostics) { *diagnostics = local_diagnostics; }
}

GhostFaceSelection::GhostFaceSelection(FiniteElementSpace &fes,
                                       const Array<int> &active,
                                       const Array<int> &cut,
                                       int extension_layers)
    : space(&fes), mesh_sequence(fes.GetMesh()->GetSequence()),
      space_sequence(fes.GetSequence()), elements(fes.GetNE())
{
   Mesh &mesh = *fes.GetMesh();
   MFEM_VERIFY(!mesh.Nonconforming() && !fes.IsVariableOrder() &&
                  dynamic_cast<const H1_FECollection *>(fes.FEColl()) &&
                  fes.GetVDim() > 0,
               "Ghost penalty: require conforming uniform-order H1 spaces.");
   MFEM_VERIFY((mesh.Dimension() == 2 || mesh.Dimension() == 3) &&
                  mesh.Dimension() == mesh.SpaceDimension(),
               "Ghost penalty: require a 2D or 3D volume mesh.");
   const auto *nodes = mesh.GetNodes();
   MFEM_VERIFY(!nodes ||
                  (!nodes->FESpace()->IsVariableOrder() &&
                   dynamic_cast<const H1_FECollection *>(
                      nodes->FESpace()->FEColl())),
               "Ghost penalty: require conforming uniform-order H1 geometry.");
   MFEM_VERIFY(active.Size() == elements && cut.Size() == elements &&
                  extension_layers >= 0,
               "Ghost penalty: invalid flag sizes or extension layers.");
   Array<int> band(cut);
   for (int element = 0; element < elements; ++element)
   {
      MFEM_VERIFY((active[element] == 0 || active[element] == 1) &&
                     (cut[element] == 0 || cut[element] == 1),
                  "Ghost penalty: active/cut flags must be zero or one.");
      GetGeometry(*mesh.GetElementTransformation(element));
      band[element] = active[element] && cut[element];
   }
#ifdef MFEM_USE_MPI
   auto *parallel_space = dynamic_cast<ParFiniteElementSpace *>(&fes);
   auto *parallel_mesh = dynamic_cast<ParMesh *>(&mesh);
   MFEM_VERIFY(!parallel_mesh || parallel_space,
               "Ghost penalty: ParMesh requires ParFiniteElementSpace.");
   std::unique_ptr<L2_FECollection> marker_collection;
   std::unique_ptr<ParFiniteElementSpace> marker_space;
   std::unique_ptr<ParGridFunction> active_field, band_field;
   if (parallel_space)
   {
      MPI_Comm communicator = parallel_space->GetComm();
      MPI_Comm_rank(communicator, &rank);
      const int local_order = fes.FEColl()->GetOrder();
      int minimum_order, maximum_order;
      MPI_Allreduce(&local_order, &minimum_order, 1, MPI_INT, MPI_MIN,
                    communicator);
      MPI_Allreduce(&local_order, &maximum_order, 1, MPI_INT, MPI_MAX,
                    communicator);
      MFEM_VERIFY(
         minimum_order == maximum_order,
         "Ghost penalty: polynomial order must agree across MPI ranks.");
      int minimum_layers, maximum_layers;
      MPI_Allreduce(&extension_layers, &minimum_layers, 1, MPI_INT, MPI_MIN,
                    communicator);
      MPI_Allreduce(&extension_layers, &maximum_layers, 1, MPI_INT, MPI_MAX,
                    communicator);
      MFEM_VERIFY(
         minimum_layers == maximum_layers,
         "Ghost penalty: extension layers must agree across MPI ranks.");
      const int geometry_order = nodes ?
                                 nodes->FESpace()->FEColl()->GetOrder() : 1;
      int minimum_geometry, maximum_geometry;
      MPI_Allreduce(&geometry_order, &minimum_geometry, 1, MPI_INT, MPI_MIN,
                    communicator);
      MPI_Allreduce(&geometry_order, &maximum_geometry, 1, MPI_INT, MPI_MAX,
                    communicator);
      MFEM_VERIFY(minimum_geometry == maximum_geometry,
                  "Ghost penalty: geometry order must agree across MPI ranks.");
      parallel_mesh->ExchangeFaceNbrNodes();
      marker_collection.reset(new L2_FECollection(0, mesh.Dimension()));
      marker_space.reset(
         new ParFiniteElementSpace(parallel_mesh, marker_collection.get()));
      active_field.reset(new ParGridFunction(marker_space.get()));
      band_field.reset(new ParGridFunction(marker_space.get()));
   }
#endif
   for (int layer = 0; layer <= extension_layers; ++layer)
   {
      faces.clear();
      local_faces = shared_faces = 0;
      Array<int> expanded(band);
      auto insert = [&](FaceElementTransformations &trans, bool neighbor_active,
                        bool neighbor_cut, bool shared)
      {
         const int first = trans.Elem1No;
         const bool selected =
            active[first] && neighbor_active && (band[first] || neighbor_cut);
         MFEM_VERIFY(faces.emplace(FaceKey(trans), selected).second,
                     "Ghost penalty: ambiguous element-pair face key; "
                     "periodic/multiple-face pairings are unsupported.");
         if (selected)
         {
            if (shared) { ++shared_faces; }
            else
            {
               ++local_faces;
            }
            expanded[first] = 1;
            if (!shared) { expanded[trans.Elem2No] = 1; }
         }
      };
      for (int face = 0; face < mesh.GetNumFaces(); ++face)
      {
         auto *trans = mesh.GetInteriorFaceTransformations(face);
         if (trans)
         {
            insert(*trans, active[trans->Elem2No], band[trans->Elem2No], false);
         }
      }
#ifdef MFEM_USE_MPI
      if (parallel_space)
      {
         Array<int> dofs;
         for (int element = 0; element < elements; ++element)
         {
            marker_space->GetElementDofs(element, dofs);
            (*active_field)(dofs[0]) = active[element];
            (*band_field)(dofs[0]) = band[element];
         }
         active_field->ExchangeFaceNbrData();
         band_field->ExchangeFaceNbrData();
         Vector active_value, band_value;
         for (int face = 0; face < parallel_mesh->GetNSharedFaces(); ++face)
         {
            auto *trans = parallel_mesh->GetSharedFaceTransformations(face);
            active_field->GetElementDofValues(trans->Elem2No, active_value);
            band_field->GetElementDofValues(trans->Elem2No, band_value);
            GetGeometry(*trans->Elem2);
            insert(*trans, active_value(0) != 0.0, band_value(0) != 0.0, true);
         }
      }
#endif
      band = expanded;
   }
}

void GhostFaceSelection::Validate() const
{
   MFEM_VERIFY(
      space->GetSequence() == space_sequence &&
         space->GetMesh()->GetSequence() == mesh_sequence &&
         space->GetNE() == elements,
      "Ghost penalty: stale selection; rebuild after mesh/space changes.");
}

bool GhostFaceSelection::IsSelected(
   const FaceElementTransformations &trans) const
{
   Validate();
   const auto found = faces.find(FaceKey(trans));
   MFEM_VERIFY(
      found != faces.end(),
      "Ghost penalty: face not in prepared selection; rebuild metadata.");
   return found->second;
}

ScalarProjectionGhostPenaltyIntegrator::ScalarProjectionGhostPenaltyIntegrator(
   std::shared_ptr<const GhostFaceSelection> selected,
   const ProjectionGhostPenaltyOptions &configuration)
    : selection(std::move(selected)), options(configuration)
{
   MFEM_VERIFY(selection && selection->GetVDim() == 1,
               "Ghost penalty: scalar integrator requires vdim=1.");
   CheckOptions(options);
}

void ScalarProjectionGhostPenaltyIntegrator::AssembleFaceMatrix(
   const FiniteElement &first_fe, const FiniteElement &second_fe,
   FaceElementTransformations &trans, DenseMatrix &matrix)
{
   const auto context = FaceContext(*selection, trans);
   const int dofs = CheckedSize(static_cast<long long>(first_fe.GetDof()) +
                               second_fe.GetDof(), context);
   CheckedSize(static_cast<long long>(dofs) * dofs, context);
   matrix.SetSize(dofs);
   matrix = 0.0;
   if (!selection->IsSelected(trans)) { return; }
   ProjectionGhostPenaltyKernel::Assemble(
      first_fe, second_fe, *trans.Elem1, *trans.Elem2, options, matrix, nullptr,
      nullptr, nullptr, context);
}

VectorProjectionGhostPenaltyIntegrator::VectorProjectionGhostPenaltyIntegrator(
   std::shared_ptr<const GhostFaceSelection> selected,
   const ProjectionGhostPenaltyOptions &configuration)
    : selection(std::move(selected)), options(configuration)
{
   MFEM_VERIFY(selection && selection->GetVDim() >= 2,
               "Ghost penalty: vector integrator requires vdim>=2.");
   CheckOptions(options);
}

void VectorProjectionGhostPenaltyIntegrator::AssembleFaceMatrix(
   const FiniteElement &first_fe, const FiniteElement &second_fe,
   FaceElementTransformations &trans, DenseMatrix &matrix)
{
   const auto context = FaceContext(*selection, trans);
   const int first_dofs = first_fe.GetDof();
   const int second_dofs = second_fe.GetDof();
   const int dofs = CheckedSize(static_cast<long long>(first_dofs) + second_dofs,
                               context);
   const int components = selection->GetVDim();
   const int vector_dofs = CheckedSize(static_cast<long long>(components) * dofs,
                                      context);
   CheckedSize(static_cast<long long>(vector_dofs) * vector_dofs, context);
   matrix.SetSize(vector_dofs);
   matrix = 0.0;
   if (!selection->IsSelected(trans)) { return; }
   DenseMatrix scalar;
   ProjectionGhostPenaltyKernel::Assemble(
      first_fe, second_fe, *trans.Elem1, *trans.Elem2, options, scalar, nullptr,
      nullptr, nullptr, context);
   for (int component = 0; component < components; ++component)
   {
      auto index = [&](int scalar_index)
      {
         return scalar_index < first_dofs
                   ? component * first_dofs + scalar_index
                   : components * first_dofs + component * second_dofs +
                        scalar_index - first_dofs;
      };
      for (int row = 0; row < dofs; ++row)
      {
         for (int column = 0; column < dofs; ++column)
         {
            matrix(index(row), index(column)) = scalar(row, column);
         }
      }
   }
}

}

#endif
