#include "cut_quadrature.hpp"
#include "fe_coll.hpp"
#include "../mesh/mesh.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace mfem
{
namespace
{
CutQuadratureReport Failure(CutQuadratureStatus status, const std::string &message)
{ return {status, message}; }

bool ValidGeometry(Geometry::Type geometry)
{ return geometry >= Geometry::POINT && geometry < Geometry::NUM_GEOMETRIES; }

bool FiniteVector(const Vector &values)
{
   for (int index = 0; index < values.Size(); ++index)
   { if (!std::isfinite(values(index))) { return false; } }
   return true;
}

bool ValidPoint(Geometry::Type geometry, const IntegrationPoint &point)
{
   const int dimension = Geometry::Dimension[geometry];
   return std::isfinite(point.x) &&
          (dimension < 2 || std::isfinite(point.y)) &&
          (dimension < 3 || std::isfinite(point.z)) &&
          Geometry::CheckPoint(geometry, point,
                               128 * std::numeric_limits<real_t>::epsilon());
}

bool SupportedElement(const FiniteElement &element)
{
   const auto geometry = element.GetGeomType();
   if (geometry != Geometry::SQUARE && geometry != Geometry::CUBE) { return false; }
   if (element.GetRangeType() != FiniteElement::SCALAR ||
       element.GetMapType() != FiniteElement::VALUE ||
       element.Space() != FunctionSpace::Qk || element.HasAnisotropicOrders() ||
       element.GetOrder() < 0 || element.GetOrder() == std::numeric_limits<int>::max())
   { return false; }
   int count = 1;
   for (int axis = 0; axis < element.GetDim(); ++axis)
   {
      if (count > std::numeric_limits<int>::max() / (element.GetOrder() + 1))
      { return false; }
      count *= element.GetOrder() + 1;
   }
   return element.GetDof() == count &&
          (dynamic_cast<const NodalFiniteElement *>(&element) ||
           dynamic_cast<const PositiveTensorFiniteElement *>(&element));
}

class PointGuard
{
public:
   explicit PointGuard(ElementTransformation &transformation)
      : transformation(transformation), point(transformation.GetIntPointPointer()) { }
   ~PointGuard() { transformation.SetIntPoint(point); }
private:
   ElementTransformation &transformation;
   const IntegrationPoint *point;
};

bool FullRank(const DenseMatrix &matrix)
{
   if (matrix.Width() > matrix.Height()) { return false; }
   DenseMatrix basis(matrix);
   real_t scale = 0;
   for (int column = 0; column < basis.Width(); ++column)
   {
      real_t norm = 0;
      for (int row = 0; row < basis.Height(); ++row)
      {
         if (!std::isfinite(basis(row, column))) { return false; }
         norm = std::hypot(norm, basis(row, column));
      }
      scale = std::max(scale, norm);
   }
   const real_t tolerance = 64 * std::numeric_limits<real_t>::epsilon() * scale;
   for (int column = 0; column < basis.Width(); ++column)
   {
      for (int previous = 0; previous < column; ++previous)
      {
         real_t product = 0;
         for (int row = 0; row < basis.Height(); ++row)
         { product += basis(row, previous) * basis(row, column); }
         for (int row = 0; row < basis.Height(); ++row)
         { basis(row, column) -= product * basis(row, previous); }
      }
      real_t norm = 0;
      for (int row = 0; row < basis.Height(); ++row)
      { norm = std::hypot(norm, basis(row, column)); }
      if (norm <= tolerance || !std::isfinite(norm)) { return false; }
      for (int row = 0; row < basis.Height(); ++row)
      { basis(row, column) /= norm; }
   }
   return true;
}

CutQuadratureReport FaceConormal(Geometry::Type geometry, int face, Vector &normal)
{
   const int dimension = Geometry::Dimension[geometry];
   normal.SetSize(dimension);
   normal = 0.;
   if (dimension == 1) { normal(0) = face == 0 ? -1 : 1; return {}; }
   const int *indices = nullptr;
   switch (geometry)
   {
      case Geometry::TRIANGLE:
         indices = Geometry::Constants<Geometry::TRIANGLE>::Edges[face]; break;
      case Geometry::SQUARE:
         indices = Geometry::Constants<Geometry::SQUARE>::Edges[face]; break;
      case Geometry::TETRAHEDRON:
         indices = Geometry::Constants<Geometry::TETRAHEDRON>::FaceVert[face]; break;
      case Geometry::CUBE:
         indices = Geometry::Constants<Geometry::CUBE>::FaceVert[face]; break;
      case Geometry::PRISM:
         indices = Geometry::Constants<Geometry::PRISM>::FaceVert[face]; break;
      case Geometry::PYRAMID:
         indices = Geometry::Constants<Geometry::PYRAMID>::FaceVert[face]; break;
      default:
         return Failure(CutQuadratureStatus::InvalidInput, "Invalid facet geometry.");
   }
   const IntegrationRule &points = *Geometries.GetVertices(geometry);
   Vector origin(dimension), first(dimension), second(dimension), center(dimension);
   points.IntPoint(indices[0]).Get(origin.GetData(), dimension);
   points.IntPoint(indices[1]).Get(first.GetData(), dimension);
   first -= origin;
   if (dimension == 2) { normal(0) = first(1); normal(1) = -first(0); }
   else
   {
      points.IntPoint(indices[2]).Get(second.GetData(), dimension);
      second -= origin;
      normal(0) = first(1)*second(2) - first(2)*second(1);
      normal(1) = first(2)*second(0) - first(0)*second(2);
      normal(2) = first(0)*second(1) - first(1)*second(0);
   }
   Geometries.GetCenter(geometry).Get(center.GetData(), dimension);
   center -= origin;
   if (normal * center > 0) { normal *= -1; }
   const real_t norm = normal.Norml2();
   if (norm == 0) { return Failure(CutQuadratureStatus::DegenerateGeometry, "Invalid reference facet."); }
   normal /= norm;
   return {};
}
}

int ReferenceLevelSet::GetDim() const
{ return ValidGeometry(GetGeometry()) ? Geometry::Dimension[GetGeometry()] : -1; }
bool ReferenceLevelSet::GetBounds(real_t &, real_t &) const { return false; }
const PolynomialLevelSet *ReferenceLevelSet::AsPolynomial() const { return nullptr; }

CutQuadratureReport PolynomialLevelSet::Set(const FiniteElement &new_element, const Vector &values)
{
   Vector copied(values);
   element = nullptr;
   dofs.SetSize(0);
   if (!SupportedElement(new_element))
   { return Failure(CutQuadratureStatus::Unsupported, "Unsupported polynomial finite element."); }
   if (copied.Size() != new_element.GetDof() || !FiniteVector(copied))
   { return Failure(CutQuadratureStatus::InvalidInput, "Invalid level-set dofs."); }
   dofs = copied;
   element = &new_element;
   return {};
}

CutQuadratureReport PolynomialLevelSet::Set(const GridFunction &function, int index)
{
   element = nullptr;
   dofs.SetSize(0);
   const FiniteElementSpace *space = function.FESpace();
   if (!space || space->GetVDim() != 1 || index < 0 || index >= space->GetNE())
   { return Failure(CutQuadratureStatus::InvalidInput, "Expected a local scalar grid function element."); }
   const FiniteElement &local_element = *space->GetFE(index);
   if (!SupportedElement(local_element))
   { return Failure(CutQuadratureStatus::Unsupported, "Unsupported grid-function space."); }
   Vector values;
   function.GetElementDofValues(index, values);
   return Set(local_element, values);
}

CutQuadratureReport PolynomialLevelSet::Project(Coefficient &coefficient,
   ElementTransformation &transformation, const FiniteElement &new_element)
{
   element = nullptr;
   dofs.SetSize(0);
   if (!SupportedElement(new_element))
   { return Failure(CutQuadratureStatus::Unsupported, "Unsupported projection space."); }
   if (transformation.GetGeometryType() != new_element.GetGeomType())
   { return Failure(CutQuadratureStatus::InvalidInput, "Projection geometry mismatch."); }
   PointGuard guard(transformation);
   L2_FECollection collection(new_element.GetOrder(), new_element.GetDim());
   const FiniteElement &nodal = *collection.FiniteElementForGeometry(new_element.GetGeomType());
   Vector values(nodal.GetDof());
   nodal.Project(coefficient, transformation, values);
   if (!FiniteVector(values))
   { return Failure(CutQuadratureStatus::NumericalFailure, "Nonfinite projected coefficient."); }
   IsoparametricTransformation identity;
   identity.SetIdentityTransformation(new_element.GetGeomType());
   DenseMatrix conversion(new_element.GetDof(), nodal.GetDof());
   new_element.Project(nodal, identity, conversion);
   Vector projected(new_element.GetDof());
   conversion.Mult(values, projected);
   return Set(new_element, projected);
}

const FiniteElement &PolynomialLevelSet::GetFE() const
{ MFEM_VERIFY(element, "Uninitialized polynomial level set."); return *element; }
Geometry::Type PolynomialLevelSet::GetGeometry() const
{ return element ? element->GetGeomType() : Geometry::INVALID; }
real_t PolynomialLevelSet::Eval(const IntegrationPoint &point) const
{
   Vector shape(GetFE().GetDof());
   element->CalcShape(point, shape);
   return shape * dofs;
}
void PolynomialLevelSet::EvalGradient(const IntegrationPoint &point, Vector &gradient) const
{
   const FiniteElement &finite_element = GetFE();
   DenseMatrix derivatives(finite_element.GetDof(), finite_element.GetDim());
   finite_element.CalcDShape(point, derivatives);
   gradient.SetSize(finite_element.GetDim());
   derivatives.MultTranspose(dofs, gradient);
}
const PolynomialLevelSet *PolynomialLevelSet::AsPolynomial() const
{ return element ? this : nullptr; }

CutQuadratureReport PolynomialLevelSet::GetBernsteinCoefficients(
   const FiniteElement &bernstein_element, Vector &coefficients) const
{
   coefficients.SetSize(0);
   if (!element) { return Failure(CutQuadratureStatus::InvalidInput, "Uninitialized level set."); }
   if (!SupportedElement(bernstein_element) ||
       !dynamic_cast<const PositiveTensorFiniteElement *>(&bernstein_element) ||
       bernstein_element.GetGeomType() != GetGeometry() ||
       bernstein_element.GetOrder() != element->GetOrder())
   { return Failure(CutQuadratureStatus::Unsupported, "Expected the matching native Bernstein space."); }
   if (const auto *positive = dynamic_cast<const PositiveTensorFiniteElement *>(element))
   {
      const auto &source_map = positive->GetDofMap();
      const auto &target_map =
         dynamic_cast<const PositiveTensorFiniteElement &>(bernstein_element).GetDofMap();
      coefficients.SetSize(dofs.Size());
      for (int index = 0; index < dofs.Size(); ++index)
      {
         coefficients(target_map.Size() ? target_map[index] : index) =
            dofs(source_map.Size() ? source_map[index] : index);
      }
      return {};
   }
   IsoparametricTransformation identity;
   identity.SetIdentityTransformation(GetGeometry());
   DenseMatrix conversion(bernstein_element.GetDof(), element->GetDof());
   bernstein_element.Project(*element, identity, conversion);
   coefficients.SetSize(bernstein_element.GetDof());
   conversion.Mult(dofs, coefficients);
   if (!FiniteVector(coefficients))
   {
      coefficients.SetSize(0);
      return Failure(CutQuadratureStatus::NumericalFailure, "Bernstein conversion failed.");
   }
   return {};
}

bool PolynomialLevelSet::GetBounds(real_t &lower, real_t &upper) const
{
   if (!element || !dynamic_cast<const PositiveTensorFiniteElement *>(element)) { return false; }
   lower = dofs.Min(); upper = dofs.Max();
   return true;
}

FunctionLevelSet::FunctionLevelSet(Geometry::Type new_geometry, ValueFunction new_value,
                                 GradientFunction new_gradient)
   : geometry(new_geometry), value(std::move(new_value)), gradient(std::move(new_gradient))
{ MFEM_VERIFY(ValidGeometry(geometry) && value && gradient, "Invalid level-set callbacks."); }
real_t FunctionLevelSet::Eval(const IntegrationPoint &point) const { return value(point); }
void FunctionLevelSet::EvalGradient(const IntegrationPoint &point, Vector &result) const
{ result.SetSize(GetDim()); gradient(point, result); }

CutDomain CutDomain::Positive(int level_set)
{ CutDomain domain; return domain.Add(level_set, LevelSetSign::Positive); }
CutDomain CutDomain::Negative(int level_set)
{ CutDomain domain; return domain.Add(level_set, LevelSetSign::Negative); }
CutDomain CutDomain::Surface(int level_set)
{ CutDomain domain; return domain.Add(level_set, LevelSetSign::Zero); }
CutDomain CutDomain::Intersection(int first, int second)
{ CutDomain domain = Surface(first); return domain.Add(second, LevelSetSign::Zero); }
CutDomain &CutDomain::Add(int level_set, LevelSetSign sign)
{ constraints.push_back({level_set, sign}); return *this; }
CutDomain &CutDomain::OnFace(int new_face) { face = new_face; return *this; }
int CutDomain::NumConstraints() const { return static_cast<int>(constraints.size()); }
const CutDomain::Constraint &CutDomain::GetConstraint(int index) const { return constraints.at(index); }
int CutDomain::Codimension() const
{
   int result = face >= 0 ? 1 : 0;
   for (const auto &constraint : constraints) { result += constraint.sign == LevelSetSign::Zero; }
   return result;
}
CutQuadratureReport CutDomain::Validate(Geometry::Type geometry, int count) const
{
   if (!ValidGeometry(geometry) || count < 0 || face < -1)
   { return Failure(CutQuadratureStatus::InvalidInput, "Invalid geometry or facet."); }
   if (face >= Geometry::NumBdrArray[geometry] || Codimension() > Geometry::Dimension[geometry])
   { return Failure(CutQuadratureStatus::InvalidInput, "Invalid facet or codimension."); }
   std::vector<bool> used(count, false);
   for (const auto &constraint : constraints)
   {
      if (constraint.level_set < 0 || constraint.level_set >= count ||
          (constraint.sign != LevelSetSign::Positive && constraint.sign != LevelSetSign::Negative &&
           constraint.sign != LevelSetSign::Zero))
      { return Failure(CutQuadratureStatus::InvalidInput, "Invalid level-set constraint."); }
      if (used[constraint.level_set])
      { return Failure(CutQuadratureStatus::InvalidInput, "Repeated level-set constraint."); }
      used[constraint.level_set] = true;
   }
   return {};
}

void CutQuadrature::Clear()
{
   geometry = Geometry::INVALID; domain = CutDomain(); region = CutRegion::Unknown;
   rule.SetSize(0); rule.SetOrder(-1); conormals.SetSize(0, 0, 0);
}
void MappedCutQuadrature::Clear()
{
   region = CutRegion::Unknown; weight_type = CutWeightType::Reference;
   rule.SetSize(0); rule.SetOrder(-1); physical_points.SetSize(0);
   physical_conormals.SetSize(0, 0, 0); measure_scaling.SetSize(0);
}
CutQuadratureGenerator::CutQuadratureGenerator(const CutQuadratureBackend &implementation)
   : backend(implementation) { }

CutQuadratureReport CutQuadratureGenerator::Generate(Geometry::Type geometry,
   const Array<const ReferenceLevelSet *> &level_sets,
   const CutQuadratureRequest &request, CutQuadrature &out) const
{
   out.Clear();
   auto report = request.domain.Validate(geometry, level_sets.Size());
   if (!report.Succeeded()) { return report; }
   if (request.order < 0) { return Failure(CutQuadratureStatus::InvalidInput, "Negative quadrature order."); }
   if (request.fixed_points)
   {
      if (request.domain.Codimension() || !request.fixed_points->GetNPoints())
      { return Failure(CutQuadratureStatus::InvalidInput, "Invalid prescribed volume nodes."); }
      for (int index = 0; index < request.fixed_points->GetNPoints(); ++index)
      {
         if (!ValidPoint(geometry, request.fixed_points->IntPoint(index)))
         { return Failure(CutQuadratureStatus::InvalidInput, "Invalid prescribed node."); }
      }
   }
   for (const auto *level_set : level_sets)
   {
      if (!level_set || level_set->GetGeometry() != geometry)
      { return Failure(CutQuadratureStatus::InvalidInput, "Level-set geometry mismatch."); }
   }
   report = backend.CheckSupport(geometry, level_sets, request);
   if (!report.Succeeded()) { return report; }
   report = backend.Generate(geometry, level_sets, request, out.rule, out.region);
   if (!report.Succeeded()) { out.Clear(); return report; }
   auto fail = [&](CutQuadratureStatus status, const char *message)
   { out.Clear(); return Failure(status, message); };
   if ((out.region != CutRegion::Full && out.region != CutRegion::Empty &&
        out.region != CutRegion::Cut) ||
       (out.region == CutRegion::Full &&
        request.domain.Codimension() > (request.domain.GetFace() >= 0 ? 1 : 0)))
   { return fail(CutQuadratureStatus::NumericalFailure, "Invalid backend region."); }
   if (request.fixed_points &&
       out.GetNPoints() != request.fixed_points->GetNPoints())
   { return fail(CutQuadratureStatus::NumericalFailure, "Backend changed prescribed nodes."); }
   out.geometry = geometry; out.domain = request.domain; out.rule.SetOrder(request.order);
   const int dimension = Geometry::Dimension[geometry];
   const int codimension = request.domain.Codimension();
   const int count = out.GetNPoints();
   if (codimension) { out.conormals.SetSize(dimension, codimension, count); out.conormals = 0.; }
   for (int index = 0; index < count; ++index)
   {
      auto &point = out.rule.IntPoint(index);
      point.index = index;
      if (!std::isfinite(point.weight) || !ValidPoint(geometry, point) ||
          (out.region == CutRegion::Empty && point.weight != 0))
      { return fail(CutQuadratureStatus::NumericalFailure, "Invalid quadrature node."); }
      if (request.fixed_points)
      {
         const auto &prescribed = request.fixed_points->IntPoint(index);
         if (point.x != prescribed.x ||
             (dimension > 1 && point.y != prescribed.y) ||
             (dimension > 2 && point.z != prescribed.z))
         { return fail(CutQuadratureStatus::NumericalFailure, "Backend moved a prescribed node."); }
      }
      if (point.weight == 0 || !codimension) { continue; }
      DenseMatrix normals(dimension, codimension);
      Vector normal(dimension);
      int column = 0;
      for (int constraint = 0; constraint < request.domain.NumConstraints(); ++constraint)
      {
         const auto &condition = request.domain.GetConstraint(constraint);
         if (condition.sign != LevelSetSign::Zero) { continue; }
         level_sets[condition.level_set]->EvalGradient(point, normal);
         if (normal.Size() != dimension || !FiniteVector(normal) || normal.Norml2() == 0)
         { return fail(CutQuadratureStatus::DegenerateGeometry, "Singular level-set gradient."); }
         normal /= normal.Norml2(); normals.SetCol(column++, normal);
      }
      if (request.domain.GetFace() >= 0)
      {
         auto face_report = FaceConormal(geometry, request.domain.GetFace(), normal);
         if (!face_report.Succeeded()) { out.Clear(); return face_report; }
         normals.SetCol(column, normal);
      }
      if (!FullRank(normals))
      { return fail(CutQuadratureStatus::DegenerateGeometry, "Dependent conormals."); }
      out.conormals(index) = normals;
   }
   return report;
}

CutQuadratureReport MapCutQuadrature(ElementTransformation &transformation,
   const CutQuadrature &reference, const CutMappingOptions &options, MappedCutQuadrature &out)
{
   out.Clear();
   if (!ValidGeometry(reference.geometry) || reference.geometry != transformation.GetGeometryType() ||
       reference.region == CutRegion::Unknown ||
       (options.data & ~(CQ_PHYSICAL_POINTS | CQ_PHYSICAL_CONORMALS | CQ_MEASURE_SCALING)) ||
       (options.weight_type != CutWeightType::Reference && options.weight_type != CutWeightType::Physical &&
        options.weight_type != CutWeightType::Integrator))
   { return Failure(CutQuadratureStatus::InvalidInput, "Invalid mapping request."); }
   const int dimension = Geometry::Dimension[reference.geometry];
   const int space_dimension = transformation.GetSpaceDim();
   const int codimension = reference.domain.Codimension();
   const int count = reference.GetNPoints();
   if (dimension < 1 || space_dimension < dimension || codimension > dimension ||
       (codimension && (reference.conormals.SizeI() != dimension ||
                       reference.conormals.SizeJ() != codimension || reference.conormals.SizeK() != count)))
   { return Failure(CutQuadratureStatus::InvalidInput, "Invalid reference conormal dimensions."); }
   PointGuard guard(transformation);
   out.rule = reference.rule; out.region = reference.region; out.weight_type = options.weight_type;
   if (options.data & CQ_PHYSICAL_POINTS) { out.physical_points.SetSize(space_dimension, count); }
   if ((options.data & CQ_PHYSICAL_CONORMALS) && codimension)
   { out.physical_conormals.SetSize(space_dimension, codimension, count); out.physical_conormals = 0.; }
   if (options.data & CQ_MEASURE_SCALING) { out.measure_scaling.SetSize(count); out.measure_scaling = 0.; }
   auto fail = [&](CutQuadratureStatus status, const char *message)
   { out.Clear(); return Failure(status, message); };
   for (int index = 0; index < count; ++index)
   {
      const auto &point = reference.rule.IntPoint(index);
      if (!std::isfinite(point.weight) || !ValidPoint(reference.geometry, point))
      { return fail(CutQuadratureStatus::InvalidInput, "Invalid reference node or weight."); }
      if (options.data & CQ_PHYSICAL_POINTS)
      {
         Vector physical(space_dimension);
         transformation.Transform(point, physical);
         if (!FiniteVector(physical))
         { return fail(CutQuadratureStatus::NumericalFailure, "Nonfinite physical point."); }
         out.physical_points.SetCol(index, physical);
      }
      if (point.weight == 0) { continue; }
      transformation.SetIntPoint(&point);
      const DenseMatrix &jacobian = transformation.Jacobian();
      if (!FullRank(jacobian))
      { return fail(CutQuadratureStatus::DegenerateGeometry, "Singular element mapping."); }
      DenseMatrix metric(dimension), inverse(dimension);
      MultAtB(jacobian, jacobian, metric);
      const real_t volume = std::sqrt(metric.Det());
      if (!(volume > 0) || !std::isfinite(volume))
      { return fail(CutQuadratureStatus::DegenerateGeometry, "Invalid element measure."); }
      real_t scaling = volume;
      if (codimension)
      {
         const DenseMatrix &normals = reference.conormals(index);
         if (!FullRank(normals))
         { return fail(CutQuadratureStatus::DegenerateGeometry, "Dependent reference conormals."); }
         CalcInverse(metric, inverse);
         DenseMatrix dual(dimension, codimension), physical(space_dimension, codimension);
         Mult(inverse, normals, dual);
         DenseMatrix gram(codimension), dual_gram(codimension);
         MultAtB(normals, normals, gram); MultAtB(normals, dual, dual_gram);
         scaling *= std::sqrt(dual_gram.Det() / gram.Det());
         if (options.data & CQ_PHYSICAL_CONORMALS)
         {
            Mult(jacobian, dual, physical);
            for (int column = 0; column < codimension; ++column)
            {
               Vector normal(space_dimension);
               physical.GetColumn(column, normal);
               if (!FiniteVector(normal) || normal.Norml2() == 0)
               { return fail(CutQuadratureStatus::DegenerateGeometry, "Invalid physical conormal."); }
               normal /= normal.Norml2(); physical.SetCol(column, normal);
            }
            out.physical_conormals(index) = physical;
         }
      }
      if (!(scaling > 0) || !std::isfinite(scaling))
      { return fail(CutQuadratureStatus::NumericalFailure, "Invalid manifold measure."); }
      if (options.data & CQ_MEASURE_SCALING) { out.measure_scaling(index) = scaling; }
      if (options.weight_type == CutWeightType::Physical) { out.rule.IntPoint(index).weight *= scaling; }
      else if (options.weight_type == CutWeightType::Integrator) { out.rule.IntPoint(index).weight *= scaling / volume; }
      if (!std::isfinite(out.rule.IntPoint(index).weight))
      { return fail(CutQuadratureStatus::NumericalFailure, "Nonfinite mapped weight."); }
   }
   return {};
}

CutQuadratureReport CutQuadratureGenerator::Generate(ElementTransformation &transformation,
   const Array<const ReferenceLevelSet *> &level_sets, const CutQuadratureRequest &request,
   const CutMappingOptions &mapping, MappedCutQuadrature &out) const
{
   out.Clear();
   CutQuadrature reference;
   auto report = Generate(transformation.GetGeometryType(), level_sets, request, reference);
   if (!report.Succeeded()) { return report; }
   auto mapped = MapCutQuadrature(transformation, reference, mapping, out);
   return mapped.Succeeded() ? report : mapped;
}

#ifndef MFEM_USE_ALGOIM
std::unique_ptr<CutQuadratureBackend> CreateCutQuadratureBackend(const std::string &) { return nullptr; }
std::vector<std::string> GetCutQuadratureBackends() { return {}; }
#endif
}
