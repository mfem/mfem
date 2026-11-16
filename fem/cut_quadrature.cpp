#include "cut_quadrature.hpp"
#include "fe_coll.hpp"
#include "../mesh/mesh.hpp"

#include <algorithm>
#include <cmath>
#include <exception>
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

void ConvertNodalDofs(const FiniteElement &source, const Vector &values,
                      const FiniteElement &target, Vector &converted)
{
   converted.SetSize(target.GetDof());
   if (dynamic_cast<const PositiveTensorFiniteElement *>(&target))
   {
      DenseMatrix interpolation(source.GetDof(), target.GetDof());
      Vector shape(target.GetDof());
      for (int index = 0; index < source.GetDof(); ++index)
      {
         target.CalcShape(source.GetNodes().IntPoint(index), shape);
         interpolation.SetRow(index, shape);
      }
      DenseMatrixInverse inverse(interpolation);
      inverse.Mult(values, converted);
   }
   else
   {
      Vector shape(source.GetDof());
      for (int index = 0; index < target.GetDof(); ++index)
      {
         source.CalcShape(target.GetNodes().IntPoint(index), shape);
         converted(index) = shape * values;
      }
   }
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

bool FactorQR(const DenseMatrix &matrix, DenseMatrix &basis,
              DenseMatrix &triangular, real_t &measure)
{
   if (matrix.Width() > matrix.Height()) { return false; }
   basis = matrix;
   triangular.SetSize(matrix.Width());
   triangular = 0.;
   measure = 1;
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
      for (int pass = 0; pass < 2; ++pass)
      {
         for (int previous = 0; previous < column; ++previous)
         {
            real_t product = 0;
            for (int row = 0; row < basis.Height(); ++row)
            { product += basis(row, previous) * basis(row, column); }
            triangular(previous, column) += product;
            for (int row = 0; row < basis.Height(); ++row)
            { basis(row, column) -= product * basis(row, previous); }
         }
      }
      real_t norm = 0;
      for (int row = 0; row < basis.Height(); ++row)
      { norm = std::hypot(norm, basis(row, column)); }
      if (norm <= tolerance || !std::isfinite(norm)) { return false; }
      triangular(column, column) = norm;
      measure *= norm;
      for (int row = 0; row < basis.Height(); ++row)
      { basis(row, column) /= norm; }
   }
   return true;
}

bool FullRank(const DenseMatrix &matrix)
{
   DenseMatrix basis, triangular;
   real_t measure;
   return FactorQR(matrix, basis, triangular, measure);
}

void SolveUpperTranspose(const DenseMatrix &triangular,
                         const DenseMatrix &right_hand_side, DenseMatrix &solution)
{
   solution.SetSize(right_hand_side.Height(), right_hand_side.Width());
   for (int column = 0; column < solution.Width(); ++column)
   {
      for (int row = 0; row < solution.Height(); ++row)
      {
         real_t value = right_hand_side(row, column);
         for (int previous = 0; previous < row; ++previous)
         { value -= triangular(previous, row) * solution(previous, column); }
         solution(row, column) = value / triangular(row, row);
      }
   }
}

}

int ReferenceLevelSet::GetDim() const
{
   return ValidGeometry(GetReferenceGeometry()) ?
          Geometry::Dimension[GetReferenceGeometry()] : -1;
}

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
   Vector projected(new_element.GetDof());
   ConvertNodalDofs(nodal, values, new_element, projected);
   return Set(new_element, projected);
}

const FiniteElement &PolynomialLevelSet::GetFE() const
{ MFEM_VERIFY(element, "Uninitialized polynomial level set."); return *element; }
Geometry::Type PolynomialLevelSet::GetReferenceGeometry() const
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
CutQuadratureReport PolynomialLevelSet::GetBernsteinCoefficients(
   const FiniteElement &bernstein_element, Vector &coefficients) const
{
   coefficients.SetSize(0);
   if (!element) { return Failure(CutQuadratureStatus::InvalidInput, "Uninitialized level set."); }
   if (!SupportedElement(bernstein_element) ||
       !dynamic_cast<const PositiveTensorFiniteElement *>(&bernstein_element) ||
       bernstein_element.GetGeomType() != GetReferenceGeometry() ||
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
   ConvertNodalDofs(*element, dofs, bernstein_element, coefficients);
   if (!FiniteVector(coefficients))
   {
      coefficients.SetSize(0);
      return Failure(CutQuadratureStatus::NumericalFailure, "Bernstein conversion failed.");
   }
   return {};
}

FunctionLevelSet::FunctionLevelSet(Geometry::Type new_geometry, ValueFunction new_value,
                                 GradientFunction new_gradient)
   : geometry(new_geometry), value(std::move(new_value)), gradient(std::move(new_gradient))
{ MFEM_VERIFY(ValidGeometry(geometry) && value && gradient, "Invalid level-set callbacks."); }
real_t FunctionLevelSet::Eval(const IntegrationPoint &point) const { return value(point); }
void FunctionLevelSet::EvalGradient(const IntegrationPoint &point, Vector &result) const
{ result.SetSize(GetDim()); gradient(point, result); }

CutDomain CutDomain::Positive(const ReferenceLevelSet &level_set)
{ CutDomain domain; return domain.Add(level_set, LevelSetSign::Positive); }
CutDomain CutDomain::Negative(const ReferenceLevelSet &level_set)
{ CutDomain domain; return domain.Add(level_set, LevelSetSign::Negative); }
CutDomain CutDomain::Surface(const ReferenceLevelSet &level_set)
{ CutDomain domain; return domain.Add(level_set, LevelSetSign::Zero); }
CutDomain &CutDomain::Add(const ReferenceLevelSet &level_set, LevelSetSign sign)
{ constraints.push_back({level_set, sign}); return *this; }
int CutDomain::NumConstraints() const { return static_cast<int>(constraints.size()); }
const CutDomain::Constraint &CutDomain::GetConstraint(int index) const { return constraints.at(index); }
int CutDomain::Codimension() const
{
   int result = 0;
   for (const auto &constraint : constraints)
   { result += constraint.sign == LevelSetSign::Zero; }
   return result;
}
CutQuadratureReport CutDomain::Validate(Geometry::Type geometry) const
{
   if (!ValidGeometry(geometry))
   { return Failure(CutQuadratureStatus::InvalidInput, "Invalid geometry."); }
   if (Codimension() > Geometry::Dimension[geometry])
   { return Failure(CutQuadratureStatus::InvalidInput, "Invalid codimension."); }
   for (int index = 0; index < NumConstraints(); ++index)
   {
      const auto &constraint = constraints[index];
      if (constraint.sign != LevelSetSign::Positive &&
          constraint.sign != LevelSetSign::Negative &&
          constraint.sign != LevelSetSign::Zero)
      { return Failure(CutQuadratureStatus::InvalidInput, "Invalid level-set constraint."); }
      if (constraint.level_set.get().GetReferenceGeometry() != geometry)
      { return Failure(CutQuadratureStatus::InvalidInput, "Level-set geometry mismatch."); }
      for (int previous = 0; previous < index; ++previous)
      {
         if (&constraints[previous].level_set.get() == &constraint.level_set.get())
         { return Failure(CutQuadratureStatus::InvalidInput, "Repeated level-set constraint."); }
      }
   }
   return {};
}

void CutQuadrature::Clear()
{
   geometry = Geometry::INVALID; codimension = 0; region = Region::Unknown;
   rule.SetSize(0); rule.SetOrder(-1); conormals.SetSize(0, 0, 0);
}
void CutQuadratureMapping::Clear()
{
   physical_points.SetSize(0);
   physical_conormals.SetSize(0, 0, 0); measure_scaling.SetSize(0);
}
CutQuadratureGenerator::CutQuadratureGenerator(const CutQuadratureBackend &implementation)
   : backend(implementation) { }

CutQuadratureReport CutQuadratureGenerator::Generate(Geometry::Type geometry,
   const CutDomain &domain, int order, CutQuadrature &out,
   const IntegrationRule *fixed_points) const try
{
   out.Clear();
   auto report = domain.Validate(geometry);
   if (!report.Succeeded()) { return report; }
   if (order < 0)
   { return Failure(CutQuadratureStatus::InvalidInput, "Negative quadrature order."); }
   if (fixed_points)
   {
      if (domain.Codimension() || !fixed_points->GetNPoints())
      { return Failure(CutQuadratureStatus::InvalidInput, "Invalid prescribed volume nodes."); }
      for (int index = 0; index < fixed_points->GetNPoints(); ++index)
      {
         if (!ValidPoint(geometry, fixed_points->IntPoint(index)))
         { return Failure(CutQuadratureStatus::InvalidInput, "Invalid prescribed node."); }
      }
   }
   report = backend.CheckSupport(geometry, domain, order, fixed_points);
   if (!report.Succeeded()) { return report; }
   report = backend.Generate(geometry, domain, order, out, fixed_points);
   if (!report.Succeeded()) { out.Clear(); return report; }
   auto fail = [&](CutQuadratureStatus status, const char *message)
   { out.Clear(); return Failure(status, message); };
   if ((out.region != CutQuadrature::Region::Full &&
        out.region != CutQuadrature::Region::Empty &&
        out.region != CutQuadrature::Region::Cut) ||
       (out.region == CutQuadrature::Region::Full && domain.Codimension() > 0))
   { return fail(CutQuadratureStatus::NumericalFailure, "Invalid backend region."); }
   if (fixed_points && out.GetNPoints() != fixed_points->GetNPoints())
   { return fail(CutQuadratureStatus::NumericalFailure, "Backend changed prescribed nodes."); }
   out.geometry = geometry;
   out.codimension = domain.Codimension();
   out.rule.SetOrder(order);
   const int dimension = Geometry::Dimension[geometry];
   const int codimension = domain.Codimension();
   const int count = out.GetNPoints();
   out.conormals.SetSize(0, 0, 0);
   if (codimension) { out.conormals.SetSize(dimension, codimension, count); out.conormals = 0.; }
   for (int index = 0; index < count; ++index)
   {
      auto &point = out.rule.IntPoint(index);
      point.index = index;
      if (!std::isfinite(point.weight) || !ValidPoint(geometry, point) ||
          (out.region == CutQuadrature::Region::Empty && point.weight != 0))
      { return fail(CutQuadratureStatus::NumericalFailure, "Invalid quadrature node."); }
      if (fixed_points)
      {
         const auto &prescribed = fixed_points->IntPoint(index);
         if (point.x != prescribed.x ||
             (dimension > 1 && point.y != prescribed.y) ||
             (dimension > 2 && point.z != prescribed.z))
         { return fail(CutQuadratureStatus::NumericalFailure, "Backend moved a prescribed node."); }
      }
      if (point.weight == 0 || !codimension) { continue; }
      DenseMatrix normals(dimension, codimension);
      Vector normal(dimension);
      int column = 0;
      for (int constraint = 0; constraint < domain.NumConstraints(); ++constraint)
      {
         const auto &condition = domain.GetConstraint(constraint);
         if (condition.sign != CutDomain::LevelSetSign::Zero) { continue; }
         condition.level_set.get().EvalGradient(point, normal);
         if (normal.Size() != dimension || !FiniteVector(normal) || normal.Norml2() == 0)
         { return fail(CutQuadratureStatus::DegenerateGeometry, "Singular level-set gradient."); }
         normal /= normal.Norml2(); normals.SetCol(column++, normal);
      }
      if (!FullRank(normals))
      { return fail(CutQuadratureStatus::DegenerateGeometry, "Dependent conormals."); }
      out.conormals(index) = normals;
   }
   return report;
}
catch (const std::exception &error)
{
   out.Clear();
   return Failure(CutQuadratureStatus::NumericalFailure, error.what());
}

CutQuadratureReport MapCutQuadrature(ElementTransformation &transformation,
   const CutQuadrature &reference, int requested_data,
   CutQuadratureMapping &out) try
{
   out.Clear();
   const int valid_data = CutQuadratureMapping::PHYSICAL_POINTS |
                          CutQuadratureMapping::PHYSICAL_CONORMALS |
                          CutQuadratureMapping::MEASURE_SCALING;
   if (!ValidGeometry(reference.geometry) || reference.geometry != transformation.GetGeometryType() ||
       reference.region == CutQuadrature::Region::Unknown ||
       (requested_data & ~valid_data))
   { return Failure(CutQuadratureStatus::InvalidInput, "Invalid mapping request."); }
   const int dimension = Geometry::Dimension[reference.geometry];
   const int space_dimension = transformation.GetSpaceDim();
   const int codimension = reference.codimension;
   const int count = reference.GetNPoints();
   if (dimension < 1 || space_dimension < dimension || codimension < 0 ||
       codimension > dimension ||
       (codimension && (reference.conormals.SizeI() != dimension ||
                       reference.conormals.SizeJ() != codimension || reference.conormals.SizeK() != count)))
   { return Failure(CutQuadratureStatus::InvalidInput, "Invalid reference conormal dimensions."); }
   PointGuard guard(transformation);
   if (requested_data & CutQuadratureMapping::PHYSICAL_POINTS)
   { out.physical_points.SetSize(space_dimension, count); }
   if ((requested_data & CutQuadratureMapping::PHYSICAL_CONORMALS) && codimension)
   { out.physical_conormals.SetSize(space_dimension, codimension, count); out.physical_conormals = 0.; }
   if (requested_data & CutQuadratureMapping::MEASURE_SCALING)
   { out.measure_scaling.SetSize(count); out.measure_scaling = 0.; }
   auto fail = [&](CutQuadratureStatus status, const char *message)
   { out.Clear(); return Failure(status, message); };
   for (int index = 0; index < count; ++index)
   {
      const auto &point = reference.rule.IntPoint(index);
      if (!std::isfinite(point.weight) || !ValidPoint(reference.geometry, point))
      { return fail(CutQuadratureStatus::InvalidInput, "Invalid reference node or weight."); }
      if (requested_data & CutQuadratureMapping::PHYSICAL_POINTS)
      {
         Vector physical(space_dimension);
         transformation.Transform(point, physical);
         if (!FiniteVector(physical))
         { return fail(CutQuadratureStatus::NumericalFailure, "Nonfinite physical point."); }
         out.physical_points.SetCol(index, physical);
      }
      if (point.weight == 0) { continue; }
      const bool map_conormals =
         codimension && (requested_data & CutQuadratureMapping::PHYSICAL_CONORMALS);
      if (!(requested_data & CutQuadratureMapping::MEASURE_SCALING) &&
          !map_conormals)
      { continue; }
      transformation.SetIntPoint(&point);
      const DenseMatrix &jacobian = transformation.Jacobian();
      DenseMatrix mapping_basis, mapping_upper;
      real_t mapping_measure;
      if (!FactorQR(jacobian, mapping_basis, mapping_upper, mapping_measure))
      { return fail(CutQuadratureStatus::DegenerateGeometry, "Singular element mapping."); }
      const real_t physical_volume = space_dimension == dimension ?
                                     std::abs(jacobian.Det()) : mapping_measure;
      if (!(physical_volume > 0) || !std::isfinite(physical_volume))
      { return fail(CutQuadratureStatus::DegenerateGeometry, "Invalid element measure."); }
      real_t scaling = physical_volume;
      if (codimension)
      {
         const DenseMatrix &normals = reference.conormals(index);
         DenseMatrix normal_basis, normal_upper;
         real_t normal_measure;
         if (!FactorQR(normals, normal_basis, normal_upper, normal_measure))
         { return fail(CutQuadratureStatus::DegenerateGeometry, "Dependent reference conormals."); }
         DenseMatrix dual;
         SolveUpperTranspose(mapping_upper, normal_basis, dual);
         DenseMatrix dual_basis, dual_upper;
         real_t dual_measure;
         if (!FactorQR(dual, dual_basis, dual_upper, dual_measure))
         { return fail(CutQuadratureStatus::DegenerateGeometry, "Dependent physical conormals."); }
         scaling = codimension == dimension ? 1 : physical_volume * dual_measure;
         if (map_conormals)
         {
            DenseMatrix original_dual;
            DenseMatrix physical(space_dimension, codimension);
            SolveUpperTranspose(mapping_upper, normals, original_dual);
            Mult(mapping_basis, original_dual, physical);
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
      if (requested_data & CutQuadratureMapping::MEASURE_SCALING)
      { out.measure_scaling(index) = scaling; }
   }
   return {};
}
catch (const std::exception &error)
{
   out.Clear();
   return Failure(CutQuadratureStatus::NumericalFailure, error.what());
}

#ifndef MFEM_USE_ALGOIM
std::unique_ptr<CutQuadratureBackend> CreateCutQuadratureBackend(const std::string &) { return nullptr; }
#endif
}
