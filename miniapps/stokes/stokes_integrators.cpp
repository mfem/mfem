#include "stokes_integrators.hpp"
#include <algorithm>
#include <cmath>

namespace mfem::stokes
{

int detail::StokesGeometryOrder(const Mesh &mesh)
{
   if (mesh.Dimension() != mesh.SpaceDimension() ||
       (mesh.Dimension() != 2 && mesh.Dimension() != 3)) { return -1; }
   for (int element = 0; element < mesh.GetNE(); ++element)
   {
      const auto geometry = mesh.GetElementGeometry(element);
      if (geometry != Geometry::TRIANGLE && geometry != Geometry::SQUARE &&
          geometry != Geometry::TETRAHEDRON && geometry != Geometry::CUBE) { return -1; }
   }
   if (!mesh.GetNodes()) { return 1; }
   const auto &space = *mesh.GetNodalFESpace();
   if (space.IsVariableOrder() || space.GetVDim() != mesh.SpaceDimension()) { return -1; }
   const auto *collection = space.FEColl();
   if (!dynamic_cast<const H1_FECollection *>(collection) &&
       !dynamic_cast<const L2_FECollection *>(collection) &&
       !dynamic_cast<const LinearFECollection *>(collection)) { return -1; }
   if (collection->GetOrder() < 1) { return -1; }
   for (int element = 0; element < mesh.GetNE(); ++element)
   {
      if (space.GetFE(element)->GetMapType() != FiniteElement::VALUE) { return -1; }
   }
   return collection->GetOrder();
}

int detail::StokesGeometryOrder(const ElementTransformation &transformation)
{
   const auto *isoparametric = dynamic_cast<const IsoparametricTransformation *>(&transformation);
   MFEM_VERIFY(isoparametric && isoparametric->GetFE(), "Stokes requires polynomial isoparametric geometry");
   const auto &element = *isoparametric->GetFE();
   const auto geometry = element.GetGeomType();
   MFEM_VERIFY((geometry == Geometry::TRIANGLE || geometry == Geometry::SQUARE ||
                geometry == Geometry::TETRAHEDRON || geometry == Geometry::CUBE) &&
               (element.Space() == FunctionSpace::Pk || element.Space() == FunctionSpace::Qk) &&
               element.GetMapType() == FiniteElement::VALUE && element.GetOrder() >= 1 &&
               !element.HasAnisotropicOrders(), "Unsupported Stokes polynomial coordinate element");
   return element.GetOrder();
}

int detail::StokesIntegrationOrder(int solution_order, int geometry_order,
                                   int dimension, int requested)
{
   MFEM_VERIFY(solution_order >= 1 && geometry_order >= 1 &&
               (requested == -1 || requested > 0), "Invalid Stokes integration order");
   return requested > 0 ? requested : 2 * solution_order + 2 + 2 * dimension * (geometry_order - 1);
}

void detail::StokesGeometryContext::Prepare(ElementTransformation &transformation)
{
   const int new_order = StokesGeometryOrder(transformation);
   const int new_dimension = transformation.GetDimension();
   if (!collection || order != new_order || dimension != new_dimension)
   {
      collection = std::make_unique<H1_FECollection>(new_order, new_dimension);
   }
   order = new_order;
   dimension = new_dimension;
   source = &transformation;
   geometry.SetFE(collection->FiniteElementForGeometry(transformation.GetGeometryType()));
   transformation.Transform(geometry.GetFE()->GetNodes(), geometry.GetPointMat());
   geometry.Reset();
}

void detail::StokesGeometryContext::SetIntPoint(const IntegrationPoint &point)
{
   MFEM_VERIFY(source, "Prepare the Stokes geometry workspace first");
   source->SetIntPoint(&point);
   geometry.SetIntPoint(&point);
}

void detail::StokesGeometryContext::CalcPhysHessian(const FiniteElement &element,
                                                   DenseMatrix &hessian)
{
   const int dofs = element.GetDof(), packed = dimension * (dimension + 1) / 2;
   DenseMatrix reference(dofs, packed), gradient(dofs, dimension);
   element.CalcHessian(geometry.GetIntPoint(), reference);
   element.CalcPhysDShape(geometry, gradient);
   const auto &mapping_hessian = geometry.Hessian();
   const auto &inverse = geometry.InverseJacobian();
   for (int node = 0; node < dofs; ++node)
   {
      for (int entry = 0; entry < packed; ++entry)
      {
         for (int axis = 0; axis < dimension; ++axis)
         {
            reference(node, entry) -= gradient(node, axis) * mapping_hessian(axis, entry);
         }
      }
   }
   hessian.SetSize(dofs, packed);
   hessian = 0.0;
   for (int first = 0; first < dimension; ++first)
   {
      for (int second = first; second < dimension; ++second)
      {
         const int entry = HessianIndex(dimension, first, second);
         for (int row = 0; row < dimension; ++row)
         {
            for (int column = 0; column < dimension; ++column)
            {
               const real_t weight = inverse(row, first) * inverse(column, second);
               for (int node = 0; node < dofs; ++node)
               {
                  hessian(node, entry) += weight * reference(node, HessianIndex(dimension, row, column));
               }
            }
         }
      }
   }
}

int HessianIndex(int dimension, int row, int column)
{
   MFEM_VERIFY((dimension == 2 || dimension == 3) && row >= 0 &&
               row < dimension && column >= 0 && column < dimension,
               "Invalid Hessian index");
   if (row > column) { std::swap(row, column); }
   return row * dimension - row * (row - 1) / 2 + column - row;
}

void EvaluateMomentumBasis(const FiniteElement &element,
                           ElementTransformation &transformation,
                           Coefficient &viscosity,
                           VectorCoefficient &viscosity_gradient,
                           DenseMatrix &residual)
{
   detail::StokesGeometryContext geometry;
   geometry.Prepare(transformation);
   EvaluateMomentumBasis(element, transformation, viscosity, viscosity_gradient, residual, geometry);
}

void EvaluateMomentumBasis(const FiniteElement &element,
                           ElementTransformation &transformation,
                           Coefficient &viscosity,
                           VectorCoefficient &viscosity_gradient,
                           DenseMatrix &residual,
                           detail::StokesGeometryContext &geometry)
{
   const int dimension = element.GetDim();
   const int dofs = element.GetDof();
   MFEM_VERIFY((dimension == 2 || dimension == 3) &&
               transformation.GetSpaceDim() == dimension &&
               viscosity_gradient.GetVDim() == dimension,
               "Momentum residual requires matching 2D or 3D volume elements");
   DenseMatrix hessian(dofs, dimension * (dimension + 1) / 2);
   DenseMatrix gradient(dofs, dimension);
   Vector mu_gradient(dimension);
   geometry.SetIntPoint(transformation.GetIntPoint());
   geometry.CalcPhysHessian(element, hessian);
   element.CalcPhysDShape(transformation, gradient);
   const IntegrationPoint &point = transformation.GetIntPoint();
   const real_t mu = viscosity.Eval(transformation, point);
   viscosity_gradient.Eval(mu_gradient, transformation, point);
   residual.SetSize(dimension, dimension * dofs);
   for (int node = 0; node < dofs; ++node)
   {
      real_t laplacian = 0.0, gradient_product = 0.0;
      for (int axis = 0; axis < dimension; ++axis)
      {
         laplacian += hessian(node, HessianIndex(dimension, axis, axis));
         gradient_product += gradient(node, axis) * mu_gradient(axis);
      }
      for (int component = 0; component < dimension; ++component)
      {
         for (int axis = 0; axis < dimension; ++axis)
         {
            real_t value = -mu * hessian(node, HessianIndex(dimension, axis, component)) -
                           gradient(node, axis) * mu_gradient(component);
            if (axis == component) { value -= mu * laplacian + gradient_product; }
            residual(axis, component * dofs + node) = value;
         }
      }
   }
}

PSPGSymGradIntegrator::PSPGSymGradIntegrator(
   Coefficient &viscosity_, VectorCoefficient &viscosity_gradient_,
   Coefficient &tau_, int integration_order_)
   : viscosity(viscosity_), viscosity_gradient(viscosity_gradient_),
     tau(tau_), integration_order(integration_order_) { }

void PSPGSymGradIntegrator::AssembleElementMatrix2(
   const FiniteElement &trial, const FiniteElement &test,
   ElementTransformation &transformation, DenseMatrix &matrix)
{
   const int dimension = trial.GetDim();
   MFEM_VERIFY((dimension == 2 || dimension == 3) &&
               transformation.GetSpaceDim() == dimension &&
               test.GetDim() == dimension &&
               viscosity_gradient.GetVDim() == dimension,
               "PSPG requires matching 2D or 3D volume elements");
   const int trial_dofs = trial.GetDof();
   const int test_dofs = test.GetDof();
   DenseMatrix test_gradient(test_dofs, dimension);
   DenseMatrix residual;
   detail::StokesGeometryContext geometry;
   geometry.Prepare(transformation);
   matrix.SetSize(test_dofs, dimension * trial_dofs);
   matrix = 0.0;
   const int rule_order = detail::StokesIntegrationOrder(std::max(trial.GetOrder(), test.GetOrder()),
                                                         geometry.GeometryOrder(), dimension, integration_order);
   const IntegrationRule &rule = IntRule ? *IntRule :
                                IntRules.Get(trial.GetGeomType(), rule_order);
   for (int point = 0; point < rule.GetNPoints(); ++point)
   {
      const IntegrationPoint &ip = rule.IntPoint(point);
      transformation.SetIntPoint(&ip);
      EvaluateMomentumBasis(trial, transformation, viscosity, viscosity_gradient, residual, geometry);
      test.CalcPhysDShape(transformation, test_gradient);
      const real_t weight = ip.weight * transformation.Weight() *
                            tau.Eval(transformation, ip);
      for (int trial_dof = 0; trial_dof < residual.Width(); ++trial_dof)
      {
         for (int test_node = 0; test_node < test_dofs; ++test_node)
         {
            real_t product = 0.0;
            for (int axis = 0; axis < dimension; ++axis)
            {
               product += test_gradient(test_node, axis) * residual(axis, trial_dof);
            }
            matrix(test_node, trial_dof) += weight * product;
         }
      }
   }
}

GLSVelocityIntegrator::GLSVelocityIntegrator(
   Coefficient &viscosity_, VectorCoefficient &viscosity_gradient_,
   Coefficient &tau_, int integration_order_)
   : viscosity(viscosity_), viscosity_gradient(viscosity_gradient_),
     tau(tau_), integration_order(integration_order_) { }

void GLSVelocityIntegrator::AssembleElementMatrix(
   const FiniteElement &element, ElementTransformation &transformation,
   DenseMatrix &matrix)
{
   matrix.SetSize(element.GetDim() * element.GetDof());
   matrix = 0.0;
   DenseMatrix residual;
   detail::StokesGeometryContext geometry;
   geometry.Prepare(transformation);
   const int order = detail::StokesIntegrationOrder(element.GetOrder(), geometry.GeometryOrder(),
                                                    element.GetDim(), integration_order);
   const IntegrationRule &rule = IntRule ? *IntRule : IntRules.Get(element.GetGeomType(), order);
   for (int point = 0; point < rule.GetNPoints(); ++point)
   {
      const IntegrationPoint &ip = rule.IntPoint(point);
      transformation.SetIntPoint(&ip);
      EvaluateMomentumBasis(element, transformation, viscosity, viscosity_gradient, residual, geometry);
      const real_t weight = ip.weight * transformation.Weight() * tau.Eval(transformation, ip);
      for (int row = 0; row < matrix.Height(); ++row)
      {
         for (int column = 0; column < matrix.Width(); ++column)
         {
            real_t product = 0.0;
            for (int axis = 0; axis < element.GetDim(); ++axis)
            {
               product += residual(axis, row) * residual(axis, column);
            }
            matrix(row, column) += weight * product;
         }
      }
   }
}

GLSForceIntegrator::GLSForceIntegrator(
   Coefficient &viscosity_, VectorCoefficient &viscosity_gradient_,
   Coefficient &tau_, VectorCoefficient &force_, int integration_order_)
   : viscosity(viscosity_), viscosity_gradient(viscosity_gradient_),
     tau(tau_), force(force_), integration_order(integration_order_) { }

void GLSForceIntegrator::AssembleRHSElementVect(
   const FiniteElement &element, ElementTransformation &transformation,
   Vector &vector)
{
   MFEM_VERIFY(force.GetVDim() == element.GetDim(), "GLS force dimension mismatch");
   vector.SetSize(element.GetDim() * element.GetDof());
   vector = 0.0;
   DenseMatrix residual;
   Vector force_value(element.GetDim());
   detail::StokesGeometryContext geometry;
   geometry.Prepare(transformation);
   const int order = detail::StokesIntegrationOrder(element.GetOrder(), geometry.GeometryOrder(),
                                                    element.GetDim(), integration_order);
   const IntegrationRule &rule = IntRule ? *IntRule : IntRules.Get(element.GetGeomType(), order);
   for (int point = 0; point < rule.GetNPoints(); ++point)
   {
      const IntegrationPoint &ip = rule.IntPoint(point);
      transformation.SetIntPoint(&ip);
      EvaluateMomentumBasis(element, transformation, viscosity, viscosity_gradient, residual, geometry);
      force.Eval(force_value, transformation, ip);
      const real_t weight = ip.weight * transformation.Weight() * tau.Eval(transformation, ip);
      for (int dof = 0; dof < vector.Size(); ++dof)
      {
         for (int axis = 0; axis < element.GetDim(); ++axis)
         {
            vector(dof) += weight * residual(axis, dof) * force_value(axis);
         }
      }
   }
}

PressureCIPIntegrator::PressureCIPIntegrator(
   Coefficient &element_size_, Coefficient &element_viscosity_,
   real_t gamma_, int integration_order_)
   : element_size(element_size_), element_viscosity(element_viscosity_),
     gamma(gamma_), integration_order(integration_order_)
{
   MFEM_VERIFY(std::isfinite(gamma) && gamma > 0.0, "CIP gamma must be positive and finite");
}

void PressureCIPIntegrator::AssembleFaceMatrix(
   const FiniteElement &first, const FiniteElement &second,
   FaceElementTransformations &transformation, DenseMatrix &matrix)
{
   const int dimension = first.GetDim();
   MFEM_VERIFY((dimension == 2 || dimension == 3) && second.GetDim() == dimension &&
               transformation.Elem1 && transformation.Elem2,
               "CIP requires an interior face of 2D or 3D volume elements");
   const int first_dofs = first.GetDof(), second_dofs = second.GetDof();
   matrix.SetSize(first_dofs + second_dofs);
   matrix = 0.0;
   DenseMatrix first_gradient(first_dofs, dimension), second_gradient(second_dofs, dimension);
   Vector normal(dimension), jump(first_dofs + second_dofs);
   const int geometry_order = std::max(detail::StokesGeometryOrder(*transformation.Elem1),
                                       detail::StokesGeometryOrder(*transformation.Elem2));
   const int order = detail::StokesIntegrationOrder(std::max(first.GetOrder(), second.GetOrder()),
                                                    geometry_order, dimension, integration_order);
   const IntegrationRule &rule = IntRule ? *IntRule : IntRules.Get(transformation.GetGeometryType(), order);
   for (int point = 0; point < rule.GetNPoints(); ++point)
   {
      const IntegrationPoint &ip = rule.IntPoint(point);
      transformation.SetAllIntPoints(&ip);
      auto &first_transformation = *transformation.Elem1;
      auto &second_transformation = *transformation.Elem2;
      first.CalcPhysDShape(first_transformation, first_gradient);
      second.CalcPhysDShape(second_transformation, second_gradient);
      CalcOrtho(transformation.Jacobian(), normal);
      const real_t measure = normal.Norml2();
      MFEM_VERIFY(std::isfinite(measure) && measure > 0.0, "Invalid CIP face geometry");
      normal /= measure;
      const real_t first_size = element_size.Eval(first_transformation, first_transformation.GetIntPoint());
      const real_t second_size = element_size.Eval(second_transformation, second_transformation.GetIntPoint());
      const real_t first_mu = element_viscosity.Eval(first_transformation, first_transformation.GetIntPoint());
      const real_t second_mu = element_viscosity.Eval(second_transformation, second_transformation.GetIntPoint());
      MFEM_VERIFY(std::isfinite(first_size) && first_size > 0.0 &&
                  std::isfinite(second_size) && second_size > 0.0 &&
                  std::isfinite(first_mu) && first_mu > 0.0 &&
                  std::isfinite(second_mu) && second_mu > 0.0,
                  "Invalid CIP element size or viscosity");
      const real_t size = std::min(first_size, second_size);
      const real_t weight = ip.weight * measure * gamma * size * size * size *
                            (0.5 / first_mu + 0.5 / second_mu);
      for (int dof = 0; dof < jump.Size(); ++dof)
      {
         jump(dof) = 0.0;
         for (int axis = 0; axis < dimension; ++axis)
         {
            jump(dof) += normal(axis) * (dof < first_dofs ?
                         first_gradient(dof, axis) : -second_gradient(dof - first_dofs, axis));
         }
      }
      for (int row = 0; row < jump.Size(); ++row)
      {
         for (int column = 0; column < jump.Size(); ++column)
         {
            matrix(row, column) += weight * jump(row) * jump(column);
         }
      }
   }
}

}
