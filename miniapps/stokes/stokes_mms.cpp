#include "stokes_mms.hpp"
#include <cmath>

namespace mfem::stokes
{
namespace
{
const real_t pi = std::acos(real_t(-1.0));

real_t SineSquaredDerivative(real_t coordinate, int derivative)
{
   if (derivative == 0) { return std::pow(std::sin(pi * coordinate), 2); }
   if (derivative == 1) { return pi * std::sin(2 * pi * coordinate); }
   if (derivative == 2) { return 2 * pi * pi * std::cos(2 * pi * coordinate); }
   return -4 * pi * pi * pi * std::sin(2 * pi * coordinate);
}
}

Mesh MakeStokesMesh(const std::string &element, int cells)
{
   MFEM_VERIFY(cells > 0, "Mesh cell count must be positive");
   if (element == "tri" || element == "quad")
   {
      return Mesh::MakeCartesian2D(cells, cells,
                                  element == "tri" ? Element::TRIANGLE : Element::QUADRILATERAL,
                                  true);
   }
   MFEM_VERIFY(element == "tet" || element == "hex", "Unknown element type");
   return Mesh::MakeCartesian3D(cells, cells, cells,
                               element == "tet" ? Element::TETRAHEDRON : Element::HEXAHEDRON);
}

ManufacturedSolution::ManufacturedSolution(int dimension_, real_t viscosity_scale_,
                                          bool variable_viscosity_, real_t shift_)
   : dimension(dimension_), viscosity_scale(viscosity_scale_), shift(shift_),
     variable_viscosity(variable_viscosity_), curl_weights(dimension),
     viscosity([this](const Vector &point) { return Viscosity(point); }),
     viscosity_gradient(dimension, [this](const Vector &, Vector &value)
     {
        value.SetSize(dimension);
        value = variable_viscosity ? viscosity_scale : 0.0;
     }),
     velocity(dimension, [this](const Vector &point, Vector &value) { Velocity(point, value); }),
     pressure([this](const Vector &point) { return Pressure(point); }),
     force(dimension, [this](const Vector &point, Vector &value) { Force(point, value); })
{
   MFEM_VERIFY(dimension == 2 || dimension == 3, "Invalid MMS dimension");
   curl_weights = 0.0;
   if (dimension == 2)
   {
      curl_weights(0, 1) = 1.0;
      curl_weights(1, 0) = -1.0;
   }
   else
   {
      curl_weights(0, 1) = 3.0;
      curl_weights(0, 2) = -2.0;
      curl_weights(1, 2) = 1.0;
      curl_weights(1, 0) = -3.0;
      curl_weights(2, 0) = 2.0;
      curl_weights(2, 1) = -1.0;
   }
}

real_t ManufacturedSolution::Derivative(const Vector &point,
                                        const std::array<int, 3> &derivatives) const
{
   real_t value = 1.0;
   for (int axis = 0; axis < dimension; ++axis)
   {
      value *= SineSquaredDerivative(point(axis), derivatives[axis]);
   }
   return value;
}

real_t ManufacturedSolution::Viscosity(const Vector &point) const
{
   return viscosity_scale * (variable_viscosity ? 1.0 + point.Sum() : 1.0);
}

real_t ManufacturedSolution::Pressure(const Vector &point) const
{
   return std::sin(pi * point(0)) * std::cos(pi * point(1)) *
          (dimension == 3 ? std::sin(pi * point(2)) : 1.0);
}

void ManufacturedSolution::Velocity(const Vector &point, Vector &value) const
{
   value.SetSize(dimension);
   for (int component = 0; component < dimension; ++component)
   {
      value(component) = shift * (component + 1);
      for (int axis = 0; axis < dimension; ++axis)
      {
         std::array<int, 3> derivatives = {};
         derivatives[axis] = 1;
         value(component) += curl_weights(component, axis) * Derivative(point, derivatives);
      }
   }
}

void ManufacturedSolution::Gradient(const Vector &point, DenseMatrix &value) const
{
   value.SetSize(dimension);
   value = 0.0;
   for (int component = 0; component < dimension; ++component)
   {
      for (int direction = 0; direction < dimension; ++direction)
      {
         for (int axis = 0; axis < dimension; ++axis)
         {
            std::array<int, 3> derivatives = {};
            ++derivatives[axis];
            ++derivatives[direction];
            value(component, direction) += curl_weights(component, axis) *
                                           Derivative(point, derivatives);
         }
      }
   }
}

void ManufacturedSolution::Force(const Vector &point, Vector &value) const
{
   value.SetSize(dimension);
   value(0) = pi * std::cos(pi * point(0)) * std::cos(pi * point(1));
   value(1) = -pi * std::sin(pi * point(0)) * std::sin(pi * point(1));
   if (dimension == 3)
   {
      value(0) *= std::sin(pi * point(2));
      value(1) *= std::sin(pi * point(2));
      value(2) = pi * std::sin(pi * point(0)) * std::cos(pi * point(1)) *
                 std::cos(pi * point(2));
   }
   DenseMatrix gradient;
   Gradient(point, gradient);
   for (int component = 0; component < dimension; ++component)
   {
      real_t laplacian = 0.0;
      for (int direction = 0; direction < dimension; ++direction)
      {
         for (int axis = 0; axis < dimension; ++axis)
         {
            std::array<int, 3> derivatives = {};
            ++derivatives[direction];
            derivatives[axis] += 2;
            laplacian += curl_weights(component, direction) * Derivative(point, derivatives);
         }
         if (variable_viscosity)
         {
            value(component) -= viscosity_scale *
                                (gradient(component, direction) + gradient(direction, component));
         }
      }
      value(component) -= Viscosity(point) * laplacian;
   }
}

StokesErrors ComputeErrors(const ParGridFunction &velocity,
                          const ParGridFunction &pressure,
                          ManufacturedSolution &exact, int integration_order)
{
   ParMesh &mesh = *velocity.ParFESpace()->GetParMesh();
   const int dimension = mesh.Dimension();
   real_t local[4] = {}, global[4] = {};
   Vector point(dimension), computed(dimension), reference(dimension);
   DenseMatrix computed_gradient(dimension), reference_gradient(dimension);
   for (int element = 0; element < mesh.GetNE(); ++element)
   {
      ElementTransformation &transformation = *mesh.GetElementTransformation(element);
      const IntegrationRule &rule = IntRules.Get(mesh.GetElementGeometry(element), integration_order);
      for (int index = 0; index < rule.GetNPoints(); ++index)
      {
         const IntegrationPoint &ip = rule.IntPoint(index);
         transformation.SetIntPoint(&ip);
         transformation.Transform(ip, point);
         const real_t weight = ip.weight * transformation.Weight();
         velocity.GetVectorValue(element, ip, computed);
         exact.Velocity(point, reference);
         computed -= reference;
         local[0] += weight * (computed * computed);
         transformation.SetIntPoint(&ip);
         velocity.GetVectorGradient(transformation, computed_gradient);
         real_t divergence = 0.0;
         for (int axis = 0; axis < dimension; ++axis) { divergence += computed_gradient(axis, axis); }
         local[3] += weight * divergence * divergence;
         exact.Gradient(point, reference_gradient);
         computed_gradient -= reference_gradient;
         local[1] += weight * computed_gradient.FNorm2();
         const real_t pressure_error = pressure.GetValue(element, ip) - exact.Pressure(point);
         local[2] += weight * pressure_error * pressure_error;
      }
   }
   MPI_Allreduce(local, global, 4, MPITypeMap<real_t>::mpi_type, MPI_SUM, mesh.GetComm());
   return {std::sqrt(global[0]), std::sqrt(global[1]),
           std::sqrt(global[2]), std::sqrt(global[3])};
}

}
