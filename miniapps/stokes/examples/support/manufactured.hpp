#ifndef MFEM_STOKES_PACKAGE_EXAMPLES_SUPPORT_MANUFACTURED_HPP
#define MFEM_STOKES_PACKAGE_EXAMPLES_SUPPORT_MANUFACTURED_HPP

#include "stokes.hpp"
#include <array>
#include <string>

namespace mfem::flow
{

Mesh MakeStokesMesh(const std::string &element, int cells);

class ManufacturedSolution
{
   int dimension;
   real_t viscosity_scale, shift;
   bool variable_viscosity;
   DenseMatrix curl_weights;
   real_t Derivative(const Vector &point, const std::array<int, 3> &derivatives) const;
public:
   FunctionCoefficient viscosity;
   VectorFunctionCoefficient viscosity_gradient;
   VectorFunctionCoefficient velocity;
   FunctionCoefficient pressure;
   VectorFunctionCoefficient force;

   ManufacturedSolution(int dimension, real_t viscosity_scale = 1.0,
                        bool variable_viscosity = false, real_t shift = 0.0);
   void Velocity(const Vector &point, Vector &value) const;
   void Gradient(const Vector &point, DenseMatrix &value) const;
   void Force(const Vector &point, Vector &value) const;
   real_t Viscosity(const Vector &point) const;
   real_t Pressure(const Vector &point) const;
};

struct StokesErrors
{
   real_t velocity_l2, velocity_h1, pressure_l2, divergence_l2;
};

StokesErrors ComputeErrors(const ParGridFunction &velocity,
                          const ParGridFunction &pressure,
                          ManufacturedSolution &exact, int integration_order);

}
#endif
