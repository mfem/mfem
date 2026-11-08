#include "stokes_mms.hpp"
#include "stokes_integrators.hpp"
#include "stokes_preconditioners.hpp"
#include "stokes_mesh_mms.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

using namespace mfem;
using namespace mfem::stokes;

namespace
{

class Tests
{
   int failures = 0, checks = 0;
public:
   std::string context;
   void Require(bool condition, const std::string &name)
   {
      ++checks;
      if (!condition)
      {
         ++failures;
         err << "FAIL rank=" << Mpi::WorldRank() << " " << context << " " << name << '\n';
      }
   }

   void Near(real_t actual, real_t expected, real_t tolerance, const std::string &name)
   {
      std::ostringstream diagnostic;
      diagnostic << std::scientific << " actual=" << actual << " expected=" << expected
                 << " tolerance=" << tolerance;
      Require(std::isfinite(actual) && std::abs(actual - expected) <=
              tolerance * (1.0 + std::abs(expected)), name + diagnostic.str());
   }

   int Finish()
   {
      int total_failures = 0, total_checks = 0;
      MPI_Allreduce(&failures, &total_failures, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(&checks, &total_checks, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
      if (Mpi::Root())
      {
         out << "Stokes tests: " << total_checks << " checks, " << total_failures << " failures\n";
      }
      return total_failures ? 1 : 0;
   }
};

real_t Norm(const Vector &vector, MPI_Comm comm = MPI_COMM_WORLD)
{
   return std::sqrt(InnerProduct(comm, vector, vector));
}

void CheckAssemblyTime(Tests &tests, const StokesSolveInfo &info)
{
   tests.Require(std::isfinite(info.assembly_seconds) && info.assembly_seconds > 0.0,
                 "Assembly time is finite and positive");
   double minimum = 0.0, maximum = 0.0;
   MPI_Allreduce(&info.assembly_seconds, &minimum, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
   MPI_Allreduce(&info.assembly_seconds, &maximum, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
   tests.Require(minimum == maximum, "Assembly time is identical on all MPI ranks");
   for (double seconds : {info.preconditioner_setup_seconds, info.direct_setup_seconds,
                          info.linear_solve_seconds, info.finalization_seconds})
   {
      tests.Require(std::isfinite(seconds) && seconds >= 0.0, "Phase time is finite and nonnegative");
      MPI_Allreduce(&seconds, &minimum, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
      MPI_Allreduce(&seconds, &maximum, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      tests.Require(minimum == maximum, "Phase time is identical on all MPI ranks");
   }
}

real_t Monomial(const Vector &point, const std::array<int, 3> &powers,
                const std::array<int, 3> &derivatives = {})
{
   real_t result = 1.0;
   for (int axis = 0; axis < point.Size(); ++axis)
   {
      if (derivatives[axis] > powers[axis]) { return 0.0; }
      for (int derivative = 0; derivative < derivatives[axis]; ++derivative)
      {
         result *= powers[axis] - derivative;
      }
      result *= std::pow(point(axis), powers[axis] - derivatives[axis]);
   }
   return result;
}

void CurveTestMesh(Mesh &mesh, int geometry_order)
{
   if (geometry_order <= 1) { return; }
   mesh.SetCurvature(geometry_order, geometry_order == 3);
   mesh.Transform([](const Vector &point, Vector &value)
   {
      const real_t pi = std::acos(real_t(-1.0));
      real_t bubble = 1.0;
      for (int axis = 0; axis < point.Size(); ++axis) { bubble *= std::sin(pi * point(axis)); }
      value = point;
      for (int axis = 0; axis < point.Size(); ++axis)
      {
         value(axis) += 0.02 * (axis + 1) / point.Size() * bubble * std::sin(2 * pi * point(axis));
      }
   });
}

void GeometryTests(Tests &tests)
{
   for (const auto shape : {Geometry::TRIANGLE, Geometry::SQUARE, Geometry::TETRAHEDRON, Geometry::CUBE})
   {
      const int dimension = Geometry::Dimension[shape];
      const bool tensor = shape == Geometry::SQUARE || shape == Geometry::CUBE;
      for (int degree : {1, 2, 3})
      {
         for (bool discontinuous : {false, true})
         {
            for (real_t scale : {real_t(1e-6), real_t(1.0), real_t(1e6)})
            {
               tests.context = "geometry shape=" + std::to_string(shape) + " degree=" +
                               std::to_string(degree) + " dg=" + std::to_string(discontinuous) +
                               " scale=" + std::to_string(scale);
               H1_FECollection continuous(degree, dimension), solution(2 * degree, dimension);
               L2_FECollection broken(degree, dimension);
               const auto &coordinate = *(discontinuous ? broken.FiniteElementForGeometry(shape) :
                                          continuous.FiniteElementForGeometry(shape));
               const real_t mixed = degree > 1 || tensor ? 0.12 : 0.0;
               const real_t quadratic = degree > 1 ? 0.08 : 0.0;
               const real_t cubic = degree > 2 ? 0.03 : 0.0;
               auto map = [&](const IntegrationPoint &point, Vector &value)
               {
                  value.SetSize(dimension);
                  value(0) = scale * (point.x + mixed * point.x * point.y + cubic * std::pow(point.x, 3));
                  value(1) = scale * (point.y + quadratic * point.x * point.x);
                  if (dimension == 3) { value(2) = scale * (point.z + mixed * point.y * point.z); }
               };
               IsoparametricTransformation source;
               source.SetFE(&coordinate);
               source.GetPointMat().SetSize(dimension, coordinate.GetDof());
               Vector value;
               for (int node = 0; node < coordinate.GetDof(); ++node)
               {
                  map(coordinate.GetNodes().IntPoint(node), value);
                  source.GetPointMat().SetCol(node, value);
               }
               detail::StokesGeometryContext prepared;
               prepared.Prepare(source);
               const auto &basis = *solution.FiniteElementForGeometry(shape);
               Vector polynomial(basis.GetDof());
               for (int node = 0; node < basis.GetDof(); ++node)
               {
                  map(basis.GetNodes().IntPoint(node), value);
                  value /= scale;
                  polynomial(node) = value(0) * value(0) + value(1) * value(dimension == 2 ? 0 : 2);
               }
               const auto &rule = IntRules.Get(shape, 2);
               for (int index = 0; index < rule.GetNPoints(); ++index)
               {
                  const auto &point = rule.IntPoint(index);
                  prepared.SetIntPoint(point);
                  auto &replacement = prepared.HessianTransformation();
                  Vector expected, actual;
                  map(point, expected);
                  replacement.Transform(point, actual);
                  actual -= expected;
                  tests.Near(actual.Normlinf() / scale, 0.0, 2e-13, "Original polynomial map preserved");
                  DenseMatrix jacobian(dimension);
                  jacobian = 0.0;
                  jacobian(0, 0) = scale * (1 + mixed * point.y + 3 * cubic * point.x * point.x);
                  jacobian(0, 1) = scale * mixed * point.x;
                  jacobian(1, 0) = scale * 2 * quadratic * point.x;
                  jacobian(1, 1) = scale;
                  if (dimension == 3)
                  {
                     jacobian(2, 1) = scale * mixed * point.z;
                     jacobian(2, 2) = scale * (1 + mixed * point.y);
                  }
                  jacobian -= replacement.Jacobian();
                  tests.Near(jacobian.FNorm() / scale, 0.0, 1e-12, "Analytic mapping Jacobian");
                  DenseMatrix mapping_hessian(dimension, dimension * (dimension + 1) / 2);
                  mapping_hessian = 0.0;
                  mapping_hessian(0, 0) = scale * 6 * cubic * point.x;
                  mapping_hessian(0, 1) = scale * mixed;
                  mapping_hessian(1, 0) = scale * 2 * quadratic;
                  if (dimension == 3) { mapping_hessian(2, 4) = scale * mixed; }
                  mapping_hessian -= replacement.Hessian();
                  tests.Near(mapping_hessian.FNorm() / scale, 0.0, 2e-11, "Analytic mapping Hessian");
                  DenseMatrix hessian;
                  prepared.CalcPhysHessian(basis, hessian);
                  Vector physical_hessian(hessian.Width());
                  hessian.MultTranspose(polynomial, physical_hessian);
                  physical_hessian *= scale * scale;
                  for (int entry = 0; entry < physical_hessian.Size(); ++entry)
                  {
                     const real_t exact = entry == 0 ? 2.0 : entry == (dimension == 2 ? 1 : 4) ? 1.0 : 0.0;
                     tests.Near(physical_hessian(entry), exact, 2e-9, "Independent physical quadratic Hessian");
                  }
               }
               tests.Require(detail::StokesIntegrationOrder(2, degree, dimension) == 6 + 2 * dimension * (degree - 1),
                             "Geometry-aware quadrature default");
               tests.Require(detail::StokesIntegrationOrder(2, degree, dimension, 19) == 19, "Explicit quadrature override");
            }
         }
      }
   }
   for (const std::string family : {"tri", "quad", "tet", "hex"})
   {
      Mesh serial = MakeStokesMesh(family, 2);
      CurveTestMesh(serial, 3);
      ParMesh mesh(MPI_COMM_WORLD, serial);
      H1_FECollection collection(3, mesh.Dimension());
      ParFiniteElementSpace space(&mesh, &collection);
      ConstantCoefficient one(1.0);
      ParBilinearForm form(&space);
      form.AddInteriorFaceIntegrator(new PressureCIPIntegrator(one, one, 0.01));
      form.Assemble();
      form.Finalize();
      std::unique_ptr<HypreParMatrix> matrix(form.ParallelAssemble());
      FunctionCoefficient linear([](const Vector &point) { return 1 + point.Sum(); });
      ParGridFunction field(&space);
      field.ProjectCoefficient(linear);
      Vector values, residual(space.GetTrueVSize());
      field.GetTrueDofs(values);
      matrix->Mult(values, residual);
      tests.context = "curved CIP " + family;
      tests.Near(Norm(residual), 0.0, 1e-10, "Representable physical linear pressure has no face jump");
      Vector first(values.Size()), second(values.Size()), first_action(values.Size()), second_action(values.Size());
      for (int entry = 0; entry < first.Size(); ++entry)
      {
         first(entry) = std::sin(real_t(entry + 1));
         second(entry) = std::cos(real_t(entry + 2));
      }
      matrix->Mult(first, first_action);
      matrix->Mult(second, second_action);
      tests.Near(InnerProduct(MPI_COMM_WORLD, first, second_action),
                 InnerProduct(MPI_COMM_WORLD, second, first_action), 1e-11, "Curved CIP symmetry");
      tests.Require(InnerProduct(MPI_COMM_WORLD, first, first_action) >= 0.0, "Curved CIP nonnegative energy");
   }
}

void OperatorTests(Tests &tests)
{
   if (!Mpi::Root()) { return; }
   for (const std::string element_type : {"tri", "quad", "tet", "hex"})
   {
      Mesh mesh = MakeStokesMesh(element_type, 1);
      const int dimension = mesh.Dimension();
      for (int vertex = 0; vertex < mesh.GetNV(); ++vertex)
      {
         real_t *point = mesh.GetVertex(vertex);
         const real_t original_x = point[0], original_y = point[1];
         point[0] += 0.17 * original_x * original_y + 0.11 * original_y;
         point[1] += 0.07 * original_x * original_y;
         if (dimension == 3) { point[2] += 0.13 * original_x * original_y * point[2]; }
      }
      for (int order : {2, 3})
      {
         H1_FECollection collection(order, dimension);
         for (bool variable_mu : {false, true})
         {
            FunctionCoefficient viscosity([variable_mu](const Vector &point)
            {
               return variable_mu ? 1.0 + point.Sum() : 1.0;
            });
            Vector gradient_value(dimension);
            gradient_value = variable_mu ? 1.0 : 0.0;
            VectorConstantCoefficient viscosity_gradient(gradient_value);
            ConstantCoefficient tau(0.37);
            PSPGSymGradIntegrator pspg(viscosity, viscosity_gradient, tau, 2 * order + 8);
            ElasticityIntegrator elasticity(viscosity, 0.0, 1.0);
            VectorDivergenceIntegrator divergence;
            DiffusionIntegrator stabilization(tau);
            for (int element = 0; element < mesh.GetNE(); ++element)
            {
               const FiniteElement &finite_element = *collection.FiniteElementForGeometry(
                                                        mesh.GetElementGeometry(element));
               ElementTransformation &transformation = *mesh.GetElementTransformation(element);
               const int dofs = finite_element.GetDof();
               const IntegrationRule &rule = IntRules.Get(finite_element.GetGeomType(), 2 * order + 8);
               elasticity.SetIntRule(&rule);
               divergence.SetIntRule(&rule);
               stabilization.SetIntRule(&rule);
               DenseMatrix coupling, stiffness, divergence_matrix, pressure_matrix;
               pspg.AssembleElementMatrix2(finite_element, finite_element, transformation, coupling);
               elasticity.AssembleElementMatrix(finite_element, transformation, stiffness);
               divergence.AssembleElementMatrix2(finite_element, finite_element, transformation,
                                                  divergence_matrix);
               stabilization.AssembleElementMatrix(finite_element, transformation, pressure_matrix);
               tests.Require(coupling.Height() == dofs && coupling.Width() == dimension * dofs,
                             "C block dimensions");
               Vector constant(dofs), action(dofs);
               constant = 1.0;
               pressure_matrix.Mult(constant, action);
               tests.Require(action.Norml2() < 1e-10, "S annihilates constants");
               DenseMatrix asymmetry(stiffness);
               for (int row = 0; row < stiffness.Height(); ++row)
               {
                  for (int column = 0; column < stiffness.Width(); ++column)
                  {
                     asymmetry(row, column) -= stiffness(column, row);
                  }
               }
               tests.Require(asymmetry.FNorm() < 1e-10 * (1.0 + stiffness.FNorm()), "A symmetric");
               Vector point(dimension), input(dimension * dofs), pressure_test(dofs);
               Vector stiffness_action(input.Size()), div_action(dofs);
               input = 0.0;
               for (int node = 0; node < dofs; ++node)
               {
                  transformation.Transform(finite_element.GetNodes().IntPoint(node), point);
                  input(node) = -point(1);
                  input(dofs + node) = point(0);
               }
               stiffness.Mult(input, stiffness_action);
               tests.Require(stiffness_action.Norml2() < 1e-10 * (1.0 + stiffness.FNorm()),
                             "Symmetric gradient annihilates rigid rotation, unlike vector Laplacian");

               for (int first = 0; first < dimension; ++first)
               {
                  for (int second = first; second < dimension; ++second)
                  {
                     std::array<int, 3> powers = {};
                     powers[first] += order - 1;
                     ++powers[second];
                     for (int component = 0; component < dimension; ++component)
                     {
                        input = 0.0;
                        for (int node = 0; node < dofs; ++node)
                        {
                           transformation.Transform(finite_element.GetNodes().IntPoint(node), point);
                           input(component * dofs + node) = Monomial(point, powers);
                        }
                        coupling.Mult(input, action);
                        divergence_matrix.Mult(input, div_action);
                        stiffness.Mult(input, stiffness_action);
                        real_t expected_energy = 0.0;
                        Vector residual_integral(dimension), divergence_integral(dimension);
                        residual_integral = 0.0;
                        divergence_integral = 0.0;
                        for (int index = 0; index < rule.GetNPoints(); ++index)
                        {
                           const IntegrationPoint &ip = rule.IntPoint(index);
                           transformation.SetIntPoint(&ip);
                           transformation.Transform(ip, point);
                           const real_t weight = ip.weight * transformation.Weight();
                           const real_t mu = variable_mu ? 1.0 + point.Sum() : 1.0;
                           Vector gradient(dimension);
                           real_t laplacian = 0.0, gradient_product = 0.0;
                           for (int axis = 0; axis < dimension; ++axis)
                           {
                              std::array<int, 3> derivatives = {};
                              derivatives[axis] = 1;
                              gradient(axis) = Monomial(point, powers, derivatives);
                              derivatives[axis] = 2;
                              laplacian += Monomial(point, powers, derivatives);
                              gradient_product += gradient(axis) * gradient_value(axis);
                           }
                           expected_energy += weight * mu * ((gradient * gradient) +
                                                              gradient(component) * gradient(component));
                           for (int axis = 0; axis < dimension; ++axis)
                           {
                              std::array<int, 3> derivatives = {};
                              ++derivatives[axis];
                              ++derivatives[component];
                              real_t residual = -mu * Monomial(point, powers, derivatives) -
                                                 gradient(axis) * gradient_value(component);
                              if (axis == component) { residual -= mu * laplacian + gradient_product; }
                              residual_integral(axis) += weight * 0.37 * residual;
                              divergence_integral(axis) += weight * point(axis) * gradient(component);
                           }
                        }
                        tests.Near(input * stiffness_action, expected_energy, 1e-10,
                                   "A analytic symmetric-gradient energy");
                        for (int axis = 0; axis < dimension; ++axis)
                        {
                           for (int node = 0; node < dofs; ++node)
                           {
                              transformation.Transform(finite_element.GetNodes().IntPoint(node), point);
                              pressure_test(node) = point(axis);
                           }
                           tests.Near(pressure_test * action, residual_integral(axis), 1e-10,
                                      "C analytic Laplacian, grad-div and variable-mu action");
                           tests.Near(pressure_test * div_action, divergence_integral(axis), 1e-10,
                                      "B analytic sign and action");
                        }
                     }
                  }
               }
            }
         }
      }
   }
}

void ExtensionOperatorTests(Tests &tests)
{
   for (const std::string element : {"tri", "quad", "tet", "hex"})
   {
      Mesh serial = MakeStokesMesh(element, 2);
      const int dimension = serial.Dimension();
      for (int order : {2, 3})
      {
         tests.context = "operators " + element + " k=" + std::to_string(order);
         H1_FECollection collection(order, dimension);
         FiniteElementSpace space(&serial, &collection);
         ConstantCoefficient one(1.0), tau(0.37);
         PressureCIPIntegrator cip(one, one, 0.01, 2 * order + 4);
         for (int face = 0; face < serial.GetNumFaces(); ++face)
         {
            auto *transformation = serial.GetInteriorFaceTransformations(face);
            if (!transformation) { continue; }
            const auto &first = *space.GetFE(transformation->Elem1No);
            const auto &second = *space.GetFE(transformation->Elem2No);
            DenseMatrix forward, reverse;
            cip.AssembleFaceMatrix(first, second, *transformation, forward);
            std::swap(transformation->Elem1, transformation->Elem2);
            std::swap(transformation->Elem1No, transformation->Elem2No);
            std::swap(transformation->Loc1, transformation->Loc2);
            cip.AssembleFaceMatrix(second, first, *transformation, reverse);
            std::swap(transformation->Elem1, transformation->Elem2);
            std::swap(transformation->Elem1No, transformation->Elem2No);
            std::swap(transformation->Loc1, transformation->Loc2);
            real_t symmetry = 0.0, orientation = 0.0;
            for (int row = 0; row < forward.Height(); ++row)
            {
               const int swapped_row = row < first.GetDof() ? second.GetDof() + row : row - first.GetDof();
               for (int column = 0; column < forward.Width(); ++column)
               {
                  const int swapped_column = column < first.GetDof() ? second.GetDof() + column : column - first.GetDof();
                  symmetry = std::max(symmetry, std::abs(forward(row, column) - forward(column, row)));
                  orientation = std::max(orientation, std::abs(forward(row, column) - reverse(swapped_row, swapped_column)));
               }
            }
            tests.Near(symmetry, 0.0, 1e-12 * (1.0 + forward.FNorm()), "CIP face symmetry");
            tests.Near(orientation, 0.0, 1e-12 * (1.0 + forward.FNorm()), "CIP side reversal with reordered DOFs");
         }
         FunctionCoefficient linear([](const Vector &point) { return 0.5 + point.Sum(); });
         FunctionCoefficient kink([](const Vector &point) { return std::abs(point(0) - 0.5); });
         BilinearForm serial_penalty(&space);
         serial_penalty.AddInteriorFaceIntegrator(new PressureCIPIntegrator(one, one, 0.01));
         serial_penalty.Assemble();
         serial_penalty.Finalize();
         GridFunction serial_field(&space);
         serial_field.ProjectCoefficient(kink);
         Vector serial_action(space.GetVSize());
         serial_penalty.Mult(serial_field, serial_action);
         const real_t serial_energy = serial_field * serial_action;
         tests.Require(serial_energy > 1e-6, "CIP detects continuous pressure gradient jump");
         ParMesh mesh(MPI_COMM_WORLD, serial);
         ParFiniteElementSpace parallel_space(&mesh, &collection);
         ParBilinearForm parallel_penalty(&parallel_space);
         parallel_penalty.AddInteriorFaceIntegrator(new PressureCIPIntegrator(one, one, 0.01));
         parallel_penalty.Assemble();
         parallel_penalty.Finalize();
         std::unique_ptr<HypreParMatrix> penalty(parallel_penalty.ParallelAssemble());
         ParGridFunction field(&parallel_space);
         Vector values, action(parallel_space.GetTrueVSize());
         for (Coefficient *coefficient : std::array<Coefficient*, 3>{&one, &linear, &kink})
         {
            field.ProjectCoefficient(*coefficient);
            field.GetTrueDofs(values);
            penalty->Mult(values, action);
            if (coefficient == &kink)
            {
               tests.Near(InnerProduct(MPI_COMM_WORLD, values, action), serial_energy,
                          1e-10, "CIP shared-face energy equals serial energy");
            }
            else
            {
               tests.Require(Norm(action) < 1e-10 * (1.0 + penalty->FNorm()), "CIP constant/linear nullspace");
            }
         }
         Vector other(values.Size()), other_action(values.Size());
         for (int dof = 0; dof < values.Size(); ++dof)
         {
            values(dof) = std::sin(dof + 1.0 + Mpi::WorldRank());
            other(dof) = std::cos(2.0 * dof + Mpi::WorldRank());
         }
         penalty->Mult(values, action);
         penalty->Mult(other, other_action);
         tests.Require(InnerProduct(MPI_COMM_WORLD, values, action) >= -1e-12,
                       "CIP distributed positive semidefinite energy");
         tests.Near(InnerProduct(MPI_COMM_WORLD, values, other_action),
                    InnerProduct(MPI_COMM_WORLD, other, action),
                    1e-10, "CIP distributed symmetry");

         if (order == 2 && (element == "quad" || element == "hex"))
         {
            FunctionCoefficient spurious([](const Vector &point)
            {
               real_t value = 1.0;
               for (int axis = 0; axis < point.Size(); ++axis)
               {
                  const real_t centered = point(axis) - 0.5;
                  value *= centered - (10.0 / 3.0) * centered * std::abs(centered);
               }
               return value;
            });
            field.ProjectCoefficient(spurious);
            field.GetTrueDofs(values);
            values /= Norm(values);
            penalty->Mult(values, action);
            tests.Require(Norm(action) < 1e-10, "Known coarse tensor C1 pressure mode is in CIP kernel");
            ParFiniteElementSpace velocity_space(&mesh, &collection, dimension, Ordering::byVDIM);
            ParMixedBilinearForm divergence(&velocity_space, &parallel_space);
            divergence.AddDomainIntegrator(new VectorDivergenceIntegrator);
            divergence.Assemble();
            divergence.Finalize();
            std::unique_ptr<HypreParMatrix> divergence_matrix(divergence.ParallelAssemble());
            Vector momentum(velocity_space.GetTrueVSize());
            divergence_matrix->MultTranspose(values, momentum);
            Array<int> boundary(mesh.bdr_attributes.Max()), essential;
            boundary = 1;
            velocity_space.GetEssentialTrueDofs(boundary, essential);
            momentum.SetSubVector(essential, 0.0);
            tests.Require(Norm(momentum) < 1e-10, "Known coarse tensor pressure mode is invisible to free velocity tests");
         }

         for (bool variable_mu : {false, true})
         {
            FunctionCoefficient viscosity([variable_mu](const Vector &point)
            { return variable_mu ? 1.0 + point.Sum() : 1.0; });
            Vector gradient_value(dimension);
            gradient_value = variable_mu ? 1.0 : 0.0;
            VectorConstantCoefficient gradient(gradient_value);
            VectorFunctionCoefficient force(dimension, [=](const Vector &point, Vector &value)
            {
               const real_t mu = variable_mu ? 1.0 + point.Sum() : 1.0;
               value.SetSize(dimension);
               for (int axis = 0; axis < dimension; ++axis)
               {
                  value(axis) = axis + 1.0 - (variable_mu ? 2.0 * point(axis) : 0.0);
               }
               value(0) -= 2.0 * mu * (dimension + 1) + (variable_mu ? 2.0 * point.Sum() : 0.0);
            });
            for (int cell = 0; cell < serial.GetNE(); ++cell)
            {
               auto &transformation = *serial.GetElementTransformation(cell);
               const auto &finite_element = *space.GetFE(cell);
               const int dofs = finite_element.GetDof();
               DenseMatrix velocity_matrix, coupling, transpose_coupling, pressure_matrix;
               GLSVelocityIntegrator velocity_integrator(viscosity, gradient, tau, 2 * order + 4);
               PSPGSymGradIntegrator coupling_integrator(viscosity, gradient, tau, 2 * order + 4);
               TransposeIntegrator transpose_integrator(new PSPGSymGradIntegrator(viscosity, gradient, tau, 2 * order + 4));
               RuleBilinearIntegrator<DiffusionIntegrator> pressure_integrator(2 * order + 4, tau);
               GLSForceIntegrator velocity_force(viscosity, gradient, tau, force, 2 * order + 4);
               ScalarVectorProductCoefficient tau_force(tau, force);
               RuleLinearIntegrator<DomainLFGradIntegrator> pressure_force(2 * order + 4, tau_force);
               velocity_integrator.AssembleElementMatrix(finite_element, transformation, velocity_matrix);
               coupling_integrator.AssembleElementMatrix2(finite_element, finite_element, transformation, coupling);
               transpose_integrator.AssembleElementMatrix2(finite_element, finite_element, transformation, transpose_coupling);
               pressure_integrator.AssembleElementMatrix(finite_element, transformation, pressure_matrix);
               Vector velocity(dimension * dofs), pressure(dofs), velocity_rhs, pressure_rhs, point(dimension);
               velocity = 0.0;
               for (int node = 0; node < dofs; ++node)
               {
                  transformation.Transform(finite_element.GetNodes().IntPoint(node), point);
                  velocity(node) = point * point;
                  pressure(node) = 0.0;
                  for (int axis = 0; axis < dimension; ++axis) { pressure(node) += (axis + 1) * point(axis); }
               }
               velocity_force.AssembleRHSElementVect(finite_element, transformation, velocity_rhs);
               pressure_force.AssembleRHSElementVect(finite_element, transformation, pressure_rhs);
               Vector velocity_action(velocity.Size()), pressure_action(pressure.Size());
               velocity_matrix.Mult(velocity, velocity_action);
               transpose_coupling.AddMult(pressure, velocity_action);
               coupling.Mult(velocity, pressure_action);
               pressure_matrix.AddMult(pressure, pressure_action);
               velocity_action -= velocity_rhs;
               pressure_action -= pressure_rhs;
               tests.Require(velocity_action.Norml2() < 1e-10 * (1.0 + velocity_rhs.Norml2()), "GLS H*u+D^T*p=F_H analytic residual");
               tests.Require(pressure_action.Norml2() < 1e-10 * (1.0 + pressure_rhs.Norml2()), "GLS D*u+S*p=G analytic residual");
               tests.Require(velocity_rhs.Norml2() > 1e-4, "GLS source test detects missing F_H");
               tests.Require(velocity_matrix.FNorm() > 1e-4 && coupling.FNorm() > 1e-4, "GLS high-order H and D nonzero");
               real_t asymmetry = 0.0, transpose_error = 0.0;
               for (int row = 0; row < velocity.Size(); ++row)
               {
                  for (int column = 0; column < velocity.Size(); ++column)
                  {
                     asymmetry = std::max(asymmetry, std::abs(velocity_matrix(row, column) - velocity_matrix(column, row)));
                  }
                  for (int column = 0; column < pressure.Size(); ++column)
                  {
                     transpose_error = std::max(transpose_error, std::abs(transpose_coupling(row, column) - coupling(column, row)));
                  }
                  velocity(row) = std::sin(row + 0.3);
               }
               velocity_matrix.Mult(velocity, velocity_action);
               tests.Require(velocity * velocity_action >= -1e-10, "GLS H positive semidefinite");
               tests.Near(asymmetry, 0.0, 1e-12 * (1.0 + velocity_matrix.FNorm()), "GLS H symmetric");
               tests.Near(transpose_error, 0.0, 1e-12 * (1.0 + coupling.FNorm()), "GLS D transpose");
            }
         }
      }
   }
}

void AddMMSBoundary(StokesSolver &solver, ParMesh &mesh, ManufacturedSolution &exact)
{
   for (int index = 0; index < mesh.bdr_attributes.Size(); ++index)
   {
      solver.AddVelocityBC(mesh.bdr_attributes[index], exact.velocity);
   }
   solver.SetVolForce(exact.force);
}

LinearSolverType ReferenceSolver()
{
#ifdef MFEM_USE_MUMPS
   return LinearSolverType::MUMPS;
#else
   return LinearSolverType::FGMRES;
#endif
}

void TimingTests(Tests &tests)
{
   Mesh serial = MakeStokesMesh("tri", 3);
   ParMesh mesh(MPI_COMM_WORLD, serial);
   H1_FECollection velocity_collection(2, 2);
   ParFiniteElementSpace velocity_space(&mesh, &velocity_collection, 2, Ordering::byVDIM);
   ManufacturedSolution exact(2, 1.0, true, 0.2);
   std::vector<LinearSolverType> backends = {LinearSolverType::GMRES, LinearSolverType::FGMRES};
#ifdef MFEM_USE_MUMPS
   backends.push_back(LinearSolverType::MUMPS);
#endif
   for (auto method : {StokesStabilizationMethod::PSPG, StokesStabilizationMethod::GLS,
                       StokesStabilizationMethod::CIP, StokesStabilizationMethod::TaylorHood})
   {
      H1_FECollection pressure_collection(method == StokesStabilizationMethod::TaylorHood ? 1 : 2, 2);
      ParFiniteElementSpace pressure_space(&mesh, &pressure_collection);
      for (auto backend : backends)
      {
         for (auto preconditioner : {StokesPreconditionerType::BlockDiagonalAMG,
                                     StokesPreconditionerType::BlockDiagonalP1AMG,
                                     StokesPreconditionerType::BlockUpperTriangularP1AMG,
                                     StokesPreconditionerType::BlockLowerTriangularP1AMG,
                                     StokesPreconditionerType::BlockDiagonalP1AMGAssembled,
                                     StokesPreconditionerType::BlockUpperTriangularP1AMGAssembled,
                                     StokesPreconditionerType::BlockLowerTriangularP1AMGAssembled})
         {
            if (backend == LinearSolverType::MUMPS && preconditioner != StokesPreconditionerType::BlockDiagonalAMG)
            {
               continue;
            }
            StokesSolverOptions options;
            options.method = method;
            options.linear_solver = backend;
            options.preconditioner = preconditioner;
            options.relative_tolerance = 1e-12;
            options.absolute_tolerance = 1e-14;
            options.max_iterations = 2000;
            options.print_level = 0;
            tests.context = std::string("timing ") + StokesStabilizationMethodName(method) + " " +
                            StokesPreconditionerName(preconditioner) + " backend=" + std::to_string(int(backend));
            auto solve_and_check = [&](StokesSolver &solver, bool assembled_here)
            {
               const double started = MPI_Wtime();
               const auto info = solver.Solve();
               const double local_elapsed = MPI_Wtime() - started;
               double elapsed = 0.0;
               MPI_Allreduce(&local_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
               CheckAssemblyTime(tests, info);
               tests.Require(info.assembly_performed == assembled_here, "Per-call assembly flag");
               tests.Require(backend == LinearSolverType::MUMPS ? info.preconditioner_setup_seconds == 0.0 :
                             info.direct_setup_seconds == 0.0, "Unused backend setup time is zero");
               tests.Require(backend == LinearSolverType::MUMPS ? info.direct_setup_seconds > 0.0 :
                             info.preconditioner_setup_seconds > 0.0, "Active backend setup is measured");
               if (assembled_here) { tests.Require(info.assembly_seconds <= elapsed, "Current assembly is inside solve"); }
               for (double seconds : {info.preconditioner_setup_seconds, info.direct_setup_seconds,
                                      info.linear_solve_seconds, info.finalization_seconds})
               {
                  tests.Require(seconds <= elapsed, "Each phase is inside the inclusive solve");
               }
               return info;
            };
            StokesSolver solver(velocity_space, pressure_space, exact.viscosity, exact.viscosity_gradient, options);
            AddMMSBoundary(solver, mesh, exact);
            const auto initial = solve_and_check(solver, true);
            tests.Require(initial.converged, "Timed initial solve converges");
            Vector initial_velocity, initial_pressure;
            solver.GetVelocity().GetTrueDofs(initial_velocity);
            solver.GetPressure().GetTrueDofs(initial_pressure);
            for (bool reassemble : {false, true})
            {
               if (reassemble) { solver.Assemble(); }
               const auto repeated = solve_and_check(solver, false);
               tests.Require(repeated.converged && repeated.iterations == initial.iterations,
                             "Repeated/explicitly assembled solve preserves convergence and iterations");
               if (!reassemble)
               {
                  tests.Require(repeated.assembly_seconds == initial.assembly_seconds,
                                "Cached solve retains historical assembly time");
               }
               Vector difference;
               solver.GetVelocity().GetTrueDofs(difference);
               difference -= initial_velocity;
               tests.Near(Norm(difference), 0.0, 1e-11, "Repeated velocity unchanged");
               solver.GetPressure().GetTrueDofs(difference);
               difference -= initial_pressure;
               tests.Near(Norm(difference), 0.0, 1e-10, "Repeated pressure unchanged");
            }
            if (preconditioner == StokesPreconditionerType::BlockDiagonalAMG)
            {
               if (backend != LinearSolverType::MUMPS)
               {
                  options.max_iterations = 1;
                  StokesSolver incomplete(velocity_space, pressure_space, exact.viscosity, exact.viscosity_gradient, options);
                  AddMMSBoundary(incomplete, mesh, exact);
                  const auto failed = solve_and_check(incomplete, true);
                  tests.Require(!failed.converged && failed.iterations == 1, "Failed solve still reports timings");
               }
               solver.DelVelocityBC();
               for (int attribute : mesh.bdr_attributes) { solver.AddVelocityBC(attribute, -1, 0.0); }
               solver.SetVolForce(0.0, 0.0);
               const auto zero = solve_and_check(solver, true);
               tests.Require(zero.converged && zero.iterations == 0, "Zero RHS timing and iteration reporting");
            }
         }
      }
   }
}

void SolverTests(Tests &tests, StokesStabilizationMethod method = StokesStabilizationMethod::PSPG,
                 StokesPreconditionerType preconditioner = StokesPreconditionerType::BlockDiagonalAMG,
                 int geometry_order = 1)
{
   for (const std::string element : {"tri", "quad", "tet", "hex"})
   {
      for (int order : {2, 3})
      {
         Mesh serial = MakeStokesMesh(element, method == StokesStabilizationMethod::CIP && (element == "quad" || element == "hex") ? 3 : 2);
         CurveTestMesh(serial, geometry_order);
         ParMesh mesh(MPI_COMM_WORLD, serial);
         const int dimension = mesh.Dimension();
         H1_FECollection collection(order, dimension);
         H1_FECollection pressure_collection(method == StokesStabilizationMethod::TaylorHood ? order - 1 : order, dimension);
         ParFiniteElementSpace velocity_space(&mesh, &collection, dimension, Ordering::byVDIM);
         ParFiniteElementSpace pressure_space(&mesh, &pressure_collection);
         ManufacturedSolution exact(dimension, 1.0, order == 3, 0.2);
         StokesSolverOptions options;
         options.method = method;
         options.preconditioner = preconditioner;
         if (method == StokesStabilizationMethod::PSPG && geometry_order > 1)
         {
            options.relative_tolerance = 1e-13;
            options.absolute_tolerance = 1e-15;
         }
         tests.context = std::string(StokesStabilizationMethodName(method)) + " " + element + " k=" + std::to_string(order);
         tests.context += std::string(" ") + StokesPreconditionerName(preconditioner);
         options.print_level = 0;
         options.linear_solver = ReferenceSolver();
         StokesSolver reference(velocity_space, pressure_space, exact.viscosity,
                                    exact.viscosity_gradient, options);
         AddMMSBoundary(reference, mesh, exact);
         const StokesSolveInfo reference_info = reference.Solve();
         CheckAssemblyTime(tests, reference_info);
         tests.Require(reference_info.geometry_order == geometry_order, "Coordinate degree diagnostic");
         tests.Require(reference_info.converged, element + " direct reference converges");
         tests.Require(reference_info.pressure_gauge_active, "Closed boundary activates gauge");
         tests.Near(reference_info.pressure_mean, 0.0, 1e-10, "Exact pressure mean constraint");
         tests.Near(reference_info.boundary_flux, 0.0, 1e-10, "Nonzero Dirichlet net flux");
         tests.Near(reference_info.gauge_multiplier, 0.0, 1e-9, "Gauge does not repair continuity");
         Vector reference_velocity, reference_pressure;
         reference.GetVelocity().GetTrueDofs(reference_velocity);
         reference.GetPressure().GetTrueDofs(reference_pressure);
         if (method == StokesStabilizationMethod::TaylorHood)
         {
            tests.Require(pressure_space.GetMaxElementOrder() == order - 1,
                          "Taylor-Hood pressure degree k-1");
            tests.Require(pressure_space.GlobalTrueVSize() < velocity_space.GlobalTrueVSize() / dimension,
                          "Taylor-Hood distinct pressure DOF count");
            ParMixedBilinearForm divergence(&velocity_space, &pressure_space);
            divergence.AddDomainIntegrator(new RuleBilinearIntegrator<VectorDivergenceIntegrator>(reference_info.integration_order));
            divergence.Assemble();
            divergence.Finalize();
            std::unique_ptr<HypreParMatrix> divergence_matrix(divergence.ParallelAssemble());
            Vector residual(pressure_space.GetTrueVSize());
            divergence_matrix->Mult(reference_velocity, residual);
            tests.Require(Norm(residual) < 1e-9, "Taylor-Hood continuity has no S or D contribution");
            ParBilinearForm elasticity(&velocity_space);
            elasticity.AddDomainIntegrator(new RuleBilinearIntegrator<ElasticityIntegrator>(reference_info.integration_order, exact.viscosity, 0.0, 1.0));
            elasticity.Assemble();
            elasticity.Finalize();
            std::unique_ptr<HypreParMatrix> elasticity_matrix(elasticity.ParallelAssemble());
            ParLinearForm forcing(&velocity_space);
            forcing.AddDomainIntegrator(new RuleLinearIntegrator<VectorDomainLFIntegrator>(reference_info.integration_order, exact.force));
            forcing.Assemble();
            Vector force(velocity_space.GetTrueVSize()), pressure_action(force.Size());
            forcing.ParallelAssemble(force);
            residual.SetSize(force.Size());
            elasticity_matrix->Mult(reference_velocity, residual);
            divergence_matrix->MultTranspose(reference_pressure, pressure_action);
            residual -= pressure_action;
            residual -= force;
            Array<int> boundary(mesh.bdr_attributes.Max()), essential;
            boundary = 1;
            velocity_space.GetEssentialTrueDofs(boundary, essential);
            residual.SetSubVector(essential, 0.0);
            tests.Require(Norm(residual) < 1e-9 * (1.0 + Norm(force)), "Taylor-Hood momentum has no H or D^T contribution");
         }
         for (LinearSolverType backend : {LinearSolverType::GMRES, LinearSolverType::FGMRES})
         {
            options.linear_solver = backend;
            if (method != StokesStabilizationMethod::PSPG)
            {
               options.relative_tolerance = 1e-12;
               options.absolute_tolerance = 1e-14;
            }
            if (preconditioner != StokesPreconditionerType::BlockDiagonalAMG)
            {
               options.relative_tolerance = 1e-13;
               options.absolute_tolerance = 1e-15;
            }
            if (method == StokesStabilizationMethod::CIP)
            {
               options.relative_tolerance = 1e-14;
               options.absolute_tolerance = 1e-16;
               options.restart_dimension = 200;
               options.max_iterations = 3000;
            }
            options.amg_elasticity_interp_refine = backend == LinearSolverType::FGMRES;
            StokesSolver iterative(velocity_space, pressure_space, exact.viscosity,
                                       exact.viscosity_gradient, options);
            AddMMSBoundary(iterative, mesh, exact);
            const StokesSolveInfo info = iterative.Solve();
            CheckAssemblyTime(tests, info);
            tests.Require(info.converged, element + " elasticity AMG Krylov convergence");
            Vector velocity_difference, pressure_difference;
            iterative.GetVelocity().GetTrueDofs(velocity_difference);
            iterative.GetPressure().GetTrueDofs(pressure_difference);
            velocity_difference -= reference_velocity;
            pressure_difference -= reference_pressure;
            tests.Require(Norm(velocity_difference) / (1.0 + Norm(reference_velocity)) < 1e-7,
                          "Krylov/MUMPS velocity agreement");
            tests.Near(Norm(pressure_difference) / (1.0 + Norm(reference_pressure)), 0.0, 1e-7,
                       "Krylov/MUMPS pressure agreement");
         }
         options.linear_solver = LinearSolverType::FGMRES;
         options.max_iterations = 1;
         StokesSolver incomplete(velocity_space, pressure_space, exact.viscosity,
                                     exact.viscosity_gradient, options);
         AddMMSBoundary(incomplete, mesh, exact);
         const StokesSolveInfo incomplete_info = incomplete.Solve();
         tests.Require(!incomplete_info.converged, "Nonconvergence is reported");
         CheckAssemblyTime(tests, incomplete_info);
         if (Mpi::Root()) { out << "SOLVERS method=" << StokesStabilizationMethodName(method) << " element=" << element << " order=" << order << '\n'; }
      }
   }
}

void PolynomialTests(Tests &tests, StokesStabilizationMethod method)
{
   for (const std::string element : {"tri", "quad", "tet", "hex"})
   {
      Mesh serial = MakeStokesMesh(element, method == StokesStabilizationMethod::CIP && (element == "quad" || element == "hex") ? 3 : 2);
      ParMesh mesh(MPI_COMM_WORLD, serial);
      const int dimension = mesh.Dimension();
      H1_FECollection velocity_collection(2, dimension);
      H1_FECollection pressure_collection(method == StokesStabilizationMethod::TaylorHood ? 1 : 2, dimension);
      ParFiniteElementSpace velocity_space(&mesh, &velocity_collection, dimension, Ordering::byVDIM);
      ParFiniteElementSpace pressure_space(&mesh, &pressure_collection);
      for (bool variable_mu : {false, true})
      {
         tests.context = std::string(StokesStabilizationMethodName(method)) + " polynomial " + element +
                         " variable_mu=" + std::to_string(variable_mu);
         FunctionCoefficient viscosity([variable_mu](const Vector &point)
         { return variable_mu ? 1.0 + point.Sum() : 1.0; });
         Vector gradient_value(dimension);
         gradient_value = variable_mu ? 1.0 : 0.0;
         VectorConstantCoefficient gradient(gradient_value);
         VectorFunctionCoefficient exact_velocity(dimension, [dimension](const Vector &point, Vector &value)
         {
            value.SetSize(dimension);
            value = 0.3;
            value(0) = point(0) * point(0) + 0.1;
            value(1) = -2.0 * point(0) * point(1) + 0.2;
         });
         FunctionCoefficient exact_pressure([](const Vector &point)
         {
            real_t value = 0.0;
            for (int axis = 0; axis < point.Size(); ++axis) { value += (axis + 1) * (point(axis) - 0.5); }
            return value;
         });
         VectorFunctionCoefficient force(dimension, [=](const Vector &point, Vector &value)
         {
            value.SetSize(dimension);
            for (int axis = 0; axis < dimension; ++axis) { value(axis) = axis + 1.0; }
            const real_t mu = variable_mu ? 1.0 + point.Sum() : 1.0;
            value(0) -= 2.0 * mu;
            if (variable_mu)
            {
               value(0) += -4.0 * point(0) + 2.0 * point(1);
               value(1) += 2.0 * point(1) + 4.0 * point(0);
            }
         });
         StokesSolverOptions options;
         options.method = method;
         options.linear_solver = ReferenceSolver();
         options.print_level = 0;
         options.relative_tolerance = 1e-12;
         options.absolute_tolerance = 1e-14;
         StokesSolver solver(velocity_space, pressure_space, viscosity, gradient, options);
         for (int index = 0; index < mesh.bdr_attributes.Size(); ++index)
         {
            solver.AddVelocityBC(mesh.bdr_attributes[index], exact_velocity);
         }
         solver.SetVolForce(force);
         tests.Require(solver.Solve().converged, "Polynomial consistency solve");
         tests.Require(solver.GetVelocity().ComputeL2Error(exact_velocity) < 1e-8, "Polynomial exact velocity and nonzero lifting");
         tests.Require(solver.GetPressure().ComputeL2Error(exact_pressure) < 1e-8, "Polynomial exact pressure and GLS source consistency");
      }
   }
}

void BoundaryTests(Tests &tests, LinearSolverType backend, StokesStabilizationMethod method = StokesStabilizationMethod::PSPG,
                   StokesPreconditionerType preconditioner = StokesPreconditionerType::BlockDiagonalAMG,
                   int geometry_order = 1, real_t relative_tolerance = 1e-12)
{
   for (int dimension : {2, 3})
   {
      Mesh serial = MakeStokesMesh(dimension == 2 ? "quad" : "hex", method == StokesStabilizationMethod::CIP ? 3 : 2);
      CurveTestMesh(serial, geometry_order);
      ParMesh mesh(MPI_COMM_WORLD, serial);
      const int order = std::max(2, geometry_order + (method == StokesStabilizationMethod::TaylorHood ? 1 : 0));
      H1_FECollection collection(order, dimension);
      H1_FECollection pressure_collection(method == StokesStabilizationMethod::TaylorHood ? order - 1 : order, dimension);
      ParFiniteElementSpace velocity_space(&mesh, &collection, dimension, Ordering::byVDIM);
      ParFiniteElementSpace pressure_space(&mesh, &pressure_collection);
      StokesSolverOptions options;
      options.method = method;
      tests.context = std::string(StokesStabilizationMethodName(method)) + " boundary dim=" + std::to_string(dimension) +
                      " backend=" + std::to_string(static_cast<int>(backend));
      options.print_level = 0;
      options.linear_solver = backend;
      options.preconditioner = preconditioner;
      tests.context += std::string(" ") + StokesPreconditionerName(preconditioner);
      options.relative_tolerance = relative_tolerance;
      options.absolute_tolerance = 0.01 * relative_tolerance;
      if (method == StokesStabilizationMethod::CIP)
      {
         options.relative_tolerance = 1e-14;
         options.absolute_tolerance = 1e-16;
         options.restart_dimension = 200;
         options.max_iterations = 3000;
      }
      Vector plug_value(dimension);
      for (int component = 0; component < dimension; ++component) { plug_value(component) = component + 1.0; }
      VectorConstantCoefficient plug(plug_value);
      ConstantCoefficient zero(0.0);
      {
         VectorFunctionCoefficient linear_velocity(dimension,
         [dimension](const Vector &point, Vector &value)
         {
            value = point;
            value(dimension - 1) *= -(dimension - 1);
         });
         std::vector<std::unique_ptr<FunctionCoefficient>> components;
         for (int component = 0; component < dimension; ++component)
         {
            components.push_back(std::make_unique<FunctionCoefficient>(
            [component, dimension](const Vector &point)
            {
               return point(component) * (component == dimension - 1 ? -(dimension - 1) : 1);
            }));
         }
         StokesSolver solver(velocity_space, pressure_space, 1.0, options);
         for (int index = 0; index < mesh.bdr_attributes.Size(); ++index)
         {
            for (int component = 0; component < dimension; ++component)
            {
               solver.AddVelocityBC(mesh.bdr_attributes[index], component, *components[component]);
            }
         }
         tests.Require(solver.Solve().converged, "Nonconstant scalar component BC solve");
         tests.Require(solver.GetVelocity().ComputeL2Error(linear_velocity) < 1e-8,
                       "Nonconstant scalar component BC reproduction");
         tests.Require(solver.GetPressure().ComputeL2Error(zero) < 1e-8,
                       "Affine divergence-free velocity has zero pressure");
      }
      for (int mode = 0; mode < 3; ++mode)
      {
         StokesSolver solver(velocity_space, pressure_space, 1.0, options);
         for (int index = 0; index < mesh.bdr_attributes.Size(); ++index)
         {
            const int attribute = mesh.bdr_attributes[index];
            if (mode == 0)
            {
               solver.AddVelocityBC(attribute, -1, 17.0);
               solver.AddVelocityBC(attribute, plug);
            }
            else if (mode == 1)
            {
               solver.AddVelocityBC(attribute, plug);
               solver.AddVelocityBC(attribute, 0, 1.0);
               for (int component = 1; component < dimension; ++component)
               {
                  solver.AddVelocityBC(attribute, component, component + 1.0);
               }
            }
            else
            {
               int normal_axis = -1;
               for (int boundary = 0; boundary < mesh.GetNBE(); ++boundary)
               {
                  if (mesh.GetBdrAttribute(boundary) != attribute) { continue; }
                  ElementTransformation &transformation = *mesh.GetBdrElementTransformation(boundary);
                  transformation.SetIntPoint(&Geometries.GetCenter(transformation.GetGeometryType()));
                  Vector normal(dimension);
                  CalcOrtho(transformation.Jacobian(), normal);
                  for (int axis = 0; axis < dimension; ++axis)
                  {
                     if (std::abs(normal(axis)) > 1e-12) { normal_axis = axis; }
                  }
               }
               int global_axis = -1;
               MPI_Allreduce(&normal_axis, &global_axis, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
               solver.AddVelocityBC(attribute, global_axis, plug_value(global_axis));
            }
         }
         const StokesSolveInfo info = solver.Solve();
         tests.Require(info.converged && info.pressure_gauge_active, "Plug/slip gauge solve");
         tests.Require(solver.GetVelocity().ComputeL2Error(plug) < 1e-8, "Component/vector/slip BC equivalence");
         tests.Near(solver.GetPressure().ComputeL2Error(zero), 0.0, 1e-8, "Plug pressure zero");
         solver.DelVelocityBC();
         for (int index = 0; index < mesh.bdr_attributes.Size(); ++index)
         {
            solver.AddVelocityBC(mesh.bdr_attributes[index], -1, 0.0);
         }
         tests.Require(solver.Solve().converged, "Clear BCs and reassemble");
      }

      FunctionCoefficient exact_pressure([](const Vector &point) { return point(0) - 1.0; });
      Vector zero_values(dimension);
      zero_values = 0.0;
      VectorConstantCoefficient zero_velocity(zero_values);
      Vector constant_force(dimension);
      constant_force = 0.0;
      constant_force(0) = 1.0;
      VectorConstantCoefficient force_coefficient(constant_force);
      for (int force_mode = 0; force_mode < 3; ++force_mode)
      {
         StokesSolver solver(velocity_space, pressure_space, 1.0, options);
         for (int index = 0; index < mesh.bdr_attributes.Size(); ++index)
         {
            const int attribute = mesh.bdr_attributes[index];
            int local_open = 0, global_open = 0;
            for (int boundary = 0; boundary < mesh.GetNBE(); ++boundary)
            {
               if (mesh.GetBdrAttribute(boundary) != attribute) { continue; }
               ElementTransformation &transformation = *mesh.GetBdrElementTransformation(boundary);
               Vector center(dimension);
               transformation.Transform(Geometries.GetCenter(transformation.GetGeometryType()), center);
               if (std::abs(center(0) - 1.0) < 1e-12) { local_open = 1; }
            }
            MPI_Allreduce(&local_open, &global_open, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
            if (!global_open) { solver.AddVelocityBC(attribute, zero_velocity); }
         }
         if (force_mode == 0) { solver.SetVolForce(force_coefficient); }
         if (force_mode == 1) { solver.SetVolForce(constant_force); }
         if (force_mode == 2) { solver.SetVolForce(1.0, 0.0, 0.0); }
         const StokesSolveInfo info = solver.Solve();
         tests.Require(info.converged && !info.pressure_gauge_active, "Open boundary determines pressure");
         tests.Near(info.pressure_mean, -0.5, 1e-8, "Do not normalize open-boundary pressure");
         tests.Require(solver.GetPressure().ComputeL2Error(exact_pressure) < 1e-8, "Force overload pressure");
         tests.Require(solver.GetVelocity().ComputeL2Error(zero_velocity) < 1e-8, "Force overload velocity");
      }
   }
}

void ConvergenceTests(Tests &tests, StokesStabilizationMethod method = StokesStabilizationMethod::PSPG,
                      int geometry_order = 1)
{
   for (const std::string element : {"tri", "quad", "tet", "hex"})
   {
      const int dimension = element == "tri" || element == "quad" ? 2 : 3;
      for (int order : {2, 3})
      {
         for (bool variable_mu : {false, true})
         {
            StokesErrors previous = {};
            int previous_cells = 0;
            const bool curved_pspg_cubic_hex = geometry_order > 1 && element == "hex" &&
                                               order == 3 && method == StokesStabilizationMethod::PSPG;
            for (int level = 0; level < 3; ++level)
            {
               const int cells = dimension == 2 ? 4 * (1 << level) :
                                 level + ((method == StokesStabilizationMethod::CIP ||
                                           method == StokesStabilizationMethod::TaylorHood || curved_pspg_cubic_hex) &&
                                          element == "hex" ? 3 : 2);
               Mesh serial = MakeStokesMesh(element, cells);
               CurveTestMesh(serial, geometry_order);
               ParMesh mesh(MPI_COMM_WORLD, serial);
               H1_FECollection collection(order, dimension);
               H1_FECollection pressure_collection(method == StokesStabilizationMethod::TaylorHood ? order - 1 : order, dimension);
               ParFiniteElementSpace velocity_space(&mesh, &collection, dimension, Ordering::byVDIM);
               ParFiniteElementSpace pressure_space(&mesh, &pressure_collection);
               ManufacturedSolution exact(dimension, 1.0, variable_mu, 0.1);
               StokesSolverOptions options;
               options.method = method;
               tests.context = std::string(StokesStabilizationMethodName(method)) + " " + element + " k=" +
                               std::to_string(order) + " cells=" + std::to_string(cells) +
                               " variable_mu=" + std::to_string(variable_mu);
               options.print_level = 0;
               options.linear_solver = ReferenceSolver();
               StokesSolver solver(velocity_space, pressure_space, exact.viscosity,
                                       exact.viscosity_gradient, options);
               AddMMSBoundary(solver, mesh, exact);
               const auto info = solver.Solve();
               tests.Require(info.converged, "MMS convergence solve");
               const auto errors = ComputeErrors(solver.GetVelocity(), solver.GetPressure(), exact,
                                                  info.integration_order + 6);
               if (level > 0)
               {
                  tests.Require(errors.velocity_l2 < previous.velocity_l2, "MMS velocity L2 decreases");
                  tests.Require(errors.velocity_h1 < previous.velocity_h1, "MMS velocity H1 decreases");
                  tests.Require(errors.pressure_l2 < previous.pressure_l2, "MMS pressure L2 decreases");
                  if (level == 2 && (method != StokesStabilizationMethod::PSPG || curved_pspg_cubic_hex))
                  {
                     const real_t denominator = std::log(real_t(cells) / previous_cells);
                     tests.Require(std::log(previous.velocity_l2 / errors.velocity_l2) / denominator > order + 0.5,
                                   "Finest MMS velocity L2 rate exceeds k+0.5");
                     tests.Require(std::log(previous.velocity_h1 / errors.velocity_h1) / denominator > order - 0.5,
                                   "Finest MMS velocity H1 rate exceeds k-0.5");
                     if (method == StokesStabilizationMethod::TaylorHood)
                     {
                        tests.Require(std::log(previous.pressure_l2 / errors.pressure_l2) / denominator > order - 0.5,
                                   "Finest Taylor-Hood pressure rate exceeds k-0.5");
                     }
                     if (curved_pspg_cubic_hex)
                     {
                        tests.Require(std::log(previous.pressure_l2 / errors.pressure_l2) / denominator > order - 0.5,
                                      "Finest curved PSPG cubic-hex pressure rate exceeds k-0.5");
                     }
                  }
               }
               if (Mpi::Root())
               {
                  out << std::setprecision(10) << "CONVERGENCE method=" << StokesStabilizationMethodName(method)
                      << " element=" << element << " order=" << order
                      << " variable_mu=" << variable_mu << " cells=" << cells
                      << " velocity_l2=" << errors.velocity_l2 << " velocity_h1=" << errors.velocity_h1
                      << " pressure_l2=" << errors.pressure_l2 << " divergence=" << errors.divergence_l2;
                  if (level > 0)
                  {
                     const real_t denominator = std::log(real_t(cells) / previous_cells);
                     out << " rate_u_l2=" << std::log(previous.velocity_l2 / errors.velocity_l2) / denominator
                         << " rate_u_h1=" << std::log(previous.velocity_h1 / errors.velocity_h1) / denominator
                         << " rate_p=" << std::log(previous.pressure_l2 / errors.pressure_l2) / denominator;
                  }
                  out << std::endl;
               }
               previous = errors;
               previous_cells = cells;
            }
         }
      }
   }
}

void SensitivityTests(Tests &tests, StokesStabilizationMethod method = StokesStabilizationMethod::PSPG,
                      int geometry_order = 1)
{
   for (const std::string element : {"quad", "tet"})
   {
      Mesh serial = MakeStokesMesh(element, method == StokesStabilizationMethod::CIP && element == "quad" ? 3 : 2);
      CurveTestMesh(serial, geometry_order);
      ParMesh mesh(MPI_COMM_WORLD, serial);
      const int dimension = mesh.Dimension();
      H1_FECollection collection(2, dimension);
      H1_FECollection pressure_collection(method == StokesStabilizationMethod::TaylorHood ? 1 : 2, dimension);
      ParFiniteElementSpace velocity_space(&mesh, &collection, dimension, Ordering::byVDIM);
      ParFiniteElementSpace pressure_space(&mesh, &pressure_collection);
      ManufacturedSolution exact(dimension, 1.0, true, 0.125);
      Vector previous_velocity, previous_pressure;
      Vector parameter_velocity, parameter_pressure;
      for (real_t constant : {0.25, 1.0, 4.0})
      {
         for (int quadrature : {12, 16})
         {
            StokesSolverOptions options;
            options.method = method;
            options.gls_tau_scale = constant;
            options.cip_gamma *= constant;
            tests.context = std::string(StokesStabilizationMethodName(method)) + " sensitivity " + element +
                            " scale=" + std::to_string(constant);
            options.print_level = 0;
            options.linear_solver = ReferenceSolver();
            options.pspg_constant = constant;
            options.integration_order = quadrature;
            StokesSolver solver(velocity_space, pressure_space, exact.viscosity,
                                    exact.viscosity_gradient, options);
            AddMMSBoundary(solver, mesh, exact);
            const auto info = solver.Solve();
            tests.Require(info.converged, "Tau/quadrature sensitivity solve");
            Vector current_velocity, current_pressure;
            solver.GetVelocity().GetTrueDofs(current_velocity);
            solver.GetPressure().GetTrueDofs(current_pressure);
            if (method == StokesStabilizationMethod::TaylorHood && quadrature == 12)
            {
               if (parameter_velocity.Size())
               {
                  parameter_velocity -= current_velocity;
                  parameter_pressure -= current_pressure;
                  tests.Require(Norm(parameter_velocity) < 1e-11 && Norm(parameter_pressure) < 1e-11,
                                "Taylor-Hood ignores stabilization parameters");
               }
               parameter_velocity = current_velocity;
               parameter_pressure = current_pressure;
            }
            if (Mpi::Root())
            {
               out << "SENSITIVITY method=" << StokesStabilizationMethodName(method) << " element=" << element
                   << " factor=" << constant << " quadrature=" << quadrature
                   << " iterations=" << info.iterations << " residual=" << info.relative_residual << '\n';
            }
            if (quadrature == 16)
            {
               previous_velocity -= current_velocity;
               previous_pressure -= current_pressure;
               tests.Require(Norm(previous_velocity) / (1.0 + Norm(current_velocity)) < 1e-6,
                             "Velocity stabilizes with quadrature");
               tests.Require(Norm(previous_pressure) / (1.0 + Norm(current_pressure)) < 1e-6,
                             "Pressure stabilizes with quadrature");
               if (method != StokesStabilizationMethod::PSPG)
               {
                  options.linear_solver = LinearSolverType::FGMRES;
                  options.relative_tolerance = method == StokesStabilizationMethod::CIP ? 1e-14 : 1e-12;
                  options.absolute_tolerance = method == StokesStabilizationMethod::CIP ? 1e-16 : 1e-14;
                  options.restart_dimension = 200;
                  options.max_iterations = 3000;
                  StokesSolver iterative(velocity_space, pressure_space, exact.viscosity,
                                         exact.viscosity_gradient, options);
                  AddMMSBoundary(iterative, mesh, exact);
                  const auto iterative_info = iterative.Solve();
                  tests.Require(iterative_info.converged, "Parameter-sweep FGMRES convergence");
                  Vector difference;
                  iterative.GetPressure().GetTrueDofs(difference);
                  difference -= current_pressure;
                  tests.Near(Norm(difference) / (1.0 + Norm(current_pressure)), 0.0, 1e-7,
                             "Parameter-sweep direct/Krylov pressure agreement");
                  if (Mpi::Root())
                  {
                     out << "SWEEP method=" << StokesStabilizationMethodName(method) << " element=" << element
                         << " factor=" << constant << " iterations=" << iterative_info.iterations
                         << " residual=" << iterative_info.relative_residual << '\n';
                  }
               }
            }
            previous_velocity = current_velocity;
            previous_pressure = current_pressure;
         }
      }
   }
}

void PartitionTests(Tests &tests, StokesStabilizationMethod method = StokesStabilizationMethod::PSPG,
                    StokesPreconditionerType preconditioner = StokesPreconditionerType::BlockDiagonalAMG,
                    int geometry_order = 1)
{
   for (const std::string element : {"tri", "quad", "tet", "hex"})
   {
      Mesh serial = MakeStokesMesh(element, method == StokesStabilizationMethod::CIP && (element == "quad" || element == "hex") ? 3 : 2);
      CurveTestMesh(serial, geometry_order);
      const int dimension = serial.Dimension();
      ManufacturedSolution exact(dimension, 1.0, true, 0.2);
      H1_FECollection collection(3, dimension);
      H1_FECollection pressure_collection(method == StokesStabilizationMethod::TaylorHood ? 2 : 3, dimension);
      StokesSolverOptions options;
      options.method = method;
      if (method != StokesStabilizationMethod::PSPG)
      {
         options.relative_tolerance = 1e-12;
         options.absolute_tolerance = 1e-14;
      }
      if (method == StokesStabilizationMethod::CIP) { options.restart_dimension = 200; options.max_iterations = 3000; }
      tests.context = std::string(StokesStabilizationMethodName(method)) + " partition " + element;
      options.preconditioner = preconditioner;
      tests.context += std::string(" ") + StokesPreconditionerName(preconditioner);
      options.print_level = 0;
      options.linear_solver = ReferenceSolver();
      StokesErrors reference_errors;
      {
         ParMesh reference_mesh(MPI_COMM_SELF, serial);
         ParFiniteElementSpace velocity_space(&reference_mesh, &collection, dimension, Ordering::byVDIM);
         ParFiniteElementSpace pressure_space(&reference_mesh, &pressure_collection);
         StokesSolver reference(velocity_space, pressure_space, exact.viscosity,
                                    exact.viscosity_gradient, options);
         AddMMSBoundary(reference, reference_mesh, exact);
         tests.Require(reference.Solve().converged, "Single-rank partition reference");
         reference_errors = ComputeErrors(reference.GetVelocity(), reference.GetPressure(), exact, 14);
      }
      ParMesh mesh(MPI_COMM_WORLD, serial);
      ParFiniteElementSpace velocity_space(&mesh, &collection, dimension, Ordering::byVDIM);
      ParFiniteElementSpace pressure_space(&mesh, &pressure_collection);
      options.linear_solver = LinearSolverType::FGMRES;
      StokesSolver parallel(velocity_space, pressure_space, exact.viscosity,
                                exact.viscosity_gradient, options);
      AddMMSBoundary(parallel, mesh, exact);
      tests.Require(parallel.Solve().converged, "Distributed partition solve");
      const auto parallel_errors = ComputeErrors(parallel.GetVelocity(), parallel.GetPressure(), exact, 14);
      tests.Near(parallel_errors.velocity_l2, reference_errors.velocity_l2, 1e-7, "Partition-invariant velocity L2");
      tests.Near(parallel_errors.velocity_h1, reference_errors.velocity_h1, 1e-7, "Partition-invariant velocity H1");
      tests.Near(parallel_errors.pressure_l2, reference_errors.pressure_l2, 1e-7, "Partition-invariant pressure L2");
      tests.Near(parallel_errors.divergence_l2, reference_errors.divergence_l2, 1e-7, "Partition-invariant divergence");
   }
}

void ReassemblyTests(Tests &tests, StokesStabilizationMethod method = StokesStabilizationMethod::PSPG,
                     StokesPreconditionerType preconditioner = StokesPreconditionerType::BlockDiagonalAMG)
{
   Mesh serial = MakeStokesMesh("quad", method == StokesStabilizationMethod::CIP ? 3 : 2);
   ParMesh mesh(MPI_COMM_WORLD, serial);
   H1_FECollection collection(2, 2);
   H1_FECollection pressure_collection(method == StokesStabilizationMethod::TaylorHood ? 1 : 2, 2);
   ParFiniteElementSpace velocity_space(&mesh, &collection, 2, Ordering::byVDIM);
   ParFiniteElementSpace pressure_space(&mesh, &pressure_collection);
   ManufacturedSolution first(2, 1.0, true, 0.2), second(2, 3.0, true, 0.2);
   ManufacturedSolution *current = &first;
   FunctionCoefficient viscosity([&](const Vector &point) { return current->Viscosity(point); });
   VectorFunctionCoefficient gradient(2, [&](const Vector &, Vector &value)
   {
      value.SetSize(2);
      value = current == &first ? 1.0 : 3.0;
   });
   VectorFunctionCoefficient force(2, [&](const Vector &point, Vector &value) { current->Force(point, value); });
   StokesSolverOptions options;
   options.method = method;
   if (method == StokesStabilizationMethod::CIP) { options.restart_dimension = 200; options.max_iterations = 3000; }
   tests.context = std::string(StokesStabilizationMethodName(method)) + " reassembly";
   options.preconditioner = preconditioner;
   tests.context += std::string(" ") + StokesPreconditionerName(preconditioner);
   options.print_level = 0;
   options.relative_tolerance = 1e-12;
   options.absolute_tolerance = 1e-14;
   StokesSolver solver(velocity_space, pressure_space, viscosity, gradient, options);
   AddMMSBoundary(solver, mesh, first);
   solver.SetVolForce(force);
   const auto initial_info = solver.Solve();
   tests.Require(initial_info.converged, "Initial borrowed-coefficient solve");
   CheckAssemblyTime(tests, initial_info);
   const auto reused_info = solver.Solve();
   tests.Require(reused_info.converged, "Cached-system solve");
   tests.Require(reused_info.assembly_seconds == initial_info.assembly_seconds,
                 "Cached-system solve retains the last assembly time");
   Vector previous_pressure;
   solver.GetPressure().GetTrueDofs(previous_pressure);
   current = &second;
   solver.Assemble();
   const auto reassembled_info = solver.Solve();
   tests.Require(reassembled_info.converged, "Reassembled variable-viscosity AMG solve");
   CheckAssemblyTime(tests, reassembled_info);
   const auto reused_reassembly = solver.Solve();
   tests.Require(reused_reassembly.converged, "Explicitly assembled cached-system solve");
   tests.Require(reused_reassembly.assembly_seconds == reassembled_info.assembly_seconds,
                 "Explicit reassembly time is retained by subsequent solves");
   Vector updated_velocity, updated_pressure;
   solver.GetVelocity().GetTrueDofs(updated_velocity);
   solver.GetPressure().GetTrueDofs(updated_pressure);
   previous_pressure -= updated_pressure;
   tests.Require(Norm(previous_pressure) > 1e-3, "Viscosity update actually changes the discrete solution");
   options.linear_solver = ReferenceSolver();
   StokesSolver reference(velocity_space, pressure_space, second.viscosity,
                              second.viscosity_gradient, options);
   AddMMSBoundary(reference, mesh, second);
   tests.Require(reference.Solve().converged, "Reassembly direct reference");
   Vector reference_velocity, reference_pressure;
   reference.GetVelocity().GetTrueDofs(reference_velocity);
   reference.GetPressure().GetTrueDofs(reference_pressure);
   updated_velocity -= reference_velocity;
   updated_pressure -= reference_pressure;
   const real_t pressure_difference = Norm(updated_pressure) / (1.0 + Norm(reference_pressure));
   if (Mpi::Root()) { out << "REASSEMBLY relative_pressure_difference=" << pressure_difference << '\n'; }
   tests.Require(Norm(updated_velocity) / (1.0 + Norm(reference_velocity)) < 1e-7,
                 "Reassembled velocity agrees with fresh reference");
   tests.Require(pressure_difference < 1e-7,
                 "Reassembled pressure agrees with fresh reference");
}

void PreconditionerOperatorTests(Tests &tests, int geometry_order = 1)
{
   tests.context = "triangular block substitution";
   for (bool gauge : {false, true})
   {
      Array<int> offsets(gauge ? 4 : 3);
      offsets[0] = 0;
      offsets[1] = 2;
      offsets[2] = 4;
      if (gauge) { offsets[3] = 5; }
      DenseMatrix velocity(2), pressure(2), coupling(2), column(2, 1), gauge_matrix(1);
      velocity = 0.0;
      velocity(0, 0) = 2.0;
      velocity(1, 1) = 3.0;
      pressure(0, 0) = 4.0;
      pressure(1, 1) = 5.0;
      pressure(0, 1) = pressure(1, 0) = 1.0;
      coupling(0, 0) = 1.0;
      coupling(0, 1) = 2.0;
      coupling(1, 0) = -3.0;
      coupling(1, 1) = 4.0;
      column(0, 0) = 2.0;
      column(1, 0) = 3.0;
      gauge_matrix(0, 0) = -2.0;
      DenseMatrixInverse velocity_inverse(velocity), pressure_inverse(pressure), gauge_inverse(gauge_matrix);
      detail::UpperTriangularPreconditioner inverse(offsets, velocity_inverse, pressure_inverse,
            coupling, gauge ? &gauge_inverse : nullptr, gauge ? &column : nullptr);
      BlockOperator triangular(offsets);
      triangular.SetBlock(0, 0, &velocity);
      triangular.SetBlock(0, 1, &coupling);
      triangular.SetBlock(1, 1, &pressure);
      if (gauge)
      {
         triangular.SetBlock(1, 2, &column);
         triangular.SetBlock(2, 2, &gauge_matrix);
      }
      Vector exact(offsets.Last()), rhs(exact.Size()), result(exact.Size());
      for (int entry = 0; entry < exact.Size(); ++entry) { exact(entry) = entry + 0.5; }
      triangular.Mult(exact, rhs);
      result = 19.0;
      inverse.Mult(rhs, result);
      result -= exact;
      tests.Require(result.Norml2() < 1e-13, "Upper coupling and gauge signs");
      inverse.Mult(rhs, rhs);
      rhs -= exact;
      tests.Require(rhs.Norml2() < 1e-13, "Upper action supports aliasing");

      DenseMatrix row(1, 2);
      row(0, 0) = column(0, 0);
      row(0, 1) = column(1, 0);
      BlockLowerTriangularPreconditioner lower_inverse(offsets);
      lower_inverse.SetDiagonalBlock(0, &velocity_inverse);
      lower_inverse.SetDiagonalBlock(1, &pressure_inverse);
      lower_inverse.SetBlock(1, 0, &coupling);
      BlockOperator lower(offsets);
      lower.SetBlock(0, 0, &velocity);
      lower.SetBlock(1, 0, &coupling);
      lower.SetBlock(1, 1, &pressure);
      if (gauge)
      {
         lower_inverse.SetDiagonalBlock(2, &gauge_inverse);
         lower_inverse.SetBlock(2, 1, &row);
         lower.SetBlock(2, 1, &row);
         lower.SetBlock(2, 2, &gauge_matrix);
      }
      lower.Mult(exact, rhs);
      for (int repeat = 0; repeat < 2; ++repeat)
      {
         result = 19.0;
         lower_inverse.Mult(rhs, result);
         result -= exact;
         tests.Require(result.Norml2() < 1e-13,
                       "Lower forward substitution, coupling/gauge signs and repeatability");
      }
      rhs *= -2.5;
      lower_inverse.Mult(rhs, result);
      result.Add(2.5, exact);
      tests.Require(result.Norml2() < 1e-13, "Lower action is fixed and linear");
   }

   for (const std::string element : {"tri", "quad", "tet", "hex"})
   {
      Mesh serial = MakeStokesMesh(element, 2);
      CurveTestMesh(serial, geometry_order);
      ParMesh mesh(MPI_COMM_WORLD, serial);
      const int dimension = mesh.Dimension();
      for (int order : {1, 3})
      {
         H1_FECollection collection(order, dimension);
         ParFiniteElementSpace space(&mesh, &collection);
         ConstantCoefficient one(1.0);
         ParBilinearForm form(&space);
         form.AddDomainIntegrator(new MassIntegrator(one));
         form.AddDomainIntegrator(new DiffusionIntegrator(one));
         form.Assemble();
         form.Finalize();
         std::unique_ptr<HypreParMatrix> matrix(form.ParallelAssemble());
         FunctionCoefficient field([](const Vector &point) { return 1.0 + point.Sum(); });
         ParGridFunction projected(&space);
         projected.ProjectCoefficient(field);
         Vector exact, rhs(space.GetTrueVSize()), result(rhs.Size()), action(rhs.Size());
         projected.GetTrueDofs(exact);
         matrix->Mult(exact, rhs);
         const real_t initial_energy = InnerProduct(MPI_COMM_WORLD, exact, rhs);
         for (int steps : {1, 3})
         {
            tests.context = "P1-AMG " + element + " order=" + std::to_string(order) +
                            " sweeps=" + std::to_string(steps);
            detail::P1AMGPreconditioner inverse(*matrix, space, {}, steps, false, false);
            result = 12.0;
            inverse.Mult(rhs, result);
            Vector first(result), scaled_rhs(rhs), repeated(rhs.Size());
            repeated = -23.0;
            inverse.Mult(rhs, repeated);
            repeated -= first;
            tests.Require(Norm(repeated) < 1e-13 * (1.0 + Norm(first)), "Repeatable zero-start action");
            scaled_rhs *= -2.5;
            inverse.Mult(scaled_rhs, repeated);
            repeated.Add(2.5, first);
            tests.Require(Norm(repeated) < 1e-12 * (1.0 + Norm(first)), "Fixed linear action");
            result -= exact;
            matrix->Mult(result, action);
            const real_t final_energy = InnerProduct(MPI_COMM_WORLD, result, action);
            tests.Require(final_energy < initial_energy, "Two-level correction reduces energy error");
            HypreSmoother smoother(*matrix, HypreSmoother::l1Jacobi, 2 * steps);
            smoother.Mult(rhs, result);
            result -= exact;
            matrix->Mult(result, action);
            const real_t smoothed_energy = InnerProduct(MPI_COMM_WORLD, result, action);
            tests.Require(smoothed_energy < initial_energy, "l1-Jacobi is convergent");
            tests.Require(final_energy < smoothed_energy, "P1 coarse correction improves smooth error");
         }
      }
   }
}

void P1AssemblyTests(Tests &tests, StokesStabilizationMethod method)
{
   for (const std::string element : {"tri", "quad", "tet", "hex"})
   {
      for (int geometry_order : {1, 2, 3})
      {
         Mesh serial = MakeStokesMesh(element, 2);
         CurveTestMesh(serial, geometry_order);
         ParMesh mesh(MPI_COMM_WORLD, serial);
         const int dimension = mesh.Dimension();
         const int quadrature = detail::StokesIntegrationOrder(2, geometry_order, dimension);
         ManufacturedSolution exact(dimension, 1.0, true, 0.2);
         FunctionCoefficient inverse_mu([&](const Vector &point) { return 1.0 / exact.Viscosity(point); });
         L2_FECollection cell_collection(0, dimension);
         ParFiniteElementSpace cell_space(&mesh, &cell_collection);
         ParGridFunction cell_sizes(&cell_space), cell_viscosities(&cell_space), cell_tau(&cell_space);
         cell_viscosities.ProjectCoefficient(exact.viscosity);
         for (int index = 0; index < mesh.GetNE(); ++index)
         {
            Array<int> dofs;
            cell_space.GetElementDofs(index, dofs);
            cell_sizes(dofs[0]) = mesh.GetElementSize(index, 1);
            cell_tau(dofs[0]) = std::pow(cell_sizes(dofs[0]), 2) / (81.0 * cell_viscosities(dofs[0]));
         }
         cell_sizes.ExchangeFaceNbrData();
         cell_viscosities.ExchangeFaceNbrData();
         GridFunctionCoefficient sizes(&cell_sizes), viscosities(&cell_viscosities), tau(&cell_tau);
         for (bool velocity_block : {false, true})
         {
            tests.context = std::string("P1 assembly ") + StokesStabilizationMethodName(method) +
                            " " + element + " geometry=" + std::to_string(geometry_order) +
                            (velocity_block ? " velocity" : " pressure");
            const int order = !velocity_block && method == StokesStabilizationMethod::TaylorHood ? 1 : 2;
            H1_FECollection collection(order, dimension);
            ParFiniteElementSpace space(&mesh, &collection, velocity_block ? dimension : 1, Ordering::byVDIM);
            auto assemble = [&](ParFiniteElementSpace &target)
            {
               ParBilinearForm form(&target);
               if (velocity_block)
               {
                  form.AddDomainIntegrator(new RuleBilinearIntegrator<ElasticityIntegrator>(
                                              quadrature, exact.viscosity, 0.0, 1.0));
                  if (method == StokesStabilizationMethod::GLS)
                  {
                     form.AddDomainIntegrator(new GLSVelocityIntegrator(
                                                 exact.viscosity, exact.viscosity_gradient, tau, quadrature));
                  }
               }
               else
               {
                  form.AddDomainIntegrator(new RuleBilinearIntegrator<MassIntegrator>(quadrature, inverse_mu));
                  if (method == StokesStabilizationMethod::PSPG || method == StokesStabilizationMethod::GLS)
                  {
                     form.AddDomainIntegrator(new RuleBilinearIntegrator<DiffusionIntegrator>(quadrature, tau));
                  }
                  else if (method == StokesStabilizationMethod::CIP)
                  {
                     form.AddInteriorFaceIntegrator(new PressureCIPIntegrator(sizes, viscosities, 0.01, quadrature));
                  }
               }
               form.Assemble(0);
               form.Finalize(0);
               return std::unique_ptr<HypreParMatrix>(form.ParallelAssemble());
            };
            auto matrix = assemble(space);
            std::vector<Array<int>> markers;
            Array<int> essential, selected;
            if (velocity_block)
            {
               markers.resize(dimension);
               for (int component = 0; component < dimension; ++component)
               {
                  markers[component].SetSize(mesh.bdr_attributes.Max());
                  markers[component] = 0;
                  markers[component][component] = 1;
                  space.GetEssentialTrueDofs(markers[component], selected, component);
                  essential.Append(selected);
               }
               essential.Sort();
               delete matrix->EliminateRowsCols(essential);
            }
            detail::P1AMGPreconditioner galerkin(*matrix, space, markers, 2, velocity_block, true);
            int assemblies = 0;
            auto assemble_coarse = [&](ParFiniteElementSpace &coarse)
            {
               ++assemblies;
               tests.Require(coarse.GetMaxElementOrder() == 1 && coarse.GetParMesh() == &mesh &&
                             coarse.GetVDim() == space.GetVDim(), "Assembly uses same-mesh P1/Q1 space");
               return assemble(coarse);
            };
            detail::P1AMGPreconditioner assembled(*matrix, space, markers, 2, velocity_block, true, assemble_coarse);
            const auto &coarse_matrix = assembled.GetCoarseMatrix();
            std::unique_ptr<HypreParMatrix> difference(Add(1.0, coarse_matrix, -1.0, galerkin.GetCoarseMatrix()));
            tests.Require(difference->FNorm() < 2e-11 * (1.0 + coarse_matrix.FNorm()),
                          "Constrained P1 assembly agrees with Galerkin at matched quadrature/weights");
            Vector rhs(space.GetTrueVSize()), result(rhs.Size()), repeated(rhs.Size());
            for (int index = 0; index < rhs.Size(); ++index) { rhs(index) = 1.0 + std::sin(index + 0.3); }
            result = 12.0;
            assembled.Mult(rhs, result);
            repeated = -23.0;
            assembled.Mult(rhs, repeated);
            repeated -= result;
            tests.Require(Norm(repeated) < 1e-13 * (1.0 + Norm(result)), "Assembled inverse is repeatable");
            rhs *= -2.5;
            assembled.Mult(rhs, repeated);
            repeated.Add(2.5, result);
            tests.Require(Norm(repeated) < 1e-12 * (1.0 + Norm(result)), "Assembled inverse is fixed and linear");
            tests.Require(assemblies == 1, "Coarse assembly occurs only once, never during Mult");
         }
      }
   }
}

void MeshMMSFieldTests(Tests &tests)
{
   for (const std::string family : {"tri", "quad", "tet", "hex"})
   {
      Mesh reference = MakeStokesMesh(family, 2);
      reference.Transform([](const Vector &point, Vector &value)
      {
         value = point;
         value(0) += 0.13 * point(0) * point(1);
         if (point.Size() == 3) { value(1) += 0.09 * point(0) * point(2); }
      });
      Mesh converted(reference);
      converted.SetCurvature(1, true);
      tests.context = "mesh MMS discontinuous coordinates " + family;
      tests.Require(converted.GetNodalFESpace()->IsDGSpace(), "Coordinates remain discontinuous");
      H1_FECollection collection(3, reference.Dimension());
      ConstantCoefficient mu(1.7);
      Vector zero(reference.Dimension());
      zero = 0.0;
      VectorConstantCoefficient gradient_mu(zero);
      for (int element = 0; element < reference.GetNE(); ++element)
      {
         auto &original = *reference.GetElementTransformation(element);
         auto &replacement = *converted.GetElementTransformation(element);
         const auto geometry = reference.GetElementGeometry(element);
         const auto &rule = IntRules.Get(geometry, 2);
         for (int index = 0; index < rule.GetNPoints(); ++index)
         {
            const auto &point = rule.IntPoint(index);
            original.SetIntPoint(&point);
            replacement.SetIntPoint(&point);
            Vector original_point, replacement_point;
            original.Transform(point, original_point);
            replacement.Transform(point, replacement_point);
            replacement_point -= original_point;
            tests.Near(replacement_point.Norml2(), 0.0, 1e-13, "Coordinate map preserved");
            DenseMatrix jacobian(replacement.Jacobian());
            jacobian -= original.Jacobian();
            tests.Near(jacobian.FNorm(), 0.0, 1e-12, "Jacobian preserved");
            DenseMatrix original_basis, replacement_basis;
            const auto &finite_element = *collection.FiniteElementForGeometry(geometry);
            const int dimension = reference.Dimension();
            const int dofs = finite_element.GetDof();
            DenseMatrix hessian(dofs, dimension * (dimension + 1) / 2);
            finite_element.CalcPhysHessian(original, hessian);
            original_basis.SetSize(dimension, dimension * dofs);
            for (int node = 0; node < dofs; ++node)
            {
               real_t laplacian = 0.0;
               for (int axis = 0; axis < dimension; ++axis)
               {
                  laplacian += hessian(node, HessianIndex(dimension, axis, axis));
               }
               for (int component = 0; component < dimension; ++component)
               {
                  for (int axis = 0; axis < dimension; ++axis)
                  {
                     original_basis(axis, component * dofs + node) = -1.7 *
                        (hessian(node, HessianIndex(dimension, axis, component)) +
                         (axis == component ? laplacian : 0.0));
                  }
               }
            }
            EvaluateMomentumBasis(finite_element, replacement, mu, gradient_mu, replacement_basis);
            replacement_basis -= original_basis;
            tests.Near(replacement_basis.FNorm(), 0.0, 1e-9, "Physical momentum Hessian preserved");
         }
      }
   }
   for (const auto &configuration : {std::make_pair(2, MeshMMSCase::NoSlip),
                                     std::make_pair(3, MeshMMSCase::NoSlip),
                                     std::make_pair(2, MeshMMSCase::Smooth),
                                     std::make_pair(3, MeshMMSCase::Smooth)})
   {
      const int dimension = configuration.first;
      const auto problem_case = configuration.second;
      for (bool variable_mu : {false, true})
      {
         MeshManufacturedSolution exact(dimension, 1.7, variable_mu, problem_case);
         tests.context = "mesh MMS dim=" + std::to_string(dimension) + " variable_mu=" + std::to_string(variable_mu) +
                         " case=" + MeshMMSCaseName(problem_case);
         for (const std::array<real_t, 3> coordinates :
              {std::array<real_t, 3>{0.31, 0.44, 0.54}, {0.5, 0.25, 0.25},
               {0.64, 0.25, 0.25}, {0.60, 0.32, 0.29}, {0.68, 0.25, 0.25}})
         {
            Vector point(dimension), value(dimension), force(dimension), gradient_mu(dimension);
            for (int axis = 0; axis < dimension; ++axis) { point(axis) = coordinates[axis]; }
            exact.Velocity(point, value);
            exact.Force(point, force);
            exact.ViscosityGradient(point, gradient_mu);
            DenseMatrix gradient;
            exact.Gradient(point, gradient);
            real_t divergence = 0.0;
            for (int axis = 0; axis < dimension; ++axis) { divergence += gradient(axis, axis); }
            tests.Near(divergence, 0.0, 1e-10, "Analytic velocity is divergence-free");
            Vector difference_force(dimension);
            difference_force = 0.0;
            const real_t step = 1e-6;
            for (int axis = 0; axis < dimension; ++axis)
            {
               Vector plus(point), minus(point), plus_value(dimension), minus_value(dimension);
               plus(axis) += step;
               minus(axis) -= step;
               exact.Velocity(plus, plus_value);
               exact.Velocity(minus, minus_value);
               for (int component = 0; component < dimension; ++component)
               {
                  tests.Near((plus_value(component) - minus_value(component)) / (2 * step),
                             gradient(component, axis), 2e-6, "Analytic velocity gradient");
               }
               tests.Near((exact.Viscosity(plus) - exact.Viscosity(minus)) / (2 * step),
                          gradient_mu(axis), 1e-7, "Analytic periodic viscosity gradient");
               DenseMatrix plus_gradient, minus_gradient;
               exact.Gradient(plus, plus_gradient);
               exact.Gradient(minus, minus_gradient);
               for (int component = 0; component < dimension; ++component)
               {
                  difference_force(component) -=
                     (exact.Viscosity(plus) * (plus_gradient(component, axis) + plus_gradient(axis, component)) -
                      exact.Viscosity(minus) * (minus_gradient(component, axis) + minus_gradient(axis, component))) / (2 * step);
               }
               difference_force(axis) += (exact.Pressure(plus) - exact.Pressure(minus)) / (2 * step);
               Vector translated(point), translated_velocity(dimension), translated_force(dimension);
               translated(axis) += axis == 0 ? 2.0 : 1.0;
               exact.Velocity(translated, translated_velocity);
               exact.Force(translated, translated_force);
               translated_velocity -= value;
               translated_force -= force;
               tests.Require(translated_velocity.Norml2() < 1e-10 * (1 + value.Norml2()), "Periodic velocity");
               tests.Require(translated_force.Norml2() < 1e-9 * (1 + force.Norml2()), "Periodic force");
               tests.Near(exact.Pressure(translated), exact.Pressure(point), 1e-12, "Periodic pressure");
               tests.Near(exact.Viscosity(translated), exact.Viscosity(point), 1e-12, "Periodic viscosity");
            }
            difference_force -= force;
            tests.Require(difference_force.Norml2() < 2e-5 * (1 + force.Norml2()), "Force equals -div(2*mu*epsilon(u))+grad(p)");
         }
         Vector point(dimension), value(dimension);
         point = 0.37;
         for (int axis = 0; axis < dimension; ++axis)
         {
            const real_t saved = point(axis);
            for (real_t side : {real_t(0.0), axis == 0 ? real_t(2.0) : real_t(1.0)})
            {
               point(axis) = side;
               exact.Velocity(point, value);
               tests.Require(value.Norml2() < 1e-12, "Exact no-slip outer boundary");
            }
            point(axis) = saved;
         }
         point(0) = 0.59;
         point(1) = 0.25;
         if (dimension == 3) { point(2) = 0.25; }
         exact.Velocity(point, value);
         if (problem_case == MeshMMSCase::NoSlip)
         {
            tests.Require(value.Norml2() == 0.0, "Exact no-slip hole collar");
         }
         else
         {
            tests.Require(value.Norml2() > 0.1, "Smooth MMS has nonzero hole velocity");
            DenseMatrix gradient;
            exact.Gradient(point, gradient);
            tests.Require(gradient.FNorm() > 0.1, "Smooth MMS exercises nonconstant velocity gradients near holes");
         }
      }
   }
}

void PressureGaugeTests(Tests &tests, StokesStabilizationMethod method)
{
   Mesh serial = MakeStokesMesh("quad", 3);
   ParMesh mesh(MPI_COMM_WORLD, serial);
   H1_FECollection velocity_collection(2, 2);
   H1_FECollection pressure_collection(method == StokesStabilizationMethod::TaylorHood ? 1 : 2, 2);
   ParFiniteElementSpace velocity_space(&mesh, &velocity_collection, 2, Ordering::byVDIM);
   ParFiniteElementSpace pressure_space(&mesh, &pressure_collection);
   Vector zero(2);
   zero = 0.0;
   VectorConstantCoefficient zero_velocity(zero);
   std::vector<LinearSolverType> backends = {LinearSolverType::GMRES, LinearSolverType::FGMRES};
#ifdef MFEM_USE_MUMPS
   backends.push_back(LinearSolverType::MUMPS);
#endif
   for (auto backend : backends)
   {
      for (auto policy : {StokesPressureGauge::Auto, StokesPressureGauge::MeanZero,
                          StokesPressureGauge::None})
      {
         StokesSolverOptions options;
         options.method = method;
         options.linear_solver = backend;
         options.pressure_gauge = policy;
         options.print_level = 0;
         options.relative_tolerance = 1e-13;
         options.absolute_tolerance = 1e-15;
         options.restart_dimension = 200;
         options.max_iterations = 2000;
         StokesSolver solver(velocity_space, pressure_space, 1.0, options);
         solver.SetVolForce(1.0, 0.0);
         for (int boundary_mode : {0, 1, 2})
         {
            const bool open_boundary = boundary_mode == 1;
            if ((policy == StokesPressureGauge::MeanZero && open_boundary) ||
                (policy == StokesPressureGauge::None && !open_boundary)) { continue; }
            tests.context = std::string(StokesStabilizationMethodName(method)) +
                            " gauge=" + StokesPressureGaugeName(policy) +
                            " backend=" + std::to_string(static_cast<int>(backend)) +
                            " boundary=" + std::to_string(boundary_mode);
            solver.DelVelocityBC();
            for (int index = 0; index < mesh.bdr_attributes.Size(); ++index)
            {
               const int attribute = mesh.bdr_attributes[index];
               if (open_boundary && attribute == 2) { continue; }
               const int direction = boundary_mode == 2 ? (attribute % 2 == 0 ? 0 : 1) : -1;
               solver.AddVelocityBC(attribute, direction, 0.0);
            }
            const auto info = solver.Solve();
            tests.Require(info.converged, "Pressure-gauge policy converges");
            tests.Require(info.pressure_gauge == policy, "Requested policy is reported");
            tests.Require(info.pressure_nullspace_detected == !open_boundary,
                          "Nullspace detection follows boundary conditions after reassembly");
            tests.Require(info.pressure_gauge_active == !open_boundary,
                          "Mean constraint follows the detected nullspace");
            tests.Near(info.pressure_mean, open_boundary ? -0.5 : 0.0, 1e-8,
                       "Natural-boundary pressure remains absolute; closed/slip pressure has zero mean");
            tests.Near(info.gauge_multiplier, 0.0, 1e-8, "Gauge does not repair continuity");
            tests.Near(info.boundary_flux, 0.0, 1e-8, "Compatible boundary flux");
            FunctionCoefficient exact_pressure([open_boundary](const Vector &point)
            {
               return point(0) - (open_boundary ? 1.0 : 0.5);
            });
            tests.Require(solver.GetPressure().ComputeL2Error(exact_pressure) < 1e-8,
                          "Explicit and automatic policies reproduce the same exact pressure");
            tests.Require(solver.GetVelocity().ComputeL2Error(zero_velocity) < 1e-8,
                          "Pressure-gauge policy preserves the exact velocity");
         }
      }
   }
   tests.Require(std::string(StokesPressureGaugeName(static_cast<StokesPressureGauge>(-1))) == "unknown",
                 "Invalid pressure-gauge enum has no CLI spelling");
}

void InvalidPressureGaugeTest(const std::string &name)
{
   StokesSolverOptions options;
   for (auto method : {StokesStabilizationMethod::PSPG, StokesStabilizationMethod::GLS,
                       StokesStabilizationMethod::CIP, StokesStabilizationMethod::TaylorHood})
   {
      if (name.find(StokesStabilizationMethodName(method)) != std::string::npos)
      {
         options.method = method;
      }
   }
   options.pressure_gauge = name.find("none") != std::string::npos ?
                            StokesPressureGauge::None : StokesPressureGauge::MeanZero;
   if (name == "gauge_enum") { options.pressure_gauge = static_cast<StokesPressureGauge>(-1); }
   Mesh serial = MakeStokesMesh("quad", 3);
   ParMesh mesh(MPI_COMM_WORLD, serial);
   H1_FECollection velocity_collection(2, 2);
   H1_FECollection pressure_collection(options.method == StokesStabilizationMethod::TaylorHood ? 1 : 2, 2);
   ParFiniteElementSpace velocity_space(&mesh, &velocity_collection, 2, Ordering::byVDIM);
   ParFiniteElementSpace pressure_space(&mesh, &pressure_collection);
   StokesSolver solver(velocity_space, pressure_space, 1.0, options);
   VectorFunctionCoefficient expansion(2, [](const Vector &point, Vector &value)
   {
      value = point;
      value *= 0.01;
   });
   const bool incompatible_flux = name.find("flux") != std::string::npos;
   const bool reassembly = name.find("reassembly") != std::string::npos;
   const bool target_open = !incompatible_flux && options.pressure_gauge == StokesPressureGauge::MeanZero;
   for (int assembly = reassembly ? 0 : 1; assembly < 2; ++assembly)
   {
      solver.DelVelocityBC();
      const bool open_boundary = assembly == 0 ? !target_open : target_open;
      for (int index = 0; index < mesh.bdr_attributes.Size(); ++index)
      {
         const int attribute = mesh.bdr_attributes[index];
         if (open_boundary && attribute == 2) { continue; }
         if (incompatible_flux) { solver.AddVelocityBC(attribute, expansion); }
         else { solver.AddVelocityBC(attribute, -1, 0.0); }
      }
      solver.SetVolForce(1e12, 0.0);
      solver.Assemble();
   }
   MFEM_ABORT("Expected pressure-gauge failure was not rejected");
}

void RobustnessTests(Tests &tests)
{
   for (auto method : {StokesStabilizationMethod::PSPG, StokesStabilizationMethod::GLS,
                       StokesStabilizationMethod::CIP, StokesStabilizationMethod::TaylorHood})
   {
      Mesh serial = MakeStokesMesh("quad", 3);
      ParMesh mesh(MPI_COMM_WORLD, serial);
      StokesSolverOptions options;
      options.method = method;
      options.linear_solver = ReferenceSolver();
      options.print_level = 0;
      const bool residual_method = method == StokesStabilizationMethod::PSPG ||
                                   method == StokesStabilizationMethod::GLS;
      for (int velocity_basis : {BasisType::GaussLobatto, BasisType::Positive})
      {
         if (residual_method && velocity_basis == BasisType::Positive) { continue; }
         H1_FECollection velocity_collection(2, 2, velocity_basis);
         H1_FECollection pressure_collection(method == StokesStabilizationMethod::TaylorHood ? 1 : 2,
                                             2, BasisType::Positive);
         ParFiniteElementSpace velocity_space(&mesh, &velocity_collection, 2, Ordering::byVDIM);
         ParFiniteElementSpace pressure_space(&mesh, &pressure_collection);
         tests.context = std::string(StokesStabilizationMethodName(method)) +
                         " velocity_basis=" + std::to_string(velocity_basis);
         for (bool open_boundary : {false, true})
         {
            for (real_t force_scale : {real_t(1.0), real_t(1e12)})
            {
               StokesSolver solver(velocity_space, pressure_space, 1.0, options);
               for (int index = 0; index < mesh.bdr_attributes.Size(); ++index)
               {
                  const int attribute = mesh.bdr_attributes[index];
                  if (open_boundary && attribute == 2) { continue; }
                  solver.AddVelocityBC(attribute, -1, 0.0);
               }
               solver.SetVolForce(force_scale, 0.0);
               const auto info = solver.Solve();
               tests.Require(info.converged, "Compatible flux at small/large forcing converges");
               tests.Require(info.pressure_gauge_active == !open_boundary,
                             "Pressure gauge is independent of force scale and basis");
               tests.Require(solver.GetVelocity().CheckFinite() == 0 &&
                             solver.GetPressure().CheckFinite() == 0 &&
                             std::isfinite(info.absolute_residual), "Finite solution and residual");
               if (!open_boundary)
               {
                  tests.Near(info.boundary_flux, 0.0, 1e-8, "Closed boundary retains zero flux");
               }
               FunctionCoefficient pressure([force_scale, open_boundary](const Vector &point)
               {
                  return force_scale * (point(0) - (open_boundary ? 1.0 : 0.5));
               });
               tests.Require(solver.GetPressure().ComputeL2Error(pressure) < 1e-8 * force_scale,
                             "Positive pressure basis reproduces linear pressure");
            }
         }
      }
   }
}

void FluxScalingTests(Tests &tests)
{
   for (int dimension : {2, 3})
   {
      for (real_t scale : {real_t(1.0), real_t(1e-3), real_t(1e-6)})
      {
         Mesh serial = MakeStokesMesh(dimension == 2 ? "quad" : "hex", 3);
         serial.Transform([scale](const Vector &point, Vector &value) { value = point; value *= scale; });
         ParMesh mesh(MPI_COMM_WORLD, serial);
         for (auto method : {StokesStabilizationMethod::PSPG, StokesStabilizationMethod::GLS,
                             StokesStabilizationMethod::CIP, StokesStabilizationMethod::TaylorHood})
         {
            H1_FECollection velocity_collection(2, dimension);
            H1_FECollection pressure_collection(method == StokesStabilizationMethod::TaylorHood ? 1 : 2,
                                                dimension);
            ParFiniteElementSpace velocity_space(&mesh, &velocity_collection, dimension, Ordering::byVDIM);
            ParFiniteElementSpace pressure_space(&mesh, &pressure_collection);
            StokesSolverOptions options;
            options.method = method;
            options.linear_solver = ReferenceSolver();
            options.relative_tolerance = 1e-13;
            options.absolute_tolerance = 1e-15 * std::pow(scale, dimension);
            options.max_iterations = 3000;
            options.restart_dimension = 200;
            options.print_level = 0;
            for (int mode = 0; mode < 3; ++mode)
            {
               tests.context = std::string(StokesStabilizationMethodName(method)) + " flux scaling dim=" +
                               std::to_string(dimension) + " scale=" + std::to_string(scale) +
                               " mode=" + std::to_string(mode);
               VectorFunctionCoefficient exact_velocity(dimension, [=](const Vector &point, Vector &value)
               {
                  value.SetSize(dimension);
                  value = 0.0;
                  if (mode == 1)
                  {
                     value = point;
                     value(dimension - 1) *= -(dimension - 1);
                  }
                  if (mode == 2)
                  {
                     const real_t horizontal = point(0) / scale, vertical = point(1) / scale;
                     value(0) = scale * horizontal * (1 - horizontal) * (1 - 2 * vertical);
                     value(1) = -scale * (1 - 2 * horizontal) * vertical * (1 - vertical);
                  }
               });
               VectorFunctionCoefficient force(dimension, [=](const Vector &point, Vector &value)
               {
                  value.SetSize(dimension);
                  value = 0.0;
                  if (mode == 2)
                  {
                     value(0) = 2 * (1 - 2 * point(1) / scale) / scale;
                     value(1) = -2 * (1 - 2 * point(0) / scale) / scale;
                  }
               });
               StokesSolver solver(velocity_space, pressure_space, 1.0, options);
               for (int attribute : mesh.bdr_attributes) { solver.AddVelocityBC(attribute, exact_velocity); }
               solver.SetVolForce(force);
               const auto info = solver.Solve();
               tests.Require(info.converged && info.pressure_gauge_active, "Compatible scaled boundary converges");
               tests.Require(solver.GetVelocity().ComputeL2Error(exact_velocity) /
                             std::pow(scale, 1.0 + 0.5 * dimension) < 1e-7,
                             "Scale-independent velocity accuracy");
               tests.Near(info.boundary_flux / std::pow(scale, dimension), 0.0, 1e-9,
                          "Scale-independent boundary flux");
               tests.Near(info.gauge_multiplier, 0.0, 1e-8, "Gauge does not repair scaled continuity");
            }
         }
      }
   }
}

void BernsteinBoundaryTests(Tests &tests)
{
   for (const std::string element : {"tri", "quad", "tet", "hex"})
   {
      Mesh serial = MakeStokesMesh(element, element == "quad" || element == "hex" ? 3 : 2);
      ParMesh mesh(MPI_COMM_WORLD, serial);
      const int dimension = mesh.Dimension();
      for (int degree : {2, 3})
      {
         for (auto method : {StokesStabilizationMethod::CIP, StokesStabilizationMethod::TaylorHood})
         {
            H1_FECollection velocity_collection(degree, dimension, BasisType::Positive);
            H1_FECollection pressure_collection(method == StokesStabilizationMethod::TaylorHood ? degree - 1 : degree,
                                                dimension, BasisType::Positive);
            ParFiniteElementSpace velocity_space(&mesh, &velocity_collection, dimension, Ordering::byVDIM);
            ParFiniteElementSpace pressure_space(&mesh, &pressure_collection);
            VectorFunctionCoefficient exact_velocity(dimension, [dimension](const Vector &point, Vector &value)
            {
               value.SetSize(dimension);
               value = 0.0;
               value(0) = point(1) * point(1);
            });
            FunctionCoefficient horizontal_velocity([](const Vector &point) { return point(1) * point(1); });
            ConstantCoefficient zero(0.0);
            StokesSolverOptions options;
            options.method = method;
            options.linear_solver = ReferenceSolver();
            options.relative_tolerance = 1e-13;
            options.absolute_tolerance = 1e-15;
            options.max_iterations = 3000;
            options.restart_dimension = 200;
            options.print_level = 0;
            StokesSolver solver(velocity_space, pressure_space, 1.0, options);
            solver.SetVolForce(-2.0, 0.0);
            for (int mode = 0; mode < 2; ++mode)
            {
               tests.context = std::string(StokesStabilizationMethodName(method)) + " Bernstein boundary " +
                               element + " degree=" + std::to_string(degree) + " mode=" + std::to_string(mode);
               solver.DelVelocityBC();
               for (int attribute : mesh.bdr_attributes)
               {
                  solver.AddVelocityBC(attribute, -1, 17.0);
                  solver.AddVelocityBC(attribute, exact_velocity);
                  if (mode == 1)
                  {
                     solver.AddVelocityBC(attribute, 0, 17.0);
                     solver.AddVelocityBC(attribute, 0, horizontal_velocity);
                     for (int component = 1; component < dimension; ++component)
                     {
                        solver.AddVelocityBC(attribute, component, zero);
                     }
                  }
               }
               const auto info = solver.Solve();
               tests.Require(info.converged && info.pressure_gauge_active, "Bernstein boundary solve converges");
               tests.Require(solver.GetVelocity().ComputeL2Error(exact_velocity) < 1e-8,
                             "Bernstein boundary reproduces quadratic velocity");
               tests.Require(solver.GetPressure().ComputeL2Error(zero) < 1e-8,
                             "Bernstein boundary preserves exact pressure");
               tests.Near(info.gauge_multiplier, 0.0, 1e-9, "Bernstein boundary has compatible flux");
            }
         }
      }
   }
}

void InvalidRobustnessTest(const std::string &name)
{
   StokesSolverOptions options;
   if (name.find("gls") != std::string::npos) { options.method = StokesStabilizationMethod::GLS; }
   if (name.find("cip") != std::string::npos) { options.method = StokesStabilizationMethod::CIP; }
   if (name.find("taylor_hood") != std::string::npos) { options.method = StokesStabilizationMethod::TaylorHood; }
   const int basis = name.find("positive") != std::string::npos ? BasisType::Positive :
                     name.find("serendipity") != std::string::npos ? BasisType::Serendipity : BasisType::GaussLobatto;
   Mesh serial = MakeStokesMesh("quad", 3);
   const auto scale_position = name.find("flux_scale");
   if (scale_position != std::string::npos)
   {
      const real_t scale = std::pow(10.0, -std::stoi(name.substr(scale_position + 10)));
      serial.Transform([scale](const Vector &point, Vector &value) { value = point; value *= scale; });
   }
   ParMesh mesh(MPI_COMM_WORLD, serial);
   H1_FECollection velocity_collection(2, 2, basis);
   H1_FECollection pressure_collection(options.method == StokesStabilizationMethod::TaylorHood ? 1 : 2, 2);
   ParFiniteElementSpace velocity_space(&mesh, &velocity_collection, 2, Ordering::byVDIM);
   ParFiniteElementSpace pressure_space(&mesh, &pressure_collection);
   const bool overflow = name == "robustness_overflow";
   const bool incompatible_flux = name.find("flux") != std::string::npos;
   const bool boundary_conflict = name.find("boundary_conflict") != std::string::npos;
   const bool boundary_nonfinite = name.find("boundary_nonfinite") != std::string::npos;
   FunctionCoefficient invalid_boundary([boundary_nonfinite](const Vector &point)
   {
      return boundary_nonfinite ? std::numeric_limits<real_t>::quiet_NaN() : 1.0 + point.Sum() * point.Sum();
   });
   StokesSolver solver(velocity_space, pressure_space, overflow ? 1e-8 : 1.0, options);
   VectorFunctionCoefficient expansion(2, [](const Vector &point, Vector &value)
   {
      value = point;
      value *= 0.01;
   });
   for (int index = 0; index < mesh.bdr_attributes.Size(); ++index)
   {
      const int attribute = mesh.bdr_attributes[index];
      if (overflow && attribute == 2) { continue; }
      if (incompatible_flux) { solver.AddVelocityBC(attribute, expansion); }
      else { solver.AddVelocityBC(attribute, -1, 0.0); }
      if (boundary_nonfinite || (boundary_conflict && attribute == 1))
      {
         solver.AddVelocityBC(attribute, 0, invalid_boundary);
      }
   }
   solver.SetVolForce(overflow ? (Mpi::Root() ? 1e308 : 1.0) : 1e12, 0.0);
   solver.Assemble();
   MFEM_ABORT("Expected robustness failure was not rejected");
}

void InvalidTest(const std::string &name)
{
   if (name.find("gauge_") == 0) { InvalidPressureGaugeTest(name); }
   if (name.find("robustness_") == 0) { InvalidRobustnessTest(name); }
   Mesh serial = MakeStokesMesh("quad", 2);
   if (name == "folded_geometry")
   {
      serial.SetCurvature(2);
      serial.Transform([](const Vector &point, Vector &value)
      {
         value = point;
         value(0) = point(0) * (1.0 - 2.0 * point(0));
      });
   }
   if (name == "coordinate_family")
   {
      auto *collection = new QuadraticFECollection;
      serial.SetNodalFESpace(new FiniteElementSpace(&serial, collection, 2));
      serial.GetNodes()->MakeOwner(collection);
   }
   ParMesh mesh(MPI_COMM_WORLD, serial);
   H1_FECollection collection(2, 2);
   ParFiniteElementSpace velocity_space(&mesh, &collection, 2,
                                       name == "ordering" ? Ordering::byNODES : Ordering::byVDIM);
   ParFiniteElementSpace pressure_space(&mesh, &collection);
   StokesSolverOptions options;
   if (name == "method") { options.method = static_cast<StokesStabilizationMethod>(-1); }
   if (name == "preconditioner") { options.preconditioner = static_cast<StokesPreconditionerType>(-1); }
   if (name == "smoothing")
   {
      options.preconditioner = StokesPreconditionerType::BlockDiagonalP1AMG;
      options.p1_smoothing_steps = 0;
   }
   if (name == "pressure_order") { options.method = StokesStabilizationMethod::TaylorHood; }
   if (name == "cip_gamma") { options.method = StokesStabilizationMethod::CIP; options.cip_gamma = 0.0; }
   if (name == "gls_tau") { options.method = StokesStabilizationMethod::GLS; options.gls_tau_scale = -1.0; }
   if (name == "cip_mesh") { options.method = StokesStabilizationMethod::CIP; }
   StokesSolver solver(velocity_space, pressure_space, name == "viscosity" ? -1.0 : 1.0, options);
   if (name == "folded_geometry") { solver.Assemble(); }
   if (name == "direction") { solver.AddVelocityBC(1, 2, 0.0); }
   if (name == "attribute") { solver.AddVelocityBC(9999, 0, 0.0); }
   if (name == "dimension")
   {
      Vector value(3);
      value = 0.0;
      VectorConstantCoefficient coefficient(value);
      solver.AddVelocityBC(1, coefficient);
   }
   if (name == "conflict")
   {
      for (int index = 0; index < mesh.bdr_attributes.Size(); ++index)
      {
         const int attribute = mesh.bdr_attributes[index];
         solver.AddVelocityBC(attribute, -1, real_t(attribute));
      }
      solver.Assemble();
   }
   if (name == "flux")
   {
      VectorFunctionCoefficient expansion(2, [](const Vector &point, Vector &value) { value = point; });
      for (int index = 0; index < mesh.bdr_attributes.Size(); ++index)
      {
         solver.AddVelocityBC(mesh.bdr_attributes[index], expansion);
      }
      solver.Assemble();
   }
   if (name == "rigid") { solver.Assemble(); }
   if (name == "cip_mesh")
   {
      for (int index = 0; index < mesh.bdr_attributes.Size(); ++index)
      {
         solver.AddVelocityBC(mesh.bdr_attributes[index], -1, 0.0);
      }
      solver.Assemble();
   }
   MFEM_ABORT("Expected invalid input was not rejected");
}

}

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();
   Device device("cpu");
   const char *suite_name = "solvers";
   const char *method_name = "pspg";
   const char *invalid_name = "";
   const char *error_log_prefix = "";
   OptionsParser args(argc, argv);
   args.AddOption(&suite_name, "-s", "--suite", "operators, solvers, boundary, flux-scaling, bernstein-boundary, pressure-gauge, convergence, sensitivity, partition, reassembly, preconditioners, preconditioner-operators");
   args.AddOption(&method_name, "-method", "--method", "pspg, cip, gls, taylor-hood");
   args.AddOption(&invalid_name, "-i", "--invalid", "Run one expected-failure scenario.");
   args.AddOption(&error_log_prefix, "-el", "--error-log", "Per-rank diagnostic file prefix for expected failures.");
   args.ParseCheck();
   std::ofstream error_log;
   if (std::string(error_log_prefix).size())
   {
      MFEM_VERIFY(std::string(invalid_name).size(), "Diagnostic files require an expected-failure scenario");
      error_log.open(std::string(error_log_prefix) + "." + std::to_string(Mpi::WorldRank()));
      int local_open = error_log.is_open() ? 1 : 0, global_open = 0;
      MPI_Allreduce(&local_open, &global_open, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
      MFEM_VERIFY(global_open, "Unable to open expected-failure diagnostics");
      err.SetStream(error_log);
      err << std::unitbuf;
   }
   if (std::string(invalid_name).size()) { InvalidTest(invalid_name); }
   Tests tests;
   const std::string selected_method(method_name);
   MFEM_VERIFY(selected_method == "pspg" || selected_method == "cip" || selected_method == "gls" ||
               selected_method == "taylor-hood", "Unknown test method");
   const StokesStabilizationMethod method = selected_method == "pspg" ? StokesStabilizationMethod::PSPG :
                               selected_method == "cip" ? StokesStabilizationMethod::CIP :
                               selected_method == "gls" ? StokesStabilizationMethod::GLS : StokesStabilizationMethod::TaylorHood;
   const std::string suite(suite_name);
   if (suite == "operators") { OperatorTests(tests); }
   else if (suite == "robustness") { RobustnessTests(tests); }
   else if (suite == "flux-scaling") { FluxScalingTests(tests); }
   else if (suite == "bernstein-boundary") { BernsteinBoundaryTests(tests); }
   else if (suite == "timing") { TimingTests(tests); }
   else if (suite == "pressure-gauge") { PressureGaugeTests(tests, method); }
   else if (suite == "geometry") { GeometryTests(tests); PreconditionerOperatorTests(tests, 2); }
   else if (suite == "curved-solvers")
   {
      for (auto preconditioner : {StokesPreconditionerType::BlockDiagonalAMG,
                                 StokesPreconditionerType::BlockDiagonalP1AMG,
                                 StokesPreconditionerType::BlockUpperTriangularP1AMG,
                                 StokesPreconditionerType::BlockLowerTriangularP1AMG})
      {
         const int degree = preconditioner == StokesPreconditionerType::BlockDiagonalAMG ||
                            preconditioner == StokesPreconditionerType::BlockDiagonalP1AMG ? 2 : 3;
         SolverTests(tests, method, preconditioner, degree);
      }
   }
   else if (suite == "curved-boundary")
   {
      BoundaryTests(tests, ReferenceSolver(), method, StokesPreconditionerType::BlockDiagonalAMG, 2);
      BoundaryTests(tests, LinearSolverType::FGMRES, method, StokesPreconditionerType::BlockLowerTriangularP1AMG, 2);
   }
   else if (suite == "curved-convergence") { ConvergenceTests(tests, method, 2); }
   else if (suite == "curved-sensitivity") { SensitivityTests(tests, method, 2); }
   else if (suite == "curved-partition") { PartitionTests(tests, method, StokesPreconditionerType::BlockLowerTriangularP1AMG, 3); }
   else if (suite == "extension-operators") { ExtensionOperatorTests(tests); }
   else if (suite == "preconditioner-operators") { PreconditionerOperatorTests(tests); }
   else if (suite == "p1-assembly") { P1AssemblyTests(tests, method); }
   else if (suite == "mesh-mms-fields") { MeshMMSFieldTests(tests); }
   else if (suite == "solvers") { SolverTests(tests, method); PolynomialTests(tests, method); }
   else if (suite == "boundary")
   {
      BoundaryTests(tests, ReferenceSolver(), method);
      BoundaryTests(tests, LinearSolverType::FGMRES, method);
   }
   else if (suite == "convergence") { ConvergenceTests(tests, method); }
   else if (suite == "sensitivity") { SensitivityTests(tests, method); }
   else if (suite == "partition") { PartitionTests(tests, method); }
   else if (suite == "reassembly") { ReassemblyTests(tests, method); }
   else if (suite == "preconditioners")
   {
      for (auto preconditioner : {StokesPreconditionerType::BlockDiagonalP1AMG,
                                 StokesPreconditionerType::BlockUpperTriangularP1AMG,
                                 StokesPreconditionerType::BlockLowerTriangularP1AMG})
      {
         SolverTests(tests, method, preconditioner);
         BoundaryTests(tests, LinearSolverType::GMRES, method, preconditioner);
         BoundaryTests(tests, LinearSolverType::FGMRES, method, preconditioner);
         PartitionTests(tests, method, preconditioner);
         ReassemblyTests(tests, method, preconditioner);
      }
   }
   else if (suite == "assembled-preconditioners")
   {
      for (auto preconditioner : {StokesPreconditionerType::BlockDiagonalP1AMGAssembled,
                                 StokesPreconditionerType::BlockUpperTriangularP1AMGAssembled,
                                 StokesPreconditionerType::BlockLowerTriangularP1AMGAssembled})
      {
         SolverTests(tests, method, preconditioner);
         BoundaryTests(tests, LinearSolverType::GMRES, method, preconditioner);
         BoundaryTests(tests, LinearSolverType::FGMRES, method, preconditioner, 2, 1e-14);
         PartitionTests(tests, method, preconditioner, 3);
         ReassemblyTests(tests, method, preconditioner);
      }
   }
   else { tests.Require(false, "Unknown suite"); }
   return tests.Finish();
}
