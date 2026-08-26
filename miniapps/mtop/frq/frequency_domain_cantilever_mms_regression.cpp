// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC.
//
// Verify the frequency-domain elasticity solver with smooth manufactured
// solutions on two- and three-dimensional cantilever beams.

#include "frequency_domain_elasticity_solver.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

using namespace mfem;

namespace
{

/// Smooth complex displacement and its derived elasticity data.
struct ManufacturedSolution
{
   ManufacturedSolution(const int dimension, const real_t omega,
                        const bool moving_support)
      : dimension(dimension), omega(omega), real_amplitude(dimension),
        imaginary_amplitude(dimension), real_shift(dimension),
        imaginary_shift(dimension), real_wave_number(dimension),
        imaginary_wave_number(dimension)
   {
      const real_t pi = std::acos(-1.0);
      const real_t real_values[3] = {0.08, -0.11, 0.06};
      const real_t imaginary_values[3] = {-0.05, 0.09, -0.07};
      real_shift = 0.0;
      imaginary_shift = 0.0;
      for (int d = 0; d < dimension; ++d)
      {
         real_amplitude(d) = real_values[d];
         imaginary_amplitude(d) = imaginary_values[d];
      }
      if (moving_support)
      {
         real_shift(dimension - 1) = 0.03;
         imaginary_shift(dimension - 1) = -0.02;
      }

      real_wave_number(0) = pi/(2.0*length);
      imaginary_wave_number(0) = pi/length;
      real_wave_number(1) = pi/height;
      imaginary_wave_number(1) = pi/(2.0*height);
      if (dimension == 3)
      {
         real_wave_number(2) = pi/width;
         imaginary_wave_number(2) = pi/(2.0*width);
      }
   }

   void Displacement(const Vector &position, const bool imaginary,
                     Vector &value) const
   {
      const Vector &amplitude = imaginary ? imaginary_amplitude :
                                real_amplitude;
      const Vector &shift = imaginary ? imaginary_shift : real_shift;
      const Vector &wave_number = imaginary ? imaginary_wave_number :
                                  real_wave_number;
      const real_t mode = EvaluateMode(position, wave_number);
      value.SetSize(dimension);
      for (int i = 0; i < dimension; ++i)
      {
         value(i) = shift(i) + amplitude(i)*mode;
      }
   }

   void Gradient(const Vector &position, const bool imaginary,
                 DenseMatrix &value) const
   {
      const Vector &amplitude = imaginary ? imaginary_amplitude :
                                real_amplitude;
      const Vector &wave_number = imaginary ? imaginary_wave_number :
                                  real_wave_number;
      Vector gradient;
      EvaluateMode(position, wave_number, &gradient);
      value.SetSize(dimension);
      for (int i = 0; i < dimension; ++i)
      {
         for (int j = 0; j < dimension; ++j)
         {
            value(i, j) = amplitude(i)*gradient(j);
         }
      }
   }

   void BodyForce(const Vector &position, const bool imaginary,
                  Vector &value) const
   {
      // For A=K-omega^2 M+i omega(alpha M+beta K), form the real or
      // imaginary part of A(u_real+i u_imaginary) in strong form.
      Vector real_displacement;
      Vector imaginary_displacement;
      Vector real_stiffness;
      Vector imaginary_stiffness;
      Displacement(position, false, real_displacement);
      Displacement(position, true, imaginary_displacement);
      Stiffness(position, false, real_stiffness);
      Stiffness(position, true, imaginary_stiffness);

      value.SetSize(dimension);
      for (int i = 0; i < dimension; ++i)
      {
         if (imaginary)
         {
            value(i) = imaginary_stiffness(i)
                       - omega*omega*density*imaginary_displacement(i)
                       + omega*(damping_alpha*density*real_displacement(i)
                                 + damping_beta*real_stiffness(i));
         }
         else
         {
            value(i) = real_stiffness(i)
                       - omega*omega*density*real_displacement(i)
                       - omega*(damping_alpha*density*
                                imaginary_displacement(i)
                                + damping_beta*imaginary_stiffness(i));
         }
      }
   }

   void Traction(const Vector &position, const Vector &normal,
                 const bool imaginary, Vector &value) const
   {
      // The matching natural data are
      // (1+i omega beta) sigma(u_real+i u_imaginary) normal.
      DenseMatrix real_stress;
      DenseMatrix imaginary_stress;
      Stress(position, false, real_stress);
      Stress(position, true, imaginary_stress);
      value.SetSize(dimension);
      value = 0.0;
      for (int i = 0; i < dimension; ++i)
      {
         for (int j = 0; j < dimension; ++j)
         {
            const real_t stress = imaginary ?
                                  imaginary_stress(i, j) +
                                  omega*damping_beta*real_stress(i, j) :
                                  real_stress(i, j) -
                                  omega*damping_beta*imaginary_stress(i, j);
            value(i) += stress*normal(j);
         }
      }
   }

   const int dimension;
   const real_t length = 4.0;
   const real_t height = 1.0;
   const real_t width = 1.0;
   const real_t lambda = 2.3;
   const real_t mu = 1.7;
   const real_t density = 0.9;
   const real_t damping_alpha = 0.08;
   const real_t damping_beta = 0.015;
   const real_t omega;
   Vector real_amplitude;
   Vector imaginary_amplitude;
   Vector real_shift;
   Vector imaginary_shift;
   Vector real_wave_number;
   Vector imaginary_wave_number;

private:
   static real_t ModeFactor(const int direction, const int derivative,
                            const real_t coordinate,
                            const real_t wave_number)
   {
      const real_t argument = wave_number*coordinate;
      if (direction == 0)
      {
         if (derivative == 0) { return std::sin(argument); }
         if (derivative == 1)
         {
            return wave_number*std::cos(argument);
         }
         return -wave_number*wave_number*std::sin(argument);
      }
      if (derivative == 0) { return std::cos(argument); }
      if (derivative == 1)
      {
         return -wave_number*std::sin(argument);
      }
      return -wave_number*wave_number*std::cos(argument);
   }

   real_t EvaluateMode(const Vector &position, const Vector &wave_number,
                       Vector *gradient = nullptr,
                       DenseMatrix *hessian = nullptr) const
   {
      real_t value = 1.0;
      for (int d = 0; d < dimension; ++d)
      {
         value *= ModeFactor(d, 0, position(d), wave_number(d));
      }
      if (gradient)
      {
         gradient->SetSize(dimension);
         for (int i = 0; i < dimension; ++i)
         {
            (*gradient)(i) = 1.0;
            for (int d = 0; d < dimension; ++d)
            {
               (*gradient)(i) *= ModeFactor(
                                    d, d == i ? 1 : 0,
                                    position(d), wave_number(d));
            }
         }
      }
      if (hessian)
      {
         hessian->SetSize(dimension);
         for (int i = 0; i < dimension; ++i)
         {
            for (int j = 0; j < dimension; ++j)
            {
               (*hessian)(i, j) = 1.0;
               for (int d = 0; d < dimension; ++d)
               {
                  const int derivative = (d == i) + (d == j);
                  (*hessian)(i, j) *= ModeFactor(
                                         d, derivative, position(d),
                                         wave_number(d));
               }
            }
         }
      }
      return value;
   }

   void Stiffness(const Vector &position, const bool imaginary,
                  Vector &value) const
   {
      const Vector &amplitude = imaginary ? imaginary_amplitude :
                                real_amplitude;
      const Vector &wave_number = imaginary ? imaginary_wave_number :
                                  real_wave_number;
      DenseMatrix hessian;
      const real_t mode = EvaluateMode(
                             position, wave_number, nullptr, &hessian);
      real_t laplacian = 0.0;
      for (int d = 0; d < dimension; ++d)
      {
         laplacian -= wave_number(d)*wave_number(d)*mode;
      }
      value.SetSize(dimension);
      for (int i = 0; i < dimension; ++i)
      {
         real_t gradient_divergence = 0.0;
         for (int j = 0; j < dimension; ++j)
         {
            gradient_divergence += hessian(i, j)*amplitude(j);
         }
         value(i) = -mu*amplitude(i)*laplacian
                    - (lambda + mu)*gradient_divergence;
      }
   }

   void Stress(const Vector &position, const bool imaginary,
               DenseMatrix &stress) const
   {
      const Vector &amplitude = imaginary ? imaginary_amplitude :
                                real_amplitude;
      const Vector &wave_number = imaginary ? imaginary_wave_number :
                                  real_wave_number;
      Vector gradient;
      EvaluateMode(position, wave_number, &gradient);
      real_t divergence = 0.0;
      for (int i = 0; i < dimension; ++i)
      {
         divergence += amplitude(i)*gradient(i);
      }
      stress.SetSize(dimension);
      for (int i = 0; i < dimension; ++i)
      {
         for (int j = 0; j < dimension; ++j)
         {
            stress(i, j) = mu*(amplitude(i)*gradient(j) +
                               amplitude(j)*gradient(i));
            if (i == j) { stress(i, j) += lambda*divergence; }
         }
      }
   }
};

class ManufacturedDisplacementCoefficient : public VectorCoefficient
{
public:
   ManufacturedDisplacementCoefficient(
      const ManufacturedSolution &solution, const bool imaginary)
      : VectorCoefficient(solution.dimension), solution_(solution),
        imaginary_(imaginary) { }

   void Eval(Vector &value, ElementTransformation &transformation,
             const IntegrationPoint &point) override
   {
      transformation.Transform(point, position_);
      solution_.Displacement(position_, imaginary_, value);
   }

private:
   const ManufacturedSolution &solution_;
   const bool imaginary_;
   Vector position_;
};

class ManufacturedBodyForceCoefficient : public VectorCoefficient
{
public:
   ManufacturedBodyForceCoefficient(
      const ManufacturedSolution &solution, const bool imaginary)
      : VectorCoefficient(solution.dimension), solution_(solution),
        imaginary_(imaginary) { }

   void Eval(Vector &value, ElementTransformation &transformation,
             const IntegrationPoint &point) override
   {
      transformation.Transform(point, position_);
      solution_.BodyForce(position_, imaginary_, value);
   }

private:
   const ManufacturedSolution &solution_;
   const bool imaginary_;
   Vector position_;
};

class ManufacturedTractionCoefficient : public VectorCoefficient
{
public:
   ManufacturedTractionCoefficient(const ManufacturedSolution &solution,
                                    const Vector &normal,
                                    const bool imaginary)
      : VectorCoefficient(solution.dimension), solution_(solution),
        normal_(normal), imaginary_(imaginary) { }

   void Eval(Vector &value, ElementTransformation &transformation,
             const IntegrationPoint &point) override
   {
      transformation.Transform(point, position_);
      solution_.Traction(position_, normal_, imaginary_, value);
   }

private:
   const ManufacturedSolution &solution_;
   const Vector normal_;
   const bool imaginary_;
   Vector position_;
};

struct SolverConfiguration
{
   // New frequency regimes only need an additional configuration row.
   std::string name;
   real_t frequency;
   std::function<void(FrequencyDomainLinearElasticitySolver &)> configure;
};

struct LevelResult
{
   long long elements = 0;
   HYPRE_BigInt dofs = 0;
   int iterations = 0;
   real_t relative_l2_error = 0.0;
   real_t relative_h1_error = 0.0;
   real_t support_error = 0.0;
};

real_t GlobalMaximum(const real_t local_value, const MPI_Comm communicator)
{
   real_t global_value = 0.0;
   MPI_Allreduce(&local_value, &global_value, 1,
                 MPITypeMap<real_t>::mpi_type, MPI_MAX, communicator);
   return global_value;
}

void AddNaturalTraction(FrequencyDomainLinearElasticitySolver &solver,
                        const ManufacturedSolution &solution,
                        const int attribute, const int direction,
                        const real_t sign)
{
   Vector normal(solution.dimension);
   normal = 0.0;
   normal(direction) = sign;
   solver.AddBoundaryLoad(
      attribute,
      std::make_shared<ManufacturedTractionCoefficient>(
         solution, normal, false),
      std::make_shared<ManufacturedTractionCoefficient>(
         solution, normal, true));
}

void AddNaturalTractions(FrequencyDomainLinearElasticitySolver &solver,
                         const ManufacturedSolution &solution)
{
   if (solution.dimension == 2)
   {
      AddNaturalTraction(solver, solution, 1, 1, -1.0);
      AddNaturalTraction(solver, solution, 2, 0, 1.0);
      AddNaturalTraction(solver, solution, 3, 1, 1.0);
   }
   else
   {
      AddNaturalTraction(solver, solution, 1, 2, -1.0);
      AddNaturalTraction(solver, solution, 2, 1, -1.0);
      AddNaturalTraction(solver, solution, 3, 0, 1.0);
      AddNaturalTraction(solver, solution, 4, 1, 1.0);
      AddNaturalTraction(solver, solution, 6, 2, 1.0);
   }
}

void ComputeErrors(const ParComplexGridFunction &numerical,
                   const ManufacturedSolution &exact, const int order,
                   real_t &relative_l2_error,
                   real_t &relative_h1_error)
{
   const ParFiniteElementSpace &space = *numerical.ParFESpace();
   ParMesh &mesh = *space.GetParMesh();
   real_t local_squares[4] = {0.0, 0.0, 0.0, 0.0};
   Vector position;
   Vector numerical_real;
   Vector numerical_imaginary;
   Vector exact_real;
   Vector exact_imaginary;
   DenseMatrix numerical_real_gradient;
   DenseMatrix numerical_imaginary_gradient;
   DenseMatrix exact_real_gradient;
   DenseMatrix exact_imaginary_gradient;

   for (int element = 0; element < mesh.GetNE(); ++element)
   {
      const FiniteElement &finite_element = *space.GetFE(element);
      ElementTransformation &transformation =
         *mesh.GetElementTransformation(element);
      const IntegrationRule &rule = IntRules.Get(
                                       finite_element.GetGeomType(),
                                       2*order + 6);
      for (int q = 0; q < rule.GetNPoints(); ++q)
      {
         const IntegrationPoint &point = rule.IntPoint(q);
         transformation.SetIntPoint(&point);
         transformation.Transform(point, position);
         numerical.real().GetVectorValue(
            transformation, point, numerical_real);
         numerical.imag().GetVectorValue(
            transformation, point, numerical_imaginary);
         numerical.real().GetVectorGradient(
            transformation, numerical_real_gradient);
         numerical.imag().GetVectorGradient(
            transformation, numerical_imaginary_gradient);
         exact.Displacement(position, false, exact_real);
         exact.Displacement(position, true, exact_imaginary);
         exact.Gradient(position, false, exact_real_gradient);
         exact.Gradient(position, true, exact_imaginary_gradient);

         real_t error_value_squared = 0.0;
         real_t exact_value_squared = 0.0;
         real_t error_gradient_squared = 0.0;
         real_t exact_gradient_squared = 0.0;
         for (int i = 0; i < exact.dimension; ++i)
         {
            const real_t real_error = numerical_real(i) - exact_real(i);
            const real_t imaginary_error =
               numerical_imaginary(i) - exact_imaginary(i);
            error_value_squared += real_error*real_error +
                                   imaginary_error*imaginary_error;
            exact_value_squared += exact_real(i)*exact_real(i) +
                                   exact_imaginary(i)*exact_imaginary(i);
            for (int j = 0; j < exact.dimension; ++j)
            {
               const real_t real_gradient_error =
                  numerical_real_gradient(i, j) - exact_real_gradient(i, j);
               const real_t imaginary_gradient_error =
                  numerical_imaginary_gradient(i, j) -
                  exact_imaginary_gradient(i, j);
               error_gradient_squared +=
                  real_gradient_error*real_gradient_error +
                  imaginary_gradient_error*imaginary_gradient_error;
               exact_gradient_squared +=
                  exact_real_gradient(i, j)*exact_real_gradient(i, j) +
                  exact_imaginary_gradient(i, j)*
                  exact_imaginary_gradient(i, j);
            }
         }
         const real_t weight = point.weight*transformation.Weight();
         local_squares[0] += weight*error_value_squared;
         local_squares[1] += weight*error_gradient_squared;
         local_squares[2] += weight*exact_value_squared;
         local_squares[3] += weight*exact_gradient_squared;
      }
   }

   real_t global_squares[4] = {0.0, 0.0, 0.0, 0.0};
   MPI_Allreduce(local_squares, global_squares, 4,
                 MPITypeMap<real_t>::mpi_type, MPI_SUM, mesh.GetComm());
   MFEM_VERIFY(global_squares[2] > 0.0 &&
               global_squares[2] + global_squares[3] > 0.0,
               "Manufactured solution must have a nonzero norm.");
   relative_l2_error = std::sqrt(global_squares[0]/global_squares[2]);
   relative_h1_error = std::sqrt(
                          (global_squares[0] + global_squares[1])/
                          (global_squares[2] + global_squares[3]));
}

real_t ComputeSupportError(const ParComplexGridFunction &numerical,
                           ParFiniteElementSpace &space,
                           const ManufacturedSolution &exact,
                           const Array<int> &essential_tdofs)
{
   ManufacturedDisplacementCoefficient real_exact(exact, false);
   ManufacturedDisplacementCoefficient imaginary_exact(exact, true);
   ParComplexGridFunction projected_exact(&space);
   projected_exact.ProjectCoefficient(real_exact, imaginary_exact);
   Vector numerical_true(2*space.GetTrueVSize());
   Vector exact_true(2*space.GetTrueVSize());
   numerical.ParallelProject(numerical_true);
   projected_exact.ParallelProject(exact_true);
   const real_t *numerical_values = numerical_true.HostRead();
   const real_t *exact_values = exact_true.HostRead();
   const int block_size = space.GetTrueVSize();
   real_t local_error = 0.0;
   for (int i = 0; i < essential_tdofs.Size(); ++i)
   {
      const int dof = essential_tdofs[i];
      local_error = std::max(
                       local_error,
                       std::abs(numerical_values[dof] - exact_values[dof]));
      local_error = std::max(
                       local_error,
                       std::abs(numerical_values[block_size + dof] -
                                exact_values[block_size + dof]));
   }
   return GlobalMaximum(local_error, space.GetComm());
}

LevelResult RunLevel(const int dimension, const int order,
                     const int refinement,
                     const SolverConfiguration &configuration,
                     const bool moving_support)
{
   ManufacturedSolution exact(dimension, configuration.frequency,
                              moving_support);
   Mesh serial_mesh;
   if (dimension == 2)
   {
      serial_mesh = Mesh::MakeCartesian2D(
                       4, 1, Element::QUADRILATERAL, true,
                       exact.length, exact.height);
   }
   else
   {
      serial_mesh = Mesh::MakeCartesian3D(
                       4, 1, 1, Element::HEXAHEDRON,
                       exact.length, exact.height, exact.width);
   }
   // Partial assembly requires a nonempty local mesh. Refine before
   // partitioning when the four-element seed is smaller than the MPI job.
   while (serial_mesh.GetNE() < Mpi::WorldSize())
   {
      serial_mesh.UniformRefinement();
   }
   ParMesh mesh(MPI_COMM_WORLD, serial_mesh);
   for (int level = 0; level < refinement; ++level)
   {
      mesh.UniformRefinement();
   }

   H1_FECollection collection(order, dimension);
   ParFiniteElementSpace space(&mesh, &collection, dimension,
                               Ordering::byNODES);
   FrequencyDomainLinearElasticitySolver solver(space);
   solver.SetLameMaterial(
      std::make_shared<ConstantCoefficient>(exact.lambda),
      std::make_shared<ConstantCoefficient>(exact.mu),
      std::make_shared<ConstantCoefficient>(exact.density));
   solver.SetFrequency(configuration.frequency);
   solver.SetRayleighDamping(exact.damping_alpha, exact.damping_beta);
   solver.SetRelTol(1.0e-12);
   solver.SetAbsTol(1.0e-14);
   solver.SetMaxIter(1000);
   solver.SetKDim(100);
   solver.SetPrintLevel(-1);
   solver.SetPreconditionerPrintLevel(-1);
   configuration.configure(solver);

   const int support_attribute = dimension == 2 ? 4 : 5;
   if (moving_support)
   {
      solver.AddDisplacementBC(
         support_attribute,
         std::make_shared<VectorConstantCoefficient>(exact.real_shift),
         std::make_shared<VectorConstantCoefficient>(exact.imaginary_shift));
   }
   else
   {
      solver.AddBoundaryID(support_attribute);
   }
   solver.AddVolumeLoad(
      1,
      std::make_shared<ManufacturedBodyForceCoefficient>(exact, false),
      std::make_shared<ManufacturedBodyForceCoefficient>(exact, true));
   AddNaturalTractions(solver, exact);

   ParComplexGridFunction displacement(&space);
   displacement = std::complex<real_t>(0.0, 0.0);
   solver.Solve(displacement);

   LevelResult result;
   ComputeErrors(displacement, exact, order, result.relative_l2_error,
                 result.relative_h1_error);
   result.support_error = ComputeSupportError(
                             displacement, space, exact,
                             solver.GetEssentialTrueDofs());
   result.iterations = solver.GetNumIterations();
   result.elements = mesh.GetGlobalNE();
   result.dofs = space.GlobalTrueVSize();
   return result;
}

bool CheckConvergence(const std::string &name,
                      const std::vector<LevelResult> &results,
                      const int order)
{
   bool passed = true;
   for (const LevelResult &result : results)
   {
      passed = passed && std::isfinite(result.relative_l2_error) &&
               std::isfinite(result.relative_h1_error) &&
               result.support_error <= 1.0e-11;
   }
   for (std::size_t level = 1; level < results.size(); ++level)
   {
      passed = passed &&
               results[level].relative_l2_error <
               results[level - 1].relative_l2_error &&
               results[level].relative_h1_error <
               results[level - 1].relative_h1_error;
   }
   const LevelResult &previous = results[results.size() - 2];
   const LevelResult &final = results.back();
   const real_t l2_rate = std::log(previous.relative_l2_error/
                                   final.relative_l2_error)/std::log(2.0);
   const real_t h1_rate = std::log(previous.relative_h1_error/
                                   final.relative_h1_error)/std::log(2.0);
   passed = passed && std::isfinite(l2_rate) && std::isfinite(h1_rate) &&
            l2_rate >= order + 0.5 && h1_rate >= order - 0.35;

   if (Mpi::Root())
   {
      for (std::size_t level = 0; level < results.size(); ++level)
      {
         const LevelResult &result = results[level];
         std::cout << "  level " << level
                   << ": elements=" << result.elements
                   << ", dofs=" << result.dofs
                   << ", iterations=" << result.iterations
                   << ", rel-L2=" << result.relative_l2_error
                   << ", rel-H1=" << result.relative_h1_error
                   << ", support-error=" << result.support_error << '\n';
      }
      std::cout << (passed ? "PASS  " : "FAIL  ") << name
                << ": final L2 rate=" << l2_rate
                << ", final H1 rate=" << h1_rate << '\n';
   }
   return passed;
}

} // namespace

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();

   const char *device_configuration = "cpu";
   const char *boundary_case = "all";
   int dimension = 0;
   int order = 2;
   int refinement_levels = 3;
   OptionsParser args(argc, argv);
   args.AddOption(&device_configuration, "-d", "--device",
                  "MFEM device configuration.");
   args.AddOption(&dimension, "-dim", "--dimension",
                  "Dimension to test: 0 for both, 2, or 3.");
   args.AddOption(&order, "-o", "--order", "H1 polynomial degree.");
   args.AddOption(&refinement_levels, "-rl", "--refinement-levels",
                  "Number of mesh levels; at least two.");
   args.AddOption(&boundary_case, "-bc", "--boundary-case",
                  "Boundary case: all, clamped, or support.");
   args.ParseCheck();

   const std::string selected_boundary_case(boundary_case);
   MFEM_VERIFY(dimension == 0 || dimension == 2 || dimension == 3,
               "Dimension must be 0, 2, or 3.");
   MFEM_VERIFY(order > 0, "The polynomial degree must be positive.");
   MFEM_VERIFY(refinement_levels >= 2,
               "At least two mesh levels are required.");
   MFEM_VERIFY(selected_boundary_case == "all" ||
               selected_boundary_case == "clamped" ||
               selected_boundary_case == "support",
               "Boundary case must be all, clamped, or support.");

   Device device(device_configuration);
   std::vector<SolverConfiguration> configurations;
   configurations.push_back(
   {
      "low-frequency PRESB", 0.05,
      [](FrequencyDomainLinearElasticitySolver &solver)
      {
         solver.SetPreconditionerType(
            FrequencyDomainLinearElasticitySolver::PreconditionerType::PRESB);
         solver.SetHInverseType(
            FrequencyDomainLinearElasticitySolver::HInverseType::
            LORMonolithicAMG);
         solver.SetLinearSolverType(
            FrequencyDomainLinearElasticitySolver::LinearSolverType::
            Automatic);
      }
   });
   // Add future high-frequency preconditioners here. The manufactured data
   // depend on each row's frequency but not on its solver implementation.

   std::vector<int> dimensions;
   if (dimension == 0)
   {
      dimensions.push_back(2);
      dimensions.push_back(3);
   }
   else
   {
      dimensions.push_back(dimension);
   }
   std::vector<bool> support_cases;
   if (selected_boundary_case == "all")
   {
      support_cases.push_back(false);
      support_cases.push_back(true);
   }
   else
   {
      support_cases.push_back(selected_boundary_case == "support");
   }

   int failures = 0;
   for (const SolverConfiguration &configuration : configurations)
   {
      for (const int active_dimension : dimensions)
      {
         for (const bool moving_support : support_cases)
         {
            std::vector<LevelResult> results;
            for (int level = 0; level < refinement_levels; ++level)
            {
               results.push_back(RunLevel(active_dimension, order, level,
                                          configuration, moving_support));
            }
            const std::string name =
               configuration.name + ", " +
               std::to_string(active_dimension) + "D, " +
               (moving_support ? "support motion" : "clamped support");
            failures += !CheckConvergence(name, results, order);
         }
      }
   }

   int global_failures = 0;
   MPI_Allreduce(&failures, &global_failures, 1, MPI_INT, MPI_MAX,
                 MPI_COMM_WORLD);
   if (Mpi::Root())
   {
      std::cout << (global_failures == 0 ?
                    "ALL TESTS PASSED\n" : "TESTS FAILED\n");
   }
   return global_failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
