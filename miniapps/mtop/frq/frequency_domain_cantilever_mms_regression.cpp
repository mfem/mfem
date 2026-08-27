// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC.
//
// Verify the frequency-domain elasticity solver with smooth manufactured
// solutions on two- and three-dimensional cantilever beams, with optional
// CSV diagnostics and ParaView fields for the numerical and exact solutions.

#include "frequency_domain_driver_utils.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <fstream>
#include <iomanip>
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
                        const bool moving_support,
                        const frequency_domain::DampingOptions &damping)
      : dimension(dimension), damping(damping), omega(omega),
        real_amplitude(dimension),
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
      // Form the real or imaginary part of
      // (K-omega^2 M+i omega C)(u_real+i u_imaginary) in strong form.
      Vector real_displacement;
      Vector imaginary_displacement;
      Vector real_stiffness;
      Vector imaginary_stiffness;
      Vector real_damping;
      Vector imaginary_damping;
      Displacement(position, false, real_displacement);
      Displacement(position, true, imaginary_displacement);
      Stiffness(position, false, real_stiffness);
      Stiffness(position, true, imaginary_stiffness);
      DampingAction(position, false, real_damping);
      DampingAction(position, true, imaginary_damping);

      value.SetSize(dimension);
      for (int i = 0; i < dimension; ++i)
      {
         if (imaginary)
         {
            value(i) = imaginary_stiffness(i)
                       - omega*omega*density*imaginary_displacement(i)
                       + omega*real_damping(i);
         }
         else
         {
            value(i) = real_stiffness(i)
                       - omega*omega*density*real_displacement(i)
                       - omega*imaginary_damping(i);
         }
      }
   }

   void Traction(const Vector &position, const Vector &normal,
                 const bool imaginary, Vector &value) const
   {
      // The matching natural data include both elastic and stiffness-like
      // damping stresses. Mass damping has no boundary traction.
      DenseMatrix real_stress;
      DenseMatrix imaginary_stress;
      DenseMatrix real_damping_stress;
      DenseMatrix imaginary_damping_stress;
      Stress(position, false, real_stress);
      Stress(position, true, imaginary_stress);
      DampingStress(position, false, real_damping_stress);
      DampingStress(position, true, imaginary_damping_stress);
      value.SetSize(dimension);
      value = 0.0;
      for (int i = 0; i < dimension; ++i)
      {
         for (int j = 0; j < dimension; ++j)
         {
            const real_t stress = imaginary ?
                                  imaginary_stress(i, j) +
                                  omega*real_damping_stress(i, j) :
                                  real_stress(i, j) -
                                  omega*imaginary_damping_stress(i, j);
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
   const frequency_domain::DampingOptions damping;
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
                  Vector &value, const real_t active_lambda,
                  const real_t active_mu) const
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
         value(i) = -active_mu*amplitude(i)*laplacian
                    - (active_lambda + active_mu)*gradient_divergence;
      }
   }

   void Stiffness(const Vector &position, const bool imaginary,
                  Vector &value) const
   {
      Stiffness(position, imaginary, value, lambda, mu);
   }

   void Stress(const Vector &position, const bool imaginary,
               DenseMatrix &stress, const real_t active_lambda,
               const real_t active_mu) const
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
            stress(i, j) = active_mu*(amplitude(i)*gradient(j) +
                                      amplitude(j)*gradient(i));
            if (i == j) { stress(i, j) += active_lambda*divergence; }
         }
      }
   }

   void Stress(const Vector &position, const bool imaginary,
               DenseMatrix &stress) const
   {
      Stress(position, imaginary, stress, lambda, mu);
   }

   void DampingAction(const Vector &position, const bool imaginary,
                      Vector &value) const
   {
      Vector displacement;
      Displacement(position, imaginary, displacement);
      if (std::string(damping.model) == "rayleigh")
      {
         Stiffness(position, imaginary, value);
         value *= damping.beta;
         value.Add(damping.alpha*density, displacement);
      }
      else
      {
         Stiffness(position, imaginary, value,
                   damping.lambda, damping.mu);
         value.Add(damping.mass, displacement);
      }
   }

   void DampingStress(const Vector &position, const bool imaginary,
                      DenseMatrix &stress) const
   {
      if (std::string(damping.model) == "rayleigh")
      {
         Stress(position, imaginary, stress);
         stress *= damping.beta;
      }
      else
      {
         Stress(position, imaginary, stress, damping.lambda, damping.mu);
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

/// Scalar magnitude of a pair of vector grid functions.
class VectorPairMagnitude : public Coefficient
{
public:
   VectorPairMagnitude(const ParGridFunction &real_part,
                       const ParGridFunction &imaginary_part)
      : real_part_(real_part), imaginary_part_(imaginary_part) { }

   real_t Eval(ElementTransformation &transformation,
               const IntegrationPoint &point) override
   {
      real_part_.GetVectorValue(transformation, point, real_value_);
      imaginary_part_.GetVectorValue(transformation, point, imaginary_value_);
      return std::sqrt(real_value_*real_value_ +
                       imaginary_value_*imaginary_value_);
   }

private:
   const ParGridFunction &real_part_;
   const ParGridFunction &imaginary_part_;
   Vector real_value_;
   Vector imaginary_value_;
};

struct LevelResult
{
   int dimension = 0;
   int level = 0;
   bool moving_support = false;
   long long elements = 0;
   HYPRE_BigInt dofs = 0;
   HYPRE_BigInt total_dofs = 0;
   real_t frequency = 0.0;
   frequency_domain::EigenDiagnostics eigen;
   real_t frequency_ratio = 0.0;
   real_t loss_factor = 0.0;
   real_t h_indicator = 0.0;
   bool converged = false;
   int outer_iterations = 0;
   int preconditioner_applications = 0;
   int h_inverse_applications = 0;
   int h_inverse_iterations = 0;
   real_t initial_residual = -1.0;
   real_t final_residual = -1.0;
   real_t relative_residual = -1.0;
   real_t relative_l2_error = 0.0;
   real_t relative_h1_error = 0.0;
   real_t support_error = 0.0;
   double eigen_time = 0.0;
   double assembly_time = 0.0;
   double operator_assembly_time = 0.0;
   double preconditioner_setup_time = 0.0;
   double solver_setup_time = 0.0;
   double solve_time = 0.0;
   double load_time = 0.0;
   double linear_solve_time = 0.0;
   double distribution_time = 0.0;
   double visualization_time = 0.0;
   FrequencyDomainLinearElasticitySolver::LinearSolverType active_solver =
      FrequencyDomainLinearElasticitySolver::LinearSolverType::Automatic;
};

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
   return frequency_domain::GlobalMaximum(local_error, space.GetComm());
}

LevelResult RunLevel(const int dimension, const int order,
                     const int refinement,
                     const bool moving_support, const real_t input_frequency,
                     const real_t frequency_factor,
                     const frequency_domain::SolverOptions &solver_options,
                     const frequency_domain::DampingOptions &damping_options,
                     const frequency_domain::EigenOptions &eigen_options,
                     const bool visualization, const char *output_prefix)
{
   const real_t length = 4.0;
   const real_t height = 1.0;
   const real_t width = 1.0;
   const real_t lambda = 2.3;
   const real_t mu = 1.7;
   const real_t density = 0.9;
   Mesh serial_mesh;
   if (dimension == 2)
   {
      serial_mesh = Mesh::MakeCartesian2D(
                       4, 1, Element::QUADRILATERAL, true,
                       length, height);
   }
   else
   {
      serial_mesh = Mesh::MakeCartesian3D(
                       4, 1, 1, Element::HEXAHEDRON,
                       length, height, width);
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
   const int support_attribute = dimension == 2 ? 4 : 5;
   Array<int> essential_boundary(mesh.bdr_attributes.Max());
   essential_boundary = 0;
   essential_boundary[support_attribute - 1] = 1;
   ConstantCoefficient lambda_coefficient(lambda);
   ConstantCoefficient mu_coefficient(mu);
   ConstantCoefficient density_coefficient(density);
   const frequency_domain::EigenDiagnostics eigen =
      frequency_domain::ComputeEigenDiagnostics(
         space, lambda_coefficient, mu_coefficient, density_coefficient,
         essential_boundary, damping_options, eigen_options);
   const real_t frequency = frequency_domain::ResolveFrequency(
                               input_frequency, frequency_factor, eigen);
   ManufacturedSolution exact(dimension, frequency, moving_support,
                              damping_options);

   FrequencyDomainLinearElasticitySolver solver(space);
   solver.SetLameMaterial(
      std::make_shared<ConstantCoefficient>(lambda),
      std::make_shared<ConstantCoefficient>(mu),
      std::make_shared<ConstantCoefficient>(density));
   solver.SetFrequency(frequency);
   damping_options.Apply(solver);
   solver_options.Apply(solver);

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
   StopWatch assembly_timer;
   assembly_timer.Start();
   solver.Assemble();
   assembly_timer.Stop();
   StopWatch solve_timer;
   solve_timer.Start();
   solver.Solve(displacement);
   solve_timer.Stop();

   LevelResult result;
   result.dimension = dimension;
   result.level = refinement;
   result.moving_support = moving_support;
   result.frequency = frequency;
   result.eigen = eigen;
   result.frequency_ratio = frequency/eigen.omega1;
   result.loss_factor = frequency_domain::LossFactor(frequency, eigen);
   result.h_indicator = frequency_domain::HIndicator(frequency, eigen);
   ComputeErrors(displacement, exact, order, result.relative_l2_error,
                 result.relative_h1_error);
   result.support_error = ComputeSupportError(
                             displacement, space, exact,
                             solver.GetEssentialTrueDofs());
   result.converged = solver.GetConverged();
   result.outer_iterations = solver.GetNumIterations();
   result.preconditioner_applications =
      solver.GetNumPreconditionerApplications();
   result.h_inverse_applications = solver.GetNumHInverseApplications();
   result.h_inverse_iterations = solver.GetNumHInverseIterations();
   result.initial_residual = solver.GetInitialNorm();
   result.final_residual = solver.GetFinalNorm();
   result.relative_residual = result.initial_residual > 0.0 ?
                              result.final_residual/result.initial_residual :
                              -1.0;
   result.elements = mesh.GetGlobalNE();
   result.dofs = space.GlobalTrueVSize();
   result.total_dofs = 2*result.dofs;
   result.active_solver = solver.GetActiveLinearSolverType();

   double visualization_time = 0.0;
   if (visualization)
   {
      StopWatch visualization_timer;
      visualization_timer.Start();
      ManufacturedDisplacementCoefficient exact_real_coefficient(exact,
                                                                   false);
      ManufacturedDisplacementCoefficient exact_imaginary_coefficient(exact,
                                                                        true);
      ParGridFunction exact_real(&space);
      ParGridFunction exact_imaginary(&space);
      exact_real.ProjectCoefficient(exact_real_coefficient);
      exact_imaginary.ProjectCoefficient(exact_imaginary_coefficient);
      ParGridFunction error_real(&space);
      ParGridFunction error_imaginary(&space);
      error_real = displacement.real();
      error_real -= exact_real;
      error_imaginary = displacement.imag();
      error_imaginary -= exact_imaginary;

      ParFiniteElementSpace scalar_space(&mesh, &collection);
      ParGridFunction numerical_magnitude(&scalar_space);
      ParGridFunction exact_magnitude(&scalar_space);
      ParGridFunction error_magnitude(&scalar_space);
      VectorPairMagnitude numerical_magnitude_coefficient(
         displacement.real(), displacement.imag());
      VectorPairMagnitude exact_magnitude_coefficient(
         exact_real, exact_imaginary);
      VectorPairMagnitude error_magnitude_coefficient(
         error_real, error_imaginary);
      numerical_magnitude.ProjectCoefficient(
         numerical_magnitude_coefficient);
      exact_magnitude.ProjectCoefficient(exact_magnitude_coefficient);
      error_magnitude.ProjectCoefficient(error_magnitude_coefficient);

      const std::string boundary_name = moving_support ? "support" : "clamped";
      const std::string collection_name =
         "frequency_domain_cantilever_mms_" + std::to_string(dimension) +
         "d_" + boundary_name + "_level_" + std::to_string(refinement);
      ParaViewDataCollection paraview(collection_name, &mesh);
      paraview.SetPrefixPath(output_prefix);
      paraview.SetLevelsOfDetail(order);
      paraview.SetDataFormat(VTKFormat::BINARY);
      paraview.SetHighOrderOutput(true);
      paraview.SetCycle(refinement);
      paraview.SetTime(frequency);
      paraview.RegisterField("displacement_real", &displacement.real());
      paraview.RegisterField("displacement_imaginary", &displacement.imag());
      paraview.RegisterField("exact_real", &exact_real);
      paraview.RegisterField("exact_imaginary", &exact_imaginary);
      paraview.RegisterField("error_real", &error_real);
      paraview.RegisterField("error_imaginary", &error_imaginary);
      paraview.RegisterField("displacement_magnitude", &numerical_magnitude);
      paraview.RegisterField("exact_magnitude", &exact_magnitude);
      paraview.RegisterField("error_magnitude", &error_magnitude);
      paraview.Save();
      visualization_timer.Stop();
      visualization_time = visualization_timer.RealTime();
      if (Mpi::Root())
      {
         std::string output_path(output_prefix);
         if (!output_path.empty() && output_path.back() != '/')
         {
            output_path += '/';
         }
         output_path += collection_name + "/" + collection_name + ".pvd";
         std::cout << "MMS ParaView collection: " << output_path << '\n';
      }
   }

   const double local_times[] =
   {
      eigen.solve_time, assembly_timer.RealTime(), solver.GetAssemblyTime(),
      solver.GetPreconditionerAssemblyTime(), solver.GetSolverSetupTime(),
      solve_timer.RealTime(), solver.GetLoadAssemblyTime(),
      solver.GetLinearSolveTime(), solver.GetSolutionDistributionTime(),
      visualization_time
   };
   double maximum_times[10];
   frequency_domain::GlobalMaximum(local_times, maximum_times, 10,
                                   mesh.GetComm());
   result.eigen_time = maximum_times[0];
   result.assembly_time = maximum_times[1];
   result.operator_assembly_time = maximum_times[2];
   result.preconditioner_setup_time = maximum_times[3];
   result.solver_setup_time = maximum_times[4];
   result.solve_time = maximum_times[5];
   result.load_time = maximum_times[6];
   result.linear_solve_time = maximum_times[7];
   result.distribution_time = maximum_times[8];
   result.visualization_time = maximum_times[9];
   return result;
}

bool CheckConvergence(const std::string &name,
                      const std::vector<LevelResult> &results,
                      const int order)
{
   bool passed = true;
   for (const LevelResult &result : results)
   {
      passed = passed && result.converged &&
               std::isfinite(result.relative_l2_error) &&
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
                   << ", displacement-dofs=" << result.dofs
                   << ", total-dofs=" << result.total_dofs
                   << ", omega1=" << result.eigen.omega1
                   << ", omega=" << result.frequency
                   << ", omega/omega1=" << result.frequency_ratio
                   << ", c1=" << result.eigen.modal_damping
                   << ", zeta1=" << result.eigen.damping_ratio
                   << ", eta1=" << result.loss_factor
                   << ", H1=" << result.h_indicator
                   << ", solver="
                   << frequency_domain::LinearSolverName(
                         result.active_solver)
                   << ", converged=" << (result.converged ? "yes" : "no")
                   << ", outer-it=" << result.outer_iterations
                   << ", pc-applications="
                   << result.preconditioner_applications
                   << ", H-applications=" << result.h_inverse_applications
                   << ", H-iterations/cycles=" << result.h_inverse_iterations
                   << ", relative-residual=" << result.relative_residual
                   << ", eigen-time=" << result.eigen_time << " s"
                   << ", assembly-time=" << result.assembly_time << " s"
                   << ", solve-time=" << result.solve_time << " s"
                   << ", linear-solve-time=" << result.linear_solve_time
                   << " s"
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

void WriteCSVHeader(std::ostream &output)
{
   output << "dimension,boundary_case,level,requested_solver,active_solver,"
             "preconditioner,h_inverse,lor_ordering,damping_model,"
             "damping_alpha,damping_beta,mass_damping,damping_lambda,"
             "damping_mu,elements,displacement_dofs,total_dofs,frequency,"
             "lambda1,"
             "omega1,frequency_ratio,modal_damping,damping_ratio,loss_factor,"
             "h_indicator,converged,outer_iterations,"
             "preconditioner_applications,h_inverse_applications,"
             "h_inverse_iterations,initial_residual,final_residual,"
             "relative_residual,relative_l2_error,relative_h1_error,"
             "support_error,eigen_time,assembly_time,operator_time,"
             "preconditioner_setup_time,solver_setup_time,solve_time,"
             "load_time,linear_solve_time,distribution_time,"
             "visualization_time\n";
}

void WriteCSVRow(std::ostream &output, const LevelResult &result,
                 const frequency_domain::SolverOptions &solver_options,
                 const frequency_domain::DampingOptions &damping_options)
{
   output << result.dimension << ','
          << (result.moving_support ? "support" : "clamped") << ','
          << result.level << ',' << solver_options.linear_solver << ','
          << frequency_domain::LinearSolverName(result.active_solver) << ','
          << (result.active_solver ==
              FrequencyDomainLinearElasticitySolver::LinearSolverType::MUMPS ?
              "none" : solver_options.preconditioner) << ','
          << (result.active_solver ==
              FrequencyDomainLinearElasticitySolver::LinearSolverType::MUMPS ?
              "none" : solver_options.h_inverse)
          << ',' << solver_options.lor_ordering << ',' << damping_options.model
          << ',' << damping_options.alpha << ',' << damping_options.beta << ','
          << damping_options.mass << ',' << damping_options.lambda << ','
          << damping_options.mu << ',' << result.elements << ',' << result.dofs
          << ',' << result.total_dofs << ',' << result.frequency << ','
          << result.eigen.lambda1 << ','
          << result.eigen.omega1 << ',' << result.frequency_ratio << ','
          << result.eigen.modal_damping << ',' << result.eigen.damping_ratio
          << ',' << result.loss_factor << ',' << result.h_indicator << ','
          << (result.converged ? 1 : 0) << ',' << result.outer_iterations
          << ',' << result.preconditioner_applications << ','
          << result.h_inverse_applications << ',' << result.h_inverse_iterations
          << ',' << result.initial_residual << ',' << result.final_residual
          << ',' << result.relative_residual << ',' << result.relative_l2_error
          << ',' << result.relative_h1_error << ',' << result.support_error
          << ',' << result.eigen_time << ',' << result.assembly_time << ','
          << result.operator_assembly_time << ','
          << result.preconditioner_setup_time << ',' << result.solver_setup_time
          << ',' << result.solve_time << ',' << result.load_time << ','
          << result.linear_solve_time << ',' << result.distribution_time << ','
          << result.visualization_time << '\n';
}

} // namespace

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();

   const char *device_configuration = "cpu";
   const char *boundary_case = "all";
   const char *output_prefix = "ParaView";
   const char *visualization_levels = "final";
   const char *csv_path = "";
   bool visualization = false;
   int dimension = 0;
   int order = 2;
   int refinement_levels = 3;
   real_t frequency = 0.05;
   real_t frequency_factor = -1.0;
   frequency_domain::SolverOptions solver_options;
   solver_options.relative_tolerance = 1.0e-12;
   solver_options.absolute_tolerance = 1.0e-14;
   solver_options.max_iterations = 1000;
   solver_options.kdim = 100;
   solver_options.print_level = -1;
   frequency_domain::DampingOptions damping_options;
   frequency_domain::EigenOptions eigen_options;
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
   args.AddOption(&frequency, "-f", "--frequency",
                  "Angular excitation frequency.");
   args.AddOption(&frequency_factor, "-ff", "--frequency-factor",
                  "Set frequency to this multiple of each level's lowest "
                  "eigenfrequency; a positive value overrides --frequency.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization", "Enable or disable ParaView output.");
   args.AddOption(&visualization_levels, "-vl", "--visualization-levels",
                  "ParaView levels: final or all.");
   args.AddOption(&output_prefix, "-out", "--output-prefix",
                  "ParaView output directory.");
   args.AddOption(&csv_path, "-csv", "--csv", "Optional CSV output file.");
   solver_options.AddOptions(args);
   damping_options.AddOptions(args);
   eigen_options.AddOptions(args);
   args.ParseCheck();

   const std::string selected_boundary_case(boundary_case);
   const std::string selected_visualization_levels(visualization_levels);
   MFEM_VERIFY(dimension == 0 || dimension == 2 || dimension == 3,
               "Dimension must be 0, 2, or 3.");
   MFEM_VERIFY(order > 0, "The polynomial degree must be positive.");
   MFEM_VERIFY(refinement_levels >= 2,
               "At least two mesh levels are required.");
   MFEM_VERIFY(selected_boundary_case == "all" ||
               selected_boundary_case == "clamped" ||
               selected_boundary_case == "support",
               "Boundary case must be all, clamped, or support.");
   MFEM_VERIFY(frequency >= 0.0, "Frequency must be nonnegative.");
   MFEM_VERIFY(frequency_factor == -1.0 || frequency_factor > 0.0,
               "Frequency factor must be positive when supplied.");
   MFEM_VERIFY(selected_visualization_levels == "final" ||
               selected_visualization_levels == "all",
               "Visualization levels must be final or all.");
   solver_options.Validate();
   damping_options.Validate();
   eigen_options.Validate();

   Device device(device_configuration);
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

   const bool csv_requested = std::string(csv_path).size() > 0;
   std::unique_ptr<std::ofstream> csv;
   int csv_open = 1;
   if (Mpi::Root() && csv_requested)
   {
      csv.reset(new std::ofstream(csv_path));
      csv_open = *csv ? 1 : 0;
      if (csv_open)
      {
         *csv << std::setprecision(17);
         WriteCSVHeader(*csv);
      }
   }
   if (csv_requested)
   {
      MPI_Bcast(&csv_open, 1, MPI_INT, 0, MPI_COMM_WORLD);
      MFEM_VERIFY(csv_open, "Unable to open CSV output file: " << csv_path);
   }

   if (Mpi::Root())
   {
      std::cout << std::setprecision(12)
                << "Requested linear solver: "
                << solver_options.linear_solver << '\n';
      if (std::string(solver_options.linear_solver) == "mumps")
      {
         std::cout << "Preconditioner: none\nH inverse: none\n";
      }
      else
      {
         std::cout << "Preconditioner: " << solver_options.preconditioner
                   << '\n'
                   << "H inverse: " << solver_options.h_inverse << '\n';
      }
      std::cout << "LOR ordering: " << solver_options.lor_ordering << '\n'
                << "Damping model: " << damping_options.model << '\n';
      if (std::string(damping_options.model) == "rayleigh")
      {
         std::cout << "Rayleigh damping alpha, beta: "
                   << damping_options.alpha << ", "
                   << damping_options.beta << '\n';
      }
      else
      {
         std::cout << "Independent damping c_M, lambda_C, mu_C: "
                   << damping_options.mass << ", "
                   << damping_options.lambda << ", "
                   << damping_options.mu << '\n';
      }
   }

   int failures = 0;
   for (const int active_dimension : dimensions)
   {
      for (const bool moving_support : support_cases)
      {
         std::vector<LevelResult> results;
         for (int level = 0; level < refinement_levels; ++level)
         {
            const bool save_level = visualization &&
               (selected_visualization_levels == "all" ||
                level == refinement_levels - 1);
            results.push_back(RunLevel(
               active_dimension, order, level, moving_support, frequency,
               frequency_factor, solver_options, damping_options,
               eigen_options, save_level, output_prefix));
            if (csv)
            {
               WriteCSVRow(*csv, results.back(), solver_options,
                           damping_options);
            }
         }
         const auto active_solver = results.back().active_solver;
         const std::string name =
            std::string(frequency_domain::LinearSolverName(active_solver)) +
            (active_solver ==
             FrequencyDomainLinearElasticitySolver::LinearSolverType::MUMPS ?
             "" : "/" + std::string(solver_options.preconditioner)) + ", " +
            std::to_string(active_dimension) + "D, " +
            (moving_support ? "support motion" : "clamped support");
         failures += !CheckConvergence(name, results, order);
      }
   }

   if (csv_requested)
   {
      int csv_written = 1;
      if (Mpi::Root())
      {
         csv->flush();
         csv_written = *csv ? 1 : 0;
      }
      MPI_Bcast(&csv_written, 1, MPI_INT, 0, MPI_COMM_WORLD);
      MFEM_VERIFY(csv_written,
                  "Unable to write CSV output file: " << csv_path);
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
