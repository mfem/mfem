#include "examples/support/mesh_manufactured.hpp"
#include "stokes.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <memory>
#include <string>

namespace mfem::flow
{
const char *MeshMMSCaseName(MeshMMSCase problem_case)
{
   switch (problem_case)
   {
      case MeshMMSCase::NoSlip: return "no-slip";
      case MeshMMSCase::Smooth: return "smooth";
   }
   return "unknown";
}

namespace
{

const real_t pi = std::acos(real_t(-1.0));
const real_t centers[5][3] = {{0.5, 0.25, 0.25}, {1.5, 0.25, 0.75},
   {1.0, 0.5, 0.5}, {0.5, 0.75, 0.75}, {1.5, 0.75, 0.25}};

struct Derivatives
{
   real_t value = 0.0;
   real_t gradient[3] = {};
   real_t hessian[3][3] = {};
   real_t third[3][3][3] = {};
};

Derivatives Product(const Derivatives &left, const Derivatives &right, int dimension)
{
   Derivatives result;
   result.value = left.value * right.value;
   for (int first = 0; first < dimension; ++first)
   {
      result.gradient[first] = left.gradient[first] * right.value + left.value * right.gradient[first];
      for (int second = 0; second < dimension; ++second)
      {
         result.hessian[first][second] = left.hessian[first][second] * right.value +
                                        left.gradient[first] * right.gradient[second] +
                                        left.gradient[second] * right.gradient[first] +
                                        left.value * right.hessian[first][second];
         for (int third = 0; third < dimension; ++third)
         {
            result.third[first][second][third] =
               left.third[first][second][third] * right.value +
               left.hessian[first][second] * right.gradient[third] +
               left.hessian[first][third] * right.gradient[second] +
               left.hessian[second][third] * right.gradient[first] +
               left.gradient[first] * right.hessian[second][third] +
               left.gradient[second] * right.hessian[first][third] +
               left.gradient[third] * right.hessian[first][second] +
               left.value * right.third[first][second][third];
         }
      }
   }
   return result;
}

Derivatives Potential(const Vector &point, int dimension, MeshMMSCase problem_case)
{
   constexpr real_t inner_squared = 0.10 * 0.10;
   constexpr real_t outer_squared = 0.20 * 0.20;
   constexpr real_t range = outer_squared - inner_squared;
   real_t coordinates[3] = {};
   for (int axis = 0; axis < dimension; ++axis)
   {
      const real_t length = axis == 0 ? 2.0 : 1.0;
      coordinates[axis] = point(axis) - length * std::floor(point(axis) / length);
   }
   Derivatives result;
   result.value = 1.0;
   for (int axis = 0; axis < dimension; ++axis)
   {
      const real_t frequency = axis == 0 ? pi : 2 * pi;
      const real_t phase = frequency * coordinates[axis];
      Derivatives factor;
      factor.value = std::pow(std::sin(0.5 * phase), 2);
      factor.gradient[axis] = 0.5 * frequency * std::sin(phase);
      factor.hessian[axis][axis] = 0.5 * frequency * frequency * std::cos(phase);
      factor.third[axis][axis][axis] = -0.5 * std::pow(frequency, 3) * std::sin(phase);
      result = Product(result, factor, dimension);
   }
   if (problem_case == MeshMMSCase::Smooth) { return result; }
   real_t distances[5] = {};
   for (int hole = 0; hole < 5; ++hole)
   {
      for (int axis = 0; axis < dimension; ++axis)
      {
         distances[hole] += std::pow(coordinates[axis] - centers[hole][axis], 2);
      }
      if (distances[hole] <= inner_squared) { return {}; }
   }
   for (int hole = 0; hole < 5; ++hole)
   {
      if (distances[hole] >= outer_squared) { continue; }
      const real_t parameter = (distances[hole] - inner_squared) / range;
      const real_t coefficients[] = {0, 0, 0, 0, 0, 126, -420, 540, -315, 70};
      real_t value = 0, first_derivative = 0, second_derivative = 0, third_derivative = 0;
      for (int degree = 9; degree >= 0; --degree)
      {
         third_derivative = third_derivative * parameter + 3 * second_derivative;
         second_derivative = second_derivative * parameter + 2 * first_derivative;
         first_derivative = first_derivative * parameter + value;
         value = value * parameter + coefficients[degree];
      }
      Derivatives factor;
      factor.value = value;
      real_t tangent[3] = {};
      for (int axis = 0; axis < dimension; ++axis)
      {
         tangent[axis] = 2 * (coordinates[axis] - centers[hole][axis]) / range;
         factor.gradient[axis] = first_derivative * tangent[axis];
      }
      for (int first = 0; first < dimension; ++first)
      {
         for (int second = 0; second < dimension; ++second)
         {
            factor.hessian[first][second] = second_derivative * tangent[first] * tangent[second] +
                                           (first == second ? 2 * first_derivative / range : 0);
            for (int third = 0; third < dimension; ++third)
            {
               factor.third[first][second][third] =
                  third_derivative * tangent[first] * tangent[second] * tangent[third] +
                  2 * second_derivative / range *
                  ((first == second ? tangent[third] : 0) +
                   (first == third ? tangent[second] : 0) +
                   (second == third ? tangent[first] : 0));
            }
         }
      }
      result = Product(result, factor, dimension);
   }
   return result;
}

real_t CurlWeight(int dimension, int component, int direction)
{
   if (dimension == 2)
   {
      return component == direction ? 0.0 : component == 0 ? 1.0 : -1.0;
   }
   const real_t weights[3][3] = {{0, 3, -2}, {-3, 0, 1}, {2, -1, 0}};
   return weights[component][direction];
}

struct MeshBoundaryInfo
{
   bool periodic = false;
   bool internal_interfaces = false;
};

struct BoundaryProjectionInfo
{
   real_t flux_before = 0.0;
   real_t correction_l2 = 0.0;
};

BoundaryProjectionInfo ProjectHoleVelocity(ParFiniteElementSpace &velocity_space,
                                          ParFiniteElementSpace &pressure_space,
                                          VectorCoefficient &exact_velocity,
                                          Array<int> &hole_marker, int quadrature_order,
                                          ParGridFunction &prescribed)
{
   const MPI_Comm comm = velocity_space.GetComm();
   ParBilinearForm mass(&velocity_space);
   auto *integrator = new RuleBilinearIntegrator<VectorMassIntegrator>(quadrature_order + 4);
   integrator->SetVDim(velocity_space.GetVDim());
   mass.AddBoundaryIntegrator(integrator, hole_marker);
   mass.Assemble();
   mass.Finalize();
   std::unique_ptr<HypreParMatrix> matrix(mass.ParallelAssemble());
   matrix->EliminateZeroRows();
   ParLinearForm load(&velocity_space);
   load.AddBoundaryIntegrator(new RuleLinearIntegrator<VectorBoundaryLFIntegrator>(
                                 quadrature_order + 4, exact_velocity), hole_marker);
   load.Assemble();
   Vector rhs(velocity_space.GetTrueVSize()), values(rhs.Size());
   load.ParallelAssemble(rhs);
   HypreDiagScale diagonal(*matrix);
   CGSolver projection(comm);
   projection.SetOperator(*matrix);
   projection.SetPreconditioner(diagonal);
   projection.SetRelTol(1e-14);
   projection.SetAbsTol(1e-15);
   projection.SetMaxIter(500);
   projection.SetPrintLevel(0);
   values = 0.0;
   projection.Mult(rhs, values);
   MFEM_VERIFY(projection.GetConverged(), "MMS boundary L2 projection did not converge");

   ParMixedBilinearForm divergence(&velocity_space, &pressure_space);
   divergence.AddDomainIntegrator(new RuleBilinearIntegrator<VectorDivergenceIntegrator>(quadrature_order));
   divergence.Assemble();
   divergence.Finalize();
   std::unique_ptr<HypreParMatrix> divergence_matrix(divergence.ParallelAssemble());
   ConstantCoefficient one(1.0);
   ParGridFunction constant(&pressure_space);
   constant.ProjectCoefficient(one);
   Vector constant_true, flux_weights(values.Size());
   constant.GetTrueDofs(constant_true);
   divergence_matrix->MultTranspose(constant_true, flux_weights);
   Array<int> boundary_dofs, selected(values.Size());
   velocity_space.GetEssentialTrueDofs(hole_marker, boundary_dofs);
   selected = 0;
   for (int dof : boundary_dofs) { selected[dof] = 1; }
   for (int dof = 0; dof < values.Size(); ++dof)
   {
      if (!selected[dof]) { flux_weights(dof) = 0.0; values(dof) = 0.0; }
   }
   BoundaryProjectionInfo info;
   info.flux_before = InnerProduct(comm, flux_weights, values);
   MFEM_VERIFY(std::isfinite(info.flux_before), "Nonfinite MMS boundary projection flux");
   if (std::abs(info.flux_before) > 1e-13)
   {
      Vector direction(values.Size());
      direction = 0.0;
      projection.Mult(flux_weights, direction);
      MFEM_VERIFY(projection.GetConverged(), "MMS flux projection did not converge");
      const real_t scale = InnerProduct(comm, flux_weights, direction);
      MFEM_VERIFY(std::isfinite(scale) && scale > 0.0, "Invalid MMS flux projection direction");
      values.Add(-info.flux_before / scale, direction);
      info.correction_l2 = std::abs(info.flux_before) / std::sqrt(scale);
   }
   prescribed.SetFromTrueDofs(values);
   return info;
}

struct BoundaryDiagnostics
{
   real_t error_l2 = 0.0;
   real_t hole_velocity_l2 = 0.0;
   real_t dof_error = 0.0;
};

BoundaryDiagnostics ComputeBoundaryDiagnostics(const ParGridFunction &velocity,
                                               MeshManufacturedSolution &exact,
                                               const ParGridFunction &prescribed,
                                               Array<int> &marker, int quadrature_order)
{
   auto &space = *velocity.ParFESpace();
   auto &mesh = *space.GetParMesh();
   const int dimension = mesh.Dimension();
   real_t local[2] = {}, global[2] = {};
   Vector point(dimension), computed(dimension), reference(dimension);
   for (int boundary = 0; boundary < mesh.GetNBE(); ++boundary)
   {
      const int attribute = mesh.GetBdrAttribute(boundary);
      if (!marker[attribute - 1]) { continue; }
      auto &transformation = *mesh.GetBdrElementTransformation(boundary);
      const auto &rule = IntRules.Get(transformation.GetGeometryType(), quadrature_order);
      for (int index = 0; index < rule.GetNPoints(); ++index)
      {
         const auto &ip = rule.IntPoint(index);
         transformation.SetIntPoint(&ip);
         transformation.Transform(ip, point);
         velocity.GetVectorValue(transformation, ip, computed);
         const real_t weight = ip.weight * transformation.Weight();
         if (attribute > 2 * dimension) { local[1] += weight * (computed * computed); }
         exact.Velocity(point, reference);
         computed -= reference;
         local[0] += weight * (computed * computed);
      }
   }
   MPI_Allreduce(local, global, 2, MPITypeMap<real_t>::mpi_type, MPI_SUM, mesh.GetComm());
   Vector difference, prescribed_true;
   velocity.GetTrueDofs(difference);
   prescribed.GetTrueDofs(prescribed_true);
   difference -= prescribed_true;
   Array<int> boundary_dofs;
   space.GetEssentialTrueDofs(marker, boundary_dofs);
   real_t local_max = 0.0, global_max = 0.0;
   for (int dof : boundary_dofs) { local_max = std::max(local_max, std::abs(difference(dof))); }
   MPI_Allreduce(&local_max, &global_max, 1, MPITypeMap<real_t>::mpi_type, MPI_MAX, mesh.GetComm());
   return {std::sqrt(global[0]), std::sqrt(global[1]), global_max};
}

MeshBoundaryInfo CheckMesh(Mesh &mesh, int dimension, MeshManufacturedSolution &exact)
{
   MFEM_VERIFY(!mesh.Nonconforming(), "MMS drivers require conforming meshes");
   const int geometry_order = detail::StokesGeometryOrder(mesh);
   MFEM_VERIFY(geometry_order > 0, "MMS drivers require uniform polynomial H1/L2 coordinate geometry");
   Vector lower, upper;
   mesh.GetBoundingBox(lower, upper, std::max(2, 2 * geometry_order));
   for (int axis = 0; axis < dimension; ++axis)
   {
      MFEM_VERIFY(std::abs(lower(axis)) < 1e-8 &&
                  std::abs(upper(axis) - (axis == 0 ? 2.0 : 1.0)) < 1e-8,
                  "MMS drivers expect the supplied 2x1[x1] box meshes");
   }
   std::array<bool, 6> outer = {};
   MeshBoundaryInfo info;
   int hole_boundaries = 0;
   for (int boundary = 0; boundary < mesh.GetNBE(); ++boundary)
   {
      const int attribute = mesh.GetBdrAttribute(boundary);
      MFEM_VERIFY(attribute >= 1 && attribute <= 2 * dimension + 5,
                  "Unexpected MMS boundary attribute");
      int first, second;
      mesh.GetFaceElements(mesh.GetBdrElementFaceIndex(boundary), &first, &second);
      if (attribute <= 2 * dimension)
      {
         if (second < 0) { outer[attribute - 1] = true; }
      }
      else
      {
         ++hole_boundaries;
         info.internal_interfaces = info.internal_interfaces || second >= 0;
      }
      auto &transformation = *mesh.GetBdrElementTransformation(boundary);
      const auto &rule = IntRules.Get(transformation.GetGeometryType(), 6 + 2 * dimension * (geometry_order - 1));
      Vector point(dimension), value(dimension);
      for (int index = 0; index < rule.GetNPoints(); ++index)
      {
         transformation.Transform(rule.IntPoint(index), point);
         exact.Velocity(point, value);
         MFEM_VERIFY(exact.GetCase() != MeshMMSCase::NoSlip || value.Normlinf() < 1e-10,
                     "MMS boundary lies outside the zero-velocity box/hole collars");
      }
   }
   const int outer_count = std::count(outer.begin(), outer.begin() + 2 * dimension, true);
   MFEM_VERIFY(outer_count == 0 || outer_count == 2 * dimension,
               "MMS drivers support fully periodic or fully nonperiodic boxes");
   info.periodic = outer_count == 0;
   MFEM_VERIFY(!info.periodic || hole_boundaries > 0,
               "Periodic MMS requires hole/interface constraints to remove the velocity nullspace");
   return info;
}

}

MeshManufacturedSolution::MeshManufacturedSolution(int dimension_, real_t viscosity_scale_,
                                                 bool variable_viscosity_, MeshMMSCase problem_case_)
   : dimension(dimension_), viscosity_scale(viscosity_scale_),
     variable_viscosity(variable_viscosity_), problem_case(problem_case_),
     viscosity([this](const Vector &point) { return Viscosity(point); }),
     viscosity_gradient(dimension, [this](const Vector &point, Vector &value) { ViscosityGradient(point, value); }),
     velocity(dimension, [this](const Vector &point, Vector &value) { Velocity(point, value); }),
     force(dimension, [this](const Vector &point, Vector &value) { Force(point, value); }),
     pressure([this](const Vector &point) { return Pressure(point); })
{
   MFEM_VERIFY((dimension == 2 || dimension == 3) &&
               std::isfinite(viscosity_scale) && viscosity_scale > 0,
               "Invalid mesh MMS dimension or viscosity");
   MFEM_VERIFY(problem_case == MeshMMSCase::NoSlip || problem_case == MeshMMSCase::Smooth,
               "Unknown mesh MMS case");
}

void MeshManufacturedSolution::Velocity(const Vector &point, Vector &value) const
{
   const auto derivatives = Potential(point, dimension, problem_case);
   value.SetSize(dimension);
   value = 0.0;
   for (int component = 0; component < dimension; ++component)
   {
      for (int direction = 0; direction < dimension; ++direction)
      {
         value(component) += CurlWeight(dimension, component, direction) * derivatives.gradient[direction];
      }
   }
}

void MeshManufacturedSolution::Gradient(const Vector &point, DenseMatrix &value) const
{
   const auto derivatives = Potential(point, dimension, problem_case);
   value.SetSize(dimension);
   value = 0.0;
   for (int component = 0; component < dimension; ++component)
   {
      for (int axis = 0; axis < dimension; ++axis)
      {
         for (int direction = 0; direction < dimension; ++direction)
         {
            value(component, axis) += CurlWeight(dimension, component, direction) *
                                      derivatives.hessian[direction][axis];
         }
      }
   }
}

real_t MeshManufacturedSolution::Viscosity(const Vector &point) const
{
   real_t product = 1.0;
   for (int axis = 0; axis < dimension; ++axis)
   {
      product *= std::cos((axis == 0 ? pi : 2 * pi) * point(axis));
   }
   return viscosity_scale * (1.0 + (variable_viscosity ? 0.25 * product : 0.0));
}

void MeshManufacturedSolution::ViscosityGradient(const Vector &point, Vector &value) const
{
   value.SetSize(dimension);
   value = 0.0;
   if (!variable_viscosity) { return; }
   for (int direction = 0; direction < dimension; ++direction)
   {
      real_t derivative = 0.25 * viscosity_scale;
      for (int axis = 0; axis < dimension; ++axis)
      {
         const real_t frequency = axis == 0 ? pi : 2 * pi;
         derivative *= axis == direction ? -frequency * std::sin(frequency * point(axis)) :
                       std::cos(frequency * point(axis));
      }
      value(direction) = derivative;
   }
}

real_t MeshManufacturedSolution::Pressure(const Vector &point) const
{
   real_t value = std::sin(pi * point(0)) * std::cos(2 * pi * point(1));
   if (dimension == 3) { value *= std::cos(2 * pi * point(2)); }
   return value;
}

void MeshManufacturedSolution::Force(const Vector &point, Vector &value) const
{
   const auto derivatives = Potential(point, dimension, problem_case);
   const real_t mu = Viscosity(point);
   Vector gradient_mu;
   ViscosityGradient(point, gradient_mu);
   value.SetSize(dimension);
   for (int component = 0; component < dimension; ++component)
   {
      real_t pressure_derivative = 1.0;
      for (int axis = 0; axis < dimension; ++axis)
      {
         const real_t frequency = axis == 0 ? pi : 2 * pi;
         const real_t phase = frequency * point(axis);
         pressure_derivative *= axis == 0 ?
                                (component == axis ? frequency * std::cos(phase) : std::sin(phase)) :
                                (component == axis ? -frequency * std::sin(phase) : std::cos(phase));
      }
      value(component) = pressure_derivative;
      for (int axis = 0; axis < dimension; ++axis)
      {
         for (int direction = 0; direction < dimension; ++direction)
         {
            value(component) -= mu * CurlWeight(dimension, component, direction) *
                                derivatives.third[direction][axis][axis];
            value(component) -= gradient_mu(axis) *
                                (CurlWeight(dimension, component, direction) * derivatives.hessian[direction][axis] +
                                 CurlWeight(dimension, axis, direction) * derivatives.hessian[direction][component]);
         }
      }
   }
}

StokesErrors ComputeMeshMMSErrors(const ParGridFunction &velocity,
                                 const ParGridFunction &pressure,
                                 MeshManufacturedSolution &exact, int integration_order)
{
   ParMesh &mesh = *velocity.ParFESpace()->GetParMesh();
   const int dimension = mesh.Dimension();
   real_t local[6] = {}, global[6] = {};
   Vector point(dimension), computed(dimension), reference(dimension);
   DenseMatrix computed_gradient(dimension), reference_gradient(dimension);
   for (int element = 0; element < mesh.GetNE(); ++element)
   {
      auto &transformation = *mesh.GetElementTransformation(element);
      const auto &rule = IntRules.Get(mesh.GetElementGeometry(element), integration_order);
      for (int index = 0; index < rule.GetNPoints(); ++index)
      {
         const auto &ip = rule.IntPoint(index);
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
         const real_t error = pressure.GetValue(element, ip) - exact.Pressure(point);
         local[2] += weight * error * error;
         local[4] += weight * error;
         local[5] += weight;
      }
   }
   MPI_Allreduce(local, global, 6, MPITypeMap<real_t>::mpi_type, MPI_SUM, mesh.GetComm());
   return {std::sqrt(global[0]), std::sqrt(global[1]),
           std::sqrt(std::max(real_t(0.0), global[2] - global[4] * global[4] / global[5])),
           std::sqrt(global[3])};
}

int RunMeshMMS(int argc, char **argv)
{
   const char *mesh_file = "", *csv_file = "";
   const char *output_directory = "stokes-mms-output";
   const char *pressure_gauge_name = "auto";
   const char *case_name = "no-slip";
   const char *solver_name = "fgmres", *preconditioner_name = "block-upper-p1-amg", *method_name = "pspg";
   int serial_refinements = 0, parallel_refinements = 0, order = 2;
   real_t viscosity_scale = 1.0;
   bool variable_viscosity = false, paraview = false;
   FlowSolverOptions options;
   IterativeOptions iterative;
   Stabilization discretization = Stabilization{PSPGOptions{}};
   PressureGauge gauge_policy = PressureGauge::Auto();
   LinearSolverType linear_solver = LinearSolverType::FGMRES;
   int integration_order = -1;
   real_t pspg_constant = 1.0, cip_gamma = 0.01, gls_tau_scale = 1.0;
   options.linear.print_level = 1;
   options.linear.print_level = 0;
   OptionsParser args(argc, argv);
   args.AddOption(&case_name, "-case", "--mms-case", "no-slip (default) or smooth (nonzero velocity on hole boundaries, without cutoffs).");
   args.AddOption(&pressure_gauge_name, "-pg", "--pressure-gauge",
                  "auto (default), mean-zero (require a pressure nullspace), or none (require no pressure nullspace).");
   args.AddOption(&mesh_file, "-m", "--mesh", "Input five-hole mesh; required. Infer 2D/3D dimension from the mesh.");
   args.AddOption(&serial_refinements, "-rs", "--refine-serial", "Uniform serial refinements before partitioning (default 0).");
   args.AddOption(&parallel_refinements, "-rp", "--refine-parallel", "Parallel uniform refinements; solve/report levels 0 through this count after serial refinement (default 0).");
   args.AddOption(&csv_file, "-csv", "--csv", "Optional rank-zero CSV report, overwritten at startup.");
   args.AddOption(&paraview, "-pv", "--paraview", "-no-pv", "--no-paraview", "Write velocity and pressure at every level as a ParaView series (off by default).");
   args.AddOption(&output_directory, "-out", "--output", "ParaView output directory; used only with --paraview (default stokes-mms-output).");
   args.AddOption(&order, "-o", "--order", "Velocity degree k>=2; pressure degree k or k-1 for Taylor-Hood.");
   args.AddOption(&method_name, "-method", "--method", "pspg, cip, gls, or taylor-hood.");
   args.AddOption(&solver_name, "-s", "--solver", "mumps, gmres, or fgmres.");
   args.AddOption(&preconditioner_name, "-pc", "--preconditioner", "block-diagonal-amg, block-diagonal-p1-amg, block-upper-p1-amg, block-lower-p1-amg; append -assembled to a P1-AMG choice for direct P1 coarse assembly.");
   args.AddOption(&iterative.p1_smoothing_steps, "-p1s", "--p1-smoothing-steps", "P1-AMG pre/post smoothing sweeps.");
   args.AddOption(&options.linear.relative_tolerance, "-rtol", "--relative-tolerance", "True-residual relative tolerance.");
   args.AddOption(&options.linear.absolute_tolerance, "-atol", "--absolute-tolerance", "True-residual absolute tolerance.");
   args.AddOption(&iterative.max_iterations, "-mi", "--max-iterations", "Total outer Krylov iteration budget.");
   args.AddOption(&iterative.restart_dimension, "-kd", "--restart", "GMRES/FGMRES restart dimension.");
   args.AddOption(&integration_order, "-q", "--quadrature-order", "Assembly quadrature order; -1 selects 2*k+2+2*dim*(g-1) for coordinate degree g.");
   args.AddOption(&pspg_constant, "-tau", "--tau-constant",
                  "Positive dimensionless PSPG multiplier C (default 1), not a raw tau.\n\t"
                  "Automatic tau_K = C*h_K^2/(mean(mu)_K*(k+1)^4), with k the velocity degree.\n\t"
                  "h_K is the local mesh size metric; mean(mu)_K is the cell-average viscosity.\n\t"
                  "Omit for the default multiplier; specifying this option overrides C only.\n\t"
                  "Used only with --method pspg.");
   args.AddOption(&cip_gamma, "-cip", "--cip-gamma",
                  "Positive dimensionless CIP multiplier gamma (default 0.01), not a raw face weight.\n\t"
                  "Automatic weight_F = gamma*h_F^3*(1/mean(mu)_K+ + 1/mean(mu)_K-)/2.\n\t"
                  "h_F = min(h_K+,h_K-), using adjacent element size metrics and cell-average viscosities.\n\t"
                  "No additional degree scaling. Omit for the default; this option overrides gamma only.\n\t"
                  "Used only with --method cip.");
   args.AddOption(&gls_tau_scale, "-gls", "--gls-tau-scale",
                  "Positive dimensionless GLS multiplier C (default 1), not a raw tau.\n\t"
                  "Automatic tau_K = C*h_K^2/(mean(mu)_K*(k+1)^4), with k the velocity degree.\n\t"
                  "h_K is the local mesh size metric; mean(mu)_K is the cell-average viscosity.\n\t"
                  "Omit for the default multiplier; specifying this option overrides C only.\n\t"
                  "Used only with --method gls.");
   args.AddOption(&iterative.amg_elasticity_interp_refine, "-air", "--amg-interp-refine", "-no-air", "--no-amg-interp-refine", "Refine elasticity AMG interpolation.");
   args.AddOption(&options.linear.print_level, "-pl", "--print-level", "Additional solver verbosity; level reports are always printed.");
   args.AddOption(&viscosity_scale, "-mu", "--mu", "Positive viscosity scale.");
   args.AddOption(&variable_viscosity, "-vm", "--variable-viscosity", "-cm", "--constant-viscosity", "Use smooth periodic variable viscosity instead of constant viscosity.");
   args.Parse();
   if (!args.Good())
   {
      if (Mpi::Root()) { args.PrintUsage(out); }
      return args.Help() ? 0 : 1;
   }
   const std::string filename(mesh_file), solver(solver_name);
   MFEM_VERIFY(!filename.empty(), "Specify an input mesh with --mesh");
   MFEM_VERIFY(!paraview || std::string(output_directory).size(), "ParaView output directory must not be empty");
   MFEM_VERIFY(serial_refinements >= 0 && parallel_refinements >= 0,
               "MMS refinement counts must be nonnegative");
   MFEM_VERIFY(order >= 2, "Invalid MMS polynomial order");
   MFEM_VERIFY(solver == "mumps" || solver == "gmres" || solver == "fgmres", "Unknown MMS solver");
   linear_solver = solver == "mumps" ? LinearSolverType::MUMPS :
                           solver == "gmres" ? LinearSolverType::GMRES : LinearSolverType::FGMRES;
   bool valid_method = false, valid_preconditioner = false;
   for (auto method : {Stabilization{PSPGOptions{pspg_constant}}, Stabilization{CIPOptions{cip_gamma}},
                       Stabilization{GLSOptions{gls_tau_scale}}, Stabilization{NoStabilization{}}})
   {
      if (std::string(method_name) == StabilizationName(method))
      {
         discretization = method;
         valid_method = true;
      }
   }
   for (auto preconditioner : {PreconditionerType::BlockDiagonalAMG,
                              PreconditionerType::BlockDiagonalP1AMG,
                              PreconditionerType::BlockUpperTriangularP1AMG,
                              PreconditionerType::BlockLowerTriangularP1AMG,
                              PreconditionerType::BlockDiagonalP1AMGAssembled,
                              PreconditionerType::BlockUpperTriangularP1AMGAssembled,
                              PreconditionerType::BlockLowerTriangularP1AMGAssembled})
   {
      if (std::string(preconditioner_name) == PreconditionerName(preconditioner))
      {
         iterative.preconditioner = preconditioner;
         valid_preconditioner = true;
      }
   }
   MFEM_VERIFY(valid_method && valid_preconditioner, "Unknown MMS method or preconditioner");
   bool valid_pressure_gauge = false;
   for (auto pressure_gauge : {PressureGauge::Auto(), PressureGauge::MeanValue(0.0),
                              PressureGauge::None()})
   {
      if (std::string(pressure_gauge_name) == PressureGaugeName(pressure_gauge))
      {
         gauge_policy = pressure_gauge;
         valid_pressure_gauge = true;
      }
   }
   MFEM_VERIFY(valid_pressure_gauge, "Unknown Stokes pressure-gauge policy");
   const std::string selected_case(case_name);
   MFEM_VERIFY(selected_case == "no-slip" || selected_case == "smooth", "Unknown mesh MMS case");
   const auto problem_case = selected_case == "smooth" ? MeshMMSCase::Smooth : MeshMMSCase::NoSlip;
   auto serial = std::make_unique<Mesh>(filename, 1, 1);
   const int dimension = serial->Dimension();
   MFEM_VERIFY((dimension == 2 || dimension == 3) && serial->SpaceDimension() == dimension,
               "MMS requires a full-dimensional 2D or 3D mesh");
   MeshManufacturedSolution exact(dimension, viscosity_scale, variable_viscosity, problem_case);
   std::ofstream csv;
   int csv_ok = 1;
   if (Mpi::Root() && std::string(csv_file).size())
   {
      csv.open(csv_file);
      csv_ok = csv.good();
      csv << "dimension,level,periodic,internal_interface_bcs,method,solver,preconditioner,elements,velocity_dofs,pressure_dofs,outer_iterations,converged,relative_residual,velocity_l2,velocity_h1,pressure_l2,divergence_l2,solve_seconds,geometry_order,integration_order,assembly_seconds,pressure_gauge,pressure_nullspace_detected,pressure_gauge_active,mms_case,boundary_l2_error,hole_velocity_l2,boundary_dof_error,boundary_flux,gauge_multiplier,projection_flux_before,projection_correction_l2,velocity_l2_rate,velocity_h1_rate,pressure_l2_rate,boundary_l2_rate"
          << ",assembly_performed,preconditioner_setup_seconds,direct_setup_seconds,linear_solve_seconds,finalization_seconds,boundary_projection_seconds,absolute_residual,order,pressure_order\n";
      csv << std::setprecision(16);
   }
   MPI_Bcast(&csv_ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
   MFEM_VERIFY(csv_ok, "Cannot open MMS CSV output");
   const MeshBoundaryInfo boundary_info = CheckMesh(*serial, dimension, exact);
   for (int refinement = 0; refinement < serial_refinements; ++refinement)
   {
      serial->UniformRefinement();
   }
   auto mesh = std::make_unique<ParMesh>(MPI_COMM_WORLD, *serial);
   serial.reset();
   std::unique_ptr<ParaViewDataCollection> visualization;
   const std::string collection_name = "stokes_mms_" + std::to_string(dimension) + "d";
   if (paraview)
   {
      visualization = std::make_unique<ParaViewDataCollection>(collection_name);
      visualization->SetPrefixPath(output_directory);
      visualization->SetHighOrderOutput(true);
      visualization->SetDataFormat(VTKFormat::BINARY);
      visualization->SetPrecision(16);
   }
   bool all_converged = true;
   std::array<real_t, 4> previous_errors = {};
   for (int level = 0; level <= parallel_refinements; ++level)
   {
      if (level > 0) { mesh->UniformRefinement(); }
      const int pressure_order = std::holds_alternative<NoStabilization>(discretization) ? order - 1 : order;
      H1_FECollection velocity_collection(order, dimension), pressure_collection(pressure_order, dimension);
      ParFiniteElementSpace velocity_space(mesh.get(), &velocity_collection, dimension, Ordering::byVDIM);
      ParFiniteElementSpace pressure_space(mesh.get(), &pressure_collection);
      Array<int> marker(mesh->bdr_attributes.Max()), hole_marker(marker.Size());
      marker = 0;
      hole_marker = 0;
      for (int attribute : mesh->bdr_attributes)
      {
         marker[attribute - 1] = !boundary_info.periodic || attribute > 2 * dimension;
         hole_marker[attribute - 1] = attribute > 2 * dimension;
      }
      ParGridFunction prescribed(&velocity_space);
      prescribed = 0.0;
      BoundaryProjectionInfo projection_info;
      double local_projection_seconds = 0.0, projection_seconds = 0.0;
      if (problem_case == MeshMMSCase::Smooth)
      {
         const double projection_started = MPI_Wtime();
         const int quadrature_order = detail::StokesIntegrationOrder(order, detail::StokesGeometryOrder(*mesh),
                                                                    dimension, integration_order);
         projection_info = ProjectHoleVelocity(velocity_space, pressure_space, exact.velocity,
                                               hole_marker, quadrature_order, prescribed);
         local_projection_seconds = MPI_Wtime() - projection_started;
      }
      MPI_Allreduce(&local_projection_seconds, &projection_seconds, 1, MPI_DOUBLE, MPI_MAX, mesh->GetComm());
      VectorGridFunctionCoefficient boundary_velocity(&prescribed);
      if (linear_solver == LinearSolverType::MUMPS) { options.linear.backend = MumpsOptions{}; }
      else { iterative.method = linear_solver; options.linear.backend = iterative; }
      FlowProblem definition(velocity_space, pressure_space, exact.viscosity);
      definition.SetViscosityGradient(exact.viscosity_gradient);
      definition.SetIntegrationOrder(integration_order);
      definition.SetStabilization(discretization);
      definition.SetPressureGauge(gauge_policy);
      BoundaryConditions boundary;
      ParGridFunction velocity(&velocity_space), pressure(&pressure_space);
      for (int index = 0; index < mesh->bdr_attributes.Size(); ++index)
      {
         const int attribute = mesh->bdr_attributes[index];
         if (!boundary_info.periodic || attribute > 2 * dimension)
         {
            if (problem_case == MeshMMSCase::Smooth) { boundary.SetVelocity(attribute, boundary_velocity); }
            else
            {
               for (int component = 0; component < dimension; ++component)
               { boundary.SetVelocity(attribute, static_cast<Direction>(component), 0.0); }
            }
         }
      }
      definition.SetBoundaryConditions(boundary);
      for (int attribute : mesh->attributes) { definition.SetBodyForce(attribute, exact.force); }
      FlowSolver problem(definition, options);
      MPI_Barrier(MPI_COMM_WORLD);
      const double started = MPI_Wtime();
      const auto info = problem.Solve(velocity, pressure);
      const double elapsed = MPI_Wtime() - started;
      double seconds = 0.0;
      MPI_Allreduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      const auto errors = ComputeMeshMMSErrors(velocity, pressure, exact, info.integration_order + 4);
      const auto boundary_errors = ComputeBoundaryDiagnostics(velocity, exact, prescribed,
                                                              marker, info.integration_order + 4);
      const std::array<real_t, 4> current_errors = {errors.velocity_l2, errors.velocity_h1,
                                                  errors.pressure_l2, boundary_errors.error_l2};
      std::array<real_t, 4> rates = {};
      for (int index = 0; index < 4; ++index)
      {
         if (level > 0 && previous_errors[index] > 0.0 && current_errors[index] > 0.0)
         {
            rates[index] = std::log2(previous_errors[index]) - std::log2(current_errors[index]);
         }
      }
      previous_errors = current_errors;
      const auto elements = mesh->GetGlobalNE();
      const auto velocity_dofs = velocity_space.GlobalTrueVSize();
      const auto pressure_dofs = pressure_space.GlobalTrueVSize();
      all_converged = all_converged && info.converged;
      if (Mpi::Root())
      {
         out << std::setprecision(12) << "MMS_LEVEL dim=" << dimension << " level=" << level
             << " periodic=" << boundary_info.periodic << " internal_interface_bcs=" << boundary_info.internal_interfaces
             << " mms_case=" << MeshMMSCaseName(problem_case)
             << " method=" << method_name << " solver=" << solver_name << " preconditioner=" << preconditioner_name
             << " pressure_gauge=" << PressureGaugeName(gauge_policy)
             << " pressure_nullspace_detected=" << info.pressure_nullspace_detected
             << " pressure_gauge_active=" << info.gauge_multiplier.has_value()
             << " order=" << order << " pressure_order=" << pressure_order
             << " geometry_order=" << info.geometry_order << " integration_order=" << info.integration_order
             << " elements=" << elements << " velocity_dofs=" << velocity_dofs << " pressure_dofs=" << pressure_dofs
             << " outer_iterations=" << info.linear_iterations << " converged=" << info.converged
             << " residual=" << info.relative_residual << " velocity_l2=" << errors.velocity_l2
             << " velocity_h1=" << errors.velocity_h1 << " pressure_l2=" << errors.pressure_l2
             << " divergence_l2=" << errors.divergence_l2 << " solve_seconds=" << seconds
             << " boundary_l2_error=" << boundary_errors.error_l2
             << " hole_velocity_l2=" << boundary_errors.hole_velocity_l2
             << " boundary_dof_error=" << boundary_errors.dof_error
             << " boundary_flux=" << info.boundary_flux << " gauge_multiplier=" << info.gauge_multiplier.value_or(0.0)
             << " projection_flux_before=" << projection_info.flux_before
             << " projection_correction_l2=" << projection_info.correction_l2
             << " velocity_l2_rate=" << rates[0] << " velocity_h1_rate=" << rates[1]
             << " pressure_l2_rate=" << rates[2] << " boundary_l2_rate=" << rates[3]
             << " assembly_seconds=" << info.assembly_seconds
             << " assembly_performed=" << info.assembly_performed
             << " preconditioner_setup_seconds=" << (linear_solver != LinearSolverType::MUMPS ? info.setup_seconds : 0.0)
             << " direct_setup_seconds=" << (linear_solver == LinearSolverType::MUMPS ? info.setup_seconds : 0.0)
             << " linear_solve_seconds=" << info.solve_seconds
             << " finalization_seconds=" << info.finalization_seconds
             << " boundary_projection_seconds=" << projection_seconds
             << " absolute_residual=" << info.absolute_residual << std::endl;
         if (csv.is_open())
         {
            csv << dimension << ',' << level << ',' << boundary_info.periodic << ',' << boundary_info.internal_interfaces
                << ',' << method_name << ',' << solver_name << ',' << preconditioner_name << ',' << elements
                << ',' << velocity_dofs << ',' << pressure_dofs << ',' << info.linear_iterations << ',' << info.converged
                << ',' << info.relative_residual << ',' << errors.velocity_l2 << ',' << errors.velocity_h1
                << ',' << errors.pressure_l2 << ',' << errors.divergence_l2 << ',' << seconds
                << ',' << info.geometry_order << ',' << info.integration_order
                << ',' << info.assembly_seconds << ',' << PressureGaugeName(gauge_policy)
                << ',' << info.pressure_nullspace_detected << ',' << info.gauge_multiplier.has_value()
                << ',' << MeshMMSCaseName(problem_case) << ',' << boundary_errors.error_l2
                << ',' << boundary_errors.hole_velocity_l2 << ',' << boundary_errors.dof_error
                << ',' << info.boundary_flux << ',' << info.gauge_multiplier.value_or(0.0)
                << ',' << projection_info.flux_before << ',' << projection_info.correction_l2
                << ',' << rates[0] << ',' << rates[1] << ',' << rates[2] << ',' << rates[3]
                << ',' << info.assembly_performed << ',' << (linear_solver != LinearSolverType::MUMPS ? info.setup_seconds : 0.0)
                << ',' << (linear_solver == LinearSolverType::MUMPS ? info.setup_seconds : 0.0) << ',' << info.solve_seconds
                << ',' << info.finalization_seconds << ',' << projection_seconds
                << ',' << info.absolute_residual << ',' << order << ',' << pressure_order << std::endl;
            csv_ok = csv.good();
         }
      }
      MPI_Bcast(&csv_ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
      MFEM_VERIFY(csv_ok, "Failed writing MMS CSV output");
      if (visualization)
      {
         visualization->SetMesh(mesh.get());
         visualization->SetCycle(level);
         visualization->SetTime(real_t(level));
         visualization->SetLevelsOfDetail(std::max(order, info.geometry_order));
         visualization->RegisterField("velocity", const_cast<ParGridFunction *>(&velocity));
         visualization->RegisterField("pressure", const_cast<ParGridFunction *>(&pressure));
         visualization->Save();
         visualization->DeregisterField("velocity");
         visualization->DeregisterField("pressure");
         const int local_error = visualization->Error();
         int global_error = DataCollection::No_Error;
         MPI_Allreduce(&local_error, &global_error, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
         MFEM_VERIFY(global_error == DataCollection::No_Error, "Failed writing MMS ParaView output");
      }
   }
   if (visualization && Mpi::Root())
   {
      out << "ParaView collection: " << visualization->GetPrefixPath()
          << collection_name << '/' << collection_name << ".pvd" << std::endl;
   }
   return all_converged ? 0 : 2;
}

}
