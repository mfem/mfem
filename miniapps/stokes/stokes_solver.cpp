#include "stokes_solver.hpp"
#include "stokes_integrators.hpp"
#include "stokes_preconditioners.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <utility>
#include <vector>

namespace mfem::stokes
{

const char *StokesPreconditionerName(StokesPreconditionerType preconditioner)
{
   switch (preconditioner)
   {
      case StokesPreconditionerType::BlockDiagonalAMG: return "block-diagonal-amg";
      case StokesPreconditionerType::BlockDiagonalP1AMG: return "block-diagonal-p1-amg";
      case StokesPreconditionerType::BlockUpperTriangularP1AMG: return "block-upper-p1-amg";
      case StokesPreconditionerType::BlockLowerTriangularP1AMG: return "block-lower-p1-amg";
      case StokesPreconditionerType::BlockDiagonalP1AMGAssembled: return "block-diagonal-p1-amg-assembled";
      case StokesPreconditionerType::BlockUpperTriangularP1AMGAssembled: return "block-upper-p1-amg-assembled";
      case StokesPreconditionerType::BlockLowerTriangularP1AMGAssembled: return "block-lower-p1-amg-assembled";
   }
   return "unknown";
}

const char *StokesStabilizationMethodName(StokesStabilizationMethod method)
{
   switch (method)
   {
      case StokesStabilizationMethod::PSPG: return "pspg";
      case StokesStabilizationMethod::CIP: return "cip";
      case StokesStabilizationMethod::GLS: return "gls";
      case StokesStabilizationMethod::TaylorHood: return "taylor-hood";
   }
   return "unknown";
}

const char *StokesPressureGaugeName(StokesPressureGauge pressure_gauge)
{
   switch (pressure_gauge)
   {
      case StokesPressureGauge::Auto: return "auto";
      case StokesPressureGauge::MeanZero: return "mean-zero";
      case StokesPressureGauge::None: return "none";
   }
   return "unknown";
}

namespace
{

Vector ZeroVector(int dimension)
{
   Vector result(dimension);
   result = 0.0;
   return result;
}

class ElementTau : public Coefficient
{
public:
   Vector values;
   real_t Eval(ElementTransformation &transformation,
               const IntegrationPoint &) override
   {
      return values(transformation.ElementNo);
   }
};

class InverseViscosity : public Coefficient
{
   Coefficient &viscosity;
public:
   explicit InverseViscosity(Coefficient &viscosity_) : viscosity(viscosity_) { }
   real_t Eval(ElementTransformation &transformation,
               const IntegrationPoint &point) override
   {
      return 1.0 / viscosity.Eval(transformation, point);
   }
};

class DiagonalInverse : public Solver
{
   const Vector &diagonal;
public:
   explicit DiagonalInverse(const Vector &diagonal_)
      : Solver(diagonal_.Size()), diagonal(diagonal_) { }
   void SetOperator(const Operator &) override
   {
      MFEM_ABORT("DiagonalInverse has a fixed diagonal");
   }
   void Mult(const Vector &input, Vector &output) const override
   {
      output.SetSize(height);
      for (int entry = 0; entry < height; ++entry)
      {
         output(entry) = input(entry) / diagonal(entry);
      }
   }
};

class RightPreconditionedOperator : public Operator
{
   const Operator &matrix, &preconditioner;
   mutable Vector work;
public:
   RightPreconditionedOperator(const Operator &matrix_, const Operator &preconditioner_)
      : Operator(matrix_.Height()), matrix(matrix_), preconditioner(preconditioner_),
        work(matrix_.Width()) { }
   void Mult(const Vector &input, Vector &output) const override
   {
      preconditioner.Mult(input, work);
      matrix.Mult(work, output);
   }
};

}

class StokesSolver::Impl
{
public:
   struct BoundaryValue
   {
      std::unique_ptr<Coefficient> owned;
      Coefficient *coefficient = nullptr;
   };

   ParFiniteElementSpace &velocity_space;
   ParFiniteElementSpace &pressure_space;
   ParMesh &mesh;
   MPI_Comm comm;
   StokesSolverOptions options;
   int dimension, rank, ranks, order, pressure_order, geometry_order, quadrature_order, max_boundary_attribute;
   long mesh_sequence, velocity_sequence, pressure_sequence;
   std::unique_ptr<Coefficient> owned_viscosity;
   std::unique_ptr<VectorCoefficient> owned_gradient, owned_force;
   Coefficient *viscosity;
   VectorCoefficient *viscosity_gradient, *force;
   std::map<std::pair<int, int>, BoundaryValue> boundary_values;
   ParGridFunction velocity, pressure;
   bool assembled = false, has_solution = false, gauge = false;
   bool pressure_nullspace_detected = false;
   double assembly_seconds = 0.0;
   ElementTau tau;
   Vector element_sizes, element_viscosities;
   Array<int> essential, offsets;
   Array<HYPRE_BigInt> gauge_partition;
   Vector lifting, moments, pressure_constant, pressure_diagonal, gauge_diagonal;
   Vector rhs, solution;
   real_t volume = 0.0;
   std::unique_ptr<HypreParMatrix> velocity_matrix, divergence, continuity;
   std::unique_ptr<HypreParMatrix> momentum_pressure, stabilization;
   std::unique_ptr<HypreParMatrix> pressure_matrix;
   std::unique_ptr<HypreParMatrix> gauge_column, gauge_row, system;

   Impl(ParFiniteElementSpace &velocity_space_,
        ParFiniteElementSpace &pressure_space_,
        const StokesSolverOptions &options_, Coefficient *viscosity_,
        VectorCoefficient *gradient_, real_t constant_viscosity)
      : velocity_space(velocity_space_), pressure_space(pressure_space_),
        mesh(*velocity_space.GetParMesh()), comm(velocity_space.GetComm()),
        options(options_), dimension(mesh.Dimension()),
        velocity(&velocity_space), pressure(&pressure_space)
   {
      MPI_Comm_rank(comm, &rank);
      MPI_Comm_size(comm, &ranks);
      mesh_sequence = mesh.GetSequence();
      velocity_sequence = velocity_space.GetSequence();
      pressure_sequence = pressure_space.GetSequence();
      Check(&mesh == pressure_space.GetParMesh(), "Spaces must share a ParMesh");
      Check((dimension == 2 || dimension == 3) && mesh.SpaceDimension() == dimension,
            "Stokes requires a 2D or 3D volume mesh");
      Check(!Device::Allows(Backend::DEVICE_MASK), "Stokes currently requires CPU execution");
      Check(!mesh.Nonconforming(), "Nonconforming Stokes meshes are not supported");
      Check(velocity_space.GetVDim() == dimension && pressure_space.GetVDim() == 1,
            "Invalid velocity or pressure vector dimension");
      Check(velocity_space.GetOrdering() == Ordering::byVDIM,
            "Elasticity AMG requires velocity Ordering::byVDIM");
      Check(dynamic_cast<const H1_FECollection*>(velocity_space.FEColl()) &&
            dynamic_cast<const H1_FECollection*>(pressure_space.FEColl()),
            "Stokes requires continuous H1 spaces");
      Check(options.method == StokesStabilizationMethod::PSPG || options.method == StokesStabilizationMethod::CIP ||
            options.method == StokesStabilizationMethod::GLS || options.method == StokesStabilizationMethod::TaylorHood,
            "Unknown Stokes discretization method");
      const int velocity_basis = static_cast<const H1_FECollection*>(velocity_space.FEColl())->GetBasisType();
      Check((options.method != StokesStabilizationMethod::PSPG &&
             options.method != StokesStabilizationMethod::GLS) ||
            (velocity_basis != BasisType::Positive && velocity_basis != BasisType::Serendipity),
            "PSPG/GLS require a velocity basis with implemented Hessians; positive/Bernstein and serendipity bases are unsupported");
      int local_order = velocity_space.GetMaxElementOrder();
      MPI_Allreduce(&local_order, &order, 1, MPI_INT, MPI_MAX, comm);
      pressure_order = options.method == StokesStabilizationMethod::TaylorHood ? order - 1 : order;
      bool valid_elements = order >= 2;
      for (int element = 0; element < mesh.GetNE(); ++element)
      {
         const auto geometry = mesh.GetElementGeometry(element);
         valid_elements = valid_elements &&
                          velocity_space.GetElementOrder(element) == order &&
                          pressure_space.GetElementOrder(element) == pressure_order &&
                          (geometry == Geometry::TRIANGLE ||
                           geometry == Geometry::SQUARE ||
                           geometry == Geometry::TETRAHEDRON ||
                           geometry == Geometry::CUBE);
      }
      Check(valid_elements, "Require uniform H1 k>=2 on tri/quad/tet/hex with pressure order k-1 for Taylor-Hood, k otherwise");
      const int local_geometry_order = detail::StokesGeometryOrder(mesh);
      Check(local_geometry_order > 0, "Stokes requires uniform polynomial H1/L2 coordinate geometry on tri/quad/tet/hex");
      int minimum_geometry_order;
      MPI_Allreduce(&local_geometry_order, &geometry_order, 1, MPI_INT, MPI_MAX, comm);
      MPI_Allreduce(&local_geometry_order, &minimum_geometry_order, 1, MPI_INT, MPI_MIN, comm);
      Check(minimum_geometry_order == geometry_order, "Variable coordinate order is not supported");
      Check((options.method != StokesStabilizationMethod::PSPG ||
             (std::isfinite(options.pspg_constant) && options.pspg_constant > 0.0)) &&
            (options.method != StokesStabilizationMethod::CIP ||
             (std::isfinite(options.cip_gamma) && options.cip_gamma > 0.0)) &&
            (options.method != StokesStabilizationMethod::GLS ||
             (std::isfinite(options.gls_tau_scale) && options.gls_tau_scale > 0.0)) &&
            std::isfinite(options.relative_tolerance) && options.relative_tolerance > 0.0 &&
            std::isfinite(options.absolute_tolerance) && options.absolute_tolerance >= 0.0 &&
            options.max_iterations > 0 && options.restart_dimension > 0 &&
            (options.integration_order == -1 || options.integration_order > 0),
            "Invalid Stokes solver options");
      Check(options.linear_solver == LinearSolverType::MUMPS ||
            options.linear_solver == LinearSolverType::GMRES ||
            options.linear_solver == LinearSolverType::FGMRES,
            "Unknown Stokes linear solver");
      Check(options.pressure_gauge == StokesPressureGauge::Auto ||
            options.pressure_gauge == StokesPressureGauge::MeanZero ||
            options.pressure_gauge == StokesPressureGauge::None,
            "Unknown Stokes pressure-gauge policy");
      Check(options.preconditioner == StokesPreconditionerType::BlockDiagonalAMG ||
            options.preconditioner == StokesPreconditionerType::BlockDiagonalP1AMG ||
            options.preconditioner == StokesPreconditionerType::BlockUpperTriangularP1AMG ||
            options.preconditioner == StokesPreconditionerType::BlockLowerTriangularP1AMG ||
            options.preconditioner == StokesPreconditionerType::BlockDiagonalP1AMGAssembled ||
            options.preconditioner == StokesPreconditionerType::BlockUpperTriangularP1AMGAssembled ||
            options.preconditioner == StokesPreconditionerType::BlockLowerTriangularP1AMGAssembled,
            "Unknown Stokes preconditioner");
      Check(options.linear_solver == LinearSolverType::MUMPS ||
            options.preconditioner == StokesPreconditionerType::BlockDiagonalAMG ||
            options.p1_smoothing_steps > 0,
            "P1-AMG smoothing steps must be positive");
#ifndef MFEM_USE_MUMPS
      Check(options.linear_solver != LinearSolverType::MUMPS,
            "MUMPS was requested but MFEM was built without MUMPS");
#endif
      quadrature_order = detail::StokesIntegrationOrder(order, geometry_order, dimension,
                                                        options.integration_order);
      int local_max = mesh.bdr_attributes.Size() ? mesh.bdr_attributes.Max() : 0;
      MPI_Allreduce(&local_max, &max_boundary_attribute, 1, MPI_INT, MPI_MAX, comm);
      if (viscosity_)
      {
         viscosity = viscosity_;
         viscosity_gradient = gradient_;
      }
      else
      {
         Check(std::isfinite(constant_viscosity) && constant_viscosity > 0.0,
               "Viscosity must be positive and finite");
         owned_viscosity = std::make_unique<ConstantCoefficient>(constant_viscosity);
         owned_gradient = std::make_unique<VectorConstantCoefficient>(ZeroVector(dimension));
         viscosity = owned_viscosity.get();
         viscosity_gradient = owned_gradient.get();
      }
      Check(viscosity_gradient && viscosity_gradient->GetVDim() == dimension,
            "Viscosity gradient dimension mismatch");
      owned_force = std::make_unique<VectorConstantCoefficient>(ZeroVector(dimension));
      force = owned_force.get();
      velocity = 0.0;
      pressure = 0.0;
   }

   void Check(bool valid, const char *message) const
   {
      int local = valid ? 1 : 0, global = 0;
      MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_MIN, comm);
      MFEM_VERIFY(global, message);
   }

   real_t Dot(const Vector &left, const Vector &right) const
   {
      return InnerProduct(comm, left, right);
   }

   real_t Norm(const Vector &vector) const
   {
      return std::sqrt(Dot(vector, vector));
   }

   void Invalidate()
   {
      assembled = false;
      has_solution = false;
      assembly_seconds = 0.0;
   }

   void CheckSequences() const
   {
      Check(mesh.GetSequence() == mesh_sequence &&
            velocity_space.GetSequence() == velocity_sequence &&
            pressure_space.GetSequence() == pressure_sequence,
            "Mesh/spaces changed; recreate the Stokes solver");
   }

   void CheckBoundary(int attribute, int direction) const
   {
      Check(direction >= -1 && direction < dimension,
            "Invalid velocity BC direction (0=x, 1=y, 2=z, -1=all)");
      int local_found = mesh.bdr_attributes.Find(attribute) >= 0 ? 1 : 0;
      int global_found = 0;
      MPI_Allreduce(&local_found, &global_found, 1, MPI_INT, MPI_MAX, comm);
      Check(attribute > 0 && global_found, "Unknown boundary attribute");
   }

   void SetBoundary(int attribute, int component, Coefficient *coefficient,
                    std::unique_ptr<Coefficient> owned = {})
   {
      auto &entry = boundary_values[{attribute, component}];
      entry.owned = std::move(owned);
      entry.coefficient = coefficient;
      Invalidate();
   }

   void BuildBoundaryLifting()
   {
      const int local_size = velocity_space.GetTrueVSize();
      lifting.SetSize(local_size);
      lifting = 0.0;
      Array<int> prescribed(local_size);
      prescribed = 0;
      essential.SetSize(0);
      std::unique_ptr<H1_FECollection> nodal_collection;
      std::unique_ptr<ParFiniteElementSpace> nodal_space;
      std::unique_ptr<PRefinementTransferOperator> boundary_transfer;
      const int basis = static_cast<const H1_FECollection*>(velocity_space.FEColl())->GetBasisType();
      if (basis == BasisType::Positive)
      {
         nodal_collection = std::make_unique<H1_FECollection>(order, dimension);
         nodal_space = std::make_unique<ParFiniteElementSpace>(&mesh, nodal_collection.get(),
                                                              dimension, velocity_space.GetOrdering());
         boundary_transfer = std::make_unique<PRefinementTransferOperator>(*nodal_space, velocity_space);
      }
      bool consistent = true;
      for (auto &item : boundary_values)
      {
         Array<int> marker(max_boundary_attribute);
         marker = 0;
         marker[item.first.first - 1] = 1;
         const int component = item.first.second;
         std::vector<Coefficient*> coefficients(dimension, nullptr);
         coefficients[component] = item.second.coefficient;
         ParGridFunction projected(nodal_space ? nodal_space.get() : &velocity_space);
         projected = 0.0;
         projected.ProjectBdrCoefficient(coefficients.data(), marker);
         Vector values;
         if (boundary_transfer)
         {
            Vector nodal_values;
            projected.GetTrueDofs(nodal_values);
            values.SetSize(local_size);
            boundary_transfer->GetTrueTransferOperator()->Mult(nodal_values, values);
         }
         else { projected.GetTrueDofs(values); }
         Array<int> selected;
         velocity_space.GetEssentialTrueDofs(marker, selected, component);
         for (int index = 0; index < selected.Size(); ++index)
         {
            const int dof = selected[index];
            const real_t value = values(dof);
            consistent = consistent && std::isfinite(value);
            if (prescribed[dof])
            {
               consistent = consistent && std::abs(lifting(dof) - value) <=
                            1e-12 * std::max({real_t(1.0), std::abs(value),
                                             std::abs(lifting(dof))});
            }
            else
            {
               prescribed[dof] = 1;
               essential.Append(dof);
               lifting(dof) = value;
            }
         }
      }
      Check(consistent, "Conflicting or nonfinite velocity boundary values");
      essential.Sort();
      CheckRigidModes();
   }

   void CheckRigidModes()
   {
      Vector lower, upper;
      mesh.GetBoundingBox(lower, upper);
      Vector global_lower(dimension), global_upper(dimension);
      MPI_Allreduce(lower.GetData(), global_lower.GetData(), dimension,
                    MPITypeMap<real_t>::mpi_type, MPI_MIN, comm);
      MPI_Allreduce(upper.GetData(), global_upper.GetData(), dimension,
                    MPITypeMap<real_t>::mpi_type, MPI_MAX, comm);
      Vector center(global_lower);
      center += global_upper;
      center *= 0.5;
      global_upper -= global_lower;
      const real_t length = global_upper.Norml2();
      Check(length > 0.0, "Degenerate Stokes domain");
      std::vector<Vector> basis;
      const int mode_count = dimension * (dimension + 1) / 2;
      for (int mode = 0; mode < mode_count; ++mode)
      {
         VectorFunctionCoefficient rigid(dimension, [&, mode](const Vector &point,
                                                              Vector &value)
         {
            value.SetSize(dimension);
            value = 0.0;
            if (mode < dimension) { value(mode) = 1.0; }
            else
            {
               int rotation = dimension;
               for (int first = 0; first < dimension; ++first)
               {
                  for (int second = first + 1; second < dimension; ++second)
                  {
                     if (mode == rotation)
                     {
                        value(first) = -(point(second) - center(second)) / length;
                        value(second) = (point(first) - center(first)) / length;
                     }
                     ++rotation;
                  }
               }
            }
         });
         ParGridFunction field(&velocity_space);
         field.ProjectCoefficient(rigid);
         Vector true_values;
         field.GetTrueDofs(true_values);
         Vector restricted(essential.Size());
         for (int index = 0; index < essential.Size(); ++index)
         {
            restricted(index) = true_values(essential[index]);
         }
         const real_t original_norm = Norm(restricted);
         for (int pass = 0; pass < 2; ++pass)
         {
            for (const auto &previous : basis)
            {
               restricted.Add(-Dot(previous, restricted), previous);
            }
         }
         const real_t norm = Norm(restricted);
         Check(norm > 1e-10 * std::max(real_t(1.0), original_norm),
               "Velocity BCs leave an unconstrained rigid translation/rotation");
         restricted /= norm;
         basis.push_back(restricted);
      }
   }

   void ValidateGeometry()
   {
      bool valid = true;
      for (int element = 0; element < mesh.GetNE(); ++element)
      {
         auto &transformation = *mesh.GetElementTransformation(element);
         const auto *isoparametric = dynamic_cast<const IsoparametricTransformation *>(&transformation);
         if (!isoparametric) { valid = false; continue; }
         const auto geometry = mesh.GetElementGeometry(element);
         for (const IntegrationRule *rule : {&IntRules.Get(geometry, quadrature_order),
                                             Geometries.GetVertices(geometry),
                                             &isoparametric->GetFE()->GetNodes()})
         {
            for (int point = 0; point < rule->GetNPoints(); ++point)
            {
               transformation.SetIntPoint(&rule->IntPoint(point));
               const auto &jacobian = transformation.Jacobian();
               const real_t determinant = jacobian.Det();
               valid = valid && std::isfinite(determinant) && determinant > 0.0;
            }
         }
      }
      Check(valid, "Invalid Stokes geometry: nonpositive/nonfinite sampled signed Jacobian");
   }

   void BuildTau()
   {
      tau.values.SetSize(mesh.GetNE());
      element_sizes.SetSize(mesh.GetNE());
      element_viscosities.SetSize(mesh.GetNE());
      const bool residual_method = options.method == StokesStabilizationMethod::PSPG || options.method == StokesStabilizationMethod::GLS;
      const real_t scale = options.method == StokesStabilizationMethod::GLS ? options.gls_tau_scale :
                           options.method == StokesStabilizationMethod::PSPG ? options.pspg_constant : 0.0;
      bool valid = true;
      real_t local_min = std::numeric_limits<real_t>::max(), local_max = 0.0;
      real_t local_mu_min = std::numeric_limits<real_t>::max(), local_mu_max = 0.0;
      Vector gradient(dimension), force_value(dimension);
      for (int element = 0; element < mesh.GetNE(); ++element)
      {
         const real_t size = mesh.GetElementSize(element, 1);
         ElementTransformation &transformation = *mesh.GetElementTransformation(element);
         const IntegrationRule &rule = IntRules.Get(mesh.GetElementGeometry(element),
                                                   quadrature_order);
         real_t integral = 0.0, measure = 0.0;
         for (int point = 0; point < rule.GetNPoints(); ++point)
         {
            const IntegrationPoint &ip = rule.IntPoint(point);
            transformation.SetIntPoint(&ip);
            const real_t weight = ip.weight * transformation.Weight();
            const real_t mu = viscosity->Eval(transformation, ip);
            local_mu_min = std::min(local_mu_min, mu);
            local_mu_max = std::max(local_mu_max, mu);
            viscosity_gradient->Eval(gradient, transformation, ip);
            force->Eval(force_value, transformation, ip);
            valid = valid && std::isfinite(mu) && mu > 0.0 &&
                    std::isfinite(weight) && transformation.Weight() > 0.0;
            for (int axis = 0; axis < dimension; ++axis)
            {
               valid = valid && std::isfinite(gradient(axis)) &&
                       std::isfinite(force_value(axis));
            }
            integral += weight * mu;
            measure += weight;
         }
         const real_t value = scale * size * size * measure /
                              (integral * std::pow(real_t(order + 1), 4));
         valid = valid && std::isfinite(value) && (!residual_method || value > 0.0) &&
                 std::isfinite(size) && size > 0.0 && measure > 0.0;
         tau.values(element) = value;
         element_sizes(element) = size;
         element_viscosities(element) = integral / measure;
         local_min = std::min(local_min, value);
         local_max = std::max(local_max, value);
      }
      Check(valid, "Nonpositive/nonfinite viscosity, geometry, force or PSPG coefficient");
      real_t global_min, global_max;
      real_t global_mu_min, global_mu_max;
      MPI_Allreduce(&local_min, &global_min, 1, MPITypeMap<real_t>::mpi_type, MPI_MIN, comm);
      MPI_Allreduce(&local_max, &global_max, 1, MPITypeMap<real_t>::mpi_type, MPI_MAX, comm);
      MPI_Allreduce(&local_mu_min, &global_mu_min, 1, MPITypeMap<real_t>::mpi_type, MPI_MIN, comm);
      MPI_Allreduce(&local_mu_max, &global_mu_max, 1, MPITypeMap<real_t>::mpi_type, MPI_MAX, comm);
      if (rank == 0 && options.print_level > 0)
      {
         out << "Stokes: method=" << StokesStabilizationMethodName(options.method)
             << " dim=" << dimension << " order=" << order << " pressure_order=" << pressure_order
             << " geometry_order=" << geometry_order
             << " quadrature=" << quadrature_order
             << " mu=[" << global_mu_min << "," << global_mu_max << "]"
             << " tau_scale=" << scale
             << " cip_gamma=" << (options.method == StokesStabilizationMethod::CIP ? options.cip_gamma : 0.0)
             << " tau=[" << global_min
             << "," << global_max << "]\n";
      }
   }

   std::unique_ptr<HypreParMatrix> AssembleVelocityBlock(ParFiniteElementSpace &space)
   {
      ParBilinearForm form(&space);
      form.AddDomainIntegrator(new RuleBilinearIntegrator<ElasticityIntegrator>(
                                  quadrature_order, *viscosity, 0.0, 1.0));
      if (options.method == StokesStabilizationMethod::GLS)
      {
         form.AddDomainIntegrator(new GLSVelocityIntegrator(
                                     *viscosity, *viscosity_gradient, tau, quadrature_order));
      }
      form.Assemble(0);
      form.Finalize(0);
      return std::unique_ptr<HypreParMatrix>(form.ParallelAssemble());
   }

   std::unique_ptr<HypreParMatrix> AssemblePressureBlock(ParFiniteElementSpace &space,
                                                       bool with_mass)
   {
      L2_FECollection cell_collection(0, dimension);
      ParFiniteElementSpace cell_space(&mesh, &cell_collection);
      ParGridFunction cell_sizes(&cell_space), cell_viscosities(&cell_space);
      if (options.method == StokesStabilizationMethod::CIP)
      {
         Array<int> dofs;
         for (int element = 0; element < mesh.GetNE(); ++element)
         {
            cell_space.GetElementDofs(element, dofs);
            cell_sizes(dofs[0]) = element_sizes(element);
            cell_viscosities(dofs[0]) = element_viscosities(element);
         }
         cell_sizes.ExchangeFaceNbrData();
         cell_viscosities.ExchangeFaceNbrData();
      }
      GridFunctionCoefficient size_coefficient(&cell_sizes), mu_coefficient(&cell_viscosities);
      InverseViscosity inverse_viscosity(*viscosity);
      ParBilinearForm form(&space);
      if (with_mass)
      {
         form.AddDomainIntegrator(new RuleBilinearIntegrator<MassIntegrator>(
                                     quadrature_order, inverse_viscosity));
      }
      if (options.method == StokesStabilizationMethod::PSPG ||
          options.method == StokesStabilizationMethod::GLS)
      {
         form.AddDomainIntegrator(new RuleBilinearIntegrator<DiffusionIntegrator>(quadrature_order, tau));
      }
      else if (options.method == StokesStabilizationMethod::CIP)
      {
         form.AddInteriorFaceIntegrator(new PressureCIPIntegrator(
                                           size_coefficient, mu_coefficient, options.cip_gamma, quadrature_order));
      }
      form.Assemble(0);
      form.Finalize(0);
      return std::unique_ptr<HypreParMatrix>(form.ParallelAssemble());
   }

   void Assemble()
   {
      const double assembly_start = MPI_Wtime();
      CheckSequences();
      Invalidate();
      ValidateGeometry();
      BuildBoundaryLifting();
      if (options.method == StokesStabilizationMethod::CIP && mesh.GetGlobalNE() <= (1 << dimension))
      {
         int local_coarse_tensor = 1, global_coarse_tensor = 0;
         for (int element = 0; element < mesh.GetNE(); ++element)
         {
            const auto geometry = mesh.GetElementGeometry(element);
            if (geometry != Geometry::SQUARE && geometry != Geometry::CUBE) { local_coarse_tensor = 0; }
         }
         Array<int> boundary_marker(max_boundary_attribute), boundary_dofs;
         boundary_marker = 1;
         velocity_space.GetEssentialTrueDofs(boundary_marker, boundary_dofs);
         for (int index = 0; index < boundary_dofs.Size(); ++index)
         {
            if (essential.Find(boundary_dofs[index]) < 0) { local_coarse_tensor = 0; }
         }
         MPI_Allreduce(&local_coarse_tensor, &global_coarse_tensor, 1, MPI_INT, MPI_MIN, comm);
         Check(!global_coarse_tensor,
               "CIP requires refinement of fully Dirichlet tensor meshes with at most 2^dim elements; the gradient-jump penalty can leave extra pressure null modes");
      }
      BuildTau();
      const bool gls = options.method == StokesStabilizationMethod::GLS;
      const bool residual_method = gls || options.method == StokesStabilizationMethod::PSPG;
      velocity_matrix = AssembleVelocityBlock(velocity_space);

      ParMixedBilinearForm divergence_form(&velocity_space, &pressure_space);
      divergence_form.AddDomainIntegrator(new RuleBilinearIntegrator<VectorDivergenceIntegrator>(
                                            quadrature_order));
      divergence_form.Assemble(0);
      divergence_form.Finalize(0);
      divergence.reset(divergence_form.ParallelAssemble());

      ParMixedBilinearForm continuity_form(&velocity_space, &pressure_space);
      continuity_form.AddDomainIntegrator(new RuleBilinearIntegrator<VectorDivergenceIntegrator>(
                                            quadrature_order));
      if (residual_method)
      {
         continuity_form.AddDomainIntegrator(new PSPGSymGradIntegrator(
                                            *viscosity, *viscosity_gradient, tau,
                                            quadrature_order));
      }
      continuity_form.Assemble(0);
      continuity_form.Finalize(0);
      continuity.reset(continuity_form.ParallelAssemble());

      stabilization.reset();
      if (options.method != StokesStabilizationMethod::TaylorHood)
      {
         stabilization = AssemblePressureBlock(pressure_space, false);
      }

      pressure_matrix = AssemblePressureBlock(pressure_space, true);
      pressure_matrix->GetDiag(pressure_diagonal);
      bool positive_diagonal = true;
      for (int index = 0; index < pressure_diagonal.Size(); ++index)
      {
         positive_diagonal = positive_diagonal && pressure_diagonal(index) > 0.0 &&
                             std::isfinite(pressure_diagonal(index));
      }
      Check(positive_diagonal, "Invalid pressure preconditioner diagonal");

      ScalarVectorProductCoefficient tau_force(tau, *force);
      ParLinearForm velocity_rhs(&velocity_space), pressure_rhs(&pressure_space);
      velocity_rhs.AddDomainIntegrator(new RuleLinearIntegrator<VectorDomainLFIntegrator>(
                                         quadrature_order, *force));
      if (gls)
      {
         velocity_rhs.AddDomainIntegrator(new GLSForceIntegrator(
                                            *viscosity, *viscosity_gradient, tau, *force, quadrature_order));
      }
      if (residual_method)
      {
         pressure_rhs.AddDomainIntegrator(new RuleLinearIntegrator<DomainLFGradIntegrator>(
                                         quadrature_order, tau_force));
      }
      velocity_rhs.Assemble();
      pressure_rhs.Assemble();
      Vector momentum_rhs(velocity_space.GetTrueVSize());
      Vector continuity_rhs(pressure_space.GetTrueVSize());
      velocity_rhs.ParallelAssemble(momentum_rhs);
      pressure_rhs.ParallelAssemble(continuity_rhs);
      Vector correction(momentum_rhs.Size());
      velocity_matrix->Mult(lifting, correction);
      momentum_rhs -= correction;
      correction.SetSize(continuity_rhs.Size());
      continuity->Mult(lifting, correction);
      continuity_rhs -= correction;

      std::unique_ptr<HypreParMatrix> eliminated_velocity(velocity_matrix->EliminateRowsCols(essential));
      std::unique_ptr<HypreParMatrix> eliminated_continuity(continuity->EliminateCols(essential));
      for (int index = 0; index < essential.Size(); ++index)
      {
         momentum_rhs(essential[index]) = lifting(essential[index]);
      }
      if (gls)
      {
         ConstantCoefficient negative_one(-1.0);
         ParMixedBilinearForm transpose_form(&velocity_space, &pressure_space);
         transpose_form.AddDomainIntegrator(new RuleBilinearIntegrator<VectorDivergenceIntegrator>(
                                              quadrature_order, negative_one));
         transpose_form.AddDomainIntegrator(new PSPGSymGradIntegrator(
                                              *viscosity, *viscosity_gradient, tau, quadrature_order));
         transpose_form.Assemble(0);
         transpose_form.Finalize(0);
         std::unique_ptr<HypreParMatrix> transpose_matrix(transpose_form.ParallelAssemble());
         momentum_pressure.reset(transpose_matrix->Transpose());
      }
      else
      {
         momentum_pressure.reset(divergence->Transpose());
         *momentum_pressure *= -1.0;
      }
      momentum_pressure->EliminateRows(essential);

      ConstantCoefficient one(1.0);
      ParGridFunction constant_field(&pressure_space);
      constant_field.ProjectCoefficient(one);
      constant_field.GetTrueDofs(pressure_constant);
      ParLinearForm moment_form(&pressure_space);
      moment_form.AddDomainIntegrator(new RuleLinearIntegrator<DomainLFIntegrator>(
                                        quadrature_order, one));
      moment_form.Assemble();
      moments.SetSize(pressure_space.GetTrueVSize());
      moment_form.ParallelAssemble(moments);
      volume = Dot(moments, pressure_constant);
      Check(std::isfinite(volume) && volume > 0.0, "Invalid domain volume");
      Vector pressure_mode(velocity_space.GetTrueVSize());
      momentum_pressure->Mult(pressure_constant, pressure_mode);
      const real_t mode_scale = momentum_pressure->FNorm() * Norm(pressure_constant);
      pressure_nullspace_detected = Norm(pressure_mode) <= 1e-11 * mode_scale;
      if (pressure_nullspace_detected)
      {
         Vector flux_weights(velocity_space.GetTrueVSize());
         divergence->MultTranspose(pressure_constant, flux_weights);
         Vector absolute_constant(pressure_constant), roundoff_weights(flux_weights.Size());
         for (int index = 0; index < absolute_constant.Size(); ++index)
         {
            absolute_constant(index) = std::abs(absolute_constant(index));
         }
         divergence->AbsMultTranspose(absolute_constant, roundoff_weights);
         const real_t prescribed_flux = Dot(flux_weights, lifting);
         real_t local_scales[2] = {}, scales[2] = {};
         for (int index = 0; index < essential.Size(); ++index)
         {
            const int dof = essential[index];
            local_scales[0] += std::abs(flux_weights(dof) * lifting(dof));
            local_scales[1] += roundoff_weights(dof) * std::abs(lifting(dof));
         }
         MPI_Allreduce(local_scales, scales, 2, MPITypeMap<real_t>::mpi_type, MPI_SUM, comm);
         const real_t flux_tolerance = 1e-10 * scales[0] +
                                       128 * std::numeric_limits<real_t>::epsilon() * scales[1];
         Check(std::isfinite(prescribed_flux) && std::isfinite(scales[0]) && std::isfinite(scales[1]) &&
               std::abs(prescribed_flux) <= flux_tolerance,
               "Prescribed velocity has incompatible net boundary flux");
      }
      Check(options.pressure_gauge != StokesPressureGauge::MeanZero || pressure_nullspace_detected,
            "Pressure gauge mean-zero requires a constant-pressure nullspace; boundary conditions already determine absolute pressure");
      Check(options.pressure_gauge != StokesPressureGauge::None || !pressure_nullspace_detected,
            "Pressure gauge none leaves an unresolved constant-pressure nullspace; use auto or mean-zero");
      gauge = pressure_nullspace_detected && options.pressure_gauge != StokesPressureGauge::None;

      const int block_count = gauge ? 3 : 2;
      offsets.SetSize(block_count + 1);
      offsets[0] = 0;
      offsets[1] = velocity_space.GetTrueVSize();
      offsets[2] = offsets[1] + pressure_space.GetTrueVSize();
      if (gauge) { offsets[3] = offsets[2] + (rank == 0 ? 1 : 0); }
      rhs.SetSize(offsets.Last());
      solution.SetSize(offsets.Last());
      rhs = 0.0;
      solution = 0.0;
      BlockVector rhs_blocks(rhs, offsets);
      rhs_blocks.GetBlock(0) = momentum_rhs;
      rhs_blocks.GetBlock(1) = continuity_rhs;
      Array2D<const HypreParMatrix*> blocks(block_count, block_count);
      for (int row = 0; row < block_count; ++row)
      {
         for (int column = 0; column < block_count; ++column)
         {
            blocks(row, column) = nullptr;
         }
      }
      blocks(0, 0) = velocity_matrix.get();
      blocks(0, 1) = momentum_pressure.get();
      blocks(1, 0) = continuity.get();
      blocks(1, 1) = stabilization.get();
      gauge_column.reset();
      gauge_row.reset();
      if (gauge)
      {
         gauge_partition.SetSize(HYPRE_AssumedPartitionCheck() ? 2 : ranks + 1);
         gauge_partition = 1;
         if (!HYPRE_AssumedPartitionCheck() || rank == 0) { gauge_partition[0] = 0; }
         const int pressure_size = pressure_space.GetTrueVSize();
         Array<int> row_indices(pressure_size + 1);
         Array<HYPRE_BigInt> column_indices(pressure_size);
         for (int index = 0; index <= pressure_size; ++index) { row_indices[index] = index; }
         column_indices = 0;
         gauge_column = std::make_unique<HypreParMatrix>(comm, pressure_size,
                           pressure_space.GlobalTrueVSize(), 1, row_indices.GetData(),
                           column_indices.GetData(), moments.GetData(),
                           pressure_space.GetTrueDofOffsets(), gauge_partition.GetData());
         gauge_row.reset(gauge_column->Transpose());
         blocks(1, 2) = gauge_column.get();
         blocks(2, 1) = gauge_row.get();
         Vector scaled(moments);
         for (int index = 0; index < scaled.Size(); ++index)
         {
            scaled(index) /= pressure_diagonal(index);
         }
         const real_t gauge_scale = Dot(moments, scaled);
         Check(std::isfinite(gauge_scale) && gauge_scale > 0.0, "Invalid gauge scale");
         gauge_diagonal.SetSize(rank == 0 ? 1 : 0);
         gauge_diagonal = gauge_scale;
      }
      system.reset(HypreParMatrixFromBlocks(blocks));
      const real_t system_norm = system->FNorm(), rhs_norm = Norm(rhs);
      Check(std::isfinite(system_norm) && std::isfinite(rhs_norm) && rhs.CheckFinite() == 0,
            "Nonfinite Stokes matrix or right-hand side");
      const double elapsed = MPI_Wtime() - assembly_start;
      MPI_Allreduce(&elapsed, &assembly_seconds, 1, MPI_DOUBLE, MPI_MAX, comm);
      assembled = true;
      if (rank == 0 && options.print_level > 0)
      {
         out << "Stokes: Nv=" << velocity_space.GlobalTrueVSize()
             << " Np=" << pressure_space.GlobalTrueVSize()
             << " pressure_gauge=" << gauge << " assembly_seconds=" << assembly_seconds << '\n';
      }
   }

   StokesSolveInfo Solve()
   {
      CheckSequences();
      StokesSolveInfo info;
      info.assembly_performed = !assembled;
      if (!assembled) { Assemble(); }
      solution = 0.0;
      info.geometry_order = geometry_order;
      info.integration_order = quadrature_order;
      info.assembly_seconds = assembly_seconds;
      bool backend_converged = true;
      double local_seconds[4] = {}, global_seconds[4] = {};
      const double setup_started = MPI_Wtime();
      double finalization_started = 0.0;
      if (options.linear_solver == LinearSolverType::MUMPS)
      {
#ifdef MFEM_USE_MUMPS
         MUMPSSolver direct(comm);
         direct.SetPrintLevel(0);
         direct.SetMatrixSymType(MUMPSSolver::UNSYMMETRIC);
         direct.SetOperator(*system);
         const double solve_started = MPI_Wtime();
         local_seconds[1] = solve_started - setup_started;
         direct.Mult(rhs, solution);
         finalization_started = MPI_Wtime();
         local_seconds[2] = finalization_started - solve_started;
#else
         MFEM_ABORT("MUMPS is unavailable");
#endif
      }
      else
      {
         const bool upper = options.preconditioner == StokesPreconditionerType::BlockUpperTriangularP1AMG ||
                            options.preconditioner == StokesPreconditionerType::BlockUpperTriangularP1AMGAssembled;
         const bool lower = options.preconditioner == StokesPreconditionerType::BlockLowerTriangularP1AMG ||
                            options.preconditioner == StokesPreconditionerType::BlockLowerTriangularP1AMGAssembled;
         const bool assemble_p1 = options.preconditioner == StokesPreconditionerType::BlockDiagonalP1AMGAssembled ||
                                  options.preconditioner == StokesPreconditionerType::BlockUpperTriangularP1AMGAssembled ||
                                  options.preconditioner == StokesPreconditionerType::BlockLowerTriangularP1AMGAssembled;
         std::unique_ptr<Solver> velocity_inverse, pressure_inverse;
         Vector gauge_scale(gauge_diagonal);
         if (options.preconditioner == StokesPreconditionerType::BlockDiagonalAMG)
         {
            auto amg = std::make_unique<HypreBoomerAMG>(*velocity_matrix);
            amg->SetElasticityOptions(&velocity_space, options.amg_elasticity_interp_refine);
            amg->SetMaxIter(1);
            amg->SetTol(0.0);
            amg->SetPrintLevel(0);
            Vector setup_rhs(velocity_matrix->Height()), setup_solution(velocity_matrix->Height());
            setup_rhs = 0.0;
            setup_solution = 0.0;
            amg->Setup(setup_rhs, setup_solution);
            velocity_inverse = std::move(amg);
            pressure_inverse = std::make_unique<DiagonalInverse>(pressure_diagonal);
         }
         else
         {
            std::vector<Array<int>> boundary_markers(dimension);
            for (auto &marker : boundary_markers)
            {
               marker.SetSize(max_boundary_attribute);
               marker = 0;
            }
            for (const auto &boundary : boundary_values)
            {
               boundary_markers[boundary.first.second][boundary.first.first - 1] = 1;
            }
            detail::P1AMGPreconditioner::CoarseAssembler assemble_velocity, assemble_pressure;
            if (assemble_p1)
            {
               assemble_velocity = [this](ParFiniteElementSpace &space)
               {
                  return AssembleVelocityBlock(space);
               };
               assemble_pressure = [this](ParFiniteElementSpace &space)
               {
                  return AssemblePressureBlock(space, true);
               };
            }
            velocity_inverse = std::make_unique<detail::P1AMGPreconditioner>(
                                  *velocity_matrix, velocity_space, boundary_markers,
                                  options.p1_smoothing_steps, true, options.amg_elasticity_interp_refine,
                                  assemble_velocity);
            pressure_inverse = std::make_unique<detail::P1AMGPreconditioner>(
                                  *pressure_matrix, pressure_space, std::vector<Array<int>>{},
                                  options.p1_smoothing_steps, false, false, assemble_pressure);
            if (gauge)
            {
               Vector scaled(moments.Size());
               pressure_inverse->Mult(moments, scaled);
               const real_t scale = Dot(moments, scaled);
               Check(std::isfinite(scale) && scale > 0.0, "Invalid P1-AMG gauge scale");
               gauge_scale = (upper || lower) ? -scale : scale;
            }
         }
         DiagonalInverse gauge_inverse(gauge_scale);
         std::unique_ptr<Solver> preconditioner;
         if (upper)
         {
            preconditioner = std::make_unique<detail::UpperTriangularPreconditioner>(
                                offsets, *velocity_inverse, *pressure_inverse, *momentum_pressure,
                                gauge ? &gauge_inverse : nullptr, gauge_column.get());
         }
         else if (lower)
         {
            auto triangular = std::make_unique<BlockLowerTriangularPreconditioner>(offsets);
            triangular->SetDiagonalBlock(0, velocity_inverse.get());
            triangular->SetDiagonalBlock(1, pressure_inverse.get());
            triangular->SetBlock(1, 0, continuity.get());
            if (gauge)
            {
               triangular->SetDiagonalBlock(2, &gauge_inverse);
               triangular->SetBlock(2, 1, gauge_row.get());
            }
            preconditioner = std::move(triangular);
         }
         else
         {
            auto diagonal = std::make_unique<BlockDiagonalPreconditioner>(offsets);
            diagonal->SetDiagonalBlock(0, velocity_inverse.get());
            diagonal->SetDiagonalBlock(1, pressure_inverse.get());
            if (gauge) { diagonal->SetDiagonalBlock(2, &gauge_inverse); }
            preconditioner = std::move(diagonal);
         }
         if (rank == 0 && options.print_level > 0)
         {
            out << "Stokes: preconditioner=" << StokesPreconditionerName(options.preconditioner) << '\n';
         }
         RightPreconditionedOperator right_preconditioned(*system, *preconditioner);
         std::unique_ptr<IterativeSolver> krylov;
         if (options.linear_solver == LinearSolverType::GMRES)
         {
            auto solver = std::make_unique<GMRESSolver>(comm);
            solver->SetKDim(options.restart_dimension);
            solver->SetOperator(right_preconditioned);
            krylov = std::move(solver);
         }
         else
         {
            auto solver = std::make_unique<FGMRESSolver>(comm);
            solver->SetKDim(options.restart_dimension);
            solver->SetOperator(*system);
            solver->SetPreconditioner(*preconditioner);
            krylov = std::move(solver);
         }
         krylov->SetRelTol(options.relative_tolerance);
         krylov->SetAbsTol(options.absolute_tolerance);
         krylov->SetMaxIter(options.max_iterations);
         krylov->SetPrintLevel(options.print_level > 1 ? 1 : -1);
         krylov->iterative_mode = false;
         const double solve_started = MPI_Wtime();
         local_seconds[0] = solve_started - setup_started;
         krylov->Mult(rhs, solution);
         if (options.linear_solver == LinearSolverType::GMRES)
         {
            Vector preconditioned_solution(solution.Size());
            preconditioner->Mult(solution, preconditioned_solution);
            solution = preconditioned_solution;
         }
         info.iterations = krylov->GetNumIterations();
         backend_converged = krylov->GetConverged();
         const real_t target = std::max(options.absolute_tolerance,
                                        options.relative_tolerance * Norm(rhs));
         Vector correction_rhs(rhs.Size()), correction_solution(rhs.Size());
         for (int correction = 0; correction < 2 && backend_converged &&
              info.iterations < options.max_iterations; ++correction)
         {
            system->Mult(solution, correction_rhs);
            correction_rhs *= -1.0;
            correction_rhs += rhs;
            const real_t residual_norm = Norm(correction_rhs);
            if (!std::isfinite(residual_norm) || residual_norm <= target) { break; }
            krylov->SetRelTol(0.0);
            krylov->SetAbsTol(0.25 * target);
            krylov->SetMaxIter(options.max_iterations - info.iterations);
            krylov->Mult(correction_rhs, correction_solution);
            if (options.linear_solver == LinearSolverType::GMRES)
            {
               Vector preconditioned_correction(correction_solution.Size());
               preconditioner->Mult(correction_solution, preconditioned_correction);
               correction_solution = preconditioned_correction;
            }
            solution += correction_solution;
            info.iterations += krylov->GetNumIterations();
            backend_converged = krylov->GetConverged();
         }
         finalization_started = MPI_Wtime();
         local_seconds[2] = finalization_started - solve_started;
      }
      Vector residual(rhs.Size());
      system->Mult(solution, residual);
      residual -= rhs;
      info.absolute_residual = Norm(residual);
      const real_t rhs_norm = Norm(rhs);
      info.relative_residual = info.absolute_residual / (rhs_norm > 0.0 ? rhs_norm : 1.0);
      const int local_valid = backend_converged && solution.CheckFinite() == 0 && residual.CheckFinite() == 0;
      int global_valid = 0;
      MPI_Allreduce(&local_valid, &global_valid, 1, MPI_INT, MPI_MIN, comm);
      info.converged = global_valid && std::isfinite(info.absolute_residual) && std::isfinite(rhs_norm) &&
                       info.absolute_residual <=
                       std::max(options.absolute_tolerance,
                                options.relative_tolerance * rhs_norm);
      BlockVector solution_blocks(solution, offsets);
      velocity.SetFromTrueDofs(solution_blocks.GetBlock(0));
      pressure.SetFromTrueDofs(solution_blocks.GetBlock(1));
      info.pressure_gauge_active = gauge;
      info.pressure_gauge = options.pressure_gauge;
      info.pressure_nullspace_detected = pressure_nullspace_detected;
      info.pressure_mean = Dot(moments, solution_blocks.GetBlock(1)) / volume;
      Vector divergence_value(pressure_space.GetTrueVSize());
      divergence->Mult(solution_blocks.GetBlock(0), divergence_value);
      info.boundary_flux = Dot(pressure_constant, divergence_value);
      if (gauge)
      {
         if (rank == 0) { info.gauge_multiplier = solution_blocks.GetBlock(2)(0); }
         MPI_Bcast(&info.gauge_multiplier, 1, MPITypeMap<real_t>::mpi_type, 0, comm);
      }
      has_solution = true;
      local_seconds[3] = MPI_Wtime() - finalization_started;
      MPI_Allreduce(local_seconds, global_seconds, 4, MPI_DOUBLE, MPI_MAX, comm);
      info.preconditioner_setup_seconds = global_seconds[0];
      info.direct_setup_seconds = global_seconds[1];
      info.linear_solve_seconds = global_seconds[2];
      info.finalization_seconds = global_seconds[3];
      if (rank == 0 && options.print_level > 0)
      {
         out << "Stokes: converged=" << info.converged << " iterations=" << info.iterations
             << " true_relative_residual=" << info.relative_residual
             << " pressure_gauge=" << StokesPressureGaugeName(info.pressure_gauge)
             << " pressure_nullspace_detected=" << info.pressure_nullspace_detected
             << " pressure_gauge_active=" << info.pressure_gauge_active
             << " pressure_mean=" << info.pressure_mean
             << " flux=" << info.boundary_flux << '\n';
      }
      return info;
   }
};

StokesSolver::StokesSolver(ParFiniteElementSpace &velocity_space,
                                 ParFiniteElementSpace &pressure_space,
                                 real_t viscosity, const StokesSolverOptions &options)
   : impl(std::make_unique<Impl>(velocity_space, pressure_space, options,
                                nullptr, nullptr, viscosity)) { }

StokesSolver::StokesSolver(ParFiniteElementSpace &velocity_space,
                                 ParFiniteElementSpace &pressure_space,
                                 Coefficient &viscosity, VectorCoefficient &viscosity_gradient,
                                 const StokesSolverOptions &options)
   : impl(std::make_unique<Impl>(velocity_space, pressure_space, options,
                                &viscosity, &viscosity_gradient, 0.0)) { }

StokesSolver::~StokesSolver() = default;

void StokesSolver::AddVelocityBC(int attribute, int direction, real_t value)
{
   impl->CheckBoundary(attribute, direction);
   impl->Check(std::isfinite(value), "Nonfinite velocity boundary value");
   for (int component = 0; component < impl->dimension; ++component)
   {
      if (direction != -1 && direction != component) { continue; }
      auto coefficient = std::make_unique<ConstantCoefficient>(value);
      auto *pointer = coefficient.get();
      impl->SetBoundary(attribute, component, pointer, std::move(coefficient));
   }
}

void StokesSolver::AddVelocityBC(int attribute, int direction, Coefficient &value)
{
   impl->CheckBoundary(attribute, direction);
   for (int component = 0; component < impl->dimension; ++component)
   {
      if (direction == -1 || direction == component)
      {
         impl->SetBoundary(attribute, component, &value);
      }
   }
}

void StokesSolver::AddVelocityBC(int attribute, VectorCoefficient &velocity)
{
   impl->CheckBoundary(attribute, -1);
   impl->Check(velocity.GetVDim() == impl->dimension, "Velocity BC dimension mismatch");
   for (int component = 0; component < impl->dimension; ++component)
   {
      auto coefficient = std::make_unique<VectorComponentCoefficient>(velocity, component);
      auto *pointer = coefficient.get();
      impl->SetBoundary(attribute, component, pointer, std::move(coefficient));
   }
}

void StokesSolver::DelVelocityBC()
{
   impl->boundary_values.clear();
   impl->Invalidate();
}

void StokesSolver::SetVolForce(VectorCoefficient &force)
{
   impl->Check(force.GetVDim() == impl->dimension, "Force dimension mismatch");
   impl->force = &force;
   impl->owned_force.reset();
   impl->Invalidate();
}

void StokesSolver::SetVolForce(const Vector &constant_force)
{
   impl->Check(constant_force.Size() == impl->dimension, "Force dimension mismatch");
   impl->owned_force = std::make_unique<VectorConstantCoefficient>(constant_force);
   impl->force = impl->owned_force.get();
   impl->Invalidate();
}

void StokesSolver::SetVolForce(real_t force_x, real_t force_y, real_t force_z)
{
   impl->Check(impl->dimension == 3 || force_z == 0.0, "Nonzero z force in 2D");
   Vector force(impl->dimension);
   force(0) = force_x;
   force(1) = force_y;
   if (impl->dimension == 3) { force(2) = force_z; }
   SetVolForce(force);
}

void StokesSolver::Assemble() { impl->Assemble(); }
StokesSolveInfo StokesSolver::Solve() { return impl->Solve(); }

const ParGridFunction &StokesSolver::GetVelocity() const
{
   MFEM_VERIFY(impl->has_solution, "Call Solve before requesting velocity");
   return impl->velocity;
}

const ParGridFunction &StokesSolver::GetPressure() const
{
   MFEM_VERIFY(impl->has_solution, "Call Solve before requesting pressure");
   return impl->pressure;
}

}
