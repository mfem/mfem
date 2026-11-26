#include "stokes.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <utility>
#include <vector>

namespace mfem::flow
{

namespace detail
{
void Check(MPI_Comm communicator, bool valid, const char *message)
{
   int local = valid ? 1 : 0, global = 0;
   MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_MIN, communicator);
   MFEM_VERIFY(global, message);
}

void CheckReplicated(MPI_Comm communicator, const std::vector<real_t> &signature)
{
   Check(communicator, signature.size() <= std::numeric_limits<int>::max(),
         "Navier-Stokes configuration is too large");
   const int count = static_cast<int>(signature.size());
   int minimum_count = 0, maximum_count = 0;
   MPI_Allreduce(&count, &minimum_count, 1, MPI_INT, MPI_MIN, communicator);
   MPI_Allreduce(&count, &maximum_count, 1, MPI_INT, MPI_MAX, communicator);
   Check(communicator, minimum_count == maximum_count,
         "Navier-Stokes configuration differs across ranks");
   Check(communicator, std::all_of(signature.begin(), signature.end(),
                                  [](real_t value) { return std::isfinite(value); }),
         "Nonfinite Navier-Stokes configuration");
   std::vector<real_t> minimum(count), maximum(count);
   MPI_Allreduce(signature.data(), minimum.data(), count, MPI_DOUBLE, MPI_MIN, communicator);
   MPI_Allreduce(signature.data(), maximum.data(), count, MPI_DOUBLE, MPI_MAX, communicator);
   Check(communicator, minimum == maximum, "Navier-Stokes configuration differs across ranks");
}
}

real_t ScalarData::Eval(ElementTransformation &transformation,
                        const IntegrationPoint &point) const
{
   if (const auto *constant = std::get_if<real_t>(&value)) { return *constant; }
   if (const auto *coefficient = std::get_if<Coefficient *>(&value))
   {
      return (*coefficient)->Eval(transformation, point);
   }
   const auto &component = std::get<std::pair<VectorCoefficient *, int>>(value);
   Vector result(component.first->GetVDim());
   component.first->Eval(result, transformation, point);
   return result(component.second);
}

bool ScalarData::Valid(int dimension) const
{
   if (const auto *constant = std::get_if<real_t>(&value)) { return std::isfinite(*constant); }
   if (const auto *coefficient = std::get_if<Coefficient *>(&value)) { return *coefficient; }
   const auto &component = std::get<std::pair<VectorCoefficient *, int>>(value);
   return component.first && component.first->GetVDim() == dimension &&
          component.second >= 0 && component.second < dimension;
}

void ScalarData::AppendSignature(std::vector<real_t> &signature) const
{
   signature.push_back(static_cast<real_t>(value.index()));
   if (const auto *constant = std::get_if<real_t>(&value)) { signature.push_back(*constant); }
   if (const auto *component = std::get_if<std::pair<VectorCoefficient *, int>>(&value))
   {
      signature.push_back(component->first->GetVDim());
      signature.push_back(component->second);
   }
}

namespace
{
void SetVector(ComponentData &data, int attribute, VectorCoefficient &value)
{
   auto first = data.lower_bound({attribute, std::numeric_limits<int>::min()});
   auto last = data.upper_bound({attribute, std::numeric_limits<int>::max()});
   data.erase(first, last);
   for (int component = 0; component < std::max(1, value.GetVDim()); ++component)
   {
      data.insert_or_assign({attribute, component}, ScalarData(value, component));
   }
}

void AppendComponents(const ComponentData &data, std::vector<real_t> &signature)
{
   signature.push_back(static_cast<real_t>(data.size()));
   for (const auto &entry : data)
   {
      signature.push_back(entry.first.first);
      signature.push_back(entry.first.second);
      entry.second.AppendSignature(signature);
   }
}

void AppendBoundary(const BoundaryConditions &boundary, std::vector<real_t> &signature)
{
   signature.push_back(static_cast<int>(boundary.GetPressureCondition()));
   AppendComponents(boundary.GetVelocity(), signature);
   AppendComponents(boundary.GetTraction(), signature);
   signature.push_back(static_cast<real_t>(boundary.GetPressure().size()));
   for (const auto &entry : boundary.GetPressure())
   {
      signature.push_back(entry.first);
      entry.second.AppendSignature(signature);
   }
}

bool Finite(const Vector &vector)
{
   for (int index = 0; index < vector.Size(); ++index)
   {
      if (!std::isfinite(vector(index))) { return false; }
   }
   return true;
}

Vector ZeroVector(int dimension)
{
   Vector value(dimension);
   value = 0.0;
   return value;
}

class AttributeForce : public VectorCoefficient
{
   const ComponentData &data;
public:
   AttributeForce(int dimension, const ComponentData &data_)
      : VectorCoefficient(dimension), data(data_) { }
   void Eval(Vector &result, ElementTransformation &transformation,
             const IntegrationPoint &point) override
   {
      result.SetSize(vdim);
      result = 0.0;
      for (int component = 0; component < vdim; ++component)
      {
         auto found = data.find({transformation.Attribute, component});
         if (found != data.end()) { result(component) = found->second.Eval(transformation, point); }
      }
   }
};

struct BrinkmanData
{
   ScalarData penalty;
   VectorCoefficient *target;
};

class AttributePenalty : public Coefficient
{
   const std::map<int, BrinkmanData> &data;
public:
   explicit AttributePenalty(const std::map<int, BrinkmanData> &data_) : data(data_) { }
   real_t Eval(ElementTransformation &transformation, const IntegrationPoint &point) override
   {
      auto found = data.find(transformation.Attribute);
      if (found == data.end()) { found = data.find(0); }
      return found == data.end() ? 0.0 : found->second.penalty.Eval(transformation, point);
   }
};

class AttributeTarget : public VectorCoefficient
{
   const std::map<int, BrinkmanData> &data;
public:
   AttributeTarget(int dimension, const std::map<int, BrinkmanData> &data_)
      : VectorCoefficient(dimension), data(data_) { }
   void Eval(Vector &result, ElementTransformation &transformation,
             const IntegrationPoint &point) override
   {
      result.SetSize(vdim);
      result = 0.0;
      auto found = data.find(transformation.Attribute);
      if (found == data.end()) { found = data.find(0); }
      if (found != data.end() && found->second.target)
      {
         found->second.target->Eval(result, transformation, point);
      }
   }
};
}

void BoundaryConditions::SetVelocity(int attribute, Direction direction, real_t value)
{ velocity.insert_or_assign({attribute, static_cast<int>(direction)}, ScalarData(value)); }
void BoundaryConditions::SetVelocity(int attribute, Direction direction, Coefficient &value)
{ velocity.insert_or_assign({attribute, static_cast<int>(direction)}, ScalarData(value)); }
void BoundaryConditions::SetVelocity(int attribute, VectorCoefficient &value)
{ SetVector(velocity, attribute, value); }
void BoundaryConditions::SetTraction(int attribute, Direction direction, real_t value)
{ traction.insert_or_assign({attribute, static_cast<int>(direction)}, ScalarData(value)); }
void BoundaryConditions::SetTraction(int attribute, Direction direction, Coefficient &value)
{ traction.insert_or_assign({attribute, static_cast<int>(direction)}, ScalarData(value)); }
void BoundaryConditions::SetTraction(int attribute, VectorCoefficient &value)
{ SetVector(traction, attribute, value); }
void BoundaryConditions::SetPressure(int attribute, real_t value)
{ pressure.insert_or_assign(attribute, ScalarData(value)); }
void BoundaryConditions::SetPressure(int attribute, Coefficient &value)
{ pressure.insert_or_assign(attribute, ScalarData(value)); }

void LinearOptions::Validate(MPI_Comm communicator) const
{
   detail::Check(communicator, std::isfinite(relative_tolerance) && relative_tolerance > 0.0 &&
                 std::isfinite(absolute_tolerance) && absolute_tolerance >= 0.0,
                 "Invalid Navier-Stokes linear tolerances");
   std::vector<real_t> signature{static_cast<real_t>(backend.index()), relative_tolerance,
                               absolute_tolerance, static_cast<real_t>(print_level)};
   if (const auto *iterative = std::get_if<IterativeOptions>(&backend))
   {
      signature.insert(signature.end(), {static_cast<real_t>(iterative->method),
                        static_cast<real_t>(iterative->preconditioner),
                        iterative->amg_elasticity_interp_refine ? 1.0 : 0.0,
                        static_cast<real_t>(iterative->max_iterations),
                        static_cast<real_t>(iterative->restart_dimension),
                        static_cast<real_t>(iterative->p1_smoothing_steps),
                        static_cast<real_t>(iterative->schur),
                        static_cast<real_t>(iterative->lsc_solver), static_cast<real_t>(iterative->lsc_amg_cycles),
                        static_cast<real_t>(iterative->pcd_solver), static_cast<real_t>(iterative->pcd_amg_cycles),
                        iterative->velocity_inverse_factory ? 1.0 : 0.0,
                        iterative->pressure_inverse_factory ? 1.0 : 0.0,
                        iterative->hierarchy ? 1.0 : 0.0});
   }
   detail::CheckReplicated(communicator, signature);
   if (const auto *iterative = std::get_if<IterativeOptions>(&backend))
   {
      detail::Check(communicator, (iterative->method == LinearSolverType::GMRES ||
                    iterative->method == LinearSolverType::FGMRES) &&
                    iterative->preconditioner >= PreconditionerType::BlockDiagonalAMG &&
                    iterative->preconditioner <= PreconditionerType::BlockLowerTriangularP1AMGAssembled &&
                    iterative->max_iterations > 0 &&
                    iterative->restart_dimension > 0 && iterative->p1_smoothing_steps > 0,
                    "Invalid Navier-Stokes iterative options");
      detail::Check(communicator, iterative->schur == SchurApproximation::WeightedMass ||
                    iterative->schur == SchurApproximation::ExperimentalReactionDiagonal ||
                    iterative->schur == SchurApproximation::ExperimentalLSC ||
                    iterative->schur == SchurApproximation::ExperimentalPCD,
                    "Invalid Navier-Stokes Schur approximation");
      detail::Check(communicator, iterative->schur != SchurApproximation::ExperimentalLSC ||
                    ((iterative->lsc_solver == LSCPressureSolver::Mumps ||
                      iterative->lsc_solver == LSCPressureSolver::BoomerAMG) && iterative->lsc_amg_cycles > 0),
                    "Invalid LSC pressure inverse options");
      detail::Check(communicator, iterative->schur != SchurApproximation::ExperimentalPCD ||
                    ((iterative->pcd_solver == PCDPressureSolver::Mumps ||
                      iterative->pcd_solver == PCDPressureSolver::BoomerAMG) && iterative->pcd_amg_cycles > 0),
                    "Invalid PCD pressure inverse options");
#ifndef MFEM_USE_MUMPS
      detail::Check(communicator, iterative->schur != SchurApproximation::ExperimentalReactionDiagonal,
                    "Experimental reaction Schur requires MUMPS");
      detail::Check(communicator, iterative->schur != SchurApproximation::ExperimentalLSC ||
                    iterative->lsc_solver != LSCPressureSolver::Mumps, "LSC MUMPS support is unavailable");
      detail::Check(communicator, iterative->schur != SchurApproximation::ExperimentalPCD ||
                    iterative->pcd_solver != PCDPressureSolver::Mumps, "PCD MUMPS support is unavailable");
#endif
   }
#ifndef MFEM_USE_MUMPS
   detail::Check(communicator, !std::holds_alternative<MumpsOptions>(backend),
                 "Navier-Stokes MUMPS support is unavailable");
#endif
}

void FlowSolverOptions::Validate(MPI_Comm communicator) const
{
   linear.Validate(communicator);
   detail::CheckReplicated(communicator,
   {
      static_cast<real_t>(nonlinear_method), static_cast<real_t>(gauge_enforcement),
      static_cast<real_t>(initial_guess), convection ? 1.0 : 0.0,
      experimental_residual ? 1.0 : 0.0,
      experimental_residual ? experimental_residual->advection_scale : 0.0,
      experimental_residual ? experimental_residual->reaction_scale : 0.0,
      static_cast<real_t>(max_iterations), relative_tolerance, absolute_tolerance,
      initial_damping, minimum_damping, line_search_reduction, armijo_constant
   });
   detail::Check(communicator, (nonlinear_method == NonlinearMethod::Newton ||
                 nonlinear_method == NonlinearMethod::Picard) && max_iterations > 0 &&
                 relative_tolerance > 0.0 && absolute_tolerance >= 0.0 &&
                 minimum_damping > 0.0 && minimum_damping <= initial_damping &&
                 initial_damping <= 1.0 && line_search_reduction > 0.0 &&
                 line_search_reduction < 1.0 && armijo_constant > 0.0 && armijo_constant < 1.0,
                 "Invalid Navier-Stokes nonlinear options");
   detail::Check(communicator, initial_guess == InitialGuess::Stokes || initial_guess == InitialGuess::Provided,
                 "Invalid Navier-Stokes initial guess policy");
   detail::Check(communicator, gauge_enforcement == GaugeEnforcement::AugmentedSystem ||
                 (gauge_enforcement == GaugeEnforcement::NullspaceProjection &&
                  std::holds_alternative<IterativeOptions>(linear.backend)),
                 "Navier-Stokes nullspace projection requires the iterative backend");
   detail::Check(communicator, !experimental_residual ||
                 (experimental_residual->advection_scale >= 0.0 && experimental_residual->reaction_scale >= 0.0),
                 "Invalid experimental residual stabilization scales");
}

class FlowProblem::Impl
{
public:
   ParFiniteElementSpace &velocity, &pressure;
   ParMesh &mesh;
   MPI_Comm comm;
   const long mesh_sequence, velocity_sequence, pressure_sequence;
   std::uint64_t revision = 0;
   std::unique_ptr<ConstantCoefficient> owned_viscosity;
   Coefficient *viscosity = nullptr;
   Vector zero;
   VectorConstantCoefficient zero_gradient;
   VectorCoefficient *gradient = nullptr;
   BoundaryConditions boundary;
   Array<int> prescribed_dofs;
   VectorCoefficient *prescribed_values = nullptr;
   VelocityConstraintSelector auxiliary_constraints;
   ComponentData forces;
   std::map<int, BrinkmanData> brinkman;
   AttributeForce force;
   AttributePenalty penalty;
   AttributeTarget target;
   Stabilization stabilization = NoStabilization{};
   PressureGauge gauge = PressureGauge::None();
   mutable std::optional<bool> resolved_gauge;
   mutable std::uint64_t gauge_revision = 0;
   int integration_order = -1;

   Impl(ParFiniteElementSpace &velocity_, ParFiniteElementSpace &pressure_)
      : velocity(velocity_), pressure(pressure_), mesh(*velocity.GetParMesh()),
        comm(velocity.GetComm()), mesh_sequence(mesh.GetSequence()),
        velocity_sequence(velocity.GetSequence()), pressure_sequence(pressure.GetSequence()),
        zero(ZeroVector(mesh.Dimension())), zero_gradient(zero), force(mesh.Dimension(), forces),
        penalty(brinkman), target(mesh.Dimension(), brinkman)
   {
      detail::Check(comm, pressure.GetParMesh() == &mesh &&
                    mesh.Dimension() == mesh.SpaceDimension() &&
                    (mesh.Dimension() == 2 || mesh.Dimension() == 3) && !mesh.Nonconforming() &&
                    velocity.GetVDim() == mesh.Dimension() && pressure.GetVDim() == 1 &&
                    velocity.GetOrdering() == Ordering::byVDIM &&
                    dynamic_cast<const H1_FECollection *>(velocity.FEColl()) &&
                    dynamic_cast<const H1_FECollection *>(pressure.FEColl()),
                    "Incompatible Navier-Stokes H1 spaces or ordering");
   }

   void CheckCurrent() const
   {
      detail::Check(comm, mesh.GetSequence() == mesh_sequence &&
                    velocity.GetSequence() == velocity_sequence &&
                    pressure.GetSequence() == pressure_sequence,
                    "Navier-Stokes mesh/spaces changed; reconstruct the problem");
   }

   void CheckAttribute(int attribute, bool boundary_attribute) const
   {
      CheckCurrent();
      const auto &attributes = boundary_attribute ? mesh.bdr_attributes : mesh.attributes;
      int local = attributes.Find(attribute) >= 0, global = 0;
      MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_MAX, comm);
      detail::Check(comm, attribute > 0 && global,
                    boundary_attribute ? "Unknown Navier-Stokes boundary attribute" :
                    "Unknown Navier-Stokes domain attribute");
   }

   void CheckComponents(const ComponentData &data, bool boundary_attribute) const
   {
      for (const auto &entry : data)
      {
         CheckAttribute(entry.first.first, boundary_attribute);
         detail::Check(comm, entry.first.second >= 0 && entry.first.second < mesh.Dimension() &&
                       entry.second.Valid(mesh.Dimension()),
                       "Invalid Navier-Stokes component or coefficient dimension");
      }
   }

   void CheckBoundary() const
   {
      std::vector<real_t> signature;
      AppendBoundary(boundary, signature);
      detail::CheckReplicated(comm, signature);
      detail::Check(comm, boundary.GetPressureCondition() == PressureCondition::OutletTraction,
                    "Navier-Stokes EssentialTrace has no supported velocity closure");
      CheckComponents(boundary.GetVelocity(), true);
      CheckComponents(boundary.GetTraction(), true);
      bool compatible = true;
      for (const auto &entry : boundary.GetTraction())
      {
         compatible = compatible && !boundary.GetVelocity().count(entry.first) &&
                      !boundary.GetPressure().count(entry.first.first);
      }
      for (const auto &entry : boundary.GetPressure())
      {
         CheckAttribute(entry.first, true);
         detail::Check(comm, entry.second.Valid(mesh.Dimension()),
                       "Invalid Navier-Stokes pressure data");
         int prescribed = 0;
         for (int component = 0; component < mesh.Dimension(); ++component)
         {
            prescribed += boundary.GetVelocity().count({entry.first, component});
         }
         compatible = compatible && prescribed < mesh.Dimension();
      }
      detail::Check(comm, compatible, "Overlapping Navier-Stokes boundary conditions");
   }

   void SetForce(int attribute, int component, const ScalarData &value)
   {
      CheckAttribute(attribute, false);
      detail::Check(comm, component >= 0 && component < mesh.Dimension() && value.Valid(mesh.Dimension()),
                    "Invalid Navier-Stokes component or coefficient dimension");
      forces.insert_or_assign({attribute, component}, value);
      ++revision;
   }

   void SetDrag(int attribute, const ScalarData &value, VectorCoefficient *target_)
   {
      CheckAttribute(attribute, false);
      detail::Check(comm, value.Valid(mesh.Dimension()) &&
                    (!target_ || target_->GetVDim() == mesh.Dimension()),
                    "Invalid Navier-Stokes Brinkman data");
      brinkman.insert_or_assign(attribute, BrinkmanData{value, target_});
      ++revision;
   }

   int IntegrationOrder() const
   {
      const int local_order = velocity.GetMaxElementOrder();
      int order = 0;
      MPI_Allreduce(&local_order, &order, 1, MPI_INT, MPI_MAX, comm);
      const int geometry_order = flow::detail::StokesGeometryOrder(mesh);
      detail::Check(comm, geometry_order > 0, "Unsupported Navier-Stokes coordinate geometry");
      detail::CheckReplicated(comm, {static_cast<real_t>(geometry_order)});
      return flow::detail::StokesIntegrationOrder(order, geometry_order, mesh.Dimension(), integration_order);
   }
};

FlowProblem::FlowProblem(ParFiniteElementSpace &velocity,
      ParFiniteElementSpace &pressure, real_t viscosity) : impl(new Impl(velocity, pressure))
{
   detail::Check(impl->comm, std::isfinite(viscosity) && viscosity > 0.0,
                 "Invalid Navier-Stokes viscosity");
   impl->owned_viscosity = std::make_unique<ConstantCoefficient>(viscosity);
   impl->viscosity = impl->owned_viscosity.get();
   impl->gradient = &impl->zero_gradient;
}

FlowProblem::FlowProblem(ParFiniteElementSpace &velocity,
      ParFiniteElementSpace &pressure, Coefficient &viscosity) : impl(new Impl(velocity, pressure))
{ impl->viscosity = &viscosity; }

FlowProblem::~FlowProblem() = default;

void FlowProblem::SetBoundaryConditions(const BoundaryConditions &conditions)
{
   impl->CheckCurrent();
   impl->boundary = conditions;
   impl->CheckBoundary();
   ++impl->revision;
}

void FlowProblem::SetPrescribedVelocityTrueDofs(const Array<int> &dofs,
      VectorCoefficient &values)
{
   impl->CheckCurrent();
   bool valid = values.GetVDim() == impl->velocity.GetVDim();
   for (int dof : dofs) { valid = valid && dof >= 0 && dof < impl->velocity.GetTrueVSize(); }
   const auto *collection = static_cast<const H1_FECollection *>(impl->velocity.FEColl());
   detail::Check(impl->comm, valid && collection->GetBasisType() != BasisType::Positive,
                 "Prescribed velocity DOFs require valid owned true indices and a nodal vector coefficient");
   impl->prescribed_dofs = dofs;
   impl->prescribed_dofs.Sort();
   impl->prescribed_dofs.Unique();
   impl->prescribed_values = &values;
   ++impl->revision;
}

void FlowProblem::ClearPrescribedVelocityTrueDofs()
{
   impl->CheckCurrent();
   impl->prescribed_dofs.SetSize(0);
   impl->prescribed_values = nullptr;
   ++impl->revision;
}

void FlowProblem::SetAuxiliaryVelocityConstraints(VelocityConstraintSelector selector)
{
   impl->CheckCurrent();
   impl->auxiliary_constraints = std::move(selector);
   ++impl->revision;
}

const Array<int> &FlowProblem::GetPrescribedVelocityTrueDofs() const
{ impl->CheckCurrent(); return impl->prescribed_dofs; }

VectorCoefficient *FlowProblem::GetPrescribedVelocityValues() const
{ impl->CheckCurrent(); return impl->prescribed_values; }

void FlowProblem::GetAuxiliaryVelocityConstraints(ParFiniteElementSpace &space,
      Array<int> &dofs) const
{
   impl->CheckCurrent();
   detail::CheckReplicated(impl->comm, {impl->auxiliary_constraints ? 1.0 : 0.0,
                                       impl->prescribed_values ? 1.0 : 0.0});
   detail::Check(impl->comm, !impl->prescribed_values || impl->auxiliary_constraints,
                 "List-based velocity constraints require an auxiliary level constraint selector");
   dofs.SetSize(0);
   if (impl->auxiliary_constraints) { impl->auxiliary_constraints(space, dofs); }
   bool valid = true;
   for (int dof : dofs) { valid = valid && dof >= 0 && dof < space.GetTrueVSize(); }
   detail::Check(impl->comm, valid, "Invalid auxiliary velocity true DOF index");
   dofs.Sort(); dofs.Unique();
}

void FlowProblem::SetBodyForce(int attribute, Direction direction, real_t value)
{ impl->SetForce(attribute, static_cast<int>(direction), ScalarData(value)); }
void FlowProblem::SetBodyForce(int attribute, Direction direction, Coefficient &value)
{ impl->SetForce(attribute, static_cast<int>(direction), ScalarData(value)); }
void FlowProblem::SetBodyForce(int attribute, VectorCoefficient &value)
{
   impl->CheckAttribute(attribute, false);
   detail::Check(impl->comm, value.GetVDim() == impl->mesh.Dimension(),
                 "Invalid Navier-Stokes component or coefficient dimension");
   SetVector(impl->forces, attribute, value);
   ++impl->revision;
}

void FlowProblem::SetBrinkman(int attribute, real_t penalty)
{
   detail::Check(impl->comm, std::isfinite(penalty) && penalty >= 0.0,
                 "Invalid Navier-Stokes Brinkman penalty");
   impl->SetDrag(attribute, ScalarData(penalty), nullptr);
}
void FlowProblem::SetBrinkman(int attribute, Coefficient &penalty)
{ impl->SetDrag(attribute, ScalarData(penalty), nullptr); }
void FlowProblem::SetBrinkman(int attribute, real_t penalty, VectorCoefficient &target)
{
   detail::Check(impl->comm, std::isfinite(penalty) && penalty >= 0.0,
                 "Invalid Navier-Stokes Brinkman penalty");
   impl->SetDrag(attribute, ScalarData(penalty), &target);
}
void FlowProblem::SetBrinkman(int attribute, Coefficient &penalty, VectorCoefficient &target)
{ impl->SetDrag(attribute, ScalarData(penalty), &target); }

void FlowProblem::SetBrinkman(Coefficient &penalty)
{
   impl->CheckCurrent();
   impl->brinkman.insert_or_assign(0, BrinkmanData{ScalarData(penalty), nullptr});
   ++impl->revision;
}

void FlowProblem::SetBrinkman(Coefficient &penalty, VectorCoefficient &target)
{
   impl->CheckCurrent();
   detail::Check(impl->comm, target.GetVDim() == impl->velocity.GetVDim(),
                 "Invalid Brinkman target dimension");
   impl->brinkman.insert_or_assign(0, BrinkmanData{ScalarData(penalty), &target});
   ++impl->revision;
}

void FlowProblem::SetViscosityGradient(VectorCoefficient &gradient)
{
   impl->CheckCurrent();
   detail::Check(impl->comm, gradient.GetVDim() == impl->mesh.Dimension(),
                 "Invalid Navier-Stokes viscosity gradient dimension");
   impl->gradient = &gradient;
   ++impl->revision;
}
void FlowProblem::SetStabilization(const Stabilization &stabilization)
{ impl->CheckCurrent(); impl->stabilization = stabilization; ++impl->revision; }
void FlowProblem::SetPressureGauge(const PressureGauge &gauge)
{
   impl->CheckCurrent();
   detail::Check(impl->comm, !gauge.GetMean() || std::isfinite(*gauge.GetMean()),
                 "Invalid Navier-Stokes pressure mean");
   impl->gauge = gauge;
   ++impl->revision;
}
void FlowProblem::SetIntegrationOrder(int order)
{
   impl->CheckCurrent();
   detail::Check(impl->comm, order == -1 || order > 0, "Invalid Navier-Stokes integration order");
   impl->integration_order = order;
   ++impl->revision;
}
void FlowProblem::NotifyCoefficientsChanged()
{ impl->CheckCurrent(); ++impl->revision; }

void FlowProblem::Validate() const
{
   impl->CheckCurrent();
   std::vector<real_t> signature{static_cast<real_t>(impl->stabilization.index()),
                               static_cast<real_t>(impl->integration_order),
                               static_cast<real_t>(impl->gauge.GetMode()),
                               impl->gauge.GetMean().value_or(0.0),
                               impl->gradient ? 1.0 : 0.0,
                               impl->prescribed_values ? 1.0 : 0.0,
                               impl->auxiliary_constraints ? 1.0 : 0.0};
   if (const auto *options = std::get_if<CIPOptions>(&impl->stabilization)) { signature.push_back(options->gamma); }
   if (const auto *options = std::get_if<PSPGOptions>(&impl->stabilization)) { signature.push_back(options->tau_constant); }
   if (const auto *options = std::get_if<GLSOptions>(&impl->stabilization)) { signature.push_back(options->tau_constant); }
   signature.push_back(impl->owned_viscosity ? 1.0 : 0.0);
   if (impl->owned_viscosity) { signature.push_back(impl->owned_viscosity->constant); }
   AppendComponents(impl->forces, signature);
   signature.push_back(static_cast<real_t>(impl->brinkman.size()));
   for (const auto &entry : impl->brinkman)
   {
      signature.push_back(entry.first);
      entry.second.penalty.AppendSignature(signature);
      signature.push_back(entry.second.target ? entry.second.target->GetVDim() : 0);
   }
   detail::CheckReplicated(impl->comm, signature);
   impl->CheckBoundary();
   impl->CheckComponents(impl->forces, false);
   const bool unstabilized = std::holds_alternative<NoStabilization>(impl->stabilization);
   const bool residual_stabilization = std::holds_alternative<PSPGOptions>(impl->stabilization) ||
                                       std::holds_alternative<GLSOptions>(impl->stabilization);
   bool valid_scale = true;
   if (const auto *options = std::get_if<CIPOptions>(&impl->stabilization)) { valid_scale = options->gamma > 0.0; }
   if (const auto *options = std::get_if<PSPGOptions>(&impl->stabilization)) { valid_scale = options->tau_constant > 0.0; }
   if (const auto *options = std::get_if<GLSOptions>(&impl->stabilization)) { valid_scale = options->tau_constant > 0.0; }
   detail::Check(impl->comm, valid_scale, "Invalid Navier-Stokes stabilization scale");
   detail::Check(impl->comm, !residual_stabilization || impl->gradient,
                 "Navier-Stokes residual stabilization requires a viscosity gradient");
   const int basis = static_cast<const H1_FECollection *>(impl->velocity.FEColl())->GetBasisType();
   detail::Check(impl->comm, !residual_stabilization ||
                 (basis != BasisType::Positive && basis != BasisType::Serendipity),
                 "Unsupported Navier-Stokes residual-stabilization velocity basis");
   const int local_order = impl->velocity.GetMaxElementOrder();
   int order = 0;
   MPI_Allreduce(&local_order, &order, 1, MPI_INT, MPI_MAX, impl->comm);
   bool valid_elements = order >= 2;
   for (int element = 0; element < impl->mesh.GetNE(); ++element)
   {
      const auto geometry = impl->mesh.GetElementGeometry(element);
      valid_elements = valid_elements && impl->velocity.GetElementOrder(element) == order &&
                       impl->pressure.GetElementOrder(element) == order - (unstabilized ? 1 : 0) &&
                       (geometry == Geometry::TRIANGLE || geometry == Geometry::SQUARE ||
                        geometry == Geometry::TETRAHEDRON || geometry == Geometry::CUBE);
   }
   detail::Check(impl->comm, valid_elements, "Unsupported Navier-Stokes element orders or pair");
   const int quadrature_order = impl->IntegrationOrder();
   bool valid_material = true;
   Vector force(impl->mesh.Dimension()), target(impl->mesh.Dimension()), gradient(impl->mesh.Dimension());
   for (int element = 0; element < impl->mesh.GetNE(); ++element)
   {
      auto &transformation = *impl->mesh.GetElementTransformation(element);
      const auto &rule = IntRules.Get(impl->mesh.GetElementGeometry(element), quadrature_order);
      for (int point_index = 0; point_index < rule.GetNPoints(); ++point_index)
      {
         const auto &point = rule.IntPoint(point_index);
         transformation.SetIntPoint(&point);
         const real_t viscosity = impl->viscosity->Eval(transformation, point);
         const real_t penalty = impl->penalty.Eval(transformation, point);
         impl->force.Eval(force, transformation, point);
         impl->target.Eval(target, transformation, point);
         if (impl->gradient) { impl->gradient->Eval(gradient, transformation, point); }
         valid_material = valid_material && std::isfinite(viscosity) && viscosity > 0.0 &&
                          std::isfinite(penalty) && penalty >= 0.0 && Finite(force) && Finite(target) &&
                          (!impl->gradient || Finite(gradient));
      }
   }
   detail::Check(impl->comm, valid_material, "Invalid sampled Navier-Stokes material or force data");
   bool valid_boundary = true;
   for (int element = 0; element < impl->mesh.GetNBE(); ++element)
   {
      auto &transformation = *impl->mesh.GetBdrElementTransformation(element);
      const auto &rule = IntRules.Get(impl->mesh.GetBdrElementGeometry(element), quadrature_order);
      for (int point_index = 0; point_index < rule.GetNPoints(); ++point_index)
      {
         const auto &point = rule.IntPoint(point_index);
         transformation.SetIntPoint(&point);
         for (const auto *data : {&impl->boundary.GetVelocity(), &impl->boundary.GetTraction()})
         {
            for (const auto &entry : *data)
            {
               if (entry.first.first == transformation.Attribute)
               {
                  valid_boundary = std::isfinite(entry.second.Eval(transformation, point)) && valid_boundary;
               }
            }
         }
         const auto found = impl->boundary.GetPressure().find(transformation.Attribute);
         if (found != impl->boundary.GetPressure().end())
         {
            valid_boundary = std::isfinite(found->second.Eval(transformation, point)) && valid_boundary;
         }
      }
   }
   detail::Check(impl->comm, valid_boundary, "Nonfinite sampled Navier-Stokes boundary data");
}

void FlowProblem::CheckSpacesUnchanged() const { impl->CheckCurrent(); }
std::uint64_t FlowProblem::GetRevision() const { return impl->revision; }
MPI_Comm FlowProblem::GetComm() const { return impl->comm; }
ParFiniteElementSpace &FlowProblem::GetVelocitySpace() const { return impl->velocity; }
ParFiniteElementSpace &FlowProblem::GetPressureSpace() const { return impl->pressure; }
Coefficient &FlowProblem::GetViscosity() const { return *impl->viscosity; }
VectorCoefficient *FlowProblem::GetViscosityGradient() const { return impl->gradient; }
VectorCoefficient &FlowProblem::GetBodyForce() const { return impl->force; }
Coefficient &FlowProblem::GetBrinkmanPenalty() const { return impl->penalty; }
VectorCoefficient &FlowProblem::GetBrinkmanTarget() const { return impl->target; }
const BoundaryConditions &FlowProblem::GetBoundaryConditions() const { return impl->boundary; }
const Stabilization &FlowProblem::GetStabilization() const { return impl->stabilization; }
const PressureGauge &FlowProblem::GetPressureGauge() const { return impl->gauge; }
void FlowProblem::ResolvePressureGauge(bool active) const
{ impl->resolved_gauge = active; impl->gauge_revision = impl->revision; }
bool FlowProblem::HasPressureGauge() const
{
   if (impl->gauge.GetMode() != PressureGauge::Mode::Auto) { return impl->gauge.GetMean().has_value(); }
   impl->CheckCurrent();
   detail::Check(impl->comm, impl->resolved_gauge && impl->gauge_revision == impl->revision,
                 "Assemble the current flow problem before querying an automatic pressure gauge");
   return *impl->resolved_gauge;
}
int FlowProblem::GetIntegrationOrder() const
{ impl->CheckCurrent(); return impl->IntegrationOrder(); }

int FlowProblem::GetConvectionIntegrationOrder() const
{
   impl->CheckCurrent();
   if (impl->integration_order > 0) { return impl->integration_order; }
   const int local_order = impl->velocity.GetMaxElementOrder();
   int order = 0;
   MPI_Allreduce(&local_order, &order, 1, MPI_INT, MPI_MAX, impl->comm);
   const int geometry = flow::detail::StokesGeometryOrder(impl->mesh);
   return std::max(impl->IntegrationOrder(), 3 * order + 2 * impl->mesh.Dimension() * (geometry - 1));
}

int FlowProblem::GetStabilizationIntegrationOrder() const
{
   impl->CheckCurrent();
   if (impl->integration_order > 0 || !std::holds_alternative<GLSOptions>(impl->stabilization))
   { return GetConvectionIntegrationOrder(); }
   const int local_order = impl->velocity.GetMaxElementOrder();
   int order = 0;
   MPI_Allreduce(&local_order, &order, 1, MPI_INT, MPI_MAX, impl->comm);
   const int geometry = flow::detail::StokesGeometryOrder(impl->mesh);
   return std::max(impl->IntegrationOrder(), 4 * order + 2 * impl->mesh.Dimension() * (geometry - 1));
}

TrueDofLayout::TrueDofLayout(const FlowProblem &problem_)
   : problem(&problem_), revision(problem_.GetRevision()),
     global_velocity_size(0), global_pressure_size(0), gauge(false)
{
   problem->Validate();
   gauge = problem->HasPressureGauge();
   int rank = 0;
   MPI_Comm_rank(GetComm(), &rank);
   offsets.SetSize(gauge ? 4 : 3);
   offsets[0] = 0;
   offsets[1] = problem->GetVelocitySpace().GetTrueVSize();
   const auto total = static_cast<long long>(offsets[1]) + problem->GetPressureSpace().GetTrueVSize() +
                      (gauge && rank == 0 ? 1 : 0);
   detail::Check(GetComm(), total <= std::numeric_limits<int>::max(), "Navier-Stokes local layout overflows int");
   offsets[2] = offsets[1] + problem->GetPressureSpace().GetTrueVSize();
   if (gauge) { offsets[3] = static_cast<int>(total); }
   global_velocity_size = problem->GetVelocitySpace().GlobalTrueVSize();
   global_pressure_size = problem->GetPressureSpace().GlobalTrueVSize();
}

void TrueDofLayout::CheckCurrent() const
{
   problem->CheckSpacesUnchanged();
   detail::Check(GetComm(), revision == problem->GetRevision(), "Stale Navier-Stokes layout or linearization");
}

void TrueDofLayout::ValidateVector(const Vector &vector) const
{
   CheckCurrent();
   detail::Check(GetComm(), vector.Size() == LocalSize() && Finite(vector),
                 "Invalid Navier-Stokes algebraic vector");
}

void FlowState::Pack(const TrueDofLayout &layout, Vector &state) const
{
   layout.CheckCurrent();
   detail::Check(layout.GetComm(), &state != &velocity && &state != &pressure,
                 "Navier-Stokes packed output must not alias primal fields");
   const auto &problem = layout.GetProblem();
   detail::Check(layout.GetComm(), velocity.ParFESpace() == &problem.GetVelocitySpace() &&
                 pressure.ParFESpace() == &problem.GetPressureSpace(),
                 "Incompatible Navier-Stokes supplied-state spaces");
   detail::Check(layout.GetComm(), gauge_multiplier.has_value() == layout.HasGauge(),
                 "Inconsistent Navier-Stokes supplied-state gauge");
   detail::CheckReplicated(layout.GetComm(), {gauge_multiplier.value_or(0.0)});
   detail::Check(layout.GetComm(), Finite(velocity) && Finite(pressure),
                 "Nonfinite Navier-Stokes supplied state");
   Vector velocity_true, pressure_true;
   velocity.GetTrueDofs(velocity_true);
   pressure.GetTrueDofs(pressure_true);
   state.SetSize(layout.LocalSize());
   const auto &offsets = layout.GetOffsets();
   for (int index = 0; index < velocity_true.Size(); ++index) { state(index) = velocity_true(index); }
   for (int index = 0; index < pressure_true.Size(); ++index) { state(offsets[1] + index) = pressure_true(index); }
   if (layout.HasGauge() && offsets[3] > offsets[2]) { state(offsets[2]) = *gauge_multiplier; }
   layout.ValidateVector(state);
}

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

namespace detail
{

ResidualStabilizationIntegrator::ResidualStabilizationIntegrator(
   const FlowProblem &problem_, ExperimentalResidualOptions options_,
   bool convection_, bool exact_, int order_)
   : problem(problem_), options(options_), convection(convection_), exact(exact_), order(order_) { }

void ResidualStabilizationIntegrator::AssembleElementVector(
   const Array<const FiniteElement *> &elements, ElementTransformation &transformation,
   const Array<const Vector *> &state, const Array<Vector *> &residual)
{ Assemble(elements, transformation, state, &residual, nullptr); }

void ResidualStabilizationIntegrator::AssembleElementGrad(
   const Array<const FiniteElement *> &elements, ElementTransformation &transformation,
   const Array<const Vector *> &state, const Array2D<DenseMatrix *> &jacobian)
{ Assemble(elements, transformation, state, nullptr, &jacobian); }

void ResidualStabilizationIntegrator::Assemble(
   const Array<const FiniteElement *> &elements, ElementTransformation &transformation,
   const Array<const Vector *> &state, const Array<Vector *> *residual,
   const Array2D<DenseMatrix *> *jacobian)
{
   const auto &velocity_element = *elements[0], &pressure_element = *elements[1];
   const int dimension = velocity_element.GetDim(), nodes = velocity_element.GetDof();
   const int sizes[2] = {dimension * nodes, pressure_element.GetDof()};
   const int total = sizes[0] + sizes[1];
   const bool gls = std::holds_alternative<GLSOptions>(problem.GetStabilization());
   const real_t scale = gls ? std::get<GLSOptions>(problem.GetStabilization()).tau_constant :
                        std::get<PSPGOptions>(problem.GetStabilization()).tau_constant;
   for (int row = 0; row < 2; ++row)
   {
      if (residual) { (*residual)[row]->SetSize(sizes[row]); *(*residual)[row] = 0.0; }
      if (jacobian)
      {
         for (int column = 0; column < 2; ++column)
         { (*jacobian)(row, column)->SetSize(sizes[row], sizes[column]); *(*jacobian)(row, column) = 0.0; }
      }
   }
   const auto &rule = IntRules.Get(velocity_element.GetGeomType(), order);
   Vector shape(nodes), velocity(dimension), target(dimension), force(dimension), strong(dimension);
   Vector velocity_average_derivative(sizes[0]); velocity_average_derivative = 0.0;
   real_t volume = 0.0, viscosity_integral = 0.0, speed_integral = 0.0, drag_integral = 0.0;
   for (int point = 0; point < rule.GetNPoints(); ++point)
   {
      const auto &ip = rule.IntPoint(point);
      transformation.SetIntPoint(&ip);
      const real_t weight = ip.weight * transformation.Weight();
      velocity_element.CalcShape(ip, shape);
      const real_t viscosity = problem.GetViscosity().Eval(transformation, ip);
      const real_t drag = problem.GetBrinkmanPenalty().Eval(transformation, ip);
      MFEM_VERIFY(std::isfinite(viscosity) && viscosity > 0.0 && std::isfinite(drag) && drag >= 0.0 &&
                  std::isfinite(weight) && transformation.Jacobian().Det() > 0.0,
                  "Invalid experimental stabilization coefficient or geometry sample");
      volume += weight;
      viscosity_integral += weight * viscosity;
      drag_integral += weight * drag * drag;
      if (convection)
      {
         for (int component = 0; component < dimension; ++component)
         {
            real_t value = 0.0;
            for (int node = 0; node < nodes; ++node) { value += shape(node) * (*state[0])(component * nodes + node); }
            speed_integral += weight * value * value;
            for (int node = 0; node < nodes; ++node)
            { velocity_average_derivative(component * nodes + node) += weight * value * shape(node); }
         }
      }
   }
   const real_t size = problem.GetVelocitySpace().GetParMesh()->GetElementSize(transformation.ElementNo, 1);
   const real_t degree = velocity_element.GetOrder() + 1.0;
   const real_t tau_zero = scale * size * size * volume / (viscosity_integral * std::pow(degree, 4));
   const real_t advective_factor = convection ? std::pow(options.advection_scale * degree / size, 2) : 0.0;
   MFEM_VERIFY(volume > 0.0 && viscosity_integral > 0.0 && drag_integral >= 0.0,
               "Invalid experimental stabilization cell averages");
   const real_t tau = 1.0 / std::sqrt(1.0 / (tau_zero * tau_zero) +
                                     advective_factor * speed_integral / volume +
                                     options.reaction_scale * options.reaction_scale * drag_integral / volume);
   velocity_average_derivative *= exact ? -tau * tau * tau * advective_factor / volume : 0.0;
   Vector zero(dimension); zero = 0.0;
   VectorConstantCoefficient zero_gradient(zero);
   auto *viscosity_gradient = problem.GetViscosityGradient();
   flow::detail::StokesGeometryContext geometry;
   geometry.Prepare(transformation);
   DenseMatrix viscous, gradient(nodes, dimension), pressure_gradient(sizes[1], dimension);
   DenseMatrix velocity_gradient(dimension), test(dimension, total), derivative(dimension, total);
   Vector unscaled(total); unscaled = 0.0;
   for (int point = 0; point < rule.GetNPoints(); ++point)
   {
      const auto &ip = rule.IntPoint(point);
      transformation.SetIntPoint(&ip);
      const real_t weight = ip.weight * transformation.Weight();
      velocity_element.CalcShape(ip, shape);
      velocity_element.CalcPhysDShape(transformation, gradient);
      pressure_element.CalcPhysDShape(transformation, pressure_gradient);
      flow::EvaluateMomentumBasis(velocity_element, transformation, problem.GetViscosity(),
                                   viscosity_gradient ? *viscosity_gradient : zero_gradient, viscous, geometry);
      const real_t drag = problem.GetBrinkmanPenalty().Eval(transformation, ip);
      problem.GetBodyForce().Eval(force, transformation, ip);
      problem.GetBrinkmanTarget().Eval(target, transformation, ip);
      velocity = 0.0; velocity_gradient = 0.0;
      for (int component = 0; component < dimension; ++component)
      {
         for (int node = 0; node < nodes; ++node)
         {
            const real_t value = (*state[0])(component * nodes + node);
            velocity(component) += shape(node) * value;
            for (int axis = 0; axis < dimension; ++axis) { velocity_gradient(component, axis) += gradient(node, axis) * value; }
         }
      }
      viscous.Mult(*state[0], strong);
      for (int axis = 0; axis < dimension; ++axis)
      {
         strong(axis) += drag * (velocity(axis) - target(axis)) - force(axis);
         for (int node = 0; node < sizes[1]; ++node) { strong(axis) += pressure_gradient(node, axis) * (*state[1])(node); }
         if (convection)
         { for (int component = 0; component < dimension; ++component) { strong(axis) += velocity(component) * velocity_gradient(axis, component); } }
      }
      for (int column = 0; column < total; ++column)
      {
         for (int axis = 0; axis < dimension; ++axis)
         {
            if (column >= sizes[0])
            { test(axis, column) = derivative(axis, column) = pressure_gradient(column - sizes[0], axis); continue; }
            const int component = column / nodes, node = column % nodes;
            real_t oseen = viscous(axis, column);
            if (axis == component)
            {
               oseen += drag * shape(node);
               if (convection)
               { for (int direction = 0; direction < dimension; ++direction) { oseen += velocity(direction) * gradient(node, direction); } }
            }
            test(axis, column) = gls ? oseen : 0.0;
            derivative(axis, column) = oseen + (convection && exact ? shape(node) * velocity_gradient(axis, component) : 0.0);
         }
      }
      for (int row = 0; row < total; ++row)
      {
         const int row_block = row < sizes[0] ? 0 : 1, local_row = row < sizes[0] ? row : row - sizes[0];
         real_t value = 0.0;
         for (int axis = 0; axis < dimension; ++axis) { value += test(axis, row) * strong(axis); }
         unscaled(row) += weight * value;
         if (residual) { (*(*residual)[row_block])(local_row) += weight * tau * value; }
         if (!jacobian) { continue; }
         for (int column = 0; column < total; ++column)
         {
            const int column_block = column < sizes[0] ? 0 : 1, local_column = column < sizes[0] ? column : column - sizes[0];
            real_t product = 0.0;
            for (int axis = 0; axis < dimension; ++axis) { product += test(axis, row) * derivative(axis, column); }
            if (gls && convection && exact && row_block == 0 && column_block == 0)
            { product += strong(row / nodes) * shape(column % nodes) * gradient(row % nodes, column / nodes); }
            (*(*jacobian)(row_block, column_block))(local_row, local_column) += weight * tau * product;
         }
      }
   }
   if (jacobian)
   {
      for (int row = 0; row < total; ++row)
      {
         const int row_block = row < sizes[0] ? 0 : 1, local_row = row < sizes[0] ? row : row - sizes[0];
         for (int column = 0; column < sizes[0]; ++column)
         { (*(*jacobian)(row_block, 0))(local_row, column) += unscaled(row) * velocity_average_derivative(column); }
      }
   }
}

}

namespace detail
{

void ConstrainCorrectionTransfer(HypreParMatrix &transfer, const Array<int> &fine,
                                 const Array<int> &coarse, bool check_compatibility)
{
   Vector ones(transfer.Width()), original(transfer.Height()), remaining(transfer.Height());
   ones = 1.0;
   if (check_compatibility) { transfer.AbsMult(ones, original); }
   std::unique_ptr<HypreParMatrix> removed(transfer.EliminateCols(coarse));
   if (check_compatibility)
   {
      transfer.AbsMult(ones, remaining);
      int valid = 1, global = 0;
      for (int dof : fine)
      { valid = valid && remaining(dof) <= 1e-12 * std::max(real_t(1.0), original(dof)); }
      MPI_Allreduce(&valid, &global, 1, MPI_INT, MPI_MIN, transfer.GetComm());
      MFEM_VERIFY(global, "Incompatible auxiliary constraints: constrain every coarse basis influencing a prescribed fine DOF");
   }
   transfer.EliminateRows(fine);
}

P1AMGPreconditioner::P1AMGPreconditioner(
   const HypreParMatrix &matrix_, ParFiniteElementSpace &space,
   const std::vector<Array<int>> &boundary_markers, int smoothing_steps,
   bool elasticity, bool refine, const CoarseAssembler &assemble,
   const Array<int> *fine_constraints, const ConstraintSelector &coarse_constraints,
   bool nonsymmetric)
   : Solver(matrix_.Height()), matrix(matrix_),
     coarse_collection(1, space.GetParMesh()->Dimension()),
     coarse_space(space.GetParMesh(), &coarse_collection, space.GetVDim(),
                  space.GetOrdering()),
     transfer(coarse_space, space, true),
     prolongation(dynamic_cast<HypreParMatrix*>(transfer.GetTrueTransferOperator())),
     smoother(matrix, HypreSmoother::l1Jacobi, smoothing_steps),
     residual(height), correction(height),
     coarse_rhs(coarse_space.GetTrueVSize()), coarse_solution(coarse_rhs.Size())
{
   MFEM_VERIFY(prolongation && smoothing_steps > 0, "Invalid P1-AMG setup");
   MFEM_VERIFY(boundary_markers.empty() ||
               boundary_markers.size() == static_cast<size_t>(space.GetVDim()),
               "P1-AMG needs one boundary marker per component");
   Array<int> fine_essential, coarse_essential, selected;
   for (int component = 0; component < static_cast<int>(boundary_markers.size()); ++component)
   {
      space.GetEssentialTrueDofs(boundary_markers[component], selected, component);
      fine_essential.Append(selected);
      coarse_space.GetEssentialTrueDofs(boundary_markers[component], selected, component);
      coarse_essential.Append(selected);
   }
   if (fine_constraints) { fine_essential.Append(*fine_constraints); }
   if (coarse_constraints)
   {
      coarse_constraints(coarse_space, selected);
      coarse_essential.Append(selected);
   }
   fine_essential.Sort(); fine_essential.Unique();
   coarse_essential.Sort(); coarse_essential.Unique();
   if (fine_constraints) { fine_identity = fine_essential; }
   ConstrainCorrectionTransfer(*prolongation, fine_essential, coarse_essential, fine_constraints != nullptr);
   if (assemble) { coarse_matrix = assemble(coarse_space); }
   else { coarse_matrix.reset(RAP(&matrix, prolongation)); }
   MFEM_VERIFY(coarse_matrix && coarse_matrix->Height() == coarse_space.GetTrueVSize() &&
               coarse_matrix->Width() == coarse_space.GetTrueVSize(),
               "Invalid P1-AMG coarse matrix");
   delete coarse_matrix->EliminateRowsCols(coarse_essential);
   Vector coarse_diagonal;
   coarse_matrix->GetDiag(coarse_diagonal);
   int valid = 1, global_valid = 0;
   for (int entry = 0; entry < coarse_diagonal.Size(); ++entry)
   {
      valid = valid && std::isfinite(coarse_diagonal(entry)) &&
              (nonsymmetric ? coarse_diagonal(entry) != 0.0 : coarse_diagonal(entry) > 0.0);
   }
   MPI_Allreduce(&valid, &global_valid, 1, MPI_INT, MPI_MIN, space.GetComm());
   MFEM_VERIFY(global_valid, "P1-AMG constraints leave a zero or invalid coarse correction mode");
   coarse_amg = std::make_unique<HypreBoomerAMG>(*coarse_matrix);
   if (nonsymmetric)
   { coarse_amg->SetSystemsOptions(coarse_space.GetVDim(), coarse_space.GetOrdering() == Ordering::byNODES); }
   else if (elasticity) { coarse_amg->SetElasticityOptions(&coarse_space, refine); }
   coarse_amg->SetMaxIter(1);
   coarse_amg->SetTol(0.0);
   coarse_amg->SetPrintLevel(0);
   coarse_amg->iterative_mode = false;
   smoother.iterative_mode = false;
   coarse_rhs = 0.0;
   coarse_solution = 0.0;
   coarse_amg->Setup(coarse_rhs, coarse_solution);
}

void P1AMGPreconditioner::SetOperator(const Operator &)
{
   MFEM_ABORT("P1AMGPreconditioner has fixed operators");
}

void P1AMGPreconditioner::Mult(const Vector &input, Vector &output) const
{
   output.SetSize(height);
   smoother.Mult(input, output);
   matrix.Mult(output, residual);
   subtract(input, residual, residual);
   prolongation->MultTranspose(residual, coarse_rhs);
   coarse_amg->Mult(coarse_rhs, coarse_solution);
   prolongation->Mult(coarse_solution, correction);
   output += correction;
   matrix.Mult(output, residual);
   subtract(input, residual, residual);
   smoother.Mult(residual, correction);
   output += correction;
   for (int dof : fine_identity) { output(dof) = input(dof); }
}

}

namespace detail
{

namespace
{

class InverseViscosity : public Coefficient
{
   Coefficient &viscosity;
public:
   explicit InverseViscosity(Coefficient &coefficient) : viscosity(coefficient) { }
   real_t Eval(ElementTransformation &transformation, const IntegrationPoint &point) override
   { return 1.0 / viscosity.Eval(transformation, point); }
};

class PressureMassIntegrator : public MassIntegrator
{
   int order;
public:
   PressureMassIntegrator(Coefficient &coefficient, int order_) : MassIntegrator(coefficient), order(order_) { }
   void AssembleElementMatrix(const FiniteElement &element, ElementTransformation &transformation,
                              DenseMatrix &matrix) override
   {
      SetIntRule(&IntRules.Get(element.GetGeomType(), order));
      MassIntegrator::AssembleElementMatrix(element, transformation, matrix);
   }
};

class VelocityMassIntegrator : public VectorMassIntegrator
{
   int order;
public:
   explicit VelocityMassIntegrator(int order_) : order(order_) { }
   void AssembleElementMatrix(const FiniteElement &element, ElementTransformation &transformation,
                              DenseMatrix &matrix) override
   {
      SetIntRule(&IntRules.Get(element.GetGeomType(), order));
      VectorMassIntegrator::AssembleElementMatrix(element, transformation, matrix);
   }
};

template <class Integrator>
class PressureQuadratureIntegrator : public Integrator
{
   int order;
public:
   template <class... Arguments>
   PressureQuadratureIntegrator(int order_, Arguments &&... arguments)
      : Integrator(std::forward<Arguments>(arguments)...), order(order_) { }
   void AssembleElementMatrix(const FiniteElement &element, ElementTransformation &transformation,
                              DenseMatrix &matrix) override
   {
      this->SetIntRule(&IntRules.Get(element.GetGeomType(), order));
      Integrator::AssembleElementMatrix(element, transformation, matrix);
   }
};

void ConstrainPCDMatrix(HypreParMatrix &matrix, const Array<int> &essential)
{
   Vector diagonal, scaling(matrix.Height()); matrix.GetDiag(diagonal); scaling = 1.0;
   std::unique_ptr<HypreParMatrix> removed(matrix.EliminateRowsCols(essential));
   for (int dof : essential) { scaling(dof) = diagonal(dof); }
   matrix.ScaleRows(scaling);
}

class PressureInflowCoefficient : public Coefficient
{
   VectorCoefficient &velocity;
   Vector value, normal;
public:
   explicit PressureInflowCoefficient(VectorCoefficient &velocity_)
      : velocity(velocity_), value(velocity_.GetVDim()), normal(value.Size()) { }
   real_t Eval(ElementTransformation &transformation, const IntegrationPoint &point) override
   {
      transformation.SetIntPoint(&point);
      velocity.Eval(value, transformation, point);
      CalcOrtho(transformation.Jacobian(), normal);
      const real_t measure = normal.Norml2();
      MFEM_VERIFY(std::isfinite(measure) && measure > 0.0, "Invalid PCD boundary measure");
      return std::max(real_t(0.0), -(value * normal) / measure);
   }
};

std::unique_ptr<Solver> PCDSymmetricInverse(const HypreParMatrix &matrix, const IterativeOptions &options)
{
   Vector diagonal; matrix.GetDiag(diagonal);
   bool valid = true;
   for (int entry = 0; entry < diagonal.Size(); ++entry)
   { valid = valid && std::isfinite(diagonal(entry)) && diagonal(entry) > 0.0; }
   Check(matrix.GetComm(), valid, "PCD requires positive finite mass and Laplacian diagonals");
   if (options.pcd_solver == PCDPressureSolver::BoomerAMG)
   { return std::make_unique<FixedAMGInverse>(matrix, options.pcd_amg_cycles, nullptr, true); }
#ifdef MFEM_USE_MUMPS
   auto inverse = std::make_unique<MUMPSSolver>(matrix.GetComm());
   inverse->SetMatrixSymType(MUMPSSolver::SYMMETRIC_POSITIVE_DEFINITE);
   inverse->SetPrintLevel(0); inverse->SetOperator(matrix);
   return inverse;
#else
   MFEM_ABORT("PCD MUMPS support is unavailable");
   return {};
#endif
}

}

PCDOperators AssemblePCDOperators(const FlowProblem &problem,
                                 const FlowSolverOptions &options, const Vector &state)
{
   const auto communicator = problem.GetComm();
   Check(communicator, std::holds_alternative<NoStabilization>(problem.GetStabilization()),
         "Experimental PCD requires unstabilized Taylor-Hood flow");
   Check(communicator, !problem.GetPrescribedVelocityValues(),
         "PCD requires boundary-attribute velocity conditions, not explicit DOF constraints");
   auto &pressure_space = problem.GetPressureSpace();
   auto &velocity_space = problem.GetVelocitySpace();
   const auto &attributes = pressure_space.GetParMesh()->bdr_attributes;
   Array<int> marker(attributes.Size() ? attributes.Max() : 0); marker = 0;
   Array<int> robin_marker(marker.Size()); robin_marker = 0;
   bool valid = true;
   for (int attribute : attributes)
   {
      int prescribed = 0;
      for (int component = 0; component < velocity_space.GetVDim(); ++component)
      { prescribed += problem.GetBoundaryConditions().GetVelocity().count({attribute, component}); }
      valid = valid && (prescribed == 0 || prescribed == velocity_space.GetVDim());
      if (prescribed == 0) { marker[attribute - 1] = 1; }
      if (prescribed == velocity_space.GetVDim()) { robin_marker[attribute - 1] = 1; }
   }
   Check(communicator, valid, "PCD requires fully prescribed or traction-only velocity boundary attributes");
   Check(communicator, state.Size() >= velocity_space.GetTrueVSize(), "PCD requires the current velocity state");
   Vector velocity_values(state.GetData(), velocity_space.GetTrueVSize());
   ParGridFunction velocity(&velocity_space); velocity.SetFromTrueDofs(velocity_values);
   VectorGridFunctionCoefficient advection(&velocity);
   PressureInflowCoefficient inflow(advection);
   ConstantCoefficient one(1.0);
   const int order = options.convection ? std::max(problem.GetIntegrationOrder(), problem.GetConvectionIntegrationOrder()) :
                     problem.GetIntegrationOrder();
   ParBilinearForm mass(&pressure_space), laplacian(&pressure_space), transport(&pressure_space);
   mass.AddDomainIntegrator(new PressureMassIntegrator(one, order));
   laplacian.AddDomainIntegrator(new PressureQuadratureIntegrator<DiffusionIntegrator>(order, one));
   transport.AddDomainIntegrator(new PressureQuadratureIntegrator<DiffusionIntegrator>(order, problem.GetViscosity()));
   transport.AddDomainIntegrator(new PressureMassIntegrator(problem.GetBrinkmanPenalty(), order));
   if (options.convection)
   {
      transport.AddDomainIntegrator(new PressureQuadratureIntegrator<ConvectionIntegrator>(order, advection));
      transport.AddBoundaryIntegrator(new PressureQuadratureIntegrator<MassIntegrator>(order, inflow), robin_marker);
   }
   PCDOperators result;
   mass.Assemble(0); mass.Finalize(0); result.mass.reset(mass.ParallelAssemble());
   laplacian.Assemble(0); laplacian.Finalize(0); result.laplacian.reset(laplacian.ParallelAssemble());
   transport.Assemble(0); transport.Finalize(0); result.convection_diffusion.reset(transport.ParallelAssemble());
   pressure_space.GetEssentialTrueDofs(marker, result.essential);
   ConstrainPCDMatrix(*result.laplacian, result.essential);
   ConstrainPCDMatrix(*result.convection_diffusion, result.essential);
   return result;
}

class PCDPressureInverse::Impl
{
public:
   HypreParMatrix pinned;
   const HypreParMatrix &transport;
   Vector constant, moments;
   std::unique_ptr<Solver> laplace_inverse, mass_inverse;
   int pinned_dof = -1, size;
   bool gauge, orientation;
   real_t norm_squared = 1.0, volume = 1.0;

   Impl(const HypreParMatrix &laplacian, const HypreParMatrix &mass, const HypreParMatrix &transport_,
        const Vector &constant_, const Vector &moments_, int gauge_size, bool gauge_,
        const IterativeOptions &options, bool transpose)
      : pinned(laplacian), transport(transport_), constant(constant_), moments(moments_),
        size(laplacian.Height() + gauge_size), gauge(gauge_), orientation(transpose)
   {
      Check(pinned.GetComm(), mass.Height() == pinned.Height() && transport.Height() == pinned.Height() &&
            mass.Width() == pinned.Width() && transport.Width() == pinned.Width(), "Invalid PCD operator dimensions");
      if (gauge)
      {
         Check(pinned.GetComm(), constant.Size() == pinned.Height() && moments.Size() == pinned.Height(),
               "Invalid PCD pressure constant or moments");
         norm_squared = InnerProduct(pinned.GetComm(), constant, constant);
         volume = InnerProduct(pinned.GetComm(), constant, moments);
         Check(pinned.GetComm(), norm_squared > 0.0 && volume > 0.0 && std::isfinite(volume),
               "Invalid PCD pressure constant or moments");
         Vector defect(pinned.Height()); pinned.Mult(constant, defect);
         Check(pinned.GetComm(), std::sqrt(InnerProduct(pinned.GetComm(), defect, defect)) <=
               1e-11 * std::max(real_t(1.0), pinned.FNorm()) * std::sqrt(norm_squared),
               "Unsupported PCD pressure nullspace");
         int rank = 0, ranks = 0, owner = 0;
         MPI_Comm_rank(pinned.GetComm(), &rank); MPI_Comm_size(pinned.GetComm(), &ranks);
         int candidate = pinned.Height() ? rank : ranks;
         MPI_Allreduce(&candidate, &owner, 1, MPI_INT, MPI_MIN, pinned.GetComm());
         Array<int> selected;
         if (rank == owner) { selected.Append(0); pinned_dof = 0; }
         std::unique_ptr<HypreParMatrix> removed(pinned.EliminateRowsCols(selected));
      }
      laplace_inverse = PCDSymmetricInverse(pinned, options);
      mass_inverse = PCDSymmetricInverse(mass, options);
   }

   void Project(Vector &value) const
   {
      if (gauge) { value.Add(-InnerProduct(pinned.GetComm(), constant, value) / norm_squared, constant); }
   }

   void Laplace(const Vector &input, Vector &output, bool transpose) const
   {
      Vector rhs(input); Project(rhs);
      if (pinned_dof >= 0) { rhs(pinned_dof) = 0.0; }
      if (transpose) { laplace_inverse->MultTranspose(rhs, output); }
      else { laplace_inverse->Mult(rhs, output); }
      if (pinned_dof >= 0) { output(pinned_dof) = 0.0; }
      Project(output);
   }

   void Apply(const Vector &input, Vector &output, bool transpose) const
   {
      Vector rhs(input.GetData(), pinned.Height()), compatible(rhs), first(pinned.Height()),
             second(pinned.Height()), pressure(pinned.Height());
      real_t multiplier = 0.0, mean_rhs = 0.0;
      if (gauge)
      {
         multiplier = InnerProduct(pinned.GetComm(), constant, rhs) / volume;
         compatible.Add(-multiplier, moments);
         if (input.Size() > pinned.Height()) { mean_rhs = input(pinned.Height()); }
         MPI_Bcast(&mean_rhs, 1, MPITypeMap<real_t>::mpi_type, 0, pinned.GetComm());
      }
      if (transpose)
      {
         mass_inverse->MultTranspose(compatible, first);
         transport.MultTranspose(first, second);
         Laplace(second, pressure, true);
      }
      else
      {
         Laplace(compatible, first, false);
         transport.Mult(first, second);
         mass_inverse->Mult(second, pressure);
      }
      if (gauge)
      { pressure.Add((mean_rhs - InnerProduct(pinned.GetComm(), moments, pressure)) / volume, constant); }
      output.SetSize(size);
      for (int entry = 0; entry < pressure.Size(); ++entry) { output(entry) = pressure(entry); }
      if (size > pressure.Size()) { output(pressure.Size()) = multiplier; }
   }
};

PCDPressureInverse::PCDPressureInverse(const HypreParMatrix &laplacian, const HypreParMatrix &mass,
   const HypreParMatrix &convection_diffusion, const Vector &constant, const Vector &moments,
   int gauge_size, bool gauge, const IterativeOptions &options, bool transpose)
   : Solver(laplacian.Height() + gauge_size), impl(std::make_unique<Impl>(laplacian, mass,
      convection_diffusion, constant, moments, gauge_size, gauge, options, transpose)) { }

PCDPressureInverse::~PCDPressureInverse() = default;
void PCDPressureInverse::SetOperator(const Operator &) { MFEM_ABORT("Fixed PCD inverse"); }
void PCDPressureInverse::Mult(const Vector &input, Vector &output) const
{ impl->Apply(input, output, impl->orientation); }
void PCDPressureInverse::MultTranspose(const Vector &input, Vector &output) const
{ impl->Apply(input, output, !impl->orientation); }

void BuildSchurOperators(const FlowProblem &problem,
                         const FlowSolverOptions &options,
                         const Array<int> &essential,
                         const Array2D<const HypreParMatrix *> &source,
                         const Array2D<const HypreParMatrix *> &blocks,
                         std::unique_ptr<HypreParMatrix> &pressure_mass,
                         std::unique_ptr<HypreParMatrix> &lsc_middle)
{
   const auto &settings = std::get<IterativeOptions>(options.linear.backend);
   const auto communicator = problem.GetComm();
   Check(communicator, settings.schur != SchurApproximation::ExperimentalPCD,
         "Use AssemblePCDOperators for state-dependent PCD assembly");
   auto &pressure_space = problem.GetPressureSpace();
   if (settings.schur == SchurApproximation::ExperimentalLSC)
   {
      Check(communicator, std::holds_alternative<NoStabilization>(problem.GetStabilization()),
            "Experimental LSC requires unstabilized Taylor-Hood flow");
      ParBilinearForm mass(&problem.GetVelocitySpace());
      mass.AddDomainIntegrator(new VelocityMassIntegrator(problem.GetIntegrationOrder()));
      mass.Assemble(0); mass.Finalize(0);
      std::unique_ptr<HypreParMatrix> mass_matrix(mass.ParallelAssemble());
      Vector diagonal; mass_matrix->GetDiag(diagonal);
      bool valid = true;
      for (int entry = 0; entry < diagonal.Size(); ++entry)
      { valid = valid && std::isfinite(diagonal(entry)) && diagonal(entry) > 0.0; }
      Check(communicator, valid, "LSC requires a finite positive velocity mass diagonal");
      std::unique_ptr<HypreParMatrix> weighted_gradient(source(1, 0)->Transpose());
      weighted_gradient->InvScaleRows(diagonal);
      weighted_gradient->EliminateRows(essential);
      pressure_mass.reset(ParMult(source(1, 0), weighted_gradient.get()));
      lsc_middle.reset(RAP(source(0, 0), weighted_gradient.get()));
   }
   else if (settings.schur == SchurApproximation::WeightedMass)
   {
      InverseViscosity weight(problem.GetViscosity());
      ParBilinearForm form(&pressure_space);
      form.AddDomainIntegrator(new PressureMassIntegrator(weight, problem.GetIntegrationOrder()));
      form.Assemble(0); form.Finalize(0);
      pressure_mass.reset(form.ParallelAssemble());
      if (blocks(1, 1))
      { pressure_mass.reset(Add(1.0, *pressure_mass, 1.0, *blocks(1, 1))); }
   }
   else
   {
      Vector diagonal;
      blocks(0, 0)->GetDiag(diagonal);
      bool valid = true;
      for (int entry = 0; entry < diagonal.Size(); ++entry)
      { valid = valid && std::isfinite(diagonal(entry)) && diagonal(entry) > 0.0; }
      Check(communicator, valid, "Reaction Schur requires strictly positive finite velocity diagonal pivots");
      HypreParMatrix scaled(*blocks(0, 1));
      scaled.InvScaleRows(diagonal);
      pressure_mass.reset(ParMult(blocks(1, 0), &scaled));
      if (blocks(1, 1))
      { pressure_mass.reset(Add(1.0, *blocks(1, 1), -1.0, *pressure_mass)); }
      else { *pressure_mass *= -1.0; }
   }
}

}

namespace detail
{

namespace
{

real_t Norm(MPI_Comm communicator, const Vector &value)
{ return std::sqrt(InnerProduct(communicator, value, value)); }

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

class BlockUpperInverse : public Solver
{
   Solver &velocity, &pressure;
   const Operator &upper, &lower;
   bool forward, diagonal;
public:
   mutable double apply_seconds = 0.0;
   BlockUpperInverse(Solver &velocity_, Solver &pressure_, const Operator &upper_, const Operator &lower_,
                     bool forward_, bool diagonal_)
      : Solver(velocity_.Height() + pressure_.Height()), velocity(velocity_), pressure(pressure_),
        upper(upper_), lower(lower_), forward(forward_), diagonal(diagonal_) { }
   void SetOperator(const Operator &) override { MFEM_ABORT("Fixed block inverse"); }
   void Mult(const Vector &input, Vector &output) const override
   {
      const double started = MPI_Wtime();
      Vector copy(input);
      output.SetSize(height);
      Vector velocity_rhs(copy.GetData(), velocity.Height());
      Vector pressure_rhs(copy.GetData() + velocity.Height(), pressure.Height());
      Vector velocity_solution(output.GetData(), velocity.Height());
      Vector pressure_solution(output.GetData() + velocity.Height(), pressure.Height());
      if (forward || diagonal)
      {
         velocity.Mult(velocity_rhs, velocity_solution);
         if (!diagonal)
         {
            Vector work(lower.Height()); lower.Mult(velocity_solution, work);
            for (int entry = 0; entry < work.Size(); ++entry) { pressure_rhs(entry) -= work(entry); }
         }
         pressure.Mult(pressure_rhs, pressure_solution);
      }
      else
      {
         pressure.Mult(pressure_rhs, pressure_solution);
         Vector pressure_field(pressure_solution.GetData(), upper.Width()), work(velocity.Height());
         upper.Mult(pressure_field, work); velocity_rhs -= work;
         velocity.Mult(velocity_rhs, velocity_solution);
      }
      apply_seconds += MPI_Wtime() - started;
   }
};

class Projector
{
   MPI_Comm communicator;
   Vector mode;
public:
   Projector(MPI_Comm communicator_, const Vector &mode_) : communicator(communicator_), mode(mode_)
   { mode /= Norm(communicator, mode); }
   void Apply(Vector &value) const
   { value.Add(-InnerProduct(communicator, mode, value), mode); }
};

class ProjectedOperator : public Operator
{
   const Operator &matrix;
   const Projector &right, &left;
public:
   ProjectedOperator(const Operator &matrix_, const Projector &right_, const Projector &left_)
      : Operator(matrix_.Height()), matrix(matrix_), right(right_), left(left_) { }
   void Mult(const Vector &input, Vector &output) const override
   {
      Vector work(input);
      right.Apply(work);
      matrix.Mult(work, output);
      left.Apply(output);
   }
};

class ProjectedInverse : public Solver
{
   const Solver &inverse;
   const Projector &right, &left;
public:
   ProjectedInverse(int size, const Solver &inverse_, const Projector &right_, const Projector &left_)
      : Solver(size), inverse(inverse_), right(right_), left(left_) { }
   void SetOperator(const Operator &) override { MFEM_ABORT("Fixed projected inverse"); }
   void Mult(const Vector &input, Vector &output) const override
   {
      Vector work(input), padded(inverse.Height()), result(inverse.Height());
      left.Apply(work);
      padded = 0.0;
      for (int entry = 0; entry < height; ++entry) { padded(entry) = work(entry); }
      inverse.Mult(padded, result);
      output.SetSize(height);
      for (int entry = 0; entry < height; ++entry) { output(entry) = result(entry); }
      right.Apply(output);
   }
};

}

class BlockFGMRESSolver::Impl
{
public:
   const FlowProblem &problem;
   FlowSolverOptions options;
   const TrueDofLayout &layout;
   Array2D<const HypreParMatrix *> blocks;
   std::vector<std::unique_ptr<HypreParMatrix>> transposed;
   std::unique_ptr<HypreParMatrix> matrix, physical, pressure_mass, pressure_auxiliary;
   std::unique_ptr<TransposeOperator> transpose_operator;
   const Operator *system_operator = nullptr;
   std::unique_ptr<HypreParMatrix> lsc_middle;
   std::unique_ptr<HypreParMatrix> pcd_mass, pcd_transport;
   Vector constant, moments, right_mode, left_mode;
   std::unique_ptr<LinearSolverContext> context;
   std::unique_ptr<Solver> velocity_inverse, pressure_degree_one, pressure_inverse;
   std::unique_ptr<BlockUpperInverse> preconditioner;
   std::unique_ptr<Projector> right, left;
   std::unique_ptr<ProjectedOperator> projected;
   std::unique_ptr<ProjectedInverse> projected_inverse;
   Vector pressure_diagonal;
   std::unique_ptr<IterativeSolver> krylov;
   std::unique_ptr<RightPreconditionedOperator> right_preconditioned;
   Solver *active_inverse = nullptr;

   Impl(const FlowProblem &problem_, const FlowSolverOptions &options_,
        const TrueDofLayout &layout_, const Array<int> &essential, const Vector &state,
        const Array2D<const HypreParMatrix *> &source, const HypreParMatrix &proxy, bool transpose,
        const P1AMGPreconditioner::CoarseAssembler &velocity_coarse,
        const P1AMGPreconditioner::CoarseAssembler &pressure_coarse)
      : problem(problem_), options(options_), layout(layout_), blocks(source.NumRows(), source.NumCols())
   {
      const auto &settings = std::get<IterativeOptions>(options.linear.backend);
      const auto communicator = problem.GetComm();
      const bool lower = settings.preconditioner == PreconditionerType::BlockLowerTriangularP1AMG ||
                         settings.preconditioner == PreconditionerType::BlockLowerTriangularP1AMGAssembled;
      const bool diagonal = settings.preconditioner == PreconditionerType::BlockDiagonalAMG ||
                            settings.preconditioner == PreconditionerType::BlockDiagonalP1AMG ||
                            settings.preconditioner == PreconditionerType::BlockDiagonalP1AMGAssembled;
      const bool assembled = settings.preconditioner == PreconditionerType::BlockDiagonalP1AMGAssembled ||
                             settings.preconditioner == PreconditionerType::BlockUpperTriangularP1AMGAssembled ||
                             settings.preconditioner == PreconditionerType::BlockLowerTriangularP1AMGAssembled;
      const bool classical = settings.preconditioner == PreconditionerType::BlockDiagonalAMG;
      Check(communicator, !assembled || (velocity_coarse && pressure_coarse), "Assembled P1 inverses require flow coarse builders");
      if (settings.method == LinearSolverType::GMRES) { krylov = std::make_unique<GMRESSolver>(communicator); }
      else { krylov = std::make_unique<FGMRESSolver>(communicator); }
      for (int row = 0; row < blocks.NumRows(); ++row)
      {
         for (int column = 0; column < blocks.NumCols(); ++column)
         {
            if (transpose && source(column, row))
            {
               transposed.emplace_back(source(column, row)->Transpose());
               blocks(row, column) = transposed.back().get();
            }
            else { blocks(row, column) = transpose ? nullptr : source(row, column); }
         }
      }
      Array2D<const HypreParMatrix *> source_blocks(source);
      matrix.reset(HypreParMatrixFromBlocks(source_blocks));
      if (transpose)
      {
         transpose_operator = std::make_unique<TransposeOperator>(*matrix);
         system_operator = transpose_operator.get();
      }
      else { system_operator = matrix.get(); }
      auto &pressure_space = problem.GetPressureSpace();
      if (layout.HasGauge())
      {
         ConstantCoefficient one(1.0);
         ParGridFunction field(&pressure_space);
         field.ProjectCoefficient(one);
         field.GetTrueDofs(constant);
         Vector gauge_one(blocks(1, 2)->Width()); gauge_one = 1.0;
         moments.SetSize(constant.Size()); blocks(1, 2)->Mult(gauge_one, moments);
      }
      if (options.gauge_enforcement == GaugeEnforcement::NullspaceProjection)
      {
         Check(communicator, layout.HasGauge(), "Nullspace projection requires a pressure mean");
         Array2D<const HypreParMatrix *> physical_blocks(2, 2);
         for (int row = 0; row < 2; ++row)
         { for (int column = 0; column < 2; ++column) { physical_blocks(row, column) = blocks(row, column); } }
         physical.reset(HypreParMatrixFromBlocks(physical_blocks));
         right_mode.SetSize(physical->Height()); right_mode = 0.0;
         for (int entry = 0; entry < constant.Size(); ++entry) { right_mode(proxy.Height() + entry) = constant(entry); }
         left_mode = right_mode;
         Vector defect(physical->Height());
         const real_t threshold = 1e-11 * std::max(real_t(1.0), physical->FNorm()) * Norm(communicator, right_mode);
         physical->Mult(right_mode, defect);
         Check(communicator, Norm(communicator, defect) <= threshold, "Unsupported projected right pressure nullspace");
         physical->MultTranspose(left_mode, defect);
         Check(communicator, Norm(communicator, defect) <= threshold, "Unsupported projected left pressure nullspace");
         right = std::make_unique<Projector>(communicator, right_mode);
         left = std::make_unique<Projector>(communicator, left_mode);
      }
      if (settings.schur == SchurApproximation::ExperimentalPCD)
      {
         auto operators = AssemblePCDOperators(problem, options, state);
         pressure_mass = std::move(operators.laplacian);
         pcd_mass = std::move(operators.mass);
         pcd_transport = std::move(operators.convection_diffusion);
      }
      else { BuildSchurOperators(problem, options, essential, source, blocks, pressure_mass, lsc_middle); }
      if (layout.HasGauge())
      {
         Array2D<const HypreParMatrix *> pressure_blocks(2, 2);
         pressure_blocks(0, 0) = pressure_mass.get();
         pressure_blocks(0, 1) = blocks(1, 2);
         pressure_blocks(1, 0) = blocks(2, 1);
         pressure_blocks(1, 1) = nullptr;
         pressure_auxiliary.reset(HypreParMatrixFromBlocks(pressure_blocks));
      }
      else { pressure_auxiliary = std::make_unique<HypreParMatrix>(*pressure_mass); }
      context = std::make_unique<LinearSolverContext>(LinearSolverContext{
         problem, layout, essential, state, blocks, proxy, *pressure_auxiliary, constant, moments,
         right ? &right_mode : nullptr, left ? &left_mode : nullptr,
         options.gauge_enforcement, transpose, settings.hierarchy});
      const auto &velocity_matrix = options.convection ? *blocks(0, 0) : proxy;
      P1AMGPreconditioner::CoarseAssembler assemble_velocity = velocity_coarse;
      if (assembled && options.convection)
      {
         assemble_velocity = [&](ParFiniteElementSpace &coarse_space)
         {
            auto coarse = velocity_coarse(coarse_space);
            PRefinementTransferOperator transfer(coarse_space, problem.GetVelocitySpace(), true);
            auto *prolongation = dynamic_cast<HypreParMatrix *>(transfer.GetTrueTransferOperator());
            Check(communicator, prolongation != nullptr, "Expected a Hypre velocity degree transfer");
            std::unique_ptr<HypreParMatrix> difference(Add(1.0, velocity_matrix, -1.0, proxy));
            std::unique_ptr<HypreParMatrix> correction(RAP(difference.get(), prolongation));
            return std::unique_ptr<HypreParMatrix>(Add(1.0, *coarse, 1.0, *correction));
         };
      }
      if (settings.velocity_inverse_factory) { velocity_inverse = settings.velocity_inverse_factory(*context); }
      else if (classical)
      {
         auto amg = std::make_unique<HypreBoomerAMG>(velocity_matrix);
         if (options.convection)
         { amg->SetSystemsOptions(problem.GetVelocitySpace().GetVDim(), problem.GetVelocitySpace().GetOrdering() == Ordering::byNODES); }
         else { amg->SetElasticityOptions(&problem.GetVelocitySpace(), settings.amg_elasticity_interp_refine); }
         amg->SetMaxIter(1); amg->SetTol(0.0); amg->SetPrintLevel(0);
         Vector zero(proxy.Height()), result(proxy.Height()); zero = 0.0; result = 0.0;
         amg->Setup(zero, result); velocity_inverse = std::move(amg);
      }
      else
      {
         auto &space = problem.GetVelocitySpace();
         std::vector<Array<int>> markers(space.GetVDim());
         const auto &attributes = space.GetParMesh()->bdr_attributes;
         for (auto &marker : markers) { marker.SetSize(attributes.Size() ? attributes.Max() : 0); marker = 0; }
         for (const auto &entry : problem.GetBoundaryConditions().GetVelocity())
         { markers[entry.first.second][entry.first.first - 1] = 1; }
         velocity_inverse = std::make_unique<flow::detail::P1AMGPreconditioner>(
                               velocity_matrix, space, markers, settings.p1_smoothing_steps, true, settings.amg_elasticity_interp_refine,
                               assembled ? assemble_velocity : P1AMGPreconditioner::CoarseAssembler{},
                               problem.GetPrescribedVelocityValues() ? &essential : nullptr,
                               [this](ParFiniteElementSpace &coarse, Array<int> &dofs)
                               { problem.GetAuxiliaryVelocityConstraints(coarse, dofs); }, options.convection);
      }
      if (settings.pressure_inverse_factory) { pressure_inverse = settings.pressure_inverse_factory(*context); }
      else if (settings.schur == SchurApproximation::ExperimentalPCD)
      {
         pressure_inverse = std::make_unique<PCDPressureInverse>(*pressure_mass, *pcd_mass, *pcd_transport,
                           constant, moments, layout.HasGauge() ? blocks(1, 2)->Width() : 0,
                           layout.HasGauge(), settings, transpose);
      }
      else if (settings.schur == SchurApproximation::ExperimentalLSC)
      {
         pressure_inverse = std::make_unique<LSCPressureInverse>(*pressure_mass, *lsc_middle, constant, moments,
                           layout.HasGauge() ? blocks(1, 2)->Width() : 0, layout.HasGauge(), settings, transpose);
      }
      else if (settings.schur == SchurApproximation::WeightedMass)
      {
         if (classical)
         {
            pressure_mass->GetDiag(pressure_diagonal);
            bool valid = true;
            for (int entry = 0; entry < pressure_diagonal.Size(); ++entry)
            {
               valid = valid && std::isfinite(pressure_diagonal(entry)) &&
                       pressure_diagonal(entry) > 0.0;
            }
            Check(communicator, valid, "Pressure mass inverse requires a finite positive diagonal");
            pressure_degree_one = std::make_unique<DiagonalInverse>(pressure_diagonal);
         }
         else { pressure_degree_one = std::make_unique<flow::detail::P1AMGPreconditioner>(
                                  *pressure_mass, pressure_space, std::vector<Array<int>>{},
                                  settings.p1_smoothing_steps, false, false,
                                  assembled ? pressure_coarse : P1AMGPreconditioner::CoarseAssembler{}); }
         pressure_inverse = std::make_unique<MassGaugeInverse>(communicator, *pressure_degree_one,
                                                               layout.HasGauge() ? blocks(1, 2) : nullptr, lower, diagonal);
      }
      else
      {
#ifdef MFEM_USE_MUMPS
         auto direct = std::make_unique<MUMPSSolver>(communicator);
         direct->SetPrintLevel(0);
         direct->SetMatrixSymType(MUMPSSolver::UNSYMMETRIC);
         direct->SetOperator(*pressure_auxiliary);
         pressure_inverse = std::move(direct);
#else
         MFEM_ABORT("Experimental reaction Schur requires MUMPS");
#endif
      }
      Check(communicator, velocity_inverse && velocity_inverse->Height() == proxy.Height() &&
            velocity_inverse->Width() == proxy.Width() && pressure_inverse &&
            pressure_inverse->Height() == pressure_auxiliary->Height() &&
            pressure_inverse->Width() == pressure_auxiliary->Width(), "Invalid iterative inverse factory result");
      preconditioner = std::make_unique<BlockUpperInverse>(*velocity_inverse, *pressure_inverse, *blocks(0, 1), *blocks(1, 0), lower, diagonal);
      if (physical)
      {
         projected = std::make_unique<ProjectedOperator>(*physical, *right, *left);
         projected_inverse = std::make_unique<ProjectedInverse>(physical->Height(), *preconditioner, *right, *left);
         active_inverse = projected_inverse.get();
      }
      else
      {
         active_inverse = preconditioner.get();
      }
      const Operator &active_matrix = projected ? static_cast<const Operator &>(*projected) : *system_operator;
      if (settings.method == LinearSolverType::GMRES)
      {
         right_preconditioned = std::make_unique<RightPreconditionedOperator>(active_matrix, *active_inverse);
         krylov->SetOperator(*right_preconditioned);
         static_cast<GMRESSolver &>(*krylov).SetKDim(settings.restart_dimension);
      }
      else
      {
         krylov->SetOperator(active_matrix); krylov->SetPreconditioner(*active_inverse);
         static_cast<FGMRESSolver &>(*krylov).SetKDim(settings.restart_dimension);
      }
      krylov->SetMaxIter(settings.max_iterations);
      krylov->SetRelTol(0.1 * options.linear.relative_tolerance);
      krylov->SetAbsTol(0.1 * options.linear.absolute_tolerance);
      krylov->SetPrintLevel(options.linear.print_level);
      krylov->iterative_mode = false;
   }

   LinearSolveResult Solve(const Vector &rhs, Vector &solution)
   {
      const auto communicator = problem.GetComm();
      LinearSolveResult result;
      const double started = MPI_Wtime(), apply_before = preconditioner->apply_seconds;
      solution.SetSize(matrix->Height()); solution = 0.0;
      const real_t tolerance = std::max(options.linear.absolute_tolerance,
                                        options.linear.relative_tolerance * Norm(communicator, rhs));
      auto solve_with_true_residual = [&](const Operator &op, const Vector &load, Vector &answer)
      {
         answer.SetSize(load.Size()); answer = 0.0;
         Vector residual(load), correction(load.Size());
         real_t norm = Norm(communicator, residual);
         const int maximum = std::get<IterativeOptions>(options.linear.backend).max_iterations;
         while (std::isfinite(norm) && norm > tolerance && result.iterations < maximum)
         {
            if (result.iterations) { ++result.residual_restarts; }
            krylov->SetMaxIter(maximum - result.iterations);
            krylov->SetRelTol(0.0);
            krylov->SetAbsTol(0.1 * tolerance);
            krylov->Mult(residual, correction);
            if (right_preconditioned)
            { Vector recovered(correction.Size()); active_inverse->Mult(correction, recovered); correction = recovered; }
            const int count = krylov->GetNumIterations();
            result.iterations += count;
            answer += correction;
            op.Mult(answer, residual); residual *= -1.0; residual += load;
            const real_t next = Norm(communicator, residual);
            if (count == 0 || !(next < norm)) { break; }
            norm = next;
         }
      };
      bool compatible = true;
      if (projected)
      {
         Vector physical_rhs(physical->Height()), correction(physical->Height());
         for (int entry = 0; entry < physical_rhs.Size(); ++entry) { physical_rhs(entry) = rhs(entry); }
         Vector original(physical_rhs);
         left->Apply(physical_rhs);
         original -= physical_rhs;
         compatible = Norm(communicator, original) <= tolerance;
         if (compatible)
         {
            solve_with_true_residual(*projected, physical_rhs, correction);
            right->Apply(correction);
            const int velocity_size = problem.GetVelocitySpace().GetTrueVSize();
            Vector pressure(correction.GetData() + velocity_size, constant.Size());
            real_t mean_rhs = rhs.Size() > physical_rhs.Size() ? rhs(physical_rhs.Size()) : 0.0;
            MPI_Bcast(&mean_rhs, 1, MPITypeMap<real_t>::mpi_type, 0, communicator);
            const real_t shift = (mean_rhs - InnerProduct(communicator, moments, pressure)) /
                                 InnerProduct(communicator, moments, constant);
            pressure.Add(shift, constant);
            for (int entry = 0; entry < correction.Size(); ++entry) { solution(entry) = correction(entry); }
         }
      }
      else { solve_with_true_residual(*system_operator, rhs, solution); }
      Vector residual(rhs.Size());
      system_operator->Mult(solution, residual); residual -= rhs;
      result.residual = Norm(communicator, residual);
      int valid = compatible && solution.CheckFinite() == 0 && std::isfinite(result.residual) && result.residual <= tolerance;
      int global = 0;
      MPI_Allreduce(&valid, &global, 1, MPI_INT, MPI_MIN, communicator);
      result.converged = global != 0;
      result.solve_seconds = MPI_Wtime() - started;
      result.preconditioner_seconds = preconditioner->apply_seconds - apply_before;
      return result;
   }
};

BlockFGMRESSolver::BlockFGMRESSolver(const FlowProblem &problem,
   const FlowSolverOptions &options, const TrueDofLayout &layout, const Array<int> &essential,
   const Vector &state, const Array2D<const HypreParMatrix *> &blocks,
   const HypreParMatrix &velocity_proxy, bool transpose,
   const P1AMGPreconditioner::CoarseAssembler &velocity_coarse,
   const P1AMGPreconditioner::CoarseAssembler &pressure_coarse)
   : impl(std::make_unique<Impl>(problem, options, layout, essential, state, blocks, velocity_proxy, transpose, velocity_coarse, pressure_coarse)) { }
BlockFGMRESSolver::~BlockFGMRESSolver() = default;
LinearSolveResult BlockFGMRESSolver::Solve(const Vector &rhs, Vector &solution) { return impl->Solve(rhs, solution); }
const HypreParMatrix &BlockFGMRESSolver::GetPressureAuxiliary() const { return *impl->pressure_auxiliary; }
const Solver &BlockFGMRESSolver::GetPreconditioner() const { return *impl->preconditioner; }


UpperTriangularPreconditioner::UpperTriangularPreconditioner(
   const Array<int> &offsets_, Solver &velocity_inverse_, Solver &pressure_inverse_,
   const Operator &momentum_pressure_, Solver *gauge_inverse_, const Operator *gauge_column_)
   : Solver(offsets_.Last()), offsets(offsets_),
     velocity_inverse(velocity_inverse_), pressure_inverse(pressure_inverse_),
     momentum_pressure(momentum_pressure_), gauge_inverse(gauge_inverse_),
     gauge_column(gauge_column_), work(offsets),
     velocity_work(velocity_inverse.Height()), pressure_work(pressure_inverse.Height())
{
   MFEM_VERIFY((offsets.Size() == 3 && !gauge_inverse && !gauge_column) ||
               (offsets.Size() == 4 && gauge_inverse && gauge_column),
               "Inconsistent upper-triangular gauge blocks");
}

void UpperTriangularPreconditioner::SetOperator(const Operator &)
{
   MFEM_ABORT("UpperTriangularPreconditioner has fixed blocks");
}

void UpperTriangularPreconditioner::Mult(const Vector &input, Vector &output) const
{
   work.Vector::operator=(input);
   output.SetSize(height);
   BlockVector result(output, offsets);
   if (gauge_inverse)
   {
      gauge_inverse->Mult(work.GetBlock(2), result.GetBlock(2));
      gauge_column->Mult(result.GetBlock(2), pressure_work);
      work.GetBlock(1) -= pressure_work;
   }
   pressure_inverse.Mult(work.GetBlock(1), result.GetBlock(1));
   momentum_pressure.Mult(result.GetBlock(1), velocity_work);
   work.GetBlock(0) -= velocity_work;
   velocity_inverse.Mult(work.GetBlock(0), result.GetBlock(0));
}

}

namespace
{

#ifdef MFEM_USE_MUMPS
class RootMumpsInverse : public Solver
{
   MPI_Comm communicator;
   int rank;
   std::vector<int> counts, offsets;
   HYPRE_BigInt partition[2] = {0, 0};
   std::unique_ptr<SparseMatrix> serial;
   std::unique_ptr<HypreParMatrix> matrix;
   std::unique_ptr<MUMPSSolver> solver;
   mutable Vector rhs, solution;
public:
   RootMumpsInverse(HypreParMatrix &source, const GeometricMultigridOptions &options)
      : Solver(source.Height()), communicator(source.GetComm())
   {
      int ranks;
      MPI_Comm_rank(communicator, &rank); MPI_Comm_size(communicator, &ranks);
      const auto global_rows = source.GetGlobalNumRows();
      detail::Check(communicator, global_rows > 0 && global_rows <= options.root_max_dofs &&
                    source.GetGlobalNumCols() == global_rows, "GMG root coarse matrix exceeds configured limits");
      SparseMatrix diagonal, offdiagonal;
      HYPRE_BigInt *column_map;
      source.GetDiag(diagonal); source.GetOffd(offdiagonal, column_map);
      long long local_count = static_cast<long long>(diagonal.NumNonZeroElems()) + offdiagonal.NumNonZeroElems(), global_nonzeros;
      MPI_Allreduce(&local_count, &global_nonzeros, 1, MPI_LONG_LONG, MPI_SUM, communicator);
      detail::Check(communicator, global_nonzeros > 0 && global_nonzeros <= options.root_max_nonzeros,
                    "GMG root coarse matrix exceeds configured limits");
      SparseMatrix local;
      source.MergeDiagAndOffd(local);
      const int local_nonzeros = local.NumNonZeroElems();
      counts.resize(ranks); offsets.resize(ranks);
      std::vector<int> nonzero_counts(ranks), nonzero_offsets(ranks);
      MPI_Allgather(&height, 1, MPI_INT, counts.data(), 1, MPI_INT, communicator);
      MPI_Allgather(&local_nonzeros, 1, MPI_INT, nonzero_counts.data(), 1, MPI_INT, communicator);
      for (int process = 1; process < ranks; ++process)
      {
         offsets[process] = offsets[process - 1] + counts[process - 1];
         nonzero_offsets[process] = nonzero_offsets[process - 1] + nonzero_counts[process - 1];
      }
      const auto first_row = source.GetRowStarts()[HYPRE_AssumedPartitionCheck() ? 0 : rank];
      detail::Check(communicator, first_row == offsets[rank] && offsets.back() + counts.back() == global_rows,
                    "GMG root coarse matrix requires contiguous rank-ordered rows");
      std::vector<int> row_lengths(height);
      for (int row = 0; row < height; ++row) { row_lengths[row] = local.GetI()[row + 1] - local.GetI()[row]; }
      std::unique_ptr<int[]> rows, columns;
      std::unique_ptr<real_t[]> entries;
      if (rank == 0)
      {
         rows = std::make_unique<int[]>(global_rows + 1);
         columns = std::make_unique<int[]>(global_nonzeros);
         entries = std::make_unique<real_t[]>(global_nonzeros);
      }
      MPI_Gatherv(row_lengths.data(), height, MPI_INT, rows.get(), counts.data(), offsets.data(), MPI_INT, 0, communicator);
      MPI_Gatherv(local.GetJ(), local_nonzeros, MPI_INT, columns.get(), nonzero_counts.data(), nonzero_offsets.data(), MPI_INT, 0, communicator);
      MPI_Gatherv(local.GetData(), local_nonzeros, MPITypeMap<real_t>::mpi_type, entries.get(), nonzero_counts.data(),
                  nonzero_offsets.data(), MPITypeMap<real_t>::mpi_type, 0, communicator);
      if (rank == 0)
      {
         int position = 0;
         for (int row = 0; row < global_rows; ++row)
         { const int length = rows[row]; rows[row] = position; position += length; }
         rows[global_rows] = position;
         serial = std::make_unique<SparseMatrix>(rows.release(), columns.release(), entries.release(), global_rows, global_rows);
         partition[1] = global_rows;
         matrix = std::make_unique<HypreParMatrix>(MPI_COMM_SELF, global_rows, partition, serial.get());
         solver = std::make_unique<MUMPSSolver>(MPI_COMM_SELF);
         solver->SetPrintLevel(0); solver->SetMatrixSymType(MUMPSSolver::SYMMETRIC_POSITIVE_DEFINITE);
         solver->SetOperator(*matrix);
         rhs.SetSize(global_rows); solution.SetSize(global_rows);
      }
   }
   void SetOperator(const Operator &) override { MFEM_ABORT("Fixed root coarse inverse"); }
   void Mult(const Vector &input, Vector &output) const override
   {
      MPI_Gatherv(input.HostRead(), height, MPITypeMap<real_t>::mpi_type, rhs.HostWrite(), counts.data(), offsets.data(),
                  MPITypeMap<real_t>::mpi_type, 0, communicator);
      if (rank == 0) { solver->Mult(rhs, solution); }
      output.SetSize(height);
      MPI_Scatterv(solution.HostRead(), counts.data(), offsets.data(), MPITypeMap<real_t>::mpi_type,
                   output.HostWrite(), height, MPITypeMap<real_t>::mpi_type, 0, communicator);
   }
   void MultTranspose(const Vector &input, Vector &output) const override { Mult(input, output); }
};
#endif

std::uint64_t MeshStamp(const ParMesh &mesh)
{
   std::uint64_t stamp = 14695981039346656037ULL;
   auto append = [&](const void *data, std::size_t count)
   {
      const auto *bytes = static_cast<const unsigned char *>(data);
      for (std::size_t index = 0; index < count; ++index)
      { stamp ^= bytes[index]; stamp *= 1099511628211ULL; }
   };
   for (int vertex = 0; vertex < mesh.GetNV(); ++vertex)
   { append(mesh.GetVertex(vertex), mesh.SpaceDimension() * sizeof(real_t)); }
   if (const auto *nodes = mesh.GetNodes())
   { append(nodes->HostRead(), nodes->Size() * sizeof(real_t)); }
   for (int element = 0; element < mesh.GetNE(); ++element)
   { const int attribute = mesh.GetAttribute(element); append(&attribute, sizeof(attribute)); }
   for (int element = 0; element < mesh.GetNBE(); ++element)
   { const int attribute = mesh.GetBdrAttribute(element); append(&attribute, sizeof(attribute)); }
   return stamp;
}

Array<int> Essential(ParFiniteElementSpace &space, const BoundaryConditions &boundary)
{
   Array<int> result, marker(space.GetParMesh()->bdr_attributes.Max()), component_dofs;
   for (int component = 0; component < space.GetVDim(); ++component)
   {
      marker = 0;
      for (const auto &entry : boundary.GetVelocity())
      { if (entry.first.second == component) { marker[entry.first.first - 1] = 1; } }
      space.GetEssentialTrueDofs(marker, component_dofs, component);
      result.Append(component_dofs);
   }
   result.Sort(); result.Unique();
   return result;
}

std::uint64_t Storage(const HypreParMatrix &matrix)
{
   SparseMatrix diagonal, offdiagonal;
   HYPRE_BigInt *columns;
   matrix.GetDiag(diagonal); matrix.GetOffd(offdiagonal, columns);
   return std::uint64_t(diagonal.NumNonZeroElems() + offdiagonal.NumNonZeroElems()) *
          (sizeof(real_t) + sizeof(HYPRE_Int)) +
          std::uint64_t(2) * (matrix.Height() + 1) * sizeof(HYPRE_Int) +
          std::uint64_t(offdiagonal.Width()) * sizeof(HYPRE_BigInt);
}

Vector JacobiWeights(const HypreParMatrix &matrix, real_t damping)
{
   Vector diagonal; matrix.GetDiag(diagonal);
   SparseMatrix local, remote;
   HYPRE_BigInt *columns;
   matrix.GetDiag(local); matrix.GetOffd(remote, columns);
   bool valid = true;
   real_t local_bound = 0.0;
   for (int row = 0; row < matrix.Height(); ++row)
   {
      real_t sum = 0.0;
      for (const SparseMatrix *part : {&local, &remote})
      {
         for (int entry = part->GetI()[row]; entry < part->GetI()[row + 1]; ++entry)
         { sum += std::abs(part->GetData()[entry]); }
      }
      valid = valid && std::isfinite(diagonal(row)) && diagonal(row) > 0.0 && std::isfinite(sum);
      if (valid) { local_bound = std::max(local_bound, sum / diagonal(row)); }
   }
   detail::Check(matrix.GetComm(), valid, "GMG requires finite positive level diagonals");
   real_t bound;
   MPI_Allreduce(&local_bound, &bound, 1, MPITypeMap<real_t>::mpi_type, MPI_MAX, matrix.GetComm());
   detail::Check(matrix.GetComm(), std::isfinite(bound) && bound > 0.0, "Invalid GMG Jacobi bound");
   for (int entry = 0; entry < diagonal.Size(); ++entry)
   { diagonal(entry) = damping / (bound * diagonal(entry)); }
   return diagonal;
}

}

void GeometricMultigridOptions::Validate(MPI_Comm communicator) const
{
   detail::CheckReplicated(communicator, {static_cast<real_t>(smoother), static_cast<real_t>(cycle),
      static_cast<real_t>(coarse_solver), static_cast<real_t>(pre_sweeps),
      static_cast<real_t>(post_sweeps), static_cast<real_t>(coarse_sweeps), damping,
      static_cast<real_t>(root_max_dofs), static_cast<real_t>(root_max_nonzeros), static_cast<real_t>(num_levels),
      static_cast<real_t>(coarse_amg_cycles), coarse_factory ? 1.0 : 0.0});
   detail::Check(communicator, smoother == GeometricSmoother::DampedJacobi && cycle == GeometricCycle::V &&
                 (coarse_solver == GeometricCoarseSolver::Mumps || coarse_solver == GeometricCoarseSolver::FixedJacobi ||
                  coarse_solver == GeometricCoarseSolver::MumpsRoot || coarse_solver == GeometricCoarseSolver::BoomerAMG ||
                  coarse_solver == GeometricCoarseSolver::User) && coarse_amg_cycles > 0 &&
                 (coarse_solver != GeometricCoarseSolver::User || coarse_factory) && root_max_dofs > 0 && root_max_nonzeros > 0 &&
                 pre_sweeps > 0 && post_sweeps > 0 && coarse_sweeps > 0 && num_levels >= 0 &&
                 std::isfinite(damping) && damping > 0.0 && damping < 1.0, "Invalid GMG options");
#ifndef MFEM_USE_MUMPS
   detail::Check(communicator, coarse_solver != GeometricCoarseSolver::Mumps &&
                 coarse_solver != GeometricCoarseSolver::MumpsRoot, "GMG MUMPS support is unavailable");
#endif
}

class UniformVelocityHierarchy::Impl
{
public:
   MPI_Comm communicator;
   H1_FECollection collection;
   std::vector<std::unique_ptr<ParMesh>> meshes;
   std::vector<std::unique_ptr<ParFiniteElementSpace>> spaces;
   std::vector<std::unique_ptr<HypreParMatrix>> transfers;
   std::vector<VelocityHierarchyLevel> levels;
   std::vector<long> mesh_sequences, space_sequences;
   std::vector<std::uint64_t> stamps;

   Impl(const ParMesh &coarse, int order, int refinements)
      : communicator(coarse.GetComm()), collection(order, coarse.Dimension())
   {
      detail::CheckReplicated(communicator, {static_cast<real_t>(order), static_cast<real_t>(refinements),
                                             static_cast<real_t>(coarse.Dimension())});
      detail::Check(communicator, order >= 2 && order <= 4 && refinements >= 1 && refinements <= 8 &&
                    (coarse.Dimension() == 2 || coarse.Dimension() == 3) && coarse.SpaceDimension() == coarse.Dimension() &&
                    !coarse.Nonconforming() && coarse.GetNE() > 0 && coarse.bdr_attributes.Size() > 0,
                    "GMG requires a nonempty conforming 2D/3D coarse mesh, order 2-4 and 1-8 refinements");
      bool supported_geometry = true;
      for (int element = 0; element < coarse.GetNE(); ++element)
      {
         const auto geometry = coarse.GetElementGeometry(element);
         supported_geometry = supported_geometry && (geometry == Geometry::TRIANGLE || geometry == Geometry::SQUARE ||
                              geometry == Geometry::TETRAHEDRON || geometry == Geometry::CUBE);
      }
      if (const auto *nodes = coarse.GetNodes())
      {
         const auto &space = *nodes->FESpace();
         supported_geometry = supported_geometry && dynamic_cast<const H1_FECollection *>(space.FEColl()) &&
                              !space.IsVariableOrder() && space.FEColl()->GetOrder() <= 2;
      }
      detail::Check(communicator, supported_geometry, "GMG requires simplex/tensor cells and affine or quadratic H1 geometry");
      for (int level = 0; level <= refinements; ++level)
      {
         meshes.push_back(std::make_unique<ParMesh>(level == 0 ? coarse : *meshes.back()));
         if (level) { meshes.back()->UniformRefinement(); }
         spaces.push_back(std::make_unique<ParFiniteElementSpace>(meshes.back().get(), &collection,
                                                                  coarse.Dimension(), Ordering::byVDIM));
         if (level)
         {
            OperatorHandle transfer(Operator::Hypre_ParCSR);
            spaces.back()->GetTrueTransferOperator(*spaces[level - 1], transfer);
            transfers.emplace_back(transfer.As<HypreParMatrix>());
            transfer.SetOperatorOwner(false);
         }
      }
      for (int level = 0; level <= refinements; ++level)
      {
         levels.push_back({spaces[level].get(), level < refinements ? transfers[level].get() : nullptr,
                           LevelOperatorConstruction::Galerkin});
         mesh_sequences.push_back(meshes[level]->GetSequence());
         space_sequences.push_back(spaces[level]->GetSequence());
         stamps.push_back(MeshStamp(*meshes[level]));
      }
   }
   void CheckCurrent() const
   {
      bool valid = true;
      for (std::size_t level = 0; level < levels.size(); ++level)
      {
         valid = valid && meshes[level]->GetSequence() == mesh_sequences[level] &&
                 spaces[level]->GetSequence() == space_sequences[level] &&
                 MeshStamp(*meshes[level]) == stamps[level];
      }
      detail::Check(communicator, valid, "GMG mesh/spaces changed; reconstruct the hierarchy");
   }
};

UniformVelocityHierarchy::UniformVelocityHierarchy(const ParMesh &mesh, int order, int refinements)
{
   detail::Check(mesh.GetComm(), order >= 2 && order <= 4, "GMG requires velocity order 2-4");
   impl = std::make_unique<Impl>(mesh, order, refinements);
}
UniformVelocityHierarchy::~UniformVelocityHierarchy() = default;
void UniformVelocityHierarchy::CheckCurrent() const { impl->CheckCurrent(); }
ParFiniteElementSpace &UniformVelocityHierarchy::GetFinestSpace() const
{ CheckCurrent(); return *impl->spaces.back(); }
const std::vector<VelocityHierarchyLevel> &UniformVelocityHierarchy::GetLevels() const
{ CheckCurrent(); return impl->levels; }

void UniformVelocityHierarchy::BuildOperators(const LinearSolverContext &context,
   std::vector<std::unique_ptr<HypreParMatrix>> &operators,
   std::vector<std::unique_ptr<HypreParMatrix>> &transfers,
   std::vector<Array<int>> &essential, int num_levels) const
{
   CheckCurrent(); context.layout.CheckCurrent();
   detail::Check(impl->communicator, context.hierarchy == this &&
                 &context.problem.GetVelocitySpace() == impl->spaces.back().get(), "Incompatible GMG finest space or provider");
   detail::CheckReplicated(impl->communicator, {static_cast<real_t>(num_levels)});
   const int available = static_cast<int>(impl->levels.size());
   detail::Check(impl->communicator, num_levels >= 0 && num_levels <= available, "Invalid GMG level count for hierarchy");
   const int count = num_levels ? num_levels : available, first = available - count;
   operators.clear(); transfers.clear(); essential.clear();
   operators.resize(count);
   for (int level = first; level < available; ++level)
   {
      if (level + 1 == available) { essential.push_back(context.essential_velocity_dofs); }
      else
      {
         auto constraints = Essential(*impl->spaces[level], context.problem.GetBoundaryConditions());
         Array<int> additional;
         context.problem.GetAuxiliaryVelocityConstraints(*impl->spaces[level], additional);
         constraints.Append(additional); constraints.Sort(); constraints.Unique();
         essential.push_back(std::move(constraints));
      }
   }
   detail::Check(impl->communicator, essential.back() == context.essential_velocity_dofs,
                 "Incompatible GMG finest correction constraints");
   operators.back() = std::make_unique<HypreParMatrix>(context.velocity_proxy);
   for (int level = 0; level + 1 < count; ++level)
   {
      transfers.push_back(std::make_unique<HypreParMatrix>(*impl->transfers[first + level]));
      flow::detail::ConstrainCorrectionTransfer(*transfers.back(), essential[level + 1], essential[level],
                                                context.problem.GetPrescribedVelocityValues() != nullptr);
   }
   for (int level = count - 2; level >= 0; --level)
   {
      operators[level].reset(RAP(operators[level + 1].get(), transfers[level].get()));
      std::unique_ptr<HypreParMatrix> removed(operators[level]->EliminateRowsCols(essential[level]));
   }
}

std::unique_ptr<HypreParMatrix> UniformVelocityHierarchy::BuildLevelOperator(
   int level, const LinearSolverContext &context) const
{
   detail::Check(impl->communicator, level >= 0 && level < static_cast<int>(impl->levels.size()), "Invalid GMG level");
   std::vector<std::unique_ptr<HypreParMatrix>> operators, transfers;
   std::vector<Array<int>> essential;
   BuildOperators(context, operators, transfers, essential);
   return std::move(operators[level]);
}

class GeometricVelocityInverse::Impl
{
public:
   const UniformVelocityHierarchy &hierarchy;
   const FlowProblem &problem;
   std::uint64_t revision;
   bool transpose;
   GeometricMultigridOptions options;
   std::vector<std::unique_ptr<HypreParMatrix>> operators, transfers;
   std::vector<Array<int>> essential;
   std::vector<Vector> weights;
   std::unique_ptr<Solver> coarse;
   mutable GeometricMultigridStatistics statistics;

   Impl(const UniformVelocityHierarchy &hierarchy_, const LinearSolverContext &context,
        const GeometricMultigridOptions &options_)
      : hierarchy(hierarchy_), problem(context.problem), revision(problem.GetRevision()),
        transpose(context.transpose), options(options_)
   {
      const double started = MPI_Wtime();
      options.Validate(problem.GetComm());
      hierarchy.BuildOperators(context, operators, transfers, essential, options.num_levels);
      for (const auto &matrix : operators)
      {
         weights.push_back(JacobiWeights(*matrix, options.damping));
         statistics.level_dofs.push_back(matrix->GetGlobalNumRows());
         statistics.level_nonzeros.push_back(matrix->NNZ());
         statistics.local_csr_bytes += Storage(*matrix);
      }
      for (const auto &matrix : transfers)
      { statistics.transfer_nonzeros += matrix->NNZ(); statistics.local_csr_bytes += Storage(*matrix); }
      const double coarse_started = MPI_Wtime();
      MPI_Comm_size(problem.GetComm(), &statistics.coarse_solver_ranks);
#ifdef MFEM_USE_MUMPS
      if (options.coarse_solver == GeometricCoarseSolver::Mumps)
      {
         auto direct = std::make_unique<MUMPSSolver>(problem.GetComm());
         direct->SetPrintLevel(0);
         direct->SetMatrixSymType(MUMPSSolver::SYMMETRIC_POSITIVE_DEFINITE);
         direct->SetOperator(*operators.front());
         coarse = std::move(direct);
      }
      else if (options.coarse_solver == GeometricCoarseSolver::MumpsRoot)
      {
         coarse = std::make_unique<RootMumpsInverse>(*operators.front(), options);
         statistics.coarse_solver_ranks = 1;
      }
#endif
      const int first_level = static_cast<int>(hierarchy.GetLevels().size() - operators.size());
      auto &coarse_space = *hierarchy.GetLevels()[first_level].space;
      if (options.coarse_solver == GeometricCoarseSolver::BoomerAMG)
      { coarse = std::make_unique<detail::FixedAMGInverse>(*operators.front(), options.coarse_amg_cycles, &coarse_space); }
      if (options.coarse_solver == GeometricCoarseSolver::User)
      {
         coarse = options.coarse_factory(*operators.front(), coarse_space);
         detail::Check(problem.GetComm(), coarse && coarse->Height() == operators.front()->Height() &&
                       coarse->Width() == operators.front()->Width(), "Invalid GMG coarse inverse factory result");
      }
      statistics.coarse_setup_seconds = MPI_Wtime() - coarse_started;
      statistics.setup_seconds = MPI_Wtime() - started;
   }
   void Smooth(int level, const Vector &rhs, Vector &solution, int sweeps) const
   {
      Vector residual(rhs.Size());
      for (int sweep = 0; sweep < sweeps; ++sweep)
      {
         operators[level]->Mult(solution, residual);
         for (int entry = 0; entry < rhs.Size(); ++entry)
         { solution(entry) += weights[level](entry) * (rhs(entry) - residual(entry)); }
      }
   }
   void Cycle(int level, const Vector &rhs, Vector &solution, bool transposed) const
   {
      solution.SetSize(rhs.Size()); solution = 0.0;
      if (level == 0)
      {
         const double started = MPI_Wtime();
         if (coarse)
         {
            Vector free_rhs(rhs);
            for (int dof : essential[level]) { free_rhs(dof) = 0.0; }
            if (transposed && (options.coarse_solver == GeometricCoarseSolver::BoomerAMG ||
                               options.coarse_solver == GeometricCoarseSolver::User))
            { coarse->MultTranspose(free_rhs, solution); }
            else { coarse->Mult(free_rhs, solution); }
         }
         else { Smooth(level, rhs, solution, options.coarse_sweeps); }
         statistics.coarse_apply_seconds += MPI_Wtime() - started;
      }
      else
      {
         Smooth(level, rhs, solution, transposed ? options.post_sweeps : options.pre_sweeps);
         Vector residual(rhs.Size()), restricted(transfers[level - 1]->Width()), correction, prolonged(rhs.Size());
         operators[level]->Mult(solution, residual); residual *= -1.0; residual += rhs;
         transfers[level - 1]->MultTranspose(residual, restricted);
         Cycle(level - 1, restricted, correction, transposed);
         transfers[level - 1]->Mult(correction, prolonged); solution += prolonged;
         Smooth(level, rhs, solution, transposed ? options.pre_sweeps : options.post_sweeps);
      }
      for (int entry : essential[level]) { solution(entry) = rhs(entry); }
   }
   void Apply(const Vector &input, Vector &output, bool transposed) const
   {
      const double started = MPI_Wtime();
      hierarchy.CheckCurrent(); problem.CheckSpacesUnchanged();
      detail::Check(problem.GetComm(), revision == problem.GetRevision(), "Stale GMG inverse; rebuild numerical levels");
      detail::Check(problem.GetComm(), input.Size() == operators.back()->Height(), "Invalid GMG input size");
      bool finite = true;
      for (int entry = 0; entry < input.Size(); ++entry) { finite = finite && std::isfinite(input(entry)); }
      detail::Check(problem.GetComm(), finite, "Nonfinite GMG input");
      Vector copy(input);
      Cycle(static_cast<int>(operators.size()) - 1, copy, output, transposed);
      ++statistics.applications; statistics.apply_seconds += MPI_Wtime() - started;
   }
};

GeometricVelocityInverse::GeometricVelocityInverse(const LinearSolverContext &context,
   const GeometricMultigridOptions &options) : Solver(context.velocity_proxy.Height())
{
   const auto *hierarchy = dynamic_cast<const UniformVelocityHierarchy *>(context.hierarchy);
   detail::Check(context.problem.GetComm(), hierarchy != nullptr, "GMG requires UniformVelocityHierarchy");
   impl = std::make_unique<Impl>(*hierarchy, context, options);
}
GeometricVelocityInverse::~GeometricVelocityInverse() = default;
void GeometricVelocityInverse::SetOperator(const Operator &)
{ MFEM_ABORT("Fixed GMG inverse; reconstruct after operator changes"); }
void GeometricVelocityInverse::Mult(const Vector &input, Vector &output) const
{ impl->Apply(input, output, impl->transpose); }
void GeometricVelocityInverse::MultTranspose(const Vector &input, Vector &output) const
{ impl->Apply(input, output, !impl->transpose); }
const GeometricMultigridStatistics &GeometricVelocityInverse::GetStatistics() const
{ return impl->statistics; }
const HypreParMatrix &GeometricVelocityInverse::GetLevelOperator(int level) const
{ return *impl->operators.at(level); }
const HypreParMatrix &GeometricVelocityInverse::GetCorrectionProlongation(int level) const
{ return *impl->transfers.at(level); }
const Array<int> &GeometricVelocityInverse::GetEssentialDofs(int level) const
{ return impl->essential.at(level); }
InverseFactory MakeGeometricVelocityInverse(const GeometricMultigridOptions &options)
{
   return [options](const LinearSolverContext &context)
   { return std::make_unique<GeometricVelocityInverse>(context, options); };
}

const char *PreconditionerName(PreconditionerType preconditioner)
{
   switch (preconditioner)
   {
      case PreconditionerType::BlockDiagonalAMG: return "block-diagonal-amg";
      case PreconditionerType::BlockDiagonalP1AMG: return "block-diagonal-p1-amg";
      case PreconditionerType::BlockUpperTriangularP1AMG: return "block-upper-p1-amg";
      case PreconditionerType::BlockLowerTriangularP1AMG: return "block-lower-p1-amg";
      case PreconditionerType::BlockDiagonalP1AMGAssembled: return "block-diagonal-p1-amg-assembled";
      case PreconditionerType::BlockUpperTriangularP1AMGAssembled: return "block-upper-p1-amg-assembled";
      case PreconditionerType::BlockLowerTriangularP1AMGAssembled: return "block-lower-p1-amg-assembled";
   }
   return "unknown";
}

const char *StabilizationName(const Stabilization &stabilization)
{
   if (std::holds_alternative<PSPGOptions>(stabilization)) { return "pspg"; }
   if (std::holds_alternative<CIPOptions>(stabilization)) { return "cip"; }
   if (std::holds_alternative<GLSOptions>(stabilization)) { return "gls"; }
   if (std::holds_alternative<NoStabilization>(stabilization)) { return "taylor-hood"; }
   return "unknown";
}

const char *PressureGaugeName(const PressureGauge &gauge)
{
   switch (gauge.GetMode())
   {
      case PressureGauge::Mode::Auto: return "auto";
      case PressureGauge::Mode::MeanValue: return *gauge.GetMean() == 0.0 ? "mean-zero" : "mean-value";
      case PressureGauge::Mode::None: return "none";
   }
   return "unknown";
}

namespace
{

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

class BoundaryScalar : public Coefficient
{
   ScalarData data;
public:
   explicit BoundaryScalar(const ScalarData &data_) : data(data_) { }
   real_t Eval(ElementTransformation &transformation, const IntegrationPoint &point) override
   { return data.Eval(transformation, point); }
};

class BoundaryTraction : public VectorCoefficient
{
   const BoundaryConditions &boundary;
public:
   BoundaryTraction(int dimension, const BoundaryConditions &boundary_)
      : VectorCoefficient(dimension), boundary(boundary_) { }
   void Eval(Vector &value, ElementTransformation &transformation,
             const IntegrationPoint &point) override
   {
      value.SetSize(vdim);
      value = 0.0;
      for (int component = 0; component < vdim; ++component)
      {
         auto entry = boundary.GetTraction().find({transformation.Attribute, component});
         if (entry != boundary.GetTraction().end())
         {
            value(component) = entry->second.Eval(transformation, point);
         }
      }
   }
};

class BoundaryPressure : public Coefficient
{
   const BoundaryConditions &boundary;
public:
   explicit BoundaryPressure(const BoundaryConditions &boundary_) : boundary(boundary_) { }
   real_t Eval(ElementTransformation &transformation, const IntegrationPoint &point) override
   {
      auto entry = boundary.GetPressure().find(transformation.Attribute);
      return entry == boundary.GetPressure().end() ? 0.0 : entry->second.Eval(transformation, point);
   }
};

class BrinkmanForce : public VectorCoefficient
{
   const FlowProblem &problem;
   Vector target;
public:
   explicit BrinkmanForce(const FlowProblem &problem_)
      : VectorCoefficient(problem_.GetVelocitySpace().GetVDim()), problem(problem_), target(vdim) { }
   void Eval(Vector &value, ElementTransformation &transformation,
             const IntegrationPoint &point) override
   {
      problem.GetBodyForce().Eval(value, transformation, point);
      const real_t penalty = problem.GetBrinkmanPenalty().Eval(transformation, point);
      if (penalty != 0.0)
      {
         problem.GetBrinkmanTarget().Eval(target, transformation, point);
         value.Add(penalty, target);
      }
   }
};

}

class detail::FlowAssembly
{
public:
   ParFiniteElementSpace &velocity_space;
   ParFiniteElementSpace &pressure_space;
   ParMesh &mesh;
   MPI_Comm comm;
   const FlowProblem &problem;
   const FlowSolverOptions &settings;
   const Stabilization &stabilization;
   VectorConstantCoefficient zero_gradient;
   BoundaryTraction traction;
   BoundaryPressure outlet_pressure;
   VectorCoefficient *traction_data = nullptr;
   Coefficient *outlet_data = nullptr;
   BrinkmanForce applied_force;
   std::unique_ptr<TrueDofLayout> layout;
   int dimension, rank, ranks, order, pressure_order, geometry_order, quadrature_order, max_boundary_attribute;
   long mesh_sequence, velocity_sequence, pressure_sequence;
   Coefficient *viscosity;
   VectorCoefficient *viscosity_gradient, *force;
   std::map<std::pair<int, int>, std::unique_ptr<Coefficient>> boundary_values;
   bool defer_residual_stabilization;
   Coefficient *brinkman_penalty;
   const Array<int> *prescribed_velocity_dofs;
   VectorCoefficient *prescribed_velocity_values;
   real_t reaction_measure = 0.0;
   bool gauge = false;
   bool pressure_nullspace_detected = false;
   double assembly_seconds = 0.0;
   ElementTau tau;
   Vector element_sizes, element_viscosities;
   Array<int> essential, offsets;
   Array<HYPRE_BigInt> gauge_partition;
   Vector lifting, moments, pressure_constant;
   Vector rhs;
   real_t volume = 0.0;
   std::unique_ptr<HypreParMatrix> velocity_matrix, divergence, continuity;
   std::unique_ptr<HypreParMatrix> momentum_pressure, pressure_stabilization;
   std::unique_ptr<HypreParMatrix> gauge_column, gauge_row, system;

   FlowAssembly(const FlowProblem &problem_, const FlowSolverOptions &settings_)
      : velocity_space(problem_.GetVelocitySpace()), pressure_space(problem_.GetPressureSpace()),
        mesh(*velocity_space.GetParMesh()), comm(problem_.GetComm()), problem(problem_), settings(settings_),
        stabilization(problem_.GetStabilization()),
        zero_gradient(ZeroVector(mesh.Dimension())), traction(mesh.Dimension(), problem_.GetBoundaryConditions()),
        outlet_pressure(problem_.GetBoundaryConditions()), applied_force(problem_), dimension(mesh.Dimension()),
        viscosity(&problem_.GetViscosity()),
        viscosity_gradient(problem_.GetViscosityGradient() ? problem_.GetViscosityGradient() : &zero_gradient),
        force(&applied_force), defer_residual_stabilization(settings_.experimental_residual.has_value()),
        brinkman_penalty(&problem_.GetBrinkmanPenalty()), prescribed_velocity_dofs(&problem_.GetPrescribedVelocityTrueDofs()),
        prescribed_velocity_values(problem_.GetPrescribedVelocityValues())
   {
      problem.Validate();
      Check(!Device::Allows(Backend::DEVICE_MASK), "Flow assembly currently requires CPU execution");
      MPI_Comm_rank(comm, &rank);
      MPI_Comm_size(comm, &ranks);
      mesh_sequence = mesh.GetSequence();
      velocity_sequence = velocity_space.GetSequence();
      pressure_sequence = pressure_space.GetSequence();
      const int local_order = velocity_space.GetMaxElementOrder();
      MPI_Allreduce(&local_order, &order, 1, MPI_INT, MPI_MAX, comm);
      pressure_order = std::holds_alternative<NoStabilization>(stabilization) ? order - 1 : order;
      geometry_order = detail::StokesGeometryOrder(mesh);
      quadrature_order = problem.GetIntegrationOrder();
      const int local_max = mesh.bdr_attributes.Size() ? mesh.bdr_attributes.Max() : 0;
      MPI_Allreduce(&local_max, &max_boundary_attribute, 1, MPI_INT, MPI_MAX, comm);
      for (const auto &entry : problem.GetBoundaryConditions().GetVelocity())
      {
         boundary_values[entry.first] = std::make_unique<BoundaryScalar>(entry.second);
      }
      traction_data = problem.GetBoundaryConditions().GetTraction().empty() ? nullptr : &traction;
      outlet_data = problem.GetBoundaryConditions().GetPressure().empty() ? nullptr : &outlet_pressure;
      Assemble();
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

   void CheckSequences() const
   {
      Check(mesh.GetSequence() == mesh_sequence &&
            velocity_space.GetSequence() == velocity_sequence &&
            pressure_space.GetSequence() == pressure_sequence,
            "Mesh/spaces changed; recreate the Stokes solver");
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
         coefficients[component] = item.second.get();
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
      if (prescribed_velocity_values)
      {
         Check(prescribed_velocity_dofs && basis != BasisType::Positive &&
               prescribed_velocity_values->GetVDim() == dimension,
               "Invalid prescribed velocity DOF data");
         bool valid = true;
         for (int dof : *prescribed_velocity_dofs)
         { valid = valid && dof >= 0 && dof < local_size; }
         Check(valid, "Invalid prescribed velocity true DOF index");
         ParGridFunction projected(&velocity_space), synchronized(&velocity_space), selected(&velocity_space);
         projected.ProjectCoefficient(*prescribed_velocity_values);
         Vector values, mask(local_size);
         projected.GetTrueDofs(values);
         mask = 0.0;
         for (int dof : *prescribed_velocity_dofs)
         {
            const real_t value = values(dof);
            consistent = consistent && std::isfinite(value);
            mask(dof) = 1.0;
            if (prescribed[dof])
            {
               consistent = consistent && std::abs(lifting(dof) - value) <=
                            1e-12 * std::max({real_t(1.0), std::abs(value), std::abs(lifting(dof))});
            }
            else
            {
               prescribed[dof] = 1;
               essential.Append(dof);
               lifting(dof) = value;
            }
         }
         synchronized.SetFromTrueDofs(values);
         selected.SetFromTrueDofs(mask);
         for (int dof = 0; dof < projected.Size(); ++dof)
         {
            if (selected(dof) != 0.0)
            {
               consistent = consistent && std::isfinite(projected(dof)) &&
                            std::abs(projected(dof) - synchronized(dof)) <=
                            1e-12 * std::max({real_t(1.0), std::abs(projected(dof)), std::abs(synchronized(dof))});
            }
         }
      }
      Check(consistent, "Conflicting or nonfinite velocity boundary values or prescribed DOF values");
      essential.Sort();
      CheckRigidModes();
   }

   void ValidateReaction()
   {
      reaction_measure = 0.0;
      if (!brinkman_penalty) { return; }
      bool valid = true;
      real_t local_measure = 0.0;
      for (int element = 0; element < mesh.GetNE(); ++element)
      {
         auto &transformation = *mesh.GetElementTransformation(element);
         const auto &rule = IntRules.Get(mesh.GetElementGeometry(element), quadrature_order);
         for (int point = 0; point < rule.GetNPoints(); ++point)
         {
            const auto &ip = rule.IntPoint(point);
            transformation.SetIntPoint(&ip);
            const real_t penalty = brinkman_penalty->Eval(transformation, ip);
            valid = valid && std::isfinite(penalty) && penalty >= 0.0;
            local_measure += std::abs(ip.weight) * transformation.Weight() * penalty;
         }
      }
      MPI_Allreduce(&local_measure, &reaction_measure, 1,
                    MPITypeMap<real_t>::mpi_type, MPI_SUM, comm);
      Check(valid && std::isfinite(reaction_measure) && reaction_measure >= 0.0,
            "Invalid sampled Brinkman reaction");
      Check(reaction_measure == 0.0 || defer_residual_stabilization ||
            std::holds_alternative<NoStabilization>(stabilization) ||
            std::holds_alternative<CIPOptions>(stabilization),
            "Positive Brinkman reaction with PSPG/GLS requires Stage D residual extensions");
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
      int reaction_samples = 0;
      if (reaction_measure > 0.0)
      {
         for (int element = 0; element < mesh.GetNE(); ++element)
         {
            reaction_samples += dimension * IntRules.Get(mesh.GetElementGeometry(element),
                                                         quadrature_order).GetNPoints();
         }
      }
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
         Vector restricted(essential.Size() + reaction_samples);
         for (int index = 0; index < essential.Size(); ++index)
         {
            restricted(index) = true_values(essential[index]);
         }
         if (reaction_measure > 0.0)
         {
            int index = essential.Size();
            Vector value(dimension);
            for (int element = 0; element < mesh.GetNE(); ++element)
            {
               auto &transformation = *mesh.GetElementTransformation(element);
               const auto &rule = IntRules.Get(mesh.GetElementGeometry(element), quadrature_order);
               for (int point = 0; point < rule.GetNPoints(); ++point)
               {
                  const auto &ip = rule.IntPoint(point);
                  transformation.SetIntPoint(&ip);
                  const real_t weight = std::abs(ip.weight) * transformation.Weight() *
                                        brinkman_penalty->Eval(transformation, ip) /
                                        reaction_measure;
                  rigid.Eval(value, transformation, ip);
                  for (int component = 0; component < dimension; ++component)
                  {
                     restricted(index++) = std::sqrt(weight) * value(component);
                  }
               }
            }
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
      const auto *pspg = std::get_if<PSPGOptions>(&stabilization);
      const auto *gls = std::get_if<GLSOptions>(&stabilization);
      const bool residual_method = pspg || gls;
      const real_t scale = pspg ? pspg->tau_constant : gls ? gls->tau_constant : 0.0;
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
      if (rank == 0 && settings.linear.print_level > 0)
      {
         out << "Stokes: method=" << StabilizationName(stabilization)
             << " dim=" << dimension << " order=" << order << " pressure_order=" << pressure_order
             << " geometry_order=" << geometry_order
             << " quadrature=" << quadrature_order
             << " mu=[" << global_mu_min << "," << global_mu_max << "]"
             << " tau_scale=" << scale
             << " cip_gamma=" << (std::holds_alternative<CIPOptions>(stabilization) ? std::get<CIPOptions>(stabilization).gamma : 0.0)
             << " tau=[" << global_min
             << "," << global_max << "]\n";
      }
   }

   std::unique_ptr<HypreParMatrix> AssembleVelocityBlock(ParFiniteElementSpace &space, bool proxy = false)
   {
      ParBilinearForm form(&space);
      form.AddDomainIntegrator(new RuleBilinearIntegrator<ElasticityIntegrator>(
                                  quadrature_order, *viscosity, 0.0, 1.0));
      if (reaction_measure > 0.0)
      {
         form.AddDomainIntegrator(new RuleBilinearIntegrator<VectorMassIntegrator>(
                                     quadrature_order, *brinkman_penalty));
      }
      if (std::holds_alternative<GLSOptions>(stabilization) && (!defer_residual_stabilization || proxy))
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
      if (std::holds_alternative<CIPOptions>(stabilization))
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
      if (!defer_residual_stabilization &&
          (std::holds_alternative<PSPGOptions>(stabilization) ||
           std::holds_alternative<GLSOptions>(stabilization)))
      {
         form.AddDomainIntegrator(new RuleBilinearIntegrator<DiffusionIntegrator>(quadrature_order, tau));
      }
      else if (std::holds_alternative<CIPOptions>(stabilization))
      {
         form.AddInteriorFaceIntegrator(new PressureCIPIntegrator(
                                           size_coefficient, mu_coefficient, std::get<CIPOptions>(stabilization).gamma, quadrature_order));
      }
      form.Assemble(0);
      form.Finalize(0);
      return std::unique_ptr<HypreParMatrix>(form.ParallelAssemble());
   }

   void Assemble()
   {
      const double assembly_start = MPI_Wtime();
      CheckSequences();
      ValidateGeometry();
      ValidateReaction();
      BuildBoundaryLifting();
      if (std::holds_alternative<CIPOptions>(stabilization) && mesh.GetGlobalNE() <= (1 << dimension))
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
      const bool gls = std::holds_alternative<GLSOptions>(stabilization) && !defer_residual_stabilization;
      const bool residual_method = !defer_residual_stabilization &&
                                   (gls || std::holds_alternative<PSPGOptions>(stabilization));
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

      pressure_stabilization.reset();
      if (!std::holds_alternative<NoStabilization>(stabilization))
      {
         pressure_stabilization = AssemblePressureBlock(pressure_space, false);
      }

      ScalarVectorProductCoefficient tau_force(tau, *force);
      ParLinearForm velocity_rhs(&velocity_space), pressure_rhs(&pressure_space);
      velocity_rhs.AddDomainIntegrator(new RuleLinearIntegrator<VectorDomainLFIntegrator>(
                                         quadrature_order, *force));
      if (traction_data)
      {
         velocity_rhs.AddBoundaryIntegrator(new RuleLinearIntegrator<VectorBoundaryLFIntegrator>(
                                              quadrature_order, *traction_data));
      }
      if (outlet_data)
      {
         velocity_rhs.AddBoundaryIntegrator(new RuleLinearIntegrator<VectorBoundaryFluxLFIntegrator>(
                                              quadrature_order, *outlet_data, -1.0));
      }
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
      Check(problem.GetPressureGauge().GetMode() != PressureGauge::Mode::MeanValue || pressure_nullspace_detected,
            "Pressure mean constraint requires a constant-pressure nullspace; boundary conditions already determine absolute pressure");
      Check(problem.GetPressureGauge().GetMode() != PressureGauge::Mode::None || !pressure_nullspace_detected,
            "Pressure gauge none leaves an unresolved constant-pressure nullspace; use auto or mean-zero");
      gauge = pressure_nullspace_detected && problem.GetPressureGauge().GetMode() != PressureGauge::Mode::None;

      const int block_count = gauge ? 3 : 2;
      offsets.SetSize(block_count + 1);
      offsets[0] = 0;
      offsets[1] = velocity_space.GetTrueVSize();
      offsets[2] = offsets[1] + pressure_space.GetTrueVSize();
      if (gauge) { offsets[3] = offsets[2] + (rank == 0 ? 1 : 0); }
      rhs.SetSize(offsets.Last());
      rhs = 0.0;
      BlockVector rhs_blocks(rhs, offsets);
      rhs_blocks.GetBlock(0) = momentum_rhs;
      rhs_blocks.GetBlock(1) = continuity_rhs;
      if (gauge && rank == 0) { rhs_blocks.GetBlock(2)(0) = problem.GetPressureGauge().GetMean().value_or(0.0) * volume; }
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
      blocks(1, 1) = pressure_stabilization.get();
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
      }
      system.reset(HypreParMatrixFromBlocks(blocks));
      const real_t system_norm = system->FNorm(), rhs_norm = Norm(rhs);
      Check(std::isfinite(system_norm) && std::isfinite(rhs_norm) && rhs.CheckFinite() == 0,
            "Nonfinite Stokes matrix or right-hand side");
      problem.ResolvePressureGauge(gauge);
      layout = std::make_unique<TrueDofLayout>(problem);
      const double elapsed = MPI_Wtime() - assembly_start;
      MPI_Allreduce(&elapsed, &assembly_seconds, 1, MPI_DOUBLE, MPI_MAX, comm);
      if (rank == 0 && settings.linear.print_level > 0)
      {
         out << "Stokes: Nv=" << velocity_space.GlobalTrueVSize()
             << " Np=" << pressure_space.GlobalTrueVSize()
             << " pressure_gauge=" << gauge << " assembly_seconds=" << assembly_seconds << '\n';
      }
   }

   const TrueDofLayout &GetLayout() const { layout->CheckCurrent(); return *layout; }
   const Array<int> &GetEssentialVelocityDofs() const { GetLayout(); return essential; }
   const Vector &GetRHS() const { GetLayout(); return rhs; }
   const HypreParMatrix &GetSystemMatrix() const { GetLayout(); return *system; }
   const HypreParMatrix *GetSystemBlock(int row, int column) const
   {
      GetLayout();
      Check(row >= 0 && column >= 0 && row < (gauge ? 3 : 2) && column < (gauge ? 3 : 2), "Invalid flow block index");
      if (row == 0 && column == 0) { return velocity_matrix.get(); }
      if (row == 0 && column == 1) { return momentum_pressure.get(); }
      if (row == 1 && column == 0) { return continuity.get(); }
      if (row == 1 && column == 1) { return pressure_stabilization.get(); }
      if (row == 1 && column == 2) { return gauge_column.get(); }
      if (row == 2 && column == 1) { return gauge_row.get(); }
      return nullptr;
   }
   std::unique_ptr<HypreParMatrix> AssembleVelocityProxy()
   {
      if (!defer_residual_stabilization) { return std::make_unique<HypreParMatrix>(*velocity_matrix); }
      auto proxy = AssembleVelocityBlock(velocity_space, true);
      std::unique_ptr<HypreParMatrix> removed(proxy->EliminateRowsCols(essential));
      return proxy;
   }
};

namespace detail
{

void UpdateFlowDiagnostics(const FlowProblem &problem, const TrueDofLayout &layout,
                           const Vector &state, const Vector &residual,
                           ParGridFunction &velocity, ParGridFunction &pressure,
                           FlowSolveReport &report)
{
   const auto norm = [&problem](const Vector &value)
   { return std::sqrt(InnerProduct(problem.GetComm(), value, value)); };
   BlockVector fields(const_cast<Vector &>(state), layout.GetOffsets());
   BlockVector equations(const_cast<Vector &>(residual), layout.GetOffsets());
   velocity.SetFromTrueDofs(fields.GetBlock(0));
   pressure.SetFromTrueDofs(fields.GetBlock(1));
   report.absolute_residual = norm(residual);
   report.relative_residual = report.absolute_residual / (report.initial_residual > 0.0 ? report.initial_residual : 1.0);
   report.momentum_residual = norm(equations.GetBlock(0));
   report.continuity_residual = norm(equations.GetBlock(1));
   if (layout.HasGauge())
   {
      real_t multiplier = 0.0;
      int rank = 0;
      MPI_Comm_rank(problem.GetComm(), &rank);
      if (rank == 0) { multiplier = fields.GetBlock(2)(0); }
      MPI_Bcast(&multiplier, 1, MPITypeMap<real_t>::mpi_type, 0, problem.GetComm());
      report.gauge_multiplier = multiplier;
      report.gauge_residual = norm(equations.GetBlock(2));
   }
   auto &mesh = *problem.GetVelocitySpace().GetParMesh();
   const int order = problem.GetIntegrationOrder();
   real_t local[4] = {}, global[4] = {};
   for (int element = 0; element < mesh.GetNE(); ++element)
   {
      auto &transformation = *mesh.GetElementTransformation(element);
      const auto &rule = IntRules.Get(mesh.GetElementGeometry(element), order);
      for (int point = 0; point < rule.GetNPoints(); ++point)
      {
         const auto &ip = rule.IntPoint(point);
         transformation.SetIntPoint(&ip);
         const real_t weight = ip.weight * transformation.Weight();
         const real_t divergence = velocity.GetDivergence(transformation);
         local[0] += weight;
         local[1] += weight * pressure.GetValue(transformation, ip);
         local[2] += weight * divergence;
         local[3] += weight * divergence * divergence;
      }
   }
   MPI_Allreduce(local, global, 4, MPITypeMap<real_t>::mpi_type, MPI_SUM, problem.GetComm());
   report.pressure_mean = global[1] / global[0];
   report.boundary_flux = global[2];
   report.divergence_l2 = std::isfinite(global[3]) ? std::sqrt(std::max(real_t(0.0), global[3])) : global[3];
   report.gauge_error = layout.HasGauge() ? std::abs(report.pressure_mean - *problem.GetPressureGauge().GetMean()) : 0.0;
}

}

namespace
{

template <typename Base> class ConvectionIntegrator : public Base
{
   int order;
public:
   explicit ConvectionIntegrator(int order_) : order(order_) { }
   void AssembleElementVector(const FiniteElement &element, ElementTransformation &transformation,
                              const Vector &state, Vector &residual) override
   {
      this->SetIntRule(&IntRules.Get(element.GetGeomType(), order));
      Base::AssembleElementVector(element, transformation, state, residual);
   }
   void AssembleElementGrad(const FiniteElement &element, ElementTransformation &transformation,
                            const Vector &state, DenseMatrix &gradient) override
   {
      this->SetIntRule(&IntRules.Get(element.GetGeomType(), order));
      Base::AssembleElementGrad(element, transformation, state, gradient);
   }
};

std::vector<HYPRE_BigInt> SparsityPattern(const HypreParMatrix &matrix)
{
   SparseMatrix diagonal, off_diagonal;
   HYPRE_BigInt *columns = nullptr;
   matrix.GetDiag(diagonal);
   matrix.GetOffd(off_diagonal, columns);
   std::vector<HYPRE_BigInt> pattern{matrix.Height(), matrix.Width(),
                                    matrix.GetGlobalNumRows(), matrix.GetGlobalNumCols()};
   for (const auto *block : {&diagonal, &off_diagonal})
   {
      pattern.push_back(block->Width());
      pattern.push_back(block->NumNonZeroElems());
      for (int row = 0; row <= block->Height(); ++row) { pattern.push_back(block->GetI()[row]); }
      for (int entry = 0; entry < block->NumNonZeroElems(); ++entry) { pattern.push_back(block->GetJ()[entry]); }
   }
   for (int column = 0; column < off_diagonal.Width(); ++column) { pattern.push_back(columns[column]); }
   return pattern;
}

}

class FlowSolver::Impl
{
public:
   const FlowProblem &problem;
   FlowSolverOptions options;
   std::unique_ptr<detail::FlowAssembly> baseline;
   std::unique_ptr<ParNonlinearForm> convection, oseen;
   std::unique_ptr<ParBlockNonlinearForm> stabilization, picard_stabilization, stokes_stabilization;
   std::uint64_t revision = 0;
   bool assembled = false;
   std::unique_ptr<HypreParMatrix> linear_matrix, linear_proxy;
   std::vector<std::unique_ptr<HypreParMatrix>> linear_owned_blocks;
   Array2D<const HypreParMatrix *> linear_blocks;
   Vector linear_rhs, linear_state;
   std::unique_ptr<detail::BlockFGMRESSolver> linear_iterative;
#ifdef MFEM_USE_MUMPS
   std::unique_ptr<MUMPSSolver> linear_direct;
#endif
   double assembly_seconds = 0.0;

   Impl(const FlowProblem &problem_, const FlowSolverOptions &options_)
      : problem(problem_), options(options_) { options.Validate(problem.GetComm()); }

   real_t Norm(const Vector &vector) const
   { return std::sqrt(InnerProduct(problem.GetComm(), vector, vector)); }

   bool All(bool valid) const
   {
      int local = valid ? 1 : 0, global = 0;
      MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_MIN, problem.GetComm());
      return global != 0;
   }

   void Assemble()
   {
      const double started = MPI_Wtime();
      problem.Validate();
      detail::Check(problem.GetComm(), options.gauge_enforcement != GaugeEnforcement::NullspaceProjection ||
                    problem.GetPressureGauge().GetMean().has_value(), "Nullspace projection requires a pressure mean");
      detail::Check(problem.GetComm(), !options.experimental_residual ||
                    std::holds_alternative<PSPGOptions>(problem.GetStabilization()) ||
                    std::holds_alternative<GLSOptions>(problem.GetStabilization()),
                    "Experimental residual stabilization requires PSPG or GLS");
      detail::Check(problem.GetComm(), !options.convection || options.experimental_residual ||
                    std::holds_alternative<NoStabilization>(problem.GetStabilization()) ||
                    std::holds_alternative<CIPOptions>(problem.GetStabilization()),
                    "Convection with PSPG/GLS requires Stage D residual and Jacobian extensions");
      assembled = false;
      linear_iterative.reset();
#ifdef MFEM_USE_MUMPS
      linear_direct.reset();
#endif
      linear_matrix.reset(); linear_proxy.reset(); linear_owned_blocks.clear();
      baseline = std::make_unique<detail::FlowAssembly>(problem, options);
      convection.reset();
      oseen.reset();
      stabilization.reset();
      picard_stabilization.reset();
      stokes_stabilization.reset();
      if (options.convection)
      {
         const int order = problem.GetConvectionIntegrationOrder();
         auto &mesh = *problem.GetVelocitySpace().GetParMesh();
         bool valid = true;
         for (int element = 0; element < mesh.GetNE(); ++element)
         {
            auto &transformation = *mesh.GetElementTransformation(element);
            const auto &rule = IntRules.Get(mesh.GetElementGeometry(element), order);
            for (int point = 0; point < rule.GetNPoints(); ++point)
            {
               transformation.SetIntPoint(&rule.IntPoint(point));
               const real_t determinant = transformation.Jacobian().Det();
               valid = valid && std::isfinite(determinant) && determinant > 0.0;
            }
         }
         detail::Check(problem.GetComm(), valid,
                       "Invalid Navier-Stokes geometry: nonpositive/nonfinite convection-quadrature signed Jacobian");
         convection = std::make_unique<ParNonlinearForm>(&problem.GetVelocitySpace());
         convection->AddDomainIntegrator(new ConvectionIntegrator<VectorConvectionNLFIntegrator>(order));
         oseen = std::make_unique<ParNonlinearForm>(&problem.GetVelocitySpace());
         oseen->AddDomainIntegrator(new ConvectionIntegrator<ConvectiveVectorConvectionNLFIntegrator>(order));
      }
      if (options.experimental_residual)
      {
         Array<ParFiniteElementSpace *> spaces(2);
         spaces[0] = &problem.GetVelocitySpace(); spaces[1] = &problem.GetPressureSpace();
         auto create = [&](bool advect, bool exact)
         {
            auto form = std::make_unique<ParBlockNonlinearForm>(spaces);
            form->AddDomainIntegrator(new detail::ResidualStabilizationIntegrator(problem,
                                      *options.experimental_residual, advect, exact,
                                      advect ? problem.GetStabilizationIntegrationOrder() : problem.GetIntegrationOrder()));
            return form;
         };
         stabilization = create(options.convection, true);
         picard_stabilization = create(options.convection, false);
         stokes_stabilization = create(false, true);
      }
      revision = problem.GetRevision();
      assembled = true;
      if (!options.convection)
      {
         const auto &layout = baseline->GetLayout();
         linear_state.SetSize(layout.LocalSize()); linear_state = 0.0;
         Residual(linear_state, linear_rhs); linear_rhs *= -1.0;
         const int count = layout.HasGauge() ? 3 : 2;
         linear_blocks.SetSize(count, count);
         const auto *gradient = StabilizationGradient(linear_state, NonlinearMethod::Newton, true);
         for (int row = 0; row < count; ++row)
         {
            for (int column = 0; column < count; ++column)
            {
               linear_owned_blocks.push_back(JacobianBlock(linear_state, row, column, NonlinearMethod::Newton, true, gradient));
               linear_blocks(row, column) = linear_owned_blocks.back().get();
            }
         }
         linear_matrix.reset(HypreParMatrixFromBlocks(linear_blocks));
         linear_proxy = baseline->AssembleVelocityProxy();
      }
      const double elapsed = MPI_Wtime() - started;
      MPI_Allreduce(&elapsed, &assembly_seconds, 1, MPI_DOUBLE, MPI_MAX, problem.GetComm());
   }

   const TrueDofLayout &GetLayout() const
   {
      detail::Check(problem.GetComm(), assembled && baseline, "Assemble before requesting the flow layout");
      return baseline->GetLayout();
   }

   void CheckLinear() const
   {
      GetLayout();
      detail::Check(problem.GetComm(), !options.convection && linear_matrix,
                    "System matrix/RHS access requires a current nonconvective flow assembly");
   }

   FlowSolveReport SolveLinear(ParGridFunction &velocity, ParGridFunction &pressure)
   {
      const double started = MPI_Wtime();
      detail::Check(problem.GetComm(), velocity.ParFESpace() == &problem.GetVelocitySpace() &&
                    pressure.ParFESpace() == &problem.GetPressureSpace(), "Incompatible flow output spaces");
      FlowSolveReport report;
      report.assembly_performed = !assembled || revision != problem.GetRevision();
      EnsureAssembled();
      report.assembly_seconds = assembly_seconds;
      report.geometry_order = baseline->geometry_order;
      report.integration_order = baseline->quadrature_order;
      report.pressure_nullspace_detected = baseline->pressure_nullspace_detected;
      report.initial_residual = Norm(linear_rhs);
      report.linear_solves = 1;
      Vector solution(linear_rhs.Size());
      double phase_started = MPI_Wtime();
      if (std::holds_alternative<IterativeOptions>(options.linear.backend))
      {
         if (!linear_iterative)
         {
            linear_iterative = std::make_unique<detail::BlockFGMRESSolver>(problem, options,
               GetLayout(), baseline->essential, linear_state, linear_blocks, *linear_proxy, false,
               [this](ParFiniteElementSpace &space) { return baseline->AssembleVelocityBlock(space, true); },
               [this](ParFiniteElementSpace &space) { return baseline->AssemblePressureBlock(space, true); });
            report.setup_seconds = MPI_Wtime() - phase_started;
         }
         const auto result = linear_iterative->Solve(linear_rhs, solution);
         report.converged = result.converged;
         report.linear_iterations = result.iterations;
         report.linear_residual_restarts = result.residual_restarts;
         report.solve_seconds = result.solve_seconds;
         report.preconditioner_seconds = result.preconditioner_seconds;
      }
      else
      {
#ifdef MFEM_USE_MUMPS
         if (!linear_direct)
         {
            linear_direct = std::make_unique<MUMPSSolver>(problem.GetComm());
            linear_direct->SetPrintLevel(0);
            linear_direct->SetMatrixSymType(MUMPSSolver::UNSYMMETRIC);
            linear_direct->SetOperator(*linear_matrix);
            report.setup_seconds = MPI_Wtime() - phase_started;
            report.symbolic_factorizations = 1;
         }
         phase_started = MPI_Wtime();
         linear_direct->Mult(linear_rhs, solution);
         report.solve_seconds = MPI_Wtime() - phase_started;
         report.converged = true;
#else
         MFEM_ABORT("Flow MUMPS support is unavailable");
#endif
      }
      phase_started = MPI_Wtime();
      Vector residual(linear_rhs.Size());
      linear_matrix->Mult(solution, residual); residual -= linear_rhs;
      report.absolute_residual = Norm(residual);
      report.relative_residual = report.absolute_residual / (report.initial_residual > 0.0 ? report.initial_residual : 1.0);
      const bool finite = All(solution.CheckFinite() == 0 && residual.CheckFinite() == 0) &&
                          std::isfinite(report.absolute_residual);
      report.converged = report.converged && finite && report.absolute_residual <=
                         std::max(options.linear.absolute_tolerance, options.linear.relative_tolerance * report.initial_residual);
      report.termination = report.converged ? TerminationReason::Converged :
                           finite ? TerminationReason::LinearFailure : TerminationReason::NonfiniteResidual;
      if (finite)
      { detail::UpdateFlowDiagnostics(problem, GetLayout(), solution, residual, velocity, pressure, report); }
      else
      {
         BlockVector fields(solution, GetLayout().GetOffsets());
         velocity.SetFromTrueDofs(fields.GetBlock(0)); pressure.SetFromTrueDofs(fields.GetBlock(1));
         if (GetLayout().HasGauge())
         {
            real_t multiplier = fields.GetBlock(2).Size() ? fields.GetBlock(2)(0) : 0.0;
            MPI_Bcast(&multiplier, 1, MPITypeMap<real_t>::mpi_type, 0, problem.GetComm());
            report.gauge_multiplier = multiplier;
         }
      }
      report.residual_history = {report.initial_residual, report.absolute_residual};
      report.linear_residuals = {report.absolute_residual};
      report.finalization_seconds = MPI_Wtime() - phase_started;
      const double local[] = {report.setup_seconds, report.solve_seconds, report.preconditioner_seconds,
                              report.finalization_seconds, MPI_Wtime() - started};
      double global[5];
      MPI_Allreduce(local, global, 5, MPI_DOUBLE, MPI_MAX, problem.GetComm());
      report.setup_seconds = global[0]; report.solve_seconds = global[1]; report.preconditioner_seconds = global[2];
      report.finalization_seconds = global[3]; report.total_seconds = global[4];
      return report;
   }

   void EnsureAssembled()
   {
      problem.CheckSpacesUnchanged();
      if (!assembled || revision != problem.GetRevision()) { Assemble(); }
   }

   void ImposeVelocity(Vector &state) const
   {
      const Vector &rhs = baseline->GetRHS();
      for (const int dof : baseline->GetEssentialVelocityDofs()) { state(dof) = rhs(dof); }
   }

   void VelocityState(const Vector &state, Vector &velocity) const
   {
      velocity.SetSize(problem.GetVelocitySpace().GetTrueVSize());
      for (int entry = 0; entry < velocity.Size(); ++entry) { velocity(entry) = state(entry); }
      ImposeVelocity(velocity);
   }

   void Residual(const Vector &state, Vector &residual, bool include_convection = true) const
   {
      residual.SetSize(state.Size());
      baseline->GetSystemMatrix().Mult(state, residual);
      residual -= baseline->GetRHS();
      if (convection && include_convection)
      {
         Vector velocity, value(problem.GetVelocitySpace().GetTrueVSize());
         VelocityState(state, velocity);
         convection->Mult(velocity, value);
         for (const int dof : baseline->GetEssentialVelocityDofs()) { value(dof) = 0.0; }
         for (int entry = 0; entry < value.Size(); ++entry) { residual(entry) += value(entry); }
      }
      if (stabilization)
      {
         auto &form = include_convection ? *stabilization : *stokes_stabilization;
         Vector physical(form.Width()), value(form.Height());
         for (int entry = 0; entry < physical.Size(); ++entry) { physical(entry) = state(entry); }
         ImposeVelocity(physical);
         form.Mult(physical, value);
         for (const int dof : baseline->GetEssentialVelocityDofs()) { value(dof) = 0.0; }
         for (int entry = 0; entry < value.Size(); ++entry) { residual(entry) += value(entry); }
      }
   }

   std::unique_ptr<HypreParMatrix> VelocityJacobian(const Vector &state, NonlinearMethod method,
                                                  bool include_convection = true) const
   {
      const auto *viscous = baseline->GetSystemBlock(0, 0);
      if (!convection || !include_convection) { return std::make_unique<HypreParMatrix>(*viscous); }
      Vector velocity;
      VelocityState(state, velocity);
      auto &form = method == NonlinearMethod::Newton ? *convection : *oseen;
      auto *gradient = dynamic_cast<HypreParMatrix *>(&form.GetGradient(velocity));
      detail::Check(problem.GetComm(), gradient != nullptr, "Expected a Hypre convection gradient");
      HypreParMatrix correction(*gradient);
      correction.EliminateRows(baseline->GetEssentialVelocityDofs());
      std::unique_ptr<HypreParMatrix> eliminated(correction.EliminateCols(baseline->GetEssentialVelocityDofs()));
      return std::unique_ptr<HypreParMatrix>(Add(1.0, *viscous, 1.0, correction));
   }

   BlockOperator *StabilizationGradient(const Vector &state, NonlinearMethod method, bool include_convection) const
   {
      if (!stabilization) { return nullptr; }
      auto &form = !include_convection ? *stokes_stabilization :
                   method == NonlinearMethod::Newton ? *stabilization : *picard_stabilization;
      Vector physical(form.Width());
      for (int entry = 0; entry < physical.Size(); ++entry) { physical(entry) = state(entry); }
      ImposeVelocity(physical);
      return &form.GetGradient(physical);
   }

   std::unique_ptr<HypreParMatrix> JacobianBlock(const Vector &state, int row, int column,
                                                NonlinearMethod method, bool include_convection,
                                                const BlockOperator *gradient) const
   {
      const auto *base = baseline->GetSystemBlock(row, column);
      auto result = row == 0 && column == 0 ? VelocityJacobian(state, method, include_convection) :
                    base ? std::make_unique<HypreParMatrix>(*base) : nullptr;
      if (gradient && row < 2 && column < 2)
      {
         const auto *block = dynamic_cast<const HypreParMatrix *>(&gradient->GetBlock(row, column));
         detail::Check(problem.GetComm(), block != nullptr, "Expected a Hypre stabilization gradient");
         auto correction = std::make_unique<HypreParMatrix>(*block);
         if (row == 0) { correction->EliminateRows(baseline->GetEssentialVelocityDofs()); }
         if (column == 0)
         { std::unique_ptr<HypreParMatrix> eliminated(correction->EliminateCols(baseline->GetEssentialVelocityDofs())); }
         result = result ? std::unique_ptr<HypreParMatrix>(Add(1.0, *result, 1.0, *correction)) : std::move(correction);
      }
      return result;
   }

   std::unique_ptr<HypreParMatrix> Jacobian(const Vector &state, NonlinearMethod method,
                                          bool include_convection = true) const
   {
      if (!convection && !stabilization) { return std::make_unique<HypreParMatrix>(baseline->GetSystemMatrix()); }
      const auto *gradient = StabilizationGradient(state, method, include_convection);
      const int count = baseline->GetLayout().HasGauge() ? 3 : 2;
      Array2D<const HypreParMatrix *> blocks(count, count);
      std::vector<std::unique_ptr<HypreParMatrix>> owned;
      for (int row = 0; row < count; ++row)
      {
         for (int column = 0; column < count; ++column)
         {
            owned.push_back(JacobianBlock(state, row, column, method, include_convection, gradient));
            blocks(row, column) = owned.back().get();
         }
      }
      return std::unique_ptr<HypreParMatrix>(HypreParMatrixFromBlocks(blocks));
   }

   detail::LinearSolveResult IterativeSolve(const Vector &state, const Vector &rhs, Vector &step,
                                            FlowSolveReport &report, bool include_convection = true) const
   {
      double started = MPI_Wtime();
      const auto *gradient = StabilizationGradient(state, options.nonlinear_method, include_convection);
      const int count = baseline->GetLayout().HasGauge() ? 3 : 2;
      Array2D<const HypreParMatrix *> blocks(count, count);
      std::vector<std::unique_ptr<HypreParMatrix>> owned;
      for (int row = 0; row < count; ++row)
      {
         for (int column = 0; column < count; ++column)
         {
            owned.push_back(JacobianBlock(state, row, column, options.nonlinear_method, include_convection, gradient));
            blocks(row, column) = owned.back().get();
         }
      }
      auto proxy = baseline->AssembleVelocityProxy();
      report.assembly_seconds += MPI_Wtime() - started;
      started = MPI_Wtime();
      auto linear_options = options;
      linear_options.convection = options.convection && include_convection;
      detail::BlockFGMRESSolver linear(problem, linear_options, baseline->GetLayout(), baseline->GetEssentialVelocityDofs(),
                                       state, blocks, *proxy, false,
                                       [this](ParFiniteElementSpace &space) { return baseline->AssembleVelocityBlock(space, true); },
                                       [this](ParFiniteElementSpace &space) { return baseline->AssemblePressureBlock(space, true); });
      report.setup_seconds += MPI_Wtime() - started;
      auto result = linear.Solve(rhs, step);
      report.solve_seconds += result.solve_seconds;
      report.preconditioner_seconds += result.preconditioner_seconds;
      report.linear_residual_restarts += result.residual_restarts;
      return result;
   }

   void Diagnostics(const Vector &state, const Vector &residual, ParGridFunction &velocity,
                    ParGridFunction &pressure, FlowSolveReport &report) const
   {
      detail::UpdateFlowDiagnostics(problem, baseline->GetLayout(), state, residual, velocity, pressure, report);
   }

   FlowSolveReport SolveNonlinear(ParGridFunction &velocity, ParGridFunction &pressure,
                     std::optional<real_t> initial_gauge)
   {
      const double started = MPI_Wtime();
      FlowSolveReport report;
      detail::Check(problem.GetComm(), velocity.ParFESpace() == &problem.GetVelocitySpace() &&
                    pressure.ParFESpace() == &problem.GetPressureSpace(), "Incompatible Navier-Stokes output spaces");
      detail::CheckReplicated(problem.GetComm(), {initial_gauge ? 1.0 : 0.0, initial_gauge.value_or(0.0)});
      report.assembly_performed = !assembled || revision != problem.GetRevision();
      EnsureAssembled();
      report.pressure_nullspace_detected = baseline->pressure_nullspace_detected;
      report.geometry_order = detail::StokesGeometryOrder(*problem.GetVelocitySpace().GetParMesh());
      report.integration_order = problem.GetIntegrationOrder();
      const auto &layout = baseline->GetLayout();
      detail::Check(problem.GetComm(), !initial_gauge ||
                    (layout.HasGauge() && options.initial_guess == InitialGuess::Provided),
                    "An initial gauge multiplier requires a provided, augmented initial guess");
      detail::Check(problem.GetComm(), options.gauge_enforcement != GaugeEnforcement::NullspaceProjection ||
                    !initial_gauge || *initial_gauge == 0.0, "Projected initial multiplier must be zero");
      report.assembly_seconds = MPI_Wtime() - started;
      bool initial_success = true;
      if (options.initial_guess == InitialGuess::Stokes)
      {
         double phase_started = MPI_Wtime();
         Vector initial_state(layout.LocalSize()), initial_residual, step(layout.LocalSize()), error(layout.LocalSize());
         initial_state = 0.0; ImposeVelocity(initial_state);
         Residual(initial_state, initial_residual, false);
         auto matrix = Jacobian(initial_state, NonlinearMethod::Newton, false);
         const real_t initial_norm = Norm(initial_residual);
         initial_residual *= -1.0;
         report.assembly_seconds += MPI_Wtime() - phase_started;
         if (std::holds_alternative<IterativeOptions>(options.linear.backend))
         {
            const auto result = IterativeSolve(initial_state, initial_residual, step, report, false);
            report.initial_linear_iterations = result.iterations;
            initial_success = result.converged;
         }
         else
         {
#ifdef MFEM_USE_MUMPS
            MUMPSSolver initial_direct(problem.GetComm());
            initial_direct.SetPrintLevel(0);
            initial_direct.SetMatrixSymType(MUMPSSolver::UNSYMMETRIC);
            phase_started = MPI_Wtime();
            initial_direct.SetOperator(*matrix);
            report.setup_seconds += MPI_Wtime() - phase_started;
            phase_started = MPI_Wtime();
            initial_direct.Mult(initial_residual, step);
            report.solve_seconds += MPI_Wtime() - phase_started;
#else
            MFEM_ABORT("Navier-Stokes MUMPS support is unavailable");
#endif
         }
         matrix->Mult(step, error); error -= initial_residual;
         initial_success = initial_success && All(step.CheckFinite() == 0 && error.CheckFinite() == 0) &&
                           Norm(error) <= std::max(options.linear.absolute_tolerance, options.linear.relative_tolerance * initial_norm);
         if (initial_success) { initial_state += step; }
         ImposeVelocity(initial_state);
         BlockVector fields(initial_state, layout.GetOffsets());
         velocity.SetFromTrueDofs(fields.GetBlock(0)); pressure.SetFromTrueDofs(fields.GetBlock(1));
         if (layout.HasGauge())
         {
            real_t multiplier = fields.GetBlock(2).Size() ? fields.GetBlock(2)(0) : 0.0;
            MPI_Bcast(&multiplier, 1, MPITypeMap<real_t>::mpi_type, 0, problem.GetComm());
            initial_gauge = multiplier;
         }
      }
      if (layout.HasGauge() && !initial_gauge) { initial_gauge = 0.0; }
      FlowState initial{velocity, pressure, initial_gauge};
      Vector state, residual;
      initial.Pack(layout, state);
      ImposeVelocity(state);
      double assembly_started = MPI_Wtime();
      Residual(state, residual);
      report.initial_residual = Norm(residual);
      report.residual_history.push_back(report.initial_residual);
      report.assembly_seconds += MPI_Wtime() - assembly_started;
      real_t norm = report.initial_residual;
      const real_t tolerance = std::max(options.absolute_tolerance, options.relative_tolerance * norm);
      if (!All(residual.CheckFinite() == 0) || !std::isfinite(norm))
      { report.termination = TerminationReason::NonfiniteResidual; }
      else if (!initial_success) { report.termination = TerminationReason::LinearFailure; }
#ifdef MFEM_USE_MUMPS
      MUMPSSolver direct(problem.GetComm());
      direct.SetPrintLevel(0);
      direct.SetMatrixSymType(MUMPSSolver::UNSYMMETRIC);
#endif
      std::vector<HYPRE_BigInt> previous_pattern;
      int rank = 0;
      MPI_Comm_rank(problem.GetComm(), &rank);
      while (report.termination == TerminationReason::NotRun && norm > tolerance &&
             report.nonlinear_iterations < options.max_iterations)
      {
         assembly_started = MPI_Wtime();
         auto jacobian = Jacobian(state, options.nonlinear_method);
         const bool finite = std::isfinite(jacobian->FNorm());
         report.assembly_seconds += MPI_Wtime() - assembly_started;
         if (!finite) { report.termination = TerminationReason::LinearFailure; break; }
         Vector rhs(residual), step(state.Size()), error(state.Size());
         rhs *= -1.0;
         int iterations = 0;
         bool linear_success = true;
         if (std::holds_alternative<IterativeOptions>(options.linear.backend))
         {
            const auto result = IterativeSolve(state, rhs, step, report);
            iterations = result.iterations;
            linear_success = result.converged;
         }
         else
         {
            const double setup_started = MPI_Wtime();
            auto pattern = SparsityPattern(*jacobian);
            const bool reuse = All(!previous_pattern.empty() && pattern == previous_pattern);
            if (!reuse) { ++report.symbolic_factorizations; }
            previous_pattern = std::move(pattern);
#ifdef MFEM_USE_MUMPS
            direct.SetReorderingReuse(reuse);
            direct.SetOperator(*jacobian);
            report.setup_seconds += MPI_Wtime() - setup_started;
            const double solve_started = MPI_Wtime();
            direct.Mult(rhs, step);
            report.solve_seconds += MPI_Wtime() - solve_started;
#else
            MFEM_ABORT("Navier-Stokes MUMPS support is unavailable");
#endif
         }
         ++report.linear_solves;
         report.linear_iterations += iterations;
         report.linear_iterations_per_step.push_back(iterations);
         jacobian->Mult(step, error);
         error += residual;
         const real_t linear_error = Norm(error);
         report.linear_residuals.push_back(linear_error);
         if (!linear_success || !All(step.CheckFinite() == 0 && error.CheckFinite() == 0) ||
             !std::isfinite(linear_error) ||
             linear_error > std::max(options.linear.absolute_tolerance, options.linear.relative_tolerance * norm))
         { report.termination = TerminationReason::LinearFailure; break; }
         bool accepted = false;
         real_t damping = options.initial_damping;
         while (true)
         {
            Vector trial(state), trial_residual;
            trial.Add(damping, step);
            ImposeVelocity(trial);
            assembly_started = MPI_Wtime();
            Residual(trial, trial_residual);
            const real_t trial_norm = Norm(trial_residual);
            report.assembly_seconds += MPI_Wtime() - assembly_started;
            if (All(trial.CheckFinite() == 0 && trial_residual.CheckFinite() == 0) &&
                std::isfinite(trial_norm) &&
                (trial_norm <= (1.0 - options.armijo_constant * damping) * norm || trial_norm <= tolerance))
            {
               state = trial;
               residual = trial_residual;
               norm = trial_norm;
               report.residual_history.push_back(norm);
               report.accepted_step_lengths.push_back(damping);
               ++report.nonlinear_iterations;
               accepted = true;
               if (rank == 0 && options.linear.print_level > 0)
               { out << "Navier-Stokes: iteration=" << report.nonlinear_iterations << " residual=" << norm << " damping=" << damping << '\n'; }
               break;
            }
            ++report.line_search_backtracks;
            if (damping <= options.minimum_damping) { break; }
            damping = std::max(options.minimum_damping, damping * options.line_search_reduction);
         }
         if (!accepted) { report.termination = TerminationReason::LineSearchFailure; }
      }
      if (report.termination == TerminationReason::NotRun)
      {
         report.converged = norm <= tolerance;
         report.termination = report.converged ? TerminationReason::Converged : TerminationReason::IterationLimit;
      }
      Diagnostics(state, residual, velocity, pressure, report);
      const double local[5] = {report.assembly_seconds, report.setup_seconds, report.solve_seconds,
                              MPI_Wtime() - started, report.preconditioner_seconds};
      double global[5];
      MPI_Allreduce(local, global, 5, MPI_DOUBLE, MPI_MAX, problem.GetComm());
      report.assembly_seconds = global[0];
      report.setup_seconds = global[1];
      report.solve_seconds = global[2];
      report.total_seconds = global[3];
      report.preconditioner_seconds = global[4];
      report.linear_iterations += report.initial_linear_iterations;
      return report;
   }
};

FlowSolver::FlowSolver(const FlowProblem &problem, const FlowSolverOptions &options)
   : impl(std::make_unique<Impl>(problem, options)) { }
FlowSolver::~FlowSolver() = default;
void FlowSolver::Assemble() { impl->Assemble(); }
const TrueDofLayout &FlowSolver::GetLayout() const { return impl->GetLayout(); }
const Array<int> &FlowSolver::GetEssentialVelocityDofs() const
{ GetLayout(); return impl->baseline->essential; }
const HypreParMatrix &FlowSolver::GetSystemMatrix() const
{ impl->CheckLinear(); return *impl->linear_matrix; }
const Vector &FlowSolver::GetRHS() const
{ impl->CheckLinear(); return impl->linear_rhs; }
const HypreParMatrix *FlowSolver::GetSystemBlock(int row, int column) const
{
   impl->CheckLinear();
   detail::CheckReplicated(impl->problem.GetComm(), {static_cast<real_t>(row), static_cast<real_t>(column)});
   detail::Check(impl->problem.GetComm(), row >= 0 && column >= 0 &&
                 row < impl->linear_blocks.NumRows() && column < impl->linear_blocks.NumCols(), "Invalid flow block index");
   return impl->linear_blocks(row, column);
}
FlowSolveReport FlowSolver::Solve(ParGridFunction &velocity, ParGridFunction &pressure,
                                 std::optional<real_t> initial_gauge_multiplier)
{
   if (impl->options.convection) { return impl->SolveNonlinear(velocity, pressure, initial_gauge_multiplier); }
   detail::Check(impl->problem.GetComm(), !initial_gauge_multiplier,
                 "An initial gauge multiplier is only supported with convection enabled");
   return impl->SolveLinear(velocity, pressure);
}

void FlowSolver::AssembleResidual(const FlowState &state, Vector &residual)
{
   impl->EnsureAssembled();
   detail::Check(impl->problem.GetComm(), &residual != &state.velocity && &residual != &state.pressure,
                 "Navier-Stokes residual output must not alias primal fields");
   Vector packed;
   state.Pack(GetLayout(), packed);
   impl->Residual(packed, residual);
}

std::unique_ptr<HypreParMatrix> FlowSolver::AssembleJacobian(const FlowState &state)
{
   impl->EnsureAssembled();
   Vector packed;
   state.Pack(GetLayout(), packed);
   return impl->Jacobian(packed, NonlinearMethod::Newton);
}

std::unique_ptr<HypreParMatrix> FlowSolver::AssembleVelocityProxy()
{
   impl->EnsureAssembled();
   return impl->baseline->AssembleVelocityProxy();
}

std::unique_ptr<HypreParMatrix> FlowSolver::AssembleJacobianBlock(const FlowState &state,
                                                                           int row, int column)
{
   impl->EnsureAssembled();
   Vector packed;
   state.Pack(GetLayout(), packed);
   detail::CheckReplicated(impl->problem.GetComm(), {static_cast<real_t>(row), static_cast<real_t>(column)});
   const int count = GetLayout().HasGauge() ? 3 : 2;
   detail::Check(impl->problem.GetComm(), row >= 0 && row < count && column >= 0 && column < count,
                 "Invalid Navier-Stokes Jacobian block");
   return impl->JacobianBlock(packed, row, column, NonlinearMethod::Newton, true,
                             impl->StabilizationGradient(packed, NonlinearMethod::Newton, true));
}

std::unique_ptr<FlowLinearization> FlowSolver::Linearize(const FlowState &state)
{
   Vector residual;
   AssembleResidual(state, residual);
   return std::make_unique<FlowLinearization>(impl->problem, state, AssembleJacobian(state), residual);
}

void AdjointOptions::Validate(MPI_Comm communicator) const
{
   linear.Validate(communicator);
   detail::CheckReplicated(communicator, {forward_relative_tolerance, forward_absolute_tolerance,
                                         allow_nonconverged_state ? 1.0 : 0.0});
   detail::Check(communicator, std::isfinite(forward_relative_tolerance) &&
                 std::isfinite(forward_absolute_tolerance) &&
                 forward_relative_tolerance > 0.0 && forward_absolute_tolerance >= 0.0,
                 "Invalid Navier-Stokes supplied-state tolerances");
}

namespace
{

real_t GlobalNorm(MPI_Comm communicator, const Vector &vector)
{ return std::sqrt(InnerProduct(communicator, vector, vector)); }

bool SameVector(MPI_Comm communicator, const Vector &first, const Vector &second)
{
   int same = first.Size() == second.Size();
   if (same)
   { for (int entry = 0; entry < first.Size(); ++entry) { same = same && first(entry) == second(entry); } }
   int global = 0;
   MPI_Allreduce(&same, &global, 1, MPI_INT, MPI_MIN, communicator);
   return global != 0;
}

FlowSolverOptions AdjointAssemblyOptions(const FlowProblem &problem,
   const FlowSolverOptions &forward, const AdjointOptions &adjoint)
{
   adjoint.Validate(problem.GetComm());
   auto options = forward;
   options.linear = adjoint.linear;
   options.Validate(problem.GetComm());
   return options;
}

int ObjectiveOrder(const FiniteElement &element, ElementTransformation &transformation)
{ return 2 * element.GetOrder() + transformation.OrderW() + 4; }

template <typename Base> class ObjectiveLoad : public Base
{
public:
   template <typename CoefficientType> explicit ObjectiveLoad(CoefficientType &coefficient) : Base(coefficient) { }
   void AssembleRHSElementVect(const FiniteElement &element, ElementTransformation &transformation,
                               Vector &vector) override
   {
      this->SetIntRule(&IntRules.Get(element.GetGeomType(), ObjectiveOrder(element, transformation)));
      Base::AssembleRHSElementVect(element, transformation, vector);
   }
};

void PressureIntegrals(const FlowState &state, int positive, int negative, real_t *global)
{
   auto &space = *state.pressure.ParFESpace();
   detail::CheckReplicated(space.GetComm(), {real_t(positive), real_t(negative)});
   real_t local[4] = {};
   for (int element = 0; element < space.GetNE(); ++element)
   {
      auto &transformation = *space.GetElementTransformation(element);
      const int slot = transformation.Attribute == positive ? 0 : transformation.Attribute == negative ? 1 : -1;
      if (slot < 0) { continue; }
      const auto &finite_element = *space.GetFE(element);
      const auto &rule = IntRules.Get(finite_element.GetGeomType(), ObjectiveOrder(finite_element, transformation));
      for (int point = 0; point < rule.GetNPoints(); ++point)
      {
         const auto &ip = rule.IntPoint(point); transformation.SetIntPoint(&ip);
         const real_t weight = ip.weight * transformation.Weight();
         local[slot] += weight;
         local[slot + 2] += weight * state.pressure.GetValue(element, ip);
      }
   }
   MPI_Allreduce(local, global, 4, MPI_DOUBLE, MPI_SUM, space.GetComm());
   detail::Check(space.GetComm(), positive != negative && global[0] > 0.0 && global[1] > 0.0,
                 "Pressure difference objective requires two distinct nonempty domain attributes");
}

class PressureWeight : public Coefficient
{
   int positive, negative;
   real_t positive_volume, negative_volume;
public:
   PressureWeight(int positive_, int negative_, real_t positive_volume_, real_t negative_volume_)
      : positive(positive_), negative(negative_), positive_volume(positive_volume_), negative_volume(negative_volume_) { }
   real_t Eval(ElementTransformation &transformation, const IntegrationPoint &) override
   { return transformation.Attribute == positive ? 1.0 / positive_volume :
            transformation.Attribute == negative ? -1.0 / negative_volume : 0.0; }
};

}

real_t KineticEnergyObjective::Evaluate(const FlowState &state) const
{
   auto &space = *state.velocity.ParFESpace();
   real_t local = 0.0, global = 0.0;
   Vector velocity;
   for (int element = 0; element < space.GetNE(); ++element)
   {
      auto &transformation = *space.GetElementTransformation(element);
      const auto &finite_element = *space.GetFE(element);
      const auto &rule = IntRules.Get(finite_element.GetGeomType(), ObjectiveOrder(finite_element, transformation));
      for (int point = 0; point < rule.GetNPoints(); ++point)
      {
         const auto &ip = rule.IntPoint(point); transformation.SetIntPoint(&ip);
         state.velocity.GetVectorValue(element, ip, velocity);
         local += 0.5 * ip.weight * transformation.Weight() * (velocity * velocity);
      }
   }
   MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, space.GetComm());
   return global;
}

void KineticEnergyObjective::AssembleGradient(const FlowState &state, const TrueDofLayout &layout,
                                             Vector &gradient) const
{
   Vector packed; state.Pack(layout, packed);
   VectorGridFunctionCoefficient coefficient(&state.velocity);
   ParLinearForm form(&layout.GetProblem().GetVelocitySpace());
   form.AddDomainIntegrator(new ObjectiveLoad<VectorDomainLFIntegrator>(coefficient));
   form.Assemble();
   gradient.SetSize(layout.LocalSize()); gradient = 0.0;
   Vector velocity(gradient.GetData(), layout.GetOffsets()[1]);
   form.ParallelAssemble(velocity);
}

MeanPressureDifferenceObjective::MeanPressureDifferenceObjective(int positive, int negative)
   : positive_attribute(positive), negative_attribute(negative) { }

real_t MeanPressureDifferenceObjective::Evaluate(const FlowState &state) const
{
   real_t integrals[4]; PressureIntegrals(state, positive_attribute, negative_attribute, integrals);
   return integrals[2] / integrals[0] - integrals[3] / integrals[1];
}

void MeanPressureDifferenceObjective::AssembleGradient(const FlowState &state, const TrueDofLayout &layout,
                                                       Vector &gradient) const
{
   Vector packed; state.Pack(layout, packed);
   real_t integrals[4]; PressureIntegrals(state, positive_attribute, negative_attribute, integrals);
   PressureWeight weight(positive_attribute, negative_attribute, integrals[0], integrals[1]);
   ParLinearForm form(&layout.GetProblem().GetPressureSpace());
   form.AddDomainIntegrator(new ObjectiveLoad<DomainLFIntegrator>(weight));
   form.Assemble();
   gradient.SetSize(layout.LocalSize()); gradient = 0.0;
   const auto &offsets = layout.GetOffsets();
   Vector pressure(gradient.GetData() + offsets[1], offsets[2] - offsets[1]);
   form.ParallelAssemble(pressure);
}

real_t ParameterDerivative::EvaluateObjectiveDerivative(const FlowState &) const { return 0.0; }

class FlowAdjointSolver::Impl
{
public:
   const FlowProblem &problem;
   AdjointOptions options;
   FlowSolverOptions assembly_options;
   FlowSolver assembler;
   std::unique_ptr<TrueDofLayout> layout;
   Vector packed, solution;
   ParGridFunction velocity, pressure;
   const ParGridFunction *source_velocity = nullptr, *source_pressure = nullptr;
   std::optional<real_t> gauge;
   std::unique_ptr<HypreParMatrix> matrix, proxy;
   Array2D<const HypreParMatrix *> blocks;
   std::vector<std::unique_ptr<HypreParMatrix>> owned_blocks;
#ifdef MFEM_USE_MUMPS
   std::unique_ptr<MUMPSSolver> direct;
#endif
   std::unique_ptr<detail::BlockFGMRESSolver> iterative;
   AdjointSolveReport report;
   bool sensitivity_ready = false;

   Impl(const FlowProblem &problem_, const FlowSolverOptions &forward,
        const AdjointOptions &options_)
      : problem(problem_), options(options_), assembly_options(AdjointAssemblyOptions(problem, forward, options)),
        assembler(problem, assembly_options), velocity(&problem.GetVelocitySpace()), pressure(&problem.GetPressureSpace()) { }

   FlowState State() const { return {velocity, pressure, gauge}; }

   void CheckSource() const
   {
      detail::Check(problem.GetComm(), bool(layout), "Prepare the Navier-Stokes adjoint before layout access");
      layout->CheckCurrent();
      Vector current;
      FlowState{*source_velocity, *source_pressure, gauge}.Pack(*layout, current);
      detail::Check(problem.GetComm(), SameVector(problem.GetComm(), current, packed),
                    "Navier-Stokes adjoint supplied state changed; prepare and solve again");
   }

   void CheckSensitivity() const
   {
      CheckSource();
      detail::Check(problem.GetComm(), sensitivity_ready,
                    "Navier-Stokes sensitivity requires a converged adjoint and valid steady state");
   }

   void Prepare(const FlowState &state)
   {
      const double started = MPI_Wtime();
      problem.Validate();
      if (!layout || layout->GetRevision() != problem.GetRevision()) { assembler.Assemble(); }
      TrueDofLayout current(problem);
      Vector values; state.Pack(current, values);
      const bool reused = layout && layout->GetRevision() == current.GetRevision() &&
                          SameVector(problem.GetComm(), values, packed);
      sensitivity_ready = false;
      report = AdjointSolveReport{};
      report.linearization_reused = reused;
      report.problem_revision = current.GetRevision();
      if (!reused)
      {
         iterative.reset();
#ifdef MFEM_USE_MUMPS
         direct.reset();
#endif
         matrix.reset(); proxy.reset(); owned_blocks.clear();
         layout = std::make_unique<TrueDofLayout>(current);
         packed = values;
         const auto &offsets = layout->GetOffsets();
         Vector velocity_values(packed.GetData(), offsets[1]);
         Vector pressure_values(packed.GetData() + offsets[1], offsets[2] - offsets[1]);
         velocity.SetFromTrueDofs(velocity_values); pressure.SetFromTrueDofs(pressure_values);
      }
      source_velocity = &state.velocity; source_pressure = &state.pressure; gauge = state.gauge_multiplier;
      Vector residual;
      assembler.AssembleResidual(State(), residual);
      report.forward_residual = GlobalNorm(problem.GetComm(), residual);
      detail::Check(problem.GetComm(), residual.CheckFinite() == 0 && std::isfinite(report.forward_residual),
                    "Nonfinite supplied Navier-Stokes residual");
      const auto &offsets = layout->GetOffsets();
      Vector momentum(residual.GetData(), offsets[1]);
      Vector continuity(residual.GetData() + offsets[1], offsets[2] - offsets[1]);
      report.momentum_residual = GlobalNorm(problem.GetComm(), momentum);
      report.continuity_residual = GlobalNorm(problem.GetComm(), continuity);
      if (layout->HasGauge())
      {
         Vector equation(residual.GetData() + offsets[2], offsets[3] - offsets[2]);
         report.gauge_residual = GlobalNorm(problem.GetComm(), equation);
      }
      ParGridFunction zero_velocity(&problem.GetVelocitySpace()), zero_pressure(&problem.GetPressureSpace());
      zero_velocity = 0.0; zero_pressure = 0.0;
      assembler.AssembleResidual({zero_velocity, zero_pressure, layout->HasGauge() ? std::optional<real_t>(0.0) : std::nullopt}, residual);
      report.forward_reference_norm = GlobalNorm(problem.GetComm(), residual);
      report.forward_tolerance = std::max(options.forward_absolute_tolerance,
                                          options.forward_relative_tolerance * report.forward_reference_norm);
      report.forward_valid = std::isfinite(report.forward_residual) && report.forward_residual <= report.forward_tolerance;
      detail::Check(problem.GetComm(), std::isfinite(report.forward_reference_norm) &&
                    (report.forward_valid || options.allow_nonconverged_state),
                    "Supplied Navier-Stokes state is not converged for the current residual");
      detail::Check(problem.GetComm(), assembly_options.gauge_enforcement != GaugeEnforcement::NullspaceProjection ||
                    (gauge && std::abs(*gauge) <= options.forward_absolute_tolerance),
                    "Projected Navier-Stokes adjoint requires a zero primal gauge multiplier");
      if (!reused) { matrix = assembler.AssembleJacobian(State()); }
      report.assembly_seconds = MPI_Wtime() - started;
   }

   AdjointSolveReport Solve(const Vector &gradient, AdjointState &adjoint)
   {
      CheckSource(); layout->ValidateVector(gradient);
      Vector rhs(gradient);
      const auto communicator = problem.GetComm();
      const real_t rhs_norm = GlobalNorm(communicator, rhs);
      detail::Check(communicator, std::isfinite(rhs_norm), "Nonfinite Navier-Stokes objective gradient norm");
      const real_t tolerance = std::max(options.linear.absolute_tolerance, options.linear.relative_tolerance * rhs_norm);
      solution.SetSize(layout->LocalSize()); solution = 0.0;
      if (assembly_options.gauge_enforcement == GaugeEnforcement::NullspaceProjection)
      {
         ConstantCoefficient one(1.0);
         ParGridFunction constant_field(&problem.GetPressureSpace()); constant_field.ProjectCoefficient(one);
         Vector constant; constant_field.GetTrueDofs(constant);
         Vector pressure_rhs(rhs.GetData() + layout->GetOffsets()[1], constant.Size());
         report.rhs_compatibility_defect = std::abs(InnerProduct(communicator, constant, pressure_rhs)) / GlobalNorm(communicator, constant);
         report.rhs_compatible = report.rhs_compatibility_defect <= tolerance;
      }
      bool converged = report.rhs_compatible;
      if (converged && std::holds_alternative<MumpsOptions>(options.linear.backend))
      {
#ifdef MFEM_USE_MUMPS
         report.setup_reused = bool(direct);
         if (!direct)
         {
            const double started = MPI_Wtime();
            direct = std::make_unique<MUMPSSolver>(communicator);
            direct->SetPrintLevel(options.linear.print_level);
            direct->SetMatrixSymType(MUMPSSolver::UNSYMMETRIC);
            direct->SetOperator(*matrix);
            report.setup_seconds = MPI_Wtime() - started;
         }
         const double started = MPI_Wtime();
         direct->MultTranspose(rhs, solution);
         report.solve_seconds = MPI_Wtime() - started;
         report.linear_iterations = 1;
#else
         detail::Check(communicator, false, "Navier-Stokes MUMPS backend is unavailable");
#endif
      }
      else if (converged)
      {
         report.setup_reused = bool(iterative);
         if (!iterative)
         {
            double started = MPI_Wtime();
            const int count = layout->HasGauge() ? 3 : 2;
            blocks.SetSize(count, count);
            for (int row = 0; row < count; ++row)
            {
               for (int column = 0; column < count; ++column)
               {
                  owned_blocks.push_back(assembler.AssembleJacobianBlock(State(), row, column));
                  blocks(row, column) = owned_blocks.back().get();
               }
            }
            proxy = assembler.AssembleVelocityProxy();
            report.assembly_seconds += MPI_Wtime() - started;
            started = MPI_Wtime();
            iterative = std::make_unique<detail::BlockFGMRESSolver>(problem, assembly_options, *layout,
               assembler.GetEssentialVelocityDofs(), packed, blocks, *proxy, true,
               [this](ParFiniteElementSpace &space) { return assembler.impl->baseline->AssembleVelocityBlock(space, true); },
               [this](ParFiniteElementSpace &space) { return assembler.impl->baseline->AssemblePressureBlock(space, true); });
            report.setup_seconds = MPI_Wtime() - started;
         }
         const auto result = iterative->Solve(rhs, solution);
         converged = result.converged;
         report.linear_iterations = result.iterations;
         report.linear_residual_restarts = result.residual_restarts;
         report.solve_seconds = result.solve_seconds;
         report.preconditioner_seconds = result.preconditioner_seconds;
      }
      Vector residual(rhs.Size()); matrix->MultTranspose(solution, residual); residual -= rhs;
      report.adjoint_residual = GlobalNorm(communicator, residual);
      int local = converged && solution.CheckFinite() == 0 && std::isfinite(report.adjoint_residual) && report.adjoint_residual <= tolerance;
      int global = 0; MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_MIN, communicator);
      report.converged = global != 0;
      report.termination = report.converged ? TerminationReason::Converged : TerminationReason::LinearFailure;
      adjoint.SetMultipliers(*layout, solution);
      report.gauge_multiplier = adjoint.GetGaugeMultiplier();
      sensitivity_ready = report.converged && report.forward_valid;
      const double local_seconds[4] = {report.assembly_seconds, report.setup_seconds, report.solve_seconds, report.preconditioner_seconds};
      double global_seconds[4]; MPI_Allreduce(local_seconds, global_seconds, 4, MPI_DOUBLE, MPI_MAX, communicator);
      report.assembly_seconds = global_seconds[0]; report.setup_seconds = global_seconds[1];
      report.solve_seconds = global_seconds[2]; report.preconditioner_seconds = global_seconds[3];
      return report;
   }
};

FlowAdjointSolver::FlowAdjointSolver(const FlowProblem &problem,
                                                               const AdjointOptions &options)
   : FlowAdjointSolver(problem, FlowSolverOptions{}, options) { }

FlowAdjointSolver::FlowAdjointSolver(const FlowProblem &problem,
   const FlowSolverOptions &forward, const AdjointOptions &options)
   : impl(std::make_unique<Impl>(problem, forward, options)) { }
FlowAdjointSolver::~FlowAdjointSolver() = default;
void FlowAdjointSolver::Prepare(const FlowState &state) { impl->Prepare(state); }
const TrueDofLayout &FlowAdjointSolver::GetLayout() const
{ impl->CheckSource(); return *impl->layout; }

AdjointSolveReport FlowAdjointSolver::Solve(const FlowState &state,
   const ObjectiveFunctional &objective, AdjointState &adjoint)
{
   impl->Prepare(state);
   impl->report.objective_value = objective.Evaluate(impl->State());
   detail::Check(impl->problem.GetComm(), std::isfinite(*impl->report.objective_value), "Nonfinite Navier-Stokes objective value");
   detail::CheckReplicated(impl->problem.GetComm(), {*impl->report.objective_value});
   Vector gradient; objective.AssembleGradient(impl->State(), *impl->layout, gradient);
   return impl->Solve(gradient, adjoint);
}

AdjointSolveReport FlowAdjointSolver::Solve(const FlowState &state,
   const Vector &objective_gradient, AdjointState &adjoint)
{ impl->Prepare(state); return impl->Solve(objective_gradient, adjoint); }

AdjointSolveReport FlowAdjointSolver::Solve(const ParGridFunction &velocity,
   const ParGridFunction &pressure, const ObjectiveFunctional &objective, AdjointState &adjoint)
{ return Solve(FlowState{velocity, pressure, std::nullopt}, objective, adjoint); }

real_t FlowAdjointSolver::ComputeDirectionalSensitivity(real_t explicit_derivative,
                                                                     const Vector &residual_direction) const
{
   impl->CheckSensitivity(); impl->layout->ValidateVector(residual_direction);
   detail::Check(impl->problem.GetComm(), std::isfinite(explicit_derivative), "Nonfinite explicit Navier-Stokes objective derivative");
   detail::CheckReplicated(impl->problem.GetComm(), {explicit_derivative});
   return explicit_derivative - InnerProduct(impl->problem.GetComm(), impl->solution, residual_direction);
}

real_t FlowAdjointSolver::ComputeDirectionalSensitivity(const ParameterDerivative &derivative) const
{
   impl->CheckSensitivity();
   Vector residual; derivative.AssembleResidualDerivative(impl->State(), *impl->layout, residual);
   const real_t explicit_derivative = derivative.EvaluateObjectiveDerivative(impl->State());
   return ComputeDirectionalSensitivity(explicit_derivative, residual);
}

void FlowAdjointSolver::ComputeParameterGradient(const Operator &jacobian,
   const Vector &explicit_gradient, Vector &total_gradient) const
{
   impl->CheckSensitivity();
   detail::Check(impl->problem.GetComm(), jacobian.Height() == impl->layout->LocalSize() &&
                 jacobian.Width() == explicit_gradient.Size() && explicit_gradient.CheckFinite() == 0,
                 "Invalid Navier-Stokes parameter derivative layout or values");
   Vector contribution(jacobian.Width()), result(explicit_gradient);
   jacobian.MultTranspose(impl->solution, contribution);
   impl->CheckSensitivity();
   detail::Check(impl->problem.GetComm(), contribution.Size() == jacobian.Width() && contribution.CheckFinite() == 0,
                 "Invalid Navier-Stokes residual parameter transpose output");
   result -= contribution; total_gradient = result;
}

void AdjointState::SetMultipliers(const TrueDofLayout &layout_, const Vector &values)
{
   layout_.ValidateVector(values);
   Vector copy(values);
   layout = std::make_unique<TrueDofLayout>(layout_);
   multipliers = copy;
   const auto &offsets = layout->GetOffsets();
   velocity.MakeRef(multipliers, 0, offsets[1]);
   pressure.MakeRef(multipliers, offsets[1], offsets[2] - offsets[1]);
}

const Vector &AdjointState::GetMultipliers() const
{
   MFEM_VERIFY(layout, "Uninitialized Navier-Stokes adjoint state");
   layout->CheckCurrent();
   return multipliers;
}

const Vector &AdjointState::GetVelocity() const { GetMultipliers(); return velocity; }
const Vector &AdjointState::GetPressure() const { GetMultipliers(); return pressure; }

std::optional<real_t> AdjointState::GetGaugeMultiplier() const
{
   GetMultipliers();
   if (!layout->HasGauge()) { return std::nullopt; }
   int rank = 0;
   MPI_Comm_rank(layout->GetComm(), &rank);
   real_t value = rank == 0 ? multipliers(layout->GetOffsets()[2]) : 0.0;
   MPI_Bcast(&value, 1, MPI_DOUBLE, 0, layout->GetComm());
   return value;
}

FlowLinearization::FlowLinearization(const FlowProblem &problem,
      const FlowState &state_, std::unique_ptr<Operator> jacobian_, const Vector &residual_)
   : Operator(0), layout(problem), residual(residual_), jacobian(std::move(jacobian_))
{
   state_.Pack(layout, state);
   layout.ValidateVector(residual);
   height = width = layout.LocalSize();
   detail::Check(layout.GetComm(), jacobian && jacobian->Height() == height && jacobian->Width() == width,
                 "Invalid Navier-Stokes linearization operator");
}

const TrueDofLayout &FlowLinearization::GetLayout() const { layout.CheckCurrent(); return layout; }
const Vector &FlowLinearization::GetState() const { layout.CheckCurrent(); return state; }
const Vector &FlowLinearization::GetResidual() const { layout.CheckCurrent(); return residual; }

void FlowLinearization::Mult(const Vector &input, Vector &output) const
{
   layout.ValidateVector(input);
   Vector copy(input);
   output.SetSize(height);
   jacobian->Mult(copy, output);
   layout.ValidateVector(output);
}

void FlowLinearization::MultTranspose(const Vector &input, Vector &output) const
{
   layout.ValidateVector(input);
   Vector copy(input);
   output.SetSize(width);
   jacobian->MultTranspose(copy, output);
   layout.ValidateVector(output);
}

}
