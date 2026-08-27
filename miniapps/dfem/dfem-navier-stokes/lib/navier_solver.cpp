// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.

#include "navier_solver.hpp"

namespace mfem
{
namespace dfem_navier
{

using namespace future;

template <int DIM>
NavierStokesOperator<DIM>::NavierStokesOperator(
   ParFiniteElementSpace &ufes, ParFiniteElementSpace &pfes,
   const IntegrationRule &ir, real_t viscosity)
   : NavierStokesOperator(ufes, pfes, ir, viscosity,
                          RheologyType::Newtonian) { }

template <int DIM>
NavierStokesOperator<DIM>::NavierStokesOperator(
   ParFiniteElementSpace &ufes, ParFiniteElementSpace &pfes,
   const IntegrationRule &ir, real_t viscosity, RheologyType rheology)
   : NavierStokesOperatorBase(ufes.GetTrueVSize() + pfes.GetTrueVSize()),
     ufes(ufes), pfes(pfes)
{
   auto &nodes = *static_cast<ParGridFunction *>(ufes.GetParMesh()->GetNodes());
   nodes_fes = nodes.ParFESpace();
   nodes.GetTrueDofs(nodes_tdofs);

   const std::vector<FieldDescriptor> inputs =
   {
      {U, &ufes}, {P, &pfes}, {Coords, nodes_fes}
   };
   const std::vector<FieldDescriptor> outputs =
   {
      {U, &ufes}, {P, &pfes}
   };
   dop = std::make_shared<DifferentiableOperator>(
            inputs, outputs, *ufes.GetParMesh());

   Array<int> domain_attributes;
   if (ufes.GetMesh()->attributes.Size() > 0)
   {
      domain_attributes.SetSize(ufes.GetMesh()->attributes.Max());
      domain_attributes = 1;
   }

   auto register_rheology = [&](auto qf)
   {
      dop->AddDomainIntegrator<LocalQFBackend>(
         qf,
         Inputs<Value<U>, Gradient<U>, Value<P>, Gradient<Coords>, Weight> {},
         Outputs<Gradient<U>, Value<U>, Value<P>> {}, ir, domain_attributes,
         Derivatives<U, P> {});
   };

   switch (rheology)
   {
      case RheologyType::Newtonian:
      {
         NewtonianNavierStokesQFunction<DIM> qf;
         qf.viscosity = viscosity;
         register_rheology(qf);
         break;
      }
      case RheologyType::PowerLaw:
      {
         RegularizedPowerLawNavierStokesQFunction<DIM> qf;
         qf.consistency = viscosity;
         qf.power_index = 0.5;  // Shear thinning behavior
         qf.regularization = 1.0e-3;
         register_rheology(qf);
         break;
      }
   }

   block_offsets.SetSize(3);
   block_offsets[0] = 0;
   block_offsets[1] = ufes.GetTrueVSize();
   block_offsets[2] = pfes.GetTrueVSize();
   block_offsets.PartialSum();
}

template <int DIM>
void NavierStokesOperator<DIM>::SetEssentialVelocityAttributes(
   const Array<int> &ess_bdr)
{
   ufes.GetEssentialTrueDofs(ess_bdr, ess_velocity_tdofs);
}

template <int DIM>
void NavierStokesOperator<DIM>::SetEssentialVelocityTrueDofs(
   const Array<int> &tdofs)
{
   ess_velocity_tdofs = tdofs;
}

template <int DIM>
const Array<int> &NavierStokesOperator<DIM>::
GetEssentialVelocityTrueDofs() const
{
   return ess_velocity_tdofs;
}

template <int DIM>
const Array<int> &NavierStokesOperator<DIM>::GetBlockOffsets() const
{
   return block_offsets;
}

template <int DIM>
void NavierStokesOperator<DIM>::Mult(const Vector &x, Vector &y) const
{
   MFEM_VERIFY(dynamic_cast<const BlockVector *>(&x),
               "NavierStokesOperator input must be a BlockVector");
   MFEM_VERIFY(dynamic_cast<BlockVector *>(&y),
               "NavierStokesOperator output must be a BlockVector");

   const auto &state = static_cast<const BlockVector &>(x);
   auto &result = static_cast<BlockVector &>(y);
   MFEM_VERIFY(state.NumBlocks() == 2 && result.NumBlocks() == 2,
               "expected velocity and pressure blocks");

   MultiVector X{state.GetBlock(0), state.GetBlock(1), nodes_tdofs};
   MultiVector Y{result.GetBlock(0), result.GetBlock(1)};
   dop->Mult(X, Y);
   result.GetBlock(0).SetSubVector(ess_velocity_tdofs, 0.0);
}

template <int DIM>
std::shared_ptr<DerivativeOperator>
NavierStokesOperator<DIM>::GetDerivative(size_t field_id,
                                         const BlockVector &state,
                                         bool use_cached_setup) const
{
   MultiVector X{state.GetBlock(0), state.GetBlock(1), nodes_tdofs};
   return dop->GetDerivative(field_id, X, use_cached_setup);
}

NavierStokesResidual::JacobianOperator::JacobianOperator(
   const HypreParMatrix &mass, HypreParMatrix &divergence,
   HypreParMatrix &pressure_gradient,
   NavierStokesOperatorBase &ns_operator,
   const BlockVector &state, real_t gamma)
   : BlockOperator(ns_operator.GetBlockOffsets())
{
   // Only Assemble() is used below, so the cached derivative-apply callbacks
   // would never be used; ask for the direct path.
   auto dRdU = ns_operator.GetDerivative(U, state, false);
   std::vector<HypreParMatrix *> velocity_blocks;
   dRdU->Assemble(velocity_blocks);
   velocity_tangent.reset(Add(1.0, mass, gamma, *velocity_blocks[U]));
   for (auto *block : velocity_blocks)
   {
      delete block;
   }
   delete velocity_tangent->EliminateRowsCols(
      ns_operator.GetEssentialVelocityTrueDofs());
   SetBlock(U, U, velocity_tangent.get());
   SetBlock(U, P, &pressure_gradient);
   SetBlock(P, U, &divergence);
}

NavierStokesResidual::NavierStokesResidual(
   NavierStokesOperatorBase &ns_operator, const HypreParMatrix &mass,
   HypreParMatrix &divergence, HypreParMatrix &pressure_gradient)
   : Operator(ns_operator.Height()), ns_operator(ns_operator), mass(mass),
     divergence(divergence), pressure_gradient(pressure_gradient),
     block_offsets(ns_operator.GetBlockOffsets()), stage_state(block_offsets),
     spatial_residual(block_offsets), mass_derivative(block_offsets[1]) { }

void NavierStokesResidual::SetParameters(real_t gamma_, const Vector *state_)
{
   MFEM_VERIFY(gamma_ > 0.0, "implicit stage coefficient should be positive");
   gamma = gamma_;
   state = state_;
}

void NavierStokesResidual::Mult(const Vector &stage_derivative,
                                Vector &residual) const
{
   MFEM_ASSERT(state, "call SetParameters() before the Newton solve");
   MFEM_VERIFY(stage_derivative.Size() == Height() &&
               residual.Size() == Height(),
               "invalid mixed stage vector size");
   BlockVector derivative_blocks(
      const_cast<Vector &>(stage_derivative), block_offsets);
   BlockVector residual_blocks(residual, block_offsets);
   const Vector &velocity_derivative = derivative_blocks.GetBlock(U);
   add(*state, gamma, velocity_derivative, stage_state.GetBlock(U));
   stage_state.GetBlock(P) = derivative_blocks.GetBlock(P);
   ns_operator.Mult(stage_state, spatial_residual);
   mass.Mult(velocity_derivative, mass_derivative);
   residual_blocks.GetBlock(U) = spatial_residual.GetBlock(U);
   residual_blocks.GetBlock(U) += mass_derivative;
   residual_blocks.GetBlock(P) = spatial_residual.GetBlock(P);
   residual_blocks.GetBlock(P) /= gamma;
}

Operator &NavierStokesResidual::GetGradient(
   const Vector &stage_derivative) const
{
   MFEM_ASSERT(state, "call SetParameters() before the Newton solve");
   BlockVector derivative_blocks(
      const_cast<Vector &>(stage_derivative), block_offsets);
   add(*state, gamma, derivative_blocks.GetBlock(U), stage_state.GetBlock(U));
   stage_state.GetBlock(P) = derivative_blocks.GetBlock(P);
   jacobian = std::make_unique<JacobianOperator>(
                 mass, divergence, pressure_gradient, ns_operator,
                 stage_state, gamma);
   return *jacobian;
}

NavierStokesEvolution::NavierStokesEvolution(
   ParFiniteElementSpace &ufes, ParFiniteElementSpace &pfes,
   NavierStokesOperatorBase &ns_operator,
   const BlockVector &initial_state)
   : TimeDependentOperator(ufes.GetTrueVSize()), ns_operator(ns_operator),
     comm(ufes.GetComm()), block_offsets(ns_operator.GetBlockOffsets()),
     state(block_offsets), residual(block_offsets), rhs(block_offsets),
     solution(block_offsets), pressure(block_offsets[2] - block_offsets[1]),
     system(block_offsets), explicit_solver(ufes.GetComm()),
     implicit_solver(ufes.GetComm()), newton_solver(ufes.GetComm()),
     explicit_preconditioner(ufes.GetComm(), block_offsets),
     implicit_preconditioner(ufes.GetComm(), block_offsets)
{
   pressure = 0.0;

   ParLinearForm ones_form(&pfes);
   ConstantCoefficient one(1.0);
   ones_form.AddDomainIntegrator(new DomainLFIntegrator(one));
   ones_form.Assemble();
   {
      std::unique_ptr<HypreParVector> ones_true(ones_form.ParallelAssemble());
      pressure_ones = *ones_true;
   }
   domain_volume = pressure_ones.Sum();
   MPI_Allreduce(MPI_IN_PLACE, &domain_volume, 1,
                 MPITypeMap<real_t>::mpi_type, MPI_SUM, comm);

   ParBilinearForm mass_form(&ufes);
   mass_form.AddDomainIntegrator(new VectorMassIntegrator);
   mass_form.Assemble();
   mass_form.Finalize();
   mass.reset(mass_form.ParallelAssemble());

   auto dRdU = ns_operator.GetDerivative(U, initial_state);
   auto dRdP = ns_operator.GetDerivative(P, initial_state);
   std::vector<HypreParMatrix *> velocity_blocks;
   std::vector<HypreParMatrix *> pressure_blocks;
   dRdU->Assemble(velocity_blocks);
   dRdP->Assemble(pressure_blocks);
   divergence.reset(velocity_blocks[P]);
   pressure_gradient.reset(pressure_blocks[U]);
   velocity_blocks[P] = nullptr;
   pressure_blocks[U] = nullptr;
   for (auto *block : velocity_blocks) { delete block; }
   for (auto *block : pressure_blocks) { delete block; }

   const Array<int> &ess_tdofs = ns_operator.GetEssentialVelocityTrueDofs();
   delete mass->EliminateRowsCols(ess_tdofs);
   delete divergence->EliminateCols(ess_tdofs);
   pressure_gradient->EliminateRows(ess_tdofs);

   system.SetBlock(U, U, mass.get());
   system.SetBlock(U, P, pressure_gradient.get());
   system.SetBlock(P, U, divergence.get());

#ifdef MFEM_USE_SINGLE
   const real_t linear_abs_tol = 1e-8;
#else
   const real_t linear_abs_tol = 1e-12;
#endif

   explicit_solver.iterative_mode = false;
   explicit_solver.SetRelTol(1.0e-10);
   explicit_solver.SetAbsTol(linear_abs_tol);
   explicit_solver.SetMaxIter(500);
   explicit_solver.SetPrintLevel(0);
   explicit_solver.SetPreconditioner(explicit_preconditioner);
   explicit_solver.SetOperator(system);

   implicit_solver.iterative_mode = false;
   implicit_solver.SetRelTol(1e-10);
   implicit_solver.SetAbsTol(linear_abs_tol);
   implicit_solver.SetMaxIter(500);
   implicit_solver.SetPrintLevel(0);
   implicit_solver.SetPreconditioner(implicit_preconditioner);

   stage_residual = std::make_unique<NavierStokesResidual>(
                       ns_operator, *mass, *divergence, *pressure_gradient);
   newton_solver.iterative_mode = true;
   newton_solver.SetOperator(*stage_residual);
   newton_solver.SetSolver(implicit_solver);
#ifdef MFEM_USE_SINGLE
   newton_solver.SetRelTol(1e-5);
   newton_solver.SetAbsTol(1e-8);
#else
   newton_solver.SetRelTol(1e-8);
   newton_solver.SetAbsTol(1e-12);
#endif
   newton_solver.SetMaxIter(25);
   newton_solver.SetPrintLevel(0);
}

NavierStokesEvolution::~NavierStokesEvolution() = default;

void NavierStokesEvolution::Mult(const Vector &u, Vector &du_dt) const
{
   state.GetBlock(U) = u;
   state.GetBlock(P) = 0.0;
   ns_operator.Mult(state, residual);
   rhs = 0.0;
   rhs.GetBlock(U) = residual.GetBlock(U);
   rhs.GetBlock(U) *= -1.0;
   solution = 0.0;
   explicit_solver.Mult(rhs, solution);
   MFEM_VERIFY(explicit_solver.GetConverged(),
               "Navier-Stokes saddle-point solve did not converge");
   du_dt = solution.GetBlock(U);
   MeanZero(solution.GetBlock(P));
   pressure = solution.GetBlock(P);
}

void NavierStokesEvolution::ImplicitSolve(const real_t gamma,
                                          const Vector &u,
                                          Vector &du_dt)
{
   stage_residual->SetParameters(gamma, &u);
   solution.GetBlock(U) = 0.0;
   solution.GetBlock(P) = pressure;
   Vector zero_rhs;
   newton_solver.Mult(zero_rhs, solution);
   MFEM_VERIFY(newton_solver.GetConverged(),
               "Navier-Stokes implicit solve did not converge");
   du_dt = solution.GetBlock(U);
   MeanZero(solution.GetBlock(P));
   pressure = solution.GetBlock(P);
}

void NavierStokesEvolution::RecoverPressure(const Vector &u) const
{
   Vector du_dt(u.Size());
   Mult(u, du_dt);
}

void NavierStokesEvolution::MeanZero(Vector &p) const
{
   const real_t integral = InnerProduct(comm, pressure_ones, p);
   p -= integral / domain_volume;
}

void NavierStokesEvolution::ProjectDivergenceFree(Vector &u) const
{
   mass->Mult(u, rhs.GetBlock(U));
   rhs.GetBlock(P) = 0.0;
   solution = 0.0;
   explicit_solver.Mult(rhs, solution);
   MFEM_VERIFY(explicit_solver.GetConverged(),
               "divergence-free projection did not converge");
   u = solution.GetBlock(U);
}

NavierStokesSolver::NavierStokesSolver(
   std::unique_ptr<ODESolver> ode_solver_,
   NavierStokesEvolution &evolution_)
   : ode_solver(std::move(ode_solver_)), evolution(evolution_)
{
   MFEM_VERIFY(ode_solver, "NavierStokesSolver requires an ODE solver");
   ode_solver->Init(evolution);
}

void NavierStokesSolver::Step(BlockVector &state, real_t &t, real_t &dt)
{
   MFEM_VERIFY(state.NumBlocks() == 2,
               "NavierStokesSolver expects velocity and pressure blocks");
   MFEM_VERIFY(state.GetBlock(U).Size() == evolution.Width(),
               "invalid velocity block size");
   MFEM_VERIFY(state.GetBlock(P).Size() == evolution.GetPressure().Size(),
               "invalid pressure block size");

   ode_solver->Step(state.GetBlock(U), t, dt);
   evolution.RecoverPressure(state.GetBlock(U));
   state.GetBlock(P) = evolution.GetPressure();
}

/// Template instantiations for 2D and 3D Navier-Stokes operators
template class NavierStokesOperator<2>;
template class NavierStokesOperator<3>;

} // namespace dfem_navier
} // namespace mfem
