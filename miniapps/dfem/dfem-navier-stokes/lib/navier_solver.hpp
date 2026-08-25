// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.

#ifndef MFEM_DFEM_NAVIER_SOLVER_HPP
#define MFEM_DFEM_NAVIER_SOLVER_HPP

#include "mfem.hpp"
#include "../../../../fem/dfem/doperator.hpp"
#include "navier_preconditioners.hpp"
#include "navier_qfunctions.hpp"

#include <memory>


/// TODO: The structure is fine for now, but we will need a refactor to make 
/// the NavierStokesSolver the only user-facing class in the API. This means that 
/// all the other classes (NavierStokesOperator, NavierStokesResidual, NavierStokesEvolution) should be incorporater in it.
/// Then we can simplify the example codes to not have to deal with instantiation of the separate components.


namespace mfem
{
namespace dfem_navier
{

/// Wrapper for the DifferentiableOperator that implements the incompressible Navier-Stokes equations.
/// This is the driver of the differentiable path, and is used in some of the other classes below to
/// implement the residual, the time-dependent operator, and the ODE solver.
template <int DIM>
class NavierStokesOperator : public Operator
{
public:
   NavierStokesOperator(ParFiniteElementSpace &ufes,
                        ParFiniteElementSpace &pfes,
                        const IntegrationRule &ir,
                        real_t viscosity);

   void SetEssentialVelocityAttributes(const Array<int> &ess_bdr);
   void SetEssentialVelocityTrueDofs(const Array<int> &tdofs);
   const Array<int> &GetEssentialVelocityTrueDofs() const;
   const Array<int> &GetBlockOffsets() const;
   void Mult(const Vector &x, Vector &y) const override;

   std::shared_ptr<future::DerivativeOperator>
   GetDerivative(size_t field_id, const BlockVector &state,
                 bool use_cached_setup = true) const;

private:
   ParFiniteElementSpace &ufes;
   ParFiniteElementSpace &pfes;
   ParFiniteElementSpace *nodes_fes = nullptr;
   Vector nodes_tdofs;
   Array<int> ess_velocity_tdofs;
   Array<int> block_offsets;
   std::shared_ptr<future::DifferentiableOperator> dop;
};

/// Operator for the Residual form of the incompressible Navier-Stokes equations.
/// This is required for the implicit solve in the ODE solver, and is the operator
/// passed to the NewtonSolver (needs Mult() and GetGradient()).
class NavierStokesResidual : public Operator
{
private:
   class JacobianOperator : public BlockOperator
   {
   public:
      JacobianOperator(const HypreParMatrix &mass,
                       HypreParMatrix &divergence,
                       HypreParMatrix &pressure_gradient,
                       NavierStokesOperator<dim> &ns_operator,
                       const BlockVector &state, real_t gamma);

   private:
      std::unique_ptr<HypreParMatrix> velocity_tangent;
   };

public:
   NavierStokesResidual(NavierStokesOperator<dim> &ns_operator,
                        const HypreParMatrix &mass,
                        HypreParMatrix &divergence,
                        HypreParMatrix &pressure_gradient);

   void SetParameters(real_t gamma, const Vector *state);
   void Mult(const Vector &stage_derivative, Vector &residual) const override;
   Operator &GetGradient(const Vector &stage_derivative) const override;

private:
   NavierStokesOperator<dim> &ns_operator;
   const HypreParMatrix &mass;
   HypreParMatrix &divergence;
   HypreParMatrix &pressure_gradient;
   Array<int> block_offsets;
   real_t gamma = 0.0;
   const Vector *state = nullptr;
   mutable BlockVector stage_state;
   mutable BlockVector spatial_residual;
   mutable Vector mass_derivative;
   mutable std::unique_ptr<JacobianOperator> jacobian;
};

/// Time dependent operator for the incompressible Navier-Stokes equations.
/// This is the TDO passed to the ODE solver, and it is responsible for
/// advancing the velocity and recovering the algebraic pressure at the new time.
class NavierStokesEvolution : public TimeDependentOperator
{
public:
   NavierStokesEvolution(ParFiniteElementSpace &ufes,
                         ParFiniteElementSpace &pfes,
                         NavierStokesOperator<dim> &ns_operator,
                         const BlockVector &initial_state);
   ~NavierStokesEvolution() override;

   void Mult(const Vector &u, Vector &du_dt) const override;
   void ImplicitSolve(real_t gamma, const Vector &u,
                      Vector &du_dt) override;
   void RecoverPressure(const Vector &u) const;
   void MeanZero(Vector &p) const;
   void ProjectDivergenceFree(Vector &u) const;
   const Vector &GetPressure() const { return pressure; }

private:
   NavierStokesOperator<dim> &ns_operator;
   MPI_Comm comm;
   Array<int> block_offsets;
   mutable BlockVector state;
   mutable BlockVector residual;
   mutable BlockVector rhs;
   mutable BlockVector solution;
   mutable Vector pressure;
   Vector pressure_ones;
   real_t domain_volume = 0.0;
   std::unique_ptr<HypreParMatrix> mass;
   std::unique_ptr<HypreParMatrix> divergence;
   std::unique_ptr<HypreParMatrix> pressure_gradient;
   BlockOperator system;
   mutable FGMRESSolver explicit_solver;
   FGMRESSolver implicit_solver;
   NewtonSolver newton_solver;
   BlockDiagonalPreconditioner explicit_preconditioner;
   BlockDiagonalPreconditioner implicit_preconditioner;
   std::unique_ptr<NavierStokesResidual> stage_residual;
};

/// Simple wrapper for an ODE solver that advances the velocity and recovers the algebraic pressure at the new time.
/// The ODESolver advances only the velocity, and the pressure would be the cached one from the last
/// Mult() or ImplicitSolve() call. 
/// This class avoids the user from having to manually call RecoverPressure() after each ODE step.
class NavierStokesSolver
{
public:
   NavierStokesSolver(std::unique_ptr<ODESolver> ode_solver,
                      NavierStokesEvolution &evolution);

   /// Advance velocity and recover the algebraic pressure at the new time.
   /// The input and output blocks are [velocity, pressure].
   void Step(BlockVector &state, real_t &t, real_t &dt);

private:
   std::unique_ptr<ODESolver> ode_solver;
   NavierStokesEvolution &evolution;
};

extern template class NavierStokesOperator<dim>;

} // namespace dfem_navier
} // namespace mfem

#endif // MFEM_DFEM_NAVIER_SOLVER_HPP