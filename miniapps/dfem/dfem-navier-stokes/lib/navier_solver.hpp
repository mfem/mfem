// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.

#ifndef MFEM_DFEM_NAVIER_SOLVER_HPP
#define MFEM_DFEM_NAVIER_SOLVER_HPP

#include "mfem.hpp"
#include "../../../../fem/dfem/doperator.hpp"
#include "navier_preconditioners.hpp"
#include "navier_qfunctions.hpp"

#include <functional>
#include <memory>


/// TODO: The structure is fine for now, but we will need a refactor to make
/// the NavierStokesSolver the only user-facing class in the API. This means that
/// all the other classes (NavierStokesOperator, NavierStokesResidual, NavierStokesEvolution) should be incorporater in it.
/// Then we can simplify the example codes to not have to deal with instantiation of the separate components.


namespace mfem
{
namespace dfem_navier
{

/// Field ids of the differentiable operator. U and P double as the block
/// indices of the [velocity, pressure] state BlockVector, so the miniapps use
/// them too; Coords is the mesh nodes field and is internal to the residual.
constexpr int U = 0;
constexpr int P = 1;
constexpr int Coords = 2;

/// Rheology model the operator differentiates; the models themselves are the
/// q-functions in navier_qfunctions.hpp.
enum class RheologyType
{
   Newtonian,
   PowerLaw,
   Bingham
};

//-----------------------------------------------------------------------------------
//    Utilities
//
// Evaluate shear rate and effective viscosity
// Just used for post-processing and visualization, and are consistent with the
// used rheology model.
//-----------------------------------------------------------------------------------

/// Shear rate gamma = sqrt(2 D:D) of a velocity field, with D = sym(grad(u)).
class ShearRateCoefficient : public Coefficient
{
public:
   ShearRateCoefficient(const ParGridFunction &velocity) : velocity(velocity) {}

   real_t Eval(ElementTransformation &T, const IntegrationPoint &ip) override;

private:
   const ParGridFunction &velocity;
   DenseMatrix strain_rate;
};


/// Evaluates the effective viscosity of a velocity field. It forms the strain
/// rate D = sym(grad(u)), which is pure kinematics, and hands it to the
/// viscosity law encoded in the selected Rheology model.
class ViscosityCoefficient : public Coefficient
{
public:
   ViscosityCoefficient(const ParGridFunction &velocity,
                        std::function<real_t(const DenseMatrix &)>
                        viscosity_function)
      : velocity(velocity),
        viscosity_function(std::move(viscosity_function)) { }

   real_t Eval(ElementTransformation &T, const IntegrationPoint &ip) override
   {
      T.SetIntPoint(&ip);
      velocity.GetVectorGradient(T, strain_rate);
      strain_rate.Symmetrize();  // In place: D = (grad(u) + grad(u)^T) / 2
      return viscosity_function(strain_rate);
   }

private:
   const ParGridFunction &velocity;
   std::function<real_t(const DenseMatrix &)> viscosity_function;
   DenseMatrix strain_rate;
};


//-----------------------------------------------------------------------------------
//    Navier-Stokes solver classes
//
// Structure and organization:
//
// NavierStokesOperator: wrapper for the DifferentiableOperator
// NavierStokesEvolution: the time-dependent operator used in the ODE solver
// NavierStokesResidual: residual form of NS solver; needed for ImplicitSolve
//                       (passed to NewtonSolver, implements Mult() and GetGradient())
// NavierStokesSolver: simple wrapper for the ODESolver, which additionally recovers
//                     the algebraic pressure at each time step (without needing the user
//                     to do it manually).
//-----------------------------------------------------------------------------------

/// Dimension-independent interface used by the solver infrastructure.
/// This is needed for type-erasure of the NavierStokesOperator, since the
/// other classes (NavierStokesResidual, NavierStokesEvolution) are effectively
/// dimension-independent (but have a reference to the NavierStokesOperator)
/// TODO: this might change if we refactor the classes as above
class NavierStokesOperatorBase : public Operator
{
public:
   using Operator::Operator;

   virtual const Array<int> &GetEssentialVelocityTrueDofs() const = 0;
   virtual const Array<int> &GetBlockOffsets() const = 0;
   virtual std::shared_ptr<future::DerivativeOperator>
   GetDerivative(size_t field_id, const BlockVector &state,
                 bool use_cached_setup = false) const = 0;
};

/// Wrapper for the DifferentiableOperator that implements the incompressible Navier-Stokes equations.
/// This is the driver of the differentiable path, and is used in some of the other classes below to
/// implement the residual, the time-dependent operator, and the ODE solver.
template <int DIM>
class NavierStokesOperator : public NavierStokesOperatorBase
{
public:
   NavierStokesOperator(ParFiniteElementSpace &ufes,
                        ParFiniteElementSpace &pfes,
                        const IntegrationRule &ir,
                        real_t viscosity);
   NavierStokesOperator(ParFiniteElementSpace &ufes,
                        ParFiniteElementSpace &pfes,
                        const IntegrationRule &ir,
                        real_t viscosity,
                        RheologyType rheology);

   void SetEssentialVelocityAttributes(const Array<int> &ess_bdr);
   void SetEssentialVelocityTrueDofs(const Array<int> &tdofs);
   const Array<int> &GetEssentialVelocityTrueDofs() const override;
   const Array<int> &GetBlockOffsets() const override;
   void Mult(const Vector &x, Vector &y) const override;

   /// Effective viscosity mu of the active rheology
   Coefficient &GetViscosity(const ParGridFunction &velocity) const;

   std::shared_ptr<future::DerivativeOperator>
   GetDerivative(size_t field_id, const BlockVector &state,
                 bool use_cached_setup = false) const override;
   // NOTE: in this case the PA cached version is inefficient
   // cuz we have to perform the setup on each GetDerivative()
   // i.e. every time we call GetGradient() in the NavierStokesResidual
   // might be useful if we have a Frozen Jacobian option for Newton.

private:
   ParFiniteElementSpace &ufes;
   ParFiniteElementSpace &pfes;
   ParFiniteElementSpace *nodes_fes = nullptr;
   Vector nodes_tdofs;
   Array<int> ess_velocity_tdofs;
   Array<int> block_offsets;
   std::shared_ptr<future::DifferentiableOperator> dop;

   /// mu(D) of the rheology selected at construction, captured next to the
   /// q-function it was configured with so the two cannot disagree.
   std::function<real_t(const DenseMatrix &)> viscosity_law;
   mutable std::unique_ptr<Coefficient> viscosity_coefficient;
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
                       NavierStokesOperatorBase &ns_operator,
                       const BlockVector &state, real_t gamma);

   private:
      std::unique_ptr<HypreParMatrix> velocity_tangent;
   };

public:
   NavierStokesResidual(NavierStokesOperatorBase &ns_operator,
                        const HypreParMatrix &mass,
                        HypreParMatrix &divergence,
                        HypreParMatrix &pressure_gradient);

   void SetParameters(real_t gamma, const Vector *state);
   void Mult(const Vector &stage_derivative, Vector &residual) const override;
   Operator &GetGradient(const Vector &stage_derivative) const override;

private:
   NavierStokesOperatorBase &ns_operator;
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
                         NavierStokesOperatorBase &ns_operator,
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
   NavierStokesOperatorBase &ns_operator;
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

// ---------------------------------------------------------------------------
// Templated NavierStokesOperator<DIM> implementation.
//
// NOTE: Moved definitions in the header to split navier_solver in two TUs
//       for 2D and 3D, so they can compile in parallel.
// ---------------------------------------------------------------------------

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
      // Add the q-function specified by selected Rheology to the differentiable operator
      dop->AddDomainIntegrator<LocalQFBackend>(
         qf,
         Inputs<Value<U>, Gradient<U>, Value<P>, Gradient<Coords>, Weight> {},
         Outputs<Gradient<U>, Value<U>, Value<P>> {}, ir, domain_attributes,
         Derivatives<U, P> {});

      // Capture the same q-function for post-processing, so the exported
      // viscosity is by construction the one the solve differentiates.
      viscosity_law = [qf](const DenseMatrix &strain_rate)
      {
         tensor<real_t, DIM, DIM> D;
         for (int i = 0; i < DIM; i++)
         {
            for (int j = 0; j < DIM; j++)
            {
               D(i, j) = strain_rate(i, j);
            }
         }
         return qf.effective_viscosity(D);
      };
   };

   switch (rheology)
   {
      case RheologyType::Newtonian:
      {
         NewtonianRheology<DIM> rheology;
         rheology.viscosity = viscosity;
         NavierStokesQFunction<NewtonianRheology<DIM>, DIM> qf{rheology};
         register_rheology(qf);
         break;
      }
      case RheologyType::PowerLaw:
      {
         RegularizedPowerLawRheology<DIM> rheology;
         rheology.consistency = viscosity;
         rheology.power_index = 0.5;  // Shear thinning behavior
         rheology.regularization = 1.0e-3;
         NavierStokesQFunction<RegularizedPowerLawRheology<DIM>, DIM> qf
         {
            rheology
         };
         register_rheology(qf);
         break;
      }
      case RheologyType::Bingham:
      {
         RegularizedBinghamRheology<DIM> rheology;
         rheology.mu_p = viscosity;
         constexpr real_t bingham_number = 2.0;
         rheology.yield_stress = bingham_number * viscosity;
         rheology.tau_regularization = 1.0e1;
         rheology.regularization = 1.0e-3;
         NavierStokesQFunction<RegularizedBinghamRheology<DIM>, DIM> qf
         {
            rheology
         };
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
Coefficient &NavierStokesOperator<DIM>::GetViscosity(
   const ParGridFunction &velocity) const
{
   MFEM_VERIFY(viscosity_law, "no rheology was registered");
   if (!viscosity_coefficient)
   {
      viscosity_coefficient =
         std::make_unique<ViscosityCoefficient>(velocity, viscosity_law);
   }
   return *viscosity_coefficient;
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

/// Instantiated by navier_solver_2d.cpp and navier_solver_3d.cpp.
extern template class NavierStokesOperator<2>;
extern template class NavierStokesOperator<3>;

} // namespace dfem_navier
} // namespace mfem

#endif // MFEM_DFEM_NAVIER_SOLVER_HPP