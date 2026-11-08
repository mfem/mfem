#ifndef MFEM_STOKES_PRECONDITIONERS_HPP
#define MFEM_STOKES_PRECONDITIONERS_HPP

#include "mfem.hpp"
#include <functional>
#include <memory>
#include <vector>

namespace mfem::stokes::detail
{

/** @brief Fixed two-level inverse with degree-one coarse AMG and l1-Jacobi.

    The fine SPD matrix and space are borrowed. Boundary markers are supplied
    per vector component, or empty for an unconstrained scalar pressure block.
    Both fine rows and coarse essential columns of the parallel true-dof
    transfer are eliminated. Each Mult starts from zero and applies the same
    number of pre/post smoothing sweeps. Coarse spaces and operators are owned.
    Elasticity interpolation is configured on the coarse velocity space only.
    An optional assembler builds the unconstrained coarse matrix directly on
    the owned P1/Q1 space. It is called only during construction and is not
    retained. Without it, the coarse matrix is formed by Galerkin projection.
    Both paths impose identical homogeneous componentwise coarse constraints.
    Construction explicitly sets up AMG; Mult does not defer hierarchy setup. */
class P1AMGPreconditioner : public Solver
{
   const HypreParMatrix &matrix;
   H1_FECollection coarse_collection;
   ParFiniteElementSpace coarse_space;
   PRefinementTransferOperator transfer;
   HypreParMatrix *prolongation;
   std::unique_ptr<HypreParMatrix> coarse_matrix;
   std::unique_ptr<HypreBoomerAMG> coarse_amg;
   HypreSmoother smoother;
   mutable Vector residual, correction, coarse_rhs, coarse_solution;

public:
   using CoarseAssembler = std::function<std::unique_ptr<HypreParMatrix>(ParFiniteElementSpace &)>;

   P1AMGPreconditioner(const HypreParMatrix &matrix,
                      ParFiniteElementSpace &space,
                      const std::vector<Array<int>> &boundary_markers,
                      int smoothing_steps, bool elasticity, bool refine,
                      const CoarseAssembler &assemble = {});
   /// Borrow the constrained coarse operator for diagnostics and verification.
   const HypreParMatrix &GetCoarseMatrix() const { return *coarse_matrix; }
   void SetOperator(const Operator &) override;
   void Mult(const Vector &input, Vector &output) const override;
};

/** @brief Back substitution with borrowed velocity/pressure diagonal inverses.

    Uses the actual constrained upper-right Stokes block. If a gauge is active,
    its diagonal inverse must include the negative Schur sign; its column is
    applied collectively even on ranks with no locally owned multiplier. */
class UpperTriangularPreconditioner : public Solver
{
   Array<int> offsets;
   Solver &velocity_inverse, &pressure_inverse;
   const Operator &momentum_pressure;
   Solver *gauge_inverse;
   const Operator *gauge_column;
   mutable BlockVector work;
   mutable Vector velocity_work, pressure_work;

public:
   UpperTriangularPreconditioner(const Array<int> &offsets,
                                Solver &velocity_inverse, Solver &pressure_inverse,
                                const Operator &momentum_pressure,
                                Solver *gauge_inverse = nullptr,
                                const Operator *gauge_column = nullptr);
   void SetOperator(const Operator &) override;
   void Mult(const Vector &input, Vector &output) const override;
};

}

#endif
