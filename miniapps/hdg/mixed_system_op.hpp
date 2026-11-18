// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_HDG_MIXED_SYSTEM_OP
#define MFEM_HDG_MIXED_SYSTEM_OP

#include "mfem.hpp"
#include "../../general/socketstream.hpp"
#include <vector>
#include <variant>

namespace mfem
{
namespace hdg
{
/// A helper operator for mixed systems
/** Class MixedSystemOperator helps with construction of a spatial operator for
    mixed systems with (anti)symmetric weak form structure common for parabolic
    and elliptic problems. These can be written as the following block system
    (not necessarily symmetric):
    \verbatim
        ┌        ┐┌   ┐   ┌    ┐
        | Mu ±Bᵀ || u | _ | bu |
        | B  Mp  || p | ̅  | bp |
        └        ┘└   ┘   └    ┘
    \endverbatim
    following the notation of MixedSystemForm.

    The mixed system discretization provided through MixedSystemForm is solved
    implicitly with an appropriate solver and preconditioner to provide
    time derivatives for time evolution of the system. Specifically,
    these configurations are set up:
    - mixed system - Schur complement preconditioner using Jacobi
                     preconditioner for the flux mass and HypreBoomerAMG in
                     parallel for the Schur complement, or UMFPackSolver
                     / GSSmoother in serial (depending on availability and
                     @a USE_DIRECT_SOLVER_SCHUR in mixed_system_op.cpp).
    - hybridized system - HypreBoomerAMG in parallel, or UMFPackSolver
                          / GSSmoother in serial (depending on availability
                          and @a USE_DIRECT_SOLVER_HYBRIDIZATION in
                          mixed_system_op.cpp)
    - reduced system - HypreBoomerAMG in parallel, or UMFPackSolver
                       / GSSmoother in serial (depending on availability
                       and @a USE_DIRECT_SOLVER_REDUCTION in
                       mixed_system_op.cpp)

    Note that MixedSystemOperator uses a BlockVector (with offsets obtained
    through GetOffsets()) for representation of the state, which has 3
    components for hybridized systems: flux, potential and trace unknowns. This
    construction speeds up the solution procedure by providing an initial guess
    of the trace unknowns to the linear solver.

    MixedSystemOperator also handles construction of the inertial term in time
    evolving cases (indicated in the constructor), which is added to the
    MixedSystemForm and its contribution is added to the right hand side as well
    every time step.
  */
class MixedSystemOperator : public TimeDependentOperator
{
   Array<int> offsets;
   const Array<int> &ess_flux_tdofs_list;
   MixedSystemForm *mixed_system;
   LinearForm *g{};
   LinearForm *f{};
   LinearForm *h{};
#ifdef MFEM_USE_MPI
   ParMixedSystemForm *pmixed_system {};
   ParLinearForm *pg{};
   ParLinearForm *pf{};
   ParLinearForm *ph{};
#endif
   const std::vector<std::variant<Coefficient*,VectorCoefficient*>> coeffs;
   bool btime_u, btime_p;
   real_t rtol{1e-6};
   real_t atol{1e-10};
   int max_iters{1000};

   FiniteElementSpace *trace_space{};

   real_t idt{};
   std::unique_ptr<Coefficient> idtcoeff;
   std::unique_ptr<BilinearForm> Mt0, Mq0;

   std::unique_ptr<Solver> prec;
   std::string prec_str;
   std::unique_ptr<IterativeSolver> solver;
   std::string solver_str;

   mutable BlockVector x, rhs;

   class SchurPreconditioner : public Solver
   {
      const MixedSystemForm *mixed_system;
#ifdef MFEM_USE_MPI
      const ParMixedSystemForm *pmixed_system {};
#endif
      const Operator *op {};

      const char *prec_str;
      mutable std::unique_ptr<BlockDiagonalPreconditioner> schur_prec;
      mutable std::unique_ptr<SparseMatrix> S;
#ifdef MFEM_USE_MPI
      mutable std::unique_ptr<HypreParMatrix> hS;
#endif
      mutable bool reconstruct {};

      void Construct();
#ifdef MFEM_USE_MPI
      void ConstructPar();
#endif

   public:
      SchurPreconditioner(const MixedSystemForm *mixed_system);
#ifdef MFEM_USE_MPI
      SchurPreconditioner(const ParMixedSystemForm *mixed_system);
#endif

      const char *GetString() const { return prec_str; }

      void SetOperator(const Operator &op_) override
      { op = &op_; reconstruct = true; }

      void Mult(const Vector &x, Vector &y) const override;
   };

   void SetupLinearSolver(real_t rtol, real_t atol, int iters);

public:
   /// Constructor
   /** @param ess_flux_tdofs_list      list of essential TDOFs for the flux
       @param mixed_system             discretization of the mixed system
       @param rhs                      right hand sides
       @param coeffs                   array of time-dependent coefficients
                                       that need to be updated for assembly
                                       of the right-hand-side linear forms
       @param btime_u                  flag for time evolving flux
       @param btime_p                  flag for time evolving potential
    */
   MixedSystemOperator(const Array<int> &ess_flux_tdofs_list,
                       MixedSystemForm *mixed_system,
                       std::vector<LinearForm*> rhs,
                       std::vector<std::variant<Coefficient*,VectorCoefficient*>> coeffs,
                       bool btime_u = false, bool btime_p = false);

#ifdef MFEM_USE_MPI
   /// Constructor (parallel)
   /** @param ess_flux_tdofs_list      list of essential TDOFs for the flux
       @param mixed_system             discretization of the mixed system
       @param rhs                      right hand side
       @param coeffs                   array of time-dependent coefficients
                                       that need to be updated for assembly
                                       of the right-hand-side linear forms
       @param btime_u                  flag for time evolving flux
       @param btime_p                  flag for time evolving potential
    */
   MixedSystemOperator(const Array<int> &ess_flux_tdofs_list,
                       ParMixedSystemForm *mixed_system,
                       std::vector<ParLinearForm*> rhs,
                       std::vector<std::variant<Coefficient*,VectorCoefficient*>> coeffs,
                       bool btime_u = false, bool btime_p = false);
#endif

   /// Destructor
   ~MixedSystemOperator();

   /// Set the tolerance of iterative solvers
   void SetTolerance(real_t rtol_, real_t atol_ = 0.) { rtol = rtol_; atol = atol_; }

   /// Set the maximal number of iterations of iterative solvers
   void SetMaxIters(int iters_) { max_iters = iters_; }

   /// Construct state vector offsets
   /** Constructs state vector offsets corresponding to the MixedSystemForm
       @p mixed_system. Note this includes trace unknonws for hybridized
       systems. */
   static Array<int> ConstructOffsets(const MixedSystemForm &mixed_system);

   /// Get the state vector offsets
   /** @see ConstructOffsets() */
   inline const Array<int>& GetOffsets() const { return offsets; }

   /// Get the associated MixedSystemForm
   inline const MixedSystemForm& GetMixedSystemForm() const { return *mixed_system; }

#ifdef MFEM_USE_MPI
   /// Get the associated ParMixedSystemForm
   inline const ParMixedSystemForm& GetParMixedSystemForm() const
   { MFEM_VERIFY(pmixed_system, "No ParMixedSystemForm!"); return *pmixed_system; }
#endif

   void ImplicitSolve(const real_t dt, const Vector &x, Vector &k) override;

   /// Updates the operator after a change of the mesh
   void Update();
};

} // namespace hdg
} // namespace mfem

#endif // MFEM_HDG_MIXED_SYSTEM_OP
