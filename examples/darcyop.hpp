// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_DARCYOP
#define MFEM_DARCYOP

#include "mfem.hpp"
#include "../general/socketstream.hpp"

namespace mfem
{

class DarcyOperator : public TimeDependentOperator
{
public:
   enum class SolverType
   {
      Default = 0,
      LBFGS,
      LBB,
      Newton,
      KINSol,
   };

private:
   Array<int> offsets;
   const Array<int> &ess_flux_tdofs_list;
   DarcyForm *darcy;
   LinearForm *g, *f, *h;
#ifdef MFEM_USE_MPI
   ParDarcyForm *pdarcy;
   ParLinearForm *pg, *pf, *ph;
#endif
   const Array<Coefficient*> &coeffs;
   SolverType solver_type;
   bool btime_u, btime_p;

   FiniteElementSpace *trace_space{};

   real_t idt{};
   Coefficient *idtcoeff{};
   BilinearForm *Mt0{}, *Mq0{};

   std::string lsolver_str;
   Solver *prec{}, *lin_prec{};
   std::string prec_str;
   IterativeSolver *solver{};
   std::string solver_str;
   IterativeSolverMonitor *monitor{};
   int monitor_step{-1};

   class SchurPreconditioner : public Solver
   {
      const DarcyForm *darcy;
#ifdef MFEM_USE_MPI
      const ParDarcyForm *pdarcy {};
#endif
      const Operator *op {};
      bool nonlinear;

      const char *prec_str;
      mutable BlockDiagonalPreconditioner *darcyPrec{};
      mutable SparseMatrix *S{};
#ifdef MFEM_USE_MPI
      mutable HypreParMatrix *hS {};
#endif
      mutable bool reconstruct {};

      void Construct(const Vector &x) const;
#ifdef MFEM_USE_MPI
      void ConstructPar(const Vector &x) const;
#endif

   public:
      SchurPreconditioner(const DarcyForm *darcy, bool nonlinear = false);
#ifdef MFEM_USE_MPI
      SchurPreconditioner(const ParDarcyForm *darcy, bool nonlinear = false);
#endif
      ~SchurPreconditioner();

      const char *GetString() const { return prec_str; }

      void SetOperator(const Operator &op_) override
      { op = &op_; reconstruct = true; }

      void Mult(const Vector &x, Vector &y) const override;
   };

public:
   class SolutionController : public IterativeSolverController
   {
   public:
      enum class Type
      {
         Native,
         Flux,
         Potential
      };

   protected:
      DarcyForm &darcy;
      const BlockVector &rhs;
      Type type;
      real_t rtol;
      int it_prev{};
      Vector sol_prev;

      bool CheckSolution(const Vector &x, const Vector &y) const;

   public:
      SolutionController(DarcyForm &darcy, const BlockVector &rhs,
                         Type type, real_t rtol);

      void MonitorSolution(int it, real_t norm, const Vector &x,
                           bool final) override;

      bool RequiresUpdatedSolution() const override { return true; }
   };

#ifdef MFEM_USE_MPI
   class ParSolutionController : public SolutionController
   {
   protected:
      ParDarcyForm &pdarcy;

   public:
      ParSolutionController(ParDarcyForm &pdarcy, const BlockVector &rhs,
                            Type type, real_t rtol);

      void MonitorSolution(int it, real_t norm, const Vector &x,
                           bool final) override;
   };
#endif //MFEM_USE_MPI

private:
   SolutionController::Type sol_type{SolutionController::Type::Native};

   class IterativeGLVis : public IterativeSolverMonitor
   {
      DarcyOperator *p;
      int step;
      socketstream q_sock, t_sock;
   public:
      IterativeGLVis(DarcyOperator *p, int step = 0);

      void MonitorSolution(int it, real_t norm, const Vector &x,
                           bool final) override;
   };

   void SetupNonlinearSolver(real_t rtol, real_t atol, int iters);
   void SetupLinearSolver(real_t rtol, real_t atol, int iters);

public:
   DarcyOperator(const Array<int> &ess_flux_tdofs_list, DarcyForm *darcy,
                 LinearForm *g, LinearForm *f, LinearForm *h, const Array<Coefficient*> &coeffs,
                 SolverType stype = SolverType::LBFGS,  bool bflux_u = true,
                 bool btime_p = true);
#ifdef MFEM_USE_MPI
   DarcyOperator(const Array<int> &ess_flux_tdofs_list, ParDarcyForm *darcy,
                 ParLinearForm *g, ParLinearForm *f, ParLinearForm *h,
                 const Array<Coefficient*> &coeffs,
                 SolverType stype = SolverType::LBFGS,  bool bflux_u = true,
                 bool btime_p = true);
#endif

   ~DarcyOperator();

   void EnableSolutionConstroller(SolutionController::Type type) { sol_type = type; }
   void EnableIterationsVisualization(int vis_step = 0) { monitor_step = vis_step; }

   static Array<int> ConstructOffsets(const DarcyForm &darcy);
   inline const Array<int>& GetOffsets() const { return offsets; }

   void ImplicitSolve(const real_t dt, const Vector &x, Vector &k) override;
};

}

#endif
