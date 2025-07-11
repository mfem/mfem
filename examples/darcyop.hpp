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
   ParDarcyForm *pdarcy {};
   ParLinearForm *pg{}, *pf{}, *ph{};
#endif
   const Array<Coefficient*> &coeffs;
   SolverType solver_type;
   bool btime_u, btime_p;
   real_t rtol{1e-6}, atol{1e-10};
   int max_iters{1000};

   FiniteElementSpace *trace_space{};

   real_t idt{};
   std::unique_ptr<Coefficient> idtcoeff;
   std::unique_ptr<BilinearForm> Mt0, Mq0;

   std::string lsolver_str;
   std::unique_ptr<Solver> prec, lin_prec;
   std::string prec_str, lin_prec_str;
   std::unique_ptr<IterativeSolver> solver;
   std::string solver_str;
   std::unique_ptr<IterativeSolverMonitor> monitor;
   int monitor_step{-1};

   mutable BlockVector x, rhs;

   class SchurPreconditioner : public Solver
   {
      const DarcyForm *darcy;
#ifdef MFEM_USE_MPI
      const ParDarcyForm *pdarcy {};
#endif
      const Operator *op {};
      bool nonlinear;

      const char *prec_str;
      mutable std::unique_ptr<BlockDiagonalPreconditioner> darcyPrec;
      mutable std::unique_ptr<SparseMatrix> S;
#ifdef MFEM_USE_MPI
      mutable std::unique_ptr<HypreParMatrix> hS;
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
      BlockVector &x;
      const BlockVector &rhs;
      Type type;
      real_t rtol;
      int it_prev{};
      Vector sol_prev;

      bool CheckSolution(const Vector &x, const Vector &y) const;
      virtual void ReduceValues(real_t diff[], int num) const { }

   public:
      SolutionController(DarcyForm &darcy, BlockVector &x, const BlockVector &rhs,
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

      void ReduceValues(real_t diff[], int num) const override;

   public:
      ParSolutionController(ParDarcyForm &pdarcy, BlockVector &x,
                            const BlockVector &rhs, Type type, real_t rtol);

      void MonitorSolution(int it, real_t norm, const Vector &x,
                           bool final) override;
   };
#endif //MFEM_USE_MPI

private:
   SolutionController::Type sol_type{SolutionController::Type::Native};

   class IterativeGLVis : public IterativeSolverMonitor
   {
   protected:
      DarcyForm &darcy;
      BlockVector &x;
      const BlockVector &rhs;
      int step;
      bool save_files;

      socketstream q_sock, t_sock;

      virtual void StreamPreamble(socketstream &ss) { }
      virtual std::string FormFilename(const char *base, int it,
                                       const char *suff = "gf");
   public:
      IterativeGLVis(DarcyForm &darcy, BlockVector &x, const BlockVector &rhs,
                     int step = 0, bool save_files = false);

      void MonitorSolution(int it, real_t norm, const Vector &x,
                           bool final) override;

      bool RequiresUpdatedSolution() const override { return true; }
   };

#ifdef MFEM_USE_MPI
   class ParIterativeGLVis : public IterativeGLVis
   {
      ParDarcyForm &pdarcy;

      void StreamPreamble(socketstream &ss) override;
      std::string FormFilename(const char *base, int it,
                               const char *suff = "gf") override;
   public:
      ParIterativeGLVis(ParDarcyForm &pdarcy_, BlockVector &x, const BlockVector &rhs,
                        int step = 0, bool save_files = false)
         : IterativeGLVis(pdarcy_, x, rhs, step), pdarcy(pdarcy_) { }
   };
#endif //MFEM_USE_MPI

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

   void SetTolerance(real_t rtol_, real_t atol_ = 0.) { rtol = rtol_; atol = atol_; }
   void SetMaxIters(int iters_) { max_iters = iters_; }

   void EnableSolutionController(SolutionController::Type type) { sol_type = type; }
   void EnableIterationsVisualization(int vis_step = 0) { monitor_step = vis_step; }

   static Array<int> ConstructOffsets(const DarcyForm &darcy);
   inline const Array<int>& GetOffsets() const { return offsets; }

   void ImplicitSolve(const real_t dt, const Vector &x, Vector &k) override;
};

}

#endif
