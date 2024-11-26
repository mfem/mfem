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

#include "../config/config.hpp"
#include "darcyform.hpp"
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
   const Array<Coefficient*> &coeffs;
   SolverType solver_type;
   bool btime_u, btime_p;

   FiniteElementSpace *trace_space{};

   real_t idt{};
   Coefficient *idtcoeff{};
   BilinearForm *Mt0{}, *Mq0{};

   const char *lsolver_str{};
   Solver *prec{};
   const char *prec_str{};
   IterativeSolver *solver{};
   const char *solver_str{};
   IterativeSolverMonitor *monitor{};
   int monitor_step{-1};
   SparseMatrix *S{};

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

public:
   DarcyOperator(const Array<int> &ess_flux_tdofs_list, DarcyForm *darcy,
                 LinearForm *g, LinearForm *f, LinearForm *h, const Array<Coefficient*> &coeffs,
                 SolverType stype = SolverType::LBFGS,  bool bflux_u = true,
                 bool btime_p = true);
   ~DarcyOperator();

   void EnableIterationsVisualization(int vis_step = 0) { monitor_step = vis_step; }

   static Array<int> ConstructOffsets(const DarcyForm &darcy);
   inline const Array<int>& GetOffsets() const { return offsets; }

   void ImplicitSolve(const real_t dt, const Vector &x, Vector &k) override;
};

}

#endif
