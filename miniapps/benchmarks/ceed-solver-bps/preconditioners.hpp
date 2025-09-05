// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef __SOLVER_BP_HPP__
#define __SOLVER_BP_HPP__

#include "mfem.hpp"
#include <memory>

namespace mfem
{

struct SolverConfig
{
   enum SolverType
   {
      JACOBI = 0,
      FA_HYPRE = 1,
      LOR_HYPRE = 2,
      FA_AMGX = 3,
      LOR_AMGX = 4
   };
   SolverType type;
   const char *amgx_config_file = "amgx/amgx.json";
   bool inner_cg = false; //<-- use inner CG iteration for coarse solver
   bool inner_sli = false; //<-- use inner SLI iteration for coarse solver
   int inner_sli_iter = 1; //<- number of iterations for the inner SLI solver
   bool coarse_smooth = false; //<- enable level 0 smoothing
   SolverConfig(SolverType type_) : type(type_) { }

   void Print()
   {
      mfem::out << "Coarse solver: ";
      switch (type)
      {
         case JACOBI: mfem::out << "Jacobi"; break;
         case FA_HYPRE: mfem::out << "Hypre (full)"; break;
         case LOR_HYPRE: mfem::out << "Hypre (LOR)"; break;
         case FA_AMGX: mfem::out << "AmgX (full)"; break;
         case LOR_AMGX: mfem::out << "AmgX (LOR)"; break;
      }
      mfem::out << std::endl;
      // If inner_sli is true inner_cg is not used, see
      // DiffusionMultigrid::ConstructCoarseOperatorAndSolver():
      if (inner_sli) { inner_cg = false; }
      mfem::out << "Inner CG:      "
                << (inner_cg ? "On" : "Off")
                << std::endl;
      mfem::out << "Inner SLI:     " << (inner_sli ? "On" : "Off") << '\n';
      mfem::out << "Coarse smooth: " << (coarse_smooth ? "On" : "Off") << '\n';
   }
};

struct DiffusionMultigrid : GeometricMultigrid
{
   Coefficient &coeff;
   int q1d_inc;
   IntegrationRules irs;
   std::unique_ptr<ParLORDiscretization> lor;
   OperatorPtr A_coarse;
   std::shared_ptr<Solver> coarse_solver, coarse_precond;
   int smoothers_cheby_order;

   DiffusionMultigrid(
      ParFiniteElementSpaceHierarchy& hierarchy,
      Coefficient &coeff_,
      Array<int>& ess_bdr,
      SolverConfig coarse_solver_config,
      int q1d_inc_ = 0,
      int smoothers_cheby_order_ = 1);

   void ConstructBilinearForm(
      ParFiniteElementSpace &fespace,
      Array<int> &ess_bdr,
      AssemblyLevel asm_lvl);

   void ConstructOperatorAndSmoother(
      ParFiniteElementSpace &fespace,
      Array<int> &ess_bdr);

   void ConstructCoarseOperatorAndSolver(
      SolverConfig config,
      ParFiniteElementSpace &fespace,
      Array<int> &ess_bdr);

   void SetSmoothersChebyshevOrder(int new_cheby_order);
   void SetInnerSLINumIter(int inner_sli_iter);
};

} // namespace mfem

#endif
