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

#ifndef MFEM_COUPLEDOP
#define MFEM_COUPLEDOP

#include "mfem.hpp"

namespace mfem
{
class CoupledOperator : public TimeDependentOperator
{
   Array<int> ess_u_tdofs_list, ess_E_tdofs_list;
   Coefficient *sigma;
   const Array<LinearForm*> &lfs;
   const Array<Coefficient*> &coeffs;
   FiniteElementSpace *u_space, *n_space, *E_space, *B_space, *tr_space;
   Array<int> offsets;
   BilinearForm *ME{}, *Mu{}, *Mn{};
   MixedBilinearForm *CE{}, *Du{};
   DiscreteLinearOperator *CdE{};
   DarcyForm *darcy{};
   real_t idt{};
   Coefficient *idtcoeff{}, *dtcoeff{};

   class ReducedOperator : public Operator
   {
      Array<int> offsets, offsets_x;
      Coefficient *sigma;
      DarcyForm *darcy;
      BlockVector *darcy_rhs{};
      Vector darcy_rhs_lin;
      FiniteElementSpace *fes_E;
      OperatorHandle op;
      Array<int> ess_tdofs_list;
      mutable OperatorHandle grad;

   public:
      ReducedOperator(Coefficient *sigma, DarcyForm *darcy, FiniteElementSpace *fes_E,
                      Operator &pl, Operator &max);
      ~ReducedOperator();

      void SetEssentialTDOFs(const Array<int> &u_tdofs_list,
                             const Array<int> &E_tdofs_list);
      void SetDarcyRHS(BlockVector &rhs) { darcy_rhs = &rhs; darcy_rhs_lin = rhs; }
      void EliminateRHS(const Vector &x, Vector &b) const;

      void Mult(const Vector &x, Vector &y) const override;
      void MultUnconstrained(const Vector &x, Vector &y) const;
      Operator& GetGradient(const Vector &x) const override;
   };

public:
   CoupledOperator(const Array<int> &bdr_u_ess, const Array<int> &bdr_E_is_ess,
                   Coefficient *sigma,
                   const Array<LinearForm*> &lfs, const Array<Coefficient*> &coeffs,
                   FiniteElementSpace *u_space, FiniteElementSpace *n_space,
                   FiniteElementSpace *E_space, FiniteElementSpace *B_space,
                   FiniteElementSpace *tr_space = NULL, real_t td = 0.5);
   ~CoupledOperator();

   static Array<int> ConstructOffsets(
      const FiniteElementSpace *u_space, const FiniteElementSpace *n_space,
      const FiniteElementSpace *E_space, const FiniteElementSpace *B_space,
      const FiniteElementSpace *tr_space = NULL);

   void ImplicitSolve(const double dt, const Vector &x, Vector &y) override;
};
}

#endif
