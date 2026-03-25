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

#ifndef MFEM_SCATTER_SOLVER
#define MFEM_SCATTER_SOLVER

#include "mfem.hpp"
#include <vector>
#include <memory>

namespace mfem
{
namespace plasma
{
class CoupledOperator : public TimeDependentOperator
{
   Array<int> ess_q_tdofs_list, ess_E_tdofs_list;
   std::vector<Coefficient*> kappa;
   std::vector<std::unique_ptr<Coefficient>> ikappa;
   Coefficient *sigma;
   FiniteElementSpace *q_space, *p_space, *E_space, *B_space, *tr_space;
   bool imex_plasma{true};

   std::vector<Array<int>> bdr_is_dirichlet_plasma;
   std::vector<Array<int>> bdr_is_neumann_maxwell;
   std::vector<VectorCoefficient*> bdr_coeffs_plasma;
   std::vector<std::unique_ptr<VectorCoefficient>> bdr_gcoeffs_plasma;
   std::vector<VectorCoefficient*> bdr_coeffs_maxwell;
   Array<int> offsets;
   BilinearForm *ME{}, *Mq{};
   NonlinearForm *Mpnl{}, *Kp{};
   BilinearForm *Mpdt{};
   LinearForm *bq{}, *bp{};
   MixedBilinearForm *CE{}, *Dq{};
   DiscreteLinearOperator *CdE{};
   DarcyForm *darcy{};
   real_t idt{};
   std::unique_ptr<Coefficient> idtcoeff;
   std::unique_ptr<FluxFunction> flux;
   std::unique_ptr<NumericalFlux> numericalFlux;

   class ReducedOperator : public Operator
   {
      Coefficient *sigma;
      DarcyForm *darcy;
      FiniteElementSpace *fes_E;
      Operator &op_pl, &op_max;

      Array<int> offsets, offsets_x;
      Vector *darcy_bp{};
      Vector darcy_bp_lin;
      OperatorHandle op;
      Array<int> ess_tdofs_list;
      mutable OperatorHandle grad;

   public:
      ReducedOperator(Coefficient *sigma, DarcyForm *darcy, FiniteElementSpace *fes_E,
                      Operator &pl, Operator &max);
      ~ReducedOperator();

      void SetEssentialTDOFs(const Array<int> &u_tdofs_list,
                             const Array<int> &E_tdofs_list);
      void EliminateRHS(const Vector &x, Vector &b) const;

      void Mult(const Vector &x, Vector &y) const override;
      void MultUnconstrained(const Vector &x, Vector &y) const;
      Operator& GetGradient(const Vector &x) const override;
   };

public:
   enum class BCType
   {
      Zero,
      Dirichlet,
      Free,
      Neumann,
   };

   CoupledOperator(std::vector<Coefficient*> kappa, Coefficient *sigma,
                   std::vector<std::pair<BCType,VectorCoefficient*>> bcs_plasma,
                   std::vector<std::pair<BCType,VectorCoefficient*>> bcs_maxwell,
                   FiniteElementSpace *q_space, FiniteElementSpace *p_space,
                   FiniteElementSpace *E_space, FiniteElementSpace *B_space,
                   FiniteElementSpace *tr_space = NULL, real_t cs = 1.,
                   real_t td = 0.5, bool imex_plasma = true);
   ~CoupledOperator();

   static Array<int> ConstructOffsets(
      const FiniteElementSpace *q_space, const FiniteElementSpace *p_space,
      const FiniteElementSpace *E_space, const FiniteElementSpace *B_space,
      const FiniteElementSpace *tr_space = NULL);

   void ProjectIC(BlockVector &x, VectorCoefficient &p) const;

   void ImplicitSolve(const double dt, const Vector &x, Vector &y) override;
};

} // namespace plasma
} // namespace mfem

#endif // MFEM_SCATTER_SOLVER
