// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_BLOCK_HYBRIDIZATION_SOLVER_HPP
#define MFEM_BLOCK_HYBRIDIZATION_SOLVER_HPP

#include "darcy_solver.hpp"

namespace mfem
{
namespace blocksolvers
{


class BlockHybridizationSolver : public DarcySolver
{
   ParFiniteElementSpace trial_space, test_space;
   ParFiniteElementSpace *c_fes;
   Array<int> hat_offsets, test_offsets, data_offsets, ipiv_offsets, mixed_dofs;
   double *data;
   int *ipiv;
   bool elimination_;
   SparseMatrix *Ct;
   HypreBoomerAMG *M;
   OperatorPtr pH;
   CGSolver solver_;

   void Init(const int ne);
   void ConstructCt(const ParFiniteElementSpace &c_space);
   void ConstructH(const std::shared_ptr<ParBilinearForm> &a,
                   const std::shared_ptr<ParMixedBilinearForm> &b,
                   const Array<int> &marker,
                   const ParFiniteElementSpace &c_space);
   void ReduceRHS(const Vector &b, const Vector &sol, BlockVector &rhs,
                  Vector &b_r) const;
   void ComputeSolution(Vector &y,
                        BlockVector &rhs,
                        const Vector &rhs_r,
                        Array<int> &block_offsets) const;
public:
   BlockHybridizationSolver(const std::shared_ptr<ParBilinearForm> &a,
                            const std::shared_ptr<ParMixedBilinearForm> &b,
                            const IterSolveParameters &param,
                            const Array<int> &ess_bdr_attr);
   ~BlockHybridizationSolver();
   virtual void Mult(const Vector &x, Vector &y) const;
   virtual void SetOperator(const Operator &op) { }
   virtual int GetNumIterations() const { return solver_.GetNumIterations(); }
};

} // namespace blocksolvers
} // namespace mfem

#endif // MFEM_BLOCK_HYBRIDIZATION_SOLVER_HPP
