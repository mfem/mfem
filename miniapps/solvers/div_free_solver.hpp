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

#ifndef MFEM_DIVFREE_SOLVER_HPP
#define MFEM_DIVFREE_SOLVER_HPP

#include "darcy_solver.hpp"

namespace mfem
{
namespace blocksolvers
{

/// Solver for B * B^T
/// Compute the product B * B^T and solve it with CG preconditioned by BoomerAMG
class BBTSolver : public Solver
{
   OperatorPtr BBT_;
   OperatorPtr BBT_prec_;
   CGSolver BBT_solver_;
public:
   BBTSolver(const HypreParMatrix &B, IterSolveParameters param);
   virtual void Mult(const Vector &x, Vector &y) const { BBT_solver_.Mult(x, y); }
   virtual void SetOperator(const Operator &op) { }
};

/// Block diagonal solver for symmetric A, each block is inverted by direct solver
class SymDirectSubBlockSolver : public DirectSubBlockSolver
{
public:
   SymDirectSubBlockSolver(const SparseMatrix& A, const SparseMatrix& block_dof)
      : DirectSubBlockSolver(A, block_dof) { }
   virtual void MultTranspose(const Vector &x, Vector &y) const { Mult(x, y); }
};

/// non-overlapping additive Schwarz smoother for saddle point systems
///                      [ M  B^T ]
///                      [ B   0  ]
class SaddleSchwarzSmoother : public Solver
{
   const SparseMatrix& agg_hdivdof_;
   const SparseMatrix& agg_l2dof_;
   OperatorPtr coarse_l2_projector_;

   Array<int> offsets_;
   mutable Array<int> offsets_loc_;
   mutable Array<int> hdivdofs_loc_;
   mutable Array<int> l2dofs_loc_;
   std::vector<OperatorPtr> solvers_loc_;
public:
   /** SaddleSchwarzSmoother solves local saddle point problems defined on a
       list of non-overlapping aggregates (of elements).
       @param M the [1,1] block of the saddle point system
       @param B the [2,1] block of the saddle point system
       @param agg_hdivdof aggregate to H(div) dofs relation table (boolean matrix)
       @param agg_l2dof aggregate to L2 dofs relation table (boolean matrix)
       @param P_l2 prolongation matrix of the discrete L2 space
       @param Q_l2 projection matrix of the discrete L2 space:
                      Q_l2 := (P_l2 W P_l2)^{-1} * P_l2 * W,
                   where W is the mass matrix of the discrete L2 space. */
   SaddleSchwarzSmoother(const HypreParMatrix& M,
                         const HypreParMatrix& B,
                         const SparseMatrix& agg_hdivdof,
                         const SparseMatrix& agg_l2dof,
                         const HypreParMatrix& P_l2,
                         const HypreParMatrix& Q_l2);
   virtual void Mult(const Vector &x, Vector &y) const;
   virtual void MultTranspose(const Vector &x, Vector &y) const { Mult(x, y); }
   virtual void SetOperator(const Operator &op) { }
};

/// Solver for local problems in SaddleSchwarzSmoother
class LocalSolver : public Solver
{
   DenseMatrix local_system_;
   DenseMatrixInverse local_solver_;
   const int offset_;
public:
   LocalSolver(const DenseMatrix &M, const DenseMatrix &B);
   virtual void Mult(const Vector &x, Vector &y) const;
   virtual void SetOperator(const Operator &op) { }
};

/// Divergence free solver.
/** Divergence free solver.
    The basic idea of the solver is to exploit a multilevel decomposition of
    Raviart-Thomas space to find a particular solution satisfying the divergence
    constraint, and then solve the remaining (divergence-free) component in the
    kernel space of the discrete divergence operator.

    For more details see
    1. Vassilevski, Multilevel Block Factorization Preconditioners (Appendix F.3),
       Springer, 2008.
    2. Voronin, Lee, Neumuller, Sepulveda, Vassilevski, Space-time discretizations
       using constrained first-order system least squares (CFOSLS).
       J. Comput. Phys. 373: 863-876, 2018. */
class DivFreeSolver : public DarcySolver
{
   const DFSData& data_;
   DFSParameters param_;
   OperatorPtr BT_;
   BBTSolver BBT_solver_;
   std::vector<Array<int>> ops_offsets_;
   Array<BlockOperator*> ops_;
   Array<BlockOperator*> blk_Ps_;
   Array<Solver*> smoothers_;
   OperatorPtr prec_;
   OperatorPtr solver_;

   void SolveParticular(const Vector& rhs, Vector& sol) const;
   void SolveDivFree(const Vector& rhs, Vector& sol) const;
   void SolvePotential(const Vector &rhs, Vector& sol) const;
public:
   DivFreeSolver(const HypreParMatrix& M, const HypreParMatrix &B,
                 const DFSData& data);
   ~DivFreeSolver();
   virtual void Mult(const Vector &x, Vector &y) const;
   virtual void SetOperator(const Operator &op) { }
   virtual int GetNumIterations() const;
};

} // namespace blocksolvers
} // namespace mfem

#endif // MFEM_DIVFREE_SOLVER_HPP
