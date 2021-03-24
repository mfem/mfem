// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
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

#include "mfem.hpp"
#include <memory>

namespace mfem
{
namespace blocksolvers
{

/// Parameters for iterative solver
struct IterSolveParameters
{
   int print_level = 0;
   int max_iter = 500;
   double abs_tol = 1e-12;
   double rel_tol = 1e-9;
};

/// Parameters for the divergence free solver
struct DFSParameters : IterSolveParameters
{
   /** There are three components in the solver: a particular solution
       satisfying the divergence constraint, the remaining div-free component of
       the flux, and the pressure. When coupled_solve == false, the three
       components will be solved one by one in the aforementioned order.
       Otherwise, they will be solved at the same time. */
   bool coupled_solve = false;
   bool verbose = false;
   IterSolveParameters coarse_solve_param;
   IterSolveParameters BBT_solve_param;
};

/// Data for the divergence free solver
struct DFSData
{
   Array<OperatorPtr> agg_hdivdof;    // agglomerates to H(div) dofs table
   Array<OperatorPtr> agg_l2dof;      // agglomerates to L2 dofs table
   Array<OperatorPtr> P_hdiv;         // Interpolation matrix for H(div) space
   Array<OperatorPtr> P_l2;           // Interpolation matrix for L2 space
   Array<OperatorPtr> P_hcurl;        // Interpolation for kernel space of div
   Array<OperatorPtr> Q_l2;           // Q_l2[l] = (W_{l+1})^{-1} P_l2[l]^T W_l
   Array<int> coarsest_ess_hdivdofs;  // coarsest level essential H(div) dofs
   Array<OperatorPtr> C;              // discrete curl: ND -> RT, map to Null(B)
   DFSParameters param;
};

/// Finite element spaces concerning divergence free solver.
/// The main usage of this class is to collect data needed for the solver.
class DFSSpaces
{
   RT_FECollection hdiv_fec_;
   L2_FECollection l2_fec_;
   std::unique_ptr<FiniteElementCollection> hcurl_fec_;
   L2_FECollection l2_0_fec_;

   std::unique_ptr<ParFiniteElementSpace> coarse_hdiv_fes_;
   std::unique_ptr<ParFiniteElementSpace> coarse_l2_fes_;
   std::unique_ptr<ParFiniteElementSpace> coarse_hcurl_fes_;
   std::unique_ptr<ParFiniteElementSpace> l2_0_fes_;

   std::unique_ptr<ParFiniteElementSpace> hdiv_fes_;
   std::unique_ptr<ParFiniteElementSpace> l2_fes_;
   std::unique_ptr<ParFiniteElementSpace> hcurl_fes_;

   std::vector<SparseMatrix> el_l2dof_;
   const Array<int>& ess_bdr_attr_;
   Array<int> all_bdr_attr_;

   int level_;
   DFSData data_;

   void MakeDofRelationTables(int level);
   void DataFinalize();
public:
   DFSSpaces(int order, int num_refine, ParMesh *mesh,
             const Array<int>& ess_attr, const DFSParameters& param);

   /** This should be called each time when the mesh (where the FE spaces are
       defined) is refined. The spaces will be updated, and the prolongation for
       the spaces and other data needed for the div-free solver are stored. */
   void CollectDFSData();

   const DFSData& GetDFSData() const { return data_; }
   ParFiniteElementSpace* GetHdivFES() const { return hdiv_fes_.get(); }
   ParFiniteElementSpace* GetL2FES() const { return l2_fes_.get(); }
};

/// Abstract solver class for Darcy's flow
class DarcySolver : public Solver
{
protected:
   Array<int> offsets_;
public:
   DarcySolver(int size0, int size1) : Solver(size0 + size1), offsets_(3)
   { offsets_[0] = 0; offsets_[1] = size0; offsets_[2] = height; }
   virtual int GetNumIterations() const = 0;
};

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
   Array<OperatorPtr> solvers_loc_;
public:
   /** SaddleSchwarzSmoother solves local saddle point problems defined on a
       list of non-overlapping aggregates (of elements).
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

/// Wrapper for the block-diagonal-preconditioned MINRES defined in ex5p.cpp
class BDPMinresSolver : public DarcySolver
{
   BlockOperator op_;
   BlockDiagonalPreconditioner prec_;
   OperatorPtr BT_;
   OperatorPtr S_;   // S_ = B diag(M)^{-1} B^T
   MINRESSolver solver_;
   Array<int> ess_zero_dofs_;
public:
   BDPMinresSolver(HypreParMatrix& M, HypreParMatrix& B,
                   IterSolveParameters param);
   virtual void Mult(const Vector & x, Vector & y) const;
   virtual void SetOperator(const Operator &op) { }
   void SetEssZeroDofs(const Array<int>& dofs) { dofs.Copy(ess_zero_dofs_); }
   virtual int GetNumIterations() const { return solver_.GetNumIterations(); }
};

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
   Array<Array<int>> ops_offsets_;
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
