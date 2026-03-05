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

#ifndef MFEM_DIVFREE_SOLVER_HPP
#define MFEM_DIVFREE_SOLVER_HPP

#include "darcy_solver.hpp"
#include <memory>

namespace mfem::blocksolvers
{

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
   using UniqueOperatorPtr = std::unique_ptr<OperatorPtr>;
   using UniqueHypreParMatrix = std::unique_ptr<HypreParMatrix>;

   std::vector<OperatorPtr> agg_hdivdof; // agglomerates to H(div) dofs table
   std::vector<OperatorPtr> agg_l2dof; // agglomerates to L2 dofs table
   std::vector<UniqueOperatorPtr> P_hdiv; // Interpolation matrix for H(div) space
   std::vector<UniqueOperatorPtr> P_l2; // Interpolation matrix for L2 space
   std::vector<UniqueOperatorPtr> P_hcurl; // Interpolation for kernel space of div
   std::vector<OperatorPtr> Q_l2; // Q_l2[l] = (W_{l+1})^{-1} P_l2[l]^T W_l
   Array<int> coarsest_ess_hdivdofs; // coarsest level essential H(div) dofs
   std::vector<OperatorPtr> C; // discrete curl: ND -> RT, map to Null(B)
   std::vector<UniqueHypreParMatrix> Ae;
   DFSParameters param;
};

/// Finite element spaces concerning divergence free solvers
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

/// Solvers for DFS
/// Solver for B * B^T
/// Compute the product B * B^T and solve it with CG preconditioned by BoomerAMG
class BBTSolver : public Solver
{
   OperatorPtr BBT_, BBT_prec_;
   CGSolver BBT_solver_;
public:
   BBTSolver(const HypreParMatrix &B, IterSolveParameters param);
   void Mult(const Vector &x, Vector &y) const override { BBT_solver_.Mult(x, y); }
   void SetOperator(const Operator &op) override { }
};

/// Block diagonal solver for symmetric A, each block is inverted by direct solver
class SymDirectSubBlockSolver : public DirectSubBlockSolver
{
public:
   SymDirectSubBlockSolver(const SparseMatrix& A, const SparseMatrix& block_dof)
      : DirectSubBlockSolver(A, block_dof) { }
   void MultTranspose(const Vector &x, Vector &y) const override { Mult(x, y); }
};

/// non-overlapping additive Schwarz smoother for saddle point systems
///                      [ M  B^T ]
///                      [ B   0  ]
class SaddleSchwarzSmoother : public Solver
{
   const SparseMatrix &agg_hdivdof_, &agg_l2dof_;
   OperatorPtr coarse_l2_projector_;

   Array<int> offsets_;
   mutable Array<int> offsets_loc_, hdivdofs_loc_, l2dofs_loc_;
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
                         const ProductOperator& Q_l2);
   void Mult(const Vector &x, Vector &y) const override;
   void MultTranspose(const Vector &x, Vector &y) const override { Mult(x, y); }
   void SetOperator(const Operator &op) override { }
};

/// Solver for local problems in SaddleSchwarzSmoother
class LocalSolver : public Solver
{
   DenseMatrix local_system_;
   DenseMatrixInverse local_solver_;
   const int offset_;
public:
   LocalSolver(const DenseMatrix &M, const DenseMatrix &B);
   void Mult(const Vector &x, Vector &y) const override;
   void SetOperator(const Operator &op) override { }
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
   std::vector<std::unique_ptr<BlockOperator>> ops_;
   std::vector<std::unique_ptr<BlockOperator>> blk_Ps_;
   std::vector<std::unique_ptr<Solver>> smoothers_;
   OperatorPtr prec_, solver_;

   void SolveParticular(const Vector& rhs, Vector& sol) const;
   void SolveDivFree(const Vector& rhs, Vector& sol) const;
   void SolvePotential(const Vector &rhs, Vector& sol) const;
public:
   DivFreeSolver(const HypreParMatrix& M, const HypreParMatrix &B,
                 const DFSData& data);
   void Mult(const Vector &x, Vector &y) const override;
   void SetOperator(const Operator &op) override { }
   int GetNumIterations() const override;
};

} // namespace mfem::blocksolvers

#endif // MFEM_DIVFREE_SOLVER_HPP
