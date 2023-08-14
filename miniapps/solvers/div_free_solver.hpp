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

#include "mfem.hpp"
#include <memory>
#include <vector>

namespace mfem
{
/// Bramble-Pasciak Conjugate Gradient
class BPCGSolver : public IterativeSolver
{
protected:
   mutable Vector r, p, g, t, r_bar, r_red, g_red;
   /// Remaining required operators
   /*  Operator list
    *  From IterativeSolver:
    *  *oper  -> A  = [M, Bt; B, 0]
    *  *prec  -> P  = diag(M0, M1)
    *  From this class:
    *  *iprec -> N  = diag(M0, 0)
    *  *pprec -> P' = P * [Id, 0; B*M0, -Id]
    */
   const Operator *iprec, *pprec;
   void UpdateVectors();

public:
   BPCGSolver() { }

#ifdef MFEM_USE_MPI
   BPCGSolver(MPI_Comm comm_) : IterativeSolver(comm_) { }
#endif

   virtual void SetOperator(const Operator &op)
   { IterativeSolver::SetOperator(op); UpdateVectors(); }

   virtual void SetPreconditioner(const Operator &pc)
   { if (Mpi::Root()) { MFEM_WARNING("No explicit preconditioner required for BPCG"); } }

   virtual void SetPreconditioner()
   { if (Mpi::Root()) { MFEM_WARNING("No explicit preconditioner required for BPCG"); } }

   virtual void SetIncompletePreconditioner(const Operator &ipc)
   { iprec = &ipc; }

   virtual void SetParticularPreconditioner(const Operator &ppc)
   { pprec = &ppc; }

   virtual void Mult(const Vector &b, Vector &x) const;
};

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

/// Parameters for the BPCG method
struct BPCGParameters : IterSolveParameters
{
   /* These are parameters for the scaling of the Q preconditioner
    * the usage of BPCG method, and the definition of the H preconditioner */
   bool use_bpcg = true;
   double q_scaling = 0.5;
   bool use_hpc = false;
};

/// Data for the divergence free solver
struct DFSData
{
   std::vector<OperatorPtr> agg_hdivdof;  // agglomerates to H(div) dofs table
   std::vector<OperatorPtr> agg_l2dof;    // agglomerates to L2 dofs table
   std::vector<OperatorPtr> P_hdiv;   // Interpolation matrix for H(div) space
   std::vector<OperatorPtr> P_l2;     // Interpolation matrix for L2 space
   std::vector<OperatorPtr> P_hcurl;  // Interpolation for kernel space of div
   std::vector<OperatorPtr> Q_l2;     // Q_l2[l] = (W_{l+1})^{-1} P_l2[l]^T W_l
   Array<int> coarsest_ess_hdivdofs;  // coarsest level essential H(div) dofs
   std::vector<OperatorPtr> C;        // discrete curl: ND -> RT, map to Null(B)
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

/// Bramble-Pasciak Solver for Darcy equation.
/** Bramble-Pasciak Solver for Darcy equation.
 *  The basic idea is to precondition the mass matrix M with a s.p.d. matrix Q
 *  such that M - Q remains s.p.d. Then we can transform the block operator into a
 *  s.p.d. operator under a modified inner product.
 *  In particular, this enable us to implement modified versions of CG iterations,
 *  that rely on efficient applications of the required transformations.
 *
 *  We offer a mass preconditioner based on a rescalling of the diagonal of the
 *  element mass matrices M_T.
 *  We consider Q_T := alpha * lambda_min * D_T, where D_T := diag(M_T), and
 *  lambda_min is the smallest eigenvalue of the following problem
 *                M_T x = lambda * D_T x.
 *  alpha is a parameter that is stricly between 0 and 1.
 *
 *  For more details, see:
 *  1. Vassilevski, Multilevel Block Factorization Preconditioners (Appendix F.3),
 *     Springer, 2008.
 *  2. James H. Bramble and Joseph E. Pasciak.
 *     A Preconditioning Technique for Indefinite Systems Resulting From Mixed
 *     Approximations of Elliptic Problems. Mathematics of Computation, 50:1–17, 1988.
 */
class BramblePasciakSolver : public DarcySolver
{
   // TODO TO be removed and included in param
   mutable bool use_bpcg;
   OperatorPtr solver_;
   // CGSolver solver_;
   // BPCGSolver bpsolver_;
   BlockOperator *oop_, *ipc_;
   ProductOperator *mop_;
   AddOperator *map_;
   ProductOperator *ppc_;
   BlockDiagonalPreconditioner *cpc_, *hpc_;
   std::unique_ptr<HypreParMatrix> M_;
   std::unique_ptr<HypreParMatrix> B_;
   std::unique_ptr<HypreParMatrix> Q_;
   Array<int> ess_zero_dofs_;

   /// User provides system.
   void Init(HypreParMatrix &M, HypreParMatrix &B,
             HypreParMatrix &Q,
             Solver &M0, Solver &M1,
             const BPCGParameters &param);

   /// Construct specific preconditioners.
   void Init(HypreParMatrix &M, HypreParMatrix &B,
             HypreParMatrix &Q,
             const BPCGParameters &param);
public:
   /// System and mass preconditioner are constructed from bilinear forms
   BramblePasciakSolver(
      const std::shared_ptr<ParBilinearForm> &mVarf,
      const std::shared_ptr<ParMixedBilinearForm> &bVarf,
      const BPCGParameters &param);

   /// System and mass preconditioner are user-provided
   BramblePasciakSolver(
      HypreParMatrix &M, HypreParMatrix &B, HypreParMatrix &Q,
      Solver &M0, Solver &M1,
      const BPCGParameters &param);

   /// Assemble a preconditioner for the mass matrix
   /** Mass preconditioner corresponds to a local re-scaling
    * based on the smallest eigenvalue of the generalized
    * eigenvalue problem locally on each element T:
    *         M_T x_T = lambda_T diag(M_T) x_T
    * and we set Q_T = 0.5 * min(lambda_T) * diag(M_T).
   */
   static HypreParMatrix *ConstructMassPreconditioner(ParBilinearForm &mVarf,
                                                      double alpha = 0.5);

   // TODO
   /// Define if BPCG will be employed in Mult
   void SetBPCG(bool use) { use_bpcg = use; }

   virtual void Mult(const Vector &x, Vector &y) const;
   virtual void SetOperator(const Operator &op) { }
   void SetEssZeroDofs(const Array<int>& dofs) { dofs.Copy(ess_zero_dofs_); }
   virtual int GetNumIterations() const;
};
} // namespace blocksolvers
} // namespace mfem

#endif // MFEM_DIVFREE_SOLVER_HPP
