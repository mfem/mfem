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

#ifndef MFEM_CONSTRAINED
#define MFEM_CONSTRAINED

#include "solvers.hpp"
#include "blockoperator.hpp"
#include "sparsemat.hpp"
#include "hypre.hpp"

namespace mfem
{

class FiniteElementSpace;
#ifdef MFEM_USE_MPI
class ParFiniteElementSpace;
#endif

/** @brief An abstract class to solve the constrained system $ Ax = f $
    subject to the constraint $ B x = r $.

    Although implementations may not use the below formulation, for
    understanding some of its methods and notation you can think of
    it as solving the saddle-point system

     (  A   B^T  )  ( x )          (  f  )
     (  B        )  ( lambda )  =  (  r  )

    Do not confuse with ConstrainedOperator, which handles only simple
    pointwise constraints and is not a Solver.

    The height and width of this object as an IterativeSolver are the same as
    just the unconstrained operator $ A $, and the Mult() interface just
    takes $ f $ as an argument. You can set $ r $ with
    SetConstraintRHS() (it defaults to zero) and get the Lagrange multiplier
    solution with GetMultiplierSolution().

    Alternatively, you can use LagrangeSystemMult() to solve the block system
    shown above.

    This abstract object unifies this interface so that derived classes can
    solve whatever linear system makes sense and the interface will provide
    uniform access to solutions, Lagrange multipliers, etc. */
class ConstrainedSolver : public IterativeSolver
{
public:
#ifdef MFEM_USE_MPI
   ConstrainedSolver(MPI_Comm comm, Operator& A_, Operator& B_);
#endif
   ConstrainedSolver(Operator& A_, Operator& B_);

   virtual ~ConstrainedSolver() { }

   void SetOperator(const Operator& op) override { }

   /** @brief Set the right-hand side r for the constraint B x = r

       (r defaults to zero if you don't call this) */
   virtual void SetConstraintRHS(const Vector& r);

   /** @brief Return the Lagrange multiplier solution in lambda

       Mult() only gives you x, this provides access to lambda

       Does not make sense unless you've already solved the constrained
       system with Mult() or LagrangeSystemMult() */
   void GetMultiplierSolution(Vector& lambda) const { lambda = multiplier_sol; }

   /** @brief Solve for $ x $ given $ f $.

       If you want to set $ r $, call SetConstraintRHS() before this.

       If you want to get $ \lambda $, call GetMultiplierSolution() after
       this.

       The base class implementation calls LagrangeSystemMult(), so derived
       classes must implement either this or LagrangeSystemMult() */
   void Mult(const Vector& f, Vector& x) const override;

   /** @brief Solve for (x, lambda) given (f, r)

       The base class implementation calls Mult(), so derived classes
       must implement either this or Mult() */
   virtual void LagrangeSystemMult(const Vector& f_and_r,
                                   Vector& x_and_lambda) const;

protected:
   Operator& A;
   Operator& B;

   mutable Vector constraint_rhs;
   mutable Vector multiplier_sol;
   mutable Vector workb;
   mutable Vector workx;

private:
   void Initialize();
};


/** @brief Perform elimination of a single constraint.

    See EliminationProjection, EliminationCGSolver

    This keeps track of primary / secondary tdofs and does small dense block
    solves to eliminate constraints from a global system.

    $ B_s^{-1} $ maps the lagrange space into secondary dofs, while
    $ -B_s^{-1} B_p $ maps primary dofs to secondary dofs. */
class Eliminator
{
public:
   Eliminator(const SparseMatrix& B, const Array<int>& lagrange_dofs,
              const Array<int>& primary_tdofs,
              const Array<int>& secondary_tdofs);

   const Array<int>& LagrangeDofs() const { return lagrange_tdofs; }
   const Array<int>& PrimaryDofs() const { return primary_tdofs; }
   const Array<int>& SecondaryDofs() const { return secondary_tdofs; }

   /// Given primary dofs in in, return secondary dofs in out
   /// This applies $ -B_s^{-1} B_p $.
   void Eliminate(const Vector& in, Vector& out) const;

   /// Transpose of Eliminate(), applies $ -B_p^T B_s^{-T} $
   void EliminateTranspose(const Vector& in, Vector& out) const;

   /// Maps Lagrange multipliers to secondary dofs, applies $ B_s^{-1} $
   void LagrangeSecondary(const Vector& in, Vector& out) const;

   /// Transpose of LagrangeSecondary()
   void LagrangeSecondaryTranspose(const Vector& in, Vector& out) const;

   /// Return $ -B_s^{-1} B_p $ explicitly assembled in mat
   void ExplicitAssembly(DenseMatrix& mat) const;

private:
   Array<int> lagrange_tdofs;
   Array<int> primary_tdofs;
   Array<int> secondary_tdofs;

   DenseMatrix Bp;
   DenseMatrix Bs;  // gets inverted in place
   LUFactors Bsinverse;
   DenseMatrix BsT;   // gets inverted in place
   LUFactors BsTinverse;
   Array<int> ipiv;
   Array<int> ipivT;
};


/** Collects action of several Eliminator objects to perform elimination of
    constraints.

    Works in parallel, but each Eliminator must be processor local, and must
    operate on disjoint degrees of freedom (ie, the primary and secondary dofs
    for one Eliminator must have no overlap with any dofs from a different
    Eliminator). */
class EliminationProjection : public Operator
{
public:
   EliminationProjection(const Operator& A, Array<Eliminator*>& eliminators);

   void Mult(const Vector& x, Vector& y) const override;

   void MultTranspose(const Vector& x, Vector& y) const override;

   /** @brief Assemble this projector as a (processor-local) SparseMatrix.

       Some day we may also want to try approximate variants. */
   SparseMatrix * AssembleExact() const;

   /** Given Lagrange multiplier right-hand-side $ g $, return
       $ \tilde{g} $ */
   void BuildGTilde(const Vector& g, Vector& gtilde) const;

   /** After a solve, recover the Lagrange multiplier. */
   void RecoverMultiplier(const Vector& primalrhs,
                          const Vector& primalvars, Vector& lm) const;

private:
   const Operator& Aop;
   Array<Eliminator*> eliminators;
};


#ifdef MFEM_USE_MPI
/** @brief Solve constrained system by eliminating the constraint; see
    ConstrainedSolver

    Solves the system with the operator $ P^T A P + Z_P $, where P is
    EliminationProjection and Z_P is the identity on the eliminated dofs. */
class EliminationSolver : public ConstrainedSolver
{
public:
   /** @brief Constructor, with explicit splitting into primary/secondary dofs.

       This constructor uses a single elimination block (per processor), which
       provides the most general algorithm but is also not scalable

       The secondary_dofs are eliminated from the system in this algorithm,
       as they can be written in terms of the primary_dofs.

       Both primary_dofs and secondary_dofs are in the local truedof numbering;
       All elimination has to be done locally on processor, though the global
       system can be parallel. */
   EliminationSolver(HypreParMatrix& A, SparseMatrix& B,
                     Array<int>& primary_dofs,
                     Array<int>& secondary_dofs);

   /** @brief Constructor, elimination is by blocks.

       Each block is eliminated independently; if the blocks are reasonably
       small this can be reasonably efficient.

       The nonzeros in B are assumed to be in disjoint rows and columns; the
       rows are identified with the constraint_rowstarts array, the secondary
       dofs are assumed to be the first nonzeros in the rows. */
   EliminationSolver(HypreParMatrix& A, SparseMatrix& B,
                     Array<int>& constraint_rowstarts);

   ~EliminationSolver();

   void Mult(const Vector& x, Vector& y) const override;

   void SetOperator(const Operator& op) override
   { MFEM_ABORT("Operator cannot be reset!"); }

   void SetPreconditioner(Solver& precond) override
   { prec = &precond; }

protected:
   /// Internal utility routine; assembles eliminated matrix explicitly
   void BuildExplicitOperator();

   /// Build preconditioner for eliminated system
   virtual Solver* BuildPreconditioner() const = 0;
   /// Select krylov solver for eliminated system
   virtual IterativeSolver* BuildKrylov() const = 0;

   HypreParMatrix& hA;
   Array<Eliminator*> eliminators;
   EliminationProjection * projector;
   HypreParMatrix * h_explicit_operator;
   mutable IterativeSolver* krylov;
   mutable Solver* prec;
};


/** EliminationSolver using CG and HypreBoomerAMG */
class EliminationCGSolver : public EliminationSolver
{
public:
   EliminationCGSolver(HypreParMatrix& A, SparseMatrix& B,
                       Array<int>& constraint_rowstarts,
                       int dimension_=0, bool reorder_=false) :
      EliminationSolver(A, B, constraint_rowstarts),
      dimension(dimension_), reorder(reorder_)
   { }

protected:
   Solver* BuildPreconditioner() const override
   {
      HypreBoomerAMG * h_prec = new HypreBoomerAMG(*h_explicit_operator);
      h_prec->SetPrintLevel(0);
      if (dimension > 0) { h_prec->SetSystemsOptions(dimension, reorder); }
      return h_prec;
   }

   IterativeSolver* BuildKrylov() const override
   { return new CGSolver(GetComm()); }

private:
   int dimension;
   bool reorder;
};

/** EliminationSolver using GMRES and HypreBoomerAMG */
class EliminationGMRESSolver : public EliminationSolver
{
public:
   EliminationGMRESSolver(HypreParMatrix& A, SparseMatrix& B,
                          Array<int>& constraint_rowstarts,
                          int dimension_=0, bool reorder_=false) :
      EliminationSolver(A, B, constraint_rowstarts),
      dimension(dimension_), reorder(reorder_)
   { }

protected:
   Solver* BuildPreconditioner() const override
   {
      HypreBoomerAMG * h_prec = new HypreBoomerAMG(*h_explicit_operator);
      h_prec->SetPrintLevel(0);
      if (dimension > 0) { h_prec->SetSystemsOptions(dimension, reorder); }
      return h_prec;
   }

   IterativeSolver* BuildKrylov() const override
   { return new GMRESSolver(GetComm()); }

private:
   int dimension;
   bool reorder;
};

/** @brief Solve constrained system with penalty method; see ConstrainedSolver.

    Only approximates the solution, better approximation with higher penalty,
    but with higher penalty the preconditioner is less effective. */
class PenaltyConstrainedSolver : public ConstrainedSolver
{
public:
   PenaltyConstrainedSolver(HypreParMatrix& A, SparseMatrix& B,
                            real_t penalty_);

   PenaltyConstrainedSolver(HypreParMatrix& A, HypreParMatrix& B,
                            real_t penalty_);

   PenaltyConstrainedSolver(HypreParMatrix& A, HypreParMatrix& B,
                            Vector& penalty_);

   ~PenaltyConstrainedSolver();

   void Mult(const Vector& x, Vector& y) const override;

   void SetOperator(const Operator& op) override
   { MFEM_ABORT("Operator cannot be reset!"); }

   void SetPreconditioner(Solver& precond) override
   { prec = &precond; }

protected:
   /// Initialize the matrix A + B*D*B^T for the constrained linear system
   /// A - original matrix (N x N square matrix)
   /// B - constraint matrix (M x N rectangular matrix)
   /// D - diagonal matrix of penalty values (M x M square matrix)
   void Initialize(HypreParMatrix& A, HypreParMatrix& B, HypreParMatrix& D);

   /// Build preconditioner for penalized system
   virtual Solver* BuildPreconditioner() const = 0;
   /// Select krylov solver for penalized system
   virtual IterativeSolver* BuildKrylov() const = 0;

   Vector penalty;
   Operator& constraintB;
   HypreParMatrix * penalized_mat;
   mutable IterativeSolver * krylov;
   mutable Solver * prec;
};


/** Uses CG and a HypreBoomerAMG preconditioner for the penalized system. */
class PenaltyPCGSolver : public PenaltyConstrainedSolver
{
public:
   PenaltyPCGSolver(HypreParMatrix& A, SparseMatrix& B, real_t penalty_,
                    int dimension=0, bool reorder=false) :
      PenaltyConstrainedSolver(A, B, penalty_),
      dimension_(dimension), reorder_(reorder)
   { }

   PenaltyPCGSolver(HypreParMatrix& A, HypreParMatrix& B, real_t penalty_,
                    int dimension=0, bool reorder=false) :
      PenaltyConstrainedSolver(A, B, penalty_),
      dimension_(dimension), reorder_(reorder)
   { }

   PenaltyPCGSolver(HypreParMatrix& A, HypreParMatrix& B, Vector& penalty_,
                    int dimension=0, bool reorder=false) :
      PenaltyConstrainedSolver(A, B, penalty_),
      dimension_(dimension), reorder_(reorder)
   { }

protected:
   Solver* BuildPreconditioner() const override
   {
      HypreBoomerAMG* h_prec = new HypreBoomerAMG(*penalized_mat);
      h_prec->SetPrintLevel(0);
      if (dimension_ > 0) { h_prec->SetSystemsOptions(dimension_, reorder_); }
      return h_prec;
   }

   IterativeSolver* BuildKrylov() const override
   { return new CGSolver(GetComm()); }

private:
   int dimension_;
   bool reorder_;
};

/** Uses GMRES and a HypreBoomerAMG preconditioner for the penalized system. */
class PenaltyGMRESSolver : public PenaltyConstrainedSolver
{
public:
   PenaltyGMRESSolver(HypreParMatrix& A, SparseMatrix& B, real_t penalty_,
                      int dimension=0, bool reorder=false) :
      PenaltyConstrainedSolver(A, B, penalty_),
      dimension_(dimension), reorder_(reorder)
   { }

   PenaltyGMRESSolver(HypreParMatrix& A, HypreParMatrix& B, real_t penalty_,
                      int dimension=0, bool reorder=false) :
      PenaltyConstrainedSolver(A, B, penalty_),
      dimension_(dimension), reorder_(reorder)
   { }

   PenaltyGMRESSolver(HypreParMatrix& A, HypreParMatrix& B, Vector& penalty_,
                      int dimension=0, bool reorder=false) :
      PenaltyConstrainedSolver(A, B, penalty_),
      dimension_(dimension), reorder_(reorder)
   { }

protected:
   Solver* BuildPreconditioner() const override
   {
      HypreBoomerAMG* h_prec = new HypreBoomerAMG(*penalized_mat);
      h_prec->SetPrintLevel(0);
      if (dimension_ > 0) { h_prec->SetSystemsOptions(dimension_, reorder_); }
      return h_prec;
   }

   IterativeSolver* BuildKrylov() const override
   { return new GMRESSolver(GetComm()); }

private:
   int dimension_;
   bool reorder_;
};

#endif

/** @brief Solve constrained system by solving original mixed system;
    see ConstrainedSolver.

    Solves the saddle-point problem with a block-diagonal preconditioner, with
    user-provided preconditioner in the top-left block and (by default) an
    identity matrix in the bottom-right.

    This is the most general ConstrainedSolver, needing only Operator objects
    to function. But in general it is not very efficient or scalable. */
class SchurConstrainedSolver : public ConstrainedSolver
{
public:
   /// Setup constrained system, with primal_pc a user-provided preconditioner
   /// for the top-left block.
#ifdef MFEM_USE_MPI
   SchurConstrainedSolver(MPI_Comm comm, Operator& A_, Operator& B_,
                          Solver& primal_pc_);
#endif
   SchurConstrainedSolver(Operator& A_, Operator& B_, Solver& primal_pc_);
   virtual ~SchurConstrainedSolver();

   void LagrangeSystemMult(const Vector& x, Vector& y) const override;

protected:
#ifdef MFEM_USE_MPI
   SchurConstrainedSolver(MPI_Comm comm, Operator& A_, Operator& B_);
#endif
   SchurConstrainedSolver(Operator& A_, Operator& B_);

   Array<int> offsets;
   BlockOperator * block_op;  // owned
   TransposeOperator * tr_B;  // owned
   Solver * primal_pc; // NOT owned
   BlockDiagonalPreconditioner * block_pc;  // owned
   Solver * dual_pc;  // owned

private:
   void Initialize();
};


#ifdef MFEM_USE_MPI
/** @brief Basic saddle-point solver with assembled blocks (ie, the
    operators are assembled HypreParMatrix objects.)

    This uses a block-diagonal preconditioner that approximates
    $ [ A^{-1} 0; 0 (B A^{-1} B^T)^{-1} ] $.

    In the top-left block, we approximate $ A^{-1} $ with HypreBoomerAMG.
    In the bottom-right, we approximate $ A^{-1} $ with the inverse of the
    diagonal of $ A $, assemble $ B diag(A)^{-1} B^T $, and use
    HypreBoomerAMG on that assembled matrix. */
class SchurConstrainedHypreSolver : public SchurConstrainedSolver
{
public:
   SchurConstrainedHypreSolver(MPI_Comm comm, HypreParMatrix& hA_,
                               HypreParMatrix& hB_, Solver * prec = nullptr,
                               int dimension=0, bool reorder=false);
   virtual ~SchurConstrainedHypreSolver();

private:
   HypreParMatrix& hA;
   HypreParMatrix& hB;
   HypreParMatrix * schur_mat;
};
#endif

/** @brief Build a matrix constraining normal components to zero.

    Given a vector space fespace, and the array constrained_att that
    includes the boundary *attributes* that are constrained to have normal
    component zero, this returns a SparseMatrix representing the
    constraints that need to be imposed.

    Each row of the returned matrix corresponds to a node that is
    constrained. The rows are arranged in (contiguous) blocks corresponding
    to a physical constraint; in 3D, a one-row constraint means the node
    is free to move along a plane, a two-row constraint means it is free
    to move along a line (e.g. the intersection of two normal-constrained
    planes), and a three-row constraint is fully constrained (equivalent
    to MFEM's usual essential boundary conditions).

    The constraint_rowstarts array is filled in to describe the structure of
    these constraints, so that (block) constraint k is encoded in rows
    constraint_rowstarts[k] to constraint_rowstarts[k + 1] - 1, inclusive,
    of the returned matrix.

    Constraints are imposed on "true" degrees of freedom, which are different
    in serial and parallel, so we need different numbering systems for the
    serial and parallel versions of this function.

    When two attributes intersect, this version will combine constraints,
    so in 2D the point at the intersection is fully constrained (ie,
    fixed in both directions). This is the wrong thing to do if the
    two boundaries are (close to) parallel at that point.

    @param[in] fespace              A vector finite element space
    @param[in] constrained_att      Boundary attributes to constrain
    @param[out] constraint_rowstarts  The rowstarts for separately
                                    eliminated constraints, possible
                                    input to EliminationCGSolver
    @param[in] parallel             Indicate that fespace is actually a
                                    ParFiniteElementSpace and the numbering
                                    in the returned matrix should be based
                                    on truedofs.

    @return a constraint matrix
*/
SparseMatrix * BuildNormalConstraints(FiniteElementSpace& fespace,
                                      Array<int>& constrained_att,
                                      Array<int>& constraint_rowstarts,
                                      bool parallel=false);

#ifdef MFEM_USE_MPI
/// Parallel wrapper for BuildNormalConstraints
SparseMatrix * ParBuildNormalConstraints(ParFiniteElementSpace& fespace,
                                         Array<int>& constrained_att,
                                         Array<int>& constraint_rowstarts);
#endif

}

#endif
