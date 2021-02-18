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

#ifndef MFEM_CONSTRAINED
#define MFEM_CONSTRAINED

#include "solvers.hpp"
#include "blockoperator.hpp"
#include "sparsemat.hpp"
#include "hypre.hpp"

namespace mfem
{

class ParFiniteElementSpace;

/** @brief An abstract class to solve the constrained system \f$ Ax = f \f$
    subject to the constraint \f$ B x = r \f$.

    Although implementations may not use the below formulation, for
    understanding some of its methods and notation you can think of
    it as solving the saddle-point system

     (  A   B^T  )  ( x )          (  f  )
     (  B        )  ( lambda )  =  (  r  )

    Do not confuse with ConstrainedOperator, which despite the similar name
    has very different functionality.

    The height and width of this object as an IterativeSolver are for
    the above saddle point system, but one can use its PrimalMult() method
    to solve just \f$ Ax = f \f$ subject to a constraint with \f$ r \f$
    defaulting to zero or set via SetConstraintRHS().

    This abstract object unifies handling of the "dual" rhs \f$ r \f$
    and the Lagrange multiplier solution \f$ \lambda \f$, so that derived
    classses can reuse that code. */
class ConstrainedSolver : public IterativeSolver
{
public:
#ifdef MFEM_USE_MPI
   ConstrainedSolver(MPI_Comm comm, Operator& A_, Operator& B_);
#endif
   ConstrainedSolver(Operator& A_, Operator& B_);

   virtual ~ConstrainedSolver() { }

   virtual void SetOperator(const Operator& op) override { }

   /** @brief Set the right-hand side r for the constraint B x = r

       (r defaults to zero if you don't call this) */
   virtual void SetConstraintRHS(const Vector& r);

   /** @brief Return the Lagrange multiplier solution in lambda

       PrimalMult() only gives you x, this provides access to lambda

       Does not make sense unless you've already solved the constrained
       system with Mult() or PrimalMult() */
   void GetMultiplierSolution(Vector& lambda) const { lambda = multiplier_sol; }

   /** @brief Solve for \f$ x \f$ given \f$ f \f$.

       If you want to set \f$ r \f$, call SetConstraintRHS() before this.

       If you want to get \f$ \lambda \f$, call GetMultiplierSolution() after
       this.

       The base class implementation calls Mult(), so derived classes must
       implement either this or Mult() */
   virtual void PrimalMult(const Vector& f, Vector& x) const;

   /** @brief Solve for (x, lambda) given (f, r)

       The base class implementation calls PrimalMult(), so derived classes
       must implement either this or PrimalMult() */
   virtual void Mult(const Vector& f_and_r, Vector& x_and_lambda) const override;

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

    \f$ B_s^{-1} \f$ maps the lagrange space into secondary dofs, while
    \f$ -B_s^{-1} B_p \f$ maps primary dofs to secondary dofs. */
class Eliminator
{
public:
   Eliminator(const SparseMatrix& B, const Array<int>& lagrange_dofs,
              const Array<int>& primary_tdofs,
              const Array<int>& secondary_tdofs);

   const Array<int>& LagrangeDofs() const { return lagrange_tdofs; }
   const Array<int>& PrimaryDofs() const { return primary_tdofs; }
   const Array<int>& SecondaryDofs() const { return secondary_tdofs; }

   /// Given primary displacements, return secondary displacements
   /// This applies \f$ -B_s^{-1} B_p \f$.
   void Eliminate(const Vector& in, Vector& out) const;

   /// Transpose of Eliminate(), applies \f$ -B_p^T B_s^{-T} \f$
   void EliminateTranspose(const Vector& in, Vector& out) const;

   /// Maps Lagrange multipliers to secondary displacements,
   /// applies \f$ B_s^{-1} \f$
   void LagrangeSecondary(const Vector& in, Vector& out) const;

   /// Transpose of LagrangeSecondary()
   void LagrangeSecondaryTranspose(const Vector& in, Vector& out) const;

   /// Return \f$ -B_s^{-1} B_p \f$ explicitly assembled in mat
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

   /** Given Lagrange multiplier right-hand-side \f$ g \f$, return
       \f$ \tilde{g} \f$ */
   void BuildGTilde(const Vector& g, Vector& gtilde) const;

   /** After a solve, recover the Lagrange multiplier. */
   void RecoverMultiplier(const Vector& disprhs,
                          const Vector& disp, Vector& lm) const;

private:
   const Operator& Aop;
   Array<Eliminator*> eliminators;
};


#ifdef MFEM_USE_MPI
/** @brief Solve constrained system by eliminating the constraint; see
    ConstrainedSolver

    Solves the system with the operator \f$ P^T A P + Z_P \f$, where P is
    EliminationProjection and Z_P is the identity on the eliminated dofs. */
class EliminationSolver : public ConstrainedSolver
{
public:
   /** @brief Constructor, with explicit splitting into primary/secondary dofs.

       This constructor uses a single elimination block (per processor), which
       provides the most general algorithm but is also not scalable

       The secondary_dofs are eliminated from the system in this algorithm,
       as they can be written in terms of the primary_dofs. */
   EliminationSolver(HypreParMatrix& A, SparseMatrix& B,
                     Array<int>& primary_dofs,
                     Array<int>& secondary_dofs);

   /** @brief Constructor, elimination is by blocks.

       The nonzeros in B are assumed to be in disjoint rows and columns; the
       rows are identified with the constraint_rowstarts array, the secondary
       dofs are assumed to be the first nonzeros in the rows. */
   EliminationSolver(HypreParMatrix& A, SparseMatrix& B,
                     Array<int>& constraint_rowstarts);

   ~EliminationSolver();

   void PrimalMult(const Vector& x, Vector& y) const override;

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
   { BuildExplicitOperator(); }

protected:
   virtual Solver* BuildPreconditioner() const override
   {
      HypreBoomerAMG * h_prec = new HypreBoomerAMG(*h_explicit_operator);
      h_prec->SetPrintLevel(0);
      if (dimension > 0) { h_prec->SetSystemsOptions(dimension, reorder); }
      return h_prec;
   }

   virtual IterativeSolver* BuildKrylov() const override
   { return new CGSolver(GetComm()); }

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
                            double penalty_);

   PenaltyConstrainedSolver(HypreParMatrix& A, HypreParMatrix& B,
                            double penalty_);

   ~PenaltyConstrainedSolver();

   void PrimalMult(const Vector& x, Vector& y) const override;

protected:
   void Initialize(HypreParMatrix& A, HypreParMatrix& B);

   /// Build preconditioner for penalized system
   virtual Solver* BuildPreconditioner() const = 0;
   /// Select krylov solver for penalized system
   virtual IterativeSolver* BuildKrylov() const = 0;

   double penalty;
   Operator& constraintB;
   HypreParMatrix * penalized_mat;
   mutable Solver * prec;
};


/** Uses CG and a HypreBoomerAMG preconditioner for the penalized system. */
class PenaltyPCGSolver : public PenaltyConstrainedSolver
{
public:
   PenaltyPCGSolver(HypreParMatrix& A, SparseMatrix& B, double penalty_,
                    int dimension=0, bool reorder=false) :
      PenaltyConstrainedSolver(A, B, penalty_),
      dimension_(dimension), reorder_(reorder)
   { }

   PenaltyPCGSolver(HypreParMatrix& A, HypreParMatrix& B, double penalty_,
                    int dimension=0, bool reorder=false) :
      PenaltyConstrainedSolver(A, B, penalty_),
      dimension_(dimension), reorder_(reorder)
   { }

protected:
   virtual Solver* BuildPreconditioner() const override
   {
      HypreBoomerAMG* h_prec = new HypreBoomerAMG(*penalized_mat);
      h_prec->SetPrintLevel(0);
      if (dimension_ > 0) { h_prec->SetSystemsOptions(dimension_, reorder_); }
      return h_prec;
   }

   virtual IterativeSolver* BuildKrylov() const override
   { return new CGSolver(GetComm()); }

private:
   int dimension_;
   bool reorder_;
};

#endif

/** @brief Solve constrained system by solving original mixed sysetm;
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

   virtual void Mult(const Vector& x, Vector& y) const override;

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
    \f$ [ A^{-1} 0; 0 (B A^{-1} B^T)^{-1} ] \f$.

    In the top-left block, we approximate \f$ A^{-1} \f$ with HypreBoomerAMG.
    In the bottom-right, we approximate \f$ A^{-1} \f$ with the inverse of the
    diagonal of \f$ A \f$, assemble \f$ B D^{-1} B^T \f$, and use
    HypreBoomerAMG on that assembled matrix. */
class SchurConstrainedHypreSolver : public SchurConstrainedSolver
{
public:
   SchurConstrainedHypreSolver(MPI_Comm comm, HypreParMatrix& hA_,
                               HypreParMatrix& hB_, int dimension=0,
                               bool reorder=false);
   virtual ~SchurConstrainedHypreSolver();

private:
   HypreParMatrix& hA;
   HypreParMatrix& hB;
   HypreParMatrix * schur_mat;
};
#endif

}

#endif
