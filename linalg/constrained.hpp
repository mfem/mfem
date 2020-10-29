// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#ifndef MFEM_CONSTRAINED
#define MFEM_CONSTRAINED

#include "solvers.hpp"
#include "blockoperator.hpp"
#include "sparsemat.hpp"
#include "hypre.hpp"

namespace mfem
{

/** @brief An abstract class to solve the constrained system \f$ Ax = f \f$
    subject to the constraint \f$ B x = r \f$.

    Although implementations may not use the below formulation, for
    understanding some of its methods and notation you can think of
    it as solving the saddle-point system

     (  A   B^T  )  ( x )         (  f  )
     (  B        )  ( lambda)  =  (  r  )

    Not to be confused with ConstrainedOperator, which is totally
    different.

    The only point for this object is to unify handling of the "dual" rhs
    r and the dual solution \f$ \labmda \f$.
 */
class ConstrainedSolver : public IterativeSolver
{
public:
   ConstrainedSystem(Operator& A_, Operator& B_);
   virtual ~ConstrainedSolver();

   virtual void SetOperator(const Operator& op) { }

   /** @brief Set the right-hand side r for the constraint B x = r

       (r defaults to zero if you don't call this)
   */
   virtual void SetDualRHS(const Vector& r);

   /** @brief Return the Lagrange multiplier solution in lambda

       Does not make sense unless you've already solved the constrained
       system with Mult() */
   void GetDualSolution(Vector& lambda) const { lambda = dual_sol; }

   /** @brief Solve for x given f

       Implementations must implement either Mult() or SaddleMult() */
   virtual void Mult(const Vector& f, Vector& x);

protected:
   /** @brief Larger, saddle-point sized system solve */
   virtual void SaddleMult(const Vector& fr, Vector& xlambda) const
   {
      mfem_error("Not Implemnted!");
   }

   Operator& A;
   Operator& B;

   Vector dual_rhs;
   mutable Vector dual_sol;
   mutable Vector workb;
   mutable Vector workx;
};


/** @brief Connects eliminated dofs to non-eliminated dofs for EliminationCGSolver

    The action of this is (for a certain ordering) \f$ [ I ; -B_s^{-1} B_p ] \f$
    where \f$ B_s \f$ is the part of B (the constraint matrix) corresponding
    to secondary degrees of freedom, and \f$ B_p \f$ is the remainder of the
    constraint matrix.

    This is P in the EliminationCGSolver algorithm

    Height is number of total displacements, Width is smaller, with some
    displacements eliminated via constraints. */
class EliminationProjection : public Operator
{
public:
   /**
      Lots to think about in this interface, but I think a cleaner version
      takes just the jac. Actually, what I want is an object that creates
      both this and the approximate version, using only jac.

      rectangular B_1 = B_p has lagrange_dofs rows, primary_contact_dofs columns
      square B_2 = B_s has lagrange_dofs rows, secondary_contact_dofs columns

      B_p maps primary displacements into lagrange space
      B_s maps secondary displacements into lagrange space
      B_s^T maps lagrange space to secondary displacements (*)
      B_s^{-1} maps lagrange space into secondary displacements
      -B_s^{-1} B_p maps primary displacements to secondary displacements
   */
   EliminationProjection(SparseMatrix& A, SparseMatrix& B,
                         Array<int>& primary_contact_dofs,
                         Array<int>& secondary_contact_dofs);

   void Mult(const Vector& in, Vector& out) const;
   void MultTranspose(const Vector& in, Vector& out) const;

   /** @brief Assemble this projector as a SparseMatrix

       Some day we may also want to try approximate variants. */
   SparseMatrix * AssembleExact() const;

   void BuildGTilde(const Vector& g, Vector& gtilde) const;

   void RecoverPressure(const Vector& disprhs,
                        const Vector& disp, Vector& pressure) const;

private:
   SparseMatrix& A_;
   SparseMatrix& B_;

   Array<int>& primary_contact_dofs_;
   Array<int>& secondary_contact_dofs_;

   DenseMatrix Bp_;
   DenseMatrix Bs_;  // gets inverted in place
   LUFactors Bsinverse_;
   /// @todo there is probably a better way to handle the B_s^{-T}
   DenseMatrix BsT_;   // gets inverted in place
   LUFactors BsTinverse_;
   Array<int> ipiv_;
   Array<int> ipivT_;
};

#ifdef MFEM_USE_MPI

/** @brief Solve constrained system by eliminating the constraint; see ConstrainedSolver

    Solves the system with the operator \f$ P^T A P \f$, where P is
    EliminationProjection.

    Currently does not work in parallel. */
class EliminationCGSolver : public ConstrainedSolver
{
public:
   EliminationCGSolver(SparseMatrix& A, SparseMatrix& B, int firstblocksize);

   /** @brief Constructor, with explicit splitting into primary/secondary dofs.

       The secondary_dofs are eliminated from the system in this algorithm,
       as they can be written in terms of the primary_dofs. */
   EliminationCGSolver(SparseMatrix& A, SparseMatrix& B, Array<int>& primary_dofs,
                       Array<int>& secondary_dofs);

   ~EliminationCGSolver();

   void Mult(const Vector& x, Vector& y) const;

private:
   /**
      This assumes the primary/secondary dofs are cleanly separated in
      the matrix, and the given index tells you where.

      We want to move away from this assumption, the first step
      is to separate its logic in this method.
   */
   void BuildSeparatedInterfaceDofs(int firstblocksize);

   void BuildPreconditioner();

   SparseMatrix& Asp_;
   SparseMatrix& Bsp_;
   Array<int> first_interface_dofs_;
   Array<int> second_interface_dofs_;
   EliminationProjection * projector_;
   HypreParMatrix * h_explicit_operator_;
   HypreBoomerAMG * prec_;
};

/** @brief Solve constrained system with penalty method; see ConstrainedSolver.

    Uses a HypreBoomerAMG preconditioner for the penalized system. Only
    approximates the solution, better approximation with higher penalty,
    but with higher penalty the preconditioner is less effective. */
class PenaltyConstrainedSolver : public ConstrainedSolver
{
public:
   PenaltyConstrainedSolver(HypreParMatrix& A, SparseMatrix& B, double penalty_);

   PenaltyConstrainedSolver(HypreParMatrix& A, HypreParMatrix& B, double penalty_);

   ~PenaltyConstrainedSolver();

   void Mult(const Vector& x, Vector& y) const;

private:
   void Initialize(HypreParMatrix& A, HypreParMatrix& B);

   double penalty;
   Operator& constraintB;
   HypreParMatrix * penalized_mat;
   HypreBoomerAMG * prec;
};

#endif

/** @brief Solve constrained system by solving original mixed sysetm;
    see ConstrainedSolver.

    Solves the saddle-point problem with a block-diagonal preconditioner, with
    user-provided preconditioner in the top-left block and an identity matrix
    in the bottom-right. */
class SchurConstrainedSolver : public ConstrainedSolver
{
public:
   SchurConstrainedSolver(Operator& A_, Operator& B_,
                          Solver& primal_pc_);
   virtual ~SchurConstrainedSolver();

   void SetOperator(const Operator& op) { }

   void Mult(const Vector& x, Vector& y) const;

private:
   Array<int> offsets;
   BlockOperator * block_op;  // owned
   TransposeOperator * tr_B;  // owned
   Solver& primal_pc;
   BlockDiagonalPreconditioner * block_pc;  // owned
   Solver * dual_pc;  // owned
};


}

#endif
