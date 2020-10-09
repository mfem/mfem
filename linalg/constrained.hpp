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

/**
   Take a vector with displacements and lagrange multiplier degrees of freedom
   (corresponding to pressures on the slave surface), eliminate the
   constraint, return vector of just displacements.

   This is P in the EliminationSolver algorithm

   Height is number of total displacements, Width is smaller, with some
   displacements eliminated via constraints.
*/
class EliminationProjection : public Operator
{
public:
   /**
      Lots to think about in this interface, but I think a cleaner version
      takes just the jac. Actually, what I want is an object that creates
      both this and the approximate version, using only jac.

      rectangular B_1 = B_m has lagrange_dofs rows, master_contact_dofs columns
      square B_2 = B_s has lagrange_dofs rows, slave_contact_dofs columns

      B_m maps master displacements into lagrange space
      B_s maps slave displacements into lagrange space
      B_s^T maps lagrange space to slave displacements (*)
      B_s^{-1} maps lagrange space into slave displacements
      -B_s^{-1} B_m maps master displacements to slave displacements
   */
   EliminationProjection(SparseMatrix& A, SparseMatrix& B,
                         Array<int>& master_contact_dofs,
                         Array<int>& slave_contact_dofs);

   void Mult(const Vector& in, Vector& out) const;
   void MultTranspose(const Vector& in, Vector& out) const;

   SparseMatrix * AssembleApproximate() const;

   void BuildGTilde(const Vector& g, Vector& gtilde) const;

   void RecoverPressure(const Vector& disprhs,
                        const Vector& disp, Vector& pressure) const;

private:
   SparseMatrix& A_;
   SparseMatrix& B_;

   Array<int>& master_contact_dofs_;
   Array<int>& slave_contact_dofs_;

   DenseMatrix Bm_;
   DenseMatrix Bs_;  // gets inverted in place
   LUFactors Bsinverse_;
   /// @todo there is probably a better way to handle the B_s^{-T}
   DenseMatrix BsT_;   // gets inverted in place
   LUFactors BsTinverse_;
   Array<int> ipiv_;
   Array<int> ipivT_;
};


class EliminationCGSolver : public IterativeSolver
{
public:
   EliminationCGSolver(SparseMatrix& A, SparseMatrix& B, int firstblocksize);

   EliminationCGSolver(SparseMatrix& A, SparseMatrix& B, Array<int>& master_dofs,
                       Array<int>& slave_dofs);

   ~EliminationCGSolver();

   void SetOperator(const Operator& op) { }

   void Mult(const Vector& x, Vector& y) const;

private:
   /**
      This assumes the master/slave dofs are cleanly separated in
      the matrix, and the given index tells you where.

      We want to move away from this assumption, the first step
      is to separate its logic in this method.
   */
   void BuildSeparatedInterfaceDofs(int firstblocksize);

   void BuildPreconditioner();

   SparseMatrix& A_;
   SparseMatrix& B_;
   Array<int> first_interface_dofs_;
   Array<int> second_interface_dofs_;
   EliminationProjection * projector_;
   HypreParMatrix * h_explicit_operator_;
   HypreBoomerAMG * prec_;
};


/**
   @todo test in parallel
*/
class PenaltyConstrainedSolver : public IterativeSolver
{
public:
   PenaltyConstrainedSolver(HypreParMatrix& A, SparseMatrix& B, double penalty_);

   PenaltyConstrainedSolver(HypreParMatrix& A, HypreParMatrix& B, double penalty_);

   ~PenaltyConstrainedSolver();

   void Mult(const Vector& x, Vector& y) const;

   void SetOperator(const Operator& op) { }

private:
   void Initialize(HypreParMatrix& A, HypreParMatrix& B);

   double penalty;
   Operator& constraintB;
   // SparseMatrix * penalized_mat;
   HypreParMatrix * penalized_mat;
   HypreBoomerAMG * prec;
};

/// @todo test in parallel
class SchurConstrainedSolver : public IterativeSolver
{
public:
   SchurConstrainedSolver(BlockOperator& block_op_,
                          Solver& primal_pc_);
   virtual ~SchurConstrainedSolver();

   void SetOperator(const Operator& op) { }

   void Mult(const Vector& x, Vector& y) const;

private:
   BlockOperator& block_op;
   Solver& primal_pc;
   BlockDiagonalPreconditioner block_pc;
   Solver * dual_pc;  // owned
};


/**
   A class to solve the constrained system

     A x = f

   subject to the constraint

     B x = r

   abstractly. Although this object may not use the below formulation,
   for understanding some of its methods and notation you can think of
   it as solving the saddle-point system

     (  A   B^T  )  ( x )         (  f  )
     (  B        )  ( lambda)  =  (  r  )

   Not to be confused with ConstrainedOperator, which is totally different.
*/
class ConstrainedSolver : public Solver
{
public:
   ConstrainedSolver(Operator& A, Operator& B);
   ~ConstrainedSolver();

   void SetOperator(const Operator& op) { }

   /** @brief Setup Schur complement solver for constrained system.

       @param prec Preconditioner for primal block.
   */
   void SetSchur(Solver& prec);

   /** @brief Set the right-hand side r for the constraint B x = r

       @todo this is not going to work for elimination?

       (r defaults to zero if you don't call this)
   */
   void SetDualRHS(const Vector& r);

   /** @brief Set up the elimination solver.

       The array secondary_dofs should contain as many entries as the rows
       in the constraint matrix B; The block of B corresponding to these
       columns will be inverted (or approximately inverted in a
       preconditioner, depending on options) to eliminate the constraint.

       @todo this can be done with only secondary_dofs given in interface
   */
   void SetElimination(Array<int>& primary_dofs,
                       Array<int>& secondary_dofs);

   /** @brief Set up a penalty solver. */
   void SetPenalty(double penalty);

   /** @brief Solve the constrained system.

       The notation follows the documentation
       for the class, the input vector f is for the primal
       part only; if you have a nonzero r, you need to set that with
       SetDualRHS(). Similarly, the output x is only for the primal system,
       while if you want the Lagrange multiplier solution you call
       GetDualSolution() after the solve. */
   void Mult(const Vector& f, Vector& x) const;

   /** @brief Return the Lagrange multiplier solution in lambda

       Does not make sense unless you've already solved the constrained
       system with Mult() */
   void GetDualSolution(Vector& lambda) { lambda = dual_sol; }

   void SetRelTol(double rtol) { subsolver->SetRelTol(rtol); }
   void SetAbsTol(double atol) { subsolver->SetAbsTol(atol); }
   void SetMaxIter(int max_it) { subsolver->SetMaxIter(max_it); }
   void SetPrintLevel(int pl) { subsolver->SetPrintLevel(pl); }

private:
   Array<int> offsets;
   BlockOperator * block_op;
   TransposeOperator * tr_B;

   IterativeSolver * subsolver;

   /// hack, @todo remove
   SparseMatrix hypre_diag;

   Vector dual_rhs;
   mutable Vector dual_sol;

   mutable Vector workb;
   mutable Vector workx;
};

}

#endif
