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

#ifndef MFEM_SOLVERS
#define MFEM_SOLVERS

#include "../config/config.hpp"
#include "operator.hpp"

#ifdef MFEM_USE_MPI
#include <mpi.h>
#endif

#ifdef MFEM_USE_SUITESPARSE
#include "sparsemat.hpp"
#include <umfpack.h>
#include <klu.h>
#endif

namespace mfem
{

/// Abstract base class for iterative solver
class IterativeSolver : public Solver
{
#ifdef MFEM_USE_MPI
private:
   int dot_prod_type; // 0 - local, 1 - global over 'comm'
   MPI_Comm comm;
#endif

protected:
   const Operator *oper;
   Solver *prec;

   int max_iter, print_level;
   double rel_tol, abs_tol;

   // stats
   mutable int final_iter, converged;
   mutable double final_norm;

   double Dot(const Vector &x, const Vector &y) const;
   double Norm(const Vector &x) const { return sqrt(Dot(x, x)); }

public:
   IterativeSolver();

#ifdef MFEM_USE_MPI
   IterativeSolver(MPI_Comm _comm);
#endif

   void SetRelTol(double rtol) { rel_tol = rtol; }
   void SetAbsTol(double atol) { abs_tol = atol; }
   void SetMaxIter(int max_it) { max_iter = max_it; }
   void SetPrintLevel(int print_lvl);

   int GetNumIterations() const { return final_iter; }
   int GetConverged() const { return converged; }
   double GetFinalNorm() const { return final_norm; }

   /// This should be called before SetOperator
   virtual void SetPreconditioner(Solver &pr);

   /// Also calls SetOperator for the preconditioner if there is one
   virtual void SetOperator(const Operator &op);
};


/// Stationary linear iteration: x <- x + B (b - A x)
class SLISolver : public IterativeSolver
{
protected:
   mutable Vector r, z;

   void UpdateVectors();

public:
   SLISolver() { }

#ifdef MFEM_USE_MPI
   SLISolver(MPI_Comm _comm) : IterativeSolver(_comm) { }
#endif

   virtual void SetOperator(const Operator &op)
   { IterativeSolver::SetOperator(op); UpdateVectors(); }

   virtual void Mult(const Vector &b, Vector &x) const;
};

/// Stationary linear iteration. (tolerances are squared)
void SLI(const Operator &A, const Vector &b, Vector &x,
         int print_iter = 0, int max_num_iter = 1000,
         double RTOLERANCE = 1e-12, double ATOLERANCE = 1e-24);

/// Preconditioned stationary linear iteration. (tolerances are squared)
void SLI(const Operator &A, Solver &B, const Vector &b, Vector &x,
         int print_iter = 0, int max_num_iter = 1000,
         double RTOLERANCE = 1e-12, double ATOLERANCE = 1e-24);


/// Conjugate gradient method
class CGSolver : public IterativeSolver
{
protected:
   mutable Vector r, d, z;

   void UpdateVectors();

public:
   CGSolver() { }

#ifdef MFEM_USE_MPI
   CGSolver(MPI_Comm _comm) : IterativeSolver(_comm) { }
#endif

   virtual void SetOperator(const Operator &op)
   { IterativeSolver::SetOperator(op); UpdateVectors(); }

   virtual void Mult(const Vector &b, Vector &x) const;
};

/// Conjugate gradient method. (tolerances are squared)
void CG(const Operator &A, const Vector &b, Vector &x,
        int print_iter = 0, int max_num_iter = 1000,
        double RTOLERANCE = 1e-12, double ATOLERANCE = 1e-24);

/// Preconditioned conjugate gradient method. (tolerances are squared)
void PCG(const Operator &A, Solver &B, const Vector &b, Vector &x,
         int print_iter = 0, int max_num_iter = 1000,
         double RTOLERANCE = 1e-12, double ATOLERANCE = 1e-24);


/// GMRES method
class GMRESSolver : public IterativeSolver
{
protected:
   int m; // see SetKDim()

public:
   GMRESSolver() { m = 50; }

#ifdef MFEM_USE_MPI
   GMRESSolver(MPI_Comm _comm) : IterativeSolver(_comm) { m = 50; }
#endif

   /// Set the number of iteration to perform between restarts, default is 50.
   void SetKDim(int dim) { m = dim; }

   virtual void Mult(const Vector &b, Vector &x) const;
};

/// FGMRES method
class FGMRESSolver : public IterativeSolver
{
protected:
   int m;

public:
   FGMRESSolver() { m = 50; }

#ifdef MFEM_USE_MPI
   FGMRESSolver(MPI_Comm _comm) : IterativeSolver(_comm) { m = 50; }
#endif

   void SetKDim(int dim) { m = dim; }

   virtual void Mult(const Vector &b, Vector &x) const;
};

/// GMRES method. (tolerances are squared)
int GMRES(const Operator &A, Vector &x, const Vector &b, Solver &M,
          int &max_iter, int m, double &tol, double atol, int printit);

/// GMRES method. (tolerances are squared)
void GMRES(const Operator &A, Solver &B, const Vector &b, Vector &x,
           int print_iter = 0, int max_num_iter = 1000, int m = 50,
           double rtol = 1e-12, double atol = 1e-24);


/// BiCGSTAB method
class BiCGSTABSolver : public IterativeSolver
{
protected:
   mutable Vector p, phat, s, shat, t, v, r, rtilde;

   void UpdateVectors();

public:
   BiCGSTABSolver() { }

#ifdef MFEM_USE_MPI
   BiCGSTABSolver(MPI_Comm _comm) : IterativeSolver(_comm) { }
#endif

   virtual void SetOperator(const Operator &op)
   { IterativeSolver::SetOperator(op); UpdateVectors(); }

   virtual void Mult(const Vector &b, Vector &x) const;
};

/// BiCGSTAB method. (tolerances are squared)
int BiCGSTAB(const Operator &A, Vector &x, const Vector &b, Solver &M,
             int &max_iter, double &tol, double atol, int printit);

/// BiCGSTAB method. (tolerances are squared)
void BiCGSTAB(const Operator &A, Solver &B, const Vector &b, Vector &x,
              int print_iter = 0, int max_num_iter = 1000,
              double rtol = 1e-12, double atol = 1e-24);


/// MINRES method
class MINRESSolver : public IterativeSolver
{
protected:
   mutable Vector v0, v1, w0, w1, q;
   mutable Vector u1; // used in the preconditioned version

public:
   MINRESSolver() { }

#ifdef MFEM_USE_MPI
   MINRESSolver(MPI_Comm _comm) : IterativeSolver(_comm) { }
#endif

   virtual void SetPreconditioner(Solver &pr)
   {
      IterativeSolver::SetPreconditioner(pr);
      if (oper) { u1.SetSize(width); }
   }

   virtual void SetOperator(const Operator &op);

   virtual void Mult(const Vector &b, Vector &x) const;
};

/// MINRES method without preconditioner. (tolerances are squared)
void MINRES(const Operator &A, const Vector &b, Vector &x, int print_it = 0,
            int max_it = 1000, double rtol = 1e-12, double atol = 1e-24);

/// MINRES method with preconditioner. (tolerances are squared)
void MINRES(const Operator &A, Solver &B, const Vector &b, Vector &x,
            int print_it = 0, int max_it = 1000,
            double rtol = 1e-12, double atol = 1e-24);


/// Newton's method for solving F(x)=b for a given operator F.
/** The method GetGradient() must be implemented for the operator F.
    The preconditioner is used (in non-iterative mode) to evaluate
    the action of the inverse gradient of the operator. */
class NewtonSolver : public IterativeSolver
{
protected:
   mutable Vector r, c;

public:
   NewtonSolver() { }

#ifdef MFEM_USE_MPI
   NewtonSolver(MPI_Comm _comm) : IterativeSolver(_comm) { }
#endif
   virtual void SetOperator(const Operator &op);

   /// Set the linear solver for inverting the Jacobian.
   /** This method is equivalent to calling SetPreconditioner(). */
   virtual void SetSolver(Solver &solver) { prec = &solver; }

   /// Solve the nonlinear system with right-hand side @a b.
   /** If `b.Size() != Height()`, then @a b is assumed to be zero. */
   virtual void Mult(const Vector &b, Vector &x) const;

   /** @brief This method can be overloaded in derived classes to implement line
       search algorithms. */
   /** The base class implementation (NewtonSolver) simply returns 1. A return
       value of 0 indicates a failure, interrupting the Newton iteration. */
   virtual double ComputeScalingFactor(const Vector &x, const Vector &b) const
   { return 1.0; }

   /** @brief This method can be overloaded in derived classes to perform
       computations that need knowledge of the newest Newton state. */
   virtual void ProcessNewState(const Vector &x) const { }
};

/** Adaptive restarted GMRES.
    m_max and m_min(=1) are the maximal and minimal restart parameters.
    m_step(=1) is the step to use for going from m_max and m_min.
    cf(=0.4) is a desired convergence factor. */
int aGMRES(const Operator &A, Vector &x, const Vector &b,
           const Operator &M, int &max_iter,
           int m_max, int m_min, int m_step, double cf,
           double &tol, double &atol, int printit);


/** Defines operators and constraints for the following optimization problem:
 *
 *    Find x that minimizes the objective function F(x), subject to
 *    C(x) = c_e,
 *    d_lo <= D(x) <= d_hi,
 *    x_lo <= x <= x_hi.
 *
 *  The operators F, C, D must take input of the same size (same width).
 *  Gradients of F, C, D might be needed, depending on the OptimizationSolver.
 *  When used with Hiop, gradients of C and D must be DenseMatrices.
 *  F always returns a scalar value, see CalcObjective(), CalcObjectiveGrad().
 *  C and D can have arbitrary heights.
 *  C and D can be NULL, meaning that their constraints are not used.
 *
 *  When used in parallel, all Vectors are assumed to be true dof vectors, and
 *  the operators are expected to be defined for tdof vectors. */
class OptimizationProblem
{
protected:
   /// Not owned, some can remain unused (NULL).
   const Operator *C, *D;
   const Vector *c_e, *d_lo, *d_hi, *x_lo, *x_hi;

public:
   const int input_size;

   /// In parallel, insize is the number of the local true dofs.
   OptimizationProblem(int insize, const Operator *C_, const Operator *D_);

   /// Objective F(x). In parallel, the result should be reduced over tasks.
   virtual double CalcObjective(const Vector &x) const = 0;
   /// The result grad is expected to enter with the correct size.
   virtual void CalcObjectiveGrad(const Vector &x, Vector &grad) const
   { MFEM_ABORT("The objective gradient is not implemented."); }

   void SetEqualityConstraint(const Vector &c);
   void SetInequalityConstraint(const Vector &dl, const Vector &dh);
   void SetSolutionBounds(const Vector &xl, const Vector &xh);

   const Operator *GetC() const { return C; }
   const Operator *GetD() const { return D; }
   const Vector *GetEqualityVec() const { return c_e; }
   const Vector *GetInequalityVec_Lo() const { return d_lo; }
   const Vector *GetInequalityVec_Hi() const { return d_hi; }
   const Vector *GetBoundsVec_Lo() const { return x_lo; }
   const Vector *GetBoundsVec_Hi() const { return x_hi; }

   int GetNumConstraints() const;
};

/// Abstract solver for OptimizationProblems.
class OptimizationSolver : public IterativeSolver
{
protected:
   const OptimizationProblem *problem;

public:
   OptimizationSolver(): IterativeSolver(), problem(NULL) { }
#ifdef MFEM_USE_MPI
   OptimizationSolver(MPI_Comm _comm): IterativeSolver(_comm), problem(NULL) { }
#endif
   virtual ~OptimizationSolver() { }

   /** This function is virtual as solvers might need to perform some initial
    *  actions (e.g. validation) with the OptimizationProblem. */
   virtual void SetOptimizationProblem(const OptimizationProblem &prob)
   { problem = &prob; }

   virtual void Mult(const Vector &xt, Vector &x) const = 0;

   virtual void SetPreconditioner(Solver &pr)
   { MFEM_ABORT("Not meaningful for this solver."); }
   virtual void SetOperator(const Operator &op)
   { MFEM_ABORT("Not meaningful for this solver."); }
};

/** SLBQP optimizer:
 *  (S)ingle (L)inearly Constrained with (B)ounds (Q)uadratic (P)rogram
 *
 *    Minimize || x-x_t ||, subject to
 *    sum w_i x_i = a,
 *    x_lo <= x <= x_hi.
 */
class SLBQPOptimizer : public OptimizationSolver
{
protected:
   Vector lo, hi, w;
   double a;

   /// Solve QP at fixed lambda
   inline double solve(double l, const Vector &xt, Vector &x, int &nclip) const
   {
      add(xt, l, w, x);
      if (problem == NULL) { x.median(lo,hi); }
      else
      {
         x.median(*problem->GetBoundsVec_Lo(),
                  *problem->GetBoundsVec_Hi());
      }
      nclip++;
      if (problem == NULL) { return Dot(w, x) - a; }
      else
      {
         Vector c(1);
         // Includes parallel communication.
         problem->GetC()->Mult(x, c);

         return c(0) - (*problem->GetEqualityVec())(0);
      }
   }

   inline void print_iteration(int it, double r, double l) const;

public:
   SLBQPOptimizer() { }

#ifdef MFEM_USE_MPI
   SLBQPOptimizer(MPI_Comm _comm) : OptimizationSolver(_comm) { }
#endif

   /** Setting an OptimizationProblem will overwrite the Vectors given by
    *  SetBounds and SetLinearConstraint. The objective function remains
    *  unchanged. */
   virtual void SetOptimizationProblem(const OptimizationProblem &prob);

   void SetBounds(const Vector &_lo, const Vector &_hi);
   void SetLinearConstraint(const Vector &_w, double _a);

   /** We let the target values play the role of the initial vector xt, from
    *  which the operator generates the optimal vector x. */
   virtual void Mult(const Vector &xt, Vector &x) const;
};


#ifdef MFEM_USE_SUITESPARSE

/// Direct sparse solver using UMFPACK
class UMFPackSolver : public Solver
{
protected:
   bool use_long_ints;
   SparseMatrix *mat;
   void *Numeric;
   SuiteSparse_long *AI, *AJ;

   void Init();

public:
   double Control[UMFPACK_CONTROL];
   mutable double Info[UMFPACK_INFO];

   /** @brief For larger matrices, if the solver fails, set the parameter @a
       _use_long_ints = true. */
   UMFPackSolver(bool _use_long_ints = false)
      : use_long_ints(_use_long_ints) { Init(); }
   /** @brief Factorize the given SparseMatrix using the defaults. For larger
       matrices, if the solver fails, set the parameter @a _use_long_ints =
       true. */
   UMFPackSolver(SparseMatrix &A, bool _use_long_ints = false)
      : use_long_ints(_use_long_ints) { Init(); SetOperator(A); }

   /** @brief Factorize the given Operator @a op which must be a SparseMatrix.

       The factorization uses the parameters set in the #Control data member.
       @note This method calls SparseMatrix::SortColumnIndices() with @a op,
       modifying the matrix if the column indices are not already sorted. */
   virtual void SetOperator(const Operator &op);

   /// Set the print level field in the #Control data member.
   void SetPrintLevel(int print_lvl) { Control[UMFPACK_PRL] = print_lvl; }

   virtual void Mult(const Vector &b, Vector &x) const;
   virtual void MultTranspose(const Vector &b, Vector &x) const;

   virtual ~UMFPackSolver();
};

/// Direct sparse solver using KLU
class KLUSolver : public Solver
{
protected:
   SparseMatrix *mat;
   klu_symbolic *Symbolic;
   klu_numeric *Numeric;

   void Init();

public:
   KLUSolver()
      : mat(0),Symbolic(0),Numeric(0)
   { Init(); }
   KLUSolver(SparseMatrix &A)
      : mat(0),Symbolic(0),Numeric(0)
   { Init(); SetOperator(A); }

   // Works on sparse matrices only; calls SparseMatrix::SortColumnIndices().
   virtual void SetOperator(const Operator &op);

   virtual void Mult(const Vector &b, Vector &x) const;
   virtual void MultTranspose(const Vector &b, Vector &x) const;

   virtual ~KLUSolver();

   mutable klu_common Common;
};

#endif // MFEM_USE_SUITESPARSE

}

#endif // MFEM_SOLVERS
