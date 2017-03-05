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

   template <class TVector>
   double Dot(const TVector &x, const TVector &y) const
   {
#ifndef MFEM_USE_MPI
     return (x * y);
#else
     if (dot_prod_type == 0)
     {
       return (x * y);
     }
     double local_dot = (x * y);
     double global_dot;

     MPI_Allreduce(&local_dot, &global_dot, 1, MPI_DOUBLE, MPI_SUM, comm);

     return global_dot;
#endif
   }

   template <class TVector>
   double Norm(const TVector &x) const { return sqrt(Dot(x, x)); }

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
template <class TVector>
class TCGSolver : public IterativeSolver
{
protected:
   mutable TVector r, d, z;

   void UpdateVectors()
   {
      r.SetSize(width);
      d.SetSize(width);
      z.SetSize(width);
   }

public:
   TCGSolver() { }

#ifdef MFEM_USE_MPI
   TCGSolver(MPI_Comm _comm) : IterativeSolver(_comm) { }
#endif

   virtual void SetOperator(const Operator &op)
   { IterativeSolver::SetOperator(op); UpdateVectors(); }

   virtual void Mult(const Vector &b, Vector &x) const
   { mfem_error("TCGSolver::Mult() Cannot be used, use Solve()"); }

   void Solve(const TVector &b, TVector &x) const
   {
      int i;
      double r0, den, nom, nom0, betanom, alpha, beta;

      if (iterative_mode)
      {
        oper->Mult(x, r);
        std::cout << "1. r.Min()     = " << r.Min() << '\n'
                  << "1. r.Max()     = " << r.Max() << '\n'
                  << "1. r.Norml2()  = " << r.Norml2() << '\n';
        subtract(b, r, r); // r = b - A x
        std::cout << "2. r.Min()     = " << r.Min() << '\n'
                  << "2. r.Max()     = " << r.Max() << '\n'
                  << "2. r.Norml2()  = " << r.Norml2() << '\n';
      }
      else
      {
         r = b;
         x = 0.0;
      }

      if (prec)
      {
         prec->Mult(r, z); // z = B r
         d = z;
      }
      else
      {
         d = r;
      }
      nom0 = nom = Dot(d, r);
      MFEM_ASSERT(IsFinite(nom), "nom = " << nom);
      std::cout << "3. nom         = " << nom << '\n';

      if (print_level == 1 || print_level == 3)
      {
         std::cout << "   Iteration : " << std::setw(3) << 0 << "  (B r, r) = "
                   << nom << (print_level == 3 ? " ...\n" : "\n");
      }

      r0 = std::max(nom*rel_tol*rel_tol, abs_tol*abs_tol);
      if (nom <= r0)
      {
         converged = 1;
         final_iter = 0;
         final_norm = sqrt(nom);
         return;
      }

      oper->Mult(d, z);  // z = A d
      std::cout << "4. z.Min()     = " << z.Min() << '\n'
                << "4. z.Max()     = " << z.Max() << '\n'
                << "4. z.Norml2()  = " << z.Norml2() << '\n';
      den = Dot(z, d);
      MFEM_ASSERT(IsFinite(den), "den = " << den);

      std::cout << "5. den         = " << den << '\n';

      if (print_level >= 0 && den < 0.0)
      {
         std::cout << "Negative denominator in step 0 of PCG: " << den << '\n';
      }

      if (den == 0.0)
      {
         converged = 0;
         final_iter = 0;
         final_norm = sqrt(nom);
         return;
      }

      // start iteration
      converged = 0;
      final_iter = max_iter;
      for (i = 1; true; )
      {
         alpha = nom/den;
         std::cout << "I1. alph       = " << alpha << '\n';
         add(x,  alpha, d, x);     //  x = x + alpha d
         std::cout << "I2. x.Min()    = " << x.Min() << '\n'
                   << "I2. x.Max()    = " << x.Max() << '\n'
                   << "I2. x.Norml2() = " << x.Norml2() << '\n';
         add(r, -alpha, z, r);     //  r = r - alpha A d
         std::cout << "I3. r.Min()    = " << r.Min() << '\n'
                   << "I3. r.Max()    = " << r.Max() << '\n'
                   << "I3. r.Norml2() = " << r.Norml2() << '\n';

         if (prec)
         {
            prec->Mult(r, z);      //  z = B r
            betanom = Dot(r, z);
         }
         else
         {
            betanom = Dot(r, r);
         }
         MFEM_ASSERT(IsFinite(betanom), "betanom = " << betanom);

         if (print_level == 1)
         {
            std::cout << "   Iteration : " << std::setw(3) << i << "  (B r, r) = "
                      << betanom << '\n';
         }

         if (betanom < r0)
         {
            if (print_level == 2)
            {
               std::cout << "Number of PCG iterations: " << i << '\n';
            }
            else if (print_level == 3)
            {
               std::cout << "   Iteration : " << std::setw(3) << i << "  (B r, r) = "
                         << betanom << '\n';
            }
            converged = 1;
            final_iter = i;
            break;
         }

         if (++i > max_iter)
         {
            break;
         }

         beta = betanom/nom;
         if (prec)
         {
            add(z, beta, d, d);   //  d = z + beta d
         }
         else
         {
            add(r, beta, d, d);
         }
         oper->Mult(d, z);       //  z = A d
         den = Dot(d, z);
         MFEM_ASSERT(IsFinite(den), "den = " << den);
         if (den <= 0.0)
         {
            if (print_level >= 0 && Dot(d, d) > 0.0)
               std::cout << "PCG: The operator is not positive definite. (Ad, d) = "
                         << den << '\n';
         }
         nom = betanom;
      }
      if (print_level >= 0 && !converged)
      {
         if (print_level != 1)
         {
            if (print_level != 3)
            {
               std::cout << "   Iteration : " << std::setw(3) << 0 << "  (B r, r) = "
                         << nom0 << " ...\n";
            }
            std::cout << "   Iteration : " << std::setw(3) << final_iter << "  (B r, r) = "
                      << betanom << '\n';
         }
         std::cout << "PCG: No convergence!" << '\n';
      }
      if (print_level >= 1 || (print_level >= 0 && !converged))
      {
         std::cout << "Average reduction factor = "
                   << pow (betanom/nom0, 0.5/final_iter) << '\n';
      }
      final_norm = sqrt(betanom);
   }
};

typedef TCGSolver<Vector> CGSolver;

/// Conjugate gradient method. (tolerances are squared)
template <class TVector>
void TCG(const Operator &A, const TVector &b, TVector &x,
        int print_iter = 0, int max_num_iter = 1000,
        double RTOLERANCE = 1e-12, double ATOLERANCE = 1e-24)
{
   TCGSolver<TVector> cg;
   cg.SetPrintLevel(print_iter);
   cg.SetMaxIter(max_num_iter);
   cg.SetRelTol(sqrt(RTOLERANCE));
   cg.SetAbsTol(sqrt(ATOLERANCE));
   cg.SetOperator(A);
   cg.Solve(b, x);
}

/// Preconditioned conjugate gradient method. (tolerances are squared)
template <class TVector>
void TPCG(const Operator &A, Solver &B, const TVector &b, TVector &x,
          int print_iter = 0, int max_num_iter = 1000,
          double RTOLERANCE = 1e-12, double ATOLERANCE = 1e-24)
{
   TCGSolver<TVector> pcg;
   pcg.SetPrintLevel(print_iter);
   pcg.SetMaxIter(max_num_iter);
   pcg.SetRelTol(sqrt(RTOLERANCE));
   pcg.SetAbsTol(sqrt(ATOLERANCE));
   pcg.SetOperator(A);
   pcg.SetPreconditioner(B);
   pcg.Solve(b, x);
}

inline void CG(const Operator &A, const Vector &b, Vector &x,
        int print_iter = 0, int max_num_iter = 1000,
        double RTOLERANCE = 1e-12, double ATOLERANCE = 1e-24)
{
  TCG<Vector>(A, b, x,
              print_iter, max_num_iter,
              RTOLERANCE, ATOLERANCE);
}

inline void PCG(const Operator &A, Solver &B, const Vector &b, Vector &x,
        int print_iter = 0, int max_num_iter = 1000,
        double RTOLERANCE = 1e-12, double ATOLERANCE = 1e-24)
{
  TPCG<Vector>(A, B, b, x,
              print_iter, max_num_iter,
              RTOLERANCE, ATOLERANCE);
}

/// GMRES method
class GMRESSolver : public IterativeSolver
{
protected:
   int m;

public:
   GMRESSolver() { m = 50; }

#ifdef MFEM_USE_MPI
   GMRESSolver(MPI_Comm _comm) : IterativeSolver(_comm) { m = 50; }
#endif

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
};


/** Adaptive restarted GMRES.
    m_max and m_min(=1) are the maximal and minimal restart parameters.
    m_step(=1) is the step to use for going from m_max and m_min.
    cf(=0.4) is a desired convergence factor. */
int aGMRES(const Operator &A, Vector &x, const Vector &b,
           const Operator &M, int &max_iter,
           int m_max, int m_min, int m_step, double cf,
           double &tol, double &atol, int printit);


/** SLBQP: (S)ingle (L)inearly Constrained with (B)ounds (Q)uadratic (P)rogram

    minimize 1/2 ||x - x_t||^2, subject to:
    lo_i <= x_i <= hi_i
    sum_i w_i x_i = a
*/
class SLBQPOptimizer : public IterativeSolver
{
protected:
   Vector lo, hi, w;
   double a;

   /// Solve QP at fixed lambda
   inline double solve(double l, const Vector &xt, Vector &x, int &nclip) const
   {
      add(xt, l, w, x);
      x.median(lo,hi);
      nclip++;
      return Dot(w,x)-a;
   }

   inline void print_iteration(int it, double r, double l) const;

public:
   SLBQPOptimizer() {}

#ifdef MFEM_USE_MPI
   SLBQPOptimizer(MPI_Comm _comm) : IterativeSolver(_comm) {}
#endif

   void SetBounds(const Vector &_lo, const Vector &_hi);
   void SetLinearConstraint(const Vector &_w, double _a);

   // For this problem type, we let the target values play the role of the
   // initial vector xt, from which the operator generates the optimal vector x.
   virtual void Mult(const Vector &xt, Vector &x) const;

   /// These are not currently meaningful for this solver and will error out.
   virtual void SetPreconditioner(Solver &pr);
   virtual void SetOperator(const Operator &op);
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
