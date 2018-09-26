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

#ifndef MFEM_HIOP
#define MFEM_HIOP

#include "linalg.hpp"
#include "../config/config.hpp"
#include "../general/globals.hpp"

#ifdef MFEM_USE_MPI
#include "operator.hpp"
#endif

#ifdef MFEM_USE_HIOP

#include "hiopInterface.hpp"
#include "hiopNlpFormulation.hpp"

namespace mfem
{

/**  Adapts the OptimizationProblem class to HIOP's interface.
 */
class HiopOptimizationProblem : public hiop::hiopInterfaceDenseConstraints
{
private:
   // Initial guess.
   const Vector *x_start;

protected:
   // Problem info.
   // Local and global number of variables and constraints.
   long long n_loc, n_glob, m_total;
   OptimizationProblem &problem; // TODO make it private after removing _simple.

   Vector constr_vals;
   DenseMatrix constr_grads;
   bool constr_info_is_current;
   void UpdateConstrValsGrads(const Vector x);

public:
   HiopOptimizationProblem(OptimizationProblem &prob)
      : x_start(NULL),
        problem(prob),
        n_loc(prob.input_size), n_glob(n_loc),
        m_total(prob.GetNumConstraints()),
        constr_vals(m_total), constr_grads(), constr_info_is_current(false),
        a_(0.)
  { 
#ifdef MFEM_USE_MPI
    //in case HiOp with MPI support is called by a serial driver.
    comm_ = MPI_COMM_WORLD;
#endif
  }

#ifdef MFEM_USE_MPI
   HiopOptimizationProblem(const MPI_Comm& _comm, OptimizationProblem &prob)
      : x_start(NULL), comm_(_comm),
        problem(prob),
        n_loc(prob.input_size), n_glob(0),
        m_total(prob.GetNumConstraints()),
        constr_vals(m_total), constr_info_is_current(false),
        a_(0.)
   {
      MPI_Allreduce(&n_loc, &n_glob, 1, MPI_LONG_LONG_INT, MPI_SUM, comm_);
   }
#endif

   /** f = 1/2 x^T A x + c^T x */
   virtual void setObjectiveFunction(const DenseMatrix &_A, const Vector &_c)
   {
      setObjectiveFunction(_A);
      setObjectiveFunction(_c);
   }
   virtual void setObjectiveFunction(const DenseMatrix &_A)
   {
      A_ = _A;
   }
   virtual void setObjectiveFunction(const Vector &_c)  { c_ = _c; }

   void setStartingPoint(const Vector &x0) { x_start = &x0; }

   /** Extraction of problem dimensions:
    *  n is the number of variables, m is the number of constraints. */
   virtual bool get_prob_sizes(long long int& n, long long int& m);

   /** Provide an primal starting point. This point is subject to adjustments
    *  internally in hiOP. */
   virtual bool get_starting_point(const long long &n, double *x0);

   virtual bool get_vars_info(const long long& n, double *xlow, double* xupp,
                              NonlinearityType* type);

   /** bounds on the constraints
    *  (clow<=-1e20 means no lower bound, cupp>=1e20 means no upper bound) */
   virtual bool get_cons_info(const long long& m, double* clow, double* cupp,
                              NonlinearityType* type);

   /** Objective function evaluation.
    *  Each rank returns the global objective value. */
   virtual bool eval_f(const long long& n, const double* x, bool new_x,
                       double& obj_value);

   /** Gradient of the objective function (local chunk). */
   virtual bool eval_grad_f(const long long &n, const double *x, bool new_x,
                            double *gradf);

   /** Evaluates a subset of the constraints cons(x). The subset is of size
    *  num_cons and is described by indexes in the idx_cons array,
    *  i.e. cons[c] = C(x)[idx_cons[c]] where c = 0 .. num_cons-1.
    *  The methods may be called multiple times, each time for a subset of the
    *  constraints, for example, for the subset containing the equalities and
    *  for the subset containing the inequalities. However, each constraint will
    *  be inquired EXACTLY once. This is done for performance considerations,
    *  to avoid temporary holders and memory copying.
    *
    *  Parameters:
    *   - n, m: the global number of variables and constraints
    *   - num_cons, idx_cons (array of size num_cons): the number and indexes of
    *     constraints to be evaluated
    *   - x: the point where the constraints are to be evaluated
    *   - new_x: whether x has been changed from the previous call to f, grad_f,
    *     or Jac
    *   - cons: array of size num_cons containing the value of the  constraints
    *     indicated by idx_cons
    *
    *  When MPI enabled, every rank populates cons, since the constraints are
    *  not distributed.
    */
   virtual bool eval_cons(const long long &n, const long long &m,
                          const long long &num_cons, const long long *idx_cons,
                          const double *x, bool new_x, double *cons);

   /** Evaluates the Jacobian of the subset of constraints indicated by
    *  idx_cons. The idx_cons is assumed to be of size num_cons.
    *  Example: if cons[c] = C(x)[idx_cons[c]] where c = 0 .. num_cons-1., then
    *  one needs to do Jac[c][j] = d cons[c] / dx_j, j = 1 .. n_loc.
    *
    *  Parameters: see eval_cons().
    *
    *  When MPI enabled, each rank computes only the local columns of the
    *  Jacobian, that is the partials with respect to local variables.
    */
   virtual bool eval_Jac_cons(const long long &n, const long long &m,
                              const long long &num_cons,
                              const long long *idx_cons,
                              const double *x, bool new_x, double **Jac);

   /**  column partitioning specification for distributed memory vectors
    *  Process P owns cols[P], cols[P]+1, ..., cols[P+1]-1, P={0,1,...,NumRanks}.
    *  Example: for a vector x of 6 elements on 3 ranks, the col partitioning is cols=[0,2,4,6].
    *  The caller manages memory associated with 'cols', array of size NumRanks+1
    */
   virtual bool get_vecdistrib_info(long long global_n, long long* cols)
   {
#ifdef MFEM_USE_MPI
      int nranks;
      int ierr = MPI_Comm_size(comm_, &nranks);
      MFEM_ASSERT(ierr==MPI_SUCCESS, "MPI_Comm_size failed with error" << ierr);

      long long* sizes = new long long[nranks];
      ierr = MPI_Allgather(&n_loc, 1, MPI_LONG_LONG_INT, sizes, 1, MPI_LONG_LONG_INT, comm_);
      MFEM_ASSERT(MPI_SUCCESS==ierr,
		"Error in MPI_Allgather of number of decision variables." << ierr);

      //compute global indeces
      cols[0]=0;
      for (int r=1; r<=nranks; r++) {
         cols[r] = sizes[r-1] + cols[r-1];
      }

      delete[] sizes;
      return true;
#else
      return false; //hiop runs in non-distributed mode 
#endif    
   }

#ifdef MFEM_USE_MPI
   virtual bool get_MPI_comm(MPI_Comm& comm_out) 
   { 
      comm_out=comm_; 
      return true;
   }
#endif

   virtual void setLinearConstraint(const Vector &_w, const double& _a)
   {
      a_ = _a;
   }

protected:
#ifdef MFEM_USE_MPI
   MPI_Comm comm_;
#endif

   //Objective function: f = 1/2 x^T A x + c^T x
   DenseMatrix A_;
   Vector c_;

   double a_;      //linear constraint rhs
};

/** Adapts the HIOP functionality to the MFEM OptimizationSolver interface.
 */
class HiopNlpOptimizer : public OptimizationSolver
{
protected:
   HiopOptimizationProblem* optProb_;

#ifdef MFEM_USE_MPI
   MPI_Comm comm_;
#endif

   virtual void allocHiopProbSpec(const long long& numvars);

public:
   HiopNlpOptimizer(); 
#ifdef MFEM_USE_MPI
   HiopNlpOptimizer(MPI_Comm _comm);
#endif
   virtual ~HiopNlpOptimizer();

   virtual void SetBounds(const Vector &_lo, const Vector &_hi);
   virtual void SetLinearConstraint(const Vector &_w, double _a);
   virtual void SetObjectiveFunction(const DenseMatrix &_A, const Vector &_c);
   virtual void SetObjectiveFunction(const DenseMatrix &_A);
   virtual void SetObjectiveFunction(const Vector &_c);

   virtual void SetOptimizationProblem(OptimizationProblem &prob);

   /** When iterative_mode is true, xt plays the role of an initial guess. */
   virtual void Mult(const Vector &xt, Vector &x) const;
};

} // mfem namespace

#endif //MFEM_USE_HIOP
#endif //MFEM_HIOP guard
