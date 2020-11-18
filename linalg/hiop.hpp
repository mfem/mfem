// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

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

/// Internal class - adapts the OptimizationProblem class to HiOp's interface.
class HiopOptimizationProblem : public hiop::hiopInterfaceDenseConstraints
{
private:

#ifdef MFEM_USE_MPI
   MPI_Comm comm_;
#endif

   // Problem info.
   const OptimizationProblem &problem;

   // Local and global number of variables and constraints.
   const long long ntdofs_loc, m_total;
   long long ntdofs_glob;

   // Initial guess.
   const Vector *x_start;

   Vector constr_vals;
   DenseMatrix constr_grads;
   bool constr_info_is_current;
   void UpdateConstrValsGrads(const Vector x);

public:
   HiopOptimizationProblem(const OptimizationProblem &prob)
      : problem(prob),
        ntdofs_loc(prob.input_size), m_total(prob.GetNumConstraints()),
        ntdofs_glob(ntdofs_loc),
        x_start(NULL),
        constr_vals(m_total), constr_grads(m_total, ntdofs_loc),
        constr_info_is_current(false)
   {
#ifdef MFEM_USE_MPI
      // Used when HiOp with MPI support is called by a serial driver.
      comm_ = MPI_COMM_WORLD;
#endif
   }

#ifdef MFEM_USE_MPI
   HiopOptimizationProblem(const MPI_Comm& _comm,
                           const OptimizationProblem &prob)
      : comm_(_comm),
        problem(prob),
        ntdofs_loc(prob.input_size), m_total(prob.GetNumConstraints()),
        ntdofs_glob(0),
        x_start(NULL),
        constr_vals(m_total), constr_grads(m_total, ntdofs_loc),
        constr_info_is_current(false)
   {
      MPI_Allreduce(&ntdofs_loc, &ntdofs_glob, 1, MPI_LONG_LONG_INT,
                    MPI_SUM, comm_);
   }
#endif

   void setStartingPoint(const Vector &x0) { x_start = &x0; }

   /** Extraction of problem dimensions:
    *  n is the number of variables, m is the number of constraints. */
   virtual bool get_prob_sizes(long long int& n, long long int& m);

   /** Provide an primal starting point. This point is subject to adjustments
    *  internally in HiOp. */
   virtual bool get_starting_point(const long long &n, double *x0);

   virtual bool get_vars_info(const long long& n, double *xlow, double* xupp,
                              NonlinearityType* type);

   /** bounds on the constraints
    *  (clow<=-1e20 means no lower bound, cupp>=1e20 means no upper bound) */
   virtual bool get_cons_info(const long long &m, double *clow, double *cupp,
                              NonlinearityType* type);

   /** Objective function evaluation.
    *  Each rank returns the global objective value. */
   virtual bool eval_f(const long long& n, const double *x, bool new_x,
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
    *  Example: if cons[c] = C(x)[idx_cons[c]] where c = 0 .. num_cons-1, then
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

   /** Specifies column partitioning for distributed memory vectors.
    *  Process p owns vector entries with indices cols[p] to cols[p+1]-1,
    *  where p = 0 .. nranks-1. The cols array is of size nranks + 1.
    *  Example: for a vector x of 6 entries (globally) on 3 ranks, the uniform
    *  column partitioning is cols=[0,2,4,6].
    */
   virtual bool get_vecdistrib_info(long long global_n, long long *cols);

#ifdef MFEM_USE_MPI
   virtual bool get_MPI_comm(MPI_Comm &comm_out)
   {
      comm_out = comm_;
      return true;
   }
#endif
};

/// Adapts the HiOp functionality to the MFEM OptimizationSolver interface.
class HiopNlpOptimizer : public OptimizationSolver
{
protected:
   HiopOptimizationProblem *hiop_problem;

#ifdef MFEM_USE_MPI
   MPI_Comm comm_;
#endif

public:
   HiopNlpOptimizer();
#ifdef MFEM_USE_MPI
   HiopNlpOptimizer(MPI_Comm _comm);
#endif
   virtual ~HiopNlpOptimizer();

   virtual void SetOptimizationProblem(const OptimizationProblem &prob);

   /// Solves the optimization problem with xt as initial guess.
   virtual void Mult(const Vector &xt, Vector &x) const;
};

} // mfem namespace

#endif //MFEM_USE_HIOP
#endif //MFEM_HIOP guard
