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
//#include "../config/config.hpp"

#ifdef MFEM_USE_MPI
#include "operator.hpp"
#endif

#ifdef MFEM_USE_HIOP

#include "hiopInterface.hpp"
#include "hiopNlpFormulation.hpp"

namespace mfem
{

class HiopProblemSpec : public hiop::hiopInterfaceDenseConstraints
{

public:
  HiopProblemSpec(const MPI_Comm& _comm, const long long& _n) 
    : comm_(_comm), n_(_n), lo_(NULL), hi_(NULL), conbody_(NULL), conrhs_(0.) {}

  /** problem dimensions: n number of variables, m number of constraints */
  virtual bool get_prob_sizes(long long int& n, long long int& m) {
    n = n_;
    m = 1;
    return true;
  };

  virtual bool get_vars_info(const long long& n, double *xlow, double* xupp, NonlinearityType* type) {
    assert(n==n_);
    //!mfem memcpy
    memcpy(xlow, lo_, n_local_*sizeof(double));
    memcpy(xupp, hi_, n_local_*sizeof(double));
    return true;
  };
  /** bounds on the constraints 
   *  (clow<=-1e20 means no lower bound, cupp>=1e20 means no upper bound) */
  virtual bool get_cons_info(const long long& m, double* clow, double* cupp, NonlinearityType* type) {
    assert(m==1);
    *clow = *cupp = conrhs_;
    return true;
  };

  /** Objective function evaluation. Each rank returns the global obj. value. */
  virtual bool eval_f(const long long& n, const double* x, bool new_x, double& obj_value) {
    //! what is the obj exactly
    return true;
  };

  /** Gradient of objective (local chunk) */
  virtual bool eval_grad_f(const long long& n, const double* x, bool new_x, double* gradf) {
    return true;
  }

  /** Evaluates a subset of the constraints cons(x) (where clow<=cons(x)<=cupp). The subset is of size
   *  'num_cons' and is described by indexes in the 'idx_cons' array. The methods may be called 
   *  multiple times, each time for a subset of the constraints, for example, for the 
   *  subset containing the equalities and for the subset containing the inequalities. However, each 
   *  constraint will be inquired EXACTLY once. This is done for performance considerations, to avoid 
   *  temporary holders and memory copying.
   *
   *  Parameters:
   *   - n, m: the global number of variables and constraints
   *   - num_cons, idx_cons (array of size num_cons): the number and indexes of constraints to be evaluated
   *   - x: the point where the constraints are to be evaluated
   *   - new_x: whether x has been changed from the previous call to f, grad_f, or Jac
   *   - cons: array of size num_cons containing the value of the  constraints indicated by idx_cons
   *  
   *  When MPI enabled, every rank populates 'cons' since the constraints are not distributed.
   */
  virtual bool eval_cons(const long long& n, const long long& m, 
			 const long long& num_cons, const long long* idx_cons,  
			 const double* x, bool new_x, 
			 double* cons) {
    assert(n==n_);
    assert(m==1);
    assert(num_cons<=1);
    if(num_cons>0) {
      assert(idx_cons[0]==0);
      //! mfem dotprod - probably better to keep conbody_ as an mfem vector and use * (note: also see Dot in IterativeSolver)
      cons[0]=0.;
      for(int it=0; it<n_local_; it++) cons[0] += x[it]*conbody_[it];
    }
#ifdef MFEM_USE_MPI
    double gcon;
    int comm_ = MPI_COMM_WORLD;
    int ierr = MPI_Allreduce(cons, &gcon, 1, MPI_DOUBLE, MPI_SUM, comm_); assert(ierr==MPI_SUCCESS);
    cons[0] = gcon;
#endif
    return true;
  };

  /** provide a primal starting point. This point is subject to adjustments internally in hiOP.*/
  virtual bool get_starting_point(const long long&n, double* x0) {

    return true;
  };


  /** Evaluates the Jacobian of the subset of constraints indicated by idx_cons and of size num_cons.
   *  Example: Assuming idx_cons[k]=i, which means that the gradient of the (i+1)th constraint is
   *  to be evaluated, one needs to do Jac[k][0]=d/dx_0 con_i(x), Jac[k][1]=d/dx_1 con_i(x), ...
   *  When MPI enabled, each rank computes only the local columns of the Jacobian, that is the partials
   *  with respect to local variables.
   *
   *  Parameters: see eval_cons
   */
  virtual bool eval_Jac_cons(const long long& n, const long long& m, 
			     const long long& num_cons, const long long* idx_cons,  
			     const double* x, bool new_x,
			     double** Jac) {
    
    return true;
  };
  
private: 
  MPI_Comm comm_;
  //members that store problem info
  long long n_; //number of variables (global)
  int n_local_; //number of variables (local to the MPI process)
  double *lo_, *hi_;
  double *conbody_, conrhs_; //these are w and a from SetLinearConstraints
};


class HiopNlpOptimizer : public IterativeSolver
{
public:
  HiopNlpOptimizer();
#ifdef MFEM_USE_MPI
  HiopNlpOptimizer(MPI_Comm _comm);
#endif
  virtual ~HiopNlpOptimizer();

  void SetBounds(const Vector &_lo, const Vector &_hi);
  void SetLinearConstraint(const Vector &_w, double _a);

  // For this problem type, we let the target values play the role of the
  // initial vector xt, from which the operator generates the optimal vector x.
  virtual void Mult(const Vector &xt, Vector &x); //! const;

  /// These are not currently meaningful for this solver and will error out.
  virtual void SetPreconditioner(Solver &pr);
  virtual void SetOperator(const Operator &op);

private:
  void allocHiopProbSpec(const long long& numvars);

private:
#ifdef MFEM_USE_MPI
  MPI_Comm comm_;
#endif
  HiopProblemSpec* optProb_;
  hiop::hiopNlpDenseConstraints* hiopInstance_;

}; //end of hiop class


} // mfem namespace

#endif //MFEM_USE_HIOP
#endif //MFEM_HIOP guard
