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

class HiopProblemSpec : public hiop::hiopInterfaceDenseConstraints
{

public:

   HiopProblemSpec(const long long& n_loc)
      : n_(n_loc), n_local_(n_loc), a_(0.), workVec_(n_loc) 
  { 
#ifdef MFEM_USE_MPI
    //in case HiOp with MPI support is called by a serial driver.
    comm_ = MPI_COMM_WORLD;
#endif
  }

#ifdef MFEM_USE_MPI
  HiopProblemSpec(const MPI_Comm& _comm, const long long& _n_local)
    : comm_(_comm), n_local_(_n_local), a_(0.), workVec_(_n_local)
  { 
     int ierr = MPI_Allreduce(&n_local_, &n_, 1, MPI_LONG_LONG_INT, MPI_SUM, comm_);
     MFEM_ASSERT(ierr==MPI_SUCCESS, "MPI_Allreduce failed with error" << ierr);
  }
#endif

  /** problem dimensions: n number of variables, m number of constraints */
  virtual bool get_prob_sizes(long long int& n, long long int& m) {
    n = n_;
    m = 1;
    return true;
  };

  virtual bool get_vars_info(const long long& n, double *xlow, double* xupp, NonlinearityType* type) {
    MFEM_ASSERT(n==n_, "global size input mismatch");
    std::memcpy(xlow, lo_.GetData(), n_local_*sizeof(double));
    std::memcpy(xupp, hi_.GetData(), n_local_*sizeof(double));
    return true;
  };
  /** bounds on the constraints 
   *  (clow<=-1e20 means no lower bound, cupp>=1e20 means no upper bound) */
  virtual bool get_cons_info(const long long& m, double* clow, double* cupp, NonlinearityType* type) {
    MFEM_ASSERT(m==1, "only one constraint should be present");
    *clow = *cupp = a_;
    return true;
  };

  /** Objective function evaluation. Each rank returns the global obj. value. */
  virtual bool eval_f(const long long& n, const double* x, bool new_x, double& obj_value) {
    MFEM_ASSERT(n==n_, "global size input mismatch");

    workVec_ = x;
    obj_value = 0.5 * (workVec_ * workVec_);

#ifdef MFEM_USE_MPI
    double loc_obj = obj_value;
    int ierr = MPI_Allreduce(&loc_obj, &obj_value, 1, MPI_DOUBLE, MPI_SUM, comm_);
    MFEM_ASSERT(ierr==MPI_SUCCESS, "MPI_Allreduce failed with error" << ierr);
#endif

    return true;
  };

  /** Gradient of objective (local chunk) */
  virtual bool eval_grad_f(const long long& n, const double* x, bool new_x, double* gradf) {
    MFEM_ASSERT(n==n_, "global size input mismatch");

    // compute gradf = x
    workVec_  = x;
    std::memcpy(gradf, workVec_.GetData(), n_local_*sizeof(double));

    return true;
  };

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
    MFEM_ASSERT(n==n_, "global size input mismatch");
    MFEM_ASSERT(m==1, "only one constraint should be present");
    MFEM_ASSERT(num_cons<=m, "num_cons should be at most m=" << m);
    if(num_cons>0) {
      MFEM_ASSERT(idx_cons[0]==0, "index of the constraint should be 0");

      workVec_ = x;
      double wtx = w_ * workVec_;

#ifdef MFEM_USE_MPI
      double loc_wtx = wtx;
      int ierr = MPI_Allreduce(&loc_wtx, &wtx, 1, MPI_DOUBLE, MPI_SUM, comm_);
      MFEM_ASSERT(ierr==MPI_SUCCESS, "MPI_Allreduce failed with error" << ierr);
#endif
      //w' * x - a
      cons[0] = wtx - a_;
    }
    return true;
  };

  /** provide a primal starting point. This point is subject to adjustments internally in hiOP.*/
  virtual bool get_starting_point(const long long&n, double* x0) {
    MFEM_ASSERT(n_local_ == w_.Size(), "Linear constraint vector not set ?!?");
    memcpy(x0, w_.GetData(), n_local_*sizeof(double));
    return true;
    //let hiop decide: return false;
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
    MFEM_ASSERT(n==n_, "global size input mismatch");
    MFEM_ASSERT(m==1, "only one constraint should be present");
    MFEM_ASSERT(num_cons<=m, "num_cons should be at most m=" << m);
    if(num_cons>0) {
      MFEM_ASSERT(idx_cons[0]==0, "index of the constraint should be 0");

      std::memcpy(Jac[0], w_.GetData(), n_local_*sizeof(double));
    }
    return true;
  };

  /**  column partitioning specification for distributed memory vectors 
   *  Process P owns cols[P], cols[P]+1, ..., cols[P+1]-1, P={0,1,...,NumRanks}.
   *  Example: for a vector x of 6 elements on 3 ranks, the col partitioning is cols=[0,2,4,6].
   *  The caller manages memory associated with 'cols', array of size NumRanks+1 
   */
  virtual bool get_vecdistrib_info(long long global_n, long long* cols) {
#ifdef MFEM_USE_MPI
    int nranks;
    int ierr = MPI_Comm_size(comm_, &nranks);
    MFEM_ASSERT(ierr==MPI_SUCCESS, "MPI_Comm_size failed with error" << ierr);
    
    long long* sizes = new long long[nranks];
    ierr = MPI_Allgather(&n_local_, 1, MPI_LONG_LONG_INT, sizes, 1, MPI_LONG_LONG_INT, comm_);
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
  };

#ifdef MFEM_USE_MPI
  virtual bool get_MPI_comm(MPI_Comm& comm_out) 
  { 
    comm_out=comm_; 
    return true;
  }
#endif

  /** Seter/geter methods below; not inherited from the HiOp interface class */
  virtual void setBounds(const Vector &_lo, const Vector &_hi) {
    lo_ = _lo;
    hi_ = _hi;
  };
  virtual void setLinearConstraint(const Vector &_w, const double& _a) {
    w_ = _w;
    a_ = _a;
  };

protected:
#ifdef MFEM_USE_MPI
  MPI_Comm comm_;
#endif

  //members that store problem info
  long long n_; //number of variables (global)
  long long n_local_; //number of variables (local to the MPI process)

  Vector lo_,hi_; //lower and upper bounds
  Vector w_;      //linear constraint coefficients 
  double a_;      //linear constraint rhs

  Vector workVec_; //used as work space of size n_local_

}; //End of HiopProblemSpec class

// Special class for HiopProblemSpec where f = ||x-xt||_2
class HiopProblemSpec_Simple : public HiopProblemSpec
{

public:

   HiopProblemSpec_Simple(const long long& n_loc)
      : HiopProblemSpec(n_loc) {}

#ifdef MFEM_USE_MPI
  HiopProblemSpec_Simple(const MPI_Comm& _comm, const long long& _n_local)
    : HiopProblemSpec(_comm, _n_local) {}
#endif

  /** Objective function evaluation. Each rank returns the global obj. value. */
  virtual bool eval_f(const long long& n, const double* x, bool new_x, double& obj_value) {
    MFEM_ASSERT(n==n_, "global size input mismatch");

    workVec_ = x;
    workVec_.Add(-1.0, xt_);
    obj_value = 0.5 * (workVec_ * workVec_);

#ifdef MFEM_USE_MPI
    double loc_obj = obj_value;
    int ierr = MPI_Allreduce(&loc_obj, &obj_value, 1, MPI_DOUBLE, MPI_SUM, comm_);
    MFEM_ASSERT(ierr==MPI_SUCCESS, "MPI_Allreduce failed with error" << ierr);
#endif

    return true;
  };

  /** Gradient of objective (local chunk) */
  virtual bool eval_grad_f(const long long& n, const double* x, bool new_x, double* gradf) {
    MFEM_ASSERT(n==n_, "global size input mismatch");

    // compute gradf = x-xt
    workVec_  = x;
    workVec_ -= xt_;
    std::memcpy(gradf, workVec_.GetData(), n_local_*sizeof(double));

    return true;
  };

  virtual void setObjectiveTarget(const Vector &_xt) {
    xt_ = _xt;
  };

protected:
  Vector xt_;     //target vector in the L2 objective

}; //End of HiopProblemSpec_Simple class

class HiopNlpOptimizer : public OptimizationSolver
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
  virtual void Mult(Vector &x) const;

  // For this problem type, xt is just used as a starting guess
  //   (if applicable) to x
  virtual void Mult(const Vector &xt, Vector &x) const{
     x = xt;
     Mult(x);
  }

protected:
  virtual void allocHiopProbSpec(const long long& numvars);

protected:
  HiopProblemSpec* optProb_;

#ifdef MFEM_USE_MPI
  MPI_Comm comm_;
#endif

}; //end of HiopNlpOptimizer class

// Special class for f = ||x-xt||_2
class HiopNlpOptimizer_Simple : public HiopNlpOptimizer
{
public:
  HiopNlpOptimizer_Simple() : HiopNlpOptimizer(),
                       optProb_Simple_(NULL) { }

#ifdef MFEM_USE_MPI
  HiopNlpOptimizer_Simple(MPI_Comm _comm) : HiopNlpOptimizer(_comm),
                                     optProb_Simple_(NULL) { }
#endif

  // Assumes xt = 0
  virtual void Mult(Vector &x) const;

  // For this problem type, we let the target values play the role of the
  // initial vector xt, from which the operator generates the optimal vector x.
  virtual void Mult(const Vector &xt, Vector &x) const;

protected:
  virtual void allocHiopProbSpec(const long long& numvars);

private:
  HiopProblemSpec_Simple* optProb_Simple_;

}; //end of HiopNlpOptimizer class


} // mfem namespace

#endif //MFEM_USE_HIOP
#endif //MFEM_HIOP guard
