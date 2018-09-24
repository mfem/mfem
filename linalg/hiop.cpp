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

#include "../config/config.hpp"
#include "hiop.hpp"


#ifdef MFEM_USE_HIOP
#include <iostream>

#include "hiopAlgFilterIPM.hpp"

using namespace hiop;

namespace mfem
{

bool HiopOptimizationProblem::get_prob_sizes(long long &n, long long &m)
{
   n = n_glob;

   const int m_loc = problem.GetNumConstraints();
   MPI_Allreduce(&m_loc, &m, 1, MPI_LONG_LONG_INT, MPI_SUM, comm_);

   return true;
}

bool HiopOptimizationProblem::get_vars_info(const long long &n,
                                            double *xlow, double *xupp,
                                            NonlinearityType *type)
{
   MFEM_ASSERT(n == n_glob, "Global input mismatch.");

   std::memcpy(xlow, problem.x_lo->GetData(), n_loc * sizeof(double));
   std::memcpy(xupp, problem.x_hi->GetData(), n_loc * sizeof(double));

   return true;
}

bool HiopOptimizationProblem::get_cons_info(const long long &m,
                                            double *clow, double *cupp,
                                            NonlinearityType *type)
{
   MFEM_ASSERT(m == m_glob, "Global constraint size mismatch.");

   const int csize = problem.c_e->Size();
   std::memcpy(clow, problem.c_e->GetData(), csize * sizeof(double));
   std::memcpy(cupp, problem.c_e->GetData(), csize * sizeof(double));
   const int dsize = problem.d_lo->Size();
   std::memcpy(clow + csize, problem.d_lo->GetData(), dsize * sizeof(double));
   std::memcpy(cupp + csize, problem.d_hi->GetData(), dsize * sizeof(double));

   return true;
}

bool HiopOptimizationProblem::eval_f(const long long &n, const double *x,
                                     bool new_x, double &obj_value)
{
   MFEM_ASSERT(n == n_glob, "Global input mismatch.");

   Vector x_vec(n_loc);
   x_vec = x;
   obj_value = problem.CalcObjective(x_vec);

#ifdef MFEM_USE_MPI
   const double loc_obj = obj_value;
   MPI_Allreduce(&loc_obj, &obj_value, 1, MPI_DOUBLE, MPI_SUM, comm_);
#endif

   return true;
}

bool HiopOptimizationProblem::eval_grad_f(const long long &n, const double *x,
                                          bool new_x, double *gradf)
{
   MFEM_ASSERT(n == n_glob, "Global input mismatch.");

   Vector x_vec(n_loc), gradf_vec(gradf, n_loc);
   x_vec = x;
   problem.CalcObjectiveGrad(x_vec, gradf_vec);

   return true;
}

bool HiopOptimizationProblem::eval_cons(const long long& n, const long long& m,
                                        const long long& num_cons,
                                        const long long* idx_cons,
                                        const double* x, bool new_x,
                                        double* cons)
{
   MFEM_ASSERT(n == n_glob, "Global input mismatch.");
   MFEM_ASSERT(m == m_glob, "Constraint size mismatch.");
   MFEM_ASSERT(num_cons <= m, "num_cons should be at most m = " << m);

   if (num_cons == 0) { return true; }

   if (new_x) { cons_vals_are_current = false; }
   Vector x_vec(n_loc);
   x_vec = x;
   ComputeConstraintValues(x_vec);

   for (int c = 0; c < num_cons; c++)
   {
      MFEM_ASSERT(idx_cons[c] < m_glob, "Constraint index is out of bounds.");
      cons[c] = cons_values(idx_cons[c]);
   }

   return true;
}

void HiopOptimizationProblem::ComputeConstraintValues(const Vector x)
{
   if (cons_vals_are_current) { return; }

   const int cheight = problem.C->Height(), dheight = problem.D->Height();
   Vector cons_C(cons_values.GetData(), cheight),
          cons_D(cons_values.GetData() + cheight, dheight);
   problem.C->Mult(x, cons_C);
   problem.D->Mult(x, cons_D);

#ifdef MFEM_USE_MPI
   Vector loc_cons(cons_values);
   MPI_Allreduce(loc_cons.GetData(), cons_values.GetData(), m_glob,
                 MPI_DOUBLE, MPI_SUM, comm_);
#endif

   cons_vals_are_current = true;
}

HiopNlpOptimizer::HiopNlpOptimizer() : OptimizationSolver(), optProb_(NULL)
{ 
#ifdef MFEM_USE_MPI
  //in case a serial driver in parallel MFEM build calls HiOp
  comm_ = MPI_COMM_WORLD;
  int initialized, nret = MPI_Initialized(&initialized); 
  MFEM_ASSERT(MPI_SUCCESS==nret, "failure in calling MPI_Initialized");
  if(!initialized) {
    nret = MPI_Init(NULL, NULL);
    MFEM_ASSERT(MPI_SUCCESS==nret, "failure in calling MPI_Init");
  }
#endif
} 

#ifdef MFEM_USE_MPI
HiopNlpOptimizer::HiopNlpOptimizer(MPI_Comm _comm) 
  : OptimizationSolver(_comm),
    optProb_(NULL),
    comm_(_comm) { }
#endif

HiopNlpOptimizer::~HiopNlpOptimizer()
{
   delete optProb_;
}

void HiopNlpOptimizer::SetOptimizationProblem(OptimizationProblem &prob)
{
   problem = &prob;

   if (optProb_) { delete optProb_; }

#ifdef MFEM_USE_MPI
   optProb_ = new HiopOptimizationProblem(comm_, prob);
#else
   optProb_ = new HiopOptimizationProblem(prob);
#endif
}

void HiopNlpOptimizer::Mult(const Vector &xt, Vector &x) const
{
   if (iterative_mode) { optProb_->setStartingPoint(xt); }

   hiop::hiopNlpDenseConstraints hiopInstance(*optProb_);

   // Set tolerance:
   hiopInstance.options->SetNumericValue("tolerance", abs_tol);
   // TODO move this somehow before the constructor of hiopInstance.
   hiopInstance.options->SetStringValue("fixed_var", "relax");
   // 0: no output; 3: not too much
   hiopInstance.options->SetIntegerValue("verbosity_level", print_level);

   // Use the IPM solver.
   hiop::hiopAlgFilterIPM solver(&hiopInstance);
   const hiop::hiopSolveStatus status = solver.run();
   final_norm = solver.getObjective();

   if (status != hiop::Solve_Success)
   {
      converged = false;
      MFEM_WARNING("HIOP returned with a non-success status: " << status);
   }

   // Copy the final solution in x.
   solver.getSolution(x.GetData());
}

void HiopNlpOptimizer_Simple::Mult(const Vector &xt, Vector &x) const
{
   long long int n_local, m;
   optProb_Simple_->get_prob_sizes(n_local, m);

   //set xt in the problemSpec to compute the objective
   optProb_Simple_->setObjectiveTarget(xt);
   HiopNlpOptimizer::Mult(xt, x);
}

void HiopNlpOptimizer::SetBounds(const Vector &_lo, const Vector &_hi)
{
   if (NULL==optProb_)
      allocHiopProbSpec(_lo.Size());

   //optProb_->setBounds(_lo, _hi);
}

void HiopNlpOptimizer::SetLinearConstraint(const Vector &_w, double _a)
{
   if (NULL==optProb_)
      allocHiopProbSpec(_w.Size());

   optProb_->setLinearConstraint(_w, _a);
}

void HiopNlpOptimizer::SetObjectiveFunction(const DenseMatrix &_A,
                                            const Vector &_c)
{
   if (NULL==optProb_)
      allocHiopProbSpec(_c.Size());
   optProb_->setObjectiveFunction(_A, _c);
}
void HiopNlpOptimizer::SetObjectiveFunction(const DenseMatrix &_A)
{
   if (NULL==optProb_)
      allocHiopProbSpec(_A.Width());
   optProb_->setObjectiveFunction(_A);
}
void HiopNlpOptimizer::SetObjectiveFunction(const Vector &_c)
{
   if (NULL==optProb_)
      allocHiopProbSpec(_c.Size());
   optProb_->setObjectiveFunction(_c);
}

void HiopNlpOptimizer::allocHiopProbSpec(const long long& numvars)
{
   MFEM_ASSERT(optProb_==NULL, "HiopProbSpec object already created");

#ifdef MFEM_USE_MPI
   optProb_ = new HiopOptimizationProblem(comm_, *problem);
#else
   optProb_ = new HiopOptimizationProblem(*problem);
#endif
}

void HiopNlpOptimizer_Simple::allocHiopProbSpec(const long long& numvars)
{
   MFEM_ASSERT(optProb_Simple_==NULL && optProb_==NULL,
                "HiopProbSpec object already created");

#ifdef MFEM_USE_MPI
   optProb_Simple_ = new HiopProblemSpec_Simple(comm_, *problem);
#else
   optProb_Simple_ = new HiopProblemSpec_Simple(*problem);
#endif

   optProb_ = optProb_Simple_;
}


} // mfem namespace
#endif // MFEM_USE_HIOP
