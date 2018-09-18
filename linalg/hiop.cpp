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
HiopNlpOptimizer::HiopNlpOptimizer() 
  : OptimizationSolver(), optProb_(NULL), use_initial_x_value(false)
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
    comm_(_comm),
    use_initial_x_value(false)
{ 

}
#endif

HiopNlpOptimizer::~HiopNlpOptimizer()
{
   if (optProb_) delete optProb_;
}

void HiopNlpOptimizer::Mult(Vector &x) const
{
   if (use_initial_x_value) optProb_->setStartingPoint(x);
   hiop::hiopNlpDenseConstraints hiopInstance(*optProb_);

   //Set tolerance:
   hiopInstance.options->SetNumericValue("tolerance", abs_tol);
   //0: no output; 3: not too much
   hiopInstance.options->SetIntegerValue("verbosity_level", print_level);
   // TODO add capability to pass in some of these arguments

   //use the IPM solver
   hiop::hiopAlgFilterIPM solver(&hiopInstance);
   hiop::hiopSolveStatus status = solver.run();
   double objective = solver.getObjective();

   MFEM_ASSERT(solver.getSolveStatus()==hiop::Solve_Success,
               "optimizer returned with a non-success status: "
                  << solver.getSolveStatus());

   //copy the solution to x
   solver.getSolution(x.GetData());
}

void HiopNlpOptimizer_Simple::Mult(Vector &x) const
{
   long long int n_local, m;
   optProb_Simple_->get_prob_sizes(n_local, m);

   // Solve assuming xt = 0
   Vector xt(n_local);
   xt = 0.;
   MFEM_WARNING("Assuming xt = 0 in HiopNlpOptimizer_Simple::Mult!");
   Mult(xt, x);
}

void HiopNlpOptimizer_Simple::Mult(const Vector &xt, Vector &x) const
{
   //set xt in the problemSpec to compute the objective
   optProb_Simple_->setObjectiveTarget(xt);
   HiopNlpOptimizer::Mult(x);
}

void HiopNlpOptimizer::SetBounds(const Vector &_lo, const Vector &_hi)
{
   if (NULL==optProb_)
      allocHiopProbSpec(_lo.Size());

   optProb_->setBounds(_lo, _hi);
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
   optProb_ = new HiopProblemSpec(comm_, numvars);
#else
   optProb_ = new HiopProblemSpec(numvars);
#endif
}

void HiopNlpOptimizer_Simple::allocHiopProbSpec(const long long& numvars)
{
   MFEM_ASSERT(optProb_Simple_==NULL && optProb_==NULL,
                "HiopProbSpec object already created");

#ifdef MFEM_USE_MPI
   optProb_Simple_ = new HiopProblemSpec_Simple(comm_, numvars);
#else
   optProb_Simple_ = new HiopProblemSpec_Simple(numvars);
#endif

   optProb_ = optProb_Simple_;
}


} // mfem namespace
#endif // MFEM_USE_HIOP
