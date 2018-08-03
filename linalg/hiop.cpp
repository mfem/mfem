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

#pragma message "Compiling " __FILE__ "..."

using namespace hiop;

namespace mfem
{ 


HiopNlpOptimizer::HiopNlpOptimizer()
  : optProb_(NULL), hiopInstance_(NULL)
{
}

#ifdef MFEM_USE_MPI
HiopNlpOptimizer::HiopNlpOptimizer(MPI_Comm _comm) 
  : IterativeSolver(_comm), comm_(_comm), optProb_(NULL), hiopInstance_(NULL)

{
};
#endif

HiopNlpOptimizer::~HiopNlpOptimizer()
{
  if(optProb_) delete optProb_;
  if(hiopInstance_) delete hiopInstance_;
}

  void HiopNlpOptimizer::Mult(const Vector &xt, Vector &x)// const
{
  //set xt in the problemSpec to compute the objective
  //todo

  //instantiate Hiop's NLP formulation (dense constraints) 
  assert(hiopInstance_==NULL);
  hiopInstance_ = new hiop::hiopNlpDenseConstraints(*optProb_);
  {
    //use the IPM solver
    hiop::hiopAlgFilterIPM solver(hiopInstance_);
    hiop::hiopSolveStatus status = solver.run();
    double objective = solver.getObjective();

    //get the solution from the solver and copy it to x
    //todo
  }

  delete hiopInstance_; 
  hiopInstance_ = NULL;
}

void HiopNlpOptimizer::SetBounds(const Vector &_lo, const Vector &_hi)
{
  if(NULL==optProb_) 
    allocHiopProbSpec(_lo.Size());
}

void HiopNlpOptimizer::SetLinearConstraint(const Vector &_w, double _a)
{
  if(NULL==optProb_) 
    allocHiopProbSpec(_w.Size());

  
}

void HiopNlpOptimizer::SetPreconditioner(Solver &pr)
{
   mfem_error("HiopNlpOptimizer::SetPreconditioner() : "
              "not meaningful for this solver");
}

void HiopNlpOptimizer::SetOperator(const Operator &op)
{
   mfem_error("HiopNlpOptimizer::SetOperator() : "
              "not meaningful for this solver");
}



void HiopNlpOptimizer::allocHiopProbSpec(const long long& numvars) {
  //! mfem assert strategy
  assert(optProb_==NULL && "HiopProbSpec object already created");
  optProb_ = new HiopProblemSpec(comm_, numvars);
};


} // mfem namespace
#endif // MFEM_USE_HIOP
