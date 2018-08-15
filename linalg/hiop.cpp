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

#ifdef MFEM_USE_MPI

#endif

using namespace hiop;

namespace mfem
{ 


HiopNlpOptimizer::HiopNlpOptimizer()
  : optProb_(NULL)//, hiopInstance_(NULL)
{
}

#ifdef MFEM_USE_MPI
HiopNlpOptimizer::HiopNlpOptimizer(MPI_Comm _comm) 
  : OptimizationSolver(_comm), comm_(_comm), optProb_(NULL)//, hiopInstance_(NULL)

{
};
#endif

HiopNlpOptimizer::~HiopNlpOptimizer()
{
  if(optProb_) delete optProb_;
  //if(hiopInstance_) delete hiopInstance_;
}

void HiopNlpOptimizer::Mult(const Vector &xt, Vector &x) const
{
  //set xt in the problemSpec to compute the objective
  optProb_->setObjectiveTarget(xt);

  //instantiate Hiop's NLP formulation (dense constraints) 
  //assert(hiopInstance_==NULL && "This should be allocated and deallocated in the Mult operator");

  hiop::hiopNlpDenseConstraints hiopInstance(*optProb_);
  hiopInstance.options->SetNumericValue("tolerance", 1e-7);
  hiopInstance.options->SetIntegerValue("verbosity_level", 0); //0: no output; 3: not too much 

  //use the IPM solver
  hiop::hiopAlgFilterIPM solver(&hiopInstance);
  hiop::hiopSolveStatus status = solver.run();
  double objective = solver.getObjective();

  MFEM_ASSERT(solver.getSolveStatus()==hiop::Solve_Success, "optimizer returned with a non-success status: " << solver.getSolveStatus());

  //copy the solution to x
  solver.getSolution(x.GetData());
}

void HiopNlpOptimizer::SetBounds(const Vector &_lo, const Vector &_hi)
{
  if(NULL==optProb_) 
    allocHiopProbSpec(_lo.Size());

  optProb_->setBounds(_lo, _hi);
}

void HiopNlpOptimizer::SetLinearConstraint(const Vector &_w, double _a)
{
  if(NULL==optProb_) 
    allocHiopProbSpec(_w.Size());

  optProb_->setLinearConstraint(_w, _a);
}

void HiopNlpOptimizer::allocHiopProbSpec(const long long& numvars) {
  //! mfem assert strategy?
  assert(optProb_==NULL && "HiopProbSpec object already created");
  optProb_ = new HiopProblemSpec(comm_, numvars);
};


} // mfem namespace
#endif // MFEM_USE_HIOP
