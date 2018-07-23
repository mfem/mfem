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

namespace mfem
{

class HiopNlpOptimizer : public IterativeSolver
{
public:
  HiopNlpOptimizer() {}; 
#ifdef MFEM_USE_MPI
  HiopNlpOptimizer(MPI_Comm _comm);
#endif

  void SetBounds(const Vector &_lo, const Vector &_hi);
  void SetLinearConstraint(const Vector &_w, double _a);

  // For this problem type, we let the target values play the role of the
  // initial vector xt, from which the operator generates the optimal vector x.
  virtual void Mult(const Vector &xt, Vector &x) const;

  /// These are not currently meaningful for this solver and will error out.
  virtual void SetPreconditioner(Solver &pr);
  virtual void SetOperator(const Operator &op);

private:
  

}; //end of hiop class


} // mfem namespace

#endif //MFEM_USE_HIOP
#endif //MFEM_HIOP guard
