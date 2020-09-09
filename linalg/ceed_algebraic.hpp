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

#ifndef MFEM_CEED_ALGEBRAIC_HPP
#define MFEM_CEED_ALGEBRAIC_HPP

#include "../config/config.hpp"

#ifdef MFEM_USE_CEED
#include "operator.hpp"
#include "../fem/libceed/ceedsolvers-utility.h"

namespace mfem
{

// forward declarations
class CeedMultigridLevel;
class BilinearForm;

class AlgebraicCeedSolver : public mfem::Solver
{
public:
   AlgebraicCeedSolver(Operator& fine_mfem_op, BilinearForm& form, 
                       Array<int>& ess_dofs);
   ~AlgebraicCeedSolver();

   /// Note that this does not rebuild the hierarchy or smoothers,
   /// just changes the finest level operator for residual computations
   void SetOperator(const Operator& op) { operators[0] = const_cast<Operator*>(&op); }

   void Mult(const Vector& x, Vector& y) const;

private:
   int num_levels;
   Operator ** operators;
   CeedMultigridLevel ** levels;
   Solver ** solvers;
   CeedOperator fine_composite_op;
};

} // namespace mfem

#endif // MFEM_USE_CEED

#endif // MFEM_CEED_ALGEBRAIC_HPP
