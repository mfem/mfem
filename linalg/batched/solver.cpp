// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "solver.hpp"

namespace mfem
{

BatchedDirectSolver::BatchedDirectSolver(const DenseTensor &A_, Mode mode_,
                                         BatchedLinAlg::Backend backend_)
   : A(A_), mode(mode_), backend(backend_)
{
   MFEM_VERIFY(A.SizeI() == A.SizeJ(), "Blocks must be square.");
   if (mode == LU)
   {
      BatchedLinAlg::Get(backend).LUFactor(A, P);
   }
   else
   {
      BatchedLinAlg::Get(backend).Invert(A);
   }
}

void BatchedDirectSolver::Mult(const Vector &x, Vector &y) const
{
   if (mode == LU)
   {
      y = x;
      BatchedLinAlg::Get(backend).LUSolve(A, P, y);
   }
   else
   {
      BatchedLinAlg::Get(backend).Mult(A, x, y);
   }
}

void BatchedDirectSolver::SetOperator(const Operator &op)
{
   MFEM_ABORT("Not supported.");
}

} // namespace mfem
