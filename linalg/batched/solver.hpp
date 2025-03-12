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

#ifndef MFEM_BATCHED_SOLVER
#define MFEM_BATCHED_SOLVER

#include "batched.hpp"
#include "../operator.hpp"

namespace mfem
{

/// @brief Solve block-diagonal systems using batched LU or inverses.
///
/// LU factorization is more numerically stable, but exposes less fine-grained
/// parallelism. Inverse matrices have worse conditioning (and increased setup
/// time), but solving the system is more efficient in parallel (e.g. on GPUs).
class BatchedDirectSolver : public Solver
{
public:
   /// %Solver mode: whether to use LU factorization or inverses.
   enum Mode
   {
      LU, ///< LU factorization.
      INVERSE ///< Inverse matrices.
   };
protected:
   DenseTensor A; ///< The LU factors/inverses of the input matrices.
   Array<int> P; ///< Pivots (needed only for LU factors).
   Mode mode; ///< Solver mode.
   BatchedLinAlg::Backend backend; ///< Requested batched linear algebra backend.
public:
   /// @brief Constructor.
   ///
   /// The DenseTensor @a A_ has dimensions $(m, m, n)$, and represents a block
   /// diagonal matrix $A$ with $n$ blocks of size $m \times m$.
   ///
   /// A deep copy is made of the input @a A_, and so it does not need to be
   /// retained by the caller.
   BatchedDirectSolver(const DenseTensor &A_, Mode mode_,
                       BatchedLinAlg::Backend backend_ =
                          BatchedLinAlg::GetActiveBackend());
   /// Sets $y = A^{-1} x$.
   void Mult(const Vector &x, Vector &y) const;
   /// Not supported (aborts).
   void SetOperator(const Operator &op);
};

} // namespace mfem

#endif
