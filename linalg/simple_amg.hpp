// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_SIMPLE_AMG
#define MFEM_SIMPLE_AMG

#include "sparsemat.hpp"
#include "hypre.hpp"

namespace mfem
{

class SimpleAMG : public Solver
{
private:
   const SparseMatrix *A;
   SparseMatrix *Ac, *R;
   Solver *coarse_solver, *smoother;

   // Parallel stuff
   HypreParMatrix *Ac_par;
   int bounds[2];

   SparseMatrix *Restriction() const;

public:
   SimpleAMG(const SparseMatrix *A, Solver *smoother, MPI_Comm comm,
             bool two_level=true);
   ~SimpleAMG();

   static HypreParMatrix *ToHypreParMatrix(SparseMatrix *B, MPI_Comm comm,
                                           int *bounds);

   void Mult(const Vector &x, Vector &y) const;

   void SetOperator(const Operator &op) { }
};

} // namespace mfem

#endif // MFEM_SIMPLE_AMG
