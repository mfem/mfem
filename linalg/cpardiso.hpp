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

#ifndef MFEM_CPARDISO
#define MFEM_CPARDISO

#include "../config/config.hpp"
#include "mkl_cluster_sparse_solver.h"
#include "operator.hpp"

namespace mfem
{
class CPardisoSolver : public Solver
{
public:
   CPardisoSolver(MPI_Comm comm);

   void SetOperator(const Operator &op) override;

   void Mult(const Vector &x, Vector &y) const override;

   void SetPrintLevel(int print_lvl);

   ~CPardisoSolver();

private:
   MPI_Comm comm_;
   int m;
   int first_row;
   int nnz_loc;
   int m_loc;
   int *csr_rowptr = nullptr;
   double *reordered_csr_nzval = nullptr;
   int *reordered_csr_colind = nullptr;

   // Internal solver memory pointer pt,
   // 32-bit: int pt[64]
   // 64-bit: long int pt[64] or void *pt[64] should be OK on both architectures
   mutable void *pt[64] = {0};

   // Solver control parameters
   mutable int iparm[64] = {0};

   mutable int maxfct, mnum, msglvl, phase, error, err_mem;

   mutable int idum;
   double ddum;

   int mtype;
   int nrhs;
};
} // namespace mfem

#endif