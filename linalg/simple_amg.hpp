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
#include "amgxsolver.hpp"
#include <memory>

namespace mfem
{

class SimpleAMG : public Solver
{
protected:
   const SparseMatrix &A;
   Solver &smoother;
   std::unique_ptr<Solver> coarse_solver;
   std::unique_ptr<SparseMatrix> Ac;
   SparseMatrix R;

   // Parallel version of local matrix, used for HypreBoomerAMG
   std::unique_ptr<HypreParMatrix> Ac_par;
   int row_starts[2];

   mutable Vector r, r_c, e_c, z;

   void FormRestriction();
public:
   enum class solverBackend{DIRECT,AMG_HYPRE,AMG_AMGX};

   SimpleAMG(const SparseMatrix &A_, Solver &smoother_, const solverBackend &backend, MPI_Comm comm, std::string="amgx.json");

   void Mult(const Vector &x, Vector &y) const;

   void SetOperator(const Operator &op) { }
};

} // namespace mfem

#endif // MFEM_SIMPLE_AMG
