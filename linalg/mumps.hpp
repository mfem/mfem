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

#ifndef MFEM_MUMPS
#define MFEM_MUMPS

#include "../config/config.hpp"

#ifdef MFEM_USE_MUMPS
#ifdef MFEM_USE_MPI
#include "operator.hpp"
#include "hypre.hpp"

#include <mpi.h>
#include "dmumps_c.h"
#include <vector>

namespace mfem
{
class MUMPSSolver : public mfem::Solver
{
public:
   // Default Constructor.
   MUMPSSolver() {}

   void SetMatrixSymType(int sym_) { sym = (sym_>2) ? 0 : sym_ ; }

   void SetPrintLevel(int print_level_) { print_level=print_level_;}

   // Factor and solve the linear system y = Op^{-1} x.
   void Mult(const Vector &x, Vector &y) const;

   void MultTranspose(const Vector &x, Vector &y) const;

   // Set the operator.
   void SetOperator(const Operator &op);

   // Default destructor.
   ~MUMPSSolver();

private:

   MPI_Comm comm;

   int numProcs;

   int myid;

   int sym=0;

   int print_level=0;

   int row_start;

   int *I;

   int *J;

   double * data;

   // MUMPS workspace
   // macro s.t. indices match MUMPS documentation
#define ICNTL(I) icntl[(I) -1]
#define INFO(I) info[(I) -1]

   DMUMPS_STRUC_C *id=nullptr;

   void SetParameters();

#if MFEM_MUMPS_VERSION >= 530

   Array<int> row_starts;

   Array<int> irhs_loc;

   Array<int> isol_loc;

   Vector sol_loc;

   int GetRowRank(int i, const Array<int> &row_starts_) const;

   void RedistributeSol(const Array<int> &row_map,
                        const Vector &x,
                        Vector &y) const;
#else
   Array<int> recv_counts;

   Array<int> displs;

   Vector rhs_glob;

#endif

}; // mfem::MUMPSSolver class

} // namespace mfem

#endif // MFEM_USE_MPI
#endif // MFEM_USE_MUMPS
#endif // MFEM_MUMPS
