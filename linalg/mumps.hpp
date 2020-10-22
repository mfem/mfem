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

/**
 * @brief MUMPS: A Parallel Sparse Direct Solver
 *
 * Interface for the distributed MUMPS solver
 */
class MUMPSSolver : public mfem::Solver
{
public:
   enum MatType
   {
      UNSYMMETRIC = 0,
      SYMMETRIC_INDEFINITE = 1,
      SYMMETRIC_POSITIVE_DEFINITE = 2
   };

   /**
    * @brief Default Constructor
    */
   MUMPSSolver() {}

   /**
    * @brief Set the Operator and perform factorization
    *
    * @a op needs to be of type HypreParMatrix.
    *
    * @param op Operator used in factorization and solve
    */
   void SetOperator(const Operator &op);

   /**
    * @brief Solve y = Op^{-1} x.
    *
    * @param x RHS vector
    * @param y Solution vector
    */
   void Mult(const Vector &x, Vector &y) const;

   /**
    * @brief Transpose Solve y = Op^{-T} x.
    *
    * @param x RHS vector
    * @param y Solution vector
    */
   void MultTranspose(const Vector &x, Vector &y) const;

   /**
    * @brief Set the error print level for MUMPS
    *
    * @param print_lvl Print level
    *
    * @note This method has to be called before SetOperator.
    */
   void SetPrintLevel(int print_lvl);

   /**
    * @brief Set the matrix type
    *
    * Supported matrix types: General, symmetric indefinite and
    * symmetric positive definite
    *
    * @param mtype Matrix type
    *
    * @note This method has to be called before SetOperator.
    */
   void SetMatrixSymType(MatType mtype);

   // Destructor
   ~MUMPSSolver();

private:

   // MPI communicator
   MPI_Comm comm;

   // Number of procs
   int numProcs;

   // local mpi id
   int myid;

   // parameter controling the matrix type
   MatType mat_type = MatType::UNSYMMETRIC;

   // parameter controling the printing level
   int print_level = 0;

   // local row offsets
   int row_start;

   // MUMPS object
   DMUMPS_STRUC_C *id=nullptr;

   // Method for setting MUMPS interal parameters
   void SetParameters();

#if MFEM_MUMPS_VERSION >= 530

   // row offests array on all procs
   Array<int> row_starts;

   // row map
   int * irhs_loc = nullptr;

   // These two methods are needed to distribute the local solution
   // vectors returned by MUMPS to the original MFEM parallel partition
   int GetRowRank(int i, const Array<int> &row_starts_) const;

   void RedistributeSol(const int * row_map,
                        const double * x,
                        double * y) const;
#else

   // Arrays needed for MPI_Gather and MPI_Scatter
   int * recv_counts = nullptr;

   int * displs = nullptr;

   double * rhs_glob = nullptr;

#endif

}; // mfem::MUMPSSolver class

} // namespace mfem

#endif // MFEM_USE_MPI
#endif // MFEM_USE_MUMPS
#endif // MFEM_MUMPS
