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

#ifndef MFEM_MUMPS
#define MFEM_MUMPS

#include "../config/config.hpp"

#ifdef MFEM_USE_MUMPS
#ifdef MFEM_USE_MPI

#include "operator.hpp"
#include "hypre.hpp"
#include <mpi.h>

#ifdef MFEM_USE_SINGLE
#include "smumps_c.h"
#else
#include "dmumps_c.h"
#endif

namespace mfem
{

/**
 * @brief MUMPS: A Parallel Sparse Direct Solver
 *
 * Interface for the distributed MUMPS solver
 */
class MUMPSSolver : public Solver
{
public:
   /// Specify the type of matrix we are applying the solver to
   enum MatType
   {
      /// General sparse matrix, no symmetry is assumed
      UNSYMMETRIC = 0,
      /// A sparse symmetric positive definite matrix
      SYMMETRIC_POSITIVE_DEFINITE = 1,
      /// A sparse symmetric matrix that is not necessarily positive definite
      SYMMETRIC_INDEFINITE = 2
   };

   /// Specify the reordering strategy for the MUMPS solver
   enum ReorderingStrategy
   {
      /// Let MUMPS automatically decide the reording strategy
      AUTOMATIC = 0,
      /// Approximate Minimum Degree with auto quasi-dense row detection is used
      AMD,
      /// Approximate Minimum Fill method will be used
      AMF,
      /// The PORD library will be used
      PORD,
      /// The METIS library will be used
      METIS,
      /// The ParMETIS library will be used
      PARMETIS,
      /// The Scotch library will be used
      SCOTCH,
      /// The PTScotch library will be used
      PTSCOTCH
   };

   /**
    * @brief Constructor with MPI_Comm parameter.
    */
   MUMPSSolver(MPI_Comm comm_);

   /**
    * @brief Constructor with a HypreParMatrix Operator.
    */
   MUMPSSolver(const Operator &op);

   /**
    * @brief Set the Operator and perform factorization
    *
    * @a op needs to be of type HypreParMatrix.
    *
    * @param op Operator used in factorization and solve
    */
   void SetOperator(const Operator &op);

   /**
    * @brief Solve $ y = Op^{-1} x $
    *
    * @param x RHS vector
    * @param y Solution vector
    */
   void Mult(const Vector &x, Vector &y) const;

   /**
    * @brief Solve $ Y_i = Op^{-T} X_i $
    *
    * @param X Array of RHS vectors
    * @param Y Array of Solution vectors
    */
   void ArrayMult(const Array<const Vector *> &X, Array<Vector *> &Y) const;

   /**
    * @brief Transpose Solve $ y = Op^{-T} x $
    *
    * @param x RHS vector
    * @param y Solution vector
    */
   void MultTranspose(const Vector &x, Vector &y) const;

   /**
    * @brief Transpose Solve $ Y_i = Op^{-T} X_i $
    *
    * @param X Array of RHS vectors
    * @param Y Array of Solution vectors
    */
   void ArrayMultTranspose(const Array<const Vector *> &X,
                           Array<Vector *> &Y) const;

   /**
    * @brief Set the error print level for MUMPS
    *
    * Supported values are:
    * - 0:  No output printed
    * - 1:  Only errors printed
    * - 2:  Errors, warnings, and main stats printed
    * - 3:  Errors, warning, main stats, and terse diagnostics printed
    * - 4:  Errors, warning, main stats, diagnostics, and input/output printed
    *
    * @param print_lvl Print level, default is 2
    *
    * @note This method has to be called before SetOperator
    */
   void SetPrintLevel(int print_lvl);

   /**
    * @brief Set the matrix type
    *
    * Supported matrix types: MUMPSSolver::UNSYMMETRIC,
    * MUMPSSolver::SYMMETRIC_POSITIVE_DEFINITE,
    * and MUMPSSolver::SYMMETRIC_INDEFINITE
    *
    * @param mtype Matrix type
    *
    * @note This method has to be called before SetOperator
    */
   void SetMatrixSymType(MatType mtype);

   /**
    * @brief Set the reordering strategy
    *
    * Supported reorderings are: MUMPSSolver::AUTOMATIC,
    * MUMPSSolver::AMD, MUMPSSolver::AMF, MUMPSSolver::PORD,
    * MUMPSSolver::METIS, MUMPSSolver::PARMETIS,
    * MUMPSSolver::SCOTCH, and MUMPSSolver::PTSCOTCH
    *
    * @param method Reordering method
    *
    * @note This method has to be called before SetOperator
    */
   void SetReorderingStrategy(ReorderingStrategy method);

   /**
    * @brief Set the flag controlling reuse of the symbolic factorization
    * for multiple operators
    *
    * @param reuse Flag to reuse symbolic factorization
    *
    * @note This method has to be called before repeated calls to SetOperator
    */
   void SetReorderingReuse(bool reuse);

   /**
    * @brief Set the tolerance for activating block low-rank (BLR) approximate
    * factorization
    *
    * @param tol Tolerance
    *
    * @note This method has to be called before SetOperator
    */
#if MFEM_MUMPS_VERSION >= 510
   void SetBLRTol(double tol);
#endif

   // Destructor
   ~MUMPSSolver();

private:
   // MPI communicator
   MPI_Comm comm;

   // Number of procs
   int numProcs;

   // MPI rank
   int myid;

   // Parameter controlling the matrix type
   MatType mat_type;

   // Parameter controlling the printing level
   int print_level;

   // Parameter controlling the reordering strategy
   ReorderingStrategy reorder_method;

   // Parameter controlling whether or not to reuse the symbolic factorization
   // for multiple calls to SetOperator
   bool reorder_reuse;

#if MFEM_MUMPS_VERSION >= 510
   // Parameter controlling the Block Low-Rank (BLR) feature in MUMPS
   double blr_tol;
#endif

   // Local row offsets
   int row_start;

   // MUMPS object
#ifdef MFEM_USE_SINGLE
   SMUMPS_STRUC_C *id;
#else
   DMUMPS_STRUC_C *id;
#endif

   /// Method for initialization
   void Init(MPI_Comm comm_);

   /// Method for setting MUMPS internal parameters
   void SetParameters();

   /// Method for configuring storage for distributed/centralized RHS and
   /// solution
   void InitRhsSol(int nrhs) const;

#if MFEM_MUMPS_VERSION >= 530
   // Row offests array on all procs
   Array<int> row_starts;

   // Row maps and storage for distributed RHS and solution
   int *irhs_loc, *isol_loc;
   mutable real_t *rhs_loc, *sol_loc;

   // These two methods are needed to distribute the local solution
   // vectors returned by MUMPS to the original MFEM parallel partition
   int GetRowRank(int i, const Array<int> &row_starts_) const;
   void RedistributeSol(const int *rmap, const real_t *x, const int lx_loc,
                        Array<Vector *> &Y) const;
#else
   // Arrays needed for MPI_Gatherv and MPI_Scatterv
   int *recv_counts, *displs;
   mutable real_t *rhs_glob;
#endif
}; // mfem::MUMPSSolver class

} // namespace mfem

#endif // MFEM_USE_MPI
#endif // MFEM_USE_MUMPS
#endif // MFEM_MUMPS
