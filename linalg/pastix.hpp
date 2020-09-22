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

/**
 * @file pastix.hpp
 *
 * @brief This file contains a RAII wrapper for the PaStiX sparse matrix class
 * and an associated PaStiX direct solver wrapper
 */


#ifndef MFEM_PASTIX
#define MFEM_PASTIX

#include "../config/config.hpp"

#ifdef MFEM_USE_MPI
#ifdef MFEM_USE_PASTIX
#include "operator.hpp"
#include "hypre.hpp"

#include <mpi.h>

#include "pastix.h"

namespace mfem
{

/**
 * @brief A memory-safe wrapper for INRIA's Sparse Matrix Package
 */
class PastixSparseMatrix : public Operator
{
public:
   /**
    * @brief Constructs a distributed sparse PaStiX matrix
    * @param[in] hypParMat The HYPRE distributed matrix to
    * @b copy into the PaStiX matrix structure
    * @note The source matrix is not modified in any way
    */
   PastixSparseMatrix(const HypreParMatrix & hypParMat);

   /**
    * @brief De-allocates all heap-allocated memory in the
    * sparse matrix structure
    */
   ~PastixSparseMatrix();

   /**
    * @brief Matrix-vector multiplication
    * @param[in] x The vector operand
    * @param[out] y The result A*x, where A is the calling object
    */
   void Mult(const Vector &x, Vector &y) const override;

   /**
    * @brief Returns a reference to the underlying structure
    */
   spmatrix_t& InternalData() {return matrix_;}
   const spmatrix_t& InternalData() const {return matrix_;}

   /**
    * @brief Returns the internal MPI communicator
    */
   // MPI_Comm GetComm() const {return matrix_.comm;}
   MPI_Comm GetComm() const {return comm_;}

private:
   /**
    * @brief The underlying matrix structure
    * @see spmatrix_t
    */
   spmatrix_t matrix_;
   MPI_Comm comm_;
};

/**
 * @brief A wrapper for the PaStiX direct solver that conforms to the
 * mfem::Solver interface
 */
class PastixSolver : public Solver
{
public:
   /**
    * @brief Constructs a new solver object
    */
   PastixSolver();

   ~PastixSolver();

   /**
    * @brief Solves the system Ax=b for x
    * @param[in] b The right-hand-side vector
    * @param[out] x The solution vector
    * @pre The matrix A has been configured with SetOperator
    */
   void Mult( const Vector& b, Vector& x) const override;

   /**
    * @brief Sets the operator A for the solver
    * @param[in] op The system matrix A
    * @pre op must be of derived type PastixSparseMatrix
    */
   void SetOperator(const Operator& op) override;

   // Set various solver options. Refer to PaStiX documentation for details.

   /**
    * @brief Sets the verbosity (print level) for the solver
    * @param[in] verb Verbosity level
    * @note Options are PastixVerboseNot = 0 (nothing),
    * PastixVerboseNo = 1 (default), and PastixVerboseYes = 2 (extended)
    */
   void SetVerbosity(const pastix_verbose_e verb) {integer_params_[IPARM_VERBOSE] = verb;}

   /**
    * @brief Sets the iterative refinement method to use
    * @param[in] refine The method
    * @note Options are PastixRefineGMRES, PastixRefineCG, PastixRefineSR, PastixRefineBiCGSTAB
    */
   void SetIterativeRefinementMethod(const pastix_refine_e refine) {integer_params_[IPARM_REFINEMENT] = refine;}

private:
   pastix_int_t    integer_params_[IPARM_SIZE];
   double          double_params_[DPARM_SIZE];
   pastix_data_t  *pastix_data_ = nullptr;
   const PastixSparseMatrix* matrix_ = nullptr;
};

} // namespace mfem

#endif // MFEM_USE_PASTIX
#endif // MFEM_USE_MPI
#endif // MFEM_PASTIX
