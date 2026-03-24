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

#ifndef MFEM_PARALLEL_DIRECT_SOLVER
#define MFEM_PARALLEL_DIRECT_SOLVER

#include "mfem.hpp"


namespace mfem
{

/**
 * @class ParallelDirectSolver
 * @brief Wrapper around parallel sparse direct solvers (MUMPS, SuperLU_DIST,
 *        STRUMPACK, CPARDISO).
 *
 * ParallelDirectSolver provides a uniform interface to several parallel sparse
 * direct solvers. The solver is selected at
 * runtime via the Type enum or a string name, while the actual availability
 * of each backend depends on how MFEM was configured and built.
 *
 * Supported backends (if enabled at configure time):
 *  - MUMPS      (MFEM_USE_MUMPS)
 *  - SuperLU    (MFEM_USE_SUPERLU)
 *  - STRUMPACK  (MFEM_USE_STRUMPACK)
 *  - CPARDISO   (MFEM_USE_MKL_CPARDISO)
 *
 * The typical usage pattern is:
 *  - Construct a ParallelDirectSolver with an MPI communicator and a backend.
 *  - Call SetOperator() with the system operator (usually a HypreParMatrix).
 *  - Call Mult() to apply the inverse (i.e. solve).
 *
 */
class ParallelDirectSolver : public Solver
{
public:
   /**
    * @brief Type of parallel direct solver to use.
    *
    * AUTO selects the first available backend at runtime according to an
    * internal priority order.
    */
   enum class Type
   {
      AUTO,
      MUMPS,
      SUPERLU,
      CPARDISO,
      STRUMPACK
   };

private:
   /// Selected solver type
   Type type;
   /// MPI communicator
   MPI_Comm comm;

#ifdef MFEM_USE_SUPERLU
   /// Row-local matrix for SuperLU.
   mutable std::unique_ptr<SuperLURowLocMatrix> superlu_mat;
#endif

#ifdef MFEM_USE_STRUMPACK
   /// Row-local matrix for STRUMPACK.
   mutable std::unique_ptr<STRUMPACKRowLocMatrix> strumpack_mat;
#endif

   /// Owning pointer to the underlying backend solver.
   std::unique_ptr<Solver> solver;

   /// Helper that constructs the underlying backend solver based on #type.
   void InitSolver();

public:
   /**
    * @brief Construct a ParallelDirectSolver from an MPI communicator and a Type.
    *
    * If @p type_ is Type::AUTO, the constructor will select the first
    * available backend according to an internal priority order and store the
    * resolved type in #type. No factorization is performed here; that happens
    * when SetOperator() is called.
    */
   ParallelDirectSolver(MPI_Comm comm_, Type type_ = Type::AUTO);
   /**
    * @brief Construct a ParallelDirectSolver from a string name.
    * Accepted strings:
    *   "auto", "mumps", "superlu", "cpardiso", "strumpack".
    * The string is converted to a Type and the other constructor is invoked.
    */
   ParallelDirectSolver(MPI_Comm comm_, const std::string &name);

   /// Virtual destructor. The underlying solver and any auxiliary matrices
   /// (SuperLURowLocMatrix, STRUMPACKRowLocMatrix) are destroyed automatically.
   virtual ~ParallelDirectSolver() { }

   /**
    * @brief Set the operator to be factored/solved by the direct solver.
    *
    * The expected dynamic type of @p op depends on the chosen backend:
    *  - MUMPS / CPARDISO :
    *      - Usually expects a HypreParMatrix (assembled parallel sparse matrix).
    *  - SUPERLU :
    *      - A SuperLURowLocMatrix is constructed internally from @p op and
    *        kept in #superlu_mat. The SuperLUSolver then uses this row-local
    *        representation.
    *  - STRUMPACK :
    *     - A STRUMPACKRowLocMatrix is constructed internally from @p op and
    *       kept in #strumpack_mat. The STRUMPACKSolver then uses this row-local
    *       representation.
    *
    * @param op The operator representing the system matrix to factor.
    */
   virtual void SetOperator(const Operator &op) override;

   /// Apply the inverse of the operator: y = A^{-1} x.
   virtual void Mult(const Vector &x, Vector &y) const override;

   virtual void SetPrintLevel(int print_lvl);

};


} // namespace mfem

#endif // MFEM_PARALLEL_DIRECT_SOLVER
