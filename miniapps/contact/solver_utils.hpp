// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
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
#include <functional>
#include <memory>
#include <vector>

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


/**
 * @class AMGFSchwarzSolver
 * @brief AMGF whose filtered subspace solver is an AdditiveSchwarz smoother
 *        with one patch per row of the gap Jacobian, instead of a direct
 *        solver.
 *
 * The patch of row @a i of the gap Jacobian @a J consists of the dofs of the
 * nonzeros of that row that lie in the filtered subspace defined by the
 * transfer operator @a P. Each MPI rank owns the patches of its local rows of
 * @a J.
 *
 * The solver is meant to precondition the reduced IP-Newton operator
 * K + J^T D J with diagonal D, and the patch of row @a i is skipped when
 * D_i is below a threshold. Since D changes in every IP-Newton iteration, the
 * patches are rebuilt from the current D in every call to SetOperator().
 */
class AMGFSchwarzSolver : public AMGFSolver
{
public:
   /**
    * @param J Gap Jacobian (not owned); only used during construction, so it
    *          need not outlive this object.
    * @param P Filtered subspace transfer operator (not owned); it must outlive
    *          this object.
    * @param get_D_ Fills its argument with the current diagonal of D, whose
    *               first J.Height() entries correspond to the local rows of
    *               @a J. It is copied, but anything it captures by reference
    *               must outlive this object.
    * @param D_threshold_ The patch of row @a i is skipped if D_i is below
    *                     this value.
    */
   AMGFSchwarzSolver(const HypreParMatrix &J, const HypreParMatrix &P,
                     std::function<void(Vector &)> get_D_,
                     real_t D_threshold_ = 0.0);

   /// Set the operator, and rebuild the Schwarz patches from the current D.
   void SetOperator(const Operator &op) override;

   /// Global number of Schwarz patches used after each SetOperator() call.
   /// The returned array is owned by this object.
   const Array<HYPRE_BigInt> & GetNumPatches() const { return num_patches; }

private:
   /// Communicator of the gap Jacobian.
   MPI_Comm comm;
   /// Global subspace dofs of the nonzeros of each local row of J.
   std::vector<Array<HYPRE_BigInt>> row_patches;
   /// Returns the current diagonal of D; see the constructor.
   std::function<void(Vector &)> get_D;
   /// The patch of row i of J is skipped if D_i is below this value.
   real_t D_threshold;
   /// Subspace solver (owned), rebuilt in every SetOperator() call.
   std::unique_ptr<AdditiveSchwarz> schwarz;
   /// Global number of patches used after each SetOperator() call.
   Array<HYPRE_BigInt> num_patches;
};


} // namespace mfem

#endif // MFEM_PARALLEL_DIRECT_SOLVER
