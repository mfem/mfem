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

#ifndef MFEM_HANDLE_HPP
#define MFEM_HANDLE_HPP

#include "../config/config.hpp"
#include "operator.hpp"
#ifdef MFEM_USE_MPI
#include "hypre.hpp"
#endif

namespace mfem
{

/// Pointer to an Operator of a specified type
/** This class provides a common interface for global, matrix-type operators to
    be used in bilinear forms, gradients of nonlinear forms, static condensation,
    hybridization, etc. The following backends are currently supported:
      - HYPRE parallel sparse matrix (Hypre_ParCSR)
      - PETSC globally assembled parallel sparse matrix (PETSC_MATAIJ)
      - PETSC parallel matrix assembled on each processor (PETSC_MATIS)
    See also Operator::Type.
*/
class OperatorHandle
{
protected:
   static const char not_supported_msg[];

   Operator      *oper;
   Operator::Type type_id;
   bool           own_oper;

   Operator::Type CheckType(Operator::Type tid);

   template <typename OpType>
   void pSet(OpType *A, bool own_A = true)
   {
      oper = A;
      type_id = A->GetType();
      own_oper = own_A;
   }

public:
   /** @brief Create an OperatorHandle with type id = Operator::MFEM_SPARSEMAT
       without allocating the actual matrix. */
   OperatorHandle()
      : oper(NULL), type_id(Operator::MFEM_SPARSEMAT), own_oper(false) { }

   /** @brief Create a OperatorHandle with a specified type id, @a tid, without
       allocating the actual matrix. */
   explicit OperatorHandle(Operator::Type tid)
      : oper(NULL), type_id(CheckType(tid)), own_oper(false) { }

   /// Create an OperatorHandle for the given OpType pointer, @a A.
   /** Presently, OpType can be SparseMatrix, HypreParMatrix, or PetscParMatrix.

       The operator ownership flag is set to the value of @a own_A.

       It is expected that @a A points to a valid object. */
   template <typename OpType>
   explicit OperatorHandle(OpType *A, bool own_A = true) { pSet(A, own_A); }

   ~OperatorHandle() { if (own_oper) { delete oper; } }

   /// Shallow copy. The ownership flag of the target is set to false.
   OperatorHandle &operator=(const OperatorHandle &master)
   {
      Clear(); oper = master.oper; type_id = master.type_id; own_oper = false;
      return *this;
   }

   /// Access the underlying Operator pointer.
   Operator *Ptr() const { return oper; }

   /// Support the use of -> to call methods of the underlying Operator.
   Operator *operator->() const { return oper; }

   /// Access the underlying Operator.
   Operator &operator*() { return *oper; }

   /// Get the currently set operator type id.
   Operator::Type Type() const { return type_id; }

   /** @brief Return the Operator pointer statically cast to a specified OpType.
       Similar to the method Get(). */
   template <typename OpType>
   OpType *As() const { return static_cast<OpType*>(oper); }

   /// Return the Operator pointer dynamically cast to a specified OpType.
   template <typename OpType>
   OpType *Is() const { return dynamic_cast<OpType*>(oper); }

   /// Return the Operator pointer statically cast to a given OpType.
   /** Similar to the method As(), however the template type OpType can be
       derived automatically from the argument @a A. */
   template <typename OpType>
   void Get(OpType *&A) const { A = static_cast<OpType*>(oper); }

   /// Return true if the OperatorHandle owns the held Operator.
   bool OwnsOperator() const { return own_oper; }

   /// Set the ownership flag for the held Operator.
   void SetOperatorOwner(bool own = true) { own_oper = own; }

   /** @brief Clear the OperatorHandle, deleting the held Operator (if owned),
       while leaving the type id unchanged. */
   void Clear()
   {
      if (own_oper) { delete oper; }
      oper = NULL;
      own_oper = false;
   }

   /// Invoke Clear() and set a new type id.
   void SetType(Operator::Type tid)
   {
      Clear();
      type_id = CheckType(tid);
   }

   /// Reset the OperatorHandle to the given OpType pointer, @a A.
   /** Presently, OpType can be SparseMatrix, HypreParMatrix, or PetscParMatrix.

       The operator ownership flag is set to the value of @a own_A.

       It is expected that @a A points to a valid object. */
   template <typename OpType>
   void Reset(OpType *A, bool own_A = true)
   {
      if (own_oper) { delete oper; }
      pSet(A, own_A);
   }

#ifdef MFEM_USE_MPI
   /** @brief Reset the OperatorHandle to hold a parallel square block-diagonal
       matrix using the currently set type id. */
   /** The operator ownership flag is set to true. */
   void MakeSquareBlockDiag(MPI_Comm comm, HYPRE_Int glob_size,
                            HYPRE_Int *row_starts, SparseMatrix *diag);

   /** @brief Reset the OperatorHandle to hold a parallel rectangular
       block-diagonal matrix using the currently set type id. */
   /** The operator ownership flag is set to true. */
   void MakeRectangularBlockDiag(MPI_Comm comm, HYPRE_Int glob_num_rows,
                                 HYPRE_Int glob_num_cols, HYPRE_Int *row_starts,
                                 HYPRE_Int *col_starts, SparseMatrix *diag);
#endif // MFEM_USE_MPI

   /// Reset the OperatorHandle to hold the product @a P^t @a A @a P.
   /** The type id of the result is determined by that of @a A and @a P. The
       operator ownership flag is set to true. */
   void MakePtAP(OperatorHandle &A, OperatorHandle &P);

   /** @brief Reset the OperatorHandle to hold the product R @a A @a P, where
       R = @a Rt^t. */
   /** The type id of the result is determined by that of @a Rt, @a A, and @a P.
       The operator ownership flag is set to true. */
   void MakeRAP(OperatorHandle &Rt, OperatorHandle &A, OperatorHandle &P);

   /// Convert the given OperatorHandle @a A to the currently set type id.
   /** The operator ownership flag is set to false if the object held by @a A
       will be held by this object as well, e.g. when the source and destination
       types are the same; otherwise it is set to true. */
   void ConvertFrom(OperatorHandle &A);

   /// Convert the given OpType pointer, @a A, to the currently set type id.
   /** This method creates a temporary OperatorHandle for @a A and invokes
       ConvertFrom(OperatorHandle &) with it. */
   template <typename OpType>
   void ConvertFrom(OpType *A)
   {
      OperatorHandle Ah(A, false);
      ConvertFrom(Ah);
   }

   /** @brief Reset the OperatorHandle to be the eliminated part of @a A after
       elimination of the essential dofs @a ess_dof_list. */
   void EliminateRowsCols(OperatorHandle &A, const Array<int> &ess_dof_list);

   /// Eliminate the rows corresponding to the essential dofs @a ess_dof_list
   void EliminateRows(const Array<int> &ess_dof_list);

   /// Eliminate columns corresponding to the essential dofs @a ess_dof_list
   void EliminateCols(const Array<int> &ess_dof_list);

   /// Eliminate essential dofs from the solution @a X into the r.h.s. @a B.
   /** The argument @a A_e is expected to be the result of the method
       EliminateRowsCols(). */
   void EliminateBC(const OperatorHandle &A_e, const Array<int> &ess_dof_list,
                    const Vector &X, Vector &B) const;
};


/// Add an alternative name for OperatorHandle -- OperatorPtr.
typedef OperatorHandle OperatorPtr;

} // namespace mfem

#endif
