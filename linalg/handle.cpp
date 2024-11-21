// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "handle.hpp"
#include "sparsemat.hpp"
#ifdef MFEM_USE_MPI
#include "petsc.hpp"
#endif

// Make sure that hypre and PETSc use the same size indices.
#if defined(MFEM_USE_MPI) && defined(MFEM_USE_PETSC)
#if ((defined(HYPRE_BIGINT) || defined(HYPRE_MIXEDINT)) && !defined(PETSC_USE_64BIT_INDICES)) || \
    (!defined(HYPRE_BIGINT) && !defined(HYPRE_MIXEDINT) && defined(PETSC_USE_64BIT_INDICES))
#error HYPRE and PETSC do not use the same size integers!
#endif
#endif

namespace mfem
{

const char OperatorHandle::not_supported_msg[] =
   "Operator::Type is not supported: type_id = ";

Operator::Type OperatorHandle::CheckType(Operator::Type tid)
{
   switch (tid)
   {
      case Operator::ANY_TYPE: break;
      case Operator::MFEM_SPARSEMAT: break;
      case Operator::Hypre_ParCSR:
#ifdef MFEM_USE_MPI
         break;
#else
         MFEM_ABORT("cannot use HYPRE parallel matrix format: "
                    "MFEM is not built with HYPRE support");
#endif
      case Operator::PETSC_MATAIJ:
      case Operator::PETSC_MATIS:
      case Operator::PETSC_MATGENERIC:
#ifdef MFEM_USE_PETSC
         break;
#else
         MFEM_ABORT("cannot use PETSc matrix formats: "
                    "MFEM is not built with PETSc support");
#endif
      default:
         MFEM_ABORT("invalid Operator::Type, type_id = " << (int)type_id);
   }
   return tid;
}

#ifdef MFEM_USE_MPI
void OperatorHandle::MakeSquareBlockDiag(MPI_Comm comm, HYPRE_BigInt glob_size,
                                         HYPRE_BigInt *row_starts,
                                         SparseMatrix *diag)
{
   if (own_oper) { delete oper; }

   switch (type_id)
   {
      case Operator::ANY_TYPE: // --> MFEM_SPARSEMAT
      case Operator::MFEM_SPARSEMAT:
         // As a parallel Operator, the SparseMatrix simply represents the local
         // Operator, without the need of any communication.
         pSet(diag, false);
         return;
      case Operator::Hypre_ParCSR:
         oper = new HypreParMatrix(comm, glob_size, row_starts, diag);
         break;
#ifdef MFEM_USE_PETSC
      case Operator::PETSC_MATAIJ:
      case Operator::PETSC_MATIS:
         // Assuming that PetscInt is the same size as HYPRE_Int, checked above.
         oper = new PetscParMatrix(comm, glob_size, (PetscInt*)row_starts, diag,
                                   type_id);
         break;
#endif
      default: MFEM_ABORT(not_supported_msg << type_id);
   }
   own_oper = true;
}

void OperatorHandle::
MakeRectangularBlockDiag(MPI_Comm comm, HYPRE_BigInt glob_num_rows,
                         HYPRE_BigInt glob_num_cols, HYPRE_BigInt *row_starts,
                         HYPRE_BigInt *col_starts, SparseMatrix *diag)
{
   if (own_oper) { delete oper; }

   switch (type_id)
   {
      case Operator::ANY_TYPE: // --> MFEM_SPARSEMAT
      case Operator::MFEM_SPARSEMAT:
         // As a parallel Operator, the SparseMatrix simply represents the local
         // Operator, without the need of any communication.
         pSet(diag, false);
         return;
      case Operator::Hypre_ParCSR:
         oper = new HypreParMatrix(comm, glob_num_rows, glob_num_cols,
                                   row_starts, col_starts, diag);
         break;
#ifdef MFEM_USE_PETSC
      case Operator::PETSC_MATAIJ:
      case Operator::PETSC_MATIS:
         // Assuming that PetscInt is the same size as HYPRE_Int, checked above.
         oper = new PetscParMatrix(comm, glob_num_rows, glob_num_cols,
                                   (PetscInt*)row_starts, (PetscInt*)col_starts, diag, type_id);
         break;
#endif
      default: MFEM_ABORT(not_supported_msg << type_id);
   }
   own_oper = true;
}
#endif // MFEM_USE_MPI

void OperatorHandle::MakePtAP(OperatorHandle &A, OperatorHandle &P)
{
   if (A.Type() != Operator::ANY_TYPE)
   {
      MFEM_VERIFY(A.Type() == P.Type(), "type mismatch in A and P");
   }
   Clear();
   switch (A.Type())
   {
      case Operator::ANY_TYPE:
         pSet(new RAPOperator(*P.Ptr(), *A.Ptr(), *P.Ptr()));
         break;
      case Operator::MFEM_SPARSEMAT:
      {
         SparseMatrix *R  = mfem::Transpose(*P.As<SparseMatrix>());
         SparseMatrix *RA = mfem::Mult(*R, *A.As<SparseMatrix>());
         delete R;
         pSet(mfem::Mult(*RA, *P.As<SparseMatrix>()));
         delete RA;
         break;
      }
#ifdef MFEM_USE_MPI
      case Operator::Hypre_ParCSR:
         pSet(mfem::RAP(A.As<HypreParMatrix>(), P.As<HypreParMatrix>()));
         break;
#ifdef MFEM_USE_PETSC
      case Operator::PETSC_MATAIJ:
      case Operator::PETSC_MATIS:
      case Operator::PETSC_MATGENERIC:
      {
         pSet(mfem::RAP(A.As<PetscParMatrix>(), P.As<PetscParMatrix>()));
         break;
      }
#endif
#endif
      default: MFEM_ABORT(not_supported_msg << A.Type());
   }
}

void OperatorHandle::MakeRAP(OperatorHandle &Rt, OperatorHandle &A,
                             OperatorHandle &P)
{
   if (A.Type() != Operator::ANY_TYPE)
   {
      MFEM_VERIFY(A.Type() == Rt.Type(), "type mismatch in A and Rt");
      MFEM_VERIFY(A.Type() == P.Type(), "type mismatch in A and P");
   }
   Clear();
   switch (A.Type())
   {
      case Operator::ANY_TYPE:
         pSet(new RAPOperator(*Rt.Ptr(), *A.Ptr(), *P.Ptr()));
         break;
      case Operator::MFEM_SPARSEMAT:
      {
         pSet(mfem::RAP(*Rt.As<SparseMatrix>(), *A.As<SparseMatrix>(),
                        *P.As<SparseMatrix>()));
         break;
      }
#ifdef MFEM_USE_MPI
      case Operator::Hypre_ParCSR:
         pSet(mfem::RAP(Rt.As<HypreParMatrix>(), A.As<HypreParMatrix>(),
                        P.As<HypreParMatrix>()));
         break;
#ifdef MFEM_USE_PETSC
      case Operator::PETSC_MATAIJ:
      case Operator::PETSC_MATIS:
      case Operator::PETSC_MATGENERIC:
      {
         pSet(mfem::RAP(Rt.As<PetscParMatrix>(), A.As<PetscParMatrix>(),
                        P.As<PetscParMatrix>()));
         break;
      }
#endif
#endif
      default: MFEM_ABORT(not_supported_msg << A.Type());
   }
}

void OperatorHandle::ConvertFrom(OperatorHandle &A)
{
   if (own_oper) { delete oper; }
   if (Type() == A.Type() || Type() == Operator::ANY_TYPE)
   {
      oper = A.Ptr();
      own_oper = false;
      return;
   }
   oper = NULL;
   switch (Type()) // target type id
   {
      case Operator::MFEM_SPARSEMAT:
      {
         oper = A.Is<SparseMatrix>();
         break;
      }
      case Operator::Hypre_ParCSR:
      {
#ifdef MFEM_USE_MPI
         oper = A.Is<HypreParMatrix>();
#endif
         break;
      }
      case Operator::PETSC_MATAIJ:
      case Operator::PETSC_MATIS:
      {
         switch (A.Type()) // source type id
         {
            case Operator::Hypre_ParCSR:
#ifdef MFEM_USE_PETSC
               oper = new PetscParMatrix(A.As<HypreParMatrix>(), Type());
#endif
               break;
            default: break;
         }
#ifdef MFEM_USE_PETSC
         if (!oper)
         {
            PetscParMatrix *pA = A.Is<PetscParMatrix>();
            if (pA->GetType() == Type()) { oper = pA; }
         }
#endif
         break;
      }
      default: break;
   }
   MFEM_VERIFY(oper != NULL, "conversion from type id = " << A.Type()
               << " to type id = " << Type() << " is not supported");
   own_oper = true;
}

void OperatorHandle::EliminateRowsCols(OperatorHandle &A,
                                       const Array<int> &ess_dof_list)
{
   Clear();
   switch (A.Type())
   {
      case Operator::ANY_TYPE:
      {
         bool own_A = A.OwnsOperator();
         A.SetOperatorOwner(false);
         A.Reset(new ConstrainedOperator(A.Ptr(), ess_dof_list, own_A));
         // Keep this object empty - this will be OK if this object is only
         // used as the A_e parameter in a call to A.EliminateBC().
         break;
      }
      case Operator::MFEM_SPARSEMAT:
      {
         const Matrix::DiagonalPolicy preserve_diag = Matrix::DIAG_KEEP;
         SparseMatrix *sA = A.As<SparseMatrix>();
         SparseMatrix *Ae = new SparseMatrix(sA->Height());
         for (int i = 0; i < ess_dof_list.Size(); i++)
         {
            sA->EliminateRowCol(ess_dof_list[i], *Ae, preserve_diag);
         }
         Ae->Finalize();
         pSet(Ae);
         break;
      }
      case Operator::Hypre_ParCSR:
      {
#ifdef MFEM_USE_MPI
         pSet(A.As<HypreParMatrix>()->EliminateRowsCols(ess_dof_list));
#else
         MFEM_ABORT("type id = Hypre_ParCSR requires MFEM_USE_MPI");
#endif
         break;
      }
      case Operator::PETSC_MATAIJ:
      case Operator::PETSC_MATIS:
      case Operator::PETSC_MATGENERIC:
      {
#ifdef MFEM_USE_PETSC
         pSet(A.As<PetscParMatrix>()->EliminateRowsCols(ess_dof_list));
#else
         MFEM_ABORT("type id = Operator::PETSC_* requires MFEM_USE_PETSC");
#endif
         break;
      }
      default: MFEM_ABORT(not_supported_msg << A.Type());
   }
}

void OperatorHandle::EliminateRows(const Array<int> &ess_dof_list)
{
   switch (Type())
   {
      case Operator::Hypre_ParCSR:
      {
#ifdef MFEM_USE_MPI
         this->As<HypreParMatrix>()->EliminateRows(ess_dof_list);
#else
         MFEM_ABORT("type id = Hypre_ParCSR requires MFEM_USE_MPI");
#endif
         break;
      }
      default:
         MFEM_ABORT(not_supported_msg << Type());
   }
}

void OperatorHandle::EliminateCols(const Array<int> &ess_dof_list)
{
   switch (Type())
   {
      case Operator::Hypre_ParCSR:
      {
#ifdef MFEM_USE_MPI
         auto Ae = this->As<HypreParMatrix>()->EliminateCols(ess_dof_list);
         delete Ae;
#else
         MFEM_ABORT("type id = Hypre_ParCSR requires MFEM_USE_MPI");
#endif
         break;
      }
      default:
         MFEM_ABORT(not_supported_msg << Type());
   }
}

void OperatorHandle::EliminateBC(const OperatorHandle &A_e,
                                 const Array<int> &ess_dof_list,
                                 const Vector &X, Vector &B) const
{
   switch (Type())
   {
      case Operator::ANY_TYPE:
      {
         ConstrainedOperator *A = Is<ConstrainedOperator>();
         MFEM_VERIFY(A != NULL, "EliminateRowsCols() is not called");
         A->EliminateRHS(X, B);
         break;
      }
      case Operator::MFEM_SPARSEMAT:
      {
         A_e.As<SparseMatrix>()->AddMult(X, B, -1.);
         As<SparseMatrix>()->PartMult(ess_dof_list, X, B);
         break;
      }
      case Operator::Hypre_ParCSR:
      {
#ifdef MFEM_USE_MPI
         mfem::EliminateBC(*As<HypreParMatrix>(), *A_e.As<HypreParMatrix>(),
                           ess_dof_list, X, B);
#else
         MFEM_ABORT("type id = Hypre_ParCSR requires MFEM_USE_MPI");
#endif
         break;
      }
      case Operator::PETSC_MATAIJ:
      case Operator::PETSC_MATIS:
      case Operator::PETSC_MATGENERIC:
      {
#ifdef MFEM_USE_PETSC
         mfem::EliminateBC(*As<PetscParMatrix>(), *A_e.As<PetscParMatrix>(),
                           ess_dof_list, X, B);
#else
         MFEM_ABORT("type id = Operator::PETSC_* requires MFEM_USE_PETSC");
#endif
         break;
      }
      default: MFEM_ABORT(not_supported_msg << Type());
   }
}

} // namespace mfem
