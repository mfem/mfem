// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_COMPLEX_BLOCK_STATIC_CONDENSATION
#define MFEM_COMPLEX_BLOCK_STATIC_CONDENSATION

#include "mfem.hpp"

namespace mfem
{


/** @brief Class that performs static condensation of interior dofs for
    multiple FE spaces for complex systems (see BlockStaticCondensation). It's used
    by the class ComplexDPGWeakForm. */
class ComplexBlockStaticCondensation
{
   int height, width;
   int nblocks; // original number of blocks
   int rblocks; // reduces number of blocks
   Mesh * mesh = nullptr;
   bool parallel = false;
   // original set of Finite Element Spaces
   Array<FiniteElementSpace *> fes;
   // indicates if the original space is already a trace space
   Array<bool> IsTraceSpace;

   // New set of "reduced" Finite Element Spaces
   // (after static condensation)
   Array<FiniteElementSpace *> tr_fes;

   Array<int> dof_offsets;
   Array<int> tdof_offsets;

   Array<int> rdof_offsets;
   Array<int> rtdof_offsets;

   // Schur complement matrix
   // S = A_ii - A_ib (A_bb)^{-1} A_bi.
   BlockMatrix * S_r = nullptr;
   BlockMatrix * S_i = nullptr;
   BlockMatrix * S_e_r = nullptr;
   BlockMatrix * S_e_i = nullptr;
   ComplexOperator * S = nullptr;

   BlockVector * y_r = nullptr;
   BlockVector * y_i = nullptr;
   Vector * y = nullptr;

   Array<ComplexDenseMatrix * > lmat;
   Array<Vector * > lvec;

   Array<int> rdof_edof;      // Map from reduced dofs to exposed dofs
   Array<int> ess_rtdof_list;

   BlockMatrix * P = nullptr; // Block Prolongation
   BlockMatrix * R = nullptr; // Block Restriction

#ifdef MFEM_USE_MPI
   BlockOperator * pS_r = nullptr;
   BlockOperator * pS_e_r = nullptr;
   BlockOperator * pS_i = nullptr;
   BlockOperator * pS_e_i = nullptr;
   // Block HypreParMatrix for Prolongation
   BlockOperator * pP = nullptr;
#endif

   bool Parallel() const { return parallel; }


   // tr_idx (trace dofs indices)
   // int_idx (interior dof indices)
   void GetReduceElementIndicesAndOffsets(int el, Array<int> & tr_idx,
                                          Array<int> & int_idx,
                                          Array<int> & offsets) const;

   void GetReduceElementVDofs(int el, Array<int> & rdofs) const;
   void GetElementVDofs(int el, Array<int> & vdofs) const;


   //  S = A_ii - A_ib (A_bb)^{-1} A_bi.
   //  y = y_i - A_ib (A_bb)^{-1} y_b
   ComplexDenseMatrix * GetLocalShurComplement(int el, const Array<int> & tr_idx,
                                               const Array<int> & int_idx,
                                               const ComplexDenseMatrix & elmat,
                                               const Vector & elvect_r,
                                               const Vector & elvect_i,
                                               Vector & rvect_r,
                                               Vector & rvect_i);

   void ComputeOffsets();

   void BuildProlongation();
#ifdef MFEM_USE_MPI
   void BuildParallelProlongation();
#endif

   // ess_tdof list for each space
   Array<Array<int> *> ess_tdofs;
   void FillEssTdofLists(const Array<int> & ess_tdof_list);

   void ConformingAssemble(int skip_zeros);

   /** Restrict a marker Array on the true FE spaces dofs to a marker Array on
    the reduced/trace true FE spaces dofs. */
   void ConvertMarkerToReducedTrueDofs(Array<int> & tdof_marker,
                                       Array<int> & rtdof_marker);

   void SetSpaces(Array<FiniteElementSpace*> & fes_);

   void Init();

public:

   ComplexBlockStaticCondensation(Array<FiniteElementSpace *> & fes_);

   ~ComplexBlockStaticCondensation();

   /** Assemble the contribution to the Schur complement from the given
       element matrix @a elmat. Save the other blocks internally: A_bb_inv, A_bi,
       and A_bi. */
   void AssembleReducedSystem(int el, ComplexDenseMatrix &elmat,
                              Vector & elvect_r, Vector & elvect_i);

   /// Finalize the construction of the Schur complement matrix.
   void Finalize(int skip_zeros = 0);

   /// Determine and save internally essential reduced true dofs.
   void SetEssentialTrueDofs(const Array<int> &ess_tdof_list);

   /// Eliminate the given reduced true dofs from the Schur complement matrix S.
   void EliminateReducedTrueDofs(const Array<int> &ess_rtdof_list,
                                 Matrix::DiagonalPolicy dpolicy);

   bool HasEliminatedBC() const
   {
#ifndef MFEM_USE_MPI
      return S_e_r;
#else
      return S_e_r || pS_e_r;
#endif

   }

   /// Return the serial Schur complement matrix.
   BlockMatrix &GetSchurMatrix_r() { return *S_r; }
   BlockMatrix &GetSchurMatrix_i() { return *S_i; }
   ComplexOperator &GetSchurComplexOperator()
   {
      if (!S)
      {
#ifndef MFEM_USE_MPI
         S = new ComplexOperator(S_r,S_i,false,false);
#else
         if (parallel)
         {
            S = new ComplexOperator(pS_r,pS_i,false,false);
         }
         else
         {
            S = new ComplexOperator(S_r,S_i,false,false);
         }
#endif
      }
      return *S;
   }

   /// Return the eliminated part of the serial Schur complement matrix.
   BlockMatrix &GetSchurMatrixElim_r() { return *S_e_r; }
   BlockMatrix &GetSchurMatrixElim_i() { return *S_e_i; }

#ifdef MFEM_USE_MPI
   /// Return the parallel Schur complement matrix.
   BlockOperator &GetParallelSchurMatrix_r() { return *pS_r; }
   BlockOperator &GetParallelSchurMatrix_i() { return *pS_i; }

   /// Return the eliminated part of the parallel Schur complement matrix.
   BlockOperator &GetParallelSchurMatrixElim_r() { return *pS_e_r; }
   BlockOperator &GetParallelSchurMatrixElim_i() { return *pS_e_i; }

   void ParallelAssemble(BlockMatrix *m_r, BlockMatrix*m_i);
#endif

   /** Form the global reduced system matrix using the given @a diag_policy.
       This method can be called after Assemble() is called. */
   void FormSystemMatrix(Operator::DiagonalPolicy diag_policy);

   /** Restrict a solution vector on the full FE space dofs to a vector on the
       reduced/trace true FE space dofs. */
   void ReduceSolution(const Vector &sol, Vector &sc_sol) const;

   /** @brief Set the reduced solution `X` and r.h.s `B` vectors from the full
    linear system solution `x` and r.h.s. `b` vectors.
      This method should be called after the internal reduced essential dofs
      have been set using SetEssentialTrueDofs() and both the Schur complement
      and its eliminated part have been finalized. */
   void ReduceSystem(Vector &x, Vector &X, Vector &B,
                     int copy_interior = 0) const;

   /** Restrict a list of true FE space dofs to a list of reduced/trace true FE
    space dofs. */
   void ConvertListToReducedTrueDofs(const Array<int> &ess_tdof_list,
                                     Array<int> &ess_rtdof_list) const;

   /** Given a solution of the reduced system 'sc_sol' and the RHS 'b' for the
       full linear system, compute the solution of the full system 'sol'. */
   void ComputeSolution(const Vector &sc_sol, Vector &sol) const;

};

}

#endif
