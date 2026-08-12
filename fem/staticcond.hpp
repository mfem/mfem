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

#ifndef MFEM_STATIC_CONDENSATION
#define MFEM_STATIC_CONDENSATION

#include "../config/config.hpp"
#include "fespace.hpp"

#ifdef MFEM_USE_MPI
#include "pfespace.hpp"
#endif

namespace mfem
{

/** Auxiliary class StaticCondensation, used to implement static condensation
    in class BilinearForm.

    Static condensation is a technique for solving linear systems by eliminating
    groups/blocks of unknowns and reducing the original system to the remaining
    interfacial unknowns. The assumption is that unknowns in one group are
    connected (in the graph of the matrix) only to unknowns in the same group
    or to interfacial unknowns but not to other groups.

    For finite element systems, the groups correspond to degrees of freedom
    (DOFs) associated with the interior of the elements. The rest of the DOFs
    (associated with the element boundaries) are interfacial.

    In block form the matrix of the system can be written as
       $$ A =
       \begin{pmatrix}
          A_{11} & A_{12} \\
          A_{21} & A_{22}
       \end{pmatrix}
       \begin{array}{l}
          \text{- groups: element interior/private DOFs} \\
          \text{- interface: element boundary/exposed DOFs}
       \end{array} $$
    where the block $ A_1 $ is itself block diagonal with small local blocks
    and it is, therefore, easily invertible.

    Starting with the block system
       $$ \begin{pmatrix}
          A_{11} & A_{12} \\
          A_{21} & A_{22}
       \end{pmatrix}
       \begin{pmatrix} X_1 \\ X_2 \end{pmatrix} =
       \begin{pmatrix} B_1 \\ B_2 \end{pmatrix} $$
    the reduced, statically condensed system is given by
        $$ S_{22} X_2 = B_2 - A_{21} A_{11}^{-1} B_1 $$
    where the Schur complement matrix $ S_{22} $ is given by
        $$ S_{22} = A_{22} - A_{21} A_{11}^{-1} A_{12}. $$
    After solving the Schur complement system, the $ X_1 $ part of the
    solution can be recovered using the formula
        $$ X_1 = A_{11}^{-1} ( B_1 - A_{12} X_2 ). $$ */
class StaticCondensation
{
   FiniteElementSpace *fes, *tr_fes;
   FiniteElementCollection *tr_fec;
   Table elem_pdof;           // Element to private dof
   int npdofs;                // Number of private dofs
   Array<int> rdof_edof;      // Map from reduced dofs to exposed dofs

   // Schur complement: S = A_ee - A_ep (A_pp)^{-1} A_pe.
   SparseMatrix *S, *S_e;
#ifdef MFEM_USE_MPI
   ParFiniteElementSpace *pfes, *tr_pfes;
   OperatorHandle pS, pS_e;
   bool Parallel() const { return (tr_pfes != NULL); }
#else
   bool Parallel() const { return false; }
#endif

   bool symm; // TODO: handle the symmetric case correctly.
   Array<int> A_offsets, A_ipiv_offsets;
   Memory<real_t> A_data;
   Memory<int> A_ipiv;

   Array<int> ess_rtdof_list;

public:
   /// Construct a StaticCondensation object.
   StaticCondensation(FiniteElementSpace *fespace);
   /// Destroy a StaticCondensation object.
   ~StaticCondensation();

   /// Return the number of vector private dofs.
   int GetNPrDofs() const { return npdofs; }
   /// Return the number of vector exposed/reduced dofs.
   int GetNExDofs() const { return tr_fes->GetVSize(); }
   /** Return true if applying the static condensation actually reduces the
       (global) number of true vector dofs. */
   bool ReducesTrueVSize() const;

   /** Prepare the StaticCondensation object to assembly: allocate the Schur
       complement matrix and the other element-wise blocks. */
   void Init(bool symmetric, bool block_diagonal);

   /// Return a pointer to the reduced/trace FE space.
   FiniteElementSpace *GetTraceFESpace() { return tr_fes; }

#ifdef MFEM_USE_MPI
   /// Return a pointer to the parallel reduced/trace FE space.
   ParFiniteElementSpace *GetParTraceFESpace() { return tr_pfes; }
#endif
   /** Assemble the contribution to the Schur complement from the given
       element matrix 'elmat'; save the other blocks internally: A_pp_inv, A_pe,
       and A_ep. */
   void AssembleMatrix(int el, const DenseMatrix &elmat);

   /** Assemble the contribution to the Schur complement from the given boundary
       element matrix 'elmat'. */
   void AssembleBdrMatrix(int el, const DenseMatrix &elmat);

   /// Finalize the construction of the Schur complement matrix.
   void Finalize();

   /// Determine and save internally essential reduced true dofs.
   void SetEssentialTrueDofs(const Array<int> &ess_tdof_list)
   { ConvertListToReducedTrueDofs(ess_tdof_list, ess_rtdof_list); }

   /// Eliminate the given reduced true dofs from the Schur complement matrix S.
   void EliminateReducedTrueDofs(const Array<int> &ess_rtdof_list,
                                 Matrix::DiagonalPolicy dpolicy);

   /// @brief Eliminate the internal reduced true dofs (set using
   /// SetEssentialTrueDofs()) from the Schur complement matrix S.
   void EliminateReducedTrueDofs(Matrix::DiagonalPolicy dpolicy)
   { EliminateReducedTrueDofs(ess_rtdof_list, dpolicy); }

   /** @brief Return true if essential boundary conditions have been eliminated
       from the Schur complement matrix. */
   bool HasEliminatedBC() const
   {
#ifndef MFEM_USE_MPI
      return S_e;
#else
      return S_e || pS_e.Ptr();
#endif
   }

   /// Return the serial Schur complement matrix.
   SparseMatrix &GetMatrix() { return *S; }

   /// Return the eliminated part of the serial Schur complement matrix.
   SparseMatrix &GetMatrixElim() { return *S_e; }

#ifdef MFEM_USE_MPI
   /// Return the parallel Schur complement matrix.
   HypreParMatrix &GetParallelMatrix() { return *pS.Is<HypreParMatrix>(); }

   /// Return the eliminated part of the parallel Schur complement matrix.
   HypreParMatrix &GetParallelMatrixElim()
   { return *pS_e.Is<HypreParMatrix>(); }

   /** @brief Return the parallel Schur complement matrix in the format
       specified by SetOperatorType(). */
   void GetParallelMatrix(OperatorHandle &S_h) const { S_h = pS; }

   /** @brief Return the eliminated part of the parallel Schur complement matrix
       in the format specified by SetOperatorType(). */
   void GetParallelMatrixElim(OperatorHandle &S_e_h) const { S_e_h = pS_e; }

   /// Set the operator type id for the parallel reduced matrix/operator.
   void SetOperatorType(Operator::Type tid)
   { pS.SetType(tid); pS_e.SetType(tid); }
#endif

   /** Given a RHS vector for the full linear system, compute the RHS for the
       reduced linear system: sc_b = b_e - A_ep A_pp_inv b_p. */
   void ReduceRHS(const Vector &b, Vector &sc_b) const;

   /** Restrict a solution vector on the full FE space dofs to a vector on the
       reduced/trace true FE space dofs. */
   void ReduceSolution(const Vector &sol, Vector &sc_sol) const;

   /** @brief Set the reduced solution `X` and r.h.s `B` vectors from the full
       linear system solution `x` and r.h.s. `b` vectors.

       This method should be called after the internal reduced essential dofs
       have been set using SetEssentialTrueDofs() and both the Schur complement
       and its eliminated part have been finalized. */
   void ReduceSystem(Vector &x, Vector &b, Vector &X, Vector &B,
                     int copy_interior = 0) const;

   /** Restrict a marker Array on the true FE space dofs to a marker Array on
       the reduced/trace true FE space dofs. */
   void ConvertMarkerToReducedTrueDofs(const Array<int> &ess_tdof_marker,
                                       Array<int> &ess_rtdof_marker) const;

   /** Restrict a list of true FE space dofs to a list of reduced/trace true FE
       space dofs. */
   void ConvertListToReducedTrueDofs(const Array<int> &ess_tdof_list_,
                                     Array<int> &ess_rtdof_list_) const
   {
      Array<int> ess_tdof_marker, ess_rtdof_marker;
      FiniteElementSpace::ListToMarker(ess_tdof_list_, fes->GetTrueVSize(),
                                       ess_tdof_marker);
      ConvertMarkerToReducedTrueDofs(ess_tdof_marker, ess_rtdof_marker);
      FiniteElementSpace::MarkerToList(ess_rtdof_marker, ess_rtdof_list_);
   }

   /** Given a solution of the reduced system 'sc_sol' and the RHS 'b' for the
       full linear system, compute the solution of the full system 'sol'. */
   void ComputeSolution(const Vector &b, const Vector &sc_sol,
                        Vector &sol) const;
};

}

#endif
