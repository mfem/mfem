// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_BLOCK_STATIC_CONDENSATION
#define MFEM_BLOCK_STATIC_CONDENSATION

#include "../config/config.hpp"
#include "fespace.hpp"

#ifdef MFEM_USE_MPI
#include "pfespace.hpp"
#endif

namespace mfem
{


class BlockStaticCondensation
{
   int height, width;
   int nblocks;
   Mesh * mesh = nullptr;
   // original set of Finite Element Spaces
   Array<FiniteElementSpace *> fes;
   // indicates if the original space is already a trace space
   Array<bool> IsTraceSpace;

   // New set of "reduced" Finite Element Spaces
   // (after static condensation)
   Array<FiniteElementSpace *> tr_fes;

   // Schur complement matrix
   // S = A_ii - A_ib (A_bb)^{-1} A_bi.
   BlockMatrix * S = nullptr;
   BlockMatrix * S_e = nullptr;

   Array<int> rdof_edof;      // Map from reduced dofs to exposed dofs
   Array<int> ess_rtdof_list;

   // tr_dofs (element dof to global dof)
   // tr_dofs (element dof to global dof)
   void GetReduceElementIndices(int el, Array<int> & tr_dofs,
                                Array<int> & tr_ldofs);

public:

   BlockStaticCondensation(Array<FiniteElementSpace *> & fes_);

   ~BlockStaticCondensation();

   void SetSpaces(Array<FiniteElementSpace*> & fes_);

   void Init();

   /** Assemble the contribution to the Schur complement from the given
       element matrix 'elmat'; save the other blocks internally: A_bb_inv, A_bi,
       and A_bi. */

   void AssembleReducedSystem(int el, const DenseMatrix &elmat,
                              const Vector & elvect);

   /// Finalize the construction of the Schur complement matrix.
   void Finalize();

   /// Determine and save internally essential reduced true dofs.
   void SetEssentialTrueDofs(const Array<int> &ess_tdof_list);

   /// Eliminate the given reduced true dofs from the Schur complement matrix S.
   void EliminateReducedTrueDofs(const Array<int> &ess_rtdof_list,
                                 Matrix::DiagonalPolicy dpolicy);

   void EliminateReducedTrueDofs(Matrix::DiagonalPolicy dpolicy);

   bool HasEliminatedBC() const
   {
      return S_e;
   }

   /// Return the serial Schur complement matrix.
   BlockMatrix &GetMatrix() { return *S; }

   /// Return the eliminated part of the serial Schur complement matrix.
   BlockMatrix &GetMatrixElim() { return *S_e; }


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
   void ConvertListToReducedTrueDofs(const Array<int> &ess_tdof_list,
                                     Array<int> &ess_rtdof_list) const;

   /** Given a solution of the reduced system 'sc_sol' and the RHS 'b' for the
       full linear system, compute the solution of the full system 'sol'. */
   void ComputeSolution(const Vector &b, const Vector &sc_sol,
                        Vector &sol) const;

};

}

#endif
