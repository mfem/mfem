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

#ifndef MFEM_BLOCKFORM
#define MFEM_BLOCKFORM

#include "../config/config.hpp"

#include "fespace.hpp"
#include "gridfunc.hpp"
#include "bilinearform.hpp"

namespace mfem
{
// square block forms
class BlockForm
{

private:
   int height, width;
   int nblocks;
   Array<int> dof_offsets;
   Array<int> tdof_offsets;
   // BilinearForms
   Array2D<BilinearForm * > bforms;
   Array2D<MixedBilinearForm * > mforms;
   Array<FiniteElementSpace *> fes;

   // Block Prolongation
   BlockMatrix * P = nullptr;
   // Block Restriction
   BlockMatrix * R = nullptr;

   BlockMatrix * mat = nullptr;
   BlockMatrix * mat_e = nullptr;

   void BuildProlongation();
   void ConformingAssemble();

   mfem::Operator::DiagonalPolicy diag_policy;

public:

   BlockForm(const Array<FiniteElementSpace*> pfes_ );

   void SetBlock(BilinearForm * bform, int row_idx, int col_idx);
   void SetBlock(MixedBilinearForm * mform, int row_idx, int col_idx);

   /// Assemble the local matrix
   void Assemble(int skip_zeros = 1);

   void FormLinearSystem(const Array<int> &ess_tdof_list, Vector &x, Vector & b,
                         OperatorHandle &A, Vector &X,
                         Vector &B, int copy_interior = 0);

   void FormSystemMatrix(const Array<int> &ess_tdof_list,
                         OperatorHandle &A);

   void RecoverFEMSolution(const Vector &X, Vector &x);

   ///  Finalizes the matrix initialization.
   void Finalize(int skip_zeros = 1);

   void EliminateVDofs(const Array<int> &vdofs,
                       Operator::DiagonalPolicy dpolicy = Operator::DIAG_ONE);

   void EliminateVDofsInRHS(const Array<int> &vdofs, const Vector &x, Vector &b);

   void SetDiagonalPolicy(Operator::DiagonalPolicy policy)
   {
      diag_policy = policy;
   }

   /// Destroys bilinear form.
   ~BlockForm();
};

}

#endif