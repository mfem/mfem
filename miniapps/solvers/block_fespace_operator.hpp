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

#ifndef MFEM_BLOCK_FESPACE_OPERATOR
#define MFEM_BLOCK_FESPACE_OPERATOR

#include "mfem.hpp"

namespace mfem
{

/// @brief Operator for block systems arising from different arbitrarily many finite element spaces.
///
///        This operator can be used with FormLinearSystem to impose boundary
///        conditions for block systems arise from mixing many types of
///        finite element spaces. Each block is intended to operate on
///        L-Vectors. For example, a block may be a BilinearForm.
class BlockFESpaceOperator : public Operator
{
   using FESVector = std::vector<const FiniteElementSpace*>;

   /// Offsets for the square "A" operator.
   Array<int> offsets;
   /// Column offsets for the prolongation operator.
   Array<int> prolongColOffsets;
   /// Row offsets for the prolongation operator.
   Array<int> restrictRowOffsets;
   /// The "A" part of "RAP".
   BlockOperator A;
   /// Maps local dofs of each block to true dofs.
   BlockOperator prolongation;
   /// Maps true dofs of each block to local dofs.
   BlockOperator restriction;
   /// Computes height for parent operator.
   static int GetHeight(const FESVector &fespaces);
   /// Computes offsets for A BlockOperator.
   static Array<int> GetBlockOffsets(const FESVector &fespaces);
   /// Computes col_offsets for prolongation operator.
   static Array<int> GetProColBlockOffsets(const FESVector &fespaces);
   /// Computes row_offsets for restriction operator.
   static Array<int> GetResRowBlockOffsets(const FESVector &fespaces);

public:
   /// @brief Constructor for BlockFESpaceOperator.
   /// @param[in] fespaces Finite element spaces for diagonal blocks. Spaces are not owned.
   BlockFESpaceOperator(const FESVector &fespaces);
   const Operator* GetProlongation () const override;
   const Operator* GetRestriction () const override;
   void Mult(const Vector &x, Vector &y) const override {A.Mult(x,y);};
   /// @brief Wraps BlockOperator::SetBlock. Eventually would like this class to inherit
   /// from BlockOperator instead, but can't easily due to ownership of offset data
   /// in BlockOperator being by reference.
   void SetBlock(int iRow, int iCol, Operator *op, real_t c = 1.0) { A.SetBlock(iRow, iCol, op, c); };
};

} // namespace mfem

#endif // MFEM_BLOCK_FESPACE_OPERATOR
