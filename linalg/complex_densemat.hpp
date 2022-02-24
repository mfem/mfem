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

#ifndef MFEM_COMPLEX_DENSEMAT
#define MFEM_COMPLEX_DENSEMAT

#include "complex_operator.hpp"

namespace mfem
{

/** @brief Specialization of the ComplexOperator built from a pair of Dense
    Matrices.

    The purpose of this specialization is to support the inverse of a
    ComplexDenseMatrix and various MatMat operations

    See ComplexOperator documentation for more information.
    Note: Only the Hermitian convention is supported
 */
class ComplexDenseMatrix : public ComplexOperator
{

public:
   ComplexDenseMatrix(DenseMatrix * A_Real, DenseMatrix * A_Imag,
                      bool ownReal, bool ownImag)
      : ComplexOperator(A_Real, A_Imag, ownReal, ownImag)
   { }

   virtual DenseMatrix & real();
   virtual DenseMatrix & imag();

   virtual const DenseMatrix & real() const;
   virtual const DenseMatrix & imag() const;

   /** Combine the blocks making up this complex operator into a single
       DenseMatrix. Note that this combined operator requires roughly
       twice the memory of the block structured operator. */
   DenseMatrix * GetSystemMatrix() const;

   virtual Type GetType() const { return Complex_DenseMat; }

   ComplexDenseMatrix * ComputeInverse();

};

/// Matrix matrix multiplication.  A = B * C.
ComplexDenseMatrix * Mult(const ComplexDenseMatrix &B,
                          const ComplexDenseMatrix &C);

/// Multiply the Complex transpose of a matrix A with a matrix B:   Ah*B
ComplexDenseMatrix * MultAtB(const ComplexDenseMatrix &A,
                             const ComplexDenseMatrix &B);

} // namespace mfem

#endif // MFEM_COMPLEX_DENSEMAT
