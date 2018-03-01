// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#ifndef MFEM_COMPLEX_OPERATOR
#define MFEM_COMPLEX_OPERATOR

#include "operator.hpp"
#include "sparsemat.hpp"

namespace mfem
{

/** @brief Mimic the action of a complex operator using two real operators.

    This operator requires vectors that are twice the length of its internally
    stored real operators, Op_Real and Op_Imag. It is assumed that these vectors
    store the real part of the vector first followed by its imaginary part.

    ComplexOperator allows one to choose a convention upon construction, which
    facilitates symmetry.

    Matrix-vector products are then computed as:

    1. When Convention::HERMITIAN is used (default)
    / y_r \   / Op_r -Op_i \ / x_r \
    |     | = |            | |     |
    \ y_i /   \ Op_i  Op_r / \ x_i /

    2. When Convention::BLOCK_SYMMETRIC is used
    / y_r \   / Op_r -Op_i \ / x_r \
    |     | = |            | |     |
    \-y_i /   \-Op_i -Op_r / \ x_i /

    Either convention can be used with a given complex operator,
    however, each of them is best suited for certain classes of
    problems.  For example:

    1. Convention::HERMITIAN, is well suited for Hermitian operators,
    i.e. operators where the real part is symmetric and the imaginary part of
    the operator is anti-symmetric, hence the name. In such cases the resulting
    2 x 2 operator will be symmetric.

    2. Convention::BLOCK_SYMMETRIC, is well suited for operators where both the
    real and imaginary parts are symmetric. In this case the resulting 2 x 2
    operator will again be symmetric. Such operators are common when studying
    damped oscillations, for example.

    Note: this class cannot be used to represent a general nonlinear complex
    operator.
 */
class ComplexOperator : public Operator
{
public:
   enum Convention
   {
      HERMITIAN,      ///< Native convention for Hermitian operators
      BLOCK_SYMMETRIC ///< Alternate convention for damping operators
   };

   /** @brief Constructs complex operator object

       Note that either @p Op_Real or @p Op_Imag can be NULL,
       thus eliminating their action (see documentation of the
       class for more details).

       In case ownership of the passed operator is transferred
       to this class through @p ownReal and @p ownImag,
       the operators will be explicitly destroyed at the end
       of the life of this object.
   */
   ComplexOperator(Operator * Op_Real, Operator * Op_Imag,
                   bool ownReal, bool ownImag,
                   Convention convention = HERMITIAN);

   virtual ~ComplexOperator();

   virtual void Mult(const Vector &x, Vector &y) const;
   virtual void MultTranspose(const Vector &x, Vector &y) const;

protected:
   // Let this be hidden from the public interface since the implementation
   // depends on internal members
   void Mult(const Vector &x_r, const Vector &x_i,
             Vector &y_r, Vector &y_i) const;
   void MultTranspose(const Vector &x_r, const Vector &x_i,
                      Vector &y_r, Vector &y_i) const;

protected:
   Operator * Op_Real_;
   Operator * Op_Imag_;

   bool ownReal_;
   bool ownImag_;

   Convention convention_;

   mutable Vector x_r_, x_i_, y_r_, y_i_;
   mutable Vector *u_, *v_;
};


/** @brief Specialization of the ComplexOperator built from a pair of Sparse
    Matrices.

    The purpose of this specialization is to construct a single SparseMatrix
    object which is equivalent to the 2x2 block system that the ComplexOperator
    mimics. The resulting SparseMatrix can then be passed along to solvers which
    require access to the CSR matrix data such as SuperLU, STRUMPACK, or similar
    sparse linear solvers.

    See ComplexOperator documentation in operator.hpp for more information.
 */
class ComplexSparseMatrix : public ComplexOperator
{
public:
   ComplexSparseMatrix(SparseMatrix * A_Real, SparseMatrix * A_Imag,
                       bool ownReal, bool ownImag,
                       Convention convention = HERMITIAN)
      : ComplexOperator(A_Real, A_Imag, ownReal, ownImag, convention)
   {}

   SparseMatrix * GetSystemMatrix() const;
};

}

#endif
