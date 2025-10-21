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

#ifndef MFEM_COMPLEX_OPERATOR
#define MFEM_COMPLEX_OPERATOR

#include "operator.hpp"
#include "sparsemat.hpp"
#ifdef MFEM_USE_MPI
#include "hypre.hpp"
#endif

#ifdef MFEM_USE_SUITESPARSE
#include <umfpack.h>
#endif

namespace mfem
{

/** @brief Mimic the action of a complex operator using two real operators.

    This operator requires vectors that are twice the length of its internally
    stored real operators, Op_Real and Op_Imag. It is assumed that these vectors
    store the real part of the vector first followed by its imaginary part.

    ComplexOperator allows one to choose a convention upon construction, which
    facilitates symmetry.

    If we let (y_r + i y_i) = (Op_r + i Op_i)(x_r + i x_i) then Matrix-vector
    products are computed as:

    1. When Convention::HERMITIAN is used (default)
    / y_r \   / Op_r -Op_i \ / x_r \
    |     | = |            | |     |
    \ y_i /   \ Op_i  Op_r / \ x_i /

    2. When Convention::BLOCK_SYMMETRIC is used
    / y_r \   / Op_r -Op_i \ / x_r \
    |     | = |            | |     |
    \-y_i /   \-Op_i -Op_r / \ x_i /
    In other words, Matrix-vector products with Convention::BLOCK_SYMMETRIC
    compute the complex conjugate of Op*x.

    Either convention can be used with a given complex operator, however, each
    of them may be best suited for different classes of problems. For example:

    1. Convention::HERMITIAN, is well suited for Hermitian operators, i.e.,
       operators where the real part is symmetric and the imaginary part of the
       operator is anti-symmetric, hence the name. In such cases the resulting 2
       x 2 operator will be symmetric.

    2. Convention::BLOCK_SYMMETRIC, is well suited for operators where both the
       real and imaginary parts are symmetric. In this case the resulting 2 x 2
       operator will also be symmetric. Such operators are common when studying
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

       Note that either @p Op_Real or @p Op_Imag can be NULL, thus eliminating
       their action (see documentation of the class for more details).

       In case ownership of the passed operator is transferred to this class
       through @p ownReal and @p ownImag, the operators will be explicitly
       destroyed at the end of the life of this object.
   */
   ComplexOperator(Operator * Op_Real, Operator * Op_Imag,
                   bool ownReal, bool ownImag,
                   Convention convention = HERMITIAN);

   virtual ~ComplexOperator();

   /** @brief Check for existence of real or imaginary part of the operator

       These methods do not check that the operators are non-zero but only that
       the operators have been set.
    */
   bool hasRealPart() const { return Op_Real_ != NULL; }
   bool hasImagPart() const { return Op_Imag_ != NULL; }

   /** @brief Real or imaginary part accessor methods

       The following accessor methods should only be called if the requested
       part of the operator is known to exist. This can be checked with
       hasRealPart() or hasImagPart().
   */
   virtual Operator & real();
   virtual Operator & imag();
   virtual const Operator & real() const;
   virtual const Operator & imag() const;

   void Mult(const Vector &x, Vector &y) const override;
   void MultTranspose(const Vector &x, Vector &y) const override;

   using Operator::Mult;
   using Operator::MultTranspose;

   virtual Type GetType() const { return Complex_Operator; }

   Convention GetConvention() const { return convention_; }

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

    See ComplexOperator documentation above for more information.
 */
class ComplexSparseMatrix : public ComplexOperator
{
public:
   ComplexSparseMatrix(SparseMatrix * A_Real, SparseMatrix * A_Imag,
                       bool ownReal, bool ownImag,
                       Convention convention = HERMITIAN)
      : ComplexOperator(A_Real, A_Imag, ownReal, ownImag, convention)
   {}

   SparseMatrix & real() override;
   SparseMatrix & imag() override;

   const SparseMatrix & real() const override;
   const SparseMatrix & imag() const override;

   /** Combine the blocks making up this complex operator into a single
       SparseMatrix. The resulting matrix can be passed to solvers which require
       access to the matrix entries themselves, such as sparse direct solvers,
       rather than simply the action of the operator. Note that this combined
       operator requires roughly twice the memory of the block structured
       operator. */
   SparseMatrix * GetSystemMatrix() const;

   Type GetType() const override { return MFEM_ComplexSparseMat; }
};

#ifdef MFEM_USE_SUITESPARSE
/** @brief Interface with UMFPack solver specialized for ComplexSparseMatrix
    This approach avoids forming a monolithic SparseMatrix which leads
    to increased memory and flops
 */
class ComplexUMFPackSolver : public Solver
{
protected:
   bool use_long_ints;
   bool transa;
   ComplexSparseMatrix *mat;

   void *Numeric;
   SuiteSparse_long *AI, *AJ;

   void Init();

public:
   real_t Control[UMFPACK_CONTROL];
   mutable real_t Info[UMFPACK_INFO];

   /** @brief For larger matrices, if the solver fails, set the parameter @a
       use_long_ints_ = true. */
   ComplexUMFPackSolver(bool use_long_ints_ = false, bool transa_ = false)
      : use_long_ints(use_long_ints_), transa(transa_) { Init(); }
   /** @brief Factorize the given ComplexSparseMatrix using the defaults.
       For larger matrices, if the solver fails, set the parameter
       @a use_long_ints_ = true. */
   ComplexUMFPackSolver(ComplexSparseMatrix &A, bool use_long_ints_ = false,
                        bool transa_ = false)
      : use_long_ints(use_long_ints_), transa(transa_) { Init(); SetOperator(A); }

   /** @brief Factorize the given Operator @a op which must be
       a ComplexSparseMatrix.

       The factorization uses the parameters set in the #Control data member.
       @note This method calls SparseMatrix::SortColumnIndices()
       for real and imag parts of the ComplexSparseMatrix,
       modifying the matrices if the column indices are not already sorted. */
   void SetOperator(const Operator &op) override;

   // Set the print level field in the #Control data member.
   void SetPrintLevel(int print_lvl) { Control[UMFPACK_PRL] = print_lvl; }

   // This determines the action of MultTranspose (see below for details)
   void SetTransposeSolve(bool transa_) { transa = transa_; }

   /** @brief This is solving the system A x = b */
   void Mult(const Vector &b, Vector &x) const override;

   /** @brief
      This is solving the system:
      A^H x = b (when transa = false)
      This is equivalent to solving the transpose block system for the
      case of Convention = HERMITIAN
      A^T x = b (when transa = true)
      This is equivalent to solving the transpose block system for the
      case of Convention = BLOCK_SYMMETRIC */
   void MultTranspose(const Vector &b, Vector &x) const override;

   virtual ~ComplexUMFPackSolver();
};

#endif

#ifdef MFEM_USE_MPI

/** @brief Specialization of the ComplexOperator built from a pair of
    HypreParMatrices.

    The purpose of this specialization is to construct a single HypreParMatrix
    object which is equivalent to the 2x2 block system that the ComplexOperator
    mimics. The resulting HypreParMatrix can then be passed along to solvers
    which require access to the CSR matrix data such as SuperLU, STRUMPACK, or
    similar sparse linear solvers.

    See ComplexOperator documentation above for more information.
 */
class ComplexHypreParMatrix : public ComplexOperator
{
public:
   ComplexHypreParMatrix(HypreParMatrix * A_Real, HypreParMatrix * A_Imag,
                         bool ownReal, bool ownImag,
                         Convention convention = HERMITIAN);

   HypreParMatrix & real() override;
   HypreParMatrix & imag() override;

   const HypreParMatrix & real() const override;
   const HypreParMatrix & imag() const override;

   /** Combine the blocks making up this complex operator into a single
       HypreParMatrix. The resulting matrix can be passed to solvers which
       require access to the matrix entries themselves, such as sparse direct
       solvers or Hypre preconditioners, rather than simply the action of the
       operator. Note that this combined operator requires roughly twice the
       memory of the block structured operator. */
   HypreParMatrix * GetSystemMatrix() const;

   Type GetType() const override { return Complex_Hypre_ParCSR; }

private:
   void getColStartStop(const HypreParMatrix * A_r,
                        const HypreParMatrix * A_i,
                        int & num_recv_procs,
                        HYPRE_BigInt *& offd_col_start_stop) const;

   MPI_Comm comm_;
   int myid_;
   int nranks_;
};

#endif // MFEM_USE_MPI

}

#endif // MFEM_COMPLEX_OPERATOR
