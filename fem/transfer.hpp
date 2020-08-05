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

#ifndef MFEM_TRANSFER_HPP
#define MFEM_TRANSFER_HPP

#include "../linalg/linalg.hpp"
#include "fespace.hpp"

#ifdef MFEM_USE_MPI
#include "pfespace.hpp"
#endif

namespace mfem
{

/// Matrix-free transfer operator between finite element spaces
class TransferOperator : public Operator
{
private:
   Operator* opr;

public:
   /// Constructs a transfer operator from \p lFESpace to \p hFESpace.
   /** No matrices are assembled, only the action to a vector is being computed.
       If both spaces' FE collection pointers are pointing to the same collection
       we assume that the grid was refined while keeping the order constant. If
       the FE collections are different, it is assumed that both spaces have are
       using the same mesh. If the first element of the high-order space is a
       `TensorBasisElement`, the optimized tensor-product transfers are used. If
       not, the general transfers used. */
   TransferOperator(const FiniteElementSpace& lFESpace,
                    const FiniteElementSpace& hFESpace);

   /// Destructor
   virtual ~TransferOperator();

   /// @brief Interpolation or prolongation of a vector \p x corresponding to the
   /// coarse space to the vector \p y corresponding to the fine space.
   virtual void Mult(const Vector& x, Vector& y) const override;

   /// Restriction by applying the transpose of the Mult method.
   /** The vector \p x corresponding to the fine space is restricted to the vector
       \p y corresponding to the coarse space. */
   virtual void MultTranspose(const Vector& x, Vector& y) const override;
};

/// Matrix-free transfer operator between finite element spaces on the same mesh
class PRefinementTransferOperator : public Operator
{
private:
   const FiniteElementSpace& lFESpace;
   const FiniteElementSpace& hFESpace;

public:
   /// @brief Constructs a transfer operator from \p lFESpace to \p hFESpace
   /// which have different FE collections.
   /** No matrices are assembled, only the action to a vector is being computed.
       The underlying finite elements need to implement the GetTransferMatrix
       methods. */
   PRefinementTransferOperator(const FiniteElementSpace& lFESpace_,
                               const FiniteElementSpace& hFESpace_);

   /// Destructor
   virtual ~PRefinementTransferOperator();

   /// @brief Interpolation or prolongation of a vector \p x corresponding to the
   /// coarse space to the vector \p y corresponding to the fine space.
   virtual void Mult(const Vector& x, Vector& y) const override;

   /// Restriction by applying the transpose of the Mult method.
   /** The vector \p x corresponding to the fine space is restricted to the vector
   \p y corresponding to the coarse space. */
   virtual void MultTranspose(const Vector& x, Vector& y) const override;
};

/// @brief Matrix-free transfer operator between finite element spaces on the same
/// mesh exploiting the tensor product structure of the finite elements
class TensorProductPRefinementTransferOperator : public Operator
{
private:
   const FiniteElementSpace& lFESpace;
   const FiniteElementSpace& hFESpace;
   int dim;
   int NE;
   int D1D;
   int Q1D;
   Array<double> B;
   Array<double> Bt;
   const Operator* elem_restrict_lex_l;
   const Operator* elem_restrict_lex_h;
   Vector mask;
   mutable Vector localL;
   mutable Vector localH;

public:
   /// @brief Constructs a transfer operator from \p lFESpace to \p hFESpace which
   /// have different FE collections.
   /** No matrices are assembled, only the action to a vector is being computed.
   The underlying finite elements need to be of the type `TensorBasisElement`. It
   is also assumed that all the elements in the spaces are of the same type. */
   TensorProductPRefinementTransferOperator(
      const FiniteElementSpace& lFESpace_,
      const FiniteElementSpace& hFESpace_);

   /// Destructor
   virtual ~TensorProductPRefinementTransferOperator();

   /// @brief Interpolation or prolongation of a vector \p x corresponding to the
   /// coarse space to the vector \p y corresponding to the fine space.
   virtual void Mult(const Vector& x, Vector& y) const override;

   /// Restriction by applying the transpose of the Mult method.
   /** The vector \p x corresponding to the fine space is restricted to the vector
   \p y corresponding to the coarse space. */
   virtual void MultTranspose(const Vector& x, Vector& y) const override;
};

#ifdef MFEM_USE_MPI
/// @brief Matrix-free transfer operator between finite element spaces working on
/// true degrees of freedom
class TrueTransferOperator : public Operator
{
private:
   const ParFiniteElementSpace& lFESpace;
   const ParFiniteElementSpace& hFESpace;
   TransferOperator* localTransferOperator;
   mutable Vector tmpL;
   mutable Vector tmpH;

public:
   /// @brief Constructs a transfer operator working on true degrees of freedom from
   /// from \p lFESpace to \p hFESpace
   TrueTransferOperator(const ParFiniteElementSpace& lFESpace_,
                        const ParFiniteElementSpace& hFESpace_);

   /// Destructor
   ~TrueTransferOperator();

   /// @brief Interpolation or prolongation of a true dof vector \p x to a true dof
   /// vector \p y.
   /** The true dof vector \p x corresponding to the coarse space is restricted to
       the true dof vector \p y corresponding to the fine space. */
   virtual void Mult(const Vector& x, Vector& y) const override;

   /// Restriction by applying the transpose of the Mult method.
   /** The true dof vector \p x corresponding to the fine space is restricted to
       the true dof vector \p y corresponding to the coarse space. */
   virtual void MultTranspose(const Vector& x, Vector& y) const override;
};
#endif

} // namespace mfem
#endif
