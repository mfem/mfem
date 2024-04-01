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

#ifndef CHANGE_BASIS_HPP
#define CHANGE_BASIS_HPP

#include "mfem.hpp"

namespace mfem
{

/// @brief Change of basis operator between L2 spaces.
///
/// This represents the change-of-basis operator from the given L2 space to a
/// space using the IntegratedGLL basis.
class ChangeOfBasis_L2 : public Operator
{
private:
   const int ne; ///< Number of elements in the mesh.
   mutable DofToQuad dof2quad; ///< 1D basis transformation.
   Array<real_t> B_1d; ///< 1D basis transformation matrix.
   Array<real_t> Bt_1d; ///< 1D basis transformation matrix transpose.
   bool no_op; ///< If the basis types are the same, the operation is a no-op.
public:
   ChangeOfBasis_L2(FiniteElementSpace &fes);
   void Mult(const Vector &x, Vector &y) const override;
   void MultTranspose(const Vector &x, Vector &y) const override;
};

/// Change of basis operator between RT spaces.
///
/// This represents the change-of-basis operator from the given RT space to a
/// space using Gauss-Lobatto as the "open" basis and IntegratedGLL as the
/// "closed" basis.
class ChangeOfBasis_RT : public Operator
{
public:
   // Should be private, nvcc limitation...
   enum Mode
   {
      NORMAL,
      TRANSPOSE,
      INVERSE
   };
private:
   FiniteElementSpace &fes; ///< The finite element space.
   const int dim; ///< Dimension of the mesh.
   const int ne; ///< Number of elements.
   const int p; ///< Polynomial degree.
   const ElementRestriction *elem_restr; ///< Element restriction operator.
   Array<real_t> Bc_1d; ///< 1D closed basis transformation matrix.
   Array<real_t> Bci_1d; ///< 1D closed basis transformation matrix inverse.
   Array<real_t> Bct_1d; ///< 1D closed basis transformation matrix transpose.
   Array<real_t> Bo_1d; ///< 1D open basis transformation matrix.
   Array<real_t> Boi_1d; ///< 1D open basis transformation matrix inverse.
   Array<real_t> Bot_1d; ///< 1D open basis transformation matrix transpose.

   mutable Vector x_l, y_l; ///< L-vector layout
   mutable Vector x_e, y_e; ///< E-vector layout

   bool no_op; ///< If the spaces are the same, the operation is a no-op.

   void Mult(const Vector &x, Vector &y, Mode mode) const;
   const real_t *GetOpenMap(Mode mode) const;
   const real_t *GetClosedMap(Mode mode) const;
public:
   ChangeOfBasis_RT(FiniteElementSpace &fes);
   void Mult(const Vector &x, Vector &y) const override;
   void MultTranspose(const Vector &x, Vector &y) const override;
   void MultInverse(const Vector &x, Vector &y) const;
   // The following should be considered private, public because of compiler
   // limitations
   void MultRT_2D(const Vector &x, Vector &y, Mode mode) const;
   void MultRT_3D(const Vector &x, Vector &y, Mode mode) const;
};

} // namespace mfem

#endif
