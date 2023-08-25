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

#ifndef MFEM_NORMAL_DERIV_RESTRICTION
#define MFEM_NORMAL_DERIV_RESTRICTION

#include "../mesh/mesh.hpp"

namespace mfem
{

class FiniteElementSpace;
enum class ElementDofOrdering;

/// @brief Class to compute (reference coordinate) face normal derivatives of a
/// L2 grid function (used internally by L2FaceRestriction).
class L2NormalDerivativeFaceRestriction
{
protected:
   const FiniteElementSpace &fes; ///< The L2 space.
   const FaceType face_type; ///< Boundary or interior spaces?
   const int dim; ///< Dimension of the mesh.
   const int nf; ///< Number of faces of the given @a face_type.
   const int ne; ///< Number of elements.
   int ne_type; ///< Number of elements with faces of type face type

   Array<int> face_to_elem; ///< Face-wise information array.
   Array<int> elem_to_face; ///< Element-wise information array.

public:
   /// @brief Constructs an L2NormalDerivativeFaceRestriction
   /// @param[in] fes_ The FiniteElementSpace on which this operates
   /// @param[in] ordering ordering of dofs
   /// @param[in] face_type_ type of faces to compute restriction for (interior or boundary)
   L2NormalDerivativeFaceRestriction(const FiniteElementSpace &fes_,
                                     const ElementDofOrdering ordering,
                                     const FaceType face_type_);

   /// @brief Computes the normal derivatives on the @a type faces of the mesh.
   /// @param[in] x The L-vector degrees of freedom.
   /// @param[out] y The face E(like)-vector degrees of freedom of the format
   /// (face_dofs x vdim x 2 x nf) where nf is the number of faces of type @a
   /// face_type. The face_dofs are ordered according to @a ordering specified
   /// in the constructor.
   void Mult(const Vector &x, Vector &y) const;

   /// @brief Computes the transpose of the action of Mult(), accumulating into
   ///        @a y with coefficient @a a.
   /// @param x Face E-vector layout (face_dofs x vdim x 2 x nf).
   /// @param y L-vector layout.
   /// @param a Optional coefficient (y = y + a*R^t*x)
   void AddMultTranspose(const Vector &x, Vector &y,
                         const double a = 1.0) const;

   /// @name Internal compute kernels. Public because of nvcc restriction.
   ///@{

   void Mult2D(const Vector &x, Vector &y) const;

   void Mult3D(const Vector &x, Vector &y) const;

   void AddMultTranspose2D(const Vector &x, Vector &y, const double a) const;

   void AddMultTranspose3D(const Vector &x, Vector &y, const double a) const;

   /// @}
};

}

#endif // MFEM_RESTRICTION
