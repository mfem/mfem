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

#ifndef MFEM_HYBRIDIZATION_EXT
#define MFEM_HYBRIDIZATION_EXT

#include "../config/config.hpp"
#include "../general/array.hpp"
#include "../linalg/operator.hpp"
#include "../linalg/vector.hpp"

#include <memory>

namespace mfem
{

/// @brief Extension class supporting Hybridization on device (GPU).
///
/// Similar to BilinearFormExtension and LinearFormExtension, this extension
/// class provides device execution capabilities for the Hybridization class.
///
/// As with the other extension classes, a limitation of this class is that it
/// requires meshes consisting only of tensor-product elements, and finite
/// element spaces without variable polynomial degrees.
class HybridizationExtension
{
   friend class Hybridization;
public:
   enum DofType : char
   {
      ESSENTIAL,
      BOUNDARY,
      INTERIOR
   };
protected:
   class Hybridization &h; ///< The associated Hybridization object.=
   int num_hat_dofs; ///< Number of Lagrange multipliers.
   mutable Vector tmp1, tmp2; ///< Temporary vectors.

   Array<int> hat_dof_gather_map;
   Array<DofType> hat_dof_marker;

   Array<int> el_to_face; ///< Element to face connectivity.
   Array<int> el_face_offsets; ///< Per-element offsets into @a el_to_face.
   Array<int> face_to_el; ///< Face-to-element connectivity.
   Array<int> face_face_offsets; ///< Face-to-face offsets.

   int n_el_face; ///< Total number of element-to-face connections.
   int n_face_face; ///< Total number of face-to-face connections.

   Vector Ct_mat; ///< Constraint matrix (transposed) stored element-wise.

   /// @name For parallel non-conforming meshes
   ///@{
   std::unique_ptr<Operator> P_pc; ///< Partially conforming prolongation.
   std::unique_ptr<Operator> P_nbr; ///< Face-neighbor prolongation.
   ///@}

   Array<int> idofs, bdofs;

   Vector Ahat, Ahat_ii, Ahat_ib, Ahat_bi, Ahat_bb;
   Array<int> Ahat_ii_piv, Ahat_bb_piv;

   /// Return the (partially) conforming prolongation on the constraint space.
   const Operator &GetProlongation() const;

public:
   /// Construct the constraint matrix.
   void ConstructC();

   template <int MID, int MBD>
   void FactorElementMatrices(Vector &AhatInvCt_mat);

   /// Form the Schur complement matrix $H$.
   void ConstructH();

   /// Compute the action of C^t x.
   void MultCt(const Vector &x, Vector &y) const;

   /// Compute the action of C x.
   void MultC(const Vector &x, Vector &y) const;

   /// @brief Assemble the element matrix A into the hybridized system matrix.
   ///
   /// @warning Using the interface will be very slow. AssembleElementMatrices()
   /// should be used instead.
   void AssembleMatrix(int el, const class DenseMatrix &elmat);

   /// @brief Assemble the boundary element matrix A into the hybridized system
   /// matrix.
   ///
   /// @warning Using the interface will be very slow. AssembleElementMatrices()
   /// should be used instead.
   void AssembleBdrMatrix(int bdr_el, const class DenseMatrix &elmat);

   /// Invert and store the element matrices Ahat.
   void AssembleElementMatrices(const class DenseTensor &el_mats);

   /// Apply the action of R mapping from "hat DOFs" to T-vector
   void MultR(const Vector &b, Vector &b_hat) const;

   /// Apply the action of R^t mapping into the "hat DOF" space.
   void MultRt(const Vector &b, Vector &b_hat) const;

   /// Apply the elementwise A_hat^{-1}.
   void MultAhatInv(Vector &x) const;

   /// Constructor.
   HybridizationExtension(class Hybridization &hybridization_);

   /// Prepare for assembly; form the constraint matrix.
   void Init(const Array<int> &ess_tdof_list);

   /// @brief Given a right-hand side on the original space, compute the
   /// corresponding right-hand side for the Lagrange multipliers.
   void ReduceRHS(const Vector &b, Vector &b_r) const;

   /// @brief Given Lagrange multipliers @a sol_r and the original right-hand
   /// side @a b, recover the solution @a sol on the original finite element
   /// space.
   void ComputeSolution(const Vector &b, const Vector &sol_r, Vector &sol) const;

   /// Destroys the stored element matrices.
   void Reset() { Ahat = 0.0; }
};

}

#endif
