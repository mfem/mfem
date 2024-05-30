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

#ifndef MFEM_HYBRIDIZATION_EXT
#define MFEM_HYBRIDIZATION_EXT

#include "../config/config.hpp"
#include "../general/array.hpp"
#include "../linalg/vector.hpp"

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
protected:
   class Hybridization &h; ///< The associated Hybridization object.=
   int num_hat_dofs; ///< Number of Lagrange multipliers.
   mutable Vector tmp1, tmp2; ///< Temporary vectors.

   Array<int> el_to_face;
   Vector Ct_mat; ///< Constraint matrix (transposed) stored element-wise.

   /// Classification of the "hat DOFs" in the broken space.
   enum HatDofType
   {
      FREE_BOUNDARY = -1,
      FREE_INTERIOR = 0,
      ESSENTIAL = 1
   };

   /// Construct the constraint matrix.
   void ConstructC();

   /// Compute the action of C^t x.
   void MultCt(const Vector &x, Vector &y) const;

public:
   /// Constructor.
   HybridizationExtension(class Hybridization &hybridization_);
   /// Prepare for assembly; form the constraint matrix.
   void Init(const Array<int> &ess_tdof_list);
   void ComputeSolution(const Vector &b, const Vector &sol_r, Vector &sol) const;
};

}

#endif
