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

#ifndef MFEM_PRESTRICTION
#define MFEM_PRESTRICTION

#include "../config/config.hpp"

#ifdef MFEM_USE_MPI

#include "restriction.hpp"

namespace mfem
{

class ParFiniteElementSpace;

/// Operator that extracts Face degrees of freedom in parallel.
/** Objects of this type are typically created and owned by FiniteElementSpace
    objects, see FiniteElementSpace::GetFaceRestriction(). */
class ParL2FaceRestriction : public L2FaceRestriction
{
public:
   ParL2FaceRestriction(const ParFiniteElementSpace&, ElementDofOrdering,
                        FaceType type,
                        L2FaceValues m = L2FaceValues::DoubleValued);
   void Mult(const Vector &x, Vector &y) const override;
   /** Fill the I array of SparseMatrix corresponding to the sparsity pattern
       given by this L2FaceRestriction. */
   void FillI(SparseMatrix &mat,
              const bool keep_nbr_block = false) const override;
   /** Fill the I array of SparseMatrix corresponding to the sparsity pattern
       given by this L2FaceRestriction. @a mat contains the interior dofs
       contribution, the @a face_mat contains the shared dofs contribution.*/
   void FillI(SparseMatrix &mat,
              SparseMatrix &face_mat) const;
   /** Fill the J and Data arrays of SparseMatrix corresponding to the sparsity
       pattern given by this L2FaceRestriction, and the values of ea_data.
       @a mat contains the interior dofs contribution, the @a face_mat contains
       the shared dofs contribution.*/
   void FillJAndData(const Vector &ea_data,
                     SparseMatrix &mat,
                     SparseMatrix &face_mat) const;

   void FillJAndData(const Vector &ea_data,
                     SparseMatrix &mat,
                     const bool keep_nbr_block = false) const override;
};

/// Operator that extracts Face degrees of freedom for NCMesh in parallel.
/** Objects of this type are typically created and owned by FiniteElementSpace
    objects, see FiniteElementSpace::GetFaceRestriction(). */
class ParNCL2FaceRestriction : public NCL2FaceRestriction
{
public:
   ParNCL2FaceRestriction(const ParFiniteElementSpace&, ElementDofOrdering,
                          FaceType type,
                          L2FaceValues m = L2FaceValues::DoubleValued);
   void Mult(const Vector &x, Vector &y) const override;
   /** Fill the I array of SparseMatrix corresponding to the sparsity pattern
       given by this L2FaceRestriction. */
   void FillI(SparseMatrix &mat,
              const bool keep_nbr_block = false) const override;
   /** Fill the I array of SparseMatrix corresponding to the sparsity pattern
       given by this L2FaceRestriction. @a mat contains the interior dofs
       contribution, the @a face_mat contains the shared dofs contribution.*/
   void FillI(SparseMatrix &mat,
              SparseMatrix &face_mat) const;
   /** Fill the J and Data arrays of SparseMatrix corresponding to the sparsity
       pattern given by this L2FaceRestriction, and the values of ea_data.
       @a mat contains the interior dofs contribution, the @a face_mat contains
       the shared dofs contribution.*/
   void FillJAndData(const Vector &ea_data,
                     SparseMatrix &mat,
                     SparseMatrix &face_mat) const;

   void FillJAndData(const Vector &ea_data,
                     SparseMatrix &mat,
                     const bool keep_nbr_block = false) const override;
};

}

#endif // MFEM_USE_MPI

#endif // MFEM_PRESTRICTION
