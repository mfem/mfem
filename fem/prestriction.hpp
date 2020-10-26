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
   void Mult(const Vector &x, Vector &y) const;
   /** Fill the I array of SparseMatrix corresponding to the sparsity pattern
       given by this ParL2FaceRestriction. */
   void FillI(SparseMatrix &mat, SparseMatrix &face_mat) const;
   /** Fill the J and Data arrays of SparseMatrix corresponding to the sparsity
       pattern given by this ParL2FaceRestriction, and the values of ea_data. */
   void FillJAndData(const Vector &ea_data,
                     SparseMatrix &mat,
                     SparseMatrix &face_mat) const;
};

}

#endif // MFEM_USE_MPI

#endif //MFEM_PRESTRICTION
