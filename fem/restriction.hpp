// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license.  We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_RESTRICTION
#define MFEM_RESTRICTION

#include "../linalg/operator.hpp"
#include "../mesh/mesh.hpp"

namespace mfem
{

class FiniteElementSpace;
enum class ElementDofOrdering;
#ifdef MFEM_USE_MPI
class ParFiniteElementSpace;
#endif

/** An enum type to specify if only e1 value is requested (SingleValued) or both
    e1 and e2 (DoubleValued). */
enum class L2FaceValues : bool {SingleValued, DoubleValued};

/// Operator that converts FiniteElementSpace L-vectors to E-vectors.
/** Objects of this type are typically created and owned by FiniteElementSpace
    objects, see FiniteElementSpace::GetElementRestriction(). */
class ElementRestriction : public Operator
{
protected:
   const FiniteElementSpace &fes;
   const int ne;
   const int vdim;
   const bool byvdim;
   const int ndofs;
   const int dof;
   const int nedofs;
   Array<int> offsets;
   Array<int> indices;

public:
   ElementRestriction(const FiniteElementSpace&, ElementDofOrdering);
   void Mult(const Vector &x, Vector &y) const;
   void MultTranspose(const Vector &x, Vector &y) const;

   /// Compute MultTranspose without applying signs based on DOF orientations.
   void MultTransposeUnsigned(const Vector &x, Vector &y) const;
};

/// Operator that converts L2 FiniteElementSpace L-vectors to E-vectors.
/** Objects of this type are typically created and owned by FiniteElementSpace
    objects, see FiniteElementSpace::GetElementRestriction(). L-vectors
    corresponding to grid functions in L2 finite element spaces differ from
    E-vectors only in the ordering of the degrees of freedom. */
class L2ElementRestriction : public Operator
{
   const int ne;
   const int vdim;
   const bool byvdim;
   const int ndof;
   const int ndofs;
public:
   L2ElementRestriction(const FiniteElementSpace&);
   void Mult(const Vector &x, Vector &y) const;
   void MultTranspose(const Vector &x, Vector &y) const;
};

/// Operator that extracts Face degrees of freedom.
/** Objects of this type are typically created and owned by FiniteElementSpace
    objects, see FiniteElementSpace::GetFaceRestriction(). */
class H1FaceRestriction : public Operator
{
protected:
   const FiniteElementSpace &fes;
   const int nf;
   const int vdim;
   const bool byvdim;
   const int ndofs;
   const int dof;
   const int nfdofs;
   Array<int> scatter_indices;
   Array<int> offsets;
   Array<int> gather_indices;

public:
   H1FaceRestriction(const FiniteElementSpace&, const ElementDofOrdering,
                     const FaceType);
   void Mult(const Vector &x, Vector &y) const;
   void MultTranspose(const Vector &x, Vector &y) const;
};

/// Operator that extracts Face degrees of freedom.
/** Objects of this type are typically created and owned by FiniteElementSpace
    objects, see FiniteElementSpace::GetFaceRestriction(). */
class L2FaceRestriction : public Operator
{
protected:
   const FiniteElementSpace &fes;
   const int nf;
   const int vdim;
   const bool byvdim;
   const int ndofs;
   const int dof;
   const L2FaceValues m;
   const int nfdofs;
   Array<int> scatter_indices1;
   Array<int> scatter_indices2;
   Array<int> offsets;
   Array<int> gather_indices;

public:
   L2FaceRestriction(const FiniteElementSpace&, const ElementDofOrdering,
                     const FaceType,
                     const L2FaceValues m = L2FaceValues::DoubleValued);
   void Mult(const Vector &x, Vector &y) const;
   void MultTranspose(const Vector &x, Vector &y) const;
};

#ifdef MFEM_USE_MPI

/// Operator that extracts Face degrees of freedom.
/** Objects of this type are typically created and owned by FiniteElementSpace
    objects, see FiniteElementSpace::GetFaceRestriction(). */
class ParL2FaceRestriction : public Operator
{
protected:
   const ParFiniteElementSpace &fes;
   const int nf;
   const int vdim;
   const bool byvdim;
   const int ndofs;
   const int dof;
   const L2FaceValues m;
   const int nfdofs;
   Array<int> scatter_indices1;
   Array<int> scatter_indices2;
   Array<int> offsets;
   Array<int> gather_indices;

public:
   ParL2FaceRestriction(const ParFiniteElementSpace&, ElementDofOrdering,
                        FaceType type,
                        L2FaceValues m = L2FaceValues::DoubleValued);
   void Mult(const Vector &x, Vector &y) const;
   void MultTranspose(const Vector &x, Vector &y) const;
};

#endif // MFEM_USE_MPI

// Convert from Native ordering to lexicographic ordering
int ToLexOrdering(const int dim, const int face_id, const int size1d,
                  const int index);

}

#endif //MFEM_RESTRICTION
