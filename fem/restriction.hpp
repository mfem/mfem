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

#ifndef MFEM_RESTRICTION
#define MFEM_RESTRICTION

#include "../linalg/operator.hpp"
#include "../mesh/mesh.hpp"

namespace mfem
{

class FiniteElementSpace;
class ParFiniteElementSpace;
enum class ElementDofOrdering;

/** An enum type to specify if only e1 value is requested (Single) or both
    e1 and e2 (Double). */
enum class L2FaceValues : bool {Single, Double};

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
public:
   L2ElementRestriction(const FiniteElementSpace&);
   void Mult(const Vector &x, Vector &y) const;
   void MultTranspose(const Vector &x, Vector &y) const;
};

/// Operator that extracts Face degrees of freedom.
/** Objects of this type are typically created and owned by FiniteElementSpace
    objects, see FiniteElementSpace::GetElementRestriction(). */
class H1FaceRestriction : public Operator
{
protected:
   const FiniteElementSpace &fes;
   int nf;//const int nf;
   const int vdim;
   const bool byvdim;
   const int ndofs;
   int dof;//const int dof;
   int nfdofs;//const int nfdofs;
   Array<int> indices;
   Array<bool> signs;

public:
   H1FaceRestriction(const FiniteElementSpace&, const ElementDofOrdering,
                     const FaceType);
   void Mult(const Vector &x, Vector &y) const;
   void MultTranspose(const Vector &x, Vector &y) const;
   static void GetFaceDofs(const int dim, const int face_id, const int dof1d,
                           Array<int> &faceMap);
};

/// Operator that extracts Face degrees of freedom.
/** Objects of this type are typically created and owned by FiniteElementSpace
    objects, see FiniteElementSpace::GetElementRestriction(). */
class L2FaceRestriction : public Operator
{
protected:
   const FiniteElementSpace &fes;
   int nf;//const int nf;
   const int vdim;
   const bool byvdim;
   const int ndofs;
   int dof;//const int dof;
   const L2FaceValues m;
   int nfdofs;//const int nfdofs;
   Array<int> indices1;
   Array<int> indices2;
   Array<bool> signs;

public:
   L2FaceRestriction(const FiniteElementSpace&, const ElementDofOrdering,
                     const FaceType, const L2FaceValues m = L2FaceValues::Double);
   void Mult(const Vector &x, Vector &y) const;
   void MultTranspose(const Vector &x, Vector &y) const;
   static int PermuteFaceL2(const int dim, const int face_id1, const int face_id2,
                            const int orientation,
                            const int size1d, const int index);
};

/// Operator that extracts Face degrees of freedom.
/** Objects of this type are typically created and owned by FiniteElementSpace
    objects, see FiniteElementSpace::GetElementRestriction(). */
class ParL2FaceRestriction : public Operator
{
protected:
   const ParFiniteElementSpace &fes;
   int nf;//const int nf;
   const int vdim;
   const bool byvdim;
   const int ndofs;
   int dof;//const int dof;
   const L2FaceValues m;
   int nfdofs;//const int nfdofs;
   Array<int> indices1;
   Array<int> indices2;

public:
   ParL2FaceRestriction(const ParFiniteElementSpace&, ElementDofOrdering,
                        FaceType type, L2FaceValues m = L2FaceValues::Double);
   void Mult(const Vector &x, Vector &y) const;
   void MultTranspose(const Vector &x, Vector &y) const;
};

}

#endif //MFEM_RESTRICTION