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

#ifndef MFEM_RESTRICTION
#define MFEM_RESTRICTION

#include "../linalg/operator.hpp"
#include "../mesh/mesh.hpp"

namespace mfem
{

class FiniteElementSpace;
enum class ElementDofOrdering;

/** An enum type to specify if only e1 value is requested (SingleValued) or both
    e1 and e2 (DoubleValued). */
enum class L2FaceValues : bool {SingleValued, DoubleValued};

/// Operator that converts FiniteElementSpace L-vectors to E-vectors.
/** Objects of this type are typically created and owned by FiniteElementSpace
    objects, see FiniteElementSpace::GetElementRestriction(). */
class ElementRestriction : public Operator
{
private:
   /** This number defines the maximum number of elements any dof can belong to
       for the FillSparseMatrix method. */
   static const int MaxNbNbr = 16;

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
   Array<int> gatherMap;

public:
   ElementRestriction(const FiniteElementSpace&, ElementDofOrdering);
   void Mult(const Vector &x, Vector &y) const;
   void MultTranspose(const Vector &x, Vector &y) const;

   /// Compute Mult without applying signs based on DOF orientations.
   void MultUnsigned(const Vector &x, Vector &y) const;
   /// Compute MultTranspose without applying signs based on DOF orientations.
   void MultTransposeUnsigned(const Vector &x, Vector &y) const;

   /// @brief Fills the E-vector y with `boolean` values 0.0 and 1.0 such that each
   /// each entry of the L-vector is uniquely represented in `y`.
   /** This means, the sum of the E-vector `y` is equal to the sum of the
       corresponding L-vector filled with ones. The boolean mask is required to
       emulate SetSubVector and its transpose on GPUs. This method is running on
       the host, since the `processed` array requires a large shared memory. */
   void BooleanMask(Vector& y) const;

   /// Fill a Sparse Matrix with Element Matrices.
   void FillSparseMatrix(const Vector &mat_ea, SparseMatrix &mat) const;

   /** Fill the I array of SparseMatrix corresponding to the sparsity pattern
       given by this ElementRestriction. */
   int FillI(SparseMatrix &mat) const;
   /** Fill the J and Data arrays of SparseMatrix corresponding to the sparsity
       pattern given by this ElementRestriction, and the values of ea_data. */
   void FillJAndData(const Vector &ea_data, SparseMatrix &mat) const;
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
   /** Fill the I array of SparseMatrix corresponding to the sparsity pattern
       given by this ElementRestriction. */
   void FillI(SparseMatrix &mat) const;
   /** Fill the J and Data arrays of SparseMatrix corresponding to the sparsity
       pattern given by this L2FaceRestriction, and the values of ea_data. */
   void FillJAndData(const Vector &ea_data, SparseMatrix &mat) const;
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
   const int ne;
   const int vdim;
   const bool byvdim;
   const int ndofs;
   const int dof;
   const int elemDofs;
   const L2FaceValues m;
   const int nfdofs;
   Array<int> scatter_indices1;
   Array<int> scatter_indices2;
   Array<int> offsets;
   Array<int> gather_indices;

   L2FaceRestriction(const FiniteElementSpace&,
                     const FaceType,
                     const L2FaceValues m = L2FaceValues::DoubleValued);

public:
   L2FaceRestriction(const FiniteElementSpace&, const ElementDofOrdering,
                     const FaceType,
                     const L2FaceValues m = L2FaceValues::DoubleValued);
   virtual void Mult(const Vector &x, Vector &y) const;
   void MultTranspose(const Vector &x, Vector &y) const;
   /** Fill the I array of SparseMatrix corresponding to the sparsity pattern
       given by this L2FaceRestriction. */
   virtual void FillI(SparseMatrix &mat, SparseMatrix &face_mat) const;
   /** Fill the J and Data arrays of SparseMatrix corresponding to the sparsity
       pattern given by this L2FaceRestriction, and the values of ea_data. */
   virtual void FillJAndData(const Vector &ea_data,
                             SparseMatrix &mat,
                             SparseMatrix &face_mat) const;
   /// This methods adds the DG face matrices to the element matrices.
   void AddFaceMatricesToElementMatrices(Vector &fea_data,
                                         Vector &ea_data) const;
};

// Return the face degrees of freedom returned in Lexicographic order.
void GetFaceDofs(const int dim, const int face_id,
                 const int dof1d, Array<int> &faceMap);

// Convert from Native ordering to lexicographic ordering
int ToLexOrdering(const int dim, const int face_id, const int size1d,
                  const int index);

// Permute dofs or quads on a face for e2 to match with the ordering of e1
int PermuteFaceL2(const int dim, const int face_id1,
                  const int face_id2, const int orientation,
                  const int size1d, const int index);

}

#endif //MFEM_RESTRICTION
