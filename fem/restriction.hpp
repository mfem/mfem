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

   /// Compute MultTranspose by setting (rather than adding) element
   /// contributions; this is a left inverse of the Mult() operation
   void MultLeftInverse(const Vector &x, Vector &y) const;

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
   const int nf; // Number of faces of the requested type
   const int ne; // Number of elements
   const int vdim; // vdim
   const bool byvdim;
   const int ndofs; // Total number of dofs
   const int dof; // Number of dofs on each face
   const int elemDofs; // Number of dofs in each element
   const FaceType type;
   const L2FaceValues m;
   const int nfdofs; // Total number of dofs on the faces
   Array<int> scatter_indices1; // Scattering indices for element 1 on each face
   Array<int> scatter_indices2; // Scattering indices for element 2 on each face
   Array<int> offsets; // offsets for the gathering indices of each dof
   Array<int> gather_indices; // gathering indices for each dof

   L2FaceRestriction(const FiniteElementSpace&,
                     const FaceType,
                     const L2FaceValues m = L2FaceValues::DoubleValued);

public:
   /** @brief Constructs an L2FaceRestriction.

       @param[in] fes      The FiniteElementSpace on which this operates
       @param[in] ordering Request a specific ordering
       @param[in] type     Request internal or boundary faces dofs
       @param[in] m        Request the face dofs for elem1, or both elem1 and
                           elem2 */
   L2FaceRestriction(const FiniteElementSpace& fes,
                     const ElementDofOrdering ordering,
                     const FaceType type,
                     const L2FaceValues m = L2FaceValues::DoubleValued);

   /** @brief Scatter the degrees of freedom, i.e. goes from L-Vector to
       face E-Vector.

       The format of y is:
       if m==L2FacesValues::DoubleValued (face_dofs x vdim x 2 x nf)
       if m==L2FacesValues::SingleValued (face_dofs x vdim x nf) */
   void Mult(const Vector &x, Vector &y) const override;

   /** @brief Gather the degrees of freedom, i.e. goes from face E-Vector to
       L-Vector. */
   void MultTranspose(const Vector &x, Vector &y) const override;

   /** Fill the I array of SparseMatrix corresponding to the sparsity pattern
       given by this L2FaceRestriction. */
   virtual void FillI(SparseMatrix &mat, const bool keep_nbr_block = false) const;

   /** Fill the J and Data arrays of SparseMatrix corresponding to the sparsity
       pattern given by this L2FaceRestriction, and the values of ea_data. */
   virtual void FillJAndData(const Vector &ea_data,
                             SparseMatrix &mat,
                             const bool keep_nbr_block = false) const;

   /// This methods adds the DG face matrices to the element matrices.
   virtual void AddFaceMatricesToElementMatrices(Vector &fea_data,
                                                 Vector &ea_data) const;
};

/** @brief Operator that extracts face degrees of freedom for non-conforming 
    meshes.

    In order to support face restrictions on non-conforming meshes, this
    operator interpolates master (coarse) face degrees of freedom onto the
    slave (fine) face. This allows face integrators to treat non-conforming
    faces just as regular conforming faces. */
class NCL2FaceRestriction : public L2FaceRestriction
{
protected:
   Array<int> interp_config; // interpolator index for each face
   int nc_size; // number of non-conforming interpolators
   Vector interpolators; // face_dofs x face_dofs x nc_size

   NCL2FaceRestriction(const FiniteElementSpace&,
                       const FaceType,
                       const L2FaceValues m = L2FaceValues::DoubleValued);

public:
   /** @brief Constructs an NCL2FaceRestriction, this is a specialization of a
       NCL2FaceRestriction for .

       @param[in] fes      The FiniteElementSpace on which this operates
       @param[in] ordering Request a specific ordering
       @param[in] type     Request internal or boundary faces dofs
       @param[in] m        Request the face dofs for elem1, or both elem1 and
                           elem2 */
   NCL2FaceRestriction(const FiniteElementSpace& fes,
                       const ElementDofOrdering ordering,
                       const FaceType type,
                       const L2FaceValues m = L2FaceValues::DoubleValued);

   /** @brief Scatter the degrees of freedom, i.e. goes from L-Vector to
       face E-Vector.

       The format of y is:
       if m==L2FacesValues::DoubleValued (face_dofs x vdim x 2 x nf)
       if m==L2FacesValues::SingleValued (face_dofs x vdim x nf) */
   void Mult(const Vector &x, Vector &y) const override;

   /** @brief Gather the degrees of freedom, i.e. goes from face E-Vector to
       L-Vector. */
   void MultTranspose(const Vector &x, Vector &y) const override;

   /** Fill the I array of SparseMatrix corresponding to the sparsity pattern
       given by this L2FaceRestriction. */
   void FillI(SparseMatrix &mat,
              const bool keep_nbr_block = false) const override;

   /** Fill the J and Data arrays of SparseMatrix corresponding to the sparsity
       pattern given by this L2FaceRestriction, and the values of ea_data. */
   void FillJAndData(const Vector &ea_data,
                     SparseMatrix &mat,
                     const bool keep_nbr_block = false) const override;

   /// This methods adds the DG face matrices to the element matrices.
   void AddFaceMatricesToElementMatrices(Vector &fea_data,
                                         Vector &ea_data) const override;

protected:
   static const int conforming = -1; // helper value

   const DenseMatrix* ComputeCoarseToFineInterpolation(const DenseMatrix* ptMat,
                                                       const int face_id1,
                                                       const int face_id2,
                                                       const int orientation);
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
