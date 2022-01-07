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

/** An enum type to specify if only e1 value is requested (SingleValued) or both
    e1 and e2 (DoubleValued). */
enum class L2FaceValues : bool {SingleValued, DoubleValued};

/** @brief Base class for operators that extracts Face degrees of freedom.

    In order to compute quantities on the faces of a mesh, it is often useful to
    extract the degrees of freedom on the faces of the elements. This class
    provides an interface for such operations.

    If the FiniteElementSpace is ordered by Ordering::byVDIM, then the expected
    format for the L-vector is (vdim x ndofs), otherwise if Ordering::byNODES
    the expected format is (ndofs x vdim), where ndofs is the total number of
    degrees of freedom.
    Since FiniteElementSpace can either be continuous or discontinuous, the
    degrees of freedom on a face can either be single valued or double valued,
    this is what we refer to as the multiplicity and is represented by the
    L2FaceValues enum type.
    The format of the output face E-vector of degrees of freedom is
    (face_dofs x vdim x multiplicity x nfaces), where face_dofs is the number of
    degrees of freedom on each face, and nfaces the number of faces of the
    requested FaceType (see FiniteElementSpace::GetNFbyType).

    @note Objects of this type are typically created and owned by
    FiniteElementSpace objects, see FiniteElementSpace::GetFaceRestriction(). */
class FaceRestriction : public Operator
{
public:
   FaceRestriction(): Operator() { }

   FaceRestriction(int h, int w): Operator(h, w) { }

   virtual ~FaceRestriction() { }

   /** @brief Extract the face degrees of freedom from @a x into @a y.

       @param[in]  x The L-vector of degrees of freedom.
       @param[out] y The degrees of freedom on the face, corresponding to a face
                     E-vector.
   */
   void Mult(const Vector &x, Vector &y) const override = 0;

   /** @brief Add the face degrees of freedom @a x to the element degrees of
       freedom @a y.

       @param[in]     x The face degrees of freedom on the face.
       @param[in,out] y The L-vector of degrees of freedom to which we add the
                        face degrees of freedom.
   */
   virtual void AddMultTranspose(const Vector &x, Vector &y) const = 0;

   /** @brief Set the face degrees of freedom in the element degrees of freedom
       @a y to the values given in @a x.

       @param[in]     x The face degrees of freedom on the face.
       @param[in,out] y The L-vector of degrees of freedom to which we add the
                        face degrees of freedom.
   */
   void MultTranspose(const Vector &x, Vector &y) const override
   {
      y = 0.0;
      AddMultTranspose(x, y);
   }
};

/// Operator that extracts Face degrees of freedom for H1 FiniteElementSpaces.
/** Objects of this type are typically created and owned by FiniteElementSpace
    objects, see FiniteElementSpace::GetFaceRestriction(). */
class H1FaceRestriction : public FaceRestriction
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
   /** @brief Constructor for a H1FaceRestriction.

       @param[in] fes The FiniteElementSpace on which this H1FaceRestriction
                      operates.
       @param[in] ordering The requested output ordering of the
                           H1FaceRestriction, either Native or Lexicographic.
       @param[in] type The requested type of faces on which this operator
                       extracts the degrees of freedom, either Interior or
                       Boundary.
   */
   H1FaceRestriction(const FiniteElementSpace& fes,
                     const ElementDofOrdering ordering,
                     const FaceType type);

   /** @brief Extract the face degrees of freedom from @a x into @a y.

       @param[in]  x The L-vector of degrees of freedom.
       @param[out] y The degrees of freedom on the face, corresponding to a face
                     E-vector.
   */
   void Mult(const Vector &x, Vector &y) const override;

   /** @brief Add the face degrees of freedom @a x to the element degrees of
       freedom @a y.

       @param[in]     x The face degrees of freedom on the face.
       @param[in,out] y The L-vector of degrees of freedom to which we add the
                        face degrees of freedom.
   */
   void AddMultTranspose(const Vector &x, Vector &y) const override;
};

/// Operator that extracts Face degrees of freedom on L2 FiniteElementSpaces.
/** Objects of this type are typically created and owned by FiniteElementSpace
    objects, see FiniteElementSpace::GetFaceRestriction(). */
class L2FaceRestriction : public FaceRestriction
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
   L2FaceRestriction(const FiniteElementSpace&,
                     const ElementDofOrdering,
                     const FaceType,
                     const L2FaceValues m = L2FaceValues::DoubleValued);

   /** @brief Extract the face degrees of freedom from @a x into @a y.

       @param[in]  x The L-vector of degrees of freedom.
       @param[out] y The degrees of freedom on the face, corresponding to a face
                     E-vector.
   */
   void Mult(const Vector &x, Vector &y) const override;

   /** @brief Add the face degrees of freedom @a x to the element degrees of
       freedom @a y.

       @param[in]     x The face degrees of freedom on the face.
       @param[in,out] y The L-vector of degrees of freedom to which we add the
                        face degrees of freedom.
   */
   void AddMultTranspose(const Vector &x, Vector &y) const override;

   /** Fill the I array of SparseMatrix corresponding to the sparsity pattern
       given by this L2FaceRestriction. */
   virtual void FillI(SparseMatrix &mat, const bool keep_nbr_block = false) const;

   /** Fill the J and Data arrays of SparseMatrix corresponding to the sparsity
       pattern given by this L2FaceRestriction, and the values of ea_data. */
   virtual void FillJAndData(const Vector &ea_data,
                             SparseMatrix &mat,
                             const bool keep_nbr_block = false) const;

   /// This methods adds the DG face matrices to the element matrices.
   void AddFaceMatricesToElementMatrices(Vector &fea_data,
                                         Vector &ea_data) const;
};


/// Operator that computes normal derivative at the face.
/** Objects of this type are typically created and owned by FiniteElementSpace
    objects, see FiniteElementSpace::GetFaceRestriction(). */
class L2FaceNormalDRestriction : public FaceRestriction
{
protected:
   const FiniteElementSpace &fes;
   const int dim;
   const int nf; // number of faces
   const int ne; // number of elements
   const int vdim; // number of vector components
   const bool byvdim; // fes.GetOrdering() == Ordering::byVDIM
   const int ndofs; // total number of dofs
   const int ndofs1d; // dofs in 1 dimension (for tensor bases)
   const int ndofs_face; // fes.GetTraceElement(0, fes.GetMesh()->GetFaceBaseGeometry(0))->GetDof()
   // dof equal to dofs per face?
   const int elemDofs; // dofs per element
   const L2FaceValues m; // are face values double valued?
   const int numfacedofs; // total number of face dofs
   const int num_values_per_point; // 2 for values and their derivatives, more for second derivatices, etc.
                                   // higher derivatives not yet implemented
   const int num_faces_per_element;

   Array<int> map_elements_to_faces;
   Array<int> map_elements_to_sides;
   Array<int> map_side_permutations;
   int num_needed_elements;
   Array<int> needed_elements;

   Vector Ge;
   Vector jac_face_factors;

   L2FaceNormalDRestriction(const FiniteElementSpace&,
                     const FaceType,
                     const L2FaceValues m = L2FaceValues::DoubleValued);

public:
   L2FaceNormalDRestriction(const FiniteElementSpace&, 
                     const ElementDofOrdering,
                     const FaceType,
                     //const IntegrationRule* Intrules,
                     const L2FaceValues m = L2FaceValues::DoubleValued);
   virtual void Mult(const Vector &x, Vector &y) const;
   void AddMultTranspose(const Vector &x, Vector &y) const;
   /** Fill the I array of SparseMatrix corresponding to the sparsity pattern
       given by this L2FaceRestriction. */

   //virtual void FillI(SparseMatrix &mat, SparseMatrix &face_mat) const;
   /** Fill the J and Data arrays of SparseMatrix corresponding to the sparsity
       pattern given by this L2FaceRestriction, and the values of ea_data. */
   //virtual void FillJAndData(const Vector &ea_data,
   //                          SparseMatrix &mat,
   //                          SparseMatrix &face_mat) const;
   /// This methods adds the DG face matrices to the element matrices.
   //void AddFaceMatricesToElementMatrices(Vector &fea_data,
   //                                      Vector &ea_data) const;
};

// Return the face degrees of freedom returned in Lexicographic order.
void GetFaceDofs(const int dim, const int face_id,
                 const int ndofs1d, Array<int> &faceMap);
void GetNormalDFaceDofStencil(const int dim, const int face_id,
                 const int ndofs1d, Array<Array<int>> &faceMap);
                 
// Convert from Native ordering to lexicographic ordering
int ToLexOrdering(const int dim, const int face_id, const int size1d,
                  const int index);

// Permute dofs or quads on a face for e2 to match with the ordering of e1
int PermuteFaceL2(const int dim, const int face_id1,
                  const int face_id2, const int orientation,
                  const int size1d, const int index);

}

#endif //MFEM_RESTRICTION
