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

#ifndef MFEM_RESTRICTION
#define MFEM_RESTRICTION

#include "../linalg/operator.hpp"
#include "../mesh/mesh.hpp"
#include "normal_deriv_restriction.hpp"

namespace mfem
{

class FiniteElementSpace;
enum class ElementDofOrdering;

class FaceQuadratureSpace;

/// Abstract base class that defines an interface for element restrictions.
class ElementRestrictionOperator : public Operator
{
public:
   /// @brief Add the E-vector degrees of freedom @a x to the L-vector degrees
   /// of freedom @a y.
   void AddMultTranspose(const Vector &x, Vector &y,
                         const real_t a = 1.0) const override = 0;
};

/// Operator that converts FiniteElementSpace L-vectors to E-vectors.
/** Objects of this type are typically created and owned by FiniteElementSpace
    objects, see FiniteElementSpace::GetElementRestriction(). */
class ElementRestriction : public ElementRestrictionOperator
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
   Array<int> gather_map;

public:
   ElementRestriction(const FiniteElementSpace&, ElementDofOrdering);
   void Mult(const Vector &x, Vector &y) const override;
   void MultTranspose(const Vector &x, Vector &y) const override;
   void AddMultTranspose(const Vector &x, Vector &y,
                         const real_t a = 1.0) const override;

   /// Compute Mult without applying signs based on DOF orientations.
   void AbsMult(const Vector &x, Vector &y) const override;

   /// Compute MultTranspose without applying signs based on DOF orientations.
   void AbsMultTranspose(const Vector &x, Vector &y) const override;

   /// @deprecated Use AbsMult() instead.
   MFEM_DEPRECATED void MultUnsigned(const Vector &x, Vector &y) const
   { AbsMult(x, y); }

   /// @deprecated Use AbsMultTranspose() instead.
   MFEM_DEPRECATED void MultTransposeUnsigned(const Vector &x, Vector &y) const
   { AbsMultTranspose(x, y); }

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
   /// @private Not part of the public interface (device kernel limitation).
   ///
   /// Performs either MultTranspose or AddMultTranspose depending on the
   /// boolean template parameter @a ADD.
   template <bool ADD> void TAddMultTranspose(const Vector &x, Vector &y) const;

   /// @name Low-level access to the underlying element-dof mappings
   ///@{
   const Array<int> &GatherMap() const { return gather_map; }
   const Array<int> &Indices() const { return indices; }
   const Array<int> &Offsets() const { return offsets; }
   ///@}
};

/// Operator that converts L2 FiniteElementSpace L-vectors to E-vectors.
/** Objects of this type are typically created and owned by FiniteElementSpace
    objects, see FiniteElementSpace::GetElementRestriction(). L-vectors
    corresponding to grid functions in L2 finite element spaces differ from
    E-vectors only in the ordering of the degrees of freedom. */
class L2ElementRestriction : public ElementRestrictionOperator
{
   const int ne;
   const int vdim;
   const bool byvdim;
   const int ndof;
   const int ndofs;
public:
   L2ElementRestriction(const FiniteElementSpace&);
   void Mult(const Vector &x, Vector &y) const override;
   void MultTranspose(const Vector &x, Vector &y) const override;
   void AddMultTranspose(const Vector &x, Vector &y,
                         const real_t a = 1.0) const override;
   /** Fill the I array of SparseMatrix corresponding to the sparsity pattern
       given by this ElementRestriction. */
   void FillI(SparseMatrix &mat) const;
   /** Fill the J and Data arrays of SparseMatrix corresponding to the sparsity
       pattern given by this L2FaceRestriction, and the values of ea_data. */
   void FillJAndData(const Vector &ea_data, SparseMatrix &mat) const;
   /// @private Not part of the public interface (device kernel limitation).
   ///
   /// Performs either MultTranspose or AddMultTranspose depending on the
   /// boolean template parameter @a ADD.
   template <bool ADD> void TAddMultTranspose(const Vector &x, Vector &y) const;
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
       @param[in]     a Scalar coefficient for addition.
   */
   void AddMultTranspose(const Vector &x, Vector &y,
                         const real_t a = 1.0) const override = 0;

   /** @brief Add the face degrees of freedom @a x to the element degrees of
       freedom @a y ignoring the signs from DOF orientation. */
   virtual void AddAbsMultTranspose(const Vector &x, Vector &y,
                                    const real_t a = 1.0) const
   {
      AddMultTranspose(x, y, a);
   }

   /// @deprecated Use AddAbsMultTranspose() instead.
   MFEM_DEPRECATED void AddMultTransposeUnsigned(const Vector &x, Vector &y,
                                                 const real_t a = 1.0) const
   {
      AddAbsMultTranspose(x, y, a);
   }

   /** @brief Add the face degrees of freedom @a x to the element degrees of
       freedom @a y. Perform the same computation as AddMultTranspose, but
       @a x is invalid after calling this method.

       @param[in,out]     x The face degrees of freedom on the face.
       @param[in,out] y The L-vector of degrees of freedom to which we add the
                        face degrees of freedom.

      @note This method is an optimization of AddMultTranspose where the @a x
      Vector is used and modified to avoid memory allocation and memcpy.
   */
   virtual void AddMultTransposeInPlace(Vector &x, Vector &y) const
   {
      AddMultTranspose(x, y);
   }

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

   void AbsMultTranspose(const Vector &x, Vector &y) const override
   {
      y = 0.0;
      AddAbsMultTranspose(x, y);
   }

   /** @brief For each face, sets @a y to the partial derivative of @a x with
              respect to the reference coordinate whose direction is
              perpendicular to the face on the reference element.

    @details This is not the normal derivative in physical coordinates, but can
             be mapped to the physical normal derivative using the element
             Jacobian and the tangential derivatives (in reference coordinates)
             which can be computed from the face values (provided by Mult).

             Note that due to the polynomial degree of the element mapping, the
             physical normal derivative may be a higher degree polynomial than
             the restriction of the values to the face. However, the normal
             derivative in reference coordinates has degree-1, and therefore can
             be exactly represented with the degrees of freedom of a face
             E-vector.

    @param[in]     x The L-vector degrees of freedom.
    @param[in,out] y The reference normal derivative degrees of freedom. Is
                     E-vector like.
    */
   virtual void NormalDerivativeMult(const Vector &x, Vector &y) const
   {
      MFEM_ABORT("Not implemented for this restriction operator.");
   }

   /** @brief Add the face reference-normal derivative degrees of freedom in @a
              x to the element degrees of freedom in @a y.

       @details see NormalDerivativeMult.

       @param[in]     x The degrees of freedom of the face reference-normal
                        derivative. Is E-vector like.
       @param[in,out] y The L-vector degrees of freedom.
   */
   virtual void NormalDerivativeAddMultTranspose(const Vector &x, Vector &y) const
   {
      MFEM_ABORT("Not implemented for this restriction operator.");
   }

   /// @brief Low-level access to the underlying gather map.
   virtual const Array<int> &GatherMap() const
   {
      MFEM_ABORT("Not implemented for this restriction operator.");
   }
};

/// @brief Operator that extracts face degrees of freedom for H1, ND, or RT
/// FiniteElementSpaces.
///
/// Objects of this type are typically created and owned by FiniteElementSpace
/// objects, see FiniteElementSpace::GetFaceRestriction().
class ConformingFaceRestriction : public FaceRestriction
{
protected:
   const FiniteElementSpace &fes;
   const int nf; // Number of faces of the requested type
   const int vdim;
   const bool byvdim;
   const int face_dofs; // Number of dofs on each face
   const int elem_dofs; // Number of dofs in each element
   const int nfdofs; // Total number of face E-vector dofs
   const int ndofs; // Total number of dofs
   Array<int> scatter_indices; // Scattering indices for element 1 on each face
   Array<int> gather_offsets; // offsets for the gathering indices of each dof
   Array<int> gather_indices; // gathering indices for each dof
   Array<int> vol_dof_map; // mapping from lexicographic to native ordering

   /** @brief Construct a ConformingFaceRestriction.

       @param[in] fes        The FiniteElementSpace on which this operates
       @param[in] f_ordering Request a specific face dof ordering
       @param[in] type       Request internal or boundary faces dofs
       @param[in] build      Request the NCL2FaceRestriction to compute the
                             scatter/gather indices. False should only be used
                             when inheriting from ConformingFaceRestriction.
   */
   ConformingFaceRestriction(const FiniteElementSpace& fes,
                             const ElementDofOrdering f_ordering,
                             const FaceType type,
                             bool build);
public:
   /** @brief Construct a ConformingFaceRestriction.

       @param[in] fes        The FiniteElementSpace on which this operates
       @param[in] f_ordering Request a specific face dof ordering
       @param[in] type       Request internal or boundary faces dofs */
   ConformingFaceRestriction(const FiniteElementSpace& fes,
                             const ElementDofOrdering f_ordering,
                             const FaceType type);

   /** @brief Scatter the degrees of freedom, i.e. goes from L-Vector to
       face E-Vector.

       @param[in]  x The L-vector degrees of freedom.
       @param[out] y The face E-Vector degrees of freedom with the given format:
                     face_dofs x vdim x nf
                     where nf is the number of interior or boundary faces
                     requested by @a type in the constructor.
                     The face_dofs are ordered according to the given
                     ElementDofOrdering. */
   void Mult(const Vector &x, Vector &y) const override
   { MultInternal(x, y); }

   /// Compute Mult without applying signs based on DOF orientations.
   void AbsMult(const Vector &x, Vector &y) const override
   { MultInternal(x, y, true); }

   /// @deprecated Use AbsMult() instead.
   MFEM_DEPRECATED void MultUnsigned(const Vector &x, Vector &y) const
   { AbsMult(x, y); }

   using FaceRestriction::AddMultTransposeInPlace;

   /** @brief Gather the degrees of freedom, i.e. goes from face E-Vector to
       L-Vector.

       @param[in]  x The face E-Vector degrees of freedom with the given format:
                     face_dofs x vdim x nf
                     where nf is the number of interior or boundary faces
                     requested by @a type in the constructor.
                     The face_dofs should be ordered according to the given
                     ElementDofOrdering
       @param[in,out] y The L-vector degrees of freedom.
       @param[in]  a Scalar coefficient for addition. */
   void AddMultTranspose(const Vector &x, Vector &y,
                         const real_t a = 1.0) const override;

   /** @brief Gather the degrees of freedom, i.e. goes from face E-Vector to
       L-Vector @b not taking into account signs from DOF orientations.

       @sa AddMultTranspose(). */
   void AddAbsMultTranspose(const Vector &x, Vector &y,
                            const real_t a = 1.0) const override;

   /// @deprecated Use AddAbsMultTranspose() instead.
   MFEM_DEPRECATED void AddMultTransposeUnsigned(const Vector &x, Vector &y) const
   {
      AddAbsMultTranspose(x, y);
   }

   void AbsMultTranspose(const Vector &x, Vector &y) const override
   {
      y = 0.0;
      AddAbsMultTranspose(x, y);
   }

private:
   /** @brief Compute the scatter indices: L-vector to E-vector, and the offsets
       for the gathering: E-vector to L-vector.

       @param[in] f_ordering Request a specific face dof ordering.
       @param[in] type       Request internal or boundary faces dofs.
   */
   void ComputeScatterIndicesAndOffsets(const ElementDofOrdering f_ordering,
                                        const FaceType type);

   /** @brief Compute the gather indices: E-vector to L-vector.

       Note: Requires the gather offsets to be computed.

       @param[in] f_ordering Request a specific face dof ordering.
       @param[in] type       Request internal or boundary faces dofs.
   */
   void ComputeGatherIndices(const ElementDofOrdering f_ordering,
                             const FaceType type);

protected:
   mutable Array<int> face_map; // Used in the computation of GetFaceDofs

   /** @brief Verify that ConformingFaceRestriction is built from a supported
       finite element space.

       @param[in] f_ordering The requested face dof ordering.
   */
   void CheckFESpace(const ElementDofOrdering f_ordering);

   /** @brief Set the scattering indices of elem1, and increment the offsets for
       the face described by the @a face.

       @param[in] face        The face information of the current face.
       @param[in] face_index  The interior/boundary face index.
       @param[in] f_ordering  Request a specific face dof ordering.
    */
   void SetFaceDofsScatterIndices(const Mesh::FaceInformation &face,
                                  const int face_index,
                                  const ElementDofOrdering f_ordering);

   /** @brief Set the gathering indices of elem1 for the interior face described
       by the @a face.

       @param[in] face        The face information of the current face.
       @param[in] face_index  The interior/boundary face index.
       @param[in] f_ordering  Request a specific face dof ordering.
    */
   void SetFaceDofsGatherIndices(const Mesh::FaceInformation &face,
                                 const int face_index,
                                 const ElementDofOrdering f_ordering);

public:
   // This method needs to be public due to 'nvcc' restriction.
   void MultInternal(const Vector &x, Vector &y,
                     const bool useAbs = false) const;
};

/// @brief Alias for ConformingFaceRestriction, for backwards compatibility and
/// as base class for ParNCH1FaceRestriction.
using H1FaceRestriction = ConformingFaceRestriction;

/// Operator that extracts Face degrees of freedom for L2 spaces.
/** Objects of this type are typically created and owned by FiniteElementSpace
    objects, see FiniteElementSpace::GetFaceRestriction(). */
class L2FaceRestriction : public FaceRestriction
{
protected:
   const FiniteElementSpace &fes;
   const ElementDofOrdering ordering;
   const int nf; // Number of faces of the requested type
   const int ne; // Number of elements
   const int vdim; // vdim
   const bool byvdim;
   const int face_dofs; // Number of dofs on each face
   const int elem_dofs; // Number of dofs in each element
   const int nfdofs; // Total number of dofs on the faces
   const int ndofs; // Total number of dofs
   const FaceType type;
   const L2FaceValues m;
   Array<int> scatter_indices1; // Scattering indices for element 1 on each face
   Array<int> scatter_indices2; // Scattering indices for element 2 on each face
   Array<int> gather_offsets; // offsets for the gathering indices of each dof
   Array<int> gather_indices; // gathering indices for each dof
   mutable std::unique_ptr<L2NormalDerivativeFaceRestriction> normal_deriv_restr;

   /** @brief Constructs an L2FaceRestriction.

       @param[in] fes        The FiniteElementSpace on which this operates
       @param[in] f_ordering Request a specific face dof ordering
       @param[in] type       Request internal or boundary faces dofs
       @param[in] m          Request the face dofs for elem1, or both elem1 and
                             elem2
       @param[in] build      Request the NCL2FaceRestriction to compute the
                             scatter/gather indices. False should only be used
                             when inheriting from L2FaceRestriction.
   */
   L2FaceRestriction(const FiniteElementSpace& fes,
                     const ElementDofOrdering f_ordering,
                     const FaceType type,
                     const L2FaceValues m,
                     bool build);

public:
   /** @brief Constructs an L2FaceRestriction.

       @param[in] fes        The FiniteElementSpace on which this operates
       @param[in] f_ordering Request a specific face dof ordering
       @param[in] type       Request internal or boundary faces dofs
       @param[in] m          Request the face dofs for elem1, or both elem1 and
                             elem2 */
   L2FaceRestriction(const FiniteElementSpace& fes,
                     const ElementDofOrdering f_ordering,
                     const FaceType type,
                     const L2FaceValues m = L2FaceValues::DoubleValued);

   /** @brief Scatter the degrees of freedom, i.e. goes from L-Vector to
       face E-Vector.

       @param[in]  x The L-vector degrees of freedom.
       @param[out] y The face E-Vector degrees of freedom with the given format:
                     if L2FacesValues::DoubleValued (face_dofs x vdim x 2 x nf)
                     if L2FacesValues::SingleValued (face_dofs x vdim x nf)
                     where nf is the number of interior or boundary faces
                     requested by @a type in the constructor.
                     The face_dofs are ordered according to the given
                     ElementDofOrdering. */
   void Mult(const Vector &x, Vector &y) const override;

   using FaceRestriction::AddMultTranspose;

   /** @brief Gather the degrees of freedom, i.e. goes from face E-Vector to
       L-Vector.

       @param[in]  x The face E-Vector degrees of freedom with the given format:
                     if L2FacesValues::DoubleValued (face_dofs x vdim x 2 x nf)
                     if L2FacesValues::SingleValued (face_dofs x vdim x nf)
                     where nf is the number of interior or boundary faces
                     requested by @a type in the constructor.
                     The face_dofs should be ordered according to the given
                     ElementDofOrdering
       @param[in,out] y The L-vector degrees of freedom.
       @param[in]  a Scalar coefficient for addition. */
   void AddMultTranspose(const Vector &x, Vector &y,
                         const real_t a = 1.0) const override;

   /** @brief Fill the I array of SparseMatrix corresponding to the sparsity
       pattern given by this L2FaceRestriction.

       @param[in,out] mat The sparse matrix for which we want to initialize the
                          row offsets.
       @param[in] keep_nbr_block When set to true the SparseMatrix will
                                 include the rows (in addition to the columns)
                                 corresponding to face-neighbor dofs. The
                                 default behavior is to disregard those rows. */
   virtual void FillI(SparseMatrix &mat, const bool keep_nbr_block = false) const;

   /** @brief Fill the J and Data arrays of the SparseMatrix corresponding to
       the sparsity pattern given by this L2FaceRestriction, and the values of
       fea_data.

       @param[in] fea_data The dense matrices representing the local operators
                           on each face. The format is:
                           face_dofs x face_dofs x 2 x nf
                           On each face the first local matrix corresponds to
                           the contribution of elem1 on elem2, and the second to
                           the contribution of elem2 on elem1.
       @param[in,out] mat The sparse matrix that is getting filled.
       @param[in] keep_nbr_block When set to true the SparseMatrix will
                                 include the rows (in addition to the columns)
                                 corresponding to face-neighbor dofs. The
                                 default behavior is to disregard those rows. */
   virtual void FillJAndData(const Vector &fea_data,
                             SparseMatrix &mat,
                             const bool keep_nbr_block = false) const;

   /** @brief This methods adds the DG face matrices to the element matrices.

       @param[in] fea_data The dense matrices representing the local operators
                           on each face. The format is:
                           face_dofs x face_dofs x 2 x nf
                           On each face the first and second local matrices
                           correspond to the contributions of elem1 and elem2 on
                           themselves respectively.
       @param[in,out] ea_data The dense matrices representing the element local
                              contributions for each element to which will be
                              added the face contributions.
                              The format is: dofs x dofs x ne, where dofs is the
                              number of dofs per element and ne the number of
                              elements. */
   virtual void AddFaceMatricesToElementMatrices(const Vector &fea_data,
                                                 Vector &ea_data) const;

   /** @brief Scatter the degrees of freedom, i.e. goes from L-Vector to
      face E-Vector.

      @param[in]  x The L-vector degrees of freedom.
      @param[out] y The face E-Vector degrees of freedom with the given format:
                    (face_dofs x vdim x 2 x nf) where nf is the number of
                    interior or boundary faces requested by @a type in the
                    constructor. The face_dofs are ordered according to the
                    given ElementDofOrdering. */
   void NormalDerivativeMult(const Vector &x, Vector &y) const override;

   /** @brief Add the face reference-normal derivative degrees of freedom in @a
              x to the element degrees of freedom in @a y.

       @details see NormalDerivativeMult.

       @param[in]     x The degrees of freedom of the face reference-normal
                        derivative. Is E-vector like.
       @param[in,out] y The L-vector degrees of freedom.
   */
   void NormalDerivativeAddMultTranspose(const Vector &x,
                                         Vector &y) const override;
private:
   /** @brief Compute the scatter indices: L-vector to E-vector, and the offsets
       for the gathering: E-vector to L-vector.
   */
   void ComputeScatterIndicesAndOffsets();

   /** @brief Compute the gather indices: E-vector to L-vector.

       Note: Requires the gather offsets to be computed.
   */
   void ComputeGatherIndices();

   /// Create the internal normal derivative restriction operator if needed.
   void EnsureNormalDerivativeRestriction() const;

protected:
   mutable Array<int> face_map; // Used in the computation of GetFaceDofs

   /** @brief Verify that L2FaceRestriction is built from an L2 FESpace.
   */
   void CheckFESpace();

   /** @brief Set the scattering indices of elem1, and increment the offsets for
       the face described by the @a face. The ordering of the face dofs of elem1
       is lexicographic relative to elem1.

       @param[in] face The face information of the current face.
       @param[in] face_index The interior/boundary face index.
    */
   void SetFaceDofsScatterIndices1(const Mesh::FaceInformation &face,
                                   const int face_index);

   /** @brief Permute and set the scattering indices of elem2, and increment the
       offsets for the face described by the @a face. The permutation orders the
       dofs of elem2 lexicographically as the ones of elem1.

       @param[in] face The face information of the current face.
       @param[in] face_index The interior/boundary face index.
    */
   void PermuteAndSetFaceDofsScatterIndices2(const Mesh::FaceInformation &face,
                                             const int face_index);

   /** @brief Permute and set the scattering indices of elem2 for the shared
       face described by the @a face. The permutation orders the dofs of elem2 as
       the ones of elem1.

       @param[in] face The face information of the current face.
       @param[in] face_index The interior/boundary face index.
    */
   void PermuteAndSetSharedFaceDofsScatterIndices2(
      const Mesh::FaceInformation &face,
      const int face_index);

   /** @brief Set the scattering indices of elem2 for the boundary face
       described by the @a face.

       @param[in] face The face information of the current face.
       @param[in] face_index The interior/boundary face index.
    */
   void SetBoundaryDofsScatterIndices2(const Mesh::FaceInformation &face,
                                       const int face_index);

   /** @brief Set the gathering indices of elem1 for the interior face described
       by the @a face.

       Note: This function modifies the offsets.

       @param[in] face The face information of the current face.
       @param[in] face_index The interior/boundary face index.
    */
   void SetFaceDofsGatherIndices1(const Mesh::FaceInformation &face,
                                  const int face_index);

   /** @brief Permute and set the gathering indices of elem2 for the interior
       face described by the @a face. The permutation orders the dofs of elem2 as
       the ones of elem1.

       Note: This function modifies the offsets.

       @param[in] face The face information of the current face.
       @param[in] face_index The interior/boundary face index.
    */
   void PermuteAndSetFaceDofsGatherIndices2(const Mesh::FaceInformation &face,
                                            const int face_index);

public:
   /** @brief Scatter the degrees of freedom, i.e. goes from L-Vector to
       face E-Vector. Should only be used with conforming faces and when:
       m == L2FacesValues::SingleValued

       @param[in]  x The L-vector degrees of freedom.
       @param[out] y The face E-Vector degrees of freedom with the given format:
                     face_dofs x vdim x nf
                     where nf is the number of interior or boundary faces
                     requested by @a type in the constructor.
                     The face_dofs are ordered according to the given
                     ElementDofOrdering. */
   void SingleValuedConformingMult(const Vector& x, Vector& y) const;

   /** @brief Scatter the degrees of freedom, i.e. goes from L-Vector to
       face E-Vector. Should only be used with conforming faces and when:
       m == L2FacesValues::DoubleValued

       @param[in]  x The L-vector degrees of freedom.
       @param[out] y The face E-Vector degrees of freedom with the given format:
                     face_dofs x vdim x 2 x nf
                     where nf is the number of interior or boundary faces
                     requested by @a type in the constructor.
                     The face_dofs are ordered according to the given
                     ElementDofOrdering. */
   virtual void DoubleValuedConformingMult(const Vector& x, Vector& y) const;

   /** @brief Gather the degrees of freedom, i.e. goes from face E-Vector to
       L-Vector. Should only be used with conforming faces and when:
       m == L2FacesValues::SingleValued

       @param[in]  x The face E-Vector degrees of freedom with the given format:
                     face_dofs x vdim x nf
                     where nf is the number of interior or boundary faces
                     requested by @a type in the constructor.
                     The face_dofs should be ordered according to the given
                     ElementDofOrdering
       @param[in,out] y The L-vector degrees of freedom. */
   void SingleValuedConformingAddMultTranspose(const Vector& x, Vector& y) const;

   /** @brief Gather the degrees of freedom, i.e. goes from face E-Vector to
       L-Vector. Should only be used with conforming faces and when:
       m == L2FacesValues::DoubleValued

       @param[in]  x The face E-Vector degrees of freedom with the given format:
                     face_dofs x vdim x 2 x nf
                     where nf is the number of interior or boundary faces
                     requested by @a type in the constructor.
                     The face_dofs should be ordered according to the given
                     ElementDofOrdering
       @param[in,out] y The L-vector degrees of freedom. */
   void DoubleValuedConformingAddMultTranspose(const Vector& x, Vector& y) const;
};

/** This struct stores which side is the master nonconforming side and the
    index of the interpolator, see InterpolationManager class below. */
struct InterpConfig
{
   uint32_t is_non_conforming : 1;
   uint32_t master_side : 1;
   uint32_t index : 30;

   // default constructor, create a conforming face with index 0.
   InterpConfig() = default;

   // Non-conforming face
   InterpConfig(int master_side, int nc_index)
      : is_non_conforming(1), master_side(master_side), index(nc_index)
   { }

   InterpConfig(const InterpConfig&) = default;

   InterpConfig &operator=(const InterpConfig &rhs) = default;
};

/** This struct stores which side is the master nonconforming side and the
    index of the interpolator, see InterpolationManager class below. */
struct NCInterpConfig
{
   int face_index;
   uint32_t is_non_conforming : 1;
   uint32_t master_side : 1;
   uint32_t index : 30;

   // default constructor.
   NCInterpConfig() = default;

   // Non-conforming face
   NCInterpConfig(int face_index, int master_side, int nc_index)
      : face_index(face_index),
        is_non_conforming(1),
        master_side(master_side),
        index(nc_index)
   { }

   // Non-conforming face
   NCInterpConfig(int face_index, InterpConfig & config)
      : face_index(face_index),
        is_non_conforming(config.is_non_conforming),
        master_side(config.master_side),
        index(config.index)
   { }

   NCInterpConfig(const NCInterpConfig&) = default;

   NCInterpConfig &operator=(const NCInterpConfig &rhs) = default;
};

/** @brief This class manages the storage and computation of the interpolations
    from master (coarse) face to slave (fine) face.
*/
class InterpolationManager
{
protected:
   const FiniteElementSpace &fes;
   const ElementDofOrdering ordering;
   Array<InterpConfig> interp_config; // interpolator index for each face
   Array<NCInterpConfig> nc_interp_config; // interpolator index for each ncface
   Vector interpolators; // face_dofs x face_dofs x num_interpolators
   int nc_cpt; // Counter for interpolators, and used as index.

   /** The interpolators are associated to a key of containing the address of
       PointMatrix and a local face identifier. */
   using Key = std::pair<const DenseMatrix*,int>;
   /// The temporary map used to store the different interpolators.
   using Map = std::map<Key, std::pair<int,const DenseMatrix*>>;
   Map interp_map; // The temporary map that stores the interpolators.

public:
   InterpolationManager() = delete;

   /** @brief main constructor.

       @param[in] fes      The FiniteElementSpace on which this operates
       @param[in] ordering Request a specific element ordering.
       @param[in] type     Request internal or boundary faces dofs
    */
   InterpolationManager(const FiniteElementSpace &fes,
                        ElementDofOrdering ordering,
                        FaceType type);

   /** @brief Register the face with @a face and index @a face_index as a
       conforming face for the interpolation of the degrees of freedom.

       @param[in] face The face information of the current face.
       @param[in] face_index The interior/boundary face index.
    */
   void RegisterFaceConformingInterpolation(const Mesh::FaceInformation &face,
                                            int face_index);

   /** @brief Register the face with @a face and index @a face_index as a
       conforming face for the interpolation of the degrees of freedom.

       @param[in] face The face information of the current face.
       @param[in] face_index The interior/boundary face index.
    */
   void RegisterFaceCoarseToFineInterpolation(const Mesh::FaceInformation &face,
                                              int face_index);

   /** @brief Transform the interpolation matrix map into a contiguous memory
       structure. */
   void LinearizeInterpolatorMapIntoVector();

   void InitializeNCInterpConfig();

   /// @brief Return the total number of interpolators.
   int GetNumInterpolators() const
   {
      return nc_cpt;
   }

   /** @brief Return an mfem::Vector containing the interpolators in the
       following format: face_dofs x face_dofs x num_interpolators. */
   const Vector& GetInterpolators() const
   {
      return interpolators;
   }

   /** @brief Return an array containing the interpolation configuration for
       each face registered with RegisterFaceConformingInterpolation and
       RegisterFaceCoarseToFineInterpolation. */
   const Array<InterpConfig>& GetFaceInterpConfig() const
   {
      return interp_config;
   }

   /** @brief Return an array containing the interpolation configuration for
       each face registered with RegisterFaceConformingInterpolation and
       RegisterFaceCoarseToFineInterpolation. */
   const Array<NCInterpConfig>& GetNCFaceInterpConfig() const
   {
      return nc_interp_config;
   }

private:
   /** @brief Returns the interpolation operator from a master (coarse) face to
       a slave (fine) face.

       @param[in] face The face information of the current face.
       @param[in] ptMat The PointMatrix describing the position and orientation
                        of the fine face in the coarse face. This PointMatrix is
                        usually obtained from the mesh through the method
                        GetNCFacesPtMat.
       @param[in] ordering  Request a specific element ordering.
       @return The dense matrix corresponding to the interpolation of the face
               degrees of freedom of the master (coarse) face to the slave
               (fine) face. */
   const DenseMatrix* GetCoarseToFineInterpolation(
      const Mesh::FaceInformation &face,
      const DenseMatrix* ptMat);
};

/** @brief Operator that extracts face degrees of freedom for L2 nonconforming
    spaces.

    In order to support face restrictions on nonconforming meshes, this
    operator interpolates master (coarse) face degrees of freedom onto the
    slave (fine) face. This allows face integrators to treat nonconforming
    faces just as regular conforming faces. */
class NCL2FaceRestriction : virtual public L2FaceRestriction
{
protected:
   InterpolationManager interpolations;
   mutable Vector x_interp;

   /** @brief Constructs an NCL2FaceRestriction, this is a specialization of a
       L2FaceRestriction for nonconforming meshes.

       @param[in] fes        The FiniteElementSpace on which this operates
       @param[in] f_ordering Request a specific face dof ordering
       @param[in] type       Request internal or boundary faces dofs
       @param[in] m          Request the face dofs for elem1, or both elem1 and
                             elem2
       @param[in] build      Request the NCL2FaceRestriction to compute the
                             scatter/gather indices. False should only be used
                             when inheriting from NCL2FaceRestriction.
   */
   NCL2FaceRestriction(const FiniteElementSpace& fes,
                       const ElementDofOrdering f_ordering,
                       const FaceType type,
                       const L2FaceValues m,
                       bool build);
public:
   /** @brief Constructs an NCL2FaceRestriction, this is a specialization of a
       L2FaceRestriction for nonconforming meshes.

       @param[in] fes        The FiniteElementSpace on which this operates
       @param[in] f_ordering Request a specific face dof ordering
       @param[in] type       Request internal or boundary faces dofs
       @param[in] m          Request the face dofs for elem1, or both elem1 and
                             elem2
   */
   NCL2FaceRestriction(const FiniteElementSpace& fes,
                       const ElementDofOrdering f_ordering,
                       const FaceType type,
                       const L2FaceValues m = L2FaceValues::DoubleValued);

   /** @brief Scatter the degrees of freedom, i.e. goes from L-Vector to
       face E-Vector.

       @param[in]  x The L-vector degrees of freedom.
       @param[out] y The face E-Vector degrees of freedom with the given format:
                     if L2FacesValues::DoubleValued (face_dofs x vdim x 2 x nf),
                     if L2FacesValues::SingleValued (face_dofs x vdim x nf),
                     where nf is the number of interior or boundary faces
                     requested by @a type in the constructor.
                     The face_dofs are ordered according to the given
                     ElementDofOrdering. */
   void Mult(const Vector &x, Vector &y) const override;

   /** @brief Gather the degrees of freedom, i.e. goes from face E-Vector to
       L-Vector.

       @param[in]  x The face E-Vector degrees of freedom with the given format:
                     if L2FacesValues::DoubleValued (face_dofs x vdim x 2 x nf),
                     if L2FacesValues::SingleValued (face_dofs x vdim x nf),
                     where nf is the number of interior or boundary faces
                     requested by @a type in the constructor.
                     The face_dofs should be ordered according to the given
                     ElementDofOrdering
       @param[in,out] y The L-vector degrees of freedom.
       @param[in]  a Scalar coefficient for addition. */
   void AddMultTranspose(const Vector &x, Vector &y,
                         const real_t a = 1.0) const override;

   /** @brief Gather the degrees of freedom, i.e. goes from face E-Vector to
       L-Vector.

       @param[in,out]  x The face E-Vector degrees of freedom with the given format:
                         if L2FacesValues::DoubleValued (face_dofs x vdim x 2 x nf),
                         if L2FacesValues::SingleValued (face_dofs x vdim x nf),
                         where nf is the number of interior or boundary faces
                         requested by @a type in the constructor.
                         The face_dofs should be ordered according to the given
                         ElementDofOrdering
       @param[in,out] y The L-vector degrees of freedom.

      @note This method is an optimization of AddMultTranspose where the @a x
      Vector is used and modified to avoid memory allocation and memcpy. */
   void AddMultTransposeInPlace(Vector &x, Vector &y) const override;

   /** @brief Fill the I array of SparseMatrix corresponding to the sparsity
       pattern given by this NCL2FaceRestriction.

       @param[in,out] mat The sparse matrix for which we want to initialize the
                          row offsets.
       @param[in] keep_nbr_block When set to true the SparseMatrix will
                                 include the rows (in addition to the columns)
                                 corresponding to face-neighbor dofs. The
                                 default behavior is to disregard those rows.

       @warning This method is not implemented yet. */
   void FillI(SparseMatrix &mat,
              const bool keep_nbr_block = false) const override;

   /** @brief Fill the J and Data arrays of the SparseMatrix corresponding to
       the sparsity pattern given by this NCL2FaceRestriction, and the values of
       ea_data.

       @param[in] fea_data The dense matrices representing the local operators
                           on each face. The format is:
                           face_dofs x face_dofs x 2 x nf.
                           On each face the first local matrix corresponds to
                           the contribution of elem1 on elem2, and the second to
                           the contribution of elem2 on elem1.
       @param[in,out] mat The sparse matrix that is getting filled.
       @param[in] keep_nbr_block When set to true the SparseMatrix will
                                 include the rows (in addition to the columns)
                                 corresponding to face-neighbor dofs. The
                                 default behavior is to disregard those rows.

       @warning This method is not implemented yet. */
   void FillJAndData(const Vector &fea_data,
                     SparseMatrix &mat,
                     const bool keep_nbr_block = false) const override;

   /** @brief This methods adds the DG face matrices to the element matrices.

       @param[in] fea_data The dense matrices representing the local operators
                           on each face. The format is:
                           face_dofs x face_dofs x 2 x nf.
                           On each face the first and second local matrices
                           correspond to the contributions of elem1 and elem2 on
                           themselves respectively.
       @param[in,out] ea_data The dense matrices representing the element local
                              contributions for each element to which will be
                              added the face contributions.
                              The format is: dofs x dofs x ne, where dofs is the
                              number of dofs per element and ne the number of
                              elements.

       @warning This method is not implemented yet. */
   void AddFaceMatricesToElementMatrices(const Vector &fea_data,
                                         Vector &ea_data) const override;

private:
   /** @brief Compute the scatter indices: L-vector to E-vector, the offsets
       for the gathering: E-vector to L-vector, and the interpolators from
       coarse to fine face for master non-comforming faces.
   */
   void ComputeScatterIndicesAndOffsets();

   /** @brief Compute the gather indices: E-vector to L-vector.

       Note: Requires the gather offsets to be computed.
   */
   void ComputeGatherIndices();

public:
   /** @brief Scatter the degrees of freedom, i.e. goes from L-Vector to
       face E-Vector. Should only be used with nonconforming faces and when:
       L2FaceValues m == L2FaceValues::DoubleValued

       @param[in]  x The L-vector degrees of freedom.
       @param[out] y The face E-Vector degrees of freedom with the given format:
                     (face_dofs x vdim x 2 x nf),
                     where nf is the number of interior or boundary faces
                     requested by @a type in the constructor.
                     The face_dofs are ordered according to the given
                     ElementDofOrdering. */
   virtual void DoubleValuedNonconformingMult(const Vector& x, Vector& y) const;

   /** @brief Apply a change of basis from coarse element basis to fine element
       basis for the coarse face dofs.

       @param[in,out] x The dofs vector that needs coarse dofs to be express in
                        term of the fine basis.
   */
   void DoubleValuedNonconformingInterpolation(Vector& x) const;

   /** @brief Apply a change of basis from fine element basis to coarse element
       basis for the coarse face dofs. Should only be used when:
       L2FaceValues m == L2FaceValues::SingleValued

       @param[in] x The dofs vector that needs coarse dofs to be express in term
                    of the coarse basis, the result is stored in x_interp.
   */
   void SingleValuedNonconformingTransposeInterpolation(const Vector& x) const;

   /** @brief Apply a change of basis from fine element basis to coarse element
       basis for the coarse face dofs. Should only be used when:
       L2FaceValues m == L2FaceValues::SingleValued

       @param[in,out] x The dofs vector that needs coarse dofs to be express in
                        term of the coarse basis, the result is stored in x.
   */
   void SingleValuedNonconformingTransposeInterpolationInPlace(Vector& x) const;

   /** @brief Apply a change of basis from fine element basis to coarse element
       basis for the coarse face dofs. Should only be used when:
       L2FaceValues m == L2FaceValues::DoubleValued

       @param[in] x The dofs vector that needs coarse dofs to be express in term
                    of the coarse basis, the result is stored in x_interp.
   */
   void DoubleValuedNonconformingTransposeInterpolation(const Vector& x) const;

   /** @brief Apply a change of basis from fine element basis to coarse element
       basis for the coarse face dofs. Should only be used when:
       L2FaceValues m == L2FaceValues::DoubleValued

       @param[in,out] x The dofs vector that needs coarse dofs to be express in
                        term of the coarse basis, the result is stored in
                        x.
   */
   void DoubleValuedNonconformingTransposeInterpolationInPlace(Vector& x) const;
};

/// Operator that extracts face degrees of freedom for L2 interface spaces.
/** Objects of this type are typically created and owned by FiniteElementSpace
    objects, see FiniteElementSpace::GetFaceRestriction(). */
class L2InterfaceFaceRestriction : public FaceRestriction
{
protected:
   const FiniteElementSpace &fes; ///< The finite element space
   const ElementDofOrdering ordering; ///< Requested ordering
   const FaceType type; ///< Face type (interior or boundary)
   const int nfaces; ///< Number of faces of the requested type
   const int vdim; ///< vdim of the space
   const bool byvdim; ///< DOF ordering (by nodes or by vdim)
   const int face_dofs; ///< Number of dofs on each face
   const int nfdofs; ///< Total number of dofs on the faces (E-vector size)
   const int ndofs; ///< Number of dofs in the space (L-vector size)
   Array<int> gather_map; ///< Gather map

public:
   /** @brief Constructs an L2InterfaceFaceRestriction.

       @param[in] fes_       The FiniteElementSpace on which this operates
       @param[in] ordering_  Request a specific face dof ordering
       @param[in] type_      Request internal or boundary faces dofs */
   L2InterfaceFaceRestriction(const FiniteElementSpace& fes_,
                              const ElementDofOrdering ordering_,
                              const FaceType type_);

   /** @brief Scatter the degrees of freedom, i.e. goes from L-Vector to
       face E-Vector.

       @param[in]  x The L-vector degrees of freedom.
       @param[out] y The face E-Vector degrees of freedom with size (face_dofs,
                     vdim, nf), where nf is the number of interior or boundary
                     faces requested by @a type in the constructor. The
                     face_dofs are ordered according to the given
                     ElementDofOrdering. */
   void Mult(const Vector &x, Vector &y) const override;

   using FaceRestriction::AddMultTranspose;

   /** @brief Gather the degrees of freedom, i.e. goes from face E-Vector to
       L-Vector.

       @param[in]     x The face E-Vector degrees of freedom with size
                        (face_dofs, vdim, nf), where nf is the number of
                        interior or boundary faces requested by @a type in the
                        constructor. The face_dofs should be ordered according
                        to the given ElementDofOrdering
       @param[in,out] y The L-vector degrees of freedom.
       @param[in]     a Scalar coefficient for addition. */
   void AddMultTranspose(const Vector &x, Vector &y,
                         const real_t a = 1.0) const override;

   const Array<int> &GatherMap() const override;
};

/** @brief Convert a dof face index from Native ordering to lexicographic
    ordering for quads and hexes.

    @param[in] dim The dimension of the element, 2 for quad, 3 for hex
    @param[in] face_id The local face identifier
    @param[in] size1d The 1D number of degrees of freedom for each dimension
    @param[in] index The native index on the face
    @return The lexicographic index on the face
*/
int ToLexOrdering(const int dim, const int face_id, const int size1d,
                  const int index);

/** @brief Compute the dof face index of elem2 corresponding to the given dof
    face index.

    @param[in] dim The dimension of the element, 2 for quad, 3 for hex
    @param[in] face_id1 The local face identifier of elem1
    @param[in] face_id2 The local face identifier of elem2
    @param[in] orientation The orientation of elem2 relative to elem1 on the
                           face
    @param[in] size1d The 1D number of degrees of freedom for each dimension
    @param[in] index The dof index on elem1
    @return The dof index on elem2 facing the dof on elem1
*/
int PermuteFaceL2(const int dim, const int face_id1,
                  const int face_id2, const int orientation,
                  const int size1d, const int index);

/// @brief Return the face-neighbor data given the L-vector @a x.
///
/// If the input vector @a x is a ParGridFunction with non-empty face-neighbor
/// data, return an alias to ParGridFunction::FaceNbrData() (avoiding an
/// unneeded call to ParGridFunction::ExchangeFaceNbrData).
///
/// Otherwise, create a temporary ParGridFunction, exchange the face-neighbor
/// data, and return the resulting vector.
///
/// If @a fes is not a parallel space, or if @a ftype is not FaceType::Interior,
/// return an empty vector.
Vector GetLVectorFaceNbrData(
   const FiniteElementSpace &fes, const Vector &x, FaceType ftype);

}

#endif // MFEM_RESTRICTION
