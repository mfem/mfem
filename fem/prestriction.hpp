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

#ifndef MFEM_PRESTRICTION
#define MFEM_PRESTRICTION

#include "../config/config.hpp"

#ifdef MFEM_USE_MPI

#include "restriction.hpp"

namespace mfem
{

class ParFiniteElementSpace;

/// Operator that extracts Face degrees of freedom for NCMesh in parallel.
/** Objects of this type are typically created and owned by
    ParFiniteElementSpace objects, see
    ParFiniteElementSpace::GetFaceRestriction(). */
class ParNCH1FaceRestriction : public H1FaceRestriction
{
protected:
   const FaceType type;
   InterpolationManager interpolations;
   mutable Vector x_interp;

public:
   /** @brief Constructs an ParNCH1FaceRestriction.

       @param[in] fes        The ParFiniteElementSpace on which this operates
       @param[in] f_ordering Request a specific face dof ordering
       @param[in] type       Request internal or boundary faces dofs */
   ParNCH1FaceRestriction(const ParFiniteElementSpace &fes,
                          ElementDofOrdering f_ordering,
                          FaceType type);

   /** @brief Scatter the degrees of freedom, i.e. goes from L-Vector to
       face E-Vector.

       @param[in]  x The L-vector degrees of freedom.
       @param[out] y The face E-Vector degrees of freedom with the given format:
                     face_dofs x vdim x nf
                     where nf is the number of interior or boundary faces
                     requested by @a type in the constructor.
                     The face_dofs are ordered according to the given
                     ElementDofOrdering. */
   void Mult(const Vector &x, Vector &y) const override;

   /** @brief Gather the degrees of freedom, i.e. goes from face E-Vector to
       L-Vector.

       @param[in]  x The face E-Vector degrees of freedom with the given format:
                     face_dofs x vdim x nf
                     where nf is the number of interior or boundary faces
                     requested by @a type in the constructor.
                     The face_dofs should be ordered according to the given
                     ElementDofOrdering.
       @param[in,out] y The L-vector degrees of freedom.
       @param[in]  a Scalar coefficient for addition. */
   void AddMultTranspose(const Vector &x, Vector &y,
                         const real_t a = 1.0) const override;

   /** @brief Gather the degrees of freedom, i.e. goes from face E-Vector to
       L-Vector.

       @param[in,out]  x The face E-Vector degrees of freedom with the given format:
                         face_dofs x vdim x nf
                         where nf is the number of interior or boundary faces
                         requested by @a type in the constructor.
                         The face_dofs should be ordered according to the given
                         ElementDofOrdering.
       @param[in,out] y The L-vector degrees of freedom.

      @note This method is an optimization of AddMultTranspose where the @a x
      Vector is used and modified to avoid memory allocation and memcpy. */
   void AddMultTransposeInPlace(Vector &x, Vector &y) const override;

private:
   /** @brief Compute the scatter indices: L-vector to E-vector, the offsets
       for the gathering: E-vector to L-vector, and the interpolators from
       coarse to fine face for master non-comforming faces.

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

public: // For nvcc
   /** @brief Apply a change of basis from coarse element basis to fine element
       basis for the coarse face dofs.

       @param[in,out] x The dofs vector that needs coarse dofs to be express in
                        term of the fine basis.
   */
   void NonconformingInterpolation(Vector& x) const;

   /** @brief Apply a change of basis from fine element basis to coarse element
       basis for the coarse face dofs.

       @param[in] x The dofs vector that needs coarse dofs to be express in term
                    of the coarse basis, the result is stored in x_interp.
   */
   void NonconformingTransposeInterpolation(const Vector& x) const;

   /** @brief Apply a change of basis from fine element basis to coarse element
       basis for the coarse face dofs.

       @param[in] x The dofs vector that needs coarse dofs to be express in term
                    of the coarse basis, the result is stored in x_interp.
   */
   void NonconformingTransposeInterpolationInPlace(Vector& x) const;
};

/// Operator that extracts Face degrees of freedom in parallel.
/** Objects of this type are typically created and owned by
    ParFiniteElementSpace objects, see
    ParFiniteElementSpace::GetFaceRestriction(). */
class ParL2FaceRestriction : virtual public L2FaceRestriction
{
protected:
   const ParFiniteElementSpace &pfes;

   /** @brief Constructs an ParL2FaceRestriction.

       @param[in] pfes_      The ParFiniteElementSpace on which this operates
       @param[in] f_ordering Request a specific face dof ordering
       @param[in] type       Request internal or boundary faces dofs
       @param[in] m          Request the face dofs for elem1, or both elem1 and
                             elem2
       @param[in] build      Request the ParL2FaceRestriction to compute the
                             scatter/gather indices. False should only be used
                             when inheriting from ParL2FaceRestriction. */
   ParL2FaceRestriction(const ParFiniteElementSpace &pfes_,
                        ElementDofOrdering f_ordering,
                        FaceType type,
                        L2FaceValues m,
                        bool build);

public:
   /** @brief Constructs an ParL2FaceRestriction.

       @param[in] fes        The ParFiniteElementSpace on which this operates
       @param[in] f_ordering Request a specific face dof ordering
       @param[in] type       Request internal or boundary faces dofs
       @param[in] m          Request the face dofs for elem1, or both elem1 and
                             elem2 */
   ParL2FaceRestriction(const ParFiniteElementSpace& fes,
                        ElementDofOrdering f_ordering,
                        FaceType type,
                        L2FaceValues m = L2FaceValues::DoubleValued);

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

   /** Fill the I array of SparseMatrix corresponding to the sparsity pattern
       given by this ParL2FaceRestriction.

       @param[in,out] mat The sparse matrix for which we want to initialize the
                          row offsets.
       @param[in] keep_nbr_block When set to true the SparseMatrix will
                                 include the rows (in addition to the columns)
                                 corresponding to face-neighbor dofs. The
                                 default behavior is to disregard those rows. */
   void FillI(SparseMatrix &mat,
              const bool keep_nbr_block = false) const override;

   /** Fill the I array of SparseMatrix corresponding to the sparsity pattern
       given by this ParL2FaceRestriction. @a mat contains the interior dofs
       contribution, the @a face_mat contains the shared dofs contribution.*/
   void FillI(SparseMatrix &mat,
              SparseMatrix &face_mat) const;

   /** Fill the J and Data arrays of SparseMatrix corresponding to the sparsity
       pattern given by this ParL2FaceRestriction, and the values of ea_data.
       @a mat contains the interior dofs contribution, the @a face_mat contains
       the shared dofs contribution.*/
   void FillJAndData(const Vector &ea_data,
                     SparseMatrix &mat,
                     SparseMatrix &face_mat) const;

   /** @brief Fill the J and Data arrays of the SparseMatrix corresponding to
       the sparsity pattern given by this ParL2FaceRestriction, and the values of
       fea_data.

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
                                 default behavior is to disregard those rows. */
   void FillJAndData(const Vector &fea_data,
                     SparseMatrix &mat,
                     const bool keep_nbr_block = false) const override;

private:
   /** @brief Compute the scatter indices: L-vector to E-vector, and the offsets
       for the gathering: E-vector to L-vector.
   */
   void ComputeScatterIndicesAndOffsets();

   /** @brief Compute the gather indices: E-vector to L-vector.

       Note: Requires the gather offsets to be computed.
   */
   void ComputeGatherIndices();

public:
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
   void DoubleValuedConformingMult(const Vector& x, Vector& y) const override;
};

/// Operator that extracts Face degrees of freedom for NCMesh in parallel.
/** Objects of this type are typically created and owned by
    ParFiniteElementSpace objects, see
    ParFiniteElementSpace::GetFaceRestriction(). */
class ParNCL2FaceRestriction
   : public NCL2FaceRestriction, public ParL2FaceRestriction
{
public:
   /** @brief Constructs an ParNCL2FaceRestriction.

       @param[in] fes        The ParFiniteElementSpace on which this operates
       @param[in] f_ordering Request a specific ordering
       @param[in] type       Request internal or boundary faces dofs
       @param[in] m          Request the face dofs for elem1, or both elem1 and
                             elem2 */
   ParNCL2FaceRestriction(const ParFiniteElementSpace& fes,
                          ElementDofOrdering f_ordering,
                          FaceType type,
                          L2FaceValues m = L2FaceValues::DoubleValued);

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

       @note @a x is used for computation. */
   void AddMultTransposeInPlace(Vector &x, Vector &y) const override;

   /** @brief Fill the I array of SparseMatrix corresponding to the sparsity
       pattern given by this ParNCL2FaceRestriction.

       @param[in,out] mat The sparse matrix for which we want to initialize the
                          row offsets.
       @param[in] keep_nbr_block When set to true the SparseMatrix will
                                 include the rows (in addition to the columns)
                                 corresponding to face-neighbor dofs. The
                                 default behavior is to disregard those rows. */
   void FillI(SparseMatrix &mat,
              const bool keep_nbr_block = false) const override;

   /** Fill the I array of SparseMatrix corresponding to the sparsity pattern
       given by this ParNCL2FaceRestriction. @a mat contains the interior dofs
       contribution, the @a face_mat contains the shared dofs contribution.

       @warning This method is not implemented yet. */
   void FillI(SparseMatrix &mat,
              SparseMatrix &face_mat) const;

   /** Fill the J and Data arrays of SparseMatrix corresponding to the sparsity
       pattern given by this ParNCL2FaceRestriction, and the values of ea_data.
       @a mat contains the interior dofs contribution, the @a face_mat contains
       the shared dofs contribution.

       @warning This method is not implemented yet. */
   void FillJAndData(const Vector &fea_data,
                     SparseMatrix &mat,
                     SparseMatrix &face_mat) const;

   /** @brief Fill the J and Data arrays of the SparseMatrix corresponding to
       the sparsity pattern given by this ParNCL2FaceRestriction, and the values
       of ea_data.

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
                                 default behavior is to disregard those rows. */
   void FillJAndData(const Vector &fea_data,
                     SparseMatrix &mat,
                     const bool keep_nbr_block = false) const override;

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
       L2FaceValues m == L2FaceValues::SingleValued

       @param[in]  x The L-vector degrees of freedom.
       @param[out] y The face E-Vector degrees of freedom with the given format:
                     (face_dofs x vdim x nf),
                     where nf is the number of interior or boundary faces
                     requested by @a type in the constructor.
                     The face_dofs are ordered according to the given
                     ElementDofOrdering. */
   void SingleValuedNonconformingMult(const Vector& x, Vector& y) const;

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
   void DoubleValuedNonconformingMult(const Vector& x, Vector& y) const override;
};

}

#endif // MFEM_USE_MPI

#endif // MFEM_PRESTRICTION
