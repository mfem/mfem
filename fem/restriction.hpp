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
   const int nf; // Number of faces of the requested type
   const int vdim;
   const bool byvdim;
   const int ndofs; // Total number of dofs
   const int dof; // Number of dofs on each face
   const int elem_dofs; // Number of dofs in each element
   const int nfdofs; // Total number of face E-vector dofs
   Array<int> scatter_indices; // Scattering indices for element 1 on each face
   Array<int> offsets; // offsets for the gathering indices of each dof
   Array<int> gather_indices; // gathering indices for each dof

   /** @brief Initialize an H1FaceRestriction.

       @param[in] fes      The FiniteElementSpace on which this operates
       @param[in] type     Request internal or boundary faces dofs */
   H1FaceRestriction(const FiniteElementSpace& fes,
                     const FaceType type);

public:
   /** @brief Construct an H1FaceRestriction.

       @param[in] fes      The FiniteElementSpace on which this operates
       @param[in] ordering Request a specific element ordering
       @param[in] type     Request internal or boundary faces dofs */
   H1FaceRestriction(const FiniteElementSpace& fes,
                     const ElementDofOrdering ordering,
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
   void Mult(const Vector &x, Vector &y) const override;

   /** @brief Gather the degrees of freedom, i.e. goes from face E-Vector to
       L-Vector.

       @param[in]  x The face E-Vector degrees of freedom with the given format:
                     face_dofs x vdim x nf
                     where nf is the number of interior or boundary faces
                     requested by @a type in the constructor.
                     The face_dofs should be ordered according to the given
                     ElementDofOrdering
       @param[out] y The L-vector degrees of freedom. */
   void AddMultTranspose(const Vector &x, Vector &y) const override;

protected:
   /** @brief Compute the scatter indices: L-vector to E-vector, and the offsets
       for the gathering: E-vector to L-vector. 

       @param[in] ordering Request a specific element ordering.
       @param[in] type     Request internal or boundary faces dofs.
   */
   void ComputeScatterIndicesAndOffsets(const ElementDofOrdering ordering,
                                        const FaceType type);

   /** @brief Compute the gather indices: E-vector to L-vector.

       Note: Requires the gather offsets to be computed.

       @param[in] ordering Request a specific element ordering.
       @param[in] type     Request internal or boundary faces dofs.
   */
   void ComputeGatherIndices(const ElementDofOrdering ordering,
                             const FaceType type);

   mutable Array<int> face_map; // Used in the computation of GetFaceDofs

   /** @brief Set the scattering indices of elem1, and increment the offsets for
       the face described by the info.

       @param[in] info The face information of the current face.
       @param[in] face_index The interior/boundary face index.
    */
   void SetFaceDofsScatterIndices(const Mesh::FaceInformation &info,
                                  const int face_index,
                                  const ElementDofOrdering ordering);

   /** @brief Set the gathering indices of elem1 for the interior face described
       by the info.

       @param[in] info The face information of the current face.
       @param[in] face_index The interior/boundary face index.
    */
   void SetFaceDofsGatherIndices(const Mesh::FaceInformation &info,
                                 const int face_index,
                                 const ElementDofOrdering ordering);
};

/// Operator that extracts Face degrees of freedom for L2 spaces.
/** Objects of this type are typically created and owned by FiniteElementSpace
    objects, see FiniteElementSpace::GetFaceRestriction(). */
class L2FaceRestriction : public FaceRestriction
{
protected:
   const FiniteElementSpace &fes;
   const int nf; // Number of faces of the requested type
   const int ne; // Number of elements
   const int vdim; // vdim
   const bool byvdim;
   const int ndofs; // Total number of dofs
   const int dof; // Number of dofs on each face
   const int elem_dofs; // Number of dofs in each element
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

       @param[in]  x The L-vector degrees of freedom.
       @param[out] y The face E-Vector degrees of freedom with the given format:
                     if L2FacesValues::DoubleValued (face_dofs x vdim x 2 x nf)
                     if L2FacesValues::SingleValued (face_dofs x vdim x nf)
                     where nf is the number of interior or boundary faces
                     requested by @a type in the constructor.
                     The face_dofs are ordered according to the given
                     ElementDofOrdering. */
   void Mult(const Vector &x, Vector &y) const override;

   /** @brief Gather the degrees of freedom, i.e. goes from face E-Vector to
       L-Vector.

       @param[in]  x The face E-Vector degrees of freedom with the given format:
                     if L2FacesValues::DoubleValued (face_dofs x vdim x 2 x nf)
                     if L2FacesValues::SingleValued (face_dofs x vdim x nf)
                     where nf is the number of interior or boundary faces
                     requested by @a type in the constructor.
                     The face_dofs should be ordered according to the given
                     ElementDofOrdering
       @param[out] y The L-vector degrees of freedom. */
   void AddMultTranspose(const Vector &x, Vector &y) const override;

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

private:
   /** @brief Compute the scatter indices: L-vector to E-vector, and the offsets
       for the gathering: E-vector to L-vector. 

       @param[in] ordering Request a specific element ordering.
       @param[in] type     Request internal or boundary faces dofs.
   */
   void ComputeScatterIndicesAndOffsets(const ElementDofOrdering ordering,
                                        const FaceType type);

   /** @brief Compute the gather indices: E-vector to L-vector.

       Note: Requires the gather offsets to be computed.

       @param[in] ordering Request a specific element ordering.
       @param[in] type     Request internal or boundary faces dofs.
   */
   void ComputeGatherIndices(const ElementDofOrdering ordering,
                             const FaceType type);

protected:
   mutable Array<int> face_map; // Used in the computation of GetFaceDofs

   /** @brief Set the scattering indices of elem1, and increment the offsets for
       the face described by the info.

       @param[in] info The face information of the current face.
       @param[in] face_index The interior/boundary face index.
    */
   void SetFaceDofsScatterIndices1(const Mesh::FaceInformation &info,
                                   const int face_index);

   /** @brief Set the scattering indices of elem2, and increment the offsets for
       the face described by the info.

       @param[in] info The face information of the current face.
       @param[in] face_index The interior/boundary face index.
    */
   void SetFaceDofsScatterIndices2(const Mesh::FaceInformation &info,
                                   const int face_index);

   /** @brief Permute and set the scattering indices of elem2, and increment the
       offsets for the face described by the info. The permutation orders the
       dofs of elem2 as the ones of elem1.

       @param[in] info The face information of the current face.
       @param[in] face_index The interior/boundary face index.
    */
   void PermuteAndSetFaceDofsScatterIndices2(const Mesh::FaceInformation &info,
                                             const int face_index);

   /** @brief Set the scattering indices of elem2 for the shared face described
       by the info.

      Note: This method uses a trick to differentiate dof location inter/shared,
      the gid index is shifted by ndofs for shared dofs.

       @param[in] info The face information of the current face.
       @param[in] face_index The interior/boundary face index.
    */
   void SetSharedFaceDofsScatterIndices2(const Mesh::FaceInformation &info,
                                         const int face_index);

   /** @brief Permute and set the scattering indices of elem2 for the shared
       face described by the info. The permutation orders the dofs of elem2 as
       the ones of elem1.

       @param[in] info The face information of the current face.
       @param[in] face_index The interior/boundary face index.
    */
   void PermuteAndSetSharedFaceDofsScatterIndices2(
      const Mesh::FaceInformation &info,
      const int face_index);

   /** @brief Set the scattering indices of elem2 for the boundary face
       described by the info.

       @param[in] info The face information of the current face.
       @param[in] face_index The interior/boundary face index.
    */
   void SetBoundaryDofsScatterIndices2(const Mesh::FaceInformation &info,
                                       const int face_index);

   /** @brief Set the gathering indices of elem1 for the interior face described
       by the info.

       Note: This function modifies the offsets.

       @param[in] info The face information of the current face.
       @param[in] face_index The interior/boundary face index.
    */
   void SetFaceDofsGatherIndices1(const Mesh::FaceInformation &info,
                                  const int face_index);

   /** @brief Set the gathering indices of elem2 for the interior face described
       by the info.

       Note: This function modifies the offsets.

       @param[in] info The face information of the current face.
       @param[in] face_index The interior/boundary face index.
    */
   void SetFaceDofsGatherIndices2(const Mesh::FaceInformation &info,
                                  const int face_index);

   /** @brief Permute and set the gathering indices of elem2 for the interior
       face described by the info. The permutation orders the dofs of elem2 as
       the ones of elem1.

       Note: This function modifies the offsets.

       @param[in] info The face information of the current face.
       @param[in] face_index The interior/boundary face index.
    */
   void PermuteAndSetFaceDofsGatherIndices2(const Mesh::FaceInformation &info,
                                            const int face_index);
};

/** @brief Operator that extracts face degrees of freedom for L2 non-conforming
    spaces.

    In order to support face restrictions on non-conforming meshes, this
    operator interpolates master (coarse) face degrees of freedom onto the
    slave (fine) face. This allows face integrators to treat non-conforming
    faces just as regular conforming faces. */
class NCL2FaceRestriction : public L2FaceRestriction
{
protected:
   NCL2FaceRestriction(const FiniteElementSpace&,
                       const FaceType,
                       const L2FaceValues m = L2FaceValues::DoubleValued);

public:
   /** @brief Constructs an NCL2FaceRestriction, this is a specialization of a
       L2FaceRestriction for non-conforming meshes.

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
       @param[out] y The L-vector degrees of freedom. */
   void AddMultTranspose(const Vector &x, Vector &y) const override;

   /** @brief Fill the I array of SparseMatrix corresponding to the sparsity
       pattern given by this NCL2FaceRestriction.

       @param[in,out] mat The sparse matrix for which we want to initialize the
                          row offsets.
       @param[in] keep_nbr_block When set to true the SparseMatrix will
                                 include the rows (in addition to the columns)
                                 corresponding to face-neighbor dofs. The
                                 default behavior is to disregard those rows. */
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
                                 default behavior is to disregard those rows. */
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
                              elements. */
   void AddFaceMatricesToElementMatrices(const Vector &fea_data,
                                         Vector &ea_data) const override;

private:
   /** @brief Compute the scatter indices: L-vector to E-vector, the offsets
       for the gathering: E-vector to L-vector, and the interpolators from
       coarse to fine face for master non-comforming faces.

       @param[in] ordering Request a specific element ordering.
       @param[in] type     Request internal or boundary faces dofs.
   */
   void ComputeScatterIndicesAndOffsets(const ElementDofOrdering ordering,
                                        const FaceType type);

   /** @brief Compute the gather indices: E-vector to L-vector.

       Note: Requires the gather offsets to be computed.

       @param[in] ordering Request a specific element ordering.
       @param[in] type     Request internal or boundary faces dofs.
   */
   void ComputeGatherIndices(const ElementDofOrdering ordering,
                             const FaceType type);

protected:

   /** This class stores which side is the master non-conforming side and the
       index of the interpolator. */
   struct InterpConfig
   {
      int config;

      // Conforming face
      // InterpConfig() : config(-1) { } // TODO: try to make this work with is_trivial
      InterpConfig() = default;

      // Non-conforming face, if nc_index is given assumes side==1 (always true
      // except for ghost faces)
      InterpConfig(int config) : config(config) { }

      // Non-conforming face
      InterpConfig(int side, int nc_index)
         : config(side==1?nc_index: -1 - nc_index)
      { }

      MFEM_HOST_DEVICE
      InterpConfig(const InterpConfig&) = default;

      MFEM_HOST_DEVICE
      InterpConfig &operator=(const InterpConfig &rhs)
      {
         this->config = rhs.config;
         return *this;
      }

      MFEM_HOST_DEVICE
      int GetNonConformingMasterSide() const
      {
         // MFEM_ASSERT(config!=-1, "This face is conforming.");
         return config<-1? 0 : 1;
      }

      MFEM_HOST_DEVICE
      int GetInterpolatorIndex() const
      {
         // MFEM_ASSERT(config!=-1, "This face is conforming.");
         return config<-1? -1-config : config;
      }
   };

   Array<InterpConfig> interp_config; // interpolator index for each face
   int nc_size; // number of non-conforming interpolators
   Vector interpolators; // face_dofs x face_dofs x nc_size
   static const int conforming = -1; // helper value

   /** @brief Register the face with @a info and index @a face_index as a
       conforming face for the interpolation of the degrees of freedom.

       @param[in] info The face information of the current face.
       @param[in] face_index The interior/boundary face index.
    */
   void RegisterFaceConformingInterpolation(const Mesh::FaceInformation &info,
                                            int face_index);

   /** The interpolators are associated to a key of containing the address of
       PointMatrix and a local face identifiant. */
   using Key = std::pair<const DenseMatrix*,int>;
   /// The temporary map used to store the different interpolators.
   using Map = std::map<Key, std::pair<int,const DenseMatrix*>>;

   /** @brief Register the face with @a info and index @a face_index as a
       conforming face for the interpolation of the degrees of freedom.

       @param[in] info The face information of the current face.
       @param[in] face_index The interior/boundary face index.
       @param[in] ordering  Request a specific element ordering.
       @param[in,out] interp_map The map that stores the interpolators.
       @param[in,out] nc_cpt A counter that stores the number of different
                             interpolators. Used as index for each interpolator.
    */
   void RegisterFaceCoarseToFineInterpolation(const Mesh::FaceInformation &info,
                                              int face_index,
                                              ElementDofOrdering ordering,
                                              Map &interp_map,
                                              int &nc_cpt);

   /** @brief Returns the interpolation operator from a master (coarse) face to
       a slave (fine) face.

       @param[in] info The face information of the current face.
       @param[in] ptMat The PointMatrix describing the position and orientation
                        of the fine face in the coarse face. This PointMatrix is
                        usually obtained from the mesh through the method
                        GetNCFacesPtMat.
       @param[in] ordering  Request a specific element ordering.
       @return The dense matrix corresponding to the interpolation of the face
               degrees of freedom of the master (coarse) face to the slave
               (fine) face. */
   const DenseMatrix* ComputeCoarseToFineInterpolation(
      const Mesh::FaceInformation &info,
      const DenseMatrix* ptMat,
      ElementDofOrdering ordering);
};

/** @brief Return the face map that extracts the degrees of freedom for the
    requested local face of a quad or hex, returned in Lexicographic order.

    @param[in] dim The dimension of the space
    @param[in] face_id The local face identifiant
    @param[in] dof1d The 1D number of degrees of freedom for each dimension
    @param[out] face_map The map that maps each face dof to an element dof
*/
void GetFaceDofs(const int dim, const int face_id,
                 const int dof1d, Array<int> &face_map);

/** @brief Convert a dof face index from Native ordering to lexicographic
    ordering for quads and hexes.

    @param[in] dim The dimension of the element, 2 for quad, 3 for hex
    @param[in] face_id The local face identifiant
    @param[in] size1d The 1D number of degrees of freedom for each dimension
    @param[in] index The native index on the face
    @return The lexicographic index on the face
*/
int ToLexOrdering(const int dim, const int face_id, const int size1d,
                  const int index);

/** @brief Compute the dof face index of elem2 corresponding to the given dof
    face index.

    @param[in] dim The dimension of the element, 2 for quad, 3 for hex
    @param[in] face_id1 The local face identifiant of elem1
    @param[in] face_id2 The local face identifiant of elem2
    @param[in] orientation The orientation of elem2 relative to elem1 on the
                           face
    @param[in] size1d The 1D number of degrees of freedom for each dimension
    @param[in] index The dof index on elem1
    @return The dof index on elem2 facing the dof on elem1
*/
int PermuteFaceL2(const int dim, const int face_id1,
                  const int face_id2, const int orientation,
                  const int size1d, const int index);

}

#endif //MFEM_RESTRICTION
