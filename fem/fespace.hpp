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

#ifndef MFEM_FESPACE
#define MFEM_FESPACE

#include "../config/config.hpp"
#include "../linalg/sparsemat.hpp"
#include "../mesh/mesh.hpp"
#include "fe_coll.hpp"
#include "restriction.hpp"
#include <iostream>
#include <unordered_map>

namespace mfem
{

/** @brief The ordering method used when the number of unknowns per mesh node
    (vector dimension) is bigger than 1. */
class Ordering
{
public:
   /// %Ordering methods:
   enum Type
   {
      byNODES, /**< loop first over the nodes (inner loop) then over the vector
                    dimension (outer loop); symbolically it can be represented
                    as: XXX...,YYY...,ZZZ... */
      byVDIM   /**< loop first over the vector dimension (inner loop) then over
                    the nodes (outer loop); symbolically it can be represented
                    as: XYZ,XYZ,XYZ,... */
   };

   template <Type Ord>
   static inline int Map(int ndofs, int vdim, int dof, int vd);

   template <Type Ord>
   static void DofsToVDofs(int ndofs, int vdim, Array<int> &dofs);
};

template <> inline int
Ordering::Map<Ordering::byNODES>(int ndofs, int vdim, int dof, int vd)
{
   MFEM_ASSERT(dof < ndofs && -1-dof < ndofs && 0 <= vd && vd < vdim, "");
   return (dof >= 0) ? dof+ndofs*vd : dof-ndofs*vd;
}

template <> inline int
Ordering::Map<Ordering::byVDIM>(int ndofs, int vdim, int dof, int vd)
{
   MFEM_ASSERT(dof < ndofs && -1-dof < ndofs && 0 <= vd && vd < vdim, "");
   return (dof >= 0) ? vd+vdim*dof : -1-(vd+vdim*(-1-dof));
}


/// Constants describing the possible orderings of the DOFs in one element.
enum class ElementDofOrdering
{
   /// Native ordering as defined by the FiniteElement.
   /** This ordering can be used by tensor-product elements when the
       interpolation from the DOFs to quadrature points does not use the
       tensor-product structure. */
   NATIVE,
   /// Lexicographic ordering for tensor-product FiniteElements.
   /** This ordering can be used only with tensor-product elements. */
   LEXICOGRAPHIC
};

// Forward declarations
class NURBSExtension;
class BilinearFormIntegrator;
class QuadratureSpace;
class QuadratureInterpolator;
class FaceQuadratureInterpolator;


/** @brief Class FiniteElementSpace - responsible for providing FEM view of the
    mesh, mainly managing the set of degrees of freedom. */
class FiniteElementSpace
{
   friend class InterpolationGridTransfer;
   friend class PRefinementTransferOperator;
   friend void Mesh::Swap(Mesh &, bool);
   friend class LORBase;

protected:
   /// The mesh that FE space lives on (not owned).
   Mesh *mesh;

   /// Associated FE collection (not owned).
   const FiniteElementCollection *fec;

   /// %Vector dimension (number of unknowns per degree of freedom).
   int vdim;

   /** Type of ordering of the vector dofs when #vdim > 1.
       - Ordering::byNODES - first nodes, then vector dimension,
       - Ordering::byVDIM  - first vector dimension, then nodes */
   Ordering::Type ordering;

   /// Number of degrees of freedom. Number of unknowns is #ndofs * #vdim.
   int ndofs;

   /** Polynomial order for each element. If empty, all elements are assumed
       to be of the default order (fec->GetOrder()). */
   Array<char> elem_order;

   int nvdofs, nedofs, nfdofs, nbdofs;
   int uni_fdof; ///< # of single face DOFs if all faces uniform; -1 otherwise
   int *bdofs; ///< internal DOFs of elements if mixed/var-order; NULL otherwise

   /** Variable order spaces only: DOF assignments for edges and faces, see
       docs in MakeDofTable. For constant order spaces the tables are empty. */
   Table var_edge_dofs;
   Table var_face_dofs; ///< NOTE: also used for spaces with mixed faces

   /** Additional data for the var_*_dofs tables: individual variant orders
       (these are basically alternate J arrays for var_edge/face_dofs). */
   Array<char> var_edge_orders, var_face_orders;

   // precalculated DOFs for each element, boundary element, and face
   mutable Table *elem_dof; // owned (except in NURBS FE space)
   mutable Table *bdr_elem_dof; // owned (except in NURBS FE space)
   mutable Table *face_dof; // owned; in var-order space contains variant 0 DOFs

   Array<int> dof_elem_array, dof_ldof_array;

   NURBSExtension *NURBSext;
   int own_ext;
   mutable Array<int> face_to_be; // NURBS FE space only

   /** Matrix representing the prolongation from the global conforming dofs to
       a set of intermediate partially conforming dofs, e.g. the dofs associated
       with a "cut" space on a non-conforming mesh. */
   mutable SparseMatrix *cP; // owned
   /// Conforming restriction matrix such that cR.cP=I.
   mutable SparseMatrix *cR; // owned
   /// A version of the conforming restriction matrix for variable-order spaces.
   mutable SparseMatrix *cR_hp; // owned
   mutable bool cP_is_set;

   /// Transformation to apply to GridFunctions after space Update().
   OperatorHandle Th;

   /// The element restriction operators, see GetElementRestriction().
   mutable OperatorHandle L2E_nat, L2E_lex;
   /// The face restriction operators, see GetFaceRestriction().
   using key_face = std::tuple<bool, ElementDofOrdering, FaceType, L2FaceValues>;
   struct key_hash
   {
      std::size_t operator()(const key_face& k) const
      {
         return std::get<0>(k)
                + 2 * (int)std::get<1>(k)
                + 4 * (int)std::get<2>(k)
                + 8 * (int)std::get<3>(k);
      }
   };
   using map_L2F = std::unordered_map<const key_face,Operator*,key_hash>;
   mutable map_L2F L2F;

   mutable Array<QuadratureInterpolator*> E2Q_array;
   mutable Array<FaceQuadratureInterpolator*> E2IFQ_array;
   mutable Array<FaceQuadratureInterpolator*> E2BFQ_array;

   /** Update counter, incremented every time the space is constructed/updated.
       Used by GridFunctions to check if they are up to date with the space. */
   long sequence;

   /** Mesh sequence number last seen when constructing the space. The space
       needs updating if Mesh::GetSequence() is larger than this. */
   long mesh_sequence;

   /// True if at least one element order changed (variable-order space only).
   bool orders_changed;

   bool relaxed_hp; // see SetRelaxedHpConformity()

   void UpdateNURBS();

   void Construct();
   void Destroy();

   void BuildElementToDofTable() const;
   void BuildBdrElementToDofTable() const;
   void BuildFaceToDofTable() const;

   /** @brief  Generates partial face_dof table for a NURBS space.

       The table is only defined for exterior faces that coincide with a
       boundary. */
   void BuildNURBSFaceToDofTable() const;

   /// Bit-mask representing a set of orders needed by an edge/face.
   typedef std::uint64_t VarOrderBits;
   static constexpr int MaxVarOrder = 8*sizeof(VarOrderBits) - 1;

   /// Return the minimum order (least significant bit set) in the bit mask.
   static int MinOrder(VarOrderBits bits);

   /// Return element order: internal version of GetElementOrder without checks.
   int GetElementOrderImpl(int i) const;

   /** In a variable order space, calculate a bitmask of polynomial orders that
       need to be represented on each edge and face. */
   void CalcEdgeFaceVarOrders(Array<VarOrderBits> &edge_orders,
                              Array<VarOrderBits> &face_orders) const;

   /** Build the table var_edge_dofs (or var_face_dofs) in a variable order
       space; return total edge/face DOFs. */
   int MakeDofTable(int ent_dim, const Array<int> &entity_orders,
                    Table &entity_dofs, Array<char> *var_ent_order);

   /// Search row of a DOF table for a DOF set of size 'ndof', return first DOF.
   int FindDofs(const Table &var_dof_table, int row, int ndof) const;

   /** In a variable order space, return edge DOFs associated with a polynomial
       order that has 'ndof' degrees of freedom. */
   int FindEdgeDof(int edge, int ndof) const
   { return FindDofs(var_edge_dofs, edge, ndof); }

   /// Similar to FindEdgeDof, but used for mixed meshes too.
   int FindFaceDof(int face, int ndof) const
   { return FindDofs(var_face_dofs, face, ndof); }

   int FirstFaceDof(int face, int variant = 0) const
   { return uni_fdof >= 0 ? face*uni_fdof : var_face_dofs.GetRow(face)[variant];}

   /// Return number of possible DOF variants for edge/face (var. order spaces).
   int GetNVariants(int entity, int index) const;

   /// Helper to encode a sign flip into a DOF index (for Hcurl/Hdiv shapes).
   static inline int EncodeDof(int entity_base, int idx)
   { return (idx >= 0) ? (entity_base + idx) : (-1-(entity_base + (-1-idx))); }

   /// Helpers to remove encoded sign from a DOF
   static inline int DecodeDof(int dof)
   { return (dof >= 0) ? dof : (-1 - dof); }

   static inline int DecodeDof(int dof, double& sign)
   { return (dof >= 0) ? (sign = 1, dof) : (sign = -1, (-1 - dof)); }

   /// Helper to get vertex, edge or face DOFs (entity=0,1,2 resp.).
   int GetEntityDofs(int entity, int index, Array<int> &dofs,
                     Geometry::Type master_geom = Geometry::INVALID,
                     int variant = 0) const;

   // Get degenerate face DOFs: see explanation in method implementation.
   int GetDegenerateFaceDofs(int index, Array<int> &dofs,
                             Geometry::Type master_geom, int variant) const;

   int GetNumBorderDofs(Geometry::Type geom, int order) const;

   /// Calculate the cP and cR matrices for a nonconforming mesh.
   void BuildConformingInterpolation() const;

   static void AddDependencies(SparseMatrix& deps, Array<int>& master_dofs,
                               Array<int>& slave_dofs, DenseMatrix& I,
                               int skipfirst = 0);

   static bool DofFinalizable(int dof, const Array<bool>& finalized,
                              const SparseMatrix& deps);

   void AddEdgeFaceDependencies(SparseMatrix &deps, Array<int>& master_dofs,
                                const FiniteElement *master_fe,
                                Array<int> &slave_dofs, int slave_face,
                                const DenseMatrix *pm) const;

   /// Replicate 'mat' in the vector dimension, according to vdim ordering mode.
   void MakeVDimMatrix(SparseMatrix &mat) const;

   /// GridFunction interpolation operator applicable after mesh refinement.
   class RefinementOperator : public Operator
   {
      const FiniteElementSpace* fespace;
      DenseTensor localP[Geometry::NumGeom];
      Table* old_elem_dof; // Owned.

   public:
      /** Construct the operator based on the elem_dof table of the original
          (coarse) space. The class takes ownership of the table. */
      RefinementOperator(const FiniteElementSpace* fespace,
                         Table *old_elem_dof/*takes ownership*/, int old_ndofs);
      RefinementOperator(const FiniteElementSpace *fespace,
                         const FiniteElementSpace *coarse_fes);
      virtual void Mult(const Vector &x, Vector &y) const;
      virtual void MultTranspose(const Vector &x, Vector &y) const;
      virtual ~RefinementOperator();
   };

   /// Derefinement operator, used by the friend class InterpolationGridTransfer.
   class DerefinementOperator : public Operator
   {
      const FiniteElementSpace *fine_fes; // Not owned.
      DenseTensor localR[Geometry::NumGeom];
      Table *coarse_elem_dof; // Owned.
      Table coarse_to_fine;
      Array<int> coarse_to_ref_type;
      Array<Geometry::Type> ref_type_to_geom;
      Array<int> ref_type_to_fine_elem_offset;

   public:
      DerefinementOperator(const FiniteElementSpace *f_fes,
                           const FiniteElementSpace *c_fes,
                           BilinearFormIntegrator *mass_integ);
      virtual void Mult(const Vector &x, Vector &y) const;
      virtual ~DerefinementOperator();
   };

   /** This method makes the same assumptions as the method:
       void GetLocalRefinementMatrices(
           const FiniteElementSpace &coarse_fes, Geometry::Type geom,
           DenseTensor &localP) const
       which is defined below. It also assumes that the coarse fes and this have
       the same vector dimension, vdim. */
   SparseMatrix *RefinementMatrix_main(const int coarse_ndofs,
                                       const Table &coarse_elem_dof,
                                       const DenseTensor localP[]) const;

   void GetLocalRefinementMatrices(Geometry::Type geom,
                                   DenseTensor &localP) const;
   void GetLocalDerefinementMatrices(Geometry::Type geom,
                                     DenseTensor &localR) const;

   /** Calculate explicit GridFunction interpolation matrix (after mesh
       refinement). NOTE: consider using the RefinementOperator class instead
       of the fully assembled matrix, which can take a lot of memory. */
   SparseMatrix* RefinementMatrix(int old_ndofs, const Table* old_elem_dof);

   /// Calculate GridFunction restriction matrix after mesh derefinement.
   SparseMatrix* DerefinementMatrix(int old_ndofs, const Table* old_elem_dof);

   /** @brief Return in @a localP the local refinement matrices that map
       between fespaces after mesh refinement. */
   /** This method assumes that this->mesh is a refinement of coarse_fes->mesh
       and that the CoarseFineTransformations of this->mesh are set accordingly.
       Another assumption is that the FEs of this use the same MapType as the FEs
       of coarse_fes. Finally, it assumes that the spaces this and coarse_fes are
       NOT variable-order spaces. */
   void GetLocalRefinementMatrices(const FiniteElementSpace &coarse_fes,
                                   Geometry::Type geom,
                                   DenseTensor &localP) const;

   /// Help function for constructors + Load().
   void Constructor(Mesh *mesh, NURBSExtension *ext,
                    const FiniteElementCollection *fec,
                    int vdim = 1, int ordering = Ordering::byNODES);

   /// Updates the internal mesh pointer. @warning @a new_mesh must be
   /// <b>topologically identical</b> to the existing mesh. Used if the address
   /// of the Mesh object has changed, e.g. in @a Mesh::Swap.
   virtual void UpdateMeshPointer(Mesh *new_mesh);

   /// Resize the elem_order array on mesh change.
   void UpdateElementOrders();

   /// @brief Copies the prolongation and restriction matrices from @a fes.
   ///
   /// Used for low order preconditioning on non-conforming meshes. If the DOFs
   /// require a permutation, it will be supplied by non-NULL @a perm. NULL @a
   /// perm indicates that no permutation is required.
   virtual void CopyProlongationAndRestriction(const FiniteElementSpace &fes,
                                               const Array<int> *perm);

public:
   /** @brief Default constructor: the object is invalid until initialized using
       the method Load(). */
   FiniteElementSpace();

   /** @brief Copy constructor: deep copy all data from @a orig except the Mesh,
       the FiniteElementCollection, ans some derived data. */
   /** If the @a mesh or @a fec pointers are NULL (default), then the new
       FiniteElementSpace will reuse the respective pointers from @a orig. If
       any of these pointers is not NULL, the given pointer will be used instead
       of the one used by @a orig.

       @note The objects pointed to by the @a mesh and @a fec parameters must be
       either the same objects as the ones used by @a orig, or copies of them.
       Otherwise, the behavior is undefined.

       @note Derived data objects, such as the conforming prolongation and
       restriction matrices, and the update operator, will not be copied, even
       if they are created in the @a orig object. */
   FiniteElementSpace(const FiniteElementSpace &orig, Mesh *mesh = NULL,
                      const FiniteElementCollection *fec = NULL);

   FiniteElementSpace(Mesh *mesh,
                      const FiniteElementCollection *fec,
                      int vdim = 1, int ordering = Ordering::byNODES)
   { Constructor(mesh, NULL, fec, vdim, ordering); }

   /// Construct a NURBS FE space based on the given NURBSExtension, @a ext.
   /** @note If the pointer @a ext is NULL, this constructor is equivalent to
       the standard constructor with the same arguments minus the
       NURBSExtension, @a ext. */
   FiniteElementSpace(Mesh *mesh, NURBSExtension *ext,
                      const FiniteElementCollection *fec,
                      int vdim = 1, int ordering = Ordering::byNODES)
   { Constructor(mesh, ext, fec, vdim, ordering); }

   /// Returns the mesh
   inline Mesh *GetMesh() const { return mesh; }

   const NURBSExtension *GetNURBSext() const { return NURBSext; }
   NURBSExtension *GetNURBSext() { return NURBSext; }
   NURBSExtension *StealNURBSext();

   bool Conforming() const { return mesh->Conforming() && cP == NULL; }
   bool Nonconforming() const { return mesh->Nonconforming() || cP != NULL; }

   /// Sets the order of the i'th finite element.
   /** By default, all elements are assumed to be of fec->GetOrder(). Once
       SetElementOrder is called, the space becomes a variable order space. */
   void SetElementOrder(int i, int p);

   /// Returns the order of the i'th finite element.
   int GetElementOrder(int i) const;

   /// Return the maximum polynomial order.
   int GetMaxElementOrder() const
   { return IsVariableOrder() ? elem_order.Max() : fec->GetOrder(); }

   /// Returns true if the space contains elements of varying polynomial orders.
   bool IsVariableOrder() const { return elem_order.Size(); }

   /// The returned SparseMatrix is owned by the FiniteElementSpace.
   const SparseMatrix *GetConformingProlongation() const;

   /// The returned SparseMatrix is owned by the FiniteElementSpace.
   const SparseMatrix *GetConformingRestriction() const;

   /** Return a version of the conforming restriction matrix for variable-order
       spaces with complex hp interfaces, where some true DOFs are not owned by
       any elements and need to be interpolated from higher order edge/face
       variants (see also @a SetRelaxedHpConformity()). */
   /// The returned SparseMatrix is owned by the FiniteElementSpace.
   const SparseMatrix *GetHpConformingRestriction() const;

   /// The returned Operator is owned by the FiniteElementSpace.
   virtual const Operator *GetProlongationMatrix() const
   { return GetConformingProlongation(); }

   /// Return an operator that performs the transpose of GetRestrictionOperator
   /** The returned operator is owned by the FiniteElementSpace. In serial this
       is the same as GetProlongationMatrix() */
   virtual const Operator *GetRestrictionTransposeOperator() const
   { return GetConformingProlongation(); }

   /// An abstract operator that performs the same action as GetRestrictionMatrix
   /** In some cases this is an optimized matrix-free implementation. The
       returned operator is owned by the FiniteElementSpace. */
   virtual const Operator *GetRestrictionOperator() const
   { return GetConformingRestriction(); }

   /// The returned SparseMatrix is owned by the FiniteElementSpace.
   virtual const SparseMatrix *GetRestrictionMatrix() const
   { return GetConformingRestriction(); }

   /// The returned SparseMatrix is owned by the FiniteElementSpace.
   virtual const SparseMatrix *GetHpRestrictionMatrix() const
   { return GetHpConformingRestriction(); }

   /// Return an Operator that converts L-vectors to E-vectors.
   /** An L-vector is a vector of size GetVSize() which is the same size as a
       GridFunction. An E-vector represents the element-wise discontinuous
       version of the FE space.

       The layout of the E-vector is: ND x VDIM x NE, where ND is the number of
       degrees of freedom, VDIM is the vector dimension of the FE space, and NE
       is the number of the mesh elements.

       The parameter @a e_ordering describes how the local DOFs in each element
       should be ordered, see ElementDofOrdering.

       For discontinuous spaces, the element restriction corresponds to a
       permutation of the degrees of freedom, implemented by the
       L2ElementRestriction class.

       The returned Operator is owned by the FiniteElementSpace. */
   const Operator *GetElementRestriction(ElementDofOrdering e_ordering) const;

   /// Return an Operator that converts L-vectors to E-vectors on each face.
   virtual const Operator *GetFaceRestriction(
      ElementDofOrdering e_ordering, FaceType,
      L2FaceValues mul = L2FaceValues::DoubleValued) const;

   /** @brief Return a QuadratureInterpolator that interpolates E-vectors to
       quadrature point values and/or derivatives (Q-vectors). */
   /** An E-vector represents the element-wise discontinuous version of the FE
       space and can be obtained, for example, from a GridFunction using the
       Operator returned by GetElementRestriction().

       All elements will use the same IntegrationRule, @a ir as the target
       quadrature points. */
   const QuadratureInterpolator *GetQuadratureInterpolator(
      const IntegrationRule &ir) const;

   /** @brief Return a QuadratureInterpolator that interpolates E-vectors to
       quadrature point values and/or derivatives (Q-vectors). */
   /** An E-vector represents the element-wise discontinuous version of the FE
       space and can be obtained, for example, from a GridFunction using the
       Operator returned by GetElementRestriction().

       The target quadrature points in the elements are described by the given
       QuadratureSpace, @a qs. */
   const QuadratureInterpolator *GetQuadratureInterpolator(
      const QuadratureSpace &qs) const;

   /** @brief Return a FaceQuadratureInterpolator that interpolates E-vectors to
       quadrature point values and/or derivatives (Q-vectors). */
   const FaceQuadratureInterpolator *GetFaceQuadratureInterpolator(
      const IntegrationRule &ir, FaceType type) const;

   /// Returns the polynomial degree of the i'th finite element.
   /** NOTE: it is recommended to use GetElementOrder in new code. */
   int GetOrder(int i) const { return GetElementOrder(i); }

   /** Return the order of an edge. In a variable order space, return the order
       of a specific variant, or -1 if there are no more variants. */
   int GetEdgeOrder(int edge, int variant = 0) const;

   /// Returns the polynomial degree of the i'th face finite element
   int GetFaceOrder(int face, int variant = 0) const;

   /// Returns vector dimension.
   inline int GetVDim() const { return vdim; }

   /// Returns number of degrees of freedom.
   inline int GetNDofs() const { return ndofs; }

   /// Return the number of vector dofs, i.e. GetNDofs() x GetVDim().
   inline int GetVSize() const { return vdim * ndofs; }

   /// Return the number of vector true (conforming) dofs.
   virtual int GetTrueVSize() const { return GetConformingVSize(); }

   /// Returns the number of conforming ("true") degrees of freedom
   /// (if the space is on a nonconforming mesh with hanging nodes).
   int GetNConformingDofs() const;

   int GetConformingVSize() const { return vdim * GetNConformingDofs(); }

   /// Return the ordering method.
   inline Ordering::Type GetOrdering() const { return ordering; }

   const FiniteElementCollection *FEColl() const { return fec; }

   /// Number of all scalar vertex dofs
   int GetNVDofs() const { return nvdofs; }
   /// Number of all scalar edge-interior dofs
   int GetNEDofs() const { return nedofs; }
   /// Number of all scalar face-interior dofs
   int GetNFDofs() const { return nfdofs; }

   /// Returns number of vertices in the mesh.
   inline int GetNV() const { return mesh->GetNV(); }

   /// Returns number of elements in the mesh.
   inline int GetNE() const { return mesh->GetNE(); }

   /// Returns number of faces (i.e. co-dimension 1 entities) in the mesh.
   /** The co-dimension 1 entities are those that have dimension 1 less than the
       mesh dimension, e.g. for a 2D mesh, the faces are the 1D entities, i.e.
       the edges. */
   inline int GetNF() const { return mesh->GetNumFaces(); }

   /// Returns number of boundary elements in the mesh.
   inline int GetNBE() const { return mesh->GetNBE(); }

   /// Returns the number of faces according to the requested type.
   /** If type==Boundary returns only the "true" number of boundary faces
       contrary to GetNBE() that returns "fake" boundary faces associated to
       visualization for GLVis.
       Similarly, if type==Interior, the "fake" boundary faces associated to
       visualization are counted as interior faces. */
   inline int GetNFbyType(FaceType type) const
   { return mesh->GetNFbyType(type); }

   /// Returns the type of element i.
   inline int GetElementType(int i) const
   { return mesh->GetElementType(i); }

   /// Returns the vertices of element i.
   inline void GetElementVertices(int i, Array<int> &vertices) const
   { mesh->GetElementVertices(i, vertices); }

   /// Returns the type of boundary element i.
   inline int GetBdrElementType(int i) const
   { return mesh->GetBdrElementType(i); }

   /// Returns ElementTransformation for the @a i-th element.
   ElementTransformation *GetElementTransformation(int i) const
   { return mesh->GetElementTransformation(i); }

   /** @brief Returns the transformation defining the @a i-th element in the
       user-defined variable @a ElTr. */
   void GetElementTransformation(int i, IsoparametricTransformation *ElTr)
   { mesh->GetElementTransformation(i, ElTr); }

   /// Returns ElementTransformation for the @a i-th boundary element.
   ElementTransformation *GetBdrElementTransformation(int i) const
   { return mesh->GetBdrElementTransformation(i); }

   int GetAttribute(int i) const { return mesh->GetAttribute(i); }

   int GetBdrAttribute(int i) const { return mesh->GetBdrAttribute(i); }

   /// Returns indices of degrees of freedom of element 'elem'.
   virtual void GetElementDofs(int elem, Array<int> &dofs) const;

   /// Returns indices of degrees of freedom for boundary element 'bel'.
   virtual void GetBdrElementDofs(int bel, Array<int> &dofs) const;

   /** @brief Returns the indices of the degrees of freedom for the specified
       face, including the DOFs for the edges and the vertices of the face. */
   /** In variable order spaces, multiple variants of DOFs can be returned.
       See @a GetEdgeDofs for more details.
       @return Order of the selected variant, or -1 if there are no more
       variants.*/
   virtual int GetFaceDofs(int face, Array<int> &dofs, int variant = 0) const;

   /** @brief Returns the indices of the degrees of freedom for the specified
       edge, including the DOFs for the vertices of the edge. */
   /** In variable order spaces, multiple sets of DOFs may exist on an edge,
       corresponding to the different polynomial orders of incident elements.
       The 'variant' parameter is the zero-based index of the desired DOF set.
       The variants are ordered from lowest polynomial degree to the highest.
       @return Order of the selected variant, or -1 if there are no more
       variants. */
   int GetEdgeDofs(int edge, Array<int> &dofs, int variant = 0) const;

   void GetVertexDofs(int i, Array<int> &dofs) const;

   void GetElementInteriorDofs(int i, Array<int> &dofs) const;

   void GetFaceInteriorDofs(int i, Array<int> &dofs) const;

   int GetNumElementInteriorDofs(int i) const;

   void GetEdgeInteriorDofs(int i, Array<int> &dofs) const;

   void DofsToVDofs(Array<int> &dofs, int ndofs = -1) const;

   void DofsToVDofs(int vd, Array<int> &dofs, int ndofs = -1) const;

   int DofToVDof(int dof, int vd, int ndofs = -1) const;

   int VDofToDof(int vdof) const
   { return (ordering == Ordering::byNODES) ? (vdof%ndofs) : (vdof/vdim); }

   static void AdjustVDofs(Array<int> &vdofs);

   /// Returns indexes of degrees of freedom in array dofs for i'th element.
   void GetElementVDofs(int i, Array<int> &vdofs) const;

   /// Returns indexes of degrees of freedom for i'th boundary element.
   void GetBdrElementVDofs(int i, Array<int> &vdofs) const;

   /// Returns indexes of degrees of freedom for i'th face element (2D and 3D).
   void GetFaceVDofs(int i, Array<int> &vdofs) const;

   /// Returns indexes of degrees of freedom for i'th edge.
   void GetEdgeVDofs(int i, Array<int> &vdofs) const;

   void GetVertexVDofs(int i, Array<int> &vdofs) const;

   void GetElementInteriorVDofs(int i, Array<int> &vdofs) const;

   void GetEdgeInteriorVDofs(int i, Array<int> &vdofs) const;

   /// (@deprecated) Use the Update() method if the space or mesh changed.
   MFEM_DEPRECATED void RebuildElementToDofTable();

   /** @brief Reorder the scalar DOFs based on the element ordering.

       The new ordering is constructed as follows: 1) loop over all elements as
       ordered in the Mesh; 2) for each element, assign new indices to all of
       its current DOFs that are still unassigned; the new indices we assign are
       simply the sequence `0,1,2,...`; if there are any signed DOFs their sign
       is preserved. */
   void ReorderElementToDofTable();

   /** @brief Return a reference to the internal Table that stores the lists of
       scalar dofs, for each mesh element, as returned by GetElementDofs(). */
   const Table &GetElementToDofTable() const { return *elem_dof; }

   /** @brief Return a reference to the internal Table that stores the lists of
       scalar dofs, for each boundary mesh element, as returned by
       GetBdrElementDofs(). */
   const Table &GetBdrElementToDofTable() const
   { if (!bdr_elem_dof) { BuildBdrElementToDofTable(); } return *bdr_elem_dof; }

   /** @brief Return a reference to the internal Table that stores the lists of
       scalar dofs, for each face in the mesh, as returned by GetFaceDofs(). In
       this context, "face" refers to a (dim-1)-dimensional mesh entity. */
   /** @note In the case of a NURBS space, the rows corresponding to interior
       faces will be empty. */
   const Table &GetFaceToDofTable() const
   { if (!face_dof) { BuildFaceToDofTable(); } return *face_dof; }

   /** @brief Initialize internal data that enables the use of the methods
       GetElementForDof() and GetLocalDofForDof(). */
   void BuildDofToArrays();

   /// Return the index of the first element that contains dof @a i.
   /** This method can be called only after setup is performed using the method
       BuildDofToArrays(). */
   int GetElementForDof(int i) const { return dof_elem_array[i]; }
   /// Return the local dof index in the first element that contains dof @a i.
   /** This method can be called only after setup is performed using the method
       BuildDofToArrays(). */
   int GetLocalDofForDof(int i) const { return dof_ldof_array[i]; }

   /** @brief Returns pointer to the FiniteElement in the FiniteElementCollection
        associated with i'th element in the mesh object. */
   virtual const FiniteElement *GetFE(int i) const;

   /** @brief Returns pointer to the FiniteElement in the FiniteElementCollection
        associated with i'th boundary face in the mesh object. */
   const FiniteElement *GetBE(int i) const;

   /** @brief Returns pointer to the FiniteElement in the FiniteElementCollection
        associated with i'th face in the mesh object.  Faces in this case refer
        to the MESHDIM-1 primitive so in 2D they are segments and in 1D they are
        points.*/
   const FiniteElement *GetFaceElement(int i) const;

   /** @brief Returns pointer to the FiniteElement in the FiniteElementCollection
        associated with i'th edge in the mesh object. */
   const FiniteElement *GetEdgeElement(int i, int variant = 0) const;

   /// Return the trace element from element 'i' to the given 'geom_type'
   const FiniteElement *GetTraceElement(int i, Geometry::Type geom_type) const;

   /** @brief Mark degrees of freedom associated with boundary elements with
       the specified boundary attributes (marked in 'bdr_attr_is_ess').
       For spaces with 'vdim' > 1, the 'component' parameter can be used
       to restricts the marked vDOFs to the specified component. */
   virtual void GetEssentialVDofs(const Array<int> &bdr_attr_is_ess,
                                  Array<int> &ess_vdofs,
                                  int component = -1) const;

   /** @brief Get a list of essential true dofs, ess_tdof_list, corresponding to the
       boundary attributes marked in the array bdr_attr_is_ess.
       For spaces with 'vdim' > 1, the 'component' parameter can be used
       to restricts the marked tDOFs to the specified component. */
   virtual void GetEssentialTrueDofs(const Array<int> &bdr_attr_is_ess,
                                     Array<int> &ess_tdof_list,
                                     int component = -1);

   /** @brief Get a list of all boundary true dofs, @a boundary_dofs. For spaces
       with 'vdim' > 1, the 'component' parameter can be used to restricts the
       marked tDOFs to the specified component. Equivalent to
       FiniteElementSpace::GetEssentialTrueDofs with all boundary attributes
       marked as essential. */
   void GetBoundaryTrueDofs(Array<int> &boundary_dofs, int component = -1);

   /// Convert a Boolean marker array to a list containing all marked indices.
   static void MarkerToList(const Array<int> &marker, Array<int> &list);

   /** @brief Convert an array of indices (list) to a Boolean marker array where all
       indices in the list are marked with the given value and the rest are set
       to zero. */
   static void ListToMarker(const Array<int> &list, int marker_size,
                            Array<int> &marker, int mark_val = -1);

   /** @brief For a partially conforming FE space, convert a marker array (nonzero
       entries are true) on the partially conforming dofs to a marker array on
       the conforming dofs. A conforming dofs is marked iff at least one of its
       dependent dofs is marked. */
   void ConvertToConformingVDofs(const Array<int> &dofs, Array<int> &cdofs);

   /** @brief For a partially conforming FE space, convert a marker array (nonzero
       entries are true) on the conforming dofs to a marker array on the
       (partially conforming) dofs. A dof is marked iff it depends on a marked
       conforming dofs, where dependency is defined by the ConformingRestriction
       matrix; in other words, a dof is marked iff it corresponds to a marked
       conforming dof. */
   void ConvertFromConformingVDofs(const Array<int> &cdofs, Array<int> &dofs);

   /** @brief Generate the global restriction matrix from a discontinuous
       FE space to the continuous FE space of the same polynomial degree. */
   SparseMatrix *D2C_GlobalRestrictionMatrix(FiniteElementSpace *cfes);

   /** @brief Generate the global restriction matrix from a discontinuous
       FE space to the piecewise constant FE space. */
   SparseMatrix *D2Const_GlobalRestrictionMatrix(FiniteElementSpace *cfes);

   /** @brief Construct the restriction matrix from the FE space given by
       (*this) to the lower degree FE space given by (*lfes) which
       is defined on the same mesh. */
   SparseMatrix *H2L_GlobalRestrictionMatrix(FiniteElementSpace *lfes);

   /** @brief Construct and return an Operator that can be used to transfer
       GridFunction data from @a coarse_fes, defined on a coarse mesh, to @a
       this FE space, defined on a refined mesh. */
   /** It is assumed that the mesh of this FE space is a refinement of the mesh
       of @a coarse_fes and the CoarseFineTransformations returned by the method
       Mesh::GetRefinementTransforms() of the refined mesh are set accordingly.
       The Operator::Type of @a T can be set to request an Operator of the set
       type. Currently, only Operator::MFEM_SPARSEMAT and Operator::ANY_TYPE
       (matrix-free) are supported. When Operator::ANY_TYPE is requested, the
       choice of the particular Operator sub-class is left to the method.  This
       method also works in parallel because the transfer operator is local to
       the MPI task when the input is a synchronized ParGridFunction. */
   void GetTransferOperator(const FiniteElementSpace &coarse_fes,
                            OperatorHandle &T) const;

   /** @brief Construct and return an Operator that can be used to transfer
       true-dof data from @a coarse_fes, defined on a coarse mesh, to @a this FE
       space, defined on a refined mesh.

       This method calls GetTransferOperator() and multiplies the result by the
       prolongation operator of @a coarse_fes on the right, and by the
       restriction operator of this FE space on the left.

       The Operator::Type of @a T can be set to request an Operator of the set
       type. In serial, the supported types are: Operator::MFEM_SPARSEMAT and
       Operator::ANY_TYPE (matrix-free). In parallel, the supported types are:
       Operator::Hypre_ParCSR and Operator::ANY_TYPE. Any other type is treated
       as Operator::ANY_TYPE: the operator representation choice is made by this
       method. */
   virtual void GetTrueTransferOperator(const FiniteElementSpace &coarse_fes,
                                        OperatorHandle &T) const;

   /** @brief Reflect changes in the mesh: update number of DOFs, etc. Also, calculate
       GridFunction transformation operator (unless want_transform is false).
       Safe to call multiple times, does nothing if space already up to date. */
   virtual void Update(bool want_transform = true);

   /// Get the GridFunction update operator.
   const Operator* GetUpdateOperator() { Update(); return Th.Ptr(); }

   /// Return the update operator in the given OperatorHandle, @a T.
   void GetUpdateOperator(OperatorHandle &T) { T = Th; }

   /** @brief Set the ownership of the update operator: if set to false, the
       Operator returned by GetUpdateOperator() must be deleted outside the
       FiniteElementSpace. */
   /** The update operator ownership is automatically reset to true when a new
       update operator is created by the Update() method. */
   void SetUpdateOperatorOwner(bool own) { Th.SetOperatorOwner(own); }

   /// Specify the Operator::Type to be used by the update operators.
   /** The default type is Operator::ANY_TYPE which leaves the choice to this
       class. The other currently supported option is Operator::MFEM_SPARSEMAT
       which is only guaranteed to be honored for a refinement update operator.
       Any other type will be treated as Operator::ANY_TYPE.
       @note This operation destroys the current update operator (if owned). */
   void SetUpdateOperatorType(Operator::Type tid) { Th.SetType(tid); }

   /// Free the GridFunction update operator (if any), to save memory.
   virtual void UpdatesFinished() { Th.Clear(); }

   /** Return update counter, similar to Mesh::GetSequence(). Used by
       GridFunction to check if it is up to date with the space. */
   long GetSequence() const { return sequence; }

   /// Return whether or not the space is discontinuous (L2)
   bool IsDGSpace() const
   {
      return dynamic_cast<const L2_FECollection*>(fec) != NULL;
   }

   /** In variable order spaces on nonconforming (NC) meshes, this function
       controls whether strict conformity is enforced in cases where coarse
       edges/faces have higher polynomial order than their fine NC neighbors.
       In the default (strict) case, the coarse side polynomial order is
       reduced to that of the lowest order fine edge/face, so all fine
       neighbors can interpolate the coarse side exactly. If relaxed == true,
       some discontinuities in the solution in such cases are allowed and the
       coarse side is not restricted. For an example, see
       https://github.com/mfem/mfem/pull/1423#issuecomment-621340392 */
   void SetRelaxedHpConformity(bool relaxed = true)
   {
      relaxed_hp = relaxed;
      orders_changed = true; // force update
      Update(false);
   }

   /// Save finite element space to output stream @a out.
   void Save(std::ostream &out) const;

   /** @brief Read a FiniteElementSpace from a stream. The returned
       FiniteElementCollection is owned by the caller. */
   FiniteElementCollection *Load(Mesh *m, std::istream &input);

   virtual ~FiniteElementSpace();
};


/// Class representing the storage layout of a QuadratureFunction.
/** Multiple QuadratureFunction%s can share the same QuadratureSpace. */
class QuadratureSpace
{
protected:
   friend class QuadratureFunction; // Uses the element_offsets.

   Mesh *mesh;
   int order;
   int size;

   const IntegrationRule *int_rule[Geometry::NumGeom];
   int *element_offsets; // scalar offsets; size = number of elements + 1

   // protected functions

   // Assuming mesh and order are set, construct the members: int_rule,
   // element_offsets, and size.
   void Construct();

public:
   /// Create a QuadratureSpace based on the global rules from #IntRules.
   QuadratureSpace(Mesh *mesh_, int order_)
      : mesh(mesh_), order(order_) { Construct(); }

   /// Read a QuadratureSpace from the stream @a in.
   QuadratureSpace(Mesh *mesh_, std::istream &in);

   virtual ~QuadratureSpace() { delete [] element_offsets; }

   /// Return the total number of quadrature points.
   int GetSize() const { return size; }

   /// Return the order of the quadrature rule(s) used by all elements.
   int GetOrder() const { return order; }

   /// Returns the mesh
   inline Mesh *GetMesh() const { return mesh; }

   /// Returns number of elements in the mesh.
   inline int GetNE() const { return mesh->GetNE(); }

   /// Get the IntegrationRule associated with mesh element @a idx.
   const IntegrationRule &GetElementIntRule(int idx) const
   { return *int_rule[mesh->GetElementBaseGeometry(idx)]; }

   /// Write the QuadratureSpace to the stream @a out.
   void Save(std::ostream &out) const;
};

inline bool UsesTensorBasis(const FiniteElementSpace& fes)
{
   return dynamic_cast<const mfem::TensorBasisElement *>(fes.GetFE(0))!=nullptr;
}

}

#endif
