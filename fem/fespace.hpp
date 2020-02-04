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

#ifndef MFEM_FESPACE
#define MFEM_FESPACE

#include "../config/config.hpp"
#include "../linalg/sparsemat.hpp"
#include "../mesh/mesh.hpp"
#include "fe_coll.hpp"
#include <iostream>

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


/** @brief Class FiniteElementSpace - responsible for providing FEM view of the
    mesh, mainly managing the set of degrees of freedom. */
class FiniteElementSpace
{
   friend class InterpolationGridTransfer;

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

   int nvdofs, nedofs, nfdofs, nbdofs;
   int *fdofs, *bdofs;

   mutable Table *elem_dof; // if NURBS FE space, not owned; otherwise, owned.
   Table *bdrElem_dof; // used only with NURBS FE spaces; not owned.

   Array<int> dof_elem_array, dof_ldof_array;

   NURBSExtension *NURBSext;
   int own_ext;

   /** Matrix representing the prolongation from the global conforming dofs to
       a set of intermediate partially conforming dofs, e.g. the dofs associated
       with a "cut" space on a non-conforming mesh. */
   mutable SparseMatrix *cP; // owned
   /// Conforming restriction matrix such that cR.cP=I.
   mutable SparseMatrix *cR; // owned
   mutable bool cP_is_set;

   /// Transformation to apply to GridFunctions after space Update().
   OperatorHandle Th;

   /// The element restriction operators, see GetElementRestriction().
   mutable OperatorHandle L2E_nat, L2E_lex;

   mutable Array<QuadratureInterpolator*> E2Q_array;

   long sequence; // should match Mesh::GetSequence

   void UpdateNURBS();

   void Construct();
   void Destroy();

   void BuildElementToDofTable() const;

   /// Helper to remove encoded sign from a DOF
   static inline int DecodeDof(int dof, double& sign)
   { return (dof >= 0) ? (sign = 1, dof) : (sign = -1, (-1 - dof)); }

   /// Helper to get vertex, edge or face DOFs (entity=0,1,2 resp.).
   void GetEntityDofs(int entity, int index, Array<int> &dofs,
                      Geometry::Type master_geom = Geometry::INVALID) const;
   // Get degenerate face DOFs: see explanation in method implementation.
   void GetDegenerateFaceDofs(int index, Array<int> &dofs,
                              Geometry::Type master_geom) const;

   /// Calculate the cP and cR matrices for a nonconforming mesh.
   void BuildConformingInterpolation() const;

   static void AddDependencies(SparseMatrix& deps, Array<int>& master_dofs,
                               Array<int>& slave_dofs, DenseMatrix& I);

   static bool DofFinalizable(int dof, const Array<bool>& finalized,
                              const SparseMatrix& deps);

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
      virtual ~RefinementOperator();
   };

   // Derefinement operator, used by the friend class InterpolationGridTransfer.
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

   // This method makes the same assumptions as the method:
   //    void GetLocalRefinementMatrices(
   //       const FiniteElementSpace &coarse_fes, Geometry::Type geom,
   //       DenseTensor &localP) const
   // which is defined below. It also assumes that the coarse fes and this have
   // the same vector dimension, vdim.
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

   // This method assumes that this->mesh is a refinement of coarse_fes->mesh
   // and that the CoarseFineTransformations of this->mesh are set accordingly.
   // Another assumption is that the FEs of this use the same MapType as the FEs
   // of coarse_fes. Finally, it assumes that the spaces this and coarse_fes are
   // NOT variable-order spaces.
   void GetLocalRefinementMatrices(const FiniteElementSpace &coarse_fes,
                                   Geometry::Type geom,
                                   DenseTensor &localP) const;

   /// Help function for constructors + Load().
   void Constructor(Mesh *mesh, NURBSExtension *ext,
                    const FiniteElementCollection *fec,
                    int vdim = 1, int ordering = Ordering::byNODES);

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

   bool Conforming() const { return mesh->Conforming(); }
   bool Nonconforming() const { return mesh->Nonconforming(); }

   /// The returned SparseMatrix is owned by the FiniteElementSpace.
   const SparseMatrix *GetConformingProlongation() const;

   /// The returned SparseMatrix is owned by the FiniteElementSpace.
   const SparseMatrix *GetConformingRestriction() const;

   /// The returned Operator is owned by the FiniteElementSpace.
   virtual const Operator *GetProlongationMatrix() const
   { return GetConformingProlongation(); }

   /// The returned SparseMatrix is owned by the FiniteElementSpace.
   virtual const SparseMatrix *GetRestrictionMatrix() const
   { return GetConformingRestriction(); }

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

   /// Returns vector dimension.
   inline int GetVDim() const { return vdim; }

   /// Returns the order of the i'th finite element
   int GetOrder(int i) const;
   /// Returns the order of the i'th face finite element
   int GetFaceOrder(int i) const;

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

   /// Returns indexes of degrees of freedom in array dofs for i'th element.
   virtual void GetElementDofs(int i, Array<int> &dofs) const;

   /// Returns indexes of degrees of freedom for i'th boundary element.
   virtual void GetBdrElementDofs(int i, Array<int> &dofs) const;

   /** Returns the indexes of the degrees of freedom for i'th face
       including the dofs for the edges and the vertices of the face. */
   virtual void GetFaceDofs(int i, Array<int> &dofs) const;

   /** Returns the indexes of the degrees of freedom for i'th edge
       including the dofs for the vertices of the edge. */
   void GetEdgeDofs(int i, Array<int> &dofs) const;

   void GetVertexDofs(int i, Array<int> &dofs) const;

   void GetElementInteriorDofs(int i, Array<int> &dofs) const;

   void GetFaceInteriorDofs(int i, Array<int> &dofs) const;

   int GetNumElementInteriorDofs(int i) const
   { return fec->DofForGeometry(mesh->GetElementBaseGeometry(i)); }

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

   void RebuildElementToDofTable();

   /** @brief Reorder the scalar DOFs based on the element ordering.

       The new ordering is constructed as follows: 1) loop over all elements as
       ordered in the Mesh; 2) for each element, assign new indices to all of
       its current DOFs that are still unassigned; the new indices we assign are
       simply the sequence `0,1,2,...`; if there are any signed DOFs their sign
       is preserved. */
   void ReorderElementToDofTable();

   void BuildDofToArrays();

   const Table &GetElementToDofTable() const { return *elem_dof; }
   const Table &GetBdrElementToDofTable() const { return *bdrElem_dof; }

   int GetElementForDof(int i) const { return dof_elem_array[i]; }
   int GetLocalDofForDof(int i) const { return dof_ldof_array[i]; }

   /// Returns pointer to the FiniteElement associated with i'th element.
   const FiniteElement *GetFE(int i) const;

   /// Returns pointer to the FiniteElement for the i'th boundary element.
   const FiniteElement *GetBE(int i) const;

   const FiniteElement *GetFaceElement(int i) const;

   const FiniteElement *GetEdgeElement(int i) const;

   /// Return the trace element from element 'i' to the given 'geom_type'
   const FiniteElement *GetTraceElement(int i, Geometry::Type geom_type) const;

   /** Mark degrees of freedom associated with boundary elements with
       the specified boundary attributes (marked in 'bdr_attr_is_ess').
       For spaces with 'vdim' > 1, the 'component' parameter can be used
       to restricts the marked vDOFs to the specified component. */
   virtual void GetEssentialVDofs(const Array<int> &bdr_attr_is_ess,
                                  Array<int> &ess_vdofs,
                                  int component = -1) const;

   /** Get a list of essential true dofs, ess_tdof_list, corresponding to the
       boundary attributes marked in the array bdr_attr_is_ess.
       For spaces with 'vdim' > 1, the 'component' parameter can be used
       to restricts the marked tDOFs to the specified component. */
   virtual void GetEssentialTrueDofs(const Array<int> &bdr_attr_is_ess,
                                     Array<int> &ess_tdof_list,
                                     int component = -1);

   /// Convert a Boolean marker array to a list containing all marked indices.
   static void MarkerToList(const Array<int> &marker, Array<int> &list);

   /** Convert an array of indices (list) to a Boolean marker array where all
       indices in the list are marked with the given value and the rest are set
       to zero. */
   static void ListToMarker(const Array<int> &list, int marker_size,
                            Array<int> &marker, int mark_val = -1);

   /** For a partially conforming FE space, convert a marker array (nonzero
       entries are true) on the partially conforming dofs to a marker array on
       the conforming dofs. A conforming dofs is marked iff at least one of its
       dependent dofs is marked. */
   void ConvertToConformingVDofs(const Array<int> &dofs, Array<int> &cdofs);

   /** For a partially conforming FE space, convert a marker array (nonzero
       entries are true) on the conforming dofs to a marker array on the
       (partially conforming) dofs. A dof is marked iff it depends on a marked
       conforming dofs, where dependency is defined by the ConformingRestriction
       matrix; in other words, a dof is marked iff it corresponds to a marked
       conforming dof. */
   void ConvertFromConformingVDofs(const Array<int> &cdofs, Array<int> &dofs);

   /** Generate the global restriction matrix from a discontinuous
       FE space to the continuous FE space of the same polynomial degree. */
   SparseMatrix *D2C_GlobalRestrictionMatrix(FiniteElementSpace *cfes);

   /** Generate the global restriction matrix from a discontinuous
       FE space to the piecewise constant FE space. */
   SparseMatrix *D2Const_GlobalRestrictionMatrix(FiniteElementSpace *cfes);

   /** Construct the restriction matrix from the FE space given by
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

   /** Reflect changes in the mesh: update number of DOFs, etc. Also, calculate
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

   /// Return update counter (see Mesh::sequence)
   long GetSequence() const { return sequence; }

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

   /// Get the IntegrationRule associated with mesh element @a idx.
   const IntegrationRule &GetElementIntRule(int idx) const
   { return *int_rule[mesh->GetElementBaseGeometry(idx)]; }

   /// Write the QuadratureSpace to the stream @a out.
   void Save(std::ostream &out) const;
};


/** @brief Base class for transfer algorithms that construct transfer Operator%s
    between two finite element (FE) spaces. */
/** Generally, the two FE spaces (domain and range) can be defined on different
    meshes. */
class GridTransfer
{
protected:
   FiniteElementSpace &dom_fes; ///< Domain FE space
   FiniteElementSpace &ran_fes; ///< Range FE space

   /** @brief Desired Operator::Type for the construction of all operators
       defined by the underlying transfer algorithm. It can be ignored by
       derived classes. */
   Operator::Type oper_type;

   OperatorHandle fw_t_oper; ///< Forward true-dof operator
   OperatorHandle bw_t_oper; ///< Backward true-dof operator

#ifdef MFEM_USE_MPI
   bool parallel;
#endif
   bool Parallel() const
   {
#ifndef MFEM_USE_MPI
      return false;
#else
      return parallel;
#endif
   }

   const Operator &MakeTrueOperator(FiniteElementSpace &fes_in,
                                    FiniteElementSpace &fes_out,
                                    const Operator &oper,
                                    OperatorHandle &t_oper);

public:
   /** Construct a transfer algorithm between the domain, @a dom_fes_, and
       range, @a ran_fes_, FE spaces. */
   GridTransfer(FiniteElementSpace &dom_fes_, FiniteElementSpace &ran_fes_);

   /// Virtual destructor
   virtual ~GridTransfer() { }

   /** @brief Set the desired Operator::Type for the construction of all
       operators defined by the underlying transfer algorithm. */
   /** The default value is Operator::ANY_TYPE which typically corresponds to
       a matrix-free operator representation. Note that derived classes are not
       required to support this setting and can ignore it. */
   void SetOperatorType(Operator::Type type) { oper_type = type; }

   /** @brief Return an Operator that transfers GridFunction%s from the domain
       FE space to GridFunction%s in the range FE space. */
   virtual const Operator &ForwardOperator() = 0;

   /** @brief Return an Operator that transfers GridFunction%s from the range
       FE space back to GridFunction%s in the domain FE space. */
   virtual const Operator &BackwardOperator() = 0;

   /** @brief Return an Operator that transfers true-dof Vector%s from the
       domain FE space to true-dof Vector%s in the range FE space. */
   /** This method is implemented in the base class, based on ForwardOperator(),
       however, derived classes can overload the construction, if necessary. */
   virtual const Operator &TrueForwardOperator()
   {
      return MakeTrueOperator(dom_fes, ran_fes, ForwardOperator(), fw_t_oper);
   }

   /** @brief Return an Operator that transfers true-dof Vector%s from the
       range FE space back to true-dof Vector%s in the domain FE space. */
   /** This method is implemented in the base class, based on
       BackwardOperator(), however, derived classes can overload the
       construction, if necessary. */
   virtual const Operator &TrueBackwardOperator()
   {
      return MakeTrueOperator(ran_fes, dom_fes, BackwardOperator(), bw_t_oper);
   }
};


/** @brief Transfer data between a coarse mesh and an embedded refined mesh
    using interpolation. */
/** The forward, coarse-to-fine, transfer uses nodal interpolation. The
    backward, fine-to-coarse, transfer is defined locally (on a coarse element)
    as B = (F^t M_f F)^{-1} F^t M_f, where F is the forward transfer matrix, and
    M_f is a mass matrix on the union of all fine elements comprising the coarse
    element. Note that the backward transfer operator, B, is a left inverse of
    the forward transfer operator, F, i.e. B F = I. Both F and B are defined in
    reference space and do not depend on the actual physical shape of the mesh
    elements.

    It is assumed that both the coarse and the fine FiniteElementSpace%s use
    compatible types of elements, e.g. finite elements with the same map-type
    (VALUE, INTEGRAL, H_DIV, H_CURL - see class FiniteElement). Generally, the
    FE spaces can have different orders, however, in order for the backward
    operator to be well-defined, the (local) number of the fine dofs should not
    be smaller than the number of coarse dofs. */
class InterpolationGridTransfer : public GridTransfer
{
protected:
   BilinearFormIntegrator *mass_integ; ///< Ownership depends on #own_mass_integ
   bool own_mass_integ; ///< Ownership flag for #mass_integ

   OperatorHandle F; ///< Forward, coarse-to-fine, operator
   OperatorHandle B; ///< Backward, fine-to-coarse, operator

public:
   InterpolationGridTransfer(FiniteElementSpace &coarse_fes,
                             FiniteElementSpace &fine_fes)
      : GridTransfer(coarse_fes, fine_fes),
        mass_integ(NULL), own_mass_integ(false)
   { }

   virtual ~InterpolationGridTransfer();

   /** @brief Assign a mass integrator to be used in the construction of the
       backward, fine-to-coarse, transfer operator. */
   void SetMassIntegrator(BilinearFormIntegrator *mass_integ_,
                          bool own_mass_integ_ = true);

   virtual const Operator &ForwardOperator();

   virtual const Operator &BackwardOperator();
};


/** @brief Transfer data between a coarse mesh and an embedded refined mesh
    using L2 projection. */
/** The forward, coarse-to-fine, transfer uses L2 projection. The backward,
    fine-to-coarse, transfer is defined locally (on a coarse element) as
    B = (F^t M_f F)^{-1} F^t M_f, where F is the forward transfer matrix, and
    M_f is the mass matrix on the union of all fine elements comprising the
    coarse element. Note that the backward transfer operator, B, is a left
    inverse of the forward transfer operator, F, i.e. B F = I. Both F and B are
    defined in physical space and, generally, vary between different mesh
    elements.

    This class currently only fully supports L2 finite element spaces and fine
    meshes that are a uniform refinement of the coarse mesh. Generally, the
    coarse and fine FE spaces can have different orders, however, in order for
    the backward operator to be well-defined, the number of the fine dofs (in a
    coarse element) should not be smaller than the number of coarse dofs.

    If used on H1 finite element spaces, the transfer will be performed locally,
    and the value of shared (interface) degrees of freedom will be determined by
    the value of the last transfer to be performed (according to the element
    numbering in the finite element space). As a consequence, the mass
    conservation properties for this operator from the L2 case do not carry over
    to H1 spaces. */
class L2ProjectionGridTransfer : public GridTransfer
{
protected:
   /** Class representing projection operator between a high-order L2 finite
       element space on a coarse mesh, and a low-order L2 finite element space
       on a refined mesh (LOR). We assume that the low-order space, fes_lor,
       lives on a mesh obtained by refining the mesh of the high-order space,
       fes_ho. */
   class L2Projection : public Operator
   {
      const FiniteElementSpace &fes_ho;
      const FiniteElementSpace &fes_lor;

      int ndof_lor, ndof_ho, nref;

      Table ho2lor;

      DenseTensor R, P;

   public:
      L2Projection(const FiniteElementSpace &fes_ho_,
                   const FiniteElementSpace &fes_lor_);
      /// Perform the L2 projection onto the LOR space
      virtual void Mult(const Vector &x, Vector &y) const;
      /// Perform the mass conservative left-inverse prolongation operation.
      /// This functionality is also provided as an Operator by L2Prolongation.
      void Prolongate(const Vector &x, Vector &y) const;
      virtual ~L2Projection() { }
   };

   /** Mass-conservative prolongation operator going in the opposite direction
       as L2Projection. This operator is a left inverse to the L2Projection. */
   class L2Prolongation : public Operator
   {
      const L2Projection &l2proj;

   public:
      L2Prolongation(const L2Projection &l2proj_) : l2proj(l2proj_) { }
      void Mult(const Vector &x, Vector &y) const
      {
         l2proj.Prolongate(x, y);
      }
      virtual ~L2Prolongation() { }
   };

   L2Projection   *F; ///< Forward, coarse-to-fine, operator
   L2Prolongation *B; ///< Backward, fine-to-coarse, operator

public:
   L2ProjectionGridTransfer(FiniteElementSpace &coarse_fes,
                            FiniteElementSpace &fine_fes)
      : GridTransfer(coarse_fes, fine_fes),
        F(NULL), B(NULL)
   { }

   virtual const Operator &ForwardOperator();

   virtual const Operator &BackwardOperator();
};


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

/** @brief A class that performs interpolation from an E-vector to quadrature
    point values and/or derivatives (Q-vectors). */
/** An E-vector represents the element-wise discontinuous version of the FE
    space and can be obtained, for example, from a GridFunction using the
    Operator returned by FiniteElementSpace::GetElementRestriction().

    The target quadrature points in the elements can be described either by an
    IntegrationRule (all mesh elements must be of the same type in this case) or
    by a QuadratureSpace. */
class QuadratureInterpolator
{
protected:
   friend class FiniteElementSpace; // Needs access to qspace and IntRule

   const FiniteElementSpace *fespace;  ///< Not owned
   const QuadratureSpace *qspace;      ///< Not owned
   const IntegrationRule *IntRule;     ///< Not owned

   mutable bool use_tensor_products;

   static const int MAX_NQ2D = 100;
   static const int MAX_ND2D = 100;
   static const int MAX_VDIM2D = 2;

   static const int MAX_NQ3D = 1000;
   static const int MAX_ND3D = 1000;
   static const int MAX_VDIM3D = 3;

public:
   enum EvalFlags
   {
      VALUES       = 1 << 0,  ///< Evaluate the values at quadrature points
      DERIVATIVES  = 1 << 1,  ///< Evaluate the derivatives at quadrature points
      /** @brief Assuming the derivative at quadrature points form a matrix,
          this flag can be used to compute and store their determinants. This
          flag can only be used in Mult(). */
      DETERMINANTS = 1 << 2
   };

   QuadratureInterpolator(const FiniteElementSpace &fes,
                          const IntegrationRule &ir);

   QuadratureInterpolator(const FiniteElementSpace &fes,
                          const QuadratureSpace &qs);

   /** @brief Disable the use of tensor product evaluations, for tensor-product
       elements, e.g. quads and hexes. */
   /** Currently, tensor product evaluations are not implemented and this method
       has no effect. */
   void DisableTensorProducts(bool disable = true) const
   { use_tensor_products = !disable; }

   /// Interpolate the E-vector @a e_vec to quadrature points.
   /** The @a eval_flags are a bitwise mask of constants from the EvalFlags
       enumeration. When the VALUES flag is set, the values at quadrature points
       are computed and stored in the Vector @a q_val. Similarly, when the flag
       DERIVATIVES is set, the derivatives are computed and stored in @a q_der.
       When the DETERMINANTS flags is set, it is assumed that the derivatives
       form a matrix at each quadrature point (i.e. the associated
       FiniteElementSpace is a vector space) and their determinants are computed
       and stored in @a q_det. */
   void Mult(const Vector &e_vec, unsigned eval_flags,
             Vector &q_val, Vector &q_der, Vector &q_det) const;

   /// Perform the transpose operation of Mult(). (TODO)
   void MultTranspose(unsigned eval_flags, const Vector &q_val,
                      const Vector &q_der, Vector &e_vec) const;

   // Compute kernels follow (cannot be private or protected with nvcc)

   /// Template compute kernel for 2D.
   template<const int T_VDIM = 0, const int T_ND = 0, const int T_NQ = 0>
   static void Eval2D(const int NE,
                      const int vdim,
                      const DofToQuad &maps,
                      const Vector &e_vec,
                      Vector &q_val,
                      Vector &q_der,
                      Vector &q_det,
                      const int eval_flags);

   /// Template compute kernel for 3D.
   template<const int T_VDIM = 0, const int T_ND = 0, const int T_NQ = 0>
   static void Eval3D(const int NE,
                      const int vdim,
                      const DofToQuad &maps,
                      const Vector &e_vec,
                      Vector &q_val,
                      Vector &q_der,
                      Vector &q_det,
                      const int eval_flags);
};

}

#endif
