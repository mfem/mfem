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

enum AssemblyType
{
   FullAssembly,
   PartialAssembly,
   NoAssembly
};

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


class NURBSExtension;

/** @brief Class FiniteElementSpace - responsible for providing FEM view of the
    mesh, mainly managing the set of degrees of freedom. */
class FiniteElementSpace
{
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

   int nvdofs, nedofs, nfdofs, nbdofs, nldofs;
   int *fdofs, *bdofs;

   mutable Table *elem_dof; // if NURBS FE space, not owned; otherwise, owned.
   Table *bdrElem_dof; // used only with NURBS FE spaces; not owned.

   Array<int> dof_elem_array, dof_ldof_array;
   Array<int> *tensor_offsets, *tensor_indices;

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
   Operator *T;
   bool own_T;

   long sequence; // should match Mesh::GetSequence

   void UpdateNURBS();

   void Construct();
   void Destroy();

   void BuildElementToDofTable() const;

   /// Helper to get vertex, edge or face DOFs (entity=0,1,2 resp.).
   void GetEntityDofs(int entity, int index, Array<int> &dofs) const;

   /// Calculate the cP and cR matrices for a nonconforming mesh.
   void BuildConformingInterpolation() const;

   static void AddDependencies(SparseMatrix& deps, Array<int>& master_dofs,
                               Array<int>& slave_dofs, DenseMatrix& I);

   static bool DofFinalizable(int dof, const Array<bool>& finalized,
                              const SparseMatrix& deps);

   void MakeVDimMatrix(SparseMatrix &mat) const;

   /// Calculate GridFunction interpolation matrix after mesh refinement.
   SparseMatrix* RefinementMatrix(int old_ndofs, const Table* old_elem_dof);

   void GetLocalDerefinementMatrices(int geom, const CoarseFineTransformations &dt,
                                     DenseTensor &localR);

   /// Calculate GridFunction restriction matrix after mesh derefinement.
   SparseMatrix* DerefinementMatrix(int old_ndofs, const Table* old_elem_dof);

   /// Help function for constructors.
   void Constructor(Mesh *mesh, NURBSExtension *ext,
                    const FiniteElementCollection *fec,
                    int vdim = 1, int ordering = Ordering::byNODES);

public:
   /** @brief Default constructor: the object is invalid until initialized using
       the method Load(). */
   FiniteElementSpace();

   /** @brief Copy constructor: deep copy all data from @a orig except the Mesh,
       the FiniteElementCollection, ans some derived data. */
   /** If the @a mesh or @a fec poiters are NULL (default), then the new
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

   const SparseMatrix *GetConformingProlongation() const;
   const SparseMatrix *GetConformingRestriction() const;

   virtual const Operator *GetProlongationMatrix()
   { return GetConformingProlongation(); }
   virtual const SparseMatrix *GetRestrictionMatrix()
   { return GetConformingRestriction(); }

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

   /// Return the number of DOFs on all elements unrolled.
   inline int GetLocalVSize() const { return vdim * nldofs; }

   /// Returns the number of conforming ("true") degrees of freedom
   /// (if the space is on a nonconforming mesh with hanging nodes).
   int GetNConformingDofs() const;

   int GetConformingVSize() const { return vdim * GetNConformingDofs(); }

   /// Return the ordering method.
   inline Ordering::Type GetOrdering() const { return ordering; }

   const FiniteElementCollection *FEColl() const { return fec; }

   int GetNVDofs() const { return nvdofs; }
   int GetNEDofs() const { return nedofs; }
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
   const FiniteElement *GetTraceElement(int i, int geom_type) const;

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

   /** Reflect changes in the mesh: update number of DOFs, etc. Also, calculate
       GridFunction transformation matrix (unless want_transform is false).
       Safe to call multiple times, does nothing if space already up to date. */
   virtual void Update(bool want_transform = true);

   /// Get the GridFunction update matrix.
   const Operator* GetUpdateOperator() { Update(); return T; }

   /** @brief Set the ownership of the update operator: if set to false, the
       Operator returned by GetUpdateOperator() must be deleted outside the
       FiniteElementSpace. */
   void SetUpdateOperatorOwner(bool own) { own_T = own; }

   /// Free GridFunction transformation matrix (if any), to save memory.
   virtual void UpdatesFinished() { if (own_T) { delete T; } T = NULL; }

   /// Return update counter (see Mesh::sequence)
   long GetSequence() const { return sequence; }

   void Save(std::ostream &out) const;

   /// Make a vector corresponding to a grid function on the finite element space
   void ToLocalVector(const Vector &v, Vector &V);
   void ToGlobalVector(const Vector &V, Vector &v);

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
   int GetSize() { return size; }

   /// Get the IntegrationRule associated with mesh element @a idx.
   const IntegrationRule &GetElementIntRule(int idx)
   { return *int_rule[mesh->GetElementBaseGeometry(idx)]; }

   /// Write the QuadratureSpace to the stream @a out.
   void Save(std::ostream &out) const;
};

}

#endif
