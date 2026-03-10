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

#ifndef MFEM_FESPACE
#define MFEM_FESPACE

#include "../config/config.hpp"
#include "../general/hash_util.hpp"
#include "../linalg/ordering.hpp"
#include "../linalg/sparsemat.hpp"
#include "../mesh/mesh.hpp"
#include "fe_coll.hpp"
#include "doftrans.hpp"
#include "restriction.hpp"
#include <iostream>
#include <unordered_map>

namespace mfem
{

/// @brief Type describing possible layouts for Q-vectors.
/// @sa QuadratureInterpolator and FaceQuadratureInterpolator.
enum class QVectorLayout
{
   /** Layout depending on the input space and the computed quantity:
       - scalar H1/L2 spaces, values: NQPT x VDIM x NE,
       - scalar H1/L2 spaces, gradients: NQPT x VDIM x DIM x NE,
       - vector RT/ND spaces, values: NQPT x SDIM x NE (vdim = 1). */
   byNODES,

   /** Layout depending on the input space and the computed quantity:
       - scalar H1/L2 spaces, values: VDIM x NQPT x NE,
       - scalar H1/L2 spaces, gradients: VDIM x DIM x NQPT x NE,
       - vector RT/ND spaces, values: SDIM x NQPT x NE (vdim = 1). */
   byVDIM
};

/// Constants describing the possible orderings of the DOFs in one element.
enum class ElementDofOrdering
{
   /// Native ordering as defined by the FiniteElement.
   /** This ordering can be used by tensor-product elements when the
       interpolation from the DOFs to quadrature points does not use the
       tensor-product structure. */
   NATIVE,
   /// Lexicographic: DOFs are listed in order of increasing x-coordinate,
   /// followed by increasing y-coordinate, and z-coordinate.
   /** This ordering is usually used with tensor-product elements, but it is
       also supported by some non-tensor elements. */
   LEXICOGRAPHIC
};

/** Represents the index of an element to p-refine, plus a change to the order
    of that element. */
struct pRefinement
{
   int index; ///< Mesh element number
   int delta; ///< Change to element order

   pRefinement() = default;

   pRefinement(int element, int change)
      : index(element), delta(change) {}
};

// Forward declarations
class NURBSExtension;
class BilinearFormIntegrator;
class QuadratureSpace;
class QuadratureInterpolator;
class FaceQuadratureInterpolator;
class PRefinementTransferOperator;
struct DerefineMatrixOp;

/** @brief Class FiniteElementSpace - responsible for providing FEM view of the
    mesh, mainly managing the set of degrees of freedom.

    @details The term "degree of freedom", or "dof" for short, can mean
    different things in different contexts. In MFEM we use "dof" to refer to
    four closely related types of data; @ref edof "edofs", @ref ldof "ldofs",
    @ref tdof "tdofs", and @ref vdof "vdofs".

    @anchor edof @par Element DoF:
    %Element dofs, sometimes referred to as @b edofs, are the expansion
    coefficients used to build the linear combination of basis functions which
    approximate a field within one element of the computational mesh. The
    arrangement of the element dofs is determined by the basis function and
    element types.
    @par
    %Element dofs are usually accessed one element at a time but they can be
    concatenated together into a global vector when minimizing access time is
    crucial. The global number of element dofs is not directly available from
    the FiniteElementSpace. It can be determined by repeatedly calling
    FiniteElementSpace::GetElementDofs and summing the lengths of the resulting
    @a dofs arrays.

    @anchor ldof @par Local DoF:
    Most basis function types share many of their element dofs with neighboring
    elements. Consequently, the global @ref edof "edof" vector suggested above
    would contain many redundant entries. One of the primary roles of the
    FiniteElementSpace is to collapse out these redundancies and
    define a unique ordering of the remaining degrees of freedom. The
    collapsed set of dofs are called @b "local dofs" or @b ldofs in
    the MFEM parlance.
    @par
    The term @b local in this context refers to the local rank in a parallel
    processing environment. MFEM can, of course, be used in sequential
    computing environments but it is designed with parallel processing in mind
    and this terminology reflects that design focus.
    @par
    When running in parallel the set of local dofs contains all of the degrees
    of freedom associated with locally owned elements. When running in serial
    all elements are locally owned so all element dofs are represented in the
    set of local dofs.
    @par
    There are two important caveats regarding local dofs. First, some basis
    function types, Nedelec and Raviart-Thomas are the prime examples, have an
    orientation associated with each basis function. The relative orientations
    of such basis functions in neighboring elements can lead to shared degrees
    of freedom with opposite signs from the point of view of these neighboring
    elements. MFEM typically chooses the orientation of the first such shared
    degree of freedom that it encounters as the default orientation for the
    corresponding local dof. When this local dof is referenced by a neighboring
    element which happens to require the opposite orientation the local dof
    index will be returned (by calls to functions such as
    FiniteElementSpace::GetElementDofs) as a negative integer. In such cases
    the actual offset into the vector of local dofs is @b -index-1 and the
    value expected by this element should have the opposite sign to the value
    stored in the local dof vector.
    @par
    The second important caveat only pertains to high order Nedelec basis
    functions when shared triangular faces are present in the mesh. In this
    very particular case the relative orientation of the face with respect to
    its two neighboring elements can lead to different definitions of the
    degrees of freedom associated with the interior of the face which cannot
    be handled by simply flipping the signs of the corresponding values. The
    DofTransformation class is designed to manage the necessary @b edof to
    @b ldof transformations in this case. In the majority of cases the
    DofTransformation is unnecessary and a NULL pointer will be returned in
    place of a pointer to this object. See DofTransformation for more
    information.

    @anchor tdof @par True DoF:
    As the name suggests "true dofs" or @b tdofs form the minimal set of data
    values needed (along with mesh and basis function definitions) to uniquely
    define a finite element discretization of a field. The number of true dofs
    determines the size of the linear systems which typically need to be solved
    in FEM simulations.
    @par
    Often the true dofs and the local dofs are identical, however, there are
    important cases where they differ significantly. The first such case is
    related to non-conforming meshes. On non-conforming meshes it is common
    for degrees of freedom associated with "hanging" nodes, edges, or faces to
    be constrained by degrees of freedom associated with another mesh entity.
    In such cases the "hanging" degrees of freedom should not be considered
    "true" degrees of freedom since their values cannot be independently
    assigned. For this reason the FiniteElementSpace must process these
    constraints and define a reduced set of "true" degrees of freedom which are
    distinct from the local degrees of freedom.
    @par
    The second important distinction arises in parallel processing. When
    distributing a linear system in parallel each degree of freedom must be
    assigned to a particular processor, its owner. From the finite element
    point of view it is convenient to distribute a computational mesh and
    define an owning processor for each element. Since degrees of freedom may
    be shared between neighboring elements they may also be shared between
    neighboring processors. Another role of the FiniteElementSpace is to
    identify the ownership of degrees of freedom which must be shared between
    processors. Therefore the set of "true" degrees of freedom must also remove
    redundant degrees of freedom which are owned by other processors.
    @par
    To summarize the set of true degrees of freedom are those degrees of
    freedom needed to solve a linear system representing the partial
    differential equation being modeled. True dofs differ from "local" dofs by
    eliminating redundancies across processor boundaries and applying
    the constraints needed to properly define fields on non-conforming meshes.

    @anchor vdof @par Vector DoF:
    %Vector dofs or @b vdofs are related to fields which are constructed using
    multiple copies of the same set of basis functions. A typical example would
    be the use of three instances of the scalar H1 basis functions to
    approximate the x, y, and z components of a displacement vector field in
    three dimensional space as often seen in elasticity simulations.
    @par
    %Vector dofs do not represent a specific index space the way the three
    previous types of dofs do. Rather they are related to modifications of
    these other index spaces to accommodate multiple copies of the underlying
    function spaces.
    @par
    When using @b vdofs, i.e. when @b vdim != 1, the FiniteElementSpace only
    manages a single set of degrees of freedom and then uses simple rules to
    determine the appropriate offsets into the full index spaces. Two ordering
    rules are supported; @b byNODES and @b byVDIM. See Ordering::Type for
    details.
    @par
    Clearly the notion of a @b vdof is relevant in each of the three contexts
    mentioned above so extra care must be taken whenever @b vdim != 1 to ensure
    that the @b edof, @b ldof, or @b tdof is being interpreted correctly.
 */
class FiniteElementSpace
{
   friend class InterpolationGridTransfer;
   friend class PRefinementTransferOperator;
   friend void Mesh::Swap(Mesh &, bool);
   friend class LORBase;
   friend struct DerefineMatrixOp;

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

   bool variableOrder = false;

   /** Polynomial order for each element. If empty, all elements are assumed
       to be of the default order (fec->GetOrder()). */
   Array<char> elem_order;

   int nvdofs, nedofs, nfdofs, nbdofs, lnedofs, lnfdofs;
   int uni_fdof; ///< # of single face DOFs if all faces uniform; -1 otherwise
   int *bdofs; ///< internal DOFs of elements if mixed/var-order; NULL otherwise

   /** Variable-order spaces only: DOF assignments for edges and faces, see
       docs in MakeDofTable. For constant order spaces the tables are empty. */
   Table var_edge_dofs;
   Table var_face_dofs; ///< NOTE: also used for spaces with mixed faces

   // Temporary data for condensing all DOFs to local DOFs.
   Table loc_var_edge_dofs, loc_var_face_dofs;

   /** Map from all DOFs of all orders on each entity to local DOFs of orders
       occurring on a local element containing the entity. */
   Array<int> all2local;

   /// Bit-mask representing a set of orders needed by an edge/face.
   typedef std::uint64_t VarOrderBits;
   static constexpr int MaxVarOrder = 8*sizeof(VarOrderBits) - 1;

   /** Additional data for the var_*_dofs tables: individual variant orders
       (these are basically alternate J arrays for var_edge/face_dofs). */
   Array<char> var_edge_orders, var_face_orders;
   Array<char> loc_var_edge_orders, loc_var_face_orders;
   Array<char> ghost_edge_orders, ghost_face_orders;

   /// Minimum order among neighboring elements.
   mutable Array<int> edge_min_nghb_order, face_min_nghb_order;

   /// Marker arrays for ghost master entities to be skipped in conforming
   /// interpolation constraints.
   Array<bool> skip_edge, skip_face;

   // precalculated DOFs for each element, boundary element, and face
   mutable Table *elem_dof; // owned (except in NURBS FE space)
   mutable Table *elem_fos; // face orientations by element index
   mutable Table *bdr_elem_dof; // owned (except in NURBS FE space)
   mutable Table *bdr_elem_fos; // bdr face orientations by bdr element index
   mutable Table *face_dof; // owned; in var-order space contains variant 0 DOFs

   mutable Array<int> dof_elem_array;
   mutable Array<int> dof_ldof_array;
   mutable Array<int> dof_bdr_elem_array;
   mutable Array<int> dof_bdr_ldof_array;

   NURBSExtension *NURBSext;
   /** array of NURBS extension for H(div) and H(curl) vector elements.
       For each direction an extension is created from the base NURBSext,
       with an increase in order in the appropriate direction. */
   Array<NURBSExtension*> VNURBSext;
   int own_ext;
   mutable Array<int> face_to_be; // NURBS FE space only

   Array<StatelessDofTransformation *> DoFTransArray;
   mutable DofTransformation DoFTrans;

   /** Matrix representing the prolongation from the global conforming dofs to
       a set of intermediate partially conforming dofs, e.g. the dofs associated
       with a "cut" space on a non-conforming mesh. */
   mutable std::unique_ptr<SparseMatrix> cP;
   /// Conforming restriction matrix such that cR.cP=I.
   mutable std::unique_ptr<SparseMatrix> cR;
   /// A version of the conforming restriction matrix for variable-order spaces.
   mutable std::unique_ptr<SparseMatrix> cR_hp;
   mutable bool cP_is_set;
   /// Operator computing the action of the transpose of the restriction.
   mutable std::unique_ptr<Operator> R_transpose;

   /** Stores the previous FiniteElementSpace, before p-refinement, in the case
       that @a PTh is constructed by PRefineAndUpdate(). */
   std::unique_ptr<FiniteElementSpace> fesPrev;

   /// Transformation to apply to GridFunctions after space Update().
   OperatorHandle Th;

   std::shared_ptr<PRefinementTransferOperator> PTh;

   /// Flag to indicate whether the last update was for p-refinement.
   bool lastUpdatePRef = false;

   /// The element restriction operators, see GetElementRestriction().
   mutable OperatorHandle L2E_nat, L2E_lex;
   /// The face restriction operators, see GetFaceRestriction().
   using key_face = std::tuple<bool, ElementDofOrdering, FaceType, L2FaceValues>;
   mutable std::unordered_map<key_face,std::unique_ptr<FaceRestriction>,
           TupleHasher> L2F;

   mutable std::unordered_map<std::tuple<ElementDofOrdering,FaceType>,
           std::unique_ptr<InterpolationManager>, TupleHasher> interpolations;

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

   /** Helper function for constructing the data in this class, for initial
       construction or updates (e.g. h- or p-refinement). */
   void Construct();

   void Destroy();

   void ConstructDoFTransArray();
   void DestroyDoFTransArray();

   void BuildElementToDofTable() const;
   void BuildBdrElementToDofTable() const;
   void BuildFaceToDofTable() const;

   /** Get all @a edges and @a faces (in 3D) on boundary elements with attribute
       marked as essential in @a bdr_attr_is_ess. */
   void GetEssentialBdrEdgesFaces(const Array<int> &bdr_attr_is_ess,
                                  std::set<int> & edges,
                                  std::set<int> & faces) const;

   /** @brief Initialize internal data that enables the use of the methods
    GetElementForDof() and GetLocalDofForDof(). */
   void BuildDofToArrays_() const;

   /** @brief Initialize internal data that enables the use of the methods
      GetBdrElementForDof() and GetBdrLocalDofForDof(). */
   void BuildDofToBdrArrays() const;

   /** @brief  Generates partial face_dof table for a NURBS space.

       The table is only defined for exterior faces that coincide with a
       boundary. */
   void BuildNURBSFaceToDofTable() const;

   /// Sets @a all2local. See documentation of @a all2local for details.
   void SetVarOrderLocalDofs();

   /// Return the minimum order (least significant bit set) in the bit mask.
   static int MinOrder(VarOrderBits bits);

   /// Return element order: internal version of GetElementOrder without checks.
   int GetElementOrderImpl(int i) const;

   /// Returns true if the space is H1 and has variable-order elements.
   bool IsVariableOrderH1() const
   {
      return variableOrder &&
             dynamic_cast<const H1_FECollection*>(fec);
   }

   /** In a variable-order space, calculate a bitmask of polynomial orders that
       need to be represented on each edge and face. */
   void CalcEdgeFaceVarOrders(
      Array<VarOrderBits> &edge_orders, Array<VarOrderBits> &face_orders,
      Array<VarOrderBits> &edge_elem_orders,
      Array<VarOrderBits> &face_elem_orders,
      Array<bool> &skip_edges, Array<bool> &skip_faces) const;

   /// Helper function for ParFiniteElementSpace.
   virtual void ApplyGhostElementOrdersToEdgesAndFaces(
      Array<VarOrderBits> &edge_orders, Array<VarOrderBits> &face_orders) const;

   /// Helper function for ParFiniteElementSpace.
   virtual void GhostFaceOrderToEdges(
      const Array<VarOrderBits> &face_orders,
      Array<VarOrderBits> &edge_orders) const { }

   /// Returns true if order propagation is done, for variable-order spaces.
   virtual bool OrderPropagation(const std::set<int> &edges,
                                 const std::set<int> &faces,
                                 Array<VarOrderBits> &edge_orders,
                                 Array<VarOrderBits> &face_orders) const
   { return edges.size() == 0 && faces.size() == 0; };

   /// Returns the number of ghost edges (nonzero in ParFiniteElementSpace).
   virtual int NumGhostEdges() const { return 0; }

   /// Returns the number of ghost faces (nonzero in ParFiniteElementSpace).
   virtual int NumGhostFaces() const { return 0; }

   /** Build the table var_edge_dofs (or var_face_dofs) in a variable-order
        space; return total edge/face DOFs. */
   int MakeDofTable(int ent_dim, const Array<VarOrderBits> &entity_orders,
                    Table &entity_dofs, Array<char> *var_ent_order);

   /// Search row of a DOF table for a DOF set of size 'ndof', return first DOF.
   int FindDofs(const Table &var_dof_table, int row, int ndof) const;

   /** In a variable-order space, return edge DOFs associated with a polynomial
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

   /// Helper to get vertex, edge or face DOFs (entity=0,1,2 resp.).
   int GetEntityDofs(int entity, int index, Array<int> &dofs,
                     Geometry::Type master_geom = Geometry::INVALID,
                     int variant = 0) const;
   /// Helper to get vertex, edge or face VDOFs (entity=0,1,2 resp.).
   int GetEntityVDofs(int entity, int index, Array<int> &dofs,
                      Geometry::Type master_geom = Geometry::INVALID,
                      int variant = 0) const;

   // Get degenerate face DOFs: see explanation in method implementation.
   int GetDegenerateFaceDofs(int index, Array<int> &dofs,
                             Geometry::Type master_geom, int variant) const;

   int GetNumBorderDofs(Geometry::Type geom, int order) const;

   /// Calculate the cP and cR matrices for a nonconforming mesh.
   void BuildConformingInterpolation() const;

   /** In variable-order spaces, enforce the minimum order rule on edges and
       faces, by adding constraints to @a deps for high-order DOFs to
       interpolate the lowest-order DOFs per mesh entity. */
   void VariableOrderMinimumRule(SparseMatrix & deps) const;

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
      Table* old_elem_fos; // Owned.

      Array<StatelessDofTransformation*> old_DoFTransArray;
      mutable DofTransformation old_DoFTrans;

      void ConstructDoFTransArray();

   public:
      /** Construct the operator based on the elem_dof table of the original
          (coarse) space. The class takes ownership of the table. */
      RefinementOperator(const FiniteElementSpace* fespace,
                         Table *old_elem_dof/*takes ownership*/,
                         Table *old_elem_fos/*takes ownership*/, int old_ndofs);
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
      // Table *coarse_elem_fos; // Owned.
      Table coarse_to_fine;
      Array<int> coarse_to_ref_type;
      Array<Geometry::Type> ref_type_to_geom;
      Array<int> ref_type_to_fine_elem_offset;

   public:
      DerefinementOperator(const FiniteElementSpace *f_fes,
                           const FiniteElementSpace *c_fes,
                           BilinearFormIntegrator *mass_integ);
      void Mult(const Vector &x, Vector &y) const override;
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
                                       const Table *coarse_elem_fos,
                                       const DenseTensor localP[]) const;

   /* This method returns the Refinement matrix (i.e., the embedding)
      from a coarse variable-order fes to a fine fes (after a geometric refinement) */
   SparseMatrix *VariableOrderRefinementMatrix(const int coarse_ndofs,
                                               const Table &coarse_elem_dof) const;

   void GetLocalRefinementMatrices(Geometry::Type geom,
                                   DenseTensor &localP) const;
   void GetLocalDerefinementMatrices(Geometry::Type geom,
                                     DenseTensor &localR) const;

   /** Calculate explicit GridFunction interpolation matrix (after mesh
       refinement). NOTE: consider using the RefinementOperator class instead
       of the fully assembled matrix, which can take a lot of memory. */
   SparseMatrix* RefinementMatrix(int old_ndofs, const Table* old_elem_dof,
                                  const Table* old_elem_fos);

   /// Calculate GridFunction restriction matrix after mesh derefinement.
   SparseMatrix* DerefinementMatrix(int old_ndofs, const Table* old_elem_dof,
                                    const Table* old_elem_fos);

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
       the FiniteElementCollection, and some derived data. */
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
                      int vdim = 1, int ordering = Ordering::byNODES);

   /// Construct a NURBS FE space based on the given NURBSExtension, @a ext.
   /** @note If the pointer @a ext is NULL, this constructor is equivalent to
       the standard constructor with the same arguments minus the
       NURBSExtension, @a ext. */
   FiniteElementSpace(Mesh *mesh, NURBSExtension *ext,
                      const FiniteElementCollection *fec,
                      int vdim = 1, int ordering = Ordering::byNODES);

   /// Copy assignment not supported
   FiniteElementSpace& operator=(const FiniteElementSpace&) = delete;

   /// Returns the mesh
   inline Mesh *GetMesh() const { return mesh; }

   const NURBSExtension *GetNURBSext() const { return NURBSext; }
   NURBSExtension *GetNURBSext() { return NURBSext; }
   NURBSExtension *StealNURBSext();

   bool Conforming() const
   {
      return NURBSext != NULL ||
             (mesh->Conforming() && cP == NULL);
   }
   bool Nonconforming() const { return !Conforming(); }

   /** Set the prolongation operator of the space to an arbitrary sparse matrix,
       creating a copy of the argument. */
   void SetProlongation(const SparseMatrix& p);

   /** Set the restriction operator of the space to an arbitrary sparse matrix,
       creating a copy of the argument. */
   void SetRestriction(const SparseMatrix& r);

   /// Sets the order of the i'th finite element.
   /** By default, all elements are assumed to be of fec->GetOrder(). Once
       SetElementOrder is called, the space becomes a variable-order space. */
   void SetElementOrder(int i, int p);

   /// Returns the order of the i'th finite element.
   int GetElementOrder(int i) const;

   /// Return the maximum polynomial order over all elements.
   virtual int GetMaxElementOrder() const
   { return IsVariableOrder() ? elem_order.Max() : fec->GetOrder(); }

   /// Returns true if the space contains elements of varying polynomial orders.
   bool IsVariableOrder() const { return variableOrder; }

   /// The returned SparseMatrix is owned by the FiniteElementSpace. The method
   /// returns nullptr if the matrix is identity.
   const SparseMatrix *GetConformingProlongation() const;

   /// The returned SparseMatrix is owned by the FiniteElementSpace.
   const SparseMatrix *GetConformingRestriction() const;

   /** Return a version of the conforming restriction matrix for variable-order
       spaces with complex hp interfaces, where some true DOFs are not owned by
       any elements and need to be interpolated from higher order edge/face
       variants (see also @a SetRelaxedHpConformity()). */
   /// The returned SparseMatrix is owned by the FiniteElementSpace.
   const SparseMatrix *GetHpConformingRestriction() const;

   /// The returned Operator is owned by the FiniteElementSpace. The method
   /// returns nullptr if the prolongation matrix is identity.
   virtual const Operator *GetProlongationMatrix() const
   { return GetConformingProlongation(); }

   /// Return an operator that performs the transpose of GetRestrictionOperator
   /** The returned operator is owned by the FiniteElementSpace.

       For a serial conforming space, this returns NULL, indicating the identity
       operator.

       For a parallel conforming space, this will return a matrix-free
       (Device)ConformingProlongationOperator.

       For a non-conforming mesh this will return a TransposeOperator wrapping
       the restriction matrix. */
   const Operator *GetRestrictionTransposeOperator() const;

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
       should be ordered in the E-vector, see ElementDofOrdering.

       For discontinuous spaces, the element restriction corresponds to a
       permutation of the degrees of freedom, implemented by the
       L2ElementRestriction class.

       The returned Operator is owned by the FiniteElementSpace. */
   const ElementRestrictionOperator *GetElementRestriction(
      ElementDofOrdering e_ordering) const;

   /** @brief Return an Operator that converts L-vectors to E-vectors on each
       face. */
   /** @warning only meshes with tensor-product elements are currently
       supported. */
   virtual const FaceRestriction *GetFaceRestriction(
      ElementDofOrdering f_ordering, FaceType,
      L2FaceValues mul = L2FaceValues::DoubleValued) const;

   const InterpolationManager &GetInterpolationManager(
      ElementDofOrdering f_ordering, FaceType type) const;

   /** @brief Return a QuadratureInterpolator that interpolates E-vectors to
       quadrature point values and/or derivatives (Q-vectors). */
   /** An E-vector represents the element-wise discontinuous version of the FE
       space and can be obtained, for example, from a GridFunction using the
       Operator returned by GetElementRestriction().

       All elements will use the same IntegrationRule, @a ir as the target
       quadrature points.

       @note The returned pointer is shared. A good practice, before using it,
       is to set all its properties to their expected values, as other parts of
       the code may also change them. That is, it's good to call
       SetOutputLayout() and DisableTensorProducts() before interpolating.

       @note If the space is not supported by QuadratureInterpolator, nullptr is
       returned. */
   const QuadratureInterpolator *GetQuadratureInterpolator(
      const IntegrationRule &ir) const;

   /** @brief Return a QuadratureInterpolator that interpolates E-vectors to
       quadrature point values and/or derivatives (Q-vectors). */
   /** An E-vector represents the element-wise discontinuous version of the FE
       space and can be obtained, for example, from a GridFunction using the
       Operator returned by GetElementRestriction().

       The target quadrature points in the elements are described by the given
       QuadratureSpace, @a qs.

       @note The returned pointer is shared. A good practice, before using it,
       is to set all its properties to their expected values, as other parts of
       the code may also change them. That is, it's good to call
       SetOutputLayout() and DisableTensorProducts() before interpolating.

       @note If the space is not supported by QuadratureInterpolator, nullptr is
       returned. */
   const QuadratureInterpolator *GetQuadratureInterpolator(
      const QuadratureSpace &qs) const;

   /** @brief Return a FaceQuadratureInterpolator that interpolates E-vectors to
       quadrature point values and/or derivatives (Q-vectors).

       @note The returned pointer is shared. A good practice, before using it,
       is to set all its properties to their expected values, as other parts of
       the code may also change them. That is, it's good to call
       SetOutputLayout() and DisableTensorProducts() before interpolating.

       @note If the space is not supported by FaceQuadratureInterpolator,
       nullptr is returned. */
   const FaceQuadratureInterpolator *GetFaceQuadratureInterpolator(
      const IntegrationRule &ir, FaceType type) const;

   /// Returns the polynomial degree of the i'th finite element.
   /** NOTE: it is recommended to use GetElementOrder in new code. */
   int GetOrder(int i) const { return GetElementOrder(i); }

   /** Return the order of an edge. In a variable-order space, return the order
       of a specific variant, or -1 if there are no more variants. */
   int GetEdgeOrder(int edge, int variant = 0) const;

   /// Returns the polynomial degree of the i'th face finite element
   int GetFaceOrder(int face, int variant = 0) const;

   /// Returns the vector dimension of the finite element space.
   /** Since the finite elements could be vector-valued, this may not be the
       dimension of an actual vector in the space; see GetVectorDim(). */
   inline int GetVDim() const { return vdim; }

   /// @brief Returns number of degrees of freedom.
   /// This is the number of @ref ldof "Local Degrees of Freedom"
   inline int GetNDofs() const { return ndofs; }

   /// @brief Return the number of vector dofs, i.e. GetNDofs() x GetVDim().
   inline int GetVSize() const { return vdim * ndofs; }

   /// @brief Return the number of vector true (conforming) dofs.
   virtual int GetTrueVSize() const { return GetConformingVSize(); }

   /// Returns the number of conforming ("true") degrees of freedom
   /// (if the space is on a nonconforming mesh with hanging nodes).
   int GetNConformingDofs() const;

   int GetConformingVSize() const { return vdim * GetNConformingDofs(); }

   /// Return the total dimension of a vector in the space
   /** This accounts for the vectorization of elements and cases where the
       elements themselves are vector-valued; see FiniteElement:GetRangeDim().
       If the finite elements are FiniteElement::SCALAR, this equals GetVDim().

       Note: For vector-valued elements, the results pads up the range dimension
       to the spatial dimension. E.g., consider a stack of 5 vector-valued
       elements each representing 2D vectors, living in a 3 dimensional space.
       Then this fucntion would give 15, not 10.
       */
   int GetVectorDim() const;

   /// Return the dimension of the curl of a GridFunction defined on this space.
   /** Note: This assumes a space dimension of 2 or 3 only. */
   int GetCurlDim() const;

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
   /// @note The returned pointer references an object owned by the associated
   /// @a Mesh that will be modified by other calls to `GetElementTransformation`.
   /// As such, this pointer should @b not be deleted by the caller.
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

   /// @anchor getdof @name Local DoF Access Members
   /// These member functions produce arrays of local degree of freedom
   /// indices, see @ref ldof. If @b vdim == 1 these indices can be used to
   /// access entries in GridFunction, LinearForm, and BilinearForm objects.
   /// If @b vdim != 1 the corresponding @ref getvdof "Get*VDofs" methods
   /// should be used instead or one of the @ref dof2vdof "DofToVDof" methods
   /// could be used to produce the appropriate offsets from these local dofs.
   ///@{

   /// @brief Returns indices of degrees of freedom of element 'elem'. The
   /// returned indices are offsets into an @ref ldof vector. See also
   /// GetElementVDofs().
   ///
   /// @note In many cases the returned DofTransformation object will be NULL.
   /// In other cases see the documentation of the DofTransformation class for
   /// guidance on its role in performing @ref edof to @ref ldof transformations
   /// on local vectors and matrices. At present the DofTransformation is only
   /// needed for Nedelec basis functions of order 2 and above on 3D elements
   /// with triangular faces.
   ///
   /// @deprecated Use of the returned object is deprecated. The returned object
   /// should @b not be deleted by the caller. If the DofTransformation is
   /// needed, use GetElementDofs(int, Array<int> &, DofTransformation &)
   /// instead.
   DofTransformation *GetElementDofs(int elem, Array<int> &dofs) const;

   /// @brief The same as GetElementDofs(), but with a user-provided
   /// DofTransformation object.
   ///
   /// The user can use DofTransformation::IsIdentity on the returned @a
   /// doftrans object to determine if the DofTransformation needs to actually
   /// be used.
   virtual void GetElementDofs(int elem, Array<int> &dofs,
                               DofTransformation &doftrans) const;

   /// @brief Returns indices of degrees of freedom for boundary element 'bel'.
   /// The returned indices are offsets into an @ref ldof vector. See also
   /// GetBdrElementVDofs().
   ///
   /// @note In many cases the returned DofTransformation object will be NULL.
   /// In other cases see the documentation of the DofTransformation class for
   /// guidance on its role in performing @ref edof to @ref ldof transformations
   /// on local vectors and matrices. At present the DofTransformation is only
   /// needed for Nedelec basis functions of order 2 and above on 3D elements
   /// with triangular faces.
   ///
   /// @deprecated Use of the returned object is deprecated. The returned object
   /// should @b not be deleted by the caller. If the DofTransformation is
   /// needed, use GetBdrElementDofs(int, Array<int> &, DofTransformation &)
   /// instead.
   DofTransformation *GetBdrElementDofs(int bel, Array<int> &dofs) const;

   /// @brief The same as GetBdrElementDofs(), but with a user-provided
   /// DofTransformation object.
   ///
   /// The user can use DofTransformation::IsIdentity on the returned @a
   /// doftrans object to determine if the DofTransformation needs to actually
   /// be used.
   virtual void GetBdrElementDofs(int bel, Array<int> &dofs,
                                  DofTransformation &doftrans) const;

   /// @brief Returns the indices of the degrees of freedom for the specified
   /// face, including the DOFs for the edges and the vertices of the face.
   ///
   /// In variable-order spaces, multiple variants of DOFs can be returned.
   /// See GetEdgeDofs() for more details.
   /// @return Order of the selected variant, or -1 if there are no more
   /// variants.
   ///
   /// The returned indices are offsets into an @ref ldof vector. See also
   /// GetFaceVDofs().
   virtual int GetFaceDofs(int face, Array<int> &dofs, int variant = 0) const;

   /// @brief Returns the indices of the degrees of freedom for the specified
   /// edge, including the DOFs for the vertices of the edge.
   ///
   /// In variable-order spaces, multiple sets of DOFs may exist on an edge,
   /// corresponding to the different polynomial orders of incident elements.
   /// The 'variant' parameter is the zero-based index of the desired DOF set.
   /// The variants are ordered from lowest polynomial degree to the highest.
   /// @return Order of the selected variant, or -1 if there are no more
   /// variants.
   ///
   /// The returned indices are offsets into an @ref ldof vector. See also
   /// GetEdgeVDofs().
   int GetEdgeDofs(int edge, Array<int> &dofs, int variant = 0) const;

   /// @brief Returns the indices of the degrees of freedom for the specified
   /// vertices.
   ///
   /// The returned indices are offsets into an @ref ldof vector. See also
   /// GetVertexVDofs().
   void GetVertexDofs(int i, Array<int> &dofs) const;

   /// @brief Returns the indices of the degrees of freedom for the interior
   /// of the specified element.
   ///
   /// Specifically this refers to degrees of freedom which are not associated
   /// with the vertices, edges, or faces of the mesh. This method may be
   /// useful in conjunction with schemes which process shared and non-shared
   /// degrees of freedom differently such as static condensation.
   ///
   /// The returned indices are offsets into an @ref ldof vector. See also
   /// GetElementInteriorVDofs().
   void GetElementInteriorDofs(int i, Array<int> &dofs) const;

   /// @brief Returns the number of degrees of freedom associated with the
   /// interior of the specified element.
   ///
   /// See GetElementInteriorDofs() for more information or to obtain the
   /// relevant indices.
   int GetNumElementInteriorDofs(int i) const;

   /// @brief Returns the indices of the degrees of freedom for the interior
   /// of the specified face.
   ///
   /// Specifically this refers to degrees of freedom which are not associated
   /// with the vertices, edges, or cell interiors of the mesh. This method may
   /// be useful in conjunction with schemes which process shared and non-shared
   /// degrees of freedom differently such as static condensation.
   ///
   /// The returned indices are offsets into an @ref ldof vector. See also
   /// GetFaceInteriorVDofs().
   void GetFaceInteriorDofs(int i, Array<int> &dofs) const;

   /// @brief Returns the indices of the degrees of freedom for the interior
   /// of the specified edge.
   ///
   /// The returned indices are offsets into an @ref ldof vector. See also
   /// GetEdgeInteriorVDofs().
   void GetEdgeInteriorDofs(int i, Array<int> &dofs) const;
   ///@}

   /** @brief Returns indices of degrees of freedom for NURBS patch index
    @a patch. Cartesian ordering is used, for the tensor-product degrees of
    freedom. */
   void GetPatchDofs(int patch, Array<int> &dofs) const;

   /// @anchor dof2vdof @name DoF To VDoF Conversion methods
   /// These methods convert between local dof and local vector dof using the
   /// appropriate relationship based on the Ordering::Type defined in this
   /// FiniteElementSpace object.
   ///
   /// These methods assume the index set has a range [0, GetNDofs()) which
   /// will be mapped to the range [0, GetVSize()). This assumption can be
   /// changed in the forward mappings by passing a value for @a ndofs which
   /// differs from that returned by GetNDofs().
   ///
   /// @note These methods, with the exception of VDofToDof(), are designed to
   /// produce the correctly encoded values when dof entries are negative,
   /// see @ref ldof for more on negative dof indices.
   ///
   /// @warning When MFEM_DEBUG is enabled at build time the forward mappings
   /// will verify that each @a dof lies in the proper range. If MFEM_DEBUG is
   /// disabled no range checking is performed.
   ///@{

   /// @brief Returns the indices of all of the VDofs for the specified
   /// dimension 'vd'.
   ///
   /// The @a ndofs parameter can be used to indicate the total number of Dofs
   /// associated with each component of @b vdim. If @a ndofs is -1 (the
   /// default value), then the number of Dofs is determined by the
   /// FiniteElementSpace::GetNDofs().
   ///
   /// @note This method does not resize the @a dofs array. It takes the range
   /// of dofs [0, dofs.Size()) and converts these to @ref vdof "vdofs" and
   /// stores the results in the @a dofs array.
   void GetVDofs(int vd, Array<int> &dofs, int ndofs = -1) const;

   /// @brief Compute the full set of @ref vdof "vdofs" corresponding to each
   /// entry in @a dofs.
   ///
   /// @details Produces a set of @ref vdof "vdofs" of
   /// length @b vdim * dofs.Size() corresponding to the entries contained in
   /// the @a dofs array.
   ///
   /// The @a ndofs parameter can be used to indicate the total number of Dofs
   /// associated with each component of @b vdim. If @a ndofs is -1 (the
   /// default value), then the number of Dofs is <determined by the
   /// FiniteElementSpace::GetNDofs().
   ///
   /// @note The @a dofs array is overwritten and resized to accomodate the
   /// new values.
   void DofsToVDofs(Array<int> &dofs, int ndofs = -1) const;

   /// @brief Compute the set of @ref vdof "vdofs" corresponding to each entry
   /// in @a dofs for the given vector index @a vd.
   ///
   /// The @a ndofs parameter can be used to indicate the total number of Dofs
   /// associated with each component of @b vdim. If @a ndofs is -1 (the
   /// default value), then the number of Dofs is <determined by the
   /// FiniteElementSpace::GetNDofs().
   ///
   /// @note The @a dofs array is overwritten with the new values but its size
   /// will not be altered.
   void DofsToVDofs(int vd, Array<int> &dofs, int ndofs = -1) const;

   /// @brief Compute a single @ref vdof corresponding to the index @a dof and
   /// the vector index @a vd.
   ///
   /// The @a ndofs parameter can be used to indicate the total number of Dofs
   /// associated with each component of @b vdim. If @a ndofs is -1 (the
   /// default value), then the number of Dofs is <determined by the
   /// FiniteElementSpace::GetNDofs().
   int DofToVDof(int dof, int vd, int ndofs = -1) const;

   /// @brief Compute the inverse of the Dof to VDof mapping for a single
   /// index @a vdof.
   ///
   /// @warning This method is only intended for use with positive indices.
   /// Passing a negative value for @a vdof will produce an invalid result.
   int VDofToDof(int vdof) const
   { return (ordering == Ordering::byNODES) ? (vdof%ndofs) : (vdof/vdim); }

   ///@}

   /// @brief Remove the orientation information encoded into an array of dofs
   /// Some basis function types have a relative orientation associated with
   /// degrees of freedom shared between neighboring elements, see @ref ldof
   /// for more information. An orientation mismatch is indicated in the dof
   /// indices by a negative index value. This method replaces such negative
   /// indices with the corresponding positive offsets.
   ///
   /// @note The name of this method reflects the fact that it is most often
   /// applied to sets of @ref vdof "Vector Dofs" but it would work equally
   /// well on sets of @ref ldof "Local Dofs".
   static void AdjustVDofs(Array<int> &vdofs);

   /// Helper to encode a sign flip into a DOF index (for Hcurl/Hdiv shapes).
   static inline int EncodeDof(int entity_base, int idx)
   { return (idx >= 0) ? (entity_base + idx) : (-1-(entity_base + (-1-idx))); }

   /// Helper to return the DOF associated with a sign encoded DOF
   static inline int DecodeDof(int dof)
   { return (dof >= 0) ? dof : (-1 - dof); }

   /// Helper to determine the DOF and sign of a sign encoded DOF
   static inline int DecodeDof(int dof, real_t& sign)
   { return (dof >= 0) ? (sign = 1, dof) : (sign = -1, (-1 - dof)); }

   /// @anchor getvdof @name Local Vector DoF Access Members
   /// These member functions produce arrays of local vector degree of freedom
   /// indices, see @ref ldof and @ref vdof. These indices can be used to
   /// access entries in GridFunction, LinearForm, and BilinearForm objects
   /// regardless of the value of @b vdim.
   /// @{

   /// @brief Returns indices of degrees of freedom for the @a i'th element.
   /// The returned indices are offsets into an @ref ldof vector with @b vdim
   /// not necessarily equal to 1. The returned indices are always ordered
   /// byNODES, irrespective of whether the space is byNODES or byVDIM.
   /// See also GetElementDofs().
   ///
   /// @note In many cases the returned DofTransformation object will be NULL.
   /// In other cases see the documentation of the DofTransformation class for
   /// guidance on its role in performing @ref edof to @ref ldof transformations
   /// on local vectors and matrices. At present the DofTransformation is only
   /// needed for Nedelec basis functions of order 2 and above on 3D elements
   /// with triangular faces.
   ///
   /// @deprecated Use of the returned object is deprecated. The returned object
   /// should @b not be deleted by the caller. If the DofTransformation is
   /// needed, use GetElementVDofs(int, Array<int> &, DofTransformation &)
   /// instead.
   DofTransformation *GetElementVDofs(int i, Array<int> &vdofs) const;

   /// @brief The same as GetElementVDofs(), but with a user-provided
   /// DofTransformation object.
   ///
   /// The user can use DofTransformation::IsIdentity on the returned @a
   /// doftrans object to determine if the DofTransformation needs to actually
   /// be used.
   void GetElementVDofs(int i, Array<int> &vdofs,
                        DofTransformation &doftrans) const;

   /// @brief Returns indices of degrees of freedom for @a i'th boundary
   /// element.
   /// The returned indices are offsets into an @ref ldof vector with @b vdim
   /// not necessarily equal to 1. See also GetBdrElementDofs().
   ///
   /// @note In many cases the returned DofTransformation object will be NULL.
   /// In other cases see the documentation of the DofTransformation class for
   /// guidance on its role in performing @ref edof to @ref ldof transformations
   /// on local vectors and matrices. At present the DofTransformation is only
   /// needed for Nedelec basis functions of order 2 and above on 3D elements
   /// with triangular faces.
   ///
   /// @deprecated Use of the returned object is deprecated. The returned object
   /// should @b not be deleted by the caller. If the DofTransformation is
   /// needed, use GetBdrElementVDofs(int, Array<int> &, DofTransformation &)
   /// instead.
   DofTransformation *GetBdrElementVDofs(int i, Array<int> &vdofs) const;

   /// @brief The same as GetBdrElementVDofs(), but with a user-provided
   /// DofTransformation object.
   ///
   /// The user can use DofTransformation::IsIdentity on the returned @a
   /// doftrans object to determine if the DofTransformation needs to actually
   /// be used.
   void GetBdrElementVDofs(int i, Array<int> &vdofs,
                           DofTransformation &doftrans) const;

   /// Returns indices of degrees of freedom in @a vdofs for NURBS patch @a i.
   void GetPatchVDofs(int i, Array<int> &vdofs) const;

   /// @brief Returns the indices of the degrees of freedom for the specified
   /// face, including the DOFs for the edges and the vertices of the face.
   ///
   /// The returned indices are offsets into an @ref ldof vector with @b vdim
   /// not necessarily equal to 1. See GetFaceDofs() for more information.
   void GetFaceVDofs(int i, Array<int> &vdofs) const;

   /// @brief Returns the indices of the degrees of freedom for the specified
   /// edge, including the DOFs for the vertices of the edge.
   ///
   /// The returned indices are offsets into an @ref ldof vector with @b vdim
   /// not necessarily equal to 1. See GetEdgeDofs() for more information.
   void GetEdgeVDofs(int i, Array<int> &vdofs) const;

   /// @brief Returns the indices of the degrees of freedom for the specified
   /// vertices.
   ///
   /// The returned indices are offsets into an @ref ldof vector with @b vdim
   /// not necessarily equal to 1. See also GetVertexDofs().
   void GetVertexVDofs(int i, Array<int> &vdofs) const;

   /// @brief Returns the indices of the degrees of freedom for the interior
   /// of the specified element.
   ///
   /// The returned indices are offsets into an @ref ldof vector with @b vdim
   /// not necessarily equal to 1. See GetElementInteriorDofs() for more
   /// information.
   void GetElementInteriorVDofs(int i, Array<int> &vdofs) const;

   /// @brief Returns the indices of the degrees of freedom for the interior
   /// of the specified edge.
   ///
   /// The returned indices are offsets into an @ref ldof vector with @b vdim
   /// not necessarily equal to 1. See also GetEdgeInteriorDofs().
   void GetEdgeInteriorVDofs(int i, Array<int> &vdofs) const;
   /// @}

   /// (@deprecated) Use the Update() method if the space or mesh changed.
   MFEM_DEPRECATED void RebuildElementToDofTable();

   /** @brief Reorder the scalar DOFs based on the element ordering.

       The new ordering is constructed as follows: 1) loop over all elements as
       ordered in the Mesh; 2) for each element, assign new indices to all of
       its current DOFs that are still unassigned; the new indices we assign are
       simply the sequence `0,1,2,...`; if there are any signed DOFs their sign
       is preserved. */
   void ReorderElementToDofTable();

   const Table *GetElementToFaceOrientationTable() const { return elem_fos; }

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

   /// Deprecated. This function is not required to be called by the user.
   MFEM_DEPRECATED void BuildDofToArrays() const { BuildDofToArrays_(); }

   /// Return the index of the first element that contains ldof index @a i.
   int GetElementForDof(int i) const { BuildDofToArrays_(); return dof_elem_array[i]; }

   /// Return the dof index within the element from GetElementForDof() for ldof index @a i.
   int GetLocalDofForDof(int i) const { BuildDofToArrays_(); return dof_ldof_array[i]; }

   /// Return the index of the first boundary element that contains ldof index @a i.
   int GetBdrElementForDof(int i) const { BuildDofToBdrArrays(); return dof_bdr_elem_array[i]; }

   /// Return the dof index within the boundary element from GetBdrElementForDof() for ldof index @a i.
   int GetBdrLocalDofForDof(int i) const { BuildDofToBdrArrays(); return dof_bdr_ldof_array[i]; }


   /** @brief Returns pointer to the FiniteElement in the FiniteElementCollection
        associated with i'th element in the mesh object.
        Note: The method has been updated to abort instead of returning NULL for
        an empty partition. */
   virtual const FiniteElement *GetFE(int i) const;

   /** @brief Return GetFE(0) if the local mesh is not empty; otherwise return a
       typical FE based on the Geometry types in the global mesh.

       This method can be used as a replacement for GetFE(0) that will be valid
       even if the local mesh is empty. */
   const FiniteElement *GetTypicalFE() const;

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

   /// @brief Return a "typical" trace element.
   ///
   /// This can be used in situations where the local mesh partition may be
   /// empty.
   const FiniteElement *GetTypicalTraceElement() const;

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
                                     int component = -1) const;

   /** @brief Get a list of all boundary true dofs, @a boundary_dofs. For spaces
       with 'vdim' > 1, the 'component' parameter can be used to restricts the
       marked tDOFs to the specified component. Equivalent to
       FiniteElementSpace::GetEssentialTrueDofs with all boundary attributes
       marked as essential. */
   void GetBoundaryTrueDofs(Array<int> &boundary_dofs, int component = -1);

   /** @brief Mark degrees of freedom associated with exterior faces of the
       mesh. For spaces with 'vdim' > 1, the 'component' parameter can be used
       to restricts the marked vDOFs to the specified component. */
   virtual void GetExteriorVDofs(Array<int> &exterior_vdofs,
                                 int component = -1) const;

   /** @brief Get a list of all true dofs on the exterior of the mesh,
       @a exterior_dofs. For spaces with 'vdim' > 1, the 'component' parameter
       can be used to restricts the marked tDOFs to the specified component. */
   virtual void GetExteriorTrueDofs(Array<int> &exterior_dofs,
                                    int component = -1) const;

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

   /** @brief Reflect changes in the mesh: update number of DOFs, etc. Also,
       calculate GridFunction transformation operator (unless want_transform is
       false). Safe to call multiple times, does nothing if space already up to
       date. */
   virtual void Update(bool want_transform = true);

   /** P-refine and update the space. If @a want_transfer, also maintain the old
       space and a transfer operator accessible by GetPrefUpdateOperator(). */
   virtual void PRefineAndUpdate(const Array<pRefinement> & refs,
                                 bool want_transfer = true);

   /** Return true iff p-refinement is supported in this space. Current support
       is only for L2 or H1 spaces on purely quadrilateral or hexahedral
       meshes. */
   bool PRefinementSupported();

   /// Get the GridFunction update operator.
   const Operator* GetUpdateOperator() { Update(); return Th.Ptr(); }

   /// Return the update operator in the given OperatorHandle, @a T.
   void GetUpdateOperator(OperatorHandle &T) { T = Th; }

   /** Returns @a PTh, the transfer operator from the previous space to the
       current space, after p-refinement. */
   std::shared_ptr<const PRefinementTransferOperator> GetPrefUpdateOperator();

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

   /// Return a flag indicating whether the last update was for p-refinement.
   bool LastUpdatePRef() const { return lastUpdatePRef; }

   /// Return whether or not the space is discontinuous (L2)
   bool IsDGSpace() const
   {
      return dynamic_cast<const L2_FECollection*>(fec) != NULL;
   }

   /** In variable-order spaces on nonconforming (NC) meshes, this function
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

   /** @brief Compute the space's node positions w.r.t. given mesh positions.
       The function uses FiniteElement::GetNodes() to obtain the reference DOF
       positions of each finite element.

       @param[in]  mesh_nodes   Mesh positions. Assumes that it has the same
                                topology & ordering as the mesh of the FE space,
                                i.e, same size as this->GetMesh()->GetNodes().
       @param[out] fes_node_pos Positions of the FE space's nodes.
       @param[in]  fes_nodes_ordering  Ordering of fes_node_pos.     */
   void GetNodePositions(const Vector &mesh_nodes, Vector &fes_node_pos,
                         int fes_nodes_ordering = Ordering::byNODES) const;

   /// Save finite element space to output stream @a out.
   void Save(std::ostream &out) const;

   /** @brief Read a FiniteElementSpace from a stream. The returned
       FiniteElementCollection is owned by the caller. */
   FiniteElementCollection *Load(Mesh *m, std::istream &input);

   virtual ~FiniteElementSpace();
};

/// @brief Return true if the mesh contains only one topology and the elements
/// are tensor elements.
inline bool UsesTensorBasis(const FiniteElementSpace& fes)
{
   Mesh & mesh = *fes.GetMesh();
   const bool mixed = mesh.GetNumGeometries(mesh.Dimension()) > 1;
   return !mixed &&
          dynamic_cast<const mfem::TensorBasisElement *>(
             fes.GetTypicalFE()) != nullptr;
}

/// @brief Return LEXICOGRAPHIC if mesh contains only one topology and the
/// elements are tensor elements, otherwise, return NATIVE.
ElementDofOrdering GetEVectorOrdering(const FiniteElementSpace& fes);

}

#endif
