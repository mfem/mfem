// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.googlecode.com.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#ifndef MFEM_NCMESH
#define MFEM_NCMESH

#include "../config/config.hpp"
#include "../general/hash.hpp"
#include "../linalg/densemat.hpp"
#include "element.hpp"
#include "vertex.hpp"
#include "../fem/geom.hpp"

namespace mfem
{

// TODO: these won't be needed once this module is purely geometric
class SparseMatrix;
class Mesh;
class IsoparametricTransformation;
class FiniteElementSpace;

/** Represents the index of an element to refine, plus a refinement type.
    The refinement type is needed for anisotropic refinement of quads and hexes.
    Bits 0,1 and 2 of 'ref_type' specify whether the element should be split
    in the X, Y and Z directions, respectively (Z is ignored for quads). */
struct Refinement
{
   int index; ///< Mesh element number
   int ref_type; ///< refinement XYZ bit mask (7 = full isotropic)

   Refinement(int index, int type = 7)
      : index(index), ref_type(type) {}
};


/** \brief A class for non-conforming AMR on higher-order hexahedral,
 *  quadrilateral or triangular meshes.
 *
 *  The class is used as follows:
 *
 *  1. NCMesh is constructed from elements of an existing Mesh. The elements
 *     are copied and become the roots of the refinement hierarchy.
 *
 *  2. Some elements are refined with the Refine() method. Both isotropic and
 *     anisotropic refinements of quads/hexes are supported.
 *
 *  3. A new Mesh is created from NCMesh containing the leaf elements.
 *     This new mesh may have non-conforming (hanging) edges and faces.
 *
 *  4. A conforming interpolation matrix is obtained using GetInterpolation().
 *     The matrix can be used to constrain the hanging DOFs so a continous
 *     solution is obtained.
 *
 *  5. Refine some more leaf elements, i.e., repeat from step 2.
 */
class NCMesh
{
public:
   NCMesh(const Mesh *mesh);

   int Dimension() const { return Dim; }

   /** Perform the given batch of refinements. Please note that in the presence
       of anisotropic splits additional refinements may be necessary to keep
       the mesh consistent. However, the function always performas at least the
       requested refinements. */
   void Refine(const Array<Refinement> &refinements);

   /** Derefine -- not implemented yet */
   //void Derefine(Element* elem);

   /** Check mesh and potentially refine some elements so that the maximum level
       of hanging nodes is not greater than 'max_level'. */
   void LimitNCLevel(int max_level);

   /** Calculate the conforming interpolation matrix P that ties slave DOFs to
       independent DOFs. P is rectangular with M rows and N columns, where M
       is the number of DOFs of the nonconforming ('cut') space, and N is the
       number of independent ('true') DOFs. If x is a solution vector containing
       the values of the independent DOFs, Px can be used to obtain the values
       of all DOFs, including the slave DOFs. */
   SparseMatrix* GetInterpolation(FiniteElementSpace* space,
                                  SparseMatrix **cR_ptr = NULL);

   /** Represents the relation of a fine element to its parent (coarse) element
       from a previous mesh state. (Note that the parent can be an indirect
       parent.) The point matrix determines where in the reference domain of the
       coarse element the fine element is located. */
   struct FineTransform
   {
      int coarse_index; ///< coarse Mesh element index
      DenseMatrix point_matrix; ///< for use in IsoparametricTransformation

      /// As an optimization, identity transform is "stored" as empty matrix.
      bool IsIdentity() const { return !point_matrix.Data(); }
   };

   /** Store the current layer of leaf elements before the mesh is refined.
       This is later used by 'GetFineTransforms' to determine the relations of
       the coarse and refined elements. */
   void MarkCoarseLevel() { leaf_elements.Copy(coarse_elements); }

   /// Free the internally stored array of coarse leaf elements.
   void ClearCoarseLevel() { coarse_elements.DeleteAll(); }

   /** Return an array of structures 'FineTransform', one for each leaf
       element. This data can be used to transfer functions from a previous
       coarse level of the mesh (marked with 'MarkCoarseLevel') to a newly
       refined state of the mesh.
       NOTE: the caller needs to free the returned array. */
   FineTransform* GetFineTransforms();

   /** Given an edge (by its vertex indices v1 and v2) return the first
       (geometric) parent edge that exists in the Mesh or -1 if there is no such
       parent. */
   int GetEdgeMaster(int v1, int v2) const;

   /** Return total number of bytes allocated. */
   long MemoryUsage();

   ~NCMesh();


protected: // interface for Mesh to be able to construct itself from us

   void GetVerticesElementsBoundary(Array<mfem::Vertex>& vertices,
                                    Array<mfem::Element*>& elements,
                                    Array<mfem::Element*>& boundary);

   void SetEdgeIndicesFromMesh(Mesh *mesh);
   void SetFaceIndicesFromMesh(Mesh *mesh);

   friend class Mesh;


protected: // implementation

   int Dim;

   /** We want vertices and edges to autodestruct when elements stop using
       (i.e., referencing) them. This base class does the reference counting. */
   struct RefCount
   {
      int ref_count;

      RefCount() : ref_count(0) {}

      int Ref() {
         return ++ref_count;
      }
      int Unref() {
         int ret = --ref_count;
         if (!ret) delete this;
         return ret;
      }
   };

   /** A vertex in the NC mesh. Elements point to vertices indirectly through
       their Nodes. */
   struct Vertex : public RefCount
   {
      double pos[3]; ///< 3D position
      int index;     ///< vertex number in the Mesh

      Vertex() {}
      Vertex(double x, double y, double z) : index(-1)
      { pos[0] = x, pos[1] = y, pos[2] = z; }
   };

   /** An NC mesh edge. Edges don't do much more than just exist. */
   struct Edge : public RefCount
   {
      int attribute; ///< boundary element attribute, -1 if internal edge
      int index;     ///< edge number in the Mesh

      Edge() : attribute(-1), index(-1) {}
      bool Boundary() const { return attribute >= 0; }
   };

   /** A Node can hold a Vertex, an Edge, or both. Elements directly point to
       their corner nodes, but edge nodes also exist and can be accessed using
       a hash-table given their two end-point node IDs. All nodes can be
       accessed in this way, with the exception of top-level vertex nodes.
       When an element is being refined, the mid-edge nodes are readily
       available with this mechanism. The new elements "sign in" into the nodes
       to have vertices and edges created for them or to just have their
       reference counts increased. The parent element "signs off" its nodes,
       which decrements the vertex and edge reference counts. Vertices and edges
       are destroyed when their reference count drops to zero. */
   struct Node : public Hashed2<Node>
   {
      Vertex* vertex;
      Edge* edge;

      Node(int id) : Hashed2<Node>(id), vertex(NULL), edge(NULL) {}

      // Bump ref count on a vertex or an edge, or create them. Used when an
      // element starts using a vertex or an edge.
      void RefVertex();
      void RefEdge();

      // Decrement ref on vertex or edge when an element is not using them
      // anymore. The vertex, edge or the whole Node can autodestruct.
      // (The hash-table pointer needs to be known then to remove the node.)
      void UnrefVertex(HashTable<Node>& nodes);
      void UnrefEdge(HashTable<Node>& nodes);

      ~Node();
   };

   struct Element;

   /** Similarly to nodes, faces can be accessed by hashing their four vertex
       node IDs. A face knows about the one or two elements that are using it.
       A face that is not on the boundary and only has one element referencing
       it is either a master or a slave face. */
   struct Face : public RefCount, public Hashed4<Face>
   {
      int attribute;    ///< boundary element attribute, -1 if internal face
      int index;        ///< face number in the Mesh
      Element* elem[2]; ///< up to 2 elements sharing the face

      Face(int id) : Hashed4<Face>(id), attribute(-1), index(-1)
      { elem[0] = elem[1] = NULL; }

      bool Boundary() const { return attribute >= 0; }

      // add or remove an element from the 'elem[2]' array
      void RegisterElement(Element* e);
      void ForgetElement(Element* e);

      // return one of elem[0] or elem[1] and make sure the other is NULL
      Element* GetSingleElement() const;

      // overloaded Unref without auto-destruction
      int Unref() { return --ref_count; }
   };

   /** This is an element in the refinement hierarchy. Each element has
       either been refined and points to its children, or is a leaf and points
       to its vertex nodes. */
   struct Element
   {
      int geom;     // Geometry::Type of the element
      int attribute;
      int ref_type; // bit mask of X,Y,Z refinements (bits 0,1,2, respectively)
      int index;    // element number in the Mesh, -1 if refined
      union
      {
         Node* node[8];  // element corners (if ref_type == 0)
         Element* child[8]; // 2-8 children (if ref_type != 0)
      };

      Element(int geom, int attr);
   };

   Array<Element*> root_elements; // initialized by constructor
   Array<Element*> leaf_elements; // finest level, updated by UpdateLeafElements
   Array<Element*> coarse_elements; // coarse level, set by MarkCoarseLevel

   Array<int> vertex_nodeId; // vertex-index to node-id map

   HashTable<Node> nodes; // associative container holding all Nodes
   HashTable<Face> faces; // associative container holding all Faces

   struct RefStackItem
   {
      Element* elem;
      int ref_type;

      RefStackItem(Element* elem, int type)
         : elem(elem), ref_type(type) {}
   };

   Array<RefStackItem> ref_stack; ///< stack of scheduled refinements

   void Refine(Element* elem, int ref_type);

   void UpdateVertices(); // update the indices of vertices and vertex_nodeId

   void GetLeafElements(Element* e);
   void UpdateLeafElements();

   void DeleteHierarchy(Element* elem);

   Element* NewHexahedron(Node* n0, Node* n1, Node* n2, Node* n3,
                          Node* n4, Node* n5, Node* n6, Node* n7,
                          int attr,
                          int fattr0, int fattr1, int fattr2,
                          int fattr3, int fattr4, int fattr5);

   Element* NewQuadrilateral(Node* n0, Node* n1, Node* n2, Node* n3,
                             int attr,
                             int eattr0, int eattr1, int eattr2, int eattr3);

   Element* NewTriangle(Node* n0, Node* n1, Node* n2,
                        int attr, int eattr0, int eattr1, int eattr2);

   Vertex* NewVertex(Node* v1, Node* v2);

   Node* GetMidEdgeVertex(Node* v1, Node* v2);
   Node* GetMidEdgeVertexSimple(Node* v1, Node* v2);
   Node* GetMidFaceVertex(Node* e1, Node* e2, Node* e3, Node* e4);

   int FaceSplitType(Node* v1, Node* v2, Node* v3, Node* v4,
                     Node* mid[4] = NULL /* optional output of mid-edge nodes*/);

   void ForceRefinement(Node* v1, Node* v2, Node* v3, Node* v4);

   void CheckAnisoFace(Node* v1, Node* v2, Node* v3, Node* v4,
                       Node* mid12, Node* mid34, int level = 0);

   void CheckIsoFace(Node* v1, Node* v2, Node* v3, Node* v4,
                     Node* e1, Node* e2, Node* e3, Node* e4, Node* midf);

   void RefElementNodes(Element *elem);
   void UnrefElementNodes(Element *elem);
   void RegisterFaces(Element* elem);

   Node* PeekAltParents(Node* v1, Node* v2);

   bool NodeSetX1(Node* node, Node** n);
   bool NodeSetX2(Node* node, Node** n);
   bool NodeSetY1(Node* node, Node** n);
   bool NodeSetY2(Node* node, Node** n);
   bool NodeSetZ1(Node* node, Node** n);
   bool NodeSetZ2(Node* node, Node** n);


   // interpolation

   struct Dependency
   {
      int dof;
      double coef;

      Dependency(int dof, double coef)
         : dof(dof), coef(coef) {}
   };

   typedef Array<Dependency> DepList;

   /** Holds temporary data for each nonconforming (FESpace-assigned) DOF
       during the interpolation algorithm. */
   struct DofData
   {
      bool finalized; ///< true if cP matrix row is known for this DOF
      DepList dep_list; ///< list of other DOFs this DOF depends on

      DofData() : finalized(false) {}
      bool Independent() const { return !dep_list.Size(); }
   };

   DofData* dof_data; ///< DOF temporary data

   FiniteElementSpace* space;

   static int find_node(Element* elem, Node* node);

   void ReorderFacePointMat(Node* v0, Node* v1, Node* v2, Node* v3,
                            Element* elem, DenseMatrix& pm);

   void AddDependencies(Array<int>& master_dofs, Array<int>& slave_dofs,
                        DenseMatrix& I);

   void ConstrainEdge(Node* v0, Node* v1, double t0, double t1,
                      Array<int>& master_dofs, int level);

   struct PointMatrix;

   void ConstrainFace(Node* v0, Node* v1, Node* v2, Node* v3,
                      const PointMatrix &pm,
                      Array<int>& master_dofs, int level);

   void ProcessMasterEdge(Node* node[2], Node* edge);
   void ProcessMasterFace(Node* node[4], Face* face);

   bool DofFinalizable(DofData& vd);


   // coarse to fine transformations

   struct Point
   {
      int dim;
      double coord[3];

      Point()
      { dim = 0; }

      Point(double x, double y)
      { dim = 2; coord[0] = x; coord[1] = y; }

      Point(double x, double y, double z)
      { dim = 3; coord[0] = x; coord[1] = y; coord[2] = z; }

      Point(const Point& p0, const Point& p1)
      {
         dim = p0.dim;
         for (int i = 0; i < dim; i++)
            coord[i] = (p0.coord[i] + p1.coord[i]) * 0.5;
      }

      Point(const Point& p0, const Point& p1, const Point& p2, const Point& p3)
      {
         dim = p0.dim;
         for (int i = 0; i < dim; i++)
            coord[i] = (p0.coord[i] + p1.coord[i] + p2.coord[i] + p3.coord[i])
               * 0.25;
      }

      Point& operator=(const Point& src)
      {
         dim = src.dim;
         for (int i = 0; i < dim; i++)
            coord[i] = src.coord[i];
         return *this;
      }
   };

   struct PointMatrix
   {
      int np;
      Point points[8];

      PointMatrix(const Point& p0, const Point& p1, const Point& p2)
      { np = 3; points[0] = p0; points[1] = p1; points[2] = p2; }

      PointMatrix(const Point& p0, const Point& p1, const Point& p2, const Point& p3)
      { np = 4; points[0] = p0; points[1] = p1; points[2] = p2; points[3] = p3; }

      PointMatrix(const Point& p0, const Point& p1, const Point& p2,
                  const Point& p3, const Point& p4, const Point& p5,
                  const Point& p6, const Point& p7)
      {
         np = 8;
         points[0] = p0; points[1] = p1; points[2] = p2; points[3] = p3;
         points[4] = p4; points[5] = p5; points[6] = p6; points[7] = p7;
      }

      Point& operator()(int i) { return points[i]; }
      const Point& operator()(int i) const { return points[i]; }

      void GetMatrix(DenseMatrix& point_matrix) const;
   };

   void GetFineTransforms(Element* elem, int coarse_index,
                          FineTransform *transforms, const PointMatrix &pm);

   int GetEdgeMaster(Node *n) const;

   // utility

   void FaceSplitLevel(Node* v1, Node* v2, Node* v3, Node* v4,
                       int& h_level, int& v_level);

   void CountSplits(Element* elem, int splits[3]);

   int CountElements(Element* elem);

};

}

#endif
