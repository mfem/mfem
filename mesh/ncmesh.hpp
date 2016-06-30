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

#ifndef MFEM_NCMESH
#define MFEM_NCMESH

#include <vector>
#include <map>
#include <iostream>

#include "../config/config.hpp"
#include "../general/hash.hpp"
#include "../linalg/densemat.hpp"
#include "element.hpp"
#include "vertex.hpp"
#include "../fem/geom.hpp"

namespace mfem
{

/** Represents the index of an element to refine, plus a refinement type.
    The refinement type is needed for anisotropic refinement of quads and hexes.
    Bits 0,1 and 2 of 'ref_type' specify whether the element should be split
    in the X, Y and Z directions, respectively (Z is ignored for quads). */
struct Refinement
{
   int index; ///< Mesh element number
   char ref_type; ///< refinement XYZ bit mask (7 = full isotropic)

   Refinement(int index, int type = 7) : index(index), ref_type(type) {}
};

/// Defines the position of a fine element within a coarse element.
struct Embedding
{
   int parent; ///< element index in the coarse mesh
   int matrix; ///< index into CoarseFineTransformations::point_matrices

   Embedding(int elem, int matrix = 0) : parent(elem), matrix(matrix) {}
};

/// Defines the coarse-fine transformations of all fine elements.
struct CoarseFineTransformations
{
   DenseTensor point_matrices;  ///< matrices for IsoparametricTransformation
   Array<Embedding> embeddings; ///< fine element positions in their parents
};


/** \brief A class for non-conforming AMR on higher-order hexahedral,
 *  quadrilateral or triangular meshes.
 *
 *  The class is used as follows:
 *
 *  1. NCMesh is constructed from elements of an existing Mesh. The elements
 *     are copied and become roots of the refinement hierarchy.
 *
 *  2. Some elements are refined with the Refine() method. Both isotropic and
 *     anisotropic refinements of quads/hexes are supported.
 *
 *  3. A new Mesh is created from NCMesh containing the leaf elements.
 *     This new mesh may have non-conforming (hanging) edges and faces.
 *
 *  4. FiniteElementSpace asks NCMesh for a list of conforming, master and
 *     slave edges/faces and creates the conforming interpolation matrix P.
 *
 *  5. A continuous/conforming solution is obtained by solving P'*A*P x = P'*b.
 *
 *  6. Repeat from step 2.
 */
class NCMesh
{
protected:
   struct Element; // forward

public:

   /** Initialize with elements from 'mesh'. If an already nonconforming mesh
       is being loaded, 'vertex_parents' must point to a stream at the appropriate
       section of the mesh file which contains the vertex hierarchy. */
   NCMesh(const Mesh *mesh, std::istream *vertex_parents = NULL);

   /// Deep copy of 'other'.
   NCMesh(const NCMesh &other);

   virtual ~NCMesh();

   int Dimension() const { return Dim; }
   int SpaceDimension() const { return spaceDim; }

   /** Perform the given batch of refinements. Please note that in the presence
       of anisotropic splits additional refinements may be necessary to keep
       the mesh consistent. However, the function always performs at least the
       requested refinements. */
   virtual void Refine(const Array<Refinement> &refinements);

   /** Check the mesh and potentially refine some elements so that the maximum
       difference of refinement levels between adjacent elements is not greater
       than 'max_nc_level'. */
   virtual void LimitNCLevel(int max_nc_level);

   /** Return a list of derefinement opportunities. Each row of the table
       contains Mesh indices of existing elements that can be derefined to form
       a single new coarse element. Row numbers are then passed to Derefine.
       This function works both in serial and parallel. */
   const Table &GetDerefinementTable();

   /** Check derefinements returned by GetDerefinementTable and mark those that
       can be done safely so that the maximum NC level condition is not violated.
       On return, level_ok.Size() == deref_table.Size() and contains 0/1s. */
   virtual void CheckDerefinementNCLevel(const Table &deref_table,
                                         Array<int> &level_ok, int max_nc_level);

   /** Perform a subset of the possible derefinements (see GetDerefinementTable).
       Note that if anisotropic refinements are present in the mesh, some of the
       derefinements may have to be skipped to preserve mesh consistency. */
   virtual void Derefine(const Array<int> &derefs);


   // master/slave lists

   /// Identifies a vertex/edge/face in both Mesh and NCMesh.
   struct MeshId
   {
      int index; ///< Mesh number
      int local; ///< local number within 'element'
      Element* element; ///< NCMesh::Element containing this vertex/edge/face

      MeshId(int index = -1, Element* element = NULL, int local = -1)
         : index(index), local(local), element(element) {}
   };

   /** Nonconforming edge/face that has more than one neighbor. The neighbors
       are stored in NCList::slaves[i], slaves_begin <= i < slaves_end. */
   struct Master : public MeshId
   {
      int slaves_begin, slaves_end; ///< slave faces

      Master(int index, Element* element, int local, int sb, int se)
         : MeshId(index, element, local), slaves_begin(sb), slaves_end(se) {}
   };

   /** Nonconforming edge/face within a bigger edge/face.
       NOTE: only the 'index' member of MeshId is currently valid for slaves. */
   struct Slave : public MeshId
   {
      int master; ///< master number (in Mesh numbering)
      int edge_flags; ///< edge orientation flags
      DenseMatrix point_matrix; ///< position within the master edge/face

      Slave(int index, Element* element, int local)
         : MeshId(index, element, local), master(-1), edge_flags(0) {}

      /// Return the point matrix oriented according to the master and slave edges
      void OrientedPointMatrix(DenseMatrix &oriented_matrix) const;
   };

   /// Lists all edges/faces in the nonconforming mesh.
   struct NCList
   {
      std::vector<MeshId> conforming;
      std::vector<Master> masters;
      std::vector<Slave> slaves;
      // TODO: switch to Arrays when fixed for non-POD types
      // TODO: make a list of unique slave matrices to save memory (+ time later)

      void Clear() { conforming.clear(); masters.clear(); slaves.clear(); }
      bool Empty() const { return !conforming.size() && !masters.size(); }
      long MemoryUsage() const;
   };

   /// Return the current list of conforming and nonconforming faces.
   const NCList& GetFaceList()
   {
      if (face_list.Empty()) { BuildFaceList(); }
      return face_list;
   }

   /// Return the current list of conforming and nonconforming edges.
   const NCList& GetEdgeList()
   {
      if (edge_list.Empty()) { BuildEdgeList(); }
      return edge_list;
   }


   // coarse/fine transforms

   /** Remember the current layer of leaf elements before the mesh is refined.
       Needed by GetRefinementTransforms(), must be called before Refine(). */
   void MarkCoarseLevel();

   /** After refinement, calculate the relation of each fine element to its
       parent coarse element. Note that Refine() or LimitNCLevel() can be called
       multiple times between MarkCoarseLevel() and this function. */
   const CoarseFineTransformations& GetRefinementTransforms();

   /** After derefinement, calculate the relations of previous fine elements
       (some of which may no longer exist) to the current leaf elements.
       Unlike for refinement, Derefine() may only be called once before this
       function so there is no MarkFineLevel(). */
   const CoarseFineTransformations& GetDerefinementTransforms();

   /// Free all internal data created by the above three functions.
   void ClearTransforms();


   // utility

   /** Given an edge (by its vertex indices v1 and v2) return the first
       (geometric) parent edge that exists in the Mesh or -1 if there is no such
       parent. */
   int GetEdgeMaster(int v1, int v2) const;

   /** Get a list of vertices (2D/3D) and edges (3D) that coincide with boundary
       elements with the specified attributes (marked in 'bdr_attr_is_ess').
       In 3D this function also reveals "hidden" boundary edges. In parallel it
       helps identifying boundary vertices/edges affected by non-local boundary
       elements. */
   virtual void GetBoundaryClosure(const Array<int> &bdr_attr_is_ess,
                                   Array<int> &bdr_vertices,
                                   Array<int> &bdr_edges);

   /// Return the type of elements in the mesh.
   int GetElementGeometry() const { return root_elements[0]->geom; }

   /// Return the distance of leaf 'i' from the root.
   int GetElementDepth(int i) const;

   /// I/O: Print the "vertex_parents" section of the mesh file (ver. >= 1.1).
   void PrintVertexParents(std::ostream &out) const;

   /// I/O: Print the "coarse_elements" section of the mesh file (ver. >= 1.1).
   void PrintCoarseElements(std::ostream &out) const;

   /** I/O: Load the vertex parent hierarchy from a mesh file. NOTE: called
       indirectly through the constructor. */
   void LoadVertexParents(std::istream &input);

   /// I/O: Load the element refinement hierarchy from a mesh file.
   void LoadCoarseElements(std::istream &input);

   /// I/O: Set positions of all vertices (used by mesh loader).
   void SetVertexPositions(const Array<mfem::Vertex> &vertices);

   /// Return total number of bytes allocated.
   long MemoryUsage() const;

   void PrintMemoryDetail() const;


protected: // interface for Mesh to be able to construct itself from NCMesh

   friend class Mesh;

   /// Return the basic Mesh arrays for the current finest level.
   void GetMeshComponents(Array<mfem::Vertex>& vertices,
                          Array<mfem::Element*>& elements,
                          Array<mfem::Element*>& boundary) const;

   /** Get edge and face numbering from 'mesh' (i.e., set all Edge::index and
       Face::index) after a new mesh was created from us. */
   virtual void OnMeshUpdated(Mesh *mesh);


protected: // implementation

   int Dim, spaceDim; ///< dimensions of the elements and the vertex coordinates
   bool Iso; ///< true if the mesh only contains isotropic refinements

   Element* CopyHierarchy(Element* elem);
   void DeleteHierarchy(Element* elem);


   // primary data

   /** We want vertices and edges to autodestruct when elements stop using
       (i.e., referencing) them. This base class does the reference counting. */
   struct RefCount
   {
      int ref_count;

      RefCount() : ref_count(0) {}

      int Ref()
      {
         return ++ref_count;
      }
      int Unref()
      {
         int ret = --ref_count;
         if (!ret) { delete this; }
         return ret;
      }
   };

   /** A vertex in the NC mesh. Elements point to vertices indirectly through
       their Nodes. */
   struct Vertex : public RefCount
   {
      int index;     ///< vertex number in the Mesh
      double pos[3]; ///< 3D position

      Vertex() {}
      Vertex(double x, double y, double z) : index(-1)
      { pos[0] = x, pos[1] = y, pos[2] = z; }
      Vertex(const Vertex& other) { std::memcpy(this, &other, sizeof(*this)); }
   };

   /** An NC mesh edge. Edges don't do much more than just exist. */
   struct Edge : public RefCount
   {
      int attribute; ///< boundary element attribute, -1 if internal edge (2D)
      int index;     ///< edge number in the Mesh

      Edge() : attribute(-1), index(-1) {}
      Edge(const Edge &other) { std::memcpy(this, &other, sizeof(*this)); }

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
      Node(const Node &other);
      ~Node();

      // Bump ref count on a vertex or an edge, or create them. Used when an
      // element starts using a vertex or an edge.
      void RefVertex();
      void RefEdge();

      // Decrement ref on vertex or edge when an element is not using them
      // anymore. The vertex, edge or the whole Node can autodestruct.
      // (The hash-table pointer needs to be known then to remove the node.)
      void UnrefVertex(HashTable<Node>& nodes);
      // Implement as a static method since we check if node == NULL.
      static void UnrefEdge(Node *node, HashTable<Node>& nodes);
   };

   /** Similarly to nodes, faces can be accessed by hashing their four vertex
       node IDs. A face knows about the one or two elements that are using it.
       A face that is not on the boundary and only has one element referencing
       it is either a master or a slave face. */
   struct Face : public RefCount, public Hashed4<Face>
   {
      int attribute;    ///< boundary element attribute, -1 if internal face
      int index;        ///< face number in the Mesh
      Element* elem[2]; ///< up to 2 elements sharing the face

      Face(int id);
      Face(const Face& other);

      bool Boundary() const { return attribute >= 0; }

      // add or remove an element from the 'elem[2]' array
      void RegisterElement(Element* e);
      void ForgetElement(Element* e);

      /// Return one of elem[0] or elem[1] and make sure the other is NULL.
      Element* GetSingleElement() const;

      // overloaded Unref without auto-destruction
      int Unref() { return --ref_count; }
   };

   /** This is an element in the refinement hierarchy. Each element has
       either been refined and points to its children, or is a leaf and points
       to its vertex nodes. */
   struct Element
   {
      char geom;     ///< Geometry::Type of the element
      char ref_type; ///< bit mask of X,Y,Z refinements (bits 0,1,2 respectively)
      char flag;     ///< generic flag/marker, can be used by algorithms
      int index;     ///< element number in the Mesh, -1 if refined
      int rank;      ///< processor number (ParNCMesh)
      int attribute;
      union
      {
         Node* node[8];  ///< element corners (if ref_type == 0)
         Element* child[8]; ///< 2-8 children (if ref_type != 0)
      };
      Element* parent; ///< parent element, NULL if this is a root element

      Element(int geom, int attr);
      Element(const Element& other) { std::memcpy(this, &other, sizeof(*this)); }
   };

   // TODO: use MemAlloc or similar to store Element, Face, Node, etc., replace
   // most pointers with indices (ints) into the allocators to save memory;
   // ref_count and attributes could be chars

   Array<Element*> root_elements; // coarsest mesh, initialized by constructor

   HashTable<Node> nodes; // associative container holding all Nodes
   HashTable<Face> faces; // associative container holding all Faces


   // secondary data

   /** Apart from the primary data structure, which is the element/node/face
       hierarchy, there is secondary data that is derived from the primary
       data and needs to be updated when the primary data changes. Update()
       takes care of that and needs to be called after refinement and
       derefinement. Secondary data includes: leaf_elements, vertex_nodeId,
       face_list, edge_list, and everything in ParNCMesh. */
   virtual void Update();

   Array<Element*> leaf_elements; // finest level, updated by UpdateLeafElements

   Array<int> vertex_nodeId; // vertex-index to node-id map, see UpdateVertices

   NCList face_list; ///< lazy-initialized list of faces, see GetFaceList
   NCList edge_list; ///< lazy-initialized list of edges, see GetEdgeList

   Array<Face*> boundary_faces; ///< subset of all faces, set by BuildFaceList
   Array<Node*> boundary_edges; ///< subset of all edges, set by BuildEdgeList

   Table element_vertex; ///< leaf-element to vertex table, see FindSetNeighbors
   int num_vertices;     ///< width of the table

   virtual void UpdateVertices(); ///< update Vertex::index and vertex_nodeId

   void CollectLeafElements(Element* elem, int state);
   void UpdateLeafElements();

   virtual void AssignLeafIndices();

   virtual bool IsGhost(const Element* elem) const { return false; }
   virtual int GetNumGhosts() const { return 0; }


   // refinement/derefinement

   struct ElemRefType
   {
      Element* elem;
      int ref_type;

      ElemRefType(Element* elem, int type)
         : elem(elem), ref_type(type) {}
   };

   Array<ElemRefType> ref_stack; ///< stack of scheduled refinements (temporary)

   Table derefinements; ///< possible derefinements, see GetDerefinementTable

   void RefineElement(Element* elem, char ref_type);
   void DerefineElement(Element* elem);

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

   mfem::Element* NewMeshElement(int geom) const;

   Node* GetMidEdgeVertex(Node* v1, Node* v2);
   Node* GetMidEdgeVertexSimple(Node* v1, Node* v2);
   Node* GetMidFaceVertex(Node* e1, Node* e2, Node* e3, Node* e4);

   int FaceSplitType(Node* v1, Node* v2, Node* v3, Node* v4,
                     Node* mid[4] = NULL /* optional output of mid-edge nodes*/)
   const;

   void ForceRefinement(Node* v1, Node* v2, Node* v3, Node* v4);

   void CheckAnisoFace(Node* v1, Node* v2, Node* v3, Node* v4,
                       Node* mid12, Node* mid34, int level = 0);

   void CheckIsoFace(Node* v1, Node* v2, Node* v3, Node* v4,
                     Node* e1, Node* e2, Node* e3, Node* e4, Node* midf);

   void RefElementNodes(Element *elem);
   void UnrefElementNodes(Element *elem);
   Face* GetFace(Element* elem, int face_no);
   void RegisterFaces(Element* elem, int *fattr = NULL);

   Node* PeekAltParents(Node* v1, Node* v2);

   bool NodeSetX1(Node* node, Node** n);
   bool NodeSetX2(Node* node, Node** n);
   bool NodeSetY1(Node* node, Node** n);
   bool NodeSetY2(Node* node, Node** n);
   bool NodeSetZ1(Node* node, Node** n);
   bool NodeSetZ2(Node* node, Node** n);

   void CollectDerefinements(Element* elem, Array<Connection> &list);

   // face/edge lists

   static int find_node(Element* elem, Node* node);
   static int find_node(Element* elem, int node_id);
   static int find_element_edge(Element* elem, int v0, int v1);
   static int find_hex_face(int a, int b, int c);

   int ReorderFacePointMat(Node* v0, Node* v1, Node* v2, Node* v3,
                           Element* elem, DenseMatrix& mat) const;
   struct PointMatrix;

   void TraverseFace(Node* v0, Node* v1, Node* v2, Node* v3,
                     const PointMatrix& pm, int level);

   void TraverseEdge(Node* v0, Node* v1, double t0, double t1, int flags,
                     int level);

   virtual void BuildFaceList();
   virtual void BuildEdgeList();

   virtual void ElementSharesEdge(Element* elem, Edge* edge) {} // ParNCMesh
   virtual void ElementSharesFace(Element* elem, Face* face) {} // ParNCMesh


   // neighbors / element_vertex table

   /** Return all vertex-, edge- and face-neighbors of a set of elements.
       The neighbors are returned as a list (neighbors != NULL), as a set
       (neighbor_set != NULL), or both. The sizes of the set arrays must match
       that of leaf_elements. The function is intended to be used for large
       sets of elements and its complexity is linear in the number of leaf
       elements in the mesh. */
   void FindSetNeighbors(const Array<char> &elem_set,
                         Array<Element*> *neighbors, /* append */
                         Array<char> *neighbor_set = NULL);

   /** Return all vertex-, edge- and face-neighbors of a single element.
       You can limit the number of elements being checked using 'search_set'.
       The complexity of the function is linear in the size of the search set.*/
   void FindNeighbors(const Element* elem,
                      Array<Element*> &neighbors, /* append */
                      const Array<Element*> *search_set = NULL);

   /** Expand a set of elements by all vertex-, edge- and face-neighbors.
       The output array 'expanded' will contain all items from 'elements'
       (provided they are in 'search_set') plus their neighbors. The neighbor
       search can be limited to the optional search set. The complexity is
       linear in the sum of the sizes of 'elements' and 'search_set'. */
   void NeighborExpand(const Array<Element*> &elements,
                       Array<Element*> &expanded,
                       const Array<Element*> *search_set = NULL);


   void CollectEdgeVertices(Node *v0, Node *v1, Array<int> &indices);
   void CollectFaceVertices(Node* v0, Node* v1, Node* v2, Node* v3,
                            Array<int> &indices);
   void BuildElementToVertexTable();

   void UpdateElementToVertexTable()
   {
      if (element_vertex.Size() < 0) { BuildElementToVertexTable(); }
   }


   // coarse/fine transformations

   struct Point
   {
      int dim;
      double coord[3];

      Point() { dim = 0; }

      Point(double x, double y)
      { dim = 2; coord[0] = x; coord[1] = y; }

      Point(double x, double y, double z)
      { dim = 3; coord[0] = x; coord[1] = y; coord[2] = z; }

      Point(const Point& p0, const Point& p1)
      {
         dim = p0.dim;
         for (int i = 0; i < dim; i++)
         {
            coord[i] = (p0.coord[i] + p1.coord[i]) * 0.5;
         }
      }

      Point(const Point& p0, const Point& p1, const Point& p2, const Point& p3)
      {
         dim = p0.dim;
         MFEM_ASSERT(p1.dim == dim && p2.dim == dim && p3.dim == dim, "");
         for (int i = 0; i < dim; i++)
         {
            coord[i] = (p0.coord[i] + p1.coord[i] + p2.coord[i] + p3.coord[i])
                       * 0.25;
         }
      }

      Point& operator=(const Point& src)
      {
         dim = src.dim;
         for (int i = 0; i < dim; i++) { coord[i] = src.coord[i]; }
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

   static PointMatrix pm_tri_identity;
   static PointMatrix pm_quad_identity;
   static PointMatrix pm_hex_identity;

   static const PointMatrix& GetGeomIdentity(int geom);

   void GetPointMatrix(int geom, const char* ref_path, DenseMatrix& matrix);

   typedef std::map<std::string, int> RefPathMap;

   void TraverseRefinements(Element* elem, int coarse_index,
                            std::string &ref_path, RefPathMap &map);

   /// storage for data returned by Get[De]RefinementTransforms()
   CoarseFineTransformations transforms;

   /// state of leaf_elements before Refine(), set by MarkCoarseLevel()
   Array<Element*> coarse_elements;

   void InitDerefTransforms();
   void SetDerefMatrixCodes(Element* parent, Array<Element*> &coarse);


   // utility

   Node* GetEdgeMaster(Node* node) const;

   static void find_face_nodes(const Face *face, Node* node[4]);

   int  EdgeSplitLevel(Node* v1, Node* v2) const;
   void FaceSplitLevel(Node* v1, Node* v2, Node* v3, Node* v4,
                       int& h_level, int& v_level) const;

   void CountSplits(Element* elem, int splits[3]) const;
   void GetLimitRefinements(Array<Refinement> &refinements, int max_level);

   int CountElements(Element* elem) const;

   int PrintElements(std::ostream &out, Element* elem, int &coarse_id) const;

   void CountObjects(int &nelem, int &nvert, int &nedges) const;


public: // TODO: maybe make this part of mfem::Geometry?

   /** This holds in one place the constants about the geometries we support
       (triangles, quads, cubes) */
   struct GeomInfo
   {
      int nv, ne, nf, nfv; // number of: vertices, edges, faces, face vertices
      int edges[12][2];    // edge vertices (up to 12 edges)
      int faces[6][4];     // face vertices (up to 6 faces)

      bool initialized;
      GeomInfo() : initialized(false) {}
      void Initialize(const mfem::Element* elem);
   };

   static GeomInfo GI[Geometry::NumGeom];

#ifdef MFEM_DEBUG
public:
   void DebugNeighbors(Array<char> &elem_set);

   /// Print the space-filling curve formed by the sequence of leaf elements.
   void DebugLeafOrder() const;
#endif

   friend class ParNCMesh/*::ElementSet*/;
};

}

#endif
