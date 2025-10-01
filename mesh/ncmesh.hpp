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

#ifndef MFEM_NCMESH
#define MFEM_NCMESH

#include "../config/config.hpp"
#include "../general/hash.hpp"
#include "../general/globals.hpp"
#include "../general/sort_pairs.hpp"
#include "../linalg/densemat.hpp"
#include "element.hpp"
#include "vertex.hpp"
#include "../fem/geom.hpp"

#include <vector>
#include <map>
#include <iostream>
#include <unordered_map>

namespace mfem
{

/** Represents the index of an element to refine, plus a refinement type.
    The refinement type is needed for anisotropic refinement of quads and hexes.
    Bits 0,1 and 2 of 'ref_type' specify whether the element should be split
    in the X, Y and Z directions, respectively (Z is ignored for quads). The
    refinement spacing or scale in each direction is a number in (0,1), with the
    default 0.5 meaning bisection. This linear scale parameter defines the
    position of the refinement between the beginning (0) and end (1) of the
    reference element in each direction. */
struct Refinement
{
   int index; ///< Mesh element number
   enum : char { X = 1, Y = 2, Z = 4, XY = 3, XZ = 5, YZ = 6, XYZ = 7 };
   using ScaledType = std::pair<char, real_t>;
   real_t s[3];  /// Refinement scale in each dimension
   Refinement() = default;
   /// Refinement type XYZ, with scale 0.5.
   Refinement(int index);
   /// Default case of empty list @a refs is XYZ with scale 0.5.
   Refinement(int index, const std::initializer_list<ScaledType> &refs);
   /// Refine element with a single type and scale in all dimensions.
   Refinement(int index, char type, real_t scale = 0.5);
   /// Return the type as char.
   char GetType() const;
   /// Set the element, type, and scale.
   void Set(int element, char type,
            real_t scale = 0.5);  /// Uses @a scale in all dimensions
   /// Set the type and scale, assuming the element is already set.
   void SetType(char type,
                real_t scale = 0.5);  /// Uses @a scale in all dimensions
   /// Set the scale in the directions for the currently set type.
   void SetScaleForType(const real_t *scale);
private :
   void SetScale(const ScaledType &ref);
};

/// Defines the position of a fine element within a coarse element.
struct Embedding
{
   /// Coarse %Element index in the coarse mesh.
   int parent;

   /** The (geom, matrix) pair determines the sub-element transformation for the
       fine element: CoarseFineTransformations::point_matrices[geom](matrix) is
       the point matrix of the region within the coarse element reference
       domain.*/
   unsigned geom : 4;
   unsigned matrix : 27;

   /// For internal use: 0 if regular fine element, 1 if parallel ghost element.
   unsigned ghost : 1;

   Embedding() = default;
   Embedding(int elem, Geometry::Type geom, int matrix = 0, bool ghost = false)
      : parent(elem), geom(geom), matrix(matrix), ghost(ghost) {}
};

/// Defines the coarse-fine transformations of all fine elements.
struct CoarseFineTransformations
{
   /// Fine element positions in their parents.
   Array<Embedding> embeddings;

   /** A "dictionary" of matrices for IsoparametricTransformation. Use
       Embedding::{geom,matrix} to access a fine element point matrix. */
   DenseTensor point_matrices[Geometry::NumGeom];

   /** Invert the 'embeddings' array: create a Table with coarse elements as
       rows and fine elements as columns. If 'want_ghosts' is false, parallel
       ghost fine elements are not included in the table. */
   void MakeCoarseToFineTable(Table &coarse_to_fine,
                              bool want_ghosts = false) const;

   void Clear();
   bool IsInitialized() const;
   long MemoryUsage() const;

   MFEM_DEPRECATED
   void GetCoarseToFineMap(const Mesh &fine_mesh, Table &coarse_to_fine) const
   { MakeCoarseToFineTable(coarse_to_fine, true); (void) fine_mesh; }
};

void Swap(CoarseFineTransformations &a, CoarseFineTransformations &b);

struct MatrixMap; // for internal use

/** @brief For a NURBS mesh with nonconforming patch topology, this struct
    provides a map from hanging vertices in the patch topology to the knotvector
    of a neighboring patch. This facilitates ensuring mesh conformity.
 */
class VertexToKnotSpan
{
public:
   /// Set the spatial dimension and number of vertices.
   void SetSize(int dimension, int numVertices);

   // The following set and get functions are for a single entry in the array of
   // data, for a hanging vertex in the patch topology, with the given 'index'.
   // The vertex index is 'v', parent vertices are 'pv', and knot-span is 'ks'.

   /// Set the data for a vertex in 2D.
   void SetVertex2D(int index, int v, int ks,
                    const std::array<int, 2> &pv);

   /// Set the data for a vertex in 3D.
   void SetVertex3D(int index, int v, const std::array<int, 2> &ks,
                    const std::array<int, 4> &pv);

   /// Set the knot-span index for a vertex in 2D.
   void SetKnotSpan2D(int index, int ks);

   /// Set the knot-span indices for a vertex in 3D.
   void SetKnotSpans3D(int index, const std::array<int, 2> &ks);

   /// Get the data for a vertex in 2D.
   void GetVertex2D(int index, int &v, int &ks,
                    std::array<int, 2> &pv) const;

   /// Get the data for a vertex in 3D.
   void GetVertex3D(int index, int &v, std::array<int, 2> &ks,
                    std::array<int, 4> &pv) const;

   /// Print all the data.
   void Print(std::ostream &os) const;

   /// Return the number of vertices.
   int Size() const { return data.NumRows(); }

   /// Return the vertex pair representing the parent edge (2D) or face (3D).
   std::pair<int, int> GetVertexParentPair(int index) const;

private:
   int dim; /// Spatial dimension
   Array2D<int> data; /// Row-wise data for each vertex.
};

/** @brief A class for non-conforming AMR. The class is not used directly by the
 *  user, rather it is an extension of the Mesh class.
 *
 *  In general, the class is used by MFEM as follows:
 *
 *  1. NCMesh is constructed from elements of an existing Mesh. The elements are
 *     copied and become roots of the refinement hierarchy.
 *
 *  2. Some elements are refined with the Refine() method. Both isotropic and
 *     anisotropic refinements of quads/hexes are supported.
 *
 *  3. A new Mesh is created from NCMesh containing the leaf elements. This new
 *     Mesh may have non-conforming (hanging) edges and faces and is the one
 *     seen by the user.
 *
 *  4. FiniteElementSpace asks NCMesh for a list of conforming, master and slave
 *     edges/faces and creates the conforming interpolation matrix P.
 *
 *  5. A continuous/conforming solution is obtained by solving P'*A*P x = P'*b.
 *
 *  6. Repeat from step 2.
 */
class NCMesh
{
protected:
   NCMesh() = default;
public:
   //// Initialize with elements from an existing Mesh.
   explicit NCMesh(const Mesh *mesh);

   /** Load from a stream. The id header is assumed to have been read already
       from \param[in] input . \param[in] version is 10 for the v1.0 NC format,
       or 1 for the legacy v1.1 format. \param[out] curved is set to 1 if the
       curvature GridFunction follows after mesh data. \param[out] is_nc (again
       treated as a boolean) is set to 0 if the legacy v1.1 format in fact
       defines a conforming mesh. See Mesh::Loader for details. */
   NCMesh(std::istream &input, int version, int &curved, int &is_nc);

   /// Deep copy of another instance.
   NCMesh(const NCMesh &other);

   /// Copy assignment not supported
   NCMesh& operator=(NCMesh&) = delete;

   virtual ~NCMesh();

   /// Return the dimension of the NCMesh.
   int Dimension() const { return Dim; }
   /// Return the space dimension of the NCMesh.
   int SpaceDimension() const { return spaceDim; }

   /// Return the number of vertices in the NCMesh.
   int GetNVertices() const { return NVertices; }
   /// Return the number of edges in the NCMesh.
   int GetNEdges() const { return NEdges; }
   /// Return the number of (2D) faces in the NCMesh.
   int GetNFaces() const { return NFaces; }
   virtual int GetNGhostElements() const { return 0; }

   /** Perform the given batch of refinements. Please note that in the presence
       of anisotropic splits additional refinements may be necessary to keep the
       mesh consistent. However, the function always performs at least the
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
       can be done safely so that the maximum NC level condition is not
       violated. On return, level_ok.Size() == deref_table.Size() and contains
       0/1s. */
   virtual void CheckDerefinementNCLevel(const Table &deref_table,
                                         Array<int> &level_ok, int max_nc_level);

   /** Perform a subset of the possible derefinements (see
       GetDerefinementTable). Note that if anisotropic refinements are present
       in the mesh, some of the derefinements may have to be skipped to preserve
       mesh consistency. */
   virtual void Derefine(const Array<int> &derefs);

   // master/slave lists

   /// Identifies a vertex/edge/face in both Mesh and NCMesh.
   struct MeshId
   {
      int index;   ///< Mesh number
      int element; ///< NCMesh::Element containing this vertex/edge/face
      signed char local; ///< local number within 'element'
      signed char geom;  ///< Geometry::Type (faces only) (char to save RAM)

      Geometry::Type Geom() const { return Geometry::Type(geom); }

      MeshId() = default;
      MeshId(int index, int element, int local, int geom = -1)
         : index(index), element(element), local(local), geom(geom) {}
   };

   /** Nonconforming edge/face that has more than one neighbor. The neighbors
       are stored in NCList::slaves[i], slaves_begin <= i < slaves_end. */
   struct Master : public MeshId
   {
      int slaves_begin, slaves_end; ///< slave faces

      Master() = default;
      Master(int index, int element, int local, int geom, int sb, int se)
         : MeshId(index, element, local, geom)
         , slaves_begin(sb), slaves_end(se) {}
   };

   /// Nonconforming edge/face within a bigger edge/face.
   struct Slave : public MeshId
   {
      int master; ///< master number (in Mesh numbering)
      unsigned matrix : 24;    ///< index into NCList::point_matrices[geom]
      unsigned edge_flags : 8; ///< orientation flags, see OrientedPointMatrix

      Slave() = default;
      Slave(int index, int element, int local, int geom)
         : MeshId(index, element, local, geom)
         , master(-1), matrix(0), edge_flags(0) {}
   };


   /// Lists all edges/faces in the nonconforming mesh.
   struct NCList
   {
      Array<MeshId> conforming; ///< All MeshIds corresponding to conformal faces
      Array<Master> masters; ///< All MeshIds corresponding to master faces
      Array<Slave> slaves; ///< All MeshIds corresponding to slave faces

      /// List of unique point matrices for each slave geometry.
      Array<DenseMatrix*> point_matrices[Geometry::NumGeom];

      void OrientedPointMatrix(const Slave &slave,
                               DenseMatrix &oriented_matrix) const;

      /// Particular MeshId type, used for allowing static casting to the
      /// appropriate child type after searching the NCList. UNRECOGNIZED
      /// denotes that an instance is not known within the NCList, meaning that
      /// it does not play a part in NC mechanics. This can be because the index
      /// did not exist in the original Mesh, or because the entry is a boundary
      /// face, whose NC status is always conforming.
      enum class MeshIdType : char {CONFORMING, MASTER, SLAVE, UNRECOGNIZED};

      /// Helper storing a reference to a MeshId type, and the face type it can
      /// be cast to
      struct MeshIdAndType
      {
         const MeshId * const id; ///< Pointer to a possible MeshId, nullptr if not found
         /// MeshIdType corresponding to the MeshId. UNRECOGNIZED if unfound.
         const MeshIdType type;
      };
      /// Return a mesh id and type for a given nc index.
      MeshIdAndType GetMeshIdAndType(int index) const;

      /// Return a face type for a given nc index.
      MeshIdType GetMeshIdType(int index) const;

      /// Given an index, check if this is a certain face type.
      bool CheckMeshIdType(int index, MeshIdType type) const;

      /// Erase the contents of the conforming, master and slave arrays.
      void Clear();
      /// Whether the NCList is empty.
      bool Empty() const
      {
         return conforming.Size() == 0
                && masters.Size() == 0
                && slaves.Size() == 0;
      }
      /// The total size of the component arrays in the NCList.
      long TotalSize() const
      {
         return conforming.Size() + masters.Size() + slaves.Size();
      }
      /// The memory usage of the three public arrays. Does not account for the
      /// inverse index.
      long MemoryUsage() const;
      ~NCList() { Clear(); }
   private:
      // Check for existence or construct the inv_index list map if necessary.
      // const because only modifies the mutable member inv_index.
      void BuildIndex() const;

      /// A lazily constructed map from index to MeshId. Built whenever
      /// GetMeshIdAndType, GetMeshIdType or CheckMeshIdType is called for the
      /// first time. The MeshIdType is stored with, to enable casting to Slave
      /// or Master elements appropriately.
      mutable std::unordered_map<int, std::pair<MeshIdType, int>> inv_index;
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

   /** Return a list of vertices (in 'conforming'); this function is provided
       for uniformity/completeness. Needed in ParNCMesh/ParFESpace. */
   const NCList& GetVertexList()
   {
      if (vertex_list.Empty()) { BuildVertexList(); }
      return vertex_list;
   }

   /// Return vertex/edge/face list (entity = 0/1/2, respectively).
   const NCList& GetNCList(int entity)
   {
      switch (entity)
      {
         case 0: return GetVertexList();
         case 1: return GetEdgeList();
         default: return GetFaceList();
      }
   }

   const VertexToKnotSpan& GetVertexToKnotSpan() const
   {
      return vertex_to_knotspan;
   }

   /// Remap knot-span indices @a vertex_to_knotspan after refinement.
   void RefineVertexToKnotSpan(const std::vector<Array<int>> &kvf,
                               const Array<KnotVector*> &kvext,
                               std::map<std::pair<int, int>,
                               std::array<int, 2>> &parentToKV);

   // coarse/fine transforms

   /** Remember the current layer of leaf elements before the mesh is refined.
       Needed by GetRefinementTransforms(), must be called before Refine(). */
   void MarkCoarseLevel();

   /** After refinement, calculate the relation of each fine element to its
       parent coarse element. Note that Refine() or LimitNCLevel() can be called
       multiple times between MarkCoarseLevel() and this function. */
   const CoarseFineTransformations& GetRefinementTransforms() const;

   /** After derefinement, calculate the relations of previous fine elements
       (some of which may no longer exist) to the current leaf elements. Unlike
       for refinement, Derefine() may only be called once before this function
       so there is no MarkFineLevel(). */
   const CoarseFineTransformations& GetDerefinementTransforms() const;

   /// Free all internal data created by the above three functions.
   void ClearTransforms();


   // grid ordering

   /** Return a space filling curve for a rectangular grid of elements.
       Implemented is a generalized Hilbert curve for arbitrary grid dimensions.
       If the width is odd, height should be odd too, otherwise one diagonal
       (vertex-neighbor) step cannot be avoided in the curve. Even dimensions
       are recommended. */
   static void GridSfcOrdering2D(int width, int height,
                                 Array<int> &coords);

   /** Return a space filling curve for a 3D rectangular grid of elements. The
       Hilbert-curve-like algorithm works well for even dimensions. For odd
       width/height/depth it tends to produce some diagonal (edge-neighbor)
       steps. Even dimensions are recommended. */
   static void GridSfcOrdering3D(int width, int height, int depth,
                                 Array<int> &coords);


   // utility

   /// Return Mesh vertex indices of an edge identified by 'edge_id'.
   void GetEdgeVertices(const MeshId &edge_id, int vert_index[2],
                        bool oriented = true) const;

   /** Return "NC" orientation of an edge. As opposed to standard Mesh edge
       orientation based on vertex IDs, "NC" edge orientation follows the local
       edge orientation within the element 'edge_id.element' and is thus
       processor independent. TODO: this seems only partially true? */
   int GetEdgeNCOrientation(const MeshId &edge_id) const;

   /** Return Mesh vertex and edge indices of a face identified by 'face_id'.
       The return value is the number of face vertices. */
   int GetFaceVerticesEdges(const MeshId &face_id,
                            int vert_index[4], int edge_index[4],
                            int edge_orientation[4]) const;

   /** Given an edge (by its vertex indices v1 and v2) return the first
       (geometric) parent edge that exists in the Mesh or -1 if there is no such
       parent. */
   int GetEdgeMaster(int v1, int v2) const;

   /** Get a list of vertices (2D/3D), edges (3D) and faces (3D) that coincide
       with boundary elements with the specified attributes (marked in
       'bdr_attr_is_ess'). In 3D this function also reveals "hidden" boundary
       edges. In parallel it helps identifying boundary vertices/edges/faces
       affected by non-local boundary elements. Hidden faces can occur for an
       internal boundary coincident to a processor boundary.
       */

   /**
    * @brief Get a list of vertices (2D/3D), edges (3D) and faces (3D) that
    * coincide with boundary elements with the specified attributes (marked in
    * 'bdr_attr_is_ess').
    *
    * @details In 3D this function also reveals "hidden" boundary edges. In
    * parallel it helps identifying boundary vertices/edges/faces affected by
    * non-local boundary elements. Hidden faces can occur for an internal
    * boundary coincident to a processor boundary.
    *
    * @param bdr_attr_is_ess Indicator if a given attribute is essential.
    * @param bdr_vertices Array of vertices that are essential.
    * @param bdr_edges Array of edges that are essential.
    * @param bdr_faces Array of faces that are essential.
    */
   virtual void GetBoundaryClosure(const Array<int> &bdr_attr_is_ess,
                                   Array<int> &bdr_vertices,
                                   Array<int> &bdr_edges, Array<int> &bdr_faces);

   /// Return element geometry type. @a index is the Mesh element number.
   Geometry::Type GetElementGeometry(int index) const
   { return elements[leaf_elements[index]].Geom(); }

   /// Return face geometry type. @a index is the Mesh face number.
   Geometry::Type GetFaceGeometry(int index) const
   { return Geometry::Type(face_geom[index]); }

   /// Return the number of root elements.
   int GetNumRootElements() { return root_state.Size(); }

   /// Return the distance of leaf @a i from the root.
   int GetElementDepth(int i) const;

   /** Return the size reduction compared to the root element (ignoring local
       stretching and curvature). */
   int GetElementSizeReduction(int i) const;

   /// Return the faces and face attributes of leaf element @a i.
   void GetElementFacesAttributes(int i, Array<int> &faces,
                                  Array<int> &fattr) const;

   /// Set the attribute of leaf element @a i, which is a Mesh element index.
   void SetAttribute(int i, int attr)
   { elements[leaf_elements[i]].attribute = attr; }

   /** I/O: Print the mesh in "MFEM NC mesh v1.0" format. If @a comments is
       non-empty, it will be printed after the first line of the file, and each
       line should begin with '#'. */
   void Print(std::ostream &out, const std::string &comments = "",
              bool nurbs=false) const;

   /// I/O: Return true if the mesh was loaded from the legacy v1.1 format.
   bool IsLegacyLoaded() const { return Legacy; }

   /// I/O: Return a map from old (v1.1) vertex indices to new vertex indices.
   void LegacyToNewVertexOrdering(Array<int> &order) const;

   /// Save memory by releasing all non-essential and cached data.
   virtual void Trim();

   /// Return total number of bytes allocated.
   long MemoryUsage() const;

   int PrintMemoryDetail() const;

   /// Return true for ParNCMesh with more than one MPI process.
   virtual bool IsParallel() const { return false; }

   using RefCoord = std::int64_t;

   static constexpr int MaxElemNodes =
      8;       ///< Number of nodes an element can have
   static constexpr int MaxElemEdges =
      12;      ///< Number of edges an element can have
   static constexpr int MaxElemFaces =
      6;       ///< Number of faces an element can have
   static constexpr int MaxElemChildren =
      10;      ///< Number of children an element can have
   static constexpr int MaxFaceNodes =
      4;      ///< Number of faces an element can have

   /**
    * @brief Given a node index, return the vertex index associated
    *
    * @param node
    * @return int
    */
   int GetNodeVertex(int node) { return nodes[node].vert_index; }

protected: // non-public interface for the Mesh class

   friend class Mesh;

   /// Fill Mesh::{vertices,elements,boundary} for the current finest level.
   void GetMeshComponents(Mesh &mesh) const;

   /** Get edge and face numbering from 'mesh' (i.e., set all Edge::index and
       Face::index) after a new mesh was created from us. */
   void OnMeshUpdated(Mesh *mesh);

   /** Delete top-level vertex coordinates if the Mesh became curved, e.g., by
       calling Mesh::SetCurvature or otherwise setting the Nodes. */
   void MakeTopologyOnly() { coordinates.DeleteAll(); }

protected: // implementation

   int Dim, spaceDim; ///< dimensions of the elements and the vertex coordinates
   int MyRank; ///< used in parallel, or when loading a parallel file in serial
   bool Iso; ///< true if the mesh only contains isotropic refinements
   int Geoms; ///< bit mask of element geometries present, see InitGeomFlags()
   bool Legacy; ///< true if the mesh was loaded from the legacy v1.1 format


   /** A Node can hold a vertex, an edge, or both. Elements directly point to
       their corner nodes, but edge nodes also exist and can be accessed using a
       hash-table given their two end-point node IDs. All nodes can be accessed
       in this way, with the exception of top-level vertex nodes. When an
       element is being refined, the mid-edge nodes are readily available with
       this mechanism. The new elements "sign in" to the nodes by increasing the
       reference counts of their vertices and edges. The parent element "signs
       off" its nodes by decrementing the ref counts. */
   struct Node : public Hashed2
   {
      char vert_refc, edge_refc;
      int vert_index, edge_index;

      Node() : vert_refc(0), edge_refc(0), vert_index(-1), edge_index(-1),
         scale(0.5), scaleSet(false) {}
      ~Node();

      bool HasVertex() const { return vert_refc > 0; }
      bool HasEdge()   const { return edge_refc > 0; }

      // decrease vertex/edge ref count, return false if Node should be deleted
      bool UnrefVertex() { --vert_refc; return vert_refc || edge_refc; }
      bool UnrefEdge()   { --edge_refc; return vert_refc || edge_refc; }

      real_t GetScale() const { return scale; }
      void SetScale(real_t s, bool overwrite = false);

   private:
      real_t scale;  ///< Scale from struct Refinement, default 0.5
      bool scaleSet; ///< Indicates whether scale is set and cannot be changed
#ifdef MFEM_USE_DOUBLE
      static constexpr real_t scaleTol = 1.0e-8; ///< Scale comparison tolerance
#else
      static constexpr real_t scaleTol = 1.0e-5; ///< Scale comparison tolerance
#endif
   };

   /** Similarly to nodes, faces can be accessed by hashing their four vertex
       node IDs. A face knows about the one or two elements that are using it. A
       face that is not on the boundary and only has one element referencing it
       is either a master or a slave face. */
   struct Face : public Hashed4
   {
      int attribute; ///< boundary element attribute, -1 if internal face
      int index;     ///< face number in the Mesh
      int elem[2];   ///< up to 2 elements sharing the face

      Face() : attribute(-1), index(-1) { elem[0] = elem[1] = -1; }

      bool Boundary() const { return attribute >= 0; }
      bool Unused() const { return elem[0] < 0 && elem[1] < 0; }

      // add or remove an element from the 'elem[2]' array
      void RegisterElement(int e);
      void ForgetElement(int e);

      /// Return one of elem[0] or elem[1] and make sure the other is -1.
      int GetSingleElement() const;
      int GetAttribute() const { return attribute; }
   };

   /** This is an element in the refinement hierarchy. Each element has either
       been refined and points to its children, or is a leaf and points to its
       vertex nodes. */
   struct Element
   {
      char geom;     ///< Geometry::Type of the element (char for storage only)
      char ref_type; ///< bit mask of X,Y,Z refinements (bits 0,1,2 respectively)
      char tet_type; ///< tetrahedron split type, currently always 0
      char flag;     ///< generic flag/marker, can be used by algorithms
      int index;     ///< element number in the Mesh, -1 if refined
      int rank;      ///< processor number (ParNCMesh), -1 if undefined/unknown
      int attribute;
      union
      {
         int node[MaxElemNodes];  ///< element corners (if ref_type == 0)
         int child[MaxElemChildren]; ///< 2-10 children (if ref_type != 0)
      };
      int parent; ///< parent element, -1 if this is a root element, -2 if free'd
      Element(Geometry::Type geom, int attr);

      Geometry::Type Geom() const { return Geometry::Type(geom); }
      bool IsLeaf() const { return !ref_type && (parent != -2); }
      int GetAttribute() const { return attribute; }
   };


   // primary data
   HashTable<Node> nodes; // associative container holding all Nodes
   HashTable<Face> faces; // associative container holding all Faces

   bool using_scaling = false; // Whether Node::scale is being used

   BlockArray<Element> elements; // storage for all Elements
   Array<int> free_element_ids;  // unused element ids - indices into 'elements'
public:
   /**
    * @brief The number of Nodes.
    *
    * @return int
    */
   int GetNumNodes() const { return nodes.Size(); }
   /**
    * @brief Access a Node
    *
    * @param i Index of the node
    * @return const Node&
    */
   const Node& GetNode(int i) const {return nodes[i]; }
   /**
    * @brief The number of faces
    *
    * @return int
    */
   int GetNumFaces() const { return faces.Size(); }
   /**
    * @brief Access a Face
    *
    * @param i Index of the face
    * @return const Face&
    */
   const Face& GetFace(int i) const {return faces[i]; }
   /**
    * @brief The number of elements
    *
    * @return int
    */
   int GetNumElements() const { return elements.Size(); }
   /**
    * @brief Access an Element
    *
    * @param i Index of the element
    * @return const Element&
    */
   const Element& GetElement(int i) const { return elements[i]; }

   /**
    * @brief Given a set of nodes defining a face, traverse the nodes structure
    * to find the nodes that make up the parent face and replace the input nodes
    * with the parent nodes. Additionally return the child index that the child
    * face would be, relative to the discovered parent face.
    * @details This method is concerned with the construction of an NCMesh
    * structure for a d-1 manifold of an existing NCMesh. It forms a key element
    * in a leaf -> root traversal of the parent ncmesh elements structure.
    *
    * @param[out] nodes The collection of nodes whose parent we are searching
    * for
    * @return int The child index corresponding to placing the face for the
    * original nodes within the face defined by the returned parent nodes. If
    * child index is -1, then the face is made up of root nodes, and nodes is
    * unchanged.
    */
   int ParentFaceNodes(std::array<int, 4> &nodes) const;

   /**
    * @brief Method for finding the nodes associated to a @a face
    * @return Nodes making up the face
    */
   std::array<int, 4> FindFaceNodes(int face) const;
   std::array<int, 4> FindFaceNodes(const Face &fa) const;
   /**
    * @brief Backwards compatible method for finding the @a node associated to a
    * @a face
    */
   MFEM_DEPRECATED void FindFaceNodes(int face, int node[4]) const;
protected:

   /** Initial traversal state (~ element orientation) for each root element
       NOTE: M = root_state.Size() is the number of root elements. NOTE: the
       first M items of 'elements' is the coarse mesh. */
   Array<int> root_state;

   /** Coordinates of top-level vertices (organized as triples). If empty, the
       Mesh is curved (Nodes != NULL) and NCMesh is topology-only. */
   Array<real_t> coordinates;

   // secondary data
   /** Apart from the primary data structure, which is the element/node/face
       hierarchy, there is secondary data that is derived from the primary data
       and needs to be updated when the primary data changes. Update() takes
       care of that and needs to be called after each refinement and
       derefinement. */
   virtual void Update();

   // set by UpdateLeafElements, UpdateVertices and OnMeshUpdated
   int NElements, NVertices, NEdges, NFaces;

   // NOTE: the serial code understands the bare minimum about ghost elements
   // and other ghost entities in order to be able to load parallel partial
   // meshes
   int NGhostElements, NGhostVertices, NGhostEdges, NGhostFaces;

   Array<int> leaf_elements; ///< finest elements, in Mesh ordering (+ ghosts)
   Array<int> leaf_sfc_index; ///< natural tree ordering of leaf elements
   Array<int> vertex_nodeId; ///< vertex-index to node-id map, see UpdateVertices

   NCList face_list; ///< lazy-initialized list of faces, see GetFaceList
   NCList edge_list; ///< lazy-initialized list of edges, see GetEdgeList
   NCList vertex_list; ///< lazy-initialized list of vertices, see GetVertexList

   Array<int> boundary_faces; ///< subset of all faces, set by BuildFaceList
   Array<char> face_geom; ///< face geometry by face index, set by OnMeshUpdated

   Table element_vertex; ///< leaf-element to vertex table, see FindSetNeighbors

   /// Update the leaf elements indices in leaf_elements
   void UpdateLeafElements();

   /** @brief This method assigns indices to vertices (Node::vert_index) that
       will be seen by the Mesh class and the rest of MFEM.

       We must be careful to:
       1. Stay compatible with the conforming code, which expects top-level
          (original) vertices to be indexed first, otherwise GridFunctions
          defined on a conforming mesh would no longer be valid when the mesh is
          converted to an NC mesh.

       2. Make sure serial NCMesh is compatible with the parallel ParNCMesh, so
          it is possible to read parallel partial solutions in serial code
          (e.g., serial GLVis). This means handling ghost elements, if present.

       3. Assign vertices in a globally consistent order for parallel meshes: if
          two vertices i,j are shared by two ranks r1,r2, and i<j on r1, then
          i<j on r2 as well. This is true for top-level vertices but also for
          the remaining shared vertices thanks to the globally consistent SFC
          ordering of the leaf elements. This property reduces communication and
          simplifies ParNCMesh. */
   void UpdateVertices(); ///< update Vertex::index and vertex_nodeId

   /** Collect the leaf elements in leaf_elements, and the ghost elements in
       ghosts. Compute and set the element indices of @a elements. On quad and
       hex refined elements tries to order leaf elements along a space-filling
       curve according to the given @a state variable. */
   void CollectLeafElements(int elem, int state, Array<int> &ghosts,
                            int &counter);

   /** Try to find a space-filling curve friendly orientation of the root
       elements: set 'root_state' based on the ordering of coarse elements. Note
       that the coarse mesh itself must be ordered as an SFC by e.g.
       Mesh::GetGeckoElementOrdering. */
   void InitRootState(int root_count);

   /** Compute the Geometry::Type present in the root elements (coarse elements)
       and set @a Geoms bitmask accordingly. */
   void InitGeomFlags();

   /// Return true if the mesh contains prism elements.
   bool HavePrisms() const { return Geoms & (1 << Geometry::PRISM); }

   /// Return true if the mesh contains pyramid elements.
   bool HavePyramids() const { return Geoms & (1 << Geometry::PYRAMID); }

   /// Return true if the mesh contains tetrahedral elements.
   bool HaveTets() const   { return Geoms & (1 << Geometry::TETRAHEDRON); }

   /// Return true if the Element @a el is a ghost element.
   bool IsGhost(const Element &el) const { return el.rank != MyRank; }

   // refinement/derefinement

   Array<Refinement> ref_stack; ///< stack of scheduled refinements (temporary)
   HashTable<Node> shadow; ///< temporary storage for reparented nodes
   Array<Triple<int, int, int> > reparents; ///< scheduled node reparents (tmp)
   Array<real_t> reparent_scale;  ///< scale associated with reparents (tmp)

   Table derefinements; ///< possible derefinements, see GetDerefinementTable


   /// Refine the element @a elem with the refinement type @a ref_type.
   void RefineElement(int elem, char ref_type);

   /// Refine one element with type and scale specified by @a ref.
   void RefineElement(const Refinement & ref);

   /// Derefine the element @a elem, does nothing on leaf elements.
   void DerefineElement(int elem);

   /// Helper function to set scale for a node with parents @a p0, @a p1.
   void SetNodeScale(int p0, int p1, real_t scale);

   /// Add an Element @a el to the NCMesh, optimized to reuse freed elements.
   int AddElement(const Element &el)
   {
      if (free_element_ids.Size())
      {
         int idx = free_element_ids.Last();
         free_element_ids.DeleteLast();
         elements[idx] = el;
         return idx;
      }
      return elements.Append(el);
   }
   int AddElement(Geometry::Type geom, int attr) { return AddElement(Element(geom,attr)); }

   // Free the element with index @a id.
   void FreeElement(int id)
   {
      free_element_ids.Append(id);
      elements[id].ref_type = 0;
      elements[id].parent = -2; // mark the element as free
   }

   int NewHexahedron(int n0, int n1, int n2, int n3,
                     int n4, int n5, int n6, int n7, int attr,
                     int fattr0, int fattr1, int fattr2,
                     int fattr3, int fattr4, int fattr5);

   int NewWedge(int n0, int n1, int n2,
                int n3, int n4, int n5, int attr,
                int fattr0, int fattr1,
                int fattr2, int fattr3, int fattr4);

   int NewTetrahedron(int n0, int n1, int n2, int n3, int attr,
                      int fattr0, int fattr1, int fattr2, int fattr3);

   int NewPyramid(int n0, int n1, int n2, int n3, int n4, int attr,
                  int fattr0, int fattr1, int fattr2, int fattr3,
                  int fattr4);

   int NewQuadrilateral(int n0, int n1, int n2, int n3, int attr,
                        int eattr0, int eattr1, int eattr2, int eattr3);

   int NewTriangle(int n0, int n1, int n2,
                   int attr, int eattr0, int eattr1, int eattr2);

   int NewSegment(int n0, int n1, int attr, int vattr1, int vattr2);

   mfem::Element* NewMeshElement(int geom) const;

   /**
    * @brief Given a quad face defined by four vertices, establish which edges
    * of this face have been split, and if so optionally return the mid points
    * of those edges.
    *
    * @param n1 The first node defining the face
    * @param n2 The second node defining the face
    * @param n3 The third node defining the face
    * @param n4 The fourth node defining the face
    * @param s returns the scale of the split
    * @param mid optional return of the edge mid points.
    * @return int 0 -- no split, 1 -- "vertical" split, 2 -- "horizontal" split
    */
   int QuadFaceSplitType(int n1, int n2, int n3, int n4, real_t & s,
                         int mid[5] = NULL /*optional output of mid-edge nodes*/) const;

   /**
    * @brief Given a tri face defined by three vertices, establish whether the
    * edges that make up this face have been split, and if so optionally return
    * the midpoints.
    * @details This is a necessary condition for this face to have been split,
    * but is not sufficient. Consider a triangle attached to three refined
    * triangles, in this scenario all edges can be split but this face not be
    * split. In this case, it is necessary to check if there is a face made up
    * of the returned midpoint nodes.
    *
    * @param n1 The first node defining the face
    * @param n2 The second node defining the face
    * @param n3 The third node defining the face
    * @param mid optional return of the edge mid points.
    * @return true Splits for all edges have been found
    * @return false
    */
   bool TriFaceSplit(int n1, int n2, int n3, int mid[3] = NULL) const;

   /**
    * @brief Determine if a Triangle face is a master face
    * @details This check requires looking for the edges making up the triangle
    * being split, if nodes exist at their midpoints, and there are vertices at
    * them, this implies the face COULD be split. To determine if it is, we then
    * check whether these midpoints have all been connected, this is required to
    * discriminate between an internal master face surrounded by nonconformal
    * refinements and a conformal boundary face surrounded by refinements.
    *
    * @param n1 The first node defining the face
    * @param n2 The second node defining the face
    * @param n3 The third node defining the face
    * @return true The face is a master
    * @return false The face is not a master
    */
   inline bool TriFaceIsMaster(int n1, int n2, int n3) const
   {
      int mid[3];
      return !(!TriFaceSplit(n1, n2, n3, mid) // The edges aren't split
               // OR none of the midpoints are connected.
               || (nodes.FindId(mid[0], mid[1]) < 0 &&
                   nodes.FindId(mid[0], mid[2]) < 0 &&
                   nodes.FindId(mid[1], mid[2]) < 0));
   }

   /**
    * @brief Determine if a Quad face is a master face
    *
    * @param n1 The first node defining the face
    * @param n2 The second node defining the face
    * @param n3 The third node defining the face
    * @param n4 The fourth node defining the face
    * @return true The quad face is a master face
    * @return false The quad face is not a master face
    */
   inline bool QuadFaceIsMaster(int n1, int n2, int n3, int n4) const
   {
      real_t s;
      return QuadFaceSplitType(n1, n2, n3, n4, s) != 0;
   }

   void ForceRefinement(int vn1, int vn2, int vn3, int vn4);

   void FindEdgeElements(int vn1, int vn2, int vn3, int vn4,
                         Array<MeshId> &prisms) const;

   void CheckAnisoPrism(int vn1, int vn2, int vn3, int vn4,
                        const Refinement *refs, int nref);

   void CheckAnisoFace(int vn1, int vn2, int vn3, int vn4,
                       int mid12, int mid34, int level = 0);

   void CheckIsoFace(int vn1, int vn2, int vn3, int vn4,
                     int en1, int en2, int en3, int en4, int midf);

   void ReparentNode(int node, int new_p1, int new_p2, real_t scale);

   int FindMidEdgeNode(int node1, int node2) const;
   int GetMidEdgeNode(int node1, int node2);

   int GetMidFaceNode(int en1, int en2, int en3, int en4);

   /**
    * @brief Add references to all nodes, edges and faces of the element
    *
    * @param elem index into elements
    */
   void ReferenceElement(int elem);
   void UnreferenceElement(int elem, Array<int> &elemFaces);

   Face* GetFace(Element &elem, int face_no);
   void RegisterFaces(int elem, int *fattr = NULL);
   void DeleteUnusedFaces(const Array<int> &elemFaces);

   void CollectDerefinements(int elem, Array<Connection> &list);

   /// Return el.node[index] correctly, even if the element is refined.
   int RetrieveNode(const Element &el, int index);

   /// Extended version of find_node: works if 'el' is refined.
   int FindNodeExt(const Element &el, int node, bool abort = true);


   // face/edge lists

   static int find_node(const Element &el, int node);
   static int find_element_edge(const Element &el, int vn0, int vn1,
                                bool abort = true);
   static int find_local_face(int geom, int a, int b, int c);

   struct Point;
   struct PointMatrix;

   int ReorderFacePointMat(int v0, int v1, int v2, int v3,
                           int elem, const PointMatrix &pm,
                           PointMatrix &reordered) const;

   void TraverseQuadFace(int vn0, int vn1, int vn2, int vn3,
                         const PointMatrix& pm, int level, Face* eface[4],
                         MatrixMap &matrix_map);
   struct TriFaceTraverseResults
   {
      bool unsplit; ///< Whether this face has no further splits.
      bool ghost_neighbor; ///< Whether the face neighbor is a ghost.
   };
   TriFaceTraverseResults TraverseTriFace(int vn0, int vn1, int vn2,
                                          const PointMatrix& pm, int level,
                                          MatrixMap &matrix_map);
   void TraverseTetEdge(int vn0, int vn1, const Point &p0, const Point &p1,
                        MatrixMap &matrix_map);
   void TraverseEdge(int vn0, int vn1, real_t t0, real_t t1, int flags,
                     int level, MatrixMap &matrix_map);

   virtual void BuildFaceList();
   virtual void BuildEdgeList();
   virtual void BuildVertexList();

   virtual void ElementSharesFace(int elem, int local, int face) {} // ParNCMesh
   virtual void ElementSharesEdge(int elem, int local, int enode) {} // ParNCMesh
   virtual void ElementSharesVertex(int elem, int local, int vnode) {} // ParNCMesh

   // neighbors / element_vertex table

   /** Return all vertex-, edge- and face-neighbors of a set of elements. The
       neighbors are returned as a list (neighbors != NULL), as a set
       (neighbor_set != NULL), or both. The sizes of the set arrays must match
       that of leaf_elements. The function is intended to be used for large sets
       of elements and its complexity is linear in the number of leaf elements
       in the mesh. */
   void FindSetNeighbors(const Array<char> &elem_set,
                         Array<int> *neighbors, /* append */
                         Array<char> *neighbor_set = NULL);

   /** Return all vertex-, edge- and face-neighbors of a single element. You can
       limit the number of elements being checked using 'search_set'. The
       complexity of the function is linear in the size of the search set.*/
   void FindNeighbors(int elem,
                      Array<int> &neighbors, /* append */
                      const Array<int> *search_set = NULL);

   /** Expand a set of elements by all vertex-, edge- and face-neighbors. The
       output array 'expanded' will contain all items from 'elems' (provided
       they are in 'search_set') plus their neighbors. The neighbor search can
       be limited to the optional search set. The complexity is linear in the
       sum of the sizes of 'elems' and 'search_set'. */
   void NeighborExpand(const Array<int> &elems,
                       Array<int> &expanded,
                       const Array<int> *search_set = NULL);


   void CollectEdgeVertices(int v0, int v1, Array<int> &indices);
   void CollectTriFaceVertices(int v0, int v1, int v2, Array<int> &indices);
   void CollectQuadFaceVertices(int v0, int v1, int v2, int v3,
                                Array<int> &indices);
   void BuildElementToVertexTable();

   void UpdateElementToVertexTable()
   {
      if (element_vertex.Size() < 0) { BuildElementToVertexTable(); }
   }

   int GetVertexRootCoord(int elem, RefCoord coord[3]) const;
   void CollectIncidentElements(int elem, const RefCoord coord[3],
                                Array<int> &list) const;

   /** Return elements neighboring to a local vertex of element 'elem'. Only
       elements from within the same refinement tree ('cousins') are returned.
       Complexity is proportional to the depth of elem's refinement tree. */
   void FindVertexCousins(int elem, int local, Array<int> &cousins) const;


   // coarse/fine transformations

   struct Point
   {
      int dim;
      real_t coord[3];

      Point() { dim = 0; }

      Point(const Point &) = default;

      Point(real_t x)
      { dim = 1; coord[0] = x; }

      Point(real_t x, real_t y)
      { dim = 2; coord[0] = x; coord[1] = y; }

      Point(real_t x, real_t y, real_t z)
      { dim = 3; coord[0] = x; coord[1] = y; coord[2] = z; }

      Point(const Point& p0, const Point& p1, real_t s = 0.5)
      {
         dim = p0.dim;
         for (int i = 0; i < dim; i++)
         {
            coord[i] = ((1.0 - s) * p0.coord[i]) + (s * p1.coord[i]);
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

   /** @brief The PointMatrix stores the coordinates of the slave face using the
       master face coordinate as reference.

       In 2D, the point matrix has the orientation of the parent edge, so its
       columns need to be flipped when applying it, see
       ApplyLocalSlaveTransformation.

       In 3D, the orientation part of Elem2Inf is encoded in the point matrix.

       The following transformation gives the relation between the reference
       quad face coordinates (xi, eta) in [0,1]^2, and the fine quad face
       coordinates (x, y):
         x = a0*(1-xi)*(1-eta) + a1*xi*(1-eta) + a2*xi*eta + a3*(1-xi)*eta
         y = b0*(1-xi)*(1-eta) + b1*xi*(1-eta) + b2*xi*eta + b3*(1-xi)*eta
   */
   struct PointMatrix
   {
      int np;
      Point points[MaxElemNodes];

      PointMatrix() : np(0) {}

      PointMatrix(const Point& p0, const Point& p1)
      { np = 2; points[0] = p0; points[1] = p1; }

      PointMatrix(const Point& p0, const Point& p1, const Point& p2)
      { np = 3; points[0] = p0; points[1] = p1; points[2] = p2; }

      PointMatrix(const Point& p0, const Point& p1, const Point& p2, const Point& p3)
      { np = 4; points[0] = p0; points[1] = p1; points[2] = p2; points[3] = p3; }

      PointMatrix(const Point& p0, const Point& p1, const Point& p2,
                  const Point& p3, const Point& p4)
      {
         np = 5;
         points[0] = p0; points[1] = p1; points[2] = p2;
         points[3] = p3; points[4] = p4;
      }
      PointMatrix(const Point& p0, const Point& p1, const Point& p2,
                  const Point& p3, const Point& p4, const Point& p5)
      {
         np = 6;
         points[0] = p0; points[1] = p1; points[2] = p2;
         points[3] = p3; points[4] = p4; points[5] = p5;
      }
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

      bool operator==(const PointMatrix &pm) const;

      void GetMatrix(DenseMatrix& point_matrix) const;
   };

   static PointMatrix pm_seg_identity;
   static PointMatrix pm_tri_identity;
   static PointMatrix pm_quad_identity;
   static PointMatrix pm_tet_identity;
   static PointMatrix pm_prism_identity;
   static PointMatrix pm_pyramid_identity;
   static PointMatrix pm_hex_identity;

   static const PointMatrix& GetGeomIdentity(Geometry::Type geom);

   void GetPointMatrix(Geometry::Type geom, const char* ref_path,
                       DenseMatrix& matrix) const;

   using RefPathMap = std::map<std::string, int>;

   void TraverseRefinements(int elem, int coarse_index,
                            std::string &ref_path, RefPathMap &map) const;

   /// storage for data returned by Get[De]RefinementTransforms()
   mutable CoarseFineTransformations transforms;

   /// state of leaf_elements before Refine(), set by MarkCoarseLevel()
   Array<int> coarse_elements;

   void InitDerefTransforms();
   void SetDerefMatrixCodes(int parent, Array<int> &fine_coarse);

   // vertex temporary data, used by GetMeshComponents
   struct TmpVertex
   {
      bool valid, visited;
      real_t pos[3];
      TmpVertex() : valid(false), visited(false) {}
   };

   mutable TmpVertex* tmp_vertex;

   const real_t *CalcVertexPos(int node) const;


   // utility

   int GetEdgeMaster(int node) const;

   /// Return directed scale in (0,1).
   inline real_t GetScale(real_t s, bool reverse) const
   { return reverse ? 1.0 - s : s; }

   /**
    * @brief Return the number of splits of this edge that have occurred in the
    * NCMesh. If zero, this means the segment is not the master of any other
    * segments.
    *
    * @param vn1 The first vertex making up the segment
    * @param vn2 The second vertex making up the segment
    * @return int The depth of splits of this segment that are present in the
    * mesh.
    */
   int EdgeSplitLevel(int vn1, int vn2) const;
   /**
    * @brief Return the number of splits of this triangle that have occurred in
    * the NCMesh. If zero, this means the triangle is neither split, nor the
    * master of a split face.
    *
    * @param vn1 The first vertex making up the triangle
    * @param vn2 The second vertex making up the triangle
    * @param vn3 The third vertex making up the triangle
    * @return int The depth of splits of this triangle that are present in the
    * mesh.
    */
   int TriFaceSplitLevel(int vn1, int vn2, int vn3) const;
   /**
    * @brief Computes the number of horizontal and vertical splits of this quad
    * that have occurred in the NCMesh. If zero, this means the quad is not the
    * master of any other quad.
    *
    * @param vn1 The first vertex making up the quad
    * @param vn2 The second vertex making up the quad
    * @param vn3 The third vertex making up the quad
    * @param vn4 The fourth vertex making up the quad
    * @param h_level The number of "horizontal" splits of the quad
    * @param v_level The number of "vertical" splits of the quad
    */
   void QuadFaceSplitLevel(int vn1, int vn2, int vn3, int vn4,
                           int& h_level, int& v_level) const;
   /**
    * @brief Returns the total number of splits of this quad that have occurred
    * in the NCMesh. If zero, this means the quad is not the master of any other
    * quad.
    * @details This is a convenience wrapper that sums the horizontal and
    * vertical levels from the full method.
    *
    * @param vn1 The first vertex making up the quad
    * @param vn2 The second vertex making up the quad
    * @param vn3 The third vertex making up the quad
    * @param vn4 The fourth vertex making up the quad
    * @return int The depth of splits of this triangle that are present in the
    * mesh. NB: An isotropic refinement has a level of 2, one horizontal split,
    * followed by a vertical split.
    */
   int QuadFaceSplitLevel(int vn1, int vn2, int vn3, int vn4) const;

   void CountSplits(int elem, int splits[3]) const;
   void GetLimitRefinements(Array<Refinement> &refinements, int max_level);

   // Checker helpers

   static void CheckSupportedGeom(Geometry::Type geom)
   {
      MFEM_VERIFY(geom == Geometry::SEGMENT ||
                  geom == Geometry::TRIANGLE || geom == Geometry::SQUARE ||
                  geom == Geometry::CUBE || geom == Geometry::PRISM ||
                  geom == Geometry::PYRAMID || geom == Geometry::TETRAHEDRON,
                  "Element type " << geom << " is not supported by NCMesh.");
   }


   // I/O

   /// Print the "vertex_parents" section of the mesh file.
   int PrintVertexParents(std::ostream *out) const;
   /// Load the vertex parent hierarchy from a mesh file.
   void LoadVertexParents(std::istream &input);

   /// Load VertexToKnotSpan data for the NC patch topology mesh of a 2D or 3D
   /// MFEM NURBS NC-patch mesh.
   void LoadVertexToKnotSpan(std::istream &input);
   void LoadVertexToKnotSpan2D(std::istream &input);
   void LoadVertexToKnotSpan3D(std::istream &input);

   /** Print the "boundary" section of the mesh file. If out == NULL, only
       return the number of boundary elements. */
   int PrintBoundary(std::ostream *out) const;
   /// Load the "boundary" section of the mesh file.
   void LoadBoundary(std::istream &input);

   /// Print the "coordinates" section of the mesh file.
   void PrintCoordinates(std::ostream &out) const;
   /// Load the "coordinates" section of the mesh file.
   void LoadCoordinates(std::istream &input);

   /// Count root elements and initialize root_state.
   void InitRootElements();
   /// Return the index of the last top-level node plus one.
   int CountTopLevelNodes() const;
   /// Return true if all root_states are zero.
   bool ZeroRootStates() const;

   /// Load the element refinement hierarchy from a legacy mesh file.
   void LoadCoarseElements(std::istream &input);
   void CopyElements(int elem, const BlockArray<Element> &tmp_elements);
   /// Load the deprecated MFEM mesh v1.1 format for backward compatibility.
   void LoadLegacyFormat(std::istream &input, int &curved, int &is_nc);

   // geometry

   /// This holds in one place the constants about the geometries we support
   struct GeomInfo
   {
      int nv, ne, nf;   // number of: vertices, edges, faces
      int edges[MaxElemEdges][2]; // edge vertices (up to 12 edges)
      int faces[MaxElemFaces][4];  // face vertices (up to 6 faces)
      int nfv[MaxElemFaces];       // number of face vertices

      bool initialized;
      GeomInfo() : initialized(false) {}
      GeomInfo(Geometry::Type geom) : GeomInfo() { InitGeom(geom); }
      void InitGeom(Geometry::Type geom);
   };

   static GeomInfo GI[Geometry::NumGeom];

   /// This is used for a NURBS mesh with this NCMesh as its patch topology.
   VertexToKnotSpan vertex_to_knotspan;

#ifdef MFEM_DEBUG
public:
   void DebugLeafOrder(std::ostream &out) const;
   void DebugDump(std::ostream &out) const;
#endif

   friend class ParNCMesh; // for ParNCMesh::ElementSet
   friend struct MatrixMap;
   friend struct PointMatrixHash;
   friend class NCSubMesh; // for faces, nodes
   friend class ParNCSubMesh; // for faces, nodes
};

}

#endif
