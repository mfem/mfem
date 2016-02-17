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

#ifndef MFEM_PNCMESH
#define MFEM_PNCMESH

#include "../config/config.hpp"

#ifdef MFEM_USE_MPI

#include <map>
#include <set>

#include "ncmesh.hpp"
#include "../general/communication.hpp"
#include "../general/sort_pairs.hpp"

namespace mfem
{

/** \brief A parallel extension of the NCMesh class.
 *
 *  The basic idea (and assumption) is that all processors share the coarsest
 *  layer ('root_elements'). This has the advantage that refinements can easily
 *  be exchanged between processors when rebalancing. Also individual elements
 *  can be uniquely identified by the index of the root element and a path in
 *  the refinement tree.
 *
 *  Each leaf element is owned by one of the processors (NCMesh::Element::rank).
 *  The underlying NCMesh stores not only elements for the current ('MyRank')
 *  processor, but also a minimal layer of adjacent "ghost" elements owned by
 *  other processors. The ghost layer is synchronized after refinement.
 *
 *  The ghost layer contains all vertex-, edge- and face-neighbors of the
 *  current processor's region. It is used to determine constraining relations
 *  and ownership of DOFs on the processor boundary. Ghost elements are never
 *  seen by the rest of MFEM as they are skipped when a Mesh is created from
 *  the NCMesh.
 *
 *  The processor that owns a vertex, edge or a face (and in turn its DOFs) is
 *  currently defined to be the one with the lowest rank in the group of
 *  processors that share the entity.
 *
 *  Vertices, edges and faces that are not shared by this ('MyRank') processor
 *  are ghosts, and are numbered after all real vertices/edges/faces, i.e.,
 *  they have indices greater than NVertices, NEdges, NFaces, respectively.
 *
 *  A shared vertex/edge/face is identified in an interprocessor message by a
 *  pair of numbers. The first number specifies an element in an ElementSet
 *  (typically sent at the beginning of the message) that contains the v/e/f.
 *  The second number is the local index of the v/e/f in that element.
 *
 *  The interface of ParNCMesh is designed for its main customer, the
 *  ParFiniteElementSpace class, which needs to know everything about the
 *  vertices, edges and faces on the processor boundary.
 *
 */
class ParNCMesh : public NCMesh
{
public:
   ParNCMesh(MPI_Comm comm, const NCMesh& ncmesh);

   /** An override of NCMesh::Refine, which is called eventually, after making
       sure that refinements that occur on the processor boundary are sent to
       the neighbor processors so they can keep their ghost layers up to date.*/
   virtual void Refine(const Array<Refinement> &refinements);

   /// To be implemented.
   virtual void LimitNCLevel(int max_level);

   /** Return a list of vertices shared by this processor and at least one other
       processor. (NOTE: only NCList::conforming will be set.) */
   const NCList& GetSharedVertices()
   {
      if (shared_vertices.Empty()) { BuildSharedVertices(); }
      return shared_vertices;
   }

   /** Return a list of edges shared by this processor and at least one other
       processor. (NOTE: this is a subset of the NCMesh::edge_list; slaves are
       empty.) */
   const NCList& GetSharedEdges()
   {
      if (edge_list.Empty()) { BuildEdgeList(); }
      return shared_edges;
   }

   /** Return a list of faces shared by this processor and another processor.
       (NOTE: this is a subset of NCMesh::face_list; slaves are empty.) */
   const NCList& GetSharedFaces()
   {
      if (face_list.Empty()) { BuildFaceList(); }
      return shared_faces;
   }

   /// Helper to get shared vertices/edges/faces ('type' == 0/1/2 resp.).
   const NCList& GetSharedList(int type)
   {
      switch (type)
      {
         case 0: return GetSharedVertices();
         case 1: return GetSharedEdges();
         default: return GetSharedFaces();
      }
   }

   /// Return (shared) face orientation relative to the owner element.
   int GetFaceOrientation(int index) const
   {
      return face_orient[index];
   }

   /// Return vertex/edge/face ('type' == 0/1/2, resp.) owner.
   int GetOwner(int type, int index) const
   {
      switch (type)
      {
         case 0: return vertex_owner[index];
         case 1: return edge_owner[index];
         default: return face_owner[index];
      }
   }

   /** Return a list of processors sharing a vertex/edge/face
       ('type' == 0/1/2, resp.) and the size of the list. */
   const int* GetGroup(int type, int index, int &size) const
   {
      const Table* table;
      switch (type)
      {
         case 0: table = &vertex_group; break;
         case 1: table = &edge_group; break;
         default: table = &face_group;
      }
      size = table->RowSize(index);
      return table->GetRow(index);
   }

   /** Returns true if 'rank' is in the processor group of a vertex/edge/face
       ('type' == 0/1/2, resp.). */
   bool RankInGroup(int type, int index, int rank) const
   {
      int size;
      const int* group = GetGroup(type, index, size);
      for (int i = 0; i < size; i++)
      {
         if (group[i] == rank) { return true; }
      }
      return false;
   }

   /// Returns true if the specified vertex/edge/face is a ghost.
   bool IsGhost(int type, int index) const
   {
      switch (type)
      {
         case 0: return index >= NVertices;
         case 1: return index >= NEdges;
         default: return index >= NFaces;
      }
   }

   /** Extension of NCMesh::GetBoundaryClosure. Filters out ghost vertices and
       ghost edges from 'bdr_vertices' and 'bdr_edges'. */
   virtual void GetBoundaryClosure(const Array<int> &bdr_attr_is_ess,
                                   Array<int> &bdr_vertices,
                                   Array<int> &bdr_edges);

   /** Extract a debugging Mesh containing all leaf elements, including ghosts.
       The debug mesh will have element attributes set to 1 for real elements
       and to 2 for ghost elements. */
   void GetDebugMesh(Mesh &debug_mesh) const;


protected:
   MPI_Comm MyComm;
   int NRanks, MyRank;

   int NVertices, NGhostVertices;
   int NEdges, NGhostEdges;
   int NFaces, NGhostFaces;
   int NElements, NGhostElements;

   // lists of vertices/edges/faces shared by us and at least one more processor
   NCList shared_vertices;
   NCList shared_edges;
   NCList shared_faces;

   // owner processor for each vertex/edge/face
   Array<int> vertex_owner;
   Array<int> edge_owner;
   Array<int> face_owner;

   // list of processors sharing each vertex/edge/face
   Table vertex_group;
   Table edge_group;
   Table face_group;

   Array<char> face_orient; // see CalcFaceOrientations

   /** Type of each leaf element:
         1 - our element (rank == MyRank),
         3 - our element, and neighbor to the ghost layer,
         2 - ghost layer element (existing element, but rank != MyRank),
         0 - element beyond the ghost layer, may not be a real element.
       See also UpdateLayers. */
   Array<char> element_type;

   Array<Element*> ghost_layer; ///< list of elements whose 'element_type' == 2.

   virtual void Update();

   /// Assigns elements to processors at the initial stage (ParMesh creation).
   int InitialPartition(int index) const
   { return index * NRanks / leaf_elements.Size(); }

   virtual void UpdateVertices();
   virtual void AssignLeafIndices();

   virtual bool IsGhost(const Element* elem) const
   { return elem->rank != MyRank; }

   virtual int GetNumGhosts() const { return NGhostElements; }

   virtual void OnMeshUpdated(Mesh *mesh);

   virtual void BuildEdgeList();
   virtual void BuildFaceList();

   virtual void ElementSharesEdge(Element* elem, Edge* edge);
   virtual void ElementSharesFace(Element* elem, Face* face);

   void BuildSharedVertices();

   void CalcFaceOrientations();

   void UpdateLayers();

   Array<Connection> index_rank; // temporary

   void AddMasterSlaveRanks(int nitems, const NCList& list);
   void MakeShared(const Table &groups, const NCList &list, NCList &shared);

   /** Uniquely encodes a set of elements in the refinement hierarchy of an
       NCMesh. Can be dumped to a stream, sent to another processor, loaded,
       and decoded to identify the same set of elements (refinements) in a
       different but compatible NCMesh. The elements don't have to be leaves,
       but they must represent subtrees of 'ncmesh_roots'. */
   class ElementSet
   {
   public:
      ElementSet(const std::set<Element*> &elements,
                 const Array<Element*> &ncmesh_roots);
      void Dump(std::ostream &os) const;

      ElementSet() {}
      ElementSet(std::istream &is) { Load(is); }
      void Load(std::istream &is);
      void Decode(Array<Element*> &elements,
                  const Array<Element*> &ncmesh_roots) const;

   protected:
      Array<unsigned char> data; ///< encoded refinement (sub-)trees

      bool EncodeTree(Element* elem, const std::set<Element*> &elements);
      void DecodeTree(Element* elem, int &pos, Array<Element*> &elements) const;

      void SetInt(int pos, int value);
      int GetInt(int pos) const;
   };

   /// Write to 'os' a processor-independent encoding of vertex/edge/face IDs.
   void EncodeMeshIds(std::ostream &os, Array<MeshId> ids[], int dim) const;

   /// Read from 'is' a processor-independent encoding of vertex/edge/face IDs.
   void DecodeMeshIds(std::istream &is, Array<MeshId> ids[], int dim,
                      bool decode_indices) const;

   Array<Element*> tmp_neighbors; // temporary used by ElementNeighborProcessors

   /** Return a list of processors that own elements in the immediate
       neighborhood of 'elem' (i.e., vertex, edge and face neighbors),
       and are not 'MyRank'. */
   void ElementNeighborProcessors(Element* elem, Array<int> &ranks);

   /** Get a list of ranks that own elements in the neighborhood of our region.
       NOTE: MyRank is not included. */
   void NeighborProcessors(Array<int> &neighbors);

   /** Traverse the (local) refinement tree and determine which subtrees are
       no longer needed, i.e., their leaves are not owned by us nor are they our
       ghosts. These subtrees are then derefined. */
   void Prune();

   /// Internal. Recursive part of Prune().
   bool PruneTree(Element* elem);


   /** Internal message; used to inform neighbors about refinement of elements
    *  adjacent to the processor boundary. This is used by the neighbors to
    *  update their ghost layers.
    */
   class NeighborRefinementMessage : public VarMessage<289>
   {
   public:
      std::vector<ElemRefType> refinements;

      void AddRefinement(Element* elem, int ref_type)
      { refinements.push_back(ElemRefType(elem, ref_type)); }

      /// Set pointer to ParNCMesh (needed to encode the message).
      void SetNCMesh(ParNCMesh* pncmesh) { this->pncmesh = pncmesh; }

      typedef std::map<int, NeighborRefinementMessage> Map;

   protected:
      ParNCMesh* pncmesh;

      virtual void Encode();
      virtual void Decode();
   };


   friend class ParMesh;
   friend class NeighborDofMessage;
};


class FiniteElementCollection; // needed for edge orientation handling

/** Represents a message about DOF assignment of vertex, edge and face DOFs on
 *  the boundary with another processor. This and other messages in this file
 *  are only exchanged between immediate neighbors. Used by
 *  ParFiniteElementSpace::GetConformingInterpolation().
 */
class NeighborDofMessage : public VarMessage<135>
{
public:
   /// Add vertex/edge/face DOFs to an outgoing message.
   void AddDofs(int type, const NCMesh::MeshId &id, const Array<int> &dofs);

   /** Set pointers to ParNCMesh & FECollection (needed to encode the message),
       set the space size to be sent. */
   void Init(ParNCMesh* pncmesh, const FiniteElementCollection* fec, int ndofs)
   { this->pncmesh = pncmesh; this->fec = fec; this->ndofs = ndofs; }

   /** Get vertex/edge/face DOFs from a received message. 'ndofs' receives
       the remote space size. */
   void GetDofs(int type, const NCMesh::MeshId& id,
                Array<int>& dofs, int &ndofs);

   typedef std::map<int, NeighborDofMessage> Map;

protected:
   typedef std::map<NCMesh::MeshId, std::vector<int> > IdToDofs;
   IdToDofs id_dofs[3];

   ParNCMesh* pncmesh;
   const FiniteElementCollection* fec;
   int ndofs;

   virtual void Encode();
   virtual void Decode();

   void ReorderEdgeDofs(const NCMesh::MeshId &id, std::vector<int> &dofs);
};

/** Used by ParFiniteElementSpace::GetConformingInterpolation() to request
 *  finished non-local rows of the P matrix. This message is only sent once
 *  to each neighbor.
 */
class NeighborRowRequest: public VarMessage<312>
{
public:
   std::set<int> rows;

   void RequestRow(int row) { rows.insert(row); }
   void RemoveRequest(int row) { rows.erase(row); }

   typedef std::map<int, NeighborRowRequest> Map;

protected:
   virtual void Encode();
   virtual void Decode();
};

/** Represents a reply to NeighborRowRequest. The reply contains a batch of
 *  P matrix rows that have been finished by the sending processor. Multiple
 *  batches may be sent depending on the number of iterations of the final part
 *  of the function ParFiniteElementSpace::GetConformingInterpolation(). All
 *  rows that are sent accumulate in the same NeighborRowReply instance.
 */
class NeighborRowReply: public VarMessage<313>
{
public:
   void AddRow(int row, const Array<int> &cols, const Vector &srow);

   bool HaveRow(int row) const { return rows.find(row) != rows.end(); }
   void GetRow(int row, Array<int> &cols, Vector &srow);

   typedef std::map<int, NeighborRowReply> Map;

protected:
   struct Row { std::vector<int> cols; Vector srow; };
   std::map<int, Row> rows;

   virtual void Encode();
   virtual void Decode();
};


// comparison operator so that MeshId can be used as key in std::map
inline bool operator< (const NCMesh::MeshId &a, const NCMesh::MeshId &b)
{
   return a.index < b.index;
}

} // namespace mfem

#endif // MFEM_USE_MPI

#endif // MFEM_PNCMESH
