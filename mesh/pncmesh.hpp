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

namespace mfem
{

class ParMesh;
class FiniteElementCollection; // for edge orientation handling
class FiniteElementSpace; // for Dof -> VDof conversion


/** \brief A parallel extension of the NCMesh class.
 *
 *  The basic idea (and assumption) is that all processors share the coarsest
 *  layer ('root_elements'). This has the advantage that refinements can easily
 *  be exchanged between processors when rebalancing since individual elements
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
 */
class ParNCMesh : public NCMesh
{
public:
   ParNCMesh(MPI_Comm comm, const NCMesh& ncmesh);

   virtual ~ParNCMesh();

   /** An override of NCMesh::Refine, which is called eventually, after making
       sure that refinements that occur on the processor boundary are sent to
       the neighbor processors so they can keep their ghost layers up to date.*/
   virtual void Refine(const Array<Refinement> &refinements);

   /// Parallel version of NCMesh::LimitNCLevel.
   virtual void LimitNCLevel(int max_nc_level);

   /** Parallel version of NCMesh::CheckDerefinementNCLevel. */
   virtual void CheckDerefinementNCLevel(const Table &deref_table,
                                         Array<int> &level_ok, int max_nc_level);

   /** Parallel reimplementation of NCMesh::Derefine, keeps ghost layers
       in sync. The interface is identical. */
   virtual void Derefine(const Array<int> &derefs);

   /** Migrate leaf elements of the global refinement hierarchy (including ghost
       elements) so that each processor owns the same number of leaves (+-1). */
   void Rebalance();


   // interface for ParFiniteElementSpace

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

   /** Returns owner processor for element 'index'. This is normally MyRank but
       for index >= NElements (i.e., for ghosts) it may be something else. */
   int ElementRank(int index) const
   { return leaf_elements[index]->rank; }


   // interface for ParMesh

   /** Populate face neighbor members of ParMesh from the ghost layer, without
       communication. */
   void GetFaceNeighbors(ParMesh &pmesh);


   // utility

   /// Use the communication pattern from last Rebalance() to send element DOFs.
   void SendRebalanceDofs(int old_ndofs, const Table &old_element_dofs,
                          long old_global_offset, FiniteElementSpace* space);

   /// Receive element DOFs sent by SendRebalanceDofs().
   void RecvRebalanceDofs(Array<int> &elements, Array<long> &dofs);

   /** Get previous indices (pre-Rebalance) of current elements. Index of -1
       indicates that an element didn't exist in the mesh before. */
   const Array<int>& GetRebalanceOldIndex() const { return old_index_or_rank; }

   /** Get previous (pre-Derefine) fine element ranks. This complements the
       CoarseFineTransformations::embeddings array in parallel. */
   const Array<int>& GetDerefineOldRanks() const { return old_index_or_rank; }

   /** Exchange element data for derefinements that straddle processor
       boundaries. 'elem_data' is enlarged and filled with ghost values. */
   template<typename Type>
   void SynchronizeDerefinementData(Array<Type> &elem_data,
                                    const Table &deref_table);

   /** Extension of NCMesh::GetBoundaryClosure. Filters out ghost vertices and
       ghost edges from 'bdr_vertices' and 'bdr_edges'. */
   virtual void GetBoundaryClosure(const Array<int> &bdr_attr_is_ess,
                                   Array<int> &bdr_vertices,
                                   Array<int> &bdr_edges);

   /** Extract a debugging Mesh containing all leaf elements, including ghosts.
       The debug mesh will have element attributes set to element rank + 1. */
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
       See also UpdateLayers(). */
   Array<char> element_type;

   Array<Element*> ghost_layer; ///< list of elements whose 'element_type' == 2.
   Array<Element*> boundary_layer; ///< list of type 3 elements

   virtual void Update();

   /// Return the processor number for a global element number.
   int Partition(long index, long total_elements) const
   { return index * NRanks / total_elements; }

   /// Helper to get the partition when the serial mesh is being split initially
   int InitialPartition(int index) const
   { return Partition(index, leaf_elements.Size()); }

   /// Return the global index of the first element owned by processor 'rank'.
   long PartitionFirstIndex(int rank, long total_elements) const
   { return (rank * total_elements + NRanks-1) / NRanks; }

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

   static int get_face_orientation(Face *face, Element* e1, Element* e2,
                                   int local[2] = NULL);
   void CalcFaceOrientations();

   void UpdateLayers();

   Array<Connection> index_rank; // temporary

   void AddMasterSlaveRanks(int nitems, const NCList& list);
   void MakeShared(const Table &groups, const NCList &list, NCList &shared);

   /** Uniquely encodes a set of leaf elements in the refinement hierarchy of
       an NCMesh. Can be dumped to a stream, sent to another processor, loaded,
       and decoded to identify the same set of elements (refinements) in a
       different but compatible NCMesh. The encoding can optionally include
       the refinement types needed to reach the leaves, so the element set can
       be decoded (recreated) even if the receiver has an incomplete tree. */
   class ElementSet
   {
   public:
      ElementSet(NCMesh *ncmesh = NULL, bool include_ref_types = false)
         : ncmesh(ncmesh), include_ref_types(include_ref_types) {}
      ElementSet(const ElementSet &other);

      void Encode(const Array<Element*> &elements);
      void Dump(std::ostream &os) const;

      void Load(std::istream &is);
      void Decode(Array<Element*> &elements) const;

      void SetNCMesh(NCMesh *ncmesh) { this->ncmesh = ncmesh; }

   protected:
      Array<unsigned char> data; ///< encoded refinement (sub-)trees
      NCMesh* ncmesh;
      bool include_ref_types;

      void EncodeTree(Element* elem);
      void DecodeTree(Element* elem, int &pos, Array<Element*> &elements) const;

      void WriteInt(int value);
      int  GetInt(int pos) const;
      void FlagElements(const Array<Element*> &elements, char flag);
   };

   /// Write to 'os' a processor-independent encoding of vertex/edge/face IDs.
   void EncodeMeshIds(std::ostream &os, Array<MeshId> ids[]);

   /// Read from 'is' a processor-independent encoding of vertex/edge/face IDs.
   void DecodeMeshIds(std::istream &is, Array<MeshId> ids[]);

   bool CheckElementType(Element* elem, int type);

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


   /** A base for internal messages used by Refine(), Derefine() and Rebalance().
    *  Allows sending values associated with elements in a set.
    *  If RefType == true, the element set is recreated on the receiving end.
    */
   template<class ValueType, bool RefTypes, int Tag>
   class ElementValueMessage : public VarMessage<Tag>
   {
   public:
      using VarMessage<Tag>::data;
      std::vector<Element*> elements;
      std::vector<ValueType> values;

      int Size() const { return elements.size(); }
      void Reserve(int size) { elements.reserve(size); values.reserve(size); }

      void Add(Element* elem, ValueType val)
      { elements.push_back(elem); values.push_back(val); }

      /// Set pointer to ParNCMesh (needed to encode the message).
      void SetNCMesh(ParNCMesh* pncmesh) { this->pncmesh = pncmesh; }

      ElementValueMessage() : pncmesh(NULL) {}

   protected:
      ParNCMesh* pncmesh;

      virtual void Encode();
      virtual void Decode();
   };

   /** Used by ParNCMesh::Refine() to inform neighbors about refinements at
    *  the processor boundary. This keeps their ghost layers synchronized.
    */
   class NeighborRefinementMessage : public ElementValueMessage<char, false, 289>
   {
   public:
      void AddRefinement(Element* elem, char ref_type) { Add(elem, ref_type); }
      typedef std::map<int, NeighborRefinementMessage> Map;
   };

   /** Used by ParNCMesh::Derefine() to keep the ghost layers synchronized.
    */
   class NeighborDerefinementMessage : public ElementValueMessage<int, false, 290>
   {
   public:
      void AddDerefinement(Element* elem, int rank) { Add(elem, rank); }
      typedef std::map<int, NeighborDerefinementMessage> Map;
   };

   /** Used in Step 2 of Rebalance() to synchronize new rank assignments in
    *  the ghost layer.
    */
   class NeighborElementRankMessage : public ElementValueMessage<int, false, 156>
   {
   public:
      void AddElementRank(Element* elem, int rank) { Add(elem, rank); }
      typedef std::map<int, NeighborElementRankMessage> Map;
   };

   /** Used by Rebalance() to send elements and their ranks. Note that
    *  RefTypes == true which means the refinement hierarchy will be recreated
    *  on the receiving side.
    */
   class RebalanceMessage : public ElementValueMessage<int, true, 157>
   {
   public:
      void AddElementRank(Element* elem, int rank) { Add(elem, rank); }
      typedef std::map<int, RebalanceMessage> Map;
   };

   /** Allows migrating element data (DOFs) after Rebalance().
    *  Used by SendRebalanceDofs and RecvRebalanceDofs.
    */
   class RebalanceDofMessage : public VarMessage<158>
   {
   public:
      std::vector<int> elem_ids, dofs;
      long dof_offset;

      void SetElements(const Array<Element*> &elems, NCMesh *ncmesh);
      void SetNCMesh(NCMesh* ncmesh) { eset.SetNCMesh(ncmesh); }

      typedef std::map<int, RebalanceDofMessage> Map;

   protected:
      ElementSet eset;

      virtual void Encode();
      virtual void Decode();
   };

   /** Assign new Element::rank to leaf elements and send them to their new
       owners, keeping the ghost layer up to date. Used by Rebalance() and
       Derefine(). */
   void RedistributeElements(Array<int> &new_ranks, int target_elements,
                             bool record_comm);

   /** Recorded communication pattern from last Rebalance. Used by
       Send/RecvRebalanceDofs to ship element DOFs. */
   RebalanceDofMessage::Map send_rebalance_dofs;
   RebalanceDofMessage::Map recv_rebalance_dofs;

   /** After Rebalance, this array holds the old element indices, or -1 if an
       element didn't exist in the mesh previously. After Derefine, it holds
       the ranks of the old (potentially non-existent) fine elements. */
   Array<int> old_index_or_rank;

   /// Stores modified point matrices created by GetFaceNeighbors
   Array<DenseMatrix*> aux_pm_store;
   void ClearAuxPM();

   static bool compare_ranks(const Element* a, const Element* b);
   static bool compare_ranks_indices(const Element* a, const Element* b);

   friend class ParMesh;
   friend class NeighborDofMessage;
};


/** Represents a message about DOF assignment of vertex, edge and face DOFs on
 *  the boundary with another processor. This and other messages below
 *  are only exchanged between immediate neighbors. Used by
 *  ParFiniteElementSpace::GetParallelConformingInterpolation().
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
