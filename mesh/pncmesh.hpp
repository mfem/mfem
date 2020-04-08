// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

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

class FiniteElementSpace;


/** \brief A parallel extension of the NCMesh class.
 *
 *  The basic idea (and assumption) is that all processors share the coarsest
 *  layer ("root elements"). This has the advantage that refinements can easily
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
 *  Vertices, edges and faces that are not owned by this ('MyRank') processor
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
   ParNCMesh(MPI_Comm comm, const NCMesh& ncmesh, int* part = NULL);

   ParNCMesh(const ParNCMesh &other);

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
       elements) so that each processor owns the same number of leaves (+-1).
       The default partitioning strategy is based on equal splitting of the
       space-filling sequence of leaf elements (custom_partition == NULL).
       Alternatively, a used-defined element-rank assignment array can be
       passed. */
   void Rebalance(const Array<int> *custom_partition = NULL);


   // interface for ParFiniteElementSpace

   int GetNElements() const { return NElements; }

   int GetNGhostVertices() const { return NGhostVertices; }
   int GetNGhostEdges() const { return NGhostEdges; }
   int GetNGhostFaces() const { return NGhostFaces; }
   int GetNGhostElements() const { return NGhostElements; }

   // Return a list of vertices/edges/faces shared by this processor and at
   // least one other processor. These are subsets of NCMesh::<entity>_list. */
   const NCList& GetSharedVertices() { GetVertexList(); return shared_vertices; }
   const NCList& GetSharedEdges() { GetEdgeList(); return shared_edges; }
   const NCList& GetSharedFaces() { GetFaceList(); return shared_faces; }

   /// Helper to get shared vertices/edges/faces ('entity' == 0/1/2 resp.).
   const NCList& GetSharedList(int entity)
   {
      switch (entity)
      {
         case 0: return GetSharedVertices();
         case 1: return GetSharedEdges();
         default: return GetSharedFaces();
      }
   }

   /// Return (shared) face orientation relative to its owner element.
   int GetFaceOrientation(int index) const
   {
      return (index < NFaces) ? face_orient[index] : 0;
   }

   typedef short GroupId;
   typedef std::vector<int> CommGroup;

   /// Return vertex/edge/face ('entity' == 0/1/2, resp.) owner.
   GroupId GetEntityOwnerId(int entity, int index)
   {
      MFEM_ASSERT(entity >= 0 && entity < 3, "");
      MFEM_ASSERT(index >= 0, "");
      if (!entity_owner[entity].Size())
      {
         GetSharedList(entity);
      }
      return entity_owner[entity][index];
   }

   /** Return the P matrix communication group ID for a vertex/edge/face.
       The groups are calculated specifically to match the P matrix
       construction algorithm and its communication pattern. */
   GroupId GetEntityGroupId(int entity, int index)
   {
      MFEM_ASSERT(entity >= 0 && entity < 3, "");
      MFEM_ASSERT(index >= 0, "");
      if (!entity_pmat_group[entity].Size())
      {
         CalculatePMatrixGroups();
      }
      return entity_pmat_group[entity][index];
   }

   /// Return a list of ranks contained in the group of the given ID.
   const CommGroup& GetGroup(GroupId id) const
   {
      MFEM_ASSERT(id >= 0, "");
      return groups[id];
   }

   /// Return true if group 'id' contains the given rank.
   bool GroupContains(GroupId id, int rank) const;

   /// Return true if the specified vertex/edge/face is a ghost.
   bool IsGhost(int entity, int index) const
   {
      if (index < 0) // special case prism edge-face constraint
      {
         MFEM_ASSERT(entity == 2, "");
         entity = 1;
         index = -1 - index;
      }
      switch (entity)
      {
         case 0: return index >= NVertices;
         case 1: return index >= NEdges;
         default: return index >= NFaces;
      }
   }

   /** Returns owner processor for element 'index'. This is normally MyRank but
       for index >= NElements (i.e., for ghosts) it may be something else. */
   int ElementRank(int index) const
   {
      return elements[leaf_elements[index]].rank;
   }


   // utility

   int GetMyRank() const { return MyRank; }

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

   /// Save memory by releasing all non-essential and cached data.
   virtual void Trim();

   /// Return total number of bytes allocated.
   long MemoryUsage(bool with_base = true) const;

   int PrintMemoryDetail(bool with_base = true) const;

   /** Extract a debugging Mesh containing all leaf elements, including ghosts.
       The debug mesh will have element attributes set to element rank + 1. */
   void GetDebugMesh(Mesh &debug_mesh) const;


protected: // interface for ParMesh

   friend class ParMesh;

   /** For compatibility with conforming code in ParMesh and ParFESpace.
       Initializes shared structures in ParMesh: gtopo, shared_*, group_s*, s*_l*.
       The ParMesh then acts as a parallel mesh cut along the NC interfaces. */
   void GetConformingSharedStructures(class ParMesh &pmesh);

   /** Populate face neighbor members of ParMesh from the ghost layer, without
       communication. */
   void GetFaceNeighbors(class ParMesh &pmesh);


protected: // implementation

   MPI_Comm MyComm;
   int NRanks, MyRank;

   int NGhostVertices, NGhostEdges, NGhostFaces;
   int NElements, NGhostElements;

   typedef std::vector<CommGroup> GroupList;
   typedef std::map<CommGroup, GroupId> GroupMap;

   GroupList groups;  // comm group list; NOTE: groups[0] = { MyRank }
   GroupMap group_id; // search index over groups

   // owner rank for each vertex, edge and face (encoded as singleton group)
   Array<GroupId> entity_owner[3];
   // P matrix comm pattern groups for each vertex/edge/face (0/1/2)
   Array<GroupId> entity_pmat_group[3];

   // ParMesh-compatible (conforming) groups for each vertex/edge/face (0/1/2)
   Array<GroupId> entity_conf_group[3];
   // ParMesh compatibility helper arrays to order groups, also temporary
   Array<int> leaf_glob_order, entity_elem_local[3];

   // lists of vertices/edges/faces shared by us and at least one more processor
   NCList shared_vertices, shared_edges, shared_faces;

   Array<char> face_orient; // see CalcFaceOrientations

   /** Type of each leaf element:
         1 - our element (rank == MyRank),
         3 - our element, and neighbor to the ghost layer,
         2 - ghost layer element (existing element, but rank != MyRank),
         0 - element beyond the ghost layer, may not be a real element.
       Note: indexed by Element::index. See also UpdateLayers(). */
   Array<char> element_type;

   Array<int> ghost_layer;    ///< list of elements whose 'element_type' == 2.
   Array<int> boundary_layer; ///< list of type 3 elements

   virtual void Update();

   virtual bool IsGhost(const Element& el) const
   { return el.rank != MyRank; }

   virtual int GetNumGhostElements() const { return NGhostElements; }
   virtual int GetNumGhostVertices() const { return NGhostVertices; }

   /// Return the processor number for a global element number.
   int Partition(long index, long total_elements) const
   { return index * NRanks / total_elements; }

   /// Helper to get the partitioning when the serial mesh gets split initially
   int InitialPartition(int index) const
   { return Partition(index, leaf_elements.Size()); }

   /// Return the global index of the first element owned by processor 'rank'.
   long PartitionFirstIndex(int rank, long total_elements) const
   { return (rank * total_elements + NRanks-1) / NRanks; }

   virtual void UpdateVertices();
   virtual void AssignLeafIndices();
   virtual void OnMeshUpdated(Mesh *mesh);

   virtual void BuildFaceList();
   virtual void BuildEdgeList();
   virtual void BuildVertexList();

   virtual void ElementSharesFace(int elem, int local, int face);
   virtual void ElementSharesEdge(int elem, int local, int enode);
   virtual void ElementSharesVertex(int elem, int local, int vnode);

   GroupId GetGroupId(const CommGroup &group);
   GroupId GetSingletonGroup(int rank);

   Array<int> tmp_owner; // temporary
   Array<char> tmp_shared_flag; // temporary
   Array<Connection> entity_index_rank[3]; // temporary

   void InitOwners(int num, Array<GroupId> &entity_owner);
   void MakeSharedList(const NCList &list, NCList &shared);

   void AddConnections(int entity, int index, const Array<int> &ranks);
   void CalculatePMatrixGroups();
   void CreateGroups(int nentities, Array<Connection> &index_rank,
                     Array<GroupId> &entity_group);

   static int get_face_orientation(Face &face, Element &e1, Element &e2,
                                   int local[2] = NULL /* optional output */);
   void CalcFaceOrientations();

   void UpdateLayers();

   void MakeSharedTable(int ngroups, int ent, Array<int> &shared_local,
                        Table &group_shared, Array<char> *entity_geom = NULL,
                        char geom = 0);

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

      void Encode(const Array<int> &elements);
      void Dump(std::ostream &os) const;

      void Load(std::istream &is);
      void Decode(Array<int> &elements) const;

      void SetNCMesh(NCMesh *ncmesh) { this->ncmesh = ncmesh; }
      const NCMesh* GetNCMesh() const { return ncmesh; }

   protected:
      Array<unsigned char> data; ///< encoded refinement (sub-)trees
      NCMesh* ncmesh;
      bool include_ref_types;

      void EncodeTree(int elem);
      void DecodeTree(int elem, int &pos, Array<int> &elements) const;

      void WriteInt(int value);
      int  GetInt(int pos) const;
      void FlagElements(const Array<int> &elements, char flag);

#ifdef MFEM_DEBUG
      mutable Array<int> ref_path;
      std::string RefPath() const;
#endif
   };

   /** Adjust some of the MeshIds before encoding for recipient 'rank', so that
       they only reference elements that exist in the recipient's ref. tree. */
   void AdjustMeshIds(Array<MeshId> ids[], int rank);

   void ChangeVertexMeshIdElement(NCMesh::MeshId &id, int elem);
   void ChangeEdgeMeshIdElement(NCMesh::MeshId &id, int elem);
   void ChangeRemainingMeshIds(Array<MeshId> &ids, int pos,
                               const Array<Pair<int, int> > &find);

   // Write/read a processor-independent encoding of vertex/edge/face IDs.
   void EncodeMeshIds(std::ostream &os, Array<MeshId> ids[]);
   void DecodeMeshIds(std::istream &is, Array<MeshId> ids[]);

   // Write/read comm groups and a list of their IDs.
   void EncodeGroups(std::ostream &os, const Array<GroupId> &ids);
   void DecodeGroups(std::istream &is, Array<GroupId> &ids);

   bool CheckElementType(int elem, int type);

   Array<int> tmp_neighbors; // temporary, used by ElementNeighborProcessors

   /** Return a list of processors that own elements in the immediate
       neighborhood of 'elem' (i.e., vertex, edge and face neighbors),
       and are not 'MyRank'. */
   void ElementNeighborProcessors(int elem, Array<int> &ranks);

   /** Get a list of ranks that own elements in the neighborhood of our region.
       NOTE: MyRank is not included. */
   void NeighborProcessors(Array<int> &neighbors);

   /** Traverse the (local) refinement tree and determine which subtrees are
       no longer needed, i.e., their leaves are not owned by us nor are they our
       ghosts. These subtrees are then derefined. */
   void Prune();

   /// Internal. Recursive part of Prune().
   bool PruneTree(int elem);


   /** A base for internal messages used by Refine(), Derefine() and Rebalance().
    *  Allows sending values associated with elements in a set.
    *  If RefType == true, the element set is recreated on the receiving end.
    */
   template<class ValueType, bool RefTypes, int Tag>
   class ElementValueMessage : public VarMessage<Tag>
   {
   public:
      using VarMessage<Tag>::data;
      std::vector<int> elements;
      std::vector<ValueType> values;

      int Size() const { return elements.size(); }
      void Reserve(int size) { elements.reserve(size); values.reserve(size); }

      void Add(int elem, ValueType val)
      { elements.push_back(elem); values.push_back(val); }

      /// Set pointer to ParNCMesh (needed to encode the message).
      void SetNCMesh(ParNCMesh* pncmesh) { this->pncmesh = pncmesh; }

      ElementValueMessage() : pncmesh(NULL) {}

   protected:
      ParNCMesh* pncmesh;

      virtual void Encode(int);
      virtual void Decode(int);
   };

   /** Used by ParNCMesh::Refine() to inform neighbors about refinements at
    *  the processor boundary. This keeps their ghost layers synchronized.
    */
   class NeighborRefinementMessage : public ElementValueMessage<char, false, 289>
   {
   public:
      void AddRefinement(int elem, char ref_type) { Add(elem, ref_type); }
      typedef std::map<int, NeighborRefinementMessage> Map;
   };

   /** Used by ParNCMesh::Derefine() to keep the ghost layers synchronized.
    */
   class NeighborDerefinementMessage : public ElementValueMessage<int, false, 290>
   {
   public:
      void AddDerefinement(int elem, int rank) { Add(elem, rank); }
      typedef std::map<int, NeighborDerefinementMessage> Map;
   };

   /** Used in Step 2 of Rebalance() to synchronize new rank assignments in
    *  the ghost layer.
    */
   class NeighborElementRankMessage : public ElementValueMessage<int, false, 156>
   {
   public:
      void AddElementRank(int elem, int rank) { Add(elem, rank); }
      typedef std::map<int, NeighborElementRankMessage> Map;
   };

   /** Used by Rebalance() to send elements and their ranks. Note that
    *  RefTypes == true which means the refinement hierarchy will be recreated
    *  on the receiving side.
    */
   class RebalanceMessage : public ElementValueMessage<int, true, 157>
   {
   public:
      void AddElementRank(int elem, int rank) { Add(elem, rank); }
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

      void SetElements(const Array<int> &elems, NCMesh *ncmesh);
      void SetNCMesh(NCMesh* ncmesh) { eset.SetNCMesh(ncmesh); }
      long MemoryUsage() const;

      typedef std::map<int, RebalanceDofMessage> Map;

   protected:
      ElementSet eset;

      virtual void Encode(int);
      virtual void Decode(int);
   };

   /** Assign new Element::rank to leaf elements and send them to their new
       owners, keeping the ghost layer up to date. Used by Rebalance() and
       Derefine(). 'target_elements' is the number of elements this rank
       is supposed to own after the exchange. If this number is not known
       a priori, the parameter can be set to -1, but more expensive communication
       (synchronous sends and a barrier) will be used in that case. */
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

   long GroupsMemoryUsage() const;

   friend class NeighborRowMessage;
};



// comparison operator so that MeshId can be used as key in std::map
inline bool operator< (const NCMesh::MeshId &a, const NCMesh::MeshId &b)
{
   return a.index < b.index;
}

// equality of MeshId is based on 'index' (element/local are not unique)
inline bool operator== (const NCMesh::MeshId &a, const NCMesh::MeshId &b)
{
   return a.index == b.index;
}

} // namespace mfem

#endif // MFEM_USE_MPI

#endif // MFEM_PNCMESH
