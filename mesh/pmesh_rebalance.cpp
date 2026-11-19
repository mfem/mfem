// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

// Implementation of ParMesh::Rebalance(const Array<int> &) for conforming
// meshes of triangles and tetrahedra: RebalanceConforming().
//
// Outline (each step is local or exchanges data with neighbors only):
//
//  0. The arguments are checked, and the ranks that share entities compare
//     their descriptions of them (CheckRebalanceArguments(),
//     CheckRebalanceGroupSizes(), CheckRebalanceSharedEntities()).
//  1. Each vertex gets a global id: the master rank of its group and the local
//     index of the vertex there, as in ParMesh::ReorientTetMesh()
//     (GetRebalanceVertexIds()).
//  2. Each vertex, edge (3D) and face gets the set of new ranks of the local
//     elements around it. The sets of the shared entities are completed over
//     the current groups, so that every rank knows, for each of its vertices
//     and edges, all the new ranks holding it, and for each face the new rank
//     of the element on the other side (GetRebalanceTarget()).
//  3. The elements, their boundary elements, vertices and shared edges are
//     packed per destination (PackRebalanceMessages()) and exchanged with the
//     non-blocking consensus algorithm (MPI_Issend, MPI_Iprobe, MPI_Ibarrier),
//     as in ParNCMesh::RedistributeElements() (ExchangeMessages()). The
//     messages are decoded in the order of the source ranks, so the result
//     does not depend on their arrival (DecodeRebalanceMessages()). The nodes
//     are moved by the transfer operator of their space, as the other grid
//     functions (ParFiniteElementSpace::Update()).
//  4. BuildFromConformingRecords() builds the new local mesh
//     (BuildLocalMeshFromRecords()) and its groups of shared vertices, edges
//     and faces (BuildGroupsFromRecords()). Inside a group, the entities are
//     sorted by their global vertex ids, so all members agree on their order.
//     The lower rank of the two ranks of a shared triangle orients it from its
//     element (refinement edge first if the element is marked) and sends it
//     to the higher rank, which adopts it (OrientSharedTriangles()). Then it
//     publishes the migration and updates the nodes (PublishRebalance()).
//
// Memory: no array is dimensioned by the number of ranks. Each vertex and
// shared edge travels with the set of ranks that hold it after the call, in
// each message that contains it, and the sets of the shared entities are
// completed over their groups. The peak on a rank (old mesh, send buffers,
// receive buffers, new mesh) is therefore of the order of the size of its
// local mesh before plus after the call, times the largest number of ranks
// that share a vertex before or after the call. When each rank that holds
// a vertex holds one of its elements, that number is at most the number of
// elements around the vertex; a vertex that a rank holds without any of its
// elements (which the rebalance keeps there) can be shared by any number of
// ranks. A vertex in elements sent to s ranks costs of the order of s^2 on
// the sending rank, and a group of s ranks of the order of s^2 on its master
// when the groups are built (GroupTopology::Create(), see
// ConformingRecords::GroupTopologyEntries()): the sizes of both are checked
// before the mesh is replaced.

#include "../config/config.hpp"

#ifdef MFEM_USE_MPI

#include "mesh_headers.hpp"
#include "../fem/pfespace.hpp"
#include "../fem/pgridfunc.hpp"
#include "../general/sets.hpp"
#include "submesh/psubmesh.hpp"

#include <algorithm>
#include <array>
#include <climits>
#include <cstdint>
#include <cstring>
#include <functional>
#include <map>
#include <set>
#include <utility>
#include <vector>

namespace mfem
{

struct ParMesh::ConformingRecords
{
   /// For each element: attribute, refinement flag (tetrahedra), previous
   /// rank and previous local index.
   Array<int> elem_attr, elem_flag, elem_old_rank, elem_old_index;
   /// Global vertex ids of the elements, Dim+1 per element, in element order.
   Array<std::int64_t> elem_vert;
   /** For each element and each of its Dim+1 faces (edges in 2D), in the
       local order of the faces: the rank of the element on the other side of
       the face, or -1 if there is none. */
   Array<int> elem_nbr;
   /// Global ids of the vertices of no element (kept by their rank).
   Array<std::int64_t> iso_vert;
   /// The new local indices of the vertices of iso_vert (set by the build).
   Array<int> iso_new;

   /// For each boundary element: attribute and Dim global vertex ids.
   Array<int> bdr_attr;
   Array<std::int64_t> bdr_vert;

   /** Vertex records: global id, spaceDim coordinates, and the ranks holding
       the vertex (none for a vertex held by this rank only). A vertex may have
       several, identical, records. */
   Array<std::int64_t> vert_id;
   Array<real_t> vert_coord;
   Array<int> vert_set_offset, vert_set;

   /** Records of the edges (3D) held by more than one rank: the two global
       vertex ids, smaller first, and the ranks holding the edge. An edge may
       have several, identical, records. */
   Array<std::int64_t> edge_vert;
   Array<int> edge_set_offset, edge_set;

   /// The groups of the new local mesh (SharedGroups()), counted before the
   /// mesh is replaced.
   std::set<std::vector<int>> groups;

   /// The vertices of the new local mesh, set by the build: their global ids,
   /// in increasing order, and the index of one of their records.
   Array<std::int64_t> new_vert_id;
   Array<int> new_vert_rec;

   ConformingRecords()
   {
      vert_set_offset.Append(0);
      edge_set_offset.Append(0);
   }

   /// The vertex of the new local mesh with the global id @a id.
   int LocalVertex(std::int64_t id) const
   {
      const int i = new_vert_id.FindSorted(id);
      MFEM_VERIFY(i >= 0, "internal error: vertex " << id << " not received");
      return i;
   }

   /// Whether the vertex @a v of the new local mesh is held by other ranks.
   bool SharedVertex(int v) const
   {
      const int r = new_vert_rec[v];
      return vert_set_offset[r+1] - vert_set_offset[r] > 1;
   }

   /// The ranks holding the vertex @a v of the new local mesh.
   std::vector<int> VertexSet(int v) const
   {
      const int r = new_vert_rec[v];
      const int *s = vert_set.GetData();
      return std::vector<int>(s + vert_set_offset[r], s + vert_set_offset[r+1]);
   }

   /** The groups of shared entities that BuildFromConformingRecords() makes
       from the records on rank @a rank: the sets of ranks (sorted, with
       @a rank) of the vertices, and in 3D of the edges, held by more than one
       rank, and the pairs of @a rank and the rank on the other side of each
       shared face (edge in 2D) of the elements. */
   std::set<std::vector<int>> SharedGroups(int rank, int dim) const;

   /** The number of entries of the largest array that GroupTopology::Create()
       allocates on rank @a rank for its @a groups, see the definition. */
   static std::int64_t GroupTopologyEntries(
      int rank, const std::set<std::vector<int>> &groups);
};

namespace
{

// MPI tags of the messages of ParMesh::RebalanceConforming()
constexpr int rebalance_sets_tag = 3801;
constexpr int rebalance_elements_tag = 3802;
constexpr int rebalance_groups_tag = 3803;
constexpr int rebalance_check_tag = 3804;
constexpr int rebalance_faces_tag = 3805;
constexpr int rebalance_rows_tag = 3806;
constexpr int rebalance_sizes_tag = 3807;

/** The largest number of entries of the arrays of a rebalance. The sizes are
    computed in 64 bits and checked against it before the arrays are filled.
    It is half of INT_MAX: an Array grows by doubling its capacity, an int,
    which then stays below INT_MAX. */
constexpr std::int64_t max_entries = INT_MAX / 2;

/** The largest number of 64-bit words of a message, a std::vector whose size
    is the count of an MPI call, an int. */
constexpr std::int64_t max_words = INT_MAX;

/** Whether an Array of @a entries entries, or a message of @a words words,
    would be too large: the rule of all the arrays and messages of a
    rebalance and of the transfer of grid functions after it. */
bool TooLarge(std::int64_t entries, std::int64_t words)
{
   return entries > max_entries || words > max_words;
}

/// Message, after a refusal of the sizes, on all ranks.
const char *const too_large_message =
   "The mesh is too large to be rebalanced: an array of the rebalance would"
   " have more than INT_MAX/2 entries, or a message more than INT_MAX words;"
   " use more ranks or a partition that sends fewer elements to a rank.";

/// Sets of ranks of rows 0..n-1, stored as a compressed table.
struct RankSets
{
   Array<int> I, J;

   /// Build the sets from (row, rank) pairs, with sorted unique ranks.
   void Build(int nrows, const Array<int> &rows, const Array<int> &ranks)
   {
      I.SetSize(nrows + 1);
      I = 0;
      for (int k = 0; k < rows.Size(); k++) { I[rows[k] + 1]++; }
      I.PartialSum();
      Array<int> pos(nrows);
      for (int r = 0; r < nrows; r++) { pos[r] = I[r]; }
      J.SetSize(I[nrows]);
      for (int k = 0; k < rows.Size(); k++) { J[pos[rows[k]]++] = ranks[k]; }
      int n = 0;
      for (int r = 0; r < nrows; r++)
      {
         const int b = I[r], e = I[r + 1];
         std::sort(J.GetData() + b, J.GetData() + e);
         I[r] = n;
         for (int k = b; k < e; k++)
         {
            if (n == I[r] || J[k] != J[n - 1]) { J[n++] = J[k]; }
         }
      }
      I[nrows] = n;
      J.SetSize(n);
   }

   int Size(int r) const { return I[r + 1] - I[r]; }
   const int *Row(int r) const { return J.GetData() + I[r]; }
};

/// Append a set of ranks: its size, then its elements.
void AppendSet(Array<int> &buf, const RankSets &sets, int row)
{
   buf.Append(sets.Size(row));
   buf.Append(sets.Row(row), sets.Size(row));
}

/** Pack an integer and a real buffer in one message of 8-byte words: the two
    sizes, the integers, then the bytes of the reals. */
void PackMessage(const Array<std::int64_t> &ints, const Array<real_t> &reals,
                 std::vector<std::int64_t> &msg)
{
   const std::size_t real_bytes = sizeof(real_t)*reals.Size();
   const std::size_t real_words = (real_bytes + 7)/8;
   msg.assign(2 + ints.Size() + real_words, 0);
   msg[0] = ints.Size();
   msg[1] = reals.Size();
   std::copy(ints.begin(), ints.end(), msg.begin() + 2);
   if (real_bytes)
   {
      std::memcpy(msg.data() + 2 + ints.Size(), reals.GetData(), real_bytes);
   }
}

/** Send the message @a send[k] to rank @a dests[k], for every k, and receive
    the messages sent to this rank, with the non-blocking consensus algorithm
    (Hoefler, Siebert and Lumsdaine, PPoPP 2010): synchronous sends, probes for
    any source, and a non-blocking barrier entered once the local sends have
    been received. The message to this rank is moved without MPI. The messages
    are returned per source rank in @a recv; the send buffers are freed. */
void ExchangeMessages(MPI_Comm comm, const Array<int> &dests,
                      std::vector<std::vector<std::int64_t>> &send,
                      std::map<int, std::vector<std::int64_t>> &recv,
                      int tag = rebalance_elements_tag)
{
   int myrank;
   MPI_Comm_rank(comm, &myrank);
   const MPI_Datatype word_type = MPITypeMap<std::int64_t>::mpi_type;
   std::vector<MPI_Request> send_req;
   send_req.reserve(dests.Size());
   for (int k = 0; k < dests.Size(); k++)
   {
      if (dests[k] == myrank)
      {
         recv[myrank].swap(send[k]);
         continue;
      }
      MFEM_VERIFY(!TooLarge(0, static_cast<std::int64_t>(send[k].size())),
                  "internal error: message size");
      send_req.push_back(MPI_REQUEST_NULL);
      MPI_Issend(send[k].data(), static_cast<int>(send[k].size()), word_type,
                 dests[k], tag, comm, &send_req.back());
   }
   MPI_Request barrier = MPI_REQUEST_NULL;
   int done = 0;
   while (!done)
   {
      int flag;
      MPI_Status status;
      MPI_Iprobe(MPI_ANY_SOURCE, tag, comm, &flag, &status);
      if (flag)
      {
         int count;
         MPI_Get_count(&status, word_type, &count);
         std::vector<std::int64_t> &msg = recv[status.MPI_SOURCE];
         MFEM_VERIFY(msg.empty(), "internal error: second message from rank "
                     << status.MPI_SOURCE);
         msg.resize(count);
         MPI_Recv(msg.data(), count, word_type, status.MPI_SOURCE,
                  tag, comm, MPI_STATUS_IGNORE);
      }
      else if (barrier == MPI_REQUEST_NULL)
      {
         int sent;
         MPI_Testall(static_cast<int>(send_req.size()), send_req.data(), &sent,
                     MPI_STATUSES_IGNORE);
         if (sent) { MPI_Ibarrier(comm, &barrier); }
      }
      else
      {
         MPI_Test(&barrier, &done, MPI_STATUS_IGNORE);
      }
   }
   send.clear();
   send.shrink_to_fit();
}

/** Rotate the triangle @a v so that its vertex with the smallest index comes
    first, keeping its orientation. */
void RotateToSmallestFirst(int *v)
{
   if (v[1] < v[0] && v[1] < v[2])
   {
      const int t = v[0]; v[0] = v[1]; v[1] = v[2]; v[2] = t;
   }
   else if (v[2] < v[0] && v[2] < v[1])
   {
      const int t = v[2]; v[2] = v[1]; v[1] = v[0]; v[0] = t;
   }
}

/// Sort the first @a n (2 or 3) entries of @a v in increasing order.
void SortSmall(int *v, int n)
{
   if (v[1] < v[0]) { std::swap(v[0], v[1]); }
   if (n == 3)
   {
      if (v[2] < v[1]) { std::swap(v[1], v[2]); }
      if (v[1] < v[0]) { std::swap(v[0], v[1]); }
   }
}

/// Group set of a face shared by ranks @a p and @a q.
std::vector<int> PairSet(int p, int q)
{
   return {std::min(p, q), std::max(p, q)};
}

/// Global numbers of elements, vertices, edges and (3D) faces of @a pmesh,
/// each shared entity counted once, on the master of its group. Collective.
std::array<long long, 4> GlobalEntityCounts(const ParMesh &pmesh)
{
   long long c[4] = {pmesh.GetNE(), pmesh.GetNV(), pmesh.GetNEdges(),
                     (pmesh.Dimension() == 3) ? pmesh.GetNFaces() : 0
                    };
   for (int g = 1; g < pmesh.GetNGroups(); g++)
   {
      if (pmesh.gtopo.IAmMaster(g)) { continue; }
      c[1] -= pmesh.GroupNVertices(g);
      c[2] -= pmesh.GroupNEdges(g);
      if (pmesh.Dimension() == 3)
      {
         c[3] -= pmesh.GroupNTriangles(g) + pmesh.GroupNQuadrilaterals(g);
      }
   }
   MPI_Allreduce(MPI_IN_PLACE, c, 4, MPI_LONG_LONG, MPI_SUM, pmesh.GetComm());
   return {c[0], c[1], c[2], c[3]};
}

/** The groups of @a gtopo shared with each neighbor: row n of @a nbr_groups
    lists those of neighbor n, in the order of (master rank, group number in
    the master), which both sides know. */
void NeighborGroups(const GroupTopology &gtopo, Table &nbr_groups)
{
   const int nnbr = gtopo.GetNumNeighbors();
   nbr_groups.MakeI(nnbr);
   for (int g = 1; g < gtopo.NGroups(); g++)
   {
      const int *lprocs = gtopo.GetGroup(g);
      for (int k = 0; k < gtopo.GetGroupSize(g); k++)
      {
         if (lprocs[k]) { nbr_groups.AddAColumnInRow(lprocs[k]); }
      }
   }
   nbr_groups.MakeJ();
   for (int g = 1; g < gtopo.NGroups(); g++)
   {
      const int *lprocs = gtopo.GetGroup(g);
      for (int k = 0; k < gtopo.GetGroupSize(g); k++)
      {
         if (lprocs[k]) { nbr_groups.AddConnection(lprocs[k], g); }
      }
   }
   nbr_groups.ShiftUpI();
   {
      auto group_less = [&gtopo](int a, int b)
      {
         const int ra = gtopo.GetGroupMasterRank(a);
         const int rb = gtopo.GetGroupMasterRank(b);
         return (ra != rb) ? (ra < rb) :
                (gtopo.GetGroupMasterGroup(a) < gtopo.GetGroupMasterGroup(b));
      };
      for (int n = 1; n < nnbr; n++)
      {
         int *row = nbr_groups.GetRow(n);
         std::sort(row, row + nbr_groups.RowSize(n), group_less);
      }
   }
}

/** The length of the row of an element in the transfer of the nodes of
    @a pmesh after a rebalance, see
    ParFiniteElementSpace::ConformingRebalanceMatrix(): its vector DOFs, and
    its face orientations where the nodal space has a DOF transformation; 0
    for a mesh without nodes. */
std::int64_t NodalRowLength(const ParMesh &pmesh)
{
   const GridFunction *nodes = pmesh.GetNodes();
   const int dim = pmesh.Dimension();
   std::int64_t nodal_row = 0;
   if (nodes)
   {
      const FiniteElementCollection *fec = nodes->FESpace()->FEColl();
      const Geometry::Type geom =
         (dim == 3) ? Geometry::TETRAHEDRON : Geometry::TRIANGLE;
      nodal_row = std::int64_t(nodes->FESpace()->GetVDim())*
                  fec->FiniteElementForGeometry(geom)->GetDof();
      const StatelessDofTransformation *t =
         fec->DofTransformationForGeometry(geom);
      if (t && !t->IsIdentity()) { nodal_row += dim + 1; }
   }
   return nodal_row;
}

} // anonymous namespace

/// The new ranks of the entities of the local mesh in a conforming rebalance.
struct ParMesh::ConformingTarget
{
   /// For each vertex, and in 3D each edge: the new ranks of the elements
   /// around it, on all the ranks that hold it.
   RankSets vert_sets, edge_sets;
   /// For each face (edge in 2D): the new ranks of the elements on its sides
   /// 1 and 2, or -1 where there is no element.
   Array<int> face_rank1, face_rank2;
   /// The vertices of no element, which stay on this rank.
   Array<int> isolated;
};

void ParMesh::GetRebalanceOldElements(Array<int> &old_rank,
                                      Array<int> &old_index) const
{
   MFEM_VERIFY(rebalance_sequence == sequence &&
               last_operation == Mesh::REBALANCE,
               "the last modification of the mesh was not Rebalance() of a"
               " conforming mesh");
   rebalance_old_rank.Copy(old_rank);
   rebalance_old_index.Copy(old_index);
}

void ParMesh::RebalanceConforming(const Array<int> &partition)
{
   CheckRebalanceArguments(partition);

   Table nbr_groups;
   NeighborGroups(gtopo, nbr_groups);
   CheckRebalanceGroupSizes(nbr_groups);

   // 1. Global vertex ids: (master rank << 32) + index of the vertex there.
   Array<std::int64_t> vert_gid;
   GetRebalanceVertexIds(vert_gid);

   const std::array<long long, 4> counts_before = GlobalEntityCounts(*this);
   CheckRebalanceSharedEntities(nbr_groups, vert_gid);

   ConformingTarget target;

   // 2. New ranks of the local elements around each vertex and edge (3D), as
   //    (entity, rank) pairs, and on each side of each face.
   int too_large = GetRebalanceTarget(partition, nbr_groups, target);

   // The length of the row of an element in the transfer of the nodes after
   // the rebalance: its sizes are checked with those of the rebalance, before
   // the mesh changes.
   const std::int64_t nodal_row = NodalRowLength(*this);
   Array<int> dests;
   std::vector<std::vector<std::int64_t>> msgs;

   // 3. Pack one message per destination: elements (in the local order),
   //    boundary elements, vertices, shared edges (3D), then the coordinates
   //    of the vertices.
   too_large = PackRebalanceMessages(partition, vert_gid, target, nodal_row,
                                     too_large, dests, msgs);
   MPI_Allreduce(MPI_IN_PLACE, &too_large, 1, MPI_INT, MPI_MAX, MyComm);
   MFEM_VERIFY(!too_large, too_large_message);

   std::map<int, std::vector<std::int64_t>> recv_msgs;
   ExchangeMessages(MyComm, dests, msgs, recv_msgs);

   // The records of all the messages received must fit, too, and so must the
   // rows of the nodal space that the update of the nodes will receive: the
   // rebalance refuses before the mesh is replaced, or completes.
   {
      std::int64_t n_ints = 0, n_reals = 0, n_elems = 0;
      for (const auto &m : recv_msgs)
      {
         MFEM_VERIFY(m.second.size() >= 3, "internal error: message size");
         n_ints += m.second[0];
         n_reals += m.second[1];
         n_elems += m.second[2];
      }
      too_large = TooLarge(std::max({n_ints, n_reals, n_elems*nodal_row}), 0);
      MPI_Allreduce(MPI_IN_PLACE, &too_large, 1, MPI_INT, MPI_MAX, MyComm);
      MFEM_VERIFY(!too_large, too_large_message);
   }

   // 4. Decode, in the order of the source ranks, and build the new mesh.
   ConformingRecords rec;
   DecodeRebalanceMessages(recv_msgs, rec);

   // The tables of the groups of the new mesh must fit too: the master of a
   // group of many ranks holds a description of the group for each of them.
   rec.groups = rec.SharedGroups(MyRank, Dim);
   const std::int64_t group_entries =
      ConformingRecords::GroupTopologyEntries(MyRank, rec.groups);
   too_large = TooLarge(group_entries, 0);
   MPI_Allreduce(MPI_IN_PLACE, &too_large, 1, MPI_INT, MPI_MAX, MyComm);
   MFEM_VERIFY(!too_large, too_large_message);

   // Nothing below refuses: from here on the mesh is replaced, with its face
   // neighbor data. A refusal above leaves the mesh, caches included, as it
   // was.
   DeleteFaceNbrData();
   BuildFromConformingRecords(rec, partition, target.isolated);
   const std::array<long long, 4> counts_after = GlobalEntityCounts(*this);
   MFEM_VERIFY(counts_after == counts_before, "internal error: the rebalance"
               " changed the global numbers of elements, vertices, edges or"
               " faces (" << counts_before[0] << ", " << counts_before[1]
               << ", " << counts_before[2] << ", " << counts_before[3]
               << " before; " << counts_after[0] << ", " << counts_after[1]
               << ", " << counts_after[2] << ", " << counts_after[3]
               << " after).");
}

void ParMesh::CheckRebalanceArguments(const Array<int> &partition) const
{
   // All ranks reach the same decision: 'meshgen', 'Dim' and 'NURBSext' are
   // the same on all ranks, the other checks are reduced.
   MFEM_VERIFY(!NURBSext, "Rebalancing is not supported for NURBS meshes.");
   MFEM_VERIFY(!ParSubMesh::IsParSubMesh(this), "Rebalancing a ParSubMesh is"
               " not supported: its maps to the parent mesh would no longer"
               " be valid.");
   MFEM_VERIFY((Dim == 2 || Dim == 3) && meshgen == 1,
               "Rebalancing a conforming mesh is only supported for meshes of"
               " triangles (2D) or tetrahedra (3D).");
   MFEM_VERIFY(Dim == 2 || el_to_face, "internal error: no faces");
   int bad = (partition.Size() != NumOfElements) ? 1 : 0;
   for (int i = 0; i < partition.Size() && !bad; i++)
   {
      if (partition[i] < 0 || partition[i] >= NRanks) { bad = 2; }
   }
   if (Nodes)
   {
      const ParFiniteElementSpace *pfes =
         dynamic_cast<const ParFiniteElementSpace *>(Nodes->FESpace());
      if (!pfes || pfes->IsVariableOrder()) { bad = 4; }
   }
   // A bound of the sizes of the arrays of (entity, rank) pairs of step 2
   // (GetRebalanceTarget()), of the sets sent to each neighbor there, and of
   // the destinations of the elements and boundary elements of step 3.
   if (TooLarge(2*std::int64_t(NumOfVertices) + NumOfEdges + GetNumFaces() +
                10*std::int64_t(NumOfElements) + NumOfBdrElements, 0))
   {
      bad |= 8;
   }
   int glob_bad = 0;
   MPI_Allreduce(&bad, &glob_bad, 1, MPI_INT, MPI_BOR, MyComm);
   MFEM_VERIFY(!(glob_bad & 1), "The size of the partition must be the"
               " number of local elements.");
   MFEM_VERIFY(!(glob_bad & 2), "Partition values must be in [0, "
               << NRanks << ").");
   MFEM_VERIFY(!(glob_bad & 4), "The nodes of the mesh must be in a"
               " ParFiniteElementSpace of uniform order.");
   MFEM_VERIFY(!(glob_bad & 8), too_large_message);
}

void ParMesh::CheckRebalanceGroupSizes(const Table &nbr_groups) const
{
   const int nnbr = gtopo.GetNumNeighbors();
   // The members of a group must list the same numbers of shared vertices,
   // edges and faces in it. They are compared first, with messages whose size
   // the receiver does not assume: the group communicators of the next steps
   // rely on these numbers. The groups themselves, and which shared vertex of
   // a group is which on each member, are taken as given, as in the other
   // parallel operations: there is nothing independent to check them against.
   std::vector<std::vector<std::int64_t>> send(nnbr);
   int too_many = 0;
   for (int n = 1; n < nnbr; n++)
   {
      for (int k = 0; k < nbr_groups.RowSize(n); k++)
      {
         const int g = nbr_groups.GetRow(n)[k];
         send[n].push_back(gtopo.GetGroupMasterRank(g));
         send[n].push_back(gtopo.GetGroupMasterGroup(g));
         send[n].push_back(group_svert.RowSize(g-1));
         send[n].push_back(group_sedge.RowSize(g-1));
         send[n].push_back((Dim == 3) ? group_stria.RowSize(g-1) : 0);
      }
      if (TooLarge(0, static_cast<std::int64_t>(send[n].size())))
      {
         too_many = 1;
      }
   }
   MPI_Allreduce(MPI_IN_PLACE, &too_many, 1, MPI_INT, MPI_MAX, MyComm);
   MFEM_VERIFY(!too_many, too_large_message);
   std::vector<MPI_Request> req(nnbr, MPI_REQUEST_NULL);
   for (int n = 1; n < nnbr; n++)
   {
      MPI_Isend(send[n].data(), static_cast<int>(send[n].size()),
                MPI_INT64_T, gtopo.GetNeighborRank(n), rebalance_sizes_tag,
                MyComm, &req[n]);
   }
   // Groups listed differently by a neighbor.
   long long bad = 0;
   std::vector<std::int64_t> theirs;
   for (int n = 1; n < nnbr; n++)
   {
      MPI_Status status;
      MPI_Probe(gtopo.GetNeighborRank(n), rebalance_sizes_tag, MyComm,
                &status);
      int count;
      MPI_Get_count(&status, MPI_INT64_T, &count);
      theirs.resize(count);
      MPI_Recv(theirs.data(), count, MPI_INT64_T, gtopo.GetNeighborRank(n),
               rebalance_sizes_tag, MyComm, MPI_STATUS_IGNORE);
      const std::vector<std::int64_t> &mine = send[n];
      if (mine.size() != theirs.size()) { bad++; continue; }
      for (std::size_t pos = 0; pos < mine.size(); pos += 5)
      {
         bad += !std::equal(mine.begin() + pos, mine.begin() + pos + 5,
                            theirs.begin() + pos);
      }
   }
   MPI_Waitall(nnbr, req.data(), MPI_STATUSES_IGNORE);
   MPI_Allreduce(MPI_IN_PLACE, &bad, 1, MPI_LONG_LONG, MPI_SUM, MyComm);
   MFEM_VERIFY(bad == 0, "The mesh cannot be rebalanced: ranks that share"
               " entities list different groups, or different numbers of"
               " shared vertices, edges or faces in a group (" << bad
               << " differences). Such a mesh is not conforming across"
               " ranks.");
}

void ParMesh::GetRebalanceVertexIds(Array<std::int64_t> &vert_gid) const
{
   vert_gid.SetSize(NumOfVertices);
   for (int i = 0; i < NumOfVertices; i++)
   {
      vert_gid[i] = (std::int64_t(MyRank) << 32) + i;
   }
   GroupCommunicator svert_comm(gtopo);
   GetSharedVertexCommunicator(0, svert_comm);
   Array<int> svert_master_rank(svert_lvert.Size());
   Array<int> svert_master_index(svert_lvert);
   for (int g = 1; g < GetNGroups(); g++)
   {
      const int rank = gtopo.GetGroupMasterRank(g);
      for (int j = 0; j < group_svert.RowSize(g-1); j++)
      {
         svert_master_rank[group_svert.GetRow(g-1)[j]] = rank;
      }
   }
   svert_comm.Bcast(svert_master_index);
   for (int s = 0; s < svert_lvert.Size(); s++)
   {
      vert_gid[svert_lvert[s]] = (std::int64_t(svert_master_rank[s]) << 32) +
                                 svert_master_index[s];
   }
}

void ParMesh::CheckRebalanceSharedEntities(
   const Table &nbr_groups, const Array<std::int64_t> &vert_gid) const
{
   const int nnbr = gtopo.GetNumNeighbors();
   // The members of a group must describe its shared entities the same way:
   // the same edges and faces, given by the global ids of their vertices, in
   // the same order, and each with its vertices in the same order (the
   // orientation of a shared edge or face, and the refinement edge of a shared
   // face), whether or not the elements next to them are marked. The rest
   // relies on it, as do the finite element spaces on the mesh. A mesh that
   // fails it is not conforming across ranks, or will not be after a
   // conforming refinement; it is refused before anything moves. The numbers
   // of entities of each group were compared before
   // (CheckRebalanceGroupSizes()).
   auto Describe = [&](int g, std::vector<std::int64_t> &desc)
   {
      desc.push_back(group_svert.RowSize(g-1));
      desc.push_back(group_sedge.RowSize(g-1));
      desc.push_back((Dim == 3) ? group_stria.RowSize(g-1) : 0);
      for (int j = 0; j < group_sedge.RowSize(g-1); j++)
      {
         const int *v =
            shared_edges[group_sedge.GetRow(g-1)[j]]->GetVertices();
         desc.push_back(vert_gid[v[0]]);
         desc.push_back(vert_gid[v[1]]);
      }
      for (int j = 0; Dim == 3 && j < group_stria.RowSize(g-1); j++)
      {
         const int *v = shared_trias[group_stria.GetRow(g-1)[j]].v;
         desc.push_back(vert_gid[v[0]]);
         desc.push_back(vert_gid[v[1]]);
         desc.push_back(vert_gid[v[2]]);
      }
   };
   std::vector<std::vector<std::int64_t>> send(nnbr);
   int too_many = 0;
   for (int n = 1; n < nnbr; n++)
   {
      for (int k = 0; k < nbr_groups.RowSize(n); k++)
      {
         Describe(nbr_groups.GetRow(n)[k], send[n]);
      }
      if (TooLarge(0, static_cast<std::int64_t>(send[n].size())))
      {
         too_many = 1;
      }
   }
   MPI_Allreduce(MPI_IN_PLACE, &too_many, 1, MPI_INT, MPI_MAX, MyComm);
   MFEM_VERIFY(!too_many, too_large_message);
   std::vector<MPI_Request> req(nnbr, MPI_REQUEST_NULL);
   for (int n = 1; n < nnbr; n++)
   {
      MPI_Isend(send[n].data(), static_cast<int>(send[n].size()),
                MPI_INT64_T, gtopo.GetNeighborRank(n), rebalance_check_tag,
                MyComm, &req[n]);
   }
   // Shared edges or faces with other vertices; with the same vertices in
   // another order.
   long long bad[2] = {0, 0};
   std::vector<std::int64_t> theirs;
   for (int n = 1; n < nnbr; n++)
   {
      MPI_Status status;
      MPI_Probe(gtopo.GetNeighborRank(n), rebalance_check_tag, MyComm,
                &status);
      int count;
      MPI_Get_count(&status, MPI_INT64_T, &count);
      theirs.resize(count);
      MPI_Recv(theirs.data(), count, MPI_INT64_T, gtopo.GetNeighborRank(n),
               rebalance_check_tag, MyComm, MPI_STATUS_IGNORE);
      const std::vector<std::int64_t> &mine = send[n];
      bool same_shape = (mine.size() == theirs.size());
      for (std::size_t pos = 0; same_shape && pos < mine.size(); )
      {
         const std::int64_t *a = mine.data() + pos, *b = theirs.data() + pos;
         if (a[0] != b[0] || a[1] != b[1] || a[2] != b[2])
         {
            same_shape = false;
            break;
         }
         // An entity with other vertices, or with the same vertices in
         // another order.
         auto Compare = [&bad](const std::int64_t *x, const std::int64_t *y,
                               int len)
         {
            if (std::equal(x, x + len, y)) { return; }
            std::int64_t sx[3], sy[3];
            std::copy(x, x + len, sx);
            std::copy(y, y + len, sy);
            std::sort(sx, sx + len);
            std::sort(sy, sy + len);
            bad[std::equal(sx, sx + len, sy) ? 1 : 0]++;
         };
         for (std::int64_t j = 0; j < a[1]; j++)
         {
            Compare(a + 3 + 2*j, b + 3 + 2*j, 2);
         }
         for (std::int64_t j = 0; j < a[2]; j++)
         {
            Compare(a + 3 + 2*a[1] + 3*j, b + 3 + 2*a[1] + 3*j, 3);
         }
         pos += 3 + 2*a[1] + 3*a[2];
      }
      if (!same_shape) { bad[0]++; }
   }
   MPI_Waitall(nnbr, req.data(), MPI_STATUSES_IGNORE);
   MPI_Allreduce(MPI_IN_PLACE, bad, 2, MPI_LONG_LONG, MPI_SUM, MyComm);
   MFEM_VERIFY(bad[0] == 0 && bad[1] == 0, "The mesh cannot be rebalanced:"
               " ranks that share entities describe them differently ("
               << bad[0] << " shared edges or faces differ; " << bad[1]
               << " shared edges or faces have their vertices in another"
               " order: another orientation, or another refinement edge),"
               " compared through the shared vertices of the groups, which"
               " are taken as given. Such a mesh is not conforming across"
               " ranks, or will not be after a conforming refinement.");
}

int ParMesh::GetRebalanceTarget(const Array<int> &partition,
                                const Table &nbr_groups,
                                ConformingTarget &target) const
{
   const int nv_el = Dim + 1;            // vertices of an element
   const int ne_el = (Dim == 3) ? 6 : 3; // edges of an element
   const int nfaces = GetNumFaces();
   const int nnbr = gtopo.GetNumNeighbors();

   Array<int> vert_rows, vert_ranks, edge_rows, edge_ranks;
   vert_rows.Reserve(nv_el*NumOfElements);
   vert_ranks.Reserve(nv_el*NumOfElements);
   if (Dim == 3)
   {
      edge_rows.Reserve(ne_el*NumOfElements);
      edge_ranks.Reserve(ne_el*NumOfElements);
   }
   for (int e = 0; e < NumOfElements; e++)
   {
      const int *v = elements[e]->GetVertices();
      for (int j = 0; j < nv_el; j++)
      {
         vert_rows.Append(v[j]);
         vert_ranks.Append(partition[e]);
      }
      if (Dim == 3)
      {
         const int *ed = el_to_edge->GetRow(e);
         for (int j = 0; j < ne_el; j++)
         {
            edge_rows.Append(ed[j]);
            edge_ranks.Append(partition[e]);
         }
      }
   }
   // The vertices that belong to no element stay on this rank (and on the
   // other ranks that share them).
   {
      Array<bool> used(NumOfVertices);
      used = false;
      for (int e = 0; e < NumOfElements; e++)
      {
         const int *v = elements[e]->GetVertices();
         for (int j = 0; j < nv_el; j++) { used[v[j]] = true; }
      }
      for (int i = 0; i < NumOfVertices; i++)
      {
         if (!used[i])
         {
            target.isolated.Append(i);
            vert_rows.Append(i);
            vert_ranks.Append(MyRank);
         }
      }
   }
   target.vert_sets.Build(NumOfVertices, vert_rows, vert_ranks);
   if (Dim == 3) { target.edge_sets.Build(NumOfEdges, edge_rows, edge_ranks); }

   // Side 2 of the shared faces is set below, from the other rank.
   target.face_rank1.SetSize(nfaces);
   target.face_rank2.SetSize(nfaces);
   for (int f = 0; f < nfaces; f++)
   {
      const FaceInfo &fi = faces_info[f];
      target.face_rank1[f] = (fi.Elem1No >= 0) ? partition[fi.Elem1No] : -1;
      target.face_rank2[f] = (fi.Elem2No >= 0) ? partition[fi.Elem2No] : -1;
   }

   // Sizes that do not fit, found before the arrays are filled (see
   // max_entries and max_words): the call then stops on all ranks, before the
   // mesh changes.
   int too_large = 0;

   // Complete the sets of the shared entities over the current groups: each
   // member of a group sends its sets for the entities of the group (in the
   // group order, the same on all members) to the other members, the groups
   // in the order of nbr_groups.
   {
      // Shared edges are faces in 2D; shared triangles exist in 3D only.
      std::vector<Array<int>> send_buf(nnbr);
      std::vector<MPI_Request> send_req(nnbr, MPI_REQUEST_NULL);
      for (int n = 1; n < nnbr; n++)
      {
         Array<int> &buf = send_buf[n];
         for (int k = 0; k < nbr_groups.RowSize(n); k++)
         {
            const int g = nbr_groups.GetRow(n)[k];
            const int *sv = group_svert.GetRow(g-1);
            for (int j = 0; j < group_svert.RowSize(g-1); j++)
            {
               AppendSet(buf, target.vert_sets, svert_lvert[sv[j]]);
            }
            const int *se = group_sedge.GetRow(g-1);
            for (int j = 0; j < group_sedge.RowSize(g-1); j++)
            {
               const int edge = sedge_ledge[se[j]];
               if (Dim == 3) { AppendSet(buf, target.edge_sets, edge); }
               else { buf.Append(target.face_rank1[edge]); }
            }
            if (Dim == 3)
            {
               const int *st = group_stria.GetRow(g-1);
               for (int j = 0; j < group_stria.RowSize(g-1); j++)
               {
                  buf.Append(target.face_rank1[sface_lface[st[j]]]);
               }
            }
         }
         MPI_Isend(buf.GetData(), buf.Size(), MPI_INT,
                   gtopo.GetNeighborRank(n), rebalance_sets_tag, MyComm,
                   &send_req[n]);
      }
      Array<int> buf;
      for (int n = 1; n < nnbr; n++)
      {
         const int nbr_rank = gtopo.GetNeighborRank(n);
         MPI_Status status;
         MPI_Probe(nbr_rank, rebalance_sets_tag, MyComm, &status);
         int count;
         MPI_Get_count(&status, MPI_INT, &count);
         buf.SetSize(count);
         MPI_Recv(buf.GetData(), count, MPI_INT, nbr_rank, rebalance_sets_tag,
                  MyComm, MPI_STATUS_IGNORE);
         // Each entry of the message adds at most one (entity, rank) pair.
         if (TooLarge(std::int64_t(std::max(vert_rows.Size(), edge_rows.Size()))
                      + count, 0))
         {
            too_large = 1;
         }
         if (too_large) { continue; }
         int pos = 0;
         auto ReadSet = [&buf, &pos](int row, Array<int> &rows,
                                     Array<int> &ranks)
         {
            const int size = buf[pos++];
            for (int i = 0; i < size; i++)
            {
               rows.Append(row);
               ranks.Append(buf[pos++]);
            }
         };
         for (int k = 0; k < nbr_groups.RowSize(n); k++)
         {
            const int g = nbr_groups.GetRow(n)[k];
            const int *sv = group_svert.GetRow(g-1);
            for (int j = 0; j < group_svert.RowSize(g-1); j++)
            {
               ReadSet(svert_lvert[sv[j]], vert_rows, vert_ranks);
            }
            const int *se = group_sedge.GetRow(g-1);
            for (int j = 0; j < group_sedge.RowSize(g-1); j++)
            {
               const int edge = sedge_ledge[se[j]];
               if (Dim == 3) { ReadSet(edge, edge_rows, edge_ranks); }
               else { target.face_rank2[edge] = buf[pos++]; }
            }
            if (Dim == 3)
            {
               const int *st = group_stria.GetRow(g-1);
               for (int j = 0; j < group_stria.RowSize(g-1); j++)
               {
                  target.face_rank2[sface_lface[st[j]]] = buf[pos++];
               }
            }
         }
         MFEM_VERIFY(pos == buf.Size(), "internal error: message size");
      }
      MPI_Waitall(nnbr, send_req.data(), MPI_STATUSES_IGNORE);
   }
   target.vert_sets.Build(NumOfVertices, vert_rows, vert_ranks);
   vert_rows.DeleteAll();
   vert_ranks.DeleteAll();
   if (Dim == 3)
   {
      target.edge_sets.Build(NumOfEdges, edge_rows, edge_ranks);
      edge_rows.DeleteAll();
      edge_ranks.DeleteAll();
   }
   return too_large;
}

int ParMesh::PackRebalanceMessages(
   const Array<int> &partition, const Array<std::int64_t> &vert_gid,
   const ConformingTarget &target, std::int64_t nodal_row, int too_large,
   Array<int> &dests, std::vector<std::vector<std::int64_t>> &msgs) const
{
   const int nv_el = Dim + 1;            // vertices of an element
   const int nf_el = Dim + 1;            // faces (edges in 2D) of an element
   const int ne_el = (Dim == 3) ? 6 : 3; // edges of an element
   const int nv_be = Dim;                // vertices of a boundary element
   const Table &elem_face = (Dim == 3) ? *el_to_face : *el_to_edge;
   const int (*elem_edge_vert)[2] =
      (Dim == 3) ? Geometry::Constants<Geometry::TETRAHEDRON>::Edges :
      Geometry::Constants<Geometry::TRIANGLE>::Edges;

   // Each boundary element goes with the element on the side its orientation
   // agrees with (its normal points out of that element), as in the ParMesh
   // constructor in 3D; that element may be on the other side of a shared
   // face.
   Array<int> bdr_dest(NumOfBdrElements);
   for (int b = 0; b < NumOfBdrElements; b++)
   {
      int f, o;
      GetBdrElementFace(b, &f, &o);
      bdr_dest[b] = (o % 2 == 0 || target.face_rank2[f] < 0) ?
                    target.face_rank1[f] : target.face_rank2[f];
   }

   dests.Reserve(NumOfElements + NumOfBdrElements);
   dests.Append(partition);
   dests.Append(bdr_dest);
   if (target.isolated.Size()) { dests.Append(MyRank); }
   dests.Sort();
   dests.Unique();
   const int ndest = dests.Size();
   Table dest_elems, dest_bdr;
   dest_elems.MakeI(ndest);
   dest_bdr.MakeI(ndest);
   for (int e = 0; e < NumOfElements; e++)
   {
      dest_elems.AddAColumnInRow(dests.FindSorted(partition[e]));
   }
   for (int b = 0; b < NumOfBdrElements; b++)
   {
      dest_bdr.AddAColumnInRow(dests.FindSorted(bdr_dest[b]));
   }
   dest_elems.MakeJ();
   dest_bdr.MakeJ();
   for (int e = 0; e < NumOfElements; e++)
   {
      dest_elems.AddConnection(dests.FindSorted(partition[e]), e);
   }
   for (int b = 0; b < NumOfBdrElements; b++)
   {
      dest_bdr.AddConnection(dests.FindSorted(bdr_dest[b]), b);
   }
   dest_elems.ShiftUpI();
   dest_bdr.ShiftUpI();

   msgs.resize(ndest);
   Array<int> vert_mark(NumOfVertices), edge_mark(NumOfEdges);
   vert_mark = -1;
   edge_mark = -1;
   Array<std::int64_t> ints;
   Array<real_t> reals;
   Array<int> dest_verts, dest_edges;
   for (int k = 0; k < ndest; k++)
   {
      const int ne_k = dest_elems.RowSize(k), nb_k = dest_bdr.RowSize(k);
      const int *el_k = dest_elems.GetRow(k), *be_k = dest_bdr.GetRow(k);

      // Distinct vertices, and edges held by more than one rank, of the
      // elements sent to this destination; each edge with its vertices.
      dest_verts.SetSize(0);
      dest_edges.SetSize(0);
      for (int i = 0; i < ne_k; i++)
      {
         const int *v = elements[el_k[i]]->GetVertices();
         for (int j = 0; j < nv_el; j++)
         {
            if (vert_mark[v[j]] != k)
            {
               vert_mark[v[j]] = k;
               dest_verts.Append(v[j]);
            }
         }
         if (Dim == 3)
         {
            const int *ed = el_to_edge->GetRow(el_k[i]);
            for (int j = 0; j < ne_el; j++)
            {
               if (target.edge_sets.Size(ed[j]) > 1 && edge_mark[ed[j]] != k)
               {
                  edge_mark[ed[j]] = k;
                  dest_edges.Append(ed[j]);
                  dest_edges.Append(v[elem_edge_vert[j][0]]);
                  dest_edges.Append(v[elem_edge_vert[j][1]]);
               }
            }
         }
      }

      // The isolated vertices, last, in the message to this rank.
      const int n_iso = (dests[k] == MyRank) ? target.isolated.Size() : 0;
      dest_verts.Append(target.isolated.GetData(), n_iso);

      // The sizes of the message, before it is packed.
      std::int64_t n_ints = 5 + std::int64_t(ne_k)*(3 + nv_el + nf_el) +
                            std::int64_t(nb_k)*(1 + nv_be);
      for (int i = 0; i < dest_verts.Size(); i++)
      {
         const int n = target.vert_sets.Size(dest_verts[i]);
         n_ints += 2 + std::int64_t((n > 1) ? n : 0);
      }
      for (int i = 0; i < dest_edges.Size(); i += 3)
      {
         n_ints += 3 + std::int64_t(target.edge_sets.Size(dest_edges[i]));
      }
      const std::int64_t n_reals = std::int64_t(dest_verts.Size())*spaceDim;
      const std::int64_t n_words =
         2 + n_ints + (n_reals*std::int64_t(sizeof(real_t)) + 7)/8;
      // The message of the rows of the nodal space to this rank, sent by
      // the update of the nodes (ExchangeRebalanceRows()).
      const std::int64_t nodal_words =
         1 + std::int64_t(ne_k)*(1 + nodal_row);
      if (too_large || TooLarge(std::max(n_ints, n_reals), n_words) ||
          TooLarge(0, nodal_words))
      {
         too_large = 1;
         break;
      }

      ints.SetSize(0);
      reals.SetSize(0);
      ints.Reserve(static_cast<int>(n_ints));
      reals.Reserve(static_cast<int>(n_reals));
      ints.Append(ne_k);
      ints.Append(nb_k);
      ints.Append(dest_verts.Size());
      ints.Append(dest_edges.Size()/3);
      ints.Append(n_iso);
      for (int i = 0; i < ne_k; i++)
      {
         const int e = el_k[i];
         const Element *el = elements[e];
         ints.Append(e);
         ints.Append(el->GetAttribute());
         ints.Append((Dim == 3) ? static_cast<const Tetrahedron *>(el)->
                     GetRefinementFlag() : 0);
         const int *v = el->GetVertices();
         for (int j = 0; j < nv_el; j++) { ints.Append(vert_gid[v[j]]); }
         const int *ef = elem_face.GetRow(e);
         for (int j = 0; j < nf_el; j++)
         {
            const int f = ef[j];
            ints.Append((faces_info[f].Elem1No == e) ? target.face_rank2[f] :
                        target.face_rank1[f]);
         }
      }
      for (int i = 0; i < nb_k; i++)
      {
         const Element *be = boundary[be_k[i]];
         ints.Append(be->GetAttribute());
         const int *v = be->GetVertices();
         for (int j = 0; j < nv_be; j++) { ints.Append(vert_gid[v[j]]); }
      }
      for (int i = 0; i < dest_verts.Size(); i++)
      {
         const int v = dest_verts[i];
         const int n = target.vert_sets.Size(v);
         ints.Append(vert_gid[v]);
         ints.Append((n > 1) ? n : 0);
         for (int j = 0; n > 1 && j < n; j++)
         {
            ints.Append(target.vert_sets.Row(v)[j]);
         }
         reals.Append(vertices[v](), spaceDim);
      }
      for (int i = 0; i < dest_edges.Size(); i += 3)
      {
         const int edge = dest_edges[i];
         std::int64_t a = vert_gid[dest_edges[i+1]];
         std::int64_t b = vert_gid[dest_edges[i+2]];
         if (b < a) { std::swap(a, b); }
         ints.Append(a);
         ints.Append(b);
         ints.Append(target.edge_sets.Size(edge));
         for (int j = 0; j < target.edge_sets.Size(edge); j++)
         {
            ints.Append(target.edge_sets.Row(edge)[j]);
         }
      }
      PackMessage(ints, reals, msgs[k]);
      MFEM_VERIFY(std::int64_t(msgs[k].size()) == n_words,
                  "internal error: message size");
   }
   return too_large;
}

void ParMesh::DecodeRebalanceMessages(
   std::map<int, std::vector<std::int64_t>> &recv_msgs,
   ConformingRecords &rec) const
{
   const int nv_el = Dim + 1, nf_el = Dim + 1, nv_be = Dim;
   for (auto it = recv_msgs.begin(); it != recv_msgs.end();
        it = recv_msgs.erase(it))
   {
      const int src = it->first;
      const std::vector<std::int64_t> &msg = it->second;
      MFEM_VERIFY(msg.size() >= 7, "internal error: message size");
      const std::int64_t n_ints = msg[0], n_reals = msg[1];
      const std::int64_t *ip = msg.data() + 2, *ip_end = ip + n_ints;
      // The reals follow the integers as bytes (see PackMessage): copy them
      // into real_t storage instead of reading them through a real_t pointer.
      const std::size_t real_bytes =
         static_cast<std::size_t>(n_reals) * sizeof(real_t);
      MFEM_VERIFY(n_ints >= 0 && n_reals >= 0 &&
                  (2 + n_ints) * sizeof(std::int64_t) + real_bytes <=
                  msg.size() * sizeof(std::int64_t),
                  "internal error: message size");
      Array<real_t> reals(static_cast<int>(n_reals));
      if (real_bytes) { std::memcpy(reals.GetData(), ip_end, real_bytes); }
      const real_t *rp = reals.GetData();
      const real_t *rp_end = rp + n_reals;
      const int ne_m = static_cast<int>(*ip++), nb_m = static_cast<int>(*ip++);
      const int nv_m = static_cast<int>(*ip++), ned_m = static_cast<int>(*ip++);
      const int niso_m = static_cast<int>(*ip++);
      for (int i = 0; i < ne_m; i++)
      {
         rec.elem_old_rank.Append(src);
         rec.elem_old_index.Append(static_cast<int>(*ip++));
         rec.elem_attr.Append(static_cast<int>(*ip++));
         rec.elem_flag.Append(static_cast<int>(*ip++));
         rec.elem_vert.Append(ip, nv_el);
         ip += nv_el;
         for (int j = 0; j < nf_el; j++)
         {
            rec.elem_nbr.Append(static_cast<int>(*ip++));
         }
      }
      for (int i = 0; i < nb_m; i++)
      {
         rec.bdr_attr.Append(static_cast<int>(*ip++));
         rec.bdr_vert.Append(ip, nv_be);
         ip += nv_be;
      }
      for (int i = 0; i < nv_m; i++)
      {
         rec.vert_id.Append(*ip++);
         const int n = static_cast<int>(*ip++);
         for (int j = 0; j < n; j++)
         {
            rec.vert_set.Append(static_cast<int>(*ip++));
         }
         rec.vert_set_offset.Append(rec.vert_set.Size());
         rec.vert_coord.Append(rp, spaceDim);
         rp += spaceDim;
      }
      for (int i = 0; i < ned_m; i++)
      {
         rec.edge_vert.Append(ip, 2);
         ip += 2;
         const int n = static_cast<int>(*ip++);
         for (int j = 0; j < n; j++)
         {
            rec.edge_set.Append(static_cast<int>(*ip++));
         }
         rec.edge_set_offset.Append(rec.edge_set.Size());
      }
      // The isolated vertices are the last niso_m vertex records.
      for (int i = nv_m - niso_m; i < nv_m; i++)
      {
         rec.iso_vert.Append(rec.vert_id[rec.vert_id.Size() - nv_m + i]);
      }
      MFEM_VERIFY(ip == ip_end && rp == rp_end,
                  "internal error: message from rank " << src);
   }
}

void ParMesh::BuildFromConformingRecords(ConformingRecords &rec,
                                         const Array<int> &partition,
                                         const Array<int> &isolated)
{
   BuildLocalMeshFromRecords(rec);
   BuildGroupsFromRecords(rec);
   PublishRebalance(rec, partition, isolated);
}

void ParMesh::BuildLocalMeshFromRecords(ConformingRecords &rec)
{
   const int nv_el = Dim + 1, nv_be = Dim;
   const int ne = rec.elem_attr.Size(), nbe = rec.bdr_attr.Size();
   const Geometry::Type elem_geom =
      (Dim == 3) ? Geometry::TETRAHEDRON : Geometry::TRIANGLE;
   const Geometry::Type bdr_geom =
      (Dim == 3) ? Geometry::TRIANGLE : Geometry::SEGMENT;

   // Vertices: the distinct global ids, in increasing order, so that the local
   // vertex numbers increase with the global ids on every rank.
   {
      Array<int> order(rec.vert_id.Size());
      for (int i = 0; i < order.Size(); i++) { order[i] = i; }
      std::stable_sort(order.begin(), order.end(), [&rec](int a, int b)
      { return rec.vert_id[a] < rec.vert_id[b]; });
      for (int i = 0; i < order.Size(); i++)
      {
         const std::int64_t id = rec.vert_id[order[i]];
         if (rec.new_vert_id.Size() == 0 || rec.new_vert_id.Last() != id)
         {
            rec.new_vert_id.Append(id);
            rec.new_vert_rec.Append(order[i]);
         }
      }
   }
   const int nv = rec.new_vert_id.Size();

   // The new local mesh. FinalizeTopology() keeps the vertex order of the
   // elements and of the boundary elements (refinement edges, normals).
   {
      Mesh mesh(Dim, nv, ne, nbe, spaceDim);
      for (int i = 0; i < nv; i++)
      {
         mesh.AddVertex(rec.vert_coord.GetData() +
                        spaceDim*rec.new_vert_rec[i]);
      }
      int v[4];
      for (int e = 0; e < ne; e++)
      {
         Element *el = mesh.NewElement(elem_geom);
         for (int j = 0; j < nv_el; j++)
         {
            v[j] = rec.LocalVertex(rec.elem_vert[nv_el*e + j]);
         }
         el->SetVertices(v);
         el->SetAttribute(rec.elem_attr[e]);
         if (Dim == 3)
         {
            static_cast<Tetrahedron *>(el)->SetRefinementFlag(rec.elem_flag[e]);
         }
         mesh.AddElement(el);
      }
      for (int b = 0; b < nbe; b++)
      {
         Element *be = mesh.NewElement(bdr_geom);
         for (int j = 0; j < nv_be; j++)
         {
            v[j] = rec.LocalVertex(rec.bdr_vert[nv_be*b + j]);
         }
         be->SetVertices(v);
         be->SetAttribute(rec.bdr_attr[b]);
         mesh.AddBdrElement(be);
      }
      mesh.FinalizeTopology(false);

      // The attribute lists, the attribute sets and 'meshgen' are global.
      attributes.Copy(mesh.attributes);
      bdr_attributes.Copy(mesh.bdr_attributes);
      attribute_sets.Copy(mesh.attribute_sets);
      bdr_attribute_sets.Copy(mesh.bdr_attribute_sets);
      const int glob_meshgen = meshgen;
      Mesh::Swap(mesh, false);
      meshgen = glob_meshgen;
   } // the previous local mesh is deleted here
   ResetLazyData();
   CoarseFineTr.Clear();
}

std::set<std::vector<int>> ParMesh::ConformingRecords::SharedGroups(
                           int rank, int dim) const
{
   std::set<std::vector<int>> sets;
   auto Add = [&sets](const Array<int> &offset, const Array<int> &set)
   {
      for (int i = 0; i + 1 < offset.Size(); i++)
      {
         if (offset[i+1] - offset[i] > 1)
         {
            sets.emplace(set.GetData() + offset[i],
                         set.GetData() + offset[i+1]);
         }
      }
   };
   Add(vert_set_offset, vert_set);
   if (dim == 3) { Add(edge_set_offset, edge_set); }
   for (int q : elem_nbr)
   {
      if (q >= 0 && q != rank) { sets.insert(PairSet(rank, q)); }
   }
   return sets;
}

/* GroupTopology::Create() lists the ranks of each group (group_lproc, and
   group_mgroupandproc with one more entry per group, the local group
   included), the groups that each neighbor shares with the rank
   (lproc_cgroup: for a group it masters, its other ranks; for another, its
   master), and, in one buffer for all the neighbors, the descriptions of the
   groups, s + 2 entries for a group of s ranks, that the master of each group
   (its smallest rank) sends to each of its other ranks: (s - 1)(s + 2)
   entries on the master, s + 2 on each other rank, and two rows per neighbor
   (2N - 1 row offsets for the N ranks of the groups, the rank included). The
   master of a group of s ranks holds of the order of s^2 entries for it,
   which the messages of the rebalance do not bound. */
std::int64_t ParMesh::ConformingRecords::GroupTopologyEntries(
   int rank, const std::set<std::vector<int>> &groups)
{
   const std::int64_t ngroups = std::int64_t(groups.size()) + 1;
   std::int64_t ranks = 1, neighbors = 0, buffer = 0;
   std::set<int> lprocs = {rank};
   for (const std::vector<int> &g : groups)
   {
      const std::int64_t s = std::int64_t(g.size());
      const bool master = (g[0] == rank);
      ranks += s;
      neighbors += master ? s - 1 : 1;
      buffer += master ? (s - 1)*(s + 2) : s + 2;
      lprocs.insert(g.begin(), g.end());
   }
   const std::int64_t buffer_rows = 2*std::int64_t(lprocs.size()) - 1;
   return std::max({ranks + ngroups, neighbors, buffer, buffer_rows});
}

void ParMesh::BuildGroupsFromRecords(ConformingRecords &rec)
{
   const int nf_el = Dim + 1;
   const int ne = rec.elem_attr.Size();
   const int nv = rec.new_vert_id.Size();

   // Shared edges (3D): local vertices, smaller first, and set of ranks.
   struct SharedEdge { int v0, v1; std::vector<int> set; };
   std::vector<SharedEdge> sedges;
   if (Dim == 3)
   {
      const int nrec = rec.edge_vert.Size()/2;
      Array<int> order(nrec);
      for (int i = 0; i < nrec; i++) { order[i] = i; }
      const std::int64_t *ev = rec.edge_vert.GetData();
      std::sort(order.begin(), order.end(), [ev](int a, int b)
      {
         return (ev[2*a] != ev[2*b]) ? (ev[2*a] < ev[2*b]) :
                (ev[2*a+1] < ev[2*b+1]);
      });
      for (int i = 0; i < nrec; i++)
      {
         const int r = order[i], p = (i > 0) ? order[i-1] : -1;
         if (p >= 0 && ev[2*p] == ev[2*r] && ev[2*p+1] == ev[2*r+1])
         {
            continue;
         }
         const int *s = rec.edge_set.GetData();
         sedges.push_back({rec.LocalVertex(ev[2*r]),
                           rec.LocalVertex(ev[2*r+1]),
                           std::vector<int>(s + rec.edge_set_offset[r],
                                            s + rec.edge_set_offset[r+1])});
      }
   }

   // Shared faces (edges in 2D), from the rank on the other side of each face
   // of the elements. Their vertex order must be the same on both ranks: in 2D
   // the smaller global id first; in 3D the face as seen from the element of
   // the lower rank, either with its refinement edge first (the marked face,
   // see ParMesh::RefineGroups) or, if that element is not marked, from its
   // vertex of smallest global id. The lower rank derives it here, and the
   // higher rank adopts it (OrientSharedTriangles(); its own element can be
   // marked differently).
   struct SharedFace { int v[3]; int key[3]; int rank; };
   std::vector<SharedFace> sfaces;
   for (int e = 0; e < ne; e++)
   {
      const int *v = elements[e]->GetVertices();
      for (int j = 0; j < nf_el; j++)
      {
         const int q = rec.elem_nbr[nf_el*e + j];
         if (q < 0 || q == MyRank) { continue; }
         SharedFace sf;
         sf.rank = q;
         if (Dim == 2)
         {
            const int *ev = Geometry::Constants<Geometry::TRIANGLE>::Edges[j];
            sf.v[0] = std::min(v[ev[0]], v[ev[1]]);
            sf.v[1] = std::max(v[ev[0]], v[ev[1]]);
            sf.v[2] = -1;
         }
         else
         {
            const int *fv =
               Geometry::Constants<Geometry::TETRAHEDRON>::FaceVert[j];
            for (int k = 0; k < 3; k++) { sf.v[k] = v[fv[k]]; }
            const Tetrahedron *tet =
               static_cast<const Tetrahedron *>(elements[e]);
            if (MyRank < q)
            {
               if (tet->GetRefinementFlag()) { tet->GetMarkedFace(j, sf.v); }
               else { RotateToSmallestFirst(sf.v); }
            }
         }
         for (int k = 0; k < 3; k++) { sf.key[k] = sf.v[k]; }
         SortSmall(sf.key, (Dim == 2) ? 2 : 3);
         sfaces.push_back(sf);
      }
   }

   // The groups: the sets of ranks of the shared entities, including this
   // rank, numbered in their lexicographic order after the local group 0.
   std::map<std::vector<int>, int> group_ids;
   for (int i = 0; i < nv; i++)
   {
      if (rec.SharedVertex(i)) { group_ids.emplace(rec.VertexSet(i), 0); }
   }
   for (const SharedEdge &se : sedges) { group_ids.emplace(se.set, 0); }
   for (const SharedFace &sf : sfaces)
   {
      group_ids.emplace(PairSet(MyRank, sf.rank), 0);
   }
#ifdef MFEM_DEBUG
   {
      // The groups are those whose tables were counted before the mesh was
      // replaced (RebalanceConforming()).
      bool counted = (group_ids.size() == rec.groups.size());
      auto it = rec.groups.begin();
      for (const auto &gi : group_ids)
      {
         if (!counted) { break; }
         counted = (gi.first == *it++);
      }
      MFEM_VERIFY(counted, "internal error: groups not counted");
   }
#endif
   ListOfIntegerSets groups;
   {
      IntegerSet group;
      group.Recreate(1, &MyRank);
      groups.Insert(group);
      int g = 1;
      for (auto &gi : group_ids)
      {
         gi.second = g++;
         group.Recreate(static_cast<int>(gi.first.size()), gi.first.data());
         groups.Insert(group);
      }
   }
   const int ngroups = groups.Size();
   gtopo.Create(groups, rebalance_groups_tag);
#ifdef MFEM_DEBUG
   {
      // The largest array of GroupTopology::Create() on this rank, from the
      // topology it made, is the one counted before the mesh was replaced.
      std::int64_t ranks = 0, conn = 0, buf = 0;
      for (int g = 0; g < ngroups; g++)
      {
         const std::int64_t s = gtopo.GetGroupSize(g);
         ranks += s;
         if (g == 0) { continue; }
         conn += gtopo.IAmMaster(g) ? s - 1 : 1;
         buf += gtopo.IAmMaster(g) ? (s - 1)*(s + 2) : s + 2;
      }
      const std::int64_t rows = 2*std::int64_t(gtopo.GetNumNeighbors()) - 1;
      const std::int64_t made = std::max({ranks + ngroups, conn, buf, rows});
      MFEM_VERIFY(made == ConformingRecords::GroupTopologyEntries(MyRank,
                                                                  rec.groups),
                  "internal error: group tables not counted");
   }
#endif

   // The shared entities of each group, sorted by their vertex numbers (which
   // increase with the global ids), with consecutive numbers.
   auto MakeGroupTable = [ngroups](Table &t, const Array<int> &count)
   {
      t.SetDims(ngroups - 1, count.Sum());
      int *I = t.GetI(), *J = t.GetJ();
      I[0] = 0;
      for (int g = 1; g < ngroups; g++) { I[g] = I[g-1] + count[g]; }
      for (int k = 0; k < I[ngroups-1]; k++) { J[k] = k; }
   };
   for (int i = 0; i < shared_edges.Size(); i++)
   {
      FreeElement(shared_edges[i]);
   }
   shared_edges.SetSize(0);
   shared_trias.SetSize(0);
   shared_quads.SetSize(0);

   {
      Array<int> vert_group(nv), count(ngroups), pos(ngroups);
      count = 0;
      for (int i = 0; i < nv; i++)
      {
         vert_group[i] = rec.SharedVertex(i) ? group_ids[rec.VertexSet(i)] : 0;
         count[vert_group[i]]++;
      }
      count[0] = 0;
      MakeGroupTable(group_svert, count);
      svert_lvert.SetSize(group_svert.Size_of_connections());
      for (int g = 1; g < ngroups; g++) { pos[g] = group_svert.GetI()[g-1]; }
      for (int i = 0; i < nv; i++)
      {
         if (vert_group[i]) { svert_lvert[pos[vert_group[i]]++] = i; }
      }
   }

   {
      struct Entry { int group, v0, v1; };
      std::vector<Entry> entries;
      for (const SharedEdge &se : sedges)
      {
         entries.push_back({group_ids[se.set], se.v0, se.v1});
      }
      for (const SharedFace &sf : sfaces)
      {
         if (Dim == 2)
         {
            entries.push_back({group_ids[PairSet(MyRank, sf.rank)], sf.v[0],
                               sf.v[1]});
         }
      }
      std::sort(entries.begin(), entries.end(),
                [](const Entry &a, const Entry &b)
      {
         return (a.group != b.group) ? (a.group < b.group) :
                (a.v0 != b.v0) ? (a.v0 < b.v0) : (a.v1 < b.v1);
      });
      Array<int> count(ngroups);
      count = 0;
      for (const Entry &en : entries) { count[en.group]++; }
      MakeGroupTable(group_sedge, count);
      shared_edges.SetSize(static_cast<int>(entries.size()));
      for (int k = 0; k < shared_edges.Size(); k++)
      {
         shared_edges[k] = new Segment(entries[k].v0, entries[k].v1, 1);
      }
   }

   {
      Array<int> count(ngroups);
      count = 0;
      if (Dim == 3)
      {
         std::vector<int> face_group(sfaces.size()), order(sfaces.size());
         for (std::size_t k = 0; k < sfaces.size(); k++)
         {
            face_group[k] = group_ids[PairSet(MyRank, sfaces[k].rank)];
            order[k] = static_cast<int>(k);
         }
         std::sort(order.begin(), order.end(), [&](int a, int b)
         {
            if (face_group[a] != face_group[b])
            {
               return face_group[a] < face_group[b];
            }
            return std::lexicographical_compare(sfaces[a].key,
                                                sfaces[a].key + 3,
                                                sfaces[b].key,
                                                sfaces[b].key + 3);
         });
         shared_trias.SetSize(static_cast<int>(sfaces.size()));
         for (int k = 0; k < shared_trias.Size(); k++)
         {
            shared_trias[k].Set(sfaces[order[k]].v);
            count[face_group[order[k]]]++;
         }

         std::map<int, std::vector<int>> faces_with;
         for (int k = 0; k < shared_trias.Size(); k++)
         {
            faces_with[sfaces[order[k]].rank].push_back(k);
         }
         OrientSharedTriangles(faces_with, rec);
      }
      MakeGroupTable(group_stria, count);
      count = 0;
      MakeGroupTable(group_squad, count);
   }
   FinalizeParTopo();
}

void ParMesh::OrientSharedTriangles(
   const std::map<int, std::vector<int>> &faces_with,
   const ConformingRecords &rec)
{
   // The lower rank of each pair sends its descriptions of their shared
   // faces, in the group order (the same on both ranks), and the higher
   // rank adopts them.
   std::vector<std::vector<std::int64_t>> send;
   std::vector<MPI_Request> req;
   send.reserve(faces_with.size());
   req.reserve(faces_with.size());
   for (const auto &fw : faces_with)
   {
      if (fw.first < MyRank) { continue; }
      send.emplace_back();
      for (int k : fw.second)
      {
         for (int i = 0; i < 3; i++)
         {
            send.back().push_back(rec.new_vert_id[shared_trias[k].v[i]]);
         }
      }
      req.emplace_back();
      MPI_Isend(send.back().data(), static_cast<int>(send.back().size()),
                MPI_INT64_T, fw.first, rebalance_faces_tag, MyComm,
                &req.back());
   }
   std::vector<std::int64_t> recv;
   for (const auto &fw : faces_with)
   {
      if (fw.first > MyRank) { continue; }
      recv.resize(3*fw.second.size());
      MPI_Recv(recv.data(), static_cast<int>(recv.size()), MPI_INT64_T,
               fw.first, rebalance_faces_tag, MyComm, MPI_STATUS_IGNORE);
      for (std::size_t j = 0; j < fw.second.size(); j++)
      {
         int *v = shared_trias[fw.second[j]].v;
         int w[3] = {rec.LocalVertex(recv[3*j]),
                     rec.LocalVertex(recv[3*j+1]),
                     rec.LocalVertex(recv[3*j+2])
                    };
         int a[3] = {v[0], v[1], v[2]}, b[3] = {w[0], w[1], w[2]};
         SortSmall(a, 3);
         SortSmall(b, 3);
         MFEM_VERIFY(std::equal(a, a + 3, b),
                     "internal error: shared face from rank "
                     << fw.first);
         for (int i = 0; i < 3; i++) { v[i] = w[i]; }
      }
   }
   MPI_Waitall(static_cast<int>(req.size()), req.data(),
               MPI_STATUSES_IGNORE);
}

void ParMesh::PublishRebalance(ConformingRecords &rec,
                               const Array<int> &partition,
                               const Array<int> &isolated)
{
   rec.iso_new.SetSize(rec.iso_vert.Size());
   for (int i = 0; i < rec.iso_vert.Size(); i++)
   {
      rec.iso_new[i] = rec.LocalVertex(rec.iso_vert[i]);
   }

   have_face_nbr_data = false;
   face_nbr_el_to_face.reset();
   face_nbr_el_ori.reset();
   last_operation = Mesh::REBALANCE;
   sequence++;

   // The migration, published before the nodal space is updated: the previous
   // location of each element (GetRebalanceOldElements()), and the schedule of
   // the transfer of grid functions (ParFiniteElementSpace::Update()).
   rebalance_old_rank.Swap(rec.elem_old_rank);
   rebalance_old_index.Swap(rec.elem_old_index);
   rebalance_dest = partition;
   rebalance_iso_old = isolated;
   rebalance_iso_new.Swap(rec.iso_new);
   rebalance_sequence = sequence;

   // Nodes: the same grid function and space, updated for the new mesh with
   // the transfer operator of the space, as after a refinement, so that the
   // other grid functions on the nodal space follow with Update().
   if (Nodes)
   {
      ParFiniteElementSpace *nodes_fes =
         static_cast<ParFiniteElementSpace *>(Nodes->FESpace());
      nodes_fes->Update();
      Nodes->Update();
      // As after the other mesh modifications, see UpdateNodes(). That takes
      // the vertex coordinates from the elements: the vertices of no element
      // keep the coordinates they arrived with.
      SetVerticesFromNodes(Nodes);
      for (int i = 0; i < rec.iso_vert.Size(); i++)
      {
         const int v = rec.LocalVertex(rec.iso_vert[i]);
         vertices[v].SetCoords(spaceDim, rec.vert_coord.GetData() +
                               spaceDim*rec.new_vert_rec[v]);
      }
      NodesUpdated();
   }
}

void ParMesh::ExchangeRebalanceRows(
   const Array<std::int64_t> &row_sizes,
   const std::function<void(int, std::int64_t *)> &fill,
   Array<int> &new_offsets, Array<std::int64_t> &new_rows) const
{
   MFEM_VERIFY(rebalance_sequence == sequence &&
               row_sizes.Size() == rebalance_dest.Size(),
               "The mesh was modified after Rebalance(), or the rows do not "
               "belong to its elements before the call.");
   const int old_ne = rebalance_dest.Size();

   // One message per new owner: the number of rows, then each row (its length,
   // its entries), in the order of the old elements. The sizes are counted in
   // 64 bits and checked on all ranks before the messages are filled; a row
   // longer than a message stops the count, which therefore cannot overflow.
   Array<int> dests(rebalance_dest);
   dests.Sort();
   dests.Unique();
   std::vector<std::int64_t> words(dests.Size(), 1);
   int too_large = 0;
   for (int i = 0; i < old_ne && !too_large; i++)
   {
      MFEM_VERIFY(row_sizes[i] >= 0, "internal error: row size");
      if (TooLarge(0, row_sizes[i])) { too_large = 1; }
      else { words[dests.FindSorted(rebalance_dest[i])] += 1 + row_sizes[i]; }
   }
   for (std::int64_t w : words) { if (TooLarge(0, w)) { too_large = 1; } }
   MPI_Allreduce(MPI_IN_PLACE, &too_large, 1, MPI_INT, MPI_MAX, MyComm);
   MFEM_VERIFY(!too_large, too_large_message);
   std::vector<std::vector<std::int64_t>> msgs(dests.Size());
   for (int k = 0; k < dests.Size(); k++)
   {
      msgs[k].reserve(words[k]);
      msgs[k].push_back(0);
   }
   for (int i = 0; i < old_ne; i++)
   {
      std::vector<std::int64_t> &m = msgs[dests.FindSorted(rebalance_dest[i])];
      m[0]++;
      m.push_back(row_sizes[i]);
      const std::size_t start = m.size();
      m.resize(start + row_sizes[i]);
      fill(i, m.data() + start);
   }
   std::map<int, std::vector<std::int64_t>> recv;
   ExchangeMessages(MyComm, dests, msgs, recv, rebalance_rows_tag);

   // The entries of the rows received (each message: the number of rows, then
   // a length and the entries of each row) must fit an array, on all ranks,
   // before it is allocated, to its exact size.
   std::int64_t total = 0;
   for (const auto &m : recv)
   {
      const std::int64_t size = static_cast<std::int64_t>(m.second.size());
      MFEM_VERIFY(size >= 1 && m.second[0] >= 0 && m.second[0] < size,
                  "internal error: rows from rank " << m.first);
      total += size - 1 - m.second[0];
   }
   too_large = TooLarge(std::max(total, std::int64_t(NumOfElements) + 1), 0);
   MPI_Allreduce(MPI_IN_PLACE, &too_large, 1, MPI_INT, MPI_MAX, MyComm);
   MFEM_VERIFY(!too_large, too_large_message);

   // The local elements are in the order of their previous rank, then of
   // their previous index: the rows of each source, in order.
   std::map<int, std::size_t> pos;
   std::map<int, std::int64_t> nrows;
   for (const auto &m : recv)
   {
      pos[m.first] = 1;
      nrows[m.first] = 0;
   }
   new_offsets.SetSize(NumOfElements + 1);
   new_offsets[0] = 0;
   new_rows.SetSize(static_cast<int>(total));
   int filled = 0;
   for (int e = 0; e < NumOfElements; e++)
   {
      const int src = rebalance_old_rank[e];
      auto it = recv.find(src);
      MFEM_VERIFY(it != recv.end(),
                  "internal error: no rows from rank " << src);
      const std::vector<std::int64_t> &m = it->second;
      std::size_t &p = pos[src];
      MFEM_VERIFY(p < m.size(), "internal error: rows from rank " << src);
      const std::int64_t n = m[p++];
      MFEM_VERIFY(n >= 0 && p + n <= m.size() && filled + n <= total,
                  "internal error: rows from rank " << src);
      std::copy(m.data() + p, m.data() + p + n, new_rows.GetData() + filled);
      filled += static_cast<int>(n);
      p += n;
      nrows[src]++;
      new_offsets[e+1] = filled;
   }
   MFEM_VERIFY(filled == total, "internal error: rows");
   for (const auto &m : recv)
   {
      MFEM_VERIFY(pos[m.first] == m.second.size() &&
                  nrows[m.first] == m.second[0],
                  "internal error: rows from rank " << m.first);
   }
}

} // namespace mfem

#endif // MFEM_USE_MPI
