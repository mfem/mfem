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

#include "../config/config.hpp"

#ifdef MFEM_USE_MPI

#include "mesh_headers.hpp"
#include "pncmesh.hpp"
#include "../general/binaryio.hpp"

#include <map>
#include <climits> // INT_MIN, INT_MAX

namespace mfem
{

using namespace bin_io;

ParNCMesh::ParNCMesh(MPI_Comm comm, const NCMesh &ncmesh, int *part)
   : NCMesh(ncmesh)
{
   MyComm = comm;
   MPI_Comm_size(MyComm, &NRanks);
   MPI_Comm_rank(MyComm, &MyRank);

   // assign leaf elements to the processors by simply splitting the
   // sequence of leaf elements into 'NRanks' parts
   for (int i = 0; i < leaf_elements.Size(); i++)
   {
      elements[leaf_elements[i]].rank = part ? part[i] : InitialPartition(i);
   }

   Update();

   // note that at this point all processors still have all the leaf elements;
   // we however may now start pruning the refinement tree to get rid of
   // branches that only contain someone else's leaves (see Prune())
}

ParNCMesh::ParNCMesh(const ParNCMesh &other)
// copy primary data only
   : NCMesh(other)
   , MyComm(other.MyComm)
   , NRanks(other.NRanks)
   , MyRank(other.MyRank)
{
   Update(); // mark all secondary stuff for recalculation
}

ParNCMesh::~ParNCMesh()
{
   ClearAuxPM();
}

void ParNCMesh::Update()
{
   NCMesh::Update();

   groups.clear();
   group_id.clear();

   CommGroup self;
   self.push_back(MyRank);
   groups.push_back(self);
   group_id[self] = 0;

   for (int i = 0; i < 3; i++)
   {
      entity_owner[i].DeleteAll();
      entity_pmat_group[i].DeleteAll();
      entity_index_rank[i].DeleteAll();
   }

   shared_vertices.Clear();
   shared_edges.Clear();
   shared_faces.Clear();

   element_type.SetSize(0);
   ghost_layer.SetSize(0);
   boundary_layer.SetSize(0);
}

void ParNCMesh::AssignLeafIndices()
{
   // This is an override of NCMesh::AssignLeafIndices(). The difference is
   // that we shift all elements we own to the beginning of the array
   // 'leaf_elements' and assign all ghost elements indices >= NElements.

   // Also note that the ordering of ghosts and non-ghosts is preserved here,
   // which is important for ParNCMesh::GetFaceNeighbors.

   // We store the original leaf ordering in 'leaf_glob_order'. This is later
   // used (and deleted) in GetConformingSharedStructures

   NCMesh::AssignLeafIndices(); // original numbering, for 'leaf_glob_order'

   int nleafs = leaf_elements.Size();

   Array<int> ghosts;
   ghosts.Reserve(nleafs);

   NElements = 0;
   for (int i = 0; i < nleafs; i++)
   {
      int elem = leaf_elements[i];
      if (elements[elem].rank == MyRank)
      {
         leaf_elements[NElements++] = elem;
      }
      else
      {
         ghosts.Append(elem);
      }
   }
   NGhostElements = ghosts.Size();

   leaf_elements.SetSize(NElements);
   leaf_elements.Append(ghosts);

   // store original (globally consistent) numbering in 'leaf_glob_order'
   leaf_glob_order.SetSize(nleafs);
   for (int i = 0; i < nleafs; i++)
   {
      leaf_glob_order[i] = elements[leaf_elements[i]].index;
   }

   // new numbering with ghost shifted to the back
   NCMesh::AssignLeafIndices();
}

void ParNCMesh::UpdateVertices()
{
   // This is an override of NCMesh::UpdateVertices. This version first
   // assigns vert_index to vertices of elements of our rank. Only these
   // vertices then make it to the Mesh in NCMesh::GetMeshComponents.
   // The remaining (ghost) vertices are assigned indices greater or equal to
   // Mesh::GetNV().

   for (node_iterator node = nodes.begin(); node != nodes.end(); ++node)
   {
      if (node->HasVertex()) { node->vert_index = -1; }
   }

   NVertices = 0;
   for (int i = 0; i < leaf_elements.Size(); i++)
   {
      Element &el = elements[leaf_elements[i]];
      if (el.rank == MyRank)
      {
         for (int j = 0; j < GI[el.Geom()].nv; j++)
         {
            int &vindex = nodes[el.node[j]].vert_index;
            if (vindex < 0) { vindex = NVertices++; }
         }
      }
   }

   vertex_nodeId.SetSize(NVertices);
   for (node_iterator node = nodes.begin(); node != nodes.end(); ++node)
   {
      if (node->HasVertex() && node->vert_index >= 0)
      {
         vertex_nodeId[node->vert_index] = node.index();
      }
   }

   NGhostVertices = 0;
   for (node_iterator node = nodes.begin(); node != nodes.end(); ++node)
   {
      if (node->HasVertex() && node->vert_index < 0)
      {
         node->vert_index = NVertices + (NGhostVertices++);
      }
   }
}

void ParNCMesh::OnMeshUpdated(Mesh *mesh)
{
   // This is an override (or extension of) NCMesh::OnMeshUpdated().
   // In addition to getting edge/face indices from 'mesh', we also
   // assign indices to ghost edges/faces that don't exist in the 'mesh'.

   // clear edge_index and Face::index
   for (node_iterator node = nodes.begin(); node != nodes.end(); ++node)
   {
      if (node->HasEdge()) { node->edge_index = -1; }
   }
   for (face_iterator face = faces.begin(); face != faces.end(); ++face)
   {
      face->index = -1;
   }

   // go assign existing edge/face indices
   NCMesh::OnMeshUpdated(mesh);

   // count ghost edges and assign their indices
   NEdges = mesh->GetNEdges();
   NGhostEdges = 0;
   for (node_iterator node = nodes.begin(); node != nodes.end(); ++node)
   {
      if (node->HasEdge() && node->edge_index < 0)
      {
         node->edge_index = NEdges + (NGhostEdges++);
      }
   }

   // count ghost faces
   NFaces = mesh->GetNumFaces();
   NGhostFaces = 0;
   for (face_iterator face = faces.begin(); face != faces.end(); ++face)
   {
      if (face->index < 0) { NGhostFaces++; }
   }

   if (Dim == 2)
   {
      // in 2D we have fake faces because of DG
      MFEM_ASSERT(NFaces == NEdges, "");
      MFEM_ASSERT(NGhostFaces == NGhostEdges, "");
   }

   // resize face_geom (default_geom is for slave faces beyond the ghost layer)
   Geometry::Type default_geom = Geometry::SQUARE;
   face_geom.SetSize(NFaces + NGhostFaces, default_geom);

   // update 'face_geom' for ghost faces, assign ghost face indices
   int nghosts = 0;
   for (int i = 0; i < NGhostElements; i++)
   {
      Element &el = elements[leaf_elements[NElements + i]]; // ghost element
      GeomInfo &gi = GI[el.Geom()];

      for (int j = 0; j < gi.nf; j++)
      {
         const int *fv = gi.faces[j];
         Face* face = faces.Find(el.node[fv[0]], el.node[fv[1]],
                                 el.node[fv[2]], el.node[fv[3]]);
         MFEM_ASSERT(face, "face not found!");

         if (face->index < 0)
         {
            face->index = NFaces + (nghosts++);

            // store the face geometry
            static const Geometry::Type types[5] =
            {
               Geometry::INVALID, Geometry::INVALID,
               Geometry::SEGMENT, Geometry::TRIANGLE, Geometry::SQUARE
            };
            face_geom[face->index] = types[gi.nfv[j]];
         }
      }
   }

   // assign valid indices also to faces beyond the ghost layer
   for (face_iterator face = faces.begin(); face != faces.end(); ++face)
   {
      if (face->index < 0) { face->index = NFaces + (nghosts++); }
   }
   MFEM_ASSERT(nghosts == NGhostFaces, "");
}

void ParNCMesh::ElementSharesFace(int elem, int local, int face)
{
   // Analogous to ElementSharesEdge.

   Element &el = elements[elem];
   int f_index = faces[face].index;

   int &owner = tmp_owner[f_index];
   owner = std::min(owner, el.rank);

   char &flag = tmp_shared_flag[f_index];
   flag |= (el.rank == MyRank) ? 0x1 : 0x2;

   entity_index_rank[2].Append(Connection(f_index, el.rank));

   // derive globally consistent face ID from the global element sequence
   int &el_loc = entity_elem_local[2][f_index];
   if (el_loc < 0 || leaf_glob_order[el.index] < leaf_glob_order[(el_loc >> 4)])
   {
      el_loc = (el.index << 4) | local;
   }
}

void ParNCMesh::BuildFaceList()
{
   // This is an extension of NCMesh::BuildFaceList() which also determines
   // face ownership and prepares face processor groups.

   face_list.Clear();
   if (Dim < 3) { return; }

   int nfaces = NFaces + NGhostFaces;

   tmp_owner.SetSize(nfaces);
   tmp_owner = INT_MAX;

   tmp_shared_flag.SetSize(nfaces);
   tmp_shared_flag = 0;

   entity_index_rank[2].SetSize(6*leaf_elements.Size() * 3/2);
   entity_index_rank[2].SetSize(0);

   entity_elem_local[2].SetSize(nfaces);
   entity_elem_local[2] = -1;

   NCMesh::BuildFaceList();

   InitOwners(nfaces, entity_owner[2]);
   MakeSharedList(face_list, shared_faces);

   tmp_owner.DeleteAll();
   tmp_shared_flag.DeleteAll();

   // create simple conforming (cut-mesh) groups now
   CreateGroups(NFaces, entity_index_rank[2], entity_conf_group[2]);
   // NOTE: entity_index_rank[2] is not deleted until CalculatePMatrixGroups

   CalcFaceOrientations();
}

void ParNCMesh::ElementSharesEdge(int elem, int local, int enode)
{
   // Called by NCMesh::BuildEdgeList when an edge is visited in a leaf element.
   // This allows us to determine edge ownership and whether it is shared
   // without duplicating all the HashTable lookups in NCMesh::BuildEdgeList().

   Element &el= elements[elem];
   int e_index = nodes[enode].edge_index;

   int &owner = tmp_owner[e_index];
   owner = std::min(owner, el.rank);

   char &flag = tmp_shared_flag[e_index];
   flag |= (el.rank == MyRank) ? 0x1 : 0x2;

   entity_index_rank[1].Append(Connection(e_index, el.rank));

   // derive globally consistent edge ID from the global element sequence
   int &el_loc = entity_elem_local[1][e_index];
   if (el_loc < 0 || leaf_glob_order[el.index] < leaf_glob_order[(el_loc >> 4)])
   {
      el_loc = (el.index << 4) | local;
   }
}

void ParNCMesh::BuildEdgeList()
{
   // This is an extension of NCMesh::BuildEdgeList() which also determines
   // edge ownership and prepares edge processor groups.

   int nedges = NEdges + NGhostEdges;

   tmp_owner.SetSize(nedges);
   tmp_owner = INT_MAX;

   tmp_shared_flag.SetSize(nedges);
   tmp_shared_flag = 0;

   entity_index_rank[1].SetSize(12*leaf_elements.Size() * 3/2);
   entity_index_rank[1].SetSize(0);

   entity_elem_local[1].SetSize(nedges);
   entity_elem_local[1] = -1;

   NCMesh::BuildEdgeList();

   InitOwners(nedges, entity_owner[1]);
   MakeSharedList(edge_list, shared_edges);

   tmp_owner.DeleteAll();
   tmp_shared_flag.DeleteAll();

   // create simple conforming (cut-mesh) groups now
   CreateGroups(NEdges, entity_index_rank[1], entity_conf_group[1]);
   // NOTE: entity_index_rank[1] is not deleted until CalculatePMatrixGroups
}

void ParNCMesh::ElementSharesVertex(int elem, int local, int vnode)
{
   // Analogous to ElementSharesEdge.

   Element &el = elements[elem];
   int v_index = nodes[vnode].vert_index;

   int &owner = tmp_owner[v_index];
   owner = std::min(owner, el.rank);

   char &flag = tmp_shared_flag[v_index];
   flag |= (el.rank == MyRank) ? 0x1 : 0x2;

   entity_index_rank[0].Append(Connection(v_index, el.rank));

   // derive globally consistent vertex ID from the global element sequence
   int &el_loc = entity_elem_local[0][v_index];
   if (el_loc < 0 || leaf_glob_order[el.index] < leaf_glob_order[(el_loc >> 4)])
   {
      el_loc = (el.index << 4) | local;
   }
}

void ParNCMesh::BuildVertexList()
{
   // This is an extension of NCMesh::BuildVertexList() which also determines
   // vertex ownership and creates vertex processor groups.

   int nvertices = NVertices + NGhostVertices;

   tmp_owner.SetSize(nvertices);
   tmp_owner = INT_MAX;

   tmp_shared_flag.SetSize(nvertices);
   tmp_shared_flag = 0;

   entity_index_rank[0].SetSize(8*leaf_elements.Size());
   entity_index_rank[0].SetSize(0);

   entity_elem_local[0].SetSize(nvertices);
   entity_elem_local[0] = -1;

   NCMesh::BuildVertexList();

   InitOwners(nvertices, entity_owner[0]);
   MakeSharedList(vertex_list, shared_vertices);

   tmp_owner.DeleteAll();
   tmp_shared_flag.DeleteAll();

   // create simple conforming (cut-mesh) groups now
   CreateGroups(NVertices, entity_index_rank[0], entity_conf_group[0]);
   // NOTE: entity_index_rank[0] is not deleted until CalculatePMatrixGroups
}

void ParNCMesh::InitOwners(int num, Array<GroupId> &entity_owner)
{
   entity_owner.SetSize(num);
   for (int i = 0; i < num; i++)
   {
      entity_owner[i] =
         (tmp_owner[i] != INT_MAX) ? GetSingletonGroup(tmp_owner[i]) : 0;
   }
}

void ParNCMesh::MakeSharedList(const NCList &list, NCList &shared)
{
   MFEM_VERIFY(tmp_shared_flag.Size(), "wrong code path");

   // combine flags of masters and slaves
   for (int i = 0; i < list.masters.Size(); i++)
   {
      const Master &master = list.masters[i];
      char &master_flag = tmp_shared_flag[master.index];
      char master_old_flag = master_flag;

      for (int j = master.slaves_begin; j < master.slaves_end; j++)
      {
         int si = list.slaves[j].index;
         if (si >= 0)
         {
            char &slave_flag = tmp_shared_flag[si];
            master_flag |= slave_flag;
            slave_flag |= master_old_flag;
         }
         else // special case: prism edge-face constraint
         {
            if (entity_owner[1][-1-si] != MyRank)
            {
               master_flag |= 0x2;
            }
         }
      }
   }

   shared.Clear();

   for (int i = 0; i < list.conforming.Size(); i++)
   {
      if (tmp_shared_flag[list.conforming[i].index] == 0x3)
      {
         shared.conforming.Append(list.conforming[i]);
      }
   }
   for (int i = 0; i < list.masters.Size(); i++)
   {
      if (tmp_shared_flag[list.masters[i].index] == 0x3)
      {
         shared.masters.Append(list.masters[i]);
      }
   }
   for (int i = 0; i < list.slaves.Size(); i++)
   {
      int si = list.slaves[i].index;
      if (si >= 0 && tmp_shared_flag[si] == 0x3)
      {
         shared.slaves.Append(list.slaves[i]);
      }
   }
}

bool operator<(const ParNCMesh::CommGroup &lhs, const ParNCMesh::CommGroup &rhs)
{
   if (lhs.size() == rhs.size())
   {
      for (unsigned i = 0; i < lhs.size(); i++)
      {
         if (lhs[i] < rhs[i]) { return true; }
      }
      return false;
   }
   return lhs.size() < rhs.size();
}

#ifdef MFEM_DEBUG
static bool group_sorted(const ParNCMesh::CommGroup &group)
{
   for (unsigned i = 1; i < group.size(); i++)
   {
      if (group[i] <= group[i-1]) { return false; }
   }
   return true;
}
#endif

ParNCMesh::GroupId ParNCMesh::GetGroupId(const CommGroup &group)
{
   if (group.size() == 1 && group[0] == MyRank)
   {
      return 0;
   }
   MFEM_ASSERT(group_sorted(group), "invalid group");
   GroupId &id = group_id[group];
   if (!id)
   {
      id = groups.size();
      groups.push_back(group);
   }
   return id;
}

ParNCMesh::GroupId ParNCMesh::GetSingletonGroup(int rank)
{
   MFEM_ASSERT(rank != INT_MAX, "invalid rank");
   static std::vector<int> group;
   group.resize(1);
   group[0] = rank;
   return GetGroupId(group);
}

bool ParNCMesh::GroupContains(GroupId id, int rank) const
{
   // TODO: would std::lower_bound() pay off here? Groups are usually small.
   const CommGroup &group = groups[id];
   for (unsigned i = 0; i < group.size(); i++)
   {
      if (group[i] == rank) { return true; }
   }
   return false;
}

void ParNCMesh::CreateGroups(int nentities, Array<Connection> &index_rank,
                             Array<GroupId> &entity_group)
{
   index_rank.Sort();
   index_rank.Unique();

   entity_group.SetSize(nentities);
   entity_group = 0;

   CommGroup group;
   group.reserve(128);

   int begin = 0, end = 0;
   while (begin < index_rank.Size())
   {
      int index = index_rank[begin].from;
      if (index >= nentities)
      {
         break; // probably creating entity_conf_group (no ghosts)
      }
      while (end < index_rank.Size() && index_rank[end].from == index)
      {
         end++;
      }
      group.resize(end - begin);
      for (int i = begin; i < end; i++)
      {
         group[i - begin] = index_rank[i].to;
      }
      entity_group[index] = GetGroupId(group);
      begin = end;
   }
}

void ParNCMesh::AddConnections(int entity, int index, const Array<int> &ranks)
{
   for (int i = 0; i < ranks.Size(); i++)
   {
      entity_index_rank[entity].Append(Connection(index, ranks[i]));
   }
}

void ParNCMesh::CalculatePMatrixGroups()
{
   // make sure all entity_index_rank[i] arrays are filled
   GetSharedVertices();
   GetSharedEdges();
   GetSharedFaces();

   int v[4], e[4], eo[4];

   Array<int> ranks;
   ranks.Reserve(256);

   // connect slave edges to master edges and their vertices
   for (int i = 0; i < shared_edges.masters.Size(); i++)
   {
      const Master &master_edge = shared_edges.masters[i];
      ranks.SetSize(0);
      for (int j = master_edge.slaves_begin; j < master_edge.slaves_end; j++)
      {
         int owner = entity_owner[1][edge_list.slaves[j].index];
         ranks.Append(groups[owner][0]);
      }
      ranks.Sort();
      ranks.Unique();

      AddConnections(1, master_edge.index, ranks);

      GetEdgeVertices(master_edge, v);
      for (int j = 0; j < 2; j++)
      {
         AddConnections(0, v[j], ranks);
      }
   }

   // connect slave faces to master faces and their edges and vertices
   for (int i = 0; i < shared_faces.masters.Size(); i++)
   {
      const Master &master_face = shared_faces.masters[i];
      ranks.SetSize(0);
      for (int j = master_face.slaves_begin; j < master_face.slaves_end; j++)
      {
         int si = face_list.slaves[j].index;
         int owner = (si >= 0) ? entity_owner[2][si] // standard face dependency
                     /*     */ : entity_owner[1][-1 - si]; // prism edge-face dep
         ranks.Append(groups[owner][0]);
      }
      ranks.Sort();
      ranks.Unique();

      AddConnections(2, master_face.index, ranks);

      int nfv = GetFaceVerticesEdges(master_face, v, e, eo);
      for (int j = 0; j < nfv; j++)
      {
         AddConnections(0, v[j], ranks);
         AddConnections(1, e[j], ranks);
      }
   }

   int nentities[3] =
   {
      NVertices + NGhostVertices,
      NEdges + NGhostEdges,
      NFaces + NGhostFaces
   };

   // compress the index-rank arrays into group representation
   for (int i = 0; i < 3; i++)
   {
      CreateGroups(nentities[i], entity_index_rank[i], entity_pmat_group[i]);
      entity_index_rank[i].DeleteAll();
   }
}

int ParNCMesh::get_face_orientation(Face &face, Element &e1, Element &e2,
                                    int local[2])
{
   // Return face orientation in e2, assuming the face has orientation 0 in e1.
   int ids[2][4];
   Element* e[2] = { &e1, &e2 };
   for (int i = 0; i < 2; i++)
   {
      // get local face number (remember that p1, p2, p3 are not in order, and
      // p4 is not stored)
      int lf = find_local_face(e[i]->Geom(),
                               find_node(*e[i], face.p1),
                               find_node(*e[i], face.p2),
                               find_node(*e[i], face.p3));
      // optional output
      if (local) { local[i] = lf; }

      // get node IDs for the face as seen from e[i]
      const int* fv = GI[e[i]->Geom()].faces[lf];
      for (int j = 0; j < 4; j++)
      {
         ids[i][j] = e[i]->node[fv[j]];
      }
   }

   return (ids[0][3] >= 0) ? Mesh::GetQuadOrientation(ids[0], ids[1])
          /*            */ : Mesh::GetTriOrientation(ids[0], ids[1]);
}

void ParNCMesh::CalcFaceOrientations()
{
   if (Dim < 3) { return; }

   // Calculate orientation of shared conforming faces.
   // NOTE: face orientation is calculated relative to its lower rank element.
   // Thanks to the ghost layer this can be done locally, without communication.

   face_orient.SetSize(NFaces);
   face_orient = 0;

   for (face_iterator face = faces.begin(); face != faces.end(); ++face)
   {
      if (face->elem[0] >= 0 && face->elem[1] >= 0 && face->index < NFaces)
      {
         Element *e1 = &elements[face->elem[0]];
         Element *e2 = &elements[face->elem[1]];

         if (e1->rank == e2->rank) { continue; }
         if (e1->rank > e2->rank) { std::swap(e1, e2); }

         face_orient[face->index] = get_face_orientation(*face, *e1, *e2);
      }
   }
}

void ParNCMesh::GetBoundaryClosure(const Array<int> &bdr_attr_is_ess,
                                   Array<int> &bdr_vertices,
                                   Array<int> &bdr_edges)
{
   NCMesh::GetBoundaryClosure(bdr_attr_is_ess, bdr_vertices, bdr_edges);

   int i, j;
   // filter out ghost vertices
   for (i = j = 0; i < bdr_vertices.Size(); i++)
   {
      if (bdr_vertices[i] < NVertices) { bdr_vertices[j++] = bdr_vertices[i]; }
   }
   bdr_vertices.SetSize(j);

   // filter out ghost edges
   for (i = j = 0; i < bdr_edges.Size(); i++)
   {
      if (bdr_edges[i] < NEdges) { bdr_edges[j++] = bdr_edges[i]; }
   }
   bdr_edges.SetSize(j);
}


//// Neighbors /////////////////////////////////////////////////////////////////

void ParNCMesh::UpdateLayers()
{
   if (element_type.Size()) { return; }

   int nleaves = leaf_elements.Size();

   element_type.SetSize(nleaves);
   for (int i = 0; i < nleaves; i++)
   {
      element_type[i] = (elements[leaf_elements[i]].rank == MyRank) ? 1 : 0;
   }

   // determine the ghost layer
   Array<char> ghost_set;
   FindSetNeighbors(element_type, NULL, &ghost_set);

   // find the neighbors of the ghost layer
   Array<char> boundary_set;
   FindSetNeighbors(ghost_set, NULL, &boundary_set);

   ghost_layer.SetSize(0);
   boundary_layer.SetSize(0);
   for (int i = 0; i < nleaves; i++)
   {
      char &etype = element_type[i];
      if (ghost_set[i])
      {
         etype = 2;
         ghost_layer.Append(leaf_elements[i]);
      }
      else if (boundary_set[i] && etype)
      {
         etype = 3;
         boundary_layer.Append(leaf_elements[i]);
      }
   }
}

bool ParNCMesh::CheckElementType(int elem, int type)
{
   Element &el = elements[elem];
   if (!el.ref_type)
   {
      return (element_type[el.index] == type);
   }
   else
   {
      for (int i = 0; i < 8 && el.child[i] >= 0; i++)
      {
         if (!CheckElementType(el.child[i], type)) { return false; }
      }
      return true;
   }
}

void ParNCMesh::ElementNeighborProcessors(int elem, Array<int> &ranks)
{
   ranks.SetSize(0); // preserve capacity

   // big shortcut: there are no neighbors if element_type == 1
   if (CheckElementType(elem, 1)) { return; }

   // ok, we do need to look for neighbors;
   // at least we can only search in the ghost layer
   tmp_neighbors.SetSize(0);
   FindNeighbors(elem, tmp_neighbors, &ghost_layer);

   // return a list of processors
   for (int i = 0; i < tmp_neighbors.Size(); i++)
   {
      ranks.Append(elements[tmp_neighbors[i]].rank);
   }
   ranks.Sort();
   ranks.Unique();
}

template<class T>
static void set_to_array(const std::set<T> &set, Array<T> &array)
{
   array.Reserve(set.size());
   array.SetSize(0);
   for (std::set<int>::iterator it = set.begin(); it != set.end(); ++it)
   {
      array.Append(*it);
   }
}

void ParNCMesh::NeighborProcessors(Array<int> &neighbors)
{
   UpdateLayers();

   // TODO: look at groups instead?

   std::set<int> ranks;
   for (int i = 0; i < ghost_layer.Size(); i++)
   {
      ranks.insert(elements[ghost_layer[i]].rank);
   }
   set_to_array(ranks, neighbors);
}


//// ParMesh compatibility /////////////////////////////////////////////////////

void ParNCMesh::MakeSharedTable(int ngroups, int ent, Array<int> &shared_local,
                                Table &group_shared, Array<char> *entity_geom,
                                char geom)
{
   const Array<GroupId> &conf_group = entity_conf_group[ent];

   group_shared.MakeI(ngroups-1);

   // count shared entities
   int num_shared = 0;
   for (int i = 0; i < conf_group.Size(); i++)
   {
      if (conf_group[i])
      {
         if (entity_geom && (*entity_geom)[i] != geom) { continue; }

         num_shared++;
         group_shared.AddAColumnInRow(conf_group[i]-1);
      }
   }

   shared_local.SetSize(num_shared);
   group_shared.MakeJ();

   // fill shared_local and group_shared
   for (int i = 0, j = 0; i < conf_group.Size(); i++)
   {
      if (conf_group[i])
      {
         if (entity_geom && (*entity_geom)[i] != geom) { continue; }

         shared_local[j] = i;
         group_shared.AddConnection(conf_group[i]-1, j);
         j++;
      }
   }
   group_shared.ShiftUpI();

   // sort the groups consistently across processors
   for (int i = 0; i < group_shared.Size(); i++)
   {
      int size = group_shared.RowSize(i);
      int *row = group_shared.GetRow(i);

      Array<int> ref_row(row, size);
      ref_row.Sort([&](const int a, const int b)
      {
         int el_loc_a = entity_elem_local[ent][shared_local[a]];
         int el_loc_b = entity_elem_local[ent][shared_local[b]];

         int lgo_a = leaf_glob_order[el_loc_a >> 4];
         int lgo_b = leaf_glob_order[el_loc_b >> 4];

         if (lgo_a != lgo_b) { return lgo_a < lgo_b; }

         return (el_loc_a & 0xf) < (el_loc_b & 0xf);
      });
   }
}

void ParNCMesh::GetConformingSharedStructures(ParMesh &pmesh)
{
   // make sure we have entity_conf_group[x] and the ordering arrays
   if (leaf_elements.Size())
   {
      for (int ent = 0; ent < Dim; ent++)
      {
         GetSharedList(ent);
         MFEM_VERIFY(entity_conf_group[ent].Size(), "internal error");
         MFEM_VERIFY(entity_elem_local[ent].Size(), "internal error");
      }
      MFEM_VERIFY(leaf_glob_order.Size(), "internal error");
   }

   // create ParMesh groups, and the map (ncmesh_group -> pmesh_group)
   Array<int> group_map(groups.size());
   {
      group_map = 0;
      IntegerSet iset;
      ListOfIntegerSets int_groups;
      for (unsigned i = 0; i < groups.size(); i++)
      {
         if (groups[i].size() > 1 || !i) // skip singleton groups
         {
            iset.Recreate(groups[i].size(), groups[i].data());
            group_map[i] = int_groups.Insert(iset);
         }
      }
      pmesh.gtopo.Create(int_groups, 822);
   }

   // renumber groups in entity_conf_group[] (due to missing singletons)
   for (int ent = 0; ent < 3; ent++)
   {
      for (int i = 0; i < entity_conf_group[ent].Size(); i++)
      {
         GroupId &ecg = entity_conf_group[ent][i];
         ecg = group_map[ecg];
      }
   }

   // create shared to local index mappings and group tables
   int ng = pmesh.gtopo.NGroups();
   MakeSharedTable(ng, 0, pmesh.svert_lvert, pmesh.group_svert);
   MakeSharedTable(ng, 1, pmesh.sedge_ledge, pmesh.group_sedge);

   Array<int> slt, slq;
   MakeSharedTable(ng, 2, slt, pmesh.group_stria, &face_geom, Geometry::TRIANGLE);
   MakeSharedTable(ng, 2, slq, pmesh.group_squad, &face_geom, Geometry::SQUARE);

   pmesh.sface_lface = slt;
   pmesh.sface_lface.Append(slq);

   // create shared_edges
   for (int i = 0; i < pmesh.shared_edges.Size(); i++)
   {
      delete pmesh.shared_edges[i];
   }
   pmesh.shared_edges.SetSize(pmesh.sedge_ledge.Size());
   for (int i = 0; i < pmesh.shared_edges.Size(); i++)
   {
      int el_loc = entity_elem_local[1][pmesh.sedge_ledge[i]];
      MeshId edge_id(-1, leaf_elements[(el_loc >> 4)], (el_loc & 0xf));

      int v[2];
      GetEdgeVertices(edge_id, v, false);
      pmesh.shared_edges[i] = new Segment(v, 1);
   }

   // create shared_trias
   pmesh.shared_trias.SetSize(slt.Size());
   for (int i = 0; i < slt.Size(); i++)
   {
      int el_loc = entity_elem_local[2][slt[i]];
      MeshId face_id(-1, leaf_elements[(el_loc >> 4)], (el_loc & 0xf));

      int v[4], e[4], eo[4];
      GetFaceVerticesEdges(face_id, v, e, eo);
      pmesh.shared_trias[i].Set(v);
   }

   // create shared_quads
   pmesh.shared_quads.SetSize(slq.Size());
   for (int i = 0; i < slq.Size(); i++)
   {
      int el_loc = entity_elem_local[2][slq[i]];
      MeshId face_id(-1, leaf_elements[(el_loc >> 4)], (el_loc & 0xf));

      int e[4], eo[4];
      GetFaceVerticesEdges(face_id, pmesh.shared_quads[i].v, e, eo);
   }

   // free the arrays, they're not needed anymore (until next mesh update)
   for (int ent = 0; ent < Dim; ent++)
   {
      entity_conf_group[ent].DeleteAll();
      entity_elem_local[ent].DeleteAll();
   }
   leaf_glob_order.DeleteAll();
}

void ParNCMesh::GetFaceNeighbors(ParMesh &pmesh)
{
   ClearAuxPM();

   const NCList &shared = (Dim == 3) ? GetSharedFaces() : GetSharedEdges();
   const NCList &full_list = (Dim == 3) ? GetFaceList() : GetEdgeList();

   Array<Element*> fnbr;
   Array<Connection> send_elems;

   int bound = shared.conforming.Size() + shared.slaves.Size();

   fnbr.Reserve(bound);
   send_elems.Reserve(bound);

   // go over all shared faces and collect face neighbor elements
   for (int i = 0; i < shared.conforming.Size(); i++)
   {
      const MeshId &cf = shared.conforming[i];
      Face* face = GetFace(elements[cf.element], cf.local);
      MFEM_ASSERT(face != NULL, "");

      MFEM_ASSERT(face->elem[0] >= 0 && face->elem[1] >= 0, "");
      Element* e[2] = { &elements[face->elem[0]], &elements[face->elem[1]] };

      if (e[0]->rank == MyRank) { std::swap(e[0], e[1]); }
      MFEM_ASSERT(e[0]->rank != MyRank && e[1]->rank == MyRank, "");

      fnbr.Append(e[0]);
      send_elems.Append(Connection(e[0]->rank, e[1]->index));
   }

   for (int i = 0; i < shared.masters.Size(); i++)
   {
      const Master &mf = shared.masters[i];
      for (int j = mf.slaves_begin; j < mf.slaves_end; j++)
      {
         const Slave &sf = full_list.slaves[j];
         if (sf.element < 0) { continue; }

         MFEM_ASSERT(mf.element >= 0, "");
         Element* e[2] = { &elements[mf.element], &elements[sf.element] };

         bool loc0 = (e[0]->rank == MyRank);
         bool loc1 = (e[1]->rank == MyRank);
         if (loc0 == loc1) { continue; }
         if (loc0) { std::swap(e[0], e[1]); }

         fnbr.Append(e[0]);
         send_elems.Append(Connection(e[0]->rank, e[1]->index));
      }
   }

   MFEM_ASSERT(fnbr.Size() <= bound, "oops, bad upper bound");

   // remove duplicate face neighbor elements and sort them by rank & index
   // (note that the send table is sorted the same way and the order is also the
   // same on different processors, this is important for ExchangeFaceNbrData)
   fnbr.Sort();
   fnbr.Unique();
   fnbr.Sort([](const Element* a, const Element* b)
   {
      return (a->rank != b->rank) ? a->rank < b->rank
             /*                */ : a->index < b->index;
   });

   // put the ranks into 'face_nbr_group'
   for (int i = 0; i < fnbr.Size(); i++)
   {
      if (!i || fnbr[i]->rank != pmesh.face_nbr_group.Last())
      {
         pmesh.face_nbr_group.Append(fnbr[i]->rank);
      }
   }
   int nranks = pmesh.face_nbr_group.Size();

   // create a new mfem::Element for each face neighbor element
   pmesh.face_nbr_elements.SetSize(0);
   pmesh.face_nbr_elements.Reserve(fnbr.Size());

   pmesh.face_nbr_elements_offset.SetSize(0);
   pmesh.face_nbr_elements_offset.Reserve(pmesh.face_nbr_group.Size()+1);

   Array<int> fnbr_index(NGhostElements);
   fnbr_index = -1;

   std::map<int, int> vert_map;
   for (int i = 0; i < fnbr.Size(); i++)
   {
      Element* elem = fnbr[i];
      mfem::Element* fne = NewMeshElement(elem->geom);
      fne->SetAttribute(elem->attribute);
      pmesh.face_nbr_elements.Append(fne);

      GeomInfo& gi = GI[(int) elem->geom];
      for (int k = 0; k < gi.nv; k++)
      {
         int &v = vert_map[elem->node[k]];
         if (!v) { v = vert_map.size(); }
         fne->GetVertices()[k] = v-1;
      }

      if (!i || elem->rank != fnbr[i-1]->rank)
      {
         pmesh.face_nbr_elements_offset.Append(i);
      }

      MFEM_ASSERT(elem->index >= NElements, "not a ghost element");
      fnbr_index[elem->index - NElements] = i;
   }
   pmesh.face_nbr_elements_offset.Append(fnbr.Size());

   // create vertices in 'face_nbr_vertices'
   {
      pmesh.face_nbr_vertices.SetSize(vert_map.size());
      tmp_vertex = new TmpVertex[nodes.NumIds()]; // TODO: something cheaper?

      std::map<int, int>::iterator it;
      for (it = vert_map.begin(); it != vert_map.end(); ++it)
      {
         pmesh.face_nbr_vertices[it->second-1].SetCoords(
            spaceDim, CalcVertexPos(it->first));
      }
      delete [] tmp_vertex;
   }

   // make the 'send_face_nbr_elements' table
   send_elems.Sort();
   send_elems.Unique();

   for (int i = 0, last_rank = -1; i < send_elems.Size(); i++)
   {
      Connection &c = send_elems[i];
      if (c.from != last_rank)
      {
         // renumber rank to position in 'face_nbr_group'
         last_rank = c.from;
         c.from = pmesh.face_nbr_group.Find(c.from);
      }
      else
      {
         c.from = send_elems[i-1].from; // avoid search
      }
   }
   pmesh.send_face_nbr_elements.MakeFromList(nranks, send_elems);

   // go over the shared faces again and modify their Mesh::FaceInfo
   for (int i = 0; i < shared.conforming.Size(); i++)
   {
      const MeshId &cf = shared.conforming[i];
      Face* face = GetFace(elements[cf.element], cf.local);

      Element* e[2] = { &elements[face->elem[0]], &elements[face->elem[1]] };
      if (e[0]->rank == MyRank) { std::swap(e[0], e[1]); }

      Mesh::FaceInfo &fi = pmesh.faces_info[cf.index];
      fi.Elem2No = -1 - fnbr_index[e[0]->index - NElements];

      if (Dim == 3)
      {
         int local[2];
         int o = get_face_orientation(*face, *e[1], *e[0], local);
         fi.Elem2Inf = 64*local[1] + o;
      }
      else
      {
         fi.Elem2Inf = 64*find_element_edge(*e[0], face->p1, face->p3) + 1;
      }
   }

   if (shared.slaves.Size())
   {
      int nfaces = NFaces, nghosts = NGhostFaces;
      if (Dim <= 2) { nfaces = NEdges, nghosts = NGhostEdges; }

      // enlarge Mesh::faces_info for ghost slaves
      MFEM_ASSERT(pmesh.faces_info.Size() == nfaces, "");
      MFEM_ASSERT(pmesh.GetNumFaces() == nfaces, "");
      pmesh.faces_info.SetSize(nfaces + nghosts);
      for (int i = nfaces; i < pmesh.faces_info.Size(); i++)
      {
         Mesh::FaceInfo &fi = pmesh.faces_info[i];
         fi.Elem1No  = fi.Elem2No  = -1;
         fi.Elem1Inf = fi.Elem2Inf = -1;
         fi.NCFace = -1;
      }
      // Note that some of the indices i >= nfaces in pmesh.faces_info will
      // remain untouched below and they will have Elem1No == -1, in particular.

      // fill in FaceInfo for shared slave faces
      for (int i = 0; i < shared.masters.Size(); i++)
      {
         const Master &mf = shared.masters[i];
         for (int j = mf.slaves_begin; j < mf.slaves_end; j++)
         {
            const Slave &sf = full_list.slaves[j];
            if (sf.element < 0) { continue; }

            MFEM_ASSERT(mf.element >= 0, "");
            Element &sfe = elements[sf.element];
            Element &mfe = elements[mf.element];

            bool sloc = (sfe.rank == MyRank);
            bool mloc = (mfe.rank == MyRank);
            if (sloc == mloc) { continue; }

            Mesh::FaceInfo &fi = pmesh.faces_info[sf.index];
            fi.Elem1No = sfe.index;
            fi.Elem2No = mfe.index;
            fi.Elem1Inf = 64 * sf.local;
            fi.Elem2Inf = 64 * mf.local;

            if (!sloc)
            {
               // 'fi' is the info for a ghost slave face with index:
               // sf.index >= nfaces
               std::swap(fi.Elem1No, fi.Elem2No);
               std::swap(fi.Elem1Inf, fi.Elem2Inf);
               // After the above swap, Elem1No refers to the local, master-side
               // element. In other words, side 1 IS NOT the side that generated
               // the face.
            }
            else
            {
               // 'fi' is the info for a local slave face with index:
               // sf.index < nfaces
               // Here, Elem1No refers to the local, slave-side element.
               // In other words, side 1 IS the side that generated the face.
            }
            MFEM_ASSERT(fi.Elem2No >= NElements, "");
            fi.Elem2No = -1 - fnbr_index[fi.Elem2No - NElements];

            const DenseMatrix* pm = full_list.point_matrices[sf.geom][sf.matrix];
            if (!sloc && Dim == 3)
            {
               // TODO: does this handle triangle faces correctly?

               // ghost slave in 3D needs flipping orientation
               DenseMatrix* pm2 = new DenseMatrix(*pm);
               std::swap((*pm2)(0,1), (*pm2)(0,3));
               std::swap((*pm2)(1,1), (*pm2)(1,3));
               aux_pm_store.Append(pm2);

               fi.Elem2Inf ^= 1;
               pm = pm2;

               // The problem is that sf.point_matrix is designed for P matrix
               // construction and always has orientation relative to the slave
               // face. In ParMesh::GetSharedFaceTransformations the result
               // would therefore be the same on both processors, which is not
               // how that function works for conforming faces. The orientation
               // of Loc1, Loc2 and Face needs to always be relative to Element
               // 1, which is the element containing the slave face on one
               // processor, but on the other it is the element containing the
               // master face. In the latter case we need to flip the pm.
            }
            else if (!sloc && Dim == 2)
            {
               fi.Elem2Inf ^= 1; // set orientation to 1
               // The point matrix (used to define "side 1" which is the same as
               // "parent side" in this case) does not require a flip since it
               // is aligned with the parent side, so NO flip is performed in
               // Mesh::ApplyLocalSlaveTransformation.
            }

            MFEM_ASSERT(fi.NCFace < 0, "");
            fi.NCFace = pmesh.nc_faces_info.Size();
            pmesh.nc_faces_info.Append(Mesh::NCFaceInfo(true, sf.master, pm));
         }
      }
   }

   // NOTE: this function skips ParMesh::send_face_nbr_vertices and
   // ParMesh::face_nbr_vertices_offset, these are not used outside of ParMesh
}

void ParNCMesh::ClearAuxPM()
{
   for (int i = 0; i < aux_pm_store.Size(); i++)
   {
      delete aux_pm_store[i];
   }
   aux_pm_store.DeleteAll();
}


//// Prune, Refine, Derefine ///////////////////////////////////////////////////

bool ParNCMesh::PruneTree(int elem)
{
   Element &el = elements[elem];
   if (el.ref_type)
   {
      bool remove[8];
      bool removeAll = true;

      // determine which subtrees can be removed (and whether it's all of them)
      for (int i = 0; i < 8; i++)
      {
         remove[i] = false;
         if (el.child[i] >= 0)
         {
            remove[i] = PruneTree(el.child[i]);
            if (!remove[i]) { removeAll = false; }
         }
      }

      // all children can be removed, let the (maybe indirect) parent do it
      if (removeAll) { return true; }

      // not all children can be removed, but remove those that can be
      for (int i = 0; i < 8; i++)
      {
         if (remove[i]) { DerefineElement(el.child[i]); }
      }

      return false; // need to keep this element and up
   }
   else
   {
      // return true if this leaf can be removed
      return el.rank < 0;
   }
}

void ParNCMesh::Prune()
{
   if (!Iso && Dim == 3)
   {
      if (MyRank == 0)
      {
         MFEM_WARNING("Can't prune 3D aniso meshes yet.");
      }
      return;
   }

   UpdateLayers();

   for (int i = 0; i < leaf_elements.Size(); i++)
   {
      // rank of elements beyond the ghost layer is unknown / not updated
      if (element_type[i] == 0)
      {
         elements[leaf_elements[i]].rank = -1;
         // NOTE: rank == -1 will make the element disappear from leaf_elements
         // on next Update, see NCMesh::CollectLeafElements
      }
   }

   // derefine subtrees whose leaves are all unneeded
   for (int i = 0; i < root_state.Size(); i++)
   {
      if (PruneTree(i)) { DerefineElement(i); }
   }

   Update();
}


void ParNCMesh::Refine(const Array<Refinement> &refinements)
{
   if (NRanks == 1)
   {
      NCMesh::Refine(refinements);
      return;
   }

   for (int i = 0; i < refinements.Size(); i++)
   {
      const Refinement &ref = refinements[i];
      MFEM_VERIFY(ref.ref_type == 7 || Dim < 3,
                  "anisotropic parallel refinement not supported yet in 3D.");
   }
   MFEM_VERIFY(Iso || Dim < 3,
               "parallel refinement of 3D aniso meshes not supported yet.");

   NeighborRefinementMessage::Map send_ref;

   // create refinement messages to all neighbors (NOTE: some may be empty)
   Array<int> neighbors;
   NeighborProcessors(neighbors);
   for (int i = 0; i < neighbors.Size(); i++)
   {
      send_ref[neighbors[i]].SetNCMesh(this);
   }

   // populate messages: all refinements that occur next to the processor
   // boundary need to be sent to the adjoining neighbors so they can keep
   // their ghost layer up to date
   Array<int> ranks;
   ranks.Reserve(64);
   for (int i = 0; i < refinements.Size(); i++)
   {
      const Refinement &ref = refinements[i];
      MFEM_ASSERT(ref.index < NElements, "");
      int elem = leaf_elements[ref.index];
      ElementNeighborProcessors(elem, ranks);
      for (int j = 0; j < ranks.Size(); j++)
      {
         send_ref[ranks[j]].AddRefinement(elem, ref.ref_type);
      }
   }

   // send the messages (overlap with local refinements)
   NeighborRefinementMessage::IsendAll(send_ref, MyComm);

   // do local refinements
   for (int i = 0; i < refinements.Size(); i++)
   {
      const Refinement &ref = refinements[i];
      NCMesh::RefineElement(leaf_elements[ref.index], ref.ref_type);
   }

   // receive (ghost layer) refinements from all neighbors
   for (int j = 0; j < neighbors.Size(); j++)
   {
      int rank, size;
      NeighborRefinementMessage::Probe(rank, size, MyComm);

      NeighborRefinementMessage msg;
      msg.SetNCMesh(this);
      msg.Recv(rank, size, MyComm);

      // do the ghost refinements
      for (int i = 0; i < msg.Size(); i++)
      {
         NCMesh::RefineElement(msg.elements[i], msg.values[i]);
      }
   }

   Update();

   // make sure we can delete the send buffers
   NeighborRefinementMessage::WaitAllSent(send_ref);
}


void ParNCMesh::LimitNCLevel(int max_nc_level)
{
   MFEM_VERIFY(max_nc_level >= 1, "'max_nc_level' must be 1 or greater.");

   while (1)
   {
      Array<Refinement> refinements;
      GetLimitRefinements(refinements, max_nc_level);

      long size = refinements.Size(), glob_size;
      MPI_Allreduce(&size, &glob_size, 1, MPI_LONG, MPI_SUM, MyComm);

      if (!glob_size) { break; }

      Refine(refinements);
   }
}

void ParNCMesh::Derefine(const Array<int> &derefs)
{
   MFEM_VERIFY(Dim < 3 || Iso,
               "derefinement of 3D anisotropic meshes not implemented yet.");

   InitDerefTransforms();

   // store fine element ranks
   old_index_or_rank.SetSize(leaf_elements.Size());
   for (int i = 0; i < leaf_elements.Size(); i++)
   {
      old_index_or_rank[i] = elements[leaf_elements[i]].rank;
   }

   // back up the leaf_elements array
   Array<int> old_elements;
   leaf_elements.Copy(old_elements);

   // *** STEP 1: redistribute elements to avoid complex derefinements ***

   Array<int> new_ranks(leaf_elements.Size());
   for (int i = 0; i < leaf_elements.Size(); i++)
   {
      new_ranks[i] = elements[leaf_elements[i]].rank;
   }

   // make the lowest rank get all the fine elements for each derefinement
   for (int i = 0; i < derefs.Size(); i++)
   {
      int row = derefs[i];
      MFEM_VERIFY(row >= 0 && row < derefinements.Size(),
                  "invalid derefinement number.");

      const int* fine = derefinements.GetRow(row);
      int size = derefinements.RowSize(row);

      int coarse_rank = INT_MAX;
      for (int j = 0; j < size; j++)
      {
         int fine_rank = elements[leaf_elements[fine[j]]].rank;
         coarse_rank = std::min(coarse_rank, fine_rank);
      }
      for (int j = 0; j < size; j++)
      {
         new_ranks[fine[j]] = coarse_rank;
      }
   }

   int target_elements = 0;
   for (int i = 0; i < new_ranks.Size(); i++)
   {
      if (new_ranks[i] == MyRank) { target_elements++; }
   }

   // redistribute elements slightly to get rid of complex derefinements
   // straddling processor boundaries *and* update the ghost layer
   RedistributeElements(new_ranks, target_elements, false);

   // *** STEP 2: derefine now, communication similar to Refine() ***

   NeighborDerefinementMessage::Map send_deref;

   // create derefinement messages to all neighbors (NOTE: some may be empty)
   Array<int> neighbors;
   NeighborProcessors(neighbors);
   for (int i = 0; i < neighbors.Size(); i++)
   {
      send_deref[neighbors[i]].SetNCMesh(this);
   }

   // derefinements that occur next to the processor boundary need to be sent
   // to the adjoining neighbors to keep their ghost layers in sync
   Array<int> ranks;
   ranks.Reserve(64);
   for (int i = 0; i < derefs.Size(); i++)
   {
      const int* fine = derefinements.GetRow(derefs[i]);
      int parent = elements[old_elements[fine[0]]].parent;

      // send derefinement to neighbors
      ElementNeighborProcessors(parent, ranks);
      for (int j = 0; j < ranks.Size(); j++)
      {
         send_deref[ranks[j]].AddDerefinement(parent, new_ranks[fine[0]]);
      }
   }
   NeighborDerefinementMessage::IsendAll(send_deref, MyComm);

   // restore old (pre-redistribution) element indices, for SetDerefMatrixCodes
   for (int i = 0; i < leaf_elements.Size(); i++)
   {
      elements[leaf_elements[i]].index = -1;
   }
   for (int i = 0; i < old_elements.Size(); i++)
   {
      elements[old_elements[i]].index = i;
   }

   // do local derefinements
   Array<int> coarse;
   old_elements.Copy(coarse);
   for (int i = 0; i < derefs.Size(); i++)
   {
      const int* fine = derefinements.GetRow(derefs[i]);
      int parent = elements[old_elements[fine[0]]].parent;

      // record the relation of the fine elements to their parent
      SetDerefMatrixCodes(parent, coarse);

      NCMesh::DerefineElement(parent);
   }

   // receive ghost layer derefinements from all neighbors
   for (int j = 0; j < neighbors.Size(); j++)
   {
      int rank, size;
      NeighborDerefinementMessage::Probe(rank, size, MyComm);

      NeighborDerefinementMessage msg;
      msg.SetNCMesh(this);
      msg.Recv(rank, size, MyComm);

      // do the ghost derefinements
      for (int i = 0; i < msg.Size(); i++)
      {
         int elem = msg.elements[i];
         if (elements[elem].ref_type)
         {
            SetDerefMatrixCodes(elem, coarse);
            NCMesh::DerefineElement(elem);
         }
         elements[elem].rank = msg.values[i];
      }
   }

   // update leaf_elements, Element::index etc.
   Update();

   UpdateLayers();

   // link old fine elements to the new coarse elements
   for (int i = 0; i < coarse.Size(); i++)
   {
      int index = elements[coarse[i]].index;
      if (element_type[index] == 0)
      {
         // this coarse element will get pruned, encode who owns it now
         index = -1 - elements[coarse[i]].rank;
      }
      transforms.embeddings[i].parent = index;
   }

   leaf_elements.Copy(old_elements);

   Prune();

   // renumber coarse element indices after pruning
   for (int i = 0; i < coarse.Size(); i++)
   {
      int &index = transforms.embeddings[i].parent;
      if (index >= 0)
      {
         index = elements[old_elements[index]].index;
      }
   }

   // make sure we can delete all send buffers
   NeighborDerefinementMessage::WaitAllSent(send_deref);
}


template<typename Type>
void ParNCMesh::SynchronizeDerefinementData(Array<Type> &elem_data,
                                            const Table &deref_table)
{
   const MPI_Datatype datatype = MPITypeMap<Type>::mpi_type;

   Array<MPI_Request*> requests;
   Array<int> neigh;

   requests.Reserve(64);
   neigh.Reserve(8);

   // make room for ghost values (indices beyond NumElements)
   elem_data.SetSize(leaf_elements.Size(), 0);

   for (int i = 0; i < deref_table.Size(); i++)
   {
      const int* fine = deref_table.GetRow(i);
      int size = deref_table.RowSize(i);
      MFEM_ASSERT(size <= 8, "");

      int ranks[8], min_rank = INT_MAX, max_rank = INT_MIN;
      for (int j = 0; j < size; j++)
      {
         ranks[j] = elements[leaf_elements[fine[j]]].rank;
         min_rank = std::min(min_rank, ranks[j]);
         max_rank = std::max(max_rank, ranks[j]);
      }

      // exchange values for derefinements that straddle processor boundaries
      if (min_rank != max_rank)
      {
         neigh.SetSize(0);
         for (int j = 0; j < size; j++)
         {
            if (ranks[j] != MyRank) { neigh.Append(ranks[j]); }
         }
         neigh.Sort();
         neigh.Unique();

         for (int j = 0; j < size; j++/*pass*/)
         {
            Type *data = &elem_data[fine[j]];

            int rnk = ranks[j], len = 1; /*j;
            do { j++; } while (j < size && ranks[j] == rnk);
            len = j - len;*/

            if (rnk == MyRank)
            {
               for (int k = 0; k < neigh.Size(); k++)
               {
                  MPI_Request* req = new MPI_Request;
                  MPI_Isend(data, len, datatype, neigh[k], 292, MyComm, req);
                  requests.Append(req);
               }
            }
            else
            {
               MPI_Request* req = new MPI_Request;
               MPI_Irecv(data, len, datatype, rnk, 292, MyComm, req);
               requests.Append(req);
            }
         }
      }
   }

   for (int i = 0; i < requests.Size(); i++)
   {
      MPI_Wait(requests[i], MPI_STATUS_IGNORE);
      delete requests[i];
   }
}

// instantiate SynchronizeDerefinementData for int and double
template void
ParNCMesh::SynchronizeDerefinementData<int>(Array<int> &, const Table &);
template void
ParNCMesh::SynchronizeDerefinementData<double>(Array<double> &, const Table &);


void ParNCMesh::CheckDerefinementNCLevel(const Table &deref_table,
                                         Array<int> &level_ok, int max_nc_level)
{
   Array<int> leaf_ok(leaf_elements.Size());
   leaf_ok = 1;

   // check elements that we own
   for (int i = 0; i < deref_table.Size(); i++)
   {
      const int *fine = deref_table.GetRow(i),
                 size = deref_table.RowSize(i);

      int parent = elements[leaf_elements[fine[0]]].parent;
      Element &pa = elements[parent];

      for (int j = 0; j < size; j++)
      {
         int child = leaf_elements[fine[j]];
         if (elements[child].rank == MyRank)
         {
            int splits[3];
            CountSplits(child, splits);

            for (int k = 0; k < Dim; k++)
            {
               if ((pa.ref_type & (1 << k)) &&
                   splits[k] >= max_nc_level)
               {
                  leaf_ok[fine[j]] = 0; break;
               }
            }
         }
      }
   }

   SynchronizeDerefinementData(leaf_ok, deref_table);

   level_ok.SetSize(deref_table.Size());
   level_ok = 1;

   for (int i = 0; i < deref_table.Size(); i++)
   {
      const int* fine = deref_table.GetRow(i),
                 size = deref_table.RowSize(i);

      for (int j = 0; j < size; j++)
      {
         if (!leaf_ok[fine[j]])
         {
            level_ok[i] = 0; break;
         }
      }
   }
}


//// Rebalance /////////////////////////////////////////////////////////////////

void ParNCMesh::Rebalance(const Array<int> *custom_partition)
{
   send_rebalance_dofs.clear();
   recv_rebalance_dofs.clear();

   Array<int> old_elements;
   leaf_elements.GetSubArray(0, NElements, old_elements);

   if (!custom_partition) // SFC based partitioning
   {
      Array<int> new_ranks(leaf_elements.Size());
      new_ranks = -1;

      // figure out new assignments for Element::rank
      long local_elems = NElements, total_elems = 0;
      MPI_Allreduce(&local_elems, &total_elems, 1, MPI_LONG, MPI_SUM, MyComm);

      long first_elem_global = 0;
      MPI_Scan(&local_elems, &first_elem_global, 1, MPI_LONG, MPI_SUM, MyComm);
      first_elem_global -= local_elems;

      for (int i = 0, j = 0; i < leaf_elements.Size(); i++)
      {
         if (elements[leaf_elements[i]].rank == MyRank)
         {
            new_ranks[i] = Partition(first_elem_global + (j++), total_elems);
         }
      }

      int target_elements = PartitionFirstIndex(MyRank+1, total_elems)
                            - PartitionFirstIndex(MyRank, total_elems);

      // assign the new ranks and send elements (plus ghosts) to new owners
      RedistributeElements(new_ranks, target_elements, true);
   }
   else // whatever partitioning the user has passed
   {
      MFEM_VERIFY(custom_partition->Size() == NElements,
                  "Size of the partition array must match the number "
                  "of local mesh elements (ParMesh::GetNE()).");

      Array<int> new_ranks;
      custom_partition->Copy(new_ranks);

      new_ranks.SetSize(leaf_elements.Size(), -1); // make room for ghosts

      RedistributeElements(new_ranks, -1, true);
   }

   // set up the old index array
   old_index_or_rank.SetSize(NElements);
   old_index_or_rank = -1;
   for (int i = 0; i < old_elements.Size(); i++)
   {
      Element &el = elements[old_elements[i]];
      if (el.rank == MyRank) { old_index_or_rank[el.index] = i; }
   }

   // get rid of elements beyond the new ghost layer
   Prune();
}

void ParNCMesh::RedistributeElements(Array<int> &new_ranks, int target_elements,
                                     bool record_comm)
{
   bool sfc = (target_elements >= 0);

   UpdateLayers();

   // *** STEP 1: communicate new rank assignments for the ghost layer ***

   NeighborElementRankMessage::Map send_ghost_ranks, recv_ghost_ranks;

   ghost_layer.Sort([&](const int a, const int b)
   {
      return elements[a].rank < elements[b].rank;
   });

   {
      Array<int> rank_neighbors;

      // loop over neighbor ranks and their elements
      int begin = 0, end = 0;
      while (end < ghost_layer.Size())
      {
         // find range of elements belonging to one rank
         int rank = elements[ghost_layer[begin]].rank;
         while (end < ghost_layer.Size() &&
                elements[ghost_layer[end]].rank == rank) { end++; }

         Array<int> rank_elems;
         rank_elems.MakeRef(&ghost_layer[begin], end - begin);

         // find elements within boundary_layer that are neighbors to 'rank'
         rank_neighbors.SetSize(0);
         NeighborExpand(rank_elems, rank_neighbors, &boundary_layer);

         // send a message with new rank assignments within 'rank_neighbors'
         NeighborElementRankMessage& msg = send_ghost_ranks[rank];
         msg.SetNCMesh(this);

         msg.Reserve(rank_neighbors.Size());
         for (int i = 0; i < rank_neighbors.Size(); i++)
         {
            int elem = rank_neighbors[i];
            msg.AddElementRank(elem, new_ranks[elements[elem].index]);
         }

         msg.Isend(rank, MyComm);

         // prepare to receive a message from the neighbor too, these will
         // be new the new rank assignments for our ghost layer
         recv_ghost_ranks[rank].SetNCMesh(this);

         begin = end;
      }
   }

   NeighborElementRankMessage::RecvAll(recv_ghost_ranks, MyComm);

   // read new ranks for the ghost layer from messages received
   NeighborElementRankMessage::Map::iterator it;
   for (it = recv_ghost_ranks.begin(); it != recv_ghost_ranks.end(); ++it)
   {
      NeighborElementRankMessage &msg = it->second;
      for (int i = 0; i < msg.Size(); i++)
      {
         int ghost_index = elements[msg.elements[i]].index;
         MFEM_ASSERT(element_type[ghost_index] == 2, "");
         new_ranks[ghost_index] = msg.values[i];
      }
   }

   recv_ghost_ranks.clear();

   // *** STEP 2: send elements that no longer belong to us to new assignees ***

   /* The result thus far is just the array 'new_ranks' containing new owners
      for elements that we currently own plus new owners for the ghost layer.
      Next we keep elements that still belong to us and send ElementSets with
      the remaining elements to their new owners. Each batch of elements needs
      to be sent together with their neighbors so the receiver also gets a
      ghost layer that is up to date (this is why we needed Step 1). */

   int received_elements = 0;
   for (int i = 0; i < leaf_elements.Size(); i++)
   {
      Element &el = elements[leaf_elements[i]];
      if (el.rank == MyRank && new_ranks[i] == MyRank)
      {
         received_elements++; // initialize to number of elements we're keeping
      }
      el.rank = new_ranks[i];
   }

   int nsent = 0, nrecv = 0; // for debug check

   RebalanceMessage::Map send_elems;
   {
      // sort elements we own by the new rank
      Array<int> owned_elements;
      owned_elements.MakeRef(leaf_elements.GetData(), NElements);
      owned_elements.Sort([&](const int a, const int b)
      {
         return elements[a].rank < elements[b].rank;
      });

      Array<int> batch;
      batch.Reserve(1024);

      // send elements to new owners
      int begin = 0, end = 0;
      while (end < NElements)
      {
         // find range of elements belonging to one rank
         int rank = elements[owned_elements[begin]].rank;
         while (end < owned_elements.Size() &&
                elements[owned_elements[end]].rank == rank) { end++; }

         if (rank != MyRank)
         {
            Array<int> rank_elems;
            rank_elems.MakeRef(&owned_elements[begin], end - begin);

            // expand the 'rank_elems' set by its neighbor elements (ghosts)
            batch.SetSize(0);
            NeighborExpand(rank_elems, batch);

            // send the batch
            RebalanceMessage &msg = send_elems[rank];
            msg.SetNCMesh(this);

            msg.Reserve(batch.Size());
            for (int i = 0; i < batch.Size(); i++)
            {
               int elem = batch[i];
               Element &el = elements[elem];

               if ((element_type[el.index] & 1) || el.rank != rank)
               {
                  msg.AddElementRank(elem, el.rank);
               }
               // NOTE: we skip 'ghosts' that are of the receiver's rank because
               // they are not really ghosts and would get sent multiple times,
               // disrupting the termination mechanism in Step 4.
            }

            if (sfc)
            {
               msg.Isend(rank, MyComm);
            }
            else
            {
               // custom partitioning needs synchronous sends
               msg.Issend(rank, MyComm);
            }
            nsent++;

            // also: record what elements we sent (excluding the ghosts)
            // so that SendRebalanceDofs can later send data for them
            if (record_comm)
            {
               send_rebalance_dofs[rank].SetElements(rank_elems, this);
            }
         }

         begin = end;
      }
   }

   // *** STEP 3: receive elements from others ***

   RebalanceMessage msg;
   msg.SetNCMesh(this);

   if (sfc)
   {
      /* We don't know from whom we're going to receive, so we need to probe.
         However, for the default SFC partitioning, we do know how many elements
         we're going to own eventually, so the termination condition is easy. */

      while (received_elements < target_elements)
      {
         int rank, size;
         RebalanceMessage::Probe(rank, size, MyComm);

         // receive message; note: elements are created as the message is decoded
         msg.Recv(rank, size, MyComm);
         nrecv++;

         for (int i = 0; i < msg.Size(); i++)
         {
            int elem_rank = msg.values[i];
            elements[msg.elements[i]].rank = elem_rank;

            if (elem_rank == MyRank) { received_elements++; }
         }

         // save the ranks we received from, for later use in RecvRebalanceDofs
         if (record_comm)
         {
            recv_rebalance_dofs[rank].SetNCMesh(this);
         }
      }

      Update();

      RebalanceMessage::WaitAllSent(send_elems);
   }
   else
   {
      /* The case (target_elements < 0) is used for custom partitioning.
         Here we need to employ the "non-blocking consensus" algorithm
         (https://scorec.rpi.edu/REPORTS/2015-9.pdf) to determine when the
         element exchange is finished. The algorithm uses a non-blocking
         barrier. */

      MPI_Request barrier = MPI_REQUEST_NULL;
      int done = 0;

      while (!done)
      {
         int rank, size;
         while (RebalanceMessage::IProbe(rank, size, MyComm))
         {
            // receive message; note: elements are created as the msg is decoded
            msg.Recv(rank, size, MyComm);
            nrecv++;

            for (int i = 0; i < msg.Size(); i++)
            {
               elements[msg.elements[i]].rank = msg.values[i];
            }

            // save the ranks we received from, for later use in RecvRebalanceDofs
            if (record_comm)
            {
               recv_rebalance_dofs[rank].SetNCMesh(this);
            }
         }

         if (barrier != MPI_REQUEST_NULL)
         {
            MPI_Test(&barrier, &done, MPI_STATUS_IGNORE);
         }
         else
         {
            if (RebalanceMessage::TestAllSent(send_elems))
            {
               int err = MPI_Ibarrier(MyComm, &barrier);

               MFEM_VERIFY(err == MPI_SUCCESS, "");
               MFEM_VERIFY(barrier != MPI_REQUEST_NULL, "");
            }
         }
      }

      Update();
   }

   NeighborElementRankMessage::WaitAllSent(send_ghost_ranks);

#ifdef MFEM_DEBUG
   int glob_sent, glob_recv;
   MPI_Reduce(&nsent, &glob_sent, 1, MPI_INT, MPI_SUM, 0, MyComm);
   MPI_Reduce(&nrecv, &glob_recv, 1, MPI_INT, MPI_SUM, 0, MyComm);

   if (MyRank == 0)
   {
      MFEM_ASSERT(glob_sent == glob_recv,
                  "(glob_sent, glob_recv) = ("
                  << glob_sent << ", " << glob_recv << ")");
   }
#endif
}


void ParNCMesh::SendRebalanceDofs(int old_ndofs,
                                  const Table &old_element_dofs,
                                  long old_global_offset,
                                  FiniteElementSpace *space)
{
   Array<int> dofs;
   int vdim = space->GetVDim();

   // fill messages (prepared by Rebalance) with element DOFs
   RebalanceDofMessage::Map::iterator it;
   for (it = send_rebalance_dofs.begin(); it != send_rebalance_dofs.end(); ++it)
   {
      RebalanceDofMessage &msg = it->second;
      msg.dofs.clear();
      int ne = msg.elem_ids.size();
      if (ne)
      {
         msg.dofs.reserve(old_element_dofs.RowSize(msg.elem_ids[0]) * ne * vdim);
      }
      for (int i = 0; i < ne; i++)
      {
         old_element_dofs.GetRow(msg.elem_ids[i], dofs);
         space->DofsToVDofs(dofs, old_ndofs);
         msg.dofs.insert(msg.dofs.end(), dofs.begin(), dofs.end());
      }
      msg.dof_offset = old_global_offset;
   }

   // send the DOFs to element recipients from last Rebalance()
   RebalanceDofMessage::IsendAll(send_rebalance_dofs, MyComm);
}


void ParNCMesh::RecvRebalanceDofs(Array<int> &elements, Array<long> &dofs)
{
   // receive from the same ranks as in last Rebalance()
   RebalanceDofMessage::RecvAll(recv_rebalance_dofs, MyComm);

   // count the size of the result
   int ne = 0, nd = 0;
   RebalanceDofMessage::Map::iterator it;
   for (it = recv_rebalance_dofs.begin(); it != recv_rebalance_dofs.end(); ++it)
   {
      RebalanceDofMessage &msg = it->second;
      ne += msg.elem_ids.size();
      nd += msg.dofs.size();
   }

   elements.SetSize(ne);
   dofs.SetSize(nd);

   // copy element indices and their DOFs
   ne = nd = 0;
   for (it = recv_rebalance_dofs.begin(); it != recv_rebalance_dofs.end(); ++it)
   {
      RebalanceDofMessage &msg = it->second;
      for (unsigned i = 0; i < msg.elem_ids.size(); i++)
      {
         elements[ne++] = msg.elem_ids[i];
      }
      for (unsigned i = 0; i < msg.dofs.size(); i++)
      {
         dofs[nd++] = msg.dof_offset + msg.dofs[i];
      }
   }

   RebalanceDofMessage::WaitAllSent(send_rebalance_dofs);
}


//// ElementSet ////////////////////////////////////////////////////////////////

ParNCMesh::ElementSet::ElementSet(const ElementSet &other)
   : ncmesh(other.ncmesh), include_ref_types(other.include_ref_types)
{
   other.data.Copy(data);
}

void ParNCMesh::ElementSet::WriteInt(int value)
{
   // helper to put an int to the data array
   data.Append(value & 0xff);
   data.Append((value >> 8) & 0xff);
   data.Append((value >> 16) & 0xff);
   data.Append((value >> 24) & 0xff);
}

int ParNCMesh::ElementSet::GetInt(int pos) const
{
   // helper to get an int from the data array
   return (int) data[pos] +
          ((int) data[pos+1] << 8) +
          ((int) data[pos+2] << 16) +
          ((int) data[pos+3] << 24);
}

void ParNCMesh::ElementSet::FlagElements(const Array<int> &elements, char flag)
{
   for (int i = 0; i < elements.Size(); i++)
   {
      int elem = elements[i];
      while (elem >= 0)
      {
         Element &el = ncmesh->elements[elem];
         if (el.flag == flag) { break; }
         el.flag = flag;
         elem = el.parent;
      }
   }
}

void ParNCMesh::ElementSet::EncodeTree(int elem)
{
   Element &el = ncmesh->elements[elem];
   if (!el.ref_type)
   {
      // we reached a leaf, mark this as zero child mask
      data.Append(0);
   }
   else
   {
      // check which subtrees contain marked elements
      int mask = 0;
      for (int i = 0; i < 8; i++)
      {
         if (el.child[i] >= 0 && ncmesh->elements[el.child[i]].flag)
         {
            mask |= 1 << i;
         }
      }

      // write the bit mask and visit the subtrees
      data.Append(mask);
      if (include_ref_types)
      {
         data.Append(el.ref_type);
      }

      for (int i = 0; i < 8; i++)
      {
         if (mask & (1 << i))
         {
            EncodeTree(el.child[i]);
         }
      }
   }
}

void ParNCMesh::ElementSet::Encode(const Array<int> &elements)
{
   FlagElements(elements, 1);

   // Each refinement tree that contains at least one element from the set
   // is encoded as HEADER + TREE, where HEADER is the root element number and
   // TREE is the output of EncodeTree().
   for (int i = 0; i < ncmesh->root_state.Size(); i++)
   {
      if (ncmesh->elements[i].flag)
      {
         WriteInt(i);
         EncodeTree(i);
      }
   }
   WriteInt(-1); // mark end of data

   FlagElements(elements, 0);
}

#ifdef MFEM_DEBUG
std::string ParNCMesh::ElementSet::RefPath() const
{
   std::ostringstream oss;
   for (int i = 0; i < ref_path.Size(); i++)
   {
      oss << "     elem " << ref_path[i] << " (";
      const Element &el = ncmesh->elements[ref_path[i]];
      for (int j = 0; j < GI[el.Geom()].nv; j++)
      {
         if (j) { oss << ", "; }
         oss << ncmesh->RetrieveNode(el, j);
      }
      oss << ")\n";
   }
   return oss.str();
}
#endif

void ParNCMesh::ElementSet::DecodeTree(int elem, int &pos,
                                       Array<int> &elements) const
{
#ifdef MFEM_DEBUG
   ref_path.Append(elem);
#endif
   int mask = data[pos++];
   if (!mask)
   {
      elements.Append(elem);
   }
   else
   {
      Element &el = ncmesh->elements[elem];
      if (include_ref_types)
      {
         int ref_type = data[pos++];
         if (!el.ref_type)
         {
            ncmesh->RefineElement(elem, ref_type);
         }
         else { MFEM_ASSERT(ref_type == el.ref_type, "") }
      }
      else
      {
         MFEM_ASSERT(el.ref_type != 0, "Path not found:\n"
                     << RefPath() << "     mask = " << mask);
      }

      for (int i = 0; i < 8; i++)
      {
         if (mask & (1 << i))
         {
            DecodeTree(el.child[i], pos, elements);
         }
      }
   }
#ifdef MFEM_DEBUG
   ref_path.DeleteLast();
#endif
}

void ParNCMesh::ElementSet::Decode(Array<int> &elements) const
{
   int root, pos = 0;
   while ((root = GetInt(pos)) >= 0)
   {
      pos += 4;
      DecodeTree(root, pos, elements);
   }
}

void ParNCMesh::ElementSet::Dump(std::ostream &os) const
{
   write<int>(os, data.Size());
   os.write((const char*) data.GetData(), data.Size());
}

void ParNCMesh::ElementSet::Load(std::istream &is)
{
   data.SetSize(read<int>(is));
   is.read((char*) data.GetData(), data.Size());
}


//// EncodeMeshIds/DecodeMeshIds ///////////////////////////////////////////////

void ParNCMesh::AdjustMeshIds(Array<MeshId> ids[], int rank)
{
   GetSharedVertices();
   GetSharedEdges();
   GetSharedFaces();

   if (!shared_edges.masters.Size() &&
       !shared_faces.masters.Size()) { return; }

   Array<bool> contains_rank(groups.size());
   for (unsigned i = 0; i < groups.size(); i++)
   {
      contains_rank[i] = GroupContains(i, rank);
   }

   Array<Pair<int, int> > find_v(ids[0].Size());
   for (int i = 0; i < ids[0].Size(); i++)
   {
      find_v[i].one = ids[0][i].index;
      find_v[i].two = i;
   }
   find_v.Sort();

   // find vertices of master edges shared with 'rank', and modify their
   // MeshIds so their element/local matches the element of the master edge
   for (int i = 0; i < shared_edges.masters.Size(); i++)
   {
      const MeshId &edge_id = shared_edges.masters[i];
      if (contains_rank[entity_pmat_group[1][edge_id.index]])
      {
         int v[2], pos, k;
         GetEdgeVertices(edge_id, v);
         for (int j = 0; j < 2; j++)
         {
            if ((pos = find_v.FindSorted(Pair<int, int>(v[j], 0))) != -1)
            {
               // switch to an element/local that is safe for 'rank'
               k = find_v[pos].two;
               ChangeVertexMeshIdElement(ids[0][k], edge_id.element);
               ChangeRemainingMeshIds(ids[0], pos, find_v);
            }
         }
      }
   }

   if (!shared_faces.masters.Size()) { return; }

   Array<Pair<int, int> > find_e(ids[1].Size());
   for (int i = 0; i < ids[1].Size(); i++)
   {
      find_e[i].one = ids[1][i].index;
      find_e[i].two = i;
   }
   find_e.Sort();

   // find vertices/edges of master faces shared with 'rank', and modify their
   // MeshIds so their element/local matches the element of the master face
   for (int i = 0; i < shared_faces.masters.Size(); i++)
   {
      const MeshId &face_id = shared_faces.masters[i];
      if (contains_rank[entity_pmat_group[2][face_id.index]])
      {
         int v[4], e[4], eo[4], pos, k;
         int nfv = GetFaceVerticesEdges(face_id, v, e, eo);
         for (int j = 0; j < nfv; j++)
         {
            if ((pos = find_v.FindSorted(Pair<int, int>(v[j], 0))) != -1)
            {
               k = find_v[pos].two;
               ChangeVertexMeshIdElement(ids[0][k], face_id.element);
               ChangeRemainingMeshIds(ids[0], pos, find_v);
            }
            if ((pos = find_e.FindSorted(Pair<int, int>(e[j], 0))) != -1)
            {
               k = find_e[pos].two;
               ChangeEdgeMeshIdElement(ids[1][k], face_id.element);
               ChangeRemainingMeshIds(ids[1], pos, find_e);
            }
         }
      }
   }
}

void ParNCMesh::ChangeVertexMeshIdElement(NCMesh::MeshId &id, int elem)
{
   Element &el = elements[elem];
   MFEM_ASSERT(el.ref_type == 0, "");

   GeomInfo& gi = GI[el.Geom()];
   for (int i = 0; i < gi.nv; i++)
   {
      if (nodes[el.node[i]].vert_index == id.index)
      {
         id.local = i;
         id.element = elem;
         return;
      }
   }
   MFEM_ABORT("Vertex not found.");
}

void ParNCMesh::ChangeEdgeMeshIdElement(NCMesh::MeshId &id, int elem)
{
   Element &old = elements[id.element];
   const int *ev = GI[old.Geom()].edges[(int) id.local];
   Node* node = nodes.Find(old.node[ev[0]], old.node[ev[1]]);
   MFEM_ASSERT(node != NULL, "Edge not found.");

   Element &el = elements[elem];
   MFEM_ASSERT(el.ref_type == 0, "");

   GeomInfo& gi = GI[el.Geom()];
   for (int i = 0; i < gi.ne; i++)
   {
      const int* ev = gi.edges[i];
      if ((el.node[ev[0]] == node->p1 && el.node[ev[1]] == node->p2) ||
          (el.node[ev[1]] == node->p1 && el.node[ev[0]] == node->p2))
      {
         id.local = i;
         id.element = elem;
         return;
      }

   }
   MFEM_ABORT("Edge not found.");
}

void ParNCMesh::ChangeRemainingMeshIds(Array<MeshId> &ids, int pos,
                                       const Array<Pair<int, int> > &find)
{
   const MeshId &first = ids[find[pos].two];
   while (++pos < find.Size() && ids[find[pos].two].index == first.index)
   {
      MeshId &other = ids[find[pos].two];
      other.element = first.element;
      other.local = first.local;
   }
}

void ParNCMesh::EncodeMeshIds(std::ostream &os, Array<MeshId> ids[])
{
   std::map<int, int> stream_id;

   // get a list of elements involved, dump them to 'os' and create the mapping
   // element_id: (Element index -> stream ID)
   {
      Array<int> elements;
      for (int type = 0; type < 3; type++)
      {
         for (int i = 0; i < ids[type].Size(); i++)
         {
            elements.Append(ids[type][i].element);
         }
      }

      ElementSet eset(this);
      eset.Encode(elements);
      eset.Dump(os);

      Array<int> decoded;
      decoded.Reserve(elements.Size());
      eset.Decode(decoded);

      for (int i = 0; i < decoded.Size(); i++)
      {
         stream_id[decoded[i]] = i;
      }
   }

   // write the IDs as element/local pairs
   for (int type = 0; type < 3; type++)
   {
      write<int>(os, ids[type].Size());
      for (int i = 0; i < ids[type].Size(); i++)
      {
         const MeshId& id = ids[type][i];
         write<int>(os, stream_id[id.element]); // TODO: variable 1-4 bytes
         write<char>(os, id.local);
      }
   }
}

void ParNCMesh::DecodeMeshIds(std::istream &is, Array<MeshId> ids[])
{
   // read the list of elements
   ElementSet eset(this);
   eset.Load(is);

   Array<int> elems;
   eset.Decode(elems);

   // read vertex/edge/face IDs
   for (int type = 0; type < 3; type++)
   {
      int ne = read<int>(is);
      ids[type].SetSize(ne);

      for (int i = 0; i < ne; i++)
      {
         int el_num = read<int>(is);
         int elem = elems[el_num];
         Element &el = elements[elem];

         MFEM_VERIFY(!el.ref_type, "not a leaf element: " << el_num);

         MeshId &id = ids[type][i];
         id.element = elem;
         id.local = read<char>(is);

         // find vertex/edge/face index
         GeomInfo &gi = GI[el.Geom()];
         switch (type)
         {
            case 0:
            {
               id.index = nodes[el.node[(int) id.local]].vert_index;
               break;
            }
            case 1:
            {
               const int* ev = gi.edges[(int) id.local];
               Node* node = nodes.Find(el.node[ev[0]], el.node[ev[1]]);
               MFEM_ASSERT(node && node->HasEdge(), "edge not found.");
               id.index = node->edge_index;
               break;
            }
            default:
            {
               const int* fv = gi.faces[(int) id.local];
               Face* face = faces.Find(el.node[fv[0]], el.node[fv[1]],
                                       el.node[fv[2]], el.node[fv[3]]);
               MFEM_ASSERT(face, "face not found.");
               id.index = face->index;
            }
         }
      }
   }
}

void ParNCMesh::EncodeGroups(std::ostream &os, const Array<GroupId> &ids)
{
   // get a list of unique GroupIds
   std::map<GroupId, GroupId> stream_id;
   for (int i = 0; i < ids.Size(); i++)
   {
      if (i && ids[i] == ids[i-1]) { continue; }
      unsigned size = stream_id.size();
      GroupId &sid = stream_id[ids[i]];
      if (size != stream_id.size()) { sid = size; }
   }

   // write the unique groups
   write<short>(os, stream_id.size());
   for (std::map<GroupId, GroupId>::iterator
        it = stream_id.begin(); it != stream_id.end(); ++it)
   {
      write<GroupId>(os, it->second);
      if (it->first >= 0)
      {
         const CommGroup &group = groups[it->first];
         write<short>(os, group.size());
         for (unsigned i = 0; i < group.size(); i++)
         {
            write<int>(os, group[i]);
         }
      }
      else
      {
         // special "invalid" group, marks forwarded rows
         write<short>(os, -1);
      }
   }

   // write the list of all GroupIds
   write<int>(os, ids.Size());
   for (int i = 0; i < ids.Size(); i++)
   {
      write<GroupId>(os, stream_id[ids[i]]);
   }
}

void ParNCMesh::DecodeGroups(std::istream &is, Array<GroupId> &ids)
{
   int ngroups = read<short>(is);
   Array<GroupId> groups(ngroups);

   // read stream groups, convert to our groups
   CommGroup ranks;
   ranks.reserve(128);
   for (int i = 0; i < ngroups; i++)
   {
      int id = read<GroupId>(is);
      int size = read<short>(is);
      if (size >= 0)
      {
         ranks.resize(size);
         for (int i = 0; i < size; i++)
         {
            ranks[i] = read<int>(is);
         }
         groups[id] = GetGroupId(ranks);
      }
      else
      {
         groups[id] = -1; // forwarded
      }
   }

   // read the list of IDs
   ids.SetSize(read<int>(is));
   for (int i = 0; i < ids.Size(); i++)
   {
      ids[i] = groups[read<GroupId>(is)];
   }
}


//// Messages //////////////////////////////////////////////////////////////////

template<class ValueType, bool RefTypes, int Tag>
void ParNCMesh::ElementValueMessage<ValueType, RefTypes, Tag>::Encode(int)
{
   std::ostringstream ostream;

   Array<int> tmp_elements;
   tmp_elements.MakeRef(elements.data(), elements.size());

   ElementSet eset(pncmesh, RefTypes);
   eset.Encode(tmp_elements);
   eset.Dump(ostream);

   // decode the element set to obtain a local numbering of elements
   Array<int> decoded;
   decoded.Reserve(tmp_elements.Size());
   eset.Decode(decoded);

   std::map<int, int> element_index;
   for (int i = 0; i < decoded.Size(); i++)
   {
      element_index[decoded[i]] = i;
   }

   write<int>(ostream, values.size());
   MFEM_ASSERT(elements.size() == values.size(), "");

   for (unsigned i = 0; i < values.size(); i++)
   {
      write<int>(ostream, element_index[elements[i]]); // element number
      write<ValueType>(ostream, values[i]);
   }

   ostream.str().swap(data);
}

template<class ValueType, bool RefTypes, int Tag>
void ParNCMesh::ElementValueMessage<ValueType, RefTypes, Tag>::Decode(int)
{
   std::istringstream istream(data);

   ElementSet eset(pncmesh, RefTypes);
   eset.Load(istream);

   Array<int> tmp_elements;
   eset.Decode(tmp_elements);

   int* el = tmp_elements.GetData();
   elements.assign(el, el + tmp_elements.Size());
   values.resize(elements.size());

   int count = read<int>(istream);
   for (int i = 0; i < count; i++)
   {
      int index = read<int>(istream);
      MFEM_ASSERT(index >= 0 && (size_t) index < values.size(), "");
      values[index] = read<ValueType>(istream);
   }

   // no longer need the raw data
   data.clear();
}

void ParNCMesh::RebalanceDofMessage::SetElements(const Array<int> &elems,
                                                 NCMesh *ncmesh)
{
   eset.SetNCMesh(ncmesh);
   eset.Encode(elems);

   Array<int> decoded;
   decoded.Reserve(elems.Size());
   eset.Decode(decoded);

   elem_ids.resize(decoded.Size());
   for (int i = 0; i < decoded.Size(); i++)
   {
      elem_ids[i] = eset.GetNCMesh()->elements[decoded[i]].index;
   }
}

static void write_dofs(std::ostream &os, const std::vector<int> &dofs)
{
   write<int>(os, dofs.size());
   // TODO: we should compress the ints, mostly they are contiguous ranges
   os.write((const char*) dofs.data(), dofs.size() * sizeof(int));
}

static void read_dofs(std::istream &is, std::vector<int> &dofs)
{
   dofs.resize(read<int>(is));
   is.read((char*) dofs.data(), dofs.size() * sizeof(int));
}

void ParNCMesh::RebalanceDofMessage::Encode(int)
{
   std::ostringstream stream;

   eset.Dump(stream);
   write<long>(stream, dof_offset);
   write_dofs(stream, dofs);

   stream.str().swap(data);
}

void ParNCMesh::RebalanceDofMessage::Decode(int)
{
   std::istringstream stream(data);

   eset.Load(stream);
   dof_offset = read<long>(stream);
   read_dofs(stream, dofs);

   data.clear();

   Array<int> elems;
   eset.Decode(elems);

   elem_ids.resize(elems.Size());
   for (int i = 0; i < elems.Size(); i++)
   {
      elem_ids[i] = eset.GetNCMesh()->elements[elems[i]].index;
   }
}


//// Utility ///////////////////////////////////////////////////////////////////

void ParNCMesh::GetDebugMesh(Mesh &debug_mesh) const
{
   // create a serial NCMesh containing all our elements (ghosts and all)
   NCMesh* copy = new NCMesh(*this);

   Array<int> &cle = copy->leaf_elements;
   for (int i = 0; i < cle.Size(); i++)
   {
      Element &el = copy->elements[cle[i]];
      el.attribute = el.rank + 1;
   }

   debug_mesh.InitFromNCMesh(*copy);
   debug_mesh.SetAttributes();
   debug_mesh.ncmesh = copy;
}

void ParNCMesh::Trim()
{
   NCMesh::Trim();

   shared_vertices.Clear();
   shared_edges.Clear();
   shared_faces.Clear();

   for (int i = 0; i < 3; i++)
   {
      entity_owner[i].DeleteAll();
      entity_pmat_group[i].DeleteAll();
      entity_index_rank[i].DeleteAll();
   }

   send_rebalance_dofs.clear();
   recv_rebalance_dofs.clear();

   old_index_or_rank.DeleteAll();

   ClearAuxPM();
}

long ParNCMesh::RebalanceDofMessage::MemoryUsage() const
{
   return (elem_ids.capacity() + dofs.capacity()) * sizeof(int);
}

template<typename K, typename V>
static long map_memory_usage(const std::map<K, V> &map)
{
   long result = 0;
   for (typename std::map<K, V>::const_iterator
        it = map.begin(); it != map.end(); ++it)
   {
      result += it->second.MemoryUsage();
      result += sizeof(std::pair<K, V>) + 3*sizeof(void*) + sizeof(bool);
   }
   return result;
}

long ParNCMesh::GroupsMemoryUsage() const
{
   long groups_size = groups.capacity() * sizeof(CommGroup);
   for (unsigned i = 0; i < groups.size(); i++)
   {
      groups_size += groups[i].capacity() * sizeof(int);
   }
   const int approx_node_size =
      sizeof(std::pair<CommGroup, GroupId>) + 3*sizeof(void*) + sizeof(bool);
   return groups_size + group_id.size() * approx_node_size;
}

template<typename Type, int Size>
static long arrays_memory_usage(const Array<Type> (&arrays)[Size])
{
   long total = 0;
   for (int i = 0; i < Size; i++)
   {
      total += arrays[i].MemoryUsage();
   }
   return total;
}

long ParNCMesh::MemoryUsage(bool with_base) const
{
   long total_groups_owners = 0;
   for (int i = 0; i < 3; i++)
   {
      total_groups_owners += entity_owner[i].MemoryUsage() +
                             entity_pmat_group[i].MemoryUsage() +
                             entity_index_rank[i].MemoryUsage();
   }

   return (with_base ? NCMesh::MemoryUsage() : 0) +
          GroupsMemoryUsage() +
          arrays_memory_usage(entity_owner) +
          arrays_memory_usage(entity_pmat_group) +
          arrays_memory_usage(entity_conf_group) +
          leaf_glob_order.MemoryUsage() +
          arrays_memory_usage(entity_elem_local) +
          shared_vertices.MemoryUsage() +
          shared_edges.MemoryUsage() +
          shared_faces.MemoryUsage() +
          face_orient.MemoryUsage() +
          element_type.MemoryUsage() +
          ghost_layer.MemoryUsage() +
          boundary_layer.MemoryUsage() +
          tmp_owner.MemoryUsage() +
          tmp_shared_flag.MemoryUsage() +
          arrays_memory_usage(entity_index_rank) +
          tmp_neighbors.MemoryUsage() +
          map_memory_usage(send_rebalance_dofs) +
          map_memory_usage(recv_rebalance_dofs) +
          old_index_or_rank.MemoryUsage() +
          aux_pm_store.MemoryUsage() +
          sizeof(ParNCMesh) - sizeof(NCMesh);
}

int ParNCMesh::PrintMemoryDetail(bool with_base) const
{
   if (with_base) { NCMesh::PrintMemoryDetail(); }

   mfem::out << GroupsMemoryUsage() << " groups\n"
             << arrays_memory_usage(entity_owner) << " entity_owner\n"
             << arrays_memory_usage(entity_pmat_group) << " entity_pmat_group\n"
             << arrays_memory_usage(entity_conf_group) << " entity_conf_group\n"
             << leaf_glob_order.MemoryUsage() << " leaf_glob_order\n"
             << arrays_memory_usage(entity_elem_local) << " entity_elem_local\n"
             << shared_vertices.MemoryUsage() << " shared_vertices\n"
             << shared_edges.MemoryUsage() << " shared_edges\n"
             << shared_faces.MemoryUsage() << " shared_faces\n"
             << face_orient.MemoryUsage() << " face_orient\n"
             << element_type.MemoryUsage() << " element_type\n"
             << ghost_layer.MemoryUsage() << " ghost_layer\n"
             << boundary_layer.MemoryUsage() << " boundary_layer\n"
             << tmp_owner.MemoryUsage() << " tmp_owner\n"
             << tmp_shared_flag.MemoryUsage() << " tmp_shared_flag\n"
             << arrays_memory_usage(entity_index_rank) << " entity_index_rank\n"
             << tmp_neighbors.MemoryUsage() << " tmp_neighbors\n"
             << map_memory_usage(send_rebalance_dofs) << " send_rebalance_dofs\n"
             << map_memory_usage(recv_rebalance_dofs) << " recv_rebalance_dofs\n"
             << old_index_or_rank.MemoryUsage() << " old_index_or_rank\n"
             << aux_pm_store.MemoryUsage() << " aux_pm_store\n"
             << sizeof(ParNCMesh) - sizeof(NCMesh) << " ParNCMesh" << std::endl;

   return leaf_elements.Size();
}

} // namespace mfem

#endif // MFEM_USE_MPI
