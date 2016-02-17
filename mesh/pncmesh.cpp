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

#include "../config/config.hpp"

#ifdef MFEM_USE_MPI

#include "mesh_headers.hpp"
#include "pncmesh.hpp"
#include "../fem/fe_coll.hpp"

#include <map>
#include <limits>

namespace mfem
{

ParNCMesh::ParNCMesh(MPI_Comm comm, const NCMesh &ncmesh)
   : NCMesh(ncmesh)
{
   MyComm = comm;
   MPI_Comm_size(MyComm, &NRanks);
   MPI_Comm_rank(MyComm, &MyRank);

   // assign leaf elements to the 'NRanks' processors
   for (int i = 0; i < leaf_elements.Size(); i++)
   {
      leaf_elements[i]->rank = InitialPartition(i);
   }

   AssignLeafIndices();
   UpdateVertices();
}

void ParNCMesh::Update()
{
   NCMesh::Update();

   shared_vertices.Clear();
   shared_edges.Clear();
   shared_faces.Clear();

   element_type.SetSize(0);
   ghost_layer.SetSize(0);
}

void ParNCMesh::AssignLeafIndices()
{
   // This is an override of NCMesh::AssignLeafIndices(). The difference is
   // that we shift all elements we own to the beginning of the array
   // 'leaf_elements' and assign all ghost elements indices >= NElements. This
   // will make the ghosts skipped in NCMesh::GetMeshComponents.

   NElements = NGhostElements = 0;
   for (int i = 0; i < leaf_elements.Size(); i++)
   {
      if (leaf_elements[i]->rank == MyRank)
      {
         std::swap(leaf_elements[NElements++], leaf_elements[i]);
      }
      else
      {
         NGhostElements++;
      }
   }

   for (int i = 0; i < leaf_elements.Size(); i++)
   {
      leaf_elements[i]->index = i;
   }
}

void ParNCMesh::UpdateVertices()
{
   // This is an override of NCMesh::UpdateVertices. This version first
   // assigns Vertex::index to vertices of elements of our rank. Only these
   // vertices then make it to the Mesh in NCMesh::GetMeshComponents.
   // The remaining (ghost) vertices are assigned indices greater or equal to
   // Mesh::GetNV().

   for (HashTable<Node>::Iterator it(nodes); it; ++it)
   {
      if (it->vertex) { it->vertex->index = -1; }
   }

   NVertices = 0;
   for (int i = 0; i < leaf_elements.Size(); i++)
   {
      Element* elem = leaf_elements[i];
      if (elem->rank == MyRank)
      {
         for (int j = 0; j < GI[(int) elem->geom].nv; j++)
         {
            int &vindex = elem->node[j]->vertex->index;
            if (vindex < 0) { vindex = NVertices++; }
         }
      }
   }

   vertex_nodeId.SetSize(NVertices);
   for (HashTable<Node>::Iterator it(nodes); it; ++it)
   {
      if (it->vertex && it->vertex->index >= 0)
      {
         vertex_nodeId[it->vertex->index] = it->id;
      }
   }

   NGhostVertices = 0;
   for (HashTable<Node>::Iterator it(nodes); it; ++it)
   {
      if (it->vertex && it->vertex->index < 0)
      {
         it->vertex->index = NVertices + (NGhostVertices++);
      }
   }
}

void ParNCMesh::OnMeshUpdated(Mesh *mesh)
{
   // This is an override (or extension of) NCMesh::OnMeshUpdated().
   // In addition to getting edge/face indices from 'mesh', we also
   // assign indices to ghost edges/faces that don't exist in the 'mesh'.

   // clear Edge:: and Face::index
   for (HashTable<Node>::Iterator it(nodes); it; ++it)
   {
      if (it->edge) { it->edge->index = -1; }
   }
   for (HashTable<Face>::Iterator it(faces); it; ++it) { it->index = -1; }

   // go assign existing edge/face indices
   NCMesh::OnMeshUpdated(mesh);

   // assign ghost edge indices
   NEdges = mesh->GetNEdges();
   NGhostEdges = 0;
   for (HashTable<Node>::Iterator it(nodes); it; ++it)
   {
      if (it->edge && it->edge->index < 0)
      {
         it->edge->index = NEdges + (NGhostEdges++);
      }
   }

   // assign ghost face indices
   NFaces = mesh->GetNFaces();
   NGhostFaces = 0;
   for (HashTable<Face>::Iterator it(faces); it; ++it)
   {
      if (it->index < 0) { it->index = NFaces + (NGhostFaces++); }
   }
}

void ParNCMesh::ElementSharesEdge(Element *elem, Edge *edge)
{
   // Called by NCMesh::BuildEdgeList when an edge is visited in a leaf element.
   // This allows us to determine edge ownership and processors that share it
   // without duplicating all the HashTable lookups in NCMesh::BuildEdgeList().

   int &owner = edge_owner[edge->index];
   owner = std::min(owner, elem->rank);

   index_rank.Append(Connection(edge->index, elem->rank));
}

void ParNCMesh::ElementSharesFace(Element* elem, Face* face)
{
   // Analogous to ElementHasEdge.

   int &owner = face_owner[face->index];
   owner = std::min(owner, elem->rank);

   index_rank.Append(Connection(face->index, elem->rank));
}

void ParNCMesh::BuildEdgeList()
{
   // This is an extension of NCMesh::BuildEdgeList() which also determines
   // edge ownership, creates edge processor groups and lists shared edges.

   int nedges = NEdges + NGhostEdges;
   edge_owner.SetSize(nedges);
   edge_owner = std::numeric_limits<int>::max();

   index_rank.SetSize(12*leaf_elements.Size() * 3/2);
   index_rank.SetSize(0);

   NCMesh::BuildEdgeList();

   AddMasterSlaveRanks(nedges, edge_list);

   index_rank.Sort();
   index_rank.Unique();
   edge_group.MakeFromList(nedges, index_rank);
   index_rank.DeleteAll();

   MakeShared(edge_group, edge_list, shared_edges);
}

void ParNCMesh::BuildFaceList()
{
   // This is an extension of NCMesh::BuildFaceList() which also determines
   // face ownership, creates face processor groups and lists shared faces.

   int nfaces = NFaces + NGhostFaces;
   face_owner.SetSize(nfaces);
   face_owner = std::numeric_limits<int>::max();

   index_rank.SetSize(6*leaf_elements.Size() * 3/2);
   index_rank.SetSize(0);

   NCMesh::BuildFaceList();

   AddMasterSlaveRanks(nfaces, face_list);

   index_rank.Sort();
   index_rank.Unique();
   face_group.MakeFromList(nfaces, index_rank);
   index_rank.DeleteAll();

   MakeShared(face_group, face_list, shared_faces);

   CalcFaceOrientations();
}

struct MasterSlaveInfo
{
   int master; // master index if this is a slave
   int slaves_begin, slaves_end; // slave list if this is a master
   MasterSlaveInfo() : master(-1), slaves_begin(0), slaves_end(0) {}
};

void ParNCMesh::AddMasterSlaveRanks(int nitems, const NCList& list)
{
   // create an auxiliary structure for each edge/face
   std::vector<MasterSlaveInfo> info(nitems);

   for (unsigned i = 0; i < list.masters.size(); i++)
   {
      const Master &mf = list.masters[i];
      info[mf.index].slaves_begin = mf.slaves_begin;
      info[mf.index].slaves_end = mf.slaves_end;
   }
   for (unsigned i = 0; i < list.slaves.size(); i++)
   {
      const Slave& sf = list.slaves[i];
      info[sf.index].master = sf.master;
   }

   // We need the processor groups of master edges/faces to contain the ranks of
   // their slaves (so that master DOFs get sent to those who share the slaves).
   // Conversely, we need the groups of slave edges/faces to contain the ranks
   // of their masters. Both can be done by appending more items to the
   // 'index_rank' array, before it is sorted and converted to the group table.
   // (Note that a master/slave edge can be shared by more than one processor.)

   int size = index_rank.Size();
   for (int i = 0; i < size; i++)
   {
      int index = index_rank[i].from;
      int rank = index_rank[i].to;

      const MasterSlaveInfo &msi = info[index];
      if (msi.master >= 0)
      {
         // 'index' is a slave, add its rank to the master's group
         index_rank.Append(Connection(msi.master, rank));
      }
      else
      {
         for (int j = msi.slaves_begin; j < msi.slaves_end; j++)
         {
            // 'index' is a master, add its rank to the groups of the slaves
            index_rank.Append(Connection(list.slaves[j].index, rank));
         }
      }
   }
}

static bool is_shared(const Table& groups, int index, int MyRank)
{
   // A vertex/edge/face is shared if its group contains more than one processor
   // and at the same time one of them is ourselves.

   int size = groups.RowSize(index);
   if (size <= 1)
   {
      return false;
   }

   const int* group = groups.GetRow(index);
   for (int i = 0; i < size; i++)
   {
      if (group[i] == MyRank) { return true; }
   }

   return false;
}

void ParNCMesh::MakeShared(const Table &groups, const NCList &list,
                           NCList &shared)
{
   shared.Clear();

   for (unsigned i = 0; i < list.conforming.size(); i++)
   {
      if (is_shared(groups, list.conforming[i].index, MyRank))
      {
         shared.conforming.push_back(list.conforming[i]);
      }
   }
   for (unsigned i = 0; i < list.masters.size(); i++)
   {
      if (is_shared(groups, list.masters[i].index, MyRank))
      {
         shared.masters.push_back(list.masters[i]);
      }
   }
   for (unsigned i = 0; i < list.slaves.size(); i++)
   {
      if (is_shared(groups, list.slaves[i].index, MyRank))
      {
         shared.slaves.push_back(list.slaves[i]);
      }
   }
}

void ParNCMesh::BuildSharedVertices()
{
   int nvertices = NVertices + NGhostVertices;
   vertex_owner.SetSize(nvertices);
   vertex_owner = std::numeric_limits<int>::max();

   index_rank.SetSize(8*leaf_elements.Size());
   index_rank.SetSize(0);

   Array<MeshId> vertex_id(nvertices);

   // similarly to edges/faces, we loop over the vertices of all leaf elements
   // to determine which processors share each vertex
   for (int i = 0; i < leaf_elements.Size(); i++)
   {
      Element* elem = leaf_elements[i];
      for (int j = 0; j < GI[(int) elem->geom].nv; j++)
      {
         Node* node = elem->node[j];
         int index = node->vertex->index;

         int &owner = vertex_owner[index];
         owner = std::min(owner, elem->rank);

         index_rank.Append(Connection(index, elem->rank));

         MeshId &id = vertex_id[index];
         id.index = (node->edge ? -1 : index);
         id.element = elem;
         id.local = j;
      }
   }

   index_rank.Sort();
   index_rank.Unique();
   vertex_group.MakeFromList(nvertices, index_rank);
   index_rank.DeleteAll();

   // create a list of shared vertices, skip obviously slave vertices
   // (for simplicity, we don't guarantee to skip all slave vertices)
   shared_vertices.Clear();
   for (int i = 0; i < nvertices; i++)
   {
      if (is_shared(vertex_group, i, MyRank) && vertex_id[i].index >= 0)
      {
         shared_vertices.conforming.push_back(vertex_id[i]);
      }
   }
}

void ParNCMesh::CalcFaceOrientations()
{
   // Calculate orientation of shared conforming faces.
   // NOTE: face orientation is calculated relative to its lower rank element.
   // Thanks to the ghost layer this can be done locally, without communication.

   face_orient.SetSize(NFaces);
   face_orient = 0;

   for (HashTable<Face>::Iterator it(faces); it; ++it)
   {
      if (it->ref_count == 2 && it->index < NFaces)
      {
         Element* e[2] = { it->elem[0], it->elem[1] };
         if (e[0]->rank == e[1]->rank) { continue; }
         if (e[0]->rank > e[1]->rank) { std::swap(e[0], e[1]); }

         int ids[2][4];
         for (int i = 0; i < 2; i++)
         {
            int f = find_hex_face(find_node(e[i], it->p1),
                                  find_node(e[i], it->p2),
                                  find_node(e[i], it->p3));

            // get node IDs for the face as seen from e[i]
            const int* fv = GI[Geometry::CUBE].faces[f];
            for (int j = 0; j < 4; j++)
            {
               ids[i][j] = e[i]->node[fv[j]]->id;
            }
         }

         face_orient[it->index] = Mesh::GetQuadOrientation(ids[0], ids[1]);
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
      element_type[i] = (leaf_elements[i]->rank == MyRank) ? 1 : 0;
   }

   // determine the ghost layer
   Array<char> ghost_set;
   FindSetNeighbors(element_type, NULL, &ghost_set);

   // find the neighbors of the ghost layer
   Array<char> boundary_set;
   FindSetNeighbors(ghost_set, NULL, &boundary_set);

   ghost_layer.SetSize(0);
   for (int i = 0; i < nleaves; i++)
   {
      if (ghost_set[i])
      {
         element_type[i] = 2;
         ghost_layer.Append(leaf_elements[i]);
      }
      else if (boundary_set[i] && element_type[i])
      {
         element_type[i] = 3;
      }
   }
}

void ParNCMesh::ElementNeighborProcessors(Element *elem,
                                          Array<int> &ranks)
{
   MFEM_ASSERT(!elem->ref_type, "not a leaf.");

   ranks.SetSize(0); // preserve capacity

   // big shortcut: there are no neighbors if element_type == 1
   if (element_type[elem->index] == 1) { return; }

   // ok, we do need to look for neighbors;
   // at least we can only search in the ghost layer
   tmp_neighbors.SetSize(0);
   FindNeighbors(elem, tmp_neighbors, &ghost_layer);

   // return a list of processors
   for (int i = 0; i < tmp_neighbors.Size(); i++)
   {
      ranks.Append(tmp_neighbors[i]->rank);
   }
   ranks.Sort();
   ranks.Unique();
}

void ParNCMesh::NeighborProcessors(Array<int> &neighbors)
{
   UpdateLayers();

   std::set<int> ranks;
   for (int i = 0; i < ghost_layer.Size(); i++)
   {
      ranks.insert(ghost_layer[i]->rank);
   }

   neighbors.DeleteAll();
   neighbors.Reserve(ranks.size());

   std::set<int>::iterator it;
   for (it = ranks.begin(); it != ranks.end(); ++it)
   {
      neighbors.Append(*it);
   }
}


//// Prune, Refine /////////////////////////////////////////////////////////////

bool ParNCMesh::PruneTree(Element* elem)
{
   if (elem->ref_type)
   {
      bool remove[8];
      bool removeAll = true;

      // determine which subtrees can be removed (and whether it's all of them)
      for (int i = 0; i < 8; i++)
      {
         remove[i] = false;
         if (elem->child[i])
         {
            remove[i] = PruneTree(elem->child[i]);
            if (!remove[i]) { removeAll = false; }
         }
      }

      // all children can be removed, let the (maybe indirect) parent do it
      if (removeAll) { return true; }

      // not all children can be removed, but remove those that can be
      for (int i = 0; i < 8; i++)
      {
         if (remove[i]) { DerefineElement(elem->child[i]); }
      }

      return false; // need to keep this element and up
   }
   else
   {
      // return true if this leaf can be removed
      return element_type[elem->index] == 0;
   }
}

void ParNCMesh::Prune()
{
   if (!Iso)
   {
      MFEM_WARNING("Can't prune aniso meshes yet.");
      return;
   }

   UpdateLayers();

   // derefine subtrees whose leaves are all unneeded
   for (int i = 0; i < root_elements.Size(); i++)
   {
      if (PruneTree(root_elements[i]))
      {
         DerefineElement(root_elements[i]);
      }
   }

   Update();
}

void ParNCMesh::Refine(const Array<Refinement> &refinements)
{
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
      Element* elem = leaf_elements[ref.index];
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
      for (unsigned i = 0; i < msg.refinements.size(); i++)
      {
         const ElemRefType &ref = msg.refinements[i];
         NCMesh::RefineElement(ref.elem, ref.ref_type);
      }
   }

   Update();

   // make sure we can delete the send buffers
   NeighborDofMessage::WaitAllSent(send_ref);
}


void ParNCMesh::LimitNCLevel(int max_level)
{
   if (NRanks > 1)
   {
      MFEM_ABORT("not implemented in parallel yet.");
   }
   NCMesh::LimitNCLevel(max_level);
}


//// ElementSet ////////////////////////////////////////////////////////////////

void ParNCMesh::ElementSet::SetInt(int pos, int value)
{
   // helper to put an int to the data array
   data[pos] = value & 0xff;
   data[pos+1] = (value >> 8) & 0xff;
   data[pos+2] = (value >> 16) & 0xff;
   data[pos+3] = (value >> 24) & 0xff;
}

int ParNCMesh::ElementSet::GetInt(int pos) const
{
   // helper to get an int from the data array
   return (int) data[pos] +
          ((int) data[pos+1] << 8) +
          ((int) data[pos+2] << 16) +
          ((int) data[pos+3] << 24);
}

bool ParNCMesh::ElementSet::EncodeTree
(Element* elem, const std::set<Element*> &elements)
{
   // is 'elem' in the set?
   if (elements.find(elem) != elements.end())
   {
      // we reached a 'leaf' of our subtree, mark this as zero child mask
      data.Append(0);
      return true;
   }
   else if (elem->ref_type)
   {
      // write a bit mask telling what subtrees contain elements from the set
      int mpos = data.Size();
      data.Append(0);

      // check the subtrees
      int mask = 0;
      for (int i = 0; i < 8; i++)
      {
         if (elem->child[i])
         {
            if (EncodeTree(elem->child[i], elements))
            {
               mask |= (unsigned char) 1 << i;
            }
         }
      }

      if (mask)
      {
         data[mpos] = mask;
      }
      else
      {
         data.DeleteLast();
      }

      return mask != 0;
   }
   return false;
}

ParNCMesh::ElementSet::ElementSet(const std::set<Element*> &elements,
                                  const Array<Element*> &ncmesh_roots)
{
   int header_pos = 0;
   data.SetSize(4);

   // Each refinement tree that contains at least one element from the set
   // is encoded as HEADER + TREE, where HEADER is the root element number and
   // TREE is the output of EncodeTree().
   for (int i = 0; i < ncmesh_roots.Size(); i++)
   {
      if (EncodeTree(ncmesh_roots[i], elements))
      {
         SetInt(header_pos, i);

         // make room for the next header
         header_pos = data.Size();
         data.SetSize(header_pos + 4);
      }
   }

   // mark end of data
   SetInt(header_pos, -1);
}

void ParNCMesh::ElementSet::DecodeTree(Element* elem, int &pos,
                                       Array<Element*> &elements) const
{
   int mask = data[pos++];
   if (!mask)
   {
      elements.Append(elem);
   }
   else
   {
      for (int i = 0; i < 8; i++)
      {
         if (mask & (1 << i))
         {
            DecodeTree(elem->child[i], pos, elements);
         }
      }
   }
}

void ParNCMesh::ElementSet::Decode(Array<Element*> &elements,
                                   const Array<Element*> &ncmesh_roots) const
{
   int root, pos = 0;
   while ((root = GetInt(pos)) >= 0)
   {
      pos += 4;
      DecodeTree(ncmesh_roots[root], pos, elements);
   }
}

template<typename T>
static inline void write(std::ostream& os, T value)
{
   os.write((char*) &value, sizeof(T));
}

template<typename T>
static inline T read(std::istream& is)
{
   T value;
   is.read((char*) &value, sizeof(T));
   return value;
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

void ParNCMesh::EncodeMeshIds(std::ostream &os, Array<MeshId> ids[],
                              int dim) const
{
   std::map<Element*, int> element_id;

   // get a list of elements involved, dump them to 'os' and create the mapping
   // element_id: (Element* -> stream ID)
   {
      std::set<Element*> elements;
      for (int type = 0; type < dim; type++)
      {
         for (int i = 0; i < ids[type].Size(); i++)
         {
            elements.insert(ids[type][i].element);
         }
      }

      ElementSet eset(elements, root_elements); // encodes 'elements'
      eset.Dump(os);

      Array<Element*> decoded;
      eset.Decode(decoded, root_elements);

      for (int i = 0; i < decoded.Size(); i++)
      {
         element_id[decoded[i]] = i;
      }
   }

   // write the IDs as element/local pairs
   for (int type = 0; type < dim; type++)
   {
      write<int>(os, ids[type].Size());
      for (int i = 0; i < ids[type].Size(); i++)
      {
         const MeshId& id = ids[type][i];
         write<int>(os, element_id[id.element]); // TODO: variable 1-4 bytes
         write<char>(os, id.local);
      }
   }
}

void ParNCMesh::DecodeMeshIds(std::istream &is, Array<MeshId> ids[], int dim,
                              bool decode_indices) const
{
   // read the list of elements
   ElementSet eset(is);

   Array<Element*> elements;
   eset.Decode(elements, root_elements);

   // read vertex/edge/face IDs
   for (int type = 0; type < dim; type++)
   {
      int ne = read<int>(is);
      ids[type].SetSize(ne);

      for (int i = 0; i < ne; i++)
      {
         int el_num = read<int>(is);
         Element* elem = elements[el_num];
         MFEM_VERIFY(!elem->ref_type, "not a leaf element: " << el_num);

         MeshId &id = ids[type][i];
         id.element = elem;
         id.local = read<char>(is);

         if (!decode_indices) { continue; }

         // find vertex/edge/face index
         GeomInfo &gi = GI[(int) elem->geom];
         switch (type)
         {
            case 0:
            {
               id.index = elem->node[id.local]->vertex->index;
               break;
            }
            case 1:
            {
               const int* ev = gi.edges[id.local];
               Node* node = nodes.Peek(elem->node[ev[0]], elem->node[ev[1]]);
               MFEM_ASSERT(node && node->edge, "edge not found.");
               id.index = node->edge->index;
               break;
            }
            default:
            {
               const int* fv = gi.faces[id.local];
               Face* face = faces.Peek(elem->node[fv[0]], elem->node[fv[1]],
                                       elem->node[fv[2]], elem->node[fv[3]]);
               MFEM_ASSERT(face, "face not found.");
               id.index = face->index;
            }
         }
      }
   }
}

//// Messages //////////////////////////////////////////////////////////////////

void NeighborDofMessage::AddDofs(int type, const NCMesh::MeshId &id,
                                 const Array<int> &dofs)
{
   MFEM_ASSERT(type >= 0 && type < 3, "");
   id_dofs[type][id].assign(dofs.GetData(), dofs.GetData() + dofs.Size());
}

void NeighborDofMessage::GetDofs(int type, const NCMesh::MeshId& id,
                                 Array<int>& dofs, int &ndofs)
{
   MFEM_ASSERT(type >= 0 && type < 3, "");
#ifdef MFEM_DEBUG
   if (id_dofs[type].find(id) == id_dofs[type].end())
   {
      MFEM_ABORT("type/ID " << type << "/" << id.index << " not found in "
                 "neighbor message. Ghost layers out of sync?");
   }
#endif
   std::vector<int> &vec = id_dofs[type][id];
   dofs.SetSize(vec.size());
   dofs.Assign(vec.data());
   ndofs = this->ndofs;
}

void NeighborDofMessage::ReorderEdgeDofs(const NCMesh::MeshId &id,
                                         std::vector<int> &dofs)
{
   // Reorder the DOFs into/from a neutral ordering, independent of local
   // edge orientation. The processor neutral edge orientation is given by
   // the element local vertex numbering, not the mesh vertex numbering.

   const int *ev = NCMesh::GI[(int) id.element->geom].edges[id.local];
   int v0 = id.element->node[ev[0]]->vertex->index;
   int v1 = id.element->node[ev[1]]->vertex->index;

   if ((v0 < v1 && ev[0] > ev[1]) || (v0 > v1 && ev[0] < ev[1]))
   {
      std::vector<int> tmp(dofs);

      int nv = fec->DofForGeometry(Geometry::POINT);
      int ne = fec->DofForGeometry(Geometry::SEGMENT);
      MFEM_ASSERT((int) dofs.size() == 2*nv + ne, "");

      // swap the two vertex DOFs
      for (int i = 0; i < 2; i++)
      {
         for (int k = 0; k < nv; k++)
         {
            dofs[nv*i + k] = tmp[nv*(1-i) + k];
         }
      }

      // reorder the edge DOFs
      int* ind = fec->DofOrderForOrientation(Geometry::SEGMENT, 0);
      for (int i = 0; i < ne; i++)
      {
         dofs[2*nv + i] = (ind[i] >= 0) ? tmp[2*nv + ind[i]]
                          /*         */ : -1 - tmp[2*nv + (-1 - ind[i])];
      }
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

void NeighborDofMessage::Encode()
{
   IdToDofs::iterator it;

   // collect vertex/edge/face IDs
   Array<NCMesh::MeshId> ids[3];
   for (int type = 0; type < 3; type++)
   {
      ids[type].Reserve(id_dofs[type].size());
      for (it = id_dofs[type].begin(); it != id_dofs[type].end(); ++it)
      {
         ids[type].Append(it->first);
      }
   }

   // encode the IDs
   std::ostringstream stream;
   pncmesh->EncodeMeshIds(stream, ids, 3);

   // dump the DOFs
   for (int type = 0; type < 3; type++)
   {
      for (it = id_dofs[type].begin(); it != id_dofs[type].end(); ++it)
      {
         if (type == 1) { ReorderEdgeDofs(it->first, it->second); }
         write_dofs(stream, it->second);
      }

      // no longer need the original data
      id_dofs[type].clear();
   }

   write<int>(stream, ndofs);

   stream.str().swap(data);
}

void NeighborDofMessage::Decode()
{
   std::istringstream stream(data);

   // decode vertex/edge/face IDs
   Array<NCMesh::MeshId> ids[3];
   pncmesh->DecodeMeshIds(stream, ids, 3, true);

   // load DOFs
   for (int type = 0; type < 3; type++)
   {
      id_dofs[type].clear();
      for (int i = 0; i < ids[type].Size(); i++)
      {
         const NCMesh::MeshId &id = ids[type][i];
         read_dofs(stream, id_dofs[type][id]);
         if (type == 1) { ReorderEdgeDofs(id, id_dofs[type][id]); }
      }
   }

   ndofs = read<int>(stream);

   // no longer need the raw data
   data.clear();
}

void NeighborRowRequest::Encode()
{
   std::ostringstream stream;

   // write the int set to the stream
   write<int>(stream, rows.size());
   for (std::set<int>::iterator it = rows.begin(); it != rows.end(); ++it)
   {
      write<int>(stream, *it);
   }

   rows.clear();
   stream.str().swap(data);
}

void NeighborRowRequest::Decode()
{
   std::istringstream stream(data);

   // read the int set from the stream
   rows.clear();
   int size = read<int>(stream);
   for (int i = 0; i < size; i++)
   {
      rows.insert(rows.end(), read<int>(stream));
   }

   data.clear();
}

void NeighborRowReply::AddRow(int row, const Array<int> &cols,
                              const Vector &srow)
{
   MFEM_ASSERT(rows.find(row) == rows.end(), "");
   Row& row_data = rows[row];
   row_data.cols.assign(cols.GetData(), cols.GetData() + cols.Size());
   row_data.srow = srow;
}

void NeighborRowReply::GetRow(int row, Array<int> &cols, Vector &srow)
{
   MFEM_ASSERT(rows.find(row) != rows.end(),
               "row " << row << " not found in neighbor message.");
   Row& row_data = rows[row];
   cols.SetSize(row_data.cols.size());
   cols.Assign(row_data.cols.data());
   srow = row_data.srow;
}

void NeighborRowReply::Encode()
{
   std::ostringstream stream;

   // dump the rows to the stream
   write<int>(stream, rows.size());
   for (std::map<int, Row>::iterator it = rows.begin(); it != rows.end(); ++it)
   {
      write<int>(stream, it->first); // row number
      Row& row_data = it->second;
      MFEM_ASSERT((int) row_data.cols.size() == row_data.srow.Size(), "");
      write_dofs(stream, row_data.cols);
      stream.write((const char*) row_data.srow.GetData(),
                   sizeof(double) * row_data.srow.Size());
   }

   rows.clear();
   stream.str().swap(data);
}

void NeighborRowReply::Decode()
{
   std::istringstream stream(data); // stream makes a copy of data

   // NOTE: there is no rows.clear() since a row reply can be received
   // repeatedly and the received rows accumulate.

   // read the rows
   int size = read<int>(stream);
   for (int i = 0; i < size; i++)
   {
      Row& row_data = rows[read<int>(stream)];
      read_dofs(stream, row_data.cols);
      row_data.srow.SetSize(row_data.cols.size());
      stream.read((char*) row_data.srow.GetData(),
                  sizeof(double) * row_data.srow.Size());
   }

   data.clear();
}

void ParNCMesh::NeighborRefinementMessage::Encode()
{
   Array<MeshId> ids;

   // abuse EncodeMeshIds() to encode the list of refinements
   ids.Reserve(refinements.size());
   for (unsigned i = 0; i < refinements.size(); i++)
   {
      const ElemRefType &ref = refinements[i];
      ids.Append(MeshId(-1, ref.elem, ref.ref_type));
   }

   std::ostringstream stream;
   pncmesh->EncodeMeshIds(stream, &ids, 1);

   stream.str().swap(data);
}

void ParNCMesh::NeighborRefinementMessage::Decode()
{
   Array<NCMesh::MeshId> ids;

   // inverse abuse to Encode()
   std::istringstream stream(data);
   pncmesh->DecodeMeshIds(stream, &ids, 1, false);

   refinements.clear();
   refinements.reserve(ids.Size());
   for (int i = 0; i < ids.Size(); i++)
   {
      AddRefinement(ids[i].element, ids[i].local);
   }

   data.clear();
}


//// Utility ///////////////////////////////////////////////////////////////////

void ParNCMesh::GetDebugMesh(Mesh &debug_mesh) const
{
   // create a serial NCMesh containing all our elements (ghosts and all)
   NCMesh* copy = new NCMesh(*this);

   Array<Element*> &cle = copy->leaf_elements;
   for (int i = 0; i < cle.Size(); i++)
   {
      cle[i]->attribute = (cle[i]->rank == MyRank) ? 1 : 2;
   }

   debug_mesh.InitFromNCMesh(*copy);
   debug_mesh.SetAttributes();
   debug_mesh.ncmesh = copy;
}


} // namespace mfem

#endif // MFEM_USE_MPI
