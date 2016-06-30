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
#include "../fem/fespace.hpp"

#include <map>
#include <climits> // INT_MIN, INT_MAX

namespace mfem
{

ParNCMesh::ParNCMesh(MPI_Comm comm, const NCMesh &ncmesh)
   : NCMesh(ncmesh)
{
   MyComm = comm;
   MPI_Comm_size(MyComm, &NRanks);
   MPI_Comm_rank(MyComm, &MyRank);

   // assign leaf elements to the processors by simply splitting the
   // sequence of leaf elements into 'NRanks' parts
   for (int i = 0; i < leaf_elements.Size(); i++)
   {
      leaf_elements[i]->rank = InitialPartition(i);
   }

   AssignLeafIndices();
   UpdateVertices();

   // note that at this point all processors still have all the leaf elements;
   // we however may now start pruning the refinement tree to get rid of
   // branches that only contain someone else's leaves (see Prune())
}

ParNCMesh::~ParNCMesh()
{
   ClearAuxPM();
}

void ParNCMesh::Update()
{
   NCMesh::Update();

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
   // 'leaf_elements' and assign all ghost elements indices >= NElements. This
   // will make the ghosts skipped in NCMesh::GetMeshComponents.

   // Also note that the ordering of ghosts and non-ghosts is preserved here,
   // which is important for ParNCMesh::GetFaceNeighbors.

   Array<Element*> ghosts;
   ghosts.Reserve(leaf_elements.Size());

   NElements = 0;
   for (int i = 0; i < leaf_elements.Size(); i++)
   {
      Element* elem = leaf_elements[i];
      if (elem->rank == MyRank)
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

   NCMesh::AssignLeafIndices();
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
   for (HashTable<Face>::Iterator it(faces); it; ++it)
   {
      it->index = -1;
   }

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
   NFaces = mesh->GetNumFaces();
   NGhostFaces = 0;
   for (HashTable<Face>::Iterator it(faces); it; ++it)
   {
      if (it->index < 0) { it->index = NFaces + (NGhostFaces++); }
   }

   if (Dim == 2)
   {
      MFEM_ASSERT(NFaces == NEdges, "");
      MFEM_ASSERT(NGhostFaces == NGhostEdges, "");
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
   edge_owner = INT_MAX;

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
   face_owner = INT_MAX;

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
   vertex_owner = INT_MAX;

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

int ParNCMesh::get_face_orientation(Face* face, Element* e1, Element* e2,
                                    int local[2])
{
   // Return face orientation in e2, assuming the face has orientation 0 in e1.
   int ids[2][4];
   Element* e[2] = { e1, e2 };
   for (int i = 0; i < 2; i++)
   {
      int lf = find_hex_face(find_node(e[i], face->p1),
                             find_node(e[i], face->p2),
                             find_node(e[i], face->p3));
      if (local) { local[i] = lf; }

      // get node IDs for the face as seen from e[i]
      const int* fv = GI[Geometry::CUBE].faces[lf];
      for (int j = 0; j < 4; j++)
      {
         ids[i][j] = e[i]->node[fv[j]]->id;
      }
   }
   return Mesh::GetQuadOrientation(ids[0], ids[1]);
}

void ParNCMesh::CalcFaceOrientations()
{
   if (Dim < 3) { return; }

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

         face_orient[it->index] = get_face_orientation(it, e[0], e[1]);
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

bool ParNCMesh::CheckElementType(Element* elem, int type)
{
   if (!elem->ref_type)
   {
      return (element_type[elem->index] == type);
   }
   else
   {
      for (int i = 0; i < 8; i++)
      {
         if (elem->child[i] &&
             !CheckElementType(elem->child[i], type)) { return false; }
      }
      return true;
   }
}

void ParNCMesh::ElementNeighborProcessors(Element *elem,
                                          Array<int> &ranks)
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
      ranks.Append(tmp_neighbors[i]->rank);
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

   std::set<int> ranks;
   for (int i = 0; i < ghost_layer.Size(); i++)
   {
      ranks.insert(ghost_layer[i]->rank);
   }
   set_to_array(ranks, neighbors);
}

bool ParNCMesh::compare_ranks_indices(const Element* a, const Element* b)
{
   return (a->rank != b->rank) ? a->rank < b->rank
          /*                */ : a->index < b->index;
}

void ParNCMesh::GetFaceNeighbors(ParMesh &pmesh)
{
   ClearAuxPM();

   const NCList &shared = (Dim == 3) ? GetSharedFaces() : GetSharedEdges();
   const NCList &full_list = (Dim == 3) ? GetFaceList() : GetEdgeList();

   Array<Element*> fnbr;
   Array<Connection> send_elems;

   int bound = shared.conforming.size() + shared.slaves.size();

   fnbr.Reserve(bound);
   send_elems.Reserve(bound);

   // go over all shared faces and collect face neighbor elements
   for (unsigned i = 0; i < shared.conforming.size(); i++)
   {
      const MeshId &cf = shared.conforming[i];
      Face* face = GetFace(cf.element, cf.local);
      MFEM_ASSERT(face != NULL, "");

      Element* e[2] = { face->elem[0], face->elem[1] };
      MFEM_ASSERT(e[0] != NULL && e[1] != NULL, "");

      if (e[0]->rank == MyRank) { std::swap(e[0], e[1]); }
      MFEM_ASSERT(e[0]->rank != MyRank && e[1]->rank == MyRank, "");

      fnbr.Append(e[0]);
      send_elems.Append(Connection(e[0]->rank, e[1]->index));
   }

   for (unsigned i = 0; i < shared.masters.size(); i++)
   {
      const Master &mf = shared.masters[i];
      for (int j = mf.slaves_begin; j < mf.slaves_end; j++)
      {
         const Slave &sf = full_list.slaves[j];

         Element* e[2] = { mf.element, sf.element };
         MFEM_ASSERT(e[0] != NULL && e[1] != NULL, "");

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
   fnbr.Sort(compare_ranks_indices);

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

   std::map<Vertex*, int> vert_map;
   for (int i = 0; i < fnbr.Size(); i++)
   {
      Element* elem = fnbr[i];
      mfem::Element* fne = NewMeshElement(elem->geom);
      fne->SetAttribute(elem->attribute);
      pmesh.face_nbr_elements.Append(fne);

      GeomInfo& gi = GI[(int) elem->geom];
      for (int k = 0; k < gi.nv; k++)
      {
         int &v = vert_map[elem->node[k]->vertex];
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
   pmesh.face_nbr_vertices.SetSize(vert_map.size());
   std::map<Vertex*, int>::iterator it;
   for (it = vert_map.begin(); it != vert_map.end(); ++it)
   {
      pmesh.face_nbr_vertices[it->second-1].SetCoords(it->first->pos);
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
   for (unsigned i = 0; i < shared.conforming.size(); i++)
   {
      const MeshId &cf = shared.conforming[i];
      Face* face = GetFace(cf.element, cf.local);

      Element* e[2] = { face->elem[0], face->elem[1] };
      if (e[0]->rank == MyRank) { std::swap(e[0], e[1]); }

      Mesh::FaceInfo &fi = pmesh.faces_info[cf.index];
      fi.Elem2No = -1 - fnbr_index[e[0]->index - NElements];

      if (Dim == 3)
      {
         int local[2];
         int o = get_face_orientation(face, e[1], e[0], local);
         fi.Elem2Inf = 64*local[1] + o;
      }
      else
      {
         fi.Elem2Inf = 64*find_element_edge(e[0], face->p1, face->p3) + 1;
      }
   }

   if (shared.slaves.size())
   {
      int nfaces = NFaces, nghosts = NGhostFaces;
      if (Dim <= 2) { nfaces = NEdges, nghosts = NGhostEdges; }

      // enlarge Mesh::faces_info for ghost slaves
      pmesh.faces_info.SetSize(nfaces + nghosts);
      for (int i = nfaces; i < pmesh.faces_info.Size(); i++)
      {
         Mesh::FaceInfo &fi = pmesh.faces_info[i];
         fi.Elem1No  = fi.Elem2No  = -1;
         fi.Elem1Inf = fi.Elem2Inf = 0;
         fi.NCFace = -1;
      }

      // fill in FaceInfo for shared slave faces
      for (unsigned i = 0; i < shared.masters.size(); i++)
      {
         const Master &mf = shared.masters[i];
         for (int j = mf.slaves_begin; j < mf.slaves_end; j++)
         {
            const Slave &sf = full_list.slaves[j];

            MFEM_ASSERT(sf.element && mf.element, "");
            bool sloc = (sf.element->rank == MyRank);
            bool mloc = (mf.element->rank == MyRank);
            if (sloc == mloc) { continue; }

            Mesh::FaceInfo &fi = pmesh.faces_info[sf.index];
            fi.Elem1No = sf.element->index;
            fi.Elem2No = mf.element->index;
            fi.Elem1Inf = 64 * sf.local;
            fi.Elem2Inf = 64 * mf.local;

            if (!sloc)
            {
               std::swap(fi.Elem1No, fi.Elem2No);
               std::swap(fi.Elem1Inf, fi.Elem2Inf);
            }
            MFEM_ASSERT(fi.Elem2No >= NElements, "");
            fi.Elem2No = -1 - fnbr_index[fi.Elem2No - NElements];

            const DenseMatrix* pm = &sf.point_matrix;
            if (!sloc && Dim == 3)
            {
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
      return elem->rank < 0;
   }
}

void ParNCMesh::Prune()
{
   if (!Iso && Dim == 3)
   {
      MFEM_WARNING("Can't prune 3D aniso meshes yet.");
      return;
   }

   UpdateLayers();

   for (int i = 0; i < leaf_elements.Size(); i++)
   {
      // rank of elements beyond the ghost layer is unknown / not updated
      if (element_type[i] == 0)
      {
         leaf_elements[i]->rank = -1;
         // NOTE: rank == -1 will make the element disappear from leaf_elements
         // on next Update, see NCMesh::CollectLeafElements
      }
   }

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
      old_index_or_rank[i] = leaf_elements[i]->rank;
   }

   // back up the leaf_elements array
   Array<Element*> old_elements;
   leaf_elements.Copy(old_elements);

   // *** STEP 1: redistribute elements to avoid complex derefinements ***

   Array<int> new_ranks(leaf_elements.Size());
   for (int i = 0; i < leaf_elements.Size(); i++)
   {
      new_ranks[i] = leaf_elements[i]->rank;
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
         int fine_rank = leaf_elements[fine[j]]->rank;
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
      Element* parent = old_elements[fine[0]]->parent;

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
      leaf_elements[i]->index = -1;
   }
   for (int i = 0; i < old_elements.Size(); i++)
   {
      old_elements[i]->index = i;
   }

   // do local derefinements
   Array<Element*> coarse;
   old_elements.Copy(coarse);
   for (int i = 0; i < derefs.Size(); i++)
   {
      const int* fine = derefinements.GetRow(derefs[i]);
      Element* parent = old_elements[fine[0]]->parent;

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
         Element* elem = msg.elements[i];
         if (elem->ref_type)
         {
            SetDerefMatrixCodes(elem, coarse);
            NCMesh::DerefineElement(elem);
         }
         elem->rank = msg.values[i];
      }
   }

   // update leaf_elements, Element::index etc.
   Update();

   UpdateLayers();

   // link old fine elements to the new coarse elements
   for (int i = 0; i < coarse.Size(); i++)
   {
      int index = coarse[i]->index;
      if (element_type[index] == 0)
      {
         // this coarse element will get pruned, encode who owns it now
         index = -1 - coarse[i]->rank;
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
         index = old_elements[index]->index;
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
         ranks[j] = leaf_elements[fine[j]]->rank;
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
      const int* fine = deref_table.GetRow(i),
                 size = deref_table.RowSize(i);

      Element* parent = leaf_elements[fine[0]]->parent;
      for (int j = 0; j < size; j++)
      {
         Element* child = leaf_elements[fine[j]];
         if (child->rank == MyRank)
         {
            int splits[3];
            CountSplits(child, splits);

            for (int k = 0; k < Dim; k++)
            {
               if ((parent->ref_type & (1 << k)) &&
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

void ParNCMesh::Rebalance()
{
   send_rebalance_dofs.clear();
   recv_rebalance_dofs.clear();

   Array<Element*> old_elements;
   leaf_elements.GetSubArray(0, NElements, old_elements);

   // figure out new assignments for Element::rank
   long local_elems = NElements, total_elems = 0;
   MPI_Allreduce(&local_elems, &total_elems, 1, MPI_LONG, MPI_SUM, MyComm);

   long first_elem_global = 0;
   MPI_Scan(&local_elems, &first_elem_global, 1, MPI_LONG, MPI_SUM, MyComm);
   first_elem_global -= local_elems;

   Array<int> new_ranks(leaf_elements.Size());
   new_ranks = -1;

   for (int i = 0, j = 0; i < leaf_elements.Size(); i++)
   {
      if (leaf_elements[i]->rank == MyRank)
      {
         new_ranks[i] = Partition(first_elem_global + (j++), total_elems);
      }
   }

   int target_elements = PartitionFirstIndex(MyRank+1, total_elems)
                         - PartitionFirstIndex(MyRank, total_elems);

   // assign the new ranks and send elements (plus ghosts) to new owners
   RedistributeElements(new_ranks, target_elements, true);

   // set up the old index array
   old_index_or_rank.SetSize(NElements);
   old_index_or_rank = -1;
   for (int i = 0; i < old_elements.Size(); i++)
   {
      Element* e = old_elements[i];
      if (e->rank == MyRank) { old_index_or_rank[e->index] = i; }
   }

   // get rid of elements beyond the new ghost layer
   Prune();
}


bool ParNCMesh::compare_ranks(const Element* a, const Element* b)
{
   return a->rank < b->rank;
}

void ParNCMesh::RedistributeElements(Array<int> &new_ranks, int target_elements,
                                     bool record_comm)
{
   UpdateLayers();

   // *** STEP 1: communicate new rank assignments for the ghost layer ***

   NeighborElementRankMessage::Map send_ghost_ranks, recv_ghost_ranks;

   ghost_layer.Sort(compare_ranks);
   {
      Array<Element*> rank_neighbors;

      // loop over neighbor ranks and their elements
      int begin = 0, end = 0;
      while (end < ghost_layer.Size())
      {
         // find range of elements belonging to one rank
         int rank = ghost_layer[begin]->rank;
         while (end < ghost_layer.Size() &&
                ghost_layer[end]->rank == rank) { end++; }

         Array<Element*> rank_elems;
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
            Element* e = rank_neighbors[i];
            msg.AddElementRank(e, new_ranks[e->index]);
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
         int ghost_index = msg.elements[i]->index;
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
      Element* e = leaf_elements[i];
      if (e->rank == MyRank && new_ranks[i] == MyRank)
      {
         received_elements++; // initialize to number of elements we're keeping
      }
      e->rank = new_ranks[i];
   }

   RebalanceMessage::Map send_elems;
   {
      // sort elements we own by the new rank
      Array<Element*> owned_elements;
      owned_elements.MakeRef(leaf_elements.GetData(), NElements);
      owned_elements.Sort(compare_ranks);

      Array<Element*> batch;
      batch.Reserve(1024);

      // send elements to new owners
      int begin = 0, end = 0;
      while (end < NElements)
      {
         // find range of elements belonging to one rank
         int rank = owned_elements[begin]->rank;
         while (end < owned_elements.Size() &&
                owned_elements[end]->rank == rank) { end++; }

         if (rank != MyRank)
         {
            Array<Element*> rank_elems;
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
               Element* e = batch[i];
               if ((element_type[e->index] & 1) || e->rank != rank)
               {
                  msg.AddElementRank(e, e->rank);
               }
               // NOTE: we skip 'ghosts' that are of the receiver's rank because
               // they are not really ghosts and would get sent multiple times,
               // disrupting the termination mechanism in Step 4.
            }

            msg.Isend(rank, MyComm);

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

   /* We don't know from whom we're going to receive so we need to probe.
      Fortunately, we do know how many elements we're going to own eventually
      so the termination condition is easy. */

   RebalanceMessage msg;
   msg.SetNCMesh(this);

   while (received_elements < target_elements)
   {
      int rank, size;
      RebalanceMessage::Probe(rank, size, MyComm);

      // receive message; note: elements are created as the message is decoded
      msg.Recv(rank, size, MyComm);

      for (int i = 0; i < msg.Size(); i++)
      {
         int elem_rank = msg.values[i];
         msg.elements[i]->rank = elem_rank;

         if (elem_rank == MyRank) { received_elements++; }
      }

      // save the ranks we received from, for later use in RecvRebalanceDofs
      if (record_comm)
      {
         recv_rebalance_dofs[rank].SetNCMesh(this);
      }
   }

   Update();

   // make sure we can delete all send buffers
   NeighborElementRankMessage::WaitAllSent(send_ghost_ranks);
   NeighborElementRankMessage::WaitAllSent(send_elems);
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

void ParNCMesh::ElementSet::FlagElements(const Array<Element*> &elements,
                                         char flag)
{
   for (int i = 0; i < elements.Size(); i++)
   {
      Element* e = elements[i];
      while (e && e->flag != flag)
      {
         e->flag = flag;
         e = e->parent;
      }
   }
}

void ParNCMesh::ElementSet::EncodeTree(Element* elem)
{
   if (!elem->ref_type)
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
         if (elem->child[i] && elem->child[i]->flag)
         {
            mask |= 1 << i;
         }
      }

      // write the bit mask and visit the subtrees
      data.Append(mask);
      if (include_ref_types)
      {
         data.Append(elem->ref_type);
      }

      for (int i = 0; i < 8; i++)
      {
         if (mask & (1 << i))
         {
            EncodeTree(elem->child[i]);
         }
      }
   }
}

void ParNCMesh::ElementSet::Encode(const Array<Element*> &elements)
{
   FlagElements(elements, 1);

   // Each refinement tree that contains at least one element from the set
   // is encoded as HEADER + TREE, where HEADER is the root element number and
   // TREE is the output of EncodeTree().
   Array<Element*> &roots = ncmesh->root_elements;
   for (int i = 0; i < roots.Size(); i++)
   {
      if (roots[i]->flag)
      {
         WriteInt(i);
         EncodeTree(roots[i]);
      }
   }
   WriteInt(-1); // mark end of data

   FlagElements(elements, 0);
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
      if (include_ref_types)
      {
         int ref_type = data[pos++];
         if (!elem->ref_type)
         {
            ncmesh->RefineElement(elem, ref_type);
         }
         else { MFEM_ASSERT(ref_type == elem->ref_type, "") }
      }
      else { MFEM_ASSERT(elem->ref_type != 0, ""); }

      for (int i = 0; i < 8; i++)
      {
         if (mask & (1 << i))
         {
            DecodeTree(elem->child[i], pos, elements);
         }
      }
   }
}

void ParNCMesh::ElementSet::Decode(Array<Element*> &elements) const
{
   int root, pos = 0;
   while ((root = GetInt(pos)) >= 0)
   {
      pos += 4;
      DecodeTree(ncmesh->root_elements[root], pos, elements);
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

void ParNCMesh::EncodeMeshIds(std::ostream &os, Array<MeshId> ids[])
{
   std::map<Element*, int> element_id;

   // get a list of elements involved, dump them to 'os' and create the mapping
   // element_id: (Element* -> stream ID)
   {
      Array<Element*> elements;
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

      Array<Element*> decoded;
      eset.Decode(decoded);

      for (int i = 0; i < decoded.Size(); i++)
      {
         element_id[decoded[i]] = i;
      }
   }

   // write the IDs as element/local pairs
   for (int type = 0; type < 3; type++)
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

void ParNCMesh::DecodeMeshIds(std::istream &is, Array<MeshId> ids[])
{
   // read the list of elements
   ElementSet eset(this);
   eset.Load(is);

   Array<Element*> elements;
   eset.Decode(elements);

   // read vertex/edge/face IDs
   for (int type = 0; type < 3; type++)
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
   pncmesh->EncodeMeshIds(stream, ids);

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
   pncmesh->DecodeMeshIds(stream, ids);

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

template<class ValueType, bool RefTypes, int Tag>
void ParNCMesh::ElementValueMessage<ValueType, RefTypes, Tag>::Encode()
{
   std::ostringstream ostream;

   Array<Element*> tmp_elements;
   tmp_elements.MakeRef(elements.data(), elements.size());

   ElementSet eset(pncmesh, RefTypes);
   eset.Encode(tmp_elements);
   eset.Dump(ostream);

   // decode the element set to obtain a local numbering of elements
   Array<Element*> decoded;
   eset.Decode(decoded);

   std::map<Element*, int> element_index;
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
void ParNCMesh::ElementValueMessage<ValueType, RefTypes, Tag>::Decode()
{
   std::istringstream istream(data);

   ElementSet eset(pncmesh, RefTypes);
   eset.Load(istream);

   Array<Element*> tmp_elements;
   eset.Decode(tmp_elements);

   Element** el = tmp_elements.GetData();
   elements.assign(el, el + tmp_elements.Size());
   values.resize(elements.size());

   int count = read<int>(istream);
   for (int i = 0; i < count; i++)
   {
      int index = read<int>(istream);
      MFEM_ASSERT(index >= 0 && (size_t)index < values.size(), "");
      values[index] = read<ValueType>(istream);
   }

   // no longer need the raw data
   data.clear();
}

void ParNCMesh::RebalanceDofMessage::SetElements(const Array<Element*> &elems,
                                                 NCMesh *ncmesh)
{
   eset.SetNCMesh(ncmesh);
   eset.Encode(elems);

   Array<Element*> decoded;
   decoded.Reserve(elems.Size());
   eset.Decode(decoded);

   elem_ids.resize(decoded.Size());
   for (int i = 0; i < decoded.Size(); i++)
   {
      elem_ids[i] = decoded[i]->index;
   }
}

void ParNCMesh::RebalanceDofMessage::Encode()
{
   std::ostringstream stream;

   eset.Dump(stream);
   write<long>(stream, dof_offset);
   write_dofs(stream, dofs);

   stream.str().swap(data);
}

void ParNCMesh::RebalanceDofMessage::Decode()
{
   std::istringstream stream(data);

   eset.Load(stream);
   dof_offset = read<long>(stream);
   read_dofs(stream, dofs);

   data.clear();

   Array<Element*> elems;
   eset.Decode(elems);

   elem_ids.resize(elems.Size());
   for (int i = 0; i < elems.Size(); i++)
   {
      elem_ids[i] = elems[i]->index;
   }
}


//// Utility ///////////////////////////////////////////////////////////////////

void ParNCMesh::GetDebugMesh(Mesh &debug_mesh) const
{
   // create a serial NCMesh containing all our elements (ghosts and all)
   NCMesh* copy = new NCMesh(*this);

   Array<Element*> &cle = copy->leaf_elements;
   for (int i = 0; i < cle.Size(); i++)
   {
      cle[i]->attribute = cle[i]->rank + 1;
   }

   debug_mesh.InitFromNCMesh(*copy);
   debug_mesh.SetAttributes();
   debug_mesh.ncmesh = copy;
}


} // namespace mfem

#endif // MFEM_USE_MPI
