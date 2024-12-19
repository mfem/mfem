// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
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
#include "../fem/fem.hpp"
#include "../general/sets.hpp"
#include "../general/sort_pairs.hpp"
#include "../general/text.hpp"
#include "../general/globals.hpp"

#include <iostream>
#include <fstream>

using namespace std;

namespace mfem
{

ParMesh::ParMesh(const ParMesh &pmesh, bool copy_nodes)
   : Mesh(pmesh, false),
     group_svert(pmesh.group_svert),
     group_sedge(pmesh.group_sedge),
     group_stria(pmesh.group_stria),
     group_squad(pmesh.group_squad),
     glob_elem_offset(-1),
     glob_offset_sequence(-1),
     gtopo(pmesh.gtopo)
{
   MyComm = pmesh.MyComm;
   NRanks = pmesh.NRanks;
   MyRank = pmesh.MyRank;

   // Duplicate the shared_edges
   shared_edges.SetSize(pmesh.shared_edges.Size());
   for (int i = 0; i < shared_edges.Size(); i++)
   {
      shared_edges[i] = pmesh.shared_edges[i]->Duplicate(this);
   }

   shared_trias = pmesh.shared_trias;
   shared_quads = pmesh.shared_quads;

   // Copy the shared-to-local index Arrays
   pmesh.svert_lvert.Copy(svert_lvert);
   pmesh.sedge_ledge.Copy(sedge_ledge);
   sface_lface = pmesh.sface_lface;

   // Do not copy face-neighbor data (can be generated if needed)
   have_face_nbr_data = false;

   // If pmesh has a ParNURBSExtension, it was copied by the Mesh copy ctor, so
   // there is no need to do anything here.

   // Copy ParNCMesh, if present
   if (pmesh.pncmesh)
   {
      pncmesh = new ParNCMesh(*pmesh.pncmesh);
      pncmesh->OnMeshUpdated(this);
   }
   else
   {
      pncmesh = NULL;
   }
   ncmesh = pncmesh;

   // Copy the Nodes as a ParGridFunction, including the FiniteElementCollection
   // and the FiniteElementSpace (as a ParFiniteElementSpace)
   if (pmesh.Nodes && copy_nodes)
   {
      FiniteElementSpace *fes = pmesh.Nodes->FESpace();
      const FiniteElementCollection *fec = fes->FEColl();
      FiniteElementCollection *fec_copy =
         FiniteElementCollection::New(fec->Name());
      ParFiniteElementSpace *pfes_copy =
         new ParFiniteElementSpace(*fes, *this, fec_copy);
      Nodes = new ParGridFunction(pfes_copy);
      Nodes->MakeOwner(fec_copy);
      *Nodes = *pmesh.Nodes;
      own_nodes = 1;
   }
}

ParMesh::ParMesh(ParMesh &&mesh) : ParMesh()
{
   Swap(mesh);
}

ParMesh& ParMesh::operator=(ParMesh &&mesh)
{
   Swap(mesh);
   return *this;
}

ParMesh::ParMesh(MPI_Comm comm, Mesh &mesh, const int *partitioning_,
                 int part_method)
   : glob_elem_offset(-1)
   , glob_offset_sequence(-1)
   , gtopo(comm)
{
   MyComm = comm;
   MPI_Comm_size(MyComm, &NRanks);
   MPI_Comm_rank(MyComm, &MyRank);

   Array<int> partitioning;
   Array<bool> activeBdrElem;

   if (partitioning_)
   {
      partitioning.MakeRef(const_cast<int *>(partitioning_), mesh.GetNE(),
                           false);
   }

   if (mesh.Nonconforming())
   {
      ncmesh = pncmesh = new ParNCMesh(comm, *mesh.ncmesh, partitioning_);

      if (!partitioning_)
      {
         partitioning.SetSize(mesh.GetNE());
         for (int i = 0; i < mesh.GetNE(); i++)
         {
            partitioning[i] = pncmesh->InitialPartition(i);
         }
      }

      pncmesh->Prune();

      Mesh::InitFromNCMesh(*pncmesh);
      pncmesh->OnMeshUpdated(this);

      pncmesh->GetConformingSharedStructures(*this);

      // SetMeshGen(); // called by Mesh::InitFromNCMesh(...) above
      meshgen = mesh.meshgen; // copy the global 'meshgen'

      mesh.attributes.Copy(attributes);
      mesh.bdr_attributes.Copy(bdr_attributes);

      // Copy attribute and bdr_attribute names
      mesh.attribute_sets.Copy(attribute_sets);
      mesh.bdr_attribute_sets.Copy(bdr_attribute_sets);

      GenerateNCFaceInfo();
   }
   else // mesh.Conforming()
   {
      Dim = mesh.Dim;
      spaceDim = mesh.spaceDim;

      ncmesh = pncmesh = NULL;

      if (!partitioning_)
      {
         // Mesh::GeneratePartitioning always uses new[] to allocate the,
         // partitioning, so we need to tell the memory manager to free it with
         // delete[] (even if a different host memory type has been selected).
         constexpr MemoryType mt = MemoryType::HOST;
         partitioning.MakeRef(mesh.GeneratePartitioning(NRanks, part_method),
                              mesh.GetNE(), mt, true);
      }

      // re-enumerate the partitions to better map to actual processor
      // interconnect topology !?

      Array<int> vert_global_local;
      NumOfVertices = BuildLocalVertices(mesh, partitioning, vert_global_local);
      NumOfElements = BuildLocalElements(mesh, partitioning, vert_global_local);

      Table *edge_element = NULL;
      NumOfBdrElements = BuildLocalBoundary(mesh, partitioning,
                                            vert_global_local,
                                            activeBdrElem, edge_element);

      SetMeshGen();
      meshgen = mesh.meshgen; // copy the global 'meshgen'

      mesh.attributes.Copy(attributes);
      mesh.bdr_attributes.Copy(bdr_attributes);

      // Copy attribute and bdr_attribute names
      mesh.attribute_sets.Copy(attribute_sets);
      mesh.bdr_attribute_sets.Copy(bdr_attribute_sets);

      NumOfEdges = NumOfFaces = 0;

      if (Dim > 1)
      {
         el_to_edge = new Table;
         NumOfEdges = Mesh::GetElementToEdgeTable(*el_to_edge);
      }

      STable3D *faces_tbl = NULL;
      if (Dim == 3)
      {
         faces_tbl = GetElementToFaceTable(1);
      }

      GenerateFaces();

      // Make sure the be_to_face array is initialized.
      // In 2D, it will be set in the above call to Mesh::GetElementToEdgeTable.
      // In 3D, it will be set in GetElementToFaceTable.
      // In 1D, we need to set it manually.
      if (Dim == 1)
      {
         be_to_face.SetSize(NumOfBdrElements);
         for (int i = 0; i < NumOfBdrElements; ++i)
         {
            be_to_face[i] = boundary[i]->GetVertices()[0];
         }
      }

      ListOfIntegerSets  groups;
      {
         // the first group is the local one
         IntegerSet group;
         group.Recreate(1, &MyRank);
         groups.Insert(group);
      }

      MFEM_ASSERT(mesh.GetNFaces() == 0 || Dim >= 3, "");

      Array<int> face_group(mesh.GetNFaces());
      Table *vert_element = mesh.GetVertexToElementTable(); // we must delete this

      FindSharedFaces(mesh, partitioning, face_group, groups);
      int nsedges = FindSharedEdges(mesh, partitioning, edge_element, groups);
      int nsvert = FindSharedVertices(partitioning, vert_element, groups);

      // build the group communication topology
      gtopo.Create(groups, 822);

      // fill out group_sface, group_sedge, group_svert
      int ngroups = groups.Size()-1, nstris, nsquads;
      BuildFaceGroup(ngroups, mesh, face_group, nstris, nsquads);
      BuildEdgeGroup(ngroups, *edge_element);
      BuildVertexGroup(ngroups, *vert_element);

      // build shared_faces and sface_lface mapping
      BuildSharedFaceElems(nstris, nsquads, mesh, partitioning, faces_tbl,
                           face_group, vert_global_local);
      delete faces_tbl;

      // build shared_edges and sedge_ledge mapping
      BuildSharedEdgeElems(nsedges, mesh, vert_global_local, edge_element);
      delete edge_element;

      // build svert_lvert mapping
      BuildSharedVertMapping(nsvert, vert_element, vert_global_local);
      delete vert_element;
   }

   if (mesh.NURBSext)
   {
      MFEM_ASSERT(mesh.GetNodes() &&
                  mesh.GetNodes()->FESpace()->GetNURBSext() == mesh.NURBSext,
                  "invalid NURBS mesh");
      NURBSext = new ParNURBSExtension(comm, mesh.NURBSext, partitioning,
                                       activeBdrElem);
   }

   if (mesh.GetNodes()) // curved mesh
   {
      if (!NURBSext)
      {
         Nodes = new ParGridFunction(this, mesh.GetNodes());
      }
      else
      {
         const FiniteElementSpace *glob_fes = mesh.GetNodes()->FESpace();
         FiniteElementCollection *nfec =
            FiniteElementCollection::New(glob_fes->FEColl()->Name());
         ParFiniteElementSpace *pfes =
            new ParFiniteElementSpace(this, nfec, glob_fes->GetVDim(),
                                      glob_fes->GetOrdering());
         Nodes = new ParGridFunction(pfes);
         Nodes->MakeOwner(nfec); // Nodes will own nfec and pfes
      }
      own_nodes = 1;

      Array<int> gvdofs, lvdofs;
      Vector lnodes;
      int element_counter = 0;
      for (int i = 0; i < mesh.GetNE(); i++)
      {
         if (partitioning[i] == MyRank)
         {
            Nodes->FESpace()->GetElementVDofs(element_counter, lvdofs);
            mesh.GetNodes()->FESpace()->GetElementVDofs(i, gvdofs);
            mesh.GetNodes()->GetSubVector(gvdofs, lnodes);
            Nodes->SetSubVector(lvdofs, lnodes);
            element_counter++;
         }
      }

      // set meaningful values to 'vertices' even though we have Nodes,
      // for compatibility (e.g., Mesh::GetVertex())
      SetVerticesFromNodes(Nodes);
   }

   have_face_nbr_data = false;
}


int ParMesh::BuildLocalVertices(const mfem::Mesh &mesh,
                                const int* partitioning,
                                Array<int> &vert_global_local)
{
   vert_global_local.SetSize(mesh.GetNV());
   vert_global_local = -1;

   int vert_counter = 0;
   for (int i = 0; i < mesh.GetNE(); i++)
   {
      if (partitioning[i] == MyRank)
      {
         Array<int> vert;
         mesh.GetElementVertices(i, vert);
         for (int j = 0; j < vert.Size(); j++)
         {
            if (vert_global_local[vert[j]] < 0)
            {
               vert_global_local[vert[j]] = vert_counter++;
            }
         }
      }
   }

   // re-enumerate the local vertices to preserve the global ordering
   vert_counter = 0;
   for (int i = 0; i < vert_global_local.Size(); i++)
   {
      if (vert_global_local[i] >= 0)
      {
         vert_global_local[i] = vert_counter++;
      }
   }

   vertices.SetSize(vert_counter);

   for (int i = 0; i < vert_global_local.Size(); i++)
   {
      if (vert_global_local[i] >= 0)
      {
         vertices[vert_global_local[i]].SetCoords(mesh.SpaceDimension(),
                                                  mesh.GetVertex(i));
      }
   }

   return vert_counter;
}

int ParMesh::BuildLocalElements(const Mesh& mesh, const int* partitioning,
                                const Array<int>& vert_global_local)
{
   const int nelems = std::count_if(partitioning,
   partitioning + mesh.GetNE(), [this](int i) { return i == MyRank;});

   elements.SetSize(nelems);

   // Determine elements, enumerating the local elements to preserve the global
   // order. This is used, e.g. by the ParGridFunction ctor that takes a global
   // GridFunction as input parameter.
   int element_counter = 0;
   for (int i = 0; i < mesh.GetNE(); i++)
   {
      if (partitioning[i] == MyRank)
      {
         elements[element_counter] = mesh.GetElement(i)->Duplicate(this);
         int *v = elements[element_counter]->GetVertices();
         int nv = elements[element_counter]->GetNVertices();
         for (int j = 0; j < nv; j++)
         {
            v[j] = vert_global_local[v[j]];
         }
         ++element_counter;
      }
   }

   return element_counter;
}

int ParMesh::BuildLocalBoundary(const Mesh& mesh, const int* partitioning,
                                const Array<int>& vert_global_local,
                                Array<bool>& activeBdrElem,
                                Table*& edge_element)
{
   int nbdry = 0;
   if (mesh.NURBSext)
   {
      activeBdrElem.SetSize(mesh.GetNBE());
      activeBdrElem = false;
   }
   // build boundary elements
   if (Dim == 3)
   {
      for (int i = 0; i < mesh.GetNBE(); i++)
      {
         int face, o, el1, el2;
         mesh.GetBdrElementFace(i, &face, &o);
         mesh.GetFaceElements(face, &el1, &el2);
         if (partitioning[(o % 2 == 0 || el2 < 0) ? el1 : el2] == MyRank)
         {
            nbdry++;
            if (mesh.NURBSext)
            {
               activeBdrElem[i] = true;
            }
         }
      }

      int bdrelem_counter = 0;
      boundary.SetSize(nbdry);
      for (int i = 0; i < mesh.GetNBE(); i++)
      {
         int face, o, el1, el2;
         mesh.GetBdrElementFace(i, &face, &o);
         mesh.GetFaceElements(face, &el1, &el2);
         if (partitioning[(o % 2 == 0 || el2 < 0) ? el1 : el2] == MyRank)
         {
            boundary[bdrelem_counter] = mesh.GetBdrElement(i)->Duplicate(this);
            int *v = boundary[bdrelem_counter]->GetVertices();
            int nv = boundary[bdrelem_counter]->GetNVertices();
            for (int j = 0; j < nv; j++)
            {
               v[j] = vert_global_local[v[j]];
            }
            bdrelem_counter++;
         }
      }
   }
   else if (Dim == 2)
   {
      edge_element = new Table;
      Transpose(mesh.ElementToEdgeTable(), *edge_element, mesh.GetNEdges());

      for (int i = 0; i < mesh.GetNBE(); i++)
      {
         int edge = mesh.GetBdrElementFaceIndex(i);
         int el1 = edge_element->GetRow(edge)[0];
         if (partitioning[el1] == MyRank)
         {
            nbdry++;
            if (mesh.NURBSext)
            {
               activeBdrElem[i] = true;
            }
         }
      }

      int bdrelem_counter = 0;
      boundary.SetSize(nbdry);
      for (int i = 0; i < mesh.GetNBE(); i++)
      {
         int edge = mesh.GetBdrElementFaceIndex(i);
         int el1 = edge_element->GetRow(edge)[0];
         if (partitioning[el1] == MyRank)
         {
            boundary[bdrelem_counter] = mesh.GetBdrElement(i)->Duplicate(this);
            int *v = boundary[bdrelem_counter]->GetVertices();
            int nv = boundary[bdrelem_counter]->GetNVertices();
            for (int j = 0; j < nv; j++)
            {
               v[j] = vert_global_local[v[j]];
            }
            bdrelem_counter++;
         }
      }
   }
   else if (Dim == 1)
   {
      for (int i = 0; i < mesh.GetNBE(); i++)
      {
         int vert = mesh.boundary[i]->GetVertices()[0];
         int el1, el2;
         mesh.GetFaceElements(vert, &el1, &el2);
         if (partitioning[el1] == MyRank)
         {
            nbdry++;
         }
      }

      int bdrelem_counter = 0;
      boundary.SetSize(nbdry);
      for (int i = 0; i < mesh.GetNBE(); i++)
      {
         int vert = mesh.boundary[i]->GetVertices()[0];
         int el1, el2;
         mesh.GetFaceElements(vert, &el1, &el2);
         if (partitioning[el1] == MyRank)
         {
            boundary[bdrelem_counter] = mesh.GetBdrElement(i)->Duplicate(this);
            int *v = boundary[bdrelem_counter]->GetVertices();
            v[0] = vert_global_local[v[0]];
            bdrelem_counter++;
         }
      }
   }

   return nbdry;
}

void ParMesh::FindSharedFaces(const Mesh &mesh, const int *partitioning,
                              Array<int> &face_group,
                              ListOfIntegerSets &groups)
{
   IntegerSet group;

   // determine shared faces
   face_group.SetSize(mesh.GetNFaces());
   for (int i = 0; i < face_group.Size(); i++)
   {
      int el[2];
      face_group[i] = -1;
      mesh.GetFaceElements(i, &el[0], &el[1]);
      if (el[1] >= 0)
      {
         el[0] = partitioning[el[0]];
         el[1] = partitioning[el[1]];
         if ((el[0] == MyRank && el[1] != MyRank) ||
             (el[0] != MyRank && el[1] == MyRank))
         {
            group.Recreate(2, el);
            face_group[i] = groups.Insert(group) - 1;
         }
      }
   }
}

int ParMesh::FindSharedEdges(const Mesh &mesh, const int *partitioning,
                             Table*& edge_element,
                             ListOfIntegerSets& groups)
{
   IntegerSet group;

   // determine shared edges
   int sedge_counter = 0;
   if (!edge_element)
   {
      edge_element = new Table;
      if (Dim == 1)
      {
         edge_element->SetDims(0,0);
      }
      else
      {
         Transpose(mesh.ElementToEdgeTable(), *edge_element, mesh.GetNEdges());
      }
   }

   for (int i = 0; i < edge_element->Size(); i++)
   {
      int me = 0, others = 0;
      for (int j = edge_element->GetI()[i]; j < edge_element->GetI()[i+1]; j++)
      {
         int k = edge_element->GetJ()[j];
         int rank = partitioning[k];
         edge_element->GetJ()[j] = rank;
         if (rank == MyRank)
         {
            me = 1;
         }
         else
         {
            others = 1;
         }
      }

      if (me && others)
      {
         sedge_counter++;
         group.Recreate(edge_element->RowSize(i), edge_element->GetRow(i));
         edge_element->GetRow(i)[0] = groups.Insert(group) - 1;
      }
      else
      {
         edge_element->GetRow(i)[0] = -1;
      }
   }

   return sedge_counter;
}

int ParMesh::FindSharedVertices(const int *partitioning, Table *vert_element,
                                ListOfIntegerSets &groups)
{
   IntegerSet group;

   // determine shared vertices
   int svert_counter = 0;
   for (int i = 0; i < vert_element->Size(); i++)
   {
      int me = 0, others = 0;
      for (int j = vert_element->GetI()[i]; j < vert_element->GetI()[i+1]; j++)
      {
         vert_element->GetJ()[j] = partitioning[vert_element->GetJ()[j]];
         if (vert_element->GetJ()[j] == MyRank)
         {
            me = 1;
         }
         else
         {
            others = 1;
         }
      }

      if (me && others)
      {
         svert_counter++;
         group.Recreate(vert_element->RowSize(i), vert_element->GetRow(i));
         vert_element->GetI()[i] = groups.Insert(group) - 1;
      }
      else
      {
         vert_element->GetI()[i] = -1;
      }
   }
   return svert_counter;
}

void ParMesh::BuildFaceGroup(int ngroups, const Mesh &mesh,
                             const Array<int> &face_group,
                             int &nstria, int &nsquad)
{
   // build group_stria and group_squad
   group_stria.MakeI(ngroups);
   group_squad.MakeI(ngroups);

   for (int i = 0; i < face_group.Size(); i++)
   {
      if (face_group[i] >= 0)
      {
         if (mesh.GetFace(i)->GetType() == Element::TRIANGLE)
         {
            group_stria.AddAColumnInRow(face_group[i]);
         }
         else
         {
            group_squad.AddAColumnInRow(face_group[i]);
         }
      }
   }

   group_stria.MakeJ();
   group_squad.MakeJ();

   nstria = nsquad = 0;
   for (int i = 0; i < face_group.Size(); i++)
   {
      if (face_group[i] >= 0)
      {
         if (mesh.GetFace(i)->GetType() == Element::TRIANGLE)
         {
            group_stria.AddConnection(face_group[i], nstria++);
         }
         else
         {
            group_squad.AddConnection(face_group[i], nsquad++);
         }
      }
   }

   group_stria.ShiftUpI();
   group_squad.ShiftUpI();
}

void ParMesh::BuildEdgeGroup(int ngroups, const Table &edge_element)
{
   group_sedge.MakeI(ngroups);

   for (int i = 0; i < edge_element.Size(); i++)
   {
      if (edge_element.GetRow(i)[0] >= 0)
      {
         group_sedge.AddAColumnInRow(edge_element.GetRow(i)[0]);
      }
   }

   group_sedge.MakeJ();

   int sedge_counter = 0;
   for (int i = 0; i < edge_element.Size(); i++)
   {
      if (edge_element.GetRow(i)[0] >= 0)
      {
         group_sedge.AddConnection(edge_element.GetRow(i)[0], sedge_counter++);
      }
   }

   group_sedge.ShiftUpI();
}

void ParMesh::BuildVertexGroup(int ngroups, const Table &vert_element)
{
   group_svert.MakeI(ngroups);

   for (int i = 0; i < vert_element.Size(); i++)
   {
      if (vert_element.GetI()[i] >= 0)
      {
         group_svert.AddAColumnInRow(vert_element.GetI()[i]);
      }
   }

   group_svert.MakeJ();

   int svert_counter = 0;
   for (int i = 0; i < vert_element.Size(); i++)
   {
      if (vert_element.GetI()[i] >= 0)
      {
         group_svert.AddConnection(vert_element.GetI()[i], svert_counter++);
      }
   }

   group_svert.ShiftUpI();
}

void ParMesh::BuildSharedFaceElems(int ntri_faces, int nquad_faces,
                                   const Mesh& mesh, const int *partitioning,
                                   const STable3D *faces_tbl,
                                   const Array<int> &face_group,
                                   const Array<int> &vert_global_local)
{
   shared_trias.SetSize(ntri_faces);
   shared_quads.SetSize(nquad_faces);
   sface_lface. SetSize(ntri_faces + nquad_faces);

   if (Dim < 3) { return; }

   int stria_counter = 0;
   int squad_counter = 0;
   for (int i = 0; i < face_group.Size(); i++)
   {
      if (face_group[i] < 0) { continue; }

      const Element *face = mesh.GetFace(i);
      const int *fv = face->GetVertices();
      switch (face->GetType())
      {
         case Element::TRIANGLE:
         {
            shared_trias[stria_counter].Set(fv);
            int *v = shared_trias[stria_counter].v;
            for (int j = 0; j < 3; j++)
            {
               v[j] = vert_global_local[v[j]];
            }
            const int lface = (*faces_tbl)(v[0], v[1], v[2]);
            sface_lface[stria_counter] = lface;
            if (meshgen == 1) // Tet-only mesh
            {
               Tetrahedron *tet = dynamic_cast<Tetrahedron *>
                                  (elements[faces_info[lface].Elem1No]);
               // mark the shared face for refinement by reorienting
               // it according to the refinement flag in the tetrahedron
               // to which this shared face belongs to.
               if (tet->GetRefinementFlag())
               {
                  tet->GetMarkedFace(faces_info[lface].Elem1Inf/64, v);
                  // flip the shared face in the processor that owns the
                  // second element (in 'mesh')
                  int gl_el1, gl_el2;
                  mesh.GetFaceElements(i, &gl_el1, &gl_el2);
                  if (MyRank == partitioning[gl_el2])
                  {
                     std::swap(v[0], v[1]);
                  }
               }
            }
            stria_counter++;
            break;
         }

         case Element::QUADRILATERAL:
         {
            shared_quads[squad_counter].Set(fv);
            int *v = shared_quads[squad_counter].v;
            for (int j = 0; j < 4; j++)
            {
               v[j] = vert_global_local[v[j]];
            }
            sface_lface[shared_trias.Size() + squad_counter] =
               (*faces_tbl)(v[0], v[1], v[2], v[3]);
            squad_counter++;
            break;
         }

         default:
            MFEM_ABORT("unknown face type: " << face->GetType());
            break;
      }
   }
}

void ParMesh::BuildSharedEdgeElems(int nedges, Mesh& mesh,
                                   const Array<int>& vert_global_local,
                                   const Table* edge_element)
{
   // The passed in mesh is still the global mesh.  "this" mesh is the
   // local partitioned mesh.

   shared_edges.SetSize(nedges);
   sedge_ledge. SetSize(nedges);

   {
      DSTable v_to_v(NumOfVertices);
      GetVertexToVertexTable(v_to_v);

      int sedge_counter = 0;
      for (int i = 0; i < edge_element->Size(); i++)
      {
         if (edge_element->GetRow(i)[0] >= 0)
         {
            Array<int> vert;
            mesh.GetEdgeVertices(i, vert);

            shared_edges[sedge_counter] =
               new Segment(vert_global_local[vert[0]],
                           vert_global_local[vert[1]], 1);

            sedge_ledge[sedge_counter] = v_to_v(vert_global_local[vert[0]],
                                                vert_global_local[vert[1]]);

            MFEM_VERIFY(sedge_ledge[sedge_counter] >= 0, "Error in v_to_v.");

            sedge_counter++;
         }
      }
   }
}

void ParMesh::BuildSharedVertMapping(int nvert,
                                     const mfem::Table *vert_element,
                                     const Array<int> &vert_global_local)
{
   // build svert_lvert
   svert_lvert.SetSize(nvert);

   int svert_counter = 0;
   for (int i = 0; i < vert_element->Size(); i++)
   {
      if (vert_element->GetI()[i] >= 0)
      {
         svert_lvert[svert_counter++] = vert_global_local[i];
      }
   }
}


// protected method, used by Nonconforming(De)Refinement and Rebalance
ParMesh::ParMesh(const ParNCMesh &pncmesh)
   : MyComm(pncmesh.MyComm)
   , NRanks(pncmesh.NRanks)
   , MyRank(pncmesh.MyRank)
   , glob_elem_offset(-1)
   , glob_offset_sequence(-1)
   , gtopo(MyComm)
   , pncmesh(NULL)
{
   Mesh::InitFromNCMesh(pncmesh);
   ReduceMeshGen();
   have_face_nbr_data = false;
}

void ParMesh::ComputeGlobalElementOffset() const
{
   if (glob_offset_sequence != sequence) // mesh has changed
   {
      long long local_elems = NumOfElements;
      MPI_Scan(&local_elems, &glob_elem_offset, 1, MPI_LONG_LONG, MPI_SUM,
               MyComm);
      glob_elem_offset -= local_elems;

      glob_offset_sequence = sequence; // don't recalculate until refinement etc.
   }
}

void ParMesh::ReduceMeshGen()
{
   int loc_meshgen = meshgen;
   MPI_Allreduce(&loc_meshgen, &meshgen, 1, MPI_INT, MPI_BOR, MyComm);
}

void ParMesh::FinalizeParTopo()
{
   // Determine sedge_ledge
   sedge_ledge.SetSize(shared_edges.Size());
   if (shared_edges.Size())
   {
      DSTable v_to_v(NumOfVertices);
      GetVertexToVertexTable(v_to_v);
      for (int se = 0; se < shared_edges.Size(); se++)
      {
         const int *v = shared_edges[se]->GetVertices();
         const int l_edge = v_to_v(v[0], v[1]);
         MFEM_ASSERT(l_edge >= 0, "invalid shared edge");
         sedge_ledge[se] = l_edge;
      }
   }

   // Determine sface_lface
   const int nst = shared_trias.Size();
   sface_lface.SetSize(nst + shared_quads.Size());
   if (sface_lface.Size())
   {
      auto faces_tbl = std::unique_ptr<STable3D>(GetFacesTable());
      for (int st = 0; st < nst; st++)
      {
         const int *v = shared_trias[st].v;
         sface_lface[st] = (*faces_tbl)(v[0], v[1], v[2]);
      }
      for (int sq = 0; sq < shared_quads.Size(); sq++)
      {
         const int *v = shared_quads[sq].v;
         sface_lface[nst+sq] = (*faces_tbl)(v[0], v[1], v[2], v[3]);
      }
   }
}

ParMesh::ParMesh(MPI_Comm comm, istream &input, bool refine, int generate_edges,
                 bool fix_orientation)
   : glob_elem_offset(-1)
   , glob_offset_sequence(-1)
   , gtopo(comm)
{
   MyComm = comm;
   MPI_Comm_size(MyComm, &NRanks);
   MPI_Comm_rank(MyComm, &MyRank);

   have_face_nbr_data = false;
   pncmesh = NULL;

   Load(input, generate_edges, refine, fix_orientation);
}

void ParMesh::Load(istream &input, int generate_edges, int refine,
                   bool fix_orientation)
{
   ParMesh::Destroy();

   // Tell Loader() to read up to 'mfem_serial_mesh_end' instead of
   // 'mfem_mesh_end', as we have additional parallel mesh data to load in from
   // the stream.
   Loader(input, generate_edges, "mfem_serial_mesh_end");

   ReduceMeshGen(); // determine the global 'meshgen'

   if (Conforming())
   {
      LoadSharedEntities(input);
   }
   else
   {
      // the ParNCMesh instance was already constructed in 'Loader'
      pncmesh = dynamic_cast<ParNCMesh*>(ncmesh);
      MFEM_ASSERT(pncmesh, "internal error");

      // in the NC case we don't need to load extra data from the file,
      // as the shared entities can be constructed from the ghost layer
      pncmesh->GetConformingSharedStructures(*this);
   }

   Finalize(refine, fix_orientation);

   EnsureParNodes();

   // note: attributes and bdr_attributes are local lists

   // TODO: NURBS meshes?
}

void ParMesh::LoadSharedEntities(istream &input)
{
   string ident;
   skip_comment_lines(input, '#');

   // read the group topology
   input >> ident;
   MFEM_VERIFY(ident == "communication_groups",
               "input stream is not a parallel MFEM mesh");
   gtopo.Load(input);

   skip_comment_lines(input, '#');

   // read and set the sizes of svert_lvert, group_svert
   {
      int num_sverts;
      input >> ident >> num_sverts;
      MFEM_VERIFY(ident == "total_shared_vertices", "invalid mesh file");
      svert_lvert.SetSize(num_sverts);
      group_svert.SetDims(GetNGroups()-1, num_sverts);
   }
   // read and set the sizes of sedge_ledge, group_sedge
   if (Dim >= 2)
   {
      skip_comment_lines(input, '#');
      int num_sedges;
      input >> ident >> num_sedges;
      MFEM_VERIFY(ident == "total_shared_edges", "invalid mesh file");
      sedge_ledge.SetSize(num_sedges);
      shared_edges.SetSize(num_sedges);
      group_sedge.SetDims(GetNGroups()-1, num_sedges);
   }
   else
   {
      group_sedge.SetSize(GetNGroups()-1, 0);   // create empty group_sedge
   }
   // read and set the sizes of sface_lface, group_{stria,squad}
   if (Dim >= 3)
   {
      skip_comment_lines(input, '#');
      int num_sface;
      input >> ident >> num_sface;
      MFEM_VERIFY(ident == "total_shared_faces", "invalid mesh file");
      sface_lface.SetSize(num_sface);
      group_stria.MakeI(GetNGroups()-1);
      group_squad.MakeI(GetNGroups()-1);
   }
   else
   {
      group_stria.SetSize(GetNGroups()-1, 0);   // create empty group_stria
      group_squad.SetSize(GetNGroups()-1, 0);   // create empty group_squad
   }

   // read, group by group, the contents of group_svert, svert_lvert,
   // group_sedge, shared_edges, group_{stria,squad}, shared_{trias,quads}
   int svert_counter = 0, sedge_counter = 0;
   for (int gr = 1; gr < GetNGroups(); gr++)
   {
      skip_comment_lines(input, '#');
#if 0
      // implementation prior to prism-dev merge
      int g;
      input >> ident >> g; // group
      if (g != gr)
      {
         mfem::err << "ParMesh::ParMesh : expecting group " << gr
                   << ", read group " << g << endl;
         mfem_error();
      }
#endif
      {
         int nv;
         input >> ident >> nv; // shared_vertices (in this group)
         MFEM_VERIFY(ident == "shared_vertices", "invalid mesh file");
         nv += svert_counter;
         MFEM_VERIFY(nv <= group_svert.Size_of_connections(),
                     "incorrect number of total_shared_vertices");
         group_svert.GetI()[gr] = nv;
         for ( ; svert_counter < nv; svert_counter++)
         {
            group_svert.GetJ()[svert_counter] = svert_counter;
            input >> svert_lvert[svert_counter];
         }
      }
      if (Dim >= 2)
      {
         int ne, v[2];
         input >> ident >> ne; // shared_edges (in this group)
         MFEM_VERIFY(ident == "shared_edges", "invalid mesh file");
         ne += sedge_counter;
         MFEM_VERIFY(ne <= group_sedge.Size_of_connections(),
                     "incorrect number of total_shared_edges");
         group_sedge.GetI()[gr] = ne;
         for ( ; sedge_counter < ne; sedge_counter++)
         {
            group_sedge.GetJ()[sedge_counter] = sedge_counter;
            input >> v[0] >> v[1];
            shared_edges[sedge_counter] = new Segment(v[0], v[1], 1);
         }
      }
      if (Dim >= 3)
      {
         int nf, tstart = shared_trias.Size(), qstart = shared_quads.Size();
         input >> ident >> nf; // shared_faces (in this group)
         for (int i = 0; i < nf; i++)
         {
            int geom, *v;
            input >> geom;
            switch (geom)
            {
               case Geometry::TRIANGLE:
                  shared_trias.SetSize(shared_trias.Size()+1);
                  v = shared_trias.Last().v;
                  for (int ii = 0; ii < 3; ii++) { input >> v[ii]; }
                  break;
               case Geometry::SQUARE:
                  shared_quads.SetSize(shared_quads.Size()+1);
                  v = shared_quads.Last().v;
                  for (int ii = 0; ii < 4; ii++) { input >> v[ii]; }
                  break;
               default:
                  MFEM_ABORT("invalid shared face geometry: " << geom);
            }
         }
         group_stria.AddColumnsInRow(gr-1, shared_trias.Size()-tstart);
         group_squad.AddColumnsInRow(gr-1, shared_quads.Size()-qstart);
      }
   }
   if (Dim >= 3)
   {
      MFEM_VERIFY(shared_trias.Size() + shared_quads.Size()
                  == sface_lface.Size(),
                  "incorrect number of total_shared_faces");
      // Define the J arrays of group_stria and group_squad -- they just contain
      // consecutive numbers starting from 0 up to shared_trias.Size()-1 and
      // shared_quads.Size()-1, respectively.
      group_stria.MakeJ();
      for (int i = 0; i < shared_trias.Size(); i++)
      {
         group_stria.GetJ()[i] = i;
      }
      group_squad.MakeJ();
      for (int i = 0; i < shared_quads.Size(); i++)
      {
         group_squad.GetJ()[i] = i;
      }
   }
}

ParMesh::ParMesh(ParMesh *orig_mesh, int ref_factor, int ref_type)
{
   MakeRefined_(*orig_mesh, ref_factor, ref_type);
}

void ParMesh::MakeRefined_(ParMesh &orig_mesh, int ref_factor, int ref_type)
{
   MyComm = orig_mesh.GetComm();
   NRanks = orig_mesh.GetNRanks();
   MyRank = orig_mesh.GetMyRank();
   face_nbr_el_to_face = nullptr;
   glob_elem_offset = -1;
   glob_offset_sequence = -1;
   gtopo = orig_mesh.gtopo;
   have_face_nbr_data = false;
   pncmesh = NULL;

   Array<int> ref_factors(orig_mesh.GetNE());
   ref_factors = ref_factor;
   Mesh::MakeRefined_(orig_mesh, ref_factors, ref_type);

   // Need to initialize:
   // - shared_edges, shared_{trias,quads}
   // - group_svert, group_sedge, group_{stria,squad}
   // - svert_lvert, sedge_ledge, sface_lface

   meshgen = orig_mesh.meshgen; // copy the global 'meshgen'

   H1_FECollection rfec(ref_factor, Dim, ref_type);
   ParFiniteElementSpace rfes(&orig_mesh, &rfec);

   // count the number of entries in each row of group_s{vert,edge,face}
   group_svert.MakeI(GetNGroups()-1); // exclude the local group 0
   group_sedge.MakeI(GetNGroups()-1);
   group_stria.MakeI(GetNGroups()-1);
   group_squad.MakeI(GetNGroups()-1);
   for (int gr = 1; gr < GetNGroups(); gr++)
   {
      // orig vertex -> vertex
      group_svert.AddColumnsInRow(gr-1, orig_mesh.GroupNVertices(gr));
      // orig edge -> (ref_factor-1) vertices and (ref_factor) edges
      const int orig_ne = orig_mesh.GroupNEdges(gr);
      group_svert.AddColumnsInRow(gr-1, (ref_factor-1)*orig_ne);
      group_sedge.AddColumnsInRow(gr-1, ref_factor*orig_ne);
      // orig face -> (?) vertices, (?) edges, and (?) faces
      const int orig_nt = orig_mesh.GroupNTriangles(gr);
      if (orig_nt > 0)
      {
         const Geometry::Type geom = Geometry::TRIANGLE;
         const int nvert = Geometry::NumVerts[geom];
         RefinedGeometry &RG =
            *GlobGeometryRefiner.Refine(geom, ref_factor, ref_factor);

         // count internal vertices
         group_svert.AddColumnsInRow(gr-1, orig_nt*rfec.DofForGeometry(geom));
         // count internal edges
         group_sedge.AddColumnsInRow(gr-1, orig_nt*(RG.RefEdges.Size()/2-
                                                    RG.NumBdrEdges));
         // count refined faces
         group_stria.AddColumnsInRow(gr-1, orig_nt*(RG.RefGeoms.Size()/nvert));
      }
      const int orig_nq = orig_mesh.GroupNQuadrilaterals(gr);
      if (orig_nq > 0)
      {
         const Geometry::Type geom = Geometry::SQUARE;
         const int nvert = Geometry::NumVerts[geom];
         RefinedGeometry &RG =
            *GlobGeometryRefiner.Refine(geom, ref_factor, ref_factor);

         // count internal vertices
         group_svert.AddColumnsInRow(gr-1, orig_nq*rfec.DofForGeometry(geom));
         // count internal edges
         group_sedge.AddColumnsInRow(gr-1, orig_nq*(RG.RefEdges.Size()/2-
                                                    RG.NumBdrEdges));
         // count refined faces
         group_squad.AddColumnsInRow(gr-1, orig_nq*(RG.RefGeoms.Size()/nvert));
      }
   }

   group_svert.MakeJ();
   svert_lvert.Reserve(group_svert.Size_of_connections());

   group_sedge.MakeJ();
   shared_edges.Reserve(group_sedge.Size_of_connections());
   sedge_ledge.SetSize(group_sedge.Size_of_connections());

   group_stria.MakeJ();
   group_squad.MakeJ();
   shared_trias.Reserve(group_stria.Size_of_connections());
   shared_quads.Reserve(group_squad.Size_of_connections());
   sface_lface.SetSize(shared_trias.Size() + shared_quads.Size());

   Array<int> rdofs;
   for (int gr = 1; gr < GetNGroups(); gr++)
   {
      // add shared vertices from original shared vertices
      const int orig_n_verts = orig_mesh.GroupNVertices(gr);
      for (int j = 0; j < orig_n_verts; j++)
      {
         rfes.GetVertexDofs(orig_mesh.GroupVertex(gr, j), rdofs);
         group_svert.AddConnection(gr-1, svert_lvert.Append(rdofs[0])-1);
      }

      // add refined shared edges; add shared vertices from refined shared edges
      const int orig_n_edges = orig_mesh.GroupNEdges(gr);
      if (orig_n_edges > 0)
      {
         const Geometry::Type geom = Geometry::SEGMENT;
         const int nvert = Geometry::NumVerts[geom];
         RefinedGeometry &RG = *GlobGeometryRefiner.Refine(geom, ref_factor);
         const int *c2h_map = rfec.GetDofMap(geom, ref_factor); // FIXME hp

         for (int e = 0; e < orig_n_edges; e++)
         {
            rfes.GetSharedEdgeDofs(gr, e, rdofs);
            MFEM_ASSERT(rdofs.Size() == RG.RefPts.Size(), "");
            // add the internal edge 'rdofs' as shared vertices
            for (int j = 2; j < rdofs.Size(); j++)
            {
               group_svert.AddConnection(gr-1, svert_lvert.Append(rdofs[j])-1);
            }
            for (int j = 0; j < RG.RefGeoms.Size(); j += nvert)
            {
               Element *elem = NewElement(geom);
               int *v = elem->GetVertices();
               for (int k = 0; k < nvert; k++)
               {
                  int cid = RG.RefGeoms[j+k]; // local Cartesian index
                  v[k] = rdofs[c2h_map[cid]];
               }
               group_sedge.AddConnection(gr-1, shared_edges.Append(elem)-1);
            }
         }
      }
      // add refined shared faces; add shared edges and shared vertices from
      // refined shared faces
      const int orig_nt = orig_mesh.GroupNTriangles(gr);
      if (orig_nt > 0)
      {
         const Geometry::Type geom = Geometry::TRIANGLE;
         const int nvert = Geometry::NumVerts[geom];
         RefinedGeometry &RG =
            *GlobGeometryRefiner.Refine(geom, ref_factor, ref_factor);
         const int num_int_verts = rfec.DofForGeometry(geom);
         const int *c2h_map = rfec.GetDofMap(geom, ref_factor); // FIXME hp

         for (int f = 0; f < orig_nt; f++)
         {
            rfes.GetSharedTriangleDofs(gr, f, rdofs);
            MFEM_ASSERT(rdofs.Size() == RG.RefPts.Size(), "");
            // add the internal face 'rdofs' as shared vertices
            for (int j = rdofs.Size()-num_int_verts; j < rdofs.Size(); j++)
            {
               group_svert.AddConnection(gr-1, svert_lvert.Append(rdofs[j])-1);
            }
            // add the internal (for the shared face) edges as shared edges
            for (int j = 2*RG.NumBdrEdges; j < RG.RefEdges.Size(); j += 2)
            {
               Element *elem = NewElement(Geometry::SEGMENT);
               int *v = elem->GetVertices();
               for (int k = 0; k < 2; k++)
               {
                  v[k] = rdofs[c2h_map[RG.RefEdges[j+k]]];
               }
               group_sedge.AddConnection(gr-1, shared_edges.Append(elem)-1);
            }
            // add refined shared faces
            for (int j = 0; j < RG.RefGeoms.Size(); j += nvert)
            {
               shared_trias.SetSize(shared_trias.Size()+1);
               int *v = shared_trias.Last().v;
               for (int k = 0; k < nvert; k++)
               {
                  int cid = RG.RefGeoms[j+k]; // local Cartesian index
                  v[k] = rdofs[c2h_map[cid]];
               }
               group_stria.AddConnection(gr-1, shared_trias.Size()-1);
            }
         }
      }
      const int orig_nq = orig_mesh.GroupNQuadrilaterals(gr);
      if (orig_nq > 0)
      {
         const Geometry::Type geom = Geometry::SQUARE;
         const int nvert = Geometry::NumVerts[geom];
         RefinedGeometry &RG =
            *GlobGeometryRefiner.Refine(geom, ref_factor, ref_factor);
         const int num_int_verts = rfec.DofForGeometry(geom);
         const int *c2h_map = rfec.GetDofMap(geom, ref_factor); // FIXME hp

         for (int f = 0; f < orig_nq; f++)
         {
            rfes.GetSharedQuadrilateralDofs(gr, f, rdofs);
            MFEM_ASSERT(rdofs.Size() == RG.RefPts.Size(), "");
            // add the internal face 'rdofs' as shared vertices
            for (int j = rdofs.Size()-num_int_verts; j < rdofs.Size(); j++)
            {
               group_svert.AddConnection(gr-1, svert_lvert.Append(rdofs[j])-1);
            }
            // add the internal (for the shared face) edges as shared edges
            for (int j = 2*RG.NumBdrEdges; j < RG.RefEdges.Size(); j += 2)
            {
               Element *elem = NewElement(Geometry::SEGMENT);
               int *v = elem->GetVertices();
               for (int k = 0; k < 2; k++)
               {
                  v[k] = rdofs[c2h_map[RG.RefEdges[j+k]]];
               }
               group_sedge.AddConnection(gr-1, shared_edges.Append(elem)-1);
            }
            // add refined shared faces
            for (int j = 0; j < RG.RefGeoms.Size(); j += nvert)
            {
               shared_quads.SetSize(shared_quads.Size()+1);
               int *v = shared_quads.Last().v;
               for (int k = 0; k < nvert; k++)
               {
                  int cid = RG.RefGeoms[j+k]; // local Cartesian index
                  v[k] = rdofs[c2h_map[cid]];
               }
               group_squad.AddConnection(gr-1, shared_quads.Size()-1);
            }
         }
      }
   }
   group_svert.ShiftUpI();
   group_sedge.ShiftUpI();
   group_stria.ShiftUpI();
   group_squad.ShiftUpI();

   FinalizeParTopo();

   if (Nodes != NULL)
   {
      // This call will turn the Nodes into a ParGridFunction
      SetCurvature(1, GetNodalFESpace()->IsDGSpace(), spaceDim,
                   GetNodalFESpace()->GetOrdering());
   }
}

ParMesh ParMesh::MakeRefined(ParMesh &orig_mesh, int ref_factor, int ref_type)
{
   ParMesh mesh;
   mesh.MakeRefined_(orig_mesh, ref_factor, ref_type);
   return mesh;
}

ParMesh ParMesh::MakeSimplicial(ParMesh &orig_mesh)
{
   ParMesh mesh;

   mesh.MyComm = orig_mesh.GetComm();
   mesh.NRanks = orig_mesh.GetNRanks();
   mesh.MyRank = orig_mesh.GetMyRank();
   mesh.glob_elem_offset = -1;
   mesh.glob_offset_sequence = -1;
   mesh.gtopo = orig_mesh.gtopo;
   mesh.have_face_nbr_data = false;
   mesh.pncmesh = NULL;
   mesh.meshgen = orig_mesh.meshgen;

   H1_FECollection fec(1, orig_mesh.Dimension());
   ParFiniteElementSpace fes(&orig_mesh, &fec);

   Array<int> vglobal(orig_mesh.GetNV());
   for (int iv=0; iv<orig_mesh.GetNV(); ++iv)
   {
      vglobal[iv] = fes.GetGlobalTDofNumber(iv);
   }
   mesh.MakeSimplicial_(orig_mesh, vglobal);

   // count the number of entries in each row of group_s{vert,edge,face}
   mesh.group_svert.MakeI(mesh.GetNGroups()-1); // exclude the local group 0
   mesh.group_sedge.MakeI(mesh.GetNGroups()-1);
   mesh.group_stria.MakeI(mesh.GetNGroups()-1);
   mesh.group_squad.MakeI(mesh.GetNGroups()-1);
   for (int gr = 1; gr < mesh.GetNGroups(); gr++)
   {
      mesh.group_svert.AddColumnsInRow(gr-1, orig_mesh.GroupNVertices(gr));
      mesh.group_sedge.AddColumnsInRow(gr-1, orig_mesh.GroupNEdges(gr));
      // Every quad gives an extra edge
      const int orig_nq = orig_mesh.GroupNQuadrilaterals(gr);
      mesh.group_sedge.AddColumnsInRow(gr-1, orig_nq);
      // Every quad is subdivided into two triangles
      mesh.group_stria.AddColumnsInRow(gr-1, 2*orig_nq);
      // Existing triangles remain unchanged
      const int orig_nt = orig_mesh.GroupNTriangles(gr);
      mesh.group_stria.AddColumnsInRow(gr-1, orig_nt);
   }
   mesh.group_svert.MakeJ();
   mesh.svert_lvert.Reserve(mesh.group_svert.Size_of_connections());

   mesh.group_sedge.MakeJ();
   mesh.shared_edges.Reserve(mesh.group_sedge.Size_of_connections());
   mesh.sedge_ledge.SetSize(mesh.group_sedge.Size_of_connections());

   mesh.group_stria.MakeJ();
   mesh.shared_trias.Reserve(mesh.group_stria.Size_of_connections());
   mesh.sface_lface.SetSize(mesh.shared_trias.Size());

   mesh.group_squad.MakeJ();

   constexpr int ntris = 2, nv_tri = 3, nv_quad = 4;

   Array<int> dofs;
   for (int gr = 1; gr < mesh.GetNGroups(); gr++)
   {
      // add shared vertices from original shared vertices
      const int orig_n_verts = orig_mesh.GroupNVertices(gr);
      for (int j = 0; j < orig_n_verts; j++)
      {
         fes.GetVertexDofs(orig_mesh.GroupVertex(gr, j), dofs);
         mesh.group_svert.AddConnection(gr-1, mesh.svert_lvert.Append(dofs[0])-1);
      }

      // add original shared edges
      const int orig_n_edges = orig_mesh.GroupNEdges(gr);
      for (int e = 0; e < orig_n_edges; e++)
      {
         int iedge, o;
         orig_mesh.GroupEdge(gr, e, iedge, o);
         Element *elem = mesh.NewElement(Geometry::SEGMENT);
         Array<int> edge_verts;
         orig_mesh.GetEdgeVertices(iedge, edge_verts);
         elem->SetVertices(edge_verts);
         mesh.group_sedge.AddConnection(gr-1, mesh.shared_edges.Append(elem)-1);
      }
      // add original shared triangles
      const int orig_nt = orig_mesh.GroupNTriangles(gr);
      for (int e = 0; e < orig_nt; e++)
      {
         int itri, o;
         orig_mesh.GroupTriangle(gr, e, itri, o);
         const int *v = orig_mesh.GetFace(itri)->GetVertices();
         mesh.shared_trias.SetSize(mesh.shared_trias.Size()+1);
         int *v2 = mesh.shared_trias.Last().v;
         for (int iv=0; iv<nv_tri; ++iv) { v2[iv] = v[iv]; }
         mesh.group_stria.AddConnection(gr-1, mesh.shared_trias.Size()-1);
      }
      // add triangles from split quads and add resulting diagonal edge
      const int orig_nq = orig_mesh.GroupNQuadrilaterals(gr);
      if (orig_nq > 0)
      {
         static const int trimap[12] =
         {
            0, 0, 0, 1,
            1, 2, 1, 2,
            2, 3, 3, 3
         };
         static const int diagmap[4] = { 0, 2, 1, 3 };
         for (int f = 0; f < orig_nq; ++f)
         {
            int iquad, o;
            orig_mesh.GroupQuadrilateral(gr, f, iquad, o);
            const int *v = orig_mesh.GetFace(iquad)->GetVertices();
            // Split quad according the smallest (global) vertex
            int vg[nv_quad];
            for (int iv=0; iv<nv_quad; ++iv) { vg[iv] = vglobal[v[iv]]; }
            int iv_min = std::min_element(vg, vg+nv_quad) - vg;
            int isplit = (iv_min == 0 || iv_min == 2) ? 0 : 1;
            // Add diagonal
            Element *diag = mesh.NewElement(Geometry::SEGMENT);
            int *v_diag = diag->GetVertices();
            v_diag[0] = v[diagmap[0 + isplit*2]];
            v_diag[1] = v[diagmap[1 + isplit*2]];
            mesh.group_sedge.AddConnection(gr-1, mesh.shared_edges.Append(diag)-1);
            // Add two new triangles
            for (int itri=0; itri<ntris; ++itri)
            {
               mesh.shared_trias.SetSize(mesh.shared_trias.Size()+1);
               int *v2 = mesh.shared_trias.Last().v;
               for (int iv=0; iv<nv_tri; ++iv)
               {
                  v2[iv] = v[trimap[itri + isplit*2 + iv*ntris*2]];
               }
               mesh.group_stria.AddConnection(gr-1, mesh.shared_trias.Size()-1);
            }
         }
      }
   }
   mesh.group_svert.ShiftUpI();
   mesh.group_sedge.ShiftUpI();
   mesh.group_stria.ShiftUpI();

   mesh.FinalizeParTopo();

   return mesh;
}

void ParMesh::Finalize(bool refine, bool fix_orientation)
{
   const int meshgen_save = meshgen; // Mesh::Finalize() may call SetMeshGen()
   // 'mesh_geoms' is local, so there's no need to save and restore it.

   Mesh::Finalize(refine, fix_orientation);

   meshgen = meshgen_save;
   // Note: if Mesh::Finalize() calls MarkTetMeshForRefinement() then the
   //       shared_trias have been rotated as necessary.

   // Setup secondary parallel mesh data: sedge_ledge, sface_lface
   FinalizeParTopo();
}

int ParMesh::GetLocalElementNum(long long global_element_num) const
{
   ComputeGlobalElementOffset();
   long long local = global_element_num - glob_elem_offset;
   if (local < 0 || local >= NumOfElements) { return -1; }
   return local;
}

long long ParMesh::GetGlobalElementNum(int local_element_num) const
{
   ComputeGlobalElementOffset();
   return glob_elem_offset + local_element_num;
}

void ParMesh::DistributeAttributes(Array<int> &attr)
{
   // Determine the largest attribute number across all processors
   int max_attr = attr.Size() ? attr.Max() : 1 /*allow empty ranks*/;
   int glb_max_attr = -1;
   MPI_Allreduce(&max_attr, &glb_max_attr, 1, MPI_INT, MPI_MAX, MyComm);

   // Create marker arrays to indicate which attributes are present
   // assuming attribute numbers are in the range [1,glb_max_attr].
   bool *attr_marker = new bool[glb_max_attr];
   bool *glb_attr_marker = new bool[glb_max_attr];
   for (int i = 0; i < glb_max_attr; i++)
   {
      attr_marker[i] = false;
   }
   for (int i = 0; i < attr.Size(); i++)
   {
      attr_marker[attr[i] - 1] = true;
   }
   MPI_Allreduce(attr_marker, glb_attr_marker, glb_max_attr,
                 MPI_C_BOOL, MPI_LOR, MyComm);
   delete [] attr_marker;

   // Translate from the marker array to a unique, sorted list of attributes
   attr.SetSize(0);
   attr.Reserve(glb_max_attr);
   for (int i = 0; i < glb_max_attr; i++)
   {
      if (glb_attr_marker[i])
      {
         attr.Append(i + 1);
      }
   }
   delete [] glb_attr_marker;
}

void ParMesh::SetAttributes()
{
   // Determine the attributes occurring in local interior and boundary elements
   Mesh::SetAttributes();

   DistributeAttributes(bdr_attributes);
   if (bdr_attributes.Size() > 0 && bdr_attributes[0] <= 0)
   {
      MFEM_WARNING("Non-positive boundary element attributes found!");
   }

   DistributeAttributes(attributes);
   if (attributes.Size() > 0 && attributes[0] <= 0)
   {
      MFEM_WARNING("Non-positive element attributes found!");
   }
}

bool ParMesh::HasBoundaryElements() const
{
   // maximum number of boundary elements over all ranks
   int maxNumOfBdrElements;
   MPI_Allreduce(&NumOfBdrElements, &maxNumOfBdrElements, 1,
                 MPI_INT, MPI_MAX, MyComm);
   return (maxNumOfBdrElements > 0);
}

void ParMesh::GroupEdge(int group, int i, int &edge, int &o) const
{
   int sedge = group_sedge.GetRow(group-1)[i];
   edge = sedge_ledge[sedge];
   int *v = shared_edges[sedge]->GetVertices();
   o = (v[0] < v[1]) ? (+1) : (-1);
}

void ParMesh::GroupTriangle(int group, int i, int &face, int &o) const
{
   int stria = group_stria.GetRow(group-1)[i];
   face = sface_lface[stria];
   // face gives the base orientation
   MFEM_ASSERT(faces[face]->GetType() == Element::TRIANGLE,
               "Expecting a triangular face.");

   o = GetTriOrientation(faces[face]->GetVertices(), shared_trias[stria].v);
}

void ParMesh::GroupQuadrilateral(int group, int i, int &face, int &o) const
{
   int squad = group_squad.GetRow(group-1)[i];
   face = sface_lface[shared_trias.Size()+squad];
   // face gives the base orientation
   MFEM_ASSERT(faces[face]->GetType() == Element::QUADRILATERAL,
               "Expecting a quadrilateral face.");

   o = GetQuadOrientation(faces[face]->GetVertices(), shared_quads[squad].v);
}

void ParMesh::GetSharedEdgeCommunicator(int ordering,
                                        GroupCommunicator& sedge_comm) const
{
   Table &gr_sedge = sedge_comm.GroupLDofTable();
   gr_sedge.SetDims(GetNGroups(), shared_edges.Size());
   gr_sedge.GetI()[0] = 0;
   for (int gr = 1; gr <= GetNGroups(); gr++)
   {
      gr_sedge.GetI()[gr] = group_sedge.GetI()[gr-1];
   }
   for (int k = 0; k < shared_edges.Size(); k++)
   {
      if (ordering == 1)
      {
         gr_sedge.GetJ()[k] =k;
      }
      else
      {
         gr_sedge.GetJ()[k] = group_sedge.GetJ()[k];
      }
   }
   sedge_comm.Finalize();
}

void ParMesh::GetSharedVertexCommunicator(int ordering,
                                          GroupCommunicator& svert_comm) const
{
   Table &gr_svert = svert_comm.GroupLDofTable();
   gr_svert.SetDims(GetNGroups(), svert_lvert.Size());
   gr_svert.GetI()[0] = 0;
   for (int gr = 1; gr <= GetNGroups(); gr++)
   {
      gr_svert.GetI()[gr] = group_svert.GetI()[gr-1];
   }
   for (int k = 0; k < svert_lvert.Size(); k++)
   {
      if (ordering == 1)
      {
         gr_svert.GetJ()[k] = k;
      }
      else
      {
         gr_svert.GetJ()[k] = group_svert.GetJ()[k];
      }
   }
   svert_comm.Finalize();
}

void ParMesh::GetSharedQuadCommunicator(int ordering,
                                        GroupCommunicator& squad_comm) const
{
   Table &gr_squad = squad_comm.GroupLDofTable();
   gr_squad.SetDims(GetNGroups(), shared_quads.Size());
   gr_squad.GetI()[0] = 0;
   for (int gr = 1; gr <= GetNGroups(); gr++)
   {
      gr_squad.GetI()[gr] = group_squad.GetI()[gr-1];
   }
   for (int k = 0; k < shared_quads.Size(); k++)
   {
      if (ordering == 1)
      {
         gr_squad.GetJ()[k] = k;
      }
      else
      {
         gr_squad.GetJ()[k] = group_squad.GetJ()[k];
      }
   }
   squad_comm.Finalize();
}

void ParMesh::GetSharedTriCommunicator(int ordering,
                                       GroupCommunicator& stria_comm) const
{
   Table &gr_stria = stria_comm.GroupLDofTable();
   gr_stria.SetDims(GetNGroups(), shared_trias.Size());
   gr_stria.GetI()[0] = 0;
   for (int gr = 1; gr <= GetNGroups(); gr++)
   {
      gr_stria.GetI()[gr] = group_stria.GetI()[gr-1];
   }
   for (int k = 0; k < shared_trias.Size(); k++)
   {
      if (ordering == 1)
      {
         gr_stria.GetJ()[k] = k;
      }
      else
      {
         gr_stria.GetJ()[k] = group_stria.GetJ()[k];
      }
   }
   stria_comm.Finalize();
}

void ParMesh::MarkTetMeshForRefinement(const DSTable &v_to_v)
{
   Array<int> order;
   GetEdgeOrdering(v_to_v, order); // local edge ordering

   // create a GroupCommunicator on the shared edges
   GroupCommunicator sedge_comm(gtopo);
   GetSharedEdgeCommunicator(0, sedge_comm);

   Array<int> sedge_ord(shared_edges.Size());
   Array<Pair<int,int> > sedge_ord_map(shared_edges.Size());
   for (int k = 0; k < shared_edges.Size(); k++)
   {
      // sedge_ledge may be undefined -- use shared_edges and v_to_v instead
      const int sedge = group_sedge.GetJ()[k];
      const int *v = shared_edges[sedge]->GetVertices();
      sedge_ord[k] = order[v_to_v(v[0], v[1])];
   }

   sedge_comm.Bcast<int>(sedge_ord, 1);

   for (int k = 0, gr = 1; gr < GetNGroups(); gr++)
   {
      const int n = group_sedge.RowSize(gr-1);
      if (n == 0) { continue; }
      sedge_ord_map.SetSize(n);
      for (int j = 0; j < n; j++)
      {
         sedge_ord_map[j].one = sedge_ord[k+j];
         sedge_ord_map[j].two = j;
      }
      SortPairs<int, int>(sedge_ord_map, n);
      for (int j = 0; j < n; j++)
      {
         const int sedge_from = group_sedge.GetJ()[k+j];
         const int *v = shared_edges[sedge_from]->GetVertices();
         sedge_ord[k+j] = order[v_to_v(v[0], v[1])];
      }
      std::sort(&sedge_ord[k], &sedge_ord[k] + n);
      for (int j = 0; j < n; j++)
      {
         const int sedge_to = group_sedge.GetJ()[k+sedge_ord_map[j].two];
         const int *v = shared_edges[sedge_to]->GetVertices();
         order[v_to_v(v[0], v[1])] = sedge_ord[k+j];
      }
      k += n;
   }

#ifdef MFEM_DEBUG
   {
      Array<Pair<int, real_t> > ilen_len(order.Size());

      for (int i = 0; i < NumOfVertices; i++)
      {
         for (DSTable::RowIterator it(v_to_v, i); !it; ++it)
         {
            int j = it.Index();
            ilen_len[j].one = order[j];
            ilen_len[j].two = GetLength(i, it.Column());
         }
      }

      SortPairs<int, real_t>(ilen_len, order.Size());

      real_t d_max = 0.;
      for (int i = 1; i < order.Size(); i++)
      {
         d_max = std::max(d_max, ilen_len[i-1].two-ilen_len[i].two);
      }

#if 0
      // Debug message from every MPI rank.
      mfem::out << "proc. " << MyRank << '/' << NRanks << ": d_max = " << d_max
                << endl;
#else
      // Debug message just from rank 0.
      real_t glob_d_max;
      MPI_Reduce(&d_max, &glob_d_max, 1, MPITypeMap<real_t>::mpi_type, MPI_MAX, 0,
                 MyComm);
      if (MyRank == 0)
      {
         mfem::out << "glob_d_max = " << glob_d_max << endl;
      }
#endif
   }
#endif

   // use 'order' to mark the tets, the boundary triangles, and the shared
   // triangle faces
   for (int i = 0; i < NumOfElements; i++)
   {
      if (elements[i]->GetType() == Element::TETRAHEDRON)
      {
         elements[i]->MarkEdge(v_to_v, order);
      }
   }

   for (int i = 0; i < NumOfBdrElements; i++)
   {
      if (boundary[i]->GetType() == Element::TRIANGLE)
      {
         boundary[i]->MarkEdge(v_to_v, order);
      }
   }

   for (int i = 0; i < shared_trias.Size(); i++)
   {
      Triangle::MarkEdge(shared_trias[i].v, v_to_v, order);
   }
}

// For a line segment with vertices v[0] and v[1], return a number with
// the following meaning:
// 0 - the edge was not refined
// 1 - the edge e was refined once by splitting v[0],v[1]
int ParMesh::GetEdgeSplittings(Element *edge, const DSTable &v_to_v,
                               int *middle)
{
   int m, *v = edge->GetVertices();

   if ((m = v_to_v(v[0], v[1])) != -1 && middle[m] != -1)
   {
      return 1;
   }
   else
   {
      return 0;
   }
}

void ParMesh::GetFaceSplittings(const int *fv, const HashTable<Hashed2> &v_to_v,
                                Array<unsigned> &codes)
{
   typedef Triple<int,int,int> face_t;
   Array<face_t> face_stack;

   unsigned code = 0;
   face_stack.Append(face_t(fv[0], fv[1], fv[2]));
   for (unsigned bit = 0; face_stack.Size() > 0; bit++)
   {
      if (bit == 8*sizeof(unsigned))
      {
         codes.Append(code);
         code = bit = 0;
      }

      const face_t &f = face_stack.Last();
      int mid = v_to_v.FindId(f.one, f.two);
      if (mid == -1)
      {
         // leave a 0 at bit 'bit'
         face_stack.DeleteLast();
      }
      else
      {
         code += (1 << bit); // set bit 'bit' to 1
         mid += NumOfVertices;
         face_stack.Append(face_t(f.three, f.one, mid));
         face_t &r = face_stack[face_stack.Size()-2];
         r = face_t(r.two, r.three, mid);
      }
   }
   codes.Append(code);
}

bool ParMesh::DecodeFaceSplittings(HashTable<Hashed2> &v_to_v, const int *v,
                                   const Array<unsigned> &codes, int &pos)
{
   typedef Triple<int,int,int> face_t;
   Array<face_t> face_stack;

   bool need_refinement = 0;
   face_stack.Append(face_t(v[0], v[1], v[2]));
   for (unsigned bit = 0, code = codes[pos++]; face_stack.Size() > 0; bit++)
   {
      if (bit == 8*sizeof(unsigned))
      {
         code = codes[pos++];
         bit = 0;
      }

      if ((code & (1 << bit)) == 0) { face_stack.DeleteLast(); continue; }

      const face_t &f = face_stack.Last();
      int mid = v_to_v.FindId(f.one, f.two);
      if (mid == -1)
      {
         mid = v_to_v.GetId(f.one, f.two);
         int ind[2] = { f.one, f.two };
         vertices.Append(Vertex());
         AverageVertices(ind, 2, vertices.Size()-1);
         need_refinement = 1;
      }
      mid += NumOfVertices;
      face_stack.Append(face_t(f.three, f.one, mid));
      face_t &r = face_stack[face_stack.Size()-2];
      r = face_t(r.two, r.three, mid);
   }
   return need_refinement;
}

void ParMesh::GenerateOffsets(int N, HYPRE_BigInt loc_sizes[],
                              Array<HYPRE_BigInt> *offsets[]) const
{
   if (HYPRE_AssumedPartitionCheck())
   {
      Array<HYPRE_BigInt> temp(N);
      MPI_Scan(loc_sizes, temp.GetData(), N, HYPRE_MPI_BIG_INT, MPI_SUM, MyComm);
      for (int i = 0; i < N; i++)
      {
         offsets[i]->SetSize(3);
         (*offsets[i])[0] = temp[i] - loc_sizes[i];
         (*offsets[i])[1] = temp[i];
      }
      MPI_Bcast(temp.GetData(), N, HYPRE_MPI_BIG_INT, NRanks-1, MyComm);
      for (int i = 0; i < N; i++)
      {
         (*offsets[i])[2] = temp[i];
         // check for overflow
         MFEM_VERIFY((*offsets[i])[0] >= 0 && (*offsets[i])[1] >= 0,
                     "overflow in offsets");
      }
   }
   else
   {
      Array<HYPRE_BigInt> temp(N*NRanks);
      MPI_Allgather(loc_sizes, N, HYPRE_MPI_BIG_INT, temp.GetData(), N,
                    HYPRE_MPI_BIG_INT, MyComm);
      for (int i = 0; i < N; i++)
      {
         Array<HYPRE_BigInt> &offs = *offsets[i];
         offs.SetSize(NRanks+1);
         offs[0] = 0;
         for (int j = 0; j < NRanks; j++)
         {
            offs[j+1] = offs[j] + temp[i+N*j];
         }
         // Check for overflow
         MFEM_VERIFY(offs[MyRank] >= 0 && offs[MyRank+1] >= 0,
                     "overflow in offsets");
      }
   }
}

void ParMesh::DeleteFaceNbrData()
{
   if (!have_face_nbr_data)
   {
      return;
   }

   have_face_nbr_data = false;
   face_nbr_group.DeleteAll();
   face_nbr_elements_offset.DeleteAll();
   face_nbr_vertices_offset.DeleteAll();
   for (int i = 0; i < face_nbr_elements.Size(); i++)
   {
      FreeElement(face_nbr_elements[i]);
   }
   face_nbr_elements.DeleteAll();
   face_nbr_vertices.DeleteAll();
   send_face_nbr_elements.Clear();
   send_face_nbr_vertices.Clear();
}

void ParMesh::SetCurvature(int order, bool discont, int space_dim, int ordering)
{
   DeleteFaceNbrData();
   space_dim = (space_dim == -1) ? spaceDim : space_dim;
   FiniteElementCollection* nfec;
   if (discont)
   {
      nfec = new L2_FECollection(order, Dim, BasisType::GaussLobatto);
   }
   else
   {
      nfec = new H1_FECollection(order, Dim);
   }
   ParFiniteElementSpace* nfes = new ParFiniteElementSpace(this, nfec, space_dim,
                                                           ordering);
   auto pnodes = new ParGridFunction(nfes);
   GetNodes(*pnodes);
   NewNodes(*pnodes, true);
   Nodes->MakeOwner(nfec);
}

void ParMesh::SetNodalFESpace(FiniteElementSpace *nfes)
{
   DeleteFaceNbrData();
   ParFiniteElementSpace *npfes = dynamic_cast<ParFiniteElementSpace*>(nfes);
   if (npfes)
   {
      SetNodalFESpace(npfes);
   }
   else
   {
      Mesh::SetNodalFESpace(nfes);
   }
}

void ParMesh::SetNodalFESpace(ParFiniteElementSpace *npfes)
{
   DeleteFaceNbrData();
   ParGridFunction *nodes = new ParGridFunction(npfes);
   SetNodalGridFunction(nodes, true);
}

void ParMesh::EnsureParNodes()
{
   if (Nodes && dynamic_cast<ParFiniteElementSpace*>(Nodes->FESpace()) == NULL)
   {
      DeleteFaceNbrData();
      ParFiniteElementSpace *pfes =
         new ParFiniteElementSpace(*Nodes->FESpace(), *this);
      ParGridFunction *new_nodes = new ParGridFunction(pfes);
      *new_nodes = *Nodes;
      if (Nodes->OwnFEC())
      {
         new_nodes->MakeOwner(Nodes->OwnFEC());
         Nodes->MakeOwner(NULL); // takes away ownership of 'fec' and 'fes'
         delete Nodes->FESpace();
      }
      delete Nodes;
      Nodes = new_nodes;
   }
}

void ParMesh::ExchangeFaceNbrData()
{
   if (have_face_nbr_data)
   {
      return;
   }

   if (Nonconforming())
   {
      // With ParNCMesh we can set up face neighbors mostly without communication.
      pncmesh->GetFaceNeighbors(*this);
      have_face_nbr_data = true;

      ExchangeFaceNbrNodes();
      return;
   }

   Table *gr_sface;
   int   *s2l_face;
   bool   del_tables = false;
   if (Dim == 1)
   {
      gr_sface = &group_svert;
      s2l_face = svert_lvert;
   }
   else if (Dim == 2)
   {
      gr_sface = &group_sedge;
      s2l_face = sedge_ledge;
   }
   else
   {
      s2l_face = sface_lface;
      if (shared_trias.Size() == sface_lface.Size())
      {
         // All shared faces are Triangular
         gr_sface = &group_stria;
      }
      else if (shared_quads.Size() == sface_lface.Size())
      {
         // All shared faced are Quadrilateral
         gr_sface = &group_squad;
      }
      else
      {
         // Shared faces contain a mixture of triangles and quads
         gr_sface = new Table;
         del_tables = true;

         // Merge the Tables group_stria and group_squad
         gr_sface->MakeI(group_stria.Size());
         for (int gr=0; gr<group_stria.Size(); gr++)
         {
            gr_sface->AddColumnsInRow(gr,
                                      group_stria.RowSize(gr) +
                                      group_squad.RowSize(gr));
         }
         gr_sface->MakeJ();
         const int nst = shared_trias.Size();
         for (int gr=0; gr<group_stria.Size(); gr++)
         {
            gr_sface->AddConnections(gr,
                                     group_stria.GetRow(gr),
                                     group_stria.RowSize(gr));
            for (int c=0; c<group_squad.RowSize(gr); c++)
            {
               gr_sface->AddConnection(gr,
                                       nst + group_squad.GetRow(gr)[c]);
            }
         }
         gr_sface->ShiftUpI();
      }
   }

   ExchangeFaceNbrData(gr_sface, s2l_face);

   if (Dim == 3)
   {
      BuildFaceNbrElementToFaceTable();
   }

   if (del_tables) { delete gr_sface; }

   if ( have_face_nbr_data ) { return; }

   have_face_nbr_data = true;

   ExchangeFaceNbrNodes();
}

void ParMesh::ExchangeFaceNbrData(Table *gr_sface, int *s2l_face)
{
   int num_face_nbrs = 0;
   for (int g = 1; g < GetNGroups(); g++)
   {
      if (gr_sface->RowSize(g-1) > 0)
      {
         num_face_nbrs++;
      }
   }

   face_nbr_group.SetSize(num_face_nbrs);

   if (num_face_nbrs == 0)
   {
      have_face_nbr_data = true;
      return;
   }

   {
      // sort face-neighbors by processor rank
      Array<Pair<int, int> > rank_group(num_face_nbrs);

      for (int g = 1, counter = 0; g < GetNGroups(); g++)
      {
         if (gr_sface->RowSize(g-1) > 0)
         {
            MFEM_ASSERT(gtopo.GetGroupSize(g) == 2, "group size is not 2!");

            const int *nbs = gtopo.GetGroup(g);
            int lproc = (nbs[0]) ? nbs[0] : nbs[1];
            rank_group[counter].one = gtopo.GetNeighborRank(lproc);
            rank_group[counter].two = g;
            counter++;
         }
      }

      SortPairs<int, int>(rank_group, rank_group.Size());

      for (int fn = 0; fn < num_face_nbrs; fn++)
      {
         face_nbr_group[fn] = rank_group[fn].two;
      }
   }

   MPI_Request *requests = new MPI_Request[2*num_face_nbrs];
   MPI_Request *send_requests = requests;
   MPI_Request *recv_requests = requests + num_face_nbrs;
   MPI_Status  *statuses = new MPI_Status[num_face_nbrs];

   int *nbr_data = new int[6*num_face_nbrs];
   int *nbr_send_data = nbr_data;
   int *nbr_recv_data = nbr_data + 3*num_face_nbrs;

   Array<int> el_marker(GetNE());
   Array<int> vertex_marker(GetNV());
   el_marker = -1;
   vertex_marker = -1;

   Array<int> fcs, cor;

   Table send_face_nbr_elemdata, send_face_nbr_facedata;

   send_face_nbr_elements.MakeI(num_face_nbrs);
   send_face_nbr_vertices.MakeI(num_face_nbrs);
   send_face_nbr_elemdata.MakeI(num_face_nbrs);
   send_face_nbr_facedata.MakeI(num_face_nbrs);
   for (int fn = 0; fn < num_face_nbrs; fn++)
   {
      int nbr_group = face_nbr_group[fn];
      int  num_sfaces = gr_sface->RowSize(nbr_group-1);
      int *sface = gr_sface->GetRow(nbr_group-1);
      for (int i = 0; i < num_sfaces; i++)
      {
         int lface = s2l_face[sface[i]];
         int el = faces_info[lface].Elem1No;
         if (el_marker[el] != fn)
         {
            el_marker[el] = fn;
            send_face_nbr_elements.AddAColumnInRow(fn);

            const int nv = elements[el]->GetNVertices();
            const int *v = elements[el]->GetVertices();
            for (int j = 0; j < nv; j++)
               if (vertex_marker[v[j]] != fn)
               {
                  vertex_marker[v[j]] = fn;
                  send_face_nbr_vertices.AddAColumnInRow(fn);
               }

            const int nf = elements[el]->GetNFaces();

            send_face_nbr_elemdata.AddColumnsInRow(fn, nv + nf + 2);
         }
      }
      send_face_nbr_facedata.AddColumnsInRow(fn, 2*num_sfaces);

      nbr_send_data[3*fn  ] = send_face_nbr_elements.GetI()[fn];
      nbr_send_data[3*fn+1] = send_face_nbr_vertices.GetI()[fn];
      nbr_send_data[3*fn+2] = send_face_nbr_elemdata.GetI()[fn];

      int nbr_rank = GetFaceNbrRank(fn);
      int tag = 0;

      MPI_Isend(&nbr_send_data[3*fn], 3, MPI_INT, nbr_rank, tag, MyComm,
                &send_requests[fn]);
      MPI_Irecv(&nbr_recv_data[3*fn], 3, MPI_INT, nbr_rank, tag, MyComm,
                &recv_requests[fn]);
   }
   send_face_nbr_elements.MakeJ();
   send_face_nbr_vertices.MakeJ();
   send_face_nbr_elemdata.MakeJ();
   send_face_nbr_facedata.MakeJ();
   el_marker = -1;
   vertex_marker = -1;
   const int nst = shared_trias.Size();
   for (int fn = 0; fn < num_face_nbrs; fn++)
   {
      int nbr_group = face_nbr_group[fn];
      int  num_sfaces = gr_sface->RowSize(nbr_group-1);
      int *sface = gr_sface->GetRow(nbr_group-1);
      for (int i = 0; i < num_sfaces; i++)
      {
         const int sf = sface[i];
         int lface = s2l_face[sf];
         int el = faces_info[lface].Elem1No;
         if (el_marker[el] != fn)
         {
            el_marker[el] = fn;
            send_face_nbr_elements.AddConnection(fn, el);

            const int nv = elements[el]->GetNVertices();
            const int *v = elements[el]->GetVertices();
            for (int j = 0; j < nv; j++)
               if (vertex_marker[v[j]] != fn)
               {
                  vertex_marker[v[j]] = fn;
                  send_face_nbr_vertices.AddConnection(fn, v[j]);
               }

            send_face_nbr_elemdata.AddConnection(fn, GetAttribute(el));
            send_face_nbr_elemdata.AddConnection(
               fn, GetElementBaseGeometry(el));
            send_face_nbr_elemdata.AddConnections(fn, v, nv);

            if (Dim == 3)
            {
               const int nf = elements[el]->GetNFaces();
               GetElementFaces(el, fcs, cor);
               send_face_nbr_elemdata.AddConnections(fn, cor, nf);
            }
         }
         send_face_nbr_facedata.AddConnection(fn, el);
         int info = faces_info[lface].Elem1Inf;
         // change the orientation in info to be relative to the shared face
         //   in 1D and 2D keep the orientation equal to 0
         if (Dim == 3)
         {
            const int *lf_v = faces[lface]->GetVertices();
            if (sf < nst) // triangle shared face
            {
               info += GetTriOrientation(shared_trias[sf].v, lf_v);
            }
            else // quad shared face
            {
               info += GetQuadOrientation(shared_quads[sf-nst].v, lf_v);
            }
         }
         send_face_nbr_facedata.AddConnection(fn, info);
      }
   }
   send_face_nbr_elements.ShiftUpI();
   send_face_nbr_vertices.ShiftUpI();
   send_face_nbr_elemdata.ShiftUpI();
   send_face_nbr_facedata.ShiftUpI();

   // convert the vertex indices in send_face_nbr_elemdata
   // convert the element indices in send_face_nbr_facedata
   for (int fn = 0; fn < num_face_nbrs; fn++)
   {
      int  num_elems  = send_face_nbr_elements.RowSize(fn);
      int *elems      = send_face_nbr_elements.GetRow(fn);
      int  num_verts  = send_face_nbr_vertices.RowSize(fn);
      int *verts      = send_face_nbr_vertices.GetRow(fn);
      int *elemdata   = send_face_nbr_elemdata.GetRow(fn);
      int  num_sfaces = send_face_nbr_facedata.RowSize(fn)/2;
      int *facedata   = send_face_nbr_facedata.GetRow(fn);

      for (int i = 0; i < num_verts; i++)
      {
         vertex_marker[verts[i]] = i;
      }

      for (int el = 0; el < num_elems; el++)
      {
         const int nv = elements[elems[el]]->GetNVertices();
         const int nf = (Dim == 3) ? elements[elems[el]]->GetNFaces() : 0;
         elemdata += 2; // skip the attribute and the geometry type
         for (int j = 0; j < nv; j++)
         {
            elemdata[j] = vertex_marker[elemdata[j]];
         }
         elemdata += nv + nf;

         el_marker[elems[el]] = el;
      }

      for (int i = 0; i < num_sfaces; i++)
      {
         facedata[2*i] = el_marker[facedata[2*i]];
      }
   }

   MPI_Waitall(num_face_nbrs, recv_requests, statuses);

   Array<int> recv_face_nbr_facedata;
   Table recv_face_nbr_elemdata;

   // fill-in face_nbr_elements_offset, face_nbr_vertices_offset
   face_nbr_elements_offset.SetSize(num_face_nbrs + 1);
   face_nbr_vertices_offset.SetSize(num_face_nbrs + 1);
   recv_face_nbr_elemdata.MakeI(num_face_nbrs);
   face_nbr_elements_offset[0] = 0;
   face_nbr_vertices_offset[0] = 0;
   for (int fn = 0; fn < num_face_nbrs; fn++)
   {
      face_nbr_elements_offset[fn+1] =
         face_nbr_elements_offset[fn] + nbr_recv_data[3*fn];
      face_nbr_vertices_offset[fn+1] =
         face_nbr_vertices_offset[fn] + nbr_recv_data[3*fn+1];
      recv_face_nbr_elemdata.AddColumnsInRow(fn, nbr_recv_data[3*fn+2]);
   }
   recv_face_nbr_elemdata.MakeJ();

   MPI_Waitall(num_face_nbrs, send_requests, statuses);

   // send and receive the element data
   for (int fn = 0; fn < num_face_nbrs; fn++)
   {
      int nbr_rank = GetFaceNbrRank(fn);
      int tag = 0;

      MPI_Isend(send_face_nbr_elemdata.GetRow(fn),
                send_face_nbr_elemdata.RowSize(fn),
                MPI_INT, nbr_rank, tag, MyComm, &send_requests[fn]);

      MPI_Irecv(recv_face_nbr_elemdata.GetRow(fn),
                recv_face_nbr_elemdata.RowSize(fn),
                MPI_INT, nbr_rank, tag, MyComm, &recv_requests[fn]);
   }

   // convert the element data into face_nbr_elements
   face_nbr_elements.SetSize(face_nbr_elements_offset[num_face_nbrs]);
   face_nbr_el_ori.reset(new Table(face_nbr_elements_offset[num_face_nbrs], 6));
   while (true)
   {
      int fn;
      MPI_Waitany(num_face_nbrs, recv_requests, &fn, statuses);

      if (fn == MPI_UNDEFINED)
      {
         break;
      }

      int  vert_off      = face_nbr_vertices_offset[fn];
      int  elem_off      = face_nbr_elements_offset[fn];
      int  num_elems     = face_nbr_elements_offset[fn+1] - elem_off;
      int *recv_elemdata = recv_face_nbr_elemdata.GetRow(fn);

      for (int i = 0; i < num_elems; i++)
      {
         Element *el = NewElement(recv_elemdata[1]);
         el->SetAttribute(recv_elemdata[0]);
         recv_elemdata += 2;
         int nv = el->GetNVertices();
         for (int j = 0; j < nv; j++)
         {
            recv_elemdata[j] += vert_off;
         }
         el->SetVertices(recv_elemdata);
         recv_elemdata += nv;
         if (Dim == 3)
         {
            int nf = el->GetNFaces();
            int * fn_ori = face_nbr_el_ori->GetRow(elem_off);
            for (int j = 0; j < nf; j++)
            {
               fn_ori[j] = recv_elemdata[j];
            }
            recv_elemdata += nf;
         }
         face_nbr_elements[elem_off++] = el;
      }
   }
   face_nbr_el_ori->Finalize();

   MPI_Waitall(num_face_nbrs, send_requests, statuses);

   // send and receive the face data
   recv_face_nbr_facedata.SetSize(
      send_face_nbr_facedata.Size_of_connections());
   for (int fn = 0; fn < num_face_nbrs; fn++)
   {
      int nbr_rank = GetFaceNbrRank(fn);
      int tag = 0;

      MPI_Isend(send_face_nbr_facedata.GetRow(fn),
                send_face_nbr_facedata.RowSize(fn),
                MPI_INT, nbr_rank, tag, MyComm, &send_requests[fn]);

      // the size of the send and receive face data is the same
      MPI_Irecv(&recv_face_nbr_facedata[send_face_nbr_facedata.GetI()[fn]],
                send_face_nbr_facedata.RowSize(fn),
                MPI_INT, nbr_rank, tag, MyComm, &recv_requests[fn]);
   }

   // transfer the received face data into faces_info
   while (true)
   {
      int fn;
      MPI_Waitany(num_face_nbrs, recv_requests, &fn, statuses);

      if (fn == MPI_UNDEFINED)
      {
         break;
      }

      int  elem_off   = face_nbr_elements_offset[fn];
      int  nbr_group  = face_nbr_group[fn];
      int  num_sfaces = gr_sface->RowSize(nbr_group-1);
      int *sface      = gr_sface->GetRow(nbr_group-1);
      int *facedata =
         &recv_face_nbr_facedata[send_face_nbr_facedata.GetI()[fn]];

      for (int i = 0; i < num_sfaces; i++)
      {
         const int sf = sface[i];
         int lface = s2l_face[sf];
         FaceInfo &face_info = faces_info[lface];
         face_info.Elem2No = -1 - (facedata[2*i] + elem_off);
         int info = facedata[2*i+1];
         // change the orientation in info to be relative to the local face
         if (Dim < 3)
         {
            info++; // orientation 0 --> orientation 1
         }
         else
         {
            int nbr_ori = info%64, nbr_v[4];
            const int *lf_v = faces[lface]->GetVertices();

            if (sf < nst) // triangle shared face
            {
               // apply the nbr_ori to sf_v to get nbr_v
               const int *perm = tri_t::Orient[nbr_ori];
               const int *sf_v = shared_trias[sf].v;
               for (int j = 0; j < 3; j++)
               {
                  nbr_v[perm[j]] = sf_v[j];
               }
               // get the orientation of nbr_v w.r.t. the local face
               nbr_ori = GetTriOrientation(lf_v, nbr_v);
            }
            else // quad shared face
            {
               // apply the nbr_ori to sf_v to get nbr_v
               const int *perm = quad_t::Orient[nbr_ori];
               const int *sf_v = shared_quads[sf-nst].v;
               for (int j = 0; j < 4; j++)
               {
                  nbr_v[perm[j]] = sf_v[j];
               }
               // get the orientation of nbr_v w.r.t. the local face
               nbr_ori = GetQuadOrientation(lf_v, nbr_v);
            }

            info = 64*(info/64) + nbr_ori;
         }
         face_info.Elem2Inf = info;
      }
   }

   MPI_Waitall(num_face_nbrs, send_requests, statuses);

   // allocate the face_nbr_vertices
   face_nbr_vertices.SetSize(face_nbr_vertices_offset[num_face_nbrs]);

   delete [] nbr_data;

   delete [] statuses;
   delete [] requests;
}

void ParMesh::ExchangeFaceNbrNodes()
{
   if (!have_face_nbr_data)
   {
      ExchangeFaceNbrData(); // calls this method at the end
   }
   else if (Nodes == NULL)
   {
      if (Nonconforming())
      {
         // with ParNCMesh we already have the vertices
         return;
      }

      int num_face_nbrs = GetNFaceNeighbors();

      if (!num_face_nbrs) { return; }

      MPI_Request *requests = new MPI_Request[2*num_face_nbrs];
      MPI_Request *send_requests = requests;
      MPI_Request *recv_requests = requests + num_face_nbrs;
      MPI_Status  *statuses = new MPI_Status[num_face_nbrs];

      // allocate buffer and copy the vertices to be sent
      Array<Vertex> send_vertices(send_face_nbr_vertices.Size_of_connections());
      for (int i = 0; i < send_vertices.Size(); i++)
      {
         send_vertices[i] = vertices[send_face_nbr_vertices.GetJ()[i]];
      }

      // send and receive the vertices
      for (int fn = 0; fn < num_face_nbrs; fn++)
      {
         int nbr_rank = GetFaceNbrRank(fn);
         int tag = 0;

         MPI_Isend(send_vertices[send_face_nbr_vertices.GetI()[fn]](),
                   3*send_face_nbr_vertices.RowSize(fn),
                   MPITypeMap<real_t>::mpi_type, nbr_rank, tag, MyComm, &send_requests[fn]);

         MPI_Irecv(face_nbr_vertices[face_nbr_vertices_offset[fn]](),
                   3*(face_nbr_vertices_offset[fn+1] -
                      face_nbr_vertices_offset[fn]),
                   MPITypeMap<real_t>::mpi_type, nbr_rank, tag, MyComm, &recv_requests[fn]);
      }

      MPI_Waitall(num_face_nbrs, recv_requests, statuses);
      MPI_Waitall(num_face_nbrs, send_requests, statuses);

      delete [] statuses;
      delete [] requests;
   }
   else
   {
      ParGridFunction *pNodes = dynamic_cast<ParGridFunction *>(Nodes);
      MFEM_VERIFY(pNodes != NULL, "Nodes are not ParGridFunction!");
      pNodes->ExchangeFaceNbrData();
   }
}

STable3D *ParMesh::GetSharedFacesTable()
{
   STable3D *sfaces_tbl = new STable3D(face_nbr_vertices.Size());
   for (int i = 0; i < face_nbr_elements.Size(); i++)
   {
      const int *v = face_nbr_elements[i]->GetVertices();
      switch (face_nbr_elements[i]->GetType())
      {
         case Element::TETRAHEDRON:
         {
            for (int j = 0; j < 4; j++)
            {
               const int *fv = tet_t::FaceVert[j];
               sfaces_tbl->Push(v[fv[0]], v[fv[1]], v[fv[2]]);
            }
            break;
         }
         case Element::WEDGE:
         {
            for (int j = 0; j < 2; j++)
            {
               const int *fv = pri_t::FaceVert[j];
               sfaces_tbl->Push(v[fv[0]], v[fv[1]], v[fv[2]]);
            }
            for (int j = 2; j < 5; j++)
            {
               const int *fv = pri_t::FaceVert[j];
               sfaces_tbl->Push4(v[fv[0]], v[fv[1]], v[fv[2]], v[fv[3]]);
            }
            break;
         }
         case Element::PYRAMID:
         {
            for (int j = 0; j < 1; j++)
            {
               const int *fv = pyr_t::FaceVert[j];
               sfaces_tbl->Push4(v[fv[0]], v[fv[1]], v[fv[2]], v[fv[3]]);
            }
            for (int j = 1; j < 5; j++)
            {
               const int *fv = pyr_t::FaceVert[j];
               sfaces_tbl->Push(v[fv[0]], v[fv[1]], v[fv[2]]);
            }
            break;
         }
         case Element::HEXAHEDRON:
         {
            // find the face by the vertices with the smallest 3 numbers
            // z = 0, y = 0, x = 1, y = 1, x = 0, z = 1
            for (int j = 0; j < 6; j++)
            {
               const int *fv = hex_t::FaceVert[j];
               sfaces_tbl->Push4(v[fv[0]], v[fv[1]], v[fv[2]], v[fv[3]]);
            }
            break;
         }
         default:
            MFEM_ABORT("Unexpected type of Element.");
      }
   }
   return sfaces_tbl;
}

template <int N>
void
ParMesh::AddTriFaces(const Array<int> &elem_vertices,
                     const std::unique_ptr<STable3D> &faces,
                     const std::unique_ptr<STable3D> &shared_faces,
                     int elem, int start, int end, const int fverts[][N])
{
   for (int i = start; i < end; ++i)
   {
      // Reference face vertices.
      const auto fv = fverts[i];
      // Element specific face vertices.
      const Vert3 elem_fv(elem_vertices[fv[0]], elem_vertices[fv[1]],
                          elem_vertices[fv[2]]);

      // Check amongst the faces of elements local to this rank for this set of vertices
      const int lf = faces->Index(elem_fv.v[0], elem_fv.v[1], elem_fv.v[2]);

      // If the face wasn't found amonst processor local elements, search the
      // ghosts for this set of vertices.
      const int sf = lf < 0 ? shared_faces->Index(elem_fv.v[0], elem_fv.v[1],
                                                  elem_fv.v[2]) : -1;
      // If find local face -> use that
      //    else if find shared face -> shift and use that
      //       else no face found -> set to -1
      const int face_to_add = lf < 0 ? (sf >= 0 ? sf + NumOfFaces : -1) : lf;

      MFEM_ASSERT(sf >= 0 ||
                  lf >= 0, "Face must be from a local or a face neighbor element");

      // Add this discovered face to the list of faces of this face neighbor element
      face_nbr_el_to_face->Push(elem, face_to_add);
   }
}

void ParMesh::BuildFaceNbrElementToFaceTable()
{
   const auto faces = std::unique_ptr<STable3D>(GetFacesTable());
   const auto shared_faces = std::unique_ptr<STable3D>(GetSharedFacesTable());

   face_nbr_el_to_face.reset(new Table(face_nbr_elements.Size(), 6));

   Array<int> v;

   // Helper for adding quadrilateral faces.
   auto add_quad_faces = [&faces, &shared_faces, &v, this]
                         (int elem, int start, int end, const int fverts[][4])
   {
      for (int i = start; i < end; ++i)
      {
         const int * const fv = fverts[i];
         int k = 0;
         int max = v[fv[0]];

         if (max < v[fv[1]]) { max = v[fv[1]], k = 1; }
         if (max < v[fv[2]]) { max = v[fv[2]], k = 2; }
         if (max < v[fv[3]]) { k = 3; }

         int v0 = -1, v1 = -1, v2 = -1;
         switch (k)
         {
            case 0:
               v0 = v[fv[1]]; v1 = v[fv[2]]; v2 = v[fv[3]];
               break;
            case 1:
               v0 = v[fv[0]]; v1 = v[fv[2]]; v2 = v[fv[3]];
               break;
            case 2:
               v0 = v[fv[0]]; v1 = v[fv[1]]; v2 = v[fv[3]];
               break;
            case 3:
               v0 = v[fv[0]]; v1 = v[fv[1]]; v2 = v[fv[2]];
               break;
         }
         int lf = faces->Index(v0, v1, v2);
         if (lf < 0)
         {
            lf = shared_faces->Index(v0, v1, v2);
            if (lf >= 0)
            {
               lf += NumOfFaces;
            }
         }
         face_nbr_el_to_face->Push(elem, lf);
      }
   };

   for (int i = 0; i < face_nbr_elements.Size(); i++)
   {
      face_nbr_elements[i]->GetVertices(v);
      switch (face_nbr_elements[i]->GetType())
      {
         case Element::TETRAHEDRON:
         {
            AddTriFaces(v, faces, shared_faces, i, 0, 4, tet_t::FaceVert);
            break;
         }
         case Element::WEDGE:
         {
            AddTriFaces(v, faces, shared_faces, i, 0, 2, pri_t::FaceVert);
            add_quad_faces(i, 2, 5, pri_t::FaceVert);
            break;
         }
         case Element::PYRAMID:
         {
            add_quad_faces(i, 0, 1, pyr_t::FaceVert);
            AddTriFaces(v, faces, shared_faces, i, 1, 5, pyr_t::FaceVert);
            break;
         }
         case Element::HEXAHEDRON:
         {
            add_quad_faces(i, 0, 6, hex_t::FaceVert);
            break;
         }
         default:
            MFEM_ABORT("Unexpected type of Element.");
      }
   }
   face_nbr_el_to_face->Finalize();
}

int ParMesh::GetFaceNbrRank(int fn) const
{
   if (Conforming())
   {
      int nbr_group = face_nbr_group[fn];
      const int *nbs = gtopo.GetGroup(nbr_group);
      int nbr_lproc = (nbs[0]) ? nbs[0] : nbs[1];
      int nbr_rank = gtopo.GetNeighborRank(nbr_lproc);
      return nbr_rank;
   }
   else
   {
      // NC: simplified handling of face neighbor ranks
      return face_nbr_group[fn];
   }
}

void
ParMesh::GetFaceNbrElementFaces(int i, Array<int> &faces,
                                Array<int> &orientations) const
{
   int el_nbr = i - GetNE();
   if (face_nbr_el_to_face != nullptr && el_nbr < face_nbr_el_to_face->Size())
   {
      face_nbr_el_to_face->GetRow(el_nbr, faces);
   }
   else
   {
      MFEM_ABORT("ParMesh::GetFaceNbrElementFaces(...) : "
                 "face_nbr_el_to_face not generated correctly.");
   }

   if (face_nbr_el_ori != nullptr && el_nbr < face_nbr_el_ori->Size())
   {
      face_nbr_el_ori->GetRow(el_nbr, orientations);
   }
   else
   {
      MFEM_ABORT("ParMesh::GetFaceNbrElementFaces(...) : "
                 "face_nbr_el_ori not generated correctly.");
   }
}

Table *ParMesh::GetFaceToAllElementTable() const
{
   const Array<int> *s2l_face;
   if (Dim == 1)
   {
      s2l_face = &svert_lvert;
   }
   else if (Dim == 2)
   {
      s2l_face = &sedge_ledge;
   }
   else
   {
      s2l_face = &sface_lface;
   }

   Table *face_elem = new Table;

   face_elem->MakeI(faces_info.Size());

   for (int i = 0; i < faces_info.Size(); i++)
   {
      if (faces_info[i].Elem2No >= 0)
      {
         face_elem->AddColumnsInRow(i, 2);
      }
      else
      {
         face_elem->AddAColumnInRow(i);
      }
   }
   for (int i = 0; i < s2l_face->Size(); i++)
   {
      face_elem->AddAColumnInRow((*s2l_face)[i]);
   }

   face_elem->MakeJ();

   for (int i = 0; i < faces_info.Size(); i++)
   {
      face_elem->AddConnection(i, faces_info[i].Elem1No);
      if (faces_info[i].Elem2No >= 0)
      {
         face_elem->AddConnection(i, faces_info[i].Elem2No);
      }
   }
   for (int i = 0; i < s2l_face->Size(); i++)
   {
      int lface = (*s2l_face)[i];
      int nbr_elem_idx = -1 - faces_info[lface].Elem2No;
      face_elem->AddConnection(lface, NumOfElements + nbr_elem_idx);
   }

   face_elem->ShiftUpI();

   return face_elem;
}

FaceElementTransformations *ParMesh::GetFaceElementTransformations(int FaceNo,
                                                                   int mask)
{
   GetFaceElementTransformations(FaceNo, FaceElemTr, Transformation,
                                 Transformation2, mask);
   return &FaceElemTr;
}

void ParMesh::GetFaceElementTransformations(int FaceNo,
                                            FaceElementTransformations &FElTr,
                                            IsoparametricTransformation &ElTr1,
                                            IsoparametricTransformation &ElTr2,
                                            int mask) const
{
   if (FaceNo < GetNumFaces())
   {
      Mesh::GetFaceElementTransformations(FaceNo, FElTr, ElTr1, ElTr2, mask);
   }
   else
   {
      const bool fill2 = mask & 10; // Elem2 and/or Loc2
      GetSharedFaceTransformationsByLocalIndex(FaceNo, FElTr, ElTr1, ElTr2,
                                               fill2);
   }
}

FaceElementTransformations *ParMesh::GetSharedFaceTransformations(int sf,
                                                                  bool fill2)
{
   GetSharedFaceTransformations(sf, FaceElemTr, Transformation,
                                Transformation2, fill2);
   return &FaceElemTr;
}

void ParMesh::GetSharedFaceTransformations(int sf,
                                           FaceElementTransformations &FElTr,
                                           IsoparametricTransformation &ElTr1,
                                           IsoparametricTransformation &ElTr2,
                                           bool fill2) const
{
   int FaceNo = GetSharedFace(sf);
   GetSharedFaceTransformationsByLocalIndex(FaceNo, FElTr, ElTr1, ElTr2, fill2);
}

FaceElementTransformations *
ParMesh::GetSharedFaceTransformationsByLocalIndex(int FaceNo, bool fill2)
{
   GetSharedFaceTransformationsByLocalIndex(FaceNo, FaceElemTr, Transformation,
                                            Transformation2, fill2);
   return &FaceElemTr;
}

void ParMesh::GetSharedFaceTransformationsByLocalIndex(
   int FaceNo, FaceElementTransformations &FElTr,
   IsoparametricTransformation &ElTr1, IsoparametricTransformation &ElTr2,
   bool fill2) const
{
   const FaceInfo &face_info = faces_info[FaceNo];
   MFEM_VERIFY(face_info.Elem2Inf >= 0, "The face must be shared.");

   bool is_slave = Nonconforming() && IsSlaveFace(face_info);
   bool is_ghost = Nonconforming() && FaceNo >= GetNumFaces();

   int mask = 0;
   FElTr.SetConfigurationMask(0);
   FElTr.Elem1 = NULL;
   FElTr.Elem2 = NULL;

   int local_face =
      is_ghost ? nc_faces_info[face_info.NCFace].MasterFace : FaceNo;
   Element::Type  face_type = GetFaceElementType(local_face);
   Geometry::Type face_geom = GetFaceGeometry(local_face);

   // setup the transformation for the first element
   FElTr.Elem1No = face_info.Elem1No;
   GetElementTransformation(FElTr.Elem1No, &ElTr1);
   FElTr.Elem1 = &ElTr1;
   mask |= FaceElementTransformations::HAVE_ELEM1;

   // setup the transformation for the second (neighbor) element
   int Elem2NbrNo;
   if (fill2)
   {
      Elem2NbrNo = -1 - face_info.Elem2No;
      // Store the "shifted index" for element 2 in FElTr.Elem2No.
      // `Elem2NbrNo` is the index of the face neighbor (starting from 0),
      // and `FElTr.Elem2No` will be offset by the number of (local)
      // elements in the mesh.
      FElTr.Elem2No = NumOfElements + Elem2NbrNo;
      GetFaceNbrElementTransformation(Elem2NbrNo, ElTr2);
      FElTr.Elem2 = &ElTr2;
      mask |= FaceElementTransformations::HAVE_ELEM2;
   }
   else
   {
      FElTr.Elem2No = -1;
   }

   // setup the face transformation if the face is not a ghost
   if (!is_ghost)
   {
      GetFaceTransformation(FaceNo, &FElTr);
      // NOTE: The above call overwrites FElTr.Loc1
      mask |= FaceElementTransformations::HAVE_FACE;
   }
   else
   {
      FElTr.SetGeometryType(face_geom);
   }

   // setup Loc1 & Loc2
   int elem_type = GetElementType(face_info.Elem1No);
   GetLocalFaceTransformation(face_type, elem_type, FElTr.Loc1.Transf,
                              face_info.Elem1Inf);
   mask |= FaceElementTransformations::HAVE_LOC1;

   if (fill2)
   {
      elem_type = face_nbr_elements[Elem2NbrNo]->GetType();
      GetLocalFaceTransformation(face_type, elem_type, FElTr.Loc2.Transf,
                                 face_info.Elem2Inf);
      mask |= FaceElementTransformations::HAVE_LOC2;
   }

   // adjust Loc1 or Loc2 of the master face if this is a slave face
   if (is_slave)
   {
      if (is_ghost || fill2)
      {
         // is_ghost -> modify side 1, otherwise -> modify side 2:
         ApplyLocalSlaveTransformation(FElTr, face_info, is_ghost);
      }
   }

   // for ghost faces we need a special version of GetFaceTransformation
   if (is_ghost)
   {
      GetGhostFaceTransformation(FElTr, face_type, face_geom);
      mask |= FaceElementTransformations::HAVE_FACE;
   }

   FElTr.SetConfigurationMask(mask);

   // This check can be useful for internal debugging, however it will fail on
   // periodic boundary faces, so we keep it disabled in general.
#if 0
#ifdef MFEM_DEBUG
   real_t dist = FElTr.CheckConsistency();
   if (dist >= 1e-12)
   {
      mfem::out << "\nInternal error: face id = " << FaceNo
                << ", dist = " << dist << ", rank = " << MyRank << '\n';
      FElTr.CheckConsistency(1); // print coordinates
      MFEM_ABORT("internal error");
   }
#endif
#endif
}

void ParMesh::GetGhostFaceTransformation(
   FaceElementTransformations &FElTr, Element::Type face_type,
   Geometry::Type face_geom) const
{
   // calculate composition of FElTr.Loc1 and FElTr.Elem1
   DenseMatrix &face_pm = FElTr.GetPointMat();
   FElTr.Reset();
   if (Nodes == NULL)
   {
      FElTr.Elem1->Transform(FElTr.Loc1.Transf.GetPointMat(), face_pm);
      FElTr.SetFE(GetTransformationFEforElementType(face_type));
   }
   else
   {
      const FiniteElement* face_el =
         Nodes->FESpace()->GetTraceElement(FElTr.Elem1No, face_geom);
      MFEM_VERIFY(dynamic_cast<const NodalFiniteElement*>(face_el),
                  "Mesh requires nodal Finite Element.");

#if 0 // TODO: handle the case of non-interpolatory Nodes
      DenseMatrix I;
      face_el->Project(Transformation.GetFE(), FElTr.Loc1.Transf, I);
      MultABt(Transformation.GetPointMat(), I, pm_face);
#else
      IntegrationRule eir(face_el->GetDof());
      FElTr.Loc1.Transform(face_el->GetNodes(), eir);
      Nodes->GetVectorValues(*FElTr.Elem1, eir, face_pm);
#endif
      FElTr.SetFE(face_el);
   }
}

ElementTransformation *ParMesh::GetFaceNbrElementTransformation(int FaceNo)
{
   GetFaceNbrElementTransformation(FaceNo, Transformation);
   return &Transformation;
}

void ParMesh::GetFaceNbrElementTransformation(
   int FaceNo, IsoparametricTransformation &ElTr) const
{
   DenseMatrix &pointmat = ElTr.GetPointMat();
   Element *elem = face_nbr_elements[FaceNo];

   ElTr.Attribute = elem->GetAttribute();
   ElTr.ElementNo = NumOfElements + FaceNo;
   ElTr.ElementType = ElementTransformation::ELEMENT;
   ElTr.mesh = this;
   ElTr.Reset();

   if (Nodes == NULL)
   {
      const int nv = elem->GetNVertices();
      const int *v = elem->GetVertices();

      pointmat.SetSize(spaceDim, nv);
      for (int k = 0; k < spaceDim; k++)
      {
         for (int j = 0; j < nv; j++)
         {
            pointmat(k, j) = face_nbr_vertices[v[j]](k);
         }
      }

      ElTr.SetFE(GetTransformationFEforElementType(elem->GetType()));
   }
   else
   {
      Array<int> vdofs;
      ParGridFunction *pNodes = dynamic_cast<ParGridFunction *>(Nodes);
      if (pNodes)
      {
         pNodes->ParFESpace()->GetFaceNbrElementVDofs(FaceNo, vdofs);
         int n = vdofs.Size()/spaceDim;
         pointmat.SetSize(spaceDim, n);
         for (int k = 0; k < spaceDim; k++)
         {
            for (int j = 0; j < n; j++)
            {
               pointmat(k,j) = (pNodes->FaceNbrData())(vdofs[n*k+j]);
            }
         }

         ElTr.SetFE(pNodes->ParFESpace()->GetFaceNbrFE(FaceNo));
      }
      else
      {
         MFEM_ABORT("Nodes are not ParGridFunction!");
      }
   }
}

real_t ParMesh::GetFaceNbrElementSize(int i, int type)
{
   return GetElementSize(GetFaceNbrElementTransformation(i), type);
}

int ParMesh::GetNSharedFaces() const
{
   if (Conforming())
   {
      switch (Dim)
      {
         case 1:  return svert_lvert.Size();
         case 2:  return sedge_ledge.Size();
         default: return sface_lface.Size();
      }
   }
   else
   {
      MFEM_ASSERT(Dim > 1, "");
      const NCMesh::NCList &shared = pncmesh->GetSharedList(Dim-1);
      return shared.conforming.Size() + shared.slaves.Size();
   }
}

int ParMesh::GetSharedFace(int sface) const
{
   if (Conforming())
   {
      switch (Dim)
      {
         case 1:  return svert_lvert[sface];
         case 2:  return sedge_ledge[sface];
         default: return sface_lface[sface];
      }
   }
   else
   {
      MFEM_ASSERT(Dim > 1, "");
      const NCMesh::NCList &shared = pncmesh->GetSharedList(Dim-1);
      int csize = shared.conforming.Size();
      return sface < csize
             ? shared.conforming[sface].index
             : shared.slaves[sface - csize].index;
   }
}

int ParMesh::GetNFbyType(FaceType type) const
{
   const_cast<ParMesh*>(this)->ExchangeFaceNbrData();
   return Mesh::GetNFbyType(type);
}

// shift cyclically 3 integers a, b, c, so that the smallest of
// order[a], order[b], order[c] is first
static inline
void Rotate3Indirect(int &a, int &b, int &c,
                     const Array<std::int64_t> &order)
{
   if (order[a] < order[b])
   {
      if (order[a] > order[c])
      {
         ShiftRight(a, b, c);
      }
   }
   else
   {
      if (order[b] < order[c])
      {
         ShiftRight(c, b, a);
      }
      else
      {
         ShiftRight(a, b, c);
      }
   }
}

void ParMesh::ReorientTetMesh()
{
   if (Dim != 3 || !(meshgen & 1))
   {
      return;
   }

   ResetLazyData();

   DSTable *old_v_to_v = NULL;
   Table *old_elem_vert = NULL;

   if (Nodes)
   {
      PrepareNodeReorder(&old_v_to_v, &old_elem_vert);
   }

   // create a GroupCommunicator over shared vertices
   GroupCommunicator svert_comm(gtopo);
   GetSharedVertexCommunicator(0, svert_comm);

   // communicate the local index of each shared vertex from the group master to
   // other ranks in the group
   Array<int> svert_master_rank(svert_lvert.Size());
   Array<int> svert_master_index(svert_lvert);
   for (int i = 0; i < group_svert.Size(); i++)
   {
      int rank = gtopo.GetGroupMasterRank(i+1);
      for (int j = 0; j < group_svert.RowSize(i); j++)
      {
         svert_master_rank[group_svert.GetRow(i)[j]] = rank;
      }
   }
   svert_comm.Bcast(svert_master_index);

   // the pairs (master rank, master local index) define a globally consistent
   // vertex ordering
   Array<std::int64_t> glob_vert_order(vertices.Size());
   {
      Array<int> lvert_svert(vertices.Size());
      lvert_svert = -1;
      for (int i = 0; i < svert_lvert.Size(); i++)
      {
         lvert_svert[svert_lvert[i]] = i;
      }

      for (int i = 0; i < vertices.Size(); i++)
      {
         int s = lvert_svert[i];
         if (s >= 0)
         {
            glob_vert_order[i] =
               (std::int64_t(svert_master_rank[s]) << 32) + svert_master_index[s];
         }
         else
         {
            glob_vert_order[i] = (std::int64_t(MyRank) << 32) + i;
         }
      }
   }

   // rotate tetrahedra so that vertex zero is the lowest (global) index vertex,
   // vertex 1 is the second lowest (global) index and vertices 2 and 3 preserve
   // positive orientation of the element
   for (int i = 0; i < NumOfElements; i++)
   {
      if (GetElementType(i) == Element::TETRAHEDRON)
      {
         int *v = elements[i]->GetVertices();

         Rotate3Indirect(v[0], v[1], v[2], glob_vert_order);

         if (glob_vert_order[v[0]] < glob_vert_order[v[3]])
         {
            Rotate3Indirect(v[1], v[2], v[3], glob_vert_order);
         }
         else
         {
            ShiftRight(v[0], v[1], v[3]);
         }
      }
   }

   // rotate also boundary triangles
   for (int i = 0; i < NumOfBdrElements; i++)
   {
      if (GetBdrElementType(i) == Element::TRIANGLE)
      {
         int *v = boundary[i]->GetVertices();

         Rotate3Indirect(v[0], v[1], v[2], glob_vert_order);
      }
   }

   const bool check_consistency = true;
   if (check_consistency)
   {
      // create a GroupCommunicator on the shared triangles
      GroupCommunicator stria_comm(gtopo);
      GetSharedTriCommunicator(0, stria_comm);

      Array<int> stria_flag(shared_trias.Size());
      for (int i = 0; i < stria_flag.Size(); i++)
      {
         const int *v = shared_trias[i].v;
         if (glob_vert_order[v[0]] < glob_vert_order[v[1]])
         {
            stria_flag[i] = (glob_vert_order[v[0]] < glob_vert_order[v[2]]) ? 0 : 2;
         }
         else // v[1] < v[0]
         {
            stria_flag[i] = (glob_vert_order[v[1]] < glob_vert_order[v[2]]) ? 1 : 2;
         }
      }

      Array<int> stria_master_flag(stria_flag);
      stria_comm.Bcast(stria_master_flag);
      for (int i = 0; i < stria_flag.Size(); i++)
      {
         const int *v = shared_trias[i].v;
         MFEM_VERIFY(stria_flag[i] == stria_master_flag[i],
                     "inconsistent vertex ordering found, shared triangle "
                     << i << ": ("
                     << v[0] << ", " << v[1] << ", " << v[2] << "), "
                     << "local flag: " << stria_flag[i]
                     << ", master flag: " << stria_master_flag[i]);
      }
   }

   // rotate shared triangle faces
   for (int i = 0; i < shared_trias.Size(); i++)
   {
      int *v = shared_trias[i].v;

      Rotate3Indirect(v[0], v[1], v[2], glob_vert_order);
   }

   // finalize
   if (!Nodes)
   {
      GetElementToFaceTable();
      GenerateFaces();
      if (el_to_edge)
      {
         NumOfEdges = GetElementToEdgeTable(*el_to_edge);
      }
   }
   else
   {
      DoNodeReorder(old_v_to_v, old_elem_vert);
      delete old_elem_vert;
      delete old_v_to_v;
   }

   // the local edge and face numbering is changed therefore we need to
   // update sedge_ledge and sface_lface.
   FinalizeParTopo();
}

void ParMesh::LocalRefinement(const Array<int> &marked_el, int type)
{
   if (pncmesh)
   {
      MFEM_ABORT("Local and nonconforming refinements cannot be mixed.");
   }

   DeleteFaceNbrData();

   InitRefinementTransforms();

   if (Dim == 3)
   {
      int uniform_refinement = 0;
      if (type < 0)
      {
         type = -type;
         uniform_refinement = 1;
      }

      // 1. Hash table of vertex to vertex connections corresponding to refined
      //    edges.
      HashTable<Hashed2> v_to_v;

      // 2. Do the red refinement.
      switch (type)
      {
         case 1:
            for (int i = 0; i < marked_el.Size(); i++)
            {
               Bisection(marked_el[i], v_to_v);
            }
            break;
         case 2:
            for (int i = 0; i < marked_el.Size(); i++)
            {
               Bisection(marked_el[i], v_to_v);

               Bisection(NumOfElements - 1, v_to_v);
               Bisection(marked_el[i], v_to_v);
            }
            break;
         case 3:
            for (int i = 0; i < marked_el.Size(); i++)
            {
               Bisection(marked_el[i], v_to_v);

               int j = NumOfElements - 1;
               Bisection(j, v_to_v);
               Bisection(NumOfElements - 1, v_to_v);
               Bisection(j, v_to_v);

               Bisection(marked_el[i], v_to_v);
               Bisection(NumOfElements-1, v_to_v);
               Bisection(marked_el[i], v_to_v);
            }
            break;
      }

      // 3. Do the green refinement (to get conforming mesh).
      int need_refinement;
      int max_faces_in_group = 0;
      // face_splittings identify how the shared faces have been split
      Array<unsigned> *face_splittings = new Array<unsigned>[GetNGroups()-1];
      for (int i = 0; i < GetNGroups()-1; i++)
      {
         const int faces_in_group = GroupNTriangles(i+1);
         face_splittings[i].Reserve(faces_in_group);
         if (faces_in_group > max_faces_in_group)
         {
            max_faces_in_group = faces_in_group;
         }
      }
      int neighbor;
      Array<unsigned> iBuf(max_faces_in_group);

      MPI_Request *requests = new MPI_Request[GetNGroups()-1];
      MPI_Status  status;

#ifdef MFEM_DEBUG_PARMESH_LOCALREF
      int ref_loops_all = 0, ref_loops_par = 0;
#endif
      do
      {
         need_refinement = 0;
         for (int i = 0; i < NumOfElements; i++)
         {
            if (elements[i]->NeedRefinement(v_to_v))
            {
               need_refinement = 1;
               Bisection(i, v_to_v);
            }
         }
#ifdef MFEM_DEBUG_PARMESH_LOCALREF
         ref_loops_all++;
#endif

         if (uniform_refinement)
         {
            continue;
         }

         // if the mesh is locally conforming start making it globally
         // conforming
         if (need_refinement == 0)
         {
#ifdef MFEM_DEBUG_PARMESH_LOCALREF
            ref_loops_par++;
#endif
            // MPI_Barrier(MyComm);
            const int tag = 293;

            // (a) send the type of interface splitting
            int req_count = 0;
            for (int i = 0; i < GetNGroups()-1; i++)
            {
               const int *group_faces = group_stria.GetRow(i);
               const int faces_in_group = group_stria.RowSize(i);
               // it is enough to communicate through the faces
               if (faces_in_group == 0) { continue; }

               face_splittings[i].SetSize(0);
               for (int j = 0; j < faces_in_group; j++)
               {
                  GetFaceSplittings(shared_trias[group_faces[j]].v, v_to_v,
                                    face_splittings[i]);
               }
               const int *nbs = gtopo.GetGroup(i+1);
               neighbor = gtopo.GetNeighborRank(nbs[0] ? nbs[0] : nbs[1]);
               MPI_Isend(face_splittings[i], face_splittings[i].Size(),
                         MPI_UNSIGNED, neighbor, tag, MyComm,
                         &requests[req_count++]);
            }

            // (b) receive the type of interface splitting
            for (int i = 0; i < GetNGroups()-1; i++)
            {
               const int *group_faces = group_stria.GetRow(i);
               const int faces_in_group = group_stria.RowSize(i);
               if (faces_in_group == 0) { continue; }

               const int *nbs = gtopo.GetGroup(i+1);
               neighbor = gtopo.GetNeighborRank(nbs[0] ? nbs[0] : nbs[1]);
               MPI_Probe(neighbor, tag, MyComm, &status);
               int count;
               MPI_Get_count(&status, MPI_UNSIGNED, &count);
               iBuf.SetSize(count);
               MPI_Recv(iBuf, count, MPI_UNSIGNED, neighbor, tag, MyComm,
                        MPI_STATUS_IGNORE);

               for (int j = 0, pos = 0; j < faces_in_group; j++)
               {
                  const int *v = shared_trias[group_faces[j]].v;
                  need_refinement |= DecodeFaceSplittings(v_to_v, v, iBuf, pos);
               }
            }

            int nr = need_refinement;
            MPI_Allreduce(&nr, &need_refinement, 1, MPI_INT, MPI_LOR, MyComm);

            MPI_Waitall(req_count, requests, MPI_STATUSES_IGNORE);
         }
      }
      while (need_refinement == 1);

#ifdef MFEM_DEBUG_PARMESH_LOCALREF
      {
         int i = ref_loops_all;
         MPI_Reduce(&i, &ref_loops_all, 1, MPI_INT, MPI_MAX, 0, MyComm);
         if (MyRank == 0)
         {
            mfem::out << "\n\nParMesh::LocalRefinement : max. ref_loops_all = "
                      << ref_loops_all << ", ref_loops_par = " << ref_loops_par
                      << '\n' << endl;
         }
      }
#endif

      delete [] requests;
      iBuf.DeleteAll();
      delete [] face_splittings;

      // 4. Update the boundary elements.
      do
      {
         need_refinement = 0;
         for (int i = 0; i < NumOfBdrElements; i++)
         {
            if (boundary[i]->NeedRefinement(v_to_v))
            {
               need_refinement = 1;
               BdrBisection(i, v_to_v);
            }
         }
      }
      while (need_refinement == 1);

      if (NumOfBdrElements != boundary.Size())
      {
         mfem_error("ParMesh::LocalRefinement :"
                    " (NumOfBdrElements != boundary.Size())");
      }

      ResetLazyData();

      const int old_nv = NumOfVertices;
      NumOfVertices = vertices.Size();

      RefineGroups(old_nv, v_to_v);

      // 5. Update the groups after refinement.
      if (el_to_face != NULL)
      {
         GetElementToFaceTable();
         GenerateFaces();
      }

      // 6. Update element-to-edge relations.
      if (el_to_edge != NULL)
      {
         NumOfEdges = GetElementToEdgeTable(*el_to_edge);
      }
   } //  'if (Dim == 3)'


   if (Dim == 2)
   {
      int uniform_refinement = 0;
      if (type < 0)
      {
         // type = -type; // not used
         uniform_refinement = 1;
      }

      // 1. Get table of vertex to vertex connections.
      DSTable v_to_v(NumOfVertices);
      GetVertexToVertexTable(v_to_v);

      // 2. Get edge to element connections in arrays edge1 and edge2
      int nedges  = v_to_v.NumberOfEntries();
      int *edge1  = new int[nedges];
      int *edge2  = new int[nedges];
      int *middle = new int[nedges];

      for (int i = 0; i < nedges; i++)
      {
         edge1[i] = edge2[i] = middle[i] = -1;
      }

      for (int i = 0; i < NumOfElements; i++)
      {
         int *v = elements[i]->GetVertices();
         for (int j = 0; j < 3; j++)
         {
            int ind = v_to_v(v[j], v[(j+1)%3]);
            (edge1[ind] == -1) ? (edge1[ind] = i) : (edge2[ind] = i);
         }
      }

      // 3. Do the red refinement.
      for (int i = 0; i < marked_el.Size(); i++)
      {
         RedRefinement(marked_el[i], v_to_v, edge1, edge2, middle);
      }

      // 4. Do the green refinement (to get conforming mesh).
      int need_refinement;
      int edges_in_group, max_edges_in_group = 0;
      // edge_splittings identify how the shared edges have been split
      int **edge_splittings = new int*[GetNGroups()-1];
      for (int i = 0; i < GetNGroups()-1; i++)
      {
         edges_in_group = GroupNEdges(i+1);
         edge_splittings[i] = new int[edges_in_group];
         if (edges_in_group > max_edges_in_group)
         {
            max_edges_in_group = edges_in_group;
         }
      }
      int neighbor, *iBuf = new int[max_edges_in_group];

      Array<int> group_edges;

      MPI_Request request;
      MPI_Status  status;
      Vertex V;
      V(2) = 0.0;

#ifdef MFEM_DEBUG_PARMESH_LOCALREF
      int ref_loops_all = 0, ref_loops_par = 0;
#endif
      do
      {
         need_refinement = 0;
         for (int i = 0; i < nedges; i++)
         {
            if (middle[i] != -1 && edge1[i] != -1)
            {
               need_refinement = 1;
               GreenRefinement(edge1[i], v_to_v, edge1, edge2, middle);
            }
         }
#ifdef MFEM_DEBUG_PARMESH_LOCALREF
         ref_loops_all++;
#endif

         if (uniform_refinement)
         {
            continue;
         }

         // if the mesh is locally conforming start making it globally
         // conforming
         if (need_refinement == 0)
         {
#ifdef MFEM_DEBUG_PARMESH_LOCALREF
            ref_loops_par++;
#endif
            // MPI_Barrier(MyComm);

            // (a) send the type of interface splitting
            for (int i = 0; i < GetNGroups()-1; i++)
            {
               group_sedge.GetRow(i, group_edges);
               edges_in_group = group_edges.Size();
               // it is enough to communicate through the edges
               if (edges_in_group != 0)
               {
                  for (int j = 0; j < edges_in_group; j++)
                  {
                     edge_splittings[i][j] =
                        GetEdgeSplittings(shared_edges[group_edges[j]], v_to_v,
                                          middle);
                  }
                  const int *nbs = gtopo.GetGroup(i+1);
                  if (nbs[0] == 0)
                  {
                     neighbor = gtopo.GetNeighborRank(nbs[1]);
                  }
                  else
                  {
                     neighbor = gtopo.GetNeighborRank(nbs[0]);
                  }
                  MPI_Isend(edge_splittings[i], edges_in_group, MPI_INT,
                            neighbor, 0, MyComm, &request);
               }
            }

            // (b) receive the type of interface splitting
            for (int i = 0; i < GetNGroups()-1; i++)
            {
               group_sedge.GetRow(i, group_edges);
               edges_in_group = group_edges.Size();
               if (edges_in_group != 0)
               {
                  const int *nbs = gtopo.GetGroup(i+1);
                  if (nbs[0] == 0)
                  {
                     neighbor = gtopo.GetNeighborRank(nbs[1]);
                  }
                  else
                  {
                     neighbor = gtopo.GetNeighborRank(nbs[0]);
                  }
                  MPI_Recv(iBuf, edges_in_group, MPI_INT, neighbor,
                           MPI_ANY_TAG, MyComm, &status);

                  for (int j = 0; j < edges_in_group; j++)
                  {
                     if (iBuf[j] == 1 && edge_splittings[i][j] == 0)
                     {
                        int *v = shared_edges[group_edges[j]]->GetVertices();
                        int ii = v_to_v(v[0], v[1]);
#ifdef MFEM_DEBUG_PARMESH_LOCALREF
                        if (middle[ii] != -1)
                        {
                           mfem_error("ParMesh::LocalRefinement (triangles) : "
                                      "Oops!");
                        }
#endif
                        need_refinement = 1;
                        middle[ii] = NumOfVertices++;
                        for (int c = 0; c < spaceDim; c++)
                        {
                           V(c) = 0.5 * (vertices[v[0]](c) + vertices[v[1]](c));
                        }
                        vertices.Append(V);
                     }
                  }
               }
            }

            int nr = need_refinement;
            MPI_Allreduce(&nr, &need_refinement, 1, MPI_INT, MPI_LOR, MyComm);
         }
      }
      while (need_refinement == 1);

#ifdef MFEM_DEBUG_PARMESH_LOCALREF
      {
         int i = ref_loops_all;
         MPI_Reduce(&i, &ref_loops_all, 1, MPI_INT, MPI_MAX, 0, MyComm);
         if (MyRank == 0)
         {
            mfem::out << "\n\nParMesh::LocalRefinement : max. ref_loops_all = "
                      << ref_loops_all << ", ref_loops_par = " << ref_loops_par
                      << '\n' << endl;
         }
      }
#endif

      for (int i = 0; i < GetNGroups()-1; i++)
      {
         delete [] edge_splittings[i];
      }
      delete [] edge_splittings;

      delete [] iBuf;

      // 5. Update the boundary elements.
      int v1[2], v2[2], bisect, temp;
      temp = NumOfBdrElements;
      for (int i = 0; i < temp; i++)
      {
         int *v = boundary[i]->GetVertices();
         bisect = v_to_v(v[0], v[1]);
         if (middle[bisect] != -1)
         {
            // the element was refined (needs updating)
            if (boundary[i]->GetType() == Element::SEGMENT)
            {
               v1[0] =           v[0]; v1[1] = middle[bisect];
               v2[0] = middle[bisect]; v2[1] =           v[1];

               boundary[i]->SetVertices(v1);
               boundary.Append(new Segment(v2, boundary[i]->GetAttribute()));
            }
            else
            {
               mfem_error("Only bisection of segment is implemented for bdr"
                          " elem.");
            }
         }
      }
      NumOfBdrElements = boundary.Size();

      ResetLazyData();

      // 5a. Update the groups after refinement.
      RefineGroups(v_to_v, middle);

      // 6. Free the allocated memory.
      delete [] edge1;
      delete [] edge2;
      delete [] middle;

      if (el_to_edge != NULL)
      {
         NumOfEdges = GetElementToEdgeTable(*el_to_edge);
         GenerateFaces();
      }
   } //  'if (Dim == 2)'

   if (Dim == 1) // --------------------------------------------------------
   {
      int cne = NumOfElements, cnv = NumOfVertices;
      NumOfVertices += marked_el.Size();
      NumOfElements += marked_el.Size();
      vertices.SetSize(NumOfVertices);
      elements.SetSize(NumOfElements);
      CoarseFineTr.embeddings.SetSize(NumOfElements);

      for (int j = 0; j < marked_el.Size(); j++)
      {
         int i = marked_el[j];
         Segment *c_seg = (Segment *)elements[i];
         int *vert = c_seg->GetVertices(), attr = c_seg->GetAttribute();
         int new_v = cnv + j, new_e = cne + j;
         AverageVertices(vert, 2, new_v);
         elements[new_e] = new Segment(new_v, vert[1], attr);
         vert[1] = new_v;

         CoarseFineTr.embeddings[i] = Embedding(i, Geometry::SEGMENT, 1);
         CoarseFineTr.embeddings[new_e] = Embedding(i, Geometry::SEGMENT, 2);
      }

      static real_t seg_children[3*2] = { 0.0,1.0, 0.0,0.5, 0.5,1.0 };
      CoarseFineTr.point_matrices[Geometry::SEGMENT].
      UseExternalData(seg_children, 1, 2, 3);

      GenerateFaces();
   } // end of 'if (Dim == 1)'

   last_operation = Mesh::REFINE;
   sequence++;

   UpdateNodes();

#ifdef MFEM_DEBUG
   CheckElementOrientation(false);
   CheckBdrElementOrientation(false);
#endif
}

void ParMesh::NonconformingRefinement(const Array<Refinement> &refinements,
                                      int nc_limit)
{
   if (NURBSext)
   {
      MFEM_ABORT("NURBS meshes are not supported. Please project the "
                 "NURBS to Nodes first with SetCurvature().");
   }

   if (!pncmesh)
   {
      MFEM_ABORT("Can't convert conforming ParMesh to nonconforming ParMesh "
                 "(you need to initialize the ParMesh from a nonconforming "
                 "serial Mesh)");
   }

   ResetLazyData();

   DeleteFaceNbrData();

   // NOTE: no check of !refinements.Size(), in parallel we would have to reduce

   // do the refinements
   pncmesh->MarkCoarseLevel();
   pncmesh->Refine(refinements);

   if (nc_limit > 0)
   {
      pncmesh->LimitNCLevel(nc_limit);
   }

   // create a second mesh containing the finest elements from 'pncmesh'
   ParMesh* pmesh2 = new ParMesh(*pncmesh);
   pncmesh->OnMeshUpdated(pmesh2);

   attributes.Copy(pmesh2->attributes);
   bdr_attributes.Copy(pmesh2->bdr_attributes);

   // Copy attribute and bdr_attribute names
   attribute_sets.Copy(pmesh2->attribute_sets);
   bdr_attribute_sets.Copy(pmesh2->bdr_attribute_sets);

   // now swap the meshes, the second mesh will become the old coarse mesh
   // and this mesh will be the new fine mesh
   Mesh::Swap(*pmesh2, false);

   delete pmesh2; // NOTE: old face neighbors destroyed here

   pncmesh->GetConformingSharedStructures(*this);

   GenerateNCFaceInfo();

   last_operation = Mesh::REFINE;
   sequence++;

   UpdateNodes();
}

bool ParMesh::NonconformingDerefinement(Array<real_t> &elem_error,
                                        real_t threshold, int nc_limit, int op)
{
   MFEM_VERIFY(pncmesh, "Only supported for non-conforming meshes.");
   MFEM_VERIFY(!NURBSext, "Derefinement of NURBS meshes is not supported. "
               "Project the NURBS to Nodes first.");

   const Table &dt = pncmesh->GetDerefinementTable();

   pncmesh->SynchronizeDerefinementData(elem_error, dt);

   Array<int> level_ok;
   if (nc_limit > 0)
   {
      pncmesh->CheckDerefinementNCLevel(dt, level_ok, nc_limit);
   }

   Array<int> derefs;
   for (int i = 0; i < dt.Size(); i++)
   {
      if (nc_limit > 0 && !level_ok[i]) { continue; }

      real_t error =
         AggregateError(elem_error, dt.GetRow(i), dt.RowSize(i), op);

      if (error < threshold) { derefs.Append(i); }
   }

   long long glob_size = ReduceInt(derefs.Size());
   if (!glob_size) { return false; }

   // Destroy face-neighbor data only when actually de-refining.
   DeleteFaceNbrData();

   pncmesh->Derefine(derefs);

   ParMesh* mesh2 = new ParMesh(*pncmesh);
   pncmesh->OnMeshUpdated(mesh2);

   attributes.Copy(mesh2->attributes);
   bdr_attributes.Copy(mesh2->bdr_attributes);

   // Copy attribute and bdr_attribute names
   attribute_sets.Copy(mesh2->attribute_sets);
   bdr_attribute_sets.Copy(mesh2->bdr_attribute_sets);

   Mesh::Swap(*mesh2, false);
   delete mesh2;

   pncmesh->GetConformingSharedStructures(*this);

   GenerateNCFaceInfo();

   last_operation = Mesh::DEREFINE;
   sequence++;

   UpdateNodes();

   return true;
}


void ParMesh::Rebalance()
{
   RebalanceImpl(NULL); // default SFC-based partition
}

void ParMesh::Rebalance(const Array<int> &partition)
{
   RebalanceImpl(&partition);
}

void ParMesh::RebalanceImpl(const Array<int> *partition)
{
   if (Conforming())
   {
      MFEM_ABORT("Load balancing is currently not supported for conforming"
                 " meshes.");
   }

   if (Nodes)
   {
      // check that Nodes use a parallel FE space, so we can call UpdateNodes()
      MFEM_VERIFY(dynamic_cast<ParFiniteElementSpace*>(Nodes->FESpace())
                  != NULL, "internal error");
   }

   DeleteFaceNbrData();

   pncmesh->Rebalance(partition);

   ParMesh* pmesh2 = new ParMesh(*pncmesh);
   pncmesh->OnMeshUpdated(pmesh2);

   attributes.Copy(pmesh2->attributes);
   bdr_attributes.Copy(pmesh2->bdr_attributes);

   // Copy attribute and bdr_attribute names
   attribute_sets.Copy(pmesh2->attribute_sets);
   bdr_attribute_sets.Copy(pmesh2->bdr_attribute_sets);

   Mesh::Swap(*pmesh2, false);
   delete pmesh2;

   pncmesh->GetConformingSharedStructures(*this);

   GenerateNCFaceInfo();

   last_operation = Mesh::REBALANCE;
   sequence++;

   UpdateNodes();
}

void ParMesh::RefineGroups(const DSTable &v_to_v, int *middle)
{
   // Refine groups after LocalRefinement in 2D (triangle meshes)

   MFEM_ASSERT(Dim == 2 && meshgen == 1, "internal error");

   Array<int> group_verts, group_edges;

   // To update the groups after a refinement, we observe that:
   // - every (new and old) vertex, edge and face belongs to exactly one group
   // - the refinement does not create new groups
   // - a new vertex appears only as the middle of a refined edge
   // - a face can be refined 2, 3 or 4 times producing new edges and faces

   int *I_group_svert, *J_group_svert;
   int *I_group_sedge, *J_group_sedge;

   I_group_svert = Memory<int>(GetNGroups()+1);
   I_group_sedge = Memory<int>(GetNGroups()+1);

   I_group_svert[0] = I_group_svert[1] = 0;
   I_group_sedge[0] = I_group_sedge[1] = 0;

   // overestimate the size of the J arrays
   J_group_svert = Memory<int>(group_svert.Size_of_connections() +
                               group_sedge.Size_of_connections());
   J_group_sedge = Memory<int>(2*group_sedge.Size_of_connections());

   for (int group = 0; group < GetNGroups()-1; group++)
   {
      // Get the group shared objects
      group_svert.GetRow(group, group_verts);
      group_sedge.GetRow(group, group_edges);

      // Check which edges have been refined
      for (int i = 0; i < group_sedge.RowSize(group); i++)
      {
         int *v = shared_edges[group_edges[i]]->GetVertices();
         const int ind = middle[v_to_v(v[0], v[1])];
         if (ind != -1)
         {
            // add a vertex
            group_verts.Append(svert_lvert.Append(ind)-1);
            // update the edges
            const int attr = shared_edges[group_edges[i]]->GetAttribute();
            shared_edges.Append(new Segment(v[1], ind, attr));
            group_edges.Append(sedge_ledge.Append(-1)-1);
            v[1] = ind;
         }
      }

      I_group_svert[group+1] = I_group_svert[group] + group_verts.Size();
      I_group_sedge[group+1] = I_group_sedge[group] + group_edges.Size();

      int *J;
      J = J_group_svert+I_group_svert[group];
      for (int i = 0; i < group_verts.Size(); i++)
      {
         J[i] = group_verts[i];
      }
      J = J_group_sedge+I_group_sedge[group];
      for (int i = 0; i < group_edges.Size(); i++)
      {
         J[i] = group_edges[i];
      }
   }

   FinalizeParTopo();

   group_svert.SetIJ(I_group_svert, J_group_svert);
   group_sedge.SetIJ(I_group_sedge, J_group_sedge);
}

void ParMesh::RefineGroups(int old_nv, const HashTable<Hashed2> &v_to_v)
{
   // Refine groups after LocalRefinement in 3D (tetrahedral meshes)

   MFEM_ASSERT(Dim == 3 && meshgen == 1, "internal error");

   Array<int> group_verts, group_edges, group_trias;

   // To update the groups after a refinement, we observe that:
   // - every (new and old) vertex, edge and face belongs to exactly one group
   // - the refinement does not create new groups
   // - a new vertex appears only as the middle of a refined edge
   // - a face can be refined multiple times producing new edges and faces

   Array<Segment *> sedge_stack;
   Array<Vert3> sface_stack;

   Array<int> I_group_svert, J_group_svert;
   Array<int> I_group_sedge, J_group_sedge;
   Array<int> I_group_stria, J_group_stria;

   I_group_svert.SetSize(GetNGroups());
   I_group_sedge.SetSize(GetNGroups());
   I_group_stria.SetSize(GetNGroups());

   I_group_svert[0] = 0;
   I_group_sedge[0] = 0;
   I_group_stria[0] = 0;

   for (int group = 0; group < GetNGroups()-1; group++)
   {
      // Get the group shared objects
      group_svert.GetRow(group, group_verts);
      group_sedge.GetRow(group, group_edges);
      group_stria.GetRow(group, group_trias);

      // Check which edges have been refined
      for (int i = 0; i < group_sedge.RowSize(group); i++)
      {
         int *v = shared_edges[group_edges[i]]->GetVertices();
         int ind = v_to_v.FindId(v[0], v[1]);
         if (ind == -1) { continue; }

         // This shared edge is refined: walk the whole refinement tree
         const int attr = shared_edges[group_edges[i]]->GetAttribute();
         do
         {
            ind += old_nv;
            // Add new shared vertex
            group_verts.Append(svert_lvert.Append(ind)-1);
            // Put the right sub-edge on top of the stack
            sedge_stack.Append(new Segment(ind, v[1], attr));
            // The left sub-edge replaces the original edge
            v[1] = ind;
            ind = v_to_v.FindId(v[0], ind);
         }
         while (ind != -1);
         // Process all edges in the edge stack
         do
         {
            Segment *se = sedge_stack.Last();
            v = se->GetVertices();
            ind = v_to_v.FindId(v[0], v[1]);
            if (ind == -1)
            {
               // The edge 'se' is not refined
               sedge_stack.DeleteLast();
               // Add new shared edge
               shared_edges.Append(se);
               group_edges.Append(sedge_ledge.Append(-1)-1);
            }
            else
            {
               // The edge 'se' is refined
               ind += old_nv;
               // Add new shared vertex
               group_verts.Append(svert_lvert.Append(ind)-1);
               // Put the left sub-edge on top of the stack
               sedge_stack.Append(new Segment(v[0], ind, attr));
               // The right sub-edge replaces the original edge
               v[0] = ind;
            }
         }
         while (sedge_stack.Size() > 0);
      }

      // Check which triangles have been refined
      for (int i = 0; i < group_stria.RowSize(group); i++)
      {
         int *v = shared_trias[group_trias[i]].v;
         int ind = v_to_v.FindId(v[0], v[1]);
         if (ind == -1) { continue; }

         // This shared face is refined: walk the whole refinement tree
         const int edge_attr = 1;
         do
         {
            ind += old_nv;
            // Add the refinement edge to the edge stack
            sedge_stack.Append(new Segment(v[2], ind, edge_attr));
            // Put the right sub-triangle on top of the face stack
            sface_stack.Append(Vert3(v[1], v[2], ind));
            // The left sub-triangle replaces the original one
            v[1] = v[0]; v[0] = v[2]; v[2] = ind;
            ind = v_to_v.FindId(v[0], v[1]);
         }
         while (ind != -1);
         // Process all faces (triangles) in the face stack
         do
         {
            Vert3 &st = sface_stack.Last();
            v = st.v;
            ind = v_to_v.FindId(v[0], v[1]);
            if (ind == -1)
            {
               // The triangle 'st' is not refined
               // Add new shared face
               shared_trias.Append(st);
               group_trias.Append(sface_lface.Append(-1)-1);
               sface_stack.DeleteLast();
            }
            else
            {
               // The triangle 'st' is refined
               ind += old_nv;
               // Add the refinement edge to the edge stack
               sedge_stack.Append(new Segment(v[2], ind, edge_attr));
               // Put the left sub-triangle on top of the face stack
               sface_stack.Append(Vert3(v[2], v[0], ind));
               // Note that the above Append() may invalidate 'v'
               v = sface_stack[sface_stack.Size()-2].v;
               // The right sub-triangle replaces the original one
               v[0] = v[1]; v[1] = v[2]; v[2] = ind;
            }
         }
         while (sface_stack.Size() > 0);
         // Process all edges in the edge stack (same code as above)
         do
         {
            Segment *se = sedge_stack.Last();
            v = se->GetVertices();
            ind = v_to_v.FindId(v[0], v[1]);
            if (ind == -1)
            {
               // The edge 'se' is not refined
               sedge_stack.DeleteLast();
               // Add new shared edge
               shared_edges.Append(se);
               group_edges.Append(sedge_ledge.Append(-1)-1);
            }
            else
            {
               // The edge 'se' is refined
               ind += old_nv;
               // Add new shared vertex
               group_verts.Append(svert_lvert.Append(ind)-1);
               // Put the left sub-edge on top of the stack
               sedge_stack.Append(new Segment(v[0], ind, edge_attr));
               // The right sub-edge replaces the original edge
               v[0] = ind;
            }
         }
         while (sedge_stack.Size() > 0);
      }

      I_group_svert[group+1] = I_group_svert[group] + group_verts.Size();
      I_group_sedge[group+1] = I_group_sedge[group] + group_edges.Size();
      I_group_stria[group+1] = I_group_stria[group] + group_trias.Size();

      J_group_svert.Append(group_verts);
      J_group_sedge.Append(group_edges);
      J_group_stria.Append(group_trias);
   }

   FinalizeParTopo();

   group_svert.SetIJ(I_group_svert, J_group_svert);
   group_sedge.SetIJ(I_group_sedge, J_group_sedge);
   group_stria.SetIJ(I_group_stria, J_group_stria);
   I_group_svert.LoseData(); J_group_svert.LoseData();
   I_group_sedge.LoseData(); J_group_sedge.LoseData();
   I_group_stria.LoseData(); J_group_stria.LoseData();
}

void ParMesh::UniformRefineGroups2D(int old_nv)
{
   Array<int> sverts, sedges;

   int *I_group_svert, *J_group_svert;
   int *I_group_sedge, *J_group_sedge;

   I_group_svert = Memory<int>(GetNGroups());
   I_group_sedge = Memory<int>(GetNGroups());

   I_group_svert[0] = 0;
   I_group_sedge[0] = 0;

   // compute the size of the J arrays
   J_group_svert = Memory<int>(group_svert.Size_of_connections() +
                               group_sedge.Size_of_connections());
   J_group_sedge = Memory<int>(2*group_sedge.Size_of_connections());

   for (int group = 0; group < GetNGroups()-1; group++)
   {
      // Get the group shared objects
      group_svert.GetRow(group, sverts);
      group_sedge.GetRow(group, sedges);

      // Process all the edges
      for (int i = 0; i < group_sedge.RowSize(group); i++)
      {
         int *v = shared_edges[sedges[i]]->GetVertices();
         const int ind = old_nv + sedge_ledge[sedges[i]];
         // add a vertex
         sverts.Append(svert_lvert.Append(ind)-1);
         // update the edges
         const int attr = shared_edges[sedges[i]]->GetAttribute();
         shared_edges.Append(new Segment(v[1], ind, attr));
         sedges.Append(sedge_ledge.Append(-1)-1);
         v[1] = ind;
      }

      I_group_svert[group+1] = I_group_svert[group] + sverts.Size();
      I_group_sedge[group+1] = I_group_sedge[group] + sedges.Size();

      sverts.CopyTo(J_group_svert + I_group_svert[group]);
      sedges.CopyTo(J_group_sedge + I_group_sedge[group]);
   }

   FinalizeParTopo();

   group_svert.SetIJ(I_group_svert, J_group_svert);
   group_sedge.SetIJ(I_group_sedge, J_group_sedge);
}

void ParMesh::UniformRefineGroups3D(int old_nv, int old_nedges,
                                    const DSTable &old_v_to_v,
                                    const STable3D &old_faces,
                                    Array<int> *f2qf)
{
   // f2qf can be NULL if all faces are quads or there are no quad faces

   Array<int> group_verts, group_edges, group_trias, group_quads;

   int *I_group_svert, *J_group_svert;
   int *I_group_sedge, *J_group_sedge;
   int *I_group_stria, *J_group_stria;
   int *I_group_squad, *J_group_squad;

   I_group_svert = Memory<int>(GetNGroups());
   I_group_sedge = Memory<int>(GetNGroups());
   I_group_stria = Memory<int>(GetNGroups());
   I_group_squad = Memory<int>(GetNGroups());

   I_group_svert[0] = 0;
   I_group_sedge[0] = 0;
   I_group_stria[0] = 0;
   I_group_squad[0] = 0;

   // compute the size of the J arrays
   J_group_svert = Memory<int>(group_svert.Size_of_connections() +
                               group_sedge.Size_of_connections() +
                               group_squad.Size_of_connections());
   J_group_sedge = Memory<int>(2*group_sedge.Size_of_connections() +
                               3*group_stria.Size_of_connections() +
                               4*group_squad.Size_of_connections());
   J_group_stria = Memory<int>(4*group_stria.Size_of_connections());
   J_group_squad = Memory<int>(4*group_squad.Size_of_connections());

   const int oface = old_nv + old_nedges;

   for (int group = 0; group < GetNGroups()-1; group++)
   {
      // Get the group shared objects
      group_svert.GetRow(group, group_verts);
      group_sedge.GetRow(group, group_edges);
      group_stria.GetRow(group, group_trias);
      group_squad.GetRow(group, group_quads);

      // Process the edges that have been refined
      for (int i = 0; i < group_sedge.RowSize(group); i++)
      {
         int *v = shared_edges[group_edges[i]]->GetVertices();
         const int ind = old_nv + old_v_to_v(v[0], v[1]);
         // add a vertex
         group_verts.Append(svert_lvert.Append(ind)-1);
         // update the edges
         const int attr = shared_edges[group_edges[i]]->GetAttribute();
         shared_edges.Append(new Segment(v[1], ind, attr));
         group_edges.Append(sedge_ledge.Append(-1)-1);
         v[1] = ind; // v[0] remains the same
      }

      // Process the triangles that have been refined
      for (int i = 0; i < group_stria.RowSize(group); i++)
      {
         int m[3];
         const int stria = group_trias[i];
         int *v = shared_trias[stria].v;
         // add the refinement edges
         m[0] = old_nv + old_v_to_v(v[0], v[1]);
         m[1] = old_nv + old_v_to_v(v[1], v[2]);
         m[2] = old_nv + old_v_to_v(v[2], v[0]);
         const int edge_attr = 1;
         shared_edges.Append(new Segment(m[0], m[1], edge_attr));
         group_edges.Append(sedge_ledge.Append(-1)-1);
         shared_edges.Append(new Segment(m[1], m[2], edge_attr));
         group_edges.Append(sedge_ledge.Append(-1)-1);
         shared_edges.Append(new Segment(m[0], m[2], edge_attr));
         group_edges.Append(sedge_ledge.Append(-1)-1);
         // update faces
         const int nst = shared_trias.Size();
         shared_trias.SetSize(nst+3);
         // The above SetSize() may invalidate 'v'
         v = shared_trias[stria].v;
         shared_trias[nst+0].Set(m[1],m[2],m[0]);
         shared_trias[nst+1].Set(m[0],v[1],m[1]);
         shared_trias[nst+2].Set(m[2],m[1],v[2]);
         v[1] = m[0]; v[2] = m[2]; // v[0] remains the same
         group_trias.Append(nst+0);
         group_trias.Append(nst+1);
         group_trias.Append(nst+2);
         // sface_lface is set later
      }

      // Process the quads that have been refined
      for (int i = 0; i < group_squad.RowSize(group); i++)
      {
         int m[5];
         const int squad = group_quads[i];
         int *v = shared_quads[squad].v;
         const int olf = old_faces(v[0], v[1], v[2], v[3]);
         // f2qf can be NULL if all faces are quads
         m[0] = oface + (f2qf ? (*f2qf)[olf] : olf);
         // add a vertex
         group_verts.Append(svert_lvert.Append(m[0])-1);
         // add the refinement edges
         m[1] = old_nv + old_v_to_v(v[0], v[1]);
         m[2] = old_nv + old_v_to_v(v[1], v[2]);
         m[3] = old_nv + old_v_to_v(v[2], v[3]);
         m[4] = old_nv + old_v_to_v(v[3], v[0]);
         const int edge_attr = 1;
         shared_edges.Append(new Segment(m[1], m[0], edge_attr));
         group_edges.Append(sedge_ledge.Append(-1)-1);
         shared_edges.Append(new Segment(m[2], m[0], edge_attr));
         group_edges.Append(sedge_ledge.Append(-1)-1);
         shared_edges.Append(new Segment(m[3], m[0], edge_attr));
         group_edges.Append(sedge_ledge.Append(-1)-1);
         shared_edges.Append(new Segment(m[4], m[0], edge_attr));
         group_edges.Append(sedge_ledge.Append(-1)-1);
         // update faces
         const int nsq = shared_quads.Size();
         shared_quads.SetSize(nsq+3);
         // The above SetSize() may invalidate 'v'
         v = shared_quads[squad].v;
         shared_quads[nsq+0].Set(m[1],v[1],m[2],m[0]);
         shared_quads[nsq+1].Set(m[0],m[2],v[2],m[3]);
         shared_quads[nsq+2].Set(m[4],m[0],m[3],v[3]);
         v[1] = m[1]; v[2] = m[0]; v[3] = m[4]; // v[0] remains the same
         group_quads.Append(nsq+0);
         group_quads.Append(nsq+1);
         group_quads.Append(nsq+2);
         // sface_lface is set later
      }

      I_group_svert[group+1] = I_group_svert[group] + group_verts.Size();
      I_group_sedge[group+1] = I_group_sedge[group] + group_edges.Size();
      I_group_stria[group+1] = I_group_stria[group] + group_trias.Size();
      I_group_squad[group+1] = I_group_squad[group] + group_quads.Size();

      group_verts.CopyTo(J_group_svert + I_group_svert[group]);
      group_edges.CopyTo(J_group_sedge + I_group_sedge[group]);
      group_trias.CopyTo(J_group_stria + I_group_stria[group]);
      group_quads.CopyTo(J_group_squad + I_group_squad[group]);
   }

   FinalizeParTopo();

   group_svert.SetIJ(I_group_svert, J_group_svert);
   group_sedge.SetIJ(I_group_sedge, J_group_sedge);
   group_stria.SetIJ(I_group_stria, J_group_stria);
   group_squad.SetIJ(I_group_squad, J_group_squad);
}

void ParMesh::UniformRefinement2D()
{
   DeleteFaceNbrData();

   const int old_nv = NumOfVertices;

   // call Mesh::UniformRefinement2D so that it won't update the nodes
   {
      const bool update_nodes = false;
      Mesh::UniformRefinement2D_base(update_nodes);
   }

   // update the groups
   UniformRefineGroups2D(old_nv);

   UpdateNodes();

#ifdef MFEM_DEBUG
   // If there are no Nodes, the orientation is checked in the call to
   // UniformRefinement2D_base() above.
   if (Nodes) { CheckElementOrientation(false); }
#endif
}

void ParMesh::UniformRefinement3D()
{
   DeleteFaceNbrData();

   const int old_nv = NumOfVertices;
   const int old_nedges = NumOfEdges;

   DSTable v_to_v(NumOfVertices);
   GetVertexToVertexTable(v_to_v);
   auto faces_tbl = std::unique_ptr<STable3D>(GetFacesTable());

   // call Mesh::UniformRefinement3D_base so that it won't update the nodes
   Array<int> f2qf;
   {
      const bool update_nodes = false;
      UniformRefinement3D_base(&f2qf, &v_to_v, update_nodes);
      // Note: for meshes that have triangular faces, v_to_v is modified by the
      //       above call to return different edge indices - this is used when
      //       updating the groups. This is needed by ReorientTetMesh().
   }

   // update the groups
   UniformRefineGroups3D(old_nv, old_nedges, v_to_v, *faces_tbl,
                         f2qf.Size() ? &f2qf : NULL);

   UpdateNodes();
}

void ParMesh::NURBSUniformRefinement(int rf, real_t tol)
{
   if (MyRank == 0)
   {
      mfem::out << "\nParMesh::NURBSUniformRefinement : Not supported yet!\n";
   }
}

void ParMesh::NURBSUniformRefinement(const Array<int> &rf, real_t tol)
{
   if (MyRank == 0)
   {
      mfem::out << "\nParMesh::NURBSUniformRefinement : Not supported yet!\n";
   }
}

void ParMesh::PrintXG(std::ostream &os) const
{
   MFEM_ASSERT(Dim == spaceDim, "2D manifolds not supported");
   if (Dim == 3 && meshgen == 1)
   {
      int i, j, nv;
      const int *ind;

      os << "NETGEN_Neutral_Format\n";
      // print the vertices
      os << NumOfVertices << '\n';
      for (i = 0; i < NumOfVertices; i++)
      {
         for (j = 0; j < Dim; j++)
         {
            os << " " << vertices[i](j);
         }
         os << '\n';
      }

      // print the elements
      os << NumOfElements << '\n';
      for (i = 0; i < NumOfElements; i++)
      {
         nv = elements[i]->GetNVertices();
         ind = elements[i]->GetVertices();
         os << elements[i]->GetAttribute();
         for (j = 0; j < nv; j++)
         {
            os << " " << ind[j]+1;
         }
         os << '\n';
      }

      // print the boundary + shared faces information
      os << NumOfBdrElements + sface_lface.Size() << '\n';
      // boundary
      for (i = 0; i < NumOfBdrElements; i++)
      {
         nv = boundary[i]->GetNVertices();
         ind = boundary[i]->GetVertices();
         os << boundary[i]->GetAttribute();
         for (j = 0; j < nv; j++)
         {
            os << " " << ind[j]+1;
         }
         os << '\n';
      }
      // shared faces
      const int sf_attr =
         MyRank + 1 + (bdr_attributes.Size() > 0 ? bdr_attributes.Max() : 0);
      for (i = 0; i < shared_trias.Size(); i++)
      {
         ind = shared_trias[i].v;
         os << sf_attr;
         for (j = 0; j < 3; j++)
         {
            os << ' ' << ind[j]+1;
         }
         os << '\n';
      }
      // There are no quad shared faces
   }

   if (Dim == 3 && meshgen == 2)
   {
      int i, j, nv;
      const int *ind;

      os << "TrueGrid\n"
         << "1 " << NumOfVertices << " " << NumOfElements
         << " 0 0 0 0 0 0 0\n"
         << "0 0 0 1 0 0 0 0 0 0 0\n"
         << "0 0 " << NumOfBdrElements+sface_lface.Size()
         << " 0 0 0 0 0 0 0 0 0 0 0 0 0\n"
         << "0.0 0.0 0.0 0 0 0.0 0.0 0 0.0\n"
         << "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n";

      // print the vertices
      for (i = 0; i < NumOfVertices; i++)
      {
         os << i+1 << " 0.0 " << vertices[i](0) << " " << vertices[i](1)
            << " " << vertices[i](2) << " 0.0\n";
      }

      // print the elements
      for (i = 0; i < NumOfElements; i++)
      {
         nv = elements[i]->GetNVertices();
         ind = elements[i]->GetVertices();
         os << i+1 << " " << elements[i]->GetAttribute();
         for (j = 0; j < nv; j++)
         {
            os << " " << ind[j]+1;
         }
         os << '\n';
      }

      // print the boundary information
      for (i = 0; i < NumOfBdrElements; i++)
      {
         nv = boundary[i]->GetNVertices();
         ind = boundary[i]->GetVertices();
         os << boundary[i]->GetAttribute();
         for (j = 0; j < nv; j++)
         {
            os << " " << ind[j]+1;
         }
         os << " 1.0 1.0 1.0 1.0\n";
      }

      // print the shared faces information
      const int sf_attr =
         MyRank + 1 + (bdr_attributes.Size() > 0 ? bdr_attributes.Max() : 0);
      // There are no shared triangle faces
      for (i = 0; i < shared_quads.Size(); i++)
      {
         ind = shared_quads[i].v;
         os << sf_attr;
         for (j = 0; j < 4; j++)
         {
            os << ' ' << ind[j]+1;
         }
         os << " 1.0 1.0 1.0 1.0\n";
      }
   }

   if (Dim == 2)
   {
      int i, j, attr;
      Array<int> v;

      os << "areamesh2\n\n";

      // print the boundary + shared edges information
      os << NumOfBdrElements + shared_edges.Size() << '\n';
      // boundary
      for (i = 0; i < NumOfBdrElements; i++)
      {
         attr = boundary[i]->GetAttribute();
         boundary[i]->GetVertices(v);
         os << attr << "     ";
         for (j = 0; j < v.Size(); j++)
         {
            os << v[j] + 1 << "   ";
         }
         os << '\n';
      }
      // shared edges
      for (i = 0; i < shared_edges.Size(); i++)
      {
         attr = shared_edges[i]->GetAttribute();
         shared_edges[i]->GetVertices(v);
         os << attr << "     ";
         for (j = 0; j < v.Size(); j++)
         {
            os << v[j] + 1 << "   ";
         }
         os << '\n';
      }

      // print the elements
      os << NumOfElements << '\n';
      for (i = 0; i < NumOfElements; i++)
      {
         attr = elements[i]->GetAttribute();
         elements[i]->GetVertices(v);

         os << attr << "   ";
         if ((j = GetElementType(i)) == Element::TRIANGLE)
         {
            os << 3 << "   ";
         }
         else if (j == Element::QUADRILATERAL)
         {
            os << 4 << "   ";
         }
         else if (j == Element::SEGMENT)
         {
            os << 2 << "   ";
         }
         for (j = 0; j < v.Size(); j++)
         {
            os << v[j] + 1 << "  ";
         }
         os << '\n';
      }

      // print the vertices
      os << NumOfVertices << '\n';
      for (i = 0; i < NumOfVertices; i++)
      {
         for (j = 0; j < Dim; j++)
         {
            os << vertices[i](j) << " ";
         }
         os << '\n';
      }
   }
   os.flush();
}

bool ParMesh::WantSkipSharedMaster(const NCMesh::Master &master) const
{
   // In 2D, this is a workaround for a CPU boundary rendering artifact. We need
   // to skip a shared master edge if one of its slaves has the same rank.

   const NCMesh::NCList &list = pncmesh->GetEdgeList();
   for (int i = master.slaves_begin; i < master.slaves_end; i++)
   {
      if (!pncmesh->IsGhost(1, list.slaves[i].index)) { return true; }
   }
   return false;
}

void ParMesh::Print(std::ostream &os, const std::string &comments) const
{
   int shared_bdr_attr;
   Array<int> nc_shared_faces;

   if (NURBSext)
   {
      Printer(os, "", comments); // does not print shared boundary
      return;
   }

   const Array<int>* s2l_face;
   if (!pncmesh)
   {
      s2l_face = ((Dim == 1) ? &svert_lvert :
                  ((Dim == 2) ? &sedge_ledge : &sface_lface));
   }
   else
   {
      s2l_face = &nc_shared_faces;
      if (Dim >= 2)
      {
         // get a list of all shared non-ghost faces
         const NCMesh::NCList& sfaces =
            (Dim == 3) ? pncmesh->GetSharedFaces() : pncmesh->GetSharedEdges();
         const int nfaces = GetNumFaces();
         for (int i = 0; i < sfaces.conforming.Size(); i++)
         {
            int index = sfaces.conforming[i].index;
            if (index < nfaces) { nc_shared_faces.Append(index); }
         }
         for (int i = 0; i < sfaces.masters.Size(); i++)
         {
            if (Dim == 2 && WantSkipSharedMaster(sfaces.masters[i])) { continue; }
            int index = sfaces.masters[i].index;
            if (index < nfaces) { nc_shared_faces.Append(index); }
         }
         for (int i = 0; i < sfaces.slaves.Size(); i++)
         {
            int index = sfaces.slaves[i].index;
            if (index < nfaces) { nc_shared_faces.Append(index); }
         }
      }
   }

   const bool set_names = attribute_sets.SetsExist() ||
                          bdr_attribute_sets.SetsExist();

   os << (!set_names ? "MFEM mesh v1.0\n" : "MFEM mesh v1.3\n");

   if (!comments.empty()) { os << '\n' << comments << '\n'; }

   // optional
   os <<
      "\n#\n# MFEM Geometry Types (see fem/geom.hpp):\n#\n"
      "# POINT       = 0\n"
      "# SEGMENT     = 1\n"
      "# TRIANGLE    = 2\n"
      "# SQUARE      = 3\n"
      "# TETRAHEDRON = 4\n"
      "# CUBE        = 5\n"
      "# PRISM       = 6\n"
      "#\n";

   os << "\ndimension\n" << Dim
      << "\n\nelements\n" << NumOfElements << '\n';
   for (int i = 0; i < NumOfElements; i++)
   {
      PrintElement(elements[i], os);
   }

   if (set_names)
   {
      os << "\nattribute_sets\n";
      attribute_sets.Print(os);
   }

   int num_bdr_elems = NumOfBdrElements;
   if (print_shared && Dim > 1)
   {
      num_bdr_elems += s2l_face->Size();
   }
   os << "\nboundary\n" << num_bdr_elems << '\n';
   for (int i = 0; i < NumOfBdrElements; i++)
   {
      PrintElement(boundary[i], os);
   }

   if (print_shared && Dim > 1)
   {
      if (bdr_attributes.Size())
      {
         shared_bdr_attr = bdr_attributes.Max() + MyRank + 1;
      }
      else
      {
         shared_bdr_attr = MyRank + 1;
      }
      for (int i = 0; i < s2l_face->Size(); i++)
      {
         // Modify the attributes of the faces (not used otherwise?)
         faces[(*s2l_face)[i]]->SetAttribute(shared_bdr_attr);
         PrintElement(faces[(*s2l_face)[i]], os);
      }
   }

   if (set_names)
   {
      os << "\nbdr_attribute_sets\n";
      bdr_attribute_sets.Print(os);
   }

   os << "\nvertices\n" << NumOfVertices << '\n';
   if (Nodes == NULL)
   {
      os << spaceDim << '\n';
      for (int i = 0; i < NumOfVertices; i++)
      {
         os << vertices[i](0);
         for (int j = 1; j < spaceDim; j++)
         {
            os << ' ' << vertices[i](j);
         }
         os << '\n';
      }
      os.flush();
   }
   else
   {
      os << "\nnodes\n";
      Nodes->Save(os);
   }

   if (set_names)
   {
      os << "\nmfem_mesh_end" << endl;
   }
}

void ParMesh::Save(const std::string &fname, int precision) const
{
   ostringstream fname_with_suffix;
   fname_with_suffix << fname << "." << setfill('0') << setw(6) << MyRank;
   ofstream ofs(fname_with_suffix.str().c_str());
   ofs.precision(precision);
   Print(ofs);
}

#ifdef MFEM_USE_ADIOS2
void ParMesh::Print(adios2stream &os) const
{
   Mesh::Print(os);
}
#endif

static void dump_element(const Element* elem, Array<int> &data)
{
   data.Append(elem->GetGeometryType());

   int nv = elem->GetNVertices();
   const int *v = elem->GetVertices();
   for (int i = 0; i < nv; i++)
   {
      data.Append(v[i]);
   }
}

void ParMesh::PrintAsOne(std::ostream &os, const std::string &comments) const
{
   int i, j, k, p, nv_ne[2], &nv = nv_ne[0], &ne = nv_ne[1], vc;
   const int *v;
   MPI_Status status;
   Array<real_t> vert;
   Array<int> ints;

   if (MyRank == 0)
   {
      os << "MFEM mesh v1.0\n";

      if (!comments.empty()) { os << '\n' << comments << '\n'; }

      // optional
      os <<
         "\n#\n# MFEM Geometry Types (see fem/geom.hpp):\n#\n"
         "# POINT       = 0\n"
         "# SEGMENT     = 1\n"
         "# TRIANGLE    = 2\n"
         "# SQUARE      = 3\n"
         "# TETRAHEDRON = 4\n"
         "# CUBE        = 5\n"
         "# PRISM       = 6\n"
         "#\n";

      os << "\ndimension\n" << Dim;
   }

   nv = NumOfElements;
   MPI_Reduce(&nv, &ne, 1, MPI_INT, MPI_SUM, 0, MyComm);
   if (MyRank == 0)
   {
      os << "\n\nelements\n" << ne << '\n';
      for (i = 0; i < NumOfElements; i++)
      {
         // processor number + 1 as attribute and geometry type
         os << 1 << ' ' << elements[i]->GetGeometryType();
         // vertices
         nv = elements[i]->GetNVertices();
         v  = elements[i]->GetVertices();
         for (j = 0; j < nv; j++)
         {
            os << ' ' << v[j];
         }
         os << '\n';
      }
      vc = NumOfVertices;
      for (p = 1; p < NRanks; p++)
      {
         MPI_Recv(nv_ne, 2, MPI_INT, p, 444, MyComm, &status);
         ints.SetSize(ne);
         if (ne)
         {
            MPI_Recv(&ints[0], ne, MPI_INT, p, 445, MyComm, &status);
         }
         for (i = 0; i < ne; )
         {
            // processor number + 1 as attribute and geometry type
            os << p+1 << ' ' << ints[i];
            // vertices
            k = Geometries.GetVertices(ints[i++])->GetNPoints();
            for (j = 0; j < k; j++)
            {
               os << ' ' << vc + ints[i++];
            }
            os << '\n';
         }
         vc += nv;
      }
   }
   else
   {
      // for each element send its geometry type and its vertices
      ne = 0;
      for (i = 0; i < NumOfElements; i++)
      {
         ne += 1 + elements[i]->GetNVertices();
      }
      nv = NumOfVertices;
      MPI_Send(nv_ne, 2, MPI_INT, 0, 444, MyComm);

      ints.Reserve(ne);
      ints.SetSize(0);
      for (i = 0; i < NumOfElements; i++)
      {
         dump_element(elements[i], ints);
      }
      MFEM_ASSERT(ints.Size() == ne, "");
      if (ne)
      {
         MPI_Send(&ints[0], ne, MPI_INT, 0, 445, MyComm);
      }
   }

   // boundary + shared boundary
   ne = NumOfBdrElements;
   if (!pncmesh)
   {
      ne += GetNSharedFaces();
   }
   else if (Dim > 1)
   {
      const NCMesh::NCList &list = pncmesh->GetSharedList(Dim - 1);
      ne += list.conforming.Size() + list.masters.Size() + list.slaves.Size();
      // In addition to the number returned by GetNSharedFaces(), include the
      // the master shared faces as well.
   }
   ints.Reserve(ne * (1 + 2*(Dim-1))); // just an upper bound
   ints.SetSize(0);

   // for each boundary and shared boundary element send its geometry type
   // and its vertices
   ne = 0;
   for (i = j = 0; i < NumOfBdrElements; i++)
   {
      dump_element(boundary[i], ints); ne++;
   }
   if (!pncmesh)
   {
      switch (Dim)
      {
         case 1:
            for (i = 0; i < svert_lvert.Size(); i++)
            {
               ints.Append(Geometry::POINT);
               ints.Append(svert_lvert[i]);
               ne++;
            }
            break;

         case 2:
            for (i = 0; i < shared_edges.Size(); i++)
            {
               dump_element(shared_edges[i], ints); ne++;
            }
            break;

         case 3:
            for (i = 0; i < shared_trias.Size(); i++)
            {
               ints.Append(Geometry::TRIANGLE);
               ints.Append(shared_trias[i].v, 3);
               ne++;
            }
            for (i = 0; i < shared_quads.Size(); i++)
            {
               ints.Append(Geometry::SQUARE);
               ints.Append(shared_quads[i].v, 4);
               ne++;
            }
            break;

         default:
            MFEM_ABORT("invalid dimension: " << Dim);
      }
   }
   else if (Dim > 1)
   {
      const NCMesh::NCList &list = pncmesh->GetSharedList(Dim - 1);
      const int nfaces = GetNumFaces();
      for (i = 0; i < list.conforming.Size(); i++)
      {
         int index = list.conforming[i].index;
         if (index < nfaces) { dump_element(faces[index], ints); ne++; }
      }
      for (i = 0; i < list.masters.Size(); i++)
      {
         int index = list.masters[i].index;
         if (index < nfaces) { dump_element(faces[index], ints); ne++; }
      }
      for (i = 0; i < list.slaves.Size(); i++)
      {
         int index = list.slaves[i].index;
         if (index < nfaces) { dump_element(faces[index], ints); ne++; }
      }
   }

   MPI_Reduce(&ne, &k, 1, MPI_INT, MPI_SUM, 0, MyComm);
   if (MyRank == 0)
   {
      os << "\nboundary\n" << k << '\n';
      vc = 0;
      for (p = 0; p < NRanks; p++)
      {
         if (p)
         {
            MPI_Recv(nv_ne, 2, MPI_INT, p, 446, MyComm, &status);
            ints.SetSize(ne);
            if (ne)
            {
               MPI_Recv(ints.GetData(), ne, MPI_INT, p, 447, MyComm, &status);
            }
         }
         else
         {
            ne = ints.Size();
            nv = NumOfVertices;
         }
         for (i = 0; i < ne; )
         {
            // processor number + 1 as bdr. attr. and bdr. geometry type
            os << p+1 << ' ' << ints[i];
            k = Geometries.NumVerts[ints[i++]];
            // vertices
            for (j = 0; j < k; j++)
            {
               os << ' ' << vc + ints[i++];
            }
            os << '\n';
         }
         vc += nv;
      }
   }
   else
   {
      nv = NumOfVertices;
      ne = ints.Size();
      MPI_Send(nv_ne, 2, MPI_INT, 0, 446, MyComm);
      if (ne)
      {
         MPI_Send(ints.GetData(), ne, MPI_INT, 0, 447, MyComm);
      }
   }

   // vertices / nodes
   MPI_Reduce(&NumOfVertices, &nv, 1, MPI_INT, MPI_SUM, 0, MyComm);
   if (MyRank == 0)
   {
      os << "\nvertices\n" << nv << '\n';
   }
   if (Nodes == NULL)
   {
      if (MyRank == 0)
      {
         os << spaceDim << '\n';
         for (i = 0; i < NumOfVertices; i++)
         {
            os << vertices[i](0);
            for (j = 1; j < spaceDim; j++)
            {
               os << ' ' << vertices[i](j);
            }
            os << '\n';
         }
         for (p = 1; p < NRanks; p++)
         {
            MPI_Recv(&nv, 1, MPI_INT, p, 448, MyComm, &status);
            vert.SetSize(nv*spaceDim);
            if (nv)
            {
               MPI_Recv(&vert[0], nv*spaceDim, MPITypeMap<real_t>::mpi_type, p, 449, MyComm,
                        &status);
            }
            for (i = 0; i < nv; i++)
            {
               os << vert[i*spaceDim];
               for (j = 1; j < spaceDim; j++)
               {
                  os << ' ' << vert[i*spaceDim+j];
               }
               os << '\n';
            }
         }
         os.flush();
      }
      else
      {
         MPI_Send(&NumOfVertices, 1, MPI_INT, 0, 448, MyComm);
         vert.SetSize(NumOfVertices*spaceDim);
         for (i = 0; i < NumOfVertices; i++)
         {
            for (j = 0; j < spaceDim; j++)
            {
               vert[i*spaceDim+j] = vertices[i](j);
            }
         }
         if (NumOfVertices)
         {
            MPI_Send(&vert[0], NumOfVertices*spaceDim, MPITypeMap<real_t>::mpi_type, 0, 449,
                     MyComm);
         }
      }
   }
   else
   {
      if (MyRank == 0)
      {
         os << "\nnodes\n";
      }
      ParGridFunction *pnodes = dynamic_cast<ParGridFunction *>(Nodes);
      if (pnodes)
      {
         pnodes->SaveAsOne(os);
      }
      else
      {
         ParFiniteElementSpace *pfes =
            dynamic_cast<ParFiniteElementSpace *>(Nodes->FESpace());
         if (pfes)
         {
            // create a wrapper ParGridFunction
            ParGridFunction ParNodes(pfes, Nodes);
            ParNodes.SaveAsOne(os);
         }
         else
         {
            mfem_error("ParMesh::PrintAsOne : Nodes have no parallel info!");
         }
      }
   }
}

void ParMesh::PrintAsSerial(std::ostream &os, const std::string &comments) const
{
   int save_rank = 0;
   Mesh serialmesh = GetSerialMesh(save_rank);
   if (MyRank == save_rank)
   {
      serialmesh.Printer(os, "", comments);
   }
   MPI_Barrier(MyComm);
}

Mesh ParMesh::GetSerialMesh(int save_rank) const
{
   if (pncmesh || NURBSext)
   {
      MFEM_ABORT("Nonconforming meshes and NURBS meshes are not yet supported.");
   }

   // Define linear H1 space for vertex numbering
   H1_FECollection fec_linear(1, Dim);
   ParMesh *pm = const_cast<ParMesh *>(this);
   ParFiniteElementSpace pfespace_linear(pm, &fec_linear);

   long long ne_glob_l = GetGlobalNE(); // needs to be called by all ranks
   MFEM_VERIFY(int(ne_glob_l) == ne_glob_l,
               "overflow in the number of elements!");
   int ne_glob = (save_rank == MyRank) ? int(ne_glob_l) : 0;

   long long nvertices = pfespace_linear.GetTrueVSize();
   long long nvertices_glob_l = 0;
   MPI_Reduce(&nvertices, &nvertices_glob_l, 1, MPI_LONG_LONG, MPI_SUM,
              save_rank, MyComm);
   int nvertices_glob = int(nvertices_glob_l);
   MFEM_VERIFY(nvertices_glob == nvertices_glob_l,
               "overflow in the number of vertices!");

   long long nbe = NumOfBdrElements;
   long long nbe_glob_l = 0;
   MPI_Reduce(&nbe, &nbe_glob_l, 1, MPI_LONG_LONG, MPI_SUM, save_rank, MyComm);
   int nbe_glob = int(nbe_glob_l);
   MFEM_VERIFY(nbe_glob == nbe_glob_l,
               "overflow in the number of boundary elements!");

   // On ranks other than save_rank, the *_glob variables are 0, so the serial
   // mesh is empty.
   Mesh serialmesh(Dim, nvertices_glob, ne_glob, nbe_glob, spaceDim);

   int n_send_recv;
   MPI_Status status;
   Array<real_t> vert;
   Array<int> ints, dofs;

   // First set the connectivity of serial mesh using the True Dofs from
   // the linear H1 space.
   if (MyRank == save_rank)
   {
      for (int e = 0; e < NumOfElements; e++)
      {
         const int attr = elements[e]->GetAttribute();
         const int geom_type = elements[e]->GetGeometryType();
         pfespace_linear.GetElementDofs(e, dofs);
         for (int j = 0; j < dofs.Size(); j++)
         {
            dofs[j] = pfespace_linear.GetGlobalTDofNumber(dofs[j]);
         }
         Element *elem = serialmesh.NewElement(geom_type);
         elem->SetAttribute(attr);
         elem->SetVertices(dofs);
         serialmesh.AddElement(elem);
      }

      for (int p = 0; p < NRanks; p++)
      {
         if (p == save_rank) { continue; }
         MPI_Recv(&n_send_recv, 1, MPI_INT, p, 444, MyComm, &status);
         ints.SetSize(n_send_recv);
         if (n_send_recv)
         {
            MPI_Recv(&ints[0], n_send_recv, MPI_INT, p, 445, MyComm, &status);
         }
         for (int i = 0; i < n_send_recv; )
         {
            int attr = ints[i++];
            int geom_type = ints[i++];
            Element *elem = serialmesh.NewElement(geom_type);
            elem->SetAttribute(attr);
            elem->SetVertices(&ints[i]); i += Geometry::NumVerts[geom_type];
            serialmesh.AddElement(elem);
         }
      }
   }
   else
   {
      n_send_recv = 0;
      for (int e = 0; e < NumOfElements; e++)
      {
         n_send_recv += 2 + elements[e]->GetNVertices();
      }
      MPI_Send(&n_send_recv, 1, MPI_INT, save_rank, 444, MyComm);
      ints.Reserve(n_send_recv);
      ints.SetSize(0);
      for (int e = 0; e < NumOfElements; e++)
      {
         const int attr = elements[e]->GetAttribute();
         const int geom_type = elements[e]->GetGeometryType();
         ints.Append(attr);
         ints.Append(geom_type);
         pfespace_linear.GetElementDofs(e, dofs);
         for (int j = 0; j < dofs.Size(); j++)
         {
            ints.Append(pfespace_linear.GetGlobalTDofNumber(dofs[j]));
         }
      }
      if (n_send_recv)
      {
         MPI_Send(&ints[0], n_send_recv, MPI_INT, save_rank, 445, MyComm);
      }
   }

   // Write out boundary elements
   if (MyRank == save_rank)
   {
      for (int e = 0; e < NumOfBdrElements; e++)
      {
         const int attr = boundary[e]->GetAttribute();
         const int geom_type = boundary[e]->GetGeometryType();
         pfespace_linear.GetBdrElementDofs(e, dofs);
         for (int j = 0; j < dofs.Size(); j++)
         {
            dofs[j] = pfespace_linear.GetGlobalTDofNumber(dofs[j]);
         }
         Element *elem = serialmesh.NewElement(geom_type);
         elem->SetAttribute(attr);
         elem->SetVertices(dofs);
         serialmesh.AddBdrElement(elem);
      }

      for (int p = 0; p < NRanks; p++)
      {
         if (p == save_rank) { continue; }
         MPI_Recv(&n_send_recv, 1, MPI_INT, p, 446, MyComm, &status);
         ints.SetSize(n_send_recv);
         if (n_send_recv)
         {
            MPI_Recv(&ints[0], n_send_recv, MPI_INT, p, 447, MyComm, &status);
         }
         for (int i = 0; i < n_send_recv; )
         {
            int attr = ints[i++];
            int geom_type = ints[i++];
            Element *elem = serialmesh.NewElement(geom_type);
            elem->SetAttribute(attr);
            elem->SetVertices(&ints[i]); i += Geometry::NumVerts[geom_type];
            serialmesh.AddBdrElement(elem);
         }
      }
   } // MyRank == save_rank
   else
   {
      n_send_recv = 0;
      for (int e = 0; e < NumOfBdrElements; e++)
      {
         n_send_recv += 2 + GetBdrElement(e)->GetNVertices();
      }
      MPI_Send(&n_send_recv, 1, MPI_INT, save_rank, 446, MyComm);
      ints.Reserve(n_send_recv);
      ints.SetSize(0);
      for (int e = 0; e < NumOfBdrElements; e++)
      {
         const int attr = boundary[e]->GetAttribute();
         const int geom_type = boundary[e]->GetGeometryType();
         ints.Append(attr);
         ints.Append(geom_type);
         pfespace_linear.GetBdrElementDofs(e, dofs);
         for (int j = 0; j < dofs.Size(); j++)
         {
            ints.Append(pfespace_linear.GetGlobalTDofNumber(dofs[j]));
         }
      }
      if (n_send_recv)
      {
         MPI_Send(&ints[0], n_send_recv, MPI_INT, save_rank, 447, MyComm);
      }
   } // MyRank != save_rank

   if (MyRank == save_rank)
   {
      for (int v = 0; v < nvertices_glob; v++)
      {
         serialmesh.AddVertex(0.0); // all other coordinates are 0 by default
      }
      serialmesh.FinalizeTopology();
   }

   // From each processor, we send element-wise vertex/dof locations and
   // overwrite the vertex/dof locations of the serial mesh.
   if (MyRank == save_rank && Nodes)
   {
      FiniteElementSpace *fespace_serial = NULL;
      // Duplicate the FE collection to make sure the serial mesh is completely
      // independent of the parallel mesh:
      auto fec_serial = FiniteElementCollection::New(
                           GetNodalFESpace()->FEColl()->Name());
      fespace_serial = new FiniteElementSpace(&serialmesh,
                                              fec_serial,
                                              spaceDim,
                                              GetNodalFESpace()->GetOrdering());
      serialmesh.SetNodalFESpace(fespace_serial);
      serialmesh.GetNodes()->MakeOwner(fec_serial);
      // The serial mesh owns its Nodes and they, in turn, own fec_serial and
      // fespace_serial.
   }

   int elem_count = 0; // To keep track of element count in serial mesh
   if (MyRank == save_rank)
   {
      Vector nodeloc;
      Array<int> ints_serial;
      for (int e = 0; e < NumOfElements; e++)
      {
         if (Nodes)
         {
            Nodes->GetElementDofValues(e, nodeloc);
            serialmesh.GetNodalFESpace()->GetElementVDofs(elem_count++, dofs);
            serialmesh.GetNodes()->SetSubVector(dofs, nodeloc);
         }
         else
         {
            GetElementVertices(e, ints);
            serialmesh.GetElementVertices(elem_count++, ints_serial);
            for (int i = 0; i < ints.Size(); i++)
            {
               const real_t *vdata = GetVertex(ints[i]);
               real_t *vdata_serial = serialmesh.GetVertex(ints_serial[i]);
               for (int d = 0; d < spaceDim; d++)
               {
                  vdata_serial[d] = vdata[d];
               }
            }
         }
      }

      for (int p = 0; p < NRanks; p++)
      {
         if (p == save_rank) { continue; }
         MPI_Recv(&n_send_recv, 1, MPI_INT, p, 448, MyComm, &status);
         vert.SetSize(n_send_recv);
         if (n_send_recv)
         {
            MPI_Recv(&vert[0], n_send_recv, MPITypeMap<real_t>::mpi_type, p, 449, MyComm,
                     &status);
         }
         for (int i = 0; i < n_send_recv; )
         {
            if (Nodes)
            {
               serialmesh.GetNodalFESpace()->GetElementVDofs(elem_count++, dofs);
               serialmesh.GetNodes()->SetSubVector(dofs, &vert[i]);
               i += dofs.Size();
            }
            else
            {
               serialmesh.GetElementVertices(elem_count++, ints_serial);
               for (int j = 0; j < ints_serial.Size(); j++)
               {
                  real_t *vdata_serial = serialmesh.GetVertex(ints_serial[j]);
                  for (int d = 0; d < spaceDim; d++)
                  {
                     vdata_serial[d] = vert[i++];
                  }
               }
            }
         }
      }
   } // MyRank == save_rank
   else
   {
      n_send_recv = 0;
      Vector nodeloc;
      for (int e = 0; e < NumOfElements; e++)
      {
         if (Nodes)
         {
            const FiniteElement *fe = Nodes->FESpace()->GetFE(e);
            n_send_recv += spaceDim*fe->GetDof();
         }
         else
         {
            n_send_recv += elements[e]->GetNVertices()*spaceDim;
         }
      }
      MPI_Send(&n_send_recv, 1, MPI_INT, save_rank, 448, MyComm);
      vert.Reserve(n_send_recv);
      vert.SetSize(0);
      for (int e = 0; e < NumOfElements; e++)
      {
         if (Nodes)
         {
            Nodes->GetElementDofValues(e, nodeloc);
            for (int j = 0; j < nodeloc.Size(); j++)
            {
               vert.Append(nodeloc(j));
            }
         }
         else
         {
            GetElementVertices(e, ints);
            for (int i = 0; i < ints.Size(); i++)
            {
               const real_t *vdata = GetVertex(ints[i]);
               for (int d = 0; d < spaceDim; d++)
               {
                  vert.Append(vdata[d]);
               }
            }
         }
      }
      if (n_send_recv)
      {
         MPI_Send(&vert[0], n_send_recv, MPITypeMap<real_t>::mpi_type, save_rank, 449,
                  MyComm);
      }
   }

   MPI_Barrier(MyComm);
   return serialmesh;
}

void ParMesh::SaveAsOne(const std::string &fname, int precision) const
{
   ofstream ofs;
   if (MyRank == 0)
   {
      ofs.open(fname);
      ofs.precision(precision);
   }
   PrintAsOne(ofs);
}

void ParMesh::PrintAsOneXG(std::ostream &os)
{
   MFEM_ASSERT(Dim == spaceDim, "2D Manifolds not supported.");
   if (Dim == 3 && meshgen == 1)
   {
      int i, j, k, nv, ne, p;
      const int *ind, *v;
      MPI_Status status;
      Array<real_t> vert;
      Array<int> ints;

      if (MyRank == 0)
      {
         os << "NETGEN_Neutral_Format\n";
         // print the vertices
         ne = NumOfVertices;
         MPI_Reduce(&ne, &nv, 1, MPI_INT, MPI_SUM, 0, MyComm);
         os << nv << '\n';
         for (i = 0; i < NumOfVertices; i++)
         {
            for (j = 0; j < Dim; j++)
            {
               os << " " << vertices[i](j);
            }
            os << '\n';
         }
         for (p = 1; p < NRanks; p++)
         {
            MPI_Recv(&nv, 1, MPI_INT, p, 444, MyComm, &status);
            vert.SetSize(Dim*nv);
            MPI_Recv(&vert[0], Dim*nv, MPITypeMap<real_t>::mpi_type, p, 445, MyComm,
                     &status);
            for (i = 0; i < nv; i++)
            {
               for (j = 0; j < Dim; j++)
               {
                  os << " " << vert[Dim*i+j];
               }
               os << '\n';
            }
         }

         // print the elements
         nv = NumOfElements;
         MPI_Reduce(&nv, &ne, 1, MPI_INT, MPI_SUM, 0, MyComm);
         os << ne << '\n';
         for (i = 0; i < NumOfElements; i++)
         {
            nv = elements[i]->GetNVertices();
            ind = elements[i]->GetVertices();
            os << 1;
            for (j = 0; j < nv; j++)
            {
               os << " " << ind[j]+1;
            }
            os << '\n';
         }
         k = NumOfVertices;
         for (p = 1; p < NRanks; p++)
         {
            MPI_Recv(&nv, 1, MPI_INT, p, 444, MyComm, &status);
            MPI_Recv(&ne, 1, MPI_INT, p, 446, MyComm, &status);
            ints.SetSize(4*ne);
            MPI_Recv(&ints[0], 4*ne, MPI_INT, p, 447, MyComm, &status);
            for (i = 0; i < ne; i++)
            {
               os << p+1;
               for (j = 0; j < 4; j++)
               {
                  os << " " << k+ints[i*4+j]+1;
               }
               os << '\n';
            }
            k += nv;
         }
         // print the boundary + shared faces information
         nv = NumOfBdrElements + sface_lface.Size();
         MPI_Reduce(&nv, &ne, 1, MPI_INT, MPI_SUM, 0, MyComm);
         os << ne << '\n';
         // boundary
         for (i = 0; i < NumOfBdrElements; i++)
         {
            nv = boundary[i]->GetNVertices();
            ind = boundary[i]->GetVertices();
            os << 1;
            for (j = 0; j < nv; j++)
            {
               os << " " << ind[j]+1;
            }
            os << '\n';
         }
         // shared faces
         const int sf_attr =
            MyRank + 1 + (bdr_attributes.Size() > 0 ? bdr_attributes.Max() : 0);
         for (i = 0; i < shared_trias.Size(); i++)
         {
            ind = shared_trias[i].v;
            os << sf_attr;
            for (j = 0; j < 3; j++)
            {
               os << ' ' << ind[j]+1;
            }
            os << '\n';
         }
         // There are no quad shared faces
         k = NumOfVertices;
         for (p = 1; p < NRanks; p++)
         {
            MPI_Recv(&nv, 1, MPI_INT, p, 444, MyComm, &status);
            MPI_Recv(&ne, 1, MPI_INT, p, 446, MyComm, &status);
            ints.SetSize(3*ne);
            MPI_Recv(&ints[0], 3*ne, MPI_INT, p, 447, MyComm, &status);
            for (i = 0; i < ne; i++)
            {
               os << p+1;
               for (j = 0; j < 3; j++)
               {
                  os << ' ' << k+ints[i*3+j]+1;
               }
               os << '\n';
            }
            k += nv;
         }
      }
      else
      {
         ne = NumOfVertices;
         MPI_Reduce(&ne, &nv, 1, MPI_INT, MPI_SUM, 0, MyComm);
         MPI_Send(&NumOfVertices, 1, MPI_INT, 0, 444, MyComm);
         vert.SetSize(Dim*NumOfVertices);
         for (i = 0; i < NumOfVertices; i++)
            for (j = 0; j < Dim; j++)
            {
               vert[Dim*i+j] = vertices[i](j);
            }
         MPI_Send(&vert[0], Dim*NumOfVertices, MPITypeMap<real_t>::mpi_type,
                  0, 445, MyComm);
         // elements
         ne = NumOfElements;
         MPI_Reduce(&ne, &nv, 1, MPI_INT, MPI_SUM, 0, MyComm);
         MPI_Send(&NumOfVertices, 1, MPI_INT, 0, 444, MyComm);
         MPI_Send(&NumOfElements, 1, MPI_INT, 0, 446, MyComm);
         ints.SetSize(NumOfElements*4);
         for (i = 0; i < NumOfElements; i++)
         {
            v = elements[i]->GetVertices();
            for (j = 0; j < 4; j++)
            {
               ints[4*i+j] = v[j];
            }
         }
         MPI_Send(&ints[0], 4*NumOfElements, MPI_INT, 0, 447, MyComm);
         // boundary + shared faces
         nv = NumOfBdrElements + sface_lface.Size();
         MPI_Reduce(&nv, &ne, 1, MPI_INT, MPI_SUM, 0, MyComm);
         MPI_Send(&NumOfVertices, 1, MPI_INT, 0, 444, MyComm);
         ne = NumOfBdrElements + sface_lface.Size();
         MPI_Send(&ne, 1, MPI_INT, 0, 446, MyComm);
         ints.SetSize(3*ne);
         for (i = 0; i < NumOfBdrElements; i++)
         {
            v = boundary[i]->GetVertices();
            for (j = 0; j < 3; j++)
            {
               ints[3*i+j] = v[j];
            }
         }
         for ( ; i < ne; i++)
         {
            v = shared_trias[i-NumOfBdrElements].v; // tet mesh
            for (j = 0; j < 3; j++)
            {
               ints[3*i+j] = v[j];
            }
         }
         MPI_Send(&ints[0], 3*ne, MPI_INT, 0, 447, MyComm);
      }
   }

   if (Dim == 3 && meshgen == 2)
   {
      int i, j, k, nv, ne, p;
      const int *ind, *v;
      MPI_Status status;
      Array<real_t> vert;
      Array<int> ints;

      int TG_nv, TG_ne, TG_nbe;

      if (MyRank == 0)
      {
         MPI_Reduce(&NumOfVertices, &TG_nv, 1, MPI_INT, MPI_SUM, 0, MyComm);
         MPI_Reduce(&NumOfElements, &TG_ne, 1, MPI_INT, MPI_SUM, 0, MyComm);
         nv = NumOfBdrElements + sface_lface.Size();
         MPI_Reduce(&nv, &TG_nbe, 1, MPI_INT, MPI_SUM, 0, MyComm);

         os << "TrueGrid\n"
            << "1 " << TG_nv << " " << TG_ne << " 0 0 0 0 0 0 0\n"
            << "0 0 0 1 0 0 0 0 0 0 0\n"
            << "0 0 " << TG_nbe << " 0 0 0 0 0 0 0 0 0 0 0 0 0\n"
            << "0.0 0.0 0.0 0 0 0.0 0.0 0 0.0\n"
            << "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n";

         // print the vertices
         nv = TG_nv;
         for (i = 0; i < NumOfVertices; i++)
         {
            os << i+1 << " 0.0 " << vertices[i](0) << " " << vertices[i](1)
               << " " << vertices[i](2) << " 0.0\n";
         }
         for (p = 1; p < NRanks; p++)
         {
            MPI_Recv(&nv, 1, MPI_INT, p, 444, MyComm, &status);
            vert.SetSize(Dim*nv);
            MPI_Recv(&vert[0], Dim*nv, MPITypeMap<real_t>::mpi_type, p, 445, MyComm,
                     &status);
            for (i = 0; i < nv; i++)
            {
               os << i+1 << " 0.0 " << vert[Dim*i] << " " << vert[Dim*i+1]
                  << " " << vert[Dim*i+2] << " 0.0\n";
            }
         }

         // print the elements
         ne = TG_ne;
         for (i = 0; i < NumOfElements; i++)
         {
            nv = elements[i]->GetNVertices();
            ind = elements[i]->GetVertices();
            os << i+1 << " " << 1;
            for (j = 0; j < nv; j++)
            {
               os << " " << ind[j]+1;
            }
            os << '\n';
         }
         k = NumOfVertices;
         for (p = 1; p < NRanks; p++)
         {
            MPI_Recv(&nv, 1, MPI_INT, p, 444, MyComm, &status);
            MPI_Recv(&ne, 1, MPI_INT, p, 446, MyComm, &status);
            ints.SetSize(8*ne);
            MPI_Recv(&ints[0], 8*ne, MPI_INT, p, 447, MyComm, &status);
            for (i = 0; i < ne; i++)
            {
               os << i+1 << " " << p+1;
               for (j = 0; j < 8; j++)
               {
                  os << " " << k+ints[i*8+j]+1;
               }
               os << '\n';
            }
            k += nv;
         }

         // print the boundary + shared faces information
         ne = TG_nbe;
         // boundary
         for (i = 0; i < NumOfBdrElements; i++)
         {
            nv = boundary[i]->GetNVertices();
            ind = boundary[i]->GetVertices();
            os << 1;
            for (j = 0; j < nv; j++)
            {
               os << " " << ind[j]+1;
            }
            os << " 1.0 1.0 1.0 1.0\n";
         }
         // shared faces
         const int sf_attr =
            MyRank + 1 + (bdr_attributes.Size() > 0 ? bdr_attributes.Max() : 0);
         // There are no shared triangle faces
         for (i = 0; i < shared_quads.Size(); i++)
         {
            ind = shared_quads[i].v;
            os << sf_attr;
            for (j = 0; j < 4; j++)
            {
               os << ' ' << ind[j]+1;
            }
            os << " 1.0 1.0 1.0 1.0\n";
         }
         k = NumOfVertices;
         for (p = 1; p < NRanks; p++)
         {
            MPI_Recv(&nv, 1, MPI_INT, p, 444, MyComm, &status);
            MPI_Recv(&ne, 1, MPI_INT, p, 446, MyComm, &status);
            ints.SetSize(4*ne);
            MPI_Recv(&ints[0], 4*ne, MPI_INT, p, 447, MyComm, &status);
            for (i = 0; i < ne; i++)
            {
               os << p+1;
               for (j = 0; j < 4; j++)
               {
                  os << " " << k+ints[i*4+j]+1;
               }
               os << " 1.0 1.0 1.0 1.0\n";
            }
            k += nv;
         }
      }
      else
      {
         MPI_Reduce(&NumOfVertices, &TG_nv, 1, MPI_INT, MPI_SUM, 0, MyComm);
         MPI_Reduce(&NumOfElements, &TG_ne, 1, MPI_INT, MPI_SUM, 0, MyComm);
         nv = NumOfBdrElements + sface_lface.Size();
         MPI_Reduce(&nv, &TG_nbe, 1, MPI_INT, MPI_SUM, 0, MyComm);

         MPI_Send(&NumOfVertices, 1, MPI_INT, 0, 444, MyComm);
         vert.SetSize(Dim*NumOfVertices);
         for (i = 0; i < NumOfVertices; i++)
            for (j = 0; j < Dim; j++)
            {
               vert[Dim*i+j] = vertices[i](j);
            }
         MPI_Send(&vert[0], Dim*NumOfVertices, MPITypeMap<real_t>::mpi_type, 0, 445,
                  MyComm);
         // elements
         MPI_Send(&NumOfVertices, 1, MPI_INT, 0, 444, MyComm);
         MPI_Send(&NumOfElements, 1, MPI_INT, 0, 446, MyComm);
         ints.SetSize(NumOfElements*8);
         for (i = 0; i < NumOfElements; i++)
         {
            v = elements[i]->GetVertices();
            for (j = 0; j < 8; j++)
            {
               ints[8*i+j] = v[j];
            }
         }
         MPI_Send(&ints[0], 8*NumOfElements, MPI_INT, 0, 447, MyComm);
         // boundary + shared faces
         MPI_Send(&NumOfVertices, 1, MPI_INT, 0, 444, MyComm);
         ne = NumOfBdrElements + sface_lface.Size();
         MPI_Send(&ne, 1, MPI_INT, 0, 446, MyComm);
         ints.SetSize(4*ne);
         for (i = 0; i < NumOfBdrElements; i++)
         {
            v = boundary[i]->GetVertices();
            for (j = 0; j < 4; j++)
            {
               ints[4*i+j] = v[j];
            }
         }
         for ( ; i < ne; i++)
         {
            v = shared_quads[i-NumOfBdrElements].v; // hex mesh
            for (j = 0; j < 4; j++)
            {
               ints[4*i+j] = v[j];
            }
         }
         MPI_Send(&ints[0], 4*ne, MPI_INT, 0, 447, MyComm);
      }
   }

   if (Dim == 2)
   {
      int i, j, k, attr, nv, ne, p;
      Array<int> v;
      MPI_Status status;
      Array<real_t> vert;
      Array<int> ints;

      if (MyRank == 0)
      {
         os << "areamesh2\n\n";

         // print the boundary + shared edges information
         nv = NumOfBdrElements + shared_edges.Size();
         MPI_Reduce(&nv, &ne, 1, MPI_INT, MPI_SUM, 0, MyComm);
         os << ne << '\n';
         // boundary
         for (i = 0; i < NumOfBdrElements; i++)
         {
            attr = boundary[i]->GetAttribute();
            boundary[i]->GetVertices(v);
            os << attr << "     ";
            for (j = 0; j < v.Size(); j++)
            {
               os << v[j] + 1 << "   ";
            }
            os << '\n';
         }
         // shared edges
         for (i = 0; i < shared_edges.Size(); i++)
         {
            attr = shared_edges[i]->GetAttribute();
            shared_edges[i]->GetVertices(v);
            os << attr << "     ";
            for (j = 0; j < v.Size(); j++)
            {
               os << v[j] + 1 << "   ";
            }
            os << '\n';
         }
         k = NumOfVertices;
         for (p = 1; p < NRanks; p++)
         {
            MPI_Recv(&nv, 1, MPI_INT, p, 444, MyComm, &status);
            MPI_Recv(&ne, 1, MPI_INT, p, 446, MyComm, &status);
            ints.SetSize(2*ne);
            MPI_Recv(&ints[0], 2*ne, MPI_INT, p, 447, MyComm, &status);
            for (i = 0; i < ne; i++)
            {
               os << p+1;
               for (j = 0; j < 2; j++)
               {
                  os << " " << k+ints[i*2+j]+1;
               }
               os << '\n';
            }
            k += nv;
         }

         // print the elements
         nv = NumOfElements;
         MPI_Reduce(&nv, &ne, 1, MPI_INT, MPI_SUM, 0, MyComm);
         os << ne << '\n';
         for (i = 0; i < NumOfElements; i++)
         {
            // attr = elements[i]->GetAttribute(); // not used
            elements[i]->GetVertices(v);
            os << 1 << "   " << 3 << "   ";
            for (j = 0; j < v.Size(); j++)
            {
               os << v[j] + 1 << "  ";
            }
            os << '\n';
         }
         k = NumOfVertices;
         for (p = 1; p < NRanks; p++)
         {
            MPI_Recv(&nv, 1, MPI_INT, p, 444, MyComm, &status);
            MPI_Recv(&ne, 1, MPI_INT, p, 446, MyComm, &status);
            ints.SetSize(3*ne);
            MPI_Recv(&ints[0], 3*ne, MPI_INT, p, 447, MyComm, &status);
            for (i = 0; i < ne; i++)
            {
               os << p+1 << " " << 3;
               for (j = 0; j < 3; j++)
               {
                  os << " " << k+ints[i*3+j]+1;
               }
               os << '\n';
            }
            k += nv;
         }

         // print the vertices
         ne = NumOfVertices;
         MPI_Reduce(&ne, &nv, 1, MPI_INT, MPI_SUM, 0, MyComm);
         os << nv << '\n';
         for (i = 0; i < NumOfVertices; i++)
         {
            for (j = 0; j < Dim; j++)
            {
               os << vertices[i](j) << " ";
            }
            os << '\n';
         }
         for (p = 1; p < NRanks; p++)
         {
            MPI_Recv(&nv, 1, MPI_INT, p, 444, MyComm, &status);
            vert.SetSize(Dim*nv);
            MPI_Recv(&vert[0], Dim*nv, MPITypeMap<real_t>::mpi_type, p, 445, MyComm,
                     &status);
            for (i = 0; i < nv; i++)
            {
               for (j = 0; j < Dim; j++)
               {
                  os << " " << vert[Dim*i+j];
               }
               os << '\n';
            }
         }
      }
      else
      {
         // boundary + shared faces
         nv = NumOfBdrElements + shared_edges.Size();
         MPI_Reduce(&nv, &ne, 1, MPI_INT, MPI_SUM, 0, MyComm);
         MPI_Send(&NumOfVertices, 1, MPI_INT, 0, 444, MyComm);
         ne = NumOfBdrElements + shared_edges.Size();
         MPI_Send(&ne, 1, MPI_INT, 0, 446, MyComm);
         ints.SetSize(2*ne);
         for (i = 0; i < NumOfBdrElements; i++)
         {
            boundary[i]->GetVertices(v);
            for (j = 0; j < 2; j++)
            {
               ints[2*i+j] = v[j];
            }
         }
         for ( ; i < ne; i++)
         {
            shared_edges[i-NumOfBdrElements]->GetVertices(v);
            for (j = 0; j < 2; j++)
            {
               ints[2*i+j] = v[j];
            }
         }
         MPI_Send(&ints[0], 2*ne, MPI_INT, 0, 447, MyComm);
         // elements
         ne = NumOfElements;
         MPI_Reduce(&ne, &nv, 1, MPI_INT, MPI_SUM, 0, MyComm);
         MPI_Send(&NumOfVertices, 1, MPI_INT, 0, 444, MyComm);
         MPI_Send(&NumOfElements, 1, MPI_INT, 0, 446, MyComm);
         ints.SetSize(NumOfElements*3);
         for (i = 0; i < NumOfElements; i++)
         {
            elements[i]->GetVertices(v);
            for (j = 0; j < 3; j++)
            {
               ints[3*i+j] = v[j];
            }
         }
         MPI_Send(&ints[0], 3*NumOfElements, MPI_INT, 0, 447, MyComm);
         // vertices
         ne = NumOfVertices;
         MPI_Reduce(&ne, &nv, 1, MPI_INT, MPI_SUM, 0, MyComm);
         MPI_Send(&NumOfVertices, 1, MPI_INT, 0, 444, MyComm);
         vert.SetSize(Dim*NumOfVertices);
         for (i = 0; i < NumOfVertices; i++)
            for (j = 0; j < Dim; j++)
            {
               vert[Dim*i+j] = vertices[i](j);
            }
         MPI_Send(&vert[0], Dim*NumOfVertices, MPITypeMap<real_t>::mpi_type,
                  0, 445, MyComm);
      }
   }
}

void ParMesh::GetBoundingBox(Vector &gp_min, Vector &gp_max, int ref)
{
   int sdim;
   Vector p_min, p_max;

   this->Mesh::GetBoundingBox(p_min, p_max, ref);

   sdim = SpaceDimension();

   gp_min.SetSize(sdim);
   gp_max.SetSize(sdim);

   MPI_Allreduce(p_min.GetData(), gp_min.GetData(), sdim,
                 MPITypeMap<real_t>::mpi_type,
                 MPI_MIN, MyComm);
   MPI_Allreduce(p_max.GetData(), gp_max.GetData(), sdim,
                 MPITypeMap<real_t>::mpi_type,
                 MPI_MAX, MyComm);
}

void ParMesh::GetCharacteristics(real_t &gh_min, real_t &gh_max,
                                 real_t &gk_min, real_t &gk_max)
{
   real_t h_min, h_max, kappa_min, kappa_max;

   this->Mesh::GetCharacteristics(h_min, h_max, kappa_min, kappa_max);

   MPI_Allreduce(&h_min, &gh_min, 1, MPITypeMap<real_t>::mpi_type, MPI_MIN,
                 MyComm);
   MPI_Allreduce(&h_max, &gh_max, 1, MPITypeMap<real_t>::mpi_type, MPI_MAX,
                 MyComm);
   MPI_Allreduce(&kappa_min, &gk_min, 1, MPITypeMap<real_t>::mpi_type, MPI_MIN,
                 MyComm);
   MPI_Allreduce(&kappa_max, &gk_max, 1, MPITypeMap<real_t>::mpi_type, MPI_MAX,
                 MyComm);
}

void ParMesh::PrintInfo(std::ostream &os)
{
   int i;
   DenseMatrix J(Dim);
   real_t h_min, h_max, kappa_min, kappa_max, h, kappa;

   if (MyRank == 0)
   {
      os << "Parallel Mesh Stats:" << '\n';
   }

   for (i = 0; i < NumOfElements; i++)
   {
      GetElementJacobian(i, J);
      h = pow(fabs(J.Weight()), 1.0/real_t(Dim));
      kappa = (Dim == spaceDim) ?
              J.CalcSingularvalue(0) / J.CalcSingularvalue(Dim-1) : -1.0;
      if (i == 0)
      {
         h_min = h_max = h;
         kappa_min = kappa_max = kappa;
      }
      else
      {
         if (h < h_min) { h_min = h; }
         if (h > h_max) { h_max = h; }
         if (kappa < kappa_min) { kappa_min = kappa; }
         if (kappa > kappa_max) { kappa_max = kappa; }
      }
   }

   real_t gh_min, gh_max, gk_min, gk_max;
   MPI_Reduce(&h_min, &gh_min, 1, MPITypeMap<real_t>::mpi_type, MPI_MIN, 0,
              MyComm);
   MPI_Reduce(&h_max, &gh_max, 1, MPITypeMap<real_t>::mpi_type, MPI_MAX, 0,
              MyComm);
   MPI_Reduce(&kappa_min, &gk_min, 1, MPITypeMap<real_t>::mpi_type, MPI_MIN, 0,
              MyComm);
   MPI_Reduce(&kappa_max, &gk_max, 1, MPITypeMap<real_t>::mpi_type, MPI_MAX, 0,
              MyComm);

   // TODO: collect and print stats by geometry

   long long ldata[5]; // vert, edge, face, elem, neighbors;
   long long mindata[5], maxdata[5], sumdata[5];

   // count locally owned vertices, edges, and faces
   ldata[0] = GetNV();
   ldata[1] = GetNEdges();
   ldata[2] = GetNFaces();
   ldata[3] = GetNE();
   ldata[4] = gtopo.GetNumNeighbors()-1;
   for (int gr = 1; gr < GetNGroups(); gr++)
   {
      if (!gtopo.IAmMaster(gr)) // we are not the master
      {
         ldata[0] -= group_svert.RowSize(gr-1);
         ldata[1] -= group_sedge.RowSize(gr-1);
         ldata[2] -= group_stria.RowSize(gr-1);
         ldata[2] -= group_squad.RowSize(gr-1);
      }
   }

   MPI_Reduce(ldata, mindata, 5, MPI_LONG_LONG, MPI_MIN, 0, MyComm);
   MPI_Reduce(ldata, sumdata, 5, MPI_LONG_LONG, MPI_SUM, 0, MyComm);
   MPI_Reduce(ldata, maxdata, 5, MPI_LONG_LONG, MPI_MAX, 0, MyComm);

   if (MyRank == 0)
   {
      os << '\n'
         << "           "
         << setw(12) << "minimum"
         << setw(12) << "average"
         << setw(12) << "maximum"
         << setw(12) << "total" << '\n';
      os << " vertices  "
         << setw(12) << mindata[0]
         << setw(12) << sumdata[0]/NRanks
         << setw(12) << maxdata[0]
         << setw(12) << sumdata[0] << '\n';
      os << " edges     "
         << setw(12) << mindata[1]
         << setw(12) << sumdata[1]/NRanks
         << setw(12) << maxdata[1]
         << setw(12) << sumdata[1] << '\n';
      if (Dim == 3)
      {
         os << " faces     "
            << setw(12) << mindata[2]
            << setw(12) << sumdata[2]/NRanks
            << setw(12) << maxdata[2]
            << setw(12) << sumdata[2] << '\n';
      }
      os << " elements  "
         << setw(12) << mindata[3]
         << setw(12) << sumdata[3]/NRanks
         << setw(12) << maxdata[3]
         << setw(12) << sumdata[3] << '\n';
      os << " neighbors "
         << setw(12) << mindata[4]
         << setw(12) << sumdata[4]/NRanks
         << setw(12) << maxdata[4] << '\n';
      os << '\n'
         << "       "
         << setw(12) << "minimum"
         << setw(12) << "maximum" << '\n';
      os << " h     "
         << setw(12) << gh_min
         << setw(12) << gh_max << '\n';
      os << " kappa "
         << setw(12) << gk_min
         << setw(12) << gk_max << '\n';
      os << std::flush;
   }
}

long long ParMesh::ReduceInt(int value) const
{
   long long local = value, global;
   MPI_Allreduce(&local, &global, 1, MPI_LONG_LONG, MPI_SUM, MyComm);
   return global;
}

void ParMesh::ParPrint(ostream &os, const std::string &comments) const
{
   if (NURBSext)
   {
      // TODO: NURBS meshes.
      Print(os, comments); // use the serial MFEM v1.0 format for now
      return;
   }

   if (Nonconforming())
   {
      // the NC mesh format works both in serial and in parallel
      Printer(os, "", comments);
      return;
   }

   // Write out serial mesh.  Tell serial mesh to delineate the end of its
   // output with 'mfem_serial_mesh_end' instead of 'mfem_mesh_end', as we will
   // be adding additional parallel mesh information.
   Printer(os, "mfem_serial_mesh_end", comments);

   // write out group topology info.
   gtopo.Save(os);

   os << "\ntotal_shared_vertices " << svert_lvert.Size() << '\n';
   if (Dim >= 2)
   {
      os << "total_shared_edges " << shared_edges.Size() << '\n';
   }
   if (Dim >= 3)
   {
      os << "total_shared_faces " << sface_lface.Size() << '\n';
   }
   os << "\n# group 0 has no shared entities\n";
   for (int gr = 1; gr < GetNGroups(); gr++)
   {
      {
         const int  nv = group_svert.RowSize(gr-1);
         const int *sv = group_svert.GetRow(gr-1);
         os << "\n# group " << gr << "\nshared_vertices " << nv << '\n';
         for (int i = 0; i < nv; i++)
         {
            os << svert_lvert[sv[i]] << '\n';
         }
      }
      if (Dim >= 2)
      {
         const int  ne = group_sedge.RowSize(gr-1);
         const int *se = group_sedge.GetRow(gr-1);
         os << "\nshared_edges " << ne << '\n';
         for (int i = 0; i < ne; i++)
         {
            const int *v = shared_edges[se[i]]->GetVertices();
            os << v[0] << ' ' << v[1] << '\n';
         }
      }
      if (Dim >= 3)
      {
         const int  nt = group_stria.RowSize(gr-1);
         const int *st = group_stria.GetRow(gr-1);
         const int  nq = group_squad.RowSize(gr-1);
         const int *sq = group_squad.GetRow(gr-1);
         os << "\nshared_faces " << nt+nq << '\n';
         for (int i = 0; i < nt; i++)
         {
            os << Geometry::TRIANGLE;
            const int *v = shared_trias[st[i]].v;
            for (int j = 0; j < 3; j++) { os << ' ' << v[j]; }
            os << '\n';
         }
         for (int i = 0; i < nq; i++)
         {
            os << Geometry::SQUARE;
            const int *v = shared_quads[sq[i]].v;
            for (int j = 0; j < 4; j++) { os << ' ' << v[j]; }
            os << '\n';
         }
      }
   }

   // Write out section end tag for mesh.
   os << "\nmfem_mesh_end" << endl;
}

void ParMesh::PrintVTU(std::string pathname,
                       VTKFormat format,
                       bool high_order_output,
                       int compression_level,
                       bool bdr)
{
   int pad_digits_rank = 6;
   DataCollection::create_directory(pathname, this, MyRank);

   std::string::size_type pos = pathname.find_last_of('/');
   std::string fname
      = (pos == std::string::npos) ? pathname : pathname.substr(pos+1);

   if (MyRank == 0)
   {
      std::string pvtu_name = pathname + "/" + fname + ".pvtu";
      std::ofstream os(pvtu_name);

      std::string data_type = (format == VTKFormat::BINARY32) ? "Float32" : "Float64";
      std::string data_format = (format == VTKFormat::ASCII) ? "ascii" : "binary";

      os << "<?xml version=\"1.0\"?>\n";
      os << "<VTKFile type=\"PUnstructuredGrid\"";
      os << " version =\"0.1\" byte_order=\"" << VTKByteOrder() << "\">\n";
      os << "<PUnstructuredGrid GhostLevel=\"0\">\n";

      os << "<PPoints>\n";
      os << "\t<PDataArray type=\"" << data_type << "\" ";
      os << " Name=\"Points\" NumberOfComponents=\"3\""
         << " format=\"" << data_format << "\"/>\n";
      os << "</PPoints>\n";

      os << "<PCells>\n";
      os << "\t<PDataArray type=\"Int32\" ";
      os << " Name=\"connectivity\" NumberOfComponents=\"1\""
         << " format=\"" << data_format << "\"/>\n";
      os << "\t<PDataArray type=\"Int32\" ";
      os << " Name=\"offsets\"      NumberOfComponents=\"1\""
         << " format=\"" << data_format << "\"/>\n";
      os << "\t<PDataArray type=\"UInt8\" ";
      os << " Name=\"types\"        NumberOfComponents=\"1\""
         << " format=\"" << data_format << "\"/>\n";
      os << "</PCells>\n";

      os << "<PCellData>\n";
      os << "\t<PDataArray type=\"Int32\" Name=\"" << "attribute"
         << "\" NumberOfComponents=\"1\""
         << " format=\"" << data_format << "\"/>\n";
      os << "</PCellData>\n";

      for (int ii=0; ii<NRanks; ii++)
      {
         std::string piece = fname + ".proc"
                             + to_padded_string(ii, pad_digits_rank) + ".vtu";
         os << "<Piece Source=\"" << piece << "\"/>\n";
      }

      os << "</PUnstructuredGrid>\n";
      os << "</VTKFile>\n";
      os.close();
   }

   std::string vtu_fname = pathname + "/" + fname + ".proc"
                           + to_padded_string(MyRank, pad_digits_rank);
   Mesh::PrintVTU(vtu_fname, format, high_order_output, compression_level, bdr);
}

int ParMesh::FindPoints(DenseMatrix& point_mat, Array<int>& elem_id,
                        Array<IntegrationPoint>& ip, bool warn,
                        InverseElementTransformation *inv_trans)
{
   const int npts = point_mat.Width();
   if (npts == 0) { return 0; }

   const bool no_warn = false;
   Mesh::FindPoints(point_mat, elem_id, ip, no_warn, inv_trans);

   // If multiple processors find the same point, we need to choose only one of
   // the processors to mark that point as found.
   // Here, we choose the processor with the minimal rank.

   Array<int> my_point_rank(npts), glob_point_rank(npts);
   for (int k = 0; k < npts; k++)
   {
      my_point_rank[k] = (elem_id[k] == -1) ? NRanks : MyRank;
   }

   MPI_Allreduce(my_point_rank.GetData(), glob_point_rank.GetData(), npts,
                 MPI_INT, MPI_MIN, MyComm);

   int pts_found = 0;
   for (int k = 0; k < npts; k++)
   {
      if (glob_point_rank[k] == NRanks) { elem_id[k] = -1; }
      else
      {
         pts_found++;
         if (glob_point_rank[k] != MyRank) { elem_id[k] = -2; }
      }
   }
   if (warn && pts_found != npts && MyRank == 0)
   {
      MFEM_WARNING((npts-pts_found) << " points were not found");
   }
   return pts_found;
}

static void PrintVertex(const Vertex &v, int space_dim, ostream &os)
{
   os << v(0);
   for (int d = 1; d < space_dim; d++)
   {
      os << ' ' << v(d);
   }
}

void ParMesh::PrintSharedEntities(const std::string &fname_prefix) const
{
   stringstream out_name;
   out_name << fname_prefix << '_' << setw(5) << setfill('0') << MyRank
            << ".shared_entities";
   ofstream os(out_name.str().c_str());
   os.precision(16);

   gtopo.Save(out);

   os << "\ntotal_shared_vertices " << svert_lvert.Size() << '\n';
   if (Dim >= 2)
   {
      os << "total_shared_edges " << shared_edges.Size() << '\n';
   }
   if (Dim >= 3)
   {
      os << "total_shared_faces " << sface_lface.Size() << '\n';
   }
   for (int gr = 1; gr < GetNGroups(); gr++)
   {
      {
         const int  nv = group_svert.RowSize(gr-1);
         const int *sv = group_svert.GetRow(gr-1);
         os << "\n# group " << gr << "\n\nshared_vertices " << nv << '\n';
         for (int i = 0; i < nv; i++)
         {
            const int lvi = svert_lvert[sv[i]];
            // os << lvi << '\n';
            PrintVertex(vertices[lvi], spaceDim, os);
            os << '\n';
         }
      }
      if (Dim >= 2)
      {
         const int  ne = group_sedge.RowSize(gr-1);
         const int *se = group_sedge.GetRow(gr-1);
         os << "\nshared_edges " << ne << '\n';
         for (int i = 0; i < ne; i++)
         {
            const int *v = shared_edges[se[i]]->GetVertices();
            // os << v[0] << ' ' << v[1] << '\n';
            PrintVertex(vertices[v[0]], spaceDim, os);
            os << " | ";
            PrintVertex(vertices[v[1]], spaceDim, os);
            os << '\n';
         }
      }
      if (Dim >= 3)
      {
         const int  nt = group_stria.RowSize(gr-1);
         const int *st = group_stria.GetRow(gr-1);
         const int  nq = group_squad.RowSize(gr-1);
         const int *sq = group_squad.GetRow(gr-1);
         os << "\nshared_faces " << nt+nq << '\n';
         for (int i = 0; i < nt; i++)
         {
            const int *v = shared_trias[st[i]].v;
#if 0
            os << Geometry::TRIANGLE;
            for (int j = 0; j < 3; j++) { os << ' ' << v[j]; }
            os << '\n';
#endif
            for (int j = 0; j < 3; j++)
            {
               PrintVertex(vertices[v[j]], spaceDim, os);
               (j < 2) ? os << " | " : os << '\n';
            }
         }
         for (int i = 0; i < nq; i++)
         {
            const int *v = shared_quads[sq[i]].v;
#if 0
            os << Geometry::SQUARE;
            for (int j = 0; j < 4; j++) { os << ' ' << v[j]; }
            os << '\n';
#endif
            for (int j = 0; j < 4; j++)
            {
               PrintVertex(vertices[v[j]], spaceDim, os);
               (j < 3) ? os << " | " : os << '\n';
            }
         }
      }
   }
}

void ParMesh::GetGlobalVertexIndices(Array<HYPRE_BigInt> &gi) const
{
   H1_FECollection fec(1, Dim); // Order 1, mesh dimension (not spatial dimension).
   ParMesh *pm = const_cast<ParMesh *>(this);
   ParFiniteElementSpace fespace(pm, &fec);

   gi.SetSize(GetNV());

   Array<int> dofs;
   for (int i=0; i<GetNV(); ++i)
   {
      fespace.GetVertexDofs(i, dofs);
      gi[i] = fespace.GetGlobalTDofNumber(dofs[0]);
   }
}

void ParMesh::GetGlobalEdgeIndices(Array<HYPRE_BigInt> &gi) const
{
   if (Dim == 1)
   {
      GetGlobalVertexIndices(gi);
      return;
   }

   ND_FECollection fec(1, Dim); // Order 1, mesh dimension (not spatial dimension).
   ParMesh *pm = const_cast<ParMesh *>(this);
   ParFiniteElementSpace fespace(pm, &fec);

   gi.SetSize(GetNEdges());

   Array<int> dofs;
   for (int i=0; i<GetNEdges(); ++i)
   {
      fespace.GetEdgeDofs(i, dofs);
      const int ldof = (dofs[0] >= 0) ? dofs[0] : -1 - dofs[0];
      gi[i] = fespace.GetGlobalTDofNumber(ldof);
   }
}

void ParMesh::GetGlobalFaceIndices(Array<HYPRE_BigInt> &gi) const
{
   if (Dim == 2)
   {
      GetGlobalEdgeIndices(gi);
      return;
   }
   else if (Dim == 1)
   {
      GetGlobalVertexIndices(gi);
      return;
   }

   RT_FECollection fec(0, Dim); // Order 0, mesh dimension (not spatial dimension).
   ParMesh *pm = const_cast<ParMesh *>(this);
   ParFiniteElementSpace fespace(pm, &fec);

   gi.SetSize(GetNFaces());

   Array<int> dofs;
   for (int i=0; i<GetNFaces(); ++i)
   {
      fespace.GetFaceDofs(i, dofs);
      const int ldof = (dofs[0] >= 0) ? dofs[0] : -1 - dofs[0];
      gi[i] = fespace.GetGlobalTDofNumber(ldof);
   }
}

void ParMesh::GetGlobalElementIndices(Array<HYPRE_BigInt> &gi) const
{
   ComputeGlobalElementOffset();

   // Cast from long long to HYPRE_BigInt
   const HYPRE_BigInt offset = glob_elem_offset;

   gi.SetSize(GetNE());
   for (int i=0; i<GetNE(); ++i)
   {
      gi[i] = offset + i;
   }
}

void ParMesh::Swap(ParMesh &other)
{
   Mesh::Swap(other, true);

   mfem::Swap(MyComm, other.MyComm);
   mfem::Swap(NRanks, other.NRanks);
   mfem::Swap(MyRank, other.MyRank);

   mfem::Swap(glob_elem_offset, other.glob_elem_offset);
   mfem::Swap(glob_offset_sequence, other.glob_offset_sequence);

   gtopo.Swap(other.gtopo);

   group_svert.Swap(other.group_svert);
   group_sedge.Swap(other.group_sedge);
   group_stria.Swap(other.group_stria);
   group_squad.Swap(other.group_squad);

   mfem::Swap(shared_edges, other.shared_edges);
   mfem::Swap(shared_trias, other.shared_trias);
   mfem::Swap(shared_quads, other.shared_quads);
   mfem::Swap(svert_lvert, other.svert_lvert);
   mfem::Swap(sedge_ledge, other.sedge_ledge);
   mfem::Swap(sface_lface, other.sface_lface);

   // Swap face-neighbor data
   mfem::Swap(have_face_nbr_data, other.have_face_nbr_data);
   mfem::Swap(face_nbr_group, other.face_nbr_group);
   mfem::Swap(face_nbr_elements_offset, other.face_nbr_elements_offset);
   mfem::Swap(face_nbr_vertices_offset, other.face_nbr_vertices_offset);
   mfem::Swap(face_nbr_elements, other.face_nbr_elements);
   mfem::Swap(face_nbr_vertices, other.face_nbr_vertices);
   mfem::Swap(send_face_nbr_elements, other.send_face_nbr_elements);
   mfem::Swap(send_face_nbr_vertices, other.send_face_nbr_vertices);
   std::swap(face_nbr_el_ori, other.face_nbr_el_ori);
   std::swap(face_nbr_el_to_face, other.face_nbr_el_to_face);

   // Nodes, NCMesh, and NURBSExtension are taken care of by Mesh::Swap
   mfem::Swap(pncmesh, other.pncmesh);

   print_shared = other.print_shared;
}

void ParMesh::Destroy()
{
   delete pncmesh;
   ncmesh = pncmesh = NULL;

   DeleteFaceNbrData();

   for (int i = 0; i < shared_edges.Size(); i++)
   {
      FreeElement(shared_edges[i]);
   }
   shared_edges.DeleteAll();

   face_nbr_el_to_face = nullptr;
}

ParMesh::~ParMesh()
{
   ParMesh::Destroy();

   // The Mesh destructor is called automatically
}

}

#endif
