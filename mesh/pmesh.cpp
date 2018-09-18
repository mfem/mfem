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
#include "../fem/fem.hpp"
#include "../general/sets.hpp"
#include "../general/sort_pairs.hpp"
#include "../general/text.hpp"
#include "../general/globals.hpp"

#include <iostream>
using namespace std;

namespace mfem
{

ParMesh::ParMesh(const ParMesh &pmesh, bool copy_nodes)
   : Mesh(pmesh, false),
     group_svert(pmesh.group_svert),
     group_sedge(pmesh.group_sedge),
     group_sface(pmesh.group_sface),
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

   // Duplicate the shared_faces
   shared_faces.SetSize(pmesh.shared_faces.Size());
   for (int i = 0; i < shared_faces.Size(); i++)
   {
      shared_faces[i] = pmesh.shared_faces[i]->Duplicate(this);
   }

   // Copy the shared-to-local index Arrays
   pmesh.svert_lvert.Copy(svert_lvert);
   pmesh.sedge_ledge.Copy(sedge_ledge);
   pmesh.sface_lface.Copy(sface_lface);

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

ParMesh::ParMesh(MPI_Comm comm, Mesh &mesh, int *partitioning_,
                 int part_method)
   : gtopo(comm)
{
   int *partitioning;

   MyComm = comm;
   MPI_Comm_size(MyComm, &NRanks);
   MPI_Comm_rank(MyComm, &MyRank);

   ncmesh = pncmesh = NULL;
   if (mesh.Nonconforming())
   {
      ncmesh = pncmesh = new ParNCMesh(comm, *mesh.ncmesh);
   }

   if (partitioning_)
   {
      partitioning = partitioning_;
   }
   else
   {
      if (mesh.Nonconforming())
      {
         // save the element partitioning before Prune()
         partitioning = new int[mesh.GetNE()];
         for (int i = 0; i < mesh.GetNE(); i++)
         {
            partitioning[i] = pncmesh->InitialPartition(i);
         }
      }
      else
      {
         partitioning = mesh.GeneratePartitioning(NRanks, part_method);
      }
   }

   Table *edge_element = NULL;
   STable3D *faces_tbl = NULL;

   Array<int> vert_global_local(mesh.GetNV());
   Array<bool> activeBdrElem;

   if (mesh.Nonconforming())
   {
      pncmesh->Prune();
      faces_tbl = Mesh::InitFromNCMesh(*pncmesh);
      pncmesh->OnMeshUpdated(this);

      // In the NC case we already have local numbering for the
      // element vertices, which has come from InitFromNCMesh.
      // So derive vert_global_local from it.

      vert_global_local = -1;
      int le = 0;
      for (int i = 0; i < mesh.GetNE(); i++)
      {
         if (partitioning[i] == MyRank)
         {
            Array<int> vert_global, vert_local;
            mesh.GetElementVertices(i, vert_global);
            elements[le++]->GetVertices(vert_local);
            for (int j = 0; j < vert_local.Size(); j++)
            {
               vert_global_local[vert_global[j]] = vert_local[j];
            }
         }
      }

      GenerateNCFaceInfo();
   }
   else
   {
      Dim = mesh.Dim;
      spaceDim = mesh.spaceDim;

      BaseGeom = mesh.BaseGeom;
      BaseBdrGeom = mesh.BaseBdrGeom;

      // fills out Mesh::vertices from partition of global mesh and
      // generates vert_global_local.
      NumOfVertices = BuildLocalVertices(mesh, partitioning,
                                         vert_global_local);

      // fills out Mesh::elements
      NumOfElements = BuildLocalElements(mesh, partitioning,
                                         vert_global_local);

      // fills out Mesh::boundary
      NumOfBdrElements = BuildLocalBoundary(mesh, partitioning,
                                            vert_global_local,
                                            activeBdrElem, edge_element);

      NumOfEdges = NumOfFaces = 0;

      if (Dim > 1)
      {
         el_to_edge = new Table;
         NumOfEdges = Mesh::GetElementToEdgeTable(*el_to_edge, be_to_edge);
      }

      if (Dim == 3)
      {
         faces_tbl = GetElementToFaceTable(1);
      }

      GenerateFaces();
   }

   meshgen = mesh.MeshGenerator();

   mesh.attributes.Copy(attributes);
   mesh.bdr_attributes.Copy(bdr_attributes);

   // Build groups.  At this point there is no difference between the
   // conforming and NC cases.

   ListOfIntegerSets  groups;
   {
      // the first group is the local one
      IntegerSet         group;
      group.Recreate(1, &MyRank);
      groups.Insert(group);
   }

   MFEM_ASSERT(mesh.GetNFaces() == 0 || Dim >= 3, "");

   Array<int> face_group(mesh.GetNFaces());
   Table *vert_element = mesh.GetVertexToElementTable(); // we must delete this

   int nsfaces = FindSharedFaces(mesh, partitioning, face_group, groups);
   int nsedges = FindSharedEdges(mesh, partitioning, edge_element, groups);
   int nsvert = FindSharedVertices(mesh, partitioning, vert_element, groups);

   // build the group communication topology
   gtopo.Create(groups, 822);

   // fill out group_sface, group_sedge, group_svert
   int ngroups = groups.Size()-1;
   BuildFaceGroup(ngroups, face_group, group_sface);
   BuildEdgeGroup(ngroups, *edge_element, group_sedge);
   BuildVertexGroup(ngroups, *vert_element, group_svert);

   // build shared_faces and sface_lface mapping
   BuildSharedFaceElems(nsfaces, mesh, partitioning, faces_tbl,
                        face_group, vert_global_local);

   // build shared_edges and sedge_ledge mapping
   BuildSharedEdgeElems(nsedges, mesh, vert_global_local,
                        edge_element);
   delete edge_element;

   // build svert_lvert mapping
   BuildSharedVertMapping(nsvert, vert_element, vert_global_local);
   delete vert_element;


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
         if (partitioning[i] == MyRank)
         {
            Nodes->FESpace()->GetElementVDofs(element_counter, lvdofs);
            mesh.GetNodes()->FESpace()->GetElementVDofs(i, gvdofs);
            mesh.GetNodes()->GetSubVector(gvdofs, lnodes);
            Nodes->SetSubVector(lvdofs, lnodes);
            element_counter++;
         }
   }

   if (partitioning)
   {
      delete [] partitioning;
   }

   have_face_nbr_data = false;
}

// protected method
ParMesh::ParMesh(const ParNCMesh &pncmesh)
   : MyComm(pncmesh.MyComm)
   , NRanks(pncmesh.NRanks)
   , MyRank(pncmesh.MyRank)
   , gtopo(MyComm)
   , pncmesh(NULL)
{
   Mesh::InitFromNCMesh(pncmesh);
   have_face_nbr_data = false;
}

ParMesh::ParMesh(MPI_Comm comm, istream &input, bool refine)
   : gtopo(comm)
{
   MyComm = comm;
   MPI_Comm_size(MyComm, &NRanks);
   MPI_Comm_rank(MyComm, &MyRank);

   have_face_nbr_data = false;
   pncmesh = NULL;

   string ident;

   // read the serial part of the mesh
   int gen_edges = 1;

   // Tell Loader() to read up to 'mfem_serial_mesh_end' instead of
   // 'mfem_mesh_end', as we have additional parallel mesh data to load in from
   // the stream.
   Loader(input, gen_edges, "mfem_serial_mesh_end");

   skip_comment_lines(input, '#');

   // read the group topology
   input >> ident;
   MFEM_VERIFY(ident == "communication_groups",
               "input stream is not a parallel MFEM mesh");
   gtopo.Load(input);

   skip_comment_lines(input, '#');

   DSTable *v_to_v = NULL;
   STable3D *faces_tbl = NULL;
   // read and set the sizes of svert_lvert, group_svert
   {
      int num_sverts;
      input >> ident >> num_sverts; // total_shared_vertices
      svert_lvert.SetSize(num_sverts);
      group_svert.SetDims(GetNGroups()-1, num_sverts);
   }
   // read and set the sizes of sedge_ledge, group_sedge
   if (Dim >= 2)
   {
      skip_comment_lines(input, '#');
      int num_sedges;
      input >> ident >> num_sedges; // total_shared_edges
      sedge_ledge.SetSize(num_sedges);
      shared_edges.SetSize(num_sedges);
      group_sedge.SetDims(GetNGroups()-1, num_sedges);
      v_to_v = new DSTable(NumOfVertices);
      GetVertexToVertexTable(*v_to_v);
   }
   else
   {
      group_sedge.SetSize(GetNGroups()-1, 0);   // create empty group_sedge
   }
   // read and set the sizes of sface_lface, group_sface
   if (Dim >= 3)
   {
      skip_comment_lines(input, '#');
      int num_sfaces;
      input >> ident >> num_sfaces; // total_shared_faces
      sface_lface.SetSize(num_sfaces);
      shared_faces.SetSize(num_sfaces);
      group_sface.SetDims(GetNGroups()-1, num_sfaces);
      faces_tbl = GetFacesTable();
   }
   else
   {
      group_sface.SetSize(GetNGroups()-1, 0);   // create empty group_sface
   }

   // read, group by group, the contents of group_svert, svert_lvert,
   // group_sedge, shared_edges, group_sface, shared_faces
   //
   // derive the contents of sedge_ledge, sface_lface
   int svert_counter = 0, sedge_counter = 0, sface_counter = 0;
   for (int gr = 1; gr < GetNGroups(); gr++)
   {
      int g;
      input >> ident >> g; // group
      if (g != gr)
      {
         mfem::err << "ParMesh::ParMesh : expecting group " << gr
                   << ", read group " << g << endl;
         mfem_error();
      }

      {
         int nv;
         input >> ident >> nv; // shared_vertices (in this group)
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
         ne += sedge_counter;
         MFEM_VERIFY(ne <= group_sedge.Size_of_connections(),
                     "incorrect number of total_shared_edges");
         group_sedge.GetI()[gr] = ne;
         for ( ; sedge_counter < ne; sedge_counter++)
         {
            group_sedge.GetJ()[sedge_counter] = sedge_counter;
            input >> v[0] >> v[1];
            shared_edges[sedge_counter] = new Segment(v[0], v[1], 1);
            sedge_ledge[sedge_counter] = (*v_to_v)(v[0], v[1]);
         }
      }
      if (Dim >= 3)
      {
         int nf;
         input >> ident >> nf; // shared_faces (in this group)
         nf += sface_counter;
         MFEM_VERIFY(nf <= group_sface.Size_of_connections(),
                     "incorrect number of total_shared_faces");
         group_sface.GetI()[gr] = nf;
         for ( ; sface_counter < nf; sface_counter++)
         {
            group_sface.GetJ()[sface_counter] = sface_counter;
            Element *sface = ReadElementWithoutAttr(input);
            shared_faces[sface_counter] = sface;
            const int *v = sface->GetVertices();
            switch (sface->GetType())
            {
               case Element::TRIANGLE:
                  sface_lface[sface_counter] = (*faces_tbl)(v[0], v[1], v[2]);
                  break;
               case Element::QUADRILATERAL:
                  sface_lface[sface_counter] =
                     (*faces_tbl)(v[0], v[1], v[2], v[3]);
                  break;
            }
         }
      }
   }
   delete faces_tbl;
   delete v_to_v;

   const bool fix_orientation = false;
   Finalize(refine, fix_orientation);

   // If the mesh has Nodes, convert them from GridFunction to ParGridFunction?

   // note: attributes and bdr_attributes are local lists

   // TODO: AMR meshes, NURBS meshes?
}

ParMesh::ParMesh(ParMesh *orig_mesh, int ref_factor, int ref_type)
   : Mesh(orig_mesh, ref_factor, ref_type),
     MyComm(orig_mesh->GetComm()),
     NRanks(orig_mesh->GetNRanks()),
     MyRank(orig_mesh->GetMyRank()),
     gtopo(orig_mesh->gtopo),
     have_face_nbr_data(false),
     pncmesh(NULL)
{
   // Need to initialize:
   // - shared_edges, shared_faces
   // - group_svert, group_sedge, group_sface
   // - svert_lvert, sedge_ledge, sface_lface

   H1_FECollection rfec(ref_factor, Dim, ref_type);
   ParFiniteElementSpace rfes(orig_mesh, &rfec);

   // count the number of entries in each row of group_s{vert,edge,face}
   group_svert.MakeI(GetNGroups()-1); // exclude the local group 0
   group_sedge.MakeI(GetNGroups()-1);
   group_sface.MakeI(GetNGroups()-1);
   for (int gr = 1; gr < GetNGroups(); gr++)
   {
      // orig vertex -> vertex
      group_svert.AddColumnsInRow(gr-1, orig_mesh->GroupNVertices(gr));
      // orig edge -> (ref_factor-1) vertices and (ref_factor) edges
      const int orig_ne = orig_mesh->GroupNEdges(gr);
      group_svert.AddColumnsInRow(gr-1, (ref_factor-1)*orig_ne);
      group_sedge.AddColumnsInRow(gr-1, ref_factor*orig_ne);
      // orig face -> (?) vertices, (?) edges, and (?) faces
      const int  orig_nf = orig_mesh->GroupNFaces(gr);
      const int *orig_sf = orig_mesh->group_sface.GetRow(gr-1);
      for (int fi = 0; fi < orig_nf; fi++)
      {
         const int orig_l_face = orig_mesh->sface_lface[orig_sf[fi]];
         const int geom = orig_mesh->GetFaceBaseGeometry(orig_l_face);
         const int nvert = Geometry::NumVerts[geom];
         RefinedGeometry &RG =
            *GlobGeometryRefiner.Refine(geom, ref_factor, ref_factor);

         // count internal vertices
         group_svert.AddColumnsInRow(gr-1, rfec.DofForGeometry(geom));
         // count internal edges
         group_sedge.AddColumnsInRow(gr-1, RG.RefEdges.Size()/2-RG.NumBdrEdges);
         // count refined faces
         group_sface.AddColumnsInRow(gr-1, RG.RefGeoms.Size()/nvert);
      }
   }

   group_svert.MakeJ();
   svert_lvert.Reserve(group_svert.Size_of_connections());

   group_sedge.MakeJ();
   shared_edges.Reserve(group_sedge.Size_of_connections());
   sedge_ledge.SetSize(group_sedge.Size_of_connections());

   group_sface.MakeJ();
   shared_faces.Reserve(group_sface.Size_of_connections());
   sface_lface.SetSize(group_sface.Size_of_connections());

   Array<int> rdofs;
   for (int gr = 1; gr < GetNGroups(); gr++)
   {
      // add shared vertices from original shared vertices
      const int orig_n_verts = orig_mesh->GroupNVertices(gr);
      for (int j = 0; j < orig_n_verts; j++)
      {
         rfes.GetVertexDofs(orig_mesh->GroupVertex(gr, j), rdofs);
         group_svert.AddConnection(gr-1, svert_lvert.Append(rdofs[0])-1);
      }

      // add refined shared edges; add shared vertices from refined shared edges
      const int orig_n_edges = orig_mesh->GroupNEdges(gr);
      const int geom = Geometry::SEGMENT;
      const int nvert = Geometry::NumVerts[geom];
      RefinedGeometry &RG = *GlobGeometryRefiner.Refine(geom, ref_factor);
      for (int e = 0; e < orig_n_edges; e++)
      {
         rfes.GetSharedEdgeDofs(gr, e, rdofs);
         MFEM_ASSERT(rdofs.Size() == RG.RefPts.Size(), "");
         // add the internal edge 'rdofs' as shared vertices
         for (int j = 2; j < rdofs.Size(); j++)
         {
            group_svert.AddConnection(gr-1, svert_lvert.Append(rdofs[j])-1);
         }
         const int *c2h_map = rfec.GetDofMap(geom);
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
      // add refined shared faces; add shared edges and shared vertices from
      // refined shared faces
      const int  orig_nf = orig_mesh->group_sface.RowSize(gr-1);
      const int *orig_sf = orig_mesh->group_sface.GetRow(gr-1);
      for (int f = 0; f < orig_nf; f++)
      {
         const int orig_l_face = orig_mesh->sface_lface[orig_sf[f]];
         const int geom = orig_mesh->GetFaceBaseGeometry(orig_l_face);
         const int nvert = Geometry::NumVerts[geom];
         RefinedGeometry &RG =
            *GlobGeometryRefiner.Refine(geom, ref_factor, ref_factor);

         rfes.GetSharedFaceDofs(gr, f, rdofs);
         MFEM_ASSERT(rdofs.Size() == RG.RefPts.Size(), "");
         // add the internal face 'rdofs' as shared vertices
         const int num_int_verts = rfec.DofForGeometry(geom);
         for (int j = rdofs.Size()-num_int_verts; j < rdofs.Size(); j++)
         {
            group_svert.AddConnection(gr-1, svert_lvert.Append(rdofs[j])-1);
         }
         const int *c2h_map = rfec.GetDofMap(geom);
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
            Element *elem = NewElement(geom);
            int *v = elem->GetVertices();
            for (int k = 0; k < nvert; k++)
            {
               int cid = RG.RefGeoms[j+k]; // local Cartesian index
               v[k] = rdofs[c2h_map[cid]];
            }
            group_sface.AddConnection(gr-1, shared_faces.Append(elem)-1);
         }
      }
   }
   group_svert.ShiftUpI();
   group_sedge.ShiftUpI();
   group_sface.ShiftUpI();

   // determine sedge_ledge
   if (shared_edges.Size() > 0)
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

   // determine sface_lface
   if (shared_faces.Size() > 0)
   {
      STable3D *faces_tbl = GetFacesTable();
      for (int sf = 0; sf < shared_faces.Size(); sf++)
      {
         int l_face;
         const int *v = shared_faces[sf]->GetVertices();
         switch (shared_faces[sf]->GetGeometryType())
         {
            case Geometry::TRIANGLE:
               l_face = (*faces_tbl)(v[0], v[1], v[2]);
               break;
            case Geometry::SQUARE:
               l_face = (*faces_tbl)(v[0], v[1], v[2], v[3]);
               break;
            default:
               MFEM_ABORT("invalid face geometry");
               l_face = -1;
         }
         MFEM_ASSERT(l_face >= 0, "invalid shared face");
         sface_lface[sf] = l_face;
      }
      delete faces_tbl;
   }
}


int ParMesh::FindSharedFaces(const Mesh &mesh, const int *partitioning,
                             Array<int>& face_group,
                             ListOfIntegerSets& groups)
{
   IntegerSet group;

   int sface_counter = 0;
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
            sface_counter++;
         }
      }
   }
   return sface_counter;
}


int ParMesh::FindSharedEdges(const Mesh &mesh, const int *partitioning,
                             Table*& edge_element,
                             ListOfIntegerSets& groups)
{
   IntegerSet group;

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

int ParMesh::FindSharedVertices(const Mesh &mesh, const int *partitioning,
                                Table* vert_element,
                                ListOfIntegerSets& groups)
{
   IntegerSet group;

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

void ParMesh::BuildFaceGroup(int ngroups, const Array<int>& face_group,
                             Table& group_sface)
{
   group_sface.MakeI(ngroups);

   for (int i = 0; i < face_group.Size(); i++)
   {
      if (face_group[i] >= 0)
      {
         group_sface.AddAColumnInRow(face_group[i]);
      }
   }

   group_sface.MakeJ();

   int sface_counter = 0;
   for (int i = 0; i < face_group.Size(); i++)
   {
      if (face_group[i] >= 0)
      {
         group_sface.AddConnection(face_group[i], sface_counter++);
      }
   }

   group_sface.ShiftUpI();
}


void ParMesh::BuildEdgeGroup(int ngroups, const Table& edge_element,
                             Table& group_sface)
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

void ParMesh::BuildVertexGroup(int ngroups, const Table& vert_element,
                               Table& group_svert)
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

void ParMesh::BuildSharedFaceElems(int nfaces, const Mesh& mesh,
                                   int* partitioning,
                                   STable3D* faces_tbl,
                                   Array<int>& face_group,
                                   Array<int>& vert_global_local)
{
   shared_faces.SetSize(nfaces);
   sface_lface. SetSize(nfaces);

   if (Dim == 3)
   {
      int sface_counter = 0;
      for (int i = 0; i < face_group.Size(); i++)
      {
         if (face_group[i] >= 0)
         {
            shared_faces[sface_counter] = mesh.GetFace(i)->Duplicate(this);
            int *v = shared_faces[sface_counter]->GetVertices();
            int nv = shared_faces[sface_counter]->GetNVertices();
            for (int j = 0; j < nv; j++)
            {
               v[j] = vert_global_local[v[j]];
            }
            switch (shared_faces[sface_counter]->GetType())
            {
               case Element::TRIANGLE:
                  sface_lface[sface_counter] = (*faces_tbl)(v[0], v[1], v[2]);
                  // mark the shared face for refinement by reorienting
                  // it according to the refinement flag in the tetrahedron
                  // to which this shared face belongs to.
                  {
                     int lface = sface_lface[sface_counter];
                     Tetrahedron *tet =
                        (Tetrahedron *)(elements[faces_info[lface].Elem1No]);
                     tet->GetMarkedFace(faces_info[lface].Elem1Inf/64, v);
                     // flip the shared face in the processor that owns the
                     // second element (in 'mesh')
                     {
                        int gl_el1, gl_el2;
                        mesh.GetFaceElements(i, &gl_el1, &gl_el2);
                        if (MyRank == partitioning[gl_el2])
                        {
                           std::swap(v[0], v[1]);
                        }
                     }
                  }
                  break;
               case Element::QUADRILATERAL:
                  sface_lface[sface_counter] =
                     (*faces_tbl)(v[0], v[1], v[2], v[3]);
                  break;
            }
            sface_counter++;
         }
      }

      delete faces_tbl;
   }
}

void ParMesh::BuildSharedEdgeElems(int nedges, Mesh& mesh,
                                   Array<int>& vert_global_local,
                                   Table* edge_element)
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



            if (sedge_ledge[sedge_counter] < 0)
            {
               cerr << "\n\n\n" << MyRank << ": ParMesh::ParMesh: "
                    << "ERROR in v_to_v\n\n" << endl;
               mfem_error();
            }

            sedge_counter++;
         }
      }
   }
}

void ParMesh::BuildSharedVertMapping(int nvert, Table* vert_element,
                                     Array<int>& vert_global_local)
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

// Fills out vert_global_local

int ParMesh::BuildVertGlobalLocal(Mesh& mesh, int* partitioning,
                                  Array<int>& vert_global_local)
{
   vert_global_local = -1;

   int vert_counter = 0;
   for (int i = 0; i < mesh.GetNE(); i++)
   {
      if (partitioning[i] == MyRank)
      {
         Array<int> vert;
         mesh.GetElementVertices(i, vert);
         for (int j = 0; j < vert.Size(); j++)
            if (vert_global_local[vert[j]] < 0)
            {
               vert_global_local[vert[j]] = vert_counter++;
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

   return vert_counter;
}

int ParMesh::BuildLocalVertices(const mfem::Mesh &mesh,
                                const int* partitioning,
                                Array<int> &vert_global_local)
{
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
   int nelems = 0;
   for (int i = 0; i < mesh.GetNE(); i++)
   {
      if (partitioning[i] == MyRank) { nelems++; }
   }

   elements.SetSize(nelems);

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
         element_counter++;
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
         int edge = mesh.GetBdrElementEdgeIndex(i);
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
         int edge = mesh.GetBdrElementEdgeIndex(i);
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

void ParMesh::GroupEdge(int group, int i, int &edge, int &o)
{
   int sedge = group_sedge.GetRow(group-1)[i];
   edge = sedge_ledge[sedge];
   int *v = shared_edges[sedge]->GetVertices();
   o = (v[0] < v[1]) ? (+1) : (-1);
}

void ParMesh::GroupFace(int group, int i, int &face, int &o)
{
   int sface = group_sface.GetRow(group-1)[i];
   face = sface_lface[sface];
   // face gives the base orientation
   if (faces[face]->GetType() == Element::TRIANGLE)
   {
      o = GetTriOrientation(faces[face]->GetVertices(),
                            shared_faces[sface]->GetVertices());
   }
   if (faces[face]->GetType() == Element::QUADRILATERAL)
   {
      o = GetQuadOrientation(faces[face]->GetVertices(),
                             shared_faces[sface]->GetVertices());
   }
}

void ParMesh::MarkTetMeshForRefinement(DSTable &v_to_v)
{
   Array<int> order;
   GetEdgeOrdering(v_to_v, order); // local edge ordering

   // create a GroupCommunicator on the shared edges
   GroupCommunicator sedge_comm(gtopo);
   {
      // initialize sedge_comm
      Table &gr_sedge = sedge_comm.GroupLDofTable(); // differs from group_sedge
      gr_sedge.SetDims(GetNGroups(), shared_edges.Size());
      gr_sedge.GetI()[0] = 0;
      for (int gr = 1; gr <= GetNGroups(); gr++)
      {
         gr_sedge.GetI()[gr] = group_sedge.GetI()[gr-1];
      }
      for (int k = 0; k < shared_edges.Size(); k++)
      {
         gr_sedge.GetJ()[k] = group_sedge.GetJ()[k];
      }
      sedge_comm.Finalize();
   }

   Array<int> sedge_ord(shared_edges.Size());
   Array<Pair<int,int> > sedge_ord_map(shared_edges.Size());
   for (int k = 0; k < shared_edges.Size(); k++)
   {
      sedge_ord[k] = order[sedge_ledge[group_sedge.GetJ()[k]]];
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
         int sedge_from = group_sedge.GetJ()[k+j];
         sedge_ord[k+j] = order[sedge_ledge[sedge_from]];
      }
      std::sort(&sedge_ord[k], &sedge_ord[k] + n);
      for (int j = 0; j < n; j++)
      {
         int sedge_to = group_sedge.GetJ()[k+sedge_ord_map[j].two];
         order[sedge_ledge[sedge_to]] = sedge_ord[k+j];
      }
      k += n;
   }

#ifdef MFEM_DEBUG
   {
      Array<Pair<int, double> > ilen_len(order.Size());

      for (int i = 0; i < NumOfVertices; i++)
      {
         for (DSTable::RowIterator it(v_to_v, i); !it; ++it)
         {
            int j = it.Index();
            ilen_len[j].one = order[j];
            ilen_len[j].two = GetLength(i, it.Column());
         }
      }

      SortPairs<int, double>(ilen_len, order.Size());

      double d_max = 0.;
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
      double glob_d_max;
      MPI_Reduce(&d_max, &glob_d_max, 1, MPI_DOUBLE, MPI_MAX, 0, MyComm);
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

   for (int i = 0; i < shared_faces.Size(); i++)
   {
      if (shared_faces[i]->GetType() == Element::TRIANGLE)
      {
         shared_faces[i]->MarkEdge(v_to_v, order);
      }
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

void ParMesh::GetFaceSplittings(Element *face, const HashTable<Hashed2> &v_to_v,
                                Array<unsigned> &codes)
{
   const int *v = face->GetVertices();
   typedef Triple<int,int,int> face_t;
   Array<face_t> face_stack;

   unsigned code = 0;
   face_stack.Append(face_t(v[0], v[1], v[2]));
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

void ParMesh::GenerateOffsets(int N, HYPRE_Int loc_sizes[],
                              Array<HYPRE_Int> *offsets[]) const
{
   if (HYPRE_AssumedPartitionCheck())
   {
      Array<HYPRE_Int> temp(N);
      MPI_Scan(loc_sizes, temp.GetData(), N, HYPRE_MPI_INT, MPI_SUM, MyComm);
      for (int i = 0; i < N; i++)
      {
         offsets[i]->SetSize(3);
         (*offsets[i])[0] = temp[i] - loc_sizes[i];
         (*offsets[i])[1] = temp[i];
      }
      MPI_Bcast(temp.GetData(), N, HYPRE_MPI_INT, NRanks-1, MyComm);
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
      Array<HYPRE_Int> temp(N*NRanks);
      MPI_Allgather(loc_sizes, N, HYPRE_MPI_INT, temp.GetData(), N,
                    HYPRE_MPI_INT, MyComm);
      for (int i = 0; i < N; i++)
      {
         Array<HYPRE_Int> &offs = *offsets[i];
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

void ParMesh::GetFaceNbrElementTransformation(
   int i, IsoparametricTransformation *ElTr)
{
   DenseMatrix &pointmat = ElTr->GetPointMat();
   Element *elem = face_nbr_elements[i];

   ElTr->Attribute = elem->GetAttribute();
   ElTr->ElementNo = NumOfElements + i;

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

      ElTr->SetFE(GetTransformationFEforElementType(elem->GetType()));
   }
   else
   {
      Array<int> vdofs;
      ParGridFunction *pNodes = dynamic_cast<ParGridFunction *>(Nodes);
      if (pNodes)
      {
         pNodes->ParFESpace()->GetFaceNbrElementVDofs(i, vdofs);
         int n = vdofs.Size()/spaceDim;
         pointmat.SetSize(spaceDim, n);
         for (int k = 0; k < spaceDim; k++)
         {
            for (int j = 0; j < n; j++)
            {
               pointmat(k,j) = (pNodes->FaceNbrData())(vdofs[n*k+j]);
            }
         }

         ElTr->SetFE(pNodes->ParFESpace()->GetFaceNbrFE(i));
      }
      else
      {
         MFEM_ABORT("Nodes are not ParGridFunction!");
      }
   }
   ElTr->FinalizeTransformation();
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

void ParMesh::ExchangeFaceNbrData()
{
   if (have_face_nbr_data)
   {
      return;
   }

   if (Nonconforming())
   {
      // with ParNCMesh we can set up face neighbors without communication
      pncmesh->GetFaceNeighbors(*this);
      have_face_nbr_data = true;

      ExchangeFaceNbrNodes();
      return;
   }

   Table *gr_sface;
   int   *s2l_face;
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
      gr_sface = &group_sface;
      s2l_face = sface_lface;
   }

   int num_face_nbrs = 0;
   for (int g = 1; g < GetNGroups(); g++)
      if (gr_sface->RowSize(g-1) > 0)
      {
         num_face_nbrs++;
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
         if (gr_sface->RowSize(g-1) > 0)
         {
#ifdef MFEM_DEBUG
            if (gtopo.GetGroupSize(g) != 2)
               mfem_error("ParMesh::ExchangeFaceNbrData() : "
                          "group size is not 2!");
#endif
            const int *nbs = gtopo.GetGroup(g);
            int lproc = (nbs[0]) ? nbs[0] : nbs[1];
            rank_group[counter].one = gtopo.GetNeighborRank(lproc);
            rank_group[counter].two = g;
            counter++;
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

            send_face_nbr_elemdata.AddColumnsInRow(fn, nv + 2);
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
         }
         send_face_nbr_facedata.AddConnection(fn, el);
         int info = faces_info[lface].Elem1Inf;
         // change the orientation in info to be relative to the shared face
         //   in 1D and 2D keep the orientation equal to 0
         if (Dim == 3)
         {
            Element *lf = faces[lface];
            const int *sf_v = shared_faces[sface[i]]->GetVertices();

            if  (lf->GetGeometryType() == Geometry::TRIANGLE)
            {
               info += GetTriOrientation(sf_v, lf->GetVertices());
            }
            else
            {
               info += GetQuadOrientation(sf_v, lf->GetVertices());
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
         const int nv = elements[el]->GetNVertices();
         elemdata += 2; // skip the attribute and the geometry type
         for (int j = 0; j < nv; j++)
         {
            elemdata[j] = vertex_marker[elemdata[j]];
         }
         elemdata += nv;

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
         face_nbr_elements[elem_off++] = el;
      }
   }

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
         int lface = s2l_face[sface[i]];
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
            Element *lf = faces[lface];
            const int *sf_v = shared_faces[sface[i]]->GetVertices();

            if  (lf->GetGeometryType() == Geometry::TRIANGLE)
            {
               // apply the nbr_ori to sf_v to get nbr_v
               const int *perm = tri_t::Orient[nbr_ori];
               for (int j = 0; j < 3; j++)
               {
                  nbr_v[perm[j]] = sf_v[j];
               }
               // get the orientation of nbr_v w.r.t. the local face
               nbr_ori = GetTriOrientation(lf->GetVertices(), nbr_v);
            }
            else
            {
               // apply the nbr_ori to sf_v to get nbr_v
               const int *perm = quad_t::Orient[nbr_ori];
               for (int j = 0; j < 4; j++)
               {
                  nbr_v[perm[j]] = sf_v[j];
               }
               // get the orientation of nbr_v w.r.t. the local face
               nbr_ori = GetQuadOrientation(lf->GetVertices(), nbr_v);
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

   have_face_nbr_data = true;

   ExchangeFaceNbrNodes();
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
                   MPI_DOUBLE, nbr_rank, tag, MyComm, &send_requests[fn]);

         MPI_Irecv(face_nbr_vertices[face_nbr_vertices_offset[fn]](),
                   3*(face_nbr_vertices_offset[fn+1] -
                      face_nbr_vertices_offset[fn]),
                   MPI_DOUBLE, nbr_rank, tag, MyComm, &recv_requests[fn]);
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

ElementTransformation* ParMesh::GetGhostFaceTransformation(
   FaceElementTransformations* FETr, int face_type, int face_geom)
{
   // calculate composition of FETr->Loc1 and FETr->Elem1
   DenseMatrix &face_pm = FaceTransformation.GetPointMat();
   if (Nodes == NULL)
   {
      FETr->Elem1->Transform(FETr->Loc1.Transf.GetPointMat(), face_pm);
      FaceTransformation.SetFE(GetTransformationFEforElementType(face_type));
   }
   else
   {
      const FiniteElement* face_el =
         Nodes->FESpace()->GetTraceElement(FETr->Elem1No, face_geom);

#if 0 // TODO: handle the case of non-interpolatory Nodes
      DenseMatrix I;
      face_el->Project(Transformation.GetFE(), FETr->Loc1.Transf, I);
      MultABt(Transformation.GetPointMat(), I, pm_face);
#else
      IntegrationRule eir(face_el->GetDof());
      FETr->Loc1.Transform(face_el->GetNodes(), eir);
      Nodes->GetVectorValues(*FETr->Elem1, eir, face_pm);
#endif
      FaceTransformation.SetFE(face_el);
   }
   FaceTransformation.FinalizeTransformation();
   return &FaceTransformation;
}

FaceElementTransformations *ParMesh::
GetSharedFaceTransformations(int sf, bool fill2)
{
   int FaceNo = GetSharedFace(sf);

   FaceInfo &face_info = faces_info[FaceNo];

   bool is_slave = Nonconforming() && IsSlaveFace(face_info);
   bool is_ghost = Nonconforming() && FaceNo >= GetNumFaces();

   NCFaceInfo* nc_info = NULL;
   if (is_slave) { nc_info = &nc_faces_info[face_info.NCFace]; }

   int local_face = is_ghost ? nc_info->MasterFace : FaceNo;
   int face_type = GetFaceElementType(local_face);
   int face_geom = GetFaceGeometryType(local_face);

   // setup the transformation for the first element
   FaceElemTr.Elem1No = face_info.Elem1No;
   GetElementTransformation(FaceElemTr.Elem1No, &Transformation);
   FaceElemTr.Elem1 = &Transformation;

   // setup the transformation for the second (neighbor) element
   if (fill2)
   {
      FaceElemTr.Elem2No = -1 - face_info.Elem2No;
      GetFaceNbrElementTransformation(FaceElemTr.Elem2No, &Transformation2);
      FaceElemTr.Elem2 = &Transformation2;
   }
   else
   {
      FaceElemTr.Elem2No = -1;
   }

   // setup the face transformation if the face is not a ghost
   FaceElemTr.FaceGeom = face_geom;
   if (!is_ghost)
   {
      FaceElemTr.Face = GetFaceTransformation(FaceNo);
      // NOTE: The above call overwrites FaceElemTr.Loc1
   }

   // setup Loc1 & Loc2
   int elem_type = GetElementType(face_info.Elem1No);
   GetLocalFaceTransformation(face_type, elem_type, FaceElemTr.Loc1.Transf,
                              face_info.Elem1Inf);

   if (fill2)
   {
      elem_type = face_nbr_elements[FaceElemTr.Elem2No]->GetType();
      GetLocalFaceTransformation(face_type, elem_type, FaceElemTr.Loc2.Transf,
                                 face_info.Elem2Inf);
   }

   // adjust Loc1 or Loc2 of the master face if this is a slave face
   if (is_slave)
   {
      // is a ghost slave? -> master not a ghost -> choose Elem1 local transf
      // not a ghost slave? -> master is a ghost -> choose Elem2 local transf
      IsoparametricTransformation &loctr =
         is_ghost ? FaceElemTr.Loc1.Transf : FaceElemTr.Loc2.Transf;

      if (is_ghost || fill2)
      {
         ApplyLocalSlaveTransformation(loctr, face_info);
      }

      if (face_type == Element::SEGMENT && fill2)
      {
         // fix slave orientation in 2D: flip Loc2 to match Loc1 and Face
         DenseMatrix &pm = FaceElemTr.Loc2.Transf.GetPointMat();
         std::swap(pm(0,0), pm(0,1));
         std::swap(pm(1,0), pm(1,1));
      }
   }

   // for ghost faces we need a special version of GetFaceTransformation
   if (is_ghost)
   {
      FaceElemTr.Face =
         GetGhostFaceTransformation(&FaceElemTr, face_type, face_geom);
   }

   return &FaceElemTr;
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
      return shared.conforming.size() + shared.slaves.size();
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
      int csize = (int) shared.conforming.size();
      return sface < csize
             ? shared.conforming[sface].index
             : shared.slaves[sface - csize].index;
   }
}

void ParMesh::ReorientTetMesh()
{
   if (Dim != 3 || !(meshgen & 1))
   {
      return;
   }

   Mesh::ReorientTetMesh();

   int *v;

   // The local edge and face numbering is changed therefore we need to
   // update sedge_ledge and sface_lface.
   {
      DSTable v_to_v(NumOfVertices);
      GetVertexToVertexTable(v_to_v);
      for (int i = 0; i < shared_edges.Size(); i++)
      {
         v = shared_edges[i]->GetVertices();
         sedge_ledge[i] = v_to_v(v[0], v[1]);
      }
   }

   // Rotate shared faces and update sface_lface.
   // Note that no communication is needed to ensure that the shared
   // faces are rotated in the same way in both processors. This is
   // automatic due to various things, e.g. the global to local vertex
   // mapping preserves the global order; also the way new vertices
   // are introduced during refinement is essential.
   {
      STable3D *faces_tbl = GetFacesTable();
      for (int i = 0; i < shared_faces.Size(); i++)
         if (shared_faces[i]->GetType() == Element::TRIANGLE)
         {
            v = shared_faces[i]->GetVertices();

            Rotate3(v[0], v[1], v[2]);

            sface_lface[i] = (*faces_tbl)(v[0], v[1], v[2]);
         }
      delete faces_tbl;
   }
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
         const int faces_in_group = GroupNFaces(i+1);
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
               const int *group_faces = group_sface.GetRow(i);
               const int faces_in_group = group_sface.RowSize(i);
               // it is enough to communicate through the faces
               if (faces_in_group == 0) { continue; }

               face_splittings[i].SetSize(0);
               for (int j = 0; j < faces_in_group; j++)
               {
                  GetFaceSplittings(shared_faces[group_faces[j]], v_to_v,
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
               const int *group_faces = group_sface.GetRow(i);
               const int faces_in_group = group_sface.RowSize(i);
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
                  const int *v = shared_faces[group_faces[j]]->GetVertices();
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

      DeleteLazyTables();

      // 5. Update the groups after refinement.
      if (el_to_face != NULL)
      {
         RefineGroups(v_to_v);
         // GetElementToFaceTable(); // Called by RefineGroups
         GenerateFaces();
      }
      NumOfVertices = vertices.Size();

      // 6. Update element-to-edge relations.
      if (el_to_edge != NULL)
      {
         NumOfEdges = GetElementToEdgeTable(*el_to_edge, be_to_edge);
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
                        for (int c = 0; c < 2; c++)
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

      DeleteLazyTables();

      // 5a. Update the groups after refinement.
      RefineGroups(v_to_v, middle);

      // 6. Free the allocated memory.
      delete [] edge1;
      delete [] edge2;
      delete [] middle;

      if (el_to_edge != NULL)
      {
         NumOfEdges = GetElementToEdgeTable(*el_to_edge, be_to_edge);
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

         CoarseFineTr.embeddings[i] = Embedding(i, 1);
         CoarseFineTr.embeddings[new_e] = Embedding(i, 2);
      }

      static double seg_children[3*2] = { 0.0,1.0, 0.0,0.5, 0.5,1.0 };
      CoarseFineTr.point_matrices.UseExternalData(seg_children, 1, 2, 3);

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
      MFEM_ABORT("ParMesh::NonconformingRefinement: NURBS meshes are not "
                 "supported. Project the NURBS to Nodes first.");
   }

   if (!pncmesh)
   {
      MFEM_ABORT("Can't convert conforming ParMesh to nonconforming ParMesh "
                 "(you need to initialize the ParMesh from a nonconforming "
                 "serial Mesh)");
   }

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

   // now swap the meshes, the second mesh will become the old coarse mesh
   // and this mesh will be the new fine mesh
   Swap(*pmesh2, false);

   delete pmesh2; // NOTE: old face neighbors destroyed here

   GenerateNCFaceInfo();

   last_operation = Mesh::REFINE;
   sequence++;

   if (Nodes) // update/interpolate curved mesh
   {
      Nodes->FESpace()->Update();
      Nodes->Update();
   }
}

bool ParMesh::NonconformingDerefinement(Array<double> &elem_error,
                                        double threshold, int nc_limit, int op)
{
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

      const int* fine = dt.GetRow(i);
      int size = dt.RowSize(i);

      double error = 0.0;
      for (int j = 0; j < size; j++)
      {
         MFEM_VERIFY(fine[j] < elem_error.Size(), "");

         double err_fine = elem_error[fine[j]];
         switch (op)
         {
            case 0: error = std::min(error, err_fine); break;
            case 1: error += err_fine; break;
            case 2: error = std::max(error, err_fine); break;
         }
      }

      if (error < threshold) { derefs.Append(i); }
   }

   long glob_size = ReduceInt(derefs.Size());
   if (glob_size)
   {
      DerefineMesh(derefs);
      return true;
   }

   return false;
}

void ParMesh::Rebalance()
{
   if (Conforming())
   {
      MFEM_ABORT("Load balancing is currently not supported for conforming"
                 " meshes.");
   }

   DeleteFaceNbrData();

   pncmesh->Rebalance();

   ParMesh* pmesh2 = new ParMesh(*pncmesh);
   pncmesh->OnMeshUpdated(pmesh2);

   attributes.Copy(pmesh2->attributes);
   bdr_attributes.Copy(pmesh2->bdr_attributes);

   Swap(*pmesh2, false);
   delete pmesh2;

   GenerateNCFaceInfo();

   last_operation = Mesh::REBALANCE;
   sequence++;

   if (Nodes) // redistribute curved mesh
   {
      Nodes->FESpace()->Update();
      Nodes->Update();
   }
}

void ParMesh::RefineGroups(const DSTable &v_to_v, int *middle)
{
   int i, attr, newv[3], ind, f_ind, *v;

   int group;
   Array<int> group_verts, group_edges, group_faces;

   // To update the groups after a refinement, we observe that:
   // - every (new and old) vertex, edge and face belongs to exactly one group
   // - the refinement does not create new groups
   // - a new vertex appears only as the middle of a refined edge
   // - a face can be refined 2, 3 or 4 times producing new edges and faces

   int *I_group_svert, *J_group_svert;
   int *I_group_sedge, *J_group_sedge;
   int *I_group_sface, *J_group_sface;

   I_group_svert = new int[GetNGroups()+1];
   I_group_sedge = new int[GetNGroups()+1];
   if (Dim == 3)
   {
      I_group_sface = new int[GetNGroups()+1];
   }
   else
   {
      I_group_sface = NULL;
   }

   I_group_svert[0] = I_group_svert[1] = 0;
   I_group_sedge[0] = I_group_sedge[1] = 0;
   if (Dim == 3)
   {
      I_group_sface[0] = I_group_sface[1] = 0;
   }

   // overestimate the size of the J arrays
   if (Dim == 3)
   {
      J_group_svert = new int[group_svert.Size_of_connections()
                              + group_sedge.Size_of_connections()];
      J_group_sedge = new int[2*group_sedge.Size_of_connections()
                              + 3*group_sface.Size_of_connections()];
      J_group_sface = new int[4*group_sface.Size_of_connections()];
   }
   else if (Dim == 2)
   {
      J_group_svert = new int[group_svert.Size_of_connections()
                              + group_sedge.Size_of_connections()];
      J_group_sedge = new int[2*group_sedge.Size_of_connections()];
      J_group_sface = NULL;
   }
   else
   {
      J_group_svert = J_group_sedge = J_group_sface = NULL;
   }

   for (group = 0; group < GetNGroups()-1; group++)
   {
      // Get the group shared objects
      group_svert.GetRow(group, group_verts);
      group_sedge.GetRow(group, group_edges);
      group_sface.GetRow(group, group_faces);

      // Check which edges have been refined
      for (i = 0; i < group_sedge.RowSize(group); i++)
      {
         v = shared_edges[group_edges[i]]->GetVertices();
         ind = middle[v_to_v(v[0], v[1])];
         if (ind != -1)
         {
            // add a vertex
            group_verts.Append(svert_lvert.Append(ind)-1);
            // update the edges
            attr = shared_edges[group_edges[i]]->GetAttribute();
            shared_edges.Append(new Segment(v[1], ind, attr));
            group_edges.Append(sedge_ledge.Append(-1)-1);
            v[1] = ind;
         }
      }

      // Check which faces have been refined
      for (i = 0; i < group_sface.RowSize(group); i++)
      {
         v = shared_faces[group_faces[i]]->GetVertices();
         ind = middle[v_to_v(v[0], v[1])];
         if (ind != -1)
         {
            attr = shared_faces[group_faces[i]]->GetAttribute();
            // add the refinement edge
            shared_edges.Append(new Segment(v[2], ind, attr));
            group_edges.Append(sedge_ledge.Append(-1)-1);
            // add a face
            f_ind = group_faces.Size();
            shared_faces.Append(new Triangle(v[1], v[2], ind, attr));
            group_faces.Append(sface_lface.Append(-1)-1);
            newv[0] = v[2]; newv[1] = v[0]; newv[2] = ind;
            shared_faces[group_faces[i]]->SetVertices(newv);

            // check if the left face has also been refined
            // v = shared_faces[group_faces[i]]->GetVertices();
            ind = middle[v_to_v(v[0], v[1])];
            if (ind != -1)
            {
               // add the refinement edge
               shared_edges.Append(new Segment(v[2], ind, attr));
               group_edges.Append(sedge_ledge.Append(-1)-1);
               // add a face
               shared_faces.Append(new Triangle(v[1], v[2], ind, attr));
               group_faces.Append(sface_lface.Append(-1)-1);
               newv[0] = v[2]; newv[1] = v[0]; newv[2] = ind;
               shared_faces[group_faces[i]]->SetVertices(newv);
            }

            // check if the right face has also been refined
            v = shared_faces[group_faces[f_ind]]->GetVertices();
            ind = middle[v_to_v(v[0], v[1])];
            if (ind != -1)
            {
               // add the refinement edge
               shared_edges.Append(new Segment(v[2], ind, attr));
               group_edges.Append(sedge_ledge.Append(-1)-1);
               // add a face
               shared_faces.Append(new Triangle(v[1], v[2], ind, attr));
               group_faces.Append(sface_lface.Append(-1)-1);
               newv[0] = v[2]; newv[1] = v[0]; newv[2] = ind;
               shared_faces[group_faces[f_ind]]->SetVertices(newv);
            }
         }
      }

      I_group_svert[group+1] = I_group_svert[group] + group_verts.Size();
      I_group_sedge[group+1] = I_group_sedge[group] + group_edges.Size();
      if (Dim == 3)
      {
         I_group_sface[group+1] = I_group_sface[group] + group_faces.Size();
      }

      int *J;
      J = J_group_svert+I_group_svert[group];
      for (i = 0; i < group_verts.Size(); i++)
      {
         J[i] = group_verts[i];
      }
      J = J_group_sedge+I_group_sedge[group];
      for (i = 0; i < group_edges.Size(); i++)
      {
         J[i] = group_edges[i];
      }
      if (Dim == 3)
      {
         J = J_group_sface+I_group_sface[group];
         for (i = 0; i < group_faces.Size(); i++)
         {
            J[i] = group_faces[i];
         }
      }
   }

   // Fix the local numbers of shared edges and faces
   {
      DSTable new_v_to_v(NumOfVertices);
      GetVertexToVertexTable(new_v_to_v);
      for (i = 0; i < shared_edges.Size(); i++)
      {
         v = shared_edges[i]->GetVertices();
         sedge_ledge[i] = new_v_to_v(v[0], v[1]);
      }
   }
   if (Dim == 3)
   {
      STable3D *faces_tbl = GetElementToFaceTable(1);
      for (i = 0; i < shared_faces.Size(); i++)
      {
         v = shared_faces[i]->GetVertices();
         sface_lface[i] = (*faces_tbl)(v[0], v[1], v[2]);
      }
      delete faces_tbl;
   }

   group_svert.SetIJ(I_group_svert, J_group_svert);
   group_sedge.SetIJ(I_group_sedge, J_group_sedge);
   if (Dim == 3)
   {
      group_sface.SetIJ(I_group_sface, J_group_sface);
   }
}

void ParMesh::RefineGroups(const HashTable<Hashed2> &v_to_v)
{
   int i, attr, ind, *v;

   int group;
   Array<int> group_verts, group_edges, group_faces;

   // To update the groups after a refinement, we observe that:
   // - every (new and old) vertex, edge and face belongs to exactly one group
   // - the refinement does not create new groups
   // - a new vertex appears only as the middle of a refined edge
   // - a face can be refined multiple times producing new edges and faces

   Array<Segment *> sedge_stack;
   Array<Triangle *> sface_stack;

   Array<int> I_group_svert, J_group_svert;
   Array<int> I_group_sedge, J_group_sedge;
   Array<int> I_group_sface, J_group_sface;

   I_group_svert.SetSize(GetNGroups()+1);
   I_group_sedge.SetSize(GetNGroups()+1);
   if (Dim == 3)
   {
      I_group_sface.SetSize(GetNGroups()+1);
   }

   I_group_svert[0] = I_group_svert[1] = 0;
   I_group_sedge[0] = I_group_sedge[1] = 0;
   if (Dim == 3)
   {
      I_group_sface[0] = I_group_sface[1] = 0;
   }

   for (group = 0; group < GetNGroups()-1; group++)
   {
      // Get the group shared objects
      group_svert.GetRow(group, group_verts);
      group_sedge.GetRow(group, group_edges);
      group_sface.GetRow(group, group_faces);

      // Check which edges have been refined
      for (i = 0; i < group_sedge.RowSize(group); i++)
      {
         v = shared_edges[group_edges[i]]->GetVertices();
         ind = v_to_v.FindId(v[0], v[1]);
         if (ind == -1) { continue; }

         // This shared edge is refined: walk the whole refinement tree
         attr = shared_edges[group_edges[i]]->GetAttribute();
         do
         {
            ind += NumOfVertices;
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
               ind += NumOfVertices;
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

      // Check which faces have been refined
      for (i = 0; i < group_sface.RowSize(group); i++)
      {
         v = shared_faces[group_faces[i]]->GetVertices();
         ind = v_to_v.FindId(v[0], v[1]);
         if (ind == -1) { continue; }

         // This shared face is refined: walk the whole refinement tree
         attr = shared_faces[group_faces[i]]->GetAttribute();
         const int edge_attr = 1;
         do
         {
            ind += NumOfVertices;
            // Add the refinement edge to the edge stack
            sedge_stack.Append(new Segment(v[2], ind, edge_attr));
            // Put the right sub-triangle on top of the face stack
            sface_stack.Append(new Triangle(v[1], v[2], ind, attr));
            // The left sub-triangle replaces the original one
            v[1] = v[0]; v[0] = v[2]; v[2] = ind;
            ind = v_to_v.FindId(v[0], v[1]);
         }
         while (ind != -1);
         // Process all faces (triangles) in the face stack
         do
         {
            Triangle *st = sface_stack.Last();
            v = st->GetVertices();
            ind = v_to_v.FindId(v[0], v[1]);
            if (ind == -1)
            {
               // The triangle 'st' is not refined
               sface_stack.DeleteLast();
               // Add new shared face
               shared_faces.Append(st);
               group_faces.Append(sface_lface.Append(-1)-1);
            }
            else
            {
               // The triangle 'st' is refined
               ind += NumOfVertices;
               // Add the refinement edge to the edge stack
               sedge_stack.Append(new Segment(v[2], ind, edge_attr));
               // Put the left sub-triangle on top of the face stack
               sface_stack.Append(new Triangle(v[2], v[0], ind, attr));
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
               ind += NumOfVertices;
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

      I_group_svert[group+1] = I_group_svert[group] + group_verts.Size();
      I_group_sedge[group+1] = I_group_sedge[group] + group_edges.Size();
      if (Dim == 3)
      {
         I_group_sface[group+1] = I_group_sface[group] + group_faces.Size();
      }

      J_group_svert.Append(group_verts);
      J_group_sedge.Append(group_edges);
      if (Dim == 3)
      {
         J_group_sface.Append(group_faces);
      }
   }

   // Fix the local numbers of shared edges and faces: sedge_ledge, sface_lface
   {
      NumOfVertices = vertices.Size();
      DSTable new_v_to_v(NumOfVertices);
      GetVertexToVertexTable(new_v_to_v);
      for (i = 0; i < shared_edges.Size(); i++)
      {
         v = shared_edges[i]->GetVertices();
         sedge_ledge[i] = new_v_to_v(v[0], v[1]);
      }
   }
   if (Dim == 3)
   {
      STable3D *faces_tbl = GetElementToFaceTable(1);
      for (i = 0; i < shared_faces.Size(); i++)
      {
         v = shared_faces[i]->GetVertices();
         sface_lface[i] = (*faces_tbl)(v[0], v[1], v[2]);
      }
      delete faces_tbl;
   }

   group_svert.SetIJ(I_group_svert, J_group_svert);
   group_sedge.SetIJ(I_group_sedge, J_group_sedge);
   if (Dim == 3)
   {
      group_sface.SetIJ(I_group_sface, J_group_sface);
   }
   I_group_svert.LoseData(); J_group_svert.LoseData();
   I_group_sedge.LoseData(); J_group_sedge.LoseData();
   if (Dim == 3)
   {
      I_group_sface.LoseData(); J_group_sface.LoseData();
   }
}

void ParMesh::QuadUniformRefinement()
{
   DeleteFaceNbrData();

   int oedge = NumOfVertices;

   // call Mesh::QuadUniformRefinement so that it won't update the nodes
   {
      GridFunction *nodes = Nodes;
      Nodes = NULL;
      Mesh::QuadUniformRefinement();
      Nodes = nodes;
   }

   // update the groups
   {
      int i, attr, ind, *v;

      int group;
      Array<int> sverts, sedges;

      int *I_group_svert, *J_group_svert;
      int *I_group_sedge, *J_group_sedge;

      I_group_svert = new int[GetNGroups()+1];
      I_group_sedge = new int[GetNGroups()+1];

      I_group_svert[0] = I_group_svert[1] = 0;
      I_group_sedge[0] = I_group_sedge[1] = 0;

      // compute the size of the J arrays
      J_group_svert = new int[group_svert.Size_of_connections()
                              + group_sedge.Size_of_connections()];
      J_group_sedge = new int[2*group_sedge.Size_of_connections()];

      for (group = 0; group < GetNGroups()-1; group++)
      {
         // Get the group shared objects
         group_svert.GetRow(group, sverts);
         group_sedge.GetRow(group, sedges);

         // Process all the edges
         for (i = 0; i < group_sedge.RowSize(group); i++)
         {
            v = shared_edges[sedges[i]]->GetVertices();
            ind = oedge + sedge_ledge[sedges[i]];
            // add a vertex
            sverts.Append(svert_lvert.Append(ind)-1);
            // update the edges
            attr = shared_edges[sedges[i]]->GetAttribute();
            shared_edges.Append(new Segment(v[1], ind, attr));
            sedges.Append(sedge_ledge.Append(-1)-1);
            v[1] = ind;
         }

         I_group_svert[group+1] = I_group_svert[group] + sverts.Size();
         I_group_sedge[group+1] = I_group_sedge[group] + sedges.Size();

         int *J;
         J = J_group_svert+I_group_svert[group];
         for (i = 0; i < sverts.Size(); i++)
         {
            J[i] = sverts[i];
         }
         J = J_group_sedge+I_group_sedge[group];
         for (i = 0; i < sedges.Size(); i++)
         {
            J[i] = sedges[i];
         }
      }

      // Fix the local numbers of shared edges
      DSTable v_to_v(NumOfVertices);
      GetVertexToVertexTable(v_to_v);
      for (i = 0; i < shared_edges.Size(); i++)
      {
         v = shared_edges[i]->GetVertices();
         sedge_ledge[i] = v_to_v(v[0], v[1]);
      }

      group_svert.SetIJ(I_group_svert, J_group_svert);
      group_sedge.SetIJ(I_group_sedge, J_group_sedge);
   }

   UpdateNodes();
}

void ParMesh::HexUniformRefinement()
{
   DeleteFaceNbrData();

   int oedge = NumOfVertices;
   int oface = oedge + NumOfEdges;

   DSTable v_to_v(NumOfVertices);
   GetVertexToVertexTable(v_to_v);
   STable3D *faces_tbl = GetFacesTable();

   // call Mesh::HexUniformRefinement so that it won't update the nodes
   {
      GridFunction *nodes = Nodes;
      Nodes = NULL;
      Mesh::HexUniformRefinement();
      Nodes = nodes;
   }

   // update the groups
   {
      int i, attr, newv[4], ind, m[5];
      Array<int> v;

      int group;
      Array<int> group_verts, group_edges, group_faces;

      int *I_group_svert, *J_group_svert;
      int *I_group_sedge, *J_group_sedge;
      int *I_group_sface, *J_group_sface;

#if 0
      I_group_svert = new int[GetNGroups()+1];
      I_group_sedge = new int[GetNGroups()+1];
      I_group_sface = new int[GetNGroups()+1];

      I_group_svert[0] = I_group_svert[1] = 0;
      I_group_sedge[0] = I_group_sedge[1] = 0;
      I_group_sface[0] = I_group_sface[1] = 0;
#else
      I_group_svert = new int[GetNGroups()];
      I_group_sedge = new int[GetNGroups()];
      I_group_sface = new int[GetNGroups()];

      I_group_svert[0] = 0;
      I_group_sedge[0] = 0;
      I_group_sface[0] = 0;
#endif

      // compute the size of the J arrays
      J_group_svert = new int[group_svert.Size_of_connections()
                              + group_sedge.Size_of_connections()
                              + group_sface.Size_of_connections()];
      J_group_sedge = new int[2*group_sedge.Size_of_connections()
                              + 4*group_sface.Size_of_connections()];
      J_group_sface = new int[4*group_sface.Size_of_connections()];

      for (group = 0; group < GetNGroups()-1; group++)
      {
         // Get the group shared objects
         group_svert.GetRow(group, group_verts);
         group_sedge.GetRow(group, group_edges);
         group_sface.GetRow(group, group_faces);

         // Process the edges that have been refined
         for (i = 0; i < group_sedge.RowSize(group); i++)
         {
            shared_edges[group_edges[i]]->GetVertices(v);
            ind = oedge + v_to_v(v[0], v[1]);
            // add a vertex
            group_verts.Append(svert_lvert.Append(ind)-1);
            // update the edges
            attr = shared_edges[group_edges[i]]->GetAttribute();
            shared_edges.Append(new Segment(v[1], ind, attr));
            group_edges.Append(sedge_ledge.Append(-1)-1);
            newv[0] = v[0]; newv[1] = ind;
            shared_edges[group_edges[i]]->SetVertices(newv);
         }

         // Process the faces that have been refined
         for (i = 0; i < group_sface.RowSize(group); i++)
         {
            shared_faces[group_faces[i]]->GetVertices(v);
            m[0] = oface+(*faces_tbl)(v[0], v[1], v[2], v[3]);
            // add a vertex
            group_verts.Append(svert_lvert.Append(m[0])-1);
            // add the refinement edges
            attr = shared_faces[group_faces[i]]->GetAttribute();
            m[1] = oedge + v_to_v(v[0], v[1]);
            m[2] = oedge + v_to_v(v[1], v[2]);
            m[3] = oedge + v_to_v(v[2], v[3]);
            m[4] = oedge + v_to_v(v[3], v[0]);
            shared_edges.Append(new Segment(m[1], m[0], attr));
            group_edges.Append(sedge_ledge.Append(-1)-1);
            shared_edges.Append(new Segment(m[2], m[0], attr));
            group_edges.Append(sedge_ledge.Append(-1)-1);
            shared_edges.Append(new Segment(m[3], m[0], attr));
            group_edges.Append(sedge_ledge.Append(-1)-1);
            shared_edges.Append(new Segment(m[4], m[0], attr));
            group_edges.Append(sedge_ledge.Append(-1)-1);
            // update faces
            newv[0] = v[0]; newv[1] = m[1]; newv[2] = m[0]; newv[3] = m[4];
            shared_faces[group_faces[i]]->SetVertices(newv);
            shared_faces.Append(new Quadrilateral(m[1],v[1],m[2],m[0],attr));
            group_faces.Append(sface_lface.Append(-1)-1);
            shared_faces.Append(new Quadrilateral(m[0],m[2],v[2],m[3],attr));
            group_faces.Append(sface_lface.Append(-1)-1);
            shared_faces.Append(new Quadrilateral(m[4],m[0],m[3],v[3],attr));
            group_faces.Append(sface_lface.Append(-1)-1);
         }

         I_group_svert[group+1] = I_group_svert[group] + group_verts.Size();
         I_group_sedge[group+1] = I_group_sedge[group] + group_edges.Size();
         I_group_sface[group+1] = I_group_sface[group] + group_faces.Size();

         int *J;
         J = J_group_svert+I_group_svert[group];
         for (i = 0; i < group_verts.Size(); i++)
         {
            J[i] = group_verts[i];
         }
         J = J_group_sedge+I_group_sedge[group];
         for (i = 0; i < group_edges.Size(); i++)
         {
            J[i] = group_edges[i];
         }
         J = J_group_sface+I_group_sface[group];
         for (i = 0; i < group_faces.Size(); i++)
         {
            J[i] = group_faces[i];
         }
      }

      // Fix the local numbers of shared edges and faces
      DSTable new_v_to_v(NumOfVertices);
      GetVertexToVertexTable(new_v_to_v);
      for (i = 0; i < shared_edges.Size(); i++)
      {
         shared_edges[i]->GetVertices(v);
         sedge_ledge[i] = new_v_to_v(v[0], v[1]);
      }

      delete faces_tbl;
      faces_tbl = GetFacesTable();
      for (i = 0; i < shared_faces.Size(); i++)
      {
         shared_faces[i]->GetVertices(v);
         sface_lface[i] = (*faces_tbl)(v[0], v[1], v[2], v[3]);
      }
      delete faces_tbl;

      group_svert.SetIJ(I_group_svert, J_group_svert);
      group_sedge.SetIJ(I_group_sedge, J_group_sedge);
      group_sface.SetIJ(I_group_sface, J_group_sface);
   }

   UpdateNodes();
}

void ParMesh::NURBSUniformRefinement()
{
   if (MyRank == 0)
   {
      mfem::out << "\nParMesh::NURBSUniformRefinement : Not supported yet!\n";
   }
}

void ParMesh::PrintXG(std::ostream &out) const
{
   MFEM_ASSERT(Dim == spaceDim, "2D manifolds not supported");
   if (Dim == 3 && meshgen == 1)
   {
      int i, j, nv;
      const int *ind;

      out << "NETGEN_Neutral_Format\n";
      // print the vertices
      out << NumOfVertices << '\n';
      for (i = 0; i < NumOfVertices; i++)
      {
         for (j = 0; j < Dim; j++)
         {
            out << " " << vertices[i](j);
         }
         out << '\n';
      }

      // print the elements
      out << NumOfElements << '\n';
      for (i = 0; i < NumOfElements; i++)
      {
         nv = elements[i]->GetNVertices();
         ind = elements[i]->GetVertices();
         out << elements[i]->GetAttribute();
         for (j = 0; j < nv; j++)
         {
            out << " " << ind[j]+1;
         }
         out << '\n';
      }

      // print the boundary + shared faces information
      out << NumOfBdrElements + shared_faces.Size() << '\n';
      // boundary
      for (i = 0; i < NumOfBdrElements; i++)
      {
         nv = boundary[i]->GetNVertices();
         ind = boundary[i]->GetVertices();
         out << boundary[i]->GetAttribute();
         for (j = 0; j < nv; j++)
         {
            out << " " << ind[j]+1;
         }
         out << '\n';
      }
      // shared faces
      for (i = 0; i < shared_faces.Size(); i++)
      {
         nv = shared_faces[i]->GetNVertices();
         ind = shared_faces[i]->GetVertices();
         out << shared_faces[i]->GetAttribute();
         for (j = 0; j < nv; j++)
         {
            out << " " << ind[j]+1;
         }
         out << '\n';
      }
   }

   if (Dim == 3 && meshgen == 2)
   {
      int i, j, nv;
      const int *ind;

      out << "TrueGrid\n"
          << "1 " << NumOfVertices << " " << NumOfElements << " 0 0 0 0 0 0 0\n"
          << "0 0 0 1 0 0 0 0 0 0 0\n"
          << "0 0 " << NumOfBdrElements+shared_faces.Size()
          << " 0 0 0 0 0 0 0 0 0 0 0 0 0\n"
          << "0.0 0.0 0.0 0 0 0.0 0.0 0 0.0\n"
          << "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n";

      // print the vertices
      for (i = 0; i < NumOfVertices; i++)
         out << i+1 << " 0.0 " << vertices[i](0) << " " << vertices[i](1)
             << " " << vertices[i](2) << " 0.0\n";

      // print the elements
      for (i = 0; i < NumOfElements; i++)
      {
         nv = elements[i]->GetNVertices();
         ind = elements[i]->GetVertices();
         out << i+1 << " " << elements[i]->GetAttribute();
         for (j = 0; j < nv; j++)
         {
            out << " " << ind[j]+1;
         }
         out << '\n';
      }

      // print the boundary information
      for (i = 0; i < NumOfBdrElements; i++)
      {
         nv = boundary[i]->GetNVertices();
         ind = boundary[i]->GetVertices();
         out << boundary[i]->GetAttribute();
         for (j = 0; j < nv; j++)
         {
            out << " " << ind[j]+1;
         }
         out << " 1.0 1.0 1.0 1.0\n";
      }

      // print the shared faces information
      for (i = 0; i < shared_faces.Size(); i++)
      {
         nv = shared_faces[i]->GetNVertices();
         ind = shared_faces[i]->GetVertices();
         out << shared_faces[i]->GetAttribute();
         for (j = 0; j < nv; j++)
         {
            out << " " << ind[j]+1;
         }
         out << " 1.0 1.0 1.0 1.0\n";
      }
   }

   if (Dim == 2)
   {
      int i, j, attr;
      Array<int> v;

      out << "areamesh2\n\n";

      // print the boundary + shared edges information
      out << NumOfBdrElements + shared_edges.Size() << '\n';
      // boundary
      for (i = 0; i < NumOfBdrElements; i++)
      {
         attr = boundary[i]->GetAttribute();
         boundary[i]->GetVertices(v);
         out << attr << "     ";
         for (j = 0; j < v.Size(); j++)
         {
            out << v[j] + 1 << "   ";
         }
         out << '\n';
      }
      // shared edges
      for (i = 0; i < shared_edges.Size(); i++)
      {
         attr = shared_edges[i]->GetAttribute();
         shared_edges[i]->GetVertices(v);
         out << attr << "     ";
         for (j = 0; j < v.Size(); j++)
         {
            out << v[j] + 1 << "   ";
         }
         out << '\n';
      }

      // print the elements
      out << NumOfElements << '\n';
      for (i = 0; i < NumOfElements; i++)
      {
         attr = elements[i]->GetAttribute();
         elements[i]->GetVertices(v);

         out << attr << "   ";
         if ((j = GetElementType(i)) == Element::TRIANGLE)
         {
            out << 3 << "   ";
         }
         else if (j == Element::QUADRILATERAL)
         {
            out << 4 << "   ";
         }
         else if (j == Element::SEGMENT)
         {
            out << 2 << "   ";
         }
         for (j = 0; j < v.Size(); j++)
         {
            out << v[j] + 1 << "  ";
         }
         out << '\n';
      }

      // print the vertices
      out << NumOfVertices << '\n';
      for (i = 0; i < NumOfVertices; i++)
      {
         for (j = 0; j < Dim; j++)
         {
            out << vertices[i](j) << " ";
         }
         out << '\n';
      }
   }
   out.flush();
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

void ParMesh::Print(std::ostream &out) const
{
   bool print_shared = true;
   int i, j, shared_bdr_attr;
   Array<int> nc_shared_faces;

   if (NURBSext)
   {
      Printer(out); // does not print shared boundary
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
         for (unsigned i = 0; i < sfaces.conforming.size(); i++)
         {
            int index = sfaces.conforming[i].index;
            if (index < nfaces) { nc_shared_faces.Append(index); }
         }
         for (unsigned i = 0; i < sfaces.masters.size(); i++)
         {
            if (Dim == 2 && WantSkipSharedMaster(sfaces.masters[i])) { continue; }
            int index = sfaces.masters[i].index;
            if (index < nfaces) { nc_shared_faces.Append(index); }
         }
         for (unsigned i = 0; i < sfaces.slaves.size(); i++)
         {
            int index = sfaces.slaves[i].index;
            if (index < nfaces) { nc_shared_faces.Append(index); }
         }
      }
   }

   out << "MFEM mesh v1.0\n";

   // optional
   out <<
       "\n#\n# MFEM Geometry Types (see mesh/geom.hpp):\n#\n"
       "# POINT       = 0\n"
       "# SEGMENT     = 1\n"
       "# TRIANGLE    = 2\n"
       "# SQUARE      = 3\n"
       "# TETRAHEDRON = 4\n"
       "# CUBE        = 5\n"
       "#\n";

   out << "\ndimension\n" << Dim
       << "\n\nelements\n" << NumOfElements << '\n';
   for (i = 0; i < NumOfElements; i++)
   {
      PrintElement(elements[i], out);
   }

   int num_bdr_elems = NumOfBdrElements;
   if (print_shared && Dim > 1)
   {
      num_bdr_elems += s2l_face->Size();
   }
   out << "\nboundary\n" << num_bdr_elems << '\n';
   for (i = 0; i < NumOfBdrElements; i++)
   {
      PrintElement(boundary[i], out);
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
      for (i = 0; i < s2l_face->Size(); i++)
      {
         // Modify the attributes of the faces (not used otherwise?)
         faces[(*s2l_face)[i]]->SetAttribute(shared_bdr_attr);
         PrintElement(faces[(*s2l_face)[i]], out);
      }
   }
   out << "\nvertices\n" << NumOfVertices << '\n';
   if (Nodes == NULL)
   {
      out << spaceDim << '\n';
      for (i = 0; i < NumOfVertices; i++)
      {
         out << vertices[i](0);
         for (j = 1; j < spaceDim; j++)
         {
            out << ' ' << vertices[i](j);
         }
         out << '\n';
      }
      out.flush();
   }
   else
   {
      out << "\nnodes\n";
      Nodes->Save(out);
   }
}

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

void ParMesh::PrintAsOne(std::ostream &out)
{
   int i, j, k, p, nv_ne[2], &nv = nv_ne[0], &ne = nv_ne[1], vc;
   const int *v;
   MPI_Status status;
   Array<double> vert;
   Array<int> ints;

   if (MyRank == 0)
   {
      out << "MFEM mesh v1.0\n";

      // optional
      out <<
          "\n#\n# MFEM Geometry Types (see mesh/geom.hpp):\n#\n"
          "# POINT       = 0\n"
          "# SEGMENT     = 1\n"
          "# TRIANGLE    = 2\n"
          "# SQUARE      = 3\n"
          "# TETRAHEDRON = 4\n"
          "# CUBE        = 5\n"
          "#\n";

      out << "\ndimension\n" << Dim;
   }

   nv = NumOfElements;
   MPI_Reduce(&nv, &ne, 1, MPI_INT, MPI_SUM, 0, MyComm);
   if (MyRank == 0)
   {
      out << "\n\nelements\n" << ne << '\n';
      for (i = 0; i < NumOfElements; i++)
      {
         // processor number + 1 as attribute and geometry type
         out << 1 << ' ' << elements[i]->GetGeometryType();
         // vertices
         nv = elements[i]->GetNVertices();
         v  = elements[i]->GetVertices();
         for (j = 0; j < nv; j++)
         {
            out << ' ' << v[j];
         }
         out << '\n';
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
            out << p+1 << ' ' << ints[i];
            // vertices
            k = Geometries.GetVertices(ints[i++])->GetNPoints();
            for (j = 0; j < k; j++)
            {
               out << ' ' << vc + ints[i++];
            }
            out << '\n';
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
      ne += ((Dim == 2) ? shared_edges : shared_faces).Size();
   }
   else if (Dim > 1)
   {
      const NCMesh::NCList &list = pncmesh->GetSharedList(Dim - 1);
      ne += list.conforming.size() + list.masters.size() + list.slaves.size();
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
      Array<Element*> &shared = (Dim == 2) ? shared_edges : shared_faces;
      for (i = 0; i < shared.Size(); i++)
      {
         dump_element(shared[i], ints); ne++;
      }
   }
   else if (Dim > 1)
   {
      const NCMesh::NCList &list = pncmesh->GetSharedList(Dim - 1);
      const int nfaces = GetNumFaces();
      for (i = 0; i < (int) list.conforming.size(); i++)
      {
         int index = list.conforming[i].index;
         if (index < nfaces) { dump_element(faces[index], ints); ne++; }
      }
      for (i = 0; i < (int) list.masters.size(); i++)
      {
         int index = list.masters[i].index;
         if (index < nfaces) { dump_element(faces[index], ints); ne++; }
      }
      for (i = 0; i < (int) list.slaves.size(); i++)
      {
         int index = list.slaves[i].index;
         if (index < nfaces) { dump_element(faces[index], ints); ne++; }
      }
   }

   MPI_Reduce(&ne, &k, 1, MPI_INT, MPI_SUM, 0, MyComm);
   if (MyRank == 0)
   {
      out << "\nboundary\n" << k << '\n';
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
            out << p+1 << ' ' << ints[i];
            k = Geometries.NumVerts[ints[i++]];
            // vertices
            for (j = 0; j < k; j++)
            {
               out << ' ' << vc + ints[i++];
            }
            out << '\n';
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
      out << "\nvertices\n" << nv << '\n';
   }
   if (Nodes == NULL)
   {
      if (MyRank == 0)
      {
         out << spaceDim << '\n';
         for (i = 0; i < NumOfVertices; i++)
         {
            out << vertices[i](0);
            for (j = 1; j < spaceDim; j++)
            {
               out << ' ' << vertices[i](j);
            }
            out << '\n';
         }
         for (p = 1; p < NRanks; p++)
         {
            MPI_Recv(&nv, 1, MPI_INT, p, 448, MyComm, &status);
            vert.SetSize(nv*spaceDim);
            if (nv)
            {
               MPI_Recv(&vert[0], nv*spaceDim, MPI_DOUBLE, p, 449, MyComm, &status);
            }
            for (i = 0; i < nv; i++)
            {
               out << vert[i*spaceDim];
               for (j = 1; j < spaceDim; j++)
               {
                  out << ' ' << vert[i*spaceDim+j];
               }
               out << '\n';
            }
         }
         out.flush();
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
            MPI_Send(&vert[0], NumOfVertices*spaceDim, MPI_DOUBLE, 0, 449, MyComm);
         }
      }
   }
   else
   {
      if (MyRank == 0)
      {
         out << "\nnodes\n";
      }
      ParGridFunction *pnodes = dynamic_cast<ParGridFunction *>(Nodes);
      if (pnodes)
      {
         pnodes->SaveAsOne(out);
      }
      else
      {
         ParFiniteElementSpace *pfes =
            dynamic_cast<ParFiniteElementSpace *>(Nodes->FESpace());
         if (pfes)
         {
            // create a wrapper ParGridFunction
            ParGridFunction ParNodes(pfes, Nodes);
            ParNodes.SaveAsOne(out);
         }
         else
         {
            mfem_error("ParMesh::PrintAsOne : Nodes have no parallel info!");
         }
      }
   }
}

void ParMesh::PrintAsOneXG(std::ostream &out)
{
   MFEM_ASSERT(Dim == spaceDim, "2D Manifolds not supported.");
   if (Dim == 3 && meshgen == 1)
   {
      int i, j, k, nv, ne, p;
      const int *ind, *v;
      MPI_Status status;
      Array<double> vert;
      Array<int> ints;

      if (MyRank == 0)
      {
         out << "NETGEN_Neutral_Format\n";
         // print the vertices
         ne = NumOfVertices;
         MPI_Reduce(&ne, &nv, 1, MPI_INT, MPI_SUM, 0, MyComm);
         out << nv << '\n';
         for (i = 0; i < NumOfVertices; i++)
         {
            for (j = 0; j < Dim; j++)
            {
               out << " " << vertices[i](j);
            }
            out << '\n';
         }
         for (p = 1; p < NRanks; p++)
         {
            MPI_Recv(&nv, 1, MPI_INT, p, 444, MyComm, &status);
            vert.SetSize(Dim*nv);
            MPI_Recv(&vert[0], Dim*nv, MPI_DOUBLE, p, 445, MyComm, &status);
            for (i = 0; i < nv; i++)
            {
               for (j = 0; j < Dim; j++)
               {
                  out << " " << vert[Dim*i+j];
               }
               out << '\n';
            }
         }

         // print the elements
         nv = NumOfElements;
         MPI_Reduce(&nv, &ne, 1, MPI_INT, MPI_SUM, 0, MyComm);
         out << ne << '\n';
         for (i = 0; i < NumOfElements; i++)
         {
            nv = elements[i]->GetNVertices();
            ind = elements[i]->GetVertices();
            out << 1;
            for (j = 0; j < nv; j++)
            {
               out << " " << ind[j]+1;
            }
            out << '\n';
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
               out << p+1;
               for (j = 0; j < 4; j++)
               {
                  out << " " << k+ints[i*4+j]+1;
               }
               out << '\n';
            }
            k += nv;
         }
         // print the boundary + shared faces information
         nv = NumOfBdrElements + shared_faces.Size();
         MPI_Reduce(&nv, &ne, 1, MPI_INT, MPI_SUM, 0, MyComm);
         out << ne << '\n';
         // boundary
         for (i = 0; i < NumOfBdrElements; i++)
         {
            nv = boundary[i]->GetNVertices();
            ind = boundary[i]->GetVertices();
            out << 1;
            for (j = 0; j < nv; j++)
            {
               out << " " << ind[j]+1;
            }
            out << '\n';
         }
         // shared faces
         for (i = 0; i < shared_faces.Size(); i++)
         {
            nv = shared_faces[i]->GetNVertices();
            ind = shared_faces[i]->GetVertices();
            out << 1;
            for (j = 0; j < nv; j++)
            {
               out << " " << ind[j]+1;
            }
            out << '\n';
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
               out << p+1;
               for (j = 0; j < 3; j++)
               {
                  out << " " << k+ints[i*3+j]+1;
               }
               out << '\n';
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
         MPI_Send(&vert[0], Dim*NumOfVertices, MPI_DOUBLE,
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
         nv = NumOfBdrElements + shared_faces.Size();
         MPI_Reduce(&nv, &ne, 1, MPI_INT, MPI_SUM, 0, MyComm);
         MPI_Send(&NumOfVertices, 1, MPI_INT, 0, 444, MyComm);
         ne = NumOfBdrElements + shared_faces.Size();
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
            v = shared_faces[i-NumOfBdrElements]->GetVertices();
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
      Array<double> vert;
      Array<int> ints;

      int TG_nv, TG_ne, TG_nbe;

      if (MyRank == 0)
      {
         MPI_Reduce(&NumOfVertices, &TG_nv, 1, MPI_INT, MPI_SUM, 0, MyComm);
         MPI_Reduce(&NumOfElements, &TG_ne, 1, MPI_INT, MPI_SUM, 0, MyComm);
         nv = NumOfBdrElements + shared_faces.Size();
         MPI_Reduce(&nv, &TG_nbe, 1, MPI_INT, MPI_SUM, 0, MyComm);

         out << "TrueGrid\n"
             << "1 " << TG_nv << " " << TG_ne << " 0 0 0 0 0 0 0\n"
             << "0 0 0 1 0 0 0 0 0 0 0\n"
             << "0 0 " << TG_nbe << " 0 0 0 0 0 0 0 0 0 0 0 0 0\n"
             << "0.0 0.0 0.0 0 0 0.0 0.0 0 0.0\n"
             << "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n";

         // print the vertices
         nv = TG_nv;
         for (i = 0; i < NumOfVertices; i++)
            out << i+1 << " 0.0 " << vertices[i](0) << " " << vertices[i](1)
                << " " << vertices[i](2) << " 0.0\n";
         for (p = 1; p < NRanks; p++)
         {
            MPI_Recv(&nv, 1, MPI_INT, p, 444, MyComm, &status);
            vert.SetSize(Dim*nv);
            MPI_Recv(&vert[0], Dim*nv, MPI_DOUBLE, p, 445, MyComm, &status);
            for (i = 0; i < nv; i++)
               out << i+1 << " 0.0 " << vert[Dim*i] << " " << vert[Dim*i+1]
                   << " " << vert[Dim*i+2] << " 0.0\n";
         }

         // print the elements
         ne = TG_ne;
         for (i = 0; i < NumOfElements; i++)
         {
            nv = elements[i]->GetNVertices();
            ind = elements[i]->GetVertices();
            out << i+1 << " " << 1;
            for (j = 0; j < nv; j++)
            {
               out << " " << ind[j]+1;
            }
            out << '\n';
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
               out << i+1 << " " << p+1;
               for (j = 0; j < 8; j++)
               {
                  out << " " << k+ints[i*8+j]+1;
               }
               out << '\n';
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
            out << 1;
            for (j = 0; j < nv; j++)
            {
               out << " " << ind[j]+1;
            }
            out << " 1.0 1.0 1.0 1.0\n";
         }
         // shared faces
         for (i = 0; i < shared_faces.Size(); i++)
         {
            nv = shared_faces[i]->GetNVertices();
            ind = shared_faces[i]->GetVertices();
            out << 1;
            for (j = 0; j < nv; j++)
            {
               out << " " << ind[j]+1;
            }
            out << " 1.0 1.0 1.0 1.0\n";
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
               out << p+1;
               for (j = 0; j < 4; j++)
               {
                  out << " " << k+ints[i*4+j]+1;
               }
               out << " 1.0 1.0 1.0 1.0\n";
            }
            k += nv;
         }
      }
      else
      {
         MPI_Reduce(&NumOfVertices, &TG_nv, 1, MPI_INT, MPI_SUM, 0, MyComm);
         MPI_Reduce(&NumOfElements, &TG_ne, 1, MPI_INT, MPI_SUM, 0, MyComm);
         nv = NumOfBdrElements + shared_faces.Size();
         MPI_Reduce(&nv, &TG_nbe, 1, MPI_INT, MPI_SUM, 0, MyComm);

         MPI_Send(&NumOfVertices, 1, MPI_INT, 0, 444, MyComm);
         vert.SetSize(Dim*NumOfVertices);
         for (i = 0; i < NumOfVertices; i++)
            for (j = 0; j < Dim; j++)
            {
               vert[Dim*i+j] = vertices[i](j);
            }
         MPI_Send(&vert[0], Dim*NumOfVertices, MPI_DOUBLE, 0, 445, MyComm);
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
         ne = NumOfBdrElements + shared_faces.Size();
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
            v = shared_faces[i-NumOfBdrElements]->GetVertices();
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
      Array<double> vert;
      Array<int> ints;


      if (MyRank == 0)
      {
         out << "areamesh2\n\n";

         // print the boundary + shared edges information
         nv = NumOfBdrElements + shared_edges.Size();
         MPI_Reduce(&nv, &ne, 1, MPI_INT, MPI_SUM, 0, MyComm);
         out << ne << '\n';
         // boundary
         for (i = 0; i < NumOfBdrElements; i++)
         {
            attr = boundary[i]->GetAttribute();
            boundary[i]->GetVertices(v);
            out << attr << "     ";
            for (j = 0; j < v.Size(); j++)
            {
               out << v[j] + 1 << "   ";
            }
            out << '\n';
         }
         // shared edges
         for (i = 0; i < shared_edges.Size(); i++)
         {
            attr = shared_edges[i]->GetAttribute();
            shared_edges[i]->GetVertices(v);
            out << attr << "     ";
            for (j = 0; j < v.Size(); j++)
            {
               out << v[j] + 1 << "   ";
            }
            out << '\n';
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
               out << p+1;
               for (j = 0; j < 2; j++)
               {
                  out << " " << k+ints[i*2+j]+1;
               }
               out << '\n';
            }
            k += nv;
         }

         // print the elements
         nv = NumOfElements;
         MPI_Reduce(&nv, &ne, 1, MPI_INT, MPI_SUM, 0, MyComm);
         out << ne << '\n';
         for (i = 0; i < NumOfElements; i++)
         {
            // attr = elements[i]->GetAttribute(); // not used
            elements[i]->GetVertices(v);
            out << 1 << "   " << 3 << "   ";
            for (j = 0; j < v.Size(); j++)
            {
               out << v[j] + 1 << "  ";
            }
            out << '\n';
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
               out << p+1 << " " << 3;
               for (j = 0; j < 3; j++)
               {
                  out << " " << k+ints[i*3+j]+1;
               }
               out << '\n';
            }
            k += nv;
         }

         // print the vertices
         ne = NumOfVertices;
         MPI_Reduce(&ne, &nv, 1, MPI_INT, MPI_SUM, 0, MyComm);
         out << nv << '\n';
         for (i = 0; i < NumOfVertices; i++)
         {
            for (j = 0; j < Dim; j++)
            {
               out << vertices[i](j) << " ";
            }
            out << '\n';
         }
         for (p = 1; p < NRanks; p++)
         {
            MPI_Recv(&nv, 1, MPI_INT, p, 444, MyComm, &status);
            vert.SetSize(Dim*nv);
            MPI_Recv(&vert[0], Dim*nv, MPI_DOUBLE, p, 445, MyComm, &status);
            for (i = 0; i < nv; i++)
            {
               for (j = 0; j < Dim; j++)
               {
                  out << " " << vert[Dim*i+j];
               }
               out << '\n';
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
         MPI_Send(&vert[0], Dim*NumOfVertices, MPI_DOUBLE,
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

   MPI_Allreduce(p_min.GetData(), gp_min, sdim, MPI_DOUBLE, MPI_MIN, MyComm);
   MPI_Allreduce(p_max.GetData(), gp_max, sdim, MPI_DOUBLE, MPI_MAX, MyComm);
}

void ParMesh::GetCharacteristics(double &gh_min, double &gh_max,
                                 double &gk_min, double &gk_max)
{
   double h_min, h_max, kappa_min, kappa_max;

   this->Mesh::GetCharacteristics(h_min, h_max, kappa_min, kappa_max);

   MPI_Allreduce(&h_min, &gh_min, 1, MPI_DOUBLE, MPI_MIN, MyComm);
   MPI_Allreduce(&h_max, &gh_max, 1, MPI_DOUBLE, MPI_MAX, MyComm);
   MPI_Allreduce(&kappa_min, &gk_min, 1, MPI_DOUBLE, MPI_MIN, MyComm);
   MPI_Allreduce(&kappa_max, &gk_max, 1, MPI_DOUBLE, MPI_MAX, MyComm);
}

void ParMesh::PrintInfo(std::ostream &out)
{
   int i;
   DenseMatrix J(Dim);
   double h_min, h_max, kappa_min, kappa_max, h, kappa;

   if (MyRank == 0)
   {
      out << "Parallel Mesh Stats:" << '\n';
   }

   for (i = 0; i < NumOfElements; i++)
   {
      GetElementJacobian(i, J);
      h = pow(fabs(J.Det()), 1.0/double(Dim));
      kappa = J.CalcSingularvalue(0) / J.CalcSingularvalue(Dim-1);
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

   double gh_min, gh_max, gk_min, gk_max;
   MPI_Reduce(&h_min, &gh_min, 1, MPI_DOUBLE, MPI_MIN, 0, MyComm);
   MPI_Reduce(&h_max, &gh_max, 1, MPI_DOUBLE, MPI_MAX, 0, MyComm);
   MPI_Reduce(&kappa_min, &gk_min, 1, MPI_DOUBLE, MPI_MIN, 0, MyComm);
   MPI_Reduce(&kappa_max, &gk_max, 1, MPI_DOUBLE, MPI_MAX, 0, MyComm);

   long ldata[5]; // vert, edge, face, elem, neighbors;
   long mindata[5], maxdata[5], sumdata[5];

   // count locally owned vertices, edges, and faces
   ldata[0] = GetNV();
   ldata[1] = GetNEdges();
   ldata[2] = GetNFaces();
   ldata[3] = GetNE();
   ldata[4] = gtopo.GetNumNeighbors()-1;
   for (int gr = 1; gr < GetNGroups(); gr++)
      if (!gtopo.IAmMaster(gr)) // we are not the master
      {
         ldata[0] -= group_svert.RowSize(gr-1);
         ldata[1] -= group_sedge.RowSize(gr-1);
         ldata[2] -= group_sface.RowSize(gr-1);
      }

   MPI_Reduce(ldata, mindata, 5, MPI_LONG, MPI_MIN, 0, MyComm);
   MPI_Reduce(ldata, sumdata, 5, MPI_LONG, MPI_SUM, 0, MyComm);
   MPI_Reduce(ldata, maxdata, 5, MPI_LONG, MPI_MAX, 0, MyComm);

   if (MyRank == 0)
   {
      out << '\n'
          << "           "
          << setw(12) << "minimum"
          << setw(12) << "average"
          << setw(12) << "maximum"
          << setw(12) << "total" << '\n';
      out << " vertices  "
          << setw(12) << mindata[0]
          << setw(12) << sumdata[0]/NRanks
          << setw(12) << maxdata[0]
          << setw(12) << sumdata[0] << '\n';
      out << " edges     "
          << setw(12) << mindata[1]
          << setw(12) << sumdata[1]/NRanks
          << setw(12) << maxdata[1]
          << setw(12) << sumdata[1] << '\n';
      if (Dim == 3)
         out << " faces     "
             << setw(12) << mindata[2]
             << setw(12) << sumdata[2]/NRanks
             << setw(12) << maxdata[2]
             << setw(12) << sumdata[2] << '\n';
      out << " elements  "
          << setw(12) << mindata[3]
          << setw(12) << sumdata[3]/NRanks
          << setw(12) << maxdata[3]
          << setw(12) << sumdata[3] << '\n';
      out << " neighbors "
          << setw(12) << mindata[4]
          << setw(12) << sumdata[4]/NRanks
          << setw(12) << maxdata[4] << '\n';
      out << '\n'
          << "       "
          << setw(12) << "minimum"
          << setw(12) << "maximum" << '\n';
      out << " h     "
          << setw(12) << gh_min
          << setw(12) << gh_max << '\n';
      out << " kappa "
          << setw(12) << gk_min
          << setw(12) << gk_max << '\n';
      out << std::flush;
   }
}

long ParMesh::ReduceInt(int value) const
{
   long local = value, global;
   MPI_Allreduce(&local, &global, 1, MPI_LONG, MPI_SUM, MyComm);
   return global;
}

void ParMesh::ParPrint(ostream &out) const
{
   if (NURBSext || pncmesh)
   {
      // TODO: AMR meshes, NURBS meshes.
      Print(out);
      return;
   }

   // Write out serial mesh.  Tell serial mesh to deliniate the end of it's
   // output with 'mfem_serial_mesh_end' instead of 'mfem_mesh_end', as we will
   // be adding additional parallel mesh information.
   Printer(out, "mfem_serial_mesh_end");

   // write out group topology info.
   gtopo.Save(out);

   out << "\ntotal_shared_vertices " << svert_lvert.Size() << '\n';
   if (Dim >= 2)
   {
      out << "total_shared_edges " << shared_edges.Size() << '\n';
   }
   if (Dim >= 3)
   {
      out << "total_shared_faces " << shared_faces.Size() << '\n';
   }
   for (int gr = 1; gr < GetNGroups(); gr++)
   {
      {
         const int  nv = group_svert.RowSize(gr-1);
         const int *sv = group_svert.GetRow(gr-1);
         out << "\n#group " << gr << "\nshared_vertices " << nv << '\n';
         for (int i = 0; i < nv; i++)
         {
            out << svert_lvert[sv[i]] << '\n';
         }
      }
      if (Dim >= 2)
      {
         const int  ne = group_sedge.RowSize(gr-1);
         const int *se = group_sedge.GetRow(gr-1);
         out << "\nshared_edges " << ne << '\n';
         for (int i = 0; i < ne; i++)
         {
            const int *v = shared_edges[se[i]]->GetVertices();
            out << v[0] << ' ' << v[1] << '\n';
         }
      }
      if (Dim >= 3)
      {
         const int  nf = group_sface.RowSize(gr-1);
         const int *sf = group_sface.GetRow(gr-1);
         out << "\nshared_faces " << nf << '\n';
         for (int i = 0; i < nf; i++)
         {
            PrintElementWithoutAttr(shared_faces[sf[i]], out);
         }
      }
   }

   // Write out section end tag for mesh.
   out << "\nmfem_mesh_end" << endl;
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

ParMesh::~ParMesh()
{
   delete pncmesh;
   ncmesh = pncmesh = NULL;

   DeleteFaceNbrData();

   for (int i = 0; i < shared_faces.Size(); i++)
   {
      FreeElement(shared_faces[i]);
   }
   for (int i = 0; i < shared_edges.Size(); i++)
   {
      FreeElement(shared_edges[i]);
   }

   // The Mesh destructor is called automatically
}

}

#endif
