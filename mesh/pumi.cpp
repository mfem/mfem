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

#include "pumi.hpp"

#ifdef MFEM_USE_PUMI
#ifdef MFEM_USE_MPI

#include "mesh_headers.hpp"
#include "../fem/fem.hpp"
#include "../general/sort_pairs.hpp"
#include "../general/text.hpp"
#include "../general/sets.hpp"

#include <iostream>
#include <sstream>
#include <fstream>
#include <limits>
#include <cmath>
#include <cstring>
#include <ctime>

using namespace std;

namespace mfem
{

PumiMesh::PumiMesh(apf::Mesh2* apf_mesh, int generate_edges, int refine,
                   bool fix_orientation)
{
   Load(apf_mesh, generate_edges, refine, fix_orientation);
}

Element *PumiMesh::ReadElement(apf::MeshEntity* Ent, const int geom,
                               apf::Downward Verts,
                               const int Attr, apf::Numbering* vert_num)
{
   Element *el;
   int nv, *v;

   // Create element in MFEM
   el = NewElement(geom);
   nv = el->GetNVertices();
   v  = el->GetVertices();

   // Fill the connectivity
   for (int i = 0; i < nv; ++i)
   {
      v[i] = apf::getNumber(vert_num, Verts[i], 0, 0);
   }

   // Assign attribute
   el->SetAttribute(Attr);

   return el;
}

void PumiMesh::CountBoundaryEntity(apf::Mesh2* apf_mesh, const int BcDim,
                                   int &NumBc)
{
   apf::MeshEntity* ent;
   apf::MeshIterator* itr = apf_mesh->begin(BcDim);

   while ((ent=apf_mesh->iterate(itr)))
   {
      apf::ModelEntity* mdEnt = apf_mesh->toModel(ent);
      if (apf_mesh->getModelType(mdEnt) == BcDim)
      {
         NumBc++;
      }
   }
   apf_mesh->end(itr);

   // Check if any boundary is detected
   if (NumBc==0)
   {
      MFEM_ABORT("no boundary detected!");
   }
}

void PumiMesh::Load(apf::Mesh2* apf_mesh, int generate_edges, int refine,
                    bool fix_orientation)
{
   int  curved = 0, read_gf = 1;

   // Add a check on apf_mesh just in case
   Clear();

   // First number vertices
   apf::Field* apf_field_crd = apf_mesh->getCoordinateField();
   apf::FieldShape* crd_shape = apf::getShape(apf_field_crd);
   apf::Numbering* v_num_loc = apf::createNumbering(apf_mesh, "VertexNumbering",
                                                    crd_shape, 1);
   // Check if it is a curved mesh
   curved = (crd_shape->getOrder() > 1) ? 1 : 0;

   // Read mesh
   ReadSCORECMesh(apf_mesh, v_num_loc, curved);
#ifdef MFEM_DEBUG
   mfem::out << "After ReadSCORECMesh" << endl;
#endif
   // at this point the following should be defined:
   //  1) Dim
   //  2) NumOfElements, elements
   //  3) NumOfBdrElements, boundary
   //  4) NumOfVertices, with allocated space in vertices
   //  5) curved
   //  5a) if curved == 0, vertices must be defined
   //  5b) if curved != 0 and read_gf != 0,
   //         'input' must point to a GridFunction
   //  5c) if curved != 0 and read_gf == 0,
   //         vertices and Nodes must be defined

   // FinalizeTopology() will:
   // - assume that generate_edges is true
   // - assume that refine is false
   // - does not check the orientation of regular and boundary elements
   FinalizeTopology();

   if (curved && read_gf)
   {
      // Check it to be only Quadratic if higher order
      Nodes = new GridFunctionPumi(this, apf_mesh, v_num_loc,
                                   crd_shape->getOrder());
      edge_vertex = NULL;
      own_nodes = 1;
      spaceDim = Nodes->VectorDim();

      // Set the 'vertices' from the 'Nodes'
      for (int i = 0; i < spaceDim; i++)
      {
         Vector vert_val;
         Nodes->GetNodalValues(vert_val, i+1);
         for (int j = 0; j < NumOfVertices; j++)
         {
            vertices[j](i) = vert_val(j);
         }
      }
   }

   // Delete numbering
   apf::destroyNumbering(v_num_loc);

   Finalize(refine, fix_orientation);
}

void PumiMesh::ReadSCORECMesh(apf::Mesh2* apf_mesh, apf::Numbering* v_num_loc,
                              const int curved)
{
   // Here fill the element table from SCOREC MESH
   // The vector of element pointers is generated with attr and connectivity

   apf::MeshIterator* itr = apf_mesh->begin(0);
   apf::MeshEntity* ent;
   NumOfVertices = 0;
   while ((ent = apf_mesh->iterate(itr)))
   {
      // IDs start from 0
      apf::number(v_num_loc, ent, 0, 0, NumOfVertices);
      NumOfVertices++;
   }
   apf_mesh->end(itr);

   Dim = apf_mesh->getDimension();
   NumOfElements = countOwned(apf_mesh,Dim);
   elements.SetSize(NumOfElements);

   // Read elements from SCOREC Mesh
   itr = apf_mesh->begin(Dim);
   unsigned int j=0;
   while ((ent = apf_mesh->iterate(itr)))
   {
      // Get vertices
      apf::Downward verts;
      apf_mesh->getDownward(ent,0,verts); // num_vert
      // Get attribute Tag vs Geometry
      int attr = 1;

      int geom_type = apf_mesh->getType(ent);
      elements[j] = ReadElement(ent, geom_type, verts, attr, v_num_loc);
      j++;
   }
   // End iterator
   apf_mesh->end(itr);

   // Read Boundaries from SCOREC Mesh
   // First we need to count them
   int BCdim = Dim - 1;
   NumOfBdrElements = 0;
   CountBoundaryEntity(apf_mesh, BCdim, NumOfBdrElements);
   boundary.SetSize(NumOfBdrElements);
   j=0;

   // Read boundary from SCOREC mesh
   itr = apf_mesh->begin(BCdim);
   while ((ent = apf_mesh->iterate(itr)))
   {
      // Check if this mesh entity is on the model boundary
      apf::ModelEntity* mdEnt = apf_mesh->toModel(ent);
      if (apf_mesh->getModelType(mdEnt) == BCdim)
      {
         apf::Downward verts;
         apf_mesh->getDownward(ent, 0, verts);
         int attr = 1;
         int geom_type = apf_mesh->getType(ent);
         boundary[j] = ReadElement( ent, geom_type, verts, attr, v_num_loc);
         j++;
      }
   }
   apf_mesh->end(itr);

   // Fill vertices
   vertices.SetSize(NumOfVertices);

   if (!curved)
   {
      apf::MeshIterator* itr = apf_mesh->begin(0);
      spaceDim = Dim;

      while ((ent = apf_mesh->iterate(itr)))
      {
         unsigned int id = apf::getNumber(v_num_loc, ent, 0, 0);
         apf::Vector3 Crds;
         apf_mesh->getPoint(ent,0,Crds);

         for (int ii=0; ii<spaceDim; ii++)
         {
            vertices[id](ii) = Crds[ii];
         }
      }
      apf_mesh->end(itr);
   }
}

// ParPumiMesh implementation
Element *ParPumiMesh::ReadElement(apf::MeshEntity* Ent, const int geom,
                                  apf::Downward Verts,
                                  const int Attr, apf::Numbering* vert_num)
{
   Element *el;
   int nv, *v;

   // Create element in MFEM
   el = NewElement(geom);
   nv = el->GetNVertices();
   v  = el->GetVertices();

   // Fill the connectivity
   for (int i = 0; i < nv; ++i)
   {
      v[i] = apf::getNumber(vert_num, Verts[i], 0, 0);
   }

   // Assign attribute
   el->SetAttribute(Attr);

   return el;
}

// This function loads a parallel PUMI mesh and returns the parallel MFEM mesh
// corresponding to it.
ParPumiMesh::ParPumiMesh(MPI_Comm comm, apf::Mesh2* apf_mesh)
{
   // Set the communicator for gtopo
   gtopo.SetComm(comm);

   MyComm = comm;
   MPI_Comm_size(MyComm, &NRanks);
   MPI_Comm_rank(MyComm, &MyRank);

   Dim = apf_mesh->getDimension();
   spaceDim = Dim;// mesh.spaceDim;

   apf::MeshIterator* itr;
   apf::MeshEntity* ent;

   // Global numbering of vertices. This is necessary to build a local numbering
   // that has the same ordering in each process.
   apf::Numbering* vLocNum =
      apf::numberOwnedDimension(apf_mesh, "AuxVertexNumbering", 0);
   apf::GlobalNumbering* VertexNumbering = apf::makeGlobal(vLocNum, true);
   apf::synchronize(VertexNumbering);

   // Take this process global vertex IDs and sort
   Array<Pair<long,int> > thisVertIds(apf_mesh->count(0));
   itr = apf_mesh->begin(0);
   for (int i = 0; (ent = apf_mesh->iterate(itr)); i++)
   {
      long id = apf::getNumber(VertexNumbering, ent, 0, 0);
      thisVertIds[i] = Pair<long,int>(id, i);
   }
   apf_mesh->end(itr);
   apf::destroyGlobalNumbering(VertexNumbering);
   thisVertIds.Sort();
   // Set thisVertIds[i].one = j where j is such that thisVertIds[j].two = i.
   // Thus, the mapping i -> thisVertIds[i].one is the inverse of the mapping
   // j -> thisVertIds[j].two.
   for (int j = 0; j < thisVertIds.Size(); j++)
   {
      const int i = thisVertIds[j].two;
      thisVertIds[i].one = j;
   }

   // Create local numbering that respects the global ordering
   apf::Field* apf_field_crd = apf_mesh->getCoordinateField();
   apf::FieldShape* crd_shape = apf::getShape(apf_field_crd);
   apf::Numbering* v_num_loc = apf::createNumbering(apf_mesh,
                                                    "LocalVertexNumbering",
                                                    crd_shape, 1);

   // Construct the numbering v_loc_num and set the coordinates of the vertices.
   NumOfVertices = thisVertIds.Size();
   vertices.SetSize(NumOfVertices);
   itr = apf_mesh->begin(0);
   for (int i = 0; (ent = apf_mesh->iterate(itr)); i++)
   {
      const int id = thisVertIds[i].one;
      // Assign as local number
      apf::number(v_num_loc, ent, 0, 0, id);

      apf::Vector3 Crds;
      apf_mesh->getPoint(ent,0,Crds);

      for (int ii=0; ii<spaceDim; ii++)
      {
         vertices[id](ii) = Crds[ii]; // Assuming the IDs are ordered and from 0
      }
   }
   apf_mesh->end(itr);
   thisVertIds.DeleteAll();

   // Fill the elements
   NumOfElements = countOwned(apf_mesh,Dim);
   elements.SetSize(NumOfElements);

   // Read elements from SCOREC Mesh
   itr = apf_mesh->begin(Dim);
   for (int j = 0; (ent = apf_mesh->iterate(itr)); j++)
   {
      // Get vertices
      apf::Downward verts;
      apf_mesh->getDownward(ent,0,verts);

      // Get attribute Tag vs Geometry
      int attr = 1;
      int geom_type = apf_mesh->getType(ent);
      elements[j] = ReadElement(ent, geom_type, verts, attr, v_num_loc);
   }
   // End iterator
   apf_mesh->end(itr);

   // Count number of boundaries by classification
   int BcDim = Dim - 1;
   itr = apf_mesh->begin(BcDim);
   NumOfBdrElements = 0;
   while ((ent=apf_mesh->iterate(itr)))
   {
      apf::ModelEntity* mdEnt = apf_mesh->toModel(ent);
      if (apf_mesh->getModelType(mdEnt) == BcDim)
      {
         NumOfBdrElements++;
      }
   }
   apf_mesh->end(itr);

   boundary.SetSize(NumOfBdrElements);
   // Read boundary from SCOREC mesh
   itr = apf_mesh->begin(BcDim);
   for (int bdr_ctr = 0; (ent = apf_mesh->iterate(itr)); )
   {
      // Check if this mesh entity is on the model boundary
      apf::ModelEntity* mdEnt = apf_mesh->toModel(ent);
      if (apf_mesh->getModelType(mdEnt) == BcDim)
      {
         apf::Downward verts;
         apf_mesh->getDownward(ent, 0, verts);
         int attr = 1 ;
         int geom_type = apf_mesh->getType(ent);
         boundary[bdr_ctr++] = ReadElement(ent, geom_type, verts, attr,
                                           v_num_loc);
      }
   }
   apf_mesh->end(itr);

   // The next two methods are called by FinalizeTopology() called below:
   // Mesh::SetMeshGen();
   // Mesh::SetAttributes();

   // This is called by the default Mesh constructor
   // Mesh::InitTables();

   this->FinalizeTopology();

   ListOfIntegerSets  groups;
   IntegerSet         group;

   // The first group is the local one
   group.Recreate(1, &MyRank);
   groups.Insert(group);

   MFEM_ASSERT(Dim >= 3 || GetNFaces() == 0,
               "[proc " << MyRank << "]: invalid state");

   // Determine shared faces
   Array<Pair<long, apf::MeshEntity*> > sfaces;
   // Initially sfaces[i].one holds the global face id.
   // Then it is replaced by the group id of the shared face.
   if (Dim > 2)
   {
      // Number the faces globally and enumerate the local shared faces
      // following the global enumeration. This way we ensure that the ordering
      // of the shared faces within each group (of processors) is the same in
      // each processor in the group.
      apf::Numbering* AuxFaceNum =
         apf::numberOwnedDimension(apf_mesh, "AuxFaceNumbering", 2);
      apf::GlobalNumbering* GlobalFaceNum = apf::makeGlobal(AuxFaceNum, true);
      apf::synchronize(GlobalFaceNum);

      itr = apf_mesh->begin(2);
      while ((ent = apf_mesh->iterate(itr)))
      {
         if (apf_mesh->isShared(ent))
         {
            long id = apf::getNumber(GlobalFaceNum, ent, 0, 0);
            sfaces.Append(Pair<long,apf::MeshEntity*>(id, ent));
         }
      }
      apf_mesh->end(itr);
      sfaces.Sort();
      apf::destroyGlobalNumbering(GlobalFaceNum);

      // Replace the global face id in sfaces[i].one with group id.
      for (int i = 0; i < sfaces.Size(); i++)
      {
         ent = sfaces[i].two;

         const int thisNumAdjs = 2;
         int eleRanks[thisNumAdjs];

         // Get the IDs
         apf::Parts res;
         apf_mesh->getResidence(ent, res);
         int kk = 0;
         for (std::set<int>::iterator itr = res.begin();
              itr != res.end(); ++itr)
         {
            eleRanks[kk++] = *itr;
         }

         group.Recreate(2, eleRanks);
         sfaces[i].one = groups.Insert(group) - 1;
      }
   }

   // Determine shared edges
   Array<Pair<long, apf::MeshEntity*> > sedges;
   // Initially sedges[i].one holds the global edge id.
   // Then it is replaced by the group id of the shared edge.
   if (Dim > 1)
   {
      // Number the edges globally and enumerate the local shared edges
      // following the global enumeration. This way we ensure that the ordering
      // of the shared edges within each group (of processors) is the same in
      // each processor in the group.
      apf::Numbering* AuxEdgeNum =
         apf::numberOwnedDimension(apf_mesh, "EdgeNumbering", 1);
      apf::GlobalNumbering* GlobalEdgeNum = apf::makeGlobal(AuxEdgeNum, true);
      apf::synchronize(GlobalEdgeNum);

      itr = apf_mesh->begin(1);
      while ((ent = apf_mesh->iterate(itr)))
      {
         if (apf_mesh->isShared(ent))
         {
            long id = apf::getNumber(GlobalEdgeNum, ent, 0, 0);
            sedges.Append(Pair<long,apf::MeshEntity*>(id, ent));
         }
      }
      apf_mesh->end(itr);
      sedges.Sort();
      apf::destroyGlobalNumbering(GlobalEdgeNum);

      // Replace the global edge id in sedges[i].one with group id.
      Array<int> eleRanks;
      for (int i = 0; i < sedges.Size(); i++)
      {
         ent = sedges[i].two;

         // Number of adjacent element
         apf::Parts res;
         apf_mesh->getResidence(ent, res);
         eleRanks.SetSize(res.size());

         // Get the IDs
         int kk = 0;
         for (std::set<int>::iterator itr = res.begin();
              itr != res.end(); itr++)
         {
            eleRanks[kk++] = *itr;
         }

         // Generate the group
         group.Recreate(eleRanks.Size(), eleRanks);
         sedges[i].one = groups.Insert(group) - 1;
      }
   }

   // Determine shared vertices
   Array<Pair<int, apf::MeshEntity*> > sverts;
   // The entries sverts[i].one hold the local vertex ids.
   Array<int> svert_group;
   {
      itr = apf_mesh->begin(0);
      while ((ent = apf_mesh->iterate(itr)))
      {
         if (apf_mesh->isShared(ent))
         {
            int vtId = apf::getNumber(v_num_loc, ent, 0, 0);
            sverts.Append(Pair<int,apf::MeshEntity*>(vtId, ent));
         }
      }
      apf_mesh->end(itr);
      sverts.Sort();

      // Determine svert_group
      svert_group.SetSize(sverts.Size());
      Array<int> eleRanks;
      for (int i = 0; i < sverts.Size(); i++)
      {
         ent = sverts[i].two;

         // Number of adjacent element
         apf::Parts res;
         apf_mesh->getResidence(ent, res);
         eleRanks.SetSize(res.size());

         // Get the IDs
         int kk = 0;
         for (std::set<int>::iterator itr = res.begin();
              itr != res.end(); itr++)
         {
            eleRanks[kk++] = *itr;
         }

         group.Recreate(eleRanks.Size(), eleRanks);
         svert_group[i] = groups.Insert(group) - 1;
      }
   }

   // Build group_stria and group_squad.
   // Also allocate shared_trias, shared_quads, and sface_lface.
   group_stria.MakeI(groups.Size()-1);
   group_squad.MakeI(groups.Size()-1);
   for (int i = 0; i < sfaces.Size(); i++)
   {
      apf::Mesh::Type ftype = apf_mesh->getType(sfaces[i].two);
      if (ftype == apf::Mesh::TRIANGLE)
      {
         group_stria.AddAColumnInRow(sfaces[i].one);
      }
      else if (ftype == apf::Mesh::QUAD)
      {
         group_squad.AddAColumnInRow(sfaces[i].one);
      }
   }
   group_stria.MakeJ();
   group_squad.MakeJ();
   {
      int nst = 0;
      for (int i = 0; i < sfaces.Size(); i++)
      {
         apf::Mesh::Type ftype = apf_mesh->getType(sfaces[i].two);
         if (ftype == apf::Mesh::TRIANGLE)
         {
            group_stria.AddConnection(sfaces[i].one, nst++);
         }
         else if (ftype == apf::Mesh::QUAD)
         {
            group_squad.AddConnection(sfaces[i].one, i-nst);
         }
      }
      shared_trias.SetSize(nst);
      shared_quads.SetSize(sfaces.Size()-nst);
      sface_lface.SetSize(sfaces.Size());
   }
   group_stria.ShiftUpI();
   group_squad.ShiftUpI();

   // Build group_sedge
   group_sedge.MakeI(groups.Size()-1);
   for (int i = 0; i < sedges.Size(); i++)
   {
      group_sedge.AddAColumnInRow(sedges[i].one);
   }
   group_sedge.MakeJ();
   for (int i = 0; i < sedges.Size(); i++)
   {
      group_sedge.AddConnection(sedges[i].one, i);
   }
   group_sedge.ShiftUpI();

   // Build group_svert
   group_svert.MakeI(groups.Size()-1);
   for (int i = 0; i < svert_group.Size(); i++)
   {
      group_svert.AddAColumnInRow(svert_group[i]);
   }
   group_svert.MakeJ();
   for (int i = 0; i < svert_group.Size(); i++)
   {
      group_svert.AddConnection(svert_group[i], i);
   }
   group_svert.ShiftUpI();

   // Build shared_trias and shared_quads. They are allocated above.
   {
      int nst = 0;
      for (int i = 0; i < sfaces.Size(); i++)
      {
         ent = sfaces[i].two;

         apf::Downward verts;
         apf_mesh->getDownward(ent,0,verts);

         int *v, nv = 0;
         apf::Mesh::Type ftype = apf_mesh->getType(ent);
         if (ftype == apf::Mesh::TRIANGLE)
         {
            v = shared_trias[nst++].v;
            nv = 3;
         }
         else if (ftype == apf::Mesh::QUAD)
         {
            v = shared_quads[i-nst].v;
            nv = 4;
         }
         for (int j = 0; j < nv; ++j)
         {
            v[j] = apf::getNumber(v_num_loc, verts[j], 0, 0);
         }
      }
   }

   // Build shared_edges and allocate sedge_ledge
   shared_edges.SetSize(sedges.Size());
   sedge_ledge. SetSize(sedges.Size());
   for (int i = 0; i < sedges.Size(); i++)
   {
      ent = sedges[i].two;

      apf::Downward verts;
      apf_mesh->getDownward(ent, 0, verts);
      int id1, id2;
      id1 = apf::getNumber(v_num_loc, verts[0], 0, 0);
      id2 = apf::getNumber(v_num_loc, verts[1], 0, 0);
      if (id1 > id2) { swap(id1,id2); }

      shared_edges[i] = new Segment(id1, id2, 1);
   }

   // Build svert_lvert
   svert_lvert.SetSize(sverts.Size());
   for (int i = 0; i < sverts.Size(); i++)
   {
      svert_lvert[i] = sverts[i].one;
   }

   // Build the group communication topology
   gtopo.Create(groups, 822);

   // Determine sedge_ledge and sface_lface
   FinalizeParTopo();

   // Set nodes for higher order mesh
   int curved = (crd_shape->getOrder() > 1) ? 1 : 0;
   if (curved) // curved mesh
   {
      GridFunctionPumi auxNodes(this, apf_mesh, v_num_loc,
                                crd_shape->getOrder());
      Nodes = new ParGridFunction(this, &auxNodes);
      Nodes->Vector::Swap(auxNodes);
      this->edge_vertex = NULL;
      own_nodes = 1;
   }
}


// GridFunctionPumi Implementation needed for high order meshes
GridFunctionPumi::GridFunctionPumi(Mesh* m, apf::Mesh2* PumiM,
                                   apf::Numbering* v_num_loc,
                                   const int mesh_order)
{
   int spDim = m->SpaceDimension();
   // Note: default BasisType for 'fec' is GaussLobatto.
   fec = new H1_FECollection(mesh_order, m->Dimension());
   int ordering = Ordering::byVDIM; // x1y1z1/x2y2z2/...
   fes = new FiniteElementSpace(m, fec, spDim, ordering);
   int data_size = fes->GetVSize();

   // Read PUMI mesh data
   this->SetSize(data_size);
   double* PumiData = this->GetData();

   apf::MeshEntity* ent;
   apf::MeshIterator* itr;

   // Assume all element type are the same i.e. tetrahedral
   const FiniteElement* H1_elem = fes->GetFE(0);
   const IntegrationRule &All_nodes = H1_elem->GetNodes();
   int nnodes = All_nodes.Size();

   // Loop over elements
   apf::Field* crd_field = PumiM->getCoordinateField();

   int nc = apf::countComponents(crd_field);
   int iel = 0;
   itr = PumiM->begin(m->Dimension());
   while ((ent = PumiM->iterate(itr)))
   {
      Array<int> vdofs;
      fes->GetElementVDofs(iel, vdofs);

      // Create PUMI element to interpolate
      apf::MeshElement* mE = apf::createMeshElement(PumiM, ent);
      apf::Element* elem = apf::createElement(crd_field, mE);

      // Vertices are already interpolated
      for (int ip = 0; ip < nnodes; ip++)
      {
         // Take parametric coordinates of the node
         apf::Vector3 param;
         param[0] = All_nodes.IntPoint(ip).x;
         param[1] = All_nodes.IntPoint(ip).y;
         param[2] = All_nodes.IntPoint(ip).z;

         // Compute the interpolating coordinates
         apf::DynamicVector phCrd(nc);
         apf::getComponents(elem, param, &phCrd[0]);

         // Fill the nodes list
         for (int kk = 0; kk < spDim; ++kk)
         {
            int dof_ctr = ip + kk * nnodes;
            PumiData[vdofs[dof_ctr]] = phCrd[kk];
         }
      }
      iel++;
      apf::destroyElement(elem);
      apf::destroyMeshElement(mE);
   }
   PumiM->end(itr);

   sequence = 0;
}

// Copy the adapted mesh to the original mesh and increase the sequence to be
// able to Call Update() methods of FESpace, Linear and Bilinear forms.
void ParPumiMesh::UpdateMesh(const ParMesh* AdaptedpMesh)
{
   // Destroy the ParMesh data fields.
   delete pncmesh;
   pncmesh = NULL;

   DeleteFaceNbrData();

   for (int i = 0; i < shared_edges.Size(); i++)
   {
      FreeElement(shared_edges[i]);
   }
   shared_quads.DeleteAll();
   shared_trias.DeleteAll();
   shared_edges.DeleteAll();
   group_svert.Clear();
   group_sedge.Clear();
   group_stria.Clear();
   group_squad.Clear();
   svert_lvert.DeleteAll();
   sedge_ledge.DeleteAll();
   sface_lface.DeleteAll();

   // Destroy the Mesh data fields.
   Destroy();

   // Assuming Dim, spaceDim, geom type is unchanged
   MFEM_ASSERT(Dim == AdaptedpMesh->Dim, "");
   MFEM_ASSERT(spaceDim == AdaptedpMesh->spaceDim, "");
   MFEM_ASSERT(meshgen == AdaptedpMesh->meshgen, "");

   NumOfVertices = AdaptedpMesh->GetNV();
   NumOfElements = AdaptedpMesh->GetNE();
   NumOfBdrElements = AdaptedpMesh->GetNBE();
   NumOfEdges = AdaptedpMesh->GetNEdges();
   NumOfFaces = AdaptedpMesh->GetNFaces();

   meshgen = AdaptedpMesh->meshgen;

   // Sequence is increased by one to trigger update in FEspace etc.
   sequence++;
   last_operation = Mesh::NONE;

   // Duplicate the elements
   elements.SetSize(NumOfElements);
   for (int i = 0; i < NumOfElements; i++)
   {
      elements[i] = AdaptedpMesh->GetElement(i)->Duplicate(this);
   }

   // Copy the vertices
   AdaptedpMesh->vertices.Copy(vertices);

   // Duplicate the boundary
   boundary.SetSize(NumOfBdrElements);
   for (int i = 0; i < NumOfBdrElements; i++)
   {
      boundary[i] = AdaptedpMesh->GetBdrElement(i)->Duplicate(this);
   }

   // Copy the element-to-face Table, el_to_face
   el_to_face = (AdaptedpMesh->el_to_face) ?
                new Table(*(AdaptedpMesh->el_to_face)) : NULL;

   // Copy the boundary-to-face Array, be_to_face.
   AdaptedpMesh->be_to_face.Copy(be_to_face);

   // Copy the element-to-edge Table, el_to_edge
   el_to_edge = (AdaptedpMesh->el_to_edge) ?
                new Table(*(AdaptedpMesh->el_to_edge)) : NULL;

   // Copy the boudary-to-edge Table, bel_to_edge (3D)
   bel_to_edge = (AdaptedpMesh->bel_to_edge) ?
                 new Table(*(AdaptedpMesh->bel_to_edge)) : NULL;

   // Copy the boudary-to-edge Array, be_to_edge (2D)
   AdaptedpMesh->be_to_edge.Copy(be_to_edge);

   // Duplicate the faces and faces_info.
   faces.SetSize(AdaptedpMesh->faces.Size());
   for (int i = 0; i < faces.Size(); i++)
   {
      Element *face = AdaptedpMesh->faces[i]; // in 1D the faces are NULL
      faces[i] = (face) ? face->Duplicate(this) : NULL;
   }
   AdaptedpMesh->faces_info.Copy(faces_info);

   // Do NOT copy the element-to-element Table, el_to_el
   el_to_el = NULL;

   // Do NOT copy the face-to-edge Table, face_edge
   face_edge = NULL;

   // Copy the edge-to-vertex Table, edge_vertex
   edge_vertex = (AdaptedpMesh->edge_vertex) ?
                 new Table(*(AdaptedpMesh->edge_vertex)) : NULL;

   // Copy the attributes and bdr_attributes
   AdaptedpMesh->attributes.Copy(attributes);
   AdaptedpMesh->bdr_attributes.Copy(bdr_attributes);

   // PUMI meshes cannot use NURBS meshes.
   MFEM_VERIFY(AdaptedpMesh->NURBSext == NULL,
               "invalid adapted mesh: it is a NURBS mesh");
   NURBSext = NULL;

   // PUMI meshes cannot use NCMesh/ParNCMesh.
   MFEM_VERIFY(AdaptedpMesh->ncmesh == NULL && AdaptedpMesh->pncmesh == NULL,
               "invalid adapted mesh: it is a non-conforming mesh");
   ncmesh = NULL;
   pncmesh = NULL;

   // Parallel Implications
   AdaptedpMesh->group_svert.Copy(group_svert);
   AdaptedpMesh->group_sedge.Copy(group_sedge);
   group_stria = AdaptedpMesh->group_stria;
   group_squad = AdaptedpMesh->group_squad;
   AdaptedpMesh->gtopo.Copy(gtopo);

   MyComm = AdaptedpMesh->MyComm;
   NRanks = AdaptedpMesh->NRanks;
   MyRank = AdaptedpMesh->MyRank;

   // Duplicate the shared_edges
   shared_edges.SetSize(AdaptedpMesh->shared_edges.Size());
   for (int i = 0; i < shared_edges.Size(); i++)
   {
      shared_edges[i] = AdaptedpMesh->shared_edges[i]->Duplicate(this);
   }

   // Duplicate the shared_trias and shared_quads
   shared_trias = AdaptedpMesh->shared_trias;
   shared_quads = AdaptedpMesh->shared_quads;

   // Copy the shared-to-local index Arrays
   AdaptedpMesh->svert_lvert.Copy(svert_lvert);
   AdaptedpMesh->sedge_ledge.Copy(sedge_ledge);
   AdaptedpMesh->sface_lface.Copy(sface_lface);

   // Do not copy face-neighbor data (can be generated if needed)
   have_face_nbr_data = false;

   // Copy the Nodes as a ParGridFunction, including the FiniteElementCollection
   // and the FiniteElementSpace (as a ParFiniteElementSpace)
   if (AdaptedpMesh->Nodes)
   {
      FiniteElementSpace *fes = AdaptedpMesh->Nodes->FESpace();
      const FiniteElementCollection *fec = fes->FEColl();
      FiniteElementCollection *fec_copy =
         FiniteElementCollection::New(fec->Name());
      ParFiniteElementSpace *pfes_copy =
         new ParFiniteElementSpace(this, fec_copy, fes->GetVDim(),
                                   fes->GetOrdering());
      Nodes = new ParGridFunction(pfes_copy);
      Nodes->MakeOwner(fec_copy);
      *Nodes = *(AdaptedpMesh->Nodes);
      own_nodes = 1;
   }
}

// Transfer a mixed vector-scalar field (i.e. velocity,pressure) and the
// magnitude of the vector field to use for mesh adaptation.
void ParPumiMesh::FieldMFEMtoPUMI(apf::Mesh2* apf_mesh,
                                  ParGridFunction* grid_vel,
                                  ParGridFunction* grid_pr,
                                  apf::Field* VelField,
                                  apf::Field* PrField,
                                  apf::Field* VelMagField)
{
   apf::FieldShape* VelFieldShape = getShape(VelField);
   int num_nodes = 4 * VelFieldShape->countNodesOn(0) + // Vertex
                   6 * VelFieldShape->countNodesOn(1) + // Edge
                   4 * VelFieldShape->countNodesOn(2) + // Triangle
                   VelFieldShape->countNodesOn(4); // Tetrahedron

   // Define integration points
   IntegrationRule pumi_nodes(num_nodes);
   int ip_cnt = 0;
   apf::Vector3 xi_crd(0.,0.,0.);

   // Create a template of dof holders coordinates in parametric coordinates.
   // The ordering is taken care of when the field is transferred to PUMI.

   // Dofs on Vertices
   IntegrationPoint& ip = pumi_nodes.IntPoint(ip_cnt++);
   double pt_crd[3] = {0., 0., 0.};
   ip.Set(pt_crd, 3);
   for (int kk = 0; kk < 3; kk++)
   {
      IntegrationPoint& ip = pumi_nodes.IntPoint(ip_cnt++);
      double pt_crd[3] = {0.,0.,0.};
      pt_crd[kk] = 1.0;
      ip.Set(pt_crd, 3);
   }
   // Dofs on Edges
   if (VelFieldShape->hasNodesIn(apf::Mesh::EDGE))
   {
      const int nn = VelFieldShape->countNodesOn(apf::Mesh::EDGE);
      for (int ii = 0; ii < 6; ii++)
      {
         for (int jj = 0; jj < nn; jj++)
         {
            VelFieldShape->getNodeXi(apf::Mesh::EDGE, jj, xi_crd);
            xi_crd[0] = 0.5 * (xi_crd[0] + 1.);// from (-1,1) to (0,1)
            double pt_crd[3] = {0., 0., 0.};
            switch (ii)
            {
               case 0:
                  pt_crd[0] = xi_crd[0];
                  break;
               case 1:
                  pt_crd[0] = 1. - xi_crd[0];
                  pt_crd[1] = xi_crd[0];
                  break;
               case 2:
                  pt_crd[1] = xi_crd[0];
                  break;
               case 3:
                  pt_crd[2] = xi_crd[0];
                  break;
               case 4:
                  pt_crd[0] = 1. - xi_crd[0];
                  pt_crd[2] = xi_crd[0];
                  break;
               case 5:
                  pt_crd[1] = 1. - xi_crd[0];
                  pt_crd[2] = xi_crd[0];
                  break;
            }
            IntegrationPoint& ip = pumi_nodes.IntPoint(ip_cnt++);
            ip.Set(pt_crd, 3);
         }
      }
   }
   // Dofs on Faces
   if (VelFieldShape->hasNodesIn(apf::Mesh::TRIANGLE))
   {
      const int nn = VelFieldShape->countNodesOn(apf::Mesh::TRIANGLE);
      for (int ii = 0; ii < 4; ii++)
      {
         for (int jj = 0; jj < nn; jj++)
         {
            VelFieldShape->getNodeXi(apf::Mesh::TRIANGLE, jj, xi_crd);
            double pt_crd[3] = {0., 0., 0.};
            switch (ii)
            {
               case 0:
                  pt_crd[0] = xi_crd[0];
                  pt_crd[1] = xi_crd[1];
                  break;
               case 1:
                  pt_crd[0] = xi_crd[0];
                  pt_crd[2] = xi_crd[2];
                  break;
               case 2:
                  pt_crd[0] = xi_crd[0];
                  pt_crd[1] = xi_crd[1];
                  pt_crd[2] = xi_crd[2];
                  break;
               case 3:
                  pt_crd[1] = xi_crd[0];
                  pt_crd[2] = xi_crd[1];
                  break;
            }
            IntegrationPoint& ip = pumi_nodes.IntPoint(ip_cnt++);
            ip.Set(pt_crd, 3);
         }
      }
   }
   MFEM_ASSERT(ip_cnt == num_nodes, "");

   // Other dofs
   apf::MeshEntity* ent;
   apf::MeshIterator* itr = apf_mesh->begin(3);
   int iel = 0;
   while ((ent = apf_mesh->iterate(itr)))
   {
      // Get the solution
      Vector u_vel, v_vel, w_vel;
      grid_vel->GetValues(iel, pumi_nodes, u_vel, 1);
      grid_vel->GetValues(iel, pumi_nodes, v_vel, 2);
      grid_vel->GetValues(iel, pumi_nodes, w_vel, 3);

      Vector pr;
      grid_pr->GetValues(iel, pumi_nodes, pr, 1);

      // Transfer
      apf::Downward vtxs;
      int num_vts = apf_mesh->getDownward(ent, 0, vtxs);
      for (int kk = 0; kk < num_vts; kk++)
      {
         double mag = u_vel[kk] * u_vel[kk] + v_vel[kk] * v_vel[kk] +
                      w_vel[kk] * w_vel[kk];
         mag = sqrt(mag);
         apf::setScalar(VelMagField, vtxs[kk], 0, mag);
         // Set vel
         double vels[3] = {u_vel[kk], v_vel[kk], w_vel[kk]};
         apf::setComponents(VelField, vtxs[kk], 0, vels);

         // Set Pr
         apf::setScalar(PrField, vtxs[kk], 0, pr[kk]);
      }

      int dofId = num_vts;

      apf::EntityShape* es = VelFieldShape->getEntityShape(apf::Mesh::TET);
      // Edge Dofs
      if (VelFieldShape->hasNodesIn(apf::Mesh::EDGE))
      {
         int ndOnEdge = VelFieldShape->countNodesOn(apf::Mesh::EDGE);
         Array<int> order(ndOnEdge);

         apf::Downward edges;
         int num_edge =  apf_mesh->getDownward(ent, apf::Mesh::EDGE, edges);
         for (int ii = 0 ; ii < num_edge; ++ii)
         {
            es->alignSharedNodes(apf_mesh, ent, edges[ii], order);
            for (int jj = 0; jj < ndOnEdge; jj++)
            {
               int cnt = dofId + order[jj];
               double mag = u_vel[cnt] * u_vel[cnt] +
                            v_vel[cnt] * v_vel[cnt] +
                            w_vel[cnt] * w_vel[cnt];
               mag = sqrt(mag);
               apf::setScalar(VelMagField, edges[ii], jj, mag);

               // Set vel
               double vels[3] = {u_vel[cnt], v_vel[cnt], w_vel[cnt]};
               apf::setComponents(VelField, edges[ii], jj, vels);

               // Set Pr
               apf::setScalar(PrField, edges[ii], jj, pr[cnt]);

            }
            // Counter
            dofId += ndOnEdge;
         }
      }
      // Face Dofs
      if (VelFieldShape->hasNodesIn(apf::Mesh::TRIANGLE))
      {
         int ndOnFace = VelFieldShape->countNodesOn(apf::Mesh::TRIANGLE);
         Array<int> order(ndOnFace);

         apf::Downward faces;
         int num_face = apf_mesh->getDownward(ent, apf::Mesh::TRIANGLE, faces);
         for (int ii = 0; ii < num_face; ii++)
         {
            if ( ndOnFace > 1)
            {
               es->alignSharedNodes(apf_mesh, ent, faces[ii], order);
            }
            else
            {
               order[0] = 0;
            }
            for (int jj = 0; jj < ndOnFace; jj++)
            {
               int cnt = dofId + order[jj];
               double mag = u_vel[cnt] * u_vel[cnt] +
                            v_vel[cnt] * v_vel[cnt] +
                            w_vel[cnt] * w_vel[cnt];
               mag = sqrt(mag);
               apf::setScalar(VelMagField, faces[ii], jj, mag);

               // Set vel
               double vels[3] = {u_vel[cnt], v_vel[cnt], w_vel[cnt]};
               apf::setComponents(VelField, faces[ii], jj, vels);

               // Set Pr
               apf::setScalar(PrField, faces[ii], jj, pr[cnt]);
            }
            // Counter
            dofId += ndOnFace;
         }
      }

      iel++;
   }
   apf_mesh->end(itr);
}

// Transfer a scalar field its magnitude to use for mesh adaptation.
void ParPumiMesh::FieldMFEMtoPUMI(apf::Mesh2* apf_mesh,
                                  ParGridFunction* grid_pr,
                                  apf::Field* PrField,
                                  apf::Field* PrMagField)
{
   apf::FieldShape* PrFieldShape = getShape(PrField);
   int num_nodes = 4 * PrFieldShape->countNodesOn(0) + // Vertex
                   6 * PrFieldShape->countNodesOn(1) + // Edge
                   4 * PrFieldShape->countNodesOn(2) + // Triangle
                   PrFieldShape->countNodesOn(4); // Tetrahedron

   // Define integration points
   IntegrationRule pumi_nodes(num_nodes);
   int ip_cnt = 0;
   apf::Vector3 xi_crd(0.,0.,0.);

   // Create a template of dof holders coordinates in parametric coordinates.
   // The ordering is taken care of when the field is transferred to PUMI.

   // Dofs on Vertices
   IntegrationPoint& ip = pumi_nodes.IntPoint(ip_cnt++);
   double pt_crd[3] = {0., 0., 0.};
   ip.Set(pt_crd, 3);
   for (int kk = 0; kk < 3; kk++)
   {
      IntegrationPoint& ip = pumi_nodes.IntPoint(ip_cnt++);
      double pt_crd[3] = {0.,0.,0.};
      pt_crd[kk] = 1.0;
      ip.Set(pt_crd, 3);
   }
   // Dofs on Edges
   if (PrFieldShape->hasNodesIn(apf::Mesh::EDGE))
   {
      const int nn = PrFieldShape->countNodesOn(apf::Mesh::EDGE);
      for (int ii = 0; ii < 6; ii++)
      {
         for (int jj = 0; jj < nn; jj++)
         {
            PrFieldShape->getNodeXi(apf::Mesh::EDGE, jj, xi_crd);
            xi_crd[0] = 0.5 * (xi_crd[0] + 1.); // from (-1,1) to (0,1)
            double pt_crd[3] = {0., 0., 0.};
            switch (ii)
            {
               case 0:
                  pt_crd[0] = xi_crd[0];
                  break;
               case 1:
                  pt_crd[0] = 1. - xi_crd[0];
                  pt_crd[1] = xi_crd[0];
                  break;
               case 2:
                  pt_crd[1] = xi_crd[0];
                  break;
               case 3:
                  pt_crd[2] = xi_crd[0];
                  break;
               case 4:
                  pt_crd[0] = 1. - xi_crd[0];
                  pt_crd[2] = xi_crd[0];
                  break;
               case 5:
                  pt_crd[1] = 1. - xi_crd[0];
                  pt_crd[2] = xi_crd[0];
                  break;
            }
            IntegrationPoint& ip = pumi_nodes.IntPoint(ip_cnt++);
            ip.Set(pt_crd, 3);
         }
      }
   }
   // Dofs on Faces
   if (PrFieldShape->hasNodesIn(apf::Mesh::TRIANGLE))
   {
      const int nn = PrFieldShape->countNodesOn(apf::Mesh::TRIANGLE);
      for (int ii = 0; ii < 4; ii++)
      {
         for (int jj = 0; jj < nn; jj++)
         {
            PrFieldShape->getNodeXi(apf::Mesh::TRIANGLE, jj, xi_crd);
            double pt_crd[3] = {0., 0., 0.};
            switch (ii)
            {
               case 0:
                  pt_crd[0] = xi_crd[0];
                  pt_crd[1] = xi_crd[1];
                  break;
               case 1:
                  pt_crd[0] = xi_crd[0];
                  pt_crd[2] = xi_crd[2];
                  break;
               case 2:
                  pt_crd[0] = xi_crd[0];
                  pt_crd[1] = xi_crd[1];
                  pt_crd[2] = xi_crd[2];
                  break;
               case 3:
                  pt_crd[1] = xi_crd[0];
                  pt_crd[2] = xi_crd[1];
                  break;
            }
            IntegrationPoint& ip = pumi_nodes.IntPoint(ip_cnt++);
            ip.Set(pt_crd, 3);
         }
      }
   }
   MFEM_ASSERT(ip_cnt == num_nodes, "");

   // Other dofs
   apf::MeshEntity* ent;
   apf::MeshIterator* itr = apf_mesh->begin(3);
   int iel = 0;
   while ((ent = apf_mesh->iterate(itr)))
   {
      // Get the solution
      Vector pr;
      grid_pr->GetValues(iel, pumi_nodes, pr, 1);

      // Transfer
      apf::Downward vtxs;
      int num_vts = apf_mesh->getDownward(ent, 0, vtxs);
      for (int kk = 0; kk < num_vts; kk++)
      {
         double mag;
         (pr[kk] >= 0. ? mag = pr[kk] : mag = -pr[kk]);
         apf::setScalar(PrMagField, vtxs[kk], 0, mag);

         // Set Pr
         apf::setScalar(PrField, vtxs[kk], 0, pr[kk]);
      }

      int dofId = num_vts;

      apf::EntityShape* es = PrFieldShape->getEntityShape(apf::Mesh::TET);
      // Edge Dofs
      if (PrFieldShape->hasNodesIn(apf::Mesh::EDGE))
      {
         int ndOnEdge = PrFieldShape->countNodesOn(apf::Mesh::EDGE);
         Array<int> order(ndOnEdge);

         apf::Downward edges;
         int num_edge =  apf_mesh->getDownward(ent, apf::Mesh::EDGE, edges);
         for (int ii = 0 ; ii < num_edge; ++ii)
         {
            es->alignSharedNodes(apf_mesh, ent, edges[ii], order);
            for (int jj = 0; jj < ndOnEdge; jj++)
            {
               int cnt = dofId + order[jj];
               double mag;
               (pr[cnt] >= 0. ? mag = pr[cnt] : mag = -pr[cnt]);
               apf::setScalar(PrMagField, edges[ii], jj, mag);

               // Set Pr
               apf::setScalar(PrField, edges[ii], jj, pr[cnt]);

            }
            // Counter
            dofId += ndOnEdge;
         }
      }

      // Face Dofs
      if (PrFieldShape->hasNodesIn(apf::Mesh::TRIANGLE))
      {
         int ndOnFace = PrFieldShape->countNodesOn(apf::Mesh::TRIANGLE);
         Array<int> order(ndOnFace);

         apf::Downward faces;
         int num_face = apf_mesh->getDownward(ent, apf::Mesh::TRIANGLE, faces);
         for (int ii = 0; ii < num_face; ii++)
         {
            if ( ndOnFace > 1)
            {
               es->alignSharedNodes(apf_mesh, ent, faces[ii], order);
            }
            else
            {
               order[0] = 0;
            }
            for (int jj = 0; jj < ndOnFace; jj++)
            {
               int cnt = dofId + order[jj];
               double mag;
               (pr[cnt] >= 0. ? mag = pr[cnt] : mag = -pr[cnt]);
               apf::setScalar(PrMagField, faces[ii], jj, mag);

               // Set Pr
               apf::setScalar(PrField, faces[ii], jj, pr[cnt]);
            }
            // Counter
            dofId += ndOnFace;
         }
      }

      iel++;
   }
   apf_mesh->end(itr);
}

// Transfer a vector field and the magnitude of the vector field to use for mesh
// adaptation
void ParPumiMesh::VectorFieldMFEMtoPUMI(apf::Mesh2* apf_mesh,
                                        ParGridFunction* grid_vel,
                                        apf::Field* VelField,
                                        apf::Field* VelMagField)
{
   apf::FieldShape* VelFieldShape = getShape(VelField);
   int num_nodes = 4 * VelFieldShape->countNodesOn(0) + // Vertex
                   6 * VelFieldShape->countNodesOn(1) + // Edge
                   4 * VelFieldShape->countNodesOn(2) + // Triangle
                   VelFieldShape->countNodesOn(4);// Tetrahedron

   // Define integration points
   IntegrationRule pumi_nodes(num_nodes);
   int ip_cnt = 0;
   apf::Vector3 xi_crd(0.,0.,0.);

   // Create a template of dof holders coordinates in parametric coordinates.
   // The ordering is taken care of when the field is transferred to PUMI.

   // Dofs on Vertices
   IntegrationPoint& ip = pumi_nodes.IntPoint(ip_cnt++);
   double pt_crd[3] = {0., 0., 0.};
   ip.Set(pt_crd, 3);
   for (int kk = 0; kk < 3; kk++)
   {
      IntegrationPoint& ip = pumi_nodes.IntPoint(ip_cnt++);
      double pt_crd[3] = {0.,0.,0.};
      pt_crd[kk] = 1.0;
      ip.Set(pt_crd, 3);
   }
   // Dofs on Edges
   if (VelFieldShape->hasNodesIn(apf::Mesh::EDGE))
   {
      const int nn = VelFieldShape->countNodesOn(apf::Mesh::EDGE);
      for (int ii = 0; ii < 6; ii++)
      {
         for (int jj = 0; jj < nn; jj++)
         {
            VelFieldShape->getNodeXi(apf::Mesh::EDGE, jj, xi_crd);
            xi_crd[0] = 0.5 * (xi_crd[0] + 1.); // from (-1,1) to (0,1)
            double pt_crd[3] = {0., 0., 0.};
            switch (ii)
            {
               case 0:
                  pt_crd[0] = xi_crd[0];
                  break;
               case 1:
                  pt_crd[0] = 1. - xi_crd[0];
                  pt_crd[1] = xi_crd[0];
                  break;
               case 2:
                  pt_crd[1] = xi_crd[0];
                  break;
               case 3:
                  pt_crd[2] = xi_crd[0];
                  break;
               case 4:
                  pt_crd[0] = 1. - xi_crd[0];
                  pt_crd[2] = xi_crd[0];
                  break;
               case 5:
                  pt_crd[1] = 1. - xi_crd[0];
                  pt_crd[2] = xi_crd[0];
                  break;
            }
            IntegrationPoint& ip = pumi_nodes.IntPoint(ip_cnt++);
            ip.Set(pt_crd, 3);
         }
      }
   }
   // Dofs on Faces
   if (VelFieldShape->hasNodesIn(apf::Mesh::TRIANGLE))
   {
      const int nn = VelFieldShape->countNodesOn(apf::Mesh::TRIANGLE);
      for (int ii = 0; ii < 4; ii++)
      {
         for (int jj = 0; jj < nn; jj++)
         {
            VelFieldShape->getNodeXi(apf::Mesh::TRIANGLE, jj, xi_crd);
            double pt_crd[3] = {0., 0., 0.};
            switch (ii)
            {
               case 0:
                  pt_crd[0] = xi_crd[0];
                  pt_crd[1] = xi_crd[1];
                  break;
               case 1:
                  pt_crd[0] = xi_crd[0];
                  pt_crd[2] = xi_crd[2];
                  break;
               case 2:
                  pt_crd[0] = xi_crd[0];
                  pt_crd[1] = xi_crd[1];
                  pt_crd[2] = xi_crd[2];
                  break;
               case 3:
                  pt_crd[1] = xi_crd[0];
                  pt_crd[2] = xi_crd[1];
                  break;
            }
            IntegrationPoint& ip = pumi_nodes.IntPoint(ip_cnt++);
            ip.Set(pt_crd, 3);
         }
      }
   }
   MFEM_ASSERT(ip_cnt == num_nodes, "");

   // Other dofs
   apf::MeshEntity* ent;
   apf::MeshIterator* itr = apf_mesh->begin(3);
   int iel = 0;
   while ((ent = apf_mesh->iterate(itr)))
   {
      // Get the solution
      Vector u_vel, v_vel, w_vel;
      grid_vel->GetValues(iel, pumi_nodes, u_vel, 1);
      grid_vel->GetValues(iel, pumi_nodes, v_vel, 2);
      grid_vel->GetValues(iel, pumi_nodes, w_vel, 3);

      // Transfer
      apf::Downward vtxs;
      int num_vts = apf_mesh->getDownward(ent, 0, vtxs);
      for (int kk = 0; kk < num_vts; kk++)
      {
         double mag = u_vel[kk] * u_vel[kk] + v_vel[kk] * v_vel[kk] +
                      w_vel[kk] * w_vel[kk];
         mag = sqrt(mag);
         apf::setScalar(VelMagField, vtxs[kk], 0, mag);
         // Set vel
         double vels[3] = {u_vel[kk], v_vel[kk], w_vel[kk]};
         apf::setComponents(VelField, vtxs[kk], 0, vels);
      }

      int dofId = num_vts;

      apf::EntityShape* es = VelFieldShape->getEntityShape(apf::Mesh::TET);
      // Edge Dofs
      if (VelFieldShape->hasNodesIn(apf::Mesh::EDGE))
      {
         int ndOnEdge = VelFieldShape->countNodesOn(apf::Mesh::EDGE);
         Array<int> order(ndOnEdge);

         apf::Downward edges;
         int num_edge =  apf_mesh->getDownward(ent, apf::Mesh::EDGE, edges);
         for (int ii = 0 ; ii < num_edge; ++ii)
         {
            es->alignSharedNodes(apf_mesh, ent, edges[ii], order);
            for (int jj = 0; jj < ndOnEdge; jj++)
            {
               int cnt = dofId + order[jj];
               double mag = u_vel[cnt] * u_vel[cnt] +
                            v_vel[cnt] * v_vel[cnt] +
                            w_vel[cnt] * w_vel[cnt];
               mag = sqrt(mag);
               apf::setScalar(VelMagField, edges[ii], jj, mag);

               // Set vel
               double vels[3] = {u_vel[cnt], v_vel[cnt], w_vel[cnt]};
               apf::setComponents(VelField, edges[ii], jj, vels);
            }
            // Counter
            dofId += ndOnEdge;
         }
      }

      // Face Dofs
      if (VelFieldShape->hasNodesIn(apf::Mesh::TRIANGLE))
      {
         int ndOnFace = VelFieldShape->countNodesOn(apf::Mesh::TRIANGLE);
         Array<int> order(ndOnFace);

         apf::Downward faces;
         int num_face = apf_mesh->getDownward(ent, apf::Mesh::TRIANGLE, faces);
         for (int ii = 0; ii < num_face; ii++)
         {
            if ( ndOnFace > 1)
            {
               es->alignSharedNodes(apf_mesh, ent, faces[ii], order);
            }
            else
            {
               order[0] = 0;
            }
            for (int jj = 0; jj < ndOnFace; jj++)
            {
               int cnt = dofId + order[jj];
               double mag = u_vel[cnt] * u_vel[cnt] +
                            v_vel[cnt] * v_vel[cnt] +
                            w_vel[cnt] * w_vel[cnt];
               mag = sqrt(mag);
               apf::setScalar(VelMagField, faces[ii], jj, mag);

               // Set vel
               double vels[3] = {u_vel[cnt], v_vel[cnt], w_vel[cnt]};
               apf::setComponents(VelField, faces[ii], jj, vels);
            }
            // Counter
            dofId += ndOnFace;
         }
      }

      iel++;
   }
   apf_mesh->end(itr);
}

void ParPumiMesh::FieldPUMItoMFEM(apf::Mesh2* apf_mesh,
                                  apf::Field* ScalarField,
                                  ParGridFunction* Pr)
{
   // Pr->Update();
   // Find local numbering
   v_num_loc = apf_mesh->findNumbering("LocalVertexNumbering");

   // Loop over field to copy
   getShape(ScalarField);
   apf::MeshEntity* ent;
   apf::MeshIterator* itr = apf_mesh->begin(0);
   while ((ent = apf_mesh->iterate(itr)))
   {
      unsigned int id = apf::getNumber(v_num_loc, ent, 0, 0);
      double fieldVal = apf::getScalar(ScalarField, ent, 0);

      (Pr->GetData())[id] = fieldVal;
   }
   apf_mesh->end(itr);

   // Check for higher order
   getShape(ScalarField);
   if ( Pr->FESpace()->GetOrder(1) > 1 )
   {
      // Assume all element type are the same i.e. tetrahedral
      const FiniteElement* H1_elem = Pr->FESpace()->GetFE(1);
      const IntegrationRule &All_nodes = H1_elem->GetNodes();
      int nnodes = All_nodes.Size();

      // Loop over elements
      int nc = apf::countComponents(ScalarField);
      int iel = 0;
      itr = apf_mesh->begin(3);
      while ((ent = apf_mesh->iterate(itr)))
      {
         Array<int> vdofs;
         Pr->FESpace()->GetElementVDofs(iel, vdofs);

         // Create PUMI element to interpolate
         apf::MeshElement* mE = apf::createMeshElement(apf_mesh, ent);
         apf::Element* elem = apf::createElement(ScalarField, mE);

         // Vertices are already interpolated
         for (int ip = 0; ip < nnodes; ip++) //num_vert
         {
            // Take parametric coordinates of the node
            apf::Vector3 param;
            param[0] = All_nodes.IntPoint(ip).x;
            param[1] = All_nodes.IntPoint(ip).y;
            param[2] = All_nodes.IntPoint(ip).z;

            // Compute the interpolating coordinates
            apf::DynamicVector phCrd(nc);
            apf::getComponents(elem, param, &phCrd[0]);

            // Fill the nodes list
            for (int kk = 0; kk < nc; ++kk)
            {
               int dof_ctr = ip + kk * nnodes;
               (Pr->GetData())[vdofs[dof_ctr]] = phCrd[kk];
            }
         }
         iel++;
         apf::destroyElement(elem);
         apf::destroyMeshElement(mE);
      }
      apf_mesh->end(itr);
   }
}

}

#endif // MFEM_USE_MPI
#endif // MFEM_USE_SCOREC
