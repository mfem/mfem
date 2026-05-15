// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "mfem.hpp"
#include "mesh-optimizer.hpp"

using namespace mfem;

// Go over the dofs and set bitwise flags in attr_marker based on boundary
// attribute the dof touches.
// attr_count tells how many attributes each dof touches.
// works for both serial and vector spaces.
void SetupDofAttributes(ParGridFunction &attr_count, Array<int> &attr_marker)
{
   ParFiniteElementSpace *pfesmarker = attr_count.ParFESpace();
   ParMesh *pmesh = pfesmarker->GetParMesh();
   attr_marker.SetSize(attr_count.Size());
   attr_marker = 0;
   Array<int> dofs;
   int nbdr_faces = pmesh->GetNFbyType(FaceType::Boundary);
   for (int f = 0; f < nbdr_faces; f++)
   {
      int attrib = pmesh->GetBdrAttribute(f);
      pfesmarker->GetBdrElementVDofs(f, dofs);
      for (int i = 0; i < dofs.Size(); i++)
      {
         int val = attr_marker[dofs[i]];
         attr_marker[dofs[i]] = val | (1 << (attrib-1));
      }
   }

   GroupCommunicator &gcomm = attr_count.ParFESpace()->GroupComm();
   gcomm.Reduce<int>(attr_marker, GroupCommunicator::BitOR);
   gcomm.Bcast(attr_marker);

   auto countBits = [](int val)
   {
      int count = 0;
      while (val)
      {
         count += val & 1;
         val >>= 1;
      }
      return count;
   };

   for (int i = 0; i < attr_count.Size(); i++)
   {
      attr_count(i) = countBits(attr_marker[i]);
   }
}

// Mark only the dofs that are on interfaces between different attributes.
// The attribute for such dofs is element attribute + max_bdr_attr

void RemoveDofsFromBdrFaceDofs(Array<int> &bdr_face_dofs,
                               Array<int> &bdr_edge_dofs)
{
   // Remove the edge dofs from the face dofs.
   for (int i = 0; i < bdr_face_dofs.Size(); i++)
   {
      int dof = bdr_face_dofs[i];
      if (bdr_edge_dofs.Find(dof) != -1)
      {
         bdr_face_dofs[i] = -1; // mark for removal
      }
   }
   bdr_face_dofs.Sort();
   bdr_face_dofs.Unique(); // remove duplicates
   // Remove the marked dof
   bdr_face_dofs.DeleteFirst(-1);
}

// Get indices of all dofs that have more than or equal to `dim` attributes.
// Essentially dofs that should not be allowed to move.
// Works for both scalar and vector spaces
Array<int> IdentifyAuxiliaryEssentialDofs(ParGridFunction &attr_count)
{
   ParFiniteElementSpace *pfespace = attr_count.ParFESpace();
   Array<int> aux_dofs;
   int dim = pfespace->GetParMesh()->SpaceDimension();
   for (int i = 0; i < attr_count.Size(); i++)
   {
      if (attr_count[i] >= dim)
      {
         aux_dofs.Append(i);
      }
   }
   return aux_dofs;
}

// Get indices of all dofs that have only the required attribute marked.
Array<int> *GetDofMatchingBdrFaceAttributes(ParGridFunction &attr_count,
                                            Array<int> &attr_marker,
                                            int attr)
{
   ParFiniteElementSpace *pfespace = attr_count.ParFESpace();
   Array<int> *facedofs = new Array<int>;
   int vdim = pfespace->GetVDim();
   MFEM_VERIFY(vdim == 1, "Only scalar spaces should be used.");

   // check if jth bit is set
   auto hasBitSet = [](int val, int j)
   {
      int mask = (1 << (j-1));
      return (val & mask) == mask;
   };

   for (int i = 0; i < attr_count.Size(); i++)
   {
      if (attr_count[i] == 1)
      {
         int val = attr_marker[i];
         if (hasBitSet(val, attr))
         {
            facedofs->Append(i);
         }
      }
   }
   facedofs->Sort();
   facedofs->Unique();
   return facedofs;
}

// Get indices of all dofs that have only the required attribute marked.
Array<int> *GetDofMatchingBdrEdgeAttributes(ParGridFunction &attr_count,
                                            Array<int> &attr_marker,
                                            int attr1, int attr2)
{
   ParFiniteElementSpace *pfespace = attr_count.ParFESpace();
   Array<int> *edgedofs = new Array<int>;
   int vdim = pfespace->GetVDim();
   MFEM_VERIFY(vdim == 1, "Only scalar spaces should be used.");

   // check if jth and kth bits are set
   auto hasAtLeastTwoBitsSet = [](int val, int j, int k)
   {
      int mask = (1 << (j-1)) | (1 << (k-1));
      return (val & mask) == mask;
   };

   for (int i = 0; i < attr_count.Size(); i++)
   {
      if (attr_count[i] == 2)
      {
         int val = attr_marker[i];
         if (hasAtLeastTwoBitsSet(val, attr1, attr2))
         {
            edgedofs->Append(i);
         }
      }
   }
   edgedofs->Sort();
   edgedofs->Unique();
   return edgedofs;
}

// Get indices of all dofs that have only the required attribute marked.

void SetupSerialDofAttributes(GridFunction &attr_count, Array<int> &attr_marker)
{
   FiniteElementSpace *fesmarker = attr_count.FESpace();
   Mesh *mesh = fesmarker->GetMesh();
   attr_marker.SetSize(attr_count.Size());
   attr_marker = 0;
   Array<int> dofs;
   int nbdr_faces = mesh->GetNFbyType(FaceType::Boundary);
   for (int f = 0; f < nbdr_faces; f++)
   {
      int attrib = mesh->GetBdrAttribute(f);
      fesmarker->GetBdrElementVDofs(f, dofs);
      for (int i = 0; i < dofs.Size(); i++)
      {
         int val = attr_marker[dofs[i]];
         attr_marker[dofs[i]] = val | (1 << (attrib-1));
      }
   }

   auto countBits = [](int val)
   {
      int count = 0;
      while (val)
      {
         count += val & 1;
         val >>= 1;
      }
      return count;
   };

   for (int i = 0; i < attr_count.Size(); i++)
   {
      attr_count(i) = countBits(attr_marker[i]);
   }
}

Mesh *SetupEdgeMesh3D(Mesh *mesh, GridFunction &attr_count_ser,
                      Array<int> &attr_marker_ser,
                      int attr1, int attr2)
{
   Array<int> facedofs(0);
   int spaceDim = mesh->SpaceDimension();
   MFEM_VERIFY(spaceDim == 3, "Only 2D meshes supported right now.");
   Array<int> edofs;
   GridFunction *x = mesh->GetNodes();
   MFEM_VERIFY(x, "Mesh nodal space not set\n");
   const FiniteElementSpace *fes = mesh->GetNodalFESpace();
   int mesh_poly_deg = fes->GetMaxElementOrder();
   int nedges = mesh->GetNEdges();

   FiniteElementSpace *fespace_attr = attr_count_ser.FESpace();

   Array<int> ev, dofs;
   int attr_marker_dofs[2], attr_count_dofs[2];
   Array<int> edge_to_include;
   for (int ei = 0; ei < nedges; ei++)
   {
      mesh->GetEdgeVertices(ei, ev);
      MFEM_VERIFY(ev.Size(), "Could not get edge vertices.");
      for (int j = 0; j < 2; j++)
      {
         int v_id = ev[j];
         fespace_attr->GetVertexDofs(v_id, dofs);
         attr_marker_dofs[j] = attr_marker_ser[dofs[0]];
         attr_count_dofs[j] = attr_count_ser[dofs[0]];
      }
      auto hasAtLeastTwoBitsSet = [](int val, int j, int k)
      {
         int mask = (1 << (j-1)) | (1 << (k-1));
         return (val & mask) == mask;
      };
      bool dof1_mask2 = hasAtLeastTwoBitsSet(attr_marker_dofs[0], attr1, attr2);
      bool dof2_mask2 = hasAtLeastTwoBitsSet(attr_marker_dofs[1], attr1, attr2);
      if ( (dof1_mask2 && dof2_mask2))
      {
         edge_to_include.Append(ei);
      }
   }

   // Setup a mesh with dummy vertices
   int nel = edge_to_include.Size();
   Vector vals;
   Mesh *intmesh = new Mesh(1, nel*2, nel, 0, spaceDim);
   {
      for (int i = 0; i < nel; i++)
      {
         for (int j = 0; j < 2; j++) // 2 vertices per element
         {
            Vector vert(spaceDim);
            vert = 0.5;
            intmesh->AddVertex(vert.GetData());
         }
         Array<int> verts(2);
         for (int d = 0; d < 2; d++)
         {
            verts[d] = i*2+d;
         }
         intmesh->AddSegment(verts, 1);
      }
      intmesh->Finalize(true, true);
      intmesh->FinalizeTopology(false);
      intmesh->SetCurvature(mesh_poly_deg, false, -1, 0);
   }

   const FiniteElementSpace *intnodespace = intmesh->GetNodalFESpace();
   GridFunction *intnodes = intmesh->GetNodes();

   for (int i = 0; i < nel; i++)
   {
      int ei = edge_to_include[i];
      fes->GetEdgeVDofs(ei, dofs);
      x->GetSubVector(dofs, vals);
      Array<int> edofs;
      intnodespace->GetElementVDofs(i, edofs);
      intnodes->SetSubVector(edofs, vals);
   }

   return intmesh;
}

// Get indices of all edges that have both the required attributes marked.

Mesh *SetupFaceMesh3D(Mesh *mesh, int attr)
{
   Array<int> facedofs(0);
   int spaceDim = mesh->SpaceDimension();
   MFEM_VERIFY(spaceDim == 3, "Only 2D meshes supported right now.");
   Array<int> fdofs;
   GridFunction *x = mesh->GetNodes();
   MFEM_VERIFY(x, "Mesh nodal space not set\n");
   const FiniteElementSpace *fes = mesh->GetNodalFESpace();
   int mesh_poly_deg = fes->GetMaxElementOrder();

   int nbdr_faces = mesh->GetNFbyType(FaceType::Boundary);
   Array<int> faces_to_include;
   for (int f = 0; f < nbdr_faces; f++)
   {
      int attrib = mesh->GetBdrAttribute(f);
      if (attrib == attr)
      {
         faces_to_include.Append(mesh->GetBdrElementFaceIndex(f));
      }
   }

   // Setup a mesh with dummy vertices
   int nel = faces_to_include.Size();
   Vector vals;
   Mesh *intmesh = new Mesh(2, nel*4, nel, 0, spaceDim);
   {
      for (int i = 0; i < nel; i++)
      {
         for (int j = 0; j < 4; j++) // 4 vertices per element
         {
            Vector vert(spaceDim);
            vert = 0.5;
            intmesh->AddVertex(vert.GetData());
         }
         Array<int> verts(4);
         for (int d = 0; d < 4; d++)
         {
            verts[d] = i*4+d;
         }
         intmesh->AddQuad(verts, 1);
      }
      intmesh->Finalize(true, true);
      intmesh->FinalizeTopology(false);
      intmesh->SetCurvature(mesh_poly_deg, false, -1, 0);
   }

   const FiniteElementSpace *intnodespace = intmesh->GetNodalFESpace();
   GridFunction *intnodes = intmesh->GetNodes();

   for (int i = 0; i < nel; i++)
   {
      int fi = faces_to_include[i];
      fes->GetFaceVDofs(fi, fdofs);
      x->GetSubVector(fdofs, vals);
      intnodespace->GetElementVDofs(i, fdofs);
      intnodes->SetSubVector(fdofs, vals);
   }

   return intmesh;
}

Mesh *SetupEdgeMesh2D(Mesh *mesh, GridFunction &attr_count_ser,
                      Array<int> &attr_marker_ser, int attr)
{
   Array<int> facedofs(0);
   int spaceDim = mesh->SpaceDimension();
   MFEM_VERIFY(spaceDim == 2, "Only 2D meshes supported right now.");
   Array<int> edofs;
   GridFunction *x = mesh->GetNodes();
   MFEM_VERIFY(x, "Mesh nodal space not set\n");
   const FiniteElementSpace *fes = mesh->GetNodalFESpace();
   int mesh_poly_deg = fes->GetMaxElementOrder();
   int nedges = mesh->GetNEdges();

   FiniteElementSpace *fespace_attr = attr_count_ser.FESpace();

   Array<int> ev, dofs;
   int attr_marker_dofs[2], attr_count_dofs[2];
   Array<int> edge_to_include;
   for (int ei = 0; ei < nedges; ei++)
   {
      mesh->GetEdgeVertices(ei, ev);
      MFEM_VERIFY(ev.Size(), "Could not get edge vertices.");
      for (int j = 0; j < 2; j++)
      {
         int v_id = ev[j];
         fespace_attr->GetVertexDofs(v_id, dofs);
         attr_marker_dofs[j] = attr_marker_ser[dofs[0]];
         attr_count_dofs[j] = attr_count_ser[dofs[0]];
      }
      auto hasAtLeastOneBitSet = [](int val, int j)
      {
         int mask = (1 << (j-1));
         return (val & mask) == mask;
      };
      bool dof1_mask2 = hasAtLeastOneBitSet(attr_marker_dofs[0], attr);
      bool dof2_mask2 = hasAtLeastOneBitSet(attr_marker_dofs[1], attr);
      if ( (dof1_mask2 && dof2_mask2))
      {
         edge_to_include.Append(ei);
      }
   }

   // Setup a mesh with dummy vertices
   int nel = edge_to_include.Size();
   Vector vals;
   Mesh *intmesh = new Mesh(1, nel*2, nel, 0, spaceDim);
   {
      for (int i = 0; i < nel; i++)
      {
         for (int j = 0; j < 2; j++) // 2 vertices per element
         {
            Vector vert(spaceDim);
            vert = 0.5;
            intmesh->AddVertex(vert.GetData());
         }
         Array<int> verts(2);
         for (int d = 0; d < 2; d++)
         {
            verts[d] = i*2+d;
         }
         intmesh->AddSegment(verts, 1);
      }
      intmesh->Finalize(true, true);
      intmesh->FinalizeTopology(false);
      intmesh->SetCurvature(mesh_poly_deg, false, -1, 0);
   }

   const FiniteElementSpace *intnodespace = intmesh->GetNodalFESpace();
   GridFunction *intnodes = intmesh->GetNodes();

   for (int i = 0; i < nel; i++)
   {
      int ei = edge_to_include[i];
      fes->GetEdgeVDofs(ei, dofs);
      x->GetSubVector(dofs, vals);
      Array<int> edofs;
      intnodespace->GetElementVDofs(i, edofs);
      intnodes->SetSubVector(edofs, vals);
   }

   return intmesh;
}

real_t GetMinDet(ParMesh *pmesh,
                 ParGridFunction &x,
                 int quad_order)
{
   int NE = pmesh->GetNE();
   ParFiniteElementSpace &pfes = *(x.ParFESpace());
   real_t min_det = infinity();
   for (int e = 0; e < NE; e++)
   {
      const IntegrationRule &ir =
         IntRulesLo.Get(pfes.GetFE(e)->GetGeomType(), quad_order);
      ElementTransformation *transf = pmesh->GetElementTransformation(e);
      for (int q = 0; q < ir.GetNPoints(); q++)
      {
         transf->SetIntPoint(&ir.IntPoint(q));
         real_t det = transf->Jacobian().Det();
         min_det = fmin(min_det, det);
      }
   }
   MPI_Allreduce(MPI_IN_PLACE, &min_det, 1, MPI_DOUBLE,
                 MPI_MIN, pfes.GetComm());
   return min_det;
}
