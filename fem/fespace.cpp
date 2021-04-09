// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

// Implementation of FiniteElementSpace

#include "../general/text.hpp"
#include "../general/forall.hpp"
#include "../mesh/mesh_headers.hpp"
#include "fem.hpp"
#include "ceed/util.hpp"

#include <cmath>
#include <cstdarg>

using namespace std;

namespace mfem
{

template <> void Ordering::
DofsToVDofs<Ordering::byNODES>(int ndofs, int vdim, Array<int> &dofs)
{
   // static method
   int size = dofs.Size();
   dofs.SetSize(size*vdim);
   for (int vd = 1; vd < vdim; vd++)
   {
      for (int i = 0; i < size; i++)
      {
         dofs[i+size*vd] = Map<byNODES>(ndofs, vdim, dofs[i], vd);
      }
   }
}

template <> void Ordering::
DofsToVDofs<Ordering::byVDIM>(int ndofs, int vdim, Array<int> &dofs)
{
   // static method
   int size = dofs.Size();
   dofs.SetSize(size*vdim);
   for (int vd = vdim-1; vd >= 0; vd--)
   {
      for (int i = 0; i < size; i++)
      {
         dofs[i+size*vd] = Map<byVDIM>(ndofs, vdim, dofs[i], vd);
      }
   }
}


FiniteElementSpace::FiniteElementSpace()
   : mesh(NULL), fec(NULL), vdim(0), ordering(Ordering::byNODES),
     ndofs(0), nvdofs(0), nedofs(0), nfdofs(0), nbdofs(0),
     bdofs(NULL),
     elem_dof(NULL), elem_fos(NULL), bdr_elem_dof(NULL), bdr_elem_fos(NULL),
     face_dof(NULL),
     NURBSext(NULL), own_ext(false),
	DoFTrans(0), VDoFTrans(vdim, ordering),
     cP(NULL), cR(NULL), cR_hp(NULL), cP_is_set(false),
     Th(Operator::ANY_TYPE),
     sequence(0), mesh_sequence(0), orders_changed(false), relaxed_hp(false)
{ }

FiniteElementSpace::FiniteElementSpace(const FiniteElementSpace &orig,
                                       Mesh *mesh,
                                       const FiniteElementCollection *fec)
   : VDoFTrans(orig.vdim, orig.ordering)
{
   mesh = mesh ? mesh : orig.mesh;
   fec = fec ? fec : orig.fec;

   NURBSExtension *NURBSext = NULL;
   if (orig.NURBSext && orig.NURBSext != orig.mesh->NURBSext)
   {
#ifdef MFEM_USE_MPI
      ParNURBSExtension *pNURBSext =
         dynamic_cast<ParNURBSExtension *>(orig.NURBSext);
      if (pNURBSext)
      {
         NURBSext = new ParNURBSExtension(*pNURBSext);
      }
      else
#endif
      {
         NURBSext = new NURBSExtension(*orig.NURBSext);
      }
   }

   Constructor(mesh, NURBSext, fec, orig.vdim, orig.ordering);
}

void FiniteElementSpace::SetElementOrder(int i, int p)
{
   MFEM_VERIFY(mesh_sequence == mesh->GetSequence(),
               "Space has not been Updated() after a Mesh change.");
   MFEM_VERIFY(i >= 0 && i < GetNE(), "Invalid element index");
   MFEM_VERIFY(p >= 0 && p <= MaxVarOrder, "Order out of range");
   MFEM_ASSERT(!elem_order.Size() || elem_order.Size() == GetNE(),
               "Internal error");

   if (elem_order.Size()) // already a variable-order space
   {
      if (elem_order[i] != p)
      {
         elem_order[i] = p;
         orders_changed = true;
      }
   }
   else // convert space to variable-order space
   {
      elem_order.SetSize(GetNE());
      elem_order = fec->GetOrder();

      elem_order[i] = p;
      orders_changed = true;
   }
}

int FiniteElementSpace::GetElementOrder(int i) const
{
   MFEM_VERIFY(mesh_sequence == mesh->GetSequence(),
               "Space has not been Updated() after a Mesh change.");
   MFEM_VERIFY(i >= 0 && i < GetNE(), "Invalid element index");
   MFEM_ASSERT(!elem_order.Size() || elem_order.Size() == GetNE(),
               "Internal error");

   return GetElementOrderImpl(i);
}

int FiniteElementSpace::GetElementOrderImpl(int i) const
{
   // (this is an internal version of GetElementOrder without asserts and checks)
   return elem_order.Size() ? elem_order[i] : fec->GetOrder();
}

void FiniteElementSpace::DofsToVDofs (Array<int> &dofs, int ndofs) const
{
   if (vdim == 1) { return; }
   if (ndofs < 0) { ndofs = this->ndofs; }

   if (ordering == Ordering::byNODES)
   {
      Ordering::DofsToVDofs<Ordering::byNODES>(ndofs, vdim, dofs);
   }
   else
   {
      Ordering::DofsToVDofs<Ordering::byVDIM>(ndofs, vdim, dofs);
   }
}

void FiniteElementSpace::DofsToVDofs(int vd, Array<int> &dofs, int ndofs) const
{
   if (vdim == 1) { return; }
   if (ndofs < 0) { ndofs = this->ndofs; }

   if (ordering == Ordering::byNODES)
   {
      for (int i = 0; i < dofs.Size(); i++)
      {
         dofs[i] = Ordering::Map<Ordering::byNODES>(ndofs, vdim, dofs[i], vd);
      }
   }
   else
   {
      for (int i = 0; i < dofs.Size(); i++)
      {
         dofs[i] = Ordering::Map<Ordering::byVDIM>(ndofs, vdim, dofs[i], vd);
      }
   }
}

int FiniteElementSpace::DofToVDof(int dof, int vd, int ndofs) const
{
   if (vdim == 1) { return dof; }
   if (ndofs < 0) { ndofs = this->ndofs; }

   if (ordering == Ordering::byNODES)
   {
      return Ordering::Map<Ordering::byNODES>(ndofs, vdim, dof, vd);
   }
   else
   {
      return Ordering::Map<Ordering::byVDIM>(ndofs, vdim, dof, vd);
   }
}

// static function
void FiniteElementSpace::AdjustVDofs (Array<int> &vdofs)
{
   int n = vdofs.Size(), *vdof = vdofs;
   for (int i = 0; i < n; i++)
   {
      int j;
      if ((j = vdof[i]) < 0)
      {
         vdof[i] = -1-j;
      }
   }
}

DofTransformation *
FiniteElementSpace::GetElementVDofs(int i, Array<int> &vdofs) const
{
   DofTransformation * doftrans = GetElementDofs(i, vdofs);
   DofsToVDofs(vdofs);
   if (vdim == 1 || doftrans == NULL)
   {
      return doftrans;
   }
   else
   {
      VDoFTrans.SetDofTransformation(*doftrans);
      return &VDoFTrans;
   }
}

DofTransformation *
FiniteElementSpace::GetBdrElementVDofs(int i, Array<int> &vdofs) const
{
   DofTransformation * doftrans = GetBdrElementDofs(i, vdofs);
   DofsToVDofs(vdofs);
   if (vdim == 1 || doftrans == NULL)
   {
      return doftrans;
   }
   else
   {
      VDoFTrans.SetDofTransformation(*doftrans);
      return &VDoFTrans;
   }
}

void FiniteElementSpace::GetFaceVDofs(int i, Array<int> &vdofs) const
{
   GetFaceDofs(i, vdofs);
   DofsToVDofs(vdofs);
}

void FiniteElementSpace::GetEdgeVDofs(int i, Array<int> &vdofs) const
{
   GetEdgeDofs(i, vdofs);
   DofsToVDofs(vdofs);
}

void FiniteElementSpace::GetVertexVDofs(int i, Array<int> &vdofs) const
{
   GetVertexDofs(i, vdofs);
   DofsToVDofs(vdofs);
}

void FiniteElementSpace::GetElementInteriorVDofs(int i, Array<int> &vdofs) const
{
   GetElementInteriorDofs(i, vdofs);
   DofsToVDofs(vdofs);
}

void FiniteElementSpace::GetEdgeInteriorVDofs(int i, Array<int> &vdofs) const
{
   GetEdgeInteriorDofs(i, vdofs);
   DofsToVDofs(vdofs);
}

void FiniteElementSpace::BuildElementToDofTable() const
{
   if (elem_dof) { return; }

   // TODO: can we call GetElementDofs only once per element?
   Table *el_dof = new Table;
   Table *el_fos = (mesh->Dimension() > 2) ? (new Table) : NULL;
   Array<int> dofs;
   Array<int> F, Fo;
   el_dof -> MakeI (mesh -> GetNE());
   if (el_fos) { el_fos -> MakeI (mesh -> GetNE()); }
   for (int i = 0; i < mesh -> GetNE(); i++)
   {
      GetElementDofs (i, dofs);
      el_dof -> AddColumnsInRow (i, dofs.Size());

      if (el_fos)
      {
         mesh->GetElementFaces(i, F, Fo);
         el_fos -> AddColumnsInRow (i, Fo.Size());
      }
   }
   el_dof -> MakeJ();
   if (el_fos) { el_fos -> MakeJ(); }
   for (int i = 0; i < mesh -> GetNE(); i++)
   {
      GetElementDofs (i, dofs);
      el_dof -> AddConnections (i, (int *)dofs, dofs.Size());

      if (el_fos)
      {
         mesh->GetElementFaces(i, F, Fo);
         el_fos -> AddConnections (i, (int *)Fo, Fo.Size());
      }
   }
   el_dof -> ShiftUpI();
   if (el_fos) { el_fos -> ShiftUpI(); }
   elem_dof = el_dof;
   elem_fos = el_fos;
}

void FiniteElementSpace::BuildBdrElementToDofTable() const
{
   if (bdr_elem_dof) { return; }

   Table *bel_dof = new Table;
   Array<int> dofs;
   bel_dof->MakeI(mesh->GetNBE());
   for (int i = 0; i < mesh->GetNBE(); i++)
   {
      GetBdrElementDofs(i, dofs);
      bel_dof->AddColumnsInRow(i, dofs.Size());
   }
   bel_dof->MakeJ();
   for (int i = 0; i < mesh->GetNBE(); i++)
   {
      GetBdrElementDofs(i, dofs);
      bel_dof->AddConnections(i, (int *)dofs, dofs.Size());
   }
   bel_dof->ShiftUpI();
   bdr_elem_dof = bel_dof;
}

void FiniteElementSpace::BuildFaceToDofTable() const
{
   // Here, "face" == (dim-1)-dimensional mesh entity.

   if (face_dof) { return; }

   if (NURBSext) { BuildNURBSFaceToDofTable(); return; }

   Table *fc_dof = new Table;
   Array<int> dofs;
   fc_dof->MakeI(mesh->GetNumFaces());
   for (int i = 0; i < fc_dof->Size(); i++)
   {
      GetFaceDofs(i, dofs, 0);
      fc_dof->AddColumnsInRow(i, dofs.Size());
   }
   fc_dof->MakeJ();
   for (int i = 0; i < fc_dof->Size(); i++)
   {
      GetFaceDofs(i, dofs, 0);
      fc_dof->AddConnections(i, (int *)dofs, dofs.Size());
   }
   fc_dof->ShiftUpI();
   face_dof = fc_dof;
}

void FiniteElementSpace::RebuildElementToDofTable()
{
   delete elem_dof;
   delete elem_fos;
   elem_dof = NULL;
   elem_fos = NULL;
   BuildElementToDofTable();
}

void FiniteElementSpace::ReorderElementToDofTable()
{
   Array<int> dof_marker(ndofs);

   dof_marker = -1;

   int *J = elem_dof->GetJ(), nnz = elem_dof->Size_of_connections();
   for (int k = 0, dof_counter = 0; k < nnz; k++)
   {
      const int sdof = J[k]; // signed dof
      const int dof = (sdof < 0) ? -1-sdof : sdof;
      int new_dof = dof_marker[dof];
      if (new_dof < 0)
      {
         dof_marker[dof] = new_dof = dof_counter++;
      }
      J[k] = (sdof < 0) ? -1-new_dof : new_dof; // preserve the sign of sdof
   }
}

void FiniteElementSpace::BuildDofToArrays()
{
   if (dof_elem_array.Size()) { return; }

   BuildElementToDofTable();

   dof_elem_array.SetSize (ndofs);
   dof_ldof_array.SetSize (ndofs);
   dof_elem_array = -1;
   for (int i = 0; i < mesh -> GetNE(); i++)
   {
      const int *dofs = elem_dof -> GetRow(i);
      const int n = elem_dof -> RowSize(i);
      for (int j = 0; j < n; j++)
      {
         if (dof_elem_array[dofs[j]] < 0)
         {
            dof_elem_array[dofs[j]] = i;
            dof_ldof_array[dofs[j]] = j;
         }
      }
   }
}

static void mark_dofs(const Array<int> &dofs, Array<int> &mark_array)
{
   for (int i = 0; i < dofs.Size(); i++)
   {
      int k = dofs[i];
      if (k < 0) { k = -1 - k; }
      mark_array[k] = -1;
   }
}

void FiniteElementSpace::GetEssentialVDofs(const Array<int> &bdr_attr_is_ess,
                                           Array<int> &ess_vdofs,
                                           int component) const
{
   Array<int> vdofs, dofs;

   ess_vdofs.SetSize(GetVSize());
   ess_vdofs = 0;

   for (int i = 0; i < GetNBE(); i++)
   {
      if (bdr_attr_is_ess[GetBdrAttribute(i)-1])
      {
         if (component < 0)
         {
            // Mark all components.
            GetBdrElementVDofs(i, vdofs);
            mark_dofs(vdofs, ess_vdofs);
         }
         else
         {
            GetBdrElementDofs(i, dofs);
            for (int d = 0; d < dofs.Size(); d++)
            { dofs[d] = DofToVDof(dofs[d], component); }
            mark_dofs(dofs, ess_vdofs);
         }
      }
   }

   // mark possible hidden boundary edges in a non-conforming mesh, also
   // local DOFs affected by boundary elements on other processors
   if (Nonconforming())
   {
      Array<int> bdr_verts, bdr_edges;
      mesh->ncmesh->GetBoundaryClosure(bdr_attr_is_ess, bdr_verts, bdr_edges);

      for (int i = 0; i < bdr_verts.Size(); i++)
      {
         if (component < 0)
         {
            GetVertexVDofs(bdr_verts[i], vdofs);
            mark_dofs(vdofs, ess_vdofs);
         }
         else
         {
            GetVertexDofs(bdr_verts[i], dofs);
            for (int d = 0; d < dofs.Size(); d++)
            { dofs[d] = DofToVDof(dofs[d], component); }
            mark_dofs(dofs, ess_vdofs);
         }
      }
      for (int i = 0; i < bdr_edges.Size(); i++)
      {
         if (component < 0)
         {
            GetEdgeVDofs(bdr_edges[i], vdofs);
            mark_dofs(vdofs, ess_vdofs);
         }
         else
         {
            GetEdgeDofs(bdr_edges[i], dofs);
            for (int d = 0; d < dofs.Size(); d++)
            { dofs[d] = DofToVDof(dofs[d], component); }
            mark_dofs(dofs, ess_vdofs);
         }
      }
   }
}

void FiniteElementSpace::GetEssentialTrueDofs(const Array<int> &bdr_attr_is_ess,
                                              Array<int> &ess_tdof_list,
                                              int component)
{
   Array<int> ess_vdofs, ess_tdofs;
   GetEssentialVDofs(bdr_attr_is_ess, ess_vdofs, component);
   const SparseMatrix *R = GetConformingRestriction();
   if (!R)
   {
      ess_tdofs.MakeRef(ess_vdofs);
   }
   else
   {
      R->BooleanMult(ess_vdofs, ess_tdofs);
   }
   MarkerToList(ess_tdofs, ess_tdof_list);
}

void FiniteElementSpace::GetBoundaryTrueDofs(Array<int> &boundary_dofs,
                                             int component)
{
   if (mesh->bdr_attributes.Size())
   {
      Array<int> ess_bdr(mesh->bdr_attributes.Max());
      ess_bdr = 1;
      GetEssentialTrueDofs(ess_bdr, boundary_dofs, component);
   }
   else
   {
      boundary_dofs.DeleteAll();
   }
}

// static method
void FiniteElementSpace::MarkerToList(const Array<int> &marker,
                                      Array<int> &list)
{
   int num_marked = 0;
   marker.HostRead(); // make sure we can read the array on host
   for (int i = 0; i < marker.Size(); i++)
   {
      if (marker[i]) { num_marked++; }
   }
   list.SetSize(0);
   list.HostWrite();
   list.Reserve(num_marked);
   for (int i = 0; i < marker.Size(); i++)
   {
      if (marker[i]) { list.Append(i); }
   }
}

// static method
void FiniteElementSpace::ListToMarker(const Array<int> &list, int marker_size,
                                      Array<int> &marker, int mark_val)
{
   list.HostRead(); // make sure we can read the array on host
   marker.SetSize(marker_size);
   marker.HostWrite();
   marker = 0;
   for (int i = 0; i < list.Size(); i++)
   {
      marker[list[i]] = mark_val;
   }
}

void FiniteElementSpace::ConvertToConformingVDofs(const Array<int> &dofs,
                                                  Array<int> &cdofs)
{
   GetConformingProlongation();
   if (cP) { cP->BooleanMultTranspose(dofs, cdofs); }
   else { dofs.Copy(cdofs); }
}

void FiniteElementSpace::ConvertFromConformingVDofs(const Array<int> &cdofs,
                                                    Array<int> &dofs)
{
   GetConformingRestriction();
   if (cR) { cR->BooleanMultTranspose(cdofs, dofs); }
   else { cdofs.Copy(dofs); }
}

SparseMatrix *
FiniteElementSpace::D2C_GlobalRestrictionMatrix (FiniteElementSpace *cfes)
{
   int i, j;
   Array<int> d_vdofs, c_vdofs;
   SparseMatrix *R;

   R = new SparseMatrix (cfes -> GetVSize(), GetVSize());

   for (i = 0; i < mesh -> GetNE(); i++)
   {
      this -> GetElementVDofs (i, d_vdofs);
      cfes -> GetElementVDofs (i, c_vdofs);

#ifdef MFEM_DEBUG
      if (d_vdofs.Size() != c_vdofs.Size())
      {
         mfem_error ("FiniteElementSpace::D2C_GlobalRestrictionMatrix (...)");
      }
#endif

      for (j = 0; j < d_vdofs.Size(); j++)
      {
         R -> Set (c_vdofs[j], d_vdofs[j], 1.0);
      }
   }

   R -> Finalize();

   return R;
}

SparseMatrix *
FiniteElementSpace::D2Const_GlobalRestrictionMatrix(FiniteElementSpace *cfes)
{
   int i, j;
   Array<int> d_dofs, c_dofs;
   SparseMatrix *R;

   R = new SparseMatrix (cfes -> GetNDofs(), ndofs);

   for (i = 0; i < mesh -> GetNE(); i++)
   {
      this -> GetElementDofs (i, d_dofs);
      cfes -> GetElementDofs (i, c_dofs);

#ifdef MFEM_DEBUG
      if (c_dofs.Size() != 1)
         mfem_error ("FiniteElementSpace::"
                     "D2Const_GlobalRestrictionMatrix (...)");
#endif

      for (j = 0; j < d_dofs.Size(); j++)
      {
         R -> Set (c_dofs[0], d_dofs[j], 1.0);
      }
   }

   R -> Finalize();

   return R;
}

SparseMatrix *
FiniteElementSpace::H2L_GlobalRestrictionMatrix (FiniteElementSpace *lfes)
{
   SparseMatrix *R;
   DenseMatrix loc_restr;
   Array<int> l_dofs, h_dofs, l_vdofs, h_vdofs;

   int vdim = lfes->GetVDim();
   R = new SparseMatrix (vdim * lfes -> GetNDofs(), vdim * ndofs);

   Geometry::Type cached_geom = Geometry::INVALID;
   const FiniteElement *h_fe = NULL;
   const FiniteElement *l_fe = NULL;
   IsoparametricTransformation T;

   for (int i = 0; i < mesh -> GetNE(); i++)
   {
      this -> GetElementDofs (i, h_dofs);
      lfes -> GetElementDofs (i, l_dofs);

      // Assuming 'loc_restr' depends only on the Geometry::Type.
      const Geometry::Type geom = mesh->GetElementBaseGeometry(i);
      if (geom != cached_geom)
      {
         h_fe = this -> GetFE (i);
         l_fe = lfes -> GetFE (i);
         T.SetIdentityTransformation(h_fe->GetGeomType());
         h_fe->Project(*l_fe, T, loc_restr);
         cached_geom = geom;
      }

      for (int vd = 0; vd < vdim; vd++)
      {
         l_dofs.Copy(l_vdofs);
         lfes->DofsToVDofs(vd, l_vdofs);

         h_dofs.Copy(h_vdofs);
         this->DofsToVDofs(vd, h_vdofs);

         R -> SetSubMatrix (l_vdofs, h_vdofs, loc_restr, 1);
      }
   }

   R -> Finalize();

   return R;
}

void FiniteElementSpace
::AddDependencies(SparseMatrix& deps, Array<int>& master_dofs,
                  Array<int>& slave_dofs, DenseMatrix& I, int skipfirst)
{
   for (int i = skipfirst; i < slave_dofs.Size(); i++)
   {
      int sdof = slave_dofs[i];
      if (!deps.RowSize(sdof)) // not processed yet
      {
         for (int j = 0; j < master_dofs.Size(); j++)
         {
            double coef = I(i, j);
            if (std::abs(coef) > 1e-12)
            {
               int mdof = master_dofs[j];
               if (mdof != sdof && mdof != (-1-sdof))
               {
                  deps.Add(sdof, mdof, coef);
               }
            }
         }
      }
   }
}

void FiniteElementSpace
::AddEdgeFaceDependencies(SparseMatrix &deps, Array<int> &master_dofs,
                          const FiniteElement *master_fe,
                          Array<int> &slave_dofs, int slave_face,
                          const DenseMatrix *pm) const
{
   // In variable-order spaces in 3D, we need to only constrain interior face
   // DOFs (this is done one level up), since edge dependencies can be more
   // complex and are primarily handled by edge-edge dependencies. The one
   // exception is edges of slave faces that lie in the interior of the master
   // face, which are not covered by edge-edge relations. This function finds
   // such edges and makes them constrained by the master face.
   // See also https://github.com/mfem/mfem/pull/1423#issuecomment-633916643

   Array<int> V, E, Eo; // TODO: LocalArray
   mesh->GetFaceVertices(slave_face, V);
   mesh->GetFaceEdges(slave_face, E, Eo);
   MFEM_ASSERT(V.Size() == E.Size(), "");

   DenseMatrix I;
   IsoparametricTransformation edge_T;
   edge_T.SetFE(&SegmentFE);

   // constrain each edge of the slave face
   for (int i = 0; i < E.Size(); i++)
   {
      int a = i, b = (i+1) % V.Size();
      if (V[a] > V[b]) { std::swap(a, b); }

      DenseMatrix &edge_pm = edge_T.GetPointMat();
      edge_pm.SetSize(2, 2);

      // copy two points from the face point matrix
      double mid[2];
      for (int j = 0; j < 2; j++)
      {
         edge_pm(j, 0) = (*pm)(j, a);
         edge_pm(j, 1) = (*pm)(j, b);
         mid[j] = 0.5*((*pm)(j, a) + (*pm)(j, b));
      }

      // check that the edge does not coincide with the master face's edge
      const double eps = 1e-14;
      if (mid[0] > eps && mid[0] < 1-eps &&
          mid[1] > eps && mid[1] < 1-eps)
      {
         int order = GetEdgeDofs(E[i], slave_dofs, 0);

         const auto *edge_fe = fec->GetFE(Geometry::SEGMENT, order);
         edge_fe->GetTransferMatrix(*master_fe, edge_T, I);

         AddDependencies(deps, master_dofs, slave_dofs, I, 0);
      }
   }
}

bool FiniteElementSpace::DofFinalizable(int dof, const Array<bool>& finalized,
                                        const SparseMatrix& deps)
{
   const int* dep = deps.GetRowColumns(dof);
   int ndep = deps.RowSize(dof);

   // are all constraining DOFs finalized?
   for (int i = 0; i < ndep; i++)
   {
      if (!finalized[dep[i]]) { return false; }
   }
   return true;
}

int FiniteElementSpace::GetDegenerateFaceDofs(int index, Array<int> &dofs,
                                              Geometry::Type master_geom,
                                              int variant) const
{
   // In NC meshes with prisms/tets, a special constraint occurs where a
   // prism/tet edge is slave to another element's face (see illustration
   // here: https://github.com/mfem/mfem/pull/713#issuecomment-495786362)
   // Rather than introduce a new edge-face constraint type, we handle such
   // cases as degenerate face-face constraints, where the point-matrix
   // rectangle has zero height. This method returns DOFs for the first edge
   // of the rectangle, duplicated in the orthogonal direction, to resemble
   // DOFs for a quadrilateral face. The extra DOFs are ignored by
   // FiniteElementSpace::AddDependencies.

   Array<int> edof;
   int order = GetEdgeDofs(-1 - index, edof, variant);

   int nv = fec->DofForGeometry(Geometry::POINT);
   int ne = fec->DofForGeometry(Geometry::SEGMENT);
   int nn = 2*nv + ne;

   dofs.SetSize(nn*nn);
   if (!dofs.Size()) { return 0; }

   dofs = edof[0];

   // copy first two vertex DOFs
   for (int i = 0; i < nv; i++)
   {
      dofs[i] = edof[i];
      dofs[nv+i] = edof[nv+i];
   }
   // copy first edge DOFs
   int face_vert = Geometry::NumVerts[master_geom];
   for (int i = 0; i < ne; i++)
   {
      dofs[face_vert*nv + i] = edof[2*nv + i];
   }

   return order;
}

int FiniteElementSpace::GetNumBorderDofs(Geometry::Type geom, int order) const
{
   // return the number of vertex and edge DOFs that precede inner DOFs
   int nv = fec->GetNumDof(Geometry::POINT, order);
   int ne = fec->GetNumDof(Geometry::SEGMENT, order);
   return Geometry::NumVerts[geom] * (nv + ne);
}

int FiniteElementSpace::GetEntityDofs(int entity, int index, Array<int> &dofs,
                                      Geometry::Type master_geom,
                                      int variant) const
{
   switch (entity)
   {
      case 0:
         GetVertexDofs(index, dofs);
         return 0;

      case 1:
         return GetEdgeDofs(index, dofs, variant);

      default:
         if (index >= 0)
         {
            return GetFaceDofs(index, dofs, variant);
         }
         else
         {
            return GetDegenerateFaceDofs(index, dofs, master_geom, variant);
         }
   }
}

void FiniteElementSpace::BuildConformingInterpolation() const
{
#ifdef MFEM_USE_MPI
   MFEM_VERIFY(dynamic_cast<const ParFiniteElementSpace*>(this) == NULL,
               "This method should not be used with a ParFiniteElementSpace!");
#endif

   if (cP_is_set) { return; }
   cP_is_set = true;

   Array<int> master_dofs, slave_dofs, highest_dofs;

   IsoparametricTransformation T;
   DenseMatrix I;

   // For each slave DOF, the dependency matrix will contain a row that
   // expresses the slave DOF as a linear combination of its immediate master
   // DOFs. Rows of independent DOFs will remain empty.
   SparseMatrix deps(ndofs);

   // Inverse dependencies for the cR_hp matrix in variable order spaces:
   // For each master edge/face with more DOF sets, the inverse dependency
   // matrix contains a row that expresses the master true DOF (lowest order)
   // as a linear combination of the highest order set of DOFs.
   SparseMatrix inv_deps(ndofs);

   // collect local face/edge dependencies
   for (int entity = 2; entity >= 1; entity--)
   {
      const NCMesh::NCList &list = mesh->ncmesh->GetNCList(entity);
      if (!list.masters.Size()) { continue; }

      // loop through all master edges/faces, constrain their slave edges/faces
      for (const NCMesh::Master &master : list.masters)
      {
         Geometry::Type master_geom = master.Geom();

         int p = GetEntityDofs(entity, master.index, master_dofs, master_geom);
         if (!master_dofs.Size()) { continue; }

         const FiniteElement *master_fe = fec->GetFE(master_geom, p);
         if (!master_fe) { continue; }

         switch (master_geom)
         {
            case Geometry::SQUARE:   T.SetFE(&QuadrilateralFE); break;
            case Geometry::TRIANGLE: T.SetFE(&TriangleFE); break;
            case Geometry::SEGMENT:  T.SetFE(&SegmentFE); break;
            default: MFEM_ABORT("unsupported geometry");
         }

         for (int si = master.slaves_begin; si < master.slaves_end; si++)
         {
            const NCMesh::Slave &slave = list.slaves[si];

            int q = GetEntityDofs(entity, slave.index, slave_dofs, master_geom);
            if (!slave_dofs.Size()) { break; }

            const FiniteElement *slave_fe = fec->GetFE(slave.Geom(), q);
            list.OrientedPointMatrix(slave, T.GetPointMat());
            slave_fe->GetTransferMatrix(*master_fe, T, I);

            // variable order spaces: face edges need to be handled separately
            int skipfirst = 0;
            if (IsVariableOrder() && entity == 2 && slave.index >= 0)
            {
               skipfirst = GetNumBorderDofs(master_geom, q);
            }

            // make each slave DOF dependent on all master DOFs
            AddDependencies(deps, master_dofs, slave_dofs, I, skipfirst);

            if (skipfirst)
            {
               // constrain internal edge DOFs if they were skipped
               const auto *pm = list.point_matrices[master_geom][slave.matrix];
               AddEdgeFaceDependencies(deps, master_dofs, master_fe,
                                       slave_dofs, slave.index, pm);
            }
         }

         // Add inverse dependencies for the cR_hp matrix; if a master has
         // more DOF sets, the lowest order set interpolates the highest one.
         if (IsVariableOrder())
         {
            int nvar = GetNVariants(entity, master.index);
            if (nvar > 1)
            {
               int q = GetEntityDofs(entity, master.index, highest_dofs,
                                     master_geom, nvar-1);
               const auto *highest_fe = fec->GetFE(master_geom, q);

               T.SetIdentityTransformation(master_geom);
               master_fe->GetTransferMatrix(*highest_fe, T, I);

               // add dependencies only for the inner dofs
               int skip = GetNumBorderDofs(master_geom, p);
               AddDependencies(inv_deps, highest_dofs, master_dofs, I, skip);
            }
         }
      }
   }

   // variable order spaces: enforce minimum rule on conforming edges/faces
   if (IsVariableOrder())
   {
      for (int entity = 1; entity < mesh->Dimension(); entity++)
      {
         const Table &ent_dofs = (entity == 1) ? var_edge_dofs : var_face_dofs;
         int num_ent = (entity == 1) ? mesh->GetNEdges() : mesh->GetNFaces();
         MFEM_ASSERT(ent_dofs.Size() == num_ent+1, "");

         // add constraints within edges/faces holding multiple DOF sets
         Geometry::Type last_geom = Geometry::INVALID;
         for (int i = 0; i < num_ent; i++)
         {
            if (ent_dofs.RowSize(i) <= 1) { continue; }

            Geometry::Type geom =
               (entity == 1) ? Geometry::SEGMENT : mesh->GetFaceGeometry(i);

            if (geom != last_geom)
            {
               T.SetIdentityTransformation(geom);
               last_geom = geom;
            }

            // get lowest order variant DOFs and FE
            int p = GetEntityDofs(entity, i, master_dofs, geom, 0);
            const auto *master_fe = fec->GetFE(geom, p);

            // constrain all higher order DOFs: interpolate lowest order function
            for (int variant = 1; ; variant++)
            {
               int q = GetEntityDofs(entity, i, slave_dofs, geom, variant);
               if (q < 0) { break; }

               const auto *slave_fe = fec->GetFE(geom, q);
               slave_fe->GetTransferMatrix(*master_fe, T, I);

               AddDependencies(deps, master_dofs, slave_dofs, I);
            }
         }
      }
   }

   deps.Finalize();
   inv_deps.Finalize();

   // DOFs that stayed independent are true DOFs
   int n_true_dofs = 0;
   for (int i = 0; i < ndofs; i++)
   {
      if (!deps.RowSize(i)) { n_true_dofs++; }
   }

   // if all dofs are true dofs leave cP and cR NULL
   if (n_true_dofs == ndofs)
   {
      cP = cR = cR_hp = NULL; // will be treated as identities
      return;
   }

   // create the conforming prolongation matrix cP
   cP = new SparseMatrix(ndofs, n_true_dofs);

   // create the conforming restriction matrix cR
   int *cR_J;
   {
      int *cR_I = Memory<int>(n_true_dofs+1);
      double *cR_A = Memory<double>(n_true_dofs);
      cR_J = Memory<int>(n_true_dofs);
      for (int i = 0; i < n_true_dofs; i++)
      {
         cR_I[i] = i;
         cR_A[i] = 1.0;
      }
      cR_I[n_true_dofs] = n_true_dofs;
      cR = new SparseMatrix(cR_I, cR_J, cR_A, n_true_dofs, ndofs);
   }

   // In var. order spaces, create the restriction matrix cR_hp which is similar
   // to cR, but has interpolation in the extra master edge/face DOFs.
   cR_hp = IsVariableOrder() ? new SparseMatrix(n_true_dofs, ndofs) : NULL;

   Array<bool> finalized(ndofs);
   finalized = false;

   Array<int> cols;
   Vector srow;

   // put identity in the prolongation matrix for true DOFs, initialize cR_hp
   for (int i = 0, true_dof = 0; i < ndofs; i++)
   {
      if (!deps.RowSize(i)) // true dof
      {
         cP->Add(i, true_dof, 1.0);
         cR_J[true_dof] = i;
         finalized[i] = true;

         if (cR_hp)
         {
            if (inv_deps.RowSize(i))
            {
               inv_deps.GetRow(i, cols, srow);
               cR_hp->AddRow(true_dof, cols, srow);
            }
            else
            {
               cR_hp->Add(true_dof, i, 1.0);
            }
         }

         true_dof++;
      }
   }

   // Now calculate cP rows of slave DOFs as combinations of cP rows of their
   // master DOFs. It is possible that some slave DOFs depend on DOFs that are
   // themselves slaves. Here we resolve such indirect constraints by first
   // calculating rows of the cP matrix for DOFs whose master DOF cP rows are
   // already known (in the first iteration these are the true DOFs). In the
   // second iteration, slaves of slaves can be 'finalized' (given a row in the
   // cP matrix), in the third iteration slaves of slaves of slaves, etc.
   bool finished;
   int n_finalized = n_true_dofs;
   do
   {
      finished = true;
      for (int dof = 0; dof < ndofs; dof++)
      {
         if (!finalized[dof] && DofFinalizable(dof, finalized, deps))
         {
            const int* dep_col = deps.GetRowColumns(dof);
            const double* dep_coef = deps.GetRowEntries(dof);
            int n_dep = deps.RowSize(dof);

            for (int j = 0; j < n_dep; j++)
            {
               cP->GetRow(dep_col[j], cols, srow);
               srow *= dep_coef[j];
               cP->AddRow(dof, cols, srow);
            }

            finalized[dof] = true;
            n_finalized++;
            finished = false;
         }
      }
   }
   while (!finished);

   // If everything is consistent (mesh, face orientations, etc.), we should
   // be able to finalize all slave DOFs, otherwise it's a serious error.
   MFEM_VERIFY(n_finalized == ndofs,
               "Error creating cP matrix: n_finalized = "
               << n_finalized << ", ndofs = " << ndofs);

   cP->Finalize();
   if (cR_hp) { cR_hp->Finalize(); }

   if (vdim > 1)
   {
      MakeVDimMatrix(*cP);
      MakeVDimMatrix(*cR);
      if (cR_hp) { MakeVDimMatrix(*cR_hp); }
   }

   if (Device::IsEnabled()) { cP->BuildTranspose(); }
}

void FiniteElementSpace::MakeVDimMatrix(SparseMatrix &mat) const
{
   if (vdim == 1) { return; }

   int height = mat.Height();
   int width = mat.Width();

   SparseMatrix *vmat = new SparseMatrix(vdim*height, vdim*width);

   Array<int> dofs, vdofs;
   Vector srow;
   for (int i = 0; i < height; i++)
   {
      mat.GetRow(i, dofs, srow);
      for (int vd = 0; vd < vdim; vd++)
      {
         dofs.Copy(vdofs);
         DofsToVDofs(vd, vdofs, width);
         vmat->SetRow(DofToVDof(i, vd, height), vdofs, srow);
      }
   }
   vmat->Finalize();

   mat.Swap(*vmat);
   delete vmat;
}


const SparseMatrix* FiniteElementSpace::GetConformingProlongation() const
{
   if (Conforming()) { return NULL; }
   if (!cP_is_set) { BuildConformingInterpolation(); }
   return cP;
}

const SparseMatrix* FiniteElementSpace::GetConformingRestriction() const
{
   if (Conforming()) { return NULL; }
   if (!cP_is_set) { BuildConformingInterpolation(); }
   return cR;
}

const SparseMatrix* FiniteElementSpace::GetHpConformingRestriction() const
{
   if (Conforming()) { return NULL; }
   if (!cP_is_set) { BuildConformingInterpolation(); }
   return IsVariableOrder() ? cR_hp : cR;
}

int FiniteElementSpace::GetNConformingDofs() const
{
   const SparseMatrix* P = GetConformingProlongation();
   return P ? (P->Width() / vdim) : ndofs;
}

const Operator *FiniteElementSpace::GetElementRestriction(
   ElementDofOrdering e_ordering) const
{
   // Check if we have a discontinuous space using the FE collection:
   if (IsDGSpace())
   {
      if (L2E_nat.Ptr() == NULL)
      {
         L2E_nat.Reset(new L2ElementRestriction(*this));
      }
      return L2E_nat.Ptr();
   }
   if (e_ordering == ElementDofOrdering::LEXICOGRAPHIC)
   {
      if (L2E_lex.Ptr() == NULL)
      {
         L2E_lex.Reset(new ElementRestriction(*this, e_ordering));
      }
      return L2E_lex.Ptr();
   }
   // e_ordering == ElementDofOrdering::NATIVE
   if (L2E_nat.Ptr() == NULL)
   {
      L2E_nat.Reset(new ElementRestriction(*this, e_ordering));
   }
   return L2E_nat.Ptr();
}

const Operator *FiniteElementSpace::GetFaceRestriction(
   ElementDofOrdering e_ordering, FaceType type, L2FaceValues mul) const
{
   const bool is_dg_space = IsDGSpace();
   const L2FaceValues m = (is_dg_space && mul==L2FaceValues::DoubleValued) ?
                          L2FaceValues::DoubleValued : L2FaceValues::SingleValued;
   key_face key = std::make_tuple(is_dg_space, e_ordering, type, m);
   auto itr = L2F.find(key);
   if (itr != L2F.end())
   {
      return itr->second;
   }
   else
   {
      Operator* res;
      if (is_dg_space)
      {
         res = new L2FaceRestriction(*this, e_ordering, type, m);
      }
      else
      {
         res = new H1FaceRestriction(*this, e_ordering, type);
      }
      L2F[key] = res;
      return res;
   }
}

const QuadratureInterpolator *FiniteElementSpace::GetQuadratureInterpolator(
   const IntegrationRule &ir) const
{
   for (int i = 0; i < E2Q_array.Size(); i++)
   {
      const QuadratureInterpolator *qi = E2Q_array[i];
      if (qi->IntRule == &ir) { return qi; }
   }

   QuadratureInterpolator *qi = new QuadratureInterpolator(*this, ir);
   E2Q_array.Append(qi);
   return qi;
}

const QuadratureInterpolator *FiniteElementSpace::GetQuadratureInterpolator(
   const QuadratureSpace &qs) const
{
   for (int i = 0; i < E2Q_array.Size(); i++)
   {
      const QuadratureInterpolator *qi = E2Q_array[i];
      if (qi->qspace == &qs) { return qi; }
   }

   QuadratureInterpolator *qi = new QuadratureInterpolator(*this, qs);
   E2Q_array.Append(qi);
   return qi;
}

const FaceQuadratureInterpolator
*FiniteElementSpace::GetFaceQuadratureInterpolator(
   const IntegrationRule &ir, FaceType type) const
{
   if (type==FaceType::Interior)
   {
      for (int i = 0; i < E2IFQ_array.Size(); i++)
      {
         const FaceQuadratureInterpolator *qi = E2IFQ_array[i];
         if (qi->IntRule == &ir) { return qi; }
      }

      FaceQuadratureInterpolator *qi = new FaceQuadratureInterpolator(*this, ir,
                                                                      type);
      E2IFQ_array.Append(qi);
      return qi;
   }
   else //Boundary
   {
      for (int i = 0; i < E2BFQ_array.Size(); i++)
      {
         const FaceQuadratureInterpolator *qi = E2BFQ_array[i];
         if (qi->IntRule == &ir) { return qi; }
      }

      FaceQuadratureInterpolator *qi = new FaceQuadratureInterpolator(*this, ir,
                                                                      type);
      E2BFQ_array.Append(qi);
      return qi;
   }
}

SparseMatrix *FiniteElementSpace::RefinementMatrix_main(
   const int coarse_ndofs, const Table &coarse_elem_dof,
   const Table *coarse_elem_fos, const DenseTensor localP[]) const
{
   /// TODO: Implement DofTransformation support

   MFEM_VERIFY(mesh->GetLastOperation() == Mesh::REFINE, "");

   Array<int> dofs, coarse_dofs, coarse_vdofs;
   Vector row;

   Mesh::GeometryList elem_geoms(*mesh);

   SparseMatrix *P;
   if (elem_geoms.Size() == 1)
   {
      const int coarse_ldof = localP[elem_geoms[0]].SizeJ();
      P = new SparseMatrix(GetVSize(), coarse_ndofs*vdim, coarse_ldof);
   }
   else
   {
      P = new SparseMatrix(GetVSize(), coarse_ndofs*vdim);
   }

   Array<int> mark(P->Height());
   mark = 0;

   const CoarseFineTransformations &rtrans = mesh->GetRefinementTransforms();

   for (int k = 0; k < mesh->GetNE(); k++)
   {
      const Embedding &emb = rtrans.embeddings[k];
      const Geometry::Type geom = mesh->GetElementBaseGeometry(k);
      const DenseMatrix &lP = localP[geom](emb.matrix);
      const int fine_ldof = localP[geom].SizeI();

      elem_dof->GetRow(k, dofs);
      coarse_elem_dof.GetRow(emb.parent, coarse_dofs);

      for (int vd = 0; vd < vdim; vd++)
      {
         coarse_dofs.Copy(coarse_vdofs);
         DofsToVDofs(vd, coarse_vdofs, coarse_ndofs);

         for (int i = 0; i < fine_ldof; i++)
         {
            int r = DofToVDof(dofs[i], vd);
            int m = (r >= 0) ? r : (-1 - r);

            if (!mark[m])
            {
               lP.GetRow(i, row);
               P->SetRow(r, coarse_vdofs, row);
               mark[m] = 1;
            }
         }
      }
   }

   MFEM_ASSERT(mark.Sum() == P->Height(), "Not all rows of P set.");
   if (elem_geoms.Size() != 1) { P->Finalize(); }
   return P;
}

void FiniteElementSpace::GetLocalRefinementMatrices(
   Geometry::Type geom, DenseTensor &localP) const
{
   const FiniteElement *fe = fec->FiniteElementForGeometry(geom);

   const CoarseFineTransformations &rtrans = mesh->GetRefinementTransforms();
   const DenseTensor &pmats = rtrans.point_matrices[geom];

   int nmat = pmats.SizeK();
   int ldof = fe->GetDof();

   IsoparametricTransformation isotr;
   isotr.SetIdentityTransformation(geom);

   // calculate local interpolation matrices for all refinement types
   localP.SetSize(ldof, ldof, nmat);
   for (int i = 0; i < nmat; i++)
   {
      isotr.SetPointMat(pmats(i));
      fe->GetLocalInterpolation(isotr, localP(i));
   }
}

SparseMatrix* FiniteElementSpace::RefinementMatrix(int old_ndofs,
                                                   const Table* old_elem_dof,
                                                   const Table* old_elem_fos)
{
   MFEM_VERIFY(GetNE() >= old_elem_dof->Size(),
               "Previous mesh is not coarser.");

   Mesh::GeometryList elem_geoms(*mesh);

   DenseTensor localP[Geometry::NumGeom];
   for (int i = 0; i < elem_geoms.Size(); i++)
   {
      GetLocalRefinementMatrices(elem_geoms[i], localP[elem_geoms[i]]);
   }

   return RefinementMatrix_main(old_ndofs, *old_elem_dof, old_elem_fos,
                                localP);
}

FiniteElementSpace::RefinementOperator::RefinementOperator
(const FiniteElementSpace* fespace, Table* old_elem_dof, Table* old_elem_fos,
 int old_ndofs)
   : fespace(fespace)
   , old_elem_dof(old_elem_dof)
   , old_elem_fos(old_elem_fos)
{
   MFEM_VERIFY(fespace->GetNE() >= old_elem_dof->Size(),
               "Previous mesh is not coarser.");

   width = old_ndofs * fespace->GetVDim();
   height = fespace->GetVSize();

   Mesh::GeometryList elem_geoms(*fespace->GetMesh());

   for (int i = 0; i < elem_geoms.Size(); i++)
   {
      fespace->GetLocalRefinementMatrices(elem_geoms[i], localP[elem_geoms[i]]);
   }

   ConstructDoFTrans();
}

FiniteElementSpace::RefinementOperator::RefinementOperator(
   const FiniteElementSpace *fespace, const FiniteElementSpace *coarse_fes)
   : Operator(fespace->GetVSize(), coarse_fes->GetVSize()),
     fespace(fespace), old_elem_dof(NULL), old_elem_fos(NULL)
{
   Mesh::GeometryList elem_geoms(*fespace->GetMesh());

   for (int i = 0; i < elem_geoms.Size(); i++)
   {
      fespace->GetLocalRefinementMatrices(*coarse_fes, elem_geoms[i],
                                          localP[elem_geoms[i]]);
   }

   // Make a copy of the coarse elem_dof Table.
   old_elem_dof = new Table(coarse_fes->GetElementToDofTable());

   // Make a copy of the coarse elem_fos Table if it exists.
   if (coarse_fes->GetElementToFaceOrientationTable())
   {
      old_elem_fos = new Table(*coarse_fes->GetElementToFaceOrientationTable());
   }

   ConstructDoFTrans();
}

FiniteElementSpace::RefinementOperator::~RefinementOperator()
{
   delete old_elem_dof;
   delete old_elem_fos;
}

void FiniteElementSpace::RefinementOperator
::ConstructDoFTrans()
{
   old_DoFTrans.SetSize(Geometry::NUM_GEOMETRIES);
   for (int i=0; i<old_DoFTrans.Size(); i++)
   {
      old_DoFTrans[i] = NULL;
   }

   const FiniteElementCollection *fec = fespace->FEColl();
   if (dynamic_cast<const ND_FECollection*>(fec))
   {
      const FiniteElement * nd_tri =
         fec->FiniteElementForGeometry(Geometry::TRIANGLE);
      if (nd_tri)
      {
         old_DoFTrans[Geometry::TRIANGLE] =
            new ND_TriDofTransformation(nd_tri->GetOrder());
      }

      const FiniteElement * nd_tet =
         fec->FiniteElementForGeometry(Geometry::TETRAHEDRON);
      if (nd_tet)
      {
         old_DoFTrans[Geometry::TETRAHEDRON] =
            new ND_TetDofTransformation(nd_tet->GetOrder());
      }
   }
}

void FiniteElementSpace::RefinementOperator
::Mult(const Vector &x, Vector &y) const
{
   Mesh* mesh = fespace->GetMesh();
   const CoarseFineTransformations &rtrans = mesh->GetRefinementTransforms();

   Array<int> dofs, vdofs, old_dofs, old_vdofs, old_Fo;

   int vdim = fespace->GetVDim();
   int old_ndofs = width / vdim;

   Vector subY, subX, subYt, subXt;

   //old_VDoFTrans.SetVDim(vdim);

   for (int k = 0; k < mesh->GetNE(); k++)
   {
      const Embedding &emb = rtrans.embeddings[k];
      const Geometry::Type geom = mesh->GetElementBaseGeometry(k);
      const DenseMatrix &lP = localP[geom](emb.matrix);

      DofTransformation *doftrans = fespace->GetElementDofs(k, dofs);
      old_elem_dof->GetRow(emb.parent, old_dofs);

      if (!doftrans)
      {
         subY.SetSize(lP.Height());

         for (int vd = 0; vd < vdim; vd++)
         {
            dofs.Copy(vdofs);
            fespace->DofsToVDofs(vd, vdofs);
            old_dofs.Copy(old_vdofs);
            fespace->DofsToVDofs(vd, old_vdofs, old_ndofs);
            x.GetSubVector(old_vdofs, subX);
            lP.Mult(subX, subY);
            y.SetSubVector(vdofs, subY);
         }
      }
      else
      {
         subYt.SetSize(lP.Height());

         old_elem_fos->GetRow(emb.parent, old_Fo);
         old_DoFTrans[geom]->SetFaceOrientations(old_Fo);
         // old_VDoFTrans.SetDofTransformation(*old_DoFTrans[geom]);

         DofTransformation *new_doftrans = NULL;
         VDofTransformation *vdoftrans =
            dynamic_cast<VDofTransformation*>(doftrans);
         if (vdoftrans)
         {
            new_doftrans = doftrans;
            doftrans = vdoftrans->GetDofTransformation();
         }

         for (int vd = 0; vd < vdim; vd++)
         {
            dofs.Copy(vdofs);
            fespace->DofsToVDofs(vd, vdofs);
            old_dofs.Copy(old_vdofs);
            fespace->DofsToVDofs(vd, old_vdofs, old_ndofs);
            x.GetSubVector(old_vdofs, subX);
            // old_VDoFTrans.InvTransformPrimal(subX, subXt);
            old_DoFTrans[geom]->InvTransformPrimal(subX, subXt);
            lP.Mult(subXt, subYt);
            doftrans->TransformPrimal(subYt, subY);
            y.SetSubVector(vdofs, subY);
         }

         if (vdoftrans)
         {
            doftrans = new_doftrans;
         }
      }
   }
}

void FiniteElementSpace::RefinementOperator
::MultTranspose(const Vector &x, Vector &y) const
{
   y = 0.0;

   Mesh* mesh = fespace->GetMesh();
   const CoarseFineTransformations &rtrans = mesh->GetRefinementTransforms();

   Array<char> processed(fespace->GetVSize());
   processed = 0;

   Array<int> f_dofs, c_dofs, f_vdofs, c_vdofs, old_Fo;

   int vdim = fespace->GetVDim();
   int old_ndofs = width / vdim;

   Vector subY, subX, subYt, subXt;

   for (int k = 0; k < mesh->GetNE(); k++)
   {
      const Embedding &emb = rtrans.embeddings[k];
      const Geometry::Type geom = mesh->GetElementBaseGeometry(k);
      const DenseMatrix &lP = localP[geom](emb.matrix);

      DofTransformation * doftrans = fespace->GetElementDofs(k, f_dofs);
      old_elem_dof->GetRow(emb.parent, c_dofs);

      if (!doftrans)
      {
         subY.SetSize(lP.Width());

         for (int vd = 0; vd < vdim; vd++)
         {
            f_dofs.Copy(f_vdofs);
            fespace->DofsToVDofs(vd, f_vdofs);
            c_dofs.Copy(c_vdofs);
            fespace->DofsToVDofs(vd, c_vdofs, old_ndofs);

            x.GetSubVector(f_vdofs, subX);

            for (int p = 0; p < f_dofs.Size(); ++p)
            {
               if (processed[DecodeDof(f_dofs[p])])
               {
                  subX[p] = 0.0;
               }
            }

            lP.MultTranspose(subX, subY);
            y.AddElementVector(c_vdofs, subY);
         }
      }
      else
      {
         subYt.SetSize(lP.Width());

         old_elem_fos->GetRow(emb.parent, old_Fo);
         old_DoFTrans[geom]->SetFaceOrientations(old_Fo);

         DofTransformation *new_doftrans = NULL;
         VDofTransformation *vdoftrans =
            dynamic_cast<VDofTransformation*>(doftrans);
         if (vdoftrans)
         {
            new_doftrans = doftrans;
            doftrans = vdoftrans->GetDofTransformation();
         }

         for (int vd = 0; vd < vdim; vd++)
         {
            f_dofs.Copy(f_vdofs);
            fespace->DofsToVDofs(vd, f_vdofs);
            c_dofs.Copy(c_vdofs);
            fespace->DofsToVDofs(vd, c_vdofs, old_ndofs);

            x.GetSubVector(f_vdofs, subX);
            old_DoFTrans[geom]->InvTransformPrimal(subX, subXt);

            for (int p = 0; p < f_dofs.Size(); ++p)
            {
               if (processed[DecodeDof(f_dofs[p])])
               {
                  subXt[p] = 0.0;
               }
            }

            lP.MultTranspose(subXt, subYt);
            doftrans->TransformPrimal(subYt, subY);
            y.AddElementVector(c_vdofs, subY);
         }

         if (vdoftrans)
         {
            doftrans = new_doftrans;
         }
      }

      for (int p = 0; p < f_dofs.Size(); ++p)
      {
         processed[DecodeDof(f_dofs[p])] = 1;
      }
   }
}

/// TODO: Implement DofTransformation support
FiniteElementSpace::DerefinementOperator::DerefinementOperator(
   const FiniteElementSpace *f_fes, const FiniteElementSpace *c_fes,
   BilinearFormIntegrator *mass_integ)
   : Operator(c_fes->GetVSize(), f_fes->GetVSize()),
     fine_fes(f_fes)
{
   MFEM_VERIFY(c_fes->GetOrdering() == f_fes->GetOrdering() &&
               c_fes->GetVDim() == f_fes->GetVDim(),
               "incompatible coarse and fine FE spaces");

   IsoparametricTransformation emb_tr;
   Mesh *f_mesh = f_fes->GetMesh();
   const CoarseFineTransformations &rtrans = f_mesh->GetRefinementTransforms();

   Mesh::GeometryList elem_geoms(*f_mesh);
   DenseTensor localP[Geometry::NumGeom], localM[Geometry::NumGeom];
   for (int gi = 0; gi < elem_geoms.Size(); gi++)
   {
      const Geometry::Type geom = elem_geoms[gi];
      DenseTensor &lP = localP[geom], &lM = localM[geom];
      const FiniteElement *fine_fe =
         f_fes->fec->FiniteElementForGeometry(geom);
      const FiniteElement *coarse_fe =
         c_fes->fec->FiniteElementForGeometry(geom);
      const DenseTensor &pmats = rtrans.point_matrices[geom];

      lP.SetSize(fine_fe->GetDof(), coarse_fe->GetDof(), pmats.SizeK());
      lM.SetSize(fine_fe->GetDof(),   fine_fe->GetDof(), pmats.SizeK());
      emb_tr.SetIdentityTransformation(geom);
      for (int i = 0; i < pmats.SizeK(); i++)
      {
         emb_tr.SetPointMat(pmats(i));
         // Get the local interpolation matrix for this refinement type
         fine_fe->GetTransferMatrix(*coarse_fe, emb_tr, lP(i));
         // Get the local mass matrix for this refinement type
         mass_integ->AssembleElementMatrix(*fine_fe, emb_tr, lM(i));
      }
   }

   Table ref_type_to_matrix;
   rtrans.GetCoarseToFineMap(*f_mesh, coarse_to_fine, coarse_to_ref_type,
                             ref_type_to_matrix, ref_type_to_geom);
   MFEM_ASSERT(coarse_to_fine.Size() == c_fes->GetNE(), "");

   const int total_ref_types = ref_type_to_geom.Size();
   int num_ref_types[Geometry::NumGeom], num_fine_elems[Geometry::NumGeom];
   Array<int> ref_type_to_coarse_elem_offset(total_ref_types);
   ref_type_to_fine_elem_offset.SetSize(total_ref_types);
   std::fill(num_ref_types, num_ref_types+Geometry::NumGeom, 0);
   std::fill(num_fine_elems, num_fine_elems+Geometry::NumGeom, 0);
   for (int i = 0; i < total_ref_types; i++)
   {
      Geometry::Type g = ref_type_to_geom[i];
      ref_type_to_coarse_elem_offset[i] = num_ref_types[g];
      ref_type_to_fine_elem_offset[i] = num_fine_elems[g];
      num_ref_types[g]++;
      num_fine_elems[g] += ref_type_to_matrix.RowSize(i);
   }
   DenseTensor localPtMP[Geometry::NumGeom];
   for (int g = 0; g < Geometry::NumGeom; g++)
   {
      if (num_ref_types[g] == 0) { continue; }
      const int fine_dofs = localP[g].SizeI();
      const int coarse_dofs = localP[g].SizeJ();
      localPtMP[g].SetSize(coarse_dofs, coarse_dofs, num_ref_types[g]);
      localR[g].SetSize(coarse_dofs, fine_dofs, num_fine_elems[g]);
   }
   for (int i = 0; i < total_ref_types; i++)
   {
      Geometry::Type g = ref_type_to_geom[i];
      DenseMatrix &lPtMP = localPtMP[g](ref_type_to_coarse_elem_offset[i]);
      int lR_offset = ref_type_to_fine_elem_offset[i]; // offset in localR[g]
      const int *mi = ref_type_to_matrix.GetRow(i);
      const int nm = ref_type_to_matrix.RowSize(i);
      lPtMP = 0.0;
      for (int s = 0; s < nm; s++)
      {
         DenseMatrix &lP = localP[g](mi[s]);
         DenseMatrix &lM = localM[g](mi[s]);
         DenseMatrix &lR = localR[g](lR_offset+s);
         MultAtB(lP, lM, lR); // lR = lP^T lM
         AddMult(lR, lP, lPtMP); // lPtMP += lP^T lM lP
      }
      DenseMatrixInverse lPtMP_inv(lPtMP);
      for (int s = 0; s < nm; s++)
      {
         DenseMatrix &lR = localR[g](lR_offset+s);
         lPtMP_inv.Mult(lR); // lR <- (P^T M P)^{-1} P^T M
      }
   }

   // Make a copy of the coarse element-to-dof Table.
   coarse_elem_dof = new Table(c_fes->GetElementToDofTable());
}

FiniteElementSpace::DerefinementOperator::~DerefinementOperator()
{
   delete coarse_elem_dof;
}

void FiniteElementSpace::DerefinementOperator
::Mult(const Vector &x, Vector &y) const
{
   Array<int> c_vdofs, f_vdofs;
   Vector loc_x, loc_y;
   DenseMatrix loc_x_mat, loc_y_mat;
   const int vdim = fine_fes->GetVDim();
   const int coarse_ndofs = height/vdim;
   for (int coarse_el = 0; coarse_el < coarse_to_fine.Size(); coarse_el++)
   {
      coarse_elem_dof->GetRow(coarse_el, c_vdofs);
      fine_fes->DofsToVDofs(c_vdofs, coarse_ndofs);
      loc_y.SetSize(c_vdofs.Size());
      loc_y = 0.0;
      loc_y_mat.UseExternalData(loc_y.GetData(), c_vdofs.Size()/vdim, vdim);
      const int ref_type = coarse_to_ref_type[coarse_el];
      const Geometry::Type geom = ref_type_to_geom[ref_type];
      const int *fine_elems = coarse_to_fine.GetRow(coarse_el);
      const int num_fine_elems = coarse_to_fine.RowSize(coarse_el);
      const int lR_offset = ref_type_to_fine_elem_offset[ref_type];
      for (int s = 0; s < num_fine_elems; s++)
      {
         const DenseMatrix &lR = localR[geom](lR_offset+s);
         fine_fes->GetElementVDofs(fine_elems[s], f_vdofs);
         x.GetSubVector(f_vdofs, loc_x);
         loc_x_mat.UseExternalData(loc_x.GetData(), f_vdofs.Size()/vdim, vdim);
         AddMult(lR, loc_x_mat, loc_y_mat);
      }
      y.SetSubVector(c_vdofs, loc_y);
   }
}

void FiniteElementSpace::GetLocalDerefinementMatrices(Geometry::Type geom,
                                                      DenseTensor &localR) const
{
   const FiniteElement *fe = fec->FiniteElementForGeometry(geom);

   const CoarseFineTransformations &dtrans =
      mesh->ncmesh->GetDerefinementTransforms();
   const DenseTensor &pmats = dtrans.point_matrices[geom];

   const int nmat = pmats.SizeK();
   const int ldof = fe->GetDof();

   IsoparametricTransformation isotr;
   isotr.SetIdentityTransformation(geom);

   // calculate local restriction matrices for all refinement types
   localR.SetSize(ldof, ldof, nmat);
   for (int i = 0; i < nmat; i++)
   {
      isotr.SetPointMat(pmats(i));
      fe->GetLocalRestriction(isotr, localR(i));
   }
}

SparseMatrix* FiniteElementSpace::DerefinementMatrix(int old_ndofs,
                                                     const Table* old_elem_dof,
                                                     const Table* old_elem_fos)
{
   /// TODO: Implement DofTransformation support

   MFEM_VERIFY(Nonconforming(), "Not implemented for conforming meshes.");
   MFEM_VERIFY(old_ndofs, "Missing previous (finer) space.");
   MFEM_VERIFY(ndofs <= old_ndofs, "Previous space is not finer.");

   Array<int> dofs, old_dofs, old_vdofs;
   Vector row;

   Mesh::GeometryList elem_geoms(*mesh);

   DenseTensor localR[Geometry::NumGeom];
   for (int i = 0; i < elem_geoms.Size(); i++)
   {
      GetLocalDerefinementMatrices(elem_geoms[i], localR[elem_geoms[i]]);
   }

   SparseMatrix *R = (elem_geoms.Size() != 1)
                     ? new SparseMatrix(ndofs*vdim, old_ndofs*vdim) // variable row size
                     : new SparseMatrix(ndofs*vdim, old_ndofs*vdim,
                                        localR[elem_geoms[0]].SizeI());

   Array<int> mark(R->Height());
   mark = 0;

   const CoarseFineTransformations &dtrans =
      mesh->ncmesh->GetDerefinementTransforms();

   MFEM_ASSERT(dtrans.embeddings.Size() == old_elem_dof->Size(), "");

   int num_marked = 0;
   for (int k = 0; k < dtrans.embeddings.Size(); k++)
   {
      const Embedding &emb = dtrans.embeddings[k];
      Geometry::Type geom = mesh->GetElementBaseGeometry(emb.parent);
      DenseMatrix &lR = localR[geom](emb.matrix);

      elem_dof->GetRow(emb.parent, dofs);
      old_elem_dof->GetRow(k, old_dofs);

      for (int vd = 0; vd < vdim; vd++)
      {
         old_dofs.Copy(old_vdofs);
         DofsToVDofs(vd, old_vdofs, old_ndofs);

         for (int i = 0; i < lR.Height(); i++)
         {
            if (!std::isfinite(lR(i, 0))) { continue; }

            int r = DofToVDof(dofs[i], vd);
            int m = (r >= 0) ? r : (-1 - r);

            if (!mark[m])
            {
               lR.GetRow(i, row);
               R->SetRow(r, old_vdofs, row);
               mark[m] = 1;
               num_marked++;
            }
         }
      }
   }

   MFEM_VERIFY(num_marked == R->Height(),
               "internal error: not all rows of R were set.");

   R->Finalize(); // no-op if fixed width
   return R;
}

void FiniteElementSpace::GetLocalRefinementMatrices(
   const FiniteElementSpace &coarse_fes, Geometry::Type geom,
   DenseTensor &localP) const
{
   // Assumptions: see the declaration of the method.

   const FiniteElement *fine_fe = fec->FiniteElementForGeometry(geom);
   const FiniteElement *coarse_fe =
      coarse_fes.fec->FiniteElementForGeometry(geom);

   const CoarseFineTransformations &rtrans = mesh->GetRefinementTransforms();
   const DenseTensor &pmats = rtrans.point_matrices[geom];

   int nmat = pmats.SizeK();

   IsoparametricTransformation isotr;
   isotr.SetIdentityTransformation(geom);

   // Calculate the local interpolation matrices for all refinement types
   localP.SetSize(fine_fe->GetDof(), coarse_fe->GetDof(), nmat);
   for (int i = 0; i < nmat; i++)
   {
      isotr.SetPointMat(pmats(i));
      fine_fe->GetTransferMatrix(*coarse_fe, isotr, localP(i));
   }
}

void FiniteElementSpace::Constructor(Mesh *mesh, NURBSExtension *NURBSext,
                                     const FiniteElementCollection *fec,
                                     int vdim, int ordering)
{
   this->mesh = mesh;
   this->fec = fec;
   this->vdim = vdim;
   this->ordering = (Ordering::Type) ordering;

   elem_dof = NULL;
   elem_fos = NULL;
   face_dof = NULL;

   sequence = 0;
   orders_changed = false;
   relaxed_hp = false;

   Th.SetType(Operator::ANY_TYPE);

   const NURBSFECollection *nurbs_fec =
      dynamic_cast<const NURBSFECollection *>(fec);
   if (nurbs_fec)
   {
      MFEM_VERIFY(mesh->NURBSext, "NURBS FE space requires a NURBS mesh.");

      if (NURBSext == NULL)
      {
         this->NURBSext = mesh->NURBSext;
         own_ext = 0;
      }
      else
      {
         this->NURBSext = NURBSext;
         own_ext = 1;
      }
      UpdateNURBS();
      cP = cR = cR_hp = NULL;
      cP_is_set = false;

      ConstructDoFTrans();
   }
   else
   {
      this->NURBSext = NULL;
      own_ext = 0;
      Construct();
   }

   BuildElementToDofTable();
}

void FiniteElementSpace::ConstructDoFTrans()
{
   DestroyDoFTrans();

   VDoFTrans.SetVDim(vdim);
   DoFTrans.SetSize(Geometry::NUM_GEOMETRIES);
   for (int i=0; i<DoFTrans.Size(); i++)
   {
      DoFTrans[i] = NULL;
   }
   if (mesh->Dimension() < 3) { return; }
   if (dynamic_cast<const ND_FECollection*>(fec))
   {
      const FiniteElement * nd_tri =
         fec->FiniteElementForGeometry(Geometry::TRIANGLE);
      if (nd_tri)
      {
         DoFTrans[Geometry::TRIANGLE] =
            new ND_TriDofTransformation(nd_tri->GetOrder());
      }

      const FiniteElement * nd_tet =
         fec->FiniteElementForGeometry(Geometry::TETRAHEDRON);
      if (nd_tet)
      {
         DoFTrans[Geometry::TETRAHEDRON] =
            new ND_TetDofTransformation(nd_tet->GetOrder());
      }
   }
}

NURBSExtension *FiniteElementSpace::StealNURBSext()
{
   if (NURBSext && !own_ext)
   {
      mfem_error("FiniteElementSpace::StealNURBSext");
   }
   own_ext = 0;

   return NURBSext;
}

void FiniteElementSpace::UpdateNURBS()
{
   MFEM_VERIFY(NURBSext, "NURBSExt not defined.");

   nvdofs = 0;
   nedofs = 0;
   nfdofs = 0;
   nbdofs = 0;
   bdofs = NULL;

   delete face_dof;
   face_dof = NULL;
   face_to_be.DeleteAll();

   dynamic_cast<const NURBSFECollection *>(fec)->Reset();

   ndofs = NURBSext->GetNDof();
   elem_dof = NURBSext->GetElementDofTable();
   bdr_elem_dof = NURBSext->GetBdrElementDofTable();

   mesh_sequence = mesh->GetSequence();
   sequence++;
}

void FiniteElementSpace::BuildNURBSFaceToDofTable() const
{
   if (face_dof) { return; }

   const int dim = mesh->Dimension();

   // Find bdr to face mapping
   face_to_be.SetSize(GetNF());
   face_to_be = -1;
   for (int b = 0; b < GetNBE(); b++)
   {
      int f = mesh->GetBdrElementEdgeIndex(b);
      face_to_be[f] = b;
   }

   // Loop over faces in correct order, to prevent a sort
   // Sort will destroy orientation info in ordering of dofs
   Array<Connection> face_dof_list;
   Array<int> row;
   for (int f = 0; f < GetNF(); f++)
   {
      int b = face_to_be[f];
      if (b == -1) { continue; }
      // FIXME: this assumes the boundary element and the face element have the
      //        same orientation.
      if (dim > 1)
      {
         const Element *fe = mesh->GetFace(f);
         const Element *be = mesh->GetBdrElement(b);
         const int nv = be->GetNVertices();
         const int *fv = fe->GetVertices();
         const int *bv = be->GetVertices();
         for (int i = 0; i < nv; i++)
         {
            MFEM_VERIFY(fv[i] == bv[i],
                        "non-matching face and boundary elements detected!");
         }
      }
      GetBdrElementDofs(b, row);
      Connection conn(f,0);
      for (int i = 0; i < row.Size(); i++)
      {
         conn.to = row[i];
         face_dof_list.Append(conn);
      }
   }
   face_dof = new Table(GetNF(), face_dof_list);
}

void FiniteElementSpace::Construct()
{
   // This method should be used only for non-NURBS spaces.
   MFEM_VERIFY(!NURBSext, "internal error");

   // Variable order space needs a nontrivial P matrix + also ghost elements
   // in parallel, we thus require the mesh to be NC.
   MFEM_VERIFY(!IsVariableOrder() || Nonconforming(),
               "Variable order space requires a nonconforming mesh.");

   elem_dof = NULL;
   elem_fos = NULL;
   bdr_elem_dof = NULL;
   bdr_elem_fos = NULL;
   face_dof = NULL;

   ndofs = 0;
   nvdofs = nedofs = nfdofs = nbdofs = 0;
   bdofs = NULL;

   cP = NULL;
   cR = NULL;
   cR_hp = NULL;
   cP_is_set = false;
   // 'Th' is initialized/destroyed before this method is called.

   int dim = mesh->Dimension();
   int order = fec->GetOrder();

   MFEM_VERIFY((mesh->GetNumGeometries(dim) > 0) || (mesh->GetNE() == 0),
               "Mesh was not correctly finalized.");

   bool mixed_elements = (mesh->GetNumGeometries(dim) > 1);
   bool mixed_faces = (dim > 2 && mesh->GetNumGeometries(2) > 1);

   Array<VarOrderBits> edge_orders, face_orders;
   if (IsVariableOrder())
   {
      // for variable order spaces, calculate orders of edges and faces
      CalcEdgeFaceVarOrders(edge_orders, face_orders);
   }
   else if (mixed_faces)
   {
      // for mixed faces we also create the var_face_dofs table, see below
      face_orders.SetSize(mesh->GetNFaces());
      face_orders = (VarOrderBits(1) << order);
   }

   // assign vertex DOFs
   if (mesh->GetNV())
   {
      nvdofs = mesh->GetNV() * fec->GetNumDof(Geometry::POINT, order);
   }

   // assign edge DOFs
   if (mesh->GetNEdges())
   {
      if (IsVariableOrder())
      {
         nedofs = MakeDofTable(1, edge_orders, var_edge_dofs, &var_edge_orders);
      }
      else
      {
         // the simple case: all edges are of the same order
         nedofs = mesh->GetNEdges() * fec->GetNumDof(Geometry::SEGMENT, order);
      }
   }

   // assign face DOFs
   if (mesh->GetNFaces())
   {
      if (IsVariableOrder() || mixed_faces)
      {
         // NOTE: for simplicity, we also use Table var_face_dofs for mixed faces
         nfdofs = MakeDofTable(2, face_orders, var_face_dofs,
                               IsVariableOrder() ? &var_face_orders : NULL);
         uni_fdof = -1;
      }
      else
      {
         // the simple case: all faces are of the same geometry and order
         uni_fdof = fec->GetNumDof(mesh->GetFaceGeometry(0), order);
         nfdofs = mesh->GetNFaces() * uni_fdof;
      }
   }

   // assign internal ("bubble") DOFs
   if (mesh->GetNE() && dim > 0)
   {
      if (IsVariableOrder() || mixed_elements)
      {
         bdofs = new int[mesh->GetNE()+1];
         bdofs[0] = 0;
         for (int i = 0; i < mesh->GetNE(); i++)
         {
            int p = GetElementOrderImpl(i);
            nbdofs += fec->GetNumDof(mesh->GetElementGeometry(i), p);
            bdofs[i+1] = nbdofs;
         }
      }
      else
      {
         // the simple case: all elements are the same
         bdofs = NULL;
         Geometry::Type geom = mesh->GetElementGeometry(0);
         nbdofs = mesh->GetNE() * fec->GetNumDof(geom, order);
      }
   }

   ndofs = nvdofs + nedofs + nfdofs + nbdofs;

   ConstructDoFTrans();

   // record the current mesh sequence number to detect refinement etc.
   mesh_sequence = mesh->GetSequence();

   // increment our sequence number to let GridFunctions know they need updating
   sequence++;

   // DOFs are now assigned according to current element orders
   orders_changed = false;

   // Do not build elem_dof Table here: in parallel it has to be constructed
   // later.
}

int FiniteElementSpace::MinOrder(VarOrderBits bits)
{
   MFEM_ASSERT(bits != 0, "invalid bit mask");
   for (int order = 0; bits != 0; order++, bits >>= 1)
   {
      if (bits & 1) { return order; }
   }
   return 0;
}

void FiniteElementSpace
::CalcEdgeFaceVarOrders(Array<VarOrderBits> &edge_orders,
                        Array<VarOrderBits> &face_orders) const
{
   MFEM_ASSERT(IsVariableOrder(), "");
   MFEM_ASSERT(Nonconforming(), "");
   MFEM_ASSERT(elem_order.Size() == mesh->GetNE(), "");

   edge_orders.SetSize(mesh->GetNEdges());  edge_orders = 0;
   face_orders.SetSize(mesh->GetNFaces());  face_orders = 0;

   // Calculate initial edge/face orders, as required by incident elements.
   // For each edge/face we accumulate in a bit-mask the orders of elements
   // sharing the edge/face.
   Array<int> E, F, ori;
   for (int i = 0; i < mesh->GetNE(); i++)
   {
      int order = elem_order[i];
      MFEM_ASSERT(order <= MaxVarOrder, "");
      VarOrderBits mask = (VarOrderBits(1) << order);

      mesh->GetElementEdges(i, E, ori);
      for (int j = 0; j < E.Size(); j++)
      {
         edge_orders[E[j]] |= mask;
      }

      if (mesh->Dimension() > 2)
      {
         mesh->GetElementFaces(i, F, ori);
         for (int j = 0; j < F.Size(); j++)
         {
            face_orders[F[j]] |= mask;
         }
      }
   }

   if (relaxed_hp)
   {
      // for relaxed conformity we don't need the masters to match the minimum
      // orders of the slaves, we can stop now
      return;
   }

   // Iterate while minimum orders propagate by master/slave relations
   // (and new orders also propagate from faces to incident edges).
   // See https://github.com/mfem/mfem/pull/1423#issuecomment-638930559
   // for an illustration of why this is necessary in hp meshes.
   bool done;
   do
   {
      done = true;

      // propagate from slave edges to master edges
      const NCMesh::NCList &edge_list = mesh->ncmesh->GetEdgeList();
      for (const NCMesh::Master &master : edge_list.masters)
      {
         VarOrderBits slave_orders = 0;
         for (int i = master.slaves_begin; i < master.slaves_end; i++)
         {
            slave_orders |= edge_orders[edge_list.slaves[i].index];
         }

         int min_order = MinOrder(slave_orders);
         if (min_order < MinOrder(edge_orders[master.index]))
         {
            edge_orders[master.index] |= (VarOrderBits(1) << min_order);
            done = false;
         }
      }

      // propagate from slave faces(+edges) to master faces(+edges)
      const NCMesh::NCList &face_list = mesh->ncmesh->GetFaceList();
      for (const NCMesh::Master &master : face_list.masters)
      {
         VarOrderBits slave_orders = 0;
         for (int i = master.slaves_begin; i < master.slaves_end; i++)
         {
            const NCMesh::Slave &slave = face_list.slaves[i];
            if (slave.index >= 0)
            {
               slave_orders |= face_orders[slave.index];

               mesh->GetFaceEdges(slave.index, E, ori);
               for (int j = 0; j < E.Size(); j++)
               {
                  slave_orders |= edge_orders[E[j]];
               }
            }
            else
            {
               // degenerate face (i.e., edge-face constraint)
               slave_orders |= edge_orders[-1 - slave.index];
            }
         }

         int min_order = MinOrder(slave_orders);
         if (min_order < MinOrder(face_orders[master.index]))
         {
            face_orders[master.index] |= (VarOrderBits(1) << min_order);
            done = false;
         }
      }

      // make sure edges support (new) orders required by incident faces
      for (int i = 0; i < mesh->GetNFaces(); i++)
      {
         mesh->GetFaceEdges(i, E, ori);
         for (int j = 0; j < E.Size(); j++)
         {
            edge_orders[E[j]] |= face_orders[i];
         }
      }
   }
   while (!done);
}

int FiniteElementSpace::MakeDofTable(int ent_dim,
                                     const Array<int> &entity_orders,
                                     Table &entity_dofs,
                                     Array<char> *var_ent_order)
{
   // The tables var_edge_dofs and var_face_dofs hold DOF assignments for edges
   // and faces of a variable order space, in which each edge/face may host
   // several DOF sets, called DOF set variants. Example: an edge 'i' shared by
   // 4 hexes of orders 2, 3, 4, 5 will hold four DOF sets, each starting at
   // indices e.g. 100, 101, 103, 106, respectively. These numbers are stored
   // in row 'i' of var_edge_dofs. Variant zero is always the lowest order DOF
   // set, followed by consecutive ranges of higher order DOFs. Variable order
   // faces are handled similarly by var_face_dofs, which holds at most two DOF
   // set variants per face. The tables are empty for constant-order spaces.

   int num_ent = entity_orders.Size();
   int total_dofs = 0;

   Array<Connection> list;
   list.Reserve(2*num_ent);

   if (var_ent_order)
   {
      var_ent_order->SetSize(0);
      var_ent_order->Reserve(num_ent);
   }

   // assign DOFs according to order bit masks
   for (int i = 0; i < num_ent; i++)
   {
      auto geom = (ent_dim == 1) ? Geometry::SEGMENT : mesh->GetFaceGeometry(i);

      VarOrderBits bits = entity_orders[i];
      for (int order = 0; bits != 0; order++, bits >>= 1)
      {
         if (bits & 1)
         {
            int dofs = fec->GetNumDof(geom, order);
            list.Append(Connection(i, total_dofs));
            total_dofs += dofs;

            if (var_ent_order) { var_ent_order->Append(order); }
         }
      }
   }

   // append a dummy row as terminator
   list.Append(Connection(num_ent, total_dofs));

   // build the table
   entity_dofs.MakeFromList(num_ent+1, list);

   return total_dofs;
}

int FiniteElementSpace::FindDofs(const Table &var_dof_table,
                                 int row, int ndof) const
{
   const int *beg = var_dof_table.GetRow(row);
   const int *end = var_dof_table.GetRow(row + 1); // terminator, see above

   while (beg < end)
   {
      // return the appropriate range of DOFs
      if ((beg[1] - beg[0]) == ndof) { return beg[0]; }
      beg++;
   }

   MFEM_ABORT("DOFs not found for ndof = " << ndof);
   return 0;
}

int FiniteElementSpace::GetEdgeOrder(int edge, int variant) const
{
   if (!IsVariableOrder()) { return fec->GetOrder(); }

   const int* beg = var_edge_dofs.GetRow(edge);
   const int* end = var_edge_dofs.GetRow(edge + 1);
   if (variant >= end - beg) { return -1; } // past last variant

   return var_edge_orders[var_edge_dofs.GetI()[edge] + variant];
}

int FiniteElementSpace::GetFaceOrder(int face, int variant) const
{
   if (!IsVariableOrder())
   {
      // face order can be different from fec->GetOrder()
      Geometry::Type geom = mesh->GetFaceGeometry(face);
      return fec->FiniteElementForGeometry(geom)->GetOrder();
   }

   const int* beg = var_face_dofs.GetRow(face);
   const int* end = var_face_dofs.GetRow(face + 1);
   if (variant >= end - beg) { return -1; } // past last variant

   return var_face_orders[var_face_dofs.GetI()[face] + variant];
}

int FiniteElementSpace::GetNVariants(int entity, int index) const
{
   MFEM_ASSERT(IsVariableOrder(), "");
   const Table &dof_table = (entity == 1) ? var_edge_dofs : var_face_dofs;

   MFEM_ASSERT(index >= 0 && index < dof_table.Size(), "");
   return dof_table.GetRow(index + 1) - dof_table.GetRow(index);
}

static const char* msg_orders_changed =
   "Element orders changed, you need to Update() the space first.";

DofTransformation *
FiniteElementSpace::GetElementDofs(int elem, Array<int> &dofs) const
{
   MFEM_VERIFY(!orders_changed, msg_orders_changed);

   if (elem_dof)
   {
      elem_dof->GetRow(elem, dofs);

      if (DoFTrans[mesh->GetElementBaseGeometry(elem)])
      {
         Array<int> Fo;
         elem_fos -> GetRow (elem, Fo);
         DoFTrans[mesh->GetElementBaseGeometry(elem)]->SetFaceOrientations(Fo);
      }
      return DoFTrans[mesh->GetElementBaseGeometry(elem)];
   }

   Array<int> V, E, Eo, F, Fo; // TODO: LocalArray

   int dim = mesh->Dimension();
   auto geom = mesh->GetElementGeometry(elem);
   int order = GetElementOrderImpl(elem);

   int nv = fec->GetNumDof(Geometry::POINT, order);
   int ne = (dim > 1) ? fec->GetNumDof(Geometry::SEGMENT, order) : 0;
   int nb = (dim > 0) ? fec->GetNumDof(geom, order) : 0;

   if (nv) { mesh->GetElementVertices(elem, V); }
   if (ne) { mesh->GetElementEdges(elem, E, Eo); }

   int nfd = 0;
   if (dim > 2 && fec->HasFaceDofs(geom, order))
   {
      mesh->GetElementFaces(elem, F, Fo);
      for (int i = 0; i < F.Size(); i++)
      {
         nfd += fec->GetNumDof(mesh->GetFaceGeometry(F[i]), order);
      }
      if (DoFTrans[mesh->GetElementBaseGeometry(elem)])
      {
	DoFTrans[mesh->GetElementBaseGeometry(elem)]
	  -> SetFaceOrientations(Fo);
      }
   }

   dofs.SetSize(0);
   dofs.Reserve(nv*V.Size() + ne*E.Size() + nfd + nb);

   if (nv) // vertex DOFs
   {
      for (int i = 0; i < V.Size(); i++)
      {
         for (int j = 0; j < nv; j++)
         {
            dofs.Append(V[i]*nv + j);
         }
      }
   }

   if (ne) // edge DOFs
   {
      for (int i = 0; i < E.Size(); i++)
      {
         int ebase = IsVariableOrder() ? FindEdgeDof(E[i], ne) : E[i]*ne;
         const int *ind = fec->GetDofOrdering(Geometry::SEGMENT, order, Eo[i]);

         for (int j = 0; j < ne; j++)
         {
            dofs.Append(EncodeDof(nvdofs + ebase, ind[j]));
         }
      }
   }

   if (nfd) // face DOFs
   {
      for (int i = 0; i < F.Size(); i++)
      {
         auto fgeom = mesh->GetFaceGeometry(F[i]);
         int nf = fec->GetNumDof(fgeom, order);

         int fbase = (var_face_dofs.Size() > 0) ? FindFaceDof(F[i], nf) : F[i]*nf;
         const int *ind = fec->GetDofOrdering(fgeom, order, Fo[i]);

         for (int j = 0; j < nf; j++)
         {
            dofs.Append(EncodeDof(nvdofs + nedofs + fbase, ind[j]));
         }
      }
   }

   if (nb) // interior ("bubble") DOFs
   {
      int bbase = bdofs ? bdofs[elem] : elem*nb;
      bbase += nvdofs + nedofs + nfdofs;

      for (int j = 0; j < nb; j++)
      {
         dofs.Append(bbase + j);
      }
   }
   return DoFTrans[mesh->GetElementBaseGeometry(elem)];
}

const FiniteElement *FiniteElementSpace::GetFE(int i) const
{
   if (i < 0 || !mesh->GetNE()) { return NULL; }
   MFEM_VERIFY(i < mesh->GetNE(),
               "Invalid element id " << i << ", maximum allowed " << mesh->GetNE()-1);

   const FiniteElement *FE =
      fec->GetFE(mesh->GetElementGeometry(i), GetElementOrderImpl(i));

   if (NURBSext)
   {
      NURBSext->LoadFE(i, FE);
   }
   else
   {
#ifdef MFEM_DEBUG
      // consistency check: fec->GetOrder() and FE->GetOrder() should return
      // the same value (for standard, constant-order spaces)
      if (!IsVariableOrder() && FE->GetDim() > 0)
      {
         MFEM_ASSERT(FE->GetOrder() == fec->GetOrder(),
                     "internal error: " <<
                     FE->GetOrder() << " != " << fec->GetOrder());
      }
#endif
   }

   return FE;
}

DofTransformation *
FiniteElementSpace::GetBdrElementDofs(int bel, Array<int> &dofs) const
{
   MFEM_VERIFY(!orders_changed, msg_orders_changed);

   if (bdr_elem_dof)
   {
      bdr_elem_dof->GetRow(bel, dofs);

      if (DoFTrans[mesh->GetBdrElementBaseGeometry(bel)])
      {
         Array<int> Fo;
         bdr_elem_fos -> GetRow (bel, Fo);
         DoFTrans[mesh->GetBdrElementBaseGeometry(bel)]->
	   SetFaceOrientations(Fo);
      }
      return DoFTrans[mesh->GetBdrElementBaseGeometry(bel)];
   }

   Array<int> V, E, Eo, Fo; // TODO: LocalArray
   int F, oF;

   int dim = mesh->Dimension();
   auto geom = mesh->GetBdrElementGeometry(bel);
   int order = fec->GetOrder();

   if (IsVariableOrder()) // determine order from adjacent element
   {
      int elem, info;
      mesh->GetBdrElementAdjacentElement(bel, elem, info);
      order = elem_order[elem];
   }

   int nv = fec->GetNumDof(Geometry::POINT, order);
   int ne = (dim > 1) ? fec->GetNumDof(Geometry::SEGMENT, order) : 0;
   int nf = (dim > 2) ? fec->GetNumDof(geom, order) : 0;

   if (nv) { mesh->GetBdrElementVertices(bel, V); }
   if (ne) { mesh->GetBdrElementEdges(bel, E, Eo); }
   if (nf)
   {
      mesh->GetBdrElementFace(bel, &F, &oF);

      if (DoFTrans[mesh->GetBdrElementBaseGeometry(bel)])
      {
         Fo.Append(oF);
         DoFTrans[mesh->GetBdrElementBaseGeometry(bel)]->
	    SetFaceOrientations(Fo);
      }
   }

   dofs.SetSize(0);
   dofs.Reserve(nv*V.Size() + ne*E.Size() + nf);

   if (nv) // vertex DOFs
   {
      for (int i = 0; i < V.Size(); i++)
      {
         for (int j = 0; j < nv; j++)
         {
            dofs.Append(V[i]*nv + j);
         }
      }
   }

   if (ne) // edge DOFs
   {
      for (int i = 0; i < E.Size(); i++)
      {
         int ebase = IsVariableOrder() ? FindEdgeDof(E[i], ne) : E[i]*ne;
         const int *ind = fec->GetDofOrdering(Geometry::SEGMENT, order, Eo[i]);

         for (int j = 0; j < ne; j++)
         {
            dofs.Append(EncodeDof(nvdofs + ebase, ind[j]));
         }
      }
   }

   if (nf) // face DOFs
   {
      int fbase = (var_face_dofs.Size() > 0) ? FindFaceDof(F, nf) : F*nf;
      const int *ind = fec->GetDofOrdering(geom, order, oF);

      for (int j = 0; j < nf; j++)
      {
         dofs.Append(EncodeDof(nvdofs + nedofs + fbase, ind[j]));
      }
   }

   return DoFTrans[mesh->GetBdrElementBaseGeometry(bel)];
}

int FiniteElementSpace::GetFaceDofs(int face, Array<int> &dofs,
                                    int variant) const
{
   MFEM_VERIFY(!orders_changed, msg_orders_changed);

   // If face_dof is already built, use it.
   // If it is not and we have a NURBS space, build the face_dof and use it.
   if ((face_dof && variant == 0) ||
       (NURBSext && (BuildNURBSFaceToDofTable(), true)))
   {
      face_dof->GetRow(face, dofs);
      return fec->GetOrder();
   }

   int order, nf, fbase;
   int dim = mesh->Dimension();
   auto fgeom = (dim > 2) ? mesh->GetFaceGeometry(face) : Geometry::INVALID;

   if (var_face_dofs.Size() > 0) // variable orders or *mixed* faces
   {
      const int* beg = var_face_dofs.GetRow(face);
      const int* end = var_face_dofs.GetRow(face + 1);
      if (variant >= end - beg) { return -1; } // past last face DOFs

      fbase = beg[variant];
      nf = beg[variant+1] - fbase;

      order = !IsVariableOrder() ? fec->GetOrder() :
              var_face_orders[var_face_dofs.GetI()[face] + variant];
      MFEM_ASSERT(fec->GetNumDof(fgeom, order) == nf, "");
   }
   else
   {
      if (variant > 0) { return -1; }
      order = fec->GetOrder();
      nf = (dim > 2) ? fec->GetNumDof(fgeom, order) : 0;
      fbase = face*nf;
   }

   // for 1D, 2D and 3D faces
   int nv = fec->GetNumDof(Geometry::POINT, order);
   int ne = (dim > 1) ? fec->GetNumDof(Geometry::SEGMENT, order) : 0;

   Array<int> V, E, Eo;
   if (nv) { mesh->GetFaceVertices(face, V); }
   if (ne) { mesh->GetFaceEdges(face, E, Eo); }

   dofs.SetSize(0);
   dofs.Reserve(V.Size() * nv + E.Size() * ne + nf);

   if (nv) // vertex DOFs
   {
      for (int i = 0; i < V.Size(); i++)
      {
         for (int j = 0; j < nv; j++)
         {
            dofs.Append(V[i]*nv + j);
         }
      }
   }
   if (ne) // edge DOFs
   {
      for (int i = 0; i < E.Size(); i++)
      {
         int ebase = IsVariableOrder() ? FindEdgeDof(E[i], ne) : E[i]*ne;
         const int *ind = fec->GetDofOrdering(Geometry::SEGMENT, order, Eo[i]);

         for (int j = 0; j < ne; j++)
         {
            dofs.Append(EncodeDof(nvdofs + ebase, ind[j]));
         }
      }
   }
   for (int j = 0; j < nf; j++)
   {
      dofs.Append(nvdofs + nedofs + fbase + j);
   }

   return order;
}

int FiniteElementSpace::GetEdgeDofs(int edge, Array<int> &dofs,
                                    int variant) const
{
   MFEM_VERIFY(!orders_changed, msg_orders_changed);

   int order, ne, base;
   if (IsVariableOrder())
   {
      const int* beg = var_edge_dofs.GetRow(edge);
      const int* end = var_edge_dofs.GetRow(edge + 1);
      if (variant >= end - beg) { return -1; } // past last edge DOFs

      base = beg[variant];
      ne = beg[variant+1] - base;

      order = var_edge_orders[var_edge_dofs.GetI()[edge] + variant];
      MFEM_ASSERT(fec->GetNumDof(Geometry::SEGMENT, order) == ne, "");
   }
   else
   {
      if (variant > 0) { return -1; }
      order = fec->GetOrder();
      ne = fec->GetNumDof(Geometry::SEGMENT, order);
      base = edge*ne;
   }

   Array<int> V; // TODO: LocalArray
   int nv = fec->GetNumDof(Geometry::POINT, order);
   if (nv) { mesh->GetEdgeVertices(edge, V); }

   dofs.SetSize(0);
   dofs.Reserve(2*nv + ne);

   for (int i = 0; i < 2; i++)
   {
      for (int j = 0; j < nv; j++)
      {
         dofs.Append(V[i]*nv + j);
      }
   }
   for (int j = 0; j < ne; j++)
   {
      dofs.Append(nvdofs + base + j);
   }

   return order;
}

void FiniteElementSpace::GetVertexDofs(int i, Array<int> &dofs) const
{
   int nv = fec->DofForGeometry(Geometry::POINT);
   dofs.SetSize(nv);
   for (int j = 0; j < nv; j++)
   {
      dofs[j] = i*nv+j;
   }
}

void FiniteElementSpace::GetElementInteriorDofs(int i, Array<int> &dofs) const
{
   MFEM_VERIFY(!orders_changed, msg_orders_changed);

   int nb = fec->GetNumDof(mesh->GetElementGeometry(i), GetElementOrderImpl(i));
   int base = bdofs ? bdofs[i] : i*nb;

   dofs.SetSize(nb);
   base += nvdofs + nedofs + nfdofs;
   for (int j = 0; j < nb; j++)
   {
      dofs[j] = base + j;
   }
}

int FiniteElementSpace::GetNumElementInteriorDofs(int i) const
{
   return fec->GetNumDof(mesh->GetElementGeometry(i),
                         GetElementOrderImpl(i));
}

void FiniteElementSpace::GetEdgeInteriorDofs(int i, Array<int> &dofs) const
{
   MFEM_VERIFY(!IsVariableOrder(), "not implemented");

   int ne = fec->DofForGeometry(Geometry::SEGMENT);
   dofs.SetSize (ne);
   for (int j = 0, k = nvdofs+i*ne; j < ne; j++, k++)
   {
      dofs[j] = k;
   }
}

void FiniteElementSpace::GetFaceInteriorDofs(int i, Array<int> &dofs) const
{
   MFEM_VERIFY(!IsVariableOrder(), "not implemented");

   int nf, base;
   if (var_face_dofs.Size() > 0) // mixed faces
   {
      base = var_face_dofs.GetRow(i)[0];
      nf = var_face_dofs.GetRow(i)[1] - base;
   }
   else
   {
      auto geom = mesh->GetFaceGeometry(0);
      nf = fec->GetNumDof(geom, fec->GetOrder());
      base = i*nf;
   }

   dofs.SetSize(nf);
   for (int j = 0; j < nf; j++)
   {
      dofs[j] = nvdofs + nedofs + base + j;
   }
}

const FiniteElement *FiniteElementSpace::GetBE(int i) const
{
   int order = fec->GetOrder();

   if (IsVariableOrder()) // determine order from adjacent element
   {
      int elem, info;
      mesh->GetBdrElementAdjacentElement(i, elem, info);
      order = elem_order[elem];
   }

   const FiniteElement *BE;
   switch (mesh->Dimension())
   {
      case 1:
         BE = fec->GetFE(Geometry::POINT, order);
         break;
      case 2:
         BE = fec->GetFE(Geometry::SEGMENT, order);
         break;
      case 3:
      default:
         BE = fec->GetFE(mesh->GetBdrElementBaseGeometry(i), order);
   }

   if (NURBSext)
   {
      NURBSext->LoadBE(i, BE);
   }

   return BE;
}

const FiniteElement *FiniteElementSpace::GetFaceElement(int i) const
{
   MFEM_VERIFY(!IsVariableOrder(), "not implemented");

   const FiniteElement *fe;
   switch (mesh->Dimension())
   {
      case 1:
         fe = fec->FiniteElementForGeometry(Geometry::POINT);
         break;
      case 2:
         fe = fec->FiniteElementForGeometry(Geometry::SEGMENT);
         break;
      case 3:
      default:
         fe = fec->FiniteElementForGeometry(mesh->GetFaceBaseGeometry(i));
   }

   if (NURBSext)
   {
      // Ensure 'face_to_be' is built:
      if (!face_dof) { BuildNURBSFaceToDofTable(); }
      MFEM_ASSERT(face_to_be[i] >= 0,
                  "NURBS mesh: only boundary faces are supported!");
      NURBSext->LoadBE(face_to_be[i], fe);
   }

   return fe;
}

const FiniteElement *FiniteElementSpace::GetEdgeElement(int i,
                                                        int variant) const
{
   MFEM_ASSERT(mesh->Dimension() > 1, "No edges with mesh dimension < 2");

   int eo = IsVariableOrder() ? GetEdgeOrder(i, variant) : fec->GetOrder();
   return fec->GetFE(Geometry::SEGMENT, eo);
}

const FiniteElement *FiniteElementSpace
::GetTraceElement(int i, Geometry::Type geom_type) const
{
   return fec->TraceFiniteElementForGeometry(geom_type);
}

FiniteElementSpace::~FiniteElementSpace()
{
   Destroy();
}

void FiniteElementSpace::Destroy()
{
   delete cR;
   delete cR_hp;
   delete cP;
   Th.Clear();
   L2E_nat.Clear();
   L2E_lex.Clear();
   for (int i = 0; i < E2Q_array.Size(); i++)
   {
      delete E2Q_array[i];
   }
   E2Q_array.SetSize(0);
   for (auto &x : L2F)
   {
      delete x.second;
   }
   for (int i = 0; i < E2IFQ_array.Size(); i++)
   {
      delete E2IFQ_array[i];
   }
   E2IFQ_array.SetSize(0);
   for (int i = 0; i < E2BFQ_array.Size(); i++)
   {
      delete E2BFQ_array[i];
   }
   E2BFQ_array.SetSize(0);

   DestroyDoFTrans();

   dof_elem_array.DeleteAll();
   dof_ldof_array.DeleteAll();

   if (NURBSext)
   {
      if (own_ext) { delete NURBSext; }
      delete face_dof;
      face_to_be.DeleteAll();
   }
   else
   {
      delete elem_dof;
      delete elem_fos;
      delete bdr_elem_dof;
      delete bdr_elem_fos;
      delete face_dof;

      delete [] bdofs;
   }
   ceed::RemoveBasisAndRestriction(this);
}

void FiniteElementSpace::DestroyDoFTrans()
{
   for (int i = 0; i < DoFTrans.Size(); i++)
   {
      delete DoFTrans[i];
   }
   DoFTrans.SetSize(0);
}

void FiniteElementSpace::GetTransferOperator(
   const FiniteElementSpace &coarse_fes, OperatorHandle &T) const
{
   // Assumptions: see the declaration of the method.

   if (T.Type() == Operator::MFEM_SPARSEMAT)
   {
      Mesh::GeometryList elem_geoms(*mesh);

      DenseTensor localP[Geometry::NumGeom];
      for (int i = 0; i < elem_geoms.Size(); i++)
      {
         GetLocalRefinementMatrices(coarse_fes, elem_geoms[i],
                                    localP[elem_geoms[i]]);
      }
      T.Reset(RefinementMatrix_main(coarse_fes.GetNDofs(),
                                    coarse_fes.GetElementToDofTable(),
                                    coarse_fes.
                                    GetElementToFaceOrientationTable(),
                                    localP));
   }
   else
   {
      T.Reset(new RefinementOperator(this, &coarse_fes));
   }
}

void FiniteElementSpace::GetTrueTransferOperator(
   const FiniteElementSpace &coarse_fes, OperatorHandle &T) const
{
   const SparseMatrix *coarse_P = coarse_fes.GetConformingProlongation();

   Operator::Type req_type = T.Type();
   GetTransferOperator(coarse_fes, T);

   if (req_type == Operator::MFEM_SPARSEMAT)
   {
      if (GetConformingRestriction())
      {
         T.Reset(mfem::Mult(*cR, *T.As<SparseMatrix>()));
      }
      if (coarse_P)
      {
         T.Reset(mfem::Mult(*T.As<SparseMatrix>(), *coarse_P));
      }
   }
   else
   {
      const int RP_case = bool(GetConformingRestriction()) + 2*bool(coarse_P);
      if (RP_case == 0) { return; }
      const bool owner = T.OwnsOperator();
      T.SetOperatorOwner(false);
      switch (RP_case)
      {
         case 1:
            T.Reset(new ProductOperator(cR, T.Ptr(), false, owner));
            break;
         case 2:
            T.Reset(new ProductOperator(T.Ptr(), coarse_P, owner, false));
            break;
         case 3:
            T.Reset(new TripleProductOperator(
                       cR, T.Ptr(), coarse_P, false, owner, false));
            break;
      }
   }
}

void FiniteElementSpace::UpdateElementOrders()
{
   const CoarseFineTransformations &cf_tr = mesh->GetRefinementTransforms();

   Array<char> new_order(mesh->GetNE());
   switch (mesh->GetLastOperation())
   {
      case Mesh::REFINE:
      {
         for (int i = 0; i < mesh->GetNE(); i++)
         {
            new_order[i] = elem_order[cf_tr.embeddings[i].parent];
         }
         break;
      }
      default:
         MFEM_ABORT("not implemented yet");
   }

   mfem::Swap(elem_order, new_order);
}

void FiniteElementSpace::Update(bool want_transform)
{
   if (!orders_changed)
   {
      if (mesh->GetSequence() == mesh_sequence)
      {
         return; // mesh and space are in sync, no-op
      }
      if (want_transform && mesh->GetSequence() != mesh_sequence + 1)
      {
         MFEM_ABORT("Error in update sequence. Space needs to be updated after "
                    "each mesh modification.");
      }
   }
   else
   {
      if (mesh->GetSequence() != mesh_sequence)
      {
         MFEM_ABORT("Updating space after both mesh change and element order "
                    "change is not supported. Please update separately after "
                    "each change.");
      }
   }

   if (NURBSext)
   {
      UpdateNURBS();
      return;
   }

   Table* old_elem_dof = NULL;
   Table* old_elem_fos = NULL;
   int old_ndofs;
   bool old_orders_changed = orders_changed;

   // save old DOF table
   if (want_transform)
   {
      old_elem_dof = elem_dof;
      old_elem_fos = elem_fos;
      elem_dof = NULL;
      elem_fos = NULL;
      old_ndofs = ndofs;
   }

   // update the 'elem_order' array if the mesh has changed
   if (IsVariableOrder() && mesh->GetSequence() != mesh_sequence)
   {
      UpdateElementOrders();
   }

   Destroy(); // calls Th.Clear()
   Construct();
   BuildElementToDofTable();

   if (want_transform)
   {
      MFEM_VERIFY(!old_orders_changed, "Interpolation for element order change "
                  "is not implemented yet, sorry.");

      // calculate appropriate GridFunction transformation
      switch (mesh->GetLastOperation())
      {
         case Mesh::REFINE:
         {
            if (Th.Type() != Operator::MFEM_SPARSEMAT)
            {
               Th.Reset(new RefinementOperator(this, old_elem_dof,
                                               old_elem_fos, old_ndofs));
               // The RefinementOperator takes ownership of 'old_elem_dof', so
               // we no longer own it:
               old_elem_dof = NULL;
            }
            else
            {
               // calculate fully assembled matrix
               Th.Reset(RefinementMatrix(old_ndofs, old_elem_dof,
                                         old_elem_fos));
            }
            break;
         }

         case Mesh::DEREFINE:
         {
            BuildConformingInterpolation();
            Th.Reset(DerefinementMatrix(old_ndofs, old_elem_dof, old_elem_fos));
            if (cP && cR)
            {
               Th.SetOperatorOwner(false);
               Th.Reset(new TripleProductOperator(cP, cR, Th.Ptr(),
                                                  false, false, true));
            }
            break;
         }

         default:
            break;
      }

      delete old_elem_dof;
   }
}

void FiniteElementSpace::UpdateMeshPointer(Mesh *new_mesh)
{
   mesh = new_mesh;
}

void FiniteElementSpace::Save(std::ostream &out) const
{
   int fes_format = 90; // the original format, v0.9
   bool nurbs_unit_weights = false;

   // Determine the format that should be used.
   if (!NURBSext)
   {
      // TODO: if this is a variable-order FE space, use fes_format = 100.
   }
   else
   {
      const NURBSFECollection *nurbs_fec =
         dynamic_cast<const NURBSFECollection *>(fec);
      MFEM_VERIFY(nurbs_fec, "invalid FE collection");
      nurbs_fec->SetOrder(NURBSext->GetOrder());
      const double eps = 5e-14;
      nurbs_unit_weights = (NURBSext->GetWeights().Min() >= 1.0-eps &&
                            NURBSext->GetWeights().Max() <= 1.0+eps);
      if ((NURBSext->GetOrder() == NURBSFECollection::VariableOrder) ||
          (NURBSext != mesh->NURBSext && !nurbs_unit_weights) ||
          (NURBSext->GetMaster().Size() != 0 ))
      {
         fes_format = 100; // v1.0 format
      }
   }

   out << (fes_format == 90 ?
           "FiniteElementSpace\n" : "MFEM FiniteElementSpace v1.0\n")
       << "FiniteElementCollection: " << fec->Name() << '\n'
       << "VDim: " << vdim << '\n'
       << "Ordering: " << ordering << '\n';

   if (fes_format == 100) // v1.0
   {
      if (!NURBSext)
      {
         // TODO: this is a variable-order FE space --> write 'element_orders'.
      }
      else if (NURBSext != mesh->NURBSext)
      {
         if (NURBSext->GetOrder() != NURBSFECollection::VariableOrder)
         {
            out << "NURBS_order\n" << NURBSext->GetOrder() << '\n';
         }
         else
         {
            out << "NURBS_orders\n";
            // 1 = do not write the size, just the entries:
            NURBSext->GetOrders().Save(out, 1);
         }
         // If periodic BCs are given, write connectivity
         if (NURBSext->GetMaster().Size() != 0 )
         {
            out <<"NURBS_periodic\n";
            NURBSext->GetMaster().Save(out);
            NURBSext->GetSlave().Save(out);
         }
         // If the weights are not unit, write them to the output:
         if (!nurbs_unit_weights)
         {
            out << "NURBS_weights\n";
            NURBSext->GetWeights().Print(out, 1);
         }
      }
      out << "End: MFEM FiniteElementSpace v1.0\n";
   }
}

FiniteElementCollection *FiniteElementSpace::Load(Mesh *m, std::istream &input)
{
   string buff;
   int fes_format = 0, ord;
   FiniteElementCollection *r_fec;

   Destroy();

   input >> std::ws;
   getline(input, buff);  // 'FiniteElementSpace'
   filter_dos(buff);
   if (buff == "FiniteElementSpace") { fes_format = 90; /* v0.9 */ }
   else if (buff == "MFEM FiniteElementSpace v1.0") { fes_format = 100; }
   else { MFEM_ABORT("input stream is not a FiniteElementSpace!"); }
   getline(input, buff, ' '); // 'FiniteElementCollection:'
   input >> std::ws;
   getline(input, buff);
   filter_dos(buff);
   r_fec = FiniteElementCollection::New(buff.c_str());
   getline(input, buff, ' '); // 'VDim:'
   input >> vdim;
   getline(input, buff, ' '); // 'Ordering:'
   input >> ord;

   NURBSFECollection *nurbs_fec = dynamic_cast<NURBSFECollection*>(r_fec);
   NURBSExtension *NURBSext = NULL;
   if (fes_format == 90) // original format, v0.9
   {
      if (nurbs_fec)
      {
         MFEM_VERIFY(m->NURBSext, "NURBS FE collection requires a NURBS mesh!");
         const int order = nurbs_fec->GetOrder();
         if (order != m->NURBSext->GetOrder() &&
             order != NURBSFECollection::VariableOrder)
         {
            NURBSext = new NURBSExtension(m->NURBSext, order);
         }
      }
   }
   else if (fes_format == 100) // v1.0
   {
      while (1)
      {
         skip_comment_lines(input, '#');
         MFEM_VERIFY(input.good(), "error reading FiniteElementSpace v1.0");
         getline(input, buff);
         filter_dos(buff);
         if (buff == "NURBS_order" || buff == "NURBS_orders")
         {
            MFEM_VERIFY(nurbs_fec,
                        buff << ": NURBS FE collection is required!");
            MFEM_VERIFY(m->NURBSext, buff << ": NURBS mesh is required!");
            MFEM_VERIFY(!NURBSext, buff << ": order redefinition!");
            if (buff == "NURBS_order")
            {
               int order;
               input >> order;
               NURBSext = new NURBSExtension(m->NURBSext, order);
            }
            else
            {
               Array<int> orders;
               orders.Load(m->NURBSext->GetNKV(), input);
               NURBSext = new NURBSExtension(m->NURBSext, orders);
            }
         }
         else if (buff == "NURBS_periodic")
         {
            Array<int> master, slave;
            master.Load(input);
            slave.Load(input);
            NURBSext->ConnectBoundaries(master,slave);
         }
         else if (buff == "NURBS_weights")
         {
            MFEM_VERIFY(NURBSext, "NURBS_weights: NURBS_orders have to be "
                        "specified before NURBS_weights!");
            NURBSext->GetWeights().Load(input, NURBSext->GetNDof());
         }
         else if (buff == "element_orders")
         {
            MFEM_VERIFY(!nurbs_fec, "section element_orders cannot be used "
                        "with a NURBS FE collection");
            MFEM_ABORT("element_orders: not implemented yet!");
         }
         else if (buff == "End: MFEM FiniteElementSpace v1.0")
         {
            break;
         }
         else
         {
            MFEM_ABORT("unknown section: " << buff);
         }
      }
   }

   Constructor(m, NURBSext, r_fec, vdim, ord);

   return r_fec;
}


void QuadratureSpace::Construct()
{
   // protected method
   int offset = 0;
   const int num_elem = mesh->GetNE();
   element_offsets = new int[num_elem + 1];
   for (int g = 0; g < Geometry::NumGeom; g++)
   {
      int_rule[g] = NULL;
   }
   for (int i = 0; i < num_elem; i++)
   {
      element_offsets[i] = offset;
      int geom = mesh->GetElementBaseGeometry(i);
      if (int_rule[geom] == NULL)
      {
         int_rule[geom] = &IntRules.Get(geom, order);
      }
      offset += int_rule[geom]->GetNPoints();
   }
   element_offsets[num_elem] = size = offset;
}

QuadratureSpace::QuadratureSpace(Mesh *mesh_, std::istream &in)
   : mesh(mesh_)
{
   const char *msg = "invalid input stream";
   string ident;

   in >> ident; MFEM_VERIFY(ident == "QuadratureSpace", msg);
   in >> ident; MFEM_VERIFY(ident == "Type:", msg);
   in >> ident;
   if (ident == "default_quadrature")
   {
      in >> ident; MFEM_VERIFY(ident == "Order:", msg);
      in >> order;
   }
   else
   {
      MFEM_ABORT("unknown QuadratureSpace type: " << ident);
      return;
   }

   Construct();
}

void QuadratureSpace::Save(std::ostream &out) const
{
   out << "QuadratureSpace\n"
       << "Type: default_quadrature\n"
       << "Order: " << order << '\n';
}


GridTransfer::GridTransfer(FiniteElementSpace &dom_fes_,
                           FiniteElementSpace &ran_fes_)
   : dom_fes(dom_fes_), ran_fes(ran_fes_),
     oper_type(Operator::ANY_TYPE),
     fw_t_oper(), bw_t_oper()
{
#ifdef MFEM_USE_MPI
   const bool par_dom = dynamic_cast<ParFiniteElementSpace*>(&dom_fes);
   const bool par_ran = dynamic_cast<ParFiniteElementSpace*>(&ran_fes);
   MFEM_VERIFY(par_dom == par_ran, "the domain and range FE spaces must both"
               " be either serial or parallel");
   parallel = par_dom;
#endif
}

const Operator &GridTransfer::MakeTrueOperator(
   FiniteElementSpace &fes_in, FiniteElementSpace &fes_out,
   const Operator &oper, OperatorHandle &t_oper)
{
   if (t_oper.Ptr())
   {
      return *t_oper.Ptr();
   }

   if (!Parallel())
   {
      const SparseMatrix *in_cP = fes_in.GetConformingProlongation();
      const SparseMatrix *out_cR = fes_out.GetConformingRestriction();
      if (oper_type == Operator::MFEM_SPARSEMAT)
      {
         const SparseMatrix *mat = dynamic_cast<const SparseMatrix *>(&oper);
         MFEM_VERIFY(mat != NULL, "Operator is not a SparseMatrix");
         if (!out_cR)
         {
            t_oper.Reset(const_cast<SparseMatrix*>(mat), false);
         }
         else
         {
            t_oper.Reset(mfem::Mult(*out_cR, *mat));
         }
         if (in_cP)
         {
            t_oper.Reset(mfem::Mult(*t_oper.As<SparseMatrix>(), *in_cP));
         }
      }
      else if (oper_type == Operator::ANY_TYPE)
      {
         const int RP_case = bool(out_cR) + 2*bool(in_cP);
         switch (RP_case)
         {
            case 0:
               t_oper.Reset(const_cast<Operator*>(&oper), false);
               break;
            case 1:
               t_oper.Reset(
                  new ProductOperator(out_cR, &oper, false, false));
               break;
            case 2:
               t_oper.Reset(
                  new ProductOperator(&oper, in_cP, false, false));
               break;
            case 3:
               t_oper.Reset(
                  new TripleProductOperator(
                     out_cR, &oper, in_cP, false, false, false));
               break;
         }
      }
      else
      {
         MFEM_ABORT("Operator::Type is not supported: " << oper_type);
      }
   }
   else // Parallel() == true
   {
#ifdef MFEM_USE_MPI
      if (oper_type == Operator::Hypre_ParCSR)
      {
         const SparseMatrix *out_R = fes_out.GetRestrictionMatrix();
         const ParFiniteElementSpace *pfes_in =
            dynamic_cast<const ParFiniteElementSpace *>(&fes_in);
         const ParFiniteElementSpace *pfes_out =
            dynamic_cast<const ParFiniteElementSpace *>(&fes_out);
         const SparseMatrix *sp_mat = dynamic_cast<const SparseMatrix *>(&oper);
         const HypreParMatrix *hy_mat;
         if (sp_mat)
         {
            SparseMatrix *RA = mfem::Mult(*out_R, *sp_mat);
            t_oper.Reset(pfes_in->Dof_TrueDof_Matrix()->
                         LeftDiagMult(*RA, pfes_out->GetTrueDofOffsets()));
            delete RA;
         }
         else if ((hy_mat = dynamic_cast<const HypreParMatrix *>(&oper)))
         {
            HypreParMatrix *RA =
               hy_mat->LeftDiagMult(*out_R, pfes_out->GetTrueDofOffsets());
            t_oper.Reset(mfem::ParMult(RA, pfes_in->Dof_TrueDof_Matrix()));
            delete RA;
         }
         else
         {
            MFEM_ABORT("unknown Operator type");
         }
      }
      else if (oper_type == Operator::ANY_TYPE)
      {
         const Operator *out_R = fes_out.GetRestrictionOperator();
         t_oper.Reset(new TripleProductOperator(
                         out_R, &oper, fes_in.GetProlongationMatrix(),
                         false, false, false));
      }
      else
      {
         MFEM_ABORT("Operator::Type is not supported: " << oper_type);
      }
#endif
   }

   return *t_oper.Ptr();
}


InterpolationGridTransfer::~InterpolationGridTransfer()
{
   if (own_mass_integ) { delete mass_integ; }
}

void InterpolationGridTransfer::SetMassIntegrator(
   BilinearFormIntegrator *mass_integ_, bool own_mass_integ_)
{
   if (own_mass_integ) { delete mass_integ; }

   mass_integ = mass_integ_;
   own_mass_integ = own_mass_integ_;
}

const Operator &InterpolationGridTransfer::ForwardOperator()
{
   if (F.Ptr())
   {
      return *F.Ptr();
   }

   // Construct F
   if (oper_type == Operator::ANY_TYPE)
   {
      F.Reset(new FiniteElementSpace::RefinementOperator(&ran_fes, &dom_fes));
   }
   else if (oper_type == Operator::MFEM_SPARSEMAT)
   {
      Mesh::GeometryList elem_geoms(*ran_fes.GetMesh());

      DenseTensor localP[Geometry::NumGeom];
      for (int i = 0; i < elem_geoms.Size(); i++)
      {
         ran_fes.GetLocalRefinementMatrices(dom_fes, elem_geoms[i],
                                            localP[elem_geoms[i]]);
      }
      F.Reset(ran_fes.RefinementMatrix_main(
                 dom_fes.GetNDofs(), dom_fes.GetElementToDofTable(),
                 dom_fes.GetElementToFaceOrientationTable(), localP));
   }
   else
   {
      MFEM_ABORT("Operator::Type is not supported: " << oper_type);
   }

   return *F.Ptr();
}

const Operator &InterpolationGridTransfer::BackwardOperator()
{
   if (B.Ptr())
   {
      return *B.Ptr();
   }

   // Construct B, if not set, define a suitable mass_integ
   if (!mass_integ && ran_fes.GetNE() > 0)
   {
      const FiniteElement *f_fe_0 = ran_fes.GetFE(0);
      const int map_type = f_fe_0->GetMapType();
      if (map_type == FiniteElement::VALUE ||
          map_type == FiniteElement::INTEGRAL)
      {
         mass_integ = new MassIntegrator;
      }
      else if (map_type == FiniteElement::H_DIV ||
               map_type == FiniteElement::H_CURL)
      {
         mass_integ = new VectorFEMassIntegrator;
      }
      else
      {
         MFEM_ABORT("unknown type of FE space");
      }
      own_mass_integ = true;
   }
   if (oper_type == Operator::ANY_TYPE)
   {
      B.Reset(new FiniteElementSpace::DerefinementOperator(
                 &ran_fes, &dom_fes, mass_integ));
   }
   else
   {
      MFEM_ABORT("Operator::Type is not supported: " << oper_type);
   }

   return *B.Ptr();
}


L2ProjectionGridTransfer::L2Projection::L2Projection(
   const FiniteElementSpace &fes_ho_, const FiniteElementSpace &fes_lor_)
   : Operator(fes_lor_.GetVSize(), fes_ho_.GetVSize()),
     fes_ho(fes_ho_),
     fes_lor(fes_lor_)
{
   Mesh *mesh_ho = fes_ho.GetMesh();
   Mesh *mesh_lor = fes_lor.GetMesh();
   int nel_ho = mesh_ho->GetNE();
   int nel_lor = mesh_lor->GetNE();

   // If the local mesh is empty, skip all computations
   if (nel_ho == 0) { return; }

   const CoarseFineTransformations &cf_tr = mesh_lor->GetRefinementTransforms();

   int nref_max = 0;
   Array<Geometry::Type> geoms;
   mesh_ho->GetGeometries(mesh_ho->Dimension(), geoms);
   for (int ig = 0; ig < geoms.Size(); ++ig)
   {
      Geometry::Type geom = geoms[ig];
      nref_max = std::max(nref_max, cf_tr.point_matrices[geom].SizeK());
   }

   // Construct the mapping from HO to LOR
   // ho2lor.GetRow(iho) will give all the LOR elements contained in iho
   ho2lor.MakeI(nel_ho);
   for (int ilor = 0; ilor < nel_lor; ++ilor)
   {
      int iho = cf_tr.embeddings[ilor].parent;
      ho2lor.AddAColumnInRow(iho);
   }
   ho2lor.MakeJ();
   for (int ilor = 0; ilor < nel_lor; ++ilor)
   {
      int iho = cf_tr.embeddings[ilor].parent;
      ho2lor.AddConnection(iho, ilor);
   }
   ho2lor.ShiftUpI();

   offsets.SetSize(nel_ho+1);
   offsets[0] = 0;
   for (int iho = 0; iho < nel_ho; ++iho)
   {
      int nref = ho2lor.RowSize(iho);
      const FiniteElement &fe_ho = *fes_ho.GetFE(iho);
      const FiniteElement &fe_lor = *fes_lor.GetFE(ho2lor.GetRow(iho)[0]);
      offsets[iho+1] = offsets[iho] + fe_ho.GetDof()*fe_lor.GetDof()*nref;
   }
   // R will contain the restriction (L^2 projection operator) defined on
   // each coarse HO element (and corresponding patch of LOR elements)
   R.SetSize(offsets[nel_ho]);
   // P will contain the corresponding prolongation operator
   P.SetSize(offsets[nel_ho]);

   IntegrationPointTransformation ip_tr;
   IsoparametricTransformation &emb_tr = ip_tr.Transf;

   for (int iho = 0; iho < nel_ho; ++iho)
   {
      Array<int> lor_els;
      int nref = ho2lor.RowSize(iho);
      ho2lor.GetRow(iho, lor_els);

      Geometry::Type geom = mesh_ho->GetElementBaseGeometry(iho);
      const FiniteElement &fe_ho = *fes_ho.GetFE(iho);
      const FiniteElement &fe_lor = *fes_lor.GetFE(lor_els[0]);
      int ndof_ho = fe_ho.GetDof();
      int ndof_lor = fe_lor.GetDof();

      Vector shape_ho(ndof_ho);
      Vector shape_lor(ndof_lor);

      emb_tr.SetIdentityTransformation(geom);
      const DenseTensor &pmats = cf_tr.point_matrices[geom];

      DenseMatrix R_iho(&R[offsets[iho]], ndof_lor*nref, ndof_ho);
      DenseMatrix P_iho(&P[offsets[iho]], ndof_ho, ndof_lor*nref);

      DenseMatrix Minv_lor(ndof_lor*nref, ndof_lor*nref);
      DenseMatrix M_mixed(ndof_lor*nref, ndof_ho);

      MassIntegrator mi;
      DenseMatrix M_lor_el(ndof_lor, ndof_lor);
      DenseMatrixInverse Minv_lor_el(&M_lor_el);
      DenseMatrix M_lor(ndof_lor*nref, ndof_lor*nref);
      DenseMatrix M_mixed_el(ndof_lor, ndof_ho);

      Minv_lor = 0.0;
      M_lor = 0.0;

      DenseMatrix RtMlor(ndof_ho, ndof_lor*nref);
      DenseMatrix RtMlorR(ndof_ho, ndof_ho);
      DenseMatrixInverse RtMlorR_inv(&RtMlorR);

      for (int iref = 0; iref < nref; ++iref)
      {
         // Assemble the low-order refined mass matrix and invert locally
         int ilor = ho2lor.GetRow(iho)[iref];
         ElementTransformation *el_tr = fes_lor.GetElementTransformation(ilor);
         mi.AssembleElementMatrix(fe_lor, *el_tr, M_lor_el);
         M_lor.CopyMN(M_lor_el, iref*ndof_lor, iref*ndof_lor);
         Minv_lor_el.Factor();
         Minv_lor_el.GetInverseMatrix(M_lor_el);
         // Insert into the diagonal of the patch LOR mass matrix
         Minv_lor.CopyMN(M_lor_el, iref*ndof_lor, iref*ndof_lor);

         // Now assemble the block-row of the mixed mass matrix associated
         // with integrating HO functions against LOR functions on the LOR
         // sub-element.

         // Create the transformation that embeds the fine low-order element
         // within the coarse high-order element in reference space
         emb_tr.SetPointMat(pmats(cf_tr.embeddings[ilor].matrix));

         int order = fe_lor.GetOrder() + fe_ho.GetOrder() + el_tr->OrderW();
         const IntegrationRule *ir = &IntRules.Get(geom, order);
         M_mixed_el = 0.0;
         for (int i = 0; i < ir->GetNPoints(); i++)
         {
            const IntegrationPoint &ip_lor = ir->IntPoint(i);
            IntegrationPoint ip_ho;
            ip_tr.Transform(ip_lor, ip_ho);
            fe_lor.CalcShape(ip_lor, shape_lor);
            fe_ho.CalcShape(ip_ho, shape_ho);
            el_tr->SetIntPoint(&ip_lor);
            // For now we use the geometry information from the LOR space
            // which means we won't be mass conservative if the mesh is curved
            double w = el_tr->Weight()*ip_lor.weight;
            shape_lor *= w;
            AddMultVWt(shape_lor, shape_ho, M_mixed_el);
         }
         M_mixed.CopyMN(M_mixed_el, iref*ndof_lor, 0);
      }
      mfem::Mult(Minv_lor, M_mixed, R_iho);

      mfem::MultAtB(R_iho, M_lor, RtMlor);
      mfem::Mult(RtMlor, R_iho, RtMlorR);
      RtMlorR_inv.Factor();
      RtMlorR_inv.Mult(RtMlor, P_iho);
   }
}

void L2ProjectionGridTransfer::L2Projection::Mult(
   const Vector &x, Vector &y) const
{
   int vdim = fes_ho.GetVDim();
   Array<int> vdofs;
   DenseMatrix xel_mat, yel_mat;
   for (int iho = 0; iho < fes_ho.GetNE(); ++iho)
   {
      int nref = ho2lor.RowSize(iho);
      int ndof_ho = fes_ho.GetFE(iho)->GetDof();
      int ndof_lor = fes_lor.GetFE(ho2lor.GetRow(iho)[0])->GetDof();
      xel_mat.SetSize(ndof_ho, vdim);
      yel_mat.SetSize(ndof_lor*nref, vdim);
      DenseMatrix R_iho(&R[offsets[iho]], ndof_lor*nref, ndof_ho);

      fes_ho.GetElementVDofs(iho, vdofs);
      x.GetSubVector(vdofs, xel_mat.GetData());
      mfem::Mult(R_iho, xel_mat, yel_mat);
      // Place result correctly into the low-order vector
      for (int iref = 0; iref < nref; ++iref)
      {
         int ilor = ho2lor.GetRow(iho)[iref];
         for (int vd=0; vd<vdim; ++vd)
         {
            fes_lor.GetElementDofs(ilor, vdofs);
            fes_lor.DofsToVDofs(vd, vdofs);
            y.SetSubVector(vdofs, &yel_mat(iref*ndof_lor,vd));
         }
      }
   }
}

void L2ProjectionGridTransfer::L2Projection::Prolongate(
   const Vector &x, Vector &y) const
{
   int vdim = fes_ho.GetVDim();
   Array<int> vdofs;
   DenseMatrix xel_mat,yel_mat;
   for (int iho = 0; iho < fes_ho.GetNE(); ++iho)
   {
      int nref = ho2lor.RowSize(iho);
      int ndof_ho = fes_ho.GetFE(iho)->GetDof();
      int ndof_lor = fes_lor.GetFE(ho2lor.GetRow(iho)[0])->GetDof();
      xel_mat.SetSize(ndof_lor*nref, vdim);
      yel_mat.SetSize(ndof_ho, vdim);
      DenseMatrix P_iho(&P[offsets[iho]], ndof_ho, ndof_lor*nref);

      // Extract the LOR DOFs
      for (int iref = 0; iref < nref; ++iref)
      {
         int ilor = ho2lor.GetRow(iho)[iref];
         for (int vd = 0; vd < vdim; ++vd)
         {
            fes_lor.GetElementDofs(ilor, vdofs);
            fes_lor.DofsToVDofs(vd, vdofs);
            x.GetSubVector(vdofs, &xel_mat(iref*ndof_lor, vd));
         }
      }
      // Locally prolongate
      mfem::Mult(P_iho, xel_mat, yel_mat);
      // Place the result in the HO vector
      fes_ho.GetElementVDofs(iho, vdofs);
      y.SetSubVector(vdofs, yel_mat.GetData());
   }
}

const Operator &L2ProjectionGridTransfer::ForwardOperator()
{
   if (!F) { F = new L2Projection(dom_fes, ran_fes); }
   return *F;
}

const Operator &L2ProjectionGridTransfer::BackwardOperator()
{
   if (!B)
   {
      if (!F) { F = new L2Projection(dom_fes, ran_fes); }
      B = new L2Prolongation(*F);
   }
   return *B;
}

} // namespace mfem
