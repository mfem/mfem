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

// Implementation of FiniteElementSpace

#include "../general/text.hpp"
#include "../general/forall.hpp"
#include "../mesh/mesh_headers.hpp"
#include "fem.hpp"

#include <cmath>
#include <cstdarg>
#include <climits>

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
     ndofs(0), nvdofs(0), nedofs(0), nfdofs(0), nbdofs(0), bdofs(NULL),
     elem_dof(NULL), bdr_elem_dof(NULL),
     NURBSext(NULL), own_ext(false),
     cP(NULL), cR(NULL), cQ(NULL), cP_is_set(false),
     Th(Operator::ANY_TYPE),
     sequence(0), orders_changed(false), relaxed_hp(false)
{ }

FiniteElementSpace::FiniteElementSpace(const FiniteElementSpace &orig,
                                       Mesh *mesh,
                                       const FiniteElementCollection *fec)
   : relaxed_hp(orig.relaxed_hp)
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
   MFEM_VERIFY(sequence == mesh->GetSequence(), "space has not been Updated()");
   MFEM_VERIFY(i >= 0 && i < GetNE(), "invalid element index");
   MFEM_VERIFY(p >= 0 && p < CHAR_MAX, "order ouf of range");
   MFEM_ASSERT(!elem_order.Size() || elem_order.Size() == GetNE(), "internal error");

   if (elem_order.Size())
   {
      if (elem_order[i] != p)
      {
         elem_order[i] = p;
         orders_changed = true;
      }
   }
   else
   {
      elem_order.SetSize(GetNE());
      elem_order = fec->DefaultOrder();

      elem_order[i] = p;
      orders_changed = true;
   }
}

int FiniteElementSpace::GetElementOrder(int i) const
{
   MFEM_VERIFY(sequence == mesh->GetSequence(), "space has not been Updated()");
   MFEM_VERIFY(i >= 0 && i < GetNE(), "invalid element index");
   MFEM_ASSERT(!elem_order.Size() || elem_order.Size() == GetNE(), "internal error");

   return elem_order.Size() ? elem_order[i] : fec->DefaultOrder();
}

int FiniteElementSpace::GetFaceOrder(int i) const
{
   Geometry::Type GeomType = mesh->GetFaceBaseGeometry(i);
   return fec->FiniteElementForGeometry(GeomType)->GetOrder();
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

void FiniteElementSpace::GetElementVDofs(int i, Array<int> &vdofs) const
{
   GetElementDofs(i, vdofs);
   DofsToVDofs(vdofs);
}

void FiniteElementSpace::GetBdrElementVDofs(int i, Array<int> &vdofs) const
{
   GetBdrElementDofs(i, vdofs);
   DofsToVDofs(vdofs);
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

void FiniteElementSpace::BuildElementToDofTable()
{
   if (elem_dof) { return; }

   // TODO: can we call GetElementDofs only once per element?
   Table *el_dof = new Table;
   Array<int> dofs;
   el_dof -> MakeI (mesh -> GetNE());
   for (int i = 0; i < mesh -> GetNE(); i++)
   {
      GetElementDofs (i, dofs);
      el_dof -> AddColumnsInRow (i, dofs.Size());
   }
   el_dof -> MakeJ();
   for (int i = 0; i < mesh -> GetNE(); i++)
   {
      GetElementDofs (i, dofs);
      el_dof -> AddConnections (i, (int *)dofs, dofs.Size());
   }
   el_dof -> ShiftUpI();
   elem_dof = el_dof;
}

void FiniteElementSpace::RebuildElementToDofTable()
{
   delete elem_dof;
   elem_dof = NULL;
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
   marker.SetSize(marker_size);
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

const int SkipDof = std::numeric_limits<int>::max();

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

void FiniteElementSpace::GetDegenerateFaceDofs(int index, Array<int> &dofs,
                                               Geometry::Type master_geom) const
{
   // In NC meshes with prisms/tets, a special constraint occurs where a
   // prism/tet edge is slave to another element's face. Rather than introduce a
   // new edge-face constraint type, we handle such cases as degenerate
   // face-face constraints, where the point-matrix rectangle has zero height.
   // This method returns DOFs for the first edge of the rectangle, duplicated
   // in the orthogonal direction, to resemble DOFs for a quadrilateral face.
   // The extra DOFs are ignored by FiniteElementSpace::AddDependencies.

   Array<int> edof;
   GetEdgeDofs(-1 - index, edof);

   int nv = fec->DofForGeometry(Geometry::POINT);
   int ne = fec->DofForGeometry(Geometry::SEGMENT);
   int nn = 2*nv + ne;

   dofs.SetSize(nn*nn);
   if (!dofs.Size()) { return; }

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
}

#if 0
void FiniteElementSpace::MaskSlaveDofs(Array<int> &dofs, const DenseMatrix &pm,
                                       int order) const
{
   MFEM_ASSERT(pm.Height() == 2 && pm.Width() == 4, "");
   double x0 = pm(0,0), x1 = pm(0,2);
   double y0 = pm(1,0), y1 = pm(1,2);

   bool bdr[4] =
   {
      (y0 == 0.0 || y0 == 1.0),
      (x1 == 0.0 || x1 == 1.0),
      (y1 == 0.0 || y1 == 1.0),
      (x0 == 0.0 || x0 == 1.0)
   };

   int nv = fec->GetNumDof(Geometry::POINT, order);
   int ne = fec->GetNumDof(Geometry::SEGMENT, order);

   for (int i = 0, prev = 3; i < 4; prev = i, i++)
   {
      if (bdr[i] || bdr[prev])
      {
         for (int j = 0; j < nv; j++)
         {
            dofs[i*nv + j] = SkipDof; // mask out vertex DOFs on boundary
         }
      }
   }

   for (int i = 0; i < 4; i++)
   {
      if (bdr[i])
      {
         for (int j = 0; j < ne; j++)
         {
            dofs[4*nv + i*ne + j] = SkipDof; // mask out edge DOFs on boundary
         }
      }
   }
}
#endif

int FiniteElementSpace::GetEntityDofs(int entity, int index, Array<int> &dofs,
                                  Geometry::Type master_geom, int variant) const
{
   switch (entity)
   {
      case 0: GetVertexDofs(index, dofs); return 0;
      case 1: return GetEdgeDofs(index, dofs, variant);
      case 2: if (index >= 0) { return GetFaceDofs(index, dofs, variant); }
              else MFEM_ABORT("FIXME"); return 0;
//         /*             */ : GetDegenerateFaceDofs(index, dofs, master_geom); // FIXME - variant

   }
   return 0;
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

   // For the restriction matrix purposes in variable order space:
   // For each edge with more DOF sets, the dependency matrix will contain
   // a row that expresses the master true DOF (lowest order)
   // as a linear combination of the highest order set of DOFs.
   SparseMatrix sped(ndofs);

   // collect local face/edge dependencies
   for (int entity = 2; entity >= 1; entity--)
   {
      const NCMesh::NCList &list = mesh->ncmesh->GetNCList(entity);
      if (!list.masters.size()) { continue; }

      // loop through all master edges/faces, constrain their slave edges/faces
      for (const NCMesh::Master &master : list.masters)
      {
         Geometry::Type master_geom = master.Geom();

         int p = GetEntityDofs(entity, master.index, master_dofs, master_geom, 0);
         if (!master_dofs.Size()) { continue; }

         const auto *master_fe = fec->GetFE(master_geom, p);
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

            int q = GetEntityDofs(entity, slave.index, slave_dofs, master_geom, 0);
            if (!slave_dofs.Size()) { break; }

            const auto *slave_fe = fec->GetFE(slave.Geom(), q);
            slave.OrientedPointMatrix(T.GetPointMat());
            slave_fe->GetTransferMatrix(*master_fe, T, I);

            // variable order spaces: face edges need to be handled separately
            int skipfirst = 0;
            if (IsVariableOrder() && entity == 2)
            {
               int nv = fec->GetNumDof(Geometry::POINT, q);
               int ne = fec->GetNumDof(Geometry::SEGMENT, q);
               skipfirst = Geometry::NumVerts[master_geom] * (nv + ne);
            }

            // make each slave DOF dependent on all master DOFs
            AddDependencies(deps, master_dofs, slave_dofs, I, skipfirst);

            // handle edge DOFs if they were skipped
            if (skipfirst)
            {
               Array<int> V, E, Eo;
               mesh->GetFaceVertices(slave.index, V);
               mesh->GetFaceEdges(slave.index, E, Eo);
               MFEM_ASSERT(V.Size() == E.Size(), "");

               IsoparametricTransformation eT;
               eT.SetFE(&SegmentFE);

               const DenseMatrix &pm = slave.point_matrix;

               // constrain each edge of the slave face
               for (int i = 0; i < E.Size(); i++)
               {
                  int a = i, b = (i+1) % V.Size();
                  if (V[a] > V[b]) { std::swap(a, b); }

                  DenseMatrix &edge_pm = eT.GetPointMat();
                  edge_pm.SetSize(2, 2);

                  // copy two points from the face point matrix
                  double mid[2];
                  for (int j = 0; j < 2; j++)
                  {
                     edge_pm(j, 0) = pm(j, a);
                     edge_pm(j, 1) = pm(j, b);
                     mid[j] = 0.5*(pm(j, a) + pm(j, b));
                  }

                  // check that the edge does not coincide with master edge
                  const double eps = 1e-14;
                  if (mid[0] > eps && mid[0] < 1-eps &&
                      mid[1] > eps && mid[1] < 1-eps)
                  {
                     int order = GetEdgeDofs(E[i], slave_dofs, 0);

                     const auto *edge_fe = fec->GetFE(Geometry::SEGMENT, order);
                     edge_fe->GetTransferMatrix(*master_fe, eT, I);

                     AddDependencies(deps, master_dofs, slave_dofs, I, 0);
                  }
               }
            }
         }

         // find dependencies for the cQ matrix; if edge/face has more sets of
         // DOFs, the lowest order (true) set interpolate the highest order set
         if (IsVariableOrder())
         {
            int variant = 0;
            for (int var = 1; ; var++)
            {
               int o = GetEntityDofs(entity, master.index, highest_dofs, master_geom, var);
               if (o == -1)
               {
                  variant = var-1;
                  break;
               }
            }
            if (variant > 0)
            {
               int q = GetEntityDofs(entity, master.index, highest_dofs, master_geom, variant);
               const auto *highest_fe = fec->GetFE(master_geom, q);
               T.SetIdentityTransformation(master_geom);
               master_fe->GetTransferMatrix(*highest_fe, T, I);
               // Add dependencies only for the edge inner dofs
               AddDependencies(sped, highest_dofs, master_dofs, I, 2);
            }
         }
      }
   }

   // variable order spaces: enforce minimum rule on conforming edges/faces
   if (IsVariableOrder())
   {
      for (int entity = 1; entity < mesh->Dimension(); entity++)
      {
         const Table &ent_dofs = (entity == 1) ? edge_dofs : face_dofs;
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
   sped.Finalize();

   // DOFs that stayed independent are true DOFs
   int n_true_dofs = 0;
   for (int i = 0; i < ndofs; i++)
   {
      if (!deps.RowSize(i)) { n_true_dofs++; }
   }

   // if all dofs are true dofs leave cP and cR NULL
   if (n_true_dofs == ndofs)
   {
      cP = cR = cQ = NULL; // will be treated as identities
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

   // in var. order spaces, create the restriction matrix cQ which is similar to
   // cR, but has interpolation in the extra master edge/face DOFs
   cQ = IsVariableOrder() ? new SparseMatrix(n_true_dofs, ndofs) : NULL;

   Array<bool> finalized(ndofs);
   finalized = false;

   Array<int> cols;
   Vector srow;

   // put identity in the prolongation matrix for true DOFs, initialize cQ
   for (int i = 0, true_dof = 0; i < ndofs; i++)
   {
      if (!deps.RowSize(i))
      {
         // true dof
         cP->Add(i, true_dof, 1.0);
         cR_J[true_dof] = i;
         finalized[i] = true;

         if (cQ)
         {
            if (sped.RowSize(i))
            {
               sped.GetRow(i, cols, srow);
               cQ->AddRow(true_dof, cols, srow);
            }
            else
            {
               cQ->Add(true_dof, i, 1.0);
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

   // if everything is consistent (mesh, face orientations, etc.), we should
   // be able to finalize all slave DOFs, otherwise it's a serious error
   if (n_finalized != ndofs)
   {
      //MFEM_ABORT("Error creating cP matrix.");
      MFEM_WARNING("Error creating cP matrix: n_finalized = " << n_finalized
                   << ", ndofs = " << ndofs);
   }

   /*out << "\n\n";
   DebugDumpDOFs(out, deps, finalized);*/

   cP->Finalize();
   if (cQ) { cQ->Finalize(); }

   if (vdim > 1)
   {
      MakeVDimMatrix(*cP);
      MakeVDimMatrix(*cR);
      MakeVDimMatrix(*cQ);
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

const SparseMatrix* FiniteElementSpace::GetConformingRestrictionInterpolation() const
{
   if (Conforming()) { return NULL; }
   if (!IsVariableOrder()) { return NULL; }
   if (!cP_is_set) { BuildConformingInterpolation(); }
   return cQ;
}

int FiniteElementSpace::GetNConformingDofs() const
{
   const SparseMatrix* P = GetConformingProlongation();
   return P ? (P->Width() / vdim) : ndofs;
}

const Operator *FiniteElementSpace::GetElementRestriction(
   ElementDofOrdering e_ordering) const
{
#if 0 // FIXME
   // Check if we have a discontinuous space using the FE collection:
   if (IsDGSpace())
   {
      if (L2E_nat.Ptr() == NULL)
      {
         L2E_nat.Reset(new L2ElementRestriction(*this));
      }
      return L2E_nat.Ptr();
   }
#endif
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
   const DenseTensor localP[]) const
{
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
                                                   const Table* old_elem_dof)
{
   MFEM_VERIFY(GetNE() >= old_elem_dof->Size(),
               "Previous mesh is not coarser.");

   Mesh::GeometryList elem_geoms(*mesh);

   DenseTensor localP[Geometry::NumGeom];
   for (int i = 0; i < elem_geoms.Size(); i++)
   {
      GetLocalRefinementMatrices(elem_geoms[i], localP[elem_geoms[i]]);
   }

   return RefinementMatrix_main(old_ndofs, *old_elem_dof, localP);
}

FiniteElementSpace::RefinementOperator::RefinementOperator
(const FiniteElementSpace* fespace, Table* old_elem_dof, int old_ndofs)
   : fespace(fespace)
   , old_elem_dof(old_elem_dof)
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
}

FiniteElementSpace::RefinementOperator::RefinementOperator(
   const FiniteElementSpace *fespace, const FiniteElementSpace *coarse_fes)
   : Operator(fespace->GetVSize(), coarse_fes->GetVSize()),
     fespace(fespace), old_elem_dof(NULL)
{
   Mesh::GeometryList elem_geoms(*fespace->GetMesh());

   for (int i = 0; i < elem_geoms.Size(); i++)
   {
      fespace->GetLocalRefinementMatrices(*coarse_fes, elem_geoms[i],
                                          localP[elem_geoms[i]]);
   }

   // Make a copy of the coarse elem_dof Table.
   old_elem_dof = new Table(coarse_fes->GetElementToDofTable());
}

FiniteElementSpace::RefinementOperator::~RefinementOperator()
{
   delete old_elem_dof;
}

void FiniteElementSpace::RefinementOperator
::Mult(const Vector &x, Vector &y) const
{
   Mesh* mesh = fespace->GetMesh();
   const CoarseFineTransformations &rtrans = mesh->GetRefinementTransforms();

   Array<int> dofs, vdofs, old_dofs, old_vdofs;

   int vdim = fespace->GetVDim();
   int old_ndofs = width / vdim;

   Vector subY, subX;

   for (int k = 0; k < mesh->GetNE(); k++)
   {
      const Embedding &emb = rtrans.embeddings[k];
      const Geometry::Type geom = mesh->GetElementBaseGeometry(k);
      const DenseMatrix &lP = localP[geom](emb.matrix);

      subY.SetSize(lP.Height());

      fespace->GetElementDofs(k, dofs);
      old_elem_dof->GetRow(emb.parent, old_dofs);

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
}

void FiniteElementSpace::RefinementOperator
::MultTranspose(const Vector &x, Vector &y) const
{
   y = 0.0;

   Mesh* mesh = fespace->GetMesh();
   const CoarseFineTransformations &rtrans = mesh->GetRefinementTransforms();

   Array<char> processed(fespace->GetVSize());
   processed = 0;

   Array<int> f_dofs, c_dofs, f_vdofs, c_vdofs;

   int vdim = fespace->GetVDim();
   int old_ndofs = width / vdim;

   Vector subY, subX;

   for (int k = 0; k < mesh->GetNE(); k++)
   {
      const Embedding &emb = rtrans.embeddings[k];
      const Geometry::Type geom = mesh->GetElementBaseGeometry(k);
      const DenseMatrix &lP = localP[geom](emb.matrix);

      fespace->GetElementDofs(k, f_dofs);
      old_elem_dof->GetRow(emb.parent, c_dofs);

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

      for (int p = 0; p < f_dofs.Size(); ++p)
      {
         processed[DecodeDof(f_dofs[p])] = 1;
      }
   }
}

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
                                                     const Table* old_elem_dof)
{
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
   sequence = mesh->GetSequence();
   Th.SetType(Operator::ANY_TYPE);

   /*const NURBSFECollection *nurbs_fec =
      dynamic_cast<const NURBSFECollection *>(fec);
   if (nurbs_fec)*/ if (0)
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
      cP = cR = cQ = NULL;
      cP_is_set = false;
   }
   else
   {
      this->NURBSext = NULL;
      own_ext = 0;
      Construct();
   }
   BuildElementToDofTable();
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
   nvdofs = 0;
   nedofs = 0;
   nfdofs = 0;
   nbdofs = 0;
   bdofs = NULL;

   //dynamic_cast<const NURBSFECollection *>(fec)->Reset();

   ndofs = NURBSext->GetNDof();
   elem_dof = NURBSext->GetElementDofTable();
   bdr_elem_dof = NURBSext->GetBdrElementDofTable();
}

void FiniteElementSpace::DumpOrders(const Table &ent_dofs, int dim)
{
   Array<int> V;
   for (int i = 0; i < ent_dofs.Size()-1; i++)
   {
      out << (dim == 1 ? "edge " : "face ") << i;
      if (dim == 2)
      {
         mesh->GetFaceVertices(i, V);
         out << " (";
         for (int v : V) { out << v << " "; }
         out << ")";
      }
      out << ": order";

      const int *beg = ent_dofs.GetRow(i);
      const int *end = ent_dofs.GetRow(i+1);

      const int *dof = beg;
      while (dof < end)
      {
         int order = (dim == 1) ? (dof[1] - dof[0] + 1)
                                : (int(sqrt(dof[1] - dof[0])) + 1);
         out << " " << order;
         dof++;
      }

      dof = beg;
      while (dof < end)
      {
         out << ", dofs";
         int base = (dim == 1) ? nvdofs : nvdofs + nedofs;
         for (int j = dof[0]; j < dof[1]; j++)
         {
            out << " " << j + base;
         }
         dof++;
      }
      out << std::endl;
   }
}

static void
Dof2Ent(const Table &table, int dof, int &index, int &variant, int &edof)
{
   for (int i = 0; i < table.Size(); i++) // FIXME O(N)
   {
      const int *e = table.GetRow(i+1);
      if (dof >= *e) { continue; }
      const int *p = table.GetRow(i);

      for (variant = 0; p < e; p++, variant++)
      {
         if (dof >= p[0] && dof < p[1])
         {
            index = i;
            edof = dof - p[0];
            return;
         }
      }
   }
   MFEM_ABORT("internal error");
}

void FiniteElementSpace::UnpackDof(int dof, int &entity, int &index,
                                   int &variant, int &edof) const
{
   MFEM_VERIFY(dof >= 0, "");
   MFEM_VERIFY(dof < ndofs, "");

   if (dof < nvdofs) // vertex DOF
   {
      int nv = fec->GetNumDof(Geometry::POINT, fec->DefaultOrder());
      entity = 0, index = dof / nv, edof = dof % nv, variant = 0;
      return;
   }
   dof -= nvdofs;

   if (dof < nedofs) // edge DOF
   {
      entity = 1;
      if (edge_dofs.Size() > 0)
      {
         Dof2Ent(edge_dofs, dof, index, variant, edof);
      }
      else
      {
         int ne = fec->GetNumDof(Geometry::SEGMENT, fec->DefaultOrder());
         index = dof / ne, edof = dof % ne, variant = 0;
      }
      return;
   }
   dof -= nedofs;

   if (dof < nfdofs) // face DOF
   {
      entity = 2;
      if (face_dofs.Size() > 0)
      {
         Dof2Ent(face_dofs, dof, index, variant, edof);
      }
      else
      {
         int nf = fec->GetNumDof(mesh->GetFaceGeometry(0), fec->DefaultOrder());
         index = dof / nf, edof = dof % nf, variant = 0;
      }
      return;
   }
   MFEM_ABORT("Cannot unpack internal DOF");
}

void FiniteElementSpace
::DebugDumpDOFs(std::ostream &os, const SparseMatrix &deps,
                const Array<bool> &finalized) const
{
   for (int i = 0; i < deps.Size(); i++)
   {
      os << i << ": ";
      if (i < (nvdofs + nedofs + nfdofs))
      {
         int ent, idx, variant, edof;
         UnpackDof(i, ent, idx, variant, edof);

         os << edof << " @ ";
         switch (ent)
         {
            case 0: os << "vertex "; break;
            case 1: os << "edge "; break;
            default: os << "face "; break;
         }
         os << idx << " (v" << variant << "); ";

         if (i < deps.Height() && deps.RowSize(i))
         {
            os << "depends on ";
            for (int j = 0; j < deps.RowSize(i); j++)
            {
               int master = deps.GetRowColumns(i)[j];
               os << master << (finalized[master] ? "" : "!");
               os << " (" << deps.GetRowEntries(i)[j] << ")";
               if (j < deps.RowSize(i)-1) { os << ", "; }
            }
            os << "; ";
         }
         else
         {
            os << "no deps; ";
         }

         os << (finalized[i] ? "finalized" : "NOT finalized");
      }
      else
      {
         os << "internal";
      }
      os << "\n";
   }
}

void FiniteElementSpace::Construct()
{
   // This method should be used only for non-NURBS spaces.
   MFEM_VERIFY(!NURBSext, "internal error");

   elem_dof = NULL;
   bdr_elem_dof = NULL;

   ndofs = 0;
   nvdofs = nedofs = nfdofs = nbdofs = 0;
   bdofs = NULL;

   cP = NULL;
   cR = NULL;
   cQ = NULL;
   cP_is_set = false;
   // 'Th' is initialized/destroyed before this method is called.

   int order = fec->DefaultOrder();

   // for full conforming hp spaces, calculate orders of edges and faces
   Array<int> edge_min_order, face_min_order;
   if (IsVariableOrder() && !relaxed_hp)
   {
      CalculateMinimumOrders(edge_min_order, face_min_order);
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
         nedofs = AssignVarOrderDofs(1, edge_min_order, edge_dofs);
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
      if (IsVariableOrder() || mesh->GetNumGeometries(2) > 1)
      {
         // note that we use Table face_dofs for mixed faces as well
         nfdofs = AssignVarOrderDofs(2, face_min_order, face_dofs);
      }
      else
      {
         // the simple case: all faces are of the same geometry and order
         Geometry::Type geom = mesh->GetFaceGeometry(0);
         nfdofs = mesh->GetNFaces() * fec->GetNumDof(geom, order);
      }
   }

   // assign internal ("bubble") DOFs
   if (mesh->GetNE())
   {
      if (IsVariableOrder() || mesh->GetNumGeometries(mesh->Dimension()) > 1)
      {
         bdofs = new int[mesh->GetNE()+1];
         bdofs[0] = 0;
         for (int i = 0; i < mesh->GetNE(); i++)
         {
            int p = IsVariableOrder() ? elem_order[i] : order;
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

   // DOFs are now assigned according to current element orders
   orders_changed = false;

   /*DumpOrders(edge_dofs, 1);
   DumpOrders(face_dofs, 2);*/

   // Do not build elem_dof Table here: in parallel it has to be constructed
   // later.
}

void FiniteElementSpace::CalculateMinimumOrders(Array<int> &edge_min_order,
                                                Array<int> &face_min_order) const
{
   MFEM_ASSERT(IsVariableOrder(), "");
   MFEM_ASSERT(elem_order.Size() == mesh->GetNE(), "");

   edge_min_order.SetSize(mesh->GetNEdges());
   face_min_order.SetSize(mesh->GetNFaces());

   edge_min_order = INT_MAX;
   face_min_order = INT_MAX;

   // initial edge/face order is the minimum of incident element orders
   Array<int> E, F, ori;
   for (int i = 0; i < mesh->GetNE(); i++)
   {
      int order = elem_order[i];

      mesh->GetElementEdges(i, E, ori);
      for (int j = 0; j < E.Size(); j++)
      {
         int &emo = edge_min_order[E[j]];
         emo = std::min(emo, order);
      }

      if (mesh->Dimension() > 2)
      {
         mesh->GetElementFaces(i, F, ori);
         for (int j = 0; j < F.Size(); j++)
         {
            int &fmo = face_min_order[F[j]];
            fmo = std::min(fmo, order);
         }
      }
   }

   // iterate while minimum orders propagate by master/slave relations
   bool done;
   do
   {
      done = true;

      // propagate from slave edges to master edges
      const NCMesh::NCList &edge_list = mesh->ncmesh->GetEdgeList();
      for (const NCMesh::Master &master : edge_list.masters)
      {
         int min_order = INT_MAX;
         for (int i = master.slaves_begin; i < master.slaves_end; i++)
         {
            const NCMesh::Slave &slave = edge_list.slaves[i];
            min_order = std::min(edge_min_order[slave.index], min_order);
         }

         if (min_order < edge_min_order[master.index])
         {
            edge_min_order[master.index] = min_order;
            done = false;
         }
      }

      // propagate from slave faces(+edges) to master faces(+edges)
      const NCMesh::NCList &face_list = mesh->ncmesh->GetFaceList();
      for (const NCMesh::Master &master : face_list.masters)
      {
         int min_order = INT_MAX;
         for (int i = master.slaves_begin; i < master.slaves_end; i++)
         {
            const NCMesh::Slave &slave = face_list.slaves[i];
            min_order = std::min(face_min_order[slave.index], min_order);

            mesh->GetFaceEdges(slave.index, E, ori);
            for (int j = 0; j < E.Size(); j++)
            {
               min_order = std::min(edge_min_order[E[j]], min_order);
            }
         }

         if (min_order < face_min_order[master.index])
         {
            face_min_order[master.index] = min_order;
            done = false;
         }
         mesh->GetFaceEdges(master.index, E, ori);
         for (int j = 0; j < E.Size(); j++)
         {
            if (min_order < edge_min_order[E[j]])
            {
               edge_min_order[E[j]] = min_order;
               done = false;
            }
         }
      }
   }
   while (!done);
}

int FiniteElementSpace::AssignVarOrderDofs(int ent_dim,
                                           const Array<int> &ent_min_order,
                                           Table &entity_dofs)
{
   Array<Connection> ent_orders;
   ent_orders.Reserve(12*mesh->GetNE() + ent_min_order.Size());

   // get a list of orders for each edge (face)
   Array<int> E, ori;
   for (int i = 0; i < mesh->GetNE(); i++)
   {
      int order = IsVariableOrder() ? elem_order[i] : fec->DefaultOrder();

      (ent_dim == 1) ? mesh->GetElementEdges(i, E, ori)
      /**/           : mesh->GetElementFaces(i, E, ori);

      for (int j = 0; j < E.Size(); j++)
      {
         ent_orders.Append(Connection(E[j], order));
      }
   }
   for (int i = 0; i < ent_min_order.Size(); i++)
   {
      ent_orders.Append(Connection(i, ent_min_order[i]));
   }

   // For each edge (face) we now have a list of polynomial orders in which it
   // needs to be represented. We assign the DOF numbers and fill the table
   // edge_dofs (face_dofs), where each row contains the indices of the
   // beginnings of the DOF ranges for the different polynomial orders.

   ent_orders.Sort();
   ent_orders.Unique();

   // assign DOFs according to orders
   int total_dofs = 0;
   for (int i = 0; i < ent_orders.Size(); i++)
   {
      auto geom = (ent_dim == 1) ? Geometry::SEGMENT
                  : mesh->GetFaceGeometry(ent_orders[i].from);

      int order = ent_orders[i].to;
      int dofs = fec->GetNumDof(geom, order);

      ent_orders[i].to = total_dofs;
      total_dofs += dofs;
   }

   // append a dummy row as terminator
   int num_ent = (ent_dim == 1) ? mesh->GetNEdges() : mesh->GetNFaces();
   ent_orders.Append(Connection(num_ent, total_dofs));

   // build the table
   entity_dofs.MakeFromList(num_ent+1, ent_orders);

   return total_dofs;
}


int FiniteElementSpace::FindDofs(const Table &dof_table, int row, int ndof) const
{
   const int *beg = dof_table.GetRow(row);
   const int *end = dof_table.GetRow(row + 1); // terminator, see above

   while (beg < end)
   {
      // return the appropriate range of DOFs
      if ((beg[1] - beg[0]) == ndof) { return beg[0]; }
      beg++;
   }

   MFEM_ABORT("DOFs not found for ndof = " << ndof);
   return 0;
}


void FiniteElementSpace::GetElementDofs(int elem, Array<int> &dofs) const
{
   //if (orders_changed) { CheckUpToDate(); }

   if (elem_dof)
   {
      elem_dof->GetRow(elem, dofs);
      return;
   }

   Array<int> V, E, Eo, F, Fo; // TODO: LocalArray

   int dim = mesh->Dimension();
   Geometry::Type geom = mesh->GetElementGeometry(elem);
   int order = IsVariableOrder() ? elem_order[elem] : fec->DefaultOrder();

   int nv = fec->GetNumDof(Geometry::POINT, order);
   int ne = (dim > 1) ? fec->GetNumDof(Geometry::SEGMENT, order) : 0;
   int nb = (dim > 0) ? fec->GetNumDof(geom, order) : 0;

   if (nv)
   {
      mesh->GetElementVertices(elem, V);
   }

   if (ne)
   {
      mesh->GetElementEdges(elem, E, Eo);
   }

   int nfd = 0;
   if (dim > 2 && fec->HasFaceDofs(geom, order))
   {
      mesh->GetElementFaces(elem, F, Fo);
      for (int i = 0; i < F.Size(); i++)
      {
         nfd += fec->GetNumDof(mesh->GetFaceGeometry(F[i]), order);
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

         int fbase = (face_dofs.Size() > 0) ? FindFaceDof(F[i], nf) : F[i]*nf;
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
}

const FiniteElement *FiniteElementSpace::GetFE(int i) const
{
   if (i < 0 || !mesh->GetNE()) { return NULL; }
   MFEM_VERIFY(i < mesh->GetNE(),
               "Invalid element id " << i << ", maximum allowed " << mesh->GetNE()-1);

   int order = IsVariableOrder() ? elem_order[i] : fec->DefaultOrder();

   const FiniteElement *FE =
      fec->GetFE(mesh->GetElementGeometry(i), order);

   if (NURBSext)
   {
      NURBSext->LoadFE(i, FE);
   }

   return FE;
}

void FiniteElementSpace::GetBdrElementDofs(int bel, Array<int> &dofs) const
{
   if (bdr_elem_dof)
   {
      bdr_elem_dof->GetRow(bel, dofs);
      return;
   }

   Array<int> V, E, Eo; // TODO: LocalArray
   int F, Fo;

   int dim = mesh->Dimension();
   auto geom = mesh->GetBdrElementGeometry(bel);
   int order = fec->DefaultOrder();

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
   if (nf) { mesh->GetBdrElementFace(bel, &F, &Fo); }

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
      int fbase = (face_dofs.Size() > 0) ? FindFaceDof(F, nf) : F*nf;
      const int *ind = fec->GetDofOrdering(geom, order, Fo);

      for (int j = 0; j < nf; j++)
      {
         dofs.Append(EncodeDof(nvdofs + nedofs + fbase, ind[j]));
      }
   }
}

int FiniteElementSpace::GetFaceDofs(int face, Array<int> &dofs, int variant) const
{
   int p, nf, fbase;

   if (face_dofs.Size() > 0) // variable orders or mixed faces
   {
      const int* beg = face_dofs.GetRow(face);
      const int* end = face_dofs.GetRow(face + 1);
      if (variant >= end - beg) { return -1; } // past last face DOFs

      fbase = beg[variant];
      nf = beg[variant+1] - fbase;

      p = std::sqrt(nf) + 1;  // FIXME: deduce 'p' from the number of face DOFs?
      MFEM_ASSERT(fec->GetNumDof(mesh->GetFaceGeometry(face), p) == nf, "");
   }
   else
   {
      if (variant > 0) { return -1; }
      p = fec->DefaultOrder();
      nf = fec->GetNumDof(mesh->GetFaceGeometry(face), p);
      fbase = face*nf;
   }

   // for 1D, 2D and 3D faces
   int dim = mesh->Dimension();
   int nv = fec->GetNumDof(Geometry::POINT, p);
   int ne = (dim > 1) ? fec->GetNumDof(Geometry::SEGMENT, p) : 0;

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
          const int *ind = fec->GetDofOrdering(Geometry::SEGMENT, p, Eo[i]);

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

   return p;
}

int FiniteElementSpace::GetEdgeDofs(int edge, Array<int> &dofs,
                                    int variant) const
{
   int p, ne, base;
   if (IsVariableOrder())
   {
      const int* beg = edge_dofs.GetRow(edge);
      const int* end = edge_dofs.GetRow(edge + 1);
      if (variant >= end - beg) { return -1; } // past last edge DOFs

      base = beg[variant];
      ne = beg[variant+1] - base;

      p = ne + 1;
      MFEM_ASSERT(fec->GetNumDof(Geometry::SEGMENT, p) == ne, "");
      // TODO: how to safely deduce 'p' from the number of edge DOFs?
   }
   else
   {
      if (variant > 0) { return -1; }
      p = fec->DefaultOrder();
      ne = fec->GetNumDof(Geometry::SEGMENT, p);
      base = edge*ne;
   }

   Array<int> V; // TODO: LocalArray
   int nv = fec->GetNumDof(Geometry::POINT, p);
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

   return p;
}

void FiniteElementSpace::GetVertexDofs(int i, Array<int> &dofs) const
{
   int j, nv;

   nv = fec->DofForGeometry(Geometry::POINT);
   dofs.SetSize(nv);
   for (j = 0; j < nv; j++)
   {
      dofs[j] = i*nv+j;
   }
}

void FiniteElementSpace::GetElementInteriorDofs(int i, Array<int> &dofs) const
{
   int j, k, nb;
   if (mesh->Dimension() == 0) { dofs.SetSize(0); return; }
   nb = fec -> DofForGeometry (mesh -> GetElementBaseGeometry (i));
   dofs.SetSize (nb);
   k = nvdofs + nedofs + nfdofs + bdofs[i];
   for (j = 0; j < nb; j++)
   {
      dofs[j] = k + j;
   }
}

void FiniteElementSpace::GetEdgeInteriorDofs(int i, Array<int> &dofs) const
{
   int j, k, ne;

   ne = fec -> DofForGeometry (Geometry::SEGMENT);
   dofs.SetSize (ne);
   for (j = 0, k = nvdofs+i*ne; j < ne; j++, k++)
   {
      dofs[j] = k;
   }
}

void FiniteElementSpace::GetFaceInteriorDofs(int i, Array<int> &dofs) const
{
   int nf, base;
   if (face_dofs.Size() > 0)
   {
      base = face_dofs.GetRow(i)[0];
      nf = face_dofs.GetRow(i)[1] - base;
   }
   else
   {
      auto geom = mesh->GetFaceGeometry(0);
      nf = fec->GetNumDof(geom, fec->DefaultOrder());
      base = i*nf;
   }

   dofs.SetSize(nf);
   for (int j = 0; j < nf; j++)
   {
      dofs[j] = nvdofs + nedofs + base + j;
   }
}

const FiniteElement *FiniteElementSpace::GetBE (int i) const
{
   const FiniteElement *BE;

   int order = fec->DefaultOrder();

   if (IsVariableOrder()) // determine order from adjacent element
   {
      int elem, info;
      mesh->GetBdrElementAdjacentElement(i, elem, info);
      order = elem_order[elem];
   }

   switch ( mesh->Dimension() )
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

   // if (NURBSext)
   //    NURBSext->LoadFaceElement(i, fe);

   return fe;
}

const FiniteElement *FiniteElementSpace::GetEdgeElement(int i) const
{
   return fec->FiniteElementForGeometry(Geometry::SEGMENT);
}

const FiniteElement *FiniteElementSpace::GetTraceElement(
   int i, Geometry::Type geom_type) const
{
#if 0
   return fec->TraceFiniteElementForGeometry(geom_type);
#else
   MFEM_ABORT("TODO");
   return NULL;
#endif
}

FiniteElementSpace::~FiniteElementSpace()
{
   Destroy();
}

void FiniteElementSpace::Destroy()
{
   delete cR;
   delete cQ;
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

   dof_elem_array.DeleteAll();
   dof_ldof_array.DeleteAll();

   if (NURBSext)
   {
      if (own_ext) { delete NURBSext; }
   }
   else
   {
      delete elem_dof;
      delete bdr_elem_dof;
      delete [] bdofs;
   }
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
   if (mesh->GetSequence() == sequence && !orders_changed)
   {
      return; // mesh and space are in sync, no-op
   }
   if (want_transform && mesh->GetSequence() != sequence + 1)
   {
      MFEM_ABORT("Error in update sequence. Space needs to be updated after "
                 "each mesh modification.");
   }
   if (mesh->GetSequence() != sequence && orders_changed)
   {
      MFEM_ABORT("Updating space after both mesh changes and element order "
                 "changes is not supported. Please update separately after "
                 "each change.");
   }

   if (NURBSext)
   {
      UpdateNURBS();
      return;
   }

   Table* old_elem_dof = NULL;
   int old_ndofs;

   // save old DOF table
   if (want_transform)
   {
      old_elem_dof = elem_dof;
      elem_dof = NULL;
      old_ndofs = ndofs;
   }

   // update the 'elem_order' array if the mesh has changed
   if (IsVariableOrder() && sequence != mesh->GetSequence())
   {
      UpdateElementOrders();
   }

   Destroy(); // calls Th.Clear()
   Construct();
   BuildElementToDofTable();

   if (want_transform)
   {
      MFEM_VERIFY(!orders_changed, "not implemented yet");

      // calculate appropriate GridFunction transformation
      switch (mesh->GetLastOperation())
      {
         case Mesh::REFINE:
         {
            if (Th.Type() != Operator::MFEM_SPARSEMAT)
            {
               Th.Reset(new RefinementOperator(this, old_elem_dof, old_ndofs));
               // The RefinementOperator takes ownership of 'old_elem_dof', so
               // we no longer own it:
               old_elem_dof = NULL;
            }
            else
            {
               // calculate fully assembled matrix
               Th.Reset(RefinementMatrix(old_ndofs, old_elem_dof));
            }
            break;
         }

         case Mesh::DEREFINE:
         {
            BuildConformingInterpolation();
            Th.Reset(DerefinementMatrix(old_ndofs, old_elem_dof));
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

   sequence = mesh->GetSequence();
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
#if 0
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
#else
      MFEM_ABORT("TODO");
#endif
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
#if 0 // FIXME
         if (NURBSext->GetOrder() != NURBSFECollection::VariableOrder)
         {
            out << "NURBS_order\n" << NURBSext->GetOrder() << '\n';
         }
         else
#endif
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

#if 0 // FIXME
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
#else
   (void) fes_format;
#endif

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
      const SparseMatrix *out_R = fes_out.GetRestrictionMatrix();
      if (oper_type == Operator::Hypre_ParCSR)
      {
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

   // Costruct F
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
                 dom_fes.GetNDofs(), dom_fes.GetElementToDofTable(), localP));
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
   : fes_ho(fes_ho_), fes_lor(fes_lor_)
{
   Mesh *mesh_ho = fes_ho.GetMesh();
   MFEM_VERIFY(mesh_ho->GetNumGeometries(mesh_ho->Dimension()) <= 1,
               "mixed meshes are not supported");

   // If the local mesh is empty, skip all computations
   if (mesh_ho->GetNE() == 0) { return; }

   const FiniteElement *fe_lor = fes_lor.GetFE(0);
   const FiniteElement *fe_ho = fes_ho.GetFE(0);
   ndof_lor = fe_lor->GetDof();
   ndof_ho = fe_ho->GetDof();

   const int nel_lor = fes_lor.GetNE();
   const int nel_ho = fes_ho.GetNE();

   nref = nel_lor/nel_ho;

   // Construct the mapping from HO to LOR
   // ho2lor.GetRow(iho) will give all the LOR elements contained in iho
   ho2lor.SetSize(nel_ho, nref);
   const CoarseFineTransformations &cf_tr =
      fes_lor.GetMesh()->GetRefinementTransforms();
   for (int ilor=0; ilor<nel_lor; ++ilor)
   {
      int iho = cf_tr.embeddings[ilor].parent;
      ho2lor.AddConnection(iho, ilor);
   }
   ho2lor.ShiftUpI();

   // R will contain the restriction (L^2 projection operator) defined on
   // each coarse HO element (and corresponding patch of LOR elements)
   R.SetSize(ndof_lor*nref, ndof_ho, nel_ho);
   // P will contain the corresponding prolongation operator
   P.SetSize(ndof_ho, ndof_lor*nref, nel_ho);

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

   IntegrationPointTransformation ip_tr;
   IsoparametricTransformation &emb_tr = ip_tr.Transf;

   Vector shape_ho(ndof_ho);
   Vector shape_lor(ndof_lor);

   const Geometry::Type geom = fe_ho->GetGeomType();
   const DenseTensor &pmats = cf_tr.point_matrices[geom];
   emb_tr.SetIdentityTransformation(geom);

   for (int iho=0; iho<nel_ho; ++iho)
   {
      for (int iref=0; iref<nref; ++iref)
      {
         // Assemble the low-order refined mass matrix and invert locally
         int ilor = ho2lor.GetRow(iho)[iref];
         ElementTransformation *el_tr = fes_lor.GetElementTransformation(ilor);
         mi.AssembleElementMatrix(*fe_lor, *el_tr, M_lor_el);
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

         int order = fe_lor->GetOrder() + fe_ho->GetOrder() + el_tr->OrderW();
         const IntegrationRule *ir = &IntRules.Get(geom, order);
         M_mixed_el = 0.0;
         for (int i = 0; i < ir->GetNPoints(); i++)
         {
            const IntegrationPoint &ip_lor = ir->IntPoint(i);
            IntegrationPoint ip_ho;
            ip_tr.Transform(ip_lor, ip_ho);
            fe_lor->CalcShape(ip_lor, shape_lor);
            fe_ho->CalcShape(ip_ho, shape_ho);
            el_tr->SetIntPoint(&ip_lor);
            // For now we use the geometry information from the LOR space
            // which means we won't be mass conservative if the mesh is curved
            double w = el_tr->Weight()*ip_lor.weight;
            shape_lor *= w;
            AddMultVWt(shape_lor, shape_ho, M_mixed_el);
         }
         M_mixed.CopyMN(M_mixed_el, iref*ndof_lor, 0);
      }
      mfem::Mult(Minv_lor, M_mixed, R(iho));

      mfem::MultAtB(R(iho), M_lor, RtMlor);
      mfem::Mult(RtMlor, R(iho), RtMlorR);
      RtMlorR_inv.Factor();
      RtMlorR_inv.Mult(RtMlor, P(iho));
   }
}

void L2ProjectionGridTransfer::L2Projection::Mult(
   const Vector &x, Vector &y) const
{
   int vdim = fes_ho.GetVDim();
   Array<int> vdofs;
   DenseMatrix xel_mat(ndof_ho, vdim);
   DenseMatrix yel_mat(ndof_lor*nref, vdim);
   for (int iho=0; iho<fes_ho.GetNE(); ++iho)
   {
      fes_ho.GetElementVDofs(iho, vdofs);
      x.GetSubVector(vdofs, xel_mat.GetData());
      mfem::Mult(R(iho), xel_mat, yel_mat);
      // Place result correctly into the low-order vector
      for (int iref=0; iref<nref; ++iref)
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
   DenseMatrix xel_mat(ndof_lor*nref, vdim);
   DenseMatrix yel_mat(ndof_ho, vdim);
   for (int iho=0; iho<fes_ho.GetNE(); ++iho)
   {
      // Extract the LOR DOFs
      for (int iref=0; iref<nref; ++iref)
      {
         int ilor = ho2lor.GetRow(iho)[iref];
         for (int vd=0; vd<vdim; ++vd)
         {
            fes_lor.GetElementDofs(ilor, vdofs);
            fes_lor.DofsToVDofs(vd, vdofs);
            x.GetSubVector(vdofs, &xel_mat(iref*ndof_lor, vd));
         }
      }
      // Locally prolongate
      mfem::Mult(P(iho), xel_mat, yel_mat);
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
