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

// Implementation of FiniteElementSpace

#include "../general/text.hpp"
#include "../general/forall.hpp"
#include "../mesh/mesh_headers.hpp"
#include "fem.hpp"

#include <cmath>
#include <cstdarg>
#include <limits>

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
     fdofs(NULL), bdofs(NULL),
     elem_dof(NULL), bdrElem_dof(NULL),
     NURBSext(NULL), own_ext(false),
     cP(NULL), cR(NULL), cP_is_set(false),
     Th(Operator::ANY_TYPE),
     sequence(0)
{ }

FiniteElementSpace::FiniteElementSpace(const FiniteElementSpace &orig,
                                       Mesh *mesh,
                                       const FiniteElementCollection *fec)
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

int FiniteElementSpace::GetOrder(int i) const
{
   Geometry::Type GeomType = mesh->GetElementBaseGeometry(i);
   return fec->FiniteElementForGeometry(GeomType)->GetOrder();
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

void FiniteElementSpace::BuildElementToDofTable() const
{
   if (elem_dof) { return; }

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
   if (mesh->ncmesh)
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
   Array<int> l_dofs, h_dofs;

   R = new SparseMatrix (lfes -> GetNDofs(), ndofs);

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

      R -> SetSubMatrix (l_dofs, h_dofs, loc_restr, 1);
   }

   R -> Finalize();

   return R;
}

void
FiniteElementSpace::AddDependencies(SparseMatrix& deps, Array<int>& master_dofs,
                                    Array<int>& slave_dofs, DenseMatrix& I)
{
   for (int i = 0; i < slave_dofs.Size(); i++)
   {
      int sdof = slave_dofs[i];
      if (!deps.RowSize(sdof)) // not processed yet?
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

void FiniteElementSpace::GetDegenerateFaceDofs(int index,
                                               Array<int> &dofs) const
{
   // In NC meshes with prisms, a special constraint occurs where a prism edge
   // is slave to a quadrilateral face. Rather than introduce a new edge-face
   // constraint type, we handle such cases as degenerate face-face constraints,
   // where the point-matrix rectangle has zero height. This method returns
   // DOFs for the first edge of the rectangle, duplicated in the orthogonal
   // direction, to resemble DOFs for a quadrilateral face. The extra DOFs are
   // ignored by FiniteElementSpace::AddDependencies.

   Array<int> edof;
   GetEdgeDofs(-1 - index, edof);

   int nv = fec->DofForGeometry(Geometry::POINT);
   int ne = fec->DofForGeometry(Geometry::SEGMENT);
   int nn = 2*nv + ne;

   dofs.SetSize(nn*nn);
   dofs = edof[0];

   // copy first two vertex DOFs
   for (int i = 0; i < nv; i++)
   {
      dofs[i] = edof[i];
      dofs[nv+i] = edof[nv+i];
   }
   // copy first edge DOFs
   for (int i = 0; i < ne; i++)
   {
      dofs[4*nv + i] = edof[2*nv + i];
   }
}

void
FiniteElementSpace::GetEntityDofs(int entity, int index, Array<int> &dofs) const
{
   switch (entity)
   {
      case 0: GetVertexDofs(index, dofs); break;
      case 1: GetEdgeDofs(index, dofs); break;
      case 2: (index >= 0) ? GetFaceDofs(index, dofs)
         /*             */ : GetDegenerateFaceDofs(index, dofs);
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

   // For each slave DOF, the dependency matrix will contain a row that
   // expresses the slave DOF as a linear combination of its immediate master
   // DOFs. Rows of independent DOFs will remain empty.
   SparseMatrix deps(ndofs);

   // collect local edge/face dependencies
   for (int entity = 1; entity <= 2; entity++)
   {
      const NCMesh::NCList &list = mesh->ncmesh->GetNCList(entity);
      if (!list.masters.size()) { continue; }

      Array<int> master_dofs, slave_dofs;

      IsoparametricTransformation T;
      DenseMatrix I;

      // loop through all master edges/faces, constrain their slave edges/faces
      for (unsigned mi = 0; mi < list.masters.size(); mi++)
      {
         const NCMesh::Master &master = list.masters[mi];

         GetEntityDofs(entity, master.index, master_dofs);
         if (!master_dofs.Size()) { continue; }

         const FiniteElement* fe = fec->FiniteElementForGeometry(master.Geom());
         if (!fe) { continue; }

         switch (master.geom)
         {
            case Geometry::SQUARE:   T.SetFE(&QuadrilateralFE); break;
            case Geometry::TRIANGLE: T.SetFE(&TriangleFE); break;
            case Geometry::SEGMENT:  T.SetFE(&SegmentFE); break;
            default: MFEM_ABORT("unsupported geometry");
         }

         for (int si = master.slaves_begin; si < master.slaves_end; si++)
         {
            const NCMesh::Slave &slave = list.slaves[si];
            GetEntityDofs(entity, slave.index, slave_dofs);
            if (!slave_dofs.Size()) { continue; }

            slave.OrientedPointMatrix(T.GetPointMat());
            T.FinalizeTransformation();
            fe->GetLocalInterpolation(T, I);

            // make each slave DOF dependent on all master DOFs
            AddDependencies(deps, master_dofs, slave_dofs, I);
         }
      }
   }

   deps.Finalize();

   // DOFs that stayed independent are true DOFs
   int n_true_dofs = 0;
   for (int i = 0; i < ndofs; i++)
   {
      if (!deps.RowSize(i)) { n_true_dofs++; }
   }

   // if all dofs are true dofs leave cP and cR NULL
   if (n_true_dofs == ndofs)
   {
      cP = cR = NULL; // will be treated as identities
      return;
   }

   // create the conforming restriction matrix cR
   int *cR_J;
   {
      int *cR_I = new int[n_true_dofs+1];
      double *cR_A = new double[n_true_dofs];
      cR_J = new int[n_true_dofs];
      for (int i = 0; i < n_true_dofs; i++)
      {
         cR_I[i] = i;
         cR_A[i] = 1.0;
      }
      cR_I[n_true_dofs] = n_true_dofs;
      cR = new SparseMatrix(cR_I, cR_J, cR_A, n_true_dofs, ndofs);
   }

   // create the conforming prolongation matrix cP
   cP = new SparseMatrix(ndofs, n_true_dofs);

   Array<bool> finalized(ndofs);
   finalized = false;

   // put identity in the restriction and prolongation matrices for true DOFs
   for (int i = 0, true_dof = 0; i < ndofs; i++)
   {
      if (!deps.RowSize(i))
      {
         cR_J[true_dof] = i;
         cP->Add(i, true_dof++, 1.0);
         finalized[i] = true;
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
   Array<int> cols;
   Vector srow;
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
      MFEM_ABORT("Error creating cP matrix.");
   }

   cP->Finalize();

   if (vdim > 1)
   {
      MakeVDimMatrix(*cP);
      MakeVDimMatrix(*cR);
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

int FiniteElementSpace::GetNConformingDofs() const
{
   const SparseMatrix* P = GetConformingProlongation();
   return P ? (P->Width() / vdim) : ndofs;
}

const Operator *FiniteElementSpace::GetElementRestriction(
   ElementDofOrdering e_ordering) const
{
   // Check if we have a discontinuous space using the FE collection:
   const L2_FECollection *dg_space = dynamic_cast<const L2_FECollection*>(fec);
   if (dg_space)
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
   int ldof = fe->GetDof(); // assuming the same FE everywhere

   IsoparametricTransformation isotr;
   isotr.SetIdentityTransformation(geom);

   // calculate local interpolation matrices for all refinement types
   localP.SetSize(ldof, ldof, nmat);
   for (int i = 0; i < nmat; i++)
   {
      isotr.GetPointMat() = pmats(i);
      isotr.FinalizeTransformation();
      fe->GetLocalInterpolation(isotr, localP(i));
   }
}

SparseMatrix* FiniteElementSpace::RefinementMatrix(int old_ndofs,
                                                   const Table* old_elem_dof)
{
   MFEM_VERIFY(ndofs >= old_ndofs, "Previous space is not coarser.");

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
   const Mesh* mesh = fespace->GetMesh();
   MFEM_VERIFY(mesh->ReduceInt(fespace->GetNDofs()) >=
               mesh->ReduceInt(old_ndofs),
               "Previous space is not coarser.");

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

   Array<int> dofs, old_dofs, old_vdofs;

   Array<char> processed(fespace->GetVSize());
   processed = 0;

   int vdim = fespace->GetVDim();
   int old_ndofs = width / vdim;

   for (int k = 0; k < mesh->GetNE(); k++)
   {
      const Embedding &emb = rtrans.embeddings[k];
      const Geometry::Type geom = mesh->GetElementBaseGeometry(k);
      const DenseMatrix &lP = localP[geom](emb.matrix);

      fespace->GetElementDofs(k, dofs);
      old_elem_dof->GetRow(emb.parent, old_dofs);

      for (int vd = 0; vd < vdim; vd++)
      {
         old_dofs.Copy(old_vdofs);
         fespace->DofsToVDofs(vd, old_vdofs, old_ndofs);

         for (int i = 0; i < dofs.Size(); i++)
         {
            double rsign, osign;
            int r = fespace->DofToVDof(dofs[i], vd);
            r = DecodeDof(r, rsign);

            if (!processed[r])
            {
               double value = 0.0;
               for (int j = 0; j < old_vdofs.Size(); j++)
               {
                  int o = DecodeDof(old_vdofs[j], osign);
                  value += x[o] * lP(i, j) * osign;
               }
               y[r] = value * rsign;
               processed[r] = 1;
            }
         }
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
         emb_tr.GetPointMat() = pmats(i);
         emb_tr.FinalizeTransformation();
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
      isotr.GetPointMat() = pmats(i);
      isotr.FinalizeTransformation();

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

   SparseMatrix *R;
   if (elem_geoms.Size() == 1)
   {
      R = new SparseMatrix(ndofs*vdim, old_ndofs*vdim,
                           localR[elem_geoms[0]].SizeI());
   }
   else
   {
      R = new SparseMatrix(ndofs*vdim, old_ndofs*vdim);
   }
   Array<int> mark(R->Height());
   mark = 0;

   const CoarseFineTransformations &dtrans =
      mesh->ncmesh->GetDerefinementTransforms();

   MFEM_ASSERT(dtrans.embeddings.Size() == old_elem_dof->Size(), "");

   int num_marked = 0;
   for (int k = 0; k < dtrans.embeddings.Size(); k++)
   {
      const Embedding &emb = dtrans.embeddings[k];
      const Geometry::Type geom = mesh->GetElementBaseGeometry(emb.parent);
      DenseMatrix &lR = localR[geom](emb.matrix);

      elem_dof->GetRow(emb.parent, dofs);
      old_elem_dof->GetRow(k, old_dofs);

      for (int vd = 0; vd < vdim; vd++)
      {
         old_dofs.Copy(old_vdofs);
         DofsToVDofs(vd, old_vdofs, old_ndofs);

         for (int i = 0; i < lR.Height(); i++)
         {
            if (lR(i, 0) == infinity()) { continue; }

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
   if (elem_geoms.Size() != 1) { R->Finalize(); }
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
      isotr.GetPointMat() = pmats(i);
      isotr.FinalizeTransformation();
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

   const NURBSFECollection *nurbs_fec =
      dynamic_cast<const NURBSFECollection *>(fec);
   if (nurbs_fec)
   {
      if (!mesh->NURBSext)
      {
         mfem_error("FiniteElementSpace::FiniteElementSpace :\n"
                    "   NURBS FE space requires NURBS mesh.");
      }

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
      cP = cR = NULL;
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
   fdofs = NULL;
   bdofs = NULL;

   dynamic_cast<const NURBSFECollection *>(fec)->Reset();

   ndofs = NURBSext->GetNDof();
   elem_dof = NURBSext->GetElementDofTable();
   bdrElem_dof = NURBSext->GetBdrElementDofTable();
}

void FiniteElementSpace::Construct()
{
   // This method should be used only for non-NURBS spaces.
   MFEM_VERIFY(!NURBSext, "internal error");

   elem_dof = NULL;
   bdrElem_dof = NULL;

   ndofs = 0;
   nedofs = nfdofs = nbdofs = 0;
   bdofs = NULL;
   fdofs = NULL;
   cP = NULL;
   cR = NULL;
   cP_is_set = false;
   // 'Th' is initialized/destroyed before this method is called.

   nvdofs = mesh->GetNV() * fec->DofForGeometry(Geometry::POINT);

   if (mesh->Dimension() > 1)
   {
      nedofs = mesh->GetNEdges() * fec->DofForGeometry(Geometry::SEGMENT);
   }

   if (mesh->GetNFaces() > 0)
   {
      bool have_face_dofs = false;
      for (int g = Geometry::DimStart[2]; g < Geometry::DimStart[3]; g++)
      {
         if (mesh->HasGeometry(Geometry::Type(g)) &&
             fec->DofForGeometry(Geometry::Type(g)) > 0)
         {
            have_face_dofs = true;
            break;
         }
      }
      if (have_face_dofs)
      {
         fdofs = new int[mesh->GetNFaces()+1];
         fdofs[0] = 0;
         for (int i = 0; i < mesh->GetNFaces(); i++)
         {
            nfdofs += fec->DofForGeometry(mesh->GetFaceBaseGeometry(i));
            fdofs[i+1] = nfdofs;
         }
      }
   }

   if (mesh->Dimension() > 0)
   {
      bdofs = new int[mesh->GetNE()+1];
      bdofs[0] = 0;
      for (int i = 0; i < mesh->GetNE(); i++)
      {
         nbdofs += fec->DofForGeometry(mesh->GetElementBaseGeometry(i));
         bdofs[i+1] = nbdofs;
      }
   }

   ndofs = nvdofs + nedofs + nfdofs + nbdofs;

   // Do not build elem_dof Table here: in parallel it has to be constructed
   // later.
}

void FiniteElementSpace::GetElementDofs(int i, Array<int> &dofs) const
{
   if (elem_dof)
   {
      elem_dof -> GetRow (i, dofs);
   }
   else
   {
      Array<int> V, E, Eo, F, Fo;
      int k, j, nv, ne, nf, nb, nfd, nd, dim;
      const int *ind;

      dim = mesh->Dimension();
      nv = fec->DofForGeometry(Geometry::POINT);
      ne = (dim > 1) ? ( fec->DofForGeometry(Geometry::SEGMENT) ) : ( 0 );
      nb = (dim > 0) ? fec->DofForGeometry(mesh->GetElementBaseGeometry(i)) : 0;
      if (nv > 0)
      {
         mesh->GetElementVertices(i, V);
      }
      if (ne > 0)
      {
         mesh->GetElementEdges(i, E, Eo);
      }
      nfd = 0;
      if (dim == 3)
      {
         if (fec->HasFaceDofs(mesh->GetElementBaseGeometry(i)))
         {
            mesh->GetElementFaces(i, F, Fo);
            for (k = 0; k < F.Size(); k++)
            {
               nfd += fec->DofForGeometry(mesh->GetFaceBaseGeometry(F[k]));
            }
         }
      }
      nd = V.Size() * nv + E.Size() * ne + nfd + nb;
      dofs.SetSize(nd);
      if (nv > 0)
      {
         for (k = 0; k < V.Size(); k++)
         {
            for (j = 0; j < nv; j++)
            {
               dofs[k*nv+j] = V[k]*nv+j;
            }
         }
         nv *= V.Size();
      }
      if (ne > 0)
      {
         // if (dim > 1)
         for (k = 0; k < E.Size(); k++)
         {
            ind = fec->DofOrderForOrientation(Geometry::SEGMENT, Eo[k]);
            for (j = 0; j < ne; j++)
            {
               if (ind[j] < 0)
               {
                  dofs[nv+k*ne+j] = -1 - ( nvdofs+E[k]*ne+(-1-ind[j]) );
               }
               else
               {
                  dofs[nv+k*ne+j] = nvdofs+E[k]*ne+ind[j];
               }
            }
         }
      }
      ne = nv + ne * E.Size();
      if (nfd > 0)
         // if (dim == 3)
      {
         for (k = 0; k < F.Size(); k++)
         {
            ind = fec->DofOrderForOrientation(mesh->GetFaceBaseGeometry(F[k]),
                                              Fo[k]);
            nf = fec->DofForGeometry(mesh->GetFaceBaseGeometry(F[k]));
            for (j = 0; j < nf; j++)
            {
               if (ind[j] < 0)
               {
                  dofs[ne+j] = -1 - ( nvdofs+nedofs+fdofs[F[k]]+(-1-ind[j]) );
               }
               else
               {
                  dofs[ne+j] = nvdofs+nedofs+fdofs[F[k]]+ind[j];
               }
            }
            ne += nf;
         }
      }
      if (nb > 0)
      {
         k = nvdofs + nedofs + nfdofs + bdofs[i];
         for (j = 0; j < nb; j++)
         {
            dofs[ne+j] = k + j;
         }
      }
   }
}

const FiniteElement *FiniteElementSpace::GetFE(int i) const
{
   if (i < 0 || !mesh->GetNE()) { return NULL; }
   MFEM_VERIFY(i < mesh->GetNE(),
               "Invalid element id " << i << ", maximum allowed " << mesh->GetNE()-1);

   const FiniteElement *FE =
      fec->FiniteElementForGeometry(mesh->GetElementBaseGeometry(i));

   if (NURBSext)
   {
      NURBSext->LoadFE(i, FE);
   }

   return FE;
}

void FiniteElementSpace::GetBdrElementDofs(int i, Array<int> &dofs) const
{
   if (bdrElem_dof)
   {
      bdrElem_dof->GetRow(i, dofs);
   }
   else
   {
      Array<int> V, E, Eo;
      int k, j, nv, ne, nf, nd, iF, oF, dim;
      const int *ind;

      dim = mesh->Dimension();
      nv = fec->DofForGeometry(Geometry::POINT);
      if (nv > 0)
      {
         mesh->GetBdrElementVertices(i, V);
      }
      ne = (dim > 1) ? ( fec->DofForGeometry(Geometry::SEGMENT) ) : ( 0 );
      if (ne > 0)
      {
         mesh->GetBdrElementEdges(i, E, Eo);
      }
      nd = V.Size() * nv + E.Size() * ne;
      nf = (dim == 3) ? (fec->DofForGeometry(
                            mesh->GetBdrElementBaseGeometry(i))) : (0);
      if (nf > 0)
      {
         nd += nf;
         mesh->GetBdrElementFace(i, &iF, &oF);
      }
      dofs.SetSize(nd);
      if (nv > 0)
      {
         for (k = 0; k < V.Size(); k++)
         {
            for (j = 0; j < nv; j++)
            {
               dofs[k*nv+j] = V[k]*nv+j;
            }
         }
         nv *= V.Size();
      }
      if (ne > 0)
      {
         // if (dim > 1)
         for (k = 0; k < E.Size(); k++)
         {
            ind = fec->DofOrderForOrientation(Geometry::SEGMENT, Eo[k]);
            for (j = 0; j < ne; j++)
            {
               if (ind[j] < 0)
               {
                  dofs[nv+k*ne+j] = -1 - ( nvdofs+E[k]*ne+(-1-ind[j]) );
               }
               else
               {
                  dofs[nv+k*ne+j] = nvdofs+E[k]*ne+ind[j];
               }
            }
         }
      }
      if (nf > 0)
         // if (dim == 3)
      {
         ne = nv + ne * E.Size();
         ind = fec->DofOrderForOrientation(
                  mesh->GetBdrElementBaseGeometry(i), oF);
         for (j = 0; j < nf; j++)
         {
            if (ind[j] < 0)
            {
               dofs[ne+j] = -1 - ( nvdofs+nedofs+fdofs[iF]+(-1-ind[j]) );
            }
            else
            {
               dofs[ne+j] = nvdofs+nedofs+fdofs[iF]+ind[j];
            }
         }
      }
   }
}

void FiniteElementSpace::GetFaceDofs(int i, Array<int> &dofs) const
{
   int j, k, nv, ne, nf, nd, dim = mesh->Dimension();
   Array<int> V, E, Eo;
   const int *ind;

   // for 1D, 2D and 3D faces
   nv = fec->DofForGeometry(Geometry::POINT);
   ne = (dim > 1) ? fec->DofForGeometry(Geometry::SEGMENT) : 0;
   if (nv > 0)
   {
      mesh->GetFaceVertices(i, V);
   }
   if (ne > 0)
   {
      mesh->GetFaceEdges(i, E, Eo);
   }
   nf = (fdofs) ? (fdofs[i+1]-fdofs[i]) : (0);
   nd = V.Size() * nv + E.Size() * ne + nf;
   dofs.SetSize(nd);
   if (nv > 0)
   {
      for (k = 0; k < V.Size(); k++)
      {
         for (j = 0; j < nv; j++)
         {
            dofs[k*nv+j] = V[k]*nv+j;
         }
      }
   }
   nv *= V.Size();
   if (ne > 0)
   {
      for (k = 0; k < E.Size(); k++)
      {
         ind = fec->DofOrderForOrientation(Geometry::SEGMENT, Eo[k]);
         for (j = 0; j < ne; j++)
         {
            if (ind[j] < 0)
            {
               dofs[nv+k*ne+j] = -1 - ( nvdofs+E[k]*ne+(-1-ind[j]) );
            }
            else
            {
               dofs[nv+k*ne+j] = nvdofs+E[k]*ne+ind[j];
            }
         }
      }
   }
   ne = nv + ne * E.Size();
   if (nf > 0)
   {
      for (j = nvdofs+nedofs+fdofs[i], k = 0; k < nf; j++, k++)
      {
         dofs[ne+k] = j;
      }
   }
}

void FiniteElementSpace::GetEdgeDofs(int i, Array<int> &dofs) const
{
   int j, k, nv, ne;
   Array<int> V;

   nv = fec->DofForGeometry(Geometry::POINT);
   if (nv > 0)
   {
      mesh->GetEdgeVertices(i, V);
   }
   ne = fec->DofForGeometry(Geometry::SEGMENT);
   dofs.SetSize(2*nv+ne);
   if (nv > 0)
   {
      for (k = 0; k < 2; k++)
      {
         for (j = 0; j < nv; j++)
         {
            dofs[k*nv+j] = V[k]*nv+j;
         }
      }
   }
   nv *= 2;
   for (j = 0, k = nvdofs+i*ne; j < ne; j++, k++)
   {
      dofs[nv+j] = k;
   }
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

void FiniteElementSpace::GetElementInteriorDofs (int i, Array<int> &dofs) const
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

void FiniteElementSpace::GetEdgeInteriorDofs (int i, Array<int> &dofs) const
{
   int j, k, ne;

   ne = fec -> DofForGeometry (Geometry::SEGMENT);
   dofs.SetSize (ne);
   for (j = 0, k = nvdofs+i*ne; j < ne; j++, k++)
   {
      dofs[j] = k;
   }
}

void FiniteElementSpace::GetFaceInteriorDofs (int i, Array<int> &dofs) const
{
   int j, k, nf;

   nf = (fdofs) ? (fdofs[i+1]-fdofs[i]) : (0);
   dofs.SetSize (nf);
   if (nf > 0)
   {
      for (j = 0, k = nvdofs+nedofs+fdofs[i]; j < nf; j++, k++)
      {
         dofs[j] = k;
      }
   }
}

const FiniteElement *FiniteElementSpace::GetBE (int i) const
{
   const FiniteElement *BE;

   switch ( mesh->Dimension() )
   {
      case 1:
         BE = fec->FiniteElementForGeometry(Geometry::POINT);
         break;
      case 2:
         BE = fec->FiniteElementForGeometry(Geometry::SEGMENT);
         break;
      case 3:
      default:
         BE = fec->FiniteElementForGeometry(
                 mesh->GetBdrElementBaseGeometry(i));
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
   return fec->TraceFiniteElementForGeometry(geom_type);
}

FiniteElementSpace::~FiniteElementSpace()
{
   Destroy();
}

void FiniteElementSpace::Destroy()
{
   delete cR;
   delete cP;
   Th.Clear();
   L2E_nat.Clear();
   L2E_lex.Clear();
   for (int i = 0; i < E2Q_array.Size(); i++)
   {
      delete E2Q_array[i];
   }
   E2Q_array.SetSize(0);

   dof_elem_array.DeleteAll();
   dof_ldof_array.DeleteAll();

   if (NURBSext)
   {
      if (own_ext) { delete NURBSext; }
   }
   else
   {
      delete elem_dof;
      delete bdrElem_dof;

      delete [] bdofs;
      delete [] fdofs;
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

void FiniteElementSpace::Update(bool want_transform)
{
   if (mesh->GetSequence() == sequence)
   {
      return; // mesh and space are in sync, no-op
   }
   if (want_transform && mesh->GetSequence() != sequence + 1)
   {
      MFEM_ABORT("Error in update sequence. Space needs to be updated after "
                 "each mesh modification.");
   }
   sequence = mesh->GetSequence();

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

   Destroy(); // calls Th.Clear()
   Construct();
   BuildElementToDofTable();

   if (want_transform)
   {
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
         emb_tr.GetPointMat() = pmats(cf_tr.embeddings[ilor].matrix);
         emb_tr.FinalizeTransformation();

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

L2ElementRestriction::L2ElementRestriction(const FiniteElementSpace &fes)
   : ne(fes.GetNE()),
     vdim(fes.GetVDim()),
     byvdim(fes.GetOrdering() == Ordering::byVDIM),
     ndof(ne > 0 ? fes.GetFE(0)->GetDof() : 0)
{
   height = vdim*ne*ndof;
   width = vdim*ne*ndof;
}

void L2ElementRestriction::Mult(const Vector &x, Vector &y) const
{
   const int NE = ne;
   const int VDIM = vdim;
   const int NDOF = ndof;
   const bool BYVDIM = byvdim;
   auto d_x = x.Read();
   auto d_y = y.Write();
   MFEM_FORALL(iel, NE,
   {
      for (int vd=0; vd<VDIM; ++vd)
      {
         for (int idof=0; idof<NDOF; ++idof)
         {
            // E-vector dimensions (dofs, vdim, elements)
            // L-vector dimensions: byVDIM:  (vdim, dofs, element)
            //                      byNODES: (dofs, elements, vdim)
            int yidx = iel*VDIM*NDOF + vd*NDOF + idof;
            int xidx;
            if (BYVDIM)
            {
               xidx = iel*NDOF*VDIM + idof*VDIM + vd;
            }
            else
            {
               xidx = vd*NE*NDOF + iel*NDOF + idof;
            }
            d_y[yidx] = d_x[xidx];
         }
      }
   });
}
void L2ElementRestriction::MultTranspose(const Vector &x, Vector &y) const
{
   const int NE = ne;
   const int VDIM = vdim;
   const int NDOF = ndof;
   const bool BYVDIM = byvdim;
   auto d_x = x.Read();
   auto d_y = y.Write();
   // Since this restriction is a permutation, the transpose is the inverse
   MFEM_FORALL(iel, NE,
   {
      for (int vd=0; vd<VDIM; ++vd)
      {
         for (int idof=0; idof<NDOF; ++idof)
         {
            // E-vector dimensions (dofs, vdim, elements)
            // L-vector dimensions: byVDIM:  (vdim, dofs, element)
            //                      byNODES: (dofs, elements, vdim)
            int xidx = iel*VDIM*NDOF + vd*NDOF + idof;
            int yidx;
            if (BYVDIM)
            {
               yidx = iel*NDOF*VDIM + idof*VDIM + vd;
            }
            else
            {
               yidx = vd*NE*NDOF + iel*NDOF + idof;
            }
            d_y[yidx] = d_x[xidx];
         }
      }
   });
}

ElementRestriction::ElementRestriction(const FiniteElementSpace &f,
                                       ElementDofOrdering e_ordering)
   : fes(f),
     ne(fes.GetNE()),
     vdim(fes.GetVDim()),
     byvdim(fes.GetOrdering() == Ordering::byVDIM),
     ndofs(fes.GetNDofs()),
     dof(ne > 0 ? fes.GetFE(0)->GetDof() : 0),
     nedofs(ne*dof),
     offsets(ndofs+1),
     indices(ne*dof)
{
   // Assuming all finite elements are the same.
   height = vdim*ne*dof;
   width = fes.GetVSize();
   const bool dof_reorder = (e_ordering == ElementDofOrdering::LEXICOGRAPHIC);
   const int *dof_map = NULL;
   if (dof_reorder && ne > 0)
   {
      for (int e = 0; e < ne; ++e)
      {
         const FiniteElement *fe = fes.GetFE(e);
         const TensorBasisElement* el =
            dynamic_cast<const TensorBasisElement*>(fe);
         if (el) { continue; }
         mfem_error("Finite element not suitable for lexicographic ordering");
      }
      const FiniteElement *fe = fes.GetFE(0);
      const TensorBasisElement* el =
         dynamic_cast<const TensorBasisElement*>(fe);
      const Array<int> &fe_dof_map = el->GetDofMap();
      MFEM_VERIFY(fe_dof_map.Size() > 0, "invalid dof map");
      dof_map = fe_dof_map.GetData();
   }
   const Table& e2dTable = fes.GetElementToDofTable();
   const int* elementMap = e2dTable.GetJ();
   // We will be keeping a count of how many local nodes point to its global dof
   for (int i = 0; i <= ndofs; ++i)
   {
      offsets[i] = 0;
   }
   for (int e = 0; e < ne; ++e)
   {
      for (int d = 0; d < dof; ++d)
      {
         const int gid = elementMap[dof*e + d];
         ++offsets[gid + 1];
      }
   }
   // Aggregate to find offsets for each global dof
   for (int i = 1; i <= ndofs; ++i)
   {
      offsets[i] += offsets[i - 1];
   }
   // For each global dof, fill in all local nodes that point to it
   for (int e = 0; e < ne; ++e)
   {
      for (int d = 0; d < dof; ++d)
      {
         const int did = (!dof_reorder)?d:dof_map[d];
         const int gid = elementMap[dof*e + did];
         const int lid = dof*e + d;
         indices[offsets[gid]++] = lid;
      }
   }
   // We shifted the offsets vector by 1 by using it as a counter.
   // Now we shift it back.
   for (int i = ndofs; i > 0; --i)
   {
      offsets[i] = offsets[i - 1];
   }
   offsets[0] = 0;
}

void ElementRestriction::Mult(const Vector& x, Vector& y) const
{
   // Assumes all elements have the same number of dofs
   const int nd = dof;
   const int vd = vdim;
   const bool t = byvdim;
   auto d_offsets = offsets.Read();
   auto d_indices = indices.Read();
   auto d_x = Reshape(x.Read(), t?vd:ndofs, t?ndofs:vd);
   auto d_y = Reshape(y.Write(), nd, vd, ne);
   MFEM_FORALL(i, ndofs,
   {
      const int offset = d_offsets[i];
      const int nextOffset = d_offsets[i+1];
      for (int c = 0; c < vd; ++c)
      {
         const double dofValue = d_x(t?c:i,t?i:c);
         for (int j = offset; j < nextOffset; ++j)
         {
            const int idx_j = d_indices[j];
            d_y(idx_j % nd, c, idx_j / nd) = dofValue;
         }
      }
   });
}

void ElementRestriction::MultTranspose(const Vector& x, Vector& y) const
{
   // Assumes all elements have the same number of dofs
   const int nd = dof;
   const int vd = vdim;
   const bool t = byvdim;
   auto d_offsets = offsets.Read();
   auto d_indices = indices.Read();
   auto d_x = Reshape(x.Read(), nd, vd, ne);
   auto d_y = Reshape(y.Write(), t?vd:ndofs, t?ndofs:vd);
   MFEM_FORALL(i, ndofs,
   {
      const int offset = d_offsets[i];
      const int nextOffset = d_offsets[i + 1];
      for (int c = 0; c < vd; ++c)
      {
         double dofValue = 0;
         for (int j = offset; j < nextOffset; ++j)
         {
            const int idx_j = d_indices[j];
            dofValue +=  d_x(idx_j % nd, c, idx_j / nd);
         }
         d_y(t?c:i,t?i:c) = dofValue;
      }
   });
}


QuadratureInterpolator::QuadratureInterpolator(const FiniteElementSpace &fes,
                                               const IntegrationRule &ir)
{
   fespace = &fes;
   qspace = NULL;
   IntRule = &ir;
   use_tensor_products = true; // not implemented yet (not used)

   if (fespace->GetNE() == 0) { return; }
   const FiniteElement *fe = fespace->GetFE(0);
   MFEM_VERIFY(dynamic_cast<const ScalarFiniteElement*>(fe) != NULL,
               "Only scalar finite elements are supported");
}

QuadratureInterpolator::QuadratureInterpolator(const FiniteElementSpace &fes,
                                               const QuadratureSpace &qs)
{
   fespace = &fes;
   qspace = &qs;
   IntRule = NULL;
   use_tensor_products = true; // not implemented yet (not used)

   if (fespace->GetNE() == 0) { return; }
   const FiniteElement *fe = fespace->GetFE(0);
   MFEM_VERIFY(dynamic_cast<const ScalarFiniteElement*>(fe) != NULL,
               "Only scalar finite elements are supported");
}

template<const int T_VDIM, const int T_ND, const int T_NQ>
void QuadratureInterpolator::Eval2D(
   const int NE,
   const int vdim,
   const DofToQuad &maps,
   const Vector &e_vec,
   Vector &q_val,
   Vector &q_der,
   Vector &q_det,
   const int eval_flags)
{
   const int nd = maps.ndof;
   const int nq = maps.nqpt;
   const int ND = T_ND ? T_ND : nd;
   const int NQ = T_NQ ? T_NQ : nq;
   const int VDIM = T_VDIM ? T_VDIM : vdim;
   MFEM_VERIFY(ND <= MAX_ND2D, "");
   MFEM_VERIFY(NQ <= MAX_NQ2D, "");
   MFEM_VERIFY(VDIM == 2 || !(eval_flags & DETERMINANTS), "");
   auto B = Reshape(maps.B.Read(), NQ, ND);
   auto G = Reshape(maps.G.Read(), NQ, 2, ND);
   auto E = Reshape(e_vec.Read(), ND, VDIM, NE);
   auto val = Reshape(q_val.Write(), NQ, VDIM, NE);
   auto der = Reshape(q_der.Write(), NQ, VDIM, 2, NE);
   auto det = Reshape(q_det.Write(), NQ, NE);
   MFEM_FORALL(e, NE,
   {
      const int ND = T_ND ? T_ND : nd;
      const int NQ = T_NQ ? T_NQ : nq;
      const int VDIM = T_VDIM ? T_VDIM : vdim;
      constexpr int max_ND = T_ND ? T_ND : MAX_ND2D;
      constexpr int max_VDIM = T_VDIM ? T_VDIM : MAX_VDIM2D;
      double s_E[max_VDIM*max_ND];
      for (int d = 0; d < ND; d++)
      {
         for (int c = 0; c < VDIM; c++)
         {
            s_E[c+d*VDIM] = E(d,c,e);
         }
      }
      for (int q = 0; q < NQ; ++q)
      {
         if (eval_flags & VALUES)
         {
            double ed[max_VDIM];
            for (int c = 0; c < VDIM; c++) { ed[c] = 0.0; }
            for (int d = 0; d < ND; ++d)
            {
               const double b = B(q,d);
               for (int c = 0; c < VDIM; c++) { ed[c] += b*s_E[c+d*VDIM]; }
            }
            for (int c = 0; c < VDIM; c++) { val(q,c,e) = ed[c]; }
         }
         if ((eval_flags & DERIVATIVES) || (eval_flags & DETERMINANTS))
         {
            // use MAX_VDIM2D to avoid "subscript out of range" warnings
            double D[MAX_VDIM2D*2];
            for (int i = 0; i < 2*VDIM; i++) { D[i] = 0.0; }
            for (int d = 0; d < ND; ++d)
            {
               const double wx = G(q,0,d);
               const double wy = G(q,1,d);
               for (int c = 0; c < VDIM; c++)
               {
                  double s_e = s_E[c+d*VDIM];
                  D[c+VDIM*0] += s_e * wx;
                  D[c+VDIM*1] += s_e * wy;
               }
            }
            if (eval_flags & DERIVATIVES)
            {
               for (int c = 0; c < VDIM; c++)
               {
                  der(q,c,0,e) = D[c+VDIM*0];
                  der(q,c,1,e) = D[c+VDIM*1];
               }
            }
            if (VDIM == 2 && (eval_flags & DETERMINANTS))
            {
               // The check (VDIM == 2) should eliminate this block when VDIM is
               // known at compile time and (VDIM != 2).
               det(q,e) = D[0]*D[3] - D[1]*D[2];
            }
         }
      }
   });
}

template<const int T_VDIM, const int T_ND, const int T_NQ>
void QuadratureInterpolator::Eval3D(
   const int NE,
   const int vdim,
   const DofToQuad &maps,
   const Vector &e_vec,
   Vector &q_val,
   Vector &q_der,
   Vector &q_det,
   const int eval_flags)
{
   const int nd = maps.ndof;
   const int nq = maps.nqpt;
   const int ND = T_ND ? T_ND : nd;
   const int NQ = T_NQ ? T_NQ : nq;
   const int VDIM = T_VDIM ? T_VDIM : vdim;
   MFEM_VERIFY(ND <= MAX_ND3D, "");
   MFEM_VERIFY(NQ <= MAX_NQ3D, "");
   MFEM_VERIFY(VDIM == 3 || !(eval_flags & DETERMINANTS), "");
   auto B = Reshape(maps.B.Read(), NQ, ND);
   auto G = Reshape(maps.G.Read(), NQ, 3, ND);
   auto E = Reshape(e_vec.Read(), ND, VDIM, NE);
   auto val = Reshape(q_val.Write(), NQ, VDIM, NE);
   auto der = Reshape(q_der.Write(), NQ, VDIM, 3, NE);
   auto det = Reshape(q_det.Write(), NQ, NE);
   MFEM_FORALL(e, NE,
   {
      const int ND = T_ND ? T_ND : nd;
      const int NQ = T_NQ ? T_NQ : nq;
      const int VDIM = T_VDIM ? T_VDIM : vdim;
      constexpr int max_ND = T_ND ? T_ND : MAX_ND3D;
      constexpr int max_VDIM = T_VDIM ? T_VDIM : MAX_VDIM3D;
      double s_E[max_VDIM*max_ND];
      for (int d = 0; d < ND; d++)
      {
         for (int c = 0; c < VDIM; c++)
         {
            s_E[c+d*VDIM] = E(d,c,e);
         }
      }
      for (int q = 0; q < NQ; ++q)
      {
         if (eval_flags & VALUES)
         {
            double ed[max_VDIM];
            for (int c = 0; c < VDIM; c++) { ed[c] = 0.0; }
            for (int d = 0; d < ND; ++d)
            {
               const double b = B(q,d);
               for (int c = 0; c < VDIM; c++) { ed[c] += b*s_E[c+d*VDIM]; }
            }
            for (int c = 0; c < VDIM; c++) { val(q,c,e) = ed[c]; }
         }
         if ((eval_flags & DERIVATIVES) || (eval_flags & DETERMINANTS))
         {
            // use MAX_VDIM3D to avoid "subscript out of range" warnings
            double D[MAX_VDIM3D*3];
            for (int i = 0; i < 3*VDIM; i++) { D[i] = 0.0; }
            for (int d = 0; d < ND; ++d)
            {
               const double wx = G(q,0,d);
               const double wy = G(q,1,d);
               const double wz = G(q,2,d);
               for (int c = 0; c < VDIM; c++)
               {
                  double s_e = s_E[c+d*VDIM];
                  D[c+VDIM*0] += s_e * wx;
                  D[c+VDIM*1] += s_e * wy;
                  D[c+VDIM*2] += s_e * wz;
               }
            }
            if (eval_flags & DERIVATIVES)
            {
               for (int c = 0; c < VDIM; c++)
               {
                  der(q,c,0,e) = D[c+VDIM*0];
                  der(q,c,1,e) = D[c+VDIM*1];
                  der(q,c,2,e) = D[c+VDIM*2];
               }
            }
            if (VDIM == 3 && (eval_flags & DETERMINANTS))
            {
               // The check (VDIM == 3) should eliminate this block when VDIM is
               // known at compile time and (VDIM != 3).
               det(q,e) = D[0] * (D[4] * D[8] - D[5] * D[7]) +
                          D[3] * (D[2] * D[7] - D[1] * D[8]) +
                          D[6] * (D[1] * D[5] - D[2] * D[4]);
            }
         }
      }
   });
}

void QuadratureInterpolator::Mult(
   const Vector &e_vec, unsigned eval_flags,
   Vector &q_val, Vector &q_der, Vector &q_det) const
{
   const int ne = fespace->GetNE();
   if (ne == 0) { return; }
   const int vdim = fespace->GetVDim();
   const int dim = fespace->GetMesh()->Dimension();
   const FiniteElement *fe = fespace->GetFE(0);
   const IntegrationRule *ir =
      IntRule ? IntRule : &qspace->GetElementIntRule(0);
   const DofToQuad &maps = fe->GetDofToQuad(*ir, DofToQuad::FULL);
   const int nd = maps.ndof;
   const int nq = maps.nqpt;
   void (*eval_func)(
      const int NE,
      const int vdim,
      const DofToQuad &maps,
      const Vector &e_vec,
      Vector &q_val,
      Vector &q_der,
      Vector &q_det,
      const int eval_flags) = NULL;
   if (vdim == 1)
   {
      if (dim == 2)
      {
         switch (100*nd + nq)
         {
            // Q0
            case 101: eval_func = &Eval2D<1,1,1>; break;
            case 104: eval_func = &Eval2D<1,1,4>; break;
            // Q1
            case 404: eval_func = &Eval2D<1,4,4>; break;
            case 409: eval_func = &Eval2D<1,4,9>; break;
            // Q2
            case 909: eval_func = &Eval2D<1,9,9>; break;
            case 916: eval_func = &Eval2D<1,9,16>; break;
            // Q3
            case 1616: eval_func = &Eval2D<1,16,16>; break;
            case 1625: eval_func = &Eval2D<1,16,25>; break;
            case 1636: eval_func = &Eval2D<1,16,36>; break;
            // Q4
            case 2525: eval_func = &Eval2D<1,25,25>; break;
            case 2536: eval_func = &Eval2D<1,25,36>; break;
            case 2549: eval_func = &Eval2D<1,25,49>; break;
            case 2564: eval_func = &Eval2D<1,25,64>; break;
         }
         if (nq >= 100 || !eval_func)
         {
            eval_func = &Eval2D<1>;
         }
      }
      else if (dim == 3)
      {
         switch (1000*nd + nq)
         {
            // Q0
            case 1001: eval_func = &Eval3D<1,1,1>; break;
            case 1008: eval_func = &Eval3D<1,1,8>; break;
            // Q1
            case 8008: eval_func = &Eval3D<1,8,8>; break;
            case 8027: eval_func = &Eval3D<1,8,27>; break;
            // Q2
            case 27027: eval_func = &Eval3D<1,27,27>; break;
            case 27064: eval_func = &Eval3D<1,27,64>; break;
            // Q3
            case 64064: eval_func = &Eval3D<1,64,64>; break;
            case 64125: eval_func = &Eval3D<1,64,125>; break;
            case 64216: eval_func = &Eval3D<1,64,216>; break;
            // Q4
            case 125125: eval_func = &Eval3D<1,125,125>; break;
            case 125216: eval_func = &Eval3D<1,125,216>; break;
         }
         if (nq >= 1000 || !eval_func)
         {
            eval_func = &Eval3D<1>;
         }
      }
   }
   else if (vdim == dim)
   {
      if (dim == 2)
      {
         switch (100*nd + nq)
         {
            // Q1
            case 404: eval_func = &Eval2D<2,4,4>; break;
            case 409: eval_func = &Eval2D<2,4,9>; break;
            // Q2
            case 909: eval_func = &Eval2D<2,9,9>; break;
            case 916: eval_func = &Eval2D<2,9,16>; break;
            // Q3
            case 1616: eval_func = &Eval2D<2,16,16>; break;
            case 1625: eval_func = &Eval2D<2,16,25>; break;
            case 1636: eval_func = &Eval2D<2,16,36>; break;
            // Q4
            case 2525: eval_func = &Eval2D<2,25,25>; break;
            case 2536: eval_func = &Eval2D<2,25,36>; break;
            case 2549: eval_func = &Eval2D<2,25,49>; break;
            case 2564: eval_func = &Eval2D<2,25,64>; break;
         }
         if (nq >= 100 || !eval_func)
         {
            eval_func = &Eval2D<2>;
         }
      }
      else if (dim == 3)
      {
         switch (1000*nd + nq)
         {
            // Q1
            case 8008: eval_func = &Eval3D<3,8,8>; break;
            case 8027: eval_func = &Eval3D<3,8,27>; break;
            // Q2
            case 27027: eval_func = &Eval3D<3,27,27>; break;
            case 27064: eval_func = &Eval3D<3,27,64>; break;
            // Q3
            case 64064: eval_func = &Eval3D<3,64,64>; break;
            case 64125: eval_func = &Eval3D<3,64,125>; break;
            case 64216: eval_func = &Eval3D<3,64,216>; break;
            // Q4
            case 125125: eval_func = &Eval3D<3,125,125>; break;
            case 125216: eval_func = &Eval3D<3,125,216>; break;
         }
         if (nq >= 1000 || !eval_func)
         {
            eval_func = &Eval3D<3>;
         }
      }
   }
   if (eval_func)
   {
      eval_func(ne, vdim, maps, e_vec, q_val, q_der, q_det, eval_flags);
   }
   else
   {
      MFEM_ABORT("case not supported yet");
   }
}

void QuadratureInterpolator::MultTranspose(
   unsigned eval_flags, const Vector &q_val, const Vector &q_der,
   Vector &e_vec) const
{
   MFEM_ABORT("this method is not implemented yet");
}

} // namespace mfem
