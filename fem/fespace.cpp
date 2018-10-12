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
   int GeomType = mesh->GetElementBaseGeometry(i);
   return fec->FiniteElementForGeometry(GeomType)->GetOrder();
}

int FiniteElementSpace::GetFaceOrder(int i) const
{
   int GeomType = mesh->GetFaceBaseGeometry(i);
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

   if (!lfes->GetNE())
   {
      R->Finalize();
      return R;
   }

   const FiniteElement *h_fe = this -> GetFE (0);
   const FiniteElement *l_fe = lfes -> GetFE (0);
   IsoparametricTransformation T;
   T.SetIdentityTransformation(h_fe->GetGeomType());
   h_fe->Project(*l_fe, T, loc_restr);

   for (int i = 0; i < mesh -> GetNE(); i++)
   {
      this -> GetElementDofs (i, h_dofs);
      lfes -> GetElementDofs (i, l_dofs);

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

void
FiniteElementSpace::GetEntityDofs(int entity, int index, Array<int> &dofs) const
{
   switch (entity)
   {
      case 0: GetVertexDofs(index, dofs); break;
      case 1: GetEdgeDofs(index, dofs); break;
      case 2: GetFaceDofs(index, dofs); break;
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
      const NCMesh::NCList &list = (entity > 1) ? mesh->ncmesh->GetFaceList()
                                   /*        */ : mesh->ncmesh->GetEdgeList();
      if (!list.masters.size()) { continue; }

      IsoparametricTransformation T;
      if (entity > 1) { T.SetFE(&QuadrilateralFE); }
      else { T.SetFE(&SegmentFE); }

      int geom = (entity > 1) ? Geometry::SQUARE : Geometry::SEGMENT;
      const FiniteElement* fe = fec->FiniteElementForGeometry(geom);
      if (!fe) { continue; }

      Array<int> master_dofs, slave_dofs;
      DenseMatrix I(fe->GetDof());

      // loop through all master edges/faces, constrain their slave edges/faces
      for (unsigned mi = 0; mi < list.masters.size(); mi++)
      {
         const NCMesh::Master &master = list.masters[mi];
         GetEntityDofs(entity, master.index, master_dofs);
         if (!master_dofs.Size()) { continue; }

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

SparseMatrix *FiniteElementSpace::RefinementMatrix_main(
   const int coarse_ndofs, const Table &coarse_elem_dof,
   const DenseTensor &localP) const
{
   MFEM_VERIFY(mesh->GetLastOperation() == Mesh::REFINE, "");

   Array<int> dofs, coarse_dofs, coarse_vdofs;
   Vector row;

   const int coarse_ldof = localP.SizeJ();
   const int fine_ldof = localP.SizeI();
   SparseMatrix *P = new SparseMatrix(GetVSize(), coarse_ndofs*vdim,
                                      coarse_ldof);

   Array<int> mark(P->Height());
   mark = 0;

   const CoarseFineTransformations &rtrans = mesh->GetRefinementTransforms();

   for (int k = 0; k < mesh->GetNE(); k++)
   {
      const Embedding &emb = rtrans.embeddings[k];
      const DenseMatrix &lP = localP(emb.matrix);

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
   return P;
}

void FiniteElementSpace::GetLocalRefinementMatrices(DenseTensor &localP) const
{
   int geom = mesh->GetElementBaseGeometry(); // assuming the same geom
   const FiniteElement *fe = fec->FiniteElementForGeometry(geom);

   const CoarseFineTransformations &rtrans = mesh->GetRefinementTransforms();

   int nmat = rtrans.point_matrices.SizeK();
   int ldof = fe->GetDof(); // assuming the same FE everywhere

   IsoparametricTransformation isotr;
   isotr.SetIdentityTransformation(geom);

   // calculate local interpolation matrices for all refinement types
   localP.SetSize(ldof, ldof, nmat);
   for (int i = 0; i < nmat; i++)
   {
      isotr.GetPointMat() = rtrans.point_matrices(i);
      isotr.FinalizeTransformation();
      fe->GetLocalInterpolation(isotr, localP(i));
   }
}

SparseMatrix* FiniteElementSpace::RefinementMatrix(int old_ndofs,
                                                   const Table* old_elem_dof)
{
   MFEM_VERIFY(ndofs >= old_ndofs, "Previous space is not coarser.");

   DenseTensor localP;
   GetLocalRefinementMatrices(localP);

   return RefinementMatrix_main(old_ndofs, *old_elem_dof, localP);
}

FiniteElementSpace::RefinementOperator::RefinementOperator
(const FiniteElementSpace* fespace, Table* old_elem_dof, int old_ndofs)
   : fespace(fespace)
   , old_elem_dof(old_elem_dof)
{
   MFEM_VERIFY(fespace->GetNDofs() >= old_ndofs,
               "Previous space is not coarser.");

   width = old_ndofs * fespace->GetVDim();
   height = fespace->GetVSize();

   fespace->GetLocalRefinementMatrices(localP);
}

FiniteElementSpace::RefinementOperator::RefinementOperator(
   const FiniteElementSpace *fespace, const FiniteElementSpace *coarse_fes)
   : Operator(fespace->GetVSize(), coarse_fes->GetVSize()),
     fespace(fespace), old_elem_dof(NULL)
{
   fespace->GetLocalRefinementMatrices(*coarse_fes, localP);
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
      const DenseMatrix &lP = localP(emb.matrix);

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

void InvertLinearTrans(IsoparametricTransformation &trans,
                       const DenseMatrix &invdfdx,
                       const IntegrationPoint &pt, Vector &x)
{
   // invert a linear transform with one Newton step
   IntegrationPoint p0;
   p0.Set3(0, 0, 0);
   trans.Transform(p0, x);

   double store[3];
   Vector v(store, x.Size());
   pt.Get(v, x.Size());
   v -= x;

   invdfdx.Mult(v, x);
}

void FiniteElementSpace::GetLocalDerefinementMatrices(DenseTensor &localR) const
{
   int geom = mesh->GetElementBaseGeometry(); // assuming the same geom
   const FiniteElement *fe = fec->FiniteElementForGeometry(geom);
   const IntegrationRule &nodes = fe->GetNodes();

   const CoarseFineTransformations &dtrans =
      mesh->ncmesh->GetDerefinementTransforms();

   int nmat = dtrans.point_matrices.SizeK();
   int ldof = fe->GetDof();
   int dim = mesh->Dimension();

   LinearFECollection linfec;
   IsoparametricTransformation isotr;
   isotr.SetFE(linfec.FiniteElementForGeometry(geom));

   DenseMatrix invdfdx(dim);
   IntegrationPoint ipt;
   Vector pt(&ipt.x, dim), shape(ldof);

   // calculate local restriction matrices for all refinement types
   localR.SetSize(ldof, ldof, nmat);
   for (int i = 0; i < nmat; i++)
   {
      DenseMatrix &lR = localR(i);
      lR = infinity(); // marks invalid rows

      isotr.GetPointMat() = dtrans.point_matrices(i);
      isotr.FinalizeTransformation();
      isotr.SetIntPoint(&nodes[0]);
      CalcInverse(isotr.Jacobian(), invdfdx);

      for (int j = 0; j < nodes.Size(); j++)
      {
         InvertLinearTrans(isotr, invdfdx, nodes[j], pt);
         if (Geometries.CheckPoint(geom, ipt)) // do we need an epsilon here?
         {
            IntegrationPoint ip;
            ip.Set(pt, dim);
            MFEM_ASSERT(dynamic_cast<const NodalFiniteElement*>(fe),
                        "only nodal FEs are implemented");
            fe->CalcShape(ip, shape); // TODO: H(curl), etc.?
            lR.SetRow(j, shape);
         }
      }
      lR.Threshold(1e-12);
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

   DenseTensor localR;
   GetLocalDerefinementMatrices(localR);

   SparseMatrix *R = new SparseMatrix(ndofs*vdim, old_ndofs*vdim,
                                      localR.SizeI());
   Array<int> mark(R->Height());
   mark = 0;

   const CoarseFineTransformations &dtrans =
      mesh->ncmesh->GetDerefinementTransforms();

   for (int k = 0; k < dtrans.embeddings.Size(); k++)
   {
      const Embedding &emb = dtrans.embeddings[k];
      DenseMatrix &lR = localR(emb.matrix);

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
            }
         }
      }
   }

   MFEM_ASSERT(mark.Sum() == R->Height(), "Not all rows of R set.");
   return R;
}

void FiniteElementSpace::GetLocalRefinementMatrices(
   const FiniteElementSpace &coarse_fes, DenseTensor &localP) const
{
   // Assumptions: see the declaration of the method.

   int fine_geom = mesh->GetElementBaseGeometry(0);
   const FiniteElement *fine_fe = fec->FiniteElementForGeometry(fine_geom);
   const FiniteElement *coarse_fe = coarse_fes.GetFE(0);

   const CoarseFineTransformations &rtrans = mesh->GetRefinementTransforms();

   int nmat = rtrans.point_matrices.SizeK();

   IsoparametricTransformation isotr;
   isotr.SetIdentityTransformation(fine_geom);

   // Calculate the local interpolation matrices for all refinement types
   localP.SetSize(fine_fe->GetDof(), coarse_fe->GetDof(), nmat);
   for (int i = 0; i < nmat; i++)
   {
      isotr.GetPointMat() = rtrans.point_matrices(i);
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
   MFEM_ASSERT(!NURBSext, "internal error");

   elem_dof = NULL;
   bdrElem_dof = NULL;

   nvdofs = mesh->GetNV() * fec->DofForGeometry(Geometry::POINT);

   if ( mesh->Dimension() > 1 )
   {
      nedofs = mesh->GetNEdges() * fec->DofForGeometry(Geometry::SEGMENT);
   }
   else
   {
      nedofs = 0;
   }

   ndofs = 0;
   nfdofs = 0;
   nbdofs = 0;
   bdofs = NULL;
   fdofs = NULL;
   cP = NULL;
   cR = NULL;
   cP_is_set = false;
   // Th is initialized/destroyed before this method is called.

   if (mesh->Dimension() == 3 && mesh->GetNE())
   {
      // Here we assume that all faces in the mesh have the same base
      // geometry -- the base geometry of the 0-th face element.
      // The class Mesh assumes the same inside GetFaceBaseGeometry(...).
      // Thus we do not need to generate all the faces in the mesh
      // if we do not need them.
      int fdof = fec->DofForGeometry(mesh->GetFaceBaseGeometry(0));
      if (fdof > 0)
      {
         fdofs = new int[mesh->GetNFaces()+1];
         fdofs[0] = 0;
         for (int i = 0; i < mesh->GetNFaces(); i++)
         {
            nfdofs += fdof;
            // nfdofs += fec->DofForGeometry(mesh->GetFaceBaseGeometry(i));
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
         int geom = mesh->GetElementBaseGeometry(i);
         nbdofs += fec->DofForGeometry(geom);
         bdofs[i+1] = nbdofs;
      }
   }

   ndofs = nvdofs + nedofs + nfdofs + nbdofs;

   // Do not build elem_dof Table here: in parallel it has to be constructed
   // later.
}

void FiniteElementSpace::GetElementDofs (int i, Array<int> &dofs) const
{
   if (elem_dof)
   {
      elem_dof -> GetRow (i, dofs);
   }
   else
   {
      Array<int> V, E, Eo, F, Fo;
      int k, j, nv, ne, nf, nb, nfd, nd;
      int *ind, dim;

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
      int k, j, nv, ne, nf, nd, iF, oF;
      int *ind, dim;

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
         ind = (fec->DofOrderForOrientation(
                   mesh->GetBdrElementBaseGeometry(i), oF));
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
   int i, int geom_type) const
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
      DenseTensor localP;
      GetLocalRefinementMatrices(coarse_fes, localP);
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
         // If periodic BCs are given write connectivity
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


} // namespace mfem
