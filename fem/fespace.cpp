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

static void AddDependencies(SparseMatrix& deps, Array<int>& master_dofs,
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

static bool DofFinalizable(int dof, const Array<bool>& finalized,
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

/** This is a helper function to get edge (type == 0) or face (type == 1) DOFs.
    The function is aware of ghost edges/faces in parallel, for which an empty
    DOF list is returned. */
void FiniteElementSpace::GetEdgeFaceDofs(int type, int index, Array<int> &dofs)
const
{
   dofs.SetSize(0);
   if (type)
   {
      if (index < mesh->GetNFaces()) { GetFaceDofs(index, dofs); }
   }
   else
   {
      if (index < mesh->GetNEdges()) { GetEdgeDofs(index, dofs); }
   }
}

void FiniteElementSpace::GetConformingInterpolation() const
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
   for (int type = 0; type <= 1; type++)
   {
      const NCMesh::NCList &list = type ? mesh->ncmesh->GetFaceList()
                                   /**/ : mesh->ncmesh->GetEdgeList();
      if (!list.masters.size()) { continue; }

      IsoparametricTransformation T;
      if (type) { T.SetFE(&QuadrilateralFE); }
      else { T.SetFE(&SegmentFE); }

      int geom = type ? Geometry::SQUARE : Geometry::SEGMENT;
      const FiniteElement* fe = fec->FiniteElementForGeometry(geom);
      if (!fe) { continue; }

      Array<int> master_dofs, slave_dofs;
      DenseMatrix I(fe->GetDof());

      // loop through all master edges/faces, constrain their slave edges/faces
      for (unsigned mi = 0; mi < list.masters.size(); mi++)
      {
         const NCMesh::Master &master = list.masters[mi];
         GetEdgeFaceDofs(type, master.index, master_dofs);
         if (!master_dofs.Size()) { continue; }

         for (int si = master.slaves_begin; si < master.slaves_end; si++)
         {
            const NCMesh::Slave &slave = list.slaves[si];
            GetEdgeFaceDofs(type, slave.index, slave_dofs);
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
   if (!cP_is_set) { GetConformingInterpolation(); }
   return cP;
}

const SparseMatrix* FiniteElementSpace::GetConformingRestriction() const
{
   if (Conforming()) { return NULL; }
   if (!cP_is_set) { GetConformingInterpolation(); }
   return cR;
}

int FiniteElementSpace::GetNConformingDofs() const
{
   const SparseMatrix* P = GetConformingProlongation();
   return P ? (P->Width() / vdim) : ndofs;
}

SparseMatrix* FiniteElementSpace::RefinementMatrix(int old_ndofs,
                                                   const Table* old_elem_dof)
{
   MFEM_VERIFY(mesh->GetLastOperation() == Mesh::REFINE, "");
   MFEM_VERIFY(ndofs >= old_ndofs, "Previous space is not coarser.");

   Array<int> dofs, old_dofs, old_vdofs;
   Vector row;

   const CoarseFineTransformations &rtrans = mesh->GetRefinementTransforms();

   int geom = mesh->GetElementBaseGeometry(); // assuming the same geom
   const FiniteElement *fe = fec->FiniteElementForGeometry(geom);

   IsoparametricTransformation isotr;
   isotr.SetIdentityTransformation(geom);

   int nmat = rtrans.point_matrices.SizeK();
   int ldof = fe->GetDof(); // assuming the same FE everywhere

   // calculate local interpolation matrices for all refinement types
   DenseTensor localP(ldof, ldof, nmat);
   for (int i = 0; i < nmat; i++)
   {
      isotr.GetPointMat() = rtrans.point_matrices(i);
      isotr.FinalizeTransformation();
      fe->GetLocalInterpolation(isotr, localP(i));
   }

   SparseMatrix *P = new SparseMatrix(ndofs*vdim, old_ndofs*vdim, ldof);

   Array<int> mark(P->Height());
   mark = 0;
   for (int k = 0; k < mesh->GetNE(); k++)
   {
      const Embedding &emb = rtrans.embeddings[k];
      DenseMatrix &lP = localP(emb.matrix);

      elem_dof->GetRow(k, dofs);
      old_elem_dof->GetRow(emb.parent, old_dofs);

      for (int vd = 0; vd < vdim; vd++)
      {
         old_dofs.Copy(old_vdofs);
         DofsToVDofs(vd, old_vdofs, old_ndofs);

         for (int i = 0; i < ldof; i++)
         {
            int r = DofToVDof(dofs[i], vd);
            int m = (r >= 0) ? r : (-1 - r);

            if (!mark[m])
            {
               lP.GetRow(i, row);
               P->SetRow(r, old_vdofs, row); // TODO: profiler says this is slow!
               mark[m] = 1;
            }
         }
      }
   }

   MFEM_ASSERT(mark.Sum() == P->Height(), "Not all rows of P set.");
   return P;
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

void FiniteElementSpace::GetLocalDerefinementMatrices(
   int geom, const CoarseFineTransformations &dt, DenseTensor &localR)
{
   const FiniteElement *fe = fec->FiniteElementForGeometry(geom);
   const IntegrationRule &nodes = fe->GetNodes();

   LinearFECollection linfec;
   IsoparametricTransformation isotr;
   isotr.SetFE(linfec.FiniteElementForGeometry(geom));

   int nmat = dt.point_matrices.SizeK();
   int ldof = fe->GetDof();
   int dim = mesh->Dimension();

   DenseMatrix invdfdx(dim);
   IntegrationPoint ipt;
   Vector pt(&ipt.x, dim), shape(ldof);

   // calculate local restriction matrices for all refinement types
   localR.SetSize(ldof, ldof, nmat);
   for (int i = 0; i < nmat; i++)
   {
      DenseMatrix &lR = localR(i);
      lR = numeric_limits<double>::infinity(); // marks invalid rows

      isotr.GetPointMat() = dt.point_matrices(i);
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
            fe->CalcShape(ip, shape); // TODO: H(curl), etc.?
            MFEM_ASSERT(dynamic_cast<const NodalFiniteElement*>(fe),
                        "only nodal FEs are implemented");
            lR.SetRow(j, shape);
         }
      }
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

   const CoarseFineTransformations &dtrans =
      mesh->ncmesh->GetDerefinementTransforms();

   int geom = mesh->ncmesh->GetElementGeometry();
   int ldof = fec->FiniteElementForGeometry(geom)->GetDof();

   DenseTensor localR;
   GetLocalDerefinementMatrices(geom, dtrans, localR);

   SparseMatrix *R = new SparseMatrix(ndofs*vdim, old_ndofs*vdim, ldof);

   Array<int> mark(R->Height());
   mark = 0;
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
            if (lR(i, 0) == numeric_limits<double>::infinity()) { continue; }

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

FiniteElementSpace::FiniteElementSpace(Mesh *mesh,
                                       const FiniteElementCollection *fec,
                                       int vdim, int ordering)
{
   this->mesh = mesh;
   this->fec = fec;
   this->vdim = vdim;
   this->ordering = (Ordering::Type) ordering;

   elem_dof = NULL;
   sequence = mesh->GetSequence();

   tensor_offsets = NULL;
   tensor_indices = NULL;

   const NURBSFECollection *nurbs_fec =
      dynamic_cast<const NURBSFECollection *>(fec);
   if (nurbs_fec)
   {
      if (!mesh->NURBSext)
      {
         mfem_error("FiniteElementSpace::FiniteElementSpace :\n"
                    "   NURBS FE space requires NURBS mesh.");
      }
      else
      {
         int Order = nurbs_fec->GetOrder();
         if (mesh->NURBSext->GetOrder() == Order)
         {
            NURBSext = mesh->NURBSext;
            own_ext = 0;
         }
         else
         {
            NURBSext = new NURBSExtension(mesh->NURBSext, Order);
            own_ext = 1;
         }
         UpdateNURBS();
         cP = cR = NULL;
         cP_is_set = false;
         T = NULL;
         own_T = true;
      }
   }
   else
   {
      NURBSext = NULL;
      own_ext = 0;
      Construct();
   }

   BuildElementToDofTable();

   // Calculate the local unrolled size.
   nldofs = 0;
   for (int e = 0; e < GetNE(); e++)
   {
      const FiniteElement *fe = GetFE(e);
      nldofs += fe->GetDof();
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
   int i;

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
   T = NULL;
   own_T = true;

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
         for (i = 0; i < mesh->GetNFaces(); i++)
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
      for (i = 0; i < mesh->GetNE(); i++)
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
   if (own_T) { delete T; }

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

   delete tensor_offsets;
   delete tensor_indices;
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

   Table* old_elem_dof= NULL;
   int old_ndofs;

   // save old DOF table
   if (want_transform)
   {
      old_elem_dof = elem_dof;
      elem_dof = NULL;
      old_ndofs = ndofs;
   }

   Destroy();
   Construct(); // sets T to NULL, own_T to true
   BuildElementToDofTable();

   if (want_transform)
   {
      // calculate appropriate GridFunction transformation
      switch (mesh->GetLastOperation())
      {
         case Mesh::REFINE:
         {
            T = RefinementMatrix(old_ndofs, old_elem_dof);
            break;
         }

         case Mesh::DEREFINE:
         {
            GetConformingInterpolation();
            T = DerefinementMatrix(old_ndofs, old_elem_dof);
            if (cP && cR)
            {
               T = new TripleProductOperator(cP, cR, T, false, false, true);
            }
            break;
         }

         default:
            break; // T stays NULL
      }
   }

   delete old_elem_dof;
}

void FiniteElementSpace::Save(std::ostream &out) const
{
   out << "FiniteElementSpace\n"
       << "FiniteElementCollection: " << fec->Name() << '\n'
       << "VDim: " << vdim << '\n'
       << "Ordering: " << ordering << '\n';
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

static void BuildDofMaps(FiniteElementSpace *fes,
                         Array<int> *&off, Array<int> *&ind)
{
   const int global_size = fes->GetVSize();
   const int local_size = fes->GetLocalVSize();
   const int vdim = fes->GetVDim();

   // Now we can allocate and fill the global map
   off = new Array<int>(global_size + 1);
   ind = new Array<int>(local_size);

   Array<int> &offsets = *off;
   Array<int> &indices = *ind;

   Array<int> global_map(local_size);
   Array<int> elem_vdof;

   int offset = 0;
   for (int e = 0; e < fes->GetNE(); e++)
   {
      const FiniteElement *fe = fes->GetFE(e);
      const int dofs = fe->GetDof();
      const TensorBasisElement *tfe = dynamic_cast<const TensorBasisElement *>(fe);
      const Array<int> &dof_map = tfe->GetDofMap();

      fes->GetElementVDofs(e, elem_vdof);

      for (int vd = 0; vd < vdim; vd++)
         for (int i = 0; i < dofs; i++)
         {
            global_map[offset + dofs*vd + i] = elem_vdof[dofs*vd + dof_map[i]];
         }
      offset += dofs * vdim;
   }

   // global_map[i] = index in global vector for local dof i
   // NOTE: multiple i values will yield same global_map[i] for shared DOF.

   // We want to now invert this map so we have indices[j] = (local dof for global dof j).

   // Zero the offset vector
   offsets = 0;

   // Keep track of how many local dof point to its global dof
   // Count how many times each dof gets hit
   for (int i = 0; i < local_size; i++)
   {
      const int g = global_map[i];
      ++offsets[g + 1];
   }
   // Aggregate the offsets
   for (int i = 1; i <= global_size; i++)
   {
      offsets[i] += offsets[i - 1];
   }

   for (int i = 0; i < local_size; i++)
   {
      const int g = global_map[i];
      indices[offsets[g]++] = i;
   }

   // Shift the offset vector back by one, since it was used as a
   // counter above.
   for (int i = global_size; i > 0; i--)
   {
      offsets[i] = offsets[i - 1];
   }
   offsets[0] = 0;
}

void FiniteElementSpace::ToLocalVector(const Vector &v, Vector &V)
{
   if ((tensor_indices == NULL) || (tensor_offsets == NULL))
   {
      BuildDofMaps(this, tensor_offsets, tensor_indices);
   }

   V.SetSize(tensor_indices->Size());

   const int size = v.Size();
   const int *offsets = tensor_offsets->GetData();
   const int *indices = tensor_indices->GetData();

   for (int i = 0; i < size; i++)
   {
      const int offset = offsets[i];
      const int next_offset = offsets[i + 1];
      const double dof_value = v(i);
      for (int j = offset; j < next_offset; j++)
      {
         V(indices[j]) = dof_value;
      }
   }
}

void FiniteElementSpace::ToGlobalVector(const Vector &V, Vector &v)
{
   if ((tensor_indices == NULL) || (tensor_offsets == NULL))
   {
      BuildDofMaps(this, tensor_offsets, tensor_indices);
   }

   v.SetSize(GetVSize());

   const int size = v.Size();
   const int *offsets = tensor_offsets->GetData();
   const int *indices = tensor_indices->GetData();

   for (int i = 0; i < size; i++)
   {
      const int offset = offsets[i];
      const int next_offset = offsets[i + 1];
      double dof_value = 0;
      for (int j = offset; j < next_offset; j++)
      {
         dof_value += V(indices[j]);
      }
      v(i) = dof_value;
   }
}

} // namespace mfem
