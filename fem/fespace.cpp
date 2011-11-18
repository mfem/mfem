// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.googlecode.com.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

// Implementation of FiniteElementSpace

#include "fem.hpp"
#include <math.h>

int FiniteElementSpace::GetOrder(int i) const
{
   int GeomType = mesh->GetElementBaseGeometry(i);
   return fec->FiniteElementForGeometry(GeomType)->GetOrder();
}


void FiniteElementSpace::DofsToVDofs (Array<int> &dofs) const
{
   int i, j, size;

   if (vdim == 1)  return;

   size = dofs.Size();
   dofs.SetSize (size * vdim);

   switch(ordering)
   {
   case Ordering::byNODES:
      for (i = 1; i < vdim; i++)
         for (j = 0; j < size; j++)
            if (dofs[j] < 0)
               dofs[size * i + j] = -1 - ( ndofs * i + (-1-dofs[j]) );
            else
               dofs[size * i + j] = ndofs * i + dofs[j];
      break;

   case Ordering::byVDIM:
      for (i = vdim-1; i >= 0; i--)
         for (j = 0; j < size; j++)
            if (dofs[j] < 0)
               dofs[size * i + j] = -1 - ( (-1-dofs[j]) * vdim + i );
            else
               dofs[size * i + j] = dofs[j] * vdim + i;
      break;
   }
}

void FiniteElementSpace::DofsToVDofs(int vd, Array<int> &dofs) const
{
   if (vdim == 1)
      return;
   if (ordering == Ordering::byNODES)
   {
      for (int i = 0; i < dofs.Size(); i++)
      {
         int dof = dofs[i];
         if (dof < 0)
            dofs[i] = -1 - ((-1-dof) + vd * ndofs);
         else
            dofs[i] = dof + vd * ndofs;
      }
   }
   else
   {
      for (int i = 0; i < dofs.Size(); i++)
      {
         int dof = dofs[i];
         if (dof < 0)
            dofs[i] = -1 - ((-1-dof) * vdim + vd);
         else
            dofs[i] = dof * vdim + vd;
      }
   }
}

int FiniteElementSpace::DofToVDof (int dof, int vd) const
{
   if (vdim == 1)
      return dof;
   if (ordering == Ordering::byNODES)
   {
      if (dof < 0)
         return -1 - ((-1-dof) + vd * ndofs);
      else
         return dof + vd * ndofs;
   }
   if (dof < 0)
      return -1 - ((-1-dof) * vdim + vd);
   else
      return dof * vdim + vd;
}

// static function
void FiniteElementSpace::AdjustVDofs (Array<int> &vdofs)
{
   int n = vdofs.Size(), *vdof = vdofs;
   for (int i = 0; i < n; i++)
   {
      int j;
      if ((j=vdof[i]) < 0)
         vdof[i] = -1-j;
   }
}

void FiniteElementSpace::GetElementVDofs(int iE, Array<int> &dofs) const
{
   GetElementDofs(iE, dofs);
   DofsToVDofs (dofs);
}

void FiniteElementSpace::GetBdrElementVDofs (int iE, Array<int> &dofs) const
{
   GetBdrElementDofs(iE, dofs);
   DofsToVDofs (dofs);
}

void FiniteElementSpace::GetFaceVDofs (int iF, Array<int> &dofs) const
{
   GetFaceDofs (iF, dofs);
   DofsToVDofs (dofs);
}

void FiniteElementSpace::GetEdgeVDofs (int iE, Array<int> &dofs) const
{
   GetEdgeDofs (iE, dofs);
   DofsToVDofs (dofs);
}

void FiniteElementSpace::GetElementInteriorVDofs (int i, Array<int> &vdofs)
   const
{
   GetElementInteriorDofs (i, vdofs);
   DofsToVDofs (vdofs);
}

void FiniteElementSpace::GetEdgeInteriorVDofs (int i, Array<int> &vdofs)
   const
{
   GetEdgeInteriorDofs (i, vdofs);
   DofsToVDofs (vdofs);
}

void FiniteElementSpace::BuildElementToDofTable()
{
   if (elem_dof)
      return;

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

void FiniteElementSpace::BuildDofToArrays()
{
   if (dof_elem_array.Size())
      return;
   BuildElementToDofTable();

   dof_elem_array.SetSize (ndofs);
   dof_ldof_array.SetSize (ndofs);
   dof_elem_array = -1;
   for (int i = 0; i < mesh -> GetNE(); i++)
   {
      const int *dofs = elem_dof -> GetRow(i);
      const int n = elem_dof -> RowSize(i);
      for (int j = 0; j < n; j++)
         if (dof_elem_array[dofs[j]] < 0)
         {
            dof_elem_array[dofs[j]] = i;
            dof_ldof_array[dofs[j]] = j;
         }
   }
}

DenseMatrix * FiniteElementSpace::LocalInterpolation
(int k, int num_c_dofs, RefinementType type, Array<int> &rows)
{
   int i,j,l=-1;

   // generate refinement data if necessary
   for (i = 0; i < RefData.Size(); i++)
      if (RefData[i] -> type == type)
      { l = i; break; }

   if (l == -1)
   { ConstructRefinementData (k,num_c_dofs,type);  l = i; }

   // determine the global indices of the fine dofs on the coarse element
   Array<int> l_dofs, g_dofs;
   int num_fine_elems = RefData[l] -> num_fine_elems;
   rows.SetSize(RefData[l] -> num_fine_dofs);

   for (i = 0; i < num_fine_elems; i++) {
      GetElementDofs(mesh->GetFineElem(k,i),g_dofs); // TWO_LEVEL_FINE
      RefData[l] -> fl_to_fc -> GetRow (i, l_dofs);
      for (j = 0; j < l_dofs.Size(); j++)
         rows[l_dofs[j]] = g_dofs[j];
   }

   return RefData[l] -> I;
}

SparseMatrix * FiniteElementSpace::GlobalRestrictionMatrix
(FiniteElementSpace * cfes, int one_vdim)
{
   int k;
   SparseMatrix * R;
   DenseMatrix * r;
   Array<int> rows, cols, rs, cs;

   if (one_vdim == -1)
      one_vdim = (ordering == Ordering::byNODES) ? 1 : 0;

   mesh -> SetState (Mesh::TWO_LEVEL_COARSE);
   if (vdim == 1 || one_vdim == 1)
      R = new SparseMatrix (cfes -> GetNDofs(), ndofs);
   else
      R = new SparseMatrix (cfes -> GetVSize(), vdim*ndofs);

   for (k = 0; k < mesh -> GetNE(); k++)
   {
      cfes -> GetElementDofs (k, rows);
      mesh -> SetState (Mesh::TWO_LEVEL_FINE);
      r = LocalInterpolation (k, rows.Size(),
                              mesh -> GetRefinementType(k), cols);
      if (vdim == 1 || one_vdim == 1)
         R -> SetSubMatrixTranspose (rows, cols, *r, 1);
      else
      {
         int d, nr = rows.Size(), nc = cols.Size();
         cfes -> DofsToVDofs (rows);
         DofsToVDofs (cols);
         for (d = 0; d < vdim; d++)
         {
            rows.GetSubArray (d*nr, nr, rs);
            cols.GetSubArray (d*nc, nc, cs);
            R -> SetSubMatrixTranspose (rs, cs, *r, 1);
         }
      }
      mesh -> SetState (Mesh::TWO_LEVEL_COARSE);
   }

   return R;
}

void FiniteElementSpace::GetEssentialVDofs (Array<int> &bdr_attr_is_ess,
                                            Array<int> &ess_dofs)
{
   int i, j, k;
   Array<int> vdofs;

   ess_dofs.SetSize (GetVSize());
   ess_dofs = 0;

   for (i = 0; i < GetNBE(); i++)
      if (bdr_attr_is_ess[GetBdrAttribute (i)-1])
      {
         GetBdrElementVDofs (i, vdofs);
         for (j = 0; j < vdofs.Size(); j++)
            if ( (k = vdofs[j]) >= 0 )
               ess_dofs[k] = -1;
            else
               ess_dofs[-1-k] = -1;
      }
}

void FiniteElementSpace::EliminateEssentialBCFromGRM
(FiniteElementSpace *cfes, Array<int> &bdr_attr_is_ess, SparseMatrix *R)
{
   int i, j, k, one_vdim;
   Array<int> dofs;

   one_vdim = (cfes -> GetNDofs() == R -> Size()) ? 1 : 0;

   mesh -> SetState (Mesh::TWO_LEVEL_COARSE);
   if (bdr_attr_is_ess.Size() != 0)
      for (i=0; i < cfes -> GetNBE(); i++)
         if (bdr_attr_is_ess[cfes -> GetBdrAttribute (i)-1])
         {
            if (one_vdim == 1)
               cfes -> GetBdrElementDofs (i, dofs);
            else
               cfes -> GetBdrElementVDofs (i, dofs);
            for (j=0; j < dofs.Size(); j++)
               if ( (k = dofs[j]) >= 0 )
                  R -> EliminateRow(k);
               else
                  R -> EliminateRow(-1-k);
         }
   R -> Finalize();
}

SparseMatrix * FiniteElementSpace::GlobalRestrictionMatrix
(FiniteElementSpace * cfes, Array<int> &bdr_attr_is_ess, int one_vdim)
{
   SparseMatrix * R;

   R = GlobalRestrictionMatrix (cfes, one_vdim);
   EliminateEssentialBCFromGRM (cfes, bdr_attr_is_ess, R);

   return R;
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
         mfem_error ("FiniteElementSpace::D2C_GlobalRestrictionMatrix (...)");
#endif

      for (j = 0; j < d_vdofs.Size(); j++)
         R -> Set (c_vdofs[j], d_vdofs[j], 1.0);
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
         R -> Set (c_dofs[0], d_dofs[j], 1.0);
   }

   R -> Finalize();

   return R;
}

SparseMatrix *
FiniteElementSpace::H2L_GlobalRestrictionMatrix (FiniteElementSpace *lfes)
{
   int i, j;
   SparseMatrix *R;
   DenseMatrix loc_restr;
   Vector shape;
   Array<int> l_dofs, h_dofs;

   R = new SparseMatrix (lfes -> GetNDofs(), ndofs);

   const FiniteElement *h_fe = this -> GetFE (0);
   const FiniteElement *l_fe = lfes -> GetFE (0);
   shape.SetSize (l_fe -> GetDof());
   loc_restr.SetSize (l_fe -> GetDof(), h_fe -> GetDof());
   for (i = 0; i < h_fe -> GetDof(); i++)
   {
      l_fe -> CalcShape (h_fe -> GetNodes().IntPoint(i), shape);
      for (j = 0; j < shape.Size(); j++)
      {
         if (fabs (shape(j)) < 1.0e-6)
            shape(j) = 0.0;
         loc_restr(j,i) = shape(j);
      }
   }

   for (i = 0; i < mesh -> GetNE(); i++)
   {
      this -> GetElementDofs (i, h_dofs);
      lfes -> GetElementDofs (i, l_dofs);

      R -> SetSubMatrix (l_dofs, h_dofs, loc_restr, 1);
   }

   R -> Finalize();

   return R;
}

FiniteElementSpace::FiniteElementSpace(FiniteElementSpace &fes)
{
   mesh = fes.mesh;
   vdim = fes.vdim;
   ndofs = fes.ndofs;
   ordering = fes.ordering;
   fec = fes.fec;
   nvdofs = fes.nvdofs;
   nedofs = fes.nedofs;
   nfdofs = fes.nfdofs;
   nbdofs = fes.nbdofs;
   fdofs = fes.fdofs;
   bdofs = fes.bdofs;
   // keep 'RefData' in 'fes'
   elem_dof = fes.elem_dof;
   bdrElem_dof = fes.bdrElem_dof;
   Swap(dof_elem_array, fes.dof_elem_array);
   Swap(dof_ldof_array, fes.dof_ldof_array);

   NURBSext = fes.NURBSext;
   own_ext = 0;

   fes.bdofs = NULL;
   fes.fdofs = NULL;
   fes.elem_dof = NULL;
   fes.bdrElem_dof = NULL;
}

FiniteElementSpace::FiniteElementSpace(Mesh *m,
                                       const FiniteElementCollection *f,
                                       int dim, int order)
{
   mesh = m;
   fec = f;
   vdim = dim;
   ordering = order;

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
      }
   }
   else
   {
      NURBSext = NULL;
      own_ext = 0;
      Constructor();
   }
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

void FiniteElementSpace::Constructor()
{
   int i;

   elem_dof = NULL;
   bdrElem_dof = NULL;

   nvdofs = mesh->GetNV() * fec->DofForGeometry(Geometry::POINT);

   if ( mesh->Dimension() > 1 )
      nedofs = mesh->GetNEdges() * fec->DofForGeometry(Geometry::SEGMENT);
   else
      nedofs = 0;

   nfdofs = 0;
   fdofs = NULL;
   if (mesh->Dimension() == 3)
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

   nbdofs = 0;
   bdofs = new int[mesh->GetNE()+1];
   bdofs[0] = 0;
   for (i = 0; i < mesh->GetNE(); i++)
   {
      nbdofs += fec->DofForGeometry(mesh->GetElementBaseGeometry(i));
      bdofs[i+1] = nbdofs;
   }

   ndofs = nvdofs + nedofs + nfdofs + nbdofs;

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
      nb = fec->DofForGeometry(mesh->GetElementBaseGeometry(i));
      if (nv > 0)
         mesh->GetElementVertices(i, V);
      if (ne > 0)
         mesh->GetElementEdges(i, E, Eo);
      nfd = 0;
      if (dim == 3)
         if (fec->HasFaceDofs(mesh->GetElementBaseGeometry(i)))
         {
            mesh->GetElementFaces(i, F, Fo);
            for (k = 0; k < F.Size(); k++)
               nfd += fec->DofForGeometry(mesh->GetFaceBaseGeometry(F[k]));
         }
      nd = V.Size() * nv + E.Size() * ne + nfd + nb;
      dofs.SetSize(nd);
      if (nv > 0)
      {
         for (k = 0; k < V.Size(); k++)
            for (j = 0; j < nv; j++)
               dofs[k*nv+j] = V[k]*nv+j;
         nv *= V.Size();
      }
      if (ne > 0)
         // if (dim > 1)
         for (k = 0; k < E.Size(); k++)
         {
            ind = fec->DofOrderForOrientation(Geometry::SEGMENT, Eo[k]);
            for (j = 0; j < ne; j++)
               if (ind[j] < 0)
                  dofs[nv+k*ne+j] = -1 - ( nvdofs+E[k]*ne+(-1-ind[j]) );
               else
                  dofs[nv+k*ne+j] = nvdofs+E[k]*ne+ind[j];
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
                  dofs[ne+j] = -1 - ( nvdofs+nedofs+fdofs[F[k]]+(-1-ind[j]) );
               else
                  dofs[ne+j] = nvdofs+nedofs+fdofs[F[k]]+ind[j];
            }
            ne += nf;
         }
      }
      k = nvdofs + nedofs + nfdofs + bdofs[i];
      for (j = 0; j < nb; j++)
      {
         dofs[ne+j] = k + j;
      }
   }
}

const FiniteElement *FiniteElementSpace::GetFE(int i) const
{
   const FiniteElement *FE =
      fec->FiniteElementForGeometry(mesh->GetElementBaseGeometry(i));

   if (NURBSext)
      NURBSext->LoadFE(i, FE);

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
         mesh->GetBdrElementVertices(i, V);
      ne = (dim > 1) ? ( fec->DofForGeometry(Geometry::SEGMENT) ) : ( 0 );
      if (ne > 0)
         mesh->GetBdrElementEdges(i, E, Eo);
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
            for (j = 0; j < nv; j++)
               dofs[k*nv+j] = V[k]*nv+j;
         nv *= V.Size();
      }
      if (ne > 0)
         // if (dim > 1)
         for (k = 0; k < E.Size(); k++)
         {
            ind = fec->DofOrderForOrientation(Geometry::SEGMENT, Eo[k]);
            for (j = 0; j < ne; j++)
               if (ind[j] < 0)
                  dofs[nv+k*ne+j] = -1 - ( nvdofs+E[k]*ne+(-1-ind[j]) );
               else
                  dofs[nv+k*ne+j] = nvdofs+E[k]*ne+ind[j];
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
               dofs[ne+j] = -1 - ( nvdofs+nedofs+fdofs[iF]+(-1-ind[j]) );
            else
               dofs[ne+j] = nvdofs+nedofs+fdofs[iF]+ind[j];
         }
      }
   }
}

void FiniteElementSpace::GetFaceDofs(int i, Array<int> &dofs) const
{
   int j, k, nv, ne, nf, nd;
   Array<int> V, E, Eo;

   nv = fec->DofForGeometry(Geometry::POINT);
   ne = fec->DofForGeometry(Geometry::SEGMENT);
   if (nv > 0)
      mesh->GetFaceVertices(i, V);
   if (ne > 0)
      mesh->GetFaceEdges(i, E, Eo);
   nf = (fdofs) ? (fdofs[i+1]-fdofs[i]) : (0);
   nd = V.Size() * nv + E.Size() * ne + nf;
   dofs.SetSize (nd);
   if (nv > 0)
      for (k = 0; k < V.Size(); k++)
         for (j = 0; j < nv; j++)
            dofs[k*nv+j] = V[k]*nv+j;
   nv *= V.Size();
   // do not take edge orientation 'Eo' into account
   if (ne > 0)
      for (k = 0; k < E.Size(); k++)
         for (j = 0; j < ne; j++)
            dofs[nv+k*ne+j] = nvdofs+E[k]*ne+j;
   ne = nv + ne * E.Size();
   if (nf > 0)
      for (j = nvdofs+nedofs+fdofs[i], k = 0; k < nf; j++, k++)
         dofs[ne+k] = j;
}

void FiniteElementSpace::GetEdgeDofs(int i, Array<int> &dofs) const
{
   int j, k, nv, ne;
   Array<int> V;

   nv = fec->DofForGeometry(Geometry::POINT);
   if (nv > 0)
      mesh->GetEdgeVertices(i, V);
   ne = fec->DofForGeometry(Geometry::SEGMENT);
   dofs.SetSize(2*nv+ne);
   if (nv > 0)
      for (k = 0; k < 2; k++)
         for (j = 0; j < nv; j++)
            dofs[k*nv+j] = V[k]*nv+j;
   nv *= 2;
   for (j = 0, k = nvdofs+i*ne; j < ne; j++, k++)
      dofs[nv+j] = k;
}

void FiniteElementSpace::GetElementInteriorDofs (int i, Array<int> &dofs) const
{
   int j, k, nb;
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
      dofs[j] = k;
}

const FiniteElement *FiniteElementSpace::GetBE (int i) const
{
   const FiniteElement *BE;

   switch ( mesh->Dimension() )
   {
   case 1:
      BE = fec->FiniteElementForGeometry(Geometry::POINT);
   case 2:
      BE = fec->FiniteElementForGeometry(Geometry::SEGMENT);
   case 3:
      BE = fec->FiniteElementForGeometry(
         mesh->GetBdrElementBaseGeometry(i));
   }

   if (NURBSext)
      NURBSext->LoadBE(i, BE);

   return BE;
}

FiniteElementSpace::~FiniteElementSpace()
{
   Destructor();
   // delete RefData
   for (int i = 0; i < RefData.Size(); i++)
      delete RefData[i];
}

void FiniteElementSpace::Destructor()
{
   dof_elem_array.DeleteAll();
   dof_ldof_array.DeleteAll();

   if (NURBSext)
   {
      if (own_ext) delete NURBSext;
   }
   else
   {
      delete elem_dof;
      delete bdrElem_dof;

      delete [] bdofs;
      delete [] fdofs;
   }
}

void FiniteElementSpace::Update()
{
   if (NURBSext)
   {
      UpdateNURBS();
   }
   else
   {
      Destructor();   // keeps RefData
      Constructor();
   }
}

FiniteElementSpace *FiniteElementSpace::SaveUpdate()
{
   FiniteElementSpace *cfes = new FiniteElementSpace(*this);
   Constructor();
   return cfes;
}

void FiniteElementSpace::ConstructRefinementData (int k, int num_c_dofs,
                                                  RefinementType type)
{
   int i,j,l;
   Array<int> dofs, g_dofs;

   RefinementData * data = new RefinementData;

   data -> type = type;
   data -> num_fine_elems = mesh -> GetNumFineElems(k);

   // we assume that each fine element has <= dofs than the
   // initial coarse element
   data -> fl_to_fc = new Table(data -> num_fine_elems, num_c_dofs);
   for (i = 0; i < data -> num_fine_elems; i++) {
      GetElementDofs(mesh -> GetFineElem(k,i), g_dofs); // TWO_LEVEL_FINE
      for (j = 0; j < g_dofs.Size(); j++)
         data -> fl_to_fc -> Push (i,dofs.Union(g_dofs[j]));
   }
   data -> fl_to_fc -> Finalize();
   data -> num_fine_dofs = dofs.Size();

   // construction of I

   // k is a coarse element index but mesh is in TWO_LEVEL_FINE state
   int geomtype = mesh->GetElementBaseGeometry(k);
   const FiniteElement *fe = fec -> FiniteElementForGeometry(geomtype);
   // const IntegrationRule &ir = fe -> GetNodes();
   int nedofs = fe -> GetDof();  // number of dofs for each element

   ElementTransformation *trans;
   // IntegrationPoint ip;
   // DenseMatrix tr;
   Array<int> row;

   // Vector shape(nedofs);
   DenseMatrix I (nedofs);
   data -> I = new DenseMatrix(data -> num_fine_dofs, nedofs);

   for (i=0; i < data -> num_fine_elems; i++) {

      trans = mesh -> GetFineElemTrans(k,i);
      // trans -> Transform(ir,tr);
      fe -> GetLocalInterpolation (*trans, I);
      data -> fl_to_fc -> GetRow(i,row);

      for (j=0; j < nedofs; j++)
      {
         /*
           ip.x = tr(0,j); ip.y = tr(1,j);       if (tr.Height()==3) ip.z = tr(2,j);
           fe -> SetCalcElemTrans(trans); shape(0) = j;
           fe -> CalcShape(ip, shape);
         */
         for (l=0; l < nedofs; l++)
            (*(data->I))(row[j],l) = I(j,l);
      }
   }

   RefData.Append(data);
}

void FiniteElementSpace::Save(ostream &out) const
{
   out << "FiniteElementSpace\n"
       << "FiniteElementCollection: " << fec->Name() << '\n'
       << "VDim: " << vdim << '\n'
       << "Ordering: " << ordering << '\n';
}
