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

#include "restriction.hpp"
#include "pfespace.hpp"
#include "pgridfunc.hpp"
#include "../general/forall.hpp"

namespace mfem
{

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

void H1FaceRestriction::GetFaceDofs(const int dim, const int face_id,
                                    const int dof1d, Array<int> &faceMap)
{
   switch (dim)
   {
      case 1:
         switch (face_id)
         {
            case 0://WEST
               faceMap[0] = 0;
               break;
            case 1://EAST
               faceMap[0] = dof1d-1;
               break;
         }
         break;
      case 2:
         switch (face_id)
         {
            case 0://SOUTH
               for (int i = 0; i < dof1d; ++i)
               {
                  faceMap[i] = i;
               }
               break;
            case 1://EAST
               for (int i = 0; i < dof1d; ++i)
               {
                  faceMap[i] = dof1d-1 + i*dof1d;
               }
               break;
            case 2://NORTH
               for (int i = 0; i < dof1d; ++i)
               {
                  faceMap[i] = (dof1d-1)*dof1d + i;//Lex
                  // faceMap[i] = (dof1d-1)*dof1d + dof1d-1 - i;
               }
               break;
            case 3://WEST
               for (int i = 0; i < dof1d; ++i)
               {
                  faceMap[i] = i*dof1d;//Lex
                  // faceMap[i] = (dof1d-1)*dof1d - i*dof1d;
               }
               break;
         }
         break;
      case 3:
         switch (face_id)
         {
            case 0://BOTTOM
               for (int i = 0; i < dof1d; ++i)
               {
                  for (int j = 0; j < dof1d; ++j)
                  {
                     faceMap[i+j*dof1d] = i + j*dof1d;//Lex
                     // faceMap[i+j*dof1d] = i + (dof1d-1-j)*dof1d;
                  }
               }
               break;
            case 1://SOUTH
               for (int i = 0; i < dof1d; ++i)
               {
                  for (int j = 0; j < dof1d; ++j)
                  {
                     faceMap[i+j*dof1d] = i + j*dof1d*dof1d;//Lex
                     // faceMap[i+j*dof1d] = (dof1d-1-i) + j*dof1d*dof1d;
                  }
               }
               break;
            case 2://EAST
               for (int i = 0; i < dof1d; ++i)
               {
                  for (int j = 0; j < dof1d; ++j)
                  {
                     faceMap[i+j*dof1d] = dof1d-1 + i*dof1d + j*dof1d*dof1d;
                  }
               }
               break;
            case 3://NORTH
               for (int i = 0; i < dof1d; ++i)
               {
                  for (int j = 0; j < dof1d; ++j)
                  {
                     faceMap[i+j*dof1d] = (dof1d-1)*dof1d + i + j*dof1d*dof1d;//Lex
                     // faceMap[i+j*dof1d] = (dof1d-1)*dof1d + (dof1d-1-i) + j*dof1d*dof1d;
                  }
               }
               break;
            case 4://WEST
               for (int i = 0; i < dof1d; ++i)
               {
                  for (int j = 0; j < dof1d; ++j)
                  {
                     faceMap[i+j*dof1d] = i*dof1d + j*dof1d*dof1d;//Lex
                     // faceMap[i+j*dof1d] = (dof1d-1-i)*dof1d + j*dof1d*dof1d;
                  }
               }
               break;
            case 5://TOP
               for (int i = 0; i < dof1d; ++i)
               {
                  for (int j = 0; j < dof1d; ++j)
                  {
                     faceMap[i+j*dof1d] = (dof1d-1)*dof1d*dof1d + i + j*dof1d;
                  }
               }
               break;
         }
         break;
   }
}

H1FaceRestriction::H1FaceRestriction(const FiniteElementSpace &fes,
                                     const ElementDofOrdering e_ordering,
                                     const FaceType type)
   : fes(fes),
     nf(fes.GetNFbyType(type)),
     vdim(fes.GetVDim()),
     byvdim(fes.GetOrdering() == Ordering::byVDIM),
     ndofs(fes.GetNDofs()),
     dof(nf > 0 ? fes.GetFaceElement(0)->GetDof() : 0),
     nfdofs(nf*dof),
     indices(nf*dof)
{
   if (nf==0) { return; }
   //if fespace == H1
   const FiniteElement *fe = fes.GetFE(0);
   const TensorBasisElement *tfe = dynamic_cast<const TensorBasisElement*>(fe);
   MFEM_VERIFY(tfe != NULL &&
               (tfe->GetBasisType()==BasisType::GaussLobatto ||
                tfe->GetBasisType()==BasisType::Positive),
               "Only Gauss-Lobatto and Bernstein basis are supported in H1FaceRestriction.");
   MFEM_VERIFY(fes.GetMesh()->Conforming(),
               "Non-conforming meshes not yet supported with partial assembly.");
   // Assuming all finite elements are using Gauss-Lobatto.
   height = vdim*nf*dof;
   width = fes.GetVSize();
   const bool dof_reorder = (e_ordering == ElementDofOrdering::LEXICOGRAPHIC);
   if (dof_reorder && nf > 0)
   {
      for (int f = 0; f < fes.GetNF(); ++f)
      {
         const FiniteElement *fe = fes.GetFaceElement(f);
         const TensorBasisElement* el =
            dynamic_cast<const TensorBasisElement*>(fe);
         if (el) { continue; }
         mfem_error("Finite element not suitable for lexicographic ordering");
      }
      const FiniteElement *fe = fes.GetFaceElement(0);
      const TensorBasisElement* el =
         dynamic_cast<const TensorBasisElement*>(fe);
      const Array<int> &fe_dof_map = el->GetDofMap();
      MFEM_VERIFY(fe_dof_map.Size() > 0, "invalid dof map");
   }
   const TensorBasisElement* el =
      dynamic_cast<const TensorBasisElement*>(fe);
   const int *dof_map = el->GetDofMap().GetData();
   const Table& e2dTable = fes.GetElementToDofTable();
   const int* elementMap = e2dTable.GetJ();
   Array<int> faceMap(dof);
   int e1, e2;
   int inf1, inf2;
   int face_id;
   int orientation;
   const int dof1d = fes.GetFE(0)->GetOrder()+1;
   const int elem_dofs = fes.GetFE(0)->GetDof();
   const int dim = fes.GetMesh()->SpaceDimension();
   int f_ind = 0;
   for (int f = 0; f < fes.GetNF(); ++f)
   {
      fes.GetMesh()->GetFaceElements(f, &e1, &e2);
      fes.GetMesh()->GetFaceInfos(f, &inf1, &inf2);
      orientation = inf1 % 64;
      face_id = inf1 / 64;
      if ((type==FaceType::Interior && (e2>=0 || (e2<0 && inf2>=0))) ||
          (type==FaceType::Boundary && e2<0 && inf2<0) )
      {
         // Assumes Gauss-Lobato basis
         if (dof_reorder)
         {
            if (orientation!=0) { mfem_error("FaceRestriction used on degenerated mesh."); }
            GetFaceDofs(dim, face_id, dof1d, faceMap);//Only for hex
         }
         else
         {
            mfem_error("FaceRestriction not yet implemented for this type of element.");
            //TODO Something with GetFaceDofs?
         }
         for (int d = 0; d < dof; ++d)
         {
            const int face_dof = faceMap[d];
            const int did = (!dof_reorder)?face_dof:dof_map[face_dof];
            const int gid = elementMap[e1*elem_dofs + did];
            const int lid = dof*f_ind + d;
            indices[lid] = gid;
         }
         f_ind++;
      }
   }
   MFEM_VERIFY(f_ind==nf, "Unexpected number of faces.");
}

void H1FaceRestriction::Mult(const Vector& x, Vector& y) const
{
   // Assumes all elements have the same number of dofs
   const int nd = dof;
   const int vd = vdim;
   const bool t = byvdim;
   auto d_indices = indices.Read();
   auto d_x = Reshape(x.Read(), t?vd:ndofs, t?ndofs:vd);
   auto d_y = Reshape(y.Write(), nd, vd, nf);
   MFEM_FORALL(i, nfdofs,
   {
      const int idx = d_indices[i];
      const int dof = i % nd;
      const int face = i / nd;
      for (int c = 0; c < vd; ++c)
      {
         d_y(dof, c, face) = d_x(t?c:idx, t?idx:c);
      }
   });
}

void H1FaceRestriction::MultTranspose(const Vector& x, Vector& y) const
{
   // Assumes all elements have the same number of dofs
   const int nd = dof;
   const int vd = vdim;
   const bool t = byvdim;
   auto d_indices = indices.Read();
   auto d_x = Reshape(x.Read(), nd, vd, nf);
   auto d_y = Reshape(y.Write(), t?vd:ndofs, t?ndofs:vd);
   MFEM_FORALL(i, nfdofs,
   {
      const int idx = d_indices[i];
      for (int c = 0; c < vd; ++c)
      {
         MFEM_ATOMIC_ADD(d_y(t?c:idx,t?idx:c), d_x(i % nd, c, i / nd));
      }
   });
}

static int ToLexOrdering2D(const int face_id, const int size1d, const int i)
{
   if (face_id==2 || face_id==3)
   {
      return size1d-1-i;
   }
   else
   {
      return i;
   }
}

static int PermuteFace2D(const int face_id1, const int face_id2,
                         const int orientation,
                         const int size1d, const int index)
{
   int new_index;
   // Convert from lex ordering
   if (face_id1==2 || face_id1==3)
   {
      new_index = size1d-1-index;
   }
   else
   {
      new_index = index;
   }
   // Permute based on face orientations
   if (orientation==1)
   {
      new_index = size1d-1-new_index;
   }
   return ToLexOrdering2D(face_id2, size1d, new_index);
}

static int ToLexOrdering3D(const int face_id, const int size1d, const int i,
                           const int j)
{
   if (face_id==2 || face_id==1 || face_id==5)
   {
      return i + j*size1d;
   }
   else if (face_id==3 || face_id==4)
   {
      return (size1d-1-i) + j*size1d;
   }
   else//face_id==0
   {
      return i + (size1d-1-j)*size1d;
   }
}

static int PermuteFace3D(const int face_id1, const int face_id2,
                         const int orientation,
                         const int size1d, const int index)
{
   int i=0, j=0, new_i=0, new_j=0;
   i = index%size1d;
   j = index/size1d;
   // Convert from lex ordering
   if (face_id1==3 || face_id1==4)
   {
      i = size1d-1-i;
   }
   else if (face_id1==0)
   {
      j = size1d-1-j;
   }
   // Permute based on face orientations
   switch (orientation)
   {
      case 0:
         new_i = i;
         new_j = j;
         break;
      case 1:
         new_i = j;
         new_j = i;
         break;
      case 2:
         new_i = j;
         new_j = (size1d-1-i);
         break;
      case 3:
         new_i = (size1d-1-i);
         new_j = j;
         break;
      case 4:
         new_i = (size1d-1-i);
         new_j = (size1d-1-j);
         break;
      case 5:
         new_i = (size1d-1-j);
         new_j = (size1d-1-i);
         break;
      case 6:
         new_i = (size1d-1-j);
         new_j = i;
         break;
      case 7:
         new_i = i;
         new_j = (size1d-1-j);
         break;
   }
   return ToLexOrdering3D(face_id2, size1d, new_i, new_j);
}

int L2FaceRestriction::PermuteFaceL2(const int dim, const int face_id1,
                                     const int face_id2, const int orientation,
                                     const int size1d, const int index)
{
   switch (dim)
   {
      case 1:
         return 0;
      case 2:
         return PermuteFace2D(face_id1, face_id2, orientation, size1d, index);
      case 3:
         return PermuteFace3D(face_id1, face_id2, orientation, size1d, index);
      default:
         mfem_error("Unsupported dimension.");
         return 0;
   }
}

L2FaceRestriction::L2FaceRestriction(const FiniteElementSpace &fes,
                                     const ElementDofOrdering e_ordering,
                                     const FaceType type,
                                     const L2FaceValues m)
   : fes(fes),
     nf(fes.GetNFbyType(type)),
     vdim(fes.GetVDim()),
     byvdim(fes.GetOrdering() == Ordering::byVDIM),
     ndofs(fes.GetNDofs()),
     dof(nf > 0 ? fes.GetTraceElement(0,
                                      fes.GetMesh()->GetFaceBaseGeometry(0))->GetDof() : 0),
     m(m),
     nfdofs(nf*dof),
     indices1(nf*dof),
     indices2(m==L2FaceValues::Double?nf*dof:0)
{
   //if fespace == L2
   const FiniteElement *fe = fes.GetFE(0);
   const TensorBasisElement *tfe = dynamic_cast<const TensorBasisElement*>(fe);
   MFEM_VERIFY(tfe != NULL &&
               (tfe->GetBasisType()==BasisType::GaussLobatto ||
                tfe->GetBasisType()==BasisType::Positive),
               "Only Gauss-Lobatto and Bernstein basis are supported in L2FaceRestriction.");
   MFEM_VERIFY(fes.GetMesh()->Conforming(),
               "Non-conforming meshes not yet supported with partial assembly.");
   if (nf==0) { return; }
   height = (m==L2FaceValues::Double? 2 : 1)*vdim*nf*dof;
   width = fes.GetVSize();
   const bool dof_reorder = (e_ordering == ElementDofOrdering::LEXICOGRAPHIC);
   if (!dof_reorder) { mfem_error("Non-Tensor L2FaceRestriction not yet implemented."); }
   if (dof_reorder && nf > 0)
   {
      for (int f = 0; f < fes.GetNF(); ++f)
      {
         const FiniteElement *fe = fes.GetTraceElement(f,
                                                       fes.GetMesh()->GetFaceBaseGeometry(f));
         const TensorBasisElement* el =
            dynamic_cast<const TensorBasisElement*>(fe);
         if (el) { continue; }
         mfem_error("Finite element not suitable for lexicographic ordering");
      }
   }
   const Table& e2dTable = fes.GetElementToDofTable();
   const int* elementMap = e2dTable.GetJ();
   Array<int> faceMap1(dof), faceMap2(dof);
   int e1, e2;
   int inf1, inf2;
   int face_id1, face_id2;
   int orientation;
   const int dof1d = fes.GetFE(0)->GetOrder()+1;
   const int elem_dofs = fes.GetFE(0)->GetDof();
   const int dim = fes.GetMesh()->SpaceDimension();
   int f_ind=0;
   for (int f = 0; f < fes.GetNF(); ++f)
   {
      fes.GetMesh()->GetFaceElements(f, &e1, &e2);
      fes.GetMesh()->GetFaceInfos(f, &inf1, &inf2);
      if (dof_reorder)
      {
         orientation = inf1 % 64;
         face_id1 = inf1 / 64;
         H1FaceRestriction::GetFaceDofs(dim, face_id1, dof1d, faceMap1);//Only for hex
         orientation = inf2 % 64;
         face_id2 = inf2 / 64;
         H1FaceRestriction::GetFaceDofs(dim, face_id2, dof1d, faceMap2);//Only for hex
      }
      else
      {
         mfem_error("FaceRestriction not yet implemented for this type of element.");
         //TODO Something with GetFaceDofs?
      }
      if ((type==FaceType::Interior && e2>=0) || (type==FaceType::Boundary && e2<0) )
      {
         for (int d = 0; d < dof; ++d)
         {
            const int face_dof = faceMap1[d];
            const int did = face_dof;
            const int gid = elementMap[e1*elem_dofs + did];
            const int lid = dof*f_ind + d;
            indices1[lid] = gid;
         }
         if (m==L2FaceValues::Double)
         {
            for (int d = 0; d < dof; ++d)
            {
               if (type==FaceType::Interior && e2>=0) //interior face
               {
                  const int pd = PermuteFaceL2(dim, face_id1, face_id2, orientation, dof1d, d);
                  const int face_dof = faceMap2[pd];
                  const int did = face_dof;
                  const int gid = elementMap[e2*elem_dofs + did];
                  const int lid = dof*f_ind + d;
                  indices2[lid] = gid;
               }
               else if (type==FaceType::Boundary && e2<0) // true boundary face
               {
                  const int lid = dof*f_ind + d;
                  indices2[lid] = -1;
               }
            }
         }
         f_ind++;
      }
   }
   MFEM_VERIFY(f_ind==nf, "Unexpected number of faces.");
}

void L2FaceRestriction::Mult(const Vector& x, Vector& y) const
{
   // Assumes all elements have the same number of dofs
   const int nd = dof;
   const int vd = vdim;
   const bool t = byvdim;

   if (m==L2FaceValues::Double)
   {
      auto d_indices1 = indices1.Read();
      auto d_indices2 = indices2.Read();
      auto d_x = Reshape(x.Read(), t?vd:ndofs, t?ndofs:vd);
      auto d_y = Reshape(y.Write(), nd, vd, 2, nf);
      MFEM_FORALL(i, nfdofs,
      {
         const int dof = i % nd;
         const int face = i / nd;
         const int idx1 = d_indices1[i];
         for (int c = 0; c < vd; ++c)
         {
            d_y(dof, c, 0, face) = d_x(t?c:idx1, t?idx1:c);
         }
         const int idx2 = d_indices2[i];
         for (int c = 0; c < vd; ++c)
         {
            d_y(dof, c, 1, face) = idx2==-1 ? 0.0 : d_x(t?c:idx2, t?idx2:c);
         }
      });
   }
   else
   {
      auto d_indices1 = indices1.Read();
      auto d_x = Reshape(x.Read(), t?vd:ndofs, t?ndofs:vd);
      auto d_y = Reshape(y.Write(), nd, vd, nf);
      MFEM_FORALL(i, nfdofs,
      {
         const int dof = i % nd;
         const int face = i / nd;
         const int idx1 = d_indices1[i];
         for (int c = 0; c < vd; ++c)
         {
            d_y(dof, c, face) = d_x(t?c:idx1, t?idx1:c);
         }
      });
   }
}

void L2FaceRestriction::MultTranspose(const Vector& x, Vector& y) const
{
   // Assumes all elements have the same number of dofs
   const int nd = dof;
   const int vd = vdim;
   const bool t = byvdim;
   if (m==L2FaceValues::Double)
   {
      auto d_indices1 = indices1.Read();
      auto d_indices2 = indices2.Read();
      auto d_x = Reshape(x.Read(), nd, vd, 2, nf);
      auto d_y = Reshape(y.ReadWrite(), t?vd:ndofs, t?ndofs:vd);
      MFEM_FORALL(i, nfdofs,
      {
         const int idx1 = d_indices1[i];
         const int idx2 = d_indices2[i];
         for (int c = 0; c < vd; ++c)
         {
            MFEM_ATOMIC_ADD(d_y(t?c:idx1,t?idx1:c), d_x(i % nd, c, 0, i / nd));
            if (idx2!=-1) { MFEM_ATOMIC_ADD(d_y(t?c:idx2,t?idx2:c), d_x(i % nd, c, 1, i / nd)); }
         }
      });
   }
   else
   {
      auto d_indices1 = indices1.Read();
      auto d_x = Reshape(x.Read(), nd, vd, nf);
      auto d_y = Reshape(y.ReadWrite(), t?vd:ndofs, t?ndofs:vd);
      MFEM_FORALL(i, nfdofs,
      {
         const int idx1 = d_indices1[i];
         for (int c = 0; c < vd; ++c)
         {
            MFEM_ATOMIC_ADD(d_y(t?c:idx1,t?idx1:c), d_x(i % nd, c, i / nd));
         }
      });
   }
}


ParL2FaceRestriction::ParL2FaceRestriction(const ParFiniteElementSpace &fes,
                                           ElementDofOrdering e_ordering,
                                           FaceType type,
                                           L2FaceValues m)
   : fes(fes),
     nf(fes.GetNFbyType(type)),
     vdim(fes.GetVDim()),
     byvdim(fes.GetOrdering() == Ordering::byVDIM),
     ndofs(fes.GetNDofs()),
     dof(nf>0 ? fes.GetTraceElement(0,
                                    fes.GetMesh()->GetFaceBaseGeometry(0))->GetDof():0),
     m(m),
     nfdofs(nf*dof),
     indices1(nf*dof),
     indices2(m==L2FaceValues::Double?nf*dof:0)
{
   if (nf==0) { return; }
   //if fespace == L2
   const FiniteElement *fe = fes.GetFE(0);
   const TensorBasisElement *tfe = dynamic_cast<const TensorBasisElement*>(fe);
   MFEM_VERIFY(tfe != NULL &&
               (tfe->GetBasisType()==BasisType::GaussLobatto ||
                tfe->GetBasisType()==BasisType::Positive),
               "Only Gauss-Lobatto and Bernstein basis are supported in ParL2FaceRestriction.");
   MFEM_VERIFY(fes.GetMesh()->Conforming(),
               "Non-conforming meshes not yet supported with partial assembly.");
   // Assuming all finite elements are using Gauss-Lobatto.
   height = (m==L2FaceValues::Double? 2 : 1)*vdim*nf*dof;
   width = fes.GetVSize();
   const bool dof_reorder = (e_ordering == ElementDofOrdering::LEXICOGRAPHIC);
   if (!dof_reorder) { mfem_error("Non-Tensor L2FaceRestriction not yet implemented."); }
   if (dof_reorder && nf > 0)
   {
      for (int f = 0; f < fes.GetNF(); ++f)
      {
         const FiniteElement *fe = fes.GetTraceElement(f,
                                                       fes.GetMesh()->GetFaceBaseGeometry(f));
         const TensorBasisElement* el =
            dynamic_cast<const TensorBasisElement*>(fe);
         if (el) { continue; }
         mfem_error("Finite element not suitable for lexicographic ordering");
      }
   }
   const Table& e2dTable = fes.GetElementToDofTable();
   const int* elementMap = e2dTable.GetJ();
   Array<int> faceMap1(dof), faceMap2(dof);
   int e1, e2;
   int inf1, inf2;
   int face_id1, face_id2;
   int orientation;
   const int dof1d = fes.GetFE(0)->GetOrder()+1;
   const int elem_dofs = fes.GetFE(0)->GetDof();
   const int dim = fes.GetMesh()->SpaceDimension();
   int f_ind=0;
   for (int f = 0; f < fes.GetNF(); ++f)
   {
      fes.GetMesh()->GetFaceElements(f, &e1, &e2);
      fes.GetMesh()->GetFaceInfos(f, &inf1, &inf2);
      if (dof_reorder)
      {
         orientation = inf1 % 64;
         face_id1 = inf1 / 64;
         H1FaceRestriction::GetFaceDofs(dim, face_id1, dof1d, faceMap1);//Only for hex
         orientation = inf2 % 64;
         face_id2 = inf2 / 64;
         H1FaceRestriction::GetFaceDofs(dim, face_id2, dof1d, faceMap2);//Only for hex
      }
      else
      {
         mfem_error("FaceRestriction not yet implemented for this type of element.");
         //TODO Something with GetFaceDofs?
      }
      if (type==FaceType::Interior && (e2>=0 || (e2<0 &&
                                                 inf2>=0) )) //Interior/shared face
      {
         for (int d = 0; d < dof; ++d)
         {
            const int face_dof = faceMap1[d];
            const int did = face_dof;
            const int gid = elementMap[e1*elem_dofs + did];
            const int lid = dof*f_ind + d;
            indices1[lid] = gid;
         }
         if (m==L2FaceValues::Double)
         {
            if (e2>=0)//interior face
            {
               for (int d = 0; d < dof; ++d)
               {
                  const int pd = L2FaceRestriction::PermuteFaceL2(dim, face_id1, face_id2,
                                                                  orientation, dof1d, d);
                  const int face_dof = faceMap2[pd];
                  const int did = face_dof;
                  const int gid = elementMap[e2*elem_dofs + did];
                  const int lid = dof*f_ind + d;
                  indices2[lid] = gid;
               }
            }
            else if (inf2>=0) //shared boundary
            {
               const int se2 = -1 - e2;
               Array<int> sharedDofs;
               fes.GetFaceNbrElementVDofs(se2, sharedDofs);
               for (int d = 0; d < dof; ++d)
               {
                  const int pd = L2FaceRestriction::PermuteFaceL2(dim, face_id1, face_id2,
                                                                  orientation, dof1d, d);
                  const int face_dof = faceMap2[pd];
                  const int did = face_dof;
                  const int gid = sharedDofs[did];
                  const int lid = dof*f_ind + d;
                  indices2[lid] = ndofs+gid; //trick to differentiate dof location inter/shared
               }
            }
         }
         f_ind++;
      }
      else if ( type==FaceType::Boundary && e2<0 && inf2<0 ) //true boundary
      {
         for (int d = 0; d < dof; ++d)
         {
            const int face_dof = faceMap1[d];
            const int did = face_dof;
            const int gid = elementMap[e1*elem_dofs + did];
            const int lid = dof*f_ind + d;
            indices1[lid] = gid;
         }
         if (m==L2FaceValues::Double)
         {
            for (int d = 0; d < dof; ++d)
            {
               const int lid = dof*f_ind + d;
               indices2[lid] = -1;
            }
         }
         f_ind++;
      }
   }
   MFEM_VERIFY(f_ind==nf, "Unexpected number of faces.");
}

void ParL2FaceRestriction::Mult(const Vector& x, Vector& y) const
{
   ParGridFunction x_gf;
   x_gf.MakeRef(const_cast<ParFiniteElementSpace*>(&fes), const_cast<Vector&>(x),
                0);
   x_gf.ExchangeFaceNbrData();

   // Assumes all elements have the same number of dofs
   const int nd = dof;
   const int vd = vdim;
   const bool t = byvdim;
   const int threshold = ndofs;

   if (m==L2FaceValues::Double)
   {
      auto d_indices1 = indices1.Read();
      auto d_indices2 = indices2.Read();
      auto d_x = Reshape(x.Read(), t?vd:ndofs, t?ndofs:vd);
      auto d_x_shared = Reshape(x_gf.FaceNbrData().Read(), t?vd:ndofs, t?ndofs:vd);
      auto d_y = Reshape(y.Write(), nd, vd, 2, nf);
      MFEM_FORALL(i, nfdofs,
      {
         const int dof = i % nd;
         const int face = i / nd;
         const int idx1 = d_indices1[i];
         for (int c = 0; c < vd; ++c)
         {
            d_y(dof, c, 0, face) = d_x(t?c:idx1, t?idx1:c);
         }
         const int idx2 = d_indices2[i];
         for (int c = 0; c < vd; ++c)
         {
            if (idx2>-1 && idx2<threshold) //interior face
            {
               d_y(dof, c, 1, face) = d_x(t?c:idx2, t?idx2:c);
            }
            else if (idx2>=threshold) //shared boundary
            {
               d_y(dof, c, 1, face) = d_x_shared(t?c:(idx2-threshold), t?(idx2-threshold):c);
            }
            else //true boundary
            {
               d_y(dof, c, 1, face) = 0.0;
            }
         }
      });
   }
   else
   {
      auto d_indices1 = indices1.Read();
      auto d_x = Reshape(x.Read(), t?vd:ndofs, t?ndofs:vd);
      auto d_y = Reshape(y.Write(), nd, vd, nf);
      MFEM_FORALL(i, nfdofs,
      {
         const int dof = i % nd;
         const int face = i / nd;
         const int idx1 = d_indices1[i];
         for (int c = 0; c < vd; ++c)
         {
            d_y(dof, c, face) = d_x(t?c:idx1, t?idx1:c);
         }
      });
   }
}

void ParL2FaceRestriction::MultTranspose(const Vector& x, Vector& y) const
{
   // Assumes all elements have the same number of dofs
   const int nd = dof;
   const int vd = vdim;
   const bool t = byvdim;
   const int threshold = ndofs;
   if (m==L2FaceValues::Double)
   {
      auto d_indices1 = indices1.Read();
      auto d_indices2 = indices2.Read();
      auto d_x = Reshape(x.Read(), nd, vd, 2, nf);
      auto d_y = Reshape(y.ReadWrite(), t?vd:ndofs, t?ndofs:vd);
      MFEM_FORALL(i, nfdofs,
      {
         const int idx1 = d_indices1[i];
         const int idx2 = d_indices2[i];
         for (int c = 0; c < vd; ++c)
         {
            MFEM_ATOMIC_ADD(d_y(t?c:idx1,t?idx1:c), d_x(i % nd, c, 0, i / nd));
            if (idx2>-1 && idx2<threshold) { MFEM_ATOMIC_ADD(d_y(t?c:idx2,t?idx2:c), d_x(i % nd, c, 1, i / nd)); }
         }
      });
   }
   else
   {
      auto d_indices1 = indices1.Read();
      auto d_x = Reshape(x.Read(), nd, vd, nf);
      auto d_y = Reshape(y.ReadWrite(), t?vd:ndofs, t?ndofs:vd);
      MFEM_FORALL(i, nfdofs,
      {
         const int idx1 = d_indices1[i];
         for (int c = 0; c < vd; ++c)
         {
            MFEM_ATOMIC_ADD(d_y(t?c:idx1,t?idx1:c), d_x(i % nd, c, i / nd));
         }
      });
   }
}

int ToLexOrdering(const int dim, const int face_id, const int size1d,
                  const int index)
{
   switch (dim)
   {
      case 1:
         return 0;
      case 2:
         return ToLexOrdering2D(face_id, size1d, index);
      case 3:
         return ToLexOrdering3D(face_id, size1d, index%size1d, index/size1d);
      default:
         mfem_error("Unsupported dimension.");
         return 0;
   }
}

}
