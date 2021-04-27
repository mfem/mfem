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

#include "../config/config.hpp"

#ifdef MFEM_USE_MPI

#include "restriction.hpp"
#include "prestriction.hpp"
#include "pgridfunc.hpp"
#include "pfespace.hpp"
#include "fespace.hpp"
#include "../general/forall.hpp"

namespace mfem
{

ParL2FaceRestriction::ParL2FaceRestriction(const ParFiniteElementSpace &fes,
                                           ElementDofOrdering e_ordering,
                                           FaceType type,
                                           L2FaceValues m)
   : L2FaceRestriction(fes, type, m)
{
   if (nf==0) { return; }
   // If fespace == L2
   const ParFiniteElementSpace &pfes =
      static_cast<const ParFiniteElementSpace&>(this->fes);
   const FiniteElement *fe = pfes.GetFE(0);
   const TensorBasisElement *tfe = dynamic_cast<const TensorBasisElement*>(fe);
   MFEM_VERIFY(tfe != NULL &&
               (tfe->GetBasisType()==BasisType::GaussLobatto ||
                tfe->GetBasisType()==BasisType::Positive),
               "Only Gauss-Lobatto and Bernstein basis are supported in "
               "ParL2FaceRestriction.");
   MFEM_VERIFY(pfes.GetMesh()->Conforming(),
               "Non-conforming meshes not yet supported with partial assembly.");
   // Assuming all finite elements are using Gauss-Lobatto dofs
   height = (m==L2FaceValues::DoubleValued? 2 : 1)*vdim*nf*dof;
   width = pfes.GetVSize();
   const bool dof_reorder = (e_ordering == ElementDofOrdering::LEXICOGRAPHIC);
   if (!dof_reorder)
   {
      MFEM_ABORT("Non-Tensor L2FaceRestriction not yet implemented.");
   }
   if (dof_reorder && nf > 0)
   {
      for (int f = 0; f < pfes.GetNF(); ++f)
      {
         const FiniteElement *fe =
            pfes.GetTraceElement(f, pfes.GetMesh()->GetFaceBaseGeometry(f));
         const TensorBasisElement* el =
            dynamic_cast<const TensorBasisElement*>(fe);
         if (el) { continue; }
         MFEM_ABORT("Finite element not suitable for lexicographic ordering");
      }
   }
   // End of verifications
   Mesh &mesh = *fes.GetMesh();
   const Table& e2dTable = pfes.GetElementToDofTable();
   const int* elementMap = e2dTable.GetJ();
   Array<int> faceMap1(dof), faceMap2(dof);
   Mesh::FaceInformation info;
   int e1, e2;
   int face_id1, face_id2;
   int orientation;
   const int dof1d = pfes.GetFE(0)->GetOrder()+1;
   const int elem_dofs = pfes.GetFE(0)->GetDof();
   const int dim = pfes.GetMesh()->SpaceDimension();
   // Computation of scatter indices
   int f_ind=0;
   for (int f = 0; f < pfes.GetNF(); ++f)
   {
      mesh.GetFaceInformation(f, info);
      e1 = info.elem_1_index;
      e2 = info.elem_2_index;
      face_id1 = info.elem_1_local_face;
      face_id2 = info.elem_2_local_face;
      orientation = info.elem_2_orientation;
      if (dof_reorder)
      {
         GetFaceDofs(dim, face_id1, dof1d, faceMap1); // only for hex
         GetFaceDofs(dim, face_id2, dof1d, faceMap2); // only for hex
      }
      else
      {
         MFEM_ABORT("FaceRestriction not yet implemented for this type of "
                    "element.");
         // TODO Something with GetFaceDofs?
         orientation = 0;          // suppress compiler warning
         face_id1 = face_id2 = 0;  // suppress compiler warning
      }
      if (type==FaceType::Interior &&
          (info.location==Mesh::FaceLocation::Interior ||
           info.location==Mesh::FaceLocation::Shared) )
      {
         for (int d = 0; d < dof; ++d)
         {
            const int face_dof = faceMap1[d];
            const int did = face_dof;
            const int gid = elementMap[e1*elem_dofs + did];
            const int lid = dof*f_ind + d;
            scatter_indices1[lid] = gid;
         }
         if (m==L2FaceValues::DoubleValued)
         {
            if (info.location==Mesh::FaceLocation::Interior)
            {
               for (int d = 0; d < dof; ++d)
               {
                  const int pd = PermuteFaceL2(dim, face_id1, face_id2,
                                               orientation, dof1d, d);
                  const int face_dof = faceMap2[pd];
                  const int did = face_dof;
                  const int gid = elementMap[e2*elem_dofs + did];
                  const int lid = dof*f_ind + d;
                  scatter_indices2[lid] = gid;
               }
            }
            else if (info.location==Mesh::FaceLocation::Shared)
            {
               Array<int> sharedDofs;
               pfes.GetFaceNbrElementVDofs(e2, sharedDofs);
               for (int d = 0; d < dof; ++d)
               {
                  const int pd = PermuteFaceL2(dim, face_id1, face_id2,
                                               orientation, dof1d, d);
                  const int face_dof = faceMap2[pd];
                  const int did = face_dof;
                  const int gid = sharedDofs[did];
                  const int lid = dof*f_ind + d;
                  // Trick to differentiate dof location inter/shared
                  scatter_indices2[lid] = ndofs+gid;
               }
            }
         }
         f_ind++;
      }
      else if (type==FaceType::Boundary &&
               info.location==Mesh::FaceLocation::Boundary)
      {
         for (int d = 0; d < dof; ++d)
         {
            const int face_dof = faceMap1[d];
            const int did = face_dof;
            const int gid = elementMap[e1*elem_dofs + did];
            const int lid = dof*f_ind + d;
            scatter_indices1[lid] = gid;
         }
         if (m==L2FaceValues::DoubleValued)
         {
            for (int d = 0; d < dof; ++d)
            {
               const int lid = dof*f_ind + d;
               scatter_indices2[lid] = -1;
            }
         }
         f_ind++;
      }
   }
   MFEM_VERIFY(f_ind==nf, "Unexpected number of faces.");
   // Computation of gather_indices
   for (int i = 0; i <= ndofs; ++i)
   {
      offsets[i] = 0;
   }
   f_ind = 0;
   for (int f = 0; f < pfes.GetNF(); ++f)
   {
      mesh.GetFaceInformation(f, info);
      e1 = info.elem_1_index;
      e2 = info.elem_2_index;
      face_id1 = info.elem_1_local_face;
      face_id2 = info.elem_2_local_face;
      orientation = info.elem_2_orientation;
      if ((type==FaceType::Interior &&
           (info.location==Mesh::FaceLocation::Interior ||
            info.location==Mesh::FaceLocation::Shared) ) ||
          (type==FaceType::Boundary &&
           info.location==Mesh::FaceLocation::Boundary) )
      {
         GetFaceDofs(dim, face_id1, dof1d, faceMap1);
         GetFaceDofs(dim, face_id2, dof1d, faceMap2);
         for (int d = 0; d < dof; ++d)
         {
            const int did = faceMap1[d];
            const int gid = elementMap[e1*elem_dofs + did];
            ++offsets[gid + 1];
         }
         if (m==L2FaceValues::DoubleValued)
         {
            if (type==FaceType::Interior &&
                info.location==Mesh::FaceLocation::Interior)
            {
               for (int d = 0; d < dof; ++d)
               {
                  const int pd = PermuteFaceL2(dim, face_id1, face_id2,
                                               orientation, dof1d, d);
                  const int did = faceMap2[pd];
                  const int gid = elementMap[e2*elem_dofs + did];
                  ++offsets[gid + 1];
               }
            }
         }
         f_ind++;
      }
   }
   MFEM_VERIFY(f_ind==nf, "Unexpected number of faces.");
   for (int i = 1; i <= ndofs; ++i)
   {
      offsets[i] += offsets[i - 1];
   }
   f_ind = 0;
   for (int f = 0; f < pfes.GetNF(); ++f)
   {
      mesh.GetFaceInformation(f, info);
      e1 = info.elem_1_index;
      e2 = info.elem_2_index;
      face_id1 = info.elem_1_local_face;
      face_id2 = info.elem_2_local_face;
      orientation = info.elem_2_orientation;
      if ((type==FaceType::Interior &&
           (info.location==Mesh::FaceLocation::Interior ||
            info.location==Mesh::FaceLocation::Shared) ) ||
          (type==FaceType::Boundary &&
           info.location==Mesh::FaceLocation::Boundary) )
      {
         GetFaceDofs(dim, face_id1, dof1d, faceMap1);
         GetFaceDofs(dim, face_id2, dof1d, faceMap2);
         for (int d = 0; d < dof; ++d)
         {
            const int did = faceMap1[d];
            const int gid = elementMap[e1*elem_dofs + did];
            const int lid = dof*f_ind + d;
            // We don't shift lid to express that it's e1 of f
            gather_indices[offsets[gid]++] = lid;
         }
         if (m==L2FaceValues::DoubleValued)
         {
            if (type==FaceType::Interior &&
                info.location==Mesh::FaceLocation::Interior)
            {
               for (int d = 0; d < dof; ++d)
               {
                  const int pd = PermuteFaceL2(dim, face_id1, face_id2,
                                               orientation, dof1d, d);
                  const int did = faceMap2[pd];
                  const int gid = elementMap[e2*elem_dofs + did];
                  const int lid = dof*f_ind + d;
                  // We shift lid to express that it's e2 of f
                  gather_indices[offsets[gid]++] = nfdofs + lid;
               }
            }
         }
         f_ind++;
      }
   }
   MFEM_VERIFY(f_ind==nf, "Unexpected number of faces.");
   for (int i = ndofs; i > 0; --i)
   {
      offsets[i] = offsets[i - 1];
   }
   offsets[0] = 0;
}

void ParL2FaceRestriction::Mult(const Vector& x, Vector& y) const
{
   const ParFiniteElementSpace &pfes =
      static_cast<const ParFiniteElementSpace&>(this->fes);
   ParGridFunction x_gf;
   x_gf.MakeRef(const_cast<ParFiniteElementSpace*>(&pfes),
                const_cast<Vector&>(x), 0);
   x_gf.ExchangeFaceNbrData();

   // Assumes all elements have the same number of dofs
   const int nd = dof;
   const int vd = vdim;
   const bool t = byvdim;
   const int threshold = ndofs;
   const int nsdofs = pfes.GetFaceNbrVSize();

   if (m==L2FaceValues::DoubleValued)
   {
      auto d_indices1 = scatter_indices1.Read();
      auto d_indices2 = scatter_indices2.Read();
      auto d_x = Reshape(x.Read(), t?vd:ndofs, t?ndofs:vd);
      auto d_x_shared = Reshape(x_gf.FaceNbrData().Read(),
                                t?vd:nsdofs, t?nsdofs:vd);
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
            if (idx2>-1 && idx2<threshold) // interior face
            {
               d_y(dof, c, 1, face) = d_x(t?c:idx2, t?idx2:c);
            }
            else if (idx2>=threshold) // shared boundary
            {
               d_y(dof, c, 1, face) = d_x_shared(t?c:(idx2-threshold),
                                                 t?(idx2-threshold):c);
            }
            else // true boundary
            {
               d_y(dof, c, 1, face) = 0.0;
            }
         }
      });
   }
   else
   {
      auto d_indices1 = scatter_indices1.Read();
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

static MFEM_HOST_DEVICE int AddNnz(const int iE, int *I, const int dofs)
{
   int val = AtomicAdd(I[iE],dofs);
   return val;
}

void ParL2FaceRestriction::FillI(SparseMatrix &mat,
                                 const bool keep_nbr_block) const
{
   if (keep_nbr_block)
   {
      return L2FaceRestriction::FillI(mat, keep_nbr_block);
   }
   const int face_dofs = dof;
   const int Ndofs = ndofs;
   auto d_indices1 = scatter_indices1.Read();
   auto d_indices2 = scatter_indices2.Read();
   auto I = mat.ReadWriteI();
   MFEM_FORALL(fdof, nf*face_dofs,
   {
      const int f  = fdof/face_dofs;
      const int iF = fdof%face_dofs;
      const int iE1 = d_indices1[f*face_dofs+iF];
      if (iE1 < Ndofs)
      {
         AddNnz(iE1,I,face_dofs);
      }
      const int iE2 = d_indices2[f*face_dofs+iF];
      if (iE2 < Ndofs)
      {
         AddNnz(iE2,I,face_dofs);
      }
   });
}

void ParL2FaceRestriction::FillI(SparseMatrix &mat,
                                 SparseMatrix &face_mat) const
{
   const int face_dofs = dof;
   const int Ndofs = ndofs;
   auto d_indices1 = scatter_indices1.Read();
   auto d_indices2 = scatter_indices2.Read();
   auto I = mat.ReadWriteI();
   auto I_face = face_mat.ReadWriteI();
   MFEM_FORALL(i, ne*elemDofs*vdim+1,
   {
      I_face[i] = 0;
   });
   MFEM_FORALL(fdof, nf*face_dofs,
   {
      const int f  = fdof/face_dofs;
      const int iF = fdof%face_dofs;
      const int iE1 = d_indices1[f*face_dofs+iF];
      if (iE1 < Ndofs)
      {
         for (int jF = 0; jF < face_dofs; jF++)
         {
            const int jE2 = d_indices2[f*face_dofs+jF];
            if (jE2 < Ndofs)
            {
               AddNnz(iE1,I,1);
            }
            else
            {
               AddNnz(iE1,I_face,1);
            }
         }
      }
      const int iE2 = d_indices2[f*face_dofs+iF];
      if (iE2 < Ndofs)
      {
         for (int jF = 0; jF < face_dofs; jF++)
         {
            const int jE1 = d_indices1[f*face_dofs+jF];
            if (jE1 < Ndofs)
            {
               AddNnz(iE2,I,1);
            }
            else
            {
               AddNnz(iE2,I_face,1);
            }
         }
      }
   });
}

void ParL2FaceRestriction::FillJAndData(const Vector &ea_data,
                                        SparseMatrix &mat,
                                        const bool keep_nbr_block) const
{
   if (keep_nbr_block)
   {
      return L2FaceRestriction::FillJAndData(ea_data, mat, keep_nbr_block);
   }
   const int face_dofs = dof;
   const int Ndofs = ndofs;
   auto d_indices1 = scatter_indices1.Read();
   auto d_indices2 = scatter_indices2.Read();
   auto mat_fea = Reshape(ea_data.Read(), face_dofs, face_dofs, 2, nf);
   auto I = mat.ReadWriteI();
   auto J = mat.WriteJ();
   auto Data = mat.WriteData();
   MFEM_FORALL(fdof, nf*face_dofs,
   {
      const int f  = fdof/face_dofs;
      const int iF = fdof%face_dofs;
      const int iE1 = d_indices1[f*face_dofs+iF];
      if (iE1 < Ndofs)
      {
         const int offset = AddNnz(iE1,I,face_dofs);
         for (int jF = 0; jF < face_dofs; jF++)
         {
            const int jE2 = d_indices2[f*face_dofs+jF];
            J[offset+jF] = jE2;
            Data[offset+jF] = mat_fea(jF,iF,1,f);
         }
      }
      const int iE2 = d_indices2[f*face_dofs+iF];
      if (iE2 < Ndofs)
      {
         const int offset = AddNnz(iE2,I,face_dofs);
         for (int jF = 0; jF < face_dofs; jF++)
         {
            const int jE1 = d_indices1[f*face_dofs+jF];
            J[offset+jF] = jE1;
            Data[offset+jF] = mat_fea(jF,iF,0,f);
         }
      }
   });
}

void ParL2FaceRestriction::FillJAndData(const Vector &ea_data,
                                        SparseMatrix &mat,
                                        SparseMatrix &face_mat) const
{
   const int face_dofs = dof;
   const int Ndofs = ndofs;
   auto d_indices1 = scatter_indices1.Read();
   auto d_indices2 = scatter_indices2.Read();
   auto mat_fea = Reshape(ea_data.Read(), face_dofs, face_dofs, 2, nf);
   auto I = mat.ReadWriteI();
   auto I_face = face_mat.ReadWriteI();
   auto J = mat.WriteJ();
   auto J_face = face_mat.WriteJ();
   auto Data = mat.WriteData();
   auto Data_face = face_mat.WriteData();
   MFEM_FORALL(fdof, nf*face_dofs,
   {
      const int f  = fdof/face_dofs;
      const int iF = fdof%face_dofs;
      const int iE1 = d_indices1[f*face_dofs+iF];
      if (iE1 < Ndofs)
      {
         for (int jF = 0; jF < face_dofs; jF++)
         {
            const int jE2 = d_indices2[f*face_dofs+jF];
            if (jE2 < Ndofs)
            {
               const int offset = AddNnz(iE1,I,1);
               J[offset] = jE2;
               Data[offset] = mat_fea(jF,iF,1,f);
            }
            else
            {
               const int offset = AddNnz(iE1,I_face,1);
               J_face[offset] = jE2-Ndofs;
               Data_face[offset] = mat_fea(jF,iF,1,f);
            }
         }
      }
      const int iE2 = d_indices2[f*face_dofs+iF];
      if (iE2 < Ndofs)
      {
         for (int jF = 0; jF < face_dofs; jF++)
         {
            const int jE1 = d_indices1[f*face_dofs+jF];
            if (jE1 < Ndofs)
            {
               const int offset = AddNnz(iE2,I,1);
               J[offset] = jE1;
               Data[offset] = mat_fea(jF,iF,0,f);
            }
            else
            {
               const int offset = AddNnz(iE2,I_face,1);
               J_face[offset] = jE1-Ndofs;
               Data_face[offset] = mat_fea(jF,iF,0,f);
            }
         }
      }
   });
}

ParNCL2FaceRestriction::ParNCL2FaceRestriction(const ParFiniteElementSpace &fes,
                                               ElementDofOrdering e_ordering,
                                               FaceType type,
                                               L2FaceValues m)
   : NCL2FaceRestriction(fes, type, m)
{
   if (nf==0) { return; }
   // If fespace==L2
   const ParFiniteElementSpace &pfes =
      static_cast<const ParFiniteElementSpace&>(this->fes);
   const FiniteElement *fe = pfes.GetFE(0);
   const TensorBasisElement *tfe = dynamic_cast<const TensorBasisElement*>(fe);
   MFEM_VERIFY(tfe != NULL &&
               (tfe->GetBasisType()==BasisType::GaussLobatto ||
                tfe->GetBasisType()==BasisType::Positive),
               "Only Gauss-Lobatto and Bernstein basis are supported in "
               "ParNCL2FaceRestriction.");
   // Assuming all finite elements are using Gauss-Lobatto dofs
   height = (m==L2FaceValues::DoubleValued? 2 : 1)*vdim*nf*dof;
   width = pfes.GetVSize();
   const bool dof_reorder = (e_ordering==ElementDofOrdering::LEXICOGRAPHIC);
   if (!dof_reorder)
   {
      MFEM_ABORT("Non-Tensor L2FaceRestriction not yet implemented.");
   }
   if (dof_reorder && nf > 0)
   {
      for (int f = 0; f < pfes.GetNF(); ++f)
      {
         const FiniteElement *fe =
            pfes.GetTraceElement(f, pfes.GetMesh()->GetFaceBaseGeometry(f));
         const TensorBasisElement* el =
            dynamic_cast<const TensorBasisElement*>(fe);
         if (el) { continue; }
         MFEM_ABORT("Finite element not suitable for lexicographic ordering");
      }
   }
   // End of verifications
   Mesh &mesh = *fes.GetMesh();
   const Table& e2dTable = pfes.GetElementToDofTable();
   const int* elementMap = e2dTable.GetJ();
   Array<int> faceMap1(dof), faceMap2(dof);
   Mesh::FaceInformation info;
   int e1, e2;
   int face_id1, face_id2;
   int orientation;
   const int dof1d = pfes.GetFE(0)->GetOrder()+1;
   const int elem_dofs = pfes.GetFE(0)->GetDof();
   const int dim = pfes.GetMesh()->SpaceDimension();
   int nc_cpt = 0;
   using Key = std::pair<const DenseMatrix*,int>;
   std::map<Key, std::pair<int,DenseMatrix*>> interp_map;
   // Computation of scatter and offsets indices
   for (int i = 0; i <= ndofs; ++i)
   {
      offsets[i] = 0;
   }
   int f_ind=0;
   for (int f = 0; f < fes.GetNF(); ++f)
   {
      mesh.GetFaceInformation(f, info);
      e1 = info.elem_1_index;
      e2 = info.elem_2_index;
      face_id1 = info.elem_1_local_face;
      face_id2 = info.elem_2_local_face;
      orientation = info.elem_2_orientation;
      if (dof_reorder)
      {
         GetFaceDofs(dim, face_id1, dof1d, faceMap1); // Only for hex
         GetFaceDofs(dim, face_id2, dof1d, faceMap2); // Only for hex
      }
      else
      {
         MFEM_ABORT("FaceRestriction not yet implemented for this type of "
                    "element.");
         // TODO Something with GetFaceDofs?
      }
      if (type==FaceType::Interior &&
          (info.location==Mesh::FaceLocation::Interior ||
           info.location==Mesh::FaceLocation::Shared) )
      {
         for (int d = 0; d < dof; ++d)
         {
            const int face_dof = faceMap1[d];
            const int gid = elementMap[e1*elem_dofs + face_dof];
            const int lid = dof*f_ind + d;
            scatter_indices1[lid] = gid;
            ++offsets[gid + 1];
         }
         if ( m==L2FaceValues::DoubleValued )
         {
            if ( info.conformity==Mesh::FaceConformity::Conforming )
            {
               interp_config[f_ind] = conforming;
            }
            else // Non-conforming face
            {
               const DenseMatrix* ptMat = mesh.GetNCFacesPtMat(info.ncface);
               Key key(ptMat, face_id2);
               auto itr = interp_map.find(key);
               if (itr==interp_map.end())
               {
                  // Computation of the interpolation matrix from coarse face to
                  // fine face.
                  // Assumes all trace elements are the same.
                  const FiniteElement *trace_fe =
                     fes.GetTraceElement(0, fes.GetMesh()->GetFaceBaseGeometry(0));
                  DenseMatrix* interp_mat = new DenseMatrix(dof,dof);
                  Vector shape(dof);
                  IntegrationPoint f_ip;
                  double x_min(0), x_max(0), y_min(0), y_max(0);
                  switch (trace_fe->GetGeomType())
                  {
                     case Geometry::SQUARE:
                     {
                        MFEM_ASSERT(ptMat->Height()==2, "Unexpected PtMat height.");
                        MFEM_ASSERT(ptMat->Width()==4, "Unexpected PtMat width.");
                        x_min = x_max = (*ptMat)(0,0);
                        y_min = y_max = (*ptMat)(1,0);
                        for (size_t v = 1; v < 4; v++)
                        {
                           const double a = (*ptMat)(0,v);
                           x_min = a < x_min ? a : x_min;
                           x_max = a > x_max ? a : x_max;
                           const double b = (*ptMat)(1,v);
                           y_min = b < y_min ? b : y_min;
                           y_max = b > y_max ? b : y_max;
                        }
                        bool invert_x = face_id2==3 || face_id2==4;
                        if (invert_x)
                        {
                           const double piv = x_min;
                           x_min = 1-x_max;
                           x_max = 1-piv;
                        }
                        bool invert_y = face_id2==0;
                        if (invert_y)
                        {
                           const double piv = y_min;
                           y_min = 1-y_max;
                           y_max = 1-piv;
                        }
                     }
                     break;
                     case Geometry::SEGMENT:
                     {
                        MFEM_ASSERT(ptMat->Height()==1, "Unexpected PtMat height.");
                        MFEM_ASSERT(ptMat->Width()==2, "Unexpected PtMat width.");
                        bool invert = face_id2==2 || face_id2==3;
                        double a = (*ptMat)(0,0);
                        double b = (*ptMat)(0,1);
                        double x1 = !invert ? a : 1.0-a;
                        double x2 = !invert ? b : 1.0-b;
                        if ( x1 < x2 )
                        {
                           x_min = x1;
                           x_max = x2;
                        }
                        else
                        {
                           x_min = x2;
                           x_max = x1;
                        }
                     }
                     break;
                     default: MFEM_ABORT("unsupported geometry");
                  }
                  const IntegrationRule & nodes = trace_fe->GetNodes();
                  for (int i = 0; i < dof; i++)
                  {
                     const IntegrationPoint &ip = nodes[i];
                     switch (trace_fe->GetGeomType())
                     {
                        case Geometry::SQUARE:
                           f_ip.y = y_min + (y_max - y_min) * ip.y;
                        case Geometry::SEGMENT:
                           f_ip.x = x_min + (x_max - x_min) * ip.x;
                           break;
                        default: MFEM_ABORT("unsupported geometry");
                     }
                     trace_fe->CalcShape(f_ip, shape);
                     for (int j = 0; j < dof; j++)
                     {
                        (*interp_mat)(i,j) = shape(j);
                     }
                  }
                  interp_map[key] = {nc_cpt, interp_mat};
                  interp_config[f_ind] = nc_cpt;
                  nc_cpt++;
               }
               else
               {
                  interp_config[f_ind] = itr->second.first;
               }
            }
            if (info.location==Mesh::FaceLocation::Interior)
            {
               for (int d = 0; d < dof; ++d)
               {
                  const int pd = PermuteFaceL2(dim, face_id1, face_id2,
                                               orientation, dof1d, d);
                  const int face_dof = faceMap2[pd];
                  const int did = face_dof;
                  const int gid = elementMap[e2*elem_dofs + did];
                  const int lid = dof*f_ind + d;
                  scatter_indices2[lid] = gid;
                  ++offsets[gid + 1];
               }
            }
            else if (info.location==Mesh::FaceLocation::Shared)
            {
               Array<int> sharedDofs;
               pfes.GetFaceNbrElementVDofs(e2, sharedDofs);
               for (int d = 0; d < dof; ++d)
               {
                  const int pd = PermuteFaceL2(dim, face_id1, face_id2,
                                               orientation, dof1d, d);
                  const int face_dof = faceMap2[pd];
                  const int did = face_dof;
                  const int gid = sharedDofs[did];
                  const int lid = dof*f_ind + d;
                  // Trick to differentiate dof location inter/shared
                  scatter_indices2[lid] = ndofs+gid;
               }
            }
         }
         f_ind++;
      }
      else if (type==FaceType::Boundary &&
               info.location==Mesh::FaceLocation::Boundary)
      {
         for (int d = 0; d < dof; ++d)
         {
            const int face_dof = faceMap1[d];
            const int gid = elementMap[e1*elem_dofs + face_dof];
            const int lid = dof*f_ind + d;
            scatter_indices1[lid] = gid;
            ++offsets[gid + 1];
         }
         if ( m==L2FaceValues::DoubleValued )
         {
            for (int d = 0; d < dof; ++d)
            {
               const int lid = dof*f_ind + d;
               scatter_indices2[lid] = -1;
            }
         }
         f_ind++;
      }
   }
   MFEM_VERIFY(f_ind==nf, "Unexpected number of " <<
               (type==FaceType::Interior? "interior" : "boundary") <<
               " faces: " << f_ind << " vs " << nf );
   for (int i = 1; i <= ndofs; ++i)
   {
      offsets[i] += offsets[i - 1];
   }
   // Computation of gather_indices
   f_ind = 0;
   for (int f = 0; f < fes.GetNF(); ++f)
   {
      mesh.GetFaceInformation(f, info);
      e1 = info.elem_1_index;
      e2 = info.elem_2_index;
      face_id1 = info.elem_1_local_face;
      face_id2 = info.elem_2_local_face;
      orientation = info.elem_2_orientation;
      if ((type==FaceType::Interior &&
           (info.location==Mesh::FaceLocation::Interior ||
            info.location==Mesh::FaceLocation::Shared) ) ||
          (type==FaceType::Boundary &&
           info.location==Mesh::FaceLocation::Boundary) )
      {
         GetFaceDofs(dim, face_id1, dof1d, faceMap1);
         GetFaceDofs(dim, face_id2, dof1d, faceMap2);
         for (int d = 0; d < dof; ++d)
         {
            const int did = faceMap1[d];
            const int gid = elementMap[e1*elem_dofs + did];
            const int lid = dof*f_ind + d;
            // We don't shift lid to express that it's e1 of f
            gather_indices[offsets[gid]++] = lid;
         }
         if (m==L2FaceValues::DoubleValued &&
             type==FaceType::Interior &&
             info.location==Mesh::FaceLocation::Interior)
         {
            for (int d = 0; d < dof; ++d)
            {
               const int pd = PermuteFaceL2(dim, face_id1, face_id2,
                                            orientation, dof1d, d);
               const int did = faceMap2[pd];
               const int gid = elementMap[e2*elem_dofs + did];
               const int lid = dof*f_ind + d;
               // We shift lid to express that it's e2 of f
               gather_indices[offsets[gid]++] = nfdofs + lid;
            }
         }
         f_ind++;
      }
   }
   MFEM_VERIFY(f_ind==nf, "Unexpected number of " <<
               (type==FaceType::Interior? "interior" : "boundary") <<
               " faces: " << f_ind << " vs " << nf );
   // Switch back offsets to their correct value
   for (int i = ndofs; i > 0; --i)
   {
      offsets[i] = offsets[i - 1];
   }
   offsets[0] = 0;
   // Transform the interpolation matrix map into a contiguous memory structure.
   nc_size = interp_map.size();
   interpolators.SetSize(dof*dof*nc_size);
   auto interp = Reshape(interpolators.HostWrite(),dof,dof,nc_size);
   for (auto val : interp_map)
   {
      const int idx = val.second.first;
      for (int i = 0; i < dof; i++)
      {
         for (int j = 0; j < dof; j++)
         {
            interp(i,j,idx) = (*val.second.second)(i,j);
         }
      }
      delete val.second.second;
   }
}

void ParNCL2FaceRestriction::Mult(const Vector& x, Vector& y) const
{
   const ParFiniteElementSpace &pfes =
      static_cast<const ParFiniteElementSpace&>(this->fes);
   ParGridFunction x_gf;
   x_gf.MakeRef(const_cast<ParFiniteElementSpace*>(&pfes),
                const_cast<Vector&>(x), 0);
   x_gf.ExchangeFaceNbrData();

   // Assumes all elements have the same number of dofs
   const int nd = dof;
   const int vd = vdim;
   const bool t = byvdim;
   const int threshold = ndofs;
   const int nsdofs = pfes.GetFaceNbrVSize();

   if ( type==FaceType::Interior && m==L2FaceValues::DoubleValued )
   {
      auto d_indices1 = scatter_indices1.Read();
      auto d_indices2 = scatter_indices2.Read();
      auto d_x = Reshape(x.Read(), t?vd:ndofs, t?ndofs:vd);
      auto d_x_shared = Reshape(x_gf.FaceNbrData().Read(),
                                t?vd:nsdofs, t?nsdofs:vd);
      auto d_y = Reshape(y.Write(), nd, vd, 2, nf);
      auto interp_config_ptr = interp_config.Read();
      auto interp = Reshape(interpolators.Read(), nd, nd, nc_size);
      MFEM_FORALL(face, nf,
      {
         double dofs[nd];
         double res[nd];
         for (int c = 0; c < vd; ++c)
         {
            for (int side = 0; side < 2; side++)
            {
               for (int dof = 0; dof<nd; dof++)
               {
                  const int i = face*nd + dof;
                  const int idx = side==0 ? d_indices1[i] : d_indices2[i];
                  if (idx>-1 && idx<threshold) // interior face
                  {
                     dofs[dof] = d_x(t?c:idx, t?idx:c);
                  }
                  else if (idx>=threshold) // shared interior face
                  {
                     dofs[dof] = d_x_shared(t?c:(idx-threshold),
                                            t?(idx-threshold):c);
                  }
                  else // true boundary
                  {
                     dofs[dof] = 0.0;
                  }
               }
               const int config = side==0 ? conforming : interp_config_ptr[face];
               if ( config==conforming ) // No interpolation needed
               {
                  for (int dof = 0; dof<nd; dof++)
                  {
                     d_y(dof, c, side, face) = dofs[dof];
                  }
               }
               else // Interpolation from coarse to fine
               {
                  for (int dofOut = 0; dofOut<nd; dofOut++)
                  {
                     res[dofOut] = 0.0;
                     for (int dofIn = 0; dofIn<nd; dofIn++)
                     {
                        res[dofOut] += interp(dofOut, dofIn, config)*dofs[dofIn];
                     }
                  }
                  for (int dof = 0; dof<nd; dof++)
                  {
                     d_y(dof, c, side, face) = res[dof];
                  }
               }
            }
         }
      });
   }
   else if ( type==FaceType::Boundary && m==L2FaceValues::DoubleValued )
   {
      auto d_indices1 = scatter_indices1.Read();
      auto d_indices2 = scatter_indices2.Read();
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
   else // Single valued
   {
      auto d_indices1 = scatter_indices1.Read();
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

void ParNCL2FaceRestriction::FillI(SparseMatrix &mat,
                                   const bool keep_nbr_block) const
{
   MFEM_ABORT("Not yet implemented.");
}

void ParNCL2FaceRestriction::FillI(SparseMatrix &mat,
                                   SparseMatrix &face_mat) const
{
   MFEM_ABORT("Not yet implemented.");
}

void ParNCL2FaceRestriction::FillJAndData(const Vector &ea_data,
                                          SparseMatrix &mat,
                                          const bool keep_nbr_block) const
{
   MFEM_ABORT("Not yet implemented.");
}

void ParNCL2FaceRestriction::FillJAndData(const Vector &ea_data,
                                          SparseMatrix &mat,
                                          SparseMatrix &face_mat) const
{
   MFEM_ABORT("Not yet implemented.");
}

} // namespace mfem

#endif
