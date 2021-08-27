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
   // Initialization of the offsets
   for (int i = 0; i <= ndofs; ++i)
   {
      offsets[i] = 0;
   }
   // Computation of scatter indices and offsets
   int f_ind=0;
   for (int f = 0; f < pfes.GetNF(); ++f)
   {
      Mesh::FaceInformation info = mesh.GetFaceInformation(f);
      if (type==FaceType::Interior && info.IsInterior())
      {
         SetFaceDofsScatterIndices1(info,f_ind);
         if (m==L2FaceValues::DoubleValued)
         {
            if (info.location==Mesh::FaceLocation::Interior)
            {
               PermuteAndSetFaceDofsScatterIndices2(info,f_ind);
            }
            else if (info.location==Mesh::FaceLocation::Shared)
            {
               PermuteAndSetSharedFaceDofsScatterIndices2(info,f_ind);
            }
         }
         f_ind++;
      }
      else if (type==FaceType::Boundary && info.IsBoundary())
      {
         SetFaceDofsScatterIndices1(info,f_ind);
         if (m==L2FaceValues::DoubleValued)
         {
            SetBoundaryDofsScatterIndices2(info,f_ind);
         }
         f_ind++;
      }
   }
   MFEM_VERIFY(f_ind==nf, "Unexpected number of faces.");
   // Summation of the offsets
   for (int i = 1; i <= ndofs; ++i)
   {
      offsets[i] += offsets[i - 1];
   }
   // Computation of gather_indices
   f_ind = 0;
   for (int f = 0; f < pfes.GetNF(); ++f)
   {
      Mesh::FaceInformation info = mesh.GetFaceInformation(f);
      if (info.IsOfFaceType(type))
      {
         SetFaceDofsGatherIndices1(info,f_ind);
         if (m==L2FaceValues::DoubleValued &&
             type==FaceType::Interior &&
             info.location==Mesh::FaceLocation::Interior)
         {
            PermuteAndSetFaceDofsGatherIndices2(info,f_ind);
         }
         f_ind++;
      }
   }
   MFEM_VERIFY(f_ind==nf, "Unexpected number of faces.");
   // Reset offsets to their correct value
   for (int i = ndofs; i > 0; --i)
   {
      offsets[i] = offsets[i - 1];
   }
   offsets[0] = 0;
}


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

   ComputeScatterIndicesAndOffsets(e_ordering, type);

   ComputeGatherIndices(e_ordering, type);
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
   MFEM_FORALL(i, ne*elem_dofs*vdim+1,
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

void ParL2FaceRestriction::ComputeScatterIndicesAndOffsets(
   const ElementDofOrdering ordering,
   const FaceType type)
{
   Mesh &mesh = *fes.GetMesh();
   const ParFiniteElementSpace &pfes =
      static_cast<const ParFiniteElementSpace&>(this->fes);

   // Initialization of the offsets
   for (int i = 0; i <= ndofs; ++i)
   {
      offsets[i] = 0;
   }

   // Computation of scatter indices and offsets
   int f_ind=0;
   for (int f = 0; f < pfes.GetNF(); ++f)
   {
      Mesh::FaceInformation face = mesh.GetFaceInformation(f);
      if (type==FaceType::Interior && face.IsInterior())
      {
         SetFaceDofsScatterIndices1(face,f_ind);
         if (m==L2FaceValues::DoubleValued)
         {
            if (face.IsShared())
            {
               PermuteAndSetSharedFaceDofsScatterIndices2(face,f_ind);
            }
            else
            {
               PermuteAndSetFaceDofsScatterIndices2(face,f_ind);
            }
         }
         f_ind++;
      }
      else if (type==FaceType::Boundary && face.IsBoundary())
      {
         SetFaceDofsScatterIndices1(face,f_ind);
         if (m==L2FaceValues::DoubleValued)
         {
            SetBoundaryDofsScatterIndices2(face,f_ind);
         }
         f_ind++;
      }
   }
   MFEM_VERIFY(f_ind==nf, "Unexpected number of faces.");

   // Summation of the offsets
   for (int i = 1; i <= ndofs; ++i)
   {
      offsets[i] += offsets[i - 1];
   }
}


void ParL2FaceRestriction::ComputeGatherIndices(
   const ElementDofOrdering ordering,
   const FaceType type)
{
   Mesh &mesh = *fes.GetMesh();
   const ParFiniteElementSpace &pfes =
      static_cast<const ParFiniteElementSpace&>(this->fes);

   // Computation of gather_indices
   int f_ind = 0;
   for (int f = 0; f < pfes.GetNF(); ++f)
   {
      Mesh::FaceInformation face = mesh.GetFaceInformation(f);
      if (face.IsOfFaceType(type))
      {
         SetFaceDofsGatherIndices1(face,f_ind);
         if (m==L2FaceValues::DoubleValued &&
             type==FaceType::Interior &&
             face.IsInterior() &&
             !face.IsShared())
         {
            PermuteAndSetFaceDofsGatherIndices2(face,f_ind);
         }
         f_ind++;
      }
   }
   MFEM_VERIFY(f_ind==nf, "Unexpected number of faces.");

   // Reset offsets to their correct value
   for (int i = ndofs; i > 0; --i)
   {
      offsets[i] = offsets[i - 1];
   }
   offsets[0] = 0;
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
   int nc_cpt = 0;
   using Key = std::pair<const DenseMatrix*,int>;
   std::map<Key, std::pair<int,const DenseMatrix*>> interp_map;
   // Initialization of the offsets
   for (int i = 0; i <= ndofs; ++i)
   {
      offsets[i] = 0;
   }
   // Computation of scatter and offsets indices
   int f_ind=0;
   for (int f = 0; f < mesh.GetNumFacesWithGhost(); ++f)
   {
      Mesh::FaceInformation info = mesh.GetFaceInformation(f);
      // We skip non-conforming master faces, as they will be treated by the
      // slave faces.
      if (info.conformity==Mesh::FaceConformity::NonConformingMaster)
      {
         continue;
      }
      if (type==FaceType::Interior && info.IsInterior())
      {
         SetFaceDofsScatterIndices1(info,f_ind);
         if ( m==L2FaceValues::DoubleValued )
         {
            if ( info.conformity==Mesh::FaceConformity::Conforming )
            {
               interp_config[f_ind] = conforming;
               if (info.location==Mesh::FaceLocation::Interior)
               {
                  PermuteAndSetFaceDofsScatterIndices2(info,f_ind);
               }
               else if (info.location==Mesh::FaceLocation::Shared)
               {
                  PermuteAndSetSharedFaceDofsScatterIndices2(info,f_ind);
               }
            }
            else // Non-conforming face
            {
               MFEM_ASSERT(e_ordering == ElementDofOrdering::LEXICOGRAPHIC,
                           "The following interpolation operator is "
                           "lexicographic.");
               const DenseMatrix* ptMat = mesh.GetNCFacesPtMat(info.ncface);
               const int face_key = info.elem_1_local_face +
                                    6*info.elem_2_local_face;
               Key key(ptMat, face_key);
               auto itr = interp_map.find(key);
               if (itr==interp_map.end())
               {
                  const DenseMatrix* interpolator =
                     ComputeCoarseToFineInterpolation(info,ptMat);
                  interp_map[key] = {nc_cpt, interpolator};
                  interp_config[f_ind] = nc_cpt;
                  nc_cpt++;
               }
               else
               {
                  interp_config[f_ind] = itr->second.first;
               }
               if (info.location==Mesh::FaceLocation::Interior)
               {
                  SetFaceDofsScatterIndices2(info,f_ind);
               }
               else if (info.location==Mesh::FaceLocation::Shared)
               {
                  SetSharedFaceDofsScatterIndices2(info,f_ind);
               }
            }
         }
         f_ind++;
      }
      else if (type==FaceType::Boundary && info.IsBoundary())
      {
         SetFaceDofsScatterIndices1(info,f_ind);
         if ( m==L2FaceValues::DoubleValued )
         {
            SetBoundaryDofsScatterIndices2(info,f_ind);
         }
         f_ind++;
      }
   }
   MFEM_VERIFY(f_ind==nf, "Unexpected number of " <<
               (type==FaceType::Interior? "interior" : "boundary") <<
               " faces: " << f_ind << " vs " << nf );
   // Summation of the offsets
   for (int i = 1; i <= ndofs; ++i)
   {
      offsets[i] += offsets[i - 1];
   }
   // Computation of gather_indices
   f_ind = 0;
   for (int f = 0; f < mesh.GetNumFacesWithGhost(); ++f)
   {
      Mesh::FaceInformation info = mesh.GetFaceInformation(f);
      if (info.conformity==Mesh::FaceConformity::NonConformingMaster)
      {
         continue;
      }
      if (info.IsOfFaceType(type))
      {
         SetFaceDofsGatherIndices1(info,f_ind);
         if (m==L2FaceValues::DoubleValued &&
             type==FaceType::Interior &&
             info.location==Mesh::FaceLocation::Interior)
         {
            if (info.conformity==Mesh::FaceConformity::Conforming)
            {
               PermuteAndSetFaceDofsGatherIndices2(info,f_ind);
            }
            else
            {
               SetFaceDofsGatherIndices2(info,f_ind);
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
      static constexpr int max_nd = 16*16;
      MFEM_VERIFY(nd<=max_nd, "Too many degrees of freedom.");
      MFEM_FORALL_3D(face, nf, nd, 1, 1,
      {
         MFEM_SHARED double dofs[max_nd];
         for (int side = 0; side < 2; side++)
         {
            const int config = side==0 ? conforming : interp_config_ptr[face];
            if ( config==conforming ) // No interpolation needed
            {
               MFEM_FOREACH_THREAD(dof,x,nd)
               {
                  const int i = face*nd + dof;
                  const int idx = side==0 ? d_indices1[i] : d_indices2[i];
                  if (idx>-1 && idx<threshold) // interior face
                  {
                     for (int c = 0; c < vd; ++c)
                     {
                        d_y(dof, c, side, face) = d_x(t?c:idx, t?idx:c);
                     }
                  }
                  else if (idx>=threshold) // shared interior face
                  {
                     const int sidx = idx-threshold;
                     for (int c = 0; c < vd; ++c)
                     {
                        d_y(dof, c, side, face) = d_x_shared(t?c:sidx, t?sidx:c);
                     }
                  }
                  else // true boundary
                  {
                     for (int c = 0; c < vd; ++c)
                     {
                        d_y(dof, c, side, face) = 0.0;
                     }
                  }
               }
            }
            else // Interpolation from coarse to fine
            {
               for (int c = 0; c < vd; ++c)
               {
                  MFEM_FOREACH_THREAD(dof,x,nd)
                  {
                     const int i = face*nd + dof;
                     const int idx = side==0 ? d_indices1[i] : d_indices2[i];
                     if (idx>-1 && idx<threshold) // interior face
                     {
                        dofs[dof] = d_x(t?c:idx, t?idx:c);
                     }
                     else if (idx>=threshold) // shared interior face
                     {
                        const int sidx = idx-threshold;
                        dofs[dof] = d_x_shared(t?c:sidx, t?sidx:c);
                     }
                     else // true boundary
                     {
                        dofs[dof] = 0.0;
                     }
                  }
                  MFEM_SYNC_THREAD;
                  MFEM_FOREACH_THREAD(dofOut,x,nd)
                  for (int dofOut = 0; dofOut<nd; dofOut++)
                  {
                     double res = 0.0;
                     for (int dofIn = 0; dofIn<nd; dofIn++)
                     {
                        res += interp(dofOut, dofIn, config)*dofs[dofIn];
                     }
                     d_y(dofOut, c, side, face) = res;
                  }
                  MFEM_SYNC_THREAD;
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
