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

ParNCH1FaceRestriction::ParNCH1FaceRestriction(const ParFiniteElementSpace &fes,
                                               ElementDofOrdering ordering,
                                               FaceType type)
   : H1FaceRestriction(fes,type),
     type(type),
     interpolations(fes,ordering,type)
{
   if (nf==0) { return; }
   // If fespace == H1
   const FiniteElement *fe = fes.GetFE(0);
   const TensorBasisElement *tfe = dynamic_cast<const TensorBasisElement*>(fe);
   MFEM_VERIFY(tfe != NULL &&
               (tfe->GetBasisType()==BasisType::GaussLobatto ||
                tfe->GetBasisType()==BasisType::Positive),
               "Only Gauss-Lobatto and Bernstein basis are supported in "
               "H1FaceRestriction.");

   // Assuming all finite elements are using Gauss-Lobatto.
   height = vdim*nf*dof;
   width = fes.GetVSize();
   const bool dof_reorder = (ordering == ElementDofOrdering::LEXICOGRAPHIC);
   if (dof_reorder && nf > 0)
   {
      for (int f = 0; f < fes.GetNF(); ++f)
      {
         const FiniteElement *fe = fes.GetFaceElement(f);
         const TensorBasisElement* el =
            dynamic_cast<const TensorBasisElement*>(fe);
         if (el) { continue; }
         MFEM_ABORT("Finite element not suitable for lexicographic ordering");
      }
      const FiniteElement *fe = fes.GetFaceElement(0);
      const TensorBasisElement* el =
         dynamic_cast<const TensorBasisElement*>(fe);
      const Array<int> &fe_dof_map = el->GetDofMap();
      MFEM_VERIFY(fe_dof_map.Size() > 0, "invalid dof map");
   }
   // End of verifications

   ComputeScatterIndicesAndOffsets(ordering, type);

   ComputeGatherIndices(ordering, type);
}

void ParNCH1FaceRestriction::Mult(const Vector &x, Vector &y) const
{
   // Assumes all elements have the same number of dofs
   const int nd = dof;
   const int vd = vdim;
   const bool t = byvdim;

   if ( type==FaceType::Boundary )
   {
      auto d_indices = scatter_indices.Read();
      auto d_x = Reshape(x.Read(), t?vd:ndofs, t?ndofs:vd);
      auto d_y = Reshape(y.Write(), nd, vd, nf);
      MFEM_FORALL(i, nfdofs,
      {
         const int dof = i % nd;
         const int face = i / nd;
         const int idx = d_indices[i];
         for (int c = 0; c < vd; ++c)
         {
            d_y(dof, c, face) = d_x(t?c:idx, t?idx:c);
         }
      });
   }
   else // type==FaceType::Interior
   {
      auto d_indices = scatter_indices.Read();
      auto d_x = Reshape(x.Read(), t?vd:ndofs, t?ndofs:vd);
      auto d_y = Reshape(y.Write(), nd, vd, nf);
      auto interp_config_ptr = interpolations.GetFaceInterpConfig().Read();
      auto interpolators = interpolations.GetInterpolators().Read();
      const int nc_size = interpolations.GetNumInterpolators();
      auto interp = Reshape(interpolators, nd, nd, nc_size);
      static constexpr int max_nd = 16*16;
      MFEM_VERIFY(nd<=max_nd, "Too many degrees of freedom.");
      MFEM_FORALL_3D(face, nf, nd, 1, 1,
      {
         MFEM_SHARED double dofs[max_nd];
         const InterpConfig conf = interp_config_ptr[face];
         const int master_side = conf.GetNonConformingMasterSide();
         const int interp_index = conf.GetInterpolatorIndex();
         const int side = 0;
         if ( interp_index==InterpConfig::conforming || side!=master_side )
         {
            MFEM_FOREACH_THREAD(dof,x,nd)
            {
               const int i = face*nd + dof;
               const int idx = d_indices[i];
               for (int c = 0; c < vd; ++c)
               {
                  d_y(dof, c, face) = d_x(t?c:idx, t?idx:c);
               }
            }
         }
         else // Interpolation from coarse to fine
         {
            for (int c = 0; c < vd; ++c)
            {
               // Load the face dofs in shared memory
               MFEM_FOREACH_THREAD(dof,x,nd)
               {
                  const int i = face*nd + dof;
                  const int idx = d_indices[i];
                  dofs[dof] = d_x(t?c:idx, t?idx:c);
               }
               MFEM_SYNC_THREAD;
               // Apply the interpolation to the face dofs
               MFEM_FOREACH_THREAD(dofOut,x,nd)
               for (int dofOut = 0; dofOut<nd; dofOut++)
               {
                  double res = 0.0;
                  for (int dofIn = 0; dofIn<nd; dofIn++)
                  {
                     res += interp(dofOut, dofIn, interp_index)*dofs[dofIn];
                  }
                  d_y(dofOut, c, face) = res;
               }
               MFEM_SYNC_THREAD;
            }
         }
      });
   }
}

void ParNCH1FaceRestriction::AddMultTranspose(const Vector &x, Vector &y) const
{
   // Assumes all elements have the same number of dofs
   const int nd = dof;
   const int vd = vdim;
   const bool t = byvdim;
   if ( type==FaceType::Interior )
   {
      // Interpolation from slave to master face dofs
      // FIXME: Currently this is modifying `x`, otherwise we need a temporary?
      // TODO: Consider different algorithm
      auto d_x = Reshape(const_cast<Vector&>(x).ReadWrite(), nd, vd, nf);
      auto interp_config_ptr = interpolations.GetFaceInterpConfig().Read();
      auto interpolators = interpolations.GetInterpolators().Read();
      const int nc_size = interpolations.GetNumInterpolators();
      auto interp = Reshape(interpolators, nd, nd, nc_size);
      static constexpr int max_nd = 16*16;
      MFEM_VERIFY(nd<=max_nd, "Too many degrees of freedom.");
      MFEM_FORALL_3D(face, nf, nd, 1, 1,
      {
         MFEM_SHARED double dofs[max_nd];
         const InterpConfig conf = interp_config_ptr[face];
         const int interp_index = conf.GetInterpolatorIndex();
         if ( interp_index!=InterpConfig::conforming )
         {
            // Interpolation from fine to coarse
            for (int c = 0; c < vd; ++c)
            {
               MFEM_FOREACH_THREAD(dof,x,nd)
               {
                  dofs[dof] = d_x(dof, c, face);
               }
               MFEM_SYNC_THREAD;
               MFEM_FOREACH_THREAD(dofOut,x,nd)
               {
                  double res = 0.0;
                  for (int dofIn = 0; dofIn<nd; dofIn++)
                  {
                     res += interp(dofIn, dofOut, interp_index)*dofs[dofIn];
                  }
                  d_x(dofOut, c, face) = res; // TODO write directly in d_y?
               }
               MFEM_SYNC_THREAD;
            }
         }
      });
   }

   // Gathering of face dofs into element dofs
   auto d_offsets = offsets.Read();
   auto d_indices = gather_indices.Read();
   auto d_x = Reshape(x.Read(), nd, vd, nf);
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
            int idx_j = d_indices[j];
            // TODO: Check if conforming?
            dofValue +=  d_x(idx_j % nd, c, idx_j / nd);
         }
         d_y(t?c:i,t?i:c) += dofValue;
      }
   });
}

void ParNCH1FaceRestriction::ComputeScatterIndicesAndOffsets(
   const ElementDofOrdering ordering,
   const FaceType type)
{
   Mesh &mesh = *fes.GetMesh();

   // Initialization of the offsets
   for (int i = 0; i <= ndofs; ++i)
   {
      offsets[i] = 0;
   }

   // Computation of scatter indices and offsets
   int f_ind = 0;
   for (int f = 0; f < mesh.GetNumFacesWithGhost(); ++f)
   {
      Mesh::FaceInformation face = mesh.GetFaceInformation(f);
      if (type==FaceType::Interior && face.IsInterior())
      {
         if ( face.IsConforming() )
         {
            interpolations.RegisterFaceConformingInterpolation(face,f_ind);
            SetFaceDofsScatterIndices(face, f_ind, ordering);
            f_ind++;
         }
         else // Non-conforming face
         {
            if (face.IsShared())
            {
               // In this case the local face is the master (coarse) face, thus
               // we need to interpolate the values on the slave (fine) face.
               interpolations.RegisterFaceCoarseToFineInterpolation(face,f_ind);
               SetFaceDofsScatterIndices(face, f_ind, ordering);
            }
            else
            {
               // Treated as a conforming face since we only extract values from
               // the local slave (fine) face.
               interpolations.RegisterFaceConformingInterpolation(face,f_ind);
               SetFaceDofsScatterIndices(face, f_ind, ordering);
            }
            f_ind++;
         }
      }
      else if (type==FaceType::Boundary && face.IsBoundary())
      {
         SetFaceDofsScatterIndices(face, f_ind, ordering);
         f_ind++;
      }
   }
   MFEM_VERIFY(f_ind==nf, "Unexpected number of faces.");

   // Summation of the offsets
   for (int i = 1; i <= ndofs; ++i)
   {
      offsets[i] += offsets[i - 1];
   }

   // Transform the interpolation matrix map into a contiguous memory structure.
   interpolations.LinearizeInterpolatorMapIntoVector();
}

void ParNCH1FaceRestriction::ComputeGatherIndices(
   const ElementDofOrdering ordering,
   const FaceType type)
{
   Mesh &mesh = *fes.GetMesh();

   // Computation of gather_indices
   int f_ind = 0;
   for (int f = 0; f < mesh.GetNumFacesWithGhost(); ++f)
   {
      Mesh::FaceInformation face = mesh.GetFaceInformation(f);
      if (face.IsOfFaceType(type))
      {
         SetFaceDofsGatherIndices(face, f_ind, ordering);
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
                                           ElementDofOrdering ordering,
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
   const bool dof_reorder = (ordering == ElementDofOrdering::LEXICOGRAPHIC);
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

   ComputeScatterIndicesAndOffsets(ordering, type);

   ComputeGatherIndices(ordering, type);
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

   // Computation of gather_indices
   int f_ind = 0;
   for (int f = 0; f < fes.GetNF(); ++f)
   {
      Mesh::FaceInformation face = mesh.GetFaceInformation(f);
      if (face.IsOfFaceType(type))
      {
         SetFaceDofsGatherIndices1(face,f_ind);
         if (m==L2FaceValues::DoubleValued &&
             type==FaceType::Interior &&
             face.IsLocal())
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
                                               ElementDofOrdering ordering,
                                               FaceType type,
                                               L2FaceValues m)
   : L2FaceRestriction(fes, type, m), interpolations(fes, ordering, type)
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
   const bool dof_reorder = (ordering==ElementDofOrdering::LEXICOGRAPHIC);
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

   ComputeScatterIndicesAndOffsets(ordering, type);

   ComputeGatherIndices(ordering, type);
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
      auto interp_config_ptr = interpolations.GetFaceInterpConfig().Read();
      auto interpolators = interpolations.GetInterpolators().Read();
      const int nc_size = interpolations.GetNumInterpolators();
      auto interp = Reshape(interpolators, nd, nd, nc_size);
      static constexpr int max_nd = 16*16;
      MFEM_VERIFY(nd<=max_nd, "Too many degrees of freedom.");
      MFEM_FORALL_3D(face, nf, nd, 1, 1,
      {
         MFEM_SHARED double dofs[max_nd];
         const InterpConfig conf = interp_config_ptr[face];
         const int master_side = conf.GetNonConformingMasterSide();
         const int interp_index = conf.GetInterpolatorIndex();
         for (int side = 0; side < 2; side++)
         {
            if ( interp_index==InterpConfig::conforming || side!=master_side )
            {
               MFEM_FOREACH_THREAD(dof,x,nd)
               {
                  const int i = face*nd + dof;
                  const int idx = side==0 ? d_indices1[i] : d_indices2[i];
                  if (idx>-1 && idx<threshold) // local interior face
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
                     if (idx>-1 && idx<threshold) // local interior face
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
                        res += interp(dofOut, dofIn, interp_index)*dofs[dofIn];
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
         for (int c = 0; c < vd; ++c)
         {
            d_y(dof, c, 1, face) = 0.0;
         }
      });
   }
   else if ( type==FaceType::Interior && m==L2FaceValues::SingleValued )
   {
      auto d_indices1 = scatter_indices1.Read();
      auto d_x = Reshape(x.Read(), t?vd:ndofs, t?ndofs:vd);
      auto d_y = Reshape(y.Write(), nd, vd, nf);
      auto interp_config_ptr = interpolations.GetFaceInterpConfig().Read();
      auto interpolators = interpolations.GetInterpolators().Read();
      const int nc_size = interpolations.GetNumInterpolators();
      auto interp = Reshape(interpolators, nd, nd, nc_size);
      static constexpr int max_nd = 16*16;
      MFEM_VERIFY(nd<=max_nd, "Too many degrees of freedom.");
      MFEM_FORALL_3D(face, nf, nd, 1, 1,
      {
         MFEM_SHARED double dofs[max_nd];
         const InterpConfig conf = interp_config_ptr[face];
         const int master_side = conf.GetNonConformingMasterSide();
         const int interp_index = conf.GetInterpolatorIndex();
         const int side = 0;
         if ( interp_index==InterpConfig::conforming || side!=master_side )
         {
            MFEM_FOREACH_THREAD(dof,x,nd)
            {
               const int i = face*nd + dof;
               const int idx = d_indices1[i];
               if (idx>-1 && idx<threshold) // interior face
               {
                  for (int c = 0; c < vd; ++c)
                  {
                     d_y(dof, c, face) = d_x(t?c:idx, t?idx:c);
                  }
               }
               else if (idx>=threshold) // shared interior face
               {
                  const int sidx = idx-threshold;
                  for (int c = 0; c < vd; ++c)
                  {
                     d_y(dof, c, face) = d_x_shared(t?c:sidx, t?sidx:c);
                  }
               }
               else // true boundary
               {
                  for (int c = 0; c < vd; ++c)
                  {
                     d_y(dof, c, face) = 0.0;
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
                  const int idx = d_indices1[i];
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
                     res += interp(dofOut, dofIn, interp_index)*dofs[dofIn];
                  }
                  d_y(dofOut, c, face) = res;
               }
               MFEM_SYNC_THREAD;
            }
         }
      });
   }
   else if ( type==FaceType::Interior && m==L2FaceValues::SingleValued )
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
   else
   {
      MFEM_ABORT("Unknown type and multiplicity combination.");
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

void ParNCL2FaceRestriction::ComputeScatterIndicesAndOffsets(
   const ElementDofOrdering ordering,
   const FaceType type)
{
   Mesh &mesh = *fes.GetMesh();

   // Initialization of the offsets
   for (int i = 0; i <= ndofs; ++i)
   {
      offsets[i] = 0;
   }

   // Computation of scatter and offsets indices
   int f_ind=0;
   for (int f = 0; f < mesh.GetNumFacesWithGhost(); ++f)
   {
      Mesh::FaceInformation face = mesh.GetFaceInformation(f);
      if (type==FaceType::Interior && face.IsInterior())
      {
         if ( face.IsConforming() )
         {
            interpolations.RegisterFaceConformingInterpolation(face,f_ind);
            SetFaceDofsScatterIndices1(face,f_ind);
            if ( m==L2FaceValues::DoubleValued )
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
         }
         else // Non-conforming face
         {
            if (face.IsShared())
            {
               // TODO: not sure what should happen.
               // We swap elem1 and elem2 to have elem1 slave and elem2 master
               // face.SwapElem1AndElem2(); // TODO: check that this is correct.
               // SetSharedFaceDofsScatterIndices1(face,f_ind);
            }
            else
            {
               SetFaceDofsScatterIndices1(face,f_ind);
            }
            interpolations.RegisterFaceCoarseToFineInterpolation(face,f_ind);
            SetFaceDofsScatterIndices2(face,f_ind);
            // if (face.IsInterior())
            // {
            //    SetFaceDofsScatterIndices2(face,f_ind);
            // }
            // else if (face.IsShared())
            // {
            //    SetSharedFaceDofsScatterIndices2(face,f_ind);
            // }
         }
         f_ind++;
      }
      else if (type==FaceType::Boundary && face.IsBoundary())
      {
         SetFaceDofsScatterIndices1(face,f_ind);
         if ( m==L2FaceValues::DoubleValued )
         {
            SetBoundaryDofsScatterIndices2(face,f_ind);
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

   // Transform the interpolation matrix map into a contiguous memory structure.
   interpolations.LinearizeInterpolatorMapIntoVector();
}

void ParNCL2FaceRestriction::ComputeGatherIndices(
   const ElementDofOrdering ordering,
   const FaceType type)
{
   Mesh &mesh = *fes.GetMesh();
   // Computation of gather_indices
   int f_ind = 0;
   for (int f = 0; f < mesh.GetNumFacesWithGhost(); ++f)
   {
      Mesh::FaceInformation face = mesh.GetFaceInformation(f);
      if (face.IsOfFaceType(type))
      {
         // I should probably shift by nfdofs on ghost faces
         if (m==L2FaceValues::DoubleValued && face.IsGhost())
         {
            /* TODO: code */
         }
         SetFaceDofsGatherIndices1(face,f_ind);
         if (m==L2FaceValues::DoubleValued &&
             type==FaceType::Interior &&
             face.IsLocal())
         {
            if (face.IsConforming())
            {
               PermuteAndSetFaceDofsGatherIndices2(face,f_ind);
            }
            else
            {
               SetFaceDofsGatherIndices2(face,f_ind);
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
}

} // namespace mfem

#endif
