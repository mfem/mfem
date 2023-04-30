// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
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
                                               ElementDofOrdering f_ordering,
                                               FaceType type)
   : H1FaceRestriction(fes, f_ordering, type, false),
     type(type),
     interpolations(fes, f_ordering, type)
{
   if (nf==0) { return; }
   x_interp.UseDevice(true);

   // Check that the space is H1 (not currently implemented for ND or RT spaces)
   const bool is_h1 = dynamic_cast<const H1_FECollection*>(fes.FEColl());
   MFEM_VERIFY(is_h1, "ParNCH1FaceRestriction is only implemented for H1 spaces.")

   CheckFESpace(f_ordering);

   ComputeScatterIndicesAndOffsets(f_ordering, type);

   ComputeGatherIndices(f_ordering, type);
}

void ParNCH1FaceRestriction::Mult(const Vector &x, Vector &y) const
{
   H1FaceRestriction::Mult(x, y);
   NonconformingInterpolation(y);
}

void ParNCH1FaceRestriction::NonconformingInterpolation(Vector& y) const
{
   // Assumes all elements have the same number of dofs
   const int nface_dofs = face_dofs;
   const int vd = vdim;
   auto d_y = Reshape(y.ReadWrite(), nface_dofs, vd, nf);
   auto &nc_interp_config = interpolations.GetNCFaceInterpConfig();
   const int num_nc_faces = nc_interp_config.Size();
   if ( num_nc_faces == 0 ) { return; }
   auto interp_config_ptr = nc_interp_config.Read();
   const int nc_size = interpolations.GetNumInterpolators();
   auto d_interp = Reshape(interpolations.GetInterpolators().Read(),
                           nface_dofs, nface_dofs, nc_size);
   static constexpr int max_nd = 16*16;
   MFEM_VERIFY(nface_dofs<=max_nd, "Too many degrees of freedom.");
   mfem::forall_2D(num_nc_faces, nface_dofs, 1, [=] MFEM_HOST_DEVICE (int nc_face)
   {
      MFEM_SHARED double dof_values[max_nd];
      const NCInterpConfig conf = interp_config_ptr[nc_face];
      if ( conf.is_non_conforming && conf.master_side == 0 )
      {
         const int interp_index = conf.index;
         const int face = conf.face_index;
         for (int c = 0; c < vd; ++c)
         {
            MFEM_FOREACH_THREAD(dof,x,nface_dofs)
            {
               dof_values[dof] = d_y(dof, c, face);
            }
            MFEM_SYNC_THREAD;
            MFEM_FOREACH_THREAD(dof_out,x,nface_dofs)
            {
               double res = 0.0;
               for (int dof_in = 0; dof_in<nface_dofs; dof_in++)
               {
                  res += d_interp(dof_out, dof_in, interp_index)*dof_values[dof_in];
               }
               d_y(dof_out, c, face) = res;
            }
            MFEM_SYNC_THREAD;
         }
      }
   });
}

void ParNCH1FaceRestriction::AddMultTranspose(const Vector &x, Vector &y,
                                              const double a) const
{
   MFEM_VERIFY(a == 1.0, "General coefficient case is not yet supported!");
   if (nf==0) { return; }
   NonconformingTransposeInterpolation(x);
   H1FaceRestriction::AddMultTranspose(x_interp, y);
}

void ParNCH1FaceRestriction::AddMultTransposeInPlace(Vector &x, Vector &y) const
{
   if (nf==0) { return; }
   NonconformingTransposeInterpolationInPlace(x);
   H1FaceRestriction::AddMultTranspose(x, y);
}

void ParNCH1FaceRestriction::NonconformingTransposeInterpolation(
   const Vector& x) const
{
   if (x_interp.Size()==0)
   {
      x_interp.SetSize(x.Size());
   }
   x_interp = x;
   NonconformingTransposeInterpolationInPlace(x_interp);
}

void ParNCH1FaceRestriction::NonconformingTransposeInterpolationInPlace(
   Vector& x) const
{
   // Assumes all elements have the same number of dofs
   const int nface_dofs = face_dofs;
   const int vd = vdim;
   if ( type==FaceType::Interior )
   {
      // Interpolation from slave to master face dofs
      auto d_x = Reshape(x.ReadWrite(), nface_dofs, vd, nf);
      auto &nc_interp_config = interpolations.GetNCFaceInterpConfig();
      const int num_nc_faces = nc_interp_config.Size();
      if ( num_nc_faces == 0 ) { return; }
      auto interp_config_ptr = nc_interp_config.Read();
      const int nc_size = interpolations.GetNumInterpolators();
      auto d_interp = Reshape(interpolations.GetInterpolators().Read(),
                              nface_dofs, nface_dofs, nc_size);
      static constexpr int max_nd = 1024;
      MFEM_VERIFY(nface_dofs<=max_nd, "Too many degrees of freedom.");
      mfem::forall_2D(num_nc_faces, nface_dofs, 1,
                      [=] MFEM_HOST_DEVICE (int nc_face)
      {
         MFEM_SHARED double dof_values[max_nd];
         const NCInterpConfig conf = interp_config_ptr[nc_face];
         const int master_side = conf.master_side;
         if ( conf.is_non_conforming && master_side==0 )
         {
            const int interp_index = conf.index;
            const int face = conf.face_index;
            // Interpolation from fine to coarse
            for (int c = 0; c < vd; ++c)
            {
               MFEM_FOREACH_THREAD(dof,x,nface_dofs)
               {
                  dof_values[dof] = d_x(dof, c, face);
               }
               MFEM_SYNC_THREAD;
               MFEM_FOREACH_THREAD(dof_out,x,nface_dofs)
               {
                  double res = 0.0;
                  for (int dof_in = 0; dof_in<nface_dofs; dof_in++)
                  {
                     res += d_interp(dof_in, dof_out, interp_index)*dof_values[dof_in];
                  }
                  d_x(dof_out, c, face) = res;
               }
               MFEM_SYNC_THREAD;
            }
         }
      });
   }
}

void ParNCH1FaceRestriction::ComputeScatterIndicesAndOffsets(
   const ElementDofOrdering f_ordering,
   const FaceType face_type)
{
   Mesh &mesh = *fes.GetMesh();

   // Initialization of the offsets
   for (int i = 0; i <= ndofs; ++i)
   {
      gather_offsets[i] = 0;
   }

   // Computation of scatter indices and offsets
   int f_ind = 0;
   for (int f = 0; f < mesh.GetNumFacesWithGhost(); ++f)
   {
      Mesh::FaceInformation face = mesh.GetFaceInformation(f);
      if ( face.IsNonconformingCoarse() )
      {
         // We skip nonconforming coarse faces as they are treated
         // by the corresponding nonconforming fine faces.
         continue;
      }
      else if (face_type==FaceType::Interior && face.IsInterior())
      {
         if ( face.IsConforming() )
         {
            interpolations.RegisterFaceConformingInterpolation(face,f_ind);
            SetFaceDofsScatterIndices(face, f_ind, f_ordering);
            f_ind++;
         }
         else // Non-conforming face
         {
            SetFaceDofsScatterIndices(face, f_ind, f_ordering);
            if ( face.element[0].conformity==Mesh::ElementConformity::Superset )
            {
               // In this case the local face is the master (coarse) face, thus
               // we need to interpolate the values on the slave (fine) face.
               interpolations.RegisterFaceCoarseToFineInterpolation(face,f_ind);
            }
            else
            {
               // Treated as a conforming face since we only extract values from
               // the local slave (fine) face.
               interpolations.RegisterFaceConformingInterpolation(face,f_ind);
            }
            f_ind++;
         }
      }
      else if (face_type==FaceType::Boundary && face.IsBoundary())
      {
         SetFaceDofsScatterIndices(face, f_ind, f_ordering);
         f_ind++;
      }
   }
   MFEM_VERIFY(f_ind==nf, "Unexpected number of faces.");

   // Summation of the offsets
   for (int i = 1; i <= ndofs; ++i)
   {
      gather_offsets[i] += gather_offsets[i - 1];
   }

   // Transform the interpolation matrix map into a contiguous memory structure.
   interpolations.LinearizeInterpolatorMapIntoVector();
   interpolations.InitializeNCInterpConfig();
}

void ParNCH1FaceRestriction::ComputeGatherIndices(
   const ElementDofOrdering f_ordering,
   const FaceType face_type)
{
   Mesh &mesh = *fes.GetMesh();

   // Computation of gather_indices
   int f_ind = 0;
   for (int f = 0; f < mesh.GetNumFacesWithGhost(); ++f)
   {
      Mesh::FaceInformation face = mesh.GetFaceInformation(f);
      if ( face.IsNonconformingCoarse() )
      {
         // We skip nonconforming coarse faces as they are treated
         // by the corresponding nonconforming fine faces.
         continue;
      }
      else if (face.IsOfFaceType(face_type))
      {
         SetFaceDofsGatherIndices(face, f_ind, f_ordering);
         f_ind++;
      }
   }
   MFEM_VERIFY(f_ind==nf, "Unexpected number of faces.");

   // Reset offsets to their correct value
   for (int i = ndofs; i > 0; --i)
   {
      gather_offsets[i] = gather_offsets[i - 1];
   }
   gather_offsets[0] = 0;
}

ParL2FaceRestriction::ParL2FaceRestriction(const ParFiniteElementSpace &fes,
                                           ElementDofOrdering f_ordering,
                                           FaceType type,
                                           L2FaceValues m,
                                           bool build)
   : L2FaceRestriction(fes, f_ordering, type, m, false)
{
   if (!build) { return; }
   if (nf==0) { return; }

   CheckFESpace(f_ordering);

   ComputeScatterIndicesAndOffsets(f_ordering, type);

   ComputeGatherIndices(f_ordering, type);
}

ParL2FaceRestriction::ParL2FaceRestriction(const ParFiniteElementSpace &fes,
                                           ElementDofOrdering f_ordering,
                                           FaceType type,
                                           L2FaceValues m)
   : ParL2FaceRestriction(fes, f_ordering, type, m, true)
{ }

void ParL2FaceRestriction::DoubleValuedConformingMult(
   const Vector& x, Vector& y) const
{
   MFEM_ASSERT(
      m == L2FaceValues::DoubleValued,
      "This method should be called when m == L2FaceValues::DoubleValued.");
   const ParFiniteElementSpace &pfes =
      static_cast<const ParFiniteElementSpace&>(this->fes);
   ParGridFunction x_gf;
   x_gf.MakeRef(const_cast<ParFiniteElementSpace*>(&pfes),
                const_cast<Vector&>(x), 0);
   x_gf.ExchangeFaceNbrData();

   // Assumes all elements have the same number of dofs
   const int nface_dofs = face_dofs;
   const int vd = vdim;
   const bool t = byvdim;
   const int threshold = ndofs;
   const int nsdofs = pfes.GetFaceNbrVSize();
   auto d_indices1 = scatter_indices1.Read();
   auto d_indices2 = scatter_indices2.Read();
   auto d_x = Reshape(x.Read(), t?vd:ndofs, t?ndofs:vd);
   auto d_x_shared = Reshape(x_gf.FaceNbrData().Read(),
                             t?vd:nsdofs, t?nsdofs:vd);
   auto d_y = Reshape(y.Write(), nface_dofs, vd, 2, nf);
   mfem::forall(nfdofs, [=] MFEM_HOST_DEVICE (int i)
   {
      const int dof = i % nface_dofs;
      const int face = i / nface_dofs;
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

void ParL2FaceRestriction::Mult(const Vector& x, Vector& y) const
{
   if (nf==0) { return; }
   if (m==L2FaceValues::DoubleValued)
   {
      DoubleValuedConformingMult(x, y);
   }
   else
   {
      SingleValuedConformingMult(x, y);
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
   const int nface_dofs = face_dofs;
   const int Ndofs = ndofs;
   auto d_indices1 = scatter_indices1.Read();
   auto d_indices2 = scatter_indices2.Read();
   auto I = mat.ReadWriteI();
   mfem::forall(nf*nface_dofs, [=] MFEM_HOST_DEVICE (int fdof)
   {
      const int f  = fdof/nface_dofs;
      const int iF = fdof%nface_dofs;
      const int iE1 = d_indices1[f*nface_dofs+iF];
      if (iE1 < Ndofs)
      {
         AddNnz(iE1,I,nface_dofs);
      }
      const int iE2 = d_indices2[f*nface_dofs+iF];
      if (iE2 < Ndofs)
      {
         AddNnz(iE2,I,nface_dofs);
      }
   });
}

void ParL2FaceRestriction::FillI(SparseMatrix &mat,
                                 SparseMatrix &face_mat) const
{
   const int nface_dofs = face_dofs;
   const int Ndofs = ndofs;
   auto d_indices1 = scatter_indices1.Read();
   auto d_indices2 = scatter_indices2.Read();
   auto I = mat.ReadWriteI();
   auto I_face = face_mat.ReadWriteI();
   mfem::forall(ne*elem_dofs*vdim+1, [=] MFEM_HOST_DEVICE (int i)
   {
      I_face[i] = 0;
   });
   mfem::forall(nf*nface_dofs, [=] MFEM_HOST_DEVICE (int fdof)
   {
      const int f  = fdof/nface_dofs;
      const int iF = fdof%nface_dofs;
      const int iE1 = d_indices1[f*nface_dofs+iF];
      if (iE1 < Ndofs)
      {
         for (int jF = 0; jF < nface_dofs; jF++)
         {
            const int jE2 = d_indices2[f*nface_dofs+jF];
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
      const int iE2 = d_indices2[f*nface_dofs+iF];
      if (iE2 < Ndofs)
      {
         for (int jF = 0; jF < nface_dofs; jF++)
         {
            const int jE1 = d_indices1[f*nface_dofs+jF];
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
   const int nface_dofs = face_dofs;
   const int Ndofs = ndofs;
   auto d_indices1 = scatter_indices1.Read();
   auto d_indices2 = scatter_indices2.Read();
   auto mat_fea = Reshape(ea_data.Read(), nface_dofs, nface_dofs, 2, nf);
   auto I = mat.ReadWriteI();
   auto J = mat.WriteJ();
   auto Data = mat.WriteData();
   mfem::forall(nf*nface_dofs, [=] MFEM_HOST_DEVICE (int fdof)
   {
      const int f  = fdof/nface_dofs;
      const int iF = fdof%nface_dofs;
      const int iE1 = d_indices1[f*nface_dofs+iF];
      if (iE1 < Ndofs)
      {
         const int offset = AddNnz(iE1,I,nface_dofs);
         for (int jF = 0; jF < nface_dofs; jF++)
         {
            const int jE2 = d_indices2[f*nface_dofs+jF];
            J[offset+jF] = jE2;
            Data[offset+jF] = mat_fea(jF,iF,1,f);
         }
      }
      const int iE2 = d_indices2[f*nface_dofs+iF];
      if (iE2 < Ndofs)
      {
         const int offset = AddNnz(iE2,I,nface_dofs);
         for (int jF = 0; jF < nface_dofs; jF++)
         {
            const int jE1 = d_indices1[f*nface_dofs+jF];
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
   const int nface_dofs = face_dofs;
   const int Ndofs = ndofs;
   auto d_indices1 = scatter_indices1.Read();
   auto d_indices2 = scatter_indices2.Read();
   auto mat_fea = Reshape(ea_data.Read(), nface_dofs, nface_dofs, 2, nf);
   auto I = mat.ReadWriteI();
   auto I_face = face_mat.ReadWriteI();
   auto J = mat.WriteJ();
   auto J_face = face_mat.WriteJ();
   auto Data = mat.WriteData();
   auto Data_face = face_mat.WriteData();
   mfem::forall(nf*nface_dofs, [=] MFEM_HOST_DEVICE (int fdof)
   {
      const int f  = fdof/nface_dofs;
      const int iF = fdof%nface_dofs;
      const int iE1 = d_indices1[f*nface_dofs+iF];
      if (iE1 < Ndofs)
      {
         for (int jF = 0; jF < nface_dofs; jF++)
         {
            const int jE2 = d_indices2[f*nface_dofs+jF];
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
      const int iE2 = d_indices2[f*nface_dofs+iF];
      if (iE2 < Ndofs)
      {
         for (int jF = 0; jF < nface_dofs; jF++)
         {
            const int jE1 = d_indices1[f*nface_dofs+jF];
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
   const ElementDofOrdering f_ordering,
   const FaceType type)
{
   Mesh &mesh = *fes.GetMesh();
   const ParFiniteElementSpace &pfes =
      static_cast<const ParFiniteElementSpace&>(this->fes);

   // Initialization of the offsets
   for (int i = 0; i <= ndofs; ++i)
   {
      gather_offsets[i] = 0;
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
      gather_offsets[i] += gather_offsets[i - 1];
   }
}


void ParL2FaceRestriction::ComputeGatherIndices(
   const ElementDofOrdering f_ordering,
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
      gather_offsets[i] = gather_offsets[i - 1];
   }
   gather_offsets[0] = 0;
}

ParNCL2FaceRestriction::ParNCL2FaceRestriction(const ParFiniteElementSpace &fes,
                                               ElementDofOrdering f_ordering,
                                               FaceType type,
                                               L2FaceValues m)
   : L2FaceRestriction(fes, f_ordering, type, m, false),
     NCL2FaceRestriction(fes, f_ordering, type, m, false),
     ParL2FaceRestriction(fes, f_ordering, type, m, false)
{
   if (nf==0) { return; }
   x_interp.UseDevice(true);

   CheckFESpace(f_ordering);

   ComputeScatterIndicesAndOffsets(f_ordering, type);

   ComputeGatherIndices(f_ordering, type);
}

void ParNCL2FaceRestriction::SingleValuedNonconformingMult(
   const Vector& x, Vector& y) const
{
   MFEM_ASSERT(
      m == L2FaceValues::SingleValued,
      "This method should be called when m == L2FaceValues::SingleValued.");
   // Assumes all elements have the same number of dofs
   const int nface_dofs = face_dofs;
   const int vd = vdim;
   const bool t = byvdim;
   const int threshold = ndofs;
   auto d_indices1 = scatter_indices1.Read();
   auto d_x = Reshape(x.Read(), t?vd:ndofs, t?ndofs:vd);
   auto d_y = Reshape(y.Write(), nface_dofs, vd, nf);
   auto interp_config_ptr = interpolations.GetFaceInterpConfig().Read();
   auto interpolators = interpolations.GetInterpolators().Read();
   const int nc_size = interpolations.GetNumInterpolators();
   auto d_interp = Reshape(interpolators, nface_dofs, nface_dofs, nc_size);
   static constexpr int max_nd = 16*16;
   MFEM_VERIFY(nface_dofs<=max_nd, "Too many degrees of freedom.");
   mfem::forall_2D(nf, nface_dofs, 1, [=] MFEM_HOST_DEVICE (int face)
   {
      MFEM_SHARED double dof_values[max_nd];
      const InterpConfig conf = interp_config_ptr[face];
      const int master_side = conf.master_side;
      const int interp_index = conf.index;
      const int side = 0;
      if ( !conf.is_non_conforming || side!=master_side )
      {
         MFEM_FOREACH_THREAD(dof,x,nface_dofs)
         {
            const int i = face*nface_dofs + dof;
            const int idx = d_indices1[i];
            if (idx>-1 && idx<threshold) // interior face
            {
               for (int c = 0; c < vd; ++c)
               {
                  d_y(dof, c, face) = d_x(t?c:idx, t?idx:c);
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
            MFEM_FOREACH_THREAD(dof,x,nface_dofs)
            {
               const int i = face*nface_dofs + dof;
               const int idx = d_indices1[i];
               if (idx>-1 && idx<threshold) // interior face
               {
                  dof_values[dof] = d_x(t?c:idx, t?idx:c);
               }
               else // true boundary
               {
                  dof_values[dof] = 0.0;
               }
            }
            MFEM_SYNC_THREAD;
            MFEM_FOREACH_THREAD(dof_out,x,nface_dofs)
            {
               double res = 0.0;
               for (int dof_in = 0; dof_in<nface_dofs; dof_in++)
               {
                  res += d_interp(dof_out, dof_in, interp_index)*dof_values[dof_in];
               }
               d_y(dof_out, c, face) = res;
            }
            MFEM_SYNC_THREAD;
         }
      }
   });
}

void ParNCL2FaceRestriction::DoubleValuedNonconformingMult(
   const Vector& x, Vector& y) const
{
   ParL2FaceRestriction::DoubleValuedConformingMult(x, y);
   NCL2FaceRestriction::DoubleValuedNonconformingInterpolation(y);
}

void ParNCL2FaceRestriction::Mult(const Vector& x, Vector& y) const
{
   if (nf==0) { return; }
   if ( type==FaceType::Interior && m==L2FaceValues::DoubleValued )
   {
      DoubleValuedNonconformingMult(x, y);
   }
   else if ( type==FaceType::Boundary && m==L2FaceValues::DoubleValued )
   {
      DoubleValuedConformingMult(x, y);
   }
   else if ( type==FaceType::Interior && m==L2FaceValues::SingleValued )
   {
      SingleValuedNonconformingMult(x, y);
   }
   else if ( type==FaceType::Boundary && m==L2FaceValues::SingleValued )
   {
      SingleValuedConformingMult(x, y);
   }
   else
   {
      MFEM_ABORT("Unknown type and multiplicity combination.");
   }
}

void ParNCL2FaceRestriction::AddMultTranspose(const Vector &x, Vector &y,
                                              const double a) const
{
   MFEM_VERIFY(a == 1.0, "General coefficient case is not yet supported!");
   if (nf==0) { return; }
   if (type==FaceType::Interior)
   {
      if ( m==L2FaceValues::DoubleValued )
      {
         DoubleValuedNonconformingTransposeInterpolation(x);
         DoubleValuedConformingAddMultTranspose(x_interp, y);
      }
      else // Single Valued
      {
         SingleValuedNonconformingTransposeInterpolation(x);
         SingleValuedConformingAddMultTranspose(x_interp, y);
      }
   }
   else
   {
      if ( m==L2FaceValues::DoubleValued )
      {
         DoubleValuedConformingAddMultTranspose(x, y);
      }
      else // Single valued
      {
         SingleValuedConformingAddMultTranspose(x, y);
      }
   }
}

void ParNCL2FaceRestriction::AddMultTransposeInPlace(Vector& x, Vector& y) const
{
   if (nf==0) { return; }
   if (type==FaceType::Interior)
   {
      if ( m==L2FaceValues::DoubleValued )
      {
         DoubleValuedNonconformingTransposeInterpolationInPlace(x);
         DoubleValuedConformingAddMultTranspose(x, y);
      }
      else if ( m==L2FaceValues::SingleValued )
      {
         SingleValuedNonconformingTransposeInterpolationInPlace(x);
         SingleValuedConformingAddMultTranspose(x, y);
      }
   }
   else
   {
      if ( m==L2FaceValues::DoubleValued )
      {
         DoubleValuedConformingAddMultTranspose(x, y);
      }
      else if ( m==L2FaceValues::SingleValued )
      {
         SingleValuedConformingAddMultTranspose(x, y);
      }
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
   const ElementDofOrdering f_ordering,
   const FaceType type)
{
   Mesh &mesh = *fes.GetMesh();

   // Initialization of the offsets
   for (int i = 0; i <= ndofs; ++i)
   {
      gather_offsets[i] = 0;
   }

   // Computation of scatter and offsets indices
   int f_ind=0;
   for (int f = 0; f < mesh.GetNumFacesWithGhost(); ++f)
   {
      Mesh::FaceInformation face = mesh.GetFaceInformation(f);
      if ( face.IsNonconformingCoarse() )
      {
         // We skip nonconforming coarse faces as they are treated
         // by the corresponding nonconforming fine faces.
         continue;
      }
      else if ( type==FaceType::Interior && face.IsInterior() )
      {
         if ( face.IsConforming() )
         {
            interpolations.RegisterFaceConformingInterpolation(face,f_ind);
            SetFaceDofsScatterIndices1(face,f_ind);
            if ( m==L2FaceValues::DoubleValued )
            {
               if ( face.IsShared() )
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
            interpolations.RegisterFaceCoarseToFineInterpolation(face,f_ind);
            SetFaceDofsScatterIndices1(face,f_ind);
            if ( m==L2FaceValues::DoubleValued )
            {
               if ( face.IsShared() )
               {
                  PermuteAndSetSharedFaceDofsScatterIndices2(face,f_ind);
               }
               else // local nonconforming slave
               {
                  PermuteAndSetFaceDofsScatterIndices2(face,f_ind);
               }
            }
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
      gather_offsets[i] += gather_offsets[i - 1];
   }

   // Transform the interpolation matrix map into a contiguous memory structure.
   interpolations.LinearizeInterpolatorMapIntoVector();
   interpolations.InitializeNCInterpConfig();
}

void ParNCL2FaceRestriction::ComputeGatherIndices(
   const ElementDofOrdering f_ordering,
   const FaceType type)
{
   Mesh &mesh = *fes.GetMesh();

   // Computation of gather_indices
   int f_ind = 0;
   for (int f = 0; f < mesh.GetNumFacesWithGhost(); ++f)
   {
      Mesh::FaceInformation face = mesh.GetFaceInformation(f);
      if ( face.IsNonconformingCoarse() )
      {
         // We skip nonconforming coarse faces as they are treated
         // by the corresponding nonconforming fine faces.
         continue;
      }
      else if ( face.IsOfFaceType(type) )
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
   MFEM_VERIFY(f_ind==nf, "Unexpected number of " <<
               (type==FaceType::Interior? "interior" : "boundary") <<
               " faces: " << f_ind << " vs " << nf );

   // Switch back offsets to their correct value
   for (int i = ndofs; i > 0; --i)
   {
      gather_offsets[i] = gather_offsets[i - 1];
   }
   gather_offsets[0] = 0;
}

} // namespace mfem

#endif
