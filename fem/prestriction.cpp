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
   : fes(fes),
     nf(fes.GetNFbyType(type)),
     vdim(fes.GetVDim()),
     byvdim(fes.GetOrdering() == Ordering::byVDIM),
     ndofs(fes.GetNDofs()),
     dof(nf>0 ?
         fes.GetTraceElement(0, fes.GetMesh()->GetFaceBaseGeometry(0))->GetDof()
         : 0),
     m(m),
     nfdofs(nf*dof),
     scatter_indices1(nf*dof),
     scatter_indices2(m==L2FaceValues::DoubleValued?nf*dof:0),
     offsets(ndofs+1),
     gather_indices((m==L2FaceValues::DoubleValued? 2 : 1)*nf*dof)
{
   if (nf==0) { return; }
   // If fespace == L2
   const FiniteElement *fe = fes.GetFE(0);
   const TensorBasisElement *tfe = dynamic_cast<const TensorBasisElement*>(fe);
   MFEM_VERIFY(tfe != NULL &&
               (tfe->GetBasisType()==BasisType::GaussLobatto ||
                tfe->GetBasisType()==BasisType::Positive),
               "Only Gauss-Lobatto and Bernstein basis are supported in "
               "ParL2FaceRestriction.");
   MFEM_VERIFY(fes.GetMesh()->Conforming(),
               "Non-conforming meshes not yet supported with partial assembly.");
   // Assuming all finite elements are using Gauss-Lobatto dofs
   height = (m==L2FaceValues::DoubleValued? 2 : 1)*vdim*nf*dof;
   width = fes.GetVSize();
   const bool dof_reorder = (e_ordering == ElementDofOrdering::LEXICOGRAPHIC);
   if (!dof_reorder)
   {
      mfem_error("Non-Tensor L2FaceRestriction not yet implemented.");
   }
   if (dof_reorder && nf > 0)
   {
      for (int f = 0; f < fes.GetNF(); ++f)
      {
         const FiniteElement *fe =
            fes.GetTraceElement(f, fes.GetMesh()->GetFaceBaseGeometry(f));
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
   // Computation of scatter indices
   int f_ind=0;
   for (int f = 0; f < fes.GetNF(); ++f)
   {
      fes.GetMesh()->GetFaceElements(f, &e1, &e2);
      fes.GetMesh()->GetFaceInfos(f, &inf1, &inf2);
      if (dof_reorder)
      {
         orientation = inf1 % 64;
         face_id1 = inf1 / 64;
         GetFaceDofs(dim, face_id1, dof1d, faceMap1); // only for hex
         orientation = inf2 % 64;
         face_id2 = inf2 / 64;
         GetFaceDofs(dim, face_id2, dof1d, faceMap2); // only for hex
      }
      else
      {
         mfem_error("FaceRestriction not yet implemented for this type of "
                    "element.");
         // TODO Something with GetFaceDofs?
         orientation = 0;          // suppress compiler warning
         face_id1 = face_id2 = 0;  // suppress compiler warning
      }
      if (type==FaceType::Interior &&
          (e2>=0 || (e2<0 && inf2>=0) )) // interior/shared face
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
            if (e2>=0) // interior face
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
            else if (inf2>=0) // shared boundary
            {
               const int se2 = -1 - e2;
               Array<int> sharedDofs;
               fes.GetFaceNbrElementVDofs(se2, sharedDofs);
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
      else if (type==FaceType::Boundary && e2<0 && inf2<0) // true boundary
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
   for (int f = 0; f < fes.GetNF(); ++f)
   {
      fes.GetMesh()->GetFaceElements(f, &e1, &e2);
      fes.GetMesh()->GetFaceInfos(f, &inf1, &inf2);
      if ((type==FaceType::Interior && (e2>=0 || (e2<0 && inf2>=0))) ||
          (type==FaceType::Boundary && e2<0 && inf2<0) )
      {
         orientation = inf1 % 64;
         face_id1 = inf1 / 64;
         GetFaceDofs(dim, face_id1, dof1d, faceMap1);
         orientation = inf2 % 64;
         face_id2 = inf2 / 64;
         GetFaceDofs(dim, face_id2, dof1d, faceMap2);
         for (int d = 0; d < dof; ++d)
         {
            const int did = faceMap1[d];
            const int gid = elementMap[e1*elem_dofs + did];
            ++offsets[gid + 1];
         }
         if (m==L2FaceValues::DoubleValued)
         {
            for (int d = 0; d < dof; ++d)
            {
               if (type==FaceType::Interior && e2>=0) // interior face
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
   for (int f = 0; f < fes.GetNF(); ++f)
   {
      fes.GetMesh()->GetFaceElements(f, &e1, &e2);
      fes.GetMesh()->GetFaceInfos(f, &inf1, &inf2);
      if ((type==FaceType::Interior && (e2>=0 || (e2<0 && inf2>=0))) ||
          (type==FaceType::Boundary && e2<0 && inf2<0) )
      {
         orientation = inf1 % 64;
         face_id1 = inf1 / 64;
         GetFaceDofs(dim, face_id1, dof1d, faceMap1);
         orientation = inf2 % 64;
         face_id2 = inf2 / 64;
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
            for (int d = 0; d < dof; ++d)
            {
               if (type==FaceType::Interior && e2>=0) // interior face
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
   ParGridFunction x_gf;
   x_gf.MakeRef(const_cast<ParFiniteElementSpace*>(&fes),
                const_cast<Vector&>(x), 0);
   x_gf.ExchangeFaceNbrData();

   // Assumes all elements have the same number of dofs
   const int nd = dof;
   const int vd = vdim;
   const bool t = byvdim;
   const int threshold = ndofs;

   if (m==L2FaceValues::DoubleValued)
   {
      auto d_indices1 = scatter_indices1.Read();
      auto d_indices2 = scatter_indices2.Read();
      auto d_x = Reshape(x.Read(), t?vd:ndofs, t?ndofs:vd);
      auto d_x_shared = Reshape(x_gf.FaceNbrData().Read(),
                                t?vd:ndofs, t?ndofs:vd);
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

void ParL2FaceRestriction::MultTranspose(const Vector& x, Vector& y) const
{
   // Assumes all elements have the same number of dofs
   const int nd = dof;
   const int vd = vdim;
   const bool t = byvdim;
   const int dofs = nfdofs;
   auto d_offsets = offsets.Read();
   auto d_indices = gather_indices.Read();
   auto d_x = Reshape(x.Read(), nd, vd, 2, nf);
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
            bool isE1 = idx_j < dofs;
            idx_j = isE1 ? idx_j : idx_j - dofs;
            dofValue +=  isE1 ?
            d_x(idx_j % nd, c, 0, idx_j / nd)
            :d_x(idx_j % nd, c, 1, idx_j / nd);
         }
         d_y(t?c:i,t?i:c) += dofValue;
      }
   });
}

} // namespace mfem

#endif
