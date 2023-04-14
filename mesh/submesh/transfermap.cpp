// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "submesh.hpp"
#include "transfermap.hpp"
#include "submesh_utils.hpp"

using namespace mfem;

TransferMap::TransferMap(const GridFunction &src,
                         const GridFunction &dst)
{
   const FiniteElementSpace *parentfes = nullptr, *subfes1 = nullptr,
                             *subfes2 = nullptr;

   if (SubMesh::IsSubMesh(src.FESpace()->GetMesh()) &&
       SubMesh::IsSubMesh(dst.FESpace()->GetMesh()))
   {
      SubMesh* src_sm = static_cast<SubMesh*>(src.FESpace()->GetMesh());
      SubMesh* dst_sm = static_cast<SubMesh*>(dst.FESpace()->GetMesh());

      // There is no immediate relation and both src and dst come from a
      // SubMesh, check if they have an equivalent root parent.
      if (SubMeshUtils::GetRootParent(*src_sm) !=
          SubMeshUtils::GetRootParent(*dst_sm))
      {
         MFEM_ABORT("Can't find a relation between the two GridFunctions");
      }

      category_ = TransferCategory::SubMeshToSubMesh;

      root_fes_.reset(new FiniteElementSpace(
                         *src.FESpace(),
                         const_cast<Mesh *>(SubMeshUtils::GetRootParent(*src_sm))));
      subfes1 = src.FESpace();
      subfes2 = dst.FESpace();

      SubMeshUtils::BuildVdofToVdofMap(*subfes1,
                                       *root_fes_,
                                       src_sm->GetFrom(),
                                       src_sm->GetParentElementIDMap(),
                                       sub1_to_parent_map_);

      SubMeshUtils::BuildVdofToVdofMap(*subfes2,
                                       *root_fes_,
                                       dst_sm->GetFrom(),
                                       dst_sm->GetParentElementIDMap(),
                                       sub2_to_parent_map_);

      z_.SetSize(root_fes_->GetVSize());
   }
   else if (SubMesh::IsSubMesh(src.FESpace()->GetMesh()))
   {
      category_ = TransferCategory::SubMeshToParent;
      SubMesh* src_sm = static_cast<SubMesh*>(src.FESpace()->GetMesh());
      subfes1 = src.FESpace();
      parentfes = dst.FESpace();
      SubMeshUtils::BuildVdofToVdofMap(*subfes1,
                                       *parentfes,
                                       src_sm->GetFrom(),
                                       src_sm->GetParentElementIDMap(),
                                       sub1_to_parent_map_);
   }
   else if (SubMesh::IsSubMesh(dst.FESpace()->GetMesh()))
   {
      category_ = TransferCategory::ParentToSubMesh;
      SubMesh* dst_sm = static_cast<SubMesh*>(dst.FESpace()->GetMesh());
      subfes1 = dst.FESpace();
      parentfes = src.FESpace();
      SubMeshUtils::BuildVdofToVdofMap(*subfes1,
                                       *parentfes,
                                       dst_sm->GetFrom(),
                                       dst_sm->GetParentElementIDMap(),
                                       sub1_to_parent_map_);
   }
   else
   {
      MFEM_ABORT("Trying to do a transfer between GridFunctions but none of them is defined on a SubMesh");
   }
}

void TransferMap::Transfer(const GridFunction &src,
                           GridFunction &dst) const
{
   if (category_ == TransferCategory::ParentToSubMesh)
   {
      // dst = S1^T src
      for (int i = 0; i < sub1_to_parent_map_.Size(); i++)
      {
         double s = 1.0;
         int j = FiniteElementSpace::DecodeDof(sub1_to_parent_map_[i], s);
         dst(i) = s * src(j);
      }

      CorrectFaceOrientations(*dst.FESpace(), src, dst);
   }
   else if (category_ == TransferCategory::SubMeshToParent)
   {
      // dst = G S1 src
      //     = G z
      //
      // G is identity if the partitioning matches

      for (int i = 0; i < sub1_to_parent_map_.Size(); i++)
      {
         double s = 1.0;
         int j = FiniteElementSpace::DecodeDof(sub1_to_parent_map_[i], s);
         dst(j) = s * src(i);
      }

      CorrectFaceOrientations(*src.FESpace(), src, dst,
                              &sub1_to_parent_map_);
   }
   else if (category_ == TransferCategory::SubMeshToSubMesh)
   {
      // dst = S2^T G (S1 src (*) S2 dst)
      //
      // G is identity if the partitioning matches

      z_ = 0.0;

      for (int i = 0; i < sub2_to_parent_map_.Size(); i++)
      {
         double s = 1.0;
         int j = FiniteElementSpace::DecodeDof(sub2_to_parent_map_[i], s);
         z_(j) = s * dst(i);
      }

      CorrectFaceOrientations(*dst.FESpace(), dst, z_,
                              &sub2_to_parent_map_);

      for (int i = 0; i < sub1_to_parent_map_.Size(); i++)
      {
         double s = 1.0;
         int j = FiniteElementSpace::DecodeDof(sub1_to_parent_map_[i], s);
         z_(j) = s * src(i);
      }

      CorrectFaceOrientations(*src.FESpace(), src, z_,
                              &sub1_to_parent_map_);

      for (int i = 0; i < sub2_to_parent_map_.Size(); i++)
      {
         double s = 1.0;
         int j = FiniteElementSpace::DecodeDof(sub2_to_parent_map_[i], s);
         dst(i) = s * z_(j);
      }

      CorrectFaceOrientations(*dst.FESpace(), z_, dst);
   }
   else
   {
      MFEM_ABORT("unknown TransferCategory: " << category_);
   }
}

void TransferMap::CorrectFaceOrientations(const FiniteElementSpace &fes,
                                          const Vector &src,
                                          Vector &dst,
                                          const Array<int> *sub_to_parent_map)
{
   const FiniteElementCollection * fec = fes.FEColl();

   SubMesh * mesh = dynamic_cast<SubMesh*>(fes.GetMesh());

   const Array<int>& parent_face_ori = mesh->GetParentFaceOrientations();

   if (parent_face_ori.Size() == 0) { return; }

   VDofTransformation vdoftrans(fes.GetVDim(),
                                fes.GetOrdering());

   int dim = mesh->Dimension();
   bool face = (dim == 3);

   Array<int> vdofs;
   Array<int> Fo(1);
   Vector face_vector;

   for (int i = 0; i < (face ? mesh->GetNumFaces() : mesh->GetNE()); i++)
   {
      if (parent_face_ori[i] == 0) { continue; }

      Geometry::Type geom = face ? mesh->GetFaceGeometry(i) :
                            mesh->GetElementGeometry(i);;

      DofTransformation * doftrans = fec->DofTransformationForGeometry(geom);

      if (doftrans == NULL) { continue; }

      vdoftrans.SetDofTransformation(*doftrans);

      Fo[0] = parent_face_ori[i];
      vdoftrans.SetFaceOrientations(Fo);

      if (face)
      {
         fes.GetFaceVDofs(i, vdofs);
      }
      else
      {
         fes.GetElementVDofs(i, vdofs);
      }

      if (sub_to_parent_map)
      {
         src.GetSubVector(vdofs, face_vector);
         vdoftrans.TransformPrimal(face_vector);
      }
      else
      {
         dst.GetSubVector(vdofs, face_vector);
         vdoftrans.InvTransformPrimal(face_vector);
      }

      for (int j = 0; j < vdofs.Size(); j++)
      {
         double s = 1.0;
         int k = FiniteElementSpace::DecodeDof(vdofs[j], s);

         if (sub_to_parent_map)
         {
            double sps = 1.0;
            int spk = FiniteElementSpace::DecodeDof((*sub_to_parent_map)[k],
                                                    sps);
            s *= sps;
            k = spk;
         }

         dst[k] = s * face_vector[j];
      }
   }
}
