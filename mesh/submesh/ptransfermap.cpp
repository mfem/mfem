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

#include "../../config/config.hpp"

#ifdef MFEM_USE_MPI

#include "psubmesh.hpp"
#include "ptransfermap.hpp"
#include "submesh_utils.hpp"

using namespace mfem;

ParTransferMap::ParTransferMap(const ParGridFunction &src,
                               const ParGridFunction &dst)
{
   const ParFiniteElementSpace *parentfes = nullptr, *subfes1 = nullptr,
                                *subfes2 = nullptr;

   if (ParSubMesh::IsParSubMesh(src.ParFESpace()->GetParMesh()) &&
       ParSubMesh::IsParSubMesh(dst.ParFESpace()->GetParMesh()))
   {
      ParSubMesh* src_sm = static_cast<ParSubMesh*>(src.ParFESpace()->GetParMesh());
      ParSubMesh* dst_sm = static_cast<ParSubMesh*>(dst.ParFESpace()->GetParMesh());

      // There is no immediate relation and both src and dst come from a
      // SubMesh, check if they have an equivalent root parent.
      if (SubMeshUtils::GetRootParent(*src_sm) !=
          SubMeshUtils::GetRootParent(*dst_sm))
      {
         MFEM_ABORT("Can't find a relation between the two GridFunctions");
      }

      category_ = TransferCategory::SubMeshToSubMesh;

      root_fes_.reset(new ParFiniteElementSpace(
                         *src.ParFESpace(),
                         *const_cast<ParMesh *>(SubMeshUtils::GetRootParent(*src_sm))));
      subfes1 = src.ParFESpace();
      subfes2 = dst.ParFESpace();

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

      root_gc_ = &root_fes_->GroupComm();
      CommunicateIndicesSet(sub1_to_parent_map_, root_fes_->GetVSize());

      z_.SetSize(root_fes_->GetVSize());
   }
   else if (ParSubMesh::IsParSubMesh(src.ParFESpace()->GetParMesh()))
   {
      category_ = TransferCategory::SubMeshToParent;
      ParSubMesh* src_sm = static_cast<ParSubMesh*>(src.ParFESpace()->GetParMesh());
      subfes1 = src.ParFESpace();
      parentfes = dst.ParFESpace();
      SubMeshUtils::BuildVdofToVdofMap(*subfes1,
                                       *parentfes,
                                       src_sm->GetFrom(),
                                       src_sm->GetParentElementIDMap(),
                                       sub1_to_parent_map_);

      root_gc_ = &parentfes->GroupComm();
      CommunicateIndicesSet(sub1_to_parent_map_, dst.Size());
   }
   else if (ParSubMesh::IsParSubMesh(dst.ParFESpace()->GetParMesh()))
   {
      category_ = TransferCategory::ParentToSubMesh;
      ParSubMesh* dst_sm = static_cast<ParSubMesh*>(dst.ParFESpace()->GetParMesh());
      subfes1 = dst.ParFESpace();
      parentfes = src.ParFESpace();
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

void ParTransferMap::Transfer(const ParGridFunction &src,
                              ParGridFunction &dst) const
{
   if (category_ == TransferCategory::ParentToSubMesh)
   {
      // dst = S1^T src
      for (int i = 0; i < sub1_to_parent_map_.Size(); i++)
      {
         dst(i) = src(sub1_to_parent_map_[i]);
      }
   }
   else if (category_ == TransferCategory::SubMeshToParent)
   {
      // dst = G S1 src
      //     = G z
      //
      // G is identity if the partitioning matches

      for (int i = 0; i < sub1_to_parent_map_.Size(); i++)
      {
         dst(sub1_to_parent_map_[i]) = src(i);
      }

      CommunicateSharedVdofs(dst);
   }
   else if (category_ == TransferCategory::SubMeshToSubMesh)
   {
      // dst = S2^T G (S1 src (*) S2 dst)
      //
      // G is identity if the partitioning matches

      z_ = 0.0;

      for (int i = 0; i < sub2_to_parent_map_.Size(); i++)
      {
         z_(sub2_to_parent_map_[i]) = dst(i);
      }

      for (int i = 0; i < sub1_to_parent_map_.Size(); i++)
      {
         z_(sub1_to_parent_map_[i]) = src(i);
      }

      CommunicateSharedVdofs(z_);

      for (int i = 0; i < sub2_to_parent_map_.Size(); i++)
      {
         dst(i) = z_(sub2_to_parent_map_[i]);
      }
   }
   else
   {
      MFEM_ABORT("unknown TransferCategory: " << category_);
   }
}

void ParTransferMap::CommunicateIndicesSet(Array<int> &map, int dst_sz)
{
   indices_set_local_.SetSize(dst_sz);
   indices_set_local_ = 0;
   for (int i = 0; i < map.Size(); i++)
   {
      indices_set_local_[map[i]] = 1;
   }
   indices_set_global_ = indices_set_local_;
   root_gc_->Reduce(indices_set_global_, GroupCommunicator::Sum);
   root_gc_->Bcast(indices_set_global_);
}

void ParTransferMap::CommunicateSharedVdofs(Vector &f) const
{
   // f is usually defined on the root vdofs

   const Table &group_ldof = root_gc_->GroupLDofTable();

   // Identify indices that were only set by other ranks and clear the dof.
   for (int i = 0; i < group_ldof.Size_of_connections(); i++)
   {
      const int j = group_ldof.GetJ()[i];
      if (indices_set_global_[j] != 0 && indices_set_local_[j] == 0)
      {
         f(j) = 0.0;
      }
   }

   // TODO: do the reduce only on dofs of interest
   root_gc_->Reduce<double>(f.HostReadWrite(), GroupCommunicator::Sum);

   // Indices that were set from this rank or other ranks have been summed up
   // and therefore need to be "averaged". Note that this results in the exact
   // value that is desired.
   for (int i = 0; i < group_ldof.Size_of_connections(); i++)
   {
      const int j = group_ldof.GetJ()[i];
      if (indices_set_global_[j] != 0)
      {
         f(j) /= indices_set_global_[j];
      }
   }

   // Indices for dofs that are shared between processors need to be divided by
   // the whole group size that share this dof.
   for (int gr = 1; gr < group_ldof.Size(); gr++)
   {
      for (int i = 0; i < group_ldof.RowSize(gr); i++)
      {
         const int j = group_ldof.GetRow(gr)[i];
         if (indices_set_global_[j] == 0)
         {
            f(j) /= root_gc_->GetGroupTopology().GetGroupSize(gr);
         }
      }
   }

   root_gc_->Bcast<double>(f.HostReadWrite());
}

#endif // MFEM_USE_MPI
