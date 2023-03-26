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

#include "submesh.hpp"
#include "submesh_utils.hpp"
#include "../../fem/gridfunc.hpp"

namespace mfem
{

SubMesh SubMesh::CreateFromDomain(const Mesh &parent,
                                  Array<int> domain_attributes)
{
   return SubMesh(parent, From::Domain, domain_attributes);
}

SubMesh SubMesh::CreateFromBoundary(const Mesh &parent,
                                    Array<int> boundary_attributes)
{
   return SubMesh(parent, From::Boundary, boundary_attributes);
}

SubMesh::SubMesh(const Mesh &parent, From from,
                 Array<int> attributes) : parent_(parent), from_(from), attributes_(attributes)
{
   if (Nonconforming())
   {
      MFEM_ABORT("SubMesh does not support non-conforming meshes");
   }

   if (from == From::Domain)
   {
      InitMesh(parent.Dimension(), parent.SpaceDimension(), 0, 0, 0);

      std::tie(parent_vertex_ids_,
               parent_element_ids_) = SubMeshUtils::AddElementsToMesh(parent_, *this,
                                                                      attributes_);
   }
   else if (from == From::Boundary)
   {
      InitMesh(parent.Dimension() - 1, parent.SpaceDimension(), 0, 0, 0);

      std::tie(parent_vertex_ids_,
               parent_element_ids_) = SubMeshUtils::AddElementsToMesh(parent_, *this,
                                                                      attributes_, true);
   }

   FinalizeTopology(true);

   if (Dim == 3)
   {
      parent_face_ids_ = SubMeshUtils::BuildFaceMap(parent, *this,
                                                    parent_element_ids_);

      Array<int> parent_face_to_be = parent.GetFaceToBdrElMap();

      for (int i = 0; i < NumOfBdrElements; i++)
      {
         int pbeid = parent_face_to_be[parent_face_ids_[GetBdrFace(i)]];
         if (pbeid != -1)
         {
            int attr = parent.GetBdrElement(pbeid)->GetAttribute();
            GetBdrElement(i)->SetAttribute(attr);
         }
         else
         {
            // This case happens when a domain is extracted, but the root parent
            // mesh didn't have a boundary element on the surface that defined
            // it's boundary. It still creates a valid mesh, so we allow it.
            GetBdrElement(i)->SetAttribute(GENERATED_ATTRIBUTE);
         }
      }
   }

   // If the parent Mesh has nodes and therefore is defined on a higher order
   // geometry, we define this SubMesh as a curved Mesh and transfer the
   // GridFunction from the parent Mesh to the SubMesh.
   const GridFunction *parent_nodes = parent.GetNodes();
   if (parent_nodes)
   {
      const FiniteElementSpace *parent_fes = parent_nodes->FESpace();

      SetCurvature(
         parent_fes->FEColl()->GetOrder(),
         parent_fes->IsDGSpace(),
         spaceDim,
         parent_fes->GetOrdering());

      Transfer(*parent.GetNodes(), *GetNodes());
   }

   SetAttributes();
   Finalize();
}

SubMesh::~SubMesh() {}

void SubMesh::Transfer(const GridFunction &src, GridFunction &dst)
{
   TransferMap map(src, dst);
   map.Transfer(src, dst);
}

TransferMap SubMesh::CreateTransferMap(const GridFunction &src,
                                       const GridFunction &dst)
{
   return TransferMap(src, dst);
}

} // namespace mfem
