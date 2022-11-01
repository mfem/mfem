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

#include "mesh_connections.hpp"
#include "mesh.hpp"

namespace mfem
{

MeshConnections::MeshConnections(const Mesh &m) :
   mesh(m)
{

}

void MeshConnections::ConnectionsOfEntity(const EntityIndex entity, EntityIndices &connected) const
{
   Table *t = GetTable(entity.dim, entity.index_type, connected.dim, connected.index_type);
   t->GetRow(entity.index, connected.indices);
}

void MeshConnections::ConnectionsOfEntities(const EntityIndices entities, 
                                            EntityIndices &connected, bool covered) const
{

}

bool MeshConnections::IsChild(const EntityIndex &parent, const EntityIndex &child) const
{

   return false;
}

void MeshConnections::ChildrenOfEntity(const EntityIndex &parent,
                                       EntityIndices &children) const
{

}

void MeshConnections::ChildrenOfEntities(const EntityIndices &parents,
                                         EntityIndices &children) const
{

}

void MeshConnections::ParentsOfEntity(const EntityIndex &child,
                                      EntityIndices &parents) const
{

}

void MeshConnections::ParentsOfAnyEntities(const EntityIndices &children,
                                           EntityIndices &parents) const
{

}

void MeshConnections::ParentsCoveredByEntities(const EntityIndices &children,
                                               EntityIndices &parents) const
{

}

void MeshConnections::NeighborsOfEntity(const EntityIndex &entity,
                                        int shared_dim, EntityIndices &neighbors) const
{

}

void MeshConnections::NeighborsOfEntities(const EntityIndices &entities,
                                          int shared_dim, EntityIndices &neighbors) const
{

}

int MeshConnections::GetAllIdxFromBdrIdx(int bdr_idx) const
{
   int all_idx;
   return all_idx;
}


int MeshConnections::GetBdrIdxFromAllIdx(int all_idx) const
{
   int bdr_idx;
   return bdr_idx;
}

Table* MeshConnections::GetTable(int row_dim, EntityIndexType row_type, 
                                 int col_dim, EntityIndexType col_type) const
{
   const int mesh_dim = mesh.Dimension();
   MFEM_ASSERT(row_type != BdrIdx || row_dim == mesh_dim - 1, "Row boundary index must be 1 less than the mesh dimension.")
   MFEM_ASSERT(col_type != BdrIdx || col_dim == mesh_dim - 1, "Col boundary index must be 1 less than the mesh dimension.")
   MFEM_ASSERT(row_dim != col_dim, "Row and Col dims must be different.  For neighbor connections use GetNeighborTable.")

   int row_i = row_type == BdrInd ? mesh_dim + 1 : row_dim;
   int col_i = col_type == BdrInd ? mesh_dim + 1 : col_dim;

   MFEM_ASSERT(row_i != col_i, "The row and col dim/type cannot be the same for GetTable.  Use GetNeighborTable for neighbors.")
   Table *t = T[row_i][col_i];
   if (t != nullptr)
   {
      return t;
   }

   return nullptr;
}


Table* MeshConnections::GetNeighborTable(int entity_dim, EntityIndexType ind_type, int shared_dim)
{
   return nullptr;
}

}