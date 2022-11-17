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
   T.resize(mesh.Dimension()+1);
   NT.resize(mesh.Dimension()+1);
   for (int i = 0; i < T.size()+1; ++i)
   {
      T[i].resize(T.size(), nullptr);
      NT[i].resize(NT.size(), nullptr);
   }
}

void MeshConnections::ConnectionsOfEntity(const EntityIndex entity, EntityIndices &connected) const
{
   Table *t = GetTable(entity.kind, connected.kind);
   t->GetRow(entity.index, connected.indices);
}

void MeshConnections::ConnectionsOfEntities(const EntityIndices entities, 
                                            EntityIndices &connected, bool covered) const
{
   Table *t = GetTable(entities.kind, connected.kind);

   //Find all the connected entities touched by the entities
   std::set<int> touched_set;
   for (int i = 0; i < entities.indices.Size(); ++i)
   {
      int row_sz = t->RowSize(entities.indices[i]);
      const int *row = t->GetRow(entities.indices[i]);
      for (int j = 0; j < row_sz; ++j)
      {
         touched_set.insert(row[j]);
      }
   }

   //In this case we want the entities to cover the entire connected entity to count
   if (entities.kind.dim > connected.kind.dim && covered) 
   {
      Table *trev = GetTable(connected.kind, entities.kind);
      //Put the entities into a hashing set for fast finding
      std::unordered_set<int> entity_set(entities.indices.begin(), entities.indices.end());

      connected.indices.SetSize(touched_set.size());
      int num_covered = 0;
      for (auto touched = touched_set.begin(); touched != touched_set.end(); ++touched)
      {
         int row_sz = trev->RowSize(*touched);
         const int *row = trev->GetRow(*touched);
         bool entity_covered = true;
         for (int j = 0; j < row_sz; ++j)
         {
            if (entity_set.count(row[j]) < 1)
            {
               entity_covered = false;
               break;
            }
         }

         if (entity_covered)
         {
            connected.indices[num_covered] = *touched;
            num_covered ++;
         }      
      }
      connected.indices.SetSize(num_covered);      
   } 
   else
   {
      connected.indices.SetSize(touched_set.size());
      std::copy(touched_set.begin(), touched_set.end(), connected_indices.begin());
   }
}

bool MeshConnections::IsChild(const EntityIndex &parent, const EntityIndex &child) const
{
   EntityIndices children;
   children.kind = child.kind;
   ChildrenOfEntity(parent, children);
   return children.Find(child.index) > -1;
}

void MeshConnections::ChildrenOfEntity(const EntityIndex &parent,
                                       EntityIndices &children) const
{
   MFEM_ASSERT(parent.kind.dim > children.kind.dim, "Dimension of parent must be > child dimension.");
   ConnectionsOfEntity(parent, children);
}

void MeshConnections::ChildrenOfEntities(const EntityIndices &parents,
                                         EntityIndices &children) const
{
   MFEM_ASSERT(parents.kind.dim > children.kind.dim, "Dimension of parent must be > child dimension.");
   ConnectionsOfEntities(parents, children, false);
}

void MeshConnections::ParentsOfEntity(const EntityIndex &child,
                                      EntityIndices &parents) const
{
   MFEM_ASSERT(parents.kind.dim > child.kind.dim, "Dimension of parent must be > child dimension.");
   ConnectionsOfEntity(child, parents);
}

void MeshConnections::ParentsOfAnyEntities(const EntityIndices &children,
                                           EntityIndices &parents) const
{
   MFEM_ASSERT(parents.kind.dim > children.kind.dim, "Dimension of parent must be > child dimension.");
   ConnectionsOfEntities(children, parents, false);
}

void MeshConnections::ParentsCoveredByEntities(const EntityIndices &children,
                                               EntityIndices &parents) const
{
   MFEM_ASSERT(parents.kind.dim > children.kind.dim, "Dimension of parent must be > child dimension.");
   ConnectionsOfEntities(children, parents, true);
}

void MeshConnections::NeighborsOfEntity(const EntityIndex &entity, EntityIndexKind shared_kind,
                                        EntityIndices &neighbors) const
{
   MFEM_ASSERT(entity.kind == neighbors.kind, "Neighbors entity must have the same kind.")
   Table *t = GetNeighborTable(entity.kind, shared_kind);
   t->GetRow(entity.index, neighbors.indices);
}

void MeshConnections::NeighborsOfEntities(const EntityIndices &entities, EntityIndexKind shared_kind, 
                                          EntityIndices &neighbors) const
{
   Table *t = GetTable(entities.kind, connected.kind);

   //Find all the connected entities touched by the entities
   //put it in a set to remove duplicates
   std::set<int> touched_set;
   for (int i = 0; i < entities.indices.Size(); ++i)
   {
      int row_sz = t->RowSize(entities.indices[i]);
      const int *row = t->GetRow(entities.indices[i]);
      for (int j = 0; j < row_sz; ++j)
      {
         touched_set.insert(row[j]);
      }
   }
   neighbors.indices.SetSize(touched_set.size());
   std::copy(touched_set.begin(), touched_set.end(), neighbors.indices.begin());   
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

Table* MeshConnections::GetTable(EntityIndexKind row_kind, EntityIndexKind col_kind) const
{
   const int mesh_dim = mesh.Dimension();
   MFEM_ASSERT(row_kind.space != Bdr || row_kind.dim == mesh_dim - 1, "Row boundary index must be 1 less than the mesh dimension.")
   MFEM_ASSERT(col_kind.space != Bdr || col_kind.dim == mesh_dim - 1, "Col boundary index must be 1 less than the mesh dimension.")
   MFEM_ASSERT(row_kind != col_kind, "Row and Col kinds must be different.  For neighbor connections use GetNeighborTable.")

   int row_i = row_kind.space == Bdr ? mesh_dim + 1 : row_kind.dim;
   int col_i = col_kind.space == Bdr ? mesh_dim + 1 : col_kind.dim;
   Table *t = T[row_i][col_i];
   if (t != nullptr)
   {
      return t;
   }

   return nullptr;
}


Table* MeshConnections::GetNeighborTable(EntityIndexKind entity_kind, EntityIndexKind shared_kind) const
{
   return nullptr;
}

}