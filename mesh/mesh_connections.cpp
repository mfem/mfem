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

bool MeshConnections::IsChild(EntityIndex &parent, EntityIndex &child)
{

   return false;
}

void MeshConnections::ChildrenOfEntity(const EntityIndex &parent, EntityIndices &children)
{

}

void MeshConnections::ChildrenOfEntities(const EntityIndices &parents, EntityIndices &children)
{

}

void MeshConnections::ParentsOfEntity(const EntityIndex &child, EntityIndices &parents)
{

}

void MeshConnections::ParentsOfAnyEntities(const EntityIndices &children, EntityIndices &parents)
{

}

void MeshConnections::ParentsCoveredByEntities(const EntityIndices &children, EntityIndices &parents)
{

}

void MeshConnections::NeighborsOfEntity(const EntityIndex &entity, int shared_dim, EntityIndices &neighbors)
{

}

void MeshConnections::NeighborsOfEntities(const EntityIndices &entities, int shared_dim, EntityIndices &neighbors)
{

}

Table* MeshConnections::GetTable(int row_dim, bool row_bndry, int col_dim, bool col_bndry)
{
   return nullptr;
}

}