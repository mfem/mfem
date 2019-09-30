// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#include "spacehierarchy.hpp"

namespace mfem
{

template <typename M, typename FES>
AbstractSpaceHierarchy<M, FES>::AbstractSpaceHierarchy(M* mesh, FES* fespace,
                                                       bool ownM, bool ownFES)
{
   AddLevel(mesh, fespace, ownM, ownFES);
}

template <typename M, typename FES>
AbstractSpaceHierarchy<M, FES>::~AbstractSpaceHierarchy()
{
   for (int i = meshes.Size() - 1; i >= 0; --i)
   {
      if (ownedFES[i])
      {
         delete fespaces[i];
      }
      if (ownedMeshes[i])
      {
         delete meshes[i];
      }
   }

   fespaces.DeleteAll();
   meshes.DeleteAll();
}

template <typename M, typename FES>
unsigned AbstractSpaceHierarchy<M, FES>::GetNumLevels() const
{
   return meshes.Size();
}

template <typename M, typename FES>
unsigned AbstractSpaceHierarchy<M, FES>::GetFinestLevelIndex() const
{
   return GetNumLevels() - 1;
}

template <typename M, typename FES>
void AbstractSpaceHierarchy<M, FES>::AddLevel(M* mesh, FES* fespace, bool ownM,
                                              bool ownFES)
{
   meshes.Append(mesh);
   fespaces.Append(fespace);
   ownedMeshes.Append(ownM);
   ownedFES.Append(ownFES);
}

template <typename M, typename FES>
void AbstractSpaceHierarchy<M, FES>::AddUniformlyRefinedLevel(int dim,
                                                              int ordering)
{
   M* mesh = new M(*meshes[GetFinestLevelIndex()]);
   mesh->UniformRefinement();
   FES* coarseFEspace = fespaces[GetFinestLevelIndex()];
   FES* fineFEspace = new FES(mesh, coarseFEspace->FEColl(), dim, ordering);
   AddLevel(mesh, fineFEspace, true, true);
}

template <typename M, typename FES>
void AbstractSpaceHierarchy<M, FES>::AddOrderRefinedLevel(
    FiniteElementCollection* fec, int dim, int ordering)
{
   M* mesh = &GetFinestMesh();
   FES* newFEspace = new FES(mesh, fec, dim, ordering);
   AddLevel(mesh, newFEspace, false, true);
}

template <typename M, typename FES>
const M& AbstractSpaceHierarchy<M, FES>::GetMeshAtLevel(unsigned level) const
{
   MFEM_ASSERT(level < meshes.Size(), "Mesh at given level does not exist.");
   return *meshes[level];
}

template <typename M, typename FES>
M& AbstractSpaceHierarchy<M, FES>::GetMeshAtLevel(unsigned level)
{
   return const_cast<M&>(
       const_cast<const AbstractSpaceHierarchy*>(this)->GetMeshAtLevel(level));
}

template <typename M, typename FES>
const M& AbstractSpaceHierarchy<M, FES>::GetFinestMesh() const
{
   return GetMeshAtLevel(GetFinestLevelIndex());
}

template <typename M, typename FES>
M& AbstractSpaceHierarchy<M, FES>::GetFinestMesh()
{
   return const_cast<M&>(
       const_cast<const AbstractSpaceHierarchy*>(this)->GetFinestMesh());
}

template <typename M, typename FES>
const FES&
AbstractSpaceHierarchy<M, FES>::GetFESpaceAtLevel(unsigned level) const
{
   MFEM_ASSERT(level < fespaces.Size(),
               "FE space at given level does not exist.");
   return *fespaces[level];
}

template <typename M, typename FES>
FES& AbstractSpaceHierarchy<M, FES>::GetFESpaceAtLevel(unsigned level)
{
   return const_cast<FES&>(
       const_cast<const AbstractSpaceHierarchy*>(this)->GetFESpaceAtLevel(
           level));
}

template <typename M, typename FES>
const FES& AbstractSpaceHierarchy<M, FES>::GetFinestFESpace() const
{
   return GetFESpaceAtLevel(GetFinestLevelIndex());
}

template <typename M, typename FES>
FES& AbstractSpaceHierarchy<M, FES>::GetFinestFESpace()
{
   return const_cast<FES&>(
       const_cast<const AbstractSpaceHierarchy*>(this)->GetFinestFESpace());
}

// Explicit template instantiation
template class AbstractSpaceHierarchy<Mesh, FiniteElementSpace>;

#ifdef MFEM_USE_MPI
template class AbstractSpaceHierarchy<ParMesh, ParFiniteElementSpace>;
#endif

} // namespace mfem