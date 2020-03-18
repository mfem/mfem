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

#include "spacehierarchy.hpp"

namespace mfem
{

SpaceHierarchy::SpaceHierarchy() {}

SpaceHierarchy::SpaceHierarchy(Mesh* mesh, FiniteElementSpace* fespace,
                               bool ownM, bool ownFES)
{
   AddLevel(mesh, fespace, ownM, ownFES);
}

SpaceHierarchy::~SpaceHierarchy()
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

int SpaceHierarchy::GetNumLevels() const { return meshes.Size(); }

int SpaceHierarchy::GetFinestLevelIndex() const { return GetNumLevels() - 1; }

void SpaceHierarchy::AddLevel(Mesh* mesh, FiniteElementSpace* fespace,
                              bool ownM, bool ownFES)
{
   meshes.Append(mesh);
   fespaces.Append(fespace);
   ownedMeshes.Append(ownM);
   ownedFES.Append(ownFES);
}

void SpaceHierarchy::AddUniformlyRefinedLevel(int dim, int ordering)
{
   MFEM_VERIFY(GetNumLevels() > 0, "There is no level which can be refined");
   Mesh* mesh = new Mesh(*GetFinestFESpace().GetMesh());
   mesh->UniformRefinement();
   FiniteElementSpace& coarseFEspace = GetFinestFESpace();
   FiniteElementSpace* fineFEspace =
      new FiniteElementSpace(mesh, coarseFEspace.FEColl(), dim, ordering);
   AddLevel(mesh, fineFEspace, true, true);
}

void SpaceHierarchy::AddOrderRefinedLevel(FiniteElementCollection* fec, int dim,
                                          int ordering)
{
   MFEM_VERIFY(GetNumLevels() > 0, "There is no level which can be refined");
   Mesh* mesh = GetFinestFESpace().GetMesh();
   FiniteElementSpace* newFEspace =
      new FiniteElementSpace(mesh, fec, dim, ordering);
   AddLevel(mesh, newFEspace, false, true);
}

const FiniteElementSpace& SpaceHierarchy::GetFESpaceAtLevel(int level) const
{
   MFEM_ASSERT(level < fespaces.Size(),
               "FE space at given level does not exist.");
   return *fespaces[level];
}

FiniteElementSpace& SpaceHierarchy::GetFESpaceAtLevel(int level)
{
   MFEM_ASSERT(level < fespaces.Size(),
               "FE space at given level does not exist.");
   return *fespaces[level];
}

const FiniteElementSpace& SpaceHierarchy::GetFinestFESpace() const
{
   return GetFESpaceAtLevel(GetFinestLevelIndex());
}

FiniteElementSpace& SpaceHierarchy::GetFinestFESpace()
{
   return GetFESpaceAtLevel(GetFinestLevelIndex());
}

#ifdef MFEM_USE_MPI
ParSpaceHierarchy::ParSpaceHierarchy() {}

ParSpaceHierarchy::ParSpaceHierarchy(ParMesh* mesh,
                                     ParFiniteElementSpace* fespace, bool ownM,
                                     bool ownFES)
   : SpaceHierarchy(mesh, fespace, ownM, ownFES)
{
}

void ParSpaceHierarchy::AddUniformlyRefinedLevel(int dim, int ordering)
{
   ParMesh* mesh = new ParMesh(*GetFinestFESpace().GetParMesh());
   mesh->UniformRefinement();
   ParFiniteElementSpace& coarseFEspace = GetFinestFESpace();
   ParFiniteElementSpace* fineFEspace =
      new ParFiniteElementSpace(mesh, coarseFEspace.FEColl(), dim, ordering);
   AddLevel(mesh, fineFEspace, true, true);
}

void ParSpaceHierarchy::AddOrderRefinedLevel(FiniteElementCollection* fec,
                                             int dim, int ordering)
{
   ParMesh* mesh = GetFinestFESpace().GetParMesh();
   ParFiniteElementSpace* newFEspace =
      new ParFiniteElementSpace(mesh, fec, dim, ordering);
   AddLevel(mesh, newFEspace, false, true);
}

const ParFiniteElementSpace&
ParSpaceHierarchy::GetFESpaceAtLevel(int level) const
{
   return static_cast<const ParFiniteElementSpace&>(
             SpaceHierarchy::GetFESpaceAtLevel(level));
}

ParFiniteElementSpace& ParSpaceHierarchy::GetFESpaceAtLevel(int level)
{
   return static_cast<ParFiniteElementSpace&>(
             SpaceHierarchy::GetFESpaceAtLevel(level));
}

const ParFiniteElementSpace& ParSpaceHierarchy::GetFinestFESpace() const
{
   return GetFESpaceAtLevel(GetFinestLevelIndex());
}

ParFiniteElementSpace& ParSpaceHierarchy::GetFinestFESpace()
{
   return GetFESpaceAtLevel(GetFinestLevelIndex());
}
#endif

} // namespace mfem
