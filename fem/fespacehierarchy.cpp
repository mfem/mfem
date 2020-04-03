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

#include "fespacehierarchy.hpp"

namespace mfem
{

FiniteElementSpaceHierarchy::FiniteElementSpaceHierarchy() {}

FiniteElementSpaceHierarchy::FiniteElementSpaceHierarchy(Mesh* mesh,
                                                         FiniteElementSpace* fespace,
                                                         bool ownM, bool ownFES)
{
   AddLevel(mesh, fespace, ownM, ownFES);
}

FiniteElementSpaceHierarchy::~FiniteElementSpaceHierarchy()
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

int FiniteElementSpaceHierarchy::GetNumLevels() const { return meshes.Size(); }

int FiniteElementSpaceHierarchy::GetFinestLevelIndex() const { return GetNumLevels() - 1; }

void FiniteElementSpaceHierarchy::AddLevel(Mesh* mesh,
                                           FiniteElementSpace* fespace,
                                           bool ownM, bool ownFES)
{
   meshes.Append(mesh);
   fespaces.Append(fespace);
   ownedMeshes.Append(ownM);
   ownedFES.Append(ownFES);
}

void FiniteElementSpaceHierarchy::AddUniformlyRefinedLevel(int dim,
                                                           int ordering)
{
   MFEM_VERIFY(GetNumLevels() > 0, "There is no level which can be refined");
   Mesh* mesh = new Mesh(*GetFinestFESpace().GetMesh());
   mesh->UniformRefinement();
   FiniteElementSpace& coarseFEspace = GetFinestFESpace();
   FiniteElementSpace* fineFEspace =
      new FiniteElementSpace(mesh, coarseFEspace.FEColl(), dim, ordering);
   AddLevel(mesh, fineFEspace, true, true);
}

void FiniteElementSpaceHierarchy::AddOrderRefinedLevel(FiniteElementCollection*
                                                       fec, int dim,
                                                       int ordering)
{
   MFEM_VERIFY(GetNumLevels() > 0, "There is no level which can be refined");
   Mesh* mesh = GetFinestFESpace().GetMesh();
   FiniteElementSpace* newFEspace =
      new FiniteElementSpace(mesh, fec, dim, ordering);
   AddLevel(mesh, newFEspace, false, true);
}

const FiniteElementSpace& FiniteElementSpaceHierarchy::GetFESpaceAtLevel(
   int level) const
{
   MFEM_ASSERT(level < fespaces.Size(),
               "FE space at given level does not exist.");
   return *fespaces[level];
}

FiniteElementSpace& FiniteElementSpaceHierarchy::GetFESpaceAtLevel(int level)
{
   MFEM_ASSERT(level < fespaces.Size(),
               "FE space at given level does not exist.");
   return *fespaces[level];
}

const FiniteElementSpace& FiniteElementSpaceHierarchy::GetFinestFESpace() const
{
   return GetFESpaceAtLevel(GetFinestLevelIndex());
}

FiniteElementSpace& FiniteElementSpaceHierarchy::GetFinestFESpace()
{
   return GetFESpaceAtLevel(GetFinestLevelIndex());
}

#ifdef MFEM_USE_MPI
ParFiniteElementSpaceHierarchy::ParFiniteElementSpaceHierarchy() {}

ParFiniteElementSpaceHierarchy::ParFiniteElementSpaceHierarchy(ParMesh* mesh,
                                                               ParFiniteElementSpace* fespace, bool ownM,
                                                               bool ownFES)
   : FiniteElementSpaceHierarchy(mesh, fespace, ownM, ownFES)
{
}

void ParFiniteElementSpaceHierarchy::AddUniformlyRefinedLevel(int dim,
                                                              int ordering)
{
   ParMesh* mesh = new ParMesh(*GetFinestFESpace().GetParMesh());
   mesh->UniformRefinement();
   ParFiniteElementSpace& coarseFEspace = GetFinestFESpace();
   ParFiniteElementSpace* fineFEspace =
      new ParFiniteElementSpace(mesh, coarseFEspace.FEColl(), dim, ordering);
   AddLevel(mesh, fineFEspace, true, true);
}

void ParFiniteElementSpaceHierarchy::AddOrderRefinedLevel(
   FiniteElementCollection* fec,
   int dim, int ordering)
{
   ParMesh* mesh = GetFinestFESpace().GetParMesh();
   ParFiniteElementSpace* newFEspace =
      new ParFiniteElementSpace(mesh, fec, dim, ordering);
   AddLevel(mesh, newFEspace, false, true);
}

const ParFiniteElementSpace&
ParFiniteElementSpaceHierarchy::GetFESpaceAtLevel(int level) const
{
   return static_cast<const ParFiniteElementSpace&>(
             FiniteElementSpaceHierarchy::GetFESpaceAtLevel(level));
}

ParFiniteElementSpace& ParFiniteElementSpaceHierarchy::GetFESpaceAtLevel(
   int level)
{
   return static_cast<ParFiniteElementSpace&>(
             FiniteElementSpaceHierarchy::GetFESpaceAtLevel(level));
}

const ParFiniteElementSpace& ParFiniteElementSpaceHierarchy::GetFinestFESpace()
const
{
   return GetFESpaceAtLevel(GetFinestLevelIndex());
}

ParFiniteElementSpace& ParFiniteElementSpaceHierarchy::GetFinestFESpace()
{
   return GetFESpaceAtLevel(GetFinestLevelIndex());
}
#endif

} // namespace mfem
