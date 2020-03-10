// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license.  We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_SPACEHIERARCHY
#define MFEM_SPACEHIERARCHY

#include "fespace.hpp"

#ifdef MFEM_USE_MPI
#include "pfespace.hpp"
#endif

namespace mfem
{

/// Class bundling a hierarchy of meshes and finite element spaces
class SpaceHierarchy
{
protected:
   Array<Mesh*> meshes;
   Array<FiniteElementSpace*> fespaces;
   Array<bool> ownedMeshes;
   Array<bool> ownedFES;

public:
   /// Constructs an empty space hierarchy
   SpaceHierarchy();

   /// Constructs a space hierarchy with the given mesh and space on the coarsest level.
   /// The ownership of the mesh and space may be transferred to the
   /// SpaceHierarchy by setting the according boolean variables.
   SpaceHierarchy(Mesh* mesh, FiniteElementSpace* fespace, bool ownM,
                  bool ownFES);

   /// Destructor deleting all meshes and spaces that are owned
   virtual ~SpaceHierarchy();

   /// Returns the number of levels in the hierarchy
   int GetNumLevels() const;

   /// Returns the index of the finest level
   int GetFinestLevelIndex() const;

   /// Adds one level to the hierarchy
   void AddLevel(Mesh* mesh, FiniteElementSpace* fespace, bool ownM,
                 bool ownFES);

   /// Adds one level to the hierarchy by uniformly refining the mesh on the
   /// previous level
   virtual void AddUniformlyRefinedLevel(int dim = 1,
                                         int ordering = Ordering::byVDIM);

   /// Adds one level to the hierarchy by using a different finite element order
   /// defined through FiniteElementCollection
   virtual void AddOrderRefinedLevel(FiniteElementCollection* fec, int dim = 1,
                                     int ordering = Ordering::byVDIM);

   /// Returns the finite element space at the given level
   virtual const FiniteElementSpace& GetFESpaceAtLevel(int level) const;

   /// Returns the finite element space at the given level
   virtual FiniteElementSpace& GetFESpaceAtLevel(int level);

   /// Returns the finite element space at the finest level
   virtual const FiniteElementSpace& GetFinestFESpace() const;

   /// Returns the finite element space at the finest level
   virtual FiniteElementSpace& GetFinestFESpace();
};

#ifdef MFEM_USE_MPI
class ParSpaceHierarchy : public SpaceHierarchy
{
public:
   /// Constructs an empty space hierarchy
   ParSpaceHierarchy();

   /// Constructs a parallel space hierarchy with the given mesh and space on
   /// level zero. The ownership of the mesh and space may be transferred to the
   /// ParSpaceHierarchy by setting the according boolean variables.
   ParSpaceHierarchy(ParMesh* mesh, ParFiniteElementSpace* fespace, bool ownM,
                     bool ownFES);

   /// Adds one level to the hierarchy by uniformly refining the mesh on the
   /// previous level
   void AddUniformlyRefinedLevel(int dim = 1,
                                 int ordering = Ordering::byVDIM) override;

   /// Adds one level to the hierarchy by using a different finite element order
   /// defined through FiniteElementCollection
   void AddOrderRefinedLevel(FiniteElementCollection* fec, int dim = 1,
                             int ordering = Ordering::byVDIM) override;

   /// Returns the finite element space at the given level
   const ParFiniteElementSpace& GetFESpaceAtLevel(int level) const override;

   /// Returns the finite element space at the given level
   ParFiniteElementSpace& GetFESpaceAtLevel(int level) override;

   /// Returns the finite element space at the finest level
   const ParFiniteElementSpace& GetFinestFESpace() const override;

   /// Returns the finite element space at the finest level
   ParFiniteElementSpace& GetFinestFESpace() override;
};
#endif

} // namespace mfem

#endif
