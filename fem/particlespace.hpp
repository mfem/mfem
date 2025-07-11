// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_PARTICLESPACE
#define MFEM_PARTICLESPACE

#include "../config/config.hpp"
#include "fespace.hpp"
#include "gslib.hpp"

#ifdef MFEM_USE_GSLIB

namespace mfem
{

class ParticleSpace
{
protected:
   const int dim;
   const Ordering::Type ordering;
   const int id_stride;

   int id_counter;
   Array<int> ids;

   Mesh *mesh; // not-owned
   FindPointsGSLIB finder;
   Vector coords;


   Array<int> last_removed_idxs;

#ifdef MFEM_USE_MPI
   MPI_Comm comm;
#endif // MFEM_USE_MPI


void AddParticles(const Vector &new_coords, const Array<int> &new_ids);
private:
   void Initialize(int seed);
   
public:

   /// Serial constructor
   // Optionally include mesh to define particles on
   ParticleSpace(int dim_, int num_particles, Ordering::Type ordering_=Ordering::byVDIM, Mesh *mesh_=nullptr, int seed=0);

#ifdef MFEM_USE_MPI
   /// Parallel constructor
   ParticleSpace(MPI_Comm comm_, int dim_, int num_particles, Ordering::Type ordering_=Ordering::byVDIM, Mesh *mesh_=nullptr, int seed=0);
#endif // MFEM_USE_MPI

   int Dimension() const { return dim; }

   Ordering::Type GetOrdering() const { return ordering; }

   int GetNP() const { return ids.Size(); }

   void SetMesh(Mesh *mesh_) { mesh = mesh_; finder.Setup(*mesh); finder.FindPoints(coords, ordering); }

   Mesh* GetMesh() { return mesh; }

   Vector& GetCoords() { return coords; }

   const Array<int>& GetIDs() const { return ids; }

   int GetID(int i) const { return ids[i]; }

   /// Append new particles to the particle set with the given coordinates
   /// All ParticleFunctions associated with this ParticleSpace must be updated!
   void AddParticles(const Vector &new_coords, bool findpts=false);

   /// Remove all particles not within mesh anymore (if mesh is provided)
   /// Returns reference of array of indices to be removed from ParticleFunctions
   const Array<int>& RemoveLostParticles();


#ifdef MFEM_USE_MPI
   // Nothing happens if mesh == nullptr
   void Redistribute(ParticleFunction *pfuncs[]);
#endif // MFEM_USE_MPI



};



} // namespace mfem


#endif // MFEM_USE_GSLIB
#endif // MFEM_PARTICLESPACE
