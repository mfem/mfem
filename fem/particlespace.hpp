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
#include "particledata.hpp"

#ifdef MFEM_USE_GSLIB

namespace mfem
{

// Can add/register N meshes (including 0!).
// This will setup FindPointsGSLIB with an arbitrary number of meshes
// (Benefit is that if one wants to Interpolate a GF, we can check if we have a Setup FindPointsGSSLIB first)
// There must be a MAIN mesh of which redistribution occurs on
// Any change to particle coordinates == All FindPointsGSLIB::FindPoints called on updated coordinates

class ParticleSpace
{

private:
   void Initialize(int seed);

protected:
   const int dim;
   const Ordering::Type ordering;
   const int id_stride;

   int id_counter;
   Array<int> ids;
   std::unique_ptr<ParticleFunction> coords;

   int redistribute_mesh_idx;
   std::vector<Mesh*> meshes;
   std::vector<FindPointsGSLIB> finder;

   using ParticleDataVar = std::variant<ParticleData<int>, ParticleData<real_t>>;

   // TODO PROBLEM: We need destroyed ParticleDatas to be set to nullptr here... 
   //               Can do that in destructor of ParticleData, but then what if ParticleSpace is destroyed beforehand?
   //               Then it can't call pspace.Deregister(...) ... This is a problem. It requires that ParticleSpace is deleted last.
   //               This kind of linkage isn't ideal... 
   // Maybe instead we just have all of the ParticleFunctions be stored here? Like
   //       ParticleData<T>& CreateParticleData(int vdim);
   // Then everything remains internal, so we don't need to deal with this substantial coupling + registering issue
   // It also becomes a lot more clear how ParticleSpace manages all the ParticleData's<T>
   // Then ParticleData REALLY then just becomes a super lightweight wrapper.... It should be specialized for Particle's though for usage
   std::vector<ParticleDataVar> pdata;

#ifdef MFEM_USE_MPI
   MPI_Comm comm;
#endif // MFEM_USE_MPI


   void AddParticles(const Vector &new_coords, const Array<int> &new_ids);


public:

   /// Serial constructor
   // Optionally include mesh to define num_particles on randomly
   ParticleSpace(int dim_, int num_particles,
                 Ordering::Type ordering_=Ordering::byVDIM, Mesh *mesh_=nullptr, int seed=0);

#ifdef MFEM_USE_MPI
   /// Parallel constructor
   ParticleSpace(MPI_Comm comm_, int dim_, int num_particles,
                 Ordering::Type ordering_=Ordering::byVDIM, Mesh *mesh_=nullptr, int seed=0);
#endif // MFEM_USE_MPI

   int Dimension() const { return dim; }

   Ordering::Type GetOrdering() const { return ordering; }

   int GetNP() const { return ids.Size(); }

   void RegisterMesh(Mesh *mesh_, bool set_as_redist=false);

   Mesh* GetMesh(int idx=0) { return meshes[i]; }

   Vector& GetCoords() { return coords; }

   const Array<int>& GetIDs() const { return ids; }

   int GetID(int i) const { return ids[i]; }

   template<typename T>
   ParticleData<T>& CreateParticleData(int vdim=1);

   ParticleFunction& CreateParticleFunction(int vdim=1);

   /// Append new particles to the particle set with the given coordinates
   //  Optionally get array of indices of the new particles, for updating any ParticleData
   void AddParticles(const Vector &new_coords, Array<int> *new_indices=nullptr);

   /// Remove all particles not within mesh anymore (if mesh is provided)
   void RemoveLostParticles();


#ifdef MFEM_USE_MPI
   // Redistribution occurs relative to the mesh at index redistribute_mesh_idx
   void Redistribute();
#endif // MFEM_USE_MPI



};



} // namespace mfem


#endif // MFEM_USE_GSLIB
#endif // MFEM_PARTICLESPACE
