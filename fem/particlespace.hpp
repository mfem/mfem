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

#include <variant>

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
   void Initialize(Mesh *mesh, int seed);

protected:
   const int dim;
   const Ordering::Type ordering;
   const int id_stride;

   int id_counter;

   ParticleArray<int> ids;
   ParticleFunction coords;

   std::vector<Mesh*> meshes;
   std::vector<FindPointsGSLIB> finders;

   // User-created data:
   // For now we only allow ParticleFunction
   // Can very easily add ParticleArray<int>
   std::vector<std::string> all_func_names;
   std::vector<ParticleFunction> all_funcs;


   void AddParticles(const Vector &new_coords, const Array<int> &new_ids, Array<int> &new_indices);


#ifdef MFEM_USE_MPI
   MPI_Comm comm;
#endif // MFEM_USE_MPI

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

   int GetID(int i) const { return ids[i]; }

   // Returns index of registered mesh
   int RegisterMesh(Mesh &mesh_);

   // TODO: If we want to expose coords for changes directly, must have a way to "Commit" changes (re-call FindPoints...)
   const ParticleFunction& GetCoords() const { return coords; }

   // Update many particle coordinates
   void UpdateCoords(const Array<int> &indices, const Vector &updated_coords);

   // Update individual particle coords
   void UpdateCoords(int i, const Vector &p_coords)
   { UpdateCoords(Array<int>({i}), p_coords); }

   const ParticleData<int>& GetIDs() const { return ids; }

   // Optionally include string name for PrintCSV
   //template<typename T>
   //ParticleData<T>& CreateParticleData(int vdim=1, std::string name="");

   ParticleFunction& CreateParticleFunction(int vdim=1, std::string name="");

   /// Append new particles to the particle set with the given coordinates
   //  Optionally get array of indices of the new particles, for updating any ParticleData
   void AddParticles(const Vector &new_coords, Array<int> *new_indices=nullptr);

   /// Remove particles at given indices
   void RemoveParticles(const Array<int> &indices);

   /// Remove all particles not within mesh anymore (at provided mesh_idx!)
   void RemoveLostParticles(int mesh_idx=0);

// Below are still TODO:
   void PrintCSV(std::string fname, int precision=16);

#ifdef MFEM_USE_MPI
   // Redistribution occurs relative to the mesh at index redistribute_mesh_idx
   void Redistribute(int mesh_idx=0);
#endif // MFEM_USE_MPI



};



} // namespace mfem


#endif // MFEM_USE_GSLIB
#endif // MFEM_PARTICLESPACE
